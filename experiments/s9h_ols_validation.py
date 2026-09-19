"""
Train-only OLS scored on the validation block of the model selection

Fits the OLS (signal + T + RH) on the training rows before the validation
block and scores it on the block. Per target, records the method with the
lower validation RMSE: the OLS or the selected ML model. Writes
ols_validation_node{N}.csv and method_selection_v6.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REF_CSV_NAME, load_node_pipeline

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "cross_node"
TARGETS = ["PM2.5", "CO", "NO2", "O3", "NO2_alpha"]
CONFIGS = ["A", "B", "C"]
ABBR = {"XGBoost": "XGB", "LightGBM": "LGBM", "RF": "RF", "DNN": "DNN", "LSTM": "LSTM"}
VAL_FRAC = 0.25  # Validation block: last fraction of the valid training rows


# -------------------------------------------------------------------------
# Test R2 on the intersection basis
# -------------------------------------------------------------------------
def e4_test_r2(e4: pd.DataFrame, gas: str, method: str, model: str | None = None, config: str | None = None) -> float:
    """Return the test R2 of one method on the intersection basis of the single-basis table."""
    sub = e4[(e4.gas == gas) & (e4.method == method) & (e4.basis == "intersection") & (e4.period == "test")]
    if method == "ml":
        # Rows tagged LSTM_v6 come from tabla_maestra_lstm_v6.csv and take precedence
        names = ["LSTM_v6", "LSTM"] if model == "LSTM" else [model]
        for name in names:
            s2 = sub[(sub.model == name) & (sub.config == config)]
            if len(s2) and pd.notna(s2.iloc[0].R2):
                return float(s2.iloc[0].R2)
        return float("nan")
    return float(sub.iloc[0].R2) if len(sub) else float("nan")


# -------------------------------------------------------------------------
# OLS on the validation block, per target and configuration
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Score the train-only OLS on the validation block used to select the ML model.")
    ap.add_argument("--node", type=int, choices=(3, 5), required=True)
    args = ap.parse_args()
    print(f"OLS on the validation block | Node {args.node}")

    cfg, utils = load_node_pipeline(args.node)
    from ml.data_loader import prepare_dataset, get_train_test_masks, get_feature_columns
    from ml.ols_refit import refit_ols_on_train
    from ml.evaluator import evaluate_model

    s1 = utils.detect_csv(cfg.DATA_S1, f"{cfg.NODE_ID}_QC_RAW_wide_data_*.csv")
    s3 = utils.detect_csv(cfg.DATA_S3, f"{cfg.NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    s4 = utils.detect_csv(cfg.DATA_S4, f"{cfg.NODE_ID}_PREPROCESS_wide_data_*.csv")
    ref = cfg.REF_DATA_DIR / REF_CSV_NAME[args.node]
    for n, p in (("s1", s1), ("s3", s3), ("s4", s4), ("ref", ref)):
        if p is None or not Path(p).exists():
            sys.exit(f"ERROR: {n} not found: {p}")
        print(f"  input {n:3s} {Path(p).name}")
    df = prepare_dataset(s1, s3, s4, ref)
    train_mask, _ = get_train_test_masks(df)
    ts_all = pd.to_datetime(df[cfg.COL_TS], utc=True)

    per_gas = pd.read_csv(OUT_DIR / f"locked_selection_per_gas_node{args.node}.csv")
    per_gas = per_gas[per_gas.rule == "min_val_RMSE"].set_index("gas")
    e4 = pd.read_csv(OUT_DIR / f"single_basis_metrics_node{args.node}.csv")

    rows = []
    for gas in TARGETS:
        ref_col = cfg.ML_REF_COLS[gas]
        sel = per_gas.loc[gas]
        for config in CONFIGS:
            # Valid rows and block as in the model selection (row mask from the OLS
            # fitted on all training rows)
            refit_ols_on_train(df, train_mask)
            feature_cols = get_feature_columns(gas, config, df)
            valid = df[feature_cols + [ref_col]].notna().all(axis=1)
            idx = np.where((valid & train_mask).values)[0]
            n_val = int(round(VAL_FRAC * len(idx)))
            inner_idx, val_idx = idx[:-n_val], idx[-n_val:]
            inner_mask = pd.Series(False, index=df.index)
            inner_mask.iloc[inner_idx] = True
            # OLS fitted on the rows before the block only
            refit_ols_on_train(df, inner_mask)
            if config in ("B", "C"):
                still_valid = df[feature_cols + [ref_col]].notna().all(axis=1).values
                inner_idx = inner_idx[still_valid[inner_idx]]
                val_idx = val_idx[still_valid[val_idx]]
            # The OLS as a method: scored on the rows of the block
            y_val = df.iloc[val_idx][ref_col].to_numpy(dtype=float)
            p_val = df.iloc[val_idx][f"c_hat_ols_{gas}"].to_numpy(dtype=float)
            ok = np.isfinite(y_val) & np.isfinite(p_val)
            m = evaluate_model(y_val[ok], p_val[ok])
            is_sel = (config == sel.locked_config)
            rows.append({
                "node": args.node, "gas": gas, "config": config,
                "n_train_inner": len(inner_idx), "n_val": len(val_idx), "n_val_scored_ols": int(ok.sum()),
                "val_start": ts_all.iloc[val_idx[0]], "val_end": ts_all.iloc[val_idx[-1]],
                "ols_val_RMSE": m["RMSE"], "ols_val_R2": m["R2"],
                "selected_config": is_sel,
                "ml_model": sel.locked_model, "ml_config": sel.locked_config,
                "ml_val_RMSE": sel.val_RMSE, "ml_n_val": int(sel.n_val), "ml_n_val_scored": int(sel.n_val_scored),
            })
            print(f"  {gas:9s} {config}: inner={len(inner_idx)} val={len(val_idx)} "
                  f"({rows[-1]['val_start']:%Y-%m-%d} .. {rows[-1]['val_end']:%Y-%m-%d}) "
                  f"OLS val RMSE={m['RMSE']:.3f} R2={m['R2']:.3f}"
                  + (f" | ML {sel.locked_model}-{sel.locked_config} val RMSE={sel.val_RMSE:.3f} (n_val of the selection={int(sel.n_val)})" if is_sel else ""))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f"ols_validation_node{args.node}.csv", index=False)

    # Summary per target, with the configuration of the selected ML model
    summ = []
    for gas in TARGETS:
        r = out[(out.gas == gas) & out.selected_config].iloc[0]
        if r.n_val != r.ml_n_val:
            print(f"  WARNING {gas}: reproduced n_val {r.n_val} != selection {r.ml_n_val}")
        ols_better = r.ols_val_RMSE < r.ml_val_RMSE
        ml_r2 = e4_test_r2(e4, gas, "ml", r.ml_model, r.ml_config)
        ols_r2 = e4_test_r2(e4, gas, "ols_train_only")
        summ.append({
            "node": args.node, "gas": gas,
            "ml_model": r.ml_model, "ml_config": r.ml_config, "ml_label": f"{ABBR[r.ml_model]}-{r.ml_config}",
            "ml_val_RMSE": r.ml_val_RMSE, "ols_val_RMSE": r.ols_val_RMSE,
            "ols_vs_ml_val_pct": 100 * (r.ols_val_RMSE / r.ml_val_RMSE - 1),
            "rows_comparable": bool(r.n_val == r.ml_n_val_scored),
            "method_selected": "OLS" if ols_better else "ML",
            "label_selected": "OLS + T/RH" if ols_better else f"{ABBR[r.ml_model]}-{r.ml_config}",
            "ml_R2_test": ml_r2, "ols_R2_test": ols_r2,
            "R2_test_selected": ols_r2 if ols_better else ml_r2,
            "n_val": int(r.n_val), "val_start": r.val_start, "val_end": r.val_end,
            "test_basis": "single_basis_metrics intersection/test (E4)",
        })
    summ = pd.DataFrame(summ)
    path = OUT_DIR / "method_selection_v6.csv"  # Only the rows of this node are rewritten
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[prev.node != args.node]
        summ = pd.concat([prev, summ], ignore_index=True).sort_values(["node", "gas"])
    summ.to_csv(path, index=False)
    pd.set_option("display.width", 220)
    print(summ[summ.node == args.node][["gas", "ml_label", "ml_val_RMSE", "ols_val_RMSE", "ols_vs_ml_val_pct",
                                          "rows_comparable", "method_selected", "ml_R2_test", "ols_R2_test", "R2_test_selected"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
