"""
Evaluation of every calibration method on common rows

Scores the manufacturer curves, the raw SPS30 reading, the Alphasense
factory calibration, the OLS and the saved ML models (not retrained) on
the same rows, for the test period and the full record. Writes
single_basis_metrics_node{N}.csv (one row per method, basis and period).
"""

from __future__ import annotations

import argparse
import ast
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REF_CSV_NAME, load_node_pipeline, versions_banner

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "cross_node"
TARGETS = ["PM2.5", "CO", "NO2", "O3", "NO2_alpha"]


# -------------------------------------------------------------------------
# Scoring and model loading
# -------------------------------------------------------------------------
def metrics_row(evaluate_model, y, pred, **meta):
    """Score one method on one row selection; metrics are NaN below 50 finite pairs."""
    valid = np.isfinite(pred) & np.isfinite(y)
    row = {**meta, "n": int(len(y)), "n_scored": int(valid.sum())}
    if valid.sum() < 50:
        row.update({k: np.nan for k in ("R2", "RMSE", "MAE", "MBE", "nRMSE", "pearson_r")})
        row["CEN_level"] = "n/a"
        return row
    m = evaluate_model(y[valid], pred[valid])
    row.update({k: m[k] for k in ("R2", "RMSE", "MAE", "MBE", "nRMSE", "pearson_r", "CEN_level")})
    return row


def load_model(cfg, row, LSTMRegressor, DNNRegressor):
    """Rebuild a saved model from its row of a master table."""
    import joblib
    path = cfg.RESULTS_S9_MODELS / Path(row["model_path"]).name
    if not path.exists():
        return None, f"missing {path.name}"
    params = ast.literal_eval(row["best_params"]) if isinstance(row["best_params"], str) else {}
    n_features = int(row["n_features"])
    if row["model"] == "LSTM":
        m = LSTMRegressor(random_state=cfg.ML_RANDOM_STATE, window=cfg.LSTM_WINDOW,
                          **{k: params[k] for k in ("lstm_units", "fc_dim", "dropout") if k in params})
        m.load_torch(path, n_features)
    elif row["model"] == "DNN":
        m = DNNRegressor(random_state=cfg.ML_RANDOM_STATE,
                         **{k: params[k] for k in ("hidden_dim_1", "hidden_dim_2", "dropout") if k in params})
        m.load_torch(path, n_features)
    else:
        m = joblib.load(path)
    return m, ""


# -------------------------------------------------------------------------
# Evaluation per target, row basis and period
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate every calibration method on one common set of test observations.")
    ap.add_argument("--node", type=int, choices=(3, 5), required=True)
    ap.add_argument("--gases", nargs="*", default=None)
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("=" * 70)
    print(f"  Single-basis evaluation | Node {args.node}")
    print("=" * 70)
    versions_banner()

    cfg, utils = load_node_pipeline(args.node)
    from ml.data_loader import prepare_dataset, get_train_test_masks, get_feature_columns
    from ml.ols_refit import refit_ols_on_train
    from ml.models import LSTMRegressor, DNNRegressor
    from ml.evaluator import evaluate_model

    s1 = utils.detect_csv(cfg.DATA_S1, f"{cfg.NODE_ID}_QC_RAW_wide_data_*.csv")
    s3 = utils.detect_csv(cfg.DATA_S3, f"{cfg.NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    s4 = utils.detect_csv(cfg.DATA_S4, f"{cfg.NODE_ID}_PREPROCESS_wide_data_*.csv")
    s5 = utils.detect_csv(cfg.DATA_S5, f"{cfg.NODE_ID}_CALIBRATED_CURVAS_FABRICANTE_wide_data_*.csv")
    s6 = utils.detect_csv(cfg.DATA_S6, f"{cfg.NODE_ID}_CALIBRATED_OLS_wide_data_*.csv")
    ref = cfg.REF_DATA_DIR / REF_CSV_NAME[args.node]
    for n, p in (("s1", s1), ("s3", s3), ("s4", s4), ("s5", s5), ("s6", s6), ("ref", ref)):
        if p is None or not Path(p).exists():
            sys.exit(f"ERROR: {n} not found: {p}")
        print(f"  input {n:3s} {Path(p).name}")

    df = prepare_dataset(s1, s3, s4, ref)
    train_mask, test_mask = get_train_test_masks(df)
    refit_ols_on_train(df, train_mask)
    ts = pd.to_datetime(df[cfg.COL_TS], utc=True)

    # Manufacturer curves (s5) and full-period OLS (s6), averaged to the ML time step
    def join_15min(csv, cols):
        d = pd.read_csv(csv, usecols=[cfg.COL_TS] + cols)
        d[cfg.COL_TS] = pd.to_datetime(d[cfg.COL_TS], utc=True)
        return d.set_index(cfg.COL_TS).resample(cfg.ML_RESAMPLE_FREQ).mean()

    fab_cols = list(cfg.CURVAS_FAB_COL_NAMES.values())
    ols6_cols = list(cfg.OLS_COL_NAMES.values())
    df = (df.set_index(ts)
            .join(join_15min(s5, fab_cols), how="left")
            .join(join_15min(s6, ols6_cols), how="left")
            .reset_index(drop=True))
    df[cfg.COL_TS] = ts.values
    print(f"  frame: {len(df)} rows; train {int(train_mask.sum())}, test {int(test_mask.sum())}")

    # Master tables of the saved models
    tables = []
    for name in ("tabla_maestra_resultados.csv", "tabla_maestra_no2alpha.csv",
                 "tabla_maestra_no2alpha_lstm.csv", "tabla_maestra_lstm_v6.csv"):
        f = cfg.RESULTS_S9 / name
        if f.exists():
            d = pd.read_csv(f)
            d["source_table"] = name
            tables.append(d)
    models = pd.concat(tables, ignore_index=True)
    models = models[models["model_path"].notna()]

    periods = {"test": test_mask, "full": pd.Series(True, index=df.index)}
    rows = []
    gases = args.gases or TARGETS
    for gas in gases:
        ref_col = cfg.ML_REF_COLS[gas]
        feats = {c: get_feature_columns(gas, c, df) for c in ("A", "B", "C")}
        valid = {c: df[feats[c] + [ref_col]].notna().all(axis=1) for c in feats}
        base_specs = []  # Methods without ML: (method, pred_col, clip_zero)
        base_specs.append(("ols_train_only", f"c_hat_ols_{gas}", False))
        base_specs.append(("ols_s6_full_period", cfg.OLS_COL_NAMES[gas], False))
        if gas in cfg.CURVAS_FAB_COL_NAMES:
            base_specs.append(("manufacturer_curve", cfg.CURVAS_FAB_COL_NAMES[gas], False))
        if gas == "PM2.5":
            base_specs.append(("sps30_raw", cfg.COL_PM25, False))
        if gas == "NO2_alpha":
            base_specs.append(("alphasense_factory", cfg.COL_ALPHA_NO2, False))
            base_specs.append(("alphasense_factory", cfg.COL_ALPHA_NO2, True))
        # Intersection basis: rows valid for A, B and C with every non-ML column present
        base_cols = [c for _, c, _ in base_specs if c in df.columns]
        inter_valid = valid["A"] & valid["B"] & valid["C"] & df[base_cols].notna().all(axis=1)
        print(f"\n  [{gas}] rows valid A/B/C = {int(valid['A'].sum())}/{int(valid['B'].sum())}/"
              f"{int(valid['C'].sum())}; intersection = {int(inter_valid.sum())} "
              f"(test: {int((inter_valid & test_mask).sum())})")
        if args.dry_run:
            continue

        # Methods without ML on three bases: own rows, rows of configuration A, intersection
        for period, pmask in periods.items():
            for method, col, clip in base_specs:
                if col not in df.columns:
                    print(f"    {method}: column {col} missing, skipped")
                    continue
                for basis, bmask in (("own", df[[col, ref_col]].notna().all(axis=1)),
                                     ("configA", valid["A"]), ("intersection", inter_valid)):
                    sel = bmask & pmask
                    y = df.loc[sel, ref_col].to_numpy(dtype=float)
                    pred = df.loc[sel, col].to_numpy(dtype=float)
                    if clip:
                        # Estimate and reference clipped at zero, the convention of s7
                        y, pred = np.clip(y, 0, None), np.clip(pred, 0, None)
                    rows.append(metrics_row(evaluate_model, y, pred, node=args.node, gas=gas,
                                            method=method, model="-", config="-", basis=basis,
                                            period=period, ref_col=ref_col, pred_col=col,
                                            clip_zero=clip, source_table="-",
                                            orig_R2_test=np.nan, repro_diff=np.nan))

        if args.no_ml:
            continue
        # ML: every saved combination, on its own rows and on the intersection
        sub = models[models.gas == gas]
        for _, r in sub.iterrows():
            config = r["config"]
            m, err = load_model(cfg, r, LSTMRegressor, DNNRegressor)
            tag = "LSTM_v6" if r["source_table"] == "tabla_maestra_lstm_v6.csv" else r["model"]
            if m is None:
                print(f"    {tag}-{config}: {err}")
                continue
            fcols = feats[config]
            for period, pmask in periods.items():
                for basis, bmask in (("own", valid[config]), ("intersection", inter_valid)):
                    sel = bmask & pmask
                    if sel.sum() < 50:
                        continue
                    X = df.loc[sel, fcols].to_numpy(dtype=float)
                    y = df.loc[sel, ref_col].to_numpy(dtype=float)
                    if r["model"] == "LSTM":
                        pred = m.predict(X, timestamps=df.loc[sel, cfg.COL_TS].values)
                    else:
                        pred = np.asarray(m.predict(X), dtype=float)
                    orig = float(r["R2_test"]) if pd.notna(r["R2_test"]) else np.nan
                    row = metrics_row(evaluate_model, y, pred, node=args.node, gas=gas,
                                      method="ml", model=tag, config=config, basis=basis,
                                      period=period, ref_col=ref_col, pred_col="-",
                                      clip_zero=False, source_table=r["source_table"],
                                      orig_R2_test=orig, repro_diff=np.nan)
                    # Reproduction check against the test R2 stored by s9
                    if basis == "own" and period == "test" and np.isfinite(orig):
                        row["repro_diff"] = row["R2"] - orig
                    rows.append(row)
            own_test = [x for x in rows if x["gas"] == gas and x["model"] == tag
                        and x["config"] == config and x["basis"] == "own" and x["period"] == "test"]
            if own_test:
                x = own_test[-1]
                print(f"    {tag}-{config}: own/test R2={x['R2']:.4f} (orig {x['orig_R2_test']:.4f}, "
                      f"diff {x['repro_diff']:+.5f}); intersection/test R2="
                      f"{next((z['R2'] for z in rows if z['gas']==gas and z['model']==tag and z['config']==config and z['basis']=='intersection' and z['period']=='test'), np.nan):.4f}")
            del m
            gc.collect()

    if args.dry_run:
        print(f"\n  dry-run OK ({(time.time() - t0) / 60:.1f} min)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"single_basis_metrics_node{args.node}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  written {out_path} ({len(out)} rows) in {(time.time() - t0) / 60:.1f} min")
    # Summary: intersection basis, test period, eight highest R2 per target
    it = out[(out.basis == "intersection") & (out.period == "test")]
    for gas in gases:
        g = it[it.gas == gas].sort_values("R2", ascending=False)
        print(f"\n  {gas} (intersection, test, n={int(g['n'].max()) if len(g) else 0}):")
        for _, x in g.head(8).iterrows():
            print(f"    {x['method']:20s} {x['model']:8s} {x['config']:2s} R2={x['R2']:+.3f} "
                  f"RMSE={x['RMSE']:.2f} n_scored={x['n_scored']}")


if __name__ == "__main__":
    main()
