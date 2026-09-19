"""
Monthly R2 of the train-only OLS and of the selected ML model

Computes R2 and RMSE per month for the OLS fitted on the training rows
and for the ML model selected on validation (lowest validation RMSE),
reloaded from its saved file; selected DNN and LSTM models are skipped.
Writes the monthly series metricas_ols_ml_mensual_v6.csv of the node.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REF_CSV_NAME, load_node_pipeline

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------
RES_DIR = Path(__file__).resolve().parents[1] / "results" / "cross_node"
TARGETS = ["PM2.5", "CO", "NO2", "O3", "NO2_alpha"]


# -------------------------------------------------------------------------
# Monthly metrics of one prediction column
# -------------------------------------------------------------------------
def monthly(df, ref_col, pred_col, min_obs, test_mask, compute_ml_metrics, label):
    """
    Compute R2 and RMSE per month, for months with at least min_obs valid rows.
    in_sample is True when the month holds no row of the test period.
    """
    rows = []
    d = df[[ref_col, pred_col, "month"]].copy()
    d["test"] = test_mask.values
    for m, g in d.groupby("month"):
        g = g.dropna(subset=[ref_col, pred_col])
        if len(g) < min_obs:
            continue
        met = compute_ml_metrics(g[ref_col].values, g[pred_col].values)
        rows.append({"month": str(m), "R2": met["R2"], "RMSE": met["RMSE"],
                     "n_obs": int(len(g)), "in_sample": bool(~g["test"].any()), **label})
    return rows


# -------------------------------------------------------------------------
# Monthly series per target
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Monthly R2 of the train-only OLS and of the ML model selected on validation.")
    ap.add_argument("--node", type=int, choices=(3, 5), required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    print(f"Monthly OLS vs selected ML | Node {args.node}")

    cfg, utils = load_node_pipeline(args.node)
    from ml.data_loader import prepare_dataset, get_train_test_masks, get_feature_columns
    from ml.ols_refit import refit_ols_on_train
    from utils import compute_ml_metrics
    import joblib

    s1 = utils.detect_csv(cfg.DATA_S1, f"{cfg.NODE_ID}_QC_RAW_wide_data_*.csv")
    s3 = utils.detect_csv(cfg.DATA_S3, f"{cfg.NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    s4 = utils.detect_csv(cfg.DATA_S4, f"{cfg.NODE_ID}_PREPROCESS_wide_data_*.csv")
    ref = cfg.REF_DATA_DIR / REF_CSV_NAME[args.node]
    df = prepare_dataset(s1, s3, s4, ref)
    train_mask, test_mask = get_train_test_masks(df)
    refit_ols_on_train(df, train_mask)  # Train-only c_hat_ols_* columns (also inputs of B and C)
    ts = pd.to_datetime(df[cfg.COL_TS], utc=True)
    df["month"] = ts.dt.to_period("M").values

    per_gas = pd.read_csv(RES_DIR / f"locked_selection_per_gas_node{args.node}.csv")
    per_gas = per_gas[per_gas.rule == "min_val_RMSE"].set_index("gas")
    val = pd.read_csv(RES_DIR / f"locked_selection_validation_node{args.node}.csv")

    rows = []
    for gas in TARGETS:
        ref_col = cfg.ML_REF_COLS[gas]
        rows += monthly(df, ref_col, f"c_hat_ols_{gas}", cfg.ML_MIN_OBS_DRIFT, test_mask, compute_ml_metrics,
                        {"gas": gas, "source": "OLS", "model": "OLS", "config": "-", "basis": "train_only"})
        sel = per_gas.loc[gas]
        if sel.locked_model in ("LSTM", "DNN"):
            print(f"  {gas}: selected model {sel.locked_model}-{sel.locked_config} (torch), skipped")
            continue
        vrow = val[(val.gas == gas) & (val.model == sel.locked_model) & (val.config == sel.locked_config)].iloc[0]
        path = cfg.RESULTS_S9_MODELS / Path(vrow["model_path"]).name
        model = joblib.load(path)
        feats = get_feature_columns(gas, sel.locked_config, df)
        ok = df[feats].notna().all(axis=1)
        pred_col = f"ml_pred_{gas}"
        df[pred_col] = np.nan
        df.loc[ok, pred_col] = np.asarray(model.predict(df.loc[ok, feats].to_numpy(float)), float)
        # Check: overall test R2 of the reloaded model against the stored value
        t = df[test_mask & ok & df[ref_col].notna()]
        r2_test = compute_ml_metrics(t[ref_col].values, t[pred_col].values)["R2"]
        print(f"  {gas}: {sel.locked_model}-{sel.locked_config} reloaded test R2 {r2_test:.3f} "
              f"(stored {sel.R2_test_locked:.3f}, n={len(t)})")
        rows += monthly(df, ref_col, pred_col, cfg.ML_MIN_OBS_DRIFT, test_mask, compute_ml_metrics,
                        {"gas": gas, "source": "ML", "model": sel.locked_model, "config": sel.locked_config,
                         "basis": "validation_selected"})
    out = pd.DataFrame(rows)[["gas", "month", "source", "model", "config", "basis", "in_sample", "R2", "RMSE", "n_obs"]]
    out_path = cfg.RESULTS_S9 / "metricas_ols_ml_mensual_v6.csv"
    out.to_csv(out_path, index=False)
    print(f"  written {out_path}")
    pd.set_option("display.width", 200)
    for gas in TARGETS:
        sub = out[out.gas == gas].pivot(index="month", columns="source", values="R2")
        if "ML" in sub:
            sub = sub[["OLS", "ML"]]
        print(f"\n  {gas} (R2 per month; test months from 2025-11):")
        print(sub.round(2).T.to_string())


if __name__ == "__main__":
    main()
