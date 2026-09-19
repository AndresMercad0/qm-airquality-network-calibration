"""
Monthly metrics of the OLS-calibrated NO2 (Alphasense cell and GM-102B)

Computes R2 and RMSE per month against the LAQN NO2 for two OLS fits: the
OLS of s6 fitted on the full period (s6_full_period) and the OLS refitted
on the training rows only (train_only). Writes the monthly series
metricas_alphasense_no2_ols_mensual_v6.csv of the node.
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
# Monthly metrics of one prediction column
# -------------------------------------------------------------------------
def monthly(df, ref_col, pred_col, min_obs, train_mask, test_mask, label):
    """
    Compute R2 and RMSE per month, for months with at least min_obs valid rows.
    in_sample is True when the month holds no row of the test period; October is
    in-sample although the rows of 31 October belong to neither set.
    """
    rows = []
    d = df[[ref_col, pred_col, "month", ]].copy()
    d["test"] = test_mask.values
    for m, g in d.groupby("month"):
        g = g.dropna(subset=[ref_col, pred_col])
        if len(g) < min_obs:
            continue
        from utils import compute_ml_metrics
        met = compute_ml_metrics(g[ref_col].values, g[pred_col].values)
        rows.append({"month": str(m), "R2": met["R2"], "RMSE": met["RMSE"],
                     "n_obs": int(len(g)), "in_sample": bool(~g["test"].any()), **label})
    return rows


# -------------------------------------------------------------------------
# Series per sensor and fitting basis
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Monthly metrics of the OLS-calibrated NO2 on two fitting bases.")
    ap.add_argument("--node", type=int, choices=(3, 5), required=True)
    args = ap.parse_args()

    cfg, utils = load_node_pipeline(args.node)
    from ml.data_loader import prepare_dataset, get_train_test_masks
    from ml.ols_refit import refit_ols_on_train

    s1 = utils.detect_csv(cfg.DATA_S1, f"{cfg.NODE_ID}_QC_RAW_wide_data_*.csv")
    s3 = utils.detect_csv(cfg.DATA_S3, f"{cfg.NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    s4 = utils.detect_csv(cfg.DATA_S4, f"{cfg.NODE_ID}_PREPROCESS_wide_data_*.csv")
    s6 = utils.detect_csv(cfg.DATA_S6, f"{cfg.NODE_ID}_CALIBRATED_OLS_wide_data_*.csv")
    ref = cfg.REF_DATA_DIR / REF_CSV_NAME[args.node]
    for n, p in (("s1", s1), ("s3", s3), ("s4", s4), ("s6", s6), ("ref", ref)):
        if p is None or not Path(p).exists():
            sys.exit(f"ERROR: {n} not found: {p}")
        print(f"  input {n:3s} {Path(p).name}")

    df = prepare_dataset(s1, s3, s4, ref)
    train_mask, test_mask = get_train_test_masks(df)
    refit_ols_on_train(df, train_mask)  # Adds c_hat_ols_* for the five targets

    # Full-period OLS (s6) averaged to the ML time step and joined by timestamp
    ols_cols = [cfg.OLS_COL_NAMES["NO2_alpha"], cfg.OLS_COL_NAMES["NO2"]]
    d6 = pd.read_csv(s6, usecols=[cfg.COL_TS] + ols_cols)
    d6[cfg.COL_TS] = pd.to_datetime(d6[cfg.COL_TS], utc=True)
    d6 = d6.set_index(cfg.COL_TS).resample(cfg.ML_RESAMPLE_FREQ).mean()
    ts = pd.to_datetime(df[cfg.COL_TS], utc=True)
    df = df.set_index(ts).join(d6, how="left").reset_index(drop=True)
    df["month"] = ts.dt.to_period("M").values

    ref_col = cfg.ML_REF_COLS["NO2_alpha"]  # LAQN NO2 (same reference for both sensors)
    rows = []
    specs = [
        ("NO2_alpha", cfg.OLS_COL_NAMES["NO2_alpha"], "s6_full_period"),
        ("NO2_alpha", "c_hat_ols_NO2_alpha", "train_only"),
        ("NO2", cfg.OLS_COL_NAMES["NO2"], "s6_full_period"),
        ("NO2", "c_hat_ols_NO2", "train_only"),
    ]
    for gas, pred_col, basis in specs:
        if pred_col not in df.columns:
            print(f"  WARNING: {pred_col} not available, skipped")
            continue
        label = {"gas": gas, "source": "OLS_alphasense" if gas == "NO2_alpha" else "OLS",
                 "model": "OLS_Alphasense" if gas == "NO2_alpha" else "OLS", "config": "-",
                 "basis": basis}
        rows += monthly(df, ref_col, pred_col, cfg.ML_MIN_OBS_DRIFT, train_mask, test_mask, label)
    out = pd.DataFrame(rows)[["gas", "month", "source", "model", "config", "basis",
                              "in_sample", "R2", "RMSE", "n_obs"]]
    out_path = cfg.RESULTS_S9 / "metricas_alphasense_no2_ols_mensual_v6.csv"
    out.to_csv(out_path, index=False)
    print(f"  written {out_path.name} ({len(out)} rows)")

    # Difference from the published monthly series
    old_path = cfg.RESULTS_S9 / "metricas_alphasense_no2_ols_mensual.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        new = out[(out.gas == "NO2_alpha") & (out.basis == "s6_full_period")]
        cmp = old.merge(new, on="month", suffixes=("_old", "_v6"))
        cmp["dR2"] = cmp["R2_v6"] - cmp["R2_old"]
        cmp["dn"] = cmp["n_obs_v6"] - cmp["n_obs_old"]
        print("\n  diff vs published series (s6_full_period basis):")
        print(cmp[["month", "R2_old", "R2_v6", "dR2", "n_obs_old", "n_obs_v6", "dn"]]
              .round(4).to_string(index=False))
    print("\n  train_only basis (NO2_alpha):")
    print(out[(out.gas == "NO2_alpha") & (out.basis == "train_only")]
          [["month", "in_sample", "R2", "n_obs"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
