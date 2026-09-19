"""
Feature table of the partner node

Builds, for the transfer experiment, the 15-minute table of the other
node: low-cost signals, Alphasense series, sensor ratios, reference
station and the OLS prediction fitted at that node. Not a pipeline step.
"""

import logging

import numpy as np
import pandas as pd

from config import (
    COL_TS, COL_TEMP, COL_RH,
    REF_DATA_DIR, PARTNER_NODE_ID, PARTNER_DATA_DIR,
    PARTNER_REF_CSV_PATTERN, PARTNER_REF_STATION_SHORT,
    POLLUTANTS_ML, ML_RESAMPLE_FREQ,
)
from utils import detect_csv, compute_abs_humidity, add_temporal_features

log = logging.getLogger("cross_node")


# -------------------------------------------------------------------------
# Reference column of each pollutant at the destination node
# -------------------------------------------------------------------------
PARTNER_REF_COLS = {
    "NO2":   "no2_high_met_one_ug_m3",
    "CO":    "co_mid_alphasense_ug_m3_aan803",
    "O3":    "o3_mid_alphasense_ug_m3_aan803",
    "PM2.5": "pm2_5_high_met_one_ug_m3",
}


# -------------------------------------------------------------------------
# Data of the destination node (steps s1, s3 and s4) and its reference
# -------------------------------------------------------------------------
def _load_partner_data():
    """Return the s1, s3 and s4 files of the partner node and its reference file."""
    dst_s1 = None
    dst_s3 = None
    dst_s4 = None
    dst_data = PARTNER_DATA_DIR

    if dst_data.exists():
        s1_dir = dst_data / "s1_qc_lowcost"
        s3_dir = dst_data / "s3_qc_alphasense"
        s4_dir = dst_data / "s4_preprocess"
        if s1_dir.exists():
            dst_s1 = detect_csv(s1_dir, f"{PARTNER_NODE_ID}_QC_RAW_wide_data_*.csv")
        if s3_dir.exists():
            dst_s3 = detect_csv(s3_dir, f"{PARTNER_NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
        if s4_dir.exists():
            dst_s4 = detect_csv(s4_dir, f"{PARTNER_NODE_ID}_PREPROCESS_wide_data_*.csv")

    ref_csv = detect_csv(REF_DATA_DIR, PARTNER_REF_CSV_PATTERN) \
        if REF_DATA_DIR.exists() else None

    return dst_s1, dst_s3, dst_s4, ref_csv


def _prepare_partner_features(dst_csv, dst_s3_csv, dst_s4_csv, ref_csv):
    """Build the 15-minute feature table of the destination node."""
    log.info("Preparing features of %s...", PARTNER_NODE_ID)
    df_dst = pd.read_csv(dst_csv, parse_dates=[COL_TS])

    # Resample to 15 minutes
    numeric_cols = df_dst.select_dtypes(include=[np.number]).columns.tolist()
    df_dst = df_dst.set_index(COL_TS)[numeric_cols].resample(ML_RESAMPLE_FREQ).mean().reset_index()

    # Alphasense series: comparators for CO and O3
    if dst_s3_csv is not None and dst_s3_csv.exists():
        log.info("Loading s3 Alphasense: %s", dst_s3_csv.name)
        df_s3 = pd.read_csv(dst_s3_csv, parse_dates=[COL_TS])
        alpha_cols = [c for c in df_s3.columns if "alphasense" in c and c != COL_TS]
        df_s3 = df_s3[[COL_TS] + alpha_cols]
        df_s3 = df_s3.set_index(COL_TS).resample(ML_RESAMPLE_FREQ).mean().reset_index()
        df_dst = df_dst.merge(df_s3, on=COL_TS, how="left")

    # Sensor ratios: inputs of the OLS
    if dst_s4_csv is not None and dst_s4_csv.exists():
        log.info("Loading s4 ratios: %s", dst_s4_csv.name)
        df_s4 = pd.read_csv(dst_s4_csv, parse_dates=[COL_TS])
        ratio_cols = [c for c in df_s4.columns if c.startswith("ratio_")]
        df_s4 = df_s4[[COL_TS] + ratio_cols]
        df_s4 = df_s4.set_index(COL_TS).resample(ML_RESAMPLE_FREQ).mean().reset_index()
        df_dst = df_dst.merge(df_s4, on=COL_TS, how="left")

    # Derived features
    if COL_TEMP in df_dst.columns and COL_RH in df_dst.columns:
        df_dst["abs_humidity"] = compute_abs_humidity(
            df_dst[COL_TEMP].values, df_dst[COL_RH].values
        )
    df_dst = add_temporal_features(df_dst, COL_TS)

    # Reference station: NO2 and PM2.5
    if ref_csv is not None:
        log.info("Loading %s reference: %s", PARTNER_REF_STATION_SHORT, ref_csv.name)
        df_ref = pd.read_csv(ref_csv, parse_dates=[COL_TS])
        ref_cols = ["no2_high_met_one_ug_m3", "pm2_5_high_met_one_ug_m3"]
        ref_available = [c for c in ref_cols if c in df_ref.columns]
        df_ref = df_ref[[COL_TS] + ref_available]
        df_ref = df_ref.set_index(COL_TS).resample(ML_RESAMPLE_FREQ).mean().reset_index()
        df_dst = df_dst.merge(df_ref, on=COL_TS, how="left")

    # OLS prediction, an input of configurations B and C
    _compute_ols_predictions(df_dst, PARTNER_REF_COLS)

    log.info("Destination dataset: %d rows at 15 min, %d columns",
             len(df_dst), len(df_dst.columns))

    return df_dst


def _compute_ols_predictions(df, ref_cols):
    """Fit the OLS of each pollutant on the destination data and store its prediction."""
    from sklearn.linear_model import LinearRegression

    ratio_map = {"NO2": "ratio_NO2", "CO": "ratio_CO", "O3": "ratio_O3"}

    for gas in POLLUTANTS_ML:
        ref_col = ref_cols.get(gas)
        if ref_col is None or ref_col not in df.columns:
            df[f"c_hat_ols_{gas}"] = np.nan
            continue

        if gas == "PM2.5":
            feat_cols = ["pm2_5_low_sps30_ug_m3", COL_TEMP, COL_RH]
        else:
            ratio_col = ratio_map.get(gas)
            if ratio_col is None or ratio_col not in df.columns:
                df[f"c_hat_ols_{gas}"] = np.nan
                continue
            feat_cols = [ratio_col, COL_TEMP, COL_RH]

        all_cols = feat_cols + [ref_col]
        valid = df[all_cols].notna().all(axis=1)
        n_valid = valid.sum()

        if n_valid < 50:
            df[f"c_hat_ols_{gas}"] = np.nan
            continue

        X = df.loc[valid, feat_cols].values
        y = df.loc[valid, ref_col].values

        model = LinearRegression().fit(X, y)
        predictions = model.predict(X)

        df[f"c_hat_ols_{gas}"] = np.nan
        df.loc[valid, f"c_hat_ols_{gas}"] = predictions

        log.info("  OLS cross-node %s: R2=%.3f, n=%d", gas, model.score(X, y), n_valid)
