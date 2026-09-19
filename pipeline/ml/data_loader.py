"""
Dataset, feature configurations and train/test split (s9)

Resamples the s1, s3 and s4 outputs and the reference station to 15 min
means, joins them and adds absolute humidity and time features. Also
returns the feature columns of each (gas, config) pair and the masks
of the chronological train/test split.
"""

import logging

import numpy as np
import pandas as pd

from config import (
    COL_TS, COL_TEMP, COL_RH, COL_PM25, COL_PM10,
    COL_NO2_RAW, COL_CO_RAW, COL_O3_VOLTAGE,
    COL_VOC_RAW, COL_ALPHA_CO, COL_ALPHA_O3, COL_ALPHA_NO2,
    COL_REF_NO2, COL_REF_PM25, REF_STATION_SHORT,
    ML_FEATURES_A_GAS, ML_FEATURES_A_PM25,
    ML_FEATURES_B_BASE_GAS, ML_FEATURES_B_BASE_PM25,
    ML_FEATURES_A_NO2_ALPHA, ML_FEATURES_B_BASE_NO2_ALPHA,
    ML_FEATURES_C_BASE,
    ML_FEATURES_D_GAS, ML_FEATURES_D_PM25, ML_FEATURES_D_STRICT,
    ML_FEATURES_S_GAS, ML_FEATURES_S_PM25, ML_FEATURES_S1,
    ML_TRAIN_END, ML_TEST_START,
    ML_RESAMPLE_FREQ,
)
from utils import compute_abs_humidity, add_temporal_features

log = logging.getLogger("s9.data_loader")


# -------------------------------------------------------------------------
# Unified 15 min dataset
# -------------------------------------------------------------------------
def _load_and_resample(csv_path, cols_to_keep, source_name):
    """Load a CSV, keep the requested columns and resample to 15 min means."""
    log.info("  Loading %s: %s", source_name, csv_path.name)
    df = pd.read_csv(csv_path, parse_dates=[COL_TS])
    available = [c for c in cols_to_keep if c in df.columns]
    df = df[[COL_TS] + available].copy()
    df = df.set_index(COL_TS).resample(ML_RESAMPLE_FREQ).mean()
    log.info("    -> %d 15-min rows, %d columns", len(df), len(df.columns))
    return df


def prepare_dataset(s1_csv, s3_csv, s4_csv, ref_csv):
    """
    Build the unified 15 min dataset of the machine learning calibration.
    Each source is resampled to 15 min means and joined on the timestamp (outer join).
    Absolute humidity and the time features are added. The timestamp is returned as a column.
    """
    log.info("Preparing the unified 15-min dataset...")

    # s1: raw low-cost signals, T, RH, VOC and PM
    s1_cols = [COL_NO2_RAW, COL_CO_RAW, COL_O3_VOLTAGE,
               COL_TEMP, COL_RH, COL_VOC_RAW, COL_PM25, COL_PM10]
    df_s1 = _load_and_resample(s1_csv, s1_cols, "s1 QC RAW")

    # s3: Alphasense concentrations after quality control (ug/m3, mid-cost)
    s3_cols = [COL_ALPHA_CO, COL_ALPHA_O3, COL_ALPHA_NO2]
    df_s3 = _load_and_resample(s3_csv, s3_cols, "s3 QC Alphasense")

    # s4: preprocessed ratios, used by the OLS refit
    s4_cols = ["ratio_CO", "ratio_NO2", "ratio_O3"]
    df_s4 = _load_and_resample(s4_csv, s4_cols, "s4 Preprocess")

    # Reference station (high-cost): NO2 and PM2.5
    ref_cols = [COL_REF_NO2, COL_REF_PM25]
    df_ref = _load_and_resample(ref_csv, ref_cols, f"{REF_STATION_SHORT} reference")

    # Outer join on the 15 min index
    df = df_s1.join(df_s3, how="outer")
    df = df.join(df_s4, how="outer")
    df = df.join(df_ref, how="outer")

    # Timestamp back as a column
    df = df.reset_index()
    df = df.rename(columns={"index": COL_TS})

    # Derived features
    df["abs_humidity"] = compute_abs_humidity(
        df[COL_TEMP].values, df[COL_RH].values
    )
    df = add_temporal_features(df, COL_TS)

    n_valid_no2 = df[COL_REF_NO2].notna().sum()
    n_valid_co = df[COL_ALPHA_CO].notna().sum()
    n_valid_pm25 = df[COL_REF_PM25].notna().sum()

    log.info("Unified dataset: %d 15-min rows", len(df))
    log.info("  NO2 reference (%s): %d valid periods", REF_STATION_SHORT, n_valid_no2)
    log.info("  CO reference (Alphasense): %d valid periods", n_valid_co)
    log.info("  PM2.5 reference (%s): %d valid periods", REF_STATION_SHORT, n_valid_pm25)

    return df


# -------------------------------------------------------------------------
# Feature configurations
# -------------------------------------------------------------------------
def get_feature_columns(gas, config, df):
    """
    Return the feature columns of a (gas, config) pair that exist in the dataset.
    Config A: raw signals, ambient, derived and time features. Config B: config A
    plus c_hat_ols_{gas}. Config C: c_hat_ols_{gas} with T, RH and time features.
    """
    ols_col = f"c_hat_ols_{gas}"

    if config == "A":
        if gas == "PM2.5":
            features = ML_FEATURES_A_PM25.copy()
        elif gas == "NO2_alpha":
            features = ML_FEATURES_A_NO2_ALPHA.copy()
        else:
            features = ML_FEATURES_A_GAS.copy()

    elif config == "B":
        if gas == "PM2.5":
            features = ML_FEATURES_B_BASE_PM25.copy()
        elif gas == "NO2_alpha":
            features = ML_FEATURES_B_BASE_NO2_ALPHA.copy()
        else:
            features = ML_FEATURES_B_BASE_GAS.copy()
        features.append(ols_col)

    elif config == "C":
        features = ML_FEATURES_C_BASE.copy()
        features = [ols_col] + features

    # Ablation configs, without the OLS prediction (ML_FEATURES_D_* and S_* in config.py)
    elif config == "D":
        if gas == "PM2.5":
            features = ML_FEATURES_D_PM25.copy()
        elif gas == "NO2_alpha":
            features = ML_FEATURES_D_STRICT.copy()
        else:
            features = ML_FEATURES_D_GAS.copy()
    elif config == "D_strict":
        features = ML_FEATURES_D_STRICT.copy()
    elif config == "S":
        if gas == "PM2.5":
            features = ML_FEATURES_S_PM25.copy()
        elif gas == "NO2_alpha":
            features = ML_FEATURES_S1["NO2_alpha"].copy()
        else:
            features = ML_FEATURES_S_GAS.copy()
    elif config == "S1":
        features = ML_FEATURES_S1[gas].copy()

    else:
        raise ValueError(f"Unknown config: {config}")

    # Only the columns present in the dataset
    available = [c for c in features if c in df.columns]
    if len(available) < len(features):
        missing = set(features) - set(available)
        log.warning("  Missing features for %s Config %s: %s", gas, config, missing)

    return available


# -------------------------------------------------------------------------
# Chronological train/test split
# -------------------------------------------------------------------------
def get_train_test_masks(df, train_end=None, test_start=None):
    """Return the boolean masks (train_mask, test_mask) of the chronological split."""
    if train_end is None:
        train_end = ML_TRAIN_END
    if test_start is None:
        test_start = ML_TEST_START

    ts = pd.to_datetime(df[COL_TS])
    # The limits take the timezone of the dataset (UTC when tz-aware)
    tz = ts.dt.tz
    train_mask = ts <= pd.Timestamp(train_end, tz=tz)
    test_mask = ts >= pd.Timestamp(test_start, tz=tz)

    log.info("Chronological split: train <= %s (%d), test >= %s (%d)",
             train_end, train_mask.sum(), test_start, test_mask.sum())

    return train_mask, test_mask
