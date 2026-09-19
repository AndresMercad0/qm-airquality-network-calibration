"""
OLS refitted on the training period (s9)

Refits C_ref = b0 + b1*signal + b2*T + b3*RH for each pollutant with the
training rows only, so that no test data leak into the ML features.
Adds the prediction over the whole dataset as c_hat_ols_{gas} and
returns the coefficients and the training R2.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import (
    COL_TEMP, COL_RH, COL_PM25, COL_ALPHA_NO2,
    ML_REF_COLS, POLLUTANTS_ML,
)

log = logging.getLogger("s9.ols_refit")

# Ratio column of each gas in s4
_RATIO_COLS = {
    "NO2": "ratio_NO2",
    "CO":  "ratio_CO",
    "O3":  "ratio_O3",
}


# -------------------------------------------------------------------------
# OLS refitted on the training period (no data leakage)
# -------------------------------------------------------------------------
def refit_ols_on_train(df, train_mask):
    """
    Refit the OLS of each pollutant on the training period only, to avoid data leakage.
    The signal is ratio_{gas} for the low-cost gases, the SPS30 PM2.5 for PM2.5 and the
    Alphasense concentration for NO2_alpha, each with T and RH. The prediction over the
    whole dataset goes to c_hat_ols_{gas}, a feature of configs B and C.
    Return the coefficients per gas.
    """
    log.info("Refitting the OLS on the training period only (anti-leakage)...")
    ols_info = {}

    for gas in POLLUTANTS_ML:
        ref_col = ML_REF_COLS[gas]

        # OLS features
        if gas == "PM2.5":
            feat_cols = [COL_PM25, COL_TEMP, COL_RH]
        elif gas == "NO2_alpha":
            feat_cols = [COL_ALPHA_NO2, COL_TEMP, COL_RH]
        else:
            ratio_col = _RATIO_COLS.get(gas)
            if ratio_col is None or ratio_col not in df.columns:
                log.warning("  %s: ratio column not available, OLS refit skipped", gas)
                df[f"c_hat_ols_{gas}"] = np.nan
                continue
            feat_cols = [ratio_col, COL_TEMP, COL_RH]

        # Valid rows of the training period
        all_cols = feat_cols + [ref_col]
        valid = df[all_cols].notna().all(axis=1)
        train_valid = valid & train_mask
        n_train = train_valid.sum()

        if n_train < 50:
            log.warning("  %s: only %d valid training obs, OLS refit skipped", gas, n_train)
            df[f"c_hat_ols_{gas}"] = np.nan
            continue

        # Fit on the training period only
        X_train = df.loc[train_valid, feat_cols].values
        y_train = df.loc[train_valid, ref_col].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on every row with valid features
        all_valid = df[feat_cols].notna().all(axis=1)
        X_all = df.loc[all_valid, feat_cols].values
        predictions = model.predict(X_all)

        df[f"c_hat_ols_{gas}"] = np.nan
        df.loc[all_valid, f"c_hat_ols_{gas}"] = predictions

        # Coefficients and training R2
        coefs = dict(zip(feat_cols, model.coef_))
        coefs["intercept"] = model.intercept_
        ols_info[gas] = {
            "coefs": coefs,
            "n_train": int(n_train),
            "r2_train": float(model.score(X_train, y_train)),
        }

        log.info("  %s OLS refit: R2_train=%.3f, n=%d, coefs=%s",
                 gas, ols_info[gas]["r2_train"], n_train,
                 {k: f"{v:.4f}" for k, v in coefs.items()})

    return ols_info
