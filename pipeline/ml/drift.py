"""
Model selection and monthly drift diagnostic (s9)

Selects the best (model, config) of each pollutant by test R2, with a
simpler model preferred within 0.05. Evaluates that fixed model and the
refitted OLS month by month (R2, RMSE); s9 writes this table as
metricas_drift_mensual.csv. The slope of the monthly R2 is logged.
"""

import logging

import numpy as np
import pandas as pd

from config import (
    COL_TS, POLLUTANTS_ML, ML_REF_COLS,
    ML_DRIFT_RAPID, ML_DRIFT_MODERATE, ML_MIN_OBS_DRIFT,
)
from utils import compute_ml_metrics
from .data_loader import get_feature_columns

log = logging.getLogger("s9.drift")


# -------------------------------------------------------------------------
# Model selection
# -------------------------------------------------------------------------
def select_best_model_per_gas(results_df):
    """
    Select the best (model, config) of each pollutant by test R2.
    Tie-break: the first simpler model within 0.05 of the best R2 is preferred
    (RF, then XGBoost and LightGBM, then DNN, then LSTM).
    """
    complexity_order = {"RF": 0, "XGBoost": 1, "LightGBM": 1, "DNN": 2, "LSTM": 3}
    best = {}

    for gas in POLLUTANTS_ML:
        gas_df = results_df[results_df["gas"] == gas].copy()
        if gas_df.empty:
            log.warning("  No results for %s", gas)
            continue

        gas_df = gas_df.sort_values("R2_test", ascending=False)
        top = gas_df.iloc[0]

        # Tie-break in favour of a simpler model with a similar R2
        for _, row in gas_df.iterrows():
            if (top["R2_test"] - row["R2_test"]) < 0.05:
                if complexity_order.get(row["model"], 99) < \
                   complexity_order.get(top["model"], 99):
                    top = row
                    break

        best[gas] = {
            "model": top["model"],
            "config": top["config"],
            "R2_test": top["R2_test"],
            "RMSE": top["RMSE"],
            "CEN_level": top["CEN_level"],
            "model_path": top["model_path"],
        }
        log.info("  Best for %s: %s Config%s (R2=%.3f, CEN=%s)",
                 gas, top["model"], top["config"],
                 top["R2_test"], top["CEN_level"])

    return best


# -------------------------------------------------------------------------
# Monthly drift diagnostic (fixed model, no retraining)
# -------------------------------------------------------------------------
def compute_monthly_drift(df, all_results, best_per_gas, train_mask):
    """
    Evaluate the selected model of each pollutant month by month, without retraining.
    Return one row per pollutant, month and source (ML or refitted OLS) with R2, RMSE and n_obs.
    """
    log.info("Computing the monthly drift...")
    ts = pd.to_datetime(df[COL_TS])
    df = df.copy()
    df["_month"] = ts.dt.to_period("M")
    months = sorted(df["_month"].unique())

    rows = []

    for gas in POLLUTANTS_ML:
        if gas not in best_per_gas:
            continue

        ref_col = ML_REF_COLS[gas]
        best_info = best_per_gas[gas]

        # Detailed result of the selected model
        best_result = None
        for r in all_results:
            if (r["gas"] == gas and r["model"] == best_info["model"]
                    and r["config"] == best_info["config"]):
                best_result = r
                break

        if best_result is None:
            continue

        trained_model = best_result["trained_model"]
        feature_cols = best_result["feature_cols"]

        # Predict once over the full dataset, then split by month
        required = feature_cols + [ref_col]
        all_valid = df[required].notna().all(axis=1)
        X_all = df.loc[all_valid, feature_cols].values
        y_pred_all = trained_model.predict(X_all)
        y_true_all = df.loc[all_valid, ref_col].values
        months_all = df.loc[all_valid, "_month"].values

        # The LSTM returns NaN where no complete window exists
        pred_valid = np.isfinite(y_pred_all)

        for month in months:
            month_mask = (months_all == month) & pred_valid
            n_obs = month_mask.sum()

            if n_obs < ML_MIN_OBS_DRIFT:
                continue

            y_true = y_true_all[month_mask]
            y_pred = y_pred_all[month_mask]

            metrics = compute_ml_metrics(y_true, y_pred)
            rows.append({
                "gas": gas,
                "month": str(month),
                "source": "ML",
                "model": best_info["model"],
                "config": best_info["config"],
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "n_obs": n_obs,
            })

        # Monthly metrics of the refitted OLS, for comparison
        ols_col = f"c_hat_ols_{gas}"
        if ols_col in df.columns:
            for month in months:
                month_mask = df["_month"] == month
                required_ols = [ols_col, ref_col]
                valid = df[required_ols].notna().all(axis=1) & month_mask
                n_obs = valid.sum()

                if n_obs < ML_MIN_OBS_DRIFT:
                    continue

                y_true = df.loc[valid, ref_col].values
                y_pred_ols = df.loc[valid, ols_col].values

                metrics = compute_ml_metrics(y_true, y_pred_ols)
                rows.append({
                    "gas": gas,
                    "month": str(month),
                    "source": "OLS",
                    "model": "OLS",
                    "config": "-",
                    "R2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                    "n_obs": n_obs,
                })

    drift_df = pd.DataFrame(rows)

    # Degradation rate (slope of R2 against the month index)
    if not drift_df.empty:
        _add_degradation_rate(drift_df)

    df.drop(columns=["_month"], inplace=True, errors="ignore")

    return drift_df


def _add_degradation_rate(drift_df):
    """
    Fit the slope of the monthly R2 against the month index, per pollutant and source.
    The slope is classified with ML_DRIFT_RAPID and ML_DRIFT_MODERATE and the table is logged.
    """
    rates = []
    for (gas, source), group in drift_df.groupby(["gas", "source"]):
        # A slope needs at least 3 months
        if len(group) < 3:
            rates.append({"gas": gas, "source": source,
                          "slope": np.nan, "classification": "insufficient"})
            continue

        x = np.arange(len(group))
        y = group["R2"].values
        valid = np.isfinite(y)
        if valid.sum() < 3:
            rates.append({"gas": gas, "source": source,
                          "slope": np.nan, "classification": "insufficient"})
            continue

        slope = np.polyfit(x[valid], y[valid], 1)[0]

        if slope < ML_DRIFT_RAPID:
            classification = "rapid"
        elif slope < ML_DRIFT_MODERATE:
            classification = "moderate"
        else:
            classification = "stable"

        rates.append({"gas": gas, "source": source,
                      "slope": slope, "classification": classification})

    rates_df = pd.DataFrame(rates)
    log.info("Degradation rates:\n%s", rates_df.to_string(index=False))

    return rates_df
