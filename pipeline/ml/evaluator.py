"""
Evaluation metrics and model diagnostics (s9)

Computes R2, RMSE, MAE, MBE, nRMSE, Pearson r and the quality level of
a prediction, the overfit ratio, the Diebold-Mariano test between two
error series, feature importances, SHAP values and the residuals by
quartile of the reference.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from utils import compute_ml_metrics, nivel_calidad_ml

log = logging.getLogger("s9.evaluator")


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
def evaluate_model(y_true, y_pred):
    """Return R2, RMSE, MAE, MBE, nRMSE, Pearson r and the CEN quality level."""
    metrics = compute_ml_metrics(y_true, y_pred)
    metrics["CEN_level"] = nivel_calidad_ml(metrics["R2"], metrics["nRMSE"])
    return metrics


def compute_overfit_ratio(r2_train, r2_test):
    """Return R2_train / R2_test. Values above 1.5 indicate overfitting."""
    if r2_test <= 0 or np.isnan(r2_test):
        return np.nan
    return r2_train / r2_test


# -------------------------------------------------------------------------
# Diebold-Mariano test
# -------------------------------------------------------------------------
def diebold_mariano_test(e_ols, e_ml, h=1):
    """
    Diebold-Mariano test of equal predictive accuracy (H0: both models are equally accurate).
    e_ols and e_ml are the prediction errors (y_true - y_pred) and h is the horizon.
    Return (statistic, p_value).
    """
    from scipy import stats

    e_ols = np.asarray(e_ols, dtype=float)
    e_ml = np.asarray(e_ml, dtype=float)

    # Loss differential (squared error)
    d = e_ols**2 - e_ml**2
    n = len(d)

    if n < 10:
        return np.nan, np.nan

    d_mean = np.mean(d)

    # Long-run variance (Newey-West with h - 1 lags)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += 2 * gamma_k
    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * stats.t.sf(abs(dm_stat), df=n - 1)

    return float(dm_stat), float(p_value)


# -------------------------------------------------------------------------
# Feature importance and SHAP
# -------------------------------------------------------------------------
def compute_feature_importance(model, feature_names, X_test=None, y_test=None):
    """
    Return a table of feature importances in descending order.
    Tree models use feature_importances_ (impurity). The DNN uses the permutation
    importance, when X_test and y_test are given.
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # Fallback: permutation importance
    if X_test is not None and y_test is not None:
        from sklearn.inspection import permutation_importance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=10, random_state=42, n_jobs=-1,
            )
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": result.importances_mean,
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    return pd.DataFrame({"feature": feature_names, "importance": np.nan})


def compute_shap_values(model, X_sample, feature_names):
    """
    Compute the SHAP values of a subsample (numpy array).
    TreeExplainer for the tree models, KernelExplainer for the DNN.
    Return a shap.Explanation, or None if shap is missing or fails.
    """
    try:
        import shap
    except ImportError:
        log.warning("shap is not installed, skipping the SHAP analysis")
        return None

    try:
        # Tree-based models
        if hasattr(model, "estimators_") or hasattr(model, "booster_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            return shap.Explanation(
                values=shap_values,
                data=X_sample,
                feature_names=feature_names,
            )

        # KernelExplainer for the DNN (slower): 50 background rows, 200 explained rows
        background = X_sample[:min(50, len(X_sample))]
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(
            X_sample[:min(200, len(X_sample))],
            nsamples=100,
        )
        return shap.Explanation(
            values=shap_values,
            data=X_sample[:min(200, len(X_sample))],
            feature_names=feature_names,
        )

    except Exception as e:
        log.warning("SHAP computation failed for %s: %s", type(model).__name__, e)
        return None


def compute_residuals_by_quartile(y_true, y_pred):
    """Summarise the residuals by quartile of the reference concentration."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_pred - y_true
    quartiles = pd.qcut(y_true, 4, labels=["Q1", "Q2", "Q3", "Q4"],
                        duplicates="drop")
    df = pd.DataFrame({"quartile": quartiles, "residual": residuals})
    summary = df.groupby("quartile")["residual"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    return summary
