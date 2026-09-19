"""
Plots of the machine learning calibration (s9)

Draws, for the best model of each pollutant, the scatter, time series,
residual, feature importance and SHAP plots; the test R2 by
configuration and by model; and the monthly drift of ML and OLS.
Saves one PNG per plot in the folder given by the caller.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (
    COL_TS, POLLUTANTS_ML, ML_REF_COLS, ML_MODELS, ML_CONFIGS,
)
from .evaluator import compute_shap_values, compute_feature_importance

log = logging.getLogger("s9.plots")

# -------------------------------------------------------------------------
# Global style and units
# -------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
})

UNITS = {"NO2": "ug/m3", "NO2_alpha": "ug/m3", "CO": "ug/m3", "O3": "ug/m3", "PM2.5": "ug/m3"}


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def generate_all_plots(df, results_df, all_results, best_per_gas, drift_df,
                       plot_dir):
    """Draw every plot of the machine learning calibration and save it in plot_dir."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    log.info("Drawing the plots in %s ...", plot_dir)

    # Best model of each gas: scatter, time series, residuals, importances and SHAP
    _plot_scatter_best(all_results, best_per_gas, plot_dir)
    _plot_timeseries_48h(df, all_results, best_per_gas, plot_dir)
    _plot_residuals(all_results, best_per_gas, plot_dir)
    _plot_residual_hist(all_results, best_per_gas, plot_dir)
    _plot_feature_importance(all_results, best_per_gas, plot_dir)
    _plot_shap_summary(all_results, best_per_gas, plot_dir)

    # Test R2 by configuration, by model and as a model x config heat map
    _plot_r2_by_config(results_df, plot_dir)
    _plot_r2_by_model(results_df, plot_dir)
    _plot_r2_heatmap(results_df, plot_dir)

    # Monthly drift
    if drift_df is not None and not drift_df.empty:
        _plot_drift(drift_df, plot_dir)

    log.info("Plots completed.")


def _get_best_result(all_results, best_per_gas, gas):
    """Return the detailed result of the best model of a gas, or None."""
    if gas not in best_per_gas:
        return None
    info = best_per_gas[gas]
    for r in all_results:
        if r["gas"] == gas and r["model"] == info["model"] and r["config"] == info["config"]:
            return r
    return None


# -------------------------------------------------------------------------
# Plots of the best model of each gas
# -------------------------------------------------------------------------
def _plot_scatter_best(all_results, best_per_gas, plot_dir):
    """Scatter of observed against predicted values on the test set."""
    for gas in POLLUTANTS_ML:
        r = _get_best_result(all_results, best_per_gas, gas)
        if r is None:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        # NaN left out (the LSTM returns NaN where no complete window exists)
        yt, yp = r["y_test"], r["y_pred_test"]
        valid = np.isfinite(yt) & np.isfinite(yp)
        ax.scatter(yt[valid], yp[valid], alpha=0.3, s=8, c="steelblue")

        lo = min(np.nanmin(yt), np.nanmin(yp))
        hi = max(np.nanmax(yt), np.nanmax(yp))
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", lw=1, label="1:1")
        ax.set_xlabel(f"Reference ({UNITS[gas]})")
        ax.set_ylabel(f"Prediction ({UNITS[gas]})")
        ax.set_title(f"{gas}: {r['model']} Config{r['config']}\n"
                     f"R2={r['R2_test']:.3f}, RMSE={r['RMSE']:.2f}")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")

        fig.savefig(plot_dir / f"scatter_{gas.replace('.', '')}.png")
        plt.close(fig)


def _plot_timeseries_48h(df, all_results, best_per_gas, plot_dir):
    """Time series of the first 48 valid periods of the test set, for each gas."""
    ts = pd.to_datetime(df[COL_TS])

    for gas in POLLUTANTS_ML:
        r = _get_best_result(all_results, best_per_gas, gas)
        if r is None:
            continue

        ref_col = ML_REF_COLS[gas]
        feature_cols = r["feature_cols"]
        model = r["trained_model"]

        # Valid rows of the test period
        required = feature_cols + [ref_col]
        valid = df[required].notna().all(axis=1)
        from config import ML_TEST_START
        test_start = pd.Timestamp(ML_TEST_START, tz=ts.dt.tz)
        test_valid = valid & (ts >= test_start)
        idx = df.index[test_valid]
        if len(idx) < 48:
            continue

        # First 48 valid rows
        block = idx[:48]
        ts_block = ts.iloc[block]
        y_ref = df.loc[block, ref_col].values
        X_block = df.loc[block, feature_cols].values
        y_pred = model.predict(X_block)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ts_block, y_ref, "k-", lw=1.2, label="Reference")
        ax.plot(ts_block, y_pred, "steelblue", lw=1, alpha=0.8,
                label=f"{r['model']} Config{r['config']}")
        ax.set_ylabel(f"{gas} ({UNITS[gas]})")
        ax.set_title(f"{gas}: time series (test)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()

        fig.savefig(plot_dir / f"timeseries_48h_{gas.replace('.', '')}.png")
        plt.close(fig)


def _plot_residuals(all_results, best_per_gas, plot_dir):
    """Residuals against the prediction, for each gas."""
    for gas in POLLUTANTS_ML:
        r = _get_best_result(all_results, best_per_gas, gas)
        if r is None:
            continue

        residuals = r["y_pred_test"] - r["y_test"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(r["y_pred_test"], residuals, alpha=0.3, s=8, c="steelblue")
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_xlabel(f"Prediction ({UNITS[gas]})")
        ax.set_ylabel(f"Residual ({UNITS[gas]})")
        ax.set_title(f"{gas}: residuals vs prediction ({r['model']} Config{r['config']})")

        fig.savefig(plot_dir / f"residuals_{gas.replace('.', '')}.png")
        plt.close(fig)


def _plot_residual_hist(all_results, best_per_gas, plot_dir):
    for gas in POLLUTANTS_ML:
        r = _get_best_result(all_results, best_per_gas, gas)
        if r is None:
            continue

        residuals = r["y_pred_test"] - r["y_test"]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(residuals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", ls="--", lw=1)
        ax.axvline(np.mean(residuals), color="orange", ls="--", lw=1,
                   label=f"MBE={np.mean(residuals):.2f}")
        ax.set_xlabel(f"Residual ({UNITS[gas]})")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{gas}: residual histogram ({r['model']} Config{r['config']})")
        ax.legend()

        fig.savefig(plot_dir / f"hist_residuals_{gas.replace('.', '')}.png")
        plt.close(fig)


def _plot_feature_importance(all_results, best_per_gas, plot_dir):
    """Bar plot of the 15 most important features of the best model of each gas."""
    for gas in POLLUTANTS_ML:
        r = _get_best_result(all_results, best_per_gas, gas)
        if r is None:
            continue

        X_test = r.get("X_test")
        if X_test is None or len(X_test) == 0:
            continue

        imp_df = compute_feature_importance(
            r["trained_model"], r["feature_cols"],
            X_test=X_test, y_test=r["y_test"],
        )
        if imp_df["importance"].isna().all():
            continue

        imp_df = imp_df.head(15)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp_df["feature"], imp_df["importance"], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"{gas}: feature importance ({r['model']} Config{r['config']})")
        ax.invert_yaxis()

        fig.savefig(plot_dir / f"feat_importance_{gas.replace('.', '')}.png")
        plt.close(fig)


def _plot_shap_summary(all_results, best_per_gas, plot_dir):
    """SHAP summary plot of the best model of each gas (skipped when shap is missing)."""
    try:
        import shap
    except ImportError:
        log.warning("shap is not installed, skipping the SHAP plots")
        return

    for gas in POLLUTANTS_ML:
        r = _get_best_result(all_results, best_per_gas, gas)
        if r is None:
            continue

        # Subsample, for speed
        n_sample = min(200, len(r["y_test"]))

        X_test = r.get("X_test")
        if X_test is None or len(X_test) == 0:
            continue

        X_sample = X_test[:n_sample]
        shap_values = compute_shap_values(
            r["trained_model"],
            X_sample,
            r["feature_cols"],
        )

        if shap_values is None:
            continue

        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, show=False)
            plt.title(f"{gas}: SHAP summary ({r['model']} Config{r['config']})")
            plt.savefig(plot_dir / f"shap_{gas.replace('.', '')}.png")
            plt.close("all")
        except Exception as e:
            log.warning("SHAP plot failed for %s: %s", gas, e)
            plt.close("all")


# -------------------------------------------------------------------------
# Comparison of configurations and models
# -------------------------------------------------------------------------
def _plot_r2_by_config(results_df, plot_dir):
    """Bar plot of the best test R2 of each feature configuration, per gas."""
    for gas in POLLUTANTS_ML:
        gas_df = results_df[results_df["gas"] == gas]
        if gas_df.empty:
            continue

        # Best model of each configuration
        best_by_config = gas_df.groupby("config")["R2_test"].max().reset_index()

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#2196F3", "#4CAF50", "#FF9800"]
        bars = ax.bar(best_by_config["config"], best_by_config["R2_test"],
                      color=colors[:len(best_by_config)])
        ax.set_ylabel("R2 (test)")
        ax.set_title(f"{gas}: best R2 by feature configuration")
        ax.set_ylim(bottom=min(0, best_by_config["R2_test"].min() - 0.1))

        for bar, val in zip(bars, best_by_config["R2_test"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        fig.savefig(plot_dir / f"r2_by_config_{gas.replace('.', '')}.png")
        plt.close(fig)


def _plot_r2_by_model(results_df, plot_dir):
    """Bar plot of the best test R2 of each model, per gas."""
    for gas in POLLUTANTS_ML:
        gas_df = results_df[results_df["gas"] == gas]
        if gas_df.empty:
            continue

        best_by_model = gas_df.groupby("model")["R2_test"].max().reset_index()
        order = ["RF", "XGBoost", "LightGBM", "DNN", "LSTM"]
        best_by_model["sort_key"] = best_by_model["model"].map(
            {m: i for i, m in enumerate(order)}
        ).fillna(99)
        best_by_model = best_by_model.sort_values("sort_key").drop(columns="sort_key")

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#4CAF50", "#FF9800", "#2196F3", "#FF5722", "#9C27B0"]
        bars = ax.bar(best_by_model["model"], best_by_model["R2_test"],
                      color=colors[:len(best_by_model)])
        ax.set_ylabel("R2 (test)")
        ax.set_title(f"{gas}: best R2 by model")
        ax.set_ylim(bottom=min(0, best_by_model["R2_test"].min() - 0.1))

        for bar, val in zip(bars, best_by_model["R2_test"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        fig.savefig(plot_dir / f"r2_by_model_{gas.replace('.', '')}.png")
        plt.close(fig)


def _plot_r2_heatmap(results_df, plot_dir):
    """Heat map of the test R2 (model x config), per gas."""
    for gas in POLLUTANTS_ML:
        gas_df = results_df[results_df["gas"] == gas]
        if gas_df.empty:
            continue

        pivot = gas_df.pivot_table(
            values="R2_test", index="model", columns="config"
        )
        # Fixed order of rows and columns
        model_order = [m for m in ["RF", "XGBoost", "LightGBM", "DNN", "LSTM"] if m in pivot.index]
        config_order = [c for c in ["A", "B", "C"] if c in pivot.columns]
        pivot = pivot.reindex(index=model_order, columns=config_order)

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=min(0, pivot.values.min()), vmax=1)

        ax.set_xticks(range(len(config_order)))
        ax.set_xticklabels([f"Config {c}" for c in config_order])
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order)

        for i in range(len(model_order)):
            for j in range(len(config_order)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=11, fontweight="bold")

        plt.colorbar(im, ax=ax, label="R2 (test)")
        ax.set_title(f"{gas}: test R2, model x configuration")

        fig.savefig(plot_dir / f"heatmap_r2_{gas.replace('.', '')}.png")
        plt.close(fig)


# -------------------------------------------------------------------------
# Monthly drift
# -------------------------------------------------------------------------
def _plot_drift(drift_df, plot_dir):
    """Monthly R2 of the selected ML model and of the refitted OLS, per gas and consolidated."""
    for gas in POLLUTANTS_ML:
        gas_drift = drift_df[drift_df["gas"] == gas]
        if gas_drift.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        for source in ["ML", "OLS"]:
            src_df = gas_drift[gas_drift["source"] == source].sort_values("month")
            if src_df.empty:
                continue
            style = "-o" if source == "ML" else "--s"
            color = "steelblue" if source == "ML" else "gray"
            label = f"{source}"
            if source == "ML" and not src_df.empty:
                label += f" ({src_df.iloc[0]['model']} Config{src_df.iloc[0]['config']})"
            ax.plot(src_df["month"], src_df["R2"], style, color=color,
                    label=label, markersize=5)

        ax.set_ylabel("R2")
        ax.set_title(f"{gas}: monthly R2 drift")
        ax.legend()
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        plt.xticks(rotation=45)

        fig.savefig(plot_dir / f"drift_{gas.replace('.', '')}.png")
        plt.close(fig)

    # Consolidated figure (2 x 2 panels)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, gas in zip(axes.flat, POLLUTANTS_ML):
        gas_drift = drift_df[drift_df["gas"] == gas]
        if gas_drift.empty:
            ax.set_title(f"{gas}: no drift data")
            continue

        for source in ["ML", "OLS"]:
            src_df = gas_drift[gas_drift["source"] == source].sort_values("month")
            if src_df.empty:
                continue
            style = "-o" if source == "ML" else "--s"
            color = "steelblue" if source == "ML" else "gray"
            ax.plot(src_df["month"], src_df["R2"], style, color=color,
                    label=source, markersize=4)

        ax.set_title(gas)
        ax.set_ylabel("R2")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Monthly R2 drift: ML vs OLS", fontsize=13)
    plt.tight_layout()
    fig.savefig(plot_dir / "drift_consolidado.png")
    plt.close(fig)
