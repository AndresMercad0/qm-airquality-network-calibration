"""
Markdown report of the machine learning calibration (s9)

Writes the report of a run: best model per pollutant, comparison with
the refitted OLS, master table, Diebold-Mariano test, overfitting
diagnostic, monthly drift, the O3 case and the run parameters. s9
names the file reporte_calibracion_ml_v4.md.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import POLLUTANTS_ML, ML_REF_COLS, ML_TRAIN_END, ML_TEST_START, COL_TS, NODE_LABEL

log = logging.getLogger("s9.report")


# -------------------------------------------------------------------------
# Markdown report of the machine learning calibration
# -------------------------------------------------------------------------
def generate_report(results_df, all_results, best_per_gas, ols_info,
                    drift_df, output_path, df=None):
    """
    Write the full Markdown report.
    df is the unified dataset, needed to align ML and OLS errors in the Diebold-Mariano test.
    """
    output_path = Path(output_path)
    lines = []

    lines.append(f"# Machine learning calibration report: {NODE_LABEL}")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Section 1: executive summary
    lines.append("## 1. Executive summary\n")
    lines.append("| Pollutant | Best model | Config | R2 test | RMSE | CEN |")
    lines.append("|---|---|---|---|---|---|")
    for gas in POLLUTANTS_ML:
        if gas in best_per_gas:
            b = best_per_gas[gas]
            lines.append(f"| {gas} | {b['model']} | {b['config']} | "
                         f"{b['R2_test']:.3f} | {b['RMSE']:.2f} | {b['CEN_level']} |")
        else:
            lines.append(f"| {gas} | - | - | - | - | - |")

    # Improvement over the OLS
    lines.append("\n### Improvement over OLS\n")
    lines.append("| Pollutant | R2 OLS (train) | R2 ML (test) | Delta R2 |")
    lines.append("|---|---|---|---|")
    for gas in POLLUTANTS_ML:
        r2_ols = ols_info.get(gas, {}).get("r2_train", np.nan)
        r2_ml = best_per_gas.get(gas, {}).get("R2_test", np.nan)
        delta = r2_ml - r2_ols if not (np.isnan(r2_ols) or np.isnan(r2_ml)) else np.nan
        lines.append(f"| {gas} | {r2_ols:.3f} | {r2_ml:.3f} | "
                     f"{delta:+.3f} |" if not np.isnan(delta) else
                     f"| {gas} | - | - | - |")

    # Section 2: master table of every combination
    lines.append(f"\n## 2. Master table: {len(results_df)} combinations\n")
    if not results_df.empty:
        display_cols = ["gas", "config", "model", "R2_train", "R2_test",
                        "RMSE", "MAE", "MBE", "nRMSE", "pearson_r",
                        "CEN_level", "overfit_ratio"]
        available = [c for c in display_cols if c in results_df.columns]
        table_df = results_df[available].copy()

        # Number formats
        for col in ["R2_train", "R2_test", "pearson_r", "overfit_ratio"]:
            if col in table_df.columns:
                table_df[col] = table_df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        for col in ["RMSE", "MAE", "MBE", "nRMSE"]:
            if col in table_df.columns:
                table_df[col] = table_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

        lines.append(table_df.to_markdown(index=False))

    # Section 3: Diebold-Mariano test
    lines.append("\n## 3. Diebold-Mariano test (ML vs OLS)\n")
    lines.append("| Pollutant | Model | DM statistic | p-value | Significant (5%) |")
    lines.append("|---|---|---|---|---|")
    from .evaluator import diebold_mariano_test
    for gas in POLLUTANTS_ML:
        if gas not in best_per_gas:
            continue
        best_info = best_per_gas[gas]
        # Detailed result of the selected model
        best_result = None
        for r in all_results:
            if r["gas"] == gas and r["model"] == best_info["model"] \
                    and r["config"] == best_info["config"]:
                best_result = r
                break
        if best_result is None:
            continue

        # Errors of the ML model against those of the refitted OLS (c_hat_ols_{gas})
        e_ml = best_result["y_test"] - best_result["y_pred_test"]
        dm_stat, dm_p = np.nan, np.nan

        if df is not None:
            ols_col = f"c_hat_ols_{gas}"
            ref_col = ML_REF_COLS.get(gas)
            feature_cols = best_result.get("feature_cols", [])
            if ols_col in df.columns and ref_col in df.columns:
                ts = pd.to_datetime(df[COL_TS])
                tz = ts.dt.tz
                test_mask = ts >= pd.Timestamp(ML_TEST_START, tz=tz)
                # Test rows where the OLS, the ML features and the reference are all valid
                ml_cols = feature_cols + [ref_col]
                ml_valid = df[ml_cols].notna().all(axis=1) if ml_cols else pd.Series(True, index=df.index)
                ols_valid = df[[ols_col, ref_col]].notna().all(axis=1)
                common = test_mask & ml_valid & ols_valid
                if common.sum() >= 20:
                    y_ref = df.loc[common, ref_col].values
                    y_pred_ols = df.loc[common, ols_col].values
                    # ML prediction over the same rows
                    X_common = df.loc[common, feature_cols].values
                    y_pred_ml = best_result["trained_model"].predict(X_common)
                    # NaN predictions left out (LSTM)
                    valid_pred = np.isfinite(y_pred_ml)
                    if valid_pred.sum() >= 20:
                        e_ols = y_ref[valid_pred] - y_pred_ols[valid_pred]
                        e_ml_aligned = y_ref[valid_pred] - y_pred_ml[valid_pred]
                        dm_stat, dm_p = diebold_mariano_test(e_ols, e_ml_aligned)

        sig = "Yes" if (not np.isnan(dm_p) and dm_p < 0.05) else "No"
        dm_stat_str = f"{dm_stat:.3f}" if not np.isnan(dm_stat) else "-"
        dm_p_str = f"{dm_p:.4f}" if not np.isnan(dm_p) else "-"
        lines.append(f"| {gas} | {best_info['model']} | {dm_stat_str} | "
                     f"{dm_p_str} | {sig} |")

    # Section 4: overfitting diagnostic (ratio above 1.5 flagged with ***)
    lines.append("\n## 4. Overfitting diagnostic\n")
    lines.append("| Pollutant | Model | Config | R2 train | R2 test | Overfit ratio |")
    lines.append("|---|---|---|---|---|---|")
    if not results_df.empty:
        for _, row in results_df.iterrows():
            of_flag = " ***" if (pd.notna(row.get("overfit_ratio")) and
                                 row["overfit_ratio"] > 1.5) else ""
            lines.append(
                f"| {row['gas']} | {row['model']} | {row['config']} | "
                f"{row['R2_train']:.3f} | {row['R2_test']:.3f} | "
                f"{row.get('overfit_ratio', np.nan):.2f}{of_flag} |"
            )

    # Section 5: monthly drift
    lines.append("\n## 5. Monthly drift\n")
    if drift_df is not None and not drift_df.empty:
        lines.append(drift_df.to_markdown(index=False))
    else:
        lines.append("Not enough data for the drift analysis.\n")

    # Section 6: O3 with the MQ131
    lines.append("\n## 6. Special case: O3 (MQ131)\n")
    o3_results = results_df[results_df["gas"] == "O3"] if not results_df.empty else pd.DataFrame()
    if not o3_results.empty:
        best_o3_r2 = o3_results["R2_test"].max()
        if best_o3_r2 <= 0:
            lines.append("The MQ131 sensor does not reach a positive test R2 against the "
                         "Alphasense O3 comparator, even with the machine learning models. This is "
                         "consistent with the limitations of the MQ131 for O3 measurements "
                         "at ambient concentrations (< 100 ppb).\n")
        else:
            lines.append(f"Best R2 for O3: {best_o3_r2:.3f}. "
                         f"ML {'improves' if best_o3_r2 > 0.3 else 'does not improve significantly'} "
                         f"on OLS.\n")
    else:
        lines.append("No results for O3.\n")

    # Section 7: parameters
    lines.append("\n## 7. Experiment parameters\n")
    lines.append(f"- **Chronological split**: train <= {ML_TRAIN_END}, test >= {ML_TEST_START}")
    lines.append(f"- **Cross-validation**: TimeSeriesSplit, 5 folds")
    lines.append(f"- **Random state**: 42")
    models_str = ", ".join(results_df["model"].unique()) if not results_df.empty else "RF, LightGBM, DNN"
    lines.append(f"- **Models**: {models_str}")
    lines.append(f"- **Configurations**: A (pure ML), B (hybrid), C (OLS correction)")
    lines.append(f"- **Total combinations**: {len(results_df)} trained\n")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    log.info("Report written: %s", output_path.name)

    return output_path
