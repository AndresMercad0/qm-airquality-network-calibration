"""
Step s9: calibration by machine learning

Builds the 15 min dataset (s1, s3, s4, reference station), splits it
chronologically and trains every pollutant, configuration and model.
Writes the CALIBRATED_ML CSV of the best models, the model files,
tabla_maestra_resultados.csv, metricas_drift_mensual.csv, plots, report.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    NODE_ID, COL_TS,
    DATA_S1, DATA_S3, DATA_S4,
    DATA_S9, RESULTS_S9, RESULTS_S9_MODELS, RESULTS_S9_PLOTS,
    REF_DATA_DIR, REF_CSV_PATTERN, REF_STATION_SHORT,
    POLLUTANTS_ML, ML_REF_COLS, ML_COL_PREFIXES,
    ML_MODELS_SUPPLEMENTARY,
)
from utils import detect_csv, extract_dates_from_csv, build_output_name, build_ml_col_name

# -------------------------------------------------------------------------
# Logging configuration (console and file)
# -------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)
log = logging.getLogger("s9")


def _setup_file_logging():
    """Add a file handler so that every log record also goes to s9_ml.log."""
    log_path = RESULTS_S9 / "s9_ml.log"
    RESULTS_S9.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logging.getLogger().addHandler(fh)
    return log_path


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------
def run_s9():
    """
    Run the machine learning calibration of the node.
    Train every pollutant, configuration and model, select the best model per
    pollutant, diagnose the monthly drift, and write the plots and the report.
    """
    log_path = _setup_file_logging()
    t_start = time.time()
    log.info("=" * 70)
    log.info("  s9: MACHINE LEARNING CALIBRATION, %s", NODE_ID)
    log.info("=" * 70)
    log.info("  Log file: %s", log_path)

    # Input files
    s1_csv = detect_csv(DATA_S1, f"{NODE_ID}_QC_RAW_wide_data_*.csv")
    s3_csv = detect_csv(DATA_S3, f"{NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    s4_csv = detect_csv(DATA_S4, f"{NODE_ID}_PREPROCESS_wide_data_*.csv")
    ref_csv = detect_csv(REF_DATA_DIR, REF_CSV_PATTERN)

    for name, csv in [("s1", s1_csv), ("s3", s3_csv), ("s4", s4_csv), (REF_STATION_SHORT, ref_csv)]:
        if csv is None:
            raise FileNotFoundError(f"CSV of {name} not found")
        log.info("Input %s: %s", name, csv.name)

    # Unified 15 min dataset
    from ml.data_loader import prepare_dataset, get_train_test_masks
    df = prepare_dataset(s1_csv, s3_csv, s4_csv, ref_csv)

    # Chronological split
    train_mask, test_mask = get_train_test_masks(df)

    # OLS refitted on the training period only, so that no test data leak into configs B and C
    from ml.ols_refit import refit_ols_on_train
    ols_info = refit_ols_on_train(df, train_mask)

    # Main models
    RESULTS_S9_MODELS.mkdir(parents=True, exist_ok=True)

    from ml.trainer import train_all_combinations
    results_df, all_results = train_all_combinations(
        df, train_mask, test_mask, RESULTS_S9_MODELS
    )

    # Supplementary models (XGBoost, LSTM)
    if ML_MODELS_SUPPLEMENTARY:
        log.info("Training the supplementary models: %s", ML_MODELS_SUPPLEMENTARY)
        sup_results_df, sup_all_results = train_all_combinations(
            df, train_mask, test_mask, RESULTS_S9_MODELS,
            models=ML_MODELS_SUPPLEMENTARY,
        )
        results_df = pd.concat([results_df, sup_results_df], ignore_index=True)
        all_results.extend(sup_all_results)

    # Best model per pollutant
    from ml.drift import select_best_model_per_gas, compute_monthly_drift
    best_per_gas = select_best_model_per_gas(results_df)

    # Monthly drift of the selected models
    drift_df = compute_monthly_drift(df, all_results, best_per_gas, train_mask)

    # Predictions of the best model over the full dataset
    df_out = df[[COL_TS]].copy()

    for gas in POLLUTANTS_ML:
        if gas not in best_per_gas:
            continue

        best_info = best_per_gas[gas]
        best_result = None
        for r in all_results:
            if r["gas"] == gas and r["model"] == best_info["model"] \
                    and r["config"] == best_info["config"]:
                best_result = r
                break

        if best_result is None:
            continue

        model = best_result["trained_model"]
        feature_cols = best_result["feature_cols"]
        col_name = build_ml_col_name(gas, best_info["model"], best_info["config"])

        # Rows where every feature is available
        valid = df[feature_cols].notna().all(axis=1)
        X = df.loc[valid, feature_cols].values
        predictions = model.predict(X)

        df_out[col_name] = np.nan
        df_out.loc[valid, col_name] = predictions

    # Refitted OLS columns, kept for comparison
    for gas in POLLUTANTS_ML:
        ols_col = f"c_hat_ols_{gas}"
        if ols_col in df.columns:
            df_out[ols_col] = df[ols_col]

    # Exports: predictions, master table of results and monthly drift
    DATA_S9.mkdir(parents=True, exist_ok=True)
    RESULTS_S9.mkdir(parents=True, exist_ok=True)

    start_date, end_date = extract_dates_from_csv(df)
    out_name = build_output_name(NODE_ID, "CALIBRATED_ML", start_date, end_date)
    out_csv = DATA_S9 / out_name
    df_out.to_csv(out_csv, index=False)
    log.info("Predictions CSV: %s", out_csv.name)

    tabla_path = RESULTS_S9 / "tabla_maestra_resultados.csv"
    results_df.to_csv(tabla_path, index=False)
    log.info("Master table: %s", tabla_path.name)

    if drift_df is not None and not drift_df.empty:
        drift_path = RESULTS_S9 / "metricas_drift_mensual.csv"
        drift_df.to_csv(drift_path, index=False)
        log.info("Monthly drift: %s", drift_path.name)

    # Plots and Markdown report
    from ml.plots import generate_all_plots
    generate_all_plots(df, results_df, all_results, best_per_gas,
                       drift_df, RESULTS_S9_PLOTS)

    from ml.report import generate_report
    report_path = RESULTS_S9 / "reporte_calibracion_ml_v4.md"
    generate_report(results_df, all_results, best_per_gas, ols_info,
                    drift_df, report_path, df=df)

    # Final summary
    elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 70)
    log.info("  s9 COMPLETED: %.0fs (%.1f min)", elapsed, elapsed / 60)
    log.info("=" * 70)
    log.info("  CSV:     %s", out_csv.name)
    log.info("  Models:  %d files", len(list(RESULTS_S9_MODELS.glob('*'))))
    log.info("  Plots:   %d figures", len(list(RESULTS_S9_PLOTS.glob('*.png'))))
    log.info("  Report:  %s", report_path.name)
    log.info("")

    log.info("  BEST MODEL PER GAS:")
    for gas in POLLUTANTS_ML:
        if gas in best_per_gas:
            b = best_per_gas[gas]
            log.info("    %6s: %8s Config%s  R2=%.3f  RMSE=%.2f  CEN=%s",
                     gas, b['model'], b['config'], b['R2_test'], b['RMSE'], b['CEN_level'])
        else:
            log.info("    %6s: no results", gas)
    log.info("=" * 70)

    return out_csv


if __name__ == "__main__":
    run_s9()
