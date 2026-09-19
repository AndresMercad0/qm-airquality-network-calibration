"""
Machine learning calibration of the Alphasense NO2-B43F only

Standalone run of the s9 training for the gas key NO2_alpha (three
configurations, five models) against the reference station NO2.
Writes tabla_maestra_no2alpha.csv and the no2_alpha_* model files; the
master table, predictions, plots and report of s9 are left untouched.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    NODE_ID,
    DATA_S1, DATA_S3, DATA_S4,
    RESULTS_S9, RESULTS_S9_MODELS,
    REF_DATA_DIR, REF_CSV_PATTERN,
    ML_MODELS, ML_MODELS_SUPPLEMENTARY,
)
from utils import detect_csv

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
log = logging.getLogger("s9_no2alpha")


def _setup_file_logging():
    log_path = RESULTS_S9 / "s9_no2alpha.log"
    RESULTS_S9.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logging.getLogger().addHandler(fh)
    return log_path


# -------------------------------------------------------------------------
# Training of the Alphasense NO2-B43F models only (gas key NO2_alpha)
# -------------------------------------------------------------------------
def run():
    """
    Train only the mid-cost Alphasense NO2-B43F against the LAQN reference.
    The results go to tabla_maestra_no2alpha.csv and the models to no2_alpha_* files.
    The master table, predictions, drift, plots and report of s9 are not rewritten.
    """
    log_path = _setup_file_logging()
    t_start = time.time()
    log.info("=" * 70)
    log.info("  s9 NO2_alpha only: %s", NODE_ID)
    log.info("=" * 70)
    log.info("  Log file: %s", log_path)

    # Input files
    s1_csv = detect_csv(DATA_S1, f"{NODE_ID}_QC_RAW_wide_data_*.csv")
    s3_csv = detect_csv(DATA_S3, f"{NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    s4_csv = detect_csv(DATA_S4, f"{NODE_ID}_PREPROCESS_wide_data_*.csv")
    ref_csv = detect_csv(REF_DATA_DIR, REF_CSV_PATTERN)
    for name, csv in [("s1", s1_csv), ("s3", s3_csv), ("s4", s4_csv), ("REF", ref_csv)]:
        if csv is None:
            raise FileNotFoundError(f"CSV of {name} not found")
        log.info("Input %s: %s", name, csv.name)

    # Unified 15 min dataset and chronological split
    from ml.data_loader import prepare_dataset, get_train_test_masks
    df = prepare_dataset(s1_csv, s3_csv, s4_csv, ref_csv)

    train_mask, test_mask = get_train_test_masks(df)

    # OLS refitted on the training period only (NO2_alpha is part of POLLUTANTS_ML)
    from ml.ols_refit import refit_ols_on_train
    ols_info = refit_ols_on_train(df, train_mask)
    if "NO2_alpha" in ols_info:
        log.info("OLS NO2_alpha (anti-leakage train-only): R2_train=%.3f",
                 ols_info["NO2_alpha"]["r2_train"])
    else:
        log.warning("OLS for NO2_alpha was not computed. Check COL_ALPHA_NO2 in s3.")

    # NO2_alpha only: 3 configurations x 5 models
    RESULTS_S9_MODELS.mkdir(parents=True, exist_ok=True)
    from ml.trainer import train_all_combinations

    log.info("Training the main models (RF/LightGBM/DNN) for NO2_alpha...")
    main_df, main_results = train_all_combinations(
        df, train_mask, test_mask, RESULTS_S9_MODELS,
        models=ML_MODELS, gases=["NO2_alpha"],
    )

    log.info("Training the supplementary models (XGBoost/LSTM) for NO2_alpha...")
    sup_df, sup_results = train_all_combinations(
        df, train_mask, test_mask, RESULTS_S9_MODELS,
        models=ML_MODELS_SUPPLEMENTARY, gases=["NO2_alpha"],
    )

    # Separate results table
    combined = pd.concat([main_df, sup_df], ignore_index=True)
    out_csv = RESULTS_S9 / "tabla_maestra_no2alpha.csv"
    combined.to_csv(out_csv, index=False)
    log.info("NO2_alpha table: %s (%d rows)", out_csv.name, len(combined))

    # Summary: the five best combinations
    log.info("")
    log.info("=" * 70)
    log.info("  NO2_alpha SUMMARY (%s)", NODE_ID)
    log.info("=" * 70)
    if not combined.empty:
        sorted_df = combined.sort_values("R2_test", ascending=False).head(5)
        for _, row in sorted_df.iterrows():
            log.info("  %s-%s: R2_test=%.3f  RMSE=%.2f  overfit=%.2f  CEN=%s",
                     row["model"], row["config"],
                     row["R2_test"], row["RMSE"], row["overfit_ratio"], row["CEN_level"])
    else:
        log.warning("No model was trained successfully.")

    elapsed = time.time() - t_start
    log.info("Total time: %.1f min", elapsed / 60)


if __name__ == "__main__":
    run()
