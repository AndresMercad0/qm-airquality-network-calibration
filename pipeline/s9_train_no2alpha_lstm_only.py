"""
LSTM training of the Alphasense NO2-B43F only

Standalone run of the s9 training restricted to the LSTM and the gas
key NO2_alpha (configurations A, B and C). Writes
tabla_maestra_no2alpha_lstm.csv and overwrites the checkpoints
no2_alpha_{a,b,c}_lstm.pt; the other result tables are left untouched.
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    NODE_ID,
    DATA_S1, DATA_S3, DATA_S4,
    RESULTS_S9, RESULTS_S9_MODELS,
    REF_DATA_DIR, REF_CSV_PATTERN,
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
log = logging.getLogger("s9_no2alpha_lstm")


def _setup_file_logging():
    log_path = RESULTS_S9 / "s9_no2alpha_lstm.log"
    RESULTS_S9.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logging.getLogger().addHandler(fh)
    return log_path


# -------------------------------------------------------------------------
# Training of the three NO2_alpha LSTM models only (configs A, B, C)
# -------------------------------------------------------------------------
def run():
    """
    Retrain only the LSTM models of NO2_alpha, one per configuration.
    The results go to tabla_maestra_no2alpha_lstm.csv and the checkpoints
    no2_alpha_{a,b,c}_lstm.pt are overwritten. The other result tables are not touched.
    """
    log_path = _setup_file_logging()
    t_start = time.time()
    log.info("=" * 70)
    log.info("  s9 NO2_alpha LSTM only: %s", NODE_ID)
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

    # OLS refitted on the training period only
    from ml.ols_refit import refit_ols_on_train
    ols_info = refit_ols_on_train(df, train_mask)
    if "NO2_alpha" in ols_info:
        log.info("OLS NO2_alpha (anti-leakage train-only): R2_train=%.3f",
                 ols_info["NO2_alpha"]["r2_train"])

    RESULTS_S9_MODELS.mkdir(parents=True, exist_ok=True)
    from ml.trainer import train_all_combinations

    log.info("Retraining only the LSTM (3 configs) for NO2_alpha...")
    df_lstm, results_lstm = train_all_combinations(
        df, train_mask, test_mask, RESULTS_S9_MODELS,
        models=["LSTM"], gases=["NO2_alpha"],
    )

    # Separate results table
    out_csv = RESULTS_S9 / "tabla_maestra_no2alpha_lstm.csv"
    df_lstm.to_csv(out_csv, index=False)
    log.info("NO2_alpha LSTM table: %s (%d rows)", out_csv.name, len(df_lstm))

    log.info("")
    log.info("=" * 70)
    log.info("  NO2_alpha LSTM SUMMARY (%s)", NODE_ID)
    log.info("=" * 70)
    if not df_lstm.empty:
        for _, row in df_lstm.sort_values("R2_test", ascending=False).iterrows():
            log.info("  %s-%s: R2_test=%.3f  RMSE=%.2f  overfit=%.2f  CEN=%s",
                     row["model"], row["config"],
                     row["R2_test"], row["RMSE"], row["overfit_ratio"], row["CEN_level"])
    else:
        log.warning("No LSTM was trained successfully.")

    elapsed = time.time() - t_start
    log.info("Total time: %.1f min", elapsed / 60)


if __name__ == "__main__":
    run()
