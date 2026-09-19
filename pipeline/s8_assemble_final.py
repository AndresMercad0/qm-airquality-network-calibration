"""
Step s8: assembly of the final CSV files

Merges on the timestamp the s3 CSV (metadata, ambient variables and clean
Alphasense data) with the calibrated columns of s5 and s6. Writes two CSV
files of the node: one with the manufacturer calibration and one with the
OLS calibration.
"""

from pathlib import Path

import pandas as pd

from config import (
    NODE_ID, DATA_S3, DATA_S5, DATA_S6, DATA_S8, COL_TS, COL_PM25, COL_PM10,
    S8_COLS_FABRICANTE, S8_COLS_OLS,
    CURVAS_FAB_COL_NAMES, OLS_COL_NAMES,
)
from utils import detect_csv, extract_dates_from_csv, build_output_name


# -------------------------------------------------------------------------
# Final assembly: one CSV per calibration method
# -------------------------------------------------------------------------
def run_s8(input_s3_csv: Path = None, input_s5_csv: Path = None,
           input_s6_csv: Path = None) -> tuple:
    """
    Merge the s3 CSV (clean Alphasense, metadata and ambient variables) with the
    s5 and s6 calibrations. File 1 holds the manufacturer calibration and file 2
    the OLS calibration. Return both paths.
    """
    if input_s3_csv is None:
        input_s3_csv = detect_csv(DATA_S3, f"{NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    if input_s5_csv is None:
        input_s5_csv = detect_csv(DATA_S5, f"{NODE_ID}_CALIBRATED_CURVAS_FABRICANTE_wide_data_*.csv")
    if input_s6_csv is None:
        input_s6_csv = detect_csv(DATA_S6, f"{NODE_ID}_CALIBRATED_OLS_wide_data_*.csv")

    for name, path in [("s3", input_s3_csv), ("s5", input_s5_csv), ("s6", input_s6_csv)]:
        if path is None or not path.exists():
            raise FileNotFoundError(f"No CSV found for {name}")

    print(f"s8, final assembly:")
    print(f"  s3={input_s3_csv.name}")
    print(f"  s5={input_s5_csv.name}")
    print(f"  s6={input_s6_csv.name}")

    df_s3 = pd.read_csv(input_s3_csv, parse_dates=[COL_TS])
    df_s5 = pd.read_csv(input_s5_csv, parse_dates=[COL_TS])
    df_s6 = pd.read_csv(input_s6_csv, parse_dates=[COL_TS])

    # Columns taken from s5 and s6
    cols_s5 = [c for c in CURVAS_FAB_COL_NAMES.values() if c in df_s5.columns]
    cols_s6 = [c for c in OLS_COL_NAMES.values() if c in df_s6.columns]

    # Master DataFrame: s3 as the base, s5 and s6 merged on the timestamp
    master = df_s3.copy()
    master = master.merge(df_s5[[COL_TS] + cols_s5], on=COL_TS, how="left")
    master = master.merge(df_s6[[COL_TS] + cols_s6], on=COL_TS, how="left")

    print(f"  s3: {len(df_s3):,}, s5: {len(df_s5):,}, s6: {len(df_s6):,} rows")
    print(f"  Master: {len(master):,} rows, {len(master.columns)} columns")

    DATA_S8.mkdir(parents=True, exist_ok=True)
    start_date, end_date = extract_dates_from_csv(master)

    # File 1: manufacturer calibration
    cols_fab = [c for c in S8_COLS_FABRICANTE if c in master.columns]
    df_fab = master[cols_fab].copy()
    df_fab = df_fab.rename(columns={
        COL_PM25: COL_PM25 + "_fabricante",
        COL_PM10: COL_PM10 + "_fabricante",
    })
    df_fab[COL_TS] = df_fab[COL_TS].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    name_fab = build_output_name(NODE_ID, "ASSEMBLED_CAL_FABRICANTE", start_date, end_date)
    csv_fab = DATA_S8 / name_fab
    df_fab.to_csv(csv_fab, index=False)

    mb_fab = csv_fab.stat().st_size / (1024 ** 2)
    print(f"\n  [1] {name_fab} ({mb_fab:.1f} MB)")
    print(f"      {len(df_fab.columns)} columns: {list(df_fab.columns)}")

    # File 2: OLS calibration
    cols_ols = [c for c in S8_COLS_OLS if c in master.columns]
    df_ols = master[cols_ols].copy()
    df_ols = df_ols.rename(columns={COL_PM10: COL_PM10 + "_fabricante"})
    df_ols[COL_TS] = df_ols[COL_TS].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    name_ols = build_output_name(NODE_ID, "ASSEMBLED_CAL_OLS", start_date, end_date)
    csv_ols = DATA_S8 / name_ols
    df_ols.to_csv(csv_ols, index=False)

    mb_ols = csv_ols.stat().st_size / (1024 ** 2)
    print(f"\n  [2] {name_ols} ({mb_ols:.1f} MB)")
    print(f"      {len(df_ols.columns)} columns: {list(df_ols.columns)}")

    return csv_fab, csv_ols


if __name__ == "__main__":
    run_s8()
