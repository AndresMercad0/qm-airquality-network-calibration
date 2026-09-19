"""
Step s2: conversion of the Alphasense signals to ug/m3

Reads the s1 CSV and converts the electrode voltages of the NO2-B43F,
OX-B431 and CO-B4 cells to ug/m3: Algorithm 1 of AAN 803-05, the NO2
cross-sensitivity correction of the ozone cell and the ideal gas law.
The output CSV holds the concentrations instead of the raw columns.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    NODE_ID, DATA_S1, DATA_S2, COL_TEMP, COL_TS,
    ALPHASENSE_CALIBRATION, NT_TEMPS, NT_VALUES,
    ALPHASENSE_RAW_COLS, ALPHASENSE_VOLTAGE_COLS,
    MOLECULAR_WEIGHTS, P_STANDARD, R_GAS,
    COL_ALPHA_NO2, COL_ALPHA_O3, COL_ALPHA_CO,
)
from utils import detect_csv, extract_dates_from_csv, build_output_name


# -------------------------------------------------------------------------
# Conversion helpers (AAN 803-05, Algorithm 1)
# -------------------------------------------------------------------------
def interpolate_nT(temp_celsius: pd.Series, gas: str) -> pd.Series:
    """Interpolate the nT(T) factor of the gas, with the temperature clamped to -30..50 C."""
    clamped = temp_celsius.clip(lower=-30.0, upper=50.0)
    return pd.Series(
        np.interp(clamped.values, NT_TEMPS, NT_VALUES[gas]),
        index=temp_celsius.index,
    )


def compute_vcorr(op1_mV, op2_mV, ez_we, ez_aux, nT):
    """Return the working electrode signal (mV) corrected with the auxiliary electrode and nT."""
    return (op1_mV - ez_we) - nT * (op2_mV - ez_aux)


def ppb_to_ug_m3(ppb, mw, temp_celsius):
    """Convert ppb to ug/m3 with the ideal gas law at the measured temperature."""
    T_K = temp_celsius + 273.15
    return ppb * mw * P_STANDARD / (R_GAS * T_K) * 1e-3


# -------------------------------------------------------------------------
# Alphasense conversion from voltage to ug/m3 (NO2-B43F, OX-B431, CO-B4)
# -------------------------------------------------------------------------
def run_s2(input_csv: Path = None) -> Path:
    """
    Convert the Alphasense voltages of the s1 CSV to ug/m3 and replace the raw columns.
    Return the path of the converted CSV.
    """
    if input_csv is None:
        input_csv = detect_csv(DATA_S1, f"{NODE_ID}_QC_RAW_wide_data_*.csv")
    if input_csv is None or not input_csv.exists():
        raise FileNotFoundError(f"No s1 CSV found in {DATA_S1}")

    print(f"s2, Alphasense conversion: {input_csv.name}")
    df = pd.read_csv(input_csv)
    n_rows = len(df)
    print(f"  {n_rows:,} rows")

    missing = [c for c in ALPHASENSE_RAW_COLS + [COL_TEMP] if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    insert_pos = df.columns.get_loc(ALPHASENSE_RAW_COLS[0])
    temp = df[COL_TEMP].astype(float)

    # Voltage from V to mV
    voltages_mV = {}
    for gas in ("no2", "o3", "co"):
        op1_col, op2_col = ALPHASENSE_VOLTAGE_COLS[gas]
        voltages_mV[gas] = {
            "op1": df[op1_col].astype(float) * 1000.0,
            "op2": df[op2_col].astype(float) * 1000.0,
        }

    # nT(T)
    nT = {gas: interpolate_nT(temp, gas) for gas in ("no2", "o3", "co")}

    # Corrected signal (Algorithm 1)
    vcorr = {}
    for gas in ("no2", "o3", "co"):
        c = ALPHASENSE_CALIBRATION[gas]
        vcorr[gas] = compute_vcorr(
            voltages_mV[gas]["op1"], voltages_mV[gas]["op2"],
            c["ez_we"], c["ez_aux"], nT[gas])

    # ppb
    no2_ppb = vcorr["no2"] / abs(ALPHASENSE_CALIBRATION["no2"]["sensitivity"])

    # O3: remove the NO2 cross-sensitivity of the OX-B431 cell (AAN 803-05, "Compensation on
    # the OX-A431 or OX-B431 sensor"). The NO2 term is subtracted in magnitude because this
    # conversion divides by abs(sensitivity); the sign of gain x no2_sens differs between nodes
    no2_ppm = no2_ppb / 1000.0
    o3_cal = ALPHASENSE_CALIBRATION["o3"]
    interference_mV = abs(o3_cal["gain"] * o3_cal["no2_sens"]) * no2_ppm
    vcorr_o3_corrected = vcorr["o3"] - interference_mV
    o3_ppb = vcorr_o3_corrected / abs(o3_cal["sensitivity"])
    co_ppb = vcorr["co"] / abs(ALPHASENSE_CALIBRATION["co"]["sensitivity"])

    # ppb to ug/m3
    no2_ug = ppb_to_ug_m3(no2_ppb, MOLECULAR_WEIGHTS["no2"], temp)
    o3_ug = ppb_to_ug_m3(o3_ppb, MOLECULAR_WEIGHTS["o3"], temp)
    co_ug = ppb_to_ug_m3(co_ppb, MOLECULAR_WEIGHTS["co"], temp)

    # Count the negative values. They are not clipped: negative instrument noise is valid and
    # clipping would bias the OLS comparator upwards; s3 removes the extremes with physical limits
    stats = {}
    for name, series in [("NO2", no2_ug), ("O3", o3_ug), ("CO", co_ug)]:
        n_valid = series.notna().sum()
        n_neg = (series < 0).sum()
        pct_neg = (n_neg / n_valid * 100) if n_valid > 0 else 0.0
        stats[name] = {"n_neg": n_neg, "pct_neg": pct_neg}

    # Replace the raw columns by the ug/m3 columns
    df.drop(columns=ALPHASENSE_RAW_COLS, inplace=True)
    df.insert(insert_pos, COL_ALPHA_NO2, no2_ug)
    df.insert(insert_pos + 1, COL_ALPHA_O3, o3_ug)
    df.insert(insert_pos + 2, COL_ALPHA_CO, co_ug)

    # Save
    DATA_S2.mkdir(parents=True, exist_ok=True)
    start_date, end_date = extract_dates_from_csv(df)
    out_name = build_output_name(NODE_ID, "ALPHASENSE_UG_M3", start_date, end_date)
    out_csv = DATA_S2 / out_name
    df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv.name}")
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    for name, col in [("NO2", COL_ALPHA_NO2), ("O3", COL_ALPHA_O3), ("CO", COL_ALPHA_CO)]:
        s = df[col]
        print(f"  {name}: mean={s.mean():.2f}, median={s.median():.2f}, "
              f"NaN={s.isna().sum():,}, negative={stats[name]['n_neg']:,}")

    return out_csv


if __name__ == "__main__":
    run_s2()
