"""
Step s5: calibration with the manufacturer curves

Reads the s4 CSV and converts the ratio RS/R0 of each metal oxide sensor
to ug/m3 with the datasheet curves and the ideal gas law. Writes the
calibrated CSV and metricas_v1v2.csv, the 15 min metrics against the s3
Alphasense data (CO, O3) and the reference station (NO2, PM2.5).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    NODE_ID, DATA_S4, DATA_S3, DATA_S5, RESULTS_S5,
    COL_TS, COL_TEMP, COL_PM25, GASES, REF_DATA_DIR, REF_CSV_PATTERN,
    COL_ALPHA_NO2, COL_ALPHA_O3, COL_ALPHA_CO,
    COL_REF_NO2, COL_REF_PM25, REF_STATION_LABEL,
    CURVAS_FAB_COL_NAMES,
    MOLECULAR_WEIGHTS, P_STANDARD, R_GAS,
    CO_CURVE_RATIO, CO_CURVE_PPM,
    NO2_CURVE_RATIO, NO2_CURVE_PPM,
    O3_CURVE_RATIO, O3_CURVE_PPB,
)
from utils import (
    detect_csv, extract_dates_from_csv, build_output_name,
    nivel_calidad,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("s5")


# -------------------------------------------------------------------------
# Manufacturer curves (ratio to concentration through the look-up tables)
# -------------------------------------------------------------------------
def ratio_to_ppm_co(ratio: pd.Series) -> pd.Series:
    """
    Convert the GM-702B ratio to ppm of CO by log-log interpolation.
    The curve decreases (higher RS/R0, lower concentration), so the table is
    reversed: np.interp needs ascending x values.
    """
    valid = ratio.where(ratio > 0)
    log_ratio = np.log10(valid)
    log_r_asc = np.log10(CO_CURVE_RATIO[::-1])
    log_c_asc = np.log10(CO_CURVE_PPM[::-1])
    log_ppm = np.interp(log_ratio, log_r_asc, log_c_asc)
    result = pd.Series(10 ** log_ppm, index=ratio.index)
    result[valid.isna()] = np.nan
    return result


def ratio_to_ppm_no2(ratio: pd.Series) -> pd.Series:
    """Convert the GM-102B ratio to ppm of NO2 by linear interpolation (ascending table)."""
    result = pd.Series(
        np.interp(ratio.values, NO2_CURVE_RATIO, NO2_CURVE_PPM),
        index=ratio.index,
    )
    result[ratio.isna()] = np.nan
    return result


def ratio_to_ppb_o3(ratio: pd.Series) -> pd.Series:
    """Convert the MQ131 ratio to ppb of O3 by log-log interpolation (ascending table)."""
    valid = ratio.where(ratio > 0)
    log_ratio = np.log10(valid)
    log_r = np.log10(O3_CURVE_RATIO)
    log_c = np.log10(O3_CURVE_PPB)
    log_ppb = np.interp(log_ratio, log_r, log_c)
    result = pd.Series(10 ** log_ppb, index=ratio.index)
    result[valid.isna()] = np.nan
    return result


def ppb_to_ugm3(ppb: pd.Series, mw: float, temp_celsius: pd.Series) -> pd.Series:
    """Convert ppb to ug/m3 with the ideal gas law."""
    T_K = temp_celsius + 273.15
    return ppb * mw * P_STANDARD / (R_GAS * T_K) * 1e-3


def ppm_to_ugm3(ppm: pd.Series, mw: float, temp_celsius: pd.Series) -> pd.Series:
    return ppb_to_ugm3(ppm * 1000, mw, temp_celsius)


# -------------------------------------------------------------------------
# Metrics against the reference
# -------------------------------------------------------------------------
def calcular_metricas(y_obs: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R2, RMSE, MAE, MBE, nRMSE (%) and the quality tier of a prediction."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_obs, y_pred)
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    mae = mean_absolute_error(y_obs, y_pred)
    mbe = float(np.mean(y_pred - y_obs))
    mean_ref = float(np.mean(y_obs))
    nrmse = (rmse / mean_ref * 100) if mean_ref > 0 else np.nan
    return {
        "r2": round(r2, 4), "rmse": round(rmse, 2), "mae": round(mae, 2),
        "mbe": round(mbe, 2), "nrmse": round(nrmse, 1),
        "nivel": nivel_calidad(r2, nrmse), "n": len(y_obs),
    }


# -------------------------------------------------------------------------
# Calibration with the manufacturer curves and baseline metrics
# -------------------------------------------------------------------------
def run_s5(input_s4_csv: Path = None, input_s3_csv: Path = None) -> Path:
    """
    Apply the manufacturer curves to the ratio RS_corr / R0 and convert to ug/m3.
    Baseline metrics are computed at 15 min against the Alphasense cells (CO, O3)
    and the reference station (NO2, PM2.5). Return the path of the calibrated CSV.
    """
    if input_s4_csv is None:
        input_s4_csv = detect_csv(DATA_S4, f"{NODE_ID}_PREPROCESS_wide_data_*.csv")
    if input_s4_csv is None or not input_s4_csv.exists():
        raise FileNotFoundError(f"No s4 CSV found in {DATA_S4}")

    print(f"s5, manufacturer curves: {input_s4_csv.name}")
    df = pd.read_csv(input_s4_csv, parse_dates=[COL_TS])
    print(f"  {len(df):,} rows")

    temp = df[COL_TEMP].astype(float)

    # Manufacturer curves (datasheet look-up tables)
    co_ppm = ratio_to_ppm_co(df["ratio_CO"])
    co_ugm3 = ppm_to_ugm3(co_ppm, MOLECULAR_WEIGHTS["co"], temp).clip(lower=0)

    no2_ppm = ratio_to_ppm_no2(df["ratio_NO2"])
    no2_ugm3 = ppm_to_ugm3(no2_ppm, MOLECULAR_WEIGHTS["no2"], temp).clip(lower=0)

    o3_ppb = ratio_to_ppb_o3(df["ratio_O3"])
    o3_ugm3 = ppb_to_ugm3(o3_ppb, MOLECULAR_WEIGHTS["o3"], temp).clip(lower=0)

    # Output DataFrame
    df_out = df[[COL_TS, COL_TEMP, "humidity_low_sht45_percentage"]].copy()
    df_out[CURVAS_FAB_COL_NAMES["CO"]] = co_ugm3.values
    df_out[CURVAS_FAB_COL_NAMES["NO2"]] = no2_ugm3.values
    df_out[CURVAS_FAB_COL_NAMES["O3"]] = o3_ugm3.values

    # Low-cost PM2.5, kept to compare it with the reference station
    if COL_PM25 in df.columns:
        df_out["pm2_5_low_sps30_ug_m3"] = df[COL_PM25].values

    for gas, col in [("CO", CURVAS_FAB_COL_NAMES["CO"]),
                     ("NO2", CURVAS_FAB_COL_NAMES["NO2"]),
                     ("O3", CURVAS_FAB_COL_NAMES["O3"])]:
        vals = df_out[col].dropna()
        log.info("  %s: %d valid, mean=%.1f, median=%.1f ug/m3",
                 gas, len(vals), vals.mean(), vals.median())

    # Save the CSV
    DATA_S5.mkdir(parents=True, exist_ok=True)
    start_date, end_date = extract_dates_from_csv(df)
    out_name = build_output_name(NODE_ID, "CALIBRATED_CURVAS_FABRICANTE", start_date, end_date)
    out_csv = DATA_S5 / out_name
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")

    # Metrics against the references
    RESULTS_S5.mkdir(parents=True, exist_ok=True)

    if input_s3_csv is None:
        input_s3_csv = detect_csv(DATA_S3, f"{NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")

    ref_csv = detect_csv(REF_DATA_DIR, REF_CSV_PATTERN)

    metricas_rows = []

    # Alphasense cells (CO, O3)
    if input_s3_csv and input_s3_csv.exists():
        df_ref = pd.read_csv(input_s3_csv, parse_dates=[COL_TS])

        df_out["t15"] = pd.to_datetime(df_out[COL_TS]).dt.floor("15min")
        df_ref["t15"] = df_ref[COL_TS].dt.floor("15min")

        est_15 = df_out.groupby("t15").agg({
            CURVAS_FAB_COL_NAMES["CO"]: "mean",
            CURVAS_FAB_COL_NAMES["O3"]: "mean",
        }).reset_index()

        ref_15 = df_ref.groupby("t15").agg({
            COL_ALPHA_CO: "mean",
            COL_ALPHA_O3: "mean",
        }).reset_index()

        merged = est_15.merge(ref_15, on="t15", how="inner")

        for gas, est_col, ref_col in [
            ("CO", CURVAS_FAB_COL_NAMES["CO"], COL_ALPHA_CO),
            ("O3", CURVAS_FAB_COL_NAMES["O3"], COL_ALPHA_O3),
        ]:
            mask = merged[est_col].notna() & merged[ref_col].notna()
            sub = merged[mask]
            if len(sub) >= 50:
                m = calcular_metricas(sub[ref_col].values, sub[est_col].values)
                m["gas"] = gas
                m["version"] = "v1v2"
                m["ref"] = ref_col
                metricas_rows.append(m)
                log.info("  %s vs Alphasense: R2=%.3f, RMSE=%.1f, nRMSE=%.1f%% [%s]",
                         gas, m["r2"], m["rmse"], m["nrmse"], m["nivel"])

    # Reference station (NO2 and PM2.5)
    if ref_csv and ref_csv.exists():
        df_ref = pd.read_csv(ref_csv, parse_dates=[COL_TS])
        df_ref["t15"] = df_ref[COL_TS].dt.floor("15min")

        df_out_t15 = df_out.copy()
        if "t15" not in df_out_t15.columns:
            df_out_t15["t15"] = pd.to_datetime(df_out_t15[COL_TS]).dt.floor("15min")

        # NO2
        ref_no2_15 = df_ref.groupby("t15").agg({COL_REF_NO2: "mean"}).reset_index()
        no2_15 = df_out_t15.groupby("t15").agg({
            CURVAS_FAB_COL_NAMES["NO2"]: "mean"
        }).reset_index()

        merged_no2 = no2_15.merge(ref_no2_15, on="t15", how="inner")
        mask = merged_no2[CURVAS_FAB_COL_NAMES["NO2"]].notna() & merged_no2[COL_REF_NO2].notna()
        sub = merged_no2[mask]
        if len(sub) >= 50:
            m = calcular_metricas(sub[COL_REF_NO2].values,
                                  sub[CURVAS_FAB_COL_NAMES["NO2"]].values)
            m["gas"] = "NO2"
            m["version"] = "v1v2"
            m["ref"] = COL_REF_NO2
            metricas_rows.append(m)
            log.info("  NO2 vs %s: R2=%.3f, RMSE=%.1f, nRMSE=%.1f%% [%s]",
                     REF_STATION_LABEL, m["r2"], m["rmse"], m["nrmse"], m["nivel"])

        # PM2.5
        if COL_REF_PM25 in df_ref.columns and "pm2_5_low_sps30_ug_m3" in df_out_t15.columns:
            ref_pm25_15 = df_ref.groupby("t15").agg({COL_REF_PM25: "mean"}).reset_index()
            pm25_15 = df_out_t15.groupby("t15").agg({
                "pm2_5_low_sps30_ug_m3": "mean"
            }).reset_index()

            merged_pm = pm25_15.merge(ref_pm25_15, on="t15", how="inner")
            mask = merged_pm["pm2_5_low_sps30_ug_m3"].notna() & merged_pm[COL_REF_PM25].notna()
            sub = merged_pm[mask]
            if len(sub) >= 50:
                m = calcular_metricas(sub[COL_REF_PM25].values,
                                      sub["pm2_5_low_sps30_ug_m3"].values)
                m["gas"] = "PM2.5"
                m["version"] = "raw_vs_ref"
                m["ref"] = COL_REF_PM25
                metricas_rows.append(m)
                log.info("  PM2.5 vs %s: R2=%.3f, RMSE=%.1f, nRMSE=%.1f%% [%s]",
                         REF_STATION_LABEL, m["r2"], m["rmse"], m["nrmse"], m["nivel"])

    if metricas_rows:
        df_met = pd.DataFrame(metricas_rows)
        df_met.to_csv(RESULTS_S5 / "metricas_v1v2.csv", index=False)
        print(f"Metrics in: {RESULTS_S5 / 'metricas_v1v2.csv'}")

    return out_csv


if __name__ == "__main__":
    run_s5()
