"""
Step s4: preprocessing of the metal oxide signals

Reads the s1 CSV and converts the raw readings of the three metal oxide
sensors into the ratio RS/R0 used by the calibration: sensor resistance,
temperature and humidity correction with the manufacturer tables, and a
dynamic night-time baseline R0. Writes the preprocessed CSV of the node.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from config import (
    NODE_ID, DATA_S1, DATA_S4, COL_TS, COL_TEMP, COL_RH,
    GASES, SENSOR_PARAMS, ADC_MAX, T_REF, RH_REF,
    W_DAYS, ALPHA_EWMA, R0_PERCENTILE, R0_NIGHT_END,
    R0_MIN_SAMPLES, MAX_CARRY_FORWARD,
)
from utils import detect_csv, extract_dates_from_csv, build_output_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("s4")


# -------------------------------------------------------------------------
# Raw signal to sensor resistance RS
# -------------------------------------------------------------------------
def calcular_rs(raw_value: pd.Series, gas: str) -> pd.Series:
    """
    Convert the raw reading to the sensor resistance RS with the voltage divider law.
    ADC counts become a voltage first (ADC * VCC / 1023); the MQ131 is read as a voltage.
    """
    params = SENSOR_PARAMS[gas]
    if params["uses_voltage_col"]:
        voltage = raw_value.copy()
    else:
        adc = raw_value.copy()
        if params["inversion"]:
            adc = ADC_MAX - adc
        adc = adc.clip(1, ADC_MAX - 1)
        voltage = adc * (params["vcc"] / ADC_MAX)

    voltage_safe = voltage.where(voltage > 0)
    rs = params["rl"] * (params["vcc"] - voltage_safe) / voltage_safe
    rs = rs.where(rs > 0)
    return rs


# -------------------------------------------------------------------------
# Temperature and humidity correction
# -------------------------------------------------------------------------
def corregir_trh(rs: pd.Series, temp: pd.Series, rh: pd.Series,
                 gas: str) -> pd.Series:
    """
    Correct RS with the manufacturer T/RH table (bilinear interpolation), normalised
    to the reference conditions. T and RH are clamped to the range of the table.
    """
    params = SENSOR_PARAMS[gas]
    interpolador = RegularGridInterpolator(
        (params["temp_grid"], params["rh_grid"]),
        params["trh_table"],
        method="linear", bounds_error=False, fill_value=None,
    )

    factor_referencia = float(interpolador([[T_REF, RH_REF]])[0])
    temp_clamped = temp.clip(params["temp_grid"].min(), params["temp_grid"].max())
    rh_clamped = rh.clip(params["rh_grid"].min(), params["rh_grid"].max())

    mascara_validos = temp.notna() & rh.notna() & rs.notna()
    puntos = np.column_stack([
        temp_clamped.fillna(T_REF).values,
        rh_clamped.fillna(RH_REF).values,
    ])
    factor_medido = interpolador(puntos)
    factor_norm = factor_medido / factor_referencia

    if params["trh_type"] == "RS/RSo":
        rs_corr = rs / factor_norm
    else:
        rs_corr = rs * factor_norm

    rs_corr = pd.Series(rs_corr, index=rs.index)
    rs_corr[~mascara_validos] = np.nan
    return rs_corr


# -------------------------------------------------------------------------
# Dynamic R0 (night-time percentile, moving window, EWMA)
# -------------------------------------------------------------------------
def calcular_r0_dinamico(df: pd.DataFrame, rs_corr_dict: dict) -> dict:
    """
    Estimate a daily R0 of each gas from the night-time values of RS_corr.
    Each day takes a percentile of the 15 min night-time means of its moving window,
    smoothed with an EWMA. Days without enough samples carry the last R0 forward.
    """
    log.info("Computing the dynamic R0 (P%d, %d day window, night-time 0-%dh)...",
             R0_PERCENTILE, W_DAYS, R0_NIGHT_END)

    df_trabajo = df.copy()
    df_trabajo["t15"] = df_trabajo[COL_TS].dt.floor("15min")
    df_trabajo["hora_del_dia"] = df_trabajo[COL_TS].dt.hour
    df_trabajo["fecha"] = df_trabajo[COL_TS].dt.date

    for gas in GASES:
        df_trabajo[f"rsc_{gas}"] = rs_corr_dict[gas].values

    datos_nocturnos = df_trabajo[df_trabajo["hora_del_dia"] < R0_NIGHT_END].copy()
    log.info("  Night-time rows: %d of %d", len(datos_nocturnos), len(df_trabajo))

    columnas_agg = {f"rsc_{gas}": "mean" for gas in GASES}
    columnas_agg["fecha"] = "first"
    datos_15min = datos_nocturnos.groupby("t15").agg(columnas_agg).reset_index()

    todas_las_fechas = sorted(df_trabajo["fecha"].unique())
    n_dias = len(todas_las_fechas)

    r0_diario_raw = {gas: {} for gas in GASES}
    registros_diagnostico = []

    for i, fecha_actual in enumerate(todas_las_fechas):
        inicio = max(0, i - W_DAYS + 1)
        fechas_ventana = set(todas_las_fechas[inicio:i + 1])
        datos_ventana = datos_15min[datos_15min["fecha"].isin(fechas_ventana)]

        registro_dia = {"date": fecha_actual, "window_days": len(fechas_ventana)}

        for gas in GASES:
            col_rsc = f"rsc_{gas}"
            valores = datos_ventana[col_rsc].dropna()
            registro_dia[f"n_{gas}"] = len(valores)

            if len(valores) >= R0_MIN_SAMPLES:
                if SENSOR_PARAMS[gas]["r0_stat"] == "upper":
                    percentil = R0_PERCENTILE
                else:
                    percentil = 100 - R0_PERCENTILE
                r0_diario_raw[gas][fecha_actual] = float(np.percentile(valores, percentil))
            else:
                r0_diario_raw[gas][fecha_actual] = None

        registros_diagnostico.append(registro_dia)

    # EWMA smoothing and carry-forward
    r0_series = {}
    for gas in GASES:
        valores_r0 = []
        r0_previo = None
        dias_carry = 0

        for fecha in todas_las_fechas:
            valor = r0_diario_raw[gas].get(fecha)
            if valor is not None and np.isfinite(valor):
                if r0_previo is not None:
                    suavizado = ALPHA_EWMA * valor + (1 - ALPHA_EWMA) * r0_previo
                else:
                    suavizado = valor
                r0_previo = suavizado
                dias_carry = 0
                valores_r0.append(suavizado)
            elif r0_previo is not None and dias_carry < MAX_CARRY_FORWARD:
                dias_carry += 1
                valores_r0.append(r0_previo)
            else:
                valores_r0.append(np.nan)

        r0_series[gas] = pd.Series(
            valores_r0, index=pd.Index(todas_las_fechas, name="date"))

        n_valid = np.isfinite(valores_r0).sum()
        r0_mediana = np.nanmedian(valores_r0)
        log.info("  R0 %-3s: %d/%d valid days (%.1f%%), median=%.0f ohm",
                 gas, n_valid, n_dias, 100 * n_valid / n_dias, r0_mediana)

    return {"r0_series": r0_series, "diagnostico": pd.DataFrame(registros_diagnostico)}


def expandir_r0_a_timestamps(r0_series: pd.Series, timestamps: pd.Series) -> pd.Series:
    """Map the daily R0 to every timestamp of its date."""
    return timestamps.dt.date.map(r0_series.to_dict()).astype(float)


# -------------------------------------------------------------------------
# Preprocessing of the metal oxide sensors (raw signal to ratio RS_corr / R0)
# -------------------------------------------------------------------------
def run_s4(input_csv: Path = None) -> Path:
    """
    Add RS, RS_corr, R0 and the ratio RS_corr / R0 of each gas to the s1 CSV.
    Return the path of the preprocessed CSV.
    """
    if input_csv is None:
        input_csv = detect_csv(DATA_S1, f"{NODE_ID}_QC_RAW_wide_data_*.csv")
    if input_csv is None or not input_csv.exists():
        raise FileNotFoundError(f"No s1 CSV found in {DATA_S1}")

    print(f"s4, preprocessing: {input_csv.name}")
    df = pd.read_csv(input_csv, parse_dates=[COL_TS])
    df = df.sort_values(COL_TS).reset_index(drop=True)

    for col in [COL_TEMP, COL_RH]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Raw signal to RS
    log.info("Computing RS...")
    rs_dict = {}
    for gas in GASES:
        params = SENSOR_PARAMS[gas]
        raw_col = params["col_voltage"] if params["uses_voltage_col"] else params["col_raw"]
        df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
        rs = calcular_rs(df[raw_col], gas)
        n_valid = rs.notna().sum()
        log.info("  RS %-3s: %d valid (%.1f%%)", gas, n_valid, 100 * n_valid / len(df))
        rs_dict[gas] = rs

    # T/RH correction
    log.info("T/RH correction...")
    rs_corr_dict = {}
    for gas in GASES:
        rs_corr = corregir_trh(rs_dict[gas], df[COL_TEMP], df[COL_RH], gas)
        log.info("  RS_corr %-3s: %d valid", gas, rs_corr.notna().sum())
        rs_corr_dict[gas] = rs_corr

    # Dynamic R0
    r0_result = calcular_r0_dinamico(df, rs_corr_dict)

    # Ratio and new columns of the DataFrame
    for gas in GASES:
        df[f"rs_{gas}"] = rs_dict[gas].values
        df[f"rsc_{gas}"] = rs_corr_dict[gas].values
        r0_expanded = expandir_r0_a_timestamps(r0_result["r0_series"][gas], df[COL_TS])
        df[f"r0_{gas}"] = r0_expanded.values
        df[f"ratio_{gas}"] = (rs_corr_dict[gas] / r0_expanded).values

    # Save
    DATA_S4.mkdir(parents=True, exist_ok=True)
    start_date, end_date = extract_dates_from_csv(df)
    out_name = build_output_name(NODE_ID, "PREPROCESS", start_date, end_date)
    out_csv = DATA_S4 / out_name
    df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv.name}")
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    for gas in GASES:
        ratio = df[f"ratio_{gas}"]
        print(f"  ratio_{gas}: {ratio.notna().sum():,} valid, "
              f"mean={ratio.mean():.4f}, median={ratio.median():.4f}")

    return out_csv


if __name__ == "__main__":
    run_s4()
