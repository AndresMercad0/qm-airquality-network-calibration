"""
Step s7: evaluation of the calibrations by cost tier

Reads the s1, s3, s5 and s6 outputs and the reference station, and scores
each estimate against its reference or comparator on 15 min means: R2,
RMSE, nRMSE, MAE, MBE, Pearson r. Writes metricas_por_nivel_de_costo.csv,
reporte_comparativa_por_costo.md and a two-week plot per pollutant.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import (
    NODE_ID, DATA_S1, DATA_S3, DATA_S5, DATA_S6, RESULTS_DIR,
    COL_TS, COL_TEMP, COL_PM25,
    COL_ALPHA_NO2, COL_ALPHA_O3, COL_ALPHA_CO,
    COL_REF_NO2, COL_REF_PM25,
    REF_DATA_DIR, REF_CSV_PATTERN, REF_STATION_LABEL, REF_STATION_SHORT,
    OLS_COL_NAMES, CURVAS_FAB_COL_NAMES,
)
from utils import detect_csv, nivel_calidad

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("s7")

RESULTS_S7 = RESULTS_DIR / "s7_evaluate"
MIN_OBS = 50  # Minimum paired 15 min periods to evaluate a comparison


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
def calcular_metricas(y_obs: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return R2, RMSE, nRMSE, MAE, MBE and Pearson r between observed and predicted."""
    r2 = r2_score(y_obs, y_pred)
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    mae = mean_absolute_error(y_obs, y_pred)
    mbe = float(np.mean(y_pred - y_obs))
    mean_ref = float(np.mean(y_obs))
    nrmse = (rmse / mean_ref * 100) if mean_ref > 0 else np.nan
    pearson_r = float(np.corrcoef(y_obs, y_pred)[0, 1])
    return {
        "r2": round(r2, 4), "rmse": round(rmse, 2), "nrmse": round(nrmse, 1),
        "mae": round(mae, 2), "mbe": round(mbe, 2),
        "pearson_r": round(pearson_r, 4),
        "nivel": nivel_calidad(r2, nrmse), "n": len(y_obs),
        "mean_ref": round(mean_ref, 2), "mean_est": round(float(np.mean(y_pred)), 2),
    }


def evaluar_par(df_15: pd.DataFrame, col_est: str, col_ref: str,
                gas: str, nivel_costo: str, metodo: str, referencia: str,
                clip_zero: bool = False) -> dict:
    """
    Evaluate one estimate against its reference on 15 min means.
    With clip_zero, both series are clipped to >= 0 before the evaluation:
    negative concentrations from the AAN 803-05 temperature correction
    (algorithm 1) are over-compensation artefacts. The stored data are not modified.
    """
    mask = df_15[col_est].notna() & df_15[col_ref].notna()
    sub = df_15[mask].copy()
    if len(sub) < MIN_OBS:
        log.warning("  %s %s: only %d obs, skipping", gas, nivel_costo, len(sub))
        return None

    est_values = sub[col_est].values.copy()
    ref_values = sub[col_ref].values.copy()

    if clip_zero:
        n_neg_est = (est_values < 0).sum()
        n_neg_ref = (ref_values < 0).sum()
        if n_neg_est > 0 or n_neg_ref > 0:
            log.info("  %-5s %-12s clip>=0: %d neg est, %d neg ref (of %d)",
                     gas, nivel_costo, n_neg_est, n_neg_ref, len(est_values))
            est_values = np.maximum(est_values, 0)
            ref_values = np.maximum(ref_values, 0)

    m = calcular_metricas(ref_values, est_values)
    m["gas"] = gas
    m["nivel_costo"] = nivel_costo
    m["metodo"] = metodo
    m["referencia"] = referencia
    m["col_estimado"] = col_est
    m["col_referencia"] = col_ref

    log.info("  %-5s %-12s R2=%.3f  RMSE=%7.2f  nRMSE=%5.1f%%  [%s]  N=%d",
             gas, nivel_costo, m["r2"], m["rmse"], m["nrmse"], m["nivel"], m["n"])
    return m


# -------------------------------------------------------------------------
# Loading and resampling to 15 min
# -------------------------------------------------------------------------
def cargar_todo():
    """Load every input CSV and return one table of 15 min means."""

    # s1: clean low-cost data (SPS30 PM2.5)
    s1_csv = detect_csv(DATA_S1, f"{NODE_ID}_QC_RAW_wide_data_*.csv")
    # s3: clean Alphasense series (mid-cost)
    s3_csv = detect_csv(DATA_S3, f"{NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    # s5: manufacturer-curve concentrations (low-cost)
    s5_csv = detect_csv(DATA_S5, f"{NODE_ID}_CALIBRATED_CURVAS_FABRICANTE_wide_data_*.csv")
    # s6: OLS concentrations (low-cost)
    s6_csv = detect_csv(DATA_S6, f"{NODE_ID}_CALIBRATED_OLS_wide_data_*.csv")
    # Reference station (high-cost)
    ref_csv = detect_csv(REF_DATA_DIR, REF_CSV_PATTERN)

    for name, path in [("s1", s1_csv), ("s3", s3_csv), ("s5", s5_csv),
                        ("s6", s6_csv), (REF_STATION_SHORT, ref_csv)]:
        if path is None or not path.exists():
            raise FileNotFoundError(f"CSV not found: {name}")
        log.info("  %s: %s", name, path.name)

    df_s1 = pd.read_csv(s1_csv, parse_dates=[COL_TS])
    df_s3 = pd.read_csv(s3_csv, parse_dates=[COL_TS])
    df_s5 = pd.read_csv(s5_csv, parse_dates=[COL_TS])
    df_s6 = pd.read_csv(s6_csv, parse_dates=[COL_TS])
    df_ref = pd.read_csv(ref_csv, parse_dates=[COL_TS])

    # t15 column for the merge
    for df in [df_s1, df_s3, df_s5, df_s6, df_ref]:
        df["t15"] = df[COL_TS].dt.floor("15min")

    # s1: SPS30 PM2.5
    pm25_15 = df_s1.groupby("t15").agg({COL_PM25: "mean"}).reset_index()
    pm25_15 = pm25_15.rename(columns={COL_PM25: "pm25_low_raw"})

    # s3: Alphasense cells
    alpha_cols = [c for c in [COL_ALPHA_NO2, COL_ALPHA_O3, COL_ALPHA_CO]
                  if c in df_s3.columns]
    alpha_15 = df_s3.groupby("t15").agg({c: "mean" for c in alpha_cols}).reset_index()

    # s5: manufacturer curves (PM2.5 is added only if s5 carries it)
    v1v2_cols = [c for c in df_s5.columns
                 if c not in [COL_TS, COL_TEMP, "humidity_low_sht45_percentage", "t15"]
                 and "pm2_5" not in c]
    s5_agg = {c: "mean" for c in v1v2_cols}
    if "pm2_5_low_sps30_ug_m3" in df_s5.columns:
        s5_agg["pm2_5_low_sps30_ug_m3"] = "mean"
    v1v2_15 = df_s5.groupby("t15").agg(s5_agg).reset_index()

    # s6: OLS
    ols_cols = [c for c in OLS_COL_NAMES.values() if c in df_s6.columns]
    v3_15 = df_s6.groupby("t15").agg({c: "mean" for c in ols_cols}).reset_index()

    # Reference station
    ref_cols = [c for c in [COL_REF_NO2, COL_REF_PM25] if c in df_ref.columns]
    ref_15 = df_ref.groupby("t15").agg({c: "mean" for c in ref_cols}).reset_index()

    # Single 15 min table (outer merge)
    base = pm25_15
    for other in [alpha_15, v1v2_15, v3_15, ref_15]:
        base = base.merge(other, on="t15", how="outer")

    log.info("  Total 15 min periods: %d", len(base))
    return base


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------
def run_s7() -> Path:
    print(f"s7: evaluation by cost tier, {NODE_ID}")
    print()

    log.info("Loading data...")
    df_15 = cargar_todo()

    resultados = []

    # NO2: mid-cost and low-cost estimates against the reference station
    log.info("=== NO2 ===")

    # Alphasense NO2-B43F with the factory parameters (negatives clipped, see evaluar_par)
    r = evaluar_par(df_15, COL_ALPHA_NO2, COL_REF_NO2,
                    "NO2", "Mid-cost", "Cal. parámetros de fábrica (AAN-803-05)", REF_STATION_LABEL,
                    clip_zero=True)
    if r:
        resultados.append(r)

    # Alphasense NO2-B43F calibrated by OLS
    col_alpha_ols = OLS_COL_NAMES.get("NO2_alpha", "")
    if col_alpha_ols and col_alpha_ols in df_15.columns:
        r = evaluar_par(df_15, col_alpha_ols, COL_REF_NO2,
                        "NO2", "Mid-cost OLS",
                        "OLS colocalización (Alphasense NO₂-B43F)", REF_STATION_LABEL,
                        clip_zero=True)
        if r:
            resultados.append(r)

    # Low-cost manufacturer curve
    r = evaluar_par(df_15, CURVAS_FAB_COL_NAMES["NO2"], COL_REF_NO2,
                    "NO2", "Low-cost curvas fab.", "Curva fabricante GM-102B", REF_STATION_LABEL)
    if r:
        resultados.append(r)

    # Low-cost OLS
    r = evaluar_par(df_15, OLS_COL_NAMES["NO2"], COL_REF_NO2,
                    "NO2", "Low-cost OLS", "OLS colocalización", REF_STATION_LABEL)
    if r:
        resultados.append(r)

    # CO: low-cost estimates against the Alphasense cell (negatives clipped)
    log.info("=== CO ===")

    r = evaluar_par(df_15, CURVAS_FAB_COL_NAMES["CO"], COL_ALPHA_CO,
                    "CO", "Low-cost curvas fab.", "Curva fabricante GM-702B", "Alphasense CO-B4 (cal. fab.)",
                    clip_zero=True)
    if r:
        resultados.append(r)

    r = evaluar_par(df_15, OLS_COL_NAMES["CO"], COL_ALPHA_CO,
                    "CO", "Low-cost OLS", "OLS colocalización", "Alphasense CO-B4 (cal. fab.)",
                    clip_zero=True)
    if r:
        resultados.append(r)

    # O3: low-cost estimates against the Alphasense cell (negatives clipped)
    log.info("=== O3 ===")

    r = evaluar_par(df_15, CURVAS_FAB_COL_NAMES["O3"], COL_ALPHA_O3,
                    "O3", "Low-cost curvas fab.", "Curva fabricante MQ131", "Alphasense OX-B431 (cal. fab.)",
                    clip_zero=True)
    if r:
        resultados.append(r)

    r = evaluar_par(df_15, OLS_COL_NAMES["O3"], COL_ALPHA_O3,
                    "O3", "Low-cost OLS", "OLS colocalización", "Alphasense OX-B431 (cal. fab.)",
                    clip_zero=True)
    if r:
        resultados.append(r)

    # PM2.5: SPS30 factory calibration and OLS against the reference station
    log.info("=== PM2.5 ===")

    r = evaluar_par(df_15, "pm25_low_raw", COL_REF_PM25,
                    "PM2.5", "Low-cost (cal. fabricante)", "Calibración del fabricante", REF_STATION_LABEL)
    if r:
        resultados.append(r)

    if OLS_COL_NAMES.get("PM2.5") and OLS_COL_NAMES["PM2.5"] in df_15.columns:
        r = evaluar_par(df_15, OLS_COL_NAMES["PM2.5"], COL_REF_PM25,
                        "PM2.5", "Low-cost OLS", "OLS colocalización", REF_STATION_LABEL)
        if r:
            resultados.append(r)

    if not resultados:
        log.warning("No metrics were produced")
        return None

    df_met = pd.DataFrame(resultados)

    # Order: NO2, CO, O3, PM2.5; within each gas, mid-cost first and low-cost OLS last
    orden_gas = {"NO2": 0, "CO": 1, "O3": 2, "PM2.5": 3}
    orden_nivel = {"Mid-cost": 0, "Mid-cost OLS": 1, "Low-cost (cal. fabricante)": 2,
                   "Low-cost curvas fab.": 3, "Low-cost OLS": 4}
    df_met["_sort_gas"] = df_met["gas"].map(orden_gas)
    df_met["_sort_nivel"] = df_met["nivel_costo"].map(orden_nivel)
    df_met = df_met.sort_values(["_sort_gas", "_sort_nivel"]).drop(
        columns=["_sort_gas", "_sort_nivel"]).reset_index(drop=True)

    RESULTS_S7.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_S7 / "metricas_por_nivel_de_costo.csv"
    df_met.to_csv(csv_path, index=False)
    log.info("CSV: %s", csv_path)

    # Two-week comparison plots
    plots_dir = RESULTS_S7 / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log.info("Drawing the comparison plots (2 weeks)...")
    plot_comparacion_2semanas(df_15, plots_dir)

    generar_reporte(df_met)

    # Summary table on the console
    print("\n" + "=" * 90)
    print(f"  COMPARISON BY COST TIER: {NODE_ID}")
    print("=" * 90)
    print(f"  {'Gas':<6} {'Tier':<18} {'Method':<28} {'R2':>6} {'r':>6} {'RMSE':>7} {'nRMSE%':>7} {'Level':>12}  N")
    print("-" * 100)
    for _, row in df_met.iterrows():
        print(f"  {row['gas']:<6} {row['nivel_costo']:<18} {row['metodo']:<28} "
              f"{row['r2']:>6.3f} {row['pearson_r']:>6.3f} {row['rmse']:>7.2f} {row['nrmse']:>6.1f}% "
              f"{row['nivel']:>12}  {row['n']}")
    print("=" * 100)

    return csv_path


# -------------------------------------------------------------------------
# Comparison plots by cost tier (2 weeks)
# -------------------------------------------------------------------------
# Series, colours and axis label per pollutant
PLOT_CONFIG = {
    "NO2": {
        "series": [
            {"col": COL_REF_NO2,                   "label": f"{REF_STATION_SHORT} (high-cost)",        "color": "#2c3e50", "lw": 1.2, "alpha": 0.9},
            {"col": COL_ALPHA_NO2,                  "label": "Alphasense NO2-B43F (factory cal.)",  "color": "#e74c3c", "lw": 0.8, "alpha": 0.7},
            {"col": OLS_COL_NAMES.get("NO2_alpha", ""), "label": "Alphasense NO2-B43F (OLS)", "color": "#e67e22", "lw": 0.8, "alpha": 0.7},
            {"col": OLS_COL_NAMES["NO2"],            "label": "Low-cost OLS",           "color": "#3498db", "lw": 0.8, "alpha": 0.7},
            {"col": CURVAS_FAB_COL_NAMES["NO2"],     "label": "Low-cost manufacturer curves",   "color": "#95a5a6", "lw": 0.5, "alpha": 0.4},
        ],
        "ylabel": "NO2 (ug/m3)",
        "clip_zero": True,
    },
    "CO": {
        "series": [
            {"col": COL_ALPHA_CO,                  "label": "Alphasense CO-B4 (factory cal.)",  "color": "#2c3e50", "lw": 1.2, "alpha": 0.9},
            {"col": OLS_COL_NAMES["CO"],            "label": "Low-cost OLS",           "color": "#e74c3c", "lw": 0.8, "alpha": 0.7},
            {"col": CURVAS_FAB_COL_NAMES["CO"],     "label": "Low-cost manufacturer curves",   "color": "#95a5a6", "lw": 0.5, "alpha": 0.4},
        ],
        "ylabel": "CO (ug/m3)",
        "clip_zero": True,
    },
    "O3": {
        "series": [
            {"col": COL_ALPHA_O3,                  "label": "Alphasense OX-B431 (factory cal.)",  "color": "#2c3e50", "lw": 1.2, "alpha": 0.9},
            {"col": OLS_COL_NAMES["O3"],            "label": "Low-cost OLS",           "color": "#27ae60", "lw": 0.8, "alpha": 0.7},
            {"col": CURVAS_FAB_COL_NAMES["O3"],     "label": "Low-cost manufacturer curves",   "color": "#95a5a6", "lw": 0.5, "alpha": 0.4},
        ],
        "ylabel": "O3 (ug/m3)",
        "clip_zero": True,
    },
    "PM2.5": {
        "series": [
            {"col": COL_REF_PM25,              "label": f"{REF_STATION_SHORT} (high-cost)",          "color": "#2c3e50", "lw": 1.2, "alpha": 0.9},
            {"col": OLS_COL_NAMES.get("PM2.5", ""), "label": "Low-cost OLS",        "color": "#3498db", "lw": 0.8, "alpha": 0.7},
            {"col": "pm25_low_raw",             "label": "SPS30 (factory cal.)",  "color": "#8e44ad", "lw": 0.5, "alpha": 0.4},
        ],
        "ylabel": "PM2.5 (ug/m3)",
        "clip_zero": False,
    },
}


def seleccionar_ventana_2semanas(df_15: pd.DataFrame) -> tuple:
    """
    Select the 2-week window with the best data coverage.
    The search is restricted to the middle third of the dataset to avoid the edges.
    """
    ts = df_15["t15"].dropna().sort_values()
    t_min, t_max = ts.min(), ts.max()
    rango_total = t_max - t_min
    # Middle third of the dataset
    t_search_start = t_min + rango_total * 0.33
    t_search_end = t_max - rango_total * 0.33 - pd.Timedelta(days=14)

    if t_search_end <= t_search_start:
        t_search_start = t_min
        t_search_end = t_max - pd.Timedelta(days=14)

    mejor_inicio = t_search_start
    mejor_count = 0

    # Candidate windows every 2 days
    candidatos = pd.date_range(t_search_start, t_search_end, freq="2D")
    for t_start in candidatos:
        t_end = t_start + pd.Timedelta(days=14)
        mask = (df_15["t15"] >= t_start) & (df_15["t15"] < t_end)
        count = df_15.loc[mask].notna().sum().sum()
        if count > mejor_count:
            mejor_count = count
            mejor_inicio = t_start

    return mejor_inicio, mejor_inicio + pd.Timedelta(days=14)


def plot_comparacion_2semanas(df_15: pd.DataFrame, save_dir: Path):
    """
    Draw one figure per pollutant with every cost tier overlaid over 2 weeks.
    Hourly means are plotted for legibility.
    """

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    })

    t_start, t_end = seleccionar_ventana_2semanas(df_15)
    log.info("  2-week window: %s to %s",
             t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d"))

    mask_window = (df_15["t15"] >= t_start) & (df_15["t15"] < t_end)
    df_win = df_15[mask_window].copy().sort_values("t15")

    # Hourly means
    df_win = df_win.set_index("t15")
    numeric_cols = df_win.select_dtypes(include=[np.number]).columns
    df_h = df_win[numeric_cols].resample("1h").mean()

    for gas, cfg in PLOT_CONFIG.items():
        series_disponibles = [s for s in cfg["series"] if s["col"] in df_h.columns]
        if not series_disponibles:
            continue

        # The manufacturer curves go to their own panel when out of scale
        series_principales = [s for s in series_disponibles if "manufacturer curves" not in s["label"]]
        series_v1v2 = [s for s in series_disponibles if "manufacturer curves" in s["label"]]

        # Out of scale: median of the curves above 3 times the 95th percentile of the rest
        v1v2_fuera_escala = False
        if series_principales and series_v1v2:
            vals_main = pd.concat([df_h[s["col"]].dropna() for s in series_principales])
            vals_v1v2 = df_h[series_v1v2[0]["col"]].dropna()
            if len(vals_main) > 0 and len(vals_v1v2) > 0:
                p95_main = vals_main.quantile(0.95)
                p50_v1v2 = vals_v1v2.median()
                if p95_main > 0 and p50_v1v2 > p95_main * 3:
                    v1v2_fuera_escala = True

        if v1v2_fuera_escala:
            fig, (ax_main, ax_v1v2) = plt.subplots(
                2, 1, figsize=(16, 8), height_ratios=[3, 1], sharex=True)
        else:
            fig, ax_main = plt.subplots(figsize=(16, 5))
            ax_v1v2 = None

        # Main panel
        for s in series_principales:
            vals = df_h[s["col"]].copy()
            if cfg["clip_zero"]:
                vals = vals.clip(lower=0)
            ax_main.plot(df_h.index, vals, label=s["label"],
                         color=s["color"], lw=s["lw"], alpha=s["alpha"])

        if not v1v2_fuera_escala:
            for s in series_v1v2:
                vals = df_h[s["col"]].copy()
                if cfg["clip_zero"]:
                    vals = vals.clip(lower=0)
                ax_main.plot(df_h.index, vals, label=s["label"],
                             color=s["color"], lw=s["lw"], alpha=s["alpha"])

        ax_main.set_ylabel(cfg["ylabel"])
        ax_main.set_title(
            f"{NODE_ID}, {gas}: comparison by cost tier "
            f"({t_start.strftime('%d %b')} to {t_end.strftime('%d %b %Y')})",
            fontweight="bold")
        ax_main.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_ylim(bottom=0)

        # Manufacturer-curve panel
        if v1v2_fuera_escala and ax_v1v2 is not None and series_v1v2:
            s = series_v1v2[0]
            vals = df_h[s["col"]].copy()
            if cfg["clip_zero"]:
                vals = vals.clip(lower=0)
            ax_v1v2.plot(df_h.index, vals, label=s["label"],
                         color="#e67e22", lw=0.8, alpha=0.7)
            ax_v1v2.set_ylabel(cfg["ylabel"])
            ax_v1v2.set_title(f"{gas} manufacturer curves (different scale)",
                              fontsize=10, fontstyle="italic")
            ax_v1v2.legend(loc="upper right", fontsize=8, framealpha=0.9)
            ax_v1v2.grid(True, alpha=0.3)
            ax_v1v2.set_ylim(bottom=0)

        # X axis format
        ax_bottom = ax_v1v2 if ax_v1v2 is not None else ax_main
        ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax_bottom.xaxis.set_minor_locator(mdates.DayLocator())
        ax_bottom.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        fname = f"{NODE_ID}_comparacion_2sem_{gas.replace('.', '')}.png"
        fig.savefig(save_dir / fname)
        plt.close(fig)
        log.info("  Plot: %s", fname)


# -------------------------------------------------------------------------
# Markdown report
# -------------------------------------------------------------------------
def generar_reporte(df_met: pd.DataFrame):
    lines = []
    a = lines.append

    a(f"# Comparative evaluation by cost tier: {NODE_ID}\n")
    a(f"> **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"> **Resolution:** 15 min means")
    a(f"> **Metrics:** R2, RMSE (ug/m3), nRMSE (%), MAE, MBE\n")

    a("## Consolidated table\n")
    a("| Gas | Cost tier | Method | Reference | R2 | Pearson r | RMSE | nRMSE% | MAE | MBE | Level | N |")
    a("|-----|---------------|--------|------------|:---:|:---------:|:----:|:------:|:---:|:---:|:-----:|:-:|")
    for _, r in df_met.iterrows():
        a(f"| {r['gas']} | {r['nivel_costo']} | {r['metodo']} | {r['referencia']} | "
          f"{r['r2']:.3f} | {r['pearson_r']:.3f} | {r['rmse']:.2f} | {r['nrmse']:.1f} | "
          f"{r['mae']:.2f} | {r['mbe']:.2f} | **{r['nivel']}** | {r['n']} |")
    a("")

    # Summary per gas
    for gas in df_met["gas"].unique():
        sub = df_met[df_met["gas"] == gas]
        a(f"### {gas}\n")
        ref = sub.iloc[0]["referencia"]
        a(f"Reference: **{ref}**\n")
        for _, r in sub.iterrows():
            a(f"- **{r['nivel_costo']}** ({r['metodo']}): "
              f"R2={r['r2']:.3f}, RMSE={r['rmse']:.2f} ug/m3, "
              f"nRMSE={r['nrmse']:.1f}%, level **{r['nivel']}**")
        a("")

    a("---")
    a(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by s7_evaluate.py*")

    report_path = RESULTS_S7 / "reporte_comparativa_por_costo.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report: %s", report_path)


if __name__ == "__main__":
    run_s7()
