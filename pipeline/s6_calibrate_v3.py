"""
Step s6: calibration by ordinary least squares (co-location)

Reads the s4 ratios, the s3 Alphasense series and the reference station,
fits C_ref = b0 + b1*signal + b2*T + b3*RH per pollutant on 15 min means
and applies it to the full series. Writes the CALIBRATED_OLS CSV,
metricas_v3.csv, coeficientes_v3.csv, config_v3.json, plots and a report.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
try:
    import seaborn as sns
except ImportError:
    sns = None
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import (
    NODE_ID, DATA_S4, DATA_S3, DATA_S6, RESULTS_S6, RESULTS_S6_PLOTS,
    COL_TS, COL_TEMP, COL_RH, COL_PM25, COL_REF_PM25,
    COL_ALPHA_NO2,
    GASES, SENSOR_PARAMS, REF_COLS, POLLUTANTS_OLS, OLS_FEATURES,
    REF_DATA_DIR, REF_CSV_PATTERN, REF_STATION_LABEL, REF_STATION_SHORT,
    IQR_FACTOR, MIN_OBS_OLS,
    R0_PERCENTILE, W_DAYS, R0_NIGHT_END, ALPHA_EWMA,
    R0_MIN_SAMPLES, MAX_CARRY_FORWARD,
    OLS_COL_NAMES, POLLUTANT_DISPLAY_NAMES,
)
from utils import (
    detect_csv, extract_dates_from_csv, build_output_name, nivel_calidad,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("s6")


# -------------------------------------------------------------------------
# Data loading and preparation (15 min means)
# -------------------------------------------------------------------------
def cargar_y_preparar(s4_csv: Path, s3_csv: Path, ref_csv: Path) -> pd.DataFrame:
    """
    Load the s4 ratios, the s3 Alphasense series and the reference station.
    Return one table of 15 min means merged on the timestamp.
    """
    log.info("Loading data...")

    df_s4 = pd.read_csv(s4_csv, parse_dates=[COL_TS])
    df_s4 = df_s4.sort_values(COL_TS).reset_index(drop=True)
    log.info("  s4: %d records", len(df_s4))

    df_s3 = pd.read_csv(s3_csv, parse_dates=[COL_TS])
    df_s3 = df_s3.sort_values(COL_TS).reset_index(drop=True)
    log.info("  s3: %d records", len(df_s3))

    df_ref = pd.read_csv(ref_csv, parse_dates=[COL_TS])
    df_ref = df_ref.sort_values(COL_TS).reset_index(drop=True)
    log.info("  %s: %d records", REF_STATION_SHORT, len(df_ref))

    # s4 as 15 min means
    df_s4["t15"] = df_s4[COL_TS].dt.floor("15min")
    cols_agg = {}
    for gas in GASES:
        cols_agg[f"ratio_{gas}"] = "mean"
    cols_agg[COL_TEMP] = "mean"
    cols_agg[COL_RH] = "mean"
    cols_agg[COL_PM25] = "mean"

    horario = df_s4.groupby("t15").agg(cols_agg).reset_index().rename(columns={"t15": "ts"})

    # T, RH and PM2.5 take the feature names of the model
    horario = horario.rename(columns={COL_TEMP: "T", COL_RH: "RH", COL_PM25: "pm25_raw"})

    # Merge with the Alphasense cells (s3), NO2 included for its own OLS
    df_s3["t15"] = df_s3[COL_TS].dt.floor("15min")
    alpha_cols = [c for c in [REF_COLS["CO"]["col"], REF_COLS["O3"]["col"], COL_ALPHA_NO2]
                  if c in df_s3.columns]
    if alpha_cols:
        alpha_15 = df_s3.groupby("t15").agg({c: "mean" for c in alpha_cols}).reset_index()
        alpha_15 = alpha_15.rename(columns={"t15": "ts"})
        horario = horario.merge(alpha_15, on="ts", how="left")

    # Alphasense NO2 takes the feature name used by the OLS
    if COL_ALPHA_NO2 in horario.columns:
        horario = horario.rename(columns={COL_ALPHA_NO2: "no2_alpha_aan803"})

    # Merge with the reference station (NO2 and PM2.5)
    df_ref["t15"] = df_ref[COL_TS].dt.floor("15min")
    hc_cols = [c for c in [REF_COLS["NO2"]["col"], REF_COLS["PM2.5"]["col"]]
               if c in df_ref.columns]
    if hc_cols:
        hc_15 = df_ref.groupby("t15").agg({c: "mean" for c in hc_cols}).reset_index()
        hc_15 = hc_15.rename(columns={"t15": "ts"})
        horario = horario.merge(hc_15, on="ts", how="left")

    log.info("  15 min periods: %d", len(horario))
    for poll in POLLUTANTS_OLS:
        ref_col = REF_COLS[poll]["col"]
        if ref_col in horario.columns:
            n_valid = horario[ref_col].notna().sum()
            log.info("  Ref %-5s: %d valid", poll, n_valid)

    return horario


# -------------------------------------------------------------------------
# OLS calibration
# -------------------------------------------------------------------------
def calibrar_ols(horario: pd.DataFrame, gas: str) -> dict:
    """Fit the OLS model C_ref = b0 + b1*signal + b2*T + b3*RH for one pollutant."""
    features = OLS_FEATURES[gas]
    col_senal = features[0]
    col_ref = REF_COLS[gas]["col"]

    if col_ref not in horario.columns:
        log.warning("  %s: reference column %s not found", gas, col_ref)
        return None
    # Periods with the reference and every feature present
    mask = horario[col_ref].notna()
    for f in features:
        mask &= horario[f].notna()

    datos = horario[mask].copy().sort_values("ts").reset_index(drop=True)
    if len(datos) < MIN_OBS_OLS:
        log.warning("  %s: only %d observations, not enough", gas, len(datos))
        return None

    # IQR filter on the sensor signal (applied to the fit only)
    q1 = datos[col_senal].quantile(0.25)
    q3 = datos[col_senal].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_FACTOR * iqr
    upper = q3 + IQR_FACTOR * iqr
    mask_iqr = datos[col_senal].between(lower, upper)
    n_exc = (~mask_iqr).sum()
    if n_exc > 0:
        log.info("  %-3s: IQR filter excludes %d/%d obs", gas, n_exc, len(datos))
    datos_fit = datos[mask_iqr].reset_index(drop=True)

    X_fit = datos_fit[features].values
    y_fit = datos_fit[col_ref].values

    modelo = LinearRegression()
    modelo.fit(X_fit, y_fit)

    coefs = {"intercept": modelo.intercept_}
    for i, f in enumerate(features):
        coefs[f] = modelo.coef_[i]

    log.info("  %-3s: OLS fit with %d obs (IQR-filtered)", gas, len(datos_fit))

    # Metrics over all the data, not only the IQR-filtered subset
    X_all = datos[features].values
    y_all = datos[col_ref].values
    y_pred_all = modelo.predict(X_all)

    r2 = r2_score(y_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_all, y_pred_all))
    mae = mean_absolute_error(y_all, y_pred_all)
    mbe = float(np.mean(y_pred_all - y_all))
    mean_ref = float(np.mean(y_all))
    nrmse = (rmse / mean_ref * 100) if mean_ref > 0 else np.nan

    log.info("  %-3s: R2=%.3f, RMSE=%.2f, nRMSE=%.1f%%, MAE=%.2f, MBE=%.2f [%s] (N=%d)",
             gas, r2, rmse, nrmse, mae, mbe, nivel_calidad(r2, nrmse), len(datos))

    return {
        "gas": gas, "modelo": modelo, "features": features,
        "signal_col": col_senal, "coefs": coefs,
        "n_ajuste": len(datos_fit), "n_eval": len(datos),
        "r2": r2, "rmse": rmse, "mae": mae,
        "mbe": mbe, "nrmse": nrmse,
        "ajuste_ts": (datos["ts"].iloc[0], datos["ts"].iloc[-1]),
        "y_obs": y_all, "y_pred": y_pred_all, "ts_ajuste": datos["ts"].values,
    }


def diagnostico_mensual(horario: pd.DataFrame, resultados: dict) -> dict:
    """Compute R2, Pearson r and RMSE of each model per calendar month."""
    diag = {}
    for gas in resultados:
        res = resultados[gas]
        modelo = res["modelo"]
        features = res["features"]
        col_ref = REF_COLS[gas]["col"]

        mask = horario[col_ref].notna()
        for f in features:
            mask &= horario[f].notna()
        sub = horario[mask].copy()
        if sub.empty:
            continue

        y_pred = modelo.predict(sub[features].values)
        sub = sub.assign(y_pred=y_pred, y_ref=sub[col_ref].values)
        sub["mes"] = sub["ts"].dt.strftime("%Y-%m")

        filas = []
        for mes, grupo in sub.groupby("mes"):
            # Months with fewer than 10 periods are skipped
            if len(grupo) < 10:
                continue
            r2 = float(r2_score(grupo["y_ref"], grupo["y_pred"]))
            corr = float(np.corrcoef(grupo["y_ref"], grupo["y_pred"])[0, 1])
            rmse = float(np.sqrt(mean_squared_error(grupo["y_ref"], grupo["y_pred"])))
            filas.append({"mes": mes, "r2": round(r2, 3), "r": round(corr, 3),
                          "rmse": round(rmse, 2), "n": len(grupo)})
        if filas:
            diag[gas] = pd.DataFrame(filas).sort_values("mes").reset_index(drop=True)
    return diag


# -------------------------------------------------------------------------
# Model applied to the full dataset
# -------------------------------------------------------------------------
def aplicar_modelo_completo(s4_csv: Path, s3_csv: Path, resultados: dict) -> pd.DataFrame:
    """Apply the OLS models at the original resolution (about 30 s)."""
    log.info("Applying the model to the full dataset...")
    df = pd.read_csv(s4_csv, parse_dates=[COL_TS])

    # The Alphasense NO2 column comes from s3
    df_s3 = pd.read_csv(s3_csv, parse_dates=[COL_TS])
    if COL_ALPHA_NO2 in df_s3.columns:
        df = df.merge(df_s3[[COL_TS, COL_ALPHA_NO2]], on=COL_TS, how="left")

    final = pd.DataFrame({COL_TS: df[COL_TS]})
    final["T"] = df[COL_TEMP].values
    final["RH"] = df[COL_RH].values

    for gas in POLLUTANTS_OLS:
        if gas not in resultados:
            log.warning("  %s: no model, skipping", gas)
            continue

        res = resultados[gas]
        modelo = res["modelo"]
        features = OLS_FEATURES[gas]

        # Feature matrix built from the original CSV columns
        feature_cols = []
        for f in features:
            if f == "T":
                feature_cols.append(df[COL_TEMP].values)
            elif f == "RH":
                feature_cols.append(df[COL_RH].values)
            elif f == "pm25_raw":
                feature_cols.append(df[COL_PM25].values)
            elif f == "no2_alpha_aan803":
                feature_cols.append(df[COL_ALPHA_NO2].values)
            else:  # ratio_CO, ratio_NO2, ratio_O3
                feature_cols.append(df[f].values)

        X = np.column_stack(feature_cols)
        mascara = np.all(np.isfinite(X), axis=1)

        estimado = np.full(len(df), np.nan)
        estimado[mascara] = modelo.predict(X[mascara])

        # Negative estimates are counted and clipped to zero
        n_negative = int(np.sum(estimado[mascara] < 0))
        estimado = np.maximum(estimado, 0)

        col_name = OLS_COL_NAMES[gas]
        final[col_name] = estimado

        n_valid = np.isfinite(estimado).sum()
        log.info("  %-10s: %d valid (%.1f%%), mean=%.1f, median=%.1f, clipped_zero=%d",
                 POLLUTANT_DISPLAY_NAMES.get(gas, gas), n_valid, 100 * n_valid / len(df),
                 np.nanmean(estimado), np.nanmedian(estimado), n_negative)

    return final


# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------
def configurar_estilo():
    if sns is not None:
        sns.set_theme(style="whitegrid", font_scale=1.1)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})


# Label of the reference or comparator of each pollutant
REF_LABELS = {
    "NO2":       REF_STATION_LABEL,
    "CO":        "Alphasense CO-B4 (factory cal.)",
    "O3":        "Alphasense OX-B431 (factory cal.)",
    "PM2.5":     REF_STATION_LABEL,
    "NO2_alpha": REF_STATION_LABEL,
}


def plot_scatter(resultados: dict, save_dir: Path):
    for gas in POLLUTANTS_OLS:
        if gas not in resultados:
            continue
        res = resultados[gas]
        display = POLLUTANT_DISPLAY_NAMES.get(gas, gas)
        ref_lbl = REF_LABELS.get(gas, "Reference")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(res["y_obs"], res["y_pred"], s=3, alpha=0.3, color="steelblue")
        lims = [min(res["y_obs"].min(), res["y_pred"].min()),
                max(res["y_obs"].max(), res["y_pred"].max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="1:1")
        ax.set_xlabel(f"{ref_lbl} (ug/m3)")
        ax.set_ylabel("OLS calibration (ug/m3)")
        ax.set_title(f"{NODE_ID} {display}: R2={res['r2']:.3f}, RMSE={res['rmse']:.1f}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(save_dir / f"{NODE_ID}_scatter_{gas}_v3.png")
        plt.close(fig)


def plot_timeseries(resultados: dict, save_dir: Path):
    for gas in POLLUTANTS_OLS:
        if gas not in resultados:
            continue
        res = resultados[gas]
        display = POLLUTANT_DISPLAY_NAMES.get(gas, gas)
        fig, ax = plt.subplots(figsize=(16, 5))
        ref_lbl = REF_LABELS.get(gas, "Reference")
        ax.plot(res["ts_ajuste"], res["y_obs"], lw=0.5, alpha=0.7, color="blue", label=ref_lbl)
        ax.plot(res["ts_ajuste"], res["y_pred"], lw=0.5, alpha=0.7, color="red", label="OLS calibration")
        ax.set_ylabel(f"{display} (ug/m3)")
        ax.set_title(f"{NODE_ID} {display}: OLS fit (R2={res['r2']:.3f})", fontweight="bold")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        fig.savefig(save_dir / f"{NODE_ID}_ts_{gas}_v3.png")
        plt.close(fig)


def plot_residuos(resultados: dict, save_dir: Path):
    """Draw four residual diagnostics: vs prediction, histogram, diurnal bias and Q-Q."""
    for gas in POLLUTANTS_OLS:
        if gas not in resultados:
            continue
        res = resultados[gas]
        residuos = res["y_pred"] - res["y_obs"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        display = POLLUTANT_DISPLAY_NAMES.get(gas, gas)
        fig.suptitle(f"{NODE_ID} {display}: residual diagnostics", fontsize=14, fontweight="bold")

        # Residuals vs prediction
        ax = axes[0, 0]
        ax.scatter(res["y_pred"], residuos, s=2, alpha=0.3)
        ax.axhline(0, color="red", ls="--")
        ax.set_xlabel("OLS calibration (ug/m3)")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs OLS calibration")

        # Histogram
        ax = axes[0, 1]
        ax.hist(residuos, bins=50, density=True, alpha=0.6, color="steelblue")
        ax.set_xlabel("Residual")
        ax.set_title(f"Distribution (MBE={res['mbe']:.2f})")

        # Mean residual by hour of the day
        ax = axes[1, 0]
        horas = pd.DatetimeIndex(res["ts_ajuste"]).hour
        df_r = pd.DataFrame({"hour": horas, "resid": residuos})
        media = df_r.groupby("hour")["resid"].mean()
        ax.bar(media.index, media.values, color="steelblue", alpha=0.6)
        ax.axhline(0, color="red", ls="--")
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel("Mean residual")
        ax.set_title("Diurnal bias")

        # Normal Q-Q plot of the standardised residuals
        ax = axes[1, 1]
        r_sorted = np.sort(residuos)
        n = len(r_sorted)
        q_teoricos = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        r_std = (r_sorted - np.mean(r_sorted)) / np.std(r_sorted)
        ax.scatter(q_teoricos, r_std, s=2, alpha=0.3)
        lim = max(abs(q_teoricos[0]), abs(q_teoricos[-1]), 3)
        ax.plot([-lim, lim], [-lim, lim], "r--")
        ax.set_xlabel("Theoretical quantile N(0,1)")
        ax.set_ylabel("Observed quantile (standardised)")
        ax.set_title("Q-Q plot")

        plt.tight_layout()
        fig.savefig(save_dir / f"{NODE_ID}_residuos_{gas}_v3.png")
        plt.close(fig)


def plot_serie_completa(final_df: pd.DataFrame, save_dir: Path):
    """Plot the hourly means of every calibrated series over the full period."""
    colores = {"CO": "#e74c3c", "NO2": "#3498db", "O3": "#27ae60", "PM2.5": "#8e44ad",
               "NO2_alpha": "#e67e22"}
    polls_ok = [p for p in POLLUTANTS_OLS if OLS_COL_NAMES[p] in final_df.columns]
    if not polls_ok:
        return
    fig, axes = plt.subplots(len(polls_ok), 1, figsize=(16, 4 * len(polls_ok)), sharex=True)
    if len(polls_ok) == 1:
        axes = [axes]
    fig.suptitle(f"{NODE_ID}: calibrated concentrations (OLS)", fontsize=14, fontweight="bold")

    for i, gas in enumerate(polls_ok):
        ax = axes[i]
        col = OLS_COL_NAMES[gas]
        ts_data = (
            pd.DataFrame({"ts": final_df[COL_TS], "val": final_df[col]})
            .set_index("ts").resample("1h").mean()
        )
        display = POLLUTANT_DISPLAY_NAMES.get(gas, gas)
        ax.plot(ts_data.index, ts_data["val"], lw=0.3, alpha=0.6, color=colores.get(gas, "steelblue"))
        ax.set_ylabel(f"{display} (ug/m3)")
        vals = final_df[col].dropna()
        ax.set_title(f"{display}: mean={vals.mean():.1f}, median={vals.median():.1f}")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(save_dir / f"{NODE_ID}_series_completa_v3.png")
    plt.close(fig)


def plot_r2_mensual(diag: dict, save_dir: Path):
    gases_ok = [g for g in POLLUTANTS_OLS if g in diag]
    if not gases_ok:
        return
    colores = {"CO": "#E07B54", "NO2": "#5B8DB8", "O3": "#6BAE75", "PM2.5": "#9B59B6",
               "NO2_alpha": "#e67e22"}
    fig, axes = plt.subplots(len(gases_ok), 1, figsize=(12, 3.8 * len(gases_ok)), squeeze=False)
    axes = axes.flatten()

    for ax, gas in zip(axes, gases_ok):
        df_m = diag[gas]
        x = np.arange(len(df_m))
        bars = ax.bar(x, df_m["r2"], color=colores.get(gas, "steelblue"),
                      alpha=0.75, edgecolor="white", linewidth=0.5)
        # R2 thresholds of the quality levels
        ax.axhline(0.70, color="#2ca02c", ls="--", lw=1.2, label="Excellent R2>=0.70")
        ax.axhline(0.50, color="#ff7f0e", ls="--", lw=1.2, label="Acceptable R2>=0.50")
        ax.axhline(0.30, color="#d62728", ls="--", lw=1.2, label="Marginal R2>=0.30")
        ax.axhline(0.0, color="black", ls="-", lw=0.8)
        for bar, n in zip(bars, df_m["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.03,
                    f"N={n}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(df_m["mes"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("R2")
        ax.set_ylim(-1.1, 1.1)
        display = POLLUTANT_DISPLAY_NAMES.get(gas, gas)
        ax.set_title(f"{display}: monthly R2")
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    fig.savefig(save_dir / "r2_mensual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Markdown report
# -------------------------------------------------------------------------
def generar_reporte(resultados: dict, final_df: pd.DataFrame,
                    diag_mensual: dict, output_dir: Path):
    lines = []
    a = lines.append

    a(f"# OLS calibration report: {NODE_ID} AQMS QMUL\n")
    a(f"> **Method:** OLS linear regression by co-location (CEN/TS 17660-1)")
    a(f"> **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    a("## 1. References\n")
    a("| Pollutant | Reference | Instrument | Type |")
    a("|-------------|-----------|-------------|------|")
    a(f"| NO2 (low-cost) | {REF_STATION_LABEL} | Chemiluminescence analyser | High-cost |")
    a("| CO | Alphasense CO-B4 | Electrochemical | Mid-cost |")
    a("| O3 | Alphasense OX-B431 | Electrochemical | Mid-cost |")
    a(f"| PM2.5 | {REF_STATION_LABEL} | Met One BAM 1020 | High-cost |")
    a(f"| NO2 (Alphasense) | {REF_STATION_LABEL} | Chemiluminescence analyser | High-cost |\n")

    a("## 2. Metrics\n")
    a("| Pollutant | R2 | RMSE | nRMSE% | MAE | MBE | Level | N |")
    a("|-------------|-----|------|--------|-----|-----|-------|---|")
    for gas in POLLUTANTS_OLS:
        if gas in resultados:
            r = resultados[gas]
            display = POLLUTANT_DISPLAY_NAMES.get(gas, gas)
            nrmse = r["nrmse"]
            a(f"| {display} | {r['r2']:.3f} | {r['rmse']:.2f} | "
              f"{nrmse:.1f} | {r['mae']:.2f} | {r['mbe']:.2f} | "
              f"**{nivel_calidad(r['r2'], nrmse)}** | {r['n_eval']} (fit: {r['n_ajuste']}) |")
    a("")

    a("## 3. Coefficients\n")
    a("| Pollutant | b0 | b1 (signal) | b2 (T) | b3 (RH) | Signal |")
    a("|-------------|-----|------------|--------|---------|-------|")
    for gas in POLLUTANTS_OLS:
        if gas in resultados:
            display = POLLUTANT_DISPLAY_NAMES.get(gas, gas)
            c = resultados[gas]["coefs"]
            sig = resultados[gas]["signal_col"]
            a(f"| {display} | {c['intercept']:.4f} | {c[sig]:.4f} | {c['T']:.4f} | {c['RH']:.4f} | {sig} |")
    a("")

    a("## 4. Monthly diagnostic\n")
    for gas in POLLUTANTS_OLS:
        if gas not in diag_mensual:
            continue
        df_m = diag_mensual[gas]
        display = POLLUTANT_DISPLAY_NAMES.get(gas, gas)
        a(f"**{display}:**\n")
        a("| Month | R2 | r | RMSE | N |")
        a("|-----|-----|---|------|---|")
        for _, row in df_m.iterrows():
            a(f"| {row['mes']} | {row['r2']:.3f} | {row['r']:.3f} | {row['rmse']:.2f} | {int(row['n'])} |")
        a("")

    a("## 5. Concentrations (full dataset)\n")
    a("| Pollutant | N | Mean | Median | P95 | Max |")
    a("|-------------|---|-------|---------|-----|-----|")
    for gas in POLLUTANTS_OLS:
        col = OLS_COL_NAMES[gas]
        if col in final_df.columns:
            display = POLLUTANT_DISPLAY_NAMES.get(gas, gas)
            v = final_df[col].dropna()
            a(f"| {display} | {len(v):,} | {v.mean():.1f} | {v.median():.1f} | "
              f"{v.quantile(0.95):.1f} | {v.max():.1f} |")
    a("")

    a("---")
    a(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by s6_calibrate_v3.py*")

    (output_dir / "reporte_calibracion_v3.md").write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------------
# Export (calibrated CSV, metrics, coefficients and configuration)
# -------------------------------------------------------------------------
def exportar(resultados: dict, final_df: pd.DataFrame, diag_mensual: dict,
             s4_csv: Path):
    log.info("Exporting...")

    # Calibrated series
    DATA_S6.mkdir(parents=True, exist_ok=True)
    start_date, end_date = extract_dates_from_csv(final_df)
    out_name = build_output_name(NODE_ID, "CALIBRATED_OLS", start_date, end_date)
    out_csv = DATA_S6 / out_name

    conc_out = final_df.copy()
    conc_out[COL_TS] = conc_out[COL_TS].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    conc_out.to_csv(out_csv, index=False)
    log.info("  Final CSV: %s", out_csv.name)

    # Metrics
    RESULTS_S6.mkdir(parents=True, exist_ok=True)
    filas = []
    for gas in POLLUTANTS_OLS:
        if gas not in resultados:
            continue
        r = resultados[gas]
        filas.append({
            "gas": gas, "r2": r["r2"], "rmse": r["rmse"],
            "nrmse_pct": r["nrmse"], "mae": r["mae"],
            "mbe": r["mbe"],
            "nivel": nivel_calidad(r["r2"], r["nrmse"]),
            "n_eval": r["n_eval"], "n_ajuste": r["n_ajuste"],
        })
    pd.DataFrame(filas).to_csv(RESULTS_S6 / "metricas_v3.csv", index=False)

    # Coefficients
    filas_c = []
    for gas in POLLUTANTS_OLS:
        if gas not in resultados:
            continue
        fila = {"gas": gas}
        fila.update(resultados[gas]["coefs"])
        filas_c.append(fila)
    pd.DataFrame(filas_c).to_csv(RESULTS_S6 / "coeficientes_v3.csv", index=False)

    # Run configuration (JSON)
    config = {
        "version": "v3", "nodo": NODE_ID,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "modelo": "OLS: C = b0 + b1*señal + b2*T + b3*RH",
        "r0": {
            "percentil": R0_PERCENTILE, "ventana_dias": W_DAYS,
            "nocturno_0_a": R0_NIGHT_END, "ewma_alpha": ALPHA_EWMA,
            "min_samples": R0_MIN_SAMPLES, "max_carry_forward": MAX_CARRY_FORWARD,
        },
        "referencias": {gas: REF_COLS[gas] for gas in POLLUTANTS_OLS},
    }
    with open(RESULTS_S6 / "config_v3.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return out_csv


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------
def run_s6(input_s4_csv: Path = None, input_s3_csv: Path = None) -> Path:
    if input_s4_csv is None:
        input_s4_csv = detect_csv(DATA_S4, f"{NODE_ID}_PREPROCESS_wide_data_*.csv")
    if input_s3_csv is None:
        input_s3_csv = detect_csv(DATA_S3, f"{NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")

    if input_s4_csv is None or not input_s4_csv.exists():
        raise FileNotFoundError(f"s4 CSV not found in {DATA_S4}")
    if input_s3_csv is None or not input_s3_csv.exists():
        raise FileNotFoundError(f"s3 CSV not found in {DATA_S3}")

    ref_csv = detect_csv(REF_DATA_DIR, REF_CSV_PATTERN)
    if ref_csv is None or not ref_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found in {REF_DATA_DIR}")

    print(f"s6: OLS calibration, s4={input_s4_csv.name}, s3={input_s3_csv.name}")

    horario = cargar_y_preparar(input_s4_csv, input_s3_csv, ref_csv)

    resultados = {}
    for gas in POLLUTANTS_OLS:
        log.info("=== %s ===", gas)
        r = calibrar_ols(horario, gas)
        if r:
            resultados[gas] = r

    diag = diagnostico_mensual(horario, resultados)

    final_df = aplicar_modelo_completo(input_s4_csv, input_s3_csv, resultados)

    out_csv = exportar(resultados, final_df, diag, input_s4_csv)

    RESULTS_S6_PLOTS.mkdir(parents=True, exist_ok=True)
    configurar_estilo()
    plot_scatter(resultados, RESULTS_S6_PLOTS)
    plot_timeseries(resultados, RESULTS_S6_PLOTS)
    plot_residuos(resultados, RESULTS_S6_PLOTS)
    plot_serie_completa(final_df, RESULTS_S6_PLOTS)
    plot_r2_mensual(diag, RESULTS_S6_PLOTS)
    log.info("  %d plots generated", len(list(RESULTS_S6_PLOTS.glob("*.png"))))

    generar_reporte(resultados, final_df, diag, RESULTS_S6)

    print("\n" + "=" * 70)
    print(f"  OLS SUMMARY: {NODE_ID}")
    print("=" * 70)
    for gas in POLLUTANTS_OLS:
        if gas in resultados:
            r = resultados[gas]
            display = POLLUTANT_DISPLAY_NAMES.get(gas, gas)
            print(f"  {display:<20} R2={r['r2']:.3f} RMSE={r['rmse']:.2f} "
                  f"nRMSE={r['nrmse']:.1f}% N={r['n_eval']} (fit: {r['n_ajuste']})")
    print(f"\n  CSV: {out_csv}")
    print("=" * 70)

    return out_csv


if __name__ == "__main__":
    run_s6()
