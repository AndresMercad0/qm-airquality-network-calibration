"""
Shared helpers of the pipeline

Quality control filters that return boolean masks (Hampel, limits, stuck
sensor, saturation, thermal instability), descriptive statistics and
quality tiers, detection and naming of the CSV files, and the features,
metrics and column names used by the machine learning code.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Quality control filters (each returns a mask that is True where flagged)
# -------------------------------------------------------------------------
def hampel_filter(series: pd.Series, h: int = 20, k: float = 4.0) -> pd.Series:
    """Flag spikes with a Hampel filter over a centred window of 2h+1 samples."""
    window = 2 * h + 1
    med = series.rolling(window=window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window=window, center=True, min_periods=1).median()
    threshold = k * 1.4826 * mad
    return ((series - med).abs() > threshold) & (~series.isna())


def flag_bounds(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((series < lo) | (series > hi)) & (~series.isna())


def flag_stuck(series: pd.Series, min_repeat: int = 10):
    """
    Flag runs of at least min_repeat identical consecutive values (NaN ignored).
    Return the mask, the number of runs and the length of the longest run.
    """
    s = series.copy()
    changed = (s != s.shift()) | (s.isna() ^ s.shift().isna())
    group_id = changed.cumsum()
    group_sizes = group_id.map(group_id.value_counts())
    mask = (group_sizes >= min_repeat) & (~s.isna())
    n_events = int(group_id[mask].nunique()) if mask.any() else 0
    max_run = int(group_sizes[mask].max()) if mask.any() else 0
    return mask, n_events, max_run


def flag_adc_saturation(series: pd.Series, sat_min: int = 0,
                        sat_max: int = 1023) -> pd.Series:
    return (series == sat_min) | (series == sat_max)


def flag_voltage_saturation(series: pd.Series, lo: float = 0.0,
                            hi: float = 4.95) -> pd.Series:
    """Flag voltages at or beyond the limits of the ADS1115 range."""
    return (series <= lo) | (series >= hi)


def flag_thermal_instability(temp: pd.Series, delta_window: int = 60,
                             delta_thresh: float = 5.0,
                             recovery_win: int = 120) -> pd.Series:
    """
    Flag thermal transients, which destabilise the Alphasense signal (AAN 803-05).
    An event is a temperature range above delta_thresh within delta_window samples.
    The event and the recovery_win samples that follow are flagged.
    """
    temp_max = temp.rolling(window=delta_window, min_periods=1).max()
    temp_min = temp.rolling(window=delta_window, min_periods=1).min()
    delta_t = temp_max - temp_min
    event = (delta_t > delta_thresh).astype(float)
    flagged = event.rolling(window=recovery_win, min_periods=1).max().astype(bool)
    return flagged


def count_thermal_events(thermal_mask: pd.Series) -> int:
    """Count the distinct thermal events (False to True transitions)."""
    transitions = thermal_mask & ~thermal_mask.shift(fill_value=False)
    return int(transitions.sum())


# -------------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------------
PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
PCT_LABELS = ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]


def compute_stats(series: pd.Series, prefix: str) -> dict:
    """Return the descriptive statistics and percentiles of the series (NaN ignored)."""
    s = series.dropna()
    result = {
        f"{prefix}_min":  round(float(s.min()), 6) if len(s) else None,
        f"{prefix}_max":  round(float(s.max()), 6) if len(s) else None,
        f"{prefix}_mean": round(float(s.mean()), 6) if len(s) else None,
        f"{prefix}_std":  round(float(s.std()), 6) if len(s) else None,
    }
    for pct, label in zip(PERCENTILES, PCT_LABELS):
        result[f"{prefix}_{label}"] = round(float(s.quantile(pct)), 6) if len(s) else None
    return result


def first_last_ts(df: pd.DataFrame, mask: pd.Series, ts_col: str = "timestamp_utc"):
    """Return the first and last timestamp of the flagged rows, or None if there are none."""
    if not mask.any():
        return None, None
    ts = df.loc[mask, ts_col]
    return str(ts.min()), str(ts.max())


def nivel_calidad(r2: float, nrmse: float) -> str:
    """Return the quality tier of a calibration from R2 and nRMSE (%)."""
    if np.isnan(r2) or np.isnan(nrmse):
        return "Inaceptable"
    if r2 >= 0.70 and nrmse < 25.0:
        return "Excelente"
    if r2 >= 0.50 and nrmse < 50.0:
        return "Aceptable"
    if r2 >= 0.30 and nrmse < 75.0:
        return "Marginal"
    return "Inaceptable"


# -------------------------------------------------------------------------
# CSV detection and naming
# -------------------------------------------------------------------------
def detect_csv(directory: Path, pattern: str = "*.csv") -> Path:
    """
    Return the CSV of the directory that matches the pattern, or None.
    More than one match raises an error.
    """
    directory = Path(directory)
    matches = sorted(directory.glob(pattern))
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise RuntimeError(f"Ambiguous input in {directory}: {pattern} matches {names}")
    return matches[0] if matches else None


def extract_dates_from_csv(df: pd.DataFrame,
                           ts_col: str = "timestamp_utc") -> tuple:
    ts = pd.to_datetime(df[ts_col])
    start = ts.min().strftime("%Y-%m-%d")
    end = ts.max().strftime("%Y-%m-%d")
    return start, end


def build_output_name(node_id: str, descriptor: str,
                      start_date: str, end_date: str) -> str:
    return f"{node_id}_{descriptor}_wide_data_{start_date}_to_{end_date}.csv"


# -------------------------------------------------------------------------
# Shared machine learning helpers (s9)
# -------------------------------------------------------------------------
def compute_abs_humidity(temp_c, rh_pct):
    """Compute the absolute humidity (g/m3) from temperature (C) and relative humidity (%)."""
    temp_c = np.asarray(temp_c, dtype=float)
    rh_pct = np.asarray(rh_pct, dtype=float)
    # Saturation vapour pressure, Magnus formula (hPa)
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    ah = (216.7 * es * rh_pct / 100.0) / (273.15 + temp_c)
    return ah


def add_temporal_features(df, ts_col="timestamp_utc"):
    """Add the cyclic hour and day-of-week features to a copy of the DataFrame."""
    ts = pd.to_datetime(df[ts_col])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def compute_ml_metrics(y_true, y_pred):
    """Compute R2, RMSE, MAE, MBE, nRMSE (%) and Pearson r over the finite pairs."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 10:
        return {k: np.nan for k in ["R2", "RMSE", "MAE", "MBE", "nRMSE", "pearson_r"]}
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mbe = float(np.mean(yp - yt))
    y_mean = float(yt.mean())
    nrmse = (rmse / y_mean * 100.0) if y_mean > 0 else np.nan
    pearson_r = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else np.nan
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MBE": mbe,
            "nRMSE": nrmse, "pearson_r": pearson_r}


def nivel_calidad_ml(r2, nrmse):
    """Return the quality tier of an ML model, with stricter thresholds than nivel_calidad."""
    if np.isnan(r2) or np.isnan(nrmse):
        return "Inaceptable"
    if r2 >= 0.80 and nrmse < 20.0:
        return "Excelente"
    if r2 >= 0.60 and nrmse < 30.0:
        return "Aceptable"
    if r2 >= 0.40 and nrmse < 50.0:
        return "Marginal"
    return "Inaceptable"


def build_ml_col_name(gas, model, config):
    """Build the name of an ML column, for example no2_low_mgsv2_rf_configA_ug_m3."""
    from config import ML_COL_PREFIXES
    prefix = ML_COL_PREFIXES[gas]
    model_lower = model.lower().replace("lightgbm", "lgbm")
    return f"{prefix}_{model_lower}_config{config}_ug_m3"
