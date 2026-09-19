"""
Shared configuration of the calibration pipeline

Resolves the node (AQMS_NODE or --node) and loads its reference station
and Alphasense calibration from config_node3.py or config_node5.py.
Defines the paths, the study window, the column names, the sensor
constants and manufacturer tables, and the QC, OLS and ML settings.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np

# -------------------------------------------------------------------------
# Node selection (AQMS_NODE environment variable or --node on the command line)
# -------------------------------------------------------------------------
VALID_NODES = ("node_3", "node_5")


def _normalise_node(value):
    value = str(value).strip().lower()
    if value in ("3", "5"):
        value = f"node_{value}"
    if value not in VALID_NODES:
        raise ValueError(f"Unknown node '{value}': expected 3 or 5")
    return value


def _node_from_argv(argv):
    for i, arg in enumerate(argv):
        if arg == "--node" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--node="):
            return arg.split("=", 1)[1]
    return None


def _resolve_node():
    from_env = os.environ.get("AQMS_NODE")
    from_argv = _node_from_argv(sys.argv[1:])
    if from_env and from_argv and _normalise_node(from_env) != _normalise_node(from_argv):
        raise RuntimeError(f"AQMS_NODE={from_env} conflicts with --node {from_argv}")
    chosen = from_env or from_argv
    if chosen is None:
        raise RuntimeError("No node selected: pass --node 3|5 or set AQMS_NODE")
    return _normalise_node(chosen)


NODE_ID = _resolve_node()
os.environ["AQMS_NODE"] = NODE_ID
NODE_LABEL = "Node " + NODE_ID.split("_")[1]

_node = importlib.import_module("config_" + NODE_ID.replace("_", ""))
_partner = importlib.import_module("config_" + _node.PARTNER_NODE_ID.replace("_", ""))

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "pipeline"

DATA_DIR = PROJECT_ROOT / "data" / NODE_ID
RESULTS_DIR = PROJECT_ROOT / "results" / NODE_ID

DATA_RAW = DATA_DIR / "raw"
DATA_S1 = DATA_DIR / "s1_qc_lowcost"
DATA_S2 = DATA_DIR / "s2_alphasense_ugm3"
DATA_S3 = DATA_DIR / "s3_qc_alphasense"
DATA_S4 = DATA_DIR / "s4_preprocess"
DATA_S5 = DATA_DIR / "s5_v1v2"
DATA_S6 = DATA_DIR / "s6_v3"
DATA_S8 = DATA_DIR / "s8_final"

RESULTS_S1 = RESULTS_DIR / "s1_qc_lowcost"
RESULTS_S3 = RESULTS_DIR / "s3_qc_alphasense"
RESULTS_S5 = RESULTS_DIR / "s5_v1v2"
RESULTS_S6 = RESULTS_DIR / "s6_v3"
RESULTS_S6_PLOTS = RESULTS_S6 / "plots"

# -------------------------------------------------------------------------
# Input data (clone of qm-airquality-network-data) and study window
# -------------------------------------------------------------------------
DATASET_DIR = Path(os.environ.get("AQMS_DATASET_DIR",
                                  PROJECT_ROOT.parent / "qm-airquality-network-data")) / "data"
STUDY_START_UTC = "2025-04-16T00:00:00Z"    # First timestamp kept (inclusive)
STUDY_STOP_UTC = "2026-02-16T00:00:00Z"     # End of the study window (exclusive)
RAW_CSV_NAME = f"{NODE_ID}_RAW_wide_data_2025-04-16_to_2026-02-15.csv"

# -------------------------------------------------------------------------
# Reference station of this node and partner node for the transfer step
# -------------------------------------------------------------------------
REF_DATA_DIR = PROJECT_ROOT / "data" / "reference_snapshot"
REF_CSV_PATTERN = _node.REF_CSV_PATTERN
REF_CSV_NAME = _node.REF_CSV_NAME
REF_STATION_LABEL = _node.REF_STATION_LABEL
REF_STATION_SHORT = _node.REF_STATION_SHORT

PARTNER_NODE_ID = _node.PARTNER_NODE_ID
PARTNER_DATA_DIR = PROJECT_ROOT / "data" / PARTNER_NODE_ID
PARTNER_REF_CSV_PATTERN = _partner.REF_CSV_PATTERN
PARTNER_REF_STATION_SHORT = _partner.REF_STATION_SHORT

# -------------------------------------------------------------------------
# Column names of the wide CSV files
# -------------------------------------------------------------------------
COL_TS = "timestamp_utc"

# Low-cost
COL_NO2_RAW = "no2_low_mgsv2_raw"
COL_CO_RAW = "co_low_mgsv2_raw"
COL_O3_RAW = "o3_low_mq131_raw"
COL_O3_VOLTAGE = "o3_low_mq131_voltage"
COL_TEMP = "temperature_low_sht45_celsius"
COL_RH = "humidity_low_sht45_percentage"
COL_VOC_RAW = "voc_low_sgp40_comp_raw"
COL_VOC_INDEX = "voc_low_sgp40_index"
COL_PM25 = "pm2_5_low_sps30_ug_m3"
COL_PM10 = "pm10_low_sps30_ug_m3"

# Mid-cost (Alphasense, after the conversion in s2)
COL_ALPHA_NO2 = "no2_mid_alphasense_ug_m3_aan803"
COL_ALPHA_O3 = "o3_mid_alphasense_ug_m3_aan803"
COL_ALPHA_CO = "co_mid_alphasense_ug_m3_aan803"

# High-cost (LAQN reference station of the node)
COL_REF_NO2 = "no2_high_met_one_ug_m3"
COL_REF_PM25 = "pm2_5_high_met_one_ug_m3"

# Gases to calibrate
GASES = ["CO", "NO2", "O3"]

# Pollutants calibrated by OLS in s6 (includes PM2.5 and the Alphasense NO2)
POLLUTANTS_OLS = ["CO", "NO2", "O3", "PM2.5", "NO2_alpha"]

# Reference column of each pollutant (OLS)
REF_COLS = {
    "NO2":       {"source": "ref",   "col": COL_REF_NO2},
    "CO":        {"source": "alpha", "col": COL_ALPHA_CO},
    "O3":        {"source": "alpha", "col": COL_ALPHA_O3},
    "PM2.5":     {"source": "ref",   "col": COL_REF_PM25},
    "NO2_alpha": {"source": "ref",   "col": COL_REF_NO2},
}

# Features of the OLS model of each pollutant
OLS_FEATURES = {
    "CO":        ["ratio_CO", "T", "RH"],
    "NO2":       ["ratio_NO2", "T", "RH"],
    "O3":        ["ratio_O3", "T", "RH"],
    "PM2.5":     ["pm25_raw", "T", "RH"],
    "NO2_alpha": ["no2_alpha_aan803", "T", "RH"],
}

# Names of the OLS-calibrated columns (s6)
OLS_COL_NAMES = {
    "CO":        "co_low_mgsv2_ug_m3_ols",
    "NO2":       "no2_low_mgsv2_ug_m3_ols",
    "O3":        "o3_low_mq131_ug_m3_ols",
    "PM2.5":     "pm2_5_low_sps30_ug_m3_ols",
    "NO2_alpha": "no2_mid_alphasense_ug_m3_ols",
}

# Display names for plots and reports
POLLUTANT_DISPLAY_NAMES = {
    "CO": "CO", "NO2": "NO2 (low-cost)", "O3": "O3",
    "PM2.5": "PM2.5", "NO2_alpha": "NO2 (Alphasense)",
}

# Names of the columns calibrated with the manufacturer curves (s5)
CURVAS_FAB_COL_NAMES = {
    "CO":  "co_low_mgsv2_ug_m3_curvas_fabricante",
    "NO2": "no2_low_mgsv2_ug_m3_curvas_fabricante",
    "O3":  "o3_low_mq131_ug_m3_curvas_fabricante",
}

# Columns of the final assembled CSVs (s8)
_S8_METADATA = [COL_TS, "node_id", "site", "subsite", "exposure", "environment",
                "latitude", "longitude"]
_S8_AMBIENT = [COL_TEMP, COL_RH, COL_PM10, COL_VOC_RAW, COL_VOC_INDEX]

# File 1: manufacturer calibration (20 columns)
S8_COLS_FABRICANTE = (_S8_METADATA + _S8_AMBIENT +
                      [COL_PM25, COL_ALPHA_NO2, COL_ALPHA_O3, COL_ALPHA_CO] +
                      list(CURVAS_FAB_COL_NAMES.values()))

# File 2: OLS calibration (20 columns)
S8_COLS_OLS = (_S8_METADATA + _S8_AMBIENT +
               [COL_ALPHA_O3, COL_ALPHA_CO] +
               list(OLS_COL_NAMES.values()))

# -------------------------------------------------------------------------
# Electrical constants of the low-cost metal oxide sensors
# -------------------------------------------------------------------------
RL_CO = 100_000       # Load resistor, R23 in the Grove v2 schematic (ohm)
RL_NO2 = 47_000       # Load resistor, R12 in the Grove v2 schematic (ohm)
RL_O3 = 1_000         # Load resistor on the MQ131 board (ohm)
VCC_GROVE = 3.3       # AP21726-3.3 regulator (V)
VCC_MQ131 = 5.0       # MQ131 supply (V)
ADC_MAX = 1023        # 10-bit ADC of the STM32F030

# Manufacturer reference conditions
T_REF = 20.0          # Temperature (C)
RH_REF = 55.0         # Relative humidity (%)

# -------------------------------------------------------------------------
# Manufacturer temperature and humidity correction tables
# -------------------------------------------------------------------------
# CO (GM-702B): ratio RS/RSo
TEMP_GRID_CO = np.array([-10, 0, 10, 20, 30, 40, 50], dtype=float)
RH_GRID_CO = np.array([30, 60, 85], dtype=float)
CO_TRH_TABLE = np.array([
    [1.70, 1.48, 1.28],   # -10 C
    [1.57, 1.33, 1.16],   #   0 C
    [1.45, 1.29, 1.09],   #  10 C
    [1.39, 1.09, 0.91],   #  20 C
    [1.13, 0.98, 0.86],   #  30 C
    [1.01, 0.88, 0.73],   #  40 C
    [0.89, 0.73, 0.67],   #  50 C
])

# NO2 (GM-102B): ratio RSo/RS (inverse)
TEMP_GRID_NO2 = np.array([-10, 0, 10, 20, 30, 40, 50], dtype=float)
RH_GRID_NO2 = np.array([30, 60, 85], dtype=float)
NO2_TRH_TABLE = np.array([
    [1.70, 1.48, 1.28],   # -10 C
    [1.57, 1.32, 1.15],   #   0 C
    [1.45, 1.29, 1.09],   #  10 C
    [1.37, 1.09, 0.91],   #  20 C
    [1.13, 0.98, 0.86],   #  30 C
    [1.01, 0.87, 0.71],   #  40 C
    [0.89, 0.73, 0.67],   #  50 C
])

# O3 (MQ131): ratio RS/RSo
TEMP_GRID_O3 = np.array([-10, 0, 10, 20, 30, 40, 50], dtype=float)
RH_GRID_O3 = np.array([20, 40, 55, 85], dtype=float)
O3_TRH_TABLE = np.array([
    [1.83, 1.72, 1.62, 1.57],   # -10 C
    [1.65, 1.55, 1.40, 1.34],   #   0 C
    [1.48, 1.35, 1.18, 1.10],   #  10 C
    [1.33, 1.18, 1.00, 0.95],   #  20 C (exact anchor at 1.00)
    [1.20, 1.03, 0.90, 0.84],   #  30 C
    [1.13, 0.95, 0.81, 0.76],   #  40 C
    [1.06, 0.94, 0.79, 0.72],   #  50 C
])

# -------------------------------------------------------------------------
# Parameters of each low-cost metal oxide sensor
# -------------------------------------------------------------------------
SENSOR_PARAMS = {
    "CO": {
        "col_raw": COL_CO_RAW,
        "rl": RL_CO,
        "vcc": VCC_GROVE,
        "inversion": False,
        "uses_voltage_col": False,
        "trh_table": CO_TRH_TABLE,
        "temp_grid": TEMP_GRID_CO,
        "rh_grid": RH_GRID_CO,
        "trh_type": "RS/RSo",
        "r0_stat": "upper",     # Reducing gas: high RS means clean air (P90)
    },
    "NO2": {
        "col_raw": COL_NO2_RAW,
        "rl": RL_NO2,
        "vcc": VCC_GROVE,
        "inversion": True,      # The firmware reports 1023 minus the ADC reading
        "uses_voltage_col": False,
        "trh_table": NO2_TRH_TABLE,
        "temp_grid": TEMP_GRID_NO2,
        "rh_grid": RH_GRID_NO2,
        "trh_type": "RSo/RS",
        "r0_stat": "lower",     # Oxidising gas: low RS means clean air (P10)
    },
    "O3": {
        "col_raw": None,        # Read as a voltage from the ADS1115
        "rl": RL_O3,
        "vcc": VCC_MQ131,
        "inversion": False,
        "uses_voltage_col": True,
        "col_voltage": COL_O3_VOLTAGE,
        "trh_table": O3_TRH_TABLE,
        "temp_grid": TEMP_GRID_O3,
        "rh_grid": RH_GRID_O3,
        "trh_type": "RS/RSo",
        "r0_stat": "lower",     # Oxidising gas: low RS means clean air (P10)
    },
}

# -------------------------------------------------------------------------
# Dynamic R0 (night-time percentile)
# -------------------------------------------------------------------------
W_DAYS = 14              # Moving window, outdoor (days)
W_DAYS_INDOOR = 21       # Moving window, indoor (days)
ALPHA_EWMA = 0.2         # EWMA smoothing factor
R0_PERCENTILE = 90       # P90 for reducing gases (CO); 100 - P90 = P10 for oxidising gases
R0_NIGHT_END = 5         # Night-time hours: 00:00 to 04:59 UTC
R0_MIN_SAMPLES = 20      # Minimum 15 min periods to compute R0 (5 night-time hours)
MAX_CARRY_FORWARD = 30   # Maximum carry-forward (days)

# -------------------------------------------------------------------------
# Quality control of the low-cost sensors (s1)
# -------------------------------------------------------------------------
ADC_SAT_MIN = 0
ADC_SAT_MAX = 1023
OZ_VOLT_LO = 0.0        # Lower limit of the MQ131 voltage (V)
OZ_VOLT_HI = 4.95       # Upper limit of the MQ131 voltage (V)
STUCK_MIN = 10          # Consecutive identical readings
HAMPEL_H = 20           # Half-width of the window (samples)
HAMPEL_K = 4.0          # MAD scale factor
PM25_MAX = 1000.0       # Upper limit (ug/m3)
PM10_MAX = 1000.0       # Upper limit (ug/m3)
TEMP_LO = -10.0         # Lower limit (C)
TEMP_HI = 70.0          # Upper limit (C)
RH_LO = 0.0             # Lower limit (%)
RH_HI = 100.0           # Upper limit (%)

# ADC and voltage columns filtered in s1
ADC_COLS = [COL_CO_RAW, COL_NO2_RAW]
VOLT_COLS = [COL_O3_VOLTAGE]

PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
PCT_LABELS = ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]

# -------------------------------------------------------------------------
# Quality control of the Alphasense cells (s3)
# -------------------------------------------------------------------------
ALPHA_HAMPEL_H = 20     # Half-width of the window (samples)
ALPHA_HAMPEL_K = 4.0    # MAD scale factor (relaxed for micro-plumes)

ALPHA_BOUNDS = {
    COL_ALPHA_NO2: (-50.0,   1000.0),
    COL_ALPHA_O3:  (-50.0,    500.0),
    COL_ALPHA_CO:  (-1000.0, 20000.0),
}

# Thermal transients destabilise the cell signal (AAN 803-05); pipeline thresholds
TEMP_DELTA_WINDOW = 60    # Window of the temperature range (samples, about 30 min at 30 s)
TEMP_DELTA_THRESH = 5.0   # Temperature range that flags an event (C)
TEMP_RECOVERY_WIN = 120   # Recovery window after an event (samples, about 1 h)

# -------------------------------------------------------------------------
# Alphasense conversion from voltage to ug/m3 (s2)
# -------------------------------------------------------------------------
ALPHASENSE_CALIBRATION = _node.ALPHASENSE_CALIBRATION

# nT(T) factors of Algorithm 1 (AAN 803-05, Table 3)
NT_TEMPS = np.array([-30, -20, -10, 0, 10, 20, 30, 40, 50], dtype=float)
NT_VALUES = {
    "no2": np.array([1.3, 1.3, 1.3, 1.3, 1.0, 0.6, 0.4, 0.2, -1.5]),   # NO2-B43F
    "o3":  np.array([0.9, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.7]),    # OX-B431
    "co":  np.array([0.7, 0.7, 0.7, 0.7, 1.0, 3.0, 3.5, 4.0, 4.5]),    # CO-B4
}

# Alphasense columns of the raw CSV
ALPHASENSE_RAW_COLS = [
    "no2_mid_alphasense_op1_raw", "no2_mid_alphasense_op1_voltage",
    "no2_mid_alphasense_op2_raw", "no2_mid_alphasense_op2_voltage",
    "o3_mid_alphasense_op1_raw",  "o3_mid_alphasense_op1_voltage",
    "o3_mid_alphasense_op2_raw",  "o3_mid_alphasense_op2_voltage",
    "co_mid_alphasense_op1_raw",  "co_mid_alphasense_op1_voltage",
    "co_mid_alphasense_op2_raw",  "co_mid_alphasense_op2_voltage",
]

ALPHASENSE_VOLTAGE_COLS = {
    "no2": ("no2_mid_alphasense_op1_voltage", "no2_mid_alphasense_op2_voltage"),
    "o3":  ("o3_mid_alphasense_op1_voltage",  "o3_mid_alphasense_op2_voltage"),
    "co":  ("co_mid_alphasense_op1_voltage",  "co_mid_alphasense_op2_voltage"),
}

# Physical constants
MOLECULAR_WEIGHTS = {"no2": 46.0055, "o3": 48.0, "co": 28.01}  # Molar mass (g/mol)
P_STANDARD = 101325.0                                          # Pressure of 1 atm (Pa)
R_GAS = 8.314                                                  # Gas constant (J/(mol K))

# -------------------------------------------------------------------------
# Manufacturer curves as look-up tables (s5)
# -------------------------------------------------------------------------
# CO (GM-702B datasheet, Fig. 2): RS/R0 falls as the concentration rises; log-log interpolation
CO_CURVE_RATIO = np.array([0.76, 0.53, 0.40, 0.28, 0.22, 0.18, 0.15, 0.13, 0.12])
CO_CURVE_PPM   = np.array([1,    5,    10,   20,   50,   100,  150,  500,  1000], dtype=float)

# NO2 (GM-102B datasheet, Fig. 4): RS/R0 rises with the concentration; linear interpolation
NO2_CURVE_RATIO = np.array([1.0, 1.2, 1.4, 1.8, 2.25, 2.65, 3.05, 3.4, 3.8, 4.15, 4.45, 4.65])
NO2_CURVE_PPM   = np.array([0,   0.5, 1,   2,   3,    4,    5,    6,   7,   8,    9,    10], dtype=float)

# O3 (MQ131 datasheet, Fig. 2): RS/R0 rises with the concentration; log-log interpolation
O3_CURVE_RATIO = np.array([1.2, 1.6, 1.8, 1.9, 2.0, 2.5, 2.8, 4.0, 6.0, 8.0])
O3_CURVE_PPB   = np.array([10,  20,  30,  40,  50,  80,  100, 200, 500, 1000], dtype=float)

# -------------------------------------------------------------------------
# OLS parameters (s6)
# -------------------------------------------------------------------------
IQR_FACTOR = 4.0         # Modified Tukey fence (relaxed for log-normal tails)
MIN_OBS_OLS = 50         # Minimum observations to fit the OLS model

# -------------------------------------------------------------------------
# Machine learning calibration (s9)
# -------------------------------------------------------------------------
DATA_S9 = DATA_DIR / "s9_ml"
RESULTS_S9 = RESULTS_DIR / "s9_ml"
RESULTS_S9_MODELS = RESULTS_S9 / "models"
RESULTS_S9_PLOTS = RESULTS_S9 / "plots"


# NO2_alpha: Alphasense NO2-B43F calibrated against the same LAQN target as the low-cost NO2
POLLUTANTS_ML = ["NO2", "NO2_alpha", "CO", "O3", "PM2.5"]

# Reference column of each pollutant (high-cost or mid-cost)
ML_REF_COLS = {
    "NO2":       COL_REF_NO2,       # LAQN reference station
    "NO2_alpha": COL_REF_NO2,       # Same target as the low-cost NO2
    "CO":        COL_ALPHA_CO,      # Alphasense CO-B4
    "O3":        COL_ALPHA_O3,      # Alphasense OX-B431
    "PM2.5":     COL_REF_PM25,      # LAQN reference station
}

# Chronological train/test split
ML_TRAIN_END = "2025-10-31"
ML_TEST_START = "2025-11-01"

# Reproducibility
ML_RANDOM_STATE = 42
ML_CV_SPLITS = 5

# Config A: raw sensor signals, ambient variables, absolute humidity and time features
ML_FEATURES_A_GAS = [
    COL_NO2_RAW, COL_CO_RAW, COL_O3_VOLTAGE,
    COL_TEMP, COL_RH,
    COL_VOC_RAW, COL_PM25,
    "abs_humidity",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

# Config A for PM2.5: SPS30 mass concentrations instead of the gas signals
ML_FEATURES_A_PM25 = [
    COL_PM25, COL_PM10,
    COL_TEMP, COL_RH,
    COL_VOC_RAW,
    "abs_humidity",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

# Config B: config A plus the OLS prediction c_hat_ols_{gas}, added at run time
ML_FEATURES_B_BASE_GAS = ML_FEATURES_A_GAS.copy()
ML_FEATURES_B_BASE_PM25 = ML_FEATURES_A_PM25.copy()

# Config C: correction of the OLS prediction c_hat_ols_{gas}, added at run time
ML_FEATURES_C_BASE = [
    COL_TEMP, COL_RH,
    "abs_humidity",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

# NO2_alpha, config A: Alphasense concentration as raw signal; low-cost hardware channels left out
ML_FEATURES_A_NO2_ALPHA = [
    COL_ALPHA_NO2,
    COL_TEMP, COL_RH,
    "abs_humidity",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
# NO2_alpha, config B: config A plus c_hat_ols_NO2_alpha, added at run time
ML_FEATURES_B_BASE_NO2_ALPHA = ML_FEATURES_A_NO2_ALPHA.copy()

# Ablation configs (A = D + S), used only by experiments/s9d_ablation.py
# D: ambient and time features, with VOC and PM2.5 as covariates as in A, and no gas signal
ML_FEATURES_D_GAS = [
    COL_TEMP, COL_RH, COL_VOC_RAW, COL_PM25, "abs_humidity",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
ML_FEATURES_D_PM25 = [
    COL_TEMP, COL_RH, COL_VOC_RAW, "abs_humidity",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
# D_strict: only T, RH, absolute humidity and time
ML_FEATURES_D_STRICT = [
    COL_TEMP, COL_RH, "abs_humidity",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
# S: only the sensor channels of A
ML_FEATURES_S_GAS = [COL_NO2_RAW, COL_CO_RAW, COL_O3_VOLTAGE]
ML_FEATURES_S_PM25 = [COL_PM25, COL_PM10]
# S1: only the channel of the target
ML_FEATURES_S1 = {
    "NO2": [COL_NO2_RAW], "CO": [COL_CO_RAW], "O3": [COL_O3_VOLTAGE],
    "PM2.5": [COL_PM25], "NO2_alpha": [COL_ALPHA_NO2],
}
ML_CONFIGS_ABLATION = ["D", "D_strict", "S", "S1"]

# Search spaces for RandomizedSearchCV
HP_RF = {
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [5, 10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, 0.8],
}
HP_RF_N_ITER = 100

HP_LGBM = {
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [3, 5, 7, 10, -1],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [5, 10, 20, 50],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.0, 0.1, 1.0],
}
HP_LGBM_N_ITER = 100
HP_LGBM_EARLY_STOPPING = 50

HP_DNN = {
    "hidden_dim_1": [32, 64, 128],
    "hidden_dim_2": [16, 32, 64],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [1e-4, 5e-4, 1e-3, 5e-3],
    "batch_size": [32, 64, 128],
    "epochs": [100, 200, 300],
}
HP_DNN_N_ITER = 40
HP_DNN_PATIENCE = 20

HP_LSTM = {
    "lstm_units": [32, 64, 128],
    "fc_dim": [16, 32, 64],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [1e-4, 5e-4, 1e-3],
    "batch_size": [32, 64],
    "epochs": [100, 200],
}
HP_LSTM_N_ITER = 20
HP_LSTM_PATIENCE = 20
LSTM_WINDOW = 96                # 24 h window at 15 min resolution (96 x 15 min)
ML_RESAMPLE_FREQ = "15min"      # Time resolution of the ML data

HP_XGBOOST = {
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.0, 0.1, 1.0],
}
HP_XGBOOST_N_ITER = 100

# Prefix of the output columns of each pollutant
ML_COL_PREFIXES = {
    "NO2":       "no2_low_mgsv2",
    "NO2_alpha": "no2_mid_alphasense",
    "CO":        "co_low_mgsv2",
    "O3":        "o3_low_mq131",
    "PM2.5":     "pm2_5_low_sps30",
}

# Main models, and supplementary models trained after them
ML_MODELS = ["RF", "LightGBM", "DNN"]
ML_MODELS_SUPPLEMENTARY = ["XGBoost", "LSTM"]
ML_CONFIGS = ["A", "B", "C"]

# Monthly drift thresholds (slope of R2 against the month index)
ML_DRIFT_RAPID = -0.05       # Below: rapid degradation
ML_DRIFT_MODERATE = -0.01    # Below: moderate degradation; above: stable

# Quality tiers of the ML models (minimum R2, maximum nRMSE in %)
ML_CEN_EXCELENTE = {"r2": 0.80, "nrmse": 20.0}
ML_CEN_ACEPTABLE = {"r2": 0.60, "nrmse": 30.0}
ML_CEN_MARGINAL = {"r2": 0.40, "nrmse": 50.0}

# Minimum observations for the evaluation
ML_MIN_OBS_TRAIN = 500
ML_MIN_OBS_TEST = 200
ML_MIN_OBS_DRIFT = 50
