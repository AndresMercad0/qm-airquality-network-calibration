"""
Training loop of the machine learning calibration (s9)

Trains each (gas, config, model) combination with RandomizedSearchCV
and TimeSeriesSplit on the training period and scores it on the test
period. Saves each model (.joblib or .pt) and returns the metrics table,
which s9 writes as tabla_maestra_resultados.csv. Logs CPU and GPU use.
"""

import logging
import subprocess
import threading
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from config import (
    POLLUTANTS_ML, ML_MODELS, ML_CONFIGS,
    ML_REF_COLS, ML_RANDOM_STATE, ML_CV_SPLITS,
    ML_MIN_OBS_TRAIN, ML_MIN_OBS_TEST,
    HP_LGBM_EARLY_STOPPING, COL_TS,
)
from .data_loader import get_feature_columns
from .models import create_model, DNNRegressor
from .evaluator import evaluate_model, compute_overfit_ratio

log = logging.getLogger("s9.trainer")


# -------------------------------------------------------------------------
# Training of one (gas, config, model) combination
# -------------------------------------------------------------------------
def _fit_single_combination(df, train_mask, test_mask, gas, config, model_name,
                            save_dir):
    """
    Train one combination with RandomizedSearchCV and TimeSeriesSplit.
    Return a dict of results, or None when the data are not enough.
    """
    ref_col = ML_REF_COLS[gas]
    feature_cols = get_feature_columns(gas, config, df)

    if not feature_cols:
        log.warning("  [%s/%s/%s] No features available, skipped",
                     gas, config, model_name)
        return None

    # Rows with every feature and the reference
    required = feature_cols + [ref_col]
    valid = df[required].notna().all(axis=1)

    train_valid = valid & train_mask
    test_valid = valid & test_mask

    n_train = train_valid.sum()
    n_test = test_valid.sum()

    if n_train < ML_MIN_OBS_TRAIN:
        log.warning("  [%s/%s/%s] Only %d training obs (min=%d), skipped",
                     gas, config, model_name, n_train, ML_MIN_OBS_TRAIN)
        return None
    if n_test < ML_MIN_OBS_TEST:
        log.warning("  [%s/%s/%s] Only %d test obs (min=%d), skipped",
                     gas, config, model_name, n_test, ML_MIN_OBS_TEST)
        return None

    X_train = df.loc[train_valid, feature_cols].values
    y_train = df.loc[train_valid, ref_col].values
    X_test = df.loc[test_valid, feature_cols].values
    y_test = df.loc[test_valid, ref_col].values

    n_features = X_train.shape[1]

    # Estimator and hyperparameter search space
    estimator, param_dist, n_iter = create_model(model_name, n_features)

    # TimeSeriesSplit keeps the temporal order in the cross-validation
    cv = TimeSeriesSplit(n_splits=ML_CV_SPLITS)

    log.info("  [%s/Config%s/%s] train=%d, test=%d, features=%d",
             gas, config, model_name, n_train, n_test, n_features)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            estimator, param_dist,
            n_iter=min(n_iter, _max_combinations(param_dist)),
            cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=ML_RANDOM_STATE,
            n_jobs=-1 if model_name not in ("DNN", "LSTM") else 1,
            verbose=0,
            refit=True,
            error_score="raise" if model_name not in ("DNN", "LSTM") else np.nan,
        )
        search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    # LightGBM: final refit with early stopping
    if model_name == "LightGBM":
        best_model = _refit_lgbm_early_stopping(
            best_model, best_params, X_train, y_train
        )

    # The LSTM needs the timestamps to detect data gaps, so that no window spans an outage
    if model_name == "LSTM":
        ts_train = df.loc[train_valid, COL_TS].values
        ts_test = df.loc[test_valid, COL_TS].values
        y_pred_train = best_model.predict(X_train, timestamps=ts_train)
        y_pred_test = best_model.predict(X_test, timestamps=ts_test)
    else:
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

    # The LSTM leaves NaN where no complete window exists: those positions are left out
    if model_name == "LSTM":
        valid_train = np.isfinite(y_pred_train)
        valid_test = np.isfinite(y_pred_test)
        metrics_train = evaluate_model(y_train[valid_train], y_pred_train[valid_train])
        metrics_test = evaluate_model(y_test[valid_test], y_pred_test[valid_test])
    else:
        metrics_train = evaluate_model(y_train, y_pred_train)
        metrics_test = evaluate_model(y_test, y_pred_test)

    overfit = compute_overfit_ratio(metrics_train["R2"], metrics_test["R2"])

    # Save the model
    model_filename = f"{gas}_{config}_{model_name}".lower()
    if model_name in ("DNN", "LSTM"):
        model_path = save_dir / f"{model_filename}.pt"
        best_model.save_torch(model_path)
    else:
        model_path = save_dir / f"{model_filename}.joblib"
        joblib.dump(best_model, model_path)

    log.info("    R2_train=%.3f, R2_test=%.3f, RMSE=%.2f, CEN=%s, overfit=%.2f",
             metrics_train["R2"], metrics_test["R2"],
             metrics_test["RMSE"], metrics_test["CEN_level"], overfit)

    return {
        "gas": gas,
        "config": config,
        "model": model_name,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "R2_train": metrics_train["R2"],
        "R2_test": metrics_test["R2"],
        "RMSE": metrics_test["RMSE"],
        "MAE": metrics_test["MAE"],
        "MBE": metrics_test["MBE"],
        "nRMSE": metrics_test["nRMSE"],
        "pearson_r": metrics_test["pearson_r"],
        "CEN_level": metrics_test["CEN_level"],
        "overfit_ratio": overfit,
        "best_params": str(best_params),
        "model_path": model_path.name,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "y_train": y_train,
        "y_pred_train": y_pred_train,
        "trained_model": best_model,
    }


def _refit_lgbm_early_stopping(model, params, X_train, y_train):
    """Refit LightGBM with early stopping, the last 15 % of the training data as eval_set."""
    from lightgbm import LGBMRegressor

    n = len(X_train)
    n_val = max(int(n * 0.15), 10)
    X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

    # Same parameters as the best estimator of the search
    refit_params = model.get_params()
    refit_model = LGBMRegressor(**refit_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refit_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__("lightgbm").early_stopping(
                    HP_LGBM_EARLY_STOPPING, verbose=False
                ),
                __import__("lightgbm").log_evaluation(period=0),
            ],
        )

    return refit_model


def _max_combinations(param_dist):
    """Estimate the maximum number of combinations of the search space."""
    total = 1
    for v in param_dist.values():
        if hasattr(v, "__len__"):
            total *= len(v)
        else:
            total *= 10  # Continuous distributions
    return total


# -------------------------------------------------------------------------
# Resource monitor (CPU and GPU)
# -------------------------------------------------------------------------
class _ResourceMonitor:
    """Daemon thread that logs the CPU and GPU usage every `interval` seconds."""

    def __init__(self, interval=20):
        self.interval = interval
        self.current_task = "initialising"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def set_task(self, task_str):
        self.current_task = task_str

    def _get_cpu_percent(self):
        try:
            # CPU counters from /proc/stat (Linux); -1 where it does not exist
            with open("/proc/stat") as f:
                line = f.readline()
            parts = line.split()
            idle = int(parts[4])
            total = sum(int(p) for p in parts[1:])
            # Previous reading kept to compute the delta
            if not hasattr(self, "_prev_idle"):
                self._prev_idle = idle
                self._prev_total = total
                return -1  # The first reading has no delta
            d_idle = idle - self._prev_idle
            d_total = total - self._prev_total
            self._prev_idle = idle
            self._prev_total = total
            if d_total == 0:
                return 0
            return round((1 - d_idle / d_total) * 100)
        except Exception:
            return -1

    def _get_gpu_info(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                timeout=5, text=True
            ).strip()
            parts = [p.strip() for p in out.split(",")]
            return f"GPU:{parts[0]}% VRAM:{parts[1]}MiB Temp:{parts[2]}C"
        except Exception:
            return "GPU:N/A"

    def _run(self):
        import time as _time
        # First CPU reading, to initialise the delta
        self._get_cpu_percent()
        _time.sleep(1)
        while not self._stop.wait(self.interval):
            cpu = self._get_cpu_percent()
            gpu = self._get_gpu_info()
            cpu_str = f"CPU:{cpu}%" if cpu >= 0 else "CPU:N/A"
            log.info("[MONITOR] %s | %s | %s", self.current_task, cpu_str, gpu)


# -------------------------------------------------------------------------
# Progress and remaining time
# -------------------------------------------------------------------------
def _estimate_remaining(model_times, pending_models, elapsed_total, completed_total):
    """Estimate the remaining time from the mean duration of each model type."""
    avg_per_type = {}
    for mtype, times in model_times.items():
        if times:
            avg_per_type[mtype] = sum(times) / len(times)

    # Fallback: global mean
    global_avg = elapsed_total / completed_total if completed_total > 0 else 0

    remaining = 0.0
    for mtype, count in pending_models.items():
        avg = avg_per_type.get(mtype, global_avg)
        remaining += avg * count

    return remaining


def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}min"


# -------------------------------------------------------------------------
# Training loop (gases x configs x models)
# -------------------------------------------------------------------------
def train_all_combinations(df, train_mask, test_mask, save_dir,
                           models=None, gases=None, configs=None):
    """
    Train every combination of gases, configs and models.
    The three lists default to POLLUTANTS_ML, ML_CONFIGS and ML_MODELS. A subset gives a partial
    run that leaves the other results untouched (the ablation passes D, D_strict, S and S1).
    Return results_df (metrics of every combination) and all_results (detailed dicts,
    arrays included).
    """
    import time as _time

    if models is None:
        models = ML_MODELS
    if gases is None:
        gases = POLLUTANTS_ML
    if configs is None:
        configs = ML_CONFIGS

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(gases) * len(configs) * len(models)
    count = 0

    # Durations per model type
    model_times = {}  # {"RF": [t1, t2, ...], "DNN": [t1, ...], ...}
    t_all_start = _time.time()

    # Pending combinations per model type
    pending_by_type = {}
    for m in models:
        pending_by_type[m] = pending_by_type.get(m, 0) + len(gases) * len(configs)

    # Resource monitoring
    monitor = _ResourceMonitor(interval=60)
    monitor.start()

    for gas in gases:
        for config in configs:
            for model_name in models:
                count += 1
                elapsed_total = _time.time() - t_all_start

                remaining = _estimate_remaining(
                    model_times, pending_by_type, elapsed_total, count - 1
                )
                eta_str = _format_time(remaining) if count > 1 else "estimating..."

                task_label = f"{gas}/Config{config}/{model_name} {count}/{total}"
                monitor.set_task(task_label)

                log.info("")
                log.info("=" * 60)
                log.info("  COMBINATION %d/%d  |  %s elapsed  |  ~%s remaining",
                         count, total, _format_time(elapsed_total), eta_str)
                log.info("=" * 60)

                t_comb_start = _time.time()
                try:
                    result = _fit_single_combination(
                        df, train_mask, test_mask,
                        gas, config, model_name, save_dir
                    )
                    if result is not None:
                        all_results.append(result)
                except Exception as e:
                    log.error("  [%s/%s/%s] Error: %s",
                              gas, config, model_name, e)

                # Duration of this combination
                t_comb = _time.time() - t_comb_start
                model_times.setdefault(model_name, []).append(t_comb)
                pending_by_type[model_name] -= 1

    monitor.stop()

    # Results table (without the large arrays)
    summary_cols = [
        "gas", "config", "model", "n_train", "n_test", "n_features",
        "R2_train", "R2_test", "RMSE", "MAE", "MBE", "nRMSE",
        "pearson_r", "CEN_level", "overfit_ratio", "best_params", "model_path",
    ]
    rows = [{k: r[k] for k in summary_cols if k in r} for r in all_results]
    results_df = pd.DataFrame(rows)

    elapsed_final = _time.time() - t_all_start
    log.info("Training complete: %d/%d successful combinations in %s",
             len(all_results), total, _format_time(elapsed_final))

    return results_df, all_results
