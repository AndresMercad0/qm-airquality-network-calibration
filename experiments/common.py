"""
Shared helpers of the experiment scripts

Loads the pipeline configured for one node, and provides the LSTM
wrapper that carries timestamps through scikit-learn, the masked RMSE
scorer and the names of the reference files used by the study.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Repository layout
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
RESULTS_CROSS = REPO_ROOT / "results" / "cross_node"

PREDICT_CHUNK = 1024    # Windows per inference block of the LSTM (bounds GPU memory)

# -------------------------------------------------------------------------
# Reference files used by the study (LAQN downloads of 15 February 2026)
# -------------------------------------------------------------------------
REF_CSV_NAME = {
    3: "node_TH2_MER_wide_data_2025-04-16_to_2026-02-15.csv",
    5: "node_TH7_KEMP_wide_data_2025-04-16_to_2026-02-15.csv",
}


# -------------------------------------------------------------------------
# Loading the pipeline of one node
# -------------------------------------------------------------------------
def reset_pipeline_modules():
    """Drop cached pipeline modules so the requested node loads its own config."""
    for mod_name in list(sys.modules.keys()):
        if mod_name in {"config", "utils", "cross_node_features"} or mod_name == "ml" \
                or mod_name.startswith("ml."):
            del sys.modules[mod_name]


def load_node_pipeline(node: int):
    """Import the pipeline configured for one node and return (config, utils)."""
    reset_pipeline_modules()
    os.environ["AQMS_NODE"] = f"node_{node}"
    sys.path = [p for p in sys.path if p != str(PIPELINE_DIR)]
    sys.path.insert(0, str(PIPELINE_DIR))
    import config as cfg
    import utils
    return cfg, utils


# -------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------
def versions_banner() -> dict:
    """Print and return the versions of Python and of the main libraries."""
    import sklearn
    import torch
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for k, v in info.items():
        print(f"  {k:14s} {v}")
    return info


def masked_neg_rmse(estimator, X, y):
    """
    Return the negative RMSE over the positions with a finite prediction.
    The LSTM returns NaN outside complete windows; fewer than 10 points give NaN.
    """
    y_pred = estimator.predict(X)
    y_true = np.asarray(y, dtype=float)
    valid = np.isfinite(y_pred) & np.isfinite(y_true)
    if valid.sum() < 10:
        return np.nan
    return -float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))


def make_ts_subclass(LSTMRegressor):
    """
    Return a subclass that carries the timestamps as the last column of X
    (epoch seconds) and hands them to fit and predict of the parent class.
    """

    class LSTMRegressorTS(LSTMRegressor):
        @staticmethod
        def _split(X):
            X = np.asarray(X)
            ts = pd.to_datetime(X[:, -1].astype("int64"), unit="s", utc=True)
            return X[:, :-1].astype(np.float64), ts.values

        def fit(self, X, y, timestamps=None):
            X_feat, ts = self._split(X)
            return super().fit(X_feat, y, timestamps=ts)

        def predict(self, X, timestamps=None):
            """Predict as the parent class, in blocks of windows to bound the GPU memory."""
            import torch
            X_feat, ts = self._split(X)
            if self.model_ is None:
                return np.full(len(X_feat), np.nan)
            device = self._get_device()
            self.model_.eval()
            X_scaled = self.scaler_.transform(X_feat)
            X_win, _, target_indices = self.create_contiguous_windows(
                X_scaled, np.zeros(len(X_feat)), self.window, ts)
            result = np.full(len(X_feat), np.nan)
            if len(X_win) == 0:
                return result
            preds = []
            with torch.no_grad():
                for start in range(0, len(X_win), PREDICT_CHUNK):
                    X_t = torch.tensor(X_win[start:start + PREDICT_CHUNK],
                                       dtype=torch.float32, device=device)
                    preds.append(self.model_(X_t).cpu().numpy().flatten())
                    del X_t
            result[target_indices] = np.concatenate(preds)
            return result

    LSTMRegressorTS.__name__ = "LSTMRegressorTS"
    return LSTMRegressorTS


def epoch_seconds(ts_values) -> np.ndarray:
    """Convert timestamps to seconds since 1970-01-01 (UTC)."""
    ts = pd.to_datetime(pd.Series(ts_values))
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    secs = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    return secs.to_numpy().astype(np.float64)
