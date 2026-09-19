"""
Model factory and PyTorch regressors (s9)

Defines the DNN and LSTM regressors as scikit-learn estimators, with
scaling, early stopping and checkpoint save and load. create_model()
returns the estimator, search space and number of search iterations
of RF, LightGBM, XGBoost, DNN or LSTM.
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from config import (
    ML_RANDOM_STATE,
    HP_RF, HP_RF_N_ITER,
    HP_LGBM, HP_LGBM_N_ITER,
    HP_DNN, HP_DNN_N_ITER,
    HP_LSTM, HP_LSTM_N_ITER, LSTM_WINDOW,
    HP_XGBOOST, HP_XGBOOST_N_ITER,
)

log = logging.getLogger("s9.models")


# -------------------------------------------------------------------------
# DNN regressor (scikit-learn wrapper of a PyTorch MLP)
# -------------------------------------------------------------------------
class DNNRegressor(BaseEstimator, RegressorMixin):
    """
    scikit-learn wrapper of the MLP used for the calibration.
    Architecture: input, two hidden layers (ReLU and dropout) and a linear output.
    Compatible with RandomizedSearchCV through get_params and set_params.
    """

    def __init__(self, n_features=1, hidden_dim_1=64, hidden_dim_2=32,
                 dropout=0.2, lr=1e-3, batch_size=64, epochs=200,
                 patience=20, random_state=42):
        self.n_features = n_features
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.scaler_ = None
        self.model_ = None
        self.train_losses_ = []

    @staticmethod
    def _get_device():
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.random_state)
        model = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim_2, 1),
        )
        return model.to(self._get_device())

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        device = self._get_device()

        # Scaler fitted on the training data only
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.n_features = X.shape[1]
        self.model_ = self._build_model()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        y_t = torch.tensor(np.asarray(y, dtype=np.float64).reshape(-1, 1),
                           dtype=torch.float32, device=device)

        n = len(X_t)
        n_val = max(int(n * 0.15), 10)  # Last 15 % kept as internal validation
        X_train, X_val = X_t[:-n_val], X_t[-n_val:]
        y_train, y_val = y_t[:-n_val], y_t[-n_val:]

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        self.train_losses_ = []

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model_(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss /= len(X_train)
            self.train_losses_.append(epoch_loss)

            # Early stopping on the validation loss
            self.model_.eval()
            with torch.no_grad():
                val_pred = self.model_(X_val)
                val_loss = criterion(val_pred, y_val).item()
            self.model_.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X):
        import torch
        device = self._get_device()
        self.model_.eval()
        X_scaled = self.scaler_.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = self.model_(X_t).cpu().numpy().flatten()
        return pred

    def save_torch(self, path):
        """Save the PyTorch weights and the scaler."""
        import torch
        import joblib
        torch.save(self.model_.state_dict(), path)
        scaler_path = str(path).replace(".pt", "_scaler.joblib")
        joblib.dump(self.scaler_, scaler_path)

    def load_torch(self, path, n_features):
        """
        Load the PyTorch weights and the scaler.
        map_location sends the weights to the available device, so that a
        checkpoint trained on a GPU can be loaded on a CPU-only machine.
        """
        import torch
        import joblib
        self.n_features = n_features
        self.model_ = self._build_model()
        device = self._get_device()
        self.model_.load_state_dict(
            torch.load(path, weights_only=True, map_location=device)
        )
        self.model_.eval()
        scaler_path = str(path).replace(".pt", "_scaler.joblib")
        self.scaler_ = joblib.load(scaler_path)
        return self


# -------------------------------------------------------------------------
# LSTM regressor (supplementary analysis)
# -------------------------------------------------------------------------
class LSTMRegressor(BaseEstimator, RegressorMixin):
    """
    scikit-learn wrapper of the LSTM, sequence to one over 24 h windows (96 periods of 15 min).
    Architecture: one LSTM layer, a fully connected ReLU layer and a linear output.
    predict() returns an array of length N, with NaN where no complete contiguous window exists.
    RandomizedSearchCV passes no timestamps to fit(), so windows may span gaps during the search.
    """

    def __init__(self, n_features=1, lstm_units=64, fc_dim=32,
                 dropout=0.2, lr=1e-3, batch_size=64, epochs=200,
                 patience=20, window=96, random_state=42):
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.window = window
        self.random_state = random_state
        self.scaler_ = None
        self.model_ = None
        self.train_losses_ = []

    @staticmethod
    def _get_device():
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.random_state)

        class _LSTMNet(nn.Module):
            def __init__(self, n_features, lstm_units, fc_dim, dropout):
                super().__init__()
                self.lstm = nn.LSTM(n_features, lstm_units, num_layers=1,
                                   batch_first=True, dropout=0)
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(lstm_units, fc_dim)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(fc_dim, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last = lstm_out[:, -1, :]
                out = self.dropout(last)
                out = self.relu(self.fc1(out))
                out = self.fc2(out)
                return out

        model = _LSTMNet(self.n_features, self.lstm_units, self.fc_dim,
                         self.dropout)
        return model.to(self._get_device())

    @staticmethod
    def create_contiguous_windows(X, y, window, timestamps=None):
        """
        Build sliding windows over temporally contiguous segments only.
        A gap of more than 30 min between consecutive timestamps starts a new segment.
        Return X_windows (n_win, window, n_features), y_windows (target of the last
        period of each window) and the index of each target in X and y.
        """
        n = len(X)
        if n < window:
            return np.empty((0, window, X.shape[1])), np.empty(0), np.empty(0, dtype=int)

        # Contiguous segments. pd.Series keeps diff() and cumsum() as Series for
        # numpy and pandas inputs alike (pd.to_datetime of an array gives a DatetimeIndex)
        if timestamps is not None:
            import pandas as pd
            ts = pd.Series(pd.to_datetime(timestamps))
            diffs = ts.diff()
            gap_mask = diffs > pd.Timedelta(minutes=30)
            segment_ids = gap_mask.cumsum().values
        else:
            segment_ids = np.zeros(n, dtype=int)

        X_windows_list = []
        y_windows_list = []
        idx_list = []

        for seg_id in np.unique(segment_ids):
            seg_mask = segment_ids == seg_id
            seg_idx = np.where(seg_mask)[0]

            if len(seg_idx) < window:
                continue

            X_seg = X[seg_idx]
            y_seg = y[seg_idx]

            for i in range(len(seg_idx) - window + 1):
                X_windows_list.append(X_seg[i:i + window])
                y_windows_list.append(y_seg[i + window - 1])
                idx_list.append(seg_idx[i + window - 1])

        if not X_windows_list:
            return np.empty((0, window, X.shape[1])), np.empty(0), np.empty(0, dtype=int)

        return (np.array(X_windows_list), np.array(y_windows_list),
                np.array(idx_list, dtype=int))

    def fit(self, X, y, timestamps=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        device = self._get_device()

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.n_features = X.shape[1]

        X_win, y_win, _ = self.create_contiguous_windows(
            X_scaled, np.asarray(y, dtype=np.float64),
            self.window, timestamps
        )

        if len(X_win) < 50:
            log.warning("LSTM: only %d contiguous windows (min 50), skipped", len(X_win))
            return self

        self.model_ = self._build_model()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_win, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_win.reshape(-1, 1), dtype=torch.float32, device=device)

        n = len(X_t)
        n_val = max(int(n * 0.15), 10)  # Last 15 % kept as internal validation
        X_train, X_val = X_t[:-n_val], X_t[-n_val:]
        y_train, y_val = y_t[:-n_val], y_t[-n_val:]

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        self.train_losses_ = []

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model_(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss /= len(X_train)
            self.train_losses_.append(epoch_loss)

            # Early stopping on the validation loss
            self.model_.eval()
            with torch.no_grad():
                val_pred = self.model_(X_val)
                val_loss = criterion(val_pred, y_val).item()
            self.model_.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X, timestamps=None):
        """
        Predict over contiguous windows.
        Return an array of length len(X), with NaN where no complete contiguous window exists.
        """
        import torch
        if self.model_ is None:
            return np.full(len(X), np.nan)

        device = self._get_device()
        self.model_.eval()
        X_scaled = self.scaler_.transform(X)

        dummy_y = np.zeros(len(X))
        X_win, _, target_indices = self.create_contiguous_windows(
            X_scaled, dummy_y, self.window, timestamps
        )

        result = np.full(len(X), np.nan)

        if len(X_win) == 0:
            log.warning("LSTM predict: 0 contiguous windows in %d rows, returning NaN", len(X))
            return result

        X_t = torch.tensor(X_win, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = self.model_(X_t).cpu().numpy().flatten()

        result[target_indices] = pred
        return result

    def score(self, X, y, sample_weight=None):
        """
        R2 over the finite predictions only, so that the NaN-padded output can be scored.
        Return -inf with fewer than 10 valid points.
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        valid = np.isfinite(y_pred)
        if valid.sum() < 10:
            return float("-inf")
        y_true = np.asarray(y)
        valid = valid & np.isfinite(y_true)
        if valid.sum() < 10:
            return float("-inf")
        return r2_score(y_true[valid], y_pred[valid])

    def save_torch(self, path):
        """Save the PyTorch weights and the scaler."""
        import torch
        import joblib
        if self.model_ is not None:
            torch.save(self.model_.state_dict(), path)
        scaler_path = str(path).replace(".pt", "_scaler.joblib")
        if self.scaler_ is not None:
            joblib.dump(self.scaler_, scaler_path)

    def load_torch(self, path, n_features):
        """
        Load the PyTorch weights and the scaler.
        map_location sends the weights to the available device, so that a
        checkpoint trained on a GPU can be loaded on a CPU-only machine.
        """
        import torch
        import joblib
        self.n_features = n_features
        self.model_ = self._build_model()
        device = self._get_device()
        self.model_.load_state_dict(
            torch.load(path, weights_only=True, map_location=device)
        )
        self.model_.eval()
        scaler_path = str(path).replace(".pt", "_scaler.joblib")
        self.scaler_ = joblib.load(scaler_path)
        return self


# -------------------------------------------------------------------------
# Model factory
# -------------------------------------------------------------------------
def _create_lgbm():
    """Create the LightGBM estimator (lazy import)."""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        random_state=ML_RANDOM_STATE,
        verbose=-1,
        n_jobs=1,  # The search already runs in parallel (n_jobs=-1)
    )


def _create_xgboost():
    """Create the XGBoost estimator (lazy import)."""
    from xgboost import XGBRegressor
    return XGBRegressor(
        random_state=ML_RANDOM_STATE,
        verbosity=0,
        n_jobs=1,  # The search already runs in parallel (n_jobs=-1)
    )


def create_model(model_name, n_features):
    """Return (estimator, param_dist, n_iter) for RF, LightGBM, DNN, XGBoost or LSTM."""
    if model_name == "RF":
        estimator = RandomForestRegressor(
            random_state=ML_RANDOM_STATE,
            n_jobs=-1,
        )
        return estimator, HP_RF, HP_RF_N_ITER

    elif model_name == "LightGBM":
        estimator = _create_lgbm()
        return estimator, HP_LGBM, HP_LGBM_N_ITER

    elif model_name == "DNN":
        estimator = DNNRegressor(
            n_features=n_features,
            random_state=ML_RANDOM_STATE,
        )
        return estimator, HP_DNN, HP_DNN_N_ITER

    elif model_name == "XGBoost":
        estimator = _create_xgboost()
        return estimator, HP_XGBOOST, HP_XGBOOST_N_ITER

    elif model_name == "LSTM":
        estimator = LSTMRegressor(
            n_features=n_features,
            window=LSTM_WINDOW,
            random_state=ML_RANDOM_STATE,
        )
        return estimator, HP_LSTM, HP_LSTM_N_ITER

    else:
        raise ValueError(f"Unknown model: {model_name}")
