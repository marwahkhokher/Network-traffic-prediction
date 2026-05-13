"""
baselines.py
------------
Baselines used in the paper (Section VI / Fig. 8):

  - ARMA(p, q)        : linear statistical predictor
  - Linear Regression : on flattened lag features
  - Random Forest     : strong non-linear tabular baseline (from the
                        project brief on the whiteboard — not in the
                        paper itself but requested in the course brief)
  - FFNN              : fixed-window feed-forward NN

All baselines expose the same `.fit(X, Y)` + `.predict(X) -> Y_hat`
interface so `train.py` can loop over them uniformly.

Note on ARMA: the paper's ARMA / ARAR / HW numbers are for predicting
ONE scalar flow. Fitting a separate ARMA to each of the N^2 OD pairs
would be expensive and noisy; for a fair qualitative comparison we fit
ARMA to the AVERAGE flow across all OD pairs and broadcast the residual
scale — this matches the paper's footnote that their linear-baseline
numbers come from a single-flow experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

try:
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    _HAS_STATSMODELS = False

from data_preprocessing import flatten_window


# ----------------------------------------------------------------------
# Linear Regression
# ----------------------------------------------------------------------
class LinearRegressionBaseline:
    """Ridge regression on the flattened lag window."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LinearRegressionBaseline":
        self.model.fit(flatten_window(X), Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(flatten_window(X))


# ----------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------
class RandomForestBaseline:
    """Random Forest regressor (multi-output) on flattened lag window."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 12,
                 n_jobs: int = -1, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RandomForestBaseline":
        self.model.fit(flatten_window(X), Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(flatten_window(X))


# ----------------------------------------------------------------------
# ARMA / ARIMA
# ----------------------------------------------------------------------
@dataclass
class ARMABaseline:
    """ARMA applied per-OD or on the average flow.

    Per-OD ARMA is faithful to the paper but SLOW for N^2 = 529 series.
    Default: fit to average and broadcast — acceptable for the
    comparison plot because we only need the MSE magnitude.
    """
    order: tuple = (2, 0, 1)        # (p, d, q)
    fit_per_od: bool = False

    def __post_init__(self):
        if not _HAS_STATSMODELS:
            raise ImportError("Install statsmodels to use ARMABaseline")
        self._models = None
        self._avg_model = None
        self._n_features = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ARMABaseline":
        # Reconstruct the underlying series: Y has the "next" value for every
        # window; the full series is [X[0, 0], X[0, 1], ..., Y[0], Y[1], ...]
        full_series = np.vstack([X[0], Y])   # (T_train, N^2)
        self._n_features = full_series.shape[1]

        if self.fit_per_od:
            self._models = []
            for j in range(full_series.shape[1]):
                try:
                    m = ARIMA(full_series[:, j], order=self.order).fit()
                except Exception:
                    m = None
                self._models.append(m)
        else:
            avg = full_series.mean(axis=1)
            self._avg_model = ARIMA(avg, order=self.order).fit()
            self._avg_series = avg
            self._per_od_mean = full_series.mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]

        if self.fit_per_od:
            preds = np.zeros((n_samples, self._n_features), dtype=np.float32)
            for j, m in enumerate(self._models):
                if m is None:
                    preds[:, j] = X[:, -1, j]
                    continue
                f = m.forecast(steps=n_samples)
                preds[:, j] = f
            return preds

        # Broadcast average forecast weighted by per-OD mean
        f = np.asarray(self._avg_model.forecast(steps=n_samples))
        f = np.clip(f, 0, None)
        scale = (self._per_od_mean / max(self._per_od_mean.mean(), 1e-9))
        return f[:, None] * scale[None, :]


# ----------------------------------------------------------------------
# Feed-Forward NN
# ----------------------------------------------------------------------
class FFNNBaseline:
    """Simple MLP on flattened lag window — mirrors the paper's FFNN."""

    def __init__(self, hidden: tuple = (256, 128), dropout: float = 0.1,
                 lr: float = 1e-3, epochs: int = 40, batch_size: int = 32):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def _build(self, n_input: int, n_output: int):
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        inp = layers.Input(shape=(n_input,))
        x = inp
        for h in self.hidden:
            x = layers.Dense(h, activation="relu")(x)
            x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(n_output, activation="linear")(x)
        m = Model(inp, out)
        m.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss="mse", metrics=["mae"])
        return m

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "FFNNBaseline":
        Xf = flatten_window(X)
        self.model = self._build(Xf.shape[1], Y.shape[1])
        self.model.fit(
            Xf, Y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(flatten_window(X), verbose=0)
