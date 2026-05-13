"""
data_preprocessing.py
---------------------
Faithful implementation of Section IV-B of Azzouni & Pujolle (2017).

Core transformations:
  1. Read traffic-matrix CSV (one row per time slot, columns y_ij flattened).
  2. Optionally trim to a range of slots.
  3. Normalise by dividing by the max value (paper: "We normalize the data
     by dividing by the maximum value.").
  4. Build sliding-window training tensors:
        X.shape == (num_samples, W, N^2)
        Y.shape == (num_samples, N^2)
     where W is the learning-window size and sample i predicts
     vector at time t = i + W from vectors at times t-W ... t-1.
  5. Split into train / test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


TRAFFIC_COL_PREFIX = "y_"


@dataclass
class TrafficData:
    """Bundle everything a model needs: raw, normalised, windowed."""
    raw: np.ndarray            # (T, N^2)    original magnitudes
    scaled: np.ndarray         # (T, N^2)    /max scaled to [0, 1]
    scale: float               # the max used for inverse-transform
    n_nodes: int               # N (sqrt of feature count)
    timestamps: Optional[pd.Series] = None


def load_traffic_csv(path: str | Path) -> TrafficData:
    """Load a traffic-matrix CSV produced by AnyLogic or the surrogate."""
    df = pd.read_csv(path)
    traffic_cols = [c for c in df.columns if c.startswith(TRAFFIC_COL_PREFIX)]
    if not traffic_cols:
        raise ValueError(
            f"No traffic columns (prefix '{TRAFFIC_COL_PREFIX}') in {path}."
        )

    raw = df[traffic_cols].to_numpy(dtype=np.float32)
    n_features = raw.shape[1]
    n_nodes = int(round(np.sqrt(n_features)))
    if n_nodes * n_nodes != n_features:
        raise ValueError(
            f"Feature count {n_features} is not a perfect square — "
            f"cannot reshape to N x N matrix."
        )

    scale = float(raw.max()) if raw.max() > 0 else 1.0
    scaled = raw / scale

    timestamps = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else None

    return TrafficData(
        raw=raw,
        scaled=scaled,
        scale=scale,
        n_nodes=n_nodes,
        timestamps=timestamps,
    )


def build_windows(
    series: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a (T, F) time series into sliding windows.

    Returns
    -------
    X : (T - window, window, F) float32
    Y : (T - window, F)         float32
    """
    if series.ndim != 2:
        raise ValueError("series must be 2-D (T, F)")
    T, F = series.shape
    if window >= T:
        raise ValueError(f"window ({window}) >= T ({T})")

    n_samples = T - window
    X = np.empty((n_samples, window, F), dtype=np.float32)
    Y = np.empty((n_samples, F), dtype=np.float32)
    for i in range(n_samples):
        X[i] = series[i : i + window]
        Y[i] = series[i + window]
    return X, Y


def train_test_split_timeseries(
    X: np.ndarray,
    Y: np.ndarray,
    test_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Respect temporal order — the last `test_frac` of samples is held out."""
    n = X.shape[0]
    n_test = max(1, int(round(n * test_frac)))
    n_train = n - n_test
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def flatten_window(X: np.ndarray) -> np.ndarray:
    """For non-recurrent baselines: (n, W, F) -> (n, W*F)."""
    n, W, F = X.shape
    return X.reshape(n, W * F)
