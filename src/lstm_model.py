"""
lstm_model.py
-------------
LSTM RNN framework for traffic-matrix prediction (Azzouni & Pujolle 2017,
Section III and IV).

The paper experiments with:
  - sizes: 200, 300, 400, 500, 600, 700 hidden units (Fig. 6)
  - depth: 1 to 6 stacked layers (Fig. 7)

We provide a configurable builder that matches those sweeps, plus
a sensible default used by `train.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


@dataclass
class LSTMConfig:
    window: int = 12              # W in the paper (number of past slots)
    n_features: int = 529         # N^2 ; 23^2 for a GEANT-sized network
    hidden_sizes: List[int] = field(default_factory=lambda: [300])
    dropout: float = 0.1
    recurrent_dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    patience: int = 8


def build_lstm(cfg: LSTMConfig) -> Model:
    """Build a stacked-LSTM regression model.

    Input : (batch, W, N^2) — sliding window of traffic vectors
    Output: (batch, N^2)    — predicted next traffic vector
    """
    inputs = layers.Input(shape=(cfg.window, cfg.n_features), name="traffic_window")
    x = inputs
    for i, units in enumerate(cfg.hidden_sizes):
        return_sequences = i < len(cfg.hidden_sizes) - 1
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=cfg.dropout,
            recurrent_dropout=cfg.recurrent_dropout,
            name=f"lstm_{i + 1}",
        )(x)

    # The paper uses a linear output — one value per OD pair, no activation
    outputs = layers.Dense(cfg.n_features, activation="linear", name="traffic_vector")(x)

    model = Model(inputs, outputs, name="TrafficMatrixLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_lstm(
    cfg: LSTMConfig,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    verbose: int = 1,
) -> tuple[Model, dict]:
    """Train the model with early stopping and LR reduction on plateau."""
    model = build_lstm(cfg)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, cfg.patience // 2),
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history.history


def predict_next(model: Model, window: np.ndarray) -> np.ndarray:
    """Predict the next traffic vector given a single window.

    Parameters
    ----------
    window : shape (W, N^2)  OR  (1, W, N^2)

    Returns
    -------
    np.ndarray of shape (N^2,)
    """
    if window.ndim == 2:
        window = window[None, ...]
    y = model.predict(window, verbose=0)
    return y[0]
