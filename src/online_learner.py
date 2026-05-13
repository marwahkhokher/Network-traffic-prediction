"""
online_learner.py
-----------------
Wraps the trained LSTM model and provides:

  1. predict(window) — same interface as the original LSTM, used by the
     streaming service.
  2. partial_fit(window, target) — incremental one-batch update, used by
     the streaming service after each new (window, target) pair is
     available. Uses a much smaller learning rate than initial training
     so a single sample can't blow up the weights.
  3. Drift detection — keeps a rolling window of recent prediction
     losses; flags drift when recent loss significantly exceeds the
     baseline established during initial training.
  4. Atomic model versioning — every retrain creates a new model file;
     the "current" symlink/file is atomically swapped so that no
     in-flight prediction ever sees a half-written model.

Two-tier learning:
  - Online (this module): one-batch updates per slot. Cheap, fast,
    keeps the model fresh against minute-by-minute drift.
  - Batch (retrain_worker.py): full retrain over the recent buffer
    on a trigger. Handles bigger distribution shifts.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class LearnerState:
    model_version: int = 0
    samples_seen_online: int = 0
    samples_since_last_retrain: int = 0
    last_online_loss: float = float("nan")
    rolling_loss_mean: float = float("nan")
    baseline_loss: Optional[float] = None
    drift_score: float = 0.0
    drift_flagged: bool = False
    last_partial_fit_ts: Optional[float] = None
    last_reload_ts: Optional[float] = None


class OnlineLearner:
    """Wraps a Keras LSTM with online-learning + drift detection."""

    def __init__(
        self,
        model_path: str | Path,
        scale: float = 1.0,
        online_lr: float = 1e-5,
        loss_window: int = 64,
        drift_multiplier: float = 2.5,
        retrain_threshold_samples: int = 500,
        state_path: Optional[str | Path] = None,
    ):
        """
        Parameters
        ----------
        model_path : path to the .keras file that the worker also writes to
        scale : the /max normalisation constant from initial training
        online_lr : learning rate for partial_fit (much smaller than initial)
        loss_window : how many recent losses to keep for drift detection
        drift_multiplier : drift flag triggers when rolling_mean > multiplier * baseline
        retrain_threshold_samples : trigger background retrain after this many new samples
        """
        self._model_path = Path(model_path)
        self._state_path = Path(state_path) if state_path else \
            self._model_path.with_suffix(".state.json")
        self._scale = scale
        self._online_lr = online_lr
        self._loss_window = loss_window
        self._drift_multiplier = drift_multiplier
        self._retrain_threshold = retrain_threshold_samples

        self._losses: deque = deque(maxlen=loss_window)
        self._lock = threading.RLock()
        self._model = None
        self._model_mtime: Optional[float] = None
        self._state = LearnerState()

        self._restore_state()
        self._load_model()
        self._configure_for_online()

    # ---------------------------------------------------------------- model io
    def _load_model(self) -> None:
        import tensorflow as tf
        if not self._model_path.exists():
            raise FileNotFoundError(f"model not found at {self._model_path}")
        try:
            self._model = tf.keras.models.load_model(self._model_path)
            self._model_mtime = self._model_path.stat().st_mtime
            self._state.last_reload_ts = time.time()
            print(f"[online_learner] loaded model from {self._model_path}")
        except Exception as e:
            raise RuntimeError(f"failed to load model: {e}")

    def _configure_for_online(self) -> None:
        """Recompile with a much smaller LR for safe online updates."""
        import tensorflow as tf
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._online_lr),
            loss="mse", metrics=["mae"],
        )

    def reload_if_updated(self) -> bool:
        """If the model file on disk is newer than our loaded copy, swap to it.

        Atomic at the file-replace level: a concurrent prediction either
        sees the old model or the new model, never a half-written one
        (the worker uses os.replace which is atomic on POSIX & Windows).
        """
        try:
            current_mtime = self._model_path.stat().st_mtime
        except FileNotFoundError:
            return False
        if self._model_mtime is None or current_mtime > self._model_mtime + 0.01:
            with self._lock:
                self._load_model()
                self._configure_for_online()
                self._state.samples_since_last_retrain = 0
                self._state.model_version += 1
                self._losses.clear()  # fresh baseline post-retrain
                self._save_state()
            return True
        return False

    # ---------------------------------------------------------------- predict
    def predict(self, window: np.ndarray) -> np.ndarray:
        """Predict the next vector from a (W, F) or (1, W, F) input."""
        if window.ndim == 2:
            window = window[None, ...]
        with self._lock:
            pred = self._model.predict(window, verbose=0)
        return pred[0]

    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        with self._lock:
            return self._model.predict(windows, verbose=0)

    # ---------------------------------------------------------------- online update
    def partial_fit(self, window: np.ndarray, target: np.ndarray) -> float:
        """One-batch online update. Returns the loss on this single sample.

        window  : shape (W, F)
        target  : shape (F,)
        """
        if window.ndim != 2 or target.ndim != 1:
            raise ValueError(
                f"expected (W, F) + (F,); got {window.shape} + {target.shape}"
            )
        X = window[None, ...]
        y = target[None, ...]

        with self._lock:
            # train_on_batch returns the (already-aggregated) loss
            metrics = self._model.train_on_batch(X, y, return_dict=True)
            loss = float(metrics.get("loss", float("nan")))
            self._losses.append(loss)
            self._state.samples_seen_online += 1
            self._state.samples_since_last_retrain += 1
            self._state.last_online_loss = loss
            self._state.last_partial_fit_ts = time.time()
            if self._losses:
                self._state.rolling_loss_mean = float(np.mean(self._losses))
            if self._state.baseline_loss is None and \
                    len(self._losses) >= self._loss_window:
                # establish baseline from the first full window
                self._state.baseline_loss = self._state.rolling_loss_mean
            self._update_drift()

            # Persist state every N samples so the out-of-process retrain worker
            # can read accurate counters. Writing every sample is wasteful;
            # writing every 10 keeps the worker within ~10 samples of truth.
            if self._state.samples_seen_online % 10 == 0 or self._state.drift_flagged:
                self._save_state()

        return loss

    def _update_drift(self) -> None:
        """Cheap rolling-mean drift detector. Production systems would use
        ADWIN or Page-Hinkley; this is sufficient for the prototype."""
        if self._state.baseline_loss is None or self._state.baseline_loss <= 0:
            self._state.drift_score = 0.0
            self._state.drift_flagged = False
            return
        ratio = self._state.rolling_loss_mean / self._state.baseline_loss
        self._state.drift_score = float(ratio)
        self._state.drift_flagged = ratio > self._drift_multiplier

    # ---------------------------------------------------------------- retrain trigger
    def should_retrain(self) -> tuple[bool, str]:
        """Return (yes_no, reason)."""
        with self._lock:
            if self._state.samples_since_last_retrain >= self._retrain_threshold:
                return True, f"sample-budget ({self._state.samples_since_last_retrain})"
            if self._state.drift_flagged:
                return True, f"drift score {self._state.drift_score:.2f}"
            return False, ""

    # ---------------------------------------------------------------- versioned save
    def save_model_atomic(self, path: Optional[Path] = None) -> Path:
        """Save the current model atomically. Used by retrain_worker."""
        target = Path(path) if path else self._model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with self._lock:
            self._model.save(tmp)
        os.replace(tmp, target)
        # update our internal mtime so reload_if_updated doesn't reload our own write
        self._model_mtime = target.stat().st_mtime
        return target

    # ---------------------------------------------------------------- state io
    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(asdict(self._state), f, indent=2, default=str)
            os.replace(tmp, self._state_path)
        except Exception as e:
            print(f"[online_learner] state save failed: {e}")

    def _restore_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)
        except Exception as e:
            print(f"[online_learner] state restore failed: {e}")

    def state(self) -> LearnerState:
        with self._lock:
            return LearnerState(**asdict(self._state))

    def state_dict(self) -> dict:
        return asdict(self.state())

    # ---------------------------------------------------------------- scaling helper
    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
