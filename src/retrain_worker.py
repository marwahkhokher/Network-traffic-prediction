"""
retrain_worker.py
-----------------
Background process that watches the streaming buffer and triggers a
full retrain whenever the trigger conditions fire:

  - N new samples have arrived since the last retrain, OR
  - online drift score exceeds the configured multiplier.

When it retrains, it:
  1. Reads the most recent K samples from the buffer's disk snapshot.
  2. Builds sliding windows.
  3. Continues training from the current model (warm-start).
  4. Atomically replaces the model file. The streaming service detects
     the new file via mtime and reloads on its next prediction.

The worker is a separate process from the FastAPI service for two
reasons:
  - Retraining blocks (it's compute-heavy); we don't want to stall
    real-time predictions while it runs.
  - If retraining ever crashes (OOM, bad data), the serving process
    keeps running with the last-known-good model.

Usage:
  python src/retrain_worker.py \\
      --buffer-snapshot results/streaming_buffer.pkl \\
      --model results/lstm_streaming.keras \\
      --window 12 \\
      --check-interval 30
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# allow running as `python src/retrain_worker.py ...`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class WorkerConfig:
    buffer_snapshot: Path
    model_path: Path
    state_path: Path
    window: int = 12
    min_samples_to_retrain: int = 500
    max_samples_per_retrain: int = 4000
    epochs_per_retrain: int = 5
    batch_size: int = 32
    retrain_lr: float = 1e-4
    check_interval_seconds: float = 30.0
    val_frac: float = 0.10


# ---------------------------------------------------------------------- I/O
def load_buffer_snapshot(path: Path) -> tuple[np.ndarray, list[float]]:
    """Read the pickled buffer snapshot the streaming service maintains."""
    if not path.exists():
        return np.empty((0, 0), dtype=np.float32), []
    try:
        with open(path, "rb") as f:
            snapshot = pickle.load(f)
        if not snapshot:
            return np.empty((0, 0), dtype=np.float32), []
        ts = [s[0] for s in snapshot]
        vecs = np.stack([s[1] for s in snapshot]).astype(np.float32)
        return vecs, ts
    except Exception as e:
        print(f"[retrain_worker] failed to load snapshot: {e}")
        return np.empty((0, 0), dtype=np.float32), []


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------- windowing
def build_windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    T, F = series.shape
    if T <= window:
        return np.empty((0, window, F), dtype=np.float32), \
               np.empty((0, F), dtype=np.float32)
    n = T - window
    X = np.empty((n, window, F), dtype=np.float32)
    Y = np.empty((n, F), dtype=np.float32)
    for i in range(n):
        X[i] = series[i: i + window]
        Y[i] = series[i + window]
    return X, Y


# ---------------------------------------------------------------------- core retrain
def run_one_retrain(cfg: WorkerConfig) -> Optional[dict]:
    """Run a single retrain pass. Returns a metrics dict, or None if skipped."""
    vecs, ts = load_buffer_snapshot(cfg.buffer_snapshot)
    if vecs.size == 0:
        print(f"[retrain_worker] skip: buffer snapshot empty "
              f"(no flush yet? check {cfg.buffer_snapshot})")
        return None

    # Use only the most recent slice
    if vecs.shape[0] > cfg.max_samples_per_retrain:
        vecs = vecs[-cfg.max_samples_per_retrain:]

    X, Y = build_windows(vecs, cfg.window)
    if X.shape[0] < 50:
        print(f"[retrain_worker] skip: only {X.shape[0]} windows "
              f"(have {vecs.shape[0]} samples, need at least {50 + cfg.window})")
        return None

    # temporal train/val split
    n = X.shape[0]
    n_val = max(1, int(n * cfg.val_frac))
    n_train = n - n_val
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    import tensorflow as tf
    model = tf.keras.models.load_model(cfg.model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.retrain_lr),
        loss="mse", metrics=["mae"],
    )

    t0 = time.time()
    history = model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        epochs=cfg.epochs_per_retrain,
        batch_size=cfg.batch_size,
        verbose=0,
    )
    train_secs = time.time() - t0

    # atomic save
    tmp_path = cfg.model_path.with_suffix(cfg.model_path.suffix + ".tmp")
    model.save(tmp_path)
    os.replace(tmp_path, cfg.model_path)

    val_loss = float(history.history.get("val_loss", [float("nan")])[-1])
    train_loss = float(history.history.get("loss", [float("nan")])[-1])
    metrics = {
        "timestamp": time.time(),
        "samples_used": int(vecs.shape[0]),
        "train_windows": int(X_tr.shape[0]),
        "val_windows": int(X_val.shape[0]),
        "epochs": cfg.epochs_per_retrain,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_seconds": round(train_secs, 2),
        "model_path": str(cfg.model_path),
    }
    print(f"[retrain_worker] retrain done: val_loss={val_loss:.6f}  "
          f"({metrics['train_windows']} windows, {train_secs:.1f}s)")

    # Append to a rolling history file for the dashboard
    history_path = cfg.model_path.parent / "retrain_history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")
    return metrics


# ---------------------------------------------------------------------- loop
def should_trigger(cfg: WorkerConfig) -> tuple[bool, str]:
    """Decide whether to fire a retrain based on the learner's state file."""
    state = load_state(cfg.state_path)
    samples_since = int(state.get("samples_since_last_retrain", 0))
    drift = bool(state.get("drift_flagged", False))

    if samples_since >= cfg.min_samples_to_retrain:
        return True, f"sample-budget ({samples_since})"
    if drift:
        return True, "drift flag"
    return False, ""


def run_loop(cfg: WorkerConfig) -> None:
    print(f"[retrain_worker] started "
          f"(buffer={cfg.buffer_snapshot}  model={cfg.model_path}  "
          f"check_every={cfg.check_interval_seconds}s)")

    stop = {"flag": False}
    last_trigger_count = -1  # avoid spamming on identical state

    def _handle_sigterm(*_):
        print("[retrain_worker] SIGTERM received, stopping after current cycle")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    while not stop["flag"]:
        try:
            trigger, reason = should_trigger(cfg)
            current_state = load_state(cfg.state_path)
            current_count = int(current_state.get("samples_since_last_retrain", 0))

            if trigger:
                # Only log when the trigger state actually changed
                if current_count != last_trigger_count:
                    print(f"[retrain_worker] trigger fired: {reason}")
                    last_trigger_count = current_count

                metrics = run_one_retrain(cfg)
                if metrics:
                    # successful retrain → reset state counters
                    state = load_state(cfg.state_path)
                    state["samples_since_last_retrain"] = 0
                    state["drift_flagged"] = False
                    state["model_version"] = int(state.get("model_version", 0)) + 1
                    state["last_worker_retrain_ts"] = time.time()
                    save_state(cfg.state_path, state)
                    last_trigger_count = -1
                # else: skip reason was printed by run_one_retrain; keep waiting
        except Exception as e:
            print(f"[retrain_worker] cycle error: {e}")

        # responsive sleep
        for _ in range(int(cfg.check_interval_seconds)):
            if stop["flag"]:
                break
            time.sleep(1.0)

    print("[retrain_worker] exited")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buffer-snapshot", default="results/streaming_buffer.pkl")
    ap.add_argument("--model", default="results/lstm_streaming.keras",
                    help="Path of the live model (same path the streaming "
                         "service reads from; we replace it atomically)")
    ap.add_argument("--state", default="results/lstm_streaming.state.json")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--min-samples", type=int, default=500,
                    help="trigger retrain after this many new samples")
    ap.add_argument("--max-samples", type=int, default=4000,
                    help="cap on samples used in a single retrain")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--retrain-lr", type=float, default=1e-4)
    ap.add_argument("--check-interval", type=float, default=30.0)
    args = ap.parse_args()

    cfg = WorkerConfig(
        buffer_snapshot=Path(args.buffer_snapshot),
        model_path=Path(args.model),
        state_path=Path(args.state),
        window=args.window,
        min_samples_to_retrain=args.min_samples,
        max_samples_per_retrain=args.max_samples,
        epochs_per_retrain=args.epochs,
        batch_size=args.batch_size,
        retrain_lr=args.retrain_lr,
        check_interval_seconds=args.check_interval,
    )
    run_loop(cfg)


if __name__ == "__main__":
    main()
