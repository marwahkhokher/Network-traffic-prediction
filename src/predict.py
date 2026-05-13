"""
predict.py
----------
Inference CLI. Loads the trained LSTM from results/lstm_model.keras
and the traffic CSV, predicts the next H slots, and saves to CSV.

Usage:
  python src/predict.py --data data/traffic_matrix.csv \
        --model results/lstm_model.keras --horizon 4 \
        --out results/forecast.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_traffic_csv


def rolling_forecast(model: tf.keras.Model, history: np.ndarray, horizon: int) -> np.ndarray:
    """Autoregressively forecast `horizon` steps from the tail of `history`.

    history : (T, F) scaled traffic vectors
    """
    W = model.input_shape[1]
    window = history[-W:].copy()  # (W, F)
    out = np.empty((horizon, history.shape[1]), dtype=np.float32)
    for h in range(horizon):
        pred = model.predict(window[None, ...], verbose=0)[0]
        out[h] = pred
        # slide the window
        window = np.vstack([window[1:], pred[None, :]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="results/lstm_model.keras")
    ap.add_argument("--horizon", type=int, default=4,
                    help="number of future timeslots to predict")
    ap.add_argument("--out", default="results/forecast.csv")
    args = ap.parse_args()

    data = load_traffic_csv(args.data)
    model = tf.keras.models.load_model(args.model)

    scaled_future = rolling_forecast(model, data.scaled, args.horizon)
    raw_future = scaled_future * data.scale  # inverse-transform

    n_features = raw_future.shape[1]
    n_nodes = int(round(np.sqrt(n_features)))
    cols = [f"y_{i}_{j}" for i in range(n_nodes) for j in range(n_nodes)]

    df = pd.DataFrame(raw_future, columns=cols)
    df.insert(0, "step_ahead", np.arange(1, args.horizon + 1))
    # if the input had timestamps, extrapolate them
    if data.timestamps is not None:
        dt = data.timestamps.iloc[-1] - data.timestamps.iloc[-2]
        future_ts = [data.timestamps.iloc[-1] + (h + 1) * dt for h in range(args.horizon)]
        df.insert(1, "timestamp", future_ts)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[predict] wrote {args.out}  ({df.shape[0]} forecast rows)")

    # summarise congestion risk per destination
    per_dest = raw_future.reshape(args.horizon, n_nodes, n_nodes).sum(axis=1)
    threshold = np.quantile(data.raw.reshape(-1, n_nodes, n_nodes).sum(axis=1), 0.95)
    flags = per_dest > threshold
    if flags.any():
        print(f"[predict] ⚠ predicted congestion on these (step, dest) pairs "
              f"(> 95th percentile historical load):")
        for t, d in zip(*np.where(flags)):
            print(f"           step +{t+1}, node {d}: "
                  f"predicted inbound = {per_dest[t, d]:.2f}")
    else:
        print("[predict] no congestion flags in forecast horizon")


if __name__ == "__main__":
    main()
