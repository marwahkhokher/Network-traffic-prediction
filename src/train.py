"""
train.py
--------
End-to-end training orchestrator.

Loads the traffic matrix CSV, normalises (/max), builds sliding windows,
fits the LSTM + the baselines, computes MSE on a held-out test split,
and saves:

  results/
    lstm_model.keras        # trained Keras model
    lstm_history.json       # training history (loss curves)
    metrics.json            # MSE/MAE for every model
    predictions.npz         # raw predictions for analysis
    comparison.png          # bar chart MSE comparison (paper Fig. 8 style)
    loss_curves.png         # training/validation loss

Usage:
  python src/train.py --data data/traffic_matrix.csv --window 12 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# allow running from project root or from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import (
    build_windows,
    load_traffic_csv,
    train_test_split_timeseries,
)
from lstm_model import LSTMConfig, train_lstm
from baselines import (
    ARMABaseline,
    FFNNBaseline,
    LinearRegressionBaseline,
    RandomForestBaseline,
)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--window", type=int, default=12,
                    help="W in the paper; number of past slots (default 12 = 3h @ 15min)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden", type=int, nargs="+", default=[300],
                    help="LSTM hidden sizes per stacked layer")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--skip_arma", action="store_true",
                    help="Skip ARMA (slow on long series)")
    ap.add_argument("--skip_rf", action="store_true", help="Skip Random Forest")
    ap.add_argument("--skip_ffnn", action="store_true", help="Skip FFNN")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load + preprocess ------------------------------------------------
    print(f"[train] loading {args.data}")
    data = load_traffic_csv(args.data)
    print(f"[train] T = {data.raw.shape[0]}, N = {data.n_nodes}, "
          f"features = {data.raw.shape[1]}, scale = {data.scale:.4f}")

    X, Y = build_windows(data.scaled, window=args.window)
    X_tr, Y_tr, X_te, Y_te = train_test_split_timeseries(X, Y, args.test_frac)
    print(f"[train] train={X_tr.shape[0]}, test={X_te.shape[0]}, W={args.window}")

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {"Y_true": Y_te}

    # ---- LSTM -------------------------------------------------------------
    print("\n[train] === LSTM ===")
    cfg = LSTMConfig(
        window=args.window,
        n_features=data.raw.shape[1],
        hidden_sizes=list(args.hidden),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    lstm_model, hist = train_lstm(cfg, X_tr, Y_tr, X_te, Y_te, verbose=2)
    Y_hat_lstm = lstm_model.predict(X_te, verbose=0)
    metrics["LSTM"] = {"MSE": mse(Y_te, Y_hat_lstm), "MAE": mae(Y_te, Y_hat_lstm)}
    predictions["LSTM"] = Y_hat_lstm
    lstm_model.save(out / "lstm_model.keras")
    with open(out / "lstm_history.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.items()}, f, indent=2)

    # loss-curves plot
    plt.figure(figsize=(7, 4))
    plt.plot(hist["loss"], label="train")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("LSTM training curves")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out / "loss_curves.png", dpi=120)
    plt.close()

    # ---- Baselines --------------------------------------------------------
    print("\n[train] === Linear Regression ===")
    lr = LinearRegressionBaseline().fit(X_tr, Y_tr)
    Y_hat_lr = lr.predict(X_te)
    metrics["LinearRegression"] = {"MSE": mse(Y_te, Y_hat_lr), "MAE": mae(Y_te, Y_hat_lr)}
    predictions["LinearRegression"] = Y_hat_lr

    if not args.skip_rf:
        print("\n[train] === Random Forest ===")
        rf = RandomForestBaseline().fit(X_tr, Y_tr)
        Y_hat_rf = rf.predict(X_te)
        metrics["RandomForest"] = {"MSE": mse(Y_te, Y_hat_rf), "MAE": mae(Y_te, Y_hat_rf)}
        predictions["RandomForest"] = Y_hat_rf

    if not args.skip_ffnn:
        print("\n[train] === FFNN ===")
        ffnn = FFNNBaseline(epochs=min(40, args.epochs)).fit(X_tr, Y_tr)
        Y_hat_ff = ffnn.predict(X_te)
        metrics["FFNN"] = {"MSE": mse(Y_te, Y_hat_ff), "MAE": mae(Y_te, Y_hat_ff)}
        predictions["FFNN"] = Y_hat_ff

    if not args.skip_arma:
        print("\n[train] === ARMA(2,1) ===")
        try:
            arma = ARMABaseline(order=(2, 0, 1), fit_per_od=False).fit(X_tr, Y_tr)
            Y_hat_arma = arma.predict(X_te)
            metrics["ARMA"] = {"MSE": mse(Y_te, Y_hat_arma), "MAE": mae(Y_te, Y_hat_arma)}
            predictions["ARMA"] = Y_hat_arma
        except Exception as e:
            print(f"[train] ARMA failed ({e}); skipping.")

    # ---- Save -------------------------------------------------------------
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(out / "predictions.npz", **predictions)

    print("\n[train] === Summary ===")
    for name, m in metrics.items():
        print(f"  {name:<18}  MSE = {m['MSE']:.6f}   MAE = {m['MAE']:.6f}")

    # ---- Comparison plot (paper Fig. 8 style) -----------------------------
    order = [n for n in ["ARMA", "LinearRegression", "RandomForest", "FFNN", "LSTM"]
             if n in metrics]
    plt.figure(figsize=(7, 4.2))
    vals = [metrics[n]["MSE"] for n in order]
    bars = plt.bar(order, vals, color=["#c44", "#e9a", "#6c6", "#69c", "#34a"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel("MSE (scaled traffic)")
    plt.title("Comparison of prediction methods")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=120)
    plt.close()

    print(f"\n[train] all artifacts saved to {out.resolve()}")


if __name__ == "__main__":
    main()
