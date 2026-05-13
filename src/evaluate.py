"""
evaluate.py
-----------
Post-hoc analysis on the saved predictions.

  - Reproduces the Fig. 8-style MSE bar chart from the paper.
  - Picks three OD flows (low / medium / high volume) and plots
    predicted vs actual over the test horizon for each model.
  - Emits per-OD MSE heatmap so operators can see which flows
    are hardest to predict.

Usage:
  python src/evaluate.py --results_dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    d = Path(args.results_dir)
    with open(d / "metrics.json") as f:
        metrics = json.load(f)
    preds = np.load(d / "predictions.npz")
    Y_true = preds["Y_true"]

    # ---- 1. Replicate Fig. 8 bar chart with log scale (easier to read) ----
    order = [n for n in ["ARMA", "LinearRegression", "RandomForest", "FFNN", "LSTM"]
             if n in metrics]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    vals = [metrics[n]["MSE"] for n in order]
    ax1.bar(order, vals, color=["#c44", "#e9a", "#6c6", "#69c", "#34a"])
    ax1.set_ylabel("MSE"); ax1.set_title("Linear scale")
    ax1.grid(axis="y", alpha=0.3)
    for x, v in enumerate(vals):
        ax1.text(x, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    ax2.bar(order, vals, color=["#c44", "#e9a", "#6c6", "#69c", "#34a"])
    ax2.set_yscale("log"); ax2.set_ylabel("MSE (log)"); ax2.set_title("Log scale")
    ax2.grid(axis="y", alpha=0.3, which="both")

    plt.suptitle("Prediction methods — MSE comparison (paper Fig. 8 style)")
    plt.tight_layout()
    plt.savefig(d / "comparison_logscale.png", dpi=120)
    plt.close()

    # ---- 2. OD-flow predicted-vs-actual plots -----------------------------
    model_names = [n for n in order if n in preds.files]
    mean_flow = Y_true.mean(axis=0)
    # pick low/med/high flow indices by mean volume
    low = int(np.argmin(mean_flow))
    high = int(np.argmax(mean_flow))
    med = int(np.argsort(mean_flow)[len(mean_flow) // 2])
    picked = {"low volume OD": low, "median volume OD": med, "high volume OD": high}

    fig, axes = plt.subplots(len(picked), 1, figsize=(11, 3 * len(picked)), sharex=True)
    if len(picked) == 1:
        axes = [axes]
    for ax, (label, idx) in zip(axes, picked.items()):
        ax.plot(Y_true[:, idx], label="actual", linewidth=2, color="black")
        for name in model_names:
            ax.plot(preds[name][:, idx], label=name, alpha=0.75, linewidth=1)
        ax.set_title(f"{label} (index {idx}, mean={mean_flow[idx]:.3f})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("test time step")
    plt.tight_layout()
    plt.savefig(d / "od_flow_predictions.png", dpi=120)
    plt.close()

    # ---- 3. Per-OD MSE heatmap for LSTM -----------------------------------
    if "LSTM" in preds.files:
        n_features = Y_true.shape[1]
        n_nodes = int(round(np.sqrt(n_features)))
        per_od = np.mean((Y_true - preds["LSTM"]) ** 2, axis=0).reshape(n_nodes, n_nodes)

        plt.figure(figsize=(6.5, 5.5))
        im = plt.imshow(per_od, cmap="magma")
        plt.colorbar(im, label="MSE (LSTM)")
        plt.xlabel("destination node j"); plt.ylabel("origin node i")
        plt.title("LSTM per-OD MSE heatmap")
        plt.tight_layout()
        plt.savefig(d / "lstm_per_od_mse.png", dpi=120)
        plt.close()

    # ---- 4. Printed summary table ----------------------------------------
    print("\n=== Final MSE/MAE Summary ===")
    print(f"{'Method':<20} {'MSE':>12} {'MAE':>12} {'vs LSTM':>12}")
    lstm_mse = metrics.get("LSTM", {}).get("MSE", None)
    for name in order:
        m = metrics[name]
        ratio = (m["MSE"] / lstm_mse) if lstm_mse else float("nan")
        print(f"{name:<20} {m['MSE']:>12.6f} {m['MAE']:>12.6f} {ratio:>11.1f}x")

    print(f"\nArtifacts saved to {d.resolve()}")


if __name__ == "__main__":
    main()
