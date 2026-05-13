"""
make_comparison_plots.py
------------------------
Generates polished, publication-quality figures comparing all models.

Outputs to results/plots/:
  01_mse_comparison.png         - clean bar chart, log scale, with ratio annotations
  02_improvement_over_baselines.png - LSTM's relative advantage chart
  03_prediction_traces.png      - actual vs each model on a high-volume OD pair
  04_error_distribution.png     - violin/box plot of per-prediction errors
  05_per_od_mse_heatmap.png     - which OD pairs are hardest for the LSTM
  06_lstm_training_curve.png    - clean train+val loss curve
  07_closed_loop_comparison.png - reactive vs proactive QoS bar chart
  08_inbound_timeline.png       - inbound traffic over time, with capacity line and overflow zones
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Project palette — colorblind-friendly, professional
COLORS = {
    "Persistence":      "#9ca3af",   # grey
    "ARMA":             "#dc2626",   # red
    "LinearRegression": "#f59e0b",   # amber
    "RandomForest":     "#10b981",   # green
    "FFNN":             "#3b82f6",   # blue
    "LSTM":             "#1e3a8a",   # navy (the hero)
    "HistoricalMean":   "#cbd5e1",
}

LABELS = {
    "Persistence":      "Persistence\n(naive)",
    "ARMA":             "ARMA(2,1)",
    "LinearRegression": "Linear\nRegression",
    "RandomForest":     "Random\nForest",
    "FFNN":             "FFNN",
    "LSTM":             "LSTM\n(this paper)",
    "HistoricalMean":   "Historical\nMean",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def order_methods(metrics: dict) -> list[str]:
    """Return methods worst-to-best by MSE."""
    return sorted(metrics.keys(), key=lambda k: -metrics[k]["MSE"])


# --------------------------------------------------------------- 01
def plot_mse_comparison(metrics: dict, out_path: Path):
    methods = order_methods(metrics)
    mses = [metrics[m]["MSE"] for m in methods]
    colors = [COLORS.get(m, "#888") for m in methods]
    labels = [LABELS.get(m, m) for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Linear scale
    bars = axes[0].bar(labels, mses, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("MSE (lower is better)")
    axes[0].set_title("Linear scale", fontsize=12)
    for b, v in zip(bars, mses):
        axes[0].text(b.get_x() + b.get_width()/2, v + max(mses)*0.01,
                     f"{v:.5f}", ha="center", va="bottom", fontsize=9)

    # Log scale with annotation
    bars2 = axes[1].bar(labels, mses, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("MSE (log scale)")
    axes[1].set_title("Log scale", fontsize=12)
    lstm_mse = metrics.get("LSTM", {}).get("MSE", min(mses))
    for b, v, m in zip(bars2, mses, methods):
        ratio = v / lstm_mse
        axes[1].text(b.get_x() + b.get_width()/2, v * 1.05,
                     f"{ratio:.2f}×" if m != "LSTM" else "1.00×\n(best)",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Model comparison — Mean Squared Error on test set",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path.name}")


# --------------------------------------------------------------- 02
def plot_improvement_over_baselines(metrics: dict, out_path: Path):
    if "LSTM" not in metrics:
        return
    lstm_mse = metrics["LSTM"]["MSE"]
    others = [m for m in metrics if m != "LSTM"]
    others.sort(key=lambda k: -metrics[k]["MSE"])

    improvements = [(metrics[m]["MSE"] - lstm_mse) / metrics[m]["MSE"] * 100 for m in others]
    colors = [COLORS.get(m, "#888") for m in others]
    labels = [LABELS.get(m, m).replace("\n", " ") for m in others]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(labels, improvements, color=colors, edgecolor="white", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LSTM error reduction vs baseline (%)")
    ax.set_title("LSTM advantage over each baseline", fontsize=14, fontweight="bold")

    for b, v in zip(bars, improvements):
        x_text = v + (1 if v >= 0 else -1)
        ha = "left" if v >= 0 else "right"
        ax.text(x_text, b.get_y() + b.get_height()/2, f"{v:+.1f}%",
                va="center", ha=ha, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path.name}")


# --------------------------------------------------------------- 03
def plot_prediction_traces(predictions: dict, out_path: Path):
    Y_true = predictions["Y_true"]
    mean_flow = Y_true.mean(axis=0)
    high_idx = int(np.argmax(mean_flow))
    n_nodes = int(round(np.sqrt(Y_true.shape[1])))
    i, j = divmod(high_idx, n_nodes)

    methods = [m for m in ["ARMA", "LinearRegression", "RandomForest", "FFNN", "LSTM"]
               if m in predictions.files]

    fig, axes = plt.subplots(len(methods), 1, figsize=(13, 2.0 * len(methods)),
                              sharex=True, sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, m in zip(axes, methods):
        ax.plot(Y_true[:, high_idx], color="black", linewidth=2.0, label="actual", alpha=0.9)
        ax.plot(predictions[m][:, high_idx], color=COLORS.get(m, "#888"),
                linewidth=1.4, label=m, alpha=0.85)
        err = np.abs(Y_true[:, high_idx] - predictions[m][:, high_idx])
        ax.fill_between(np.arange(len(err)), Y_true[:, high_idx] - err/2,
                        Y_true[:, high_idx] + err/2,
                        color=COLORS.get(m, "#888"), alpha=0.15)
        ax.set_ylabel(LABELS.get(m, m).replace("\n", " "), fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("test time step (15-min slots)")
    fig.suptitle(f"Prediction traces — high-volume OD pair (node {i} → node {j})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path.name}")


# --------------------------------------------------------------- 04
def plot_error_distribution(predictions: dict, out_path: Path):
    Y_true = predictions["Y_true"]
    methods = [m for m in ["Persistence", "ARMA", "LinearRegression",
                            "RandomForest", "FFNN", "LSTM"]
               if m in predictions.files]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    data = []
    for m in methods:
        errs = (Y_true - predictions[m]).flatten()
        # Keep only non-zero pairs to avoid the zero-diagonal bias
        errs = errs[Y_true.flatten() > 0]
        data.append(errs)

    parts = ax.violinplot(data, positions=range(len(methods)),
                          showmedians=True, widths=0.85)
    for pc, m in zip(parts["bodies"], methods):
        pc.set_facecolor(COLORS.get(m, "#888"))
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS.get(m, m).replace("\n", " ") for m in methods])
    ax.set_ylabel("Per-prediction error  (truth − predicted)")
    ax.set_title("Error distribution per model — narrower & tighter is better",
                 fontsize=14, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path.name}")


# --------------------------------------------------------------- 05
def plot_per_od_heatmap(predictions: dict, out_path: Path):
    if "LSTM" not in predictions.files:
        return
    Y_true = predictions["Y_true"]
    n_nodes = int(round(np.sqrt(Y_true.shape[1])))
    per_od = np.mean((Y_true - predictions["LSTM"])**2, axis=0).reshape(n_nodes, n_nodes)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(per_od, cmap="magma", aspect="auto")
    ax.set_xlabel("Destination node j")
    ax.set_ylabel("Origin node i")
    ax.set_title("LSTM per-OD MSE — which OD pairs are hardest to predict?",
                 fontsize=13, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MSE")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path.name}")


# --------------------------------------------------------------- 06
def plot_lstm_training_curve(history_path: Path, out_path: Path):
    with open(history_path) as f:
        hist = json.load(f)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(hist["loss"], color=COLORS["LSTM"], linewidth=2, label="train loss")
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], color="#dc2626", linewidth=2, label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("LSTM training curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path.name}")


# --------------------------------------------------------------- 07
def plot_closed_loop(out_path: Path, results_dir: Path):
    """Reactive vs proactive QoS bar chart, computed live."""
    sys.path.insert(0, "src")
    from data_preprocessing import load_traffic_csv, build_windows, train_test_split_timeseries
    from closed_loop_demo import run_closed_loop, ClosedLoopConfig

    d = load_traffic_csv("data/traffic_matrix.csv")
    X, Y = build_windows(d.scaled, window=12)
    Xtr, Ytr, Xte, Yte = train_test_split_timeseries(X, Y, 0.15)

    preds = np.load(results_dir / "predictions.npz")
    cfg = ClosedLoopConfig()

    # Run closed-loop using each model's predictions
    methods = [m for m in ["Persistence", "ARMA", "LinearRegression", "FFNN", "LSTM"]
               if m in preds.files]
    results = {}
    for m in methods:
        r = run_closed_loop(Yte, preds[m], d.n_nodes, cfg, Ytr)
        results[m] = r

    # Save results JSON
    closed_loop_summary = {
        m: {
            "reactive_events": r.reactive_events,
            "proactive_events": r.proactive_events,
            "event_reduction_pct": r.event_reduction_pct,
            "reactive_overflow": r.reactive_total_overflow,
            "proactive_overflow": r.proactive_total_overflow,
            "overflow_reduction_pct": r.overflow_reduction_pct,
            "throttle_actions": r.proactive_throttle_actions,
        }
        for m, r in results.items()
    }
    with open(results_dir / "closed_loop_metrics.json", "w") as f:
        json.dump(closed_loop_summary, f, indent=2)

    # Bar chart: each method's congestion-reduction percentage
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    labels = [LABELS.get(m, m).replace("\n", " ") for m in methods]
    colors = [COLORS.get(m, "#888") for m in methods]

    # Left: congestion event reduction
    reductions = [results[m].event_reduction_pct for m in methods]
    bars = axes[0].bar(labels, reductions, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("Congestion events reduced (%)")
    axes[0].set_title("Closed-loop QoS effectiveness — fewer congestion events",
                      fontsize=12, fontweight="bold")
    axes[0].axhline(0, color="black", linewidth=0.6)
    for b, v in zip(bars, reductions):
        axes[0].text(b.get_x() + b.get_width()/2, v + (1 if v >= 0 else -2),
                     f"{v:+.1f}%", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")

    # Right: overflow reduction
    overflows = [results[m].overflow_reduction_pct for m in methods]
    bars = axes[1].bar(labels, overflows, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_ylabel("Total overflow volume reduced (%)")
    axes[1].set_title("Closed-loop QoS effectiveness — less spillover traffic",
                      fontsize=12, fontweight="bold")
    axes[1].axhline(0, color="black", linewidth=0.6)
    for b, v in zip(bars, overflows):
        axes[1].text(b.get_x() + b.get_width()/2, v + (1 if v >= 0 else -2),
                     f"{v:+.1f}%", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")

    fig.suptitle("Predictor → QoS controller — operational impact of each forecasting method",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path.name}")
    return results


# --------------------------------------------------------------- 08
def plot_inbound_timeline(closed_loop_results: dict, out_path: Path):
    """Show actual inbound traffic over time on a heavily-loaded node,
    with capacity line, overflow zones, and the LSTM-throttled version."""
    if not closed_loop_results or "LSTM" not in closed_loop_results:
        return
    r = closed_loop_results["LSTM"]
    # Pick the node with most reactive overflow
    node_overflow = r.reactive_overflow.sum(axis=0)
    busy_node = int(np.argmax(node_overflow))

    fig, ax = plt.subplots(figsize=(13, 5.2))
    t = np.arange(r.reactive_inbound.shape[0])
    cap = r.capacities[busy_node]

    ax.plot(t, r.reactive_inbound[:, busy_node], color="#dc2626", linewidth=1.6,
            label="inbound traffic — reactive (no LSTM)", alpha=0.85)
    ax.plot(t, r.proactive_inbound[:, busy_node], color=COLORS["LSTM"], linewidth=1.6,
            label="inbound traffic — LSTM-guided QoS", alpha=0.85)

    ax.axhline(cap, color="black", linewidth=1.2, linestyle="--",
               label=f"node capacity = {cap:.2f}")

    # Highlight overflow zones
    over_a = r.reactive_inbound[:, busy_node] > cap
    over_b = r.proactive_inbound[:, busy_node] > cap
    ax.fill_between(t, cap, r.reactive_inbound[:, busy_node],
                    where=over_a, color="#dc2626", alpha=0.20,
                    label="overflow (reactive)")
    ax.fill_between(t, cap, r.proactive_inbound[:, busy_node],
                    where=over_b, color=COLORS["LSTM"], alpha=0.20,
                    label="overflow (proactive — much smaller)")

    ax.set_xlabel("Test time step (15-min slots)")
    ax.set_ylabel("Inbound traffic (scaled)")
    ax.set_title(f"Inbound traffic at busy node {busy_node} — reactive vs LSTM-guided QoS",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path.name}")


def plot_qos_strategies(out_path: Path, results_dir: Path):
    """Compare the 5 QoS strategies, all using LSTM predictions."""
    qos_path = results_dir / "qos_strategies_metrics.json"
    if not qos_path.exists():
        return
    with open(qos_path) as f:
        data = json.load(f)

    # Order by event reduction
    order = sorted(data.keys(), key=lambda k: -data[k]["event_reduction_pct"])

    strat_colors = {
        "Hybrid":     "#1e3a8a",
        "RateLimit":  "#3b82f6",
        "Prioritize": "#10b981",
        "Reroute":    "#f59e0b",
        "NoAction":   "#9ca3af",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: event reduction
    vals = [data[s]["event_reduction_pct"] for s in order]
    colors = [strat_colors.get(s, "#888") for s in order]
    bars = axes[0].bar(order, vals, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Congestion events reduced (%)")
    axes[0].set_title("By QoS strategy", fontsize=12)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width()/2,
                     v + (1.5 if v >= 0 else -3),
                     f"{v:+.1f}%", ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontweight="bold", fontsize=10)

    # Right: overflow reduction
    vals2 = [data[s]["overflow_reduction_pct"] for s in order]
    bars2 = axes[1].bar(order, vals2, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Overflow volume reduced (%)")
    axes[1].set_title("By QoS strategy", fontsize=12)
    for b, v in zip(bars2, vals2):
        axes[1].text(b.get_x() + b.get_width()/2,
                     v + (1.5 if v >= 0 else -3),
                     f"{v:+.1f}%", ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontweight="bold", fontsize=10)

    fig.suptitle("QoS strategies compared — all using LSTM predictions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path.name}")


def plot_monte_carlo(out_path: Path, results_dir: Path):
    """Visualise Monte Carlo confidence intervals."""
    mc_path = results_dir / "monte_carlo.json"
    if not mc_path.exists():
        return
    with open(mc_path) as f:
        mc = json.load(f)
    summary = mc["summary"]
    if "strategies" not in summary:
        return

    order = sorted(summary["strategies"].keys(),
                   key=lambda s: -summary["strategies"][s]["event_reduction_pct"]["mean"])
    means = [summary["strategies"][s]["event_reduction_pct"]["mean"] for s in order]
    stds  = [summary["strategies"][s]["event_reduction_pct"]["std"]  for s in order]

    strat_colors = {
        "Hybrid":     "#1e3a8a",
        "RateLimit":  "#3b82f6",
        "Prioritize": "#10b981",
        "Reroute":    "#f59e0b",
        "NoAction":   "#9ca3af",
    }
    colors = [strat_colors.get(s, "#888") for s in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(order, means, yerr=stds, capsize=8,
                   color=colors, edgecolor="white", linewidth=1.5,
                   error_kw={"linewidth": 2, "ecolor": "black"})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Event reduction (%)")
    ax.set_title(f"Monte Carlo: event reduction across {summary['n_trials']} seeds  "
                 f"(mean ± std)", fontsize=14, fontweight="bold")
    for b, m, s in zip(bars, means, stds):
        y = m + s + 2 if m >= 0 else m - s - 4
        ax.text(b.get_x() + b.get_width()/2, y,
                f"{m:.1f}±{s:.1f}", ha="center",
                fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path.name}")


# --------------------------------------------------------------- main
def main():
    setup_style()
    results_dir = Path("results")
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "metrics.json") as f:
        metrics = json.load(f)
    predictions = np.load(results_dir / "predictions.npz")

    print(f"Generating plots → {out_dir}/")
    plot_mse_comparison(metrics, out_dir / "01_mse_comparison.png")
    plot_improvement_over_baselines(metrics, out_dir / "02_improvement_over_baselines.png")
    plot_prediction_traces(predictions, out_dir / "03_prediction_traces.png")
    plot_error_distribution(predictions, out_dir / "04_error_distribution.png")
    plot_per_od_heatmap(predictions, out_dir / "05_per_od_mse_heatmap.png")
    if (results_dir / "lstm_history.json").exists():
        plot_lstm_training_curve(results_dir / "lstm_history.json",
                                  out_dir / "06_lstm_training_curve.png")
    cl_results = plot_closed_loop(out_dir / "07_closed_loop_comparison.png", results_dir)
    plot_inbound_timeline(cl_results, out_dir / "08_inbound_timeline.png")
    plot_qos_strategies(out_dir / "09_qos_strategies.png", results_dir)
    plot_monte_carlo(out_dir / "10_monte_carlo.png", results_dir)

    print(f"\nAll plots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
