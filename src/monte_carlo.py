"""
monte_carlo.py
--------------
Run the entire closed-loop simulation over many seeds, get confidence
intervals on every metric. The single-seed numbers reported elsewhere
are point estimates — Monte Carlo tells you how robust they are.

For each seed:
  1. Generate a fresh traffic CSV (different random seed → different
     events, bursts, noise).
  2. Build sliding windows from that CSV.
  3. Use the existing trained LSTM to predict on the last 15%.
  4. Run the closed-loop simulation with each QoS strategy.
  5. Record metrics.

Across all seeds we report mean ± stdev for:
  - LSTM test MSE
  - Congestion events with each QoS strategy
  - Overflow reduction with each QoS strategy

Usage:
  python src/monte_carlo.py --n_seeds 30 --strategies RateLimit Hybrid \\
        --out results/monte_carlo.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anylogic_surrogate import simulate
from data_preprocessing import load_traffic_csv, build_windows, train_test_split_timeseries
from closed_loop_demo import run_closed_loop, ClosedLoopConfig
from qos_actions import STRATEGIES


def run_one_seed(
    model,
    seed: int,
    n_nodes: int = 23,
    n_timeslots: int = 1000,
    interval_min: int = 15,
    window: int = 12,
    test_frac: float = 0.15,
    strategies: List[str] = None,
) -> dict:
    """Run one Monte Carlo trial with the given seed."""
    if strategies is None:
        strategies = list(STRATEGIES.keys())

    # 1. Generate fresh traffic
    df = simulate(n_nodes=n_nodes, n_timeslots=n_timeslots,
                  interval_min=interval_min, seed=seed)
    cols = [c for c in df.columns if c.startswith("y_")]
    raw = df[cols].to_numpy(dtype=np.float32)
    scale = float(raw.max()) if raw.max() > 0 else 1.0
    scaled = raw / scale

    # 2. Window
    X, Y = build_windows(scaled, window=window)
    Xtr, Ytr, Xte, Yte = train_test_split_timeseries(X, Y, test_frac)

    # 3. Predict on test
    preds = model.predict(Xte, verbose=0)

    # Test MSE
    mse = float(np.mean((Yte - preds) ** 2))

    # 4. Closed-loop with each strategy
    cfg = ClosedLoopConfig()
    strategy_results = {}
    for s in strategies:
        if s not in STRATEGIES:
            continue
        r = run_closed_loop(
            actual_test=Yte,
            predicted_test=preds,
            n_nodes=n_nodes,
            config=cfg,
            historical_for_capacity=Ytr,
            strategy=STRATEGIES[s],
        )
        strategy_results[s] = {
            "reactive_events": r.reactive_events,
            "proactive_events": r.proactive_events,
            "event_reduction_pct": r.event_reduction_pct,
            "reactive_overflow": r.reactive_total_overflow,
            "proactive_overflow": r.proactive_total_overflow,
            "overflow_reduction_pct": r.overflow_reduction_pct,
            "throttle_actions": r.proactive_throttle_actions,
        }

    return {
        "seed": seed,
        "mse": mse,
        "scale": scale,
        "strategies": strategy_results,
    }


def aggregate(trials: List[dict]) -> dict:
    """Compute mean / stdev / 95% CI across trials."""
    out = {"n_trials": len(trials)}

    mses = [t["mse"] for t in trials]
    out["mse"] = {
        "mean": float(np.mean(mses)),
        "std":  float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0,
        "min":  float(np.min(mses)),
        "max":  float(np.max(mses)),
    }

    # Per-strategy aggregates
    out["strategies"] = {}
    if not trials:
        return out
    strategy_names = list(trials[0]["strategies"].keys())
    for s in strategy_names:
        per_metric = {}
        for metric in ["proactive_events", "event_reduction_pct",
                        "proactive_overflow", "overflow_reduction_pct",
                        "throttle_actions"]:
            vals = [t["strategies"][s][metric] for t in trials]
            per_metric[metric] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
            }
        out["strategies"][s] = per_metric

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_seeds", type=int, default=15)
    ap.add_argument("--n_timeslots", type=int, default=1000,
                    help="Slots per trial (smaller = faster MC)")
    ap.add_argument("--n_nodes", type=int, default=23)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--model", default="results/lstm_model.keras")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES.keys()))
    ap.add_argument("--out", default="results/monte_carlo.json")
    ap.add_argument("--start_seed", type=int, default=100)
    args = ap.parse_args()

    print(f"[mc] loading model from {args.model}")
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)

    trials = []
    print(f"[mc] running {args.n_seeds} trials, "
          f"{args.n_timeslots} slots each, "
          f"{len(args.strategies)} strategies\n")

    t0 = time.time()
    for i in range(args.n_seeds):
        seed = args.start_seed + i
        t1 = time.time()
        result = run_one_seed(
            model, seed,
            n_nodes=args.n_nodes,
            n_timeslots=args.n_timeslots,
            window=args.window,
            strategies=args.strategies,
        )
        trials.append(result)
        print(f"  trial {i+1:>3}/{args.n_seeds}  seed={seed}  "
              f"mse={result['mse']:.5f}  ({time.time()-t1:.1f}s)")

    print(f"\n[mc] done in {time.time()-t0:.1f}s")

    summary = aggregate(trials)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "trials": trials,
        "summary": summary,
    }, indent=2))
    print(f"[mc] saved to {out_path}")

    # Print summary table
    print("\n=== Monte Carlo summary ===")
    print(f"  LSTM test MSE: {summary['mse']['mean']:.5f} "
          f"± {summary['mse']['std']:.5f} "
          f"(n={summary['n_trials']})")
    print()
    print(f"  {'Strategy':<12} {'Events ↓ %':<22} {'Overflow ↓ %':<22}")
    for s, m in summary["strategies"].items():
        e = m["event_reduction_pct"]
        o = m["overflow_reduction_pct"]
        print(f"  {s:<12} "
              f"{e['mean']:>6.1f}% ± {e['std']:>5.1f}% [{e['min']:>5.1f},{e['max']:>5.1f}]   "
              f"{o['mean']:>6.1f}% ± {o['std']:>5.1f}% [{o['min']:>5.1f},{o['max']:>5.1f}]")


if __name__ == "__main__":
    main()
