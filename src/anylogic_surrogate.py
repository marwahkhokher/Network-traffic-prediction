"""
anylogic_surrogate.py
---------------------
A Python simulator that mimics what the AnyLogic model produces.

Why this exists:
  The AnyLogic Personal Learning Edition is free but requires Java + a GUI
  install. When grading or quickly iterating on the ML side, it helps to
  have a pure-Python equivalent that generates the same CSV format.

The physical model (mirrors anylogic/src/TrafficGenerator.java):

  For each ordered pair (i, j) of the N nodes, traffic volume at time t is:

    y_ij(t) =  base_ij
             * diurnal(t)
             * weekly(t)
             * (1 + gaussian_noise)
             * event_multiplier(t)
             + burst_ij(t)

  Where:
    - base_ij        : gravity-model base rate (pop_i * pop_j / distance_ij)
    - diurnal(t)     : sinusoid with peak at 14:00, trough at 04:00
    - weekly(t)      : lower on weekends
    - event(t)       : random flash-crowd events (exams, live streams...)
    - burst_ij(t)    : AR(1) self-correlated bursts per OD pair

Each matrix row is flattened and written as one CSV line:
    t_index, y_00, y_01, ..., y_{N-1,N-1}

Usage:
  python src/anylogic_surrogate.py --nodes 23 --timeslots 2016 \
        --interval_min 15 --out data/traffic_matrix.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def gravity_base_matrix(n_nodes: int, seed: int = 42) -> np.ndarray:
    """Build a base-traffic matrix using the classic gravity model.

    Each node gets a random 'population' (traffic weight). OD volume
    is proportional to pop_i * pop_j divided by a random 'distance'.
    Diagonal is zero (no self-traffic).
    """
    rng = np.random.default_rng(seed)
    pops = rng.uniform(0.3, 1.0, size=n_nodes)
    # random symmetric distances
    coords = rng.uniform(0, 100, size=(n_nodes, 2))
    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    dists += np.eye(n_nodes)  # avoid divide-by-zero on diagonal
    base = np.outer(pops, pops) / (dists ** 1.2)
    np.fill_diagonal(base, 0.0)
    # normalise so max OD-pair base is ~1.0
    base *= (1.0 / base.max())
    return base


def diurnal(hour: float) -> float:
    """Daily pattern: peak ~14:00, trough ~04:00. Range roughly [0.3, 1.3]."""
    # cosine with peak at hour=14
    return 0.8 + 0.5 * math.cos((hour - 14.0) * math.pi / 12.0)


def weekly(day_of_week: int) -> float:
    """Weekends are ~70% of weekday load."""
    return 0.70 if day_of_week >= 5 else 1.0


def simulate(
    n_nodes: int,
    n_timeslots: int,
    interval_min: int,
    seed: int = 7,
    event_prob: float = 0.06,
    noise_sd: float = 0.18,
    burst_ar_coef: float = 0.4,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    base = gravity_base_matrix(n_nodes, seed=seed)

    # persistent per-OD burst state (AR(1))
    burst = np.zeros((n_nodes, n_nodes))

    # event state — when triggered, affects specific OD pairs for a few slots
    event_multiplier = np.ones((n_nodes, n_nodes))
    event_ttl = np.zeros((n_nodes, n_nodes), dtype=int)

    n_features = n_nodes * n_nodes
    out = np.zeros((n_timeslots, n_features), dtype=np.float32)

    slots_per_day = int(round(24 * 60 / interval_min))

    for t in range(n_timeslots):
        hour = (t % slots_per_day) * interval_min / 60.0
        dow = (t // slots_per_day) % 7

        d = diurnal(hour)
        w = weekly(dow)

        # Decay and possibly trigger events
        event_ttl = np.maximum(event_ttl - 1, 0)
        event_multiplier = np.where(event_ttl > 0, event_multiplier, 1.0)

        if rng.random() < event_prob:
            # trigger a flash-crowd: pick a destination, amplify inbound traffic
            # SHORT (1-3 slots) but LARGE (4-10x) - hard to predict
            dst = rng.integers(0, n_nodes)
            duration = rng.integers(1, 4)
            mult = rng.uniform(4.0, 10.0)
            event_multiplier[:, dst] = mult
            event_ttl[:, dst] = duration

        # AR(1) burst — small autocorrelation so it doesn't dominate
        innovation = rng.normal(0, 0.20, size=(n_nodes, n_nodes))
        burst = burst_ar_coef * burst + innovation
        np.fill_diagonal(burst, 0.0)

        noise = 1.0 + rng.normal(0, noise_sd, size=(n_nodes, n_nodes))

        y = base * d * w * noise * event_multiplier + np.maximum(burst, 0)
        y = np.clip(y, 0.0, None)
        np.fill_diagonal(y, 0.0)

        out[t] = y.reshape(-1)

    # DataFrame with a timestamp column for readability
    cols = [f"y_{i}_{j}" for i in range(n_nodes) for j in range(n_nodes)]
    df = pd.DataFrame(out, columns=cols)
    df.insert(0, "t_index", np.arange(n_timeslots))
    df.insert(
        1,
        "timestamp",
        pd.date_range(
            start="2025-01-01", periods=n_timeslots, freq=f"{interval_min}min"
        ),
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, default=23, help="number of PoPs (default 23 = GEANT)")
    ap.add_argument("--timeslots", type=int, default=2016, help="number of time slots to simulate")
    ap.add_argument("--interval_min", type=int, default=15, help="minutes per slot")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="data/traffic_matrix.csv")
    args = ap.parse_args()

    print(f"[anylogic_surrogate] simulating {args.nodes}x{args.nodes} matrix, "
          f"{args.timeslots} slots @ {args.interval_min}min = "
          f"{args.timeslots * args.interval_min / 60 / 24:.1f} days")

    df = simulate(
        n_nodes=args.nodes,
        n_timeslots=args.timeslots,
        interval_min=args.interval_min,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[anylogic_surrogate] wrote {out_path}  "
          f"({df.shape[0]} rows, {df.shape[1] - 2} traffic features)")


if __name__ == "__main__":
    main()
