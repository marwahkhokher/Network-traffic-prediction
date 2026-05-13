"""
closed_loop_demo.py
-------------------
Simulates the closed-loop QoS pipeline: LSTM forecast → QoS strategy →
mitigated traffic → measured congestion.

Originally only supported a single rate-limit strategy. Now plugs into
qos_actions.py so any registered strategy (RateLimit, Reroute, Prioritize,
Hybrid) can be evaluated against the same baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qos_actions import (
    QosConfig,
    QosStrategy,
    RateLimitStrategy,
    STRATEGIES,
)


@dataclass
class ClosedLoopConfig:
    capacity_quantile: float = 0.90
    """A node's capacity is set at this quantile of its historical
    inbound load. Values < 1.0 mean some natural congestion."""

    throttle_factor: float = 0.5
    """When predictor warns of overload, scale traffic by this factor."""

    threshold_safety_margin: float = 0.80
    """Trigger action when predicted load > capacity × this margin."""

    reroute_fraction: float = 0.30
    priority_quantile: float = 0.70

    def to_qos_config(self) -> QosConfig:
        return QosConfig(
            safety_margin=self.threshold_safety_margin,
            throttle_factor=self.throttle_factor,
            reroute_fraction=self.reroute_fraction,
            priority_quantile=self.priority_quantile,
        )


@dataclass
class ClosedLoopResult:
    n_nodes: int = 0
    n_slots: int = 0
    capacities: np.ndarray = field(default_factory=lambda: np.zeros(0))
    strategy_name: str = "RateLimit"

    # Reactive (no prediction)
    reactive_inbound: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    reactive_overflow: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    reactive_events: int = 0
    reactive_total_overflow: float = 0.0
    reactive_experience: float = 1.0

    # Proactive (predictor-guided)
    proactive_inbound: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    proactive_overflow: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    proactive_events: int = 0
    proactive_total_overflow: float = 0.0
    proactive_experience: float = 1.0
    proactive_throttle_actions: int = 0

    # Predicted inbound (for inspection)
    predicted_inbound: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def event_reduction_pct(self) -> float:
        if self.reactive_events == 0:
            return 0.0
        return 100.0 * (self.reactive_events - self.proactive_events) / self.reactive_events

    @property
    def overflow_reduction_pct(self) -> float:
        if self.reactive_total_overflow == 0:
            return 0.0
        return 100.0 * (self.reactive_total_overflow - self.proactive_total_overflow) \
               / self.reactive_total_overflow


def _matrix_to_inbound(mat_flat: np.ndarray, n_nodes: int) -> np.ndarray:
    """Sum each column j over origins i → inbound load at each destination.

    mat_flat shape: (T, N²)   →   returns (T, N)
    """
    T = mat_flat.shape[0]
    return mat_flat.reshape(T, n_nodes, n_nodes).sum(axis=1)


def _experience_score(overflow: np.ndarray, capacities: np.ndarray) -> float:
    """Naive QoE proxy: 1.0 minus average overflow ratio.
    overflow shape: (T, N) ; capacities shape: (N,) ; returns scalar in [0, 1]."""
    if capacities.sum() == 0:
        return 1.0
    rel = np.clip(overflow / np.maximum(capacities[None, :], 1e-9), 0, 1)
    return float(1.0 - rel.mean())


def run_closed_loop(
    actual_test: np.ndarray,           # (T, N²)  ground-truth scaled traffic
    predicted_test: np.ndarray,        # (T, N²)  predictor outputs for those slots
    n_nodes: int,
    config: Optional[ClosedLoopConfig] = None,
    historical_for_capacity: Optional[np.ndarray] = None,
    strategy: Optional[QosStrategy] = None,
) -> ClosedLoopResult:
    """Run the two scenarios and return a comparison.

    `strategy` selects the QoS action applied when the predictor flags an
    overload. Defaults to RateLimitStrategy for backward compatibility.
    Pass any registered strategy from `qos_actions.STRATEGIES`.
    """
    cfg = config or ClosedLoopConfig()
    strat = strategy or RateLimitStrategy()
    T = actual_test.shape[0]

    actual_inbound = _matrix_to_inbound(actual_test, n_nodes)        # (T, N)
    predicted_inbound = _matrix_to_inbound(predicted_test, n_nodes)  # (T, N)

    # Capacity per node from historical data (or test if not provided)
    cap_source = historical_for_capacity if historical_for_capacity is not None else actual_test
    cap_inbound = _matrix_to_inbound(cap_source, n_nodes)
    capacities = np.quantile(cap_inbound, cfg.capacity_quantile, axis=0)  # (N,)

    # ---- Scenario A: Reactive ----
    overflow_a = np.maximum(actual_inbound - capacities[None, :], 0.0)
    events_a = int(np.sum(overflow_a > 0))
    total_overflow_a = float(overflow_a.sum())

    # ---- Scenario B: Proactive (apply chosen strategy) ----
    actual_3d = actual_test.reshape(T, n_nodes, n_nodes)
    proactive_3d, throttle_count = strat.apply(
        actual_3d, predicted_inbound, capacities, cfg.to_qos_config(),
    )
    proactive_flat = proactive_3d.reshape(T, n_nodes * n_nodes)
    proactive_inbound = _matrix_to_inbound(proactive_flat, n_nodes)
    overflow_b = np.maximum(proactive_inbound - capacities[None, :], 0.0)
    events_b = int(np.sum(overflow_b > 0))
    total_overflow_b = float(overflow_b.sum())

    return ClosedLoopResult(
        n_nodes=n_nodes,
        n_slots=T,
        capacities=capacities,
        strategy_name=strat.name,
        reactive_inbound=actual_inbound,
        reactive_overflow=overflow_a,
        reactive_events=events_a,
        reactive_total_overflow=total_overflow_a,
        reactive_experience=_experience_score(overflow_a, capacities),
        proactive_inbound=proactive_inbound,
        proactive_overflow=overflow_b,
        proactive_events=events_b,
        proactive_total_overflow=total_overflow_b,
        proactive_experience=_experience_score(overflow_b, capacities),
        proactive_throttle_actions=throttle_count,
        predicted_inbound=predicted_inbound,
    )


def summary_table(result: ClosedLoopResult) -> str:
    """Produce a pretty text summary."""
    return (
        f"┌─────────────────────────────┬─────────────┬─────────────┬─────────────┐\n"
        f"│ Metric                      │ Reactive    │ Proactive   │ Improvement │\n"
        f"│                             │ (no LSTM)   │ (LSTM-QoS)  │             │\n"
        f"├─────────────────────────────┼─────────────┼─────────────┼─────────────┤\n"
        f"│ Congestion events           │ {result.reactive_events:>11,} │ {result.proactive_events:>11,} │ "
        f"{result.event_reduction_pct:>10.1f}% │\n"
        f"│ Total overflow volume       │ {result.reactive_total_overflow:>11.2f} │ "
        f"{result.proactive_total_overflow:>11.2f} │ {result.overflow_reduction_pct:>10.1f}% │\n"
        f"│ User experience score       │ {result.reactive_experience:>11.4f} │ "
        f"{result.proactive_experience:>11.4f} │ "
        f"{(result.proactive_experience - result.reactive_experience) * 100:>+10.2f}% │\n"
        f"│ QoS throttle actions taken  │ {'—':>11} │ {result.proactive_throttle_actions:>11,} │ "
        f"{'—':>11} │\n"
        f"└─────────────────────────────┴─────────────┴─────────────┴─────────────┘"
    )


if __name__ == "__main__":
    import json
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_traffic_csv, build_windows, train_test_split_timeseries

    d = load_traffic_csv("data/traffic_matrix.csv")
    X, Y = build_windows(d.scaled, window=12)
    Xtr, Ytr, Xte, Yte = train_test_split_timeseries(X, Y, 0.15)

    preds = np.load("results/predictions.npz")
    if "LSTM" not in preds.files:
        raise SystemExit("Train the LSTM first.")

    result = run_closed_loop(
        actual_test=Yte,
        predicted_test=preds["LSTM"],
        n_nodes=d.n_nodes,
        historical_for_capacity=Ytr,
    )
    print(summary_table(result))
