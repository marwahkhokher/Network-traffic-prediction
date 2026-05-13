"""
qos_actions.py
--------------
A library of QoS / traffic-engineering strategies that can be triggered
by the LSTM's predictions when overload is forecast.

Each strategy implements the same interface so the closed-loop simulation
can plug them in interchangeably:

    strategy.apply(actual, predicted_inbound, capacities, topology) -> (modified, action_count)

Strategies provided:
  * NoAction              — baseline; do nothing (reactive scenario)
  * RateLimit             — throttle inbound traffic to predicted-overloaded dest
  * Reroute               — push some predicted-overload traffic via alternate paths
  * Prioritize            — protect high-priority OD pairs, throttle low-priority
  * Hybrid                — combine rerouting + prioritized throttling

These are simplified abstractions of real network operations:
  - RateLimit ↔ rate-shaping at ingress (token bucket, policers)
  - Reroute   ↔ MPLS-TE / SDN flow steering
  - Prioritize↔ DiffServ classes / QoS marking
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class QosConfig:
    safety_margin: float = 0.80
    """Trigger action when predicted load > capacity × this margin."""

    throttle_factor: float = 0.5
    """For rate-limit: scale factor applied to throttled traffic."""

    reroute_fraction: float = 0.30
    """For reroute: fraction of overload-causing traffic shifted to alternate paths."""

    priority_quantile: float = 0.70
    """For prioritize: OD pairs above this volume quantile are 'high priority'
    and protected; the rest are throttleable."""


class QosStrategy(ABC):
    """Apply some traffic-engineering action when overload is predicted."""

    name: str = "abstract"

    @abstractmethod
    def apply(
        self,
        actual_matrix: np.ndarray,        # (T, N, N)  ground-truth scaled traffic
        predicted_inbound: np.ndarray,    # (T, N)     LSTM-predicted dest inbound
        capacities: np.ndarray,           # (N,)
        config: QosConfig,
        topology: Optional["NetworkTopology"] = None,
    ) -> Tuple[np.ndarray, int]:
        """Return the post-QoS matrix plus number of actions taken."""
        ...


# ------------------------------------------------------------------ NoAction
class NoActionStrategy(QosStrategy):
    name = "NoAction"

    def apply(self, actual_matrix, predicted_inbound, capacities, config, topology=None):
        return actual_matrix.copy(), 0


# ------------------------------------------------------------------ RateLimit
class RateLimitStrategy(QosStrategy):
    name = "RateLimit"

    def apply(self, actual_matrix, predicted_inbound, capacities, config, topology=None):
        out = actual_matrix.copy()
        T, N, _ = out.shape
        actions = 0
        threshold = capacities * config.safety_margin
        flagged = predicted_inbound > threshold[None, :]   # (T, N) bool
        for t in range(T):
            for j in np.where(flagged[t])[0]:
                out[t, :, j] *= config.throttle_factor
                actions += 1
        return out, actions


# ------------------------------------------------------------------ Reroute
class RerouteStrategy(QosStrategy):
    name = "Reroute"

    def apply(self, actual_matrix, predicted_inbound, capacities, config, topology=None):
        out = actual_matrix.copy()
        T, N, _ = out.shape
        actions = 0
        threshold = capacities * config.safety_margin
        frac = config.reroute_fraction
        flagged = predicted_inbound > threshold[None, :]

        # Lookup of underloaded alternate destinations per slot
        for t in range(T):
            overloaded = np.where(flagged[t])[0]
            if len(overloaded) == 0:
                continue

            # Underloaded destinations (those well below capacity)
            inbound_t = out[t].sum(axis=0)
            spare = np.maximum(capacities - inbound_t, 0.0)
            order = np.argsort(-spare)  # most-spare first

            for j in overloaded:
                # Don't reroute to itself or to other overloaded
                candidates = [k for k in order
                              if k != j and k not in overloaded and spare[k] > 0]
                if not candidates:
                    continue
                alt = candidates[0]
                # Move `frac` of inbound to j onto alt
                shifted = out[t, :, j] * frac
                out[t, :, j] -= shifted
                out[t, :, alt] += shifted
                actions += 1
                # Update spare so next iteration has fresh info
                spare[alt] -= shifted.sum()

        return out, actions


# ------------------------------------------------------------------ Prioritize
class PrioritizeStrategy(QosStrategy):
    """Throttle only the low-priority OD pairs; keep high-priority untouched."""
    name = "Prioritize"

    def apply(self, actual_matrix, predicted_inbound, capacities, config, topology=None):
        out = actual_matrix.copy()
        T, N, _ = out.shape
        actions = 0
        threshold = capacities * config.safety_margin

        # Compute per-OD priority from average volume across the actual matrix
        pair_mean = actual_matrix.mean(axis=0)              # (N, N)
        # nonzero so we don't include diagonal as "low"
        nonzero = pair_mean[pair_mean > 0]
        cutoff = (np.quantile(nonzero, config.priority_quantile)
                  if nonzero.size else 0.0)
        # 'high priority' if its mean volume is in the top quantile
        high_priority_mask = pair_mean >= cutoff           # (N, N) bool

        flagged = predicted_inbound > threshold[None, :]   # (T, N)
        for t in range(T):
            for j in np.where(flagged[t])[0]:
                # Throttle only the LOW priority origins for this destination
                low_pri = ~high_priority_mask[:, j]
                out[t, low_pri, j] *= config.throttle_factor
                actions += int(low_pri.sum())              # one action per OD throttled
        return out, actions


# ------------------------------------------------------------------ Hybrid
class HybridStrategy(QosStrategy):
    """Reroute first; if still overloaded, prioritized throttle."""
    name = "Hybrid"

    def __init__(self):
        self.reroute = RerouteStrategy()
        self.prio = PrioritizeStrategy()

    def apply(self, actual_matrix, predicted_inbound, capacities, config, topology=None):
        # First pass: reroute
        rerouted, a1 = self.reroute.apply(
            actual_matrix, predicted_inbound, capacities, config, topology
        )
        # Recompute predicted inbound from the rerouted matrix (best estimate)
        new_inbound = rerouted.sum(axis=1)                  # (T, N)
        # Second pass: prioritize-throttle anything still over threshold
        out, a2 = self.prio.apply(rerouted, new_inbound, capacities, config, topology)
        return out, a1 + a2


# ---------------------------------------------------------------- registry
STRATEGIES = {
    "NoAction":   NoActionStrategy(),
    "RateLimit":  RateLimitStrategy(),
    "Reroute":    RerouteStrategy(),
    "Prioritize": PrioritizeStrategy(),
    "Hybrid":     HybridStrategy(),
}
