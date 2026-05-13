"""
topology.py
-----------
Lightweight network topology model for visualisation and resilience
experiments. Lets the user disable nodes or cut links and re-evaluate
how the LSTM's predictions hold up.

A real GÉANT-like topology has 23 PoPs in European capitals. We use a
ring + crosslinks approximation that gives plausible alternate paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Set, Tuple

import numpy as np


@dataclass
class NetworkTopology:
    n_nodes: int = 23
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    links: Set[Tuple[int, int]] = field(default_factory=set)
    link_capacity: dict = field(default_factory=dict)
    disabled_nodes: Set[int] = field(default_factory=set)
    disabled_links: Set[Tuple[int, int]] = field(default_factory=set)
    node_labels: list = field(default_factory=list)

    @classmethod
    def geant_like(cls, n: int = 23) -> "NetworkTopology":
        """Build a GÉANT-inspired topology: ring backbone + chord links.

        Nodes are placed in a circle. Each node connects to its two ring
        neighbours and to two further-out chord neighbours, giving an
        average degree of about 4 — similar to GÉANT's link density.
        """
        rng = np.random.default_rng(42)
        # Circular positions
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 1.0
        positions = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
        # tiny perturbation so labels don't perfectly overlap
        positions += rng.uniform(-0.03, 0.03, size=positions.shape)

        links: Set[Tuple[int, int]] = set()
        capacities: dict = {}

        # ring
        for i in range(n):
            a, b = sorted([i, (i + 1) % n])
            links.add((a, b))
            capacities[(a, b)] = 1.0

        # chord links (skip 5 and skip 11)
        for i in range(n):
            for skip in (5, 11):
                a, b = sorted([i, (i + skip) % n])
                if (a, b) not in links and a != b:
                    links.add((a, b))
                    capacities[(a, b)] = 0.7

        labels = [f"PoP-{i:02d}" for i in range(n)]

        return cls(
            n_nodes=n,
            positions=positions,
            links=links,
            link_capacity=capacities,
            node_labels=labels,
        )

    # ------------------------------------------------------------------
    # State changes
    # ------------------------------------------------------------------
    def disable_node(self, i: int):
        if 0 <= i < self.n_nodes:
            self.disabled_nodes.add(i)

    def enable_node(self, i: int):
        self.disabled_nodes.discard(i)

    def cut_link(self, a: int, b: int):
        self.disabled_links.add(tuple(sorted([a, b])))

    def restore_link(self, a: int, b: int):
        self.disabled_links.discard(tuple(sorted([a, b])))

    def reset(self):
        self.disabled_nodes.clear()
        self.disabled_links.clear()

    # ------------------------------------------------------------------
    # Active topology
    # ------------------------------------------------------------------
    def active_links(self) -> Iterable[Tuple[int, int]]:
        """Yield links that are currently up."""
        for (a, b) in self.links:
            if a in self.disabled_nodes or b in self.disabled_nodes:
                continue
            if (a, b) in self.disabled_links:
                continue
            yield (a, b)

    def adjacency(self) -> dict:
        """Return adjacency dict {node: set(neighbour)} excluding disabled."""
        adj = {i: set() for i in range(self.n_nodes) if i not in self.disabled_nodes}
        for a, b in self.active_links():
            adj[a].add(b)
            adj[b].add(a)
        return adj

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def shortest_path(self, src: int, dst: int) -> Optional[list]:
        """BFS shortest path in the active topology."""
        if src == dst:
            return [src]
        if src in self.disabled_nodes or dst in self.disabled_nodes:
            return None
        adj = self.adjacency()
        if src not in adj or dst not in adj:
            return None
        # BFS
        from collections import deque
        prev = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                # reconstruct
                path = []
                while u is not None:
                    path.append(u)
                    u = prev[u]
                return list(reversed(path))
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        return None

    def reachable_pairs(self) -> int:
        """Count how many ordered (src, dst) pairs still have a path."""
        adj = self.adjacency()
        count = 0
        for s in adj:
            # BFS from s
            seen = {s}
            stack = [s]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v); stack.append(v)
            count += len(seen) - 1  # exclude s→s
        return count

    def total_pairs(self) -> int:
        live = self.n_nodes - len(self.disabled_nodes)
        return live * (live - 1)

    def connectivity_pct(self) -> float:
        total = self.total_pairs()
        if total == 0:
            return 0.0
        return 100.0 * self.reachable_pairs() / total
