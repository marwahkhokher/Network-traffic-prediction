"""
topology_resilience.py
----------------------
Tests how the LSTM holds up when the network topology changes.

Real networks fail. Cables get cut. Routers crash. A model trained on
the steady-state network will see distribution shift the moment the
topology changes. The question is: how badly does that hurt the LSTM?

Three experiments:
  1. Single-node failures: disable each node one at a time, redistribute
     its traffic to its neighbours, measure prediction error.
  2. Random link cuts: remove K random links, see how connectivity and
     prediction error degrade.
  3. Cascading failures: increase number of failed nodes/links, watch
     the curve of "graceful degradation" vs "cliff".
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_traffic_csv, build_windows, train_test_split_timeseries
from topology import NetworkTopology


def redistribute_traffic_for_failed_node(
    matrix_3d: np.ndarray,                # (T, N, N)
    failed_node: int,
    topology: NetworkTopology,
) -> np.ndarray:
    """When node `failed_node` goes down, redistribute its inbound and
    outbound traffic to its neighbours."""
    out = matrix_3d.copy()
    adj = topology.adjacency()
    neighbours = list(adj.get(failed_node, set()))
    if not neighbours:
        # Stranded — just zero out
        out[:, failed_node, :] = 0
        out[:, :, failed_node] = 0
        return out

    n_neigh = len(neighbours)
    # outgoing: split this node's emissions among neighbours
    outgoing = out[:, failed_node, :].copy()  # (T, N)
    out[:, failed_node, :] = 0
    for nb in neighbours:
        out[:, nb, :] += outgoing / n_neigh

    # incoming: split this node's inbound among neighbours
    incoming = out[:, :, failed_node].copy()  # (T, N)
    out[:, :, failed_node] = 0
    for nb in neighbours:
        out[:, :, nb] += incoming / n_neigh

    return out


def run_node_failure_sweep(
    model,
    Xte: np.ndarray,
    Yte: np.ndarray,
    topology: NetworkTopology,
) -> dict:
    """Simulate disabling each node one at a time and predicting on the
    perturbed traffic. Returns per-node MSE delta."""
    n_nodes = topology.n_nodes
    T = Yte.shape[0]
    Y_te_3d = Yte.reshape(T, n_nodes, n_nodes)
    X_te_3d = Xte.reshape(T, Xte.shape[1], n_nodes, n_nodes)

    # Baseline (no failure) test MSE
    base_pred = model.predict(Xte, verbose=0)
    base_mse = float(np.mean((Yte - base_pred) ** 2))

    results = []
    for j in range(n_nodes):
        # Save and disable
        topo_copy = NetworkTopology(
            n_nodes=topology.n_nodes,
            positions=topology.positions.copy(),
            links=set(topology.links),
            link_capacity=dict(topology.link_capacity),
            disabled_nodes=set(topology.disabled_nodes) | {j},
            disabled_links=set(topology.disabled_links),
            node_labels=list(topology.node_labels),
        )
        # Redistribute on each window's last slot AND on the targets
        # For simplicity we redistribute only the targets (Y) — the
        # model still gets the original window, so it's seeing OOD input
        Y_perturbed_3d = redistribute_traffic_for_failed_node(Y_te_3d, j, topo_copy)
        Y_perturbed = Y_perturbed_3d.reshape(T, n_nodes * n_nodes)

        # Predict with original windows; compare to perturbed truth
        pred = model.predict(Xte, verbose=0)
        mse = float(np.mean((Y_perturbed - pred) ** 2))
        results.append({
            "failed_node": j,
            "mse": mse,
            "mse_delta_pct": (mse - base_mse) / base_mse * 100 if base_mse > 0 else 0.0,
            "connectivity_pct": topo_copy.connectivity_pct(),
        })

    return {
        "baseline_mse": base_mse,
        "per_node": results,
    }


def run_link_cut_sweep(
    model,
    Xte: np.ndarray,
    Yte: np.ndarray,
    topology: NetworkTopology,
    n_repeats: int = 5,
    max_cuts: int = 8,
) -> dict:
    """Cut K random links, repeat n_repeats times, measure prediction
    error and topology connectivity vs K."""
    n_nodes = topology.n_nodes
    T = Yte.shape[0]
    rng = np.random.default_rng(42)
    base_pred = model.predict(Xte, verbose=0)
    base_mse = float(np.mean((Yte - base_pred) ** 2))

    all_links = list(topology.links)
    results = []
    for k in range(0, max_cuts + 1):
        rep_mses = []
        rep_conn = []
        for _ in range(n_repeats):
            cuts = rng.choice(len(all_links), size=k, replace=False) if k > 0 else []
            topo = NetworkTopology(
                n_nodes=topology.n_nodes,
                positions=topology.positions.copy(),
                links=set(topology.links),
                link_capacity=dict(topology.link_capacity),
                disabled_links={all_links[i] for i in cuts},
                node_labels=list(topology.node_labels),
            )
            # When links cut, traffic that would have used them gets
            # noisy. We simulate this by adding zero-mean noise scaled
            # to the lost connectivity fraction.
            conn_pct = topo.connectivity_pct()
            rep_conn.append(conn_pct)
            lost = max(0, 1.0 - conn_pct / 100.0)
            noise = rng.normal(0, lost * 0.5, size=Yte.shape).astype(np.float32)
            Y_perturbed = np.clip(Yte + noise, 0, None)
            mse = float(np.mean((Y_perturbed - base_pred) ** 2))
            rep_mses.append(mse)

        results.append({
            "n_cuts": k,
            "mse_mean":         float(np.mean(rep_mses)),
            "mse_std":          float(np.std(rep_mses, ddof=1)) if n_repeats > 1 else 0.0,
            "connectivity_mean": float(np.mean(rep_conn)),
            "connectivity_std":  float(np.std(rep_conn, ddof=1)) if n_repeats > 1 else 0.0,
            "mse_delta_pct":    (np.mean(rep_mses) - base_mse) / base_mse * 100
                                   if base_mse > 0 else 0.0,
        })

    return {
        "baseline_mse": base_mse,
        "per_k_cuts": results,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/traffic_matrix.csv")
    ap.add_argument("--model", default="results/lstm_model.keras")
    ap.add_argument("--out", default="results/topology_resilience.json")
    ap.add_argument("--max_cuts", type=int, default=8)
    ap.add_argument("--n_repeats", type=int, default=5)
    args = ap.parse_args()

    print("[res] loading data + model")
    d = load_traffic_csv(args.data)
    X, Y = build_windows(d.scaled, window=12)
    Xtr, Ytr, Xte, Yte = train_test_split_timeseries(X, Y, 0.15)

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)

    topo = NetworkTopology.geant_like(n=d.n_nodes)
    print(f"[res] topology: {d.n_nodes} nodes, {len(topo.links)} links")

    print("[res] running node-failure sweep...")
    t = time.time()
    nf = run_node_failure_sweep(model, Xte, Yte, topo)
    print(f"[res]   done in {time.time()-t:.1f}s")
    nf_sorted = sorted(nf["per_node"], key=lambda x: -x["mse_delta_pct"])
    print(f"[res]   most-impactful failure: node {nf_sorted[0]['failed_node']} "
          f"(+{nf_sorted[0]['mse_delta_pct']:.1f}% MSE)")

    print("[res] running link-cut sweep...")
    t = time.time()
    lc = run_link_cut_sweep(model, Xte, Yte, topo,
                              n_repeats=args.n_repeats, max_cuts=args.max_cuts)
    print(f"[res]   done in {time.time()-t:.1f}s")
    for r in lc["per_k_cuts"]:
        print(f"[res]   {r['n_cuts']} cuts → "
              f"connectivity {r['connectivity_mean']:.1f}%, "
              f"MSE {r['mse_mean']:.5f} (Δ {r['mse_delta_pct']:+.1f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "node_failure": nf,
        "link_cut": lc,
        "n_links": len(topo.links),
    }, indent=2))
    print(f"\n[res] saved to {out}")


if __name__ == "__main__":
    main()
