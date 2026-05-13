"""
network_animation.py
--------------------
Build an animated Plotly figure showing the network topology with node
colours pulsing as traffic load changes over a time window. Optionally
overlay congestion warnings and active QoS actions.

This is the visual story for the viva demo: instead of a CSV, you watch
the network breathe.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from topology import NetworkTopology


def build_animation(
    topology: NetworkTopology,
    inbound_actual: np.ndarray,        # (T, N)
    inbound_predicted: Optional[np.ndarray] = None,
    inbound_proactive: Optional[np.ndarray] = None,
    capacities: Optional[np.ndarray] = None,
    qos_actions_per_slot: Optional[list] = None,  # list of sets of dest indices
    title: str = "Network — traffic over time",
    interval_ms: int = 400,
) -> go.Figure:
    """Return a Plotly figure with frames the user can play through."""
    n_nodes = topology.n_nodes
    T = inbound_actual.shape[0]
    pos = topology.positions

    if capacities is None:
        capacities = np.full(n_nodes, inbound_actual.max())

    # ---- static link traces ----
    edge_x, edge_y = [], []
    for a, b in topology.active_links():
        edge_x.extend([pos[a, 0], pos[b, 0], None])
        edge_y.extend([pos[a, 1], pos[b, 1], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(120,120,120,0.4)", width=1.5),
        hoverinfo="skip", showlegend=False,
    )

    # ---- frames ----
    frames = []
    for t in range(T):
        load = inbound_actual[t]
        load_ratio = load / np.maximum(capacities, 1e-9)
        # Colour by load_ratio: green < 0.7 ≤ amber < 1.0 ≤ red
        node_colors = []
        sizes = []
        labels = []
        outline_w = []
        outline_c = []
        for j in range(n_nodes):
            r = load_ratio[j]
            if j in topology.disabled_nodes:
                node_colors.append("#444"); outline_c.append("#000"); outline_w.append(2)
            elif r >= 1.0:
                node_colors.append("#dc2626"); outline_c.append("#7f1d1d"); outline_w.append(3)
            elif r >= 0.7:
                node_colors.append("#f59e0b"); outline_c.append("#92400e"); outline_w.append(2)
            else:
                node_colors.append("#10b981"); outline_c.append("#064e3b"); outline_w.append(1)
            sizes.append(20 + 30 * float(min(r, 1.5)))
            label_parts = [f"<b>{topology.node_labels[j] if j < len(topology.node_labels) else f'node {j}'}</b>",
                           f"load: {load[j]:.2f}",
                           f"cap:  {capacities[j]:.2f}",
                           f"util: {r*100:.0f}%"]
            if inbound_predicted is not None and t < inbound_predicted.shape[0]:
                label_parts.append(f"<i>predicted: {inbound_predicted[t, j]:.2f}</i>")
            if qos_actions_per_slot is not None and t < len(qos_actions_per_slot) and j in qos_actions_per_slot[t]:
                label_parts.append("⚙️ QoS active")
                outline_c[-1] = "#1e3a8a"; outline_w[-1] = 4
            labels.append("<br>".join(label_parts))

        safe_sizes = np.asarray(sizes, dtype=float)
        safe_sizes = np.nan_to_num(safe_sizes, nan=10.0, posinf=30.0, neginf=10.0)
        safe_sizes = np.clip(safe_sizes, 8, 40).tolist()

        node_trace = go.Scatter(
            x=pos[:, 0], y=pos[:, 1], mode="markers+text",
            marker=dict(
                size=safe_sizes,
                color=node_colors,
                line=dict(color=outline_c, width=outline_w)
            ),
            text=[topology.node_labels[j].replace("PoP-", "") if j < len(topology.node_labels) else str(j)
                for j in range(n_nodes)],
            textposition="middle center",
            textfont=dict(color="white", size=9, family="DejaVu Sans"),
            hoverinfo="text",
            hovertext=labels,
            showlegend=False,
        )

    # ---- initial figure ----
    if len(frames) == 0:
        return go.Figure()

    initial = frames[0].data
    fig = go.Figure(
        data=list(initial),
        frames=frames,
    )

    # Layout
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.3, 1.3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.3, 1.3], scaleanchor="x", scaleratio=1),
        plot_bgcolor="rgba(245,247,250,1)",
        margin=dict(l=20, r=20, t=60, b=80),
        height=620,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.1, "y": -0.05, "xanchor": "right", "yanchor": "top",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": interval_ms, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 100}}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.15, "y": -0.05, "len": 0.85,
            "currentvalue": {"prefix": "slot: ", "visible": True, "xanchor": "right"},
            "steps": [
                {"args": [[str(t)], {"frame": {"duration": 0, "redraw": True},
                                       "mode": "immediate"}],
                 "label": str(t), "method": "animate"}
                for t in range(T)
            ],
        }],
    )

    # Legend (manual, since markers don't reliably show coloured legend)
    fig.add_annotation(
        x=1.0, y=1.18, xref="paper", yref="paper", showarrow=False,
        text="🟢 light load (<70% cap)   🟡 high load (70–100%)   🔴 overload (>100%)   "
             "<span style='color:#1e3a8a'>⚙️ QoS active</span>",
        font=dict(size=11),
    )
    return fig


def build_topology_figure(topology: NetworkTopology, title: str = "Network topology") -> go.Figure:
    """Static topology view, no time dimension. Used on the topology page."""
    pos = topology.positions
    edge_x, edge_y, edge_color = [], [], []
    for a, b in topology.links:
        cut = (a, b) in topology.disabled_links or a in topology.disabled_nodes or b in topology.disabled_nodes
        edge_x.extend([pos[a, 0], pos[b, 0], None])
        edge_y.extend([pos[a, 1], pos[b, 1], None])
        edge_color.append("#ef4444" if cut else "#888")

    # Build per-segment traces so cuts can render red and dashed
    fig = go.Figure()
    for (a, b) in topology.links:
        cut_link = (a, b) in topology.disabled_links
        cut_node = a in topology.disabled_nodes or b in topology.disabled_nodes
        is_cut = cut_link or cut_node
        fig.add_trace(go.Scatter(
            x=[pos[a, 0], pos[b, 0]],
            y=[pos[a, 1], pos[b, 1]],
            mode="lines",
            line=dict(color="#ef4444" if is_cut else "#9ca3af",
                      width=2.5 if cut_link else 1.5,
                      dash="dot" if is_cut else "solid"),
            hoverinfo="skip", showlegend=False,
        ))

    # Nodes
    colors = ["#444" if j in topology.disabled_nodes else "#1e3a8a"
              for j in range(topology.n_nodes)]
    sizes  = [22 if j in topology.disabled_nodes else 30 for j in range(topology.n_nodes)]
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1], mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(color="white", width=2)),
        text=[topology.node_labels[j].replace("PoP-", "") if j < len(topology.node_labels) else str(j)
              for j in range(topology.n_nodes)],
        textposition="middle center", textfont=dict(color="white", size=9),
        hovertext=[f"<b>{topology.node_labels[j] if j < len(topology.node_labels) else j}</b><br>"
                   f"{'❌ DISABLED' if j in topology.disabled_nodes else '✅ active'}"
                   for j in range(topology.n_nodes)],
        hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.3, 1.3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.3, 1.3], scaleanchor="x", scaleratio=1),
        plot_bgcolor="rgba(245,247,250,1)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
    )
    return fig
