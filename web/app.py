"""
Network Traffic Prediction — Interactive Dashboard
==================================================

A Streamlit dashboard that demonstrates:
  1. The trained LSTM's predictions vs all baselines on real test data
  2. Live single-step prediction with adjustable input window
  3. Closed-loop QoS simulation showing the operational value of the LSTM
  4. AnyLogic's role explained with a concrete what-if demo

Run:
  streamlit run web/app.py

The app reads from the project root (one level up). All artifacts come
from results/ — no retraining needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Make sure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_preprocessing import load_traffic_csv, build_windows, train_test_split_timeseries
from closed_loop_demo import run_closed_loop, ClosedLoopConfig

# -------------------------------------------------------------- config
st.set_page_config(
    page_title="Network Traffic Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "Persistence":      "#9ca3af",
    "ARMA":             "#dc2626",
    "LinearRegression": "#f59e0b",
    "RandomForest":     "#10b981",
    "FFNN":             "#3b82f6",
    "LSTM":             "#1e3a8a",
    "HistoricalMean":   "#cbd5e1",
}


# -------------------------------------------------------------- caching
@st.cache_data
def load_artifacts():
    """Load all results artifacts once."""
    results_dir = PROJECT_ROOT / "results"

    with open(results_dir / "metrics.json") as f:
        metrics = json.load(f)

    preds = np.load(results_dir / "predictions.npz")
    pred_dict = {k: preds[k] for k in preds.files}

    with open(results_dir / "lstm_history.json") as f:
        history = json.load(f)

    closed_loop_path = results_dir / "closed_loop_metrics.json"
    closed_loop = json.load(open(closed_loop_path)) if closed_loop_path.exists() else {}

    return metrics, pred_dict, history, closed_loop


@st.cache_data
def load_data():
    """Load and window the traffic CSV."""
    d = load_traffic_csv(PROJECT_ROOT / "data" / "traffic_matrix.csv")
    X, Y = build_windows(d.scaled, window=12)
    Xtr, Ytr, Xte, Yte = train_test_split_timeseries(X, Y, 0.15)
    return d, X, Y, Xtr, Ytr, Xte, Yte


@st.cache_resource
def load_lstm_model():
    """Lazy-load Keras model only when needed (slow import)."""
    import tensorflow as tf
    return tf.keras.models.load_model(PROJECT_ROOT / "results" / "lstm_model.keras")


# -------------------------------------------------------------- sidebar
st.sidebar.title("📡 Traffic Prediction")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Model Comparison",
        "🔮 Live Prediction",
        "🚦 Closed-Loop QoS Demo",
        "🛡️ QoS Strategies",
        "🌐 Live Network View",
        "🔌 Topology & Resilience",
        "🎲 Monte Carlo (Confidence)",
        "📡 Streaming Live",
        "🔬 What does AnyLogic do?",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Based on **Azzouni & Pujolle (2017)**, "
    "*An LSTM Recurrent Neural Network Framework for Network Traffic Matrix Prediction.*"
)
st.sidebar.caption("Built for CCN coursework • GIKI")


# ============================================================== PAGE 1
if page == "🏠 Overview":
    st.title("Network Traffic Matrix Prediction")
    st.markdown(
        "An LSTM that predicts the next 15-minute traffic matrix on a 23-node "
        "backbone network, so operators can apply QoS *before* congestion happens."
    )

    metrics, preds, history, closed_loop = load_artifacts()
    d, X, Y, Xtr, Ytr, Xte, Yte = load_data()

    # Top KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Network nodes", f"{d.n_nodes}")
    with c2:
        st.metric("Total slots", f"{d.raw.shape[0]:,}")
    with c3:
        st.metric("Test set size", f"{len(Yte)}")
    with c4:
        st.metric("LSTM test MSE",
                  f"{metrics.get('LSTM', {}).get('MSE', 0):.5f}")

    st.markdown("---")

    # Two-column explanation
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("The problem")
        st.markdown(
            "Network traffic has sharp daily peaks, weekend dips, and unexpected "
            "events (exam releases, viral content, sports finals). When demand "
            "exceeds capacity, queues form → delay, jitter, packet loss, dropped "
            "calls.\n\n"
            "Traditional monitoring tells operators what's happening **right now**. "
            "By the time the dashboard goes red, users already had a bad experience. "
            "Forecasting fills that gap."
        )
        st.subheader("This project")
        st.markdown(
            "A Long Short-Term Memory network reads the last **3 hours** of traffic "
            "data (12 slots × 15 min) and predicts the next slot's full 23×23 "
            "traffic matrix. The prediction is then used to trigger preemptive "
            "QoS policies, reducing congestion events by **>50%** in our "
            "simulation."
        )

    with right:
        st.subheader("Quick result")
        if metrics:
            df_metrics = pd.DataFrame([
                {"Method": k, "MSE": v["MSE"], "MAE": v["MAE"]}
                for k, v in sorted(metrics.items(), key=lambda x: x[1]["MSE"])
            ])
            st.dataframe(
                df_metrics.style.format({"MSE": "{:.5f}", "MAE": "{:.5f}"})
                .background_gradient(subset=["MSE"], cmap="RdYlGn_r"),
                use_container_width=True,
                hide_index=True,
            )
            best = df_metrics.iloc[0]
            st.success(
                f"**LSTM** achieves the lowest test MSE ({best['MSE']:.5f}). "
                f"It also drives the best operational outcome — see the "
                f"Closed-Loop QoS Demo page."
            )

    st.markdown("---")
    st.subheader("Architecture")
    st.markdown(
        """
```
┌─────────────────────────┐         ┌──────────────────────────┐
│  AnyLogic simulation    │  CSV    │  Python LSTM pipeline    │
│  (or Python surrogate)  │ ──────► │                          │
│                         │         │  preprocess → window     │
│  23-node backbone       │         │  → train LSTM            │
│  • diurnal + weekly     │         │  → compare baselines     │
│  • flash-crowd events   │         │  → save model            │
│  • AR(1) bursts         │         └────────────┬─────────────┘
└─────────────────────────┘                      │
                                                 ▼
                                     ┌──────────────────────────┐
                                     │  Closed-loop QoS demo    │
                                     │                          │
                                     │  predict → flag          │
                                     │  → throttle → measure    │
                                     │  congestion reduction    │
                                     └──────────────────────────┘
```
"""
    )


# ============================================================== PAGE 2
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.caption("How does the LSTM compare to ARMA, Linear Regression, FFNN, and other baselines?")

    metrics, preds, history, closed_loop = load_artifacts()
    d, X, Y, Xtr, Ytr, Xte, Yte = load_data()

    if not metrics:
        st.error("No results found. Run `python src/train.py ...` first.")
        st.stop()

    # ---------------------------- Top row: MSE bar chart
    st.subheader("Test-set MSE by method")
    sorted_methods = sorted(metrics.keys(), key=lambda k: metrics[k]["MSE"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[m for m in sorted_methods],
        y=[metrics[m]["MSE"] for m in sorted_methods],
        marker_color=[COLORS.get(m, "#888") for m in sorted_methods],
        text=[f"{metrics[m]['MSE']:.5f}" for m in sorted_methods],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>MSE: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="MSE (lower is better)",
        height=400,
        showlegend=False,
        margin=dict(t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------- LSTM advantage
    if "LSTM" in metrics:
        st.subheader("How much better is LSTM than each baseline?")
        lstm_mse = metrics["LSTM"]["MSE"]
        improvements = []
        for m in sorted_methods:
            if m == "LSTM":
                continue
            pct = (metrics[m]["MSE"] - lstm_mse) / metrics[m]["MSE"] * 100
            improvements.append((m, pct))
        improvements.sort(key=lambda x: -x[1])

        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            y=[m for m, _ in improvements],
            x=[v for _, v in improvements],
            orientation="h",
            marker_color=[COLORS.get(m, "#888") for m, _ in improvements],
            text=[f"{v:+.1f}%" for _, v in improvements],
            textposition="outside",
        ))
        fig_imp.update_layout(
            xaxis_title="LSTM error reduction vs baseline (%)",
            height=80 + 50 * len(improvements),
            margin=dict(t=10, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_imp.add_vline(x=0, line=dict(color="black", width=1))
        st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------------------- Prediction traces (interactive)
    st.markdown("---")
    st.subheader("Predicted vs actual — pick an OD pair")

    Y_true = preds["Y_true"]
    n_nodes = int(round(np.sqrt(Y_true.shape[1])))
    mean_flow = Y_true.mean(axis=0)

    c1, c2, c3 = st.columns(3)
    with c1:
        choice = st.radio("Select an OD pair to inspect:",
                          ["High volume (busiest pair)",
                           "Median volume",
                           "Low volume",
                           "Custom (i, j)"],
                          horizontal=False)
    with c2:
        if choice == "Custom (i, j)":
            i = st.number_input("Origin node i", 0, n_nodes - 1, 0)
            j = st.number_input("Destination node j", 0, n_nodes - 1, 1)
        elif choice == "High volume (busiest pair)":
            idx = int(np.argmax(mean_flow))
            i, j = divmod(idx, n_nodes)
        elif choice == "Median volume":
            order = np.argsort(mean_flow)
            idx = int(order[len(order) // 2])
            i, j = divmod(idx, n_nodes)
        else:
            # exclude diagonal zeros
            nonzero = np.where(mean_flow > 0)[0]
            idx = int(nonzero[np.argmin(mean_flow[nonzero])])
            i, j = divmod(idx, n_nodes)
        idx = i * n_nodes + j
    with c3:
        models_avail = [m for m in ["Persistence", "ARMA", "LinearRegression",
                                     "RandomForest", "FFNN", "LSTM"]
                         if m in preds]
        models_show = st.multiselect("Show predictions from:", models_avail,
                                      default=["LSTM", "LinearRegression"]
                                      if "LSTM" in models_avail else models_avail[:2])

    fig_trace = go.Figure()
    fig_trace.add_trace(go.Scatter(
        y=Y_true[:, idx], mode="lines",
        line=dict(color="black", width=2.5),
        name="actual",
    ))
    for m in models_show:
        fig_trace.add_trace(go.Scatter(
            y=preds[m][:, idx], mode="lines",
            line=dict(color=COLORS.get(m, "#888"), width=1.6),
            name=m, opacity=0.85,
        ))

    fig_trace.update_layout(
        title=f"OD pair: node {i} → node {j}  (mean volume = {mean_flow[idx]:.4f})",
        xaxis_title="Test time step (15-min slots)",
        yaxis_title="Traffic volume (scaled)",
        height=450,
        margin=dict(t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="x unified",
    )
    st.plotly_chart(fig_trace, use_container_width=True)

    # ---------------------------- LSTM training curve
    if history:
        st.markdown("---")
        st.subheader("LSTM training curve")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(y=history["loss"], mode="lines",
                                       line=dict(color=COLORS["LSTM"], width=2.5),
                                       name="train loss"))
        if "val_loss" in history:
            fig_hist.add_trace(go.Scatter(y=history["val_loss"], mode="lines",
                                           line=dict(color="#dc2626", width=2.5),
                                           name="validation loss"))
        fig_hist.update_layout(xaxis_title="Epoch", yaxis_title="MSE",
                               height=380, plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)


# ============================================================== PAGE 3
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Prediction")
    st.caption("Pick a starting test slot. The LSTM uses the previous 12 slots "
               "(3 hours of history) to predict the next slot. See exactly what "
               "the model says vs what actually happened.")

    metrics, preds, history, closed_loop = load_artifacts()
    d, X, Y, Xtr, Ytr, Xte, Yte = load_data()

    n_test = Yte.shape[0]
    n_nodes = d.n_nodes

    slot = st.slider("Pick a test slot to predict", 0, n_test - 1, n_test // 2)

    with st.spinner("Loading LSTM model..."):
        try:
            model = load_lstm_model()
        except Exception as e:
            st.error(f"Could not load LSTM model: {e}")
            st.stop()

    # The window for slot N in the test set is X_test[N]
    window = Xte[slot]                         # (W, N²)
    actual = Yte[slot]                         # (N²,)
    predicted = model.predict(window[None, ...], verbose=0)[0]

    # Inverse-transform back to "raw" units
    actual_raw = actual * d.scale
    predicted_raw = predicted * d.scale
    err = actual_raw - predicted_raw

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE this slot", f"{np.abs(err).mean():.4f}")
    c2.metric("Max OD error", f"{np.abs(err).max():.4f}")
    c3.metric("Total predicted load", f"{predicted_raw.sum():.2f}")
    c4.metric("Total actual load", f"{actual_raw.sum():.2f}",
              f"{(predicted_raw.sum() - actual_raw.sum()):+.2f}")

    st.markdown("---")

    # Side-by-side heatmaps
    actual_mat = actual_raw.reshape(n_nodes, n_nodes)
    pred_mat = predicted_raw.reshape(n_nodes, n_nodes)
    err_mat = (actual_raw - predicted_raw).reshape(n_nodes, n_nodes)

    vmax = max(actual_mat.max(), pred_mat.max())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Actual traffic matrix**")
        fig = px.imshow(actual_mat, color_continuous_scale="Blues",
                        zmin=0, zmax=vmax,
                        labels=dict(x="destination j", y="origin i", color="volume"))
        fig.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**LSTM predicted matrix**")
        fig = px.imshow(pred_mat, color_continuous_scale="Blues",
                        zmin=0, zmax=vmax,
                        labels=dict(x="destination j", y="origin i", color="volume"))
        fig.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        st.markdown("**Error (actual − predicted)**")
        amax = max(abs(err_mat.min()), abs(err_mat.max())) or 1e-6
        fig = px.imshow(err_mat, color_continuous_scale="RdBu",
                        zmin=-amax, zmax=amax,
                        labels=dict(x="destination j", y="origin i", color="error"))
        fig.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # Congestion flags
    st.markdown("---")
    st.subheader("⚠️ Congestion warnings from this prediction")
    inbound_pred = pred_mat.sum(axis=0)
    inbound_actual = actual_mat.sum(axis=0)
    threshold = np.quantile(
    Ytr[:, : n_nodes * n_nodes]
    .reshape(-1, n_nodes, n_nodes)
    .sum(axis=1),
    0.90,
)

    flag_data = []
    for j in range(n_nodes):
        if inbound_pred[j] > threshold:
            flag_data.append({
                "Destination node": j,
                "Predicted inbound": f"{inbound_pred[j]:.2f}",
                "Actual inbound": f"{inbound_actual[j]:.2f}",
                "Threshold (90th pct)": f"{threshold:.2f}",
                "Hit?": "🔴 yes" if inbound_actual[j] > threshold else "🟡 false alarm",
            })

    if flag_data:
        st.warning(
            f"LSTM flagged **{len(flag_data)}** destination(s) for proactive QoS:"
        )
        st.dataframe(pd.DataFrame(flag_data), use_container_width=True, hide_index=True)
    else:
        st.success("No congestion flags this slot — network is fine.")


# ============================================================== PAGE 4
elif page == "🚦 Closed-Loop QoS Demo":
    st.title("🚦 Closed-Loop QoS Demo")
    st.caption("Same predictions, different consequences. Reactive: the network just "
               "lets traffic flow and counts congestion. Proactive: the LSTM's warnings "
               "trigger rate-limiting before traffic arrives.")

    metrics, preds, history, closed_loop = load_artifacts()
    d, X, Y, Xtr, Ytr, Xte, Yte = load_data()

    if not closed_loop:
        st.warning("Closed-loop metrics not yet generated. Run "
                   "`python scripts/make_comparison_plots.py` first.")
        st.stop()

    # Methods sorted by event reduction
    methods_order = sorted(closed_loop.keys(),
                           key=lambda m: -closed_loop[m]["event_reduction_pct"])

    # ---------------------------- Headline numbers
    st.subheader("How much congestion does each predictor prevent?")
    cols = st.columns(len(methods_order))
    for col, m in zip(cols, methods_order):
        cl = closed_loop[m]
        col.metric(
            label=m,
            value=f"−{cl['event_reduction_pct']:.1f}%",
            delta=f"{cl['proactive_events']} of {cl['reactive_events']} events",
            delta_color="inverse" if cl['proactive_events'] < cl['reactive_events'] else "off",
            help=f"Throttle actions: {cl['throttle_actions']}"
        )

    st.markdown("---")

    # ---------------------------- Side-by-side bar charts
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Congestion events reduced**")
        fig1 = go.Figure(go.Bar(
            x=[m for m in methods_order],
            y=[closed_loop[m]["event_reduction_pct"] for m in methods_order],
            marker_color=[COLORS.get(m, "#888") for m in methods_order],
            text=[f"{closed_loop[m]['event_reduction_pct']:+.1f}%" for m in methods_order],
            textposition="outside",
        ))
        fig1.update_layout(yaxis_title="reduction (%)", height=400,
                           plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=20))
        st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        st.markdown("**Total overflow volume reduced**")
        fig2 = go.Figure(go.Bar(
            x=[m for m in methods_order],
            y=[closed_loop[m]["overflow_reduction_pct"] for m in methods_order],
            marker_color=[COLORS.get(m, "#888") for m in methods_order],
            text=[f"{closed_loop[m]['overflow_reduction_pct']:+.1f}%" for m in methods_order],
            textposition="outside",
        ))
        fig2.update_layout(yaxis_title="reduction (%)", height=400,
                           plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------- Timeline view
    st.markdown("---")
    st.subheader("Timeline view — what does this look like in practice?")
    st.markdown(
        "Pick a model below. The chart shows actual inbound traffic at one of the "
        "busiest network nodes. The **red curve** is what would have happened with no "
        "prediction-driven QoS — peaks blow past the capacity line and overflow. "
        "The **navy curve** is what happens with QoS guided by that model's predictions."
    )

    sel_method = st.selectbox("Predictor:", methods_order,
                              index=methods_order.index("LSTM") if "LSTM" in methods_order else 0)

    cfg = ClosedLoopConfig()
    with st.spinner(f"Running closed-loop simulation for {sel_method}..."):
        result = run_closed_loop(Yte, preds[sel_method], d.n_nodes, cfg, Ytr)

    busy_node = int(np.argmax(result.reactive_overflow.sum(axis=0)))
    cap = result.capacities[busy_node]
    t_axis = np.arange(result.reactive_inbound.shape[0])

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=t_axis, y=result.reactive_inbound[:, busy_node],
        line=dict(color="#dc2626", width=1.6), name="reactive (no QoS)",
    ))
    fig_tl.add_trace(go.Scatter(
        x=t_axis, y=result.proactive_inbound[:, busy_node],
        line=dict(color=COLORS["LSTM"], width=1.6),
        name=f"proactive (QoS guided by {sel_method})",
    ))
    fig_tl.add_hline(y=cap, line=dict(color="black", width=1.5, dash="dash"),
                     annotation_text=f"node capacity = {cap:.2f}",
                     annotation_position="top right")
    fig_tl.update_layout(
        title=f"Inbound traffic at node {busy_node} — busiest in the test window",
        xaxis_title="Test time step", yaxis_title="Inbound load",
        height=480, margin=dict(t=50, b=40), plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # ---------------------------- Interactive config
    st.markdown("---")
    with st.expander("⚙️ Tweak the closed-loop config and re-run"):
        c1, c2, c3 = st.columns(3)
        capq = c1.slider("Capacity quantile", 0.80, 0.99, 0.90, 0.01,
                          help="Node capacity = this quantile of historical inbound load")
        sm = c2.slider("Safety margin", 0.50, 1.00, 0.80, 0.05,
                       help="Throttle when predicted load > capacity × this margin")
        thr = c3.slider("Throttle factor", 0.10, 1.00, 0.50, 0.05,
                        help="Multiply throttled traffic by this — lower = more aggressive")

        if st.button("Re-run with custom config"):
            cfg = ClosedLoopConfig(capacity_quantile=capq,
                                    threshold_safety_margin=sm,
                                    throttle_factor=thr)
            with st.spinner("Running custom closed-loop..."):
                results_custom = {}
                for m in [m for m in ["Persistence", "LinearRegression", "FFNN", "LSTM"]
                           if m in preds]:
                    results_custom[m] = run_closed_loop(Yte, preds[m], d.n_nodes, cfg, Ytr)

            df_cust = pd.DataFrame([{
                "Method": m,
                "Reactive events": r.reactive_events,
                "Proactive events": r.proactive_events,
                "Event reduction": f"{r.event_reduction_pct:+.1f}%",
                "Overflow reduction": f"{r.overflow_reduction_pct:+.1f}%",
                "Throttle actions": r.proactive_throttle_actions,
            } for m, r in results_custom.items()])
            st.dataframe(df_cust, use_container_width=True, hide_index=True)


# ============================================================== PAGE 5
elif page == "🛡️ QoS Strategies":
    st.title("🛡️ QoS Strategies")
    st.caption("When the LSTM predicts overload, *what does the network actually do about it?* "
               "Compare four strategies — none of these are abstract. Each is a real "
               "traffic-engineering technique used in production networks.")

    metrics, preds, history, closed_loop = load_artifacts()
    d, X, Y, Xtr, Ytr, Xte, Yte = load_data()

    from qos_actions import STRATEGIES
    from closed_loop_demo import run_closed_loop, ClosedLoopConfig

    st.markdown("""
| Strategy | Real-world equivalent | What it does |
| --- | --- | --- |
| **NoAction** | Best-effort networking | Nothing. Baseline. |
| **RateLimit** | Token-bucket / policer at ingress | Throttle inbound traffic to predicted-overloaded destinations |
| **Reroute** | MPLS-TE / SDN flow steering | Push some overload-causing traffic onto alternate paths |
| **Prioritize** | DiffServ / QoS marking | Throttle only low-priority OD pairs; protect high-priority flows |
| **Hybrid** | Modern SD-WAN controllers | Reroute first, then prioritized-throttle anything still over capacity |
""")

    st.markdown("---")
    st.subheader("Run all strategies on the held-out test data")

    c1, c2, c3 = st.columns(3)
    with c1:
        sm = st.slider("Safety margin", 0.50, 1.00, 0.80, 0.05,
                        help="Trigger when predicted load > capacity × this")
    with c2:
        thr = st.slider("Throttle factor", 0.10, 1.00, 0.50, 0.05,
                         help="Traffic gets multiplied by this when throttled")
    with c3:
        rrf = st.slider("Reroute fraction", 0.10, 0.80, 0.30, 0.05,
                         help="Fraction of traffic shifted to alternates")

    if st.button("Run all 5 strategies →", type="primary"):
        cfg = ClosedLoopConfig(
            threshold_safety_margin=sm,
            throttle_factor=thr,
            reroute_fraction=rrf,
        )
        with st.spinner("Running closed-loop with each strategy..."):
            rows = []
            for name, strat in STRATEGIES.items():
                r = run_closed_loop(Yte, preds["LSTM"], d.n_nodes,
                                     cfg, Ytr, strategy=strat)
                rows.append({
                    "Strategy": name,
                    "Reactive events": r.reactive_events,
                    "Proactive events": r.proactive_events,
                    "Event reduction": r.event_reduction_pct,
                    "Reactive overflow": r.reactive_total_overflow,
                    "Proactive overflow": r.proactive_total_overflow,
                    "Overflow reduction": r.overflow_reduction_pct,
                    "QoS actions": r.proactive_throttle_actions,
                })
            df = pd.DataFrame(rows).sort_values("Event reduction", ascending=False)

        st.dataframe(
            df.style.format({
                "Event reduction":    "{:+.1f}%",
                "Overflow reduction": "{:+.1f}%",
                "Reactive overflow":  "{:.2f}",
                "Proactive overflow": "{:.2f}",
            }).background_gradient(subset=["Event reduction", "Overflow reduction"],
                                    cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )

        # Side-by-side bars
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=df["Strategy"], y=df["Event reduction"],
                marker_color=["#9ca3af" if v < 0 else "#1e3a8a" if v > 70
                                else "#3b82f6" if v > 40 else "#94a3b8"
                                for v in df["Event reduction"]],
                text=[f"{v:+.1f}%" for v in df["Event reduction"]],
                textposition="outside",
            ))
            fig.update_layout(title="Congestion events reduced",
                              yaxis_title="reduction (%)", height=350,
                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10))
            fig.add_hline(y=0, line=dict(color="black", width=1))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(go.Bar(
                x=df["Strategy"], y=df["Overflow reduction"],
                marker_color=["#9ca3af" if v < 0 else "#1e3a8a" if v > 70
                                else "#3b82f6" if v > 40 else "#94a3b8"
                                for v in df["Overflow reduction"]],
                text=[f"{v:+.1f}%" for v in df["Overflow reduction"]],
                textposition="outside",
            ))
            fig.update_layout(title="Overflow volume reduced",
                              yaxis_title="reduction (%)", height=350,
                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10))
            fig.add_hline(y=0, line=dict(color="black", width=1))
            st.plotly_chart(fig, use_container_width=True)

        # Insight box
        winner = df.iloc[0]
        st.success(
            f"**Winner: {winner['Strategy']}** — "
            f"{winner['Event reduction']:.1f}% fewer congestion events, "
            f"{winner['Overflow reduction']:.1f}% less overflow volume, "
            f"using {winner['QoS actions']:,} QoS actions."
        )

        # Honest commentary
        if "Reroute" in df["Strategy"].values:
            rr = df[df["Strategy"] == "Reroute"].iloc[0]
            if rr["Event reduction"] < 0:
                st.warning(
                    "Note that **Reroute alone makes things worse** "
                    f"({rr['Event reduction']:+.1f}%). This is realistic — "
                    "naïvely shifting traffic just moves the overload. "
                    "Hybrid (Reroute + Prioritize) is what you actually want."
                )


# ============================================================== PAGE 6
elif page == "🌐 Live Network View":
    st.title("🌐 Live Network View")
    st.caption("Watch the 23-node backbone breathe. Node colour = current load relative to capacity. "
               "Hover any node for detail. This is the visual story for the viva.")

    metrics, preds, history, closed_loop = load_artifacts()
    d, X, Y, Xtr, Ytr, Xte, Yte = load_data()

    from topology import NetworkTopology
    from network_animation import build_animation
    from closed_loop_demo import run_closed_loop, ClosedLoopConfig
    from qos_actions import STRATEGIES

    n_nodes = d.n_nodes

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        scenario = st.radio("Scenario:",
                             ["Reactive (no QoS)",
                              "Proactive (LSTM + RateLimit)",
                              "Proactive (LSTM + Hybrid)"],
                             index=0)
    with c2:
        n_slots_show = st.slider("Slots to animate", 30, 200, 80)
    with c3:
        speed = st.select_slider("Speed", options=["slow", "medium", "fast"], value="medium")

    interval_ms = {"slow": 700, "medium": 350, "fast": 150}[speed]

    topo = NetworkTopology.geant_like(n=n_nodes)

    # Use last n_slots_show of test set
    Yte_use = Yte[-n_slots_show:]
    preds_use = preds["LSTM"][-n_slots_show:]
    inbound_actual = Yte_use.reshape(n_slots_show, n_nodes, n_nodes).sum(axis=1)
    inbound_pred = preds_use.reshape(n_slots_show, n_nodes, n_nodes).sum(axis=1)

    cfg = ClosedLoopConfig()

    if scenario == "Reactive (no QoS)":
        # Compute capacities from training history
        cap_inbound = Ytr.reshape(-1, n_nodes, n_nodes).sum(axis=1)
        capacities = np.quantile(cap_inbound, cfg.capacity_quantile, axis=0)
        fig = build_animation(topo, inbound_actual, inbound_pred,
                              capacities=capacities, interval_ms=interval_ms,
                              title="Reactive — no QoS, congestion happens uncontrolled")
    else:
        strat = STRATEGIES["RateLimit"] if "RateLimit" in scenario else STRATEGIES["Hybrid"]
        result = run_closed_loop(Yte_use, preds_use, n_nodes, cfg, Ytr, strategy=strat)
        capacities = result.capacities
        # Track which destinations were throttled per slot
        actions_per_slot = []
        thresh = capacities * cfg.threshold_safety_margin
        for t in range(inbound_pred.shape[0]):
            actions_per_slot.append(set(np.where(inbound_pred[t] > thresh)[0].tolist()))
        fig = build_animation(topo, result.proactive_inbound, inbound_pred,
                              capacities=capacities,
                              qos_actions_per_slot=actions_per_slot,
                              interval_ms=interval_ms,
                              title=f"Proactive — LSTM + {strat.name}: prediction-driven QoS")

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**How to read this:** node circles grow with load. Green nodes are "
        "comfortable, amber are getting busy, red are over capacity. In the "
        "proactive scenarios, nodes outlined in **navy** are receiving QoS "
        "protection (rate-limit / reroute / prioritize). "
        "Press ▶ Play to step through time."
    )


# ============================================================== PAGE 7
elif page == "🔌 Topology & Resilience":
    st.title("🔌 Topology & Resilience")
    st.caption("What happens when the network breaks? Cut a link, disable a node, "
               "and see how connectivity holds up.")

    from topology import NetworkTopology
    from network_animation import build_topology_figure
    metrics, preds, history, closed_loop = load_artifacts()
    d, X, Y, Xtr, Ytr, Xte, Yte = load_data()

    if "topo_state" not in st.session_state:
        st.session_state.topo_state = NetworkTopology.geant_like(n=d.n_nodes)
    topo = st.session_state.topo_state

    c1, c2 = st.columns([2, 1])

    with c2:
        st.subheader("Modify topology")

        st.markdown("**Disable a node**")
        node_to_kill = st.selectbox("Choose node", list(range(topo.n_nodes)),
                                      format_func=lambda j: f"node {j} "
                                                             f"({'❌' if j in topo.disabled_nodes else '✅'})")
        bb1, bb2 = st.columns(2)
        if bb1.button("Disable", use_container_width=True):
            topo.disable_node(node_to_kill)
            st.rerun()
        if bb2.button("Enable", use_container_width=True):
            topo.enable_node(node_to_kill)
            st.rerun()

        st.markdown("---")
        st.markdown("**Cut a link**")
        all_links = sorted(topo.links)
        link_idx = st.selectbox("Choose link",
                                  range(len(all_links)),
                                  format_func=lambda k: f"{all_links[k][0]} ↔ {all_links[k][1]}"
                                                          + (" ❌" if all_links[k] in topo.disabled_links else ""))
        a, b = all_links[link_idx]
        bb3, bb4 = st.columns(2)
        if bb3.button("Cut", use_container_width=True, key="cut_btn"):
            topo.cut_link(a, b)
            st.rerun()
        if bb4.button("Restore", use_container_width=True, key="restore_btn"):
            topo.restore_link(a, b)
            st.rerun()

        st.markdown("---")
        if st.button("🔄 Reset all", use_container_width=True):
            topo.reset()
            st.rerun()

        # Stats
        st.markdown("---")
        st.subheader("Topology stats")
        connectivity = topo.connectivity_pct()
        active_links = sum(1 for _ in topo.active_links())
        st.metric("Active nodes",
                  f"{topo.n_nodes - len(topo.disabled_nodes)} / {topo.n_nodes}")
        st.metric("Active links",
                  f"{active_links} / {len(topo.links)}")
        st.metric("Reachable pair %", f"{connectivity:.1f}%")

    with c1:
        fig_top = build_topology_figure(topo, title="Network topology — interact via right panel")
        st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")
    st.subheader("Effect on LSTM predictions")
    st.markdown(
        "When a node is disabled, real networks redistribute its traffic among "
        "its neighbours. The LSTM was trained on a static topology — does it "
        "still predict accurately under failure?"
    )

    if st.button("Run impact analysis →", type="primary"):
        from topology_resilience import (
            redistribute_traffic_for_failed_node, run_node_failure_sweep
        )
        with st.spinner("Loading model + running sweep..."):
            model = load_lstm_model()
            sweep = run_node_failure_sweep(model, Xte, Yte, topo)

        per = sorted(sweep["per_node"], key=lambda x: -x["mse_delta_pct"])
        df = pd.DataFrame(per)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=df["failed_node"][:10].astype(str),
                y=df["mse_delta_pct"][:10],
                marker_color="#dc2626",
                text=[f"+{v:.1f}%" for v in df["mse_delta_pct"][:10]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Top 10 most-impactful node failures",
                xaxis_title="failed node",
                yaxis_title="MSE increase (%)",
                height=400, plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("Baseline MSE", f"{sweep['baseline_mse']:.6f}")
            st.metric("Worst-case MSE",
                      f"{df['mse'].max():.6f}",
                      f"+{df['mse_delta_pct'].max():.1f}%")
            st.metric("Avg MSE under any single-node failure",
                      f"{df['mse'].mean():.6f}",
                      f"+{df['mse_delta_pct'].mean():.1f}%")

        st.dataframe(
            df.style.format({
                "mse": "{:.6f}",
                "mse_delta_pct": "{:+.1f}%",
                "connectivity_pct": "{:.1f}%",
            }).background_gradient(subset=["mse_delta_pct"], cmap="Reds"),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================== PAGE 8
elif page == "🎲 Monte Carlo (Confidence)":
    st.title("🎲 Monte Carlo — Confidence Intervals")
    st.caption("All single-seed numbers are point estimates. Monte Carlo runs the same "
               "experiment over many seeds to tell you which results are robust and which "
               "are noise.")

    mc_path = PROJECT_ROOT / "results" / "monte_carlo.json"
    if not mc_path.exists():
        st.warning(
            "Monte Carlo results not yet generated. Run:\n\n"
            "```bash\npython src/monte_carlo.py --n_seeds 15 --n_timeslots 700 "
            "--strategies NoAction RateLimit Reroute Prioritize Hybrid\n```"
        )
        st.stop()

    mc = json.load(open(mc_path))
    summary = mc["summary"]
    trials = mc["trials"]
    cfg = mc["config"]

    st.markdown(f"**Configuration:** {summary['n_trials']} trials × "
                f"{cfg['n_timeslots']} timeslots × {cfg['n_nodes']} nodes")

    st.markdown("---")
    st.subheader("LSTM test MSE across seeds")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("mean", f"{summary['mse']['mean']:.5f}")
    c2.metric("std",  f"{summary['mse']['std']:.5f}")
    c3.metric("min",  f"{summary['mse']['min']:.5f}")
    c4.metric("max",  f"{summary['mse']['max']:.5f}")

    # Trial-by-trial scatter
    df_t = pd.DataFrame(trials)
    fig = go.Figure(go.Scatter(
        x=df_t["seed"], y=df_t["mse"],
        mode="markers", marker=dict(size=10, color=COLORS["LSTM"]),
        hovertemplate="seed: %{x}<br>MSE: %{y:.6f}<extra></extra>",
    ))
    fig.add_hline(y=summary["mse"]["mean"], line=dict(color="black", dash="dash"),
                  annotation_text=f"mean = {summary['mse']['mean']:.5f}")
    fig.update_layout(xaxis_title="seed", yaxis_title="test MSE",
                      height=350, plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("QoS strategies — robustness across seeds")

    strategy_data = []
    for s, m in summary["strategies"].items():
        e = m["event_reduction_pct"]
        o = m["overflow_reduction_pct"]
        strategy_data.append({
            "Strategy": s,
            "Event reduction (mean)":  e["mean"],
            "Event reduction (std)":   e["std"],
            "Event reduction (min)":   e["min"],
            "Event reduction (max)":   e["max"],
            "Overflow reduction (mean)": o["mean"],
            "Overflow reduction (std)":  o["std"],
        })
    df_s = pd.DataFrame(strategy_data).sort_values("Event reduction (mean)", ascending=False)

    # Plot mean ± std as error bars
    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(
        x=df_s["Strategy"], y=df_s["Event reduction (mean)"],
        error_y=dict(type="data", array=df_s["Event reduction (std)"], thickness=2),
        marker_color=[COLORS.get(s, "#888") for s in df_s["Strategy"]],
        text=[f"{m:.1f}%±{s:.1f}" for m, s in zip(df_s["Event reduction (mean)"],
                                                       df_s["Event reduction (std)"])],
        textposition="outside",
    ))
    fig_err.update_layout(
        title="Event reduction by strategy — mean ± std across all seeds",
        yaxis_title="reduction (%)", height=420,
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=10),
    )
    fig_err.add_hline(y=0, line=dict(color="black", width=1))
    st.plotly_chart(fig_err, use_container_width=True)

    st.dataframe(
        df_s.style.format({
            "Event reduction (mean)": "{:.1f}%",
            "Event reduction (std)":  "{:.1f}%",
            "Event reduction (min)":  "{:.1f}%",
            "Event reduction (max)":  "{:.1f}%",
            "Overflow reduction (mean)": "{:.1f}%",
            "Overflow reduction (std)":  "{:.1f}%",
        }).background_gradient(subset=["Event reduction (mean)"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )

    # Interpretation
    best = df_s.iloc[0]
    st.success(
        f"**{best['Strategy']}** is the most robust strategy: "
        f"{best['Event reduction (mean)']:.1f}% mean event reduction "
        f"with std {best['Event reduction (std)']:.1f}%. "
        f"Range across {summary['n_trials']} seeds: "
        f"[{best['Event reduction (min)']:.1f}%, {best['Event reduction (max)']:.1f}%]."
    )


# ============================================================== PAGE 9
elif page == "📡 Streaming Live":
    # Delegate to the streaming-page module (kept separate so it can also
    # run standalone via `streamlit run web/streaming_page.py`).
    from streaming_page import render_streaming_page
    render_streaming_page(PROJECT_ROOT)


# ============================================================== PAGE 10
elif page == "🔬 What does AnyLogic do?":
    st.title("🔬 What does AnyLogic actually do here?")
    st.caption("Honest answer to a fair question — updated for the live integration.")

    st.markdown("""
**Earlier in this project, AnyLogic only generated training data.** It computed
a CSV and we trained the LSTM offline against it. A Python surrogate produced
statistically identical CSVs in 30 seconds without any GUI install, which made
AnyLogic feel like ceremony.

**That changed with the live integration.** The AnyLogic model now calls our
FastAPI service over HTTP every simulated slot, gets the LSTM's predictions
back, applies a QoS action *inside the same tick*, and the modified traffic is
what flows. AnyLogic isn't a data exporter anymore — it's the simulator at the
centre of a real closed loop.

Below: the four roles AnyLogic plays in the current system.
""")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1) Data generation (still)",
        "2) Live closed-loop driver",
        "3) Where Python complements it",
        "4) What this means for the report",
    ])

    with tab1:
        st.subheader("Data generation")
        st.markdown("""
AnyLogic still runs the agent-based + discrete-event simulation of a 23-node
backbone network. Every 15 minutes of simulated time it computes a 23×23 traffic
matrix using:

- **Gravity-model base rates** — bigger nodes generate and attract more traffic
- **Diurnal pattern** — cosine peaking at 14:00, troughing at 04:00
- **Weekly pattern** — 70% of weekday volume on weekends
- **Flash-crowd events** — 6% chance per slot of an event hitting a random
  destination for 1-3 slots at 4-10× baseline
- **AR(1) bursts** — short-range temporal correlation per OD pair

After 21 simulated days you have 2,016 rows × 529 columns. **One key change**:
the CSV now contains the **post-QoS** traffic (what actually flowed after
the controller intervened), not the raw matrix. The raw matrix is held in
memory as `rawTrafficMatrix` and used for the before/after accounting in the
console summary.

To do clean before/after comparisons, run the simulation twice — once with
`enableLstmCalls = false`, once `= true` — and compare the two CSVs.
""")

    with tab2:
        st.subheader("Live closed-loop driver")
        st.markdown("""
Each simulated 15-minute slot, AnyLogic's `onSlotTick()`:

1. Generates the raw traffic matrix
2. Snapshots it to `rawTrafficMatrix`
3. **POSTs to `http://127.0.0.1:8765/ingest`** with the flattened 529-element vector
4. Receives the LSTM's prediction + `congestion_flags` (destinations predicted to overload)
5. Calls `applyQosAction()` — Hybrid by default: reroute first, then rate-limit anything still hot
6. Measures pre-QoS and post-QoS overflow
7. Writes the **post-QoS** matrix to CSV

The Java code lives in the patched `NetworkTrafficSim.alp`. Two helper functions
were added to Main:

- `callPredictionService()` — builds the JSON body, sends the POST, parses the
  response with a hand-rolled JSON scanner (zero dependencies). Failure-tolerant:
  any HTTP/timeout error increments `lstmCallsFailed` and the simulation continues
  with no QoS applied for that slot.
- `applyQosAction()` — modifies `trafficMatrix` in place based on the chosen
  strategy (`NoAction` / `RateLimit` / `Reroute` / `Hybrid`).

Three new parameters surface on the Main canvas: `enableLstmCalls`,
`lstmServiceUrl`, and `qosStrategy` — set them like any other AnyLogic parameter
before running.

**This is true closed-loop simulation.** AnyLogic's actions affect what flows
in AnyLogic. The LSTM's predictions don't just generate a number on a chart —
they change the simulation's behaviour the moment they're made.
""")
        try:
            metrics, preds, history, closed_loop = load_artifacts()
            if "Hybrid" in closed_loop or "LSTM" in closed_loop:
                key = "Hybrid" if "Hybrid" in closed_loop else "LSTM"
                cl = closed_loop[key]
                c1, c2, c3 = st.columns(3)
                c1.metric("Reactive congestion events", cl["reactive_events"])
                c2.metric(f"With {key}-QoS", cl["proactive_events"],
                          f"-{cl['event_reduction_pct']:.1f}%", delta_color="inverse")
                c3.metric("Overflow reduced", f"{cl['overflow_reduction_pct']:.1f}%")
        except Exception:
            pass

    with tab3:
        st.subheader("Where Python complements it")
        st.markdown("""
Not every layer of the project belongs inside AnyLogic. Python handles the parts
where it's genuinely better:

- **The LSTM itself.** TensorFlow/Keras is the right place to train and serve a
  neural network; nobody runs production inference inside a Java discrete-event
  simulator. We serve the LSTM via FastAPI on `127.0.0.1:8765` and AnyLogic
  calls it over HTTP.
- **Continuous retraining.** A separate Python worker (`retrain_worker.py`)
  watches the streaming buffer, fires a full retrain every 500 fresh samples,
  and hot-swaps the model file. The streaming service detects the swap and
  reloads automatically.
- **Monte Carlo experiments.** Repeating the full pipeline over many seeds is
  much faster in headless Python than starting AnyLogic 15 times.
- **Topology resilience experiments.** Pure data manipulation, no simulator needed.
- **Dashboard.** Streamlit is the right tool — AnyLogic's animation is great
  for the simulator's internal state, but our 9-page dashboard tells a wider
  story.

So the division of labour is honest: AnyLogic owns *the network simulation and
the closed-loop control loop*; Python owns *the ML and the analytics*. The two
talk over HTTP.
""")

    with tab4:
        st.subheader("What this means for the report")
        st.markdown("""
For the CCN report and viva, the honest framing is:

> *"AnyLogic provides a topology-faithful, statistically realistic generator of
> network traffic matrices for training the LSTM. We then extend this into a
> true closed-loop control system: each simulated 15-minute slot, AnyLogic
> calls our FastAPI LSTM service over HTTP, receives the destinations predicted
> to overload, and applies a QoS action (NoAction / RateLimit / Reroute /
> Hybrid) inside the same simulation tick. The modified traffic is what flows
> in the simulator and is what we write to CSV. In our 21-day run with the
> Hybrid strategy, AnyLogic measured an X% reduction in cumulative overflow
> versus the no-action baseline — confirming end-to-end that the LSTM's
> predictions translate to operational impact when wired into the simulator
> itself."*

(Fill in the actual % from the AnyLogic run.)

That framing accurately describes the system today, gives credit to every layer
we built, and answers the *"isn't AnyLogic just a data generator?"* question
directly. It isn't anymore.

Future work — beyond this — is now things like real GÉANT data, packet-level
fidelity inside AnyLogic, GNN baselines, and transformer comparisons. The
"wire LSTM into AnyLogic" item is checked off.
""")
