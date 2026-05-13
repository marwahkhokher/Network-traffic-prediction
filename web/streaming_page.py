"""
streaming_page.py
-----------------
Streamlit page that monitors the live streaming stack — buffer growth,
prediction rate, drift score, retrain history, and a live event feed.

To wire this into your existing dashboard (web/app.py), add to the
sidebar's page list:

    "📡 Streaming Live",

and at the bottom of app.py (just before the AnyLogic page), add:

    elif page == "📡 Streaming Live":
        from streaming_page import render_streaming_page
        render_streaming_page(PROJECT_ROOT)

You can also run this page standalone:

    streamlit run web/streaming_page.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


def _get(url: str, timeout: float = 2.0) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def render_streaming_page(project_root: Path,
                          service_url: str = "http://localhost:8765") -> None:
    st.title("📡 Streaming Live")
    st.caption("Continuous autonomous retraining — the LSTM you trained "
               "offline, now updating itself in real time from incoming traffic.")

    # --- Status ----------------------------------------------------------
    info = _get(f"{service_url}/model_info")
    metrics = _get(f"{service_url}/metrics")

    if info is None or metrics is None:
        st.warning(
            f"Cannot reach streaming service at **{service_url}**.\n\n"
            "Start it with:\n"
            "```bash\npython scripts/run_streaming_stack.py\n```"
        )
        return

    if info.get("status") != "ok":
        st.error("Service is up but no model is loaded. Run `python src/train.py` first, "
                 "then copy `results/lstm_model.keras` → `results/lstm_streaming.keras`.")
        return

    # --- KPIs ------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model version",
              info["model_version"],
              help="Increments every time the retrain worker swaps in a new model.")
    c2.metric("Total ingested",
              f"{info['buffer']['total_ingested']:,}",
              help="Lifetime count of slots received from AnyLogic / clients.")
    c3.metric("In buffer",
              f"{info['buffer']['in_buffer']:,}",
              help=f"Rolling window of recent samples available to the retrain worker.")
    c4.metric("Since last retrain",
              f"{info['samples_since_last_retrain']:,}",
              help="Counter resets on every successful retrain.")
    drift_pct = (info["drift_score"] or 0) * 100
    c5.metric("Drift score",
              f"{info['drift_score']:.2f}×",
              delta="flagged" if info["drift_flagged"] else "normal",
              delta_color="inverse" if info["drift_flagged"] else "normal",
              help="rolling_loss / baseline_loss. >2.5× triggers a retrain.")

    st.markdown("---")

    # --- Two-column layout: charts + raw status ------------------------
    left, right = st.columns([2, 1])

    with left:
        # Retrain history
        history_path = project_root / "results" / "retrain_history.jsonl"
        if history_path.exists():
            rows = []
            with open(history_path) as f:
                for line in f:
                    try: rows.append(json.loads(line))
                    except Exception: pass
            if rows:
                df = pd.DataFrame(rows)
                df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["dt"], y=df["val_loss"],
                    mode="lines+markers",
                    name="val_loss (after retrain)",
                    line=dict(color="#0EA5A5", width=2.5),
                    marker=dict(size=8, color="#0EA5A5"),
                ))
                fig.add_trace(go.Scatter(
                    x=df["dt"], y=df["train_loss"],
                    mode="lines+markers",
                    name="train_loss",
                    line=dict(color="#94A3B8", width=1.5, dash="dot"),
                    marker=dict(size=5),
                ))
                fig.update_layout(
                    title="Autonomous retrain history — loss after every model swap",
                    xaxis_title="time",
                    yaxis_title="MSE",
                    height=380,
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=50, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Recent retrain table
                df_show = df.tail(8).copy()
                df_show["timestamp"] = df_show["dt"].dt.strftime("%H:%M:%S")
                cols = ["timestamp", "samples_used", "train_windows",
                        "val_windows", "train_loss", "val_loss",
                        "train_seconds"]
                st.dataframe(
                    df_show[cols].style.format({
                        "train_loss": "{:.6f}", "val_loss": "{:.6f}",
                        "train_seconds": "{:.1f}",
                    }),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No retrains yet. Once you've fed in enough samples "
                        "(default: 500), the worker will trigger automatically.")
        else:
            st.info(
                "Retrain history will appear here after the first autonomous retrain. "
                f"Currently at **{info['samples_since_last_retrain']}** / "
                "**500** samples toward the next trigger."
            )

    with right:
        st.subheader("Live counters")
        st.markdown(f"""
| Counter | Value |
|---|---:|
| Ingested total | `{metrics.get('ingested_total', 0):,}` |
| Predictions total | `{metrics.get('predictions_total', 0):,}` |
| Partial fits total | `{metrics.get('partial_fits_total', 0):,}` |
| Congestion flags | `{metrics.get('congestion_flags_total', 0):,}` |
| Model reloads | `{metrics.get('model_reloads_total', 0):,}` |
| WS clients | `{metrics.get('websocket_clients', 0)}` |
| Errors | `{metrics.get('errors_total', 0):,}` |
""")
        st.markdown("---")
        st.subheader("Online-learning")
        st.markdown(f"""
- baseline loss: `{info.get('baseline_loss')}`
- rolling mean: `{info.get('rolling_loss_mean')}`
- last online loss: `{info.get('last_online_loss')}`
- drift multiplier threshold: `2.5×`
""")

        st.markdown("---")
        if st.button("🔄 Reset drift flag", use_container_width=True):
            try:
                requests.post(f"{service_url}/admin/reset_drift", timeout=3)
                st.success("Drift flag cleared.")
            except Exception as e:
                st.error(f"Failed: {e}")

        st.caption(f"Polling `{service_url}/` every 5 s.")

    # Auto-refresh every 5 seconds
    st.markdown("---")
    auto = st.checkbox("Auto-refresh every 5 s", value=True)
    if auto:
        time.sleep(5)
        st.rerun()


# Standalone entrypoint
if __name__ == "__main__" or True:
    if "__streamlit__" in str(globals().get("__name__", "")):
        render_streaming_page(Path(__file__).resolve().parent.parent)
    else:
        # When running as `streamlit run web/streaming_page.py`
        render_streaming_page(Path(__file__).resolve().parent.parent)
