"""
streaming_app.py
----------------
FastAPI service that ties together the streaming buffer, the online
learner, and the AnyLogic ↔ cloud loop.

Endpoints
=========

  POST  /ingest             one new traffic vector (sync prediction + online update)
  POST  /ingest_batch       a small batch
  GET   /predict_next       prediction from the current tail of the buffer
  GET   /model_info         current model version, samples seen, drift state, etc.
  POST  /admin/reset_drift  manually clear the drift flag
  WS    /stream             live broadcast of every prediction + congestion flag
  GET   /                   simple status HTML
  GET   /metrics            Prometheus-style counters

Architecture
============

  AnyLogic / live monitor
        |
        | HTTP POST /ingest  { "vector": [...], "timestamp": 17... }
        v
  ┌────────────────────┐         ┌──────────────────────┐
  │ FastAPI streaming  │ uses    │  StreamingBuffer     │
  │ service (this)     │────────►│  (in-memory deque +  │
  │                    │         │   disk snapshot)     │
  │ - predicts         │         └──────────────────────┘
  │ - partial_fit                          ▲
  │ - emits WS msgs    │                   │ snapshot file
  │                    │         ┌──────────────────────┐
  │ - polls model file │         │  retrain_worker.py   │
  │   for hot reload   │◄────────┤  (separate process)  │
  └────────────────────┘  atomic │                      │
            ▲             swap   │  - polls snapshot    │
            │                    │  - full retrain      │
            │ WS push            │  - atomic model swap │
            v                    └──────────────────────┘
   Streamlit dashboard /
   AnyLogic policy hook

Run:
  uvicorn api.streaming_app:app --reload --port 8765
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np

# make src/ importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streaming_buffer import get_global_buffer       # noqa: E402
from online_learner import OnlineLearner             # noqa: E402

try:
    from qos_actions import STRATEGIES               # noqa: E402
    _HAVE_QOS = True
except Exception:
    _HAVE_QOS = False


# ====================================================================== config
MODEL_PATH      = Path(os.environ.get("STREAMING_MODEL", "results/lstm_streaming.keras"))
STATE_PATH      = Path(os.environ.get("STREAMING_STATE", "results/lstm_streaming.state.json"))
BUFFER_PATH     = Path(os.environ.get("STREAMING_BUFFER", "results/streaming_buffer.pkl"))
SCALE_PATH      = Path(os.environ.get("STREAMING_SCALE", "results/scale.txt"))
WINDOW          = int(os.environ.get("STREAMING_WINDOW", "12"))
BUFFER_CAPACITY = int(os.environ.get("STREAMING_BUFFER_CAPACITY", "8000"))
ONLINE_LR       = float(os.environ.get("STREAMING_ONLINE_LR", "1e-5"))


# ====================================================================== state
class AppState:
    learner: Optional[OnlineLearner] = None
    websocket_clients: Set[WebSocket] = set()
    metrics: dict = {
        "ingested_total":          0,
        "predictions_total":       0,
        "partial_fits_total":      0,
        "congestion_flags_total":  0,
        "model_reloads_total":     0,
        "websocket_broadcasts":    0,
        "errors_total":            0,
        "started_at":              None,
    }


STATE = AppState()


def _load_scale() -> float:
    if SCALE_PATH.exists():
        try:
            return float(SCALE_PATH.read_text().strip())
        except Exception:
            pass
    return 1.0


def _json_safe_float(v) -> Optional[float]:
    """JSON forbids NaN/Inf — replace with None so the response is valid."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    import math
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _json_safe_list(arr: np.ndarray) -> list:
    """Same for numpy arrays — NaN/Inf cells become 0.0 (safer for downstream)."""
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr.astype(float).tolist()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise learner + buffer; start background tasks."""
    scale = _load_scale()
    if not MODEL_PATH.exists():
        # On a fresh project, bootstrap from the offline-trained model.
        alt = PROJECT_ROOT / "results" / "lstm_model.keras"
        if alt.exists():
            print(f"[streaming_app] bootstrapping {MODEL_PATH} from {alt}")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(alt, MODEL_PATH)
        else:
            print(f"[streaming_app] WARNING: no model at {MODEL_PATH} and no "
                  f"offline fallback at {alt}. Run `python src/train.py` first.")

    if MODEL_PATH.exists():
        STATE.learner = OnlineLearner(
            model_path=MODEL_PATH,
            state_path=STATE_PATH,
            scale=scale,
            online_lr=ONLINE_LR,
        )

    # Materialise the shared buffer (also restores from disk if a snapshot exists)
    get_global_buffer(capacity=BUFFER_CAPACITY, flush_path=BUFFER_PATH)

    STATE.metrics["started_at"] = time.time()

    # Background: every 5 s, check whether the worker has written a newer model
    async def _reload_watcher():
        while True:
            await asyncio.sleep(5.0)
            try:
                if STATE.learner and STATE.learner.reload_if_updated():
                    STATE.metrics["model_reloads_total"] += 1
                    await _broadcast({
                        "event": "model_reloaded",
                        "version": STATE.learner.state().model_version,
                        "timestamp": time.time(),
                    })
                    print(f"[streaming_app] hot-reloaded model "
                          f"(version {STATE.learner.state().model_version})")
            except Exception as e:
                STATE.metrics["errors_total"] += 1
                print(f"[streaming_app] reload watcher error: {e}")

    watcher_task = asyncio.create_task(_reload_watcher())
    print(f"[streaming_app] startup complete — listening for traffic.")
    try:
        yield
    finally:
        watcher_task.cancel()
        get_global_buffer().force_flush()
        print("[streaming_app] shutdown complete.")


app = FastAPI(
    title="Network Traffic Prediction — Streaming Service",
    description=(
        "Real-time ingestion + prediction + online learning for the "
        "LSTM traffic-matrix forecaster. Pairs with retrain_worker.py."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ====================================================================== schemas
class IngestRequest(BaseModel):
    vector: List[float] = Field(
        ...,
        description="Flattened N² traffic vector for one slot, in raw "
                    "(unscaled) units. Will be normalised internally.",
    )
    timestamp: Optional[float] = None
    do_partial_fit: bool = Field(
        True,
        description="If true and at least W previous samples exist, "
                    "use this sample as a partial_fit target.",
    )


class IngestBatchRequest(BaseModel):
    vectors: List[List[float]]
    timestamps: Optional[List[float]] = None
    do_partial_fit: bool = True


class PredictResponse(BaseModel):
    prediction_raw: List[float]
    prediction_scaled: List[float]
    n_nodes: int
    congestion_flags: List[dict]
    model_version: int
    buffer_size: int
    used_partial_fit: bool
    partial_fit_loss: Optional[float] = None


# ====================================================================== helpers
async def _broadcast(msg: dict) -> None:
    """Push a message to every connected websocket client. Drop on failure."""
    if not STATE.websocket_clients:
        return
    payload = json.dumps(msg, default=str)
    dead = []
    for ws in list(STATE.websocket_clients):
        try:
            await ws.send_text(payload)
            STATE.metrics["websocket_broadcasts"] += 1
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.websocket_clients.discard(ws)


def _detect_congestion(pred_scaled: np.ndarray, n_nodes: int,
                       quantile: float = 0.95) -> List[dict]:
    """Quick congestion-flag heuristic: destinations whose predicted
    inbound load exceeds the quantile of the buffered history."""
    buffer = get_global_buffer()
    flags: List[dict] = []
    if len(buffer) < 50:
        return flags
    try:
        recent = buffer.get_recent(min(2000, len(buffer)))
        recent_in = recent.reshape(-1, n_nodes, n_nodes).sum(axis=1)
        thresholds = np.quantile(recent_in, quantile, axis=0)
    except Exception:
        return flags

    pred_mat = pred_scaled.reshape(n_nodes, n_nodes)
    pred_in = pred_mat.sum(axis=0)
    for j in np.where(pred_in > thresholds)[0]:
        flags.append({
            "destination": int(j),
            "predicted_inbound": float(pred_in[j]),
            "threshold": float(thresholds[j]),
            "excess_ratio": float(pred_in[j] / thresholds[j]),
        })
    return flags


# ====================================================================== endpoints
@app.get("/", response_class=HTMLResponse)
def index():
    state = STATE.learner.state() if STATE.learner else None
    buf = get_global_buffer().stats()
    uptime = (time.time() - STATE.metrics["started_at"]) if STATE.metrics["started_at"] else 0
    return f"""
<!DOCTYPE html><html><head><title>Streaming Service</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; background: #0B1A3D;
       color: #F8FAFC; margin: 0; padding: 40px; }}
.card {{ background: #071228; border-left: 4px solid #0EA5A5;
         padding: 20px 30px; border-radius: 4px; margin-bottom: 16px; max-width: 800px; }}
h1 {{ color: #0EA5A5; }}
table {{ border-collapse: collapse; }} td {{ padding: 4px 16px 4px 0; }}
.label {{ color: #94A3B8; }} .v {{ font-family: Consolas, monospace; color: #F4A03F; }}
a {{ color: #0EA5A5; }}
</style></head>
<body>
<h1>Network Traffic — Streaming Service</h1>
<div class="card">
  <h3>Status</h3>
  <table>
    <tr><td class="label">model loaded</td><td class="v">{'yes' if STATE.learner else 'NO'}</td></tr>
    <tr><td class="label">model path</td><td class="v">{MODEL_PATH}</td></tr>
    <tr><td class="label">model version</td><td class="v">{state.model_version if state else '-'}</td></tr>
    <tr><td class="label">window W</td><td class="v">{WINDOW}</td></tr>
    <tr><td class="label">samples in buffer</td><td class="v">{buf.in_buffer}</td></tr>
    <tr><td class="label">total ingested</td><td class="v">{buf.total_ingested}</td></tr>
    <tr><td class="label">samples since retrain</td><td class="v">{state.samples_since_last_retrain if state else '-'}</td></tr>
    <tr><td class="label">drift score</td><td class="v">{(state.drift_score if state else 0):.2f}</td></tr>
    <tr><td class="label">drift flagged</td><td class="v">{(state.drift_flagged if state else False)}</td></tr>
    <tr><td class="label">uptime</td><td class="v">{uptime:.0f}s</td></tr>
  </table>
</div>
<div class="card">
  <h3>Endpoints</h3>
  <ul>
    <li><code>POST /ingest</code> — submit one traffic vector</li>
    <li><code>POST /ingest_batch</code> — submit many</li>
    <li><code>GET /predict_next</code> — predict from current tail</li>
    <li><code>GET /model_info</code> — JSON status</li>
    <li><code>GET /metrics</code> — counters</li>
    <li><code>WS /stream</code> — live event feed</li>
  </ul>
</div>
</body></html>
"""


@app.get("/model_info")
def model_info():
    if STATE.learner is None:
        return {"status": "no_model"}
    buf = get_global_buffer().stats()
    state = STATE.learner.state()

    def _clean(v):
        """JSON can't encode NaN/Inf; map them to None."""
        if isinstance(v, float):
            import math
            if math.isnan(v) or math.isinf(v):
                return None
        return v

    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_version": state.model_version,
        "samples_seen_online": state.samples_seen_online,
        "samples_since_last_retrain": state.samples_since_last_retrain,
        "last_online_loss": _clean(state.last_online_loss),
        "rolling_loss_mean": _clean(state.rolling_loss_mean),
        "baseline_loss": _clean(state.baseline_loss),
        "drift_score": _clean(state.drift_score),
        "drift_flagged": state.drift_flagged,
        "buffer": {
            "in_buffer": buf.in_buffer,
            "total_ingested": buf.total_ingested,
            "last_ingest_ts": _clean(buf.last_ingest_ts),
            "n_features": buf.n_features,
        },
        "window": WINDOW,
        "scale": STATE.learner.scale,
    }


@app.get("/metrics")
def metrics():
    """Prometheus-style metrics (text format would be exposed at /metrics
    in a real deployment; here we return JSON for simplicity)."""
    out = dict(STATE.metrics)
    if STATE.learner:
        out["model_version"] = STATE.learner.state().model_version
    out["buffer_size"] = len(get_global_buffer())
    out["websocket_clients"] = len(STATE.websocket_clients)
    return out


@app.post("/ingest", response_model=PredictResponse)
async def ingest(req: IngestRequest):
    if STATE.learner is None:
        raise HTTPException(503, "no model loaded")
    vec = np.asarray(req.vector, dtype=np.float32)
    n_features = vec.size
    n_nodes = int(round(np.sqrt(n_features)))
    if n_nodes * n_nodes != n_features:
        raise HTTPException(400, f"vector length {n_features} is not a perfect square")

    # Normalise into the same /max space the model was trained on
    scaled = vec / STATE.learner.scale

    buffer = get_global_buffer()
    buffer.append(scaled, req.timestamp)
    STATE.metrics["ingested_total"] += 1

    # Predict using the previous W samples (window ending BEFORE this slot).
    # If we don't have W yet, predict zeros so the response shape stays consistent.
    # We then optionally use the newly-arrived sample as a partial_fit target,
    # with the previous-W window as input.
    used_partial_fit = False
    partial_loss = None
    prediction_scaled = np.zeros(n_features, dtype=np.float32)

    if len(buffer) > WINDOW:
        # Window ending right BEFORE the just-arrived sample
        recent = buffer.get_last_n(WINDOW + 1)            # (W+1, F)
        input_window = recent[:-1]                         # (W, F)
        target       = recent[-1]                          # (F,)

        # 1. Predict (what would we have said for this slot, given the prior W?)
        prediction_scaled = STATE.learner.predict(input_window)
        STATE.metrics["predictions_total"] += 1

        # 2. Partial-fit using the just-arrived sample as ground truth
        if req.do_partial_fit:
            try:
                partial_loss = STATE.learner.partial_fit(input_window, target)
                STATE.metrics["partial_fits_total"] += 1
                used_partial_fit = True
            except Exception as e:
                STATE.metrics["errors_total"] += 1
                print(f"[streaming_app] partial_fit failed: {e}")

    # 3. Detect congestion
    flags = _detect_congestion(prediction_scaled, n_nodes)
    STATE.metrics["congestion_flags_total"] += len(flags)

    response = PredictResponse(
        prediction_raw=_json_safe_list(prediction_scaled * STATE.learner.scale),
        prediction_scaled=_json_safe_list(prediction_scaled),
        n_nodes=n_nodes,
        congestion_flags=flags,
        model_version=STATE.learner.state().model_version,
        buffer_size=len(buffer),
        used_partial_fit=used_partial_fit,
        partial_fit_loss=_json_safe_float(partial_loss),
    )

    # 4. Broadcast over WS (fire-and-forget)
    await _broadcast({
        "event": "prediction",
        "timestamp": time.time(),
        "model_version": response.model_version,
        "buffer_size": response.buffer_size,
        "partial_fit_loss": _json_safe_float(partial_loss),
        "n_congestion_flags": len(flags),
        "congestion_flags": flags[:5],  # top 5 only
    })

    return response


@app.post("/ingest_batch")
async def ingest_batch(req: IngestBatchRequest):
    if STATE.learner is None:
        raise HTTPException(503, "no model loaded")
    arr = np.asarray(req.vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(400, "vectors must be a 2-D array")

    summary = {"n": arr.shape[0], "predictions": 0, "partial_fits": 0, "errors": 0}
    for i in range(arr.shape[0]):
        ts = req.timestamps[i] if req.timestamps else None
        try:
            await ingest(IngestRequest(
                vector=arr[i].tolist(),
                timestamp=ts,
                do_partial_fit=req.do_partial_fit,
            ))
            summary["predictions"] += 1
            if req.do_partial_fit:
                summary["partial_fits"] += 1
        except Exception as e:
            summary["errors"] += 1
            print(f"[streaming_app] batch ingest error at index {i}: {e}")
    return summary


@app.get("/predict_next")
def predict_next():
    """Predict the next slot from the current buffer tail, without
    ingesting anything new. Useful for AnyLogic to query 'what would
    happen next' without contributing to online updates."""
    if STATE.learner is None:
        raise HTTPException(503, "no model loaded")
    buffer = get_global_buffer()
    if len(buffer) < WINDOW:
        raise HTTPException(412, f"need at least {WINDOW} samples; have {len(buffer)}")
    window = buffer.get_last_n(WINDOW)
    pred_scaled = STATE.learner.predict(window)
    n_features = pred_scaled.size
    n_nodes = int(round(np.sqrt(n_features)))
    flags = _detect_congestion(pred_scaled, n_nodes)
    STATE.metrics["predictions_total"] += 1
    return {
        "prediction_raw": _json_safe_list(pred_scaled * STATE.learner.scale),
        "prediction_scaled": _json_safe_list(pred_scaled),
        "n_nodes": n_nodes,
        "congestion_flags": flags,
        "model_version": STATE.learner.state().model_version,
    }


@app.post("/admin/reset_drift")
def reset_drift():
    if STATE.learner is None:
        raise HTTPException(503, "no model loaded")
    # Clears the drift flag by saving a fresh state file
    state = STATE.learner.state()
    state.drift_flagged = False
    state.drift_score = 0.0
    STATE.learner._state = state              # type: ignore[attr-defined]
    STATE.learner._save_state()               # type: ignore[attr-defined]
    return {"ok": True, "drift_flagged": False}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    STATE.websocket_clients.add(ws)
    try:
        # send initial hello
        await ws.send_json({
            "event": "connected",
            "model_version": STATE.learner.state().model_version if STATE.learner else 0,
            "buffer_size": len(get_global_buffer()),
        })
        # keep the connection open
        while True:
            await asyncio.sleep(30)
            await ws.send_json({"event": "heartbeat", "timestamp": time.time()})
    except WebSocketDisconnect:
        pass
    finally:
        STATE.websocket_clients.discard(ws)
