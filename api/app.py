"""
app.py — FastAPI prediction service

Serves the trained LSTM as an HTTP endpoint so an operator (or the
AnyLogic simulation running as a real-time loop) can push in the
most recent W traffic vectors and receive the predicted next vector.

Endpoints:
  GET  /health        -> liveness + model info
  POST /predict       -> body: {"window": [[...], [...], ...]}  (W rows, F cols, scaled 0..1)
                         returns: {"predicted_vector": [...], "congestion_flags": [...]}
  POST /predict_raw   -> same as /predict but input/output in raw byte rates; the
                         service applies the /max scaler recorded at training time.

Run:
  uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf


MODEL_PATH = os.environ.get("LSTM_MODEL_PATH", "results/lstm_model.keras")
SCALE_PATH = os.environ.get("TRAFFIC_SCALE_PATH", "results/scale.txt")

app = FastAPI(
    title="Network Traffic Matrix Prediction API",
    description=(
        "LSTM-based traffic-matrix forecaster built on Azzouni & Pujolle (2017). "
        "Predicts the next traffic vector from a sliding window of W past vectors."
    ),
    version="1.0.0",
)


# ------------------------------------------------------------------
# Model loading (lazy)
# ------------------------------------------------------------------
_state: dict = {"model": None, "scale": 1.0, "window": None, "n_features": None}


def _load_model() -> tf.keras.Model:
    if _state["model"] is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python src/train.py ...` first."
            )
        m = tf.keras.models.load_model(MODEL_PATH)
        _state["model"] = m
        _state["window"] = m.input_shape[1]
        _state["n_features"] = m.input_shape[2]

        if Path(SCALE_PATH).exists():
            _state["scale"] = float(Path(SCALE_PATH).read_text().strip())
    return _state["model"]


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------
class PredictRequest(BaseModel):
    window: List[List[float]] = Field(
        ...,
        description="W rows x F columns. Rows = past timeslots (oldest first). "
                    "For /predict values must be pre-scaled in [0, 1]. "
                    "For /predict_raw values are raw bytes/sec or similar.",
    )


class PredictResponse(BaseModel):
    predicted_vector: List[float]
    predicted_matrix_shape: List[int]
    congestion_flags: List[dict]
    scale_used: Optional[float] = None


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        m = _load_model()
        return {
            "status": "ok",
            "model_path": MODEL_PATH,
            "window": _state["window"],
            "n_features": _state["n_features"],
            "n_nodes": int(round(np.sqrt(_state["n_features"] or 1))),
            "scale": _state["scale"],
        }
    except RuntimeError as e:
        return {"status": "degraded", "error": str(e)}


def _predict_core(window_arr: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    model = _load_model()
    W, F = _state["window"], _state["n_features"]

    if window_arr.shape != (W, F):
        raise HTTPException(
            status_code=400,
            detail=f"window shape {window_arr.shape} != expected ({W}, {F})",
        )

    pred_scaled = model.predict(window_arr[None, ...], verbose=0)[0]
    # congestion heuristic: flag OD pairs where the predicted value is in the
    # top 5% of values in the window (rough "spike ahead" signal).
    thresh = float(np.quantile(window_arr, 0.95))
    flags_idx = np.where(pred_scaled > thresh)[0]
    n_nodes = int(round(np.sqrt(F)))
    flags = [
        {
            "origin": int(k // n_nodes),
            "destination": int(k % n_nodes),
            "predicted": float(pred_scaled[k]),
            "threshold": thresh,
        }
        for k in flags_idx
    ]
    return pred_scaled, flags


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    arr = np.asarray(req.window, dtype=np.float32)
    pred_scaled, flags = _predict_core(arr)
    n_nodes = int(round(np.sqrt(pred_scaled.size)))
    return PredictResponse(
        predicted_vector=pred_scaled.tolist(),
        predicted_matrix_shape=[n_nodes, n_nodes],
        congestion_flags=flags,
    )


@app.post("/predict_raw", response_model=PredictResponse)
def predict_raw(req: PredictRequest):
    """Same as /predict but applies the training-time /max scaler."""
    _load_model()
    scale = _state["scale"] or 1.0
    arr = np.asarray(req.window, dtype=np.float32) / scale
    pred_scaled, flags = _predict_core(arr)
    pred_raw = pred_scaled * scale
    for f in flags:
        f["predicted"] = f["predicted"] * scale
        f["threshold"] = f["threshold"] * scale
    n_nodes = int(round(np.sqrt(pred_raw.size)))
    return PredictResponse(
        predicted_vector=pred_raw.tolist(),
        predicted_matrix_shape=[n_nodes, n_nodes],
        congestion_flags=flags,
        scale_used=scale,
    )
