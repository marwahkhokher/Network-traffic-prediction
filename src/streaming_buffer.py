"""
streaming_buffer.py
-------------------
A thread-safe rolling buffer that holds the most recent traffic-matrix
vectors as they arrive in real time. Used by:
  - streaming_app.py    (writer): appends each ingested row.
  - online_learner.py   (reader): reads tail windows for partial_fit.
  - retrain_worker.py   (reader): reads larger windows for full retrain.

Design choices:
  - In-memory deque with maxlen → bounded memory, O(1) append.
  - Read-write lock implemented with a single RLock (the workloads here
    are append-heavy and read-light, so contention is negligible).
  - Periodic flush to disk (pickle) for crash recovery. Atomic via
    os.replace() so a crash mid-write doesn't corrupt the snapshot.
  - Subscriber pattern: components register a callback that fires
    whenever the buffer grows past certain milestones.
"""
from __future__ import annotations

import os
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np


@dataclass
class BufferStats:
    total_ingested: int = 0
    in_buffer: int = 0
    last_ingest_ts: Optional[float] = None
    last_flush_ts: Optional[float] = None
    n_features: Optional[int] = None


class StreamingBuffer:
    """A bounded rolling buffer of (timestamp, vector) tuples."""

    def __init__(
        self,
        capacity: int = 10_000,
        flush_path: Optional[str | Path] = None,
        flush_every_n: int = 25,
    ):
        self._buffer: deque = deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._capacity = capacity
        self._flush_path = Path(flush_path) if flush_path else None
        self._flush_every_n = flush_every_n
        self._subscribers: List[Callable[[int], None]] = []
        self._stats = BufferStats()

        if self._flush_path and self._flush_path.exists():
            self._restore_from_disk()

    # ---------------------------------------------------------------- append
    def append(self, vector: np.ndarray, timestamp: Optional[float] = None) -> int:
        """Append one vector; return the new buffer length."""
        if vector.ndim != 1:
            raise ValueError(f"expected 1-D vector, got shape {vector.shape}")
        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            self._buffer.append((ts, vector.astype(np.float32)))
            self._stats.total_ingested += 1
            self._stats.in_buffer = len(self._buffer)
            self._stats.last_ingest_ts = ts
            if self._stats.n_features is None:
                self._stats.n_features = vector.shape[0]
            should_flush = (
                self._flush_path
                and self._stats.total_ingested % self._flush_every_n == 0
            )
            new_len = len(self._buffer)
            subs = list(self._subscribers)

        if should_flush:
            self._flush_to_disk()
        for cb in subs:
            try:
                cb(new_len)
            except Exception as e:  # subscriber failure must not crash ingestion
                print(f"[streaming_buffer] subscriber error: {e}")

        return new_len

    def append_batch(
        self, vectors: np.ndarray, timestamps: Optional[List[float]] = None
    ) -> int:
        if vectors.ndim != 2:
            raise ValueError(f"expected (n, F) batch, got shape {vectors.shape}")
        n = vectors.shape[0]
        if timestamps is None:
            now = time.time()
            timestamps = [now + i * 1e-6 for i in range(n)]
        for v, ts in zip(vectors, timestamps):
            self.append(v, ts)
        return len(self._buffer)

    # ---------------------------------------------------------------- reads
    def get_last_n(self, n: int) -> np.ndarray:
        """Return the most recent n vectors as a (n, F) array.

        Raises ValueError if fewer than n are available.
        """
        with self._lock:
            if n > len(self._buffer):
                raise ValueError(
                    f"requested {n} but only {len(self._buffer)} in buffer"
                )
            rows = list(self._buffer)[-n:]
        return np.stack([r[1] for r in rows])

    def get_tail_window(self, window: int) -> Optional[np.ndarray]:
        """Same as get_last_n but returns None if not yet enough samples."""
        with self._lock:
            if len(self._buffer) < window:
                return None
            rows = list(self._buffer)[-window:]
        return np.stack([r[1] for r in rows])

    def get_recent(self, max_samples: int) -> np.ndarray:
        """Return up to `max_samples` most recent vectors (whatever is available)."""
        with self._lock:
            rows = list(self._buffer)[-max_samples:]
        if not rows:
            return np.empty((0, self._stats.n_features or 0), dtype=np.float32)
        return np.stack([r[1] for r in rows])

    def get_timestamps(self, max_samples: int) -> List[float]:
        with self._lock:
            return [ts for ts, _ in list(self._buffer)[-max_samples:]]

    # ---------------------------------------------------------------- stats
    def stats(self) -> BufferStats:
        with self._lock:
            return BufferStats(
                total_ingested=self._stats.total_ingested,
                in_buffer=len(self._buffer),
                last_ingest_ts=self._stats.last_ingest_ts,
                last_flush_ts=self._stats.last_flush_ts,
                n_features=self._stats.n_features,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    # ---------------------------------------------------------------- subscribers
    def subscribe(self, callback: Callable[[int], None]) -> None:
        """Register callback fired on each append. Callback receives buffer length."""
        with self._lock:
            self._subscribers.append(callback)

    # ---------------------------------------------------------------- persistence
    def _flush_to_disk(self) -> None:
        if not self._flush_path:
            return
        self._flush_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._flush_path.with_suffix(self._flush_path.suffix + ".tmp")
        with self._lock:
            snapshot = list(self._buffer)
        try:
            with open(tmp, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self._flush_path)
            with self._lock:
                self._stats.last_flush_ts = time.time()
        except Exception as e:
            print(f"[streaming_buffer] flush failed: {e}")
            if tmp.exists():
                try: tmp.unlink()
                except Exception: pass

    def _restore_from_disk(self) -> None:
        try:
            with open(self._flush_path, "rb") as f:
                snapshot = pickle.load(f)
            with self._lock:
                self._buffer.clear()
                for ts, vec in snapshot[-self._capacity:]:
                    self._buffer.append((ts, vec))
                self._stats.total_ingested = len(self._buffer)
                self._stats.in_buffer = len(self._buffer)
                if self._buffer:
                    self._stats.n_features = self._buffer[0][1].shape[0]
                    self._stats.last_ingest_ts = self._buffer[-1][0]
            print(f"[streaming_buffer] restored {len(self._buffer)} rows from {self._flush_path}")
        except Exception as e:
            print(f"[streaming_buffer] could not restore from {self._flush_path}: {e}")

    def force_flush(self) -> None:
        self._flush_to_disk()


# ---------------------------------------------------------------- singleton
_GLOBAL_BUFFER: Optional[StreamingBuffer] = None


def get_global_buffer(
    capacity: int = 10_000,
    flush_path: Optional[str | Path] = None,
) -> StreamingBuffer:
    """Lazy singleton so the FastAPI service and the retrain worker
    can share the same buffer object inside one process. Across
    processes, both rely on the disk snapshot for synchronisation."""
    global _GLOBAL_BUFFER
    if _GLOBAL_BUFFER is None:
        _GLOBAL_BUFFER = StreamingBuffer(capacity=capacity, flush_path=flush_path)
    return _GLOBAL_BUFFER
