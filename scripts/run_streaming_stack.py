"""
run_streaming_stack.py
----------------------
One-command launcher that starts both halves of the streaming stack:

  1. The FastAPI streaming service  (api/streaming_app.py via uvicorn)
  2. The background retrain worker  (src/retrain_worker.py)

Both run as subprocesses so that:
  - if the worker crashes, the serving keeps going
  - SIGINT (Ctrl+C) cleanly stops both

Usage from project root:
  python scripts/run_streaming_stack.py
  python scripts/run_streaming_stack.py --port 8765 --check-interval 30 --min-samples 200

Or, equivalently, launch the two manually in separate terminals:
  uvicorn api.streaming_app:app --port 8765
  python src/retrain_worker.py --min-samples 500 --check-interval 30
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765,
                    help="port for the FastAPI streaming service")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model", default="results/lstm_streaming.keras")
    ap.add_argument("--state", default="results/lstm_streaming.state.json")
    ap.add_argument("--buffer", default="results/streaming_buffer.pkl")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--min-samples", type=int, default=500)
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--check-interval", type=float, default=30.0)
    ap.add_argument("--no-worker", action="store_true",
                    help="skip launching the retrain worker (service only)")
    ap.add_argument("--no-service", action="store_true",
                    help="skip launching the FastAPI service (worker only)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["STREAMING_MODEL"] = args.model
    env["STREAMING_STATE"] = args.state
    env["STREAMING_BUFFER"] = args.buffer
    env["STREAMING_WINDOW"] = str(args.window)

    procs: list[tuple[str, subprocess.Popen]] = []

    if not args.no_service:
        service_cmd = [
            sys.executable, "-m", "uvicorn",
            "api.streaming_app:app",
            "--host", args.host, "--port", str(args.port),
        ]
        print(f"[stack] launching streaming service: {' '.join(service_cmd)}")
        procs.append(("service", subprocess.Popen(service_cmd, cwd=root, env=env)))
        # Give the service a moment to come up before the worker tries to touch shared files
        time.sleep(2.0)

    if not args.no_worker:
        worker_cmd = [
            sys.executable, "src/retrain_worker.py",
            "--buffer-snapshot", args.buffer,
            "--model", args.model,
            "--state", args.state,
            "--window", str(args.window),
            "--min-samples", str(args.min_samples),
            "--max-samples", str(args.max_samples),
            "--epochs", str(args.epochs),
            "--check-interval", str(args.check_interval),
        ]
        print(f"[stack] launching retrain worker:  {' '.join(worker_cmd)}")
        procs.append(("worker", subprocess.Popen(worker_cmd, cwd=root, env=env)))

    if not procs:
        print("[stack] nothing to launch (both --no-service and --no-worker set).")
        return 1

    print(f"[stack] streaming stack up — service on port {args.port}.  Ctrl+C to stop.")

    stopping = {"flag": False}

    def _handle(_signum, _frame):
        if stopping["flag"]:
            return
        stopping["flag"] = True
        print("\n[stack] stopping subprocesses…")
        for name, p in procs:
            try:
                p.send_signal(signal.SIGINT)
            except Exception as e:
                print(f"[stack]   {name}: send_signal failed: {e}")

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    # Wait for any subprocess to exit; if one dies, take the rest down too
    while not stopping["flag"]:
        for name, p in procs:
            rc = p.poll()
            if rc is not None:
                print(f"[stack] subprocess '{name}' exited with code {rc}")
                stopping["flag"] = True
                break
        time.sleep(1.0)

    # Final cleanup
    deadline = time.time() + 10
    for name, p in procs:
        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            pass
    for name, p in procs:
        remaining = max(0.5, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[stack]   {name}: forcing kill")
            try: p.kill()
            except Exception: pass

    print("[stack] all subprocesses stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
