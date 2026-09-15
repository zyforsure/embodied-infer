"""Latency/throughput benchmark for an embodied-infer server.

Usage::

    python scripts/bench_orin.py <host> <port> [warmup] [sequential] [workers] [per_worker]

Reports mean/p50/p95/p99/min/max latency plus throughput for both a sequential
loop and a concurrent burst.  p99 is the number that matters for a real-time
control loop: the loop is bound by the tail, not the mean.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from embodied_infer_deploy.client import InferenceClient

HOST, PORT = sys.argv[1], int(sys.argv[2])
WARMUP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
SEQ = int(sys.argv[4]) if len(sys.argv) > 4 else 20
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
PER_WORKER = int(sys.argv[6]) if len(sys.argv) > 6 else 10

IMAGE = np.zeros((224, 224, 3), dtype=np.uint8)
IMAGES = [("head", IMAGE, 0), ("left_wrist", IMAGE, 0), ("right_wrist", IMAGE, 0)]
STATE = np.zeros(14, dtype=np.float32)
PROMPT = "pick up the block and place it"


def one_request(client, step):
    start = time.perf_counter()
    response = client.infer(
        control_step=step, timestamp_ns=time.time_ns(), instruction=PROMPT,
        images=IMAGES, state=STATE, timeout=60.0,
    )
    elapsed = (time.perf_counter() - start) * 1000.0
    actions = np.asarray(response["actions"])
    assert np.isfinite(actions).all(), "non-finite actions"
    return elapsed


def stats(latencies):
    arr = np.sort(np.asarray(latencies))
    return {
        "n": int(arr.size),
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
        "stdev_ms": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 2),
    }


def main():
    client = InferenceClient(HOST, PORT, request_timeout=60.0)
    try:
        meta = client.metadata
        for i in range(WARMUP):
            one_request(client, i)
        seq_lat = [one_request(client, i) for i in range(SEQ)]
        seq = stats(seq_lat)
        seq["throughput_rps"] = round(1000.0 / seq["mean_ms"], 2)

        def worker(w):
            c = InferenceClient(HOST, PORT, request_timeout=60.0)
            try:
                return [one_request(c, i) for i in range(PER_WORKER)]
            finally:
                c.close()

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(worker, range(WORKERS)))
        wall = time.perf_counter() - start
        conc_lat = [x for r in results for x in r]
        conc = stats(conc_lat)
        conc["workers"] = WORKERS
        conc["wall_s"] = round(wall, 2)
        conc["throughput_rps"] = round(len(conc_lat) / wall, 2)
        print(json.dumps({
            "backend": meta.get("backend"),
            "model": meta.get("model"),
            "execution_host": meta.get("execution_host"),
            "action_horizon": meta.get("action_horizon"),
            "vision_batching_mode": (meta.get("vision_batching") or {}).get("mode"),
            "sequential": seq,
            "concurrent": conc,
        }, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
