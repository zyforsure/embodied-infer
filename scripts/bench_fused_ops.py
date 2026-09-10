# -*- coding: utf-8 -*-
"""Microbenchmark for the fused operator plugin.

Times the NumPy reference path against the torch-native path on operator
shapes taken from the split TurboVLA stack:

* vision ViT block: batch 3 (three cameras), 201 tokens, hidden 1024,
  16 heads x 64, MLP 1024 -> 4096 -> 1024;
* text BERT block: 24 tokens, hidden 768, 12 heads x 64,
  MLP 768 -> 3072 -> 768.

Usage:
    PYTHONPATH=python python scripts/bench_fused_ops.py [--iters 50]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from embodied_infer_deploy.plugins import FusedOpsConfig, FusedOpsPlugin  # noqa: E402


def bench(fn, iters: int) -> float:
    fn()  # warmup
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iters


def bench_block(name: str, batch: int, tokens: int, hidden: int, heads: int,
                mlp_hidden: int, iters: int) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(batch, tokens, hidden)).astype(np.float32)
    head_dim = hidden // heads
    weights = tuple(rng.normal(size=(hidden, hidden)).astype(np.float32)
                    for _ in range(3))
    w1 = rng.normal(size=(mlp_hidden, hidden)).astype(np.float32)
    b1 = rng.normal(size=(mlp_hidden,)).astype(np.float32)
    w2 = rng.normal(size=(hidden, mlp_hidden)).astype(np.float32)
    b2 = rng.normal(size=(hidden,)).astype(np.float32)
    ln_w = rng.normal(size=(hidden,)).astype(np.float32)
    ln_b = rng.normal(size=(hidden,)).astype(np.float32)
    q = rng.normal(size=(batch, heads, tokens, head_dim)).astype(np.float32)
    k = rng.normal(size=(batch, heads, tokens, head_dim)).astype(np.float32)
    v = rng.normal(size=(batch, heads, tokens, head_dim)).astype(np.float32)

    numpy_plugin = FusedOpsPlugin(config=FusedOpsConfig(enabled=True, prefer="numpy"))
    native_plugin = FusedOpsPlugin(config=FusedOpsConfig(enabled=True))
    try:
        backend = native_plugin.metadata()["backend"]
        rows = []
        ops = [
            ("qkv", lambda p: p.project_qkv(x, weights)),
            ("attention", lambda p: p.attend(q, k, v)),
            ("gelu_mlp", lambda p: p.gelu_mlp(x, w1, b1, w2, b2)),
            ("layer_norm", lambda p: p.layer_norm(x, ln_w, ln_b, residual=x)),
        ]
        for op_name, fn in ops:
            numpy_ms = bench(lambda: fn(numpy_plugin), iters)
            native_ms = bench(lambda: fn(native_plugin), iters)
            rows.append((op_name, numpy_ms, native_ms))
        print(f"[{name}] batch={batch} tokens={tokens} hidden={hidden} "
              f"heads={heads} mlp={mlp_hidden} native_backend={backend}")
        for op_name, numpy_ms, native_ms in rows:
            speedup = numpy_ms / native_ms if native_ms > 0 else float("inf")
            print(f"  {op_name:11s} numpy {numpy_ms:8.3f} ms   "
                  f"native {native_ms:8.3f} ms   speedup {speedup:5.2f}x")
    finally:
        numpy_plugin.close()
        native_plugin.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    bench_block("turbovla-vision-vit", batch=3, tokens=201, hidden=1024,
                heads=16, mlp_hidden=4096, iters=args.iters)
    bench_block("turbovla-text-bert", batch=1, tokens=24, hidden=768,
                heads=12, mlp_hidden=3072, iters=args.iters)


if __name__ == "__main__":
    main()