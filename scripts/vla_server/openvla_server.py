#!/usr/bin/env python3
"""Minimal TCP inference server for OpenVLA-7B (runs on the 4090 gateway).

Wire protocol: 4-byte big-endian length prefix + msgpack payload.

Request
    {"images": {name: jpeg_bytes}, "instruction": str,
     "state": [f32, ...], "horizon": int}

Response
    {"actions": [[f32, ...], ...],   # (horizon, action_dim)
     "timing": {...}, "error": str | None}

`horizon == 1` uses the official `predict_action` path (semantically exact).
`horizon > 1` asks the decoder for `horizon * action_dim` action tokens in a
single `generate` call, so the vision tower and the prompt prefill run once and
only the shallow LM head keeps stepping.  That is the "multi-token prediction"
shape: one encode, many steps.  It is NOT a trained MTP head - the base
checkpoint was only supervised on a single 7-token action - so steps 2..H are
extrapolations, not ground-truth-quality predictions.
"""

from __future__ import annotations

import argparse
import io
import socket
import socketserver
import struct
import threading
import time
import traceback

import msgpack
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def recv_msg(conn: socket.socket):
    header = conn.recv(4)
    if len(header) < 4:
        return None
    (size,) = struct.unpack(">I", header)
    buf = bytearray()
    while len(buf) < size:
        chunk = conn.recv(min(65536, size - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return msgpack.unpackb(bytes(buf), raw=False)


def send_msg(conn: socket.socket, obj: dict) -> None:
    payload = msgpack.packb(obj, use_bin_type=True)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def _patch_transformers_compat() -> None:
    """transformers>=4.52 dropped the legacy `_supports_*` class attributes that
    the 4.40-era OpenVLA checkpoint reads during init; restore them so loading
    does not crash with AttributeError on newer transformers."""
    from transformers.modeling_utils import PreTrainedModel

    for attr in (
        "_supports_sdpa",
        "_supports_flash_attn_2",
        "_supports_cache_class",
        "_supports_static_cache",
        "_supports_quantized_cache",
        "_supports_flex_attn",
    ):
        if not hasattr(PreTrainedModel, attr):
            setattr(PreTrainedModel, attr, False)


class OpenVLAWorker:
    def __init__(self, model_path: str, unnorm_key: str | None, action_dim: int = 7):
        self.model_path = model_path
        self.action_dim = action_dim
        _patch_transformers_compat()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
        ).cuda().eval()

        stats = self._all_stats()
        self.available_keys = sorted(stats.keys())
        if unnorm_key and unnorm_key in stats:
            self.unnorm_key = unnorm_key
        else:
            # fall back to the bridge dataset the base checkpoint ships with
            self.unnorm_key = self._pick_default(stats)
        self.norm_stats = stats
        self.lock = threading.Lock()

    def _all_stats(self) -> dict:
        for getter in (
            lambda: self.model.norm_stats,
            lambda: self.model.config.norm_stats,
        ):
            try:
                value = getter()
                if isinstance(value, dict) and value:
                    return value
            except Exception:
                continue
        return {}

    @staticmethod
    def _pick_default(stats: dict) -> str:
        for key in sorted(stats):
            if "bridge" in key or "buds" in key:
                return key
        return sorted(stats)[0] if stats else None

    def _unnormalize(self, normalized: np.ndarray) -> np.ndarray:
        """normalized in [-1, 1] -> dataset action units."""
        entry = self.norm_stats.get(self.unnorm_key)
        if entry is None:
            return normalized
        # openVLA nests the stats under an "action" key; tolerate both shapes.
        stats = entry.get("action", entry) if isinstance(entry, dict) else entry
        high = np.asarray(stats["q99"], dtype=np.float64)
        low = np.asarray(stats["q01"], dtype=np.float64)
        mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
        clipped = np.clip(normalized, -1.0, 1.0)
        actions = 0.5 * (clipped + 1.0) * (high - low) + low
        # dims outside the mask have no meaningful scale: keep normalized value
        if mask.shape == actions.shape[-1:]:
            actions = np.where(mask, actions, clipped)
        return actions.astype(np.float32)

    def _prepare(self, request: dict):
        """Returns (prepared_inputs, image, elapsed_seconds)."""
        t0 = time.perf_counter()
        images = []
        for name in sorted(request.get("images", {})):
            raw = request["images"][name]
            if isinstance(raw, (bytes, bytearray)):
                images.append(Image.open(io.BytesIO(bytes(raw))).convert("RGB"))
            else:
                arr = np.asarray(raw, dtype=np.uint8)
                images.append(Image.fromarray(arr))
        if not images:
            raise ValueError("request carried no images")
        instruction = request.get("instruction", "") or ""
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        # The base OpenVLA checkpoint is single-view; extra cameras are ignored.
        inputs = self.processor(prompt, images[0])
        # Only floating-point tensors (pixel_values) go to bf16; token ids and
        # attention masks must stay integer or the embedding lookup fails.
        prepared = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                v = v.to("cuda")
                if v.dtype.is_floating_point:
                    v = v.to(dtype=torch.bfloat16)
            prepared[k] = v
        return prepared, time.perf_counter() - t0

    def infer(self, request: dict) -> dict:
        horizon = max(1, int(request.get("horizon", 1)))
        timing: dict[str, float] = {}
        t_start = time.perf_counter()
        inputs, t_prepare = self._prepare(request)
        timing["prepare_s"] = t_prepare

        with self.lock, torch.inference_mode():
            t1 = time.perf_counter()
            if horizon == 1:
                action = self.model.predict_action(
                    **inputs, unnorm_key=self.unnorm_key, do_sample=False
                )
                actions = np.asarray(action, dtype=np.float32).reshape(1, self.action_dim)
            else:
                # MTP-style: one encode/prefill, decode horizon*action_dim tokens,
                # then discretize->bin centers exactly like predict_action.
                n_tokens = self.action_dim * horizon
                if not torch.all(inputs["input_ids"][:, -1] == 29871):
                    inputs = dict(inputs)
                    pad = torch.full(
                        (inputs["input_ids"].shape[0], 1), 29871,
                        dtype=torch.long, device=inputs["input_ids"].device,
                    )
                    inputs["input_ids"] = torch.cat((inputs["input_ids"], pad), dim=1)
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=n_tokens,
                    min_new_tokens=n_tokens,
                    do_sample=False,
                )
                token_ids = (
                    generated[0, -n_tokens:].detach().cpu().numpy()
                    .reshape(horizon, self.action_dim)
                )
                vocab_size = getattr(self.model, "vocab_size", None) or int(
                    self.model.config.vocab_size
                )
                bin_centers = getattr(self.model, "bin_centers", None)
                if bin_centers is None:
                    n_bins = int(getattr(self.model.config, "n_action_bins", 256))
                    _bins = np.linspace(-1, 1, n_bins)
                    bin_centers = 0.5 * (_bins[:-1] + _bins[1:])
                bin_centers = np.asarray(bin_centers, dtype=np.float32)
                discretized = vocab_size - token_ids
                discretized = np.clip(discretized - 1, 0, bin_centers.shape[0] - 1)
                normalized = bin_centers[discretized].astype(np.float32)
                actions = self._unnormalize(normalized)
            timing["generate_s"] = time.perf_counter() - t1

        timing["total_s"] = time.perf_counter() - t_start
        return {
            "actions": actions.astype(np.float32).tolist(),
            "horizon": int(actions.shape[0]),
            "action_dim": int(actions.shape[1]),
            "unnorm_key": self.unnorm_key,
            "timing": timing,
            "error": None,
        }


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        worker: OpenVLAWorker = self.server.worker  # type: ignore[attr-defined]
        try:
            while True:
                request = recv_msg(self.request)
                if request is None:
                    return
                try:
                    response = worker.infer(request)
                except Exception as exc:  # keep the server alive
                    response = {
                        "actions": None,
                        "horizon": 0,
                        "action_dim": 0,
                        "unnorm_key": getattr(worker, "unnorm_key", None),
                        "timing": {},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    traceback.print_exc()
                send_msg(self.request, response)
        except (ConnectionResetError, BrokenPipeError):
            return


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr, worker):
        self.worker = worker
        super().__init__(addr, Handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/shared-data/models/openvla-7b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--action-dim", type=int, default=7)
    args = parser.parse_args()

    print(f"[openvla] loading {args.model} ...", flush=True)
    t0 = time.perf_counter()
    worker = OpenVLAWorker(args.model, args.unnorm_key, args.action_dim)
    print(
        f"[openvla] loaded in {time.perf_counter() - t0:.1f}s; "
        f"unnorm_key={worker.unnorm_key}; available={worker.available_keys}",
        flush=True,
    )
    server = Server((args.host, args.port), worker)
    print(f"[openvla] listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
