#!/usr/bin/env python3
"""Minimal TCP inference server for NVIDIA GR00T-N1-2B (runs on the 4090 gateway).

Wire protocol: 4-byte big-endian length prefix + msgpack payload (same as
openvla_server.py, so a single thin client backend serves both).

Request
    {"images": {name: jpeg_bytes}, "instruction": str,
     "state": [f32, ...], "horizon": int}

Response
    {"actions": [[f32, ...], ...], "horizon": int, "action_dim": int,
     "timing": {...}, "error": str | None}

The checkpoint is served through Isaac-GR00T (n1-release) Gr00tPolicy with the
gr1_arms_only embodiment.  Run with:
    PYTHONPATH=/shared-data/Isaac-GR00T-n1 python groot_server.py ...
"""

from __future__ import annotations

import argparse
import io
import json
import os
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

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

# gr1_arms_only layout (order must match the data-config keys).
STATE_KEYS = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
ACTION_KEYS = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
VIDEO_KEY = "video.ego_view"
LANG_KEY = "annotation.human.action.task_description"


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


def _dims_from_metadata(model_path: str, embodiment: str):
    """Read per-key state/action dims from the checkpoint metadata; fall back to
    the standard GR1 arms+hands layout (7,7,6,6) if unavailable."""
    try:
        meta = json.load(open(os.path.join(model_path, "experiment_cfg", "metadata.json")))
        stats = meta[embodiment]["statistics"]

        def dims(section, keys):
            out = []
            for key in keys:
                short = key.split(".", 1)[1]
                out.append(int(len(stats[section][short]["max"])))
            return out

        return dims("state", STATE_KEYS), dims("action", ACTION_KEYS)
    except Exception:
        return [7, 7, 6, 6], [7, 7, 6, 6]


class GrootWorker:
    def __init__(self, model_path, embodiment_tag, data_config, denoising_steps, device):
        cfg = DATA_CONFIG_MAP[data_config]
        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            modality_config=cfg.modality_config(),
            modality_transform=cfg.transform(),
            denoising_steps=denoising_steps,
            device=device,
        )
        self.state_dims, self.action_dims = _dims_from_metadata(model_path, embodiment_tag)
        self.total_state = sum(self.state_dims)
        self.lock = threading.Lock()

    def _split_state(self, flat):
        flat = np.asarray(flat, dtype=np.float32).reshape(-1)
        padded = np.zeros(self.total_state, dtype=np.float32)
        n = min(flat.shape[0], self.total_state)
        padded[:n] = flat[:n]
        parts, i = [], 0
        for d in self.state_dims:
            parts.append(padded[i:i + d])
            i += d
        return parts

    def _action_part(self, action_dict, full_key):
        if full_key in action_dict:
            arr = action_dict[full_key]
        else:
            short = full_key.split(".", 1)[1]
            cand = [v for k, v in action_dict.items() if k.endswith(short)]
            if not cand:
                raise KeyError(f"{full_key} not in action dict {sorted(action_dict)}")
            arr = cand[0]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    def infer(self, request: dict) -> dict:
        timing: dict[str, float] = {}
        t_start = time.perf_counter()
        t0 = time.perf_counter()

        images = request.get("images", {})
        if not images:
            raise ValueError("request carried no images")
        raw = images[sorted(images)[0]]
        if isinstance(raw, (bytes, bytearray)):
            img = np.asarray(Image.open(io.BytesIO(bytes(raw))).convert("RGB"), dtype=np.uint8)
        else:
            img = np.asarray(raw, dtype=np.uint8)
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.moveaxis(img, 0, -1)
        instruction = request.get("instruction", "") or ""
        state_parts = self._split_state(request.get("state", []))

        obs = {VIDEO_KEY: img[None], LANG_KEY: instruction}
        for key, part in zip(STATE_KEYS, state_parts):
            obs[key] = part[None]
        timing["prepare_s"] = time.perf_counter() - t0

        with self.lock, torch.inference_mode():
            t1 = time.perf_counter()
            action_dict = self.policy.get_action(obs)
            timing["generate_s"] = time.perf_counter() - t1

        parts = [self._action_part(action_dict, k) for k in ACTION_KEYS]
        actions = np.concatenate(parts, axis=-1).astype(np.float32)
        timing["total_s"] = time.perf_counter() - t_start
        return {
            "actions": actions.tolist(),
            "horizon": int(actions.shape[0]),
            "action_dim": int(actions.shape[1]),
            "timing": timing,
            "error": None,
        }


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        worker: GrootWorker = self.server.worker  # type: ignore[attr-defined]
        try:
            while True:
                request = recv_msg(self.request)
                if request is None:
                    return
                try:
                    response = worker.infer(request)
                except Exception as exc:
                    response = {
                        "actions": None, "horizon": 0, "action_dim": 0,
                        "timing": {}, "error": f"{type(exc).__name__}: {exc}",
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
    parser.add_argument("--model", default="/shared-data/models/gr00t-n1-2b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8022)
    parser.add_argument("--embodiment-tag", default="gr1")
    parser.add_argument("--data-config", default="gr1_arms_only")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[groot] loading {args.model} ({args.data_config}/{args.embodiment_tag}) ...", flush=True)
    t0 = time.perf_counter()
    worker = GrootWorker(args.model, args.embodiment_tag, args.data_config,
                         args.denoising_steps, args.device)
    print(f"[groot] loaded in {time.perf_counter() - t0:.1f}s; "
          f"state_dims={worker.state_dims} action_dims={worker.action_dims}", flush=True)
    server = Server((args.host, args.port), worker)
    print(f"[groot] listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
