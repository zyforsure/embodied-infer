"""Client for the Pi05 float TCP server used on the 4090.

The server executes the real Pi05 checkpoint; this gateway can run on Orin and
still feed its result through the normal embodied-infer/RoboTwin pipeline.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from typing import Any

import numpy as np

from ...core import BackendResult, ModelSpec

MAGIC = b"P05R"


def _jpeg(image: Any) -> bytes:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Pi05 image must be rank 3, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Pi05 image must be HWC RGB, got {array.shape}")
    array = np.ascontiguousarray(array.astype(np.uint8, copy=False))
    try:
        from PIL import Image
        import io
        output = io.BytesIO()
        Image.fromarray(array, mode="RGB").save(output, format="JPEG", quality=95)
        return output.getvalue()
    except ImportError:
        import cv2
        ok, encoded = cv2.imencode(".jpg", array[..., ::-1])
        if not ok:
            raise RuntimeError("failed to encode Pi05 image as JPEG")
        return encoded.tobytes()


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        value = sock.recv(size - len(chunks))
        if not value:
            raise ConnectionError("Pi05 TCP server closed the connection")
        chunks.extend(value)
    return bytes(chunks)


class Pi05TcpBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.host = str(config.get("host", "192.168.10.142"))
        self.port = int(config.get("port", 8012))
        self.timeout = float(config.get("timeout", 120.0))
        self._lock = threading.RLock()
        self._sock: socket.socket | None = None
        self._seq = 0
        self._spec = ModelSpec(
            name=str(config.get("model_name", "Pi05-4090")), backend="pi05-tcp",
            raw_state_dim=18, model_state_dim=14, raw_action_dim=18, model_action_dim=14,
            action_horizon=int(config.get("action_horizon", 16)),
            camera_order=("head", "left_wrist", "right_wrist"), action_representation="absolute",
            control_period_ns=int(config.get("control_period_ns", 100_000_000)),
            extras={"protocol": "pi05-float-tcp", "remote_model": "pi05", "execution_host": self.host},
        )

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def metadata(self) -> dict[str, Any]:
        return self.spec.metadata()

    def _connect(self) -> socket.socket:
        if self._sock is None:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self._sock.settimeout(self.timeout)
        return self._sock

    def _request(self, metadata: dict[str, Any], blobs: list[bytes]) -> tuple[dict[str, Any], list[bytes]]:
        payload = json.dumps({**metadata, "blob_sizes": [len(x) for x in blobs]}, separators=(",", ":")).encode()
        packet = MAGIC + struct.pack("!I", len(payload)) + payload + b"".join(blobs)
        sock = self._connect()
        try:
            sock.sendall(packet)
            if _recv_exact(sock, 4) != MAGIC:
                raise RuntimeError("invalid Pi05 TCP response magic")
            size = struct.unpack("!I", _recv_exact(sock, 4))[0]
            if size <= 0 or size > 1_000_000:
                raise RuntimeError(f"invalid Pi05 TCP response metadata size {size}")
            response = json.loads(_recv_exact(sock, size))
            blobs_out = [_recv_exact(sock, int(n)) for n in response.get("blob_sizes", [])]
            return response, blobs_out
        except Exception:
            self.close()
            raise

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        state = np.zeros(18, dtype=np.float32)
        state[:14] = np.asarray(request["state"], dtype=np.float32)
        blobs = [_jpeg(request["images"][name]) for name in self.spec.camera_order]
        with self._lock:
            seq = self._seq
            self._seq += 1
            started = time.perf_counter()
            response, outputs = self._request({"type": "infer", "seq": seq, "state": state.tolist(),
                                               "prompt": str(request.get("instruction", "")), "seed": int(request.get("control_step", 0))}, blobs)
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("error", "Pi05 TCP inference failed")))
        if len(outputs) != 1:
            raise RuntimeError(f"Pi05 TCP response expected one action blob, got {len(outputs)}")
        shape = tuple(int(x) for x in response.get("shape", [50, 18]))
        actions = np.frombuffer(outputs[0], dtype=np.float32).reshape(shape)
        actions = actions[: self.spec.action_horizon, : self.spec.model_action_dim]
        actions = self.spec.validate_actions(actions)
        return BackendResult(actions, "absolute", self.spec.control_period_ns,
                             {"client_infer_ms": (time.perf_counter() - started) * 1000.0,
                              "remote_elapsed_ms": float(response.get("elapsed_ms", 0.0))})

    def reset(self) -> None:
        self._seq = 0

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                finally:
                    self._sock = None


def create_backend(config: dict[str, Any]) -> Pi05TcpBackend:
    return Pi05TcpBackend(config)


__all__ = ["Pi05TcpBackend", "create_backend"]
