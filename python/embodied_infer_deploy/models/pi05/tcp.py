"""Client for the Pi05 float TCP server used on the 4090.

The server executes the real Pi05 checkpoint; this gateway can run on Orin and
still feed its result through the normal embodied-infer/RoboTwin pipeline.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import numpy as np

from ...core import BackendResult, ModelSpec
from ...images import encode_jpeg
from ...plugins import VisionBatchPlugin
from ...protocol import recv_frame, send_frame
from .common import pi05_spec

MAGIC = b"P05R"


class Pi05TcpBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.host = str(config.get("host", "192.168.10.142"))
        self.port = int(config.get("port", 8012))
        self.timeout = float(config.get("timeout", 120.0))
        self.persistent_connection = bool(config.get("persistent_connection", True))
        self._lock = threading.RLock()
        self._sock: socket.socket | None = None
        self._seq = 0
        self.vision_plugin = VisionBatchPlugin.from_config(config)
        self._spec = pi05_spec(
            "pi05-tcp",
            config,
            default_model_name="Pi05-4090",
            extras={"protocol": "pi05-float-tcp", "remote_model": "pi05",
                    "execution_host": self.host,
                    "persistent_connection": self.persistent_connection},
        )

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def metadata(self) -> dict[str, Any]:
        value = self.spec.metadata()
        value["vision_batching"] = self.vision_plugin.metadata()
        return value

    def _connect(self) -> socket.socket:
        if self._sock is None:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self._sock.settimeout(self.timeout)
        return self._sock

    def _request(self, metadata: dict[str, Any], blobs: list[bytes]) -> tuple[dict[str, Any], list[bytes]]:
        sock = self._connect()
        try:
            send_frame(sock, MAGIC, metadata, blobs)
            return recv_frame(sock, MAGIC)
        except Exception:
            self._close_socket()
            raise
        finally:
            if not self.persistent_connection:
                self._close_socket()

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        state = np.zeros(self.spec.raw_state_dim, dtype=np.float32)
        state[: self.spec.model_state_dim] = np.asarray(request["state"], dtype=np.float32)
        blobs = [encode_jpeg(request["images"][name]) for name in self.spec.camera_order]
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
            self._close_socket()
            self.vision_plugin.close()

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


def create_backend(config: dict[str, Any]) -> Pi05TcpBackend:
    return Pi05TcpBackend(config)


__all__ = ["Pi05TcpBackend", "create_backend"]
