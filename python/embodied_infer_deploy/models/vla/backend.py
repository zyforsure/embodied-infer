"""Remote VLA backend over a length-prefixed msgpack TCP channel.

The heavy model lives on a GPU gateway (4090); this backend only marshals the
observation, so the Jetson side stays thin.  One class covers every remote
architecture - which model is served is decided by the config.
"""

from __future__ import annotations

import io
import socket
import struct
import time
from typing import Any

import msgpack
import numpy as np
from PIL import Image

from ...core import BackendResult, ModelSpec


def _recv_msg(conn: socket.socket):
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


def _send_msg(conn: socket.socket, obj: dict) -> None:
    payload = msgpack.packb(obj, use_bin_type=True)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def _encode_image(value: Any, image_size: tuple[int, int] | None = None) -> bytes:
    """Accept HWC/CHW uint8 arrays or raw bytes and return JPEG bytes.

    ``image_size`` is an optional ``(width, height)`` the frame is resized to;
    some servers (e.g. GR00T gr1_arms_only) reject other resolutions.
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)  # CHW -> HWC
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    image = Image.fromarray(array)
    if image_size is not None and image.size != tuple(image_size):
        image = image.resize(tuple(image_size), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class VlaRemoteBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.host: str = config["host"]
        self.port: int = int(config["port"])
        self.arch: str = config.get("arch", "openvla")
        self.timeout_s: float = float(config.get("timeout_s", 30.0))
        self.persistent: bool = bool(config.get("persistent_connection", True))
        self._conn: socket.socket | None = None

        camera_order = tuple(config.get("camera_order") or ("front",))
        size = config.get("image_size")
        self._image_size = tuple(int(v) for v in size) if size else None
        horizon = int(config.get("action_horizon", 1))
        model_action_dim = int(config.get("model_action_dim", 7))
        model_state_dim = int(config.get("model_state_dim", 7))

        self.spec = ModelSpec(
            name=config.get("name", f"{self.arch}-remote"),
            backend=config.get("backend", f"{self.arch}-tcp"),
            raw_state_dim=int(config.get("raw_state_dim", model_state_dim)),
            model_state_dim=model_state_dim,
            raw_action_dim=int(config.get("raw_action_dim", model_action_dim)),
            model_action_dim=model_action_dim,
            action_horizon=horizon,
            camera_order=camera_order,
            action_representation=config.get("action_representation", "absolute"),
            control_period_ns=int(config.get("control_period_ns", 0)),
            extras={
                "arch": self.arch,
                "host": self.host,
                "port": self.port,
                **config.get("extras", {}),
            },
        )
        self._horizon = horizon

    @property
    def metadata(self) -> dict[str, Any]:
        return self.spec.metadata()

    # -- transport ---------------------------------------------------------
    def _connect(self) -> socket.socket:
        conn = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return conn

    def _request_once(self, payload: dict) -> dict:
        conn = self._conn
        if conn is None:
            conn = self._connect()
            if self.persistent:
                self._conn = conn
        try:
            _send_msg(conn, payload)
            return _recv_msg(conn)
        except Exception:
            # a stale pooled socket should not fail the control step
            self._close()
            if not self.persistent:
                raise
            conn = self._connect()
            self._conn = conn
            _send_msg(conn, payload)
            return _recv_msg(conn)

    def _close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    # -- ModelBackend ------------------------------------------------------
    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        started = time.perf_counter()

        images = {}
        for name in self.spec.camera_order:
            if name in request.get("images", {}):
                images[name] = _encode_image(request["images"][name], self._image_size)
        if not images:
            raise ValueError(f"no known cameras in request: {sorted(request.get('images', {}))}")

        payload = {
            "arch": self.arch,
            "images": images,
            "instruction": request.get("instruction", "") or "",
            "state": np.asarray(request["state"], dtype=np.float32).tolist(),
            "horizon": self._horizon,
        }

        response = self._request_once(payload)
        if response is None:
            raise RuntimeError("remote VLA server closed the connection")
        if response.get("error"):
            raise RuntimeError(f"remote VLA error: {response['error']}")

        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        # pad or trim to the declared horizon so the contract always holds
        if actions.shape[0] < self.spec.action_horizon:
            pad = np.repeat(actions[-1:], self.spec.action_horizon - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        actions = actions[: self.spec.action_horizon]

        timing = dict(response.get("timing") or {})
        timing["round_trip_s"] = time.perf_counter() - started
        return BackendResult(
            actions=self.spec.validate_actions(actions),
            representation=self.spec.action_representation,
            control_period_ns=self.spec.control_period_ns,
            timing=timing,
        )

    def reset(self) -> None:
        self._close()


def create_backend(config: dict[str, Any]) -> VlaRemoteBackend:
    return VlaRemoteBackend(config)


__all__ = ["VlaRemoteBackend", "create_backend"]
