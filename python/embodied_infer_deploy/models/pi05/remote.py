"""OpenPI Pi0.5 remote policy backend.

Pi0.5's official server uses a small WebSocket + msgpack-numpy protocol.  This
module implements that protocol without importing OpenPI, so the core package
can remain lightweight while a Pi05 environment is installed separately on
the model host.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

import msgpack
import numpy as np
from websockets.sync.client import connect

from ...core import BackendResult, ModelSpec
from ...images import as_chw_uint8
from ...plugins import VisionBatchPlugin
from .common import pi05_spec


def _pack_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise TypeError(f"unsupported Pi05 dtype: {value.dtype}")
        return {
            b"__ndarray__": True,
            b"data": np.ascontiguousarray(value).tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot msgpack value of type {type(value)!r}")


def _unpack_object(value: Any) -> Any:
    if isinstance(value, Mapping) and value.get(b"__ndarray__"):
        return np.ndarray(
            buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=tuple(value[b"shape"])
        )
    if isinstance(value, Mapping) and value.get(b"__npgeneric__"):
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def _pack(value: Mapping[str, Any]) -> bytes:
    return msgpack.packb(dict(value), default=_pack_default, use_bin_type=True)


def _unpack(value: bytes | bytearray | memoryview) -> Any:
    return msgpack.unpackb(value, raw=False, object_hook=_unpack_object, strict_map_key=False)


class Pi05RemoteBackend:
    """Adapter for an OpenPI Pi0.5 policy server."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.host = str(config.get("host", "127.0.0.1"))
        self.port = int(config.get("port", 8000))
        self.api_key = config.get("api_key")
        self.open_timeout = float(config.get("open_timeout", 10.0))
        # A Pi05 action-expert request can legitimately take longer than the
        # websocket library's 20 s keepalive window on a remote accelerator.
        self.request_timeout = float(config.get("request_timeout", 300.0))
        self.ping_interval = config.get("ping_interval", None)
        self.ping_timeout = config.get("ping_timeout", None)
        self._lock = threading.RLock()
        self._connection = None
        self._metadata: dict[str, Any] = {}
        # The stock OpenPI websocket endpoint is monolithic today.  The same
        # plugin is still attached so a split ``encode_vision`` RPC can be
        # enabled without changing the Pi05 adapter contract.
        self.vision_plugin = VisionBatchPlugin.from_config(config)
        self._spec = pi05_spec(
            "pi05-remote",
            config,
            default_model_name="Pi05",
            extras={"pi05_protocol": "openpi-websocket"},
        )

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def metadata(self) -> dict[str, Any]:
        # Keep connection lazy; startup checks can still inspect the local spec.
        value = {**self.spec.metadata(), **self._metadata}
        value["vision_batching"] = self.vision_plugin.metadata()
        return value

    def _ensure_connected(self) -> None:
        if self._connection is not None:
            return
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        self._connection = connect(
            f"ws://{self.host}:{self.port}",
            additional_headers=headers,
            open_timeout=self.open_timeout,
            ping_interval=None if self.ping_interval is None else float(self.ping_interval),
            ping_timeout=None if self.ping_timeout is None else float(self.ping_timeout),
            max_size=64 * 1024 * 1024,
            compression=None,
        )
        hello = _unpack(self._connection.recv(timeout=self.open_timeout))
        if not isinstance(hello, Mapping):
            raise RuntimeError("Pi05 server hello must be a map")
        self._metadata = dict(hello)

    def _request(self, observation: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._ensure_connected()
            try:
                self._connection.send(_pack(observation))
                response = self._connection.recv(timeout=self.request_timeout)
            except Exception:
                self.close()
                raise
        if isinstance(response, str):
            raise RuntimeError(f"Pi05 server error: {response}")
        value = _unpack(response)
        if not isinstance(value, Mapping):
            raise RuntimeError("Pi05 response must be a map")
        return dict(value)

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        images = request["images"]
        observation = {
            # The stock OpenPI server expects the post-repack observation map.
            "state": np.asarray(request["state"], dtype=np.float32),
            "images": {
                "cam_high": as_chw_uint8(images["head"]),
                "cam_left_wrist": as_chw_uint8(images["left_wrist"]),
                "cam_right_wrist": as_chw_uint8(images["right_wrist"]),
            },
            "prompt": str(request.get("instruction", "")),
        }
        started = time.perf_counter()
        response = self._request(observation)
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        # Nero/Pi05 emits the 16-D RoboTwin command vector. The shared
        # S600/TurboVLA contract exposes 14 model coordinates, so retain those
        # coordinates here; the downstream adapter reconstructs held joints.
        if actions.ndim == 2 and actions.shape[1] == 16 and self.spec.model_action_dim == 14:
            actions = actions[:, :14]
        actions = self.spec.validate_actions(actions)
        timing = {"client_infer_ms": (time.perf_counter() - started) * 1000.0}
        server_timing = response.get("server_timing")
        if isinstance(server_timing, Mapping):
            timing.update({str(k): float(v) for k, v in server_timing.items()})
        return BackendResult(
            actions=actions,
            representation=self.spec.action_representation,
            control_period_ns=self.spec.control_period_ns,
            timing=timing,
        )

    def reset(self) -> None:
        # OpenPI's stock websocket server is stateless between requests.
        return None

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            self.vision_plugin.close()


def create_backend(config: dict[str, Any]) -> Pi05RemoteBackend:
    return Pi05RemoteBackend(config)


__all__ = ["Pi05RemoteBackend", "create_backend"]
