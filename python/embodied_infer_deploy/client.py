"""Synchronous reconnecting WebSocket client."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from websockets.sync.client import connect

from .protocol import (
    ProtocolError,
    decode_message,
    encode_message,
    make_infer_request,
    make_health_request,
    parse_action_response,
    parse_health_response,
)


class InferenceClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        api_key: str | None = None,
        open_timeout: float = 10.0,
        request_timeout: float = 2.0,
    ) -> None:
        self.uri = f"ws://{host}:{port}"
        self.api_key = api_key
        self.open_timeout = open_timeout
        self.request_timeout = request_timeout
        self._connection = None
        self._metadata: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._next_request_id = 1

    @property
    def metadata(self) -> dict[str, Any]:
        self._ensure_connected()
        return dict(self._metadata)

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def reset(self) -> None:
        with self._lock:
            self._ensure_connected_locked()
            message = {"protocol": "embodied-infer/1", "type": "reset"}
            self._connection.send(encode_message(message))
            response = decode_message(self._connection.recv(timeout=self.request_timeout))
            if response.get("type") != "reset_ok":
                raise ProtocolError(str(response.get("message", "reset failed")))

    def health(self) -> dict[str, Any]:
        """Return readiness and monotonic service counters."""
        with self._lock:
            self._ensure_connected_locked()
            self._connection.send(encode_message(make_health_request()))
            response = decode_message(self._connection.recv(timeout=self.request_timeout))
            return parse_health_response(response)

    def infer(
        self,
        *,
        control_step: int,
        timestamp_ns: int,
        instruction: str,
        images: list[tuple[str, np.ndarray, int]],
        state: np.ndarray,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        request_timeout = self.request_timeout if timeout is None else timeout
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            request = make_infer_request(
                request_id=request_id,
                control_step=control_step,
                timestamp_ns=timestamp_ns,
                instruction=instruction,
                images=images,
                state=state,
                timeout_ms=request_timeout * 1000.0,
            )
            started = time.perf_counter()
            try:
                self._ensure_connected_locked()
                self._connection.send(encode_message(request))
                raw = self._connection.recv(timeout=request_timeout)
                response = parse_action_response(decode_message(raw))
            except Exception:
                if self._connection is not None:
                    self._connection.close()
                    self._connection = None
                raise
            response["client_round_trip_ms"] = (time.perf_counter() - started) * 1000.0
            if response.get("request_id") != request_id:
                raise ProtocolError(
                    f"response request_id {response.get('request_id')} != {request_id}"
                )
            return response

    def _ensure_connected(self) -> None:
        with self._lock:
            self._ensure_connected_locked()

    def _ensure_connected_locked(self) -> None:
        if self._connection is not None:
            return
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self._connection = connect(
            self.uri,
            additional_headers=headers,
            open_timeout=self.open_timeout,
            max_size=64 * 1024 * 1024,
            compression=None,
        )
        hello = decode_message(self._connection.recv(timeout=self.open_timeout))
        if hello.get("type") != "hello" or not isinstance(hello.get("metadata"), dict):
            self._connection.close()
            self._connection = None
            raise ProtocolError("server did not send a valid hello message")
        self._metadata = hello["metadata"]
