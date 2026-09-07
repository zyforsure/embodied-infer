"""WebSocket inference server for Orin or a discrete-GPU workstation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from websockets.sync.server import serve

from .models import create_model, model_registry
from .protocol import (
    ProtocolError,
    decode_message,
    encode_message,
    make_action_response,
    make_error,
    make_hello,
    parse_infer_request,
)


def load_factory(spec: str):
    if ":" not in spec:
        raise ValueError("backend factory must use module:callable syntax")
    module_name, attribute = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"backend factory {spec!r} is not callable")
    return factory


class InferenceServer:
    def __init__(self, backend, *, api_key: str | None = None) -> None:
        self.backend = backend
        self.api_key = api_key
        self._backend_lock = threading.Lock()

    def _validate_shapes(self, request, actions=None) -> None:
        metadata = self.backend.metadata
        state_dim = int(metadata.get("model_state_dim", metadata.get("state_dim", -1)))
        if request["state"].shape != (state_dim,):
            raise ProtocolError(
                f"wire state must have shape [{state_dim}], got "
                f"{request['state'].shape}"
            )
        if actions is None:
            return
        action_dim = int(metadata.get(
            "model_action_dim", metadata.get("action_dim", -1)
        ))
        horizon = int(metadata.get("action_horizon", -1))
        if actions.shape != (horizon, action_dim):
            raise RuntimeError(
                f"backend actions must have shape [{horizon},{action_dim}], "
                f"got {actions.shape}"
            )

    def handler(self, websocket) -> None:
        if self.api_key:
            authorization = websocket.request.headers.get("Authorization")
            if authorization != f"Bearer {self.api_key}":
                websocket.close(code=1008, reason="unauthorized")
                return
        websocket.send(encode_message(make_hello(self.backend.metadata)))
        for payload in websocket:
            request_id = 0
            try:
                message = decode_message(payload)
                request_id = int(message.get("request_id", 0))
                if message.get("type") == "reset":
                    with self._backend_lock:
                        self.backend.reset()
                    websocket.send(encode_message({
                        "protocol": "embodied-infer/1", "type": "reset_ok"
                    }))
                    continue
                request = parse_infer_request(message)
                started = time.perf_counter()
                deadline = started + request["timeout_ms"] / 1000.0
                self._validate_shapes(request)
                with self._backend_lock:
                    queue_ms = (time.perf_counter() - started) * 1000.0
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("request deadline expired in server queue")
                    result = self.backend.infer(request)
                if time.perf_counter() >= deadline:
                    raise TimeoutError("request deadline expired during inference")
                self._validate_shapes(request, result.actions)
                total_ms = (time.perf_counter() - started) * 1000.0
                timing = dict(result.timing)
                timing.update({"server_queue_ms": queue_ms, "server_total_ms": total_ms})
                response = make_action_response(
                    request,
                    result.actions,
                    representation=result.representation,
                    control_period_ns=result.control_period_ns,
                    timing=timing,
                )
            except ProtocolError as exc:
                response = make_error(str(exc), request_id, "invalid_request")
            except TimeoutError as exc:
                response = make_error(str(exc), request_id, "deadline_exceeded")
            except Exception as exc:
                traceback.print_exc()
                response = make_error(str(exc), request_id, "backend_error")
            websocket.send(encode_message(response))


def main() -> None:
    parser = argparse.ArgumentParser()
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--model",
        help="registered model name; use --list-models to inspect names",
    )
    backend.add_argument(
        "--backend-factory",
        help="legacy or custom module:callable factory",
    )
    parser.add_argument("--backend-config")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=44091)
    parser.add_argument("--api-key-env", default="EMBODIED_INFER_API_KEY")
    args = parser.parse_args()

    if args.list_models:
        print("\n".join(model_registry.names()))
        return
    if not args.backend_config:
        parser.error("--backend-config is required unless --list-models is used")
    config = json.loads(Path(args.backend_config).read_text(encoding="utf-8"))
    model_backend = (
        load_factory(args.backend_factory)(config)
        if args.backend_factory
        else create_model(args.model or "turbovla-tensorrt", config)
    )
    application = InferenceServer(model_backend, api_key=os.getenv(args.api_key_env))
    print(
        f"embodied-infer server listening on {args.host}:{args.port} "
        f"backend={model_backend.metadata.get('backend')}",
        flush=True,
    )
    with serve(
        application.handler,
        args.host,
        args.port,
        max_size=64 * 1024 * 1024,
        compression=None,
        ping_interval=20,
        ping_timeout=20,
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
