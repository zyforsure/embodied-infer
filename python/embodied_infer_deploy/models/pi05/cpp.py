"""Embodied.cpp Pi05/VLA ZMQ backend.

This adapter keeps the wire protocol at the model boundary while exposing the
same ``ModelBackend`` contract as the WebSocket and HBM backends.  It is useful
on Jetson/Orin where the C++ daemon owns the CUDA/GGUF execution context.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...core import BackendResult, ModelSpec


def _proto_classes():
    """Build the tiny vla.proto schema without requiring protoc at install time."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    pool = descriptor_pool.Default()
    try:
        image = pool.FindMessageTypeByName("vla.Image")
        request = pool.FindMessageTypeByName("vla.PredictRequest")
        response = pool.FindMessageTypeByName("vla.PredictResponse")
        return (message_factory.GetMessageClass(image),
                message_factory.GetMessageClass(request),
                message_factory.GetMessageClass(response))
    except KeyError:
        pass
    file = descriptor_pb2.FileDescriptorProto(
        name="embodied_infer_vla.proto", package="vla", syntax="proto3")
    enum = file.message_type.add(name="Image").enum_type.add(name="Encoding")
    for name, number in (("JPEG", 0), ("RGB_U8", 1), ("F32_RGB_01", 2)):
        enum.value.add(name=name, number=number)
    image = file.message_type[-1]
    image.field.add(name="encoding", number=1, label=1, type=14,
                    type_name=".vla.Image.Encoding")
    image.field.add(name="height", number=2, label=1, type=13)
    image.field.add(name="width", number=3, label=1, type=13)
    image.field.add(name="data", number=4, label=1, type=12)

    def msg(name, fields):
        m = file.message_type.add(name=name)
        for fname, number, label, typ, type_name in fields:
            f = m.field.add(name=fname, number=number, label=label, type=typ)
            if type_name:
                f.type_name = type_name
        return m

    msg("PredictRequest", [
        ("images", 1, 3, 11, ".vla.Image"), ("lang_tokens", 2, 3, 5, ""),
        ("state", 3, 3, 2, ""), ("noise", 4, 3, 2, ""),
        ("request_id", 5, 1, 4, ""), ("language_text", 13, 1, 9, ""),
    ])
    msg("PredictResponse", [
        ("request_id", 1, 1, 4, ""), ("action_chunk", 2, 3, 2, ""),
        ("chunk_size", 3, 1, 13, ""), ("action_dim", 4, 1, 13, ""),
        ("latency_ms_total", 5, 1, 2, ""), ("error", 7, 1, 9, ""),
        ("latency_ms_inference", 6, 1, 2, ""),
        ("latency_ms_prefill", 8, 1, 2, ""),
        ("latency_ms_denoise", 9, 1, 2, ""),
        ("latency_ms_vision", 10, 1, 2, ""),
    ])
    pool.AddSerializedFile(file.SerializeToString())
    return (message_factory.GetMessageClass(pool.FindMessageTypeByName("vla.Image")),
            message_factory.GetMessageClass(pool.FindMessageTypeByName("vla.PredictRequest")),
            message_factory.GetMessageClass(pool.FindMessageTypeByName("vla.PredictResponse")))


class Pi05CppBackend:
    """Client for ``Embodied.cpp/serving/server.cpp`` (REQ/REP over ZMQ)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.address = str(config.get("address", "tcp://127.0.0.1:5555"))
        self.server_state_dim = int(config.get("server_state_dim", 32))
        self.server_action_dim = int(config.get("server_action_dim", 32))
        self.server_horizon = int(config.get("server_horizon", 50))
        self.tokenizer_path = config.get("tokenizer_path")
        self.require_tokens = bool(config.get("require_tokens", False))
        self.token_ids = config.get("token_ids")
        self.fallback_lang_length = int(config.get("fallback_lang_length", 1))
        self.timeout_ms = int(config.get("timeout_ms", 900_000))
        self._lock = threading.RLock()
        self._step = 0
        self._sock = None
        self._pb = None
        self._spec = ModelSpec(
            name=str(config.get("model_name", "Pi05-EmbodiedCpp")),
            backend="pi05-cpp-zmq",
            raw_state_dim=int(config.get("raw_state_dim", 18)),
            model_state_dim=int(config.get("model_state_dim", 14)),
            raw_action_dim=int(config.get("raw_action_dim", 18)),
            model_action_dim=int(config.get("model_action_dim", 14)),
            action_horizon=int(config.get("action_horizon", 16)),
            camera_order=tuple(config.get("camera_order", ["head", "left_wrist", "right_wrist"])),
            action_representation="absolute", control_period_ns=int(config.get("control_period_ns", 100_000_000)),
            extras={"protocol": "embodied.cpp-vla/1", "native_action_dim": int(config.get("native_action_dim", 16)),
                    "server_state_dim": self.server_state_dim, "server_action_dim": self.server_action_dim},
        )

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def metadata(self) -> dict[str, Any]:
        return self.spec.metadata()

    def _ensure(self):
        if self._sock is not None:
            return
        import zmq
        self._pb = _proto_classes()
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.connect(self.address)

    @staticmethod
    def _image(value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim != 3:
            raise ValueError(f"Pi05 image must be rank 3, got {image.shape}")
        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] != 3:
            raise ValueError(f"Pi05 image must be HWC RGB, got {image.shape}")
        image = image.astype(np.float32, copy=False)
        if image.size and float(np.max(image)) > 1.5:
            image = image / 255.0
        return np.ascontiguousarray(np.clip(image, 0.0, 1.0), dtype=np.float32)

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        with self._lock:
            self._ensure()
            Image, PredictRequest, PredictResponse = self._pb
            req = PredictRequest(request_id=self._step)
            self._step += 1
            state = np.zeros(self.server_state_dim, dtype=np.float32)
            state[: self.spec.model_state_dim] = np.asarray(request["state"], dtype=np.float32)
            req.state.extend(state.tolist())
            instruction = str(request.get("instruction", ""))
            token_ids = self.token_ids
            if token_ids is None and self.tokenizer_path:
                try:
                    import sentencepiece as spm
                    processor = spm.SentencePieceProcessor(model_file=str(Path(self.tokenizer_path)))
                    token_ids = processor.encode(instruction.strip().replace("_", " "), out_type=int)
                except Exception as exc:
                    if self.require_tokens:
                        raise RuntimeError(f"cannot load Pi05 tokenizer {self.tokenizer_path}: {exc}") from exc
            if token_ids is None and self.require_tokens:
                raise RuntimeError("Pi05 C++ backend requires tokenizer_path; install sentencepiece and configure tokenizer_path")
            if token_ids is None:
                # This keeps protocol smoke tests useful with a server that
                # accepts token IDs, while production Pi05 should configure a
                # real PaliGemma sentencepiece tokenizer.
                token_ids = [0] * max(1, self.fallback_lang_length)
            req.lang_tokens.extend(int(x) for x in token_ids)
            req.language_text = instruction
            for name in self.spec.camera_order:
                image = self._image(request["images"][name])
                item = req.images.add(encoding=2, height=image.shape[0], width=image.shape[1])
                item.data = image.tobytes()
            started = time.perf_counter()
            self._sock.send(req.SerializeToString())
            raw = self._sock.recv()
            response = PredictResponse()
            response.ParseFromString(raw)
            if response.error:
                raise RuntimeError(f"Embodied.cpp Pi05 server error: {response.error}")
            native_dim = int(response.action_dim or self.server_action_dim)
            horizon = int(response.chunk_size or self.server_horizon)
            values = np.asarray(response.action_chunk, dtype=np.float32).reshape(horizon, native_dim)
            values = values[: self.spec.action_horizon, : self.spec.model_action_dim]
            if values.shape != (self.spec.action_horizon, self.spec.model_action_dim):
                raise RuntimeError(f"C++ action shape {values.shape} cannot map to {self.spec.action_horizon}x{self.spec.model_action_dim}")
            values = self.spec.validate_actions(values)
            timing = {"client_infer_ms": (time.perf_counter() - started) * 1000.0,
                      "server_total_ms": float(response.latency_ms_total),
                      "server_inference_ms": float(response.latency_ms_inference)}
            return BackendResult(values, self.spec.action_representation, self.spec.control_period_ns, timing)

    def reset(self) -> None:
        self._step = 0

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                self._sock.close(0)
                self._sock = None


def create_backend(config: dict[str, Any]) -> Pi05CppBackend:
    return Pi05CppBackend(config)


__all__ = ["Pi05CppBackend", "create_backend"]
