"""Versioned, non-pickle wire format for embodied observations and actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import socket
import struct
from typing import Any

import msgpack
import numpy as np

PROTOCOL_VERSION = "embodied-infer/1"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_TENSOR_BYTES = 48 * 1024 * 1024
MAX_TENSOR_RANK = 8
MAX_IMAGES = 16
MAX_INSTRUCTION_BYTES = 16 * 1024
MAX_TIMEOUT_MS = 10 * 60 * 1000

_WIRE_DTYPES = {
    "u8": np.dtype("u1"),
    "i32": np.dtype("<i4"),
    "f32": np.dtype("<f4"),
}
_NUMPY_TO_WIRE = {
    np.dtype("u1"): "u8",
    np.dtype("i4"): "i32",
    np.dtype("f4"): "f32",
}


class ProtocolError(ValueError):
    pass


def encode_message(message: Mapping[str, Any]) -> bytes:
    payload = msgpack.packb(dict(message), use_bin_type=True)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    return payload


def decode_message(payload: bytes | bytearray | memoryview) -> dict[str, Any]:
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    try:
        value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    except (msgpack.ExtraData, msgpack.FormatError, msgpack.StackError,
            ValueError) as exc:
        raise ProtocolError(f"invalid MessagePack payload: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("top-level payload must be a map")
    if value.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol {value.get('protocol')!r}; "
            f"expected {PROTOCOL_VERSION!r}"
        )
    return value


def pack_array(value: Any, dtype: str | None = None) -> dict[str, Any]:
    array = np.asarray(value)
    if dtype is None:
        normalized = np.dtype(array.dtype.name)
        try:
            dtype = _NUMPY_TO_WIRE[normalized]
        except KeyError as exc:
            raise ProtocolError(f"unsupported tensor dtype {array.dtype}") from exc
    if dtype not in _WIRE_DTYPES:
        raise ProtocolError(f"unsupported wire dtype {dtype!r}")
    array = np.ascontiguousarray(array, dtype=_WIRE_DTYPES[dtype])
    if array.ndim > MAX_TENSOR_RANK:
        raise ProtocolError(f"tensor rank exceeds {MAX_TENSOR_RANK}")
    if array.nbytes > MAX_TENSOR_BYTES:
        raise ProtocolError(f"tensor exceeds {MAX_TENSOR_BYTES} bytes")
    return {"dtype": dtype, "shape": list(array.shape), "data": array.tobytes()}


def unpack_array(tensor: Mapping[str, Any], *, copy: bool = False) -> np.ndarray:
    if not isinstance(tensor, Mapping):
        raise ProtocolError("tensor must be a map")
    dtype_name = tensor.get("dtype")
    if dtype_name not in _WIRE_DTYPES:
        raise ProtocolError(f"unsupported wire dtype {dtype_name!r}")
    shape = tensor.get("shape")
    data = tensor.get("data")
    if (not isinstance(shape, list) or len(shape) > MAX_TENSOR_RANK or
            not all(isinstance(item, int) and item >= 0 for item in shape)):
        raise ProtocolError("tensor shape is invalid")
    if not isinstance(data, bytes) or len(data) > MAX_TENSOR_BYTES:
        raise ProtocolError("tensor data is invalid or too large")
    elements = 1
    for extent in shape:
        elements *= extent
        if elements > MAX_TENSOR_BYTES:
            raise ProtocolError("tensor shape is too large")
    dtype = _WIRE_DTYPES[dtype_name]
    if elements * dtype.itemsize != len(data):
        raise ProtocolError("tensor shape does not match its byte length")
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return array.copy() if copy else array


def make_hello(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "type": "hello",
        "metadata": dict(metadata),
    }


def make_health_request(request_id: int = 0) -> dict[str, Any]:
    """Create a lifecycle/readiness probe for a running inference service."""
    if not isinstance(request_id, int) or request_id < 0:
        raise ProtocolError("request_id must be a non-negative integer")
    return {
        "protocol": PROTOCOL_VERSION,
        "type": "health",
        "request_id": request_id,
    }


def parse_health_response(message: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a service health response."""
    if message.get("type") != "health":
        raise ProtocolError("message is not a health response")
    status = message.get("status")
    if status not in {"starting", "ready", "draining", "stopped"}:
        raise ProtocolError("health status is invalid")
    if not isinstance(message.get("metadata"), Mapping):
        raise ProtocolError("health metadata must be a map")
    for key in ("uptime_ms", "requests_total", "requests_failed"):
        value = message.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ProtocolError(f"health field {key!r} is invalid")
        if float(value) < 0:
            raise ProtocolError(f"health field {key!r} must be non-negative")
    return dict(message)


def make_infer_request(
    *,
    request_id: int,
    control_step: int,
    timestamp_ns: int,
    instruction: str,
    images: Sequence[tuple[str, np.ndarray, int]],
    state: np.ndarray,
    timeout_ms: float,
) -> dict[str, Any]:
    if request_id < 0 or control_step < 0 or timestamp_ns < 0:
        raise ProtocolError("request counters and timestamp must be non-negative")
    if not instruction or len(instruction.encode("utf-8")) > MAX_INSTRUCTION_BYTES:
        raise ProtocolError("instruction must not be empty")
    if not math.isfinite(timeout_ms) or not 0 < timeout_ms <= MAX_TIMEOUT_MS:
        raise ProtocolError(f"timeout_ms must be in (0,{MAX_TIMEOUT_MS}]")
    if len(images) > MAX_IMAGES:
        raise ProtocolError(f"at most {MAX_IMAGES} images are allowed")
    packed_images = []
    for name, image, image_timestamp_ns in images:
        if not name or image_timestamp_ns < 0:
            raise ProtocolError("image name must be non-empty and timestamp non-negative")
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
            raise ProtocolError(f"image {name!r} must have HWC layout")
        packed_images.append({
            "name": str(name),
            "timestamp_ns": int(image_timestamp_ns),
            "color": "gray" if image.shape[-1] == 1 else
                     "rgba" if image.shape[-1] == 4 else "rgb",
            "tensor": pack_array(image, "u8"),
        })
    return {
        "protocol": PROTOCOL_VERSION,
        "type": "infer",
        "request_id": int(request_id),
        "control_step": int(control_step),
        "timestamp_ns": int(timestamp_ns),
        "timeout_ms": float(timeout_ms),
        "instruction": instruction,
        "images": packed_images,
        "state": pack_array(state, "f32"),
    }


def parse_infer_request(message: Mapping[str, Any]) -> dict[str, Any]:
    if message.get("type") != "infer":
        raise ProtocolError("message is not an inference request")
    required_ints = ("request_id", "control_step", "timestamp_ns")
    for key in required_ints:
        if not isinstance(message.get(key), int) or message[key] < 0:
            raise ProtocolError(f"{key} must be a non-negative integer")
    instruction = message.get("instruction")
    if (not isinstance(instruction, str) or not instruction or
            len(instruction.encode("utf-8")) > MAX_INSTRUCTION_BYTES):
        raise ProtocolError("instruction is empty or too large")
    timeout_ms = message.get("timeout_ms")
    if (not isinstance(timeout_ms, (int, float)) or
            not math.isfinite(timeout_ms) or
            not 0 < timeout_ms <= MAX_TIMEOUT_MS):
        raise ProtocolError(f"timeout_ms must be in (0,{MAX_TIMEOUT_MS}]")
    raw_images = message.get("images")
    if not isinstance(raw_images, list) or len(raw_images) > MAX_IMAGES:
        raise ProtocolError(f"images must be a list of at most {MAX_IMAGES} entries")
    images: dict[str, np.ndarray] = {}
    image_timestamps: dict[str, int] = {}
    for item in raw_images:
        if (not isinstance(item, Mapping) or
                not isinstance(item.get("name"), str) or not item["name"]):
            raise ProtocolError("image entry is invalid")
        name = item["name"]
        if name in images:
            raise ProtocolError(f"duplicate image name {name!r}")
        array = unpack_array(item.get("tensor", {}))
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ProtocolError(f"image {name!r} must be HWC uint8")
        images[name] = array
        image_timestamp = item.get("timestamp_ns")
        if not isinstance(image_timestamp, int) or image_timestamp < 0:
            raise ProtocolError(f"image {name!r} timestamp is invalid")
        image_timestamps[name] = image_timestamp
    state = unpack_array(message.get("state", {}))
    if state.dtype != np.dtype("<f4") or state.ndim != 1:
        raise ProtocolError("state must be a one-dimensional float32 tensor")
    if not np.isfinite(state).all():
        raise ProtocolError("state contains NaN or infinity")
    return {
        "request_id": message["request_id"],
        "control_step": message["control_step"],
        "timestamp_ns": message["timestamp_ns"],
        "timeout_ms": float(timeout_ms),
        "instruction": instruction,
        "images": images,
        "image_timestamps": image_timestamps,
        "state": state,
    }


def make_action_response(
    request: Mapping[str, Any],
    actions: np.ndarray,
    *,
    representation: str = "absolute",
    control_period_ns: int = 0,
    timing: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or not np.isfinite(actions).all():
        raise ProtocolError("actions must be a finite [steps, action_dim] tensor")
    if representation not in {"absolute", "delta", "relative"}:
        raise ProtocolError(f"invalid action representation {representation!r}")
    return {
        "protocol": PROTOCOL_VERSION,
        "type": "action",
        "request_id": int(request["request_id"]),
        "first_control_step": int(request["control_step"]),
        "source_timestamp_ns": int(request["timestamp_ns"]),
        "control_period_ns": int(control_period_ns),
        "representation": representation,
        "actions": pack_array(actions, "f32"),
        "timing": {str(key): float(value) for key, value in (timing or {}).items()},
    }


def parse_action_response(message: Mapping[str, Any]) -> dict[str, Any]:
    if message.get("type") == "error":
        raise ProtocolError(str(message.get("message", "remote inference failed")))
    if message.get("type") != "action":
        raise ProtocolError("message is not an action response")
    actions = unpack_array(message.get("actions", {}), copy=True)
    if actions.dtype != np.dtype("<f4") or actions.ndim != 2:
        raise ProtocolError("actions must be a [steps, action_dim] float32 tensor")
    if not np.isfinite(actions).all():
        raise ProtocolError("actions contain NaN or infinity")
    return {**message, "actions": actions}


def make_error(message: str, request_id: int = 0, code: str = "invalid_request") -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "type": "error",
        "request_id": int(request_id),
        "code": str(code),
        "message": str(message),
    }


# --------------------------------------------------------------------------- #
# Length-prefixed TCP framing
#
# Raw socket transports (the Pi05 float TCP gateway, the S600 HBM proxy)
# share one framing scheme: a 4-byte magic, a big-endian u32 metadata
# length, a compact JSON metadata object carrying ``blob_sizes``, then the
# raw blob bytes.  Keeping the framing here lets gateways reuse it without
# adopting the versioned msgpack envelope above.
# --------------------------------------------------------------------------- #


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        value = sock.recv(size - len(chunks))
        if not value:
            raise ConnectionError("peer closed the connection")
        chunks.extend(value)
    return bytes(chunks)


def send_frame(
    sock: socket.socket,
    magic: bytes,
    metadata: Mapping[str, Any],
    blobs: Sequence[bytes] = (),
) -> None:
    payload = json.dumps(
        {**metadata, "blob_sizes": [len(blob) for blob in blobs]},
        separators=(",", ":"),
    ).encode()
    sock.sendall(magic + struct.pack("!I", len(payload)) + payload + b"".join(blobs))


def recv_frame(
    sock: socket.socket,
    magic: bytes,
    *,
    max_metadata_bytes: int = 1_000_000,
) -> tuple[dict[str, Any], list[bytes]]:
    if recv_exact(sock, len(magic)) != magic:
        raise ProtocolError("invalid frame magic")
    size = struct.unpack("!I", recv_exact(sock, 4))[0]
    if size <= 0 or size > max_metadata_bytes:
        raise ProtocolError(f"invalid frame metadata size {size}")
    metadata = json.loads(recv_exact(sock, size))
    if not isinstance(metadata, dict):
        raise ProtocolError("frame metadata must be a JSON object")
    blobs = [recv_exact(sock, int(n)) for n in metadata.get("blob_sizes", [])]
    return metadata, blobs
