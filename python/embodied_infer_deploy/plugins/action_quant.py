"""Action-aware per-channel mixed-precision quantization (QVLA style).

Generic LLM/MLLM quantization optimizes for data fidelity; VLA actions are not
distributed that way.  QVLA / AutoQVLA observe that a few action-sensitive
channels dominate downstream control quality, so those channels deserve more
bit-width than the rest.  This plugin implements that idea as a model-neutral,
training-free post-training quantizer:

* ``fit`` computes per-channel min/max scale/zero-point along a channel axis;
* channels with the highest supplied ``importance`` are allocated
  ``sensitive_bits`` and the rest ``default_bits``;
* ``quantize`` / ``dequantize`` round-trip any tensor through that state.

The quantized values are returned as floats holding integer values so a backend
can cast them to the matching storage dtype (uint8/int16/...); the framework
keeps the math dtype-agnostic and dependency-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActionQuantConfig:
    enabled: bool = False
    sensitive_bits: int = 16
    default_bits: int = 8
    sensitive_ratio: float = 0.25
    axis: int = 0

    @classmethod
    def from_mapping(cls, value: Any) -> "ActionQuantConfig":
        if isinstance(value, bool):
            return cls(enabled=value)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("action_quant must be a boolean or JSON object")
        return cls(
            enabled=bool(value.get("enabled", True)),
            sensitive_bits=int(value.get("sensitive_bits", 16)),
            default_bits=int(value.get("default_bits", 8)),
            sensitive_ratio=float(value.get("sensitive_ratio", 0.25)),
            axis=int(value.get("axis", 0)),
        )

    def __post_init__(self) -> None:
        for bits in (self.sensitive_bits, self.default_bits):
            if bits not in (4, 8, 16):
                raise ValueError("action_quant bit-widths must be 4, 8, or 16")
        if not 0.0 <= self.sensitive_ratio <= 1.0:
            raise ValueError("action_quant.sensitive_ratio must be in [0, 1]")
        if self.sensitive_bits < self.default_bits:
            raise ValueError("sensitive_bits must be >= default_bits")


@dataclass
class ActionQuantState:
    """Per-channel quantization parameters computed by :meth:`fit`."""

    scales: np.ndarray
    zeros: np.ndarray
    bits: np.ndarray
    axis: int = 0

    @property
    def channels(self) -> int:
        return int(self.scales.shape[0])


class ActionQuantPlugin:
    """Training-free channel-wise mixed-precision quantizer."""

    def __init__(self, config: ActionQuantConfig | None = None) -> None:
        self.config = config or ActionQuantConfig()
        self.mode = "native" if self.config.enabled else "disabled"
        self._state: ActionQuantState | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ActionQuantPlugin":
        return cls(ActionQuantConfig.from_mapping(config.get("action_quant")))

    @property
    def native(self) -> bool:
        return self.mode == "native"

    def _channel_matrix(self, weights: Any, axis: int) -> tuple[np.ndarray, int]:
        arr = np.asarray(weights, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError("action_quant requires a tensor with a channel axis")
        if axis < 0:
            axis += arr.ndim
        if not 0 <= axis < arr.ndim:
            raise ValueError(f"action_quant axis {axis} out of range for rank {arr.ndim}")
        channels = arr.shape[axis]
        return np.moveaxis(arr, axis, 0).reshape(channels, -1), channels

    def fit(self, weights: Any, importance: Any) -> ActionQuantState:
        """Calibrate per-channel scales, zero-points, and bit-widths."""
        matrix, channels = self._channel_matrix(weights, self.config.axis)
        importance_arr = np.asarray(importance, dtype=np.float32)
        if importance_arr.shape != (channels,):
            raise ValueError(
                f"importance must have shape [{channels}], got {importance_arr.shape}"
            )
        bits = np.full(channels, int(self.config.default_bits), dtype=np.int32)
        sensitive_count = max(1, int(round(channels * self.config.sensitive_ratio)))
        if channels > 0 and sensitive_count > 0:
            top = np.argsort(importance_arr)[-sensitive_count:]
            bits[top] = int(self.config.sensitive_bits)

        mins = matrix.min(axis=1)
        maxs = matrix.max(axis=1)
        scales = np.zeros(channels, dtype=np.float32)
        zeros = np.zeros(channels, dtype=np.float32)
        for channel in range(channels):
            qmax = float((1 << int(bits[channel])) - 1)
            span = float(maxs[channel] - mins[channel])
            scales[channel] = (span / qmax) if span > 1e-12 else 1.0
            zero = round(-float(mins[channel]) / float(scales[channel]))
            zeros[channel] = float(np.clip(zero, 0.0, qmax))
        self._state = ActionQuantState(
            scales=scales, zeros=zeros, bits=bits, axis=self.config.axis
        )
        return self._state

    def quantize(self, weights: Any, state: ActionQuantState | None = None) -> np.ndarray:
        """Quantize ``weights`` into integer-valued floats using ``state``."""
        state = state or self._require_state()
        matrix, channels = self._channel_matrix(weights, state.axis)
        if channels != state.channels:
            raise ValueError(
                f"expected {state.channels} channels, got {channels}"
            )
        output = np.empty_like(matrix, dtype=np.float32)
        for channel in range(channels):
            qmax = float((1 << int(state.bits[channel])) - 1)
            values = np.round(matrix[channel] / float(state.scales[channel])) + float(state.zeros[channel])
            output[channel] = np.clip(values, 0.0, qmax)
        return np.moveaxis(output, 0, state.axis)

    def dequantize(self, quantized: Any, state: ActionQuantState | None = None) -> np.ndarray:
        """Approximate the original tensor from ``quantize`` output."""
        state = state or self._require_state()
        matrix, channels = self._channel_matrix(quantized, state.axis)
        if channels != state.channels:
            raise ValueError(
                f"expected {state.channels} channels, got {channels}"
            )
        output = np.empty_like(matrix, dtype=np.float32)
        for channel in range(channels):
            output[channel] = (
                matrix[channel] - float(state.zeros[channel])
            ) * float(state.scales[channel])
        return np.moveaxis(output, 0, state.axis)

    def error(self, weights: Any, state: ActionQuantState | None = None) -> np.ndarray:
        """Per-channel mean-squared round-trip error."""
        state = state or self._require_state()
        source, _ = self._channel_matrix(weights, state.axis)
        target, _ = self._channel_matrix(self.dequantize(self.quantize(weights, state), state), state.axis)
        return ((target - source) ** 2).mean(axis=1)

    def _require_state(self) -> ActionQuantState:
        if self._state is None:
            raise RuntimeError("action_quant plugin has not been fit() yet")
        return self._state

    def reset(self) -> None:
        self._state = None

    def close(self) -> None:
        self.reset()

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "mode": self.mode,
            "sensitive_bits": self.config.sensitive_bits,
            "default_bits": self.config.default_bits,
            "sensitive_ratio": self.config.sensitive_ratio,
            "axis": self.config.axis,
            "calibrated": self._state is not None,
        }


__all__ = ["ActionQuantConfig", "ActionQuantState", "ActionQuantPlugin"]
