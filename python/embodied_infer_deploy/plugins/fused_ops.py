# -*- coding: utf-8 -*-
"""Fused transformer operators for PyTorch-side VLA stages.

TensorRT engines already fuse operators internally, so the split TurboVLA
vision/text/fusion/action stack does not need these kernels.  They cover the
stages that still run plain array code in the deploy layer: action heads,
PyTorch reference backends, adapter glue, and training-side tooling.  The
operator set mirrors the transformer blocks used across TurboVLA (ViT
vision encoder, BERT text encoder, fusion transformer, action expert):

* fused QKV projection: one packed GEMM instead of three separate ones;
* fused scaled-dot-product attention (SDPA) instead of an explicit
  QK^T -> softmax -> AV kernel chain;
* fused GELU-MLP (addmm + tanh-GELU epilogue + addmm);
* fused residual-add + LayerNorm.

Every operator keeps the house native/fallback boundary: with torch
available the native path runs fused kernels (CUDA when present), otherwise
the NumPy reference computes the same result, so enabling the plugin never
breaks a backend.  ``native_fn`` lets a model adapter inject engine-native
kernels per operator name ("qkv", "attention", "gelu_mlp", "layer_norm").
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_TORCH: Any = None
_TORCH_CHECKED = False


def _torch() -> Any:
    global _TORCH, _TORCH_CHECKED
    if not _TORCH_CHECKED:
        try:
            import torch

            _TORCH = torch
        except Exception:  # pragma: no cover - torch absent is an env choice
            _TORCH = None
        _TORCH_CHECKED = True
    return _TORCH


@dataclass(frozen=True)
class FusedOpsConfig:
    enabled: bool = False
    prefer: str = "auto"  # auto | torch | numpy

    @classmethod
    def from_mapping(cls, value: Any) -> "FusedOpsConfig":
        if isinstance(value, bool):
            return cls(enabled=value)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("fused_ops must be a boolean or JSON object")
        return cls(
            enabled=bool(value.get("enabled", True)),
            prefer=str(value.get("prefer", "auto")),
        )

    def __post_init__(self) -> None:
        if self.prefer not in ("auto", "torch", "numpy"):
            raise ValueError("fused_ops.prefer must be auto, torch, or numpy")


# --------------------------------------------------------------------------- #
# NumPy reference operators
# --------------------------------------------------------------------------- #

def qkv_reference(hidden: np.ndarray, weights: tuple[np.ndarray, ...],
                  biases: tuple[np.ndarray | None, ...] | None = None,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate Q/K/V projections, PyTorch linear layout (y = x @ W.T + b)."""

    if len(weights) != 3:
        raise ValueError(f"qkv expects 3 weight matrices, got {len(weights)}")
    if biases is None:
        biases = (None, None, None)
    outputs = []
    for weight, bias in zip(weights, biases):
        value = np.asarray(hidden, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T
        if bias is not None:
            value = value + np.asarray(bias, dtype=np.float32)
        outputs.append(np.asarray(value, dtype=np.float32))
    return outputs[0], outputs[1], outputs[2]


def attention_reference(query: np.ndarray, key: np.ndarray, value: np.ndarray,
                        *, causal: bool = False, scale: float | None = None,
                        ) -> np.ndarray:
    """Scaled dot-product attention on [B, H, T, D] float32 arrays."""

    q = np.asarray(query, dtype=np.float32)
    k = np.asarray(key, dtype=np.float32)
    v = np.asarray(value, dtype=np.float32)
    head_dim = q.shape[-1]
    factor = float(scale) if scale is not None else 1.0 / float(np.sqrt(head_dim))
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) * factor
    if causal:
        t_q, t_k = scores.shape[-2], scores.shape[-1]
        mask = np.tri(t_q, t_k, k=t_k - t_q, dtype=bool)
        scores = np.where(mask[None, None], scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", weights, v).astype(np.float32)


def gelu_mlp_reference(hidden: np.ndarray, w1: np.ndarray, b1: np.ndarray | None,
                       w2: np.ndarray, b2: np.ndarray | None) -> np.ndarray:
    """tanh-GELU MLP with PyTorch linear layout."""

    x = np.asarray(hidden, dtype=np.float32) @ np.asarray(w1, dtype=np.float32).T
    if b1 is not None:
        x = x + np.asarray(b1, dtype=np.float32)
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    x = 0.5 * x * (1.0 + np.tanh(inner))
    x = x @ np.asarray(w2, dtype=np.float32).T
    if b2 is not None:
        x = x + np.asarray(b2, dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def layer_norm_reference(hidden: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                         *, residual: np.ndarray | None = None,
                         eps: float = 1e-6) -> np.ndarray:
    """LayerNorm over the last dimension with optional fused residual add."""

    x = np.asarray(hidden, dtype=np.float32)
    if residual is not None:
        x = x + np.asarray(residual, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(var + float(eps))
    return (x * np.asarray(weight, dtype=np.float32)
            + np.asarray(bias, dtype=np.float32)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Plugin
# --------------------------------------------------------------------------- #

_OPS = ("qkv", "attention", "gelu_mlp", "layer_norm")


class FusedOpsPlugin:
    """Fused transformer operators with a torch-native / NumPy boundary."""

    def __init__(
        self,
        config: FusedOpsConfig | None = None,
        *,
        native_fn: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.config = config or FusedOpsConfig()
        self._overrides = dict(native_fn or {})
        unknown = set(self._overrides) - set(_OPS)
        if unknown:
            raise ValueError(f"unknown fused operator overrides: {sorted(unknown)}")
        self._lock = threading.Lock()
        self._device: Any = None
        self._device_ready = False
        self._packed: dict[tuple[Any, ...], Any] = {}
        self._calls = {name: [0, 0] for name in _OPS}  # [native, fallback]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        native_fn: Mapping[str, Callable[..., Any]] | None = None,
    ) -> "FusedOpsPlugin":
        return cls(config=FusedOpsConfig.from_mapping(config.get("fused_ops")),
                   native_fn=native_fn)

    # -- mode bookkeeping -------------------------------------------------- #

    @property
    def mode(self) -> str:
        if not self.config.enabled:
            return "disabled"
        return "native" if self._torch_available() else "fallback"

    @property
    def native(self) -> bool:
        return self.mode == "native"

    def _torch_available(self) -> bool:
        return self.config.prefer != "numpy" and _torch() is not None

    def _use_native(self, op: str) -> bool:
        if not self.config.enabled:
            return False
        if op in self._overrides:
            return True
        return self._torch_available()

    def _record(self, op: str, native: bool) -> None:
        with self._lock:
            self._calls[op][0 if native else 1] += 1

    def _resolve_device(self) -> Any:
        if not self._device_ready:
            torch = _torch()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._device_ready = True
        return self._device

    def _tensor(self, array: np.ndarray) -> Any:
        torch = _torch()
        return torch.as_tensor(np.ascontiguousarray(array, dtype=np.float32),
                               device=self._resolve_device())

    # -- operators ---------------------------------------------------------- #

    def project_qkv(self, hidden: np.ndarray,
                    weights: tuple[np.ndarray, ...],
                    biases: tuple[np.ndarray | None, ...] | None = None,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (q, k, v); the native path packs one GEMM per call."""

        override = self._overrides.get("qkv")
        if self.config.enabled and override is not None:
            self._record("qkv", True)
            return override(hidden, weights, biases)
        if self._use_native("qkv"):
            torch = _torch()
            key = tuple((w.__array_interface__["data"][0], w.shape, str(w.dtype))
                        for w in weights)
            packed = self._packed.get(key)
            if packed is None:
                packed = torch.cat([self._tensor(w) for w in weights], dim=0)
                self._packed[key] = packed
            x = self._tensor(hidden)
            out_features = [w.shape[0] for w in weights]
            flat = x.reshape(-1, x.shape[-1])
            if biases is not None and any(b is not None for b in biases):
                bias = torch.cat([
                    self._tensor(b) if b is not None
                    else torch.zeros(out, device=x.device)
                    for b, out in zip(biases, out_features)
                ])
                fused = torch.addmm(bias, flat, packed.t())
            else:
                fused = flat @ packed.t()
            parts = torch.split(fused, out_features, dim=-1)
            result = tuple(
                part.reshape(*x.shape[:-1], out).cpu().numpy().astype(np.float32)
                for part, out in zip(parts, out_features)
            )
            self._record("qkv", True)
            return result  # type: ignore[return-value]
        self._record("qkv", False)
        return qkv_reference(hidden, weights, biases)

    def attend(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
               *, causal: bool = False, scale: float | None = None) -> np.ndarray:
        """Scaled dot-product attention on [B, H, T, D] arrays."""

        override = self._overrides.get("attention")
        if self.config.enabled and override is not None:
            self._record("attention", True)
            return override(query, key, value, causal=causal, scale=scale)
        if self._use_native("attention"):
            torch = _torch()
            q = self._tensor(query)
            k = self._tensor(key)
            v = self._tensor(value)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal,
                scale=scale if scale is not None else None,
            )
            self._record("attention", True)
            return out.cpu().numpy().astype(np.float32)
        self._record("attention", False)
        return attention_reference(query, key, value, causal=causal, scale=scale)

    def gelu_mlp(self, hidden: np.ndarray, w1: np.ndarray,
                 b1: np.ndarray | None, w2: np.ndarray,
                 b2: np.ndarray | None) -> np.ndarray:
        """tanh-GELU MLP with fused addmm epilogues in the native path."""

        override = self._overrides.get("gelu_mlp")
        if self.config.enabled and override is not None:
            self._record("gelu_mlp", True)
            return override(hidden, w1, b1, w2, b2)
        if self._use_native("gelu_mlp"):
            torch = _torch()
            x = self._tensor(hidden)
            flat = x.reshape(-1, x.shape[-1])
            w1_t = self._tensor(w1)
            w2_t = self._tensor(w2)
            if b1 is not None:
                x = torch.addmm(self._tensor(b1), flat, w1_t.t())
            else:
                x = flat @ w1_t.t()
            x = torch.nn.functional.gelu(x, approximate="tanh")
            if b2 is not None:
                x = torch.addmm(self._tensor(b2), x, w2_t.t())
            else:
                x = x @ w2_t.t()
            out = x.reshape(*np.asarray(hidden).shape[:-1], w2.shape[0])
            self._record("gelu_mlp", True)
            return out.cpu().numpy().astype(np.float32)
        self._record("gelu_mlp", False)
        return gelu_mlp_reference(hidden, w1, b1, w2, b2)

    def layer_norm(self, hidden: np.ndarray, weight: np.ndarray,
                   bias: np.ndarray, *, residual: np.ndarray | None = None,
                   eps: float = 1e-6) -> np.ndarray:
        """LayerNorm with optional fused residual add."""

        override = self._overrides.get("layer_norm")
        if self.config.enabled and override is not None:
            self._record("layer_norm", True)
            return override(hidden, weight, bias, residual=residual, eps=eps)
        if self._use_native("layer_norm"):
            torch = _torch()
            x = self._tensor(hidden)
            if residual is not None:
                x = x + self._tensor(residual)
            out = torch.nn.functional.layer_norm(
                x, (x.shape[-1],), self._tensor(weight), self._tensor(bias),
                float(eps),
            )
            self._record("layer_norm", True)
            return out.cpu().numpy().astype(np.float32)
        self._record("layer_norm", False)
        return layer_norm_reference(hidden, weight, bias, residual=residual,
                                    eps=eps)

    # -- lifecycle ----------------------------------------------------------- #

    def reset(self) -> None:
        with self._lock:
            self._packed.clear()
            self._calls = {name: [0, 0] for name in _OPS}

    def close(self) -> None:
        self.reset()

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            calls = {name: {"native": counts[0], "fallback": counts[1]}
                     for name, counts in self._calls.items()}
            packed = len(self._packed)
        backend = "none"
        if self._device_ready and _torch() is not None:
            backend = f"torch-{self._device.type}"
        elif _torch() is not None:
            backend = "torch"
        return {
            "enabled": bool(self.config.enabled),
            "mode": self.mode,
            "prefer": self.config.prefer,
            "backend": backend,
            "packed_weights": packed,
            "calls": calls,
        }


__all__ = [
    "FusedOpsConfig",
    "FusedOpsPlugin",
    "attention_reference",
    "gelu_mlp_reference",
    "layer_norm_reference",
    "qkv_reference",
]