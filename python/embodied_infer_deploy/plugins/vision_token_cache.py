"""Per-camera visual-token reuse and pruning (VLA-Cache / EfficientVLA style).

Successive robot frames usually change very little.  VLA-Cache observes that
unchanged visual tokens can reuse their previous computation instead of being
re-encoded every step, and EfficientVLA / TEAM-VLA additionally prune redundant
tokens before the Transformer sees them.  This plugin provides both ideas as a
single model-neutral cache:

* a per-camera fingerprint cache reuses an injected ``encoder`` result when the
  incoming frame is byte-identical to the last one for that camera;
* an optional ``pruner`` removes a configurable fraction of freshly encoded
  tokens (only on cache misses, so pruning never changes a reused result).

When no ``encoder`` is injected the plugin reports ``mode=fallback`` and
``encode`` returns the raw image unchanged, preserving the backend's original
path exactly like the vision-batching plugin.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VisionTokenCacheConfig:
    enabled: bool = False
    max_entries: int = 32
    prune_ratio: float = 0.0

    @classmethod
    def from_mapping(cls, value: Any) -> "VisionTokenCacheConfig":
        if isinstance(value, bool):
            return cls(enabled=value)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("vision_token_cache must be a boolean or JSON object")
        return cls(
            enabled=bool(value.get("enabled", True)),
            max_entries=int(value.get("max_entries", 32)),
            prune_ratio=float(value.get("prune_ratio", 0.0)),
        )

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError("vision_token_cache.max_entries must be positive")
        if not 0.0 <= self.prune_ratio < 1.0:
            raise ValueError("vision_token_cache.prune_ratio must be in [0, 1)")


class VisionTokenCachePlugin:
    """Reuse unchanged per-camera vision tokens and optionally prune tokens."""

    def __init__(
        self,
        encoder: Callable[[np.ndarray], Any] | None = None,
        pruner: Callable[[Any, float], Any] | None = None,
        config: VisionTokenCacheConfig | None = None,
    ) -> None:
        self.config = config or VisionTokenCacheConfig()
        self.encoder = encoder if self.config.enabled else None
        self.pruner = (
            pruner
            if (self.config.enabled and self.config.prune_ratio > 0.0)
            else None
        )
        self.mode = (
            "native"
            if self.encoder is not None
            else ("fallback" if self.config.enabled else "disabled")
        )
        self._cache: OrderedDict[str, tuple[bytes, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        encoder: Callable[[np.ndarray], Any] | None = None,
        pruner: Callable[[Any, float], Any] | None = None,
    ) -> "VisionTokenCachePlugin":
        return cls(encoder=encoder, pruner=pruner, config=VisionTokenCacheConfig.from_mapping(
            config.get("vision_token_cache")
        ))

    @property
    def native(self) -> bool:
        return self.mode == "native"

    @staticmethod
    def _fingerprint(image: Any) -> bytes:
        return hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()

    def encode(self, camera: str, image: Any, *, deadline: float | None = None) -> tuple[Any, bool]:
        """Return ``(tokens, reused)`` for one camera frame.

        ``reused=True`` means the tokens came from the cache and the caller can
        skip encoding entirely.  In fallback mode this returns
        ``(image, False)`` and the caller keeps its original path.
        """
        del deadline  # deadline enforcement belongs to the request scheduler
        if not self.config.enabled:
            return image, False
        fingerprint = self._fingerprint(image)
        with self._lock:
            cached = self._cache.get(camera)
            if cached is not None and cached[0] == fingerprint:
                self._hits += 1
                return cached[1], True
        if self.encoder is None:
            return image, False
        tokens = self.encoder(image)
        if self.pruner is not None:
            tokens = self.pruner(tokens, 1.0 - self.config.prune_ratio)
        with self._lock:
            self._misses += 1
            self._cache[camera] = (fingerprint, tokens)
            self._cache.move_to_end(camera)
            while len(self._cache) > self.config.max_entries:
                self._cache.popitem(last=False)
        return tokens, False

    def reset(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def close(self) -> None:
        self.reset()

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            entries = len(self._cache)
            hits = self._hits
            misses = self._misses
        total = hits + misses
        return {
            "enabled": bool(self.config.enabled),
            "mode": self.mode,
            "max_entries": self.config.max_entries,
            "prune_ratio": self.config.prune_ratio,
            "pruning": self.pruner is not None,
            "entries": entries,
            "hits": hits,
            "misses": misses,
            "reuse_rate": (hits / total) if total else 0.0,
        }


__all__ = ["VisionTokenCacheConfig", "VisionTokenCachePlugin"]
