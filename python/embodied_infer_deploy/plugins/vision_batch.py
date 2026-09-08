"""Model-agnostic continuous batching for a Vision/ViT stage.

The plugin deliberately stops at the vision boundary.  Text/KV-cache and
action-expert state stays request-local, which makes it safe to share the
plugin between Pi05 and TurboVLA adapters.  A backend can provide an
``encode_vision(batch)`` callable for native batching; without one the plugin
reports a compatibility fallback and leaves the backend's original path
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future
from typing import Any, Callable, Mapping, Sequence

import numpy as np

@dataclass(frozen=True)
class VisionBatchConfig:
    enabled: bool = False
    max_batch_size: int = 4
    batch_wait_ms: float = 2.0
    max_queue_size: int = 32

    @classmethod
    def from_mapping(cls, value: Any) -> "VisionBatchConfig":
        if isinstance(value, bool):
            return cls(enabled=value)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("vision_batching must be a boolean or JSON object")
        return cls(
            enabled=bool(value.get("enabled", True)),
            max_batch_size=int(value.get("max_batch_size", 4)),
            batch_wait_ms=float(value.get("batch_wait_ms", 2.0)),
            max_queue_size=int(value.get("max_queue_size", 32)),
        )


class VisionBatchPlugin:
    """Attach continuous vision batching to a model adapter.

    ``encoder`` is intentionally an injectable callable.  Production
    TensorRT/HBM adapters pass their native vision entry point, while remote
    Pi05/TurboVLA gateways can leave it unset until the server exposes a split
    vision RPC.  This keeps the same configuration and observability contract
    in both cases.
    """

    def __init__(self, encoder: Callable[[np.ndarray], Any] | None = None,
                 config: VisionBatchConfig | None = None) -> None:
        self.config = config or VisionBatchConfig()
        self.encoder = encoder if self.config.enabled else None
        self.mode = "native" if self.encoder is not None else (
            "fallback" if self.config.enabled else "disabled"
        )
        # Import lazily: model packages import this plugin while the registry
        # is still being initialized, so a top-level Pi05 import would cycle.
        batcher_type = None
        if self.encoder is not None:
            from ..models.pi05.vision_batch import ViTContinuousBatcher
            batcher_type = ViTContinuousBatcher
        self._batcher = (
            batcher_type(
                self.encoder,
                max_batch_size=self.config.max_batch_size,
                batch_wait_ms=self.config.batch_wait_ms,
                max_queue_size=self.config.max_queue_size,
            )
            if self.encoder is not None else None
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, encoder=None) -> "VisionBatchPlugin":
        return cls(encoder=encoder, config=VisionBatchConfig.from_mapping(
            config.get("vision_batching")
        ))

    @property
    def native(self) -> bool:
        return self.mode == "native"

    def submit(self, images: Sequence[Any], *, deadline: float | None = None) -> Future:
        if self._batcher is not None:
            return self._batcher.submit(images, deadline=deadline)
        future: Future = Future()
        try:
            from ..models.pi05.vision_batch import ViTContinuousBatcher
            future.set_result(ViTContinuousBatcher._canonical(images))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def encode(self, images: Sequence[Any], *, deadline: float | None = None) -> Any:
        """Encode one request, using the shared batch queue when available."""
        return self.submit(images, deadline=deadline).result()

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "mode": self.mode,
            "max_batch_size": self.config.max_batch_size,
            "batch_wait_ms": self.config.batch_wait_ms,
            "max_queue_size": self.config.max_queue_size,
        }

    def close(self) -> None:
        if self._batcher is not None:
            self._batcher.close()


__all__ = ["VisionBatchConfig", "VisionBatchPlugin"]
