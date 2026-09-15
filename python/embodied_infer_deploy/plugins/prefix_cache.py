"""Instruction prefix cache for reuse across control steps.

VLA episodes usually keep the same natural-language instruction for many
consecutive control steps.  Re-tokenizing (and, on a KV-cache-capable engine,
re-encoding) the instruction every step is pure waste.  This plugin caches an
opaque per-instruction *prefix* and hands it back to the backend, so the prompt
is only prepared once per episode.  The technique follows the prefix-KV-cache
portion of BLURR (arXiv 2512.11769).

The plugin is deliberately model-neutral.  A backend injects a ``prefixer``
callable that returns whatever the engine consumes (token ids, a tensor, a KV
handle).  When no ``prefixer`` is available the plugin reports
``mode=fallback`` and ``get_or_compute`` returns ``(None, False)`` so the
backend keeps its original monolithic path untouched.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefixCacheConfig:
    enabled: bool = False
    max_entries: int = 1024
    normalize: bool = True

    @classmethod
    def from_mapping(cls, value: Any) -> "PrefixCacheConfig":
        if isinstance(value, bool):
            return cls(enabled=value)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("prefix_cache must be a boolean or JSON object")
        return cls(
            enabled=bool(value.get("enabled", True)),
            max_entries=int(value.get("max_entries", 1024)),
            normalize=bool(value.get("normalize", True)),
        )

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError("prefix_cache.max_entries must be positive")


class PrefixCachePlugin:
    """Cache one opaque prefix per normalized instruction string."""

    def __init__(
        self,
        prefixer: Callable[[str], Any] | None = None,
        config: PrefixCacheConfig | None = None,
    ) -> None:
        self.config = config or PrefixCacheConfig()
        self.prefixer = prefixer if self.config.enabled else None
        self.mode = (
            "native"
            if self.prefixer is not None
            else ("fallback" if self.config.enabled else "disabled")
        )
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, prefixer: Callable[[str], Any] | None = None
    ) -> "PrefixCachePlugin":
        return cls(prefixer=prefixer, config=PrefixCacheConfig.from_mapping(
            config.get("prefix_cache")
        ))

    @property
    def native(self) -> bool:
        return self.mode == "native"

    def _key(self, text: str) -> str:
        value = str(text)
        if self.config.normalize:
            value = " ".join(value.split()).strip()
        return value

    def lookup(self, text: str) -> Any | None:
        """Return a cached prefix without changing cache statistics."""
        with self._lock:
            return self._cache.get(self._key(text))

    def put(self, text: str, prefix: Any) -> None:
        """Explicitly store a prefix for ``text`` (no statistics update)."""
        if not self.config.enabled:
            return
        with self._lock:
            key = self._key(text)
            self._cache[key] = prefix
            self._cache.move_to_end(key)
            while len(self._cache) > self.config.max_entries:
                self._cache.popitem(last=False)

    def get_or_compute(self, text: str) -> tuple[Any | None, bool]:
        """Return ``(prefix, hit)``.

        A ``hit`` means the prefix was reused.  On a miss with a native
        ``prefixer`` the prefix is computed, cached, and returned with
        ``hit=False``.  In fallback mode this returns ``(None, False)`` and the
        caller keeps its original execution path.
        """
        if not self.config.enabled:
            return None, False
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return self._cache[key], True
        if self.prefixer is None:
            return None, False
        prefix = self.prefixer(text)
        with self._lock:
            self._misses += 1
            self._cache[key] = prefix
            self._cache.move_to_end(key)
            while len(self._cache) > self.config.max_entries:
                self._cache.popitem(last=False)
        return prefix, False

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
            "normalize": self.config.normalize,
            "entries": entries,
            "hits": hits,
            "misses": misses,
            "hit_rate": (hits / total) if total else 0.0,
        }


__all__ = ["PrefixCacheConfig", "PrefixCachePlugin"]
