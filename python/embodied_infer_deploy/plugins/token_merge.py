"""Redundant visual-token merging (TEAM-VLA style).

TEAM-VLA (Token Expand-and-Merge) compresses the visual token stream
without training: tokens near high-attention regions are kept (and in the
paper, expanded with local detail), while redundant tokens are merged.
This plugin implements the model-neutral core of that idea:

* the most salient tokens (attention scores when the backend provides
  them, otherwise the leading tokens) are always protected;
* among the rest, the most cosine-similar pairs are greedily averaged
  until ``merge_ratio`` of the stream has been merged away.

A backend with an engine-native merge kernel can inject ``merger``;
otherwise the built-in NumPy merger runs, which is already training-free
and checkpoint-agnostic.  Non-array token streams pass through unchanged
so enabling the plugin never breaks an opaque token format.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TokenMergeConfig:
    enabled: bool = False
    merge_ratio: float = 0.0
    salient_ratio: float = 0.1

    @classmethod
    def from_mapping(cls, value: Any) -> "TokenMergeConfig":
        if isinstance(value, bool):
            return cls(enabled=value)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("token_merge must be a boolean or JSON object")
        return cls(
            enabled=bool(value.get("enabled", True)),
            merge_ratio=float(value.get("merge_ratio", 0.0)),
            salient_ratio=float(value.get("salient_ratio", 0.1)),
        )

    def __post_init__(self) -> None:
        if not 0.0 <= self.merge_ratio < 1.0:
            raise ValueError("token_merge.merge_ratio must be in [0, 1)")
        if not 0.0 <= self.salient_ratio <= 1.0:
            raise ValueError("token_merge.salient_ratio must be in [0, 1]")


def _cosine_merge(tokens: np.ndarray, keep: int, protected: int) -> np.ndarray:
    """Greedily average the most similar token pairs beyond ``protected``."""

    merged = tokens.astype(np.float32, copy=True)
    while len(merged) > max(keep, protected):
        body = merged[protected:]
        norm = np.linalg.norm(body, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        similarity = (body / norm) @ (body / norm).T
        np.fill_diagonal(similarity, -np.inf)
        first, second = np.unravel_index(int(np.argmax(similarity)), similarity.shape)
        averaged = (body[first] + body[second]) / 2.0
        merged = np.concatenate(
            [merged[:protected], np.delete(body, (first, second), axis=0),
             averaged[None]],
            axis=0,
        )
    return merged


class TokenMergePlugin:
    """Compress a token stream by merging redundant tokens."""

    def __init__(
        self,
        merger: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
        config: TokenMergeConfig | None = None,
    ) -> None:
        self.config = config or TokenMergeConfig()
        self.merger = merger if self.config.enabled else None
        self.mode = "native" if self.config.enabled else "disabled"
        self._lock = threading.Lock()
        self._tokens_in = 0
        self._tokens_out = 0

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        merger: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
    ) -> "TokenMergePlugin":
        return cls(merger=merger, config=TokenMergeConfig.from_mapping(
            config.get("token_merge")
        ))

    def compress(
        self, tokens: Any, saliency: np.ndarray | None = None
    ) -> np.ndarray | Any:
        """Return the compressed token stream.

        ``saliency`` is one attention-style score per token; the top
        ``salient_ratio`` tokens are protected from merging.  Streams that
        are not 2D arrays pass through unchanged.
        """

        array = np.asarray(tokens) if isinstance(tokens, np.ndarray) else None
        if (
            not self.config.enabled
            or self.config.merge_ratio <= 0.0
            or array is None
            or array.ndim != 2
        ):
            return tokens
        count = len(array)
        keep = max(1, int(np.ceil(count * (1.0 - self.config.merge_ratio))))
        protected = min(count, int(round(count * self.config.salient_ratio)))
        if saliency is not None and protected > 0:
            order = np.argsort(np.asarray(saliency, dtype=np.float32))[::-1]
            array = array[order]
        merger = self.merger or _cosine_merge
        merged = merger(array, keep, protected)
        with self._lock:
            self._tokens_in += count
            self._tokens_out += len(merged)
        return merged

    def reset(self) -> None:
        with self._lock:
            self._tokens_in = 0
            self._tokens_out = 0

    def close(self) -> None:
        self.reset()

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            tokens_in = self._tokens_in
            tokens_out = self._tokens_out
        return {
            "enabled": bool(self.config.enabled),
            "mode": self.mode,
            "merge_ratio": self.config.merge_ratio,
            "salient_ratio": self.config.salient_ratio,
            "native_merger": self.merger is not None,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "compression": (tokens_out / tokens_in) if tokens_in else 1.0,
        }


__all__ = ["TokenMergeConfig", "TokenMergePlugin"]
