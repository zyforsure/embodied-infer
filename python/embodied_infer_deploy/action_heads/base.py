"""Shared contracts for pluggable VLA action-generation paradigms."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core import ModelSpec

# Opaque conditioning produced by a backend encoder: token ids, embeddings,
# KV handles, or anything the injected head callables understand.  The
# pipeline never inspects the values; only the head callables do.
ActionContext = dict[str, Any]


class ActionHeadUnavailableError(RuntimeError):
    """Raised when a head is used without its backend-native callable."""


@runtime_checkable
class ActionHead(Protocol):
    """One action-generation paradigm shared by all VLA backends."""

    @property
    def paradigm(self) -> str: ...

    def decode(self, context: ActionContext, spec: ModelSpec) -> np.ndarray: ...

    def reset(self) -> None: ...

    def metadata(self) -> dict[str, Any]: ...


def validate_chunk(actions: Any, spec: ModelSpec, *, owner: str) -> np.ndarray:
    """Apply the ModelSpec action contract to one decoded chunk."""

    try:
        return spec.validate_actions(actions)
    except ValueError as exc:
        raise ValueError(f"{owner} produced an invalid action chunk: {exc}") from exc


__all__ = [
    "ActionContext",
    "ActionHead",
    "ActionHeadUnavailableError",
    "validate_chunk",
]
