"""Autoregressive discrete action decoding (PI0-FAST / starVLA QwenFast).

The backbone emits one discrete action token per step; a FAST-style
detokenizer turns the finished token sequence into a continuous action
chunk.  Roughly 30-60 tokens cover a full chunk at high fidelity, so the
loop is bounded by ``max_tokens`` and terminates early on EOS.

The backend injects two native callables:

- ``step_fn(context, tokens) -> int | None``: run one decode step and return
  the next token id, or ``None`` to signal EOS.
- ``detokenize_fn(tokens, spec) -> np.ndarray``: map the token sequence to
  an ``[action_horizon, model_action_dim]`` continuous chunk.

Without both callables the head raises :class:`ActionHeadUnavailableError`
instead of fabricating actions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import ModelSpec
from .base import ActionContext, ActionHeadUnavailableError, validate_chunk


@dataclass(frozen=True)
class AutoregressiveConfig:
    max_tokens: int = 60
    eos_token: int | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "AutoregressiveConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("autoregressive head config must be a JSON object")
        eos = value.get("eos_token")
        return cls(
            max_tokens=int(value.get("max_tokens", 60)),
            eos_token=None if eos is None else int(eos),
        )

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("autoregressive.max_tokens must be positive")


class AutoregressiveTokenHead:
    """PI0-FAST style token-by-token action generation."""

    def __init__(
        self,
        step_fn: Callable[[ActionContext, list[int]], int | None] | None = None,
        detokenize_fn: Callable[[list[int], ModelSpec], Any] | None = None,
        config: AutoregressiveConfig | None = None,
    ) -> None:
        self.config = config or AutoregressiveConfig()
        self.step_fn = step_fn
        self.detokenize_fn = detokenize_fn
        self.last_token_count = 0

    @property
    def paradigm(self) -> str:
        return "autoregressive"

    def decode(self, context: ActionContext, spec: ModelSpec) -> np.ndarray:
        if self.step_fn is None or self.detokenize_fn is None:
            raise ActionHeadUnavailableError(
                "autoregressive head requires step_fn and detokenize_fn"
            )
        tokens: list[int] = []
        for _ in range(self.config.max_tokens):
            token = self.step_fn(context, list(tokens))
            if token is None or token == self.config.eos_token:
                break
            tokens.append(int(token))
        self.last_token_count = len(tokens)
        if not tokens:
            raise ValueError("autoregressive head produced no action tokens")
        actions = self.detokenize_fn(tokens, spec)
        return validate_chunk(actions, spec, owner="autoregressive detokenizer")

    def reset(self) -> None:
        self.last_token_count = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "paradigm": self.paradigm,
            "max_tokens": self.config.max_tokens,
            "last_token_count": self.last_token_count,
        }


__all__ = ["AutoregressiveConfig", "AutoregressiveTokenHead"]
