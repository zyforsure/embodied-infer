"""Hierarchical dual-system action generation (GR00T / SmolVLA style).

System2 (the VLM) performs slow high-level reasoning: task planning, scene
understanding, subgoal conditioning.  System1 is a fast action head --
typically flow matching or regression -- that executes real-time chunks
conditioned on the System2 latent.  Thinking and acting stay decoupled:
System2 replans only when the instruction changes or
``refresh_interval_s`` elapses, while System1 runs every control step.

The backend injects:

- ``plan_fn(context) -> Any``: the System2 latent for the current context.
- a fast :class:`ActionHead` (usually :class:`FlowMatchingHead` or
  :class:`ParallelRegressionHead`) whose callables read the injected
  ``context["system2_latent"]``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import ModelSpec
from .base import ActionContext, ActionHead, ActionHeadUnavailableError


@dataclass(frozen=True)
class DualSystemConfig:
    refresh_interval_s: float = 1.0
    instruction_key: str = "instruction"

    @classmethod
    def from_mapping(cls, value: Any) -> "DualSystemConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("dual_system head config must be a JSON object")
        return cls(
            refresh_interval_s=float(value.get("refresh_interval_s", 1.0)),
            instruction_key=str(value.get("instruction_key", "instruction")),
        )

    def __post_init__(self) -> None:
        if self.refresh_interval_s < 0:
            raise ValueError("dual_system.refresh_interval_s must be non-negative")


class DualSystemHead:
    """System2 planner + System1 fast head with latent caching."""

    def __init__(
        self,
        plan_fn: Callable[[ActionContext], Any] | None = None,
        fast_head: ActionHead | None = None,
        config: DualSystemConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or DualSystemConfig()
        self.plan_fn = plan_fn
        self.fast_head = fast_head
        self._clock = clock
        self._latent: Any = None
        self._latent_instruction: Any = None
        self._latent_time = float("-inf")
        self.system2_calls = 0

    @property
    def paradigm(self) -> str:
        return "dual_system"

    def _needs_replan(self, context: ActionContext) -> bool:
        instruction = context.get(self.config.instruction_key)
        if self.system2_calls == 0:
            return True
        if instruction != self._latent_instruction:
            return True
        elapsed = self._clock() - self._latent_time
        return elapsed >= self.config.refresh_interval_s

    def decode(self, context: ActionContext, spec: ModelSpec) -> np.ndarray:
        if self.plan_fn is None or self.fast_head is None:
            raise ActionHeadUnavailableError(
                "dual_system head requires plan_fn and a fast_head"
            )
        if self._needs_replan(context):
            self._latent = self.plan_fn(context)
            self._latent_instruction = context.get(self.config.instruction_key)
            self._latent_time = self._clock()
            self.system2_calls += 1
        conditioned = dict(context)
        conditioned["system2_latent"] = self._latent
        return self.fast_head.decode(conditioned, spec)

    def reset(self) -> None:
        self._latent = None
        self._latent_instruction = None
        self._latent_time = float("-inf")
        self.system2_calls = 0
        if self.fast_head is not None:
            self.fast_head.reset()

    def metadata(self) -> dict[str, Any]:
        fast = self.fast_head.metadata() if self.fast_head is not None else {}
        return {
            "paradigm": self.paradigm,
            "refresh_interval_s": self.config.refresh_interval_s,
            "system2_calls": self.system2_calls,
            "fast_head": fast,
        }


__all__ = ["DualSystemConfig", "DualSystemHead"]
