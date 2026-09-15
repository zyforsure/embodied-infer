"""Flow-matching continuous action generation (Pi0 / Pi0.5 / starVLA QwenFM).

A flow-matching action expert models the *distribution* over continuous
action chunks.  Generation integrates the learned velocity field from pure
noise to a sample:

    x_0 ~ N(0, noise_scale^2)
    x_{t+dt} = x_t + dt * velocity_fn(context, x_t, t),  t: 0 -> 1

Because the head samples a distribution it captures action uncertainty and
produces diverse chunks; ``seed`` makes the sampling deterministic for
reproducible serving and tests.

The backend injects one native callable:

- ``velocity_fn(context, actions, t) -> np.ndarray``: the action expert's
  velocity prediction at flow time ``t`` for the noisy chunk ``actions``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import ModelSpec
from .base import ActionContext, ActionHeadUnavailableError, validate_chunk


@dataclass(frozen=True)
class FlowMatchingConfig:
    num_steps: int = 10
    noise_scale: float = 1.0
    seed: int | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "FlowMatchingConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("flow_matching head config must be a JSON object")
        seed = value.get("seed")
        return cls(
            num_steps=int(value.get("num_steps", 10)),
            noise_scale=float(value.get("noise_scale", 1.0)),
            seed=None if seed is None else int(seed),
        )

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("flow_matching.num_steps must be positive")
        if self.noise_scale <= 0:
            raise ValueError("flow_matching.noise_scale must be positive")


class FlowMatchingHead:
    """Pi0 style Euler integration of a learned velocity field."""

    def __init__(
        self,
        velocity_fn: Callable[[ActionContext, np.ndarray, float], Any] | None = None,
        config: FlowMatchingConfig | None = None,
    ) -> None:
        self.config = config or FlowMatchingConfig()
        self.velocity_fn = velocity_fn
        self._rng = np.random.default_rng(self.config.seed)
        self.last_num_steps = 0

    @property
    def paradigm(self) -> str:
        return "flow_matching"

    def decode(self, context: ActionContext, spec: ModelSpec) -> np.ndarray:
        if self.velocity_fn is None:
            raise ActionHeadUnavailableError(
                "flow_matching head requires velocity_fn"
            )
        shape = (spec.action_horizon, spec.model_action_dim)
        actions = self._rng.normal(0.0, self.config.noise_scale, shape).astype(np.float32)
        dt = 1.0 / self.config.num_steps
        for index in range(self.config.num_steps):
            t = index * dt
            velocity = np.asarray(
                self.velocity_fn(context, actions, t), dtype=np.float32
            )
            if velocity.shape != shape:
                raise ValueError(
                    f"flow_matching velocity must have shape {shape}, "
                    f"got {velocity.shape}"
                )
            actions = actions + dt * velocity
        self.last_num_steps = self.config.num_steps
        return validate_chunk(actions, spec, owner="flow_matching head")

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.config.seed)
        self.last_num_steps = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "paradigm": self.paradigm,
            "num_steps": self.config.num_steps,
            "noise_scale": self.config.noise_scale,
            "deterministic": self.config.seed is not None,
        }


__all__ = ["FlowMatchingConfig", "FlowMatchingHead"]
