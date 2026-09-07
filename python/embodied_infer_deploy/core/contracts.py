"""Model-neutral contracts shared by serving, robots, and simulators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

ActionRepresentation = Literal["absolute", "delta", "relative"]


@dataclass(frozen=True)
class ModelSpec:
    """Static tensor and semantic contract for one model implementation."""

    name: str
    backend: str
    raw_state_dim: int
    model_state_dim: int
    raw_action_dim: int
    model_action_dim: int
    action_horizon: int
    camera_order: tuple[str, ...]
    action_representation: ActionRepresentation = "absolute"
    control_period_ns: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dimensions = (
            self.raw_state_dim,
            self.model_state_dim,
            self.raw_action_dim,
            self.model_action_dim,
            self.action_horizon,
        )
        if not self.name or not self.backend:
            raise ValueError("model name and backend must be non-empty")
        if any(value <= 0 for value in dimensions):
            raise ValueError("model dimensions and action horizon must be positive")
        if not self.camera_order or len(set(self.camera_order)) != len(self.camera_order):
            raise ValueError("camera_order must contain unique camera names")
        if self.action_representation not in {"absolute", "delta", "relative"}:
            raise ValueError("unsupported action representation")
        if self.control_period_ns < 0:
            raise ValueError("control_period_ns must be non-negative")

    def metadata(self) -> dict[str, Any]:
        return {
            **self.extras,
            "backend": self.backend,
            "model": self.name,
            "raw_state_dim": self.raw_state_dim,
            "model_state_dim": self.model_state_dim,
            "raw_action_dim": self.raw_action_dim,
            "model_action_dim": self.model_action_dim,
            # Compatibility aliases are the model tensors carried on the wire.
            "state_dim": self.model_state_dim,
            "action_dim": self.model_action_dim,
            "action_horizon": self.action_horizon,
            "action_representation": self.action_representation,
            "camera_order": list(self.camera_order),
            "control_period_ns": self.control_period_ns,
        }

    def validate_request(self, request: dict[str, Any]) -> None:
        state = np.asarray(request["state"])
        if state.shape != (self.model_state_dim,):
            raise ValueError(
                f"model state must have shape [{self.model_state_dim}], got {state.shape}"
            )
        missing = [name for name in self.camera_order if name not in request["images"]]
        if missing:
            raise ValueError(f"missing required cameras: {missing}")

    def validate_actions(self, actions: Any) -> np.ndarray:
        value = np.asarray(actions, dtype=np.float32)
        expected = (self.action_horizon, self.model_action_dim)
        if value.shape != expected:
            raise ValueError(f"model actions must have shape {expected}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError("model actions contain NaN or infinity")
        return np.ascontiguousarray(value)


@dataclass
class BackendResult:
    actions: np.ndarray
    representation: ActionRepresentation = "absolute"
    control_period_ns: int = 0
    timing: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class ModelBackend(Protocol):
    @property
    def spec(self) -> ModelSpec: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    def infer(self, request: dict[str, Any]) -> BackendResult: ...

    def reset(self) -> None: ...
