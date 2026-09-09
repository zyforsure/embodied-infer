"""Raw Dazz/S600 state contract used by RoboTwin and real hardware."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RawRobotContract:
    raw_state_dim: int
    raw_action_dim: int
    model_state_indices: tuple[int, ...]
    left_arm_slice: slice
    left_gripper_slice: slice
    right_arm_slice: slice
    right_gripper_slice: slice

    _SLICE_FIELDS = (
        "left_arm_slice",
        "left_gripper_slice",
        "right_arm_slice",
        "right_gripper_slice",
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RawRobotContract":
        """Build a contract from a JSON-style config mapping.

        Slices are two-element ``[start, stop]`` lists so a new embodiment
        can be described entirely in configuration and registered through
        :class:`RobotRegistry` without new Python code.
        """

        if not isinstance(value, Mapping):
            raise TypeError("robot contract must be a JSON object")
        kwargs: dict[str, Any] = {}
        for field in cls._SLICE_FIELDS:
            raw = value.get(field)
            if raw is None:
                raise ValueError(f"robot contract requires {field}")
            kwargs[field] = _as_slice(raw, field)
        contract = cls(
            raw_state_dim=int(value["raw_state_dim"]),
            raw_action_dim=int(value["raw_action_dim"]),
            model_state_indices=tuple(
                int(index) for index in value["model_state_indices"]
            ),
            **kwargs,
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.raw_state_dim <= 0 or self.raw_action_dim <= 0:
            raise ValueError("raw state/action dims must be positive")
        if not self.model_state_indices:
            raise ValueError("model_state_indices must be non-empty")
        if len(set(self.model_state_indices)) != len(self.model_state_indices):
            raise ValueError("model_state_indices must be unique")
        for index in self.model_state_indices:
            if not 0 <= index < self.raw_action_dim:
                raise ValueError(
                    f"model_state_indices out of range: {index}"
                )
        for field in self._SLICE_FIELDS:
            item = getattr(self, field)
            start = item.start or 0
            stop = item.stop or 0
            if not (0 <= start < stop <= self.raw_action_dim):
                raise ValueError(
                    f"{field} must satisfy 0 <= start < stop <= raw_action_dim"
                )

    @property
    def model_dim(self) -> int:
        return len(self.model_state_indices)

    @property
    def held_indices(self) -> tuple[int, ...]:
        controlled = set(self.model_state_indices)
        return tuple(
            index for index in range(self.raw_action_dim) if index not in controlled
        )

    @property
    def command_action_dim(self) -> int:
        slices = (
            self.left_arm_slice,
            self.left_gripper_slice,
            self.right_arm_slice,
            self.right_gripper_slice,
        )
        return sum((item.stop or 0) - (item.start or 0) for item in slices)

    def validate_raw_state(self, value: Any) -> np.ndarray:
        state = np.asarray(value, dtype=np.float32).reshape(-1)
        if state.shape != (self.raw_state_dim,) or not np.isfinite(state).all():
            raise ValueError(
                f"raw state must have {self.raw_state_dim} finite values, got {state.shape}"
            )
        return np.ascontiguousarray(state)

    def state_to_model(self, value: Any) -> np.ndarray:
        state = self.validate_raw_state(value)
        return np.ascontiguousarray(state[list(self.model_state_indices)])

    def actions_to_raw(self, actions: Any, current_state: Any) -> np.ndarray:
        model_actions = np.asarray(actions, dtype=np.float32)
        if (model_actions.ndim != 2 or model_actions.shape[1] != self.model_dim or
                not np.isfinite(model_actions).all()):
            raise ValueError(
                f"model actions must have finite shape [T,{self.model_dim}]"
            )
        current = self.validate_raw_state(current_state)
        raw = np.repeat(current[None, :], model_actions.shape[0], axis=0)
        raw[:, list(self.model_state_indices)] = model_actions
        return np.ascontiguousarray(raw)

    def action_dict(self, action: Any) -> dict[str, np.ndarray]:
        value = np.asarray(action, dtype=np.float32).reshape(self.raw_action_dim)
        return {
            "raw_action": value.copy(),
            "left_arm_joint_state": value[self.left_arm_slice],
            "left_ee_joint_state": value[self.left_gripper_slice],
            "right_arm_joint_state": value[self.right_arm_slice],
            "right_ee_joint_state": value[self.right_gripper_slice],
        }


# Mapping fixed by the existing TurboVLA calibration pipeline. The full target
# is 18D, while the currently named RoboTwin qpos command fields cover 16D.
# Indices 16 and 17 stay opaque until the device schema names them explicitly.
DAZZ_S600_CONTRACT = RawRobotContract(
    raw_state_dim=18,
    raw_action_dim=18,
    model_state_indices=(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 7, 15),
    left_arm_slice=slice(0, 7),
    left_gripper_slice=slice(7, 8),
    right_arm_slice=slice(8, 15),
    right_gripper_slice=slice(15, 16),
)


def _as_slice(value: Any, field: str) -> slice:
    if isinstance(value, slice):
        return value
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f"{field} must be a [start, stop] pair")
    return slice(int(value[0]), int(value[1]))
