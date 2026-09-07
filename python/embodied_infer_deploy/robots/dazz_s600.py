"""Raw Dazz/S600 state contract used by RoboTwin and real hardware."""

from __future__ import annotations

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
