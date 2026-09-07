"""RoboTwin observation and action ordering adapter."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

RAW_STATE_DIM = 18
MODEL_STATE_DIM = 14

# This is the mapping used to build the existing TurboVLA calibration set.
# Unmapped raw coordinates are held at their current values on action decode.
MODEL_STATE_INDEX = np.asarray(
    [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 7, 15], dtype=np.int64
)


def _camera_value(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("rgb", value.get("color"))
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"camera image must be rank 3, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"RoboTwin camera must be HWC RGB, got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.uint8)


class RobotTwinAdapter:
    camera_aliases = {
        "head": ("head_camera", "cam_head", "cam_high"),
        "left_wrist": ("left_camera", "cam_left_wrist", "left_wrist"),
        "right_wrist": ("right_camera", "cam_right_wrist", "right_wrist"),
    }

    def __init__(self, client, *, default_instruction: str = "") -> None:
        self.client = client
        self.default_instruction = default_instruction

    @staticmethod
    def state_to_model_order(state: Any) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (RAW_STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(
                f"RoboTwin raw state must have {RAW_STATE_DIM} finite values, "
                f"got {state.shape}"
            )
        return np.ascontiguousarray(state[MODEL_STATE_INDEX], dtype=np.float32)

    @staticmethod
    def actions_to_raw_order(actions: Any, current_state: Any) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if (actions.ndim != 2 or actions.shape[1] != MODEL_STATE_DIM or
                not np.isfinite(actions).all()):
            raise ValueError(
                f"model actions must have finite shape [T,{MODEL_STATE_DIM}], "
                f"got {actions.shape}"
            )
        current = np.asarray(current_state, dtype=np.float32).reshape(-1)
        if current.shape != (RAW_STATE_DIM,) or not np.isfinite(current).all():
            raise ValueError(
                f"current raw state must have {RAW_STATE_DIM} finite values"
            )
        raw = np.repeat(current[None, :], actions.shape[0], axis=0)
        raw[:, MODEL_STATE_INDEX] = actions
        return np.ascontiguousarray(raw, dtype=np.float32)

    def extract_images(self, observation: dict[str, Any]) -> list[tuple[str, np.ndarray, int]]:
        camera_root = observation.get("observation", observation).get(
            "vision", observation.get("observation", observation)
        )
        now = time.time_ns()
        images = []
        for canonical, aliases in self.camera_aliases.items():
            found = None
            for alias in aliases:
                if alias in camera_root:
                    found = camera_root[alias]
                    break
            if found is None:
                raise KeyError(f"missing RoboTwin camera; tried {aliases}")
            images.append((canonical, _camera_value(found), now))
        return images

    def extract_state(self, observation: dict[str, Any]) -> np.ndarray:
        if "joint_action" in observation:
            state = observation["joint_action"].get("vector")
        else:
            state = observation.get("state")
        if state is None:
            raise KeyError("RoboTwin observation has no state or joint_action.vector")
        raw_state = np.asarray(state, dtype=np.float32).reshape(-1)
        self.state_to_model_order(raw_state)
        return raw_state

    def infer(
        self,
        observation: dict[str, Any],
        *,
        instruction: str | None = None,
        control_step: int = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        instruction = instruction or observation.get("instruction") or \
            observation.get("language") or self.default_instruction
        raw_state = self.extract_state(observation)
        response = self.client.infer(
            control_step=control_step,
            timestamp_ns=time.time_ns(),
            instruction=str(instruction),
            images=self.extract_images(observation),
            state=self.state_to_model_order(raw_state),
            timeout=timeout,
        )
        response["raw_actions"] = self.actions_to_raw_order(
            response["actions"], raw_state
        )
        return response

    @staticmethod
    def action_dict(action: Any) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(RAW_STATE_DIM)
        return {
            "left_arm_joint_state": action[:7],
            "left_ee_joint_state": action[7:8],
            "right_arm_joint_state": action[8:15],
            "right_ee_joint_state": action[15:16],
        }
