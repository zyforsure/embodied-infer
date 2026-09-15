"""RoboTwin observation/action adapter at the 18D robot boundary."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...robots import DAZZ_S600_CONTRACT, RawRobotContract
from ...images import as_hwc_uint8

# Default RoboTwin camera naming; other simulators inject their own alias
# map through the constructor instead of editing this class.
ROBOTWIN_CAMERA_ALIASES = {
    "head": ("head_camera", "cam_head", "cam_high"),
    "left_wrist": ("left_camera", "cam_left_wrist", "left_wrist"),
    "right_wrist": ("right_camera", "cam_right_wrist", "right_wrist"),
}


def _camera_value(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("rgb", value.get("color"))
    return as_hwc_uint8(value)


class RoboTwinAdapter:
    def __init__(
        self,
        client,
        *,
        default_instruction: str = "",
        robot_contract: RawRobotContract = DAZZ_S600_CONTRACT,
        camera_aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.client = client
        self.default_instruction = default_instruction
        self.robot_contract = robot_contract
        aliases = camera_aliases if camera_aliases is not None else ROBOTWIN_CAMERA_ALIASES
        self.camera_aliases = {
            str(name): tuple(candidates) for name, candidates in aliases.items()
        }

    def state_to_model_order(self, state: Any) -> np.ndarray:
        return self.robot_contract.state_to_model(state)

    def actions_to_raw_order(self, actions: Any, current_state: Any) -> np.ndarray:
        return self.robot_contract.actions_to_raw(actions, current_state)

    def extract_images(
        self, observation: dict[str, Any]
    ) -> list[tuple[str, np.ndarray, int]]:
        observation_root = observation.get("observation", observation)
        camera_root = observation_root.get("vision", observation_root)
        now = time.time_ns()
        images = []
        for canonical, aliases in self.camera_aliases.items():
            found = next(
                (camera_root[alias] for alias in aliases if alias in camera_root), None
            )
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
        return self.robot_contract.validate_raw_state(state)

    def infer(
        self,
        observation: dict[str, Any],
        *,
        instruction: str | None = None,
        control_step: int = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        instruction = (
            instruction
            or observation.get("instruction")
            or observation.get("language")
            or self.default_instruction
        )
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

    def action_dict(self, action: Any) -> dict[str, np.ndarray]:
        return self.robot_contract.action_dict(action)
