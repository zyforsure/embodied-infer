"""S600 bimanual model-order adapter and conservative action checks."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

import numpy as np

from ..robots import DAZZ_S600_CONTRACT

RAW_STATE_DIM = DAZZ_S600_CONTRACT.raw_state_dim
MODEL_STATE_DIM = DAZZ_S600_CONTRACT.model_dim
MODEL_STATE_INDEX = np.asarray(DAZZ_S600_CONTRACT.model_state_indices)


class S600Adapter:
    """Bridge the 18D robot boundary to the proven 14D TurboVLA graph.

    Model order is [left6, right6, left_gripper, right_gripper]. Coordinates
    absent from the model, including joint 7, are held from current feedback.
    """

    camera_order = ("head", "left_wrist", "right_wrist")

    def __init__(self, client) -> None:
        self.client = client

    @staticmethod
    def validate_raw_state(state: Any) -> np.ndarray:
        try:
            return DAZZ_S600_CONTRACT.validate_raw_state(state)
        except ValueError as exc:
            raise ValueError(
                f"S600 raw state must contain {RAW_STATE_DIM} finite values"
            ) from exc

    @staticmethod
    def state_to_model_order(state: Any) -> np.ndarray:
        return DAZZ_S600_CONTRACT.state_to_model(state)

    @staticmethod
    def actions_to_raw_order(actions: Any, current_state: Any) -> np.ndarray:
        return DAZZ_S600_CONTRACT.actions_to_raw(actions, current_state)

    @staticmethod
    def _image(value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.moveaxis(array, 0, -1)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"S600 image must be HWC RGB, got {array.shape}")
        return np.ascontiguousarray(array, dtype=np.uint8)

    def infer(
        self,
        images: Mapping[str, Any],
        state: Any,
        instruction: str,
        *,
        control_step: int = 0,
        timestamp_ns: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        timestamp_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
        missing = [name for name in self.camera_order if name not in images]
        if missing:
            raise KeyError(f"missing S600 cameras: {missing}")
        raw_state = self.validate_raw_state(state)
        response = self.client.infer(
            control_step=control_step,
            timestamp_ns=timestamp_ns,
            instruction=instruction,
            images=[
                (name, self._image(images[name]), timestamp_ns)
                for name in self.camera_order
            ],
            state=self.state_to_model_order(raw_state),
            timeout=timeout,
        )
        response["raw_actions"] = self.actions_to_raw_order(
            response["actions"], raw_state
        )
        return response

    @staticmethod
    def clamp_first_action(
        action: Any,
        current_state: Any,
        *,
        max_step_deg: float = 1.0,
        enable_gripper: bool = False,
    ) -> np.ndarray:
        """Last client-side check before handing a target to the CAN driver."""
        action = np.asarray(action, dtype=np.float32).reshape(RAW_STATE_DIM).copy()
        current = S600Adapter.validate_raw_state(current_state)
        if not np.isfinite(action).all():
            raise ValueError("S600 action must contain only finite values")
        if max_step_deg <= 0:
            raise ValueError("max_step_deg must be positive")
        delta = math.radians(max_step_deg)
        joint_indices = MODEL_STATE_INDEX[:12]
        gripper_indices = MODEL_STATE_INDEX[12:]
        action[joint_indices] = np.clip(
            action[joint_indices],
            current[joint_indices] - delta,
            current[joint_indices] + delta,
        )
        action[gripper_indices] = (
            (action[gripper_indices] > 0.5).astype(np.float32)
            if enable_gripper else current[gripper_indices]
        )
        held = np.ones(RAW_STATE_DIM, dtype=bool)
        held[MODEL_STATE_INDEX] = False
        action[held] = current[held]
        return action
