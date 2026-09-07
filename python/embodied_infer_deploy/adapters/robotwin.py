"""Compatibility wrapper for the historical RoboTwin adapter path."""

import numpy as np

from ..robots import DAZZ_S600_CONTRACT
from ..simulators.robotwin.adapter import RoboTwinAdapter

RAW_STATE_DIM = DAZZ_S600_CONTRACT.raw_state_dim
MODEL_STATE_DIM = DAZZ_S600_CONTRACT.model_dim
MODEL_STATE_INDEX = np.asarray(
    DAZZ_S600_CONTRACT.model_state_indices, dtype=np.int64
)

class RobotTwinAdapter(RoboTwinAdapter):
    """Backwards-compatible facade with the original static helper API."""

    @staticmethod
    def state_to_model_order(state):
        return DAZZ_S600_CONTRACT.state_to_model(state)

    @staticmethod
    def actions_to_raw_order(actions, current_state):
        return DAZZ_S600_CONTRACT.actions_to_raw(actions, current_state)

    @staticmethod
    def action_dict(action):
        return DAZZ_S600_CONTRACT.action_dict(action)

__all__ = [
    "MODEL_STATE_DIM",
    "MODEL_STATE_INDEX",
    "RAW_STATE_DIM",
    "RoboTwinAdapter",
    "RobotTwinAdapter",
]
