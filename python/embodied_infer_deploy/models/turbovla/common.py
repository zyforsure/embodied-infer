from __future__ import annotations

from typing import Any

from ...core import ModelSpec
from ...robots import DAZZ_S600_CONTRACT


def turbovla_spec(
    backend: str,
    config: dict[str, Any],
    *,
    default_period_ns: int,
) -> ModelSpec:
    return ModelSpec(
        name=str(config.get("model_name", "TurboVLA-RoboTwin")),
        backend=backend,
        raw_state_dim=DAZZ_S600_CONTRACT.raw_state_dim,
        model_state_dim=14,
        raw_action_dim=DAZZ_S600_CONTRACT.raw_action_dim,
        model_action_dim=14,
        action_horizon=50,
        camera_order=tuple(config.get(
            "camera_order", ["head", "left_wrist", "right_wrist"]
        )),
        action_representation="absolute",
        control_period_ns=int(config.get("control_period_ns", default_period_ns)),
        extras={
            "model_input_shape": [1, 14],
            "model_output_shape": [1, 50, 14],
            "robot_command_dim": DAZZ_S600_CONTRACT.command_action_dim,
        },
    )
