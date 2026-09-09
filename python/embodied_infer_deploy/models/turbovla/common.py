from __future__ import annotations

from typing import Any

from ...core import ModelSpec
from ...robots import DEFAULT_ROBOT, get_robot_contract
from ...plugins import VisionBatchPlugin


def turbovla_spec(
    backend: str,
    config: dict[str, Any],
    *,
    default_period_ns: int,
) -> ModelSpec:
    vision_batching = VisionBatchPlugin.from_config(config).metadata()
    robot = get_robot_contract(str(config.get("robot", DEFAULT_ROBOT)))
    return ModelSpec(
        name=str(config.get("model_name", "TurboVLA-RoboTwin")),
        backend=backend,
        raw_state_dim=robot.raw_state_dim,
        model_state_dim=14,
        raw_action_dim=robot.raw_action_dim,
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
            "robot_command_dim": robot.command_action_dim,
            "vision_batching": vision_batching,
        },
    )
