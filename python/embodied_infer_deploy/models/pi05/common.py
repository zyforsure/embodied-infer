"""Shared ModelSpec construction for the Pi0.5 backends.

All four transports (OpenPI websocket, Embodied.cpp ZMQ, HBM, TCP gateway)
expose the same model-side contract; only the extras differ.  Raw
dimensions default from the selected robot contract.
"""

from __future__ import annotations

from typing import Any

from ...core import ModelSpec
from ...robots import DEFAULT_ROBOT, get_robot_contract


def pi05_spec(
    backend: str,
    config: dict[str, Any],
    *,
    default_model_name: str,
    extras: dict[str, Any] | None = None,
    default_period_ns: int = 100_000_000,
) -> ModelSpec:
    robot = get_robot_contract(str(config.get("robot", DEFAULT_ROBOT)))
    merged_extras = {"robot_command_dim": robot.command_action_dim}
    if extras:
        merged_extras.update(extras)
    return ModelSpec(
        name=str(config.get("model_name", default_model_name)),
        backend=backend,
        raw_state_dim=int(config.get("raw_state_dim", robot.raw_state_dim)),
        model_state_dim=int(config.get("model_state_dim", 14)),
        raw_action_dim=int(config.get("raw_action_dim", robot.raw_action_dim)),
        model_action_dim=int(config.get("model_action_dim", 14)),
        action_horizon=int(config.get("action_horizon", 16)),
        camera_order=tuple(config.get(
            "camera_order", ["head", "left_wrist", "right_wrist"]
        )),
        action_representation="absolute",
        control_period_ns=int(config.get("control_period_ns", default_period_ns)),
        extras=merged_extras,
    )


__all__ = ["pi05_spec"]
