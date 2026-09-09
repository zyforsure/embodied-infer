"""Small dependency-free RoboTwin episode loop used by plugins and tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ...robots import DAZZ_S600_CONTRACT, RawRobotContract


def flatten_qpos_action(
    action: Any,
    *,
    contract: RawRobotContract = DAZZ_S600_CONTRACT,
) -> np.ndarray:
    """Convert an XPolicyLab-style action dictionary to RoboTwin qpos order."""
    if not isinstance(action, Mapping):
        return np.asarray(action, dtype=np.float32).reshape(-1)
    required = contract.qpos_fields
    missing = [key for key in required if key not in action]
    if missing:
        raise KeyError(f"RoboTwin qpos action is missing fields: {missing}")
    return np.concatenate([
        np.asarray(action[key], dtype=np.float32).reshape(-1) for key in required
    ])


def run_episode(
    environment: Any,
    policy: Any,
    *,
    max_steps: int | None = None,
    on_step: Callable[[int, Any], None] | None = None,
    action_transform: Callable[[Any], Any] | None = None,
) -> int:
    """Run one RoboTwin-style episode and return executed environment steps."""
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    environment.reset()
    policy.reset()
    executed = 0
    while not environment.is_episode_end():
        policy.update_obs(environment.get_obs())
        actions = policy.get_action()
        if not actions:
            raise RuntimeError("policy returned an empty action chunk")
        for action in actions:
            environment_action = (
                action_transform(action) if action_transform is not None else action
            )
            environment.take_action(environment_action)
            executed += 1
            if on_step is not None:
                on_step(executed, environment_action)
            if environment.is_episode_end() or (
                max_steps is not None and executed >= max_steps
            ):
                return executed
    return executed
