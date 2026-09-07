"""Dependency-free RoboTwin-compatible environment for the quickstart.

The real RoboTwin integration is loaded through ``--env-factory``.  This
small deterministic environment makes a fresh checkout executable before
SAPIEN/RoboTwin is installed and is also useful for CI and protocol smoke
tests.
"""

from __future__ import annotations

import numpy as np


class DemoRoboTwinEnvironment:
    def __init__(self, episode_steps: int = 8, image_size: int = 64) -> None:
        if episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.episode_steps = int(episode_steps)
        self.image_size = int(image_size)
        self.steps = 0
        self.raw_state = np.zeros(18, dtype=np.float32)
        self.actions: list[np.ndarray] = []

    def reset(self):
        self.steps = 0
        self.actions.clear()
        self.raw_state.fill(0)
        return self.get_obs()

    def is_episode_end(self) -> bool:
        return self.steps >= self.episode_steps

    def get_obs(self) -> dict:
        # Distinct colors make camera ordering mistakes visible in a recording.
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image[..., 0] = min(255, self.steps * 8)
        return {
            "instruction": "move safely in the demo scene",
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image.copy()},
                "right_camera": {"rgb": image.copy()},
            },
            "joint_action": {"vector": self.raw_state.copy()},
        }

    def take_action(self, action):
        value = np.asarray(action, dtype=np.float32).reshape(-1)
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError("demo RoboTwin action must be finite 16D qpos")
        self.actions.append(value.copy())
        self.raw_state[:16] = value
        self.steps += 1

    def close(self) -> None:
        return None


def create_demo_env(**kwargs) -> DemoRoboTwinEnvironment:
    """Factory target usable from the CLI: ``...demo_env:create_demo_env``."""
    return DemoRoboTwinEnvironment(**kwargs)
