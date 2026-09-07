from __future__ import annotations

from typing import Any

import numpy as np

from .base import BackendResult


class MockBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.state_dim = int(config.get("state_dim", 14))
        self.action_dim = int(config.get("action_dim", 14))
        self.raw_state_dim = int(config.get("raw_state_dim", self.state_dim))
        self.raw_action_dim = int(config.get("raw_action_dim", self.action_dim))
        self.horizon = int(config.get("action_horizon", 8))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "mock",
            "model": "mock-vla",
            "raw_state_dim": self.raw_state_dim,
            "model_state_dim": self.state_dim,
            "raw_action_dim": self.raw_action_dim,
            "model_action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.horizon,
            "action_representation": "absolute",
            "camera_order": ["head", "left_wrist", "right_wrist"],
            "control_period_ns": 20_000_000,
        }

    def infer(self, request: dict[str, Any]) -> BackendResult:
        state = np.asarray(request["state"], dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"expected state [{self.state_dim}], got {state.shape}")
        target = np.resize(np.tanh(state), self.action_dim)
        actions = np.repeat(target[None], self.horizon, axis=0)
        return BackendResult(actions=actions, control_period_ns=20_000_000)

    def reset(self) -> None:
        return None


def create_backend(config: dict[str, Any]) -> MockBackend:
    return MockBackend(config)
