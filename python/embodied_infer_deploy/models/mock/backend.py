from __future__ import annotations

from typing import Any

import numpy as np

from ...core import BackendResult, ModelSpec


class MockBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        model_state_dim = int(config.get("model_state_dim", config.get("state_dim", 14)))
        model_action_dim = int(config.get("model_action_dim", config.get("action_dim", 14)))
        self._spec = ModelSpec(
            name=str(config.get("model_name", "mock-vla")),
            backend="mock",
            raw_state_dim=int(config.get("raw_state_dim", model_state_dim)),
            model_state_dim=model_state_dim,
            raw_action_dim=int(config.get("raw_action_dim", model_action_dim)),
            model_action_dim=model_action_dim,
            action_horizon=int(config.get("action_horizon", 8)),
            camera_order=tuple(config.get(
                "camera_order", ["head", "left_wrist", "right_wrist"]
            )),
            control_period_ns=int(config.get("control_period_ns", 20_000_000)),
        )

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def metadata(self) -> dict[str, Any]:
        return self.spec.metadata()

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        target = np.resize(np.tanh(request["state"]), self.spec.model_action_dim)
        actions = np.repeat(target[None], self.spec.action_horizon, axis=0)
        return BackendResult(
            actions=self.spec.validate_actions(actions),
            representation=self.spec.action_representation,
            control_period_ns=self.spec.control_period_ns,
        )

    def reset(self) -> None:
        return None


def create_backend(config: dict[str, Any]) -> MockBackend:
    return MockBackend(config)
