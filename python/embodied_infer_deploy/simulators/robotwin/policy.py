"""Framework-neutral remote policy for RoboTwin-style environments."""

from __future__ import annotations

from typing import Any

from .adapter import RoboTwinAdapter


class RemoteRoboTwinPolicy:
    def __init__(
        self,
        adapter: RoboTwinAdapter,
        *,
        exec_horizon: int = 1,
        timeout: float = 10.0,
    ) -> None:
        if exec_horizon <= 0:
            raise ValueError("exec_horizon must be positive")
        self.adapter = adapter
        self.exec_horizon = exec_horizon
        self.timeout = timeout
        self.current: dict[str, Any] | None = None
        self.control_step = 0

    def update_obs(self, observation: dict[str, Any]) -> None:
        self.current = observation

    def reset(self) -> None:
        self.current = None
        self.control_step = 0
        self.adapter.client.reset()

    def get_action(self) -> list[dict[str, Any]]:
        if self.current is None:
            raise RuntimeError("update_obs must be called before get_action")
        response = self.adapter.infer(
            self.current,
            control_step=self.control_step,
            timeout=self.timeout,
        )
        actions = response["raw_actions"][: self.exec_horizon]
        self.control_step += len(actions)
        return [self.adapter.action_dict(action) for action in actions]
