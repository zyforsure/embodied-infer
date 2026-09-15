"""Protocol every simulator observation adapter satisfies.

An adapter bridges one simulator's observation schema to the canonical
embodied-infer request: named HWC uint8 cameras, a raw robot state, and an
instruction.  New simulators (ManiSkill, LIBERO, ...) implement this
protocol -- typically by subclassing ``RoboTwinAdapter`` with a different
``camera_aliases`` map -- and plug into the same policy/runner/CLI stack.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ObservationAdapter(Protocol):
    camera_aliases: dict[str, tuple[str, ...]]

    def extract_images(
        self, observation: dict[str, Any]
    ) -> list[tuple[str, np.ndarray, int]]: ...

    def extract_state(self, observation: dict[str, Any]) -> np.ndarray: ...

    def infer(
        self,
        observation: dict[str, Any],
        *,
        instruction: str | None = None,
        control_step: int = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]: ...

    def action_dict(self, action: Any) -> dict[str, np.ndarray]: ...


__all__ = ["ObservationAdapter"]
