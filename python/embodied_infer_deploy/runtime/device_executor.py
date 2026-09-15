"""Device executor boundary used by staged model runners.

This mirrors Embodied.cpp's model ABI in Python: a runner asks a device for a
named stage, and the device is responsible for engine/stream/buffer lifetime.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DeviceExecutor(Protocol):
    def run(
        self, stage: str, feeds: Mapping[str, np.ndarray]
    ) -> Mapping[str, np.ndarray]: ...


class PassthroughDeviceExecutor:
    """Test and migration helper that echoes inputs with a stage prefix."""

    def run(
        self, stage: str, feeds: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {f"{stage}:{name}": np.asarray(value) for name, value in feeds.items()}


__all__ = ["DeviceExecutor", "PassthroughDeviceExecutor"]
