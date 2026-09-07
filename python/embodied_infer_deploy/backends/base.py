from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class BackendResult:
    actions: np.ndarray
    representation: str = "absolute"
    control_period_ns: int = 0
    timing: dict[str, float] = field(default_factory=dict)


class Backend(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def infer(self, request: dict[str, Any]) -> BackendResult: ...

    def reset(self) -> None: ...
