"""Request and batch contracts shared by the scheduler and runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceRequest:
    """Immutable request envelope passed between the gateway and runner.

    ``payload`` keeps the legacy backend request shape for now.  As model
    runners migrate, it will become a strongly typed observation instead of a
    wire dict.
    """

    payload: dict[str, Any]
    deadline: float | None = None

    @property
    def request_id(self) -> int:
        return int(self.payload.get("request_id", 0))

    @property
    def timeout_ms(self) -> float:
        return float(self.payload.get("timeout_ms", 300_000))

    def as_backend_request(self) -> dict[str, Any]:
        return dict(self.payload)


__all__ = ["InferenceRequest"]
