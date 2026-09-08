"""Runner contracts and compatibility wrappers.

A runner turns a scheduled request batch into one result per request.  Model
implementations should eventually provide a staged runner; the sequential
wrapper keeps legacy ``ModelBackend`` objects working while those migrations
land.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .contracts import BackendResult
from .request import InferenceRequest
from .scheduler import ScheduledBatch


@runtime_checkable
class ModelRunner(Protocol):
    def execute(self, batch: ScheduledBatch) -> list[BackendResult]: ...


class SequentialModelRunner:
    """Compatibility runner for any existing non-batching ``ModelBackend``."""

    def __init__(self, backend) -> None:
        self.backend = backend

    def execute(self, batch: ScheduledBatch) -> list[BackendResult]:
        return [self.backend.infer(request.as_backend_request()) for request in batch.requests]


__all__ = ["ModelRunner", "SequentialModelRunner"]
