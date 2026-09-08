"""Batching mock runner used to exercise the new scheduler without a device."""

from __future__ import annotations

from typing import Any

from ...core import BackendResult, ModelRunner, ScheduledBatch
from .backend import MockBackend


class MockModelRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.backend = MockBackend(config)
        self.batch_sizes: list[int] = []

    def execute(self, batch: ScheduledBatch) -> list[BackendResult]:
        self.batch_sizes.append(len(batch))
        return [
            self.backend.infer(request.as_backend_request())
            for request in batch.requests
        ]


def create_runner(config: dict[str, Any]) -> MockModelRunner:
    return MockModelRunner(config)


__all__ = ["MockModelRunner", "create_runner"]
