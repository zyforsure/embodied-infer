"""Copy this module into a named model directory and implement its methods."""

from __future__ import annotations

from typing import Any

from ...core import BackendResult, ModelSpec


class TemplateBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        del config
        raise NotImplementedError("configure the model and construct its ModelSpec")

    @property
    def spec(self) -> ModelSpec:
        raise NotImplementedError

    @property
    def metadata(self) -> dict[str, Any]:
        return self.spec.metadata()

    def infer(self, request: dict[str, Any]) -> BackendResult:
        raise NotImplementedError

    def reset(self) -> None:
        return None


def create_backend(config: dict[str, Any]) -> TemplateBackend:
    return TemplateBackend(config)
