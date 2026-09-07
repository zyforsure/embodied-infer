"""Lazy model registry inspired by vLLM's architecture registry."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from ..core import ModelBackend

ModelFactory = Callable[[dict[str, Any]], ModelBackend]
FactoryTarget = ModelFactory | str
ENTRY_POINT_GROUP = "embodied_infer.models"
_VALID_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ModelRegistry:
    def __init__(self) -> None:
        self._targets: dict[str, FactoryTarget] = {}
        self._entry_points_loaded = False

    def register(self, name: str, target: FactoryTarget, *, replace: bool = False) -> None:
        if not _VALID_NAME.fullmatch(name):
            raise ValueError(f"invalid model registry name {name!r}")
        if not callable(target) and ":" not in target:
            raise ValueError("lazy factory target must use module:callable syntax")
        if name in self._targets and not replace:
            raise ValueError(f"model {name!r} is already registered")
        self._targets[name] = target

    def names(self) -> tuple[str, ...]:
        self.load_entry_points()
        return tuple(sorted(self._targets))

    def resolve(self, name: str) -> ModelFactory:
        self.load_entry_points()
        try:
            target = self._targets[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown model {name!r}; available: {', '.join(self.names())}"
            ) from exc
        if callable(target):
            return target
        module_name, attribute = target.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
        if not callable(factory):
            raise TypeError(f"model factory {target!r} is not callable")
        self._targets[name] = factory
        return factory

    def create(self, name: str, config: dict[str, Any]) -> ModelBackend:
        backend = self.resolve(name)(config)
        if not isinstance(backend, ModelBackend):
            raise TypeError(f"model factory {name!r} returned an invalid backend")
        return backend

    def load_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for item in entry_points(group=ENTRY_POINT_GROUP):
            self.register(item.name, item.load())


model_registry = ModelRegistry()
model_registry.register(
    "mock", "embodied_infer_deploy.models.mock.backend:create_backend"
)
model_registry.register(
    "turbovla-tensorrt",
    "embodied_infer_deploy.models.turbovla.tensorrt:create_backend",
)
model_registry.register(
    "turbovla-s600-hbm",
    "embodied_infer_deploy.models.turbovla.s600_hbm:create_backend",
)
model_registry.register(
    "turbovla-s600-remote",
    "embodied_infer_deploy.models.turbovla.s600_remote:create_backend",
)
model_registry.register(
    "turbovla-s100-remote",
    "embodied_infer_deploy.models.turbovla.s100_remote:create_backend",
)


def create_model(name: str, config: dict[str, Any]) -> ModelBackend:
    return model_registry.create(name, config)
