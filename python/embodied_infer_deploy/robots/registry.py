"""Lazy robot contract registry, mirroring the model registry.

Built-ins ship as module constants (the company robot is the 18D Dazz
S600); third-party embodiments register either a config mapping or a
``module:callable`` target through the ``embodied_infer.robots`` entry
point group, so deployment code selects a robot by name without importing
vendor libraries it does not use.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from typing import Any

from .dazz_s600 import DAZZ_S600_CONTRACT, RawRobotContract

RobotFactory = Callable[[], RawRobotContract]
FactoryTarget = RawRobotContract | RobotFactory | str
ENTRY_POINT_GROUP = "embodied_infer.robots"
DEFAULT_ROBOT = "dazz-s600"
_VALID_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class RobotRegistry:
    def __init__(self) -> None:
        self._targets: dict[str, FactoryTarget] = {}
        self._entry_points_loaded = False

    def register(
        self, name: str, target: FactoryTarget, *, replace: bool = False
    ) -> None:
        if not _VALID_NAME.fullmatch(name):
            raise ValueError(f"invalid robot registry name {name!r}")
        if (
            not isinstance(target, RawRobotContract)
            and not callable(target)
            and not (isinstance(target, str) and ":" in target)
        ):
            raise ValueError(
                "robot target must be a RawRobotContract, a factory, "
                "or module:callable syntax"
            )
        if name in self._targets and not replace:
            raise ValueError(f"robot {name!r} is already registered")
        self._targets[name] = target

    def register_config(
        self, name: str, config: Mapping[str, Any], *, replace: bool = False
    ) -> RawRobotContract:
        """Register one embodiment described purely by a config mapping."""

        contract = RawRobotContract.from_mapping(config)
        self.register(name, contract, replace=replace)
        return contract

    def names(self) -> tuple[str, ...]:
        self.load_entry_points()
        return tuple(sorted(self._targets))

    def resolve(self, name: str) -> RawRobotContract:
        self.load_entry_points()
        try:
            target = self._targets[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown robot {name!r}; available: {', '.join(self.names())}"
            ) from exc
        if isinstance(target, RawRobotContract):
            return target
        if isinstance(target, str):
            module_name, attribute = target.split(":", 1)
            target = getattr(importlib.import_module(module_name), attribute)
        contract = target() if callable(target) else target
        if not isinstance(contract, RawRobotContract):
            raise TypeError(f"robot factory for {name!r} returned an invalid contract")
        self._targets[name] = contract
        return contract

    def load_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for item in entry_points(group=ENTRY_POINT_GROUP):
            self.register(item.name, item.load())


robot_registry = RobotRegistry()
robot_registry.register(DEFAULT_ROBOT, DAZZ_S600_CONTRACT)


def get_robot_contract(name: str = DEFAULT_ROBOT) -> RawRobotContract:
    return robot_registry.resolve(name)


__all__ = [
    "DEFAULT_ROBOT",
    "ENTRY_POINT_GROUP",
    "RobotRegistry",
    "get_robot_contract",
    "robot_registry",
]
