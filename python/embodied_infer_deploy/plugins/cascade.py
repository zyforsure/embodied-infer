"""Light/heavy model scheduling per control step (SP-VLA style).

SP-VLA shows that not every control step deserves the full VLA: easy steps
can be handled by a lightweight action generator while hard steps are
escalated to the big model.  This plugin owns only the routing decision:

* ``scorer(context) -> float`` is a backend-injected difficulty score in
  ``[0, 1]`` (state novelty, action disagreement, prediction entropy...);
* ``route`` returns ``"light"`` below ``difficulty_threshold`` and
  ``"heavy"`` otherwise.

Without a scorer the plugin reports ``mode=fallback`` and always routes
``"heavy"`` -- the backend's original, full-model path -- so enabling the
plugin is never a correctness risk until a difficulty signal exists.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

Route = Literal["light", "heavy"]


@dataclass(frozen=True)
class CascadeConfig:
    enabled: bool = False
    difficulty_threshold: float = 0.5

    @classmethod
    def from_mapping(cls, value: Any) -> "CascadeConfig":
        if isinstance(value, bool):
            return cls(enabled=value)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("cascade must be a boolean or JSON object")
        return cls(
            enabled=bool(value.get("enabled", True)),
            difficulty_threshold=float(value.get("difficulty_threshold", 0.5)),
        )

    def __post_init__(self) -> None:
        if not 0.0 <= self.difficulty_threshold <= 1.0:
            raise ValueError("cascade.difficulty_threshold must be in [0, 1]")


class CascadePlugin:
    """Route each control step to a light or heavy model path."""

    def __init__(
        self,
        scorer: Callable[[Any], float] | None = None,
        config: CascadeConfig | None = None,
    ) -> None:
        self.config = config or CascadeConfig()
        self.scorer = scorer if self.config.enabled else None
        self.mode = (
            "native"
            if self.scorer is not None
            else ("fallback" if self.config.enabled else "disabled")
        )
        self._lock = threading.Lock()
        self._light = 0
        self._heavy = 0

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        scorer: Callable[[Any], float] | None = None,
    ) -> "CascadePlugin":
        return cls(scorer=scorer, config=CascadeConfig.from_mapping(
            config.get("cascade")
        ))

    @property
    def native(self) -> bool:
        return self.mode == "native"

    def route(self, context: Any) -> Route:
        """Return ``"light"`` or ``"heavy"`` for one control step."""

        choice: Route = "heavy"
        if self.scorer is not None:
            score = float(self.scorer(context))
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"cascade scorer must return [0, 1], got {score}")
            if score < self.config.difficulty_threshold:
                choice = "light"
        with self._lock:
            if choice == "light":
                self._light += 1
            else:
                self._heavy += 1
        return choice

    def reset(self) -> None:
        with self._lock:
            self._light = 0
            self._heavy = 0

    def close(self) -> None:
        self.reset()

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            light = self._light
            heavy = self._heavy
        total = light + heavy
        return {
            "enabled": bool(self.config.enabled),
            "mode": self.mode,
            "difficulty_threshold": self.config.difficulty_threshold,
            "light_routes": light,
            "heavy_routes": heavy,
            "light_rate": (light / total) if total else 0.0,
        }


__all__ = ["CascadeConfig", "CascadePlugin", "Route"]
