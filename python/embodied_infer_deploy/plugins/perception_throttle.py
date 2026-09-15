"""Perception/action decoupling with a throttled slow path (Reflex style).

Reflex observes that visual perception changes slowly compared with the
flow-matching action denoiser, so streaming systems can run the expensive
perception stack at a lower cadence while the action expert consumes the
cached perception output every control step.  This plugin owns exactly
that boundary:

* ``perception_fn(context)`` is the backend's slow perception entry point
  (vision encoder, LLM prefix, or a fused perception stack);
* ``get`` recomputes it only when ``refresh_steps`` control steps have
  elapsed or ``refresh_interval_s`` has passed, and otherwise replays the
  cached perception result.

Without an injected ``perception_fn`` the plugin reports ``mode=fallback``
and returns ``(None, False)`` so the backend keeps its monolithic path.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerceptionThrottleConfig:
    enabled: bool = False
    refresh_steps: int = 5
    refresh_interval_s: float = 0.0

    @classmethod
    def from_mapping(cls, value: Any) -> "PerceptionThrottleConfig":
        if isinstance(value, bool):
            return cls(enabled=value)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("perception_throttle must be a boolean or JSON object")
        return cls(
            enabled=bool(value.get("enabled", True)),
            refresh_steps=int(value.get("refresh_steps", 5)),
            refresh_interval_s=float(value.get("refresh_interval_s", 0.0)),
        )

    def __post_init__(self) -> None:
        if self.refresh_steps < 1:
            raise ValueError("perception_throttle.refresh_steps must be positive")
        if self.refresh_interval_s < 0:
            raise ValueError(
                "perception_throttle.refresh_interval_s must be non-negative"
            )


class PerceptionThrottlePlugin:
    """Cache slow perception output across fast action steps."""

    def __init__(
        self,
        perception_fn: Callable[[Any], Any] | None = None,
        config: PerceptionThrottleConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or PerceptionThrottleConfig()
        self.perception_fn = perception_fn if self.config.enabled else None
        self.mode = (
            "native"
            if self.perception_fn is not None
            else ("fallback" if self.config.enabled else "disabled")
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._cached: Any = None
        self._refreshes = 0
        self._reuses = 0
        self._last_refresh_time = float("-inf")
        self._calls_since_refresh = self.config.refresh_steps

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        perception_fn: Callable[[Any], Any] | None = None,
    ) -> "PerceptionThrottlePlugin":
        return cls(
            perception_fn=perception_fn,
            config=PerceptionThrottleConfig.from_mapping(
                config.get("perception_throttle")
            ),
        )

    @property
    def native(self) -> bool:
        return self.mode == "native"

    def _due(self) -> bool:
        if self._refreshes == 0:
            return True
        if self._calls_since_refresh >= self.config.refresh_steps:
            return True
        if self.config.refresh_interval_s > 0:
            elapsed = self._clock() - self._last_refresh_time
            return elapsed >= self.config.refresh_interval_s
        return False

    def get(self, context: Any) -> tuple[Any, bool]:
        """Return ``(perception, fresh)`` for one control step."""

        if not self.config.enabled or self.perception_fn is None:
            return None, False
        with self._lock:
            self._calls_since_refresh += 1
            if self._due():
                self._calls_since_refresh = 0
                self._last_refresh_time = self._clock()
                self._refreshes += 1
                refresh = True
            else:
                self._reuses += 1
                return self._cached, False
        # Compute outside the lock so concurrent fast steps are not blocked
        # by the slow perception stack; the last completed refresh wins.
        perception = self.perception_fn(context)
        with self._lock:
            self._cached = perception
        return perception, True

    def reset(self) -> None:
        with self._lock:
            self._cached = None
            self._refreshes = 0
            self._reuses = 0
            self._last_refresh_time = float("-inf")
            self._calls_since_refresh = self.config.refresh_steps

    def close(self) -> None:
        self.reset()

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            refreshes = self._refreshes
            reuses = self._reuses
        total = refreshes + reuses
        return {
            "enabled": bool(self.config.enabled),
            "mode": self.mode,
            "refresh_steps": self.config.refresh_steps,
            "refresh_interval_s": self.config.refresh_interval_s,
            "refreshes": refreshes,
            "reuses": reuses,
            "reuse_rate": (reuses / total) if total else 0.0,
        }


__all__ = ["PerceptionThrottleConfig", "PerceptionThrottlePlugin"]
