"""Cross-request micro-pipeline (ActionFlow style).

ActionFlow reframes one VLA inference as a *macro-pipeline of micro-requests*:
the monolithic forward pass is split into ordered stages, and consecutive
stages of *different* requests are allowed to run in parallel.  On an edge
device this keeps otherwise idle engines busy (e.g. the action expert runs for
request N while the vision encoder already processes request N+1), raising FPS
without changing model weights and without accuracy loss.

This plugin is a bounded, thread-safe stage pipeline:

* ``stages`` is an ordered sequence of callables ``payload -> payload``;
* every stage has its own worker thread(s), connected by bounded queues;
* submitting several requests overlaps their stages across the workers.

When no ``stages`` are provided the plugin reports ``mode=fallback`` and every
``submit``/``run`` becomes an identity passthrough, so a backend can attach it
unconditionally and enable it later via configuration.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MicroPipelineConfig:
    enabled: bool = False
    max_in_flight: int = 8
    workers: int = 1

    @classmethod
    def from_mapping(cls, value: Any) -> "MicroPipelineConfig":
        if isinstance(value, bool):
            return cls(enabled=value)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("micro_pipeline must be a boolean or JSON object")
        return cls(
            enabled=bool(value.get("enabled", True)),
            max_in_flight=int(value.get("max_in_flight", 8)),
            workers=int(value.get("workers", 1)),
        )

    def __post_init__(self) -> None:
        if self.max_in_flight < 1:
            raise ValueError("micro_pipeline.max_in_flight must be positive")
        if self.workers < 1:
            raise ValueError("micro_pipeline.workers must be positive")


class MicroPipelinePlugin:
    """Overlap the stages of a fixed-function model across concurrent requests."""

    def __init__(
        self,
        stages: Sequence[Callable[[Any], Any]] | None = None,
        config: MicroPipelineConfig | None = None,
    ) -> None:
        self.config = config or MicroPipelineConfig()
        self.stages = tuple(stages or ())
        self.mode = (
            "native"
            if (self.config.enabled and self.stages)
            else ("fallback" if self.config.enabled else "disabled")
        )
        self._queues: list[queue.Queue] = []
        self._workers: list[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._submitted = 0
        self._processed = 0
        if self.mode == "native":
            self._start()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        stages: Sequence[Callable[[Any], Any]] | None = None,
    ) -> "MicroPipelinePlugin":
        return cls(stages=stages, config=MicroPipelineConfig.from_mapping(
            config.get("micro_pipeline")
        ))

    @property
    def native(self) -> bool:
        return self.mode == "native"

    def _start(self) -> None:
        # queue[i] feeds stage i; queue[-1] is the final input queue.  Each item
        # is ``(future, payload)`` carried through to the terminal stage, which
        # is the single place a request future gets resolved.
        count = len(self.stages)
        self._queues = [
            queue.Queue(maxsize=self.config.max_in_flight) for _ in range(count + 1)
        ]
        for index, stage in enumerate(self.stages):
            in_queue = self._queues[index]
            out_queue = self._queues[index + 1]
            is_last = index == count - 1
            for _ in range(self.config.workers):
                worker = threading.Thread(
                    target=self._stage_worker,
                    args=(index, stage, in_queue, out_queue, is_last),
                    name=f"micro-pipeline-{index}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)

    def _stage_worker(
        self,
        index: int,
        stage: Callable[[Any], Any],
        in_queue: "queue.Queue",
        out_queue: "queue.Queue",
        is_last: bool,
    ) -> None:
        del index
        while not self._stop.is_set():
            try:
                item = in_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            future, payload = item
            if future.cancelled():
                self._mark_processed()
                continue
            try:
                value = stage(payload)
            except Exception as exc:  # pragma: no cover - exercised by tests
                future.set_exception(exc)
                self._mark_processed()
                continue
            if is_last:
                future.set_result(value)
                self._mark_processed()
                continue
            while not self._stop.is_set():
                try:
                    out_queue.put((future, value), timeout=0.05)
                    break
                except queue.Full:
                    continue

    def _mark_processed(self) -> None:
        with self._lock:
            self._processed += 1

    def submit(self, payload: Any, *, deadline: float | None = None) -> Future:
        """Enqueue one request; returns its completion future.

        ``deadline`` is accepted for API symmetry with the request scheduler but
        enforcement stays in the scheduler layer.
        """
        del deadline
        future: Future = Future()
        if self.mode != "native":
            future.set_result(payload)
            return future
        with self._lock:
            self._submitted += 1
        self._queues[0].put((future, payload))
        return future

    def run(self, payload: Any, *, deadline: float | None = None) -> Any:
        """Submit one request and block for its result."""
        return self.submit(payload, deadline=deadline).result()

    def close(self, *, timeout: float = 2.0) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        for worker in self._workers:
            worker.join(timeout=timeout)
        self._workers.clear()
        self._queues.clear()

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            submitted = self._submitted
            processed = self._processed
        return {
            "enabled": bool(self.config.enabled),
            "mode": self.mode,
            "stages": len(self.stages),
            "workers": self.config.workers,
            "max_in_flight": self.config.max_in_flight,
            "submitted": submitted,
            "processed": processed,
            "in_flight": submitted - processed,
        }


__all__ = ["MicroPipelineConfig", "MicroPipelinePlugin"]
