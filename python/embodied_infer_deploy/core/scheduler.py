"""Request-level inference scheduler.

This is the first refactor step toward vLLM-Omni-style scheduling: requests are
no longer coalesced inside a model plugin, but by a dedicated scheduler that
owns the queue, deadlines, batching window, and backpressure.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import queue
import threading
import time
from collections.abc import Mapping
from typing import Any, Sequence

from .request import InferenceRequest


@dataclass(frozen=True)
class BatchPolicy:
    max_batch_size: int = 4
    batch_wait_ms: float = 2.0
    max_queue_size: int = 32

    def __post_init__(self) -> None:
        if self.max_batch_size < 1 or self.max_queue_size < 1:
            raise ValueError("max_batch_size and max_queue_size must be positive")
        if self.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")

    @classmethod
    def from_mapping(cls, value):
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            return cls(
                max_batch_size=int(value.get("max_batch_size", 4)),
                batch_wait_ms=float(value.get("batch_wait_ms", 2.0)),
                max_queue_size=int(value.get("max_queue_size", 32)),
            )
        raise TypeError("batch_policy must be a mapping or None")


@dataclass(frozen=True)
class ScheduledBatch:
    requests: tuple[InferenceRequest, ...]

    def __len__(self) -> int:
        return len(self.requests)


class _QueueItem:
    __slots__ = ("request", "future")

    def __init__(self, request: InferenceRequest, future: Future) -> None:
        self.request = request
        self.future = future


class InferenceScheduler:
    """Submit requests and let a single worker dispatch batches to a runner.

    The worker is deliberately the only path that touches a non-reentrant
    model runner.  A single queue plus a small wait window gives vLLM-Omni-like
    request coalescing without giving model code its own background thread.
    """

    def __init__(self, runner, *, policy: BatchPolicy | None = None) -> None:
        if not hasattr(runner, "execute"):
            raise TypeError("scheduler runner must expose execute(batch)")
        self.runner = runner
        self.policy = policy or BatchPolicy()
        self._queue: queue.Queue[_QueueItem | None] = queue.Queue(
            maxsize=self.policy.max_queue_size
        )
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._expired = 0
        self._thread = threading.Thread(
            target=self._worker,
            name="inference-scheduler",
            daemon=True,
        )
        self._thread.start()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "expired": self._expired,
                "queue_depth": self._queue.qsize(),
            }

    def submit(
        self,
        request: InferenceRequest | dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> Future:
        """Enqueue one request and return the future completion value.

        ``deadline`` is an absolute ``time.monotonic()`` timestamp.  If omitted,
        it is derived from ``timeout_ms`` in the request payload.
        """
        envelope = (
            request
            if isinstance(request, InferenceRequest)
            else InferenceRequest(dict(request), deadline=deadline)
        )
        if deadline is not None:
            envelope = InferenceRequest(envelope.payload, deadline=deadline)
        elif envelope.deadline is None:
            envelope = InferenceRequest(
                envelope.payload,
                deadline=time.monotonic() + envelope.timeout_ms / 1000.0,
            )
        future: Future = Future()
        try:
            self._queue.put_nowait(_QueueItem(envelope, future))
        except queue.Full as exc:
            future.set_exception(
                OverflowError("inference scheduler queue is full")
            )
            with self._lock:
                self._failed += 1
            raise OverflowError("inference scheduler queue is full") from exc
        with self._lock:
            self._submitted += 1
        return future

    def close(self, *, timeout: float = 2.0) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def __enter__(self) -> "InferenceScheduler":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                return

            batch = [item]
            started = time.monotonic()
            while len(batch) < self.policy.max_batch_size:
                remaining = self.policy.batch_wait_ms / 1000.0 - (
                    time.monotonic() - started
                )
                if remaining <= 0:
                    break
                try:
                    pending = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if pending is None:
                    self._stop.set()
                    break
                batch.append(pending)

            valid: list[_QueueItem] = []
            for pending in batch:
                if pending.future.cancelled():
                    continue
                if (
                    pending.request.deadline is not None
                    and time.monotonic() >= pending.request.deadline
                ):
                    pending.future.set_exception(
                        TimeoutError("request deadline expired in scheduler queue")
                    )
                    with self._lock:
                        self._expired += 1
                    continue
                valid.append(pending)

            if not valid:
                continue
            try:
                outputs = self.runner.execute(
                    ScheduledBatch(tuple(item.request for item in valid))
                )
                if len(outputs) != len(valid):
                    raise RuntimeError(
                        f"scheduler runner returned {len(outputs)} outputs, "
                        f"expected {len(valid)}"
                    )
            except Exception as exc:
                for pending in valid:
                    pending.future.set_exception(exc)
                with self._lock:
                    self._failed += len(valid)
                continue
            for pending, output in zip(valid, outputs):
                pending.future.set_result(output)
            with self._lock:
                self._completed += len(valid)


__all__ = ["BatchPolicy", "InferenceScheduler", "ScheduledBatch"]
