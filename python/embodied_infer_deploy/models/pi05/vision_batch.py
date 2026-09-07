"""Continuous batching primitives for Pi05's ViT/SigLIP stage.

Requests are coalesced for a short window and encoded as one tensor
``[batch, views, 3, height, width]``.  The component is backend agnostic: a
TensorRT, torch, or HBM encoder can be supplied as ``encoder``.  LLM and
action-expert requests remain per-session because their KV/cache state differs.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class _VisionRequest:
    images: np.ndarray
    future: Future
    deadline: float | None


class ViTContinuousBatcher:
    def __init__(
        self,
        encoder: Callable[[np.ndarray], Any],
        *,
        max_batch_size: int = 4,
        batch_wait_ms: float = 2.0,
        max_queue_size: int = 32,
    ) -> None:
        if max_batch_size < 1 or max_queue_size < 1 or batch_wait_ms < 0:
            raise ValueError("batch sizes must be positive and batch_wait_ms non-negative")
        self.encoder = encoder
        self.max_batch_size = int(max_batch_size)
        self.batch_wait_s = float(batch_wait_ms) / 1000.0
        self._queue: queue.Queue[_VisionRequest | None] = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="pi05-vit-batcher", daemon=True)
        self._thread.start()

    @staticmethod
    def _canonical(images: Sequence[Any]) -> np.ndarray:
        value = np.asarray(images, dtype=np.float32)
        if value.ndim == 4 and value.shape[1] == 3:
            pass  # [views, 3, H, W]
        elif value.ndim == 4 and value.shape[-1] == 3:
            value = np.moveaxis(value, -1, 1)  # [views, 3, H, W]
        else:
            raise ValueError(f"ViT request must be [views,3,H,W] or [views,H,W,3], got {value.shape}")
        if value.shape[0] < 1 or value.shape[2] < 1 or value.shape[3] < 1:
            raise ValueError("ViT request contains an empty view or image")
        if not np.isfinite(value).all():
            raise ValueError("ViT request contains NaN or infinity")
        if float(np.max(value, initial=0.0)) > 1.5:
            value = value / 255.0
        return np.ascontiguousarray(np.clip(value, 0.0, 1.0), dtype=np.float32)

    def submit(self, images: Sequence[Any], *, deadline: float | None = None) -> Future:
        future: Future = Future()
        request = _VisionRequest(self._canonical(images), future, deadline)
        try:
            self._queue.put_nowait(request)
        except queue.Full as exc:
            future.set_exception(OverflowError("Pi05 ViT continuous-batching queue is full"))
            raise OverflowError("Pi05 ViT continuous-batching queue is full") from exc
        return future

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if first is None:
                return
            batch = [first]
            started = time.monotonic()
            while len(batch) < self.max_batch_size:
                remaining = self.batch_wait_s - (time.monotonic() - started)
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self._stop.set()
                    break
                batch.append(item)
            valid: list[_VisionRequest] = []
            for item in batch:
                if item.future.cancelled():
                    continue
                if item.deadline is not None and time.monotonic() >= item.deadline:
                    item.future.set_exception(TimeoutError("Pi05 ViT request deadline expired before batch execution"))
                else:
                    valid.append(item)
            if not valid:
                continue
            try:
                shapes = {tuple(item.images.shape) for item in valid}
                if len(shapes) != 1:
                    raise ValueError("continuous batching requires equal view/image shapes")
                inputs = np.stack([item.images for item in valid], axis=0)
                outputs = self.encoder(inputs)
                if isinstance(outputs, np.ndarray) and outputs.shape[0] != len(valid):
                    raise ValueError(f"ViT encoder returned batch {outputs.shape[0]}, expected {len(valid)}")
                if isinstance(outputs, (list, tuple)) and len(outputs) != len(valid):
                    raise ValueError(f"ViT encoder returned {len(outputs)} outputs, expected {len(valid)}")
                for index, item in enumerate(valid):
                    item.future.set_result(outputs[index])
            except Exception as exc:
                for item in valid:
                    item.future.set_exception(exc)

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def __enter__(self) -> "ViTContinuousBatcher":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["ViTContinuousBatcher"]
