"""Request-level TurboVLA runners.

These runners consume scheduled batches instead of owning their own queue.  The
current split implementation batches the vision stage and falls back to serial
continuation because the text/fusion/action TensorRT engines are still fixed
batch=1.  Replacing those engines with dynamic profiles is the next model-side
step.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ...core import BackendResult, ModelRunner, ScheduledBatch
from .common import turbovla_spec


class TurboVlaSplitRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        from .split_tensorrt import SplitTensorRTPolicy

        self.policy = SplitTensorRTPolicy(
            config["split_engines"],
            config["tokenizer"],
            config.get("prefix_cache"),
            stats_path=config.get("stats"),
            cuda_graph=bool(config.get("cuda_graph", False)),
            vision_token_cache=config.get("vision_token_cache"),
            perception_throttle=config.get("perception_throttle"),
            cascade=config.get("cascade"),
        )
        self._spec = turbovla_spec(
            "turbovla-tensorrt", config, default_period_ns=100_000_000
        )

    @property
    def spec(self):
        return self._spec

    def _request_images(self, request: dict[str, Any]) -> np.ndarray:
        images = [request["images"][name] for name in self._spec.camera_order]
        value = np.asarray(images, dtype=np.float32)
        if value.ndim != 4:
            raise ValueError(f"expected three camera views, got {value.shape}")
        if value.shape[-1] == 3:
            value = np.moveaxis(value, -1, 1)
        if float(np.max(value, initial=0.0)) > 1.5:
            value = value / 255.0
        return np.ascontiguousarray(value)

    def execute(self, batch: ScheduledBatch) -> list[BackendResult]:
        image_batch = np.stack(
            [self._request_images(request.as_backend_request()) for request in batch.requests],
            axis=0,
        )
        vision_started = time.perf_counter()
        try:
            vision_tokens = self.policy.encode_vision(image_batch)
        except Exception:
            if image_batch.shape[0] == 1:
                raise
            # Temporary fallback until the split Vision TensorRT engine is rebuilt
            # with a true dynamic batch profile.  This keeps request-level
            # scheduling usable while model conversion catches up.
            vision_tokens = np.stack(
                [self.policy.encode_vision(sample[None])[0] for sample in image_batch],
                axis=0,
            )
        vision_ms = (time.perf_counter() - vision_started) * 1000.0
        if vision_tokens.shape[0] != len(batch):
            raise RuntimeError(
                f"TurboVLA vision batch returned {vision_tokens.shape[0]}, "
                f"expected {len(batch)}"
            )

        results: list[BackendResult] = []
        for index, request in enumerate(batch.requests):
            payload = request.as_backend_request()
            normalized, timing = self.policy.predict_from_vision(
                vision_tokens[index], payload["state"], payload["instruction"]
            )
            normalized = np.asarray(normalized, dtype=np.float32)
            if normalized.shape != (1, 50, 14):
                raise RuntimeError(f"unexpected TurboVLA output shape {normalized.shape}")
            actions = self._spec.validate_actions(
                self.policy.unnormalize_actions(normalized)[0]
            )
            timing = {**timing, "vision_ms": vision_ms}
            results.append(
                BackendResult(
                    actions=actions,
                    representation=self._spec.action_representation,
                    control_period_ns=self._spec.control_period_ns,
                    timing=timing,
                )
            )
        return results


__all__ = ["TurboVlaSplitRunner"]
