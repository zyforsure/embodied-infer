"""TurboVLA TensorRT backend for RTX 4090 and Jetson AGX Orin."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ...core import BackendResult, ModelSpec
from .common import turbovla_spec


class TurboVlaTensorRtBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        required = ("runtime_root", "engine", "tokenizer", "stats")
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise ValueError(f"missing TurboVLA backend settings: {missing}")
        runtime_root = Path(config["runtime_root"]).expanduser().resolve()
        if not runtime_root.is_dir():
            raise FileNotFoundError(f"runtime_root does not exist: {runtime_root}")
        sys.path.insert(0, str(runtime_root))
        from trt_policy import TensorRTPolicy

        self.policy = TensorRTPolicy(
            config["engine"],
            config["tokenizer"],
            config["stats"],
            cuda_graph=bool(config.get("cuda_graph", False)),
        )
        self._spec = turbovla_spec(
            "turbovla-tensorrt", config, default_period_ns=100_000_000
        )
        if len(self.spec.camera_order) != 3:
            raise ValueError("TurboVLA requires exactly three camera names")

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def metadata(self) -> dict[str, Any]:
        return self.spec.metadata()

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        images = [request["images"][name] for name in self.spec.camera_order]
        normalized, timing = self.policy.predict_timed(
            images, request["state"], request["instruction"]
        )
        normalized = np.asarray(normalized, dtype=np.float32)
        if normalized.shape != (1, 50, 14):
            raise RuntimeError(f"unexpected TurboVLA output shape {normalized.shape}")
        actions = self.spec.validate_actions(
            self.policy.unnormalize_actions(normalized)[0]
        )
        return BackendResult(
            actions=actions,
            representation=self.spec.action_representation,
            control_period_ns=self.spec.control_period_ns,
            timing=timing,
        )

    def reset(self) -> None:
        return None


def create_backend(config: dict[str, Any]) -> TurboVlaTensorRtBackend:
    return TurboVlaTensorRtBackend(config)
