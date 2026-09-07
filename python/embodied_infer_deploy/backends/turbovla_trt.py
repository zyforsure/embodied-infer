"""Adapter for the existing TurboVLA TensorRTPolicy used on AGX Orin/4090."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .base import BackendResult


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
        self.camera_order = list(config.get(
            "camera_order", ["head", "left_wrist", "right_wrist"]
        ))
        if len(self.camera_order) != 3:
            raise ValueError("TurboVLA requires exactly three camera names")
        self.control_period_ns = int(config.get("control_period_ns", 100_000_000))
        self._metadata = {
            "backend": "turbovla-tensorrt",
            "model": str(config.get("model_name", "TurboVLA-RoboTwin")),
            "raw_state_dim": 18,
            "model_state_dim": 14,
            "raw_action_dim": 18,
            "model_action_dim": 14,
            # Compatibility aliases describe the tensors carried on the wire.
            "state_dim": 14,
            "action_dim": 14,
            "action_horizon": 50,
            "action_representation": "absolute",
            "camera_order": self.camera_order,
            "control_period_ns": self.control_period_ns,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def infer(self, request: dict[str, Any]) -> BackendResult:
        if request["state"].shape != (14,):
            raise ValueError(f"TurboVLA state must be [14], got {request['state'].shape}")
        try:
            images = [request["images"][name] for name in self.camera_order]
        except KeyError as exc:
            raise ValueError(f"missing required camera {exc.args[0]!r}") from exc
        normalized, timing = self.policy.predict_timed(
            images, request["state"], request["instruction"]
        )
        normalized = np.asarray(normalized, dtype=np.float32)
        if normalized.shape != (1, 50, 14):
            raise RuntimeError(f"unexpected TurboVLA output shape {normalized.shape}")
        actions = self.policy.unnormalize_actions(normalized)[0]
        if not np.isfinite(actions).all():
            raise RuntimeError("TurboVLA returned NaN or infinity")
        return BackendResult(
            actions=actions,
            representation="absolute",
            control_period_ns=self.control_period_ns,
            timing=timing,
        )

    def reset(self) -> None:
        return None


def create_backend(config: dict[str, Any]) -> TurboVlaTensorRtBackend:
    return TurboVlaTensorRtBackend(config)
