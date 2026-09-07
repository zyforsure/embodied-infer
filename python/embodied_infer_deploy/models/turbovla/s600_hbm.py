"""Direct backend for the proven four-graph S600 HBM runtime."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from ...core import BackendResult, ModelSpec
from .common import turbovla_spec


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("embodied_s600_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load S600 runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class S600HbmBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        script = Path(config["runtime_script"]).expanduser().resolve()
        self.module = _load_module(script)
        self.engine = self.module.TurboVLAEngine(
            Path(config["model_root"]), Path(config["assets"])
        )
        self.input_color = config.get("input_color", "rgb")
        if self.input_color not in {"rgb", "bgr"}:
            raise ValueError("input_color must be rgb or bgr")
        model_config = {**config, "model_name": config.get("model_name", "TurboVLA-S600")}
        self._spec = turbovla_spec(
            "s600-hbm-direct", model_config, default_period_ns=50_000_000
        )

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def metadata(self) -> dict[str, Any]:
        return self.spec.metadata()

    def _images(self, request: dict[str, Any]) -> np.ndarray:
        import cv2

        mean = np.asarray([0.485, 0.456, 0.406], np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], np.float32)[:, None, None]
        values = []
        for name in self.spec.camera_order:
            image = request["images"][name]
            if self.input_color == "bgr":
                image = image[..., ::-1]
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            chw = image.astype(np.float32).transpose(2, 0, 1) / 255.0
            values.append((chw - mean) / std)
        return np.ascontiguousarray(np.stack(values)[None], dtype=np.float32)

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        normalized_state = self.module.normalize_state(request["state"], self.engine)
        normalized_actions = self.engine.infer(
            self._images(request), normalized_state, request["instruction"]
        )
        actions = self.spec.validate_actions(
            self.engine.denormalize_actions(normalized_actions)
        )
        return BackendResult(
            actions=actions,
            representation=self.spec.action_representation,
            control_period_ns=self.spec.control_period_ns,
        )

    def reset(self) -> None:
        return None


def create_backend(config: dict[str, Any]) -> S600HbmBackend:
    return S600HbmBackend(config)
