"""D-Robotics HBM loader for Pi05.

The runtime is intentionally imported lazily because it exists only on S100 /
S600 images.  Loading is separated from inference so a service health check
can report an actionable ``nash-p``/``nash-e`` mismatch instead of appearing
ready after merely finding files on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...core import BackendResult, ModelSpec
from ...plugins import VisionBatchPlugin


class Pi05HbmBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.paths = {name: Path(value) for name, value in {
            "vision": config.get("vision_hbm"), "llm": config.get("llm_hbm"),
            "expert": config.get("expert_hbm")}.items() if value}
        self.expected_march = str(config.get("expected_march", "nash-e"))
        self._models: dict[str, Any] = {}
        self._load_error: str | None = None
        self.vision_plugin = VisionBatchPlugin.from_config(config)
        self._spec = ModelSpec(
            name=str(config.get("model_name", "Pi05-HBM")), backend="pi05-hbm",
            raw_state_dim=int(config.get("raw_state_dim", 18)), model_state_dim=int(config.get("model_state_dim", 14)),
            raw_action_dim=int(config.get("raw_action_dim", 18)), model_action_dim=int(config.get("model_action_dim", 14)),
            action_horizon=int(config.get("action_horizon", 16)),
            camera_order=tuple(config.get("camera_order", ["head", "left_wrist", "right_wrist"])),
            action_representation="absolute", control_period_ns=int(config.get("control_period_ns", 100000000)),
            extras={"expected_march": self.expected_march, "hbm_components": list(self.paths)},
        )
        if bool(config.get("load_on_init", True)):
            self._load()

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def metadata(self) -> dict[str, Any]:
        result = self.spec.metadata()
        result["vision_batching"] = self.vision_plugin.metadata()
        result.update({"hbm_ready": not self._load_error and len(self._models) == len(self.paths),
                       "hbm_loaded_components": sorted(self._models)})
        if self._load_error:
            result["hbm_error"] = self._load_error
        return result

    def _load(self) -> None:
        missing = [f"{name}={path}" for name, path in self.paths.items() if not path.is_file()]
        if missing:
            self._load_error = "missing HBM files: " + ", ".join(missing)
            return
        try:
            from hbm_runtime import HB_HBMRuntime
            for name, path in self.paths.items():
                self._models[name] = HB_HBMRuntime(str(path))
        except Exception as exc:
            message = str(exc)
            if "march" in message.lower() or "incompatible" in message.lower():
                message = (f"HBM target mismatch (expected {self.expected_march}); {message}. "
                           "Recompile all three graphs with the board march; do not reuse nash-p artifacts on S100/nash-e.")
            self._models.clear()
            self._load_error = message

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        if self._load_error or len(self._models) != len(self.paths):
            raise RuntimeError(self._load_error or "Pi05 HBM components are not loaded")
        raise NotImplementedError(
            "Pi05 HBM tensor bindings are model-release specific; configure the compiled "
            "SigLIP -> Gemma LLM -> Action Expert graph bindings before enabling inference"
        )

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self.vision_plugin.close()


def create_backend(config: dict[str, Any]) -> Pi05HbmBackend:
    return Pi05HbmBackend(config)


__all__ = ["Pi05HbmBackend", "create_backend"]
