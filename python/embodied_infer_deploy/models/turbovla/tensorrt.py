"""TurboVLA TensorRT backend for RTX 4090 and Jetson AGX Orin."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...core import BackendResult, ModelSpec
from ...plugins import VisionBatchPlugin
from .common import turbovla_spec


class TurboVlaTensorRtBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        required = ("runtime_root", "tokenizer", "stats")
        if not config.get("split_engines"):
            required += ("engine",)
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise ValueError(f"missing TurboVLA backend settings: {missing}")
        runtime_root = Path(config["runtime_root"]).expanduser().resolve()
        if not runtime_root.is_dir():
            raise FileNotFoundError(f"runtime_root does not exist: {runtime_root}")
        sys.path.insert(0, str(runtime_root))
        if config.get("split_engines"):
            from .split_tensorrt import SplitTensorRTPolicy
            self.policy = SplitTensorRTPolicy(
                config["split_engines"], config["tokenizer"],
                config.get("prefix_cache"), stats_path=config["stats"],
                cuda_graph=bool(config.get("cuda_graph", False)),
                vision_token_cache=config.get("vision_token_cache"),
                perception_throttle=config.get("perception_throttle"),
                cascade=config.get("cascade"),
            )
            self.supports_concurrent_infer = True
        else:
            from trt_policy import TensorRTPolicy
            self.policy = TensorRTPolicy(
                config["engine"], config["tokenizer"], config["stats"],
                cuda_graph=bool(config.get("cuda_graph", False)),
            )
            self.supports_concurrent_infer = False
        encoder = getattr(self.policy, "encode_vision", None)
        self.vision_plugin = VisionBatchPlugin.from_config(config, encoder=encoder)
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
        value = self.spec.metadata()
        value["vision_batching"] = self.vision_plugin.metadata()
        plugin_metadata = getattr(self.policy, "plugin_metadata", None)
        if callable(plugin_metadata):
            value.update(plugin_metadata())
        else:
            prefix_cache = getattr(self.policy, "prefix_cache", None)
            if prefix_cache is not None:
                value["prefix_cache"] = prefix_cache.metadata()
        return value

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        images = [request["images"][name] for name in self.spec.camera_order]
        deadline = time.monotonic() + float(request.get("timeout_ms", 300_000)) / 1000.0
        if self.vision_plugin.native and callable(getattr(self.policy, "predict_from_vision", None)):
            vision = self.vision_plugin.encode(images, deadline=deadline)
            normalized, timing = self.policy.predict_from_vision(
                vision, request["state"], request["instruction"]
            )
        else:
            # Compatibility path for existing TensorRT releases whose policy
            # still owns image preprocessing and the full forward call.
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

    def create_runner(self):
        """Return a request-level runner when this backend is split-engine based."""
        if not hasattr(self.policy, "predict_from_vision"):
            return None
        from .runner import TurboVlaSplitRunner
        return TurboVlaSplitRunner({
            **self._config,
            "model_name": self.spec.name,
            "camera_order": list(self.spec.camera_order),
        })

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self.vision_plugin.close()


def create_backend(config: dict[str, Any]) -> TurboVlaTensorRtBackend:
    return TurboVlaTensorRtBackend(config)
