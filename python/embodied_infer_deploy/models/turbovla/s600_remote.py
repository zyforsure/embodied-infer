"""Gateway backend for the existing length-prefixed S600 HBM service."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from ...core import BackendResult, ModelSpec
from ...plugins import VisionBatchPlugin
from .common import turbovla_spec


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("embodied_s600_proxy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load S600 proxy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class S600HbmRemoteBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        module = _load_module(Path(config["proxy_script"]).expanduser().resolve())
        self.policy = module.TurboVLAProxyPolicy(
            s600_host=config["s600_host"],
            s600_port=int(config.get("s600_port", 5702)),
            bert_path=Path(config["tokenizer"]),
            dino_preprocessor=Path(config["dino_preprocessor"]),
            timeout=float(config.get("upstream_timeout", 120.0)),
            dataset_statistics=Path(config["stats"]),
            statistics_key=config.get("statistics_key", "new_embodiment"),
            normalization_mode=config.get("normalization_mode", "min_max"),
            binary_threshold=float(config.get("binary_threshold", 0.49)),
        )
        self.vision_plugin = VisionBatchPlugin.from_config(
            config, encoder=getattr(self.policy, "encode_vision", None)
        )
        model_config = {**config, "model_name": config.get("model_name", "TurboVLA-S600")}
        self._spec = turbovla_spec(
            "s600-hbm-remote", model_config, default_period_ns=100_000_000
        )

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def metadata(self) -> dict[str, Any]:
        value = self.spec.metadata()
        value["vision_batching"] = self.vision_plugin.metadata()
        return value

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self.spec.validate_request(request)
        example = {
            "image": [request["images"][name] for name in self.spec.camera_order],
            "state": np.asarray(request["state"], dtype=np.float32),
            "lang": request["instruction"],
        }
        output = self.policy.predict_action(
            [example], state_space="raw_model", return_action_space="model_raw"
        )
        actions = np.asarray(output["actions"], dtype=np.float32)
        if actions.shape == (1, 50, 14):
            actions = actions[0]
        actions = self.spec.validate_actions(actions)
        timing = output.get("latency_ms", {})
        return BackendResult(
            actions=actions,
            representation=self.spec.action_representation,
            control_period_ns=self.spec.control_period_ns,
            timing={str(key): float(value) for key, value in timing.items()},
        )

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self.vision_plugin.close()


def create_backend(config: dict[str, Any]) -> S600HbmRemoteBackend:
    return S600HbmRemoteBackend(config)
