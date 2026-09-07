"""Gateway backend for the existing length-prefixed S600 HBM server."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from .base import BackendResult


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
        self.camera_order = list(config.get(
            "camera_order", ["head", "left_wrist", "right_wrist"]
        ))
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
        self.control_period_ns = int(config.get("control_period_ns", 100_000_000))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "s600-hbm-remote",
            "model": "TurboVLA-S600",
            "raw_state_dim": 18,
            "model_state_dim": 14,
            "raw_action_dim": 18,
            "model_action_dim": 14,
            "state_dim": 14,
            "action_dim": 14,
            "action_horizon": 50,
            "action_representation": "absolute",
            "camera_order": self.camera_order,
            "control_period_ns": self.control_period_ns,
        }

    def infer(self, request: dict[str, Any]) -> BackendResult:
        example = {
            "image": [request["images"][name] for name in self.camera_order],
            "state": np.asarray(request["state"], dtype=np.float32),
            "lang": request["instruction"],
        }
        output = self.policy.predict_action(
            [example], state_space="raw_model", return_action_space="model_raw"
        )
        actions = np.asarray(output["actions"], dtype=np.float32)
        if actions.shape == (1, 50, 14):
            actions = actions[0]
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise RuntimeError(f"unexpected S600 proxy output {actions.shape}")
        timing = output.get("latency_ms", {})
        return BackendResult(
            actions=actions,
            representation="absolute",
            control_period_ns=self.control_period_ns,
            timing={str(key): float(value) for key, value in timing.items()},
        )

    def reset(self) -> None:
        return None


def create_backend(config: dict[str, Any]) -> S600HbmRemoteBackend:
    return S600HbmRemoteBackend(config)
