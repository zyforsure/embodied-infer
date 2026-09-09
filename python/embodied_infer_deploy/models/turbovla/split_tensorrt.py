"""Split TurboVLA TensorRT policy.

The exported TurboVLA graph is split into Vision, text, fusion and action
expert engines.  Keeping the Vision entry point separate lets the shared
``VisionBatchPlugin`` coalesce requests while text/fusion/action remain
request-local.
"""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...plugins import PrefixCacheConfig, PrefixCachePlugin


class _DynamicTensorRTEngine:
    """Minimal TensorRT runner for the dynamic-batch Vision engine.

    The shared ``ViTContinuousBatcher`` already coalesces requests to
    ``[B,3,3,H,W]``.  The existing runtime only supports fixed-shape engines,
    so this small class owns the dynamic optimization-profile bindings for
    the Vision stage while text/fusion/action keep using the static runtime.
    """

    def __init__(self, path: str | Path) -> None:
        import tensorrt as trt
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError:
            cuda = None

        self.trt = trt
        self.cuda = cuda
        self.backend = "pycuda" if cuda is not None else "torch"
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.backend == "pycuda":
            self.stream = cuda.Stream()
        else:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("PyCUDA is unavailable and torch.cuda is not available")
            self.torch = torch
            self.stream = torch.cuda.Stream()
        if hasattr(self.context, "set_optimization_profile_async"):
            stream_handle = self.stream.handle if self.backend == "pycuda" else self.stream.cuda_stream
            self.context.set_optimization_profile_async(0, stream_handle)

        self.inputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        self.dtypes = {
            name: np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            for name in self.inputs + self.outputs
        }
        self.dynamic_inputs = {
            name
            for name in self.inputs
            if any(dim < 0 for dim in self.engine.get_tensor_shape(name))
        }
        self.host: dict[tuple[str, tuple[int, ...]], Any] = {}
        self.device: dict[tuple[str, tuple[int, ...]], Any] = {}
        self.host_torch: dict[tuple[str, tuple[int, ...]], Any] = {}

    def _ensure_buffer(self, name: str, shape: tuple[int, ...]) -> None:
        key = (name, shape)
        if key in self.host:
            return
        dtype = self.dtypes[name]
        if self.backend == "pycuda":
            host = self.cuda.pagelocked_empty(shape, dtype=dtype)
            device = self.cuda.mem_alloc(host.nbytes)
            self.host[key] = host
            self.device[key] = device
            address = int(device)
        else:
            torch_dtype = self.torch.from_numpy(np.empty((), dtype=dtype)).dtype
            host_tensor = self.torch.empty(shape, dtype=torch_dtype, pin_memory=True)
            device_tensor = self.torch.empty(shape, dtype=torch_dtype, device="cuda")
            self.host_torch[key] = host_tensor
            self.host[key] = host_tensor.numpy()
            self.device[key] = device_tensor
            address = device_tensor.data_ptr()
        self.context.set_tensor_address(name, address)

    def infer(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        for name in self.inputs:
            value = np.asarray(feeds[name], dtype=self.dtypes[name], order="C")
            if name in self.dynamic_inputs:
                self.context.set_input_shape(name, value.shape)
            values[name] = value

        for name in self.inputs:
            shape = values[name].shape if name in self.dynamic_inputs else tuple(
                self.engine.get_tensor_shape(name)
            )
            self._ensure_buffer(name, shape)

        output_shapes: dict[str, tuple[int, ...]] = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                shape = tuple(self.engine.get_tensor_shape(name))
            output_shapes[name] = shape
            self._ensure_buffer(name, shape)

        if self.backend == "pycuda":
            for name in self.inputs:
                key = (name, values[name].shape if name in self.dynamic_inputs else tuple(
                    self.engine.get_tensor_shape(name)
                ))
                self.host[key][...] = values[name]
                self.cuda.memcpy_htod_async(self.device[key], self.host[key], self.stream)
            if not self.context.execute_async_v3(self.stream.handle):
                raise RuntimeError("dynamic TensorRT execution failed")
            for name in self.outputs:
                key = (name, output_shapes[name])
                self.cuda.memcpy_dtoh_async(self.host[key], self.device[key], self.stream)
            self.stream.synchronize()
        else:
            with self.torch.cuda.stream(self.stream):
                for name in self.inputs:
                    key = (name, values[name].shape if name in self.dynamic_inputs else tuple(
                        self.engine.get_tensor_shape(name)
                    ))
                    self.host[key][...] = values[name]
                    self.device[key].copy_(self.host_torch[key], non_blocking=True)
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("dynamic TensorRT execution failed")
                for name in self.outputs:
                    key = (name, output_shapes[name])
                    self.host_torch[key].copy_(self.device[key], non_blocking=True)
            self.stream.synchronize()

        return {name: np.asarray(self.host[(name, output_shapes[name])].copy())
                for name in self.outputs}


class SplitTensorRTPolicy:
    def __init__(self, engines: dict[str, str | Path], tokenizer_path: str | Path,
                 prefix_cache: Any = None, *,
                 stats_path: str | Path | None = None, cuda_graph: bool = False):
        from engine_runtime import TensorRTEngine
        from transformers import AutoTokenizer

        required = ("vision", "text", "fusion", "action")
        missing = [name for name in required if not engines.get(name)]
        if missing:
            raise ValueError(f"missing split TurboVLA engines: {missing}")
        self.vision_engine = _DynamicTensorRTEngine(engines["vision"])
        self.text_engine = TensorRTEngine(engines["text"], cuda_graph=cuda_graph)
        self.fusion_engine = TensorRTEngine(engines["fusion"], cuda_graph=cuda_graph)
        self.action_engine = TensorRTEngine(engines["action"], cuda_graph=cuda_graph)
        self._continuation_lock = threading.Lock()
        self.prefix_cache = PrefixCachePlugin(
            prefixer=self._encode_text,
            config=PrefixCacheConfig.from_mapping(prefix_cache),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, use_fast=True
        )
        vocab = self.tokenizer.get_vocab()
        self.special = [vocab.get(x) for x in ("[CLS]", "[SEP]", ".", "?")
                        if vocab.get(x) is not None]
        self.mean = np.asarray([0.485, 0.456, 0.406], np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], np.float32)
        self.state_min = np.zeros(14, np.float32)
        self.state_max = np.ones(14, np.float32)
        self.action_min = np.zeros(14, np.float32)
        self.action_max = np.ones(14, np.float32)
        self.action_mask = np.ones(14, dtype=bool)
        if stats_path:
            import json
            payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))["new_embodiment"]
            self.state_min = np.asarray(payload["state"]["min"], np.float32)
            self.state_max = np.asarray(payload["state"]["max"], np.float32)
            self.action_min = np.asarray(payload["action"]["min"], np.float32)
            self.action_max = np.asarray(payload["action"]["max"], np.float32)
            self.action_mask = np.asarray(payload["action"].get("mask", [True] * 14), dtype=bool)

    def _text(self, prompt: str) -> dict[str, np.ndarray]:
        tok = self.tokenizer([str(prompt)], padding="max_length", truncation=True,
                             max_length=256, return_tensors="np")
        ids = tok["input_ids"].astype(np.int32)
        mask = tok["attention_mask"].astype(bool)
        self_mask = np.eye(256, dtype=bool)[None]
        position = np.zeros((1, 256), np.int32)
        specials = set(self.special)
        previous = 0
        for col in np.flatnonzero(np.isin(ids[0], list(specials))):
            if col == 0 or col == 255:
                self_mask[0, col, col] = True
                position[0, col] = 0
            else:
                self_mask[0, previous + 1:col + 1, previous + 1:col + 1] = True
                position[0, previous + 1:col + 1] = np.arange(col - previous, dtype=np.int32)
            previous = int(col)
        return {"input_ids": ids, "token_attention_mask": mask,
                "self_attention_mask": self_mask, "position_ids": position}

    def _encode_text(self, prompt: str):
        """Tokenize + run the text engine; cached per instruction (BLURR)."""

        text_inputs = self._text(prompt)
        return text_inputs, self.text_engine.infer(text_inputs)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=np.float32).reshape(14).copy()
        valid = self.action_mask & (self.state_max != self.state_min)
        value[valid] = 2 * (value[valid] - self.state_min[valid]) / (
            self.state_max[valid] - self.state_min[valid]) - 1
        value[valid] = np.clip(value[valid], -1, 1)
        return value.reshape(1, 14)

    def encode_vision(self, batch: np.ndarray) -> np.ndarray:
        """Run the split Vision engine and return ``[B,588,256]`` tokens."""
        value = np.asarray(batch, dtype=np.float32)
        if value.ndim != 5 or value.shape[1] != 3 or value.shape[2] != 3:
            raise ValueError(f"expected [B,3,3,H,W] vision batch, got {value.shape}")
        images = value
        resized = []
        for sample in images:
            sample_views = []
            for image in sample:
                chw = np.transpose(image, (1, 2, 0))
                chw = cv2.resize(chw, (224, 224), interpolation=cv2.INTER_LINEAR)
                sample_views.append(((chw - self.mean) / self.std).transpose(2, 0, 1))
            resized.append(np.stack(sample_views))
        feeds = {"images": np.ascontiguousarray(np.stack(resized), dtype=np.float32)}
        result = self.vision_engine.infer(feeds)
        return np.asarray(result["visual_tokens"], dtype=np.float32)

    def predict_from_vision(self, vision_tokens: np.ndarray, state: np.ndarray,
                            prompt: str) -> tuple[np.ndarray, dict[str, float]]:
        start = time.perf_counter()
        with self._continuation_lock:
            bundle, _hit = self.prefix_cache.get_or_compute(str(prompt))
            if bundle is None:
                bundle = self._encode_text(prompt)
            text_inputs, text = bundle
            fusion = self.fusion_engine.infer({
                "visual_tokens": np.asarray(vision_tokens, dtype=np.float32).reshape(1, 588, 256),
                "text_tokens": text["text_tokens"],
                "text_key_padding_mask": text["text_key_padding_mask"],
                "text_self_attention_masks": text_inputs["self_attention_mask"],
            })
            actions = self.action_engine.infer({
                "condition": fusion["condition"],
                "state": self._normalize_state(state),
            })["actions"]
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (1, 50, 14) or not np.isfinite(actions).all():
            raise RuntimeError(f"unexpected split TensorRT action output: {actions.shape}")
        return actions, {"split_policy_total_ms": (time.perf_counter() - start) * 1000}

    def predict_timed(self, images, state: np.ndarray, prompt: str):
        value = np.asarray(images, dtype=np.float32)
        if value.ndim != 4:
            raise ValueError(f"expected three images, got {value.shape}")
        if value.shape[-1] == 3:
            value = np.moveaxis(value, -1, 1)
        if float(np.max(value, initial=0.0)) > 1.5:
            value = value / 255.0
        tokens = self.encode_vision(value[None])
        return self.predict_from_vision(tokens, state, prompt)

    def unnormalize_actions(self, normalized_actions: np.ndarray,
                            binary_threshold: float = 0.49) -> np.ndarray:
        normalized = np.clip(np.asarray(normalized_actions, dtype=np.float32), -1, 1)
        continuous = 0.5 * (normalized + 1) * (self.action_max - self.action_min) + self.action_min
        binary = (normalized > binary_threshold).astype(np.float32)
        return np.where(self.action_mask, continuous, binary)
