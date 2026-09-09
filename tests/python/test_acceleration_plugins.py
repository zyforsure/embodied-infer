"""Contract tests for the four pluggable acceleration plugins."""

import time

import numpy as np
import pytest

from embodied_infer_deploy.plugins import (
    ActionQuantConfig,
    ActionQuantPlugin,
    MicroPipelineConfig,
    MicroPipelinePlugin,
    PrefixCacheConfig,
    PrefixCachePlugin,
    VisionTokenCacheConfig,
    VisionTokenCachePlugin,
)


# --------------------------------------------------------------------------- #
# Prefix cache
# --------------------------------------------------------------------------- #

def test_prefix_cache_reuses_instruction_prefix():
    calls = []

    def prefixer(text):
        calls.append(text)
        return {"tokens": [1, 2, 3], "text": text}

    plugin = PrefixCachePlugin(prefixer=prefixer, config=PrefixCacheConfig(enabled=True))
    try:
        prefix, hit = plugin.get_or_compute("pick up the cup")
        assert prefix == {"tokens": [1, 2, 3], "text": "pick up the cup"}
        assert hit is False

        again, hit = plugin.get_or_compute("pick up the cup")
        assert again == prefix
        assert hit is True
        assert calls == ["pick up the cup"]  # computed exactly once
        assert plugin.metadata()["hit_rate"] == 0.5
    finally:
        plugin.close()


def test_prefix_cache_normalizes_whitespace():
    calls = []

    plugin = PrefixCachePlugin(
        prefixer=lambda text: calls.append(text) or len(text),
        config=PrefixCacheConfig(enabled=True, normalize=True),
    )
    try:
        _, hit = plugin.get_or_compute("  pick   up the cup  ")
        assert hit is False
        _, hit = plugin.get_or_compute("pick up the cup")
        assert hit is True
        assert len(calls) == 1
    finally:
        plugin.close()


def test_prefix_cache_falls_back_without_prefixer():
    plugin = PrefixCachePlugin.from_config({"prefix_cache": True})
    try:
        assert plugin.mode == "fallback"
        assert plugin.get_or_compute("any instruction") == (None, False)
        assert plugin.metadata()["mode"] == "fallback"
    finally:
        plugin.close()


def test_prefix_cache_disabled_is_noop():
    plugin = PrefixCachePlugin(prefixer=lambda text: text, config=PrefixCacheConfig())
    try:
        assert plugin.mode == "disabled"
        assert plugin.get_or_compute("any instruction") == (None, False)
    finally:
        plugin.close()


def test_prefix_cache_evicts_beyond_capacity():
    plugin = PrefixCachePlugin(
        prefixer=lambda text: len(text),
        config=PrefixCacheConfig(enabled=True, max_entries=2),
    )
    try:
        plugin.get_or_compute("a")
        plugin.get_or_compute("b")
        plugin.get_or_compute("c")
        assert plugin.metadata()["entries"] == 2
        assert plugin.lookup("a") is None  # oldest entry evicted
    finally:
        plugin.close()


# --------------------------------------------------------------------------- #
# Vision token cache
# --------------------------------------------------------------------------- #

def test_vision_token_cache_reuses_unchanged_frames():
    calls = []

    def encoder(image):
        calls.append(image)
        return np.arange(4, dtype=np.float32)

    plugin = VisionTokenCachePlugin(
        encoder=encoder, config=VisionTokenCacheConfig(enabled=True)
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        tokens, reused = plugin.encode("head", image)
        assert reused is False
        assert tokens.shape == (4,)

        tokens_again, reused = plugin.encode("head", image)
        assert reused is True
        assert np.array_equal(tokens_again, tokens)
        assert len(calls) == 1  # encoder ran once
        assert plugin.metadata()["reuse_rate"] == 0.5
    finally:
        plugin.close()


def test_vision_token_cache_reencodes_changed_frames():
    def encoder(image):
        return np.sum(image)

    plugin = VisionTokenCachePlugin(
        encoder=encoder, config=VisionTokenCacheConfig(enabled=True)
    )
    try:
        _, reused = plugin.encode("head", np.zeros((4, 4, 3), dtype=np.uint8))
        assert reused is False
        _, reused = plugin.encode("head", np.ones((4, 4, 3), dtype=np.uint8))
        assert reused is False
        assert plugin.metadata()["misses"] == 2
    finally:
        plugin.close()


def test_vision_token_cache_prunes_only_fresh_tokens():
    prune_calls = []

    def encoder(image):
        return np.arange(10, dtype=np.float32)

    def pruner(tokens, keep):
        prune_calls.append(keep)
        return tokens[: max(1, int(len(tokens) * keep))]

    plugin = VisionTokenCachePlugin(
        encoder=encoder,
        pruner=pruner,
        config=VisionTokenCacheConfig(enabled=True, prune_ratio=0.5),
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        tokens, reused = plugin.encode("head", image)
        assert reused is False
        assert tokens.shape == (5,)
        _, reused = plugin.encode("head", image)
        assert reused is True
        assert len(prune_calls) == 1  # pruning skips reused results
        assert plugin.metadata()["pruning"] is True
    finally:
        plugin.close()


def test_vision_token_cache_falls_back_without_encoder():
    plugin = VisionTokenCachePlugin.from_config({"vision_token_cache": True})
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        assert plugin.mode == "fallback"
        value, reused = plugin.encode("head", image)
        assert reused is False
        assert value is image  # original path preserved
    finally:
        plugin.close()


# --------------------------------------------------------------------------- #
# Micro pipeline
# --------------------------------------------------------------------------- #

def test_micro_pipeline_composes_stages_in_order():
    plugin = MicroPipelinePlugin(
        stages=[lambda x: x + 1, lambda x: x * 2],
        config=MicroPipelineConfig(enabled=True),
    )
    try:
        assert plugin.mode == "native"
        assert plugin.run(10) == 22
        assert plugin.metadata()["processed"] == 1
    finally:
        plugin.close()


def test_micro_pipeline_overlaps_requests():
    delay = 0.05

    def make_stage():
        def stage(value):
            time.sleep(delay)
            return value

        return stage

    plugin = MicroPipelinePlugin(
        stages=[make_stage(), make_stage(), make_stage()],
        config=MicroPipelineConfig(enabled=True, max_in_flight=8),
    )
    try:
        started = time.monotonic()
        futures = [plugin.submit(i) for i in range(3)]
        results = [future.result(timeout=2.0) for future in futures]
        elapsed = time.monotonic() - started

        assert results == [0, 1, 2]
        # 3 stages x 3 requests: sequential = 0.45s, pipelined ~= 0.25s.
        assert elapsed < 0.34, f"expected pipelining, took {elapsed:.3f}s"
    finally:
        plugin.close()


def test_micro_pipeline_propagates_stage_errors():
    def boom(value):
        raise RuntimeError("stage failed")

    plugin = MicroPipelinePlugin(
        stages=[lambda x: x + 1, boom], config=MicroPipelineConfig(enabled=True)
    )
    try:
        future = plugin.submit(1)
        with pytest.raises(RuntimeError, match="stage failed"):
            future.result(timeout=1.0)
    finally:
        plugin.close()


def test_micro_pipeline_falls_back_to_identity():
    plugin = MicroPipelinePlugin.from_config({"micro_pipeline": True})
    try:
        assert plugin.mode == "fallback"
        assert plugin.run({"payload": 1}) == {"payload": 1}
    finally:
        plugin.close()


# --------------------------------------------------------------------------- #
# Action-aware quantization
# --------------------------------------------------------------------------- #

def test_action_quant_roundtrip_preserves_shape_and_is_bounded():
    weights = np.stack([np.linspace(-1.0, 1.0, 256) for _ in range(4)]).astype(np.float32)
    importance = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    plugin = ActionQuantPlugin(ActionQuantConfig(enabled=True))
    try:
        state = plugin.fit(weights, importance)
        quantized = plugin.quantize(weights, state)
        recovered = plugin.dequantize(quantized, state)
        assert recovered.shape == weights.shape
        assert np.isfinite(recovered).all()
        # 8-bit default quantization keeps relative error comfortably below 1%.
        assert np.max(np.abs(recovered - weights)) < 0.01
        assert plugin.metadata()["calibrated"] is True
    finally:
        plugin.close()


def test_action_quant_protects_sensitive_channels():
    weights = np.stack([np.linspace(-1.0, 1.0, 512) for _ in range(4)]).astype(np.float32)
    # Channel 3 is the only action-sensitive channel and receives 16 bits.
    importance = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    plugin = ActionQuantPlugin(
        ActionQuantConfig(enabled=True, sensitive_bits=16, default_bits=8)
    )
    try:
        state = plugin.fit(weights, importance)
        errors = plugin.error(weights, state)
        assert state.bits[3] == 16
        assert (state.bits[:3] == 8).all()
        assert errors[3] < errors[0] * 0.1
    finally:
        plugin.close()


def test_action_quant_quantized_values_fit_bit_range():
    weights = np.stack([np.linspace(-2.0, 2.0, 64) for _ in range(2)]).astype(np.float32)
    importance = np.array([1.0, 0.0], dtype=np.float32)

    plugin = ActionQuantPlugin(
        ActionQuantConfig(enabled=True, sensitive_bits=16, default_bits=8)
    )
    try:
        state = plugin.fit(weights, importance)
        quantized = plugin.quantize(weights, state)
        matrix = np.moveaxis(quantized, state.axis, 0).reshape(2, -1)
        for channel in range(2):
            qmax = (1 << int(state.bits[channel])) - 1
            assert matrix[channel].min() >= 0.0
            assert matrix[channel].max() <= qmax
    finally:
        plugin.close()


def test_action_quant_rejects_invalid_config_and_inputs():
    with pytest.raises(ValueError):
        ActionQuantConfig(default_bits=3)
    with pytest.raises(ValueError):
        ActionQuantConfig(sensitive_bits=8, default_bits=16)
    with pytest.raises(ValueError):
        ActionQuantConfig(sensitive_ratio=1.5)

    plugin = ActionQuantPlugin(ActionQuantConfig(enabled=True))
    try:
        weights = np.zeros((3, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            plugin.fit(weights, np.zeros(2, dtype=np.float32))
        with pytest.raises(RuntimeError):
            plugin.quantize(weights)  # not fit yet
    finally:
        plugin.close()
