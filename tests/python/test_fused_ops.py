# -*- coding: utf-8 -*-
"""Contract tests for the fused operator plugin."""

import importlib.util

import numpy as np
import pytest

from embodied_infer_deploy.plugins import FusedOpsConfig, FusedOpsPlugin
from embodied_infer_deploy.plugins.fused_ops import (
    attention_reference,
    gelu_mlp_reference,
    layer_norm_reference,
    qkv_reference,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None

RNG = np.random.default_rng(7)
HIDDEN = RNG.normal(size=(2, 5, 16)).astype(np.float32)
W_Q = RNG.normal(size=(16, 16)).astype(np.float32)
W_K = RNG.normal(size=(16, 16)).astype(np.float32)
W_V = RNG.normal(size=(16, 16)).astype(np.float32)
BIASES = tuple(RNG.normal(size=(16,)).astype(np.float32) for _ in range(3))


def make_plugin(**overrides):
    config = FusedOpsConfig(enabled=True, **overrides)
    return FusedOpsPlugin(config=config)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def test_config_defaults_disabled():
    assert FusedOpsConfig().enabled is False
    assert FusedOpsConfig.from_mapping(None).enabled is False
    assert FusedOpsConfig.from_mapping(True).enabled is True
    assert FusedOpsConfig.from_mapping({"enabled": True, "prefer": "torch"}).prefer == "torch"


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        FusedOpsConfig(enabled=True, prefer="gpu")
    with pytest.raises(TypeError):
        FusedOpsConfig.from_mapping("yes")


def test_from_config_reads_fused_ops_section():
    plugin = FusedOpsPlugin.from_config({"fused_ops": {"enabled": True}})
    try:
        assert plugin.config.enabled is True
    finally:
        plugin.close()
    assert FusedOpsPlugin.from_config({}).config.enabled is False


def test_unknown_override_rejected():
    with pytest.raises(ValueError):
        FusedOpsPlugin(config=FusedOpsConfig(enabled=True),
                       native_fn={"softmax": lambda *a: a})


# --------------------------------------------------------------------------- #
# NumPy reference correctness (always runs, no torch needed)
# --------------------------------------------------------------------------- #

def test_qkv_fallback_matches_separate_projections():
    plugin = make_plugin(prefer="numpy")
    try:
        q, k, v = plugin.project_qkv(HIDDEN, (W_Q, W_K, W_V), BIASES)
        assert q.shape == HIDDEN.shape
        np.testing.assert_allclose(q, HIDDEN @ W_Q.T + BIASES[0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(k, HIDDEN @ W_K.T + BIASES[1], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(v, HIDDEN @ W_V.T + BIASES[2], rtol=1e-5, atol=1e-6)
        assert plugin.mode == "fallback"
        assert plugin.metadata()["calls"]["qkv"] == {"native": 0, "fallback": 1}
    finally:
        plugin.close()


def test_attention_fallback_matches_manual_softmax():
    q = RNG.normal(size=(1, 2, 4, 8)).astype(np.float32)
    k = RNG.normal(size=(1, 2, 4, 8)).astype(np.float32)
    v = RNG.normal(size=(1, 2, 4, 8)).astype(np.float32)
    out = attention_reference(q, k, v)
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(8.0)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    np.testing.assert_allclose(out, weights @ v, rtol=1e-5, atol=1e-6)


def test_attention_causal_mask_blocks_future():
    q = np.zeros((1, 1, 3, 2), dtype=np.float32)
    k = np.eye(3, 2, dtype=np.float32)[None, None]
    v = np.arange(6, dtype=np.float32).reshape(1, 1, 3, 2)
    out = attention_reference(q, k, v, causal=True)
    # first query can only attend to the first key/value row
    np.testing.assert_allclose(out[0, 0, 0], v[0, 0, 0], rtol=1e-5, atol=1e-6)
    full = attention_reference(q, k, v, causal=False)
    assert not np.allclose(out[0, 0, 0], full[0, 0, 0])


def test_gelu_mlp_fallback_matches_manual_tanh_gelu():
    w1 = RNG.normal(size=(32, 16)).astype(np.float32)
    b1 = RNG.normal(size=(32,)).astype(np.float32)
    w2 = RNG.normal(size=(8, 32)).astype(np.float32)
    b2 = RNG.normal(size=(8,)).astype(np.float32)
    out = gelu_mlp_reference(HIDDEN, w1, b1, w2, b2)
    x = HIDDEN @ w1.T + b1
    x = 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    np.testing.assert_allclose(out, x @ w2.T + b2, rtol=1e-5, atol=1e-6)
    assert out.shape == (2, 5, 8)


def test_layer_norm_fallback_with_residual():
    weight = RNG.normal(size=(16,)).astype(np.float32)
    bias = RNG.normal(size=(16,)).astype(np.float32)
    residual = RNG.normal(size=(2, 5, 16)).astype(np.float32)
    out = layer_norm_reference(HIDDEN, weight, bias, residual=residual, eps=1e-5)
    x = HIDDEN + residual
    x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(out, x * weight + bias, rtol=1e-4, atol=1e-6)


def test_disabled_plugin_still_computes_via_fallback():
    plugin = FusedOpsPlugin(config=FusedOpsConfig())
    try:
        q, _, _ = plugin.project_qkv(HIDDEN, (W_Q, W_K, W_V))
        np.testing.assert_allclose(q, HIDDEN @ W_Q.T, rtol=1e-5, atol=1e-6)
        assert plugin.mode == "disabled"
        assert plugin.native is False
    finally:
        plugin.close()


def test_injected_native_override_is_used():
    calls = []

    def fake_native(hidden, weights, biases):
        calls.append(hidden.shape)
        return qkv_reference(hidden, weights, biases)

    plugin = FusedOpsPlugin(config=FusedOpsConfig(enabled=True),
                            native_fn={"qkv": fake_native})
    try:
        plugin.project_qkv(HIDDEN, (W_Q, W_K, W_V))
        assert calls == [HIDDEN.shape]
        assert plugin.metadata()["calls"]["qkv"]["native"] == 1
    finally:
        plugin.close()


def test_reset_clears_counters_and_packed_cache():
    plugin = make_plugin(prefer="numpy")
    try:
        plugin.project_qkv(HIDDEN, (W_Q, W_K, W_V))
        plugin.reset()
        meta = plugin.metadata()
        assert meta["calls"]["qkv"] == {"native": 0, "fallback": 0}
        assert meta["packed_weights"] == 0
    finally:
        plugin.close()


# --------------------------------------------------------------------------- #
# Native parity (needs torch; skipped otherwise)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_qkv_native_matches_fallback():
    plugin = make_plugin()
    try:
        q_n, k_n, v_n = plugin.project_qkv(HIDDEN, (W_Q, W_K, W_V), BIASES)
        q_f, k_f, v_f = qkv_reference(HIDDEN, (W_Q, W_K, W_V), BIASES)
        for native, fallback in ((q_n, q_f), (k_n, k_f), (v_n, v_f)):
            np.testing.assert_allclose(native, fallback, rtol=1e-4, atol=1e-5)
        meta = plugin.metadata()
        assert meta["calls"]["qkv"] == {"native": 1, "fallback": 0}
        assert meta["packed_weights"] == 1
        # second call reuses the packed weight cache
        plugin.project_qkv(HIDDEN, (W_Q, W_K, W_V), BIASES)
        assert plugin.metadata()["packed_weights"] == 1
    finally:
        plugin.close()


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_attention_native_matches_fallback():
    plugin = make_plugin()
    try:
        q = RNG.normal(size=(2, 4, 8, 16)).astype(np.float32)
        k = RNG.normal(size=(2, 4, 8, 16)).astype(np.float32)
        v = RNG.normal(size=(2, 4, 8, 16)).astype(np.float32)
        np.testing.assert_allclose(
            plugin.attend(q, k, v), attention_reference(q, k, v),
            rtol=1e-4, atol=1e-5,
        )
        np.testing.assert_allclose(
            plugin.attend(q, k, v, causal=True),
            attention_reference(q, k, v, causal=True),
            rtol=1e-4, atol=1e-5,
        )
    finally:
        plugin.close()


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_gelu_mlp_and_layer_norm_native_match_fallback():
    plugin = make_plugin()
    try:
        w1 = RNG.normal(size=(32, 16)).astype(np.float32)
        b1 = RNG.normal(size=(32,)).astype(np.float32)
        w2 = RNG.normal(size=(8, 32)).astype(np.float32)
        b2 = RNG.normal(size=(8,)).astype(np.float32)
        np.testing.assert_allclose(
            plugin.gelu_mlp(HIDDEN, w1, b1, w2, b2),
            gelu_mlp_reference(HIDDEN, w1, b1, w2, b2),
            rtol=1e-4, atol=1e-5,
        )
        weight = RNG.normal(size=(16,)).astype(np.float32)
        bias = RNG.normal(size=(16,)).astype(np.float32)
        residual = RNG.normal(size=(2, 5, 16)).astype(np.float32)
        np.testing.assert_allclose(
            plugin.layer_norm(HIDDEN, weight, bias, residual=residual),
            layer_norm_reference(HIDDEN, weight, bias, residual=residual),
            rtol=1e-4, atol=1e-5,
        )
    finally:
        plugin.close()