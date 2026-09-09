"""Contract tests for the four pluggable VLA action paradigms."""

import numpy as np
import pytest

from embodied_infer_deploy.action_heads import (
    ActionHeadUnavailableError,
    AutoregressiveConfig,
    AutoregressiveTokenHead,
    DualSystemConfig,
    DualSystemHead,
    FlowMatchingConfig,
    FlowMatchingHead,
    ParallelRegressionHead,
    VLAPipeline,
    create_head,
)
from embodied_infer_deploy.core import ModelSpec


def make_spec() -> ModelSpec:
    return ModelSpec(
        name="unit-test-vla",
        backend="test",
        raw_state_dim=14,
        model_state_dim=14,
        raw_action_dim=14,
        model_action_dim=14,
        action_horizon=8,
        camera_order=("head",),
        control_period_ns=20_000_000,
    )


SPEC = make_spec()


# --------------------------------------------------------------------------- #
# Autoregressive (PI0-FAST style)
# --------------------------------------------------------------------------- #

def test_autoregressive_decodes_until_eos():
    script = iter([10, 11, 12, 99])

    def step_fn(context, tokens):
        return next(script)

    def detokenize_fn(tokens, spec):
        assert tokens == [10, 11, 12]
        return np.full((spec.action_horizon, spec.model_action_dim), 0.5)

    head = AutoregressiveTokenHead(
        step_fn=step_fn,
        detokenize_fn=detokenize_fn,
        config=AutoregressiveConfig(max_tokens=60, eos_token=99),
    )
    actions = head.decode({}, SPEC)
    assert actions.shape == (8, 14)
    assert head.last_token_count == 3


def test_autoregressive_respects_max_tokens_without_eos():
    def step_fn(context, tokens):
        return len(tokens)

    head = AutoregressiveTokenHead(
        step_fn=step_fn,
        detokenize_fn=lambda tokens, spec: np.zeros(
            (spec.action_horizon, spec.model_action_dim), dtype=np.float32
        ),
        config=AutoregressiveConfig(max_tokens=5),
    )
    head.decode({}, SPEC)
    assert head.last_token_count == 5


def test_autoregressive_requires_native_callables():
    head = AutoregressiveTokenHead()
    with pytest.raises(ActionHeadUnavailableError):
        head.decode({}, SPEC)


def test_autoregressive_rejects_empty_token_stream():
    head = AutoregressiveTokenHead(
        step_fn=lambda context, tokens: None,
        detokenize_fn=lambda tokens, spec: np.zeros(
            (SPEC.action_horizon, SPEC.model_action_dim), dtype=np.float32
        ),
    )
    with pytest.raises(ValueError, match="no action tokens"):
        head.decode({}, SPEC)


# --------------------------------------------------------------------------- #
# Parallel regression (OpenVLA-OFT style)
# --------------------------------------------------------------------------- #

def test_regression_single_shot_chunk():
    calls = []

    def regress_fn(context, spec):
        calls.append(context["instruction"])
        return np.linspace(
            0.0, 1.0, spec.action_horizon * spec.model_action_dim
        ).reshape(spec.action_horizon, spec.model_action_dim)

    head = ParallelRegressionHead(regress_fn=regress_fn)
    actions = head.decode({"instruction": "pick"}, SPEC)
    assert actions.shape == (8, 14)
    assert calls == ["pick"]  # one parallel forward, no temporal loop


def test_regression_requires_regress_fn():
    with pytest.raises(ActionHeadUnavailableError):
        ParallelRegressionHead().decode({}, SPEC)


def test_regression_validates_chunk_shape():
    head = ParallelRegressionHead(regress_fn=lambda context, spec: np.zeros((4, 14)))
    with pytest.raises(ValueError, match="invalid action chunk"):
        head.decode({}, SPEC)


# --------------------------------------------------------------------------- #
# Flow matching (Pi0 / Pi0.5 style)
# --------------------------------------------------------------------------- #

def test_flow_matching_integrates_velocity_field():
    steps_seen = []

    def velocity_fn(context, actions, t):
        steps_seen.append(round(t, 6))
        return np.ones_like(actions)

    head = FlowMatchingHead(
        velocity_fn=velocity_fn,
        config=FlowMatchingConfig(num_steps=10, seed=0),
    )
    actions = head.decode({}, SPEC)
    assert actions.shape == (8, 14)
    assert steps_seen == [round(i / 10, 6) for i in range(10)]
    # Constant unit velocity integrates exactly to noise + 1.
    noise = np.random.default_rng(0).normal(0.0, 1.0, (8, 14)).astype(np.float32)
    np.testing.assert_allclose(actions, noise + 1.0, rtol=1e-5, atol=1e-6)


def test_flow_matching_seed_is_deterministic():
    velocity_fn = lambda context, actions, t: -actions
    config = FlowMatchingConfig(num_steps=4, seed=7)
    first = FlowMatchingHead(velocity_fn=velocity_fn, config=config).decode({}, SPEC)
    second = FlowMatchingHead(velocity_fn=velocity_fn, config=config).decode({}, SPEC)
    np.testing.assert_array_equal(first, second)


def test_flow_matching_requires_velocity_fn():
    with pytest.raises(ActionHeadUnavailableError):
        FlowMatchingHead().decode({}, SPEC)


# --------------------------------------------------------------------------- #
# Dual system (GR00T / SmolVLA style)
# --------------------------------------------------------------------------- #

def test_dual_system_caches_system2_latent():
    plans = []
    latents_seen = []

    def plan_fn(context):
        plans.append(context["instruction"])
        return {"subgoal": context["instruction"]}

    def regress_fn(context, spec):
        latents_seen.append(context["system2_latent"])
        return np.zeros((spec.action_horizon, spec.model_action_dim), dtype=np.float32)

    clock = iter([0.0, 0.1, 0.2, 0.3])
    head = DualSystemHead(
        plan_fn=plan_fn,
        fast_head=ParallelRegressionHead(regress_fn=regress_fn),
        config=DualSystemConfig(refresh_interval_s=1.0),
        clock=lambda: next(clock),
    )
    head.decode({"instruction": "pick"}, SPEC)
    head.decode({"instruction": "pick"}, SPEC)
    assert plans == ["pick"]  # System2 replanned only once
    assert latents_seen[0] is latents_seen[1]
    assert head.metadata()["system2_calls"] == 1


def test_dual_system_replans_on_instruction_change():
    plans = []
    clock = iter([0.0, 0.1, 0.2, 0.3])
    head = DualSystemHead(
        plan_fn=lambda context: plans.append(context["instruction"]) or 1,
        fast_head=ParallelRegressionHead(
            regress_fn=lambda context, spec: np.zeros(
                (spec.action_horizon, spec.model_action_dim), dtype=np.float32
            )
        ),
        config=DualSystemConfig(refresh_interval_s=60.0),
        clock=lambda: next(clock),
    )
    head.decode({"instruction": "pick"}, SPEC)
    head.decode({"instruction": "place"}, SPEC)
    assert plans == ["pick", "place"]


def test_dual_system_reset_clears_latent():
    plans = []
    clock = iter([0.0, 0.1, 0.2, 0.3])
    head = DualSystemHead(
        plan_fn=lambda context: plans.append(1) or 1,
        fast_head=ParallelRegressionHead(
            regress_fn=lambda context, spec: np.zeros(
                (spec.action_horizon, spec.model_action_dim), dtype=np.float32
            )
        ),
        clock=lambda: next(clock),
    )
    head.decode({"instruction": "pick"}, SPEC)
    head.reset()
    head.decode({"instruction": "pick"}, SPEC)
    assert plans == [1, 1]


# --------------------------------------------------------------------------- #
# Pipeline and factory
# --------------------------------------------------------------------------- #

def make_request() -> dict:
    return {
        "state": np.zeros(14, dtype=np.float32),
        "images": {"head": np.zeros((4, 4, 3), dtype=np.uint8)},
        "instruction": "pick",
    }


def test_pipeline_runs_encode_decode_validate():
    head = create_head(
        "regression",
        {"regress": lambda context, spec: np.full(
            (spec.action_horizon, spec.model_action_dim), 0.25, dtype=np.float32
        )},
    )
    pipeline = VLAPipeline(
        SPEC,
        head,
        encoder_fn=lambda request: {"instruction": request["instruction"]},
    )
    result = pipeline.infer(make_request())
    assert result.actions.shape == (8, 14)
    assert result.timing["encode_ms"] >= 0.0
    assert result.timing["decode_ms"] >= 0.0
    metadata = pipeline.metadata()
    assert metadata["action_head"]["paradigm"] == "regression"
    assert metadata["inference_count"] == 1


def test_pipeline_validates_request_before_encoding():
    pipeline = VLAPipeline(
        SPEC,
        ParallelRegressionHead(regress_fn=lambda context, spec: np.zeros((8, 14))),
        encoder_fn=lambda request: {},
    )
    bad = make_request()
    bad["state"] = np.zeros(7, dtype=np.float32)
    with pytest.raises(ValueError, match="model state"):
        pipeline.infer(bad)


def test_create_head_builds_dual_system_with_nested_fast_head():
    head = create_head(
        "dual_system",
        {
            "plan": lambda context: "latent",
            "velocity": lambda context, actions, t: np.zeros_like(actions),
        },
        {
            "refresh_interval_s": 0.5,
            "fast": {"kind": "flow_matching", "num_steps": 2, "seed": 0},
        },
    )
    assert isinstance(head, DualSystemHead)
    assert isinstance(head.fast_head, FlowMatchingHead)
    actions = head.decode({"instruction": "pick"}, SPEC)
    assert actions.shape == (8, 14)


def test_create_head_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown action head kind"):
        create_head("diffusion", {})
