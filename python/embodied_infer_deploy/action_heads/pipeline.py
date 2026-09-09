"""Standard VLA inference pipeline shared by all action paradigms.

One pipeline wires three pieces together, mirroring the unified
``forward``/``predict_action`` contract of starVLA and the staged pipeline
of vLLM-Omni:

    request --encoder_fn--> ActionContext --ActionHead.decode--> ActionChunk

The encoder is the only model-specific piece: it turns the canonical
request (images, state, instruction) into whatever conditioning the
backbone produces (token ids, embeddings, KV handles).  The head is chosen
by paradigm, and the resulting chunk is validated against the model's
``ModelSpec`` before it reaches the serving layer.  Adapting a new model
therefore means writing one encoder and picking one head -- the scheduler,
server, and safety layers never change.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from ..core import BackendResult, ModelSpec
from .autoregressive import AutoregressiveConfig, AutoregressiveTokenHead
from .base import ActionContext, ActionHead
from .dual_system import DualSystemConfig, DualSystemHead
from .flow_matching import FlowMatchingConfig, FlowMatchingHead
from .regression import ParallelRegressionHead

HEAD_KINDS = ("autoregressive", "regression", "flow_matching", "dual_system")


def create_head(
    kind: str,
    fns: Mapping[str, Callable[..., Any]],
    config: Mapping[str, Any] | None = None,
) -> ActionHead:
    """Build one action head from a paradigm name and native callables.

    ``fns`` carries the backend-native callables: ``step`` / ``detokenize``
    for autoregressive, ``regress`` for regression, ``velocity`` for flow
    matching, and ``plan`` plus a nested ``fast`` config for dual system.
    """

    config = config or {}
    if kind == "autoregressive":
        return AutoregressiveTokenHead(
            step_fn=fns.get("step"),
            detokenize_fn=fns.get("detokenize"),
            config=AutoregressiveConfig.from_mapping(config),
        )
    if kind == "regression":
        return ParallelRegressionHead(regress_fn=fns.get("regress"))
    if kind == "flow_matching":
        return FlowMatchingHead(
            velocity_fn=fns.get("velocity"),
            config=FlowMatchingConfig.from_mapping(config),
        )
    if kind == "dual_system":
        fast_config = dict(config.get("fast", {"kind": "flow_matching"}))
        fast_kind = str(fast_config.pop("kind", "flow_matching"))
        fast_head = create_head(fast_kind, fns, fast_config)
        return DualSystemHead(
            plan_fn=fns.get("plan"),
            fast_head=fast_head,
            config=DualSystemConfig.from_mapping(config),
        )
    raise ValueError(
        f"unknown action head kind {kind!r}; available: {', '.join(HEAD_KINDS)}"
    )


class VLAPipeline:
    """Encode -> decode -> validate pipeline for one model backend."""

    def __init__(
        self,
        spec: ModelSpec,
        head: ActionHead,
        encoder_fn: Callable[[dict[str, Any]], ActionContext],
    ) -> None:
        if not callable(encoder_fn):
            raise TypeError("VLAPipeline requires a callable encoder_fn")
        self._spec = spec
        self.head = head
        self.encoder_fn = encoder_fn
        self.inference_count = 0

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def infer(self, request: dict[str, Any]) -> BackendResult:
        self._spec.validate_request(request)
        started = time.perf_counter()
        context = self.encoder_fn(request)
        if not isinstance(context, dict):
            raise TypeError("encoder_fn must return an ActionContext dict")
        encoded = time.perf_counter()
        actions = self.head.decode(context, self._spec)
        finished = time.perf_counter()
        self.inference_count += 1
        return BackendResult(
            actions=actions,
            representation=self._spec.action_representation,
            control_period_ns=self._spec.control_period_ns,
            timing={
                "encode_ms": (encoded - started) * 1000.0,
                "decode_ms": (finished - encoded) * 1000.0,
            },
        )

    def reset(self) -> None:
        self.head.reset()

    def metadata(self) -> dict[str, Any]:
        return {
            **self._spec.metadata(),
            "action_head": self.head.metadata(),
            "inference_count": self.inference_count,
        }


__all__ = ["HEAD_KINDS", "VLAPipeline", "create_head"]
