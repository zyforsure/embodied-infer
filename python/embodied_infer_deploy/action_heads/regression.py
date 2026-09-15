"""Parallel continuous action regression (OpenVLA-OFT / starVLA QwenOFT).

A lightweight MLP action head regresses the entire action chunk in one
forward pass from pretrained action-token embeddings.  There is no temporal
dependency between steps, so every timestep of the chunk decodes in
parallel; this avoids discretization loss and scales to high-dimensional
action spaces.

The backend injects one native callable:

- ``regress_fn(context, spec) -> np.ndarray``: return the full
  ``[action_horizon, model_action_dim]`` chunk in a single call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..core import ModelSpec
from .base import ActionContext, ActionHeadUnavailableError, validate_chunk


class ParallelRegressionHead:
    """OpenVLA-OFT style one-shot regression of the full action chunk."""

    def __init__(
        self,
        regress_fn: Callable[[ActionContext, ModelSpec], Any] | None = None,
    ) -> None:
        self.regress_fn = regress_fn

    @property
    def paradigm(self) -> str:
        return "regression"

    def decode(self, context: ActionContext, spec: ModelSpec) -> np.ndarray:
        if self.regress_fn is None:
            raise ActionHeadUnavailableError(
                "regression head requires regress_fn"
            )
        actions = self.regress_fn(context, spec)
        return validate_chunk(actions, spec, owner="regression head")

    def reset(self) -> None:
        return None

    def metadata(self) -> dict[str, Any]:
        return {"paradigm": self.paradigm}


__all__ = ["ParallelRegressionHead"]
