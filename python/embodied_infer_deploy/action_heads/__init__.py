"""Pluggable action-generation paradigms for VLA models.

Four paradigms cover the open-source VLA landscape, following the unified
forward/predict contract used by starVLA and the staged pipeline model of
vLLM-Omni:

- ``AutoregressiveTokenHead``: PI0-FAST style discrete action tokens with a
  pluggable detokenizer.
- ``ParallelRegressionHead``: OpenVLA-OFT style one-shot continuous
  regression of the whole action chunk.
- ``FlowMatchingHead``: Pi0 / Pi0.5 style flow-matching integration from
  noise to a continuous action distribution.
- ``DualSystemHead``: GR00T / SmolVLA style hierarchical System2 planner +
  System1 fast executor.

A backend builds one :class:`VLAPipeline` from a ``ModelSpec``, an encoder
callable, and one head; the serving and scheduling layers never see which
paradigm is in use.
"""

from .base import ActionContext, ActionHead, ActionHeadUnavailableError
from .autoregressive import AutoregressiveConfig, AutoregressiveTokenHead
from .regression import ParallelRegressionHead
from .flow_matching import FlowMatchingConfig, FlowMatchingHead
from .dual_system import DualSystemConfig, DualSystemHead
from .pipeline import HEAD_KINDS, VLAPipeline, create_head

__all__ = [
    "ActionContext",
    "ActionHead",
    "ActionHeadUnavailableError",
    "AutoregressiveConfig",
    "AutoregressiveTokenHead",
    "ParallelRegressionHead",
    "FlowMatchingConfig",
    "FlowMatchingHead",
    "DualSystemConfig",
    "DualSystemHead",
    "HEAD_KINDS",
    "VLAPipeline",
    "create_head",
]
