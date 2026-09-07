from __future__ import annotations

"""Compatibility imports; use embodied_infer_deploy.core for new code."""

from ..core import BackendResult, ModelBackend

Backend = ModelBackend

__all__ = ["Backend", "BackendResult"]
