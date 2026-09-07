"""Compatibility wrapper for the historical backend import path."""

from ..models.turbovla.tensorrt import TurboVlaTensorRtBackend, create_backend

__all__ = ["TurboVlaTensorRtBackend", "create_backend"]
