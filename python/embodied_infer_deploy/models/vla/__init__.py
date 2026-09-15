"""Thin remote backends for VLA families served from a GPU gateway."""

from .backend import VlaRemoteBackend, create_backend

__all__ = ["VlaRemoteBackend", "create_backend"]
