"""Compatibility wrapper for the historical backend import path."""

from ..models.turbovla.s600_remote import S600HbmRemoteBackend, create_backend

__all__ = ["S600HbmRemoteBackend", "create_backend"]
