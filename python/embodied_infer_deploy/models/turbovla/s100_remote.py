"""TurboVLA gateway backend for the S100 HBM service.

S100 exposes the same four-stage TurboVLA HBM wire contract as the existing
S600 deployment, so the hardware generation is pure identity: registry name,
metadata, and defaults.  Endpoint configuration resolves through the shared
host/port convention (``host``/``port``, ``s100_host``/``s100_port``, or the
legacy ``s600_host`` pair); no key translation is required.
"""

from __future__ import annotations

from typing import Any

from .s600_remote import S600HbmRemoteBackend


class S100HbmRemoteBackend(S600HbmRemoteBackend):
    backend_name = "s100-hbm-remote"
    default_model_name = "TurboVLA-S100"
    default_hardware = "s100"


def create_backend(config: dict[str, Any]) -> S100HbmRemoteBackend:
    return S100HbmRemoteBackend(config)


__all__ = ["S100HbmRemoteBackend", "create_backend"]
