"""TurboVLA gateway backend for the S100 HBM service.

S100 exposes the same four-stage TurboVLA HBM wire contract as the existing
S600 deployment.  The distinction is kept in the registry and metadata so a
deployment can be selected explicitly, while the proven proxy implementation
continues to own tokenization, DINO preprocessing, normalization, and the
length-prefixed HBM transport.
"""

from __future__ import annotations

from typing import Any

from ...core import ModelSpec
from .common import turbovla_spec
from .s600_remote import S600HbmRemoteBackend


class S100HbmRemoteBackend(S600HbmRemoteBackend):
    def __init__(self, config: dict[str, Any]) -> None:
        # The historical proxy names its endpoint arguments s600_host/s600_port;
        # translate the explicit S100 config without duplicating its protocol
        # and preprocessing implementation.
        translated = dict(config)
        translated["s600_host"] = config.get("s100_host", config.get("s600_host"))
        translated["s600_port"] = int(config.get("s100_port", config.get("s600_port", 5702)))
        translated["model_name"] = config.get("model_name", "TurboVLA-S100")
        super().__init__(translated)
        self._spec = turbovla_spec(
            "s100-hbm-remote", translated, default_period_ns=100_000_000
        )


def create_backend(config: dict[str, Any]) -> S100HbmRemoteBackend:
    return S100HbmRemoteBackend(config)


__all__ = ["S100HbmRemoteBackend", "create_backend"]
