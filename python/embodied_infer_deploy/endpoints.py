"""Config-driven network endpoint discovery.

Backends and the doctor CLI share one convention: an endpoint is either a
bare ``host``/``port`` pair or a prefixed ``<stem>_host``/``<stem>_port``
pair (``s600_host``, ``s100_host``, ...).  Hardware generations differ only
in the stem, so new boards plug in without key-translation shims.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def discover_endpoints(config: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """Return (stem, host, port) for every host/port pair in the config."""

    endpoints: list[tuple[str, str, int]] = []
    for key, value in sorted(config.items()):
        if key != "host" and not key.endswith("_host"):
            continue
        stem = "" if key == "host" else key[: -len("_host")]
        port = config.get(f"{stem}_port" if stem else "port")
        if value and port is not None:
            endpoints.append((stem, str(value), int(port)))
    return endpoints


def resolve_endpoint(
    config: Mapping[str, Any], *, default_port: int
) -> tuple[str, int]:
    """Resolve the single upstream endpoint a backend should dial.

    Prefers bare ``host``/``port`` and falls back to the first prefixed
    pair, so legacy ``s600_host``/``s100_host`` configs keep working.
    """

    endpoints = discover_endpoints(config)
    if not endpoints:
        if any(key.endswith("_host") for key in config):
            return str(
                next(config[key] for key in sorted(config) if key.endswith("_host"))
            ), default_port
        raise ValueError("config requires host/port or a *_host/*_port pair")
    _, host, port = endpoints[0]
    return host, port


__all__ = ["discover_endpoints", "resolve_endpoint"]
