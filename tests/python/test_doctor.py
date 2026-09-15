"""Tests for config-driven doctor endpoint discovery."""

from embodied_infer_deploy.cli import DOCTOR_PROFILES
from embodied_infer_deploy.endpoints import discover_endpoints, resolve_endpoint


def test_discovers_bare_and_prefixed_host_pairs():
    endpoints = discover_endpoints({
        "host": "127.0.0.1",
        "port": 8000,
        "s600_host": "192.168.10.10",
        "s600_port": 5702,
        "tokenizer": "/models/tokenizer.model",
    })
    assert endpoints == [
        ("", "127.0.0.1", 8000),
        ("s600", "192.168.10.10", 5702),
    ]


def test_ignores_host_without_port_and_empty_values():
    assert discover_endpoints({"host": "127.0.0.1"}) == []
    assert discover_endpoints({"s600_host": "", "s600_port": 5702}) == []


def test_resolve_endpoint_accepts_any_hardware_stem():
    assert resolve_endpoint(
        {"s100_host": "192.168.10.20", "s100_port": 5703}, default_port=5702
    ) == ("192.168.10.20", 5703)
    assert resolve_endpoint(
        {"s600_host": "192.168.10.10"}, default_port=5702
    ) == ("192.168.10.10", 5702)
    assert resolve_endpoint(
        {"host": "10.0.0.1", "port": 9000, "s600_host": "192.168.10.10",
         "s600_port": 5702},
        default_port=5702,
    ) == ("10.0.0.1", 9000)


def test_doctor_profiles_reference_registered_models():
    from embodied_infer_deploy.models import model_registry

    available = set(model_registry.names())
    for profile, (model, default_config, _) in DOCTOR_PROFILES.items():
        assert model in available, f"{profile} -> {model}"
        assert default_config.endswith(".json")
