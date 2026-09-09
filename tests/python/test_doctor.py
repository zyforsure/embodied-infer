"""Tests for config-driven doctor endpoint discovery."""

from embodied_infer_deploy.cli import DOCTOR_PROFILES, _config_endpoints


def test_discovers_bare_and_prefixed_host_pairs():
    endpoints = _config_endpoints({
        "host": "127.0.0.1",
        "port": 8000,
        "s600_host": "192.168.10.10",
        "s600_port": 5702,
        "tokenizer": "/models/tokenizer.model",
    })
    assert endpoints == [
        ("service", "127.0.0.1", 8000),
        ("s600-service", "192.168.10.10", 5702),
    ]


def test_ignores_host_without_port_and_empty_values():
    assert _config_endpoints({"host": "127.0.0.1"}) == []
    assert _config_endpoints({"s600_host": "", "s600_port": 5702}) == []


def test_doctor_profiles_reference_registered_models():
    from embodied_infer_deploy.models import model_registry

    available = set(model_registry.names())
    for profile, (model, default_config, _) in DOCTOR_PROFILES.items():
        assert model in available, f"{profile} -> {model}"
        assert default_config.endswith(".json")
