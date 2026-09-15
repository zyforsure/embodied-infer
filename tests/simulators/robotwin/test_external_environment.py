"""Opt-in observation smoke test against an installed RoboTwin environment."""

import importlib
import json
import os

import pytest

from embodied_infer_deploy.simulators.robotwin import RoboTwinAdapter


class _NoInferenceClient:
    pass


def _load_factory(target):
    module_name, attribute = target.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


@pytest.mark.robotwin
def test_external_robotwin_observation_contract():
    target = os.getenv("EMBODIED_INFER_ROBOTWIN_FACTORY")
    if not target:
        pytest.skip("set EMBODIED_INFER_ROBOTWIN_FACTORY=module:factory to enable")
    kwargs = json.loads(os.getenv("EMBODIED_INFER_ROBOTWIN_KWARGS", "{}"))
    environment = _load_factory(target)(**kwargs)
    try:
        environment.reset()
        observation = environment.get_obs()
        adapter = RoboTwinAdapter(_NoInferenceClient())
        assert adapter.extract_state(observation).shape == (18,)
        assert [name for name, _, _ in adapter.extract_images(observation)] == [
            "head",
            "left_wrist",
            "right_wrist",
        ]
    finally:
        close = getattr(environment, "close", None)
        if close is not None:
            close()
