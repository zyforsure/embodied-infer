"""Contract tests for the cross-embodiment robot registry."""

import numpy as np
import pytest

from embodied_infer_deploy.robots import (
    DAZZ_S600_CONTRACT,
    RawRobotContract,
    RobotRegistry,
    get_robot_contract,
    robot_registry,
)


MOBILE_10D_CONFIG = {
    "raw_state_dim": 10,
    "raw_action_dim": 10,
    "model_state_indices": [0, 1, 2, 3, 4, 5, 8],
    "left_arm_slice": [0, 4],
    "left_gripper_slice": [4, 5],
    "right_arm_slice": [5, 9],
    "right_gripper_slice": [9, 10],
}


def test_builtin_dazz_s600_is_default():
    assert get_robot_contract() is DAZZ_S600_CONTRACT
    assert get_robot_contract("dazz-s600") is DAZZ_S600_CONTRACT
    assert "dazz-s600" in robot_registry.names()


def test_register_config_builds_working_contract():
    registry = RobotRegistry()
    contract = registry.register_config("mobile-10d", MOBILE_10D_CONFIG)

    state = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(
        contract.state_to_model(state), [0, 1, 2, 3, 4, 5, 8]
    )
    assert contract.held_indices == (6, 7, 9)
    assert contract.command_action_dim == 10

    model_actions = np.ones((3, 7), dtype=np.float32)
    raw = contract.actions_to_raw(model_actions, state)
    assert raw.shape == (3, 10)
    np.testing.assert_array_equal(raw[0, [6, 7, 9]], state[[6, 7, 9]])
    np.testing.assert_array_equal(raw[0, [0, 1, 2, 3, 4, 5, 8]], np.ones(7))


def test_register_config_rejects_invalid_contracts():
    registry = RobotRegistry()
    with pytest.raises(ValueError, match="unique"):
        registry.register_config(
            "dup", {**MOBILE_10D_CONFIG, "model_state_indices": [0, 0, 2, 3, 4, 5, 8]}
        )
    with pytest.raises(ValueError, match="out of range"):
        registry.register_config(
            "oob", {**MOBILE_10D_CONFIG, "model_state_indices": [0, 1, 2, 3, 4, 5, 42]}
        )
    with pytest.raises(ValueError, match="start < stop"):
        registry.register_config(
            "badslice", {**MOBILE_10D_CONFIG, "left_arm_slice": [4, 4]}
        )
    with pytest.raises(ValueError, match="requires left_arm_slice"):
        registry.register_config(
            "missing", {key: value for key, value in MOBILE_10D_CONFIG.items()
                        if key != "left_arm_slice"}
        )


def test_duplicate_registration_requires_replace():
    registry = RobotRegistry()
    registry.register_config("mobile-10d", MOBILE_10D_CONFIG)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_config("mobile-10d", MOBILE_10D_CONFIG)
    updated = {**MOBILE_10D_CONFIG, "model_state_indices": [0, 1, 2, 3, 4, 5, 6]}
    contract = registry.register_config("mobile-10d", updated, replace=True)
    assert contract.model_state_indices == (0, 1, 2, 3, 4, 5, 6)


def test_unknown_robot_lists_available_names():
    registry = RobotRegistry()
    registry.register("dazz-s600", DAZZ_S600_CONTRACT)
    with pytest.raises(KeyError, match="available: dazz-s600"):
        registry.resolve("atlas")


def test_lazy_factory_target_resolves_once():
    calls = []
    registry = RobotRegistry()
    registry.register(
        "lazy",
        lambda: calls.append(1) or RawRobotContract.from_mapping(MOBILE_10D_CONFIG),
    )
    first = registry.resolve("lazy")
    second = registry.resolve("lazy")
    assert first is second
    assert calls == [1]


def test_registry_rejects_bad_targets():
    registry = RobotRegistry()
    with pytest.raises(ValueError, match="invalid robot registry name"):
        registry.register("Bad Name", DAZZ_S600_CONTRACT)
    with pytest.raises(ValueError, match="RawRobotContract"):
        registry.register("bad-target", 42)
    registry.register("wrong-type", lambda: {"not": "a contract"})
    with pytest.raises(TypeError, match="invalid contract"):
        registry.resolve("wrong-type")
