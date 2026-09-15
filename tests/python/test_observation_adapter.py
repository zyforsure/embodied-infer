"""Contract tests for the simulator observation adapter boundary."""

import numpy as np
import pytest

from embodied_infer_deploy.robots import DAZZ_S600_CONTRACT
from embodied_infer_deploy.simulators import ObservationAdapter
from embodied_infer_deploy.simulators.robotwin import (
    RoboTwinAdapter,
    flatten_qpos_action,
)


def make_observation(**cameras) -> dict:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "observation": {name: {"rgb": image} for name in cameras},
        "joint_action": {"vector": np.zeros(18, dtype=np.float32)},
    }


def test_adapter_satisfies_observation_protocol():
    adapter = RoboTwinAdapter(client=object())
    assert isinstance(adapter, ObservationAdapter)


def test_custom_camera_aliases_without_source_changes():
    adapter = RoboTwinAdapter(
        client=object(),
        camera_aliases={
            "head": ("front",),
            "left_wrist": ("wrist_l",),
            "right_wrist": ("wrist_r",),
        },
    )
    images = adapter.extract_images(
        make_observation(front=None, wrist_l=None, wrist_r=None)
    )
    assert [name for name, _, _ in images] == ["head", "left_wrist", "right_wrist"]


def test_missing_camera_reports_tried_aliases():
    adapter = RoboTwinAdapter(client=object(), camera_aliases={"head": ("front",)})
    with pytest.raises(KeyError, match="front"):
        adapter.extract_images(make_observation(other=None))


def test_qpos_fields_are_single_source_of_truth():
    action = DAZZ_S600_CONTRACT.action_dict(np.arange(18, dtype=np.float32))
    assert set(DAZZ_S600_CONTRACT.qpos_fields) <= set(action)
    flat = flatten_qpos_action(action)
    expected = np.concatenate(
        [action[name] for name in DAZZ_S600_CONTRACT.qpos_fields]
    )
    np.testing.assert_array_equal(flat, expected)


def test_flatten_qpos_action_reports_contract_fields():
    with pytest.raises(KeyError, match="left_arm_joint_state"):
        flatten_qpos_action({"raw_action": np.zeros(18)})
