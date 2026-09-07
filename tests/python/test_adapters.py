import unittest

import numpy as np

from embodied_infer_deploy.adapters.robotwin import (
    MODEL_STATE_INDEX,
    RobotTwinAdapter,
)
from embodied_infer_deploy.adapters.s600 import S600Adapter


class AdapterTests(unittest.TestCase):
    def test_robotwin_maps_18d_state_and_holds_unmodelled_coordinates(self):
        raw = np.arange(18, dtype=np.float32)
        model = RobotTwinAdapter.state_to_model_order(raw)
        np.testing.assert_array_equal(model, raw[MODEL_STATE_INDEX])

        predicted = model + 100
        restored = RobotTwinAdapter.actions_to_raw_order(predicted[None], raw)[0]
        np.testing.assert_array_equal(restored[MODEL_STATE_INDEX], predicted)
        held = np.ones(18, dtype=bool)
        held[MODEL_STATE_INDEX] = False
        np.testing.assert_array_equal(restored[held], raw[held])

    def test_s600_safety_clamps_joints_and_holds_grippers(self):
        state = np.zeros(18, dtype=np.float32)
        state[[7, 15]] = [1, 0]
        state[[6, 14, 16, 17]] = [6, 14, 16, 17]
        action = np.ones(18, dtype=np.float32)
        checked = S600Adapter.clamp_first_action(
            action, state, max_step_deg=1.0, enable_gripper=False
        )
        np.testing.assert_allclose(
            checked[MODEL_STATE_INDEX[:12]], np.deg2rad(1.0), rtol=1e-6
        )
        np.testing.assert_array_equal(checked[[7, 15]], state[[7, 15]])
        np.testing.assert_array_equal(
            checked[[6, 14, 16, 17]], state[[6, 14, 16, 17]]
        )

    def test_s600_rejects_wrong_state(self):
        with self.assertRaises(ValueError):
            S600Adapter.validate_raw_state(np.zeros(14, dtype=np.float32))

    def test_robotwin_action_dict_uses_seven_joint_arms(self):
        action = RobotTwinAdapter.action_dict(np.arange(18, dtype=np.float32))
        np.testing.assert_array_equal(action["left_arm_joint_state"], np.arange(7))
        np.testing.assert_array_equal(action["left_ee_joint_state"], [7])
        np.testing.assert_array_equal(action["right_arm_joint_state"], np.arange(8, 15))
        np.testing.assert_array_equal(action["right_ee_joint_state"], [15])
        np.testing.assert_array_equal(action["raw_action"], np.arange(18))


if __name__ == "__main__":
    unittest.main()
