import numpy as np

from embodied_infer_deploy.robots import DAZZ_S600_CONTRACT
from embodied_infer_deploy.simulators.robotwin import (
    RemoteRoboTwinPolicy,
    RoboTwinAdapter,
    flatten_qpos_action,
    run_episode,
)


class FakeInferenceClient:
    def __init__(self):
        self.reset_count = 0
        self.requests = []

    def reset(self):
        self.reset_count += 1

    def infer(self, **request):
        self.requests.append(request)
        actions = np.repeat((request["state"] + 0.25)[None, :], 4, axis=0)
        return {"actions": actions.astype(np.float32), "request_id": 1}


class FakeRoboTwinEnvironment:
    def __init__(self, episode_steps=3):
        self.episode_steps = episode_steps
        self.steps = 0
        self.actions = []
        self.raw_state = np.arange(18, dtype=np.float32)

    def reset(self):
        self.steps = 0
        self.actions.clear()

    def is_episode_end(self):
        return self.steps >= self.episode_steps

    def get_obs(self):
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        return {
            "instruction": "pick up the block",
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            },
            "joint_action": {"vector": self.raw_state.copy()},
        }

    def take_action(self, action):
        self.actions.append(action)
        self.steps += 1


def test_robotwin_closed_loop_uses_18d_boundary_and_14d_model_request():
    client = FakeInferenceClient()
    adapter = RoboTwinAdapter(client, default_instruction="fallback")
    policy = RemoteRoboTwinPolicy(adapter, exec_horizon=2, timeout=1.0)
    environment = FakeRoboTwinEnvironment(episode_steps=3)

    executed = run_episode(
        environment, policy, action_transform=flatten_qpos_action
    )

    assert executed == 3
    assert client.reset_count == 1
    assert len(client.requests) == 2
    assert all(request["state"].shape == (14,) for request in client.requests)
    assert all(len(request["images"]) == 3 for request in client.requests)
    assert all(action.shape == (16,) for action in environment.actions)

    first_raw = adapter.actions_to_raw_order(
        client.requests[0]["state"][None] + 0.25,
        environment.raw_state,
    )[0]
    np.testing.assert_array_equal(
        first_raw[list(DAZZ_S600_CONTRACT.held_indices)],
        environment.raw_state[list(DAZZ_S600_CONTRACT.held_indices)],
    )


def test_runner_rejects_empty_action_chunks():
    class EmptyPolicy:
        def reset(self):
            pass

        def update_obs(self, observation):
            pass

        def get_action(self):
            return []

    environment = FakeRoboTwinEnvironment(episode_steps=1)
    try:
        run_episode(environment, EmptyPolicy())
    except RuntimeError as exc:
        assert "empty action chunk" in str(exc)
    else:
        raise AssertionError("empty action chunks must fail")
