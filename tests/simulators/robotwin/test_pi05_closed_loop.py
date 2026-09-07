import numpy as np

from embodied_infer_deploy.models.pi05 import remote as pi05_remote
from embodied_infer_deploy.models.pi05.remote import Pi05RemoteBackend
from embodied_infer_deploy.simulators.robotwin import (
    RemoteRoboTwinPolicy,
    RoboTwinAdapter,
    create_demo_env,
    flatten_qpos_action,
    run_episode,
)


class _FakePi05Connection:
    def __init__(self):
        self.pending = [pi05_remote._pack({"policy": "pi05-test"})]
        self.observations = []

    def send(self, payload):
        observation = pi05_remote._unpack(payload)
        self.observations.append(observation)
        state = np.asarray(observation["state"], dtype=np.float32)
        actions = np.repeat(state[None, :], 15, axis=0)
        self.pending.append(pi05_remote._pack({"actions": actions}))

    def recv(self, timeout=None):
        return self.pending.pop(0)

    def close(self):
        return None


class _BackendClient:
    def __init__(self, backend):
        self.backend = backend

    def reset(self):
        self.backend.reset()

    def infer(self, *, instruction, images, state, **_kwargs):
        result = self.backend.infer({
            "instruction": instruction,
            "images": {name: image for name, image, _timestamp in images},
            "state": state,
        })
        return {"actions": result.actions}


def test_pi05_backend_runs_complete_robotwin_demo_loop(monkeypatch):
    connection = _FakePi05Connection()
    monkeypatch.setattr(pi05_remote, "connect", lambda *args, **kwargs: connection)
    backend = Pi05RemoteBackend({})
    adapter = RoboTwinAdapter(_BackendClient(backend))
    policy = RemoteRoboTwinPolicy(adapter, exec_horizon=2)
    environment = create_demo_env(episode_steps=4, image_size=12)

    executed = run_episode(environment, policy, action_transform=flatten_qpos_action)

    assert executed == 4
    assert len(connection.observations) == 2
    assert connection.observations[0]["state"].shape == (14,)
    assert connection.observations[0]["images"]["cam_high"].shape == (3, 12, 12)
    assert all(action.shape == (16,) for action in environment.actions)
    assert all(np.isfinite(action).all() for action in environment.actions)
