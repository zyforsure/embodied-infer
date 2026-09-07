import numpy as np

from embodied_infer_deploy.simulators.robotwin import create_demo_env


def test_demo_environment_is_download_and_run_ready():
    env = create_demo_env(episode_steps=1, image_size=8)
    obs = env.reset()
    assert obs["joint_action"]["vector"].shape == (18,)
    assert obs["observation"]["head_camera"]["rgb"].shape == (8, 8, 3)
    env.take_action(np.zeros(16, dtype=np.float32))
    assert env.is_episode_end()
    env.close()
