from __future__ import annotations


def eval_one_episode(task_env, model_client):
    model_client.call(func_name="reset")
    while not task_env.is_episode_end():
        model_client.call(func_name="update_obs", obs=task_env.get_obs())
        actions = model_client.call(func_name="get_action")
        for action in actions:
            task_env.take_action(action)
            if task_env.is_episode_end():
                break


def eval_one_episode_batch(task_env, model_client):
    model_client.call(func_name="reset")
    while not task_env.is_episode_end():
        indices = task_env.get_running_env_idx_list()
        model_client.call(
            func_name="update_obs_batch", obs=task_env.get_obs_batch(indices)
        )
        chunks = model_client.call(
            func_name="get_action_batch", env_idx_list=indices
        )
        for step in range(len(chunks[0])):
            task_env.take_action_batch([chunk[step] for chunk in chunks], indices)
            if task_env.is_episode_end():
                break
