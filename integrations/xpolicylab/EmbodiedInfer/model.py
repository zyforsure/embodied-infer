from __future__ import annotations

import os

from XPolicyLab.model_template import ModelTemplate

from embodied_infer_deploy.adapters import RobotTwinAdapter
from embodied_infer_deploy.client import InferenceClient


class Model(ModelTemplate):
    """XPolicyLab model plugin for an embodied-infer remote backend."""

    def __init__(self, model_cfg):
        self.config = model_cfg
        api_key_env = model_cfg.get("api_key_env", "EMBODIED_INFER_API_KEY")
        self.client = InferenceClient(
            model_cfg.get("host", "192.168.10.162"),
            int(model_cfg.get("port", 44091)),
            api_key=os.getenv(api_key_env),
            request_timeout=float(model_cfg.get("timeout", 10.0)),
        )
        self.adapter = RobotTwinAdapter(
            self.client,
            default_instruction=str(model_cfg.get("default_instruction", "")),
        )
        self.current = None
        self.control_step = 0
        self.exec_horizon = int(model_cfg.get("exec_horizon", 1))
        if self.exec_horizon < 1:
            raise ValueError("exec_horizon must be positive")
        metadata = self.client.metadata
        dimensions = (
            int(metadata.get("raw_state_dim", -1)),
            int(metadata.get("model_state_dim", metadata.get("state_dim", -1))),
            int(metadata.get("raw_action_dim", -1)),
            int(metadata.get("model_action_dim", metadata.get("action_dim", -1))),
        )
        if dimensions != (18, 14, 18, 14):
            raise ValueError(
                "RoboTwin requires raw/model dimensions 18/14 -> 18/14, "
                f"got {metadata}"
            )

    def update_obs(self, obs):
        self.current = obs

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("EmbodiedInfer XPolicyLab plugin supports batch size 1")
        self.current = obs_list[0]

    def reset(self):
        self.current = None
        self.control_step = 0
        self.client.reset()

    def get_action(self):
        if self.current is None:
            raise RuntimeError("update_obs must be called before get_action")
        response = self.adapter.infer(
            self.current,
            control_step=self.control_step,
            timeout=float(self.config.get("timeout", 10.0)),
        )
        actions = response["raw_actions"][:self.exec_horizon]
        self.control_step += len(actions)
        return [self.adapter.action_dict(action) for action in actions]

    def get_action_batch(self, env_idx_list=None):
        indices = env_idx_list or [0]
        if len(indices) != 1:
            raise ValueError("EmbodiedInfer XPolicyLab plugin supports batch size 1")
        return [self.get_action()]
