from __future__ import annotations

import os

from XPolicyLab.model_template import ModelTemplate

from embodied_infer_deploy.client import InferenceClient
from embodied_infer_deploy.robots import get_robot_contract
from embodied_infer_deploy.simulators.robotwin import (
    RemoteRoboTwinPolicy,
    RoboTwinAdapter,
)


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
        self.adapter = RoboTwinAdapter(
            self.client,
            default_instruction=str(model_cfg.get("default_instruction", "")),
        )
        self.exec_horizon = int(model_cfg.get("exec_horizon", 1))
        if self.exec_horizon < 1:
            raise ValueError("exec_horizon must be positive")
        metadata = self.client.metadata
        contract = get_robot_contract(str(model_cfg.get("robot", "dazz-s600")))
        dimensions = (
            int(metadata.get("raw_state_dim", -1)),
            int(metadata.get("model_state_dim", metadata.get("state_dim", -1))),
            int(metadata.get("raw_action_dim", -1)),
            int(metadata.get("model_action_dim", metadata.get("action_dim", -1))),
        )
        expected = (
            contract.raw_state_dim,
            contract.model_dim,
            contract.raw_action_dim,
            contract.model_dim,
        )
        if dimensions != expected:
            raise ValueError(
                f"RoboTwin requires raw/model dimensions {expected}, got {metadata}"
            )
        self.policy = RemoteRoboTwinPolicy(
            self.adapter,
            exec_horizon=self.exec_horizon,
            timeout=float(model_cfg.get("timeout", 10.0)),
        )

    def update_obs(self, obs):
        self.policy.update_obs(obs)

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("EmbodiedInfer XPolicyLab plugin supports batch size 1")
        self.policy.update_obs(obs_list[0])

    def reset(self):
        self.policy.reset()

    def get_action(self):
        return self.policy.get_action()

    def get_action_batch(self, env_idx_list=None):
        indices = env_idx_list or [0]
        if len(indices) != 1:
            raise ValueError("EmbodiedInfer XPolicyLab plugin supports batch size 1")
        return [self.get_action()]
