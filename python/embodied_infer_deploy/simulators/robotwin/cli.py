"""Run a RoboTwin-style environment against an embodied-infer server."""

from __future__ import annotations

import argparse
import importlib
import json
import os

from ...client import InferenceClient
from ...robots import DEFAULT_ROBOT, get_robot_contract
from .adapter import RoboTwinAdapter
from .policy import RemoteRoboTwinPolicy
from .runner import flatten_qpos_action, run_episode


def _load_factory(target: str):
    if ":" not in target:
        raise ValueError("environment factory must use module:callable syntax")
    module_name, attribute = target.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"environment factory {target!r} is not callable")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-factory", required=True)
    parser.add_argument("--env-kwargs", default="{}", help="JSON object")
    parser.add_argument("--host", default="192.168.10.162")
    parser.add_argument("--port", type=int, default=44091)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--exec-horizon", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--default-instruction", default="")
    parser.add_argument("--robot", default=DEFAULT_ROBOT)
    parser.add_argument(
        "--camera-aliases",
        default=None,
        help='JSON map of canonical camera to alias list, e.g. '
        '\'{"head": ["cam_high"]}\' for non-RoboTwin simulators',
    )
    parser.add_argument(
        "--action-format",
        choices=("flat-qpos", "dict"),
        default="flat-qpos",
    )
    parser.add_argument("--api-key-env", default="EMBODIED_INFER_API_KEY")
    args = parser.parse_args()

    kwargs = json.loads(args.env_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--env-kwargs must decode to a JSON object")
    environment = _load_factory(args.env_factory)(**kwargs)
    client = InferenceClient(
        args.host,
        args.port,
        api_key=os.getenv(args.api_key_env),
        request_timeout=args.timeout,
    )
    camera_aliases = (
        json.loads(args.camera_aliases) if args.camera_aliases else None
    )
    adapter = RoboTwinAdapter(
        client,
        default_instruction=args.default_instruction,
        robot_contract=get_robot_contract(args.robot),
        camera_aliases=camera_aliases,
    )
    policy = RemoteRoboTwinPolicy(
        adapter, exec_horizon=args.exec_horizon, timeout=args.timeout
    )
    try:
        executed = run_episode(
            environment,
            policy,
            max_steps=args.max_steps,
            action_transform=(
                flatten_qpos_action if args.action_format == "flat-qpos" else None
            ),
        )
        print(json.dumps({"executed_steps": executed}, ensure_ascii=False))
    finally:
        client.close()
        close = getattr(environment, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
