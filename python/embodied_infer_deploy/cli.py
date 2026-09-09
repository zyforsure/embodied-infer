"""Single entry point for a fresh checkout.

The command intentionally separates software checks from physical motion:
``sim`` is fully self-contained, ``server`` starts an inference service, and
``real`` performs a read-only gateway/action-contract check.  A CAN driver is
never enabled implicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .client import InferenceClient
from .models import create_model, model_registry
from .robots import get_robot_contract
from .server import main as server_main
from .simulators.robotwin.cli import main as robotwin_main
from .simulators.robotwin import (
    RoboTwinAdapter,
    RemoteRoboTwinPolicy,
    create_demo_env,
    flatten_qpos_action,
    run_episode,
)


ROOT = Path(__file__).resolve().parents[2]

# Doctor profiles: name -> (registry model, default config, needs TCP service).
# The argparse choices and the check logic both derive from this table, so a
# new deployment profile is one line here instead of edits in three places.
DOCTOR_PROFILES: dict[str, tuple[str, str, bool]] = {
    "mock": ("mock", "mock.example.json", False),
    "orin": ("turbovla-tensorrt", "orin-turbovla.example.json", False),
    "s600-remote": ("turbovla-s600-remote", "s600-hbm-remote.example.json", True),
    "s600-direct": ("turbovla-s600-hbm", "s600-hbm-direct.example.json", True),
    "s100": ("turbovla-s100-remote", "s100-hbm-remote.example.json", True),
    "pi05": ("pi05-remote", "pi05-remote.example.json", True),
    "pi05-cpp": ("pi05-cpp", "pi05-cpp-orin.example.json", True),
    "pi05-hbm": ("pi05-hbm", "pi05-hbm-s100.example.json", False),
    "pi05-tcp": ("pi05-tcp", "pi05-tcp-orin.example.json", True),
}


def _config_endpoints(config: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """Discover (name, host, port) from host/port config key pairs.

    Recognizes both bare ``host``/``port`` and prefixed ``s600_host``/
    ``s600_port`` style keys, so doctor checks stay config-driven instead
    of hardcoding one endpoint convention per profile.
    """

    endpoints: list[tuple[str, str, int]] = []
    for key, value in sorted(config.items()):
        if key != "host" and not key.endswith("_host"):
            continue
        stem = "" if key == "host" else key[: -len("_host")]
        port = config.get(f"{stem}_port" if stem else "port")
        if value and port is not None:
            name = f"{stem}-service" if stem else "service"
            endpoints.append((name, str(value), int(port)))
    return endpoints


def _config_path(value: str | None, default_name: str) -> Path:
    path = Path(value) if value else ROOT / "config" / default_name
    return path.expanduser().resolve()


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return value


def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _doctor(args: argparse.Namespace) -> int:
    profile = args.profile
    model_name, default_config, needs_service = DOCTOR_PROFILES[profile]
    config_path = _config_path(args.config, default_config)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("python", sys.version_info >= (3, 10), sys.version.split()[0])
    try:
        config = _read_config(config_path)
        check("config", True, str(config_path))
    except Exception as exc:
        config = {}
        check("config", False, str(exc))
    for key in ("runtime_root", "engine", "tokenizer", "stats"):
        if key in config:
            path = Path(str(config[key])).expanduser()
            check(key, path.exists(), f"{path} ({'found' if path.exists() else 'missing'})")
    endpoints = _config_endpoints(config)
    for name, host, port in endpoints:
        check(name, _port_open(host, port), f"{host}:{port}")
    if needs_service and not endpoints:
        check("service", False, "missing host/port pair in config")
    if args.server_host:
        check("inference-service", _port_open(args.server_host, args.server_port),
              f"{args.server_host}:{args.server_port}")
    try:
        backend = create_model(model_name, config)
        check("backend", True, json.dumps(backend.metadata, ensure_ascii=False))
    except Exception as exc:
        # A missing vendor runtime/model is actionable, but should not hide the
        # other diagnostics (and mock remains completely offline).
        check("backend", profile == "mock", str(exc))

    passed = all(item["ok"] for item in checks)
    if args.json:
        print(json.dumps({"profile": profile, "ok": passed, "checks": checks},
                         indent=2, ensure_ascii=False))
    else:
        print(f"embodied-infer doctor [{profile}] {'OK' if passed else 'NOT READY'}")
        for item in checks:
            print(f"  [{'ok' if item['ok'] else '!!'}] {item['name']}: {item['detail']}")
    return 0 if passed else 1


def _server(args: argparse.Namespace) -> int:
    argv = ["embodied-infer-server", "--model", args.model,
            "--backend-config", str(_config_path(args.config, args.default_config)),
            "--host", args.host, "--port", str(args.port)]
    if args.api_key_env:
        argv.extend(["--api-key-env", args.api_key_env])
    old = sys.argv
    try:
        sys.argv = argv
        server_main()
    finally:
        sys.argv = old
    return 0


def _wait_client(host: str, port: int, timeout: float = 20.0) -> InferenceClient:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        client = InferenceClient(host, port, request_timeout=2.0)
        try:
            client.metadata
            return client
        except Exception as exc:
            last = exc
            client.close()
            time.sleep(0.15)
    raise RuntimeError(f"inference server did not become ready: {last}")


def _sim(args: argparse.Namespace) -> int:
    config = _config_path(args.config, args.default_config)
    command = [sys.executable, "-m", "embodied_infer_deploy.server",
               "--model", args.model, "--backend-config", str(config),
               "--host", args.host, "--port", str(args.port)]
    process = subprocess.Popen(command, cwd=str(ROOT), env=os.environ.copy())
    client = None
    environment = None
    try:
        client = _wait_client(args.host, args.port)
        environment = create_demo_env(episode_steps=args.steps, image_size=args.image_size)
        adapter = RoboTwinAdapter(client, default_instruction=args.instruction)
        policy = RemoteRoboTwinPolicy(adapter, exec_horizon=1, timeout=float(args.timeout))
        executed = run_episode(environment, policy, action_transform=flatten_qpos_action)
        print(json.dumps({"mode": "demo", "executed_steps": executed,
                          "requests": executed, "metadata": client.metadata},
                         indent=2, ensure_ascii=False))
        return 0
    finally:
        if client is not None:
            client.close()
        if environment is not None:
            environment.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _real(args: argparse.Namespace) -> int:
    """Read-only action-contract check against a running inference gateway."""
    client = InferenceClient(args.host, args.port, request_timeout=args.timeout)
    try:
        metadata = client.metadata
        state = np.zeros(18, dtype=np.float32)
        image = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
        adapter = RoboTwinAdapter(client, default_instruction=args.instruction)
        response = adapter.infer({
            "instruction": args.instruction,
            "head_camera": image,
            "left_camera": image,
            "right_camera": image,
            "state": state,
        }, timeout=args.timeout)
        target = get_robot_contract().actions_to_raw(response["actions"], state)[0]
        safe = np.asarray(target, dtype=np.float32).copy()
        safe[:18] = np.clip(safe, -np.inf, np.inf)
        print(json.dumps({"mode": "read-only", "motion_enabled": False,
                          "metadata": metadata,
                          "model_action_shape": list(response["actions"].shape),
                          "raw_action_dim": int(target.size),
                          "first_action_finite": bool(np.isfinite(safe).all()),
                          "max_step_deg": args.max_step_deg},
                         indent=2, ensure_ascii=False))
        return 0
    finally:
        client.close()


def _robotwin(args: argparse.Namespace) -> int:
    argv = ["embodied-infer-robotwin", "--env-factory", args.env_factory,
            "--env-kwargs", args.env_kwargs, "--host", args.host,
            "--port", str(args.port), "--timeout", str(args.timeout),
            "--exec-horizon", str(args.exec_horizon)]
    if args.max_steps is not None:
        argv.extend(["--max-steps", str(args.max_steps)])
    if args.default_instruction:
        argv.extend(["--default-instruction", args.default_instruction])
    argv.extend(["--action-format", args.action_format])
    old = sys.argv
    try:
        sys.argv = argv
        robotwin_main()
    finally:
        sys.argv = old
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="embodied-infer")
    sub = parser.add_subparsers(dest="command", required=True)
    models = sub.add_parser("models", help="list installed model adapters")
    models.set_defaults(func=lambda _args: (print("\n".join(model_registry.names())) or 0))

    doctor = sub.add_parser("doctor", help="check software, model files, and services")
    doctor.add_argument("--profile", choices=tuple(DOCTOR_PROFILES), default="mock")
    doctor.add_argument("--config")
    doctor.add_argument("--server-host")
    doctor.add_argument("--server-port", type=int, default=44091)
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=_doctor)

    server = sub.add_parser("server", help="start an inference server")
    server.add_argument("--model", default="mock")
    server.add_argument("--config")
    server.add_argument("--default-config", default="mock.example.json")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=44091)
    server.add_argument("--api-key-env", default="EMBODIED_INFER_API_KEY")
    server.set_defaults(func=_server)

    sim = sub.add_parser("sim", help="run the built-in RoboTwin-compatible simulation")
    sim.add_argument("--model", default="mock", choices=model_registry.names())
    sim.add_argument("--config")
    sim.add_argument("--default-config", default="mock.example.json")
    sim.add_argument("--host", default="127.0.0.1")
    sim.add_argument("--port", type=int, default=44091)
    sim.add_argument("--steps", type=int, default=8)
    sim.add_argument("--image-size", type=int, default=64)
    sim.add_argument("--instruction", default="move safely in the demo scene")
    sim.add_argument("--timeout", type=float, default=10.0,
                     help="per-inference timeout in seconds (remote Pi05 may need >2s)")
    sim.set_defaults(func=_sim)

    real = sub.add_parser("real", help="read-only S600 gateway and contract check")
    real.add_argument("--host", default="192.168.10.162")
    real.add_argument("--port", type=int, default=44091)
    real.add_argument("--timeout", type=float, default=10.0)
    real.add_argument("--image-size", type=int, default=224)
    real.add_argument("--instruction", default="move safely")
    real.add_argument("--max-step-deg", type=float, default=1.0)
    real.set_defaults(func=_real)

    robotwin = sub.add_parser("robotwin", help="run an external RoboTwin environment")
    robotwin.add_argument("--env-factory", required=True)
    robotwin.add_argument("--env-kwargs", default="{}")
    robotwin.add_argument("--host", default="192.168.10.162")
    robotwin.add_argument("--port", type=int, default=44091)
    robotwin.add_argument("--timeout", type=float, default=10.0)
    robotwin.add_argument("--exec-horizon", type=int, default=1)
    robotwin.add_argument("--max-steps", type=int)
    robotwin.add_argument("--default-instruction", default="")
    robotwin.add_argument("--action-format", choices=("flat-qpos", "dict"), default="flat-qpos")
    robotwin.set_defaults(func=_robotwin)
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
