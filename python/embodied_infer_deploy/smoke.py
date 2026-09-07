from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .client import InferenceClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=44091)
    parser.add_argument("--prompt", default="move safely")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    client = InferenceClient(
        args.host, args.port, api_key=args.api_key, request_timeout=args.timeout
    )
    metadata = client.metadata
    state_dim = int(metadata.get("model_state_dim", metadata["state_dim"]))
    cameras = list(metadata.get(
        "camera_order", ["head", "left_wrist", "right_wrist"]
    ))
    now = time.time_ns()
    response = client.infer(
        control_step=0,
        timestamp_ns=now,
        instruction=args.prompt,
        images=[
            (name, np.zeros((224, 224, 3), dtype=np.uint8), now)
            for name in cameras
        ],
        state=np.zeros(state_dim, dtype=np.float32),
        timeout=args.timeout,
    )
    summary = {
        "metadata": metadata,
        "request_id": response["request_id"],
        "action_shape": list(response["actions"].shape),
        "actions_finite": bool(np.isfinite(response["actions"]).all()),
        "client_round_trip_ms": response["client_round_trip_ms"],
        "server_timing": response.get("timing", {}),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
