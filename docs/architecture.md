# Architecture

The repository separates model execution, robot semantics, simulator parsing,
transport, and scheduling. A model implementation must never need to import
RoboTwin or a CAN vendor SDK.

Within a model backend, the optional operator scheduler runs the dependency
graph for stages such as image decode/normalize, vision encoder, language
tokens, action expert, and action projection. The request scheduler remains
the outer control-loop queue; the operator scheduler is bounded, device-aware
parallelism inside one inference request.

The final hardware boundary is typed separately in `hardware_output.hpp`. It
supports joint position, joint torque, joint position+torque, end-effector pose
(`[x,y,z,qx,qy,qz,qw]`), and end-effector pose+joint torque. The adapter checks
dimensions, finite values, frame, control mode, and non-zero quaternions before
a robot plugin serializes the command for CAN, EtherCAT, ROS 2, or a simulator.
Inverse kinematics and device calibration remain plugin responsibilities.

```text
RoboTwin / S600 / another embodiment
                 |
          raw robot contract
      observation + current state
                 |
        simulator/robot adapter
                 |
      model-shaped wire request
                 |
       WebSocket serving layer
                 |
        lazy ModelRegistry
                 |
 models/mock | models/turbovla | external plugin
                 |
            BackendResult
                 |
       raw action reconstruction
                 |
       simulator / safety driver
```

## Package ownership

| Directory | Owns | Must not own |
| --- | --- | --- |
| `core/` | `ModelSpec`, backend protocol, result types | model libraries, simulator APIs |
| `models/<family>/` | checkpoint preprocessing and accelerator execution | robot joint ordering, CAN writes |
| `robots/` | raw state/action schema and model-index projection | image preprocessing, networking |
| `simulators/robotwin/` | observation aliases, episode loop, action dictionary | TensorRT/HBM imports |
| `protocol.py`, `client.py`, `server.py` | versioned transport and serving | model-specific tensor code |
| `integrations/` | framework-specific thin plugins | duplicated inference logic |

This follows the typed adapter boundary used by Embodied.cpp and the lazy,
per-model organization used by vLLM/vLLM-Omni. The implementations here are
independent and do not vendor code from either project.

## Adding a model

1. Copy `models/_template` to `models/<family>`.
2. Construct an immutable `ModelSpec` and implement `infer()`.
3. Register a lazy `module:create_backend` target in `models/registry.py`, or
   expose the factory through the `embodied_infer.models` entry-point group.
4. Add contract tests that run without loading accelerator libraries.
5. Put device-specific parity tests behind an explicit test marker.

The registry intentionally imports model modules only when selected. An Orin
server therefore does not import S600 vendor libraries, and a local protocol
test does not require TensorRT.

## RoboTwin testing

The default suite runs a complete fake RoboTwin episode through the real
adapter and policy code. It checks the 18D raw boundary, 14D model request,
three-camera ordering, chunk execution, and held-coordinate behavior.

An installed RoboTwin environment can be checked without committing machine
paths:

```bash
export EMBODIED_INFER_ROBOTWIN_FACTORY='my_robotwin_setup:create_env'
export EMBODIED_INFER_ROBOTWIN_KWARGS='{"task":"beat_block_hammer"}'
pytest -m robotwin tests/simulators/robotwin/test_external_environment.py
```

The factory must return an object implementing `reset()`, `get_obs()`, and
optionally `close()`. This smoke test only reads one observation; it does not
execute an action.

To run a real closed-loop episode against the server:

```bash
embodied-infer-robotwin \
  --env-factory my_robotwin_setup:create_env \
  --env-kwargs '{"task":"beat_block_hammer"}' \
  --host 192.168.10.162 --port 44091 \
  --exec-horizon 1 --max-steps 100
```

The CLI defaults to flattening the four XPolicyLab action fields into the qpos
vector expected by RoboTwin's base `take_action()`. Use `--action-format dict`
only when an environment wrapper consumes XPolicyLab dictionaries directly.
Keep `exec-horizon=1` for initial action-order and safety validation.

The adapter retains the complete reconstructed 18D target under
`action["raw_action"]`. The currently named qpos fields contain 16 executable
values (left 7 + gripper + right 7 + gripper); raw indices 16 and 17 remain
held but are not sent to an actuator until their device-level meaning is
defined. Server metadata reports this separately as `robot_command_dim=16`.
