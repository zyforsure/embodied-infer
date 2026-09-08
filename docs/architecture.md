# Architecture

## 统一分层（RoboTwin、真机与数据集）

下面是当前实现采用的端到端边界。每一层只依赖下一层公开的类型契约，
因此同一个 RoboTwin policy 可以切换 4090、Orin、S100、S600 或远程 Pi05，
而不需要修改模型代码。

```text
RoboTwin / 真机 / 数据集
        |
        v
具身适配层
18D 原始状态 + 多相机
        |
        v
服务层
协议校验 / 鉴权 / metadata / 生命周期 / 监控
        |
        v
请求调度层
有界队列 / newest-wins / deadline
        |
        v
Engine
预处理 / Backend / 后处理 / 安全检查
        |
        v
算子调度层
图像预处理 -> Vision Encoder -> LLM -> Action Expert -> Projection
        |
        v
动作适配层
14D/16D 模型动作 -> 18D 原始动作 -> ActionBuffer
        |
        v
RoboTwin / Orin / S100 / S600
```

### 层间 contract

| 层 | 输入 | 输出 | 关键不变量 |
| --- | --- | --- | --- |
| 具身适配层 | RoboTwin observation、机器人反馈、数据集样本 | canonical observation | 原始 state 固定为 18D；相机规范名为 `head`、`left_wrist`、`right_wrist`；图像为 HWC RGB |
| 服务层 | MessagePack `embodied-infer/1` | hello、health、action、error | 版本、尺寸、时间戳、有限值和鉴权先校验；hello 发布完整 `ModelSpec.metadata` |
| 请求调度层 | canonical observation + deadline | 一个待执行请求 | C++ `Scheduler` 默认容量 1、`keep_latest`；过期请求在进入 backend 前取消 |
| Engine | observation、取消 token、deadline | `ActionChunk` | 串行保护非重入 backend；执行 observation/action processor 和安全过滤 |
| 算子调度层 | 一个模型请求的 operator DAG | stage outputs/timing | 按依赖调度 CPU/GPU/NPU/IO lane；失败和 deadline 向下游传播 |
| 动作适配层 | 14D 或原生 16D chunk、当前 18D state | 18D raw target + executable command | 未建模坐标从当前反馈回填；输出进入 `ActionBuffer` 前做 finite、限位、跃变检查 |
| 设备/仿真层 | 18D raw target 或 typed hardware command | RoboTwin、关节位置/力矩、末端位姿 | 不在模型 backend 中调用 CAN；真实运动必须由设备侧显式开启 |

### Pi05 operator DAG

Pi05 的 OpenPI 远程 backend 和本地 HBM backend 共用上面的请求与动作边界。
本地 HBM 实现按以下依赖图执行，KV cache 只在 LLM/Action Expert 之间传递，
不会泄漏到服务协议：

```text
decode/resize/normalize (CPU)
          |
          +--> SigLIP vision encoder (GPU/NPU) --+
          |                                      |
tokenize(prompt) (CPU) --------------------------+--> Gemma LLM (NPU)
                                                 |
                          state + noisy action + KV cache
                                                 v
                                      Action Expert (NPU)
                                                 |
                                      16D native action
                                                 v
                                  Projection / safety (CPU)
```

服务层只需要知道 `model_action_dim`（Pi05 共享 contract 为 14，原生 head
可为 16）和 `action_horizon`；具体 HBM tensor 名称、量化和 runtime 由
`models/pi05/` 实现负责。

### 服务生命周期与监控

WebSocket 建连后服务先发送 `hello`。客户端可发送 `type=health` 获取
`status`（`starting|ready|draining|stopped`）、`uptime_ms`、成功/失败请求计数
和同一份 metadata；`type=reset` 用于清理 backend 状态。推理响应带有
`server_queue_ms`、`server_total_ms` 以及 backend stage timing，便于区分
排队、预处理、模型执行和仿真耗时。

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

## Vision continuous-batching plugin

`plugins/vision_batch.py` provides one model-neutral Vision-stage plugin for
Pi05 and TurboVLA. Enable it in a backend config with:

```json
"vision_batching": {
  "enabled": true,
  "max_batch_size": 4,
  "batch_wait_ms": 2.0,
  "max_queue_size": 32
}
```

When the selected runtime exposes `encode_vision(batch)` (and a matching
`predict_from_vision(...)` continuation), the plugin uses native continuous
batching with a tensor shaped `[B, views, 3, H, W]`. Requests retain their own
language tokens, KV cache, state, and action-expert denoising context. If a
runtime only exposes a monolithic `predict(images, state, prompt)` API, the
same configuration safely reports `mode=fallback` and preserves the original
execution path; it does not claim a speedup until a split Vision API exists.

This boundary follows the serving patterns used by vLLM-style schedulers:
batch only the stateless, shape-compatible front-end, enforce a bounded queue
and deadline, and keep autoregressive or diffusion state request-local.

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
