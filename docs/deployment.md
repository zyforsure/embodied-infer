# Orin / S600 / S100 / RTX 4090 deployment

This guide maps the existing verified setup into the new runtime boundary. It
does not replace the tested S600 safety wrapper or the RoboTwin simulator.

## Topology

```text
RTX 4090 (RoboTwin/SAPIEN)
  XPolicyLab / embodied-infer RoboTwinAdapter
                |
                | WebSocket + MessagePack, default port 44091
                v
Jetson AGX Orin (TensorRT)
  embodied-infer server + TurboVLA TensorRT plan

S600 real robot (optional alternative backend)
  existing HBM service / turbovla_s600_server.py :5702
                ^
                | existing length-prefixed NPZ protocol
                |
RTX 4090 gateway
  embodied-infer s600-hbm-remote backend
```

The previously validated services remain separate:

- Orin TensorRT WebSocket: `192.168.10.162:44090`;
- Embodied.cpp ZeroMQ service: port `5555`;
- S600 HBM server / proxy path: the existing `5702` deployment.

The new protocol intentionally defaults to `44091` so a first smoke test cannot
take over the old service. After parity is established, change the port or
switch the RoboTwin policy configuration.

## S100 HBM backend

S100 is available as the explicit `turbovla-s100-remote` model. Its four-stage
HBM service uses the same `[1,3,3,224,224]` image tensor, `[1,256]` text
tensors, `[1,14]` state tensor, and `[1,50,14]` action output as the validated
S600 service. Use `config/s100-hbm-remote.example.json` and keep the S100 HBM
process on its own port (the existing deployment uses `5702`). The gateway
proxy performs tokenizer/DINO preprocessing and then reuses the common
18D/14D adapter, so RoboTwin can switch between S600 and S100 by changing only
the registered model and endpoint config.

## 1. Orin TensorRT server

Install the Python package on Orin and point the backend at the existing fixed
shape plan. The correctness-first plan from the deployment log keeps DINOv3
vision blocks 0--9 in FP32; do not silently replace it with the all-FP16
experiment that produced NaNs.

```bash
cd /home/nvidia/embodied-infer
python3 -m pip install -e .
python3 -m embodied_infer_deploy.server \
  --model turbovla-tensorrt \
  --backend-config config/orin-turbovla.example.json \
  --host 0.0.0.0 --port 44091
```

Run a smoke request from the 4090:

```bash
python -m embodied_infer_deploy.smoke \
  --host 192.168.10.162 --port 44091
```

Expected metadata is `raw_state_dim=18`, `model_state_dim=14`,
`raw_action_dim=18`, `model_action_dim=14`, `action_horizon=50`, and camera
order `head,left_wrist,right_wrist`. The legacy `state_dim` and `action_dim`
aliases describe the 14D tensors carried between the adapter and model server.
`robot_command_dim=16` distinguishes the currently executable RoboTwin qpos
fields from the complete reconstructed 18D raw target.

## 2. RoboTwin on RTX 4090

Copy `integrations/xpolicylab/EmbodiedInfer` into the RoboTwin `XPolicyLab/policy`
tree (or add it to `PYTHONPATH`), then set:

```yaml
policy_name: EmbodiedInfer
host: 192.168.10.162
port: 44091
exec_horizon: 1
```

The adapter requires the 18D raw `joint_action.vector`. It applies the same
calibration mapping used by the deployed plan:

```text
model[0:14] <- raw[[0,1,2,3,4,5,8,9,10,11,12,13,7,15]]
```

The model order is `left6,right6,left_gripper,right_gripper`. Each `[50,14]`
prediction is expanded back to `[50,18]`; unmapped coordinates are copied from
the current raw state. This explicitly holds both seventh arm joints and any
other raw coordinates outside the trained action head before `take_action`.

For a first episode, keep `exec_horizon: 1`. Once the end-to-end action order,
gripper polarity, and timing are confirmed, increase it to amortize network
round trips. This is separate from the model horizon of 50.

## 3. Existing S600 HBM service through a 4090 gateway

Keep the S600 service and its safety wrapper unchanged. Start the known-good
S600 command in dry-run mode first:

```bash
./run_turbovla_real_robot.sh \
  --skip-can \
  --state-values '0,0,0,0,0,0,0,0,0,0,0,0,1,1' \
  --prompt 'put the box' \
  --cycles 1
```

On the 4090, configure `config/s600-hbm-remote.example.json` with the actual
S600 IP and paths, then expose it through the new versioned WebSocket boundary:

```bash
python -m embodied_infer_deploy.server \
  --model turbovla-s600-remote \
  --backend-config config/s600-hbm-remote.example.json \
  --host 0.0.0.0 --port 44091
```

This path preserves the four HBM graph contract and returns `[50,14]` absolute
model-order actions to the adapter, which reconstructs `[50,18]` raw targets.
It does not enable CAN motion. The existing S600 process must still be
explicitly started with `--enable-motion`, and the first physical test should
retain `--action-steps 1`, `--max-step-deg 1.0`, the default speed and
acceleration limits, and `--enable-gripper` disabled.

## 4. Direct S600 server mode

If Python dependencies and HBM runtime are available on the S600-side host,
use `config/s600-hbm-direct.example.json` with the direct backend. This avoids
the 4090 gateway but requires importing the existing real-robot script and its
vendor runtime in the server process. It is intended for later latency tuning,
not the first safety validation.

```bash
python -m embodied_infer_deploy.server \
  --model turbovla-s600-hbm \
  --backend-config config/s600-hbm-direct.example.json \
  --host 0.0.0.0 --port 44091
```

## 5. Measurements and acceptance gates

Record, per request:

- client round-trip time;
- server queue and total time;
- model stage timings when the backend exposes them;
- control step and source timestamp;
- action shape, finite-value check, and representation.

For RoboTwin, report both raw policy latency and full closed-loop FPS. The
existing test measured about `4.55 ms` TensorRT execute p50 on a 4090 but only
`1.86 FPS` for a full RoboTwin loop because simulation, preprocessing, network,
and temporal ensemble were included. These numbers must not be conflated.

## 6. Troubleshooting

- **Protocol mismatch**: check that both sides report `embodied-infer/1` and
  that the client connects to `44091`, not the legacy `44090` service.
- **Missing camera**: verify names are `head`, `left_wrist`, `right_wrist`; the
  adapter accepts common RoboTwin aliases but the wire protocol is canonical.
- **NaN actions**: use the mixed-precision Orin plan with vision blocks 0--9
  FP32 and inspect `server_timing`; do not enable motion.
- **S600 state mismatch**: the robot-facing adapter requires exactly 18 raw
  values; all four HBM graphs require exactly 14 model-order values. Inspect
  the explicit index map above. Unmodelled coordinates stay at current
  feedback and are not sent to the model.
- **Unsafe motion**: return to `--skip-can`, then read-only CAN, then one action
  step with the documented limits. Gripper motion requires an explicit flag.
