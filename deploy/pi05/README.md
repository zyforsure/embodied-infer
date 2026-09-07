# Pi0.5 / Pi05

The Pi05 adapter speaks the OpenPI WebSocket protocol directly. The model host
can run OpenPI's stock server, while RoboTwin and the robot-facing adapters
remain in this repository. Pi05 receives the canonical observation:

```text
state: [14]
images.cam_high: [3,H,W] uint8
images.cam_left_wrist: [3,H,W] uint8
images.cam_right_wrist: [3,H,W] uint8
prompt: string
```

Start an OpenPI Pi05 server using its own checkpoint/environment, then point
the gateway at it:

```bash
embodied-infer doctor --profile pi05 --config config/pi05-remote.example.json
embodied-infer server --model pi05-remote \
  --config config/pi05-remote.example.json \
  --host 0.0.0.0 --port 44091
```

The adapter accepts Pi05's usual `[15,14]` action chunk by default and
reconstructs the complete 18D raw target before RoboTwin qpos flattening.
Change `action_horizon` only when the loaded Pi05 checkpoint/config reports a
different horizon.

## Embodied.cpp / Orin

`pi05-cpp` is the framework gateway for the Embodied.cpp ZMQ server used on
Jetson/Orin. It translates the common 18D/14D request into the protobuf
`PredictRequest`, schedules the request through the normal service and Engine
layers, and maps the native action chunk back to the shared contract:

```bash
embodied-infer sim --model pi05-cpp \
  --config config/pi05-cpp-orin-remote.example.json --steps 2 --timeout 20
```

The checked Orin process at `192.168.10.162:5555` currently identifies itself
as `hy_vla`, so this command validates the Embodied.cpp transport and RoboTwin
closed loop, not a Pi05 checkpoint. Set `server_arch` to `pi05` only after the
Orin daemon is restarted with a Pi05 GGUF; metadata exposes
`model_identity_verified` for this distinction.

`pi05-hbm` is the board-local D-Robotics loader. It refuses to report ready
when files are missing or when HBRT reports a target-march mismatch. The
current 4090 artifacts are `nash-p`; S100 is `nash-e`, therefore they must be
recompiled for S100 before enabling the HBM inference bindings.
