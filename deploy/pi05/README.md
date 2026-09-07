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
