# Pi0.5 / Pi05

The Pi05 model family is isolated from TurboVLA and communicates with an
OpenPI policy host through the native WebSocket + msgpack-numpy protocol.
`remote.py` only depends on NumPy, msgpack, and websockets; OpenPI itself and
the checkpoint stay on the Pi05 model host.

The adapter contract is explicit:

- RoboTwin raw state/action: 18 dimensions;
- Pi05 model state/action: 14 dimensions;
- native action horizon: 16 steps (the shared config defaults to 16);
- native action width: 16, projected to the shared 14D model contract;
- RoboTwin qpos command after reconstruction: 16 dimensions.

Use `config/pi05-remote.example.json` as the gateway configuration template.
