# Hardware-in-the-loop validation

The following read-only checks were run from the development workstation on
2026-09-07. They do not enable CAN or send robot motion commands.

| Target | Result | Evidence |
| --- | --- | --- |
| Orin `192.168.10.162:44090` | Pass | Legacy TurboVLA WebSocket hello, then zero-image inference returned finite `[1,50,14]`; four-request p50 was 228.69 ms. |
| Orin + RoboTwin adapter | Pass | Three demo-environment steps completed through the legacy Orin server; each reconstructed action was finite 16D qpos. |
| S600 HBM `192.168.10.252:5702` | Pass | Length-prefixed NPZ request with the deployed six-input contract returned finite `(1,50,14)` actions. |
| S600 + RoboTwin adapter | Pass (contract) | Two demo-environment steps completed through S600 HBM with synthetic zero preprocessing; reconstructed actions were finite 16D qpos. |
| New embodied-infer service `:44091` on Orin | Not started | Port is closed; SSH key authentication is required to deploy the new service. |

The S600 adapter test above validates transport, tensor shapes, inference, and
18D/14D/18D action reconstruction. A full camera-accurate S600 RoboTwin run
still requires the private tokenizer, DINO preprocessor, statistics file, and
the Linux RoboTwin/SAPIEN environment. The existing Orin service uses the
legacy WebSocket protocol on `44090`; the new versioned protocol intentionally
uses `44091` to avoid taking over that service.
