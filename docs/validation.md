# Hardware-in-the-loop validation

The following read-only checks were run from the development workstation on
2026-09-07. They do not enable CAN or send robot motion commands.

| Target | Result | Evidence |
| --- | --- | --- |
| Orin `192.168.10.162:44090` | Pass | Legacy TurboVLA WebSocket hello, then three zero-image requests returned finite `[1,50,14]`; measured round trips were about 221–338 ms. |
| Orin + RoboTwin adapter | Pass | Three demo-environment steps completed through the legacy Orin server; each reconstructed action was finite 16D qpos. |
| S600 HBM `192.168.10.252:5702` | Pass | Length-prefixed NPZ request with the deployed six-input contract returned finite `(1,50,14)` actions. |
| S600 + RoboTwin adapter | Pass (contract) | Two demo-environment steps completed through S600 HBM with synthetic zero preprocessing; reconstructed actions were finite 16D qpos. |
| A800 Pi05 OpenPI `172.16.0.23:8012` | Pass | Updated `Pi05RemoteBackend` completed a real cross-host request and returned finite `(16,14)` actions; first request was about 24.9 s including accelerator warm-up. |
| A800 Pi05 + RoboTwin adapter | Pass (smoke) | Two real demo-environment control steps completed through the A800 Pi05 service; actions were reconstructed to finite 16D qpos. |
| S100/S600 HBM endpoint `192.168.10.252:5702` | Pass (service smoke) | Three zero-input length-prefixed NPZ requests returned finite `(1,50,14)` actions; measured round trips were about 4.53 s. Existing deployment notes use this address for both S100 and S600 paths, so board identity is not inferred from the shared endpoint. |
| Pi05 HBM artifacts on RTX 4090 | Pass (compile + header load) | The 4090 `pi05_qat` environment has `leap_llm`; shared storage contains `pi05_siglip_ptq.hbm`, `pi05_gemma_llm_ptq.hbm` (about 2.9 GB), and `pi05_gemma_expert_ptq.hbm` (about 441 MB). The recorded `hb_compile` logs end with successful `compile_hbo`/`link_models`; the `oe370` HBDK loader also opened all three `nash-p` HBM files and reported their graph names and tensor contracts. |
| Pi05 local HBM on Orin/S100/S600 | Blocked | The artifacts now exist on the 4090 shared storage, but no board-side Pi05 HBM server contract or credentials for Orin/S100/S600 deployment are available. Existing reachable endpoints expose TurboVLA contracts, not the Pi05 Gemma/Expert graph interface. |
| New embodied-infer service `:44091` on Orin | Not started | Port is closed; SSH key authentication is required to deploy the new service. |

The S600 adapter test above validates transport, tensor shapes, inference, and
18D/14D/18D action reconstruction. A full camera-accurate S600 RoboTwin run
still requires the private tokenizer, DINO preprocessor, statistics file, and
the Linux RoboTwin/SAPIEN environment. The existing Orin service uses the
legacy WebSocket protocol on `44090`; the new versioned protocol intentionally
uses `44091` to avoid taking over that service.

The Pi05 OpenPI rows use the official WebSocket service and a deterministic
RoboTwin-compatible demo environment. They are separate from the HBM artifact
check above. HBM compilation is now evidenced for Vision/SigLIP, Gemma LLM,
and the action expert on the 4090; board execution still requires the matching
HBRT runtime and a Pi05-specific server that wires vision tokens, KV caches,
and the 16x16 native action head.

The Orin and S100/S600 rows above validate the currently deployed TurboVLA
services, not a Pi05 HBM port. A Pi05 local-HBM acceptance row requires the
three HBM files to be copied to the target board, successful HBRT load/execute
of all graphs, and a RoboTwin closed-loop request through the Pi05 server.
