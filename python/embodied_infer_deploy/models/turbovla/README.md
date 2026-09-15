# TurboVLA

One model family with three execution targets:

- `tensorrt.py`: RTX 4090 and Jetson AGX Orin TensorRT;
- `s600_hbm.py`: direct four-graph S600 HBM runtime;
- `s600_remote.py`: gateway to the existing S600 HBM server.

All targets expose the same model contract: three RGB cameras, 14D model
state, and a `[50,14]` absolute action chunk. The robot/simulator adapter owns
the separate 18D raw contract.
