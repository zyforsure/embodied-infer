# Deployment configuration

The JSON files are templates, not credentials or model artifacts:

- `orin-turbovla.example.json`: TensorRT plan on Jetson AGX Orin;
- `s600-hbm-direct.example.json`: direct import of the verified S600 HBM
  runtime;
- `s600-hbm-remote.example.json`: 4090 gateway to the existing S600 HBM server;
- `s100-hbm-remote.example.json`: 4090 gateway to the S100 TurboVLA HBM server;
- `pi05-remote.example.json`: OpenPI Pi0.5 WebSocket policy server;
- `mock.example.json`: local protocol smoke tests;
- `accelerations.example.json`: optional pluggable acceleration plugins
  (prefix cache, vision-token cache, micro-pipeline, action-aware quantization).
- `vla-pipeline.example.json`: standard VLA pipeline knobs -- pick one
  `action_head` paradigm (`autoregressive`, `regression`, `flow_matching`,
  or `dual_system`) and tune its `head_config` block; see
  `docs/vla-pipeline.md`.

Replace paths and addresses for your machine. Do not commit plans, HBM files,
tokenizers, normalization statistics, API keys, or robot logs.

TurboVLA configs expose an 18D robot/simulator contract and a 14D model
contract. These dimensions are fixed by the checked adapter mapping, not by an
implicit truncate or reshape operation.

Select a config through the model registry, for example:

```bash
python -m embodied_infer_deploy.server \
  --model turbovla-tensorrt \
  --backend-config config/orin-turbovla.example.json
```

Run `python -m embodied_infer_deploy.server --list-models` to list built-ins.
