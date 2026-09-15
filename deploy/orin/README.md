# Jetson AGX Orin

Copy the repository to Orin and install it with `scripts/install.sh`. Edit
`config/orin-turbovla.example.json` (or provide a private config path), then:

```bash
embodied-infer doctor --profile orin --config /etc/embodied-infer/orin.json
embodied-infer server --model turbovla-tensorrt \
  --config /etc/embodied-infer/orin.json \
  --default-config orin-turbovla.example.json \
  --host 0.0.0.0 --port 44091
```

The engine, tokenizer, statistics, and TurboVLA runtime are intentionally not
vendored. They are hardware/model artifacts and must be mounted or copied to
the paths in the private config. The mixed-precision engine with DINOv3 blocks
0--9 in FP32 is the correctness-first choice documented in `docs/deployment.md`.
