# S100 HBM / RoboTwin

S100 uses the same four-stage TurboVLA HBM request contract as the existing
S600 service. Select it explicitly as `turbovla-s100-remote`; the backend keeps
the S100 endpoint separate in configuration while reusing the tested proxy and
18D/14D RoboTwin adapter.

On the 4090 gateway, install the private tokenizer, DINO preprocessor,
statistics file, and proxy script, then run:

```bash
embodied-infer doctor --profile s100 --config config/s100-hbm-remote.example.json
embodied-infer server \
  --model turbovla-s100-remote \
  --config config/s100-hbm-remote.example.json \
  --host 0.0.0.0 --port 44091
```

The S100 HBM service itself is normally started on `192.168.10.252:5702`
with the four files `vision.hbm`, `text.hbm`, `fusion.hbm`, and `action.hbm`
and `--staged-loading` when device memory requires it. This repository never
starts CAN motion; use RoboTwin first with `--exec-horizon 1`.
