# RTX 4090 / RoboTwin

The repository ships a dependency-free closed-loop demo, so the first check is
always:

```bash
embodied-infer sim
```

For a real RoboTwin checkout, install RoboTwin/XPolicyLab in its own
environment and expose its environment factory on `PYTHONPATH`:

```bash
embodied-infer robotwin \
  --env-factory my_robotwin_entry:create_env \
  --env-kwargs '{"task_name":"place_object"}' \
  --host 192.168.10.162 --port 44091 \
  --exec-horizon 1
```

The server can run locally with the mock model while wiring the simulator:

```bash
embodied-infer server --model mock --config config/mock.example.json
```

On the 4090, the production model is normally remote on Orin; keep the
RoboTwin process and model server as separate processes. `doctor` reports
missing simulator/model files without changing the existing RoboTwin tree.
