# embodied-infer

`embodied-infer` is a small C++20 runtime foundation for real-time embodied
AI policy inference. It gives robot applications one stable boundary for
observations and action chunks while model execution remains replaceable.

The repository includes a complete, dependency-free runtime core, a
deterministic mock backend, TurboVLA deployment backends, and a Pi0.5/Pi05
OpenPI remote backend. Production checkpoints remain external model artifacts;
the model-specific adapters keep those dependencies isolated from the core.

## Why another runtime?

Robot policy deployment has concerns that a plain model `forward()` call does
not cover:

- incoming camera and state streams can outrun inference;
- a prediction can be correct but too old to execute;
- action chunks overlap and newer predictions should replace stale ones;
- delta, relative, and absolute actions must not be confused;
- model output must pass finite-value, joint-limit, and slew-rate checks;
- robot, simulator, and model code should not share one dependency graph.

This project handles those concerns without choosing a tensor engine or wire
protocol for you.

## Features

- typed multi-camera observations, proprioception, extensible tensors, and
  time-indexed action chunks;
- synchronous `Engine` with deadlines, cooperative cancellation, metrics, and
  serialized access to non-reentrant accelerator contexts;
- asynchronous `Scheduler` with bounded queues and a `keep_latest` policy;
- processor chains for preprocessing, action decoding, and safety policies;
- absolute, delta, and relative action semantics in the type system;
- `ActionBuffer` for receding-horizon execution and newest-chunk-wins merging;
- zero mandatory third-party dependencies;
- installable CMake target `embodied::infer`;
- lazy Python model registry with one directory per model family and external
  entry-point support;
- RoboTwin closed-loop integration tests at the 18D raw robot boundary;
- Linux, macOS, and Windows CI plus sanitizer coverage.

## Architecture

```text
sensors / simulator / dataset
             |
             v
       Observation (typed)
             |
     bounded Scheduler          newest observation wins
             |
   observation processors       validate / resize / normalize / tokenize
             |
          Backend               GGUF / ONNX Runtime / TensorRT / remote
             |
      action processors         decode / denormalize / safety filter
             |
        ActionChunk             time-indexed trajectory
             |
        ActionBuffer            merge overlap / discard stale actions
             |
      robot control loop
```

The core intentionally does not depend on ROS, gRPC, CUDA, or a model library.
Adapters own those dependencies and translate into `Observation`; backends own
tensor-engine details and return `ActionChunk`.

## End-to-end inference framework

The deployable system is organized as five layers. A service may run on the
4090 gateway, Orin, S100, or S600; the wire contract is the same while the
model adapter and accelerator runtime remain device-specific.

```text
Robot / RoboTwin / dataset
        |
1. Embodiment adapter: raw 18D state + camera aliases -> observation
        |
2. Inference service: protocol/version, auth, IDs, metadata, metrics, lifecycle
        |
3. Request scheduler: bounded newest-wins queue -> Engine -> processors
        |
4. Operator scheduler: preprocessing -> vision -> language/action expert
   dependency DAG with priority and CPU/GPU/NPU/IO lane limits
        |
5. Action adapter: model chunk -> raw 18D target -> ActionBuffer -> actuator
```

The service layer owns transport and lifecycle, not model tensors. The lazy
registry loads only the selected family (`mock`, `turbovla`, or `pi05`), so an
Orin process does not import S600 HBM libraries. Responses carry metadata,
action shape, representation, timing, and request IDs, allowing one RoboTwin
client to switch between TensorRT, HBM, and OpenPI services.

For Pi05, the production path is an OpenPI WebSocket service on the model host.
The backend repacks the canonical request into `state`, `images`, and `prompt`,
disables websocket keepalive during long accelerator calls by default, and
projects the native 16-step response to the shared 14D model contract. The
A800 vision HBM artifact is compiled separately with `hb_compile`; LLM/action
expert HBM remains an optional device-specific backend until its runtime and
server contract are available.

## Build

Requirements: CMake 3.20+ and a C++20 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/embodied-infer-demo
```

On multi-config generators, run the demo from `build/Release/` and pass
`-C Release` to CTest.

## Minimal use

```cpp
#include <embodied/infer/runtime.hpp>

namespace ei = embodied::infer;

auto backend = std::make_shared<ei::MockBackend>();
auto engine = std::make_shared<ei::Engine>(backend);
engine->add_action_processor(std::make_unique<ei::ActionSafetyFilter>(
    ei::ActionSafetyConfig{
        .lower_bounds = {-1.0F},
        .upper_bounds = {1.0F},
        .maximum_delta = {0.05F},
    }));

ei::Scheduler scheduler(engine);  // defaults to queue size 1, keep latest
ei::Observation obs;
obs.request_id = 1;
obs.control_step = 100;
obs.instruction = "pick up the cup";
obs.proprioception = std::vector<float>(7, 0.0F);

ei::InferenceOptions options;
options.deadline = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(100);
auto result = scheduler.submit(std::move(obs), options).get();
```

See [`examples/demo.cpp`](examples/demo.cpp) for the runnable loop and
[`docs/backend-guide.md`](docs/backend-guide.md) for a real backend template.
See [`docs/architecture.md`](docs/architecture.md) for the model directory,
plugin, robot contract, and simulator testing boundaries.

For the existing RTX 4090 RoboTwin, Jetson AGX Orin TensorRT, and S600 HBM
deployments, see [`docs/deployment.md`](docs/deployment.md). The Python
deployment package is optional and keeps model-specific dependencies out of the
C++ core.

## Download and run

The fastest end-to-end check requires no simulator, CUDA, or model download:

```bash
git clone https://github.com/zyforsure/embodied-infer.git
cd embodied-infer
python3 -m pip install -e .
embodied-infer sim
```

If the Python scripts directory is not on `PATH`, the equivalent command is
`python3 -m embodied_infer_deploy sim` (or `python -m embodied_infer_deploy sim`
on Windows).

Windows PowerShell users can run `.\scripts\install.ps1` first. The command
starts a local mock inference server, drives the built-in RoboTwin-compatible
environment, validates the 18D→14D→18D mapping, and exits with a JSON summary.
Use `embodied-infer doctor --profile orin|s600-remote|s100|pi05` before connecting to
hardware. `embodied-infer real` is deliberately read-only; CAN motion is never
enabled by a download or by a default command.

For the actual simulator, replace the demo factory with RoboTwin's own factory:

```bash
embodied-infer robotwin --env-factory my_robotwin_entry:create_env \
  --host 192.168.10.162 --port 44091 --exec-horizon 1
```

Platform-specific notes and private model-artifact configuration are in
[`deploy/`](deploy/) and [`docs/deployment.md`](docs/deployment.md).

That deployment has two intentionally different dimensionalities: the
RoboTwin/S600 boundary is 18D, while the deployed TurboVLA engine consumes and
produces 14D model tensors. The adapters perform the explicit 18-to-14 mapping
and hold every coordinate not predicted by the model.

## Design choices from existing work

The runtime was designed after examining:

- [Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp), especially its
  typed adapter boundary, model abstraction, and phase timing;
- [LeRobot asynchronous inference](https://github.com/huggingface/lerobot/tree/main/src/lerobot/async_inference),
  which uses a one-element observation queue and replaces the oldest pending
  observation;
- [LeRobot Real-Time Chunking](https://github.com/huggingface/lerobot/blob/main/docs/source/policy_rtc_README.md)
  and the [RTC paper](https://arxiv.org/abs/2506.07339), which motivate
  time-aware overlapping chunks;
- [OpenPI remote inference](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)
  and its action chunk broker;
- [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T), particularly policy
  validation and absolute/relative/delta action representations.

No source from those projects is vendored here. See
[`docs/design.md`](docs/design.md) for decisions and boundaries.

## Roadmap

- ONNX Runtime backend with pinned-buffer reuse;
- TensorRT backend with CUDA graph capture;
- GGUF bridge for Embodied.cpp-compatible model implementations;
- optional gRPC transport and ROS 2 adapters;
- RTC-style flow-policy prefix conditioning behind an optional interface;
- tracing exporters and latency histograms.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
