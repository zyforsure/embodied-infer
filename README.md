# embodied-infer

`embodied-infer` is a small C++20 runtime foundation for real-time embodied
AI policy inference. It gives robot applications one stable boundary for
observations and action chunks while model execution remains replaceable.

The repository currently includes a complete, dependency-free runtime core
and a deterministic mock backend. It does **not** claim built-in execution of
pi0, GR00T, OpenVLA, or other production checkpoints yet. Those integrations
belong in backend modules built on the public `Backend` interface.

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
