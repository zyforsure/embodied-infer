# Backend guide

A backend owns model loading, tensor allocation, preprocessing that cannot be
expressed as a generic processor, and accelerator execution. It exposes the
resolved model contract through `spec()` and implements one inference call.

```cpp
#include <embodied/infer/backend.hpp>

class OnnxVlaBackend final : public embodied::infer::Backend {
public:
    explicit OnnxVlaBackend(const std::filesystem::path& model_path) {
        // Load sessions and inspect model metadata here.
        spec_ = {
            "my-vla", 14, 14, 50, 3, 20'000'000,
            embodied::infer::ActionRepresentation::delta,
        };
    }

    std::string_view name() const noexcept override { return "onnxruntime"; }
    const embodied::infer::ModelSpec& spec() const noexcept override {
        return spec_;
    }

    embodied::infer::Result<embodied::infer::ActionChunk> infer(
        const embodied::infer::Observation& observation,
        const embodied::infer::InferenceOptions& options) override {
        if (options.cancellation.cancellation_requested()) {
            return embodied::infer::Result<embodied::infer::ActionChunk>::failure(
                {embodied::infer::StatusCode::cancelled, "cancelled"});
        }

        // Convert inputs, run the session, and copy or move output values.
        embodied::infer::ActionChunk output;
        output.steps = spec_.action_horizon;
        output.action_dim = spec_.action_dim;
        output.values.resize(output.steps * output.action_dim);
        return embodied::infer::Result<embodied::infer::ActionChunk>::success(
            std::move(output));
    }

private:
    embodied::infer::ModelSpec spec_;
};
```

The engine fills request, timing, and action-representation metadata from the
observation and `ModelSpec`. The backend must fill only the action shape,
values, and optional model-specific outputs.

## Integration checklist

1. Validate checkpoint metadata during construction and publish exact
   dimensions in `ModelSpec`.
2. Reuse input/output buffers; do not allocate large tensors on every call.
3. Check `deadline` before expensive phases and `CancellationToken` where the tensor
   engine permits cooperative cancellation.
4. Return `backend_error` for runtime execution failures and malformed output.
5. Put normalization and tokenization in processors when they are robot/model
   configuration, or inside the backend when they are inseparable from the
   model artifact.
6. Add `DeltaToAbsolute` before safety filtering for delta-action policies.
7. Test exact tensor names, shapes, action ordering, and numerical parity
   against the model's reference implementation.
