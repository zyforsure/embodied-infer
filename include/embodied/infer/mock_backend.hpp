#pragma once

#include "embodied/infer/backend.hpp"

#include <chrono>

namespace embodied::infer {

struct MockBackendConfig {
    ModelSpec spec{"mock-vla", 7, 7, 8, 0};
    std::chrono::milliseconds simulated_latency{0};
};

class MockBackend final : public Backend {
public:
    explicit MockBackend(MockBackendConfig config = {});

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] const ModelSpec& spec() const noexcept override;
    Result<ActionChunk> infer(const Observation& observation,
                              const InferenceOptions& options) override;

private:
    MockBackendConfig config_;
};

}  // namespace embodied::infer
