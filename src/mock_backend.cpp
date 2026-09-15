#include "embodied/infer/mock_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <thread>

namespace embodied::infer {
namespace {

bool deadline_reached(const InferenceOptions& options) {
    return options.deadline.has_value() &&
           std::chrono::steady_clock::now() >= *options.deadline;
}

}  // namespace

MockBackend::MockBackend(MockBackendConfig config) : config_(std::move(config)) {}

std::string_view MockBackend::name() const noexcept {
    return "mock";
}

const ModelSpec& MockBackend::spec() const noexcept {
    return config_.spec;
}

Result<ActionChunk> MockBackend::infer(const Observation& observation,
                                       const InferenceOptions& options) {
    auto remaining = config_.simulated_latency;
    constexpr auto quantum = std::chrono::milliseconds(1);
    while (remaining.count() > 0) {
        if (options.cancellation.cancellation_requested()) {
            return Result<ActionChunk>::failure(
                {StatusCode::cancelled, "mock inference was cancelled"});
        }
        if (deadline_reached(options)) {
            return Result<ActionChunk>::failure(
                {StatusCode::deadline_exceeded,
                 "mock inference deadline exceeded"});
        }
        const auto delay = std::min(remaining, quantum);
        std::this_thread::sleep_for(delay);
        remaining -= delay;
    }

    ActionChunk actions;
    actions.steps = config_.spec.action_horizon;
    actions.action_dim = config_.spec.action_dim;
    actions.values.resize(actions.steps * actions.action_dim);

    const float instruction_bias = static_cast<float>(
        observation.instruction.size() % 7U) * 0.01F;
    for (std::size_t dimension = 0; dimension < actions.action_dim;
         ++dimension) {
        const float start = dimension < observation.proprioception.size()
            ? observation.proprioception[dimension]
            : 0.0F;
        const float target = std::tanh(start) + instruction_bias;
        for (std::size_t step = 0; step < actions.steps; ++step) {
            const float alpha = static_cast<float>(step + 1) /
                                static_cast<float>(actions.steps);
            actions.at(step, dimension) = start + alpha * (target - start);
        }
    }
    return Result<ActionChunk>::success(std::move(actions));
}

}  // namespace embodied::infer
