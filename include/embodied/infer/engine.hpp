#pragma once

#include "embodied/infer/backend.hpp"
#include "embodied/infer/processors.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace embodied::infer {

struct MetricsSnapshot {
    std::uint64_t requests = 0;
    std::uint64_t succeeded = 0;
    std::uint64_t failed = 0;
    std::uint64_t deadline_exceeded = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t dropped = 0;
    double average_latency_ms = 0.0;
    double maximum_latency_ms = 0.0;
};

class Engine {
public:
    explicit Engine(std::shared_ptr<Backend> backend);

    Engine& add_observation_processor(
        std::unique_ptr<ObservationProcessor> processor);
    Engine& add_action_processor(std::unique_ptr<ActionProcessor> processor);

    Result<ActionChunk> infer(Observation observation,
                              const InferenceOptions& options = {});
    Status reset();

    [[nodiscard]] const ModelSpec& spec() const noexcept;
    [[nodiscard]] MetricsSnapshot metrics() const noexcept;

    void record_dropped() noexcept;

private:
    void record_result(const Status& status, double latency_ms) noexcept;

    std::shared_ptr<Backend> backend_;
    std::vector<std::unique_ptr<ObservationProcessor>> observation_processors_;
    std::vector<std::unique_ptr<ActionProcessor>> action_processors_;
    mutable std::mutex backend_mutex_;

    std::atomic<std::uint64_t> requests_{0};
    std::atomic<std::uint64_t> succeeded_{0};
    std::atomic<std::uint64_t> failed_{0};
    std::atomic<std::uint64_t> deadline_exceeded_{0};
    std::atomic<std::uint64_t> cancelled_{0};
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<std::uint64_t> latency_us_{0};
    std::atomic<std::uint64_t> maximum_latency_us_{0};
};

}  // namespace embodied::infer
