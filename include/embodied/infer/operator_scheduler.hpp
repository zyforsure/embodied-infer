#pragma once

#include "embodied/infer/status.hpp"
#include "embodied/infer/types.hpp"

#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace embodied::infer {

// A logical execution lane. Backends may map gpu/npu to a concrete stream or
// accelerator context; the core deliberately does not depend on CUDA/HBDK.
enum class OperatorDevice { cpu, gpu, npu, io };

using OperatorFunction = std::function<Status()>;

struct OperatorSpec {
    std::string name;
    OperatorDevice device = OperatorDevice::cpu;
    int priority = 0;  // larger values are dispatched first when ready
    std::vector<std::size_t> dependencies;
    OperatorFunction function;
};

class OperatorGraph {
public:
    // Returns the stable node id used by subsequent dependencies.
    std::size_t add(OperatorSpec spec);

    [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }
    [[nodiscard]] const OperatorSpec& at(std::size_t id) const;
    [[nodiscard]] const std::vector<OperatorSpec>& nodes() const noexcept {
        return nodes_;
    }

    // Checks references, duplicate edges, and cycles before execution.
    [[nodiscard]] Status validate() const;

private:
    std::vector<OperatorSpec> nodes_;
};

struct OperatorSchedulerConfig {
    std::size_t cpu_workers = 1;
    std::size_t gpu_workers = 1;
    std::size_t npu_workers = 1;
    std::size_t io_workers = 1;
};

struct OperatorScheduleOptions {
    std::optional<std::chrono::steady_clock::time_point> deadline;
    CancellationToken cancellation;
};

struct OperatorReport {
    std::vector<std::string> completed;
    std::string failed_operator;
    std::size_t peak_parallelism = 0;
};

class OperatorScheduler {
public:
    explicit OperatorScheduler(OperatorSchedulerConfig config = {});

    [[nodiscard]] Result<OperatorReport> run(
        const OperatorGraph& graph,
        const OperatorScheduleOptions& options = {}) const;

private:
    OperatorSchedulerConfig config_;
};

}  // namespace embodied::infer
