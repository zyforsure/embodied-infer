#pragma once

#include "embodied/infer/status.hpp"
#include "embodied/infer/types.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

namespace embodied::infer {

struct BufferedAction {
    std::uint64_t control_step = 0;
    std::uint64_t request_id = 0;
    ActionRepresentation representation = ActionRepresentation::absolute;
    std::vector<float> values;
};

// Merges overlapping chunks by control step. A newer prediction replaces an
// older prediction for the same future step, matching receding-horizon use.
class ActionBuffer {
public:
    explicit ActionBuffer(std::size_t action_dim,
                          std::size_t maximum_buffered_steps = 256);

    Status push(const ActionChunk& chunk);
    Result<BufferedAction> pop(std::uint64_t control_step);
    void clear() noexcept;

    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::size_t action_dim_;
    std::size_t maximum_buffered_steps_;
    mutable std::mutex mutex_;
    std::map<std::uint64_t, BufferedAction> actions_;
    std::uint64_t minimum_control_step_ = 0;
    std::uint64_t latest_request_id_ = 0;
};

}  // namespace embodied::infer
