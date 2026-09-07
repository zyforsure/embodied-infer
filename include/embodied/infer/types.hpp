#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace embodied::infer {

class CancellationToken {
public:
    CancellationToken() = default;

    [[nodiscard]] bool cancellation_requested() const noexcept {
        return state_ && state_->load(std::memory_order_relaxed);
    }

private:
    explicit CancellationToken(std::shared_ptr<std::atomic_bool> state)
        : state_(std::move(state)) {}

    std::shared_ptr<std::atomic_bool> state_;
    friend class CancellationSource;
};

class CancellationSource {
public:
    CancellationSource() : state_(std::make_shared<std::atomic_bool>(false)) {}

    [[nodiscard]] CancellationToken token() const {
        return CancellationToken(state_);
    }

    void request_cancellation() noexcept {
        state_->store(true, std::memory_order_relaxed);
    }

private:
    std::shared_ptr<std::atomic_bool> state_;
};

enum class PixelFormat {
    rgb_u8,
    rgba_u8,
    gray_u8,
};

struct Image {
    std::string name;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    PixelFormat format = PixelFormat::rgb_u8;
    std::vector<std::byte> data;
    std::uint64_t timestamp_ns = 0;
};

struct Tensor {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
};

struct Observation {
    std::uint64_t request_id = 0;
    std::uint64_t control_step = 0;
    std::uint64_t timestamp_ns = 0;
    std::string instruction;
    std::vector<Image> images;
    std::vector<float> proprioception;
    std::unordered_map<std::string, Tensor> extra_inputs;
};

enum class ActionRepresentation {
    absolute,
    delta,
    relative,
};

struct ActionChunk {
    std::uint64_t request_id = 0;
    std::uint64_t first_control_step = 0;
    std::uint64_t source_timestamp_ns = 0;
    std::uint64_t control_period_ns = 0;
    std::size_t steps = 0;
    std::size_t action_dim = 0;
    ActionRepresentation representation = ActionRepresentation::absolute;
    std::vector<float> values;
    std::unordered_map<std::string, Tensor> extra_outputs;

    [[nodiscard]] float& at(std::size_t step, std::size_t dimension) {
        return values.at(step * action_dim + dimension);
    }

    [[nodiscard]] const float& at(std::size_t step,
                                  std::size_t dimension) const {
        return values.at(step * action_dim + dimension);
    }
};

struct ModelSpec {
    std::string name;
    std::size_t state_dim = 0;
    std::size_t action_dim = 0;
    std::size_t action_horizon = 0;
    std::size_t minimum_images = 0;
    std::uint64_t control_period_ns = 0;
    ActionRepresentation action_representation =
        ActionRepresentation::absolute;
};

struct InferenceOptions {
    std::optional<std::chrono::steady_clock::time_point> deadline;
    CancellationToken cancellation;
};

}  // namespace embodied::infer
