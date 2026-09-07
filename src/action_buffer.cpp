#include "embodied/infer/action_buffer.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>

namespace embodied::infer {

ActionBuffer::ActionBuffer(std::size_t action_dim,
                           std::size_t maximum_buffered_steps)
    : action_dim_(action_dim),
      maximum_buffered_steps_(maximum_buffered_steps) {}

Status ActionBuffer::push(const ActionChunk& chunk) {
    if (action_dim_ == 0 || maximum_buffered_steps_ == 0) {
        return {StatusCode::invalid_argument,
                "action buffer dimensions must be non-zero"};
    }
    if (chunk.action_dim != action_dim_ || chunk.steps == 0 ||
        chunk.values.size() != chunk.steps * chunk.action_dim) {
        return {StatusCode::invalid_argument,
                "action chunk shape does not match the buffer"};
    }

    std::lock_guard lock(mutex_);
    if (chunk.request_id < latest_request_id_) {
        return {StatusCode::cancelled,
                "action chunk is older than the latest buffered request"};
    }
    latest_request_id_ = std::max(latest_request_id_, chunk.request_id);
    for (std::size_t index = 0; index < chunk.steps; ++index) {
        const auto step = chunk.first_control_step + index;
        if (step < minimum_control_step_) {
            continue;
        }
        BufferedAction action;
        action.control_step = step;
        action.request_id = chunk.request_id;
        action.representation = chunk.representation;
        const auto begin = chunk.values.begin() +
                           static_cast<std::ptrdiff_t>(index * action_dim_);
        action.values.assign(begin,
                             begin + static_cast<std::ptrdiff_t>(action_dim_));
        auto found = actions_.find(step);
        if (found == actions_.end() || found->second.request_id <= chunk.request_id) {
            actions_[step] = std::move(action);
        }
    }

    while (actions_.size() > maximum_buffered_steps_) {
        actions_.erase(std::prev(actions_.end()));
    }
    return Status::success();
}

Result<BufferedAction> ActionBuffer::pop(std::uint64_t control_step) {
    std::lock_guard lock(mutex_);
    actions_.erase(actions_.begin(), actions_.lower_bound(control_step));
    const auto next_step = control_step ==
            std::numeric_limits<std::uint64_t>::max()
        ? control_step
        : control_step + 1;
    minimum_control_step_ = std::max(minimum_control_step_, next_step);
    auto found = actions_.find(control_step);
    if (found == actions_.end()) {
        return Result<BufferedAction>::failure(
            {StatusCode::not_ready, "no action is ready for this control step"});
    }
    auto action = std::move(found->second);
    actions_.erase(found);
    return Result<BufferedAction>::success(std::move(action));
}

void ActionBuffer::clear() noexcept {
    std::lock_guard lock(mutex_);
    actions_.clear();
    minimum_control_step_ = 0;
    latest_request_id_ = 0;
}

std::size_t ActionBuffer::size() const noexcept {
    std::lock_guard lock(mutex_);
    return actions_.size();
}

}  // namespace embodied::infer
