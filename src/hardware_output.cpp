#include "embodied/infer/hardware_output.hpp"

#include <cmath>
#include <stdexcept>

namespace embodied::infer {
namespace {

bool finite(const std::vector<float>& values) {
    for (const auto value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

}  // namespace

Status HardwareOutputSpec::validate() const {
    if (joint_count == 0) {
        return {StatusCode::invalid_argument, "joint_count must be greater than zero"};
    }
    if (frame.empty()) {
        return {StatusCode::invalid_argument, "hardware output frame must not be empty"};
    }
    return Status::success();
}

Status HardwareCommand::validate(const HardwareOutputSpec& spec) const {
    const auto spec_status = spec.validate();
    if (!spec_status.ok()) return spec_status;
    if (type != spec.type || mode != spec.mode || frame != spec.frame) {
        return {StatusCode::invalid_argument, "hardware command contract does not match spec"};
    }
    const auto primary_size = spec.primary_dimension();
    const bool primary_required = mode != HardwareControlMode::torque;
    if (primary_required && primary.size() != primary_size) {
        return {StatusCode::invalid_argument, "hardware primary command has invalid dimension"};
    }
    if (!primary.empty() && !finite(primary)) {
        return {StatusCode::invalid_argument, "hardware primary command contains NaN or infinity"};
    }
    const bool torque_required = mode != HardwareControlMode::position;
    if (torque_required && torque.size() != spec.joint_count) {
        return {StatusCode::invalid_argument, "hardware torque command has invalid dimension"};
    }
    if (!torque.empty() && !finite(torque)) {
        return {StatusCode::invalid_argument, "hardware torque command contains NaN or infinity"};
    }
    if (type == HardwareOutputType::end_effector_pose && primary_required) {
        const auto norm = std::sqrt(primary[3] * primary[3] + primary[4] * primary[4] +
                                    primary[5] * primary[5] + primary[6] * primary[6]);
        if (!std::isfinite(norm) || norm < 1.0e-5F) {
            return {StatusCode::invalid_argument, "end-effector quaternion must be non-zero"};
        }
    }
    return Status::success();
}

HardwareOutputAdapter::HardwareOutputAdapter(HardwareOutputSpec spec)
    : spec_(std::move(spec)) {
    const auto status = spec_.validate();
    if (!status.ok()) throw std::invalid_argument(status.message());
}

Result<HardwareCommand> HardwareOutputAdapter::encode(
    const ActionChunk& actions,
    std::size_t step,
    std::uint64_t control_step,
    std::uint64_t timestamp_ns) const {
    if (step >= actions.steps) {
        return Result<HardwareCommand>::failure(
            {StatusCode::invalid_argument, "hardware action step is out of range"});
    }
    const auto expected = spec_.command_dimension();
    if (actions.action_dim != expected || actions.values.size() != actions.steps * actions.action_dim) {
        return Result<HardwareCommand>::failure(
            {StatusCode::invalid_argument, "action chunk dimension does not match hardware output spec"});
    }
    HardwareCommand command;
    command.control_step = control_step;
    command.timestamp_ns = timestamp_ns;
    command.type = spec_.type;
    command.mode = spec_.mode;
    command.frame = spec_.frame;
    const auto offset = step * actions.action_dim;
    const auto primary_size = spec_.primary_dimension();
    if (spec_.mode != HardwareControlMode::torque) {
        command.primary.assign(actions.values.begin() + static_cast<std::ptrdiff_t>(offset),
                               actions.values.begin() + static_cast<std::ptrdiff_t>(offset + primary_size));
    }
    if (spec_.mode != HardwareControlMode::position) {
        const auto torque_offset = offset + (spec_.mode == HardwareControlMode::position_torque ? primary_size : 0);
        command.torque.assign(actions.values.begin() + static_cast<std::ptrdiff_t>(torque_offset),
                              actions.values.begin() + static_cast<std::ptrdiff_t>(torque_offset + spec_.joint_count));
    }
    const auto status = command.validate(spec_);
    if (!status.ok()) return Result<HardwareCommand>::failure(status);
    return Result<HardwareCommand>::success(std::move(command));
}

}  // namespace embodied::infer
