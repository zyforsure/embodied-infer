#pragma once

#include "embodied/infer/status.hpp"
#include "embodied/infer/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace embodied::infer {

enum class HardwareOutputType { joint, end_effector_pose };

// `position_torque` is useful for impedance controllers. `torque` alone is a
// valid command for torque-controlled joints; pose commands always carry a
// pose and may optionally carry a separate joint-torque vector.
enum class HardwareControlMode { position, torque, position_torque };

struct HardwareOutputSpec {
    HardwareOutputType type = HardwareOutputType::joint;
    HardwareControlMode mode = HardwareControlMode::position;
    std::size_t joint_count = 0;
    std::string frame = "base";

    [[nodiscard]] std::size_t pose_dimension() const noexcept {
        return type == HardwareOutputType::end_effector_pose ? 7 : 0;
    }
    [[nodiscard]] std::size_t primary_dimension() const noexcept {
        return type == HardwareOutputType::joint ? joint_count : 7;
    }
    [[nodiscard]] std::size_t command_dimension() const noexcept {
        const auto primary = primary_dimension();
        if (mode == HardwareControlMode::position) return primary;
        if (mode == HardwareControlMode::torque) {
            return type == HardwareOutputType::joint ? joint_count : joint_count;
        }
        return primary + joint_count;
    }

    [[nodiscard]] Status validate() const;
};

struct HardwareCommand {
    std::uint64_t control_step = 0;
    std::uint64_t timestamp_ns = 0;
    HardwareOutputType type = HardwareOutputType::joint;
    HardwareControlMode mode = HardwareControlMode::position;
    std::string frame = "base";
    // Joint positions for joint output, or [x,y,z,qx,qy,qz,qw] for pose output.
    std::vector<float> primary;
    // Optional joint torques. Empty means that no torque command is present.
    std::vector<float> torque;

    [[nodiscard]] Status validate(const HardwareOutputSpec& spec) const;
};

class HardwareOutputAdapter {
public:
    explicit HardwareOutputAdapter(HardwareOutputSpec spec);

    [[nodiscard]] const HardwareOutputSpec& spec() const noexcept {
        return spec_;
    }

    // Encodes one action step. The ActionChunk representation is preserved;
    // coordinate transforms (IK, calibration, impedance) belong in a robot
    // plugin and must not be hidden in this generic boundary.
    [[nodiscard]] Result<HardwareCommand> encode(
        const ActionChunk& actions,
        std::size_t step,
        std::uint64_t control_step,
        std::uint64_t timestamp_ns = 0) const;

private:
    HardwareOutputSpec spec_;
};

}  // namespace embodied::infer
