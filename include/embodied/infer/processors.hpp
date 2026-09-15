#pragma once

#include "embodied/infer/status.hpp"
#include "embodied/infer/types.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace embodied::infer {

class ObservationProcessor {
public:
    virtual ~ObservationProcessor() = default;
    virtual Status reset() { return Status::success(); }
    virtual Status process(Observation& observation,
                           const ModelSpec& spec) = 0;
};

class ActionProcessor {
public:
    virtual ~ActionProcessor() = default;
    virtual Status reset() { return Status::success(); }
    virtual Status process(const Observation& observation,
                           ActionChunk& actions,
                           const ModelSpec& spec) = 0;
};

struct ObservationValidationConfig {
    bool require_instruction = true;
    bool reject_non_finite = true;
    std::size_t maximum_image_bytes = 64U * 1024U * 1024U;
};

class ObservationValidator final : public ObservationProcessor {
public:
    explicit ObservationValidator(ObservationValidationConfig config = {});
    Status process(Observation& observation, const ModelSpec& spec) override;

private:
    ObservationValidationConfig config_;
};

class DeltaToAbsolute final : public ActionProcessor {
public:
    Status process(const Observation& observation,
                   ActionChunk& actions,
                   const ModelSpec& spec) override;
};

struct ActionSafetyConfig {
    std::vector<float> lower_bounds;
    std::vector<float> upper_bounds;
    std::vector<float> maximum_delta;
    bool reject_non_finite = true;
};

class ActionSafetyFilter final : public ActionProcessor {
public:
    explicit ActionSafetyFilter(ActionSafetyConfig config);
    Status process(const Observation& observation,
                   ActionChunk& actions,
                   const ModelSpec& spec) override;

private:
    ActionSafetyConfig config_;
};

}  // namespace embodied::infer
