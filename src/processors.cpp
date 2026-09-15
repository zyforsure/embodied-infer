#include "embodied/infer/processors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace embodied::infer {
namespace {

std::size_t channels(PixelFormat format) {
    switch (format) {
        case PixelFormat::rgb_u8:
            return 3;
        case PixelFormat::rgba_u8:
            return 4;
        case PixelFormat::gray_u8:
            return 1;
    }
    return 0;
}

bool all_finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

Status validate_vector_size(const std::vector<float>& values,
                            std::size_t dimensions,
                            const char* name) {
    if (!values.empty() && values.size() != 1 && values.size() != dimensions) {
        return {StatusCode::invalid_argument,
                std::string(name) + " must be empty, scalar, or action_dim sized"};
    }
    return Status::success();
}

float element_or(const std::vector<float>& values,
                 std::size_t index,
                 float fallback) {
    if (values.empty()) {
        return fallback;
    }
    return values.size() == 1 ? values.front() : values[index];
}

}  // namespace

ObservationValidator::ObservationValidator(ObservationValidationConfig config)
    : config_(config) {}

Status ObservationValidator::process(Observation& observation,
                                     const ModelSpec& spec) {
    if (spec.action_dim == 0 || spec.action_horizon == 0) {
        return {StatusCode::not_ready,
                "backend model spec has an empty action shape"};
    }
    if (config_.require_instruction && observation.instruction.empty()) {
        return {StatusCode::invalid_argument, "instruction must not be empty"};
    }
    if (observation.proprioception.size() != spec.state_dim) {
        return {StatusCode::invalid_argument,
                "proprioception size does not match model state_dim"};
    }
    if (observation.images.size() < spec.minimum_images) {
        return {StatusCode::invalid_argument,
                "observation has fewer images than the model requires"};
    }
    if (config_.reject_non_finite &&
        !all_finite(observation.proprioception)) {
        return {StatusCode::invalid_argument,
                "proprioception contains NaN or infinity"};
    }

    for (const auto& image : observation.images) {
        if (image.width == 0 || image.height == 0) {
            return {StatusCode::invalid_argument,
                    "image width and height must be non-zero"};
        }
        const auto pixel_count = static_cast<std::uint64_t>(image.width) *
                                 static_cast<std::uint64_t>(image.height);
        const auto channel_count = channels(image.format);
        if (channel_count == 0 ||
            pixel_count > std::numeric_limits<std::uint64_t>::max() /
                              channel_count) {
            return {StatusCode::invalid_argument,
                    "image shape overflows its byte count"};
        }
        const auto byte_count = pixel_count * channel_count;
        if (byte_count > config_.maximum_image_bytes) {
            return {StatusCode::invalid_argument,
                    "image exceeds maximum_image_bytes"};
        }
        if (image.data.size() != byte_count) {
            return {StatusCode::invalid_argument,
                    "image data size does not match its shape and format"};
        }
    }

    for (const auto& [name, tensor] : observation.extra_inputs) {
        std::uint64_t elements = 1;
        for (const auto extent : tensor.shape) {
            if (extent <= 0 ||
                elements > std::numeric_limits<std::uint64_t>::max() /
                               static_cast<std::uint64_t>(extent)) {
                return {StatusCode::invalid_argument,
                        "extra input '" + name + "' has an invalid shape"};
            }
            elements *= static_cast<std::uint64_t>(extent);
        }
        if (elements != tensor.values.size()) {
            return {StatusCode::invalid_argument,
                    "extra input '" + name + "' shape does not match its data"};
        }
        if (config_.reject_non_finite && !all_finite(tensor.values)) {
            return {StatusCode::invalid_argument,
                    "extra input '" + name + "' contains NaN or infinity"};
        }
    }
    return Status::success();
}

Status DeltaToAbsolute::process(const Observation& observation,
                                ActionChunk& actions,
                                const ModelSpec& spec) {
    if (actions.representation != ActionRepresentation::delta) {
        return Status::success();
    }
    if (actions.action_dim != spec.action_dim ||
        actions.values.size() != actions.steps * actions.action_dim ||
        observation.proprioception.size() < actions.action_dim) {
        return {StatusCode::invalid_argument,
                "delta actions cannot be decoded with the supplied state"};
    }

    std::vector<float> previous(observation.proprioception.begin(),
                                observation.proprioception.begin() +
                                    static_cast<std::ptrdiff_t>(
                                        actions.action_dim));
    for (std::size_t step = 0; step < actions.steps; ++step) {
        for (std::size_t dimension = 0; dimension < actions.action_dim;
             ++dimension) {
            previous[dimension] += actions.at(step, dimension);
            actions.at(step, dimension) = previous[dimension];
        }
    }
    actions.representation = ActionRepresentation::absolute;
    return Status::success();
}

ActionSafetyFilter::ActionSafetyFilter(ActionSafetyConfig config)
    : config_(std::move(config)) {}

Status ActionSafetyFilter::process(const Observation& observation,
                                   ActionChunk& actions,
                                   const ModelSpec& spec) {
    if (actions.representation != ActionRepresentation::absolute) {
        return {StatusCode::invalid_argument,
                "action safety filtering requires absolute actions"};
    }
    if (actions.action_dim != spec.action_dim ||
        actions.steps != spec.action_horizon ||
        actions.values.size() != actions.steps * actions.action_dim) {
        return {StatusCode::backend_error,
                "backend returned an action chunk with an invalid shape"};
    }

    for (const auto* pair : {
             &config_.lower_bounds,
             &config_.upper_bounds,
             &config_.maximum_delta}) {
        const char* name = pair == &config_.lower_bounds
            ? "lower_bounds"
            : pair == &config_.upper_bounds ? "upper_bounds" : "maximum_delta";
        Status status = validate_vector_size(*pair, spec.action_dim, name);
        if (!status.ok()) {
            return status;
        }
    }

    std::vector<float> previous(spec.action_dim, 0.0F);
    for (std::size_t dimension = 0; dimension < spec.action_dim; ++dimension) {
        if (dimension < observation.proprioception.size()) {
            previous[dimension] = observation.proprioception[dimension];
        }
    }

    for (std::size_t step = 0; step < actions.steps; ++step) {
        for (std::size_t dimension = 0; dimension < actions.action_dim;
             ++dimension) {
            float value = actions.at(step, dimension);
            if (!std::isfinite(value)) {
                if (config_.reject_non_finite) {
                    return {StatusCode::backend_error,
                            "backend returned NaN or infinity"};
                }
                value = previous[dimension];
            }

            const float lower = element_or(
                config_.lower_bounds, dimension,
                -std::numeric_limits<float>::infinity());
            const float upper = element_or(
                config_.upper_bounds, dimension,
                std::numeric_limits<float>::infinity());
            const float delta = element_or(
                config_.maximum_delta, dimension,
                std::numeric_limits<float>::infinity());
            if (lower > upper || delta < 0.0F) {
                return {StatusCode::invalid_argument,
                        "action safety bounds are inconsistent"};
            }

            value = std::clamp(value, lower, upper);
            value = std::clamp(value,
                               previous[dimension] - delta,
                               previous[dimension] + delta);
            value = std::clamp(value, lower, upper);
            actions.at(step, dimension) = value;
            previous[dimension] = value;
        }
    }
    return Status::success();
}

}  // namespace embodied::infer
