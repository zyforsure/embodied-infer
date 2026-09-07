#pragma once

#include "embodied/infer/status.hpp"
#include "embodied/infer/types.hpp"

#include <string_view>

namespace embodied::infer {

class Backend {
public:
    virtual ~Backend() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual const ModelSpec& spec() const noexcept = 0;
    virtual Result<ActionChunk> infer(const Observation& observation,
                                      const InferenceOptions& options) = 0;
    virtual Status reset() { return Status::success(); }
};

}  // namespace embodied::infer
