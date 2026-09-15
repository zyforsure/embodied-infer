#include "embodied/infer/runtime.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

int main() {
    namespace ei = embodied::infer;
    using namespace std::chrono_literals;

    ei::MockBackendConfig backend_config;
    backend_config.spec = {
        "demo-vla", 7, 7, 8, 0, 20'000'000, ei::ActionRepresentation::absolute};
    backend_config.simulated_latency = 2ms;

    auto backend = std::make_shared<ei::MockBackend>(backend_config);
    auto engine = std::make_shared<ei::Engine>(backend);
    engine->add_action_processor(std::make_unique<ei::ActionSafetyFilter>(
        ei::ActionSafetyConfig{
            .lower_bounds = {-1.0F},
            .upper_bounds = {1.0F},
            .maximum_delta = {0.05F},
        }));

    ei::Scheduler scheduler(engine);
    ei::Observation observation;
    observation.request_id = 42;
    observation.control_step = 100;
    observation.timestamp_ns = 123456789;
    observation.instruction = "place the red block in the tray";
    observation.proprioception = {0.0F, 0.1F, -0.1F, 0.2F,
                                  0.0F, 0.05F, 0.0F};

    ei::InferenceOptions options;
    options.deadline = std::chrono::steady_clock::now() + 50ms;
    auto result = scheduler.submit(std::move(observation), options).get();
    if (!result.ok()) {
        std::cerr << "inference failed: " << result.status().message() << '\n';
        return 1;
    }

    auto chunk = std::move(result).value();
    ei::ActionBuffer action_buffer(chunk.action_dim);
    if (const auto status = action_buffer.push(chunk); !status.ok()) {
        std::cerr << "buffer failed: " << status.message() << '\n';
        return 1;
    }

    std::cout << "model=" << engine->spec().name
              << " request=" << chunk.request_id
              << " shape=[" << chunk.steps << ',' << chunk.action_dim << "]\n";
    auto action = action_buffer.pop(chunk.first_control_step);
    if (!action.ok()) {
        std::cerr << "action unavailable: " << action.status().message() << '\n';
        return 1;
    }

    std::cout << "first_action=";
    for (float value : action.value().values) {
        std::cout << ' ' << std::fixed << std::setprecision(3) << value;
    }
    const auto metrics = engine->metrics();
    std::cout << "\navg_latency_ms=" << std::setprecision(2)
              << metrics.average_latency_ms << '\n';
}
