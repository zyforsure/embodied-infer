#include "embodied/infer/runtime.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <atomic>

namespace ei = embodied::infer;
using namespace std::chrono_literals;

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __FILE__ << ':' << __LINE__                            \
                      << ": check failed: " #condition << '\n';                \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

ei::Observation valid_observation(std::uint64_t request_id = 1,
                                  std::uint64_t control_step = 10) {
    ei::Observation observation;
    observation.request_id = request_id;
    observation.control_step = control_step;
    observation.timestamp_ns = 987654321;
    observation.instruction = "move";
    observation.proprioception = {0.0F, 0.1F, -0.1F};
    return observation;
}

std::shared_ptr<ei::MockBackend> mock_backend(
    std::chrono::milliseconds latency = 0ms,
    ei::ActionRepresentation representation =
        ei::ActionRepresentation::absolute) {
    ei::MockBackendConfig config;
    config.spec = {"test", 3, 3, 4, 0, 10'000'000, representation};
    config.simulated_latency = latency;
    return std::make_shared<ei::MockBackend>(config);
}

void test_validation_and_metrics() {
    auto engine = std::make_shared<ei::Engine>(mock_backend());
    auto invalid = valid_observation();
    invalid.proprioception[1] = std::numeric_limits<float>::quiet_NaN();
    const auto bad = engine->infer(std::move(invalid));
    CHECK(!bad.ok());
    CHECK(bad.status().code() == ei::StatusCode::invalid_argument);

    const auto good = engine->infer(valid_observation(2, 20));
    CHECK(good.ok());
    CHECK(good.value().request_id == 2);
    CHECK(good.value().first_control_step == 20);
    CHECK(good.value().steps == 4);
    CHECK(good.value().action_dim == 3);
    CHECK(good.value().control_period_ns == 10'000'000);

    const auto metrics = engine->metrics();
    CHECK(metrics.requests == 2);
    CHECK(metrics.succeeded == 1);
    CHECK(metrics.failed == 1);
}

void test_deadline_and_cancellation() {
    auto engine = std::make_shared<ei::Engine>(mock_backend(20ms));
    ei::InferenceOptions timeout;
    timeout.deadline = std::chrono::steady_clock::now() + 2ms;
    const auto expired = engine->infer(valid_observation(), timeout);
    CHECK(!expired.ok());
    CHECK(expired.status().code() == ei::StatusCode::deadline_exceeded);

    ei::CancellationSource stop;
    stop.request_cancellation();
    ei::InferenceOptions cancelled;
    cancelled.cancellation = stop.token();
    const auto result = engine->infer(valid_observation(), cancelled);
    CHECK(!result.ok());
    CHECK(result.status().code() == ei::StatusCode::cancelled);
}

void test_action_processors() {
    ei::Observation observation = valid_observation();
    ei::ActionChunk chunk;
    chunk.steps = 2;
    chunk.action_dim = 3;
    chunk.representation = ei::ActionRepresentation::delta;
    chunk.values = {0.5F, -0.5F, 0.1F, 0.5F, 0.5F, 0.1F};
    ei::ModelSpec spec{"delta", 3, 3, 2, 0, 0,
                       ei::ActionRepresentation::delta};

    ei::DeltaToAbsolute decoder;
    CHECK(decoder.process(observation, chunk, spec).ok());
    CHECK(chunk.representation == ei::ActionRepresentation::absolute);
    CHECK(std::abs(chunk.at(0, 0) - 0.5F) < 0.0001F);
    CHECK(std::abs(chunk.at(1, 0) - 1.0F) < 0.0001F);

    ei::ActionSafetyFilter safety({
        .lower_bounds = {-0.25F},
        .upper_bounds = {0.25F},
        .maximum_delta = {0.1F},
    });
    CHECK(safety.process(observation, chunk, spec).ok());
    CHECK(std::abs(chunk.at(0, 0) - 0.1F) < 0.0001F);
    CHECK(std::abs(chunk.at(1, 0) - 0.2F) < 0.0001F);
}

ei::ActionChunk chunk(std::uint64_t request_id,
                      std::uint64_t first_step,
                      float seed) {
    ei::ActionChunk result;
    result.request_id = request_id;
    result.first_control_step = first_step;
    result.steps = 3;
    result.action_dim = 2;
    result.values = {seed, seed + 1, seed + 2,
                     seed + 3, seed + 4, seed + 5};
    return result;
}

void test_action_buffer() {
    ei::ActionBuffer buffer(2, 4);
    CHECK(buffer.push(chunk(1, 10, 1.0F)).ok());
    CHECK(buffer.push(chunk(2, 11, 20.0F)).ok());
    CHECK(buffer.size() == 4);

    auto step_11 = buffer.pop(11);
    CHECK(step_11.ok());
    CHECK(step_11.value().request_id == 2);
    CHECK(step_11.value().values[0] == 20.0F);

    const auto stale = buffer.push(chunk(1, 11, 100.0F));
    CHECK(!stale.ok());
    CHECK(stale.code() == ei::StatusCode::cancelled);
    auto step_12 = buffer.pop(12);
    CHECK(step_12.ok());
    CHECK(step_12.value().request_id == 2);
    CHECK(step_12.value().values[0] == 22.0F);

    auto old = buffer.pop(10);
    CHECK(!old.ok());
    CHECK(old.status().code() == ei::StatusCode::not_ready);
}

void test_scheduler_overflow() {
    auto engine = std::make_shared<ei::Engine>(mock_backend(40ms));
    ei::Scheduler scheduler(engine, {1, ei::OverflowPolicy::keep_latest});

    auto first = scheduler.submit(valid_observation(1));
    const auto wait_until = std::chrono::steady_clock::now() + 1s;
    while (engine->metrics().requests == 0 &&
           std::chrono::steady_clock::now() < wait_until) {
        std::this_thread::sleep_for(1ms);
    }
    CHECK(engine->metrics().requests == 1);
    auto second = scheduler.submit(valid_observation(2));
    auto third = scheduler.submit(valid_observation(3));

    CHECK(first.get().ok());
    const auto replaced = second.get();
    CHECK(!replaced.ok());
    CHECK(replaced.status().code() == ei::StatusCode::cancelled);
    CHECK(third.get().ok());
    CHECK(engine->metrics().dropped == 1);
}

void test_operator_scheduler() {
    ei::OperatorGraph graph;
    std::atomic<int> parallel{0};
    std::atomic<int> peak{0};
    const auto enter = [&] {
        const auto now = ++parallel;
        auto observed = peak.load();
        while (observed < now && !peak.compare_exchange_weak(observed, now)) {
        }
    };
    const auto leave = [&] { --parallel; };
    const auto vision = graph.add({"vision", ei::OperatorDevice::gpu, 5, {}, [&] {
        enter(); std::this_thread::sleep_for(5ms); leave(); return ei::Status::success();
    }});
    const auto proprio = graph.add({"proprio", ei::OperatorDevice::cpu, 1, {}, [&] {
        enter(); std::this_thread::sleep_for(5ms); leave(); return ei::Status::success();
    }});
    graph.add({"fusion", ei::OperatorDevice::cpu, 10, {vision, proprio}, [&] {
        return ei::Status::success();
    }});
    ei::OperatorScheduler scheduler({2, 1, 1, 1});
    const auto result = scheduler.run(graph);
    CHECK(result.ok());
    CHECK(result.value().completed.size() == 3);
    CHECK(result.value().peak_parallelism >= 2);

    ei::OperatorGraph cycle;
    cycle.add({"a", ei::OperatorDevice::cpu, 0, {1}, [] { return ei::Status::success(); }});
    cycle.add({"b", ei::OperatorDevice::cpu, 0, {0}, [] { return ei::Status::success(); }});
    const auto bad = scheduler.run(cycle);
    CHECK(!bad.ok());
    CHECK(bad.status().code() == ei::StatusCode::invalid_argument);
}

void test_hardware_outputs() {
    ei::HardwareOutputAdapter joints({ei::HardwareOutputType::joint,
                                      ei::HardwareControlMode::position_torque,
                                      2, "base"});
    ei::ActionChunk joint_chunk;
    joint_chunk.steps = 1;
    joint_chunk.action_dim = 4;
    joint_chunk.values = {0.1F, 0.2F, 1.0F, 2.0F};
    auto joint = joints.encode(joint_chunk, 0, 10);
    CHECK(joint.ok());
    CHECK(joint.value().primary.size() == 2);
    CHECK(joint.value().torque.size() == 2);

    ei::HardwareOutputAdapter pose({ei::HardwareOutputType::end_effector_pose,
                                    ei::HardwareControlMode::position,
                                    7, "left_base"});
    ei::ActionChunk pose_chunk;
    pose_chunk.steps = 1;
    pose_chunk.action_dim = 7;
    pose_chunk.values = {0.0F, 0.1F, 0.2F, 0.0F, 0.0F, 0.0F, 1.0F};
    auto command = pose.encode(pose_chunk, 0, 11);
    CHECK(command.ok());
    CHECK(command.value().primary.size() == 7);

    pose_chunk.values[6] = 0.0F;
    CHECK(!pose.encode(pose_chunk, 0, 11).ok());
}

}  // namespace

int main() {
    test_validation_and_metrics();
    test_deadline_and_cancellation();
    test_action_processors();
    test_action_buffer();
    test_scheduler_overflow();
    test_operator_scheduler();
    test_hardware_outputs();

    if (failures != 0) {
        std::cerr << failures << " test checks failed\n";
        return 1;
    }
    std::cout << "all runtime tests passed\n";
}
