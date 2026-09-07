#include "embodied/infer/engine.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace embodied::infer {
namespace {

bool expired(const InferenceOptions& options) {
    return options.deadline.has_value() &&
           std::chrono::steady_clock::now() >= *options.deadline;
}

Status deadline_status() {
    return {StatusCode::deadline_exceeded, "inference deadline exceeded"};
}

Status cancellation_status() {
    return {StatusCode::cancelled, "inference was cancelled"};
}

}  // namespace

Engine::Engine(std::shared_ptr<Backend> backend) : backend_(std::move(backend)) {
    if (!backend_) {
        throw std::invalid_argument("backend must not be null");
    }
    observation_processors_.push_back(
        std::make_unique<ObservationValidator>());
}

Engine& Engine::add_observation_processor(
    std::unique_ptr<ObservationProcessor> processor) {
    if (!processor) {
        throw std::invalid_argument("observation processor must not be null");
    }
    observation_processors_.insert(observation_processors_.end() - 1,
                                   std::move(processor));
    return *this;
}

Engine& Engine::add_action_processor(
    std::unique_ptr<ActionProcessor> processor) {
    if (!processor) {
        throw std::invalid_argument("action processor must not be null");
    }
    action_processors_.push_back(std::move(processor));
    return *this;
}

Result<ActionChunk> Engine::infer(Observation observation,
                                  const InferenceOptions& options) {
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    requests_.fetch_add(1, std::memory_order_relaxed);

    auto finish_error = [&](Status status) {
        const auto elapsed = std::chrono::duration<double, std::milli>(
            Clock::now() - started).count();
        record_result(status, elapsed);
        return Result<ActionChunk>::failure(std::move(status));
    };

    if (options.cancellation.cancellation_requested()) {
        return finish_error(cancellation_status());
    }
    if (expired(options)) {
        return finish_error(deadline_status());
    }

    // A backend and its processor state are serialized by default. Most
    // accelerator execution contexts are not re-entrant.
    std::lock_guard lock(backend_mutex_);

    if (options.cancellation.cancellation_requested()) {
        return finish_error(cancellation_status());
    }
    if (expired(options)) {
        return finish_error(deadline_status());
    }

    for (auto& processor : observation_processors_) {
        Status status = processor->process(observation, backend_->spec());
        if (!status.ok()) {
            return finish_error(std::move(status));
        }
    }

    if (options.cancellation.cancellation_requested()) {
        return finish_error(cancellation_status());
    }
    if (expired(options)) {
        return finish_error(deadline_status());
    }

    auto result = backend_->infer(observation, options);
    if (!result.ok()) {
        return finish_error(result.status());
    }

    auto actions = std::move(result).value();
    actions.request_id = observation.request_id;
    actions.first_control_step = observation.control_step;
    actions.source_timestamp_ns = observation.timestamp_ns;
    actions.control_period_ns = backend_->spec().control_period_ns;
    actions.representation = backend_->spec().action_representation;

    for (auto& processor : action_processors_) {
        Status status = processor->process(observation, actions, backend_->spec());
        if (!status.ok()) {
            return finish_error(std::move(status));
        }
    }

    if (options.cancellation.cancellation_requested()) {
        return finish_error(cancellation_status());
    }
    if (expired(options)) {
        return finish_error(deadline_status());
    }

    const auto elapsed = std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
    record_result(Status::success(), elapsed);
    return Result<ActionChunk>::success(std::move(actions));
}

Status Engine::reset() {
    std::lock_guard lock(backend_mutex_);
    for (auto& processor : observation_processors_) {
        Status status = processor->reset();
        if (!status.ok()) {
            return status;
        }
    }
    for (auto& processor : action_processors_) {
        Status status = processor->reset();
        if (!status.ok()) {
            return status;
        }
    }
    return backend_->reset();
}

const ModelSpec& Engine::spec() const noexcept {
    return backend_->spec();
}

MetricsSnapshot Engine::metrics() const noexcept {
    MetricsSnapshot snapshot;
    snapshot.requests = requests_.load(std::memory_order_relaxed);
    snapshot.succeeded = succeeded_.load(std::memory_order_relaxed);
    snapshot.failed = failed_.load(std::memory_order_relaxed);
    snapshot.deadline_exceeded =
        deadline_exceeded_.load(std::memory_order_relaxed);
    snapshot.cancelled = cancelled_.load(std::memory_order_relaxed);
    snapshot.dropped = dropped_.load(std::memory_order_relaxed);
    const auto latency = latency_us_.load(std::memory_order_relaxed);
    snapshot.average_latency_ms = snapshot.requests == 0
        ? 0.0
        : static_cast<double>(latency) /
              static_cast<double>(snapshot.requests) / 1000.0;
    snapshot.maximum_latency_ms = static_cast<double>(
        maximum_latency_us_.load(std::memory_order_relaxed)) / 1000.0;
    return snapshot;
}

void Engine::record_dropped() noexcept {
    dropped_.fetch_add(1, std::memory_order_relaxed);
}

void Engine::record_result(const Status& status, double latency_ms) noexcept {
    if (status.ok()) {
        succeeded_.fetch_add(1, std::memory_order_relaxed);
    } else {
        failed_.fetch_add(1, std::memory_order_relaxed);
        if (status.code() == StatusCode::deadline_exceeded) {
            deadline_exceeded_.fetch_add(1, std::memory_order_relaxed);
        } else if (status.code() == StatusCode::cancelled) {
            cancelled_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    const auto elapsed_us = static_cast<std::uint64_t>(latency_ms * 1000.0);
    latency_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
    auto previous = maximum_latency_us_.load(std::memory_order_relaxed);
    while (previous < elapsed_us &&
           !maximum_latency_us_.compare_exchange_weak(
               previous, elapsed_us, std::memory_order_relaxed)) {
    }
}

}  // namespace embodied::infer
