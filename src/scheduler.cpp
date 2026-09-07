#include "embodied/infer/scheduler.hpp"

#include <stdexcept>
#include <utility>

namespace embodied::infer {

Scheduler::Scheduler(std::shared_ptr<Engine> engine, SchedulerConfig config)
    : engine_(std::move(engine)), config_(config) {
    if (!engine_) {
        throw std::invalid_argument("engine must not be null");
    }
    if (config_.queue_capacity == 0) {
        throw std::invalid_argument("queue_capacity must be greater than zero");
    }
    worker_ = std::thread(&Scheduler::worker_loop, this);
}

Scheduler::~Scheduler() {
    shutdown();
}

std::future<Result<ActionChunk>> Scheduler::submit(
    Observation observation,
    InferenceOptions options) {
    Task task{std::move(observation), std::move(options), {}};
    auto future = task.promise.get_future();

    {
        std::lock_guard lock(mutex_);
        if (closed_) {
            cancel_task(task, "scheduler is shut down");
            return future;
        }

        if (queue_.size() >= config_.queue_capacity) {
            if (config_.overflow_policy == OverflowPolicy::reject_newest) {
                task.promise.set_value(Result<ActionChunk>::failure(
                    {StatusCode::queue_full, "inference queue is full"}));
                engine_->record_dropped();
                return future;
            }
            while (queue_.size() >= config_.queue_capacity) {
                cancel_task(queue_.front(),
                            "request was superseded by a newer observation");
                queue_.pop_front();
                engine_->record_dropped();
            }
        }

        queue_.push_back(std::move(task));
    }
    condition_.notify_one();
    return future;
}

void Scheduler::shutdown() {
    {
        std::lock_guard lock(mutex_);
        if (closed_) {
            return;
        }
        closed_ = true;
        for (auto& task : queue_) {
            cancel_task(task, "scheduler shut down before inference started");
        }
        queue_.clear();
    }
    condition_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void Scheduler::worker_loop() {
    while (true) {
        Task task;
        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, [this] { return closed_ || !queue_.empty(); });
            if (closed_ && queue_.empty()) {
                return;
            }
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        task.promise.set_value(
            engine_->infer(std::move(task.observation), task.options));
    }
}

void Scheduler::cancel_task(Task& task, const char* message) {
    task.promise.set_value(Result<ActionChunk>::failure(
        {StatusCode::cancelled, message}));
}

}  // namespace embodied::infer
