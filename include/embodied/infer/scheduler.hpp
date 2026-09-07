#pragma once

#include "embodied/infer/engine.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

namespace embodied::infer {

enum class OverflowPolicy {
    keep_latest,
    reject_newest,
};

struct SchedulerConfig {
    std::size_t queue_capacity = 1;
    OverflowPolicy overflow_policy = OverflowPolicy::keep_latest;
};

class Scheduler {
public:
    Scheduler(std::shared_ptr<Engine> engine, SchedulerConfig config = {});
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    std::future<Result<ActionChunk>> submit(
        Observation observation,
        InferenceOptions options = {});
    void shutdown();

private:
    struct Task {
        Observation observation;
        InferenceOptions options;
        std::promise<Result<ActionChunk>> promise;
    };

    void worker_loop();
    static void cancel_task(Task& task, const char* message);

    std::shared_ptr<Engine> engine_;
    SchedulerConfig config_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Task> queue_;
    bool closed_ = false;
    std::thread worker_;
};

}  // namespace embodied::infer
