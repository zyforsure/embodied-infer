#include "embodied/infer/operator_scheduler.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace embodied::infer {
namespace {

std::size_t lane(OperatorDevice device) {
    return static_cast<std::size_t>(device);
}

std::vector<std::size_t> capacities(const OperatorSchedulerConfig& config) {
    return {config.cpu_workers, config.gpu_workers, config.npu_workers,
            config.io_workers};
}

bool expired(const OperatorScheduleOptions& options) {
    return options.deadline.has_value() &&
           std::chrono::steady_clock::now() >= *options.deadline;
}

}  // namespace

std::size_t OperatorGraph::add(OperatorSpec spec) {
    if (spec.name.empty()) {
        throw std::invalid_argument("operator name must not be empty");
    }
    if (!spec.function) {
        throw std::invalid_argument("operator function must not be empty");
    }
    nodes_.push_back(std::move(spec));
    return nodes_.size() - 1;
}

const OperatorSpec& OperatorGraph::at(std::size_t id) const {
    return nodes_.at(id);
}

Status OperatorGraph::validate() const {
    std::vector<std::size_t> indegree(nodes_.size());
    for (std::size_t id = 0; id < nodes_.size(); ++id) {
        std::unordered_set<std::size_t> seen;
        for (const auto dependency : nodes_[id].dependencies) {
            if (dependency >= nodes_.size()) {
                return {StatusCode::invalid_argument,
                        "operator dependency is out of range"};
            }
            if (!seen.insert(dependency).second) {
                return {StatusCode::invalid_argument,
                        "operator has duplicate dependency"};
            }
            ++indegree[id];
        }
    }
    std::queue<std::size_t> ready;
    for (std::size_t id = 0; id < indegree.size(); ++id) {
        if (indegree[id] == 0) ready.push(id);
    }
    std::size_t visited = 0;
    while (!ready.empty()) {
        const auto id = ready.front();
        ready.pop();
        ++visited;
        for (std::size_t child = 0; child < nodes_.size(); ++child) {
            if (std::find(nodes_[child].dependencies.begin(),
                          nodes_[child].dependencies.end(), id) !=
                nodes_[child].dependencies.end() &&
                --indegree[child] == 0) {
                ready.push(child);
            }
        }
    }
    if (visited != nodes_.size()) {
        return {StatusCode::invalid_argument, "operator graph contains a cycle"};
    }
    return Status::success();
}

OperatorScheduler::OperatorScheduler(OperatorSchedulerConfig config)
    : config_(config) {
    for (const auto capacity : capacities(config_)) {
        if (capacity == 0) {
            throw std::invalid_argument("operator worker capacity must be > 0");
        }
    }
}

Result<OperatorReport> OperatorScheduler::run(
    const OperatorGraph& graph, const OperatorScheduleOptions& options) const {
    const auto valid = graph.validate();
    if (!valid.ok()) return Result<OperatorReport>::failure(valid);
    if (options.cancellation.cancellation_requested() || expired(options)) {
        return Result<OperatorReport>::failure(
            {expired(options) ? StatusCode::deadline_exceeded
                              : StatusCode::cancelled,
             expired(options) ? "operator schedule deadline exceeded"
                               : "operator schedule was cancelled"});
    }
    if (graph.size() == 0) return Result<OperatorReport>::success({});

    struct Ready {
        int priority;
        std::size_t id;
        bool operator<(const Ready& other) const {
            if (priority != other.priority) return priority < other.priority;
            return id > other.id;
        }
    };
    std::vector<std::size_t> remaining(graph.size());
    std::vector<std::vector<std::size_t>> children(graph.size());
    for (std::size_t id = 0; id < graph.size(); ++id) {
        remaining[id] = graph.at(id).dependencies.size();
        for (const auto dep : graph.at(id).dependencies) children[dep].push_back(id);
    }

    std::priority_queue<Ready> ready;
    for (std::size_t id = 0; id < graph.size(); ++id) {
        if (remaining[id] == 0) ready.push({graph.at(id).priority, id});
    }

    std::mutex mutex;
    std::condition_variable condition;
    const auto limits = capacities(config_);
    std::array<std::size_t, 4> active{};
    std::size_t running = 0;
    std::size_t finished = 0;
    std::size_t peak = 0;
    bool stop = false;
    Status failure = Status::success();
    std::string failed_operator;
    std::vector<std::string> completed;

    const auto worker = [&] {
        while (true) {
            std::size_t id = 0;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [&] {
                    if (stop || finished == graph.size()) return true;
                    if (ready.empty()) return false;
                    for (std::size_t i = 0; i < ready.size(); ++i) {
                        // std::priority_queue has no iteration; waking workers
                        // optimistically is safe and avoids a second queue.
                        break;
                    }
                    return true;
                });
                if (stop || finished == graph.size()) return;

                // Select the highest-priority ready node whose device has room.
                std::vector<Ready> deferred;
                bool selected = false;
                while (!ready.empty()) {
                    const auto candidate = ready.top();
                    ready.pop();
                    const auto device = lane(graph.at(candidate.id).device);
                    if (active[device] < limits[device]) {
                        id = candidate.id;
                        ++active[device];
                        ++running;
                        peak = std::max(peak, running);
                        selected = true;
                        break;
                    }
                    deferred.push_back(candidate);
                }
                for (const auto item : deferred) ready.push(item);
                if (!selected) continue;
            }

            Status result;
            try {
                if (options.cancellation.cancellation_requested() || expired(options)) {
                    result = {expired(options) ? StatusCode::deadline_exceeded
                                               : StatusCode::cancelled,
                              expired(options) ? "operator schedule deadline exceeded"
                                                : "operator schedule was cancelled"};
                } else {
                    result = graph.at(id).function();
                }
            } catch (const std::exception& error) {
                result = {StatusCode::internal_error, error.what()};
            } catch (...) {
                result = {StatusCode::internal_error, "operator threw an unknown exception"};
            }

            {
                std::lock_guard lock(mutex);
                const auto device = lane(graph.at(id).device);
                --active[device];
                --running;
                if (!result.ok() && !stop) {
                    stop = true;
                    failure = result;
                    failed_operator = graph.at(id).name;
                } else if (!stop) {
                    ++finished;
                    completed.push_back(graph.at(id).name);
                    for (const auto child : children[id]) {
                        if (--remaining[child] == 0) {
                            ready.push({graph.at(child).priority, child});
                        }
                    }
                }
            }
            condition.notify_all();
        }
    };

    const auto worker_count = config_.cpu_workers + config_.gpu_workers +
                              config_.npu_workers + config_.io_workers;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) workers.emplace_back(worker);
    condition.notify_all();
    for (auto& thread : workers) thread.join();

    if (!failure.ok()) {
        return Result<OperatorReport>::failure(
            {failure.code(), failed_operator + ": " + failure.message()});
    }
    return Result<OperatorReport>::success(
        {std::move(completed), {}, peak});
}

}  // namespace embodied::infer
