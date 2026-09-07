#pragma once

#include <optional>
#include <string>
#include <utility>

namespace embodied::infer {

enum class StatusCode {
    ok = 0,
    invalid_argument,
    not_ready,
    deadline_exceeded,
    queue_full,
    cancelled,
    backend_error,
    internal_error,
};

class Status {
public:
    Status() = default;
    Status(StatusCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    [[nodiscard]] bool ok() const noexcept { return code_ == StatusCode::ok; }
    [[nodiscard]] StatusCode code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

    static Status success() { return {}; }

private:
    StatusCode code_ = StatusCode::ok;
    std::string message_;
};

template <typename T>
class Result {
public:
    static Result success(T value) {
        return Result(Status::success(), std::move(value));
    }

    static Result failure(Status status) {
        return Result(std::move(status), std::nullopt);
    }

    [[nodiscard]] bool ok() const noexcept { return status_.ok(); }
    [[nodiscard]] const Status& status() const noexcept { return status_; }
    [[nodiscard]] const T& value() const& { return value_.value(); }
    [[nodiscard]] T& value() & { return value_.value(); }
    [[nodiscard]] T&& value() && { return std::move(value_.value()); }

private:
    Result(Status status, std::optional<T> value)
        : status_(std::move(status)), value_(std::move(value)) {}

    Status status_;
    std::optional<T> value_;
};

}  // namespace embodied::infer
