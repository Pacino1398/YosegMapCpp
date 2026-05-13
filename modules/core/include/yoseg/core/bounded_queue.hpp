#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

namespace yoseg::core {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity > 0 ? capacity : 1) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_not_full_.wait(lock, [&] { return queue_.size() < capacity_ || closed_; });
        if (closed_) {
            return;
        }
        queue_.push(std::move(item));
        cv_not_empty_.notify_one();
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_not_empty_.wait(lock, [&] { return !queue_.empty() || closed_; });
        if (queue_.empty()) {
            return false;
        }
        out = std::move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    std::size_t capacity_;
    bool closed_ = false;
    std::mutex mu_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::queue<T> queue_;
};

} // namespace yoseg::core
