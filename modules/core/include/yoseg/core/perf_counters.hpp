#pragma once

#include <atomic>
#include <cstdint>

namespace yoseg::core {

struct PerfCounters {
    std::atomic<std::uint64_t> copy_bytes{0};
    std::atomic<std::uint64_t> copy_pre_bytes{0};
    std::atomic<std::uint64_t> copy_post_bytes{0};
    std::atomic<std::uint64_t> copy_plan_bytes{0};
    std::atomic<std::uint64_t> copy_ros_bytes{0};
    std::atomic<std::uint64_t> alloc_count{0};

    void add_copy(std::uint64_t bytes) {
        copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }

    void add_copy_pre(std::uint64_t bytes) {
        copy_pre_bytes.fetch_add(bytes, std::memory_order_relaxed);
        add_copy(bytes);
    }

    void add_copy_post(std::uint64_t bytes) {
        copy_post_bytes.fetch_add(bytes, std::memory_order_relaxed);
        add_copy(bytes);
    }

    void add_copy_plan(std::uint64_t bytes) {
        copy_plan_bytes.fetch_add(bytes, std::memory_order_relaxed);
        add_copy(bytes);
    }

    void add_copy_ros(std::uint64_t bytes) {
        copy_ros_bytes.fetch_add(bytes, std::memory_order_relaxed);
        add_copy(bytes);
    }

    void add_alloc(std::uint64_t n = 1) {
        alloc_count.fetch_add(n, std::memory_order_relaxed);
    }
};

} // namespace yoseg::core
