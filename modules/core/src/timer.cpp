#include "yoseg/core/timer.hpp"

#include <iostream>

namespace yoseg::core {

ScopeTimer::ScopeTimer(const char* name)
    : name_(name), start_(std::chrono::steady_clock::now()) {}

ScopeTimer::~ScopeTimer() {
    const auto end = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
    std::cout << "[TIMER] " << name_ << ": " << ms << " ms\n";
}

} // namespace yoseg::core
