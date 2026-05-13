#pragma once

#include <chrono>

namespace yoseg::core {

class ScopeTimer {
public:
    explicit ScopeTimer(const char* name);
    ~ScopeTimer();

private:
    const char* name_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace yoseg::core
