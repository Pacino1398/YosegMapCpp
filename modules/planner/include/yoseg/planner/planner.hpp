#pragma once

#include "yoseg/core/perf_counters.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace yoseg::planner {

struct PlannerInput {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> occupancy;
};

struct GridPoint {
    int x = 0;
    int y = 0;
};

struct PlannerOutput {
    bool path_found = false;
    std::vector<GridPoint> path;
};

struct PlannerConfig {
    int max_iterations = 20000;
    float heuristic_weight = 1.0f;
};

class IPlanner {
public:
    virtual ~IPlanner() = default;
    virtual bool run(const PlannerInput& input, PlannerOutput& output) = 0;
};

void init_planner();
void set_perf_counters(yoseg::core::PerfCounters* perf);
void set_planner_config(const PlannerConfig& cfg);
void set_planner(std::unique_ptr<IPlanner> planner);
bool run_planner(const PlannerInput& input, PlannerOutput& output);

} // namespace yoseg::planner
