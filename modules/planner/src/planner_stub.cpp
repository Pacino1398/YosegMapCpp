#include "yoseg/planner/planner.hpp"
#include "dstar_lite_planner.hpp"

#include <iostream>
#include <memory>
#include <utility>

namespace yoseg::planner {

namespace {
PlannerConfig g_cfg{};
core::PerfCounters* g_perf = nullptr;
std::unique_ptr<IPlanner> g_planner = std::make_unique<DStarLitePlanner>(g_cfg);
} // namespace

void init_planner() {
    std::cout << "planner stub initialized\n";
}

void set_perf_counters(core::PerfCounters* perf) {
    g_perf = perf;
}

void set_planner_config(const PlannerConfig& cfg) {
    g_cfg = cfg;
    g_planner = std::make_unique<DStarLitePlanner>(g_cfg);
    if (g_perf != nullptr) {
        g_perf->add_alloc();
    }
}

void set_planner(std::unique_ptr<IPlanner> planner) {
    if (planner) {
        g_planner = std::move(planner);
    }
}

bool run_planner(const PlannerInput& input, PlannerOutput& output) {
    if (g_perf != nullptr) {
        g_perf->add_copy_plan(static_cast<std::uint64_t>(input.occupancy.size()));
    }
    return g_planner->run(input, output);
}

} // namespace yoseg::planner
