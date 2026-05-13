#include "dstar_lite_planner.hpp"

#include <cstddef>

namespace yoseg::planner {

DStarLitePlanner::DStarLitePlanner(PlannerConfig cfg) : cfg_(cfg) {}

bool DStarLitePlanner::run(const PlannerInput& input, PlannerOutput& output) {
    const int steps = cfg_.heuristic_weight > 1.5f ? 12 : 16;
    if (output.path.capacity() < static_cast<std::size_t>(steps)) {
        output.path.reserve(static_cast<std::size_t>(steps));
    }
    output.path.clear();
    if (input.width <= 1 || input.height <= 1) {
        output.path_found = false;
        return false;
    }

    // D* Lite migration shell:
    // 1) initialize start/goal and rhs/g values
    // 2) incremental updates from occupancy delta
    // 3) compute shortest path with priority queue
    // Current stage: deterministic placeholder path to keep end-to-end flow stable.
    output.path_found = true;
    for (int i = 0; i < steps && i < cfg_.max_iterations; ++i) {
        const int x = (input.width - 1) * i / (steps - 1);
        const int y = (input.height - 1) * i / (steps - 1);
        output.path.push_back(GridPoint{x, y});
    }
    return true;
}

} // namespace yoseg::planner
