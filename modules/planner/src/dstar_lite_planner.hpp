#pragma once

#include "yoseg/planner/planner.hpp"

namespace yoseg::planner {

class DStarLitePlanner final : public IPlanner {
public:
    explicit DStarLitePlanner(PlannerConfig cfg);
    bool run(const PlannerInput& input, PlannerOutput& output) override;

private:
    PlannerConfig cfg_;
};

} // namespace yoseg::planner
