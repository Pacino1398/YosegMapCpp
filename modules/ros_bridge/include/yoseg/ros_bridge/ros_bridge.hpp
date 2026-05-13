#pragma once

#include "yoseg/core/perf_counters.hpp"
#include "yoseg/planner/planner.hpp"

#include <string>

namespace yoseg::ros_bridge {

struct PublishConfig {
    bool enabled = false;
    std::string frame_id = "map";
    double rate_hz = 5.0;
    std::string occ_topic = "/octomap/occupancy";
    std::string cloud_topic = "/octomap/points";
    float cell_size = 1.0f;
};

void init_ros_bridge(const PublishConfig& cfg);
void set_perf_counters(yoseg::core::PerfCounters* perf);
bool publish(const yoseg::planner::PlannerOutput& planner_output);
void shutdown_ros_bridge();

} // namespace yoseg::ros_bridge
