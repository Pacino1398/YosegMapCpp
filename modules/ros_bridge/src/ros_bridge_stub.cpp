#include "yoseg/ros_bridge/ros_bridge.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#if defined(YOSEG_HAS_ROS2) && YOSEG_HAS_ROS2
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/header.hpp>
#endif

namespace yoseg::ros_bridge {

namespace {
PublishConfig g_cfg{};
bool g_initialized = false;
core::PerfCounters* g_perf = nullptr;
std::chrono::steady_clock::time_point g_last_pub{};

#if defined(YOSEG_HAS_ROS2) && YOSEG_HAS_ROS2
std::shared_ptr<rclcpp::Node> g_node;
rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr g_occ_pub;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr g_cloud_pub;
#endif

inline bool should_publish_now() {
    const auto now = std::chrono::steady_clock::now();
    const double period_ms = 1000.0 / std::max(0.1, g_cfg.rate_hz);
    if (g_last_pub.time_since_epoch().count() == 0) {
        g_last_pub = now;
        return true;
    }
    const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_last_pub).count();
    if (static_cast<double>(dt_ms) >= period_ms) {
        g_last_pub = now;
        return true;
    }
    return false;
}

} // namespace

void init_ros_bridge(const PublishConfig& cfg) {
    g_cfg = cfg;
    g_initialized = true;
    g_last_pub = {};

#if defined(YOSEG_HAS_ROS2) && YOSEG_HAS_ROS2
    if (g_cfg.enabled) {
        if (!rclcpp::ok()) {
            rclcpp::init(0, nullptr);
        }
        g_node = std::make_shared<rclcpp::Node>("yosegmapcpp_bridge");
        g_occ_pub = g_node->create_publisher<nav_msgs::msg::OccupancyGrid>(g_cfg.occ_topic, 10);
        g_cloud_pub = g_node->create_publisher<sensor_msgs::msg::PointCloud2>(g_cfg.cloud_topic, 10);
        std::cout << "ros bridge initialized (ros2), occ=" << g_cfg.occ_topic << ", cloud=" << g_cfg.cloud_topic
                  << ", frame_id=" << g_cfg.frame_id << ", rate_hz=" << g_cfg.rate_hz
                  << ", grid_units=dimensionless, occ_resolution=1.0(nominal)\n";
        return;
    }
#endif
    std::cout << "ros bridge initialized (stub), enabled=" << (g_cfg.enabled ? "true" : "false")
              << ", rate_hz=" << g_cfg.rate_hz << ", frame_id=" << g_cfg.frame_id << "\n";
}

void set_perf_counters(core::PerfCounters* perf) {
    g_perf = perf;
}

bool publish(const yoseg::planner::PlannerOutput& planner_output) {
    if (!g_initialized || !g_cfg.enabled) {
        return true;
    }
    if (!should_publish_now()) {
        return true;
    }

#if defined(YOSEG_HAS_ROS2) && YOSEG_HAS_ROS2
    if (g_node && g_occ_pub && g_cloud_pub) {
        rclcpp::spin_some(g_node);
        auto stamp = g_node->now();

        int max_x = 0;
        int max_y = 0;
        for (const auto& p : planner_output.path) {
            if (p.x > max_x) max_x = p.x;
            if (p.y > max_y) max_y = p.y;
        }
        const int w = std::max(1, max_x + 1);
        const int h = std::max(1, max_y + 1);

        nav_msgs::msg::OccupancyGrid occ{};
        occ.header.stamp = stamp;
        occ.header.frame_id = g_cfg.frame_id;
        occ.info.resolution = 1.0f;  // nominal placeholder: dimensionless grid units
        occ.info.width = static_cast<std::uint32_t>(w);
        occ.info.height = static_cast<std::uint32_t>(h);
        occ.info.origin.orientation.w = 1.0;
        occ.data.assign(static_cast<std::size_t>(w * h), 0);
        for (const auto& p : planner_output.path) {
            if (p.x >= 0 && p.x < w && p.y >= 0 && p.y < h) {
                occ.data[static_cast<std::size_t>(p.y * w + p.x)] = 100;
            }
        }
        g_occ_pub->publish(occ);

        sensor_msgs::msg::PointCloud2 cloud{};
        cloud.header.stamp = stamp;
        cloud.header.frame_id = g_cfg.frame_id;
        cloud.height = 1;
        cloud.width = static_cast<std::uint32_t>(planner_output.path.size());
        cloud.is_bigendian = false;
        cloud.is_dense = true;
        cloud.point_step = 12;  // x,y,z float32
        cloud.row_step = cloud.point_step * cloud.width;
        cloud.fields.resize(3);
        cloud.fields[0].name = "x";
        cloud.fields[0].offset = 0;
        cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud.fields[0].count = 1;
        cloud.fields[1].name = "y";
        cloud.fields[1].offset = 4;
        cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud.fields[1].count = 1;
        cloud.fields[2].name = "z";
        cloud.fields[2].offset = 8;
        cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud.fields[2].count = 1;
        cloud.data.resize(static_cast<std::size_t>(cloud.row_step), 0);
        for (std::size_t i = 0; i < planner_output.path.size(); ++i) {
            const float x = static_cast<float>(planner_output.path[i].x);  // grid units
            const float y = static_cast<float>(planner_output.path[i].y);  // grid units
            const float z = 0.0f;
            std::memcpy(cloud.data.data() + i * 12 + 0, &x, sizeof(float));
            std::memcpy(cloud.data.data() + i * 12 + 4, &y, sizeof(float));
            std::memcpy(cloud.data.data() + i * 12 + 8, &z, sizeof(float));
        }
        g_cloud_pub->publish(cloud);

        if (g_perf != nullptr) {
            g_perf->add_copy_ros(static_cast<std::uint64_t>(planner_output.path.size() * 12));
        }
        return true;
    }
#endif

    // Stub fallback path.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (g_perf != nullptr) {
        g_perf->add_copy_ros(static_cast<std::uint64_t>(planner_output.path.size() * sizeof(yoseg::planner::GridPoint)));
    }
    std::cout << "[ros_stub] publish path points=" << planner_output.path.size() << "\n";
    return true;
}

void shutdown_ros_bridge() {
#if defined(YOSEG_HAS_ROS2) && YOSEG_HAS_ROS2
    g_occ_pub.reset();
    g_cloud_pub.reset();
    g_node.reset();
    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }
#endif
    g_initialized = false;
}

} // namespace yoseg::ros_bridge

