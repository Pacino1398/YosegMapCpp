#include "yoseg/capture/capture.hpp"
#include "yoseg/core/bounded_queue.hpp"
#include "yoseg/core/perf_counters.hpp"
#include "yoseg/infer/infer.hpp"
#include "yoseg/planner/planner.hpp"
#include "yoseg/ros_bridge/ros_bridge.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <deque>
#include <cstddef>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct RuntimeConfig {
    std::string source = "0";
    std::string backend = "rknn";
    std::string model = "./weights/model.rknn";
    int frames = 120;
    int queue_capacity = 4;
    int warmup_frames = 8;
    bool ros_enabled = false;
    double ros_rate_hz = 5.0;
    std::string ros_occ_topic = "/octomap/occupancy";
    std::string ros_cloud_topic = "/octomap/points";
    float ros_cell_size = 1.0f;
    bool ros_cell_size_overridden = false;
    std::string profile_out;
    float obstacle_thres = 0.5f;
    int planner_max_iters = 20000;
    int net_w = 640;
    int net_h = 640;
};

struct FramePacket {
    int frame_id = 0;
    yoseg::capture::Frame frame;
    Clock::time_point t0{};
    Clock::time_point t1{};
};

RuntimeConfig parse_args(int argc, char** argv) {
    RuntimeConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
            cfg.source = argv[++i];
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            cfg.backend = argv[++i];
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cfg.model = argv[++i];
        } else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            cfg.frames = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--queue-capacity") == 0 && i + 1 < argc) {
            cfg.queue_capacity = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--warmup-frames") == 0 && i + 1 < argc) {
            cfg.warmup_frames = std::max(0, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--ros-enable") == 0) {
            cfg.ros_enabled = true;
        } else if (std::strcmp(argv[i], "--ros-rate") == 0 && i + 1 < argc) {
            cfg.ros_rate_hz = std::max(0.1, std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--ros-occ-topic") == 0 && i + 1 < argc) {
            cfg.ros_occ_topic = argv[++i];
        } else if (std::strcmp(argv[i], "--ros-cloud-topic") == 0 && i + 1 < argc) {
            cfg.ros_cloud_topic = argv[++i];
        } else if (std::strcmp(argv[i], "--ros-cell-size") == 0 && i + 1 < argc) {
            cfg.ros_cell_size = std::max(0.001f, static_cast<float>(std::atof(argv[++i])));
            cfg.ros_cell_size_overridden = true;
        } else if (std::strcmp(argv[i], "--profile-out") == 0 && i + 1 < argc) {
            cfg.profile_out = argv[++i];
        } else if (std::strcmp(argv[i], "--obstacle-thres") == 0 && i + 1 < argc) {
            cfg.obstacle_thres = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--planner-max-iters") == 0 && i + 1 < argc) {
            cfg.planner_max_iters = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--net-w") == 0 && i + 1 < argc) {
            cfg.net_w = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--net-h") == 0 && i + 1 < argc) {
            cfg.net_h = std::max(1, std::atoi(argv[++i]));
        }
    }
    return cfg;
}
} // namespace

int main(int argc, char** argv) {
    const RuntimeConfig cfg = parse_args(argc, argv);
    yoseg::core::PerfCounters perf{};

    yoseg::planner::init_planner();
    yoseg::planner::PlannerConfig planner_cfg;
    planner_cfg.max_iterations = cfg.planner_max_iters;
    planner_cfg.heuristic_weight = 1.0f;
    yoseg::planner::set_perf_counters(&perf);
    yoseg::planner::set_planner_config(planner_cfg);

    yoseg::infer::PostprocessConfig post_cfg;
    post_cfg.obstacle_threshold = cfg.obstacle_thres;
    post_cfg.grid_width = cfg.net_w;
    post_cfg.grid_height = cfg.net_h;
    yoseg::infer::set_postprocess_config(post_cfg);
    yoseg::infer::set_perf_counters(&perf);

    yoseg::infer::PreprocessConfig pre_cfg;
    pre_cfg.target_width = cfg.net_w;
    pre_cfg.target_height = cfg.net_h;
    pre_cfg.bgr_to_rgb = false;
    yoseg::infer::set_preprocess_config(pre_cfg);

    yoseg::ros_bridge::PublishConfig ros_cfg;
    ros_cfg.enabled = cfg.ros_enabled;
    ros_cfg.rate_hz = cfg.ros_rate_hz;
    ros_cfg.frame_id = "map";
    ros_cfg.occ_topic = cfg.ros_occ_topic;
    ros_cfg.cloud_topic = cfg.ros_cloud_topic;
    ros_cfg.cell_size = 1.0f;  // fixed nominal placeholder for dimensionless grid mapping
    yoseg::ros_bridge::set_perf_counters(&perf);
    yoseg::ros_bridge::init_ros_bridge(ros_cfg);

    yoseg::capture::CaptureSource capture;
    if (!capture.open(cfg.source)) {
        std::cerr << "failed to open capture source\n";
        return 1;
    }
    (void)capture.warmup(cfg.warmup_frames, cfg.net_w, cfg.net_h, 3);
    auto engine = yoseg::infer::create_engine(cfg.backend);
    if (!engine || !engine->init(cfg.model)) {
        std::cerr << "failed to init infer engine\n";
        return 1;
    }

    std::cout << "realtime runner started\n";
    std::cout << "source=" << cfg.source << ", backend=" << engine->name() << ", model=" << cfg.model << "\n";
    std::cout << "grid_units=dimensionless, occ_resolution=1.0(nominal)\n";
    if (cfg.ros_cell_size_overridden) {
        std::cout << "warning: --ros-cell-size is ignored in current dimensionless-grid mode\n";
    }
    const std::string header =
        "frame,pre_ms,infer_ms,post_ms,plan_ms,ros_ms,total_ms,moving_fps,copy_pre,copy_post,copy_plan,copy_ros,copy_total,alloc_count";
    std::cout << header << "\n";
    std::ofstream profile_file;
    if (!cfg.profile_out.empty()) {
        profile_file.open(cfg.profile_out, std::ios::out | std::ios::trunc);
        if (profile_file.is_open()) {
            profile_file << header << "\n";
        }
    }

    const std::size_t pool_size = static_cast<std::size_t>(cfg.queue_capacity);
    std::vector<FramePacket> packet_pool(pool_size);
    yoseg::core::BoundedQueue<int> free_ids(pool_size);
    yoseg::core::BoundedQueue<int> ready_ids(pool_size);
    std::mutex io_mu;

    const std::size_t expected_frame_bytes =
        static_cast<std::size_t>(cfg.net_w) * static_cast<std::size_t>(cfg.net_h) * static_cast<std::size_t>(3);
    for (std::size_t i = 0; i < pool_size; ++i) {
        packet_pool[i].frame.data.resize(expected_frame_bytes);
        free_ids.push(static_cast<int>(i));
    }
    perf.add_alloc(static_cast<std::uint64_t>(pool_size));

    std::thread producer([&] {
        for (int frame_id = 0; frame_id < cfg.frames; ++frame_id) {
            int slot_id = -1;
            if (!free_ids.pop(slot_id)) {
                break;
            }
            FramePacket& p = packet_pool[static_cast<std::size_t>(slot_id)];
            p.frame_id = frame_id;
            p.t0 = Clock::now();
            if (!capture.read(p.frame)) {
                free_ids.push(slot_id);
                break;
            }
            p.t1 = Clock::now();
            ready_ids.push(slot_id);
        }
        ready_ids.close();
    });

    std::thread consumer([&] {
        yoseg::infer::PostprocessOutput post_output;
        yoseg::planner::PlannerInput planner_input;
        yoseg::planner::PlannerOutput planner_output;
        yoseg::infer::InferOutput output;
        const std::size_t occupancy_size =
            static_cast<std::size_t>(std::max(1, cfg.net_w)) * static_cast<std::size_t>(std::max(1, cfg.net_h));
        post_output.occupancy.resize(occupancy_size);
        planner_input.occupancy.resize(occupancy_size);
        planner_output.path.reserve(32);
        output.tensors.reserve(4);
        std::deque<double> fps_window;
        int summary_count = 0;
        double summary_total_ms = 0.0;
        double summary_max_ms = 0.0;
        std::uint64_t last_copy_pre = perf.copy_pre_bytes.load(std::memory_order_relaxed);
        std::uint64_t last_copy_post = perf.copy_post_bytes.load(std::memory_order_relaxed);
        std::uint64_t last_copy_plan = perf.copy_plan_bytes.load(std::memory_order_relaxed);
        std::uint64_t last_copy_ros = perf.copy_ros_bytes.load(std::memory_order_relaxed);
        int slot_id = -1;
        while (ready_ids.pop(slot_id)) {
            FramePacket& p = packet_pool[static_cast<std::size_t>(slot_id)];
            if (p.frame.width != cfg.net_w || p.frame.height != cfg.net_h || p.frame.channels != 3) {
                std::lock_guard<std::mutex> lock(io_mu);
                std::cerr << "frame format drift detected: " << p.frame.width << "x" << p.frame.height
                          << "x" << p.frame.channels << "\n";
                p.frame.data.clear();
                free_ids.push(slot_id);
                continue;
            }
            yoseg::infer::InferInput raw_input{p.frame.width, p.frame.height, p.frame.channels, std::move(p.frame.data)};
            const auto t2_start = Clock::now();
            yoseg::infer::InferInput input;
            if (!yoseg::infer::preprocess_move(std::move(raw_input), input)) {
                p.frame.data = std::move(input.data);
                free_ids.push(slot_id);
                continue;
            }
            const auto t2 = Clock::now();

            output.tensors.clear();
            if (!engine->run(input, output)) {
                p.frame.data = std::move(input.data);
                free_ids.push(slot_id);
                continue;
            }
            const auto t3 = Clock::now();

            if (!yoseg::infer::postprocess(input, output, post_output)) {
                p.frame.data = std::move(input.data);
                free_ids.push(slot_id);
                continue;
            }
            const auto t4 = Clock::now();

            planner_input.width = post_output.map_width;
            planner_input.height = post_output.map_height;
            planner_input.occupancy.swap(post_output.occupancy);
            (void)yoseg::planner::run_planner(planner_input, planner_output);
            const auto t5 = Clock::now();

            (void)yoseg::ros_bridge::publish(planner_output);
            const auto t6 = Clock::now();

            const double pre_ms = Ms(p.t1 - p.t0).count() + Ms(t2 - t2_start).count();
            const double infer_ms = Ms(t3 - t2).count();
            const double post_ms = Ms(t4 - t3).count();
            const double plan_ms = Ms(t5 - t4).count();
            const double ros_ms = Ms(t6 - t5).count();
            const double total_ms = Ms(t6 - p.t0).count();
            fps_window.push_back(total_ms);
            if (fps_window.size() > 30) {
                fps_window.pop_front();
            }
            double moving_fps = 0.0;
            double sum_ms = 0.0;
            for (double v : fps_window) {
                sum_ms += v;
            }
            if (sum_ms > 1e-6) {
                moving_fps = (1000.0 * static_cast<double>(fps_window.size())) / sum_ms;
            }

            const auto copy_bytes = perf.copy_bytes.load(std::memory_order_relaxed);
            const auto copy_pre = perf.copy_pre_bytes.load(std::memory_order_relaxed);
            const auto copy_post = perf.copy_post_bytes.load(std::memory_order_relaxed);
            const auto copy_plan = perf.copy_plan_bytes.load(std::memory_order_relaxed);
            const auto copy_ros = perf.copy_ros_bytes.load(std::memory_order_relaxed);
            const auto alloc_count = perf.alloc_count.load(std::memory_order_relaxed);
            const std::string row = std::to_string(p.frame_id) + "," + std::to_string(pre_ms) + "," +
                                    std::to_string(infer_ms) + "," + std::to_string(post_ms) + "," +
                                    std::to_string(plan_ms) + "," + std::to_string(ros_ms) + "," +
                                    std::to_string(total_ms) + "," + std::to_string(moving_fps) + "," +
                                    std::to_string(copy_pre) + "," + std::to_string(copy_post) + "," +
                                    std::to_string(copy_plan) + "," + std::to_string(copy_ros) + "," +
                                    std::to_string(copy_bytes) + "," +
                                    std::to_string(alloc_count);
            std::lock_guard<std::mutex> lock(io_mu);
            std::cout << row << "\n";
            if (profile_file.is_open()) {
                profile_file << row << "\n";
            }
            ++summary_count;
            summary_total_ms += total_ms;
            if (total_ms > summary_max_ms) {
                summary_max_ms = total_ms;
            }
            if (summary_count >= 100) {
                const double avg_ms = summary_total_ms / static_cast<double>(summary_count);
                const double avg_fps = avg_ms > 1e-6 ? 1000.0 / avg_ms : 0.0;
                const std::uint64_t cur_pre = perf.copy_pre_bytes.load(std::memory_order_relaxed);
                const std::uint64_t cur_post = perf.copy_post_bytes.load(std::memory_order_relaxed);
                const std::uint64_t cur_plan = perf.copy_plan_bytes.load(std::memory_order_relaxed);
                const std::uint64_t cur_ros = perf.copy_ros_bytes.load(std::memory_order_relaxed);
                const std::uint64_t d_pre = cur_pre - last_copy_pre;
                const std::uint64_t d_post = cur_post - last_copy_post;
                const std::uint64_t d_plan = cur_plan - last_copy_plan;
                const std::uint64_t d_ros = cur_ros - last_copy_ros;
                last_copy_pre = cur_pre;
                last_copy_post = cur_post;
                last_copy_plan = cur_plan;
                last_copy_ros = cur_ros;
                const std::string summary_row =
                    "#summary,last_n=100,avg_ms=" + std::to_string(avg_ms) + ",max_ms=" + std::to_string(summary_max_ms) +
                    ",avg_fps=" + std::to_string(avg_fps) +
                    ",delta_copy_pre=" + std::to_string(d_pre) +
                    ",delta_copy_post=" + std::to_string(d_post) +
                    ",delta_copy_plan=" + std::to_string(d_plan) +
                    ",delta_copy_ros=" + std::to_string(d_ros);
                std::cout << summary_row << "\n";
                if (profile_file.is_open()) {
                    profile_file << summary_row << "\n";
                }
                summary_count = 0;
                summary_total_ms = 0.0;
                summary_max_ms = 0.0;
            }
            p.frame.data = std::move(input.data);
            free_ids.push(slot_id);
        }
    });

    producer.join();
    consumer.join();

    capture.close();
    yoseg::ros_bridge::shutdown_ros_bridge();
    if (profile_file.is_open()) {
        profile_file.close();
    }
    return 0;
}

