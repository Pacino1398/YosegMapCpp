# YosegMapCpp

独立 C++ 重构仓库，不改动原始 `YosegMap` 目录。

## 目标

- 图像获取与推理链路迁移到 C++（RK3588 优先 NPU）
- 路径规划迁移到 C++（实时增量更新）
- ROS 发布迁移到 C++（低延迟、低抖动）

## 目录

- `apps/realtime_runner`：主程序入口
- `modules/capture`：采集与预处理
- `modules/infer`：推理抽象（后续接 RKNN）
- `modules/planner`：路径规划核心
- `modules/ros_bridge`：ROS2 发布
- `modules/core`：公共类型、计时与配置
- `benchmarks`：性能基准
- `configs`：运行配置
- `docs`：迁移设计文档

## 快速开始

```bash
cmake -S . -B build_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_ninja -j
./build_ninja/apps/realtime_runner/realtime_runner --source 0 --backend cpu --model noop --frames 120 --profile-out ./profile.csv
```

说明：
- 若构建时检测到 OpenCV，`--source` 会优先使用真实摄像头/视频流读取。
- 若 OpenCV 不可用或打开失败，自动回退到合成帧，保证流水线与 profiling 可运行。
- `--profile-out` 可将分段耗时 CSV 同步写入文件。
- 开启 RKNN 运行时编译入口：`cmake -S . -B build -DYOSEG_WITH_RKNN=ON`。

## 当前状态（2026-05-13）

- 已完成 C++ 工程骨架与模块拆分：`capture/infer/planner/ros_bridge/core`。
- 已完成本地可运行链路（CPU 后端烟测通过）：
  - 程序可启动并输出实时 CSV 指标。
  - 已验证 `realtime_runner.exe` 可执行。
- 已落地性能基建：
  - 有界队列流水线（producer/consumer）。
  - `FramePacket` 对象池复用。
  - `preprocess_move`、`occupancy swap`、路径容器预留容量。
  - 单一路径 warmup：`capture.warmup(...)`。
  - 指标输出：`moving_fps`、`copy_pre/post/plan/ros/total`、`alloc_count`。
  - 每 100 帧摘要行：`#summary ... avg_ms/max_ms/avg_fps/delta_copy_*`。

## 运行示例

CPU 本地验证（推荐先跑通）：

```bash
./build_ninja/apps/realtime_runner/realtime_runner --frames 200 --warmup-frames 8 --queue-capacity 4 --backend cpu --model noop --profile-out ./profile.csv
```

ROS 相关参数（realtime_runner）：

- `--ros-enable`
- `--ros-rate <hz>`
- `--ros-occ-topic <topic>`
- `--ros-cloud-topic <topic>`
- `--ros-cell-size <meters>`（兼容参数，当前无量纲网格模式下忽略）

RKNN 路径（需要有效 `.rknn` 模型）：

```bash
./build_ninja/apps/realtime_runner/realtime_runner --backend rknn --model ./weights/xxx.rknn --frames 200
```

若 `--backend rknn` 但模型不存在，程序会报 `failed to init infer engine`，这是预期保护行为。

## 无量纲网格约定（当前默认）

- 当前采用无量纲网格建图：`640x640` 像素直接映射为导航网格索引。
- 工程命名语义统一为 `grid units`，内部 `x/y` 表示网格索引，不代表真实米制。
- ROS2 `OccupancyGrid.info.resolution` 固定发布 `1.0` 作为名义占位值（非物理尺度）。
- ROS2 `PointCloud2` 当前发布的是 `grid units` 坐标（非米制坐标）。
- 后续接入真实标定（如 15 米等不同高度）时，仅需增加线性比例转换模块，不影响现有规划与发布主流程。

## ROS2 启用（Phase 3）

当前 `ros_bridge` 已实现“双路径”：

- 检测到 ROS2 依赖并启用编译开关时：发布真实 `OccupancyGrid + PointCloud2`
- 无 ROS2 依赖时：自动回退到 stub 发布（仅流程占位，不发真实话题）

启用方式：

```bash
cmake -S . -B build_ninja -G Ninja -DYOSEG_WITH_ROS2=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_ninja -j
```

运行示例（ROS2）：

```bash
./build_ninja/apps/realtime_runner/realtime_runner \
  --backend cpu --model noop --frames 200 \
  --ros-enable --ros-rate 5 \
  --ros-occ-topic /octomap/occupancy \
  --ros-cloud-topic /octomap/points \
  --ros-cell-size 1.0
```

## 已知限制

- 当前环境未自动找到 OpenCV 时，采集会回退到合成帧（仍可用于性能与流程验证）。
- YOLOv5-seg 后处理与 D* Lite 仍为迁移骨架实现，需继续替换为完整算法。
- RK3588/NPU 实机验证未在本 README 阶段执行，后续单独验收。
- ROS2 真实话题发布已具备代码路径，但需在具备 ROS2 依赖的环境做最终联调验收。

## 迁移原则

- 不修改旧仓库代码
- 每个阶段都要和旧链路做输出一致性对齐
- 优先保证 RK3588 可部署能力，再做桌面增强
