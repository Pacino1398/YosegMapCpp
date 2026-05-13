# Python to C++ Mapping

本仓库为独立迁移实现，原仓库保持不变。

## 模块映射

- `app/inference/*` -> `modules/infer`
- `app/planning/*` -> `modules/planner`
- `app/mapping/*` -> `modules/planner`（地图与代价场）
- `ROS2 发布逻辑` -> `modules/ros_bridge`
- `realtime 主循环` -> `apps/realtime_runner`

## 迁移原则

1. RK3588 部署优先：推理以 RKNN/NPU 为第一目标。
2. 非推理热点 C++ 化：预处理、后处理、规划、消息组包改为 C++。
3. 实时优先：流水线解耦、允许丢帧保低延迟。
4. 可验证：每阶段保留耗时统计与结果对齐脚本。

## 近期里程碑

1. `IInferEngine` 抽象与 `rknn` 引擎接入。
2. `CaptureSource` 接入真实视频流（OpenCV/GStreamer）。
3. `Planner` 核心迁移并提供增量更新 API。
4. `rclcpp` Publisher + QoS + 复用缓冲区。
