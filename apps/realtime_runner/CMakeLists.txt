add_executable(realtime_runner
  src/main.cpp
)

target_link_libraries(realtime_runner PRIVATE
  yoseg_core
  yoseg_capture
  yoseg_infer
  yoseg_planner
  yoseg_ros_bridge
)
