cmake_minimum_required(VERSION 3.20)
project(YosegMapCpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YOSEG_WITH_RKNN "Enable RKNN runtime integration" OFF)
option(YOSEG_WITH_ROS2 "Enable ROS2 publisher integration" OFF)
option(YOSEG_ENABLE_BENCHMARKS "Enable benchmarks" ON)
option(YOSEG_WITH_OPENCV "Enable OpenCV capture path" ON)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(modules/core)
add_subdirectory(modules/capture)
add_subdirectory(modules/infer)
add_subdirectory(modules/planner)
add_subdirectory(modules/ros_bridge)
add_subdirectory(apps/realtime_runner)

if(YOSEG_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
