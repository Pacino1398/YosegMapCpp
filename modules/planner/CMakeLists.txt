add_library(yoseg_planner
  src/planner_stub.cpp
  src/dstar_lite_planner.cpp
)

target_include_directories(yoseg_planner PUBLIC include)
target_link_libraries(yoseg_planner PUBLIC yoseg_core)
