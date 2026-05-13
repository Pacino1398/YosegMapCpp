add_library(yoseg_ros_bridge
  src/ros_bridge_stub.cpp
)

target_include_directories(yoseg_ros_bridge PUBLIC include)
target_link_libraries(yoseg_ros_bridge PUBLIC yoseg_core yoseg_planner)

if(YOSEG_WITH_ROS2)
  find_package(rclcpp QUIET)
  find_package(nav_msgs QUIET)
  find_package(sensor_msgs QUIET)
  find_package(std_msgs QUIET)
  if(rclcpp_FOUND AND nav_msgs_FOUND AND sensor_msgs_FOUND AND std_msgs_FOUND)
    target_compile_definitions(yoseg_ros_bridge PRIVATE YOSEG_HAS_ROS2=1)
    target_link_libraries(yoseg_ros_bridge PUBLIC rclcpp::rclcpp)
    target_link_libraries(yoseg_ros_bridge PUBLIC nav_msgs::nav_msgs__rosidl_generator_cpp)
    target_link_libraries(yoseg_ros_bridge PUBLIC sensor_msgs::sensor_msgs__rosidl_generator_cpp)
    target_link_libraries(yoseg_ros_bridge PUBLIC std_msgs::std_msgs__rosidl_generator_cpp)
    message(STATUS "yoseg_ros_bridge: ROS2 enabled")
  else()
    message(STATUS "yoseg_ros_bridge: ROS2 packages not found, using stub publisher")
  endif()
endif()
