cmake_minimum_required(VERSION 3.8)
project(construct_motion_tests)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(straight_line_moveit_test src/straight_line_moveit_test.cpp)
target_compile_features(straight_line_moveit_test PRIVATE cxx_std_17)
ament_target_dependencies(
  straight_line_moveit_test
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS straight_line_moveit_test DESTINATION lib/${PROJECT_NAME})

ament_package()
