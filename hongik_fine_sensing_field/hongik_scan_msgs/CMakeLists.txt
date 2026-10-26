cmake_minimum_required(VERSION 3.12)
project(hongik_scan_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ProfileRaw.msg"
  "msg/CrossSectionFeatures.msg"
  "msg/SeamPolyline.msg"
  "action/AutoTeach.action"
  "action/PostWeldScan.action"
  DEPENDENCIES std_msgs geometry_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
