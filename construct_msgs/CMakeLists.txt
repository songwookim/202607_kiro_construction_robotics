cmake_minimum_required(VERSION 3.8)
project(construct_msgs)

# ROS 2 Humble on Ubuntu 22.04 uses the CPython 3.10 ABI.  Conda on this
# workstation may expose a newer python3 first in PATH, which produces an
# extension module that rclpy cannot import.  Resolve one 3.10 interpreter and
# make both modern and legacy CMake Python discovery use it.
find_package(Python3 3.10 EXACT REQUIRED COMPONENTS Interpreter)
set(
  PYTHON_EXECUTABLE
  "${Python3_EXECUTABLE}"
  CACHE FILEPATH "Python interpreter used by ROS interface generators" FORCE
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/CartesianPath.action"
  "msg/DigitalIoState.msg"
  "srv/SetDigitalOutput.srv"
  DEPENDENCIES geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
