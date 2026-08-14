cmake_minimum_required(VERSION 3.22)
project(construct_moveit_config)

find_package(ament_cmake REQUIRED)

ament_package()

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})
