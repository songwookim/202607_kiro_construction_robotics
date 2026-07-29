cmake_minimum_required(VERSION 3.8)
project(construct_tesseract)

find_package(ament_cmake REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tesseract_environment QUIET)
find_package(tesseract_motion_planners QUIET COMPONENTS ompl trajopt simple)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

if(tesseract_environment_FOUND)
  add_executable(environment_check src/environment_check.cpp)
  target_compile_features(environment_check PRIVATE cxx_std_17)
  target_link_libraries(environment_check PRIVATE tesseract::tesseract_environment)
  install(TARGETS environment_check DESTINATION lib/${PROJECT_NAME})

  if(tesseract_motion_planners_FOUND)
    add_executable(motion_planning_check src/motion_planning_check.cpp)
    target_compile_features(motion_planning_check PRIVATE cxx_std_17)
    target_link_libraries(
      motion_planning_check
      PRIVATE
        tesseract::tesseract_environment
        tesseract::tesseract_motion_planners_ompl
        tesseract::tesseract_motion_planners_trajopt
        tesseract::tesseract_motion_planners_simple)
    install(TARGETS motion_planning_check DESTINATION lib/${PROJECT_NAME})

    add_executable(dual_arm_rviz_demo src/dual_arm_rviz_demo.cpp)
    target_compile_features(dual_arm_rviz_demo PRIVATE cxx_std_17)
    ament_target_dependencies(
      dual_arm_rviz_demo
      moveit_msgs
      rclcpp
      trajectory_msgs
      visualization_msgs)
    target_link_libraries(
      dual_arm_rviz_demo
      tesseract::tesseract_environment
      tesseract::tesseract_motion_planners_ompl
      tesseract::tesseract_motion_planners_simple)
    install(TARGETS dual_arm_rviz_demo DESTINATION lib/${PROJECT_NAME})

    add_executable(dual_arm_constrained_demo src/dual_arm_constrained_demo.cpp)
    target_compile_features(dual_arm_constrained_demo PRIVATE cxx_std_17)
    ament_target_dependencies(
      dual_arm_constrained_demo
      moveit_msgs
      rclcpp
      trajectory_msgs
      visualization_msgs)
    target_link_libraries(
      dual_arm_constrained_demo
      tesseract::tesseract_environment)
    install(TARGETS dual_arm_constrained_demo DESTINATION lib/${PROJECT_NAME})
  endif()
else()
  message(WARNING
    "Tesseract Robotics was not found. construct_tesseract will install only "
    "its reproducible source-build files; see README.md.")
endif()

install(FILES README.md tesseract_humble.repos
  DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/patch_arm64.sh
  DESTINATION share/${PROJECT_NAME}/scripts)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

ament_package()
