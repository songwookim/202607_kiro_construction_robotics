cmake_minimum_required(VERSION 3.8)
project(construct_tesseract)

find_package(ament_cmake REQUIRED)
find_package(tesseract_environment QUIET)

if(tesseract_environment_FOUND)
  add_executable(environment_check src/environment_check.cpp)
  target_compile_features(environment_check PRIVATE cxx_std_17)
  target_link_libraries(environment_check PRIVATE tesseract::tesseract_environment)
  install(TARGETS environment_check DESTINATION lib/${PROJECT_NAME})
else()
  message(WARNING
    "Tesseract Robotics was not found. construct_tesseract will install only "
    "its reproducible source-build files; see README.md.")
endif()

install(FILES README.md tesseract_humble.repos
  DESTINATION share/${PROJECT_NAME})

ament_package()
