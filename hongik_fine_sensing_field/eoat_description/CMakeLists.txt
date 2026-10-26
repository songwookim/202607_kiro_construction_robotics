cmake_minimum_required(VERSION 3.12)
project(eoat_description)

find_package(ament_cmake REQUIRED)

# install-only package: xacro, single STL mesh, calibration YAML
install(DIRECTORY urdf meshes calib
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
