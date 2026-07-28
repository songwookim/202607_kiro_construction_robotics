# KIRO Tesseract validation

This package keeps the normal KIRO workspace buildable when Tesseract is not
installed. When a Tesseract overlay is sourced, it also builds
`environment_check`, which loads the real KIRO URDF and SRDF.

The versions in `tesseract_humble.repos` match the upstream ROS 2 Humble
0.22.x release line.

## Build the Tesseract overlay

```bash
mkdir -p /tmp/kiro_tesseract_ws/src
cd /tmp/kiro_tesseract_ws
vcs import src < /home/irs/ros2_ws/src/construct_robot_ros2/construct_tesseract/tesseract_humble.repos
rosdep install --from-paths src --ignore-src -iry --rosdistro humble
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

Full TrajOpt/Task Composer planning requires `coinor-libipopt-dev`. On this
ARM64 host, the core Environment, Bullet collision, SRDF, URDF and kinematics
packages can be built without IPOPT:

```bash
colcon build --packages-up-to tesseract_environment --packages-skip ifopt \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

## Validate the KIRO model

```bash
source /opt/ros/humble/setup.bash
source /tmp/kiro_tesseract_ws/install/setup.bash
cd /home/irs/ros2_ws
colcon build --packages-select construct_tesseract
source install/setup.bash
export TESSERACT_RESOURCE_PATH=/home/irs/ros2_ws/install
ros2 run construct_tesseract environment_check \
  /home/irs/ros2_ws/install/construct_description/share/construct_description/urdf_0528/construct_robot_0528.urdf \
  /home/irs/ros2_ws/install/construct_moveit_config/share/construct_moveit_config/config/construct_robot_0528.srdf
```

After IPOPT is installed, build through `tesseract_motion_planners` and use
the upstream `FreespacePipeline` or OMPL examples as the starting point for
the first trajectory-planning integration.
