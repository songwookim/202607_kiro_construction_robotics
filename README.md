# KIRO Construction Robotics

ROS 2 Humble packages for the dual-arm KIRO construction robot.

## Packages

- `construct_description`: URDF, CAD source CSV and mesh assets
- `construct_moveit_config`: MoveIt 2, ros2_control and RViz configuration
- `construct_robot_bringup`: controller and hardware bringup
- `construct_msgs`: Cartesian 6D pose path action
- `construct_robot`: dry-run Cartesian action client/server
- `construct_tesseract`: Tesseract Robotics model validation and pinned setup

## Build and test

```bash
cd /home/irs/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-up-to construct_robot construct_moveit_config
source install/setup.bash
colcon test --packages-select \
  construct_description construct_msgs construct_robot \
  construct_robot_bringup construct_moveit_config
colcon test-result --verbose
```

## Cartesian 6D pose action dry run

Terminal 1:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 run construct_robot cartesian_path_server
```

Terminal 2:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 run construct_robot cartesian_path_client \
  --ros-args -p use_sim_time:=false
```

The goal is an ordered `geometry_msgs/Pose[]`. Feedback contains the current
interpolated 6D pose, waypoint index and progress. The result contains the
final pose and sampled path. The server is deliberately dry-run only and does
not command hardware.

See `construct_tesseract/README.md` for the ARM64/Humble Tesseract setup and
model-validation command.
