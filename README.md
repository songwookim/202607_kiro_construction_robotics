# KIRO Construction Robotics

ROS 2 Humble packages for the dual-arm KIRO construction robot.

## Packages

- `construct_description`: URDF, CAD source CSV and mesh assets
- `construct_moveit_config`: MoveIt 2, ros2_control and RViz configuration
- `construct_robot_bringup`: controller and hardware bringup
- `construct_msgs`: Cartesian 6D pose path action
- `construct_robot`: Cartesian action client/server with MoveIt/RViz integration
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

## Cartesian 6D pose action with RViz

First launch MoveIt and RViz with fake hardware:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 launch construct_moveit_config moveit.launch.py \
  use_fake_left_hardware:=true use_fake_right_hardware:=true
```

Then start the action server. It publishes weld-point markers and the planned
robot trajectory to RViz:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 run construct_robot cartesian_path_server --ros-args \
  -p use_moveit:=true -p execute_motion:=true -p planning_frame:=World
```

Finally send a straight laser-scanner scenario from the current right-arm TCP:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 run construct_robot cartesian_path_client \
  --planning-group right_manipulator \
  --scenario laser-live-straight
```

The goal is an ordered `geometry_msgs/Pose[]`. Feedback contains the current
interpolated 6D pose, waypoint index and progress. The result contains the
final pose and sampled path. Keep `execute_motion:=false` for visualization
without controller execution.

See `construct_tesseract/README.md` for the ARM64/Humble Tesseract setup and
model-validation command.


```
ROS_DOMAIN_ID=162 ros2 run construct_motion_tests straight_line_moveit_test \
  --ros-args \
  -p group:=right_manipulator \
  -p axis:="'y'" \
  -p distance:=0.01 \
  -p speed:=0.02 \
  -p execute:=false # true시 plan and execute
```
