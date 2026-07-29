# KIRO Construction Robotics

ROS 2 Humble packages for the dual-arm KIRO construction robot.

## Packages

- `construct_description`: URDF, ros2_control, and optimized mesh assets
- `construct_moveit_config`: MoveIt 2, ros2_control and RViz configuration
- `construct_msgs`: Cartesian 6D pose path action
- `construct_robot`: hardware launch and Cartesian action/MoveIt utilities
- `construct_tesseract`: Tesseract Robotics model validation and pinned setup

## Build and test

```bash
cd /home/irs/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-up-to construct_robot construct_moveit_config
source install/setup.bash
colcon test --packages-select \
  construct_description construct_msgs construct_robot \
  construct_moveit_config
colcon test-result --verbose
```

Standalone ros2_control launch:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 launch construct_robot control.launch.py
```

## Cartesian 6D pose action with RViz

The complete fake-hardware GUI scenario is one command:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 launch construct_robot weld_action_gui.launch.py
```

The pinned Tesseract dependency overlay lives at
`/home/irs/ros2_ws/src/tesseract_ws`. Its `COLCON_IGNORE` keeps it out of the
main workspace build; build it explicitly with `--base-paths src`.

Use **Acquire weld points**, inspect the three live TCP-relative poses, then
use **Plan + execute right arm**. The GUI reports Action feedback/result while
RViz displays the yellow weld points, red seam, right-arm trajectory and an
RGB frame at every 6D pose (X red, Y green, Z blue). The same scanner output
is available as `geometry_msgs/PoseArray` on `/weld_6d_poses`.

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

## Viser browser debug viewer

Viser can replace the RViz visualization window while retaining the ROS and
MoveIt backend. Start the complete fake-hardware stack without RViz:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 launch construct_robot viser_debug.launch.py
```

Open `http://localhost:8080`, then send the right-arm scanner scenario:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 run construct_robot cartesian_path_client \
  --planning-group right_manipulator \
  --scenario laser-live-straight
```

The browser shows the live dual-arm robot, orange weld points, an RGB 6D frame
and label at every point, the red weld seam, and the planned left/right TCP
paths. The control panel reports ROS topic age and provides layer visibility,
trajectory playback/scrubbing, and manual sliders for all robot joints.

The viewer listens to:

- `/joint_states` for the live robot
- `/weld_6d_poses` for scanner/weld poses
- `/display_planned_path` for MoveIt trajectories

Planning-only visualization is the default. To send the trajectory to fake or
real controllers, explicitly use `execute_motion:=true`. The Viser Python
packages `viser` and `yourdfpy` are optional runtime dependencies.

The viewer can also be attached to an already-running ROS stack:

```bash
ros2 run construct_robot viser_viewer --port 8080
```

```
ROS_DOMAIN_ID=162 ros2 run construct_motion_tests straight_line_moveit_test \
  --ros-args \
  -p group:=right_manipulator \
  -p axis:="'y'" \
  -p distance:=0.01 \
  -p speed:=0.02 \
  -p execute:=false # true시 plan and execute
```
