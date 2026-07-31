# KIRO Construction Robotics

ROS 2 Humble packages for the dual-arm KIRO construction robot.

The complete node/topic/service/action and RB controller data flow is documented
in [`docs/ROS_GRAPH.md`](docs/ROS_GRAPH.md).

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
use **1 · Plan Preview**, inspect the trajectory in RViz or Viser, then use
**2 · Execute Approved Plan**. Any pose or speed edit invalidates the approval.
RViz displays the yellow weld points, red seam, right-arm trajectory and an
RGB frame at every 6D pose (X red, Y green, Z blue). The same scanner output
is available as `geometry_msgs/PoseArray` on `/weld_6d_poses`.

The path table is editable. Select a waypoint to change its XYZ/quaternion,
duplicate/delete/reorder it, nudge it along a World axis, or capture the
physical right TCP into the table. Every edit is immediately republished to
RViz and Viser. **Generate circle** creates a World-YZ circle around the
current right TCP. With **TCP +Z faces center**, every waypoint gets a distinct
6D orientation which continuously looks toward the circle center.

**Show planned path** toggles the compact waypoint spheres, RGB 6D axes, labels,
and connecting line in both RViz and Viser. Weaving is applied to the current
acquired/taught seam, rather than creating a second unrelated line. Amplitude,
cycles, samples per cycle, and Tool/World transverse axis are editable.

The launch starts RViz, Viser, and the weld GUI together by default. Viser is
at `http://localhost:8080`; its cyan transparent robot always follows live
`/joint_states`, while the solid robot can preview a planned trajectory.

To use the GUI's **Robot Connect / Robot Disconnect** buttons, start the
supervised launch. The GUI remains open while the supervisor starts or stops
the REAL-RB MoveIt/ros2_control child stack:

```bash
ros2 launch construct_robot weld_supervised.launch.py \
  initial_connected:=false \
  right_robot_ip:=192.168.1.10
```

Connect asks for confirmation, keeps ARC/nonzero outputs locked OFF, and uses
the entered right RB IP. Disconnect stops MoveIt/ros2_control and the hardware
connection without closing the GUI. The GUI reports `ROBOT CONNECTED` only
after live right-arm `/joint_states` arrive. With the ordinary
`weld_action_gui.launch.py`, the connection buttons intentionally report that
no supervisor is available.

The GUI launch also starts the H600 Modbus bridge derived from `~/test.py`.
It serves the 201/202/204/205/206 command registers and publishes decoded
211–213 feedback on `/h600/status`. Its safe defaults are port 1502, ARC
disabled, and nonzero setpoints disabled:

```bash
# Safe fake-hardware integration; RViz + Viser + weld GUI
ros2 launch construct_robot weld_action_gui.launch.py \
  use_fake_right_hardware:=true execute_motion:=true

# Physical right RB; keep ARC disabled for the first motion test
ros2 launch construct_robot weld_action_gui.launch.py \
  use_fake_right_hardware:=false \
  use_initial_right_positions:=false \
  right_robot_ip:=192.168.1.10 \
  execute_motion:=true

# Enable welding only after motion and the H600 map are verified
ros2 launch construct_robot weld_action_gui.launch.py \
  use_fake_right_hardware:=false \
  use_initial_right_positions:=false \
  right_robot_ip:=192.168.1.10 \
  execute_motion:=true \
  allow_arc_output:=true \
  allow_nonzero_setpoints:=true
```

When enabled in the GUI, ARC/ready/gas turn on only after planning succeeds and
immediately before trajectory execution. They are forced off after success,
failure, disconnect, or node shutdown.

The independent Wireshark-style H600 console shows register values, decoded
status bits, RX/TX MBAP+PDU frames, transaction/unit/function/register fields,
raw HEX, and CSV export:

```bash
# Start bridge and diagnostic GUI
ros2 launch construct_robot h600_console.launch.py port:=1502

# When weld_action_gui.launch.py already owns the bridge/port
ros2 launch construct_robot h600_console.launch.py start_bridge:=false
```

See `docs/ADD_VISER_HOME_COMMAND.md` for a small, follow-along example that adds
a collision-checked “move right arm to initial pose” Viser button.

See `docs/CONTINUOUS_CIRCLE.md` for the exact waypoint, quaternion,
interpolation, and multi-lap changes used to make circular welding continuous.

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

CLI weaving test:

```bash
ros2 run construct_robot cartesian_path_client \
  --planning-group right_manipulator \
  --scenario laser-live-weave \
  --velocity-scale 0.2
```

The goal is an ordered `geometry_msgs/Pose[]`. Feedback contains the current
interpolated 6D pose, waypoint index and progress. The result contains the
final pose and sampled path. Keep `execute_motion:=false` for visualization
without controller execution.

See `construct_tesseract/README.md` for the ARM64/Humble Tesseract setup and
model-validation command.

## Viser browser debug viewer

Viser can run beside RViz while retaining the same ROS and MoveIt backend.
Start the complete fake-hardware stack with both viewers:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 launch construct_robot viser_debug.launch.py
```

Use `use_rviz:=false` when only the browser viewer is needed.

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
