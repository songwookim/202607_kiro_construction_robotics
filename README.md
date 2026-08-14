# KIRO Construction Robotics

ROS 2 Humble packages for the dual-arm KIRO construction robot.

The complete node/topic/service/action and RB controller data flow is documented
in [`docs/ROS_GRAPH.md`](docs/ROS_GRAPH.md). Path-generation equations, MoveIt
conversion, speed scaling, and ros2_control diagrams are in
[`docs/MOTION_MATH_AND_CONTROL_FLOW.md`](docs/MOTION_MATH_AND_CONTROL_FLOW.md).

## Packages

- `construct_description`: URDF, ros2_control, and optimized mesh assets
- `construct_moveit_config`: MoveIt 2, ros2_control and RViz configuration
- `construct_msgs`: Cartesian 6D pose path action
- `construct_robot`: welding GUI, hardware launch, and Cartesian motion server
- `construct_tesseract`: Tesseract Robotics model validation and pinned setup

## Build and test

```bash
cd /home/irs/ros2_ws
source src/construct_robot_ros2/scripts/use_ros_python.bash
/usr/bin/colcon build --packages-up-to construct_robot construct_moveit_config \
  --symlink-install \
  --cmake-args \
  -DPYTHON_EXECUTABLE=/home/irs/ros2_ws/.venv/bin/python \
  -DPython3_EXECUTABLE=/home/irs/ros2_ws/.venv/bin/python
/usr/bin/colcon test --packages-select \
  construct_description construct_msgs construct_robot \
  construct_moveit_config
/usr/bin/colcon test-result --verbose
```

ROS 2 Humble uses the CPython 3.10 ABI. Always source
`scripts/use_ros_python.bash` in a new terminal; it selects the workspace
`.venv` and removes Conda Python 3.14 from `PATH`.

## Cartesian 6D pose action with RViz

The normal GUI starts both physical RB connections immediately:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 launch construct_robot weld_action_gui.launch.py
```

This command is **not a fake-hardware test**. It targets LEFT
`192.168.1.11` and RIGHT `192.168.1.12` and activates the arm trajectory
controllers. Do not start it unattended around a motion-enabled robot.

The pinned Tesseract dependency overlay lives at
`/home/irs/ros2_ws/src/tesseract_ws`. Its `COLCON_IGNORE` keeps it out of the
main workspace build; build it explicitly with `--base-paths src`.

Use **Acquire weld points**, inspect the three live TCP-relative poses, then
use **1 · Plan Preview**, inspect the trajectory in RViz, then use
**2 · Execute Approved Plan**. Any pose or speed edit invalidates the approval.
RViz displays the yellow weld points, red seam, right-arm trajectory and an
RGB frame at every 6D pose (X red, Y green, Z blue). The same scanner output
is available as `geometry_msgs/PoseArray` on `/weld_6d_poses`.

The path table is editable. Select a waypoint to change its XYZ/quaternion,
duplicate/delete/reorder it, nudge it along a World axis, or capture the
physical right TCP into the table. Every edit is immediately republished to
RViz. **Generate circle** creates a World-YZ circle around the
current right TCP. With **TCP +Z faces center**, every waypoint gets a distinct
6D orientation which continuously looks toward the circle center.

**Show planned path** toggles the compact waypoint spheres, RGB 6D axes, labels,
and connecting line in RViz. Weaving is applied to the current
acquired/taught seam, rather than creating a second unrelated line. Amplitude,
cycles, samples per cycle, and Tool/World transverse axis are editable.

The launch starts RViz and the weld GUI together by default. RViz follows live
`/joint_states` and previews planned trajectories from MoveIt.

The normal GUI launch immediately attempts both physical RB connections and
starts MoveIt, RViz, and the GUI in one lifecycle:

```bash
ros2 launch construct_robot weld_action_gui.launch.py \
  left_robot_ip:=192.168.1.11 \
  right_robot_ip:=192.168.1.12
```

Connection success requires live left/right RBPodo `system_state`, not merely
`/joint_states`. The GUI has no Connect/Disconnect controls and shows only
`LEFT O/X` and `RIGHT O/X`. Both arm states and commands are initialized from
the measured robot pose; no configured initial pose is selected or executed.
Once both arms and MoveGroup are ready, the GUI asks RViz to copy its current
PlanningScene state into the orange Goal State.

Welding uses the direct Hi-COMM TCP client integrated into `weld_action_gui`.
The Cartesian action server is motion-only; D-WELD SET/ON/OFF and welding
feedback are handled by `hicomm_welder.py` and explicit Sequence Builder steps.

```bash
ros2 launch construct_robot weld_action_gui.launch.py \
  right_robot_ip:=192.168.1.12 \
  execute_motion:=true
```

See [`docs/RAINBOW_CONTROL_BOX_IO.md`](docs/RAINBOW_CONTROL_BOX_IO.md) for the
live DI monitor, guarded DO controls, and a safe procedure for identifying
control-box wiring.

See [`docs/ADD_LEFT_RB11_CONNECTION.md`](docs/ADD_LEFT_RB11_CONNECTION.md) for
the atomic `.11` left + `.12` right connection controls, fake RViz
startup, measured-pose synchronization, and connection-failure recovery.

See `docs/CONTINUOUS_CIRCLE.md` for the exact waypoint, quaternion,
interpolation, and multi-lap changes used to make circular welding continuous.

First launch MoveIt and RViz with fake hardware:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 launch construct_moveit_config moveit.launch.py \
  use_fake_left_hardware:=true use_fake_right_hardware:=true
```

The goal is an ordered `geometry_msgs/Pose[]`. Feedback contains the current
interpolated 6D pose, waypoint index and progress. The result contains the
final pose and sampled path. Keep `execute_motion:=false` for visualization
without controller execution.

See `construct_tesseract/README.md` for the ARM64/Humble Tesseract setup and
model-validation command.
