# KIRO ROS 2 graph and debugging map

This document separates the planning, visualization, control, and RB hardware
layers. All names below assume the default namespace.

## End-to-end path

```mermaid
flowchart LR
  GUI["weld_action_gui<br/>or cartesian_path_client"]
  AS["cartesian_path_action_server"]
  MG["move_group"]
  ET["MoveIt execute_trajectory"]
  RTC["right_manipulator_controller"]
  CM["controller_manager / ros2_control"]
  HW["RBPodoHardwareInterface"]
  RB["RB Control Box<br/>192.168.1.10"]
  JS["/joint_states"]
  RSP["robot_state_publisher"]
  TF["/tf, /tf_static"]
  RVIZ["RViz"]
  VISER["Viser"]

  GUI -->|"/cartesian_path action"| AS
  AS -->|"/compute_cartesian_path service"| MG
  AS -->|"/execute_trajectory action"| ET
  ET -->|"/right_manipulator_controller/<br/>follow_joint_trajectory action"| RTC
  RTC -->|"position command interfaces"| CM
  CM --> HW
  HW -->|"TCP / move_servo_j"| RB
  RB -->|"joint position feedback"| HW
  CM --> JS
  JS --> RSP
  RSP --> TF
  JS --> RVIZ
  TF --> RVIZ
  JS --> VISER
  AS -->|"/weld_6d_poses<br/>/weld_path_markers<br/>/display_planned_path"| RVIZ
  AS -->|"/weld_6d_poses<br/>/display_planned_path"| VISER
```

The normal MoveIt execution path does **not** call RB Podo `move_j` or `move_l`
actions. The joint trajectory controller writes ros2_control position command
interfaces. `RBPodoHardwareInterface::write()` then streams those positions to
the control box using `move_servo_j`.

## Application nodes

| Node | Receives | Sends | Purpose |
|---|---|---|---|
| `/weld_action_gui` | `World → right_manipulator_ee_point` TF; `/cartesian_path` feedback/result | `/cartesian_path` goals; `/weld_path_markers`; `/weld_6d_poses` | Acquire/generate/edit a weld path and request right-arm planning |
| `/laser_weld_path_action_client` | TF for live scenarios; `/cartesian_path` feedback/result | `/cartesian_path` goal | Command-line scenario client |
| `/cartesian_path_action_server` | `/cartesian_path` goals; `/compute_cartesian_path` response; `/execute_trajectory` result | weld visualization topics; `/display_planned_path`; MoveIt service/action requests | Validate, visualize, plan, optionally execute |
| `/kiro_viser_debug_viewer` | `/joint_states`; `/weld_6d_poses`; `/display_planned_path` | Browser websocket/HTTP only | Browser robot/path debugging |
| `/h600_modbus_bridge` | H600 Modbus FC03/06/16; `/h600/set_command` | `/h600/status`; Modbus responses | Safe H600 command register and feedback bridge |
| `/h600_modbus_gui` | `/h600/status`; `/h600/traffic`; operator command form | `/h600/set_command` requests | Register dashboard and packet inspector |
| `/robot_state_publisher` | `/joint_states`, `robot_description` | `/tf`, `/tf_static` | Joint state to link-frame transforms |
| `/move_group` | `/joint_states`, planning scene and requests | planning services/actions, `/display_planned_path`, monitored scene | MoveIt planning and execution coordinator |

## Custom Cartesian action

Action name and type:

```text
/cartesian_path
construct_msgs/action/CartesianPath
```

Goal:

| Field | Meaning |
|---|---|
| `waypoints` | Ordered `geometry_msgs/Pose[]` in `World` |
| `planning_group` | `right_manipulator` or `left_manipulator` |
| `interpolation_step` | Maximum Cartesian sample spacing in metres |
| `velocity_scale` | `(0, 1]`; scales trajectory time, velocity and acceleration |
| `execute_requested` | `false` previews/approves; `true` requests controller execution |
| `reuse_approved_plan` | Execute only the last exactly matching path/speed plan |
| `visualize_path` | Show/hide compact 6D waypoint markers and connecting line |
| `enable_arc` | Request H600 ARC only around actual trajectory execution |
| `weld_*_raw` | H600 current, voltage, and V-offset raw register values |

Feedback contains `current_pose`, `waypoint_index`, and `progress`. Result
contains `success`, `message`, `final_pose`, and `sampled_path`.

ROS actions expand internally to send-goal, get-result and cancel services plus
feedback/status topics. Use `ros2 action` commands instead of depending on
those generated endpoint names directly.

## Visualization contracts

| Topic | Type | Publisher | Consumers |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | `joint_state_broadcaster` | MoveIt, robot_state_publisher, RViz, Viser |
| `/tf` | `tf2_msgs/TFMessage` | robot_state_publisher and static publishers | RViz, GUI TF buffer |
| `/weld_path_markers` | `visualization_msgs/MarkerArray` | GUI and action server | RViz |
| `/weld_6d_poses` | `geometry_msgs/PoseArray` | GUI and action server | RViz and Viser |
| `/display_planned_path` | `moveit_msgs/DisplayTrajectory` | action server, MoveIt/Tesseract demos | RViz and Viser |
| `/monitored_planning_scene` | `moveit_msgs/PlanningScene` | move_group | RViz MotionPlanning display |
| `/h600/status` | `construct_msgs/WelderStatus` | H600 bridge | GUI and diagnostic tools |
| `/h600/traffic` | `construct_msgs/ModbusTrace` | H600 bridge | Wireshark-style H600 GUI |

RViz and Viser are synchronized exactly when both display **live state**:
both consume `/joint_states` and use `World`. An RViz MotionPlanning gizmo is
a local goal-state interactive marker; it is not robot state and is therefore
not mirrored to Viser. For plan preview, Viser automatically selects and plays
new `/display_planned_path` messages. Select `Live joint states` in Viser to
compare actual/fake-controller state with RViz.

The two viewers have independent cameras and animation clocks. Equal camera
position or frame-perfect preview animation is not a ROS synchronization
contract.

## MoveIt and controller interfaces

The action server uses:

- `/compute_cartesian_path`
  (`moveit_msgs/srv/GetCartesianPath`)
- `/execute_trajectory`
  (`moveit_msgs/action/ExecuteTrajectory`)

MoveIt sends trajectories to:

- `/right_manipulator_controller/follow_joint_trajectory`
- `/left_manipulator_controller/follow_joint_trajectory`
- `/robot_head_controller/follow_joint_trajectory`

All use `control_msgs/action/FollowJointTrajectory`. Controller lifecycle and
hardware interfaces are managed by `/controller_manager`; inspect its services
with:

```bash
ros2 service list | grep controller_manager
ros2 control list_controllers
ros2 control list_hardware_interfaces
```

## RB hardware layer

With `use_fake_right_hardware:=false`, the right ros2_control system loads
`rbpodo_hardware/RBPodoHardwareInterface` with:

```text
hardware_namespace = right
robot_ip = 192.168.1.10
```

The hardware plugin exposes joint position/velocity/effort command interfaces
and position/effort state interfaces. The trajectory controller currently
claims only the six right-arm **position** command interfaces.

The plugin also creates `/right_rbpodo_hardware` with lower-level services such
as `eval`, `task_*`, `set_speed_bar`, and controller configuration, plus
`move_j`, `move_l`, `move_jb2`, and `move_pb` actions. These are available for
direct RB Podo workflows but are separate from this MoveIt trajectory path.
The left robot uses the equivalent `/left_rbpodo_hardware/*` names.

## Commands for self-debugging

Start a safe fake-hardware stack:

```bash
source /home/irs/ros2_ws/install/setup.bash
ros2 launch construct_robot viser_debug.launch.py
```

Inspect the runtime graph:

```bash
ros2 node list
ros2 node info /cartesian_path_action_server
ros2 node info /weld_action_gui
ros2 topic info /joint_states --verbose
ros2 topic info /display_planned_path --verbose
ros2 service type /compute_cartesian_path
ros2 action info /cartesian_path
ros2 action info /execute_trajectory
ros2 action info /right_manipulator_controller/follow_joint_trajectory
```

Watch data:

```bash
ros2 topic hz /joint_states
ros2 topic echo /weld_6d_poses --once
ros2 topic echo /display_planned_path --once
ros2 topic echo /tf --field transforms
```

Check frames:

```bash
ros2 run tf2_ros tf2_echo World right_manipulator_ee_point
ros2 run tf2_tools view_frames
```

Check controller ownership:

```bash
ros2 control list_controllers
ros2 control list_hardware_components
ros2 control list_hardware_interfaces
```

For actual hardware, verify the controller list, TCP reachability, operation
mode, and a low velocity scale before enabling `execute_motion:=true`.

Viser renders two robot instances. The solid robot follows the selected source
(live, planned trajectory, or manual joints); the cyan transparent robot always
follows the latest physical/fake `/joint_states`.

## H600 Modbus/ARC

The bridge implements the register behavior prototyped in `~/test.py`:

- command: 201 ready, 202 bit3 gas/bit0 ARC, 204 current, 205 voltage,
  206 V-offset
- feedback: 211 status, 212 current, 213 voltage
- Modbus functions: FC03, FC06, FC16
- ROS service: `/h600/set_command`
- ROS status topic: `/h600/status`

ARC ON requires all of these conditions:

1. `allow_arc_output:=true` on the bridge.
2. An H600 Modbus client is currently connected.
3. `robot_ready` is true.
4. For nonzero values, launch `allow_nonzero_setpoints:=true` and send the
   request safety flag.
5. MoveIt planning completed fully and `execute_motion` is true.

Inspect it with:

```bash
ros2 topic echo /h600/status
ros2 topic echo /h600/traffic
ros2 service type /h600/set_command
ros2 node info /h600_modbus_bridge
```
