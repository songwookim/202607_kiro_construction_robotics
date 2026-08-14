# Dual RB startup

The user-facing launch uses both physical RB arms by default:

- left control box: `192.168.1.11`
- right control box: `192.168.1.12`

```bash
ros2 launch construct_robot weld_action_gui.launch.py
```

There is no connection service or Connect/Disconnect button. The GUI launch
starts MoveIt, ros2_control, and the Cartesian server directly and defaults
both arm components to `rbpodo_hardware/RBPodoHardwareInterface`.
Diagnostic launch profiles can set
`use_fake_left_hardware` and `use_fake_right_hardware` to `true` explicitly.

## Startup sequence

1. Both RBPodo hardware components connect to their command and state ports.
2. Each hardware component reads measured `jnt_ang` values during activation.
3. ros2_control initializes its position states and commands from those
   measured values.
4. `joint_state_broadcaster` publishes the synchronized state.
5. The trajectory controllers become active, followed by MoveGroup.
6. The GUI shows `LEFT O` and `RIGHT O` only after fresh hardware feedback,
   MoveGroup, and the corresponding controller are ready.
7. After both arms are ready, the GUI publishes one
   `std_msgs/msg/Empty` on `/rviz/moveit/update_goal_state`. RViz then copies
   its own current PlanningScene state into the orange Goal State.

No configured initial-position file is used by this path. The robot is not
commanded to a saved pose during startup.

## Connection indication

The GUI has no connection controls. It displays only:

```text
LEFT  O/X    RIGHT  O/X
```

`O` requires all six finite measured arm positions on `/joint_states`, a ready
`/move_action`, and, when execution is enabled, the arm's
`FollowJointTrajectory` action. `system_state` remains the detailed IO/status
source but is not the sole connection indicator. If feedback becomes stale the
indication returns to `X`.

## Planning-only diagnosis

To connect and plan without activating the arm trajectory controllers:

```bash
ros2 launch construct_robot weld_action_gui.launch.py execute_motion:=false
```

This still opens the RB connections and reads the measured joint state. It
does not enable trajectory execution.
