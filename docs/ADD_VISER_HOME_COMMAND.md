# Add a collision-checked “move right arm to initial pose” button

Keep visualization and robot commands separate while learning the code. The
Viser button should send a MoveIt `MoveGroup` goal, not write directly to the
`FollowJointTrajectory` controller. MoveIt then checks joint limits and
collisions before it forwards a trajectory to the active controller.

Start with fake right-arm hardware and ARC disabled. The target below comes
from `construct_description/config/initial_positions.yaml`.

## 1. Add imports to `viser_viewer.py`

```python
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionClient
```

## 2. Create the MoveIt client in `RosViserBridge.__init__`

Put this after `super().__init__(...)`:

```python
self._move_group_client = ActionClient(
    self,
    MoveGroup,
    "/move_action",
)
```

## 3. Add an armed command folder in `_build_gui`

The extra checkbox prevents an accidental browser click from moving hardware.

```python
with self._server.gui.add_folder(
    "Robot commands",
    expand_by_default=False,
):
    enable_commands = self._server.gui.add_checkbox(
        "Enable real robot commands",
        initial_value=False,
    )
    go_initial = self._server.gui.add_button(
        "Move RIGHT arm to initial pose",
    )
    command_status = self._server.gui.add_text(
        "Command status",
        "disarmed",
        disabled=True,
    )

    @go_initial.on_click
    def _go_initial(_event):
        if not enable_commands.value:
            command_status.value = "REJECTED: enable commands first"
            return
        command_status.value = "sending collision-checked MoveIt goal"
        self.move_right_to_initial(command_status)
```

## 4. Add this method to `RosViserBridge`

```python
def move_right_to_initial(self, status_handle):
    joint_names = (
        "right_manipulator_joint1",
        "right_manipulator_joint2",
        "right_manipulator_joint3",
        "right_manipulator_joint4",
        "right_manipulator_joint5",
        "right_manipulator_joint6",
    )
    target = (
        3.14,
        0.314159,
        1.43117,
        1.1002556,
        0.261799,
        2.89725,
    )
    if not self._move_group_client.wait_for_server(timeout_sec=2.0):
        status_handle.value = "ERROR: /move_action unavailable"
        return

    constraints = Constraints()
    for name, position in zip(joint_names, target):
        joint = JointConstraint()
        joint.joint_name = name
        joint.position = position
        joint.tolerance_above = 0.005
        joint.tolerance_below = 0.005
        joint.weight = 1.0
        constraints.joint_constraints.append(joint)

    goal = MoveGroup.Goal()
    goal.request.group_name = "right_manipulator"
    goal.request.num_planning_attempts = 5
    goal.request.allowed_planning_time = 5.0
    goal.request.max_velocity_scaling_factor = 0.05
    goal.request.max_acceleration_scaling_factor = 0.05
    goal.request.start_state.is_diff = True
    goal.request.goal_constraints = [constraints]
    goal.planning_options.plan_only = False

    future = self._move_group_client.send_goal_async(goal)

    def goal_response(done):
        handle = done.result()
        if not handle.accepted:
            status_handle.value = "REJECTED by MoveIt"
            return
        status_handle.value = "accepted; planning/executing"
        result_future = handle.get_result_async()

        def finished(result_done):
            error_code = result_done.result().result.error_code.val
            status_handle.value = f"finished; MoveIt code={error_code}"

        result_future.add_done_callback(finished)

    future.add_done_callback(goal_response)
```

The first real-hardware test should use 5% velocity, an operator at the
emergency stop, a verified collision-free initial pose, and the H600 console
showing ARC/GAS/READY all OFF.

## 5. Verify the control path before clicking

```bash
ros2 control list_controllers
ros2 action info /move_action
ros2 action info \
  /right_manipulator_controller/follow_joint_trajectory
ros2 topic hz /joint_states
```

`right_manipulator_controller` must be `active`; both action servers must
exist; and `/joint_states` must update from the physical RB controller.
