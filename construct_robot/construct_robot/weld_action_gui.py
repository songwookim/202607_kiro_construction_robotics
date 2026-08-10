import copy
import math
from pathlib import Path
import queue
import signal
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseArray
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, DisplayTrajectory, JointConstraint
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rbpodo_msgs.msg import SystemState
from rbpodo_msgs.srv import SetDigitalOutput
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray

from construct_msgs.action import CartesianPath
from construct_msgs.msg import WelderStatus
from construct_robot.cartesian_path_common import (
    circle_waypoints,
    linear_pose_waypoints,
    pose_is_valid,
    straight_waypoints,
    tip_link_for_group,
    weaving_from_path,
)
from construct_robot.cartesian_path_server import make_weld_visualization


OBSERVED_H600_IO_CANDIDATES = frozenset((0, 4, 8, 9, 10, 12, 13))
TOUCH_INPUT_PORT = 0
ARM_JOINT_NAMES = {
    arm: frozenset(
        f"{arm}_manipulator_joint{index}" for index in range(1, 7)
    )
    for arm in ("left", "right")
}


def _finite_float(value, description):
    """Return a finite float while rejecting YAML booleans and bad values."""
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a number") from error
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def save_initial_state_yaml(path, planning_group, joint_names, positions, tcp):
    """Atomically save a captured joint state and its TCP pose as YAML."""
    path = Path(path)
    document = {
        "format_version": 1,
        "planning_group": planning_group,
        "joint_state": {
            "names": list(joint_names),
            "positions_rad": [float(value) for value in positions],
        },
        "tcp_pose_world": {
            "position_m": {
                "x": float(tcp.position.x),
                "y": float(tcp.position.y),
                "z": float(tcp.position.z),
            },
            "orientation_xyzw": {
                "x": float(tcp.orientation.x),
                "y": float(tcp.orientation.y),
                "z": float(tcp.orientation.z),
                "w": float(tcp.orientation.w),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            yaml.safe_dump(document, stream, sort_keys=False)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_initial_state_yaml(path):
    """Load and validate a TCP teaching YAML file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("YAML root must be a mapping")
    if document.get("format_version") != 1:
        raise ValueError("unsupported or missing format_version (expected 1)")

    planning_group = document.get("planning_group")
    if planning_group not in ("left_manipulator", "right_manipulator"):
        raise ValueError(
            "planning_group must be left_manipulator or right_manipulator"
        )
    arm = planning_group.removesuffix("_manipulator")

    joint_state = document.get("joint_state")
    if not isinstance(joint_state, dict):
        raise ValueError("joint_state must be a mapping")
    names = joint_state.get("names")
    positions = joint_state.get("positions_rad")
    if not isinstance(names, list) or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError("joint_state.names must be a list of joint names")
    if set(names) != ARM_JOINT_NAMES[arm] or len(names) != 6:
        raise ValueError(
            f"joint_state.names must contain the six {arm} arm joints"
        )
    if not isinstance(positions, list) or len(positions) != len(names):
        raise ValueError(
            "joint_state.positions_rad must match joint_state.names"
        )
    positions = tuple(
        _finite_float(value, f"position for {name}")
        for name, value in zip(names, positions)
    )

    tcp_data = document.get("tcp_pose_world")
    if not isinstance(tcp_data, dict):
        raise ValueError("tcp_pose_world must be a mapping")
    position = tcp_data.get("position_m")
    orientation = tcp_data.get("orientation_xyzw")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise ValueError(
            "TCP position_m and orientation_xyzw must be mappings"
        )
    tcp = Pose()
    for field in ("x", "y", "z"):
        setattr(
            tcp.position,
            field,
            _finite_float(position.get(field), f"TCP position {field}"),
        )
    for field in ("x", "y", "z", "w"):
        setattr(
            tcp.orientation,
            field,
            _finite_float(orientation.get(field), f"TCP orientation {field}"),
        )
    norm = math.sqrt(
        tcp.orientation.x ** 2
        + tcp.orientation.y ** 2
        + tcp.orientation.z ** 2
        + tcp.orientation.w ** 2
    )
    if norm < 1e-9:
        raise ValueError("TCP orientation quaternion must be non-zero")
    return planning_group, tuple(names), positions, tcp


class WeldActionNode(Node):
    """ROS interface used by the editable weld-path GUI."""

    def __init__(self, ui):
        super().__init__("weld_action_gui")
        self.ui = ui
        self.declare_parameter("expected_execute_motion", True)
        self.declare_parameter("robot_feedback_timeout", 5.0)
        self.declare_parameter("left_robot_ip", "192.168.1.11")
        self.declare_parameter("right_robot_ip", "192.168.1.10")
        self.client = ActionClient(self, CartesianPath, "cartesian_path")
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )
        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )
        self.trajectory_clients = {
            "left": ActionClient(
                self,
                FollowJointTrajectory,
                "/left_manipulator_controller/follow_joint_trajectory",
            ),
            "right": ActionClient(
                self,
                FollowJointTrajectory,
                "/right_manipulator_controller/follow_joint_trajectory",
            ),
        }
        self.digital_output_client = self.create_client(
            SetDigitalOutput,
            "/right_rbpodo_hardware/set_digital_output",
        )
        self.controller_list_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_handle = None
        self.initial_planned_trajectory = None
        self.request_execution = False
        self.execute_motion_enabled = self.get_parameter(
            "expected_execute_motion"
        ).value
        self.expect_robot_feedback = {"left": True, "right": True}
        self.robot_feedback_seen = {"left": False, "right": False}
        self.robot_ready_reported = {"left": False, "right": False}
        self.controller_states = {"left": None, "right": None}
        self.controller_state_future = None
        self.latest_joint_positions = {}
        self.last_robot_feedback_at = {"left": None, "right": None}
        startup_deadline = time.monotonic() + 90.0
        self.connection_deadline = {
            "left": startup_deadline,
            "right": startup_deadline,
        }
        self.rviz_goal_refresh_pending = True
        marker_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            "weld_path_markers",
            marker_qos,
        )
        self.pose_publisher = self.create_publisher(
            PoseArray,
            "weld_6d_poses",
            marker_qos,
        )
        self.display_trajectory_publisher = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            marker_qos,
        )
        self.rviz_goal_refresh_publisher = self.create_publisher(
            Empty,
            "/rviz/moveit/update_goal_state",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.create_subscription(
            WelderStatus,
            "/h600/status",
            self._welder_status,
            10,
        )
        self.create_timer(0.5, self._check_robot_feedback)
        self.create_subscription(
            SystemState,
            "/right_rbpodo_hardware/system_state",
            lambda message: self._system_state(message, "right"),
            10,
        )
        self.create_subscription(
            SystemState,
            "/left_rbpodo_hardware/system_state",
            lambda message: self._system_state(message, "left"),
            10,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state,
            10,
        )
        self.ui.post(
            self.ui.set_execution_configuration,
            self.get_parameter("expected_execute_motion").value,
            self.get_parameter("left_robot_ip").value,
            self.get_parameter("right_robot_ip").value,
        )

    def _system_state(self, message, arm):
        self.ui.post(
            self.ui.update_touch_input,
            arm,
            bool(message.digital_in[TOUCH_INPUT_PORT]),
        )
        if arm == "right":
            self.ui.post(
                self.ui.update_control_box_io,
                tuple(message.digital_in),
                tuple(message.digital_out),
            )
        if not self.expect_robot_feedback[arm]:
            return
        self.last_robot_feedback_at[arm] = time.monotonic()
        self.robot_feedback_seen[arm] = True

    def _joint_state(self, message):
        """Use complete finite measured arm states as connection feedback."""
        positions = dict(zip(message.name, message.position))
        self.latest_joint_positions.update(
            {
                name: position
                for name, position in positions.items()
                if math.isfinite(position)
            }
        )
        received_at = time.monotonic()
        for arm, expected_names in ARM_JOINT_NAMES.items():
            if expected_names.issubset(positions) and all(
                math.isfinite(positions[name]) for name in expected_names
            ):
                self.last_robot_feedback_at[arm] = received_at
                self.robot_feedback_seen[arm] = True

    def capture_touch_pose(self, planning_group, source):
        try:
            pose = self._current_tcp_pose(planning_group)
        except TransformException as error:
            self.ui.post(
                self.ui.error,
                f"Touch TCP capture failed: {error}",
            )
            return
        self.ui.post(
            self.ui.apply_touch_capture,
            pose,
            planning_group,
            source,
        )

    def set_digital_output(self, port, value):
        if not self.digital_output_client.wait_for_service(timeout_sec=2.0):
            self.ui.post(
                self.ui.digital_output_result,
                port,
                False,
                "/right_rbpodo_hardware/set_digital_output unavailable",
            )
            return
        request = SetDigitalOutput.Request()
        request.port = port
        request.value = value
        future = self.digital_output_client.call_async(request)
        future.add_done_callback(
            lambda result: self._digital_output_result(result, port)
        )

    def _digital_output_result(self, future, port):
        try:
            response = future.result()
            self.ui.post(
                self.ui.digital_output_result,
                port,
                response.success,
                response.message,
            )
        except Exception as error:
            self.ui.post(
                self.ui.digital_output_result,
                port,
                False,
                str(error),
            )

    def _check_robot_feedback(self):
        self._request_controller_states()
        feedback_timeout = max(
            1.0,
            float(self.get_parameter("robot_feedback_timeout").value),
        )
        move_group_ready = self.move_group_client.server_is_ready()
        for arm in ("left", "right"):
            last_feedback = self.last_robot_feedback_at[arm]
            deadline = self.connection_deadline[arm]
            feedback_is_fresh = (
                last_feedback is not None
                and time.monotonic() - last_feedback <= feedback_timeout
            )
            controller_ready = (
                not self.execute_motion_enabled
                or (
                    self.controller_states[arm] == "active"
                    and self.trajectory_clients[arm].server_is_ready()
                )
            )
            stack_ready = (
                self.robot_feedback_seen[arm]
                and feedback_is_fresh
                and move_group_ready
                and controller_ready
            )
            if stack_ready and not self.robot_ready_reported[arm]:
                self.robot_ready_reported[arm] = True
                self.connection_deadline[arm] = None
                self.ui.post(self.ui.robot_feedback_connected, arm)
                continue
            if self.robot_ready_reported[arm] and not stack_ready:
                self.robot_ready_reported[arm] = False
                self.rviz_goal_refresh_pending = True
                detail = self._not_ready_detail(
                    arm,
                    feedback_is_fresh,
                    move_group_ready,
                    controller_ready,
                )
                self.ui.post(self.ui.robot_feedback_lost, arm, detail)
                self.get_logger().warning(
                    f"{arm.upper()} CONNECTION X · {detail}"
                )
                continue
            if (
                self.expect_robot_feedback[arm]
                and not self.robot_ready_reported[arm]
                and deadline is not None
                and time.monotonic() > deadline
            ):
                self.connection_deadline[arm] = None
                detail = (
                    "measured joint feedback received, but MoveIt/controllers did "
                    "not become ready"
                    if self.robot_feedback_seen[arm]
                    else "no fresh complete measured joint state received"
                )
                self.ui.post(self.ui.robot_feedback_lost, arm)
                self.get_logger().error(
                    f"{arm.upper()} CONNECTION X · {detail}"
                )
                continue
            if (
                not self.expect_robot_feedback[arm]
                or not self.robot_feedback_seen[arm]
                or last_feedback is None
                or feedback_is_fresh
            ):
                continue
            self.robot_feedback_seen[arm] = False
            if self.robot_ready_reported[arm]:
                self.robot_ready_reported[arm] = False
                self.rviz_goal_refresh_pending = True
                self.ui.post(self.ui.robot_feedback_lost, arm)

        expected_real_arms = tuple(
            arm
            for arm in ("left", "right")
            if self.expect_robot_feedback[arm]
        )
        all_expected_arms_ready = (
            bool(expected_real_arms)
            and move_group_ready
            and all(
                self.robot_ready_reported[arm]
                for arm in expected_real_arms
            )
        )
        if (
            self.rviz_goal_refresh_pending
            and all_expected_arms_ready
            and self.rviz_goal_refresh_publisher.get_subscription_count() > 0
        ):
            # This invokes RViz's own "Goal State = <current>" callback. It
            # changes only the orange query state and sends no robot command.
            self.rviz_goal_refresh_publisher.publish(Empty())
            self.rviz_goal_refresh_pending = False
            self.get_logger().info(
                "Requested RViz Goal State refresh from current state"
            )

    def _request_controller_states(self):
        if not self.controller_list_client.service_is_ready():
            return
        if (
            self.controller_state_future is not None
            and not self.controller_state_future.done()
        ):
            return
        self.controller_state_future = self.controller_list_client.call_async(
            ListControllers.Request()
        )
        self.controller_state_future.add_done_callback(
            self._controller_states_received
        )

    def _controller_states_received(self, future):
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(
                f"Failed to read controller states: {error}"
            )
            return
        states = {
            controller.name: controller.state
            for controller in response.controller
        }
        for arm in ("left", "right"):
            name = f"{arm}_manipulator_controller"
            self.controller_states[arm] = states.get(name)

    def _not_ready_detail(
        self,
        arm,
        feedback_is_fresh,
        move_group_ready,
        controller_ready,
    ):
        if not feedback_is_fresh:
            return "measured joint feedback timeout"
        if not move_group_ready:
            return "MoveGroup action unavailable"
        if self.controller_states[arm] != "active":
            return (
                f"{arm}_manipulator_controller state="
                f"{self.controller_states[arm] or 'unknown'}"
            )
        if not controller_ready:
            return "FollowJointTrajectory action unavailable"
        return "stack not ready"

    def _current_tcp_pose(self, planning_group):
        transform = self.tf_buffer.lookup_transform(
            "World",
            tip_link_for_group(planning_group),
            rclpy.time.Time(),
            timeout=Duration(seconds=1.0),
        )
        source = transform.transform
        pose = Pose()
        pose.position.x = source.translation.x
        pose.position.y = source.translation.y
        pose.position.z = source.translation.z
        pose.orientation = source.rotation
        return pose

    def publish_points(self, points, visible=True):
        displayed_points = points if visible else []
        markers, pose_array = make_weld_visualization(
            displayed_points,
            "World",
            self.get_clock().now().to_msg(),
        )
        self.marker_publisher.publish(markers)
        self.pose_publisher.publish(pose_array)

    def acquire_points(
        self,
        reference,
        axis,
        distance,
        count,
        explicit_position,
        visible,
        planning_group,
    ):
        try:
            tcp = self._current_tcp_pose(planning_group)
            if explicit_position is not None:
                (
                    tcp.position.x,
                    tcp.position.y,
                    tcp.position.z,
                ) = explicit_position
            points = straight_waypoints(
                tcp,
                distance,
                count,
                axis,
                reference,
            )
        except (TransformException, ValueError) as error:
            self.ui.post(
                self.ui.error,
                f"Straight path acquisition failed: {error}",
            )
            return
        self.publish_points(points, visible)
        self.ui.post(self.ui.set_new_points, points, "straight")
        start_description = (
            "current TCP"
            if explicit_position is None
            else (
                "World XYZ "
                f"({explicit_position[0]:.3f}, "
                f"{explicit_position[1]:.3f}, "
                f"{explicit_position[2]:.3f})"
            )
        )
        self.ui.post(
            self.ui.log,
            f"Acquired straight seam · start={start_description} · "
            f"{reference} {axis.upper()} · "
            f"distance={distance * 1000.0:.1f} mm · {count} poses",
        )

    def generate_circle(
        self,
        normal_axis,
        radius,
        count,
        closed,
        face_center,
        visible,
        planning_group,
    ):
        try:
            tcp = self._current_tcp_pose(planning_group)
            points = circle_waypoints(
                tcp,
                radius,
                count,
                closed,
                face_center,
                normal_axis,
            )
        except (TransformException, ValueError) as error:
            self.ui.post(self.ui.error, f"Circle generation failed: {error}")
            return
        self.publish_points(points, visible)
        self.ui.post(self.ui.set_new_points, points, "circle")
        description = (
            f"{count} unique points"
            f"{' + closing point' if closed else ''}, radius={radius:.3f} m"
        )
        orientation = (
            "TCP +Z faces center"
            if face_center
            else "fixed TCP orientation"
        )
        self.ui.post(
            self.ui.log,
            f"Generated World-{normal_axis.upper()} normal circle · "
            f"{description} · {orientation}",
        )

    def capture_initial_state(self, planning_group):
        arm = "left" if planning_group.startswith("left") else "right"
        joint_names = [
            f"{arm}_manipulator_joint{index}" for index in range(1, 7)
        ]
        try:
            positions = [self.latest_joint_positions[name] for name in joint_names]
            tcp = self._current_tcp_pose(planning_group)
        except KeyError:
            self.ui.post(self.ui.error, "Complete measured joint state is unavailable")
            return
        except TransformException as error:
            self.ui.post(self.ui.error, f"Initial TCP capture failed: {error}")
            return
        self.ui.post(
            self.ui.apply_initial_state,
            planning_group,
            joint_names,
            positions,
            tcp,
        )

    def plan_initial_state(
        self,
        planning_group,
        joint_names,
        positions,
        velocity_scale,
    ):
        self.initial_planned_trajectory = None
        try:
            current_positions = [
                self.latest_joint_positions[name] for name in joint_names
            ]
        except KeyError:
            self.ui.post(
                self.ui.error,
                "Complete measured joint state is unavailable",
            )
            return
        maximum_delta = max(
            abs(current - target)
            for current, target in zip(current_positions, positions)
        )
        if maximum_delta <= 0.002:
            self.ui.post(
                self.ui.pipeline_result,
                "Already at captured initial position · no plan required",
            )
            return
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(self.ui.error, "MoveGroup action server unavailable")
            return
        goal = MoveGroup.Goal()
        goal.request.group_name = planning_group
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.start_state.is_diff = True
        goal.request.max_velocity_scaling_factor = velocity_scale
        goal.request.max_acceleration_scaling_factor = velocity_scale
        constraints = Constraints()
        for name, position in zip(joint_names, positions):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = position
            constraint.tolerance_above = 0.001
            constraint.tolerance_below = 0.001
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        self.ui.post(
            self.ui.pipeline_waiting,
            "Planning to captured initial joint angles",
        )
        future = self.move_group_client.send_goal_async(goal)
        target_positions = tuple(positions)
        future.add_done_callback(
            lambda result: self._initial_plan_goal_response(
                result,
                planning_group,
                target_positions,
                velocity_scale,
            )
        )

    def _initial_plan_goal_response(
        self,
        future,
        planning_group,
        target_positions,
        velocity_scale,
    ):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Initial-position plan failed: {error}",
            )
            return
        if not goal_handle.accepted:
            self.ui.post(self.ui.error, "Initial-position plan was rejected")
            return
        goal_handle.get_result_async().add_done_callback(
            lambda result: self._initial_plan_result(
                result,
                planning_group,
                target_positions,
                velocity_scale,
            )
        )

    def _initial_plan_result(
        self,
        future,
        planning_group,
        target_positions,
        velocity_scale,
    ):
        try:
            result = future.result().result
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Initial-position plan failed: {error}",
            )
            return
        if result.error_code.val != 1:
            self.ui.post(
                self.ui.error,
                "Initial-position plan failed "
                f"(MoveIt code {result.error_code.val})",
            )
            return
        display = DisplayTrajectory()
        display.trajectory_start = result.trajectory_start
        display.trajectory.append(result.planned_trajectory)
        self.initial_planned_trajectory = copy.deepcopy(
            result.planned_trajectory
        )
        self.display_trajectory_publisher.publish(display)
        self.ui.post(
            self.ui.initial_position_plan_ready,
            planning_group,
            target_positions,
            velocity_scale,
            "Initial-position plan shown in RViz · ready to execute",
        )

    def execute_initial_plan(self):
        if not self.execute_motion_enabled:
            self.ui.post(
                self.ui.error,
                "Initial-position execution is disabled by launch "
                "configuration",
            )
            return
        trajectory = self.initial_planned_trajectory
        if trajectory is None:
            self.ui.post(self.ui.error, "Plan the initial position first")
            return
        if not self.execute_trajectory_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(
                self.ui.error,
                "ExecuteTrajectory action server unavailable",
            )
            return
        self.initial_planned_trajectory = None
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        self.ui.post(
            self.ui.pipeline_waiting,
            "Executing the approved initial-position plan",
        )
        future = self.execute_trajectory_client.send_goal_async(goal)
        future.add_done_callback(self._initial_execute_goal_response)

    def _initial_execute_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Initial-position execution failed: {error}",
            )
            return
        if not goal_handle.accepted:
            self.ui.post(self.ui.error, "Initial-position execution rejected")
            return
        goal_handle.get_result_async().add_done_callback(
            self._initial_execute_result
        )

    def _initial_execute_result(self, future):
        try:
            result = future.result().result
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Initial-position execution failed: {error}",
            )
            return
        if result.error_code.val != 1:
            self.ui.post(
                self.ui.error,
                "Initial-position execution failed "
                f"(MoveIt code {result.error_code.val})",
            )
            return
        self.ui.post(
            self.ui.initial_position_execution_finished,
            "Robot reached the captured initial position",
        )

    def generate_weave(
        self,
        source_points,
        amplitude,
        cycles,
        samples_per_cycle,
        transverse_axis,
        visible,
    ):
        try:
            points = weaving_from_path(
                source_points,
                amplitude,
                cycles,
                samples_per_cycle,
                transverse_axis,
            )
        except ValueError as error:
            self.ui.post(self.ui.error, f"Weave generation failed: {error}")
            return
        self.publish_points(points, visible)
        self.ui.post(self.ui.set_new_points, points, "weave")
        self.ui.post(
            self.ui.log,
            f"Applied weave to taught seam · amplitude=±{amplitude:.3f} m, "
            f"cycles={cycles}, axis={transverse_axis}",
        )

    def capture_tcp(self, replace_index, visible, planning_group):
        try:
            pose = self._current_tcp_pose(planning_group)
        except TransformException as error:
            self.ui.post(self.ui.error, f"TCP capture failed: {error}")
            return
        self.ui.post(
            self.ui.apply_captured_tcp,
            pose,
            replace_index,
            visible,
        )

    def capture_linear_tcp(self, endpoint_index, planning_group):
        try:
            pose = self._current_tcp_pose(planning_group)
        except TransformException as error:
            self.ui.post(self.ui.error, f"TCP capture failed: {error}")
            return
        self.ui.post(self.ui.apply_linear_tcp, endpoint_index, pose)

    def generate_tcp_line(self, start, end, count, visible):
        try:
            points = linear_pose_waypoints(start, end, count)
        except ValueError as error:
            self.ui.post(self.ui.error, f"TCP line generation failed: {error}")
            return
        distance = math.sqrt(
            (end.position.x - start.position.x) ** 2
            + (end.position.y - start.position.y) ** 2
            + (end.position.z - start.position.z) ** 2
        )
        self.publish_points(points, visible)
        self.ui.post(self.ui.set_new_points, points, "tcp_line")
        self.ui.post(
            self.ui.log,
            f"Generated endpoint-to-endpoint linear 6D path · "
            f"distance={distance * 1000.0:.1f} mm · {count} poses",
        )

    def send(
        self,
        points,
        velocity_scale,
        interpolation_step,
        visualize_path,
        enable_arc,
        current_raw,
        voltage_raw,
        v_offset_raw,
        preflow_seconds,
        postflow_seconds,
        require_welding_feedback,
        execute_requested,
        reuse_approved_plan,
        planning_group,
    ):
        if not points:
            self.ui.post(self.ui.error, "Create weld points first")
            return
        if not self.client.wait_for_server(timeout_sec=3.0):
            self.ui.post(
                self.ui.error,
                "cartesian_path action server unavailable",
            )
            return
        goal = CartesianPath.Goal()
        goal.planning_group = planning_group
        goal.interpolation_step = interpolation_step
        goal.velocity_scale = velocity_scale
        goal.execute_requested = execute_requested
        goal.reuse_approved_plan = reuse_approved_plan
        goal.visualize_path = visualize_path
        goal.enable_arc = enable_arc
        goal.weld_current_raw = current_raw
        goal.weld_voltage_raw = voltage_raw
        goal.weld_v_offset_raw = v_offset_raw
        goal.weld_preflow_seconds = preflow_seconds
        goal.weld_postflow_seconds = postflow_seconds
        goal.require_welding_feedback = require_welding_feedback
        goal.waypoints = points
        self.request_execution = execute_requested
        self.ui.post(
            self.ui.begin,
            velocity_scale,
            execute_requested,
        )
        future = self.client.send_goal_async(
            goal,
            feedback_callback=self.feedback,
        )
        future.add_done_callback(self.goal_response)

    def feedback(self, message):
        feedback = message.feedback
        self.ui.post(
            self.ui.progress,
            feedback.progress,
            feedback.waypoint_index,
            feedback.current_pose,
            feedback.phase,
        )

    def goal_response(self, future):
        try:
            self.goal_handle = future.result()
        except Exception as error:
            self.ui.post(self.ui.error, str(error))
            return
        if not self.goal_handle.accepted:
            self.ui.post(self.ui.error, "Action goal rejected")
            return
        operation = (
            "approved trajectory execution"
            if self.request_execution
            else "MoveIt plan preview"
        )
        self.ui.post(self.ui.log, f"Action accepted · {operation}")
        result = self.goal_handle.get_result_async()
        result.add_done_callback(self.result)

    def result(self, future):
        result = future.result().result
        if result.success:
            self.ui.post(
                self.ui.finish,
                f"SUCCESS · {len(result.sampled_path)} samples · "
                f"{result.message}",
                self.request_execution,
            )
        else:
            self.ui.post(self.ui.error, result.message)

    def cancel(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.ui.post(self.ui.log, "Cancel requested")

    def _welder_status(self, message):
        self.ui.post(self.ui.update_welder_status, message)


class WeldActionGui:
    """Tk GUI for acquiring, editing, visualizing, and running weld paths."""

    POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")

    def _create_toggle_section(
        self,
        parent,
        key,
        title,
        expanded=False,
    ):
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, pady=2)
        button = ttk.Button(
            container,
            command=lambda selected=key: self.toggle_motion_section(selected),
        )
        button.pack(fill=tk.X)
        body = ttk.Frame(container, padding=(8, 5))
        self.motion_sections[key] = {
            "body": body,
            "button": button,
            "title": title,
            "expanded": bool(expanded),
        }
        if expanded:
            body.pack(fill=tk.X)
        self._refresh_motion_section_button(key)
        return body

    def _refresh_motion_section_button(self, key):
        section = self.motion_sections[key]
        marker = "▼" if section["expanded"] else "▶"
        section["button"].configure(
            text=f"{marker}  {section['title']}",
        )

    def toggle_motion_section(self, key):
        section = self.motion_sections[key]
        section["expanded"] = not section["expanded"]
        if section["expanded"]:
            section["body"].pack(fill=tk.X)
        else:
            section["body"].pack_forget()
        self._refresh_motion_section_button(key)
        self.root.after_idle(self._update_scroll_region)

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Editable Cartesian Action")
        self.root.geometry("1240x940")
        self._closing = False
        self._ui_queue = queue.SimpleQueue()
        self.points = []
        self.weave_source = []
        self.weave_base_paths = {"linear": [], "circle": []}
        self.path_kind = "empty"
        self.execution_allowed = False
        self.robot_connected = {"left": False, "right": False}
        self.plan_approved = False
        self.linear_tcp_endpoints = [None, None]
        self.initial_joint_state = None
        self.initial_plan_ready = False
        self.pose_variables = {
            name: tk.StringVar(value="0.0") for name in self.POSE_FIELDS
        }
        self.radius_mm = tk.DoubleVar(value=20.0)
        self.circle_count = tk.IntVar(value=16)
        self.close_circle = tk.BooleanVar(value=True)
        self.circle_face_center = tk.BooleanVar(value=True)
        self.circle_axis = tk.StringVar(value="X")
        self.nudge_mm = tk.DoubleVar(value=5.0)
        self.velocity_percent = tk.DoubleVar(value=20.0)
        self.interpolation_step_mm = tk.DoubleVar(value=5.0)
        self.show_path = tk.BooleanVar(value=True)
        self.weave_amplitude_mm = tk.DoubleVar(value=3.0)
        self.weave_cycles = tk.IntVar(value=4)
        self.weave_samples = tk.IntVar(value=8)
        self.weave_axis = tk.StringVar(value="tool_y")
        self.weave_base = tk.StringVar(value="linear")
        self.straight_reference = tk.StringVar(value="world")
        self.straight_axis = tk.StringVar(value="+X")
        self.straight_start_mode = tk.StringVar(value="Current TCP")
        self.straight_start_x = tk.DoubleVar(value=0.0)
        self.straight_start_y = tk.DoubleVar(value=0.0)
        self.straight_start_z = tk.DoubleVar(value=0.0)
        self.straight_distance_mm = tk.DoubleVar(value=200.0)
        self.straight_count = tk.IntVar(value=5)
        self.tcp_line_count = tk.IntVar(value=10)
        self.tcp_line_direction = tk.StringVar(value="TCP 1 → TCP 2")
        self.planning_group = tk.StringVar(value="right_manipulator")
        self.enable_arc = tk.BooleanVar(value=False)
        self.require_welding_feedback = tk.BooleanVar(value=True)
        self.weld_preflow_seconds = tk.DoubleVar(value=0.5)
        self.weld_postflow_seconds = tk.DoubleVar(value=0.5)
        self.h600_connected = False
        self.last_action_phase = ""
        self.previous_control_box_io = None
        self.touch_input_states = {"left": None, "right": None}
        self.touch_input_rising_edges = {"left": 0, "right": 0}
        self.last_touch_pose = None
        self.motion_sections = {}
        self.control_box_io_labels = {}
        self.pending_do_ports = set()
        self.unlock_all_do_ports = tk.BooleanVar(value=False)
        self.weld_current_raw = tk.IntVar(value=0)
        self.weld_voltage_raw = tk.IntVar(value=0)
        self.weld_v_offset_raw = tk.IntVar(value=0)
        self.robot_ips = {
            "left": "192.168.1.11",
            "right": "192.168.1.10",
        }

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 18, "bold"))
        style.configure("Step.TLabel", font=("Sans", 11, "bold"))

        scroll_container = ttk.Frame(self.root)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        self.content_canvas = tk.Canvas(
            scroll_container,
            highlightthickness=0,
        )
        content_scrollbar = ttk.Scrollbar(
            scroll_container,
            orient=tk.VERTICAL,
            command=self.content_canvas.yview,
        )
        self.content_canvas.configure(yscrollcommand=content_scrollbar.set)
        content_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.content_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        outer = ttk.Frame(self.content_canvas, padding=16)
        self.content_window = self.content_canvas.create_window(
            (0, 0),
            window=outer,
            anchor=tk.NW,
        )
        outer.bind("<Configure>", self._update_scroll_region)
        self.content_canvas.bind("<Configure>", self._resize_scroll_content)
        self.root.bind_all("<MouseWheel>", self._scroll_content)
        ttk.Label(
            outer,
            text="Welding Interface",
            style="Title.TLabel",
        ).pack(anchor=tk.W)

        arm_selection = ttk.Frame(outer)
        arm_selection.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(
            arm_selection,
            text="Cartesian arm:",
            style="Step.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Combobox(
            arm_selection,
            textvariable=self.planning_group,
            values=("right_manipulator", "left_manipulator"),
            state="readonly",
            width=22,
        ).pack(side=tk.LEFT)
        self.planning_group.trace_add("write", self.arm_changed)

        motion_tests = self._create_toggle_section(
            outer, "motion_test", "Motion Test", expanded=True
        )
        straight = ttk.Frame(motion_tests)
        straight.pack(fill=tk.X, pady=2)
        ttk.Button(
            straight,
            text="Generate linear path",
            command=self.acquire,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(straight, text="reference").pack(side=tk.LEFT)
        ttk.Combobox(
            straight,
            textvariable=self.straight_reference,
            values=("world", "tool"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 7))
        ttk.Label(straight, text="axis").pack(side=tk.LEFT)
        ttk.Combobox(
            straight,
            textvariable=self.straight_axis,
            values=("+X", "-X", "+Y", "-Y", "+Z", "-Z"),
            state="readonly",
            width=4,
        ).pack(side=tk.LEFT, padx=(3, 7))
        ttk.Label(straight, text="distance mm").pack(side=tk.LEFT)
        ttk.Spinbox(
            straight,
            from_=0.1,
            to=5000,
            increment=1,
            textvariable=self.straight_distance_mm,
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 6))
        ttk.Label(straight, text="points").pack(side=tk.LEFT)
        ttk.Spinbox(
            straight,
            from_=2,
            to=200,
            increment=1,
            textvariable=self.straight_count,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 0))

        controls = ttk.Frame(motion_tests)
        controls.pack(fill=tk.X, pady=2)
        ttk.Button(
            controls,
            text="Generate circle",
            command=self.generate_circle,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(controls, text="axis").pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=self.circle_axis,
            values=("X", "Y", "Z"),
            state="readonly",
            width=3,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(controls, text="radius (mm)").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls,
            from_=1,
            to=200,
            increment=1,
            textvariable=self.radius_mm,
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(controls, text="unique points").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls,
            from_=4,
            to=200,
            increment=1,
            textvariable=self.circle_count,
            width=5,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Checkbutton(
            controls,
            text="close path",
            variable=self.close_circle,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            controls,
            text="TCP +Z faces center",
            variable=self.circle_face_center,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            motion_tests,
            text="show planned path",
            variable=self.show_path,
            command=self.toggle_path_visibility,
        ).pack(anchor=tk.W, pady=(3, 0))

        welding_tests = self._create_toggle_section(
            outer, "welding_test", "Welding Test", expanded=True
        )
        tcp_line = ttk.Frame(welding_tests)
        tcp_line.pack(fill=tk.X, pady=2)
        ttk.Button(
            tcp_line,
            text="Capture TCP 1",
            command=lambda: self.capture_linear_tcp(0),
        ).pack(side=tk.LEFT, padx=(0, 5))
        self.tcp_1_status = ttk.Label(tcp_line, text="not captured")
        self.tcp_1_status.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(
            tcp_line,
            text="Capture TCP 2",
            command=lambda: self.capture_linear_tcp(1),
        ).pack(side=tk.LEFT, padx=(0, 5))
        self.tcp_2_status = ttk.Label(tcp_line, text="not captured")
        self.tcp_2_status.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(tcp_line, text="points").pack(side=tk.LEFT)
        ttk.Spinbox(
            tcp_line,
            from_=2,
            to=200,
            increment=1,
            textvariable=self.tcp_line_count,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Combobox(
            tcp_line,
            textvariable=self.tcp_line_direction,
            values=("TCP 1 → TCP 2", "TCP 2 → TCP 1"),
            state="readonly",
            width=13,
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.generate_tcp_line_button = ttk.Button(
            tcp_line,
            text="Generate linear path",
            command=self.acquire_two_tcp,
            state=tk.DISABLED,
        )
        self.generate_tcp_line_button.pack(side=tk.LEFT)

        teaching = ttk.LabelFrame(outer, text="TCP Teaching")
        teaching.pack(fill=tk.X, pady=(7, 0))
        ttk.Button(
            teaching,
            text="Get current TCP and set initial position",
            command=self.capture_initial_state,
        ).pack(side=tk.LEFT, padx=3)
        self.plan_initial_button = ttk.Button(
            teaching,
            text="1 · Plan initial position",
            command=self.plan_initial_state,
            state=tk.DISABLED,
        )
        self.plan_initial_button.pack(side=tk.LEFT, padx=3)
        self.execute_initial_button = ttk.Button(
            teaching,
            text="2 · Execute initial plan",
            command=self.execute_initial_plan,
            state=tk.DISABLED,
        )
        self.execute_initial_button.pack(side=tk.LEFT, padx=3)
        ttk.Button(
            teaching,
            text="Load from YAML",
            command=self.load_initial_state,
        ).pack(side=tk.LEFT, padx=3)
        self.initial_state_status = ttk.Label(teaching, text="not captured")
        self.initial_state_status.pack(side=tk.LEFT, padx=(12, 0))
        self.path_summary = ttk.Label(teaching, text="empty path")

        weaving = ttk.Frame(welding_tests)
        weaving.pack(fill=tk.X, pady=2)
        ttk.Button(
            weaving,
            text="Generate weave path",
            command=self.generate_weave,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(weaving, text="base").pack(side=tk.LEFT)
        ttk.Combobox(
            weaving,
            textvariable=self.weave_base,
            values=("linear", "circle"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 8))
        for label, variable, start, end in (
            ("one-side amplitude mm", self.weave_amplitude_mm, 0.1, 50),
            ("weave count", self.weave_cycles, 1, 30),
            ("samples/cycle", self.weave_samples, 4, 30),
        ):
            ttk.Label(weaving, text=label).pack(side=tk.LEFT)
            ttk.Spinbox(
                weaving,
                from_=start,
                to=end,
                textvariable=variable,
                width=6,
            ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(weaving, text="transverse axis").pack(side=tk.LEFT)
        ttk.Combobox(
            weaving,
            textvariable=self.weave_axis,
            values=(
                "tool_x",
                "tool_y",
                "tool_z",
                "world_x",
                "world_y",
                "world_z",
            ),
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(3, 8))
        self.weave_summary = ttk.Label(
            weaving,
            text="Apply after teaching a seam",
        )
        self.weave_summary.pack(side=tk.LEFT, padx=(8, 0))

        planned_path = self._create_toggle_section(
            outer, "planned_path", "Planned Path · World frame"
        )
        columns = ("id",) + self.POSE_FIELDS
        self.table = ttk.Treeview(
            planned_path,
            columns=columns,
            show="headings",
            height=4,
            selectmode="browse",
        )
        for name in columns:
            self.table.heading(name, text=name.upper())
            self.table.column(
                name,
                width=48 if name == "id" else 105,
                anchor=tk.CENTER,
            )
        self.table.pack(fill=tk.X)
        ttk.Button(
            planned_path,
            text="Delete All",
            command=self.clear_path,
        ).pack(anchor=tk.E, pady=(5, 0))

        robot_status = ttk.LabelFrame(outer, text="Robot connection")
        robot_status.pack(fill=tk.X, pady=(10, 0))
        self.robot_connection_labels = {}
        for arm in ("left", "right"):
            label = tk.Label(
                robot_status,
                text=f"Connect {arm.upper()} (IP): X",
                width=32,
                relief=tk.SOLID,
                borderwidth=1,
                bg="#fce8e6",
                fg="#b3261e",
                font=("Sans", 11, "bold"),
            )
            label.pack(side=tk.LEFT, padx=6, pady=6)
            self.robot_connection_labels[arm] = label

        head_label = tk.Label(
            robot_status,
            text="Connect HEAD (future): –",
            width=26,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#eeeeee",
        )
        head_label.pack(side=tk.LEFT, padx=6, pady=6)

        io_monitor = self._create_toggle_section(
            outer,
            "digital_io",
            "Digital I/O · DI0 = TOUCH · ports 0..15",
        )
        for io_row, kind in enumerate(("DI", "DO")):
            ttk.Label(
                io_monitor,
                text=kind,
                font=("Sans", 10, "bold"),
            ).grid(row=io_row, column=0, padx=(6, 4), pady=3)
            for port in range(16):
                candidate = port in OBSERVED_H600_IO_CANDIDATES
                label = tk.Label(
                    io_monitor,
                    text=f"{port:02d}\n–",
                    width=4,
                    relief=tk.SOLID,
                    borderwidth=2 if candidate else 1,
                    bg="#dbeafe" if candidate else "#eeeeee",
                    font=("Monospace", 9, "bold" if candidate else "normal"),
                )
                label.grid(row=io_row, column=port + 1, padx=2, pady=3)
                self.control_box_io_labels[(kind, port)] = label
                if kind == "DO":
                    label.configure(cursor="hand2")
                    label.bind(
                        "<Button-1>",
                        lambda _event, selected=port: (
                            self.request_do_toggle(selected)
                        ),
                    )
        self.control_box_io_status = ttk.Label(
            io_monitor,
            text="Waiting for /right_rbpodo_hardware/system_state",
        )
        self.control_box_io_status.grid(
            row=2,
            column=0,
            columnspan=17,
            sticky=tk.W,
            padx=6,
            pady=(2, 5),
        )
        ttk.Checkbutton(
            io_monitor,
            text="Unlock clicking non-candidate DO ports",
            variable=self.unlock_all_do_ports,
            command=self.confirm_all_do_unlock,
        ).grid(
            row=3,
            column=0,
            columnspan=12,
            sticky=tk.W,
            padx=6,
            pady=(0, 5),
        )
        ttk.Button(
            io_monitor,
            text="Candidate DO all OFF",
            command=self.candidate_outputs_off,
        ).grid(
            row=3,
            column=12,
            columnspan=5,
            sticky=tk.E,
            padx=6,
            pady=(0, 5),
        )

        execution = ttk.Frame(outer)
        execution.pack(fill=tk.X, pady=(12, 0))
        self.plan_button = ttk.Button(
            execution,
            text="1 · Plan Preview",
            command=self.plan_preview,
            state=tk.DISABLED,
        )
        self.plan_button.pack(side=tk.LEFT, padx=(0, 8))
        self.execute_button = ttk.Button(
            execution,
            text="2 · Execute Approved Plan",
            command=self.execute_approved,
            state=tk.DISABLED,
        )
        self.execute_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            execution,
            text="Cancel",
            command=self.cancel,
        ).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(
            execution,
            text="velocity scale",
        ).pack(side=tk.LEFT)
        ttk.Scale(
            execution,
            from_=1,
            to=100,
            variable=self.velocity_percent,
            command=self.update_speed_label,
            length=220,
        ).pack(side=tk.LEFT, padx=(7, 5))
        self.speed_label = ttk.Label(execution, text="20%")
        self.speed_label.pack(side=tk.LEFT)

        planning_settings = ttk.Frame(outer)
        planning_settings.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(planning_settings, text="Cartesian interpolation step mm").pack(
            side=tk.LEFT
        )
        ttk.Spinbox(
            planning_settings,
            from_=0.5,
            to=20.0,
            increment=0.5,
            textvariable=self.interpolation_step_mm,
            width=6,
            command=self.invalidate_approved_plan,
        ).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(
            planning_settings,
            text="trajectory profile: TOTG + Ruckig jerk smoothing",
        ).pack(side=tk.LEFT)

        ttk.Label(
            outer,
            text="Action feedback",
            style="Step.TLabel",
        ).pack(anchor=tk.W, pady=(12, 5))
        self.bar = ttk.Progressbar(outer, maximum=100)
        self.bar.pack(fill=tk.X)
        self.feedback_label = ttk.Label(
            outer,
            text="waypoint: –    pose: –",
        )
        self.feedback_label.pack(anchor=tk.W, pady=4)

        ttk.Label(
            outer,
            text="Pipeline status",
            style="Step.TLabel",
        ).pack(anchor=tk.W, pady=(8, 5))
        self.pipeline_status = tk.Label(
            outer,
            text="WAITING · ready",
            anchor=tk.W,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#eeeeee",
            font=("Sans", 10, "bold"),
        )
        self.pipeline_status.pack(fill=tk.X, ipady=5)

        self.node = WeldActionNode(self)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
        )
        self.executor_thread.start()
        signal.signal(
            signal.SIGINT,
            lambda _signum, _frame: self.root.after(0, self.close),
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.check_ros)

    def post(self, callback, *args):
        self._ui_queue.put((callback, args))

    def _drain_ui_queue(self):
        while True:
            try:
                callback, args = self._ui_queue.get_nowait()
            except queue.Empty:
                return
            callback(*args)

    def _update_scroll_region(self, _event=None):
        self.content_canvas.configure(
            scrollregion=self.content_canvas.bbox("all")
        )

    def _resize_scroll_content(self, event):
        self.content_canvas.itemconfigure(
            self.content_window,
            width=event.width,
        )

    def _scroll_content(self, event):
        if event.delta:
            self.content_canvas.yview_scroll(
                int(-event.delta / 120),
                "units",
            )

    def arm_changed(self, *_args):
        if not hasattr(self, "node"):
            return
        group = self.planning_group.get()
        if group != "right_manipulator":
            self.enable_arc.set(False)
        self.linear_tcp_endpoints = [None, None]
        self.tcp_1_status.configure(text="not captured")
        self.tcp_2_status.configure(text="not captured")
        self.generate_tcp_line_button.configure(state=tk.DISABLED)
        self.path_kind = "empty"
        self.weave_source = []
        self.weave_base_paths = {"linear": [], "circle": []}
        self.initial_joint_state = None
        self.initial_plan_ready = False
        self.plan_initial_button.configure(state=tk.DISABLED)
        self.execute_initial_button.configure(state=tk.DISABLED)
        self.initial_state_status.configure(text="not captured")
        self.set_points([])
        self.node.publish_points([], self.show_path.get())
        self._refresh_execution_controls()
        self.log(f"Cartesian arm changed to {group} · path cleared")

    def _selected_arm(self):
        return (
            "left"
            if self.planning_group.get() == "left_manipulator"
            else "right"
        )

    def _selected_robot_connected(self):
        return self.robot_connected[self._selected_arm()]

    def _refresh_execution_controls(self):
        selected_arm = self._selected_arm()
        connected = self.robot_connected[selected_arm]
        for arm, value in self.robot_connected.items():
            self.robot_connection_labels[arm].configure(
                text=(
                    f"Connect {arm.upper()} ({self.robot_ips[arm]}): "
                    f"{'O' if value else 'X'}"
                ),
                bg="#e6f4ea" if value else "#fce8e6",
                fg="#137333" if value else "#b3261e",
            )
        self.plan_button.configure(
            state=tk.NORMAL if self.points and connected else tk.DISABLED
        )
        self.execute_button.configure(
            state=(
                tk.NORMAL
                if (
                    self.plan_approved
                    and self.execution_allowed
                    and connected
                )
                else tk.DISABLED
            )
        )
        self._refresh_initial_position_controls()

    def _refresh_initial_position_controls(self):
        if not hasattr(self, "plan_initial_button"):
            return
        can_plan = (
            self.initial_joint_state is not None
            and self._selected_robot_connected()
        )
        self.plan_initial_button.configure(
            state=tk.NORMAL if can_plan else tk.DISABLED
        )
        can_execute = (
            self.initial_plan_ready
            and self.initial_joint_state is not None
            and self.execution_allowed
            and self._selected_robot_connected()
        )
        self.execute_initial_button.configure(
            state=tk.NORMAL if can_execute else tk.DISABLED
        )

    def log(self, text):
        if text.startswith("ERROR"):
            self._set_pipeline_status("ERROR", text.removeprefix("ERROR · "))
        elif text.startswith(("SUCCESS", "RESULT")):
            self._set_pipeline_status("RESULT", text)
        else:
            self._set_pipeline_status("WAITING", text)

    def _set_pipeline_status(self, state, message):
        colors = {
            "WAITING": ("#eeeeee", "#202124"),
            "ERROR": ("#fce8e6", "#b3261e"),
            "RESULT": ("#e6f4ea", "#137333"),
        }
        background, foreground = colors[state]
        self.pipeline_status.configure(
            text=f"{state} · {message}",
            bg=background,
            fg=foreground,
        )

    def pipeline_waiting(self, message):
        self._set_pipeline_status("WAITING", message)

    def pipeline_result(self, message):
        self._set_pipeline_status("RESULT", message)

    def error(self, text):
        self.log(f"ERROR · {text}")
        self.plan_approved = False
        state = (
            tk.NORMAL
            if self.points and self._selected_robot_connected()
            else tk.DISABLED
        )
        self.plan_button.configure(state=state)
        self.execute_button.configure(state=tk.DISABLED)

    @staticmethod
    def _pose_values(pose):
        p, q = pose.position, pose.orientation
        return (p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    def set_points(self, points, selected_index=0):
        self.invalidate_approved_plan()
        self.points = copy.deepcopy(list(points))
        self.table.delete(*self.table.get_children())
        for index, pose in enumerate(self.points, 1):
            values = tuple(f"{value:.5f}" for value in self._pose_values(pose))
            self.table.insert("", tk.END, values=(index,) + values)
        self.plan_button.configure(
            state=(
                tk.NORMAL
                if self.points and self._selected_robot_connected()
                else tk.DISABLED
            ),
        )
        children = self.table.get_children()
        if children:
            selected_index = min(max(selected_index, 0), len(children) - 1)
            self.table.selection_set(children[selected_index])
            self.table.focus(children[selected_index])
            self.table.see(children[selected_index])
        self.path_summary.configure(
            text=f"{self.path_kind} · {len(self.points)} poses"
        )

    def set_new_points(self, points, kind):
        if kind != "weave":
            self.weave_source = copy.deepcopy(list(points))
        if kind == "circle":
            self.weave_base_paths["circle"] = copy.deepcopy(list(points))
        elif kind == "tcp_line":
            self.weave_base_paths["linear"] = copy.deepcopy(list(points))
        self.path_kind = kind
        self.set_points(points)

    def set_execution_configuration(self, execute_motion, left_ip, right_ip):
        self.execution_allowed = execute_motion
        self.robot_ips = {"left": left_ip, "right": right_ip}
        self.robot_connected = {"left": False, "right": False}
        self._refresh_execution_controls()
        self.log(
            f"Connecting LEFT {left_ip} + RIGHT {right_ip} · "
            "waiting for measured feedback and planning readiness"
        )

    def robot_feedback_connected(self, arm):
        self.robot_connected[arm] = True
        self._refresh_execution_controls()
        self.log(
            f"READY · {arm}-arm feedback and MoveIt/controller available"
        )

    def robot_feedback_lost(self, arm, detail="measured joint feedback timeout"):
        self.robot_connected[arm] = False
        self.invalidate_approved_plan()
        if self._selected_arm() == arm:
            self.initial_plan_ready = False
        self._refresh_execution_controls()
        if self._selected_arm() == arm:
            self.plan_button.configure(state=tk.DISABLED)
        self.log(f"ERROR · {arm}-arm unavailable · {detail}")

    def invalidate_approved_plan(self):
        self.plan_approved = False
        if hasattr(self, "execute_button"):
            self.execute_button.configure(state=tk.DISABLED)

    def selected_index(self):
        selection = self.table.selection()
        if not selection:
            return None
        return int(self.table.item(selection[0], "values")[0]) - 1

    def load_selected(self, _event=None):
        index = self.selected_index()
        if index is None:
            return
        for name, value in zip(
            self.POSE_FIELDS,
            self._pose_values(self.points[index]),
        ):
            self.pose_variables[name].set(f"{value:.6f}")

    def publish_edits(self, selected_index):
        if self.path_kind != "weave":
            self.weave_source = copy.deepcopy(self.points)
        self.set_points(self.points, selected_index)
        self.node.publish_points(self.points, self.show_path.get())
        self.log(f"Published edited path · {len(self.points)} poses")

    def toggle_path_visibility(self):
        self.node.publish_points(self.points, self.show_path.get())
        state = "ON" if self.show_path.get() else "OFF"
        self.log(f"Planned path visualization {state}")

    def apply_selected(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        try:
            values = [
                float(self.pose_variables[name].get())
                for name in self.POSE_FIELDS
            ]
        except ValueError:
            self.error("Pose fields must be numeric")
            return
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = values[:3]
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = values[3:]
        if not pose_is_valid(pose):
            self.error("Pose must be finite with a non-zero quaternion")
            return
        self.points[index] = pose
        self.publish_edits(index)

    def duplicate_selected(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        self.points.insert(index + 1, copy.deepcopy(self.points[index]))
        self.publish_edits(index + 1)

    def delete_selected(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        self.points.pop(index)
        self.publish_edits(max(0, index - 1))

    def move_selected(self, offset):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        destination = index + offset
        if destination < 0 or destination >= len(self.points):
            return
        self.points[index], self.points[destination] = (
            self.points[destination],
            self.points[index],
        )
        self.publish_edits(destination)

    def nudge(self, axis, direction):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        try:
            distance = float(self.nudge_mm.get()) * 0.001 * direction
        except (ValueError, tk.TclError):
            self.error("Nudge distance must be numeric")
            return
        position = self.points[index].position
        setattr(position, axis, getattr(position, axis) + distance)
        self.publish_edits(index)

    # Acquire a straight seam from an axis and a World/tool reference frame.
    def acquire(self):
        try:
            reference = self.straight_reference.get()
            direction = self.straight_axis.get()
            axis = direction[-1].lower()
            sign = -1.0 if direction.startswith("-") else 1.0
            distance = (
                float(self.straight_distance_mm.get()) * 0.001 * sign
            )
            count = int(self.straight_count.get())
            explicit_position = None
            if self.straight_start_mode.get() == "World XYZ":
                explicit_position = (
                    float(self.straight_start_x.get()),
                    float(self.straight_start_y.get()),
                    float(self.straight_start_z.get()),
                )
        except (ValueError, tk.TclError):
            self.error("Straight position/distance/count must be numeric")
            return
        self.log(
            f"Reading current {self.planning_group.get()} TCP and generating "
            f"{reference} {direction} straight seam"
        )
        threading.Thread(
            target=self.node.acquire_points,
            args=(
                reference,
                axis,
                distance,
                count,
                explicit_position,
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def generate_circle(self):
        try:
            radius = float(self.radius_mm.get()) * 0.001
            count = int(self.circle_count.get())
        except (ValueError, tk.TclError):
            self.error("Circle radius/count must be numeric")
            return
        threading.Thread(
            target=self.node.generate_circle,
            args=(
                self.circle_axis.get().lower(),
                radius,
                count,
                bool(self.close_circle.get()),
                bool(self.circle_face_center.get()),
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def generate_weave(self):
        base_kind = self.weave_base.get()
        source = self.weave_base_paths.get(base_kind, [])
        if len(source) < 2:
            self.error(
                f"Generate a {base_kind} base path before applying weave"
            )
            return
        try:
            amplitude = float(self.weave_amplitude_mm.get()) * 0.001
            cycles = int(self.weave_cycles.get())
            samples = int(self.weave_samples.get())
        except (ValueError, tk.TclError):
            self.error("Weave settings must be numeric")
            return
        self.weave_source = copy.deepcopy(source)
        seam_length = sum(
            (
                (
                    second.position.x - first.position.x
                ) ** 2
                + (
                    second.position.y - first.position.y
                ) ** 2
                + (
                    second.position.z - first.position.z
                ) ** 2
            ) ** 0.5
            for first, second in zip(source[:-1], source[1:])
        )
        pitch_mm = seam_length * 1000.0 / max(cycles, 1)
        self.weave_summary.configure(
            text=(
                f"±{amplitude * 1000.0:.1f} mm · "
                f"pitch≈{pitch_mm:.1f} mm · {cycles} cycles"
            )
        )
        threading.Thread(
            target=self.node.generate_weave,
            args=(
                copy.deepcopy(source),
                amplitude,
                cycles,
                samples,
                self.weave_axis.get(),
                self.show_path.get(),
            ),
            daemon=True,
        ).start()

    def append_tcp(self):
        threading.Thread(
            target=self.node.capture_tcp,
            args=(
                None,
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def capture_initial_state(self):
        self.pipeline_waiting("Capturing current TCP and measured joint angles")
        threading.Thread(
            target=self.node.capture_initial_state,
            args=(self.planning_group.get(),),
            daemon=True,
        ).start()

# /home/irs/ros2_ws/src/construct_robot_ros2/construct_description/config
    def _initial_state_yaml_path(self, planning_group=None):
        group = planning_group or self.planning_group.get()
        return (
            Path.home()
            / "ros2_ws"
            / "src"
            / "construct_robot_ros2"
            / "construct_description"
            / "config"
            / f"{group}_initial_state.yaml"
        )

    def load_initial_state(self):
        default_path = self._initial_state_yaml_path()
        path = filedialog.askopenfilename(
            title="Load TCP teaching state",
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
            filetypes=(("YAML", "*.yaml *.yml"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            planning_group, joint_names, positions, tcp = (
                load_initial_state_yaml(path)
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.error(f"Failed to load initial state YAML: {error}")
            return
        if planning_group != self.planning_group.get():
            self.error(
                f"YAML is for {planning_group}; selected arm is "
                f"{self.planning_group.get()}"
            )
            return
        self.apply_initial_state(
            planning_group,
            joint_names,
            positions,
            tcp,
            save_to_yaml=False,
        )
        self.log(f"Loaded TCP teaching state from {path}")

    def apply_initial_state(
        self,
        planning_group,
        joint_names,
        positions,
        tcp,
        save_to_yaml=True,
    ):
        self.initial_joint_state = (
            planning_group,
            tuple(joint_names),
            tuple(positions),
        )
        self.initial_plan_ready = False
        angles = ", ".join(f"{math.degrees(value):.1f}°" for value in positions)
        self.initial_state_status.configure(text=f"captured joints: {angles}")
        self._refresh_initial_position_controls()
        values = self._pose_values(tcp)
        saved_message = ""
        save_error = None
        if save_to_yaml:
            path = self._initial_state_yaml_path(planning_group)
            try:
                save_initial_state_yaml(
                    path,
                    planning_group,
                    joint_names,
                    positions,
                    tcp,
                )
                saved_message = f" · saved to {path}"
            except (OSError, ValueError, yaml.YAMLError) as error:
                save_error = error
        self.pipeline_result(
            f"Initial state captured · TCP World XYZ="
            f"({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}) m"
            f"{saved_message}"
        )
        if save_error is not None:
            self.error(
                f"Initial state captured, but YAML save failed: {save_error}"
            )

    def plan_initial_state(self):
        if self.initial_joint_state is None:
            self.error("Capture an initial position first")
            return
        if not self._selected_robot_connected():
            self.error("Connect the selected REAL RB robot first")
            return
        group, joint_names, positions = self.initial_joint_state
        if group != self.planning_group.get():
            self.error("Captured initial position belongs to another arm")
            return
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        threading.Thread(
            target=self.node.plan_initial_state,
            args=(
                group,
                joint_names,
                positions,
                max(0.01, min(1.0, self.velocity_percent.get() / 100.0)),
            ),
            daemon=True,
        ).start()

    def initial_position_plan_ready(
        self,
        planning_group,
        target_positions,
        velocity_scale,
        message,
    ):
        if self.initial_joint_state is None:
            return
        group, _joint_names, positions = self.initial_joint_state
        if (
            group != planning_group
            or tuple(positions) != tuple(target_positions)
            or group != self.planning_group.get()
            or not math.isclose(
                velocity_scale,
                max(0.01, min(1.0, self.velocity_percent.get() / 100.0)),
            )
        ):
            self.log("Discarded stale initial-position plan")
            return
        self.initial_plan_ready = True
        self._refresh_initial_position_controls()
        self.pipeline_result(message)

    def execute_initial_plan(self):
        if not self.initial_plan_ready:
            self.error(
                "Plan and inspect the initial-position trajectory first"
            )
            return
        if not self.execution_allowed:
            self.error("Robot execution is disabled by launch configuration")
            return
        if not self._selected_robot_connected():
            self.error("Connect the selected REAL RB robot first")
            return
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        threading.Thread(
            target=self.node.execute_initial_plan,
            daemon=True,
        ).start()

    def initial_position_execution_finished(self, message):
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        self.pipeline_result(message)

    def capture_linear_tcp(self, endpoint_index):
        self.log(
            f"Reading current {self.planning_group.get()} TCP as "
            f"TCP {endpoint_index + 1}..."
        )
        threading.Thread(
            target=self.node.capture_linear_tcp,
            args=(endpoint_index, self.planning_group.get()),
            daemon=True,
        ).start()

    def apply_linear_tcp(self, endpoint_index, pose):
        self.linear_tcp_endpoints[endpoint_index] = copy.deepcopy(pose)
        position = pose.position
        status = (
            f"({position.x:.3f}, {position.y:.3f}, {position.z:.3f})"
        )
        label = self.tcp_1_status if endpoint_index == 0 else self.tcp_2_status
        label.configure(text=status)
        ready = all(pose is not None for pose in self.linear_tcp_endpoints)
        self.generate_tcp_line_button.configure(
            state=tk.NORMAL if ready else tk.DISABLED
        )
        self.log(
            f"Captured TCP {endpoint_index + 1} · World XYZ {status}"
        )

    def acquire_two_tcp(self):
        if any(pose is None for pose in self.linear_tcp_endpoints):
            self.error("Capture both TCP 1 and TCP 2 first")
            return
        try:
            count = int(self.tcp_line_count.get())
        except (ValueError, tk.TclError):
            self.error("TCP line point count must be an integer")
            return
        if self.tcp_line_direction.get() == "TCP 1 → TCP 2":
            start, end = self.linear_tcp_endpoints
        else:
            end, start = self.linear_tcp_endpoints
        self.log(
            f"Generating linear 6D path · {self.tcp_line_direction.get()}"
        )
        threading.Thread(
            target=self.node.generate_tcp_line,
            args=(
                copy.deepcopy(start),
                copy.deepcopy(end),
                count,
                self.show_path.get(),
            ),
            daemon=True,
        ).start()

    def replace_with_tcp(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint to replace")
            return
        threading.Thread(
            target=self.node.capture_tcp,
            args=(
                index,
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def apply_captured_tcp(self, pose, replace_index, visible):
        if replace_index is None:
            self.points.append(copy.deepcopy(pose))
            selected_index = len(self.points) - 1
            action = "Appended"
        else:
            self.points[replace_index] = copy.deepcopy(pose)
            selected_index = replace_index
            action = "Replaced"
        self.path_kind = "taught"
        self.weave_source = copy.deepcopy(self.points)
        self.set_points(self.points, selected_index)
        self.node.publish_points(self.points, visible)
        self.log(
            f"{action} current {self.planning_group.get()} TCP · "
            "World 6D pose"
        )

    def reverse_path(self):
        if len(self.points) < 2:
            self.error("Path needs at least two poses")
            return
        self.points.reverse()
        if self.path_kind != "weave":
            self.weave_source = copy.deepcopy(self.points)
        self.publish_edits(0)
        self.log("Reversed seam direction")

    def restore_weave_source(self):
        if not self.weave_source:
            self.error("No source seam is available")
            return
        self.path_kind = "source"
        self.set_points(self.weave_source)
        self.node.publish_points(self.points, self.show_path.get())
        self.log("Restored the seam used before weaving")

    def clear_path(self):
        self.path_kind = "empty"
        self.weave_source = []
        self.weave_base_paths = {"linear": [], "circle": []}
        self.set_points([])
        self.node.publish_points([], self.show_path.get())
        self.log("Cleared taught path")

    def update_speed_label(self, _value=None):
        self.invalidate_approved_plan()
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        self.speed_label.configure(
            text=f"{self.velocity_percent.get():.0f}%"
        )

    def plan_preview(self):
        if not self._selected_robot_connected():
            self.error("Connect the robot and wait for live /joint_states")
            return
        self._send_path(execute_requested=False)

    def execute_approved(self):
        if not self.plan_approved:
            self.error("Plan Preview is required before execution")
            return
        if not self.execution_allowed:
            self.error("Server execution is disabled by launch configuration")
            return
        if not self._selected_robot_connected():
            self.error("Connect the REAL RB robot first")
            return
        if self.enable_arc.get() and not self.h600_connected:
            self.error("H600 must be connected on TCP/502 before welding")
            return
        if self.enable_arc.get() and not messagebox.askyesno(
            "Confirm physical welding sequence",
            "Execute TCP1 approach, enable H600 welding, move to TCP2, "
            "then disable welding?",
        ):
            self.log("Physical welding sequence canceled")
            return
        self._send_path(execute_requested=True)

    def _send_path(self, execute_requested):
        speed = max(0.01, min(1.0, self.velocity_percent.get() / 100.0))
        planning_group = self.planning_group.get()
        if self.enable_arc.get() and planning_group != "right_manipulator":
            self.error("H600 ARC is allowed only for right_manipulator")
            return
        try:
            current_raw = int(self.weld_current_raw.get())
            voltage_raw = int(self.weld_voltage_raw.get())
            v_offset_raw = int(self.weld_v_offset_raw.get())
            preflow_seconds = float(self.weld_preflow_seconds.get())
            postflow_seconds = float(self.weld_postflow_seconds.get())
            interpolation_step = (
                float(self.interpolation_step_mm.get()) * 0.001
            )
        except (ValueError, tk.TclError):
            self.error("H600 raw values/timing are invalid")
            return
        if not 0.0 <= preflow_seconds <= 10.0:
            self.error("H600 pre-flow must be in 0..10 seconds")
            return
        if not 0.0 <= postflow_seconds <= 10.0:
            self.error("H600 post-flow must be in 0..10 seconds")
            return
        if not 0.0005 <= interpolation_step <= 0.02:
            self.error("Cartesian interpolation step must be 0.5..20 mm")
            return
        threading.Thread(
            target=self.node.send,
            args=(
                copy.deepcopy(self.points),
                speed,
                interpolation_step,
                self.show_path.get(),
                self.enable_arc.get(),
                current_raw,
                voltage_raw,
                v_offset_raw,
                preflow_seconds,
                postflow_seconds,
                self.require_welding_feedback.get(),
                execute_requested,
                execute_requested,
                planning_group,
            ),
            daemon=True,
        ).start()

    def update_welder_status(self, message):
        self.h600_connected = bool(
            message.server_running and message.client_connected
        )

    def update_control_box_io(self, digital_in, digital_out):
        current = (tuple(digital_in), tuple(digital_out))
        previous = self.previous_control_box_io
        changes = []
        for kind, values, old_values in (
            ("DI", current[0], previous[0] if previous else None),
            ("DO", current[1], previous[1] if previous else None),
        ):
            for port, value in enumerate(values):
                changed = (
                    old_values is not None and value != old_values[port]
                )
                candidate = port in OBSERVED_H600_IO_CANDIDATES
                if value:
                    background = "#81c995"
                elif changed:
                    background = "#fdd663"
                elif candidate:
                    background = "#dbeafe"
                else:
                    background = "#eeeeee"
                self.control_box_io_labels[(kind, port)].configure(
                    text=f"{port:02d}\n{'ON' if value else 'OFF'}",
                    bg=background,
                )
                if changed:
                    changes.append(
                        f"{kind}{port}={'ON' if value else 'OFF'}"
                    )
        self.previous_control_box_io = current
        active_inputs = [
            str(index) for index, value in enumerate(current[0]) if value
        ]
        active_outputs = [
            str(index) for index, value in enumerate(current[1]) if value
        ]
        self.control_box_io_status.configure(
            text=(
                f"Active DI: {', '.join(active_inputs) or 'none'} · "
                f"Active DO: {', '.join(active_outputs) or 'none'}"
            )
        )
        if changes:
            self.log("Rainbow control-box I/O changed · " + ", ".join(changes))

    def update_touch_input(self, arm, active):
        previous = self.touch_input_states[arm]
        active = bool(active)
        self.touch_input_states[arm] = active
        if previous is not None and active and not previous:
            self.touch_input_rising_edges[arm] += 1
            self._handle_touch_event(arm, f"{arm.upper()} DI0")

    def _handle_touch_event(self, arm, source):
        planning_group = f"{arm}_manipulator"
        self.root.bell()
        self.pipeline_waiting(
            f"TOUCH DETECTED · source={source} · "
            f"capturing {planning_group} TCP"
        )
        self.node.capture_touch_pose(planning_group, source)

    def apply_touch_capture(self, pose, planning_group, source):
        self.last_touch_pose = copy.deepcopy(pose)
        values = self._pose_values(pose)
        self.pipeline_result(
            f"TOUCH TCP CAPTURED · {planning_group} · "
            f"World XYZ=({values[0]:.6f}, {values[1]:.6f}, "
            f"{values[2]:.6f}) m"
        )

    def confirm_all_do_unlock(self):
        if not self.unlock_all_do_ports.get():
            return
        if not messagebox.askyesno(
            "Unlock all Rainbow DO ports",
            "Unknown outputs may operate gas, inching, ARC, or another "
            "actuator. Allow clicking every DO0..15 port?",
        ):
            self.unlock_all_do_ports.set(False)

    def request_do_toggle(self, port):
        if self.previous_control_box_io is None:
            self.error("Rainbow control-box state is not available")
            return
        if port in self.pending_do_ports:
            return
        candidate = port in OBSERVED_H600_IO_CANDIDATES
        if not candidate and not self.unlock_all_do_ports.get():
            self.error(
                f"DO{port} is locked · enable non-candidate DO clicking first"
            )
            return
        current = bool(self.previous_control_box_io[1][port])
        target = not current
        if not messagebox.askyesno(
            f"Toggle Rainbow DO{port}",
            f"Command control-box DO{port}: "
            f"{'ON' if current else 'OFF'} → {'ON' if target else 'OFF'}?\n\n"
            "This is a physical output and may operate connected equipment.",
        ):
            return
        self.pending_do_ports.add(port)
        label = self.control_box_io_labels[("DO", port)]
        label.configure(bg="#fdd663", text=f"{port:02d}\nWAIT")
        self.log(
            f"Rainbow DO{port} command requested · "
            f"{'ON' if target else 'OFF'}"
        )
        threading.Thread(
            target=self.node.set_digital_output,
            args=(port, target),
            daemon=True,
        ).start()

    def candidate_outputs_off(self):
        if not messagebox.askyesno(
            "Force candidate outputs OFF",
            "Command DO4, DO8, DO9, DO10, DO12, and DO13 to OFF?",
        ):
            return
        for port in sorted(OBSERVED_H600_IO_CANDIDATES):
            self.pending_do_ports.add(port)
            threading.Thread(
                target=self.node.set_digital_output,
                args=(port, False),
                daemon=True,
            ).start()
        self.log("Rainbow candidate DO all-OFF requested")

    def digital_output_result(self, port, success, message):
        self.pending_do_ports.discard(port)
        prefix = "OK" if success else "REJECTED"
        self.log(f"Rainbow DO{port} {prefix} · {message}")
        if not success and self.previous_control_box_io is not None:
            value = self.previous_control_box_io[1][port]
            self.control_box_io_labels[("DO", port)].configure(
                text=f"{port:02d}\n{'ON' if value else 'OFF'}",
                bg=(
                    "#81c995"
                    if value
                    else (
                        "#dbeafe"
                        if port in OBSERVED_H600_IO_CANDIDATES
                        else "#eeeeee"
                    )
                ),
            )

    def begin(self, velocity_scale, execute_requested):
        self.bar["value"] = 0
        self.last_action_phase = ""
        self.plan_button.configure(state=tk.DISABLED)
        self.execute_button.configure(state=tk.DISABLED)
        operation = (
            "EXECUTE exact approved plan"
            if execute_requested
            else "PLAN PREVIEW for RViz"
        )
        self.pipeline_waiting(
            f"{operation} · "
            f"speed={velocity_scale:.0%}"
        )

    def progress(self, value, waypoint, pose, phase):
        self.bar["value"] = value * 100
        position = pose.position
        self.feedback_label.configure(
            text=(
                f"{phase or 'PATH'} · waypoint: {waypoint + 1} · "
                f"progress: {value:.0%} · "
                f"pose: ({position.x:.3f}, {position.y:.3f}, "
                f"{position.z:.3f})"
            ),
        )
        if phase and phase != self.last_action_phase:
            self.last_action_phase = phase
            self.log(f"Sequence phase · {phase}")

    def finish(self, text, was_execution):
        self.bar["value"] = 100
        self.plan_button.configure(
            state=(
                tk.NORMAL
                if self.points and self._selected_robot_connected()
                else tk.DISABLED
            )
        )
        self.plan_approved = not was_execution
        self.execute_button.configure(
            state=(
                tk.NORMAL
                if (
                    self.plan_approved
                    and self.execution_allowed
                    and self._selected_robot_connected()
                )
                else tk.DISABLED
            )
        )
        if self.plan_approved:
            self.pipeline_result(
                f"{text} · plan approved; inspect RViz, then execute"
            )
        else:
            self.pipeline_result(text)

    def cancel(self):
        self.node.cancel()

    def close(self):
        if self._closing:
            return
        self._closing = True
        self.root.quit()
        self.root.destroy()

    def shutdown_ros(self):
        if rclpy.ok():
            rclpy.shutdown()
        self.executor_thread.join(timeout=1.0)
        self.node.destroy_node()

    def check_ros(self):
        self._drain_ui_queue()
        if not rclpy.ok():
            self.root.destroy()
            return
        self.root.after(50, self.check_ros)

    def mainloop(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    gui = WeldActionGui()
    try:
        gui.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        gui.shutdown_ros()
