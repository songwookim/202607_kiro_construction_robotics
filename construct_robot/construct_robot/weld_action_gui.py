import copy
import math
import queue
import signal
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
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


class WeldActionNode(Node):
    """ROS interface used by the editable weld-path GUI."""

    def __init__(self, ui):
        super().__init__("weld_action_gui")
        self.ui = ui
        self.declare_parameter("expected_execute_motion", True)
        self.declare_parameter("robot_feedback_timeout", 5.0)
        self.client = ActionClient(self, CartesianPath, "cartesian_path")
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
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
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_handle = None
        self.request_execution = False
        self.execute_motion_enabled = self.get_parameter(
            "expected_execute_motion"
        ).value
        self.expect_robot_feedback = {"left": True, "right": True}
        self.robot_feedback_seen = {"left": False, "right": False}
        self.robot_ready_reported = {"left": False, "right": False}
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
                or self.trajectory_clients[arm].server_is_ready()
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
            f"Generated World-YZ circle · {description} · {orientation}",
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
        goal.interpolation_step = 0.005
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
        self.path_kind = "empty"
        self.execution_allowed = False
        self.robot_connected = {"left": False, "right": False}
        self.plan_approved = False
        self.linear_tcp_endpoints = [None, None]
        self.pose_variables = {
            name: tk.StringVar(value="0.0") for name in self.POSE_FIELDS
        }
        self.radius_mm = tk.DoubleVar(value=20.0)
        self.circle_count = tk.IntVar(value=16)
        self.close_circle = tk.BooleanVar(value=True)
        self.circle_face_center = tk.BooleanVar(value=True)
        self.nudge_mm = tk.DoubleVar(value=5.0)
        self.velocity_percent = tk.DoubleVar(value=20.0)
        self.show_path = tk.BooleanVar(value=True)
        self.weave_amplitude_mm = tk.DoubleVar(value=3.0)
        self.weave_cycles = tk.IntVar(value=4)
        self.weave_samples = tk.IntVar(value=8)
        self.weave_axis = tk.StringVar(value="tool_y")
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
        self.touch_sensor_arm = tk.StringVar(value="right")
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

        motion_tests = ttk.LabelFrame(
            outer,
            text="Motion test generators · click a row to expand/collapse",
        )
        motion_tests.pack(fill=tk.X)
        straight = self._create_toggle_section(
            motion_tests,
            "straight",
            "Straight path",
            expanded=True,
        )
        ttk.Button(
            straight,
            text="Acquire straight path",
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
        ttk.Label(straight, text="start").pack(side=tk.LEFT)
        ttk.Combobox(
            straight,
            textvariable=self.straight_start_mode,
            values=("Current TCP", "World XYZ"),
            state="readonly",
            width=11,
        ).pack(side=tk.LEFT, padx=(3, 5))
        for label, variable in (
            ("X", self.straight_start_x),
            ("Y", self.straight_start_y),
            ("Z", self.straight_start_z),
        ):
            ttk.Label(straight, text=label).pack(side=tk.LEFT)
            ttk.Entry(
                straight,
                textvariable=variable,
                width=6,
            ).pack(side=tk.LEFT, padx=(2, 4))
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

        tcp_line = self._create_toggle_section(
            motion_tests,
            "tcp_line",
            "TCP-to-TCP linear path",
        )
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

        controls = self._create_toggle_section(
            motion_tests,
            "circle",
            "Circle path",
        )
        ttk.Button(
            controls,
            text="Generate circle",
            command=self.generate_circle,
        ).pack(side=tk.LEFT, padx=(0, 10))
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
            to=64,
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
            controls,
            text="show planned path",
            variable=self.show_path,
            command=self.toggle_path_visibility,
        ).pack(side=tk.LEFT, padx=(14, 0))

        teaching = ttk.Frame(outer)
        teaching.pack(fill=tk.X, pady=(7, 0))
        ttk.Label(
            teaching,
            text="TCP teaching:",
            style="Step.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            teaching,
            text="Append current TCP",
            command=self.append_tcp,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            teaching,
            text="Replace selected ← TCP",
            command=self.replace_with_tcp,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            teaching,
            text="Reverse seam",
            command=self.reverse_path,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            teaching,
            text="Restore source seam",
            command=self.restore_weave_source,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            teaching,
            text="Clear",
            command=self.clear_path,
        ).pack(side=tk.LEFT, padx=3)
        self.path_summary = ttk.Label(teaching, text="empty path")
        self.path_summary.pack(side=tk.LEFT, padx=(14, 0))

        weaving = self._create_toggle_section(
            motion_tests,
            "weave",
            "Weave path",
        )
        ttk.Button(
            weaving,
            text="Apply weave to current path",
            command=self.generate_weave,
        ).pack(side=tk.LEFT, padx=(0, 8))
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

        ttk.Label(
            outer,
            text="Editable path · World frame",
            style="Step.TLabel",
        ).pack(anchor=tk.W, pady=(12, 5))
        columns = ("id",) + self.POSE_FIELDS
        self.table = ttk.Treeview(
            outer,
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
        self.table.bind("<<TreeviewSelect>>", self.load_selected)

        editor = ttk.Frame(outer)
        editor.pack(fill=tk.X, pady=(6, 0))
        for name in self.POSE_FIELDS:
            ttk.Label(editor, text=name).pack(side=tk.LEFT)
            ttk.Entry(
                editor,
                textvariable=self.pose_variables[name],
                width=9,
            ).pack(side=tk.LEFT, padx=(2, 5))
        ttk.Button(
            editor,
            text="Apply selected",
            command=self.apply_selected,
        ).pack(side=tk.LEFT, padx=(6, 0))

        edit_buttons = ttk.Frame(outer)
        edit_buttons.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            edit_buttons,
            text="Duplicate",
            command=self.duplicate_selected,
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            edit_buttons,
            text="Delete",
            command=self.delete_selected,
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            edit_buttons,
            text="Move up",
            command=lambda: self.move_selected(-1),
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            edit_buttons,
            text="Move down",
            command=lambda: self.move_selected(1),
        ).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(edit_buttons, text="nudge (mm)").pack(side=tk.LEFT)
        ttk.Spinbox(
            edit_buttons,
            from_=0.1,
            to=100,
            increment=0.5,
            textvariable=self.nudge_mm,
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 6))
        for axis in ("X", "Y", "Z"):
            ttk.Button(
                edit_buttons,
                text=f"−{axis}",
                command=lambda value=axis.lower(): self.nudge(value, -1),
                width=4,
            ).pack(side=tk.LEFT, padx=1)
            ttk.Button(
                edit_buttons,
                text=f"+{axis}",
                command=lambda value=axis.lower(): self.nudge(value, 1),
                width=4,
            ).pack(side=tk.LEFT, padx=1)

        robot_status = ttk.LabelFrame(outer, text="Robot connection")
        robot_status.pack(fill=tk.X, pady=(10, 0))
        self.robot_connection_labels = {}
        for arm in ("left", "right"):
            label = tk.Label(
                robot_status,
                text=f"{arm.upper()}  X",
                width=14,
                relief=tk.SOLID,
                borderwidth=1,
                bg="#fce8e6",
                fg="#b3261e",
                font=("Sans", 11, "bold"),
            )
            label.pack(side=tk.LEFT, padx=6, pady=6)
            self.robot_connection_labels[arm] = label

        io_monitor = ttk.LabelFrame(
            outer,
            text=(
                "Rainbow control-box digital I/O · raw ports 0..15 · "
                "touch DI0 · other observed candidates: 4, 8, 9, 10, 12, 13"
            ),
        )
        io_monitor.pack(fill=tk.X, pady=(7, 0))
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

        touch_calibration = ttk.LabelFrame(
            outer,
            text="TCP touch calibration · simulated contact + real DI0",
        )
        touch_calibration.pack(fill=tk.X, pady=(7, 0))
        touch_controls = ttk.Frame(touch_calibration)
        touch_controls.pack(fill=tk.X, padx=6, pady=(5, 3))
        ttk.Label(touch_controls, text="sensor control box").pack(
            side=tk.LEFT,
        )
        touch_arm_selector = ttk.Combobox(
            touch_controls,
            textvariable=self.touch_sensor_arm,
            values=("right", "left"),
            state="readonly",
            width=7,
        )
        touch_arm_selector.pack(side=tk.LEFT, padx=(4, 10))
        touch_arm_selector.bind(
            "<<ComboboxSelected>>",
            self.touch_sensor_arm_changed,
        )
        ttk.Button(
            touch_controls,
            text="Simulate TOUCH now",
            command=self.simulate_touch,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            touch_controls,
            text=(
                "A rising DI0 or the simulate button captures the selected "
                "arm TCP in World"
            ),
        ).pack(side=tk.LEFT)
        self.touch_input_status = tk.Label(
            touch_calibration,
            text="RIGHT DI00 TOUCH: waiting for robot state",
            anchor=tk.W,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#eeeeee",
            font=("Sans", 10, "bold"),
        )
        self.touch_input_status.pack(fill=tk.X, padx=6, pady=3)
        self.touch_pose_status = ttk.Label(
            touch_calibration,
            text="Last touch TCP: not captured",
        )
        self.touch_pose_status.pack(fill=tk.X, padx=6, pady=(2, 6))

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
            text="weld travel speed",
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

        welder = ttk.Frame(outer)
        welder.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(
            welder,
            text="H600 ARC during execution",
            variable=self.enable_arc,
            command=self.invalidate_approved_plan,
        ).pack(side=tk.LEFT, padx=(0, 10))
        for label, variable in (
            ("current raw", self.weld_current_raw),
            ("voltage raw", self.weld_voltage_raw),
            ("V offset", self.weld_v_offset_raw),
        ):
            ttk.Label(welder, text=label).pack(side=tk.LEFT)
            ttk.Spinbox(
                welder,
                from_=0,
                to=65535,
                textvariable=variable,
                width=7,
            ).pack(side=tk.LEFT, padx=(3, 8))
        self.welder_status = ttk.Label(
            welder,
            text="H600: waiting",
        )
        self.welder_status.pack(side=tk.LEFT, padx=(12, 0))
        weld_timing = ttk.Frame(outer)
        weld_timing.pack(fill=tk.X, pady=(4, 0))
        ttk.Checkbutton(
            weld_timing,
            text="Require H600 welding feedback before motion",
            variable=self.require_welding_feedback,
        ).pack(side=tk.LEFT, padx=(0, 10))
        for label, variable in (
            ("pre-flow s", self.weld_preflow_seconds),
            ("post-flow s", self.weld_postflow_seconds),
        ):
            ttk.Label(weld_timing, text=label).pack(side=tk.LEFT)
            ttk.Spinbox(
                weld_timing,
                from_=0.0,
                to=10.0,
                increment=0.1,
                textvariable=variable,
                width=5,
            ).pack(side=tk.LEFT, padx=(3, 9))
        ttk.Label(
            weld_timing,
            text=(
                "Sequence: ARC OFF approach → TCP1 → pre-flow/ARC ON → "
                "TCP2 → ARC OFF/post-flow"
            ),
        ).pack(side=tk.LEFT, padx=(10, 0))

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
        self.status = tk.Text(
            outer,
            height=6,
            bg="#101820",
            fg="#d5f5e3",
        )
        self.status.pack(fill=tk.X)
        self.log("Ready · edits publish immediately to RViz")

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
                text=f"{arm.upper()}  {'O' if value else 'X'}",
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

    def log(self, text):
        self.status.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self.status.see(tk.END)

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
        self.path_kind = kind
        self.set_points(points)

    def set_execution_configuration(self, execute_motion):
        self.execution_allowed = execute_motion
        self.robot_connected = {"left": False, "right": False}
        self._refresh_execution_controls()
        self.log(
            "Connecting LEFT 192.168.1.11 + RIGHT 192.168.1.10 · "
            "waiting for measured feedback and planning readiness"
        )

    def robot_feedback_connected(self, arm):
        self.robot_connected[arm] = True
        self._refresh_execution_controls()
        self.log(
            f"READY · {arm}-arm feedback and MoveIt/controller available"
        )

    def robot_feedback_lost(self, arm):
        self.robot_connected[arm] = False
        self.invalidate_approved_plan()
        self._refresh_execution_controls()
        if self._selected_arm() == arm:
            self.plan_button.configure(state=tk.DISABLED)
        self.log(f"ERROR · {arm}-arm measured joint feedback timeout")

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
        if len(self.points) < 2:
            self.error("Teach or acquire at least two seam points first")
            return
        try:
            amplitude = float(self.weave_amplitude_mm.get()) * 0.001
            cycles = int(self.weave_cycles.get())
            samples = int(self.weave_samples.get())
        except (ValueError, tk.TclError):
            self.error("Weave settings must be numeric")
            return
        source = (
            self.weave_source
            if self.path_kind == "weave" and self.weave_source
            else self.points
        )
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
        self.set_points([])
        self.node.publish_points([], self.show_path.get())
        self.log("Cleared taught path")

    def update_speed_label(self, _value=None):
        self.invalidate_approved_plan()
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
        except (ValueError, tk.TclError):
            self.error("H600 raw values/timing are invalid")
            return
        if not 0.0 <= preflow_seconds <= 10.0:
            self.error("H600 pre-flow must be in 0..10 seconds")
            return
        if not 0.0 <= postflow_seconds <= 10.0:
            self.error("H600 post-flow must be in 0..10 seconds")
            return
        threading.Thread(
            target=self.node.send,
            args=(
                copy.deepcopy(self.points),
                speed,
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
        connection = (
            f"connected {message.client_address}"
            if message.client_connected
            else "disconnected"
        )
        self.welder_status.configure(
            text=(
                f"H600: {connection} · welding={message.welding} · "
                f"I={message.current_feedback_raw} "
                f"V={message.voltage_feedback_raw}"
            )
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

    def touch_sensor_arm_changed(self, _event=None):
        arm = self.touch_sensor_arm.get()
        state = self.touch_input_states.get(arm)
        self._refresh_touch_status(arm, state)
        self.log(f"Touch sensor source changed to {arm.upper()} DI0")

    def _refresh_touch_status(self, arm, active):
        state = "waiting" if active is None else ("ON" if active else "OFF")
        count = self.touch_input_rising_edges[arm]
        self.touch_input_status.configure(
            text=(
                f"{arm.upper()} DI{TOUCH_INPUT_PORT:02d} TOUCH: {state} · "
                f"detected contacts: {count}"
            ),
            bg="#81c995" if active else "#eeeeee",
        )

    def update_touch_input(self, arm, active):
        previous = self.touch_input_states[arm]
        active = bool(active)
        self.touch_input_states[arm] = active
        if arm == self.touch_sensor_arm.get():
            self._refresh_touch_status(arm, active)
        if previous is not None and active and not previous:
            self.touch_input_rising_edges[arm] += 1
            if arm == self.touch_sensor_arm.get():
                self._refresh_touch_status(arm, active)
                self._handle_touch_event(f"{arm.upper()} DI0")

    def simulate_touch(self):
        self._handle_touch_event("SIMULATED TOUCH")

    def _handle_touch_event(self, source):
        planning_group = self.planning_group.get()
        self.touch_input_status.configure(
            text=f"TOUCH DETECTED · {source} · capturing TCP...",
            bg="#81c995",
        )
        self.root.bell()
        self.log(
            f"TOUCH DETECTED · source={source} · "
            f"capturing {planning_group} TCP"
        )
        self.node.capture_touch_pose(planning_group, source)

    def apply_touch_capture(self, pose, planning_group, source):
        self.last_touch_pose = copy.deepcopy(pose)
        values = self._pose_values(pose)
        self.touch_input_status.configure(
            text=f"TOUCH CAPTURED · {source} · {planning_group}",
            bg="#81c995",
        )
        self.touch_pose_status.configure(
            text=(
                f"Last touch TCP · {planning_group} · {source} · "
                f"World XYZ=({values[0]:.6f}, {values[1]:.6f}, "
                f"{values[2]:.6f}) m · Q=({values[3]:.6f}, "
                f"{values[4]:.6f}, {values[5]:.6f}, {values[6]:.6f})"
            )
        )
        self.log(
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
        self.log(
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
        self.log(text)
        if self.plan_approved:
            self.log(
                "Plan approved · inspect RViz, then press "
                "Execute Approved Plan"
            )

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
