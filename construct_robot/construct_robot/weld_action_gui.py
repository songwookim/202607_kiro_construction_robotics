import copy
import math
import signal
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray

from construct_msgs.action import CartesianPath
from construct_msgs.msg import WelderStatus
from construct_msgs.srv import SetRobotConnection
from construct_robot.cartesian_path_common import (
    circle_waypoints,
    linear_pose_waypoints,
    pose_is_valid,
    straight_waypoints,
    weaving_from_path,
)
from construct_robot.cartesian_path_server import make_weld_visualization


RIGHT_JOINT_NAMES = tuple(
    f"right_manipulator_joint{index}" for index in range(1, 7)
)


def complete_right_joint_positions(message):
    positions = dict(zip(message.name, message.position))
    if not all(
        name in positions and math.isfinite(positions[name])
        for name in RIGHT_JOINT_NAMES
    ):
        return None
    return tuple(float(positions[name]) for name in RIGHT_JOINT_NAMES)


class WeldActionNode(Node):
    """ROS interface used by the editable weld-path GUI."""

    def __init__(self, ui):
        super().__init__("weld_action_gui")
        self.ui = ui
        self.declare_parameter("expected_execute_motion", False)
        self.declare_parameter("expected_robot_connected", True)
        self.declare_parameter(
            "expected_right_robot_ip",
            "192.168.1.10",
        )
        self.client = ActionClient(self, CartesianPath, "cartesian_path")
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )
        self.connection_client = self.create_client(
            SetRobotConnection,
            "/weld_stack/set_robot_connection",
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_handle = None
        self.initial_goal_handle = None
        self.request_execution = False
        self.joint_state_lock = threading.Lock()
        self.right_joint_positions = None
        self.robot_feedback_seen = False
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
        self.create_subscription(
            WelderStatus,
            "/h600/status",
            self._welder_status,
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
            self.get_parameter("expected_robot_connected").value,
            self.get_parameter("expected_right_robot_ip").value,
        )

    def _joint_state(self, message):
        positions = complete_right_joint_positions(message)
        if positions is None:
            return
        with self.joint_state_lock:
            self.right_joint_positions = positions
        if not self.robot_feedback_seen:
            self.robot_feedback_seen = True
            self.ui.post(self.ui.robot_feedback_connected)

    def current_right_joint_positions(self):
        with self.joint_state_lock:
            return self.right_joint_positions

    def move_right_to_joint_positions(self, positions, velocity_scale):
        if len(positions) != len(RIGHT_JOINT_NAMES) or not all(
            math.isfinite(value) for value in positions
        ):
            self.ui.post(self.ui.initial_move_error, "Invalid saved pose")
            return
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(
                self.ui.initial_move_error,
                "MoveIt /move_action unavailable",
            )
            return

        constraints = Constraints()
        for name, position in zip(RIGHT_JOINT_NAMES, positions):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = position
            constraint.tolerance_above = 0.005
            constraint.tolerance_below = 0.005
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)

        goal = MoveGroup.Goal()
        goal.request.group_name = "right_manipulator"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = velocity_scale
        goal.request.max_acceleration_scaling_factor = velocity_scale
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        future = self.move_group_client.send_goal_async(goal)
        future.add_done_callback(self._initial_goal_response)

    def _initial_goal_response(self, future):
        try:
            self.initial_goal_handle = future.result()
        except Exception as error:
            self.ui.post(
                self.ui.initial_move_error,
                f"Initial pose goal failed: {error}",
            )
            return
        if not self.initial_goal_handle.accepted:
            self.ui.post(
                self.ui.initial_move_error,
                "Initial pose goal rejected by MoveIt",
            )
            return
        self.ui.post(
            self.ui.log,
            "Initial pose goal accepted · planning/executing",
        )
        result = self.initial_goal_handle.get_result_async()
        result.add_done_callback(self._initial_move_result)

    def _initial_move_result(self, future):
        try:
            error_code = future.result().result.error_code.val
        except Exception as error:
            self.ui.post(
                self.ui.initial_move_error,
                f"Initial pose result failed: {error}",
            )
            return
        self.ui.post(self.ui.initial_move_finished, error_code)

    def _current_tcp_pose(self):
        transform = self.tf_buffer.lookup_transform(
            "World",
            "right_manipulator_ee_point",
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
    ):
        try:
            tcp = self._current_tcp_pose()
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
    ):
        try:
            tcp = self._current_tcp_pose()
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

    def capture_tcp(self, replace_index, visible):
        try:
            pose = self._current_tcp_pose()
        except TransformException as error:
            self.ui.post(self.ui.error, f"TCP capture failed: {error}")
            return
        self.ui.post(
            self.ui.apply_captured_tcp,
            pose,
            replace_index,
            visible,
        )

    def capture_linear_tcp(self, endpoint_index):
        try:
            pose = self._current_tcp_pose()
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

    def set_robot_connection(self, connect, right_robot_ip):
        if not self.connection_client.wait_for_service(timeout_sec=1.5):
            self.ui.post(
                self.ui.error,
                "Robot connection supervisor unavailable. Start "
                "weld_supervised.launch.py",
            )
            return
        if connect:
            self.robot_feedback_seen = False
        request = SetRobotConnection.Request()
        request.connect = connect
        request.right_robot_ip = right_robot_ip
        future = self.connection_client.call_async(request)
        future.add_done_callback(
            lambda result: self._connection_response(result, connect)
        )

    def _connection_response(self, future, connect):
        try:
            response = future.result()
            if response.accepted:
                self.ui.post(
                    self.ui.connection_change_accepted,
                    connect,
                    response.message,
                )
            else:
                self.ui.post(self.ui.error, response.message)
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Robot connection request failed: {error}",
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
        execute_requested,
        reuse_approved_plan,
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
        goal.planning_group = "right_manipulator"
        goal.interpolation_step = 0.005
        goal.velocity_scale = velocity_scale
        goal.execute_requested = execute_requested
        goal.reuse_approved_plan = reuse_approved_plan
        goal.visualize_path = visualize_path
        goal.enable_arc = enable_arc
        goal.weld_current_raw = current_raw
        goal.weld_voltage_raw = voltage_raw
        goal.weld_v_offset_raw = v_offset_raw
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

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("KIRO Laser Weld · Editable Right Arm Action")
        self.root.geometry("1240x940")
        self._closing = False
        self.points = []
        self.weave_source = []
        self.path_kind = "empty"
        self.execution_allowed = False
        self.robot_connected = True
        self.plan_approved = False
        self.saved_initial_joints = None
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
        self.tcp_line_direction = tk.StringVar(value="TCP 2 → TCP 1")
        self.enable_arc = tk.BooleanVar(value=False)
        self.right_robot_ip = tk.StringVar(value="192.168.1.10")
        self.weld_current_raw = tk.IntVar(value=0)
        self.weld_voltage_raw = tk.IntVar(value=0)
        self.weld_v_offset_raw = tk.IntVar(value=0)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 18, "bold"))
        style.configure("Step.TLabel", font=("Sans", 11, "bold"))

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="KIRO Editable Welding Action Console",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text=(
                "Laser/GUI poses → /cartesian_path → MoveIt → "
                "right_manipulator controller"
            ),
        ).pack(anchor=tk.W, pady=(2, 12))

        straight = ttk.LabelFrame(
            outer,
            text="1 · Acquire straight seam with respect to axis",
        )
        straight.pack(fill=tk.X)
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

        tcp_line = ttk.LabelFrame(
            outer,
            text="TCP-to-TCP linear seam",
        )
        tcp_line.pack(fill=tk.X, pady=(7, 0))
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

        controls = ttk.LabelFrame(outer, text="Circle seam")
        controls.pack(fill=tk.X, pady=(7, 0))
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

        weaving = ttk.LabelFrame(
            outer,
            text="2 · Apply weaving to the current taught seam",
        )
        weaving.pack(fill=tk.X, pady=(7, 0))
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
            height=8,
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

        hardware_mode = ttk.Frame(outer)
        hardware_mode.pack(fill=tk.X, pady=(10, 0))
        self.execution_mode = ttk.Label(
            hardware_mode,
            text="Execution mode: waiting for launch configuration",
            font=("Sans", 10, "bold"),
        )
        self.execution_mode.pack(side=tk.LEFT)
        ttk.Label(
            hardware_mode,
            text="right RB IP",
        ).pack(side=tk.LEFT, padx=(18, 4))
        ttk.Entry(
            hardware_mode,
            textvariable=self.right_robot_ip,
            width=15,
        ).pack(side=tk.LEFT)
        ttk.Button(
            hardware_mode,
            text="Robot Connect",
            command=self.connect_robot,
        ).pack(side=tk.LEFT, padx=(8, 3))
        ttk.Button(
            hardware_mode,
            text="Robot Disconnect",
            command=self.disconnect_robot,
        ).pack(side=tk.LEFT, padx=3)

        initial_pose = ttk.Frame(outer)
        initial_pose.pack(fill=tk.X, pady=(7, 0))
        ttk.Label(
            initial_pose,
            text="Right arm initial pose:",
            style="Step.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 7))
        ttk.Button(
            initial_pose,
            text="Save CURRENT as initial",
            command=self.save_current_as_initial,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.go_initial_button = ttk.Button(
            initial_pose,
            text="Go to saved initial",
            command=self.go_to_saved_initial,
            state=tk.DISABLED,
        )
        self.go_initial_button.pack(side=tk.LEFT, padx=(0, 9))
        self.initial_pose_status = ttk.Label(
            initial_pose,
            text="not saved",
        )
        self.initial_pose_status.pack(side=tk.LEFT)

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
        self.status.pack(fill=tk.BOTH, expand=True)
        self.log(
            "Ready · edits publish immediately to both RViz and Viser"
        )

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
        self.root.after(0, callback, *args)

    def log(self, text):
        self.status.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self.status.see(tk.END)

    def error(self, text):
        self.log(f"ERROR · {text}")
        self.plan_approved = False
        state = tk.NORMAL if self.points else tk.DISABLED
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
                if self.points and self.robot_connected
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

    def set_execution_configuration(
        self,
        execute_motion,
        robot_connected,
        right_robot_ip,
    ):
        self.execution_allowed = execute_motion
        self.robot_connected = robot_connected
        self.right_robot_ip.set(right_robot_ip)
        if not robot_connected:
            text = "ROBOT DISCONNECTED · REAL RB"
            color = "#b3261e"
        elif execute_motion:
            text = "ROBOT CONNECTED · EXECUTION ENABLED · REAL RB"
            color = "#b06000"
        else:
            text = "ROBOT CONNECTED · PLAN-ONLY · REAL RB"
            color = "#b3261e"
        self.execute_button.configure(
            state=(
                tk.NORMAL
                if (
                    execute_motion
                    and robot_connected
                    and self.plan_approved
                )
                else tk.DISABLED
            )
        )
        self.execution_mode.configure(text=text, foreground=color)
        self._update_go_initial_button()
        self.log(text)

    def _update_go_initial_button(self):
        enabled = (
            self.execution_allowed
            and self.robot_connected
            and self.saved_initial_joints is not None
        )
        self.go_initial_button.configure(
            state=tk.NORMAL if enabled else tk.DISABLED
        )

    def save_current_as_initial(self):
        positions = self.node.current_right_joint_positions()
        if positions is None:
            self.error(
                "Cannot save initial pose: incomplete right-arm /joint_states"
            )
            return
        self.saved_initial_joints = tuple(positions)
        values = ", ".join(f"{value:.3f}" for value in positions)
        self.initial_pose_status.configure(
            text=f"saved [{values}] rad"
        )
        self._update_go_initial_button()
        self.log("Saved CURRENT right-arm joint pose as initial")

    def go_to_saved_initial(self):
        if self.saved_initial_joints is None:
            self.error("Save the current right-arm pose first")
            return
        if not self.execution_allowed:
            self.error("Execution is disabled by launch configuration")
            return
        if not self.robot_connected:
            self.error("Connect the REAL RB robot first")
            return
        confirmed = messagebox.askyesno(
            "Move REAL right arm to saved initial pose?",
            (
                "Move the physical right arm to the saved six-joint "
                "pose through MoveIt?\n\n"
                "Verify the workspace and emergency stop first."
            ),
            icon="warning",
        )
        if not confirmed:
            return
        velocity_scale = max(
            0.01,
            min(1.0, self.velocity_percent.get() / 100.0),
        )
        self.go_initial_button.configure(state=tk.DISABLED)
        self.initial_pose_status.configure(text="planning/executing...")
        self.log(
            f"Going to saved initial pose · speed={velocity_scale:.0%}"
        )
        threading.Thread(
            target=self.node.move_right_to_joint_positions,
            args=(self.saved_initial_joints, velocity_scale),
            daemon=True,
        ).start()

    def initial_move_finished(self, error_code):
        if error_code == 1:
            self.initial_pose_status.configure(text="reached saved initial")
            self.log("SUCCESS · reached saved right-arm initial pose")
        else:
            self.initial_pose_status.configure(
                text=f"MoveIt error code {error_code}"
            )
            self.error(
                f"Failed to reach saved initial pose · code={error_code}"
            )
        self._update_go_initial_button()

    def initial_move_error(self, message):
        self.initial_pose_status.configure(text=message)
        self.error(message)
        self._update_go_initial_button()

    def connect_robot(self):
        right_robot_ip = self.right_robot_ip.get().strip()
        if not right_robot_ip:
            self.error("Enter the right RB IP before connecting")
            return
        confirmed = messagebox.askyesno(
            "Connect REAL RB robot?",
            (
                f"Connect to the physical right RB at {right_robot_ip}?\n\n"
                "ARC remains locked OFF. Verify the emergency stop and "
                "workspace."
            ),
            icon="warning",
        )
        if not confirmed:
            return
        self.log(f"Connecting REAL RB at {right_robot_ip}...")
        threading.Thread(
            target=self.node.set_robot_connection,
            args=(True, right_robot_ip),
            daemon=True,
        ).start()

    def disconnect_robot(self):
        confirmed = messagebox.askyesno(
            "Disconnect REAL RB robot?",
            "Stop MoveIt/ros2_control and disconnect the physical right arm?",
            icon="warning",
        )
        if not confirmed:
            return
        self.log("Disconnecting REAL RB...")
        threading.Thread(
            target=self.node.set_robot_connection,
            args=(False, self.right_robot_ip.get().strip()),
            daemon=True,
        ).start()

    def connection_change_accepted(self, connected, message):
        self.robot_connected = connected
        if connected:
            text = "ROBOT CONNECTING · REAL RB"
            color = "#b06000"
            if self.points:
                self.plan_button.configure(state=tk.NORMAL)
        else:
            text = "ROBOT DISCONNECTED · REAL RB"
            color = "#b3261e"
            self.invalidate_approved_plan()
        self.execution_mode.configure(text=text, foreground=color)
        self._update_go_initial_button()
        self.log(message)

    def robot_feedback_connected(self):
        self.robot_connected = True
        mode = (
            "EXECUTION ENABLED"
            if self.execution_allowed
            else "PLAN-ONLY"
        )
        self.execution_mode.configure(
            text=f"ROBOT CONNECTED · {mode} · REAL RB",
            foreground="#137333",
        )
        if self.points:
            self.plan_button.configure(state=tk.NORMAL)
        self._update_go_initial_button()
        self.log("Live right-arm /joint_states received")

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
            "Reading current right TCP orientation and generating "
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
            args=(None, self.show_path.get()),
            daemon=True,
        ).start()

    def capture_linear_tcp(self, endpoint_index):
        self.log(f"Reading current right TCP as TCP {endpoint_index + 1}...")
        threading.Thread(
            target=self.node.capture_linear_tcp,
            args=(endpoint_index,),
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
            args=(index, self.show_path.get()),
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
        self.log(f"{action} current right TCP · World 6D pose")

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
        self._send_path(execute_requested=False)

    def execute_approved(self):
        if not self.plan_approved:
            self.error("Plan Preview is required before execution")
            return
        if not self.execution_allowed:
            self.error("Server execution is disabled by launch configuration")
            return
        if not self.robot_connected:
            self.error("Connect the REAL RB robot first")
            return
        self._send_path(execute_requested=True)

    def _send_path(self, execute_requested):
        speed = max(0.01, min(1.0, self.velocity_percent.get() / 100.0))
        try:
            current_raw = int(self.weld_current_raw.get())
            voltage_raw = int(self.weld_voltage_raw.get())
            v_offset_raw = int(self.weld_v_offset_raw.get())
        except (ValueError, tk.TclError):
            self.error("H600 raw values must be integers")
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
                execute_requested,
                execute_requested,
            ),
            daemon=True,
        ).start()

    def update_welder_status(self, message):
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

    def begin(self, velocity_scale, execute_requested):
        self.bar["value"] = 0
        self.plan_button.configure(state=tk.DISABLED)
        self.execute_button.configure(state=tk.DISABLED)
        operation = (
            "EXECUTE exact approved plan"
            if execute_requested
            else "PLAN PREVIEW for RViz + Viser"
        )
        self.log(
            f"{operation} · "
            f"speed={velocity_scale:.0%}"
        )

    def progress(self, value, waypoint, pose):
        self.bar["value"] = value * 100
        position = pose.position
        self.feedback_label.configure(
            text=(
                f"waypoint: {waypoint + 1} · progress: {value:.0%} · "
                f"pose: ({position.x:.3f}, {position.y:.3f}, "
                f"{position.z:.3f})"
            ),
        )

    def finish(self, text, was_execution):
        self.bar["value"] = 100
        self.plan_button.configure(
            state=tk.NORMAL if self.points else tk.DISABLED
        )
        self.plan_approved = not was_execution
        self.execute_button.configure(
            state=(
                tk.NORMAL
                if self.plan_approved and self.execution_allowed
                else tk.DISABLED
            )
        )
        self.log(text)
        if self.plan_approved:
            self.log(
                "Plan approved · inspect RViz/Viser, then press "
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
        if not rclpy.ok():
            self.root.destroy()
            return
        self.root.after(200, self.check_ros)

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
