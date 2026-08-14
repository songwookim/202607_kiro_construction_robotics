import copy
import math
import queue
import signal
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, DisplayTrajectory, JointConstraint
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rbpodo_msgs.msg import SystemState
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray

from construct_msgs.action import CartesianPath
from construct_msgs.msg import WelderStatus
from construct_robot.cartesian_path_common import (
    linear_pose_waypoints,
    tip_link_for_group,
)
from construct_robot.cartesian_path_server import make_weld_visualization


TOUCH_INPUT_PORT = 0
MOVEIT_SUCCESS = 1


@dataclass
class WeldingScenarioModel:
    """ROS-independent scenario data and approval state."""

    planning_group: str = "right_manipulator"
    initial_joint_names: tuple = ()
    initial_joint_positions: tuple = ()
    initial_tcp: Pose = None
    tcp_endpoints: list = field(default_factory=lambda: [None, None])
    last_touch_pose: Pose = None
    approved_signature: tuple = None

    def reset_for_group(self, planning_group):
        self.planning_group = planning_group
        self.initial_joint_names = ()
        self.initial_joint_positions = ()
        self.initial_tcp = None
        self.tcp_endpoints = [None, None]
        self.last_touch_pose = None
        self.approved_signature = None

    def set_initial(self, joint_names, positions, tcp):
        self.initial_joint_names = tuple(joint_names)
        self.initial_joint_positions = tuple(positions)
        self.initial_tcp = copy.deepcopy(tcp)

    def set_endpoint(self, index, pose):
        self.tcp_endpoints[index] = copy.deepcopy(pose)
        self.approved_signature = None

    def record_touch(self, pose, save_endpoint):
        self.last_touch_pose = copy.deepcopy(pose)
        if not save_endpoint:
            return None
        try:
            index = self.tcp_endpoints.index(None)
        except ValueError:
            index = 1
        self.set_endpoint(index, pose)
        return index

    def path(self, unique_points):
        if any(pose is None for pose in self.tcp_endpoints):
            raise ValueError("Capture both TCP1 and TCP2 first")
        return linear_pose_waypoints(
            self.tcp_endpoints[0],
            self.tcp_endpoints[1],
            unique_points,
        )

    def signature(self, settings):
        pose_values = []
        for pose in self.tcp_endpoints:
            if pose is None:
                return None
            pose_values.extend(
                (
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
            )
        return (self.planning_group, tuple(pose_values), tuple(settings))


class WeldingScenarioNode(Node):
    """ROS interface for the guarded TCP-to-TCP welding scenario."""

    def __init__(self, ui):
        super().__init__("welding_scenario_gui")
        self.ui = ui
        self.declare_parameter("expected_execute_motion", True)
        self.declare_parameter("left_robot_ip", "192.168.1.11")
        self.declare_parameter("right_robot_ip", "192.168.1.12")
        self.latest_joint_positions = {}
        self.last_joint_state_at = {"left": None, "right": None}
        self.touch_states = {"left": None, "right": None}
        self.welder_status = None
        self.weld_goal_handle = None
        self.weld_request_is_execution = False
        self.return_trajectory = None

        self.cartesian_client = ActionClient(
            self, CartesianPath, "cartesian_path"
        )
        self.move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory"
        )
        self.trajectory_clients = {
            arm: ActionClient(
                self,
                FollowJointTrajectory,
                f"/{arm}_manipulator_controller/follow_joint_trajectory",
            )
            for arm in ("left", "right")
        }
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        transient_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/weld_path_markers", transient_qos
        )
        self.pose_publisher = self.create_publisher(
            PoseArray, "/weld_6d_poses", transient_qos
        )
        self.display_publisher = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", transient_qos
        )
        self.create_subscription(JointState, "/joint_states", self._joints, 10)
        for arm in ("left", "right"):
            self.create_subscription(
                SystemState,
                f"/{arm}_rbpodo_hardware/system_state",
                lambda message, selected=arm: self._system_state(
                    message, selected
                ),
                10,
            )
        self.create_subscription(
            WelderStatus, "/h600/status", self._welder_status, 10
        )
        self.create_timer(0.5, self._publish_readiness)
        self.ui.post(
            self.ui.configure_connections,
            self.get_parameter("expected_execute_motion").value,
            self.get_parameter("left_robot_ip").value,
            self.get_parameter("right_robot_ip").value,
        )

    def _joints(self, message):
        now = time.monotonic()
        for name, position in zip(message.name, message.position):
            if math.isfinite(position):
                self.latest_joint_positions[name] = position
        for arm in ("left", "right"):
            names = [f"{arm}_manipulator_joint{i}" for i in range(1, 7)]
            if all(name in self.latest_joint_positions for name in names):
                self.last_joint_state_at[arm] = now

    def _system_state(self, message, arm):
        active = bool(message.digital_in[TOUCH_INPUT_PORT])
        previous = self.touch_states[arm]
        self.touch_states[arm] = active
        self.ui.post(self.ui.update_touch_state, arm, active)
        if previous is not None and active and not previous:
            self.ui.post(self.ui.handle_touch_rising, arm)

    def _welder_status(self, message):
        self.welder_status = message
        self.ui.post(self.ui.update_welder_status, message)

    def _publish_readiness(self):
        now = time.monotonic()
        moveit_ready = self.cartesian_client.server_is_ready()
        execute_expected = self.get_parameter("expected_execute_motion").value
        for arm in ("left", "right"):
            feedback_ready = (
                self.last_joint_state_at[arm] is not None
                and now - self.last_joint_state_at[arm] < 2.0
            )
            controller_ready = (
                not execute_expected
                or self.trajectory_clients[arm].server_is_ready()
            )
            self.ui.post(
                self.ui.update_robot_connection,
                arm,
                feedback_ready and moveit_ready and controller_ready,
            )

    def current_tcp(self, planning_group):
        transform = self.tf_buffer.lookup_transform(
            "World",
            tip_link_for_group(planning_group),
            rclpy.time.Time(),
            timeout=Duration(seconds=1.0),
        ).transform
        pose = Pose()
        pose.position.x = transform.translation.x
        pose.position.y = transform.translation.y
        pose.position.z = transform.translation.z
        pose.orientation = transform.rotation
        return pose

    def capture_initial(self, planning_group):
        arm = planning_group.removesuffix("_manipulator")
        names = [f"{arm}_manipulator_joint{i}" for i in range(1, 7)]
        try:
            positions = [self.latest_joint_positions[name] for name in names]
            tcp = self.current_tcp(planning_group)
        except KeyError:
            self.ui.post(self.ui.error, "Measured six-joint state is unavailable")
            return
        except TransformException as error:
            self.ui.post(self.ui.error, f"TCP lookup failed: {error}")
            return
        self.ui.post(self.ui.initial_captured, names, positions, tcp)

    def capture_tcp(self, planning_group, purpose, endpoint_index=None):
        try:
            pose = self.current_tcp(planning_group)
        except TransformException as error:
            self.ui.post(self.ui.error, f"TCP lookup failed: {error}")
            return
        self.ui.post(self.ui.tcp_captured, purpose, endpoint_index, pose)

    def publish_path(self, points, visible=True):
        markers, poses = make_weld_visualization(
            points if visible else [], "World", self.get_clock().now().to_msg()
        )
        self.marker_publisher.publish(markers)
        self.pose_publisher.publish(poses)

    def send_weld(self, points, settings, execute_requested):
        if not self.cartesian_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(self.ui.error, "CartesianPath action unavailable")
            return
        goal = CartesianPath.Goal()
        goal.waypoints = points
        goal.planning_group = settings["planning_group"]
        goal.interpolation_step = settings["interpolation_step"]
        goal.velocity_scale = settings["velocity_scale"]
        goal.execute_requested = execute_requested
        goal.reuse_approved_plan = execute_requested
        goal.visualize_path = True
        goal.enable_arc = True
        goal.weld_current_a = float(settings["current_raw"])
        goal.weld_voltage_out_condition = 1
        goal.weld_voltage = float(settings["voltage_raw"]) / 10.0
        goal.weld_initial_wait = settings["preflow"]
        goal.weld_finish_wait = settings["postflow"]
        goal.require_welding_feedback = settings["require_feedback"]
        self.weld_request_is_execution = execute_requested
        self.ui.post(
            self.ui.waiting,
            "Executing guarded welding sequence"
            if execute_requested
            else "Planning welding sequence for RViz",
        )
        future = self.cartesian_client.send_goal_async(
            goal, feedback_callback=self._weld_feedback
        )
        future.add_done_callback(self._weld_goal_response)

    def _weld_feedback(self, message):
        feedback = message.feedback
        self.ui.post(
            self.ui.weld_feedback,
            feedback.progress,
            feedback.phase,
            feedback.waypoint_index,
        )

    def _weld_goal_response(self, future):
        try:
            self.weld_goal_handle = future.result()
        except Exception as error:
            self.ui.post(self.ui.error, f"Weld action failed: {error}")
            return
        if not self.weld_goal_handle.accepted:
            self.ui.post(self.ui.error, "Weld action goal rejected")
            return
        self.weld_goal_handle.get_result_async().add_done_callback(
            self._weld_result
        )

    def _weld_result(self, future):
        try:
            result = future.result().result
        except Exception as error:
            self.ui.post(self.ui.error, f"Weld action failed: {error}")
            return
        self.ui.post(
            self.ui.weld_result,
            result.success,
            result.message,
            self.weld_request_is_execution,
        )

    @staticmethod
    def _joint_constraints(joint_names, positions):
        constraints = Constraints()
        for name, position in zip(joint_names, positions):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = position
            joint.tolerance_above = 0.001
            joint.tolerance_below = 0.001
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        return constraints

    def plan_return(self, planning_group, joint_names, positions):
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(self.ui.error, "MoveGroup action unavailable")
            return
        goal = MoveGroup.Goal()
        goal.request.group_name = planning_group
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints.append(
            self._joint_constraints(joint_names, positions)
        )
        goal.planning_options.plan_only = True
        self.ui.post(self.ui.waiting, "Planning return to initial joint state")
        future = self.move_group_client.send_goal_async(goal)
        future.add_done_callback(self._return_goal_response)

    def _return_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as error:
            self.ui.post(self.ui.error, f"Return plan failed: {error}")
            return
        if not handle.accepted:
            self.ui.post(self.ui.error, "Return plan rejected")
            return
        handle.get_result_async().add_done_callback(self._return_plan_result)

    def _return_plan_result(self, future):
        try:
            result = future.result().result
        except Exception as error:
            self.ui.post(self.ui.error, f"Return plan failed: {error}")
            return
        if result.error_code.val != MOVEIT_SUCCESS:
            self.ui.post(
                self.ui.error,
                f"Return plan failed with MoveIt code {result.error_code.val}",
            )
            return
        self.return_trajectory = copy.deepcopy(result.planned_trajectory)
        display = DisplayTrajectory()
        display.trajectory_start = result.trajectory_start
        display.trajectory.append(result.planned_trajectory)
        self.display_publisher.publish(display)
        self.ui.post(self.ui.return_plan_result, True)

    def execute_return(self):
        if self.return_trajectory is None:
            self.ui.post(self.ui.error, "Plan the return trajectory first")
            return
        if not self.execute_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(self.ui.error, "ExecuteTrajectory action unavailable")
            return
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = copy.deepcopy(self.return_trajectory)
        self.return_trajectory = None
        self.ui.post(self.ui.waiting, "Moving to captured initial position")
        future = self.execute_client.send_goal_async(goal)
        future.add_done_callback(self._execute_return_response)

    def _execute_return_response(self, future):
        try:
            handle = future.result()
        except Exception as error:
            self.ui.post(self.ui.error, f"Return execution failed: {error}")
            return
        if not handle.accepted:
            self.ui.post(self.ui.error, "Return execution rejected")
            return
        handle.get_result_async().add_done_callback(self._execute_return_result)

    def _execute_return_result(self, future):
        try:
            result = future.result().result
        except Exception as error:
            self.ui.post(self.ui.error, f"Return execution failed: {error}")
            return
        if result.error_code.val != MOVEIT_SUCCESS:
            self.ui.post(
                self.ui.error,
                f"Return execution failed with code {result.error_code.val}",
            )
            return
        self.ui.post(self.ui.return_execution_result)

    def cancel(self):
        if self.weld_goal_handle is not None:
            self.weld_goal_handle.cancel_goal_async()


class WeldingScenarioGui:
    """Step-by-step welding operation GUI with explicit plan/execute gates."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Welding Scenario")
        self.root.geometry("1180x820")
        self.queue = queue.SimpleQueue()
        self.closing = False
        self.model = WeldingScenarioModel()
        self.robot_connected = {"left": False, "right": False}
        self.robot_ips = {"left": "", "right": ""}
        self.execution_allowed = False
        self.h600_connected = False
        self.weld_plan_approved = False
        self.return_plan_approved = False

        self.planning_group = tk.StringVar(value="right_manipulator")
        self.touch_sensing = tk.BooleanVar(value=False)
        self.save_touch_tcp = tk.BooleanVar(value=True)
        self.unique_points = tk.IntVar(value=20)
        self.current_raw = tk.IntVar(value=0)
        self.voltage_raw = tk.IntVar(value=0)
        self.v_offset_raw = tk.IntVar(value=0)
        self.preflow = tk.DoubleVar(value=0.5)
        self.postflow = tk.DoubleVar(value=0.5)
        self.require_feedback = tk.BooleanVar(value=True)
        self.velocity_percent = tk.DoubleVar(value=15.0)
        self.interpolation_step_mm = tk.DoubleVar(value=2.0)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 18, "bold"))
        style.configure("Step.TLabelframe.Label", font=("Sans", 11, "bold"))
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer, text="TCP-to-TCP Welding Scenario", style="Title.TLabel"
        ).pack(anchor=tk.W)

        status = ttk.LabelFrame(outer, text="Connections")
        status.pack(fill=tk.X, pady=(8, 5))
        self.connection_labels = {}
        for arm in ("right", "left"):
            label = tk.Label(
                status,
                text=f"Connect {arm.upper()} (IP): X",
                width=32,
                relief=tk.SOLID,
                borderwidth=1,
                bg="#fce8e6",
                fg="#b3261e",
                font=("Sans", 10, "bold"),
            )
            label.pack(side=tk.LEFT, padx=5, pady=5)
            self.connection_labels[arm] = label
        self.h600_label = tk.Label(
            status,
            text="H600: X",
            width=30,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#fce8e6",
            fg="#b3261e",
            font=("Sans", 10, "bold"),
        )
        self.h600_label.pack(side=tk.LEFT, padx=5, pady=5)

        step0 = ttk.LabelFrame(outer, text="0 · Initial position", style="Step.TLabelframe")
        step0.pack(fill=tk.X, pady=4)
        ttk.Label(step0, text="arm").pack(side=tk.LEFT, padx=(6, 3), pady=6)
        arm_box = ttk.Combobox(
            step0,
            textvariable=self.planning_group,
            values=("right_manipulator", "left_manipulator"),
            state="readonly",
            width=22,
        )
        arm_box.pack(side=tk.LEFT, padx=(0, 8))
        arm_box.bind("<<ComboboxSelected>>", self.group_changed)
        ttk.Button(
            step0,
            text="Set current pose as initial",
            command=self.capture_initial,
        ).pack(side=tk.LEFT, padx=4)
        self.initial_label = ttk.Label(step0, text="not captured")
        self.initial_label.pack(side=tk.LEFT, padx=10)

        step1 = ttk.LabelFrame(
            outer,
            text="1 · TCP1 → TCP2 and touch sensing",
            style="Step.TLabelframe",
        )
        step1.pack(fill=tk.X, pady=4)
        ttk.Button(
            step1, text="Capture TCP1", command=lambda: self.capture_endpoint(0)
        ).grid(row=0, column=0, padx=5, pady=5)
        self.tcp_labels = [
            ttk.Label(step1, text="TCP1: not captured"),
            ttk.Label(step1, text="TCP2: not captured"),
        ]
        self.tcp_labels[0].grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Button(
            step1, text="Capture TCP2", command=lambda: self.capture_endpoint(1)
        ).grid(row=0, column=2, padx=5, pady=5)
        self.tcp_labels[1].grid(row=0, column=3, padx=5, sticky=tk.W)
        ttk.Checkbutton(
            step1, text="Use DI0 touch sensing", variable=self.touch_sensing
        ).grid(row=1, column=0, columnspan=2, padx=5, sticky=tk.W)
        ttk.Checkbutton(
            step1,
            text="Save touched TCP pose as next TCP endpoint",
            variable=self.save_touch_tcp,
        ).grid(row=1, column=2, columnspan=2, padx=5, sticky=tk.W)
        self.touch_label = ttk.Label(step1, text="DI0 touch: waiting")
        self.touch_label.grid(row=2, column=0, columnspan=3, padx=5, pady=4, sticky=tk.W)
        ttk.Label(step1, text="linear unique points").grid(row=2, column=3, sticky=tk.E)
        ttk.Spinbox(
            step1, from_=2, to=200, textvariable=self.unique_points, width=6
        ).grid(row=2, column=4, padx=5)

        step2 = ttk.LabelFrame(outer, text="2 · Welding parameters", style="Step.TLabelframe")
        step2.pack(fill=tk.X, pady=4)
        for column, (label, variable, upper) in enumerate(
            (
                ("current raw", self.current_raw, 65535),
                ("voltage raw", self.voltage_raw, 65535),
                ("V offset raw", self.v_offset_raw, 65535),
                ("pre-flow s", self.preflow, 10),
                ("post-flow s", self.postflow, 10),
            )
        ):
            ttk.Label(step2, text=label).grid(row=0, column=column * 2, padx=(6, 2), pady=6)
            ttk.Spinbox(
                step2, from_=0, to=upper, textvariable=variable, width=7
            ).grid(row=0, column=column * 2 + 1, padx=(0, 6))
        ttk.Checkbutton(
            step2,
            text="Require H600 welding ON/OFF feedback",
            variable=self.require_feedback,
        ).grid(row=1, column=0, columnspan=5, padx=6, sticky=tk.W)

        step3 = ttk.LabelFrame(
            outer,
            text="3 · ARC ON + linear welding motion + ARC OFF",
            style="Step.TLabelframe",
        )
        step3.pack(fill=tk.X, pady=4)
        ttk.Label(step3, text="velocity").pack(side=tk.LEFT, padx=(6, 2), pady=6)
        ttk.Scale(
            step3,
            from_=1,
            to=100,
            variable=self.velocity_percent,
            command=self.settings_changed,
            length=170,
        ).pack(side=tk.LEFT, padx=4)
        self.speed_label = ttk.Label(step3, text="15%")
        self.speed_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(step3, text="interpolation mm").pack(side=tk.LEFT)
        ttk.Spinbox(
            step3,
            from_=0.5,
            to=20,
            increment=0.5,
            textvariable=self.interpolation_step_mm,
            width=6,
        ).pack(side=tk.LEFT, padx=4)
        self.plan_weld_button = ttk.Button(
            step3, text="Plan welding path in RViz", command=self.plan_weld
        )
        self.plan_weld_button.pack(side=tk.LEFT, padx=6)
        self.execute_weld_button = ttk.Button(
            step3,
            text="Execute welding scenario",
            command=self.execute_weld,
            state=tk.DISABLED,
        )
        self.execute_weld_button.pack(side=tk.LEFT, padx=6)

        step4 = ttk.LabelFrame(
            outer,
            text="4 · Return to initial position",
            style="Step.TLabelframe",
        )
        step4.pack(fill=tk.X, pady=4)
        self.plan_return_button = ttk.Button(
            step4,
            text="Plan return in RViz",
            command=self.plan_return,
            state=tk.DISABLED,
        )
        self.plan_return_button.pack(side=tk.LEFT, padx=6, pady=6)
        self.execute_return_button = ttk.Button(
            step4,
            text="Move to initial position",
            command=self.execute_return,
            state=tk.DISABLED,
        )
        self.execute_return_button.pack(side=tk.LEFT, padx=6)

        feedback = ttk.LabelFrame(outer, text="Scenario feedback")
        feedback.pack(fill=tk.X, pady=(8, 4))
        self.progress = ttk.Progressbar(feedback, maximum=100)
        self.progress.pack(fill=tk.X, padx=6, pady=(6, 3))
        self.phase_label = ttk.Label(feedback, text="phase: –")
        self.phase_label.pack(anchor=tk.W, padx=6, pady=(0, 5))
        self.pipeline_label = tk.Label(
            outer,
            text="WAITING · capture initial position",
            anchor=tk.W,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#eeeeee",
            font=("Sans", 10, "bold"),
        )
        self.pipeline_label.pack(fill=tk.X, ipady=6)
        ttk.Button(
            outer,
            text="Request cancel (not an emergency stop)",
            command=self.cancel,
        ).pack(anchor=tk.E, pady=5)

        for variable in (
            self.unique_points,
            self.current_raw,
            self.voltage_raw,
            self.v_offset_raw,
            self.preflow,
            self.postflow,
            self.require_feedback,
            self.interpolation_step_mm,
        ):
            variable.trace_add("write", self.settings_changed)

        self.node = WeldingScenarioNode(self)
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(
            target=self.executor.spin, daemon=True
        )
        self.executor_thread.start()
        signal.signal(
            signal.SIGINT, lambda _signum, _frame: self.root.after(0, self.close)
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.check_ros)

    def post(self, callback, *args):
        self.queue.put((callback, args))

    def check_ros(self):
        while True:
            try:
                callback, args = self.queue.get_nowait()
            except queue.Empty:
                break
            callback(*args)
        if rclpy.ok() and not self.closing:
            self.root.after(100, self.check_ros)

    def configure_connections(self, execution_allowed, left_ip, right_ip):
        self.execution_allowed = execution_allowed
        self.robot_ips = {"left": left_ip, "right": right_ip}
        self.waiting(f"Connecting RIGHT {right_ip} and LEFT {left_ip}")

    def selected_arm(self):
        return self.planning_group.get().removesuffix("_manipulator")

    def update_robot_connection(self, arm, connected):
        self.robot_connected[arm] = connected
        self.connection_labels[arm].configure(
            text=f"Connect {arm.upper()} ({self.robot_ips[arm]}): {'O' if connected else 'X'}",
            bg="#e6f4ea" if connected else "#fce8e6",
            fg="#137333" if connected else "#b3261e",
        )

    def update_welder_status(self, message):
        self.h600_connected = bool(
            message.server_running and message.client_connected
        )
        detail = (
            f"O · {message.client_address} · welding={message.welding}"
            if self.h600_connected
            else "X · waiting for TCP/502 client"
        )
        self.h600_label.configure(
            text=f"H600: {detail}",
            bg="#e6f4ea" if self.h600_connected else "#fce8e6",
            fg="#137333" if self.h600_connected else "#b3261e",
        )

    def set_pipeline(self, state, message):
        colors = {
            "WAITING": ("#eeeeee", "#202124"),
            "ERROR": ("#fce8e6", "#b3261e"),
            "RESULT": ("#e6f4ea", "#137333"),
        }
        background, foreground = colors[state]
        self.pipeline_label.configure(
            text=f"{state} · {message}", bg=background, fg=foreground
        )

    def waiting(self, message):
        self.set_pipeline("WAITING", message)

    def error(self, message):
        self.set_pipeline("ERROR", message)

    def result(self, message):
        self.set_pipeline("RESULT", message)

    def group_changed(self, _event=None):
        self.model.reset_for_group(self.planning_group.get())
        self.initial_label.configure(text="not captured")
        for index, label in enumerate(self.tcp_labels, 1):
            label.configure(text=f"TCP{index}: not captured")
        self.invalidate_weld_plan()
        self.plan_return_button.configure(state=tk.DISABLED)
        self.execute_return_button.configure(state=tk.DISABLED)
        self.node.return_trajectory = None
        self.node.publish_path([])
        self.waiting("Arm changed; capture initial position and TCP endpoints")

    def capture_initial(self):
        self.waiting("Capturing current TCP and measured joint angles")
        threading.Thread(
            target=self.node.capture_initial,
            args=(self.planning_group.get(),),
            daemon=True,
        ).start()

    def initial_captured(self, names, positions, tcp):
        self.model.set_initial(names, positions, tcp)
        angles = ", ".join(f"{math.degrees(value):.1f}°" for value in positions)
        self.initial_label.configure(text=f"joints: {angles}")
        self.return_plan_approved = False
        self.node.return_trajectory = None
        self.plan_return_button.configure(state=tk.DISABLED)
        self.execute_return_button.configure(state=tk.DISABLED)
        self.result("Initial joint state and TCP captured")

    def capture_endpoint(self, index):
        self.waiting(f"Capturing TCP{index + 1}")
        threading.Thread(
            target=self.node.capture_tcp,
            args=(self.planning_group.get(), "endpoint", index),
            daemon=True,
        ).start()

    def tcp_captured(self, purpose, endpoint_index, pose):
        if purpose == "touch":
            saved = self.model.record_touch(pose, self.save_touch_tcp.get())
            if saved is None:
                self.result("Touch TCP captured without changing endpoints")
                return
            endpoint_index = saved
        self.model.set_endpoint(endpoint_index, pose)
        position = pose.position
        self.tcp_labels[endpoint_index].configure(
            text=(
                f"TCP{endpoint_index + 1}: ({position.x:.4f}, "
                f"{position.y:.4f}, {position.z:.4f})"
            )
        )
        self.invalidate_weld_plan()
        self.publish_preview_path()
        self.result(f"TCP{endpoint_index + 1} captured")

    def update_touch_state(self, arm, active):
        if arm == self.selected_arm():
            self.touch_label.configure(
                text=f"{arm.upper()} DI0 touch: {'ON' if active else 'OFF'}"
            )

    def handle_touch_rising(self, arm):
        if not self.touch_sensing.get() or arm != self.selected_arm():
            return
        self.waiting(f"{arm.upper()} DI0 TOUCH detected; capturing TCP")
        self.root.bell()
        threading.Thread(
            target=self.node.capture_tcp,
            args=(self.planning_group.get(), "touch", None),
            daemon=True,
        ).start()

    def settings(self):
        try:
            settings = {
                "planning_group": self.planning_group.get(),
                "velocity_scale": max(
                    0.01, min(1.0, float(self.velocity_percent.get()) / 100.0)
                ),
                "interpolation_step": float(self.interpolation_step_mm.get())
                * 0.001,
                "current_raw": int(self.current_raw.get()),
                "voltage_raw": int(self.voltage_raw.get()),
                "v_offset_raw": int(self.v_offset_raw.get()),
                "preflow": float(self.preflow.get()),
                "postflow": float(self.postflow.get()),
                "require_feedback": bool(self.require_feedback.get()),
            }
        except (ValueError, tk.TclError) as error:
            raise ValueError("Welding parameters must be numeric") from error
        if not 0.0005 <= settings["interpolation_step"] <= 0.02:
            raise ValueError("Interpolation step must be 0.5..20 mm")
        if not 0.0 <= settings["preflow"] <= 10.0:
            raise ValueError("Pre-flow must be 0..10 seconds")
        if not 0.0 <= settings["postflow"] <= 10.0:
            raise ValueError("Post-flow must be 0..10 seconds")
        return settings

    def settings_tuple(self, settings):
        return (
            settings["velocity_scale"],
            settings["interpolation_step"],
            settings["current_raw"],
            settings["voltage_raw"],
            settings["v_offset_raw"],
            settings["preflow"],
            settings["postflow"],
            settings["require_feedback"],
            int(self.unique_points.get()),
        )

    def settings_changed(self, _value=None, *_args):
        if hasattr(self, "speed_label"):
            try:
                self.speed_label.configure(
                    text=f"{float(self.velocity_percent.get()):.0f}%"
                )
            except (ValueError, tk.TclError):
                pass
        self.invalidate_weld_plan()

    def invalidate_weld_plan(self):
        self.weld_plan_approved = False
        self.model.approved_signature = None
        if hasattr(self, "execute_weld_button"):
            self.execute_weld_button.configure(state=tk.DISABLED)

    def publish_preview_path(self):
        try:
            points = self.model.path(int(self.unique_points.get()))
        except (ValueError, tk.TclError):
            return
        self.node.publish_path(points)

    def plan_weld(self):
        if not self.robot_connected[self.selected_arm()]:
            self.error("Selected robot is not connected")
            return
        try:
            settings = self.settings()
            points = self.model.path(int(self.unique_points.get()))
        except (ValueError, tk.TclError) as error:
            self.error(str(error))
            return
        signature = self.model.signature(self.settings_tuple(settings))
        self.node.publish_path(points)
        self.model.approved_signature = signature
        threading.Thread(
            target=self.node.send_weld,
            args=(points, settings, False),
            daemon=True,
        ).start()

    def execute_weld(self):
        if not self.execution_allowed:
            self.error("Execution is disabled by launch configuration")
            return
        if not self.robot_connected[self.selected_arm()]:
            self.error("Selected robot is not connected")
            return
        if not self.h600_connected:
            self.error("H600 TCP/502 client is not connected")
            return
        try:
            settings = self.settings()
            points = self.model.path(int(self.unique_points.get()))
        except (ValueError, tk.TclError) as error:
            self.error(str(error))
            return
        if self.model.signature(self.settings_tuple(settings)) != self.model.approved_signature:
            self.error("Settings or TCP poses changed; plan again in RViz")
            return
        if not messagebox.askyesno(
            "Execute physical welding",
            "Execute the approved TCP1 approach, ARC ON weld to TCP2, "
            "and ARC OFF sequence on the real robot?",
        ):
            self.waiting("Welding execution canceled")
            return
        threading.Thread(
            target=self.node.send_weld,
            args=(points, settings, True),
            daemon=True,
        ).start()

    def weld_feedback(self, progress, phase, waypoint):
        self.progress["value"] = progress * 100.0
        self.phase_label.configure(
            text=f"phase: {phase or 'PATH'} · waypoint {waypoint + 1}"
        )

    def weld_result(self, success, message, was_execution):
        if not success:
            self.invalidate_weld_plan()
            if was_execution and self.model.initial_joint_names:
                self.plan_return_button.configure(state=tk.NORMAL)
            self.error(message)
            return
        if was_execution:
            self.invalidate_weld_plan()
            self.plan_return_button.configure(state=tk.NORMAL)
            self.result(f"Welding complete and ARC safe-off · {message}")
        else:
            self.weld_plan_approved = True
            self.execute_weld_button.configure(state=tk.NORMAL)
            self.result(f"Welding plan approved in RViz · {message}")

    def plan_return(self):
        if not self.model.initial_joint_names:
            self.error("Capture the initial position first")
            return
        threading.Thread(
            target=self.node.plan_return,
            args=(
                self.planning_group.get(),
                self.model.initial_joint_names,
                self.model.initial_joint_positions,
            ),
            daemon=True,
        ).start()

    def return_plan_result(self, success):
        self.return_plan_approved = success
        self.execute_return_button.configure(
            state=tk.NORMAL if success else tk.DISABLED
        )
        self.result("Return trajectory approved and shown in RViz")

    def execute_return(self):
        if not self.return_plan_approved:
            self.error("Plan the return trajectory first")
            return
        if not self.execution_allowed:
            self.error("Execution is disabled by launch configuration")
            return
        if not messagebox.askyesno(
            "Move to initial position",
            "Execute the RViz-approved return trajectory on the real robot?",
        ):
            self.waiting("Return execution canceled")
            return
        self.return_plan_approved = False
        self.execute_return_button.configure(state=tk.DISABLED)
        threading.Thread(target=self.node.execute_return, daemon=True).start()

    def return_execution_result(self):
        self.result("Robot returned to the captured initial position")

    def cancel(self):
        self.node.cancel()
        self.waiting(
            "Cancel requested; use the robot emergency stop for immediate stop"
        )

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()

    def shutdown_ros(self):
        self.executor.shutdown(timeout_sec=2.0)
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.executor_thread.join(timeout=1.0)


def main(args=None):
    rclpy.init(args=args)
    gui = WeldingScenarioGui()
    try:
        gui.mainloop()
    finally:
        gui.shutdown_ros()
