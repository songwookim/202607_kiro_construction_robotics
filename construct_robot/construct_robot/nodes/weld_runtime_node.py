"""ROS 2 runtime node used by the Tkinter weld production GUI.

Moved verbatim from ``construct_robot.gui.weld_action_gui``.  The node still
reports back to the GUI through ``self.ui.post(...)``; that relationship is
unchanged.
"""

import copy
import math
import threading
import time

import rclpy
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Point, Pose, PoseArray, TransformStamped
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, DisplayTrajectory, JointConstraint
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rbpodo_msgs.msg import SystemState
from rbpodo_msgs.srv import MoveStop, SetDigitalOutput, SetRobotPower
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float64MultiArray, UInt8
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from wide_sensing_msgs.msg import WideSensingResult

from construct_msgs.action import CartesianPath
from construct_msgs.msg import DigitalIoState
from construct_msgs.srv import SetDigitalOutput as FastechSetDigitalOutput
from construct_robot.core.cartesian_path_common import (
    _quaternion_rotate_vector,
    circle_waypoints,
    circular_weaving_from_path,
    linear_pose_waypoints,
    midpoint_pose,
    named_tcp_linear_waypoints,
    pose_is_valid,
    pose_with_rpy_offset,
    quaternion_angular_distance,
    scale_trajectory_speed,
    straight_waypoints,
    tip_link_for_group,
    weaving_from_path,
)
from construct_robot.nodes.cartesian_path_server import make_weld_visualization
from construct_robot.core.keyboard_jog import keyboard_velocity_vector
from construct_robot.nodes.keyboard_servo import KeyboardServoBridge
from construct_robot.core.seam_geometry import (
    _pose_position_tuple,
    _unit_vector,
    wide_sensing_path_poses,
)
from construct_robot.core.task_teaching_model import (
    TCP_POSE_TEACHING_POSES,
    TOUCH_GUARDED_TEACHING_POSES,
)
from construct_robot.io.teaching_yaml import ARM_JOINT_NAMES


FASTECH_TOUCH_INPUT_PORT = 4
FASTECH_TOUCH_OUTPUT_PORT = 0

# Kept for the Controller Digital I/O test panel and later legacy inspection.
# Production touch sensing no longer consumes these Rainbow ports.
LEGACY_RAINBOW_TOUCH_INPUT_PORT = 8
LEGACY_RAINBOW_TOUCH_OUTPUT_PORT = 4

HEAD_JOINT_NAME_ORDER = (
    "robot_head_rev_joint1",
    "robot_head_rev_joint2",
)
HEAD_JOINT_NAMES = frozenset(HEAD_JOINT_NAME_ORDER)
CONTROLLED_JOINT_NAMES = {
    **ARM_JOINT_NAMES,
    "head": HEAD_JOINT_NAMES,
}
CONTROLLER_NAMES = {
    "left": "left_manipulator_controller",
    "right": "right_manipulator_controller",
    "head": "robot_head_controller",
}
KEYBOARD_VELOCITY_CONTROLLER_NAMES = {
    "left": "left_cartesian_velocity_controller",
    "right": "right_cartesian_velocity_controller",
}

# The RB jog command is latched in the controller.  Keep a ROS-thread
# deadman independent of Tk so a delayed/missed KeyRelease cannot leave it
# running.  Tk refreshes this lease only while a direction key is held.
KEYBOARD_VELOCITY_DEADMAN_TIMEOUT_S = 0.25
KEYBOARD_VELOCITY_INITIAL_DEADMAN_TIMEOUT_S = 0.80
KEYBOARD_ZERO_BURST_COUNT = 5
KEYBOARD_TF_LOOKUP_TIMEOUT_S = 0.05
# "servo": keyboard_jog_node streams into the active JTC (RB stays in Servo-J).
# "native_jog": exchange JTC for the Cartesian velocity controller (jog_robot_l).
KEYBOARD_TEACHING_BACKENDS = ("servo", "native_jog")


class WeldGuiNode(Node):
    """ROS interface used by the editable weld-path GUI."""

    def __init__(self, ui):
        super().__init__("weld_action_gui")
        self.ui = ui
        self.declare_parameter("expected_execute_motion", True)
        self.declare_parameter("robot_feedback_timeout", 5.0)
        self.declare_parameter("left_robot_ip", "192.168.1.11")
        self.declare_parameter("right_robot_ip", "192.168.1.12")
        self.declare_parameter("use_fake_head_hardware", False)
        self.declare_parameter("hicomm_source_ip", "192.168.1.2")
        self.declare_parameter("hicomm_welder_ip", "192.168.1.10")
        self.declare_parameter("hicomm_port", 60000)
        self.declare_parameter("fastech_ip", "192.168.0.3")
        self.declare_parameter("fastech_board_id", 0)
        self.declare_parameter("fastech_poll_period_s", 0.01)
        self.declare_parameter(
            "wide_sensing_result_topic",
            "/wide_sensing/output/result",
        )
        self.declare_parameter("keyboard_teaching_backend", "servo")
        self.cartesian_motion_client = ActionClient(
            self, CartesianPath, "cartesian_path"
        )
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
        self.cartesian_planning_client = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
        )
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.declare_parameter("teaching_fk_tf_position_tolerance_mm", 1.0)
        self.declare_parameter("teaching_fk_tf_orientation_tolerance_deg", 0.5)
        self.declare_parameter("teaching_joint_state_max_age_ms", 250.0)
        self.declare_parameter("teaching_tf_max_age_ms", 250.0)
        self.joint_trajectory_clients = {
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
            "head": ActionClient(
                self,
                FollowJointTrajectory,
                "/robot_head_controller/follow_joint_trajectory",
            ),
        }
        self.keyboard_velocity_publishers = {
            arm: self.create_publisher(
                Float64MultiArray,
                f"/{KEYBOARD_VELOCITY_CONTROLLER_NAMES[arm]}/commands",
                1,
            )
            for arm in ("left", "right")
        }
        self.keyboard_velocity_lock = threading.Lock()
        self.keyboard_velocity_command = {
            "arm": None,
            "values": (0.0,) * 6,
            "refreshed_monotonic": time.monotonic(),
            "deadman_timeout_s": KEYBOARD_VELOCITY_INITIAL_DEADMAN_TIMEOUT_S,
            "zero_burst_remaining": 0,
        }
        self.create_timer(0.02, self._publish_keyboard_velocity)
        self.joint_trajectory_cancel_clients = {
            device: self.create_client(
                CancelGoal,
                f"/{CONTROLLER_NAMES[device]}/follow_joint_trajectory/"
                "_action/cancel_goal",
            )
            for device in ("left", "right", "head")
        }
        self.legacy_digital_output_client = self.create_client(
            SetDigitalOutput,
            "/right_rbpodo_hardware/set_digital_output",
        )
        self.fastech_touch_enable_client = self.create_client(
            SetBool,
            "/touch/enable",
        )
        self.fastech_set_output_client = self.create_client(
            FastechSetDigitalOutput,
            "/fastech/set_output",
        )
        self.fastech_connect_client = self.create_client(
            Trigger,
            "/fastech/connect",
        )
        self.fastech_disconnect_client = self.create_client(
            Trigger,
            "/fastech/disconnect",
        )
        self.move_stop_clients = {
            arm: self.create_client(
                MoveStop,
                f"/{arm}_rbpodo_hardware/move_stop",
            )
            for arm in ("left", "right")
        }
        self.servo_hold_release_clients = {
            arm: self.create_client(
                Trigger,
                f"/{arm}_rbpodo_hardware/release_servo_hold",
            )
            for arm in ("left", "right")
        }
        self.robot_power_clients = {
            arm: self.create_client(
                SetRobotPower,
                f"/{arm}_rbpodo_hardware/set_robot_power",
            )
            for arm in ("left", "right")
        }
        self.controller_list_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
        )
        self.controller_switch_client = self.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.active_motion_goal = None
        self.active_touch_probe = None
        self.touch_probe_edge_pose = None
        self.touch_probe_stop_requested = threading.Event()
        self.touch_probe_cancel_event = threading.Event()
        self.touch_probe_controller_deactivated = False
        self.touch_stop_lock = threading.Lock()
        self.active_touch_guard = None
        self.touch_guard_stop_lock = threading.Lock()
        self.touch_guard_triggered = threading.Event()
        self.touch_guard_stop_complete = threading.Event()
        self.touch_guard_stop_success = False
        self.fastech_io_connected = False
        self.fastech_touch_input_state = None
        self.node_touch_input_states = {"left": None, "right": None}
        self.legacy_node_touch_input_states = {"left": None, "right": None}
        self.legacy_node_digital_outputs = {"left": None, "right": None}
        self.initial_planned_trajectory = None
        self.initial_planned_pose_name = None
        self.initial_planned_group = None
        self.initial_planned_target = None
        self.latest_rviz_display = None
        self.latest_rviz_display_at = None
        self.request_execution = False
        self.execute_motion_enabled = self.get_parameter(
            "expected_execute_motion"
        ).value
        controlled_devices = ("left", "right", "head")
        self.expect_robot_feedback = {
            device: True for device in controlled_devices
        }
        self.robot_feedback_seen = {
            device: False for device in controlled_devices
        }
        self.robot_ready_reported = {
            device: False for device in controlled_devices
        }
        self.controller_states = {
            device: None for device in controlled_devices
        }
        self.keyboard_controller_states = {
            arm: None for arm in ("left", "right")
        }
        self.latest_robot_motion_state = {
            arm: None for arm in ("left", "right")
        }
        self.controller_state_future = None
        self.latest_joint_positions = {}
        backend = str(self.get_parameter("keyboard_teaching_backend").value)
        if backend not in KEYBOARD_TEACHING_BACKENDS:
            raise ValueError(
                f"keyboard_teaching_backend must be one of "
                f"{KEYBOARD_TEACHING_BACKENDS}, got {backend!r}"
            )
        self.keyboard_servo = (
            KeyboardServoBridge(
                self,
                {arm: CONTROLLER_NAMES[arm] for arm in ("left", "right")},
                on_status=self._keyboard_servo_status,
            )
            if backend == "servo"
            else None
        )
        self.measured_joint_snapshot_lock = threading.Lock()
        self.measured_joint_snapshots = {}
        self.last_measured_joints_at = {}
        self.last_motion_state_at = {}
        self.last_robot_feedback_at = {
            device: None for device in controlled_devices
        }
        startup_deadline = time.monotonic() + 90.0
        self.connection_deadline = {
            device: startup_deadline for device in controlled_devices
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
        self.create_subscription(
            DisplayTrajectory,
            "/display_planned_path",
            self._display_trajectory_received,
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
        self.create_subscription(
            UInt8,
            "/keyboard_teaching/arrow_state",
            self._keyboard_arrow_state,
            10,
        )
        self.create_subscription(
            WideSensingResult,
            str(self.get_parameter("wide_sensing_result_topic").value),
            self._wide_sensing_result,
            10,
        )
        fastech_qos = QoSProfile(depth=1)
        fastech_qos.reliability = ReliabilityPolicy.RELIABLE
        fastech_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Bool,
            "/touch/contact",
            self._fastech_touch_contact,
            fastech_qos,
        )
        self.create_subscription(
            DigitalIoState,
            "/fastech/io_state",
            self._fastech_io_state,
            fastech_qos,
        )
        self.ui.post(
            self.ui.set_execution_configuration,
            self.get_parameter("expected_execute_motion").value,
            self.get_parameter("left_robot_ip").value,
            self.get_parameter("right_robot_ip").value,
            self.get_parameter("use_fake_head_hardware").value,
            self.get_parameter("hicomm_source_ip").value,
            self.get_parameter("hicomm_welder_ip").value,
            self.get_parameter("hicomm_port").value,
        )
        self.ui.post(
            self.ui.set_fastech_configuration,
            self.get_parameter("fastech_ip").value,
            self.get_parameter("fastech_board_id").value,
            self.get_parameter("fastech_poll_period_s").value,
        )

    def _keyboard_arrow_state(self, message):
        mask = int(message.data) & 0x0F
        # Safety path stays entirely in the ROS executor: do not wait for the
        # Tk queue to notice a physical release before publishing velocity
        # zero. Multiple simultaneous arrows are also treated as STOP.
        valid_single_arrow = mask in (0x01, 0x02, 0x04, 0x08)
        if not valid_single_arrow:
            with self.keyboard_velocity_lock:
                moving = any(
                    abs(value) > 1e-12
                    for value in self.keyboard_velocity_command["values"]
                )
            if moving:
                self.clear_keyboard_velocity()
                self.get_logger().info(
                    "KEYBOARD PHYSICAL RELEASE · immediate ROS velocity zero"
                )
        # Do not coalesce physical edges: a short release between two presses
        # is precisely the safety event that must never be discarded.
        self.ui.post(
            self.ui.keyboard_arrow_state_received,
            mask,
        )

    def _wide_sensing_result(self, message):
        self.ui.post(
            self.ui.update_wide_sensing_result,
            copy.deepcopy(message),
        )

    def resolve_wide_sensing_segment(
        self,
        segment,
        source_frame,
        planning_group,
        offset_m,
        reverse,
    ):
        source_frame = str(source_frame).strip()
        if not source_frame:
            raise ValueError("Wide Sensing source frame is empty")
        if source_frame == "World":
            world_from_sensor = TransformStamped()
            world_from_sensor.header.frame_id = "World"
            world_from_sensor.child_frame_id = "World"
            world_from_sensor.transform.rotation.w = 1.0
        else:
            world_from_sensor = self.tf_buffer.lookup_transform(
                "World",
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
        current_tcp = self._current_tcp_pose(planning_group)
        return wide_sensing_path_poses(
            segment.start,
            segment.end,
            world_from_sensor,
            current_tcp.orientation,
            offset_m=offset_m,
            reverse=reverse,
        )

    def resolve_keyboard_velocity(
        self,
        planning_group,
        selection,
        direction,
        linear_speed_m_s,
        angular_speed_rad_s,
        reference,
    ):
        # World-frame translation and all rotations do not need the current
        # TCP attitude.  Avoid a blocking TCP TF lookup on the common path.
        current = Pose()
        current.orientation.w = 1.0
        if str(reference).strip().lower() == "tool":
            transform = self.tf_buffer.lookup_transform(
                "World",
                tip_link_for_group(planning_group),
                rclpy.time.Time(),
                timeout=Duration(seconds=KEYBOARD_TF_LOOKUP_TIMEOUT_S),
            )
            current.orientation = transform.transform.rotation
        world_velocity = keyboard_velocity_vector(
            current.orientation,
            selection,
            direction,
            linear_speed_m_s,
            angular_speed_rad_s,
            reference,
        )
        arm = planning_group.removesuffix("_manipulator")
        base_frame = f"{arm}_manipulator_base_link"
        base_from_world = self.tf_buffer.lookup_transform(
            base_frame,
            "World",
            rclpy.time.Time(),
            timeout=Duration(seconds=KEYBOARD_TF_LOOKUP_TIMEOUT_S),
        )
        rotation = base_from_world.transform.rotation
        return (
            *_quaternion_rotate_vector(rotation, world_velocity[:3]),
            *_quaternion_rotate_vector(rotation, world_velocity[3:]),
        )

    def set_keyboard_velocity(self, arm, values):
        values = tuple(float(value) for value in values)
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError("keyboard velocity must contain six finite values")
        with self.keyboard_velocity_lock:
            self.keyboard_velocity_command = {
                "arm": arm,
                "values": values,
                "refreshed_monotonic": time.monotonic(),
                "deadman_timeout_s": KEYBOARD_VELOCITY_INITIAL_DEADMAN_TIMEOUT_S,
                "zero_burst_remaining": 0,
            }
        # jog_robot_l is latched; publish once on start/direction/speed change.
        # Re-streaming identical non-zero messages can queue stale motion ahead
        # of the release zero when the subscriber is briefly delayed.
        if arm in self.keyboard_velocity_publishers:
            self._publish_keyboard_velocity(force=True)

    def refresh_keyboard_velocity(self, arm):
        """Renew an active jog lease without changing its command."""
        with self.keyboard_velocity_lock:
            command = self.keyboard_velocity_command
            if command["arm"] != arm or not any(
                abs(value) > 1e-12 for value in command["values"]
            ):
                return False
            command["refreshed_monotonic"] = time.monotonic()
            command["deadman_timeout_s"] = KEYBOARD_VELOCITY_DEADMAN_TIMEOUT_S
            return True

    def clear_keyboard_velocity(self):
        with self.keyboard_velocity_lock:
            arm = self.keyboard_velocity_command["arm"]
            self.keyboard_velocity_command = {
                "arm": arm,
                "values": (0.0,) * 6,
                "refreshed_monotonic": time.monotonic(),
                "deadman_timeout_s": KEYBOARD_VELOCITY_INITIAL_DEADMAN_TIMEOUT_S,
                "zero_burst_remaining": KEYBOARD_ZERO_BURST_COUNT,
            }
        if arm in self.keyboard_velocity_publishers:
            self._publish_keyboard_velocity(force=True)

    def keyboard_teaching_uses_servo(self):
        return self.keyboard_servo is not None

    def keyboard_velocity_controller_ready(self, arm):
        if self.keyboard_servo is not None:
            return self.keyboard_servo.available(arm)
        return arm in self.keyboard_velocity_publishers

    def wait_until_keyboard_command_stopped(self, arm, timeout, stable_s=0.1):
        """Servo mode: the JTC reference, not RB's lagging motion, must stop."""
        return self.keyboard_servo.command_stopped(arm, timeout, stable_s)

    def restart_keyboard_servo(self, arm):
        """Servo-mode stop fallback; never RB move_stop while Servo-J streams."""
        return self.keyboard_servo.restart(arm)

    def _keyboard_servo_status(self, arm, code, text):
        message = f"{arm.upper()} keyboard Servo · {text}"
        if code in (0, 6):
            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)
        self.ui.post(self.ui.log, message)

    def keyboard_velocity_feedback_ready(self, arm, maximum_age_s=0.25):
        received_at = self.last_robot_feedback_at.get(arm)
        if received_at is None or time.monotonic() - received_at > maximum_age_s:
            return False
        return CONTROLLED_JOINT_NAMES[arm].issubset(self.latest_joint_positions)

    def wait_for_keyboard_velocity_feedback(self, arm, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.keyboard_velocity_feedback_ready(arm):
                return True
            time.sleep(0.02)
        return False

    def _publish_keyboard_velocity(self, force=False):
        expired_arm = None
        with self.keyboard_velocity_lock:
            arm = self.keyboard_velocity_command["arm"]
            values = tuple(self.keyboard_velocity_command["values"])
            refreshed = float(self.keyboard_velocity_command.get(
                "refreshed_monotonic", 0.0
            ))
            timeout_s = float(self.keyboard_velocity_command.get(
                "deadman_timeout_s", KEYBOARD_VELOCITY_DEADMAN_TIMEOUT_S
            ))
            if (
                arm in self.keyboard_velocity_publishers
                and any(abs(value) > 1e-12 for value in values)
                and time.monotonic() - refreshed
                > timeout_s
            ):
                values = (0.0,) * 6
                self.keyboard_velocity_command["values"] = values
                self.keyboard_velocity_command["refreshed_monotonic"] = time.monotonic()
                self.keyboard_velocity_command["zero_burst_remaining"] = (
                    KEYBOARD_ZERO_BURST_COUNT
                )
                expired_arm = arm
            zero_burst = int(self.keyboard_velocity_command.get(
                "zero_burst_remaining", 0
            ))
            if not any(abs(value) > 1e-12 for value in values) and zero_burst > 0:
                self.keyboard_velocity_command["zero_burst_remaining"] = zero_burst - 1
                publish_zero_burst = True
            else:
                publish_zero_burst = False
        if arm not in self.keyboard_velocity_publishers:
            return
        # Servo is not latched: it needs the twist every cycle while a key is
        # held and halts on its own when the stream stops.
        servo_streaming = self.keyboard_servo is not None and any(
            abs(value) > 1e-12 for value in values
        )
        if (
            not force
            and expired_arm is None
            and not publish_zero_burst
            and not servo_streaming
        ):
            return
        if self.keyboard_servo is not None:
            self.keyboard_servo.publish_twist(arm, values)
        else:
            message = Float64MultiArray()
            message.data = list(values)
            self.keyboard_velocity_publishers[arm].publish(message)
        if expired_arm is not None:
            self.get_logger().warning(
                f"{expired_arm.upper()} keyboard velocity deadman expired; "
                "published explicit zero"
            )
            self.ui.post(self.ui.keyboard_velocity_deadman_stopped, expired_arm)

    def _fastech_touch_contact(self, message):
        if not self.fastech_io_connected:
            return
        self.update_fastech_touch_input(message.data)

    def _fastech_io_state(self, message):
        self.fastech_io_connected = bool(message.connected)
        if not message.connected:
            self.clear_fastech_touch_input()
        self.ui.post_latest(
            "fastech_io",
            self.ui.update_fastech_io,
            message,
        )

    @staticmethod
    def _call_service_sync(client, request, service_name, timeout=3.0):
        if not client.wait_for_service(timeout_sec=1.0):
            return False, f"ROS service unavailable: {service_name}"
        completed = threading.Event()
        result = {}

        def response_ready(future):
            try:
                result["response"] = future.result()
            except Exception as error:
                result["error"] = str(error)
            finally:
                completed.set()

        client.call_async(request).add_done_callback(response_ready)
        if not completed.wait(timeout):
            return False, f"ROS service timeout: {service_name}"
        if "error" in result:
            return False, result["error"]
        response = result["response"]
        return bool(response.success), str(response.message)

    def set_fastech_output_sync(self, channel, enabled):
        if int(channel) == FASTECH_TOUCH_OUTPUT_PORT:
            request = SetBool.Request()
            request.data = bool(enabled)
            return self._call_service_sync(
                self.fastech_touch_enable_client,
                request,
                "/touch/enable",
            )
        request = FastechSetDigitalOutput.Request()
        request.channel = int(channel)
        request.value = bool(enabled)
        return self._call_service_sync(
            self.fastech_set_output_client,
            request,
            "/fastech/set_output",
        )

    def set_both_robot_power_sync(self, enable):
        """Send left and right arm-power requests before waiting for either."""
        pending = {}
        results = {}
        completed = threading.Event()
        result_lock = threading.Lock()
        for arm, client in self.robot_power_clients.items():
            service = f"/{arm}_rbpodo_hardware/set_robot_power"
            if not client.wait_for_service(timeout_sec=1.0):
                results[arm] = (False, f"ROS service unavailable: {service}")
                continue
            request = SetRobotPower.Request()
            request.enable = bool(enable)
            pending[arm] = client.call_async(request)
        if not pending:
            return results

        def response_ready(arm, future):
            try:
                response = future.result()
                value = (bool(response.success), str(response.message))
            except Exception as error:
                value = (False, str(error))
            with result_lock:
                results[arm] = value
                if all(name in results for name in pending):
                    completed.set()

        for arm, future in pending.items():
            future.add_done_callback(
                lambda done, selected=arm: response_ready(selected, done)
            )
        timeout = 35.0 if enable else 8.0
        if not completed.wait(timeout):
            for arm in pending:
                results.setdefault(arm, (False, "robot power service timeout"))
        return results

    def set_fastech_connection_sync(self, connect):
        client = (
            self.fastech_connect_client
            if connect
            else self.fastech_disconnect_client
        )
        service_name = "/fastech/connect" if connect else "/fastech/disconnect"
        return self._call_service_sync(
            client,
            Trigger.Request(),
            service_name,
            timeout=5.0,
        )

    def _system_state(self, message, arm):
        """Retain Rainbow controller I/O as a legacy monitor/test path."""
        self.latest_robot_motion_state[arm] = int(message.robot_state)
        self.last_motion_state_at[arm] = time.monotonic()
        self.legacy_node_touch_input_states[arm] = bool(
            message.digital_in[LEGACY_RAINBOW_TOUCH_INPUT_PORT]
        )
        self.legacy_node_digital_outputs[arm] = tuple(message.digital_out)
        if arm == "right":
            self.ui.post_latest(
                "right_control_box_io",
                self.ui.update_control_box_io,
                tuple(message.digital_in),
                tuple(message.digital_out),
            )
        if not self.expect_robot_feedback[arm]:
            return
        self.last_robot_feedback_at[arm] = time.monotonic()
        self.robot_feedback_seen[arm] = True

    def update_fastech_touch_input(self, touch_active):
        """Consume the production Fastech touch edge from /touch/contact."""
        touch_active = bool(touch_active)
        probe = self.active_touch_probe
        guard = self.active_touch_guard
        arm = (
            probe[0]
            if probe is not None
            else guard[0]
            if guard is not None
            else "right"
        )
        previous_touch = self.fastech_touch_input_state
        self.fastech_touch_input_state = touch_active
        self.node_touch_input_states["left"] = touch_active
        self.node_touch_input_states["right"] = touch_active
        if (
            probe is not None
            and probe[0] == arm
            and previous_touch is False
            and touch_active
            and not self.touch_probe_stop_requested.is_set()
        ):
            # Latch before starting the worker.  Contact inputs can bounce
            # OFF/ON while the tool settles; only the first edge belongs to
            # this probe.
            self.touch_probe_stop_requested.set()
            try:
                # Latch the TCP at the Fastech DI0 edge. Waiting for measured
                # standstill before reading TF records braking overshoot as if
                # it were the physical contact point, especially along Y.
                self.touch_probe_edge_pose = self._current_tcp_pose(probe[2])
                self.ui.post(
                    self.ui.apply_touch_edge_capture,
                    copy.deepcopy(self.touch_probe_edge_pose),
                    probe[2],
                    probe[1],
                    copy.deepcopy(probe[3]),
                )
            except TransformException as error:
                self.touch_probe_edge_pose = None
                self.ui.post(
                    self.ui.log,
                    "Fastech DI0 edge TCP latch failed; stopped TCP will be "
                    f"used: {error}",
                )
            self.get_logger().warning(
                f"Fastech DI{FASTECH_TOUCH_INPUT_PORT} rising edge · {arm} · "
                f"stopping active probe {probe[1]}"
            )
            threading.Thread(
                target=self.stop_touch_probe_and_capture,
                daemon=True,
            ).start()
        if (
            guard is not None
            and guard[0] == arm
            and previous_touch is False
            and touch_active
        ):
            threading.Thread(
                target=self.stop_touch_guarded_motion,
                daemon=True,
            ).start()
        if previous_touch is None or previous_touch != touch_active:
            self.ui.post(
                self.ui.update_touch_input,
                arm,
                touch_active,
            )

    def clear_fastech_touch_input(self):
        """Invalidate production touch state after Ethernet disconnect."""
        self.fastech_touch_input_state = None
        self.node_touch_input_states["left"] = None
        self.node_touch_input_states["right"] = None

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
        for arm, expected_names in CONTROLLED_JOINT_NAMES.items():
            if expected_names.issubset(positions) and all(
                math.isfinite(positions[name]) for name in expected_names
            ):
                self.last_robot_feedback_at[arm] = received_at
                self.last_measured_joints_at[arm] = received_at
                self.robot_feedback_seen[arm] = True
                with self.measured_joint_snapshot_lock:
                    self.measured_joint_snapshots[arm] = {
                        "positions": {name: float(positions[name]) for name in expected_names},
                        "received_monotonic": received_at,
                        "stamp_sec": (float(message.header.stamp.sec)
                                      + float(message.header.stamp.nanosec) * 1e-9),
                    }

    def _display_trajectory_received(self, message):
        """Keep the latest non-empty trajectory displayed by MoveIt/RViz."""
        if not message.trajectory:
            return
        if not any(
            trajectory.joint_trajectory.points
            for trajectory in message.trajectory
        ):
            return
        self.latest_rviz_display = copy.deepcopy(message)
        self.latest_rviz_display_at = time.monotonic()

    def latest_rviz_plan(self):
        if self.latest_rviz_display is None:
            return None, None
        age = time.monotonic() - self.latest_rviz_display_at
        return copy.deepcopy(self.latest_rviz_display), age

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
        """Legacy Rainbow controller output command used by its test panel."""
        if not self.legacy_digital_output_client.wait_for_service(timeout_sec=2.0):
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
        future = self.legacy_digital_output_client.call_async(request)
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

    def _set_legacy_digital_output_sync(self, port, value):
        """Blocking Rainbow output command retained only for legacy use."""
        if not self.legacy_digital_output_client.wait_for_service(timeout_sec=2.0):
            return False, "RBPodo set_digital_output service unavailable"
        request = SetDigitalOutput.Request()
        request.port = int(port)
        request.value = bool(value)
        event = threading.Event()
        outcome = {}

        def completed(future):
            try:
                response = future.result()
                outcome["value"] = (response.success, response.message)
            except Exception as error:
                outcome["value"] = (False, str(error))
            event.set()

        self.legacy_digital_output_client.call_async(request).add_done_callback(
            completed
        )
        if not event.wait(timeout=3.0):
            return False, "RBPodo digital output command timed out"
        success, message = outcome["value"]
        if not success:
            return False, message
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            outputs = self.legacy_node_digital_outputs.get("right")
            if outputs is not None and 0 <= int(port) < len(outputs):
                if bool(outputs[int(port)]) == bool(value):
                    return True, f"{message} · system_state confirmed"
            time.sleep(0.02)
        return False, (
            f"{message} · DO{int(port)} command accepted, but system_state "
            "confirmation timed out"
        )

    @staticmethod
    def _call_service_and_wait(client, request, description):
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f"{description} service unavailable"
        event = threading.Event()
        outcome = {}

        def completed(future):
            try:
                response = future.result()
                outcome["value"] = (response.success, response.message)
            except Exception as error:
                outcome["value"] = (False, str(error))
            event.set()

        client.call_async(request).add_done_callback(completed)
        if not event.wait(timeout=10.0):
            return False, f"{description} timed out"
        return outcome["value"]

    def _send_action_goal_and_wait(
        self,
        client,
        goal,
        description,
        *,
        result_timeout=300.0,
        on_accepted=None,
        feedback_callback=None,
    ):
        """Submit an action goal and block only the calling worker thread."""
        if not client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError(f"{description} action server unavailable")
        accepted = threading.Event()
        finished = threading.Event()
        outcome = {}

        def result_ready(future):
            try:
                outcome["result"] = future.result().result
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        def goal_ready(future):
            try:
                handle = future.result()
                if not handle.accepted:
                    outcome["error"] = f"{description} goal rejected"
                    finished.set()
                    return
                outcome["handle"] = handle
                self.active_motion_goal = handle
                if on_accepted is not None:
                    on_accepted(handle)
                handle.get_result_async().add_done_callback(result_ready)
            except Exception as error:
                outcome["error"] = str(error)
                finished.set()
            finally:
                accepted.set()

        send_arguments = {}
        if feedback_callback is not None:
            send_arguments["feedback_callback"] = feedback_callback
        client.send_goal_async(goal, **send_arguments).add_done_callback(goal_ready)
        if not accepted.wait(timeout=5.0):
            raise TimeoutError(f"{description} goal response timed out")
        if not finished.wait(timeout=result_timeout):
            handle = outcome.get("handle")
            if handle is not None:
                try:
                    handle.cancel_goal_async()
                except Exception:
                    pass
            raise TimeoutError(f"{description} timed out")
        handle = outcome.get("handle")
        if handle is self.active_motion_goal:
            self.active_motion_goal = None
        if "error" in outcome:
            raise RuntimeError(outcome["error"])
        return outcome["result"]

    def _check_robot_feedback(self):
        self._request_controller_states()
        feedback_timeout = max(
            1.0,
            float(self.get_parameter("robot_feedback_timeout").value),
        )
        move_group_ready = self.move_group_client.server_is_ready()
        for arm in ("left", "right", "head"):
            last_feedback = self.last_robot_feedback_at[arm]
            deadline = self.connection_deadline[arm]
            feedback_is_fresh = (
                last_feedback is not None
                and time.monotonic() - last_feedback <= feedback_timeout
            )
            velocity_mode_ready = (
                arm in self.keyboard_controller_states
                and self.keyboard_controller_states[arm] == "active"
            )
            controller_ready = (
                not self.execute_motion_enabled
                or velocity_mode_ready
                or (
                    self.controller_states[arm] == "active"
                    and self.joint_trajectory_clients[arm].server_is_ready()
                )
            )
            stack_ready = (
                self.robot_feedback_seen[arm]
                and feedback_is_fresh
                and (arm == "head" or move_group_ready)
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
                    "measured joint feedback received, but required controllers "
                    "did not become ready"
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
        for arm, name in CONTROLLER_NAMES.items():
            self.controller_states[arm] = states.get(name)
        for arm, name in KEYBOARD_VELOCITY_CONTROLLER_NAMES.items():
            self.keyboard_controller_states[arm] = states.get(name)

    def _not_ready_detail(
        self,
        arm,
        feedback_is_fresh,
        move_group_ready,
        controller_ready,
    ):
        if not feedback_is_fresh:
            return "measured joint feedback timeout"
        if arm != "head" and not move_group_ready:
            return "MoveGroup action unavailable"
        if (
            arm in self.keyboard_controller_states
            and self.keyboard_controller_states[arm] == "active"
        ):
            return "stack not ready while Cartesian velocity controller is active"
        if self.controller_states[arm] != "active":
            return (
                f"{CONTROLLER_NAMES[arm]} state="
                f"{self.controller_states[arm] or 'unknown'}"
            )
        if not controller_ready:
            return "FollowJointTrajectory action unavailable"
        return "stack not ready"

    def _current_tcp_transform(self, planning_group):
        return self.tf_buffer.lookup_transform(
            "World",
            tip_link_for_group(planning_group),
            rclpy.time.Time(),
            timeout=Duration(seconds=1.0),
        )

    def _current_tcp_pose(self, planning_group):
        transform = self._current_tcp_transform(planning_group)
        source = transform.transform
        pose = Pose()
        pose.position.x = source.translation.x
        pose.position.y = source.translation.y
        pose.position.z = source.translation.z
        pose.orientation = source.rotation
        return pose

    def resolve_tcp_joint_state(
        self,
        planning_group,
        target_pose,
        expected_joint_names,
        endpoint,
        teaching_name,
    ):
        """Use MoveIt IK to make a named pose's joints match its corrected TCP."""
        while rclpy.ok() and not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning(
                f"Waiting for /compute_ik to resolve corrected {teaching_name} TCP"
            )
        if not rclpy.ok():
            return
        request = GetPositionIK.Request()
        request.ik_request.group_name = planning_group
        request.ik_request.robot_state.is_diff = True
        request.ik_request.ik_link_name = tip_link_for_group(planning_group)
        request.ik_request.pose_stamped.header.frame_id = "World"
        request.ik_request.pose_stamped.header.stamp = (
            self.get_clock().now().to_msg()
        )
        request.ik_request.pose_stamped.pose = copy.deepcopy(target_pose)
        # This call resolves and persists a kinematic seed only.  A sensed weld
        # point is intentionally on the workpiece and can be rejected as a
        # collision if scene geometry is present.  Actual Plan/Execute still
        # performs normal collision checking before any physical movement.
        request.ik_request.avoid_collisions = False
        request.ik_request.timeout = Duration(seconds=3.0).to_msg()
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                outcome["response"] = future.result()
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        self.ik_client.call_async(request).add_done_callback(
            response_ready
        )
        while rclpy.ok() and not finished.wait(timeout=0.2):
            pass
        if not rclpy.ok():
            return
        if "error" in outcome:
            self.ui.post(
                self.ui.corrected_tcp_joint_state_failed,
                endpoint,
                teaching_name,
                outcome["error"],
            )
            return
        response = outcome["response"]
        if response.error_code.val != 1:
            self.ui.post(
                self.ui.corrected_tcp_joint_state_failed,
                endpoint,
                teaching_name,
                f"MoveIt IK error code {response.error_code.val}",
            )
            return
        resolved = dict(zip(
            response.solution.joint_state.name,
            response.solution.joint_state.position,
        ))
        missing = [name for name in expected_joint_names if name not in resolved]
        if missing:
            self.ui.post(
                self.ui.corrected_tcp_joint_state_failed,
                endpoint,
                teaching_name,
                "corrected TCP IK omitted joints: " + ", ".join(missing),
            )
            return
        self.ui.post(
            self.ui.apply_corrected_tcp_joint_state,
            endpoint,
            teaching_name,
            planning_group,
            tuple(expected_joint_names),
            tuple(resolved[name] for name in expected_joint_names),
            copy.deepcopy(target_pose),
        )

    def resolve_tcp_joint_states(self, targets):
        """Resolve related corrected named poses serially for deterministic YAML."""
        for (
            endpoint,
            planning_group,
            target_pose,
            joint_names,
            teaching_name,
        ) in targets:
            self.resolve_tcp_joint_state(
                planning_group,
                target_pose,
                joint_names,
                endpoint,
                teaching_name,
            )

    def publish_points(self, points, visible=True):
        displayed_points = points if visible else []
        markers, pose_array = make_weld_visualization(
            displayed_points,
            "World",
            self.get_clock().now().to_msg(),
        )
        self.marker_publisher.publish(markers)
        self.pose_publisher.publish(pose_array)

    def publish_seam_comparison(self, raw_points, corrected_points, visible=True):
        """Show raw seam opaque and offset-corrected seam translucent."""
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)
        if visible:
            for marker_id, points, color in (
                (100, raw_points, (1.0, 0.05, 0.02, 1.0)),
                (101, corrected_points, (0.0, 0.7, 1.0, 0.38)),
            ):
                line = Marker()
                line.header.frame_id = "World"
                line.header.stamp = stamp
                # moveit.rviz already enables this namespace.
                line.ns = "weld_seam"
                line.id = marker_id
                line.type = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = 0.006 if marker_id == 100 else 0.010
                line.color.r, line.color.g, line.color.b, line.color.a = color
                line.points = [
                    Point(
                        x=pose.position.x,
                        y=pose.position.y,
                        z=pose.position.z,
                    )
                    for pose in points
                ]
                markers.markers.append(line)
        self.marker_publisher.publish(markers)
        pose_array = PoseArray()
        pose_array.header.frame_id = "World"
        pose_array.header.stamp = stamp
        pose_array.poses = list(raw_points)
        self.pose_publisher.publish(pose_array)

    def publish_touch_geometry(self, endpoint, wall, floor, seam_point):
        """Show two Fastech DI0 contacts, their midpoint, and reconstructed seam point."""
        endpoint = str(endpoint).strip().lower()
        if endpoint not in ("start", "goal"):
            raise ValueError(f"unknown touch endpoint: {endpoint}")
        if not all(pose_is_valid(pose) for pose in (wall, floor, seam_point)):
            raise ValueError("touch visualization poses must be valid")
        midpoint = midpoint_pose(wall, floor)
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        base_id = 300 if endpoint == "start" else 400
        namespace = "seam_touch_geometry"

        # Delete only this endpoint's old diagnostic markers so the weld path
        # and the other endpoint remain visible.
        for marker_id in range(base_id, base_id + 16):
            marker = Marker()
            marker.header.frame_id = "World"
            marker.header.stamp = stamp
            marker.ns = namespace
            marker.id = marker_id
            marker.action = Marker.DELETE
            markers.markers.append(marker)

        items = (
            ("WALL TOUCH", wall, (1.0, 0.08, 0.05, 1.0)),
            ("FLOOR TOUCH", floor, (0.05, 0.3, 1.0, 1.0)),
            ("SEAM POINT", seam_point, (0.0, 1.0, 0.2, 1.0)),
            ("1:1 MIDPOINT", midpoint, (1.0, 0.8, 0.0, 1.0)),
        )
        for index, (label_text, pose, color) in enumerate(items):
            sphere = Marker()
            sphere.header.frame_id = "World"
            sphere.header.stamp = stamp
            sphere.ns = namespace
            sphere.id = base_id + index * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = copy.deepcopy(pose)
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.014
            (
                sphere.color.r,
                sphere.color.g,
                sphere.color.b,
                sphere.color.a,
            ) = color
            markers.markers.append(sphere)

            label = Marker()
            label.header.frame_id = "World"
            label.header.stamp = stamp
            label.ns = namespace
            label.id = base_id + index * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = pose.position.x
            label.pose.position.y = pose.position.y
            label.pose.position.z = pose.position.z + 0.022
            label.pose.orientation.w = 1.0
            label.scale.z = 0.018
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text = f"{endpoint.upper()} {label_text}"
            markers.markers.append(label)

        connection = Marker()
        connection.header.frame_id = "World"
        connection.header.stamp = stamp
        connection.ns = namespace
        connection.id = base_id + 12
        connection.type = Marker.LINE_STRIP
        connection.action = Marker.ADD
        connection.scale.x = 0.003
        connection.color.r = connection.color.g = connection.color.b = 0.9
        connection.color.a = 0.8
        connection.points = [
            Point(x=wall.position.x, y=wall.position.y, z=wall.position.z),
            Point(
                x=midpoint.position.x,
                y=midpoint.position.y,
                z=midpoint.position.z,
            ),
            Point(x=floor.position.x, y=floor.position.y, z=floor.position.z),
        ]
        markers.markers.append(connection)
        self.marker_publisher.publish(markers)

    def acquire_points(
        self,
        reference,
        axis,
        distance,
        count,
        explicit_position,
        rpy_offset,
        rpy_reference,
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
            tcp = pose_with_rpy_offset(
                tcp, *rpy_offset, reference=rpy_reference
            )
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

    def capture_initial_state(self, planning_group, pose_name="robot_start"):
        try:
            joint_names, positions, tcp, provenance = self.capture_measured_teaching_snapshot(
                planning_group, pose_name)
        except (RuntimeError, ValueError, TransformException) as error:
            self.ui.post(self.ui.error, f"Teaching capture rejected: {error}")
            return
        self.ui.post(
            self.ui.apply_initial_state,
            pose_name,
            planning_group,
            joint_names,
            positions,
            tcp,
            True,
            provenance,
        )

    def _fk_pose_for_joints(self, planning_group, joint_names, positions):
        if not self.fk_client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError("/compute_fk unavailable")
        request = GetPositionFK.Request()
        request.header.frame_id = "World"
        request.fk_link_names = [tip_link_for_group(planning_group)]
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(joint_names)
        request.robot_state.joint_state.position = [float(value) for value in positions]
        finished = threading.Event()
        outcome = {}

        def done(future):
            try:
                outcome["response"] = future.result()
            except Exception as error:
                outcome["error"] = error
            finished.set()

        self.fk_client.call_async(request).add_done_callback(done)
        if not finished.wait(timeout=3.0):
            raise RuntimeError("/compute_fk timed out")
        if "error" in outcome:
            raise RuntimeError(f"/compute_fk failed: {outcome['error']}")
        response = outcome["response"]
        if response.error_code.val != 1 or not response.pose_stamped:
            raise RuntimeError(f"/compute_fk returned code {response.error_code.val}")
        pose = response.pose_stamped[0]
        if pose.header.frame_id != "World":
            raise RuntimeError(f"FK frame {pose.header.frame_id} is not World")
        return copy.deepcopy(pose.pose)

    def validate_named_pose_recall(self, pose_name, planning_group,
                                   joint_names, positions, saved_tcp):
        """Reject legacy/stale q/TCP pairs before any named-pose planning."""
        if not pose_is_valid(saved_tcp):
            raise ValueError("saved TCP pose is unavailable")
        arm = planning_group.removesuffix("_manipulator")
        if set(joint_names) != ARM_JOINT_NAMES[arm] or len(joint_names) != 6:
            raise ValueError("saved six-joint state is incomplete")
        fk_tcp = self._fk_pose_for_joints(planning_group, joint_names, positions)
        position_error_mm = math.dist(
            _pose_position_tuple(saved_tcp), _pose_position_tuple(fk_tcp))*1000.0
        orientation_error_deg = math.degrees(quaternion_angular_distance(
            saved_tcp.orientation, fk_tcp.orientation))
        with self.measured_joint_snapshot_lock:
            current = copy.deepcopy(self.measured_joint_snapshots.get(arm))
        current_q = ([current["positions"].get(name) for name in joint_names]
                     if current else None)
        try:
            current_tcp = self._current_tcp_pose(planning_group)
            current_tcp_values = self.ui._pose_values(current_tcp)
        except TransformException:
            current_tcp_values = "N/A"
        mode = "CARTESIAN" if pose_name in TCP_POSE_TEACHING_POSES else "JOINT"
        self.get_logger().info(
            f"NAMED POSE RECALL · pose_name={pose_name} · execution_mode={mode} · "
            f"current_q={current_q} · saved_q={list(positions)} · "
            f"current_tcp_tf={current_tcp_values} · "
            f"saved_tcp={self.ui._pose_values(saved_tcp)} · "
            f"fk_saved_q={self.ui._pose_values(fk_tcp)} · "
            f"saved_tcp_vs_fk_saved_q_error_mm={position_error_mm:.2f} · "
            f"orientation_error_deg={orientation_error_deg:.2f}")
        if (position_error_mm > float(self.get_parameter("teaching_fk_tf_position_tolerance_mm").value)
                or orientation_error_deg > float(self.get_parameter("teaching_fk_tf_orientation_tolerance_deg").value)):
            raise ValueError(
                f"{pose_name} saved TCP/FK mismatch {position_error_mm:.2f} mm / "
                f"{orientation_error_deg:.2f} deg; re-teach this pose")
        return fk_tcp

    def capture_measured_teaching_snapshot(self, planning_group, pose_name):
        """One post-standstill joint sample is the sole canonical pose source."""
        arm = planning_group.removesuffix("_manipulator")
        joint_names = tuple(f"{arm}_manipulator_joint{i}" for i in range(1, 7))
        with self.keyboard_velocity_lock:
            keyboard_arm = self.keyboard_velocity_command["arm"]
        if keyboard_arm == arm:
            self.clear_keyboard_velocity()
        if not self.wait_until_arm_stopped(arm, timeout=2.5):
            raise RuntimeError("arm did not reach measured standstill")
        standstill_at = time.monotonic()
        deadline = standstill_at + 1.0
        snapshot = None
        while time.monotonic() < deadline:
            with self.measured_joint_snapshot_lock:
                candidate = copy.deepcopy(self.measured_joint_snapshots.get(arm))
            if candidate and candidate["received_monotonic"] > standstill_at:
                snapshot = candidate
                break
            time.sleep(0.01)
        if snapshot is None:
            raise RuntimeError("no new complete six-joint sample after standstill")
        age_ms = (time.monotonic()-snapshot["received_monotonic"])*1000.0
        if age_ms > float(self.get_parameter("teaching_joint_state_max_age_ms").value):
            raise RuntimeError(f"measured joint state is stale ({age_ms:.1f} ms)")
        positions = tuple(snapshot["positions"][name] for name in joint_names)
        fk_tcp = self._fk_pose_for_joints(planning_group, joint_names, positions)
        observed = self._current_tcp_transform(planning_group)
        tf_tcp = Pose()
        tf_tcp.position.x = observed.transform.translation.x
        tf_tcp.position.y = observed.transform.translation.y
        tf_tcp.position.z = observed.transform.translation.z
        tf_tcp.orientation = observed.transform.rotation
        tf_stamp_s = (float(observed.header.stamp.sec)
                      + float(observed.header.stamp.nanosec)*1e-9)
        tf_age_ms = (self.get_clock().now().nanoseconds*1e-9-tf_stamp_s)*1000.0
        position_error_mm = math.dist(
            _pose_position_tuple(fk_tcp), _pose_position_tuple(tf_tcp))*1000.0
        orientation_error_deg = math.degrees(quaternion_angular_distance(
            fk_tcp.orientation, tf_tcp.orientation))
        self.get_logger().info(
            f"TEACH CAPTURE · {pose_name} · q_measured={list(positions)} rad · "
            f"FK TCP={self.ui._pose_values(fk_tcp)} · "
            f"TF TCP={self.ui._pose_values(tf_tcp)} · "
            f"FK/TF error={position_error_mm:.2f} mm / "
            f"{orientation_error_deg:.2f} deg · joint_state_age_ms={age_ms:.1f} · "
            f"tf_age_ms={tf_age_ms:.1f}")
        if (position_error_mm > float(self.get_parameter("teaching_fk_tf_position_tolerance_mm").value)
                or orientation_error_deg > float(self.get_parameter("teaching_fk_tf_orientation_tolerance_deg").value)):
            raise RuntimeError(f"FK/TF mismatch = {position_error_mm:.2f} mm / {orientation_error_deg:.2f} deg")
        tf_max_age_ms = float(
            self.get_parameter("teaching_tf_max_age_ms").value
        )
        tf_fresh = tf_age_ms <= tf_max_age_ms
        if not tf_fresh:
            # FK from the fresh, complete six-joint snapshot is the canonical
            # saved TCP.  robot_state_publisher may publish the matching TF at
            # a lower cadence; once FK/TF geometry agrees above, timestamp age
            # alone must not silently keep an older WAIT teaching active.
            self.get_logger().warning(
                f"TEACH CAPTURE · {pose_name} · TF timestamp is stale "
                f"({tf_age_ms:.1f} ms), accepted because fresh-joint FK and "
                "observed TF agree"
            )
        provenance = {
            "capture_source": "measured_joint_fk",
            "tcp_source": "moveit_fk",
            "joint_state_timestamp": snapshot["stamp_sec"],
            "joint_state_age_ms": age_ms,
            "tf_age_ms": tf_age_ms,
            "tf_fresh": tf_fresh,
            "tf_fk_position_error_mm": position_error_mm,
            "tf_fk_orientation_error_deg": orientation_error_deg,
        }
        self.get_logger().info(f"TEACH CAPTURE · {pose_name} · SAVE=ACCEPTED")
        return joint_names, positions, fk_tcp, provenance

    def execute_touch_probe(
        self,
        planning_group,
        probe_kind,
        direction,
        distance,
        velocity_scale,
        interpolation_step,
    ):
        """Execute a straight World-vector probe path; the GUI cancels on Fastech DI0."""
        arm = planning_group.removesuffix("_manipulator")
        try:
            start = self._current_tcp_pose(planning_group)
            self.touch_probe_cancel_event.set()
            self.touch_probe_cancel_event = threading.Event()
            self.active_touch_probe = (
                arm,
                probe_kind,
                planning_group,
                copy.deepcopy(start),
                velocity_scale,
                interpolation_step,
            )
            self.touch_probe_edge_pose = None
            self.touch_probe_controller_deactivated = False
            self.touch_probe_stop_requested.clear()
            # MoveIt's GetCartesianPath already interpolates this segment using
            # max_step.  Supplying every 1 mm point here duplicated that work
            # and made a 50 mm probe spend many seconds in PLAN PREVIEW.
            count = 2
            if isinstance(direction, str):
                points = straight_waypoints(
                    start, distance, count, direction.lower(), "world"
                )
            else:
                dx, dy, dz = _unit_vector(direction, "touch probe direction")
                goal = copy.deepcopy(start)
                goal.position.x += float(distance) * dx
                goal.position.y += float(distance) * dy
                goal.position.z += float(distance) * dz
                points = [copy.deepcopy(start), goal]
        except (TransformException, ValueError) as error:
            self.active_touch_probe = None
            self.ui.post(self.ui.touch_probe_failed, str(error))
            return
        self.publish_points(points, True)
        self.submit_cartesian_motion(
            points,
            velocity_scale,
            interpolation_step,
            True,
            True,
            False,
            planning_group,
        )

    def stop_touch_probe_and_capture(self):
        """Stop command streaming, confirm standstill, then capture TCP."""
        if not self.touch_stop_lock.acquire(blocking=False):
            return
        try:
            self._stop_touch_probe_and_capture_locked()
        finally:
            self.touch_stop_lock.release()

    def _stop_touch_probe_and_capture_locked(self):
        """Serialized implementation for a Fastech DI0 rising edge."""
        probe = self.active_touch_probe
        if probe is None:
            return
        arm, kind, planning_group, start, speed, interpolation = probe
        cancel_event = self.touch_probe_cancel_event
        stationary, controller_deactivated = self._stop_motion_on_touch(
            arm, f"probe {kind}"
        )
        self.touch_probe_controller_deactivated = controller_deactivated
        if not stationary:
            self.active_touch_probe = None
            self.ui.post(
                self.ui.touch_probe_failed,
                "Touch stop not confirmed; automatic controller restore/retract inhibited",
            )
            return
        try:
            stopped_pose = self._current_tcp_pose(planning_group)
        except TransformException as error:
            self.active_touch_probe = None
            self.ui.post(self.ui.touch_probe_failed, str(error))
            return
        touched = (
            copy.deepcopy(self.touch_probe_edge_pose)
            if self.touch_probe_edge_pose is not None
            else copy.deepcopy(stopped_pose)
        )
        edge_values = (
            touched.position.x, touched.position.y, touched.position.z
        )
        stopped_values = (
            stopped_pose.position.x,
            stopped_pose.position.y,
            stopped_pose.position.z,
        )
        braking_mm = tuple(
            (stopped_values[index] - edge_values[index]) * 1000.0
            for index in range(3)
        )
        self.ui.post(
            self.ui.log,
            f"Fastech DI0 CONTACT LATCH · {kind} · edge XYZ="
            f"({edge_values[0]:.6f}, {edge_values[1]:.6f}, "
            f"{edge_values[2]:.6f}) m · braking delta="
            f"({braking_mm[0]:+.3f}, {braking_mm[1]:+.3f}, "
            f"{braking_mm[2]:+.3f}) mm",
        )
        self.touch_probe_edge_pose = None
        self.ui.post(
            self.ui.apply_touch_capture,
            touched,
            planning_group,
            f"automatic probe:{kind}",
            start,
            stopped_pose,
            cancel_event,
        )

    def _stop_motion_on_touch(self, arm, label):
        """Apply the seam-probe controlled-stop ladder to any guarded move."""
        handle = self.active_motion_goal
        action_finished = threading.Event()
        controller_deactivated = False
        if handle is not None:
            try:
                handle.get_result_async().add_done_callback(
                    lambda _future: action_finished.set()
                )
                handle.cancel_goal_async()
            except Exception as error:
                self.ui.post(
                    self.ui.log,
                    f"Fastech DI0 {label} action cancel warning: {error}",
                )

        # Prefer action cancellation while keeping the trajectory controller
        # active.  Deactivate/reactivate mode switches caused a visible kick at
        # contact and before retract.  Escalate only when cancellation cannot
        # establish standstill promptly.
        cancel_success, cancel_message = self.cancel_controller_goals(arm)
        self.ui.post(
            self.ui.log,
            f"Fastech DI0 direct trajectory cancel: "
            f"{'OK' if cancel_success else 'FAILED'} · {cancel_message}",
        )
        action_finished.wait(timeout=0.25)
        stationary = (
            cancel_success
            and self.wait_until_arm_stopped(arm, timeout=1.0)
        )
        if cancel_success and stationary:
            self.ui.post(
                self.ui.log,
                "Fastech DI0 smooth stop: action canceled · controller kept active",
            )
        else:
            controller_success, controller_message = self.switch_arm_controller(
                arm, False
            )
            controller_deactivated = bool(controller_success)
            # Deactivation already requests RB task_stop in the hardware
            # interface. Do not immediately issue a second task_stop.
            stationary = (
                controller_success
                and self.wait_until_arm_stopped(arm, timeout=1.0)
            )
            direct_stop_success, direct_stop_message = True, "not needed after controller stop"
            if not stationary:
                direct_stop_success, direct_stop_message = (
                    self.request_direct_motion_stop(arm)
                )
                stationary = self.wait_until_arm_stopped(arm)
            self.ui.post(
                self.ui.log,
                f"Fastech DI0 fallback controller stop: "
                f"{'OK' if controller_success else 'FAILED'} · "
                f"{controller_message}",
            )
            self.ui.post(
                self.ui.log,
                f"Fastech DI0 fallback RBPodo move_stop: "
                f"{'OK' if direct_stop_success else 'FAILED'} · "
                f"{direct_stop_message}",
            )
        if handle is not None and not action_finished.wait(timeout=2.0):
            self.ui.post(
                self.ui.log,
                "Touch action cleanup still pending; automatic restore/retract inhibited",
            )
            return False, controller_deactivated
        return bool(stationary), controller_deactivated

    def restore_touch_controller(self, arm):
        """Never resume position control until fresh standstill and RB Idle."""
        if not self.wait_until_arm_stopped(arm):
            return False, "measured standstill unavailable; controller left inactive"
        if not self.wait_for_robot_idle(arm):
            return False, "fresh RB Idle unavailable; controller left inactive"
        return self.switch_arm_controller(arm, True)

    def cancel_controller_goals(self, arm):
        """Cancel every active FollowJointTrajectory goal for one arm."""
        client = self.joint_trajectory_cancel_clients.get(arm)
        if client is None or not client.wait_for_service(timeout_sec=0.25):
            return False, f"{arm} trajectory cancel service unavailable"
        request = CancelGoal.Request()
        # Zero UUID + zero timestamp means cancel all goals.
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                response = future.result()
                outcome["code"] = int(response.return_code)
                outcome["count"] = len(response.goals_canceling)
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        client.call_async(request).add_done_callback(response_ready)
        if not finished.wait(timeout=1.0):
            return False, "cancel response timed out"
        if "error" in outcome:
            return False, outcome["error"]
        success = outcome.get("code") == CancelGoal.Response.ERROR_NONE
        return success, (
            f"return_code={outcome.get('code')} · "
            f"goals_canceling={outcome.get('count', 0)}"
        )

    def stop_sequence_equipment(self, devices):
        """Cancel every robot goal and escalate until measured motion stops."""
        # Operator STOP controls motion and welding outputs only. Keep Fastech
        # DO0 exactly as-is so stopping a robot does not disable touch sensing.
        results = [f"Fastech DO{FASTECH_TOUCH_OUTPUT_PORT} unchanged"]
        for device in tuple(dict.fromkeys(devices)):
            canceled, cancel_message = self.cancel_controller_goals(device)
            stationary = self.wait_until_device_stopped(device, timeout=1.5)
            if stationary:
                results.append(
                    f"{device}: stationary · cancel="
                    f"{'OK' if canceled else cancel_message}"
                )
            else:
                deactivated, deactivate_message = self.switch_arm_controller(
                    device, False
                )
                if device in ("left", "right"):
                    direct, direct_message = self.request_direct_motion_stop(device)
                else:
                    direct, direct_message = False, "head has no RBPodo move_stop"
                stopped_after_fallback = self.wait_until_device_stopped(device)
                results.append(
                    f"{device}: fallback stop="
                    f"{'OK' if stopped_after_fallback else 'FAILED'} · "
                    f"controller_off={deactivated} ({deactivate_message}) · "
                    f"move_stop={direct} ({direct_message})"
                )
        self.ui.post(self.ui.sequence_hard_stop_finished, results)

    def stop_touch_guarded_motion(self):
        """Stop a guarded named move using the seam-probe stop ladder."""
        if not self.touch_guard_stop_lock.acquire(blocking=False):
            return
        try:
            guard = self.active_touch_guard
            if guard is None:
                return
            arm, pose_name = guard
            self.touch_guard_triggered.set()
            self.touch_guard_stop_success = False
            stationary, controller_deactivated = self._stop_motion_on_touch(
                arm, f"guarded named pose {pose_name}"
            )
            restored = True
            restore_message = "controller remained active"
            if controller_deactivated and stationary:
                restored, restore_message = self.restore_touch_controller(arm)
                self.ui.post(
                    self.ui.log,
                    f"Fastech DI0 guarded motion controller restore: "
                    f"{'OK' if restored else 'FAILED'} · {restore_message}",
                )
            self.touch_guard_stop_success = bool(stationary and restored)
            if not stationary:
                self.ui.post(
                    self.ui.error,
                    f"Fastech DI0 detected during {pose_name}, but standstill was not confirmed",
                )
            elif not restored:
                self.ui.post(
                    self.ui.error,
                    f"Fastech DI0 stopped {pose_name}, but controller restore failed: "
                    f"{restore_message}",
                )
        finally:
            # The Fastech DI0 edge terminates only this guarded execution.  Drop the
            # guard as soon as stop/recovery finishes so a later command can
            # move again (including a deliberate retraction while Fastech DI0 is
            # still high).  A new stop requires Fastech DI0 to release and rise again.
            if self.active_touch_guard == guard:
                self.active_touch_guard = None
            self.touch_guard_stop_complete.set()
            self.touch_guard_stop_lock.release()

    def request_direct_motion_stop(self, arm):
        """Request RBPodo move_stop; this is a controlled stop, not E-stop."""
        client = self.move_stop_clients.get(arm)
        if client is None or not client.wait_for_service(timeout_sec=0.25):
            return False, f"/{arm}_rbpodo_hardware/move_stop unavailable"
        request = MoveStop.Request()
        request.timeout = 2.0
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                response = future.result()
                outcome["success"] = bool(response.success)
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        client.call_async(request).add_done_callback(response_ready)
        if not finished.wait(timeout=3.0):
            return False, "service response timed out"
        if "error" in outcome:
            return False, outcome["error"]
        return outcome.get("success", False), "controlled move_stop completed"

    def switch_arm_controller(self, arm, activate):
        """Deactivate to stop command streaming, or reactivate for return."""
        client = self.controller_switch_client
        controller = CONTROLLER_NAMES[arm]
        if not client.wait_for_service(timeout_sec=0.5):
            return False, "/controller_manager/switch_controller unavailable"
        request = SwitchController.Request()
        if activate:
            request.activate_controllers = [controller]
        else:
            request.deactivate_controllers = [controller]
        request.strictness = SwitchController.Request.BEST_EFFORT
        request.activate_asap = True
        request.timeout.sec = 3
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                outcome["success"] = bool(future.result().ok)
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        client.call_async(request).add_done_callback(response_ready)
        if not finished.wait(timeout=4.0):
            return False, f"{controller} switch timed out"
        if "error" in outcome:
            return False, outcome["error"]
        action = "activated" if activate else "deactivated"
        if not outcome.get("success", False):
            return False, f"{controller} failed to become {action}"
        expected_state = "active" if activate else "inactive"
        if not self.wait_for_controller_state(controller, expected_state):
            return False, (
                f"{controller} did not report {expected_state} after switch"
            )
        return True, f"{controller} {action}"

    def set_keyboard_velocity_controller_enabled(self, arm, enable):
        """Hand keyboard teaching the arm, or return it to trajectories."""
        if self.keyboard_servo is not None:
            return self._set_keyboard_servo_enabled(arm, enable)
        if not enable:
            self.clear_keyboard_velocity()
            # A fixed 100 ms delay does not prove braking has finished.
            # JTC must sample a stationary measured pose when taking ownership.
            if not self.wait_until_arm_stopped(arm, timeout=3.0):
                return False, "Keyboard exit blocked: measured joints have not stopped"
        trajectory_controller = CONTROLLER_NAMES[arm]
        velocity_controller = KEYBOARD_VELOCITY_CONTROLLER_NAMES[arm]
        activate = velocity_controller if enable else trajectory_controller
        deactivate = trajectory_controller if enable else velocity_controller
        client = self.controller_switch_client
        if not client.wait_for_service(timeout_sec=1.0):
            return False, "/controller_manager/switch_controller unavailable"
        request = SwitchController.Request()
        request.activate_controllers = [activate]
        request.deactivate_controllers = [deactivate]
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout.sec = 3
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                outcome["success"] = bool(future.result().ok)
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        client.call_async(request).add_done_callback(response_ready)
        if not finished.wait(timeout=4.0):
            return False, "keyboard controller exchange timed out"
        if "error" in outcome:
            return False, outcome["error"]
        if not outcome.get("success", False):
            return False, (
                f"controller_manager rejected {deactivate} -> {activate}"
            )
        if not self.wait_for_controller_state(activate, "active"):
            return False, f"{activate} did not become active"
        if not self.wait_for_controller_state(deactivate, "inactive"):
            return False, f"{deactivate} did not become inactive"
        return True, f"{deactivate} -> {activate}"

    def _set_keyboard_servo_enabled(self, arm, enable):
        """Start/stop the jog stream on the arm's JTC; no controller exchange."""
        if enable:
            if not self.wait_for_controller_state(
                CONTROLLER_NAMES[arm], "active", timeout=1.0
            ):
                return False, f"{CONTROLLER_NAMES[arm]} is not active"
            return self.keyboard_servo.enable(arm)
        self.clear_keyboard_velocity()
        stopped = self.wait_until_arm_stopped(arm, timeout=3.0)
        # Always leave Servo mode, even if still coasting: stopping Servo
        # holds the last command and restores the JTC's normal start state.
        success, message = self.keyboard_servo.disable(arm)
        if not stopped:
            return False, (
                "measured joints had not stopped before Servo exit; "
                f"{message}"
            )
        return success, message

    def wait_for_controller_state(self, controller, expected, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            finished = threading.Event()
            outcome = {}

            def response_ready(future):
                try:
                    outcome["response"] = future.result()
                except Exception as error:
                    outcome["error"] = str(error)
                finished.set()

            self.controller_list_client.call_async(
                ListControllers.Request()
            ).add_done_callback(response_ready)
            if finished.wait(timeout=0.5) and "response" in outcome:
                states = {
                    item.name: item.state
                    for item in outcome["response"].controller
                }
                if states.get(controller) == expected:
                    return True
            time.sleep(0.05)
        return False

    def wait_until_arm_stopped(self, arm, timeout=3.0, stable_duration_s=0.30):
        """Confirm measured joints remain still before capturing the touch."""
        return self.wait_until_device_stopped(
            arm, timeout, stable_duration_s=stable_duration_s
        )

    def wait_for_robot_idle(self, arm, timeout=2.5):
        """Wait until the RB motion task has fully left its moving state."""
        deadline = time.monotonic() + timeout
        stable_since = None
        while time.monotonic() < deadline:
            now = time.monotonic()
            received_at = self.last_motion_state_at.get(arm)
            fresh = received_at is not None and now - received_at <= 0.25
            if fresh and self.latest_robot_motion_state.get(arm) == 1:
                stable_since = stable_since or now
                if now - stable_since >= 0.15:
                    return True
            else:
                stable_since = None
            time.sleep(0.02)
        return False

    def wait_until_device_stopped(
        self, device, timeout=3.0, stable_duration_s=0.30
    ):
        """Confirm a controlled arm or head remains measurably stationary."""
        names = tuple(sorted(CONTROLLED_JOINT_NAMES[device]))
        deadline = time.monotonic() + timeout
        previous = None
        stable_since = None
        while time.monotonic() < deadline:
            if device in ("left", "right"):
                received_at = self.last_measured_joints_at.get(device)
                if received_at is None or time.monotonic() - received_at > 0.25:
                    previous = None
                    stable_since = None
                    time.sleep(0.02)
                    continue
            try:
                current = tuple(self.latest_joint_positions[name] for name in names)
            except KeyError:
                time.sleep(0.02)
                continue
            now = time.monotonic()
            if previous is not None:
                maximum_delta = max(
                    abs(value - old)
                    for value, old in zip(current, previous)
                )
                if maximum_delta <= 2e-5:
                    stable_since = stable_since or now
                    if now - stable_since >= float(stable_duration_s):
                        return True
                else:
                    stable_since = None
            previous = current
            time.sleep(0.02)
        return False

    def return_touch_probe(
        self,
        planning_group,
        touched,
        start,
        speed,
        step,
        probe_kind,
        settle_seconds,
        cancel_event=None,
    ):
        """Execute the reverse probe path back to its captured start pose."""
        if cancel_event is None:
            cancel_event = self.touch_probe_cancel_event
        try:
            arm = planning_group.removesuffix("_manipulator")
            self.ui.post(
                self.ui.log,
                f"Fastech DI0 {probe_kind} standstill dwell · "
                f"{settle_seconds:.1f} seconds",
            )
            time.sleep(settle_seconds)
            if cancel_event.is_set() or cancel_event is not self.touch_probe_cancel_event:
                raise RuntimeError("Touch probe canceled; automatic retract inhibited")
            if self.touch_probe_controller_deactivated:
                activated, activation_message = self.restore_touch_controller(arm)
                if not activated:
                    raise RuntimeError(activation_message)
                self.touch_probe_controller_deactivated = False
            if not self.wait_until_arm_stopped(arm):
                raise RuntimeError("Standstill lost before retract")
            if cancel_event.is_set() or cancel_event is not self.touch_probe_cancel_event:
                raise RuntimeError("Touch probe canceled; automatic retract inhibited")
            # The captured contact/stopped pose can precede settling or a
            # controller exchange. Plan from the actual pose at retract time.
            current_pose = self._current_tcp_pose(planning_group)
            points = linear_pose_waypoints(current_pose, start, 2)
            success, message = self.run_sequence_cartesian_motion(
                {
                    "planning_group": planning_group,
                    "interpolation_step": step,
                    "velocity_scale": speed,
                    "points": points,
                },
                True,
            )
        except (RuntimeError, ValueError, TransformException) as error:
            success, message = False, str(error)
        if cancel_event is self.touch_probe_cancel_event:
            self.active_touch_probe = None
        self.ui.post(
            self.ui.touch_probe_return_finished,
            success,
            message,
            probe_kind,
        )

    def clear_touch_probe(self, cancel_return=True):
        """Disarm contact detection; explicit stop/failure also cancels retract."""
        if cancel_return:
            self.touch_probe_cancel_event.set()
        self.active_touch_probe = None
        self.touch_probe_edge_pose = None
        self.touch_probe_stop_requested.set()

    def stop_auto_motion(self, arm):
        """Stop any auto-seam motion and restore an idle active controller."""
        self.clear_touch_probe()
        handle = self.active_motion_goal
        if handle is not None:
            try:
                handle.cancel_goal_async()
            except Exception as error:
                self.ui.post(self.ui.log, f"STOP AUTO cancel warning: {error}")
        stopped, message = self.switch_arm_controller(arm, False)
        if not stopped:
            fallback, fallback_message = self.request_direct_motion_stop(arm)
            message = f"{message}; fallback={fallback}: {fallback_message}"
        stationary = self.wait_until_arm_stopped(arm)
        activated, activation_message = False, "stop unconfirmed; automatic restore inhibited"
        if stationary and stopped:
            activated, activation_message = self.restore_touch_controller(arm)
        self.active_touch_probe = None
        success = stationary and activated
        self.ui.post(
            self.ui.auto_seam_stop_finished,
            success,
            f"{message}; {activation_message}",
        )

    def plan_initial_state(
        self,
        planning_group,
        joint_names,
        positions,
        velocity_scale,
        pose_name=None,
        target_tcp=None,
    ):
        self.initial_planned_trajectory = None
        self.initial_planned_pose_name = None
        self.initial_planned_group = None
        self.initial_planned_target = None
        try:
            self.validate_named_pose_recall(
                pose_name, planning_group, joint_names, positions, target_tcp)
        except (RuntimeError, ValueError, TransformException) as error:
            self.ui.post(self.ui.error, f"Named pose recall blocked: {error}")
            return
        self.initial_planned_target = (
            planning_group, tuple(joint_names), tuple(positions),
            copy.deepcopy(target_tcp))
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
        tcp_target = bool(
            pose_name in TCP_POSE_TEACHING_POSES
            and pose_is_valid(target_tcp)
        )
        if tcp_target:
            try:
                current_tcp = self._current_tcp_pose(planning_group)
            except TransformException as error:
                self.ui.post(self.ui.error, f"Current TCP lookup failed: {error}")
                return
            target_delta = math.sqrt(sum(
                (
                    getattr(current_tcp.position, axis)
                    - getattr(target_tcp.position, axis)
                ) ** 2
                for axis in ("x", "y", "z")
            ))
            orientation_delta = quaternion_angular_distance(
                current_tcp.orientation, target_tcp.orientation
            )
            already_at_target = (
                target_delta <= 0.001 and orientation_delta <= 0.01
            )
        else:
            maximum_delta = max(
                abs(current - target)
                for current, target in zip(current_positions, positions)
            )
            already_at_target = maximum_delta <= 0.002
        if already_at_target:
            self.ui.post(
                self.ui.pipeline_result,
                "Already at selected corrected XYZ + taught orientation"
                if tcp_target
                else "Already at selected taught pose · no plan required",
            )
            return
        if tcp_target:
            self._plan_named_tcp_linear(
                planning_group,
                current_tcp,
                target_tcp,
                tuple(positions),
                velocity_scale,
                pose_name,
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
            "Planning to selected taught joint angles",
        )
        future = self.move_group_client.send_goal_async(goal)
        target_positions = tuple(positions)
        future.add_done_callback(
            lambda result: self._initial_plan_goal_response(
                result,
                planning_group,
                target_positions,
                velocity_scale,
                pose_name,
            )
        )

    def _plan_named_tcp_linear(
        self,
        planning_group,
        current_tcp,
        target_tcp,
        target_positions,
        velocity_scale,
        pose_name,
    ):
        """Plan current TCP→named TCP as linear XYZ plus quaternion SLERP."""
        if not self.cartesian_planning_client.wait_for_service(timeout_sec=3.0):
            self.ui.post(
                self.ui.error, "/compute_cartesian_path service unavailable"
            )
            return
        try:
            waypoints = named_tcp_linear_waypoints(current_tcp, target_tcp)
        except ValueError as error:
            self.ui.post(self.ui.error, f"Named TCP path failed: {error}")
            return
        request = GetCartesianPath.Request()
        request.header.frame_id = "World"
        request.start_state.is_diff = True
        request.group_name = planning_group
        request.link_name = tip_link_for_group(planning_group)
        # The current state is already the path start.  Send every sampled
        # pose after it so MoveIt follows the explicit SLERP sequence.
        request.waypoints = copy.deepcopy(waypoints[1:])
        request.max_step = 0.005
        request.jump_threshold = 0.0
        request.avoid_collisions = True
        self.ui.post(
            self.ui.pipeline_waiting,
            f"Planning named TCP linear path · {len(waypoints)} SLERP samples",
        )
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                outcome["response"] = future.result()
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        self.cartesian_planning_client.call_async(request).add_done_callback(
            response_ready
        )
        if not finished.wait(timeout=30.0):
            self.ui.post(self.ui.error, "Named TCP Cartesian planning timed out")
            return
        if "error" in outcome:
            self.ui.post(
                self.ui.error,
                f"Named TCP Cartesian planning failed: {outcome['error']}",
            )
            return
        response = outcome["response"]
        if response.fraction < 0.999:
            self.ui.post(
                self.ui.error,
                f"Named TCP Cartesian path planned only "
                f"{response.fraction:.1%}",
            )
            return
        scale_trajectory_speed(response.solution, velocity_scale)
        display = DisplayTrajectory()
        display.trajectory_start = response.start_state
        display.trajectory.append(response.solution)
        self.initial_planned_trajectory = copy.deepcopy(response.solution)
        self.initial_planned_pose_name = pose_name
        self.initial_planned_group = planning_group
        self.display_trajectory_publisher.publish(display)
        self.ui.post(
            self.ui.initial_position_plan_ready,
            planning_group,
            target_positions,
            velocity_scale,
            "Named TCP linear plan shown in RViz · XYZ linear + orientation SLERP",
        )

    def _initial_plan_goal_response(
        self,
        future,
        planning_group,
        target_positions,
        velocity_scale,
        pose_name,
    ):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Taught-pose plan failed: {error}",
            )
            return
        if not goal_handle.accepted:
            self.ui.post(self.ui.error, "Taught-pose plan was rejected")
            return
        goal_handle.get_result_async().add_done_callback(
            lambda result: self._initial_plan_result(
                result,
                planning_group,
                target_positions,
                velocity_scale,
                pose_name,
            )
        )

    def _initial_plan_result(
        self,
        future,
        planning_group,
        target_positions,
        velocity_scale,
        pose_name,
    ):
        try:
            result = future.result().result
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Taught-pose plan failed: {error}",
            )
            return
        if result.error_code.val != 1:
            self.ui.post(
                self.ui.error,
                "Taught-pose plan failed "
                f"(MoveIt code {result.error_code.val})",
            )
            return
        display = DisplayTrajectory()
        display.trajectory_start = result.trajectory_start
        display.trajectory.append(result.planned_trajectory)
        self.initial_planned_trajectory = copy.deepcopy(
            result.planned_trajectory
        )
        self.initial_planned_pose_name = pose_name
        self.initial_planned_group = planning_group
        self.display_trajectory_publisher.publish(display)
        self.ui.post(
            self.ui.initial_position_plan_ready,
            planning_group,
            target_positions,
            velocity_scale,
            "Taught-pose plan shown in RViz · ready to execute",
        )

    def execute_initial_plan(self):
        if not self.execute_motion_enabled:
            self.ui.post(
                self.ui.error,
                "Taught-pose execution is disabled by launch "
                "configuration",
            )
            return
        trajectory = self.initial_planned_trajectory
        if trajectory is None:
            self.ui.post(self.ui.error, "Plan the selected taught pose first")
            return
        target = self.initial_planned_target
        if target is None:
            self.ui.post(self.ui.error, "Saved named-pose target is unavailable; re-plan")
            return
        try:
            self.validate_named_pose_recall(self.initial_planned_pose_name, *target)
        except (RuntimeError, ValueError, TransformException) as error:
            self.ui.post(self.ui.error, f"Named pose recall blocked: {error}")
            return
        if not self.execute_trajectory_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(
                self.ui.error,
                "ExecuteTrajectory action server unavailable",
            )
            return
        pose_name = self.initial_planned_pose_name
        planning_group = self.initial_planned_group
        self.initial_planned_trajectory = None
        self.initial_planned_pose_name = None
        self.initial_planned_group = None
        self.initial_planned_target = None
        touch_guarded = bool(
            pose_name in TOUCH_GUARDED_TEACHING_POSES and planning_group
        )
        if touch_guarded:
            arm = planning_group.removesuffix("_manipulator")
            if self.node_touch_input_states.get(arm) is None:
                self.ui.post(
                    self.ui.error,
                    "Fastech DI0 is unavailable; guarded execution was not started",
                )
                return
            if self.node_touch_input_states.get(arm):
                self.ui.post(
                    self.ui.log,
                    "Fastech DI0 is already ON; new taught-pose execution is allowed. "
                    "Guard will stop only after Fastech DI0 releases and rises again",
                )
            self.touch_guard_triggered.clear()
            self.touch_guard_stop_complete.clear()
            self.touch_guard_stop_success = False
            self.active_touch_guard = (arm, pose_name)
            self.ui.post(
                self.ui.log,
                f"Fastech DI0 stop guard armed for approved taught pose: {pose_name}",
            )
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        self.ui.post(
            self.ui.pipeline_waiting,
            "Executing the approved taught-pose plan",
        )
        future = self.execute_trajectory_client.send_goal_async(goal)
        future.add_done_callback(
            lambda result: self._initial_execute_goal_response(
                result, touch_guarded
            )
        )

    def _initial_execute_goal_response(self, future, touch_guarded=False):
        try:
            goal_handle = future.result()
        except Exception as error:
            if touch_guarded:
                self.active_touch_guard = None
            self.ui.post(
                self.ui.error,
                f"Taught-pose execution failed: {error}",
            )
            return
        if not goal_handle.accepted:
            if touch_guarded:
                self.active_touch_guard = None
            self.ui.post(self.ui.error, "Taught-pose execution rejected")
            return
        self.active_motion_goal = goal_handle
        if touch_guarded and self.touch_guard_triggered.is_set():
            goal_handle.cancel_goal_async()
        goal_handle.get_result_async().add_done_callback(
            lambda result: self._initial_execute_result(result, touch_guarded)
        )

    def _initial_execute_result(self, future, touch_guarded=False):
        self.active_motion_goal = None
        try:
            result = future.result().result
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Taught-pose execution failed: {error}",
            )
            if touch_guarded:
                self.active_touch_guard = None
            return
        if touch_guarded and self.touch_guard_triggered.is_set():
            stop_complete = self.touch_guard_stop_complete.wait(timeout=5.0)
            stop_success = self.touch_guard_stop_success
            self.active_touch_guard = None
            if not stop_complete:
                self.ui.post(
                    self.ui.error,
                    "Guarded taught-pose action ended, but Fastech DI0 stop cleanup timed out",
                )
            elif not stop_success:
                self.ui.post(
                    self.ui.error,
                    "Guarded taught-pose action ended, but controller recovery failed",
                )
            else:
                self.ui.post(
                    self.ui.pipeline_result,
                    "Guarded taught-pose execution stopped by Fastech DI0 · "
                    "execution closed · ready for the next command",
                )
            return
        if touch_guarded:
            self.active_touch_guard = None
        if result.error_code.val != 1:
            self.ui.post(
                self.ui.error,
                "Taught-pose execution failed "
                f"(MoveIt code {result.error_code.val})",
            )
            return
        self.ui.post(
            self.ui.initial_position_execution_finished,
            "Robot reached the selected taught pose",
        )

    def release_servo_hold_after_arrival(self, arm):
        """End the post-trajectory Servo-J hold once the arm is stationary.

        Not called automatically: releasing the hold moves the RB servo->Idle
        settle to arrival, but every following trajectory then starts from RB
        Idle and gets the mirror-image Idle->servo start kick.  Kept for
        attended diagnostics of that RB mode transition.
        """
        if not self.wait_until_arm_stopped(arm, timeout=2.0):
            self.ui.post(
                self.ui.log,
                f"{arm.upper()} Servo-J hold kept · arm not stationary after arrival",
            )
            return False, "arm not stationary"
        client = self.servo_hold_release_clients.get(arm)
        if client is None or not client.wait_for_service(timeout_sec=0.5):
            self.ui.post(
                self.ui.log,
                f"{arm.upper()} Servo-J hold release unavailable · "
                "hardware without release_servo_hold",
            )
            return False, "release_servo_hold unavailable"
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                response = future.result()
                outcome["success"] = bool(response.success)
                outcome["message"] = str(response.message)
            except Exception as error:
                outcome["success"] = False
                outcome["message"] = str(error)
            finished.set()

        client.call_async(Trigger.Request()).add_done_callback(response_ready)
        if not finished.wait(timeout=2.0):
            outcome = {"success": False, "message": "release response timed out"}
        self.ui.post(
            self.ui.log,
            f"{arm.upper()} Servo-J hold after taught pose · "
            f"{'released' if outcome['success'] else 'kept'} · {outcome['message']}",
        )
        return outcome["success"], outcome["message"]

    def generate_weave(
        self,
        source_points,
        amplitude,
        cycles,
        samples_per_cycle,
        transverse_axis,
        pattern,
        visible,
        transverse_vector=None,
    ):
        """Preview a weave.

        ``transverse_vector`` is the sensed weave direction (``e_w``) when the
        seam has been touch-corrected.  It must be threaded through here, not
        just into the sequence builder: without it the preview draws a weave
        about a generic tool/world axis while the executed weld runs about the
        wall/floor bisector, which on a fillet joint differ by 45 degrees.
        """
        try:
            if pattern == "circle":
                points = circular_weaving_from_path(
                    source_points, amplitude, cycles, samples_per_cycle,
                    transverse_axis, transverse_vector,
                )
            else:
                points = weaving_from_path(
                    source_points, amplitude, cycles, samples_per_cycle,
                    transverse_axis, transverse_vector, pattern=pattern,
                )
        except ValueError as error:
            self.ui.post(self.ui.error, f"Weave generation failed: {error}")
            return
        self.publish_points(points, visible)
        self.ui.post(self.ui.set_new_points, points, "weave")
        if transverse_vector is None:
            source = f"axis={transverse_axis}"
        else:
            source = (
                "sensed e_w=("
                f"{transverse_vector[0]:+.6f}, {transverse_vector[1]:+.6f}, "
                f"{transverse_vector[2]:+.6f})"
            )
        self.ui.post(
            self.ui.log,
            f"Applied {pattern} weave to taught seam · "
            f"radius/amplitude={amplitude:.3f} m, cycles={cycles}, "
            f"{source}",
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
            name = "weld_start" if endpoint_index == 0 else "weld_end"
            joint_names, positions, pose, provenance = self.capture_measured_teaching_snapshot(
                planning_group, name)
        except (RuntimeError, ValueError, TransformException) as error:
            self.ui.post(self.ui.error, f"Teaching capture rejected: {error}")
            return
        self.ui.post(
            self.ui.apply_linear_tcp,
            endpoint_index,
            pose,
            planning_group,
            tuple(joint_names),
            tuple(positions),
            provenance,
        )

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

    def submit_cartesian_motion(
        self,
        points,
        velocity_scale,
        interpolation_step,
        visualize_path,
        execute_requested,
        reuse_approved_plan,
        planning_group,
        tcp_speed_m_s=0.0,
        linear_motion_profile=False,
    ):
        if not points:
            self.ui.post(self.ui.error, "Create weld points first")
            return
        if not self.cartesian_motion_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(
                self.ui.error,
                "cartesian_path action server unavailable",
            )
            return
        goal = CartesianPath.Goal()
        goal.planning_group = planning_group
        goal.interpolation_step = interpolation_step
        goal.velocity_scale = velocity_scale
        goal.tcp_speed_m_s = float(tcp_speed_m_s)
        goal.execute_requested = execute_requested
        goal.reuse_approved_plan = reuse_approved_plan
        goal.visualize_path = visualize_path
        goal.linear_motion_profile = bool(linear_motion_profile)
        goal.waypoints = points
        self.request_execution = execute_requested
        self.ui.post(
            self.ui.begin,
            velocity_scale,
            execute_requested,
            tcp_speed_m_s,
        )
        self.active_motion_goal = None
        future = self.cartesian_motion_client.send_goal_async(
            goal,
            feedback_callback=self._cartesian_feedback_received,
        )
        future.add_done_callback(self._cartesian_goal_response)

    def run_sequence_cartesian_motion(self, step, execute_requested):
        """Plan or execute one stored path and block only the worker thread."""
        goal = CartesianPath.Goal()
        goal.planning_group = step["planning_group"]
        goal.interpolation_step = step["interpolation_step"]
        goal.velocity_scale = step["velocity_scale"]
        goal.tcp_speed_m_s = float(step.get("tcp_speed_m_s", 0.0))
        goal.execute_requested = bool(execute_requested)
        # A weld path pre-planned before ARC ON executes that exact approved
        # trajectory instead of re-planning while the arc is burning.
        goal.reuse_approved_plan = bool(
            execute_requested and step.get("reuse_approved_plan", False)
        )
        goal.visualize_path = True
        goal.linear_motion_profile = bool(step.get("linear_motion_profile", False))
        goal.waypoints = copy.deepcopy(step["points"])
        goal.waypoint_hold_s = list(step.get("waypoint_hold_s", []))
        touch_guarded = bool(execute_requested and step.get("touch_guard", False))
        arm = step["planning_group"].removesuffix("_manipulator")
        guard_name = step.get("path_kind", "Cartesian approach")
        if touch_guarded:
            if self.node_touch_input_states.get(arm) is None:
                return False, (
                    f"Fastech DI0 is unavailable; {guard_name} was not started"
                )
            if self.node_touch_input_states.get(arm):
                if step.get("accept_initial_touch", False):
                    return True, (
                        f"{guard_name} already at START contact (Fastech DI0 ON) · "
                        "approach skipped and weld stages will continue"
                    )
                if not step.get("allow_initial_touch_motion", False):
                    return False, (
                        f"Fastech DI0 is already ON; {guard_name} was not started"
                    )
                self.ui.post(
                    self.ui.log,
                    f"{guard_name} starts while Fastech DI0 is ON · Cartesian "
                    "retraction allowed; guard waits for a new rising edge",
                )
            self.touch_guard_triggered.clear()
            self.touch_guard_stop_complete.clear()
            self.touch_guard_stop_success = False
            self.active_motion_goal = None
            self.active_touch_guard = (arm, guard_name)
            self.ui.post(
                self.ui.log,
                f"Fastech DI0 stop guard armed only for {guard_name}",
            )
        def trajectory_feedback(message):
            feedback = message.feedback
            phase = str(feedback.phase)
            # CartesianPath also emits dense PLAN_PREVIEW feedback before
            # physical execution.  Never use those virtual poses as actual TCP
            # samples for welding control/logging.
            if "PLAN" in phase.upper() or "PREVIEW" in phase.upper():
                return
            self.ui.record_weld_tcp_sample(
                feedback.current_pose,
                progress=feedback.progress,
                waypoint_index=feedback.waypoint_index,
                phase=phase,
            )

        try:
            result = self._send_action_goal_and_wait(
                self.cartesian_motion_client,
                goal,
                "Cartesian motion",
                on_accepted=(
                    lambda handle: handle.cancel_goal_async()
                    if touch_guarded and self.touch_guard_triggered.is_set()
                    else None
                ),
                feedback_callback=(
                    trajectory_feedback
                    if execute_requested and step.get("record_tcp_trajectory", False)
                    else None
                ),
            )
            if touch_guarded and self.touch_guard_triggered.is_set():
                if not self.touch_guard_stop_complete.wait(timeout=5.0):
                    return False, f"{guard_name} Fastech DI0 stop confirmation timed out"
                if not self.touch_guard_stop_success:
                    return False, f"{guard_name} Fastech DI0 standstill was not confirmed"
                continue_after_touch = bool(
                    step.get("continue_after_touch", False)
                )
                return continue_after_touch, (
                    f"{guard_name} stopped by Fastech DI0 · "
                    + (
                        "continuing with unguarded weld stages"
                        if continue_after_touch
                        else "sequence stopped"
                    )
                )
            return bool(result.success), result.message
        except (RuntimeError, TimeoutError) as error:
            return False, str(error)
        finally:
            if touch_guarded and self.active_touch_guard == (arm, guard_name):
                self.active_touch_guard = None

    def _cleaner_corrected_joint_target(self, step, target_pose):
        """Resolve a shifted cleaner TCP before either preview or execution."""
        if not self.ik_client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError("/compute_ik unavailable for cleaner correction")
        request = GetPositionIK.Request()
        request.ik_request.group_name = step["planning_group"]
        request.ik_request.robot_state.is_diff = True
        request.ik_request.robot_state.joint_state.name = list(step["joint_names"])
        request.ik_request.robot_state.joint_state.position = list(step["positions"])
        request.ik_request.ik_link_name = tip_link_for_group(step["planning_group"])
        request.ik_request.pose_stamped.header.frame_id = "World"
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose = copy.deepcopy(target_pose)
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout = Duration(seconds=3.0).to_msg()
        finished = threading.Event()
        outcome = {}

        def ready(future):
            try:
                outcome["response"] = future.result()
            except Exception as error:
                outcome["error"] = error
            finished.set()

        self.ik_client.call_async(request).add_done_callback(ready)
        if not finished.wait(timeout=4.0):
            raise RuntimeError("/compute_ik timed out for cleaner correction")
        if "error" in outcome:
            raise RuntimeError(f"/compute_ik failed: {outcome['error']}")
        response = outcome["response"]
        if response.error_code.val != 1:
            raise RuntimeError(f"Cleaner correction IK code {response.error_code.val}")
        resolved = dict(zip(
            response.solution.joint_state.name,
            response.solution.joint_state.position,
        ))
        try:
            positions = tuple(resolved[name] for name in step["joint_names"])
        except KeyError as error:
            raise RuntimeError(f"Cleaner correction IK omitted {error}") from error
        actual = self._fk_pose_for_joints(
            step["planning_group"], step["joint_names"], positions,
        )
        error_mm = math.dist(
            _pose_position_tuple(actual), _pose_position_tuple(target_pose)
        ) * 1000.0
        error_deg = math.degrees(quaternion_angular_distance(
            actual.orientation, target_pose.orientation,
        ))
        if error_mm > 2.0 or error_deg > 3.0:
            raise RuntimeError(
                f"Cleaner correction IK/FK error {error_mm:.2f} mm / {error_deg:.2f} deg"
            )
        return positions

    def run_sequence_named_pose(self, step, execute_requested):
        """Plan or plan-and-execute one taught joint pose."""
        if step.get("resolve_tcp_from_joints"):
            try:
                step = dict(step)
                step["tcp_pose"] = self._fk_pose_for_joints(
                    step["planning_group"], step["joint_names"], step["positions"]
                )
            except (RuntimeError, ValueError) as error:
                return False, f"Cleaner teaching FK failed: {error}"
        if step.get("resolve_target_tcp_ik"):
            try:
                positions = self._cleaner_corrected_joint_target(step, step["tcp_pose"])
                step = dict(step, positions=positions)
            except (RuntimeError, ValueError) as error:
                return False, f"Corrected named pose IK failed: {error}"
        try:
            self.validate_named_pose_recall(
                step.get("pose_name"), step["planning_group"],
                step["joint_names"], step["positions"], step.get("tcp_pose"))
        except (RuntimeError, ValueError, TransformException) as error:
            return False, f"Named pose recall blocked: {error}"
        offset = step.get("cleaner_world_offset_m")
        if offset is not None:
            try:
                corrected = copy.deepcopy(step["tcp_pose"])
                for axis in ("x", "y", "z"):
                    setattr(
                        corrected.position, axis,
                        getattr(corrected.position, axis) + float(offset[axis]),
                    )
                corrected_positions = self._cleaner_corrected_joint_target(step, corrected)
                step = dict(step, tcp_pose=corrected, positions=corrected_positions)
                self.ui.post(
                    self.ui.log,
                    f"Cleaner corrected {step['pose_label']} · World XYZ offset "
                    f"({float(offset['x'])*1000:+.1f}, "
                    f"{float(offset['y'])*1000:+.1f}, "
                    f"{float(offset['z'])*1000:+.1f}) mm · "
                    f"target=({corrected.position.x:.4f}, "
                    f"{corrected.position.y:.4f}, {corrected.position.z:.4f}) m",
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                return False, f"Cleaner correction blocked: {error}"
        tcp_target = bool(
            not step.get("use_joint_planning", False)
            and step.get("pose_name") in TCP_POSE_TEACHING_POSES
            and pose_is_valid(step.get("tcp_pose"))
        )
        if tcp_target:
            try:
                current_tcp = self._current_tcp_pose(step["planning_group"])
                points = named_tcp_linear_waypoints(
                    current_tcp, step["tcp_pose"]
                )
            except (TransformException, ValueError) as error:
                return False, f"Named TCP linear path failed: {error}"
            return self.run_sequence_cartesian_motion(
                {
                    "planning_group": step["planning_group"],
                    "interpolation_step": 0.005,
                    "velocity_scale": step["velocity_scale"],
                    "tcp_speed_m_s": float(
                        step.get("tcp_speed_m_s", 0.0)
                    ),
                    "points": points,
                    "path_kind": f"{step['pose_label']} TCP linear",
                    "touch_guard": bool(step.get("touch_guard", True)),
                    "continue_after_touch": bool(
                        step.get("continue_after_touch", False)
                    ),
                    "allow_initial_touch_motion": True,
                },
                execute_requested,
            )
        goal = MoveGroup.Goal()
        goal.request.group_name = step["planning_group"]
        goal.request.num_planning_attempts = int(
            step.get("planning_attempts", 5)
        )
        goal.request.allowed_planning_time = float(
            step.get("planning_time", 5.0)
        )
        goal.request.start_state.is_diff = True
        goal.request.max_velocity_scaling_factor = step["velocity_scale"]
        goal.request.max_acceleration_scaling_factor = step["velocity_scale"]
        constraints = Constraints()
        for name, position in zip(step["joint_names"], step["positions"]):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = position
            constraint.tolerance_above = 0.001
            constraint.tolerance_below = 0.001
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = not bool(execute_requested)
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        touch_guarded = bool(
            execute_requested
            and step.get("pose_name") != "weld_goal_wait"
            and (
                step.get("touch_guard", False)
                or step.get("pose_name") in TOUCH_GUARDED_TEACHING_POSES
            )
        )
        if execute_requested and step.get("pose_name") == "weld_goal_wait":
            self.ui.post(
                self.ui.log,
                "GOAL WAIT retract · Fastech touch guard disabled; "
                "DI contact will not cancel this motion",
            )
        arm = step["planning_group"].removesuffix("_manipulator")
        if touch_guarded:
            if self.node_touch_input_states.get(arm) is None:
                return False, (
                    "Fastech DI0 is unavailable; guarded named motion was not started"
                )
            if self.node_touch_input_states.get(arm):
                self.ui.post(
                    self.ui.log,
                    f"{step['pose_label']} starts while Fastech DI0 is ON · "
                    "allowed for contact retraction; guard waits for a new rising edge",
                )
            self.touch_guard_triggered.clear()
            self.touch_guard_stop_complete.clear()
            self.touch_guard_stop_success = False
            self.active_touch_guard = (arm, step["pose_name"])
            self.ui.post(
                self.ui.log,
                f"Fastech DI0 stop guard armed for {step['pose_label']} approach",
            )
        try:
            result = self._send_action_goal_and_wait(
                self.move_group_client,
                goal,
                "MoveGroup",
                on_accepted=(
                    lambda handle: handle.cancel_goal_async()
                    if touch_guarded and self.touch_guard_triggered.is_set()
                    else None
                ),
            )
            if touch_guarded and self.touch_guard_triggered.is_set():
                if not self.touch_guard_stop_complete.wait(timeout=5.0):
                    return False, "Fastech DI0 stop confirmation timed out"
                if not self.touch_guard_stop_success:
                    return False, "Fastech DI0 standstill was not confirmed"
                continue_after_touch = bool(
                    step.get("continue_after_touch", False)
                )
                return continue_after_touch, (
                    f"{step['pose_label']} stopped by Fastech DI0 · "
                    + (
                        "continuing sequence"
                        if continue_after_touch
                        else "sequence stopped"
                    )
                )
            success = result.error_code.val == 1
            if success and not execute_requested:
                display = DisplayTrajectory()
                display.trajectory_start = result.trajectory_start
                display.trajectory.append(result.planned_trajectory)
                self.display_trajectory_publisher.publish(display)
            return success, (
                f"{step['pose_label']} "
                f"{'reached' if execute_requested else 'planned in RViz'} · "
                "stored joint state"
                if success
                else f"MoveIt code {result.error_code.val}"
            )
        except (RuntimeError, TimeoutError) as error:
            return False, str(error)
        finally:
            if touch_guarded and self.active_touch_guard == (
                arm, step["pose_name"]
            ):
                self.active_touch_guard = None

    def run_sequence_head_motion(self, step, execute_requested):
        """Move J the head to target joint1/joint2 angles (no MoveIt plan)."""
        if not execute_requested:
            return True, (
                "Head move planned · target "
                f"({math.degrees(step['joint1_rad']):.1f}°, "
                f"{math.degrees(step['joint2_rad']):.1f}°)"
            )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(HEAD_JOINT_NAME_ORDER)
        point = JointTrajectoryPoint()
        point.positions = [
            float(step["joint1_rad"]),
            float(step["joint2_rad"]),
        ]
        duration = max(0.1, float(step.get("duration", 2.0)))
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int(
            round((duration - int(duration)) * 1e9)
        )
        goal.trajectory.points = [point]
        try:
            result = self._send_action_goal_and_wait(
                self.joint_trajectory_clients["head"],
                goal,
                "Head motion",
            )
            success = result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
            return success, (
                "Head reached target joint angles"
                if success
                else f"Head motion failed: {result.error_string or result.error_code}"
            )
        except (RuntimeError, TimeoutError) as error:
            return False, str(error)

    def run_sequence_dual_arm_pose(self, step, execute_requested):
        """Plan both arms together as one collision-checked dual_arm target."""
        names = tuple(step["joint_names"])
        positions = tuple(step["positions"])
        expected = {
            f"{arm}_manipulator_joint{index}"
            for arm in ("left", "right") for index in range(1, 7)
        }
        if (step.get("planning_group") != "dual_arm" or len(names) != 12
                or set(names) != expected or len(positions) != 12
                or not all(math.isfinite(value) for value in positions)):
            return False, "Dual-arm target must contain 12 finite named joints"
        goal = MoveGroup.Goal()
        goal.request.group_name = "dual_arm"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 8.0
        goal.request.start_state.is_diff = True
        goal.request.max_velocity_scaling_factor = step["velocity_scale"]
        goal.request.max_acceleration_scaling_factor = step["velocity_scale"]
        constraints = Constraints()
        for name, position in zip(names, positions):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = float(position)
            constraint.tolerance_above = 0.001
            constraint.tolerance_below = 0.001
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = not bool(execute_requested)
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        try:
            result = self._send_action_goal_and_wait(
                self.move_group_client, goal, "Dual-arm pose",
            )
            if result.error_code.val != 1:
                return False, f"Dual-arm MoveIt code {result.error_code.val}"
            if not execute_requested:
                display = DisplayTrajectory()
                display.trajectory_start = result.trajectory_start
                display.trajectory.append(result.planned_trajectory)
                self.display_trajectory_publisher.publish(display)
            return True, f"{step['pose_label']} {'reached' if execute_requested else 'planned'}"
        except (RuntimeError, TimeoutError) as error:
            return False, str(error)

    def run_sequence_spray_motion(self, step, execute_requested):
        """Resolve the taught left-arm center with FK for identical plan/run geometry."""
        try:
            center = self._fk_pose_for_joints(
                "left_manipulator", step["joint_names"], step["positions"],
            )
            if step["spray_kind"] == "line":
                points = straight_waypoints(
                    center, step["distance_m"], 2, "x", "world",
                )
            elif step["spray_kind"] == "circle":
                points = circle_waypoints(
                    center, step["radius_m"], step["unique_points"],
                    step["closed"], step["face_center"], "x",
                )
            else:
                raise ValueError("Unknown spray motion")
            motion = dict(
                planning_group="left_manipulator", points=points,
                interpolation_step=0.005,
                velocity_scale=step["velocity_scale"],
                tcp_speed_m_s=0.0, path_kind=step["path_kind"],
                touch_guard=False,
            )
            return self.run_sequence_cartesian_motion(motion, execute_requested)
        except (RuntimeError, ValueError, TransformException) as error:
            return False, f"Spray path generation failed: {error}"

    def run_sequence_planned_trajectory(self, step, execute_requested):
        """Preview or execute trajectories captured from RViz without replanning."""
        display = DisplayTrajectory()
        display.model_id = step.get("model_id", "")
        display.trajectory_start = copy.deepcopy(step["trajectory_start"])
        display.trajectory = copy.deepcopy(step["trajectories"])
        if not execute_requested:
            self.display_trajectory_publisher.publish(display)
            return True, (
                f"stored RViz plan previewed · "
                f"{len(display.trajectory)} trajectory(s)"
            )
        for index, trajectory in enumerate(display.trajectory, start=1):
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = copy.deepcopy(trajectory)
            try:
                result = self._send_action_goal_and_wait(
                    self.execute_trajectory_client,
                    goal,
                    f"RViz trajectory {index}",
                )
            except (RuntimeError, TimeoutError) as error:
                return False, str(error)
            if result.error_code.val != 1:
                return False, (
                    f"RViz trajectory {index} failed · "
                    f"MoveIt code {result.error_code.val}"
                )
        return True, (
            f"executed exact stored RViz plan · "
            f"{len(display.trajectory)} trajectory(s)"
        )

    def _cartesian_feedback_received(self, message):
        feedback = message.feedback
        self.ui.post(
            self.ui.progress,
            feedback.progress,
            feedback.waypoint_index,
            feedback.current_pose,
            feedback.phase,
        )

    def _cartesian_goal_response(self, future):
        try:
            self.active_motion_goal = future.result()
        except Exception as error:
            self.active_motion_goal = None
            self.ui.post(self.ui.error, str(error))
            return
        if not self.active_motion_goal.accepted:
            self.active_motion_goal = None
            self.ui.post(self.ui.error, "Action goal rejected")
            return
        operation = (
            "approved trajectory execution"
            if self.request_execution
            else "MoveIt plan preview"
        )
        self.ui.post(self.ui.log, f"Action accepted · {operation}")
        result = self.active_motion_goal.get_result_async()
        result.add_done_callback(
            lambda completed, handle=self.active_motion_goal: (
                self._cartesian_result_received(completed, handle)
            )
        )

    def _cartesian_result_received(self, future, goal_handle=None):
        if (
            goal_handle is not None
            and goal_handle is not self.active_motion_goal
        ):
            self.ui.post(
                self.ui.log,
                "Ignored completion from a superseded Cartesian goal",
            )
            return
        result = future.result().result
        if goal_handle is self.active_motion_goal:
            self.active_motion_goal = None
        if result.success:
            self.ui.post(
                self.ui.finish,
                f"SUCCESS · {len(result.sampled_path)} samples · "
                f"{result.message}",
                self.request_execution,
            )
        elif self.active_touch_probe is not None:
            self.ui.post(
                self.ui.log,
                f"Expected probe trajectory interruption · {result.message}",
            )
        else:
            self.ui.post(self.ui.error, result.message)

    def cancel_active_motion(self):
        if self.active_motion_goal is not None:
            self.active_motion_goal.cancel_goal_async()
            self.ui.post(self.ui.log, "Cancel requested")
