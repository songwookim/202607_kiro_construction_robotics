import copy
import math
import threading
import time

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from construct_msgs.action import CartesianPath
from construct_robot.cartesian_path_common import (
    PLANNING_GROUP_TIPS,
    pose_is_valid,
    scale_trajectory_speed,
    tip_link_for_group,
)


FUTURE_POLL_PERIOD = 0.01
PLANNING_TIMEOUT = 30.0
EXECUTION_TIMEOUT = 120.0


def interpolate_pose(start: Pose, goal: Pose, ratio: float) -> Pose:
    """Interpolate position and use shortest-path normalized quaternion lerp."""
    pose = Pose()
    pose.position.x = start.position.x + (goal.position.x - start.position.x) * ratio
    pose.position.y = start.position.y + (goal.position.y - start.position.y) * ratio
    pose.position.z = start.position.z + (goal.position.z - start.position.z) * ratio

    start_quaternion = (
        start.orientation.x,
        start.orientation.y,
        start.orientation.z,
        start.orientation.w,
    )
    goal_quaternion = (
        goal.orientation.x,
        goal.orientation.y,
        goal.orientation.z,
        goal.orientation.w,
    )
    if sum(a * b for a, b in zip(start_quaternion, goal_quaternion)) < 0.0:
        goal_quaternion = tuple(-component for component in goal_quaternion)
    quaternion = [
        start_component + (goal_component - start_component) * ratio
        for start_component, goal_component
        in zip(start_quaternion, goal_quaternion)
    ]
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm < 1e-12:
        quaternion = [0.0, 0.0, 0.0, 1.0]
    else:
        quaternion = [component / norm for component in quaternion]
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = quaternion
    return pose


def rotate_vector(quaternion, vector):
    """Rotate a 3-vector by a normalized geometry_msgs Quaternion."""
    qx, qy, qz, qw = (
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    )
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = (qx / norm, qy / norm, qz / norm, qw / norm)
    vx, vy, vz = vector
    # q * v * conjugate(q)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def make_weld_visualization(waypoints, frame, stamp):
    """Build compact RViz 6D-pose markers and corresponding PoseArray."""
    pose_array = PoseArray()
    pose_array.header.frame_id = frame
    pose_array.header.stamp = stamp
    pose_array.poses = list(waypoints)

    markers = MarkerArray()
    delete = Marker()
    delete.action = Marker.DELETEALL
    markers.markers.append(delete)

    if waypoints:
        line = Marker()
        line.header.frame_id = frame
        line.header.stamp = stamp
        line.ns = "weld_seam"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.006
        line.color.r = 1.0
        line.color.g = 0.05
        line.color.b = 0.02
        line.color.a = 1.0
        line.points = [pose.position for pose in waypoints]
        markers.markers.append(line)

    for index, pose in enumerate(waypoints):
        point = Marker()
        point.header.frame_id = frame
        point.header.stamp = stamp
        point.ns = "weld_points"
        point.id = index + 1
        point.type = Marker.SPHERE
        point.action = Marker.ADD
        point.pose = pose
        point.scale.x = point.scale.y = point.scale.z = 0.025
        point.color.r, point.color.g, point.color.b, point.color.a = 1.0, 0.75, 0.0, 1.0
        markers.markers.append(point)

        axes = (
            ("weld_frame_x", (1.0, 0.0, 0.0), (1.0, 0.05, 0.05)),
            ("weld_frame_y", (0.0, 1.0, 0.0), (0.05, 1.0, 0.05)),
            ("weld_frame_z", (0.0, 0.0, 1.0), (0.05, 0.25, 1.0)),
        )
        for namespace, local_axis, color in axes:
            direction = rotate_vector(pose.orientation, local_axis)
            arrow = Marker()
            arrow.header.frame_id = frame
            arrow.header.stamp = stamp
            arrow.ns = namespace
            arrow.id = index + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.points = [
                Point(x=pose.position.x, y=pose.position.y, z=pose.position.z),
                Point(
                    x=pose.position.x + 0.08 * direction[0],
                    y=pose.position.y + 0.08 * direction[1],
                    z=pose.position.z + 0.08 * direction[2],
                ),
            ]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = (
                0.008,
                0.016,
                0.024,
            )
            arrow.color.r, arrow.color.g, arrow.color.b = color
            arrow.color.a = 1.0
            markers.markers.append(arrow)

        label = Marker()
        label.header.frame_id = frame
        label.header.stamp = stamp
        label.ns = "weld_labels"
        label.id = index + 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = pose.position.x
        label.pose.position.y = pose.position.y
        label.pose.position.z = pose.position.z + 0.06
        label.pose.orientation.w = 1.0
        label.scale.z = 0.05
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f"W{index + 1}"
        markers.markers.append(label)
    return markers, pose_array


class CartesianPathActionServer(Node):
    """Visualize, plan, and optionally execute Cartesian paths."""

    def __init__(self) -> None:
        super().__init__("cartesian_path_action_server")
        self.declare_parameter("use_moveit", False)
        self.declare_parameter("execute_motion", True)
        self.declare_parameter("planning_frame", "World")
        self._approved_plan_lock = threading.Lock()
        self._approved_plan_signature = None
        self._approved_plan_response = None
        self._execute_handle_lock = threading.Lock()
        self._active_execute_handle = None
        callback_group = ReentrantCallbackGroup()
        marker_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._marker_publisher = self.create_publisher(
            MarkerArray, "weld_path_markers", marker_qos
        )
        self._pose_publisher = self.create_publisher(
            PoseArray, "weld_6d_poses", marker_qos
        )
        self._display_publisher = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", marker_qos
        )
        self._cartesian_client = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
            callback_group=callback_group,
        )
        self._execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
            callback_group=callback_group,
        )
        self._server = ActionServer(
            self,
            CartesianPath,
            "cartesian_path",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=callback_group,
        )
        self.get_logger().info(
            "Cartesian path action server ready "
            f"(use_moveit={self.get_parameter('use_moveit').value}, "
            f"execute_motion={self.get_parameter('execute_motion').value})"
        )

    def publish_weld_markers(self, waypoints, visible=True):
        frame = self.get_parameter("planning_frame").value
        stamp = self.get_clock().now().to_msg()
        if not visible:
            waypoints = []
        markers, pose_array = make_weld_visualization(
            waypoints, frame, stamp
        )
        self._pose_publisher.publish(pose_array)
        self._marker_publisher.publish(markers)
        self.get_logger().info(
            f"Published {len(waypoints)} weld 6D frames on "
            "/weld_path_markers and /weld_6d_poses"
        )

    @staticmethod
    def _wait_for_future(future, timeout, operation):
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{operation} timed out after {timeout:.0f} s")
            time.sleep(FUTURE_POLL_PERIOD)
        return future.result()

    def plan_with_moveit(
        self,
        request,
        waypoints=None,
        start_state=None,
        publish=True,
    ):
        if not self._cartesian_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/compute_cartesian_path service unavailable")
        cartesian = GetCartesianPath.Request()
        cartesian.header.frame_id = self.get_parameter("planning_frame").value
        if start_state is None:
            cartesian.start_state.is_diff = True
        else:
            cartesian.start_state = copy.deepcopy(start_state)
        cartesian.group_name = request.planning_group
        cartesian.link_name = tip_link_for_group(request.planning_group)
        cartesian.waypoints = (
            request.waypoints if waypoints is None else waypoints
        )
        cartesian.max_step = request.interpolation_step
        cartesian.jump_threshold = 0.0
        cartesian.avoid_collisions = True
        response = self._wait_for_future(
            self._cartesian_client.call_async(cartesian),
            PLANNING_TIMEOUT,
            "MoveIt Cartesian planning",
        )
        if response is None:
            raise RuntimeError("MoveIt Cartesian service returned no response")
        scale_trajectory_speed(response.solution, request.velocity_scale)

        if publish:
            self.publish_trajectories(
                response.start_state,
                [response.solution],
            )
        self.get_logger().info(
            f"MoveIt path fraction={response.fraction:.3f}, "
            f"points={len(response.solution.joint_trajectory.points)}, "
            f"velocity_scale={request.velocity_scale:.2f}"
        )
        return response

    def publish_trajectories(self, start_state, trajectories):
        display = DisplayTrajectory()
        display.model_id = "construct_robot_0528"
        display.trajectory_start = copy.deepcopy(start_state)
        display.trajectory.extend(copy.deepcopy(trajectories))
        self._display_publisher.publish(display)

    @staticmethod
    def plan_signature(request):
        """Return the path/planning values which define an approved plan."""
        pose_values = []
        for pose in request.waypoints:
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
        return (
            request.planning_group,
            request.interpolation_step,
            request.velocity_scale,
            tuple(pose_values),
        )

    def approve_plan(self, request, response):
        with self._approved_plan_lock:
            self._approved_plan_signature = self.plan_signature(request)
            self._approved_plan_response = copy.deepcopy(response)

    def approved_plan(self, request):
        signature = self.plan_signature(request)
        with self._approved_plan_lock:
            if (
                self._approved_plan_response is None
                or signature != self._approved_plan_signature
            ):
                raise RuntimeError(
                    "No matching approved plan. Press Plan Preview again "
                    "after every path or speed change."
                )
            return copy.deepcopy(self._approved_plan_response)

    def consume_approved_plan(self):
        with self._approved_plan_lock:
            self._approved_plan_signature = None
            self._approved_plan_response = None

    def execute_moveit_trajectory(self, trajectory):
        if not self._execute_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("/execute_trajectory action unavailable")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        execute_handle = self._wait_for_future(
            self._execute_client.send_goal_async(goal),
            PLANNING_TIMEOUT,
            "MoveIt execution goal submission",
        )
        if not execute_handle.accepted:
            raise RuntimeError("MoveIt execution goal rejected")
        with self._execute_handle_lock:
            self._active_execute_handle = execute_handle
        try:
            result_wrapper = self._wait_for_future(
                execute_handle.get_result_async(),
                EXECUTION_TIMEOUT,
                "MoveIt trajectory execution",
            )
        finally:
            with self._execute_handle_lock:
                if self._active_execute_handle is execute_handle:
                    self._active_execute_handle = None
        return result_wrapper.result

    def goal_callback(self, goal_request):
        if goal_request.planning_group not in PLANNING_GROUP_TIPS:
            self.get_logger().warning(
                f"Rejected unsupported planning group: "
                f"{goal_request.planning_group}"
            )
            return GoalResponse.REJECT
        if (
            not goal_request.waypoints
            or not math.isfinite(goal_request.interpolation_step)
            or goal_request.interpolation_step <= 0.0
            or not math.isfinite(goal_request.velocity_scale)
            or goal_request.velocity_scale <= 0.0
            or goal_request.velocity_scale > 1.0
        ):
            self.get_logger().warning(
                "Rejected empty path, interpolation step, or velocity scale"
            )
            return GoalResponse.REJECT
        if not all(pose_is_valid(pose) for pose in goal_request.waypoints):
            self.get_logger().warning("Rejected non-finite pose or zero quaternion")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        with self._execute_handle_lock:
            execute_handle = self._active_execute_handle
        if execute_handle is not None:
            try:
                execute_handle.cancel_goal_async()
                self.get_logger().warning(
                    "Forwarded Cartesian cancel to /execute_trajectory"
                )
            except Exception as error:
                self.get_logger().error(
                    f"Failed to cancel /execute_trajectory: {error}"
                )
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        request = goal_handle.request
        result = CartesianPath.Result()
        self.publish_weld_markers(
            request.waypoints,
            request.visualize_path,
        )

        moveit_plan = None
        if self.get_parameter("use_moveit").value:
            try:
                if request.reuse_approved_plan:
                    moveit_plan = self.approved_plan(request)
                    self.get_logger().info(
                        "Using the exact matching GUI-approved trajectory"
                    )
                else:
                    moveit_plan = self.plan_with_moveit(request)
                if moveit_plan.fraction < 0.999:
                    goal_handle.abort()
                    result.success = False
                    result.message = (
                        f"MoveIt planned only {moveit_plan.fraction:.1%} "
                        "of the Cartesian path"
                    )
                    result.final_pose = request.waypoints[-1]
                    return result
                if not request.execute_requested:
                    self.approve_plan(request, moveit_plan)
            except RuntimeError as error:
                goal_handle.abort()
                result.success = False
                result.message = str(error)
                result.final_pose = request.waypoints[-1]
                return result
        start = request.waypoints[0]
        sampled_path = []
        segment_count = len(request.waypoints)

        for waypoint_index, target in enumerate(request.waypoints):
            distance = math.sqrt(
                (target.position.x - start.position.x) ** 2
                + (target.position.y - start.position.y) ** 2
                + (target.position.z - start.position.z) ** 2
            )
            samples = max(1, math.ceil(distance / request.interpolation_step))
            for sample_index in range(1, samples + 1):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = "Cartesian path canceled"
                    result.sampled_path = sampled_path
                    result.final_pose = sampled_path[-1] if sampled_path else start
                    return result

                current = interpolate_pose(start, target, sample_index / samples)
                sampled_path.append(current)
                feedback = CartesianPath.Feedback()
                feedback.current_pose = current
                feedback.waypoint_index = waypoint_index
                feedback.progress = float(
                    (waypoint_index + sample_index / samples) / segment_count
                )
                feedback.phase = "PLAN PREVIEW"
                goal_handle.publish_feedback(feedback)
            start = target

        executed = False
        if moveit_plan is not None and request.execute_requested:
            if not self.get_parameter("execute_motion").value:
                goal_handle.abort()
                result.success = False
                result.message = (
                    "Execution requested but the server was launched with "
                    "execute_motion:=false"
                )
                result.final_pose = request.waypoints[-1]
                result.sampled_path = sampled_path
                return result
            if request.reuse_approved_plan:
                self.consume_approved_plan()
            try:
                execute_result = self.execute_moveit_trajectory(
                    moveit_plan.solution
                )
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = "Cartesian execution canceled"
                    result.final_pose = request.waypoints[-1]
                    result.sampled_path = sampled_path
                    return result
                if execute_result.error_code.val != 1:
                    goal_handle.abort()
                    result.success = False
                    result.message = (
                        "MoveIt execution failed with code "
                        f"{execute_result.error_code.val}"
                    )
                    result.final_pose = request.waypoints[-1]
                    result.sampled_path = sampled_path
                    return result
                executed = True
            except RuntimeError as error:
                goal_handle.abort()
                result.success = False
                result.message = str(error)
                result.final_pose = request.waypoints[-1]
                result.sampled_path = sampled_path
                return result

        goal_handle.succeed()
        result.success = True
        if executed:
            completion = "planned and executed on the active controller"
        elif moveit_plan is not None:
            completion = (
                "planned preview approved for matching path and speed"
            )
        else:
            completion = "sampled only; MoveIt is disabled"
        result.message = (
            f"Path for '{request.planning_group}' {completion} · "
            f"{len(sampled_path)} samples at "
            f"{request.velocity_scale:.0%} speed"
        )
        result.final_pose = request.waypoints[-1]
        result.sampled_path = sampled_path
        return result


def main(args=None):
    rclpy.init(args=args)
    node = CartesianPathActionServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop worker threads before destroying actions, services and
        # publishers.  Otherwise a reconnect SIGINT can interrupt
        # destroy_node() while an executor thread still owns an entity.
        executor.shutdown(timeout_sec=2.0)
        executor.remove_node(node)
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()
