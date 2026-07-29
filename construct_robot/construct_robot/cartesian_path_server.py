import math
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
    """Build conspicuous RViz markers and the corresponding PoseArray."""
    pose_array = PoseArray()
    pose_array.header.frame_id = frame
    pose_array.header.stamp = stamp
    pose_array.poses = list(waypoints)

    markers = MarkerArray()
    delete = Marker()
    delete.action = Marker.DELETEALL
    markers.markers.append(delete)

    line = Marker()
    line.header.frame_id = frame
    line.header.stamp = stamp
    line.ns = "weld_seam"
    line.id = 0
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.scale.x = 0.018
    line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.05, 0.02, 1.0
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
        point.scale.x = point.scale.y = point.scale.z = 0.06
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
                    x=pose.position.x + 0.25 * direction[0],
                    y=pose.position.y + 0.25 * direction[1],
                    z=pose.position.z + 0.25 * direction[2],
                ),
            ]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.022, 0.045, 0.065
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
        label.pose.position.z = pose.position.z + 0.16
        label.pose.orientation.w = 1.0
        label.scale.z = 0.13
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f"W{index + 1}"
        markers.markers.append(label)
    return markers, pose_array


class CartesianPathActionServer(Node):
    """Visualize, plan, and optionally execute Cartesian weld paths."""

    def __init__(self) -> None:
        super().__init__("cartesian_path_action_server")
        self.declare_parameter("use_moveit", False)
        self.declare_parameter("execute_motion", False)
        self.declare_parameter("planning_frame", "World")
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

    def publish_weld_markers(self, waypoints):
        frame = self.get_parameter("planning_frame").value
        stamp = self.get_clock().now().to_msg()
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

    def plan_with_moveit(self, request):
        if not self._cartesian_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/compute_cartesian_path service unavailable")
        cartesian = GetCartesianPath.Request()
        cartesian.header.frame_id = self.get_parameter("planning_frame").value
        cartesian.start_state.is_diff = True
        cartesian.group_name = request.planning_group
        cartesian.link_name = tip_link_for_group(request.planning_group)
        cartesian.waypoints = request.waypoints
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

        display = DisplayTrajectory()
        display.model_id = "construct_robot_0528"
        display.trajectory_start = response.start_state
        display.trajectory.append(response.solution)
        self._display_publisher.publish(display)
        self.get_logger().info(
            f"MoveIt path fraction={response.fraction:.3f}, "
            f"points={len(response.solution.joint_trajectory.points)}"
        )
        return response

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
        result_wrapper = self._wait_for_future(
            execute_handle.get_result_async(),
            EXECUTION_TIMEOUT,
            "MoveIt trajectory execution",
        )
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
        ):
            self.get_logger().warning(
                "Rejected empty path or invalid interpolation step"
            )
            return GoalResponse.REJECT
        if not all(pose_is_valid(pose) for pose in goal_request.waypoints):
            self.get_logger().warning("Rejected non-finite pose or zero quaternion")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        request = goal_handle.request
        result = CartesianPath.Result()
        self.publish_weld_markers(request.waypoints)

        moveit_response = None
        if self.get_parameter("use_moveit").value:
            try:
                moveit_response = self.plan_with_moveit(request)
                if moveit_response.fraction < 0.999:
                    goal_handle.abort()
                    result.success = False
                    result.message = (
                        f"MoveIt planned only {moveit_response.fraction:.1%} "
                        "of the scanner path"
                    )
                    result.final_pose = request.waypoints[-1]
                    return result
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
                goal_handle.publish_feedback(feedback)
                time.sleep(0.01)
            start = target

        if moveit_response is not None and self.get_parameter("execute_motion").value:
            try:
                execute_result = self.execute_moveit_trajectory(
                    moveit_response.solution
                )
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
            except RuntimeError as error:
                goal_handle.abort()
                result.success = False
                result.message = str(error)
                result.final_pose = request.waypoints[-1]
                result.sampled_path = sampled_path
                return result

        goal_handle.succeed()
        result.success = True
        result.message = (
            f"Path for '{request.planning_group}' completed "
            f"with {len(sampled_path)} samples"
        )
        result.final_pose = request.waypoints[-1]
        result.sampled_path = sampled_path
        return result


def main(args=None):
    rclpy.init(args=args)
    node = CartesianPathActionServer()
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
