import math
import time

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from construct_msgs.action import CartesianPath


def interpolate_pose(start: Pose, goal: Pose, ratio: float) -> Pose:
    """Linearly interpolate position and normalized quaternion components."""
    pose = Pose()
    pose.position.x = start.position.x + (goal.position.x - start.position.x) * ratio
    pose.position.y = start.position.y + (goal.position.y - start.position.y) * ratio
    pose.position.z = start.position.z + (goal.position.z - start.position.z) * ratio

    quaternion = [
        start.orientation.x + (goal.orientation.x - start.orientation.x) * ratio,
        start.orientation.y + (goal.orientation.y - start.orientation.y) * ratio,
        start.orientation.z + (goal.orientation.z - start.orientation.z) * ratio,
        start.orientation.w + (goal.orientation.w - start.orientation.w) * ratio,
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


class CartesianPathActionServer(Node):
    """Dry-run Cartesian path server; it never commands robot hardware."""

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
        markers = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)

        line = Marker()
        line.header.frame_id = frame
        line.ns = "weld_seam"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.008
        line.color.r = 1.0
        line.color.g = 0.15
        line.color.b = 0.05
        line.color.a = 1.0
        line.points = [pose.position for pose in waypoints]
        markers.markers.append(line)

        for index, pose in enumerate(waypoints):
            point = Marker()
            point.header.frame_id = frame
            point.ns = "weld_points"
            point.id = index + 1
            point.type = Marker.SPHERE
            point.action = Marker.ADD
            point.pose = pose
            point.scale.x = point.scale.y = point.scale.z = 0.035
            point.color.r = 1.0
            point.color.g = 0.8
            point.color.b = 0.0
            point.color.a = 1.0
            markers.markers.append(point)
        self._marker_publisher.publish(markers)
        self.get_logger().info(
            f"Published {len(waypoints)} weld points on /weld_path_markers"
        )

    def plan_with_moveit(self, request):
        if not self._cartesian_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/compute_cartesian_path service unavailable")
        cartesian = GetCartesianPath.Request()
        cartesian.header.frame_id = self.get_parameter("planning_frame").value
        cartesian.start_state.is_diff = True
        cartesian.group_name = request.planning_group
        cartesian.link_name = (
            "right_manipulator_ee_point"
            if request.planning_group == "right_manipulator"
            else "left_manipulator_ee_point"
        )
        cartesian.waypoints = request.waypoints
        cartesian.max_step = request.interpolation_step
        cartesian.jump_threshold = 0.0
        cartesian.avoid_collisions = True
        future = self._cartesian_client.call_async(cartesian)
        while not future.done():
            time.sleep(0.01)
        response = future.result()
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
        send_future = self._execute_client.send_goal_async(goal)
        while not send_future.done():
            time.sleep(0.01)
        execute_handle = send_future.result()
        if not execute_handle.accepted:
            raise RuntimeError("MoveIt execution goal rejected")
        result_future = execute_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.01)
        return result_future.result().result

    def goal_callback(self, goal_request):
        if not goal_request.waypoints or goal_request.interpolation_step <= 0.0:
            self.get_logger().warning("Rejected empty path or non-positive step")
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
