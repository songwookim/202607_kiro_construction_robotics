import math
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node

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
        self._server = ActionServer(
            self,
            CartesianPath,
            "cartesian_path",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )
        self.get_logger().info("Dry-run Cartesian path action server ready")

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
        start = Pose()
        start.orientation.w = 1.0
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

        goal_handle.succeed()
        result.success = True
        result.message = (
            f"Dry-run path for '{request.planning_group}' completed "
            f"with {len(sampled_path)} samples"
        )
        result.final_pose = request.waypoints[-1]
        result.sampled_path = sampled_path
        return result


def main(args=None):
    rclpy.init(args=args)
    node = CartesianPathActionServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
