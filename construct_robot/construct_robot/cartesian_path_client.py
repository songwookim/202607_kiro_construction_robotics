import argparse

import rclpy
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.node import Node

from construct_msgs.action import CartesianPath


def make_pose(x: float, y: float, z: float) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.w = 1.0
    return pose


class CartesianPathActionClient(Node):
    def __init__(self, planning_group: str) -> None:
        super().__init__("cartesian_path_action_client")
        self._planning_group = planning_group
        self._client = ActionClient(self, CartesianPath, "cartesian_path")
        self.exit_code = 1

    def send_demo_path(self):
        goal = CartesianPath.Goal()
        goal.planning_group = self._planning_group
        goal.interpolation_step = 0.02
        goal.waypoints = [
            make_pose(0.10, 0.00, 0.20),
            make_pose(0.15, 0.05, 0.25),
            make_pose(0.20, 0.00, 0.30),
        ]
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Cartesian path action server unavailable")
            rclpy.shutdown()
            return
        future = self._client.send_goal_async(goal, feedback_callback=self.on_feedback)
        future.add_done_callback(self.on_goal_response)

    def on_feedback(self, message):
        feedback = message.feedback
        self.get_logger().info(
            f"feedback progress={feedback.progress:.2f}, "
            f"waypoint={feedback.waypoint_index}, "
            f"pose=({feedback.current_pose.position.x:.3f}, "
            f"{feedback.current_pose.position.y:.3f}, "
            f"{feedback.current_pose.position.z:.3f})"
        )

    def on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            rclpy.shutdown()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def on_result(self, future):
        result = future.result().result
        self.exit_code = 0 if result.success else 1
        self.get_logger().info(
            f"result success={result.success}, samples={len(result.sampled_path)}, "
            f"final_pose=({result.final_pose.position.x:.3f}, "
            f"{result.final_pose.position.y:.3f}, "
            f"{result.final_pose.position.z:.3f}): {result.message}"
        )
        rclpy.shutdown()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning-group", default="left_manipulator")
    parsed, ros_args = parser.parse_known_args(args=args)
    rclpy.init(args=ros_args)
    node = CartesianPathActionClient(parsed.planning_group)
    node.send_demo_path()
    rclpy.spin(node)
    node.destroy_node()
    return node.exit_code
