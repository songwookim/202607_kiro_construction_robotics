import argparse
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

from construct_msgs.action import CartesianPath
from construct_robot.cartesian_path_common import (
    PLANNING_GROUP_TIPS,
    tip_link_for_group,
    weaving_waypoints,
)


def make_pose(x: float, y: float, z: float, q=(0.0, 0.0, 0.0, 1.0)) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = q
    return pose


class CartesianPathActionClient(Node):
    def __init__(self, planning_group, scenario, options) -> None:
        super().__init__("laser_weld_path_action_client")
        self._planning_group = planning_group
        self._scenario = scenario
        self._options = options
        self._client = ActionClient(self, CartesianPath, "cartesian_path")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.exit_code = 1

    def scanner_path_from_current_tcp(self):
        tip = tip_link_for_group(self._planning_group)
        deadline = time.monotonic() + 5.0
        transform = None
        while time.monotonic() < deadline:
            try:
                transform = self._tf_buffer.lookup_transform(
                    "World", tip, rclpy.time.Time(), timeout=Duration(seconds=0.2)
                )
                break
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)
        if transform is None:
            raise RuntimeError(f"Unable to transform World -> {tip}")

        p = transform.transform.translation
        q = transform.transform.rotation
        quaternion = (q.x, q.y, q.z, q.w)
        # A scanner seam beginning 10 mm from the current TCP and extending
        # 30 mm along World Y. Keeping the current orientation makes this a
        # reachable visualization/execution test from any valid start pose.
        return [
            make_pose(p.x, p.y + 0.01, p.z, quaternion),
            make_pose(p.x, p.y + 0.02, p.z, quaternion),
            make_pose(p.x, p.y + 0.03, p.z, quaternion),
        ]

    def send_demo_path(self):
        goal = CartesianPath.Goal()
        goal.planning_group = self._planning_group
        goal.interpolation_step = 0.02
        goal.velocity_scale = self._options.velocity_scale
        goal.execute_requested = not self._options.plan_only
        goal.reuse_approved_plan = self._options.execute_approved
        goal.visualize_path = not self._options.hide_path
        goal.enable_arc = self._options.enable_arc
        goal.weld_current_a = float(self._options.current_raw)
        goal.weld_voltage_out_condition = 1
        goal.weld_voltage = float(self._options.voltage_raw) / 10.0
        if self._scenario == "laser-live-straight":
            try:
                goal.waypoints = self.scanner_path_from_current_tcp()
            except RuntimeError as error:
                self.get_logger().error(str(error))
                rclpy.shutdown()
                return
            self.get_logger().info(
                "Laser scanner supplied 3 live TCP-relative collinear 6D poses"
            )
        elif self._scenario == "laser-live-weave":
            try:
                straight = self.scanner_path_from_current_tcp()
            except RuntimeError as error:
                self.get_logger().error(str(error))
                rclpy.shutdown()
                return
            start = straight[0]
            goal.waypoints = weaving_waypoints(
                start,
                length=0.08,
                amplitude=0.003,
                cycles=4,
                samples_per_cycle=8,
            )
            self.get_logger().info(
                "Generated live 80 mm seam with ±3 mm, 4-cycle weaving"
            )
        elif self._scenario == "laser-straight":
            # Simulated scanner output: three weld TCP poses on one straight
            # seam, with a constant tool orientation.
            scanner_orientation = (0.0, 0.70710678, 0.0, 0.70710678)
            goal.waypoints = [
                make_pose(0.40, -0.30, 1.20, scanner_orientation),
                make_pose(0.45, -0.30, 1.20, scanner_orientation),
                make_pose(0.50, -0.30, 1.20, scanner_orientation),
            ]
            self.get_logger().info(
                "Laser scanner supplied 3 collinear weld 6D poses"
            )
        else:
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
            f"{feedback.current_pose.position.z:.3f}), "
            f"q=({feedback.current_pose.orientation.x:.3f}, "
            f"{feedback.current_pose.orientation.y:.3f}, "
            f"{feedback.current_pose.orientation.z:.3f}, "
            f"{feedback.current_pose.orientation.w:.3f})"
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
            f"{result.final_pose.position.z:.3f}), "
            f"q=({result.final_pose.orientation.x:.3f}, "
            f"{result.final_pose.orientation.y:.3f}, "
            f"{result.final_pose.orientation.z:.3f}, "
            f"{result.final_pose.orientation.w:.3f}): {result.message}"
        )
        rclpy.shutdown()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planning-group",
        choices=tuple(PLANNING_GROUP_TIPS),
        default="left_manipulator",
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "demo",
            "laser-straight",
            "laser-live-straight",
            "laser-live-weave",
        ),
        default="laser-straight",
    )
    parser.add_argument("--velocity-scale", type=float, default=0.2)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--plan-only", action="store_true")
    execution.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--hide-path", action="store_true")
    parser.add_argument("--enable-arc", action="store_true")
    parser.add_argument("--current-raw", type=int, default=0)
    parser.add_argument("--voltage-raw", type=int, default=0)
    parser.add_argument("--v-offset-raw", type=int, default=0)
    parsed, ros_args = parser.parse_known_args(args=args)
    if not 0.0 < parsed.velocity_scale <= 1.0:
        parser.error("--velocity-scale must be in (0, 1]")
    for name in ("current_raw", "voltage_raw", "v_offset_raw"):
        if not 0 <= getattr(parsed, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be 0..65535")
    rclpy.init(args=ros_args)
    node = CartesianPathActionClient(
        parsed.planning_group,
        parsed.scenario,
        parsed,
    )
    node.send_demo_path()
    rclpy.spin(node)
    node.destroy_node()
    return node.exit_code
