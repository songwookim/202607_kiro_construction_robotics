import math
import time

import rclpy
from moveit_msgs.msg import RobotState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


RIGHT_JOINT_NAMES = tuple(
    f"right_manipulator_joint{index}" for index in range(1, 7)
)


def has_complete_right_state(message):
    positions = dict(zip(message.name, message.position))
    return all(
        name in positions and math.isfinite(positions[name])
        for name in RIGHT_JOINT_NAMES
    )


class RvizGoalStateSync(Node):
    """Initialize MoveIt's RViz goal state from live joint feedback."""

    def __init__(self):
        super().__init__("rviz_goal_state_sync")
        self.declare_parameter("poll_period", 0.2)
        self.declare_parameter("rviz_settle_delay", 2.0)
        self._timer = None
        self._rviz_seen_at = None
        self._waiting_for_rviz_logged = False
        self._latest_joint_state = None
        self._publisher = self.create_publisher(
            RobotState,
            "/rviz/moveit/update_custom_goal_state",
            10,
        )
        self._subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "Waiting for complete right-arm /joint_states before syncing "
            "the RViz Goal State"
        )

    def _on_joint_state(self, message):
        if not has_complete_right_state(message):
            return
        goal_state = RobotState()
        goal_state.joint_state = message
        goal_state.is_diff = False
        self._latest_joint_state = goal_state
        if self._timer is not None:
            return
        period = max(
            0.1,
            float(self.get_parameter("poll_period").value),
        )
        self._timer = self.create_timer(period, self._publish_update)
        self.get_logger().info(
            "Live right-arm state received; initializing RViz Goal State"
        )

    def _publish_update(self):
        if self._publisher.get_subscription_count() == 0:
            if not self._waiting_for_rviz_logged:
                self.get_logger().info(
                    "Waiting for the MoveIt RViz Goal State subscriber"
                )
                self._waiting_for_rviz_logged = True
            return
        now = time.monotonic()
        if self._rviz_seen_at is None:
            self._rviz_seen_at = now
            self.get_logger().info(
                "MoveIt RViz subscriber found; waiting for plugin startup"
            )
            return
        settle_delay = max(
            0.0,
            float(self.get_parameter("rviz_settle_delay").value),
        )
        if now - self._rviz_seen_at < settle_delay:
            return
        if self._latest_joint_state is None:
            return
        self._publisher.publish(self._latest_joint_state)
        self._timer.cancel()
        self.get_logger().info(
            "RViz custom Goal State synchronized to measured robot state"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RvizGoalStateSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
