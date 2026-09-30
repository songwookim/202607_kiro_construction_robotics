"""Stream keyboard Cartesian jogs into each arm's JointTrajectoryController.

The JTC stays active during keyboard teaching, so RB never leaves Servo-J
(no Idle<->Servo-J start/end kicks).  This node owns the integrated joint
command itself: every cycle it ramps the requested base-frame TCP twist,
converts it to a joint step with the arm Jacobian, and sends the JTC a short
trajectory from that command.  No state is fed back through other processes,
so the jog speed is exact and cannot stutter from message timing.

Per arm (``left``/``right``):
  /<arm>_keyboard_jog/delta_twist_cmds  geometry_msgs/TwistStamped (in)
  /<arm>_keyboard_jog/start, /stop      std_srvs/Trigger
  /<arm>_keyboard_jog/status            std_msgs/Int8 (keyboard_jog_kinematics codes)
"""

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from construct_robot.core.keyboard_jog_kinematics import (
    JOG_HALTED_AT_JOINT_LIMIT,
    JOG_HALTED_AT_SINGULARITY,
    JOG_OK,
    ArmChain,
    TwistRamp,
    jog_step,
)


ARMS = ("left", "right")


def _duration(seconds):
    whole = int(seconds)
    return DurationMsg(sec=whole, nanosec=int(round((seconds - whole) * 1e9)))


class ArmJog:
    def __init__(self, node, arm, chain):
        self.node = node
        self.arm = arm
        self.chain = chain
        self.base_frame = f"{arm}_manipulator_base_link"
        self.active = False
        self.q_cmd = None
        self.reference = None
        self.reference_at = None
        self.twist = np.zeros(6)
        self.twist_at = None
        self.moving = False
        self.status = JOG_OK
        self.ramp = TwistRamp(
            node.get_parameter("max_linear_accel").value,
            node.get_parameter("max_angular_accel").value,
        )
        controller = f"{arm}_manipulator_controller"
        prefix = f"/{arm}_keyboard_jog"
        self.trajectory_publisher = node.create_publisher(
            JointTrajectory, f"/{controller}/joint_trajectory", 10
        )
        self.status_publisher = node.create_publisher(Int8, f"{prefix}/status", 10)
        node.create_subscription(
            JointTrajectoryControllerState, f"/{controller}/controller_state",
            self._controller_state, 10,
        )
        node.create_subscription(
            TwistStamped, f"{prefix}/delta_twist_cmds", self._twist, 10
        )
        node.create_service(Trigger, f"{prefix}/start", self._start)
        node.create_service(Trigger, f"{prefix}/stop", self._stop)

    # ------------------------------------------------------------- inputs
    def _controller_state(self, message):
        names = list(message.joint_names)
        positions = list(message.reference.positions) or list(message.desired.positions)
        if names and len(positions) == len(names):
            order = dict(zip(names, positions))
            if all(name in order for name in self.chain.joint_names):
                self.reference = np.array([order[name] for name in self.chain.joint_names])
                self.reference_at = self.node.get_clock().now()

    def _twist(self, message):
        frame = message.header.frame_id
        if frame not in ("", self.base_frame):
            self.node.get_logger().warning(
                f"{self.arm} keyboard jog ignores twist in frame '{frame}' "
                f"(expected {self.base_frame})",
                throttle_duration_sec=2.0,
            )
            return
        t = message.twist
        values = np.array([t.linear.x, t.linear.y, t.linear.z,
                           t.angular.x, t.angular.y, t.angular.z], dtype=float)
        if not np.all(np.isfinite(values)):
            return
        self.twist = values
        self.twist_at = self.node.get_clock().now()

    def _start(self, _request, response):
        max_age = self.node.get_parameter("reference_max_age_s").value
        if self.reference_at is None or (
            self.node.get_clock().now() - self.reference_at
        ).nanoseconds * 1e-9 > max_age:
            response.success = False
            response.message = f"no fresh {self.arm}_manipulator_controller reference"
            return response
        self.q_cmd = self.reference.copy()
        self.twist = np.zeros(6)
        self.twist_at = None
        self.ramp.reset()
        self.moving = False
        self.active = True
        self._set_status(JOG_OK)
        response.success = True
        response.message = f"{self.arm} keyboard jog started from JTC reference"
        return response

    def _stop(self, _request, response):
        if self.active and self.moving:
            self._publish_hold()
        self.active = False
        self.moving = False
        self.ramp.reset()
        response.success = True
        response.message = f"{self.arm} keyboard jog stopped"
        return response

    # --------------------------------------------------------------- cycle
    def tick(self, dt):
        if not self.active:
            return
        timeout = self.node.get_parameter("command_timeout_s").value
        fresh = self.twist_at is not None and (
            self.node.get_clock().now() - self.twist_at
        ).nanoseconds * 1e-9 <= timeout
        target = self.twist if fresh else np.zeros(6)
        twist = self.ramp.step(target, dt)
        if not np.any(twist):
            if self.moving:
                self._publish_hold()
                self.moving = False
            if not np.any(target):
                self._set_status(JOG_OK)
            return
        dq, status = jog_step(
            self.chain, self.q_cmd, twist, dt,
            velocity_scale=self.node.get_parameter("joint_velocity_scale").value,
            limit_margin=self.node.get_parameter("joint_limit_margin_rad").value,
            slow_sigma=self.node.get_parameter("singularity_slow_sigma").value,
            stop_sigma=self.node.get_parameter("singularity_stop_sigma").value,
        )
        self._set_status(status)
        if status in (JOG_HALTED_AT_SINGULARITY, JOG_HALTED_AT_JOINT_LIMIT):
            self.ramp.reset()
            if self.moving:
                self._publish_hold()
                self.moving = False
            return
        self.q_cmd = self.q_cmd + dq
        self._publish_segment(dq / dt, dt)
        self.moving = True

    def _publish_segment(self, velocity, dt):
        # Two points: the next command, plus one extrapolated cycle so a late
        # message never lets the JTC stall; the next message replaces both.
        message = JointTrajectory()
        message.joint_names = list(self.chain.joint_names)
        for step in (1, 2):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in self.q_cmd + velocity * dt * (step - 1)]
            point.velocities = [float(v) for v in velocity]
            point.time_from_start = _duration(dt * step)
            message.points.append(point)
        self.trajectory_publisher.publish(message)

    def _publish_hold(self):
        message = JointTrajectory()
        message.joint_names = list(self.chain.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in self.q_cmd]
        point.velocities = [0.0] * len(self.q_cmd)
        point.time_from_start = _duration(
            self.node.get_parameter("cycle_s").value
        )
        message.points.append(point)
        self.trajectory_publisher.publish(message)

    def _set_status(self, status):
        if status != self.status:
            self.status = status
            self.status_publisher.publish(Int8(data=int(status)))


class KeyboardJogNode(Node):
    def __init__(self):
        super().__init__("keyboard_jog_node")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("arms", list(ARMS))
        # 4 JTC cycles at 125 Hz: each segment is sampled several times
        # before the next one replaces it.
        self.declare_parameter("cycle_s", 0.032)
        self.declare_parameter("command_timeout_s", 0.2)
        self.declare_parameter("max_linear_accel", 0.5)      # m/s^2
        self.declare_parameter("max_angular_accel", 3.0)     # rad/s^2
        self.declare_parameter("joint_velocity_scale", 0.5)  # of URDF limits
        self.declare_parameter("joint_limit_margin_rad", 0.05)
        self.declare_parameter("singularity_slow_sigma", 0.05)
        self.declare_parameter("singularity_stop_sigma", 0.015)
        self.declare_parameter("reference_max_age_s", 0.5)
        description = str(self.get_parameter("robot_description").value)
        if not description:
            raise RuntimeError("keyboard_jog_node needs the robot_description parameter")
        self.arms = {}
        for arm in self.get_parameter("arms").value:
            chain = ArmChain(
                description,
                f"{arm}_manipulator_base_link",
                f"{arm}_manipulator_ee_point",
            )
            self.arms[arm] = ArmJog(self, arm, chain)
        self.dt = float(self.get_parameter("cycle_s").value)
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"Keyboard jog streaming ready for {', '.join(self.arms)} "
            f"({1.0 / self.dt:.1f} Hz)"
        )

    def _tick(self):
        for arm in self.arms.values():
            arm.tick(self.dt)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = KeyboardJogNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
