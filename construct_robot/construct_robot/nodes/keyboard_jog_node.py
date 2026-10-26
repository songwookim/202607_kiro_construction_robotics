"""Stream keyboard Cartesian jogs into each arm's JointTrajectoryController.

The JTC stays active during keyboard teaching, so RB never leaves Servo-J
(no Idle<->Servo-J start/end kicks).  This node owns the integrated joint
command itself: every cycle it ramps the requested base-frame TCP twist,
converts it to a joint step with the arm Jacobian, and sends the JTC a short
trajectory.  No state is fed back through other processes, so the jog speed
is exact and cannot stutter from message timing.

Every message is stamped and starts *on the spline the JTC is already
executing*: point 0 is this node's own last trajectory sampled at the stamp
(exactly how the JTC interpolates it), point 1 is the next integrated
command one cycle later.  The JTC (2.x) samples a stamped message inside its
points, so a new message continues the motion seamlessly.  An unstamped
message would instead be anchored at the JTC's previous command for its
first control cycle: one 8 ms cycle without motion at every message, i.e.
a velocity ripple at the stream rate (0, 1.33v, 1.33v, 1.33v at 4 cycles per
message).  A twist change is streamed at once instead of waiting for the next
cycle, and each message ends with one constant-velocity cycle plus a smooth
stop, so a late or missing cycle neither stalls nor kicks the arm.

If the JTC reference ever leaves this node's trajectory (a hold after a
tolerance violation, another goal), the stream restarts from the JTC
reference instead of jumping back onto its own trajectory.

Per arm (``left``/``right``):
  /<arm>_keyboard_jog/delta_twist_cmds  geometry_msgs/TwistStamped (in)
  /<arm>_keyboard_jog/start, /stop      std_srvs/Trigger
  /<arm>_keyboard_jog/status            std_msgs/Int8 (keyboard_jog_kinematics codes)
"""

from collections import deque

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from builtin_interfaces.msg import Time as TimeMsg
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
# Published trajectories kept to compare against later JTC references.
PLAN_HISTORY = 6


def _duration(seconds):
    whole = int(seconds)
    return DurationMsg(sec=whole, nanosec=int(round((seconds - whole) * 1e9)))


def _stamp(seconds):
    sec, nanosec = divmod(int(round(seconds * 1e9)), 1_000_000_000)
    return TimeMsg(sec=sec, nanosec=nanosec)


def _seconds(time_message):
    return float(time_message.sec) + float(time_message.nanosec) * 1e-9


class StreamPlan:
    """Knots of one published trajectory, sampled the way the JTC samples it.

    Points carry positions and velocities only, so the JTC interpolates each
    segment with a cubic Hermite polynomial; past the last point it holds
    that point's position.
    """

    def __init__(self, times, positions, velocities):
        self.times = list(times)
        self.positions = [np.asarray(q, dtype=float) for q in positions]
        self.velocities = [np.asarray(v, dtype=float) for v in velocities]

    @property
    def start(self):
        return self.times[0]

    def sample(self, t):
        """Commanded (position, velocity) at absolute time t."""
        if t <= self.times[0]:
            return self.positions[0].copy(), self.velocities[0].copy()
        for index in range(len(self.times) - 1):
            t0, t1 = self.times[index], self.times[index + 1]
            if t < t1:
                return self._hermite(index, t - t0, t1 - t0)
        return self.positions[-1].copy(), np.zeros_like(self.positions[-1])

    def _hermite(self, index, s, duration):
        q0, v0 = self.positions[index], self.velocities[index]
        q1, v1 = self.positions[index + 1], self.velocities[index + 1]
        # Same coefficients as joint_trajectory_controller's cubic spline.
        c2 = (-3.0 * q0 + 3.0 * q1 - 2.0 * v0 * duration - v1 * duration) / duration ** 2
        c3 = (2.0 * q0 - 2.0 * q1 + v0 * duration + v1 * duration) / duration ** 3
        position = q0 + v0 * s + c2 * s ** 2 + c3 * s ** 3
        velocity = v0 + 2.0 * c2 * s + 3.0 * c3 * s ** 2
        return position, velocity


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
        self.reference_stamp = None
        self.twist = np.zeros(6)
        self.twist_at = None
        self.twist_stamp = None
        self.last_twist_stamp = None
        self.moving = False
        self.status = JOG_OK
        self.plans = deque(maxlen=PLAN_HISTORY)
        self.last_tick = None
        self.checked_reference_stamp = None
        self.ramp = TwistRamp(
            node.get_parameter("max_linear_accel").value,
            node.get_parameter("max_angular_accel").value,
        )
        controller = f"{arm}_manipulator_controller"
        prefix = f"/{arm}_keyboard_jog"
        self.trajectory_publisher = node.create_publisher(
            JointTrajectory, f"/{controller}/joint_trajectory", 1
        )
        self.status_publisher = node.create_publisher(Int8, f"{prefix}/status", 10)
        node.create_subscription(
            JointTrajectoryControllerState, f"/{controller}/controller_state",
            self._controller_state, 1,
        )
        node.create_subscription(
            TwistStamped, f"{prefix}/delta_twist_cmds", self._twist, 1
        )
        node.create_service(Trigger, f"{prefix}/start", self._start)
        node.create_service(Trigger, f"{prefix}/stop", self._stop)

    def _parameter(self, name):
        return self.node.get_parameter(name).value

    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------- inputs
    def _controller_state(self, message):
        names = list(message.joint_names)
        positions = list(message.reference.positions) or list(message.desired.positions)
        if names and len(positions) == len(names):
            order = dict(zip(names, positions))
            if all(name in order for name in self.chain.joint_names):
                self.reference = np.array([order[name] for name in self.chain.joint_names])
                self.reference_at = self.node.get_clock().now()
                header = getattr(message, "header", None)
                stamp = getattr(header, "stamp", None)
                self.reference_stamp = (
                    _seconds(stamp) if stamp is not None and (stamp.sec or stamp.nanosec) else None
                )

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
        stamp_message = getattr(message.header, "stamp", None)
        stamp = _seconds(stamp_message) if stamp_message is not None else 0.0
        if stamp:
            age = self._now() - stamp
            if (age < 0.0 or age > self._parameter("command_timeout_s")
                    or (self.last_twist_stamp is not None and stamp < self.last_twist_stamp)):
                self.node.get_logger().warning(
                    f"{self.arm} keyboard jog discarded stale/out-of-order twist (age {age * 1000:.1f} ms)",
                    throttle_duration_sec=2.0,
                )
                return
            self.last_twist_stamp = stamp
            if age > 0.04:
                self.node.get_logger().warning(
                    f"{self.arm} keyboard jog twist delivery delayed {age * 1000:.1f} ms",
                    throttle_duration_sec=2.0,
                )
        changed = not np.array_equal(values, self.twist)
        self.twist = values
        self.twist_at = self.node.get_clock().now()
        self.twist_stamp = stamp or None
        if changed and self.active:
            # Press, release and direction changes reach the JTC now, not at
            # the next stream cycle.
            self.tick(self._parameter("cycle_s"))

    def _start(self, _request, response):
        max_age = self._parameter("reference_max_age_s")
        if self.reference_at is None or (
            self.node.get_clock().now() - self.reference_at
        ).nanoseconds * 1e-9 > max_age:
            response.success = False
            response.message = f"no fresh {self.arm}_manipulator_controller reference"
            return response
        self.q_cmd = self.reference.copy()
        self.twist = np.zeros(6)
        self.twist_at = None
        self.twist_stamp = None
        self.last_twist_stamp = self._now()
        self.ramp.reset()
        self.moving = False
        self.plans.clear()
        self.last_tick = None
        self.checked_reference_stamp = self.reference_stamp
        self.active = True
        self._set_status(JOG_OK)
        self._activity_changed()
        response.success = True
        response.message = f"{self.arm} keyboard jog started from JTC reference"
        return response

    def _stop(self, _request, response):
        if self.active and self.moving:
            now = self._now()
            position, velocity = self._command_at(now)
            self._publish_stop(now, position, velocity)
        self.active = False
        self.moving = False
        self.ramp.reset()
        self._activity_changed()
        response.success = True
        response.message = f"{self.arm} keyboard jog stopped"
        return response

    def _activity_changed(self):
        sync = getattr(self.node, "_sync_timer", None)
        if sync is not None:
            sync()

    # --------------------------------------------------------------- cycle
    def tick(self, dt):
        if not self.active:
            return
        now = self._now()
        gap = None if self.last_tick is None else now - self.last_tick
        self.last_tick = now
        elapsed = dt if gap is None else min(max(gap, 0.0), dt)
        timeout = self._parameter("command_timeout_s")
        fresh = self.twist_at is not None and (
            self.node.get_clock().now() - self.twist_at
        ).nanoseconds * 1e-9 <= timeout
        if self.twist_stamp is not None:
            fresh = fresh and 0.0 <= now - self.twist_stamp <= timeout
        target = self.twist if fresh else np.zeros(6)
        # Read every cycle so the ramp can be tuned live (ros2 param set).
        self.ramp.linear_accel = self._parameter("max_linear_accel")
        self.ramp.angular_accel = self._parameter("max_angular_accel")
        if self._follow_jtc_reference():
            self.ramp.reset()
        position, velocity = self._command_at(now)
        if gap is not None and gap > 2.0 * dt and self.plans:
            # A stalled cycle let the JTC run into the stop tail: ramp on from
            # the twist it actually commands, never jump back to the old one.
            self.ramp.current = self.chain.jacobian(position) @ velocity
            if gap > max(3.0 * dt, 0.04):
                self.node.get_logger().warning(
                    f"{self.arm} keyboard jog stream delayed {gap * 1000:.1f} ms "
                    f"(target {dt * 1000:.1f} ms)",
                    throttle_duration_sec=2.0,
                )
        twist = self.ramp.step(target, elapsed)
        if not np.any(twist):
            if self.moving:
                self._publish_stop(now, position, velocity)
                self.moving = False
            if not np.any(target):
                self._set_status(JOG_OK)
            return
        dq, status = jog_step(
            self.chain, position, twist, dt,
            velocity_scale=self._parameter("joint_velocity_scale"),
            limit_margin=self._parameter("joint_limit_margin_rad"),
            slow_sigma=self._parameter("singularity_slow_sigma"),
            stop_sigma=self._parameter("singularity_stop_sigma"),
        )
        self._set_status(status)
        if status in (JOG_HALTED_AT_SINGULARITY, JOG_HALTED_AT_JOINT_LIMIT):
            self.ramp.reset()
            if self.moving:
                self._publish_stop(now, position, velocity)
                self.moving = False
            return
        self.q_cmd = position + dq
        self._publish_segment(now, position, velocity, dq / dt, dt)
        self.moving = True

    def _command_at(self, now):
        """The JTC command at `now` while it follows this node's stream."""
        if not self.plans:
            return self.q_cmd.copy(), np.zeros_like(self.q_cmd)
        return self.plans[-1].sample(now)

    def _follow_jtc_reference(self):
        """Restart from the JTC reference once it no longer follows the stream.

        Returns True when the stream was re-anchored.
        """
        stamp = self.reference_stamp
        if (
            not self.plans or stamp is None or self.reference is None
            or stamp == self.checked_reference_stamp
        ):
            return False
        self.checked_reference_stamp = stamp
        plan = next((plan for plan in reversed(self.plans) if plan.start <= stamp), None)
        if plan is None:
            return False
        expected, _velocity = plan.sample(stamp)
        deviation = float(np.max(np.abs(expected - self.reference)))
        if deviation <= self._parameter("reference_deviation_rad"):
            return False
        self.node.get_logger().warning(
            f"{self.arm} JTC reference left the keyboard stream by {deviation:.4f} rad; "
            "continuing from the JTC reference",
            throttle_duration_sec=1.0,
        )
        self.q_cmd = self.reference.copy()
        self.plans.clear()
        return True

    def _publish(self, now, times, positions, velocities):
        message = JointTrajectory()
        message.header.stamp = _stamp(now)
        message.joint_names = list(self.chain.joint_names)
        for t, q, v in zip(times, positions, velocities):
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in q]
            point.velocities = [float(value) for value in v]
            point.time_from_start = _duration(t)
            message.points.append(point)
        self.plans.append(StreamPlan([now + t for t in times], positions, velocities))
        self.trajectory_publisher.publish(message)

    def _publish_segment(self, now, position, velocity, target_velocity, dt):
        # 0: where the JTC is now, 1: the next command, 2: one more cycle at
        # the same velocity (covers a late cycle), 3: smooth stop if the
        # stream itself stalls.  The next message replaces 2 and 3.
        tail = self._parameter("stall_stop_s")
        target = position + target_velocity * dt
        extended = target + target_velocity * dt
        stopped = extended + target_velocity * (0.5 * tail)
        zero = np.zeros_like(target_velocity)
        self._publish(
            now,
            (0.0, dt, 2.0 * dt, 2.0 * dt + tail),
            (position, target, extended, stopped),
            (velocity, target_velocity, target_velocity, zero),
        )

    def _publish_stop(self, now, position, velocity):
        # After the release ramp the velocity is already ~0; a halt at a
        # singularity or joint limit settles over the old 32 ms hold.
        settle = 2.0 * self._parameter("cycle_s")
        self._publish(
            now,
            (0.0, settle),
            (position, position + velocity * (0.5 * settle)),
            (velocity, np.zeros_like(velocity)),
        )

    def _set_status(self, status):
        if status != self.status:
            self.status = status
            self.status_publisher.publish(Int8(data=int(status)))


class KeyboardJogNode(Node):
    def __init__(self):
        super().__init__("keyboard_jog_node")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("arms", list(ARMS))
        # One JTC cycle (125 Hz).  Stamped messages continue the spline, so
        # the rate sets how finely a ramping twist is re-integrated; presses,
        # releases and direction changes are streamed at once regardless.
        self.declare_parameter("cycle_s", 0.008)
        self.declare_parameter("command_timeout_s", 0.2)
        # 100 mm/s in ~67 ms; RB Servo-J filtering smooths the edges.
        self.declare_parameter("max_linear_accel", 1.5)      # m/s^2
        self.declare_parameter("max_angular_accel", 6.0)     # rad/s^2
        self.declare_parameter("joint_velocity_scale", 0.5)  # of URDF limits
        self.declare_parameter("joint_limit_margin_rad", 0.05)
        self.declare_parameter("singularity_slow_sigma", 0.05)
        self.declare_parameter("singularity_stop_sigma", 0.015)
        self.declare_parameter("reference_max_age_s", 0.5)
        # Deceleration at the end of every message: used only if this node
        # stops streaming while a key is held.
        self.declare_parameter("stall_stop_s", 0.1)
        self.declare_parameter("reference_deviation_rad", 0.003)
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
        # Runs only while an arm is in keyboard teaching (see _sync_timer).
        self.timer = self.create_timer(self.dt, self._tick)
        self._sync_timer()
        self.get_logger().info(
            f"Keyboard jog streaming ready for {', '.join(self.arms)} "
            f"({1.0 / self.dt:.1f} Hz)"
        )

    def _tick(self):
        for arm in self.arms.values():
            arm.tick(self.dt)

    def _sync_timer(self):
        """Stream only while some arm jogs: idle cycles cost CPU for nothing."""
        if any(arm.active for arm in self.arms.values()):
            if self.timer.is_canceled():
                self.timer.reset()
        elif not self.timer.is_canceled():
            self.timer.cancel()


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
