"""MoveIt Servo backend for keyboard teaching.

The native backend exchanges the arm's JTC for a Cartesian velocity
controller (RB ``jog_robot_l``).  Each exchange moves RB between its Servo-J
and Idle hold modes, and RB's first Servo-J sample after Idle kicks the arm
(~1e-3 rad on J2, "start kick") although the commanded position is
continuous.  This backend keeps the JTC active instead: MoveIt Servo turns
keyboard twists into JTC topic trajectories, so RB never leaves Servo-J.

Two details were required on the real RB arm:

* Humble Servo re-seeds from its joint topic every cycle.  RB Servo-J lags its
  command, so seeding from measured joints stalls the jog.  Servo is fed its
  own last command instead, seeded from the JTC reference when it starts.
* The JTC must start each streamed trajectory from its last command
  (``open_loop_control``).  Re-seeding from measured joints drifts against the
  command and snaps when the stream stops.  It is enabled only while keyboard
  teaching owns the arm and restored on exit.
"""

import threading
import time

from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import TwistStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory


KEYBOARD_SERVO_ARMS = ("left", "right")
KEYBOARD_SERVO_FEED_PERIOD_S = 0.02
KEYBOARD_SERVO_REFERENCE_TIMEOUT_S = 1.0
KEYBOARD_SERVO_SERVICE_TIMEOUT_S = 3.0
SERVO_STATUS_TEXT = {
    0: "OK",
    1: "slowing near singularity",
    2: "HALTED at singularity",
    3: "slowing near collision",
    4: "HALTED for collision",
    5: "HALTED at joint limit",
    6: "leaving singularity",
}


def keyboard_servo_name(arm):
    return f"{arm}_keyboard_servo"


def keyboard_servo_base_frame(arm):
    return f"{arm}_manipulator_base_link"


class KeyboardServoBridge:
    """Per-arm Servo control, twist output and self-fed joint topic."""

    def __init__(self, node, controller_names, on_status=None):
        self.node = node
        self.controller_names = dict(controller_names)
        self.on_status = on_status
        self.lock = threading.Lock()
        self.feeds = {
            arm: {
                "active": False,
                "own": False,
                "names": None,
                "positions": None,
            }
            for arm in KEYBOARD_SERVO_ARMS
        }
        self.status = {arm: 0 for arm in KEYBOARD_SERVO_ARMS}
        self.twist_publishers = {}
        self.feed_publishers = {}
        self.start_clients = {}
        self.stop_clients = {}
        self.parameter_clients = {}
        for arm in KEYBOARD_SERVO_ARMS:
            servo = keyboard_servo_name(arm)
            controller = self.controller_names[arm]
            self.twist_publishers[arm] = node.create_publisher(
                TwistStamped, f"/{servo}/delta_twist_cmds", 10
            )
            self.feed_publishers[arm] = node.create_publisher(
                JointState, f"/{servo}/joint_states", 10
            )
            self.start_clients[arm] = node.create_client(
                Trigger, f"/{servo}/start_servo"
            )
            self.stop_clients[arm] = node.create_client(
                Trigger, f"/{servo}/stop_servo"
            )
            self.parameter_clients[arm] = node.create_client(
                SetParameters, f"/{controller}/set_parameters"
            )
            node.create_subscription(
                JointTrajectoryControllerState,
                f"/{controller}/controller_state",
                lambda message, arm=arm: self._controller_state(arm, message),
                10,
            )
            node.create_subscription(
                JointTrajectory,
                f"/{controller}/joint_trajectory",
                lambda message, arm=arm: self._servo_output(arm, message),
                50,
            )
            node.create_subscription(
                Int8,
                f"/{servo}/status",
                lambda message, arm=arm: self._servo_status(arm, message),
                10,
            )
        node.create_timer(KEYBOARD_SERVO_FEED_PERIOD_S, self._publish_feeds)

    # ------------------------------------------------------------------ feed
    def _controller_state(self, arm, message):
        positions = list(message.reference.positions) or list(
            message.desired.positions
        )
        if not positions:
            return
        with self.lock:
            feed = self.feeds[arm]
            if not feed["active"] or feed["own"]:
                return
            feed["names"] = list(message.joint_names)
            feed["positions"] = positions
        self._publish_feed(arm)

    def _servo_output(self, arm, message):
        if not message.points:
            return
        with self.lock:
            feed = self.feeds[arm]
            if not feed["active"]:
                return
            feed["names"] = list(message.joint_names)
            feed["positions"] = list(message.points[-1].positions)
            feed["own"] = True
        self._publish_feed(arm)

    def _publish_feeds(self):
        for arm in KEYBOARD_SERVO_ARMS:
            self._publish_feed(arm)

    def _publish_feed(self, arm):
        with self.lock:
            feed = self.feeds[arm]
            if not feed["active"] or feed["positions"] is None:
                return
            names, positions = feed["names"], feed["positions"]
        # Other joints (second arm, head) keep their measured values so
        # Servo's collision checking sees the real scene.
        merged = dict(self.node.latest_joint_positions)
        merged.update(zip(names, positions))
        message = JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(merged)
        message.position = [float(value) for value in merged.values()]
        self.feed_publishers[arm].publish(message)

    def _servo_status(self, arm, message):
        code = int(message.data)
        with self.lock:
            if code == self.status[arm]:
                return
            self.status[arm] = code
        text = SERVO_STATUS_TEXT.get(code, f"status {code}")
        if self.on_status is not None:
            self.on_status(arm, code, text)

    # --------------------------------------------------------------- control
    def available(self, arm):
        return arm in KEYBOARD_SERVO_ARMS and all(
            clients[arm].service_is_ready()
            for clients in (
                self.start_clients,
                self.stop_clients,
                self.parameter_clients,
            )
        )

    def enable(self, arm):
        """Seed the feed from the JTC reference and start Servo."""
        if not self.available(arm):
            return False, (
                f"/{keyboard_servo_name(arm)} or "
                f"/{self.controller_names[arm]} parameters unavailable"
            )
        ok, message = self.set_open_loop(arm, True)
        if not ok:
            return False, message
        with self.lock:
            self.feeds[arm].update(
                active=True, own=False, names=None, positions=None
            )
        deadline = time.monotonic() + KEYBOARD_SERVO_REFERENCE_TIMEOUT_S
        while time.monotonic() < deadline:
            with self.lock:
                seeded = self.feeds[arm]["positions"] is not None
            if seeded:
                break
            time.sleep(0.02)
        else:
            self._abort(arm)
            return False, f"no {self.controller_names[arm]} reference received"
        # Let Servo's state monitor receive a few seeded samples first.
        time.sleep(0.1)
        ok, message = self._trigger(self.start_clients[arm], "start_servo")
        if not ok:
            self._abort(arm)
            return False, message
        return True, (
            f"{keyboard_servo_name(arm)} started · "
            f"{self.controller_names[arm]} stays active"
        )

    def disable(self, arm):
        """Stop Servo, then restore the JTC's closed-loop start state."""
        stop_ok, stop_message = self._trigger(self.stop_clients[arm], "stop_servo")
        with self.lock:
            self.feeds[arm].update(active=False, own=False)
        loop_ok, loop_message = self.set_open_loop(arm, False)
        ok = stop_ok and loop_ok
        return ok, f"{stop_message}; {loop_message}"

    def publish_twist(self, arm, values):
        """Publish a robot-base-frame twist (m/s, rad/s) to the arm's Servo."""
        message = TwistStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = keyboard_servo_base_frame(arm)
        (
            message.twist.linear.x,
            message.twist.linear.y,
            message.twist.linear.z,
            message.twist.angular.x,
            message.twist.angular.y,
            message.twist.angular.z,
        ) = (float(value) for value in values)
        self.twist_publishers[arm].publish(message)

    def set_open_loop(self, arm, enabled):
        controller = self.controller_names[arm]
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="open_loop_control",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL,
                    bool_value=bool(enabled),
                ),
            )
        ]
        response, error = self._call(self.parameter_clients[arm], request)
        if error is not None:
            return False, f"{controller} open_loop_control: {error}"
        results = list(response.results)
        if not results or not results[0].successful:
            reason = results[0].reason if results else "no result"
            return False, f"{controller} open_loop_control rejected: {reason}"
        return True, f"{controller} open_loop_control={bool(enabled)}"

    def _abort(self, arm):
        with self.lock:
            self.feeds[arm].update(active=False, own=False)
        self.set_open_loop(arm, False)

    def _trigger(self, client, name):
        response, error = self._call(client, Trigger.Request())
        if error is not None:
            return False, f"{name}: {error}"
        if not response.success:
            return False, f"{name} refused: {response.message}"
        return True, f"{name} OK"

    @staticmethod
    def _call(client, request):
        if not client.wait_for_service(timeout_sec=0.5):
            return None, f"{client.srv_name} unavailable"
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                outcome["response"] = future.result()
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        client.call_async(request).add_done_callback(response_ready)
        if not finished.wait(timeout=KEYBOARD_SERVO_SERVICE_TIMEOUT_S):
            return None, "service response timed out"
        if "error" in outcome:
            return None, outcome["error"]
        return outcome["response"], None
