"""GUI-side client for keyboard teaching over RB Servo-J streaming.

The native backend exchanges the arm's JTC for a Cartesian velocity
controller (RB ``jog_robot_l``).  Each exchange moves RB between its Servo-J
and Idle hold modes, and RB's first Servo-J sample after Idle kicks the arm
(~1e-3 rad on J2, "start kick") although the commanded position is
continuous.  This backend keeps the JTC active instead:
:mod:`construct_robot.nodes.keyboard_jog_node` turns keyboard twists into JTC
topic trajectories, so RB never leaves Servo-J.

The JTC must start each streamed trajectory from its last command
(``open_loop_control``); re-seeding from measured joints drifts against the
command and snaps when the stream stops.  It is enabled only while keyboard
teaching owns the arm and restored on exit.

Keyboard teaching also switches RB Servo-J to a tighter filter profile: the
welding default follows the command ~160 ms late (28 mm run-on after a 100 mm/s
release); the keyboard profile ~75 ms with the same smoothness.  The welding
default is restored on exit, so trajectories never run on the keyboard profile.
"""

import threading
import time

from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import TwistStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rbpodo_msgs.srv import SetJointPositionControllerConfig
from std_msgs.msg import Int8
from std_srvs.srv import Trigger


KEYBOARD_SERVO_ARMS = ("left", "right")
KEYBOARD_SERVO_SERVICE_TIMEOUT_S = 3.0
KEYBOARD_SERVO_COMMAND_STILL_RAD = 1e-7
# RB Servo-J (t1, t2, gain, alpha).  WELDING matches rbpodo_hardware's
# JointPositionControllerConfig defaults used by every trajectory.
SERVO_J_WELDING_PROFILE = (0.04, 0.10, 0.5, 0.2)
SERVO_J_KEYBOARD_PROFILE = (0.02, 0.10, 0.8, 0.5)
SERVO_STATUS_TEXT = {
    0: "OK",
    1: "slowing near singularity",
    2: "HALTED at singularity",
    5: "HALTED at joint limit",
}


def keyboard_jog_prefix(arm):
    return f"/{arm}_keyboard_jog"


def keyboard_servo_base_frame(arm):
    return f"{arm}_manipulator_base_link"


class KeyboardServoBridge:
    """Per-arm jog start/stop, twist output and command-standstill check."""

    def __init__(self, node, controller_names, on_status=None):
        self.node = node
        self.controller_names = dict(controller_names)
        self.on_status = on_status
        self.lock = threading.Lock()
        self.status = {arm: 0 for arm in KEYBOARD_SERVO_ARMS}
        # Last JTC reference and when it last changed: keyboard stop checks
        # use the command, since RB Servo-J settles ~0.5 s behind it.
        self.references = {arm: None for arm in KEYBOARD_SERVO_ARMS}
        self.reference_changed_at = {arm: 0.0 for arm in KEYBOARD_SERVO_ARMS}
        self.twist_publishers = {}
        self.start_clients = {}
        self.stop_clients = {}
        self.parameter_clients = {}
        self.servo_j_clients = {}
        for arm in KEYBOARD_SERVO_ARMS:
            prefix = keyboard_jog_prefix(arm)
            controller = self.controller_names[arm]
            self.twist_publishers[arm] = node.create_publisher(
                TwistStamped, f"{prefix}/delta_twist_cmds", 10
            )
            self.start_clients[arm] = node.create_client(Trigger, f"{prefix}/start")
            self.stop_clients[arm] = node.create_client(Trigger, f"{prefix}/stop")
            self.parameter_clients[arm] = node.create_client(
                SetParameters, f"/{controller}/set_parameters"
            )
            self.servo_j_clients[arm] = node.create_client(
                SetJointPositionControllerConfig,
                f"/{arm}_rbpodo_hardware/set_joint_position_controller_config",
            )
            node.create_subscription(
                JointTrajectoryControllerState,
                f"/{controller}/controller_state",
                lambda message, arm=arm: self._controller_state(arm, message),
                10,
            )
            node.create_subscription(
                Int8,
                f"{prefix}/status",
                lambda message, arm=arm: self._jog_status(arm, message),
                10,
            )

    def _controller_state(self, arm, message):
        positions = list(message.reference.positions) or list(
            message.desired.positions
        )
        if not positions:
            return
        with self.lock:
            previous = self.references[arm]
            if previous is None or max(
                abs(a - b) for a, b in zip(previous, positions)
            ) > KEYBOARD_SERVO_COMMAND_STILL_RAD:
                self.references[arm] = positions
                self.reference_changed_at[arm] = time.monotonic()

    def _jog_status(self, arm, message):
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
        """Switch the JTC to open-loop starts, then start the jog stream."""
        if not self.available(arm):
            return False, (
                f"{keyboard_jog_prefix(arm)} or "
                f"/{self.controller_names[arm]} parameters unavailable"
            )
        ok, message = self.set_open_loop(arm, True)
        if not ok:
            return False, message
        ok, message = self._trigger(self.start_clients[arm], "jog start")
        if not ok:
            self.set_open_loop(arm, False)
            return False, message
        # Lower RB lag is a comfort feature: keyboard teaching still works on
        # the welding profile if the hardware service is unavailable.
        _profile_ok, profile_message = self.set_servo_j_profile(
            arm, SERVO_J_KEYBOARD_PROFILE
        )
        return True, (
            f"{arm} keyboard jog started · "
            f"{self.controller_names[arm]} stays active · {profile_message}"
        )

    def disable(self, arm):
        """Stop the jog stream, then restore the JTC's closed-loop start state."""
        stop_ok, stop_message = self._trigger(self.stop_clients[arm], "jog stop")
        loop_ok, loop_message = self.set_open_loop(arm, False)
        profile_ok, profile_message = self.set_servo_j_profile(
            arm, SERVO_J_WELDING_PROFILE
        )
        return (
            stop_ok and loop_ok and profile_ok,
            f"{stop_message}; {loop_message}; {profile_message}",
        )

    def set_servo_j_profile(self, arm, profile):
        t1, t2, gain, alpha = profile
        name = "keyboard" if profile == SERVO_J_KEYBOARD_PROFILE else "welding"
        response, error = self._call(
            self.servo_j_clients[arm],
            SetJointPositionControllerConfig.Request(t1=t1, t2=t2, gain=gain, alpha=alpha),
        )
        if error is not None or not response.success:
            return False, f"Servo-J {name} profile NOT applied ({error or 'refused'})"
        return True, f"Servo-J {name} profile"

    def restart(self, arm):
        """Stop and restart the jog: it holds the last command, RB stays in Servo-J."""
        stop_ok, stop_message = self._trigger(self.stop_clients[arm], "jog stop")
        if not stop_ok:
            return False, stop_message
        ok, message = self._trigger(self.start_clients[arm], "jog start")
        return ok, f"{stop_message}; {message}"

    def command_stopped(self, arm, timeout, stable_s):
        """True once the JTC reference has not changed for stable_s."""
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                changed_at = self.reference_changed_at[arm]
                seen = self.references[arm] is not None
            now = time.monotonic()
            if seen and now - changed_at >= stable_s:
                return True
            if now >= deadline:
                return False
            time.sleep(0.01)

    def publish_twist(self, arm, values):
        """Publish a robot-base-frame twist (m/s, rad/s) to the arm's jog stream."""
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
