"""Servo-J keyboard jog backend: JTC stays active, RB never leaves Servo-J."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

from builtin_interfaces.msg import Time

from construct_robot.nodes.keyboard_servo import KeyboardServoBridge
from construct_robot.nodes.weld_runtime_node import (
    KEYBOARD_ZERO_BURST_COUNT,
    WeldGuiNode,
)


class FakeFuture:
    def __init__(self, response):
        self.response = response

    def result(self):
        return self.response

    def add_done_callback(self, callback):
        callback(self)


def fake_client(response, name="srv"):
    return SimpleNamespace(
        srv_name=name,
        service_is_ready=Mock(return_value=True),
        wait_for_service=Mock(return_value=True),
        call_async=Mock(return_value=FakeFuture(response)),
    )


def bridge_with(start_ok=True, parameter_ok=True):
    bridge = object.__new__(KeyboardServoBridge)
    bridge.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)),
    )
    bridge.controller_names = {"right": "right_manipulator_controller"}
    bridge.on_status = None
    bridge.lock = threading.Lock()
    bridge.status = {"right": 0}
    bridge.references = {"right": None}
    bridge.reference_changed_at = {"right": 0.0}
    bridge.twist_publishers = {"right": Mock()}
    bridge.start_clients = {"right": fake_client(
        SimpleNamespace(success=start_ok, message="" if start_ok else "no fresh reference"))}
    bridge.stop_clients = {"right": fake_client(SimpleNamespace(success=True, message=""))}
    bridge.parameter_clients = {"right": fake_client(SimpleNamespace(
        results=[SimpleNamespace(successful=parameter_ok, reason="read-only")]))}
    return bridge


def open_loop_values(bridge):
    return [
        call.args[0].parameters[0].value.bool_value
        for call in bridge.parameter_clients["right"].call_async.call_args_list
    ]


def reference_state(positions):
    return SimpleNamespace(
        joint_names=[f"right_manipulator_joint{i}" for i in range(1, 7)],
        reference=SimpleNamespace(positions=positions),
        desired=SimpleNamespace(positions=[]),
    )


def test_enable_sets_open_loop_before_starting_the_jog():
    bridge = bridge_with()
    order = []
    original = bridge.set_open_loop
    bridge.set_open_loop = lambda arm, on: (order.append(("loop", on)), original(arm, on))[1]
    bridge.start_clients["right"].call_async.side_effect = lambda request: (
        order.append("start"), FakeFuture(SimpleNamespace(success=True, message="")))[1]
    assert bridge.enable("right")[0]
    assert order == [("loop", True), "start"]


def test_rejected_open_loop_never_starts_the_jog():
    bridge = bridge_with(parameter_ok=False)
    success, message = bridge.enable("right")
    assert not success and "rejected" in message
    bridge.start_clients["right"].call_async.assert_not_called()


def test_failed_jog_start_restores_closed_loop_jtc():
    bridge = bridge_with(start_ok=False)
    success, message = bridge.enable("right")
    assert not success and "no fresh reference" in message
    assert open_loop_values(bridge) == [True, False]


def test_disable_stops_the_jog_before_restoring_closed_loop():
    bridge = bridge_with()
    order = []
    bridge.stop_clients["right"].call_async.side_effect = lambda request: (
        order.append("stop"), FakeFuture(SimpleNamespace(success=True, message="")))[1]
    original = bridge.set_open_loop
    bridge.set_open_loop = lambda arm, on: (order.append(("loop", on)), original(arm, on))[1]
    assert bridge.disable("right")[0]
    assert order == ["stop", ("loop", False)]


def test_restart_stops_then_starts_without_touching_open_loop():
    bridge = bridge_with()
    assert bridge.restart("right")[0]
    bridge.stop_clients["right"].call_async.assert_called_once()
    bridge.start_clients["right"].call_async.assert_called_once()
    assert open_loop_values(bridge) == []


def test_twist_is_published_in_robot_base_frame():
    bridge = bridge_with()
    bridge.publish_twist("right", (0.001, 0.0, -0.002, 0.0, 0.1, 0.0))
    message = bridge.twist_publishers["right"].publish.call_args.args[0]
    assert message.header.frame_id == "right_manipulator_base_link"
    assert (message.twist.linear.x, message.twist.linear.z, message.twist.angular.y) == (
        0.001, -0.002, 0.1)


def velocity_node(servo):
    node = object.__new__(WeldGuiNode)
    node.keyboard_servo = servo
    node.keyboard_velocity_publishers = {"right": Mock(), "left": Mock()}
    node.keyboard_velocity_lock = threading.Lock()
    node.keyboard_velocity_command = {
        "arm": "right", "values": (0.0,) * 6, "refreshed_monotonic": time.monotonic(),
        "deadman_timeout_s": 10.0, "zero_burst_remaining": 0,
    }
    node.ui = SimpleNamespace(post=Mock(), keyboard_velocity_deadman_stopped=Mock())
    node.get_logger = lambda: Mock()
    return node


def test_servo_backend_streams_held_key_every_timer_tick():
    servo = Mock()
    node = velocity_node(servo)
    node.keyboard_velocity_command["values"] = (0.005, 0, 0, 0, 0, 0)
    for _ in range(3):
        WeldGuiNode._publish_keyboard_velocity(node)
    assert servo.publish_twist.call_count == 3
    node.keyboard_velocity_publishers["right"].publish.assert_not_called()


def test_native_backend_keeps_latched_single_publish():
    node = velocity_node(None)
    node.keyboard_velocity_command["values"] = (0.005, 0, 0, 0, 0, 0)
    for _ in range(3):
        WeldGuiNode._publish_keyboard_velocity(node)
    node.keyboard_velocity_publishers["right"].publish.assert_not_called()


def test_servo_backend_release_sends_zero_burst_then_goes_quiet():
    servo = Mock()
    node = velocity_node(servo)
    node.keyboard_velocity_command["zero_burst_remaining"] = KEYBOARD_ZERO_BURST_COUNT
    for _ in range(KEYBOARD_ZERO_BURST_COUNT + 3):
        WeldGuiNode._publish_keyboard_velocity(node)
    assert servo.publish_twist.call_count == KEYBOARD_ZERO_BURST_COUNT
    assert all(call.args[1] == (0.0,) * 6 for call in servo.publish_twist.call_args_list)


def test_servo_backend_deadman_expiry_publishes_zero():
    servo = Mock()
    node = velocity_node(servo)
    node.keyboard_velocity_command.update(
        values=(0.005, 0, 0, 0, 0, 0), refreshed_monotonic=time.monotonic() - 1.0,
        deadman_timeout_s=0.25)
    WeldGuiNode._publish_keyboard_velocity(node)
    assert servo.publish_twist.call_args.args[1] == (0.0,) * 6
    node.ui.post.assert_called_once()


def switching_node(stopped=True):
    node = object.__new__(WeldGuiNode)
    node.keyboard_servo = Mock()
    node.keyboard_servo.enable.return_value = (True, "started")
    node.keyboard_servo.disable.return_value = (True, "stopped")
    node.controller_switch_client = Mock()
    node.wait_for_controller_state = Mock(return_value=True)
    node.wait_until_arm_stopped = Mock(return_value=stopped)
    node.clear_keyboard_velocity = Mock()
    return node


def test_servo_enable_never_switches_controllers():
    node = switching_node()
    assert WeldGuiNode.set_keyboard_velocity_controller_enabled(node, "right", True) == (
        True, "started")
    node.controller_switch_client.call_async.assert_not_called()
    node.wait_for_controller_state.assert_called_once_with(
        "right_manipulator_controller", "active", timeout=1.0)


def test_servo_enable_requires_active_jtc():
    node = switching_node()
    node.wait_for_controller_state.return_value = False
    success, message = WeldGuiNode.set_keyboard_velocity_controller_enabled(node, "right", True)
    assert not success and "not active" in message
    node.keyboard_servo.enable.assert_not_called()


def test_servo_disable_always_leaves_servo_even_if_still_moving():
    node = switching_node(stopped=False)
    success, message = WeldGuiNode.set_keyboard_velocity_controller_enabled(node, "right", False)
    assert not success and "had not stopped" in message
    node.clear_keyboard_velocity.assert_called_once()
    node.keyboard_servo.disable.assert_called_once_with("right")
    node.controller_switch_client.call_async.assert_not_called()


def test_command_stopped_follows_jtc_reference_changes():
    bridge = bridge_with()
    bridge._controller_state("right", reference_state([0.1] * 6))
    assert not bridge.command_stopped("right", timeout=0.02, stable_s=0.1)
    time.sleep(0.12)
    bridge._controller_state("right", reference_state([0.1] * 6))  # unchanged
    assert bridge.command_stopped("right", timeout=0.0, stable_s=0.1)


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def selection_gui(held=True, servo=True):
    from construct_robot.gui.weld_action_gui import WeldActionGui

    gui = object.__new__(WeldActionGui)
    gui.root = SimpleNamespace(focus_get=lambda: object(), after_cancel=Mock())
    gui.keyboard_jog_enabled = Var(True)
    gui.keyboard_jog_selection = Var("Z")
    gui.keyboard_jog_linear_speed = Var(90.0)
    gui.keyboard_jog_angular_speed = Var(30.0)
    gui.keyboard_jog_frame = Var("World")
    gui.planning_group = Var("right_manipulator")
    gui.keyboard_jog_status = SimpleNamespace(set=Mock())
    gui.keyboard_velocity_active_key = "Up"
    gui.keyboard_velocity_arm = "right"
    gui.keyboard_velocity_switching = False
    gui.keyboard_teaching_capture_in_progress = False
    gui.keyboard_release_after_id = None
    gui.keyboard_ros_physical_key = "Up" if held else None
    gui.keyboard_ros_input_last_at = time.monotonic()
    gui.keyboard_ros_dispatching = False
    gui.keyboard_stop_generation = 0
    gui.sequence_running = False
    gui.log = Mock()
    gui.error = Mock()
    gui.post = Mock()
    gui.node = Mock(active_motion_goal=None)
    gui.node.keyboard_teaching_uses_servo.return_value = servo
    gui.node.resolve_keyboard_velocity.return_value = (0.09, 0.0, 0.0, 0.0, 0.0, 0.0)
    return gui


def test_axis_key_switches_held_jog_without_stopping_in_servo_mode():
    gui = selection_gui()
    assert gui.keyboard_jog_selection_key(SimpleNamespace(keysym="1")) == "break"
    assert gui.keyboard_jog_selection.get() == "X"
    assert gui.node.resolve_keyboard_velocity.call_args.args[1] == "X"
    gui.node.set_keyboard_velocity.assert_called_once_with(
        "right", (0.09, 0.0, 0.0, 0.0, 0.0, 0.0))
    gui.node.clear_keyboard_velocity.assert_not_called()
    assert gui.keyboard_velocity_active_key == "Up"


def test_axis_key_after_release_only_changes_selection():
    gui = selection_gui(held=False)
    gui.keyboard_jog_selection_key(SimpleNamespace(keysym="2"))
    assert gui.keyboard_jog_selection.get() == "Y"
    gui.node.set_keyboard_velocity.assert_not_called()
    gui.node.clear_keyboard_velocity.assert_called_once()


def test_native_backend_axis_key_still_stops_the_jog():
    gui = selection_gui(servo=False)
    gui.keyboard_jog_selection_key(SimpleNamespace(keysym="1"))
    gui.node.set_keyboard_velocity.assert_not_called()
    gui.node.clear_keyboard_velocity.assert_called_once()


def test_servo_stop_check_uses_command_and_never_move_stop():
    gui = selection_gui()
    gui.post = lambda fn, *args: fn(*args)
    gui.keyboard_velocity_active_key = None
    gui.keyboard_stop_generation = 7
    gui.node.wait_until_keyboard_command_stopped.return_value = False
    gui.node.restart_keyboard_servo.return_value = (True, "restarted")
    started = []
    import construct_robot.gui.weld_action_gui as module
    original = module.threading.Thread
    module.threading.Thread = lambda target, args, daemon: SimpleNamespace(
        start=lambda: (started.append(target), target(*args)))
    try:
        gui._verify_keyboard_jog_stop_worker("right", 7)
    finally:
        module.threading.Thread = original
    gui.node.wait_until_keyboard_command_stopped.assert_called_once()
    gui.node.wait_until_arm_stopped.assert_not_called()
    gui.node.restart_keyboard_servo.assert_called_once_with("right")
    gui.node.request_direct_motion_stop.assert_not_called()
