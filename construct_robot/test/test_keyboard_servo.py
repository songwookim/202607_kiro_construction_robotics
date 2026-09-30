"""MoveIt Servo keyboard backend: JTC stays active, RB never leaves Servo-J."""

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
        latest_joint_positions={"left_manipulator_joint1": 0.5},
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=Time)
        ),
    )
    bridge.controller_names = {"right": "right_manipulator_controller"}
    bridge.on_status = None
    bridge.lock = threading.Lock()
    bridge.feeds = {"right": {"active": False, "own": False, "names": None, "positions": None}}
    bridge.status = {"right": 0}
    bridge.feed_publishers = {"right": Mock()}
    bridge.twist_publishers = {"right": Mock()}
    bridge.start_clients = {"right": fake_client(
        SimpleNamespace(success=start_ok, message="" if start_ok else "no robot state"))}
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


def enable_with_reference(bridge, positions):
    def deliver():
        time.sleep(0.05)
        bridge._controller_state("right", reference_state(positions))

    threading.Thread(target=deliver).start()
    return bridge.enable("right")


def test_enable_seeds_feed_from_jtc_reference_then_starts_servo():
    bridge = bridge_with()
    success, _message = enable_with_reference(bridge, [0.1] * 6)
    assert success
    assert open_loop_values(bridge) == [True]
    bridge.start_clients["right"].call_async.assert_called_once()
    feed = bridge.feed_publishers["right"].publish.call_args.args[0]
    positions = dict(zip(feed.name, feed.position))
    assert positions["right_manipulator_joint1"] == 0.1
    assert positions["left_manipulator_joint1"] == 0.5  # other arm stays measured


def test_servo_integrates_from_its_own_command_not_the_reference():
    bridge = bridge_with()
    enable_with_reference(bridge, [0.1] * 6)
    output = SimpleNamespace(
        joint_names=[f"right_manipulator_joint{i}" for i in range(1, 7)],
        points=[SimpleNamespace(positions=[0.2] * 6)],
    )
    bridge._servo_output("right", output)
    bridge._controller_state("right", reference_state([0.15] * 6))  # lagging JTC
    bridge._publish_feed("right")
    feed = bridge.feed_publishers["right"].publish.call_args.args[0]
    assert dict(zip(feed.name, feed.position))["right_manipulator_joint1"] == 0.2


def test_rejected_open_loop_never_starts_servo():
    bridge = bridge_with(parameter_ok=False)
    success, message = bridge.enable("right")
    assert not success and "rejected" in message
    bridge.start_clients["right"].call_async.assert_not_called()


def test_failed_servo_start_restores_closed_loop_jtc():
    bridge = bridge_with(start_ok=False)
    success, message = enable_with_reference(bridge, [0.1] * 6)
    assert not success and "no robot state" in message
    assert open_loop_values(bridge) == [True, False]
    assert not bridge.feeds["right"]["active"]


def test_disable_stops_servo_before_restoring_closed_loop():
    bridge = bridge_with()
    enable_with_reference(bridge, [0.1] * 6)
    order = []
    bridge.stop_clients["right"].call_async.side_effect = lambda request: (
        order.append("stop"), FakeFuture(SimpleNamespace(success=True, message="")))[1]
    original = bridge.set_open_loop
    bridge.set_open_loop = lambda arm, enabled: (order.append(("loop", enabled)), original(arm, enabled))[1]
    assert bridge.disable("right")[0]
    assert order == ["stop", ("loop", False)]
    assert not bridge.feeds["right"]["active"]


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
