"""Taught-pose arrival ends the post-trajectory Servo-J hold (Keyboard ON kick fix)."""

from types import SimpleNamespace
from unittest.mock import Mock

from construct_robot.nodes.weld_runtime_node import WeldGuiNode


class FakeFuture:
    def __init__(self, response):
        self.response = response

    def result(self):
        return self.response

    def add_done_callback(self, callback):
        callback(self)


def node_with(stopped=True, available=True, response=None):
    node = object.__new__(WeldGuiNode)
    node.ui = SimpleNamespace(post=lambda fn, *a: fn(*a), log=Mock())
    node.wait_until_arm_stopped = Mock(return_value=stopped)
    client = SimpleNamespace(
        wait_for_service=Mock(return_value=available),
        call_async=Mock(return_value=FakeFuture(
            response or SimpleNamespace(success=True, message="released")
        )),
    )
    node.servo_hold_release_clients = {"right": client, "left": client}
    return node, client


def test_hold_is_released_only_after_measured_standstill():
    node, client = node_with()
    assert WeldGuiNode.release_servo_hold_after_arrival(node, "right") == (True, "released")
    node.wait_until_arm_stopped.assert_called_once_with("right", timeout=2.0)
    client.call_async.assert_called_once()
    assert "released" in node.ui.log.call_args.args[0]


def test_hold_is_kept_while_arm_still_moves():
    node, client = node_with(stopped=False)
    assert WeldGuiNode.release_servo_hold_after_arrival(node, "right")[0] is False
    client.call_async.assert_not_called()


def test_missing_hardware_service_keeps_existing_behaviour():
    node, client = node_with(available=False)
    assert WeldGuiNode.release_servo_hold_after_arrival(node, "right") == (
        False, "release_servo_hold unavailable"
    )
    client.call_async.assert_not_called()


def test_refused_release_is_reported_as_kept():
    node, _client = node_with(response=SimpleNamespace(
        success=False, message="trajectory target is still changing; hold kept"))
    success, message = WeldGuiNode.release_servo_hold_after_arrival(node, "right")
    assert not success and "still changing" in message
    assert "kept" in node.ui.log.call_args.args[0]
