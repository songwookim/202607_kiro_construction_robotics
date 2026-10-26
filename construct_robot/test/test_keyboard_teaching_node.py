"""X11 arrow poller: edges at once, unchanged state only as a 10 Hz heartbeat."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from construct_robot.nodes import keyboard_teaching_node as module


def poller(masks, period=0.001):
    node = object.__new__(module.KeyboardTeachingNode)
    sequence = iter(masks)
    node._stop = threading.Event()

    def read_mask():
        try:
            return next(sequence)
        except StopIteration:
            node._stop.set()
            return masks[-1]  # unchanged: the run just ends

    node._keyboard = SimpleNamespace(read_mask=read_mask)
    node._publisher = Mock()
    node._period = period
    return node


def published(node):
    return [call.args[0].data for call in node._publisher.publish.call_args_list]


def test_every_edge_is_published_and_repeats_are_not(monkeypatch):
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    node = poller([0, 0, 4, 4, 4, 0, 0, 1, 5, 4])
    node._poll_loop()
    assert published(node) == [0, 4, 0, 1, 5, 4]


def test_unchanged_state_heartbeat_at_ten_hz(monkeypatch):
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    node = poller([0] * 60, period=0.005)
    node._stop.wait = lambda delay: clock.__setitem__(0, clock[0] + delay)
    node._poll_loop()
    assert len(published(node)) in (3, 4)  # 0.3 s of polls at 5 ms


def test_axis_keys_are_published_once_per_press(monkeypatch):
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    node = poller([0])
    states = iter([(0, set()), (0, {ord("1")}), (0, {ord("1")}), (0, set()), (0, {ord("1"), ord("3")})])

    def read_state():
        try:
            return next(states)
        except StopIteration:
            node._stop.set()
            return 0, {ord("1"), ord("3")}

    node._keyboard = SimpleNamespace(read_state=read_state)
    node._selection_publisher = Mock()
    node._poll_loop()
    codes = [call.args[0].data for call in node._selection_publisher.publish.call_args_list]
    assert codes == [ord("1"), ord("1"), ord("3")]


def test_v_held_and_quickly_repressed_produces_exactly_two_events(monkeypatch):
    assert module.SELECTION_KEYSYMS[ord("v")] == ord("v")
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    node = poller([0])
    states = iter([set(), {ord("v")}, {ord("v")}, set(), {ord("v")}])

    def read_state():
        try:
            return 0, next(states)
        except StopIteration:
            node._stop.set()
            return 0, {ord("v")}

    node._keyboard = SimpleNamespace(read_state=read_state)
    node._selection_publisher = Mock()
    node._poll_loop()
    assert [call.args[0].data for call in node._selection_publisher.publish.call_args_list] == [ord("v")] * 2
