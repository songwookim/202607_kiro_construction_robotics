import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from construct_robot.gui import weld_action_gui


def test_bad_queued_callback_does_not_discard_later_or_coalesced_updates():
    gui = object.__new__(weld_action_gui.WeldActionGui)
    gui.root = SimpleNamespace(report_callback_exception=Mock())
    gui._ui_queue = queue.Queue()
    gui._latest_ui_updates_lock = threading.Lock()
    received = []
    gui._latest_ui_updates = {"telemetry": (received.append, ("latest",))}
    gui._ui_queue.put((Mock(side_effect=KeyError("destroyed widget")), ()))
    gui._ui_queue.put((received.append, ("next",)))
    gui._drain_ui_queue()
    assert received == ["next", "latest"]
    gui.root.report_callback_exception.assert_called_once()


def test_ros_ui_bridge_is_rescheduled_even_if_dispatch_fails(monkeypatch):
    gui = object.__new__(weld_action_gui.WeldActionGui)
    gui.root = SimpleNamespace(after=Mock())
    gui._closing = False
    gui._drain_ui_queue = Mock(side_effect=RuntimeError("dispatch failed"))
    monkeypatch.setattr(weld_action_gui.rclpy, "ok", lambda: True)
    with pytest.raises(RuntimeError):
        gui.check_ros()
    gui.root.after.assert_called_once_with(10, gui.check_ros)


def test_stop_sends_cancel_and_controller_stop_before_feedback_save(monkeypatch):
    order = []
    gui = object.__new__(weld_action_gui.WeldActionGui)
    gui.weld_feedback_lock = threading.Lock()
    gui._stop_keyboard_wire = Mock()
    gui.hicomm_client = SimpleNamespace(
        inhibit_outputs=lambda: order.append("arc inhibited"), latest_status=lambda: None)
    gui.node = SimpleNamespace(
        clear_touch_probe=Mock(), cancel_active_motion=lambda: order.append("cancel"),
        stop_sequence_equipment=lambda devices: order.append("controller stop"))
    gui._finish_weld_feedback_record = lambda *a: order.append("save")
    gui.hicomm_gas_enabled = SimpleNamespace(set=Mock())
    gui.hicomm_arc_on_button = SimpleNamespace(configure=Mock())
    gui.hicomm_test_status = SimpleNamespace(configure=Mock())
    gui.sequence_status = SimpleNamespace(configure=Mock())
    gui.robot_connected = {"left": True}
    gui.pipeline_waiting = Mock()
    monkeypatch.setattr(weld_action_gui.threading, "Thread", lambda target, args, **kw:
                        SimpleNamespace(start=lambda: target(*args)))
    gui.stop_sequence()
    assert order == ["arc inhibited", "cancel", "controller stop", "save"]
