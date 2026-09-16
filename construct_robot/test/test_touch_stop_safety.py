from types import SimpleNamespace
import threading
from unittest.mock import Mock

import pytest

from construct_robot.weld_action_gui import WeldActionGui, WeldGuiNode
from construct_robot import weld_action_gui


def node_stub():
    return SimpleNamespace(
        touch_probe_cancel_event=threading.Event(),
        active_motion_goal=None,
        ui=SimpleNamespace(post=Mock(), log=Mock(), touch_probe_failed=Mock()),
        cancel_controller_goals=Mock(return_value=(True, "canceled")),
        wait_until_arm_stopped=Mock(return_value=True),
        wait_for_robot_idle=Mock(return_value=True),
        switch_arm_controller=Mock(return_value=(True, "switched")),
        request_direct_motion_stop=Mock(return_value=(True, "stopped")),
    )


def test_cancel_stop_keeps_controller_active():
    node = node_stub()
    assert WeldGuiNode._stop_motion_on_touch(node, "right", "probe") == (True, False)
    node.switch_arm_controller.assert_not_called()
    node.request_direct_motion_stop.assert_not_called()


def test_deactivation_stop_does_not_repeat_direct_stop():
    node = node_stub()
    node.wait_until_arm_stopped.side_effect = [False, True]
    assert WeldGuiNode._stop_motion_on_touch(node, "right", "probe") == (True, True)
    node.switch_arm_controller.assert_called_once_with("right", False)
    node.request_direct_motion_stop.assert_not_called()


def test_direct_stop_remains_available_if_deactivation_does_not_stop():
    node = node_stub()
    node.wait_until_arm_stopped.side_effect = [False, False, True]
    assert WeldGuiNode._stop_motion_on_touch(node, "right", "probe") == (True, True)
    node.request_direct_motion_stop.assert_called_once_with("right")


@pytest.mark.parametrize("stationary,idle", [(False, True), (True, False)])
def test_restore_requires_standstill_and_idle(stationary, idle):
    node = node_stub()
    node.wait_until_arm_stopped.return_value = stationary
    node.wait_for_robot_idle.return_value = idle
    assert not WeldGuiNode.restore_touch_controller(node, "right")[0]
    node.switch_arm_controller.assert_not_called()


def test_restore_after_confirmed_idle():
    node = node_stub()
    assert WeldGuiNode.restore_touch_controller(node, "right")[0]
    node.switch_arm_controller.assert_called_once_with("right", True)


def test_failed_probe_stop_does_not_reactivate_or_capture():
    node = node_stub()
    node.active_touch_probe = ("right", "wall", "right_manipulator", None, .01, .001)
    node._stop_motion_on_touch = Mock(return_value=(False, True))
    node._current_tcp_pose = Mock()
    WeldGuiNode._stop_touch_probe_and_capture_locked(node)
    assert node.active_touch_probe is None
    assert node.touch_probe_controller_deactivated
    node.switch_arm_controller.assert_not_called()
    node._current_tcp_pose.assert_not_called()


@pytest.mark.parametrize("canceled", [False, True])
def test_retract_uses_fresh_pose_and_honors_stop(monkeypatch, canceled):
    node = node_stub()
    node.ui.touch_probe_return_finished = Mock()
    # Normal capture disarms detection too; None is NOT a cancellation.
    node.active_touch_probe = object()
    node.touch_probe_stop_requested = threading.Event()
    WeldGuiNode.clear_touch_probe(node, cancel_return=canceled)
    node.touch_probe_controller_deactivated = True
    node.restore_touch_controller = Mock(return_value=(True, "restored"))
    current, old_contact, start = object(), object(), object()
    node._current_tcp_pose = Mock(return_value=current)
    node.run_sequence_cartesian_motion = Mock(return_value=(True, "returned"))
    waypoints = Mock(return_value=[])
    monkeypatch.setattr(weld_action_gui, "linear_pose_waypoints", waypoints)
    WeldGuiNode.return_touch_probe(
        node, "right_manipulator", old_contact, start, .01, .001, "wall", 0,
    )
    if canceled:
        node.restore_touch_controller.assert_not_called()
        node.run_sequence_cartesian_motion.assert_not_called()
    else:
        waypoints.assert_called_once_with(current, start, 2)
        node.run_sequence_cartesian_motion.assert_called_once()


def test_stop_after_normal_capture_cancels_pending_return():
    node = node_stub()
    node.touch_probe_stop_requested = threading.Event()
    pending_return = node.touch_probe_cancel_event
    WeldGuiNode.clear_touch_probe(node, cancel_return=False)
    assert not pending_return.is_set()
    WeldGuiNode.clear_touch_probe(node)
    assert pending_return.is_set()


def test_stop_during_dwell_prevents_controller_restore(monkeypatch):
    node = node_stub()
    node.ui.touch_probe_return_finished = Mock()
    node.restore_touch_controller = Mock()
    node.run_sequence_cartesian_motion = Mock()
    node.touch_probe_stop_requested = threading.Event()
    node.touch_probe_controller_deactivated = True
    monkeypatch.setattr(weld_action_gui.time, "sleep", lambda _: WeldGuiNode.clear_touch_probe(node))
    WeldGuiNode.return_touch_probe(node, "right_manipulator", None, None, .01, .001, "wall", .7)
    node.restore_touch_controller.assert_not_called()
    node.run_sequence_cartesian_motion.assert_not_called()


def test_operator_motion_stop_does_not_change_touch_enable_output():
    node = node_stub()
    node.ui._set_fastech_output_sync = Mock()
    node.ui.sequence_hard_stop_finished = Mock()
    WeldGuiNode.stop_sequence_equipment(node, ())
    node.ui._set_fastech_output_sync.assert_not_called()


def test_delete_all_sequence_steps_resets_builder(monkeypatch):
    gui = SimpleNamespace(
        sequence_running=False,
        sequence_steps=[{"type": "sleep"}, {"type": "sleep"}],
        sequence_parallel_slot=SimpleNamespace(set=Mock()),
        sequence_status=SimpleNamespace(configure=Mock()),
        refresh_sequence_table=Mock(),
        log=Mock(),
        error=Mock(),
    )
    monkeypatch.setattr(weld_action_gui.messagebox, "askyesno", lambda *_args: True)
    WeldActionGui.delete_all_sequence_steps(gui)
    assert gui.sequence_steps == []
    gui.sequence_parallel_slot.set.assert_called_once_with(1)
    gui.refresh_sequence_table.assert_called_once_with()
