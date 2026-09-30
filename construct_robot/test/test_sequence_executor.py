"""Sequence execution without Tk (application.sequence_executor)."""

import ast
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import Pose
import pytest

from construct_robot.application import sequence_executor as executor_module
from construct_robot.application.sequence_executor import (
    SequenceExecutor,
    record_step_conditions,
)
from construct_robot.core.sequence_model import SequenceModel


def make_host(fake_arc=False, **overrides):
    """A plain host double: shared state plus mocked runtime operations."""
    host = SimpleNamespace(
        sequence_model=SequenceModel(),
        sequence_stop_requested=False,
        weld_motion_done_event=threading.Event(),
        weld_motion_success=False,
        weld_arc_established_event=threading.Event(),
        weld_arc_on_done_event=threading.Event(),
        weld_arc_on_success=False,
        weld_feedback_lock=threading.Lock(),
        _weld_feedback_stopped=True,
        fake_arc_enabled=SimpleNamespace(get=lambda: fake_arc),
        fastech_connected=True,
        robot_connected={"left": True, "right": True},
        execution_allowed=True,
        hicomm_connected=True,
        hicomm_client=SimpleNamespace(
            inhibit_outputs=Mock(), clear_outputs=Mock(),
            latest_status=Mock(return_value="LATEST"),
            set_command_bit=Mock(), connected=True,
        ),
        node=SimpleNamespace(
            cancel_active_motion=Mock(),
            run_sequence_named_pose=Mock(return_value=(True, "named ok")),
            run_sequence_cartesian_motion=Mock(return_value=(True, "motion ok")),
            run_sequence_head_motion=Mock(return_value=(True, "head ok")),
            _set_legacy_digital_output_sync=Mock(return_value=(True, "legacy ok")),
        ),
        post=lambda callback, *args: callback(*args),
        log=Mock(),
        error=Mock(),
        _set_sequence_status=Mock(),
        _sequence_finished=Mock(),
        _begin_weld_feedback_record=Mock(),
        _finish_weld_feedback_record=Mock(),
        _record_actual_tcp_until_motion_done=Mock(),
        _pending_weld_final_status=Mock(return_value="FINAL"),
        _mark_weld_motion_timing=Mock(),
        _execute_hicomm_weld=Mock(return_value=(True, "ARC")),
        _execute_triggered_arc_off=Mock(return_value=(True, "ARC OFF")),
        _execute_software_crater=Mock(return_value=(True, "crater")),
        _software_crater_restore=Mock(),
        _execute_custom_hot_start=Mock(return_value=(True, "hot start")),
        _set_fastech_output_sync=Mock(return_value=(True, "fastech ok")),
    )
    for name, value in overrides.items():
        setattr(host, name, value)
    executor = SequenceExecutor(host, touch_io_backend="fastech_ethernet")
    # The executor calls its sibling entry points through the host.
    host._sequence_worker = executor.run_worker
    host._sequence_worker_body = executor.run_groups
    host._run_sequence_step = executor.run_step
    host._interruptible_wait = executor.interruptible_wait
    return host, executor


def run(executor, steps, execute=True):
    indices = list(range(len(steps)))
    executor.host.sequence_model.start(indices, execute)
    executor.run_worker(steps, indices, execute)


def named(slot):
    return {"type": "named_pose", "planning_group": "right_manipulator", "parallel_slot": slot}


def test_executor_imports_no_tkinter_or_gui():
    tree = ast.parse(Path(executor_module.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    for name in imported:
        assert not (name or "").startswith(
            ("tkinter", "construct_robot.gui", "construct_robot.nodes")
        ), name


def test_same_slot_members_run_in_parallel_and_groups_run_in_order():
    host, executor = make_host()
    barrier = threading.Barrier(2, timeout=2.0)
    order = []

    def head(step, execute):
        barrier.wait()  # would time out if the slot-1 members ran serially
        order.append(("head", step["parallel_slot"]))
        return True, "head ok"

    def named_pose(step, execute):
        if step["parallel_slot"] == 1:
            barrier.wait()
        order.append(("named", step["parallel_slot"]))
        return True, "named ok"

    host.node.run_sequence_head_motion = head
    host.node.run_sequence_named_pose = named_pose
    run(executor, [named(1), {"type": "head_motion", "parallel_slot": 1}, named(2)])

    assert order[-1] == ("named", 2)
    assert sorted(order[:2]) == [("head", 1), ("named", 1)]
    host._sequence_finished.assert_called_once_with(True, "complete")
    assert [c.args[0] for c in host._set_sequence_status.call_args_list] == [
        "Parallel slot 1 · group 1/2 · 2 task(s)",
        "Parallel slot 2 · group 2/2 · 1 task(s)",
    ]


def test_failure_inhibits_cancels_and_skips_later_groups():
    host, executor = make_host()
    host.node.run_sequence_named_pose = Mock(side_effect=[(False, "blocked"), (True, "ok")])
    run(executor, [named(1), named(2)])

    assert host.node.run_sequence_named_pose.call_count == 1
    host.hicomm_client.inhibit_outputs.assert_called_once()
    host.node.cancel_active_motion.assert_called_once()
    host.hicomm_client.clear_outputs.assert_called_once()
    host._finish_weld_feedback_record.assert_any_call("failed: blocked", "LATEST")
    host._sequence_finished.assert_called_once_with(False, "blocked")
    host._set_fastech_output_sync.assert_not_called()  # touch-enable DO0 untouched


def test_fake_arc_failure_leaves_hicomm_outputs_alone():
    host, executor = make_host(fake_arc=True)
    host.node.run_sequence_named_pose = Mock(return_value=(False, "blocked"))
    run(executor, [named(1)])
    host.hicomm_client.inhibit_outputs.assert_not_called()
    host.hicomm_client.clear_outputs.assert_not_called()
    host.node.cancel_active_motion.assert_called_once()


def test_operator_stop_prevents_the_next_group():
    host, executor = make_host()

    def stop_after(step, execute):
        host.sequence_stop_requested = True
        return True, "named ok"

    host.node.run_sequence_named_pose = stop_after
    run(executor, [named(1), {"type": "head_motion", "parallel_slot": 2}])
    host.node.run_sequence_head_motion.assert_not_called()
    host._sequence_finished.assert_called_once_with(False, "stopped by operator")
    host._finish_weld_feedback_record.assert_any_call("stopped", "LATEST")


def test_weld_motion_waits_for_arc_and_aborts_when_arc_on_fails():
    host, executor = make_host()
    host.weld_arc_on_done_event.set()
    host.weld_arc_on_success = False
    step = {"type": "motion", "weld_scenario_stage": "weld_motion"}
    assert executor.run_step(step, True) == (False, "weld motion aborted: ARC ON failed")
    host.node.run_sequence_cartesian_motion.assert_not_called()

    host.weld_arc_established_event.set()
    assert executor.run_step(step, True) == (True, "motion ok")
    assert [c.args[0] for c in host._mark_weld_motion_timing.call_args_list] == [
        "start", "complete",
    ]


def test_timed_outputs_turn_off_and_route_by_backend():
    host, executor = make_host()
    fastech = {"type": "digital_output", "io_backend": "fastech_ethernet",
               "port": 6, "value": True, "duration": 0.01}
    assert executor.run_step(fastech, True)[0]
    assert [c.args for c in host._set_fastech_output_sync.call_args_list] == [(6, True), (6, False)]
    legacy = {"type": "digital_output", "port": 2, "value": True}
    assert executor.run_step(legacy, True) == (True, "Legacy Rainbow DO2 ON confirmed")
    host.node._set_legacy_digital_output_sync.assert_called_once_with(2, True)
    assert executor.run_step(fastech, False) == (
        True, "Equipment output command planned (no output sent)"
    )


def test_cleaner_outputs_are_forced_off_after_execution():
    host, executor = make_host()
    steps = [{"type": "digital_output", "io_backend": "fastech_ethernet", "port": 7,
              "value": True, "task_cleaner_output": True, "parallel_slot": 1},
             named(2)]
    run(executor, steps)
    assert host._set_fastech_output_sync.call_args_list[-1].args == (7, False)


def test_worker_exception_releases_run_and_reports_internal_error():
    host, executor = make_host()
    host._begin_weld_feedback_record = Mock(side_effect=RuntimeError("boom"))
    steps = [{"type": "digital_weld", "command": "on", "settings": {}, "parallel_slot": 1}]
    run(executor, steps)
    assert host.sequence_stop_requested is True
    assert host.weld_motion_done_event.is_set()
    host.hicomm_client.inhibit_outputs.assert_called_once()
    host._sequence_finished.assert_called_once_with(False, "internal sequence error: boom")


@pytest.mark.parametrize("changes,steps,message", [
    (dict(fastech_connected=False),
     [{"type": "digital_output", "io_backend": "fastech_ethernet", "port": 0, "value": False}],
     "Connect Fastech Ethernet"),
    (dict(robot_connected={"right": False}), [named(1)], "disconnected: right"),
    (dict(execution_allowed=False), [named(1)], "execution disabled"),
    (dict(hicomm_connected=False), [{"type": "gas", "enabled": False}], "Connect Hi-COMM"),
    ({}, [named(1), {"type": "motion", "planning_group": "right_manipulator", "parallel_slot": 1}],
     "Only one robot motion is allowed in each parallel slot: 1"),
])
def test_execution_preflight_errors(changes, steps, message):
    host, executor = make_host(**changes)
    assert message in executor.execution_preflight_error(steps, work_cycle=False)


def test_preflight_passes_and_begin_starts_worker(monkeypatch):
    host, executor = make_host()
    steps = [named(1)]
    assert executor.execution_preflight_error(steps, work_cycle=False) is None
    started = []

    class InlineThread:
        def __init__(self, target, args, daemon):
            started.append((target, args, daemon))

        def start(self):
            pass

    monkeypatch.setattr(executor_module.threading, "Thread", InlineThread)
    host.sequence_stop_requested = True
    executor.begin(steps, [0], True, fake_arc_snapshot=True)
    assert host._sequence_fake_arc_snapshot is True
    assert host._weld_feedback_stopped is False
    assert host.sequence_stop_requested is False
    assert host.sequence_model.running
    host._set_sequence_status.assert_called_once_with("EXECUTE running · 1 step(s)")
    assert started == [(host._sequence_worker, (steps, [0], True), True)]
    # While running, the Execute-time FAKE ARC snapshot wins over the live toggle;
    # the GUI exposes sequence_running as a property over its SequenceModel.
    host.sequence_running = host.sequence_model.running
    assert executor.fake_arc() is True
    host.sequence_running = False
    assert executor.fake_arc() is False


def test_step_conditions_record_motion_waypoints():
    pose = Pose()
    pose.position.x, pose.orientation.w = 0.5, 1.0
    recorded = record_step_conditions(
        [{"type": "motion", "points": (pose,), "parallel_slot": 3, "extra": 1}], [4],
    )
    assert recorded == [{
        "sequence_number": 5, "type": "motion", "parallel_slot": 3,
        "waypoint_count": 1,
        "waypoints": [{
            "position_m": {"x": 0.5, "y": 0.0, "z": 0.0},
            "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }],
    }]
