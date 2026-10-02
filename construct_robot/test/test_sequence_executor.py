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


@pytest.mark.parametrize("kind", [
    "planned_trajectory", "named_pose", "dual_arm_pose", "spray_motion", "head_motion",
])
@pytest.mark.parametrize("execute", [False, True])
def test_motion_dispatch_preserves_runtime_method_and_preview_flag(kind, execute):
    host, executor = make_host()
    handler = Mock(return_value=(True, "runtime result"))
    setattr(host.node, f"run_sequence_{kind}", handler)
    step = {"type": kind}
    assert executor.run_step(step, execute) == (True, "runtime result")
    handler.assert_called_once_with(step, execute)
    host._execute_hicomm_weld.assert_not_called()
    host.hicomm_client.set_command_bit.assert_not_called()


def test_motion_dispatch_does_not_invoke_unlisted_runtime_methods():
    host, executor = make_host()
    host.node.run_sequence_unlisted = Mock()
    assert executor.run_step({"type": "unlisted"}, True) == (False, "unsupported sequence step")
    host.node.run_sequence_unlisted.assert_not_called()


def test_grouping_preserves_first_seen_slot_order_and_separate_sleep():
    host, executor = make_host()
    host._run_sequence_step = Mock(return_value=(True, "preview"))
    steps = [
        {"type": "head_motion", "parallel_slot": 9}, named(3), named(9),
        {"type": "sleep", "parallel_slot": 9, "seconds": 0},
        {"type": "head_motion", "parallel_slot": 3},
    ]
    # Slots are first-appearance ordered, not sorted or split at repeated keys.
    assert executor.execution_preflight_error(steps, work_cycle=False) is None
    run(executor, steps, execute=False)
    assert [call.args[0] for call in host._set_sequence_status.call_args_list] == [
        "Parallel slot 9 · group 1/3 · 2 task(s)",
        "Parallel slot 3 · group 2/3 · 2 task(s)",
        "Parallel slot sleep · group 3/3 · 1 task(s)",
    ]
    host._sequence_finished.assert_called_once_with(True, "complete")


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


def weld_scenario_steps():
    arc_on = {"type": "digital_weld", "command": "on", "settings": {}, "parallel_slot": 1,
              "weld_scenario_id": "weld-1", "weld_scenario_stage": "arc_on"}
    motion = {"type": "motion", "planning_group": "right_manipulator", "parallel_slot": 1,
              "points": ("a", "b"), "weld_scenario_id": "weld-1",
              "weld_scenario_stage": "weld_motion"}
    return [arc_on, motion]


def test_weld_path_is_planned_before_arc_on_and_executed_from_that_plan():
    host, executor = make_host()
    order = []

    def cartesian(step, execute):
        order.append(("cartesian", execute, bool(step.get("reuse_approved_plan"))))
        return True, "planned" if not execute else "executed"

    def arc(kind, settings, conditions=None):
        order.append(("arc", kind))
        host.weld_arc_on_success = True
        host.weld_arc_established_event.set()
        host.weld_arc_on_done_event.set()
        return True, "ARC"

    host.node.run_sequence_cartesian_motion = cartesian
    host._execute_hicomm_weld = arc
    run(executor, weld_scenario_steps())

    assert order[0] == ("cartesian", False, False)   # plan only, before ARC ON
    assert ("arc", "on") in order
    assert order.index(("cartesian", True, True)) > order.index(("arc", "on"))
    host._sequence_finished.assert_called_once_with(True, "complete")


def test_failed_weld_path_planning_never_sends_arc_on():
    host, executor = make_host()
    host.node.run_sequence_cartesian_motion = Mock(return_value=(False, "IK failed"))
    run(executor, weld_scenario_steps())
    host._execute_hicomm_weld.assert_not_called()
    host.node.run_sequence_cartesian_motion.assert_called_once()
    host._sequence_finished.assert_called_once_with(
        False, "weld path planning failed before ARC ON: IK failed"
    )


def test_node_reuses_approved_plan_only_for_execution():
    from construct_robot.nodes.weld_runtime_node import WeldGuiNode

    node = object.__new__(WeldGuiNode)
    node.node_touch_input_states = {"right": False}
    node.ui = SimpleNamespace(post=Mock(), log=Mock(), record_weld_tcp_sample=Mock())
    node.cartesian_motion_client = object()
    goals = []
    node._send_action_goal_and_wait = lambda client, goal, name, **kw: (
        goals.append(goal) or SimpleNamespace(success=True, message="ok")
    )
    pose = Pose(); pose.orientation.w = 1.0
    step = {"planning_group": "right_manipulator", "interpolation_step": 0.001,
            "velocity_scale": 0.2, "points": (pose, pose), "reuse_approved_plan": True}
    WeldGuiNode.run_sequence_cartesian_motion(node, step, False)
    WeldGuiNode.run_sequence_cartesian_motion(node, step, True)
    assert [g.reuse_approved_plan for g in goals] == [False, True]
    assert [g.execute_requested for g in goals] == [False, True]


def test_cleaner_failure_turns_off_only_owned_output():
    from unittest.mock import Mock
    from construct_robot.gui.weld_action_gui import WeldActionGui
    gui = object.__new__(WeldActionGui)
    gui.sequence_stop_requested = False
    gui.fake_arc_enabled = SimpleNamespace(get=lambda: False)
    gui.post = Mock()
    gui.error = Mock()
    gui._set_sequence_status = Mock()
    gui._sequence_finished = Mock()
    gui._run_sequence_step = Mock(return_value=(False, "test failure"))
    gui._set_fastech_output_sync = Mock(return_value=(True, "OFF"))
    gui._finish_weld_feedback_record = Mock()
    gui.hicomm_client = SimpleNamespace(inhibit_outputs=Mock(), clear_outputs=Mock(), latest_status=Mock())
    gui.node = SimpleNamespace(cancel_active_motion=Mock())
    gui._sequence_worker([{"type": "digital_output", "port": 7, "value": True,
                           "task_cleaner_output": True}], [0], True)
    gui._set_fastech_output_sync.assert_called_once_with(7, False)


@pytest.mark.parametrize("waited", [True, False])
def test_wire_feed_pulse_stops_on_completion_or_interrupt(waited):
    host, executor = make_host()
    host.hicomm_client.allow_outputs = Mock()
    host._interruptible_wait = Mock(return_value=waited)
    step = {"type": "wire_feed", "duration": 0.25}
    assert executor.run_step(step, False)[0]
    host.hicomm_client.set_command_bit.assert_not_called()
    assert executor.run_step(step, True)[0] is waited
    host._interruptible_wait.assert_called_once_with(0.25)
    assert host.hicomm_client.set_command_bit.call_args_list == [
        ((executor_module.BIT_REVERSE, False),),
        ((executor_module.BIT_FORWARD, True),),
        ((executor_module.BIT_FORWARD, False),),
    ]


def test_wire_feed_turns_off_when_wait_raises():
    host, executor = make_host()
    host.hicomm_client.allow_outputs = Mock()
    host._interruptible_wait = Mock(side_effect=RuntimeError("wait failed"))
    with pytest.raises(RuntimeError, match="wait failed"):
        executor.run_step({"type": "wire_feed", "duration": 0.25}, True)
    host.hicomm_client.set_command_bit.assert_called_with(executor_module.BIT_FORWARD, False)


def test_wire_feed_preflight_requires_hicomm():
    host, executor = make_host(hicomm_connected=False)
    assert executor.execution_preflight_error([{"type": "wire_feed"}], True) == "Connect Hi-COMM before wire feed"
