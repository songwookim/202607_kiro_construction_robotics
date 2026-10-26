"""Start touch is mandatory before ARC, bounded, and modifies only this pass's execution."""

import copy
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from construct_robot.application.scan_start_touch import seek_start_touch, register_touch_height
from construct_robot.application.scan_weld_cycle import expand_scan_weld_cycle
from construct_robot.core.sequence_model import validate_managed_weld_sequence
from test_scan_weld_cycle import cycle
from test_sequence_executor import make_host


def setup_probe(number=1):
    rows = expand_scan_weld_cycle(cycle(rescan_after_passes=False))
    steps = [row for row in rows if row.get("scan_cycle_phase") == "weld"]
    contact = next(s for s in steps if s.get("scan_cycle_pass") == number and s.get("weld_scenario_stage") == "start_contact")
    nominal = copy.deepcopy(contact["start_touch_nominal"])
    host, executor = make_host(fake_arc=True)
    node = host.node
    node.node_touch_input_states = {"right": False}
    node.touch_guard_triggered = threading.Event()
    node.touch_guard_stop_complete = threading.Event()
    node.touch_guard_stop_success = False
    node.touch_guard_edge_pose = None
    current = copy.deepcopy(nominal)
    current.position.z += .006
    node._current_tcp_pose = Mock(side_effect=lambda _group: copy.deepcopy(current))
    host._interruptible_wait = Mock(return_value=True)
    host.hicomm_client.latest_status = Mock(side_effect=lambda: dict(
        arc_ack=False, wcr_detected=False, timestamp_monotonic=time.monotonic(), feedback_current_a=0, wire_feed_m_min=0))
    node.probe_outcome = "touch"

    def move(step, execute):
        if execute and step.get("path_kind") == "start_touch_slow_world_z_descent":
            if node.probe_outcome == "stop":
                host.sequence_stop_requested = True
                return False, "STOP"
            if node.probe_outcome != "missing":
                edge = copy.deepcopy(nominal)
                edge.position.z -= .002
                if node.probe_outcome == "lateral":
                    edge.position.x += .002
                node.touch_guard_edge_pose = edge
                node.touch_guard_triggered.set()
                node.touch_guard_stop_complete.set()
                node.touch_guard_stop_success = node.probe_outcome != "moving"
        return True, "planned / moved"
    node.run_sequence_cartesian_motion = Mock(side_effect=move)
    return host, executor, steps, contact


def test_plan_shows_vertical_search_without_outputs_or_registration():
    host, _executor, _steps, contact = setup_probe()
    assert seek_start_touch(host, contact, False)[0]
    host._set_fastech_output_sync.assert_not_called()
    host._interruptible_wait.assert_not_called()
    points = host.node.run_sequence_cartesian_motion.call_args.args[0]["points"]
    nominal = contact["start_touch_nominal"]
    assert points[-1].position.z == pytest.approx(nominal.position.z - .010)
    assert "start_touch_result" not in contact


def test_touch_uses_the_executed_weld_attitude_not_an_older_source_attitude():
    from construct_robot.application.scan_start_touch import configure_start_touch, START_TOUCH
    _host, _executor, steps, contact = setup_probe()
    motion = next(s for s in steps if s.get("scan_cycle_pass") == 1 and s.get("weld_scenario_stage") == "weld_motion")
    motion["points"][0].orientation.x = .1
    motion["points"][0].orientation.w = .99498743710662
    configure_start_touch([contact, motion], START_TOUCH)
    assert contact["start_touch_nominal"].orientation == motion["points"][0].orientation


@pytest.mark.parametrize("field,value", [("touch_guard", False), ("accept_initial_touch", True),
                                        ("allow_initial_touch_motion", True), ("continue_after_touch", False)])
def test_start_touch_guard_cannot_be_disabled_in_a_generated_cycle(field, value):
    _host, _executor, steps, contact = setup_probe()
    contact[field] = value
    with pytest.raises(ValueError):
        validate_managed_weld_sequence(steps, require_complete=True)


def test_fresh_touch_is_slow_vertical_and_disables_sensor_before_success():
    host, _executor, _steps, contact = setup_probe()
    assert seek_start_touch(host, contact, True)[0]
    assert [c.args for c in host._set_fastech_output_sync.call_args_list] == [(0, True), (0, False)]
    descent = host.node.run_sequence_cartesian_motion.call_args.args[0]
    assert descent["tcp_speed_m_s"] == .0005 and descent["touch_guard"]
    assert descent["accept_initial_touch"] is False
    assert descent["allow_initial_touch_motion"] is False
    assert descent["points"][0].position.x == contact["start_touch_nominal"].position.x
    assert contact["start_touch_result"]["delta_world_m"] == pytest.approx([0, 0, -.002])


@pytest.mark.parametrize("outcome", ["missing", "moving", "lateral", "stop"])
def test_no_contact_or_unsafe_stop_never_produces_registration(outcome):
    host, _executor, _steps, contact = setup_probe()
    host.node.probe_outcome = outcome
    assert seek_start_touch(host, contact, True)[0] is False
    assert "start_touch_result" not in contact
    assert host._set_fastech_output_sync.call_args.args == (0, False)
    host._execute_hicomm_weld.assert_not_called()


@pytest.mark.parametrize("state", [None, True])
def test_unknown_or_initially_high_input_cannot_skip_contact(state):
    host, _executor, _steps, contact = setup_probe()
    host.node.node_touch_input_states["right"] = state
    assert seek_start_touch(host, contact, True)[0] is False
    host.node.run_sequence_cartesian_motion.assert_not_called()
    assert host._set_fastech_output_sync.call_args.args == (0, False)


@pytest.mark.parametrize("status", [dict(arc_ack=True, wcr_detected=False),
                                  dict(arc_ack=False, wcr_detected=False, timestamp_monotonic=0)])
def test_active_or_stale_welder_feedback_blocks_touch_enable(status):
    host, _executor, _steps, contact = setup_probe()
    host.hicomm_client.latest_status.side_effect = None
    host.hicomm_client.latest_status.return_value = status
    assert seek_start_touch(host, contact, True)[0] is False
    host._set_fastech_output_sync.assert_not_called()
    host.node.run_sequence_cartesian_motion.assert_not_called()


def test_touch_off_failure_blocks_arc_even_after_a_good_contact():
    host, _executor, _steps, contact = setup_probe()
    host._set_fastech_output_sync.side_effect = [(True, "enabled"), (False, "readback mismatch")]
    ok, text = seek_start_touch(host, contact, True)
    assert not ok and "OFF failed" in text
    assert "start_touch_result" not in contact


def test_lost_input_cancels_motion_and_blocks_arc():
    host, _executor, _steps, contact = setup_probe()
    def lost_input(_step, _execute):
        host.node.node_touch_input_states["right"] = None
        assert host.node.cancel_active_motion.wait_called.wait(timeout=1)
        return True, "motion canceled"
    called = threading.Event()
    host.node.cancel_active_motion.side_effect = called.set
    host.node.cancel_active_motion.wait_called = called
    host.node.run_sequence_cartesian_motion.side_effect = lost_input
    ok, text = seek_start_touch(host, contact, True)
    assert not ok and "connection lost" in text
    assert host.node.cancel_active_motion.called
    assert host._set_fastech_output_sync.call_args.args == (0, False)
    assert "start_touch_result" not in contact


@pytest.mark.parametrize("number", [1, 4])
def test_registration_moves_this_pass_geometry_and_crater_once(number):
    host, _executor, steps, contact = setup_probe(number)
    before = copy.deepcopy(steps)
    assert seek_start_touch(host, contact, True)[0]
    register_touch_height(steps, contact)
    for old, new in zip(before, steps):
        if old.get("scan_cycle_pass") != number or old.get("weld_scenario_stage") in ("weld_wait", "start_wait", "start_safe", "start_contact"):
            for a, b in zip(old.get("points", ()), new.get("points", ())):
                assert a == b
            continue
        for key in ("usable_seam_start", "usable_seam_goal", "endpoint", "expected_start_tcp", "tcp_pose"):
            if old.get(key) is not None:
                assert new[key].position.z - old[key].position.z == pytest.approx(-.002)
                assert new[key].position.x == old[key].position.x
                assert new[key].orientation == old[key].orientation
        for a, b in zip(old.get("points", ()), new.get("points", ())):
            assert b.position.z - a.position.z == pytest.approx(-.002)
        assert new.get("waypoint_hold_s") == old.get("waypoint_hold_s")
        assert new.get("weld_weave_transverse_vector") == old.get("weld_weave_transverse_vector")
    with pytest.raises(ValueError, match="already applied"):
        register_touch_height(steps, contact)


@pytest.mark.parametrize("outcome", ["touch", "missing", "moving", "stop"])
def test_executor_registers_and_replans_before_arc_or_blocks_it(outcome, tmp_path, monkeypatch):
    from construct_robot.application.sequence_executor import record_step_conditions
    from construct_robot.io.scan_cycle_log import ScanCycleLog
    host, executor, all_steps, contact = setup_probe()
    steps = [s for s in all_steps if s.get("scan_cycle_pass") == 1]
    monkeypatch.setenv("KIRO_SCAN_CYCLE_LOG_ROOT", str(tmp_path / "cycles"))
    journal = ScanCycleLog(tmp_path / "cycles", record_step_conditions(steps, range(len(steps))), {}, True)
    host._scan_cycle_journal = journal
    host.node.probe_outcome = outcome
    motion = next(s for s in steps if s.get("weld_scenario_stage") == "weld_motion")
    original_z = motion["usable_seam_start"].position.z
    checks = []
    original_move = host.node.run_sequence_cartesian_motion.side_effect
    def move(step, execute):
        if step.get("weld_scenario_stage") == "weld_motion":
            assert step["usable_seam_start"].position.z == pytest.approx(original_z - .002)
            checks.append("execute" if execute else "preplan")
        return original_move(step, execute)
    host.node.run_sequence_cartesian_motion.side_effect = move
    def arc(kind, _settings, conditions=None):
        if kind == "on":
            assert checks == ["preplan"]
            assert conditions["weld_start_touch"]["delta_world_m"][2] == pytest.approx(-.002)
            checks.append("arc")
            host.weld_arc_on_success = True
            host.weld_arc_established_event.set()
            host.weld_arc_on_done_event.set()
        return True, "mock ARC"
    host._execute_hicomm_weld.side_effect = arc
    executor.run_groups(steps, list(range(len(steps))), True)
    if outcome == "touch":
        assert checks == ["preplan", "arc", "execute"]
        assert host._sequence_finished.call_args.args[0] is True
        summary = json.loads((journal.directory / "summary.json").read_text())
        artifact = next(a for a in summary["artifacts"] if a["kind"] == "start_touch_registration")
        assert artifact["pass_number"] == 1
        assert artifact["delta_world_m"] == pytest.approx([0, 0, -.002])
    else:
        assert checks == []
        host._execute_hicomm_weld.assert_not_called()
        assert host._sequence_finished.call_args.args[0] is False


def test_failed_registration_logging_blocks_arc(monkeypatch):
    from construct_robot.application import sequence_executor
    host, executor, all_steps, _contact = setup_probe()
    steps = [s for s in all_steps if s.get("scan_cycle_pass") == 1]
    def record(_host, _event, **data):
        if data.get("kind") == "start_touch_registration":
            host._scan_cycle_log_error = "disk write failed"
            return False
        return True
    monkeypatch.setattr(sequence_executor, "record_cycle_event", record)
    executor.run_groups(steps, list(range(len(steps))), True)
    host._execute_hicomm_weld.assert_not_called()
    assert host._sequence_finished.call_args.args[0] is False


def test_touch_goes_all_the_way_to_the_reference_tcp_then_descends_slowly_from_it():
    host, _executor, _steps, contact = setup_probe()
    calls, move = [], host.node.run_sequence_cartesian_motion.side_effect

    def record(step, execute):                       # the probe dict is updated in place between legs
        calls.append(copy.deepcopy(step))
        return move(step, execute)
    host.node.run_sequence_cartesian_motion = Mock(side_effect=record)
    assert seek_start_touch(host, contact, True)[0]
    nominal = contact["start_touch_nominal"]
    assert [c["path_kind"] for c in calls] == ["start_touch_move_to_reference_tcp",
                                                "start_touch_slow_world_z_descent"]
    to_ref, slow = calls
    assert to_ref["points"] == (nominal,) and to_ref["continue_after_touch"] is False
    assert to_ref["tcp_speed_m_s"] == contact["start_touch_approach_speed_m_s"]   # the taught approach speed
    assert slow["points"][0].position.z == pytest.approx(nominal.position.z - .010)
    assert slow["points"][0].position.x == pytest.approx(nominal.position.x)
    assert slow["tcp_speed_m_s"] == pytest.approx(.0005)
    assert contact["start_touch_result"]["delta_world_m"][2] == pytest.approx(-.002)


def test_a_contact_before_the_reference_tcp_blocks_the_pass():
    host, _executor, _steps, contact = setup_probe()
    nominal = contact["start_touch_nominal"]
    node = host.node
    kinds = []

    def move(step, execute):
        kinds.append(step["path_kind"])
        if execute and step["path_kind"] == "start_touch_move_to_reference_tcp":
            edge = copy.deepcopy(nominal)
            edge.position.z += .001                                            # surface above START
            node.touch_guard_edge_pose = edge
            node.touch_guard_triggered.set()
            return False, "touch guard stop"
        return True, "moved"
    node.run_sequence_cartesian_motion = Mock(side_effect=move)
    ok, message = seek_start_touch(host, contact, True)
    assert not ok and "before the reference TCP" in message
    assert "start_touch_slow_world_z_descent" not in kinds and "start_touch_result" not in contact
    host._set_fastech_output_sync.assert_called_with(0, False)                 # sensor off afterwards


def test_plan_preview_goes_to_the_reference_tcp_then_down():
    host, _executor, _steps, contact = setup_probe()
    assert seek_start_touch(host, contact, False)[0]
    points = host.node.run_sequence_cartesian_motion.call_args.args[0]["points"]
    assert points[0] == contact["start_touch_nominal"] and len(points) == 2
