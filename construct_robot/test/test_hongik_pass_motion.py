"""Original delivered TCP file motion; scripted GUI and no physical robot."""

import copy
import math
import tkinter as tk
from types import SimpleNamespace as NS
from unittest.mock import Mock

from geometry_msgs.msg import Pose
import pytest
import yaml

from construct_robot.application.direct_scan import load_pass_file, pass_motion_steps
from construct_robot.gui import precision_sensing_test_panel as panel_module


DOC = {"schema": "xy_endpoint_review_v1", "pass": 1, "frame_id": "World", "requires_ik": True,
       "current_start": {"position_m": dict(x=.0146, y=-1.5476, z=.5874),
                         "orientation_xyzw": dict(x=0., y=math.sqrt(.5), z=0., w=math.sqrt(.5))},
       "current_goal": {"position_m": dict(x=.0037, y=-1.7467, z=.5847),
                        "orientation_xyzw": dict(x=0., y=1., z=0., w=0.)}}


def write(tmp_path, doc=DOC):
    path = tmp_path / "pass_1.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def values(pose):
    return tuple(getattr(pose.position, a) for a in "xyz"), tuple(getattr(pose.orientation, a) for a in "xyzw")


def test_original_positions_and_each_orientation_are_used_without_wire_offset(tmp_path):
    loaded = load_pass_file(write(tmp_path))
    current = Pose()
    current.orientation.w = 1.
    steps = pass_motion_steps(loaded, current, "START → GOAL", velocity_scale=.3, tcp_speed_m_s=.005)
    assert len(steps) == 2
    assert [row["parallel_slot"] for row in steps] == [1, 2]
    for target, row in ((loaded.start, steps[0]), (loaded.goal, steps[1])):
        xyz, q = values(row["points"][-1])
        assert xyz == pytest.approx(target.position)
        assert q == pytest.approx(target.quaternion)
        assert row["type"] == "motion" and row["planning_group"] == "right_manipulator"
        assert row["touch_guard"] and not row["continue_after_touch"]
        assert not row["allow_initial_touch_motion"]
    assert values(steps[1]["points"][0])[0] == pytest.approx(loaded.start.position)
    assert steps[1]["tcp_speed_m_s"] == .005 and steps[1]["linear_motion_profile"]


@pytest.mark.parametrize("mode", ["START", "GOAL"])
def test_single_endpoint_modes_do_not_traverse_the_pass(tmp_path, mode):
    loaded = load_pass_file(write(tmp_path))
    current = Pose(); current.orientation.w = 1.
    steps = pass_motion_steps(loaded, current, mode, velocity_scale=.3, tcp_speed_m_s=.005)
    assert len(steps) == 1
    target = loaded.start if mode == "START" else loaded.goal
    assert values(steps[0]["points"][-1])[0] == pytest.approx(target.position)


@pytest.mark.parametrize("change", ["frame", "schema", "nan", "quaternion"])
def test_invalid_or_nonworld_file_is_refused(tmp_path, change):
    doc = copy.deepcopy(DOC)
    if change == "frame": doc["frame_id"] = "robot_base_link"
    elif change == "schema": doc["schema"] = "other"
    elif change == "nan": doc["current_goal"]["position_m"]["z"] = float("nan")
    else: doc["current_goal"]["orientation_xyzw"] = dict.fromkeys("xyzw", 0.)
    with pytest.raises(ValueError): load_pass_file(write(tmp_path, doc))


def test_file_change_requires_reload_and_original_is_not_written(tmp_path):
    path = write(tmp_path)
    before = path.read_bytes()
    loaded = load_pass_file(path)
    loaded.require_unchanged()
    assert path.read_bytes() == before
    path.write_bytes(before + b"\n# edited\n")
    with pytest.raises(ValueError, match="changed after loading"): loaded.require_unchanged()


@pytest.mark.parametrize("speed", [float("nan"), 0., -.01])
def test_invalid_speed_is_refused(tmp_path, speed):
    loaded = load_pass_file(write(tmp_path))
    with pytest.raises(ValueError, match="speed"):
        pass_motion_steps(loaded, Pose(), "START → GOAL", velocity_scale=.3, tcp_speed_m_s=speed)


def panel(monkeypatch, tmp_path, *, server_running=False):
    p = panel_module.PrecisionSensingTestPanel.__new__(panel_module.PrecisionSensingTestPanel)
    interp = tk.Tcl()
    p.pass_file = tk.StringVar(interp, str(write(tmp_path)))
    p.kiro_pass = load_pass_file(p.pass_file.get())
    p.pass_mode = tk.StringVar(interp, "START → GOAL")
    p.pass_speed = tk.StringVar(interp, "5")
    p.pass_status = tk.StringVar(interp)
    p.status_service = tk.StringVar(interp, "/status")
    p.busy, p.capturing = False, False
    p._generation = 0
    current = Pose(); current.orientation.w = 1.
    node = NS(_path_start_tcp_pose=Mock(return_value=current),
              precision_sensing_test=NS(server_status=Mock(return_value=(True, "status", {
                  "running": "True" if server_running else "False", "awaiting_confirm": "False"}))))
    p.host = NS(node=node, velocity_percent=tk.StringVar(interp, "30"),
                post=lambda cb, *args: cb(*args), run_sequence=Mock(), stop_sequence=Mock(), log=Mock(), error=Mock(),
                sequence_execution_preflight_error=Mock(return_value=None))
    p._motion_blocker = Mock(return_value=None)
    monkeypatch.setattr(panel_module.threading, "Thread", lambda target, **kw: NS(start=target))
    return p


def test_plan_and_execute_use_the_existing_motion_only_sequence(monkeypatch, tmp_path):
    p = panel(monkeypatch, tmp_path)
    p.run_pass_motion(False)
    assert p.host.run_sequence.call_args.args == (True, False)
    plan_rows = p.host.run_sequence.call_args.kwargs["steps_override"]
    p.run_pass_motion(True)
    assert p.host.run_sequence.call_args.args == (True, True)
    execute_rows = p.host.run_sequence.call_args.kwargs["steps_override"]
    assert [values(row["points"][-1]) for row in plan_rows] == [values(row["points"][-1]) for row in execute_rows]
    assert all(row["type"] == "motion" for row in execute_rows)
    assert not p.capturing


def test_running_hongik_server_blocks_pass_motion(monkeypatch, tmp_path):
    p = panel(monkeypatch, tmp_path, server_running=True)
    p.run_pass_motion(True)
    p.host.run_sequence.assert_not_called()
    p.host.node._path_start_tcp_pose.assert_not_called()
    assert "Wait for Hongik motion" in p.pass_status.get() and not p.capturing


def test_motion_started_during_preparation_blocks_late_execution(monkeypatch, tmp_path):
    p = panel(monkeypatch, tmp_path)
    p._motion_blocker.side_effect = [None, "Another motion started"]
    p.run_pass_motion(True)
    p.host.run_sequence.assert_not_called()
    assert "Another motion started" in p.pass_status.get()


def test_stop_during_preparation_never_starts_a_late_sequence(monkeypatch, tmp_path):
    p = panel(monkeypatch, tmp_path)
    callbacks = []
    p.host.post = lambda cb, *args: callbacks.append((cb, args))
    p.run_pass_motion(True)
    p.stop_pass_motion()
    for cb, args in callbacks: cb(*args)
    p.host.stop_sequence.assert_called_once()
    p.host.run_sequence.assert_not_called()
    assert not p.capturing


def test_shared_preflight_failure_is_visible_on_scan_tab(monkeypatch, tmp_path):
    p = panel(monkeypatch, tmp_path)
    p.host.sequence_execution_preflight_error.return_value = "Connect Fastech Ethernet before executing touch-sensing"
    p.run_pass_motion(True)
    assert "Connect Fastech Ethernet" in p.pass_status.get()
    p.host.run_sequence.assert_not_called()


def test_two_moves_pass_real_preflight_and_execute_sequentially_without_arc(tmp_path):
    from test_sequence_executor import make_host
    loaded = load_pass_file(write(tmp_path))
    current = Pose(); current.orientation.w = 1.
    steps = pass_motion_steps(loaded, current, "START → GOAL", velocity_scale=.3, tcp_speed_m_s=.005)
    host, executor = make_host(hicomm_connected=False, hicomm_client=None)
    assert executor.execution_preflight_error(steps, work_cycle=False) is None
    executor.run_groups(steps, [0, 1], True)
    calls = host.node.run_sequence_cartesian_motion.call_args_list
    assert len(calls) == 2
    assert [(c.args[0]["parallel_slot"], c.args[1]) for c in calls] == [(1, True), (2, True)]
    host._execute_hicomm_weld.assert_not_called()
    host._mark_weld_motion_timing.assert_not_called()
