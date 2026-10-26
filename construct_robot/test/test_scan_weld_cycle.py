"""Four-pass scan/weld sequencing without equipment or ROS action submission."""

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from construct_robot.application.scan_weld_cycle import (
    PASS4_WEAVE, build_scan_weld_cycle, expand_scan_weld_cycle, validate_scan_weld_cycle,
)
from construct_robot.application.sequence_executor import contains_weld_command, attach_execution_conditions
from construct_robot.core.sequence_model import validate_managed_weld_sequence
from construct_robot.io.precision_sensing_results import inspect_document
from test_weld_sequence_builder import MOTION, scenario
from test_sequence_executor import make_host, run


@pytest.fixture(autouse=True)
def cycle_log_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("KIRO_SCAN_CYCLE_LOG_ROOT", str(tmp_path / "cycle_logs"))


def cycle(*, rescan_after_passes=True, traverse_enabled=True, saved_folder=None):
    inputs = []
    for k in range(1, 5):
        inp = scenario(approach_mode="taught_wait", settings={
            "custom_hot_start_enabled": True, "custom_hot_start_hold_s": 0.2,
            "custom_hot_start_percent": 10.0, "software_crater_enabled": True,
            "software_crater_ratio_percent": 30, "software_crater_voltage_v": 20,
            "software_crater_hold_s": 0.4,
        })
        inp = copy.deepcopy(inp)
        # Distinct current corrected pass geometry, not four copies of P4.
        inp.endpoints.start.position.z = k * 0.001
        inp.endpoints.goal.position.z = k * 0.001
        for pose in (inp.endpoints.start, inp.endpoints.goal, inp.endpoints.start_wait[3], inp.goal_wait[3], inp.finish[3]):
            pose.orientation.x = pose.orientation.y = pose.orientation.z = 0.0
            pose.orientation.w = 1.0
        inputs.append((inp, replace(MOTION, fixed_world_y_tilt_deg=0, fixed_world_z_tilt_deg=0)))
    scan = dict(a=(0, 0, 0), b=(0.2, 0, 0), frame="World",
                action_name="precision", status_service="status", log_topic="log")
    if saved_folder is not None:
        from construct_robot.application.scan_weld_cycle import saved_scan_config
        scan = dict(saved_scan_config(saved_folder), plate_normal_world=[0, 0, 1])
    post = dict(action_name="post", status_service="status", log_topic="log",
                measure_speed_mps=0, confirm_mode="each", direction="", tool_orientation="pass")
    clean = [dict(type="sleep", seconds=0.1, torch_clean_scenario=True),
             dict(type="digital_output", io_backend="fastech_ethernet", port=7,
                  value=False, torch_clean_scenario=True)]
    return build_scan_weld_cycle(scan, post, inputs, clean, rescan_after_passes=rescan_after_passes,
                                 traverse_enabled=traverse_enabled)


def inspection(tmp_path, *, unresolved=False, reverse=False):
    document = dict(error_code=0, error_message="PASS_UNRESOLVED" if unresolved else "PASS",
                    paths=[], data_file_path=str(tmp_path / "precision_sensing.npz"))
    (tmp_path / "precision_sensing.npz").touch()
    (tmp_path / "kiro").mkdir(exist_ok=True)
    (tmp_path / "sections.json").write_text('{"base_frame":"World","frame":{"n":[0,0,1]}}')
    for k in range(1, 5):
        points = [dict(position=dict(x=x, y=0.3, z=k * 0.01),
                       orientation=dict(x=0, y=0, z=0, w=1)) for x in (1.0, 1.2)]
        document["paths"].append(dict(waypoints=points[::-1] if reverse else points))
        info = dict(derived=dict(frames=dict(tcp_frame="right_manipulator_ee_point", tcp_to_aim_point_mm=21.2)))
        (tmp_path / "kiro" / f"pass_{k}_info.yaml").write_text(yaml.safe_dump(info))
    return inspect_document(document, "World", run_dir=tmp_path)


def test_visible_order_recipes_and_hot_crater_snapshot():
    rows = cycle()
    phases = []
    for row in rows:
        phase = row["scan_cycle_phase"], row["scan_cycle_pass"]
        if not phases or phases[-1] != phase:
            phases.append(phase)
    assert phases == [("precision", 0), ("weld", 1), ("clean", 1), ("raw", 1),
                      ("weld", 2), ("clean", 2), ("raw", 2), ("weld", 3),
                      ("clean", 3), ("raw", 3), ("weld", 4), ("raw", 4)]
    passes = [r for r in rows if r["type"] == "scan_weld_pass"]
    assert [(r["settings"]["current_a"], r["settings"]["voltage"], r["weld_tcp_speed_mm_s"],
             r["weld_weave_enabled"]) for r in passes] == [
        (190, 24.5, 5, False), (230, 26.5, 7, False), (230, 26.5, 7, False), (220, 26, 5, True)]
    recipe_keys = [k for k in passes[0]["settings"] if ("hot_start" in k or "crater" in k)
                   and k not in ("software_crater_current_a", "hot_start_current_a")]
    assert all({key: p["settings"][key] for key in recipe_keys} ==
               {key: passes[0]["settings"][key] for key in recipe_keys} for p in passes)
    passes[0]["settings"]["software_crater_hold_s"] = 9
    assert passes[1]["settings"]["software_crater_hold_s"] == 0.4


def test_initial_scan_only_version_retains_four_welds_three_cleans_and_validation():
    rows = cycle(rescan_after_passes=False)
    assert sum(r["type"] == "precision_sensing_scan" for r in rows) == 1
    assert sum(r["type"] == "scan_weld_pass" for r in rows) == 4
    assert not any(r["type"] == "hongik_post_weld_scan" for r in rows)
    assert {r["scan_cycle_pass"] for r in rows if r["scan_cycle_phase"] == "clean"} == {1, 2, 3}
    expanded = expand_scan_weld_cycle(rows)
    assert validate_managed_weld_sequence(expanded, require_complete=True)
    assert [r["weld_tcp_speed_mm_s"] for r in expanded if r.get("weld_scenario_stage") == "weld_motion"] == [5, 7, 7, 5]
    with pytest.raises(ValueError):
        validate_scan_weld_cycle(rows[:-1])
    # Deleting the raw rows from a full cycle cannot silently change its mode.
    with pytest.raises(ValueError):
        validate_scan_weld_cycle([r for r in cycle() if r["type"] != "hongik_post_weld_scan"])


@pytest.mark.parametrize("rescans", [True, False])
def test_capping_preset_reaches_plan_and_fresh_scan_execution(rescans, tmp_path):
    rows = cycle(rescan_after_passes=rescans)
    p4 = next(row for row in rows if row.get("scan_cycle_pass") == 4 and row["type"] == "scan_weld_pass")
    assert p4["scenario"].weave_amplitude_mm == 4.0
    assert p4["scenario"].weave_pitch_mm == 4.0
    assert p4["scenario"].weave_left_dwell_s == p4["scenario"].weave_right_dwell_s == 0.2
    for sensed in (None, inspection(tmp_path)):
        expanded = expand_scan_weld_cycle(rows, sensed)
        motions = [row for row in expanded if row.get("weld_scenario_stage") == "weld_motion"]
        assert [row["weld_weave_enabled"] for row in motions] == [False, False, False, True]
        for name, value in PASS4_WEAVE.items():
            key = "weld_" + name
            assert motions[-1][key] == value


def test_initial_scan_only_execution_still_finishes_four_fake_arc_sessions(tmp_path):
    rows = cycle(rescan_after_passes=False)
    host, executor = make_host(fake_arc=True)
    host._prepare_scan_weld_cycle = lambda steps, execute: expand_scan_weld_cycle(steps, inspection(tmp_path))
    host._run_sequence_step = Mock(return_value=(True, "completed"))
    attach_execution_conditions(rows, {"mode": "sequence_execute"})
    run(executor, rows)
    assert host._sequence_finished.call_args.args[0] is True
    assert all(c.args[0]["type"] != "hongik_post_weld_scan" for c in host._run_sequence_step.call_args_list)
    assert sum(c.args[0] == "completed (fake ARC)" for c in host._finish_weld_feedback_record.call_args_list) == 4


def test_hongik_execute_sends_only_action_inputs_and_plan_never_submits_a_scan(tmp_path):
    from construct_robot.gui.weld_action_gui import WeldActionGui
    sensed = inspection(tmp_path)
    submitted = []
    def scan(*, a, b, frame, action_name, status_service, log_topic, on_progress,
             on_result, on_error, on_finished, cancel_requested):
        submitted.append((a, b, frame))
        assert not cancel_requested()
        return sensed
    panel = SimpleNamespace(busy=False, direct=None, bridge=SimpleNamespace(busy=False, scan=scan),
                            _set_busy=Mock(), show_result=Mock())
    host = SimpleNamespace(precision_sensing_test_panel=panel, sequence_stop_requested=False,
                           node=SimpleNamespace(scanner_live=SimpleNamespace(active=False)),
                           log=Mock(), post=lambda f, *args: f(*args))
    rows = cycle(rescan_after_passes=False)
    rows[0]["scan"].update(source="hongik", plate_normal_world=[0, 0, 1])
    WeldActionGui._prepare_scan_weld_cycle(host, rows, False)
    assert not submitted
    expanded = WeldActionGui._prepare_scan_weld_cycle(host, rows, True)
    assert len(submitted) == 1
    assert len([r for r in expanded if r.get("weld_scenario_stage") == "weld_motion"]) == 4


def test_expansion_is_serial_between_passes_and_preserves_weld_groups():
    rows = cycle()
    expanded = expand_scan_weld_cycle(rows)
    assert validate_managed_weld_sequence(expanded, require_complete=True)
    motions = [r for r in expanded if r.get("weld_scenario_stage") == "weld_motion"]
    assert [r["weld_tcp_speed_mm_s"] for r in motions] == [5, 7, 7, 5]
    assert [r["weld_weave_enabled"] for r in motions] == [False, False, False, True]
    assert [r["usable_seam_start"].position.z for r in motions] == pytest.approx([0.001, 0.002, 0.003, 0.004])
    for k in range(1, 5):
        weld = [r for r in expanded if r.get("scan_cycle_phase") == "weld" and r["scan_cycle_pass"] == k]
        raw = next(r for r in expanded if r["type"] == "hongik_post_weld_scan" and r["scan_cycle_pass"] == k)
        assert max(r["parallel_slot"] for r in weld) < raw["parallel_slot"]
        assert any(r.get("weld_scenario_stage") == "arc_off" for r in weld)
        assert any(r.get("weld_scenario_stage") == "software_crater" for r in weld)
        assert any(r.get("weld_scenario_stage") == "custom_hot_start" for r in weld)
    assert all(r["resolve_target_tcp_ik"] for r in expanded if r["type"] == "named_pose")


def test_fresh_result_replaces_xyz_preserves_attitude_wait_offsets_and_binds_run(tmp_path):
    rows = cycle()
    before = copy.deepcopy(rows)
    sensed = inspection(tmp_path)
    expanded = expand_scan_weld_cycle(rows, sensed)
    for k in range(1, 5):
        weld = [r for r in expanded if r.get("weld_scenario_stage") == "weld_motion" and r["scan_cycle_pass"] == k][0]
        pose = weld["usable_seam_start"]
        assert (pose.position.x, pose.position.y, pose.position.z) == pytest.approx((1, 0.3, k * 0.01 - 0.0212))
        assert pose.orientation.w == 1
        original = next(r for r in before if r["type"] == "scan_weld_pass" and r["scan_cycle_pass"] == k)["scenario"]
        new_wait = next(r for r in expanded if r.get("weld_scenario_stage") == "start_wait" and r["scan_cycle_pass"] == k)["tcp_pose"]
        for axis in "xyz":
            assert getattr(new_wait.position, axis) - getattr(pose.position, axis) == pytest.approx(
                getattr(original.endpoints.start_wait[3].position, axis) - getattr(original.endpoints.start.position, axis))
    scans = [r for r in expanded if r["type"] == "hongik_post_weld_scan"]
    assert [r["scan"]["after_pass"] for r in scans] == [1, 2, 3, 4]
    assert all(r["scan"]["run_dir"] == str(tmp_path) and r["scan"]["dry_run"] is False for r in scans)
    assert next(r for r in rows if r["type"] == "scan_weld_pass")["scenario"].endpoints.start.position.x == 0.5


@pytest.mark.parametrize("mode", ["unresolved", "missing_info", "reversed", "missing_file", "three_passes"])
def test_bad_measurement_prevents_any_weld_expansion(tmp_path, mode):
    sensed = inspection(tmp_path, unresolved=mode == "unresolved", reverse=mode == "reversed")
    if mode == "missing_info":
        (tmp_path / "kiro" / "pass_3_info.yaml").unlink()
    if mode == "missing_file":
        (tmp_path / "precision_sensing.npz").unlink()
    if mode == "three_passes":
        sensed.document["paths"].pop()
    with pytest.raises((ValueError, FileNotFoundError)):
        expand_scan_weld_cycle(cycle(), sensed)


def test_deleted_or_reordered_macro_cannot_run():
    rows = cycle()
    with pytest.raises(ValueError):
        validate_scan_weld_cycle(rows[1:])
    rows[1]["parallel_slot"] = rows[0]["parallel_slot"]
    with pytest.raises(ValueError):
        validate_scan_weld_cycle(rows)


def test_macro_preflight_requires_welder_and_right_robot():
    rows = cycle()
    host, executor = make_host(hicomm_connected=False)
    assert contains_weld_command(rows)
    assert executor.execution_preflight_error(rows, False) == "Connect Hi-COMM"
    host.hicomm_connected = True
    host.robot_connected["right"] = False
    assert "right" in executor.execution_preflight_error(rows, False)


def test_plan_expands_without_scan_and_keeps_output_calls_disabled():
    rows = cycle()
    host, executor = make_host()
    current = copy.deepcopy(rows[1]["scenario"].endpoints.start)
    current.position.z += .010
    host.node._current_tcp_pose = Mock(return_value=current)
    host._prepare_scan_weld_cycle = lambda steps, execute: expand_scan_weld_cycle(steps)
    host._run_hongik_post_weld_scan = Mock(return_value=(True, "planned scan"))
    run(executor, rows, execute=False)
    host._execute_hicomm_weld.assert_not_called()
    host._execute_software_crater.assert_not_called()
    host._run_hongik_post_weld_scan.assert_called()
    assert all(call.args[1] is False for call in host._run_hongik_post_weld_scan.call_args_list)
    host._set_fastech_output_sync.assert_not_called()
    host._sequence_finished.assert_called_once_with(True, "complete")


def test_initial_scan_failure_never_calls_weld_or_cleaner():
    host, executor = make_host()
    host._prepare_scan_weld_cycle = Mock(side_effect=RuntimeError("scan failed"))
    run(executor, cycle())
    host.node.run_sequence_named_pose.assert_not_called()
    host._execute_hicomm_weld.assert_not_called()
    host._set_fastech_output_sync.assert_not_called()
    assert host._sequence_finished.call_args.args[0] is False


def test_raw_scan_failure_stops_before_next_pass(tmp_path):
    rows = cycle()
    sensed = inspection(tmp_path)
    host, executor = make_host()
    host._prepare_scan_weld_cycle = lambda steps, execute: expand_scan_weld_cycle(steps, sensed)
    visited = []
    def member(step, execute):
        visited.append((step.get("scan_cycle_pass"), step.get("weld_scenario_stage"), step["type"]))
        return (False, "raw scan failed") if step["type"] == "hongik_post_weld_scan" else (True, "ok")
    host._run_sequence_step = member
    attach_execution_conditions(rows, {"mode": "sequence_execute"})
    run(executor, rows)
    assert max(k for k, _, _ in visited) == 1
    assert host._sequence_finished.call_args.args == (False, "raw scan failed")
    conditions = host._begin_weld_feedback_record.call_args.args[1]
    assert conditions["steps"][0]["scan_cycle_run_dir"] == str(tmp_path)


def test_second_pass_failure_restores_second_pass_recipe(tmp_path):
    sensed = inspection(tmp_path)
    host, executor = make_host()
    host._prepare_scan_weld_cycle = lambda steps, execute: expand_scan_weld_cycle(steps, sensed)
    host._run_sequence_step = lambda step, execute: ((False, "P2 failed")
        if step.get("weld_scenario_stage") == "software_crater" and step["scan_cycle_pass"] == 2
        else (True, "ok"))
    run(executor, cycle())
    assert host._software_crater_restore.call_args.args[0]["current_a"] == 230


@pytest.mark.parametrize("scan_source", ["hongik", "local_1g"])
@pytest.mark.parametrize("rescans", [True, False])
@pytest.mark.parametrize("traverse", [True, False])
def test_gui_build_snapshots_each_current_pass_without_motion_or_arc(scan_source, rescans, traverse, tmp_path):
    from construct_robot.gui.weld_action_gui import WeldActionGui
    from construct_robot.application.weld_sequence_builder import cleaner_cut_only_tokens, build_cleaner_sequence_steps
    from construct_robot.io.teaching_yaml import teaching_config_dir, load_cleaner_order, load_initial_state_yaml
    from test_weld_sequence_builder import teaching
    refs, corrected = {}, {}
    for k in range(1, 5):
        t = teaching()
        mapping = {"start_wait": "weld_start_wait", "start": "weld_start",
                   "goal_wait": "weld_goal_wait", "goal": "weld_end"}
        corrected[k] = {endpoint: copy.deepcopy(t[name][3]) for endpoint, name in mapping.items()}
        for pose in corrected[k].values():
            pose.position.z += k * 0.01
        finish = t["weld_finish"]
        entry = dict(planning_group=finish[0], joint_state=dict(names=list(finish[1]), positions_rad=list(finish[2])),
                     tcp_pose_world=dict(position_m=dict(x=0.75, y=0.1, z=0.4), orientation_xyzw=dict(x=0, y=0, z=0, w=1)))
        refs[k] = dict(joint_states={end: t[name][1:3] for end, name in mapping.items()},
                       additional_pose_entries={"weld_wait": entry, "weld_finish": entry})
    def var(value):
        item = SimpleNamespace(value=value)
        item.get = lambda: item.value
        item.set = lambda new: setattr(item, "value", new)
        return item
    panel = SimpleNamespace(busy=False, bridge=SimpleNamespace(busy=False), direct=None,
        a=[var(0), var(0), var(0)], b=[var(200), var(0), var(0)], frame=var("World"),
        action=var("precision"), status_service=var("status"), log_topic=var("log"),
        post_weld_orientation=var("Sensing pass"), post_weld_speed=var("0"), post_weld_confirm=var("each"),
        post_weld_action=var("post"), post_weld_direction=var("Auto"))
    if scan_source == "local_1g":
        paths = [tmp_path / "c.yaml", tmp_path / "d.yaml"]
        for path in paths:
            path.write_text("local scan teaching")
        cfg = dict(source="local_1g", start_path=str(paths[0]), end_path=str(paths[1]),
                   relative="measured_layout", apply_correction=True, plate_normal_world=(0, 0, 1))
        panel.local_cycle_scan_config = lambda: (cfg, cfg)
    folder = teaching_config_dir() / "torch_cleaner_teaching"
    def build_cleaner(cut_only=False):
        tokens = load_cleaner_order(folder)
        if cut_only:
            tokens = cleaner_cut_only_tokens(tokens)
        return build_cleaner_sequence_steps(folder, tokens, 5, load_initial_state_yaml)
    cleaner_builder = Mock(side_effect=build_cleaner)
    def snapshot(**kwargs):
        inp = scenario(approach_mode="taught_wait")
        assert kwargs["return_inputs"] and not kwargs["append"]
        return replace(inp, endpoints=replace(inp.endpoints, start=kwargs["teaching_poses"]["weld_start"][3],
                                               goal=kwargs["teaching_poses"]["weld_end"][3])), MOTION
    host = SimpleNamespace(sequence_running=False, precision_sensing_test_panel=panel,
        SCAN_CYCLE_SOURCE="local_1g" if scan_source == "hongik" else "hongik",
        hongik_cycle_scan_enabled=var(scan_source == "hongik"),
        scan_cycle_traverse_enabled=var(traverse),
        four_pass_references=refs, four_pass_corrected=corrected, error=Mock(), log=Mock(),
        _validate_four_pass_source_hashes=Mock(), build_sensed_weld_sequence=snapshot,
        weave_axis=var("tool_y"), refresh_sequence_table=Mock(),
        **{name: var(None) for name in PASS4_WEAVE},
        torch_cleaner_panel=SimpleNamespace(build_sequence_steps=cleaner_builder))
    assert WeldActionGui.build_scan_weld_cycle(host, rescan_after_passes=rescans) is True
    host.error.assert_not_called()
    assert {name: getattr(host, name).get() for name in PASS4_WEAVE} == PASS4_WEAVE
    host._validate_four_pass_source_hashes.assert_called_once()
    cleaner_builder.assert_called_once_with(cut_only=True)
    assert sum(row["type"] == "hongik_post_weld_scan" for row in host.sequence_steps) == (4 if rescans else 0)
    assert host.sequence_steps[0]["scan"]["source"] == scan_source
    passes = [row for row in host.sequence_steps if row["type"] == "scan_weld_pass"]
    assert [bool(row["start_touch_probe"]) for row in passes] == [traverse] * 4
    host.scan_cycle_traverse_enabled.set(not traverse)
    # The next checkbox selection cannot change an already built execution.
    contacts = [row for row in expand_scan_weld_cycle(host.sequence_steps)
                if row.get("weld_scenario_stage") == "start_contact"]
    assert [bool(row.get("start_touch_probe")) for row in contacts] == [traverse] * 4
    assert [row["touch_guard"] for row in contacts] == [traverse] * 4
    if scan_source == "local_1g":
        assert host.sequence_steps[0]["scan"]["apply_correction"] is True
        assert host.sequence_steps[0]["scan"]["relative"] == "measured_layout"
    for k in (1, 2, 3):
        cleaner_rows = [r for r in host.sequence_steps if r.get("scan_cycle_phase") == "clean" and r["scan_cycle_pass"] == k]
        labels = [r["pose_label"] for r in cleaner_rows if r["type"] == "named_pose"]
        assert labels == ["Cleaner start", *(f"Cleaner clear1_{i}" for i in range(1, 6)), "Cleaner start"]
        assert {r["port"] for r in cleaner_rows if r["type"] == "digital_output"} == {7}
        # 1-3, DO7 on, 1-4, DO7 off, wire feed (waited), DO7 on, 1-5, DO7 off
        feed = next(i for i, row in enumerate(cleaner_rows) if row["type"] == "wire_feed")
        assert cleaner_rows[feed]["duration"] == 0.75
        assert cleaner_rows[feed - 1]["type"] == "digital_output" and cleaner_rows[feed - 1]["value"] is False
        assert cleaner_rows[feed + 1]["type"] == "digital_output" and cleaner_rows[feed + 1]["value"] is True
        poses = [(i, r["pose_label"]) for i, r in enumerate(cleaner_rows) if r["type"] == "named_pose"]
        assert [label for i, label in poses if i < feed][-1] == "Cleaner clear1_4"
        assert [label for i, label in poses if i > feed][0] == "Cleaner clear1_5"
    assert [r["scenario"].endpoints.start.position.z for r in host.sequence_steps
            if r["type"] == "scan_weld_pass"] == pytest.approx([0.11, 0.12, 0.13, 0.14])



def local_cycle(tmp_path):
    for name in ("c.yaml", "d.yaml"):
        (tmp_path / name).write_text("x")
    inputs = [args for args in (cycle_inputs())]
    common = dict(source="local_1g", start_path=str(tmp_path / "c.yaml"), end_path=str(tmp_path / "d.yaml"),
                  tcp_speed_m_s=0.03, velocity_scale=0.3, plate_normal_world=(0, 0, 1))
    clean = [dict(type="sleep", seconds=0.1, torch_clean_scenario=True)]
    return build_scan_weld_cycle(dict(common, relative="measured_layout", apply_correction=True), dict(common),
                                 inputs, clean)


def cycle_inputs():
    rows = cycle()
    return [(r["scenario"], r["motion"]) for r in rows if r["type"] == "scan_weld_pass"]


def test_local_1g_cycle_needs_no_hongik_points_and_expands_from_the_local_targets(tmp_path):
    rows = local_cycle(tmp_path)
    assert rows[0]["type"] == "precision_sensing_scan" and rows[0]["scan"]["source"] == "local_1g"
    q = (0.0, 0.0, 0.0, 1.0)
    targets = {k: {"start": ((0.1, 0.2, 0.3 + k * 0.01), q), "goal": ((0.3, 0.2, 0.3 + k * 0.01), q),
                   "start_wait": ((0.1, 0.2, 0.4), q), "goal_wait": ((0.3, 0.2, 0.4), q),
                   "weld_wait": ((0.1, 0.2, 0.43), q), "weld_finish": ((0.3, 0.2, 0.43), q)} for k in range(1, 5)}
    targets[4]["weave_geometry"] = {"normal_world": (0, 0, 1)}
    expanded = expand_scan_weld_cycle(rows, None, local={"scan_dir": str(tmp_path / "s"), "targets": targets})
    for k in range(1, 5):
        steps = [s for s in expanded if s.get("scan_cycle_pass") == k and s.get("weld_scenario_id")]
        named = {s["pose_name"]: s for s in steps if s["type"] == "named_pose"}
        assert named["weld_finish"]["tcp_pose"].position.z == pytest.approx(0.43)
        assert named["weld_finish"]["resolve_target_tcp_ik"] is True
        weld = next(s for s in steps if s.get("weld_scenario_stage") == "weld_motion")
        assert weld["points"][0].position.z == pytest.approx(0.3 + k * 0.01)
    raw = [s for s in expanded if s["type"] == "hongik_post_weld_scan"]
    assert len(raw) == 4 and all(s["scan"]["source"] == "local_1g" for s in raw)


def test_local_1g_cycle_refuses_missing_scan_poses(tmp_path):
    with pytest.raises(ValueError, match="local scan poses"):
        build_scan_weld_cycle(
            dict(source="local_1g", start_path=str(tmp_path / "x.yaml"), end_path=str(tmp_path / "y.yaml")), {},
            cycle_inputs(), [dict(type="sleep", seconds=0.1, torch_clean_scenario=True)])


def test_local_raw_scan_rows_run_the_local_post_weld_scan():
    from construct_robot.gui.weld_action_gui import WeldActionGui
    panel = SimpleNamespace(run_post_weld_scan=Mock(return_value=(True, "saved")))
    host = SimpleNamespace(precision_sensing_test_panel=panel)
    step = {"scan": {"source": "local_1g", "start_path": "c", "end_path": "d"}, "scan_cycle_pass": 2}
    assert WeldActionGui._run_hongik_post_weld_scan(host, step, True) == (True, "saved")
    panel.run_post_weld_scan.assert_called_once_with(step["scan"], True)


def test_fresh_pass4_weave_is_symmetric_in_the_scanned_plate_plane(tmp_path):
    import numpy as np
    rows = local_cycle(tmp_path)
    # A tilted surface, skewed seam and tool pose must override the manual world/tool axis.
    normal = np.array([0.2, -0.3, 1.0])
    normal /= np.linalg.norm(normal)
    q = (0.7071067811865476, 0, 0, 0.7071067811865476)
    start, goal = np.array([0.1, 0.2, 0.3]), np.array([0.3, 0.24, 0.32])
    targets = {k: {"start": (start, q), "goal": (goal, q),
                   "start_wait": (start + [0, 0, .01], q), "goal_wait": (goal + [0, 0, .01], q),
                   "weld_wait": (start + [0, 0, .07], q), "weld_finish": (goal + [0, 0, .07], q)} for k in range(1, 5)}
    targets[4]["weave_geometry"] = {"normal_world": normal.tolist()}
    expanded = expand_scan_weld_cycle(rows, local={"scan_dir": str(tmp_path), "targets": targets})
    motion = next(r for r in expanded if r.get("weld_scenario_stage") == "weld_motion" and r["scan_cycle_pass"] == 4)
    direction = np.asarray(motion["weld_weave_transverse_vector"])
    travel = goal - start
    assert abs(direction @ normal) < 1e-12 and abs(direction @ travel) < 1e-12
    assert motion["weld_weave_reference"] == "scan_plate_plane"
    np.testing.assert_allclose(motion["weld_plate_normal_world"], normal)
    from construct_robot.application.sequence_executor import record_step_conditions
    recorded = record_step_conditions([motion], [0])[0]
    assert recorded["weld_weave_reference"] == "scan_plate_plane"
    np.testing.assert_allclose(recorded["weld_plate_normal_world"], normal)
    np.testing.assert_allclose(recorded["weld_weave_transverse_vector"], direction)
    offsets = []
    for p in motion["usable_weld_points"]:
        xyz = np.array([getattr(p.position, a) for a in "xyz"])
        centre = start + travel * float((xyz - start) @ travel) / float(travel @ travel)
        delta = xyz - centre
        assert abs(delta @ normal) < 1e-12
        offsets.append(float(delta @ direction))
    amplitude = motion["weld_weave_amplitude_mm"] * .001
    assert max(offsets) == pytest.approx(amplitude) and min(offsets) == pytest.approx(-amplitude)
    # Plan uses the same geometry rule on its stored teaching.
    planned = expand_scan_weld_cycle(rows)
    preview = next(r for r in planned if r.get("weld_scenario_stage") == "weld_motion" and r["scan_cycle_pass"] == 4)
    assert abs(np.dot(preview["weld_weave_transverse_vector"], rows[0]["scan"]["plate_normal_world"])) < 1e-12


def test_no_scan_cycle_uses_frozen_teaching_without_any_scan_stage(tmp_path):
    from construct_robot.application.scan_weld_cycle import validate_saved_scan
    for k in range(1, 5):
        (tmp_path / f"pass_{k}.yaml").write_text(f"pass: {k}\n")
    rows = cycle(rescan_after_passes=False, saved_folder=tmp_path)
    assert rows[0]["type"] == "saved_scan_load"
    assert not any(r["type"] in ("precision_sensing_scan", "hongik_post_weld_scan") for r in rows)
    validate_scan_weld_cycle(rows)
    expanded = expand_scan_weld_cycle(rows)
    assert validate_managed_weld_sequence(expanded, require_complete=True)
    welds = [r for r in expanded if r.get("weld_scenario_stage") == "weld_motion"]
    assert len(welds) == 4
    assert [r["weld_tcp_speed_mm_s"] for r in welds] == [5, 7, 7, 5]
    validate_saved_scan(rows[0]["scan"])
    (tmp_path / "pass_2.yaml").write_text("changed\n")
    with pytest.raises(ValueError, match="changed after Build"):
        validate_saved_scan(rows[0]["scan"])


def test_no_scan_refuses_missing_teaching_and_rescans(tmp_path):
    from construct_robot.application.scan_weld_cycle import saved_scan_config
    with pytest.raises(FileNotFoundError):
        saved_scan_config(tmp_path)
    for k in range(1, 5):
        (tmp_path / f"pass_{k}.yaml").write_text(f"pass: {k}\n")
    with pytest.raises(ValueError, match="cannot contain rescans"):
        cycle(saved_folder=tmp_path)


def test_hongik_result_with_only_start_and_goal_per_pass_drives_the_passes(tmp_path):
    """Current PrecisionSensing interface: paths[k-1] = {pass_number, start, goal} (wire tips); the pass keeps its
    taught torch attitude and only the positions move."""
    from construct_robot.io.precision_sensing_results import inspect_document
    document = dict(error_code=0, error_message="PASS", paths=[], data_file_path=str(tmp_path / "precision_sensing.npz"))
    (tmp_path / "precision_sensing.npz").touch()
    (tmp_path / "kiro").mkdir()
    (tmp_path / "sections.json").write_text('{"base_frame":"World","frame":{"n":[0,0,1]}}')
    for k in range(1, 5):
        end = lambda x: dict(position=dict(x=x, y=0.3, z=k * 0.01), orientation=dict(x=0, y=0, z=0, w=1))
        document["paths"].append(dict(pass_number=k, start=end(1.0), goal=end(1.2)))
        info = dict(derived=dict(frames=dict(tcp_frame="right_manipulator_ee_point", tcp_to_aim_point_mm=21.2)))
        (tmp_path / "kiro" / f"pass_{k}_info.yaml").write_text(yaml.safe_dump(info))
    inspection = inspect_document(document, "World", run_dir=tmp_path)
    wire = [p for p in inspection.poses if p.kind == "wire"]
    assert len(wire) == 8 and [p.pass_number for p in wire] == [1, 1, 2, 2, 3, 3, 4, 4]
    rows = expand_scan_weld_cycle(cycle(rescan_after_passes=False), inspection)
    assert any(r.get("scan_cycle_phase") == "weld" for r in rows)
