"""TASK › Scan keyboard teaching: reference mode (I O P J / K L M U = pass 1-4 start / goal references) or
scan pose mode (I = scan start A, J = scan end B, O / K = local C / D); no robot motion."""

import math
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import Pose
import pytest
import yaml

from construct_robot.gui import precision_sensing_test_panel as panel_module
from construct_robot.gui.precision_sensing_test_panel import (
    ORIENTATION_KEYS, PrecisionSensingTestPanel, REFERENCE_KEYS, SCAN_KEY_POINTS,
)
from construct_robot.gui.weld_action_gui import WeldActionGui
from test_torch_cleaner_sequence import keyboard_gui


def test_scan_key_map_does_not_change_the_cleaner_keys():
    assert SCAN_KEY_POINTS == {"i": "start", "j": "end", "o": "local_start", "k": "local_end"}


def scan_gui(arm="right", enabled=True):
    gui = keyboard_gui(cleaner=False, arm=arm)
    gui._scan_task_selected = lambda: True
    gui.keyboard_jog_enabled = SimpleNamespace(get=lambda: enabled)
    gui.precision_sensing_test_panel = SimpleNamespace(capture_scan_point=Mock())
    return gui


@pytest.mark.parametrize("key,name", [("i", "start"), ("j", "end"), ("o", "local_start"), ("k", "local_end")])
def test_scan_tab_keys_save_their_point(key, name):
    gui = scan_gui()
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym=key)) == "break"
    gui.precision_sensing_test_panel.capture_scan_point.assert_called_once_with(name)
    gui.torch_cleaner_panel.capture.assert_not_called()


@pytest.mark.parametrize("gui", [scan_gui(arm=None), scan_gui(arm="left"), scan_gui(enabled=False)],
                         ids=["no-teaching", "left-arm", "teaching-disabled"])
def test_scan_keys_need_right_arm_keyboard_teaching(gui):
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="i")) == "break"
    gui.precision_sensing_test_panel.capture_scan_point.assert_not_called()
    gui.error.assert_called_once()


def test_other_teaching_keys_are_not_saved_on_the_scan_tab():
    gui = scan_gui()
    for key in ("p", "l", "m", "u"):
        assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym=key)) == "break"
    gui.precision_sensing_test_panel.capture_scan_point.assert_not_called()
    gui.torch_cleaner_panel.capture.assert_not_called()


def test_scan_keys_are_refused_during_motion():
    gui = scan_gui()
    gui.node.active_motion_goal = object()
    gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="j"))
    gui.precision_sensing_test_panel.capture_scan_point.assert_not_called()
    gui.error.assert_called_once()


def reference_gui(on=True, mode=None, **kw):
    gui = scan_gui(**kw)
    gui.precision_sensing_test_panel.capture_reference = Mock()
    gui.precision_sensing_test_panel.capture_orientation = Mock()
    mode = mode or ("reference" if on else "scan_poses")
    gui.precision_sensing_test_panel.scan_key_mode = SimpleNamespace(get=lambda: mode)
    return gui


@pytest.mark.parametrize("key", sorted(ORIENTATION_KEYS))
def test_orientation_mode_keys_save_the_pass_orientations(key):
    gui = reference_gui(mode="orientation")
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym=key)) == "break"
    gui.precision_sensing_test_panel.capture_orientation.assert_called_once_with(ORIENTATION_KEYS[key])
    gui.precision_sensing_test_panel.capture_reference.assert_not_called()
    gui.precision_sensing_test_panel.capture_scan_point.assert_not_called()


def test_orientation_keys_map_i_to_passes_1_and_3():
    assert ORIENTATION_KEYS == {"i": (1, 3), "o": (2,), "p": (4,)}


def test_orientation_mode_ignores_the_other_keys():
    gui = reference_gui(mode="orientation")
    for key in ("j", "k", "l", "m", "u"):
        assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym=key)) == "break"
    gui.precision_sensing_test_panel.capture_orientation.assert_not_called()
    assert "I=P1+P3 orientation" in gui.keyboard_jog_status.set.call_args[0][0]


def test_reference_keys_cover_four_starts_and_four_goals():
    assert sorted(REFERENCE_KEYS.values()) == [(k, end) for k in (1, 2, 3, 4) for end in ("goal", "start")]
    assert [REFERENCE_KEYS[key] for key in "iopj"] == [(k, "start") for k in (1, 2, 3, 4)]
    assert [REFERENCE_KEYS[key] for key in "klmu"] == [(k, "goal") for k in (1, 2, 3, 4)]


@pytest.mark.parametrize("key", sorted(REFERENCE_KEYS))
def test_reference_mode_keys_save_the_pass_reference(key):
    gui = reference_gui()
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym=key)) == "break"
    pass_k, end = REFERENCE_KEYS[key]
    gui.precision_sensing_test_panel.capture_reference.assert_called_once_with(end, pass_k)
    gui.precision_sensing_test_panel.capture_scan_point.assert_not_called()
    gui.torch_cleaner_panel.capture.assert_not_called()


def test_reference_mode_off_keeps_the_scan_pose_keys():
    gui = reference_gui(on=False)
    gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="o"))
    gui.precision_sensing_test_panel.capture_scan_point.assert_called_once_with("local_start")
    gui.precision_sensing_test_panel.capture_reference.assert_not_called()


@pytest.mark.parametrize("kw", [{"arm": "left"}, {"enabled": False}], ids=["left-arm", "teaching-disabled"])
def test_reference_keys_need_right_arm_keyboard_teaching(kw):
    gui = reference_gui(**kw)
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="u")) == "break"
    gui.precision_sensing_test_panel.capture_reference.assert_not_called()
    gui.error.assert_called_once()


def test_reference_keys_are_refused_during_motion():
    gui = reference_gui()
    gui.node.active_motion_goal = object()
    gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="m"))
    gui.precision_sensing_test_panel.capture_reference.assert_not_called()
    gui.error.assert_called_once()


def tcp_pose():
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = 0.5, -1.0, 0.9
    # 90 deg about +X: the TCP z axis points along World -Y.
    pose.orientation.x, pose.orientation.w = math.sqrt(0.5), math.sqrt(0.5)
    return pose


def capture_panel(monkeypatch, tmp_path, frame="World"):
    interp = tk.Tcl()
    panel = object.__new__(PrecisionSensingTestPanel)
    names = [f"right_manipulator_joint{i}" for i in range(1, 7)]
    offset = SimpleNamespace(transform=SimpleNamespace(translation=SimpleNamespace(x=0.0, y=0.0, z=0.020)))
    node = SimpleNamespace(
        capture_measured_teaching_snapshot=Mock(return_value=(names, [0.1] * 6, tcp_pose(), {"capture_source": "test"})),
        tf_buffer=SimpleNamespace(lookup_transform=Mock(return_value=offset)),
    )
    panel.host = SimpleNamespace(node=node, post=lambda callback, *args: callback(*args), log=Mock())
    panel.busy, panel.capturing = False, False
    panel.scan_folder = tmp_path
    panel.frame = tk.StringVar(interp, frame)
    panel.status = tk.StringVar(interp, "")
    panel.scan_poses_text = tk.StringVar(interp, "")
    panel.a = [tk.StringVar(interp) for _ in range(3)]
    panel.b = [tk.StringVar(interp) for _ in range(3)]
    panel.show_error = Mock()
    monkeypatch.setattr(panel_module.threading, "Thread", lambda target, **kw: SimpleNamespace(start=target))
    return panel


def test_sensor_pose_capture_keeps_action_seam_points_separate(monkeypatch, tmp_path):
    panel = capture_panel(monkeypatch, tmp_path)
    panel.capture_scan_point("end")
    panel.show_error.assert_not_called()
    lookup = panel.host.node.tf_buffer.lookup_transform.call_args[0]
    assert lookup[:2] == ("right_manipulator_ee_point", "hongik_weld_link")
    # TCP + R(q)·(0, 0, 20 mm): the reference follows the rotated torch axis.
    assert [v.get() for v in panel.b] == ["", "", ""]
    assert [v.get() for v in panel.a] == ["", "", ""]
    document = yaml.safe_load((tmp_path / "scan_end.yaml").read_text())
    assert document["planning_group"] == "right_manipulator"
    assert document["joint_state"]["positions_rad"] == [0.1] * 6
    provenance = document["capture_provenance"]
    assert provenance["scan_point_world_m"] == pytest.approx([0.5, -1.02, 0.9])
    assert provenance["scan_point_frame"] == "hongik_weld_link" and provenance["capture_source"] == "test"
    assert not panel.capturing
    assert "B TCP (500.0, -1000.0, 900.0) mm" in panel.scan_poses_text.get()

    reloaded = capture_panel(monkeypatch, tmp_path)
    reloaded.load_scan_points()
    assert [v.get() for v in reloaded.b] == ["", "", ""]
    assert [v.get() for v in reloaded.a] == ["", "", ""]


def test_seam_capture_fills_action_fields_and_does_not_replace_sensor_pose(monkeypatch, tmp_path):
    panel = capture_panel(monkeypatch, tmp_path)
    panel.capture_scan_point("end")
    original = (tmp_path / "scan_end.yaml").read_bytes()
    panel.capture_scan_point("seam_end")
    panel.show_error.assert_not_called()
    assert [float(v.get()) for v in panel.b] == pytest.approx([500.0, -1020.0, 900.0])
    assert (tmp_path / "scan_end.yaml").read_bytes() == original
    document = yaml.safe_load((tmp_path / "seam_end.yaml").read_text())
    assert document["capture_provenance"]["teaching_kind"] == "hongik_seam_point"
    reloaded = capture_panel(monkeypatch, tmp_path)
    reloaded.load_scan_points()
    assert [float(v.get()) for v in reloaded.b] == pytest.approx([500.0, -1020.0, 900.0])


def test_capture_is_refused_outside_world_or_while_busy(monkeypatch, tmp_path):
    panel = capture_panel(monkeypatch, tmp_path, frame="link0")
    panel.capture_scan_point("start")
    panel.busy = True
    panel.frame.set("World")
    panel.capture_scan_point("start")
    assert panel.show_error.call_count == 2
    panel.host.node.capture_measured_teaching_snapshot.assert_not_called()
    assert not (tmp_path / "scan_start.yaml").exists()


def test_failed_capture_reports_and_keeps_previous_values(monkeypatch, tmp_path):
    panel = capture_panel(monkeypatch, tmp_path)
    panel.a[0].set("123.000")
    panel.host.node.tf_buffer.lookup_transform.side_effect = RuntimeError("hongik_weld_link does not exist")
    panel.capture_scan_point("start")
    assert "hongik_weld_link" in panel.show_error.call_args[0][0]
    assert panel.a[0].get() == "123.000" and not panel.capturing
    assert not (tmp_path / "scan_start.yaml").exists()


def test_scan_tab_selection_uses_the_third_task_tab():
    gui = object.__new__(WeldActionGui)
    gui.task_notebook = SimpleNamespace(master="task", tabs=lambda: ("w", "c", "s"), select=lambda: "s")
    gui.operator_notebook = SimpleNamespace(select=lambda: "task")
    assert gui._scan_task_selected() and not gui._cleaner_task_selected()


def test_seam_tilt_matches_the_hongik_1g_rule():
    from construct_robot.core.precision_sensing_inspection import seam_tilt_deg
    assert seam_tilt_deg((0, 0, 0), (0.1, 0, 0)) == pytest.approx((0.0, 0.1))
    assert seam_tilt_deg((0, 0, 0.67), (0, 0, 0.57))[0] == pytest.approx(90.0)
    assert seam_tilt_deg((0, 0, 0), (0.1, 0, 0.1))[0] == pytest.approx(45.0)
    with pytest.raises(ValueError):
        seam_tilt_deg((1, 2, 3), (1, 2, 3))


def test_start_scan_refuses_a_vertical_seam_before_contacting_the_server():
    from test_precision_sensing_test import panel_state
    panel = panel_state()
    for var, value in zip(panel.a, ("61.6", "-1715.9", "670.8")):
        var.set(value)
    for var, value in zip(panel.b, ("61.6", "-1715.9", "569.0")):
        var.set(value)
    panel.start_scan()
    panel.bridge.scan.assert_not_called()
    assert "90° from horizontal" in panel.show_error.call_args[0][0]


def test_local_points_are_saved_separately_and_do_not_touch_a_b(monkeypatch, tmp_path):
    panel = capture_panel(monkeypatch, tmp_path)
    panel.capture_scan_point("local_start")
    panel.show_error.assert_not_called()
    assert (tmp_path / "scan_local_start.yaml").is_file() and not (tmp_path / "scan_start.yaml").exists()
    assert [v.get() for v in panel.a + panel.b] == [""] * 6
    assert "Local scan poses (World): C TCP (500.0, -1000.0, 900.0) mm · D TCP not taught" in panel.scan_poses_text.get()


def test_key_reference_capture_uses_the_key_pass_not_the_selector(monkeypatch, tmp_path):
    panel = capture_panel(monkeypatch, tmp_path)
    panel.reference_pass = tk.StringVar(panel.frame._root, "1")
    panel.correction_text = tk.StringVar(panel.frame._root, "")
    panel.show_correction = Mock()
    panel.host.error = Mock()
    save = Mock(return_value=tmp_path / "root1_reference.yaml")
    monkeypatch.setattr(panel_module.scan_correction, "save_reference", save)
    panel.capture_reference("goal", 3)
    (pass_k, label, position, quat, _source), _ = save.call_args
    assert (pass_k, label) == (3, "goal")
    assert position == pytest.approx((0.5, -1.0, 0.9))
    assert quat == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
    panel.host.node.capture_measured_teaching_snapshot.assert_called_once_with("right_manipulator", "pass3_ref_goal")
    assert not panel.capturing and "P3 goal reference saved" in panel.correction_text.get()


def test_tcp_calibration_mode_i_adds_a_sample():
    gui = reference_gui(mode="tcp_cal")
    gui.precision_sensing_test_panel.capture_tcp_sample = Mock()
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="i")) == "break"
    gui.precision_sensing_test_panel.capture_tcp_sample.assert_called_once_with()
    gui.precision_sensing_test_panel.capture_orientation.assert_not_called()


def test_new_truth_capture_stores_each_pass_joint_seed_without_changing_taught_attitudes(monkeypatch, tmp_path):
    panel = capture_panel(monkeypatch, tmp_path)
    panel.reference_pass = tk.StringVar(panel.frame._root, "1")
    panel.correction_text = tk.StringVar(panel.frame._root, "")
    panel.show_correction = Mock()
    panel.host.error = Mock()
    monkeypatch.setattr(panel_module.scan_correction, "load_reference_set", lambda: {"id": "new_set"})
    save = Mock(return_value=tmp_path / "root1_reference.yaml")
    monkeypatch.setattr(panel_module.scan_correction, "save_reference", save)
    attitude = Mock()
    monkeypatch.setattr(panel_module.scan_four_pass, "save_pass_orientation", attitude)
    for number in range(1, 5):
        panel.capture_reference("start", number)
        assert save.call_args.args[:2] == (number, "start")
        names, values = save.call_args.kwargs["joints"]
        assert names == [f"right_manipulator_joint{i}" for i in range(1, 7)]
        assert values == [0.1] * 6
    attitude.assert_not_called()             # pending captures cannot alter active path attitudes


def test_truth_compute_uses_its_scan_and_p1_anchor_instead_of_the_legacy_p2_selector(monkeypatch, tmp_path):
    panel = capture_panel(monkeypatch, tmp_path)
    interp = panel.frame._root
    panel.correction_text = tk.StringVar(interp, "")
    panel.correction_anchor = tk.StringVar(interp, "2")
    panel.relative_model = tk.StringVar(interp, "hongik_tcp")
    panel.apply_correction = tk.BooleanVar(interp, False)
    panel.show_correction, panel.show_estimate = Mock(), Mock()
    monkeypatch.setattr(panel_module.scan_correction, "load_reference_set", lambda: {"relative_model": "design_layout"})
    fit = Mock(return_value={"anchor_pass": 1, "source_scan": "/captured_scan", "report": ["checked"]})
    monkeypatch.setattr(panel_module.scan_correction, "compute_reference_set_correction", fit)
    legacy = Mock()
    monkeypatch.setattr(panel_module.scan_correction, "compute_correction", legacy)
    panel.compute_correction()
    fit.assert_called_once_with()
    legacy.assert_not_called()
    assert panel.correction_anchor.get() == "1" and panel.apply_correction.get()
    assert panel.relative_model.get() == "design_layout"
    assert panel.estimate_scan_dir == "/captured_scan"
    panel.show_estimate.assert_called_once_with("/captured_scan")


def local_cycle_panel(monkeypatch, tmp_path):
    panel = capture_panel(monkeypatch, tmp_path)
    interp = panel.frame._root
    paths = {}
    for name in ("local_start", "local_end"):
        paths[name] = tmp_path / f"{name}.yaml"
        paths[name].write_text("scan teaching")
    panel.scan_point_path = lambda name: paths[name]
    monkeypatch.setattr(panel_module, "load_scan_pose", lambda path: str(path))
    monkeypatch.setattr(panel_module, "check_scan_line", Mock())
    panel.scan_speed_mm_s = tk.StringVar(interp, "30")
    panel.host.velocity_percent = tk.StringVar(interp, "30")
    panel.relative_model = tk.StringVar(interp, "hongik_tcp")
    panel.apply_correction = tk.BooleanVar(interp, False)
    panel.estimate_job = tk.StringVar(interp, "")
    panel.weld_direction = tk.StringVar(interp, "E->S")
    panel.estimator = tk.StringVar(interp, "kiro")
    panel.show_correction = Mock()
    return panel


def test_scan_four_passes_automatically_applies_saved_truth_and_freezes_it(monkeypatch, tmp_path):
    panel = local_cycle_panel(monkeypatch, tmp_path)
    saved = {"reference_set": {"relative_model": "measured_layout"}, "seam_frame": {"n": [0, 0, 1]}}
    monkeypatch.setattr(panel_module.scan_correction, "load_correction", lambda: saved)
    validate = Mock()
    monkeypatch.setattr(panel_module.scan_correction, "check_reference_set_applied", validate)
    scan, raw = panel.local_cycle_scan_config()
    validate.assert_called_once_with(saved)
    assert scan["apply_correction"] is True and scan["relative"] == "measured_layout"
    assert panel.apply_correction.get() is True and panel.relative_model.get() == "measured_layout"
    panel.show_correction.assert_called_once_with()
    assert "apply_correction" not in raw and "relative" not in raw   # post-weld scans remain raw recording
    panel.apply_correction.set(False)
    panel.relative_model.set("hongik_tcp")
    assert scan["apply_correction"] is True and scan["relative"] == "measured_layout"


@pytest.mark.parametrize("saved", [None, {"reference_set": {"relative_model": "measured_layout"}}],
                         ids=["no-correction", "pending-truth"])
def test_scan_four_passes_never_silently_builds_without_ready_truth(monkeypatch, tmp_path, saved):
    panel = local_cycle_panel(monkeypatch, tmp_path)
    monkeypatch.setattr(panel_module.scan_correction, "load_correction", lambda: saved)
    if saved is not None:
        monkeypatch.setattr(panel_module.scan_correction, "check_reference_set_applied",
                            Mock(side_effect=ValueError("truth captures pending")))
    with pytest.raises(ValueError, match="No saved scan correction|truth captures pending"):
        panel.local_cycle_scan_config()
    panel.show_correction.assert_not_called()
    assert panel.apply_correction.get() is False
