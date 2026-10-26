"""Task switching and persistence without robot or welder connections."""

import tkinter as tk
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from construct_robot.application import welding_tasks
from construct_robot.application.welding_tasks import WeldingTaskController
from construct_robot.core.multipass import MultiPassState
from construct_robot.core.sequence_model import SequenceModel
from construct_robot.core.task_teaching_model import TeachingState, TEACHING_POSES
from construct_robot.gui.weld_action_gui import WeldActionGui


@pytest.fixture
def task_host(tmp_path, monkeypatch):
    monkeypatch.setattr(welding_tasks, "teaching_config_dir", lambda: tmp_path / "config")
    monkeypatch.setattr(welding_tasks.Path, "home", lambda: tmp_path)
    interpreter = tk.Tcl()
    host = SimpleNamespace(
        welding_position="1G", teaching_state=TeachingState(TEACHING_POSES),
        _multipass_state=MultiPassState(), sequence_model=SequenceModel(),
        sequence_running=False, active_weld_feedback_session=None,
        keyboard_teaching_capture_in_progress=False, seam_auto_running=False,
        keyboard_velocity_switching=False, keyboard_wire_active_key=None,
        hicomm_inching_direction=None, robot_power_busy=False,
        points=[], show_path=tk.BooleanVar(interpreter, value=True),
        raw_two_touch_seam=[], corrected_two_touch_seam=[],
        four_pass_folder=tk.StringVar(interpreter, value="1g-passes"),
        weld_current_raw=tk.IntVar(interpreter, value=190),
        weave_amplitude_mm=tk.DoubleVar(interpreter, value=3),
        initial_plan_ready=True, execution_allowed=True,
        node=SimpleNamespace(active_motion_goal=None, publish_points=Mock(),
                             publish_seam_comparison=Mock()),
        _auto_load_teaching_states=Mock(), invalidate_approved_plan=Mock(),
        teaching_pose_changed=Mock(), set_points=Mock(), refresh_sequence_table=Mock(),
        _update_seam_yaw_status=Mock(), update_speed_label=Mock(),
        _update_motion_profile_label=Mock(), log=Mock(),
        weld_parameter_panel=SimpleNamespace(refresh=Mock()),
    )
    host._weld_feedback_directory = lambda: welding_tasks.task_feedback_directory(host.welding_position)
    host._latest_weld_feedback_path = lambda: host._weld_feedback_directory() / "latest_weld_feedback.log"
    host.welding_tasks = WeldingTaskController(host)
    return host


def test_paths_keep_existing_1g_files_and_separate_3g(task_host, tmp_path):
    assert welding_tasks.task_config_directory("1G") == tmp_path / "config"
    assert welding_tasks.task_config_directory("3G") == tmp_path / "config/welding_3g"
    assert task_host._weld_feedback_directory() == tmp_path / "ros2_ws/weld_feedback"
    task_host.welding_tasks.switch("3G")
    assert task_host._weld_feedback_directory() == tmp_path / "ros2_ws/weld_feedback/3g"
    assert task_host.weld_parameter_panel.latest_log_path == task_host._latest_weld_feedback_path()


def test_tasks_retain_independent_teaching_correction_sequence_and_settings(task_host):
    host = task_host
    host.teaching_state.store("weld_start", "1g-start")
    host._multipass_state.corrected[1] = {"start": "1g-corrected"}
    host.sequence_model.steps.append({"type": "sleep", "duration": 1})
    host.weld_current_raw.set(200)
    original = host.teaching_state
    host.welding_tasks.switch("3G")
    assert host.teaching_state is not original
    assert not any(host.teaching_state.poses.values())
    assert host._multipass_state.corrected == {}
    assert host.sequence_model.steps == []
    assert host.weld_current_raw.get() == 190
    assert not host.initial_plan_ready and host.execution_allowed
    host.invalidate_approved_plan.assert_called()
    host.node.publish_seam_comparison.assert_called_with([], [], True)
    host.teaching_state.store("weld_start", "3g-start")
    host._multipass_state.corrected[1] = {"start": "3g-corrected"}
    host.sequence_model.steps.append({"type": "sleep", "duration": 3})
    host.weld_current_raw.set(120)
    host.welding_tasks.switch("1G")
    assert host.teaching_state is original
    assert host.teaching_state.poses["weld_start"] == "1g-start"
    assert host._multipass_state.corrected[1]["start"] == "1g-corrected"
    assert host.sequence_model.steps[0]["duration"] == 1
    assert host.weld_current_raw.get() == 200
    host.welding_tasks.switch("3G")
    assert host.teaching_state.poses["weld_start"] == "3g-start"
    assert host._multipass_state.corrected[1]["start"] == "3g-corrected"
    assert host.sequence_model.steps[0]["duration"] == 3
    assert host.weld_current_raw.get() == 120


def test_settings_persist_and_reload_in_their_own_task(task_host):
    host = task_host
    host.welding_tasks.switch("3G")
    host.weld_current_raw.set(115)
    host.weave_amplitude_mm.set(1.5)
    path = host.welding_tasks.save_settings()
    document = yaml.safe_load(path.read_text())
    assert document["welding_position"] == "3G"
    assert document["values"]["weld_current_raw"] == 115
    host.welding_tasks.switch("1G")
    host.welding_tasks.states.pop("3G")
    host.welding_tasks.switch("3G")
    assert host.weld_current_raw.get() == 115
    assert host.weave_amplitude_mm.get() == 1.5
    assert "3g" in host.four_pass_folder.get()


@pytest.mark.parametrize("attribute,value", [
    ("sequence_running", True), ("active_weld_feedback_session", {}),
    ("keyboard_teaching_capture_in_progress", True), ("seam_auto_running", True),
    ("keyboard_velocity_switching", True), ("keyboard_wire_active_key", "f"),
    ("hicomm_inching_direction", "forward"), ("robot_power_busy", True),
    ("multi_pass_registration", {"phase": "waiting_start"}),
])
def test_active_operations_prevent_switching(task_host, attribute, value):
    setattr(task_host, attribute, value)
    with pytest.raises(ValueError, match="Stop the active operation"):
        task_host.welding_tasks.switch("3G")
    assert task_host.welding_position == "1G"
    task_host._auto_load_teaching_states.assert_not_called()


def test_active_motion_and_scan_prevent_switching(task_host):
    task_host.node.active_motion_goal = object()
    with pytest.raises(ValueError, match="Stop the active operation"):
        task_host.welding_tasks.switch("3G")
    task_host.node.active_motion_goal = None
    task_host.precision_sensing_test_panel = SimpleNamespace(busy=True)
    with pytest.raises(ValueError, match="Stop the active operation"):
        task_host.welding_tasks.switch("3G")


def test_invalid_task_settings_restore_previous_working_state(task_host):
    path = welding_tasks.task_config_directory("3G") / "task_settings.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("format_version: 1\nwelding_position: 1G\nvalues: {}\n")
    task_host.teaching_state.store("weld_start", "1g-start")
    with pytest.raises(ValueError, match="Invalid 3G"):
        task_host.welding_tasks.switch("3G")
    assert task_host.welding_position == "1G"
    assert task_host.teaching_state.poses["weld_start"] == "1g-start"


def test_gui_teaching_and_touch_paths_follow_task(task_host):
    host = SimpleNamespace(welding_position="3G", planning_group=SimpleNamespace(get=lambda: "right_manipulator"))
    for method in (WeldActionGui._initial_state_yaml_path,
                   WeldActionGui._seam_reference_yaml_path, WeldActionGui._seam_touch_yaml_path):
        arguments = {"pose_name": "weld_start"} if method == WeldActionGui._initial_state_yaml_path else {}
        path = method(host, **arguments)
        assert path.parent == welding_tasks.task_config_directory("3G")
        host.welding_position = "1G"
        assert method(host, **arguments).parent == welding_tasks.task_config_directory("1G")
        host.welding_position = "3G"


def test_3g_does_not_build_the_1g_scan_cycle():
    host = SimpleNamespace(welding_position="3G", error=Mock())
    assert WeldActionGui.build_scan_weld_cycle(host) is False
    assert "1G sensing workflow" in host.error.call_args[0][0]


def test_shared_welding_controls_mount_in_each_task_page():
    from tkinter import ttk

    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display unavailable: {error}")
    root.withdraw()
    try:
        notebook = ttk.Notebook(root)
        host = SimpleNamespace(_operator_page_canvases={})
        pages = [WeldActionGui._create_operator_scroll_page(host, notebook, title)
                 for title in ("Welding · 1G", "Torch Cleaner", "Scan", "Welding · 3G")]
        controls = ttk.Frame(notebook)
        for page in (pages[0], pages[3], pages[0]):
            controls.pack_forget()
            controls.pack(in_=page, fill=tk.BOTH, expand=True)
            assert str(controls.pack_info()["in"]) == str(page)
        assert tuple(notebook.tab(tab, "text") for tab in notebook.tabs()) == (
            "Welding · 1G", "Torch Cleaner", "Scan", "Welding · 3G")
    finally:
        root.destroy()


def test_late_ik_result_cannot_save_into_another_task(tmp_path, monkeypatch):
    import construct_robot.gui.weld_action_gui as gui_module

    save_pose = Mock()
    monkeypatch.setattr(gui_module, "save_initial_state_yaml", save_pose)
    host = SimpleNamespace(welding_position="3G", log=Mock(), error=Mock())
    WeldActionGui.apply_corrected_tcp_joint_state(
        host, "start", "weld_start", "right_manipulator", (), (), None,
        task_position="1G")
    save_pose.assert_not_called()
    host.log.assert_called_once()
    WeldActionGui.corrected_tcp_joint_state_failed(
        host, "start", "weld_start", "old failure", task_position="1G")
    host.error.assert_not_called()


def test_save_from_cleaner_keeps_welding_speed_scale(task_host):
    host = task_host
    host._active_teaching_task = "cleaner"
    host._teaching_task_settings = {"welding": ("XY", "World", 5, 3, 25)}
    path = host.welding_tasks.save_settings()
    assert yaml.safe_load(path.read_text())["values"]["velocity_percent"] == 25
