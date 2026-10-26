"""ARC runtime and weld-feedback recording without Tk (weld_execution_controller)."""

import ast
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import Pose

from construct_robot.application import weld_execution_controller as controller_module
from construct_robot.application.weld_execution_controller import (
    WeldExecutionController,
    WeldFeedbackRecorder,
    weld_status_snapshot,
)
from construct_robot.core.weld_config import (
    DEFAULT_DIGITAL_WELD_SETTINGS,
    validate_digital_weld_settings,
)

STATUS = {"output_state_name": "main_weld", "feedback_current_a": 198,
          "feedback_voltage_v": 24.5, "wire_feed_m_min": 8.0, "sequence_stage": "done"}


def settings(**changes):
    values = dict(DEFAULT_DIGITAL_WELD_SETTINGS)
    values.update(custom_hot_start_enabled=False, software_crater_enabled=False)
    values.update(changes)
    return validate_digital_weld_settings(values)


def pose(x):
    result = Pose()
    result.position.x, result.position.y, result.position.z = x, 0.25, 0.10
    result.orientation.w = 1.0
    return result


def make_host(tmp_path=None, fake=False, **overrides):
    """Plain host double: shared ARC/feedback state plus a recording client."""
    calls = []
    client = SimpleNamespace(
        connected=True,
        arc_set=lambda **kw: calls.append(("arc_set",)),
        arc_on=lambda **kw: calls.append(("arc_on", kw)) or dict(STATUS),
        arc_off=lambda **kw: calls.append(("arc_off", kw)) or dict(STATUS, output_state_name="idle"),
        set_arc=lambda value: calls.append(("set_arc", value)),
        update_setpoints=lambda current, voltage: calls.append(("update_setpoints", current, voltage)),
        latest_status=lambda: dict(STATUS),
        comm_alive=lambda: True,
        inhibit_outputs=lambda: calls.append(("inhibit_outputs",)),
    )
    host = SimpleNamespace(
        calls=calls,
        hicomm_client=client,
        sequence_stop_requested=False,
        weld_feedback_lock=threading.Lock(),
        active_weld_feedback_session=None,
        _weld_feedback_stopped=False,
        weld_arc_established_event=threading.Event(),
        weld_arc_on_done_event=threading.Event(),
        weld_arc_on_success=False,
        weld_motion_done_event=threading.Event(),
        weld_motion_success=False,
        post=lambda callback, *args: callback(*args),
        log=lambda message: calls.append(("log", message)),
        error=lambda message: calls.append(("error", message)),
        _teaching_snapshot_document=lambda: {},
        _touch_snapshot_document=lambda: {},
        _weld_feedback_directory=lambda: Path(tmp_path or "/nonexistent"),
        _weld_status_snapshot=weld_status_snapshot,
        node=SimpleNamespace(_current_tcp_pose=Mock(return_value=pose(0.5))),
    )
    for name, value in overrides.items():
        setattr(host, name, value)
    controller = WeldExecutionController(host, fake_arc=lambda: fake)
    recorder = WeldFeedbackRecorder(host, tcp_sample_period_s=0.01)
    # The controller calls its sibling operations through the host.
    host._execute_fake_arc = controller.execute_fake_arc
    host._execute_hicomm_weld = controller.execute_hicomm_weld
    host._software_crater_restore = controller.software_crater_restore
    host._begin_weld_feedback_record = recorder.begin
    host._finish_weld_feedback_record = recorder.finish
    host._mark_arc_off_control = recorder.mark_arc_off_control
    host._software_crater_record = recorder.record_software_crater
    host._custom_hot_start_record = recorder.record_custom_hot_start
    return host, controller, recorder


def test_background_finish_detaches_before_saving_and_freezes_end_time():
    import time
    host, _controller, recorder = make_host()
    entered, release = threading.Event(), threading.Event()
    saved = []

    def slow_save(session, result, final_status, ended, ended_monotonic):
        entered.set()
        assert release.wait(2)
        saved.append((session, ended, ended_monotonic))
        return "saved.log"

    recorder._finish_session = slow_save
    recorder.begin(settings())
    session = host.active_weld_feedback_session
    try:
        future = recorder.finish("operator stop", background=True)
        detached_at = time.monotonic()
        assert host.active_weld_feedback_session is None
        assert entered.wait(1)
        assert not future.done()
        # A new session can collect feedback while the older one is saving.
        recorder.begin(settings())
        assert host.active_weld_feedback_session is not session
        release.set()
        assert future.result(timeout=1) == "saved.log"
        assert saved[0][0] is session
        assert saved[0][2] <= detached_at
    finally:
        release.set()
        recorder.shutdown()


def test_controller_imports_no_tkinter_or_gui():
    tree = ast.parse(Path(controller_module.__file__).read_text(encoding="utf-8"))
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


def test_fake_arc_on_sets_the_same_handshake_events_without_the_welder():
    host, controller, _recorder = make_host(fake=True)
    assert controller.execute_hicomm_weld("on", settings())[0]
    assert host.weld_arc_on_success
    assert host.weld_arc_established_event.is_set() and host.weld_arc_on_done_event.is_set()
    assert controller.execute_hicomm_weld("off", settings()) == (
        True, "FAKE ARC OFF · no command sent to welder",
    )
    assert host.calls == []


def test_real_arc_on_orders_recipe_record_handshake_and_events():
    host, controller, _recorder = make_host()
    success, message = controller.execute_hicomm_weld("on", settings(), {"mode": "sequence_execute"})
    assert success and message.startswith("ARC established")
    assert [c[0] for c in host.calls] == ["arc_set", "arc_on", "log"]
    assert host.calls[1][1] == {"wait_recognition": True, "wait_welding": True,
                                "wait_established": True, "timeout": 5.0}
    session = host.active_weld_feedback_session
    assert session["execution_conditions"]["mode"] == "sequence_execute"
    assert "arc_established_elapsed_s" in session["arc_off_control"]
    assert host.weld_arc_on_success
    assert host.weld_arc_established_event.is_set() and host.weld_arc_on_done_event.is_set()


def test_arc_on_failure_releases_waiters_without_establishment():
    host, controller, _recorder = make_host()
    host.hicomm_client.arc_on = Mock(side_effect=TimeoutError("not established"))
    assert controller.execute_hicomm_weld("on", settings()) == (False, "not established")
    assert not host.weld_arc_on_success
    assert host.weld_arc_on_done_event.is_set()
    assert not host.weld_arc_established_event.is_set()
    assert ("set_arc", False) not in host.calls  # ON timeout never clears ARC by itself


def test_disconnected_arc_on_fails_closed():
    host, controller, _recorder = make_host(hicomm_client=None)
    assert controller.execute_hicomm_weld("on", settings()) == (False, "Hi-COMM disconnected")
    assert host.weld_arc_on_done_event.is_set() and not host.weld_arc_on_success


def test_arc_off_without_finalize_keeps_session_and_pending_status():
    host, controller, recorder = make_host()
    recorder.begin(settings())
    success, _message = controller.execute_hicomm_weld("off", settings(), finalize_feedback=False)
    assert success
    assert ("arc_off", {"timeout": 5.0, "wait_idle": True, "wait_sequence_clear": True}) in host.calls
    session = host.active_weld_feedback_session
    assert session["pending_final_status"]["output_state_name"] == "idle"
    assert "command_elapsed_s" in session["arc_off_control"]
    assert "sequence_clear_elapsed_s" in session["arc_off_control"]


def test_arc_off_failure_clears_arc_and_finalizes_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(controller_module.subprocess, "Popen", Mock(return_value=Mock(wait=Mock(return_value=0))))
    host, controller, recorder = make_host(tmp_path)
    recorder.begin(settings())
    host.hicomm_client.arc_off = Mock(side_effect=TimeoutError("not clear"))
    assert controller.execute_hicomm_weld("off", settings()) == (False, "not clear")
    assert ("set_arc", False) in host.calls
    assert host.active_weld_feedback_session is None
    log = (tmp_path / "latest_weld_feedback.log").read_text(encoding="utf-8")
    assert "result=ARC OFF failed: not clear" in log


def test_triggered_arc_off_waits_for_establishment_and_fails_closed():
    host, controller, recorder = make_host()
    recorder.begin(settings())
    host.weld_arc_on_done_event.set()  # ARC ON finished but failed
    step = {"usable_seam_start": pose(0.5), "usable_seam_goal": pose(0.7), "settings": settings()}
    assert controller.execute_triggered_arc_off(step) == (
        False, "ARC OFF watcher aborted: ARC ON failed before establishment",
    )
    assert host.active_weld_feedback_session["arc_off_control"]["fallback"] == (
        "arc_on_failed_before_establishment"
    )
    assert not any(call[0] == "arc_off" for call in host.calls)


def test_triggered_arc_off_uses_projected_setpoint_lead_before_goal():
    host, controller, recorder = make_host()
    recorder.begin(settings())
    host.weld_arc_established_event.set()
    host.node._current_tcp_pose = Mock(side_effect=[pose(0.49 + 0.02 * i) for i in range(13)])
    step = {"usable_seam_start": pose(0.5), "usable_seam_goal": pose(0.7),
            "arc_off_delay_s": 0.12, "tcp_speed_m_s": 0.006,
            "path_to_seam_speed_factor": 1.0, "settings": settings()}
    success, message = controller.execute_triggered_arc_off(step)
    assert success and message.startswith("pre-GOAL ARC OFF · lead=0.72 mm")
    control = host.active_weld_feedback_session["arc_off_control"]
    assert control["speed_source"] == "tcp_setpoint_projected_along_seam"
    assert abs(control["calculated_pre_off_distance_m"] - 0.00072) < 1e-12
    # Lead-out motion continues: the session is kept for the executor to finish.
    assert host.active_weld_feedback_session["pending_final_status"] is not None


def test_fake_triggered_arc_off_waits_for_motion_completion():
    host, controller, _recorder = make_host(fake=True)
    host.weld_motion_done_event.set()
    host.weld_motion_success = True
    assert controller.execute_triggered_arc_off({}) == (
        True, "FAKE ARC OFF · no command sent to welder · weld motion completed",
    )


def test_software_crater_is_skipped_in_fake_mode_and_fails_closed_without_client():
    host, controller, recorder = make_host(fake=True)
    assert controller.execute_software_crater({"settings": settings(software_crater_enabled=True)})[0]
    host, controller, recorder = make_host(hicomm_client=None)
    recorder.begin(settings(software_crater_enabled=True))
    success, message = controller.execute_software_crater(
        {"settings": settings(software_crater_enabled=True)}
    )
    assert not success and "Hi-COMM feedback unavailable" in message
    assert host.active_weld_feedback_session["software_crater_control"]["status"] == "FAILED"


def test_custom_hot_start_requires_established_arc():
    host, controller, recorder = make_host()
    recorder.begin(settings(custom_hot_start_enabled=True))
    success, message = controller.execute_custom_hot_start(
        {"settings": settings(custom_hot_start_enabled=True), "planning_group": "right_manipulator"}
    )
    assert (success, message) == (False, "Custom Hot Start blocked: ARC was not established")
    assert host.active_weld_feedback_session["custom_hot_start"]["status"] == "ARC_NOT_ESTABLISHED"


def test_feedback_session_respects_stop_and_saves_once(tmp_path, monkeypatch):
    popen = Mock(return_value=Mock(wait=Mock(return_value=0)))
    monkeypatch.setattr(controller_module.subprocess, "Popen", popen)
    host, _controller, recorder = make_host(tmp_path, _weld_feedback_stopped=True)
    recorder.begin(settings())
    assert host.active_weld_feedback_session is None
    host._weld_feedback_stopped = False
    recorder.begin(settings())
    first = host.active_weld_feedback_session
    recorder.begin(settings(current_a=150))  # an active session is never replaced
    assert host.active_weld_feedback_session is first
    recorder.mark_arc_off_control(fallback="x")
    recorder.mark_arc_off_control(extra=1)  # command time is stamped once
    assert set(first["arc_off_control"]) == {"fallback", "command_elapsed_s", "extra"}
    path = recorder.finish("completed", STATUS)
    assert path is not None and path.parent == tmp_path
    assert (tmp_path / "latest_weld_feedback.log").is_file()
    popen.assert_called_once()
    assert recorder.finish("again") is None


def test_feedback_uses_task_and_directory_from_session_start(tmp_path, monkeypatch):
    monkeypatch.setattr(controller_module.subprocess, "Popen", Mock(return_value=Mock(wait=Mock(return_value=0))))
    original = tmp_path / "3g"
    host, _controller, recorder = make_host(original, welding_position="3G")
    recorder.begin(settings())
    assert host.active_weld_feedback_session["execution_conditions"]["welding_position"] == "3G"
    host.welding_position = "1G"
    host._weld_feedback_directory = lambda: tmp_path / "1g"
    try:
        path = recorder.finish("completed", STATUS)
        assert path.parent == original
        latest = original / "latest_weld_feedback.log"
        assert "welding_position=3G" in latest.read_text()
        assert not (tmp_path / "1g/latest_weld_feedback.log").exists()
    finally:
        recorder.shutdown()


def test_measured_joint_callback_saves_both_arms_by_name_with_tcp_session_clock(tmp_path, monkeypatch):
    from construct_robot.gui.weld_action_gui import WeldActionGui, WeldGuiNode
    from construct_robot.io.weld_feedback_plot import parse_weld_trajectory_log
    monkeypatch.setattr(controller_module.subprocess, "Popen", Mock(return_value=Mock(wait=Mock(return_value=0))))
    host, _controller, recorder = make_host(tmp_path, fake=True)
    host._weld_feedback_recorder = recorder
    host.record_weld_joint_sample = lambda arm, snapshot: WeldActionGui.record_weld_joint_sample(host, arm, snapshot)
    node = SimpleNamespace(ui=host, latest_joint_positions={}, last_robot_feedback_at={},
                           last_measured_joints_at={}, robot_feedback_seen={},
                           measured_joint_snapshot_lock=threading.Lock(), measured_joint_snapshots={})
    names = [f"{arm}_manipulator_joint{i}" for i in (6, 3, 1, 5, 2, 4) for arm in ("left", "right")]
    message = SimpleNamespace(name=names, position=[float(n[-1]) for n in names],
                              header=SimpleNamespace(stamp=SimpleNamespace(sec=123, nanosec=500000000)))
    WeldGuiNode._joint_state(node, message)  # receiving joints alone never starts a feedback session
    assert host.active_weld_feedback_session is None
    recorder.begin(settings(), {"mode": "sequence_execute", "fake_arc": True})
    recorder.record_tcp_sample(pose(.5), phase="ACTUAL_TF", tf_stamp_s=123.5)
    WeldGuiNode._joint_state(node, message)
    session = host.active_weld_feedback_session
    assert len(session["joint_samples"]) == 2
    for sample in session["joint_samples"]:
        assert sample["positions_rad"] == [1., 2., 3., 4., 5., 6.]
        assert sample["joint_stamp_s"] == 123.5
        assert sample["elapsed_s"] >= 0
    # Missing / non-finite joints must not be replaced with the previous values.
    partial = SimpleNamespace(name=names[:1], position=[1.], header=message.header)
    WeldGuiNode._joint_state(node, partial)
    invalid = SimpleNamespace(name=names[1::2], position=[float("nan")] * 6, header=message.header)
    WeldGuiNode._joint_state(node, invalid)
    assert len(session["joint_samples"]) == 2
    path = recorder.finish("completed (fake ARC)")
    text = path.read_text()
    section = text.split("[joint_trajectory]", 1)[1]
    assert "source=/joint_states" in section and "position_unit=rad" in section
    rows = [line.split() for line in section.splitlines() if len(line.split()) == 9 and line.split()[2] in ("left", "right")]
    assert len(rows) == 2
    assert {row[2] for row in rows} == {"left", "right"}
    assert all([float(v) for v in row[3:]] == [1., 2., 3., 4., 5., 6.] for row in rows)
    assert len(parse_weld_trajectory_log(path)["actual"]) == 1  # existing TCP reader remains compatible
    WeldGuiNode._joint_state(node, message)
    assert len(session["joint_samples"]) == 2  # finished session stays immutable
    recorder.begin(settings())
    recorder.record_joint_sample("right", node.measured_joint_snapshots["right"])
    assert host.active_weld_feedback_session["joint_samples"] == []  # no pre-session cached sample


def plot_report(tmp_path, output, returncode=0):
    from construct_robot.application.weld_execution_controller import WeldFeedbackRecorder
    host = SimpleNamespace(calls=[])
    host.post = lambda fn, *args: fn(*args)
    host.log = lambda message: host.calls.append(("log", message))
    host.error = lambda message: host.calls.append(("error", message))
    recorder = object.__new__(WeldFeedbackRecorder)
    recorder.host = host
    history = tmp_path / "weld_feedback_20261002_172811_139.log"
    plot_log = history.with_suffix(".plot.log")
    plot_log.write_text(output, encoding="utf-8")
    recorder._report_feedback_plot(Mock(wait=Mock(return_value=returncode)), plot_log, history)
    return host.calls


def test_plot_report_logs_saved_pngs(tmp_path):
    calls = plot_report(tmp_path, f"{tmp_path}/weld_feedback_20261002_172811_139.png\n"
                                  f"{tmp_path}/latest_weld_feedback.png\n")
    assert calls[0][0] == "log" and "PNG SAVED" in calls[0][1]
    assert "latest_weld_feedback" not in calls[0][1]


def test_plot_report_errors_when_matplotlib_is_broken(tmp_path):
    output = ("AttributeError: _ARRAY_API not found\n"
              "SKIPPED weld_feedback_20261002_172811_139.log · feedback plot: matplotlib is unavailable\n")
    calls = plot_report(tmp_path, output, returncode=1)
    assert calls[0][0] == "error" and "PNG NOT saved" in calls[0][1]
    assert "matplotlib is unavailable" in calls[0][1]


def test_plot_report_missing_tcp_trajectory_is_only_a_note(tmp_path):
    calls = plot_report(tmp_path, f"{tmp_path}/weld_feedback_20261002_172811_139.png\n"
                        "SKIPPED weld_feedback_20261002_172811_139.log · 3D trajectory plot: no recorded actual TCP samples\n")
    assert calls[0][0] == "log" and "no recorded actual TCP samples" in calls[0][1]


def test_software_crater_confirms_against_the_first_crater_tx_not_the_latest_repeat():
    """The welder TX repeats the crater setpoint every cycle; samples arrive just before each repeat.
    Measuring from the latest repeat skipped every sample (20261019 16:02: timeout with a 95 A echo)."""
    import time
    host, controller, recorder = make_host()
    s = settings(software_crater_enabled=True, software_crater_ratio_percent=50.0,
                 software_crater_voltage_v=20.0, software_crater_hold_s=0.05)
    recorder.begin(s)
    session = host.active_weld_feedback_session
    target, voltage = round(s["current_a"] * 0.5), 200
    frame = bytearray(55)
    frame[0] = 0x01
    frame[3:5] = target.to_bytes(2, "little")
    frame[5:7] = voltage.to_bytes(2, "little")
    stop = threading.Event()

    def welder():
        started = False
        while not stop.wait(0.02):
            if not any(c[0] == "update_setpoints" for c in host.calls):
                continue
            now = time.monotonic() - session["started_monotonic"]
            with host.weld_feedback_lock:
                if started:                                   # sample first, then the repeated TX frame
                    session["samples"].append({"elapsed_s": now, "feedback_current_a": target + 2,
                                               "wcr_detected": True, "set_current_a": target, "set_voltage_v": 20.0})
                session["tx_frames"].append({"elapsed_s": now + 0.001, "raw_hex": bytes(frame).hex(" ")})
            started = True
    thread = threading.Thread(target=welder, daemon=True)
    thread.start()
    client = host.hicomm_client
    client.snapshot = lambda: SimpleNamespace(command=0x01)
    host.weld_motion_done_event.set()
    host.weld_motion_success = True
    endpoint = pose(0.5)
    host.node.wait_until_arm_stopped = lambda arm, timeout: True
    host.node._current_tcp_transform = lambda group: SimpleNamespace(
        transform=SimpleNamespace(translation=endpoint.position))
    try:
        ok, message = controller.execute_software_crater({"settings": s, "endpoint": endpoint})
    finally:
        stop.set()
        thread.join(1)
    assert ok, message
    assert "HOLD complete" in message
