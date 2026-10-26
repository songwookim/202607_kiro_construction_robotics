"""Direct 1G scan: motion steps, recording and orchestration with fake robot / sensor (no hardware)."""

import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import Pose, Transform
import numpy as np
import pytest
from rclpy.time import Time

from construct_robot.application import direct_scan
from construct_robot.application.direct_scan import DirectScanController
from construct_robot.core.scan_motion import ScanPose, build_scan_steps, check_scan_line
from construct_robot.nodes import scan_recorder
from construct_robot.nodes.scan_recorder import ScanRecorder, transform_matrix, world_points

NAMES = tuple(f"right_manipulator_joint{i}" for i in range(1, 7))


def pose(x, y, z, q=(0.0, 1.0, 0.0, 0.0)):
    p = Pose()
    p.position.x, p.position.y, p.position.z = x, y, z
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p


def scan_pose(x, y, z, q=(0.0, 1.0, 0.0, 0.0), joint=0.1):
    return ScanPose(NAMES, (joint,) * 6, pose(x, y, z, q))


START, END = scan_pose(0.06, -1.54, 0.90), scan_pose(0.06, -1.72, 0.90, joint=0.2)


def build(**extra):
    kwargs = dict(clearance_m=0.05, tcp_speed_m_s=0.01, velocity_scale=0.3)
    kwargs.update(extra)
    return build_scan_steps(pose(0.3, -1.6, 0.7), START, END, **kwargs)


def xyz(p):
    return (p.position.x, p.position.y, p.position.z)


def test_steps_lift_approach_record_line_and_retract():
    steps = build()
    assert [s.name for s in steps] == ["lift", "approach start", "line", "retract"]
    assert [s.record for s in steps] == [False, False, True, False]
    lift = steps[0].step
    assert xyz(lift["points"][0]) == pytest.approx((0.3, -1.6, 0.7))
    assert xyz(lift["points"][-1]) == pytest.approx((0.3, -1.6, 0.75))
    assert lift["tcp_speed_m_s"] == 0.0 and not lift["linear_motion_profile"]
    approach = steps[1].step
    assert approach["type"] == "named_pose" and approach["use_joint_planning"]
    assert approach["positions"] == (0.1,) * 6 and approach["joint_names"] == NAMES
    line = steps[2].step
    assert line["tcp_speed_m_s"] == 0.01 and line["linear_motion_profile"] and not line["touch_guard"]
    assert xyz(line["points"][0]) == pytest.approx(xyz(START.tcp))
    assert xyz(line["points"][-1]) == pytest.approx(xyz(END.tcp))
    retract = steps[3].step
    assert xyz(retract["points"][-1]) == pytest.approx((0.06, -1.72, 0.95))


def test_line_keeps_the_start_orientation():
    tilted_end = scan_pose(0.06, -1.72, 0.90, q=(0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)))
    steps = build_scan_steps(pose(0.3, -1.6, 0.7), START, tilted_end,
                             clearance_m=0.0, tcp_speed_m_s=0.01, velocity_scale=0.3)
    assert [s.name for s in steps] == ["approach start", "line"]
    for point in steps[1].step["points"]:
        assert (point.orientation.x, point.orientation.y, point.orientation.z, point.orientation.w) == \
            pytest.approx((0.0, 1.0, 0.0, 0.0))
    assert check_scan_line(START, tilted_end)[2] == pytest.approx(90.0)


@pytest.mark.parametrize("end, message", [
    (scan_pose(0.06, -1.54, 0.80), "from horizontal"),
    (scan_pose(0.06, -1.541, 0.90), "apart"),
])
def test_bad_scan_lines_are_refused(end, message):
    with pytest.raises(ValueError, match=message):
        check_scan_line(START, end)


@pytest.mark.parametrize("extra", [{"clearance_m": 0.5}, {"tcp_speed_m_s": 0.0}, {"tcp_speed_m_s": 1.0}])
def test_bad_scan_settings_are_refused(extra):
    with pytest.raises(ValueError):
        build(**extra)


def identity_tf(z=0.5):
    t = Transform()
    t.translation.z = z
    t.rotation.w = 1.0
    return t


def test_world_points_use_the_laser_plane_and_the_frame_pose():
    T = transform_matrix(identity_tf(0.5))
    points = world_points([10.0, 20.0], [300.0, 400.0], np.array([True, False]), T)
    np.testing.assert_allclose(points, [[0.01, 0.0, 0.8]])


def fake_recorder_node(fail_stamps=()):
    def lookup(_target, _source, time, timeout=None):
        if time.nanoseconds in fail_stamps:
            raise RuntimeError("no TF")
        return SimpleNamespace(transform=identity_tf())
    return SimpleNamespace(create_publisher=Mock(return_value=SimpleNamespace(publish=Mock())),
                           tf_buffer=SimpleNamespace(lookup_transform=lookup),
                           get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=1)))


def valid(x, z):
    return np.isfinite(z) & (z > 232.0)


def test_recorder_pairs_frames_with_tf_and_saves(tmp_path):
    recorder = ScanRecorder(fake_recorder_node(fail_stamps={2}), valid, out_root=str(tmp_path))
    recorder.add(SimpleNamespace(x=[0.0], z=[300.0]), 0)        # not recording yet: ignored
    recorder.start()
    for stamp in (1, 2, 3):
        recorder.add(SimpleNamespace(x=[0.0, 5.0], z=[300.0, 0.0]), stamp)
    stats = recorder.stop()
    assert stats == {"frames": 2, "points": 2, "tf_failures": 1, "unresolved_frames": 0}
    np.testing.assert_allclose(recorder.all_points(), [[0.0, 0.0, 0.8]] * 2)
    folder = recorder.save({"status": "OK"})
    data = np.load(f"{folder}/scan.npz")
    assert data["stamp_ns"].tolist() == [1, 3] and data["T_world_optical"].shape == (2, 4, 4)
    assert data["points_frame_index"].tolist() == [0, 1]
    meta = json.loads(open(f"{folder}/scan.json").read())
    assert meta["status"] == "OK" and meta["frames"] == 2 and meta["world_frame"] == "World"


def test_recorder_fails_fast_on_stamps_older_than_the_buffer():
    calls = []

    def lookup(_target, _source, time, timeout=None):
        calls.append((time.nanoseconds, timeout))
        if time.nanoseconds == 0:                     # latest available TF
            return SimpleNamespace(transform=identity_tf(), header=SimpleNamespace(stamp=Time(nanoseconds=10).to_msg()))
        if timeout is None:
            raise RuntimeError("extrapolation")
        return SimpleNamespace(transform=identity_tf())
    node = fake_recorder_node()
    node.tf_buffer = SimpleNamespace(lookup_transform=lookup)
    recorder = ScanRecorder(node, valid)
    with pytest.raises(RuntimeError):
        recorder._lookup(5)                          # older than the buffer: no waiting lookup
    assert all(timeout is None for _stamp, timeout in calls)
    calls.clear()
    recorder._lookup(20)                             # newer than the buffer: wait for it
    assert calls[-1][1] is not None


class FakeLive:
    def __init__(self, streaming=False):
        self.streaming, self.calls = streaming, []

    def stop(self):
        self.calls.append("stop")
        self.streaming = False


class FakeSession:
    def __init__(self, events, frames=5, status="OK"):
        self.events, self.frames, self.status = events, frames, status

    def __call__(self, sensor, meta):
        self.events.append(("session", meta["tcp_speed_m_s"]))
        return self

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")
        return {"status": self.status, "message": "", "folder": "/tmp/scan_folder" if self.frames else None,
                "stats": {"frames": self.frames, "points": 10 * self.frames, "tf_failures": 0},
                "sensor": {"driver_stats": {"queue_drops": 0}, "restore": "restored"}}


def controller(monkeypatch, *, live=None, fail_step=None, frames=5):
    poses = {"start": START, "end": END}
    monkeypatch.setattr(direct_scan, "load_scan_pose", lambda path: poses[path])
    live = live or FakeLive()
    order, events = [], []
    session = FakeSession(events, frames=frames)

    def run_motion(step, execute):
        assert execute
        name = step.get("path_kind", step.get("pose_label"))
        recording = bool(events) and events[-1] == "start"
        order.append((name, recording))
        return (name != fail_step), "failed" if name == fail_step else "ok"
    node = SimpleNamespace(scanner_live=live, precision_sensing_test=SimpleNamespace(busy=False),
                           _path_start_tcp_pose=lambda group: pose(0.3, -1.6, 0.7),
                           run_sequence_cartesian_motion=run_motion, run_sequence_named_pose=run_motion,
                           cancel_active_motion=Mock())
    host = SimpleNamespace(node=node, post=lambda callback, *args: callback(*args))
    return DirectScanController(host, session), live, order, events


def run(scan, **extra):
    finished = Mock()
    kwargs = dict(start_path="start", end_path="end", clearance_m=0.05, tcp_speed_m_s=0.01, velocity_scale=0.3,
                  sensor={"ip": "192.168.3.15", "port": 1096, "calib_file": "c.yaml", "gain": 0},
                  on_progress=Mock(), on_finished=finished)
    kwargs.update(extra)
    scan.run(**kwargs)
    return finished.call_args[0][0]


def test_direct_scan_records_only_the_line_in_the_recorder_process(monkeypatch):
    scan, live, order, events = controller(monkeypatch)
    result = run(scan)
    assert result["status"] == "OK" and result["folder"] == "/tmp/scan_folder"
    assert order == [("Scan lift", False), ("Scan start", False), ("Scan line", True), ("Scan retract", False)]
    assert events == [("session", 0.01), "start", "stop"]
    assert live.calls == [] and not scan.busy


def test_direct_scan_hands_the_sensor_from_the_live_view_to_the_recorder(monkeypatch):
    scan, live, _order, _events = controller(monkeypatch, live=FakeLive(streaming=True))
    assert run(scan)["status"] == "OK"
    assert live.calls == ["stop"]


def test_failure_during_the_line_stops_the_recorder_and_keeps_partial_data(monkeypatch):
    scan, _live, order, events = controller(monkeypatch, fail_step="Scan line")
    result = run(scan)
    assert result["status"] == "ERROR" and "partial data saved" in result["message"]
    assert len(order) == 3 and events[-1] == "stop" and result["folder"] == "/tmp/scan_folder"


def test_a_line_without_frames_is_an_error(monkeypatch):
    scan, _live, _order, _events = controller(monkeypatch, frames=0)
    result = run(scan)
    assert result["status"] == "ERROR" and "not recorded" in result["message"]


def test_failed_approach_never_records(monkeypatch):
    scan, _live, order, events = controller(monkeypatch, fail_step="Scan start")
    result = run(scan)
    assert result["status"] == "ERROR" and result["folder"] is None
    assert events == [] and order[-1] == ("Scan start", False)


def test_vertical_scan_poses_are_refused_before_any_motion(monkeypatch):
    scan, live, order, events = controller(monkeypatch)
    monkeypatch.setattr(direct_scan, "load_scan_pose",
                        lambda path: {"start": START, "end": scan_pose(0.06, -1.54, 0.80)}[path])
    result = run(scan)
    assert result["status"] == "ERROR" and "horizontal" in result["message"]
    assert order == [] and events == [] and live.calls == []


def test_cancel_stops_before_the_next_motion(monkeypatch):
    scan, _live, order, _events = controller(monkeypatch)
    original = scan.host.node.run_sequence_cartesian_motion

    def cancel_after_lift(step, execute):
        outcome = original(step, execute)
        scan.cancel()
        return outcome
    scan.host.node.run_sequence_cartesian_motion = cancel_after_lift
    result = run(scan)
    assert result["status"] == "ERROR" and "canceled" in result["message"]
    assert order == [("Scan lift", False)]
    scan.host.node.cancel_active_motion.assert_called()


def test_local_scan_button_skips_the_lift(monkeypatch, tmp_path):
    import tkinter as tk
    from construct_robot.gui import precision_sensing_test_panel as panel_module
    from test_precision_sensing_test import panel_state
    for which in ("start", "end", "local_start", "local_end"):
        (tmp_path / f"scan_{which}.yaml").write_text("x: 1\n")
    interp = tk.Tcl()
    panel = panel_state()
    panel.capturing, panel.scan_folder = False, tmp_path
    panel.direct_clearance = tk.StringVar(interp, "50")
    panel.direct_status = tk.StringVar(interp, "")
    panel.direct = SimpleNamespace(run=Mock())
    panel.scan_speed_mm_s = tk.StringVar(interp, "8")
    panel.host.velocity_percent = tk.DoubleVar(interp, 30.0)
    panel.host.scanner_connection_panel = SimpleNamespace(
        ip=tk.StringVar(interp, "192.168.3.15"), port=tk.StringVar(interp, "1096"),
        calib=tk.StringVar(interp, "c.yaml"), gain=tk.StringVar(interp, "0"),
        sensor_status=tk.StringVar(interp, ""), _stopped=Mock(), _set_busy=Mock())
    monkeypatch.setattr(panel_module.threading, "Thread",
                        lambda target, kwargs, **kw: SimpleNamespace(start=lambda: target(**kwargs)))
    for local, clearance in ((True, 0.0), (False, 0.05)):
        panel.direct.run.reset_mock()
        panel.start_direct_scan(local=local)
        sent = panel.direct.run.call_args.kwargs
        assert sent["clearance_m"] == pytest.approx(clearance)
        assert sent["tcp_speed_m_s"] == pytest.approx(0.008) and sent["velocity_scale"] == pytest.approx(0.3)
        expected = ("scan_local_start.yaml", "scan_local_end.yaml") if local else ("scan_start.yaml", "scan_end.yaml")
        assert (sent["start_path"].name, sent["end_path"].name) == expected


def test_estimate_summary_lists_ends_and_pass_start_goal():
    from construct_robot.application.direct_scan import estimate_summary
    text = estimate_summary({
        "status": "OK", "out_dir": "/scan/hongik_estimate",
        "ends": {"S": {"point_mm": [1, 2, 3], "sigma_mm": 0.05, "two_sided": True},
                 "E": {"point_m": [0.2, 0.0, 0.0], "sigma_m": None, "two_sided": False}},
        "passes": [{"k": 1, "role": "root", "start_World_m": [0.1, 0.2, 0.3], "goal_World_m": [0.4, 0.5, 0.6]}]})
    assert "end S: (1.0, 2.0, 3.0) mm · sigma 0.05 mm · two-sided" in text
    assert "end E: (200.0, 0.0, 0.0) mm · sigma unknown · NOT identified" in text
    assert "pass 1 (root): start TCP (100.0, 200.0, 300.0) → goal TCP (400.0, 500.0, 600.0) mm World" in text
    assert "kiro/pass_<k>.yaml" in text


def test_latest_scan_dir_needs_a_recording(tmp_path):
    from construct_robot.application.direct_scan import latest_scan_dir
    with pytest.raises(ValueError):
        latest_scan_dir(tmp_path)
    for name in ("20261014-140000_scan", "20261014-150000_scan"):
        (tmp_path / name).mkdir()
    (tmp_path / "20261014-140000_scan" / "scan.npz").write_bytes(b"x")
    assert latest_scan_dir(tmp_path).name == "20261014-140000_scan"


def test_hongik_env_preserves_ros_paths_and_appends_system_packages():
    from construct_robot.nodes.hongik_scan_estimate import hongik_env
    env = hongik_env({"PYTHONPATH": "/ws/install/lib"})
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONPATH"] == "/ws/install/lib:/usr/lib/python3/dist-packages"


def test_deferred_recorder_resolves_every_frame_at_stop(tmp_path):
    recorder = ScanRecorder(fake_recorder_node(fail_stamps={2}), valid, out_root=str(tmp_path), deferred=True,
                            settle_s=0.0)
    recorder.start()
    for stamp in (1, 2, 3):
        recorder.add(SimpleNamespace(x=[0.0], z=[300.0]), stamp)
    assert recorder.frames == []                          # nothing resolved while recording
    stats = recorder.stop()
    assert stats == {"frames": 2, "points": 2, "tf_failures": 1, "unresolved_frames": 0}
    assert [f[0] for f in recorder.frames] == [1, 3]


def test_return_pass_scans_back_to_the_start_and_records_both_lines():
    steps = build(return_pass=True)
    assert [s.name for s in steps] == ["lift", "approach start", "line", "return line", "retract"]
    assert [s.record for s in steps] == [False, False, True, True, False]
    back = steps[3].step
    assert xyz(back["points"][0]) == pytest.approx(xyz(END.tcp))
    assert xyz(back["points"][-1]) == pytest.approx(xyz(START.tcp))
    assert back["tcp_speed_m_s"] == 0.01 and back["linear_motion_profile"]
    assert xyz(steps[4].step["points"][-1]) == pytest.approx((0.06, -1.54, 0.95))


def test_one_recording_spans_the_forward_and_return_pass(monkeypatch):
    scan, _live, order, events = controller(monkeypatch)
    result = run(scan, return_pass=True)
    assert result["status"] == "OK"
    assert order == [("Scan lift", False), ("Scan start", False), ("Scan line", True),
                     ("Scan return line", True), ("Scan retract", False)]
    assert events == [("session", 0.01), "start", "stop"]


def test_cancel_between_passes_still_stops_the_recorder(monkeypatch):
    scan, _live, order, events = controller(monkeypatch)
    original = scan.host.node.run_sequence_cartesian_motion

    def cancel_after_forward(step, execute):
        outcome = original(step, execute)
        if step["path_kind"] == "Scan line":
            scan.cancel()
        return outcome
    scan.host.node.run_sequence_cartesian_motion = cancel_after_forward
    result = run(scan, return_pass=True)
    assert result["status"] == "ERROR" and "canceled" in result["message"]
    assert events[-1] == "stop" and result["folder"] == "/tmp/scan_folder"


def test_split_passes_drops_the_turnaround():
    from construct_robot.nodes.hongik_scan_estimate import split_passes
    t = np.arange(0.0, 50.0, 0.1)
    s = np.concatenate([np.linspace(0.0, 0.2, 200), np.full(100, 0.2), np.linspace(0.2, 0.0, 200)])
    passes = split_passes(t, s)
    assert [sign for _idx, sign in passes] == [1, -1]
    assert all(abs(s[idx[-1]] - s[idx[0]]) > 0.18 for idx, _sign in passes)
    assert not set(passes[0][0]) & set(range(205, 295))         # the standstill at the far end is no pass


def test_estimate_poses_read_the_kiro_pass_files(tmp_path):
    import yaml
    from construct_robot.application.direct_scan import estimate_poses
    kiro = tmp_path / "hongik_estimate" / "kiro"
    kiro.mkdir(parents=True)
    pose_doc = {"position_m": {"x": 0.1, "y": 0.2, "z": 0.3}, "orientation_xyzw": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0}}
    (kiro / "pass_1.yaml").write_text(yaml.safe_dump({"pass": 1, "current_start": pose_doc, "current_goal": pose_doc}))
    (kiro / "pass_1_info.yaml").write_text("x: 1\n")
    poses = estimate_poses(tmp_path)
    assert [label for label, _p in poses] == ["Pass 1 start", "Pass 1 goal"]
    assert xyz(poses[0][1]) == pytest.approx((0.1, 0.2, 0.3)) and poses[0][1].orientation.y == 1.0


def test_move_to_estimated_pose_plans_a_tcp_line_through_the_sequence_runner(monkeypatch):
    import tkinter as tk
    from test_precision_sensing_test import panel_state
    interp = tk.Tcl()
    panel = panel_state()
    panel.capturing = False
    panel.estimate_text = tk.StringVar(interp, "")
    panel.pose_choice = tk.StringVar(interp, "Pass 1 start")
    panel.estimated_poses = [("Pass 1 start", pose(0.06, -1.55, 0.58, q=(0.0, 0.0, 0.0, 1.0)))]
    panel.host.node._current_tcp_pose = lambda group: pose(0.06, -1.50, 0.70)
    panel.host.velocity_percent = tk.DoubleVar(interp, 20.0)
    panel.estimate_scan_dir = "/scan"
    panel.apply_correction = tk.BooleanVar(interp, False)
    panel.relative_model = tk.StringVar(interp, "hongik_tcp")
    from construct_robot.application import scan_correction
    calls = []

    def target(scan_dir, label, current, correction, relative, orientations=None):
        calls.append((scan_dir, label, correction, relative))
        assert orientations == {}
        moved = pose(0.06, -1.55, 0.58)
        moved.orientation = current.orientation          # no taught pass orientation: the current one is kept
        return moved, {"corrected": False, "aim_m": None, "pass_orientation": False}
    monkeypatch.setattr(scan_correction, "estimated_move_target", target)
    from construct_robot.application import scan_four_pass
    monkeypatch.setattr(scan_four_pass, "load_pass_orientations", lambda: {})
    panel.host.run_sequence = Mock()
    panel.host.log = Mock()
    panel.move_to_estimated_pose(False)
    run_all, execute = panel.host.run_sequence.call_args[0]
    step = panel.host.run_sequence.call_args.kwargs["steps_override"][0]
    assert run_all and not execute
    assert calls == [("/scan", "Pass 1 start", None, "hongik_tcp")]
    assert step["type"] == "motion" and step["velocity_scale"] == pytest.approx(0.2)
    assert step["touch_guard"] and not step["continue_after_touch"] and not step["allow_initial_touch_motion"]
    assert xyz(step["points"][0]) == pytest.approx((0.06, -1.50, 0.70))
    assert xyz(step["points"][-1]) == pytest.approx((0.06, -1.55, 0.58))
    for point in step["points"]:                          # target orientation ignored: the current one is kept
        assert (point.orientation.y, point.orientation.w) == pytest.approx((1.0, 0.0))
    panel.host.robot_connected = {"right": False}
    panel.host.run_sequence.reset_mock()
    panel.move_to_estimated_pose(True)                     # execute needs the right arm
    panel.host.run_sequence.assert_not_called()


