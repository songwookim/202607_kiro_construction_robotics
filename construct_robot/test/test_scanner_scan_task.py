"""Scanner live view (1. CONNECTION) and post-weld scan (2. TASK › Scan); no robot or sensor I/O."""

import math
import sys
import threading
import tkinter as tk
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from construct_robot.nodes import scanner_live
from construct_robot.nodes import precision_sensing_test as module
from test_precision_sensing_test import bridge, completed, panel_state


def valid_mask(x, z):
    return np.isfinite(x) & np.isfinite(z) & (z > 232.0) & (z < 546.0)


def test_profile_points_keep_valid_samples_in_metres_on_the_laser_plane():
    points, ratio = scanner_live.profile_points(
        [-10.0, 0.0, 5.0, np.nan], [300.0, 100.0, 400.0, 300.0], valid_mask)
    np.testing.assert_allclose(points, [[-0.01, 0.0, 0.3], [0.005, 0.0, 0.4]], rtol=1e-6)
    assert ratio == 0.5
    empty, ratio = scanner_live.profile_points([], [], valid_mask)
    assert empty.shape == (0, 3) and ratio == 0.0


def test_raw_profile_preserves_columns_and_original_mm_without_range_filtering():
    raw, quality = scanner_live.raw_profile_points(
        [-10.0, 0.0, 5.0, 10.0, np.nan], [300.0, 100.0, 0.0, np.nan, 400.0], valid_mask)
    assert len(raw) == 5
    np.testing.assert_array_equal(raw["column_index"], [0, 1, 2, 3, 4])
    np.testing.assert_allclose(raw["raw_z_mm"], [300.0, 100.0, 0.0, np.nan, 400.0], equal_nan=True)
    assert raw["x"][0] == pytest.approx(-0.010)
    assert raw["z"][1] == pytest.approx(0.100)  # outside the valid range, still drawable in raw view
    assert raw["sensor_valid"].tolist() == [1, 0, 0, 0, 0]
    assert np.isnan(raw["z"][2])  # zero-range sentinel must not appear as a real point at the sensor origin
    assert quality == {"columns": 5, "drawable": 2, "zero_range": 1, "nonfinite": 2}
    empty, quality = scanner_live.raw_profile_points([], [], valid_mask)
    assert len(empty) == 0 and quality["drawable"] == 0


def test_raw_profile_rejects_mismatched_or_multidimensional_arrays():
    for x, z in (([1.0], []), ([[1.0]], [[300.0]])):
        with pytest.raises(ValueError, match="equal-length"):
            scanner_live.raw_profile_points(x, z, valid_mask)


def test_raw_pointcloud_layout_roundtrips_invalid_values_and_column_numbers():
    from rclpy.time import Time
    from sensor_msgs.msg import PointField
    live = live_bridge()
    raw, _ = scanner_live.raw_profile_points([12.0, -3.0], [600.0, 0.0], valid_mask)
    stamp = Time(seconds=100).to_msg()
    message = live._cloud(raw, raw=True, stamp=stamp)
    assert message.header.frame_id == scanner_live.SENSOR_FRAME
    assert message.header.stamp == stamp
    assert (message.width, message.height, message.point_step, message.row_step) == (2, 1, 28, 56)
    assert not message.is_dense and not message.is_bigendian
    assert message.fields[-2].name == "column_index" and message.fields[-2].datatype == PointField.UINT32
    received = np.frombuffer(bytes(message.data), scanner_live.RAW_PROFILE_DTYPE)
    assert received["raw_z_mm"].tolist() == [600.0, 0.0]
    assert received["z"][0] == pytest.approx(0.6)
    assert np.isnan(received["z"][1])


def test_calib_transform_reads_laser_link_to_optical(tmp_path):
    path = tmp_path / "calib.yaml"
    path.write_text("parent_frame: laser_link\nxyz: [0.1, 0.2, 0.3]\nrpy: [0.0, 0.0, 1.5707963267948966]\n")
    xyz, q = scanner_live.calib_transform(path)
    assert xyz == (0.1, 0.2, 0.3)
    assert q == pytest.approx((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    path.write_text("parent_frame: flange\nxyz: [0, 0, 0]\nrpy: [0, 0, 0]\n")
    with pytest.raises(ValueError, match="laser_link"):
        scanner_live.calib_transform(path)


class FakeClient:
    instances = []
    server_values = {"LaserMode": 0}

    def __init__(self, ip, port, recv_timeout):
        self.calls = []
        self.values = FakeClient.server_values
        self.param_backup = dict(self.values)
        self.param_backup_text = "LaserMode=0 Min=0 Max=1 Step=1\n"
        self.connected = True
        FakeClient.instances.append(self)

    def connect(self):
        return self

    def __enter__(self):
        return self.connect()

    def __exit__(self, *args):
        self.close()

    def init_session(self, init, backup_dir):
        self.calls.append(("init", init["LaserMode"], init["GainVal"]))
        self.values.update(init)
        return {}

    def start(self):
        self.calls.append("start")

    def get_frame(self, timeout):
        raise TimeoutError

    def stop(self):
        self.calls.append("stop")

    def drain(self):
        return []

    def set_params(self, values, check):
        self.calls.append(("restore", values))
        self.values.update(values)
        return SimpleNamespace(error=0)

    def read_params(self):
        return "", dict(self.values)

    def close(self):
        self.calls.append("close")
        self.connected = False


def fake_driver(monkeypatch):
    params = SimpleNamespace(
        INIT_STANDARD={"LaserMode": 1, "GainVal": 250},
        parse_params_text_with_meta=lambda text: ({}, {}),
        writable_params=lambda backup, meta: dict(backup),
        diff_params=lambda wanted, actual: {k: v for k, v in wanted.items() if actual.get(k) != v},
    )
    FakeClient.instances = []
    FakeClient.server_values = {"LaserMode": 0}
    monkeypatch.setattr(scanner_live, "_driver", lambda: (params, FakeClient, valid_mask))


def live_bridge(sensing_busy=False):
    from rclpy.time import Time
    node = SimpleNamespace(create_publisher=Mock(return_value=SimpleNamespace(publish=Mock())),
                           precision_sensing_test=SimpleNamespace(busy=sensing_busy),
                           get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=100)),
                           ui=SimpleNamespace(post=Mock(), post_latest=Mock()))
    return scanner_live.ScannerLiveBridge(node)


def test_frame_stream_publishes_raw_and_valid_clouds_with_receive_timestamp(monkeypatch):
    from rclpy.time import Time
    raw_publisher, valid_publisher = SimpleNamespace(publish=Mock()), SimpleNamespace(publish=Mock())
    live = live_bridge()
    live.publisher, live.raw_publisher = valid_publisher, raw_publisher
    live.node.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=100))
    monkeypatch.setattr(scanner_live.time, "monotonic_ns", lambda: 2_000_000_000)
    record = SimpleNamespace(x=[10.0, 20.0, 30.0], z=[300.0, 100.0, 0.0], n_points=3,
                             t_rx_ns=1_950_000_000)
    def get_frame(timeout):
        live._stop.set()
        return record
    live._frames(SimpleNamespace(get_frame=get_frame), valid_mask, Mock(), Mock())
    raw = raw_publisher.publish.call_args[0][0]
    valid = valid_publisher.publish.call_args[0][0]
    assert raw.width == 3 and valid.width == 1
    assert raw.header.stamp == Time(nanoseconds=99_950_000_000).to_msg()
    assert valid.header.stamp == raw.header.stamp


def start_args(**extra):
    return dict(dict(ip="192.168.3.15", port=1096, calib_file="", gain=0, backup_dir="/tmp",
                     on_status=Mock(), on_stopped=Mock()), **extra)


def test_live_view_initialises_streams_and_restores_the_backup(monkeypatch):
    fake_driver(monkeypatch)
    live = live_bridge()
    live.start(**start_args(gain=200))
    client = FakeClient.instances[-1]
    assert live.active
    assert client.calls[:2] == [("init", 0, 200), "start"]
    report = live.stop()
    assert not live.active
    assert ("restore", {"LaserMode": 0}) in client.calls and client.calls[-1] == "close"
    assert "1 parameters restored" in report
    assert live.stop() == "Scanner live view is not running"
    assert live.publisher.publish.call_args[0][0].width == 0
    assert live.raw_publisher.publish.call_args[0][0].width == 0


def test_laser_command_alone_never_initialises_or_streams(monkeypatch):
    fake_driver(monkeypatch)
    live = live_bridge()
    assert "ON" in live.set_laser("192.168.3.15", 1096, True)
    client = FakeClient.instances[-1]
    assert client.calls == [("restore", {"LaserMode": 1}), "close"]
    assert not live.active and not live.streaming and live.laser_mode == 1
    live.set_laser("192.168.3.15", 1096, False)
    assert live.laser_mode == 0 and FakeClient.server_values["LaserMode"] == 0


def test_visualization_keeps_the_laser_state_set_before_start(monkeypatch):
    fake_driver(monkeypatch)
    live = live_bridge()
    live.set_laser("192.168.3.15", 1096, True)
    live.start(**start_args())
    client = FakeClient.instances[-1]
    assert client.calls[:2] == [("init", 1, 250), "start"]
    live.stop()
    assert FakeClient.server_values["LaserMode"] == 1
    assert live.laser_mode == 1 and not live.streaming
    live.shutdown()
    assert FakeClient.server_values["LaserMode"] == 0


def test_laser_toggle_uses_same_live_socket_and_is_preserved_on_visualization_stop(monkeypatch):
    fake_driver(monkeypatch)
    live = live_bridge()
    live.start(**start_args())
    client = FakeClient.instances[-1]
    assert live.laser_mode == 0
    live.set_laser("192.168.3.15", 1096, True)
    assert len(FakeClient.instances) == 1
    assert live.streaming and live.laser_mode == 1
    assert client.param_backup["LaserMode"] == 0  # immutable backup
    live.stop()
    assert FakeClient.server_values["LaserMode"] == 1
    live.shutdown()
    assert FakeClient.server_values["LaserMode"] == 0


def test_explicit_laser_off_is_not_undone_by_backup_restore(monkeypatch):
    fake_driver(monkeypatch)
    FakeClient.server_values["LaserMode"] = 1
    live = live_bridge()
    live.start(**start_args())
    live.set_laser("192.168.3.15", 1096, False)
    assert live.streaming
    live.stop()
    assert FakeClient.server_values["LaserMode"] == 0


def test_emergency_shutdown_turns_off_laser_and_releases_visualization(monkeypatch):
    fake_driver(monkeypatch)
    FakeClient.server_values["LaserMode"] = 1
    live = live_bridge()
    live.start(**start_args())
    live.shutdown()
    assert not live.active and live.laser_mode == 0
    assert FakeClient.server_values["LaserMode"] == 0


def test_laser_control_refuses_sensor_owned_by_scan_or_changed_ip(monkeypatch):
    fake_driver(monkeypatch)
    with pytest.raises(RuntimeError, match="owns the sensor"):
        live_bridge(sensing_busy=True).set_laser("192.168.3.15", 1096, True)
    assert not FakeClient.instances
    live = live_bridge()
    live.start(**start_args())
    with pytest.raises(RuntimeError, match="changing the sensor"):
        live.set_laser("192.168.3.16", 1096, True)
    live.stop()


def test_laser_readback_mismatch_is_reported_instead_of_claiming_success(monkeypatch):
    fake_driver(monkeypatch)
    monkeypatch.setattr(FakeClient, "set_params", lambda self, values, check: SimpleNamespace(error=0))
    live = live_bridge()
    with pytest.raises(RuntimeError, match="not confirmed"):
        live.set_laser("192.168.3.15", 1096, True)
    assert live.laser_mode == 0 and not live.active
    assert FakeClient.instances[-1].calls[-1] == "close"


def test_pending_sensor_read_excludes_scan_without_blocking_ui_ownership_check(monkeypatch):
    fake_driver(monkeypatch)
    live = live_bridge()
    entered, release = threading.Event(), threading.Event()
    def read_params(client):
        entered.set()
        assert release.wait(3.0)
        return "", {"LaserMode": 0}
    monkeypatch.setattr(FakeClient, "read_params", read_params)
    worker = threading.Thread(target=lambda: live.check("192.168.3.15", 1096))
    worker.start()
    try:
        assert entered.wait(1.0)
        assert live.active and not live.streaming
        b = bridge(monkeypatch)
        b.node.scanner_live = live
        with pytest.raises(RuntimeError, match="one client"):
            b.scan(a=(0, 0, 0), b=(1, 0, 0), frame="World", action_name="/a", status_service="/s",
                   log_topic="/l", on_progress=Mock(), on_result=Mock(), on_error=Mock(), on_finished=Mock())
    finally:
        release.set()
        worker.join(timeout=2.0)
    assert not live.active


def test_live_view_and_hongik_scan_exclude_each_other(monkeypatch):
    fake_driver(monkeypatch)
    with pytest.raises(RuntimeError, match="owns the sensor"):
        live_bridge(sensing_busy=True).start(**start_args())
    assert FakeClient.instances == []
    b = bridge(monkeypatch)
    b.node.scanner_live = SimpleNamespace(active=True)
    with pytest.raises(RuntimeError, match="one client"):
        b.scan(a=(0, 0, 0), b=(1, 0, 0), frame="World", action_name="/a", status_service="/s",
               log_topic="/l", on_progress=Mock(), on_result=Mock(), on_error=Mock(), on_finished=Mock())
    assert not b.busy


def test_server_status_parses_fields(monkeypatch):
    b = bridge(monkeypatch, "adapter=moveit_jtc base_frame=World running=False ready=True awaiting_confirm=False")
    success, _message, fields = b.server_status("/status")
    assert success and fields["base_frame"] == "World" and fields["ready"] == "True"
    b.node.destroy_client.assert_called_once()


def fake_post_weld(monkeypatch):
    package = ModuleType("hongik_scan_msgs")
    action = ModuleType("hongik_scan_msgs.action")
    action.PostWeldScan = SimpleNamespace(Goal=lambda: SimpleNamespace(
        run_dir="", after_pass=0, measure_speed_mps=0.0, confirm_mode="", dry_run=False, direction="", tool_orientation=""))
    package.action = action
    monkeypatch.setitem(sys.modules, "hongik_scan_msgs", package)
    monkeypatch.setitem(sys.modules, "hongik_scan_msgs.action", action)


def post_weld_args(**extra):
    return dict(dict(run_dir=" run_1 ", after_pass=2, measure_speed_mps=0.045, confirm_mode="each", dry_run=True,
                     action_name="/post", status_service="/status", log_topic="/log", on_progress=Mock(),
                     on_confirm=Mock(), on_result=Mock(), on_error=Mock(), on_finished=Mock()), **extra)


@pytest.mark.parametrize("direction", ["", "S->E", "E->S"])
@pytest.mark.parametrize("tool_orientation", ["pass", "least_rotation"])
def test_post_weld_scan_sends_goal_and_reports_confirmation(monkeypatch, direction, tool_orientation):
    fake_post_weld(monkeypatch)
    b = bridge(monkeypatch)
    monkeypatch.setattr(module, "message_to_ordereddict", lambda value: value)
    result = {"status": "DRY_RUN", "message": "planned", "out_dir": "", "run_dir": "run_1"}
    handle = SimpleNamespace(accepted=True, get_result_async=lambda: completed(SimpleNamespace(result=result)))
    sent = {}

    def send_goal_async(goal, feedback_callback):
        sent["goal"] = goal
        feedback_callback(SimpleNamespace(feedback=SimpleNamespace(
            phase="confirm", progress_text="CONFIRM move?", awaiting_confirmation=True)))
        return completed(handle)
    client = SimpleNamespace(wait_for_server=lambda **kw: True, send_goal_async=send_goal_async)
    monkeypatch.setattr(module, "ActionClient", lambda *args: client)
    b.node.ui.post_latest = lambda key, callback, *args: callback(*args)
    args = post_weld_args(direction=direction, tool_orientation=tool_orientation)
    b.post_weld_scan(**args)
    goal = sent["goal"]
    assert goal.direction == direction
    assert goal.tool_orientation == tool_orientation
    assert (goal.run_dir, goal.after_pass, goal.measure_speed_mps, goal.confirm_mode, goal.dry_run) == (
        "run_1", 2, 0.045, "each", True)
    assert [c.args[0] for c in args["on_confirm"].call_args_list] == [True, False]
    args["on_result"].assert_called_once_with(result)
    args["on_error"].assert_not_called()
    assert not b.busy and b.node._motion_action_calls == set()


def test_post_weld_scan_rejects_bad_input_before_contacting_the_server(monkeypatch):
    fake_post_weld(monkeypatch)
    b = bridge(monkeypatch)
    client = Mock()
    monkeypatch.setattr(module, "ActionClient", client)
    for extra in ({"confirm_mode": "none"}, {"measure_speed_mps": -1.0}, {"measure_speed_mps": "nan"},
                  {"direction": "bad"}, {"tool_orientation": "bad"}):
        args = post_weld_args(**extra)
        b.post_weld_scan(**args)
        args["on_error"].assert_called_once()
    client.assert_not_called()
    b.node.create_client.assert_not_called()


def test_confirm_move_sends_setbool(monkeypatch):
    b = bridge(monkeypatch)
    requests = []
    b.node.create_client = Mock(return_value=SimpleNamespace(
        wait_for_service=lambda **kw: True,
        call_async=lambda req: requests.append(req) or completed(SimpleNamespace(success=True, message="ok"))))
    assert b.confirm_move("/confirm", False) == (True, "ok")
    assert requests[0].data is False


def post_weld_panel(monkeypatch, *, dry_run, live_active=False, connected=True):
    panel = panel_state()
    interp = tk.Tcl()
    panel.host.robot_connected = {"right": connected}
    panel.host.node.scanner_live = SimpleNamespace(active=live_active)
    panel.host.node.precision_sensing_test.post_weld_scan = Mock()
    for field, value in (("post_weld_run", ""), ("post_weld_pass", "1"), ("post_weld_speed", "45"),
                         ("post_weld_confirm", "each"), ("post_weld_action", "/post"), ("post_weld_status", ""),
                         ("post_weld_direction", "Auto"), ("post_weld_orientation", "Sensing pass")):
        setattr(panel, field, tk.StringVar(interp, value))
    panel.post_weld_dry_run = tk.BooleanVar(interp, dry_run)
    monkeypatch.setattr("construct_robot.gui.precision_sensing_test_panel.threading.Thread",
                        lambda target, kwargs, **kw: SimpleNamespace(start=lambda: target(**kwargs)))
    return panel


def test_panel_dry_run_needs_no_robot_but_motion_does(monkeypatch):
    panel = post_weld_panel(monkeypatch, dry_run=True, connected=False)
    panel.start_post_weld()
    sent = panel.bridge.post_weld_scan.call_args.kwargs
    assert sent["measure_speed_mps"] == pytest.approx(0.045) and sent["dry_run"] is True
    panel = post_weld_panel(monkeypatch, dry_run=False, connected=False)
    panel.start_post_weld()
    panel.bridge.post_weld_scan.assert_not_called()
    assert "Right robot" in panel.post_weld_status.get()


def test_panel_blocks_scans_while_live_view_owns_the_sensor(monkeypatch):
    panel = post_weld_panel(monkeypatch, dry_run=True, live_active=True)
    panel.start_post_weld()
    panel.bridge.post_weld_scan.assert_not_called()
    assert "live view" in panel.post_weld_status.get()
    panel.start_scan()
    panel.bridge.scan.assert_not_called()
    assert "live view" in panel.show_error.call_args[0][0]


@pytest.mark.parametrize("choice,direction", [("Auto", ""), ("S->E", "S->E"), ("E->S", "E->S")])
def test_panel_forwards_selected_post_weld_direction(monkeypatch, choice, direction):
    panel = post_weld_panel(monkeypatch, dry_run=True)
    panel.post_weld_direction.set(choice)
    panel.start_post_weld()
    assert panel.bridge.post_weld_scan.call_args.kwargs["direction"] == direction
    assert panel.bridge.post_weld_scan.call_args.kwargs["tool_orientation"] == "pass"


def test_panel_forwards_least_rotation_orientation(monkeypatch):
    panel = post_weld_panel(monkeypatch, dry_run=True)
    panel.post_weld_orientation.set("Least rotation")
    panel.start_post_weld()
    assert panel.bridge.post_weld_scan.call_args.kwargs["tool_orientation"] == "least_rotation"


def test_old_post_weld_message_requires_gui_restart_before_submission(monkeypatch):
    fake_post_weld(monkeypatch)
    sys.modules["hongik_scan_msgs.action"].PostWeldScan.Goal = lambda: SimpleNamespace()
    b = bridge(monkeypatch)
    client = Mock()
    monkeypatch.setattr(module, "ActionClient", client)
    args = post_weld_args()
    b.post_weld_scan(**args)
    assert "restart the GUI" in args["on_error"].call_args.args[0]
    client.assert_not_called()
    b.node.create_client.assert_not_called()
