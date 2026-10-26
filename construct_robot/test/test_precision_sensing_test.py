from concurrent.futures import Future
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import threading
import tkinter as tk

from geometry_msgs.msg import Pose, TransformStamped
from rclpy.time import Time
import pytest

from construct_robot.core.precision_sensing_inspection import SensingPose
from construct_robot.gui.precision_sensing_test_panel import PrecisionSensingTestPanel
from construct_robot.nodes import precision_sensing_test as module


def completed(value):
    future = Future()
    future.set_result(value)
    return future


def bridge(monkeypatch, status="base_frame=World running=False ready=True"):
    monkeypatch.setattr(module, "TransformBroadcaster", Mock())
    node = SimpleNamespace(
        create_publisher=Mock(return_value=SimpleNamespace(publish=Mock())),
        create_timer=Mock(), create_subscription=Mock(), destroy_subscription=Mock(), destroy_client=Mock(),
        active_motion_goal=None, _motion_action_lock=threading.Lock(), _motion_action_calls=set(),
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=100)),
        get_parameter=lambda name: SimpleNamespace(value=250.0),
        ui=SimpleNamespace(post=lambda callback, *args: callback(*args), post_latest=Mock()),
        create_client=Mock(return_value=SimpleNamespace(
            wait_for_service=lambda **kw: True,
            call_async=lambda req: completed(SimpleNamespace(success=True, message=status)))),
    )
    b = module.PrecisionSensingTestBridge(node)
    return b


def fake_interface(monkeypatch):
    package = ModuleType("welding_interfaces")
    action = ModuleType("welding_interfaces.action")
    action.PrecisionSensing = SimpleNamespace(Goal=lambda: SimpleNamespace(
        welding_position=0, pose_a=Pose(), pose_b=Pose()))
    package.action = action
    monkeypatch.setitem(__import__("sys").modules, "welding_interfaces", package)
    monkeypatch.setitem(__import__("sys").modules, "welding_interfaces.action", action)


def scan_args():
    return dict(a=(0.1, 0.2, 0.3), b=(0.1, 0.4, 0.3), frame="World", action_name="/scan",
                status_service="/status", log_topic="/log", on_progress=Mock(), on_result=Mock(),
                on_error=Mock(), on_finished=Mock())


def test_scan_checks_frame_and_readiness_before_submitting(monkeypatch):
    fake_interface(monkeypatch)
    client = Mock()
    monkeypatch.setattr(module, "ActionClient", client)
    for status in ("base_frame=other running=False ready=True", "base_frame=World running=False ready=False",
                   "base_frame=World running=True ready=True"):
        b = bridge(monkeypatch, status)
        args = scan_args()
        b.scan(**args)
        args["on_error"].assert_called_once()
        args["on_finished"].assert_called_once()
        assert not b.busy
    client.assert_not_called()


def test_cancel_during_preflight_never_sends_late_goal(monkeypatch):
    fake_interface(monkeypatch)
    b = bridge(monkeypatch)
    client = SimpleNamespace(wait_for_server=lambda **kw: (b.cancel() or True), send_goal_async=Mock())
    monkeypatch.setattr(module, "ActionClient", lambda *args: client)
    args = scan_args()
    b.scan(**args)
    client.send_goal_async.assert_not_called()
    assert "canceled before submission" in args["on_error"].call_args[0][0]
    assert b.node._motion_action_calls == set()


def test_sequence_stop_before_bridge_busy_never_sends_late_goal(monkeypatch):
    fake_interface(monkeypatch)
    b = bridge(monkeypatch)
    client = SimpleNamespace(wait_for_server=lambda **kw: True, send_goal_async=Mock())
    monkeypatch.setattr(module, "ActionClient", lambda *args: client)
    args = dict(scan_args(), cancel_requested=lambda: True)
    b.scan(**args)
    client.send_goal_async.assert_not_called()
    assert "canceled before submission" in args["on_error"].call_args[0][0]
    assert not b.busy


def test_scan_sends_only_precision_goal_and_delivers_failed_result(monkeypatch):
    fake_interface(monkeypatch)
    b = bridge(monkeypatch)
    result = {"error_code": 3, "error_message": "DATA_FAIL", "paths": [], "data_file_path": ""}
    monkeypatch.setattr(module, "message_to_ordereddict", lambda value: value)
    handle = SimpleNamespace(accepted=True, get_result_async=lambda: completed(SimpleNamespace(result=result)))
    client = SimpleNamespace(wait_for_server=lambda **kw: True,
                             send_goal_async=Mock(return_value=completed(handle)))
    monkeypatch.setattr(module, "ActionClient", lambda *args: client)
    args = scan_args()
    b.scan(**args)
    goal = client.send_goal_async.call_args[0][0]
    assert goal.welding_position == 1
    assert goal.pose_a.position.x == 0.1
    assert goal.pose_b.position.y == 0.4
    assert goal.pose_a.orientation.w == 1.0
    assert args["on_result"].call_args[0][0].poses == ()
    args["on_error"].assert_not_called()
    assert b.node.active_motion_goal is None


def test_sequence_scan_returns_result_without_waiting_for_tk_callbacks(monkeypatch):
    fake_interface(monkeypatch)
    b = bridge(monkeypatch)
    result = {"error_code": 3, "error_message": "DATA_FAIL", "paths": [], "data_file_path": ""}
    monkeypatch.setattr(module, "message_to_ordereddict", lambda value: value)
    b._send = Mock(return_value=result)
    args = dict(scan_args(), on_result=None, on_error=None, on_finished=None)
    inspection = b.scan(**args)
    assert inspection.document == result
    assert not b.busy
    b._send.side_effect = RuntimeError("sensor disconnected")
    with pytest.raises(RuntimeError, match="sensor disconnected"):
        b.scan(**args)
    assert not b.busy


def test_passive_rviz_preserves_frames_quaternions_and_clears(monkeypatch):
    b = bridge(monkeypatch)
    rows = tuple(SensingPose(1, str(i), "wire", "World", (i * 0.005, 0.0, 0.0), (0., 0., 0., 1.)) for i in range(2))
    tcp = SensingPose(1, "START", "tcp", "World", (0., 0., -0.0212), (0., 0., 0., 1.))
    inspection = SimpleNamespace(document={"paths": [None]}, poses=rows + (tcp,), frame="World")
    b.visualize(inspection, tcp)
    sent = b.tf.sendTransform.call_args[0][0]
    by_name = {msg.child_frame_id: msg for msg in sent}
    selected = by_name["hongik_test/selected_tcp"]
    assert selected.header.frame_id == "World"
    assert selected.transform.translation.z == -0.0212
    assert selected.transform.rotation.w == 1.0
    assert by_name["hongik_test/pass_1_wire_start"].transform.translation.z == 0.0
    line = b.markers.publish.call_args[0][0].markers[1]
    assert line.points[1].x == 0.005
    b.clear()
    assert b._transforms == ()
    assert b.markers.publish.call_args[0][0].markers[0].action == module.Marker.DELETEALL


def test_current_tcp_requires_fresh_robot_tf(monkeypatch):
    b = bridge(monkeypatch)
    transform = TransformStamped()
    transform.header.stamp = Time(seconds=100).to_msg()
    transform.transform.translation.x = 0.7
    b.node.tf_buffer = SimpleNamespace(lookup_transform=Mock(return_value=transform))
    assert b.current_tcp("World") == (0.7, 0.0, 0.0)
    transform.header.stamp = Time(seconds=98).to_msg()
    with pytest.raises(RuntimeError, match="stale"):
        b.current_tcp("World")


def panel_state():
    interp = tk.Tcl()
    panel = object.__new__(PrecisionSensingTestPanel)
    panel.host = SimpleNamespace(sequence_running=False, seam_auto_running=False,
        node=SimpleNamespace(active_motion_goal=None, precision_sensing_test=SimpleNamespace(scan=Mock())),
        keyboard_velocity_arm=None, keyboard_velocity_switching=False,
        keyboard_jog_enabled=tk.BooleanVar(interp, False), execution_allowed=True, robot_connected={"right": True})
    panel.busy = False
    panel.a = [tk.StringVar(interp, v) for v in ("100", "200", "300")]
    panel.b = [tk.StringVar(interp, v) for v in ("100", "400", "300")]
    for field, value in (("frame", "World"), ("action", "/scan"), ("status_service", "/status"), ("log_topic", "/log"), ("status", "")):
        setattr(panel, field, tk.StringVar(interp, value))
    panel.show_error = Mock()
    panel.clear = Mock()
    panel._set_busy = Mock()
    return panel


def test_gui_converts_mm_to_m_and_blocks_scan_during_teaching(monkeypatch):
    panel = panel_state()
    monkeypatch.setattr("construct_robot.gui.precision_sensing_test_panel.threading.Thread",
                        lambda target, kwargs, **kw: SimpleNamespace(start=lambda: target(**kwargs)))
    panel.start_scan()
    sent = panel.bridge.scan.call_args.kwargs
    assert sent["a"] == (0.1, 0.2, 0.3)
    assert sent["b"] == (0.1, 0.4, 0.3)
    panel.bridge.scan.reset_mock()
    panel.host.keyboard_jog_enabled.set(True)
    panel.start_scan()
    panel.bridge.scan.assert_not_called()
    assert "Disable Keyboard Teaching" in panel.show_error.call_args[0][0]


def test_gui_preview_state_does_not_mutate_production_paths(monkeypatch):
    panel = panel_state()
    panel.host.points = ["production path"]
    panel.host.four_pass_corrected = {1: "current corrected state"}
    panel.host.sequence_running = True
    panel.start_scan()
    panel.bridge.scan.assert_not_called()
    assert panel.host.points == ["production path"]
    assert panel.host.four_pass_corrected == {1: "current corrected state"}


def test_gui_comparison_uses_mm_and_selected_pose_convention():
    panel = panel_state()
    interp = tk.Tcl()
    panel.selected_pose = lambda: SensingPose(1, "0", "wire", "World", (0.1, 0.2, 0.3), (0., 0., 0., 1.))
    panel.expected = [tk.StringVar(interp, v) for v in ("103", "204", "300", "0", "0", "0", "-1")]
    panel.position_tol = tk.StringVar(interp, "4.9")
    panel.angle_tol = tk.StringVar(interp, "1")
    panel.comparison = tk.StringVar(interp)
    panel.compare()
    assert "OUTSIDE TOLERANCE" in panel.comparison.get()
    assert "position error=5.0000 mm" in panel.comparison.get()
    assert "orientation error=0.0000 deg" in panel.comparison.get()
    panel.position_tol.set("5.1")
    panel.compare()
    assert "WITHIN TOLERANCE" in panel.comparison.get()


def test_stale_background_file_response_cannot_replace_a_new_scan(monkeypatch):
    panel = panel_state()
    panel._generation = 1
    pending = []
    panel.host.post = lambda callback, *args: pending.append((callback, args))
    monkeypatch.setattr("construct_robot.gui.precision_sensing_test_panel.threading.Thread",
                        lambda target, **kw: SimpleNamespace(start=target))
    success = Mock()
    panel._worker(lambda: "old file", success)
    panel._generation = 2
    for callback, args in pending:
        callback(*args)
    success.assert_not_called()
