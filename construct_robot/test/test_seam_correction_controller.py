"""Seam-correction / touch-probe workflow without Tk (seam_correction_controller)."""

import ast
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import Pose
import pytest

from construct_robot.application import seam_correction_controller as controller_module
from construct_robot.application.seam_correction_controller import SeamCorrectionController


class UiError(Exception):
    """Stands in for tk.TclError: an operator-setting read failure."""


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def pose(x, y, z):
    result = Pose()
    result.position.x, result.position.y, result.position.z = x, y, z
    result.orientation.y, result.orientation.w = 0.9238795, 0.3826834
    return result


JOINTS = tuple(f"right_manipulator_joint{i}" for i in range(1, 7))


def stored(p):
    return ("right_manipulator", JOINTS, (0.0,) * 6, p)


TOUCHES = {
    "start_wall": pose(0.500, 0.2505, 0.110), "start_floor": pose(0.502, 0.240, 0.0985),
    "goal_wall": pose(0.700, 0.2507, 0.111), "goal_floor": pose(0.698, 0.241, 0.0987),
}


class InlineThread:
    def __init__(self, target, args=(), daemon=None):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


@pytest.fixture
def inline_threads(monkeypatch):
    monkeypatch.setattr(controller_module.threading, "Thread", InlineThread)


def make_host(**overrides):
    calls = []
    host = SimpleNamespace(
        calls=calls,
        keyboard_velocity_arm=None,
        keyboard_velocity_switching=False,
        post=lambda callback, *args: callback(*args),
        log=lambda message: calls.append(("log", message)),
        error=lambda message: calls.append(("error", message)),
        pipeline_waiting=lambda message: calls.append(("waiting", message)),
        pipeline_result=lambda message: calls.append(("result", message)),
        _set_corner_touch_status=lambda text: calls.append(("status", text)),
        _set_auto_seam_button_enabled=lambda enabled: calls.append(("auto_button", enabled)),
        _set_stop_auto_seam_button_enabled=lambda enabled: calls.append(("stop_button", enabled)),
        _set_auto_seam_status=lambda text: calls.append(("auto_status", text)),
        _pose_values=lambda p: (p.position.x, p.position.y, p.position.z,
                                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w),
        planning_group=Var("right_manipulator"),
        touch_probe_distance_mm=Var(30.0),
        touch_probe_speed_percent=Var(2.0),
        touch_settle_seconds=Var(0.7),
        wall_probe_axis=Var("World Y"),
        wall_probe_sign=Var("+"),
        floor_probe_axis=Var("World Z"),
        floor_probe_sign=Var("-"),
        touch_sensing_enabled=Var(False),
        execution_allowed=True,
        robot_connected={"right": True},
        touch_input_states={"right": False},
        taught_robot_poses={
            "weld_start_wait": stored(pose(0.50, 0.20, 0.30)),
            "weld_start": stored(pose(0.50, 0.25, 0.10)),
            "weld_goal_wait": stored(pose(0.70, 0.20, 0.30)),
            "weld_end": stored(pose(0.70, 0.25, 0.10)),
        },
        seam_auto_running=False,
        seam_auto_expected_kind=None,
        seam_auto_stage_success=False,
        seam_auto_stage_event=threading.Event(),
        seam_auto_returned_kinds=set(),
        seam_auto_move_to_end_requested=False,
        automatic_probe_kind=None,
        pass_probe_touch_yaml_target=None,
        seam_probe_touches={name: None for name in TOUCHES},
        seam_probe_starts={name: None for name in TOUCHES},
        seam_probe_stops={name: None for name in TOUCHES},
        corrected_seam_geometry=None,
        hicomm_client=SimpleNamespace(set_arc=lambda value: calls.append(("set_arc", value))),
        _set_fastech_output_sync=Mock(return_value=(True, "ok")),
        _persist_seam_touch_yaml=Mock(return_value="touch.yaml"),
        _ensure_seam_teaching_reference=lambda require_complete=False: {
            "weld_start": stored(pose(0.50, 0.25, 0.10)),
            "weld_end": stored(pose(0.70, 0.25, 0.10)),
        },
        node=SimpleNamespace(
            node_touch_input_states={"right": False},
            execute_touch_probe=Mock(),
            return_touch_probe=Mock(),
            run_sequence_named_pose=Mock(return_value=(True, "reached")),
            stop_auto_motion=Mock(),
            clear_touch_probe=Mock(),
            publish_touch_geometry=Mock(),
        ),
    )
    for name, value in overrides.items():
        setattr(host, name, value)
    controller = SeamCorrectionController(host, touch_output_port=0, ui_error_types=(UiError,))
    # Sibling operations are reached through the host, as on the GUI.
    for host_name, method in (
        ("_signal_auto_seam_stage", controller.signal_automatic_stage),
        ("_finish_automatic_seam_correction", controller.finish_automatic_correction),
        ("_wait_for_touch_release", controller.wait_for_touch_release),
        ("_launch_automatic_seam_stage", controller.launch_automatic_stage),
        ("_resolve_probe_direction", controller.resolve_probe_direction),
        ("_publish_touch_geometry_if_ready", lambda endpoint, point=None: False),
        ("start_automatic_touch_probe", controller.start_touch_probe),
    ):
        setattr(host, host_name, method)
    host._complete_automatic_seam_correction = Mock()
    return host, controller


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


def workflow(host):
    return (
        ({"pose_label": "START WAIT", "pose_name": "weld_start_wait"}, ("start_wall", "start_floor")),
        ({"pose_label": "GOAL WAIT", "pose_name": "weld_goal_wait"}, ("goal_wall", "goal_floor")),
    )


def test_worker_runs_waits_and_probes_in_order_then_completes():
    host, controller = make_host(seam_auto_running=True)
    probed = []

    def launch(kind):
        probed.append(kind)
        host.seam_probe_touches[kind] = TOUCHES[kind]
        host.seam_auto_returned_kinds.add(kind)
        controller.signal_automatic_stage(True, kind)

    host._launch_automatic_seam_stage = launch
    controller.automatic_correction_worker(workflow(host))

    assert probed == ["start_wall", "start_floor", "goal_wall", "goal_floor"]
    labels = [c.args[0]["pose_label"] for c in host.node.run_sequence_named_pose.call_args_list]
    assert labels == ["START WAIT", "GOAL WAIT"]
    host._complete_automatic_seam_correction.assert_called_once_with()


def test_stage_timeout_and_incomplete_checkpoint_fail_the_run():
    class TimedOut(threading.Event):
        def wait(self, timeout=None):
            assert timeout == 180.0
            return False

    host, controller = make_host(seam_auto_running=True, seam_auto_stage_event=TimedOut())
    host._launch_automatic_seam_stage = lambda kind: None
    controller.automatic_correction_worker(workflow(host))
    assert ("error", "Automatic seam correction stopped: start_wall timed out") in host.calls
    assert host.seam_auto_running is False

    host, controller = make_host(seam_auto_running=True)
    host._launch_automatic_seam_stage = lambda kind: controller.signal_automatic_stage(True, kind)
    controller.automatic_correction_worker(workflow(host))
    assert ("error", "Automatic seam correction stopped: start_wall incomplete: "
            "touch_saved=False, returned=False") in host.calls


def test_touch_release_wait_times_out_while_contact_stays_on():
    host, controller = make_host(touch_input_states={"right": True})
    assert controller.wait_for_touch_release(timeout=0.05) is False
    host.touch_input_states["right"] = False
    assert controller.wait_for_touch_release(timeout=0.05) is True


def test_stale_stage_results_are_ignored():
    host, controller = make_host(seam_auto_running=True, seam_auto_expected_kind="goal_wall")
    controller.signal_automatic_stage(True, "start_wall")
    assert not host.seam_auto_stage_event.is_set()
    controller.signal_automatic_stage(True, "goal_wall")
    assert host.seam_auto_stage_event.is_set() and host.seam_auto_stage_success


def test_stop_releases_the_worker_and_stops_robot_motion(inline_threads):
    host, controller = make_host(seam_auto_running=True, automatic_probe_kind="goal_wall",
                                 seam_auto_expected_kind="goal_wall")
    controller.stop_automatic_correction()
    assert host.seam_auto_running is False and host.automatic_probe_kind is None
    assert host.seam_auto_stage_event.is_set() and not host.seam_auto_stage_success
    assert ("stop_button", False) in host.calls
    host.node.stop_auto_motion.assert_called_once_with("right")


def test_touch_probe_failure_clears_probe_and_signals_failure():
    host, controller = make_host(seam_auto_running=True, automatic_probe_kind="start_floor",
                                 seam_auto_expected_kind="start_floor")
    controller.touch_probe_failed("no contact")
    host.node.clear_touch_probe.assert_called_once_with()
    assert host.seam_auto_stage_event.is_set() and not host.seam_auto_stage_success
    assert ("error", "start_floor probe failed: no contact") in host.calls


def test_start_touch_probe_enables_do0_then_launches_signed_probe(inline_threads):
    host, controller = make_host()
    host._confirm_touch_probe = Mock(return_value=True)
    controller.start_touch_probe("goal_floor")
    host._set_fastech_output_sync.assert_called_once_with(0, True)
    assert ("set_arc", False) in host.calls
    assert host.automatic_probe_kind == "goal_floor"
    host.node.execute_touch_probe.assert_called_once_with(
        "right_manipulator", "goal_floor", (-0.0, -0.0, -1.0), 0.03, 0.02, 0.001,
    )


@pytest.mark.parametrize("changes,message", [
    ({"touch_probe_distance_mm": Var(UiError("bad"))}, "Touch probe distance, speed, or settle time is invalid"),
    ({"touch_probe_distance_mm": Var(500.0)}, "Touch probe max travel must be in 1..200 mm"),
    ({"touch_input_states": {"right": True}}, "Fastech DI4 is already ON; release the touch signal before probing"),
    ({"automatic_probe_kind": "start_wall"}, "Another Fastech DI4 touch probe is already active"),
    # Keyboard teaching keeps the JTC active (Servo-J jog stream); probing must wait.
    ({"keyboard_velocity_arm": "right"}, "Disable Keyboard Teaching before touch probing"),
    ({"keyboard_velocity_switching": True}, "Disable Keyboard Teaching before touch probing"),
])
def test_start_touch_probe_rejects_unsafe_requests(changes, message):
    host, controller = make_host(**changes)
    controller.start_touch_probe("goal_floor", skip_confirmation=True)
    assert ("error", message) in host.calls
    host._set_fastech_output_sync.assert_not_called()
    host.node.execute_touch_probe.assert_not_called()


def test_probe_direction_resolution():
    host, controller = make_host()
    assert controller.resolve_probe_direction("floor") == ((0.0, 0.0, 1.0), "World Z")
    host.wall_probe_axis = Var("AUTO seam normal")
    direction, label = controller.resolve_probe_direction("wall")
    assert direction == pytest.approx((-0.0, 1.0, 0.0)) and label.startswith("AUTO seam-normal")
    with pytest.raises(ValueError, match="unknown probe surface"):
        controller.resolve_probe_direction("ceiling")


def test_corrected_geometry_uses_the_sensed_planes():
    host, controller = make_host(seam_probe_touches=dict(TOUCHES))
    reference = host._ensure_seam_teaching_reference()
    geometry = controller.compute_touch_corrected_geometry(
        reference, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 0.0, 0.0,
    )
    assert host.corrected_seam_geometry is geometry
    # START lies on the sensed wall (y ~= wall touches) and base (z ~= floor touches).
    assert geometry.start.position.y == pytest.approx(0.2505, abs=2e-4)
    assert geometry.start.position.z == pytest.approx(0.0985, abs=2e-4)


def test_automatic_capture_stores_contact_and_starts_return(inline_threads):
    host, controller = make_host(automatic_probe_kind="start_wall")
    contact, start, stopped = TOUCHES["start_wall"], pose(0.5, 0.22, 0.13), pose(0.5, 0.2505, 0.1102)
    controller.apply_touch_capture(contact, "right_manipulator", "automatic probe:start_wall",
                                   start, stopped, "cancel")
    assert host.seam_probe_touches["start_wall"] is not contact
    assert host.seam_probe_stops["start_wall"].position.z == pytest.approx(0.1102)
    assert host.automatic_probe_kind is None
    host.node.clear_touch_probe.assert_called_once_with(cancel_return=False)
    args = host.node.return_touch_probe.call_args.args
    assert args[0] == "right_manipulator" and args[5:] == ("start_wall", 0.7, "cancel")
    assert args[3] == pytest.approx(0.02)

    controller.touch_probe_return_finished(True, "returned", "start_wall")
    assert "start_wall" in host.seam_auto_returned_kinds
