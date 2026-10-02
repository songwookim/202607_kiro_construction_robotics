"""Four-pass registration workflow without Tk (application.multipass_controller)."""

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import Pose
import pytest
import yaml

from construct_robot.application import multipass_controller as controller_module
from construct_robot.application.multipass_controller import MultipassController
from construct_robot.application.sequence_executor import pose_execution_conditions

NAMES = tuple(f"right_manipulator_joint{i}" for i in range(1, 7))
ENDPOINTS = (("start_wait", "weld_start_wait"), ("start", "weld_start"),
             ("goal_wait", "weld_goal_wait"), ("goal", "weld_end"))


def pose(x, y, z, yaw_deg=0.0):
    result = Pose()
    result.position.x, result.position.y, result.position.z = float(x), float(y), float(z)
    result.orientation.z = math.sin(math.radians(yaw_deg) / 2)
    result.orientation.w = math.cos(math.radians(yaw_deg) / 2)
    return result


def xyz(p):
    return (p.position.x, p.position.y, p.position.z)


def pass_poses(number):
    start = pose(0.01 * (number - 1), 0.004 * number, 0.002 * number)
    goal = pose(0.11 + 0.008 * (number - 1), 0.004 * number, 0.0025 * number)
    return {
        "start_wait": pose(start.position.x, start.position.y - 0.03, start.position.z + 0.025),
        "start": start,
        "goal_wait": pose(goal.position.x, goal.position.y + 0.03, goal.position.z + 0.025),
        "goal": goal,
    }


def entry(p):
    return {"planning_group": "right_manipulator",
            "joint_state": {"names": list(NAMES), "positions_rad": [0.1] * 6},
            "tcp_pose_world": pose_execution_conditions(p)}


def write_pass_folder(folder):
    for number in range(1, 5):
        poses = {name: entry(pass_poses(number)[endpoint]) for endpoint, name in ENDPOINTS}
        poses["weld_finish"] = entry(pose(0.3, 0.1, 0.2))
        folder.joinpath(f"pass_{number}.yaml").write_text(yaml.safe_dump({
            "schema": "construct_robot_pass_teaching_v1", "pass": number,
            "requires_ik": False, "poses": poses,
        }))


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class InlineThread:
    def __init__(self, target, args=(), daemon=None):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


@pytest.fixture(autouse=True)
def inline_threads(monkeypatch):
    monkeypatch.setattr(controller_module.threading, "Thread", InlineThread)
    monkeypatch.setattr(controller_module.time, "sleep", lambda _s: None)


def make_host(folder, **overrides):
    calls = []
    host = SimpleNamespace(
        calls=calls,
        post=lambda callback, *args: callback(*args),
        log=lambda message: calls.append(("log", message)),
        error=lambda message: calls.append(("error", message)),
        pipeline_result=lambda message: calls.append(("result", message)),
        _set_four_pass_status=lambda text: calls.append(("status", text)),
        _confirm_multi_pass_registration=lambda number: True,
        _confirm_corrected_pass_endpoint=lambda number, endpoint: True,
        _set_keyboard_jog_controls_enabled=lambda enabled: calls.append(("keyboard_controls", enabled)),
        _focus_keyboard_teaching=lambda: calls.append(("focus",)),
        _pose_execution_conditions=pose_execution_conditions,
        _selected_pass=lambda: host.selected,
        selected=1,
        four_pass_folder=Var(str(folder)),
        velocity_percent=Var(30.0),
        keyboard_jog_enabled=Var(False),
        keyboard_jog_status=Var(""),
        keyboard_jog_enable_changed=lambda: calls.append(("keyboard_enable_changed",)),
        keyboard_velocity_arm=None,
        keyboard_velocity_switching=False,
        keyboard_teaching_capture_in_progress=True,
        sequence_running=False,
        execution_allowed=True,
        robot_connected={"right": True},
        four_pass_references={}, four_pass_loaded_folder=None, four_pass_corrected={},
        four_pass_output_folder=None, four_pass_history=[], multi_pass_registration=None,
        taught_robot_poses={name: None for name in (
            "robot_start", "weld_wait", "weld_finish", *(n for _e, n in ENDPOINTS))},
        teaching_capture_provenance={},
        node=SimpleNamespace(
            active_motion_goal=None,
            _current_tcp_pose=lambda group: pose(0.0, -0.05, 0.05),
            _path_start_tcp_pose=lambda group: pose(0.0, -0.05, 0.05),
            run_sequence_cartesian_motion=Mock(return_value=(True, "reached")),
            clear_keyboard_velocity=Mock(),
            set_keyboard_velocity_controller_enabled=Mock(return_value=(True, "ok")),
            resolve_tcp_joint_states=Mock(),
        ),
    )
    for name, value in overrides.items():
        setattr(host, name, value)
    controller = MultipassController(host)
    # Sibling operations are reached through the host, as on the GUI.
    for host_name, method in (
        ("load_four_pass_references", controller.load_references),
        ("_load_latest_sequential_four_pass_state", controller.load_latest_sequential_state),
        ("_validate_four_pass_source_hashes", controller.validate_source_hashes),
        ("_multi_pass_translated_wait", controller.translated_wait),
        ("_run_multi_pass_tcp_move", controller.run_tcp_move),
        ("_multi_pass_start_wait_worker", controller.start_wait_worker),
        ("_multi_pass_start_wait_finished", controller.start_wait_finished),
        ("_enable_multi_pass_keyboard_teaching", controller.enable_keyboard_teaching),
        ("_multi_pass_goal_wait_worker", controller.goal_wait_worker),
        ("_multi_pass_goal_wait_finished", controller.goal_wait_finished),
        ("_complete_multi_pass_registration", controller.complete_registration),
        ("_multi_pass_end_worker", controller.end_worker),
        ("_multi_pass_end_finished", controller.end_finished),
        ("_save_sequential_four_pass_state", controller.save_sequential_state),
        ("_go_to_corrected_pass_endpoint_worker", controller.corrected_endpoint_worker),
        ("_go_to_corrected_pass_endpoint_finished", controller.corrected_endpoint_finished),
        ("_selected_pass_teaching_path", controller.selected_pass_teaching_path),
        ("_load_saved_pass_teaching", controller.load_saved_pass_teaching),
        ("_apply_selected_pass_teaching", controller.apply_selected_pass_teaching),
    ):
        setattr(host, host_name, method)
    return host, controller


def measured(host, number, dx=0.003, yaw_deg=8.0):
    """Measured START/GOAL: predicted pair translated and rotated about START."""
    start = copy.deepcopy(host.four_pass_corrected[number]["start"])
    goal = copy.deepcopy(host.four_pass_corrected[number]["goal"])
    length = math.dist(xyz(start), xyz(goal))
    angle = math.radians(yaw_deg)
    start.position.x += dx
    goal.position.x = start.position.x + length * math.cos(angle)
    goal.position.y = start.position.y + length * math.sin(angle)
    goal.position.z = start.position.z
    return start, goal


def register(host, controller, number, **kwargs):
    host.selected = number
    controller.run_correction()
    start, goal = measured(host, number, **kwargs)
    controller.finish_keyboard_capture("i", (NAMES, (0.0,) * 6, start, {"key": "i"}), None)
    controller.finish_keyboard_capture("j", (NAMES, (0.0,) * 6, goal, {"key": "j"}), None)
    return start, goal


def test_controller_imports_no_tkinter_or_gui():
    tree = ast.parse(Path(controller_module.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    for name in imported:
        assert not (name or "").startswith(("tkinter", "construct_robot.gui", "construct_robot.nodes")), name


@pytest.mark.parametrize("anchor,unchanged,updated", [
    (1, (), (2, 3, 4)), (2, (1,), (3, 4)), (3, (1, 2), (4,)), (4, (1, 2, 3), ()),
])
def test_registration_is_cumulative_from_the_anchor_pass(tmp_path, anchor, unchanged, updated):
    write_pass_folder(tmp_path)
    host, controller = make_host(tmp_path)
    assert controller.load_references()
    before = copy.deepcopy(host.four_pass_corrected)

    start, goal = register(host, controller, anchor)

    assert not [c for c in host.calls if c[0] == "error"]
    assert xyz(host.four_pass_corrected[anchor]["start"]) == pytest.approx(xyz(start))
    assert xyz(host.four_pass_corrected[anchor]["goal"]) == pytest.approx(xyz(goal))
    for number in unchanged:
        assert host.four_pass_corrected[number]["start"] == before[number]["start"]
    for number in updated:
        assert host.four_pass_corrected[number]["start"] != before[number]["start"]
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert manifest["current_anchor_pass"] == anchor
    assert manifest["history"][-1]["later_passes_updated"] == list(updated)
    assert host.multi_pass_registration is None
    assert host.calls[-1] == ("result", f"Pass {anchor} correction saved · Weld end reached")


def test_motion_order_start_wait_retract_goal_wait_then_weld_end(tmp_path):
    write_pass_folder(tmp_path)
    host, controller = make_host(tmp_path)
    controller.load_references()
    register(host, controller, 2)
    motions = host.node.run_sequence_cartesian_motion.call_args_list
    assert [(c.args[0]["path_kind"], c.args[0]["touch_guard"]) for c in motions] == [
        ("Pass 2 corrected logged START WAIT", False),
        ("Pass 2 retract to corrected logged START WAIT", False),
        ("Pass 2 corrected logged GOAL WAIT", True),
        ("Pass 2 Weld end", False),
    ]
    assert ("keyboard_enable_changed",) in host.calls
    host.node.set_keyboard_velocity_controller_enabled.assert_called_with("right", False)


def test_direction_change_over_30_degrees_is_rejected(tmp_path):
    write_pass_folder(tmp_path)
    host, controller = make_host(tmp_path)
    controller.load_references()
    before = copy.deepcopy(host.four_pass_corrected)
    register(host, controller, 2, yaw_deg=40.0)
    errors = [c[1] for c in host.calls if c[0] == "error"]
    assert errors and "limit 30.0" in errors[-1]
    assert host.four_pass_corrected == before
    assert not (tmp_path / "manifest.yaml").exists()


def test_changed_source_reference_is_rejected(tmp_path):
    write_pass_folder(tmp_path)
    host, controller = make_host(tmp_path)
    controller.load_references()
    (tmp_path / "pass_3.yaml").write_text("tampered")
    with pytest.raises(ValueError, match="Pass 3 reference changed"):
        controller.validate_source_hashes()
    controller.run_correction()
    assert ("error", "Cannot start multi-pass correction: Pass 3 reference changed "
            "after loading; reload references") in host.calls
    host.node.run_sequence_cartesian_motion.assert_not_called()


def test_start_wait_failure_ends_session_and_stale_worker_does_nothing(tmp_path):
    write_pass_folder(tmp_path)
    host, controller = make_host(tmp_path)
    controller.load_references()
    host.node.run_sequence_cartesian_motion.return_value = (False, "touch stop")
    controller.run_correction()
    assert host.multi_pass_registration is None
    assert ("error", "Pass 1 START WAIT move failed: touch stop") in host.calls
    host.node.run_sequence_cartesian_motion.reset_mock()
    controller.goal_wait_worker(1, pose(0, 0, 0))  # session already ended (STOP/failure)
    host.node.run_sequence_cartesian_motion.assert_not_called()


def test_capture_key_must_match_the_phase(tmp_path):
    write_pass_folder(tmp_path)
    host, controller = make_host(tmp_path)
    controller.load_references()
    controller.run_correction()
    controller.finish_keyboard_capture("j", (NAMES, (0.0,) * 6, pose(0, 0, 0), {}), None)
    assert ("error", "Pass 1 expects I capture, not J") in host.calls
    assert host.multi_pass_registration["phase"] == "waiting_start_capture"


def test_selected_pass_teaching_save_load_and_apply(tmp_path):
    write_pass_folder(tmp_path)
    host, controller = make_host(tmp_path)
    controller.load_references()
    source_before = (tmp_path / "pass_2.yaml").read_bytes()
    host.selected = 3
    for endpoint, name in ENDPOINTS:
        tcp = copy.deepcopy(pass_poses(3)[endpoint])
        tcp.position.z += 0.004
        host.taught_robot_poses[name] = ("right_manipulator", NAMES, (0.2,) * 6, tcp)
    controller.save_teaching_to_selected_pass()
    assert (tmp_path / "pass_2.yaml").read_bytes() == source_before
    saved = yaml.safe_load((tmp_path / "pass_3.yaml").read_text())
    assert saved["status"] == "manual_teaching_saved"
    assert saved["poses"]["weld_start"]["tcp_pose_world"]["position_m"]["z"] == pytest.approx(0.010)

    host.linear_tcp_endpoints = [None, None]
    host._invalidate_seam_correction_runtime = Mock()
    host._seam_reference_yaml_path = lambda group: tmp_path / "seam_ref.yaml"
    host._initial_state_yaml_path = lambda group, name: tmp_path / f"init_{name}.yaml"
    host.teaching_pose_changed = Mock()
    for name in host.taught_robot_poses:
        host.taught_robot_poses[name] = None
    controller.load_saved_teaching_for_selected_pass()
    assert host.taught_robot_poses["weld_start"][3].position.z == pytest.approx(0.010)
    host.node.resolve_tcp_joint_states.assert_not_called()  # requires_ik False: exact restore
    assert (tmp_path / "init_weld_end.yaml").is_file()
    assert host.linear_tcp_endpoints[0].position.z == pytest.approx(0.010)
