import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import Pose
import pytest
import yaml

from construct_robot.weld_action_gui import (
    WeldActionGui,
    correct_remaining_passes,
    read_pass_teaching_reference,
    read_weld_pass_reference,
)


def pose(x, y, z, yaw_deg=0.0):
    result = Pose()
    result.position.x, result.position.y, result.position.z = (
        float(x), float(y), float(z)
    )
    half = math.radians(yaw_deg) * 0.5
    result.orientation.z = math.sin(half)
    result.orientation.w = math.cos(half)
    return result


def pass_state():
    result = {}
    for number in range(1, 5):
        start = pose(
            0.01 * (number - 1), 0.004 * number,
            0.002 * number, yaw_deg=2.0 * number,
        )
        goal = pose(
            0.11 + 0.008 * (number - 1), 0.004 * number,
            0.0025 * number, yaw_deg=3.0 * number,
        )
        result[number] = {
            "start_wait": pose(
                start.position.x, start.position.y - 0.030,
                start.position.z + 0.025, yaw_deg=10.0 + number,
            ),
            "start": start,
            "goal_wait": pose(
                goal.position.x, goal.position.y + 0.030,
                goal.position.z + 0.025, yaw_deg=-10.0 - number,
            ),
            "goal": goal,
        }
    return result


def translated_measurement(state, anchor, dx=0.003, dy=-0.002, yaw_deg=8.0):
    start = state[anchor]["start"]
    goal = state[anchor]["goal"]
    length = math.dist(
        tuple(getattr(start.position, axis) for axis in ("x", "y", "z")),
        tuple(getattr(goal.position, axis) for axis in ("x", "y", "z")),
    )
    measured_start = pose(
        start.position.x + dx, start.position.y + dy, start.position.z + 0.001
    )
    angle = math.radians(yaw_deg)
    measured_goal = pose(
        measured_start.position.x + length * math.cos(angle),
        measured_start.position.y + length * math.sin(angle),
        measured_start.position.z,
    )
    return measured_start, measured_goal


def xyz(value):
    return tuple(getattr(value.position, axis) for axis in ("x", "y", "z"))


@pytest.mark.parametrize("anchor", [1, 2, 3, 4])
def test_selected_anchor_changes_only_itself_and_later_passes(anchor):
    current = pass_state()
    original = copy.deepcopy(current)
    measured_start, measured_goal = translated_measurement(current, anchor)

    corrected, metadata = correct_remaining_passes(
        current, anchor, measured_start, measured_goal
    )

    assert xyz(corrected[anchor]["start"]) == pytest.approx(xyz(measured_start))
    assert xyz(corrected[anchor]["goal"]) == pytest.approx(xyz(measured_goal))
    for number in range(1, anchor):
        assert corrected[number] == original[number]
    for number in range(anchor + 1, 5):
        assert xyz(corrected[number]["start"]) != pytest.approx(
            xyz(original[number]["start"])
        )
    assert metadata["later_passes_updated"] == list(range(anchor + 1, 5))
    assert current == original


def test_sequential_pass_two_uses_current_pass_three_and_four_state():
    original = pass_state()
    p1_start, p1_goal = translated_measurement(original, 1, yaw_deg=10.0)
    after_one, _ = correct_remaining_passes(original, 1, p1_start, p1_goal)
    before_second = copy.deepcopy(after_one)
    p2_start, p2_goal = translated_measurement(
        after_one, 2, dx=-0.002, dy=0.003, yaw_deg=14.0
    )

    cumulative, _ = correct_remaining_passes(
        after_one, 2, p2_start, p2_goal
    )
    incorrectly_from_source, _ = correct_remaining_passes(
        original, 2, p2_start, p2_goal
    )

    assert cumulative[1] == before_second[1]
    assert xyz(cumulative[3]["start"]) != pytest.approx(
        xyz(incorrectly_from_source[3]["start"])
    )
    assert xyz(cumulative[4]["goal"]) != pytest.approx(
        xyz(incorrectly_from_source[4]["goal"])
    )


def test_start_and_goal_are_separate_anchors_and_rotate_later_offsets():
    current = pass_state()
    measured_start, measured_goal = translated_measurement(
        current, 2, dx=0.006, dy=-0.001, yaw_deg=12.0
    )
    old_start_offset = tuple(
        xyz(current[3]["start"])[index] - xyz(current[2]["start"])[index]
        for index in range(3)
    )
    old_goal_offset = tuple(
        xyz(current[3]["goal"])[index] - xyz(current[2]["goal"])[index]
        for index in range(3)
    )

    corrected, metadata = correct_remaining_passes(
        current, 2, measured_start, measured_goal
    )
    new_start_offset = tuple(
        xyz(corrected[3]["start"])[index] - xyz(measured_start)[index]
        for index in range(3)
    )
    new_goal_offset = tuple(
        xyz(corrected[3]["goal"])[index] - xyz(measured_goal)[index]
        for index in range(3)
    )

    assert new_start_offset != pytest.approx(old_start_offset)
    assert new_goal_offset != pytest.approx(old_goal_offset)
    assert metadata["start_translation_m"] != pytest.approx(
        metadata["goal_translation_m"]
    )


def test_welding_orientations_follow_minimal_seam_rotation():
    current = pass_state()
    measured_start, measured_goal = translated_measurement(current, 1, yaw_deg=15.0)
    corrected, metadata = correct_remaining_passes(
        current, 1, measured_start, measured_goal
    )

    assert metadata["direction_change_deg"] > 10.0
    for number in range(1, 5):
        for endpoint in ("start_wait", "start", "goal_wait", "goal"):
            assert corrected[number][endpoint].orientation != (
                current[number][endpoint].orientation
            )


def test_large_direction_change_is_rejected():
    current = pass_state()
    start = copy.deepcopy(current[2]["start"])
    goal = pose(start.position.x, start.position.y + 0.1, start.position.z)
    with pytest.raises(ValueError, match="limit 30.0"):
        correct_remaining_passes(current, 2, start, goal)


def test_multi_pass_wait_comes_from_selected_log_and_is_corrected():
    gui = object.__new__(WeldActionGui)
    current = pass_state()
    gui.four_pass_corrected = current
    gui.four_pass_references = copy.deepcopy(current)

    translated = gui._multi_pass_translated_wait(3, "start")

    assert translated == current[3]["start_wait"]
    assert translated.orientation != current[3]["start"].orientation
    assert math.dist(xyz(translated), xyz(current[3]["start"])) >= 0.020


def test_wait_offsets_propagate_with_selected_start_and_goal_anchors():
    current = pass_state()
    measured_start, measured_goal = translated_measurement(current, 2, yaw_deg=12.0)
    corrected, _ = correct_remaining_passes(
        current, 2, measured_start, measured_goal
    )

    assert math.dist(
        xyz(corrected[2]["start_wait"]), xyz(corrected[2]["start"])
    ) == pytest.approx(
        math.dist(xyz(current[2]["start_wait"]), xyz(current[2]["start"]))
    )
    assert math.dist(
        xyz(corrected[2]["goal_wait"]), xyz(corrected[2]["goal"])
    ) == pytest.approx(
        math.dist(xyz(current[2]["goal_wait"]), xyz(current[2]["goal"]))
    )
    assert corrected[1] == current[1]


def test_multi_pass_wait_arrival_auto_enables_existing_keyboard_path():
    gui = object.__new__(WeldActionGui)
    gui.multi_pass_registration = {"pass": 2}
    gui.keyboard_velocity_arm = None
    gui.keyboard_velocity_switching = False
    gui.keyboard_jog_enabled = SimpleNamespace(set=Mock())
    gui.four_pass_status = SimpleNamespace(set=Mock())
    gui.keyboard_jog_status = SimpleNamespace(set=Mock())
    gui.keyboard_jog_enable_changed = Mock()

    gui._enable_multi_pass_keyboard_teaching(2, "j")

    gui.keyboard_jog_enabled.set.assert_called_once_with(True)
    gui.keyboard_jog_enable_changed.assert_called_once_with()
    assert "J capture" in gui.four_pass_status.set.call_args.args[0]


def test_completed_log_reads_tcp_joint_snapshots_and_provenance(tmp_path):
    def encoded(point):
        return {
            "planning_group": "right_manipulator",
            "joint_state": {
                "names": [f"joint_{index}" for index in range(6)],
                "positions_rad": [0.1 * index for index in range(6)],
            },
            "tcp_pose_world": {
                "position_m": {"x": point[0], "y": point[1], "z": point[2]},
                "orientation_xyzw": {"x": 0, "y": 0, "z": 0, "w": 1},
            },
        }

    source = tmp_path / "1.log"
    source.write_text(
        "WELD FEEDBACK LOG\nresult=completed\n\n[teaching_snapshot_yaml]\n"
        + yaml.safe_dump({
            "weld_start_wait": encoded((-0.02, 0, 0.03)),
            "weld_start": encoded((0, 0, 0)),
            "weld_goal_wait": encoded((0.12, 0, 0.03)),
            "weld_end": encoded((0.1, 0, 0)),
        }),
        encoding="utf-8",
    )

    parsed = read_weld_pass_reference(source)

    assert parsed["goal"].position.x == pytest.approx(0.1)
    assert set(parsed["joint_states"]) == {
        "start_wait", "start", "goal_wait", "goal",
    }
    assert parsed["start_wait"].position.z == pytest.approx(0.03)
    assert len(parsed["sha256"]) == 64


def gui_with_sources(tmp_path):
    gui = object.__new__(WeldActionGui)
    gui.four_pass_loaded_folder = tmp_path
    gui.four_pass_references = {}
    current = pass_state()
    for number in range(1, 5):
        path = tmp_path / f"{number}.log"
        path.write_text(f"immutable pass {number}", encoding="utf-8")
        gui.four_pass_references[number] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **{
                endpoint: copy.deepcopy(current[number][endpoint])
                for endpoint in ("start_wait", "start", "goal_wait", "goal")
            },
            "joint_states": {
                endpoint: (
                    tuple(
                        f"right_manipulator_joint{index}"
                        for index in range(1, 7)
                    ),
                    tuple(0.01 * index for index in range(6)),
                )
                for endpoint in ("start_wait", "start", "goal_wait", "goal")
            },
        }
    gui.four_pass_corrected = current
    gui.four_pass_output_folder = None
    gui.four_pass_history = []
    gui.taught_robot_poses = {
        name: None for name in ("weld_start_wait", "weld_start", "weld_goal_wait", "weld_end")
    }
    return gui


def test_sequential_save_never_overwrites_source_logs(tmp_path):
    gui = gui_with_sources(tmp_path)
    originals = {
        number: Path(gui.four_pass_references[number]["path"]).read_bytes()
        for number in range(1, 5)
    }
    previous = copy.deepcopy(gui.four_pass_corrected)
    measured_start, measured_goal = translated_measurement(previous, 2)
    corrected, transform = correct_remaining_passes(
        previous, 2, measured_start, measured_goal
    )
    session = {
        "pass": 2,
        "previous": previous,
        "measured_start": measured_start,
        "measured_goal": measured_goal,
    }

    gui._save_sequential_four_pass_state(corrected, session, transform)

    for number in range(1, 5):
        assert Path(gui.four_pass_references[number]["path"]).read_bytes() == originals[number]
    manifest = yaml.safe_load((gui.four_pass_output_folder / "manifest.yaml").read_text())
    assert manifest["source_logs_immutable"] is True
    assert manifest["current_anchor_pass"] == 2
    assert manifest["history"][-1]["later_passes_updated"] == [3, 4]
    loader = object.__new__(WeldActionGui)
    loader.log = Mock()
    restored, restored_folder, history = (
        loader._load_latest_sequential_four_pass_state(
            tmp_path, gui.four_pass_references
        )
    )
    assert restored_folder == gui.four_pass_output_folder
    assert xyz(restored[3]["start"]) == pytest.approx(
        xyz(corrected[3]["start"])
    )
    assert len(history) == 1


def test_source_hash_mismatch_is_rejected(tmp_path):
    gui = gui_with_sources(tmp_path)
    Path(gui.four_pass_references[3]["path"]).write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="Pass 3 reference changed"):
        gui._validate_four_pass_source_hashes()


def test_save_and_load_selected_pass_teaching_is_separate_from_source_logs(tmp_path):
    gui = gui_with_sources(tmp_path)
    gui.selected_pass_number = SimpleNamespace(get=lambda: 3)
    names = tuple(f"right_manipulator_joint{index}" for index in range(1, 7))
    gui.taught_robot_poses = {
        "weld_start_wait": (
            "right_manipulator", names, (0.1,) * 6, pose(0.31, 0.01, 0.08)
        ),
        "weld_start": (
            "right_manipulator", names, (0.2,) * 6, pose(0.32, 0.02, 0.03)
        ),
        "weld_goal_wait": (
            "right_manipulator", names, (0.3,) * 6, pose(0.41, 0.01, 0.08)
        ),
        "weld_end": (
            "right_manipulator", names, (0.4,) * 6, pose(0.42, 0.02, 0.03)
        ),
    }
    gui.teaching_capture_provenance = {}
    gui.four_pass_status = SimpleNamespace(set=Mock())
    gui.pipeline_result = Mock()
    gui.error = Mock()
    source_before = (tmp_path / "3.log").read_bytes()

    gui.save_teaching_to_selected_pass()

    gui.error.assert_not_called()
    saved_path = tmp_path / "pass_teaching" / "pass_3_teaching.yaml"
    assert saved_path.is_file()
    assert (tmp_path / "3.log").read_bytes() == source_before
    saved_reference = read_pass_teaching_reference(saved_path, 3)
    assert saved_reference["reference_kind"] == "saved_pass_teaching"
    loaded, joints, used_path = gui._load_saved_pass_teaching(3)
    assert used_path == saved_path
    assert xyz(loaded["start"]) == pytest.approx((0.32, 0.02, 0.03))
    assert joints["goal"][1] == pytest.approx((0.4,) * 6)

    gui.taught_robot_poses = {
        name: None
        for name in ("weld_start_wait", "weld_start", "weld_goal_wait", "weld_end")
    }
    gui.linear_tcp_endpoints = [None, None]
    gui._invalidate_seam_correction_runtime = Mock()
    gui._seam_reference_yaml_path = Mock(return_value=tmp_path / "selected_ref.yaml")
    gui._initial_state_yaml_path = Mock(
        side_effect=lambda _group, name: tmp_path / f"selected_{name}.yaml"
    )
    gui.teaching_pose_changed = Mock()
    gui.node = SimpleNamespace(resolve_tcp_joint_states=Mock())
    gui.error.reset_mock()

    gui.load_saved_teaching_for_selected_pass()

    gui.error.assert_not_called()
    gui.node.resolve_tcp_joint_states.assert_not_called()
    assert xyz(gui.taught_robot_poses["weld_start"][3]) == pytest.approx(
        (0.32, 0.02, 0.03)
    )
    assert gui.taught_robot_poses["weld_end"][2] == pytest.approx((0.4,) * 6)


def test_save_selected_pass_teaching_requires_no_loaded_reference_set(tmp_path):
    gui = object.__new__(WeldActionGui)
    gui.selected_pass_number = SimpleNamespace(get=lambda: 2)
    gui.four_pass_folder = SimpleNamespace(get=lambda: str(tmp_path))
    gui.four_pass_loaded_folder = None
    gui.four_pass_references = {}
    gui.four_pass_corrected = {}
    names = tuple(f"right_manipulator_joint{index}" for index in range(1, 7))
    gui.taught_robot_poses = {
        "weld_start_wait": (
            "right_manipulator", names, (0.1,) * 6, pose(0.0, -0.03, 0.03)
        ),
        "weld_start": (
            "right_manipulator", names, (0.2,) * 6, pose(0.0, 0.0, 0.0)
        ),
        "weld_goal_wait": (
            "right_manipulator", names, (0.3,) * 6, pose(0.1, 0.03, 0.03)
        ),
        "weld_end": (
            "right_manipulator", names, (0.4,) * 6, pose(0.1, 0.0, 0.0)
        ),
    }
    gui.teaching_capture_provenance = {}
    gui.four_pass_status = SimpleNamespace(set=Mock())
    gui.pipeline_result = Mock()
    gui.error = Mock()

    gui.save_teaching_to_selected_pass()

    gui.error.assert_not_called()
    saved_path = tmp_path / "pass_teaching" / "pass_2_teaching.yaml"
    document = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
    assert document["pass"] == 2
    assert "source_log" not in document
    assert xyz(gui.four_pass_corrected[2]["start"]) == pytest.approx(
        (0.0, 0.0, 0.0)
    )


def test_apply_selected_pass_loads_all_four_poses_into_teaching_detail(tmp_path):
    gui = gui_with_sources(tmp_path)
    gui.selected_pass_number = SimpleNamespace(get=lambda: 2)
    gui.linear_tcp_endpoints = [None, None]
    gui.seam_teaching_reference = None
    gui._invalidate_seam_correction_runtime = Mock()
    gui._seam_reference_yaml_path = Mock(return_value=tmp_path / "seam_reference.yaml")
    gui.node = SimpleNamespace(resolve_tcp_joint_states=Mock())
    gui.teaching_pose_changed = Mock()
    gui.four_pass_status = SimpleNamespace(set=Mock())
    gui.pipeline_result = Mock()
    gui.error = Mock()

    gui.apply_selected_pass_correction()

    gui.error.assert_not_called()
    for endpoint, pose_name in (
        ("start_wait", "weld_start_wait"),
        ("start", "weld_start"),
        ("goal_wait", "weld_goal_wait"),
        ("goal", "weld_end"),
    ):
        assert gui.taught_robot_poses[pose_name][3] == (
            gui.four_pass_corrected[2][endpoint]
        )
    gui.teaching_pose_changed.assert_called_once_with()
    assert gui.linear_tcp_endpoints[0] == gui.four_pass_corrected[2]["start"]
    assert gui.linear_tcp_endpoints[1] == gui.four_pass_corrected[2]["goal"]


def test_load_four_references_accepts_pass_teaching_folder(tmp_path):
    source_gui = gui_with_sources(tmp_path)
    source_gui.selected_pass_number = SimpleNamespace(get=lambda: 1)
    names = tuple(f"right_manipulator_joint{index}" for index in range(1, 7))
    source_gui.taught_robot_poses = {
        "weld_start_wait": (
            "right_manipulator", names, (0.1,) * 6, pose(0.0, -0.03, 0.03)
        ),
        "weld_start": (
            "right_manipulator", names, (0.2,) * 6, pose(0.0, 0.0, 0.0)
        ),
        "weld_goal_wait": (
            "right_manipulator", names, (0.3,) * 6, pose(0.1, 0.03, 0.03)
        ),
        "weld_end": (
            "right_manipulator", names, (0.4,) * 6, pose(0.1, 0.0, 0.0)
        ),
    }
    source_gui.teaching_capture_provenance = {}
    source_gui.four_pass_status = SimpleNamespace(set=Mock())
    source_gui.pipeline_result = Mock()
    source_gui.error = Mock()
    source_gui.save_teaching_to_selected_pass()
    template = yaml.safe_load(
        (tmp_path / "pass_teaching" / "pass_1_teaching.yaml").read_text()
    )
    for number in range(2, 5):
        document = copy.deepcopy(template)
        document["pass"] = number
        document["source_log"] = str(tmp_path / f"{number}.log")
        document["source_log_sha256"] = source_gui.four_pass_references[number][
            "sha256"
        ]
        (tmp_path / "pass_teaching" / f"pass_{number}_teaching.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )

    loader = object.__new__(WeldActionGui)
    loader.four_pass_folder = SimpleNamespace(
        get=lambda: str(tmp_path / "pass_teaching")
    )
    loader.four_pass_status = SimpleNamespace(set=Mock())
    loader.log = Mock()
    loader.error = Mock()

    assert loader.load_four_pass_references() is True
    loader.error.assert_not_called()
    assert loader.four_pass_loaded_folder == tmp_path / "pass_teaching"
    assert all(
        reference["reference_kind"] == "saved_pass_teaching"
        for reference in loader.four_pass_references.values()
    )
    assert xyz(loader.four_pass_corrected[4]["goal"]) == pytest.approx(
        (0.1, 0.0, 0.0)
    )

    # Browsing the normal work root must resolve the same YAML files before
    # considering its N.log fallbacks.  The helper logs deliberately are not
    # parseable feedback logs, so this also proves YAML precedence.
    root_loader = object.__new__(WeldActionGui)
    root_loader.four_pass_folder = SimpleNamespace(get=lambda: str(tmp_path))
    root_loader.four_pass_status = SimpleNamespace(set=Mock())
    root_loader.log = Mock()
    root_loader.error = Mock()

    assert root_loader.load_four_pass_references() is True
    root_loader.error.assert_not_called()
    assert root_loader.four_pass_loaded_folder == tmp_path
    assert all(
        reference["reference_kind"] == "saved_pass_teaching"
        for reference in root_loader.four_pass_references.values()
    )


def test_registration_completion_updates_state_without_welder_calls(tmp_path):
    gui = gui_with_sources(tmp_path)
    measured_start, measured_goal = translated_measurement(gui.four_pass_corrected, 3)
    gui.multi_pass_registration = {
        "pass": 3,
        "previous": copy.deepcopy(gui.four_pass_corrected),
        "measured_start": measured_start,
        "measured_goal": measured_goal,
    }
    gui._save_sequential_four_pass_state = Mock()
    gui._validate_four_pass_source_hashes = Mock()
    gui.four_pass_status = SimpleNamespace(set=Mock())
    gui.keyboard_jog_status = SimpleNamespace(set=Mock())
    gui.pipeline_result = Mock()
    gui.error = Mock()
    gui.hicomm_client = Mock()

    gui._complete_multi_pass_registration()

    gui.error.assert_not_called()
    gui.hicomm_client.assert_not_called()
    assert gui.multi_pass_registration is None
    assert xyz(gui.four_pass_corrected[3]["start"]) == pytest.approx(
        xyz(measured_start)
    )
