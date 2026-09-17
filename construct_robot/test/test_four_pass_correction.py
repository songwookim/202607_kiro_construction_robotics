import copy
import hashlib
import math
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import Pose
import pytest
import yaml

from construct_robot.weld_action_gui import (
    WeldActionGui, correct_four_pass_references, read_weld_pass_reference,
)


def pose(x, y, z, qw=1.0):
    result = Pose()
    result.position.x, result.position.y, result.position.z = (
        float(x), float(y), float(z)
    )
    result.orientation.w = qw
    return result


def reference(start, goal):
    return {"start": start, "goal": goal, "path": "source.log", "sha256": "test"}


def test_four_pass_correction_preserves_distinct_offsets_and_orientations():
    references = {
        1: reference(pose(0, 0, 0), pose(.1, 0, 0)),
        2: reference(pose(0, .01, .002), pose(.08, .01, .003)),
        3: reference(pose(.01, -.02, .004), pose(.09, -.025, .005)),
        4: reference(pose(.02, .03, .006), pose(.07, .035, .007)),
    }
    references[2]["start"].orientation.x = .1
    references[2]["start"].orientation.w = math.sqrt(.99)
    original = copy.deepcopy(references)
    angle = math.radians(10)
    corrected_root_start = pose(.03, -.04, .01)
    corrected_root_goal = pose(
        .03 + .1 * math.cos(angle), -.04 + .1 * math.sin(angle), .012
    )
    result, angle_deg = correct_four_pass_references(
        references, corrected_root_start, corrected_root_goal
    )
    assert set(result) == {1, 2, 3, 4}
    assert angle_deg == pytest.approx(10.0, abs=.1)
    assert result[1]["start"].position == corrected_root_start.position
    assert result[1]["goal"].position == corrected_root_goal.position
    assert result[2]["start"].orientation == original[2]["start"].orientation
    assert result[2]["start"].position.y > corrected_root_start.position.y
    assert result[3]["start"].position.y < corrected_root_start.position.y
    assert references == original


def test_four_pass_correction_rejects_missing_pass_and_large_root_rotation():
    references = {number: reference(pose(0, 0, 0), pose(.1, 0, 0))
                  for number in range(1, 5)}
    with pytest.raises(ValueError, match="Exactly four"):
        correct_four_pass_references({1: references[1]}, pose(0, 0, 0), pose(.1, 0, 0))
    with pytest.raises(ValueError, match="over 30 degrees"):
        correct_four_pass_references(references, pose(0, 0, 0), pose(0, .1, 0))


def test_completed_log_reference_reads_tcp_pose_and_provenance(tmp_path):
    def encoded(point):
        return {
            "planning_group": "right_manipulator",
            "tcp_pose_world": {
                "position_m": {"x": point[0], "y": point[1], "z": point[2]},
                "orientation_xyzw": {"x": 0, "y": 0, "z": 0, "w": 1},
            },
        }
    source = tmp_path / "1.log"
    source.write_text(
        "WELD FEEDBACK LOG\nresult=completed\n\n[teaching_snapshot_yaml]\n"
        + yaml.safe_dump({
            "weld_start": encoded((0, 0, 0)),
            "weld_end": encoded((.1, 0, 0)),
        }), encoding="utf-8"
    )
    parsed = read_weld_pass_reference(source)
    assert parsed["start"].position.x == 0
    assert parsed["goal"].position.x == pytest.approx(.1)
    assert len(parsed["sha256"]) == 64

    source.write_text("WELD FEEDBACK LOG\nresult=failed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completed weld"):
        read_weld_pass_reference(source)


def test_gui_saves_four_predicted_passes_without_overwriting_source_logs(tmp_path):
    gui = object.__new__(WeldActionGui)
    gui.seam_auto_running = False
    gui.four_pass_folder = SimpleNamespace(get=lambda: str(tmp_path))
    gui.four_pass_loaded_folder = tmp_path.resolve()
    gui.four_pass_references = {}
    for number in range(1, 5):
        path = tmp_path / f"{number}.log"
        path.write_text(f"original pass {number}", encoding="utf-8")
        gui.four_pass_references[number] = {
            **reference(
                pose(0, number * .01, 0), pose(.1, number * .01, 0)
            ),
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    gui.computed_seam_endpoints = {
        "start": pose(.01, .01, .001),
        "goal": pose(.11, .01, .002),
    }
    gui.corrected_two_touch_seam = [pose(.01, .01, .001), pose(.11, .01, .002)]
    wait = ("right_manipulator", (), (), pose(0, 0, .1))
    gui.taught_robot_poses = {
        "weld_start_wait": wait,
        "weld_goal_wait": wait,
    }
    gui.four_pass_status = SimpleNamespace(set=Mock())
    gui.error = Mock()
    gui.log = Mock()
    gui.pipeline_result = Mock()

    gui.apply_four_pass_correction()

    gui.error.assert_not_called()
    output = next(tmp_path.glob("corrected_*"))
    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    assert manifest["status"] == "root_xyz_measured_other_passes_predicted"
    assert len(manifest["passes"]) == 4
    assert len(list(output.glob("pass_*.yaml"))) == 4
    pass_two = yaml.safe_load((output / "pass_2.yaml").read_text())
    assert pass_two["status"] == "root_propagated_unverified"
    assert pass_two["corrected_start"]["position_m"]["y"] == pytest.approx(.02)
    assert (tmp_path / "2.log").read_text() == "original pass 2"


def test_selected_pass_touch_saves_only_its_own_result(tmp_path):
    gui = object.__new__(WeldActionGui)
    source = tmp_path / "2.log"
    source.write_text("original log", encoding="utf-8")
    predicted_start, predicted_goal = pose(0, 0.01, 0.005), pose(.1, 0.01, 0.005)
    record = {
        "status": "root_propagated_unverified",
        "source_log_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "corrected_start": WeldActionGui._pose_execution_conditions(predicted_start),
        "corrected_goal": WeldActionGui._pose_execution_conditions(predicted_goal),
    }
    result_path = tmp_path / "pass_2.yaml"
    result_path.write_text(yaml.safe_dump(record), encoding="utf-8")
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump({
        "status": "root_xyz_measured_other_passes_predicted",
        "passes": [{"pass": 2, "file": "pass_2.yaml"}],
    }), encoding="utf-8")
    gui.active_pass_probe = {
        "number": 2, "folder": tmp_path,
        "reference": {
            "weld_start": ("right_manipulator", (), (), predicted_start),
            "weld_end": ("right_manipulator", (), (), predicted_goal),
        },
    }
    gui.seam_probe_touches = {
        name: pose(0, 0, 0) for name in
        ("start_wall", "start_floor", "goal_wall", "goal_floor")
    }
    gui.wall_probe_sign = SimpleNamespace(get=lambda: "-")
    gui.floor_probe_sign = SimpleNamespace(get=lambda: "-")
    gui._seam_geometry_settings = Mock(return_value=(
        gui.active_pass_probe["reference"], (0, 1, 0), (0, 0, 1),
        "AUTO XY", "World Z",
    ))
    measured_start, measured_goal = pose(.001, .011, .006), pose(.101, .011, .006)
    gui._compute_touch_corrected_seam_geometry = Mock(return_value=
        SimpleNamespace(start=measured_start, goal=measured_goal))
    gui.four_pass_corrected = {}
    gui.four_pass_status = SimpleNamespace(set=Mock())
    gui.pipeline_result = Mock()

    gui._save_selected_pass_touch_result()

    saved = yaml.safe_load(result_path.read_text())
    assert saved["status"] == "pass_physically_probed_unverified_for_weld"
    assert saved["corrected_start"]["position_m"]["x"] == pytest.approx(.001)
    assert set(saved["touch_provenance"]["contacts"]) == set(gui.seam_probe_touches)
    assert source.read_text() == "original log"
    assert gui.four_pass_corrected[2]["goal"].position.x == pytest.approx(.101)
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert manifest["passes"][0]["status"] == saved["status"]


def test_apply_selected_pass_requires_matching_loaded_source_log(tmp_path):
    gui = object.__new__(WeldActionGui)
    source = tmp_path / "2.log"
    source.write_text("original log", encoding="utf-8")
    original_start, original_goal = pose(0, .01, 0), pose(.1, .01, 0)
    corrected_start, corrected_goal = pose(.001, .012, .002), pose(.101, .012, .002)
    (tmp_path / "pass_2.yaml").write_text(yaml.safe_dump({
        "source_log_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "status": "pass_physically_probed_unverified_for_weld",
        "corrected_start": WeldActionGui._pose_execution_conditions(corrected_start),
        "corrected_goal": WeldActionGui._pose_execution_conditions(corrected_goal),
    }), encoding="utf-8")
    gui.selected_pass_number = SimpleNamespace(get=lambda: 2)
    gui.four_pass_output_folder = tmp_path
    gui.four_pass_references = {2: {
        **reference(original_start, original_goal), "path": str(source),
    }}
    gui.taught_robot_poses = {
        "weld_start": ("right_manipulator", (), (), pose(.5, 0, 0)),
        "weld_end": ("right_manipulator", (), (), original_goal),
    }
    gui.error = Mock()
    gui._invalidate_seam_correction_runtime = Mock()
    gui.four_pass_status = SimpleNamespace(set=Mock())
    gui.pipeline_result = Mock()

    gui.apply_selected_pass_correction()
    gui.error.assert_called_once()
    gui._invalidate_seam_correction_runtime.assert_not_called()

    gui.taught_robot_poses["weld_start"] = (
        "right_manipulator", (), (), original_start
    )
    gui.error.reset_mock()
    gui.apply_selected_pass_correction()
    gui.error.assert_not_called()
    assert gui.taught_robot_poses["weld_start"][3].position.x == pytest.approx(.001)
    assert gui.taught_robot_poses["weld_end"][3].position.z == pytest.approx(.002)
