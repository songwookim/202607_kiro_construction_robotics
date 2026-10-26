import copy
import json
import math

import pytest
import yaml

from construct_robot.core.precision_sensing_inspection import (
    SensingPose, inspect_paths, pose_difference, tcp_wire_error_mm,
)
from construct_robot.io.precision_sensing_results import load_result


def result_document():
    return {"error_code": 0, "error_message": "PASS", "data_file_path": "",
            "paths": [{"waypoints": [
                {"position": {"x": x, "y": 0.0, "z": 0.1},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}
                for x in (0.0, 0.005, 0.010)]}]}


def test_inspection_preserves_wire_pose_convention_and_units():
    poses, notes = inspect_paths(result_document(), "World")
    assert poses[-1].position == (0.010, 0.0, 0.1)
    assert poses[-1].kind == "wire"
    assert poses[-1].quaternion == (0.0, 0.0, 0.0, 1.0)
    assert "length=10.000 mm, max step=5.000 mm" in notes[1]


@pytest.mark.parametrize("field,value", [("x", float("nan")), ("y", float("inf"))])
def test_bad_positions_cannot_be_published(field, value):
    document = result_document()
    document["paths"][0]["waypoints"][0]["position"][field] = value
    with pytest.raises(ValueError, match="finite"):
        inspect_paths(document, "World")


def test_bad_quaternion_and_empty_success_are_rejected():
    document = result_document()
    document["paths"][0]["waypoints"][1]["orientation"]["w"] = 0.0
    with pytest.raises(ValueError, match="unit norm"):
        inspect_paths(document, "World")
    document["paths"] = []
    with pytest.raises(ValueError, match="no paths"):
        inspect_paths(document, "World")


def test_failed_action_does_not_publish_attached_paths():
    document = result_document()
    document["error_code"] = 3
    poses, notes = inspect_paths(document, "World")
    assert poses == ()
    assert "Failed result" in notes[-1]


def test_zero_length_and_spacing_diagnostics():
    document = result_document()
    document["paths"][0]["waypoints"] *= 2
    _, notes = inspect_paths(document, "World")
    assert "CHECK: spacing exceeds 5 mm" in notes[1]
    document["paths"][0]["waypoints"] = [copy.deepcopy(document["paths"][0]["waypoints"][0])] * 2
    with pytest.raises(ValueError, match="zero length"):
        inspect_paths(document, "World")


def test_reference_error_accounts_for_quaternion_double_cover_and_frame():
    pose = SensingPose(1, "START", "tcp", "World", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    reference = SensingPose(1, "ref", "tcp", "World", (0.003, 0.004, 0.0), (0.0, 0.0, 0.0, -1.0))
    assert pose_difference(pose, reference) == pytest.approx((5.0, 0.0))
    wrong = SensingPose(1, "ref", "wire", "World", reference.position, reference.quaternion)
    with pytest.raises(ValueError, match="convention"):
        pose_difference(pose, wrong)


def test_tcp_wire_check_uses_torch_local_z_without_comparing_surface_orientation():
    tcp = SensingPose(1, "START", "tcp", "World", (1.0, 0.0, 0.0),
                      (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)))
    wire = SensingPose(1, "0", "wire", "World", (1.0212, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert tcp_wire_error_mm(tcp, wire, 21.2) < 1e-10


def test_result_file_loads_tcp_endpoints_and_detects_mixed_frame(tmp_path):
    document = result_document()
    document["error_message"] = "PASS_UNRESOLVED: end_S"
    path = tmp_path / "precision_sensing_result.yaml"
    path.write_text(yaml.safe_dump(document))
    (tmp_path / "sections.json").write_text(json.dumps({"base_frame": "World", "length_mm": 10.0}))
    (tmp_path / "kiro").mkdir()
    tcp = {"schema": "xy_endpoint_review_v1", "pass": 1, "frame_id": "World"}
    for label, x in (("current_start", 0.0), ("current_goal", 0.01)):
        tcp[label] = {"position_m": {"x": x, "y": 0.0, "z": 0.0788},
                      "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}
    (tmp_path / "kiro/pass_1.yaml").write_text(yaml.safe_dump(tcp))
    (tmp_path / "kiro/pass_1_info.yaml").write_text(yaml.safe_dump(
        {"derived": {"frames": {"tcp_to_aim_point_mm": 21.2}}}))
    result = load_result(path, "World")
    assert len(result.poses) == 5
    assert [p.kind for p in result.poses[-2:]] == ["tcp", "tcp"]
    assert result.sections["length_mm"] == 10.0
    assert any("UNRESOLVED" in note for note in result.notes)
    assert sum("error=0.000000 mm" in note for note in result.notes) == 2
    with pytest.raises(ValueError, match="differs"):
        load_result(path, "robot_base")
