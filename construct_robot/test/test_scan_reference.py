"""Root-pass reference correction: wire-tip offsets in the seam frame, orientation-independent targets."""

import json
import math

from geometry_msgs.msg import Pose
import numpy as np
import pytest

from construct_robot.application import tcp_calibration as _tcp_cal


@pytest.fixture(autouse=True)
def no_calibrated_wire_tip(monkeypatch):
    """Independent of the cell's scan_teaching/wire_tip_offset.yaml."""
    monkeypatch.setattr(_tcp_cal, "load_offset", lambda path=None: None)
import yaml

from construct_robot.application import scan_correction as sc
from construct_robot.core.scan_reference import (
    SeamFrame, aim_from_tcp, corrected_aim, reference_offsets, tcp_from_aim, torch_angles, torch_axis,
)

DOWN = (1.0, 0.0, 0.0, 0.0)                       # 180 deg about x: torch axis = World -z


def quat_axis_angle(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return (*(axis * math.sin(h)), math.cos(h))


def compose(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


TILTED = compose(quat_axis_angle((1, 0, 0), 42.0), DOWN)     # torch tilted 42 deg in the y-z plane
FRAME = SeamFrame.from_estimate((0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
L = 0.0212


@pytest.fixture(autouse=True)
def hongik_tcp_to_wire_tip(monkeypatch):
    """These tests check the general mechanism with the estimate's tcp_to_aim (KIRO uses 0, see test below)."""
    monkeypatch.setattr(sc, "TCP_TO_WIRE_TIP_M", None)


def test_tcp_aim_round_trip_and_vertical_axis():
    np.testing.assert_allclose(torch_axis(DOWN), (0, 0, -1), atol=1e-12)
    tcp = np.array([0.0, -1.5, 0.6])
    aim = aim_from_tcp(tcp, TILTED, L)
    assert np.linalg.norm(aim - tcp) == pytest.approx(L)
    np.testing.assert_allclose(tcp_from_aim(aim, TILTED, L), tcp, atol=1e-12)


def test_seam_frame_is_right_handed_with_the_normal_up():
    frame = SeamFrame.from_estimate((-0.05, -0.998, -0.014), (0.0, 0.0, -1.0))
    for v in (frame.u, frame.w, frame.n):
        assert np.linalg.norm(v) == pytest.approx(1.0)
    assert frame.n[2] > 0.99 and abs(frame.u @ frame.n) < 1e-12
    np.testing.assert_allclose(np.cross(frame.n, frame.u), frame.w, atol=1e-12)
    np.testing.assert_allclose(frame.to_local(frame.to_world((1, 2, 3))), (1, 2, 3), atol=1e-12)


def test_offsets_are_taken_at_the_wire_tip_and_reproduce_the_reference():
    est_aim = {"start": np.array([0.015, -1.539, 0.566]), "goal": np.array([0.005, -1.738, 0.563])}
    true_offset = {"start": np.array([6.0, -19.0, 3.0]) * 1e-3, "goal": np.array([9.0, -20.0, 3.0]) * 1e-3}
    # the operator measures with a vertical torch; the estimate's torch is tilted - must not matter
    reference = {e: (tcp_from_aim(est_aim[e] + FRAME.to_world(true_offset[e]), DOWN, L), DOWN) for e in ("start", "goal")}
    offsets = reference_offsets(reference, {"aim_start": est_aim["start"], "aim_goal": est_aim["goal"]}, FRAME, L)
    for end in ("start", "goal"):
        np.testing.assert_allclose(offsets[end]["seam_m"], true_offset[end], atol=1e-12)
        seam = {e: offsets[e]["seam_m"] for e in ("start", "goal")}
        aim = corrected_aim(est_aim[end], end, seam, FRAME)
        np.testing.assert_allclose(tcp_from_aim(aim, DOWN, L), reference[end][0], atol=1e-12)


def test_torch_angles_follow_hongik_conventions():
    frame = SeamFrame.from_estimate((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))          # travel +y, w = n x u = -x
    push = compose(quat_axis_angle((1, 0, 0), 30.0), DOWN)
    np.testing.assert_allclose(torch_axis(push), (0.0, 0.5, -math.sqrt(0.75)), atol=1e-12)   # leans along travel
    travel, work = torch_angles(push, frame)
    assert travel == pytest.approx(30.0) and work == pytest.approx(0.0, abs=1e-9)
    assert torch_angles(push, frame, travel_sign=-1.0)[0] == pytest.approx(-30.0)              # E -> S: drag
    toward_x = compose(quat_axis_angle((0, 1, 0), -20.0), DOWN)
    assert torch_axis(toward_x)[0] > 0.0                                                      # leans to +x = -w
    assert torch_angles(toward_x, frame, bevel_sign_w=1)[1] == pytest.approx(-20.0)
    assert torch_angles(toward_x, frame, bevel_sign_w=-1)[1] == pytest.approx(20.0)


def write_estimate(tmp_path, aim_start, aim_goal, quat=TILTED, quat_pass2=None):
    out = tmp_path / "scan" / "hongik_estimate"
    (out / "kiro").mkdir(parents=True)
    (out / "estimate.json").write_text(json.dumps({"u": [0.0, -1.0, 0.0], "n_plate": [0.0, 0.0, 1.0], "bevel_sign_w": -1}))
    for k, dz in ((1, 0.0), (2, 0.004)):
        pose = {}
        q = quat_pass2 if (k == 2 and quat_pass2 is not None) else quat
        for end, aim in (("start", aim_start), ("goal", aim_goal)):
            aim = np.asarray(aim) + np.array([0.0, 0.0, dz])
            tcp = tcp_from_aim(aim, q, L)
            pose[f"current_{end}"] = {"position_m": dict(zip("xyz", map(float, tcp))),
                                      "orientation_xyzw": dict(zip("xyzw", map(float, q)))}
        (out / "kiro" / f"pass_{k}.yaml").write_text(yaml.safe_dump({"pass": k, **pose}))
        info = {"derived": {"role": "root" if k == 1 else "fill", "frames": {
            "tcp_to_aim_point_mm": L * 1e3,
            "aim_point_World_m": {"start": (np.asarray(aim_start) + [0, 0, dz]).tolist(),
                                  "goal": (np.asarray(aim_goal) + [0, 0, dz]).tolist()}}}}
        (out / "kiro" / f"pass_{k}_info.yaml").write_text(yaml.safe_dump(info))
    return tmp_path / "scan"


def test_application_correction_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    shift = np.array([-0.020, -0.005, 0.003])
    for end, aim in (("start", (0.015, -1.539, 0.566)), ("goal", (0.005, -1.738, 0.563))):
        sc.save_reference_end(end, tcp_from_aim(np.asarray(aim) + shift, DOWN, L), DOWN, "test")
    doc = sc.compute_correction(scan)
    assert doc["reference_torch_angles_deg"]["start"]["travel"] == pytest.approx(0.0, abs=1e-6)
    correction = sc.load_correction()
    reference = sc.load_reference()
    for end in ("start", "goal"):                    # pass 1 reproduces the measured TCP with the reference torch
        tcp, _aim = sc.pass_target_tcp(scan, 1, end, DOWN, correction)
        np.testing.assert_allclose(tcp, reference[end][0], atol=1e-9)
    tcp2, aim2 = sc.pass_target_tcp(scan, 2, "start", DOWN, correction)   # other passes: the same shift
    np.testing.assert_allclose(aim2, np.array([0.015, -1.539, 0.570]) + shift, atol=1e-9)
    files = sc.write_corrected_kiro(scan, correction)
    corrected = yaml.safe_load(open(files[0]))
    assert corrected["schema"] == sc.SCHEMA and corrected["frame_id"] == "World"
    np.testing.assert_allclose([corrected["current_start"]["position_m"][a] for a in "xyz"], reference["start"][0], atol=1e-9)


def test_move_target_keeps_the_current_orientation_and_uses_the_aim_point(tmp_path):
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    current = Pose()
    current.position.x, current.position.y, current.position.z = 0.0, -1.5, 0.7
    current.orientation.x, current.orientation.w = 1.0, 0.0                      # vertical torch
    target, info = sc.estimated_move_target(scan, "Pass 1 start", current, None)
    assert (target.orientation.x, target.orientation.w) == (1.0, 0.0)
    # vertical torch: TCP straight above the wire tip by L (not Hongik's tilted TCP)
    np.testing.assert_allclose([target.position.x, target.position.y, target.position.z], (0.015, -1.539, 0.566 + L), atol=1e-9)
    assert not info["corrected"]
    with pytest.raises(ValueError):
        sc.parse_pose_label("Pass x middle")


def test_relative_models_differ_when_hongik_tilts_a_pass_differently(tmp_path):
    lean = compose(quat_axis_angle((0, 1, 0), 29.0), TILTED)            # pass 2 work angle 29 deg off the root's
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563), quat_pass2=lean)
    detail = sc.load_estimate_detail(scan)
    p1, p2 = detail["passes"]
    for end in ("start", "goal"):
        tcp_model, _ = sc.pass_target_tcp(scan, 2, end, DOWN, None, "hongik_tcp")
        aim_model, implied_aim = sc.pass_target_tcp(scan, 2, end, DOWN, None, "hongik_aim")
        root, _ = sc.pass_target_tcp(scan, 1, end, DOWN, None, "hongik_tcp")
        # hongik_tcp keeps Hongik's TCP displacement; hongik_aim keeps its wire-tip displacement
        np.testing.assert_allclose(tcp_model - root, p2[f"tcp_{end}"] - p1[f"tcp_{end}"], atol=1e-12)
        np.testing.assert_allclose(implied_aim - aim_from_tcp(root, DOWN, L), p2[f"aim_{end}"] - p1[f"aim_{end}"], atol=1e-12)
        assert np.linalg.norm(tcp_model - aim_model) > 0.005
    with pytest.raises(ValueError):
        sc.pass_target_tcp(scan, 2, "start", DOWN, None, "unknown")


def test_history_archive_accumulates_small_results(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    (scan / "hongik_estimate" / "estimate.json").write_text(json.dumps(
        {"u": [0.0, -1.0, 0.0], "n_plate": [0.0, 0.0, 1.0], "ends": {"S": {"point_m": [0, 0, 0], "sigma_m": 1e-4, "two_sided": True}}}))
    dest = sc.archive_to_history(scan, note="test")
    assert dest == tmp_path / "teaching" / "history" / "scan"
    assert (dest / "kiro" / "pass_1.yaml").is_file()
    assert yaml.safe_load((dest / "summary.yaml").read_text())["ends_m"]["S"]["two_sided"] is True


def test_scan_results_live_in_the_workspace(monkeypatch, tmp_path):
    from construct_robot.io import scan_paths
    monkeypatch.setenv("KIRO_SCAN_ROOT", str(tmp_path / "scan_data" / "kiro_scan"))
    root = scan_paths.ensure_scan_results_root()
    assert root.is_dir() and (root.parent / "COLCON_IGNORE").is_file() and (root.parent / ".gitignore").read_text() == "*\n"
    monkeypatch.delenv("KIRO_SCAN_ROOT")
    assert str(scan_paths.scan_results_root()).endswith("ros2_ws/scan_data/kiro_scan")


def test_scan_axis_follows_the_first_motion_not_the_taught_points():
    from construct_robot.nodes.hongik_scan_estimate import scan_axis, split_passes
    s = np.concatenate([np.linspace(0.0, -0.2, 200), np.full(50, -0.2), np.linspace(-0.2, 0.0, 200)])
    origins = np.stack([0.02 + 0.0 * s, -1.5 + s, 0.9 + 0.0 * s], axis=1)
    axis = scan_axis(origins)
    np.testing.assert_allclose(axis, (0.0, -1.0, 0.0), atol=1e-9)              # first motion goes -y
    t = np.arange(len(s)) * 0.01
    assert [sign for _i, sign in split_passes(t, (origins - origins[0]) @ axis)] == [1, -1]


def test_references_follow_their_position_and_passes_get_their_remainder(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    lean = compose(quat_axis_angle((0, 1, 0), 29.0), TILTED)
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563), quat_pass2=lean)
    shift = np.array([-0.015, -0.007, 0.003])
    # the operator labels the ends the other way round (weld direction): matching must use the position
    sc.save_reference(1, "goal", tcp_from_aim(np.array([0.015, -1.539, 0.566]) + shift, DOWN, L), DOWN, "t")
    sc.save_reference(1, "start", tcp_from_aim(np.array([0.005, -1.738, 0.563]) + shift, DOWN, L), DOWN, "t")
    p2_goal = np.array([0.002, -1.7385, 0.5705])                       # measured pass 2 at the plan's goal end
    sc.save_reference(2, "start", p2_goal, DOWN, "t")
    doc = sc.compute_correction(scan)
    assert {(m["pass"], m["end"]) for m in doc["matches"]} == {(1, "start"), (1, "goal"), (2, "goal")}
    assert any("used as the" in n for n in doc["notes"])
    correction = sc.load_correction()
    for model in ("hongik_tcp", "hongik_aim"):          # measured_layout: its own test
        goal, _ = sc.pass_target_tcp(scan, 2, "goal", DOWN, correction, model)
        np.testing.assert_allclose(goal, p2_goal, atol=1e-9)              # the measured pass reproduces
        # the unmeasured start of pass 2 takes the goal's remainder
        root_only = dict(correction, pass_deltas_seam_m={})
        shift = goal - sc.pass_target_tcp(scan, 2, "goal", DOWN, root_only, model)[0]
        np.testing.assert_allclose(sc.pass_target_tcp(scan, 2, "start", DOWN, correction, model)[0],
                                   sc.pass_target_tcp(scan, 2, "start", DOWN, root_only, model)[0] + shift, atol=1e-12)


def test_legacy_root_reference_file_is_read(tmp_path):
    path = tmp_path / "root1_reference.yaml"
    pose = {"position_m": {"x": 0.1, "y": 0.2, "z": 0.3}, "orientation_xyzw": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}}
    path.write_text(yaml.safe_dump({"format_version": 1, "start": pose, "goal": pose}))
    refs = sc.load_references(path)
    assert [(r["pass"], r["label"]) for r in refs] == [(1, "start"), (1, "goal")]
    sc.save_reference(3, "goal", (0.0, 0.0, 0.0), DOWN, "t", path)
    assert yaml.safe_load(path.read_text())["format_version"] == 2 and len(sc.load_references(path)) == 3


def test_saving_an_end_replaces_that_end_and_relabels_the_other(tmp_path):
    path = tmp_path / "refs.yaml"
    sc.save_reference(1, "goal", (0.0, -1.745, 0.586), DOWN, "old", path)     # labelled goal, lies at y -1.745
    sc.save_reference(1, "start", (0.0, -1.547, 0.592), DOWN, "old", path)
    sc.save_reference(1, "goal", (0.0, -1.547, 0.590), DOWN, "new", path)     # new goal at y -1.547
    refs = {r["label"]: r for r in sc.load_references(path)}
    assert set(refs) == {"start", "goal"} and len(sc.load_references(path)) == 2
    assert refs["goal"]["source"] == "new" and refs["goal"]["position"][2] == pytest.approx(0.590)
    assert refs["start"]["position"][1] == pytest.approx(-1.745)               # the far entry became the start


def test_kiro_uses_hongiks_tcp_to_wire_tip_by_default():
    assert sc.TCP_TO_WIRE_TIP_M is None


def test_measured_layout_places_the_passes_relative_to_the_corrected_root(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    for k, dz in ((1, 0.0), (2, 0.0047)):                      # operator: pass 2 = 3.6 mm lateral, 4.7 mm up
        for end, aim in (("start", (0.015, -1.539, 0.566)), ("goal", (0.005, -1.738, 0.563))):
            tip = np.array(aim) + FRAME.to_world((0.0, 0.0036 if k == 2 else 0.0, dz)) + [0.002, 0, 0]
            sc.save_reference(k, end, tcp_from_aim(tip, DOWN, L), DOWN, "t")
    layout = sc.save_pass_layout(scan)
    np.testing.assert_allclose(layout[2]["start"], (0.0, 0.0036, 0.0047), atol=1e-12)
    sc.compute_correction(scan)
    correction = sc.load_correction()
    for q in (DOWN, TILTED):
        tcp1, aim1 = sc.pass_target_tcp(scan, 1, "goal", q, correction, "measured_layout")
        tcp2, aim2 = sc.pass_target_tcp(scan, 2, "goal", q, correction, "measured_layout")
        np.testing.assert_allclose(FRAME.to_local(aim2 - aim1), (0.0, 0.0036, 0.0047), atol=1e-12)


def test_a_pass_measured_with_its_own_orientation_corrects_the_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    for end, aim in (("start", (0.015, -1.539, 0.566)), ("goal", (0.005, -1.738, 0.563))):
        sc.save_reference(1, end, tcp_from_aim(np.array(aim), DOWN, L), DOWN, "t")
        sc.save_reference(2, end, tcp_from_aim(np.array(aim) + [0, 0, 0.0047], DOWN, L), DOWN, "t")
    sc.save_pass_layout(scan)
    sc.compute_correction(scan)
    predicted = sc.pass_target_tcp(scan, 2, "goal", TILTED, sc.load_correction(), "measured_layout")[0]
    true_p2 = predicted - np.array([0.0, 0.0, 0.003])              # P2 at its own tilt sits 3 mm lower
    sc.save_reference(2, "goal", true_p2, TILTED, "P2 orientation")
    sc.compute_correction(scan)
    np.testing.assert_allclose(sc.pass_target_tcp(scan, 2, "goal", TILTED, sc.load_correction(), "measured_layout")[0],
                               true_p2, atol=1e-9)


def test_design_layout_places_the_beads_from_the_root_toward_the_90_degree_face(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))   # bevel on -w, w = +X here
    _tcp1, aim1 = sc.pass_target_tcp(scan, 1, "goal", DOWN, None, "design_layout")
    _tcp2, aim2 = sc.pass_target_tcp(scan, 2, "goal", DOWN, None, "design_layout")
    np.testing.assert_allclose(aim2 - aim1, [0.002, 0.0, 0.005], atol=1e-9)        # P2: 2 mm to the face, 5 up
    (tmp_path / "teaching").mkdir()
    (tmp_path / "teaching" / "pass_design_layout.yaml").write_text("passes_mm:\n  2: [0.0, -4.0, 6.5]\n")
    _tcp2, aim2 = sc.pass_target_tcp(scan, 2, "goal", DOWN, None, "design_layout")
    np.testing.assert_allclose(aim2 - aim1, [-0.004, 0.0, 0.0065], atol=1e-9)
    assert sc.RELATIVE_MODELS[0] == "design_layout"


@pytest.mark.parametrize("bevel", [-1, 1])
@pytest.mark.parametrize("tip", [L, np.array([0.002, -0.019, 0.029])])
def test_pass2_anchor_recovers_root_and_reproduces_both_measured_ends(tmp_path, monkeypatch, bevel, tip):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    monkeypatch.setattr(sc, "wire_tip_model", lambda _length: tip)
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    est = scan / "hongik_estimate" / "estimate.json"
    est.write_text(json.dumps(dict(json.loads(est.read_text()), bevel_sign_w=bevel)))
    offset = np.array([0.0, -bevel * 0.002, 0.005])
    true_root, tcp2 = {}, {}
    for end, root in (("start", [0.016, -1.541, 0.558]), ("goal", [0.009, -1.739, 0.557])):
        true_root[end] = np.array(root)
        tcp2[end] = tcp_from_aim(true_root[end] + FRAME.to_world(offset), TILTED, tip)
        # Labels are reversed: only positions determine which end the reference belongs to.
        label = "goal" if end == "start" else "start"
        sc.save_reference(2, label, tcp2[end], TILTED, "operator P2")
        # Previous root XYZ is deliberately wrong. Its attitude remains useful.
        sc.save_reference(1, label, true_root[end] + [0.01, 0, 0.02], DOWN, "old root")
    doc = sc.compute_correction(scan, anchor_pass=2)
    assert doc["anchor_pass"] == 2
    assert doc["anchor_references"]["start"]["source"] == "operator P2"
    assert doc["anchor_references"]["start"]["saved_as"] == "goal"
    correction = sc.load_correction()
    for end in sc.ENDS:
        _, root = sc.pass_target_tcp(scan, 1, end, DOWN, correction, "design_layout")
        np.testing.assert_allclose(root, true_root[end], atol=1e-12)
        for model in ("design_layout", "hongik_aim", "hongik_tcp"):
            actual_tcp, _ = sc.pass_target_tcp(scan, 2, end, TILTED, correction, model)
            np.testing.assert_allclose(actual_tcp, tcp2[end], atol=1e-12)
        # Different execution attitude must preserve the bead centre, not the measured TCP XYZ.
        _, aim2 = sc.pass_target_tcp(scan, 2, end, DOWN, correction, "design_layout")
        np.testing.assert_allclose(FRAME.to_local(aim2 - root), offset, atol=1e-12)
    # Recompute and TCP-model refresh must retain the selected anchor, never revert to old root XYZ.
    assert sc.compute_correction(scan)["anchor_pass"] == 2
    monkeypatch.setattr(sc, "wire_tip_model", lambda _length: np.array([0.001, -0.018, 0.030]))
    assert sc.refresh_correction_for_wire_tip()["anchor_pass"] == 2
    refreshed = sc.load_correction()
    for end in sc.ENDS:
        np.testing.assert_allclose(sc.pass_target_tcp(scan, 2, end, TILTED, refreshed, "design_layout")[0],
                                   tcp2[end], atol=1e-12)


def test_pass2_anchor_needs_both_ends_and_can_work_without_old_root_references(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    sc.save_reference(2, "start", (0.01, -1.54, 0.58), TILTED, "new P2")
    with pytest.raises(ValueError, match="pass 2 start and goal"):
        sc.compute_correction(scan, anchor_pass=2)
    assert not sc.correction_path().exists()
    sc.save_reference(2, "goal", (0.003, -1.738, 0.58), TILTED, "new P2")
    sc.compute_correction(scan, anchor_pass=2)
    prior = sc.correction_path().read_bytes()
    with pytest.raises(ValueError, match="pass 1 or pass 2"):
        sc.compute_correction(scan, anchor_pass=3)
    assert sc.correction_path().read_bytes() == prior


def test_pass2_calibration_snapshots_design_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    for end, y in (("start", -1.54), ("goal", -1.738)):
        sc.save_reference(2, end, (0.01, y, 0.58), TILTED, "P2")
    sc.compute_correction(scan, anchor_pass=2)
    correction = sc.load_correction()
    # A layout edit for a future calibration cannot alter this correction's known P2-to-root geometry.
    sc.design_layout_path().write_text("passes_mm:\n  2: [0.0, 20.0, 50.0]\n")
    _, root = sc.pass_target_tcp(scan, 1, "start", DOWN, correction, "design_layout")
    _, p2 = sc.pass_target_tcp(scan, 2, "start", TILTED, correction, "design_layout")
    np.testing.assert_allclose(FRAME.to_local(p2 - root), [0.0, 0.002, 0.005], atol=1e-12)
