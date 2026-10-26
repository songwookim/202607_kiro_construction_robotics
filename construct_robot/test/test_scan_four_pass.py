"""Four-pass teaching from the corrected scan estimate + taught pass orientations (no robot motion)."""

import json

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
from construct_robot.application import scan_four_pass as fp
from construct_robot.core.scan_reference import aim_from_tcp, tcp_from_aim
from construct_robot.io.teaching_yaml import read_pass_teaching_reference
from test_scan_reference import DOWN, L, TILTED, compose, quat_axis_angle

DZ = {1: 0.0, 2: 0.004, 3: 0.004, 4: 0.009}
AIM = {"start": np.array([0.005, -1.738, 0.563]), "goal": np.array([0.015, -1.539, 0.566])}
NAMES = [f"right_manipulator_joint{i}" for i in range(1, 7)]
ZERO = {"offsets_seam_m": {e: np.zeros(3) for e in ("start", "goal")}, "pass_deltas_seam_m": {},
        "reference_orientation_xyzw": DOWN}


def write_four_pass_estimate(tmp_path):
    out = tmp_path / "scan" / "hongik_estimate"
    (out / "kiro").mkdir(parents=True)
    (out / "estimate.json").write_text(json.dumps({"u": [0.0, -1.0, 0.0], "n_plate": [0.0, 0.0, 1.0], "bevel_sign_w": -1}))
    for k, dz in DZ.items():
        aims = {end: (AIM[end] + [0, 0, dz]).tolist() for end in AIM}
        pose = {f"current_{end}": {"position_m": dict(zip("xyz", map(float, tcp_from_aim(aims[end], TILTED, L)))),
                                   "orientation_xyzw": dict(zip("xyzw", map(float, TILTED)))} for end in AIM}
        (out / "kiro" / f"pass_{k}.yaml").write_text(yaml.safe_dump({"pass": k, **pose}))
        (out / "kiro" / f"pass_{k}_info.yaml").write_text(yaml.safe_dump({"derived": {"role": "root" if k == 1 else "fill",
            "frames": {"tcp_to_aim_point_mm": L * 1e3, "aim_point_World_m": aims}}}))
    return tmp_path / "scan"


@pytest.fixture
def teaching(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "TCP_TO_WIRE_TIP_M", None)       # the general mechanism (KIRO: 0)
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    monkeypatch.setattr(fp, "scan_teaching_dir", lambda: tmp_path / "teaching")
    monkeypatch.setattr(fp, "weld_feedback_dir", lambda: tmp_path / "weld_feedback")
    # the measured layout = the estimate's own pass heights (seam normal = World z here)
    monkeypatch.setattr(sc, "load_pass_layout",
                        lambda path=None: {k: {e: np.array([0.0, 0.0, DZ[k]]) for e in ("start", "goal")} for k in (2, 3, 4)})
    return tmp_path


Q13 = compose(quat_axis_angle((1, 0, 0), 10.0), DOWN)
Q2 = compose(quat_axis_angle((1, 0, 0), -15.0), DOWN)
Q4 = compose(quat_axis_angle((0, 1, 0), 5.0), DOWN)


def teach_all():
    seed = (NAMES, [0.1 * i for i in range(6)])
    fp.save_pass_orientation((1, 3), Q13, seed, (0, -1.7, 0.6), "I")
    fp.save_pass_orientation((2,), Q2, seed, (0, -1.7, 0.6), "O")
    fp.save_pass_orientation((4,), Q4, seed, (0, -1.7, 0.6), "P")
    return fp.load_pass_orientations()


def test_one_capture_serves_several_passes_and_only_orientation_is_kept(teaching):
    orientations = teach_all()
    assert sorted(orientations) == [1, 2, 3, 4]
    np.testing.assert_allclose(orientations[1]["quat"], orientations[3]["quat"])
    assert orientations[3]["captured_for"] == (1, 3) and orientations[2]["captured_for"] == (2,)
    fp.save_pass_orientation((2,), Q4, (NAMES, [0.0] * 6), (0, 0, 0), "O again")
    assert fp.load_pass_orientations()[2]["source"] == "O again"


def test_missing_orientations_are_refused_and_no_correction_uses_the_sensing(teaching):
    scan = write_four_pass_estimate(teaching)
    fp.save_pass_orientation((1, 3), Q13, (NAMES, [0.0] * 6), (0, 0, 0), "I")
    with pytest.raises(ValueError, match="pass\\(es\\) 2, 4"):
        fp.four_pass_targets(scan, ZERO, fp.load_pass_orientations())
    targets = fp.four_pass_targets(scan, None, teach_all())     # no correction: the sensed wire tips as they are
    np.testing.assert_allclose(aim_from_tcp(targets[1]["start"][0], targets[1]["start"][1], L), AIM["start"], atol=1e-12)


def test_pass_targets_keep_the_wire_tip_and_waits_sit_above_pass_4(teaching):
    scan = write_four_pass_estimate(teaching)
    orientations = teach_all()
    targets = fp.four_pass_targets(scan, ZERO, orientations, relative="hongik_aim")   # Hongik's wire-tip layout
    for k, row in targets.items():
        q = orientations[k]["quat"]
        for end in ("start", "goal"):
            tcp, quat = row[end]
            np.testing.assert_allclose(quat, q)
            np.testing.assert_allclose(aim_from_tcp(tcp, q, L), AIM[end] + [0, 0, DZ[k]], atol=1e-12)
            wait, wq = row[f"{end}_wait"]
            np.testing.assert_allclose(wq, q)
            if k in fp.WAIT_OFFSET_FROM_WELD_M:           # pass 2: 3 mm -X, 3 mm +Z of its own weld point
                np.testing.assert_allclose(wait, row[end][0] + [-0.003, 0, 0.003], atol=1e-12)
            else:
                np.testing.assert_allclose(aim_from_tcp(wait, q, L), AIM[end] + [0, 0, DZ[4] + 0.010], atol=1e-12)
        for pose_name, wait in (("weld_wait", "start_wait"), ("weld_finish", "goal_wait")):
            tcp, quat = row[pose_name]
            np.testing.assert_allclose(quat, q)                               # each pass's own orientation
            np.testing.assert_allclose(tcp, row[wait][0] + [0, 0, 0.060], atol=1e-12)


def test_written_pass_files_load_as_4_pass_teaching(teaching):
    scan = write_four_pass_estimate(teaching)
    orientations = teach_all()
    folder, targets = fp.write_four_pass_teaching(scan, ZERO, orientations)
    assert folder == teaching / "weld_feedback" / "scan_four_pass_scan" and fp.latest_four_pass_dir() == folder
    for k in fp.PASSES:
        ref = read_pass_teaching_reference(folder / f"pass_{k}.yaml", k)
        assert ref["requires_ik"] is True and ref["reference_kind"] == "saved_pass_teaching"
        for endpoint in ("start_wait", "start", "goal_wait", "goal"):
            p, o = ref[endpoint].position, ref[endpoint].orientation
            np.testing.assert_allclose((p.x, p.y, p.z), targets[k][endpoint][0])
            np.testing.assert_allclose((o.x, o.y, o.z, o.w), orientations[k]["quat"])
            assert ref["joint_states"][endpoint] == (tuple(NAMES), tuple(orientations[k]["joints"][1]))
    assert (sc.history_dir(scan) / "four_pass" / "pass_4.yaml").is_file()
    p4 = read_pass_teaching_reference(folder / "pass_4.yaml", 4)
    assert p4["weave_geometry"]["reference"] == "scan_plate_plane"
    np.testing.assert_allclose(p4["weave_geometry"]["normal_world"], [0, 0, 1])
    (folder / "npz").mkdir()
    (folder / "npz" / "initial_scan.npz").write_bytes(b"raw scan must stay out of teaching history")
    (folder / "cycle.log").write_text("immutable execution log")
    fp.write_four_pass_teaching(scan, ZERO, orientations)
    assert sorted(p.name for p in (sc.history_dir(scan) / "four_pass").iterdir()) == [
        f"pass_{k}.yaml" for k in range(1, 5)]
    assert (folder / "cycle.log").read_text() == "immutable execution log"


def test_plate_weave_uses_the_measured_plane_normal_before_seam_orthogonalization(teaching):
    scan = write_four_pass_estimate(teaching)
    path = scan / "hongik_estimate" / "estimate.json"
    doc = json.loads(path.read_text())
    doc["n_plate"] = [0.1, 0.2, 1.0]
    path.write_text(json.dumps(doc))
    detail = sc.load_estimate_detail(scan)
    normal = np.array(doc["n_plate"]) / np.linalg.norm(doc["n_plate"])
    assert not np.allclose(normal, detail["frame"].n)
    p4 = fp.four_pass_targets(scan, ZERO, teach_all())[4]
    direction = np.asarray(p4["weave_geometry"]["transverse_vector_world"])
    np.testing.assert_allclose(p4["weave_geometry"]["normal_world"], normal)
    assert abs(direction @ normal) < 1e-12
    assert abs(direction @ (p4["goal"][0] - p4["start"][0])) < 1e-12


def test_pass2_origin_calibration_propagates_design_to_all_four_pass_files(teaching):
    scan = write_four_pass_estimate(teaching)
    orientations = teach_all()
    for end in ("start", "goal"):
        # Stale P1/P4 positions must not override a newly calibrated groove origin.
        sc.save_reference(1, end, AIM[end] + [0.01, 0, 0.02], Q13, "old root")
        sc.save_reference(4, end, AIM[end] + [0.02, 0, 0.03], Q4, "old cap")
        tip2 = AIM[end] + [0.003, 0, -0.008] + [0.002, 0, 0.005]
        sc.save_reference(2, end, tcp_from_aim(tip2, Q2, L), Q2, "P2 fitted to specimen")
    sc.compute_correction(scan, anchor_pass=2)
    correction = sc.load_correction()
    folder, targets = fp.write_four_pass_teaching(scan, correction, orientations, out_dir=teaching / "calibrated")
    expected_design = {1: [0, 0, 0], 2: [0.002, 0, 0.005], 3: [-0.004, 0, 0.0065], 4: [0, 0, 0.010]}
    for k in fp.PASSES:
        written = yaml.safe_load((folder / f"pass_{k}.yaml").read_text())
        assert written["generated_from"]["position_anchor_pass"] == 2
        assert written["requires_ik"] is True
        for end in ("start", "goal"):
            tcp, q = targets[k][end]
            np.testing.assert_allclose(aim_from_tcp(tcp, q, L),
                                       AIM[end] + [0.003, 0, -0.008] + expected_design[k], atol=1e-12)
            np.testing.assert_allclose(q, orientations[k]["quat"])


def test_estimated_moves_use_the_taught_pass_orientation(teaching):
    scan = write_four_pass_estimate(teaching)
    orientations = teach_all()
    current = Pose()
    current.orientation.w = 1.0
    target, info = sc.estimated_move_target(scan, "Pass 2 goal", current, ZERO, orientations=orientations)
    o = target.orientation
    assert info["pass_orientation"]
    np.testing.assert_allclose((o.x, o.y, o.z, o.w), orientations[2]["quat"])
    p = target.position
    np.testing.assert_allclose(aim_from_tcp((p.x, p.y, p.z), orientations[2]["quat"], L), AIM["goal"] + [0, 0, DZ[2]],
                               atol=1e-12)
    _target, info = sc.estimated_move_target(scan, "Pass 2 goal", current, ZERO, orientations={})
    assert not info["pass_orientation"]


def test_orientation_angles_are_reported_in_the_seam_frame(teaching):
    scan = write_four_pass_estimate(teaching)
    angles = fp.orientation_angles(scan, teach_all())
    assert set(angles) == {1, 2, 3, 4}
    assert angles[1] == angles[3] and angles[1] != angles[2]


def test_generated_files_load_in_4_pass_and_apply_resolves_every_pose_by_ik(teaching, monkeypatch):
    from test_multipass_controller import InlineThread, make_host
    from construct_robot.application import multipass_controller as controller_module
    scan = write_four_pass_estimate(teaching)
    orientations = teach_all()
    folder, targets = fp.write_four_pass_teaching(scan, ZERO, orientations, out_dir=teaching / "fp")
    host, controller = make_host(folder)
    assert controller.load_references()                     # "Load 4 references"
    assert not any(kind == "error" for kind, *_ in host.calls)
    monkeypatch.setattr(controller_module.threading, "Thread", InlineThread)
    host.selected = 2
    host.linear_tcp_endpoints = [None, None]
    host._invalidate_seam_correction_runtime = lambda *a, **k: None
    host._seam_reference_yaml_path = lambda group: teaching / "seam_ref.yaml"
    host._initial_state_yaml_path = lambda group, name: teaching / f"init_{name}.yaml"
    host.teaching_pose_changed = lambda: None
    controller.load_saved_teaching_for_selected_pass()
    (ik_targets,), _ = host.node.resolve_tcp_joint_states.call_args
    by_name = {t[4]: t for t in ik_targets}
    assert set(by_name) == {"weld_start_wait", "weld_start", "weld_goal_wait", "weld_end", "weld_wait", "weld_finish"}
    for name, key in (("weld_wait", "weld_wait"), ("weld_finish", "weld_finish"), ("weld_start", "start")):
        p = by_name[name][2].position
        np.testing.assert_allclose((p.x, p.y, p.z), targets[2][key][0])
    assert by_name["weld_wait"][0] == "start" and by_name["weld_finish"][0] == "goal"
    for name, target in by_name.items():                      # IK starts from the file's joints (arm branch)
        assert target[5] == tuple(orientations[2]["joints"][1])


def test_seeded_ik_refuses_another_arm_branch_and_retries_from_the_current_joints(monkeypatch):
    import math
    from types import SimpleNamespace
    from unittest.mock import Mock
    from builtin_interfaces.msg import Time
    from construct_robot.nodes import weld_runtime_node as node_module
    monkeypatch.setattr(node_module.rclpy, "ok", lambda: True)
    stored = (4.95, 1.25, 1.38, 5.11, 0.10, -0.25)              # the vertical capture used as the seed
    working = (5.24, 1.25, 1.92, 3.79, 1.18, 1.26)               # the tilted pose the arm is in (2026-10-15 log)
    flipped = (-1.16, 2.92, -1.73, -3.76, 5.34, 1.87)            # the elbow-flipped weld_wait solution

    def run(solution, current):
        posted, seeds = [], []
        response = SimpleNamespace(error_code=SimpleNamespace(val=1), solution=SimpleNamespace(
            joint_state=SimpleNamespace(name=list(NAMES), position=list(solution))))

        def call_async(request):
            seeds.append(tuple(request.ik_request.robot_state.joint_state.position))
            return SimpleNamespace(add_done_callback=lambda cb: cb(SimpleNamespace(result=lambda: response)))
        host = SimpleNamespace(ik_client=SimpleNamespace(wait_for_service=lambda timeout_sec: True, call_async=call_async),
                               get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)),
                               get_logger=Mock(), latest_joint_positions=dict(zip(NAMES, current)),
                               ui=SimpleNamespace(post=lambda fn, *a: posted.append((fn, a)),
                                                  apply_corrected_tcp_joint_state="apply",
                                                  corrected_tcp_joint_state_failed="failed"))
        host._solve_tcp_ik = lambda *a: node_module.WeldGuiNode._solve_tcp_ik(host, *a)
        node_module.WeldGuiNode.resolve_tcp_joint_state(host, "right_manipulator", Pose(), NAMES, "goal", "weld_finish",
                                                         stored)
        return posted, seeds
    posted, seeds = run(flipped, working)                         # flipped from both seeds: refused, not saved
    assert posted == [("failed", posted[0][1])] and posted[0][1][2].count("another arm branch") == 2
    assert seeds == [stored, working]
    posted, seeds = run(working, working)                         # the wrist turns ~80-90 deg from the stored seed
    assert posted[0][0] == "apply" and seeds == [stored]
    turned = (working[0] - 2 * math.pi,) + working[1:]           # KDL: the same pose one turn away on joint 1
    posted, seeds = run(turned, working)
    assert posted[0][0] == "apply" and posted[0][1][4][0] == pytest.approx(working[0])
    far = tuple(v + math.radians(70) for v in working[:3]) + working[3:]
    posted, seeds = run(far, far)                                 # elbow 70 deg off the stored seed -> retry, accepted
    assert posted[0][0] == "apply" and seeds == [stored, far]


def test_corrected_named_pose_ik_keeps_the_wrist_near_the_stored_joints(monkeypatch):
    """KDL answered the P2 weld_wait with the flipped wrist (281 deg from the stored joints) -> no plan (2026-10-15)."""
    import math
    from types import SimpleNamespace
    from construct_robot.nodes import weld_runtime_node as node_module
    stored = (4.95, 1.25, 1.37, 5.115, 0.107, -0.246)
    flipped = (4.96 - 2 * math.pi, 1.22, 1.79 - 2 * math.pi, 0.20 - 2 * math.pi, -1.02, 4.81)
    upright = (4.96, 1.22, 1.79, 3.35, 1.02, 1.67)

    def solve(step, _pose):                                      # the wrist-turned seeds reach the upright wrist
        return upright if step["positions"][4] < 0 else flipped
    host = SimpleNamespace(latest_joint_positions={}, _cleaner_corrected_joint_target=solve)
    step = {"positions": stored, "joint_names": [f"right_manipulator_joint{i}" for i in range(1, 7)]}
    chosen = node_module.WeldGuiNode._closest_corrected_joint_target(host, step, None)
    assert chosen == pytest.approx(upright)
    host._cleaner_corrected_joint_target = lambda step, _pose: flipped
    with pytest.raises(RuntimeError, match="re-teach"):
        node_module.WeldGuiNode._closest_corrected_joint_target(host, step, None)
