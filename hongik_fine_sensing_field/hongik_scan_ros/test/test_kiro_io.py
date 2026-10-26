"""hongik_scan_ros.kiro_io 단위 시험 (ROS 실행 없음): KIRO PrecisionSensing goal → 실행 입력, 끝 상태·분류 → error_code·메시지 (r3 §4.2 표 전체),
4패스 계획 → WeldingPath 자세 (+Z 단면 판 법선 토치 쪽, +X 다음 점 투영, 마지막 = 앞), 간격, precision_sensing.npz 왕복 (키·값·원자적 쓰기).
welding_interfaces 가 있으면 상수·메시지 형식도 그 생성 타입과 대조한다."""
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STACK = os.path.dirname(PKG)
for p in (PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"), os.path.join(STACK, "scan_core", "tests"),
          os.path.join(STACK, "vc3dz_driver")):
    if p not in sys.path:
        sys.path.insert(0, p)

from g1_synth import synth_estimate, synth_prior  # noqa: E402
from hongik_scan_ros import kiro_io as K  # noqa: E402
from scan_core.g1_outputs import write_outputs  # noqa: E402
from scan_core.weld_plan import WeldProcess  # noqa: E402

FAKE_CFG = os.path.join(STACK, "scan_l0", "tests", "data", "hongik_rb20_fake_moveit.yaml")


def _pose(x, y, z, q=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=z), orientation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3]))


def _goal(n, a=(0.69, 0.03, -0.03), b=(1.01, -0.03, -0.03), qa=(0, 0, 0, 1), qb=(0, 0, 0, 1)):
    return SimpleNamespace(welding_position=n, pose_a=_pose(*a, q=qa), pose_b=_pose(*b, q=qb))


# --------------------------------------------------------------------------------------
# goal
# --------------------------------------------------------------------------------------
def test_goal_1g_becomes_run_inputs_with_the_positions_as_rough_points():
    i1 = K.goal_to_run_inputs(_goal(1), g1_pipeline="core")
    assert (i1.welding_position, i1.position, i1.start_m, i1.end_m) == (1, "1G", (0.69, 0.03, -0.03), (1.01, -0.03, -0.03))
    assert K.SUPPORTED_POSITIONS == {1: "1G"}                                  # this package: the 1G core only
    # orientation is not used (KIRO Q2): any quaternion gives the same inputs
    q = (0.3, -0.2, 0.1, math.sqrt(1 - 0.14))
    assert K.goal_to_run_inputs(_goal(1, qa=q, qb=q), g1_pipeline="core") == i1
    assert any("orientations are not used" in n for n in i1.notes) and any("start pose = the arm pose at goal acceptance" in n
                                                                           for n in i1.notes)


@pytest.mark.parametrize("n,label", [(2, "2G"), (3, "3G"), (4, "4G"), (5, "5G"), (6, "6G"), (11, "1F"), (12, "2F"), (13, "3F"), (14, "4F"),
                                     (0, "not a PrecisionSensing position"), (7, "not a PrecisionSensing position"),
                                     (255, "not a PrecisionSensing position")])
def test_unsupported_welding_positions_are_invalid_goals_with_a_concrete_message(n, label):
    with pytest.raises(K.InvalidGoal) as e:
        K.goal_to_run_inputs(_goal(n), g1_pipeline="core")
    assert str(e.value) == f"welding_position {n} ({label}) not supported: 1G (core) only"
    oc = K.refused_outcome(f"{e.value} (nothing sent)", "input")
    assert (oc.action, oc.error_code, oc.deliver_paths) == ("abort", K.ERROR_INVALID_GOAL, False)
    assert oc.error_message == f"REFUSED: welding_position {n} ({label}) not supported: 1G (core) only (nothing sent)"


@pytest.mark.parametrize("bad", [(float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0)])
def test_non_finite_positions_are_invalid_goals(bad):
    with pytest.raises(K.InvalidGoal, match="pose_b.position is not finite"):
        K.goal_to_run_inputs(_goal(1, b=bad), g1_pipeline="core")


def test_rough_goal_builds_the_job_from_the_job_file_only():
    """The KIRO goal carries no design values: build_rough_job takes the specimen / tolerances / process from the job file and the
    points from pose_a / pose_b (base frame)."""
    from hongik_scan_ros.auto_teach_server import build_rough_job
    from scan_l0 import config as C
    inp = K.goal_to_run_inputs(_goal(1), g1_pipeline="core")
    job_file = str(C.JOBS_DIR / "groove_10T_1G.yaml")
    job, doc = build_rough_job(None, job_file, K.rough_goal(inp), base_frame="link0", gravity_axis_base=[0.0, 0.0, -1.0],
                               start=np.array(inp.start_m), end=np.array(inp.end_m))
    _, doc0 = C.load_job(job_file)
    assert np.allclose(job.start, inp.start_m) and np.allclose(job.end, inp.end_m) and job.position == "1G"
    assert doc["specimen"] == dict(doc0["specimen"]) and doc.get("qc_targets") == doc0.get("qc_targets")


# --------------------------------------------------------------------------------------
# result codes (r3 §4.2)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("status,category,code", [
    ("REFUSED", "input", 5), ("REFUSED", "config", 255), ("REFUSED", "motion", 1), ("REFUSED", None, 1), ("REFUSED", "sensor", 2),
    ("ABORTED", None, 1), ("ABORTED", "motion", 1), ("ABORTED", "sensor", 2),
    ("DATA_FAIL", None, 3), ("DATA_FAIL", "sensing", 3), ("DATA_FAIL", "waypoint", 4), ("ERROR", None, 255), ("ERROR", "sensing", 3),
    ("ERROR", "internal", 255), ("SPEC_FAIL", None, 3), ("ALIGN_FAIL", None, 3)])
def test_failure_table(status, category, code):
    oc = K.run_outcome(status, "the concrete reason (nothing sent)", category=category)
    assert (oc.action, oc.error_code, oc.deliver_paths) == ("abort", code, False)
    assert oc.error_message == f"{status}: the concrete reason (nothing sent)"
    assert oc.category == (category or K.STATUS_DEFAULT_CATEGORY[status])


def test_model_misfit_is_sensing_failed_without_paths():
    """§4.9: a run whose done-test quantity rests on a segment that does not fit the fused model ends DATA_FAIL / 'sensing' with a
    MODEL_MISFIT reason → KIRO SENSING_FAILED, no paths, the reason in the message."""
    oc = K.run_outcome("DATA_FAIL", "MODEL_MISFIT after the pass budget 4: station window s=180.0 mm: this seam in the window: 2 ln BF "
                                    "-812.0 < -k^2 (31 profiles) (affects root_lateral, root_normal)", category="sensing")
    assert (oc.action, oc.error_code, oc.deliver_paths) == ("abort", K.ERROR_SENSING_FAILED, False)
    assert oc.error_message.startswith("DATA_FAIL: MODEL_MISFIT") and "station window s=180.0 mm" in oc.error_message


def test_unidentified_end_is_sensing_failed_without_paths():
    """User 2026-10-11 ('identification decides'): an end that no measurement identified at the end of the budget → DATA_FAIL /
    'sensing' → SENSING_FAILED, no paths; an identified joint with a quantity above tolerance stays PASS_UNRESOLVED → SUCCESS + paths."""
    oc = K.run_outcome("DATA_FAIL", "end(s) E not identified by the measurements after the pass budget 4: every pass (pass1, pass2, "
                                    "pass3, pass4) does not show this seam - no weld paths (sections of what was measured are written)",
                       category="sensing")
    assert (oc.action, oc.error_code, oc.deliver_paths) == ("abort", K.ERROR_SENSING_FAILED, False)
    assert "not identified" in oc.error_message
    un = K.run_outcome("PASS_UNRESOLVED", "pass budget 4 used; unresolved: ['end_S'] (max k sigma/tol 1.06)")
    assert (un.action, un.error_code, un.deliver_paths) == ("succeed", K.SUCCESS, True)


def test_success_unresolved_and_cancel():
    ok = K.run_outcome("PASS", "done after 2 pass(es): every k sigma <= tol")
    assert (ok.action, ok.error_code, ok.deliver_paths, ok.error_message) == (
        "succeed", 0, True, "PASS: done after 2 pass(es): every k sigma <= tol")
    # PASS_UNRESOLVED (D13): SUCCESS + paths + a message that lists the unresolved quantities (Teach1G's reason) until KIRO answers Q4
    un = K.run_outcome("PASS_UNRESOLVED", "pass budget 4 used; unresolved: ['end_E', 'gap_mm']")
    assert (un.action, un.error_code, un.deliver_paths) == ("succeed", K.SUCCESS, True)
    assert un.error_message.startswith("PASS_UNRESOLVED: ") and "end_E" in un.error_message and "gap_mm" in un.error_message
    # cancel: canceled with the temporary code 255, whatever the status the run ended with
    for st in ("ABORTED", "PASS"):
        c = K.run_outcome(st, "cancel requested by the operator (D7): move 'transit_S_to_E'", cancelled=True)
        assert (c.action, c.error_code, c.deliver_paths) == ("canceled", K.CANCEL_CODE, False) and K.CANCEL_CODE == 255


def test_no_mapping_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="DRY_RUN"):
        K.run_outcome("DRY_RUN", "dry run")
    with pytest.raises(ValueError, match="not one of"):
        K.run_outcome("REFUSED", "x", category="other")
    with pytest.raises(ValueError, match="no PrecisionSensing result mapping"):
        K.run_outcome("SOMETHING_NEW", "x")
    assert set(K.CATEGORY_CODES) == set(K.CATEGORIES)


def test_codes_match_the_welding_interfaces_action():
    act = pytest.importorskip("welding_interfaces.action")
    R = act.PrecisionSensing.Result
    assert (R.SUCCESS, R.ERROR_MOTION_FAILED, R.ERROR_SENSOR_FAILED, R.ERROR_SENSING_FAILED, R.ERROR_WAYPOINT_GENERATION_FAILED,
            R.ERROR_INVALID_GOAL, R.ERROR_INTERNAL) == (K.SUCCESS, K.ERROR_MOTION_FAILED, K.ERROR_SENSOR_FAILED, K.ERROR_SENSING_FAILED,
                                                       K.ERROR_WAYPOINT_GENERATION_FAILED, K.ERROR_INVALID_GOAL, K.ERROR_INTERNAL)
    G = act.PrecisionSensing.Goal
    assert {n: getattr(G, f"POSITION_{lab}") for n, lab in K.POSITION_NAMES.items()} == {n: n for n in K.POSITION_NAMES}
    g = G()
    g.welding_position = 1
    g.pose_a.position.x, g.pose_b.position.x = 0.7, 1.0
    assert K.goal_to_run_inputs(g, g1_pipeline="core").start_m == (0.7, 0.0, 0.0)


# --------------------------------------------------------------------------------------
# WeldingPath frames
# --------------------------------------------------------------------------------------
def _check_frames(P, Z, R):
    N = P.shape[0]
    for i in range(N):
        assert np.allclose(R[i].T @ R[i], np.eye(3), atol=1e-12) and np.linalg.det(R[i]) == pytest.approx(1.0)
        assert np.allclose(np.cross(R[i][:, 2], R[i][:, 0]), R[i][:, 1], atol=1e-12)            # +Y = Z x X
    for i in range(N - 1):
        z = Z[i] / np.linalg.norm(Z[i])
        assert np.allclose(R[i][:, 2], z, atol=1e-12)
        d = P[i + 1] - P[i]
        x = d - (d @ z) * z
        assert np.allclose(R[i][:, 0], x / np.linalg.norm(x), atol=1e-12)
    assert np.array_equal(R[N - 1], R[N - 2])


def test_waypoint_frames_on_a_curved_tilted_path():
    s = np.linspace(0.0, 0.3, 61)
    P = np.stack([0.7 + s, 0.02 * np.sin(8 * s), -0.05 + 0.1 * s], axis=1)              # wavy, climbing
    Z = np.tile(np.array([0.0, -0.1, 1.0]), (s.size, 1))                               # not unit, not perpendicular to the path
    R = K.waypoint_frames(P, Z)
    _check_frames(P, Z, R)
    q = K.Rotation.from_matrix(R).as_quat()
    assert np.allclose(K.Rotation.from_quat(q).as_matrix(), R, atol=1e-12)


def test_waypoint_frames_refuse_undefined_directions():
    with pytest.raises(ValueError, match=">= 2 waypoints"):
        K.waypoint_frames(np.zeros((1, 3)), np.array([[0, 0, 1.0]]))
    P = np.array([[0, 0, 0], [0, 0, 0.005], [0.005, 0, 0.005]], dtype=float)      # step 0 -> 1 along +Z only
    with pytest.raises(ValueError, match="waypoints 0 and 1"):
        K.waypoint_frames(P, np.tile([0, 0, 1.0], (3, 1)))


@pytest.fixture(scope="module")
def g1_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("g1_out")
    pr = synth_prior()
    est = synth_estimate(pr)
    res = write_outputs(est, pr, WeldProcess(), out, k=pr.k, base_frame="link0")
    return SimpleNamespace(out=out, plan=res["plan"], prior=pr, est=est)


def test_plan_paths_are_the_pass_lines_in_the_documented_weldingpath_form(g1_run):
    """result.paths[k-1] = the straight line of kiro/pass_<k>.yaml (start -> goal, torch tip) in WeldingPath's documented form: robot
    base frame, evenly <= 5 mm (station step) in the welding order, +Z = the pass's plate normal (torch side), +X = along the line,
    constant; first / last waypoint = the pass_<k>.yaml start / goal positions in the base frame."""
    from scan_core.kiro_plan import pass_endpoints
    plan = g1_run.plan
    paths = K.plan_paths(plan)
    assert [p.k for p in paths] == [1, 2, 3, 4] == [wp.k for wp in plan.passes]
    n_mean = np.mean([st.n for st in plan.stations], axis=0)
    n_mean /= np.linalg.norm(n_mean)
    from scan_core.weld_line import weld_line
    for pa, wp in zip(paths, plan.passes):
        e = pass_endpoints(plan, wp)
        assert np.allclose(pa.pos_m[0], e["start"]["tip_base_m"], atol=1e-15) and np.allclose(pa.pos_m[-1], e["goal"]["tip_base_m"], atol=1e-15)
        ln = weld_line(plan, wp)                                         # THE delivered line (D14 checks it): its wire-end line
        assert np.array_equal(pa.pos_m[0], ln.tip_start) and np.allclose(pa.pos_m[-1], ln.tip_goal, atol=1e-15)
        line = pa.pos_m[-1] - pa.pos_m[0]
        L = float(np.linalg.norm(line))
        steps = np.linalg.norm(np.diff(pa.pos_m, axis=0), axis=1)
        assert np.allclose(steps, steps[0], rtol=1e-9) and steps.max() <= 0.005 + 1e-12 and pa.pos_m.shape[0] == math.ceil(L * 1e3 / 5.0 - 1e-9) + 1
        off = (pa.pos_m - pa.pos_m[0]) - np.outer((pa.pos_m - pa.pos_m[0]) @ (line / L), line / L)
        assert np.max(np.linalg.norm(off, axis=1)) < 1e-12                               # a straight line
        ds = np.diff(pa.s_mm)
        assert (np.all(ds > 0) if wp.direction == "S->E" else np.all(ds < 0)), wp.direction
        assert pa.s_mm[0] == pytest.approx(e["start"]["s_mm"]) and pa.s_mm[-1] == pytest.approx(e["goal"]["s_mm"])
        _check_frames(pa.pos_m, np.tile(n_mean, (pa.pos_m.shape[0], 1)), pa.R)
        assert np.allclose(pa.R, pa.R[0], atol=1e-12) and np.allclose(pa.R[0][:, 0], line / L, atol=1e-6)
        assert float(pa.R[0][:, 2] @ n_mean) > 0.9999                                       # +Z = plate normal, torch side
        assert np.allclose(K.Rotation.from_quat(pa.quat_xyzw).as_matrix(), pa.R, atol=1e-9)


def test_paths_are_the_wire_tip_aim_line_and_the_yaml_tcp_is_stick_out_back_along_z_tcp(g1_run, tmp_path):
    """KIRO WeldingPath 정의 (welding_interfaces/msg/WeldingPath.msg: 'Every pose.position is the welding wire-tip TCP position'):
    result.paths = 와이어 끝 (비드 겨냥점) 선, 계획에 토치 모형이 있어도 그대로 (표면 틀 자세). kiro/pass_<k>.yaml = 토치 TCP (토치 팁 끝 중심)
    = 겨냥점 − (노즐 끝 + stick_out_mm) · z_TCP (돌출 = 노즐 끝에서, 2026-10-13): yaml TCP 위치 + (노즐 끝 + stick_out) · z_TCP = paths 의
    첫 / 끝 점 (사용자 2026-10-12 밤) — 두 TCP 롤 모두."""
    import dataclasses
    import yaml
    from scan_core.kiro_plan import kiro_output_config, resolve_world_from_base, write_kiro_outputs
    from scipy.spatial.transform import Rotation
    c = _kcfg("hongik_rb20_fake_moveit.yaml")                                 # the fixed log-fitted World <- link0 (not identity)
    tip_m = (_lab_nozzle_mm() + 20.0) * 1e-3                                  # TCP -> wire end: nozzle end + stick-out 20 mm
    wt, notes = resolve_world_from_base(c, "link0")
    T_bW = np.linalg.inv(np.block([[wt.R, wt.t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    for roll in (0.0, 180.0):
        plan = dataclasses.replace(g1_run.plan, torch=c.torch.with_stick_out(20.0, "test"), torch_roll_deg=roll)
        paths = K.plan_paths(plan)
        for pa, pa0 in zip(paths, K.plan_paths(g1_run.plan)):                    # the torch model does not move the wire-tip line
            assert np.array_equal(pa.pos_m, pa0.pos_m) and np.array_equal(pa.R, pa0.R)
        out = tmp_path / f"roll{roll:.0f}"
        ko = write_kiro_outputs(plan, out, c, world_from_base=wt, notes=notes)
        assert ko["stick_out_mm"] == 20.0 and "aim-point (wire-tip) lines" in ko["tcp_definition"]
        assert ko["nozzle_end_mm"] == pytest.approx(1.2, abs=1e-3) and ko["tcp_to_aim_point_mm"] == pytest.approx(tip_m * 1e3)
        for pa in paths:
            y = yaml.safe_load((out / "kiro" / f"pass_{pa.k}.yaml").read_text(encoding="utf-8"))
            for idx, key in ((0, "current_start"), (-1, "current_goal")):
                p = np.array([y[key]["position_m"][c_] for c_ in "xyz"])
                R = Rotation.from_quat([y[key]["orientation_xyzw"][c_] for c_ in "xyzw"]).as_matrix()
                aim_W = p + tip_m * R[:, 2]                                       # yaml TCP + (nozzle + stick) x z_TCP = the aim point
                assert np.allclose((T_bW @ np.append(aim_W, 1.0))[:3], pa.pos_m[idx], atol=1e-12), (roll, pa.k, key)


def test_end_inset_moves_every_weld_plan_output_and_leaves_the_sensing_alone(tmp_path):
    """User 2026-10-12: the weld design input weld_design.end_inset_mm {start, goal} moves every layer's start and goal inward along
    the seam by those distances (start = where the pass begins in its welding direction) in every weld-plan output: weld_plan.json /
    .npz (dense path), kiro/pass_<k>.yaml, the info files (which record the measured ends and the inset) and the action result paths.
    Default 0 = start / goal exactly at the measured ends. Nothing to do with sensing: the same estimate gives the same sections and the
    same plan sections."""
    import yaml
    from scan_core.kiro_plan import kiro_output_config, pass_endpoints, resolve_world_from_base, write_kiro_outputs
    from scan_core.weld_plan import WeldDesign
    pr = synth_prior()
    est = synth_estimate(pr)
    c = _kcfg("hongik_lab_rb20_moveit.yaml")                                     # World := link0: the files are in the base frame
    tip_m = (_lab_nozzle_mm() + 20.0) * 1e-3                                     # TCP -> wire end: nozzle end + stick-out 20 mm
    proc = WeldProcess(direction="alternate")                                    # both welding directions
    runs = {}
    for tag, ins in (("zero", None), ("inset", {"start": 10.0, "goal": 6.0})):
        out = tmp_path / tag
        design = WeldDesign.from_doc({"end_inset_mm": ins}) if ins else None
        res = write_outputs(est, pr, proc, out, k=pr.k, base_frame="link0", design=design, torch=c.torch)
        wt, notes = resolve_world_from_base(c, "link0")
        write_kiro_outputs(res["plan"], out, c, world_from_base=wt, notes=notes)
        runs[tag] = (out, res["plan"])
    (o0, p0), (o1, p1) = runs["zero"], runs["inset"]
    # sensing untouched: same sections, same plan sections and aim points
    assert json.loads((o0 / "sections.json").read_text(encoding="utf-8")) == json.loads((o1 / "sections.json").read_text(encoding="utf-8"))
    j0, j1 = (json.loads((o / "weld_plan.json").read_text(encoding="utf-8")) for o in (o0, o1))
    assert j0["stations"] == j1["stations"] and [q["sections"] for q in j0["passes"]] == [q["sections"] for q in j1["passes"]]
    assert j0["weld_design"]["end_inset_mm"]["start"] == 0.0 and j0["weld_design"]["end_inset_mm"]["goal"] == 0.0
    assert (j1["weld_design"]["end_inset_mm"]["start"], j1["weld_design"]["end_inset_mm"]["goal"]) == (10.0, 6.0)
    assert any("weld line ends: every pass starts 10 mm and ends 6 mm inside" in ln for ln in p1.summary_lines())
    u = np.asarray(p0.ends["u"], dtype=float)
    z0, z1 = np.load(o0 / "weld_plan.npz"), np.load(o1 / "weld_plan.npz")
    paths0, paths1 = K.plan_paths(p0), K.plan_paths(p1)
    assert {wp.direction for wp in p1.passes} == {"S->E", "E->S"}
    for wp0, wp1, pa0, pa1, q0, q1 in zip(p0.passes, p1.passes, paths0, paths1, j0["passes"], j1["passes"]):
        k, sg = wp1.k, (1.0 if wp1.direction == "S->E" else -1.0)
        e0, e1 = pass_endpoints(p0, wp0), pass_endpoints(p1, wp1)
        # default 0: the line starts / ends at the measured ends (the layer's ends)
        assert e0["start"]["s_mm"] == e0["measured"]["start"]["s_mm"] and e0["goal"]["s_mm"] == e0["measured"]["goal"]["s_mm"]
        assert e1["measured"]["start"]["s_mm"] == e0["start"]["s_mm"] and e1["inset_mm"] == {"start": 10.0, "goal": 6.0}
        # weld_plan.npz: the dense path starts 10 mm and ends 6 mm inside (along the seam, in the pass's direction)
        P0, P1 = z0[f"p{k}_pos_m"], z1[f"p{k}_pos_m"]
        assert float((P1[0] - P0[0]) @ u) == pytest.approx(sg * 0.010, abs=1e-9)
        assert float((P1[-1] - P0[-1]) @ u) == pytest.approx(-sg * 0.006, abs=1e-9)
        assert np.allclose(z1[f"p{k}_tcp_pos_m"], P1 - tip_m * z1[f"p{k}_R"][:, :, 2], atol=1e-15)   # TCP = tip - 21.2 mm torch axis
        # weld_plan.json: the pass's weld line with the measured ends and the inset
        wl = q1["weld_line"]
        assert wl["start_s_mm"] == pytest.approx(wl["measured_ends"]["start"]["s_mm"] + sg * 10.0)
        assert wl["goal_s_mm"] == pytest.approx(wl["measured_ends"]["goal"]["s_mm"] - sg * 6.0)
        assert wl["end_inset_mm"] == {"start": 10.0, "goal": 6.0} and q0["weld_line"]["end_inset_mm"] == {"start": 0.0, "goal": 0.0}
        # kiro/pass_<k>.yaml: start / goal moved by the inset along the seam
        y0, y1 = (yaml.safe_load((o / "kiro" / f"pass_{k}.yaml").read_text(encoding="utf-8")) for o in (o0, o1))
        pos = lambda y, key: np.array([y[key]["position_m"][c_] for c_ in "xyz"])            # noqa: E731
        assert float((pos(y1, "current_start") - pos(y0, "current_start")) @ u) == pytest.approx(sg * 0.010, abs=1e-9)
        assert float((pos(y1, "current_goal") - pos(y0, "current_goal")) @ u) == pytest.approx(-sg * 0.006, abs=1e-9)
        assert y1["current_start"]["orientation_xyzw"] == y0["current_start"]["orientation_xyzw"]          # the inset moves, never turns
        # the info file records the measured ends and the inset
        info = yaml.safe_load((o1 / "kiro" / f"pass_{k}_info.yaml").read_text(encoding="utf-8"))["derived"]["straight_line"]
        assert info["end_inset_mm"]["start"] == 10.0 and info["end_inset_mm"]["goal"] == 6.0
        assert info["measured_ends"]["start"]["s_mm"] == pytest.approx(e0["start"]["s_mm"])
        assert info["measured_ends"]["start"]["end"] == ("S" if sg > 0 else "E")
        assert info["s_start_mm"] == pytest.approx(info["measured_ends"]["start"]["s_mm"] + sg * 10.0)
        assert info["length_mm"] == pytest.approx(float(np.linalg.norm(pos(y0, "current_goal") - pos(y0, "current_start"))) * 1e3 - 16.0,
                                                  abs=1e-6)
        # the action result paths: the wire-tip lines (first / last waypoint = the yaml TCP start / goal + stick_out x z_TCP, World := link0)
        from scipy.spatial.transform import Rotation
        zt = Rotation.from_quat([y1["current_start"]["orientation_xyzw"][c_] for c_ in "xyzw"]).as_matrix()[:, 2]
        assert np.allclose(pa1.pos_m[0], pos(y1, "current_start") + tip_m * zt, atol=1e-12)
        assert np.allclose(pa1.pos_m[-1], pos(y1, "current_goal") + tip_m * zt, atol=1e-12)
        assert float((pa1.pos_m[0] - pa0.pos_m[0]) @ u) == pytest.approx(sg * 0.010, abs=1e-9)
    readme = (o1 / "kiro" / "README.md").read_text(encoding="utf-8")
    assert "end inset (job weld_design.end_inset_mm): start 10 mm, goal 6 mm" in readme
    assert "end inset" not in (o0 / "kiro" / "README.md").read_text(encoding="utf-8")


def test_ends_from_goal_follow_the_run_s_end():
    """The run takes pose_a as its taught start, pose_b as its taught end; the outputs' S is the taught point the run reports
    (AutoTeachResult.s_taught_point: 1G core = the first-visited point, legacy = 'start')."""
    assert K.ends_from_goal("start") == {"S": "pose_a", "E": "pose_b"}
    assert K.ends_from_goal("end") == {"S": "pose_b", "E": "pose_a"}
    with pytest.raises(ValueError, match="s_taught_point"):
        K.ends_from_goal("first")
    from scan_l0.autoteach import AutoTeachResult
    assert AutoTeachResult().s_taught_point == "start" and AutoTeachResult().to_dict()["s_taught_point"] == "start"


@pytest.fixture(scope="module")
def g1_run_e_to_s(tmp_path_factory):
    out = tmp_path_factory.mktemp("g1_out_es")
    pr = synth_prior()
    est = synth_estimate(pr)
    res = write_outputs(est, pr, WeldProcess(direction="E->S"), out, k=pr.k, base_frame="link0")
    return SimpleNamespace(out=out, plan=res["plan"])


def test_welding_path_order_is_the_plan_weld_direction(g1_run, g1_run_e_to_s):
    """WeldingPath order = the plan's weld direction setting (default S->E: from the S end; E->S: from the E end) — recorded in the
    plan (weld_plan.json 'weld_direction') and in the npz meta."""
    for run, setting in ((g1_run, "S->E"), (g1_run_e_to_s, "E->S")):
        plan = run.plan
        wd = K.weld_direction_of(plan)
        assert wd["setting"] == setting and set(wd["passes"].values()) == {setting} and sorted(wd["passes"]) == [1, 2, 3, 4]
        doc = json.loads((run.out / "weld_plan.json").read_text())
        assert doc["weld_direction"]["setting"] == setting and setting in doc["weld_direction"]["options"]
        S_end, E_end = np.asarray(plan.ends["S_base_m"]), np.asarray(plan.ends["E_base_m"])
        first, last = (S_end, E_end) if setting == "S->E" else (E_end, S_end)
        for pa in K.plan_paths(plan):
            assert np.linalg.norm(pa.pos_m[0] - first) < np.linalg.norm(pa.pos_m[0] - last)
            assert np.linalg.norm(pa.pos_m[-1] - last) < np.linalg.norm(pa.pos_m[-1] - first)
            assert (np.all(np.diff(pa.s_mm) > 0) if setting == "S->E" else np.all(np.diff(pa.s_mm) < 0))
    assert K.weld_direction_of(None) is None
    oc = K.run_outcome("PASS", "done")
    m = K.npz_meta(outcome=oc, status="PASS", reason="done", welding_position=1, position="1G", base_frame="link0", k=3.0,
                   out_dir="/x", run_id="x", ends_from_goal=K.ends_from_goal("end"), weld_direction=K.weld_direction_of(g1_run_e_to_s.plan))
    assert m["ends_from_goal"] == {"S": "pose_b", "E": "pose_a"} and m["weld_direction"]["setting"] == "E->S"
    assert "S" in m["frames"] and "pose_a" in m["frames"]["S"] and "weld_direction" in m["waypoints"]["position"]


def test_welding_path_messages(g1_run):
    pytest.importorskip("welding_interfaces.msg")
    paths = K.plan_paths(g1_run.plan)
    msgs = K.welding_paths_msgs(paths)
    assert len(msgs) == 4
    for m, pa in zip(msgs, paths):
        P = np.array([[w.position.x, w.position.y, w.position.z] for w in m.waypoints])
        Q = np.array([[w.orientation.x, w.orientation.y, w.orientation.z, w.orientation.w] for w in m.waypoints])
        assert np.array_equal(P, pa.pos_m) and np.array_equal(Q, pa.quat_xyzw)


# --------------------------------------------------------------------------------------
# precision_sensing.npz
# --------------------------------------------------------------------------------------
def _meta(oc, out, n_paths, src):
    return K.npz_meta(outcome=oc, status="PASS", reason="done", welding_position=1, position="1G", base_frame="link0", k=3.0,
                      out_dir=str(out), run_id=os.path.basename(str(out)), inputs={"pose_a_m": [0.9, -0.2, 0.05]},
                      sections_source=src, unresolved=[], n_paths=n_paths, notes=["n"])


def test_npz_round_trip(g1_run):
    plan, out = g1_run.plan, g1_run.out
    paths = K.plan_paths(plan)
    sections, src = K.sections_arrays(out)
    assert src == "sections.npz"
    oc = K.run_outcome("PASS", "done")
    path = K.write_precision_npz(out, K.precision_npz_arrays(_meta(oc, out, len(paths), src), plan=plan, paths=paths, sections=sections))
    assert os.path.isabs(path) and path == str((out / K.NPZ_NAME).resolve())
    assert not [f for f in os.listdir(out) if f.startswith(".")]                       # atomic write leaves no temporary file
    assert {"sections.json", "sections.npz", "weld_plan.json", "weld_plan.npz", K.NPZ_NAME} <= set(os.listdir(out))
    d = K.load_precision_npz(path)
    m = d["meta"]
    assert m["schema"] == K.NPZ_SCHEMA and m["error_code"] == 0 and m["error_message"] == "PASS: done" and m["n_paths"] == 4
    assert m["base_frame"] == "link0" and m["sections_source"] == "sections.npz" and "orientation" in m["waypoints"]
    assert int(d["n_paths"]) == 4
    with np.load(out / "sections.npz", allow_pickle=False) as z:
        for k in z.files:
            assert np.array_equal(d[f"sec_{k}"], z[k], equal_nan=z[k].dtype.kind == "f"), k
    sj = json.loads((out / "sections.json").read_text())
    assert d["sec_summary"]["length_mm"] == sj["length_mm"] and "stations" not in d["sec_summary"]
    with np.load(out / "weld_plan.npz", allow_pickle=False) as wpz:
        for pa, wp in zip(paths, plan.passes):
            k = pa.k
            M, N = pa.pos_m.shape[0], len(wp.s_mm)
            assert np.array_equal(d[f"p{k}_wp_pos_m"], pa.pos_m) and np.array_equal(d[f"p{k}_wp_quat_xyzw"], pa.quat_xyzw)
            assert np.array_equal(d[f"p{k}_wp_s_mm"], pa.s_mm) and np.array_equal(d[f"p{k}_aim_pos_m"], np.asarray(wp.aim_base_m))
            assert np.array_equal(d[f"p{k}_s_mm"], wp.s_mm) and d[f"p{k}_R_torch"].shape == (N, 3, 3)
            assert np.array_equal(d[f"p{k}_R_torch"], np.asarray(wp.R_base_tcp))
            for key in ("speed_mps", "area_mm2", "layer_to_mm"):
                assert np.array_equal(d[f"p{k}_{key}"], np.asarray(getattr(wp, key), dtype=float))
            assert np.array_equal(d[f"p{k}_aim_sigma_m"], np.asarray(plan.aim_sigma_m[k]), equal_nan=True)
            if wp.kind == "cap":
                assert np.array_equal(d[f"p{k}_weave_amp_mm"], wp.weave_amp_mm) and np.all(d[f"p{k}_weave_amp_mm"] > 0)
            else:
                assert np.all(np.isnan(d[f"p{k}_weave_amp_mm"])) and np.all(np.isnan(d[f"p{k}_weave_width_mm"]))
            for a, b in (("dense_t_s", "t_s"), ("dense_pos_m", "pos_m"), ("dense_R", "R")):
                assert np.array_equal(d[f"p{k}_{a}"], wpz[f"p{k}_{b}"])
            info = d[f"p{k}_info"]
            assert (info["k"], info["role"], info["kind"], info["direction"], info["n_waypoints"], info["n_sections"]) == (
                k, wp.role, wp.kind, wp.direction, M, N)
            assert info["spacing_mm"]["max_along_seam"] <= 5.0 + 1e-9


def test_npz_with_sections_only_and_without_anything(g1_run, tmp_path):
    oc = K.run_outcome("DATA_FAIL", "weld plan not verified (D14)", category="waypoint")
    sections, src = K.sections_arrays(g1_run.out)
    a = K.precision_npz_arrays(_meta(oc, tmp_path, 0, src), sections=sections)
    p = K.write_precision_npz(tmp_path, a)
    d = K.load_precision_npz(p)
    assert int(d["n_paths"]) == 0 and not [k for k in d if k.startswith("p") and k[1].isdigit()]
    assert d["meta"]["error_code"] == K.ERROR_WAYPOINT_GENERATION_FAILED and "sec_s_m" in d
    assert K.sections_arrays(tmp_path / "nothing_here") == ({}, None)


# --------------------------------------------------------------------------------------
# KIRO-format outputs (scan_core.kiro_plan): platform configs, the fitted World <- link0 against KIRO's logged poses, npz carriage
# --------------------------------------------------------------------------------------
CFG_DIR = os.path.join(PKG, "config")
PLATFORMS = ("hongik_kiro.yaml", "hongik_lab_rb20_moveit.yaml", "hongik_rb20_fake_moveit.yaml")
KIRO_SAMPLES = os.path.join(STACK, "scan_core", "tests", "data", "kiro_samples")


# KIRO package: only hongik_kiro.yaml ships; the lab and fake-arm platform values are test data (scan_l0/tests/data)
PLATFORM_FILES = {"hongik_kiro.yaml": os.path.join(CFG_DIR, "hongik_kiro.yaml"),
                  "hongik_lab_rb20_moveit.yaml": os.path.join(STACK, "scan_l0", "tests", "data", "hongik_lab_rb20_values.yaml"),
                  "hongik_rb20_fake_moveit.yaml": os.path.join(STACK, "scan_l0", "tests", "data", "hongik_rb20_fake_moveit.yaml")}


def _platform(name):
    import yaml
    with open(PLATFORM_FILES[name], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["/**"]["ros__parameters"]


LAB_URDF = os.path.join(STACK, "scan_l0", "tests", "data", "rb20_1900es_u_eoat.urdf")
_NOZZLE = {}


def _lab_nozzle_mm() -> float:
    """Our torch's gas nozzle end along +z_TCP, derived as the run derives it on its arm (2026-10-13: the stick-out is measured from
    it): from the lab URDF's collision geometry (scan_core.weld_clearance.nozzle_end_mm) - the CAD rim 1.2 mm beyond the TCP."""
    if "mm" not in _NOZZLE:
        from scan_core.weld_clearance import nozzle_end_mm, tool_body_from_urdf
        _NOZZLE["mm"] = nozzle_end_mm(tool_body_from_urdf(open(LAB_URDF, "r", encoding="utf-8").read(), "torch_tcp"))
    return _NOZZLE["mm"]


def _kcfg(name):
    """kiro_output_config of the platform, its weld torch with the nozzle end as a run has it (teach_1g derives it on the arm)."""
    import dataclasses
    from scan_core.kiro_plan import kiro_output_config
    c = kiro_output_config(_platform(name))
    return dataclasses.replace(c, torch=c.torch.with_nozzle_end(_lab_nozzle_mm(), "the lab URDF collision geometry (test)"))


@pytest.mark.parametrize("name", PLATFORMS)
def test_platform_configs_carry_the_kiro_output_frame_chain(name):
    """World <- base and the weld torch are platform config values, one code path: KIRO (and the fake arm that mirrors it) = the
    fixed log-fitted World <- arm base (KIRO's torso and base do not move); the lab = World := our base link0 (identity). The weld
    plan's torch is OUR 1st-bracket torch in EVERY config (user + KIRO agreement 2026-10-12): TCP = the torch tip end = the CAD
    contact-tip end (KIRO 2026-10-12 night), axes identity (+z = nozzle axis, x = travel in the bend plane), the tip offset is the
    process stick-out (job weld_plan.stick_out_mm, not a platform value); lab / fake: tcp_frame torch_tcp on this arm,
    flange <- TCP = the URDF chain (eoat_yaw -90 deg, the torch turned 180 deg in its clamp since 2026-10-13); KIRO:
    right_manipulator_ee_point, the same torch turned 180 deg in its clamp at the same eoat_yaw -90 deg (KIRO's setup identified
    2026-10-13, URDF v2 r4; KIRO re-calibrates the TCP on site). TF is only an opt-in override."""
    from scan_core.kiro_plan import kiro_output_config, resolve_world_from_base
    from scipy.spatial.transform import Rotation
    from hongik_scan_ros.platform_config import MoveItJtcConfig
    p = _platform(name)
    c = kiro_output_config(p)
    assert c is not None and c.world_frame == "World" and not c.world_from_tf
    assert not ({"orientation_policy", "reference_dir"} & set(p["kiro_output"]))   # the 'reference' policy is removed (2026-10-12)
    assert not ({"tcp_frame", "ee_from_tip"} & set(p["kiro_output"]))           # the TCP is the weld torch's (weld_torch)
    wb = c.world_from_base
    if name == "hongik_lab_rb20_moveit.yaml":
        assert np.array_equal(wb.R, np.eye(3)) and np.array_equal(wb.t, np.zeros(3)) and wb.status == "lab definition"
    else:
        # the fixed log-fitted World <- arm base with the fitted 0.016 deg rotation (KIRO's poses at 0.09 mm), provenance
        assert np.allclose(wb.t * 1e3, [-249.83, -499.71, 1330.42], atol=0.005) and wb.status.startswith("fixed")
        assert math.degrees(K.Rotation.from_matrix(wb.R).magnitude()) == pytest.approx(0.0161, abs=1e-4)
        assert wb.source.startswith("FK fit World <- link0")
    # the weld torch: ours in every config
    t = c.torch
    assert c.ee_from_tip is t.tcp_from_tip and t.on_this_arm is True
    assert np.array_equal(t.tcp_from_tip.R, np.eye(3)) and np.array_equal(t.tcp_from_tip.t, np.zeros(3)) and t.stick_out_mm == 0.0
    assert "xyz_m" not in p["weld_torch"]["tip_in_tcp"] and t.tcp_from_tip.status.startswith("axes of our torch's TCP")
    assert t.status.startswith("nominal TCP = the CAD contact-tip end of our torch")
    want = Rotation.from_euler("z", -90.0, degrees=True)                           # the lab mounting = KIRO's (2026-10-13, r4)
    # the CAD contact-tip end turned 180 deg in its clamp about the torch body axis (eoat x -142.21 mm): KIRO's setup (URDF v2 r4) and
    # the lab robot model since 2026-10-13
    p_tcp = want.apply([-0.237857, -0.000452, 0.297172])
    if name != "hongik_kiro.yaml":                                                # the lab robot model: the virtual arm's URDF chain
        import io
        from scan_core.fk_urdf import chain_from_joints, parse_urdf_joints
        T_u = np.eye(4)
        for j_ in chain_from_joints(parse_urdf_joints(io.StringIO(open(LAB_URDF, encoding="utf-8").read())), "tcp", "torch_tcp").joints:
            T_u = T_u @ j_.origin()
        assert np.allclose(T_u[:3, 3], p_tcp, atol=2e-6) and np.allclose(t.flange_from_tcp.R, T_u[:3, :3], atol=1e-9)
    assert np.allclose(t.flange_from_tcp.t, p_tcp, atol=2e-6), (t.flange_from_tcp.t, p_tcp)
    assert np.degrees(np.arccos(t.flange_from_tcp.R[2, 2])) == pytest.approx(30.0, abs=1e-3)   # nozzle axis 30 deg off the flange axis
    if name == "hongik_kiro.yaml":
        assert c.tcp_frame == t.tcp_frame == "right_manipulator_ee_point" and t.flange_frame == "hongik_flange"
        assert t.flange_from_tcp.status.startswith("nominal CAD; KIRO re-calibrates the TCP on site")
    else:
        assert c.tcp_frame == t.tcp_frame == "torch_tcp" == p["torch_frame"] and t.flange_frame == "tcp"
    assert MoveItJtcConfig.from_params(p).weld_torch.to_dict() == t.to_dict()   # the MoveIt adapter's D14 tool reads the same model
    assert "base_frame" not in p["kiro_output"]["world_from_base"]               # the base is read from MoveIt at run time

    def tf_never(a, b):
        raise AssertionError("the fixed World <- base must not need TF")
    for rt in sorted({p["base_frame"], "world", "robot_base_link"} - {"World"}):    # any runtime planning frame other than World
        wt, _ = resolve_world_from_base(c, rt, tf_never)
        assert np.array_equal(wt.R, wb.R) and np.array_equal(wt.t, wb.t) and wt.base_frame == rt
    wt, _ = resolve_world_from_base(c, "World", tf_never)                        # the planning frame is World: identity
    assert np.array_equal(wt.R, np.eye(3)) and np.array_equal(wt.t, np.zeros(3))


def test_kiro_config_reproduces_kiros_logged_world_poses():
    """The KIRO config's World <- base, tested in our tree without the KIRO robot: KIRO's 26 logged teaching snapshots (joint values)
    through our RB20-1900ES-U FK, the KIRO config World <- arm base (fitted translation + fitted 0.016 deg rotation) and the same fit's
    flange <- right_manipulator_ee_point of KIRO's OLD URDF (the chiseling-gun frame the logs were computed with - void as a torch
    model since 2026-10-12, used here only to check World <- base) land on KIRO's logged tcp_pose_world: <= 0.1 mm / 0.007 deg
    (rms 0.05 mm / 0.004 deg). Without the rotation (World z up exactly) it would be 0.34 mm. The config's weld torch (ours) does NOT
    reproduce the logged poses: the logs were not recorded with that frame."""
    from scan_core.fk_urdf import rb20_1900es_u_chain
    from scan_core.kiro_plan import kiro_output_config
    with open(os.path.join(KIRO_SAMPLES, "teaching_snapshots.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    snaps = data["snapshots"]
    assert len(snaps) == 26
    kc = kiro_output_config(_platform("hongik_kiro.yaml"))
    wt = kc.world_from_base
    T_cfg = np.eye(4)
    T_cfg[:3, :3], T_cfg[:3, 3] = wt.R, wt.t
    T_fe = np.asarray(data["flange_T_ee_point"], dtype=float)                    # the logs' own (old) ee_point fit, not the config torch
    assert not np.allclose(kc.torch.flange_from_tcp.t, T_fe[:3, 3], atol=0.05)     # the config's weld torch is ours, far from it
    chain = rb20_1900es_u_chain()
    T_up = T_cfg.copy()
    T_up[:3, :3] = np.eye(3)
    for T_Wb, pos_max, rot_max, pos_rms in ((T_cfg, 0.1, 0.007, 0.06), (np.asarray(data["World_T_link0_fitted"], dtype=float), 0.1, 0.007,
                                                                          0.06), (T_up, 0.4, 0.02, 0.35)):
        dp, dr = [], []
        for sn in snaps:
            T = T_Wb @ chain.fk(sn["joints_rad"]) @ T_fe
            dp.append(float(np.linalg.norm(T[:3, 3] - np.asarray(sn["tcp_pose_world"]["position_m"]))) * 1e3)
            R_log = K.Rotation.from_quat(sn["tcp_pose_world"]["orientation_xyzw"]).as_matrix()
            dr.append(math.degrees(K.Rotation.from_matrix(R_log.T @ T[:3, :3]).magnitude()))
        assert max(dp) < pos_max and max(dr) < rot_max and math.sqrt(float(np.mean(np.square(dp)))) < pos_rms, (max(dp), max(dr))
    # pass_1.yaml current_start is the taught weld_start of pass 1 (the same pose)
    import yaml
    with open(os.path.join(KIRO_SAMPLES, "pass_1.yaml"), "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)["current_start"]
    ws = [sn for sn in snaps if sn["name"] == "p1:weld_start"][0]["tcp_pose_world"]
    assert [d["position_m"][c] for c in "xyz"] == ws["position_m"]


def test_kiro_outputs_ride_in_the_npz_meta_and_arrays(g1_run, tmp_path):
    """The action result has no slot for the KIRO-format files: the npz meta names them and the npz carries their text."""
    from scan_core.kiro_plan import kiro_output_config, resolve_world_from_base, write_kiro_outputs
    c = _kcfg("hongik_rb20_fake_moveit.yaml")
    tip_m = (_lab_nozzle_mm() + 20.0) * 1e-3                                     # TCP -> wire end: nozzle end + stick-out 20 mm
    wt, notes = resolve_world_from_base(c, "link0")
    ko = {"written": True, **write_kiro_outputs(g1_run.plan, tmp_path, c, world_from_base=wt, notes=notes)}
    m = K.kiro_meta(ko)
    assert m["written"] and m["frame_id"] == "World" and "orientation_policy" not in m and m["torch_roll"]["deg"] == 0.0 and "NECK_RULE" in m["torch_roll"]["reason"]
    assert m["stick_out_mm"] == 20.0 and m["stick_out_source"].startswith("job weld_plan.stick_out_mm 20 mm (the wire stick-out from the "
                                                                          "GAS NOZZLE END")
    assert m["nozzle_end_mm"] == pytest.approx(1.2, abs=1e-3) and m["tcp_to_aim_point_mm"] == pytest.approx(tip_m * 1e3)
    assert "torch tip end centre" in m["tcp_definition"] and "no field" in m["result_slot"]
    assert all(np.allclose(p_["tip_start_World_m"], p_["start_World_m"], atol=tip_m + 0.001) for p_ in m["passes"])
    assert "not_carried" in m["action_mapping"] and len(m["action_mapping"]["ask_kiro"]) == 3
    arr = K.kiro_npz_arrays(ko)
    assert sorted(arr) == sorted(["kiro_readme_md"] + [f"kiro_pass_{k}_yaml" for k in (1, 2, 3, 4)]
                                 + [f"kiro_pass_{k}_info_yaml" for k in (1, 2, 3, 4)] + [f"kiro_pass_{k}_start_pose" for k in (1, 2, 3, 4)]
                                 + [f"kiro_pass_{k}_goal_pose" for k in (1, 2, 3, 4)])
    import yaml
    for k in (1, 2, 3, 4):                                                         # the npz poses = the yaml poses
        y = yaml.safe_load((tmp_path / "kiro" / f"pass_{k}.yaml").read_text(encoding="utf-8"))
        for key, nm in (("current_start", "start"), ("current_goal", "goal")):
            want = [y[key]["position_m"][c] for c in "xyz"] + [y[key]["orientation_xyzw"][c] for c in "xyzw"]
            assert np.array_equal(arr[f"kiro_pass_{k}_{nm}_pose"], want)
    # result.paths <-> pass_<k>.yaml: the first / last waypoint (wire tip) = the yaml start / goal TCP + stick_out x z_TCP, base frame
    wt_inv = np.linalg.inv(np.block([[wt.R, wt.t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    for pa in K.plan_paths(g1_run.plan):
        for idx, nm in ((0, "start"), (-1, "goal")):
            pose = arr[f"kiro_pass_{pa.k}_{nm}_pose"]
            aim = pose[:3] + tip_m * K.Rotation.from_quat(pose[3:]).as_matrix()[:, 2]
            assert np.allclose((wt_inv @ np.append(aim, 1.0))[:3], pa.pos_m[idx], atol=1e-12)
    oc = K.run_outcome("PASS", "done")
    paths = K.plan_paths(g1_run.plan)
    meta = K.npz_meta(outcome=oc, status="PASS", reason="done", welding_position=1, position="1G", base_frame="link0", k=3.0,
                      out_dir=str(tmp_path), run_id="x", n_paths=len(paths), kiro_outputs=m)
    p = K.write_precision_npz(tmp_path, K.precision_npz_arrays(meta, plan=g1_run.plan, paths=paths, kiro=arr))
    d = K.load_precision_npz(p)
    assert d["meta"]["kiro_outputs"]["files"] == ko["files"]
    assert str(d["kiro_pass_1_yaml"]) == (tmp_path / "kiro" / "pass_1.yaml").read_text(encoding="utf-8")
    assert d["p1_info"]["speed_const_mm_s"] == pytest.approx(g1_run.plan.passes[0].speed_const_mps * 1e3)
    assert d["meta"]["kiro_outputs"]["weld_torch"]["tcp_frame"] == "torch_tcp"
    assert "p1_tcp_R" not in d                                                   # this plan was made without a torch model
    # a plan with the weld torch model (the run passes the platform's weld_torch): the npz also has its TCP poses per section
    import dataclasses
    plan_t = dataclasses.replace(g1_run.plan, torch=c.torch.with_stick_out(20.0, "test"))    # as plan_weld makes it
    d2 = K.precision_npz_arrays(meta, plan=plan_t, paths=paths, kiro=arr)
    for wp in plan_t.passes:
        R_tip, R_tcp = d2[f"p{wp.k}_R_torch"], d2[f"p{wp.k}_tcp_R"]
        assert np.array_equal(R_tcp, R_tip)                                       # our torch: identity axes, TCP 21.2 mm back along z
        assert np.allclose(d2[f"p{wp.k}_tcp_pos_m"], d2[f"p{wp.k}_aim_pos_m"] - tip_m * R_tip[:, :, 2], atol=1e-12)
    assert K.kiro_meta({"written": False, "reason": "no World"})["reason"] == "no World" and K.kiro_meta(None) is None
    assert K.kiro_npz_arrays({"written": False}) == {}


# --------------------------------------------------------------------------------------
# precision_sensing_result.yaml (user 2026-10-13): the full result message
# --------------------------------------------------------------------------------------
def _ps_result(n_paths: int, code: int = 0, message: str = "PASS: x", data_file: str = "/r/precision_sensing.npz"):
    act = pytest.importorskip("welding_interfaces.action")
    pytest.importorskip("rosidl_runtime_py")
    from geometry_msgs.msg import Pose
    from welding_interfaces.msg import WeldingPath
    rng = np.random.default_rng(7)
    r = act.PrecisionSensing.Result()
    r.error_code, r.error_message, r.data_file_path = code, message, data_file
    for _k in range(n_paths):
        w = WeldingPath()
        poses = []
        for _i in range(int(rng.integers(2, 80))):
            ps = Pose()
            ps.position.x, ps.position.y, ps.position.z = (float(v) for v in rng.normal(0.0, 0.5, 3))
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            ps.orientation.x, ps.orientation.y, ps.orientation.z, ps.orientation.w = (float(v) for v in q)
            poses.append(ps)
        w.waypoints = poses
        r.paths.append(w)
    return r


@pytest.mark.parametrize("n_paths, code, message", [(4, 0, "PASS_UNRESOLVED: unresolved: [end_S] - 'quoted' # not a comment"),
                                                    (0, 255, "ABORTED: cancel requested by the operator (D7)\nsecond line"),
                                                    (1, 4, "")])
def test_result_yaml_is_the_full_message_and_reads_back_to_it(tmp_path, n_paths, code, message):
    act = pytest.importorskip("welding_interfaces.action")
    r = _ps_result(n_paths, code, message, data_file="" if n_paths == 0 else "/r/precision_sensing.npz")
    p = K.write_result_yaml(tmp_path, r)
    assert p == str(tmp_path / K.RESULT_YAML_NAME) and os.path.isabs(p)
    text = open(p, encoding="utf-8").read()
    assert text == K.result_yaml_text(r)
    head = text.splitlines()[:2]
    assert head[0] == (f"# welding_interfaces/action/PrecisionSensing result: the full message as the server returned it; error_code "
                       f"{code} = {K.CODE_NAMES[code]}") and head[1].startswith("# paths[k-1] = pass k (WeldingPath)")
    doc = K.load_result_yaml(p)
    assert list(doc) == list(act.PrecisionSensing.Result.get_fields_and_field_types())     # every field, in the message's order
    assert len(doc["paths"]) == n_paths
    for w, d in zip(r.paths, doc["paths"]):
        assert [[wp["position"][c] for c in "xyz"] for wp in d["waypoints"]] == [[getattr(q.position, c) for c in "xyz"]
                                                                                for q in w.waypoints]
        assert [[wp["orientation"][c] for c in "xyzw"] for wp in d["waypoints"]] == [[getattr(q.orientation, c) for c in "xyzw"]
                                                                                    for q in w.waypoints]
    assert K.result_message(doc, act.PrecisionSensing.Result) == r                     # exact round trip (floats, text)


def test_result_yaml_is_written_only_into_an_existing_run_folder(tmp_path):
    r = _ps_result(1)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        K.write_result_yaml(tmp_path / "no_run", r)
    assert not (tmp_path / "no_run").exists()
    (tmp_path / K.RESULT_YAML_NAME).write_text("old", encoding="utf-8")                # replaced atomically, no temporary left
    K.write_result_yaml(tmp_path, r)
    assert os.listdir(tmp_path) == [K.RESULT_YAML_NAME]
    act = pytest.importorskip("welding_interfaces.action")
    with pytest.raises(Exception):
        K.result_message({**K.load_result_yaml(tmp_path / K.RESULT_YAML_NAME), "unknown_field": 1}, act.PrecisionSensing.Result)
