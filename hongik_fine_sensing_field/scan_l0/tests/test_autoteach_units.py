"""auto-teach 순수 도우미 단위 테스트: 거친 작업 (안쪽 들임 없음) / 납품 작업 구성, 프로브 판정, 광학 분기 뒤집기, confirm 처리,
드웰 분류·탐색 순서, 패스 잔차 회귀, delta_cb. 로봇·센서 없음. (2026-10-01 흐름의 계획·자세 입력은 test_autoteach_serpentine_units.py)"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import CONFIG

from scan_core.frames import unit
from scan_core.localize import PassFit, StaticLocalization
from scan_core.path_gen import sensor_frames
from scan_core.seam_features import CrossSectionFeatures, SeamScanFeatures
from scan_core.specimen import GrooveSpec
from scan_core.types import Calib, JobError, ScanJob
from scan_l0 import config as C
from scan_l0.session import ScanSession
from scan_l0.sensor_view import sensor_view
from scan_l0.autoteach import (CONFIRM_MODES, _robust_slope, classify_localization, confirm_move, confirm_needed, move_plan_text,
                               flip_optical_branch, make_coarse_job, make_refined_job, pass_residuals, probe_verdict, search_order)


G_LAB = (0.0, 0.0, -1.0)          # 연구실 플랫폼 중력 (platform TF — 판정 함수에는 기본값이 없다)


@pytest.fixture
def at_cfg():
    return C.load_settings(CONFIG).auto_teach_config()


@pytest.fixture
def view():
    return sensor_view(C.load_settings(CONFIG).z_valid_mm)


@pytest.fixture
def calib() -> Calib:
    return Calib.lab_initial(eoat_yaw_rad=-math.pi / 2, opt_yaw_branch=-1)


def rough_job(**kw) -> ScanJob:
    d = dict(start=[0.6, -0.19, 0.07], end=[0.62, 0.19, 0.075], surface_normal=[0.05, 0.0, 0.998], standoff_m=0.32, pitch_deg=-5.0,
             lead_in_m=0.04, lead_out_m=0.04, table_z_m=-0.05, job_id="rough", clamp_height_m=0.03)
    d.update(kw)
    return ScanJob(**d)


# --------------------------------------------------------------------------------------
# job_0 (coarse)
# --------------------------------------------------------------------------------------
def test_make_coarse_job_relevels_and_projects_without_inset(view, calib):
    j0 = rough_job()
    jc = make_coarse_job(j0, view, depth_m=0.010, opening_m=0.0108)
    np.testing.assert_allclose(jc.surface_normal, [0, 0, 1], atol=1e-12)                 # 1G: n0 = up (재수평화)
    assert abs(float(jc.unit_dir @ jc.normal_unit)) < 1e-12                               # u0 ⟂ n0 (path_gen 요구)
    L0 = j0.length_m
    assert jc.length_m == pytest.approx(L0, abs=1e-12)                                    # 안쪽 들임 없음 (2026-10-01)
    d0 = j0.end - j0.start
    u0 = unit(d0 - np.array([0, 0, d0[2]]))
    np.testing.assert_allclose(jc.start, j0.start, atol=1e-12)                           # 거친 시점 S0 자체
    np.testing.assert_allclose(jc.end, j0.start + u0 * L0, atol=1e-12)
    assert jc.standoff_m == pytest.approx(view.coarse_range_m(0.010, 0.0, 0.0108)) and jc.pitch_deg == 0.0   # 센서 시야 모형
    assert jc.incidence_tilt_deg == 0.0
    assert jc.lead_in_m == jc.lead_out_m == 0.001 and jc.speed_mps == 0.03
    assert jc.clamp_height_m == 0.03 and jc.table_z_m == -0.05 and jc.plate_top_measured_m is None
    assert jc.job_id == "rough_coarse" and jc.resolved_orientation() == "flat_1G"
    # 툴 프레임이 강체 (비직교 u/n 이었다면 path_gen 이 비틀린 자세를 만든다)
    T = sensor_frames(jc, calib).T_base_flange(0.0)
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
    # 옵션
    jc2 = make_coarse_job(j0, view, range_m=0.38, relevel=False)
    assert jc2.standoff_m == 0.38 and abs(float(jc2.unit_dir @ jc2.normal_unit)) < 1e-12
    assert abs(float(jc2.normal_unit @ np.array([0, 0, 1.0]))) < 1.0 - 1e-6              # 거친 법선 유지 (u 에 직교 투영은 u 쪽)
    with pytest.raises(JobError):
        make_coarse_job(rough_job(end=[0.6, -0.19, 0.17]), view)                       # 거친 두 점이 판 법선 방향 (u0 정의 불가)


# --------------------------------------------------------------------------------------
# job_1 (refined) / shift
# --------------------------------------------------------------------------------------
def _fit(flags=(), s_a=0.01, s_b=0.36, depth_mm=9.95) -> PassFit:
    c = np.array([0.6, 0.0, 0.04])
    u = np.array([0.0, 1.0, 0.0])
    n = unit([0.05, 0.0, 1.0])
    fit = PassFit(c=c, u=u, n=n, s_a=s_a, s_b=s_b, line_rms_m=1e-4, plane_rms_m=3e-5, n_inliers=50)
    fit.feature_medians = {"depth_mm": depth_mm, "root_gap_mm": 5.0}
    fit.flags = list(flags)
    return fit


def test_make_refined_job_root_points_standoff_and_leads(at_cfg):
    j0 = rough_job(plate_top_measured_m=0.049)
    fit = _fit()
    job, info = make_refined_job(j0, fit, standoff_plate_m=0.34, pitch_deg=-5.0, overscan_m=0.02, rough_start=j0.start,
                                 rough_end=j0.end, plate_top_m=0.052, spec=GrooveSpec(bevel_side="auto"), job_id="refined")
    np.testing.assert_allclose(job.start, fit.c + fit.u * 0.01, atol=1e-12)
    np.testing.assert_allclose(job.end, fit.c + fit.u * 0.36, atol=1e-12)
    np.testing.assert_allclose(job.surface_normal, fit.n, atol=1e-12)
    assert job.standoff_m == pytest.approx(0.34 + 0.00995)                                 # 판 상면 standoff + 그루브 깊이 (루트 점 기준)
    # calib 없이: 대칭 최악값 = 착점 오버스캔 + standoff·tan|pitch| (calib 이 있으면 착점 기하 — serpentine 단위 테스트)
    assert job.lead_in_m == job.lead_out_m == pytest.approx(0.02 + 0.34 * math.tan(math.radians(5.0)))
    assert job.pitch_deg == -5.0 and job.plate_top_measured_m == 0.052                     # 올리기만
    assert job.clamp_height_m == 0.03 and job.job_id == "refined"
    assert info["ends"] == {"start": "observed", "end": "observed"} and info["depth_m"] == pytest.approx(0.00995)
    # 끝이 관측되지 않은 쪽은 거친 점의 직선 투영
    fit2 = _fit(flags=("end_not_observed_first", "end_not_observed_last"), s_a=0.05, s_b=0.30)
    job2, info2 = make_refined_job(j0, fit2, standoff_plate_m=0.34, pitch_deg=-5.0, overscan_m=0.02, rough_start=j0.start,
                                   rough_end=j0.end)
    assert info2["ends"] == {"start": "rough_projected", "end": "rough_projected"}
    assert job2.start[1] == pytest.approx(j0.start[1]) and job2.end[1] == pytest.approx(j0.end[1])
    assert job2.start[2] == pytest.approx(0.04) and job2.end[2] == pytest.approx(0.04)     # 루트 직선 위 (높이 z=0.04)
    # 깊이 중앙값이 없으면 spec 두께
    fit3 = _fit(depth_mm=float("nan"))
    job3, info3 = make_refined_job(j0, fit3, standoff_plate_m=0.34, pitch_deg=0.0, overscan_m=0.02,
                                   spec=GrooveSpec(thickness_mm=12.0, bevel_side="auto"))
    assert job3.standoff_m == pytest.approx(0.352) and job3.lead_in_m == pytest.approx(0.02)
    # 불완전한 fit / 빈 구간은 거부
    with pytest.raises(ValueError):
        make_refined_job(j0, PassFit(c=np.full(3, np.nan), u=np.full(3, np.nan), n=np.full(3, np.nan)), standoff_plate_m=0.34,
                         pitch_deg=0.0, overscan_m=0.02)
    with pytest.raises(ValueError):
        make_refined_job(j0, _fit(s_a=0.3, s_b=0.3), standoff_plate_m=0.34, pitch_deg=0.0, overscan_m=0.02)


def test_make_refined_job_top_offset_report(at_cfg):
    j0 = rough_job()
    fit = _fit()
    P_root = np.array([[0.6, 0.1, 0.04], [0.6, 0.2, 0.04], [np.nan] * 3])
    P_top = P_root + np.array([0.003, 0.0, 0.010])
    job, info = make_refined_job(j0, fit, standoff_plate_m=0.34, pitch_deg=-5.0, overscan_m=0.02, top_polyline=P_top,
                                 root_polyline=P_root)
    np.testing.assert_allclose(info["top_offset"], [0.003, 0.0, 0.010], atol=1e-12)
    np.testing.assert_allclose(info["top_start"], job.start + [0.003, 0.0, 0.010], atol=1e-12)


# --------------------------------------------------------------------------------------
# probe / branch flip
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dx_x, dx_y, sgn, expect", [
    (-8.0, 0.3, 1.0, "confirmed"), (-7.0, -2.5, 1.0, "confirmed"), (-9.4, 0.0, 1.0, "confirmed"), (-5.0, 0.0, 1.0, "confirmed"),
    (-8.0, 3.5, 1.0, "confirmed"),                                           # |dx_Y| ≤ s/2 = 4
    (8.1, 0.5, 1.0, "flip"), (6.6, 0.0, 1.0, "flip"),
    (8.0, 0.0, -1.0, "confirmed"), (-8.0, 0.0, -1.0, "flip"),               # X 스텝 부호 (그루브를 가운데 쪽으로)
    (-8.0, 4.5, 1.0, "inconsistent"), (0.4, 0.1, 1.0, "inconsistent"), (-2.5, 0.0, 1.0, "inconsistent"),
    (float("nan"), 0.0, 1.0, "inconsistent"),
])
def test_probe_verdict_mapping(dx_x, dx_y, sgn, expect):
    """4d P4: 판정 띠 = 가설 중간점 (스텝의 절반, 원칙 2) — step 8 mm 에서 확인 −8 ± 4, 뒤집힘 +8 ± 4, Y ≤ 4."""
    v, note = probe_verdict(dx_x, dx_y, 8.0, sgn)
    assert v == expect and note.startswith("probe")


def test_flip_optical_branch_mirrors_x_and_y(calib):
    c2 = flip_optical_branch(calib)
    T, T2 = calib.T_flange_opt, c2.T_flange_opt
    np.testing.assert_allclose(T2[:3, 0], -T[:3, 0], atol=1e-12)
    np.testing.assert_allclose(T2[:3, 1], -T[:3, 1], atol=1e-12)
    np.testing.assert_allclose(T2[:3, 2], T[:3, 2], atol=1e-12)
    np.testing.assert_allclose(T2[:3, 3], T[:3, 3], atol=1e-12)
    np.testing.assert_allclose(c2.T_flange_eoat, calib.T_flange_eoat)
    assert c2.eoat_yaw_rad == calib.eoat_yaw_rad and "branch_flipped" in c2.source
    # = Calib.lab_initial(opt_yaw_branch=+1)
    other = Calib.lab_initial(eoat_yaw_rad=-math.pi / 2, opt_yaw_branch=+1)
    np.testing.assert_allclose(c2.T_flange_opt, other.T_flange_opt, atol=1e-12)
    np.testing.assert_allclose(flip_optical_branch(c2).T_flange_opt, T, atol=1e-12)


def test_flip_optical_branch_keeps_every_other_calib_field():
    """The flip changes T_flange<-opt (and the source) only: every other field of the file calib - the 1G 'g1:' block (time offset,
    sigmas, edge response, noise priors) and the tool body - stays (dataclasses.replace; a field-by-field copy dropped new fields)."""
    import dataclasses
    from conftest import CALIB
    c = C.load_calib(CALIB)
    assert c.noise_priors and c.edge_response and c.he_sigma_m is not None           # the lab file states the g1 block
    c.body_points_eoat, c.body_source = np.eye(3), "test body"
    c.residual_rms_m, c.sensor_param_hash = 1.0e-4, "test hash"                       # non-default values so a loss shows
    c2 = flip_optical_branch(c)
    for f in dataclasses.fields(Calib):
        if f.name in ("T_flange_opt", "T_flange_eoat", "source"):
            continue
        a, b = getattr(c, f.name), getattr(c2, f.name)
        assert a is b or (np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b), f.name
    np.testing.assert_allclose(c2.T_flange_eoat, c.T_flange_eoat)


# --------------------------------------------------------------------------------------
# confirm
# --------------------------------------------------------------------------------------
def test_confirm_move_modes():
    log = []
    assert confirm_move("none", "x", 1.0, source=lambda: "n", log=log.append) is True and log == []
    C_ = 0.05                                                               # 'large' 문턱 = 안전 정책 여유 (AutoTeach 가 준다)
    assert confirm_move("large", "x", C_ - 1e-6, source=lambda: "n", log=log.append, threshold_m=C_) is True and log == []
    assert confirm_move("large", "x", 0.06, source=lambda: "y", log=log.append, threshold_m=C_) is True and any("CONFIRM x" in ln for ln in log)
    assert confirm_move("large", "x", 0.06, source=lambda: "n", log=log.append, threshold_m=C_) is False
    with pytest.raises(ValueError, match="threshold_m"):
        confirm_move("large", "x", 0.06, source=lambda: "y")
    assert confirm_move("each", "x", 0.001, source=lambda: "Y", log=log.append) is True
    assert confirm_move("each", "x", 0.001, source=lambda: "", log=log.append, timeout_s=0.05) is False      # 타임아웃 = 중단
    assert confirm_move("each", "x", 0.001, source=lambda: "q", log=log.append) is False
    with pytest.raises(ValueError):
        confirm_move("maybe", "x", 0.001)


def test_confirm_move_auto_mode_asks_only_joint_detours():
    """'auto' (사용자 2026-10-08, 반복 실기): 관절 직선은 자동, 우회 관절 이동 (OMPL 등) 만 묻는다; 스캔·국부 이동은 자동."""
    from scan_l0.autoteach import joint_move_kind
    assert joint_move_kind("joint-space straight line") == "joint"
    assert joint_move_kind("MoveIt OMPL (265 waypoints; the joint-space straight line is blocked: ...)") == "joint_detour"
    assert joint_move_kind("sampled via-point path (the joint-space straight line is blocked: x)") == "joint_detour"
    log = []
    for kind in ("local", "scan", "joint"):
        assert confirm_move("auto", "m", 1.0, source=lambda: "n", log=log.append, kind=kind) is True
    assert log == []
    assert confirm_move("auto", "support_probe4", 1.0, source=lambda: "n", log=log.append, kind="joint_detour") is False
    assert any("CONFIRM support_probe4" in ln for ln in log)
    assert confirm_move("scan", "x", 1.0, source=lambda: "y", log=log.append, kind="joint_detour") is True      # scan 도 묻는다


def test_confirm_move_scan_mode_asks_only_scan_and_joint_moves():
    """'scan' (사용자 2026-10-08): 시편을 가로지르는 스캔 이동과 관절 공간 이동만 묻고, 정밀 보정 같은 짧은 국부 이동은 자동 — 거리와 무관."""
    log = []
    assert confirm_move("scan", "S_corr1", 0.5, source=lambda: "n", log=log.append, kind="local") is True and log == []
    assert confirm_move("scan", "pass1_E_to_S", 0.38, source=lambda: "y", log=log.append, kind="scan") is True
    assert any("CONFIRM pass1_E_to_S" in ln for ln in log)
    assert confirm_move("scan", "acquire_S", 0.9, source=lambda: "n", log=log.append, kind="joint") is False
    assert confirm_move("scan", "probe_X", 0.006, source=lambda: "n", log=log.append) is True       # 기본 kind = 국부 이동


def test_confirm_needed_is_the_unchanged_decision_rule():
    """프롬프트 문구를 바꾸며 판단 규칙을 따로 뺐다 (묻는 이동에서만 확인 문구용 계획) — 판단은 나누기 전 confirm_move (476cc54: 'scan'
    742caae + 'auto'·'joint_detour' 575eb13 병합) 의 식 그대로. 모드·종류를 모두 돈다 (새 모드는 이 표에 넣어야 통과)."""
    assert CONFIRM_MODES == ("each", "large", "scan", "auto", "none")

    def old_rule(mode, d, thr, kind):          # 476cc54 confirm_move 의 조건 (CONFIRM_KINDS_SCAN / _AUTO 를 펼쳐 적음)
        return not (mode == "none" or (mode == "large" and d <= thr) or (mode == "scan" and kind not in ("scan", "joint", "joint_detour"))
                    or (mode == "auto" and kind not in ("joint_detour",)))
    # 같은 규칙을 표로도 (거리와 무관한 모드: 묻는 종류)
    asks = {"each": {"local", "scan", "joint", "joint_detour"}, "scan": {"scan", "joint", "joint_detour"}, "auto": {"joint_detour"},
            "none": set()}
    for mode in CONFIRM_MODES:
        for d in (0.0, 0.049, 0.05, 0.051, 1.893):
            for kind in ("local", "scan", "joint", "joint_detour"):
                want = (d > 0.05) if mode == "large" else (kind in asks[mode])
                assert old_rule(mode, d, 0.05, kind) == want, (mode, d, kind)
                assert confirm_needed(mode, d, 0.05, kind) == want, (mode, d, kind)
                log = []
                asked = not confirm_move(mode, "x", d, source=lambda: "n", log=log.append, threshold_m=0.05, kind=kind)
                assert asked == bool(log) == want, (mode, d, kind)
    with pytest.raises(ValueError, match="threshold_m"):
        confirm_needed("large", 0.1)
    with pytest.raises(ValueError):
        confirm_needed("maybe", 0.1)


# 2026-10-08 g1_b start_pose (실기 기록 hongik_auto_teach/g1_b): 프롬프트 'CONFIRM start_pose: move 1893.0 mm' = 'large' 판단 값 = 현 929.2 mm
# + 툴 레버 317.0 mm × 174.23° (G1B_*). 기록한 계획 (joint_plan) 은 path_length_m 0.94965 (재시간 경로 변수 S)·duration_s 31.95·226 점뿐이다.
# 플랜지 통계는 그 기록에서 다시 푼 값: 시작 플랜지 자세의 IK 해 26 개 중 목표 관절까지 관절 직선이 226 점·S 0.9496476 m·31.955 s 를 그대로
# 내는 해 하나 → 플랜지 경로 949.6 mm (모든 점에서 병진 > 회전 × 0.2 m 라 S 와 같다), 회전 176.6°, 가장 많이 도는 관절 wrist1 128.4°.
G1B_CHORD_M, G1B_ROT_DEG, G1B_LEVER_M = 0.9291609737219987, 174.2282490620713, 0.31696907227362103
G1B_RECORDED_PLAN = {"kind": "joint", "how": "joint-space straight line", "n_points": 226, "path_length_m": 0.9496476250536787,
                     "duration_s": 31.954920835122625, "v_cruise_mps": 0.03, "time_scale": 1.0}
G1B_JOINT_PLAN = {**G1B_RECORDED_PLAN, "route": "straight", "route_text": "joint-space straight line", "n_waypoints": 2,
                  "flange_translation_m": 0.9496476250536792, "flange_rotation_deg": 176.56352199687217, "max_joint": "wrist1",
                  "max_joint_rotation_deg": 128.38145065307617}


def test_move_plan_text_joint_moves_state_route_translation_rotation_duration_and_largest_joint():
    t = move_plan_text(G1B_JOINT_PLAN)
    assert t == ("joint-space straight line: flange path 949.6 mm, flange rotation 176.6 deg, duration 32.0 s, largest joint rotation "
                 "wrist1 128.4 deg"), t
    # 재시간 경로 변수 S 는 이동 거리로 쓰지 않는다: 회전이 큰 경로 (합성 — S 가 병진보다 길다) 에서도 문구는 플랜지 병진
    rot_heavy = {**G1B_JOINT_PLAN, "path_length_m": 1.31}
    assert move_plan_text(rot_heavy) == t and "1310" not in move_plan_text(rot_heavy)
    ompl = {**G1B_JOINT_PLAN, "route": "OMPL", "route_text": "MoveIt OMPL (14 waypoints)", "n_waypoints": 14, "flange_translation_m": 2.734,
            "how": "MoveIt OMPL (14 waypoints; the joint-space straight line is blocked: point 40/300 invalid)"}
    assert move_plan_text(ompl).startswith("MoveIt OMPL (14 waypoints): flange path 2734.0 mm, ")
    # 짧은 설명이 없는 계획 (다른 백엔드): 경로 종류와 웨이포인트 수에서
    bare = {k: v for k, v in ompl.items() if k != "route_text"}
    assert move_plan_text(bare).startswith("MoveIt OMPL (14 waypoints): ")
    # 통계를 내지 않는 계획 (g1_b 가 실제로 기록한 계획 그대로): 없는 값을 지어내지 않는다 (S 를 플랜지 경로로 쓰지 않는다)
    t = move_plan_text(G1B_RECORDED_PLAN)
    assert t == ("joint-space straight line: flange path not reported by the planner, flange rotation not reported by the planner, "
                 "duration 32.0 s"), t


def test_move_plan_text_cartesian_moves():
    moveit = {"kind": "cartesian", "n_points": 191, "length_m": 0.3805, "rotation_deg": 0.21, "duration_s": 10.41}
    assert move_plan_text(moveit, chord_m=0.3805, rot_deg=0.21) == \
        "Cartesian straight line: flange path 380.5 mm, flange rotation 0.2 deg, duration 10.4 s"
    sim = {"kind": "cartesian", "path_length_m": 0.006, "duration_s": 0.9}            # 회전을 내지 않는 계획: 현의 회전
    assert move_plan_text(sim, chord_m=0.006, rot_deg=1.5) == \
        "Cartesian straight line: flange path 6.0 mm, flange rotation 1.5 deg, duration 0.9 s"
    # 계획이 없다 — 왜 없는지를 구조로 (호출자의 check_leg 결과) 구분한다. 직선이므로 경로 = 현
    # (1) 백엔드가 보내기 전에 계획하는 기능이 없다 (RB 제어박스 어댑터: 제어박스가 보낼 때 계획)
    t = move_plan_text(None, chord_m=0.040, rot_deg=-2.0)
    assert t == ("Cartesian straight line (the motion backend plans it only when sent - no duration before this prompt): flange path "
                 "40.0 mm, flange rotation 2.0 deg"), t
    # (2) 기능은 있는데 지금 자세에서 거부: 사유를 밝히고, 보낼 때 다시 계획해 원인이 남으면 거부된다고 말한다 ('planned when sent' 가 아니다)
    why = "the dense straight-line IK jumps 0.054 rad at point 12 - refusing"
    t = move_plan_text(None, chord_m=0.040, rot_deg=-2.0, refused=why)
    assert t == (f"Cartesian straight line, NOT plannable from the current pose before this prompt ({why}) - planned again when sent "
                 "and refused there if the cause remains: flange path 40.0 mm, flange rotation 2.0 deg"), t
    assert move_plan_text(None, chord_m=0.04, refused="").startswith("Cartesian straight line, NOT plannable from the current pose "
                                                                     "before this prompt (no reason given)")
    # (3) 계획은 있는데 시간을 내지 않는 플래너: 지어내지 않는다
    assert move_plan_text({"kind": "cartesian", "length_m": 0.04}, chord_m=0.04, rot_deg=0.0) == \
        "Cartesian straight line: flange path 40.0 mm, flange rotation 0.0 deg, duration not reported by the planner"


def test_confirm_move_prints_the_plan_text_not_the_decision_value():
    log = []
    assert confirm_move("each", "start_pose", 1.893, source=lambda: "y", log=log.append, kind="joint",
                        text=move_plan_text(G1B_JOINT_PLAN), extra="(joint-space move via MoveIt)")
    assert log[0].startswith("CONFIRM start_pose: joint-space straight line: flange path 949.6 mm, flange rotation 176.6 deg, "
                             "duration 32.0 s, largest joint rotation wrist1 128.4 deg (joint-space move via MoveIt) -> answer with the service "
                             "~/confirm_move (std_srvs/SetBool: data true = execute, false = abort"), log[0]
    assert "1893" not in log[0] and "move 1893" not in log[0]
    log2 = []
    assert confirm_move("each", "x", 0.0123, source=lambda: "y", log=log2.append)        # 문구가 없으면 예전 형식 (판단 값)
    assert log2[0].startswith("CONFIRM x: move 12.3 mm -> ")


def _confirm_stub(mode, *, lever=0.40, clearance=0.05):
    from scan_l0.autoteach import AutoTeach
    at = AutoTeach.__new__(AutoTeach)
    at.policy = SimpleNamespace(clearance_m=clearance, basis={"lever_m": lever})
    at.sess = SimpleNamespace(robot=SimpleNamespace())
    at.on_event = None
    at.confirm_mode = mode
    at.confirm_source = lambda: "y"
    at.lines = []
    at.log = at.lines.append
    return at


def test_autoteach_confirm_states_the_plan_and_labels_the_large_value_only_in_large_mode():
    lever, chord, rot = G1B_LEVER_M, G1B_CHORD_M, G1B_ROT_DEG     # 현 + 레버 × 회전 = 1893.0 mm (g1_b 의 프롬프트 값)
    pt = move_plan_text(G1B_JOINT_PLAN)
    at = _confirm_stub("each", lever=lever)
    at._confirm("start_pose", chord, extra="(x)", rot_deg=rot, kind="joint", plan_text=pt)
    assert at.lines[0].startswith(f"CONFIRM start_pose: {pt} (x) -> ") and "1893" not in at.lines[0], at.lines
    # 'large': 판단 값은 시작 → 끝 순변위의 상한이다 (경로 길이가 아니다 — g1_b 의 계획 경로 949.6 mm 는 현 929.2 mm 보다 길다)
    at = _confirm_stub("large", lever=lever)
    at._confirm("start_pose", chord, extra="(x)", rot_deg=rot, kind="joint", plan_text=pt)
    assert at.lines[0].startswith(f"CONFIRM start_pose: {pt} [asked because a tool point up to 317 mm from the flange can end up to "
                                  f"1893.0 mm from where it starts (start-to-end bound = flange chord 929.2 mm + 317 mm x 174.2 deg; not "
                                  f"a path length) > the 'large' threshold 50.0 mm] (x) -> "), at.lines
    assert "may move up to" not in at.lines[0]
    # 판단은 그대로: 'large' 에서 현은 작아도 회전이 문턱을 넘기면 묻고 (4d P10), 문턱 아래면 묻지 않는다
    at = _confirm_stub("large", lever=0.40)
    at._confirm("turn", 0.0, rot_deg=10.0, kind="local", plan_text="p")
    assert len(at.lines) == 2 and "can end up to 69.8 mm from where it starts" in at.lines[0]
    at = _confirm_stub("large", lever=0.40)
    at._confirm("nudge", 0.004, rot_deg=1.0, kind="local", plan_text="p")
    assert at.lines == []
    # 거절 사유도 계획을 말한다
    at = _confirm_stub("each", lever=lever)
    at.confirm_source = lambda: "n"
    with pytest.raises(Exception, match=r"operator declined move 'start_pose' \(joint-space straight line: flange path 949.6 mm"):
        at._confirm("start_pose", chord, rot_deg=rot, kind="joint", plan_text=pt)


def test_autoteach_confirm_plans_a_local_move_only_when_it_asks():
    """국부 이동의 계획 (prepare_leg) 은 묻는 이동에서만 프롬프트 전에 — 묻지 않는 이동은 예전처럼 센서를 켜기 직전에 (_local_move)."""
    calls = []

    def plan_text():
        calls.append(1)
        return move_plan_text({"kind": "cartesian", "length_m": 0.006, "rotation_deg": 0.0, "duration_s": 0.9})
    at = _confirm_stub("scan")
    at._confirm("probe_X", 0.006, kind="local", plan_text=plan_text)
    assert calls == [] and at.lines == []
    at._confirm("pass1_S_to_E", 0.38, kind="scan", plan_text=plan_text)
    assert calls == [1] and at.lines[0].startswith("CONFIRM pass1_S_to_E: Cartesian straight line: flange path 6.0 mm, "), at.lines


def test_check_leg_reports_capability_and_result_by_structure():
    """확인 문구용 계획 (check_leg): 보내지도 저장하지도 않는다. 기능 없음 (None) / 결과 {ok, reason, plan} 를 구조로 돌려준다 —
    move_plan_text 가 '보낼 때 계획하는 백엔드' 와 '지금 자세에서 거부' 를 문자열 맞추기 없이 구분한다."""
    from scan_l0.executor import check_leg
    w = SimpleNamespace(T_base_flange=np.eye(4), accel_mps2=0.1)

    def sess(be):
        return SimpleNamespace(robot=SimpleNamespace(backend=be))
    assert check_leg(sess(SimpleNamespace(move_linear=lambda *a, **k: None)), w, 0.03) is None       # RB 제어박스 어댑터 모양
    assert check_leg(SimpleNamespace(robot=None), w, 0.03) is None
    calls = []

    class Be:
        _prepared = "untouched"

        def check_linear(self, T, v, a, *, timeout_s=None):
            calls.append((v, a, timeout_s))
            return {"ok": True, "reason": "", "plan": {"kind": "cartesian", "length_m": 0.04, "duration_s": 1.6}}
    be = Be()
    r = check_leg(sess(be), w, 0.03, timeout_s=7.0)
    assert r == {"ok": True, "reason": "", "plan": {"kind": "cartesian", "length_m": 0.04, "duration_s": 1.6}}
    assert calls == [(0.03, 0.1, 7.0)] and be._prepared == "untouched"
    be.check_linear = lambda *a, **k: {"ok": False, "reason": "straight line collides with the scene: specimen", "plan": None}
    assert check_leg(sess(be), w, 0.03) == {"ok": False, "reason": "straight line collides with the scene: specimen", "plan": None}

    def boom(*a, **k):
        raise RuntimeError("tf lookup failed")
    be.check_linear = boom
    assert check_leg(sess(be), w, 0.03) == {"ok": False, "reason": "RuntimeError: tf lookup failed", "plan": None}


def test_joint_path_stats_separates_flange_translation_and_rotation():
    """관절 경로의 플랜지 병진·회전을 따로 (재시간 경로 변수 S = 점마다 max(병진, 회전 × 반경) 의 누적은 이동 거리가 아니다)."""
    from scan_core.trajectory import densify_joint_path, joint_path_stats, retime_joint_path
    r = 0.6

    def fk(q):                                               # 기초 관절 하나가 돌리는 플랜지 (반경 r), 둘째 관절은 플랜지에 닿지 않는다
        c, s_ = math.cos(q[0]), math.sin(q[0])
        T = np.eye(4)
        T[:2, :2] = [[c, -s_], [s_, c]]
        T[:3, 3] = [r * c, r * s_, 0.0]
        return T
    Q = densify_joint_path([[0.0, 0.0], [math.pi / 2, -0.1]], 0.01)
    st = joint_path_stats(Q, fk, ["base", "wrist3"])
    assert st["flange_translation_m"] == pytest.approx(r * math.pi / 2, rel=1e-4)           # 호 (현 0.849 m 가 아니다)
    assert st["flange_rotation_deg"] == pytest.approx(90.0, abs=1e-6)
    assert st["max_joint"] == "base" and st["max_joint_index"] == 0 and st["max_joint_rotation_deg"] == pytest.approx(90.0)
    assert st["joint_rotation_deg"][1] == pytest.approx(math.degrees(0.1)) and st["max_joint_net_deg"] == pytest.approx(90.0)
    rt = retime_joint_path(Q, fk, 0.05, 0.2, rotation_radius_m=1.0)
    assert rt.S == pytest.approx(math.pi / 2, rel=1e-4) and rt.S > 1.6 * st["flange_translation_m"]       # 섞은 값 (예전 'path')
    assert joint_path_stats(Q[:1], fk)["flange_translation_m"] == 0.0


# --------------------------------------------------------------------------------------
# dwell classification / search order
# --------------------------------------------------------------------------------------
def _loc(n_valid=1000, groove=False, left=True, right=True, plate_ok=True) -> StaticLocalization:
    f = CrossSectionFeatures(ok=True, n_valid=n_valid, min_points=40, groove_found=groove,     # 추출기 최소 점 수 (4d P5)
                             plate_left={"tilt_deg": 0.0} if left else None, plate_right={"tilt_deg": 0.0} if right else None)
    loc = StaticLocalization(features=f, T_base_opt=np.eye(4), opt_origin_base=np.zeros(3), groove_found=groove)
    loc.diag = {"n_valid": n_valid}
    if plate_ok:
        loc.plate_normal_base = np.array([0, 0, 1.0])
        loc.standoff_m = 0.4
    return loc


def test_classify_localization_and_search_order():
    assert classify_localization(_loc(n_valid=5)) == "no_surface"
    assert classify_localization(_loc(groove=True)) == "groove"
    assert classify_localization(_loc(left=True, right=False)) == "plate_edge"
    assert classify_localization(_loc(left=False, right=True)) == "plate_edge"
    assert classify_localization(_loc()) == "plate_only"
    assert classify_localization(_loc(plate_ok=False)) == "no_surface"
    steps = (75.0, -75.0, 150.0, -150.0)
    assert search_order(steps, _loc()) == [75.0, -75.0, 150.0, -150.0]
    assert search_order(steps, _loc(left=True, right=False)) == [-75.0, -150.0, 75.0, 150.0]     # 왼 판만 → 그루브는 −X 쪽
    assert search_order(steps, _loc(left=False, right=True)) == [75.0, 150.0, -75.0, -150.0]


# --------------------------------------------------------------------------------------
# pass residuals
# --------------------------------------------------------------------------------------
def test_pass_residuals_regression_signs():
    M = 40
    s = np.linspace(0.0, 0.3, M)
    u = np.array([0.0, 1.0, 0.0])
    T = np.tile(np.eye(4), (M, 1, 1))
    T[:, :3, :3] = np.column_stack([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])            # X=-x, Y=+y, Z=-z (looking down)
    T[:, :3, 3] = np.array([0.6, 0.0, 0.45]) + np.outer(s, u)
    sf = SeamScanFeatures(n_profiles=M, indices=np.arange(M), ok=np.ones(M, dtype=bool))
    sf.groove_centre_x_mm = 2.0 + 20.0 * s                  # +20 mm/m drift → psi = atan(0.02) = +1.146 deg
    sf.groove_centre_z_mm = 400.0 - 5.0 * s                 # range decreases along the travel → theta +0.286 deg (approaching)
    sf.root_centre_x_mm = sf.groove_centre_x_mm + 2.9
    sf.tilt_deg = np.full(M, 0.7)
    result = SimpleNamespace(T_base_opt=T)
    r = pass_residuals(result, sf, None, u, target_standoff_m=0.40, gravity_axis_base=G_LAB)
    assert r["psi_deg"] == pytest.approx(math.degrees(math.atan(0.02)), abs=1e-6)
    assert r["theta_deg"] == pytest.approx(math.degrees(math.atan(0.005)), abs=1e-6)
    assert r["roll_deg"] == pytest.approx(0.7) and r["n_ok"] == M
    assert r["lateral_m"] == pytest.approx(np.median(sf.groove_centre_x_mm) * 1e-3)
    assert r["lateral_root_m"] == pytest.approx(r["lateral_m"] + 2.9e-3)
    assert r["theta_source"] == "plate_offset_vs_travel" and math.isnan(r["standoff_m"])   # standoff 는 판 법선 fit 이 있을 때만
    # 그루브가 거의 없으면 NaN
    sf.ok[:] = False
    r2 = pass_residuals(result, sf, None, u, target_standoff_m=0.40, gravity_axis_base=G_LAB)
    assert all(math.isnan(r2[k]) for k in ("psi_deg", "theta_deg", "roll_deg", "lateral_m"))


def test_robust_slope_ignores_outliers():
    x = np.linspace(0, 1, 50)
    y = 3.0 * x + 1.0
    y[5] += 100.0
    y[30] -= 50.0
    assert _robust_slope(x, y) == pytest.approx(3.0, abs=1e-9)
    assert math.isnan(_robust_slope(np.array([1.0, 1.0]), np.array([0.0, 1.0])))


# --------------------------------------------------------------------------------------
# delta_cb
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# review 2026-09-30: CB command frame (C1/C3), start-pose gate (C8), leg phase tags (M-2), psi consistency (M-9),
# tilt-aware pass residuals (M-1), confirm drain (M-11), ts tolerance (M-13), 3G measured plane (C6)
# --------------------------------------------------------------------------------------


def test_start_pose_mismatch_thresholds():
    from poses import from_mm_deg as from_move_l
    from scan_l0.executor import start_pose_mismatch

    tol = (0.001, 0.1)                                                     # 플랫폼 도착 허용 (SafetyPolicy.goal_tol_m / goal_tol_deg)
    A = from_move_l([600.0, 0.0, 500.0, 180.0, 0.0, 0.0])
    assert start_pose_mismatch(A, from_move_l([600.5, 0.0, 500.0, 180.0, 0.0, 0.0]), *tol) is None          # 0.5 mm OK
    assert start_pose_mismatch(A, from_move_l([600.0, 0.0, 500.0, 180.0, 0.05, 0.0]), *tol) is None         # 0.05 deg OK
    assert "1.2 mm" in (start_pose_mismatch(A, from_move_l([601.2, 0.0, 500.0, 180.0, 0.0, 0.0]), *tol) or "")
    assert "0.20 deg" in (start_pose_mismatch(A, from_move_l([600.0, 0.0, 500.0, 180.0, 0.2, 0.0]), *tol) or "")
    assert start_pose_mismatch(None, A, *tol) is None


def test_leg_phase_tags_in_plane_and_rotation_as_scan():
    from scan_l0.autoteach import AutoTeach
    ph = AutoTeach._leg_phase
    assert ph(+0.05, 0.0) == "ascend" and ph(-0.02, 0.0) == "descend"
    e = 0.0021                                                             # 도착 허용 (안전 정책 pose_tol_m)
    assert ph(+0.05, 0.001, tol_m=e) == "ascend"                           # 면내 ≤ 도착 허용은 순수 +n 으로 본다 (도착 오차)
    assert ph(+0.05, 0.003, tol_m=e) == "scan" and ph(-0.02, 0.01, tol_m=e) == "scan"   # 면내 > 도착 허용 → 'scan' (clamp 면제 아님)
    assert ph(+0.05, 0.001) == "scan"                                      # 허용 없음 = 면내 성분이 조금이라도 있으면 'scan'
    assert ph(+0.05, 0.0, roll_deg=0.5) == "scan"                          # 재자세 → 'scan'
    assert ph(0.0, 0.0) == "scan"


def test_pass_residuals_theta_is_tilt_aware_and_standoff_uses_gravity_axis():
    M = 40
    s = np.linspace(0.0, 0.3, M)
    u = np.array([0.0, 1.0, 0.0])
    T = np.tile(np.eye(4), (M, 1, 1))
    T[:, :3, :3] = np.column_stack([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
    T[:, :3, 3] = np.array([0.6, 0.0, 0.45]) + np.outer(s, u)
    sf = SeamScanFeatures(n_profiles=M, indices=np.arange(M), ok=np.ones(M, dtype=bool))
    sf.groove_centre_x_mm = np.zeros(M)
    sf.groove_centre_z_mm = 400.0 - 5.0 * s                                # 판이 진행하며 5 mm/m 올라온다 (접근 = +)
    sf.root_centre_x_mm = np.zeros(M)
    sf.tilt_deg = np.zeros(M)
    result = SimpleNamespace(T_base_opt=T)
    # fit 이 있으면 theta = 판 법선 standoff 의 진행 드리프트 (base 3D)
    fit = PassFit(c=np.array([0.6, 0.0, 0.05]), u=u, n=np.array([0.0, 0.0, 1.0]))
    r = pass_residuals(result, sf, fit, u, target_standoff_m=0.40, gravity_axis_base=G_LAB)
    assert r["theta_source"] == "plate_normal_standoff_vs_s" and r["theta_deg"] == pytest.approx(math.degrees(math.atan(0.005)), abs=1e-6)
    assert r["theta_raw_z_deg"] == pytest.approx(r["theta_deg"], abs=1e-6)             # 수평 진행·평판이면 원시 z 드리프트와 같다
    # 기울어진 법선: theta 는 법선 방향 standoff 로 (원시 z 가 아니라) — n 을 진행축 둘레로 기울여도 진행 중 standoff 변화율만 본다
    n_t = unit([0.0, -math.sin(math.radians(2.0)), math.cos(math.radians(2.0))])
    fit2 = PassFit(c=np.array([0.6, 0.0, 0.05]), u=unit([0.0, math.cos(math.radians(2.0)), math.sin(math.radians(2.0))]), n=n_t)
    r2 = pass_residuals(result, sf, fit2, u, target_standoff_m=0.40, gravity_axis_base=G_LAB)
    assert np.isfinite(r2["theta_deg"]) and r2["theta_source"] == "plate_normal_standoff_vs_s"
    # fit 없음 + 중력축이 −z 가 아닐 때: theta 는 진행선에 수직인 연직 방향 m (= up, 여기서 +x) 의 상부 중심 오프셋 드리프트
    g = np.array([-1.0, 0.0, 0.0])                                         # 'up' = +x (진행 +y 는 여전히 수평)
    r3 = pass_residuals(result, sf, None, u, target_standoff_m=0.40, gravity_axis_base=g)
    assert r3["theta_source"] == "plate_offset_vs_travel" and r3["theta_deg"] == pytest.approx(0.0, abs=1e-9)
    # F-C10: fit 없음 + 수직 진행 (중력 ∥ 진행: 3G) → theta NaN, 'unavailable', standoff 없음
    g_v = np.array([0.0, -1.0, 0.0])                                       # 'up' = +y = 진행 방향
    r3v = pass_residuals(result, sf, None, u, target_standoff_m=0.40, gravity_axis_base=g_v)
    assert r3v["theta_source"] == "unavailable" and math.isnan(r3v["theta_deg"]) and "standoff_measured_m" not in r3v
    assert math.isnan(r3v["standoff_m"])
    # 기운 진행 (진행이 위로 10°): 진행 자체의 높이 변화는 빠지고 판 vs 진행 피치만 남는다
    a = math.radians(10.0)
    u_i = np.array([0.0, math.cos(a), math.sin(a)])
    T_i = T.copy()
    T_i[:, :3, 3] = np.array([0.6, 0.0, 0.45]) + np.outer(s, u_i)
    sf_i = SeamScanFeatures(n_profiles=M, indices=np.arange(M), ok=np.ones(M, dtype=bool))
    sf_i.groove_centre_x_mm = np.zeros(M)
    sf_i.groove_centre_z_mm = np.full(M, 400.0)                            # 센서 거리 일정 (판이 진행과 평행) → 피치 0
    sf_i.root_centre_x_mm = np.zeros(M)
    sf_i.tilt_deg = np.zeros(M)
    r5 = pass_residuals(SimpleNamespace(T_base_opt=T_i), sf_i, None, u_i, target_standoff_m=0.40, gravity_axis_base=G_LAB)
    assert r5["theta_source"] == "plate_offset_vs_travel" and abs(r5["theta_deg"]) < 0.05, r5


def test_stall_flag_tolerance_is_a_frame_period_minus_two_timestamp_quanta():
    """촬영 멈춤 표시 문턱은 설정이 아니라 프레임 주기 − 타임스탬프 양자 둘 (2026-10-08 4d): 100 fps → 8 ms (실기 떨림 ≤ 6 ms)."""
    from scan_core.clock_sync import stall_tol_ms
    assert stall_tol_ms(100.0) == pytest.approx(8.0) and stall_tol_ms(20.0) == pytest.approx(48.0)
    assert not hasattr(C.Settings({}), "ts_tol_ms")


def test_3g_measured_plate_plane_raises_envelope_floor(tmp_path):
    """C6: 3G 드웰의 실측 판 평면(점·법선) → job_ref 평면을 로봇 쪽으로만 옮긴다 (raise-only). 거친 평면보다 20 mm 로봇 쪽인
    실측 평면 뒤에는, 실측 평면에서 235 mm 에 끝나는 로컬 이동이 never_cross_plate_plane 으로 REFUSED (거친 평면 기준 255 mm 는 통과)."""
    import os
    from conftest import CALIB, JOBS
    from scan_core.path_gen import EnvelopeError, sensor_frames
    from scan_l0.autoteach import AutoTeach, make_coarse_job
    from scan_l0.planning import make_local_move_plan

    settings = C.load_settings(CONFIG)
    calib = C.load_calib(CALIB)
    from torch_fixtures import CAD_TORCH_KEYPOINT                                    # the 3G envelope rule's test, built on the CAD
    job0, doc = C.load_job(os.path.join(JOBS, "groove_10T_3G.yaml"))                   # torch's keypoint (explicit fixture)
    job0.torch_lateral_m = tuple(CAD_TORCH_KEYPOINT["torch_lateral_m"])
    job0.torch_protrusion_m = CAD_TORCH_KEYPOINT["torch_protrusion_m"]
    job0.validate()
    at = AutoTeach(settings, job0, doc, calib, out_dir=str(tmp_path), log=lambda s: None,
                   session=ScanSession(robot=None, sensor=None, owned=False))
    at.orientation = job0.resolved_orientation()
    assert at.orientation == "vertical_3G"
    at.up = job0.up_unit
    at.job_c = make_coarse_job(ScanJob.from_dict({**C.job_to_dict(job0), "safe_clearance_m": 0.05}), at._sv(), depth_m=at._depth_spec_m())   # 한계 L ≈ 정책 여유
    at.u0, at.n0 = at.job_c.unit_dir, at.job_c.normal_unit
    at.job_ref = ScanJob.from_dict(C.job_to_dict(at.job_c))
    n0 = at.n0
    start0 = at.job_ref.start.copy()

    def loc_on_plane(shift_m: float) -> StaticLocalization:
        p0 = start0 + n0 * shift_m
        lat = np.cross(n0, at.u0)
        pts = np.array([p0 + lat * x + at.u0 * 0.001 * (i % 3) for i, x in enumerate(np.linspace(-0.05, 0.05, 21))])
        f = CrossSectionFeatures(ok=True, n_valid=1000, groove_found=True, plate_left={"tilt_deg": 0.0}, plate_right={"tilt_deg": 0.0})
        loc = StaticLocalization(features=f, T_base_opt=np.eye(4), opt_origin_base=p0 + n0 * 0.40, groove_found=True)
        loc.plate_points_base = pts
        loc.plate_normal_base = n0.copy()
        loc.plate_dir_base = lat
        loc.standoff_m = 0.40
        loc.diag = {"n_valid": 1000}
        return loc

    # 1) 실측 평면이 거친 평면보다 20 mm 로봇 쪽 → job_ref 가 옮겨진다
    at._raise_plate_top(loc_on_plane(+0.020))
    np.testing.assert_allclose(at.job_ref.start, start0 + n0 * 0.020, atol=1e-9)
    np.testing.assert_allclose(at.job_ref.end, at.job_c.end + n0 * 0.020, atol=1e-9)
    assert at.plate_top is None and at.plate_plane_measured is not None
    # 2) 더 먼 평면(−10 mm) 은 무시 (raise-only) — job_ref 그대로
    at._raise_plate_top(loc_on_plane(-0.010))
    np.testing.assert_allclose(at.job_ref.start, start0 + n0 * 0.020, atol=1e-9)
    # 3) 포락선: 실측 평면에서 235 mm 에 끝나는 판 쪽 이동은 REFUSED, 거친 평면 기준(255 mm) 이면 통과했을 이동
    job_rough = ScanJob.from_dict(C.job_to_dict(at.job_c))
    fr = sensor_frames(job_rough, calib)
    T_from = fr.T_base_flange(0.05)
    d0 = float(((T_from @ calib.T_flange_opt)[:3, 3] - start0) @ n0)     # 광학 원점 → 거친 평면 거리
    T_to = np.array(T_from, dtype=float, copy=True)
    T_to[:3, 3] -= n0 * (d0 - 0.255)
    _plan, stats = make_local_move_plan(T_from, T_to, "approach_normal", job_rough, calib, speed_mps=0.02, accel_mps2=0.2)
    assert stats["min_plate_dist_opt"] == pytest.approx(0.255, abs=1e-6)
    with pytest.raises(EnvelopeError, match="never_cross_plate_plane"):
        make_local_move_plan(T_from, T_to, "approach_normal", at.job_ref, calib, speed_mps=0.02, accel_mps2=0.2)


def test_pass_cloud_only_when_someone_listens_and_after_the_pass_result():
    """처리된 패스 점구름 (RViz, 사용자 2026-10-08): 지금 볼 쪽이 있을 때만 (pass_cloud_wanted — ROS 서버: ~/pass_cloud 구독자 수) 간격을
    정하고, 패스 결과를 늦추지 않는다 — 작업 프로세스 (Future) 또는 서버 안 대체 경로 (판정 뒤 스레드) 가 다 만들면 콜백이 이벤트를 낸다.
    작업 프로세스의 재구성 결과는 vis_cloud 가 한 번 쓰고 비운다."""
    import threading
    from concurrent.futures import Future
    from scan_core.weld_plan import WeldProcess
    from scan_l0 import post_scan
    from scan_l0.autoteach import AutoTeach, AutoTeachResult
    at = AutoTeach.__new__(AutoTeach)
    at.doc, at.res = {}, AutoTeachResult()
    at.job0 = SimpleNamespace(base_frame="link0")
    events, got = [], threading.Event()
    at.on_event = lambda name, data: (events.append((name, data)), got.set())
    at._leg_cloud, at._vis_clouds = None, []
    step = WeldProcess.from_dict(None).dense_step_mm * 1e-3
    for wanted, exp in ((None, None), (lambda: False, None), (lambda: 1 / 0, None), (lambda: True, step)):
        at.pass_cloud_wanted = wanted
        assert at._vis_step_m() == exp
    at.on_event = None
    assert at._vis_step_m() is None                                  # 이벤트를 받을 쪽이 없으면 묻지도 않는다
    at.on_event = lambda name, data: (events.append((name, data)), got.set())
    # 재구성 결과 (2 프로파일 × 3 점, 실측 자세 = 단위 자세 + 심 방향 1 mm)
    T = np.repeat(np.eye(4)[None], 2, axis=0)
    T[1, 1, 3] = 1e-3
    res = SimpleNamespace(x_mm=np.array([[-1.0, 0.0, 1.0]] * 2), z_mm=np.full((2, 3), 300.0), valid=np.ones((2, 3), bool), T_base_opt=T)
    # 서버 안 대체 경로: (재구성 결과, 간격) → 판정 뒤 스레드
    at._leg_cloud = (res, step)
    at._emit_pass_cloud(1, "pass_1")
    assert at._leg_cloud is None and got.wait(5.0)
    (name, d), = events
    assert name == "pass_cloud" and d["k"] == 1 and d["base_frame"] == "link0" and d["points"].shape == (6, 3)
    np.testing.assert_allclose(d["points"][:, 2], 0.3, atol=1e-6)
    # 작업 프로세스 경로: Future (패스 결과 뒤 맡긴 vis_cloud) — 아직 안 끝났으면 기다리지 않고, 끝나면 콜백이 낸다
    events.clear()
    got.clear()
    fut = Future()
    at._leg_cloud = fut
    at._emit_pass_cloud(2, "pass_2")
    assert events == []
    post_scan._VIS_SRC = (res, step)
    out = post_scan.vis_cloud()
    assert post_scan._VIS_SRC is None and post_scan.vis_cloud() is None and out[0].shape == (6, 3)
    fut.set_result(out)
    assert got.wait(5.0) and events[0][1]["k"] == 2
    assert [(k, n_, w_) for k, n_, _t, w_ in sorted(at._vis_clouds)] == [(1, 6, "server thread"), (2, 6, "worker")]
