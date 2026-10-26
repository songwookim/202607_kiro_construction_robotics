"""P2 레이저 선 방향 (툴 yaw) 정렬 — 단위 (2026-10-09): 유도량 (연구실 값), S_align 목표 불변식, 측정 심 수직 보정, 강체 변환 국소화,
장면 판단 활성 규칙, X 축 프로브 판정, w 방향 스케일 읽기.

모두 기하·모형에서 유도한 값과 비교한다 (조정 상수 없음): ψ_tol = asin(측면 허용 / (FOV/2)), b* = √2·tol_root / (ψ_tol − θ_goal),
m_E = (FOV/2)·sin ψ + e_hit — r = 443 mm 에서 1.82°, 49.9 mm, 5.9 mm (명세 1 절의 1.81° / 50.4 mm / 5.9 mm 와 같은 식)."""
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import CALIB, JOBS

from scan_core.frames import make_transform, rot_x, rot_z, unit
from scan_core.localize import (StaticLocalization, apply_correction, localize_static, relative_correction, transform_localization)
from scan_core.path_gen import plate_frame
from scan_core.seam_track import SeamEstimate, yaw_to_seam_rad
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, static_profile
from scan_core.types import Calib
from scan_l0 import config as C
from scan_l0.autoteach import AutoTeach, probe_verdict_x, site_pre_laterals
from scan_l0.planning import seam_align_target
from scan_l0.session import ScanSession

STACK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAB_CONFIG = os.path.join(STACK, "scan_l0", "tests", "data", "hongik_lab_rb20_values.yaml")   # 연구실 플랫폼 값 (TCP 30 mm/s; 시험 자료)
R0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)       # X_opt = base +y (선이 심을 가로지름), Z_opt = −z
H = 0.05
T_PLATE = plate_frame([0.6, 0.0, H], [1, 0, 0], [0, 0, 1])           # 심 = base +x, 판 법선 +z
CFG = SynthConfig(n_points=1408, seed=1, sigma_z_mm=0.0, dropout_prob=0.0)
U = np.array([1.0, 0.0, 0.0])
N = np.array([0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def calib() -> Calib:
    return C.load_calib(CALIB)


@pytest.fixture(scope="module")
def lab_at(tmp_path_factory):
    """연구실 플랫폼 설정 (TCP 30 mm/s) 의 AutoTeach — 이동 없이 유도량만 (안전 정책·센서 시야 모형·1G 거친 작업)."""
    settings = C.load_settings(LAB_CONFIG)
    calib = C.attach_tool_body(C.load_calib(CALIB), settings)
    job0, doc = C.load_job(os.path.join(JOBS, "groove_10T_1G.yaml"))
    at = AutoTeach(settings, job0, doc, calib, out_dir=str(tmp_path_factory.mktemp("at")), log=lambda s: None,
                   session=ScanSession(robot=None, sensor=None, owned=False))
    lim = (settings.params.get("moveit_jtc") or {}).get("limits") or {}
    at.cfg = SimpleNamespace(v_eff_max_mps=float(lim["max_tcp_speed_mps"]))
    at._apply_safety_policy(at.job0)
    at.orientation = "flat_1G"
    at._derive_jobs(at.job0)
    return at


def _pose(calib, yaw_deg: float, lateral_m: float, *, standoff: float = 0.443, pitch_deg: float = 0.0, s: float = 0.0):
    """광학 자세: 판 법선 둘레 yaw, 광학 X 둘레 pitch (광축이 진행 방향으로 기움 — 입사 기울기), 광축이 심 점 s 에서 측면 lateral 을 standoff
    로 겨눈다 → 플랜지 자세."""
    R = rot_z(math.radians(yaw_deg)) @ R0 @ rot_x(math.radians(pitch_deg))
    target = np.array([0.6 + s, -lateral_m, H])
    o = target - standoff * R[:, 2]
    return make_transform(R, o) @ np.linalg.inv(calib.T_flange_opt)


def _loc(calib, T_bf):
    scene = GrooveScene.from_spec(GrooveSpec(bevel_side="left"), yc=0.0)
    x, z = static_profile(T_bf, calib, scene, T_PLATE, CFG)
    loc = localize_static(T_bf, calib, x, z, GrooveSpec(bevel_side="auto"), travel_dir_hint=U, plate_normal_hint=N)
    assert loc.groove_found and loc.plate_ok
    return loc


def _axis_hit(T_bf, calib):
    T = T_bf @ calib.T_flange_opt
    o, Z = T[:3, 3], T[:3, 2]
    t = (H - o[2]) / Z[2]
    return o + Z * t


def _est(u=U, n=N, **kw):
    d = dict(c=np.array([0.6, 0.0, H]), u=np.asarray(u, float), n=np.asarray(n, float), track="right", n_frames=30, n_inliers=30,
             coverage=1.0, mad_m=2e-5, s_first=-0.05, s_last=0.0, sections=np.zeros((0, 2)), length_m=0.05,
             sigma_psi_model=math.sqrt(2) * (1e-3 / 3) / 0.05, source="S_track", w=np.cross(n, u))
    d.update(kw)
    return SeamEstimate(**d)


# --------------------------------------------------------------------------------------
# 유도량 (1 절)
# --------------------------------------------------------------------------------------
def test_lab_derived_quantities(lab_at):
    at = lab_at
    r = 0.443
    assert at._sv().fov_m(r) == pytest.approx(0.1869, abs=2e-4)
    assert at._hit_tol_m(r) == pytest.approx(2.97e-3, abs=0.05e-3)
    from scan_core import measure as _ms
    assert at._tol_lat_m(r) == pytest.approx(max(at._hit_tol_m(r), 3.0 * _ms.quant_sigma_mm(443.0) * 1e-3))
    psi = at._yaw_tol_rad(r)
    assert math.degrees(psi) == pytest.approx(1.82, abs=0.02)
    assert math.degrees(at._theta_goal_rad()) == pytest.approx(0.2)
    assert at._tol_root_m() == pytest.approx(1.0e-3)                     # qc_targets root_position_mm.tol
    b, basis = at._track_length_m(r)
    assert basis["b_star_m"] == pytest.approx(math.sqrt(2) * 1e-3 / (psi - math.radians(0.2)), rel=1e-9)
    assert b == pytest.approx(0.050, abs=0.001) and b == basis["b_star_m"]          # 템플릿 대략점 (380 mm) 은 상한에 걸리지 않는다
    assert basis["b_floor_m"] < b <= basis["cap_m"] and at._baseline_refusal(basis) is None
    assert at._footprint_m(r, psi, 0.0) == pytest.approx(5.9e-3, abs=0.1e-3)       # m_E
    est = _est()
    assert math.degrees(at._psi_res_rad(est, 0.0)) == pytest.approx(math.degrees(psi), abs=0.01)   # k·σψ(b*) + θ_goal = ψ_tol
    assert at._psi_res_rad(est, math.radians(2.0)) == pytest.approx(at._psi_res_rad(est) + math.radians(2.0))


def test_no_design_root_tolerance_starts_at_the_measurement_floor_and_refuses_without_room(lab_at):
    """root_position_mm 이 없는 작업 (2026-10-07 전 3G 정련 작업 등, D5): 다른 설계 허용 (갭) 으로 바꾸지 않는다 — 이동 전 σ_0 = 측정 바닥
    (센서 양자화) 로 시작하고 S 추적의 잰 잔차가 기선을 늘린다 (_track_extension). 기하 상한이 측정 바닥 기선보다 짧으면 이동 전 거부 (D8:
    대략점·설계만의 함수)."""
    from scan_core import measure as _ms
    at = lab_at
    saved = at.qc_targets
    try:
        at.qc_targets = {k: v for k, v in saved.items() if k != "root_position_mm"}
        assert "root_gap_mm" in at.qc_targets and at._tol_root_m() is None
        b, bb = at._track_length_m(0.443)
        sig_q = _ms.quant_sigma_mm(443.0) * 1e-3
        assert bb["sigma_sec0_m"] == pytest.approx(sig_q) and "floor" in bb["sigma_sec0_source"]
        assert b == pytest.approx(bb["b_floor_m"]) and 0.004 < b < 0.008
        assert at._baseline_refusal(bb) is None
        assert at._track_fit_args(U, 0.443, "S_track")["tol_root_m"] is None
    finally:
        at.qc_targets = saved
    _b, bb = at._track_length_m(0.443)
    for cap in (0.5 * bb["b_floor_m"], 0.0, -0.01):                    # 기하 상한 < 측정 바닥 기선 (≤ 0 포함) → 이동 전 거부
        why = at._baseline_refusal(dict(bb, cap_m=cap))
        assert why and "cannot be measured" in why and "farther apart" in why
    assert at._baseline_refusal(dict(bb, cap_m=bb["b_floor_m"])) is None


def test_track_extension_from_the_measured_residual(lab_at, calib, monkeypatch):
    """S 추적 연장 (D3/D5): 잰 단면 잡음이 요구하는 기선 b_need(σ) 보다 직선 위 단면 범위가 짧으면 그 모자람만큼 (레이저 평면 법선 Y 기준
    심 방향 환산), 그루브가 늦게 나타나 직선이 없으면 계획 기선만큼 — 기하 상한 안에서, 착점 도착 허용보다 큰 이동만."""
    at = lab_at
    r = 0.443
    b, bb = at._track_length_m(r)
    T = _pose(calib, 0.0, 0.0)                                          # Y_opt ∥ 심 (U)
    monkeypatch.setattr(at, "_T_now", lambda: T)
    k = float(at.settings.judgement_k)
    e_hit = at._hit_tol_m(r)
    est = _est(sigma_sec_m=1e-3 / k, length_m=b, sigma_sec_source="design")
    assert at._track_extension(est, "ok", {}, bb, b, r, U) == (0.0, "")             # 설계 기선 그대로 — 연장 없음
    est2 = _est(sigma_sec_m=1.5e-3 / k, length_m=b, sigma_sec_source="measured")   # 잰 잡음이 설계 상한의 1.5 배
    e, why = at._track_extension(est2, "ok", {}, bb, b, r, U)
    assert e == pytest.approx(min(0.5 * b, bb["cap_m"] - b), rel=1e-6) and "measured section noise" in why
    est3 = _est(sigma_sec_m=1e-3 / k, length_m=b - 0.5 * e_hit)                     # 도착 허용 안의 모자람: 움직이지 않는다
    assert at._track_extension(est3, "ok", {}, bb, b, r, U)[0] == 0.0
    e4, why4 = at._track_extension(None, "late", {"why": "x"}, bb, b, r, U)
    assert e4 == pytest.approx(min(b, bb["cap_m"] - b)) and "late" in why4
    assert at._track_extension(None, "none", {}, bb, b, r, U) == (0.0, "")
    assert at._track_extension(est2, "ok", {}, bb, bb["cap_m"], r, U)[0] == 0.0     # 기하 상한에 닿았다 — 여유에 전파


# --------------------------------------------------------------------------------------
# 측정 심에 수직인 보정 (localize.relative_correction seam_dir) · 강체 변환 국소화
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("pitch", [0.0, 25.0])
def test_correction_perpendicular_to_the_measured_seam_does_not_slide_along_it(calib, pitch):
    """선이 심에 12° 기운 드웰 (측면 30 mm, 광축이 진행 방향으로 pitch 기움): 예전 보정 (선 방향 e_lat) 은 착점을 심 방향으로 Δl·tan ψ
    밀고, 측정 심 수직 보정은 착점의 심 방향 이동이 0 이고 한 번에 선이 착점에서 심을 지난다 (기운 광축이어도)."""
    T = _pose(calib, 12.0, 0.030, pitch_deg=pitch)
    loc = _loc(calib, T)
    target = loc.standoff_m
    old = relative_correction(loc, target_standoff_m=target)
    new = relative_correction(loc, target_standoff_m=target, seam_dir=U)
    assert new.aim == "top_centre_perp_seam" and old.aim == "top_centre"
    A0 = _axis_hit(T, calib)
    A_old = _axis_hit(apply_correction(T, calib, old), calib)
    A_new = _axis_hit(apply_correction(T, calib, new), calib)
    assert abs(float((A_old - A0) @ U)) > 5e-3                          # 예전: 기운 선을 따라 심 방향으로 미끄러진다
    assert abs(float((A_new - A0) @ U)) < 1e-6                          # 새: 심 방향 이동 0
    loc2 = _loc(calib, apply_correction(T, calib, new))
    assert abs(loc2.lateral_offset_m) < 0.2e-3                          # 한 번에 착점에서 심을 지난다
    assert new.uncapped["dropped_along_travel_m"] == pytest.approx(float((np.asarray(new.target_base) - loc.opt_origin_base) @ U))


def test_seam_dir_none_is_the_previous_correction(calib):
    loc = _loc(calib, _pose(calib, 7.0, -0.02, pitch_deg=10.0))
    a = relative_correction(loc, target_standoff_m=0.42)
    b = relative_correction(loc, target_standoff_m=0.42, seam_dir=None)
    assert a.to_dict() == b.to_dict()
    n = loc.plate_normal_base
    delta = loc.top_centre_base + n * 0.42 - loc.opt_origin_base
    e = loc.lateral_axis_base
    np.testing.assert_allclose(a.d_base, float(delta @ e) * e + float(delta @ n) * n, atol=1e-12)


def test_transform_localization_moves_sensor_quantities_only(calib):
    T = _pose(calib, 15.0, 0.025)
    loc = _loc(calib, T)
    G = loc.top_centre_base
    from scan_l0.planning import local_target_from_correction
    T1 = local_target_from_correction(T, np.zeros(3), roll_axis_base=N, roll_deg=-15.0, pivot_base=G)
    lr = transform_localization(loc, T1 @ np.linalg.inv(T))
    ref = _loc(calib, T1)                                                # 그 자세에서 실제로 다시 잰 것
    np.testing.assert_allclose(lr.top_centre_base, G, atol=1e-12)
    np.testing.assert_allclose(lr.T_base_opt, T1 @ calib.T_flange_opt, atol=1e-12)
    assert lr.standoff_m == pytest.approx(loc.standoff_m, abs=1e-6) and lr.standoff_m == pytest.approx(ref.standoff_m, abs=0.2e-3)
    np.testing.assert_allclose(lr.axis_hit_base, ref.axis_hit_base, atol=0.2e-3)
    assert lr.lateral_offset_m == pytest.approx(ref.lateral_offset_m, abs=0.3e-3)
    assert abs(float(lr.lateral_axis_base @ U)) < 1e-6                   # 선이 심에 수직이 됐다
    assert loc.T_base_opt is not lr.T_base_opt and loc.diag.get("transformed_by") is None


# --------------------------------------------------------------------------------------
# S_align 목표 (planning.seam_align_target, 4.3 절)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("yaw,lat,pitch", [(14.8, 0.035, 0.0), (-17.4, -0.045, 0.0), (10.0, 0.02, 30.0)])
def test_seam_align_target_invariants(calib, yaw, lat, pitch):
    T = _pose(calib, yaw, lat, pitch_deg=pitch)
    loc = _loc(calib, T)
    est = _est()
    target = loc.standoff_m                                              # 거리 보정 0 → 착점이 정확히 G
    T_to, info = seam_align_target(T, calib, loc, est, target, 1.0)
    G = loc.top_centre_base
    R = (T_to @ calib.T_flange_opt)[:3, :3]
    assert info["delta_deg"] == pytest.approx(-yaw, abs=0.05)
    # 렌더한 드웰의 잰 판 법선·롤은 N·0 과 µm·µrad 수준으로 다르다 (단위 불변식의 정확한 1e-9 는 아래 3G 구성 시험)
    assert abs(math.degrees(yaw_to_seam_rad(R, N, U))) < 1e-3                        # 레이저 선 ⟂ 측정 심
    A = _axis_hit(T_to, calib)
    assert np.linalg.norm((A - G)[:2]) < 0.05e-3                                    # 착점 = G (측면 규약), 심 방향 미끄럼 없음
    o = (T_to @ calib.T_flange_opt)[:3, 3]
    assert float((o - G) @ N) == pytest.approx(target, abs=0.05e-3)                    # 판 거리 = 목표
    assert float(R[:, 2] @ N) == pytest.approx(float(loc.T_base_opt[:3, 2] @ N), abs=1e-9)        # 레이저 기울기 (Z·n) 불변
    # 1G: 회전축이 연직이라 모든 툴 점의 높이가 불변 (회전만 — 평행이동 전)
    T1 = info["T_rot"]
    for p in (np.zeros(3), np.array([0.1, 0.05, -0.2])):
        hp = lambda TT: float((TT[:3, :3] @ p + TT[:3, 3])[2])
        assert hp(T1) == pytest.approx(hp(T), abs=1e-12)
    # 부분 회전: 남은 yaw = (1 − frac)·δ
    T_h, info_h = seam_align_target(T, calib, loc, est, target, 0.5)
    Rh = (T_h @ calib.T_flange_opt)[:3, :3]
    assert abs(math.degrees(yaw_to_seam_rad(Rh, N, U))) == pytest.approx(0.5 * abs(info["delta_deg"]), abs=0.05)
    assert info_h["applied_deg"] == pytest.approx(0.5 * info["delta_deg"])
    T_0, info_0 = seam_align_target(T, calib, loc, est, target, 0.0)
    assert _rot(T_0, T) < 0.01 and info_0["applied_deg"] == 0.0                         # 회전 없음 (잰 롤 µrad 만)


def test_seam_align_target_on_a_rolled_plate_keeps_the_groove_centred(calib):
    """판이 심 둘레로 12° 롤 + 선 yaw 4.5° (가상 팔 2026-10-09 roll12: 보정 기준을 지금 광축 착점으로 두면 원점 둘레 롤이 선을 r·sin θ = 86 mm
    옆으로 옮겨 그루브가 시야 끝에 갔다): 회전 정렬 + 수직 보정 뒤 레이저 선이 판에 평행 (롤 0), 그루브가 광축 위 (측면 0), 선 ⟂ 심."""
    from scipy.spatial.transform import Rotation
    n_r = Rotation.from_rotvec(U * math.radians(12.0)).apply(N)
    tp = plate_frame([0.6, 0.0, H], U, n_r)
    scene = GrooveScene.from_spec(GrooveSpec(bevel_side="left"), yc=0.0)
    w = np.cross(n_r, U)

    def at(T):
        x, z = static_profile(T, calib, scene, tp, CFG)
        return localize_static(T, calib, x, z, GrooveSpec(bevel_side="auto"), travel_dir_hint=U, plate_normal_hint=N)
    R = rot_z(math.radians(4.5)) @ R0
    o = np.array([0.6, 0.0, H]) + w * 0.02 - 0.443 * R[:, 2]
    T = make_transform(R, o) @ np.linalg.inv(calib.T_flange_opt)
    loc = at(T)
    assert abs(loc.roll_signed_deg) > 10.0
    T2, _info = seam_align_target(T, calib, loc, _est(u=U, n=n_r, w=w), loc.standoff_m, 1.0)
    loc2 = at(T2)
    assert abs(loc2.roll_signed_deg) < 0.2 and abs(loc2.lateral_offset_m) < 0.3e-3, (loc2.roll_signed_deg, loc2.lateral_offset_m)
    assert abs(math.degrees(yaw_to_seam_rad((T2 @ calib.T_flange_opt)[:3, :3], n_r, U))) < 0.5


def _rot(Ta, Tb) -> float:
    R = Ta[:3, :3].T @ Tb[:3, :3]
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def test_seam_align_target_3g_keeps_plate_distance_and_tilt_not_height(calib):
    """3G (판 법선 수평): 법선 둘레 회전은 판 거리·레이저 기울기 (Z·n) 를 지키지만 높이는 바뀐다 (사전검사가 포락선 표본으로 본다)."""
    n3 = np.array([0.0, -1.0, 0.0])
    u3 = np.array([0.0, 0.0, 1.0])
    w3 = np.cross(n3, u3)
    psi = math.radians(12.0)
    X = math.cos(psi) * w3 + math.sin(psi) * u3
    tilt = math.radians(35.0)
    Z = unit(-n3 * math.cos(tilt) - u3 * math.sin(tilt))                 # 내려다본다
    X = unit(X - Z * float(X @ Z))
    R = np.column_stack([X, np.cross(Z, X), Z])
    G = np.array([0.5, 0.70, 0.45])
    o = G - Z * 0.443
    T = make_transform(R, o) @ np.linalg.inv(calib.T_flange_opt)
    loc = StaticLocalization(features=None, T_base_opt=T @ calib.T_flange_opt, opt_origin_base=o, groove_found=True)
    loc.plate_normal_base = n3
    loc.top_centre_base = G.copy()
    loc.plate_points_base = np.array([G + w3 * d for d in (-0.05, 0.05)])
    loc.axis_hit_base = G.copy()
    loc.lateral_axis_base = unit(X - n3 * float(X @ n3))
    loc.standoff_m = float((o - G) @ n3)
    loc.roll_signed_deg = 0.0
    loc.lateral_offset_m = 0.0
    T_to, info = seam_align_target(T, calib, loc, _est(u=u3, n=n3, c=G), loc.standoff_m, 1.0)
    To = T_to @ calib.T_flange_opt
    assert abs(yaw_to_seam_rad(To[:3, :3], n3, u3)) < 1e-9
    assert float((To[:3, 3] - G) @ n3) == pytest.approx(loc.standoff_m, abs=1e-9)
    t = float((G - To[:3, 3]) @ n3) / float(To[:3, 2] @ n3)
    np.testing.assert_allclose(To[:3, 3] + To[:3, 2] * t, G, atol=1e-9)              # 착점 = G
    assert float(To[:3, 2] @ n3) == pytest.approx(float(Z @ n3), abs=1e-9)
    assert abs(float(To[:3, 3] @ u3) - float(o @ u3)) > 1e-3                         # 높이는 바뀐다


# --------------------------------------------------------------------------------------
# 장면 판단 활성 (4.7 절) — 그 드웰의 선 발자국이 측정으로 확인된 시편 구간 안일 때만 'judge'
# --------------------------------------------------------------------------------------
def _dwell_loc(calib, yaw, s, lateral=0.0):
    T = _pose(calib, yaw, lateral, s=s)
    loc = _loc(calib, T)
    loc.diag["raw_z_axis_mm"] = 443.0
    return loc


def test_scene_activation_rule(lab_at, calib):
    at = lab_at
    at.calib = calib
    r = 0.443
    est = _est(c=np.array([0.6, 0.0, H]), s_first=-0.050, s_last=0.0)              # 추적이 −50..0 mm 를 일관되게 읽었다
    # S_trk: 지금 yaw 15° 의 발자국 (93·sin15 + e_hit ≈ 27 mm) 이 s_first 뒤로 넘어가면 기록만, 안이면 판단
    f15 = at._footprint_m(r, math.radians(15.0), 0.0)
    mode = at._scene_trk(est, "ok")
    assert mode(_dwell_loc(calib, 15.0, s=0.0)) == "judge"                           # 0 − 27 ≥ −50
    assert mode(_dwell_loc(calib, 15.0, s=-0.050 + 0.5 * f15)) == "record"
    assert at._scene_trk(None, "late") == "record" and at._scene_trk(None, "groove_mismatch") == "judge"
    # S_al / S 사이트: ψ_res 발자국
    psi_res = at._psi_res_rad(est)
    ms = at._scene_site(est, psi_res)
    assert ms(_dwell_loc(calib, 0.0, s=0.0)) == "judge"
    assert ms(_dwell_loc(calib, 0.0, s=-0.049)) == "record"
    # E: 트랜싯이 일관되게 읽은 단면 (s ≤ s_last_T) 이면 판단, 그 너머면 기록만; 트랜싯 맞춤이 없으면 예전대로 판단
    est_T = _est(c=np.array([0.6, 0.0, H]), s_first=-0.3, s_last=0.0, source="transit")
    me = at._scene_E(est_T)
    assert me(_dwell_loc(calib, 0.0, s=-0.002)) == "judge" and me(_dwell_loc(calib, 0.0, s=0.004)) == "record"
    assert at._scene_E(None) == "judge"


# --------------------------------------------------------------------------------------
# X 축 프로브 판정 · w 방향 스케일 읽기
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dx,sgn,expect", [(-6.3, 1.0, "confirmed"), (6.4, -1.0, "confirmed"), (6.3, 1.0, "flip"), (-6.1, -1.0, "flip"),
                                          (0.2, 1.0, "inconsistent"), (float("nan"), 1.0, "inconsistent"), (-12.0, 1.0, "inconsistent")])
def test_probe_verdict_x(dx, sgn, expect):
    v, note = probe_verdict_x(dx, 6.3, sgn)
    assert v == expect, note


def test_site_pre_laterals_along_the_seam_normal():
    """선이 심에 ψ 기운 드웰에서 w 로 d 만큼 보정해 수렴: e 방향 읽기 (예전) 는 센서 lat 대 운동학 d·cos ψ 로 cos²ψ 만큼 어긋나고,
    w 방향 읽기는 같다 (스케일 판정이 거짓으로 어긋나지 않는다)."""
    psi = math.radians(10.0)
    w = np.array([0.0, 1.0, 0.0])
    e = np.array([math.sin(psi), math.cos(psi), 0.0])                    # 선 방향 (w 에서 ψ)
    lat = 0.030                                                          # 선 위 착점 → 그루브 30 mm
    d = lat * math.cos(psi)                                              # w 로 옮길 양 (착점이 심에)

    def L(A, lat_off):
        loc = StaticLocalization(features=None, T_base_opt=np.eye(4), opt_origin_base=A + np.array([0, 0, 0.44]))
        loc.lateral_axis_base = e
        loc.axis_hit_base = np.asarray(A, dtype=float)
        loc.lateral_offset_m = lat_off
        return loc
    A0 = np.array([0.6, 0.0, 0.05])
    arr = L(A0, lat)
    fin = L(A0 + w * d, 0.0)
    s_old, k_old = site_pre_laterals(arr, arr, fin)
    s_new, k_new = site_pre_laterals(arr, arr, fin, w=w)
    assert s_old == pytest.approx(lat) and k_old == pytest.approx(lat * math.cos(psi) ** 2)
    assert s_new == pytest.approx(k_new) == pytest.approx(d)


def test_record_only_dwell_raises_the_safety_floor(lab_at, calib):
    """D9: 장면을 기록만 하는 드웰 (첫 드웰 S 등 — 선이 시편 위라는 측정 전) 도 안전 바닥 (1G 판 상면, 올리기만) 을 갱신한다 — 그 뒤의 S 추적·
    probe X·S 정렬이 실측 바닥으로 포락선 검사를 받게. 장면 판단 (기록만) 과는 따로."""
    at = lab_at
    saved = (at.plate_top, at.job_ref.plate_top_measured_m)
    try:
        at.plate_top = None
        at.job_ref.plate_top_measured_m = None
        T = _pose(calib, 5.0, 0.0)
        scene = GrooveScene.from_spec(GrooveSpec(bevel_side="left"), yc=0.0)
        x, z = static_profile(T, calib, scene, T_PLATE, CFG)
        n0 = len(at.res.decisions)
        loc = at._localize(x, z, T, "S", scene="record")
        assert loc.diag["scene_mode"] == "record" and loc.plate_ok
        assert at.plate_top == pytest.approx(H, abs=0.5e-3) and at.job_ref.plate_top_measured_m == at.plate_top
        assert any(d["step"] == "floor" for d in at.res.decisions[n0:])
        # 올리기만: 더 낮게 읽은 기록 드웰은 바닥을 내리지 않는다
        at.plate_top = at.job_ref.plate_top_measured_m = H + 0.002
        at._localize(x, z, T, "S_again", scene="record")
        assert at.plate_top == pytest.approx(H + 0.002)
    finally:
        at.plate_top, at.job_ref.plate_top_measured_m = saved


def test_the_torch_keypoint_comes_from_the_robot_model_when_the_platform_gives_it(tmp_path):
    """MoveIt adapter tool_geometry: with the weld torch on the arm, the platform calib carries the torch's TCP frame in the eoat frame
    (TF flange <- TCP . tip axes) and its nozzle end (the URDF collision geometry); the safety policy then puts the wire end = TCP +
    (nozzle end + the job's process stick-out weld_plan.stick_out_mm) along the torch axis into the job's torch keypoint
    (torch_lateral_m = its eoat x / y, torch_protrusion_m = its eoat z beyond the calibrated optical origin) instead of the job / config
    values - one torch point for the envelope, the weld plan's aim and D14. Without it the job and config values stay."""
    settings = C.load_settings(LAB_CONFIG)
    job0, doc = C.load_job(os.path.join(JOBS, "groove_10T_1G.yaml"))
    lim = (settings.params.get("moveit_jtc") or {}).get("limits") or {}

    def run(calib):
        at = AutoTeach(settings, job0, doc, calib, out_dir=str(tmp_path / f"at{id(calib)}"), log=lambda s: None,
                       session=ScanSession(robot=None, sensor=None, owned=False))
        at.cfg = SimpleNamespace(v_eff_max_mps=float(lim["max_tcp_speed_mps"]))
        return at, at._apply_safety_policy(at.job0)

    c0 = C.attach_tool_body(C.load_calib(CALIB), settings)
    _at, j0 = run(c0)
    assert tuple(j0.torch_lateral_m) == tuple(job0.torch_lateral_m)
    assert j0.torch_protrusion_m == pytest.approx(float(settings.safety["torch_protrusion_m"]))
    c1 = C.attach_tool_body(C.load_calib(CALIB), settings)
    a = np.array([0.5, 0.0, math.sqrt(3.0) / 2.0])                            # e.g. our torch: nozzle axis 30 deg off z_eoat
    T = np.eye(4)
    T[:3, 2], T[:3, 1] = a, [0.0, -1.0, 0.0]
    T[:3, 0] = np.cross(T[:3, 1], a)
    T[:3, 3] = [-0.04655, -0.00045, 0.29717]                                    # the TCP (contact-tip end) in the eoat frame
    c1.torch_tcp_eoat, c1.torch_nozzle_end_mm = T, 1.2
    c1.torch_tip_source = "test: the weld torch from TF"
    at1, j1 = run(c1)
    stick = float(doc["weld_plan"]["stick_out_mm"])                            # the job's process stick-out (from the nozzle end)
    tip = T[:3, 3] + a * (1.2 + stick) * 1e-3                                   # the wire end
    assert np.allclose(j1.torch_lateral_m, tip[:2])
    assert j1.torch_protrusion_m == pytest.approx(tip[2] - float(c1.p_opt_eoat[2]))
    from scan_core.path_gen import tool_keypoints
    kp = tool_keypoints(np.eye(4), c1, j1)                                        # the envelope's tip keypoint = that point
    assert np.allclose(np.linalg.inv(c1.T_flange_eoat) @ np.append(kp["tip"], 1.0), np.append(tip, 1.0))
    msg = (f"torch keypoint from the robot model: eoat ({tip[0] * 1e3:+.2f}, {tip[1] * 1e3:+.2f}, {tip[2] * 1e3:+.2f}) mm")
    assert any(d["step"] == "safety" and d["msg"].startswith(msg) and f"(nozzle end 1.20 + stick-out {stick:g}) mm" in d["msg"]
               for d in at1.res.decisions)
