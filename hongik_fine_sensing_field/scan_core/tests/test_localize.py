"""localize: 기울기 인지(I2) 정적 국소화, 측 무관 그루브 중심, 상대 보정 수렴(I3)·캡, S/E 쌍 → 심 방향, 패스 정련, 충실도 게이트."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from scan_core.frames import angle_between, inv_transform, make_transform, rot_x, rot_y, rot_z, transform_points, unit
from scan_core.localize import (PassFit, Verdict, apply_correction, fidelity_gate, height_m,
                                localize_static, median_profile, plate_distance_m, plate_points_from_scan,
                                polyline_from_scan_features, ransac_line_3d, refine_from_pass, relative_correction,
                                seam_from_pair, _rot_axis)
from scan_core.path_gen import plate_frame
from scan_core.reconstruct import reconstruct_scan
from scan_core.seam_features import extract_cross_section_features, extract_seam_features_along_scan
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, PlaneScene, SynthConfig, simulate_scan, static_profile
from scan_core.types import Calib, ScanJob

R0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)       # X_opt = base +y (라인이 그루브를 가로지름), Z_opt = −z
H_PLATE = 0.05                                                         # 판 상면 base z
T_PLATE = plate_frame([0.6, 0.0, H_PLATE], [1, 0, 0], [0, 0, 1])
CFG = SynthConfig(n_points=1408, seed=1, sigma_z_mm=0.02, dropout_prob=0.01)
SPEC_AUTO = GrooveSpec(bevel_side="auto")


def _pose(calib: Calib, R: np.ndarray, o) -> np.ndarray:
    """광학 자세 (R, o) → 플랜지 자세."""
    return make_transform(R, o) @ inv_transform(calib.T_flange_opt)


def _deg(a, b) -> float:
    return math.degrees(angle_between(a, b))


# --------------------------------------------------------------------------------------
# 1. 시간 중앙값
# --------------------------------------------------------------------------------------
def test_median_profile_ignores_nan_and_zero_sentinel():
    rng = np.random.default_rng(0)
    x = np.linspace(-50, 50, 200)
    recs = []
    for k in range(9):
        z = 400.0 + 0.1 * x + rng.normal(0, 0.02, x.size)
        z[10] = np.nan if k % 2 else 0.0             # 열 10: 절반 NaN / 절반 센티널 → 전부 무효
        z[20] = np.nan if k < 3 else z[20]           # 열 20: 3개만 NaN → 유효
        recs.append(SimpleNamespace(x=x.copy(), z=z))
    xm, zm, n_ok = median_profile(recs)
    assert xm.shape == (200,) and n_ok == 199 and np.isnan(zm[10]) and np.isfinite(zm[20])
    assert np.abs(zm[np.isfinite(zm)] - (400.0 + 0.1 * xm[np.isfinite(zm)])).max() < 0.05
    # ProfileFrame 스타일 (.x_mm/.z_mm)
    xm2, zm2, n2 = median_profile([SimpleNamespace(x_mm=x, z_mm=400.0 + 0 * x)])
    assert n2 == 200 and np.allclose(zm2, 400.0)
    with pytest.raises(ValueError):
        median_profile([])


# --------------------------------------------------------------------------------------
# 2. 정적 국소화 — (a) 기울기 인지 높이/거리 (I2 증명)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("roll", [8.0, -8.0])
def test_localize_flat_plate_tilt_aware_height_and_standoff(calib_lab, roll):
    """센서가 중력 대비 롤 ±8°, 피치 10°: 판 높이·수직 standoff 는 정확, 원시 z 는 > 5 mm 다르다."""
    pitch = 10.0
    R = rot_y(math.radians(pitch)) @ rot_x(math.radians(roll)) @ R0
    true_standoff = 0.34
    T_bf = _pose(calib_lab, R, (0.6, 0.0, H_PLATE + true_standoff))
    x, z = static_profile(T_bf, calib_lab, PlaneScene(h0=0.0), T_PLATE, CFG)
    loc = localize_static(T_bf, calib_lab, x, z, SPEC_AUTO, travel_dir_hint=(1, 0, 0))
    assert loc.plate_ok and not loc.groove_found and "no_groove" in loc.flags
    assert abs(loc.plate_top_height_m - H_PLATE) < 0.2e-3, loc.summary()          # I1 바닥 = 측정 판 상면
    assert abs(loc.standoff_m - true_standoff) < 0.3e-3, loc.summary()             # I2: n 방향 거리
    assert loc.diag["raw_z_axis_mm"] - true_standoff * 1e3 > 5.0                    # 원시 z 는 range → 기울면 커진다
    assert abs(loc.diag["raw_z_axis_mm"] - true_standoff * 1e3 / (math.cos(math.radians(roll)) * math.cos(math.radians(pitch)))) < 0.5
    assert _deg(loc.plate_normal_base, [0, 0, 1]) < 0.05
    assert abs(loc.roll_deg - roll) < 0.1 and abs(loc.roll_signed_deg + roll) < 0.1
    assert abs(loc.incidence_deg - math.degrees(math.acos(math.cos(math.radians(roll)) * math.cos(math.radians(pitch))))) < 0.1
    # 판 점의 높이는 전부 판 상면 (up·P, I2); plate_distance_m 도 0
    assert np.abs(height_m(loc.plate_points_base) - H_PLATE).max() < 0.2e-3
    assert np.abs(plate_distance_m(loc.plate_points_base, loc.plate_points_base[0], loc.plate_normal_base)).max() < 0.2e-3
    assert loc.plate_height_spread_m < 0.3e-3
    # 힌트: plate_normal_hint(up) 도 같은 결과
    loc2 = localize_static(T_bf, calib_lab, x, z, SPEC_AUTO, plate_normal_hint=(0, 0, 1))
    assert abs(loc2.standoff_m - true_standoff) < 0.3e-3 and loc2.diag["normal_source"] == "plate_normal_hint"
    d = loc.to_dict()
    assert d["standoff_m"] == pytest.approx(loc.standoff_m) and "features" in d and isinstance(d["flags"], list)


def test_plate_top_floor_is_conservative_on_a_plate_tilted_vs_gravity(calib_lab):
    """Minor-1 (I1): the plate is rolled 2 deg about the travel axis, so its height varies by ~+-2.4 mm across the FOV.
    The floor plate_top_height_m is the highest plate-inlier height (the extractor's inliers are already inside its band; 4d P5 —
    formerly the 95th percentile), the median is kept separately for reporting, and the floor is never below the median."""
    tilt = 2.0
    n_p = unit([0.0, -math.sin(math.radians(tilt)), math.cos(math.radians(tilt))])
    T_plate = plate_frame([0.6, 0.0, H_PLATE], [1, 0, 0], n_p)
    T_bf = _pose(calib_lab, R0, (0.6, 0.0, H_PLATE + 0.34))
    x, z = static_profile(T_bf, calib_lab, PlaneScene(h0=0.0), T_plate, CFG)
    loc = localize_static(T_bf, calib_lab, x, z, SPEC_AUTO, travel_dir_hint=(1, 0, 0))
    assert loc.plate_ok and _deg(loc.plate_normal_base, n_p) < 0.05
    hP = height_m(loc.plate_points_base)
    assert loc.plate_height_spread_m > 3e-3                                      # the line really sees a tilted plate
    assert loc.plate_top_height_median_m == pytest.approx(float(np.median(hP)), abs=1e-9)
    assert loc.plate_top_height_m == pytest.approx(float(hP.max()), abs=1e-9)
    assert loc.plate_top_height_m - loc.plate_top_height_median_m > 1.5e-3       # conservative: well above the centre height
    assert loc.plate_top_height_m <= hP.max() + 1e-9 and loc.plate_top_height_m >= loc.plate_top_height_median_m
    assert abs(loc.plate_top_height_median_m - H_PLATE) < 0.3e-3                 # line centre is over the plate origin
    d = loc.to_dict()
    assert d["plate_top_height_median_m"] == pytest.approx(loc.plate_top_height_median_m)
    assert "plate_top_quantile" not in loc.diag and "median" in loc.summary()
    # flat plate: quantile and median coincide within the noise (0.02 mm sigma)
    xf, zf = static_profile(T_bf, calib_lab, PlaneScene(h0=0.0), T_PLATE, CFG)
    locf = localize_static(T_bf, calib_lab, xf, zf, SPEC_AUTO, travel_dir_hint=(1, 0, 0))
    assert abs(locf.plate_top_height_m - locf.plate_top_height_median_m) < 0.1e-3
    assert locf.plate_top_height_m >= locf.plate_top_height_median_m


def test_localize_without_hint_documents_pitch_limitation(calib_lab):
    """힌트 없음: n = −Z_opt 의 판 직선 직교 성분 → 피치 10° 가 그대로 n 오차가 되어 standoff 가 (1/cos−1)·S 만큼 틀린다 (플래그)."""
    R = rot_y(math.radians(10.0)) @ R0
    T_bf = _pose(calib_lab, R, (0.6, 0.0, H_PLATE + 0.34))
    x, z = static_profile(T_bf, calib_lab, PlaneScene(h0=0.0), T_PLATE, CFG)
    loc = localize_static(T_bf, calib_lab, x, z, None)
    assert "normal_from_optical_axis_unverified_pitch" in loc.flags and loc.diag["normal_source"] == "optical_axis"
    assert abs(_deg(loc.plate_normal_base, [0, 0, 1]) - 10.0) < 0.2
    assert abs(loc.standoff_m - 0.34 / math.cos(math.radians(10.0))) < 0.5e-3
    # 판 높이는 힌트와 무관하게 정확 (base 점의 up 성분)
    assert abs(loc.plate_top_height_m - H_PLATE) < 0.2e-3


def test_localize_no_plate_and_bad_input(calib_lab):
    T_bf = _pose(calib_lab, R0, (0.6, 0.0, 0.4))
    loc = localize_static(T_bf, calib_lab, np.full(1408, np.nan), np.full(1408, np.nan), SPEC_AUTO)
    assert not loc.plate_ok and "no_plate" in loc.flags and loc.valid_ratio == 0.0
    corr = relative_correction(loc, target_standoff_m=0.34)
    assert corr.is_zero and "no plate" in corr.reason
    # 센서 유효 창 밖(z < 233 mm) 은 원시 z 로만 걸러진다
    x = np.linspace(-60, 60, 1408)
    loc2 = localize_static(T_bf, calib_lab, x, np.full(1408, 200.0), SPEC_AUTO)
    assert not loc2.plate_ok and loc2.valid_ratio == 0.0


# --------------------------------------------------------------------------------------
# (b) 그루브: yaw 0/5/15°, 베벨 left/right (측 무관)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("yaw", [0.0, 5.0, 15.0])
def test_localize_groove_centres_side_agnostic(calib_lab, side, yaw):
    yc = 0.006
    spec_true = GrooveSpec(bevel_side=side)
    scene = GrooveScene.from_spec(spec_true, yc=yc, angle_deg=yaw)
    R = rot_x(math.radians(3.0)) @ R0                                  # 롤 3°
    T_bf = _pose(calib_lab, R, (0.6, 0.0, H_PLATE + 0.40))
    x, z = static_profile(T_bf, calib_lab, scene, T_PLATE, CFG)
    loc = localize_static(T_bf, calib_lab, x, z, SPEC_AUTO, travel_dir_hint=(1, 0, 0))
    assert loc.groove_found and loc.plate_ok, loc.summary()
    run = spec_true.bevel_run_mm * 1e-3
    top_true = np.array([0.6, yc - run / 2 if side == "left" else yc + run / 2, H_PLATE])
    root_true = np.array([0.6, yc, H_PLATE - spec_true.thickness_mm * 1e-3])
    assert np.abs(loc.top_centre_base - top_true).max() < 0.3e-3, (loc.top_centre_base - top_true)
    assert np.abs(loc.root_centre_base - root_true).max() < 0.3e-3, (loc.root_centre_base - root_true)
    assert abs(loc.standoff_m - 0.40) < 0.3e-3 and abs(loc.plate_top_height_m - H_PLATE) < 0.2e-3
    # 측면 오프셋 = 상부 중심 − 광축∩판 (라인 방향); rot_x(+3°) 는 Z_opt(−z) 를 +y 쪽으로 기울여 광축이 +y 로 400·tan3° 이동
    axis_y = 0.40 * math.tan(math.radians(3.0))
    assert abs(loc.lateral_offset_m - (top_true[1] - axis_y)) < 0.3e-3
    # 측 무관: 추출기는 쪽을 판정하되 spec 비교는 None
    assert loc.features.bevel_side == side and loc.features.spec_deltas["bevel_side_match"] is None
    assert abs(loc.diag["root_depth_along_n_mm"] - 10.0) < 0.3


# --------------------------------------------------------------------------------------
# (c) 상대 보정: 핸드아이 2 mm / 1° 오차에서 ≤ 2 스텝 수렴, 캡 준수
# --------------------------------------------------------------------------------------
def _perturbed_calib(calib: Calib, t_mm: float = 2.0, rot_deg: float = 1.0) -> Calib:
    ax = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    t = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0) * t_mm * 1e-3
    return Calib(T_flange_opt=calib.T_flange_opt @ make_transform(_rot_axis(ax, math.radians(rot_deg)), t))


def test_relative_correction_converges_in_two_steps_with_perturbed_calib(calib_lab):
    calib_true = _perturbed_calib(calib_lab)
    scene = GrooveScene.from_spec(GrooveSpec(bevel_side="left"), yc=0.012)
    T_bf = _pose(calib_true, rot_x(math.radians(4.0)) @ R0, (0.6, 0.0, H_PLATE + 0.38))   # 롤 4°, standoff 380, 측면 12 mm
    target = 0.34
    hist = []
    for k in range(3):
        x, z = static_profile(T_bf, calib_true, scene, T_PLATE, CFG)
        loc = localize_static(T_bf, calib_lab, x, z, SPEC_AUTO, travel_dir_hint=(1, 0, 0))     # 공칭 캘리브로 국소화
        o_true = (T_bf @ calib_true.T_flange_opt)[:3, 3]
        hist.append((k, loc.diag["x_c_mm"], loc.roll_deg, loc.standoff_m, o_true[2] - H_PLATE))
        if abs(loc.diag["x_c_mm"]) <= 1.0 and abs(loc.roll_deg) <= 0.5 and abs(loc.standoff_m - target) <= 2e-3:
            break
        corr = relative_correction(loc, target_standoff_m=target)                  # 기본: 잰 만큼 그대로 (상한 없음)
        assert not corr.clipped
        T_bf_new = apply_correction(T_bf, calib_lab, corr)
        # I3: 델타는 현재 자세 기준 — 광학 원점(공칭) 이동 = d_base
        o_nom_prev = (T_bf @ calib_lab.T_flange_opt)[:3, 3]
        o_nom_new = (T_bf_new @ calib_lab.T_flange_opt)[:3, 3]
        assert np.allclose(o_nom_new - o_nom_prev, corr.d_base, atol=1e-9)
        T_bf = T_bf_new
    assert k <= 2, hist                                                  # 2 스텝 이내 수렴
    assert hist[0][2] == pytest.approx(4.0, abs=0.1) and hist[0][3] == pytest.approx(0.38, abs=0.5e-3)
    assert abs(hist[-1][4] - target) <= 2e-3                             # 진짜 수직 거리도 목표 근처
    # 첫 스텝은 잰 만큼: 판 쪽 40 mm (0.38 → 0.34), 롤 −4° — 상한에 걸리지 않는다
    x, z = static_profile(_pose(calib_true, rot_x(math.radians(4.0)) @ R0, (0.6, 0.0, H_PLATE + 0.38)), calib_true, scene, T_PLATE, CFG)
    loc0 = localize_static(_pose(calib_true, rot_x(math.radians(4.0)) @ R0, (0.6, 0.0, H_PLATE + 0.38)), calib_lab, x, z, SPEC_AUTO,
                           travel_dir_hint=(1, 0, 0))
    c0 = relative_correction(loc0, target_standoff_m=target)
    assert not c0.clipped and c0.d_along_n_m == pytest.approx(-0.04, abs=1e-3)
    assert abs(c0.roll_deg + 4.0) < 0.1 and c0.aim == "top_centre" and np.allclose(unit(c0.roll_axis_base), loc0.T_base_opt[:3, 1])
    assert c0.to_dict()["clipped"] == c0.clipped


def test_relative_correction_caps_and_evidence_gate(calib_lab):
    # 큰 측면 오프셋(60 mm) + 큰 롤(10°): 상한을 주면 (면내 40 mm, 회전 7° 및 스윙, 판 쪽 30 mm, 멀어짐 50 mm) 줄인다; 기본은 잰 만큼
    scene = GrooveScene.from_spec(GrooveSpec(bevel_side="right"), yc=0.060)
    T_bf = _pose(calib_lab, rot_x(math.radians(10.0)) @ R0, (0.6, 0.0, H_PLATE + 0.40))
    x, z = static_profile(T_bf, calib_lab, scene, T_PLATE, CFG)
    loc = localize_static(T_bf, calib_lab, x, z, SPEC_AUTO, travel_dir_hint=(1, 0, 0))
    assert loc.groove_found
    caps = {"in_plane_m": 0.04, "along_n_toward_plate_m": 0.03, "along_n_away_m": 0.05, "rotation_deg": 7.0}
    full = relative_correction(loc, target_standoff_m=0.34)
    assert not full.clipped and full.d_in_plane_m > 0.05 and abs(full.roll_deg) > 9.0
    c = relative_correction(loc, target_standoff_m=0.34, caps=caps)
    assert "in_plane" in c.clipped and c.d_in_plane_m == pytest.approx(0.04, abs=1e-9)
    assert "rotation" in c.clipped and abs(c.roll_deg) <= 7.0 and c.swing_m <= 0.04 + 1e-9
    assert c.d_along_n_m == pytest.approx(-0.03) and "along_n_toward_plate" in c.clipped
    # 면내 이동은 라인 방향(e_lat) 성분만 — 진행 방향 성분은 버린다 (프로파일로 관측 불가)
    t_dir = np.cross(loc.plate_normal_base, loc.lateral_axis_base)
    assert abs(float(c.d_base @ t_dir)) < 1e-9
    # 사용자 캡
    c2 = relative_correction(loc, target_standoff_m=0.34, caps={"in_plane_m": 0.01, "rotation_deg": 2.0, "along_n_toward_plate_m": 0.005})
    assert c2.d_in_plane_m == pytest.approx(0.01) and abs(c2.roll_deg) <= 2.0 and c2.d_along_n_m == pytest.approx(-0.005)
    # 증거 게이트: standoff < target 이면 판 쪽 이동 없음(멀어지기는 허용, away 캡)
    c3 = relative_correction(loc, target_standoff_m=0.50, caps=caps)
    assert c3.d_along_n_m == pytest.approx(0.05) and "along_n_away" in c3.clipped
    # 그루브 없음(평판): 판 쪽 이동 금지, 면내는 롤 스윙만
    xp, zp = static_profile(T_bf, calib_lab, PlaneScene(h0=0.0), T_PLATE, CFG)
    locp = localize_static(T_bf, calib_lab, xp, zp, SPEC_AUTO, travel_dir_hint=(1, 0, 0))
    c4 = relative_correction(locp, target_standoff_m=0.34, caps=caps)
    assert not locp.groove_found and c4.aim == "axis_hit" and c4.d_along_n_m == 0.0 and "along_n_evidence_gate" in c4.clipped
    assert abs(c4.roll_deg + 10.0) < 0.5 or "rotation" in c4.clipped


# --------------------------------------------------------------------------------------
# (d) S/E 쌍 → u, n, psi, theta
# --------------------------------------------------------------------------------------
def test_seam_from_pair_recovers_yaw_pitch_direction_normal(calib_lab):
    tilt = 1.0                                                   # 판이 base y 둘레로 1° 기울어 S→E 로 내려감
    n_p = unit([math.sin(math.radians(tilt)), 0.0, math.cos(math.radians(tilt))])
    u_p = unit([math.cos(math.radians(tilt)), 0.0, -math.sin(math.radians(tilt))])
    T_plate = plate_frame([0.6, 0.0, H_PLATE], u_p, n_p)
    yaw = 3.0
    scene = GrooveScene.from_spec(GrooveSpec(bevel_side="left", weld_length_mm=1000.0), yc=0.0, angle_deg=yaw)   # 판 x ∈ ±0.5
    L = 0.38
    locs = []
    for sx in (0.0, L):
        T_bf = _pose(calib_lab, R0, (0.6 + sx, 0.0, H_PLATE + 0.40))    # 같은 자세, 수평 진행 (base +x)
        x, z = static_profile(T_bf, calib_lab, scene, T_plate, CFG)
        loc = localize_static(T_bf, calib_lab, x, z, SPEC_AUTO, travel_dir_hint=(1, 0, 0))
        assert loc.groove_found, loc.summary()
        locs.append(loc)
    pr = seam_from_pair(locs[0], locs[1])
    assert pr["ok"], pr["notes"]
    u_true = T_plate[:3, :3] @ np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
    assert _deg(pr["u"], u_true) < 0.1
    assert _deg(pr["n"], n_p) < 0.1
    assert abs(pr["psi_deg"] - yaw) < 0.1, pr                    # + = S→E 로 +X_opt 쪽으로 흐름
    assert abs(pr["theta_deg"] + tilt) < 0.1, pr                 # 판이 내려감 → −
    assert abs(pr["theta_standoff_deg"] - pr["theta_deg"]) < 0.1  # 수평 진행: 높이 차 = standoff 차
    assert abs(pr["L_m"] - L) < 0.03
    # 판 높이 차이도 I2 (base 점) 로: L·tan(1°)
    assert abs((locs[0].plate_top_height_m - locs[1].plate_top_height_m) - L * math.tan(math.radians(tilt))) < 0.3e-3
    assert pr["travel_horizontal"] is True and abs(pr["theta_height_deg"] - pr["theta_deg"]) < 1e-9
    assert pr["notes"] == []
    # 실패 경로
    bad = seam_from_pair(locs[0], locs[0])
    assert not bad["ok"] and any("coincide" in n for n in bad["notes"])


def test_seam_from_pair_vertical_travel_uses_standoff_theta(calib_lab):
    """C10 (3G): vertical plate n = (0,-1,0), seam along u = (0,0,1); the same optical pose at two heights. The plate-height
    difference equals the travel (theta_height = +45 deg, meaningless), so theta_deg must come from the standoff difference
    (~0) and no 'travel not horizontal?' note may be emitted. gravity_axis_base is the job's."""
    T_plate = plate_frame([0.6, 0.7, 0.3], [0, 0, 1], [0, -1, 0])
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])       # X_opt = +x (across the seam), Z_opt = +y (toward the plate)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12
    scene = GrooveScene.from_spec(GrooveSpec(bevel_side="left", weld_length_mm=1000.0), yc=0.0)
    L = 0.38
    locs = []
    for sz in (0.0, L):
        T_bf = _pose(calib_lab, R, (0.6, 0.7 - 0.40, 0.3 + sz))
        x, z = static_profile(T_bf, calib_lab, scene, T_plate, CFG)
        loc = localize_static(T_bf, calib_lab, x, z, SPEC_AUTO, travel_dir_hint=(0, 0, 1))
        assert loc.groove_found and loc.plate_ok, loc.summary()
        assert _deg(loc.plate_normal_base, [0, -1, 0]) < 0.1 and abs(loc.standoff_m - 0.40) < 0.3e-3
        locs.append(loc)
    pr = seam_from_pair(locs[0], locs[1], gravity_axis_base=(0.0, 0.0, -1.0))
    assert pr["ok"], pr["notes"]
    assert _deg(pr["u"], [0, 0, 1]) < 0.1 and _deg(pr["n"], [0, -1, 0]) < 0.1
    assert pr["travel_horizontal"] is False
    assert abs(pr["theta_height_deg"] - 45.0) < 0.5                           # diagnostic only: heights differ by L
    assert abs(pr["theta_deg"]) < 0.2, pr                                      # from the standoff difference (C10)
    assert pr["theta_deg"] == pr["theta_standoff_deg"]
    assert pr["notes"] == [], pr["notes"]
    assert abs(pr["psi_deg"]) < 0.2
    # a different gravity axis makes the same travel 'horizontal' again -> height theta is used
    pr2 = seam_from_pair(locs[0], locs[1], gravity_axis_base=(0.0, 1.0, 0.0))
    assert pr2["travel_horizontal"] is True and pr2["theta_deg"] == pr2["theta_height_deg"]


# --------------------------------------------------------------------------------------
# (e) 패스 정련
# --------------------------------------------------------------------------------------
def test_refine_from_pass_synthetic_line_plane_with_outliers():
    rng = np.random.default_rng(3)
    u_true = unit([1.0, 0.05, 0.01])
    n_true = unit(np.array([0, 0, 1.0]) - float(np.array([0, 0, 1.0]) @ u_true) * u_true)
    e = np.cross(n_true, u_true)
    c_true = np.array([0.6, 0.1, 0.04])
    M = 1000
    s = np.linspace(-0.02, 0.40, M)
    P = c_true + np.outer(s, u_true) + rng.normal(0, 0.05e-3, (M, 3))
    gf = (s >= 0.0) & (s <= 0.38)
    P[~gf] = np.nan
    out = rng.random(M) < 0.05
    P[out & gf] += rng.uniform(-5e-3, 5e-3, (int((out & gf).sum()), 3))
    K = 5000
    Q = (c_true + n_true * 0.010 + np.outer(rng.uniform(-0.02, 0.40, K), u_true) + np.outer(rng.uniform(-0.08, 0.08, K), e)
         + rng.normal(0, 0.05e-3, (K, 3)))
    outq = rng.random(K) < 0.05
    Q[outq] += np.outer(rng.uniform(-3e-3, 3e-3, int(outq.sum())), n_true)
    feats = {"root_gap_mm": np.full(M, 5.0), "depth_mm": np.full(M, 10.0), "top_opening_mm": np.full(M, 10.77),
             "bevel_angle_deg": np.full(M, 30.0), "hi_lo_mm": np.full(M, 0.1)}
    u_prior = _rot_axis(n_true, math.radians(5.0)) @ u_true
    n_prior = _rot_axis(u_true, math.radians(3.0)) @ n_true
    fit = refine_from_pass(P, Q, gf, feats, GrooveSpec(bevel_side="auto"), u_prior=u_prior, n_prior=n_prior,
                           ransac_line_thr_m=0.5e-3, ransac_plane_thr_m=0.5e-3)
    assert fit.ok, fit.flags
    assert _deg(fit.u, u_true) < 0.05 and float(fit.u @ u_true) > 0
    assert _deg(fit.n, n_true) < 0.1 and float(fit.n @ n_true) > 0
    assert np.linalg.norm(fit.start - c_true) < 1e-3 and np.linalg.norm(fit.end - (c_true + 0.38 * u_true)) < 1e-3
    assert abs(fit.length_m - 0.38) < 1e-3
    assert not any(f.startswith("end_not_observed") for f in fit.flags)
    assert fit.line_rms_m < 0.15e-3 and fit.plane_rms_m < 0.1e-3
    assert 0.9 <= fit.diag["line_inlier_ratio"] <= 0.97 and fit.n_inliers > 800
    assert fit.root_polyline_fit.shape == (M, 3) and np.isnan(fit.root_polyline_fit[0]).all()
    clean = gf & ~out                                          # 아웃라이어 행은 투영점의 u 방향 위치가 그만큼 밀린다
    assert np.abs(fit.root_polyline_fit[clean] - (c_true + np.outer(s[clean], u_true))).max() < 0.3e-3
    assert fit.feature_medians["root_gap_mm"] == 5.0 and np.isnan(fit.feature_medians["wall_shear_k"])
    assert fit.feature_medians["ok_fraction"] == pytest.approx(gf.mean())
    d = fit.to_dict()
    assert d["length_m"] == pytest.approx(fit.length_m) and len(d["start"]) == 3
    # 점 부족
    few = refine_from_pass(P[:5], Q, gf[:5], None, None, u_prior=u_prior, n_prior=n_prior)
    assert not few.ok and "few_groove_profiles" in few.flags
    c3, u3, inl3, rms3 = ransac_line_3d(np.zeros((3, 3)), 1e-3)
    assert np.isnan(u3).all() and inl3.sum() == 0


def test_refine_from_pass_range_is_line_inlier_extent_without_end_flags():
    """2026-10-08 (4d): refine_from_pass gives only a provisional range — the min/max of the line inliers' s (gaps and
    outliers do not move it) — and sets no end flags; end observation is decided by identify_seam_ends."""
    rng = np.random.default_rng(7)
    u_true = unit([1.0, 0.0, 0.0])
    n_true = np.array([0.0, 0.0, 1.0])
    c_true = np.array([0.6, 0.0, 0.04])
    M = 400
    s = np.linspace(0.0, 0.40, M)
    P = c_true + np.outer(s, u_true) + rng.normal(0, 0.03e-3, (M, 3))
    Q = c_true + n_true * 0.01 + np.outer(rng.uniform(0.0, 0.40, 2000), u_true) + np.outer(rng.uniform(-0.05, 0.05, 2000), [0, 1, 0])
    gf = np.zeros(M, bool)
    gf[30:370] = True
    gf[200:240] = False                                        # a long gap does not split the range
    P[365] += u_true * 0.02 + n_true * 0.005                   # an off-line point beyond the run does not stretch it
    f = refine_from_pass(P, Q, gf, None, GrooveSpec(bevel_side="auto", weld_length_mm=380.0), u_prior=u_true, n_prior=n_true)
    assert f.ok and not any(fl.startswith("end_") for fl in f.flags), f.flags
    assert np.linalg.norm(f.start - (c_true + u_true * s[30])) < 0.5e-3
    assert np.linalg.norm(f.end - (c_true + u_true * s[369])) < 0.5e-3
    assert "run" not in f.diag and "extent_vs_spec_mm" not in f.diag


def test_feature_medians_follow_seam_scan_features_selection():
    """Minor-2: for a SeamScanFeatures input the pass fit's wall_shear_k / floor_visible_fraction are SeamScanFeatures.medians()
    (ok & in_cruise selection), not a recomputation over every ok profile including ramps."""
    from scan_core.localize import _feature_medians
    from scan_core.seam_features import SeamScanFeatures
    M = 8
    sf = SeamScanFeatures(n_profiles=M, indices=np.arange(M), ok=np.ones(M, bool), in_cruise=np.array([0, 0, 1, 1, 1, 1, 0, 0], bool),
                          top_opening_mm=np.full(M, 10.77), root_gap_mm=np.full(M, 5.0), bevel_angle_deg=np.full(M, 30.0),
                          depth_mm=np.full(M, 10.0), hi_lo_mm=np.zeros(M), tilt_deg=np.zeros(M),
                          wall_shear_k=np.array([0.2, 0.2, 0.01, 0.01, 0.02, 0.02, 0.2, 0.2]),
                          floor_visible=np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
                          wall_points_n=np.full(M, 6), features=[SimpleNamespace(ok=True, groove_found=True, occluded_floor=True,
                                                                                 wall_shear_k=0.2)] * M)
    ref = sf.medians()
    fm = _feature_medians(sf, np.ones(M, bool))
    assert fm["wall_shear_k"] == ref["wall_shear_k"] == pytest.approx(0.015)          # cruise profiles only
    assert fm["floor_visible_fraction"] == ref["floor_visible_fraction"] == 1.0
    assert fm["root_gap_mm"] == 5.0 and fm["ok_fraction"] == 1.0 and fm["n_groove_profiles"] == M
    assert fidelity_gate(fm, ZERO_RES, GrooveSpec(bevel_side="auto"), "fidelity", extent=EXT_OK).status == "PASS"
    # dict / None inputs keep the keys
    fd = _feature_medians({"root_gap_mm": [5.0, 5.2, np.nan]}, np.array([True, True, False]))
    assert fd["root_gap_mm"] == pytest.approx(5.1) and np.isnan(fd["wall_shear_k"]) and fd["ok_fraction"] == pytest.approx(2 / 3)
    fn = _feature_medians(None, np.zeros(0, bool))
    assert np.isnan(fn["wall_shear_k"]) and fn["ok_fraction"] == 0.0 and fn["n_groove_profiles"] == 0


def test_fidelity_gate_square_groove_does_not_require_bevel_angle():
    """Minor-4: bevel_angle_deg is required/evaluated only for specimen kinds with a bevel (GROOVE_KINDS: single_bevel,
    single_v). A square groove (no bevel lines -> NaN median) is not DATA_FAIL and the angle is reported as not evaluated."""
    sq = GrooveSpec(kind="square", bevel_side="auto", root_gap_mm=5.0)
    assert sq.n_bevels == 0 and sq.bevel_angle_deg == 0.0 and sq.top_opening_mm == 5.0
    fm = {"root_gap_mm": 5.0, "top_opening_mm": 5.0, "depth_mm": 10.0, "bevel_angle_deg": float("nan"), "hi_lo_mm": 0.1,
          "wall_shear_k": float("nan")}
    v = fidelity_gate(fm, ZERO_RES, sq, "fidelity", extent=EXT_OK)
    assert v.status == "PASS", v.summary()
    assert v.checks["bevel_angle_deg"][2] is None and not any("bevel_angle_deg missing" in n for n in v.notes)
    assert v.checks["geometry_identity_mm"][2] is None and abs(v.checks["geometry_identity_mm"][0]) < 1e-9
    # a stray finite bevel reading on a square groove is ignored (not evaluated), never a SPEC_FAIL
    assert fidelity_gate(dict(fm, bevel_angle_deg=25.0), ZERO_RES, sq, "fidelity", extent=EXT_OK).checks["bevel_angle_deg"][2] is None
    assert fidelity_gate(dict(fm, bevel_angle_deg=25.0), ZERO_RES, sq, "fidelity", extent=EXT_OK).status == "PASS"
    # bevelled kinds still require it
    for kind, side in (("single_bevel", "auto"), ("single_v", "auto")):
        sp = GrooveSpec(kind=kind, bevel_side=side)
        v2 = fidelity_gate(dict(fm, top_opening_mm=sp.top_opening_mm), ZERO_RES, sp, "fidelity", extent=EXT_OK)
        assert v2.status == "DATA_FAIL" and v2.checks["bevel_angle_deg"][2] is False and "bevel_angle_deg missing" in v2.notes


def test_refine_from_simulated_scan_root_polyline_and_gate(calib_lab):
    """1G 짧은 합성 스캔 (루트 중심 y=+3 mm): 스캔 → 프로파일별 특징 → 루트 폴리라인·판 점(I2) → 정련 → fine 게이트 PASS."""
    spec_true = GrooveSpec(bevel_side="left")
    yc = 0.003
    scene = GrooveScene.from_spec(spec_true, yc=yc)
    job = ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, -0.14, 0.05], surface_normal=[0, 0, 1], table_z_m=-0.05,
                  specimen_top_z_m=0.05, lead_in_m=0.02, lead_out_m=0.02, speed_mps=0.05, accel_mps2=0.5,
                  profile_fps=20.0, pitch_deg=0.0, job_id="loc_syn")
    cfg = SynthConfig(n_points=704, seed=5, sigma_z_mm=0.02, dropout_prob=0.01, robot_poll_rate_hz=0.0)
    syn = simulate_scan(job, calib_lab, scene, cfg)
    res = reconstruct_scan(syn.frames, syn.robot, calib_lab, job, fps_nominal=20.0)
    sf = extract_seam_features_along_scan(res, SPEC_AUTO, centre="root", keep_features=True)
    P, gf = polyline_from_scan_features(res, sf)
    Q = plate_points_from_scan(res, sf.features, sf.indices)
    assert P.shape == (sf.n_profiles, 3) and gf.sum() == sf.n_ok and Q.shape[0] > 100 * sf.n_ok
    fit = refine_from_pass(P, Q, gf, sf, SPEC_AUTO, u_prior=job.unit_dir, n_prior=[0, 0, 1],
                           ransac_line_thr_m=1e-3, ransac_plane_thr_m=0.5e-3)
    assert fit.ok, fit.flags
    assert _deg(fit.u, job.unit_dir) < 0.15 and _deg(fit.n, [0, 0, 1]) < 0.1
    # 루트 폴리라인 = 판 프레임 y≈yc, z≈−t (I2: T_base_opt·[x_c,0,z_c])
    Pp = transform_points(inv_transform(res.T_base_plate), fit.root_polyline_fit[gf])
    assert abs(np.median(Pp[:, 1]) - yc) < 0.5e-3 and abs(np.median(Pp[:, 2]) + 0.010) < 0.5e-3
    # 판 점 높이 (I2) = 판 상면
    assert abs(np.median(height_m(Q)) - 0.05) < 0.2e-3
    assert not any(fl.startswith("end_") for fl in fit.flags)    # 끝 관측 여부는 identify_seam_ends 가 정한다 (시편이 스캔보다 길다)
    assert fit.plane_rms_m < 0.1e-3 and fit.feature_medians["ok_fraction"] > 0.8
    # 게이트: 시편이 스캔보다 길어 끝 미관측 → DATA_FAIL, 형상 (갭·깊이·개구) 은 설계 ±50 % 안 (베벨 쪽은 검사 대상 아님)
    v = fidelity_gate(fit.feature_medians, {"psi_deg": 0.0, "theta_deg": 0.0, "roll_deg": 0.0, "lateral_m": 0.0, "standoff_m": 0.0},
                      SPEC_AUTO, "fidelity", extent={"first_observed": False, "last_observed": False, "length_mm": 60.0})
    assert v.status == "DATA_FAIL" and set(v.failed) == {"end_first_observed", "end_last_observed"}, v.summary()
    assert all(v.checks[k][2] is True for k in ("root_gap_mm", "depth_mm", "top_opening_mm"))
    assert "bevel_side" not in " ".join(v.checks)


# --------------------------------------------------------------------------------------
# (f) 충실도 게이트 · 수렴
# --------------------------------------------------------------------------------------
IDEAL = {"root_gap_mm": 5.0, "top_opening_mm": 10.77, "depth_mm": 10.0, "bevel_angle_deg": 30.0, "hi_lo_mm": 0.1,
         "wall_shear_k": float("nan")}
ZERO_RES = {"psi_deg": 0.0, "theta_deg": 0.0, "roll_deg": 0.0, "lateral_m": 0.0, "standoff_m": 0.0}
EXT_OK = {"first_observed": True, "last_observed": True, "length_mm": 380.0, "spacing_mm": 0.3}


def test_fidelity_gate_design_match_is_half_the_nominal_and_placement_is_diagnostic():
    """판정 기준 (scan_core.measure): 특징 오인식 = 설계 공칭 ±50 % 밖, 작업 허용치는 보고 기준, 센서 배치는 진단 (예전 ALIGN_FAIL·
    coarse/fine 밴드 없음)."""
    spec = GrooveSpec(bevel_side="auto")
    v = fidelity_gate(IDEAL, ZERO_RES, spec, "fidelity", extent=EXT_OK)
    assert v.status == "PASS" and v.checks["wall_shear_k"][2] is None and v.checks["root_gap_mm"] == (5.0, (2.5, 7.5), True)
    assert v.to_dict()["status"] == "PASS" and "PASS" in v.summary() and not any("bevel_side" in k for k in v.checks)
    # 갭 7 = 설계 5 의 +40 %: 측정을 믿고 보고 (작업 허용치 0.5 밖), 8 = +60 %: 오인식
    v = fidelity_gate(dict(IDEAL, root_gap_mm=7.0), ZERO_RES, spec, "fidelity", extent=EXT_OK, qc_targets={"root_gap_mm": {"tol": 0.5}})
    assert v.status == "PASS" and v.specimen["root_gap_mm"]["within"] is False
    assert fidelity_gate(dict(IDEAL, root_gap_mm=8.0), ZERO_RES, spec, "fidelity", extent=EXT_OK).failed == ["root_gap_mm"]
    # 센서 배치 잔차는 진단 (큰 psi·lateral 도 PASS)
    v = fidelity_gate(IDEAL, dict(ZERO_RES, psi_deg=4.0, lateral_m=0.012), spec, "fidelity", extent=EXT_OK)
    assert v.status == "PASS" and v.checks["placement_psi_deg"] == (4.0, None, None)
    # 데이터 부족: 필수 중앙값 NaN
    assert fidelity_gate(dict(IDEAL, depth_mm=float("nan")), ZERO_RES, spec, "fidelity", extent=EXT_OK).status == "DATA_FAIL"
    # 공칭은 spec 에서: 갭 8 시편이면 8 이 공칭
    spec8 = GrooveSpec(bevel_side="right", root_gap_mm=8.0)
    fm8 = dict(IDEAL, root_gap_mm=8.0, top_opening_mm=spec8.top_opening_mm)
    assert fidelity_gate(fm8, ZERO_RES, spec8, "fidelity", extent=EXT_OK).status == "PASS"
    with pytest.raises(ValueError):
        fidelity_gate(IDEAL, ZERO_RES, spec, "medium")


def test_specimen_auto_side_and_extractor_side_agnostic():
    """GrooveSpec 'auto' = 측 무관: 단면은 left 로 생성, 비교는 None; left/right 는 기존 동작."""
    s = GrooveSpec(bevel_side="auto")
    assert s.side_agnostic and s.resolved_bevel_side() == "left" and s.top_corners_mm() == GrooveSpec(bevel_side="left").top_corners_mm()
    assert GrooveSpec(bevel_side="unknown").side_agnostic and not GrooveSpec(bevel_side="right").side_agnostic
    assert GrooveSpec.from_dict(s.to_dict()).bevel_side == "auto"
    x = np.linspace(-80, 80, 1200)
    z = 430.0 + 0.0 * x
    f = extract_cross_section_features(x, z, s)
    assert f.ok and not f.groove_found and f.spec_deltas is None
    assert f.plate_inlier_idx().size == 1200 and "plate_inlier_idx_left" not in f.to_dict()
    # 그루브 프로파일: 쪽은 판정하되 spec 비교는 None ('auto'), 'left' spec 은 기존처럼 True/False
    calib = Calib.lab_initial()
    T_bf = _pose(calib, R0, (0.6, 0.0, H_PLATE + 0.40))
    xg, zg = static_profile(T_bf, calib, GrooveScene.from_spec(GrooveSpec(bevel_side="right")), T_PLATE, CFG)
    fg = extract_cross_section_features(xg, zg, s)
    assert fg.ok and fg.groove_found and fg.bevel_side == "right" and fg.spec_deltas["bevel_side_match"] is None
    assert extract_cross_section_features(xg, zg, GrooveSpec(bevel_side="left")).spec_deltas["bevel_side_match"] is False
    idx = fg.plate_inlier_idx()
    assert idx.size > 500 and np.all(np.isfinite(zg[idx])) and (np.abs(xg[idx]) > 2.0).all()      # 그루브 안 열은 판이 아님
