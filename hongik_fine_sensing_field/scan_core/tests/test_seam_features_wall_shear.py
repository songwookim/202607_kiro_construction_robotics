"""seam_features 정렬 진단: 직각 벽 전단 wall_shear_k (de-tilt 프레임 dx'/dh') 와 바닥(뒷댐재) 가시성 floor_visible.

기하: 평면 벽 ∩ 레이저 평면은 레이저 평면이 판 법선(순수 요) 또는 가로축(순수 피치)을 포함하는 한 판 상면에 수직이다.
요·피치가 함께 있을 때만 벽이 기울고 그 크기는 k = tan(yaw)·sin(pitch) (GROOVE_PROFILE_ANALYSIS §5 H2 '요+피치').
합성 센서(광선 부채꼴)에서는 광선이 벽 쪽으로 흘러야 벽에 점이 생기므로 (정렬·중심 자세에서는 수직 벽에 광선이 닿지 않아
wall_points_n = 0 → k = NaN = 미평가) 그루브를 센서 중심에서 ±50 mm 비켜 놓아 부채꼴 기울기로 벽을 비춘다."""
import json
import math

import numpy as np
import pytest

from scan_core.frames import make_transform, rot_x, rot_y, rot_z
from scan_core.localize import fidelity_gate
from scan_core.path_gen import plate_frame
from scan_core.reconstruct import reconstruct_scan
from scan_core.seam_features import extract_cross_section_features, extract_seam_features_along_scan
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, simulate_profile, simulate_scan
from scan_core.types import ScanJob

from test_seam_features import REAL_NPZ

T_PLATE = plate_frame([0, 0, 0], [1, 0, 0], [0, 0, 1])
R0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)       # X_s = plate y (가로), Y_s = plate x (진행), Z_s = -plate z
K_GATE = 0.03                       # 시험 기준값: 눈에 띄는 벽 기울기 (게이트는 wall_shear_k 를 진단으로만 보고한다)
EXT_OK = {"first_observed": True, "last_observed": True, "length_mm": 380.0, "spacing_mm": 0.3}
# 부채꼴 광선이 직각 벽(베벨 반대쪽)을 비추는 센서 가로 오프셋 [m] (센서를 -y 로 옮기면 그루브가 센서 +x 에 보인다)
LIT_OFF = {"left": -0.05, "right": +0.05}
ZERO_RES = {"psi_deg": 0.0, "theta_deg": 0.0, "roll_deg": 0.0, "lateral_m": 0.0, "standoff_m": 0.0}


def _profile(spec, *, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, y_off_m=0.0, z_m=0.43, noise=0.02, dropout=0.02, seed=1):
    """요 = 판 법선(base z) 회전, 피치 = 가로축(base y = X_s) 회전, 롤 = 진행축(base x) 회전."""
    R = rot_z(math.radians(yaw_deg)) @ rot_y(math.radians(pitch_deg)) @ rot_x(math.radians(roll_deg)) @ R0
    T_bo = make_transform(R, (0.0, y_off_m, z_m))
    sc = GrooveScene.from_spec(spec)
    return simulate_profile(T_bo, sc, T_PLATE, SynthConfig(n_points=1408, seed=seed, sigma_z_mm=noise, dropout_prob=dropout))


def _yaw_pitch_profile(side: str, yaw_deg: float, pitch_deg: float, **kw):
    """측 대칭 자세: left 는 (−yaw, +pitch, −off), right 는 (+yaw, +pitch, +off) → 두 경우 모두 벽이 비춰진다."""
    sgn = -1.0 if side == "left" else 1.0
    return _profile(GrooveSpec(bevel_side=side), yaw_deg=sgn * yaw_deg, pitch_deg=pitch_deg, y_off_m=LIT_OFF[side], **kw)


# --------------------------------------------------------------------------------------
# 정렬 자세
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("side", ["left", "right"])
def test_aligned_wall_vertical_and_floor_visible(side):
    """정렬 자세(벽이 비춰지도록 비킴): 벽 점 ≥ 4, |k| < 0.01, 벽 = 베벨 반대쪽, 바닥 보임 · 폭 ≈ 루트갭."""
    spec = GrooveSpec(bevel_side=side)
    x, z = _profile(spec, y_off_m=LIT_OFF[side])
    f = extract_cross_section_features(x, z, spec)
    assert f.ok and f.groove_found and f.bevel_side == side, f.report()
    assert f.wall_points_n >= 4 and f.wall_side == ("right" if side == "left" else "left"), f.report()
    assert np.isfinite(f.wall_shear_k) and abs(f.wall_shear_k) < 0.01, f.report()
    assert f.wall_span_mm > 3.0
    assert f.floor_visible is True and f.floor_expected_mm == 5.0
    assert 4.0 <= f.floor_span_mm <= 5.6, f.report()
    assert "wall_ambiguous" not in f.flags
    assert abs(f.plate_slope_left) < 0.01 and abs(f.plate_slope_right) < 0.01
    rep = f.report()
    assert "align: wall(" in rep and "floor visible=True" in rep and "wall-ambiguous" not in rep
    d = f.to_dict()
    for k in ("wall_shear_k", "wall_points_n", "wall_span_mm", "wall_side", "floor_visible", "floor_span_mm",
              "floor_expected_mm", "plate_slope_left", "plate_slope_right",
              # 기존 키 유지
              "root_gap_mm", "bevel_side", "hi_lo_ambiguous", "occluded_floor", "flags"):
        assert k in d, k
    assert d["floor_visible"] is True and isinstance(d["wall_points_n"], int)
    json.dumps(d)


@pytest.mark.parametrize("roll", [0.0, 8.0, -8.0])
def test_centred_aligned_wall_not_lit_is_not_evaluated(roll):
    """센서 중심 자세(롤 포함): 수직 벽에 광선이 닿지 않아 wall_points_n = 0, k = NaN(미평가); 바닥은 보인다.
    fidelity_gate 는 NaN 을 '미평가'(None) 로 다루고 floor_visible_fraction 은 통과."""
    spec = GrooveSpec(bevel_side="left")
    x, z = _profile(spec, roll_deg=roll, y_off_m=0.006)
    f = extract_cross_section_features(x, z, spec)
    assert f.ok and f.groove_found and f.tilt_deg == pytest.approx(roll, abs=0.1)
    assert f.wall_points_n == 0 and np.isnan(f.wall_shear_k) and f.wall_side == "none"
    assert f.floor_visible is True and 4.5 <= f.floor_span_mm <= 5.6
    assert f.plate_slope_left == pytest.approx(math.tan(math.radians(roll)), abs=0.005)
    assert "shear k=n/a" in f.report()
    fm = {"root_gap_mm": f.root_gap_mm, "depth_mm": f.depth_mm, "bevel_angle_deg": f.bevel_angle_deg,
          "top_opening_mm": f.top_opening_mm, "hi_lo_mm": f.hi_lo_mm, "wall_shear_k": f.wall_shear_k,
          "floor_visible_fraction": 1.0 if f.floor_visible else 0.0, "ok_fraction": 1.0}
    v = fidelity_gate(fm, ZERO_RES, spec, "fidelity", extent=EXT_OK)
    assert v.checks["wall_shear_k"][2] is None and v.status == "PASS"


@pytest.mark.parametrize("side", ["left", "right"])
def test_pure_yaw_does_not_lean_the_wall(side):
    """순수 요 15° (피치 0): 벽 ∩ 레이저 평면은 여전히 수직 → k ≈ 0 또는 미평가; 대신 루트갭이 1/cos(yaw) 로 넓어 보인다."""
    spec = GrooveSpec(bevel_side=side)
    sgn = -1.0 if side == "left" else 1.0
    x, z = _profile(spec, yaw_deg=sgn * 15.0, y_off_m=LIT_OFF[side])
    f = extract_cross_section_features(x, z, spec)
    assert f.ok and f.groove_found, f.report()
    assert np.isnan(f.wall_shear_k) or abs(f.wall_shear_k) < 0.01, f.report()
    assert 5.0 / math.cos(math.radians(15.0)) - 0.2 <= f.root_gap_mm <= 5.0 / math.cos(math.radians(15.0)) + 0.4, f.report()
    assert f.floor_visible is True


# --------------------------------------------------------------------------------------
# 요 + 피치 → 벽 전단
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("yaw,pitch", [(10.0, 10.0), (15.0, 15.0), (20.0, 20.0)])
def test_yaw_pitch_wall_shear_matches_tan_yaw_sin_pitch_and_is_side_agnostic(yaw, pitch):
    """|k| = tan(yaw)·sin(pitch) (±0.01), 왼/오른 베벨 대칭 자세에서 |k| 가 0.01 이내로 같다; 15°/15° 는 |k| > 0.05 > 게이트."""
    pred = math.tan(math.radians(yaw)) * math.sin(math.radians(pitch))
    ks = {}
    for side in ("left", "right"):
        x, z = _yaw_pitch_profile(side, yaw, pitch)
        f = extract_cross_section_features(x, z, GrooveSpec(bevel_side=side))
        assert f.ok and f.groove_found and f.bevel_side == side, f.report()
        assert f.wall_points_n >= 4 and f.wall_side == ("right" if side == "left" else "left"), f.report()
        assert np.isfinite(f.wall_shear_k) and abs(abs(f.wall_shear_k) - pred) < 0.01, (f.report(), pred)
        assert "wall_ambiguous" not in f.flags
        assert f.floor_visible is True
        ks[side] = f.wall_shear_k
    assert abs(abs(ks["left"]) - abs(ks["right"])) < 0.01, ks
    assert np.sign(ks["left"]) == -np.sign(ks["right"])                  # 거울 자세 → 부호 반대
    if (yaw, pitch) == (15.0, 15.0):
        assert abs(ks["left"]) > 0.05 and abs(ks["right"]) > 0.05 and abs(ks["left"]) > K_GATE


def test_wall_shear_with_side_agnostic_spec_and_square_groove():
    """spec bevel_side='unknown' 이어도 같은 k; 직각 그루브(kind='square', bevel 'none') 는 비춰진 벽으로 평가하고 플래그 없음."""
    x, z = _yaw_pitch_profile("left", 15.0, 15.0)
    f_known = extract_cross_section_features(x, z, GrooveSpec(bevel_side="left"))
    f_unknown = extract_cross_section_features(x, z, GrooveSpec(bevel_side="unknown"))
    assert f_unknown.bevel_side == "left" and f_unknown.wall_side == "right"
    assert f_unknown.wall_shear_k == pytest.approx(f_known.wall_shear_k, abs=1e-6)
    assert "wall_ambiguous" not in f_unknown.flags
    # 직각 그루브: 양쪽 다 직각 벽 → 점 많은(비춰진) 쪽
    sq = GrooveSpec(kind="square", bevel_side="unknown", root_gap_mm=5.0)
    x, z = _profile(sq, yaw_deg=-15.0, pitch_deg=15.0, y_off_m=LIT_OFF["left"])
    f = extract_cross_section_features(x, z, sq)
    assert f.ok and f.groove_found and f.bevel_side == "none", f.report()
    assert f.wall_points_n >= 4 and f.wall_side == "right"
    assert abs(abs(f.wall_shear_k) - math.tan(math.radians(15.0)) * math.sin(math.radians(15.0))) < 0.01, f.report()
    assert "wall_ambiguous" not in f.flags


def test_real_lab_profile_single_bevel_and_shear_alarm():
    """실기 중앙값 프로파일(수동 자세, 요+피치, 판 기울기 8°): 직각 벽이 요·피치로 기울어 보여도 단일 베벨(left) 과 직각 벽(right)
    으로 구분하고, 그 벽의 전단 |k| > 0.03 으로 자세 경보 (분석 문서 §8). (2026-10-02 이전 추출기는 기운 벽을 둘째 베벨로 보아
    'both' + wall_ambiguous 였다 — 판 끝 연속성 연장·판 끝에 이어진 점만 쓰는 그룹화 뒤 단일 베벨로 바로잡힘.)"""
    d = np.load(REAL_NPZ)
    f = extract_cross_section_features(d["x_mm"].astype(float), d["z_mm"].astype(float), GrooveSpec(bevel_side="unknown"),
                                       sensor_tilt_hint=8.0)
    assert f.ok and f.groove_found and f.bevel_side == "left", f.report()
    assert "wall_ambiguous" not in f.flags and f.wall_side == "right"
    assert np.isfinite(f.wall_shear_k) and abs(f.wall_shear_k) > K_GATE and f.wall_points_n >= 4
    assert f.floor_visible is True and f.floor_span_mm > 3.0
    assert f.plate_slope_left == pytest.approx(math.tan(math.radians(f.plate_left["tilt_deg"])), abs=1e-6)
    rep = f.report()
    assert "(wall-ambiguous)" not in rep and "AMBIGUOUS" not in rep
    json.dumps(f.to_dict())


# --------------------------------------------------------------------------------------
# 바닥 가시성
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("side", ["left", "right"])
def test_deep_narrow_gap_in_fan_shadow_floor_not_visible(side):
    """루트갭 1 mm, 광선이 벽에서 멀어지는 쪽으로 흐르는 오프셋 → 바닥이 벽 그늘에 가려 floor_visible False (+ occluded_floor)."""
    spec = GrooveSpec(bevel_side=side, root_gap_mm=1.0)
    x, z = _profile(spec, y_off_m=-LIT_OFF[side])
    f = extract_cross_section_features(x, z, spec)
    assert f.ok and f.groove_found, f.report()
    assert f.floor_visible is False and f.floor_expected_mm == 1.0, f.report()
    assert f.floor_span_mm < 0.6 * 1.0 and "occluded_floor" in f.flags
    # 같은 갭이라도 광선이 벽 쪽으로 흐르면 바닥이 보인다
    x, z = _profile(spec, y_off_m=LIT_OFF[side])
    f2 = extract_cross_section_features(x, z, spec)
    assert f2.ok and f2.floor_visible is True and f2.floor_span_mm >= 0.6, f2.report()


@pytest.mark.parametrize("side", ["left", "right"])
def test_dark_backing_bar_dropout_floor_not_visible(side):
    """판 상면보다 8 mm 이상 깊은 열을 모두 NaN (뒷댐재 무반환) → 바닥 대역은 베벨 하단 잔여점뿐 → floor_visible False."""
    spec = GrooveSpec(bevel_side=side)
    x, z = _profile(spec, y_off_m=0.0)
    z_top = float(np.nanmedian(z[np.abs(x) > 30.0]))
    drop = z > z_top + 8.0
    x2, z2 = x.copy(), z.copy()
    x2[drop] = np.nan
    z2[drop] = np.nan
    f = extract_cross_section_features(x2, z2, spec)
    assert f.ok and f.groove_found, f.report()
    assert f.floor_visible is False and f.floor_span_mm < 3.0, f.report()
    assert "floor visible=False" in f.report()
    # 그루브가 없으면 바닥 대역을 탐색하지 않는다 → None
    xp = np.linspace(-80, 80, 1200)
    f0 = extract_cross_section_features(xp, 430.0 + 0.05 * xp)
    assert not f0.groove_found and f0.floor_visible is None and np.isnan(f0.floor_span_mm)


# --------------------------------------------------------------------------------------
# 스캔 전체 → 중앙값 전파
# --------------------------------------------------------------------------------------
def _scan(calib, spec, *, yc, angle_deg, pitch_deg):
    scene = GrooveScene.from_spec(spec, yc=yc, angle_deg=angle_deg)
    job = ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, -0.14, 0.05], surface_normal=[0, 0, 1], table_z_m=-0.05,
                  specimen_top_z_m=0.05, lead_in_m=0.02, lead_out_m=0.02, speed_mps=0.05, accel_mps2=0.5,
                  profile_fps=20.0, pitch_deg=pitch_deg, job_id="groove_syn_wall")
    cfg = SynthConfig(n_points=1408, seed=5, sigma_z_mm=0.02, dropout_prob=0.01, robot_poll_rate_hz=0.0)
    syn = simulate_scan(job, calib, scene, cfg)
    res = reconstruct_scan(syn.frames, syn.robot, calib, job, fps_nominal=20.0)
    return extract_seam_features_along_scan(res, spec, stride=2, centre="root", keep_features=True)


def test_scan_medians_aligned_wall_not_evaluated_floor_visible(calib_lab):
    """정렬된 1G 스캔: 프로파일별 배열 채워짐, wall_shear_k 중앙값 NaN(미평가), floor_visible_fraction ≈ 1, wall_points_median 0."""
    spec = GrooveSpec(bevel_side="left")
    sf = _scan(calib_lab, spec, yc=0.003, angle_deg=0.0, pitch_deg=0.0)
    assert sf.n_ok >= 0.9 * sf.n_profiles, sf.summary()
    M = sf.n_profiles
    assert sf.wall_shear_k.shape == (M,) and sf.floor_visible.shape == (M,) and sf.floor_span_mm.shape == (M,)
    assert sf.wall_points_n.shape == (M,) and sf.wall_points_n.dtype.kind == "i"
    assert sf.in_cruise.shape == (M,) and sf.in_cruise.dtype == bool and 0 < int(sf.in_cruise.sum()) < M
    m = sf.medians()
    assert np.isnan(m["wall_shear_k"]) and m["wall_points_median"] == 0.0
    assert m["floor_visible_fraction"] >= 0.9
    assert np.nanmedian(sf.floor_span_mm[sf.ok]) == pytest.approx(5.0, abs=0.5)
    assert all(np.isnan(sf.wall_shear_k[sf.ok]))
    assert set(np.unique(sf.floor_visible[sf.ok])) <= {0.0, 1.0}
    d = sf.to_dict()
    for k in ("wall_shear_k", "wall_points_n", "floor_visible", "floor_span_mm", "in_cruise"):
        assert k in d and len(d[k]) == M
    assert set(d["medians"]) >= {"wall_shear_k", "floor_visible_fraction", "wall_points_median", "root_gap_mm"}
    json.dumps(d)
    assert "align: wall shear k n/a" in sf.summary() and "floor visible 1.00" in sf.summary()
    v = fidelity_gate(m, ZERO_RES, spec, "fidelity", extent=EXT_OK)
    assert v.checks["wall_shear_k"][2] is None


@pytest.mark.parametrize("side,yc,angle", [("left", +0.05, -15.0), ("right", -0.05, +15.0)])
def test_scan_medians_yaw_pitch_wall_shear_propagates_to_the_gate_diagnostic(calib_lab, side, yc, angle):
    """그루브 축 15° (요) + 잡 피치 15°: 크루즈 프로파일의 wall_shear_k 중앙값 ≈ tan15·sin15 = 0.069 → fidelity_gate 진단 값 (센서
    배치는 판정이 아니다 — 예전 fine 게이트의 0.03 문턱은 없앴다)."""
    spec = GrooveSpec(bevel_side=side)
    sf = _scan(calib_lab, spec, yc=yc, angle_deg=angle, pitch_deg=15.0)
    assert sf.n_ok >= 0.9 * sf.n_profiles, sf.summary()
    m = sf.medians()
    pred = math.tan(math.radians(15.0)) * math.sin(math.radians(15.0))
    assert np.isfinite(m["wall_shear_k"]) and abs(abs(m["wall_shear_k"]) - pred) < 0.01, sf.summary()
    assert m["wall_points_median"] >= 4 and m["floor_visible_fraction"] >= 0.9
    sel = sf.ok & sf.in_cruise
    assert np.isfinite(sf.wall_shear_k[sel]).mean() > 0.8
    # 개별 프로파일 CrossSectionFeatures 도 같은 값을 갖는다 (localize._feature_medians 가 읽는 경로)
    ks = np.array([f.wall_shear_k for f in sf.features])
    assert np.nanmedian(ks[sel]) == pytest.approx(m["wall_shear_k"], abs=1e-9)
    v = fidelity_gate(m, ZERO_RES, spec, "fidelity", extent=EXT_OK)
    assert v.checks["wall_shear_k"][2] is None and v.checks["wall_shear_k"][0] == pytest.approx(m["wall_shear_k"])
