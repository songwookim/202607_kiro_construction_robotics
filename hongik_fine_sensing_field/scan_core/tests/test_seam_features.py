"""seam_features: 합성 그루브 프로파일(8° 롤, 노이즈 0.02 mm, NaN 드롭아웃)에서 갭/베벨/깊이 복원 (0.2 mm / 1°),
실기 시간 중앙값 프로파일(rec_plate_sh100_gain300) 처리, 스캔 전체 폴리라인."""
import json
import math
import os

import numpy as np
import pytest

from scan_core.frames import inv_transform, make_transform, rot_x, transform_points
from scan_core.path_gen import plate_frame
from scan_core.reconstruct import reconstruct_scan
from scan_core.seam_features import (HI_LO_AMBIGUOUS_MM, CrossSectionFeatures, Line2D, extract_cross_section_features,
                                     extract_seam_features_along_scan, fit_line_pca, ransac_line)
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, simulate_profile, simulate_scan
from scan_core.types import ScanJob

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REAL_NPZ = os.path.join(DATA, "real_groove_median_sh100_gain300.npz")

T_PLATE = plate_frame([0, 0, 0], [1, 0, 0], [0, 0, 1])
R0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)       # X_s = plate y (라인이 그루브를 가로지름), Z_s = -plate z


def _profile(spec: GrooveSpec, roll_deg: float, y_off_m: float = 0.006, z_m: float = 0.43, noise: float = 0.02,
             dropout: float = 0.02, seed: int = 1, angle_deg: float = 0.0):
    sc = GrooveScene.from_spec(spec, angle_deg=angle_deg)
    T_bo = make_transform(rot_x(math.radians(roll_deg)) @ R0, (0.0, y_off_m, z_m))
    return simulate_profile(T_bo, sc, T_PLATE, SynthConfig(n_points=1408, seed=seed, sigma_z_mm=noise, dropout_prob=dropout))


# --------------------------------------------------------------------------------------
# 직선 도우미
# --------------------------------------------------------------------------------------
def test_line_helpers():
    x = np.linspace(0, 10, 50)
    h = 0.5 * x + 1.0
    ln = fit_line_pca(x, h)
    assert ln.angle_deg == pytest.approx(math.degrees(math.atan(0.5)))
    assert ln.h_at(4.0) == pytest.approx(3.0) and ln.rms_mm == pytest.approx(0.0, abs=1e-9)
    h2 = h.copy()
    h2[::7] += 5.0                                        # 스파이크
    ln2, inl = ransac_line(x, h2, thr_mm=0.1, rng=np.random.default_rng(0))
    assert inl.sum() == (h2 == h).sum() and ln2.angle_deg == pytest.approx(ln.angle_deg, abs=1e-6)
    v = Line2D([2.0, 0.0], [0.0, 1.0])
    p = ln.intersect(v)
    assert p[0] == pytest.approx(2.0) and p[1] == pytest.approx(2.0)
    assert v.angle_from_vertical_deg == pytest.approx(0.0) and Line2D([0, 0], [1, 0]).angle_from_vertical_deg == 90.0
    r = ln.rotated(math.atan(0.5))
    assert r.angle_deg == pytest.approx(0.0, abs=1e-9)
    assert ransac_line(x[:2], h[:2], min_pts=3)[0] is None


# --------------------------------------------------------------------------------------
# 합성 프로파일 → 치수 복원
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("roll", [0.0, 8.0, -8.0])
def test_synthetic_groove_recovers_gap_bevel_depth(side, roll):
    spec = GrooveSpec(bevel_side=side)
    x, z = _profile(spec, roll)
    assert np.isnan(z).sum() > 10                                   # 드롭아웃 존재
    f = extract_cross_section_features(x, z, spec)
    assert f.ok and f.groove_found and not f.insufficient_points, f.report()
    assert f.tilt_deg == pytest.approx(roll, abs=0.1)
    assert f.bevel_side == side
    assert abs(f.root_gap_mm - 5.0) <= 0.2, f.report()
    assert abs(f.bevel_angle_deg - 30.0) <= 1.0
    assert abs(f.depth_mm - 10.0) <= 0.2
    assert abs(f.top_opening_mm - spec.top_opening_mm) <= 0.25
    assert abs(f.hi_lo_mm) <= 0.1
    assert not f.hi_lo_ambiguous and "AMBIGUOUS" not in f.report()
    # 공칭 비교 · QC 통과
    assert f.spec_deltas["bevel_side_match"] is True
    assert all(v["pass"] for k, v in f.qc.items() if k != "hi_lo_mm") and f.qc["hi_lo_mm"]["pass"]
    # 센서 프레임 코너/중심이 유효하고 루트 중심이 상부 중심보다 멀다(깊다)
    assert f.root_centre_z_mm > f.groove_centre_z_mm + 8.0
    d = f.to_dict()
    assert d["ok"] is True and isinstance(d["flags"], list) and d["hi_lo_ambiguous"] is False


def test_synthetic_hi_lo_lifted_plate_reported_and_gap_extrapolated_to_plate_bottom():
    """오른 판(직각 쪽)이 1.5 mm 들린 경우: hi-lo +1.5, 갭 5 (spec 두께로 판 바닥까지 외삽), 깊이 L 10 / R 11.5, QC hi-lo FAIL."""
    for side in ("left", "right"):
        spec = GrooveSpec(bevel_side=side, hi_lo_mm=1.5)
        x, z = _profile(spec, 8.0)
        f = extract_cross_section_features(x, z, spec)
        assert f.ok and f.hi_lo_mm == pytest.approx(1.5, abs=0.1) and f.bevel_side == side
        assert abs(f.root_gap_mm - 5.0) <= 0.25, f.report()
        assert f.depth_left_mm == pytest.approx(10.0, abs=0.2) and f.depth_right_mm == pytest.approx(11.5, abs=0.2)
        assert f.qc["hi_lo_mm"]["pass"] is False and f.qc["hi_lo_mm"]["flag"] is False       # 1 < 1.5 ≤ 2
        assert not f.hi_lo_ambiguous
    # 큰 단차(3 mm): 모호성 플래그 + 보고서 문구
    spec = GrooveSpec(bevel_side="left", hi_lo_mm=3.0)
    x, z = _profile(spec, 8.0)
    f = extract_cross_section_features(x, z, spec)
    assert f.hi_lo_mm == pytest.approx(3.0, abs=0.1) and f.hi_lo_ambiguous and "hi_lo_ambiguous" in f.flags
    rep = f.report()
    assert "AMBIGUOUS" in rep and "backing strip" in rep and "obliquely" in rep and f.qc["hi_lo_mm"]["flag"] is True


def test_synthetic_oblique_groove_axis_and_offset_center():
    """그루브 축이 진행축과 5° (H2) — 단면 자체는 여전히 그루브; 센서 중심 오정렬 30 mm 에서도 복원."""
    spec = GrooveSpec(bevel_side="left")
    x, z = _profile(spec, 8.0, y_off_m=-0.03, angle_deg=5.0)
    f = extract_cross_section_features(x, z, spec)
    assert f.ok and abs(f.root_gap_mm - 5.0) <= 0.25 and abs(f.bevel_angle_deg - 30.0) <= 1.5
    assert abs(f.depth_mm - 10.0) <= 0.2


def test_no_groove_and_insufficient_points_paths():
    # 평판만: no_groove
    x = np.linspace(-80, 80, 1200)
    z = 430.0 + 0.14 * x
    f = extract_cross_section_features(x, z)
    assert f.ok and not f.groove_found and "no_groove" in f.flags and f.hi_lo_mm == pytest.approx(0.0, abs=1e-6)
    assert f.tilt_deg == pytest.approx(math.degrees(math.atan(0.14)), abs=1e-3)
    # 점 부족
    f2 = extract_cross_section_features(x[:10], z[:10])
    assert not f2.ok and f2.insufficient_points and "insufficient_points" in f2.flags
    # 전부 NaN
    f3 = extract_cross_section_features(np.full(100, np.nan), np.full(100, np.nan))
    assert not f3.ok and f3.n_valid == 0
    # 힌트와 크게 다르면 노트
    f4 = extract_cross_section_features(x, z, sensor_tilt_hint=0.0)
    assert any("sensor_tilt_hint" in n for n in f4.notes)
    # 판 각도 불일치(한쪽이 판이 아님) → plates_not_parallel, ok=False
    xg = np.linspace(-80, 80, 1200)
    zg = np.where(xg < 0, 430.0 + 0.0 * xg, 430.0 + 0.3 * xg)
    f5 = extract_cross_section_features(xg, zg)
    assert not f5.ok and "plates_not_parallel" in f5.flags


# --------------------------------------------------------------------------------------
# 실기 프로파일 (시간 중앙값, 수동 자세: 기울고 중심 오정렬)
# --------------------------------------------------------------------------------------
def test_real_median_profile_runs_and_finds_plates_and_transition():
    d = np.load(REAL_NPZ)
    x, z = d["x_mm"].astype(float), d["z_mm"].astype(float)
    assert x.shape == (1408,) and int(d["n_frames"]) == 300
    spec = GrooveSpec(bevel_side="unknown")
    f = extract_cross_section_features(x, z, spec, sensor_tilt_hint=8.0)
    rep = f.report()
    assert f.ok and f.n_valid == 1408, rep
    # 두 판 상면 직선, 기울기 8~9° 근방 (왼 7.6°, 오른 8.5°)
    assert f.plate_left is not None and f.plate_right is not None
    assert 7.0 <= f.plate_left["tilt_deg"] <= 9.5 and 7.0 <= f.plate_right["tilt_deg"] <= 9.5
    assert 7.5 <= f.tilt_deg <= 9.0
    assert f.plate_left["rms_mm"] < 0.15 and f.plate_right["rms_mm"] < 0.15
    assert f.plate_left["x_min"] < -80 and f.plate_right["x_max"] > 90
    # 그루브/전이 구간이 x ≈ 0~9 mm 근방 (왼 판 끝 ≈ -3, 오른 판 시작 ≈ 8.7)
    assert f.groove_found
    assert -5.0 <= f.plate_left["x_max"] <= 2.0 and 6.0 <= f.plate_right["x_min"] <= 12.0
    assert -3.0 <= f.groove_centre_x_mm <= 9.0
    assert 6.0 <= f.top_opening_mm <= 15.0
    # hi-lo 는 유한값 (장면 해석은 불확실 → 공칭 단정 금지); > 2 mm 면 보고서가 모호성을 명시
    assert np.isfinite(f.hi_lo_mm)
    if abs(f.hi_lo_mm) > HI_LO_AMBIGUOUS_MM:
        assert "AMBIGUOUS" in rep and "backing strip" in rep
    else:
        assert "AMBIGUOUS" not in rep
    assert np.isfinite(f.depth_mm) and f.depth_mm > 5.0
    assert np.isfinite(f.root_centre_z_mm) and f.root_centre_z_mm > f.groove_centre_z_mm
    assert f.spec_deltas is not None and np.isfinite(f.spec_deltas["hi_lo_mm"])
    # 직렬화 가능
    import json
    json.dumps(f.to_dict())


# --------------------------------------------------------------------------------------
# 스캔 전체 → 폴리라인
# --------------------------------------------------------------------------------------
def test_seam_features_along_synthetic_scan(calib_lab):
    """1G 짧은 스캔(GrooveScene, 루트 중심 y = +3 mm 오프셋): 프로파일별 특징 + base 폴리라인이 판 프레임에서 y≈3 mm, z≈−10 mm."""
    spec = GrooveSpec(bevel_side="left")
    yc = 0.003
    scene = GrooveScene.from_spec(spec, yc=yc)
    job = ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, -0.14, 0.05], surface_normal=[0, 0, 1], table_z_m=-0.05,
                  specimen_top_z_m=0.05, lead_in_m=0.02, lead_out_m=0.02, speed_mps=0.05, accel_mps2=0.5,
                  profile_fps=20.0, pitch_deg=0.0, job_id="groove_syn")
    cfg = SynthConfig(n_points=704, seed=5, sigma_z_mm=0.02, dropout_prob=0.01, robot_poll_rate_hz=0.0)
    syn = simulate_scan(job, calib_lab, scene, cfg)
    res = reconstruct_scan(syn.frames, syn.robot, calib_lab, job, fps_nominal=20.0)
    sf = extract_seam_features_along_scan(res, spec, stride=2, centre="root", keep_features=True)
    assert sf.n_profiles == (res.n_profiles + 1) // 2 and sf.n_ok >= 0.8 * sf.n_profiles, sf.summary()
    m = sf.medians()
    assert abs(m["root_gap_mm"] - 5.0) <= 0.3 and abs(m["bevel_angle_deg"] - 30.0) <= 1.5 and abs(m["depth_mm"] - 10.0) <= 0.3
    assert abs(m["hi_lo_mm"]) <= 0.2
    # 폴리라인: 판 프레임(원점 start, x=u, z=n)으로 옮기면 y ≈ yc, z ≈ −t (루트 중심 = 바닥)
    P = transform_points(inv_transform(res.T_base_plate), sf.polyline_base_m)
    assert P.shape[0] == sf.n_ok
    assert np.abs(P[:, 1] - yc).max() < 0.5e-3, P[:, 1]
    assert np.abs(P[:, 2] + 0.010).max() < 0.5e-3
    assert sf.straightness_rms_mm < 0.3
    assert len(sf.features) == sf.n_profiles and isinstance(sf.features[0], CrossSectionFeatures)
    d = sf.to_dict()
    assert d["n_ok"] == sf.n_ok and len(d["polyline_base_m"]) == sf.n_ok and "root_gap_mm" in d["medians"]
    # 상부 중심 폴리라인은 루트 중심보다 베벨 쪽(−y)으로 bevel_run/2 이동, 높이는 판 상면
    sf_top = extract_seam_features_along_scan(res, spec, stride=4, centre="top")
    Pt = transform_points(inv_transform(res.T_base_plate), sf_top.polyline_base_m)
    assert abs(np.median(Pt[:, 1]) - (yc - 0.5 * spec.bevel_run_mm * 1e-3)) < 0.5e-3
    assert np.abs(Pt[:, 2]).max() < 0.5e-3
    with pytest.raises(ValueError):
        extract_seam_features_along_scan(res, spec, centre="middle")
    # 프로세스 병렬 (2026-10-03 at11: 서버 안 순차 ~100 s): 프로파일마다 독립·시드 고정이라 결과가 순차와 똑같다
    from scan_core import seam_features as SF
    old_min = SF.PARALLEL_MIN_PROFILES
    SF.PARALLEL_MIN_PROFILES = 8
    try:
        sf2 = extract_seam_features_along_scan(res, spec, stride=2, centre="root", keep_features=True, workers=2)
    finally:
        SF.PARALLEL_MIN_PROFILES = old_min
        SF.shutdown_feature_pool()
    assert sf2.n_ok == sf.n_ok and np.array_equal(sf2.polyline_base_m, sf.polyline_base_m)
    for k in ("root_centre_x_mm", "root_gap_mm", "depth_mm", "bevel_angle_deg", "ok"):
        assert np.array_equal(getattr(sf2, k), getattr(sf, k), equal_nan=True), k
    js = lambda fs: [json.dumps(f.to_dict(), sort_keys=True, default=str) for f in fs]    # NaN 도 같은 글자로 비교  # noqa: E731
    assert js(sf2.features) == js(sf.features)


def test_points_above_the_plates_flag_a_foreign_object():
    """판 상면보다 높은 점 = 레이저 선 위의 이물 (2026-10-05 at21: 센서 앞에 늘어진 케이블, 판 위 ~60 mm). 빈 그루브·단차 1.5 mm 판은 0."""
    from scan_core.seam_features import points_above_plates
    for spec in (GrooveSpec(bevel_side="left"), GrooveSpec(bevel_side="right", hi_lo_mm=1.5)):
        x, z = _profile(spec, 6.0)
        f = extract_cross_section_features(x, z, spec)
        assert points_above_plates(x, z, f)["n"] == 0
    spec = GrooveSpec(bevel_side="left")
    x, z = _profile(spec, 6.0)
    z = np.array(z, dtype=float, copy=True)
    plate_z = float(np.nanmedian(z))
    cable = (x > 30.0) & (x < 36.0)                       # 폭 6 mm, 판 위 60 mm (센서에 더 가까움)
    z[cable] = plate_z - 60.0
    z[(x > 36.0) & (x < 42.0)] = np.nan                   # 그림자
    f = extract_cross_section_features(x, z, spec)
    obj = points_above_plates(x, z, f)
    assert obj["n"] >= 10 and obj["max_height_mm"] == pytest.approx(60.0, abs=5.0)   # 판 6° 롤: 중앙값 z 기준으로 넣은 높이 ±3.5 mm
    assert 29.0 < obj["x_min_mm"] < obj["x_max_mm"] < 37.0
