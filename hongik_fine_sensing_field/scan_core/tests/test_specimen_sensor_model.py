"""GrooveSpec (시편 모델) · GrooveScene (합성 프리미티브) · sensor_model (FOV/스탠드오프)."""
import math

import numpy as np
import pytest

from scan_core.frames import make_transform, rot_x
from scan_core.path_gen import plate_frame
from scan_core.sensor_model import (VC_NANO_3DZ_LARGE, SensorModel, choose_standoff, dx_mm, fov_x_mm,
                                    required_fov_for_groove_mm)
from scan_core.specimen import DEFAULT_QC_TARGETS, GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, simulate_profile
from scan_core.validity import dX_mm


# --------------------------------------------------------------------------------------
# GrooveSpec
# --------------------------------------------------------------------------------------
def test_groove_spec_defaults_match_user_specimen():
    s = GrooveSpec()
    assert s.kind == "single_bevel" and s.thickness_mm == 10.0 and s.bevel_angle_deg == 30.0
    assert s.root_gap_mm == 5.0 and s.root_face_mm == 0.0 and s.backing and s.backing_thickness_mm == 10.0
    assert s.weld_length_mm == 380.0 and s.plate_width_mm == 300.0 and s.hi_lo_mm == 0.0
    # 상부 개구 = gap + t·tan30 = 5 + 5.774 ≈ 10.77, 깊이 = t
    assert s.bevel_run_mm == pytest.approx(10.0 * math.tan(math.radians(30.0)))
    assert s.top_opening_mm == pytest.approx(10.7735, abs=1e-3)
    assert s.groove_depth_mm == 10.0
    assert s.top_corners_mm() == (pytest.approx(-2.5 - 5.7735, abs=1e-3), pytest.approx(2.5))
    assert s.root_edges_mm() == (-2.5, 2.5)
    assert "top opening 10.77" in s.summary()


def test_groove_spec_cross_section_single_bevel_left_and_right():
    s = GrooveSpec(bevel_side="left")
    x = np.array([-20.0, -8.3, -8.27, -5.4, -2.6, -2.5, -2.4, 0.0, 2.4, 2.5, 2.6, 20.0])
    h = s.nominal_cross_section(x)
    assert h[0] == 10.0 and h[1] == 10.0                            # 왼 판 상면 (코너 바깥)
    assert 9.9 < h[2] <= 10.0                                        # 코너 직전
    # 베벨면: 루트 모서리(x=-2.5, h=0) 에서 코너(x=-8.27, h=10) 까지 선형
    assert h[3] == pytest.approx((5.4 - 2.5) / s.bevel_run_mm * 10.0, abs=1e-6)
    assert h[5] == pytest.approx(0.0) and h[4] == pytest.approx(0.1 / s.bevel_run_mm * 10.0, abs=1e-6)
    assert np.all(h[6:9] == 0.0)                                     # 바닥 (뒷댐재 상면)
    assert np.all(h[9:] == 10.0)                                     # 직각 판: x ≥ 2.5 가 판
    # 우측 베벨은 좌우 대칭
    sr = GrooveSpec(bevel_side="right")
    np.testing.assert_allclose(sr.nominal_cross_section(-x), h, atol=1e-9)
    # hi_lo 는 right 판 상면에만
    sh = GrooveSpec(bevel_side="left", hi_lo_mm=1.5)
    hh = sh.nominal_cross_section(x)
    assert hh[0] == 10.0 and hh[-1] == 11.5 and hh[7] == 0.0
    # 루트페이스: 루트 모서리에서 rf 만큼 수직
    srf = GrooveSpec(root_face_mm=2.0)
    assert srf.nominal_cross_section(np.array([-2.5]))[0] == pytest.approx(2.0)
    assert srf.top_opening_mm == pytest.approx(5.0 + 8.0 * math.tan(math.radians(30.0)))
    # 뒷댐재 없음 → 갭 NaN
    assert np.isnan(GrooveSpec(backing=False).nominal_cross_section(np.array([0.0]))[0])
    # V 그루브 / 직각
    sv = GrooveSpec(kind="single_v")
    assert sv.top_opening_mm == pytest.approx(5.0 + 2 * 5.7735, abs=1e-3)
    sq = GrooveSpec(kind="square", bevel_angle_deg=30.0)
    assert sq.bevel_angle_deg == 0.0 and sq.top_opening_mm == 5.0


def test_groove_spec_validation_and_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        GrooveSpec(kind="double_v")
    with pytest.raises(ValueError):
        GrooveSpec(bevel_side="top")
    with pytest.raises(ValueError):
        GrooveSpec(root_face_mm=11.0)
    with pytest.raises(ValueError):
        GrooveSpec(bevel_angle_deg=90.0)
    s = GrooveSpec(bevel_side="right", hi_lo_mm=0.5, notes=["taught 2026-09-30"])
    d = s.to_dict()
    assert d["derived"]["top_opening_mm"] == pytest.approx(s.top_opening_mm)
    s2 = GrooveSpec.from_dict(d)
    assert s2 == s
    p = tmp_path / "spec.yaml"
    s.save_yaml(p)
    assert GrooveSpec.load_yaml(p) == s
    # {specimen: {...}} 래퍼도 허용
    import yaml
    with open(tmp_path / "job.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"job": {}, "specimen": d}, f)
    assert GrooveSpec.load_yaml(tmp_path / "job.yaml") == s
    assert set(DEFAULT_QC_TARGETS) == {"root_gap_mm", "bevel_angle_deg", "groove_depth_mm", "hi_lo_mm"}


# --------------------------------------------------------------------------------------
# GrooveScene (synth primitive)
# --------------------------------------------------------------------------------------
def test_groove_scene_heightmap_and_extent():
    spec = GrooveSpec()
    sc = GrooveScene.from_spec(spec, yc=0.002)
    assert sc.x_range == (-0.19, 0.19) and sc.y_range == (pytest.approx(0.002 - 0.15), pytest.approx(0.002 + 0.15))
    yp = np.array([0.002, 0.002 + 0.0024, 0.002 + 0.0026, 0.002 - 0.005, -0.1, 0.16 + 0.002])
    xp = np.zeros_like(yp)
    h = sc.height(xp, yp)
    assert h[0] == pytest.approx(-0.010) and h[1] == pytest.approx(-0.010)        # 바닥 = 판 상면 − t
    assert h[2] == pytest.approx(0.0)                                              # 직각 판 상면
    assert -0.010 < h[3] < 0.0                                                     # 베벨면
    assert h[4] == pytest.approx(0.0)
    assert np.isnan(h[5])                                                          # 판 밖
    # 그루브가 진행축과 각을 이루면 루트 중심 y 가 x 에 따라 이동
    sc2 = GrooveScene.from_spec(spec, yc=0.0, angle_deg=5.0)
    assert sc2.root_centre_y(0.1) == pytest.approx(0.1 * math.tan(math.radians(5.0)))
    assert sc2.height(np.array([0.1]), np.array([sc2.root_centre_y(0.1)]))[0] == pytest.approx(-0.010)


def test_groove_scene_ray_trace_shows_groove_and_wall_occlusion():
    """센서가 판 위 430 mm 에서 8° 롤(라인축 = 판 y): 그루브가 프로파일에 나타나고 직각 벽 쪽 바닥 일부는 그늘(가림)."""
    spec = GrooveSpec(bevel_side="left")
    sc = GrooveScene(spec=spec)                                             # 무한 판 (롤 8° 에서 FOV 가 ±150 mm 를 넘음)
    T_plate = plate_frame([0, 0, 0], [1, 0, 0], [0, 0, 1])
    R0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)          # X_s = plate y, Z_s = -plate z
    from scan_core.seam_features import ransac_line

    def _depth_below_plate(x, z):
        """판 상면 RANSAC 직선(롤 때문에 z 가 x 에 따라 변함) 대비 더 먼 쪽 잔차 [mm] (+ = 판보다 깊음)."""
        ln, _ = ransac_line(x, -z, thr_mm=0.3, rng=np.random.default_rng(0))
        return -ln.dist(x, -z) / ln.d[0] * ln.d[0]          # 수직 거리 부호: 판 아래(멀리) 가 +

    def _plate_pts(T_bo, x, z):
        """센서 점 → 판 프레임 (y_mm, h_mm) (T_plate = I)."""
        o, X, Z = T_bo[:3, 3], T_bo[:3, 0], T_bo[:3, 2]
        P = o + (x[:, None] * 1e-3) * X + (z[:, None] * 1e-3) * Z
        return P[:, 1] * 1e3, P[:, 2] * 1e3

    for tilt in (0.0, 8.0, -8.0):
        T_bo = make_transform(rot_x(math.radians(tilt)) @ R0, (0.0, 0.0, 0.43))
        x, z = simulate_profile(T_bo, sc, T_plate, SynthConfig(n_points=1408, seed=0))
        assert np.isfinite(z).all()
        dep = _depth_below_plate(x, z)
        deep = dep > 8.0                                    # 바닥 근방 (10 mm 아래)
        assert 25 < deep.sum() < 120                        # 바닥 폭 5 mm / dx 0.13 ≈ 38 열 (+ 베벨 하부)
        # 상부 개구 폭 ≈ 10.8 mm: 판 레벨보다 0.5 mm 이상 깊은 열의 x 범위
        low = dep > 0.5
        width = x[low].max() - x[low].min()
        assert 9.5 < width < 13.0
    # 가림 — 광선은 가상 레이저 원점에서 퍼지는 부채꼴이라 센서 중심에서 60 mm 떨어진 열은 ≈7° 기울어 있다
    # (롤 8° 는 그 열에서 부채꼴 기울기와 상쇄되어 거의 수직 → 롤 대신 횡 오프셋으로 검사). 직각 벽 y = +2.5 (법선 −y):
    #  센서를 +y 60 mm 에 두면 그루브 열의 광선이 −y 로 진행하며 하강 → 벽은 등 뒤, 벽 옆 바닥 띠 ≈ 10·tan7° ≈ 1.2 mm 는 그늘
    #  센서를 −y 60 mm 에 두면 광선이 +y 로 진행 → 벽면이 조명되어 y ≈ 2.5, h ∈ (−10, 0) 인 벽 점이 생긴다
    #  센서가 그루브 바로 위면 그늘도 벽 점도 없다
    prof = {}
    for yo in (0.06, -0.06, 0.0):
        T_bo = make_transform(R0, (0.0, yo, 0.43))
        x, z = simulate_profile(T_bo, sc, T_plate, SynthConfig(n_points=1408, seed=0))
        prof[yo] = _plate_pts(T_bo, x, z)
    y0, h0 = prof[0.0]
    assert np.any((y0 > 1.3) & (y0 < 2.45) & (h0 < -9.5))
    assert not np.any((np.abs(y0 - 2.5) < 0.05) & (h0 < -1.0) & (h0 > -9.0))
    ys, hs = prof[0.06]
    assert not np.any((ys > 1.3) & (ys < 2.45) & (hs < -9.5))          # 그늘
    assert ys[hs < -9.5].max() < 1.6
    assert not np.any((np.abs(ys - 2.5) < 0.05) & (hs < -1.0) & (hs > -9.0))
    yw, hw = prof[-0.06]
    assert ((np.abs(yw - 2.5) < 0.05) & (hw < -1.0) & (hw > -9.0)).sum() >= 5     # 벽 점
    assert np.any((yw > 1.3) & (yw < 2.45) & (hw < -9.5))


# --------------------------------------------------------------------------------------
# sensor_model
# --------------------------------------------------------------------------------------
def test_sensor_model_fov_dx_and_consistency_with_validity():
    m = VC_NANO_3DZ_LARGE
    assert m.n_columns == 1408 and m.z_nominal_mm == (250.0, 525.0) and m.z_calibrated_mm == (232.6, 546.5)
    # 실측 fit 0.37 z + 23: 425 mm 에서 180.25 (실기 x 범위 180.9 mm)
    assert fov_x_mm(425.0) == pytest.approx(180.25)
    assert fov_x_mm(250.0) == pytest.approx(115.5) and fov_x_mm(525.0) == pytest.approx(217.25)
    # 공칭(데이터시트) 105@250 → 215@525
    assert m.fov_x_nominal_mm(250.0) == pytest.approx(105.0) and m.fov_x_nominal_mm(525.0) == pytest.approx(215.0)
    # dx = fov/1408 은 validity.dX_mm 의 knot 와 일치 (0.082@250, 0.155@525 ±0.001)
    assert dx_mm(250.0) == pytest.approx(dX_mm(250.0), abs=1e-3)
    assert dx_mm(525.0) == pytest.approx(dX_mm(525.0), abs=1e-3)
    assert m.z_for_fov(fov_x_mm(320.0)) == pytest.approx(320.0)
    assert m.in_nominal_range(320.0) and not m.in_nominal_range(240.0) and m.in_calibrated_range(240.0)
    d = m.to_dict()
    assert d["name"].startswith("VC nano 3D-Z") and d["z_nominal_mm"] == [250.0, 525.0]


def test_choose_standoff_for_groove():
    spec = GrooveSpec()
    req = required_fov_for_groove_mm(spec.top_opening_mm, plate_margin_mm=60.0)
    assert req == pytest.approx(130.77, abs=0.01)
    ch = choose_standoff(req, margin_mm=0.0, preferred_mm=320.0)
    assert ch.z_min_for_fov_mm == pytest.approx((130.77 - 23.0) / 0.37, abs=0.1)     # ≈ 291 mm
    assert 290.0 < ch.z_min_for_fov_mm < 292.0
    assert ch.recommended_mm == 320.0 and ch.fov_at_recommended_mm == pytest.approx(141.4)
    assert ch.feasible and ch.notes == []
    assert "recommended standoff 320" in ch.summary()
    # 선호값이 너무 가까우면 올린다
    ch2 = choose_standoff(req, margin_mm=10.0, preferred_mm=260.0)
    assert ch2.recommended_mm == pytest.approx(ch2.z_min_for_fov_mm) and ch2.recommended_mm > 260.0
    assert any("raised" in n for n in ch2.notes)
    # 범위 상단으로도 못 채우면 feasible=False, 상단으로 클램프
    ch3 = choose_standoff(400.0)
    assert not ch3.feasible and ch3.recommended_mm == 525.0
    with pytest.raises(ValueError):
        choose_standoff(0.0)
    # 사용자 정의 모델
    sm = SensorModel(fov_slope=0.4, fov_intercept=5.0)
    assert sm.fov_x_mm(250.0) == pytest.approx(105.0)
