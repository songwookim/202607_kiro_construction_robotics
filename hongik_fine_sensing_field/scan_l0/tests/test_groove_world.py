"""GrooveWorld (합성 그루브 세계): 플랜지 자세 (pose_fn) → 렌더링된 프로파일이 시편 설계치와 맞고(깊이 10, 개구 10.8, 갭 5), yaw 20° 에서
개구가 1/cos20° 배로 보이고, 판 끝 밖은 전부 NaN, 베벨 좌/우가 정확히 거울상이며, 기울인 자세에서도 base 3D 점의 높이는
판 상면(raw z 는 높이가 아님, I2). FakeSensor(frame_fn=world.frame) 스트림이 자세를 따라간다."""
import math
import time

import numpy as np
import pytest

from groove_world import (GrooveWorld, LAB_E_TRUE, LAB_S_TRUE, PoseBox, lab_spec, make_world, true_root_polyline)
from scan_core.seam_features import extract_cross_section_features
from scan_core.types import Calib
from scan_l0.config import load_calib

STANDOFF = 0.34


@pytest.fixture
def calib(calib_path) -> Calib:
    return load_calib(calib_path)


def _world(calib, **kw) -> GrooveWorld:
    return make_world(calib, pose_fn=PoseBox(), **kw)


def _place(world: GrooveWorld, T_opt) -> np.ndarray:
    """그 광학 자세의 플랜지 자세를 세계의 PoseBox 에 놓는다."""
    world.pose_fn.T = world.flange_for_opt(T_opt)
    return world.pose_fn.T


def _look_down(world: GrooveWorld, s=None, **kw):
    """세계의 심 점 s(기본 중앙)를 standoff 0.34 m 에서 내려다보는 자세에 놓고 프레임 0 을 렌더."""
    s = world.L / 2 if s is None else s
    _place(world, world.opt_pose(s, STANDOFF, **kw))
    return world.frame(0)


def test_lab_spec_and_true_root_polyline():
    sl, sr = lab_spec("left"), lab_spec("right")
    assert sl.thickness_mm == 10.0 and sl.bevel_angle_deg == 30.0 and sl.root_gap_mm == 5.0 and sl.backing
    assert sl.weld_length_mm == 380.0 and sl.plate_width_mm == 600.0             # 사용자 2026-10-09: 연구실 1G 시편 전체 폭 600 mm
    assert lab_spec("left", three_g=True).plate_width_mm == 300.0                   # 3G 폭은 아직 확인 안 됨 (3G 템플릿)
    assert sl.bevel_side == "left" and sr.bevel_side == "right" and sl.top_opening_mm == pytest.approx(10.77, abs=0.01)
    P = true_root_polyline(LAB_S_TRUE, LAB_E_TRUE, (0, 0, 1), sl, [0.0, 0.19, 0.38])
    np.testing.assert_allclose(P, [[0.6, -0.19, 0.04], [0.6, 0.0, 0.04], [0.6, 0.19, 0.04]], atol=1e-12)


def test_nominal_pose_renders_design_groove(calib):
    w = _world(calib)
    assert w.L == pytest.approx(0.38) and w.true_plate_top_height() == pytest.approx(0.05)
    np.testing.assert_allclose(w.lateral, [-1.0, 0.0, 0.0], atol=1e-12)        # n×u = −x_base (= 공칭 센서 +X)
    x, z = _look_down(w)
    assert x.shape == (1408,) and z.shape == (1408,) and x.dtype == np.float32 and z.dtype == np.float32
    v = np.isfinite(z)
    assert v.sum() > 1300 and np.isnan(x[~v]).all()                           # 드롭아웃 ~1 %, 무효는 x/z 모두 NaN
    assert abs(float(np.nanmedian(z)) - STANDOFF * 1e3) < 0.5                  # 판 상면이 광학축 거리 340 mm
    f = extract_cross_section_features(x, z, w.spec)
    assert f.ok and f.groove_found and not f.insufficient_points, f.report()
    assert abs(f.depth_mm - 10.0) <= 0.3, f.report()
    assert abs(f.top_opening_mm - w.spec.top_opening_mm) <= 0.3, f.report()
    assert abs(f.root_gap_mm - 5.0) <= 0.3 and abs(f.bevel_angle_deg - 30.0) <= 1.5
    assert f.bevel_side == "left" and f.n_floor_points > 20 and not f.occluded_floor
    assert abs(f.root_centre_x_mm) < 0.5 and abs(f.tilt_deg) < 0.1               # 그루브가 x=0, 판이 수평
    assert f.spec_deltas["bevel_side_match"] is True
    # 노이즈는 프레임마다 다르지만 자세 캐시는 1 번만 렌더
    x1, z1 = w.frame(1)
    assert w.n_rendered == 1 and w.n_frames == 2
    assert float(np.nanmax(np.abs(z - z1))) > 0.0
    x0b, z0b = w.frame(0)
    np.testing.assert_array_equal(z0b, z)                                        # (seed, i) 결정적
    # 진실값 바닥 폴리라인
    P = w.true_root_polyline([0.0, w.L])
    np.testing.assert_allclose(P[:, 2], 0.04)
    np.testing.assert_allclose(P[:, 0], 0.6)


def test_yaw_20_widens_apparent_opening(calib):
    w = _world(calib)
    x, z = _look_down(w, yaw_deg=20.0)
    f = extract_cross_section_features(x, z, w.spec)
    assert f.ok and f.groove_found, f.report()
    ratio = f.top_opening_mm / w.spec.top_opening_mm
    assert ratio == pytest.approx(1.0 / math.cos(math.radians(20.0)), abs=0.03), f.report()
    assert abs(f.depth_mm - 10.0) <= 0.3                                         # 깊이는 그대로


def test_beyond_plate_end_is_all_nan(calib):
    w = _world(calib)
    for s in (w.L + 0.2, -0.2):
        x, z = _look_down(w, s=s)
        assert np.isnan(z).all() and np.isnan(x).all(), s
    x, z = _look_down(w, lateral_m=0.5 * w.plate_len_m + 0.100)                # 판 가장자리에서 가로 100 mm 밖 (시야 반폭보다 멀리)
    assert np.isnan(z).all()
    # 판 끝 1 mm 안쪽은 보이고 1 mm 바깥은 안 보인다 ('판 끝' 이 관측 가능)
    x, z = _look_down(w, s=w.L - 0.001)
    assert np.isfinite(z).sum() > 1300
    x, z = _look_down(w, s=w.L + 0.001)
    assert np.isnan(z).all()


def test_bevel_sides_render_mirrored(calib):
    wl = _world(calib, bevel_side="left", sigma_z_mm=0.0, dropout=0.0)
    wr = make_world(calib, bevel_side="right", sigma_z_mm=0.0, dropout=0.0, pose_fn=wl.pose_fn)
    xl, zl = _look_down(wl)
    xr, zr = wr.frame(0)                                                         # 같은 자세
    np.testing.assert_allclose(xl, -xr[::-1], atol=1e-4)
    np.testing.assert_allclose(zl, zr[::-1], atol=1e-4)
    fl = extract_cross_section_features(xl, zl, wl.spec)
    fr = extract_cross_section_features(xr, zr, wr.spec)
    assert fl.bevel_side == "left" and fr.bevel_side == "right"
    assert fl.left_edge_x_mm == pytest.approx(-fr.right_edge_x_mm, abs=0.2)
    assert fl.right_edge_x_mm == pytest.approx(-fr.left_edge_x_mm, abs=0.2)
    assert fl.depth_mm == pytest.approx(fr.depth_mm, abs=0.1)


@pytest.mark.parametrize("kw", [dict(roll_deg=8.0), dict(pitch_deg=10.0), dict(roll_deg=-6.0, pitch_deg=-5.0)])
def test_tilted_pose_heights_from_base_points_not_raw_z(calib, kw):
    """I2: 기울인 센서에서 raw z 는 판 위에서도 수십 mm 변하지만, base 3D 점의 높이 up·P 는 판 상면 0.050 / 바닥 0.040."""
    w = _world(calib, sigma_z_mm=0.0, dropout=0.0)
    x, z = _look_down(w, **kw)
    v = np.isfinite(z)
    P = w.point_base(x, z)
    h = w.height(P)[v]
    h_top, h_floor = w.true_plate_top_height(), w.true_plate_top_height() - 0.010
    on_top = np.abs(h - h_top) < 1e-4
    on_floor = np.abs(h - h_floor) < 1e-4
    assert on_top.sum() > 1000 and on_floor.sum() > 10                         # 판 상면 / 그루브 바닥
    assert float(h.max()) <= h_top + 1e-4 and float(h.min()) >= h_floor - 1e-4  # 나머지는 베벨면(그 사이)
    np.testing.assert_allclose(w.plate_distance(P[v][on_top]), 0.0, atol=1e-4)
    np.testing.assert_allclose(w.plate_distance(P[v][on_floor]), -0.010, atol=1e-4)
    if "roll_deg" in kw and len(kw) == 1:
        assert float(np.ptp(z[v])) > 10.0                                       # raw z 는 높이가 아니다
        f = extract_cross_section_features(x, z, w.spec)
        assert f.tilt_deg == pytest.approx(kw["roll_deg"], abs=0.2)


def test_frame_follows_the_pose_and_cache_key_rounding(calib):
    w = _world(calib)
    _look_down(w)
    T = w.opt_pose(w.L / 2, STANDOFF, lateral_m=0.005)
    T_bf = _place(w, T)
    x, z = w.frame(0)
    f = extract_cross_section_features(x, z, w.spec)
    assert f.groove_found and f.root_centre_x_mm == pytest.approx(-5.0, abs=0.4)   # 센서를 +X 로 5 mm → 심은 x=−5
    assert w.n_rendered == 2
    T1 = np.array(T_bf, copy=True)
    T1[0, 3] = round(float(T_bf[0, 3]), 5) + 2e-6                                 # 0.002 mm: 캐시 키 (0.01 mm) 안
    w.pose_fn.T = T1
    w.frame(1)
    w.pose_fn.T = np.array(T1, copy=True)
    w.pose_fn.T[0, 3] += 2e-5                                                    # 0.02 mm: 새 렌더
    w.frame(2)
    assert w.n_rendered in (3, 4) and w.n_rendered >= 3
    # 자세가 없는 세계는 render / synthetic_pass 만
    with pytest.raises(ValueError, match="no pose_fn"):
        make_world(calib).frame(0)


def test_fake_sensor_streams_world_frames(calib):
    from vc3dz_driver.client import VC3DZClient
    from vc3dz_driver.fake_sensor import FakeSensor

    w = _world(calib)
    _look_down(w)
    with FakeSensor(frame_fn=w.frame, params={"AutoTriggerFPS": 100.0}) as fs:
        with VC3DZClient("127.0.0.1", fs.port, recv_timeout=2.0) as c:
            c.start()
            recs = []
            t_end = time.monotonic() + 5.0
            while len(recs) < 10 and time.monotonic() < t_end:
                try:
                    recs.append(c.get_frame(timeout=0.5))
                except Exception:
                    pass
            # 스트리밍 중 자세를 판 밖으로 옮기면 이후 프레임은 전부 NaN
            _place(w, w.opt_pose(w.L + 0.2, STANDOFF))
            c.drain()
            time.sleep(0.15)
            c.drain()
            later = c.get_frame(timeout=1.0)
            c.stop()
    assert len(recs) >= 10
    rc = [r.result_counter for r in recs]
    assert rc == list(range(rc[0], rc[0] + len(rc)))
    f = extract_cross_section_features(recs[0].x, recs[0].z, w.spec)
    assert f.groove_found and abs(f.depth_mm - 10.0) <= 0.3
    assert np.isnan(later.z).all()


def test_end_sag_lowers_only_the_given_plate_end_and_the_lab_fixtures_stand_where_the_user_says(calib):
    """EndSagScene (P2 시험 세계): −y 판의 E 끝 마지막 5 mm 가 끝에서 1.0 mm 낮고 9° 기운다 (2026-10-08 실기 E 실패의 판 끝 처짐) —
    다른 판·다른 구간은 그대로. lateral_edge_clamps: 시편 바깥 옆 가장자리 (심에서 ±280–300 mm — 연구실 시편 폭 600 mm 의 반) 의 판 위
    25 mm 상자 — 경로 위·경로 끝 근처에는 없다 (사용자 2026-10-09: 클램프는 시편 가장자리, 심에서 ~300 mm). end_fixture: 일반 '모르는 고정물' — 끝 너머 6 mm, 옆 ±67 mm, 판 위 18 mm."""
    from groove_world import EndSagScene, end_fixture, lateral_edge_clamps
    L = float(np.linalg.norm(np.subtract(LAB_E_TRUE, LAB_S_TRUE)))
    W = float(lab_spec().plate_width_mm) * 1e-3                         # 세계의 기본 폭 = 설계 폭
    sag = lambda sc: EndSagScene(sc, L - 0.005, L, -W / 2, -0.010, 0.0010, 9.0)
    w0 = _world(calib)
    w1 = _world(calib, scene_wrap=sag)
    xs = np.array([L - 0.006, L - 0.005, L - 0.0025, L - 0.0001])
    for y in (-0.05, +0.05):
        h0 = w0.scene.height(xs, np.full(xs.size, y))
        h1 = w1.scene.height(xs, np.full(xs.size, y))
        if y < 0:
            expect = np.maximum(0.0, 0.0010 - (L - xs) * math.tan(math.radians(9.0)))
            expect[0] = 0.0
            np.testing.assert_allclose(h0 - h1, expect, atol=1e-12)
            assert (h0 - h1)[-1] == pytest.approx(0.0010 - 0.0001 * math.tan(math.radians(9.0)))
        else:
            np.testing.assert_allclose(h1, h0, atol=0)                  # +y 판은 그대로
    assert np.isnan(w1.scene.height(np.array([L + 0.002]), np.array([-0.05]))).all()   # 판 끝 너머는 여전히 없음
    cl = lateral_edge_clamps(L, W / 2)
    assert len(cl) == 4 and all(0.280 - 1e-9 <= abs(c.yc) - 0.5 * c.sy and abs(c.yc) + 0.5 * c.sy <= 0.300 + 1e-9 for c in cl)
    assert all(0.0 < c.xc - 0.5 * c.sx and c.xc + 0.5 * c.sx < L for c in cl)                    # 판 위 (끝 너머가 아니다)
    w2 = _world(calib, extras=cl)                                       # 시편 가장자리 = 심에서 300 mm (클램프가 누르는 곳)
    assert w2.plate_len_m == pytest.approx(0.60)
    # 클램프 위 25 mm, 경로 (y = 0) 와 경로 양끝 근처 (옆 ±90 mm 안) 는 판 그대로, 끝 너머는 비어 있다
    h = w2.scene.height(np.array([0.030, 0.030, 0.002, L - 0.002, L + 0.017]), np.array([0.290, 0.0, 0.09, -0.09, 0.09]))
    assert h[0] == pytest.approx(0.025) and abs(h[1]) < 0.02 and abs(h[2]) < 0.02 and abs(h[3]) < 0.02 and np.isnan(h[4])
    fx = end_fixture(L, "S")
    assert len(fx) == 2 and all(c.xc < 0.0 for c in fx) and {round(abs(c.yc), 3) for c in fx} == {0.067}
    w3 = _world(calib, extras=fx)
    hf = w3.scene.height(np.array([-0.011, -0.011, -0.011]), np.array([0.067, -0.067, 0.0]))
    assert hf[0] == pytest.approx(0.018) and hf[1] == pytest.approx(0.018) and np.isnan(hf[2])
    assert len(end_fixture(L, "both")) == 4 and all(c.xc > L for c in end_fixture(L, "E"))
