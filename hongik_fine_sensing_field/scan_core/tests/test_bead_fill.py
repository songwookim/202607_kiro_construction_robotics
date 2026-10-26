"""bead_fill: 다층 멀티패스 충전 상태 정의 하나로 합성 장면·3D 프린트 인서트·진실값 (2026-10-03, KIRO 공정 패스 1–3 충전 + 4 위빙 캡)."""
import dataclasses
import math
import struct

import numpy as np
import pytest

from scan_core.bead_fill import (KIRO_FILL_FRACTIONS, CapSurface, FillLayout, FillSegment, FillSurface, FilledGrooveScene,
                                 groove_area_below_mm2, insert_mesh, insert_ring_mm, kiro_pass_states, level_for_area_fraction,
                                 mesh_check, print_layout, tailor_to_measured, wall_x_mm, write_stl)
from scan_core.frames import make_transform
from scan_core.path_gen import plate_frame
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, simulate_profile

SPEC = GrooveSpec(bevel_side="left")
T_PLATE = plate_frame([0, 0, 0], [1, 0, 0], [0, 0, 1])
R0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)       # X_s = 판 y (가로), Y_s = 판 x (진행), Z_s = −판 z


def test_groove_walls_area_and_fill_levels():
    xl, xr = wall_x_mm(SPEC, np.array([0.0, 10.0]))
    assert xl.tolist() == pytest.approx([-2.5, -2.5 - 10 * math.tan(math.radians(30))]) and xr.tolist() == [2.5, 2.5]
    A = float(groove_area_below_mm2(SPEC, 10.0))
    assert A == pytest.approx(10 * (5.0 + SPEC.top_opening_mm) / 2)            # 사다리꼴 = 깊이 × (갭 + 개구) / 2 = 78.87
    lv = [level_for_area_fraction(SPEC, f) for f in KIRO_FILL_FRACTIONS]
    assert lv == pytest.approx([3.54, 6.53, 9.02], abs=0.01)
    for f, z in zip(KIRO_FILL_FRACTIONS, lv):
        assert float(groove_area_below_mm2(SPEC, z)) == pytest.approx(f * A, rel=1e-9)
    st = kiro_pass_states(SPEC)
    assert [s.pass_index for s in st] == [1, 2, 3, 4] and st[-1].kind == "cap"


def test_fill_surface_meets_the_walls_at_the_toes_and_truth_areas():
    srf = FillSurface(2, 6.5, crown_mm=0.6, peak_frac=0.3, tilt_mm=0.4)
    xl, zl, xr, zr = srf.toes_mm(SPEC, 0.0, 50.0)
    assert (float(zl), float(zr)) == pytest.approx((6.3, 6.7)) and float(xl) == pytest.approx(float(wall_x_mm(SPEC, 6.3)[0]))
    assert float(srf.height_mm(SPEC, xl, 0.0, 50.0)) == pytest.approx(6.3) and float(srf.height_mm(SPEC, xr, 0.0, 50.0)) == pytest.approx(6.7)
    xpk = float(xl + 0.3 * (xr - xl))
    assert float(srf.height_mm(SPEC, xpk, 0.0, 50.0)) == pytest.approx(6.3 + 0.3 * 0.4 + 0.6)
    assert np.isnan(srf.height_mm(SPEC, xr + 0.01, 0.0, 50.0))
    lay = FillLayout(spec=SPEC, segments=[FillSegment(0, 50, srf, "p2"), FillSegment(50, 100, None, "empty"),
                                          FillSegment(100, 150, CapSurface(reinforcement_mm=1.5, overlap_mm=1.5, ripple_amp_mm=0.0), "cap")])
    A = float(groove_area_below_mm2(SPEC, 10.0))
    t_empty, t_fill, t_cap = lay.truth_at(75.0), lay.truth_at(25.0), lay.truth_at(125.0)
    assert t_empty["kind"] == "empty" and t_empty["remaining_area_mm2"] == pytest.approx(A, abs=0.02)
    # 남은 면적 = 전체 − (토우선 아래 면적 + 크라운 면적 (두 조각 포물선 → 2/3 · 높이 · 폭))
    w = float(xr - xl)
    below = float(groove_area_below_mm2(SPEC, 6.3)) + 0.5 * (float(xr) - float(wall_x_mm(SPEC, 6.7)[0])) * 0.0
    assert t_fill["kind"] == "fill" and t_fill["fill_level_mm"] == 6.5 and t_fill["toes_mm"][0][1] == pytest.approx(6.3)
    filled = A - t_fill["remaining_area_mm2"]
    assert below + 2 / 3 * 0.6 * w < filled < below + 2 / 3 * 0.6 * w + 0.4 * w        # 기운 토우선 아래 삼각형 몫 ≤ 0.4·폭
    # 캡: 남은 면적 0, 보강 = 2/3 · 1.5 · (개구 + 2 · 1.5)
    assert t_cap["remaining_area_mm2"] == pytest.approx(0.0, abs=1e-6)
    assert t_cap["reinforcement_area_mm2"] == pytest.approx(2 / 3 * 1.5 * (SPEC.top_opening_mm + 3.0), rel=2e-3)
    assert t_cap["toes_mm"][0][0] == pytest.approx(-2.5 - 5.774 - 1.5, abs=0.01) and t_cap["toes_mm"][1][0] == pytest.approx(4.0, abs=0.01)
    # 저장·복원
    lay2 = FillLayout.from_dict(lay.to_dict())
    x = np.linspace(-12, 8, 400)
    for s in (10.0, 75.0, 120.0):
        assert np.allclose(lay2.height_mm(x, s), lay.height_mm(x, s), equal_nan=True)


def test_filled_scene_equals_the_groove_scene_where_nothing_is_filled_and_adds_spatter():
    lay = FillLayout(spec=SPEC, segments=[FillSegment(100, 200, FillSurface(1, 3.5, crown_mm=0.5), "p1")])
    sc = FilledGrooveScene.from_layout(lay)
    g = GrooveScene.from_spec(SPEC)
    yp = np.linspace(-0.02, 0.02, 801)
    for s_mm, same in ((50.0, True), (150.0, False), (300.0, True)):
        xp = np.full_like(yp, sc.x_range[0] + s_mm * 1e-3)
        assert np.allclose(sc.height(xp, yp), g.height(xp, yp), equal_nan=True) == same
    xp = np.full_like(yp, sc.x_range[0] + 0.150)
    h = sc.height(xp, yp)
    assert float(np.nanmax(h[np.abs(yp) < 0.0025])) == pytest.approx(-10e-3 + 4.0e-3, abs=2e-5)   # 판 상면 0, 크라운 꼭대기 3.5 + 0.5
    at = (np.array([sc.x_range[0] + 0.150]), np.array([0.0]))
    h0 = float(sc.height(*at)[0])
    lay.spatter.append((150.0, 0.0, 0.5))                                     # 반지름 0.5 mm 반구: 꼭대기가 표면 + 0.5
    assert float(sc.height(*at)[0]) == pytest.approx(h0 + 0.5e-3, abs=1e-9)
    n0 = len(lay.spatter)
    lay.add_random_spatter(5, seed=1)
    assert len(lay.spatter) == n0 + 5


@pytest.mark.parametrize("label", ["P1a_convex", "P2b_flat_tilted", "CAPa_weave"])
def test_synthetic_laser_profile_lands_on_the_filled_surface(label):
    lay = print_layout(SPEC)
    seg = next(s for s in lay.segments if s.label.endswith(label))
    sc = FilledGrooveScene.from_layout(lay)
    s_mid = (seg.s0_mm + seg.s1_mm) / 2
    T_bo = make_transform(R0, (sc.x_range[0] + s_mid * 1e-3, 0.0, 0.40))
    x, z = simulate_profile(T_bo, sc, T_PLATE, SynthConfig(n_points=1408, seed=3))
    ok = np.isfinite(x) & np.isfinite(z) & (z > 0)
    P = T_bo[:3, 3] + np.outer(x[ok] * 1e-3, T_bo[:3, 0]) + np.outer(z[ok] * 1e-3, T_bo[:3, 2])
    q = (P - T_PLATE[:3, 3]) @ T_PLATE[:3, :3]
    resid = q[:, 2] - sc.height(q[:, 0], q[:, 1])
    # 수직 벽·캡 가장자리 위 점은 높이맵이 불연속이라 제외 — 98 % 이상이 표면 위 (이분법 허용치 안)
    assert ok.sum() > 1000 and float(np.mean(np.abs(resid) < 2e-5)) > 0.98, float(np.mean(np.abs(resid) < 2e-5))
    # 그루브 안 최고점 = 진실 (충전 표면) — 판 프레임 높이 [mm] 로 비교
    inn = np.abs(q[:, 1]) < 0.002
    tr = lay.truth_at(s_mid)
    top_mm = (float(np.max(q[inn, 2])) + 10e-3) * 1e3
    assert top_mm > (tr["fill_level_mm"] or 10.0) - 0.05


def test_reflection_band_option_moves_a_stretch_of_points_and_is_off_by_default():
    sc = GrooveScene.from_spec(SPEC)
    T_bo = make_transform(R0, (0.0, 0.0, 0.40))
    x0, z0 = simulate_profile(T_bo, sc, T_PLATE, SynthConfig(seed=5, sigma_z_mm=0.02))
    x1, z1 = simulate_profile(T_bo, sc, T_PLATE, SynthConfig(seed=5, sigma_z_mm=0.02, reflection_prob=0.0))
    assert np.array_equal(z0, z1, equal_nan=True) and np.array_equal(x0, x1, equal_nan=True)
    x2, z2 = simulate_profile(T_bo, sc, T_PLATE, SynthConfig(seed=5, reflection_prob=1.0, reflection_depth_mm=(3.0, 3.0),
                                                             reflection_width_cols=(20, 20)))
    xr, zr = simulate_profile(T_bo, sc, T_PLATE, SynthConfig(seed=5))
    d = (z2 - zr)[np.isfinite(z2) & np.isfinite(zr)]
    assert int(np.sum(np.abs(d - 3.0) < 1e-3)) >= 1 and int(np.sum(np.abs(d) > 1e-3)) <= 20


def test_print_inserts_are_closed_meshes_inside_the_groove_with_clearance(tmp_path):
    lay = print_layout(SPEC)
    assert len(lay.segments) == 8 and lay.segments[0].s0_mm == 0 and lay.segments[-1].s1_mm == pytest.approx(380.0)
    assert [s.surface.pass_index for s in lay.segments] == [1, 1, 2, 2, 3, 3, 4, 4]
    c = 0.2
    for seg in lay.segments:
        V, F = insert_mesh(lay, seg, clearance_mm=c)
        chk = mesh_check(V, F)
        assert chk["watertight"] and chk["volume_mm3"] > 0, (seg.label, chk)
        assert V[:, 2].min() == 0.0 and V[:, 2].max() == pytest.approx(seg.length_mm)
        # 부피 ≈ 단면 넓이의 심 방향 적분
        ss = np.linspace(0, seg.length_mm, 41)
        areas = []
        for s in ss:
            r = insert_ring_mm(lay, seg, float(s), clearance_mm=c)
            areas.append(0.5 * float(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])))
            assert areas[-1] > 0                                                # 반시계
            g = r[r[:, 1] <= SPEC.thickness_mm - 1e-9]                         # 그루브 안 부분은 벽에서 clearance 안쪽
            xl, xr = wall_x_mm(SPEC, g[:, 1])
            assert np.all(g[:, 0] >= xl + c - 1e-6) and np.all(g[:, 0] <= xr - c + 1e-6), seg.label
        assert chk["volume_mm3"] == pytest.approx(float(np.trapz(areas, ss)), rel=5e-3)
        if seg.surface.kind == "cap":                                           # 날개가 판 상면에, 그루브 안은 바닥에서 0.3 mm 위
            assert V[:, 1].min() == pytest.approx(0.3)
            r = insert_ring_mm(lay, seg, 0.0)
            wing = r[np.isclose(r[:, 1], SPEC.thickness_mm)]
            assert wing[:, 0].min() < float(wall_x_mm(SPEC, 10.0)[0]) - 1.4 and wing[:, 0].max() > 2.5 + 1.4   # 날개가 판에 1.4 mm 이상
        else:
            assert V[:, 1].min() == 0.0
    # 베벨 (spec left = −X) 쪽 벽은 위로 갈수록 바깥 (세워 출력: S 끝에서 E 를 보면 −X 가 오른쪽)
    r = insert_ring_mm(lay, lay.segments[4], 0.0)
    lo, hi = r[r[:, 1] < 1.0], r[r[:, 1] > 8.0]
    assert hi[:, 0].min() < lo[:, 0].min() - 4.0                               # −X 쪽 (베벨) 은 위로 넓어진다
    assert hi[:, 0].max() == pytest.approx(lo[:, 0].max(), abs=1e-9)          # +X 쪽 (직각 벽) 은 수직
    # STL
    V, F = insert_mesh(lay, lay.segments[0])
    p = write_stl(tmp_path / "a.stl", V, F, name="seg1")
    data = p.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    assert n == F.shape[0] and len(data) == 84 + 50 * n


def test_tailor_to_measured_uses_the_narrowest_groove_of_each_segment():
    bins = [{"from_S_mm": 20.0 * j, "to_S_mm": 20.0 * (j + 1), "root_gap_mm": 5.0 + 0.02 * j, "top_opening_mm": 11.3,
             "depth_mm": 10.0 - 0.01 * j} for j in range(19)]
    lay = tailor_to_measured(print_layout(SPEC), bins)
    s1 = lay.segments[0].spec                                                 # 0–47.5 mm: 구간 0, 1, 2
    assert s1.root_gap_mm == pytest.approx(5.0) and s1.thickness_mm == pytest.approx(9.98)
    assert s1.bevel_angle_deg == pytest.approx(math.degrees(math.atan((11.3 - 5.0) / 9.98)))
    s8 = lay.segments[-1].spec                                                # 332.5–380: 구간 16, 17, 18
    assert s8.root_gap_mm == pytest.approx(5.32) and s8.thickness_mm == pytest.approx(9.82)
    r = insert_ring_mm(lay, lay.segments[0], 0.0, clearance_mm=0.2, chamfer_mm=0.5)
    assert r[:, 1].min() == 0.0 and np.isclose(r[r[:, 1] == 0.0, 0].ptp(), 5.0 - 0.4 - 1.0)   # 바닥 폭 = 갭 − 2·여유 − 2·모따기
    assert all(mesh_check(*insert_mesh(lay, s))["watertight"] for s in lay.segments[::3])
    with pytest.raises(ValueError, match="no fill surface"):
        insert_ring_mm(lay, dataclasses.replace(lay.segments[0], surface=None), 0.0)
