"""support_plane_ends: 세운 판이 테이블에서 끝나면 끝 = 루트 직선 ∩ 테이블 면 (2026-10-06 g3_07/g3_08)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from scan_core.localize import EndParams, PassFit, support_plane_ends

TABLE_Z = -0.100
# 끝 구역 지연 = 깊이 (루트가 판 앞면 10 mm 뒤) × tan(40° 내려다봄) — 측정 끝이 모자랄 수 있는 길이 (지지면 교점 이동의 바깥 한계)
PR = EndParams(lag_m=0.010 * math.tan(math.radians(40.0)))


def _scene(table_tilt_deg=0.0, n_profiles=120):
    """판 y = 0 (법선 −y, 센서 쪽), 심 = z 축 (루트는 판 앞면 10 mm 뒤), 테이블 z = TABLE_Z (y < 0 쪽), 센서는 40° 내려다봄."""
    t = math.radians(40.0)
    X = np.array([1.0, 0.0, 0.0])
    Z = np.array([0.0, math.cos(t), -math.sin(t)])
    Y = np.cross(Z, X)
    R = np.column_stack([X, Y, Z])
    a = math.radians(table_tilt_deg)
    nt = np.array([0.0, math.sin(a), math.cos(a)])               # 테이블 법선 (기울이면 판과의 각이 줄어든다)
    hits = np.linspace(-0.16, 0.25, n_profiles)                 # 광축이 판 앞면에 닿는 높이
    xs, zs, Ts = [], [], []
    ang = np.radians(np.linspace(-9.0, 9.0, 301))
    for hz in hits:
        o = np.array([0.0, 0.0, hz]) - Z * (0.40 / math.cos(t))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, o
        D = np.outer(np.tan(ang), X) + Z
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        tp = -o[1] / D[:, 1]
        P = o + D * tp[:, None]
        below = (P - np.array([0, 0, TABLE_Z])) @ nt < 0
        tt = ((np.array([0, 0, TABLE_Z]) - o) @ nt) / (D @ nt)
        P[below] = o + D[below] * tt[below, None]
        xs.append((P - o) @ X * 1e3)
        zs.append((P - o) @ Z * 1e3)
        Ts.append(T)
    res = SimpleNamespace(x_mm=np.array(xs), z_mm=np.array(zs), T_base_opt=np.array(Ts), valid=None)
    fit = PassFit(c=np.array([0.0, 0.010, 0.0]), u=np.array([0.0, 0.0, 1.0]), n=np.array([0.0, -1.0, 0.0]))
    fit.diag = {"plate_plane_n": [0.0, -1.0, 0.0], "plate_plane_d": 0.0,
                "ends": {"first": {"observed": True, "occluded": False, "s_m": TABLE_Z + 0.002, "alt_s_m": 0.20 - 0.30},
                         "last": {"observed": True, "occluded": False, "s_m": 0.20, "alt_s_m": TABLE_Z + 0.002 + 0.30}}}
    fit.s_a, fit.s_b = TABLE_Z + 0.002, 0.20
    return res, fit


def test_end_moves_to_the_table_under_a_standing_plate():
    res, fit = _scene()
    support_plane_ends(res, None, fit, params=PR)
    sp = fit.diag["support_planes"]["first"]
    assert sp["applied"] and "end_at_support_first" in fit.flags
    assert sp["height_m"] == pytest.approx(TABLE_Z, abs=2e-4) and sp["shift_m"] == pytest.approx(0.002, abs=3e-4)
    assert fit.s_a == pytest.approx(TABLE_Z, abs=2e-4) and fit.diag["ends"]["first"]["s_m_measured"] == pytest.approx(TABLE_Z + 0.002)
    assert fit.diag["ends"]["last"]["alt_s_m"] == pytest.approx(TABLE_Z + 0.30, abs=2e-4)        # 대안 끝도 새 끝 기준
    assert "last" not in fit.diag["support_planes"] or not fit.diag["support_planes"]["last"]["applied"]


def test_a_surface_more_horizontal_than_vertical_is_a_support():
    """지지면 판정 = 가설 중간점 (scan_core.measure 원칙 2, 2026-10-07): 법선이 중력과 45° 안이면 수평에 더 가깝다 — 25° 기운 받침도
    지지면이고, 판 끝은 루트 직선이 그 면과 만나는 곳 (루트가 판 앞면보다 10 mm 뒤라 테이블 높이보다 10·tan 25° 아래)."""
    res, fit = _scene(table_tilt_deg=25.0)
    support_plane_ends(res, None, fit, params=PR)
    sp = fit.diag["support_planes"]["first"]
    assert sp["applied"] and fit.s_a == pytest.approx(TABLE_Z - 0.010 * math.tan(math.radians(25.0)), abs=3e-4)


def test_a_support_far_beyond_the_end_zone_does_not_move_the_end():
    """교점이 끝 구역 지연 + k·σ 보다 멀면 (측정 끝이 그만큼 모자랄 수 없다) 옮기지 않는다 — 가상 팔 3G 의 40° 경사 받침이 아래 끝을
    34.6 mm 옮겨 길이 415 mm 가 됐다 (2026-10-07)."""
    res, fit = _scene()
    support_plane_ends(res, None, fit, params=EndParams(lag_m=0.001))
    sp = fit.diag["support_planes"]["first"]
    assert not sp["applied"] and sp["shift_m"] == pytest.approx(0.002, abs=3e-4) and fit.s_a == pytest.approx(TABLE_Z + 0.002)


def test_inclined_surface_or_lying_plate_is_not_a_support():
    res, fit = _scene(table_tilt_deg=60.0)                                                    # 수직에 더 가깝다 (중력과 60°)
    support_plane_ends(res, None, fit, params=PR)
    assert not fit.diag["support_planes"]["first"]["applied"] and fit.s_a == pytest.approx(TABLE_Z + 0.002)
    res, fit = _scene()
    fit.diag["plate_plane_n"] = [0.0, 0.0, 1.0]                                              # 누운 판 (1G): 적용 안 함
    support_plane_ends(res, None, fit, params=PR)
    assert "support_planes" not in fit.diag and fit.s_a == pytest.approx(TABLE_Z + 0.002)
