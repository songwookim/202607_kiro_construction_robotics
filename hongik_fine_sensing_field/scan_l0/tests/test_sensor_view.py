"""센서 시야 모형 (scan_l0.sensor_view, 사용자 2026-10-07 '센서 모형에서 유도'): 거친 거리·패스 거리·측면 탐색 간격이 설정 상수 없이
센서 창·광축 교차 거리·FOV·설계 정보·도착 허용에서 나온다."""
import math

import pytest

from scan_core.sensor_model import VC_NANO_3DZ_LARGE
from scan_l0.sensor_view import sensor_view


@pytest.fixture
def v():
    return sensor_view((233.1, 544.0), pose_tol_m=0.0021)


def test_pass_distance_puts_the_groove_mid_depth_at_the_optical_cross(v):
    assert v.cross_m == pytest.approx(VC_NANO_3DZ_LARGE.optical_cross_mm * 1e-3)
    d = 0.010
    plate = v.pass_plate_m(d)
    assert plate == pytest.approx(0.337 - 0.005)
    assert 0.5 * (plate + (plate + d)) == pytest.approx(v.cross_m)                    # 상면·루트 가운데 = 광축 교차


def test_coarse_range_tolerates_the_largest_rough_point_error_in_every_direction(v):
    """거친 거리 = 대략 착점 오차를 시선 방향 (창 양끝) 과 측면 (FOV/2 − 개구/2) 어느 쪽으로든 가장 크게 견디는 거리."""
    d, o = 0.010, 0.0108
    near = 0.2366
    vv = sensor_view((233.1, 544.0), pose_tol_m=0.0021, near_m=near)

    def tol(r, t):
        dt = d / math.cos(math.radians(t))
        return min(r - dt - near - 0.0021, 0.544 - 0.0021 - dt - r, 0.5 * (vv.fov_m(r) - o))
    for t in (0.0, 20.0, 35.0):
        r = vv.coarse_range_m(d, t, o)
        assert vv.coarse_tolerance_m(d, t, o) == pytest.approx(tol(r, t), abs=1e-9)
        assert tol(r, t) >= max(tol(r - 0.002, t), tol(r + 0.002, t)) - 1e-9            # 최댓값 (격자 해상도 안)
        assert vv.coarse_plate_m(d, t, o) == pytest.approx(r * math.cos(math.radians(t)))
    r0 = vv.coarse_range_m(d, 0.0, o)
    assert 0.43 < r0 < 0.46 and vv.coarse_tolerance_m(d, 0.0, o) > 0.08                 # 연구실: ≈ 443 mm, ±88 mm
    assert vv.fov_m(r0) > vv.fov_m(vv.pass_plate_m(d))                                  # 거친 단계가 시야가 더 넓다


def test_search_steps_tile_the_specimen_width_with_views_overlapping_by_the_opening(v):
    r, o = 0.532, 0.0108
    fov = v.fov_m(r)
    steps = v.search_steps_m(r, o, 0.300)
    s = fov - o
    assert steps[:2] == pytest.approx([s, -s])
    k = len(steps) // 2
    assert (k - 1) * s + 0.5 * (fov - o) < 0.300 <= k * s + 0.5 * (fov - o)           # 마지막 시야까지 시편 폭을 덮는다
    assert v.search_steps_m(r, o, 0.05) == []                                         # 첫 시야가 이미 시편 폭을 덮는다
    assert len(v.search_steps_m(0.30, o, 0.60)) > len(v.search_steps_m(0.53, o, 0.60))   # 가까우면 시야가 좁아 스텝이 많다


def test_bad_window_is_refused():
    with pytest.raises(ValueError):
        sensor_view((544.0, 233.1))
    with pytest.raises(ValueError, match="optical cross"):
        sensor_view((350.0, 544.0))
