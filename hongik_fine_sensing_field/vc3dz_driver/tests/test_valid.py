import numpy as np
import pytest

from vc3dz_driver import valid as V


def test_dx_of_z_endpoints():
    assert V.dx_of_z(233.1) == pytest.approx(0.082)
    assert V.dx_of_z(544.0) == pytest.approx(0.155)
    assert 0.082 < float(V.dx_of_z(320.0)) < 0.155
    assert V.dx_of_z(100.0) == pytest.approx(0.082)      # 클램프


def test_valid_mask_rules():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    z = np.array([300.0, 0.0, np.nan, 233.0, 544.0, 233.2, 300.0])
    m = V.valid_mask(x, z)
    assert m.tolist() == [True, False, False, False, False, True, False]
    # z=0 센티널(finite x)은 isfinite 만으로는 걸러지지 않음을 확인
    assert bool(np.isfinite(x[1]) and np.isfinite(z[1]))


def test_valid_mask_on_real_smartshape_sample(sample_frames):
    assert len(sample_frames) == 4
    n_zero = 0
    for f in sample_frames:
        m = V.valid_mask(f.x, f.z)
        zeros = (f.z == 0.0)
        n_zero += int(zeros.sum())
        assert np.isfinite(f.x[zeros]).all()          # 센티널 = finite x + z=0
        assert not m[zeros].any()
        assert m.sum() >= 0.95 * f.x.size              # 평판 실측: 유효점 ≥95 %
        assert f.z[m].min() > 233.1 and f.z[m].max() < 544.0
    assert n_zero == 7                                  # 추출한 4 프레임의 z=0 행 수


def test_real_sample_reflections_removed(sample_frames):
    """실측 광택 평판(2026-07-24 172212): 음의 X 점프(-1~-2 mm) + Z 점프(10~15 mm) 반사 사례.
    필터 후 남는 음의 X 스텝은 설계 문턱 k·dX(z)(≈0.45 mm) 이내여야 하고, 큰 반사 점프는 모두 제거."""
    for f in sample_frames:
        x = f.x.astype(float); z = f.z.astype(float)
        raw_big_neg = int((np.diff(x) < -0.5).sum())
        assert raw_big_neg > 0                                # 원본에는 큰 음의 점프가 있다(반사 사례)
        r = V.reflection_filter(x, z, column_indexed=False)
        xs, zs = x[r.keep], z[r.keep]
        d = np.diff(xs)
        tol = 5.0 * V.dx_of_z(zs[1:])
        assert np.all(d > -tol), f'frame {f.index}: negative x step beyond tolerance after filter'
        assert int((d < -0.5).sum()) == 0
        assert r.reflection_x.sum() > 0
        assert r.stats['reflection_removed_ratio'] < 0.10     # 반사 사례 프레임: 5 % 내외 제거
        assert r.keep.sum() > 0.9 * x.size


def test_reflection_filter_negative_x_jump_and_isolated_z():
    n = 200
    dx = float(V.dx_of_z(320.0))
    x = (np.arange(n) - n / 2) * dx
    z = np.full(n, 320.0)
    # 반사 런: 열 100..104 가 x 뒤로 24 mm 점프 + 다른 z (다중 라인)
    x_bad = x.copy(); z_bad = z.copy()
    x_bad[100:105] -= 24.0
    z_bad[100:105] = 400.0
    # 고립 z 점프(3열, x 는 정상)
    z_bad[150:153] += 10.0
    # 무효점 섞기
    z_bad[10] = 0.0; x_bad[20] = np.nan; z_bad[20] = np.nan
    r = V.reflection_filter(x_bad, z_bad, column_indexed=True, n_columns=n)
    assert r.reflection_x[100:105].all()
    assert not r.reflection_x[[99, 105]].any()
    assert r.isolated_z[150:153].all()
    assert not r.valid[[10, 20]].any()
    assert r.keep.sum() == n - 5 - 3 - 2
    assert r.stats['n_valid'] == n - 2 and r.stats['column_indexed'] == 1.0


def test_reflection_filter_preserves_real_step_edge():
    """판 에지(연속 Z 단차, 런 길이 ≥ min_run)는 보존되어야 한다."""
    n = 300
    dx = float(V.dx_of_z(320.0))
    x = (np.arange(n) - n / 2) * dx
    z = np.where(np.arange(n) < 150, 320.0, 310.0)     # 10 mm 단차, 양쪽 150 열
    r = V.reflection_filter(x, z, column_indexed=True, n_columns=n)
    assert r.keep.all()


def test_clean_profile_sets_nan():
    x = np.array([0.0, 0.1, 0.2, 0.3]); z = np.array([300.0, 0.0, 300.0, 300.0])
    xo, zo, r = V.clean_profile(x, z)
    assert np.isnan(zo[1]) and np.isnan(xo[1]) and np.isfinite(zo[[0, 2, 3]]).all()
