import numpy as np
import pytest

from scan_core.clock_sync import CounterClock, PoseBuffer, TransformBuffer, counter_gaps, dts_histogram, ts_integrity_flags
from scan_core.frames import transform_from_xyz_rpy


def _synthetic_stream(n=3000, fps=100.0, drift_ppm=50.0, lat=0.005, jitter=0.0003, seed=0):
    rng = np.random.default_rng(seed)
    T = (1.0 / fps) * (1 + drift_ppm * 1e-6)
    k = np.arange(n)
    t_acq = 1000.0 + T * k
    t_rx = t_acq + lat + rng.exponential(jitter, n)
    return k, t_acq, t_rx, T


@pytest.mark.parametrize("seed", range(5))
def test_counter_clock_stamp_error_below_0p2ms(seed):
    k, t_acq, t_rx, T = _synthetic_stream(seed=seed)
    clk = CounterClock(fps_nominal=100.0, min_frames=2500)
    clk.push_many(k, t_rx)
    assert clk.ready() and clk.fit()
    assert clk.locked
    # dt = 고정 지연을 알면 진짜 취득시각 복원 (하한 포락선 편향 = p5(지터) ≈ 15 us)
    est = clk.stamp(k, dt_s=0.005)
    err = est - t_acq
    assert np.max(np.abs(err)) < 0.2e-3
    assert abs(clk.slope_ppm() - 50.0) < 5.0
    st = clk.residual_stats()
    assert st["locked"] and st["p5_ms"] == pytest.approx(0.0, abs=1e-9)
    assert abs(clk.fps_measured() - 100.0 / (1 + 50e-6)) < 1e-3


def test_counter_clock_requires_span_or_frames():
    k, t_acq, t_rx, T = _synthetic_stream(n=500)
    clk = CounterClock(fps_nominal=100.0, min_frames=2500, min_span_s=25.0)
    clk.push_many(k, t_rx)
    assert not clk.ready() and not clk.fit()
    with pytest.raises(RuntimeError):
        clk.stamp(0)
    assert clk.fit(force=True) and clk.locked
    p = clk.provisional_stamp(10)
    assert abs(p - (t_acq[10] + 0.005)) < 1e-3


def test_warmup_mode_auto_locks():
    k, t_acq, t_rx, T = _synthetic_stream(n=200)
    clk = CounterClock(fps_nominal=100.0, min_frames=150, min_span_s=1e9, mode="warmup_fixed")
    for kk, tt in zip(k, t_rx):
        clk.push(kk, tt)
    assert clk.locked and clk.fit_n == 150


def test_ts_integrity_flags_and_gaps():
    k = np.array([0, 1, 2, 3, 5, 6])
    ts = np.array([0, 10, 20, 30, 50, 63])
    f = ts_integrity_flags(ts, k, fps=100.0, tol_ms=1.5)
    assert f.tolist() == [False, False, False, False, False, True]
    assert counter_gaps(k) == 1
    assert dts_histogram([0, 10, 20, 31, 41]) == {10: 3, 11: 1}


def test_pose_buffer_interpolates_and_no_extrapolation():
    buf = PoseBuffer(fk=lambda q: transform_from_xyz_rpy(q[:3], (0, 0, 0)))
    assert buf.push(1.0, [0, 0, 0])
    assert not buf.push(1.005, [0, 0, 0])          # change-detection: 같은 스냅샷 재수신
    assert buf.push(1.01, [0.1, 0, 0])
    assert not buf.push(1.005, [0.2, 0, 0])        # 단조 위반
    q = buf.q_at(1.005)
    assert np.allclose(q, [0.05, 0, 0])
    assert buf.q_at(0.5) is None and buf.q_at(2.0) is None
    assert np.allclose(buf.T_at(1.005)[:3, 3], [0.05, 0, 0])
    # 정지 상태(25 ms 이상 변화 없음)의 중복 샘플은 채택되어 보간 범위를 연장한다
    assert not buf.push(1.02, [0.1, 0, 0])
    assert buf.push(1.04, [0.1, 0, 0])
    assert np.allclose(buf.q_at(1.03), [0.1, 0, 0])


def test_transform_buffer_slerp_and_many():
    buf = TransformBuffer()
    buf.push(0.0, transform_from_xyz_rpy((0, 0, 0), (0, 0, 0)))
    buf.push(1.0, transform_from_xyz_rpy((1, 0, 0), (0, 0, np.pi / 2)))
    T = buf.T_at(0.5)
    assert np.allclose(T[:3, 3], [0.5, 0, 0])
    assert np.allclose(T[:3, :3], transform_from_xyz_rpy((0, 0, 0), (0, 0, np.pi / 4))[:3, :3])
    Ts, uns = buf.T_at_many([-0.1, 0.25, 1.0, 1.5])
    assert uns.tolist() == [True, False, False, True]
    assert np.allclose(Ts[1][:3, 3], [0.25, 0, 0])


def test_stall_model_shifts_each_segment_and_flags_only_the_stall():
    from scan_core.clock_sync import stall_model
    k = np.arange(1000, 1400)
    t_acq = 0.01 * (k - k[0])
    t_acq[90:] += 0.028                                             # 센서가 28 ms 촬영을 멈춤
    ts = np.floor(t_acq * 1000.0 + 0.3).astype(int) + 123456
    ts[0] += 5                                                      # 실기 첫 프레임 어긋남 (−4…−5 ms 뒤 구간)
    sm = stall_model(ts, k, 100.0, tol_ms=8.0)
    assert [st["index"] for st in sm.stalls] == [1, 90]
    assert np.flatnonzero(sm.flags).tolist() == [90]
    t_model = 0.01 * (sm.k_eff - sm.k_eff[1])
    assert np.max(np.abs(t_model[1:] - (t_acq[1:] - t_acq[1]))) < 1.1e-3      # 1 ms 양자화 이내
    assert sm.to_dict()["n_stalls"] == 2 and abs(sm.stalls[1]["ms"] - 28.0) <= 1.0
    clean = stall_model(np.floor(0.01 * (k - k[0]) * 1000.0).astype(int), k, 100.0)
    assert clean.stalls == [] and np.allclose(clean.k_eff, k)


def test_envelope_slope_ignores_a_receive_backlog_that_biases_ols():
    from scan_core.clock_sync import CounterClock
    rng = np.random.default_rng(4)
    k = np.arange(1500)
    T_true = 0.01 * (1 + 25e-6)
    t_acq = 50.0 + T_true * k
    t_rx = t_acq + 0.004 + rng.exponential(0.003, k.size)
    late = (t_acq >= 50.6) & (t_acq < 50.83)
    t_rx[late] = np.maximum(t_rx[late], 50.834)                      # 0.23 s 수신 막힘 → 한꺼번에 도착
    c = CounterClock(fps_nominal=100.0)
    c.push_many(k, t_rx)
    c.fit(force=True)
    rs = c.residual_stats()
    assert rs["slope_method"] == "envelope" and abs(c.slope_ppm() - 25.0) < 40.0
    assert abs(rs["slope_ppm_ols"] - 25.0) > 200.0
    ols = CounterClock(fps_nominal=100.0, slope_method="ols")
    ols.push_many(k, t_rx)
    ols.fit(force=True)
    assert abs(ols.slope_ppm() - rs["slope_ppm_ols"]) < 1e-6
