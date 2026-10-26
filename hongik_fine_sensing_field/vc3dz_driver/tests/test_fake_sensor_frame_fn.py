"""FakeSensor.frame_fn: 프레임 내용을 함수(연결별 프레임 인덱스 → (x, z))로 만들어 스트리밍/그랩; 카운터·타임스탬프·
프로토콜은 고정 목록 경로와 같고, frame_fn=None 이면 고정 목록 순환이 그대로다."""
import logging
import threading
import time

import numpy as np
import pytest

from vc3dz_driver.client import VC3DZClient
from vc3dz_driver.fake_sensor import FakeSensor, synthetic_frames

N = 1408


def _collect(c, n, timeout=10.0):
    out = []
    t_end = time.monotonic() + timeout
    while len(out) < n and time.monotonic() < t_end:
        try:
            out.append(c.get_frame(timeout=0.5))
        except Exception:
            pass
    return out


def _make_frame(i: int):
    """열 x 는 고정, z 는 300 + i (프레임 인덱스를 값에 새김), 앞 3 열은 NaN(무효)."""
    x = ((np.arange(N) - N / 2) * 0.1).astype(np.float32)
    z = np.full(N, 300.0 + i, dtype=np.float32)
    x[:3] = np.nan
    z[:3] = np.nan
    return x, z


def test_frame_at_uses_frame_fn_or_fixed_list():
    frames = synthetic_frames(3, 50)
    fs = FakeSensor(frames=frames)
    assert fs.frame_fn is None
    for i in range(7):
        x, z = fs.frame_at(i)
        np.testing.assert_array_equal(x, frames[i % 3][0])
        np.testing.assert_array_equal(z, frames[i % 3][1])
    fs2 = FakeSensor(frames=frames, frame_fn=lambda i: (np.arange(4) * 1.0, np.full(4, 10.0 + i)))
    x, z = fs2.frame_at(5)
    assert x.dtype == np.float32 and z.dtype == np.float32 and x.shape == (4,)
    assert z[0] == 15.0
    assert len(fs2.frames) == 3                       # 고정 목록은 그대로 보관(사용만 안 함)
    with pytest.raises(ValueError):
        FakeSensor(frame_fn=lambda i: (np.zeros(3), np.zeros(4))).frame_at(0)


def test_stream_and_grab_honour_frame_fn_with_counters_intact():
    calls = []
    lock = threading.Lock()

    def fn(i):
        with lock:
            calls.append(i)
        return _make_frame(i)

    with FakeSensor(frame_fn=fn, params={'AutoTriggerFPS': 200.0}) as fs:
        with VC3DZClient('127.0.0.1', fs.port, recv_timeout=2.0) as c:
            c.start()
            recs = _collect(c, 30)
            c.stop()
            time.sleep(0.1)
            n_sent = fs.stats['frames_sent']
            time.sleep(0.1)
            assert fs.stats['frames_sent'] == n_sent          # RESET 후 정지
    assert len(recs) >= 30
    rc = np.array([r.result_counter for r in recs])
    assert rc[0] == 0 and (np.diff(rc) == 1).all()            # 카운터 0부터 1씩
    ts = np.array([r.ts_ms for r in recs], dtype=np.int64)
    assert set(np.diff(ts).tolist()) <= {5}                    # 200 fps → 5 ms
    for r in recs:
        x_exp, z_exp = _make_frame(r.result_counter)
        assert r.n_points == N and r.x.dtype == np.float32
        np.testing.assert_array_equal(r.x, x_exp)              # NaN 포함 그대로 전달
        np.testing.assert_array_equal(r.z, z_exp)
        assert np.isnan(r.x[:3]).all() and float(r.z[10]) == 300.0 + r.result_counter
    # frame_fn 은 연결별 인덱스 0, 1, 2, … 로 순서대로 호출된다
    with lock:
        assert calls[:len(recs)] == list(range(len(recs)))
        assert len(calls) == fs.stats['frames_sent']
    assert c.stats['format_errors'] == 0 and c.stats['checksum_mismatch_data'] == 0


def test_frame_fn_index_continues_across_stop_start_and_mode5_line_counter():
    seen = []
    with FakeSensor(frame_fn=lambda i: (seen.append(i), _make_frame(i))[1],
                    params={'AutoTriggerFPS': 300.0, 'DataMode': 5}, line_counter_variant=True) as fs:
        with VC3DZClient('127.0.0.1', fs.port) as c:
            c.start()
            recs1 = _collect(c, 10)
            c.stop()
            c.drain()
            n1 = fs.stats['frames_sent']
            c.start()
            recs2 = _collect(c, 10)
            c.stop()
    assert len(recs1) >= 10 and len(recs2) >= 10
    assert all(r.data_mode == 5 and r.line_counter == r.result_counter for r in recs1 + recs2)
    # 재개 후 첫 프레임의 인덱스(= z 에 새겨진 값)는 정지 전까지 보낸 수 n1 이상 (인덱스는 리셋되지 않음)
    assert float(recs2[0].z[10]) - 300.0 >= n1
    assert seen == list(range(len(seen)))


def test_fixed_list_fallback_unchanged_when_frame_fn_none():
    frames = synthetic_frames(4, 120)
    with FakeSensor(frames=frames, params={'AutoTriggerFPS': 300.0}) as fs:
        with VC3DZClient('127.0.0.1', fs.port) as c:
            c.start()
            recs = _collect(c, 12)
            c.stop()
    assert len(recs) >= 12
    for r in recs:
        np.testing.assert_array_equal(r.x, frames[r.result_counter % 4][0])
        np.testing.assert_array_equal(r.z, frames[r.result_counter % 4][1])


def test_frame_fn_exception_stops_stream_but_keeps_connection(caplog):
    def bad(i):
        if i >= 3:
            raise RuntimeError('boom')
        return _make_frame(i)

    # ROS 의 launch_testing 플러그인이 로거 클래스를 LaunchLogger(propagate=False) 로 바꾸므로 루트 캡처에 의존하지
    # 않고 이 로거에 caplog 핸들러를 직접 붙인다.
    lg = logging.getLogger('vc3dz_driver.fake_sensor')
    lg.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.ERROR, logger='vc3dz_driver.fake_sensor'):
            with FakeSensor(frame_fn=bad, params={'AutoTriggerFPS': 300.0}) as fs:
                with VC3DZClient('127.0.0.1', fs.port, recv_timeout=2.0) as c:
                    c.start()
                    recs = _collect(c, 3, timeout=3.0)
                    time.sleep(0.2)
                    assert fs.stats['frames_sent'] == 3
                    _, kv = c.read_params()                    # 명령 채널은 살아 있다
                    assert kv['NbrLines'] == -1
                    c.stop()
    finally:
        lg.removeHandler(caplog.handler)
    assert len(recs) == 3 and [r.result_counter for r in recs] == [0, 1, 2]
    assert any('frame_fn failed' in m for m in caplog.messages)


@pytest.mark.parametrize('stall_late_ms', [None, 4.0])
def test_stall_late_ms_reports_a_late_exposure_as_a_ts_step(stall_late_ms):
    """stall_late_ms: 스트림이 예정 촬영 시각보다 늦으면 (여기서는 프레임 5 의 frame_fn 이 주기 10 ms 의 4 배를 씀 → 프레임 6 이
    ≈ 30 ms 늦다) 실기 센서처럼 ts 가 실제 촬영 (frame_fn 호출) 시각만큼 계단으로 뛰고 그 뒤 격자도 밀린다 (따라잡기 몰아 보내기 없음),
    카운터는 1 씩 그대로. 기본 (None) 은 기존 동작: ts 는 이상 격자, 늦은 프레임은 몰아서 보낸다 (프레임 내용과 ts 가 어긋남)."""
    calls = {}

    def fn(i):
        calls[i] = time.perf_counter()
        if i == 5:
            time.sleep(0.040)
        return _make_frame(i)

    with FakeSensor(frame_fn=fn, params={'AutoTriggerFPS': 100.0}, stall_late_ms=stall_late_ms) as fs:
        with VC3DZClient('127.0.0.1', fs.port, recv_timeout=2.0) as c:
            c.start()
            recs = _collect(c, 20)
            c.stop()
    assert len(recs) >= 20
    rc = np.array([r.result_counter for r in recs])
    assert rc[0] == 0 and (np.diff(rc) == 1).all()                 # 카운터는 멈춤과 무관하게 1 씩
    ts = np.array([r.ts_ms for r in recs], dtype=float)
    t_ms = np.array([calls[int(k)] for k in rc]) * 1e3
    i6 = int(np.where(rc == 6)[0][0])
    if stall_late_ms is None:
        assert set(np.diff(ts).tolist()) == {10.0}                   # 이상 격자 (늦어도)
        assert fs.stats['stalls'] == 0
        return
    assert fs.stats['stalls'] >= 1 and fs.stats['stall_ms'] >= 25.0
    # ts = 실제 촬영 시각: 보고하지 않는 늦음 ≤ stall_late_ms + 정수 ms 절삭
    assert np.abs((ts - ts[0]) - (t_ms - t_ms[0])).max() <= stall_late_ms + 1.5
    assert ts[i6] - ts[i6 - 1] >= 10.0 + 25.0                        # 프레임 6 의 계단 (≈ 30 ms 늦음)
    assert np.diff(t_ms)[i6:].min() >= 9.0                           # 계단 뒤 따라잡기 없음 (촬영 간격 ≥ 주기)


def test_latency_ms_holds_each_frame_until_a_fixed_time_after_its_exposure():
    """latency_ms: 촬영 (frame_fn 호출) 에서 송신까지 고정 지연 (실기 센서 처리 지연 — 호스트는 calib dt_s 로 보정). 렌더가 빠르든
    (자세 캐시 적중) 느리든 수신 - 촬영 ≥ 지연 (하한이 렌더 시간에 따라 갈리지 않는다)."""
    calls = {}

    def fn(i):
        calls[i] = time.monotonic()
        if i % 3 == 0:
            time.sleep(0.008)                                         # 느린 렌더 (이동) 와 빠른 렌더 (정지) 가 섞임
        return _make_frame(i)

    with FakeSensor(frame_fn=fn, params={'AutoTriggerFPS': 50.0}, latency_ms=12.0) as fs:
        with VC3DZClient('127.0.0.1', fs.port, recv_timeout=2.0) as c:
            c.start()
            recs = _collect(c, 15)
            c.stop()
    assert len(recs) >= 15 and fs.stats['frames_sent'] >= 15
    lat_ms = np.array([r.t_rx_ns * 1e-6 - calls[r.result_counter] * 1e3 for r in recs])
    slow = np.array([r.result_counter % 3 == 0 for r in recs])
    assert slow.any() and (~slow).any()
    assert lat_ms[slow].min() >= 11.5 and lat_ms[~slow].min() >= 11.5, lat_ms   # 렌더 시간과 무관하게 지연 이상 (기본은 ≈ 0 / 8 ms)


@pytest.mark.parametrize('stall_late_ms', [None, 4.0])
def test_timed_frame_fn_gets_the_capture_instant_its_stamp_and_latency_count_from(stall_late_ms):
    """timed_frame_fn: frame_fn(i, t_capture_ns) gets the frame's capture instant on the sensor's clock — the stream grid's, after a
    reported stall the stalled one — whatever this thread's lateness (frame 5's render takes 4 periods): the stamps' differences are
    the capture instants' differences (to the 1 ms stamp), and each frame leaves the latency after its capture instant (not after
    the call). A frame_fn rendering the pose of that instant shows what the stamp says however late it runs (2026-10-13)."""
    cap, call = {}, {}

    def fn(i, t_capture_ns):
        cap[i] = t_capture_ns
        call[i] = time.monotonic_ns()
        if i == 5:
            time.sleep(0.040)
        return _make_frame(i)

    with FakeSensor(frame_fn=fn, params={'AutoTriggerFPS': 100.0}, stall_late_ms=stall_late_ms, latency_ms=15.0,
                    timed_frame_fn=True) as fs:
        with VC3DZClient('127.0.0.1', fs.port, recv_timeout=2.0) as c:
            c.start()
            recs = _collect(c, 20)
            c.stop()
    assert len(recs) >= 20
    rc = np.array([r.result_counter for r in recs])
    ts = np.array([r.ts_ms for r in recs], dtype=float)
    tc = np.array([cap[int(k)] for k in rc], dtype=float) * 1e-6
    assert np.abs((ts - ts[0]) - (tc - tc[0])).max() <= 1.0                # stamps = capture instants (1 ms stamp)
    late = np.array([call[int(k)] - cap[int(k)] for k in rc]) * 1e-6
    if stall_late_ms is None:
        assert late.max() >= 20.0                                           # frames after frame 5 were rendered late …
    else:
        assert fs.stats['stalls'] >= 1 and late.max() <= stall_late_ms + 2.0  # … or the capture stalled (the stamps step with it)
    lat = np.array([r.t_rx_ns * 1e-6 - cap[r.result_counter] * 1e-6 for r in recs])
    on_time = late + 1.0 < 15.0
    assert on_time.any() and lat[on_time].min() >= 14.5, lat               # … and each frame left the latency after its capture
    log_cap = {int(r_[2]): int(r_[4]) for r_ in fs.capture_log}
    assert all(log_cap[int(k)] == cap[int(k)] for k in rc)                  # the capture log holds the capture instant
