"""용접 경로 도달성 확인의 조밀 IK 추종 (moveit_adapter.dense_tracks) 은 실행을 막지 않는다 (2026-10-09): 프로세스 풀 규칙 (scan_core.proc_pool) —
죽은 자식 → 그 줄은 여기서 같은 계산, 한계 (RESULT_TIMEOUT_S, 전체에 한 번) 안에 답이 없는 멈춘 풀 → 그 줄은 '확인하지 않음' (그 풀의 자식은
끝나고 다음에 쓸 때 새 풀). 예전: 한계 없는 f.result() — 결과를 쓰다 죽은 자식·멈춘 자식이면 서버가 영원히 기다렸다. ROS 그래프 없이 돈다."""
import os
import time

import pool_probe
from hongik_scan_ros.moveit_adapter import dense_tracks
from scan_core import proc_pool as PP
from scan_core import seam_features as SF


def _until(cond, limit_s: float) -> bool:
    t_end = time.monotonic() + limit_s
    while not cond() and time.monotonic() < t_end:
        time.sleep(0.01)
    return bool(cond())


def test_dense_tracks_computes_answers_recomputes_dead_children_and_skips_a_stalled_pool(monkeypatch):
    SF.shutdown_feature_pool()
    try:
        tracks, unchecked, note = dense_tracks([(2, 10), None, (3, 4)], fn=pow, workers=2)
        assert tracks == [1024, None, 81] and unchecked == {} and note is None
        # 풀의 자식이 죽는다 → 그 줄은 여기서 같은 계산 (OwnedPool 감시: 그 풀의 Future 는 바로 BrokenProcessPool)
        tracks, unchecked, _ = dense_tracks([(os.getpid(), 5), (os.getpid(), 6)], fn=pool_probe.die_in_child, workers=2)
        assert tracks == [10, 12] and unchecked == {}
        # 살았지만 답이 없는 풀 → 한계 뒤 '확인하지 않음', 이 호출은 한계보다 오래 막히지 않고 그 풀의 자식은 끝난다
        monkeypatch.setattr(PP, "RESULT_TIMEOUT_S", 2.0)                 # 시험 시간만 (한계의 값이 아니라 한계 뒤를 본다)
        pool = SF.worker_pool(2)
        assert [f.result(60.0) for f in [pool.submit(int) for _ in range(2)]] == [0, 0]
        kids = pool.children()
        t0 = time.monotonic()
        tracks, unchecked, _ = dense_tracks([(3600.0,), None, (2, 3)], fn=pool_probe.sleep_or_pow, workers=2)
        dt = time.monotonic() - t0
        assert list(unchecked) == [0] and "not checked" in unchecked[0] and tracks == [None, None, 8], (tracks, unchecked)
        assert dt < PP.RESULT_TIMEOUT_S + 30.0, dt
        assert _until(lambda: not any(k.is_alive() for k in kids), 30.0) and SF._POOL is not pool
    finally:
        SF.shutdown_feature_pool()
