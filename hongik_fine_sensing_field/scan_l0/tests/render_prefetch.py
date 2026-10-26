"""렌더 선계산 (시험 전용, 시뮬레이션 시계 2단계) — 시뮬레이션 시계에서 바닥이 되는 직렬 렌더를 없앤다.

사실 (명세 §7):
* 가짜 센서 프레임은 (촬영 순간의 자세, seed, i) 의 순수 함수다 (groove_world.GrooveWorld.frame / clean_frame / render).
* 가상 시계에서 팔 틱 시각 (t0_ns + k·8 ms) 과 촬영 시각 (c0_ns + j·period_ns) 은 정수 ns 로 정확히 정해지고, 같은 시각이면 주 스레드
  (순번 0, 팔 재생) 의 틱이 먼저다 — 촬영 j 가 보는 자세 = 틱 k = floor((c_j − t0)/8 ms) 의 자세 (재생이 끝난 뒤는 Q[-1]).

흐름: 이동을 시작할 때 가상 팔이 on_trajectory(t0_ns, T, Q, period) 를, 스트림을 시작·멈출 때 가짜 센서가 on_stream 을 알린다. 예측기는
촬영마다 틱의 자세를 가상 팔과 **같은 함수** (sim_arm.playback_t / playback_q, arm.fk) 로 계산해 (비트까지 같은 자세) 렌더를 spawn 프로세스
풀에 맡긴다 (작업자 수는 속도 손잡이일 뿐 — SCAN_RENDER_WORKERS, 0 = 끔). 결과는 **정확한 자세 바이트**를 키로 둔다 (_pending).

clean_frame 의 조회 순서 (groove_world): ① 기존 반올림 키 캐시 (오늘과 같음) → ② 없으면 이 예측기의 정확한 자세 결과 → ③ 둘 다 없으면
동기 렌더. 그래서 '키당 처음 본 자세로 렌더' 의미와 n_rendered 가 그대로다. 제출마다 세계 렌더 인자의 지문을 함께 두고 쓸 때 지금 세계의
지문과 다르면 버린다 (시험이 세계를 바꾼 경우). 세계의 render 를 인스턴스에서 바꾼 시험 (3G 장착 오차 모형) 은 선계산하지 않는다.
계산 중인 결과는 스트림 스레드가 기다린다 (R — 그동안 시뮬 시간은 멈춘다). 예측이 틀리면 (정지·래치·중단) 동기 렌더 — 어느 경로든 내용은 같다
(E7: 켬/끔 바이트 동일).
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import pickle
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

_POOL: Optional[ProcessPoolExecutor] = None
_POOL_N = 0
_POOL_LOCK = threading.Lock()
_WORKER_WORLDS: dict = {}                 # 작업 프로세스: 지문 → 렌더 전용 세계 복제


def n_workers() -> int:
    try:
        return max(0, int(os.environ.get("SCAN_RENDER_WORKERS", "4")))
    except ValueError:
        return 0


def _pool() -> ProcessPoolExecutor:
    global _POOL, _POOL_N
    with _POOL_LOCK:
        n = n_workers()
        if _POOL is None or _POOL_N != n:
            if _POOL is not None:
                _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"))
            _POOL_N = n
        return _POOL


def _render_job(fp: str, blob: bytes, Ts: list) -> list:
    """작업 프로세스: 렌더 전용 세계 복제 (지문별) 로 광학 자세 목록을 렌더한다."""
    w = _WORKER_WORLDS.get(fp)
    if w is None:
        w = pickle.loads(blob)
        if len(_WORKER_WORLDS) > 8:
            _WORKER_WORLDS.clear()
        _WORKER_WORLDS[fp] = w
    return [w.render(np.asarray(T, dtype=float)) for T in Ts]


def world_render_blob(world) -> tuple:
    """세계의 렌더 인자 (자세 출처·잠금·캐시 제외) → (지문, 피클)."""
    from groove_world import GrooveWorld
    d = dict(world.__dict__)
    for k in ("pose_fn", "_lock", "_cache", "_prefetch", "last_T_base_flange", "n_frames", "n_rendered"):
        d.pop(k, None)
    clone = GrooveWorld.__new__(GrooveWorld)
    clone.__dict__.update(d)
    clone._cache = {}
    clone._lock = None
    clone._prefetch = None
    blob = pickle.dumps(clone, protocol=4)
    return hashlib.sha1(blob).hexdigest(), blob


class RenderPrefetcher:
    """가상 팔의 궤적 + 가짜 센서의 촬영 격자 → 촬영 자세 예측 → spawn 풀 렌더 (모듈 문서)."""

    CHUNK = 1                            # 작업 하나 = 자세 하나 (먼저 필요한 프레임이 묶음을 기다리지 않게)

    def __init__(self, world, arm):
        self.world = world
        self.arm = arm
        self._lock = threading.Lock()
        self._pending: dict = {}             # 정확한 T_base_flange 바이트 → (future, 인덱스, 지문)
        self._stream: Optional[tuple] = None  # (c0_ns, period_ns)
        self.stats = {"trajectories": 0, "predicted": 0, "jobs": 0, "hits": 0, "misses": 0, "stale": 0, "errors": 0}

    @property
    def enabled(self) -> bool:
        return n_workers() > 0 and "render" not in self.world.__dict__

    # ---- 알림 ----
    def on_stream(self, event: str, c0_ns: Optional[int] = None, period_ns: Optional[int] = None) -> None:
        with self._lock:
            self._stream = (int(c0_ns), int(period_ns)) if event == "start" else None

    def on_trajectory(self, t0_ns: int, T, Q, period_s: float) -> None:
        """가상 팔 재생 시작 (주 스레드, 시뮬 시간 정지 중): 이 이동 동안의 촬영 자세를 예측해 렌더를 맡긴다."""
        from sim_arm import playback_q, playback_t
        with self._lock:
            st = self._stream
            self._pending.clear()                      # 앞 이동의 예측은 버린다
        if st is None or not self.enabled:
            return
        self.stats["trajectories"] += 1
        c0, per = st
        T = np.asarray(T, dtype=float)
        Q = np.asarray(Q, dtype=float)
        tick = int(round(float(period_s) * 1e9))
        scale = float(self.arm.cfg.time_scale)
        t_end = int(t0_ns) + int(math.ceil(float(T[-1]) * scale * 1e9)) + tick
        j0 = max(0, -(-(int(t0_ns) - c0) // per))
        fk = self.arm.fk()
        T_fo = self.world.calib.T_flange_opt
        keys_seen = set(self.world._cache.keys())
        poses: list = []                                # (정확한 바이트, T_base_opt)
        seen_exact: set = set()
        c = c0 + j0 * per
        while c <= t_end + per:
            k = (c - int(t0_ns)) // tick
            ks = [k] if (k <= 0 or c - (int(t0_ns) + k * tick) >= 1000) else [k, k - 1]
            for kk in ks:
                tk = int(t0_ns) + kk * tick
                ts = playback_t(t0_ns, tk, scale)
                q = Q[-1] if ts >= T[-1] else playback_q(T, Q, ts)
                Tbf = np.asarray(fk(np.asarray(q, dtype=float).copy()), dtype=float).reshape(4, 4)
                ex = Tbf.tobytes()
                key = tuple(np.round(Tbf[:3, 3], 5).tolist()) + tuple(np.round(Tbf[:3, :3].ravel(), 5).tolist())
                if ex in seen_exact or key in keys_seen:
                    continue
                seen_exact.add(ex)
                keys_seen.add(key)
                poses.append((ex, Tbf @ T_fo))
            c += per
        if not poses:
            return
        fp, blob = world_render_blob(self.world)
        pool = _pool()
        pend = {}
        for i0 in range(0, len(poses), self.CHUNK):
            chunk = poses[i0:i0 + self.CHUNK]
            fut = pool.submit(_render_job, fp, blob, [p[1] for p in chunk])
            self.stats["jobs"] += 1
            for i, (ex, _T) in enumerate(chunk):
                pend[ex] = (fut, i, fp)
        with self._lock:
            self._pending.update(pend)
        self.stats["predicted"] += len(poses)

    # ---- 조회 (clean_frame 의 캐시 미스) ----
    def take(self, T_bf: np.ndarray):
        with self._lock:
            ent = self._pending.pop(np.asarray(T_bf, dtype=float).reshape(4, 4).tobytes(), None)
            streaming = self._stream is not None
        if ent is None:
            if streaming:
                self.stats["misses"] += 1
            return None
        fut, i, fp = ent
        try:
            res = fut.result()
        except Exception:  # noqa: BLE001 - the synchronous render is the fallback
            self.stats["errors"] += 1
            return None
        if fp != world_render_blob(self.world)[0]:
            self.stats["stale"] += 1
            return None
        self.stats["hits"] += 1
        return res[i]

    def close(self) -> None:
        with self._lock:
            for fut, _i, _fp in self._pending.values():
                fut.cancel()
            self._pending.clear()
            self._stream = None
