"""시간 모델 (설계 §3.4 / §10.3, C03·C06).

CounterClock : t_acq(k) = a_lo + T_est·k,  k = sensor_result_counter (센서 촬영 멈춤 보정 뒤 k_eff, 아래).
    T_est = 호스트 수신시각 t_rx 의 **하한 포락선** 기울기 (slope_method 'envelope': 0.5 s 창마다 t_rx − T_nom·k 최솟값 → 창 최솟값
            쌍 기울기의 중앙값). 수신 지연은 늦는 쪽 한 방향이고 한 패스 안에서도 고르지 않다 (2026-10-03 MoveIt 실기 at10: 녹화
            초반 계획 계산 동안 수신이 최대 230 ms 밀려 OLS 기울기가 −900…−3700 ppm, 포락선은 −12…+83 ppm). 창이 3 개 미만이면 OLS.
    a_lo  = 잔차의 5-퍼센타일 (하한 포락선).
ts_ms (센서 시계, 1 ms 분해능) 는 **촬영 멈춤** 검출에 쓴다 (stall_model): r = (ts − ts0) − (k − k0)·1000/fps 의 계단 (|Δr| >
    stall_step_ms) 사이 구간마다 r 의 중앙값만큼 k 를 옮긴다 (k_eff = k + r_seg·fps/1000). at10 패스 1: 호스트가 230 ms 수신을 못 하자
    센서가 28 ms 촬영을 멈췄고 (ts 계단 +28 ms, 그 뒤 t_rx 하한도 +28 ms), k 만 쓰는 모델은 그 뒤 모든 프레임을 28 ms (30 mm/s 에서
    0.84 mm) 틀린 시각에 두었다. 무결성 플래그 (ts_flag) 는 ts_tol 을 넘는 멈춤 직후 프레임만 (사건 표시; 데이터는 보정해 쓴다).
PoseBuffer / TransformBuffer : 호스트 수신시각 축에서 관절각/플랜지 포즈 선형 보간, 외삽 금지(None).
    CB systemSTAT.time 은 폴링 20회에서 변하지 않는 것이 실측되었으므로 시간축으로 절대 쓰지 않는다.
LiveCounterClock : 위 대응 (stall_model → k_eff, 하한 포락선) 의 온라인 (인과) 형 — 실시간 센서 가드가 프레임마다 촬영 시각을 추정해 그 시각의
    실측 자세와 짝짓는다 (S2, 2026-10-09; 예전에는 받은 순간의 자세). slerp_transform = TransformBuffer 와 같은 보간 (위치 선형·회전 측지선).
"""
from __future__ import annotations

import bisect
import math
from typing import Callable, Optional, Sequence

import numpy as np

from .frames import make_transform

__all__ = ["CounterClock", "StallModel", "stall_model", "envelope_slope", "ts_integrity_flags", "dts_histogram", "counter_gaps",
           "PoseBuffer", "TransformBuffer", "LiveCounterClock", "slerp_transform"]


class CounterClock:
    """결과 카운터 기반 시각 모델."""

    def __init__(self, fps_nominal: float = 100.0, min_frames: int = 2500, min_span_s: float = 25.0,
                 mode: str = "full_span", envelope_percentile: float = 5.0, slope_method: str = "envelope",
                 envelope_window_s: float = 0.5):
        if mode not in ("full_span", "warmup_fixed"):
            raise ValueError("mode must be 'full_span' or 'warmup_fixed'")
        if slope_method not in ("envelope", "ols"):
            raise ValueError("slope_method must be 'envelope' or 'ols'")
        self.slope_method = slope_method
        self.envelope_window_s = float(envelope_window_s)
        self.slope_ols: Optional[float] = None
        self.slope_used: str = ""
        self.fps_nominal = float(fps_nominal)
        self.T_nom = 1.0 / self.fps_nominal
        self.min_frames = int(min_frames)
        self.min_span_s = float(min_span_s)
        self.mode = mode
        self.envelope_percentile = float(envelope_percentile)
        self._k: list[float] = []
        self._t: list[float] = []
        self.T_est: float = self.T_nom
        self.a_lo: Optional[float] = None
        self.locked: bool = False
        self.fit_span_s: float = 0.0
        self.fit_n: int = 0
        self._resid: Optional[np.ndarray] = None

    # ---- 누적 ----
    def push(self, k: float, t_rx_s: float) -> None:
        self._k.append(float(k))                 # 정수 카운터 또는 촬영 멈춤 보정 k_eff
        self._t.append(float(t_rx_s))
        if self.mode == "warmup_fixed" and not self.locked and self.ready():
            self.fit()

    def push_many(self, ks: Sequence[int], ts: Sequence[float]) -> None:
        for k, t in zip(ks, ts):
            self.push(k, t)

    def reset_samples(self) -> None:
        self._k.clear()
        self._t.clear()

    @property
    def n(self) -> int:
        return len(self._k)

    @property
    def span_s(self) -> float:
        return (self._t[-1] - self._t[0]) if len(self._t) >= 2 else 0.0

    def ready(self) -> bool:
        return self.n >= 2 and (self.n >= self.min_frames or self.span_s >= self.min_span_s)

    # ---- 회귀 ----
    def fit(self, force: bool = False) -> bool:
        """전구간 OLS → T_est, a_lo 고정. 조건 미달이면 False (force=True 면 n≥2 로 강행)."""
        if self.n < 2:
            return False
        if not force and not self.ready():
            return False
        K = np.asarray(self._k, dtype=float)
        T = np.asarray(self._t, dtype=float)
        k0, t0 = K[0], T[0]
        slope, _ = np.polyfit(K - k0, T - t0, 1)
        self.slope_ols = float(slope)
        self.slope_used = "ols"
        if self.slope_method == "envelope":
            win = max(2, int(round(self.envelope_window_s * self.fps_nominal)))
            se = envelope_slope(K, T, self.T_nom, win)
            if se is not None:
                slope = se
                self.slope_used = "envelope"
        self.T_est = float(slope)
        resid = T - self.T_est * K
        self.a_lo = float(np.percentile(resid, self.envelope_percentile))
        self._resid = resid - self.a_lo
        self.fit_span_s = float(T[-1] - T[0])
        self.fit_n = int(len(K))
        self.locked = True
        return True

    def stamp(self, k, dt_s: float = 0.0):
        """t_acq = a_lo + T_est·k − dt. locked 전 호출 금지."""
        if not self.locked or self.a_lo is None:
            raise RuntimeError("CounterClock not locked: call fit() first")
        return self.a_lo + self.T_est * np.asarray(k, dtype=float) - float(dt_s)

    def provisional_stamp(self, k):
        """진단·라이브 표시 전용: 공칭 기울기 + 하한 포락선."""
        if self.n == 0:
            raise RuntimeError("no samples")
        K = np.asarray(self._k, dtype=float)
        T = np.asarray(self._t, dtype=float)
        return float(np.min(T - self.T_nom * K)) + self.T_nom * np.asarray(k, dtype=float)

    def inverse(self, t_s) -> np.ndarray:
        if not self.locked or self.a_lo is None:
            raise RuntimeError("CounterClock not locked")
        return (np.asarray(t_s, dtype=float) - self.a_lo) / self.T_est

    def slope_ppm(self) -> float:
        return (self.T_est / self.T_nom - 1.0) * 1e6

    def fps_measured(self) -> float:
        return 1.0 / self.T_est

    def residual_stats(self) -> dict:
        """잔차(수신 지연 − a_lo) 통계 [ms]."""
        if self._resid is None:
            return {"n": self.n, "locked": False}
        r = self._resid * 1e3
        return {
            "n": int(r.size), "locked": True, "span_s": self.fit_span_s,
            "rms_ms": float(np.sqrt(np.mean(r ** 2))), "mean_ms": float(np.mean(r)),
            "p5_ms": float(np.percentile(r, 5)), "p50_ms": float(np.percentile(r, 50)),
            "p95_ms": float(np.percentile(r, 95)), "max_ms": float(np.max(r)), "min_ms": float(np.min(r)),
            "T_est_s": self.T_est, "a_lo_s": self.a_lo, "slope_ppm": self.slope_ppm(), "slope_method": self.slope_used,
            "slope_ppm_ols": (self.slope_ols / self.T_nom - 1.0) * 1e6 if self.slope_ols is not None else None,
        }

    def to_dict(self) -> dict:
        return {"fps_nominal": self.fps_nominal, "T_est_s": self.T_est, "a_lo_s": self.a_lo, "locked": self.locked,
                "fit_span_s": self.fit_span_s, "fit_n": self.fit_n, "slope_ppm": self.slope_ppm() if self.locked else None,
                "mode": self.mode, "min_frames": self.min_frames, "min_span_s": self.min_span_s, "slope_method": self.slope_used,
                "slope_ppm_ols": (self.slope_ols / self.T_nom - 1.0) * 1e6 if self.slope_ols is not None else None}


def envelope_slope(K, T, T_nom: float, window: int) -> Optional[float]:
    """하한 포락선 기울기 [s/카운트]: window 개씩 t − T_nom·k 의 최솟값 점 → 그 점들 쌍 기울기의 중앙값 (Theil–Sen) + T_nom.
    창이 3 개 미만이면 None (호출자가 OLS 로)."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    n = K.size
    if n < 3 * window:
        return None
    r = (T - T[0]) - T_nom * (K - K[0])
    idx = np.array([i0 + int(np.argmin(r[i0:i0 + window])) for i0 in range(0, n - window + 1, window)])
    kk, rr = K[idx] - K[0], r[idx]
    i, j = np.triu_indices(idx.size, 1)
    dk = kk[j] - kk[i]
    ok = dk > 0
    if not np.any(ok):
        return None
    return float(T_nom + np.median((rr[j] - rr[i])[ok] / dk[ok]))


class StallModel:
    """stall_model 결과: k_eff (촬영 멈춤 보정 카운터), 구간 경계, 멈춤 목록, 무결성 플래그 (ts_tol 을 넘는 멈춤 직후 프레임)."""

    def __init__(self, k_eff: np.ndarray, flags: np.ndarray, stalls: list, offsets_ms: np.ndarray):
        self.k_eff = k_eff
        self.flags = flags
        self.stalls = stalls
        self.offsets_ms = offsets_ms

    @property
    def total_stall_ms(self) -> float:
        return float(sum(st["ms"] for st in self.stalls))

    def to_dict(self) -> dict:
        return {"n_stalls": len(self.stalls), "total_stall_ms": self.total_stall_ms, "stalls": list(self.stalls[:20]),
                "n_flagged": int(np.sum(self.flags))}


TS_QUANTUM_MS = 1.0     # 센서 타임스탬프 단위 [ms] (VC nano 3D-Z 프로토콜: 정수 ms)


def stall_tol_ms(fps: float) -> float:
    """촬영 멈춤 사건 표시 문턱 [ms] = 프레임 주기 − 타임스탬프 양자 둘 (두 시각의 양자화 떨림을 빼고 거의 한 프레임을 잃은 멈춤).
    100 fps → 8 ms (실기 ts 떨림 ≤ 6 ms, 2026-09-30)."""
    return 1000.0 / float(fps) - 2.0 * TS_QUANTUM_MS


def stall_model(ts_ms, k, fps: float, *, step_ms: Optional[float] = None, tol_ms: Optional[float] = None) -> StallModel:
    """센서 시계 ts_ms 로 촬영 멈춤 (트리거 누락·메모리 풀 정지) 을 찾아 k 를 보정한다.

    r = (ts − ts0) − (k − k0)·1000/fps. 양자화 떨림 (두 시각 각각 TS_QUANTUM_MS: |Δr| < 2 양자) 보다 큰 계단 (|Δr| > step_ms, 기본 3 양자 —
    4c 까지와 같은 값; 4d 검토: 2 양자로 바꾸면 기록 패스 3 개에서 구간 경계가 하나씩 늘었다)
    사이를 한 구간으로 보고, 구간마다
    r 의 중앙값만큼 k 를 옮긴다: k_eff = k + r_seg·fps/1000 (첫 프레임의 −4…−5 ms 어긋남도 같은 규칙으로 맞춰진다).
    flags: |계단| > tol_ms (기본 stall_tol_ms(fps)) 인 멈춤 직후 프레임 (사건 표시; 프레임은 보정된 시각으로 그대로 쓴다)."""
    step_ms = 3.0 * TS_QUANTUM_MS if step_ms is None else float(step_ms)
    tol_ms = stall_tol_ms(fps) if tol_ms is None else float(tol_ms)
    ts = np.asarray(ts_ms, dtype=float)
    kk = np.asarray(k, dtype=float)
    n = ts.size
    if n == 0:
        return StallModel(np.zeros(0), np.zeros(0, dtype=bool), [], np.zeros(0))
    Tms = 1000.0 / float(fps)
    r = (ts - ts[0]) - (kk - kk[0]) * Tms
    starts = [0] + [int(i) + 1 for i in np.where(np.abs(np.diff(r)) > float(step_ms))[0]]
    bounds = starts + [n]
    off = np.zeros(n)
    for a, b in zip(bounds[:-1], bounds[1:]):
        off[a:b] = float(np.median(r[a:b]))
    flags = np.zeros(n, dtype=bool)
    stalls = []
    for a in starts[1:]:
        d = float(off[a] - off[a - 1])
        if abs(d) > float(tol_ms):
            flags[a] = True
        stalls.append({"index": int(a), "k": int(kk[a]), "ms": d})
    return StallModel(kk + off / Tms, flags, stalls, off)


class LiveCounterClock:
    """reconstruct 의 시각 대응의 온라인 (인과) 형 — 녹화되는 대로 프레임마다 촬영 시각 [s, 호스트 monotonic] 을 낸다 (실시간 센서 가드, S2).
    녹화 (스트림) 마다 reset() — 카운터·센서 시각의 격자가 녹화마다 새로 시작한다.

      촬영 멈춤 (stall_model 과 같은 규칙): r = (ts − ts0) − (k − k0)·1000/fps 의 이웃 계단 |Δr| > step_ms (기본 3 양자) 가 새 구간을 열고,
                구간 보정 = 그 구간에서 지금까지 받은 r 의 중앙값 → k_eff = k + r_seg·fps/1000
      시각     (CounterClock.provisional_stamp 와 같은 공칭 기울기 + 하한 포락선): 포락선 = 지금까지 t_rx − T_nom·(k_eff − k0) 의 최솟값
      t_acq    = 포락선 + T_nom·(k_eff − k0) − dt_s       (reconstruct: a_lo + T_est·k_eff − dt_s)
    프레임 k 의 값은 k 까지 받은 것으로만 정한다 (뒤에 더 빨리 받은 프레임이 오면 포락선이 내려가 이후 프레임에 쓰인다). 첫 프레임 뒤로는 언제나
    t_acq ≤ t_rx − dt_s (그 프레임 자신이 포락선 후보)."""

    def __init__(self, fps: float, dt_s: float = 0.0, step_ms: Optional[float] = None):
        self.fps = float(fps)
        self.T = 1.0 / self.fps
        self.Tms = 1000.0 / self.fps
        self.dt_s = float(dt_s)
        self.step_ms = 3.0 * TS_QUANTUM_MS if step_ms is None else float(step_ms)
        self.reset()

    def reset(self) -> None:
        self._k0: Optional[int] = None
        self._ts0 = 0.0
        self._r_prev: Optional[float] = None
        self._seg: list = []                       # 지금 구간의 r (정렬 — 중앙값)
        self._lo: Optional[float] = None
        self.n = 0

    def push(self, k: int, ts_ms: float, t_rx_s: float) -> float:
        """프레임 하나 (결과 카운터, 센서 시각 ms, 호스트 수신 시각 s) → 촬영 시각 추정 [s]."""
        k = int(k)
        ts = float(ts_ms)
        if self._k0 is None:
            self._k0, self._ts0 = k, ts
        dk = k - self._k0
        r = (ts - self._ts0) - dk * self.Tms
        if self._r_prev is not None and abs(r - self._r_prev) > self.step_ms:
            self._seg = []
        self._r_prev = r
        bisect.insort(self._seg, r)
        m = len(self._seg)
        med = self._seg[m // 2] if m % 2 else 0.5 * (self._seg[m // 2 - 1] + self._seg[m // 2])
        u = (dk + med / self.Tms) * self.T          # 첫 프레임부터의 장치 시각 = T_nom·(k_eff − k0)
        e = float(t_rx_s) - u
        self._lo = e if self._lo is None else min(self._lo, e)
        self.n += 1
        return self._lo + u - self.dt_s


def ts_integrity_flags(ts_ms, k, fps: float, tol_ms: float = 1.5) -> np.ndarray:
    """|ts_k − ts_0 − (k−k_0)·1000/fps| > tol → True (트리거 누락·메모리 풀 정지 의심 → in_cruise=False)."""
    ts = np.asarray(ts_ms, dtype=float)
    kk = np.asarray(k, dtype=float)
    if ts.size == 0:
        return np.zeros(0, dtype=bool)
    return np.abs((ts - ts[0]) - (kk - kk[0]) * 1000.0 / float(fps)) > float(tol_ms)


def dts_histogram(ts_ms) -> dict:
    """Δts 히스토그램 {ms: count}. 첫 접속 판정: {9,10,11} ms 집중이면 정상 스트림."""
    d = np.diff(np.asarray(ts_ms, dtype=float))
    if d.size == 0:
        return {}
    v, c = np.unique(np.round(d), return_counts=True)
    return {int(a): int(b) for a, b in zip(v, c)}


def counter_gaps(k) -> int:
    """result_counter 불연속으로 잃은 프레임 수 (Σ(Δk − 1) for Δk > 1)."""
    kk = np.asarray(k, dtype=np.int64)
    if kk.size < 2:
        return 0
    d = np.diff(kk)
    return int(np.sum(d[d > 1] - 1))


class PoseBuffer:
    """관절각(또는 임의 벡터) 호스트 시각 보간 버퍼.

    change-detection: 이동 중에는 값이 바뀐 첫 샘플만 채택 (같은 10 ms 스냅샷을 500 Hz 로 여러 번 받으면
    계단 보간이 되므로). 단, 마지막 채택 샘플 후 stationary_keep_s (기본 25 ms = 2.5 틱) 동안 변화가 없으면
    로봇이 정지한 것이므로 중복 샘플도 채택해 보간 유효범위를 연장한다 (정지 중 프레임이 unsynced 가 되지 않게).
    """

    def __init__(self, horizon_s: Optional[float] = None, fk: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 change_detect: bool = True, atol: float = 0.0, stationary_keep_s: float = 0.025):
        self.h = horizon_s
        self.fk = fk
        self.change_detect = change_detect
        self.atol = float(atol)
        self.stationary_keep_s = float(stationary_keep_s)
        self._t: list[float] = []
        self._q: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._t)

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self._t, dtype=float)

    def push(self, t_rx_s: float, q) -> bool:
        q = np.asarray(q, dtype=float).reshape(-1)
        t = float(t_rx_s)
        if self._t and t <= self._t[-1]:
            return False                       # 단조 위반 → 버림
        if (self.change_detect and self._q and np.allclose(q, self._q[-1], atol=self.atol, rtol=0.0)
                and (t - self._t[-1]) < self.stationary_keep_s):
            return False                       # 같은 스냅샷 재수신 (이동 중)
        self._t.append(t)
        self._q.append(q)
        if self.h is not None:
            while self._t and self._t[0] < t - self.h:
                self._t.pop(0)
                self._q.pop(0)
        return True

    def q_at(self, t: float) -> Optional[np.ndarray]:
        """선형 보간. 범위 밖 → None (외삽 금지 → unsynced)."""
        if len(self._t) < 2:
            return None
        k = bisect.bisect_left(self._t, float(t))
        if k == 0:
            return self._q[0].copy() if self._t[0] == t else None
        if k >= len(self._t):
            return None
        t0, t1 = self._t[k - 1], self._t[k]
        w = (t - t0) / (t1 - t0)
        return (1.0 - w) * self._q[k - 1] + w * self._q[k]

    def T_at(self, t: float) -> Optional[np.ndarray]:
        if self.fk is None:
            raise RuntimeError("PoseBuffer has no fk callable")
        q = self.q_at(t)
        return None if q is None else np.asarray(self.fk(q), dtype=float).reshape(4, 4)


class TransformBuffer:
    """플랜지 포즈(4x4) 호스트 시각 보간 버퍼: 위치 선형 · 회전 slerp. 외삽 금지.
    change-detection / stationary_keep_s 의미는 PoseBuffer 와 같다."""

    def __init__(self, horizon_s: Optional[float] = None, change_detect: bool = True,
                 stationary_keep_s: float = 0.025):
        self.h = horizon_s
        self.change_detect = change_detect
        self.stationary_keep_s = float(stationary_keep_s)
        self._t: list[float] = []
        self._T: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._t)

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self._t, dtype=float)

    @property
    def positions(self) -> np.ndarray:
        return np.asarray([T[:3, 3] for T in self._T], dtype=float).reshape(-1, 3)

    @property
    def transforms(self) -> np.ndarray:
        return np.asarray(self._T, dtype=float).reshape(-1, 4, 4)

    def push(self, t_rx_s: float, T_base_flange) -> bool:
        T = np.asarray(T_base_flange, dtype=float).reshape(4, 4)
        t = float(t_rx_s)
        if self._t and t <= self._t[-1]:
            return False
        if (self.change_detect and self._T and np.allclose(T, self._T[-1], atol=0.0, rtol=0.0)
                and (t - self._t[-1]) < self.stationary_keep_s):
            return False
        self._t.append(t)
        self._T.append(T.copy())
        if self.h is not None:
            while self._t and self._t[0] < t - self.h:
                self._t.pop(0)
                self._T.pop(0)
        return True

    def T_at(self, t: float) -> Optional[np.ndarray]:
        if len(self._t) < 2:
            return None
        k = bisect.bisect_left(self._t, float(t))
        if k == 0:
            return self._T[0].copy() if self._t[0] == t else None
        if k >= len(self._t):
            return None
        t0, t1 = self._t[k - 1], self._t[k]
        w = (t - t0) / (t1 - t0)
        return _lerp_T(self._T[k - 1], self._T[k], w)

    def T_at_many(self, ts) -> tuple[np.ndarray, np.ndarray]:
        """여러 시각 일괄 보간 → (T[N,4,4], unsynced[N]). 범위 밖은 최근접 값을 넣고 unsynced=True."""
        from scipy.spatial.transform import Rotation, Slerp

        ts = np.asarray(ts, dtype=float).reshape(-1)
        N = ts.size
        out = np.tile(np.eye(4), (N, 1, 1))
        uns = np.ones(N, dtype=bool)
        if len(self._t) < 2 or N == 0:
            return out, uns
        tb = np.asarray(self._t, dtype=float)
        Tb = np.asarray(self._T, dtype=float)
        inside = (ts >= tb[0]) & (ts <= tb[-1])
        uns = ~inside
        if np.any(inside):
            tq = ts[inside]
            idx = np.clip(np.searchsorted(tb, tq, side="right") - 1, 0, len(tb) - 2)
            w = (tq - tb[idx]) / (tb[idx + 1] - tb[idx])
            p = (1.0 - w)[:, None] * Tb[idx, :3, 3] + w[:, None] * Tb[idx + 1, :3, 3]
            rots = Rotation.from_matrix(Tb[:, :3, :3])
            R = Slerp(tb, rots)(tq).as_matrix()
            Ti = np.tile(np.eye(4), (tq.size, 1, 1))
            Ti[:, :3, :3] = R
            Ti[:, :3, 3] = p
            out[inside] = Ti
        if np.any(uns):
            nearest = np.clip(np.searchsorted(tb, ts[uns]), 0, len(tb) - 1)
            out[uns] = Tb[nearest]
        return out, uns


def slerp_transform(T0, T1, w: float) -> np.ndarray:
    """두 자세 사이 보간 (TransformBuffer.T_at / _lerp_T 와 같은 값: 위치 선형, 회전 측지선 R0·exp(w·log(R0ᵀR1))) — 닫힌 식 (로드리게스)
    이라 scipy Slerp 객체를 만들지 않는다 (실시간 가드가 프레임마다 부른다). 회전 차이가 π 근처면 (축이 수치로 불안정) _lerp_T."""
    T0 = np.asarray(T0, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    w = float(w)
    R0 = T0[:3, :3]
    Rr = R0.T @ T1[:3, :3]
    c = max(-1.0, min(1.0, 0.5 * (float(Rr[0, 0] + Rr[1, 1] + Rr[2, 2]) - 1.0)))
    th = math.acos(c)
    sn = math.sin(th)
    if th > 0.0 and sn < 1e-6:
        return _lerp_T(T0, T1, w)
    out = np.eye(4)
    if th > 0.0:
        k = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) / (2.0 * sn)
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        a = w * th
        out[:3, :3] = R0 @ (np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K))
    else:
        out[:3, :3] = R0
    out[:3, 3] = (1.0 - w) * T0[:3, 3] + w * T1[:3, 3]
    return out


def _lerp_T(T0: np.ndarray, T1: np.ndarray, w: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    key = Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    R = Slerp([0.0, 1.0], key)(float(w)).as_matrix()
    return make_transform(R, (1.0 - w) * T0[:3, 3] + w * T1[:3, 3])
