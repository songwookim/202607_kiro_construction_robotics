"""시간오프셋 dt — 다속도 왕복 히스테리시스 (설계 §5.6.3, C14).

타깃: 진행방향에 30–45° 기울인 직선 단차(≥5 mm) 또는 Ø40 구.
각 패스(방향 ±, 속도 v)에서 에지 위치 s_edge(진행좌표, 판 프레임 x)를 서브프로파일 정밀도로 얻는다:
  프로파일 i (진행좌표 s_i) 에서 단차의 라인방향 위치 y_i 검출 → (s_i, y_i) 직선 피팅 → y = y_ref 에서의 s.
모델:  s⁺ = s0 + v·dt + c/2,  s⁻ = s0 − v·dt − c/2   ⇒  Δ(v) = s⁺ − s⁻ = 2·v·dt + c
3 속도(20/40/80 mm/s) 이상 전체 패스 OLS: 미지수 (s0, dt, c). σ_dt 는 잔차 공분산에서.
합격: dt 산포 ≤ 2 ms. dt 는 음수 허용.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..frames import inv_transform, transform_points

__all__ = ["PassEdge", "HysteresisResult", "fit_dt", "estimate_dt_hysteresis", "step_edge_y_in_profile",
           "inclined_edge_crossing", "edge_position_from_result"]


@dataclass
class PassEdge:
    v_mps: float          # 순항속도 크기 (>0, 실측 권장)
    direction: int        # +1: start→end, −1: 반대
    s_edge_m: float       # 이 패스에서 측정한 에지 진행좌표 [m] (판 프레임 x, 항상 u 기준)
    tag: str = ""


@dataclass
class HysteresisResult:
    dt_s: float
    c_m: float
    sigma_dt_s: float
    sigma_c_m: float
    s0_m: float
    n_passes: int
    n_speeds: int
    deltas: list = field(default_factory=list)      # [(v, Δ)]
    residuals_m: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def dt_ms(self) -> float:
        return self.dt_s * 1e3

    def to_dict(self) -> dict:
        return {"dt_s": self.dt_s, "dt_ms": self.dt_ms, "c_m": self.c_m, "sigma_dt_s": self.sigma_dt_s,
                "sigma_c_m": self.sigma_c_m, "s0_m": self.s0_m, "n_passes": self.n_passes, "n_speeds": self.n_speeds,
                "deltas": [(float(v), float(d)) for v, d in self.deltas]}


def fit_dt(v_mps: Sequence[float], delta_m: Sequence[float]) -> tuple[float, float, float, float]:
    """속도별 Δ(v) 만으로 OLS Δ = 2·v·dt + c → (dt, c, σ_dt, σ_c). 2 속도면 σ = NaN."""
    v = np.asarray(v_mps, dtype=float)
    d = np.asarray(delta_m, dtype=float)
    A = np.column_stack([2.0 * v, np.ones_like(v)])
    sol, res, rank, _ = np.linalg.lstsq(A, d, rcond=None)
    dt, c = float(sol[0]), float(sol[1])
    n = v.size
    if n > 2:
        r = d - A @ sol
        s2 = float(r @ r) / (n - 2)
        cov = s2 * np.linalg.inv(A.T @ A)
        return dt, c, float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    return dt, c, float("nan"), float("nan")


def estimate_dt_hysteresis(passes: Sequence[PassEdge]) -> HysteresisResult:
    """전체 패스 OLS: s = s0 + dir·(v·dt + c/2). ≥2 속도 · 양방향 필요."""
    if len(passes) < 4:
        raise ValueError("need at least 4 passes (2 speeds x 2 directions)")
    v = np.array([abs(p.v_mps) for p in passes], dtype=float)
    d = np.array([1.0 if p.direction >= 0 else -1.0 for p in passes], dtype=float)
    s = np.array([p.s_edge_m for p in passes], dtype=float)
    speeds = np.unique(np.round(v, 6))
    if speeds.size < 2 or not (np.any(d > 0) and np.any(d < 0)):
        raise ValueError("need at least 2 distinct speeds and both directions")
    A = np.column_stack([np.ones_like(v), d * v, 0.5 * d])
    sol, _, _, _ = np.linalg.lstsq(A, s, rcond=None)
    s0, dt, c = (float(x) for x in sol)
    r = s - A @ sol
    n, p = s.size, 3
    if n > p:
        s2 = float(r @ r) / (n - p)
        cov = s2 * np.linalg.pinv(A.T @ A)
        sig_dt, sig_c = float(np.sqrt(max(cov[1, 1], 0.0))), float(np.sqrt(max(cov[2, 2], 0.0)))
    else:
        sig_dt = sig_c = float("nan")
    deltas = []
    for sp in speeds:
        sel = np.isclose(v, sp, atol=1e-6)
        fwd, bwd = s[sel & (d > 0)], s[sel & (d < 0)]
        if fwd.size and bwd.size:
            deltas.append((float(sp), float(fwd.mean() - bwd.mean())))
    return HysteresisResult(dt_s=dt, c_m=c, sigma_dt_s=sig_dt, sigma_c_m=sig_c, s0_m=s0, n_passes=int(n),
                            n_speeds=int(speeds.size), deltas=deltas, residuals_m=r)


# --------------------------------------------------------------------------------------
# 에지 검출
# --------------------------------------------------------------------------------------
def step_edge_y_in_profile(y: np.ndarray, h: np.ndarray, min_step_m: float = 0.002, x: Optional[np.ndarray] = None):
    """프로파일 1개 (라인방향 y, 높이 h[, 진행방향 x]) 에서 단차 위치.
    x 없음 → y_edge (float); x 있음 → (x_edge, y_edge) (중간높이 교차점을 x·y 모두 선형보간). 없으면 NaN."""
    y = np.asarray(y, dtype=float)
    h = np.asarray(h, dtype=float)
    xx = None if x is None else np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(h)
    if xx is not None:
        ok &= np.isfinite(xx)
    nan_out = float("nan") if xx is None else (float("nan"), float("nan"))
    if ok.sum() < 4:
        return nan_out
    y, h = y[ok], h[ok]
    if xx is not None:
        xx = xx[ok]
    order = np.argsort(y)
    y, h = y[order], h[order]
    if xx is not None:
        xx = xx[order]
    dh = np.diff(h)
    j = int(np.argmax(np.abs(dh)))
    if abs(dh[j]) < min_step_m:
        return nan_out
    lo_side = h[max(0, j - 10): j + 1]
    hi_side = h[j + 1: j + 12]
    mid = 0.5 * (np.median(lo_side) + np.median(hi_side))
    h0, h1 = h[j], h[j + 1]
    w = 0.5 if h1 == h0 else min(1.0, max(0.0, float((mid - h0) / (h1 - h0))))
    y_e = float(y[j] + w * (y[j + 1] - y[j]))
    if xx is None:
        return y_e
    return float(xx[j] + w * (xx[j + 1] - xx[j])), y_e


def inclined_edge_crossing(s: np.ndarray, y_edge: np.ndarray, y_ref: float = 0.0) -> tuple[float, float, float]:
    """(s_i, y_i) 직선 피팅 y = y0 + m·s → y=y_ref 인 s (서브프로파일 정밀도). 반환 (s_cross, slope m, rms)."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y_edge, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    s, y = s[ok], y[ok]
    A = np.column_stack([np.ones_like(s), s])
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y0, m = float(sol[0]), float(sol[1])
    if abs(m) < 1e-9:
        return float("nan"), m, float("nan")
    r = y - A @ sol
    return (y_ref - y0) / m, m, float(np.sqrt(np.mean(r ** 2)))


def edge_position_from_result(result, T_base_plate: Optional[np.ndarray] = None, y_ref: float = 0.0,
                              min_step_m: float = 0.002, only_cruise: bool = True,
                              frame_mask: Optional[np.ndarray] = None) -> dict:
    """ScanResult → 기울인 단차 에지의 진행좌표 s_edge (판 프레임 x, y=y_ref 에서).
    프로파일마다 에지 교차점 (x_e, y_e) 를 3D 점열에서 직접 보간하고 (x_e, y_e) 직선 피팅 → y=y_ref 의 x.
    frame_mask: 왕복 스캔에서 패스별 프레임 선택. 반환 {'s_edge_m','slope','rms_m','n_profiles','s','y_edge'}."""
    from ..reconstruct import profiles_to_base

    T_bp = result.T_base_plate if T_base_plate is None else np.asarray(T_base_plate, dtype=float)
    T_pb = inv_transform(T_bp)
    P = profiles_to_base(result.T_base_flange, result.calib.T_flange_opt, result.x_mm, result.z_mm, result.valid)
    N = P.shape[0]
    s_list, y_list = [], []
    for i in range(N):
        if only_cruise and not result.in_cruise[i]:
            continue
        if frame_mask is not None and not frame_mask[i]:
            continue
        pts = P[i]
        ok = np.all(np.isfinite(pts), axis=1)
        if ok.sum() < 4:
            continue
        q = transform_points(T_pb, pts[ok])
        x_e, y_e = step_edge_y_in_profile(q[:, 1], q[:, 2], min_step_m=min_step_m, x=q[:, 0])
        if np.isfinite(y_e) and np.isfinite(x_e):
            s_list.append(x_e)
            y_list.append(y_e)
    s_arr, y_arr = np.array(s_list), np.array(y_list)
    s_cross, slope, rms = inclined_edge_crossing(s_arr, y_arr, y_ref=y_ref)
    return {"s_edge_m": s_cross, "slope": slope, "rms_m": rms, "n_profiles": int(s_arr.size), "s": s_arr, "y_edge": y_arr}
