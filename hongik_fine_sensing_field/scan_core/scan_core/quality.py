"""재구성 무결성 사실 · 평면 적합.

예전 §5.7 지표·자동 판정 (compute_metrics / judge: PASS|RETRY|EXPOSURE_ADJUST|FAIL) 은 2026-10-08 (4d P2) 에 뺐다 — 판정이 아무것도
정하지 않았고 (기록된 패스 57 개 모두 FAIL, PASS 실행 포함) 구간마다 0.3–0.4 s 를 썼다. 각 문턱은 유도된 검사로 대체됐다:
충분성 = station_coverage, 판 평면 = 정련의 LMedS 평면, 표면 거리 = 가드 G0, 자세 = pass_posture, 시계 = 시계 모형. 남는 것은 무결성
사실 (integrity_facts) — 기록만, 다시 스캔하거나 멈추지 않는다 (점 수가 바뀐 프레임은 재구성이 이미 가린다).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .clock_sync import counter_gaps
from .types import QCReport, ScanResult

__all__ = ["integrity_facts", "integrity_report", "ransac_plane"]


def integrity_facts(result: ScanResult, *, auto_trigger_error: int = 0) -> dict:
    """재구성의 무결성 사실 (판정 없음): 프레임 수, 카운터 공백, 타임스탬프 무결성 위반, AutoTriggerError, 순항 창 안의 포즈 공백
    (스트림 양끝 = 이동 전후는 total 로만), 점 수와 그 변화, 시계 잠금."""
    k = result.result_counter
    clk = result.meta.get("clock", {}) or {}
    m = {"n_frames": int(k.size), "frames_dropped": counter_gaps(k), "ts_integrity_violations": int(result.ts_flag.sum()),
         "auto_trigger_error": int(auto_trigger_error), "unsynced_frames_total": int(result.unsynced.sum()),
         "n_points": int(result.n_points), "n_points_changed": bool(result.meta.get("n_points_changed", False)),
         "clock_locked": bool(clk.get("locked", False))}
    if result.in_cruise.any():
        tc = result.t_acq_s[result.in_cruise]
        win = (result.t_acq_s >= tc.min()) & (result.t_acq_s <= tc.max())
        m["unsynced_frames"] = int((result.unsynced & win).sum())
    else:
        m["unsynced_frames"] = m["unsynced_frames_total"]
    return m


def integrity_report(result: ScanResult, *, auto_trigger_error: int = 0) -> QCReport:
    """qc.json 용 QCReport: 무결성 사실만 (verdict 'UNJUDGED' — 판정하지 않는다)."""
    return QCReport(metrics=integrity_facts(result, auto_trigger_error=auto_trigger_error), verdict="UNJUDGED")


# --------------------------------------------------------------------------------------
def ransac_plane(P: np.ndarray, inlier_mm: Optional[float] = None, iters: int = 200, max_points: int = 20000,
                 rng: Optional[np.random.Generator] = None, *, k: Optional[float] = None, floor_mm: Optional[float] = None) -> dict:
    """RANSAC 평면 → {normal, d, rms_mm(인라이어), inlier_ratio, n, sigma_mm, thr_mm}.  평면: normal·p + d = 0 (m).

    inlier_mm None (기본) = **적응 문턱** (scan_core.measure 원칙 1, 2026-10-07): 가설은 잔차 중앙값이 가장 작은 것 (LMedS), σ = LMedS
    척도 (1.4826·med|r|), 인라이어 = |r| ≤ k·max(σ, floor_mm) (floor_mm = 센서 양자화 dX/√12, 모르면 유효 창 가운데 값). 인라이어 SVD
    정련 뒤 σ·문턱을 다시 잰다. inlier_mm 을 주면 그 고정 문턱 (최대 인라이어 수 가설)."""
    from .measure import K_DEFAULT, lmeds_sigma, quant_sigma_mm
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    nan_out = {"normal": np.array([np.nan] * 3), "d": np.nan, "rms_mm": np.nan, "inlier_ratio": 0.0, "n": int(P.shape[0]),
               "sigma_mm": np.nan, "thr_mm": np.nan}
    if P.shape[0] < 3:
        return nan_out
    rng = rng or np.random.default_rng(0)
    if P.shape[0] > max_points:
        P = P[rng.choice(P.shape[0], max_points, replace=False)]
    M = P.shape[0]
    adaptive = inlier_mm is None
    kk = float(K_DEFAULT if k is None else k)
    fl = float(quant_sigma_mm(None) if floor_mm is None else floor_mm) * 1e-3
    best_n, best_d = None, 0.0
    best_cost = np.inf
    for _ in range(iters):
        idx = rng.choice(M, 3, replace=False)
        a, b, c = P[idx]
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        nrm /= nn
        d = -float(nrm @ a)
        dist = np.abs(P @ nrm + d)
        cost = float(np.median(dist)) if adaptive else -float((dist < inlier_mm * 1e-3).sum())
        if cost < best_cost:
            best_cost, best_n, best_d = cost, nrm, d
    if best_n is None:
        return dict(nan_out, n=M)

    def thr_of(dist):
        if not adaptive:
            return inlier_mm * 1e-3, float("nan")
        s = lmeds_sigma(dist, 1)
        return kk * max(s, fl), s

    nrm, d = best_n, best_d
    for _ in range(2):                                  # 인라이어 SVD 정련 (적응이면 σ·문턱도 정련된 평면에서 다시)
        dist = P @ nrm + d
        thr, _s = thr_of(dist)
        inl = np.abs(dist) < thr
        if inl.sum() < 3:
            break
        Q = P[inl]
        c = Q.mean(axis=0)
        _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
        nrm = Vt[-1]
        if nrm[2] < 0:
            nrm = -nrm
        d = -float(nrm @ c)
    dist = P @ nrm + d
    thr, sig = thr_of(dist)
    inl = np.abs(dist) < thr
    rms = float(np.sqrt(np.mean(dist[inl] ** 2))) * 1e3 if inl.any() else np.nan
    return {"normal": nrm, "d": d, "rms_mm": rms, "inlier_ratio": float(inl.mean()), "n": M,
            "sigma_mm": float(sig) * 1e3 if np.isfinite(sig) else float("nan"), "thr_mm": float(thr) * 1e3}
