"""프로파일 유효성 마스크 · dX(z) · 열 단조성(반사 후보) 필터.

VC nano 3D-Z (12/24) large:
  * 무효점은 매뉴얼상 NaN, 연구실 펌웨어에서는 교정범위 밖 열이 finite x + z=0 센티널로도 관측됨
    → valid = isfinite(x) & isfinite(z) & (z > 0) & (233.1 < z < 544.0)
  * 열 간격 dX(z) 는 z 에 따라 0.082 mm@250 → 0.155 mm@525 (고정 열 인덱스, 고정 X 격자 아님)
  * 점 수는 서브헤더 NumberOfPoints 로 결정 — 1408 을 하드코딩하지 않는다.
"""
from __future__ import annotations

import numpy as np

Z_VALID_MM_DEFAULT: tuple[float, float] = (233.1, 544.0)
DX_KNOTS_Z_MM = (250.0, 525.0)
DX_KNOTS_MM = (0.082, 0.155)


def dX_mm(z_mm) -> np.ndarray:
    """열 간격 [mm] (z [mm] 의 선형 보간, 범위 밖은 클램프)."""
    return np.interp(np.asarray(z_mm, dtype=float), DX_KNOTS_Z_MM, DX_KNOTS_MM)


def dx_linear_coeffs() -> tuple[float, float]:
    """dX(z) ≈ a + b·z 의 (a, b) [mm, mm/mm] — synth 광선 모델용 (클램프 없는 직선)."""
    b = (DX_KNOTS_MM[1] - DX_KNOTS_MM[0]) / (DX_KNOTS_Z_MM[1] - DX_KNOTS_Z_MM[0])
    a = DX_KNOTS_MM[0] - DX_KNOTS_Z_MM[0] * b
    return a, b


def valid_mask(x_mm, z_mm, z_valid_mm=Z_VALID_MM_DEFAULT) -> np.ndarray:
    """유효 마스크 (임의 shape, 원소별)."""
    x = np.asarray(x_mm, dtype=float)
    z = np.asarray(z_mm, dtype=float)
    zlo, zhi = float(z_valid_mm[0]), float(z_valid_mm[1])
    with np.errstate(invalid="ignore"):
        return np.isfinite(x) & np.isfinite(z) & (z > 0.0) & (z > zlo) & (z < zhi)


def column_monotonic_mask(x_mm, z_mm, k: float = 5.0, z_fallback_mm: float = 320.0) -> np.ndarray:
    """이웃 열과 |Δx| > k·dX(z) 인 열 = 반사 후보 → False. 1-D 프로파일 또는 (N, n_pts)."""
    x = np.asarray(x_mm, dtype=float)
    z = np.asarray(z_mm, dtype=float)
    ok = np.ones(x.shape, dtype=bool)
    if x.shape[-1] < 2:
        return ok
    dx = np.abs(np.diff(x, axis=-1))
    zz = np.where(np.isfinite(z[..., :-1]) & (z[..., :-1] > 0), z[..., :-1], z_fallback_mm)
    thr = k * dX_mm(zz)
    with np.errstate(invalid="ignore"):
        bad = np.isfinite(dx) & (dx > thr)
    ok[..., :-1] &= ~bad
    ok[..., 1:] &= ~bad
    return ok


def valid_point_ratio(valid: np.ndarray) -> np.ndarray:
    """프레임별 유효열 비율 (분모 = NumberOfPoints)."""
    v = np.asarray(valid, dtype=bool)
    if v.ndim == 1:
        return np.array([v.mean()]) if v.size else np.array([np.nan])
    return v.mean(axis=1) if v.shape[1] else np.full(v.shape[0], np.nan)
