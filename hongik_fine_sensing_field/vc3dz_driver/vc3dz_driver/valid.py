"""프로파일 유효 마스크 + 호스트측 반사 필터 (설계 §3.5).

- 무효점: NaN(매뉴얼, TransferAllPos_MM=1) **및** finite x + z=0.000 센티널(교정 범위 밖, 실측) →
  값 기반 마스크 `isfinite(x)&isfinite(z)&(z>0)&(zmin<z<zmax)` 로 모드 0/1 어느 쪽에서도 동일 동작.
- 반사 필터: (3) 열 단조성 — x 는 열 순서로 단조증가(실측 1.2M 쌍 중 Δx≤0 0건). 최장증가부분열(LIS)
  에 속하지 않고 이웃 LIS 점과 k·dX(z) 이상 어긋나는 점을 반사 후보로 제거.
  (4) Z 점프 |Δz| > z_jump 로 나뉘는 런 중 길이 < min_run 인 고립 런 제거(갭·베벨 코너는 런 길이로 보존).
- dX(z): 0.082 mm/px @233 → 0.155 mm/px @544 선형 보간(FOV ≈ 0.37·Z + 23 mm / 1408).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

Z_VALID_MM: Tuple[float, float] = (233.1, 544.0)   # 교정 Z 범위(실측 233–544), 경계 제외
DX_AT_ZMIN = 0.082     # mm/px @ 233 mm
DX_AT_ZMAX = 0.155     # mm/px @ 544 mm
NOMINAL_POINTS = 1408  # 풀 ROI 기준값(참고용; 하드코딩해서 판단하지 말 것)


def dx_of_z(z) -> np.ndarray:
    """열 간격 dX(z) [mm/px], 선형 보간(범위 밖은 클램프)."""
    z = np.asarray(z, dtype=np.float64)
    t = np.clip((z - Z_VALID_MM[0]) / (Z_VALID_MM[1] - Z_VALID_MM[0]), 0.0, 1.0)
    return DX_AT_ZMIN + t * (DX_AT_ZMAX - DX_AT_ZMIN)


def valid_mask(x, z, zmin: float = Z_VALID_MM[0], zmax: float = Z_VALID_MM[1]) -> np.ndarray:
    """유효점 마스크. isfinite 단독 불충분(z=0 센티널) — 값 기반."""
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        return np.isfinite(x) & np.isfinite(z) & (z > 0.0) & (z > zmin) & (z < zmax)


def _lis_indices(v: np.ndarray) -> np.ndarray:
    """비감소(non-decreasing) 최장 부분열의 인덱스(patience sorting, O(n log n))."""
    n = v.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    tails_val = []          # tails_val[k] = 길이 k+1 인 부분열의 최소 끝값
    tails_idx = []
    prev = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        k = bisect.bisect_right(tails_val, v[i])
        if k == len(tails_val):
            tails_val.append(v[i]); tails_idx.append(i)
        else:
            tails_val[k] = v[i]; tails_idx[k] = i
        prev[i] = tails_idx[k - 1] if k > 0 else -1
    out = []
    i = tails_idx[-1]
    while i >= 0:
        out.append(i); i = prev[i]
    return np.asarray(out[::-1], dtype=np.int64)


@dataclass
class FilterResult:
    keep: np.ndarray                 # bool [N]: 유효 & 반사 아님
    valid: np.ndarray                # bool [N]: 값 기반 유효 마스크(2항)
    reflection_x: np.ndarray         # bool [N]: 열 단조성 위반으로 제거
    isolated_z: np.ndarray           # bool [N]: 고립 Z 런으로 제거
    stats: Dict[str, float] = field(default_factory=dict)


def reflection_filter(x, z, k_dx: float = 5.0, z_jump_mm: float = 3.0, min_run: int = 5,
                      column_indexed: Optional[bool] = None, n_columns: Optional[int] = None,
                      zmin: float = Z_VALID_MM[0], zmax: float = Z_VALID_MM[1]) -> FilterResult:
    """유효 마스크 + 열 단조성(x) + 고립 Z 런 제거.

    column_indexed: True 면 배열 인덱스 = 픽셀 열(TransferAllPos_MM=1); None 이면 N==n_columns 로 추정.
    n_columns: ROI_DX_PIX(모드 0 폴백에서 valid_point_ratio 분모). None 이면 N.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    n = x.size
    valid = valid_mask(x, z, zmin, zmax)
    refl = np.zeros(n, dtype=bool)
    iso = np.zeros(n, dtype=bool)
    if column_indexed is None:
        column_indexed = (n_columns is not None and n == n_columns)
    denom = float(n_columns) if n_columns else float(max(n, 1))

    vi = np.flatnonzero(valid)
    if vi.size >= 3:
        xv = x[vi]
        zv = z[vi]
        lis = _lis_indices(xv)                          # xv 내 인덱스
        in_lis = np.zeros(vi.size, dtype=bool)
        in_lis[lis] = True
        if not in_lis.all():
            tol = k_dx * dx_of_z(zv)
            # 각 비-LIS 점의 좌/우 LIS 이웃
            lis_pos = np.flatnonzero(in_lis)
            cand = np.flatnonzero(~in_lis)
            left = np.searchsorted(lis_pos, cand, side='left') - 1
            right = left + 1
            lo = np.where(left >= 0, xv[lis_pos[np.clip(left, 0, lis_pos.size - 1)]], -np.inf)
            hi = np.where(right < lis_pos.size, xv[lis_pos[np.clip(right, 0, lis_pos.size - 1)]], np.inf)
            bad = (xv[cand] < lo - tol[cand]) | (xv[cand] > hi + tol[cand])
            refl[vi[cand[bad]]] = True
            # 2차: 남은 점들 사이에서도 x_j >= max(x_<j) - tol 을 보장(비-LIS 점 두 개가 서로 역행하는 경우)
            kept = np.flatnonzero(~refl[vi])
            if kept.size >= 2:
                xs = xv[kept]
                prev_max = np.concatenate([[-np.inf], np.maximum.accumulate(xs)[:-1]])
                bad2 = xs < prev_max - tol[kept]
                refl[vi[kept[bad2]]] = True

        # (4) 고립 Z 런: 남은 점을 순서대로, |Δz| > z_jump 에서 분할
        keep_i = vi[~refl[vi]]
        if keep_i.size >= 2:
            zz = z[keep_i]
            brk = np.flatnonzero(np.abs(np.diff(zz)) > z_jump_mm) + 1
            starts = np.concatenate([[0], brk])
            ends = np.concatenate([brk, [keep_i.size]])
            for s, e in zip(starts, ends):
                run_len = (keep_i[e - 1] - keep_i[s] + 1) if column_indexed else (e - s)
                if run_len < min_run and (e - s) < keep_i.size:   # 전체가 한 런이면 보존
                    iso[keep_i[s:e]] = True

    keep = valid & ~refl & ~iso
    n_valid = int(valid.sum())
    removed = int(refl.sum() + iso.sum())
    stats = {
        'n_points': float(n),
        'n_valid': float(n_valid),
        'valid_point_ratio': n_valid / denom,
        'n_reflection_x': float(refl.sum()),
        'n_isolated_z': float(iso.sum()),
        'reflection_removed_ratio': removed / max(n_valid, 1),
        'column_indexed': float(bool(column_indexed)),
    }
    return FilterResult(keep, valid, refl, iso, stats)


def clean_profile(x, z, **kw) -> Tuple[np.ndarray, np.ndarray, FilterResult]:
    """keep 마스크를 적용해 NaN 처리한 (x, z) 복사본과 FilterResult 반환."""
    r = reflection_filter(x, z, **kw)
    xo = np.asarray(x, dtype=np.float64).copy()
    zo = np.asarray(z, dtype=np.float64).copy()
    xo[~r.keep] = np.nan
    zo[~r.keep] = np.nan
    return xo, zo, r
