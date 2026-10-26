"""심 S→E 좌표 계약 (2026-10-09, 실기 1G 배치 분석 3.4: 마지막 전체 패스가 E→S 인 실행의 용접 계획이 'S->E' 표기로 E 에서 S 로 갔다).

원인은 축과 끝 좌표가 서로 다른 기준에서 온 것이다 — 축은 기준 패스의 정련 직선 u (패스 **진행 방향**), 끝은 다른 직선의 S→E 스칼라.
그래서 하류 (패스 병합, 용접 계획 단면, 시편 보고, 측정 충분성) 는 모두 이 한 곳에서 같은 규칙으로 좌표를 만든다:

  * 끝은 base 점 S, E (관측된 판 끝) 로만 주고받는다 — 다른 직선 기준의 스칼라는 넘기지 않는다.
  * 축 u = unit(E − S) (판 법선 n 을 주면 n 에 직교화: 단면 좌표 (u, w = n × u, n) 이 직교).
  * 심 방향 좌표 s = (P − S)·u — S = 0, E = L = (E − S)·u > 0. 패스 진행 방향과 무관하다.
  * 끝의 불확실성 σ 는 측정 (판 끝 식별의 σ); 없으면 착점 간격/√12 (measure.end_sigma_m, 같은 원칙 1). 계약 검사의 허용은 k·σ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .measure import end_sigma_m

__all__ = ["SeamFrame", "SeamFrameError", "seam_frame", "frame_from_extent", "end_sigma_from_extent"]


class SeamFrameError(ValueError):
    """S→E 좌표를 만들 수 없다 (끝 점 없음·같은 점·법선과 나란함)."""


@dataclass(frozen=True)
class SeamFrame:
    """S→E 계약 좌표. S, E = base 끝 점 [m], u = S→E 단위 축, n = 판 법선 (없으면 None), σ = 끝 위치 불확실성 [m]."""
    S: np.ndarray
    E: np.ndarray
    u: np.ndarray
    n: Optional[np.ndarray] = None
    sigma_S_m: float = float("nan")
    sigma_E_m: float = float("nan")

    @property
    def length_m(self) -> float:
        """L = (E − S)·u (> 0)."""
        return float((self.E - self.S) @ self.u)

    def s(self, P) -> np.ndarray:
        """점 (…, 3) → S 에서 S→E 로 잰 심 방향 좌표 [m] (유한하지 않은 행은 NaN)."""
        A = np.asarray(P, dtype=float)
        flat = A.reshape(-1, 3)
        out = np.full(flat.shape[0], np.nan)
        fin = np.all(np.isfinite(flat), axis=1)
        out[fin] = (flat[fin] - self.S) @ self.u
        return out.reshape(A.shape[:-1]) if A.ndim > 1 else float(out[0])

    def point(self, s_m) -> np.ndarray:
        """심 방향 좌표 → 축 위 base 점."""
        s = np.asarray(s_m, dtype=float)
        return self.S + s[..., None] * self.u

    def sigma_m(self, end: str) -> float:
        return float(self.sigma_S_m if end == "S" else self.sigma_E_m)

    def to_dict(self) -> dict:
        return {"S_base_m": [float(v) for v in self.S], "E_base_m": [float(v) for v in self.E], "u": [float(v) for v in self.u],
                "length_mm": self.length_m * 1e3, "sigma_S_mm": self.sigma_S_m * 1e3, "sigma_E_mm": self.sigma_E_m * 1e3}


def _vec3(v, what: str) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size != 3 or not np.all(np.isfinite(a)):
        raise SeamFrameError(f"{what} is not a finite base point: {v!r}")
    return a


def seam_frame(S, E, n=None, *, sigma_S_m: float = float("nan"), sigma_E_m: float = float("nan")) -> SeamFrame:
    """끝 점 S, E (+ 판 법선 n) → SeamFrame. E − S 가 0 이거나 법선과 나란하면 SeamFrameError."""
    S_, E_ = _vec3(S, "S end"), _vec3(E, "E end")
    d = E_ - S_
    n_ = None
    if n is not None:
        n_ = _vec3(n, "plate normal")
        nn = float(np.linalg.norm(n_))
        if nn <= 0.0:
            raise SeamFrameError("plate normal has zero length")
        n_ = n_ / nn
        d = d - float(d @ n_) * n_
    L = float(np.linalg.norm(d))
    if not (L > 0.0):
        raise SeamFrameError(f"S and E do not span the seam (|E - S| along the plate = {L * 1e3:.3f} mm)")
    return SeamFrame(S=S_, E=E_, u=d / L, n=n_, sigma_S_m=float(sigma_S_m), sigma_E_m=float(sigma_E_m))


def end_sigma_from_extent(extent: dict, end: str) -> float:
    """끝 X 의 위치 σ [m]: 그 끝 식별의 σ (extent[X]['sigma_m']) — 없거나 0 이면 착점 간격/√12 (extent['spacing_mm'])."""
    rec = (extent or {}).get(end) or {}
    sg = rec.get("sigma_m")
    try:
        sg = float(sg)
    except (TypeError, ValueError):
        sg = float("nan")
    if math.isfinite(sg) and sg > 0.0:
        return sg
    try:
        sp = float((extent or {}).get("spacing_mm")) * 1e-3
    except (TypeError, ValueError):
        sp = float("nan")
    return end_sigma_m(sp)


def frame_from_extent(extent: dict, n=None) -> SeamFrame:
    """패스 extent ({S: {point, sigma_m}, E: {…}, spacing_mm}) → SeamFrame. 끝 점이 없으면 SeamFrameError."""
    if not isinstance(extent, dict) or not isinstance(extent.get("S"), dict) or not isinstance(extent.get("E"), dict):
        raise SeamFrameError("extent has no S/E end records")
    for X in ("S", "E"):
        if extent[X].get("point") is None:
            raise SeamFrameError(f"extent {X} end has no base point")
    return seam_frame(extent["S"]["point"], extent["E"]["point"], n,
                      sigma_S_m=end_sigma_from_extent(extent, "S"), sigma_E_m=end_sigma_from_extent(extent, "E"))
