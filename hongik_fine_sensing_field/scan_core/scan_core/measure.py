"""측정 모형 — 판정 기준의 공통 원천 (사용자 2026-10-07 4단계: 판정에 작위적 상수 없음).

명시값은 통계 배수 k 하나 (플랫폼 judgement.k, 사용자 결정 3). 나머지 판정 기준은 아래 원칙에서만 나온다:

  1. k·σ          '이 점·값이 모형과 맞나' — σ 는 데이터에서 잰 강건 표준편차 (1.4826·MAD), 하한은 센서 양자화 dX(z)/√12
                  (센서 모형 validity.dX_mm). 인라이어·수렴·일관성·끝 위치 등.
  2. 가설 중간점  '설계 특징인가 / 증거가 있나' — 공칭 ±50 % (설계 특징 vs 특징 없음의 중간), 설계가 기대하는 점 개수의 절반 이상.
                  시편 편차는 그 안이면 측정을 믿고 보고한다 (사용자 2026-10-07 결정: 공칭 ±50 %).
  3. 기하·센서    길이·창은 센서 열 간격·시야·착점 간격·설계 치수에서.
  4. 용접 경로    측정 충분성 = 용접 경로 단면 창 (weld_plan station) 마다 유효 측정 (사용자 2026-10-07 결정).
  5. 안전 정책    도착 허용 e, 유령점 버스트 + 1 (scan_l0.safety).
  6. 상대 비교    후보 사이의 선택.
  7. 수치 해상도  격자 간격·반복 횟수 (판정 기준이 아니다).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .validity import Z_VALID_MM_DEFAULT, dX_mm

__all__ = ["K_DEFAULT", "MAD_TO_SIGMA", "DESIGN_HALF", "robust_sigma", "quant_sigma_mm", "noise_sigma", "k_sigma",
           "design_match", "design_band", "evidence_count", "station_coverage", "end_sigma_m", "lmeds_sigma", "HALF_TURN_DEG"]

K_DEFAULT = 3.0          # 통계 배수 (사용자 2026-10-07 결정; 플랫폼 judgement.k 가 덮어쓴다)
MAD_TO_SIGMA = 1.4826    # 정규분포에서 MAD → σ (통계 상수)
DESIGN_HALF = 0.5        # 가설 중간점: '설계 값' 과 '없음 (0)' 사이의 경계 (원칙 2)
HALF_TURN_DEG = 45.0     # 가설 중간점 (각도): '나란함 (0°)' 과 '수직 (90°)' 사이 — 면이 수평에 가까운가 수직에 가까운가
_RAYLEIGH_MEDIAN = math.sqrt(2.0 * math.log(2.0))   # 수직 두 성분이 N(0, σ²) 인 거리의 중앙값 = σ·√(2 ln 2)


def robust_sigma(r, min_n: int = 3) -> float:
    """1.4826·MAD (중앙값 둘레). 표본이 min_n 보다 적으면 NaN."""
    a = np.asarray(r, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size < int(min_n):
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(a - np.median(a))))


def lmeds_sigma(r, dof: int = 1) -> float:
    """최소 중앙값 (LMedS) 척도 — 모형까지의 잔차 r 에서 σ: 부호 있는 거리 (dof 1, 평면) σ = 1.4826·med|r|, 직선까지 거리 (dof 2, 수직
    두 성분) σ = med|r| / √(2 ln 2). 이상점이 절반 미만이면 흔들리지 않는다 (RANSAC 인라이어 문턱 = k·σ, 원칙 1)."""
    a = np.abs(np.asarray(r, dtype=float).ravel())
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    m = float(np.median(a))
    return m / _RAYLEIGH_MEDIAN if int(dof) == 2 else MAD_TO_SIGMA * m


def quant_sigma_mm(z_mm) -> float:
    """센서 양자화 잡음 하한 [mm] = 열 간격 dX(z)/√12 (z 를 모르면 유효 창 가운데)."""
    z = float(z_mm) if (z_mm is not None and math.isfinite(float(z_mm))) else 0.5 * sum(Z_VALID_MM_DEFAULT)
    return float(dX_mm(z)) / math.sqrt(12.0)


def noise_sigma(r, z_mm=None, *, unit: str = "mm") -> float:
    """잰 강건 σ 와 센서 양자화 하한 중 큰 값 (unit 'mm' 또는 'm' — 잔차의 단위)."""
    s = robust_sigma(r)
    q = quant_sigma_mm(z_mm) * (1e-3 if unit == "m" else 1.0)
    return float(max(s, q)) if math.isfinite(s) else float(q)


def k_sigma(r, z_mm=None, *, k: Optional[float] = None, unit: str = "mm") -> float:
    """k × noise_sigma — '모형과 맞나' 의 문턱 (원칙 1)."""
    return float((K_DEFAULT if k is None else float(k)) * noise_sigma(r, z_mm, unit=unit))


def design_band(nominal: float) -> float:
    """설계 일치 반폭 = |공칭| × ½ (원칙 2)."""
    return float(abs(float(nominal)) * DESIGN_HALF)


def design_match(value, nominal) -> bool:
    """측정값이 설계 특징으로 볼 수 있나: |값 − 공칭| ≤ |공칭|/2 (공칭이 0 이하·NaN 이면 판정 안 함 → True)."""
    v, n = float(value), float(nominal)
    if not (math.isfinite(n) and n > 0.0):
        return True
    return bool(math.isfinite(v) and abs(v - n) <= design_band(n))


def evidence_count(expected: float) -> int:
    """'증거가 있다' 의 최소 점 개수 = 설계가 기대하는 개수의 절반 (원칙 2), 최소 1."""
    e = float(expected)
    return max(1, int(math.ceil(DESIGN_HALF * e))) if math.isfinite(e) else 1


def end_sigma_m(spacing_m: float) -> float:
    """판 끝 위치의 σ [m]: 끝은 이웃 두 프로파일 사이 어딘가 (균일) → 착점 간격/√12."""
    sp = float(spacing_m)
    return sp / math.sqrt(12.0) if (math.isfinite(sp) and sp > 0.0) else float("nan")


def station_coverage(s_m, valid, s_from: float, s_to: float, *, step_m: float, window_m: float, min_profiles: int,
                     spacing_m: float = float("nan")) -> dict:
    """용접 경로 단면 (s_from..s_to, step_m 간격; 각 단면은 ±window_m/2 창) 마다 유효 프로파일 수 (원칙 4). 단면이 '측정됨' = 그 창에서
    가능한 수 (용접 계획의 min_profiles 와 창/착점 간격 중 작은 값) 의 절반 이상 (원칙 2: 측정됨 vs 없음의 중간 — 산발 결측에는 멈추지
    않고, 통째로 빈 단면은 잡는다). 끝 단면은 창이 시편 밖으로 나간 비율만큼 줄인다.
    반환 {n_stations, min_count, median_count, need, short: [(s_station, count), …], ok}."""
    s = np.asarray(s_m, dtype=float)
    v = np.asarray(valid, dtype=bool) & np.isfinite(s)
    a, b = float(min(s_from, s_to)), float(max(s_from, s_to))
    if not (math.isfinite(a) and math.isfinite(b)) or b <= a or step_m <= 0.0:
        return {"n_stations": 0, "min_count": 0, "short": [], "ok": False}
    n = max(1, int(round((b - a) / float(step_m)))) + 1
    st = np.linspace(a, b, n)
    sv = np.sort(s[v])
    half = 0.5 * float(window_m)
    lo = np.searchsorted(sv, st - half, side="left")
    hi = np.searchsorted(sv, st + half, side="right")
    cnt = hi - lo
    full = float(min_profiles)
    sp = float(spacing_m)
    if math.isfinite(sp) and sp > 0.0:
        full = min(full, math.floor(float(window_m) / sp + 1e-9))
    full = max(1.0, full)
    inside = (np.minimum(st + half, b) - np.maximum(st - half, a)) / max(float(window_m), 1e-12)
    need = np.maximum(1, np.ceil(DESIGN_HALF * np.clip(inside, 0.0, 1.0) * full - 1e-9)).astype(int)
    short = [(float(x), int(c)) for x, c, nd in zip(st, cnt, need) if c < nd]
    return {"n_stations": int(n), "min_count": int(cnt.min()) if cnt.size else 0,
            "median_count": float(np.median(cnt)) if cnt.size else 0.0, "need": int(math.ceil(DESIGN_HALF * full - 1e-9)),
            "short": short, "ok": not short}
