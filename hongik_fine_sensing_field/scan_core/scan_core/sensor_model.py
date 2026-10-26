"""VC nano 3D-Z (12/24) large 센서 모델: FOV(z)·열 간격 dx(z)·z 범위·스탠드오프 선택.

출처
  * 공칭(데이터시트): FOV x 105 mm @ z 250 → 215 mm @ z 525, z 범위 250~525 mm, 1408 열.
  * 실측 fit (2026-09-30, 흑피 강판 @ z≈425 mm: x −85.7…95.2 → 180.9 mm; validity.dX_mm 의 knot 0.082@250 / 0.155@525
    ×1408 = 115.5 / 218.2): fov_x_mm(z) ≈ 0.37·z + 23  (공칭보다 근거리에서 ~10 % 넓음, 원거리에서는 일치).
  * 교정 범위(펌웨어 1.9.7 실기): 232.6~546.5 mm (validity.Z_VALID_MM_DEFAULT 는 안쪽 233.1~544.0 을 쓴다).

센서 z 축은 **range**(센서에서 멀수록 큼)다. x 는 레이저 라인 방향 [mm]. 무효점은 NaN (또는 z=0 센티널).

이 모듈은 구조가 센서 특성에 의존하지 않도록 순수 기하(시야·표본 간격)만 제공한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

__all__ = ["SensorModel", "VC_NANO_3DZ_LARGE", "fov_x_mm", "dx_mm", "choose_standoff", "StandoffChoice",
           "required_fov_for_groove_mm"]


@dataclass(frozen=True)
class SensorModel:
    name: str = "VC nano 3D-Z (12/24) large"
    n_columns: int = 1408
    fov_slope: float = 0.37            # fov_x_mm(z) = fov_slope·z + fov_intercept  (실측 fit)
    fov_intercept: float = 23.0
    z_nominal_mm: tuple = (250.0, 525.0)        # 데이터시트 측정 범위
    z_calibrated_mm: tuple = (232.6, 546.5)     # 실기 교정 범위 (펌웨어 1.9.7)
    fov_nominal_mm: tuple = (105.0, 215.0)      # 데이터시트 FOV @ z_nominal (참고)
    optical_cross_mm: float = 337.0             # 광축 교차 근방 (설계 §3 참고값)

    # ---- 기하 ----
    def fov_x_mm(self, z_mm) -> np.ndarray:
        """레이저 라인 방향 시야 폭 [mm] (z [mm] range 의 1차 함수, 실측 fit)."""
        return self.fov_slope * np.asarray(z_mm, dtype=float) + self.fov_intercept

    def fov_x_nominal_mm(self, z_mm) -> np.ndarray:
        """데이터시트 공칭 FOV (105@250 → 215@525 선형 보간, 범위 밖 외삽)."""
        z0, z1 = self.z_nominal_mm
        f0, f1 = self.fov_nominal_mm
        return f0 + (np.asarray(z_mm, dtype=float) - z0) * (f1 - f0) / (z1 - z0)

    def dx_mm(self, z_mm) -> np.ndarray:
        """열 간격 [mm] = fov/n_columns (고정 열 인덱스, z 에 따라 변함)."""
        return self.fov_x_mm(z_mm) / float(self.n_columns)

    def z_for_fov(self, fov_mm: float) -> float:
        """주어진 FOV 폭을 얻는 최소 z [mm] (fit 역함수)."""
        return (float(fov_mm) - self.fov_intercept) / self.fov_slope

    def in_nominal_range(self, z_mm: float) -> bool:
        return self.z_nominal_mm[0] <= float(z_mm) <= self.z_nominal_mm[1]

    def in_calibrated_range(self, z_mm: float) -> bool:
        return self.z_calibrated_mm[0] <= float(z_mm) <= self.z_calibrated_mm[1]

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("z_nominal_mm", "z_calibrated_mm", "fov_nominal_mm"):
            d[k] = [float(v) for v in getattr(self, k)]
        return d


VC_NANO_3DZ_LARGE = SensorModel()


def fov_x_mm(z_mm, model: SensorModel = VC_NANO_3DZ_LARGE) -> np.ndarray:
    return model.fov_x_mm(z_mm)


def dx_mm(z_mm, model: SensorModel = VC_NANO_3DZ_LARGE) -> np.ndarray:
    return model.dx_mm(z_mm)


# --------------------------------------------------------------------------------------
# 스탠드오프 선택
# --------------------------------------------------------------------------------------
@dataclass
class StandoffChoice:
    required_fov_mm: float
    margin_mm: float
    z_min_for_fov_mm: float          # FOV 요구를 만족하는 최소 z (margin 포함)
    recommended_mm: float            # max(z_min_for_fov, preferred) → 공칭 범위로 클램프
    fov_at_recommended_mm: float
    dx_at_recommended_mm: float
    preferred_mm: float
    z_range_mm: tuple
    feasible: bool
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["z_range_mm"] = [float(v) for v in self.z_range_mm]
        return d

    def summary(self) -> str:
        s = (f"required FOV {self.required_fov_mm:.1f} mm (+{self.margin_mm:.0f} margin) -> z >= {self.z_min_for_fov_mm:.0f} mm; "
             f"recommended standoff {self.recommended_mm:.0f} mm (FOV {self.fov_at_recommended_mm:.1f} mm, "
             f"dx {self.dx_at_recommended_mm:.3f} mm/col), range {self.z_range_mm[0]:.1f}~{self.z_range_mm[1]:.1f} mm")
        if not self.feasible:
            s += "  [NOT FEASIBLE]"
        return s


def choose_standoff(required_fov_mm: float, margin_mm: float = 10.0, *, preferred_mm: float = 320.0,
                    model: SensorModel = VC_NANO_3DZ_LARGE, use_calibrated_range: bool = False) -> StandoffChoice:
    """요구 시야폭 → 권장 스탠드오프 [mm].

    z_min = model.z_for_fov(required + margin); recommended = clamp(max(z_min, preferred), z_range).
    예: 그루브 상부 개구 10.8 mm + 양쪽 모재 60 mm = 130.8 mm → z ≥ ~291 mm; 기본 320 mm 는 FOV ≈ 141 mm.
    z_range 상단에서도 FOV 가 부족하면 feasible=False (ValueError 대신 플래그; 호출자가 판단).
    """
    req = float(required_fov_mm)
    if not (math.isfinite(req) and req > 0):
        raise ValueError(f"required_fov_mm must be > 0, got {required_fov_mm!r}")
    zr = model.z_calibrated_mm if use_calibrated_range else model.z_nominal_mm
    z_min = model.z_for_fov(req + float(margin_mm))
    notes = []
    rec = max(z_min, float(preferred_mm))
    feasible = True
    if rec > zr[1]:
        notes.append(f"required FOV {req + margin_mm:.1f} mm exceeds FOV {float(model.fov_x_mm(zr[1])):.1f} mm at z_max {zr[1]:.1f} mm")
        feasible = z_min <= zr[1]
        rec = zr[1]
    if rec < zr[0]:
        rec = zr[0]
    if z_min > float(preferred_mm):
        notes.append(f"preferred standoff {preferred_mm:.0f} mm too close for the required FOV; raised to {rec:.0f} mm")
    return StandoffChoice(required_fov_mm=req, margin_mm=float(margin_mm), z_min_for_fov_mm=float(z_min),
                          recommended_mm=float(rec), fov_at_recommended_mm=float(model.fov_x_mm(rec)),
                          dx_at_recommended_mm=float(model.dx_mm(rec)), preferred_mm=float(preferred_mm),
                          z_range_mm=(float(zr[0]), float(zr[1])), feasible=feasible, notes=notes)


def required_fov_for_groove_mm(top_opening_mm: float, plate_margin_mm: float = 60.0,
                               lateral_misalignment_mm: float = 0.0) -> float:
    """그루브 검증에 필요한 시야폭: 상부 개구 + 양쪽 모재 각 plate_margin + 2×센서 중심 오정렬 여유."""
    return float(top_opening_mm) + 2.0 * float(plate_margin_mm) + 2.0 * float(lateral_misalignment_mm)
