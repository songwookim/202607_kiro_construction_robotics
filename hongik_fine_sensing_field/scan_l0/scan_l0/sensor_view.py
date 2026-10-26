"""센서 시야 모형 (사용자 2026-10-07 '센서 모형에서 유도'): 거리·탐색 간격을 장치 사양·설계 정보·안전 정책에서 유도한다 — 설정 상수 없음.

  유효 거리 창    [z_near, z_far]   플랫폼 sensor.z_valid_mm (센서 교정 창)
  광축 교차 거리  r*                센서 모형 optical_cross_mm (삼각측량 정확도가 가장 좋은 거리)
  시야 폭         FOV(r)            센서 모형 (실측 fit)
  도착 허용       e                 안전 정책 SafetyPolicy.pose_tol_m (도착 자세 오차가 툴 점으로 커진 값)
  가까운 한계     z_min             안전 정책 standoff_min (창 하한 + 정지 거리 = 가드 G0 사전검사 거리)

  패스 (정밀):   그루브 깊이 가운데가 r* 에 오게 — 판 상면 거리 = r* − 깊이/2 (작업 standoff = 루트까지 광축 거리 = 그 + 깊이).
                 툴이 판·지지면에서 안전 한계를 지키는지는 계획 쪽 (포락선·끝 자세 선택) 이 본다.
  거친 단계:     대략 착점의 오차를 모든 방향으로 가장 크게 견디는 거리 (넓은 시야는 측면 오차를, 창 가운데는 시선 방향 오차를 견딘다):
                 r_c = argmax_r min(δ_가까이(r), δ_멀리(r), δ_측면(r)) —
                   δ_멀리   = z_far − e − 깊이/cos t − r      (착점이 판 상면이면 루트가 창 먼 쪽 안)
                   δ_가까이 = r − 깊이/cos t − z_min − e      (착점이 루트면 판 상면이 가드 G0 거리 밖)
                   δ_측면   = FOV(r)/2 − 개구/2                (그루브 전체가 시야 안)
                 연구실 (깊이 10, 개구 10.8, 기울기 0): r_c ≈ 444 mm, 어느 방향으로든 ≈ 88 mm 어긋난 착점까지 첫 시야에 그루브. 착점까지
                 광축 거리이고 보정 목표 (판 법선 거리) = r_c·cos t.
  측면 탐색:     간격 = FOV(r) − 개구 (이웃 시야가 개구만큼 겹쳐 그루브 전체가 어느 한 시야에 든다), 범위 = 시편 폭 (착점이 시편 위면
                 그루브는 그 폭 안).
  보이는 면 없음: 거친 거리에서 아무것도 안 보이면 광축을 따라 r* 로 다가가 다시 본다 (신호·정확도가 가장 좋은 거리). 너무 가까운 쪽은
                 가드 (G0 = z_near + 정지 거리) 가 막으므로 물러나는 규칙은 없다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from scan_core.sensor_model import VC_NANO_3DZ_LARGE, SensorModel

__all__ = ["SensorView", "sensor_view"]


@dataclass(frozen=True)
class SensorView:
    z_near_m: float
    z_far_m: float
    cross_m: float
    pose_tol_m: float = 0.0
    near_m: Optional[float] = None          # 가까운 한계 (안전 정책 standoff_min = 창 하한 + 정지 거리); None = 창 하한
    model: SensorModel = VC_NANO_3DZ_LARGE

    def fov_m(self, range_m: float) -> float:
        """거리 range_m 에서 레이저 라인 방향 시야 폭 [m] (센서 모형 fit)."""
        return float(self.model.fov_x_mm(float(range_m) * 1e3)) * 1e-3

    def pass_plate_m(self, depth_m: float) -> float:
        """패스: 판 상면까지 광축 거리 = r* − 깊이/2 (그루브 깊이 가운데가 광축 교차 거리)."""
        return float(self.cross_m) - 0.5 * max(0.0, float(depth_m))

    def _coarse(self, depth_m: float, tilt_deg: float, opening_m: float) -> tuple:
        d = max(0.0, float(depth_m)) / math.cos(math.radians(float(tilt_deg)))
        e = float(self.pose_tol_m)
        near = float(self.z_near_m if self.near_m is None else self.near_m)
        lo, hi = near + e + d, float(self.z_far_m) - e - d
        if not (hi > lo):
            raise ValueError(f"the sensor window {near * 1e3:.0f}-{self.z_far_m * 1e3:.0f} mm cannot hold the groove depth {d * 1e3:.0f} mm")
        best = None
        for r in np.linspace(lo, hi, max(2, int(round((hi - lo) / 5e-4)) + 1)):      # 0.5 mm 격자 (수치 해상도)
            tol = min(r - lo, hi - r, 0.5 * (self.fov_m(r) - max(0.0, float(opening_m))))
            if best is None or tol > best[1]:
                best = (float(r), float(tol))
        return best

    def tolerance_at_m(self, range_m: float, depth_m: float, tilt_deg: float = 0.0, opening_m: float = 0.0) -> float:
        """거리 range_m 에서 첫 시야가 견디는 대략 착점 오차 [m] (시선 방향 양쪽·측면 중 가장 작은 쪽; 음수 = 그루브가 창 밖)."""
        d = max(0.0, float(depth_m)) / math.cos(math.radians(float(tilt_deg)))
        e = float(self.pose_tol_m)
        near = float(self.z_near_m if self.near_m is None else self.near_m)
        r = float(range_m)
        return min(r - (near + e + d), float(self.z_far_m) - e - d - r, 0.5 * (self.fov_m(r) - max(0.0, float(opening_m))))

    def coarse_range_m(self, depth_m: float, tilt_deg: float = 0.0, opening_m: float = 0.0) -> float:
        """거친 단계: 교시 착점까지 광축 거리 (대략 착점 오차를 모든 방향으로 가장 크게 견디는 거리)."""
        return self._coarse(depth_m, tilt_deg, opening_m)[0]

    def coarse_tolerance_m(self, depth_m: float, tilt_deg: float = 0.0, opening_m: float = 0.0) -> float:
        """그 거리에서 첫 시야가 견디는 착점 오차 (시선 방향·측면 중 작은 쪽) [m]."""
        return self._coarse(depth_m, tilt_deg, opening_m)[1]

    def coarse_plate_m(self, depth_m: float, tilt_deg: float = 0.0, opening_m: float = 0.0) -> float:
        """거친 단계 보정 목표: 판 법선 거리 = r_c·cos t."""
        return self.coarse_range_m(depth_m, tilt_deg, opening_m) * math.cos(math.radians(float(tilt_deg)))

    def search_steps_m(self, range_m: float, opening_m: float, width_m: float) -> list:
        """측면 탐색 스텝 [m] (+s, −s, +2s, −2s, …): s = FOV(r) − 개구 — 이웃 시야가 개구만큼 겹쳐, k 스텝까지 개구 전체가 어느 한
        시야에 드는 그루브 중심 범위가 k·s + FOV/2 − 개구/2 로 끊김 없이 늘어난다. 그 범위가 시편 폭 width (착점이 시편 위면 그루브는
        그 안) 에 닿을 때까지. 첫 시야가 이미 덮으면 빈 목록."""
        fov = self.fov_m(range_m)
        s = fov - max(0.0, float(opening_m))
        if not (s > 0.0):
            return []
        out, k = [], 1
        while (k - 1) * s + 0.5 * (fov - float(opening_m)) < float(width_m):
            out += [k * s, -k * s]
            k += 1
        return out

    def summary(self, depth_m: float, opening_m: float, width_m: float) -> str:
        rc, tol = self._coarse(depth_m, 0.0, opening_m)
        rp = self.pass_plate_m(depth_m)
        st = self.search_steps_m(rc, opening_m, width_m)
        return (f"sensor view: window {self.z_near_m * 1e3:.1f}-{self.z_far_m * 1e3:.1f} mm, optical cross {self.cross_m * 1e3:.0f} mm; "
                f"coarse range {rc * 1e3:.1f} mm (tolerates a rough point {tol * 1e3:.0f} mm off along the sight or across the laser "
                f"line; FOV {self.fov_m(rc) * 1e3:.0f} mm), pass plate {rp * 1e3:.1f} mm (groove mid-depth at the optical cross; "
                f"FOV {self.fov_m(rp) * 1e3:.0f} mm); lateral search "
                + (f"{len(st) // 2} x +-{st[0] * 1e3:.0f} mm (FOV - opening) over the specimen width {width_m * 1e3:.0f} mm"
                   if st else "none (one view covers the specimen width)"))

    def as_dict(self) -> dict:
        return {"z_near_m": self.z_near_m, "z_far_m": self.z_far_m, "cross_m": self.cross_m, "pose_tol_m": self.pose_tol_m,
                "near_m": self.near_m, "model": self.model.name}


def sensor_view(z_valid_mm, *, pose_tol_m: float = 0.0, near_m: Optional[float] = None,
                model: Optional[SensorModel] = None) -> SensorView:
    """플랫폼 유효 창 (sensor.z_valid_mm) + 센서 모형 + 안전 정책 (도착 허용, 가까운 한계 standoff_min) → SensorView."""
    m = model or VC_NANO_3DZ_LARGE
    z0, z1 = float(z_valid_mm[0]), float(z_valid_mm[1])
    if not (0.0 < z0 < z1):
        raise ValueError(f"sensor.z_valid_mm must be [near, far] with 0 < near < far, got {z_valid_mm!r}")
    cross = float(m.optical_cross_mm)
    if not (z0 < cross < z1):
        raise ValueError(f"sensor model optical cross {cross} mm is outside the valid window {z0}-{z1} mm")
    return SensorView(z_near_m=z0 * 1e-3, z_far_m=z1 * 1e-3, cross_m=cross * 1e-3, pose_tol_m=float(pose_tol_m),
                      near_m=None if near_m is None else float(near_m), model=m)
