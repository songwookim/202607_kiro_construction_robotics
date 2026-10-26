"""비드 모델: 용접 파라미터 → 비드 단면적·폭·볼록 단면 모양 (다층 용접 경로 계획의 공정 입력).

계획기 (weld_plan) 는 이 인터페이스 (BeadModel) 만 쓴다. 기본은 물리 모델 (MassBalanceBeadModel: 질량 보존 + 원호 비드 단면);
KIRO 데이터셋 (용접 파라미터 — 토치 각도·속도·장비 설정값 — 별 비드 단면적·폭, 사용자 2026-10-04) 으로 만든 모델이 같은
인터페이스로 갈아 끼워진다 — 그러면 같은 계획기가 그루브 자세·형상·치수에 따라 그 모델로 층·비드 배치와 속도를 정한다.

인터페이스 (PassParams p 는 패스 하나의 공정 조건):
  area_mm2(p)              비드 용착 단면적 [mm²]
  width_mm(p)              비드가 덮는 폭 [mm] (한 층에 비드 하나가 감당하는 최대 층 폭)
  speed_for_area(p, a)     p 의 다른 조건 (WFS·각도·장비 설정) 은 그대로 두고 단면적 a 를 내는 진행 속도 [m/s]
  crown_ratio(p)           (optional) the convex free surface of the pass's bead: its height over its toe chord / the chord length
                           (the free bead's h / w; scan_core.bead_geometry puts every bead's free surface at this ratio over its own
                           toe chord in the groove). Without it: the ratio of the free bead whose area and width the model gives
                           (crown_ratio_of). A data-based model fits it from post-weld bead profiles (bead_geometry.crown_ratio_from_profiles).

Bead cross-section (2026-10-13, user + KIRO: beads are convex): the free bead is a circular-arc segment over its toe chord (width w,
height h = kappa w) - the circular arc fits GMAW bead sections better than the parabola when the wire-feed / travel speed ratio is
above 12.5 (Xiong, Zhang, Gao, Wu 2013, Robot. Comput.-Integr. Manuf. 29:417-423; parabola / circular arc / cosine models also in
Suryakumar et al. 2011, Comput.-Aided Des. 43:331-344); KIRO's WPS runs at 28 / 26 / 28 (root / fill / cap: 140 / 180 / 170 mm/s
wire at 5 / 7 / 6 mm/s). The one shape parameter is the bead's TOE (flank) ANGLE theta_toe - the angle between the toe chord and the
arc's tangent at the toe (user 2026-10-13 night; WeldProcess.bead_toe_angle_deg): kappa = tan(theta_toe / 2) / 2. Segment area
a = g(kappa) w^2, g(kappa) = (1/4 + kappa^2)^2 / (4 kappa^2) (theta - sin theta cos theta), theta = theta_toe = 2 atan(2 kappa). It
sets the free bead's width (the bead count) and every bead's dome over its toe chord (bead_geometry) - an assumption until KIRO's
post-weld profiles calibrate it (bead_geometry.crown_ratio_from_profiles: kappa_p -> theta_toe = 2 atan(2 kappa_p)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["PassParams", "BeadModel", "MassBalanceBeadModel", "arc_area_ratio", "arc_ratio_for_area_ratio", "crown_ratio_of",
           "crown_ratio_for_toe_angle", "toe_angle_for_crown_ratio"]


def crown_ratio_for_toe_angle(toe_angle_deg: float) -> float:
    """kappa (the arc's height over its chord / the chord) of a circular arc meeting its chord at the toe angle theta_toe [deg]."""
    return 0.5 * math.tan(0.5 * math.radians(float(toe_angle_deg)))


def toe_angle_for_crown_ratio(kappa: float) -> float:
    """theta_toe [deg] = 2 atan(2 kappa): the toe angle of the circular arc with height kappa x its chord."""
    return math.degrees(2.0 * math.atan(2.0 * float(kappa)))


def arc_area_ratio(kappa: float) -> float:
    """g(kappa): area of the circular-arc segment over a chord c with height kappa c, divided by c^2 (module docstring)."""
    k = float(kappa)
    if k <= 0.0:
        return 0.0
    th = 2.0 * math.atan(2.0 * k)
    return (0.25 + k * k) ** 2 / (4.0 * k * k) * (th - math.sin(th) * math.cos(th))


def arc_ratio_for_area_ratio(r: float) -> float:
    """The inverse of arc_area_ratio (monotone increasing): kappa with g(kappa) = r (r >= 0)."""
    r = float(r)
    if not r > 0.0:
        return 0.0
    lo, hi = 0.0, 1.0
    while arc_area_ratio(hi) < r:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if arc_area_ratio(mid) < r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def crown_ratio_of(model, p: "PassParams") -> float:
    """The pass's crown ratio (bead height over its toe chord / chord): the model's crown_ratio(p) if it has one, otherwise the
    circular-arc segment of the model's area and width (the same free bead)."""
    f = getattr(model, "crown_ratio", None)
    if callable(f):
        return float(f(p))
    w = float(model.width_mm(p))
    return arc_ratio_for_area_ratio(float(model.area_mm2(p)) / (w * w)) if w > 0.0 else 0.0


@dataclass(frozen=True)
class PassParams:
    """패스 하나의 공정 조건 (WPS 의 역할별 값 + 자세). settings = 장비 설정값 등 ((키, 값) 쌍) — 데이터 기반 모델의 입력."""
    role: str                       # root | fill | cap
    wfs_mpm: float                  # 와이어 송급 속도 [m/min]
    speed_mps: float                # 진행 속도 [m/s]
    travel_angle_deg: float = 0.0
    work_angle_deg: float = 0.0
    weave_amp_mm: float = 0.0
    position: str = "1G"
    settings: tuple = field(default_factory=tuple)

    def with_speed(self, speed_mps: float) -> "PassParams":
        return PassParams(self.role, self.wfs_mpm, float(speed_mps), self.travel_angle_deg, self.work_angle_deg, self.weave_amp_mm,
                          self.position, self.settings)


@runtime_checkable
class BeadModel(Protocol):
    def area_mm2(self, p: PassParams) -> float: ...

    def width_mm(self, p: PassParams) -> float: ...

    def speed_for_area(self, p: PassParams, area_mm2: float) -> float: ...


@dataclass
class MassBalanceBeadModel:
    """물리 기본 모델 (KIRO 데이터 전).

    단면적 (질량 보존): 송급된 와이어 부피 중 η 가 용착된다 → a = η · (π d²/4) · WFS / v.
    폭 (평판 위 원호 단면, 토우각 θ → 높이/폭 κ = tan(θ/2)/2; module docstring): a = g(κ)·w² → w = sqrt(a / g(κ)), 위빙하면 + 2·진폭
    (위빙 비드 = 그 폭의 원호, crown_ratio = g⁻¹(a / W²)).
    속도 역산: WFS 고정 (같은 용착률) 에서 v = η·(π d²/4)·WFS / a.
    가정 값 η, θ 는 데이터로 바꾼다 (자세·각도 의존성은 이 모델에 없다 — 데이터 기반 모델이 채울 자리)."""
    wire_diameter_mm: float = 1.2
    deposition_efficiency: float = 0.95
    toe_angle_deg: float = 30.0

    def __post_init__(self):
        if not (self.wire_diameter_mm > 0.0 and 0.0 < self.deposition_efficiency <= 1.0 and 0.0 < self.toe_angle_deg < 180.0):
            raise ValueError("bead model: wire diameter > 0, 0 < deposition efficiency <= 1, 0 < toe angle < 180 deg")

    @property
    def aspect_ratio(self) -> float:
        """The free bead's height / width (kappa) of the toe angle."""
        return crown_ratio_for_toe_angle(self.toe_angle_deg)

    @property
    def wire_area_mm2(self) -> float:
        return math.pi * self.wire_diameter_mm ** 2 / 4.0

    def _rate_mm3_s(self, p: PassParams) -> float:
        return self.deposition_efficiency * self.wire_area_mm2 * float(p.wfs_mpm) * 1000.0 / 60.0

    def area_mm2(self, p: PassParams) -> float:
        return self._rate_mm3_s(p) / (float(p.speed_mps) * 1e3)

    def width_mm(self, p: PassParams) -> float:
        return math.sqrt(self.area_mm2(p) / arc_area_ratio(self.aspect_ratio)) + 2.0 * float(p.weave_amp_mm)

    def crown_ratio(self, p: PassParams) -> float:
        """Height over the toe chord / chord of the pass's free bead: kappa without weave, flatter with weave (the same area over
        the woven width)."""
        if float(p.weave_amp_mm) == 0.0:
            return float(self.aspect_ratio)
        w = self.width_mm(p)
        return arc_ratio_for_area_ratio(self.area_mm2(p) / (w * w))

    def speed_for_area(self, p: PassParams, area_mm2: float) -> float:
        return self._rate_mm3_s(p) / max(float(area_mm2), 1e-9) * 1e-3

    def describe(self) -> dict:
        return {"model": "mass_balance", "wire_diameter_mm": self.wire_diameter_mm, "deposition_efficiency": self.deposition_efficiency,
                "toe_angle_deg": self.toe_angle_deg, "aspect_ratio": self.aspect_ratio,
                "section": "circular-arc segment meeting its toe chord at the toe angle (height / width = tan(toe / 2) / 2)"}
