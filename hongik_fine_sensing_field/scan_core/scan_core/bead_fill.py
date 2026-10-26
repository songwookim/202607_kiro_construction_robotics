"""다층 멀티패스 충전 상태의 단면 형상 정의 (KIRO 공정 2026-10-03: 패스 1–3 충전 + 패스 4 위빙 캡).

**정의 하나로 세 가지를 만든다** (부분 충전 그루브 추출기 개발용, 사용자 2026-10-03 — 1·2·3 패스 시편이 없어 3D 프린트 인서트와
합성 데이터로 개발):
  1. 합성 스캔 장면 — FilledGrooveScene (synth.GrooveScene 확장; simulate_scan 이 그대로 광선 추적).
  2. 3D 프린트 인서트 — insert_mesh / write_stl (실제 그루브에 끼워 넣는 조각; 측정한 그루브 치수에 맞춘다).
  3. 추출기 진실값 — FillLayout.truth_at (남은 단면적·캡 보강 단면적·토우 위치·충전 높이).

좌표 (GrooveSpec.nominal_cross_section 과 같다): 단면 x [mm] = 루트갭 중심 0 (+x = 판 프레임 +y), 높이 z [mm] = 뒷댐재 상면 0,
판 상면 t (= spec.thickness_mm). 심 방향 s [mm] = 레이아웃 기준 (S 판 끝 0 → E).

충전 표면 (패스 k 뒤, 위에 보이는 면): 양 벽에 닿는 토우 두 점을 잇는 직선 + 크라운 (꼭대기 위치 peak_frac 의 두 조각 포물선;
+ 볼록 / − 오목) + 심 방향 비드 물결. 캡 (패스 4 위빙): 상부 모서리 바깥으로 overlap 만큼 판 상면을 덮고 판 상면 위
reinforcement 까지 올라오는 볼록면 + 위빙 셰브런 무늬. 수치 기본값 (KIRO_FILL_FRACTIONS 등) 은 KIRO 데이터셋 전의 가정이다.
"""
from __future__ import annotations

import dataclasses
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .specimen import GrooveSpec
from .synth import GrooveScene

__all__ = ["KIRO_FILL_FRACTIONS", "FillSurface", "CapSurface", "FillSegment", "FillLayout", "FilledGrooveScene",
           "wall_x_mm", "groove_area_below_mm2", "level_for_area_fraction", "kiro_pass_states", "print_layout",
           "tailor_to_measured", "insert_ring_mm", "insert_mesh", "mesh_check", "write_stl"]

# 패스 1–3 뒤 그루브 단면적 비율 (10T·30°·갭 5 에서 토우 높이 ≈ 3.5 / 6.5 / 9.0 mm; 남은 ~13 % + 보강 = 위빙 캡). KIRO 데이터셋 전 가정.
KIRO_FILL_FRACTIONS = (0.27, 0.57, 0.87)


# --------------------------------------------------------------------------------------
# 그루브 벽
# --------------------------------------------------------------------------------------
def _bevelled(spec: GrooveSpec) -> tuple:
    side = spec.resolved_bevel_side()
    bl = spec.kind == "single_v" or (spec.kind == "single_bevel" and side == "left")
    br = spec.kind == "single_v" or (spec.kind == "single_bevel" and side == "right")
    return bl, br


def wall_x_mm(spec: GrooveSpec, z_mm) -> tuple:
    """높이 z [mm] 에서 왼쪽·오른쪽 벽의 x [mm] (베벨면은 루트페이스 위에서 바깥으로 기운다; 직각 벽은 ±gap/2)."""
    z = np.asarray(z_mm, dtype=float)
    g = 0.5 * spec.root_gap_mm
    run = np.clip(z - spec.root_face_mm, 0.0, None) * math.tan(math.radians(spec.bevel_angle_deg))
    bl, br = _bevelled(spec)
    return -g - (run if bl else 0.0 * run), g + (run if br else 0.0 * run)


def groove_area_below_mm2(spec: GrooveSpec, z_mm) -> np.ndarray:
    """뒷댐재 상면 0 부터 높이 z 까지 두 벽 사이 단면적 [mm²] (z 는 [0, t] 로 자른다)."""
    z = np.clip(np.asarray(z_mm, dtype=float), 0.0, spec.thickness_mm)
    r = np.clip(z - spec.root_face_mm, 0.0, None)
    nb = sum(_bevelled(spec))
    return spec.root_gap_mm * z + nb * math.tan(math.radians(spec.bevel_angle_deg)) * r * r / 2.0


def level_for_area_fraction(spec: GrooveSpec, frac: float) -> float:
    """그루브 단면적의 frac 을 채우는 수평 높이 [mm] (이분법)."""
    target = float(frac) * float(groove_area_below_mm2(spec, spec.thickness_mm))
    lo, hi = 0.0, spec.thickness_mm
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if groove_area_below_mm2(spec, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _peak_shape(u, p: float) -> np.ndarray:
    """u ∈ [0, 1] 에서 0 → 1 (u = p) → 0 인 두 조각 포물선 (꼭대기 위치 p)."""
    p = min(max(float(p), 1e-3), 1.0 - 1e-3)
    u = np.asarray(u, dtype=float)
    a = np.where(u <= p, (u - p) / p, (u - p) / (1.0 - p))
    return 1.0 - a * a


# --------------------------------------------------------------------------------------
# 충전 표면
# --------------------------------------------------------------------------------------
@dataclass
class FillSurface:
    """충전 패스 (1–3) 뒤 그루브 안의 위 표면. level/tilt 는 토우 (표면이 벽에 닿는 점) 높이 [mm, 뒷댐재 상면 기준]."""
    pass_index: int = 1
    level_mm: float = 3.5          # 두 토우의 평균 높이 (세그먼트 시작 s = 0 에서)
    crown_mm: float = 0.0          # 토우를 잇는 선 위 크라운 (+ 볼록 / − 오목)
    peak_frac: float = 0.5         # 크라운 꼭대기 위치: 왼쪽 토우 0 → 오른쪽 토우 1
    tilt_mm: float = 0.0           # 오른쪽 토우 − 왼쪽 토우 높이 (비드가 한쪽 벽으로 쏠림)
    level_ramp_mm: float = 0.0     # 세그먼트 끝에서 level 변화 (심 방향 선형 — 충전량 변화)
    ripple_amp_mm: float = 0.0     # 심 방향 비드 물결 진폭 (토우에서 0)
    ripple_pitch_mm: float = 2.5
    kind: str = "fill"

    def level_at(self, s_rel, seg_len_mm: float) -> np.ndarray:
        f = np.clip(np.asarray(s_rel, dtype=float) / seg_len_mm, 0.0, 1.0) if seg_len_mm > 0 else 0.0
        return self.level_mm + self.level_ramp_mm * f

    def toes_mm(self, spec: GrooveSpec, s_rel, seg_len_mm: float) -> tuple:
        """(x_left, z_left, x_right, z_right) [mm] — 토우 (표면 ∩ 벽)."""
        lev = self.level_at(s_rel, seg_len_mm)
        zl, zr = lev - 0.5 * self.tilt_mm, lev + 0.5 * self.tilt_mm
        return wall_x_mm(spec, zl)[0], zl, wall_x_mm(spec, zr)[1], zr

    def height_mm(self, spec: GrooveSpec, x_mm, s_rel, seg_len_mm: float) -> np.ndarray:
        x, s = np.broadcast_arrays(np.asarray(x_mm, dtype=float), np.asarray(s_rel, dtype=float))
        xl, zl, xr, zr = self.toes_mm(spec, s, seg_len_mm)
        u = (x - xl) / (xr - xl)
        inside = (u >= 0.0) & (u <= 1.0)
        uc = np.clip(u, 0.0, 1.0)
        h = zl + (zr - zl) * uc + self.crown_mm * _peak_shape(uc, self.peak_frac)
        if self.ripple_amp_mm:
            h = h + self.ripple_amp_mm * np.sin(2.0 * math.pi * s / self.ripple_pitch_mm) * 4.0 * uc * (1.0 - uc)
        return np.where(inside, h, np.nan)

    def max_height_mm(self, seg_len_mm: float) -> float:
        lev = max(self.level_mm, self.level_mm + self.level_ramp_mm)
        return lev + 0.5 * abs(self.tilt_mm) + max(self.crown_mm, 0.0) + abs(self.ripple_amp_mm)


@dataclass
class CapSurface:
    """캡 패스 (4, 위빙): 상부 모서리 바깥으로 overlap 만큼 판 상면을 덮고 판 상면 위 reinforcement 까지. min_edge_mm > 0 이면
    보강 높이가 그보다 얇은 가장자리 (토우 쪽 칼날) 를 잘라 낸다 — 인쇄 인서트는 칼날 가장자리를 만들 수 없다 (진실값도 같은 정의)."""
    pass_index: int = 4
    reinforcement_mm: float = 1.5  # 판 상면 위 최대 높이
    overlap_mm: float = 1.5        # 상부 모서리 바깥 판 상면을 덮는 폭 (양쪽)
    peak_frac: float = 0.5
    ripple_amp_mm: float = 0.15    # 위빙 셰브런 무늬 진폭 (가장자리에서 0)
    ripple_pitch_mm: float = 3.0
    chevron_slope: float = 0.5     # 무늬 위상 = s + slope·|x − 중심| (무늬 선이 V 자로 휜다)
    min_edge_mm: float = 0.0
    kind: str = "cap"

    def extent_mm(self, spec: GrooveSpec) -> tuple:
        xl, xr = wall_x_mm(spec, spec.thickness_mm)
        return float(xl) - self.overlap_mm, float(xr) + self.overlap_mm

    def height_mm(self, spec: GrooveSpec, x_mm, s_rel, seg_len_mm: float) -> np.ndarray:
        x, s = np.broadcast_arrays(np.asarray(x_mm, dtype=float), np.asarray(s_rel, dtype=float))
        xl, xr = self.extent_mm(spec)
        u = (x - xl) / (xr - xl)
        inside = (u >= 0.0) & (u <= 1.0)
        uc = np.clip(u, 0.0, 1.0)
        bump = self.reinforcement_mm * _peak_shape(uc, self.peak_frac)
        h = spec.thickness_mm + spec.hi_lo_mm * uc + bump
        if self.ripple_amp_mm:
            xc = 0.5 * (xl + xr)
            h = h + self.ripple_amp_mm * np.cos(2.0 * math.pi * (s + self.chevron_slope * np.abs(x - xc)) / self.ripple_pitch_mm) \
                * 4.0 * uc * (1.0 - uc)
        if self.min_edge_mm > 0.0:
            inside = inside & (bump >= self.min_edge_mm)
        return np.where(inside, h, np.nan)

    def max_height_mm(self, seg_len_mm: float = 0.0) -> float:
        return self.reinforcement_mm + abs(self.ripple_amp_mm)


Surface = Union[FillSurface, CapSurface]


def _surface_from_dict(d: Optional[dict]) -> Optional[Surface]:
    if not d:
        return None
    d = dict(d)
    cls = CapSurface if d.get("kind") == "cap" else FillSurface
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


# --------------------------------------------------------------------------------------
# 레이아웃
# --------------------------------------------------------------------------------------
@dataclass
class FillSegment:
    """심 방향 구간 [s0, s1) 의 충전 상태 (surface None = 빈 그루브). spec: 이 구간의 실제 그루브 치수 (인쇄 맞춤; None = 레이아웃 spec)."""
    s0_mm: float
    s1_mm: float
    surface: Optional[Surface] = None
    label: str = ""
    spec: Optional[GrooveSpec] = None

    @property
    def length_mm(self) -> float:
        return float(self.s1_mm - self.s0_mm)

    def to_dict(self) -> dict:
        return {"s0_mm": self.s0_mm, "s1_mm": self.s1_mm, "label": self.label,
                "surface": dataclasses.asdict(self.surface) if self.surface is not None else None,
                "spec": self.spec.to_dict() if self.spec is not None else None}

    @classmethod
    def from_dict(cls, d: dict) -> "FillSegment":
        return cls(s0_mm=float(d["s0_mm"]), s1_mm=float(d["s1_mm"]), label=str(d.get("label", "")),
                   surface=_surface_from_dict(d.get("surface")),
                   spec=GrooveSpec.from_dict(d["spec"]) if d.get("spec") else None)


@dataclass
class FillLayout:
    """시편 하나의 충전 상태 배치. spatter: [(s_mm, x_mm, r_mm)] 표면 위 반구 (합성 전용 — 인쇄 해상도 밖)."""
    spec: GrooveSpec = field(default_factory=GrooveSpec)
    segments: list = field(default_factory=list)
    spatter: list = field(default_factory=list)

    def spec_of(self, seg: Optional[FillSegment]) -> GrooveSpec:
        return seg.spec if (seg is not None and seg.spec is not None) else self.spec

    def segment_at(self, s_mm: float) -> Optional[FillSegment]:
        for seg in self.segments:
            if seg.s0_mm <= s_mm < seg.s1_mm:
                return seg
        return None

    def _base_height_mm(self, x, s) -> np.ndarray:
        h = np.array(self.spec.nominal_cross_section(x), dtype=float, copy=True)
        for seg in self.segments:
            m = (s >= seg.s0_mm) & (s < seg.s1_mm)
            if not np.any(m):
                continue
            sp = self.spec_of(seg)
            hn = np.asarray(sp.nominal_cross_section(x[m]), dtype=float)
            if seg.surface is not None:
                hf = seg.surface.height_mm(sp, x[m], s[m] - seg.s0_mm, seg.length_mm)
                hn = np.where(np.isfinite(hf), np.fmax(hn, hf), hn)
            h[m] = hn
        return h

    def height_mm(self, x_mm, s_mm) -> np.ndarray:
        """단면 x, 심 방향 s → 표면 높이 [mm] (뒷댐재 상면 0; 그루브·충전·캡·스패터)."""
        x, s = np.broadcast_arrays(np.asarray(x_mm, dtype=float), np.asarray(s_mm, dtype=float))
        x, s = np.array(x, copy=True), np.array(s, copy=True)
        h = self._base_height_mm(x, s)
        for s0, x0, r in self.spatter:
            d2 = (x - x0) ** 2 + (s - s0) ** 2
            m = d2 < r * r
            if np.any(m):
                hb = float(self._base_height_mm(np.array([x0], float), np.array([s0], float))[0])
                h[m] = np.fmax(h[m], hb + np.sqrt(r * r - d2[m]))
        return h

    def truth_at(self, s_mm: float, dx_mm: float = 0.005) -> dict:
        """심 위치 s 의 진실값: 남은 단면적 (판 상면선 아래·두 벽 사이의 빈 곳), 캡 보강 단면적 (판 상면선 위), 토우, 충전 높이."""
        seg = self.segment_at(float(s_mm))
        sp = self.spec_of(seg)
        t = sp.thickness_mm
        xl_t, xr_t = (float(v) for v in wall_x_mm(sp, t))
        x = np.arange(xl_t - 8.0, xr_t + 8.0 + dx_mm, dx_mm)
        h = self.height_mm(x, np.full_like(x, float(s_mm)))
        top = t + np.where(x > 0.0, sp.hi_lo_mm, 0.0)                   # 판 상면선 (nominal_cross_section 과 같은 규약)
        within = (x >= xl_t) & (x <= xr_t)
        hh = np.where(np.isfinite(h), h, 0.0)
        out = {"s_mm": float(s_mm), "segment": seg.label if seg is not None else None,
               "kind": (seg.surface.kind if (seg is not None and seg.surface is not None) else "empty"),
               "pass_index": (seg.surface.pass_index if (seg is not None and seg.surface is not None) else 0),
               "groove_area_mm2": float(groove_area_below_mm2(sp, t)),
               "remaining_area_mm2": float(np.sum(np.clip(top - hh, 0.0, None)[within]) * dx_mm),
               "reinforcement_area_mm2": float(np.sum(np.clip(hh - top, 0.0, None)) * dx_mm),
               "fill_level_mm": None, "toes_mm": None}
        if seg is not None and seg.surface is not None:
            srel = float(s_mm) - seg.s0_mm
            if seg.surface.kind == "fill":
                xl, zl, xr, zr = (float(v) for v in seg.surface.toes_mm(sp, srel, seg.length_mm))
                out["fill_level_mm"] = float(seg.surface.level_at(srel, seg.length_mm))
                out["toes_mm"] = [[xl, zl], [xr, zr]]
            else:
                hf = seg.surface.height_mm(sp, x, np.full_like(x, srel), seg.length_mm)
                ok = np.isfinite(hf)
                if np.any(ok):
                    i0, i1 = int(np.flatnonzero(ok)[0]), int(np.flatnonzero(ok)[-1])
                    out["toes_mm"] = [[float(x[i0]), float(top[i0])], [float(x[i1]), float(top[i1])]]
        return out

    def to_dict(self) -> dict:
        return {"spec": self.spec.to_dict(), "segments": [s.to_dict() for s in self.segments],
                "spatter": [list(map(float, b)) for b in self.spatter]}

    @classmethod
    def from_dict(cls, d: dict) -> "FillLayout":
        return cls(spec=GrooveSpec.from_dict(d["spec"]), segments=[FillSegment.from_dict(s) for s in d.get("segments", [])],
                   spatter=[tuple(map(float, b)) for b in d.get("spatter", [])])

    def add_random_spatter(self, n: int, *, r_range=(0.25, 0.6), x_halfwidth_mm: float = 15.0, seed: int = 0) -> "FillLayout":
        """표면 위 무작위 스패터 반구 n 개 (판 상면·비드 위; 합성 전용)."""
        rng = np.random.default_rng(seed)
        L0 = min((s.s0_mm for s in self.segments), default=0.0)
        L1 = max((s.s1_mm for s in self.segments), default=self.spec.weld_length_mm)
        for _ in range(int(n)):
            self.spatter.append((float(rng.uniform(L0, L1)), float(rng.uniform(-x_halfwidth_mm, x_halfwidth_mm)),
                                 float(rng.uniform(*r_range))))
        return self


# --------------------------------------------------------------------------------------
# 합성 장면
# --------------------------------------------------------------------------------------
@dataclass
class FilledGrooveScene(GrooveScene):
    """GrooveScene + 충전 레이아웃. 심 방향 s [mm] = (그루브 축 진행 좌표 − s_origin_m)·1e3·s_sign (기본: x_range 시작 = S 끝)."""
    layout: Optional[FillLayout] = None
    s_origin_m: Optional[float] = None
    s_sign: float = 1.0

    @classmethod
    def from_layout(cls, layout: FillLayout, yc: float = 0.0, angle_deg: float = 0.0, h_top: float = 0.0,
                    centered: bool = True) -> "FilledGrooveScene":
        base = GrooveScene.from_spec(layout.spec, yc=yc, angle_deg=angle_deg, h_top=h_top, centered=centered)
        return cls(spec=layout.spec, yc=yc, angle_deg=angle_deg, h_top=h_top, x_range=base.x_range, y_range=base.y_range,
                   outside=base.outside, layout=layout)

    def along(self, xp, yp) -> np.ndarray:
        a = math.radians(self.angle_deg)
        return np.asarray(xp, dtype=float) * math.cos(a) + (np.asarray(yp, dtype=float) - self.yc) * math.sin(a)

    def s_mm(self, xp, yp) -> np.ndarray:
        s0 = self.x_range[0] if self.s_origin_m is None else self.s_origin_m
        if not np.isfinite(s0):
            s0 = 0.0
        return (self.along(xp, yp) - s0) * 1e3 * self.s_sign

    def height(self, xp, yp):
        if self.layout is None:
            return super().height(xp, yp)
        xp = np.asarray(xp, dtype=float)
        yp = np.asarray(yp, dtype=float)
        h = self.h_top - self.spec.thickness_mm * 1e-3 + self.layout.height_mm(self.lateral(xp, yp) * 1e3, self.s_mm(xp, yp)) * 1e-3
        inside = (xp >= self.x_range[0]) & (xp <= self.x_range[1]) & (yp >= self.y_range[0]) & (yp <= self.y_range[1])
        return np.where(inside, h, self.outside)


# --------------------------------------------------------------------------------------
# 공정 상태
# --------------------------------------------------------------------------------------
def kiro_pass_states(spec: GrooveSpec, *, fractions=KIRO_FILL_FRACTIONS, crowns=(0.6, 0.4, 0.3),
                     cap: Optional[CapSurface] = None) -> list:
    """KIRO 공정 (패스 1–3 충전 + 패스 4 위빙 캡) 의 공칭 상태 [FillSurface ×3, CapSurface] — 데이터셋 전 가정값."""
    out = [FillSurface(pass_index=k + 1, level_mm=level_for_area_fraction(spec, f), crown_mm=c, ripple_amp_mm=0.1)
           for k, (f, c) in enumerate(zip(fractions, crowns))]
    out.append(cap if cap is not None else CapSurface())
    return out


def print_layout(spec: Optional[GrooveSpec] = None, *, segment_mm: float = 47.5, n_segments: int = 8) -> FillLayout:
    """3D 프린트 인서트 1 벌 (S 끝부터 세그먼트 8 개 × 47.5 mm = 380 mm): 패스 1·2·3·캡 상태를 두 개씩, 모양을 바꿔 가며.

      1 P1a 볼록 (+0.8)             2 P1b 오목 (−0.3) + 충전 높이 경사 (+1.0 mm)
      3 P2a 볼록 (+0.6), 꼭대기가 베벨 쪽   4 P2b 평평, 베벨 쪽 토우가 0.6 mm 높음 (기운 비드)
      5 P3a 볼록 (+0.5)             6 P3b 오목 (−0.3) + 충전 높이 경사 (−0.8 mm)
      7 CAPa 위빙 캡 보강 1.5 / 덮음 3.0   8 CAPb 위빙 캡 보강 2.2 / 덮음 3.0, 꼭대기가 베벨 쪽
    충전 패스는 비드 물결 0.1 mm / 2.5 mm, 캡은 셰브런 0.15 mm / 3 mm. 캡 가장자리는 노즐 한 줄 (0.45 mm) 보다 얇은 칼날을 잘라
    낸다 (인쇄) — 그래서 덮는 폭을 넉넉히 해 날개가 판 상면에 1.4 mm 이상 걸리게 한다 (판 모서리에서 흔들리지 않게)."""
    spec = spec or GrooveSpec(bevel_side="left")
    lv = [level_for_area_fraction(spec, f) for f in KIRO_FILL_FRACTIONS]
    bevel_left = spec.resolved_bevel_side() == "left"
    toward_bevel = 0.35 if bevel_left else 0.65                     # 꼭대기를 베벨 쪽으로 (u = 0 은 왼쪽 토우)
    tilt_bevel_up = -0.6 if bevel_left else 0.6                     # tilt = 오른쪽 − 왼쪽 토우 높이
    rip = dict(ripple_amp_mm=0.1, ripple_pitch_mm=2.5)
    states = [
        ("P1a_convex", FillSurface(1, lv[0], crown_mm=0.8, **rip)),
        ("P1b_concave_ramp", FillSurface(1, lv[0], crown_mm=-0.3, level_ramp_mm=1.0, **rip)),
        ("P2a_convex_bevel_peak", FillSurface(2, lv[1], crown_mm=0.6, peak_frac=toward_bevel, **rip)),
        ("P2b_flat_tilted", FillSurface(2, lv[1], crown_mm=0.0, tilt_mm=tilt_bevel_up, **rip)),
        ("P3a_convex", FillSurface(3, lv[2], crown_mm=0.5, **rip)),
        ("P3b_concave_ramp", FillSurface(3, lv[2], crown_mm=-0.3, level_ramp_mm=-0.8, **rip)),
        ("CAPa_weave", CapSurface(reinforcement_mm=1.5, overlap_mm=3.0, min_edge_mm=0.45)),
        ("CAPb_weave_high", CapSurface(reinforcement_mm=2.2, overlap_mm=3.0, peak_frac=0.4 if bevel_left else 0.6,
                                       min_edge_mm=0.45)),
    ][:int(n_segments)]
    segs = [FillSegment(s0_mm=i * segment_mm, s1_mm=(i + 1) * segment_mm, surface=srf, label=f"{i + 1}_{name}")
            for i, (name, srf) in enumerate(states)]
    return FillLayout(spec=spec, segments=segs)


def tailor_to_measured(layout: FillLayout, bins: list) -> FillLayout:
    """세그먼트마다 실측 그루브 치수 (S 끝 기준 구간 중앙값, auto-teach specimen_report 의 bins: from_S_mm/to_S_mm/root_gap_mm/
    top_opening_mm/depth_mm) 로 spec 을 맞춘다 — 겹치는 구간의 **최소** 갭·개구·깊이 (인서트가 어디서든 들어가고, 매달린 캡이 바닥에
    닿지 않게). 베벨각 = atan((개구 − 갭)/깊이) (거리로 계산 — 각도 측정보다 강하다). 구간이 없으면 레이아웃 spec 그대로."""
    out = []
    for seg in layout.segments:
        rows = [b for b in bins if b.get("from_S_mm") is not None and b["to_S_mm"] > seg.s0_mm and b["from_S_mm"] < seg.s1_mm
                and all(np.isfinite(b.get(k, float("nan"))) for k in ("root_gap_mm", "top_opening_mm", "depth_mm"))]
        sp = layout.spec
        if rows:
            g = min(b["root_gap_mm"] for b in rows)
            o = min(b["top_opening_mm"] for b in rows)
            d = min(b["depth_mm"] for b in rows)
            nb = max(1, sum(_bevelled(sp)))
            bev = math.degrees(math.atan(max(o - g, 0.0) / nb / d))
            sp = dataclasses.replace(sp, root_gap_mm=float(g), thickness_mm=float(d), bevel_angle_deg=float(bev), hi_lo_mm=0.0,
                                     notes=[f"tailored to measured bins {[round(b['from_S_mm']) for b in rows]} mm from S: min gap "
                                            f"{g:.2f}, min opening {o:.2f}, min depth {d:.2f}"])
        out.append(dataclasses.replace(seg, spec=sp))
    return FillLayout(spec=layout.spec, segments=out, spatter=list(layout.spatter))


# --------------------------------------------------------------------------------------
# 3D 프린트 인서트
# --------------------------------------------------------------------------------------
def _wall_chain(spec: GrooveSpec, side: int, z0: float, z1: float, inset: float) -> list:
    """벽 (side −1 왼쪽 / +1 오른쪽) 을 z0 → z1 로 따라가는 점 (안쪽으로 inset) — 루트페이스 꺾임이 사이에 있으면 넣는다."""
    zs = [z0]
    rf = spec.root_face_mm
    if min(z0, z1) < rf < max(z0, z1):
        zs.append(rf)
    zs.append(z1)
    pts = []
    for z in zs:
        xl, xr = wall_x_mm(spec, z)
        pts.append((float(xl) + inset, z) if side < 0 else (float(xr) - inset, z))
    return pts


def insert_ring_mm(layout: FillLayout, seg: FillSegment, s_rel: float, *, clearance_mm: float = 0.2, chamfer_mm: float = 0.5,
                   n_top: int = 81, cap_floor_gap_mm: float = 0.3) -> np.ndarray:
    """세그먼트의 심 위치 s_rel 에서 인서트 단면 다각형 (K, 2) [(x, z) mm, 반시계]. 벽에서 clearance 만큼 안쪽, 아래 모서리 모따기.
    충전 인서트: 뒷댐재 위에 놓인다 (바닥 z = 0). 캡 인서트: 날개가 판 상면 (z = t) 에 걸리고 그루브 안 부분은 바닥에서
    cap_floor_gap 만큼 떠 있다 (두 곳에 동시에 닿아 흔들리지 않게)."""
    sp = layout.spec_of(seg)
    srf = seg.surface
    if srf is None:
        raise ValueError(f"segment {seg.label!r} has no fill surface (empty groove): nothing to print")
    c, ch, L = float(clearance_mm), float(chamfer_mm), seg.length_mm
    if srf.kind == "fill":
        xl, zl, xr, zr = (float(v) for v in srf.toes_mm(sp, s_rel, L))
        x0, x1 = xl + c, xr - c
        for _ in range(8):                      # 끝점 높이의 벽에서 정확히 c 안쪽 (오목 비드는 토우 옆이 토우보다 낮다 — 베벨벽은 아래로 좁아진다)
            h0, h1 = (float(v) for v in srf.height_mm(sp, np.array([x0, x1]), float(s_rel), L))
            x0 = max(x0, float(wall_x_mm(sp, h0)[0]) + c)
            x1 = min(x1, float(wall_x_mm(sp, h1)[1]) - c)
        xs = np.linspace(x0, x1, int(n_top))
        hs = srf.height_mm(sp, xs, np.full(xs.shape, float(s_rel)), L)
        if not np.all(np.isfinite(hs)) or float(np.min(hs)) <= ch + 0.2:
            raise ValueError(f"segment {seg.label!r}: insert top too low for the chamfer ({np.nanmin(hs):.2f} mm)")
        zb = 0.0
        right_up = _wall_chain(sp, +1, zb + ch, float(hs[-1]), c)[:-1]
        left_down = _wall_chain(sp, -1, float(hs[0]), zb + ch, c)[1:]
        top = list(zip(xs[::-1].tolist(), hs[::-1].tolist()))
        wings_r, wings_l = [], []
    else:
        t = sp.thickness_mm
        xe0, xe1 = srf.extent_mm(sp)
        xx = np.linspace(xe0, xe1, 4001)
        hf = srf.height_mm(sp, xx, np.full(xx.shape, float(s_rel)), L)
        ok = np.flatnonzero(np.isfinite(hf))
        x_tl, x_tr = float(xx[ok[0]]), float(xx[ok[-1]])
        xw_l, xw_r = (float(v) for v in wall_x_mm(sp, t))
        if not (x_tl < xw_l + c and x_tr > xw_r - c):
            raise ValueError(f"segment {seg.label!r}: cap narrower than the opening")
        xs = np.linspace(x_tl, x_tr, int(n_top))
        hs = srf.height_mm(sp, xs, np.full(xs.shape, float(s_rel)), L)
        hs = np.where(np.isfinite(hs), hs, t + srf.min_edge_mm)
        zb = float(cap_floor_gap_mm)
        right_up = _wall_chain(sp, +1, zb + ch, t, c)
        wings_r = [(x_tr, t)]
        top = list(zip(xs[::-1].tolist(), hs[::-1].tolist()))
        wings_l = [(x_tl, t)]
        left_down = _wall_chain(sp, -1, t, zb + ch, c)
    xl0, xr0 = (float(v) for v in wall_x_mm(sp, zb))
    bottom = [(xl0 + c + ch, zb), (xr0 - c - ch, zb)]
    ring = bottom + right_up + wings_r + top + wings_l + left_down
    return np.asarray(ring, dtype=float)


def _ear_clip(poly: np.ndarray) -> list:
    """단순 다각형 (반시계) 귀 자르기 삼각분할 → 정점 인덱스 삼각형 목록 (반시계). 일직선 정점은 넓이 0 귀로 처리 (T 접합 없음)."""
    P = np.asarray(poly, dtype=float)
    idx = list(range(P.shape[0]))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c_):
        d1, d2, d3 = cross(a, b, p), cross(b, c_, p), cross(c_, a, p)
        return d1 >= -1e-12 and d2 >= -1e-12 and d3 >= -1e-12

    scale = float(np.ptp(P[:, 0]) * np.ptp(P[:, 1])) or 1.0
    eps = 1e-12 * scale
    while len(idx) > 3:
        n = len(idx)
        crs = [cross(P[idx[k - 1]], P[idx[k]], P[idx[(k + 1) % n]]) for k in range(n)]
        reflex = [idx[k] for k in range(n) if crs[k] < -eps]          # 귀 안에 들어갈 수 있는 정점은 오목 정점뿐
        chosen = None
        for k in range(n):                                            # 1) 볼록 귀 (안에 오목 정점 없음)
            if crs[k] <= eps:
                continue
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            if any(inside(P[m], P[i0], P[i1], P[i2]) for m in reflex if m not in (i0, i1, i2)):
                continue
            chosen = k
            break
        if chosen is None:                                            # 2) 일직선 정점 (넓이 0 귀)
            chosen = next((k for k in range(n) if abs(crs[k]) <= eps), None)
        if chosen is None:
            raise ValueError("ear clipping failed (polygon not simple?)")
        k = chosen
        tris.append((idx[k - 1], idx[k], idx[(k + 1) % n]))
        idx.pop(k)
    tris.append(tuple(idx))
    return tris


def insert_mesh(layout: FillLayout, seg: FillSegment, *, ds_mm: Optional[float] = None, **ring_kw) -> tuple:
    """세그먼트 인서트의 닫힌 삼각망 (V (n,3), F (m,3)) [mm]. 좌표: X = 단면 x, Y = 높이 z, Z = 심 방향 s (세그먼트 시작 0) —
    단면을 출력 평면 (XY) 에 둔 **세워 출력** 방향 (Z = 0 첫 층 = S 쪽 끝). 법선은 바깥 (부피 > 0)."""
    srf = seg.surface
    L = seg.length_mm
    if ds_mm is None:
        pitch = float(getattr(srf, "ripple_pitch_mm", 0.0) or 0.0)
        ds_mm = min(0.5, pitch / 10.0) if (getattr(srf, "ripple_amp_mm", 0.0) and pitch > 0) else 0.5
        if getattr(srf, "level_ramp_mm", 0.0):
            ds_mm = min(ds_mm, 0.5)
    n_s = max(2, int(math.ceil(L / float(ds_mm))) + 1)
    ss = np.linspace(0.0, L, n_s)
    rings = [insert_ring_mm(layout, seg, float(s), **ring_kw) for s in ss]
    K = rings[0].shape[0]
    if any(r.shape[0] != K for r in rings):
        raise ValueError("ring vertex count changes along the segment")
    V = np.concatenate([np.column_stack([r[:, 0], r[:, 1], np.full(K, s)]) for r, s in zip(rings, ss)])
    i = np.arange(n_s - 1)[:, None]
    j = np.arange(K)[None, :]
    a = (i * K + j).ravel()
    b = (i * K + (j + 1) % K).ravel()
    c_ = ((i + 1) * K + (j + 1) % K).ravel()
    d = ((i + 1) * K + j).ravel()
    F = [np.column_stack([a, b, c_]), np.column_stack([a, c_, d])]
    cap0 = np.asarray(_ear_clip(rings[0]), dtype=np.int64)
    cap1 = np.asarray(_ear_clip(rings[-1]), dtype=np.int64)
    F.append(cap0[:, ::-1])                                       # Z = 0 면: 바깥 법선 −Z
    F.append(cap1 + (n_s - 1) * K)                                # Z = L 면: +Z
    return V, np.concatenate(F).astype(np.int64)


def mesh_check(V: np.ndarray, F: np.ndarray) -> dict:
    """닫힌 다양체 검사 (모든 모서리가 반대 방향으로 정확히 두 번) + 부피 [mm³] (발산정리; 바깥 법선이면 > 0)."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fwd = {tuple(x) for x in e.tolist()}
    n_dup = len(e) - len(fwd)
    unmatched = sum(1 for (p, q) in fwd if (q, p) not in fwd)
    tri = V[F]
    vol = float(np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0)
    return {"watertight": n_dup == 0 and unmatched == 0, "duplicate_edges": int(n_dup), "unmatched_edges": int(unmatched),
            "volume_mm3": vol, "n_triangles": int(F.shape[0]),
            "bbox_mm": [V.min(axis=0).round(3).tolist(), V.max(axis=0).round(3).tolist()]}


def write_stl(path, V: np.ndarray, F: np.ndarray, name: str = "bead_insert") -> Path:
    """이진 STL [mm]."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tri = np.asarray(V, dtype=np.float64)[np.asarray(F)]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 0)
    rec = np.zeros(tri.shape[0], dtype=np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = nrm
    rec["v"] = tri
    with open(p, "wb") as f:
        f.write(name.encode("ascii", "replace")[:80].ljust(80, b" "))
        f.write(struct.pack("<I", tri.shape[0]))
        f.write(rec.tobytes())
    return p
