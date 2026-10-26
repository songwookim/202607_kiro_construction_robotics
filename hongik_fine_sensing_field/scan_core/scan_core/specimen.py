"""시편 모델: 용접 전 그루브 시편 (GrooveSpec) — 공칭 단면·치수·YAML 입출력.

사용자 시편 (2026-09-30): 모재 10T 두 장, 일면개선(한쪽 모서리만 30° 베벨, 반대쪽은 직각), 루트갭 5 mm,
루트페이스 ≈ 0 (있다면 미지), 용접부 아래 뒷댐재 10T 부착, 용접선 길이 380 mm, 모재 폭 300 mm.

단면 좌표 (plate-aligned frame, mm):
  x = 0 : 루트갭 중심,  x < 0 : 'left' 판,  x > 0 : 'right' 판  (라인 방향, 센서 +x 와 같은 쪽으로 두는 것이 관례)
  h    : 뒷댐재 상면(= 그루브 바닥) 기준 높이. 판 상면 = thickness (+ hi_lo 는 right 판에만 적용: hi_lo = right − left).
  베벨 각 bevel_angle_deg 는 판 법선(수직) 기준. 베벨 판의 루트 모서리는 x = ∓gap/2, 상부 코너는 그보다
  (thickness − root_face)·tan(bevel) 만큼 바깥.
  뒷댐재가 없으면(backing=False) 갭 구간은 NaN (표면 없음).

이 모델은 **위치·스케일 관계**(갭·두께·베벨 위치)만 온톨로지로 강제하고, 상세 형상은 관측(seam_features)으로 얻는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

__all__ = ["GrooveSpec", "GROOVE_KINDS", "BEVEL_SIDES", "SIDE_AGNOSTIC", "DEFAULT_QC_TARGETS"]

GROOVE_KINDS = ("single_bevel", "single_v", "square")
# 'unknown' / 'auto' = 베벨 쪽을 모른다(측에 무관하게 처리; 추출기가 프로파일마다 판정). 공칭 단면 생성 시에는 left 로 가정.
BEVEL_SIDES = ("left", "right", "unknown", "auto")
SIDE_AGNOSTIC = ("unknown", "auto")

# 검증 목표 (job YAML qc_targets 기본값): 공칭 ± 허용, hi-lo 는 목표 ≤ target, 초과 경고 warn
DEFAULT_QC_TARGETS = {
    "root_gap_mm": {"tol": 0.5},
    "bevel_angle_deg": {"tol": 2.0},
    "groove_depth_mm": {"tol": 0.5},
    "hi_lo_mm": {"target_max": 1.0, "warn_above": 2.0},
}


@dataclass
class GrooveSpec:
    kind: str = "single_bevel"          # single_bevel | single_v | square
    thickness_mm: float = 10.0          # 모재 두께 t
    bevel_angle_deg: float = 30.0       # 판 법선 기준 베벨각 (single_v 는 양쪽 동일)
    bevel_side: str = "left"            # 베벨 판이 놓인 쪽 ('unknown'/'auto' 이면 측 무관: 단면 생성 시 left 로 가정하고 표시)
    root_gap_mm: float = 5.0
    root_face_mm: float = 0.0           # 미지 → 0 (full bevel)
    backing: bool = True
    backing_thickness_mm: float = 10.0
    weld_length_mm: float = 380.0       # 용접선(시편) 길이
    plate_width_mm: float = 300.0       # 모재 폭 합 (양쪽 판, 심에 수직) — 옛 입력 키 (읽기만; plate_widths_mm 가 있으면 그 합으로 유도)
    hi_lo_mm: float = 0.0               # 공칭 판 상면 단차 (right − left)
    name: str = "groove_10T_single_bevel_30deg_gap5"
    notes: list = field(default_factory=list)
    # 판마다의 폭 [left, right] mm (설계 입력, 심에 수직, 루트 갭 제외; 값 하나 = 두 판 같음). 없으면 옛 합 plate_width_mm 의 반씩
    plate_widths_mm: Optional[list] = None

    def __post_init__(self):
        if self.kind not in GROOVE_KINDS:
            raise ValueError(f"kind must be one of {GROOVE_KINDS}, got {self.kind!r}")
        if self.bevel_side not in BEVEL_SIDES:
            raise ValueError(f"bevel_side must be one of {BEVEL_SIDES}, got {self.bevel_side!r}")
        for k in ("thickness_mm", "root_gap_mm", "root_face_mm", "backing_thickness_mm", "weld_length_mm", "plate_width_mm"):
            v = float(getattr(self, k))
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError(f"{k} must be finite and >= 0, got {v!r}")
            setattr(self, k, v)
        self.bevel_angle_deg = float(self.bevel_angle_deg)
        self.hi_lo_mm = float(self.hi_lo_mm)
        if not (0.0 <= self.bevel_angle_deg < 90.0):
            raise ValueError(f"bevel_angle_deg must be in [0, 90), got {self.bevel_angle_deg}")
        if self.root_face_mm > self.thickness_mm:
            raise ValueError("root_face_mm > thickness_mm")
        if self.kind == "square":
            self.bevel_angle_deg = 0.0
        self.backing = bool(self.backing)
        self.notes = list(self.notes)
        w = self.plate_widths_mm
        if w is None:
            w = [0.5 * self.plate_width_mm] * 2
        elif isinstance(w, (int, float)):
            w = [float(w)] * 2
        w = [float(x) for x in w]
        if len(w) == 1:
            w = w * 2
        if len(w) != 2 or not all(math.isfinite(x) and x >= 0.0 for x in w):
            raise ValueError(f"plate_widths_mm must be one or two finite widths >= 0 (left, right plate), got {self.plate_widths_mm!r}")
        self.plate_widths_mm = w
        self.plate_width_mm = w[0] + w[1]                  # the total of both plates (older consumers)

    # ---- 유도 치수 ----
    @property
    def bevel_height_mm(self) -> float:
        """베벨면의 수직 높이 = t − root_face."""
        return self.thickness_mm - self.root_face_mm

    @property
    def bevel_run_mm(self) -> float:
        """베벨면의 수평 폭 (한쪽) = (t − root_face)·tan(bevel)."""
        return self.bevel_height_mm * math.tan(math.radians(self.bevel_angle_deg))

    @property
    def n_bevels(self) -> int:
        return {"single_bevel": 1, "single_v": 2, "square": 0}[self.kind]

    @property
    def top_opening_mm(self) -> float:
        """상부 개구 폭 = gap + n_bevels·(t − root_face)·tan(bevel). (10T/30°/gap 5 → 5 + 5.77 = 10.77 mm)"""
        return self.root_gap_mm + self.n_bevels * self.bevel_run_mm

    @property
    def groove_depth_mm(self) -> float:
        """판 상면에서 그루브 바닥(뒷댐재 상면)까지 = t (루트페이스가 있어도 바닥은 뒷댐재 상면)."""
        return self.thickness_mm

    @property
    def side_agnostic(self) -> bool:
        """베벨 쪽을 비교 대상으로 삼지 않는다 ('unknown' | 'auto')."""
        return self.bevel_side in SIDE_AGNOSTIC

    def resolved_bevel_side(self) -> str:
        return "left" if self.side_agnostic else self.bevel_side

    # ---- 공칭 단면 ----
    def nominal_cross_section(self, x_mm) -> np.ndarray:
        """x [mm] (루트갭 중심 0) → 높이 h [mm] (뒷댐재 상면 0, 판 상면 t, right 판은 t + hi_lo). 가림 없는 표면 높이.

        수직면(직각 모서리, 루트페이스)은 한 x 에 여러 높이가 있으므로 반개구간 규약: 판 쪽 x 는 판 높이, 갭 안쪽 x 는 바닥.
        """
        x = np.asarray(x_mm, dtype=float)
        g = 0.5 * self.root_gap_mm
        t = self.thickness_mm
        rf = self.root_face_mm
        side = self.resolved_bevel_side()
        h = np.full(x.shape, np.nan if not self.backing else 0.0, dtype=float)
        for plate in ("left", "right"):
            sgn = -1.0 if plate == "left" else 1.0          # 판이 놓인 방향
            off = self.hi_lo_mm if plate == "right" else 0.0
            # 판 좌표: d = 루트 모서리에서 판 안쪽으로 들어간 거리 (≥ 0)
            d = sgn * x - g
            on_plate = d >= 0.0
            top = t + off
            beveled = (self.kind == "single_v") or (self.kind == "single_bevel" and plate == side)
            if beveled and self.bevel_run_mm > 0.0:
                run = self.bevel_run_mm
                frac = np.clip(d / run, 0.0, 1.0)
                h_plate = off + rf + frac * (t - rf)      # d=0: 루트페이스 상단, d=run: 판 상면
            else:
                h_plate = np.full(x.shape, top, dtype=float)
            h = np.where(on_plate, h_plate, h)
        return h

    def top_corners_mm(self) -> tuple:
        """상부 코너 x 위치 (left, right) [mm]."""
        g = 0.5 * self.root_gap_mm
        side = self.resolved_bevel_side()
        run = self.bevel_run_mm
        if self.kind == "single_v":
            return (-g - run, g + run)
        if self.kind == "single_bevel":
            return (-g - run, g) if side == "left" else (-g, g + run)
        return (-g, g)

    def root_edges_mm(self) -> tuple:
        g = 0.5 * self.root_gap_mm
        return (-g, g)

    # ---- 직렬화 ----
    def to_dict(self) -> dict:
        d = asdict(self)
        d["derived"] = {"top_opening_mm": self.top_opening_mm, "groove_depth_mm": self.groove_depth_mm,
                        "bevel_run_mm": self.bevel_run_mm, "top_corners_mm": list(self.top_corners_mm())}
        return d

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "GrooveSpec":
        d = dict(d or {})
        d.pop("derived", None)
        names = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in names}
        if kw.get("plate_widths_mm") is None and "plate_width_mm" in kw:
            note = (f"specimen.plate_width_mm ({float(kw['plate_width_mm']):g} mm, the total of both plates) is deprecated: give "
                    f"plate_widths_mm [left, right] (the width of each plate); read as two plates of "
                    f"{0.5 * float(kw['plate_width_mm']):g} mm")
            kw["notes"] = list(kw.get("notes") or []) + [note]
        elif kw.get("plate_widths_mm") is not None:
            kw.pop("plate_width_mm", None)                 # derived from the widths
        return cls(**kw)

    def deprecation_notes(self) -> list:
        """Notes about deprecated input keys (English) — for the run log."""
        return [n_ for n_ in self.notes if "deprecated" in str(n_)]

    def specimen_width_mm(self) -> float:
        """The specimen's width across the seam: left plate + root gap + right plate."""
        return float(self.plate_widths_mm[0]) + float(self.root_gap_mm) + float(self.plate_widths_mm[1])

    def save_yaml(self, path) -> None:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, path) -> "GrooveSpec":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        if "specimen" in d and isinstance(d["specimen"], dict):
            d = d["specimen"]
        return cls.from_dict(d)

    def summary(self) -> str:
        return (f"{self.name}: {self.kind} t={self.thickness_mm:g} mm bevel={self.bevel_angle_deg:g} deg ({self.bevel_side}) "
                f"gap={self.root_gap_mm:g} mm root_face={self.root_face_mm:g} mm backing={self.backing} "
                f"({self.backing_thickness_mm:g}T) -> top opening {self.top_opening_mm:.2f} mm, depth {self.groove_depth_mm:.1f} mm; "
                f"weld length {self.weld_length_mm:g} mm, plate widths left {self.plate_widths_mm[0]:g} / right "
                f"{self.plate_widths_mm[1]:g} mm, hi-lo nominal {self.hi_lo_mm:g} mm")
