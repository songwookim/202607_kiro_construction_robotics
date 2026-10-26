"""적응 자가 티칭(auto-teach)의 핵심 수학 — 정적 국소화 · 상대 보정 · 심 방향 · 패스 정련 · 충실도 게이트.

순수 numpy (I/O·로봇 없음). 설계: docs/design_panel_2026-09-30_adaptive_seam_teaching.md (§4.2~4.9, §5.3~5.4, "Revision").

불변 조건 (사용자 요구, 2026-09-30):
  I1  로봇·프로파일러는 판 상면(1G) 아래로 내려가지 않는다. 바닥 = **센서가 측정한** 판 상면
      (StaticLocalization.plate_top_height_m; 첫 국소화 후 올리기만 한다) — 입력된 숫자가 아니다.
  I2  **기울기 인지 기하**: 높이(1G)·판 거리(3G) 는 반드시 base 3D 점에서 계산한다:
          P_base = T_base←flange(jnt_ang FK) · T_flange←opt · [x, 0, z, 1]ᵀ (mm→m)
          height = up · P (up = −gravity_axis_base),  distance = n · (P − 기준점) (n = 측정한 판 법선)
      프로파일 원시 z(광축 방향 range) 는 높이/여유거리로 쓰지 않는다 — 프로파일러가 중력·판에 대해 기울어져
      있을 수 있기 때문. 원시 z 는 센서 자체의 유효 창(233~546 mm; validity.valid_mask) 판정에만 쓴다.
      (이 모듈에서 원시 z 가 쓰이는 곳: valid_mask 적용, diag 보고용 raw_z_*; 둘 다 주석으로 표시.)
  I3  보정은 현재 **측정된** 플랜지 자세에 대한 작은 델타(Correction) 다. 모델에서 계산한 절대 목표가 아니다.
  I4  모든 이동은 보내기 전에 포락선 검사(path_gen.check_envelope) — 이 모듈은 이동을 만들지 않는다.
      (Correction 을 적용한 자세 apply_correction(...) 를 호출자가 검사·거부한다. 자동 수정 없음.)

프레임: 광학 프레임 X_opt = 레이저 라인, Z_opt = range(판 쪽), Y_opt = 레이저 평면 법선.
베벨 쪽(left/right) 은 어디에도 게이트 조건이 아니다 (측 무관: GrooveSpec.bevel_side 'auto'/'unknown').
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .frames import inv_transform, make_transform, unit
from .path_gen import laser_plane_line_param
from .quality import ransac_plane
from .seam_features import (CrossSectionFeatures, SeamScanFeatures, _plain, extract_cross_section_features, extract_on_design_window,
                            matches_design)
from .measure import (DESIGN_HALF, HALF_TURN_DEG, K_DEFAULT, design_band, design_match, end_sigma_m, evidence_count,
                      lmeds_sigma, quant_sigma_mm, robust_sigma)
from .specimen import GrooveSpec
from .types import Calib
from .validity import Z_VALID_MM_DEFAULT, dX_mm, valid_mask

__all__ = [
    "median_profile", "profile_points_base", "height_m", "plate_distance_m",
    "StaticLocalization", "localize_static",
    "Correction", "relative_correction", "apply_correction", "transform_localization",
    "seam_from_pair",
    "PassFit", "ransac_line_3d", "refine_from_pass", "plate_points_from_scan", "polyline_from_scan_features",
    "EndParams", "identify_seam_ends",
    "Verdict", "fidelity_gate",
]


# --------------------------------------------------------------------------------------
# 기하 도우미 (I2)
# --------------------------------------------------------------------------------------
def profile_points_base(T_base_opt: np.ndarray, x_mm, z_mm) -> np.ndarray:
    """프로파일 점 (x, z) [mm] → base 3D 점 [m]: P = T_base←opt · [x·1e-3, 0, z·1e-3, 1]ᵀ. (N,3)"""
    T = np.asarray(T_base_opt, dtype=float)
    x = np.asarray(x_mm, dtype=float).reshape(-1) * 1e-3
    z = np.asarray(z_mm, dtype=float).reshape(-1) * 1e-3
    # np.outer(a, b) = a[:, None] * b[None, :] 와 같은 곱 (같은 값) — 패스마다 수천 번 불려 함수 오버헤드만 덜어낸다 (2026-10-08)
    return T[:3, 3] + x[:, None] * T[:3, 0] + z[:, None] * T[:3, 2]


def height_m(points_base, gravity_axis_base=(0.0, 0.0, -1.0)) -> np.ndarray:
    """기울기 인지 높이 = up · P (up = −중력축). (…,3) → (…). [I2]"""
    up = -unit(gravity_axis_base)
    return np.asarray(points_base, dtype=float) @ up


def plate_distance_m(points_base, plate_point_base, plate_normal_base) -> np.ndarray:
    """판 평면까지의 부호 있는 거리 = n · (P − p0) (n 은 센서 쪽 +). [I2]"""
    n = unit(plate_normal_base)
    p0 = np.asarray(plate_point_base, dtype=float).reshape(3)
    return (np.asarray(points_base, dtype=float) - p0) @ n


def _rot_axis(axis, ang_rad: float) -> np.ndarray:
    """단위축 둘레 Rodrigues 회전행렬."""
    a = unit(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(ang_rad) * K + (1.0 - math.cos(ang_rad)) * (K @ K)


def _angle_deg(a, b) -> float:
    ua, ub = unit(a), unit(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(ua @ ub)))))


def _nanmedian(v) -> float:
    v = np.asarray(v, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


# --------------------------------------------------------------------------------------
# 1. 시간 중앙값 프로파일
# --------------------------------------------------------------------------------------
def median_profile(records) -> tuple:
    """레코드 목록(.x/.z 또는 .x_mm/.z_mm, mm) → (x_med, z_med, n_valid_cols): 열별 nanmedian.

    z ≤ 0 센티널·NaN 은 무시. 유효 열 = 중앙값 x, z 가 finite 이고 z > 0 (센서 유효 창 검사는 호출자/localize_static).
    """
    xs, zs = [], []
    for r in records:
        x = getattr(r, "x", None)
        z = getattr(r, "z", None)
        if x is None or z is None:
            x, z = getattr(r, "x_mm"), getattr(r, "z_mm")
        xs.append(np.asarray(x, dtype=float).reshape(-1))
        zs.append(np.asarray(z, dtype=float).reshape(-1))
    if not xs:
        raise ValueError("median_profile: no records")
    n = min(a.size for a in xs)
    X = np.stack([a[:n] for a in xs])
    Z = np.stack([a[:n] for a in zs])
    with np.errstate(invalid="ignore"):
        bad = ~(np.isfinite(X) & np.isfinite(Z) & (Z > 0.0))
    X = np.where(bad, np.nan, X)
    Z = np.where(bad, np.nan, Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_med = np.nanmedian(X, axis=0)
        z_med = np.nanmedian(Z, axis=0)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(x_med) & np.isfinite(z_med) & (z_med > 0.0)
    return x_med, z_med, int(ok.sum())


# --------------------------------------------------------------------------------------
# 2. 정적 국소화
# --------------------------------------------------------------------------------------


@dataclass
class StaticLocalization:
    """정지 자세 1개(시간 중앙값 프로파일)의 국소화 결과 (base, m). 모든 높이·거리는 I2 (base 3D 점) 로 계산."""
    features: CrossSectionFeatures
    T_base_opt: np.ndarray
    opt_origin_base: np.ndarray
    plate_points_base: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # 판 인라이어 열의 base 3D 점
    plate_dir_base: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))    # 레이저 평면 ∩ 판 = 판 직선 방향
    plate_normal_base: np.ndarray = field(default_factory=lambda: np.full(3, np.nan)) # 판 법선 (센서 쪽 +)
    plate_top_height_m: float = float("nan")      # 1G I1 바닥: up · plate_points 의 **상위 분위수**(95 %) — 기울어진 판에서도 보수적
    plate_top_height_median_m: float = float("nan")  # up · plate_points 의 중앙값 (보고용; 바닥으로 쓰지 않는다)
    plate_height_spread_m: float = float("nan")   # max−min of up·plate_points (판이 중력 대비 기울면 커진다; 3G 라도 레이저 라인이
                                                  # 수평이면 ≈0 — 1G/3G 선택자로 쓰지 말 것: 방향은 ScanJob/진행방향에서)
    top_centre_base: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))   # 상부 개구 중심
    root_centre_base: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))  # 루트/바닥 중심
    axis_hit_base: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))     # 광축 ∩ 판 평면
    standoff_m: float = float("nan")              # 광학 원점 → 판 평면 거리 (n 방향) — 원시 z 아님
    incidence_deg: float = float("nan")           # angle(−Z_opt, n)
    roll_deg: float = float("nan")                # 레이저 평면 안 판 직선 기울기 = features.tilt_deg
    roll_signed_deg: float = float("nan")         # Y_opt 둘레 X_opt → 판 직선 부호 있는 회전각 (보정 명령용)
    lateral_offset_m: float = float("nan")        # 그루브(상부) 중심의 광축 대비 면내 오프셋 (라인 방향 e_lat 부호)
    lateral_axis_base: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))  # e_lat = X_opt 의 판면 투영
    valid_ratio: float = float("nan")
    flags: list = field(default_factory=list)
    groove_found: bool = False
    diag: dict = field(default_factory=dict)

    @property
    def plate_ok(self) -> bool:
        return bool(np.all(np.isfinite(self.plate_normal_base)) and np.isfinite(self.standoff_m))

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "opt_origin_base", "plate_dir_base", "plate_normal_base", "plate_top_height_m", "plate_top_height_median_m",
            "plate_height_spread_m", "top_centre_base", "root_centre_base", "axis_hit_base", "standoff_m", "incidence_deg", "roll_deg",
            "roll_signed_deg", "lateral_offset_m", "lateral_axis_base", "valid_ratio", "flags", "groove_found", "diag")}
        d["n_plate_points"] = int(self.plate_points_base.shape[0])
        d["T_base_opt"] = self.T_base_opt
        d["features"] = self.features.to_dict()
        return _plain(d)

    def summary(self) -> str:
        return (f"localize: groove={self.groove_found} plate_top_h={self.plate_top_height_m * 1e3:.2f} mm "
                f"(median {self.plate_top_height_median_m * 1e3:.2f}) standoff={self.standoff_m * 1e3:.2f} mm incidence={self.incidence_deg:.2f} deg roll={self.roll_deg:+.2f} deg "
                f"lateral={self.lateral_offset_m * 1e3:+.2f} mm valid={self.valid_ratio:.2f} flags={self.flags}")


def localize_static(T_base_flange, calib: Calib, x_mm, z_mm, spec: Optional[GrooveSpec], *,
                    gravity_axis_base=(0.0, 0.0, -1.0), travel_dir_hint=None, plate_normal_hint=None,
                    z_valid_mm=Z_VALID_MM_DEFAULT, features_kw: Optional[dict] = None) -> StaticLocalization:
    """정지 자세 프로파일 1개 → StaticLocalization.

    * 2-D 작업은 seam_features.extract_cross_section_features; 모든 점은 T_base←opt = T_base←flange · calib.T_flange←opt 로
      base 로 옮긴다 (I2).
    * 판 법선 n (우선순위):
        1. plate_normal_hint 가 있으면 n = unit(hint − (hint·d) d)  (1G 에서 up 을 주면 판=수평 가정)
        2. travel_dir_hint u(대략의 심 방향) 가 있으면 n = unit(d × u): d 와 u 가 모두 판면 안이므로 u 의 면내(yaw) 오차는
           영향이 없고, u 의 면외 성분(대략 점의 높이 오차)만 n 을 d 둘레로 기울인다.
        3. 힌트가 없으면 n = unit(−Z_opt 의 d 직교 성분): **한계** — 프로파일 1개로는 판이 레이저 라인 둘레로 기울어진
           정도(피치/입사각)를 볼 수 없으므로 광축이 판에 수직이라고 가정하는 것과 같다. 센서가 판에 대해 θ 만큼
           피치되어 있으면 standoff 가 (1−cos θ) 만큼 과대·top/root 점은 정확하지만 n 이 θ 만큼 틀린다. 플래그로 표시.
      n 은 항상 센서 쪽(n·(origin − plate_point) > 0) 으로 향한다.
    * standoff_m = median(n · (origin − plate_points)) — 원시 z 가 아니다 (I2).
    * plate_top_height_m = **95 % 분위수**(up · plate_points) (1G I1 바닥): 판이 중력 대비 기울어져 있으면(롤/피치 허용) 라인
      중앙의 중앙값보다 FOV 안·탐색점의 판이 더 높으므로 바닥은 보수적으로 잡는다. 중앙값은 plate_top_height_median_m (보고용).
      3G 에서 높이는 바닥이 아니다(판 거리 n·(P−p) 가 바닥) — plate_height_spread_m 은 수평 라인이면 3G 라도 ≈0 이므로 선택자가 아니다.
    """
    T_bf = np.asarray(T_base_flange, dtype=float).reshape(4, 4)
    T_bo = T_bf @ calib.T_flange_opt
    R = T_bo[:3, :3]
    o = T_bo[:3, 3].copy()
    X_opt, Y_opt, Z_opt = R[:, 0], R[:, 1], R[:, 2]
    up = -unit(gravity_axis_base)

    x = np.asarray(x_mm, dtype=float).reshape(-1)
    z = np.asarray(z_mm, dtype=float).reshape(-1)
    n_cols = int(x.size)
    # 원시 z 사용 (허용 범위): 센서 자체의 교정 유효 창 판정에만 쓴다.
    v = valid_mask(x, z, z_valid_mm)
    xv = np.where(v, x, np.nan)
    zv = np.where(v, z, np.nan)
    # 레이저 평면 기울기 보정 (seam_features depth_scale): 판 법선 추정 = 힌트 또는 X_opt × 진행 힌트 (라인의 면내 기울기는 작다)
    fkw = dict(features_kw or {})
    if "depth_scale" not in fkw:
        n_est = None
        if plate_normal_hint is not None:
            n_est = unit(plate_normal_hint)
        elif travel_dir_hint is not None:
            cx = np.cross(X_opt, unit(travel_dir_hint))
            n_est = unit(cx) if float(np.linalg.norm(cx)) > 1e-6 else None
        if n_est is not None:
            fkw["depth_scale"] = float(math.sqrt(max(0.0, 1.0 - min(1.0, abs(float(n_est @ Y_opt))) ** 2)))
    f = extract_cross_section_features(xv, zv, spec, **fkw)
    window = None
    if spec is not None and not matches_design(f, spec):
        # 설계와 맞지 않으면 파인 자리 후보 둘레 창에서 다시 (g3_14 E: 판 표면 물결이 판을 둘로 나눠 개구 118.5 mm 로 읽었다)
        f2, window = extract_on_design_window(xv, zv, spec, **fkw)
        if f2 is not None:
            f = f2
            f.flags = list(f.flags) + ["design_window"]

    loc = StaticLocalization(features=f, T_base_opt=T_bo, opt_origin_base=o)
    loc.valid_ratio = float(v.sum() / n_cols) if n_cols else 0.0
    flags = list(f.flags)
    diag = {"n_cols": n_cols, "n_valid": int(v.sum()), "x_c_mm": float(f.groove_centre_x_mm),
            "z_c_mm": float(f.groove_centre_z_mm), "x_root_mm": float(f.root_centre_x_mm),
            "z_root_mm": float(f.root_centre_z_mm), "tilt_deg": float(f.tilt_deg), "bevel_side_detected": f.bevel_side,
            "hi_lo_mm": float(f.hi_lo_mm), "normal_source": None, "design_window_mm": window}

    idx = f.plate_inlier_idx()
    if idx.size < 3 or not np.isfinite(f.tilt_deg):
        flags.append("no_plate")
        loc.flags = flags
        loc.diag = diag
        return loc

    # ---- 판 점 · 판 직선 (base) ----
    P = profile_points_base(T_bo, xv[idx], zv[idx])
    loc.plate_points_base = P
    phi = math.radians(float(f.tilt_deg))                     # tilt = atan(dz/dx) (양 판 평균)
    d = unit(R @ np.array([math.cos(phi), 0.0, math.sin(phi)]))
    if float(d @ X_opt) < 0:
        d = -d
    p_ref = np.median(P, axis=0)
    if plate_normal_hint is not None:
        h = unit(plate_normal_hint)
        n = unit(h - float(h @ d) * d)
        diag["normal_source"] = "plate_normal_hint"
    elif travel_dir_hint is not None:
        u = unit(travel_dir_hint)
        c = np.cross(d, u)
        n = unit(c) if np.linalg.norm(c) > 1e-9 else unit(-Z_opt - float((-Z_opt) @ d) * d)
        diag["normal_source"] = "travel_dir_hint"
        diag["yaw_line_vs_hint_deg"] = _angle_deg(d, u)
    else:
        n = unit(-Z_opt - float((-Z_opt) @ d) * d)
        diag["normal_source"] = "optical_axis"
        flags.append("normal_from_optical_axis_unverified_pitch")
    if float(n @ (o - p_ref)) < 0.0:
        n = -n
    loc.plate_dir_base = d
    loc.plate_normal_base = n

    # ---- I2: 거리·높이는 base 3D 점에서 ----
    loc.standoff_m = float(np.median((o - P) @ n))            # 판 평면 거리 (n 방향), 원시 z 아님
    hP = P @ up
    loc.plate_top_height_median_m = float(np.median(hP))      # 보고용 (라인 중앙 부근의 판 높이)
    loc.plate_top_height_m = float(np.max(hP))               # 1G I1 바닥: 판 인라이어 (추출기 띠 안) 중 가장 높은 점 (4d P5, 예전 95 % 분위수)
    loc.plate_height_spread_m = float(hP.max() - hP.min())
    loc.incidence_deg = _angle_deg(-Z_opt, n)
    loc.roll_deg = float(f.tilt_deg)
    # Y_opt 둘레로 X_opt 를 판 직선 d 에 얹는 부호 있는 각 (d 는 레이저 평면 안에 있으므로 X_opt × d ∥ Y_opt)
    loc.roll_signed_deg = math.degrees(math.atan2(float(np.cross(X_opt, d) @ Y_opt), float(X_opt @ d)))
    denom = float(n @ Z_opt)
    if abs(denom) > 1e-6:
        t = float(n @ (p_ref - o)) / denom
        loc.axis_hit_base = o + t * Z_opt
    e_lat = X_opt - float(X_opt @ n) * n
    loc.lateral_axis_base = unit(e_lat) if np.linalg.norm(e_lat) > 1e-9 else np.full(3, np.nan)

    # 진단용 원시 z (허용 범위: 보고 전용, 제어·안전에 쓰지 않음): 광축 근처 판 열의 z 중앙값
    near = np.argsort(np.abs(xv[idx]))[: min(20, idx.size)]
    diag["raw_z_axis_mm"] = float(np.median(zv[idx][near]))
    diag["raw_z_vs_standoff_mm"] = diag["raw_z_axis_mm"] - loc.standoff_m * 1e3
    diag["plate_height_spread_mm"] = loc.plate_height_spread_m * 1e3
    diag["plate_top_height_median_mm"] = loc.plate_top_height_median_m * 1e3

    # ---- 그루브 중심 ----
    if f.groove_found and np.isfinite(f.groove_centre_x_mm) and np.isfinite(f.groove_centre_z_mm):
        loc.top_centre_base = profile_points_base(T_bo, [f.groove_centre_x_mm], [f.groove_centre_z_mm])[0]
        if np.isfinite(f.root_centre_x_mm) and np.isfinite(f.root_centre_z_mm):
            loc.root_centre_base = profile_points_base(T_bo, [f.root_centre_x_mm], [f.root_centre_z_mm])[0]
        else:
            loc.root_centre_base = loc.top_centre_base.copy()
        if np.all(np.isfinite(loc.axis_hit_base)) and np.all(np.isfinite(loc.lateral_axis_base)):
            loc.lateral_offset_m = float((loc.top_centre_base - loc.axis_hit_base) @ loc.lateral_axis_base)
        loc.groove_found = bool(f.ok)
        if not f.ok:
            flags.append("groove_unreliable")
        # 그루브 깊이(판 상면 − 루트 중심, n 방향) 진단 — I2 (base 점)
        diag["root_depth_along_n_mm"] = float((loc.top_centre_base - loc.root_centre_base) @ n) * 1e3
    loc.flags = flags
    loc.diag = diag
    return loc


# --------------------------------------------------------------------------------------
# 3. 상대 보정 (I3)
# --------------------------------------------------------------------------------------
@dataclass
class Correction:
    """현재 측정 플랜지 자세에 적용할 작은 델타 (base, m / deg). apply_correction 으로 새 플랜지 자세를 만든다.

    새 광학 자세: 원점 o' = o + d_base, 회전 R' = Rot(roll_axis_base, roll_deg) · R  (원점 둘레 회전 + 이동).
    이는 '심 점(pivot_base) 둘레로 roll 만큼 회전한 뒤 원점을 목표로 옮긴다' 와 최종 자세가 같다.
    d_base 는 광학 원점의 **순 이동**(회전으로 인한 원점 스윙 포함) — 캡은 여기에 건다.
    """
    d_base: np.ndarray
    roll_axis_base: np.ndarray
    roll_deg: float
    pivot_base: np.ndarray
    predicted_origin_motion_m: float = 0.0
    swing_m: float = 0.0                       # 롤이 원점을 심 점 둘레로 스윙시키는 거리 (|o−pivot|·|roll|)
    d_in_plane_m: float = 0.0
    d_along_n_m: float = 0.0                   # +: 판에서 멀어짐, −: 판 쪽
    clipped: list = field(default_factory=list)
    reason: str = ""
    aim: str = ""
    target_base: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    uncapped: dict = field(default_factory=dict)

    @property
    def is_zero(self) -> bool:
        return bool(np.linalg.norm(self.d_base) < 1e-9 and abs(self.roll_deg) < 1e-9)

    def to_dict(self) -> dict:
        return _plain({k: getattr(self, k) for k in (
            "d_base", "roll_axis_base", "roll_deg", "pivot_base", "predicted_origin_motion_m", "swing_m", "d_in_plane_m",
            "d_along_n_m", "clipped", "reason", "aim", "target_base", "uncapped")})


def relative_correction(loc: StaticLocalization, *, target_standoff_m: float, caps: Optional[dict] = None,
                        seam_dir=None) -> Correction:
    """StaticLocalization → 현재 자세 기준 상대 보정 (I3).

    seam_dir (P2, 2026-10-09): 측정한 심 방향 (scan_core.seam_track — S 추적·트랜싯 스트림). 주면 면내 보정 축이 레이저 선 방향 e_lat 이
    아니라 측정 심에 수직인 w = unit(n × seam_dir) (seam_dir 는 이 드웰의 판 법선 n 평면에 투영) — 크기 = 보정 (원점 둘레 롤 + 평행이동)
    뒤 광축 착점이 심 위에 오는 w 이동 (p − o)·w + h·(Z'·w)/(Z'·n) (Z' = 롤 뒤 광축, h = 목표 판 거리), 버리는 성분은 심 방향
    (uncapped['dropped_along_travel_m'] = delta·u). 레이저 선이 심에 대해 기울어 있어도 착점의 심 방향 이동이 0 이고 선 발자국이 그대로다
    (기운 선을 따라 끝 쪽으로 미끄러지지 않는다: 2026-10-08 b2 t04 Δl 38.7 mm → Δs −10.2 mm). 새 착점이 심 위에 오므로 한 번에 수렴한다.
    None = 예전 그대로 (e_lat). 법선 증거 게이트·롤·캡 규칙은 같다.

    이동 = (심 점 + n·target_standoff) − 광학 원점, 면내/법선 성분으로 분리 — 기본은 잰 만큼 그대로 (상한 없음, 2026-10-07: 안전은
    호출자의 가드·포락선이 보고, 잰 값은 센서 시야·거리 창 안에서만 나온다). caps 를 주면 그 상한으로 줄인다:
      * 면내: |·| ≤ caps['in_plane_m'] (초과 시 축소, 'in_plane').
      * 법선, 판 쪽(−n): groove_found 이고 standoff_m > target 일 때만 (증거 게이트, 아니면 0 'along_n_evidence_gate'),
        ≤ caps['along_n_toward_plate_m']. 판 반대쪽(+n): ≤ caps['along_n_away_m'].
      * 롤: Y_opt(레이저 평면 법선) 둘레로 X_opt 를 판 직선에 얹는 각 (= −tilt). |·| ≤ caps['rotation_deg'].
        롤이 심 점 둘레 회전이면 원점이 |o−p|·θ 스윙하므로 면내 상한이 있으면 이 스윙도 계상 →
        필요하면 롤 스텝을 줄인다 (면내 이동이 축소되면 같은 비율로 롤도 축소해 기하를 일관되게 유지).
    심 점 = 상부 개구 중심(groove_found) 또는 광축∩판 (그루브 없음: 면내 이동은 롤 스윙만).
    판을 못 찾았으면 0 보정 (reason).
    """
    c = {"in_plane_m": math.inf, "along_n_toward_plate_m": math.inf, "along_n_away_m": math.inf, "rotation_deg": math.inf}
    c.update(caps or {})
    R = loc.T_base_opt[:3, :3]
    o = loc.opt_origin_base
    Y_opt = R[:, 1]
    zero = Correction(d_base=np.zeros(3), roll_axis_base=Y_opt.copy(), roll_deg=0.0, pivot_base=np.full(3, np.nan))
    if not loc.plate_ok or "no_plate" in loc.flags:
        zero.reason = "no plate measured: no correction (retreat along +n is the only allowed move, decided by the caller)"
        return zero
    n = loc.plate_normal_base
    if loc.groove_found and np.all(np.isfinite(loc.top_centre_base)):
        p = loc.top_centre_base
        aim = "top_centre"
    elif np.all(np.isfinite(loc.axis_hit_base)):
        p = loc.axis_hit_base
        aim = "axis_hit"
    else:
        zero.reason = "no seam point / axis hit"
        return zero
    target = p + n * float(target_standoff_m)
    delta = target - o
    d_n = float(delta @ n)
    # 면내 성분 중 레이저 라인 방향(e_lat) 만 보정한다. 진행 방향(레이저 평면 법선의 면내 투영) 성분은 프로파일 1개로는
    # 관측할 수 없고(핸드아이 피치 오차가 그대로 standoff·sin 으로 나타나 매 스텝 반복될 뿐), 심 위 위치는 패스가 정한다.
    e_lat = loc.lateral_axis_base if np.all(np.isfinite(loc.lateral_axis_base)) else unit(R[:, 0] - float(R[:, 0] @ n) * n)
    d_lat = None
    if seam_dir is not None:
        sd = np.asarray(seam_dir, dtype=float).reshape(3)
        sd = sd - n * float(sd @ n)
        if float(np.linalg.norm(sd)) > 1e-9 and np.all(np.isfinite(sd)):
            t_dir = unit(sd)
            e_lat = unit(np.cross(n, t_dir))           # w: 측정 심에 수직 (판 안)
            aim = aim + "_perp_seam"
            # 보정 뒤 광축 착점이 심 위에 오게 (레이저 선이 착점에서 심을 지난다): 보정은 원점 둘레 롤 (apply_correction) + 평행이동이라 보정
            # 뒤 광축 Z' = Rot(Y_opt, θ)·Z, 원점의 판 거리 = h (목표) → 착점 = 원점 + Z'·h/(−Z'·n). (A' − p)·w = 0 이려면
            # (o' − p)·w = h·(Z'·w)/(Z'·n). Z'·w 는 롤을 고친 뒤에도 남는 광축의 판면 성분 (3G 내려다보기·1G 입사 기울기) × 선이 심에 남은
            # yaw — 원점 기준 (p − o)·w 만 쓰면 그만큼 (3G 35°·ψ 0.66° → 3 mm) 착점이 심에서 비껴 수렴 허용을 넘었다 (가상 팔 2026-10-09).
            # 롤이 있으면 착점 (지금 광축) 기준도 틀리다 (롤이 원점 둘레로 선을 r·sin θ 옮긴다 — 판 12° 롤에서 86 mm). e_lat 경로는 Z'·e = 0
            # (롤 뒤 X' 가 판 안) 이라 예전 식과 같다.
            Zr = R[:, 2]
            th0 = float(loc.roll_signed_deg) if np.isfinite(loc.roll_signed_deg) else 0.0
            if th0 != 0.0:
                Zr = _rot_axis(R[:, 1], math.radians(th0)) @ Zr
            zn = float(Zr @ n)
            if abs(zn) > 1e-6:
                d_lat = float((p - o) @ e_lat) + float(target_standoff_m) * float(Zr @ e_lat) / zn
        else:
            t_dir = np.cross(n, e_lat)
    else:
        t_dir = np.cross(n, e_lat)
    if d_lat is None:
        d_lat = float(delta @ e_lat)
    d_ip = d_lat * e_lat
    clipped = []
    uncapped = {"d_in_plane_m": abs(d_lat), "d_along_n_m": d_n, "dropped_along_travel_m": float(delta @ t_dir)}
    # 면내 캡
    ip_norm = float(np.linalg.norm(d_ip))
    s_ip = 1.0
    if ip_norm > c["in_plane_m"] > 0:
        s_ip = c["in_plane_m"] / ip_norm
        d_ip = d_ip * s_ip
        clipped.append("in_plane")
    # 법선 캡 + 증거 게이트
    if d_n < 0.0:
        if not (loc.groove_found and loc.standoff_m > float(target_standoff_m)):
            d_n = 0.0
            clipped.append("along_n_evidence_gate")
        elif -d_n > c["along_n_toward_plate_m"]:
            d_n = -float(c["along_n_toward_plate_m"])
            clipped.append("along_n_toward_plate")
    elif d_n > c["along_n_away_m"]:
        d_n = float(c["along_n_away_m"])
        clipped.append("along_n_away")
    # 롤
    theta = float(loc.roll_signed_deg) if np.isfinite(loc.roll_signed_deg) else 0.0
    uncapped["roll_deg"] = theta
    lever = float(np.linalg.norm(o - p))
    s_rot = 1.0
    if abs(theta) > c["rotation_deg"] > 0:
        s_rot = min(s_rot, c["rotation_deg"] / abs(theta))
    swing_full = lever * math.radians(abs(theta))
    if math.isfinite(c["in_plane_m"]) and swing_full > c["in_plane_m"] > 0:
        s_rot = min(s_rot, c["in_plane_m"] / swing_full)
    s_rot = min(s_rot, s_ip)
    if s_rot < 1.0:
        theta *= s_rot
        clipped.append("rotation")
    d_base = d_ip + d_n * n
    corr = Correction(d_base=d_base, roll_axis_base=Y_opt.copy(), roll_deg=theta, pivot_base=p.copy(),
                      predicted_origin_motion_m=float(np.linalg.norm(d_base)), swing_m=lever * math.radians(abs(theta)),
                      d_in_plane_m=float(np.linalg.norm(d_ip)), d_along_n_m=d_n, clipped=clipped,
                      aim=aim, target_base=target, uncapped=uncapped)
    corr.reason = (f"aim={aim} in_plane {corr.d_in_plane_m * 1e3:.1f} mm along_n {d_n * 1e3:+.1f} mm roll {theta:+.2f} deg"
                   + (f" clipped={clipped}" if clipped else ""))
    return corr


def transform_localization(loc: StaticLocalization, T_rigid) -> StaticLocalization:
    """툴이 강체 변환 T_rigid (base, 4×4: T_new = T_rigid · T_old) 만큼 움직였을 때 같은 장면을 읽은 국소화 (P2 S_align: 그루브 점 G 를 지나는
    판 법선 n 둘레 회전 — 판 평면과 G 가 불변인 이동). 센서에 붙은 양 (광학 원점, T_base_opt, 판 직선 방향 = 레이저 평면 ∩ 판, 측면축 =
    X_opt 의 판면 투영) 은 T_rigid 로 옮기고, 장면 양 (그루브 상부·루트 중심, 판 점, 판 법선) 은 그대로 둔다. 광축 ∩ 판, 판 거리, 입사각,
    측면 오프셋은 새 자세와 같은 판 평면으로 다시 계산한다 (롤은 판 법선 둘레 회전에서 불변 — 그대로). 판 평면이 불변이 아닌 이동에는 쓰지
    않는다 (장면 양을 다시 재야 한다)."""
    import copy
    T = np.asarray(T_rigid, dtype=float).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    out = copy.copy(loc)
    out.diag = dict(loc.diag or {})
    out.flags = list(loc.flags)
    out.T_base_opt = T @ np.asarray(loc.T_base_opt, dtype=float)
    out.opt_origin_base = R @ np.asarray(loc.opt_origin_base, dtype=float) + t
    if np.all(np.isfinite(loc.plate_dir_base)):
        out.plate_dir_base = R @ np.asarray(loc.plate_dir_base, dtype=float)
    out.diag["transformed_by"] = T.tolist()
    if not loc.plate_ok:
        return out
    n = unit(np.asarray(loc.plate_normal_base, dtype=float))
    P = np.asarray(loc.plate_points_base, dtype=float).reshape(-1, 3)
    p_ref = np.median(P, axis=0) if P.shape[0] else np.asarray(loc.axis_hit_base, dtype=float)
    Rn = out.T_base_opt[:3, :3]
    o, X, Z = out.opt_origin_base, Rn[:, 0], Rn[:, 2]
    out.standoff_m = float(np.median((o - P) @ n)) if P.shape[0] else float((o - p_ref) @ n)
    out.incidence_deg = _angle_deg(-Z, n)
    denom = float(n @ Z)
    out.axis_hit_base = (o + Z * (float(n @ (p_ref - o)) / denom)) if abs(denom) > 1e-6 else np.full(3, np.nan)
    e_lat = X - float(X @ n) * n
    out.lateral_axis_base = unit(e_lat) if np.linalg.norm(e_lat) > 1e-9 else np.full(3, np.nan)
    if loc.groove_found and np.all(np.isfinite(loc.top_centre_base)) and np.all(np.isfinite(out.axis_hit_base)):
        out.lateral_offset_m = float((np.asarray(loc.top_centre_base) - out.axis_hit_base) @ out.lateral_axis_base)
    return out


def apply_correction(T_base_flange, calib: Calib, corr: Correction) -> np.ndarray:
    """현재 측정 플랜지 자세 + Correction → 새 플랜지 자세 (base). 호출자는 이 자세를 반드시 포락선 검사한다 (I4)."""
    T_bf = np.asarray(T_base_flange, dtype=float).reshape(4, 4)
    T_bo = T_bf @ calib.T_flange_opt
    R_new = _rot_axis(corr.roll_axis_base, math.radians(float(corr.roll_deg))) @ T_bo[:3, :3] if abs(corr.roll_deg) > 0 else T_bo[:3, :3]
    o_new = T_bo[:3, 3] + np.asarray(corr.d_base, dtype=float).reshape(3)
    return make_transform(R_new, o_new) @ inv_transform(calib.T_flange_opt)


# --------------------------------------------------------------------------------------
# 4. 두 정적 국소화(S, E) → 심 방향·법선·잔차
# --------------------------------------------------------------------------------------
def seam_from_pair(loc_S: StaticLocalization, loc_E: StaticLocalization, *, gravity_axis_base=(0.0, 0.0, -1.0)) -> dict:
    """같은 자세로 찍은 S/E 두 국소화 → {u, n, psi_deg, theta_deg, theta_standoff_deg, theta_height_deg, L_m, P_S, P_E, notes, ok}.

    u = unit(P_E − P_S) (상부 개구 중심; 핸드아이 병진은 정확히 상쇄). n = 두 판 직선과 u 의 외적 평균 (센서 쪽 +).
    psi_deg  = atan((x_cE − x_cS)/L): **센서 프레임** 그루브 중심 x 의 변화 → 레이저 라인 vs 심 yaw (핸드아이 무관).
               + = S→E 진행 시 그루브가 +X_opt 쪽으로 흐른다.
    theta_height_deg = atan((h_E − h_S)/L): 판 상면 높이(up·P, I2) 차 → 판 vs 진행 피치. + = S→E 로 판이 올라간다.
               **진행이 수직보다 수평에 가까울 때 (|u·up| < sin 45°, 원칙 2 중간점) 만 쓴다** — 수직 진행(3G) 이면 h_E − h_S ≈ ±L (C10).
    theta_standoff_deg = atan((standoff_S − standoff_E)/L): + = S→E 로 판에 가까워진다 (1G 수평 진행이면 theta_height 와 같다).
    theta_deg = theta_height_deg (수평 쪽 진행이고 finite) 아니면 theta_standoff_deg. gravity_axis_base 는 ScanJob 의 값을 넘긴다.
    notes = 실패 이유만 (판·그루브 없음, 두 점이 겹침); 거리 하한은 호출자가 먼저 본다 (착점 도착 허용·설계 길이·패스 시야 여유에서 유도).
    """
    notes = []
    ok = True
    for name, l in (("S", loc_S), ("E", loc_E)):
        if not l.plate_ok:
            notes.append(f"{name}: no plate")
            ok = False
        if not l.groove_found:
            notes.append(f"{name}: groove not found")
            ok = False
    out = {"u": np.full(3, np.nan), "n": np.full(3, np.nan), "psi_deg": float("nan"), "theta_deg": float("nan"),
           "theta_standoff_deg": float("nan"), "theta_height_deg": float("nan"), "travel_horizontal": None,
           "L_m": float("nan"), "P_S": loc_S.top_centre_base, "P_E": loc_E.top_centre_base,
           "n_S": np.full(3, np.nan), "n_E": np.full(3, np.nan), "notes": notes, "ok": False}
    if not ok:
        return out
    P_S, P_E = loc_S.top_centre_base, loc_E.top_centre_base
    dP = P_E - P_S
    L = float(np.linalg.norm(dP))
    if not L > 0.0:
        notes.append("S and E coincide")
        return out
    u = dP / L
    up = -unit(gravity_axis_base)
    horizontal = abs(float(u @ up)) < math.sin(math.radians(HALF_TURN_DEG))   # 진행이 수평(1G) 쪽인가 — 높이 차 θ 는 이때만 의미
    nS = np.cross(loc_S.plate_dir_base, u)
    nE = np.cross(loc_E.plate_dir_base, u)
    nS = unit(nS) if np.linalg.norm(nS) > 1e-9 else loc_S.plate_normal_base
    nE = unit(nE) if np.linalg.norm(nE) > 1e-9 else loc_E.plate_normal_base
    if float(nS @ (loc_S.opt_origin_base - P_S)) < 0:
        nS = -nS
    if float(nE @ (loc_E.opt_origin_base - P_E)) < 0:
        nE = -nE
    n = unit(nS + nE)
    n = unit(n - float(n @ u) * u)
    x_cS, x_cE = loc_S.diag.get("x_c_mm", float("nan")), loc_E.diag.get("x_c_mm", float("nan"))
    psi = math.degrees(math.atan2(x_cE - x_cS, L * 1e3))
    hS, hE = loc_S.plate_top_height_median_m, loc_E.plate_top_height_median_m
    theta_h = math.degrees(math.atan2(hE - hS, L)) if np.isfinite(hS) and np.isfinite(hE) else float("nan")
    theta_so = math.degrees(math.atan2(loc_S.standoff_m - loc_E.standoff_m, L))
    theta = theta_h if (horizontal and np.isfinite(theta_h)) else theta_so   # 수직 진행(3G) 은 standoff 차로 (C10)
    # 진단 값 (판정 없음)
    out["n_S_vs_n_E_deg"] = _angle_deg(nS, nE)
    dtravel = loc_E.opt_origin_base - loc_S.opt_origin_base
    if np.linalg.norm(dtravel) > 0.0:
        out["travel_vs_seam_deg"] = _angle_deg(dtravel, u)
    out.update({"u": u, "n": n, "n_S": nS, "n_E": nE, "psi_deg": psi, "theta_deg": theta, "theta_standoff_deg": theta_so,
                "theta_height_deg": theta_h, "travel_horizontal": bool(horizontal), "L_m": L, "ok": True,
                "roll_S_deg": loc_S.roll_deg, "roll_E_deg": loc_E.roll_deg})
    return out


# --------------------------------------------------------------------------------------
# 5. 패스 정련
# --------------------------------------------------------------------------------------
def ransac_line_3d(P: np.ndarray, thr_m: Optional[float] = None, iters: int = 300, rng: Optional[np.random.Generator] = None,
                   min_pts: int = 5, *, k: float = K_DEFAULT, floor_m: Optional[float] = None, info: Optional[dict] = None) -> tuple:
    """3D RANSAC 직선 + 인라이어 PCA 정련 → (c (3,), u (3,), inlier mask, rms_m). 점 부족 시 (nan, nan, zeros, nan).

    thr_m None (기본) = **적응 문턱** (scan_core.measure 원칙 1, 2026-10-07): 가설은 직선까지 거리의 중앙값이 가장 작은 것 (LMedS),
    σ = LMedS 척도 (2 자유도: med / √(2 ln 2)), 인라이어 = 거리 ≤ k·max(σ, floor_m) (floor_m = 센서 양자화 dX/√12 [m], 모르면 유효 창
    가운데 값). 정련마다 σ·문턱을 다시 잰다. thr_m 을 주면 그 고정 문턱 (최대 인라이어 수 가설). info (dict) 에 sigma_m·thr_m."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    M = P.shape[0]
    nan3 = np.full(3, np.nan)
    if M < max(2, min_pts):
        return nan3, nan3, np.zeros(M, dtype=bool), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    adaptive = thr_m is None
    fl = float(quant_sigma_mm(None) * 1e-3 if floor_m is None else floor_m)

    def perp_dist(c_, u_):
        rel = P - c_
        return np.linalg.norm(rel - np.outer(rel @ u_, u_), axis=1)

    def thr_of(dist):
        if not adaptive:
            return float(thr_m), float("nan")
        s_ = lmeds_sigma(dist, 2)
        return float(k) * max(s_, fl), s_

    best_cost, best = np.inf, None
    for _ in range(int(iters)):
        i, j = rng.choice(M, 2, replace=False)
        d = P[j] - P[i]
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-9:
            continue
        d /= nrm
        dist = perp_dist(P[i], d)
        cost = float(np.median(dist)) if adaptive else -float((dist < thr_m).sum())
        if cost < best_cost:
            best_cost, best = cost, (P[i].copy(), d.copy())
    if best is None:
        return nan3, nan3, np.zeros(M, dtype=bool), float("nan")
    c, u = best
    if int((perp_dist(c, u) < thr_of(perp_dist(c, u))[0]).sum()) < min_pts:
        return nan3, nan3, np.zeros(M, dtype=bool), float("nan")
    inl = None
    for _ in range(3):
        dist = perp_dist(c, u)
        inl_new = dist < thr_of(dist)[0]
        if inl_new.sum() < min_pts:
            break
        inl = inl_new
        Q = P[inl]
        c = Q.mean(axis=0)
        _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
        u = Vt[0]
    if inl is None:
        return nan3, nan3, np.zeros(M, dtype=bool), float("nan")
    dist = perp_dist(c, u)
    thr, sig = thr_of(dist)
    inl = dist < thr
    rms = float(np.sqrt(np.mean(dist[inl] ** 2))) if inl.any() else float("nan")
    if info is not None:
        info.update(sigma_m=float(sig), thr_m=float(thr), rule="lmeds" if adaptive else "fixed")
    return c, unit(u), inl, rms


@dataclass
class PassFit:
    """한 패스(스캔)의 심 직선·판 평면·범위 정련 결과 (base, m)."""
    c: np.ndarray
    u: np.ndarray
    n: np.ndarray
    line_rms_m: float = float("nan")
    plane_rms_m: float = float("nan")
    s_a: float = float("nan")               # groove_found 구간 시작 (u 방향, c 기준)
    s_b: float = float("nan")
    n_inliers: int = 0
    n_plane_inliers: int = 0
    root_polyline_fit: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    feature_medians: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    diag: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(np.all(np.isfinite(self.u)) and np.all(np.isfinite(self.n)) and np.isfinite(self.s_a) and np.isfinite(self.s_b))

    @property
    def start(self) -> np.ndarray:
        return self.c + self.u * self.s_a

    @property
    def end(self) -> np.ndarray:
        return self.c + self.u * self.s_b

    @property
    def length_m(self) -> float:
        return float(self.s_b - self.s_a)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("c", "u", "n", "line_rms_m", "plane_rms_m", "s_a", "s_b", "n_inliers",
                                           "n_plane_inliers", "feature_medians", "flags", "diag")}
        d["start"] = self.start if self.ok else np.full(3, np.nan)
        d["end"] = self.end if self.ok else np.full(3, np.nan)
        d["length_m"] = self.length_m
        d["n_polyline"] = int(self.root_polyline_fit.shape[0])
        return _plain(d)


def _feature_medians(feats, groove_found: np.ndarray) -> dict:
    """특징 중앙값 dict. SeamScanFeatures 이면 medians() 그대로 (wall_shear_k / floor_visible_fraction 은 ok ∧ in_cruise 선택 —
    램프/프리롤 프로파일 제외; 요약(summary) 과 게이트가 같은 값을 본다). dict 이면 배열별 nanmedian."""
    out = {}
    if feats is None:
        pass
    elif isinstance(feats, SeamScanFeatures):
        out.update(feats.medians())
    elif isinstance(feats, dict):
        for k, v in feats.items():
            a = np.asarray(v, dtype=float).reshape(-1)
            out[k] = _nanmedian(a) if a.size else float("nan")
    out.setdefault("wall_shear_k", float("nan"))        # 추출기가 아직 계산하지 않음 → NaN = 미평가
    out["ok_fraction"] = float(np.mean(groove_found)) if groove_found.size else 0.0
    out["n_groove_profiles"] = int(groove_found.sum())
    return out


def refine_from_pass(polyline_root_base, plate_points_base, groove_found, feats, spec: Optional[GrooveSpec], *,
                     u_prior, n_prior, ransac_line_thr_m: Optional[float] = None, ransac_plane_thr_m: Optional[float] = None,
                     min_profiles: int = 10, seed: int = 0, k: float = K_DEFAULT,
                     floor_m: Optional[float] = None, min_line_pts: int = 5) -> PassFit:
    """패스의 루트 폴리라인(M,3) + 판 점(K,3) + groove_found(M,) + 특징 → PassFit.

    * 심 직선: groove_found 점의 3D RANSAC + 인라이어 PCA → c, u (u 는 u_prior 쪽). 인라이어 문턱 기본 = k × 이 패스의 LMedS 척도
      (하한 floor_m = 센서 양자화 dX/√12; 2026-10-07 — 예전 고정 0.5 mm). diag line_sigma_m / plane_sigma_m 에 척도.
    * 판 평면: quality.ransac_plane (같은 적응 문턱) → n (n_prior = 센서 쪽 방향으로 정렬) 뒤 u 에 직교화 (판이 살짝 비평행이어도 u 우선).
    * 범위: s_i = (P_i − c)·u; 직선 인라이어의 최소·최대 → s_a, s_b (임시 범위 — 끝 판정·관측 여부는 identify_seam_ends 가 정한다).
    * root_polyline_fit: 각 점을 직선에 투영 (finite 점만, 나머지 NaN).
    """
    P = np.asarray(polyline_root_base, dtype=float).reshape(-1, 3)
    M = P.shape[0]
    gf = np.asarray(groove_found, dtype=bool).reshape(-1)
    if gf.size != M:
        raise ValueError("groove_found length must match the polyline")
    finite = np.all(np.isfinite(P), axis=1)
    ok = gf & finite
    nan3 = np.full(3, np.nan)
    fit = PassFit(c=nan3.copy(), u=nan3.copy(), n=nan3.copy(), root_polyline_fit=np.full((M, 3), np.nan))
    fit.feature_medians = _feature_medians(feats, gf)
    rng = np.random.default_rng(seed)
    if ok.sum() < min_profiles:
        fit.flags.append("few_groove_profiles")
        return fit
    linfo: dict = {}
    c, u, inl, rms = ransac_line_3d(P[ok], ransac_line_thr_m, rng=rng, min_pts=int(min_line_pts), k=k, floor_m=floor_m, info=linfo)
    fit.diag["line_sigma_m"] = float(linfo.get("sigma_m", float("nan")))
    fit.diag["line_thr_m"] = float(linfo.get("thr_m", float("nan")))
    if not np.all(np.isfinite(u)):
        fit.flags.append("line_fit_failed")
        return fit
    if float(u @ unit(u_prior)) < 0:
        u = -u
    inl_full = np.zeros(M, dtype=bool)
    inl_full[np.nonzero(ok)[0][inl]] = True                  # 직선 인라이어 (전체 인덱스)
    fit.c, fit.u, fit.line_rms_m, fit.n_inliers = c, u, rms, int(inl.sum())
    fit.diag["line_inlier_ratio"] = float(inl.mean())
    fit.diag["u_vs_prior_deg"] = _angle_deg(u, u_prior)
    # 판 평면
    Q = np.asarray(plate_points_base, dtype=float).reshape(-1, 3)
    Q = Q[np.all(np.isfinite(Q), axis=1)]
    if Q.shape[0] >= 3:
        pl = ransac_plane(Q, inlier_mm=None if ransac_plane_thr_m is None else ransac_plane_thr_m * 1e3, rng=rng, k=k,
                          floor_mm=None if floor_m is None else floor_m * 1e3)
        fit.diag["plane_sigma_m"] = float(pl.get("sigma_mm", float("nan"))) * 1e-3
        fit.diag["plane_thr_m"] = float(pl.get("thr_mm", float("nan"))) * 1e-3
        n = np.asarray(pl["normal"], dtype=float)
        if np.all(np.isfinite(n)):
            d_pl = float(pl["d"])
            if float(n @ unit(n_prior)) < 0:
                n = -n
                d_pl = -d_pl
            # 측정 판 상면 평면 n·P + d = 0 (n 센서 쪽; 판 끝 식별 identify_seam_ends 의 기준 평면)
            fit.diag["plate_plane_n"] = n.copy()
            fit.diag["plate_plane_d"] = d_pl
            fit.diag["n_vs_u_deg"] = abs(90.0 - _angle_deg(n, u))
            n = unit(n - float(n @ u) * u)
            fit.n = n
            fit.plane_rms_m = float(pl["rms_mm"]) * 1e-3
            fit.n_plane_inliers = int(round(pl["inlier_ratio"] * pl["n"]))
            fit.diag["plane_inlier_ratio"] = float(pl["inlier_ratio"])
            fit.diag["n_vs_prior_deg"] = _angle_deg(n, n_prior)
        else:
            fit.flags.append("plane_fit_failed")
    else:
        fit.flags.append("no_plate_points")
    # 범위
    s = np.full(M, np.nan)
    s[finite] = (P[finite] - c) @ u
    seg = s[inl_full]                                        # 아웃라이어의 u 방향 성분이 범위를 밀지 않도록 직선 인라이어만
    fit.s_a, fit.s_b = float(seg.min()), float(seg.max())
    fit.root_polyline_fit[finite] = c + np.outer(s[finite], u)
    perp = P[ok] - fit.root_polyline_fit[ok]
    fit.diag["polyline_max_dev_m"] = float(np.linalg.norm(perp, axis=1).max()) if ok.any() else float("nan")
    return fit


def plate_points_from_scan(result, features_list, indices=None) -> np.ndarray:
    """ScanResult + 프로파일별 CrossSectionFeatures(plate_inlier_idx) → 판 인라이어 base 3D 점 (K,3) (I2).

    indices: features_list[j] 가 대응하는 프로파일 인덱스 (None → j). extract_seam_features_along_scan(keep_features=True)
    의 features 와 indices 를 그대로 넣는다.
    """
    idx = np.arange(len(features_list)) if indices is None else np.asarray(indices, dtype=np.int64).reshape(-1)
    pts = []
    for j, f in enumerate(features_list):
        if f is None:
            continue
        cols = f.plate_inlier_idx()
        if cols.size == 0:
            continue
        i = int(idx[j])
        x = np.asarray(result.x_mm[i], dtype=float)[cols]
        z = np.asarray(result.z_mm[i], dtype=float)[cols]
        pts.append(profile_points_base(result.T_base_opt[i], x, z))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def polyline_from_scan_features(result, sf: SeamScanFeatures, centre: str = "root") -> tuple:
    """SeamScanFeatures → (polyline (M,3) base, groove_found (M,)) — 모든 프로파일(실패는 NaN) 을 같은 길이로.

    폴리라인 점 = T_base_opt[i] · [x_c, 0, z_c] (I2; ScanResult.s_m 은 쓰지 않는다).
    """
    if centre not in ("root", "top"):
        raise ValueError("centre must be 'root' or 'top'")
    xs = sf.root_centre_x_mm if centre == "root" else sf.groove_centre_x_mm
    zs = sf.root_centre_z_mm if centre == "root" else sf.groove_centre_z_mm
    M = int(sf.indices.size)
    P = np.full((M, 3), np.nan)
    for j in range(M):
        xc, zc = float(xs[j]), float(zs[j])
        if np.isfinite(xc) and np.isfinite(zc):
            P[j] = profile_points_base(result.T_base_opt[int(sf.indices[j])], [xc], [zc])[0]
    gf = np.asarray(sf.ok, dtype=bool) & np.all(np.isfinite(P), axis=1)
    return P, gf


@dataclass
class EndParams:
    """판 끝 식별 (identify_seam_ends · support_plane_ends) 의 입력 — **판정 문턱은 없다** (scan_core.measure 원칙, 사용자 2026-10-07).
    명시 입력은 다른 곳에서 정해진 값뿐이다: 통계 배수 k (플랫폼 judgement.k), 용접 경로 단면 창 (작업 weld_plan station_window), 끝
    구역 지연 (깊이 × tan 입사 — 기운 레이저 면이 루트를 가리는 기하 구간), 센서 유령점 버스트 (안전 정책 sensor.ghost_burst_frames),
    판으로 인정하는 최소 면 길이 (특징 추출기). 나머지는 이 패스의 측정 (판마다 상면 모형과 강건 σ, 루트 직선·바닥 점의 강건 σ, 착점
    간격, 센서 열 간격 dX(z)) 과 설계 (갭) 에서 유도해 diag['end_derived'] 에 남긴다. 기본값은 설정 기본과 같다."""
    k: float = K_DEFAULT
    window_m: float = 0.010             # 용접 경로 단면 창 (WeldProcess.station_window_mm)
    lag_m: float = 0.0                  # 끝 구역 지연 [m] (scan_l0.autoteach.end_zone_lag_m)
    ghost_burst_frames: int = 2         # 유령점 최대 연속 프레임 (sensor.ghost_burst_frames)
    plate_min_len_m: float = 0.005      # 판 최소 면 길이 (seam_features.plate_min_len_mm)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "EndParams":
        d = dict(d or {})
        kw = {}
        for k_, v in d.items():
            if k_ in cls.__dataclass_fields__ and v is not None:
                kw[k_] = int(v) if k_ == "ghost_burst_frames" else float(v)
        return cls(**kw)

    def to_dict(self) -> dict:
        return {k_: (int(getattr(self, k_)) if k_ == "ghost_burst_frames" else float(getattr(self, k_))) for k_ in self.__dataclass_fields__}

    @property
    def confirm_profiles(self) -> int:
        """유령점이 아님을 확인하는 연속 프로파일 수 = 버스트 + 1 (안전 정책과 같은 원천)."""
        return max(1, int(self.ghost_burst_frames) + 1)

    @property
    def look_m(self) -> float:
        """끝 증거 구간 [m] = 끝 구역 지연 + 용접 경로 단면 창 (판 끝 너머·안쪽에서 모서리·바닥·가림 증거를 모으는 거리)."""
        return max(0.0, float(self.lag_m)) + float(self.window_m)


def _line_on_plane_u(T_bo: np.ndarray, plane_n: np.ndarray, plane_d: float, c: np.ndarray, u: np.ndarray, w: np.ndarray,
                     v: float) -> float:
    """프로파일 i 의 레이저 평면 ∩ 평면 (plane_n·X + plane_d = 0) 위에서 측면 좌표 v 인 점의 u 좌표 (판 상면 착점선의 그 측면 위치).
    해가 없으면 NaN."""
    o, Y = T_bo[:3, 3], T_bo[:3, 1]
    A = np.vstack([plane_n, Y, w])
    b = np.array([-plane_d, float(Y @ o), v + float(w @ c)])
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return float("nan")
    return float((X - c) @ u)


def _profile_z_mm(result, rows) -> float:
    """프로파일들의 센서 z 중앙값 [mm] (센서 열 간격 dX(z)·양자화 하한의 기준)."""
    zs = []
    valid = getattr(result, "valid", None)
    for i in rows:
        z = np.asarray(result.z_mm[int(i)], dtype=float)
        v = np.isfinite(z) & (z > 0)
        if valid is not None:
            v &= np.asarray(valid[int(i)], dtype=bool)
        if v.any():
            zs.append(float(np.median(z[v])))
    return float(np.median(zs)) if zs else float("nan")


_SAMPLE_PROFILES = 64        # 수치 해상도: 패스 전체 통계 (센서 z) 를 고르게 뽑은 이만큼의 프로파일에서 잰다


def identify_seam_ends(result, sf, fit: PassFit, polyline_root_base, groove_ok, spec: Optional[GrooveSpec] = None,
                       params: Optional[EndParams] = None) -> PassFit:
    """**모델 기반·기울기 인식 판 끝 식별** (사용자 2026-10-01; 1G·3G 공통 — 판 평면은 이 패스에서 측정한 평면). fit 을 제자리에서
    고친다: s_a/s_b (u 좌표, c 기준), flags (end_not_observed_first/last, plates_staggered, end_occluded_first/last), diag['ends'].
    판정 기준은 scan_core.measure 원칙에서 유도한다 (사용자 2026-10-07: 작위적 상수 없음) — diag['end_derived'] 에 값과 근거:
      * **면 판별** (이 점이 판 상면인가·그루브 바닥인가·다른 물체인가) = 원칙 2 가설 중간점: 설계상 판 상면과 그루브 바닥은 깊이만큼
        떨어진 두 면이므로 어느 면 '위' = 그 면에서 깊이/2 안. 측정 잡음 (k·σ) 이 아니다 — 실기 재생 (2026-10-07) 에서 로봇 진동으로
        프로파일 전체가 ±0.35 mm, 판 휨 ±0.3 mm, 판 끝 모서리 버 0.3–1 mm 가 났고, 이것들은 다른 면이 아니다. 판 위 물체 (클램프)·
        테이블·지그는 수 mm 이상 다른 면이다.
      * **추정** (직선·끝 위치 σ·좌우 끝 차의 유의성) = 원칙 1 k·σ.

    a. 프로파일별 ON-SPECIMEN 판정 (base 3D, I2): 양 판 상면이 모두 있고 각각 측면 폭 ≥ 판 최소 면 길이 (추출기) 이며 그 점의 절반
       이상이 판 상면 위 (원칙 2); 그루브 루트 중심이 정련 심 직선에서 갭/2 안 (루트가 이 그루브의 갭 안 — 면 판별; 갭을 모르면
       k·σ_루트); 갭·깊이·개구가 설계 공칭 ±50 % (원칙 2, measure.design_match). 판 끝 너머의 바닥/테이블 (낮음), 클램프 (높음), 임의
       각도의 지그는 이 판정을 통과하지 못한다.
    b. 끝 = 관측된 심의 **가운데**를 포함하는 연속 on-specimen 구간의 경계 (가운데에서 바깥으로 키운다). 결측은 양 판 상면이 내내 보일
       때만, 용접 경로 단면 창 (window_m) 까지 잇는다 — 판 상면이 없는 프로파일은 절대 넘지 않는다 (그루브 특징만 흔들리는 반사 구간,
       2026-10-02 at06).
    c. 판별 sub-profile 끝 위치 = 증거 괄호의 중점, 판마다 (좌/우) 따로. 하한 (U_on) = 판 상면 점의 가장 바깥 u (바깥쪽
       on-specimen 프로파일과, 그 너머에서 그 판 대역에 아직 판 상면 점이 있는 프로파일) 와 **그루브 바닥 점** 이 있는 가장 바깥 u
       (끝 증거 구간 look_m = 지연 + 단면 창 안). 상한 (U_off) = 그 판 대역에 판 상면 점이 없는 칸 (칸 = 판 최소 면 길이) 의 판 상면
       착점선 u 와 바닥 점이 없는 첫 프로파일의 루트 착점 t 중 안쪽 것 — 하한 바깥에 있는 후보만 (안전망). 바닥 점 = 루트 직선에서
       측면 ± 갭/4 (갭 가운데 절반, 원칙 2) 이고 루트 직선에서 깊이/2 안 (면 판별); '있음' = 그 대역의 열 수 (폭 / 측정 열 간격) 의 절반
       이상 (원칙 2), '같은 바닥' = 직전 바닥 프로파일과 중앙값 차가 k·σ_이웃 안 (원칙 1 — 천천히 떠도는 바닥은 따라가고 다른 면의 계단은
       끊는다), '끝' = 바닥 없는 프로파일이 유령점 버스트 + 1 개 연속.
       레이저 면이 진행 방향으로 기울면 상면과 바닥의 착점선이 깊이·tan 만큼 어긋나 한 끝에서는 상면이 먼저 사라지는데, 그때 그루브
       특징 추출은 바닥이 아직 있어도 실패하므로 '그루브가 사라진 첫 프로파일' 은 상한이 아니다 (라운드 4 F2). 끝 근처에서 바닥이
       보이지 않으면 (뒷댐재 없음) 그 규칙 (groove_present 전이) 을 상한으로 쓴다. 끝 σ = 괄호 폭/√12 (괄호가 열려 있으면 착점
       간격/√12, 원칙 1). 좌/우 판 끝의 차가 k·√(σ_좌² + σ_우²) 를 넘으면 (실제로 다름) 'plates_staggered' — 용접 경로 끝 (s_m) =
       안쪽 판 (그루브가 온전한 곳), 아니면 평균. 판 길이 검사용 끝 (s_plates_m) 은 언제나 두 판 끝의 평균 (시편 엇갈림은 보고
       대상이지 식별 오류가 아니다).
    d. '관측됨' = 끝 너머로 서로 다른 위치 (착점 간격의 절반 넘게 떨어진) 의 비시편 프로파일이 유령점 버스트 + 1 개 이상 계속 스캔됨
       (센서 안전 정책과 같은 확인 수; 멈춘 로봇의 같은 자리 프레임은 하나); 아니면 'end_not_observed_*'.
    e. 끝 너머 look_m 안의 비시편 프로파일을 바깥쪽 순서로 보아, 판보다 낮은 면 (테이블) 이 먼저 드러나기 전에 판 상면보다 깊이/2 넘게 위의 점이 판 측면 범위 안에 '면 증거' (판 최소 면 길이의
       열 수 절반) 이상 나오는 프로파일이 유령점 버스트 + 1 개 연속이면 (판 위 물체가 끝을 가림) 'end_occluded_*' — 그 끝은 관측으로
       치지 않고 추정 끝 (다른 끝 ± 설계 길이) 을 보고한다 (게이트는 auto_teach.accept_inferred_end 가 아니면 PASS 하지 않는다). 설계
       길이 교차 검사 ('end_ambiguous' + 대안 끝) 는 허용치를 가진 fidelity_gate 가 한다 — 여기서는 대안 끝 (다른 끝 ± 설계 길이) 을
       diag 에 담는다 (절대 조용히 바꾸지 않는다)."""
    pr = params if params is not None else EndParams()
    kk = float(pr.k)
    P = np.asarray(polyline_root_base, dtype=float).reshape(-1, 3)
    M = P.shape[0]
    gok = np.asarray(groove_ok, dtype=bool).reshape(-1) & np.all(np.isfinite(P), axis=1)
    fit.flags = [f for f in fit.flags if not (f.startswith("end_not_observed") or f.startswith("end_occluded")
                                              or f == "plates_staggered")]
    pn = np.asarray(fit.diag.get("plate_plane_n", np.full(3, np.nan)), dtype=float)
    pdd = float(fit.diag.get("plate_plane_d", float("nan")))
    if not (np.all(np.isfinite(fit.c)) and np.all(np.isfinite(fit.u)) and np.all(np.isfinite(pn)) and np.isfinite(pdd)):
        fit.flags.append("ends_not_identified")
        return fit
    c, u = np.asarray(fit.c, dtype=float), unit(fit.u)
    w = unit(np.cross(pn, u))                                   # 측면 축 (판 평면 안, 심에 수직)
    idx = np.asarray(sf.indices, dtype=np.int64)
    T_all = np.asarray(result.T_base_opt, dtype=float)
    T = T_all[idx]
    t = np.asarray(laser_plane_line_param(T, c, u), dtype=float).reshape(-1)
    feats = sf.features if sf.features is not None else [None] * M
    valid = getattr(result, "valid", None)

    def all_points(j: int) -> np.ndarray:
        i = int(idx[j])
        x = np.asarray(result.x_mm[i], dtype=float)
        z = np.asarray(result.z_mm[i], dtype=float)
        v = np.isfinite(x) & np.isfinite(z) & (z > 0)
        if valid is not None:
            v &= np.asarray(valid[i], dtype=bool)
        return profile_points_base(T_all[i], x[v], z[v]) if v.any() else np.zeros((0, 3))

    def side_points(j: int, side: str) -> np.ndarray:
        f = feats[j]
        cols = getattr(f, f"plate_inlier_idx_{side}", None) if f is not None else None
        if cols is None or np.asarray(cols).size == 0:
            return np.zeros((0, 3))
        i = int(idx[j])
        cols = np.asarray(cols, dtype=np.int64)
        return profile_points_base(T_all[i], np.asarray(result.x_mm[i], dtype=float)[cols], np.asarray(result.z_mm[i], dtype=float)[cols])

    def sample(mask: np.ndarray) -> np.ndarray:
        j_ = np.flatnonzero(mask)
        return j_[:: max(1, int(math.ceil(j_.size / _SAMPLE_PROFILES)))] if j_.size else j_

    # ---- 0. 이 패스의 측정과 설계 ----
    tf_ = np.sort(t[np.isfinite(t)])
    sp = float(np.median(np.abs(np.diff(tf_)))) if tf_.size >= 2 else float("nan")       # 착점 간격
    z_mm = _profile_z_mm(result, idx[sample(gok)])
    # 센서 열 간격 = 이 패스의 측정 (추출기가 프로파일마다 잰 dx 의 중앙값; 없으면 센서 모형 dX(z)) — 증거 점 개수의 기준
    dxs = [float(getattr(feats[j], "dx_mm", float("nan")) or float("nan")) for j in sample(gok) if feats[j] is not None]
    dxs = [v_ for v_ in dxs if np.isfinite(v_) and v_ > 0.0]
    dx_plate_mm = float(np.median(dxs)) if dxs else (float(dX_mm(z_mm)) if np.isfinite(z_mm) else float("nan"))
    q_m = max(quant_sigma_mm(z_mm), dx_plate_mm / math.sqrt(12.0) if np.isfinite(dx_plate_mm) else 0.0) * 1e-3   # 양자화 하한
    gap_m = float("nan")
    if spec is not None and float(spec.root_gap_mm) > 0.0:
        gap_m = float(spec.root_gap_mm) * 1e-3
    elif getattr(sf, "root_gap_mm", None) is not None:
        rg = np.asarray(sf.root_gap_mm, dtype=float)
        rg = rg[np.isfinite(rg)]
        gap_m = float(np.median(rg)) * 1e-3 if rg.size else float("nan")
    depth_m = float("nan")
    if spec is not None and float(spec.groove_depth_mm) > 0.0:
        depth_m = float(spec.groove_depth_mm) * 1e-3
    elif getattr(sf, "depth_mm", None) is not None:
        dd = np.asarray(sf.depth_mm, dtype=float)
        dd = dd[np.isfinite(dd)]
        depth_m = float(np.median(dd)) * 1e-3 if dd.size else float("nan")
    # 면 판별 (원칙 2): 판 상면 ↔ 그루브 바닥 (깊이만큼 떨어진 설계 면) 의 중간 = 깊이/2. 깊이를 모르면 판 평면 적합의 k·σ (통계)
    surf_tol = 0.5 * depth_m if np.isfinite(depth_m) else kk * max(float(fit.diag.get("plane_sigma_m", q_m) or q_m), q_m)
    # 루트 직선 잔차 σ (판 법선·측면 성분 중 큰 것; 하한 양자화)
    sig_root = q_m
    if int(gok.sum()) >= 3:
        rel = P[gok] - c
        perp = rel - np.outer(rel @ u, u)
        sig_root = max([q_m] + [s_ for s_ in (robust_sigma(perp @ pn), robust_sigma(perp @ w)) if np.isfinite(s_)])
    # 루트가 이 그루브의 루트인가 = 면 판별 (원칙 2): 갭 안 (정련 직선에서 설계 갭/2). 갭을 모르면 통계 (k·σ_루트)
    line_tol = 0.5 * gap_m if (np.isfinite(gap_m) and gap_m > 0.0) else kk * sig_root

    def h_plate(Q: np.ndarray) -> np.ndarray:
        """판 상면 (이 패스의 판 평면) 에서 판 법선 방향 높이."""
        return Q @ pn + pdd

    plate_ev = evidence_count(pr.plate_min_len_m * 1e3 / dx_plate_mm) if np.isfinite(dx_plate_mm) else 1
    side_bin = float(pr.plate_min_len_m)
    confirm = pr.confirm_profiles
    # 끝 증거 구간 = 지연 + 단면 창, 그리고 적어도 확인 프로파일 수만큼의 착점 (+ 반 칸: 위치 사이 중간) — 성긴 프레임 (CPU 부하로 착점
    # 간격 5.5–6 mm, 2026-10-04 e2e) 에서도 가림·관측 확인이 가능하게
    look = max(pr.look_m, (confirm + DESIGN_HALF) * sp) if np.isfinite(sp) else pr.look_m

    # ---- a. on-specimen 판정 ----
    nom = None
    if spec is not None:
        nom = {"root_gap_mm": spec.root_gap_mm, "depth_mm": spec.groove_depth_mm, "top_opening_mm": spec.top_opening_mm}
    on = np.zeros(M, dtype=bool)
    plane_present = np.zeros(M, dtype=bool)
    groove_present = np.zeros(M, dtype=bool)                    # 그루브 바닥·루트가 심 직선 위에 있고 설계 밴드 안
    reason = [""] * M
    bands = [dict() for _ in range(M)]                          # 판별 측면 대역 (v_min, v_max) — 판 상면 위 점
    med_h = [dict() for _ in range(M)]                          # 판별 판 상면 점 높이의 중앙값 (끝 괄호의 국소 기준)
    spread = []                                                 # 판 상면 점이 그 프로파일 중앙값에서 흩어진 양 (점 잡음)
    for j in range(M):
        # 그루브 증거 (판과 별도로: 판 끝에서 어느 쪽이 먼저 사라졌는지가 끝 위치의 괄호를 정한다)
        g_reason = ""
        if not gok[j]:
            g_reason = "no_groove"
        else:
            rel = P[j] - c
            if float(np.linalg.norm(rel - u * float(rel @ u))) > line_tol:
                g_reason = "root_off_line"
            elif nom is not None:
                bad = []
                for k_, v0 in nom.items():
                    val = float(getattr(sf, k_)[j]) if getattr(sf, k_, None) is not None else float("nan")
                    if k_ == "root_gap_mm" and not np.isfinite(val):
                        continue                                    # 갭 판독 불가 (베벨면 미검출, gap_unresolved_no_bevel): 개구·깊이로
                    if not (np.isfinite(val) and design_match(val, v0)):
                        bad.append(k_)
                if bad:
                    g_reason = "out_of_design_band:" + ",".join(bad)
        groove_present[j] = g_reason == ""
        ok_sides = 0
        for side in ("left", "right"):
            Q = side_points(j, side)
            if Q.shape[0] == 0:
                continue
            hq_ = h_plate(Q)
            inl = np.abs(hq_) <= surf_tol
            if inl.sum() < 2:
                continue
            med_h[j][side] = float(np.median(hq_[inl]))
            if gok[j]:
                spread.append(hq_[inl] - med_h[j][side])
            vv = (Q[inl] - c) @ w
            if float(vv.max() - vv.min()) >= pr.plate_min_len_m:
                plane_present[j] = True
                bands[j][side] = (float(vv.min()), float(vv.max()))
                if float(inl.mean()) >= DESIGN_HALF:                # 판 점의 절반 이상이 판 상면 위 (원칙 2)
                    ok_sides += 1
        if ok_sides < 2:
            reason[j] = "plates_off_plane_or_missing"
            continue
        if g_reason:
            reason[j] = g_reason
            continue
        on[j] = True
    fit.diag["on_specimen_n"] = int(on.sum())
    # 판 끝 괄호·'아래 면' 증거의 정밀도 (원칙 1): 끝 프로파일 자신의 판 상면 높이 (med_h) 를 기준으로, 판 점 잡음 σ_점 (프로파일 안 점이
    # 그 프로파일 중앙값에서 흩어진 강건 σ) 과 이웃 프로파일 사이 판 높이 흔들림 σ_이웃 (연속 on-specimen 프로파일 중앙값 차/√2 의 강건 σ —
    # 로봇 진동) 의 k 배. 판 끝을 지나는 몇 프로파일만 보므로 국소 기준이 휨·진동을 따라간다.
    s_pt = robust_sigma(np.concatenate(spread)) if spread else float("nan")
    steps = []
    for side in ("left", "right"):
        mm_ = np.array([med_h[j].get(side, np.nan) if on[j] else np.nan for j in range(M)])
        d_ = np.diff(mm_)
        steps.append(d_[np.isfinite(d_)] / math.sqrt(2.0))
    steps = np.concatenate(steps) if steps else np.zeros(0)
    s_step = robust_sigma(steps) if steps.size >= 3 else 0.0
    sig_edge = math.hypot(max(s_pt, q_m) if np.isfinite(s_pt) else q_m, s_step if np.isfinite(s_step) else 0.0)
    edge_tol = kk * sig_edge
    # 바닥 증거 (c): 루트 직선에서 측면 ± 갭/4 (갭 가운데 절반 — 벽 모서리 제외, 원칙 2) 이고 루트 직선에서 판 법선 방향 깊이/2 안 (판
    # 상면과의 중간 — 면 판별, 원칙 2) 의 점. 직선 기준이라 끝 프로파일 하나의 (판 끝에 걸쳐 틀어진) 루트 점에 좌우되지 않는다. 갭 = spec
    # 공칭, 없으면 이 패스의 루트 갭 중앙값.
    floor_half = 0.5 * DESIGN_HALF * gap_m if (np.isfinite(gap_m) and gap_m > 0.0) else float("nan")
    dx_floor_mm = (dx_plate_mm * float(dX_mm(z_mm + (depth_m * 1e3 if np.isfinite(depth_m) else 0.0))) / float(dX_mm(z_mm))
                   if (np.isfinite(z_mm) and np.isfinite(dx_plate_mm)) else dx_plate_mm)   # 바닥은 깊이만큼 멀다 (열 간격 ∝ dX(z))
    floor_need = evidence_count(2.0 * floor_half * 1e3 / dx_floor_mm) if (np.isfinite(floor_half) and np.isfinite(dx_floor_mm)) else 1
    _floor_cache: dict = {}

    def floor_band(j: int) -> tuple:
        """프로파일 j 의 갭 가운데 대역 점 (u 좌표, 루트 직선에서 판 법선 방향 거리) — 그루브 바닥 후보 (깊이/2 안)."""
        if j in _floor_cache:
            return _floor_cache[j]
        out_ = (np.zeros(0), np.zeros(0))
        if np.isfinite(floor_half):
            A = all_points(j)
            if A.shape[0]:
                rel = A - c
                s_ = rel @ u
                perp = rel - np.outer(s_, u)
                r_ = perp @ pn
                sel = (np.abs(perp @ w) <= floor_half) & (np.abs(r_) <= surf_tol)
                out_ = (s_[sel], r_[sel])
        _floor_cache[j] = out_
        return out_

    # 같은 바닥이 이어지나 (원칙 1): 이웃 프로파일 바닥 중앙값의 차 — on-specimen 프로파일 (고르게 뽑은 이웃 쌍) 에서 잰 강건 σ 의 k 배
    # (하한 양자화). 바닥이 천천히 떠도는 것 (실기 3G 아래 끝 20 mm 동안 0.1–0.4 mm, 로봇 진동) 은 따라가고, 다른 면 (판 끝 너머
    # 1–5 mm 다른 받침·테이블, 실기 at19b) 의 계단은 끊는다. 프로파일 안 점의 흩어짐 (판 끝 반사 구간에서 ±0.3 mm) 은 중앙값이 견딘다.
    diffs = []
    if np.isfinite(floor_half):
        for j in sample(on):
            j = int(j)
            if j + 1 < M and on[j + 1]:
                _, r0 = floor_band(j)
                _, r1 = floor_band(j + 1)
                if r0.size >= floor_need and r1.size >= floor_need:
                    diffs.append(float(np.median(r1)) - float(np.median(r0)))
    s_fstep = robust_sigma(np.array(diffs)) if len(diffs) >= 3 else float("nan")
    floor_step_tol = kk * max(s_fstep, q_m) if np.isfinite(s_fstep) else float("nan")
    if not np.isfinite(floor_step_tol):
        floor_half = float("nan")                                  # 바닥을 잴 수 없다: 그루브 전이 규칙만

    def floor_profile(j: int) -> Optional[tuple]:
        """프로파일 j 의 그루브 바닥 (u 좌표들, 높이 중앙값) — 갭 가운데 대역에 바닥 후보가 열 수의 절반 이상일 때, 아니면 None."""
        if not np.isfinite(floor_half):
            return None
        s_, r_ = floor_band(j)
        return (s_, float(np.median(r_))) if r_.size >= floor_need else None

    fit.diag["end_derived"] = {
        "k": kk, "spacing_m": sp, "sensor_z_mm": z_mm, "dX_mm": dx_plate_mm, "quant_sigma_m": q_m, "surface_tol_m": surf_tol,
        "plate_point_sigma_m": s_pt, "plate_step_sigma_m": s_step, "edge_tol_m": edge_tol,
        "root_sigma_m": sig_root, "line_tol_m": line_tol, "floor_half_band_m": floor_half, "floor_min_points": int(floor_need),
        "floor_step_sigma_m": s_fstep, "floor_step_tol_m": floor_step_tol,
        "plate_min_len_m": float(pr.plate_min_len_m), "plate_evidence_points": int(plate_ev), "look_m": look,
        "bridge_m": float(pr.window_m), "confirm_profiles": int(confirm)}
    if not on.any():
        fit.flags.append("ends_not_identified")
        fit.s_a = fit.s_b = float("nan")
        return fit
    # ---- b. 가운데를 포함하는 연속 구간 ----
    on_idx = np.flatnonzero(on & np.isfinite(t))
    t_on = t[on_idx]
    mid = 0.5 * (float(t_on.min()) + float(t_on.max()))
    j0 = int(on_idx[int(np.argmin(np.abs(t_on - mid)))])

    def grow(step: int) -> int:
        last = j0
        k_ = j0 + step
        while 0 <= k_ < M:
            if on[k_]:
                last = k_
                k_ += step
                continue
            k2 = k_
            while 0 <= k2 < M and not on[k2]:
                k2 += step
            if not (0 <= k2 < M):
                break
            seg = range(k_, k2, step)
            if not all(plane_present[q] for q in seg):
                break
            dist = abs(float(t[k2]) - float(t[last]))
            if dist > pr.window_m:                                    # 양 판이 보이는 동안의 결측 (반사) 은 단면 창까지만 잇는다
                break
            bridged.append({"from_t_m": float(t[last]), "to_t_m": float(t[k2]), "n": len(seg),
                            "reasons": sorted({reason[q] for q in seg if reason[q]})})
            last = k2
            k_ = k2 + step
        return last

    bridged: list = []
    jA, jB = grow(-1), grow(+1)
    fit.diag["bridged_gaps"] = bridged
    # ---- c/d/e. 끝마다 ----
    ends = {}
    for which, j_end, step in (("first", jA, -1), ("last", jB, +1)):
        sgn = float(np.sign(t[jB] - t[jA])) if np.isfinite(t[jB] - t[jA]) and t[jB] != t[jA] else 1.0
        out_sign = sgn * step                                   # 바깥쪽 = u 증가(+1) / 감소(−1)
        # d. 관측: 끝 너머 연속 비시편 프로파일 (유령점 버스트 + 1 개 이상이면 관측)
        beyond = []
        k_ = j_end + step
        while 0 <= k_ < M and not on[k_]:
            if np.isfinite(t[k_]):
                beyond.append(k_)
            k_ += step
        span = max((abs(float(t[q]) - float(t[j_end])) for q in beyond), default=0.0)
        # 서로 다른 위치의 너머 프로파일 수: 앞 것에서 착점 간격의 절반 넘게 나아간 것만 센다 (멈춘 로봇이 같은 곳에서 찍은 프레임은 한
        # 관측 — 2026-10-06 g3_07 패스 1: 끝 너머 5 프레임이 0.03 mm 안, 실제 끝은 17 mm 더)
        n_distinct, t_prev = 0, float(t[j_end])
        for q in beyond:
            if abs(float(t[q]) - t_prev) > DESIGN_HALF * sp:
                n_distinct += 1
                t_prev = float(t[q])
        observed = bool(n_distinct >= confirm)
        # c. 판별 판 모서리. 끝면은 수직이므로 판 상면과 그루브 바닥(루트) 의 끝은 같은 u 다 → 끝 = 증거의 괄호 [LB, UB] 의 중점.
        #    하한 LB (끝이 적어도 여기까지): 판 상면 인라이어가 있는 가장 바깥 u (마지막 on-specimen 프로파일 + 요로 일부만 판 위인
        #    너머 프로파일), 바닥 점이 있는 가장 바깥 u. 상한 UB (끝이 이보다 안쪽): 판 대역에 판 상면이 없는 칸의 판 상면 착점선 u,
        #    바닥 점이 없는 첫 너머 프로파일의 루트 착점 t. 레이저 면이 진행 방향으로 기울면 (피치·입사) 상면 착점선과 바닥 착점선이
        #    깊이·tan 만큼 어긋난다 — 한 끝에서는 상면이 먼저, 다른 끝에서는 바닥이 먼저 판을 벗어난다. 바닥 증거는 **바닥 점을 직접**
        #    본다 (리뷰 라운드 4 F2). 끝 근처에서 바닥이 보이지 않으면 (뒷댐재 없음·어두운 바닥) 그루브가 사라진 첫 프로파일의 t 를
        #    상한으로 쓴다. 안전망: LB 바깥에 있지 않은 상한 후보는 버린다 (모순된 증거; diag 'UB_dropped'). 대역 전체가 판 상면 위인
        #    너머 프로파일 (그루브 없는 동일 평면 연장 = 런오프 탭, 또는 상면 착점선이 바닥보다 뒤처짐) 은 상면 증거로 쓰지 않는다.
        # 증거 구간의 바깥 끝: 끝 프로파일에서 look, 단 판 상면이 먼저 사라져도 그루브 (벽·루트) 가 계속 보이면 판은 아직 있다 — 그 마지막
        # 그루브 프로파일에서 look (3G 아래 끝: 지지면 반사로 판 상면 점이 끝 ~25 mm 앞에서 사라지고 그루브·바닥은 끝까지 보인다, 2026-10-06
        # g3_07; 1G 판 끝 너머 받침은 바닥만 있고 벽이 없다). 그루브 결측은 단면 창까지 잇는다.
        t_gr = float(t[j_end])
        for q in beyond:
            if abs(float(t[q]) - t_gr) > pr.window_m:
                break
            if groove_present[q]:
                t_gr = float(t[q])
        far = (max(float(t[j_end]), t_gr) if out_sign > 0 else min(float(t[j_end]), t_gr)) + out_sign * look

        def outer(a: float, b: float) -> float:
            return max(a, b) if out_sign > 0 else min(a, b)

        def outermost(a: np.ndarray) -> float:
            return float(a.max()) if out_sign > 0 else float(a.min())

        # 바닥이 보이는 가장 바깥 프로파일: 마지막 on-specimen 프로파일부터 안쪽으로 look 안 (끝에 걸친 프로파일은 그루브로 잡혀도
        # 바닥이 없을 수 있다 — 그 루트 점은 바닥이 아니다)
        j_fl, fp_fl = None, None
        k_ = j_end
        while 0 <= k_ < M and np.isfinite(floor_half):
            if np.isfinite(t[k_]) and abs(float(t[k_]) - float(t[j_end])) > look:
                break
            fp = floor_profile(k_)
            if fp is not None:
                j_fl, fp_fl = k_, fp
                break
            k_ -= step
        lb_floor = float(t[j_end])
        ub_floor = float("nan")
        if j_fl is not None:
            floor_rule = "floor_points"
            lb_floor = outermost(fp_fl[0])
            ref_q = fp_fl[1]                                      # 따라가는 바닥 높이 (직전 바닥 프로파일의 중앙값)
            miss: list = []                                       # 바닥 없는 연속 프로파일 (서로 다른 위치) — 유령점 버스트 + 1 이면 끝
            q = j_fl + step
            while 0 <= q < M:
                if np.isfinite(t[q]):
                    if out_sign * (float(t[q]) - far) > 0:
                        break
                    fp = floor_profile(q)
                    if fp is not None and abs(fp[1] - ref_q) <= floor_step_tol * (len(miss) + 1):
                        lb_floor = outer(lb_floor, outermost(fp[0]))
                        ref_q = fp[1]
                        miss = []
                    else:
                        if not miss or abs(float(t[q]) - float(t[miss[-1]])) > DESIGN_HALF * sp:
                            miss.append(q)
                        if len(miss) >= confirm:
                            ub_floor = float(t[miss[0]])
                            break
                q += step
            if miss and not np.isfinite(ub_floor):
                ub_floor = float(t[miss[0]])                      # 구간 끝까지 바닥이 다시 나오지 않았다
        else:
            floor_rule = "groove_presence"                      # 바닥이 보이지 않는다 (뒷댐재 없음 등): 그루브 전이 (이전 규칙)
            for q in beyond:
                if not groove_present[q]:
                    ub_floor = float(t[q])
                    break
        side_edges = {}
        for side, (v_lo, v_hi) in bands[j_end].items():
            ref = med_h[j_end].get(side, 0.0)                     # 끝 프로파일의 이 판 상면 높이 (국소 기준)
            Q = side_points(j_end, side)
            Qi = Q[np.abs(h_plate(Q) - ref) <= edge_tol]
            if Qi.shape[0] == 0:
                Qi = Q
            uu = (Qi - c) @ u
            top_on = float(uu.max()) if out_sign > 0 else float(uu.min())
            LB = outer(top_on, lb_floor)
            UB_top = float("nan")
            nb = max(2, int(math.ceil((v_hi - v_lo) / side_bin)))
            edges_v = np.linspace(v_lo, v_hi, nb + 1)
            centres = 0.5 * (edges_v[:-1] + edges_v[1:])
            for q in beyond:
                if out_sign * (float(t[q]) - far) > 0:
                    break
                A = all_points(q)
                on_bin = np.zeros(nb, dtype=bool)
                umax_on = -np.inf if out_sign > 0 else np.inf
                if A.shape[0]:
                    vq = (A - c) @ w
                    onp = (vq >= v_lo) & (vq <= v_hi) & (np.abs(h_plate(A) - ref) <= edge_tol)
                    if onp.any():
                        b = np.clip(((vq[onp] - v_lo) / (v_hi - v_lo) * nb).astype(int), 0, nb - 1)
                        on_bin[b] = True
                        uq = (A[onp] - c) @ u
                        umax_on = float(uq.max()) if out_sign > 0 else float(uq.min())
                if on_bin.all():
                    continue                                    # 대역 전체가 판 상면 위: 판 끝 증거 아님 (탭 / 상면 착점선 지연)
                line_u = np.array([_line_on_plane_u(T[q], pn, pdd - ref, c, u, w, vb) for vb in centres])
                off = ~on_bin & np.isfinite(line_u)
                if off.any():
                    ub_q = float(line_u[off].min()) if out_sign > 0 else float(line_u[off].max())
                    UB_top = ub_q if not np.isfinite(UB_top) else (min(UB_top, ub_q) if out_sign > 0 else max(UB_top, ub_q))
                if on_bin.any():                                # 부분 (요): 판 위 점은 모서리 안
                    LB = outer(LB, umax_on)
                else:
                    break                                       # 이 대역에 판 상면이 더는 없다
            cand = [x for x in (UB_top, ub_floor) if np.isfinite(x)]
            ubs = [x for x in cand if out_sign * (x - LB) > 0]  # 안전망: 하한 바깥의 상한만 (모순된 증거는 버린다)
            UB = (min(ubs) if out_sign > 0 else max(ubs)) if ubs else float("nan")
            edge = 0.5 * (LB + UB) if np.isfinite(UB) else LB
            # 끝 σ (원칙 1): 괄호 [판 위 마지막, 판 밖 첫] 안 어딘가 (균일) → 폭/√12; 괄호가 열려 있으면 착점 간격/√12
            sig_e = abs(UB - LB) / math.sqrt(12.0) if np.isfinite(UB) else end_sigma_m(sp)
            side_edges[side] = {"U_on": LB, "U_off": UB, "UB_top": UB_top, "UB_floor": ub_floor, "LB_floor": lb_floor,
                                "floor_rule": floor_rule, "UB_dropped": len(cand) - len(ubs), "edge": edge, "sigma_m": float(sig_e)}
        stagger_tol = float("nan")
        if side_edges:
            es = [v_["edge"] for v_ in side_edges.values()]
            s_plates = float(np.mean(es))
            stagger = float(max(es) - min(es)) if len(es) == 2 else 0.0
            sg = [v_["sigma_m"] for v_ in side_edges.values() if np.isfinite(v_["sigma_m"])]
            stagger_tol = kk * math.sqrt(sum(x * x for x in sg)) if len(sg) == 2 else float("nan")
            if len(es) == 2 and np.isfinite(stagger_tol) and stagger > stagger_tol:
                s_end = min(es) if out_sign > 0 else max(es)     # 안쪽 판 (그루브가 온전한 곳)
                if "plates_staggered" not in fit.flags:
                    fit.flags.append("plates_staggered")
            else:
                s_end = s_plates
        else:
            stagger = float("nan")
            s_end = s_plates = float(t[j_end])
        # e. 가림: 끝 너머 look 안의 비시편 프로파일을 바깥쪽 순서로 본다 (판 측면 범위 안의 점만). 점의 높이 = 끝 프로파일의 판 상면
        #    (국소 기준) 에서 판 법선 방향 거리. '위' = 깊이/2 넘게 (다른 면 — 판 위 물체; 면 판별, 원칙 2: 판 끝 모서리의 버 0.3–1 mm 는
        #    물체가 아니다), '아래' = 끝 괄호 정밀도 (k·σ) 넘게 (판 상면이 아닌 낮은 면 — 판 끝면·테이블):
        #    * 위 점이 '면 증거' (판 최소 면 길이의 열 수 절반) 이상 → 판 위 물체가 끝을 덮음 → 'end_occluded'.
        #    * 그보다 먼저 판 아래 점 (테이블·바닥) 이 그만큼 보이면 → 판이 끝나고 아래 면이 드러남 → 끝 관측 (멈춤; 그 너머 물체는 무관).
        #      아래 점은 **판 대역 (각 판 상면의 측면 범위) 안에서만** 센다 — 두 판 사이 그루브 바닥은 늘 판보다 낮으므로, 클램프 모서리에
        #      걸친 과도 프로파일이 그루브 바닥 일부를 보이면 '아래 면이 먼저' 로 오판해 클램프 모서리를 판 끝으로 관측했다 (2026-10-02,
        #      e2e 가림 끝 시험). 판 위 물체 (위 점) 는 그루브 위를 포함한 전체 범위에서 센다.
        #    * 점이 없거나 판 높이 점뿐 (그늘·허공·물체 모서리에 걸친 프로파일) → 다음 프로파일.
        #    * 위 점 증거는 유령점 버스트 + 1 개의 **연속** 프로파일에서 나와야 가림 (판 높이 점뿐인 프로파일이 연속을 끊는다, 빈
        #      프로파일은 건너뜀) — 반짝이는 개선면 반사로 생기는 산발 유령점이 판 끝을 '가림' 으로 만들지 않게 (2026-10-02 at06).
        #    * 한 프로파일에 위 점과 아래 면 점이 함께 있으면 **많은 쪽** 을 따른다 — 판 끝 모서리의 작은 돌출·반사 몇 점이 테이블보다
        #      먼저 세어져 판 끝을 '가림' 으로 만들었다 (2026-10-02 at07 패스 3). 클램프는 수백 점으로 덮는다.
        #    첫 프로파일 하나만 보면 안 된다: 레이저 면이 진행 방향으로 기울어 그 프로파일이 물체 모서리에 걸쳐 점이 모자라면 가림을 놓치고
        #    물체 모서리를 판 끝으로 '관측' 해 버린다 (판 끝을 덮은 클램프 → 길이가 클램프 폭만큼 짧게, 2026-10-01).
        occluded = False
        occl_look: list = []                                         # 진단: 너머 프로파일별 (t, 위 점, 판 대역 아래 점)
        if beyond and bands[j_end]:
            v_all_lo = min(b_[0] for b_ in bands[j_end].values())
            v_all_hi = max(b_[1] for b_ in bands[j_end].values())
            plate_bands = list(bands[j_end].values())
            refs = [med_h[j_end].get(s_, 0.0) for s_ in bands[j_end]]
            ref_top = max(refs) if refs else 0.0                      # 위 = 높은 판보다, 아래 = 낮은 판보다
            ref_low = min(refs) if refs else 0.0
            look_q = [q for q in beyond if abs(float(t[q]) - float(t[j_end])) <= look] or beyond[:1]
            counts = []                                                  # 너머 프로파일별 (위 점, 판 대역 아래 점), 빈 프로파일은 None
            for q in look_q:
                A = all_points(q)
                if not A.shape[0]:
                    counts.append(None)
                    continue
                vq = (A - c) @ w
                dq = h_plate(A)
                inside = (vq >= v_all_lo) & (vq <= v_all_hi)
                on_plates = np.zeros(vq.shape, dtype=bool)
                for lo_, hi_ in plate_bands:
                    on_plates |= (vq >= lo_) & (vq <= hi_)
                counts.append((int(np.sum(inside & (dq - ref_top > surf_tol))), int(np.sum(on_plates & (dq - ref_low < -edge_tol)))))
            # 연속 위 점 프로파일 수 (반사 유령점은 산발). 창 안 프로파일이 적으면 그 수가 상한인데, 끝 바로 너머의 증거 없는 프로파일
            # (빈 것·판 높이 점뿐: 그늘, 물체 모서리에 걸침 — 위 주석) 은 상한에서 뺀다: 성긴 프레임 (촬영 멈춤·빠른 속도) 에서 창에 3 개뿐이고
            # 첫 것이 클램프 벽에 걸치면 남은 연속 2 개로 3 을 못 채워 클램프 모서리를 판 끝으로 '관측' 했다 (2026-10-04, e2e 가림 끝 시험,
            # CPU 부하로 착점 간격 5.7–6 mm). 첫 증거 뒤의 연속 요구는 그대로라 산발 유령점 배제는 같다.
            lead = next((i for i, n in enumerate(counts) if n is not None and max(n) >= plate_ev), len(counts))
            need = max(1, min(confirm, len(look_q) - lead))
            run_above = 0
            for q, n in zip(look_q, counts):
                if n is None:
                    continue
                n_up, n_dn = n
                if len(occl_look) < 64:
                    occl_look.append([round(float(t[q]), 5), n_up, n_dn])
                if n_up >= plate_ev and n_up >= n_dn:                   # 위 점이 우세할 때만 가림 증거
                    run_above += 1
                    if run_above >= need:
                        occluded = True
                        break
                    continue
                if n_dn >= plate_ev:                                    # 아래 면 (판 대역) 이 우세 → 판 끝 관측
                    break
                run_above = 0                                         # 판 높이 점뿐인 프로파일: 위 점 연속이 끊김 (빈 프로파일은 위에서 건너뜀)
        if occluded:
            observed = False
            fit.flags.append(f"end_occluded_{which}")
        if not observed:
            fit.flags.append(f"end_not_observed_{which}")
        # 끝 위치의 σ (원칙 1): 판마다 괄호 σ 중 큰 것 (판이 없으면 착점 간격/√12)
        sg_all = [v_["sigma_m"] for v_ in side_edges.values() if np.isfinite(v_["sigma_m"])]
        sig_end = max(sg_all) if sg_all else end_sigma_m(sp)
        ends[which] = {"observed": observed, "occluded": occluded, "occlusion_look": occl_look,
                       "s_m": float(s_end), "s_plates_m": float(s_plates), "point": (c + u * s_end).tolist(), "sigma_m": float(sig_end),
                       "off_specimen_span_m": float(span), "profile_inside": int(j_end), "n_beyond": len(beyond),
                       "n_beyond_distinct": int(n_distinct),
                       "sides": side_edges, "stagger_m": stagger, "stagger_tol_m": stagger_tol, "t_inside_m": float(t[j_end]),
                       "next_reason": reason[beyond[0]] if beyond else None}
    # 설계 길이로 본 대안 끝 (보고; 교차 검사·추정 끝은 게이트/호출자)
    if spec is not None and spec.weld_length_mm > 0:
        Ls = spec.weld_length_mm * 1e-3
        d_ab = float(np.sign(ends["last"]["s_m"] - ends["first"]["s_m"]) or 1.0)
        ends["first"]["alt_s_m"] = ends["last"]["s_m"] - d_ab * Ls
        ends["last"]["alt_s_m"] = ends["first"]["s_m"] + d_ab * Ls
        for wh in ("first", "last"):
            ends[wh]["alt_point"] = (c + u * ends[wh]["alt_s_m"]).tolist()
    fit.s_a, fit.s_b = sorted((ends["first"]["s_m"], ends["last"]["s_m"]))
    hr = t[np.isfinite(t)]
    fit.diag["hit_range_m"] = [float(hr.min()), float(hr.max())] if hr.size else [float("nan")] * 2
    fit.diag["spacing_m"] = sp
    fit.diag["ends"] = ends
    fit.diag["run"] = [int(jA), int(jB)]
    fit.diag["end_params"] = pr.to_dict()
    fit.diag["off_specimen_reasons"] = {r_: int(sum(1 for x in reason if x == r_)) for r_ in sorted(set(reason)) if r_}
    # 진단: 판 끝과 연속 구간 경계 사이 (시편 안이지만 on-specimen 이 아닌) 프로파일의 사유
    t_run = sorted((float(t[jA]), float(t[jB])))
    gap_r = {}
    for wh in ("first", "last"):
        se_ = float(ends[wh]["s_m"])
        lo_, hi_ = (se_, t_run[0]) if se_ < t_run[0] else (t_run[1], se_)
        sel = [q for q in range(M) if np.isfinite(t[q]) and lo_ < float(t[q]) < hi_]
        gap_r[wh] = {"span_m": float(hi_ - lo_), "n": len(sel),
                     "reasons": {r_: int(sum(1 for q in sel if reason[q] == r_)) for r_ in sorted({reason[q] for q in sel})}}
    fit.diag["end_to_run_reasons"] = gap_r
    return fit


def support_plane_ends(result, sf, fit: PassFit, *, gravity_axis_base=(0.0, 0.0, -1.0), params: Optional[EndParams] = None) -> PassFit:
    """판 끝이 **지지면** (테이블·받침) 에서 끝나면 끝 = 루트 직선 ∩ 그 면 (사용자 2026-10-06: 3G 시편은 테이블에 서 있고 그루브 아래
    끝이 테이블과 바로 만난다). identify_seam_ends 뒤에, 관측된 (가려지지 않은) 끝마다: 레이저 평면 ∩ 루트 직선 착점이 끝 너머인
    프로파일 (스캔이 본 끝 너머 전부) 의 점 중 판 상면보다 센서 쪽으로 깊이/2 넘게 앞의 점 (판 상면이 아닌 면 — identify_seam_ends 의
    면 판별) → 적응 RANSAC 평면 (LMedS, k·σ). 판정 기준은
    scan_core.measure 원칙에서 (사용자 2026-10-07):
      * 판이 서 있다 = 판 법선이 중력과 45° 넘게 (수평에 더 가깝다, 원칙 2 각도 중간점) — 누운 판 (1G) 은 지지면 끝 없음.
      * 면이 수평이다 = 법선이 중력과 45° 안, 판을 가로지른다 = 판 법선과 45° 넘게 (1G 의 판 끝 너머 경사면은 서 있는 판에서만 보므로 무관).
      * 증거 = 인라이어가 유령점 버스트 + 1 프로파일 × '면 증거' (판 최소 면 길이 열 수의 절반) 이상이고 점의 절반 이상 (원칙 2).
      * 교점이 이 끝의 증거와 맞다 = 판 상면 증거 (U_on) 에서 안쪽으로 k·σ_끝 넘게 들어가지 않고, 바깥으로는 끝 구역 지연 (깊이 × tan
        입사 — 끝 근처에서 루트가 가려져 측정 끝이 모자랄 수 있는 길이) + k·σ_끝 과 스캔이 본 끝 너머 범위 (가장 먼 너머 프로파일의 루트
        착점) 를 넘지 않는다 (본 적 없는 곳으로 외삽하지 않는다).
    이면 그 끝을 교점으로 바꾼다 ('end_at_support_<first|last>'; 판 길이 끝 s_plates_m 도 — 지지면이 두 판을 함께 끝낸다).
    g3_07/g3_08: 아래 끝 근처 ~25 mm 는 레이저가 판에서 테이블로 반사돼 판 상면 점이 사라지고 끝을 바닥 점으로만 잡아 1–3 mm
    짧았다. diag['support_planes'] 에 면 (법선·점·높이·σ·점 수) 을 남긴다 (테이블 높이 실측 — 테이블 높이 입력 없이 가기 위한 자료)."""
    pr = params if params is not None else EndParams()
    kk = float(pr.k)
    ends = (fit.diag or {}).get("ends")
    pn = np.asarray((fit.diag or {}).get("plate_plane_n", np.full(3, np.nan)), dtype=float)
    pdd = float((fit.diag or {}).get("plate_plane_d", float("nan")))
    if not ends or not (np.all(np.isfinite(fit.c)) and np.all(np.isfinite(fit.u)) and np.all(np.isfinite(pn)) and np.isfinite(pdd)):
        return fit
    c, u = np.asarray(fit.c, dtype=float), unit(fit.u)
    up = -unit(gravity_axis_base)
    cos_half = math.cos(math.radians(HALF_TURN_DEG))
    if abs(float(pn @ up)) > cos_half:                              # 누운 판 (1G): 지지면 끝 없음
        return fit
    der = (fit.diag or {}).get("end_derived") or {}
    q_m = float(der.get("quant_sigma_m", quant_sigma_mm(None) * 1e-3))
    front_min = float(der.get("surface_tol_m", float("nan")))                 # 판 상면이 아닌 면 (면 판별: 깊이/2 넘게 앞)
    if not np.isfinite(front_min):
        front_min = kk * max(float((fit.diag or {}).get("plane_sigma_m", q_m) or q_m), q_m)
    need_pts = pr.confirm_profiles * int(der.get("plate_evidence_points", 1) or 1)
    T_all = np.asarray(result.T_base_opt, dtype=float)
    o = T_all[:, :3, 3]
    Y = T_all[:, :3, 1]
    den = Y @ u                                                     # 레이저 평면 ∩ 루트 직선: c + u t, (c + u t − o)·Y = 0
    t_hit = np.where(np.abs(den) > 1e-9, -np.einsum("ij,ij->i", c[None, :] - o, Y) / np.where(np.abs(den) > 1e-9, den, 1.0), np.nan)
    valid = getattr(result, "valid", None)
    s_all = [e["s_m"] for e in ends.values() if np.isfinite(e.get("s_m", float("nan")))]
    if len(s_all) < 2:
        return fit
    s_lo = min(s_all)
    planes = {}
    changed = False
    for which, e in ends.items():
        if not e.get("observed") or e.get("occluded"):
            continue
        s_end = float(e["s_m"])
        out = -1.0 if s_end <= s_lo + 1e-9 else 1.0                 # 바깥 방향 (u 좌표)
        beyond = np.flatnonzero(np.isfinite(t_hit) & ((t_hit - s_end) * out > 0.0))
        if beyond.size == 0:
            continue
        pts = []
        for i in beyond:
            x = np.asarray(result.x_mm[i], dtype=float)
            z = np.asarray(result.z_mm[i], dtype=float)
            v = np.isfinite(x) & np.isfinite(z) & (z > 0)
            if valid is not None:
                v &= np.asarray(valid[i], dtype=bool)
            if v.any():
                P = profile_points_base(T_all[i], x[v], z[v])
                d = P @ pn + pdd
                pts.append(P[d > front_min])
        P = np.vstack(pts) if pts else np.zeros((0, 3))
        info = {"n_points": int(P.shape[0]), "applied": False, "need_points": int(need_pts), "front_min_m": float(front_min)}
        if P.shape[0] >= need_pts:
            pl = ransac_plane(P, iters=200, k=kk, floor_mm=q_m * 1e3)
            ns = np.asarray(pl["normal"], dtype=float)
            if np.all(np.isfinite(ns)):
                ang = math.degrees(math.acos(min(1.0, abs(float(ns @ pn)))))
                p0 = -float(pl["d"]) * ns                           # 평면 위의 점 (원점에서 가장 가까운)
                dn = float(u @ ns)
                s_sup = float(((p0 - c) @ ns) / dn) if abs(dn) > 1e-6 else float("nan")
                shift = (s_sup - s_end) * out if np.isfinite(s_sup) else float("nan")
                level = abs(float(up @ ns)) >= cos_half
                h_sup = float(up @ (c + u * s_sup)) if (level and np.isfinite(s_sup)) else float("nan")   # 교점 (시편 발) 높이
                sides = e.get("sides") or {}
                u_on = [float(v_["U_on"]) for v_ in sides.values() if np.isfinite(float(v_.get("U_on", float("nan"))))]
                lb = (max(u_on) if out > 0 else min(u_on)) if u_on else s_end
                lb_shift = (lb - s_end) * out - kk * float(e.get("sigma_m", 0.0) or 0.0)     # 안쪽 한계 (판 상면 증거 − k·σ_끝)
                t_far = float(np.max(t_hit[beyond] * out)) * out
                # 바깥 한계: 스캔이 본 너머, 그리고 측정 끝이 모자랄 수 있는 만큼 — 판 끝 근처에서 루트·상면이 가려지는 기하 구간 (끝 구역
                # 지연 = 깊이 × tan 입사) + k·σ_끝 (실기 g3_07/08 보정 0.6–2.6 mm, 지연 ≈ 7 mm; 가상 팔 3G 의 40° 경사 받침은 34.6 mm 를
                # 옮기려 했다)
                far_shift = min((t_far - s_end) * out,
                                max(0.0, float(pr.lag_m)) + kk * float(e.get("sigma_m", 0.0) or 0.0))
                n_inl = int(round(float(pl["inlier_ratio"]) * float(pl["n"])))
                info.update(normal=ns.tolist(), point=p0.tolist(), rms_mm=float(pl["rms_mm"]), sigma_mm=float(pl.get("sigma_mm", float("nan"))),
                            inlier_ratio=float(pl["inlier_ratio"]), n_inliers=n_inl, angle_to_plate_deg=ang, s_m=s_sup, shift_m=shift,
                            shift_range_m=[float(lb_shift), float(far_shift)], height_m=h_sup, level=bool(level),
                            ok=bool(level and ang > HALF_TURN_DEG and float(pl["inlier_ratio"]) >= DESIGN_HALF and n_inl >= need_pts
                                    and np.isfinite(shift) and lb_shift <= shift <= far_shift))
                if info["ok"]:
                    # 교점 위치의 σ (원칙 1): 지지면 오프셋의 표준오차 (σ_면/√인라이어) 를 루트 직선 방향으로 (1/|u·n_면|), 하한 양자화
                    sig_sup = max(q_m, float(pl.get("sigma_mm", float("nan"))) * 1e-3 / math.sqrt(max(n_inl, 1)) / max(abs(dn), 1e-6))
                    e["sigma_m_measured"] = e.get("sigma_m")
                    e["sigma_m"] = float(sig_sup) if np.isfinite(sig_sup) else e.get("sigma_m")
                    e["s_m_measured"] = s_end
                    e["s_m"] = s_sup
                    e["s_plates_m"] = s_sup
                    e["point"] = (c + u * s_sup).tolist()
                    e["support_plane"] = True
                    info["applied"] = True
                    fit.flags.append(f"end_at_support_{which}")
                    changed = True
        planes[which] = info
    if planes:
        fit.diag["support_planes"] = planes
    if changed:
        f_, l_ = ends["first"], ends["last"]
        if "alt_s_m" in f_ and "alt_s_m" in l_:
            Ls = abs(float(f_["alt_s_m"]) - float(l_["s_m_measured"] if "s_m_measured" in l_ else l_["s_m"]))   # 설계 길이
            d_ab = float(np.sign(l_["s_m"] - f_["s_m"]) or 1.0)
            f_["alt_s_m"] = l_["s_m"] - d_ab * Ls
            l_["alt_s_m"] = f_["s_m"] + d_ab * Ls
            for wh in ("first", "last"):
                ends[wh]["alt_point"] = (c + u * ends[wh]["alt_s_m"]).tolist()
        fit.s_a, fit.s_b = sorted((f_["s_m"], l_["s_m"]))
    return fit


# --------------------------------------------------------------------------------------
# 6. 충실도 게이트
# --------------------------------------------------------------------------------------
_REQUIRED_FEATURES = ("root_gap_mm", "top_opening_mm", "depth_mm", "bevel_angle_deg")
_BEVEL_ONLY_FEATURES = ("bevel_angle_deg",)     # 베벨이 있는 종류(spec.n_bevels > 0; specimen.GROOVE_KINDS) 에서만 필수/평가
# 설계 치수 ↔ 작업 qc_targets 의 허용치 키 (설계 정보 — 보고 기준이자 '측정이 이 허용치를 구분하나' 의 기준)
_TOL_KEY = {"root_gap_mm": "root_gap_mm", "depth_mm": "groove_depth_mm", "bevel_angle_deg": "bevel_angle_deg",
            "top_opening_mm": "top_opening_mm"}


def _tol_or_nan(v) -> float:
    """qc_targets 의 tol: 없음·None (YAML null) = 보고만 → NaN (판정 안 함)."""
    return float("nan") if v is None else float(v)


def _required_features(spec: GrooveSpec) -> tuple:
    """필수 특징 — 베벨 각은 시편 종류에 베벨이 있을 때만 (square 그루브: 베벨 없음 → 미평가)."""
    if spec.n_bevels <= 0:
        return tuple(k for k in _REQUIRED_FEATURES if k not in _BEVEL_ONLY_FEATURES)
    return _REQUIRED_FEATURES


@dataclass
class Verdict:
    status: str                                   # PASS | ALIGN_FAIL | SPEC_FAIL | DATA_FAIL
    checks: dict = field(default_factory=dict)    # name → (value, limit, ok | None=미평가)
    notes: list = field(default_factory=list)
    stage: str = ""
    specimen: dict = field(default_factory=dict)  # 시편 대비 설계 보고 (판정 아님): 치수별 {median, along_min, along_max, design, tol, …}

    @property
    def failed(self) -> list:
        return [k for k, (_, _, ok) in self.checks.items() if ok is False]

    def to_dict(self) -> dict:
        return _plain({"status": self.status, "stage": self.stage, "notes": list(self.notes), "specimen": dict(self.specimen),
                       "checks": {k: {"value": v, "limit": lim, "ok": ok} for k, (v, lim, ok) in self.checks.items()}})

    def summary(self) -> str:
        return f"{self.stage} gate: {self.status}" + (f" failed={self.failed}" if self.failed else "") + \
            ("" if not self.notes else " notes=" + "; ".join(self.notes))


def _fin(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def fidelity_gate(feature_medians: dict, residuals: dict, spec: Optional[GrooveSpec], stage: str = "fidelity", *,
                  qc_targets: Optional[dict] = None, extent: Optional[dict] = None, accept_inferred_end: bool = False,
                  k: float = K_DEFAULT, coverage: Optional[dict] = None, validity: Optional[dict] = None) -> Verdict:
    """특징 중앙값 + 단면 충분성 + 양끝 → Verdict (충실도 우선 게이트, 사용자 2026-10-01; 판정 기준은 scan_core.measure 원칙, 2026-10-07).
    stage 는 'fidelity' 만 (예전 'coarse'/'fine' 정렬 게이트는 없앴다). 자세한 규칙은 _fidelity_gate. spec 필수 (설계 정보 없이 판정하지 않는다)."""
    if stage != "fidelity":
        raise ValueError("fidelity_gate: only stage 'fidelity' exists (the old coarse/fine alignment gates were removed)")
    if spec is None:
        raise ValueError("fidelity_gate: the specimen design (GrooveSpec) is required")
    return _fidelity_gate(feature_medians, residuals, spec, qc_targets=qc_targets, extent=extent,
                          accept_inferred_end=accept_inferred_end, k=k, coverage=coverage, validity=validity)


def _fidelity_gate(feature_medians: dict, residuals: dict, spec: GrooveSpec, *, qc_targets: Optional[dict] = None,
                   extent: Optional[dict] = None, accept_inferred_end: bool = False, k: float = K_DEFAULT,
                   coverage: Optional[dict] = None, validity: Optional[dict] = None) -> Verdict:
    """충실도 우선 게이트. 상태: DATA_FAIL (식별 미완) > SPEC_FAIL (측정이 설계를 구분하지 못함·특징 오인식·끝 모호) > PASS.
    판정 기준 (scan_core.measure 원칙; 설정 상수 없음 — 사용자 2026-10-07):
      * 필수 특징 (갭·깊이·개구, 베벨이 있으면 베벨각) 중앙값이 없으면 DATA_FAIL.
      * 단면 충분성 (coverage, 원칙 4): 용접 경로 단면 창마다 그루브 (+ 뒷댐재가 있으면 바닥·양쪽 루트 모서리) 유효 프로파일이
        단면 계획의 최소 수 이상 — 아니면 DATA_FAIL (모자란 단면을 노트에). coverage 없음 = 미평가.
      * 양끝 관측 (가려진 끝은 accept_inferred_end 일 때만 추정 끝) — 아니면 DATA_FAIL.
      * 데이터 유효성 (validity): 그루브 개구 ↔ 유효 프로파일 끝 여유 ≥ 판 맞춤에 필요한 판 길이 (plate_min_mm), 그루브 상부 원시 z 가
        센서 창 안 (도착 허용 만큼 안쪽, 바닥은 깊이만큼 더 멀다) — 아니면 DATA_FAIL. validity 없음 = 미평가.
      * 특징 오인식 (원칙 2): 갭·깊이·개구 중앙값이 설계 공칭 ±50 % 밖 → SPEC_FAIL. 그 안의 시편 편차는 보고만 (사용자 2026-10-02).
      * 측정이 설계 허용치를 구분하나 (원칙 1): 용접 경로가 쓰는 단면 창 중앙값의 불확도 (√(π/2)·σ/√n, σ = 프로파일 간 잡음 —
        심 방향 추세 제거, n = 단면 창 유효 프로파일 수의 중앙값) 에 k 를 곱한 값이 그 치수의 작업 허용치 (qc_targets) 보다 크면
        SPEC_FAIL. 허용치가 없는 치수는 보고만.
      * 길이: 양끝을 같은 패스에서 봤을 때 |길이 − 설계| ≤ 작업 허용치 (qc_targets weld_length_mm.tol, 없으면 0) 면 PASS 항목, 그
        위 k·σ_길이 (σ_길이 = √(σ_끝₁² + σ_끝₂²), σ_끝 = 끝 증거 괄호 폭/√12 — 괄호가 열려 있으면 착점 간격/√12) 까지는 보고,
        넘으면 'end_ambiguous' (SPEC_FAIL — 대안 끝 보고, 바꾸지 않는다). 처음 어긋나면 다음 패스로 다시 재고 그것도
        어긋나면 멈춘다 (확인 후 멈춤 — autoteach, 사용자 2026-10-06).
    시편 대비 설계 (치수 중앙값·구간 범위 vs 작업 허용치) 는 Verdict.specimen 에 보고만. 센서 배치 잔차·벽 전단은 진단 (ok None)."""
    tg = {kk: dict(v) for kk, v in (qc_targets or {}).items() if isinstance(v, dict)}
    fm = dict(feature_medians or {})
    rs = dict(residuals or {})
    checks: dict = {}
    notes: list = []
    data_fail = spec_fail = False

    def get(d, key):
        v = d.get(key, float("nan"))
        return float(v) if _fin(v) else float("nan")

    def tol_of(key):
        return _tol_or_nan((tg.get(_TOL_KEY.get(key, key)) or {}).get("tol"))

    has_bevel = spec.n_bevels > 0
    # ---- 데이터: 필수 특징 ----
    for key in _required_features(spec):
        if not _fin(fm.get(key)):
            checks[key] = (float("nan"), None, False)
            notes.append(f"{key} missing")
            data_fail = True
    # ---- 데이터: 단면 충분성 (원칙 4) ----
    ue = ((coverage or {}).get("groove") or {}).get("unjudged_ends_m")
    if ue and max(ue) > 0.0:
        notes.append(f"sections near the ends not judged for coverage (measured only while accelerating or hidden by the laser-plane "
                     f"lag): {ue[0] * 1e3:.1f} / {ue[1] * 1e3:.1f} mm at the low / high end - the weld path extrapolates there")
    for kind, cov in (coverage or {}).items():
        if cov is None:
            continue
        if kind in ("floor", "root_edges") and not spec.backing:
            checks[f"coverage_{kind}"] = (cov.get("min_count"), cov.get("need"), None)
            continue
        good = bool(cov.get("ok"))
        checks[f"coverage_{kind}"] = (cov.get("min_count"), cov.get("need"), good)
        if not good:
            sh = cov.get("short") or []
            notes.append(f"{kind}: {len(sh)}/{cov.get('n_stations')} weld-plan sections have fewer than {cov.get('need')} valid "
                         f"profiles (s {', '.join(f'{x * 1e3:.0f}' for x, _c in sh[:6])}{' ...' if len(sh) > 6 else ''} mm)")
            data_fail = True
    # ---- 데이터: 양끝 관측 (가려진 끝은 accept_inferred_end 일 때만 추정 끝으로) ----
    inferred_used = []
    if extent is None:
        checks["ends_observed"] = (None, True, False)
        notes.append("extent not evaluated")
        data_fail = True
    else:
        occl = list(extent.get("occluded") or [])
        for which in ("first", "last"):
            obs = bool(extent.get(f"{which}_observed", False))
            if which in occl:
                inf = extent.get(f"{which}_inferred")
                if accept_inferred_end and inf is not None:
                    checks[f"end_{which}_observed"] = (False, True, None)
                    notes.append(f"end_occluded ({which}): an object on the plate covers the end; inferred end {inf} accepted "
                                 "(auto_teach.accept_inferred_end)")
                    inferred_used.append(which)
                else:
                    checks[f"end_{which}_observed"] = (False, True, False)
                    checks[f"end_occluded_{which}"] = (True, False, False)
                    notes.append(f"end_occluded ({which}): an object on the plate covers the end - inferred end {inf} reported, "
                                 "not accepted (set auto_teach.accept_inferred_end to accept it)")
                    data_fail = True
                continue
            checks[f"end_{which}_observed"] = (obs, True, obs)
            if not obs:
                notes.append(f"{which} plate end not observed in this pass (seam continues beyond the scanned range)")
                data_fail = True
    # ---- 데이터 유효성: 판 맞춤에 필요한 판 길이가 FOV 안에 있나, 센서 창 ----
    vd = dict(validity or {})
    fmar = get(fm, "fov_margin_mm")
    pmin = vd.get("plate_min_mm")
    if np.isfinite(fmar) and pmin is not None:
        good = fmar >= float(pmin)
        checks["fov_margin_mm"] = (fmar, float(pmin), bool(good))
        data_fail |= not good
    else:
        checks["fov_margin_mm"] = (fmar, pmin, None)
    zc = get(fm, "sensor_z_mm")
    zwin = vd.get("z_valid_mm")
    if np.isfinite(zc) and zwin is not None:
        e_mm = float(vd.get("pose_tol_mm", 0.0))
        dpt = get(fm, "depth_mm")
        zlo = float(zwin[0]) + e_mm
        zhi = float(zwin[1]) - e_mm - (dpt if np.isfinite(dpt) else 0.0)
        good = zlo <= zc <= zhi
        checks["sensor_z_mm"] = (zc, (zlo, zhi), bool(good))
        data_fail |= not good
    else:
        checks["sensor_z_mm"] = (zc, None, None)
    # ---- 길이 교차 검사 (양끝이 같은 패스에서 관측됐을 때) — 어긋나면 'end_ambiguous' ----
    if extent is not None:
        # 판 길이 (양끝 모두 두 판 끝의 평균) 를 설계와 비교한다 — 판 끝이 엇갈린 시편에서 용접 경로 끝 (안쪽 판) 사이 길이는 엇갈림만큼
        # 짧지만 그것은 시편 편차 (보고) 이지 끝 식별 오류가 아니다 (2026-10-07 실기 재생: 엇갈림 0.4–1.8 mm)
        L = extent.get("plate_length_mm", extent.get("length_mm"))
        if not _fin(L):
            L = extent.get("length_mm")
        tolL = _tol_or_nan((tg.get("weld_length_mm") or {}).get("tol"))
        tolL = 0.0 if not np.isfinite(tolL) else tolL
        sp = get(extent, "spacing_mm")
        sig_sp = end_sigma_m(sp * 1e-3) * 1e3 if np.isfinite(sp) else float("nan")
        se = [get(extent, f"{w}_sigma_mm") for w in ("first", "last")]
        se = [v if np.isfinite(v) else sig_sp for v in se]          # 끝마다 증거 괄호의 σ (없으면 착점 간격/√12)
        sig_L = math.hypot(*se) if all(np.isfinite(v) for v in se) else float("nan")
        ambL = tolL + (float(k) * sig_L if np.isfinite(sig_L) else 0.0)
        # 보고 대역 (사용자 2026-10-06 결정, qc_targets weld_length_mm.ambiguous): 허용치 < 차 ≤ 이 값이면 보고만 — 끝 판독 산포 (~1 mm/끝)
        # 나 실제 시편 편차 (판 끝 엇갈림 등). 4a 에서 tol + k·σ 로 줄였다가 2026-10-08 1G 실기 (엇갈림 2.5 mm, 판 길이 381.4 mm) 가
        # SPEC_FAIL 로 멈춰 되돌림. 측정 산포가 더 크면 그 값 (tol + k·σ) 이 쓰인다
        ambU = _tol_or_nan((tg.get("weld_length_mm") or {}).get("ambiguous"))
        if np.isfinite(ambU):
            ambL = max(ambL, float(ambU))
        both = bool(extent.get("first_observed")) and bool(extent.get("last_observed"))
        if inferred_used:
            checks["weld_length_mm"] = (float(L) if _fin(L) else float("nan"), None, None)
            notes.append(f"length from the inferred {inferred_used} end (= design length by construction): not a measurement")
        elif both and _fin(L) and spec.weld_length_mm > 0:
            dev = abs(float(L) - spec.weld_length_mm)
            good = dev <= ambL
            # 허용치 안 = PASS, 허용치 < 차 ≤ 허용치 + k·√2·σ_끝 = 보고만 (끝 판독 산포 또는 실제 시편 편차), 넘으면 end_ambiguous
            checks["weld_length_mm"] = (float(L), (spec.weld_length_mm - tolL, spec.weld_length_mm + tolL), True if dev <= tolL else
                                        (None if good else False))
            if good and dev > tolL:
                notes.append(f"length {float(L):.2f} mm differs from the design {spec.weld_length_mm:.2f} mm by {dev:.2f} mm (> tol "
                             f"{tolL:.2f}, <= {ambL:.2f} mm = " + (f"report band qc_targets weld_length_mm.ambiguous {float(ambU):g} mm"
                             if np.isfinite(ambU) and float(ambU) >= ambL - 1e-9 else
                             f"tol + k {float(k):g} x length sigma {sig_L:.2f} mm") + "): reported, measured ends kept")
            if not good:
                checks["end_ambiguous"] = (True, False, False)
                notes.append(f"end_ambiguous: measured length {float(L):.2f} mm vs design {spec.weld_length_mm:.2f} mm, off by more than "
                             f"{ambL:.2f} mm (" + (f"max of the report band {float(ambU):g} mm and " if np.isfinite(ambU) else "")
                             + f"tol {tolL:.2f} + k x length sigma {sig_L:.2f} mm from the two end-evidence brackets) - "
                             f"alternative ends (other end +- design length): first {extent.get('first_alt')}, last "
                             f"{extent.get('last_alt')} (not substituted)")
            spec_fail |= not good
        else:
            checks["weld_length_mm"] = (float(L) if _fin(L) else float("nan"), None, None)
    # ---- 특징 오인식 (원칙 2: 공칭 ±50 %) + 시편 대비 설계 보고 ----
    nominal = {"root_gap_mm": spec.root_gap_mm, "depth_mm": spec.groove_depth_mm, "bevel_angle_deg": spec.bevel_angle_deg,
               "top_opening_mm": spec.top_opening_mm}
    along_key = {"root_gap_mm": "root_gap", "depth_mm": "depth", "top_opening_mm": "top_opening"}
    specimen: dict = {}
    for key, nom in nominal.items():
        if key in _BEVEL_ONLY_FEATURES and not has_bevel:
            checks[key] = (get(fm, key), None, None)
            continue
        v = get(fm, key)
        if not np.isfinite(v):
            continue
        tol = tol_of(key)
        if key in along_key:                                      # 거리: 설계 특징이 아니면 (±50 % 밖) 측정 실패 (특징 오인식)
            half = design_band(nom)
            match = design_match(v, nom)
            checks[key] = (v, (nom - half, nom + half), bool(match))
            spec_fail |= not match
        else:
            checks[key] = (v, (nom - tol, nom + tol) if np.isfinite(tol) else None, None)   # 각도: 보고만
        row = {"median": v, "design": float(nom), "tol": tol if np.isfinite(tol) else None,
               "within": (bool(abs(v - nom) <= tol) if np.isfinite(tol) else None)}
        if key in along_key:
            lo, hi = get(fm, f"{along_key[key]}_along_min_mm"), get(fm, f"{along_key[key]}_along_max_mm")
            row.update(along_min=lo, along_max=hi)
            if np.isfinite(tol) and np.isfinite(lo) and np.isfinite(hi):
                row["along_within"] = bool(lo >= nom - tol and hi <= nom + tol)
        specimen[key] = row
    hl = get(fm, "hi_lo_mm")
    tmax = _tol_or_nan((tg.get("hi_lo_mm") or {}).get("target_max"))
    if np.isfinite(hl):
        checks["hi_lo_mm"] = (hl, tmax if np.isfinite(tmax) else None, None)    # 보고만 (판 단차는 시편 성질)
        specimen["hi_lo_mm"] = {"median": hl, "target_max": tmax if np.isfinite(tmax) else None,
                                "within": bool(abs(hl) <= tmax) if np.isfinite(tmax) else None,
                                "along_min": get(fm, "hi_lo_along_min_mm"), "along_max": get(fm, "hi_lo_along_max_mm")}
    gross_k = [key for key in along_key if checks.get(key, (0, 0, None))[2] is False]
    if gross_k:
        notes.append(f"{', '.join(gross_k)} outside the design +-50 % (half the nominal - between 'the designed feature' and 'no "
                     "feature'): a misidentified feature (measurement failure)")
    out_of = [key for key, r in specimen.items() if r.get("within") is False or r.get("along_within") is False]
    if out_of:
        notes.append("specimen deviates from the design (reported, not a scan failure): " + "; ".join(
            f"{key} median {specimen[key]['median']:.3f}" + (f" along {specimen[key]['along_min']:.3f}..{specimen[key]['along_max']:.3f}"
                                                         if np.isfinite(specimen[key].get('along_min', float('nan'))) else "")
            + (f" (design {specimen[key]['design']:.2f} +- {specimen[key]['tol']:.2f})" if specimen[key].get("tol") is not None else "")
            for key in out_of))
    # 심 방향 강건 표준편차: 보고만 (시편의 실제 변화를 포함한다)
    for key in ("root_gap_robust_std_mm", "depth_robust_std_mm"):
        checks[key] = (get(fm, key), None, None)
    # ---- 측정이 설계 허용치를 구분하나 (원칙 1): 용접 경로가 쓰는 값 = 단면 창의 중앙값 — 그 불확도 √(π/2)·σ/√n (n = 단면 창의 최소
    #      유효 프로파일 수의 중앙값 — 용접 경로 단면 하나가 쓰는 수, 없으면 1) 에 k 를 곱한 값이 그 치수의 작업 허용치 이하 ----
    n_st = max(1, int(((coverage or {}).get("groove") or {}).get("median_count") or 1))
    med_se = math.sqrt(math.pi / 2.0) / math.sqrt(n_st)
    for key, nk in (("root_gap_mm", "root_gap_noise_mm"), ("depth_mm", "depth_noise_mm"), ("top_opening_mm", "top_opening_noise_mm")):
        v = get(fm, nk)
        tol = tol_of(key)
        lim = tol / (float(k) * med_se) if np.isfinite(tol) else None      # 허용되는 프로파일 간 잡음 σ
        if np.isfinite(v) and lim is not None:
            good = v <= lim
            checks[nk] = (v, lim, bool(good))
            if not good:
                notes.append(f"{nk} {v:.3f} mm: k {float(k):g} x the section-median uncertainty ({v * med_se:.3f} mm over {n_st} "
                             f"profiles) > the design tolerance {tol:.2f} mm - the measurement cannot resolve the tolerance")
            spec_fail |= not good
        else:
            checks[nk] = (v, lim, None)
    # 루트 위치 (용접 경로의 겨냥점; 사용자 2026-10-07 — 작업 qc_targets root_position_mm): 측면·판 법선 성분 각각 같은 규칙
    tol_rp = _tol_or_nan((tg.get("root_position_mm") or {}).get("tol"))
    for nk in ("root_lateral_noise_mm", "root_normal_noise_mm"):
        v = get(fm, nk)
        lim = tol_rp / (float(k) * med_se) if np.isfinite(tol_rp) else None
        if np.isfinite(v) and lim is not None:
            good = v <= lim
            checks[nk] = (v, lim, bool(good))
            if not good:
                notes.append(f"{nk} {v:.3f} mm: k {float(k):g} x the section-median uncertainty ({v * med_se:.3f} mm over {n_st} "
                             f"profiles) > the root position tolerance {tol_rp:.2f} mm - the weld path aim point is not resolved")
            spec_fail |= not good
        else:
            checks[nk] = (v, lim, None)
    # ---- 진단 전용: 센서 배치 (게이트 아님) ----
    for key in ("psi_deg", "theta_deg", "roll_deg", "lateral_m", "standoff_m"):
        checks[f"placement_{key}"] = (get(rs, key), None, None)
    checks["wall_shear_k"] = (get(fm, "wall_shear_k"), None, None)
    g, d_, b, op = (get(fm, key) for key in ("root_gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm"))
    if not has_bevel:
        b = 0.0
    h_bev = max(d_ - spec.root_face_mm, 0.0) if np.isfinite(d_) else float("nan")     # 개선면 높이 = 깊이 − 루트면 (설계)
    if all(np.isfinite(v) for v in (g, d_, b, op)):
        checks["geometry_identity_mm"] = (op - g - spec.n_bevels * h_bev * math.tan(math.radians(b)), None, None)
    if has_bevel and all(np.isfinite(v) for v in (g, d_, op)) and h_bev > 0:
        # 거리 세 개 (개구·갭·깊이) 와 설계 루트면으로부터의 베벨각 — 보이는 베벨면 일부에 맞춘 기울기보다 거리 측정이 강건하다
        # (사용자 2026-10-02)
        checks["bevel_from_distances_deg"] = (math.degrees(math.atan((op - g) / (spec.n_bevels * h_bev))), None, None)
    status = "DATA_FAIL" if data_fail else ("SPEC_FAIL" if spec_fail else "PASS")
    return Verdict(status=status, checks=checks, notes=notes, stage="fidelity", specimen=specimen)


