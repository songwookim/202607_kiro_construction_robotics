"""scan_core 데이터 타입 (ROS 비의존, SI 단위: m, s, rad; 프로파일 원본만 mm 유지).

* ScanJob   : 스캔 요청 (base_frame 기준 시점·종점 + 파라미터). 필드명은
              플랫폼 YAML (hongik_scan_ros/config/*.yaml) 의 키와 일치:
              scan_defaults.{standoff_m, speed_mps, accel_mps2, profile_fps, lead_in_m, lead_out_m,
                             pitch_deg, incidence_tilt_deg, bidirectional, table_z_m,
                             specimen_top_z_m, gravity_axis_base, vertical_direction, reference_point,
                             direction_tol_deg, check_torch_tip},
              safety.{max_laser_tilt_deg, torch_protrusion_m}, 최상위 progression_sign; 안전 한계 safe_clearance_m /
              standoff_min_m / pose_tol_m 은 scan_l0.safety 가 플랫폼 정책에서 유도해 채운다,
              auto_teach.clamp_height_m (포락선 clamp_clearance 여유); plate_top_measured_m 은 로컬라이제이션이 채우는
              센서 실측 판 상면(높이축 up·P, 원시 z 아님)으로 바닥 규칙의 specimen_top_z_m 을 올리기만 한다.
              plate_plane_measured {point_m, normal} 은 센서 실측 판 평면(3G 바닥) — 판 평면 규칙이 (start, surface_normal)
              평면과 이 평면 중 로봇에 더 가까운 쪽을 쓴다 (merge_plate_plane: 로봇 쪽으로만 채택).
              방향 규칙(R4): 1G 는 진행방향 ⟂ 중력축, 3G 는 진행방향 ∥ 중력축 (direction_tol_deg 이내) — 주어진 용접 자세
              (job 문서 job.position '1G'|'3G' → orientation flat_1G|vertical_3G, POSITIONS) 와 플랫폼 gravity_axis_base 의
              일관성 검사다 (자세를 배치에서 추론하지 않는다; 'auto' 는 auto-teach·scan 이 거부).
              모든 높이 규칙(시편 상단·transit·3G 바닥)은 플랜지가 아니라 **광학 원점**(reference_point='optical') 기준.
* Calib     : T_flange←opt (4x4, m), dt_s, x_sign, T_flange←eoat(= Rz(eoat_yaw)).
* ProfileFrame / RobotSample : 드라이버·폴러가 만들어 넘기는 원시 샘플.
* ScanResult : 재구성 결과 (§5.8/§5.9 계약).
* QCReport   : 품질 지표 + 판정.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

from .measure import HALF_TURN_DEG

from .frames import (
    inv_transform, make_transform, rot_z, transform_from_xyz_rpy, xyz_rpy_from_transform, unit,
    angle_between, is_transform,
)

ORIENTATIONS = ("flat_1G", "vertical_3G", "auto")
# 용접 자세(welding position) = 사용자 명시 입력 (2026-10-01 3차 결정): job 문서의 job.position '1G' | '3G' 가 단일 출처이고
# ScanJob.orientation (flat_1G | vertical_3G) 으로 옮겨진다. 거친 점 배치에서 추론하지 않는다 (플랫폼마다 좌표계가 다르다 — KIRO 양팔).
# 'auto' (= 미지정) 는 auto-teach / scan 이 거부한다; 방향 규칙(1G 진행 ⟂ 중력, 3G 진행 ∥ 중력) 은 주어진 자세의 일관성 검사다.
POSITIONS = {"1G": "flat_1G", "3G": "vertical_3G"}
POSITION_REFUSAL = "give position: 1G|3G"


def orientation_from_position(position) -> str:
    """'1G' | '3G' (대소문자 무시) → 'flat_1G' | 'vertical_3G'. 그 외 JobError."""
    p = str(position).strip().upper() if position is not None else ""
    if p not in POSITIONS:
        raise JobError(f"position must be one of {sorted(POSITIONS)}, got {position!r} ({POSITION_REFUSAL})")
    return POSITIONS[p]


def position_from_orientation(orientation) -> Optional[str]:
    """'flat_1G' → '1G', 'vertical_3G' → '3G', 'auto'/None → None (미지정)."""
    for k, v in POSITIONS.items():
        if orientation == v:
            return k
    return None
VERTICAL_DIRECTIONS = ("down", "up")          # 3G 진행 방향: down = 위 점 → 아래 점 (기본), up = 아래 점 → 위 점
REFERENCE_POINTS = ("optical",)               # 높이·바닥 규칙의 기준점 (광학 원점만 지원; 플랜지 기준 금지)
GRAVITY_AXIS_BASE_DEFAULT = (0.0, 0.0, -1.0)  # base 프레임에서의 중력 방향 (아래쪽)
INCIDENCE_TILT_MAX_DEG = 35.0                 # |incidence_tilt_deg| 상한. 처음 25° (광택면 유효점 >90 % 구간 −25°~0°, 설계 §5.2);
                                              # 2026-10-05 사용자: 3G 아래 끝을 높은 곳에서 보려면 ≥ 25° → 35° (거친 standoff 0.40 m 에서
                                              # 원시 거리 0.40/cos35° + 그루브 ≈ 500 mm < 교정 창 544 mm − 여유). 실측 데이터로 확인 중
SPECULAR_TILT_DEG = -12.0                     # 정반사 −θ_tri/2 ≈ −12° (설계 §5.2; 피치 규약에서 상속, P1 실측으로 확정)
SPECULAR_TILT_BAND_DEG = 3.0                  # 이 대역 안이면 경고


class IncidenceTiltWarning(UserWarning):
    """incidence_tilt_deg 가 정반사 대역(−12°±3°) 근방."""


class JobError(ValueError):
    """ScanJob 입력 검증 실패."""


def _plate_plane(v) -> Optional[dict]:
    """plate_plane_measured 입력 → {'point_m': [x,y,z], 'normal': [단위]} (None 허용). dict 또는 (point, normal) 쌍."""
    if v is None:
        return None
    if isinstance(v, dict):
        p, n = v.get("point_m", v.get("point")), v.get("normal")
    else:
        try:
            p, n = v
        except (TypeError, ValueError):
            raise JobError(f"plate_plane_measured must be {{point_m, normal}} or (point, normal), got {v!r}")
    p = _vec3(p, "plate_plane_measured.point_m")
    n = _vec3(n, "plate_plane_measured.normal")
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        raise JobError("plate_plane_measured.normal must be non-zero")
    return {"point_m": [float(x) for x in p], "normal": [float(x) for x in n / nn]}


def merge_plate_plane(prev, new) -> Optional[dict]:
    """판 평면 raise-only 결합 (I1): 이전 평면이 없으면 새 평면, 있으면 새 평면이 이전 평면보다 **로봇 쪽**
    (n_prev · (p_new − p_prev) > 0) 일 때만 새 평면. 반환 {'point_m', 'normal'} 또는 None."""
    a = _plate_plane(prev)
    b = _plate_plane(new)
    if a is None:
        return b
    if b is None:
        return a
    n_a = np.asarray(a["normal"])
    if float(n_a @ (np.asarray(b["point_m"]) - np.asarray(a["point_m"]))) > 0.0:
        n_b = np.asarray(b["normal"])
        if float(n_b @ n_a) < 0.0:                       # 같은 반구 (로봇 쪽 +) 로
            b = {"point_m": b["point_m"], "normal": [float(x) for x in -n_b]}
        return b
    return a


def _vec3(v, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise JobError(f"{name} must be a finite 3-vector, got {v!r}")
    return a


# --------------------------------------------------------------------------------------
# ScanJob
# --------------------------------------------------------------------------------------
@dataclass
class ScanJob:
    """직선 스캔 작업 정의. start/end 는 base_frame 기준 판 표면 위 점 [m]."""

    start: np.ndarray
    end: np.ndarray
    surface_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    orientation: str = "auto"              # 'flat_1G' | 'vertical_3G' | 'auto'(법선으로 판정)
    standoff_m: float = 0.32
    speed_mps: float = 0.03                # 목표 순항속도 (CB 명령값은 default_speed 보정 후 결정)
    accel_mps2: float = 0.2
    lead_in_m: float = 0.04
    lead_out_m: float = 0.04
    pitch_deg: float = -5.0                # 라인축(y_eoat) 둘레 피치, 0 ~ -8° 권장 (정반사 회피; 순수 회전 → 프로파일 중심이 s 방향으로 standoff·tan 만큼 이동)
    incidence_tilt_deg: float = 0.0        # 입사 기울기 [deg], 라인축(X_s) 둘레 회전이되 광학 Z 가 계속 심(seam) 점을 향하도록 원점이 반지름 standoff 호 위를 이동.
                                           #   3G: + = 센서가 심 점을 **내려다봄** (광학 원점이 심 점보다 standoff·sin(tilt) 위, 바닥 규칙에 안전한 방향),
                                           #       − = 올려다봄 (원점이 아래 점보다 낮아져 3G_floor 위반 가능). 기본 0; 3G 작업파일 기본 +10.
                                           #   1G: pitch_deg 와 같은 회전 방향(라인축 둘레)으로 합산되며(레이저 기울기 = |pitch + tilt|), pitch 와 달리 원점이 호 위를 이동.
                                           #   두 값을 함께 주면 합성된다. |tilt| ≤ 35°, |pitch|+|tilt| ≤ max_laser_tilt_deg, −12°±3° 는 정반사 경고.
    profile_fps: float = 100.0
    progression_sign: int = 1              # x_eoat = ±d̂ (센서 앞장 +1 / 토치 앞장 -1)
    safe_clearance_m: float = 0.10         # 한계 L = 정책 여유 + 정지 거리 (scan_l0.safety 가 플랫폼 값으로 채운다): 모든 툴 점 ↔
                                           # 판·지지면·테이블, transit 높이, 판 법선 접근 거리
    table_z_m: Optional[float] = None      # base 기준 테이블 상판 z (있으면 하한 검사)
    specimen_top_z_m: Optional[float] = None   # base 기준 시편 최상단 z (1G: 없으면 start/end z 의 최댓값)
    bidirectional: bool = False
    max_laser_tilt_deg: float = 30.0       # 광학 Z 축 vs (-n) / (1G: base -Z) 허용각
    torch_protrusion_m: float = 0.168      # 토치 팁 키포인트의 광학 원점 너머 +z_eoat 거리 (팁 z_eoat = p_opt z + 이 값). 기본값 (와 아래
                                           # torch_lateral_m) = 예전 곧은 자리표시자, CAD 창면 원점 (0.147, Calib.lab_initial) 기준 — 단위 시험의
                                           # 장면이 이 값에 맞춰져 있다. 실제 작업 템플릿·플랫폼 safety 는 우리 토치의 CAD 와이어 끝 (torch_tcp +
                                           # 20 mm; eoat_lab_01 광학 원점 기준 0.1924, 측면 (−0.0366, −0.0005)) — test_torch_model_consistency
    torch_lateral_m: tuple = (-0.100, 0.0) # 토치 팁의 (x, y)_eoat (기본 = 예전 자리표시자; 작업 파일이 우리 토치 값을 준다)
    standoff_min_m: float = 0.245          # 광학 원점 ↔ 판 최소 거리 (scan_l0.safety: 센서 유효창 하한 + 정지 거리)
    pose_tol_m: float = 0.0                # 도착 자세 허용 (툴 점; scan_l0.safety): 한계에 붙여 계획하는 목표는 한계 + 이 값에 둔다
    mode: str = "standard"                 # precise | standard | fast | prescan
    job_id: str = ""
    base_frame: str = "link0"
    # ---- R4: 중력축·방향 규칙 (기준점 = 광학 원점) ----
    gravity_axis_base: np.ndarray = field(default_factory=lambda: np.array(GRAVITY_AXIS_BASE_DEFAULT))
    vertical_direction: str = "down"       # 3G: 'down'(위→아래, 리드인은 위 점 위쪽에만, 리드아웃 없음) | 'up'(아래→위, 리드인 없음)
    reference_point: str = "optical"       # 높이·바닥·transit 규칙 기준점: 'optical' = profiler_optical_frame 원점
    direction_tol_deg: float = 20.0        # 1G: u ⟂ g, 3G: u ∥ g 허용 편차 — 자세 입력 확인용 (1G/3G 는 90° 차이; 2026-10-04 5 → 20:
    #                                        시편은 대체로 수평/수직일 뿐이고 대략 점도 부정확해 3G 4.5° 기울기가 5.7–8° 로 거부됐다)
    check_torch_tip: bool = True           # 토치 팁을 추가 키포인트로 검사 (시편 상단·transit 규칙, 별도 메시지)
    # ---- 클램프 여유·센서 실측 판 상면 (설계 패널 2026-09-30 revision #5, 포락선 규칙 clamp_clearance) ----
    clamp_height_m: float = 0.0            # 1G: 판 위(in-plane) 구간에서 모든 툴 키포인트 높이 ≥ plate_top + clamp_height
                                           # 3G: 모든 키포인트의 판 평면 거리 ≥ clamp_height. 0 = 추가 여유 없음 (≥ 0)
    plate_top_measured_m: Optional[float] = None
                                           # 센서로 실측한 판 상면 높이 [m] (높이축 = up·P, base 3D 점에서 계산 — 원시 z 값 아님).
                                           # 바닥 규칙(1G_above_specimen_top, never_below_table, clamp_clearance)에서
                                           # specimen_top_z_m 을 덮어쓰거나 올린다: plate_top = max(specimen_top_z_m, plate_top_measured_m).
                                           # 로컬라이제이션 코드가 첫 측정 후 채우고 이후에는 올리기만 한다(내리지 않음).
    plate_plane_measured: Optional[dict] = None
                                           # 센서로 실측한 판 평면 {point_m: [3], normal: [3]} (base, normal = 로봇 쪽 +; 3G 바닥).
                                           # 있으면 path_gen.check_envelope 의 판 평면 규칙(never_cross_plate_plane, 3G clamp_clearance)
                                           # 이 (start, surface_normal) 평면과 이 평면 중 **더 가까운** 거리를 쓴다 (보수적; 시점/종점이
                                           # 루트 점이면 그 평면은 판 상면보다 그루브 깊이만큼 뒤에 있다). 채택은 merge_plate_plane
                                           # (로봇 쪽으로만, raise-only). (point, normal) 쌍도 받아 dict 로 정규화한다.
    floor_z_m: Optional[float] = None
                                           # 3G 바닥 높이 [m] (up·P): 있으면 start/end 중 낮은 점 대신 이 값. auto-teach 가 채운다 — 첫 패스
                                           # 전에는 대략 아래 점, 첫 패스 (아래 → 위) 가 시편 아래 끝을 잰 뒤에는 그 끝 (사용자 2026-10-05:
                                           # 대략 점은 교시마다 달라 임의적이다 — 대략 점은 처음 보정까지만 기준).

    def __post_init__(self):
        self.start = _vec3(self.start, "start")
        self.end = _vec3(self.end, "end")
        self.surface_normal = _vec3(self.surface_normal, "surface_normal")
        self.gravity_axis_base = _vec3(self.gravity_axis_base, "gravity_axis_base")
        self.progression_sign = int(self.progression_sign)
        self.check_torch_tip = bool(self.check_torch_tip)
        self.plate_plane_measured = _plate_plane(self.plate_plane_measured)

    # ---- 유도량 ----
    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def unit_dir(self) -> np.ndarray:
        return unit(self.end - self.start)

    @property
    def normal_unit(self) -> np.ndarray:
        return unit(self.surface_normal)

    @property
    def gravity_unit(self) -> np.ndarray:
        """base 프레임 중력 단위벡터 (아래쪽)."""
        return unit(self.gravity_axis_base)

    @property
    def up_unit(self) -> np.ndarray:
        """위쪽 단위벡터 = −g."""
        return -self.gravity_unit

    def height(self, p) -> np.ndarray:
        """점(들)의 높이 = up·p (기본 중력축이면 base z 와 같다). (…,3) → (…)."""
        return np.asarray(p, dtype=float) @ self.up_unit

    def direction_angles_deg(self) -> dict:
        """진행방향 u 와 중력축의 관계: {'u_vs_gravity_deg', 'horizontal_dev_deg', 'vertical_dev_deg'}."""
        u = self.unit_dir
        g = self.gravity_unit
        ang = math.degrees(angle_between(u, g))            # 0(아래로) … 180(위로)
        return {"u_vs_gravity_deg": ang, "horizontal_dev_deg": abs(90.0 - ang),
                "vertical_dev_deg": min(ang, 180.0 - ang)}

    @property
    def position(self) -> Optional[str]:
        """용접 자세 '1G' | '3G' (orientation 이 flat_1G / vertical_3G 일 때), 'auto' 면 None (= 미지정; auto-teach·scan 거부)."""
        return position_from_orientation(self.orientation)

    def resolved_orientation(self) -> str:
        """'auto' 이면 법선과 중력축으로 판정: |n·g| ≥ cos45° → flat_1G, 아니면 vertical_3G."""
        if self.orientation in ("flat_1G", "vertical_3G"):
            return self.orientation
        if self.orientation != "auto":
            raise JobError(f"orientation must be one of {ORIENTATIONS}, got {self.orientation!r}")
        n = self.normal_unit
        return "flat_1G" if abs(float(np.dot(n, self.gravity_unit))) >= math.cos(math.radians(45.0)) else "vertical_3G"

    def upper_point_index(self) -> int:
        """start(0)/end(1) 중 더 높은 점 (같으면 0)."""
        return 0 if float(self.height(self.start)) >= float(self.height(self.end)) else 1

    def plate_top_z_m(self) -> Optional[float]:
        """바닥 규칙에 쓰는 '올려진' 판 상면 높이 = max(specimen_top_z_m, plate_top_measured_m); 둘 다 없으면 None.

        실측값은 입력값을 올리기만 한다(둘 중 큰 값). 높이축은 up·P (gravity_axis_base 기준) 이며 센서 원시 z 가 아니다.
        """
        vals = [float(v) for v in (self.specimen_top_z_m, self.plate_top_measured_m) if v is not None]
        return max(vals) if vals else None

    def plate_plane(self) -> Optional[tuple]:
        """실측 판 평면 (point (3,), 단위 normal (3,)) 또는 None."""
        pp = self.plate_plane_measured
        if pp is None:
            return None
        return np.asarray(pp["point_m"], dtype=float), np.asarray(pp["normal"], dtype=float)

    def validate(self, min_length_m: float = 0.05, max_perp_err_deg: float = 5.0) -> "ScanJob":
        """§5.1 입력 검증 + R4 방향 규칙. 실패 시 JobError."""
        L = self.length_m
        if L < min_length_m:
            raise JobError(f"|end-start| = {L:.4f} m < {min_length_m} m")
        n = self.normal_unit
        u = self.unit_dir
        perp = math.degrees(abs(math.asin(max(-1.0, min(1.0, float(np.dot(u, n)))))))
        if perp > max_perp_err_deg:
            raise JobError(f"travel direction is not perpendicular to surface_normal: {perp:.2f} deg > {max_perp_err_deg}")
        if self.vertical_direction not in VERTICAL_DIRECTIONS:
            raise JobError(f"vertical_direction must be one of {VERTICAL_DIRECTIONS}, got {self.vertical_direction!r}")
        if self.reference_point not in REFERENCE_POINTS:
            raise JobError(f"reference_point must be one of {REFERENCE_POINTS} (profiler optical origin), "
                           f"got {self.reference_point!r}")
        if not (isinstance(self.direction_tol_deg, (int, float)) and math.isfinite(self.direction_tol_deg)
                and 0.0 <= self.direction_tol_deg < 45.0):
            raise JobError(f"direction_tol_deg must be in [0, 45), got {self.direction_tol_deg!r}")
        # R4 방향 규칙: 1G 는 수평 경로(u ⟂ g), 3G 는 수직 경로(u ∥ g)
        orientation = self.resolved_orientation()
        da = self.direction_angles_deg()
        # 일관성 검사: 주어진 자세(position) vs 플랫폼 중력축 (gravity_axis_base) — 메시지에 둘 다 쓴다
        given = (f"given position {position_from_orientation(self.orientation)} ({self.orientation})"
                 if self.orientation in ("flat_1G", "vertical_3G") else f"orientation {self.orientation!r} resolved to {orientation}")
        g_txt = f"platform gravity_axis_base {[round(float(v), 4) for v in self.gravity_axis_base]}"
        if orientation == "flat_1G" and da["horizontal_dev_deg"] > self.direction_tol_deg:
            raise JobError(
                f"{given} is inconsistent with the start/end layout: 1G requires horizontal path (travel direction "
                f"perpendicular to the {g_txt}): deviation {da['horizontal_dev_deg']:.2f} deg > direction_tol_deg="
                f"{self.direction_tol_deg} - check the welding position input and the platform gravity axis")
        if orientation == "vertical_3G" and da["vertical_dev_deg"] > self.direction_tol_deg:
            raise JobError(
                f"{given} is inconsistent with the start/end layout: 3G requires vertical path (travel direction "
                f"parallel to the {g_txt}): deviation {da['vertical_dev_deg']:.2f} deg > direction_tol_deg="
                f"{self.direction_tol_deg} - check the welding position input and the platform gravity axis")
        if self.standoff_m < self.torch_protrusion_m + self.safe_clearance_m:
            raise JobError(
                f"standoff_m={self.standoff_m} < torch_protrusion_m + safe_clearance_m = "
                f"{self.torch_protrusion_m + self.safe_clearance_m:.3f} (torch would come too close to the plate)")
        if self.standoff_m < self.standoff_min_m:
            raise JobError(f"standoff_m={self.standoff_m} < standoff_min_m={self.standoff_min_m}")
        for name in ("speed_mps", "accel_mps2", "profile_fps", "lead_in_m", "lead_out_m", "safe_clearance_m"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise JobError(f"{name} must be > 0, got {v!r}")
        if self.progression_sign not in (1, -1):
            raise JobError(f"progression_sign must be +1 or -1, got {self.progression_sign}")
        ch = self.clamp_height_m
        if not (isinstance(ch, (int, float)) and not isinstance(ch, bool) and math.isfinite(ch) and ch >= 0.0):
            raise JobError(f"clamp_height_m must be a finite number >= 0, got {ch!r}")
        pt = self.plate_top_measured_m
        if pt is not None and not (isinstance(pt, (int, float)) and not isinstance(pt, bool) and math.isfinite(pt)):
            raise JobError(f"plate_top_measured_m must be None or a finite height [m], got {pt!r}")
        le = self.floor_z_m
        if le is not None and not (isinstance(le, (int, float)) and not isinstance(le, bool) and math.isfinite(le)):
            raise JobError(f"floor_z_m must be None or a finite height [m], got {le!r}")
        if abs(self.pitch_deg) > self.max_laser_tilt_deg:
            raise JobError(f"|pitch_deg|={abs(self.pitch_deg)} > max_laser_tilt_deg={self.max_laser_tilt_deg}")
        tilt = self.incidence_tilt_deg
        if not (isinstance(tilt, (int, float)) and math.isfinite(tilt)):
            raise JobError(f"incidence_tilt_deg must be a finite number, got {tilt!r}")
        if abs(tilt) > INCIDENCE_TILT_MAX_DEG:
            raise JobError(f"|incidence_tilt_deg|={abs(tilt)} > {INCIDENCE_TILT_MAX_DEG} deg")
        if abs(self.pitch_deg) + abs(tilt) > self.max_laser_tilt_deg:
            raise JobError(f"|pitch_deg|+|incidence_tilt_deg|={abs(self.pitch_deg) + abs(tilt)} > "
                           f"max_laser_tilt_deg={self.max_laser_tilt_deg} (the two rotations compose about the laser-line axis)")
        if tilt != 0.0 and abs(tilt - SPECULAR_TILT_DEG) < SPECULAR_TILT_BAND_DEG:
            warnings.warn(f"incidence_tilt_deg={tilt} is within {SPECULAR_TILT_BAND_DEG} deg of the specular direction "
                          f"-theta_tri/2 = {SPECULAR_TILT_DEG} deg (design 5.2): expect saturated/lost points",
                          IncidenceTiltWarning, stacklevel=2)
        if orientation == "vertical_3G":
            if abs(float(np.dot(n, self.gravity_unit))) > math.sin(math.radians(HALF_TURN_DEG)):   # 원칙 2 중간점
                raise JobError("vertical_3G requested but surface_normal is not roughly horizontal (perpendicular to gravity)")
        return self

    def lead_distance_required_m(self, settle_s: float = 0.5) -> float:
        """lead ≥ v²/(2a) + v·settle (설계 §5.2)."""
        return self.speed_mps ** 2 / (2.0 * self.accel_mps2) + self.speed_mps * settle_s

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("start", "end", "surface_normal", "gravity_axis_base"):
            d[k] = [float(v) for v in getattr(self, k)]
        d["torch_lateral_m"] = [float(v) for v in self.torch_lateral_m]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScanJob":
        d = dict(d)
        if "torch_lateral_m" in d:
            d["torch_lateral_m"] = tuple(d["torch_lateral_m"])
        return cls(**d)


# --------------------------------------------------------------------------------------
# Calib
# --------------------------------------------------------------------------------------
# 설계 §2.4 초기값 (연구실 EOAT CAD): eoat_base_link → laser_link, laser_link → profiler_optical_frame
LASER_JOINT_XYZ = (0.057294, 0.0, 0.023)
LASER_JOINT_RPY = (math.pi, 0.0, math.pi)              # R = diag(-1, 1, -1)
OPT_JOINT_XYZ_INITIAL = (0.045, 0.0, -0.124)           # 창면 추정 (12, 0, 147) mm 의 역변환
OPT_JOINT_RPY_INITIAL = (math.pi, 0.0, math.pi / 2)    # +π/2 → (X,Y,Z)_opt = (+y, -x, +z)_eoat (연구실 +X 부호 실측 확정 2026-10-02;
                                                        # 이전 초기값 −π/2 → (-y, +x, +z) 는 광축 둘레 180° 반대)


def g1_block_fields(block) -> dict:
    """캘리브 파일의 'g1:' 블록 {dt_s, dt_sigma_s, he_sigma_m, laser_pitch_sigma_rad, edge_response, noise_priors} (1G core 전용) → Calib
    선택 필드 (없는 키는 None)."""
    b = dict(block or {})

    def f(key):
        v = b.get(key)
        return None if v is None else float(v)
    er = b.get("edge_response")
    npr = b.get("noise_priors")
    return {"g1_dt_s": f("dt_s"), "dt_sigma_s": f("dt_sigma_s"), "he_sigma_m": f("he_sigma_m"),
            "laser_pitch_sigma_rad": f("laser_pitch_sigma_rad"), "edge_response": dict(er) if isinstance(er, dict) else None,
            "noise_priors": dict(npr) if isinstance(npr, dict) else None}


@dataclass
class Calib:
    """센서 캘리브레이션.

    T_flange_opt : T_flange←opt (4x4, m). 점 변환 P_base = T_base←flange(t)·T_flange←opt·[x,0,z,1]ᵀ 에만 사용.
    T_flange_eoat: T_flange←eoat = (xyz 0, rpy 0 0 eoat_yaw). 툴 목표 자세는 eoat 프레임으로 구성.
    dt_s         : 시간오프셋 (stamp = a_lo + T_est·k − dt).  x_sign: 센서 +X 부호 실측값(±1).
    """

    T_flange_opt: np.ndarray
    dt_s: float = 0.0
    x_sign: int = 1
    eoat_yaw_rad: float = 0.0
    T_flange_eoat: Optional[np.ndarray] = None
    parent_frame: str = "laser_link"
    source: str = ""
    date: str = ""
    residual_rms_m: float = float("nan")
    sensor_param_hash: str = ""
    # 1G core 전용 (G1_CORE_SPEC §7 #27/#28; 캘리브 파일의 'g1:' 블록 — 사용자 승인 2026-10-09, 새 1G 경로 전용): 시간 오프셋 g1_dt_s [s]
    # (None = 위 dt_s), 그 σ [s], 핸드아이 자세 부류 오프셋의 σ [m] (계획 오버스캔). 없으면 None: 1G core 는 이동 전에 거부한다. 3G·legacy
    # 경로는 읽지 않는다 (그 경로의 시간 오프셋은 dt_s 그대로)
    g1_dt_s: Optional[float] = None
    dt_sigma_s: Optional[float] = None
    he_sigma_m: Optional[float] = None
    # 레이저 평면의 레이저 선 축 둘레 회전 (센서 피치) 캘리브 σ [rad] — 한 공구 자세로 잰 끝면 기울기는 이것과 구별되지 않는다 (끝면 모형이
    # 기울기 σ 에 더한다). 없으면 None: 기울기의 캘리브 몫을 모른다고 보고한다
    laser_pitch_sigma_rad: Optional[float] = None
    # 센서의 깊이 계단 응답 (레이저 띠 퍼짐 σ_L, 삼각측량 각 θ, 그 σ) — 1G 끝 모형 (scan_core.end_faces) 의 캘리브 상수. 없으면 None:
    # 끝 모형이 스캔에서 온라인으로 추정한다
    edge_response: Optional[dict] = None
    # 1G 추정기 분산 성분의 보정 사전 (끝 판독의 크로싱·판독 산포, 구간 오프셋 산포 — 로봇 + 센서 모집단, 속도 모형 a + b·v², ν0;
    # scan_core.noise_priors). 없으면 None: 추정기가 실행 자신의 자료만으로 (작은 표본) 다룬다
    noise_priors: Optional[dict] = None
    # 툴 몸체 볼록 껍질 점 (eoat 프레임, (K,3) m; scan_core.tool_body) — 테이블 규칙·3G 입사 선택이 몸체의 가장 낮은 점을 본다.
    # 캘리브 값이 아니라 EOAT 형상이다 (설정 safety.tool_body_mesh 에서 붙인다; 파일·to_dict 에는 개수와 출처만)
    body_points_eoat: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    body_source: str = ""
    # The weld torch from the platform's robot model (MoveIt adapter tool_geometry, when the weld torch is on this arm): its torch-axis
    # frame at the TCP in the eoat frame (4x4: origin = the TCP = the contact-tip end, z = the torch axis toward the work; TF flange <-
    # TCP . the tip axes) and its gas nozzle end along +z [mm] (this robot's URDF collision geometry). With them the envelope's torch
    # keypoint = the wire end = TCP + (nozzle end + the job's process stick-out) x z (AutoTeach._apply_safety_policy) instead of the
    # job's torch_lateral_m and the config's safety.torch_protrusion_m. EOAT geometry, not a calibration value
    torch_tcp_eoat: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    torch_nozzle_end_mm: Optional[float] = field(default=None, compare=False, repr=False)
    torch_tip_source: str = ""

    def __post_init__(self):
        self.T_flange_opt = np.asarray(self.T_flange_opt, dtype=float).reshape(4, 4)
        if not is_transform(self.T_flange_opt, tol=1e-5):
            raise ValueError("T_flange_opt is not a rigid transform")
        if self.T_flange_eoat is None:
            self.T_flange_eoat = make_transform(rot_z(float(self.eoat_yaw_rad)), (0.0, 0.0, 0.0))
        else:
            self.T_flange_eoat = np.asarray(self.T_flange_eoat, dtype=float).reshape(4, 4)
        self.x_sign = int(self.x_sign)

    @property
    def T_eoat_opt(self) -> np.ndarray:
        """T_eoat←opt = T_flange←eoat⁻¹ · T_flange←opt."""
        return inv_transform(self.T_flange_eoat) @ self.T_flange_opt

    @property
    def p_opt_eoat(self) -> np.ndarray:
        """광학 원점의 eoat 좌표 (초기 ≈ (0.012, 0, 0.147) m)."""
        return self.T_eoat_opt[:3, 3].copy()

    @classmethod
    def from_laser_link(cls, opt_xyz=OPT_JOINT_XYZ_INITIAL, opt_rpy=OPT_JOINT_RPY_INITIAL,
                        eoat_yaw_rad: float = 0.0, laser_xyz=LASER_JOINT_XYZ, laser_rpy=LASER_JOINT_RPY,
                        **kw) -> "Calib":
        """캘리브 YAML 저장 규약(laser_link→profiler_optical_frame, eoat_yaw 하류)에서 T_flange←opt 유도."""
        T_fe = make_transform(rot_z(float(eoat_yaw_rad)))
        T_el = transform_from_xyz_rpy(laser_xyz, laser_rpy)
        T_lo = transform_from_xyz_rpy(opt_xyz, opt_rpy)
        return cls(T_flange_opt=T_fe @ T_el @ T_lo, eoat_yaw_rad=float(eoat_yaw_rad), T_flange_eoat=T_fe,
                   parent_frame="laser_link", **kw)

    @classmethod
    def lab_initial(cls, eoat_yaw_rad: float = -math.pi / 2, opt_yaw_branch: int = +1, **kw) -> "Calib":
        """연구실 초기 추정값. opt_yaw_branch=+1 → rpy(π,0,+π/2) (2026-10-02 실측 확정), −1 → rpy(π,0,−π/2) (이전 초기값)."""
        rpy = (math.pi, 0.0, float(opt_yaw_branch) * math.pi / 2)
        return cls.from_laser_link(opt_rpy=rpy, eoat_yaw_rad=eoat_yaw_rad, source="lab_initial_cad_estimate", **kw)

    def to_dict(self) -> dict:
        xyz, rpy = xyz_rpy_from_transform(self.T_flange_opt)
        return {
            "T_flange_opt": self.T_flange_opt.tolist(),
            "flange_opt_xyz_m": xyz.tolist(),
            "flange_opt_rpy_rad": rpy.tolist(),
            "T_flange_eoat": self.T_flange_eoat.tolist(),
            "eoat_yaw_rad": float(self.eoat_yaw_rad),
            "dt_s": float(self.dt_s),
            "x_sign": int(self.x_sign),
            "parent_frame": self.parent_frame,
            "source": self.source,
            "date": self.date,
            "residual_rms_m": float(self.residual_rms_m),
            "sensor_param_hash": self.sensor_param_hash,
            "g1": {"dt_s": None if self.g1_dt_s is None else float(self.g1_dt_s),
                   "dt_sigma_s": None if self.dt_sigma_s is None else float(self.dt_sigma_s),
                   "he_sigma_m": None if self.he_sigma_m is None else float(self.he_sigma_m),
                   "laser_pitch_sigma_rad": None if self.laser_pitch_sigma_rad is None else float(self.laser_pitch_sigma_rad),
                   "edge_response": None if self.edge_response is None else dict(self.edge_response),
                   "noise_priors": None if self.noise_priors is None else dict(self.noise_priors)},
            "body_points_n": 0 if self.body_points_eoat is None else int(np.asarray(self.body_points_eoat).shape[0]),
            "body_source": self.body_source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calib":
        if "T_flange_opt" in d:
            T = np.asarray(d["T_flange_opt"], dtype=float)
        else:
            T = transform_from_xyz_rpy(d["flange_opt_xyz_m"], d["flange_opt_rpy_rad"])
        T_fe = np.asarray(d["T_flange_eoat"], dtype=float) if d.get("T_flange_eoat") is not None else None
        return cls(T_flange_opt=T, dt_s=float(d.get("dt_s", 0.0)), x_sign=int(d.get("x_sign", 1)),
                   eoat_yaw_rad=float(d.get("eoat_yaw_rad", 0.0)), T_flange_eoat=T_fe,
                   parent_frame=str(d.get("parent_frame", "laser_link")), source=str(d.get("source", "")),
                   date=str(d.get("date", "")), residual_rms_m=float(d.get("residual_rms_m", float("nan"))),
                   sensor_param_hash=str(d.get("sensor_param_hash", "")),
                   **g1_block_fields(d.get("g1")))

    def save_yaml(self, path) -> None:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path) -> "Calib":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


# --------------------------------------------------------------------------------------
# 원시 샘플
# --------------------------------------------------------------------------------------
@dataclass
class ProfileFrame:
    """센서 RESULT 메시지 1개 (DataMode 4). x_mm/z_mm 길이 = 서브헤더 NumberOfPoints."""

    x_mm: np.ndarray
    z_mm: np.ndarray
    ts_ms: int                 # 센서 u64 ms since boot (무결성 검사 전용)
    result_counter: int        # sensor_result_counter (시간모델의 k)
    t_rx_s: float              # 호스트 수신시각 [s] (monotonic)
    global_counter: Optional[int] = None
    line_counter: Optional[int] = None

    def __post_init__(self):
        self.x_mm = np.asarray(self.x_mm, dtype=np.float32).reshape(-1)
        self.z_mm = np.asarray(self.z_mm, dtype=np.float32).reshape(-1)
        if self.x_mm.shape != self.z_mm.shape:
            raise ValueError("x_mm and z_mm must have the same length")

    @property
    def n_points(self) -> int:
        return int(self.x_mm.shape[0])


@dataclass
class RobotSample:
    """로봇 상태 샘플 (change-detection 통과분만). T_base_flange 는 jnt_ang FK (m)."""

    t_rx_s: float
    T_base_flange: np.ndarray
    jnt_ang_rad: Optional[np.ndarray] = None

    def __post_init__(self):
        self.T_base_flange = np.asarray(self.T_base_flange, dtype=float).reshape(4, 4)


# --------------------------------------------------------------------------------------
# 결과
# --------------------------------------------------------------------------------------
@dataclass
class QCReport:
    metrics: dict = field(default_factory=dict)
    verdict: str = "UNJUDGED"         # PASS | RETRY | EXPOSURE_ADJUST | FAIL | UNJUDGED
    failures: list = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"metrics": _jsonable(self.metrics), "verdict": self.verdict, "failures": list(self.failures),
                "thresholds": _jsonable(self.thresholds), "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, d: dict) -> "QCReport":
        return cls(metrics=dict(d.get("metrics", {})), verdict=str(d.get("verdict", "UNJUDGED")),
                   failures=list(d.get("failures", [])), thresholds=dict(d.get("thresholds", {})),
                   notes=list(d.get("notes", [])))


@dataclass
class ScanResult:
    """재구성 결과 (§5.9 계약). 배열 첫 축 = 프로파일 N, 둘째 축 = n_pts (NumberOfPoints)."""

    x_mm: np.ndarray               # [N, n_pts] float32, 무효값 원본 유지
    z_mm: np.ndarray               # [N, n_pts]
    valid: np.ndarray              # [N, n_pts] bool
    result_counter: np.ndarray     # [N] int64
    ts_ms: np.ndarray              # [N] int64
    t_rx_s: np.ndarray             # [N] float64
    t_acq_s: np.ndarray            # [N] float64 (= a_lo + T_est·k − dt)
    T_base_flange: np.ndarray      # [N,4,4]
    T_base_opt: np.ndarray         # [N,4,4]
    in_cruise: np.ndarray          # [N] bool (실측 등속 & 무결성)
    unsynced: np.ndarray           # [N] bool (포즈 보간 범위 밖)
    ts_flag: np.ndarray            # [N] bool (ts 무결성 위반)
    speed_mps: np.ndarray          # [N] 실측 TCP 속도 (플랜지 위치 차분)
    s_m: np.ndarray                # [N] 진행좌표 (u·(opt_origin − start)); job 없으면 NaN
    base_frame: str = "link0"
    T_base_scan: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_base_plate: np.ndarray = field(default_factory=lambda: np.eye(4))
    ds_mm: float = float("nan")
    dx_mm: float = float("nan")
    calib: Optional[Calib] = None
    qc: Optional[QCReport] = None
    meta: dict = field(default_factory=dict)

    @property
    def n_profiles(self) -> int:
        return int(self.x_mm.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.x_mm.shape[1]) if self.x_mm.ndim == 2 else 0

    def opt_origins(self) -> np.ndarray:
        return self.T_base_opt[:, :3, 3]

    def array_fields(self) -> dict:
        return {k: getattr(self, k) for k in (
            "x_mm", "z_mm", "valid", "result_counter", "ts_ms", "t_rx_s", "t_acq_s", "T_base_flange",
            "T_base_opt", "in_cruise", "unsynced", "ts_flag", "speed_mps", "s_m", "T_base_scan", "T_base_plate")}


def _jsonable(o: Any):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o
