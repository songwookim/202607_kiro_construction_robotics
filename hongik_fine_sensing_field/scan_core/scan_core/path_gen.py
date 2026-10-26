"""경로 생성 (설계 §5.3) + 안전 포락선 (사용자 결정: 충돌계획 대신 '시편 위쪽'·'3G 바닥' 규칙, R4).

프레임 구성 규칙 (C10, §2.4/§6.2 #4):
  u   = (end − start)/L (진행 단위벡터, s 좌표축),  n = 판 법선(로봇 쪽을 향함)
  travel_sign = 실제 이동 방향 부호 (+1: start→end, −1: end→start; 3G 에서 vertical_direction 이 정하는 방향)
  eoat 목표 자세:  x_e = R_tilt·(progression_sign·travel_sign·u),  z_e = R_tilt·(−n),  y_e = z_e × x_e,
                  R_base←e = [x_e y_e z_e]·Ry(pitch)
                  (피치 = 레이저 라인축(y_e) 둘레; progression_sign 은 '실제 이동 방향' 기준 센서 앞장(+1)/토치 앞장(−1))
  입사 기울기:     R_tilt = 라인축 X_s = n×u_travel 둘레 회전 φ (job.incidence_tilt_deg = τ):
                  3G: φ = τ·sign(up·u_travel) → z_e = −(n·cosτ + up·sinτ), 즉 τ>0 이면 센서가 심 점을 **내려다보고**
                      광학 원점은 심 점보다 standoff·sinτ 위 (바닥 규칙에 안전한 방향), τ<0 이면 올려다봄(원점이 아래로).
                  1G: φ = −progression_sign·τ → pitch 와 같은 회전 방향(라인축 둘레)이라 레이저 기울기는 |pitch+τ| 로 합산.
                  pitch 는 원점 고정 순수 회전(프로파일 중심이 s 방향으로 이동), tilt 는 심 점을 중심으로 한 반지름 standoff 호 이동
                  (Z_s 가 계속 심 점을 향함). 둘을 함께 주면 합성된다.
  광학 원점 위치:  o_opt(s) = start + u·s + d_opt,  d_opt = standoff·R_tilt·n  (τ=0 이면 n·standoff)   (s ∈ [s_scan_from, s_scan_to])
  진행좌표:        s_of(p) = u·(p − start − d_opt)  (광학 원점 → 그 원점이 바라보는 심 점의 s; τ=0 이면 u·(p−start))
  T_base←eoat(s) = [R_e | o_opt(s) − R_e·p_opt],  p_opt = T_eoat←opt[0:3,3]  (광학 프레임은 원점 위치만 사용)
  T_base←flange(s) = T_base←eoat(s) · T_flange←eoat⁻¹
  → 진행방향을 +Y_opt 로 하드코딩하지 않으며 profiler_optical_joint 의 ±π/2 분기와 무관하게 툴 목표가 같다.

기준점 (R4): 모든 높이 규칙은 **광학 원점**(profiler_optical_frame 원점, reference_point='optical') 을 기준으로 하고,
  토치 팁은 job.check_torch_tip 일 때 추가 키포인트로 별도 메시지와 함께 검사한다. 높이 = up·p, up = −gravity_axis_base
  (기본 [0,0,−1] 이면 높이 = base z). table_z_m / specimen_top_z_m 도 같은 높이축 값이다.

방향·리드 규칙 (R4):
  1G (시편 눕힘): u ⟂ g (direction_tol_deg), 리드인/리드아웃 양쪽, −up 하강 접근 / +up 상승 이탈.
  3G (시편 세움): u ∥ g. 광학 원점과 토치 팁 (1G 시편 상단 규칙과 같은 키포인트) 은 어떤 순간에도 floor 아래로 내려가지 않는다 — floor = 실측 시편
      아래 끝 (auto-teach 가 첫 패스 뒤 job.floor_z_m 에 넣는다; 그 전에는 대략 아래 점) 또는 start/end 중 낮은 점의 높이.
      vertical_direction='down'(기본): 위 점 → 아래 점, 리드인은 위 점 위쪽에만, 리드아웃 없음 (아래 점에서 정확히 정지;
                                       감속 구간 프레임은 valid 이지만 in_cruise=False),
                                       이탈 = 판 법선(수평 성분) 방향으로 후퇴 후 상승.
      vertical_direction='up'        : 아래 점 → 위 점, 리드인 없음(아래 점에서 출발), 리드아웃은 위 점 위쪽.
      접근/후퇴 벡터 = safe_clearance·unit(n − (n·up)up) : 판 법선의 수평 성분 → 높이 변화 없음(바닥 규칙 보장).
  safe_clearance_m = 한계 L (scan_l0.safety: 플랫폼 정책 여유 + 정지 거리) — 아래 모든 '여유' 는 이 값 하나다. 한계에 붙여 계획하는
      목표 (바닥으로 줄인 끝, transit 들어올림) 는 job.pose_tol_m (플랫폼 도착 허용) 만큼 더 띄운다.
  transit (1G/3G 공통): 광학 원점 높이 ≥ max(specimen_top, 위 점 높이) + safe_clearance (1G 는 + standoff 도 더함:
      스캔 높이 위에서 이동), 토치 팁(플래그) ≥ max(specimen_top, 위 점 높이) + safe_clearance.

안전 포락선 (EnvelopeError, 위반 규칙명을 메시지에 포함):
  never_below_table        모든 키포인트 + 툴 몸체 (calib.body_points_eoat: EOAT 메시 껍질) 높이 ≥ table_z + safe_clearance
                           (1G: 실측 판 상면 plate_top_measured_m 이 있으면 그 값으로도 올림)
  never_cross_plate_plane  광학 원점 판거리 ≥ standoff_min, 토치 팁·툴 몸체 판거리 ≥ safe_clearance
  1G_above_specimen_top    1G: 광학 원점 (및 토치 팁, 플래그) 높이 ≥ plate_top = max(specimen_top_z_m, plate_top_measured_m)
  3G_floor                 3G: 광학 원점 (및 토치 팁, 플래그) 높이 ≥ floor (home 왕복 구간 제외, 그 외 모든
                           구간·보간 샘플). floor = job.floor_z_m (있으면) 또는 start/end 중 낮은 점
  transit_above            transit 구간: 광학 원점 ≥ z_safe_opt, 토치 팁(플래그) ≥ z_safe
  laser_tilt               스캔 구간: 광학 Z vs −n (1G: vs g 도) ≤ max_laser_tilt_deg
  clamp_clearance          1G: 판 위(in-plane) 구간에서 모든 키포인트 높이 ≥ plate_top + job.clamp_height_m
                               (in-plane = phase scan/approach_normal/retreat_normal 의 전 샘플 + 그 외 구간('start'·'ascend' 제외)의
                               '작업 높이' 샘플(광학 원점 높이 ≤ plate_top + standoff + 10 mm) → 클램프 대역으로 내려가는 descend 는
                               실행 전에 거부되고, 대역을 벗어나는 ascend 는 항상 허용)
                           3G: 모든 키포인트 판 평면 거리 ≥ clamp_height_m ('start'·retreat_normal 제외: +n 후퇴는 항상 허용;
                               판 평면 = 실측 판 평면이 있으면 더 가까운 쪽, S8)
                           토치 팁은 check_torch_tip 일 때만 ('(torch keypoint)' 메시지). clamp 0 이면 plate_top 자체가 하한.
모든 웨이포인트와 ≤5 mm 보간 샘플(광학 원점 궤적)에서 검사한다.
판 평면 (never_cross_plate_plane, 3G clamp_clearance): 기본은 (start, surface_normal) 평면. job.plate_plane_measured (센서 실측
  판 평면, 로봇 쪽으로만 채택) 가 있으면 두 평면 거리 중 **작은 쪽** — 시점/종점이 루트(그루브 바닥) 점인 정련 작업에서 판 상면은 루트
  평면보다 그루브 깊이만큼 로봇 쪽이고, 거친 법선과 실측 법선이 다르면 멀리서 갈라진다 (리뷰 S8). 3G_floor 는 중력 높이 규칙
  (아래 점 높이) 이라 판 평면과 무관하다.

판 상면(1G 바닥)의 출처 (I1/I2): 포락선은 센서 데이터를 직접 보지 않는다. 로컬라이제이션 코드가 base 3D 점
  P_base = T_base←flange · T_flange←opt · [x, 0, z, 1] 의 높이 up·P 로 판 상면을 구해 ScanJob.plate_top_measured_m 에 넣고,
  여기서는 plate_top = max(specimen_top_z_m, plate_top_measured_m) 만 쓴다 (실측값은 올리기만 한다). 원시 프로파일 z 는
  광축 거리이지 높이가 아니므로(프로파일러가 중력·판에 대해 기울어 있을 수 있음) 어떤 높이·거리 규칙에도 쓰지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .frames import (angle_between, interpolate_transforms, inv_transform, make_transform, rot_y, rotation_error,
                     transform_points, unit)
from .tool_body import body_points_base
from .types import SPECULAR_TILT_BAND_DEG, SPECULAR_TILT_DEG, Calib, JobError, ScanJob

__all__ = ["EnvelopeError", "ScanFrames", "sensor_frames", "plate_frame", "Waypoint", "TransitPlan",
           "build_transit_plan", "check_envelope", "tool_keypoints", "scan_s_range", "expected_cruise_s_range",
           "ramp_zones_m", "incidence_tilt_rotation", "PHASES", "RULES", "IN_PLANE_PHASES", "MAX_SAMPLE_STEP_M",
           "laser_plane_line_param", "pass_overscan"]

PHASES = ("start", "home", "ascend", "transit", "descend", "approach_normal", "scan", "retreat_normal")
TRANSIT_PHASES = ("transit",)
IN_PLANE_PHASES = ("scan", "approach_normal", "retreat_normal")   # 판 위 이동: clamp_clearance 를 전 샘플에 적용
CLAMP_EXEMPT_PHASES_1G = ("start", "ascend")                       # 1G: 현재 자세·판에서 멀어지는(+up) 구간은 클램프 면제
CLAMP_EXEMPT_PHASES_3G = ("start", "retreat_normal")               # 3G: 현재 자세·판에서 멀어지는(+n) 구간은 클램프 면제
WORKING_HEIGHT_TOL_M = 0.010     # 1G '작업 높이' 판정 여유: 광학 원점 높이 ≤ plate_top + standoff + 이 값이면 in-plane 샘플로 본다
MAX_SAMPLE_STEP_M = 0.005        # 포락선 보간 샘플 간격 상한 (R4-e)
FLOOR_TOL_M = 1e-6               # 3G 바닥 규칙 수치 허용 (아래 점에서 정확히 정지하므로 등호 허용)

RULES = {
    "never_below_table": "all tool keypoints and the tool body (EOAT mesh hull, when the calib carries it) height >= "
                         "table_z_m + safe_clearance_m "
                         "(flat_1G: also >= plate_top_measured_m, the sensor-measured plate top, when set)",
    "never_cross_plate_plane": "optical origin plate distance >= standoff_min_m; torch tip and tool body >= safe_clearance_m",
    "1G_above_specimen_top": "flat_1G: optical origin (and torch tip if check_torch_tip) height >= "
                             "plate_top = max(specimen_top_z_m, plate_top_measured_m)",
    "3G_floor": "vertical_3G: optical origin (and torch tip if check_torch_tip) height >= floor = "
                "job.floor_z_m (auto-teach: the lower rough point, then the measured lower specimen end) or min(height(start), height(end)) "
                "at all times "
                "(except home legs)",
    "transit_above": "transit legs: optical origin height >= z_safe_opt_m; torch tip (flag) >= z_safe_m",
    "laser_tilt": "scan legs: optical Z vs -n (flat_1G: also vs gravity) <= max_laser_tilt_deg",
    "clamp_clearance": "flat_1G: in-plane legs (scan/approach_normal/retreat_normal, and working-height samples of any "
                       "leg except start/ascend): every tool keypoint height >= plate_top + clamp_height_m; "
                       "vertical_3G: every tool keypoint plate-plane distance >= clamp_height_m (except start/retreat_normal); "
                       "torch tip only if check_torch_tip",
    "direction_1G": "flat_1G: travel direction perpendicular to gravity within direction_tol_deg (ScanJob.validate)",
    "direction_3G": "vertical_3G: travel direction parallel to gravity within direction_tol_deg (ScanJob.validate)",
}


class EnvelopeError(ValueError):
    """안전 포락선 위반 (메시지에 rule=<이름> 포함)."""


def plate_frame(start, u, n) -> np.ndarray:
    """T_base←plate: 원점 start, x = u, z = n, y = z × x (진행부호 무관, 재구성·캘리브 좌표)."""
    u = unit(u)
    n = unit(n)
    y = np.cross(n, u)
    return make_transform(np.column_stack([u, y, n]), start)


# --------------------------------------------------------------------------------------
# 스캔 s 구간 (리드 처리)
# --------------------------------------------------------------------------------------
def scan_s_range(job: ScanJob) -> tuple[float, float, int]:
    """(s_scan_from, s_scan_to, travel_sign). s 는 u=(end−start)/L 축, start 가 0, end 가 L.

    1G: (−lead_in, L+lead_out, +1).
    3G 'down': 위 점 → 아래 점, 리드는 위 점 바깥쪽(lead_in_m)에만, 아래 점에서 정지.
    3G 'up'  : 아래 점 → 위 점, 리드는 위 점 바깥쪽(lead_out_m)에만, 아래 점에서 출발.
    """
    L = job.length_m
    if job.resolved_orientation() != "vertical_3G":
        return -float(job.lead_in_m), L + float(job.lead_out_m), 1
    start_upper = job.upper_point_index() == 0
    if job.vertical_direction == "down":
        if start_upper:
            return -float(job.lead_in_m), L, 1
        return L + float(job.lead_in_m), 0.0, -1
    if job.vertical_direction == "up":
        if start_upper:
            return L, -float(job.lead_out_m), -1
        return 0.0, L + float(job.lead_out_m), 1
    raise JobError(f"vertical_direction must be 'down' or 'up', got {job.vertical_direction!r}")


def expected_cruise_s_range(job: ScanJob, settle_s: float = 0.0) -> tuple[float, float]:
    """순항이 기대되는 s 구간 ∩ [0, L]: 양끝에서 램프 거리 v²/(2a)+v·settle 만큼 안쪽."""
    s_from, s_to, _ = scan_s_range(job)
    d = job.speed_mps ** 2 / (2.0 * job.accel_mps2) + job.speed_mps * float(settle_s)
    lo, hi = sorted((s_from, s_to))
    lo, hi = max(0.0, lo + d), min(job.length_m, hi - d)
    return (lo, hi) if hi > lo else (0.0, 0.0)


def ramp_zones_m(job: ScanJob, settle_s: float = 0.0) -> list:
    """[0, L] 안에서 가감속(순항 아님)이 예상되는 구간 목록 [(lo, hi), …] (프레임은 valid 이지만 in_cruise=False)."""
    lo, hi = expected_cruise_s_range(job, settle_s)
    L = job.length_m
    zones = []
    if lo > 0.0:
        zones.append((0.0, float(lo)))
    if hi < L:
        zones.append((float(hi), float(L)))
    return zones


# --------------------------------------------------------------------------------------
# 스캔 기하
# --------------------------------------------------------------------------------------
@dataclass
class ScanFrames:
    """스캔 기하 (base 프레임, m)."""

    start: np.ndarray
    end: np.ndarray
    u: np.ndarray                      # s 축 (start→end)
    n: np.ndarray
    L: float
    standoff_m: float
    pitch_deg: float
    progression_sign: int
    R_base_eoat: np.ndarray            # 피치 포함
    R_base_sensor_nominal: np.ndarray  # [X_s Y_s Z_s] = [n×u_travel, u_travel, −n] (참고용)
    p_opt_eoat: np.ndarray
    T_flange_eoat: np.ndarray
    T_flange_opt: np.ndarray
    T_base_scan: np.ndarray            # 원점 start, x = progression_sign·travel_sign·u, z = n
    T_base_plate: np.ndarray           # 원점 start, x = u, z = n
    orientation: str = "flat_1G"
    travel_sign: int = 1               # +1: start→end, −1: end→start (3G 방향 규칙)
    s_scan_from: float = 0.0
    s_scan_to: float = 0.0
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))   # −gravity (단위)
    floor_h: Optional[float] = None    # 3G: 툴 키포인트 높이 하한 = job.floor_z_m 또는 min(height(start), height(end))
    incidence_tilt_deg: float = 0.0    # 입사 기울기 τ (부호 규약: 모듈 docstring)
    tilt_phi_rad: float = 0.0          # X_s 둘레 실제 회전각 φ (3G: τ·sign(up·u_t), 1G: −progression_sign·τ)
    d_opt: Optional[np.ndarray] = None # 심 점 → 광학 원점 오프셋 = standoff·R_tilt·n (None → n·standoff)

    def __post_init__(self):
        if self.d_opt is None:
            self.d_opt = self.n * self.standoff_m
        self.d_opt = np.asarray(self.d_opt, dtype=float).reshape(3)

    @property
    def u_travel(self) -> np.ndarray:
        return self.travel_sign * self.u

    def height(self, p) -> np.ndarray:
        return np.asarray(p, dtype=float) @ self.up

    def seam_point(self, s: float) -> np.ndarray:
        """광학 Z 가 향하는 판 위의 점 (심 점) = start + u·s."""
        return self.start + self.u * float(s)

    def opt_origin(self, s: float) -> np.ndarray:
        return self.seam_point(s) + self.d_opt

    @property
    def origin_height_offset_m(self) -> float:
        """광학 원점 높이 − 심 점 높이 (= up·d_opt; 3G τ>0 → +standoff·sinτ)."""
        return float(self.up @ self.d_opt)

    def T_base_eoat(self, s: float) -> np.ndarray:
        R = self.R_base_eoat
        return make_transform(R, self.opt_origin(s) - R @ self.p_opt_eoat)

    def T_base_flange(self, s: float) -> np.ndarray:
        return self.T_base_eoat(s) @ inv_transform(self.T_flange_eoat)

    def T_base_opt(self, s: float) -> np.ndarray:
        return self.T_base_flange(s) @ self.T_flange_opt

    def T_base_flange_many(self, s_values) -> np.ndarray:
        return np.stack([self.T_base_flange(s) for s in np.asarray(s_values, dtype=float).reshape(-1)])

    def s_of(self, points_base) -> np.ndarray:
        """광학 원점(base) → 그 원점이 바라보는 심 점의 진행좌표 s = u·(p − start − d_opt) (τ=0 이면 u·(p − start))."""
        P = np.asarray(points_base, dtype=float)
        return (P - self.start - self.d_opt) @ self.u

    def laser_tilt_deg(self) -> float:
        """광학 Z 축 vs −n."""
        Z_opt = (self.R_base_eoat @ self.T_flange_eoat[:3, :3].T @ self.T_flange_opt[:3, :3])[:, 2]
        return math.degrees(angle_between(Z_opt, -self.n))


def _rodrigues(axis: np.ndarray, phi: float) -> np.ndarray:
    """단위축 axis 둘레 각 phi [rad] 회전행렬."""
    a = unit(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(phi) * K + (1.0 - math.cos(phi)) * (K @ K)


def incidence_tilt_rotation(job: ScanJob, n: np.ndarray, u_t: np.ndarray, orientation: str) -> tuple[np.ndarray, float]:
    """입사 기울기 τ = job.incidence_tilt_deg → (R_tilt, φ). 라인축 X_s = n × u_t 둘레 회전.

    3G: φ = τ·sign(up·u_t) → R_tilt·(−n) = −(n cosτ + up sinτ): τ>0 이면 광학 Z 가 아래를 향하고(내려다봄) 원점은 심 점 위.
    1G: φ = −progression_sign·τ → pitch_deg(y_e = −progression_sign·X_s 둘레)와 같은 회전 방향 → 레이저 기울기 |pitch + τ|.
    """
    tau = math.radians(float(job.incidence_tilt_deg))
    if tau == 0.0:
        return np.eye(3), 0.0
    X_s = unit(np.cross(n, u_t))
    if orientation == "vertical_3G":
        sigma = 1.0 if float(job.up_unit @ u_t) >= 0.0 else -1.0
        phi = sigma * tau
    else:
        phi = -float(job.progression_sign) * tau
    return _rodrigues(X_s, phi), float(phi)


def sensor_frames(job: ScanJob, calib: Calib) -> ScanFrames:
    """ScanJob + Calib → ScanFrames (job.validate() 포함, 방향 규칙·입사 기울기 포함)."""
    job.validate()
    u = job.unit_dir
    n = job.normal_unit
    L = job.length_m
    sign = int(job.progression_sign)
    orientation = job.resolved_orientation()
    s_from, s_to, travel_sign = scan_s_range(job)
    u_t = travel_sign * u
    R_tilt, phi = incidence_tilt_rotation(job, n, u_t, orientation)
    x_e = R_tilt @ (sign * u_t)
    z_e = R_tilt @ (-n)
    y_e = np.cross(z_e, x_e)
    R_e = np.column_stack([x_e, y_e, z_e]) @ rot_y(math.radians(job.pitch_deg))
    X_s = np.cross(n, u_t)
    R_s = np.column_stack([X_s, u_t, -n])
    x_sc = sign * u_t
    T_scan = make_transform(np.column_stack([x_sc, np.cross(n, x_sc), n]), job.start)
    up = job.up_unit
    floor_h = None
    if orientation == "vertical_3G":
        floor_h = (float(job.floor_z_m) if job.floor_z_m is not None
                   else float(min(job.height(job.start), job.height(job.end))))
    d_opt = float(job.standoff_m) * (R_tilt @ n)
    return ScanFrames(
        start=job.start.copy(), end=job.end.copy(), u=u, n=n, L=L, standoff_m=float(job.standoff_m),
        pitch_deg=float(job.pitch_deg), progression_sign=sign, R_base_eoat=R_e, R_base_sensor_nominal=R_s,
        p_opt_eoat=calib.p_opt_eoat, T_flange_eoat=calib.T_flange_eoat.copy(), T_flange_opt=calib.T_flange_opt.copy(),
        T_base_scan=T_scan, T_base_plate=plate_frame(job.start, u, n),
        orientation=orientation, travel_sign=int(travel_sign), s_scan_from=float(s_from), s_scan_to=float(s_to),
        up=up, floor_h=floor_h, incidence_tilt_deg=float(job.incidence_tilt_deg), tilt_phi_rad=phi, d_opt=d_opt,
    )


# --------------------------------------------------------------------------------------
# 레이저 착점 (laser hit point) · 오버스캔 (사용자 2026-10-01: 오버스캔은 광학 원점이 아니라 착점에서 잰다)
# --------------------------------------------------------------------------------------


def laser_plane_line_param(T_base_opt, line_point, line_dir):
    """레이저 평면 ∩ 직선 → 직선 매개변수 t.

    레이저 평면 = 광학 원점을 지나고 X_opt (라인) 와 Z_opt (광축) 가 펼치는 평면 (법선 Y_opt). 직선 q(t) = line_point + unit(line_dir)·t.
    t = ((o − q0)·Y) / (u·Y). (…,4,4) 로 벡터화된다. 직선이 평면과 나란하면 NaN. 피치·입사 기울기·롤이 모두 반영된다 —
    프로파일이 '실제로 자른' 심 위치 (착점) 이며, 광학 원점의 진행좌표 (ScanFrames.s_of) 와는 standoff·tan(피치) 정도 다르다."""
    T = np.asarray(T_base_opt, dtype=float)
    o = T[..., :3, 3]
    Y = T[..., :3, 1]
    u = unit(line_dir)
    q0 = np.asarray(line_point, dtype=float).reshape(3)
    den = Y @ u
    num = np.sum((o - q0) * Y, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(den) > 1e-9, num / np.where(np.abs(den) > 1e-9, den, 1.0), np.nan)
    return float(t) if np.ndim(t) == 0 else t


def pass_overscan(job: ScanJob, calib: Calib, overscan_m: float, *, top_offset_m: float = 0.0,
                  floor_h: Optional[float] = None, min_overscan_m: Optional[float] = None,
                  overscan_first_m: Optional[float] = None, overscan_last_m: Optional[float] = None,
                  floor_first_extra_m: float = 0.0) -> dict:
    """패스 job (start = 진행 시작 끝, end = 진행 끝 끝; travel_sign +1) → **착점**이 start 앞 overscan 에서 end 뒤 overscan 까지
    지나가도록 하는 광학 원점의 job 매개변수 [s_from, s_to] 와 그 플랜지 자세.

    착점 = 레이저 평면 ∩ 측정 판 상면 위의 심 직선 q(t) = start + n·top_offset_m + u·t (start/end 가 루트(그루브 바닥) 점이면
    top_offset_m = 그루브 깊이). 툴 자세가 패스 내내 같으므로 착점 매개변수 h(s) = s + δ (δ = hit_offset_m, 피치·입사 기울기·
    progression_sign 으로 정해지며 진행 방향이 바뀌고 툴 자세를 유지하면 부호가 바뀐다) → s_from = −overscan − δ,
    s_to = L + overscan − δ. overscan_first_m / overscan_last_m 로 끝마다 다르게 줄 수 있다 (0 = 착점이 그 점에서 정확히 시작/끝:
    판 안쪽 점, 예: auto-teach 패스 1 의 보정된 E).

    floor_h (3G 바닥 = 실측 아래 끝 또는 아래 주어진 점의 높이): 가장 낮은 툴 키포인트가 그 아래로 가면 그 끝의 오버스캔을 바닥까지
    줄인다 (광학 원점·토치 팁; 툴 자세가 패스 내내 같아 팁 높이는 원점 높이 + 상수, 원점 높이는 s 에 선형). 반환 dict 의 overscan_first_m /
    overscan_last_m 이 실제 착점 오버스캔이며, 그 끝에 필요한 오버스캔 (min(계획, min_overscan_m); min_overscan_m 없으면 계획 그대로 —
    auto-teach 는 끝 판정이 요구하는 길이를 준다) 보다 작아진 끝은 'floor_blocked' 에 담긴다 (호출자가 'not reachable under the 3G floor
    rule' 로 거부한다; 여기서는 예외 없음). floor_first_extra_m: 첫 끝 (패스 시작) 에만 더하는 바닥 여유 — 패스 시작까지의 로컬 이동이 시작점보다
    낮게 지나가는 만큼 (auto-teach: 기운 3G 판에서 현재 판 거리의 면내 구간).

    job.table_z_m (2026-10-05, 임시 입력 — 목표는 테이블 높이 없이): 모든 키포인트와 툴 몸체 (calib.body_points_eoat) 가
    table_z + safe_clearance_m 아래로 가지 않도록 같은 방식으로 끝을 줄인다 (3G 토치 위 자세에서 센서 카메라 끝이 원점보다 ~175 mm 아래).
    두 바닥 중 높은 쪽이 끝을 정하고 'floor_source' 에 남는다. 줄인 끝은 바닥 + job.pose_tol_m 에 계획한다 (실측 도착이 계획보다
    그만큼 어긋나도 다음 구간의 시작이 바닥 검사에 걸리지 않게)."""
    fr = sensor_frames(job, calib)
    if fr.travel_sign != 1:
        raise ValueError("pass_overscan: the pass job must travel from start to end (travel_sign +1; set vertical_direction for 3G)")
    L = fr.L
    q0 = fr.start + fr.n * float(top_offset_m)
    delta = float(laser_plane_line_param(fr.T_base_opt(0.0), q0, fr.u))
    if not math.isfinite(delta):
        raise ValueError("pass_overscan: laser plane is parallel to the seam (no hit point)")
    ov_a = float(overscan_m if overscan_first_m is None else overscan_first_m)
    ov_b = float(overscan_m if overscan_last_m is None else overscan_last_m)
    s_from = -ov_a - delta
    s_to = L + ov_b - delta
    out = {"hit_offset_m": delta, "overscan_m": float(overscan_m), "overscan_first_m": ov_a, "overscan_last_m": ov_b,
           "planned_first_m": ov_a, "planned_last_m": ov_b, "floor_limited": [], "floor_blocked": [], "floor_h": floor_h,
           "L_m": L, "top_offset_m": float(top_offset_m)}
    origin_floors = []                                             # (광학 원점 높이 하한, 출처)
    if floor_h is not None:
        h0 = float(fr.height(fr.opt_origin(0.0)))
        kp = tool_keypoints(fr.T_base_flange(0.0), calib, job)
        names = ("opt",) + (("tip",) if job.check_torch_tip else ())          # 1G 시편 상단 규칙과 같은 키포인트
        below = max(0.0, h0 - min(float(fr.height(kp[nm])) for nm in names))     # 토치 팁이 원점보다 아래인 만큼
        origin_floors.append((float(floor_h) + below, "3G floor"))
        out["floor_keypoint_below_origin_m"] = below
    if job.table_z_m is not None:
        below_t = max(0.0, lowest_below_origin(fr.T_base_flange(0.0), calib, job))   # 모든 키포인트 + 몸체
        out["table_floor_h"] = float(job.table_z_m) + float(job.safe_clearance_m)
        out["table_below_origin_m"] = below_t
        origin_floors.append((out["table_floor_h"] + below_t, "table"))
    out["floor_source"] = {}
    if origin_floors:
        dh = float(fr.up @ fr.u)                                   # 원점 높이의 s 기울기
        h0 = float(fr.height(fr.opt_origin(0.0)))
        origin_floor, src_ = max(origin_floors, key=lambda t: t[0])
        for which in ("first", "last"):
            s_ = s_from if which == "first" else s_to
            h_ = h0 + dh * s_
            f_ = origin_floor + (max(0.0, float(floor_first_extra_m)) if which == "first" else 0.0) + float(job.pose_tol_m)
            if h_ >= f_ - FLOOR_TOL_M or abs(dh) < 1e-9:
                continue
            s_lim = (f_ - h0) / dh                                 # 가장 낮은 키포인트가 바닥에 닿는 매개변수
            if which == "first":
                got = -(s_lim + delta)                             # 착점 매개변수 s_lim + δ 가 start 앞 got 만큼
                s_from = s_lim
            else:
                got = (s_lim + delta) - L
                s_to = s_lim
            out[f"overscan_{which}_m"] = float(got)
            out["floor_limited"].append(which)
            out["floor_source"][which] = src_
            planned = ov_a if which == "first" else ov_b
            need = planned if min_overscan_m is None else min(planned, float(min_overscan_m))
            if got < need - 1e-9:
                out["floor_blocked"].append(which)
    out.update(frames=fr, s_from=float(s_from), s_to=float(s_to), hit_from=float(s_from + delta), hit_to=float(s_to + delta),
               T_from=fr.T_base_flange(s_from), T_to=fr.T_base_flange(s_to), origin_from=fr.opt_origin(s_from),
               origin_to=fr.opt_origin(s_to), hit_point_from=q0 + fr.u * (s_from + delta), hit_point_to=q0 + fr.u * (s_to + delta))
    return out


# --------------------------------------------------------------------------------------
# 키포인트
# --------------------------------------------------------------------------------------
def tool_keypoints(T_base_flange: np.ndarray, calib: Calib, job: ScanJob) -> dict[str, np.ndarray]:
    """플랜지 포즈 → 툴 키포인트 (base, m): flange, eoat, opt(광학 원점 = 기준점), tip(토치 팁 추정)."""
    T_bf = np.asarray(T_base_flange, dtype=float)
    T_be = T_bf @ calib.T_flange_eoat
    p_opt = calib.p_opt_eoat
    tip_eoat = np.array([job.torch_lateral_m[0], job.torch_lateral_m[1], p_opt[2] + job.torch_protrusion_m])
    return {
        "flange": T_bf[:3, 3].copy(),
        "eoat": T_be[:3, 3].copy(),
        "opt": transform_points(T_be, p_opt),
        "tip": transform_points(T_be, tip_eoat),
    }


def _keypoints_many(Ts: np.ndarray, calib: Calib, job: ScanJob) -> dict[str, np.ndarray]:
    """(N,4,4) → 각 키포인트 (N,3)."""
    Ts = np.asarray(Ts, dtype=float)
    T_be = Ts @ calib.T_flange_eoat
    p_opt = calib.p_opt_eoat
    tip_eoat = np.array([job.torch_lateral_m[0], job.torch_lateral_m[1], p_opt[2] + job.torch_protrusion_m])
    return {
        "flange": Ts[:, :3, 3],
        "eoat": T_be[:, :3, 3],
        "opt": np.einsum("nij,j->ni", T_be[:, :3, :3], p_opt) + T_be[:, :3, 3],
        "tip": np.einsum("nij,j->ni", T_be[:, :3, :3], tip_eoat) + T_be[:, :3, 3],
    }


def body_min_heights(Ts: np.ndarray, calib: Calib, up: np.ndarray) -> Optional[np.ndarray]:
    """플랜지 포즈 (N,4,4) → 툴 몸체 (EOAT 메시 껍질) 의 가장 낮은 높이 (N,). 몸체가 없으면 None."""
    B = getattr(calib, "body_points_eoat", None)
    if B is None:
        return None
    P = body_points_base(np.asarray(Ts, dtype=float) @ calib.T_flange_eoat, B)
    return (P @ np.asarray(up, dtype=float)).min(axis=1)


def body_min_plate_dist(Ts: np.ndarray, calib: Calib, origin: np.ndarray, normal: np.ndarray,
                        measured_plane=None) -> Optional[np.ndarray]:
    """플랜지 포즈 (N,4,4) → 툴 몸체의 판 평면 (origin, normal; 로봇 쪽 +) 최소 거리 (N,). 실측 판 평면 (점, 법선) 이 있으면 둘 중
    가까운 쪽. 몸체가 없으면 None."""
    B = getattr(calib, "body_points_eoat", None)
    if B is None:
        return None
    P = body_points_base(np.asarray(Ts, dtype=float) @ calib.T_flange_eoat, B)
    n = np.asarray(normal, dtype=float)
    d = ((P - np.asarray(origin, dtype=float)) @ n).min(axis=1)
    if measured_plane is not None:
        d = np.minimum(d, ((P - np.asarray(measured_plane[0], dtype=float)) @ np.asarray(measured_plane[1], dtype=float)).min(axis=1))
    return d


def lowest_below_origin(T_base_flange: np.ndarray, calib: Calib, job: ScanJob, *, body: bool = True,
                        names: tuple = ("flange", "eoat", "opt", "tip")) -> float:
    """한 포즈에서 광학 원점 높이 − (키포인트 names + 툴 몸체 중 가장 낮은 높이) [m] (≥ 0 이면 원점보다 그만큼 아래가 있다)."""
    kp = tool_keypoints(T_base_flange, calib, job)
    up = job.up_unit
    h0 = float(kp["opt"] @ up)
    low = min(float(kp[k] @ up) for k in names)
    if body:
        bh = body_min_heights(np.asarray(T_base_flange, dtype=float)[None], calib, up)
        if bh is not None:
            low = min(low, float(bh[0]))
    return h0 - low


# --------------------------------------------------------------------------------------
# 웨이포인트 · 계획
# --------------------------------------------------------------------------------------
@dataclass
class Waypoint:
    name: str
    phase: str                     # 이 웨이포인트로 '들어오는' 구간의 성격 (PHASES)
    T_base_flange: np.ndarray
    speed_mps: float
    accel_mps2: float
    motion: str = "move_l"         # move_l | move_j (home 복귀 등)
    s_from_m: Optional[float] = None
    s_to_m: Optional[float] = None
    floor_rule: bool = True        # 3G 바닥 규칙 적용 여부 (home 왕복 구간만 False)

    def xyz_rpy(self) -> tuple[np.ndarray, np.ndarray]:
        from .frames import xyz_rpy_from_transform
        return xyz_rpy_from_transform(self.T_base_flange)

    def opt_origin(self, calib: Calib) -> np.ndarray:
        """이 웨이포인트에서의 광학 원점 (base, m)."""
        return transform_points(self.T_base_flange @ calib.T_flange_eoat, calib.p_opt_eoat)


@dataclass
class TransitPlan:
    waypoints: list
    frames: ScanFrames
    orientation: str
    z_safe_m: float                       # 토치 팁·일반 키포인트 transit 높이 하한 = max(specimen_top, 위 점) + clearance
    z_safe_opt_m: Optional[float]         # 광학 원점 transit 높이 하한 (1G: z_safe + standoff, 3G: z_safe)
    specimen_top_z_m: float
    scan_leg_indices: list = field(default_factory=list)   # 스캔 구간 웨이포인트 인덱스(도착점)
    notes: list = field(default_factory=list)
    floor_z_m: Optional[float] = None     # 3G: 툴 키포인트 높이 하한 (job.floor_z_m 또는 start/end 중 낮은 점), 1G: None
    reference_point: str = "optical"
    vertical_direction: str = "down"
    travel_sign: int = 1
    s_scan_from_m: float = 0.0
    s_scan_to_m: float = 0.0
    rules: list = field(default_factory=list)              # 이 계획에 적용된 규칙명 (RULES 키)
    calib: Optional[Calib] = None

    def scan_waypoints(self) -> list:
        return [self.waypoints[i] for i in self.scan_leg_indices]

    def opt_origins(self) -> Optional[np.ndarray]:
        if self.calib is None:
            return None
        return np.stack([w.opt_origin(self.calib) for w in self.waypoints])

    def summary(self) -> str:
        lines = [f"orientation={self.orientation} reference={self.reference_point} travel_sign={self.travel_sign:+d} "
                 f"s_scan=[{self.s_scan_from_m:+.3f} → {self.s_scan_to_m:+.3f}] m",
                 f"z_safe={self.z_safe_m:.3f} m z_safe_opt={self.z_safe_opt_m if self.z_safe_opt_m is not None else float('nan'):.3f} m "
                 f"specimen_top={self.specimen_top_z_m:.3f} m floor={'-' if self.floor_z_m is None else f'{self.floor_z_m:.3f} m'}",
                 "rules: " + ", ".join(self.rules)]
        for i, w in enumerate(self.waypoints):
            p = w.T_base_flange[:3, 3]
            o = w.opt_origin(self.calib) if self.calib is not None else None
            ostr = f" opt=({o[0]:+.3f},{o[1]:+.3f},{o[2]:+.3f})" if o is not None else ""
            lines.append(f"[{i:2d}] {w.name:<16s} {w.phase:<15s} {w.motion:<6s} v={w.speed_mps:.3f} a={w.accel_mps2:.2f} "
                         f"flange=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}){ostr}")
        for n_ in self.notes:
            lines.append(f"note: {n_}")
        return "\n".join(lines)


def _specimen_top(job: ScanJob, orientation: str) -> float:
    """바닥 규칙의 판 상면 높이 = max(specimen_top_z_m, plate_top_measured_m) (실측값은 올리기만 한다).
    1G 에서 둘 다 없으면 start/end 높이의 최댓값. 3G 에서는 판 상단 모서리 높이로 해석되며(plate_top_measured_m 도 같은 높이축),
    3G 클램프는 높이가 아니라 판 평면 거리로 검사한다."""
    top = job.plate_top_z_m()
    if top is not None:
        return float(top)
    if orientation == "flat_1G":
        return float(max(job.height(job.start), job.height(job.end)))
    raise EnvelopeError("vertical_3G requires specimen_top_z_m (top edge height of the standing plate in base frame)")


def build_transit_plan(job: ScanJob, calib: Calib, T_home_flange: Optional[np.ndarray] = None, *,
                       transit_speed_mps: float = 0.05, descend_speed_mps: float = 0.02,
                       transit_accel_mps2: float = 0.2, check: bool = True,
                       sample_step_m: float = MAX_SAMPLE_STEP_M) -> TransitPlan:
    """접근·리드·스캔·이탈 웨이포인트 생성 (+ 포락선 검사). 모든 높이 규칙의 기준점 = 광학 원점.

    1G: home → home_above(up) → above_lead_in(z_safe_opt, 스캔 자세) → lead_in(−up 하강) → [scan] → above_exit(상승) → home_above → home
    3G: home → home_above → above_pre_in(z_safe_opt) → pre_in(수직 하강, 판에서 수평 n·clearance 만큼 떨어진 위치)
        → lead_in|scan_start(−n_h 접근) → [scan: 'down' 위→아래(리드아웃 없음) | 'up' 아래→위(리드인 없음)]
        → pre_exit(+n_h 후퇴, 높이 불변) → above_pre_exit(상승) → home_above → home
    '위쪽' 규칙은 transit 구간에만 적용; 3G 바닥 규칙은 home 왕복을 제외한 모든 구간에 적용.
    transit 들어올림 목표는 한계 + job.pose_tol_m (도착 허용) 에 둔다.
    """
    fr = sensor_frames(job, calib)
    orientation = fr.orientation
    top = _specimen_top(job, orientation)
    clearance = float(job.safe_clearance_m)
    lift_margin_m = float(job.pose_tol_m)
    n = fr.n
    up = fr.up
    h_upper = float(max(job.height(job.start), job.height(job.end)))
    z_safe = max(top, h_upper) + clearance                                   # 토치 팁·일반 키포인트 하한
    z_safe_opt = z_safe + (float(job.standoff_m) if orientation == "flat_1G" else 0.0)   # 광학 원점 하한
    s_in, s_out = fr.s_scan_from, fr.s_scan_to
    T_in = fr.T_base_flange(s_in)
    T_out = fr.T_base_flange(s_out)
    rules = ["never_below_table", "never_cross_plate_plane", "transit_above", "laser_tilt", "clamp_clearance"]

    if orientation == "vertical_3G":
        rules += ["direction_3G", "3G_floor"]
        # 법선이 로봇(베이스 원점) 쪽을 향해야 한다: 판 위 점에서 베이스로 향하는 수평 벡터와 같은 방향
        to_base = -job.start.copy()
        to_base -= up * float(to_base @ up)
        if np.linalg.norm(to_base) > 1e-6 and float(np.dot(n, to_base)) <= 0.0:
            raise EnvelopeError("vertical_3G: surface_normal points away from the robot base; "
                                "the normal must point from the plate toward the robot (flip surface_normal)")
        n_h = n - up * float(n @ up)                        # 법선의 수평 성분 (접근/후퇴는 높이 불변)
        if np.linalg.norm(n_h) < 1e-6:
            raise EnvelopeError("vertical_3G: surface_normal has no horizontal component")
        approach_vec = unit(n_h) * clearance
    else:
        rules += ["direction_1G", "1G_above_specimen_top"]
        approach_vec = None

    def lift_to_safe(T: np.ndarray) -> np.ndarray:
        # 광학 원점 ≥ z_safe_opt (+margin), 토치 팁(플래그) ≥ z_safe (+margin) 가 되도록 up 방향으로 들어올림
        kp = tool_keypoints(T, calib, job)
        lift = max(0.0, z_safe_opt + lift_margin_m - float(fr.height(kp["opt"])))
        if job.check_torch_tip:
            lift = max(lift, z_safe + lift_margin_m - float(fr.height(kp["tip"])))
        T2 = np.array(T, dtype=float, copy=True)
        T2[:3, 3] += up * lift
        return T2

    def shifted(T: np.ndarray, d: np.ndarray) -> np.ndarray:
        T2 = np.array(T, dtype=float, copy=True)
        T2[:3, 3] += d
        return T2

    vt, vd, at = float(transit_speed_mps), float(descend_speed_mps), float(transit_accel_mps2)
    wps: list[Waypoint] = []
    notes: list[str] = []

    if T_home_flange is not None:
        T_home = np.asarray(T_home_flange, dtype=float).reshape(4, 4)
        wps.append(Waypoint("home", "start", T_home, 0.0, 0.0, "move_j", floor_rule=False))
        # home_above 는 항상 별도 배열로 만들고(transit 구간 보정이 홈 포즈를 건드리지 않도록) 마지막에 홈과 같으면 제거
        T_home_above = lift_to_safe(T_home)
        wps.append(Waypoint("home_above", "ascend", T_home_above, vd, at, floor_rule=False))
    else:
        T_home = None
        T_home_above = None

    lead_at_start = abs(s_in - (0.0 if fr.travel_sign > 0 else fr.L)) > 1e-12
    lead_at_end = abs(s_out - (fr.L if fr.travel_sign > 0 else 0.0)) > 1e-12
    name_in = "lead_in" if lead_at_start else "scan_start"
    name_out = "lead_out" if lead_at_end else "scan_end"

    if orientation == "flat_1G":
        above_in = lift_to_safe(T_in)
        above_out = lift_to_safe(T_out)
        wps.append(Waypoint("above_lead_in", "transit" if wps else "start", above_in, vt, at))
        wps.append(Waypoint(name_in, "descend", T_in, vd, at))
    else:
        pre_in = shifted(T_in, approach_vec)
        pre_out = shifted(T_out, approach_vec)
        above_in = lift_to_safe(pre_in)
        above_out = lift_to_safe(pre_out)
        wps.append(Waypoint("above_pre_in", "transit" if wps else "start", above_in, vt, at))
        wps.append(Waypoint("pre_in", "descend", pre_in, vd, at))
        wps.append(Waypoint(name_in, "approach_normal", T_in, vd, at))

    scan_idx = []
    wps.append(Waypoint(name_out, "scan", T_out, float(job.speed_mps), float(job.accel_mps2), s_from_m=s_in, s_to_m=s_out))
    scan_idx.append(len(wps) - 1)
    exit_side = "out"
    if job.bidirectional:
        wps.append(Waypoint(name_in + "_return", "scan", T_in, float(job.speed_mps), float(job.accel_mps2),
                            s_from_m=s_out, s_to_m=s_in))
        scan_idx.append(len(wps) - 1)
        exit_side = "in"

    if orientation == "flat_1G":
        above_exit = above_in if exit_side == "in" else above_out
        wps.append(Waypoint("above_exit", "ascend", above_exit, vd, at))
    else:
        pre_exit = pre_in if exit_side == "in" else pre_out
        above_exit = above_in if exit_side == "in" else above_out
        wps.append(Waypoint("pre_exit", "retreat_normal", pre_exit, vd, at))
        wps.append(Waypoint("above_pre_exit", "ascend", above_exit, vd, at))

    if T_home is not None:
        wps.append(Waypoint("home_above", "transit", T_home_above, vt, at, floor_rule=False))
        wps.append(Waypoint("home", "descend", T_home, vd, at, floor_rule=False))

    # transit 구간 보정: 양끝 자세가 다르면 slerp 도중 광학 원점·토치 팁이 끝점보다 낮아질 수 있다(3G 홈 왕복 등).
    # 구간을 보간해 부족분을 구하고 양끝 웨이포인트를 같은 양만큼 up 으로 들어올린다 (배열 공유 → 같은 점은 함께 이동).
    for i in range(1, len(wps)):
        if wps[i].phase not in TRANSIT_PHASES:
            continue
        T0, T1 = wps[i - 1].T_base_flange, wps[i].T_base_flange
        span = max(float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])), rotation_error(T0[:3, :3], T1[:3, :3]) * 0.5)
        nsmp = int(min(4000, max(2, math.ceil(span / min(sample_step_m, MAX_SAMPLE_STEP_M)) + 1)))
        kp = _keypoints_many(interpolate_transforms(T0, T1, nsmp), calib, job)
        need = z_safe_opt + lift_margin_m - float(fr.height(kp["opt"]).min())
        if job.check_torch_tip:
            need = max(need, z_safe + lift_margin_m - float(fr.height(kp["tip"]).min()))
        if need > 0.0:
            lifted = []
            for T in (T0, T1):
                if T is T_home or any(T is L_ for L_ in lifted):
                    continue
                T[:3, 3] += up * need
                lifted.append(T)
            notes.append(f"transit leg → {wps[i].name}: endpoints raised {need * 1e3:.1f} mm for orientation swing "
                         "(optical origin / torch tip)")
    if T_home is not None and np.allclose(T_home_above, T_home):
        wps = [w for w in wps if w.name != "home_above"]
    scan_idx = [i for i, w in enumerate(wps) if w.phase == "scan"]

    lead_req = job.lead_distance_required_m()
    leads_used = [v for v, used in ((job.lead_in_m, lead_at_start), (job.lead_out_m, lead_at_end)) if used]
    if leads_used and min(leads_used) < lead_req:
        notes.append(f"lead {min(leads_used):.3f} m < v^2/(2a)+0.5v = {lead_req:.3f} m: cruise may not cover [0, L]")
    if orientation == "vertical_3G":
        zones = ", ".join(f"[{a:.4f}, {b:.4f}]" for a, b in ramp_zones_m(job))
        if job.vertical_direction == "down":
            notes.append(f"3G down: no lead-out below the lower point; decel zone s∈{zones} m inside the "
                         "specimen (frames valid but in_cruise=False)")
        else:
            notes.append(f"3G up: no lead-in below the lower point; accel zone s∈{zones} m inside the "
                         "specimen (frames valid but in_cruise=False)")
        notes.append(f"3G floor (optical origin, torch tip) = {fr.floor_h:.4f} m "
                     f"({'job floor_z_m' if job.floor_z_m is not None else 'lower given point'}); "
                     "approach/retreat along horizontal plate normal "
                     f"{np.round(unit(approach_vec), 4).tolist()}")
    if fr.incidence_tilt_deg != 0.0:
        dh = fr.origin_height_offset_m
        if orientation == "vertical_3G":
            look = "DOWNWARD (origin above the seam point)" if dh > 0 else "UPWARD (origin below the seam point)"
            notes.append(f"incidence tilt {fr.incidence_tilt_deg:+.1f} deg: sensor looks {look} by "
                         f"{abs(dh) * 1e3:.1f} mm = standoff*sin(tilt); sign convention '+ = looking downward' "
                         "to be confirmed with the operator ('3G slightly upward' request)")
        else:
            notes.append(f"incidence tilt {fr.incidence_tilt_deg:+.1f} deg (1G): composes with pitch {fr.pitch_deg:+.1f} deg "
                         f"about the laser-line axis; optical origin shifted {float(fr.d_opt @ fr.u) * 1e3:+.1f} mm along travel")
        if abs(fr.incidence_tilt_deg - SPECULAR_TILT_DEG) < SPECULAR_TILT_BAND_DEG:
            notes.append(f"WARNING: incidence tilt near specular -theta_tri/2 = {SPECULAR_TILT_DEG} deg")

    plan = TransitPlan(waypoints=wps, frames=fr, orientation=orientation, z_safe_m=z_safe, z_safe_opt_m=z_safe_opt,
                       specimen_top_z_m=top, scan_leg_indices=scan_idx, notes=notes, floor_z_m=fr.floor_h,
                       reference_point=job.reference_point, vertical_direction=job.vertical_direction,
                       travel_sign=fr.travel_sign, s_scan_from_m=s_in, s_scan_to_m=s_out, rules=rules, calib=calib)
    if check:
        check_envelope(plan, job, calib, sample_step_m=sample_step_m)
    return plan


def check_envelope(plan: TransitPlan, job: ScanJob, calib: Calib, *, sample_step_m: float = MAX_SAMPLE_STEP_M,
                   max_samples_per_leg: int = 4000) -> dict:
    """모든 웨이포인트 + 구간 보간 샘플(≤ 5 mm, 광학 원점 궤적)에 포락선 규칙 적용. 위반 시 EnvelopeError(rule=…).
    여유는 job.safe_clearance_m (한계 L) 하나. 첫 웨이포인트 (지금 실측 자세) 의 바닥 규칙은 job.pose_tol_m (플랫폼 도착 허용) 만큼
    봐준다 — 이미 거기 있는 로봇의 도착 오차. 첫 구간은 출발 자세가 한계 아래로 들어가 있는 만큼만 (도착 허용 상한) — 출발보다 더 낮게
    가지 않는다 (4d 검토: 바닥 옆 제자리 회전이 첫 구간 전체에 도착 허용을 받아 툴 몸체가 출발보다 낮게 한계 아래를 스쳤다).
    반환: 요약 통계 (min_opt_height, min_tip_height, floor_margin_min, …)."""
    clearance = float(job.safe_clearance_m)
    start_tol = max(float(job.pose_tol_m), FLOOR_TOL_M)
    fr = plan.frames
    orientation = plan.orientation
    # 판 상면은 올리기만 한다: 계획을 만든 뒤 job 의 실측 판 상면(plate_top_measured_m)이 올라갔으면 그 값을 따른다
    top = float(plan.specimen_top_z_m)
    job_top = job.plate_top_z_m()
    if job_top is not None and float(job_top) > top:
        top = float(job_top)
    n = fr.n
    up = fr.up
    start = fr.start
    wps = plan.waypoints
    if not wps:
        raise EnvelopeError("empty plan")
    meas_plane = job.plate_plane() if hasattr(job, "plate_plane") else None

    def plate_dist(P: np.ndarray) -> np.ndarray:
        """판 평면 거리 (로봇 쪽 +): (start, n) 평면과 실측 판 평면(있으면) 중 작은 값 — 보수적 (S8)."""
        d = (P - start) @ n
        if meas_plane is not None:
            d = np.minimum(d, (P - meas_plane[0]) @ meas_plane[1])
        return d

    sample_step_m = min(float(sample_step_m), MAX_SAMPLE_STEP_M)
    floor = plan.floor_z_m
    clamp = float(job.clamp_height_m)
    clamp_on = "clamp_clearance" in plan.rules
    # 1G 작업 높이 상한: 광학 원점이 이 높이 이하이면 '판 위(in-plane)' 샘플로 보고 클램프를 적용한다
    # (스캔면 = max(plate_top, start/end 높이) 위 standoff; transit 높이 z_safe_opt 는 그보다 safe_clearance 만큼 더 높다)
    working_h_max = (max(top, float(fr.height(fr.start)), float(fr.height(fr.end))) + float(job.standoff_m)
                     + WORKING_HEIGHT_TOL_M)
    # 1G 실측 판 상면: 모든 키포인트(플래그 무관)가 그 아래로 내려가지 못하는 물리적 바닥 (never_below_table 을 올린다)
    measured_floor = (float(job.plate_top_measured_m) if (orientation == "flat_1G" and job.plate_top_measured_m is not None)
                      else None)
    stats = {"min_z_keypoint": math.inf, "min_opt_height": math.inf, "min_tip_height": math.inf,
             "min_plate_dist_opt": math.inf, "min_plate_dist_tip": math.inf,
             "floor_margin_min": math.inf if floor is not None else float("nan"),
             "max_laser_tilt_deg": 0.0, "n_samples": 0, "sample_step_m": sample_step_m,
             "plate_top_m": float(top), "plate_top_measured_m": measured_floor,
             "clamp_height_m": clamp, "clamp_margin_min": math.inf,
             "plate_plane": "start/surface_normal" + (" + measured plate plane (closer of the two)" if meas_plane is not None else ""),
             "reference_point": plan.reference_point, "rules": list(plan.rules)}

    def fail(rule: str, msg: str, wp: Waypoint):
        raise EnvelopeError(f"[{wp.name}/{wp.phase}] rule={rule}: {msg}")

    def check_clamp(kp: dict, phase: str, wp: Waypoint):
        """clamp_clearance: 1G 높이(up·P) / 3G 판 평면 거리(n·(P−start)) — 모두 base 3D 키포인트에서 계산 (I2)."""
        names = ("flange", "eoat", "opt") + (("tip",) if job.check_torch_tip else ())
        if orientation == "flat_1G":
            if phase in CLAMP_EXEMPT_PHASES_1G:
                return
            h_opt = fr.height(kp["opt"])
            sel = np.ones(h_opt.shape[0], dtype=bool) if phase in IN_PLANE_PHASES else (h_opt <= working_h_max)
            if not sel.any():
                return
            lim = top + clamp
            for k in names:
                hk = fr.height(kp[k])[sel]
                stats["clamp_margin_min"] = min(stats["clamp_margin_min"], float(hk.min()) - lim)
                if hk.min() < lim - 1e-9:
                    what = "optical origin" if k == "opt" else ("torch tip" if k == "tip" else k)
                    fail("clamp_clearance",
                         f"{what} height {hk.min():.4f} m < plate_top+clamp_height={top:.4f}+{clamp:.4f}={lim:.4f} m "
                         "during in-plane motion" + (" (torch keypoint)" if k == "tip" else ""), wp)
        else:
            if phase in CLAMP_EXEMPT_PHASES_3G:
                return
            for k in names:
                dk = plate_dist(kp[k])
                stats["clamp_margin_min"] = min(stats["clamp_margin_min"], float(dk.min()) - clamp)
                if dk.min() < clamp - 1e-9:
                    what = "optical origin" if k == "opt" else ("torch tip" if k == "tip" else k)
                    fail("clamp_clearance",
                         f"{what} distance to plate plane {dk.min():.4f} m < clamp_height={clamp:.4f} m"
                         + (" (torch keypoint)" if k == "tip" else ""), wp)

    def floor_margins(Ts: np.ndarray, wp: Waypoint) -> dict:
        """바닥 규칙별 최소 여유 [m] (음수 = 한계 아래): 'kp' 키포인트 ≥ 테이블+여유 / 실측 판 상면, 'body' 툴 몸체 ≥ 테이블+여유,
        'floor' 3G 바닥 (광학 원점·토치 팁). 적용되지 않는 규칙은 없다."""
        out: dict = {}
        kp_ = _keypoints_many(Ts, calib, job)
        if job.table_z_m is not None or measured_floor is not None:
            zl = -math.inf
            if job.table_z_m is not None:
                zl = float(job.table_z_m) + clearance
            if measured_floor is not None and measured_floor > zl:
                zl = measured_floor
            out["kp"] = float(np.stack([fr.height(kp_[k]) for k in kp_]).min()) - zl
            hb = body_min_heights(Ts, calib, up)
            if hb is not None and job.table_z_m is not None:
                out["body"] = float(hb.min()) - (float(job.table_z_m) + clearance)
        if floor is not None and wp.floor_rule:
            names_ = ("opt",) + (("tip",) if job.check_torch_tip else ())
            out["floor"] = min(float(fr.height(kp_[k]).min()) for k in names_) - float(floor)
        return out

    def check_samples(Ts: np.ndarray, wp: Waypoint, phase: str, tol=FLOOR_TOL_M):
        """tol: 바닥 규칙의 허용 [m] — 수 하나 또는 {'kp', 'body', 'floor'} (첫 구간: 출발 자세의 결손만큼)."""
        tol_kp, tol_body, tol_floor = ((tol.get("kp", FLOOR_TOL_M), tol.get("body", FLOOR_TOL_M), tol.get("floor", FLOOR_TOL_M))
                                       if isinstance(tol, dict) else (tol, tol, tol))
        kp = _keypoints_many(Ts, calib, job)
        allh = np.stack([fr.height(kp[k]) for k in kp])            # (4, N) 높이
        h_opt = fr.height(kp["opt"])
        h_tip = fr.height(kp["tip"])
        stats["min_z_keypoint"] = min(stats["min_z_keypoint"], float(allh.min()))
        stats["min_opt_height"] = min(stats["min_opt_height"], float(h_opt.min()))
        stats["min_tip_height"] = min(stats["min_tip_height"], float(h_tip.min()))
        stats["n_samples"] += int(Ts.shape[0])
        h_body = body_min_heights(Ts, calib, up)
        if h_body is not None:
            stats["min_body_height"] = min(stats.get("min_body_height", math.inf), float(h_body.min()))
        # 테이블 (모든 키포인트); 1G 실측 판 상면이 있으면 그 값으로 바닥을 올린다 (판은 테이블 위에 놓여 있으므로 항상 ≥ 테이블)
        if job.table_z_m is not None or measured_floor is not None:
            zlim = -math.inf
            src = []
            if job.table_z_m is not None:
                zlim = float(job.table_z_m) + clearance
                src.append(f"table_z+clearance={zlim:.4f} m")
            if measured_floor is not None and measured_floor > zlim:
                zlim = measured_floor
                src.append(f"measured plate top={measured_floor:.4f} m")
            if allh.min() < zlim - tol_kp:                           # 지금 실측 자세는 도착 허용, 첫 구간은 출발의 결손만큼
                fail("never_below_table",
                     f"tool keypoint height {allh.min():.4f} m below floor {zlim:.4f} m ({'; '.join(src)})", wp)
            if h_body is not None and job.table_z_m is not None and h_body.min() < float(job.table_z_m) + clearance - tol_body:
                fail("never_below_table",
                     f"tool body (EOAT mesh, {calib.body_source or 'calib body points'}) lowest point {h_body.min():.4f} m below "
                     f"table_z+clearance={float(job.table_z_m) + clearance:.4f} m - the sensor housing / torch would come too close "
                     "to the table", wp)
        # 판 평면 (로봇 쪽): 광학 원점 ≥ standoff_min, 토치 팁·툴 몸체 ≥ 한계 L
        d_opt = plate_dist(kp["opt"])
        d_tip = plate_dist(kp["tip"])
        stats["min_plate_dist_opt"] = min(stats["min_plate_dist_opt"], float(d_opt.min()))
        stats["min_plate_dist_tip"] = min(stats["min_plate_dist_tip"], float(d_tip.min()))
        if d_opt.min() < job.standoff_min_m - 1e-9:
            fail("never_cross_plate_plane",
                 f"optical origin distance to plate plane {d_opt.min():.4f} m < standoff_min_m={job.standoff_min_m}", wp)
        if d_tip.min() < clearance - 1e-9:
            fail("never_cross_plate_plane",
                 f"torch tip distance to plate plane {d_tip.min():.4f} m < safe_clearance {clearance:.4f} m (tool would come too close "
                 "to the plate)", wp)
        d_body = body_min_plate_dist(Ts, calib, start, n, meas_plane)
        if d_body is not None:
            stats["min_plate_dist_body"] = min(stats.get("min_plate_dist_body", math.inf), float(d_body.min()))
            if d_body.min() < clearance - 1e-9:
                fail("never_cross_plate_plane",
                     f"tool body (EOAT mesh) distance to plate plane {d_body.min():.4f} m < safe_clearance {clearance:.4f} m", wp)
        # 1G: 광학 원점(기준점)·토치 팁(플래그) 시편 상단 아래 금지
        if orientation == "flat_1G":
            if h_opt.min() < top - 1e-9:
                fail("1G_above_specimen_top",
                     f"optical origin height {h_opt.min():.4f} m below specimen_top_z={top:.4f} m", wp)
            if job.check_torch_tip and h_tip.min() < top - 1e-9:
                fail("1G_above_specimen_top",
                     f"torch tip height {h_tip.min():.4f} m below specimen_top_z={top:.4f} m (torch keypoint)", wp)
        # 클램프 여유 (1G: 판 위 구간 높이, 3G: 판 평면 거리) — 모든 키포인트
        if clamp_on:
            check_clamp(kp, phase, wp)
        # 3G 바닥: 광학 원점은 어떤 순간에도 아래 점 아래로 금지 (home 왕복 제외)
        if floor is not None and wp.floor_rule:
            names = ("opt",) + (("tip",) if job.check_torch_tip else ())          # 1G_above_specimen_top 과 같은 키포인트
            lows = {k: float(fr.height(kp[k]).min()) for k in names}
            low = min(lows, key=lows.get)
            margin = lows[low] - floor
            stats["floor_margin_min"] = min(stats["floor_margin_min"], margin)
            if margin < -tol_floor:
                what = "optical origin" if low == "opt" else ("torch tip" if low == "tip" else low)
                src = ("job floor_z_m" if job.floor_z_m is not None else "the lower of start/end - the lower given point")
                fail("3G_floor", f"{what} height {lows[low]:.4f} m below floor_z={floor:.4f} m ({src}; the tool may not go below it)"
                     + (" (torch keypoint)" if low == "tip" else ""), wp)
        # transit: 광학 원점 ≥ z_safe_opt, 토치 팁(플래그) ≥ z_safe
        if phase in TRANSIT_PHASES:
            if plan.z_safe_opt_m is not None and h_opt.min() < plan.z_safe_opt_m - 1e-9:
                fail("transit_above",
                     f"transit optical origin height {h_opt.min():.4f} m < z_safe_opt={plan.z_safe_opt_m:.4f} m "
                     "(max(specimen_top, upper point) + safe_clearance" + (" + standoff" if orientation == "flat_1G" else "") + ")", wp)
            if job.check_torch_tip and h_tip.min() < plan.z_safe_m - 1e-9:
                fail("transit_above",
                     f"transit torch tip height {h_tip.min():.4f} m < specimen_top+clearance={plan.z_safe_m:.4f} m (torch keypoint)", wp)
        # scan: 레이저 기울기
        if phase == "scan":
            R_bo = Ts[:, :3, :3] @ calib.T_flange_opt[:3, :3]
            Z_opt = R_bo[:, :, 2]
            cos_n = np.clip(Z_opt @ (-n), -1.0, 1.0)
            tilt = np.degrees(np.arccos(cos_n))
            stats["max_laser_tilt_deg"] = max(stats["max_laser_tilt_deg"], float(tilt.max()))
            if tilt.max() > job.max_laser_tilt_deg + 1e-6:
                fail("laser_tilt", f"laser tilt vs plate normal {tilt.max():.2f} deg > max_laser_tilt_deg={job.max_laser_tilt_deg}", wp)
            if orientation == "flat_1G":
                tilt_g = np.degrees(np.arccos(np.clip(Z_opt @ (-up), -1.0, 1.0)))
                if tilt_g.max() > job.max_laser_tilt_deg + 1e-6:
                    fail("laser_tilt", f"laser tilt vs gravity {tilt_g.max():.2f} deg > max_laser_tilt_deg={job.max_laser_tilt_deg} "
                         "(plate too inclined for flat_1G; declare vertical_3G?)", wp)

    # 첫 웨이포인트(현재 위치) 단독 검사
    check_samples(wps[0].T_base_flange[None], wps[0], wps[0].phase, start_tol)
    # 첫 구간: 출발 자세의 결손 (한계 아래로 들어간 양, 도착 허용 상한) 만큼만 — 출발보다 낮게 가지 않는다
    m0 = floor_margins(wps[0].T_base_flange[None], wps[1] if len(wps) > 1 else wps[0])
    first_tol = {k_: min(start_tol, max(0.0, -float(v_)) + FLOOR_TOL_M) for k_, v_ in m0.items()}
    for i in range(1, len(wps)):
        T0 = wps[i - 1].T_base_flange
        T1 = wps[i].T_base_flange
        dist = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        ang = rotation_error(T0[:3, :3], T1[:3, :3])
        span = max(dist, ang * 0.5)                      # 회전은 반경 0.5 m 기준 호 길이로 환산 (광학 원점 오프셋 ≤ 0.5 m)
        nsmp = int(min(max_samples_per_leg, max(2, math.ceil(span / sample_step_m) + 1)))
        Ts = interpolate_transforms(T0, T1, nsmp)
        check_samples(Ts, wps[i], wps[i].phase, first_tol if i == 1 else FLOOR_TOL_M)
    return stats
