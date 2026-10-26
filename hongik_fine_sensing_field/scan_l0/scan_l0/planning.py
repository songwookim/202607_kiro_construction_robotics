"""계획: scan_core.path_gen.build_transit_plan 래핑 + 로컬 이동 계획 (make_local_move_plan: 자동 티칭의 드웰 배치·프로브·보정·
측면 탐색·S→E 이동·패스 등 모든 ad-hoc 이동).

로컬 이동 (설계 패널 §3 "Envelope for every ad-hoc move"):
  2-웨이포인트 TransitPlan (Waypoint("now","start", T_from) → Waypoint(name, phase, T_to)) 에 기준 작업(job_ref) 의
  frames / rules / z_safe / specimen_top(올려진 판 상면) / floor 를 build_transit_plan 과 똑같이 실어 check_envelope 로 검사한다.
  phase 태그: 판 위 이동(측면·u 방향·프로브·패스) = 'scan' (laser_tilt + 판 + 바닥 + 클램프 규칙), ±n 이동 = 'descend'/'ascend'.
  위반은 EnvelopeError 로 거부하며 목표를 절대 고치지 않는다 (I4). 보정 목표는 local_target_from_correction 으로
  **현재 측정된 플랜지 포즈** 에 작은 델타(이동 + 피벗 둘레 회전)를 적용해 만든다 (I3; 모델에서 절대 목표를 다시 계산하지 않음).
  피벗 기본값은 광학 원점 — scan_core.localize.apply_correction 과 같은 규약 (Correction.d_base = 원점의 순 이동).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from scan_core.frames import is_transform, make_transform, unit
from scan_core.localize import Correction
from scan_core.path_gen import (MAX_SAMPLE_STEP_M, PHASES, TransitPlan, Waypoint, _specimen_top, build_transit_plan,
                                check_envelope, sensor_frames)
from scan_core.types import Calib, ScanJob

__all__ = ["make_plan", "make_local_move_plan", "local_target_from_correction", "seam_align_target", "LOCAL_MOVE_PHASES"]

LOCAL_MOVE_PHASES = tuple(p for p in PHASES if p not in ("start", "home"))   # 로컬 이동 구간에 허용되는 phase 태그


def make_plan(job: ScanJob, calib: Calib, T_home_flange: Optional[np.ndarray] = None, *,
              transit_speed_mps: float = 0.05, descend_speed_mps: float = 0.02, accel_mps2: float = 0.2,
              v_eff_max_mps: float = 0.05, check: bool = True) -> tuple:
    """(TransitPlan, envelope stats). 접근/이탈 속도는 v_eff_max 로 상한 클램프."""
    vt = min(float(transit_speed_mps), float(v_eff_max_mps))
    vd = min(float(descend_speed_mps), float(v_eff_max_mps))
    plan = build_transit_plan(job, calib, T_home_flange, transit_speed_mps=vt, descend_speed_mps=vd,
                              transit_accel_mps2=float(accel_mps2), check=False)
    stats = check_envelope(plan, job, calib) if check else {}
    return plan, stats


def _as_transform(T, name: str) -> np.ndarray:
    T = np.array(T, dtype=float, copy=True).reshape(4, 4)
    if not is_transform(T, tol=1e-5):
        raise ValueError(f"{name} is not a rigid 4x4 transform")
    return T


def make_local_move_plan(T_from, T_to, phase: str, job_ref: ScanJob, calib: Calib, *,
                         floor_z_m: Optional[float] = None, specimen_top_z_m: Optional[float] = None,
                         speed_mps: float, accel_mps2: float, name: str = "local",
                         sample_step_m: float = MAX_SAMPLE_STEP_M) -> tuple:
    """현재 플랜지 포즈 T_from → 목표 T_to 한 구간의 로컬 이동 계획 (TransitPlan, envelope stats). 위반 시 EnvelopeError.

    phase: 'scan' = 판 위(in-plane) 이동 (laser_tilt + 판 평면 + 바닥 + clamp_clearance 규칙 전부),
           'descend'/'ascend' = ±n 이동, 'transit' = z_safe 위 이동, 'approach_normal'/'retreat_normal' (3G ±n_h).
    job_ref: 규칙·프레임·z_safe 의 출처 (자동 티칭에서는 거친(rough) 작업). 그 specimen_top_z_m / plate_top_measured_m /
             clamp_height_m / table_z_m / check_torch_tip 과 안전 한계 (safe_clearance_m = 플랫폼 정책 여유 + 정지 거리, pose_tol_m;
             scan_l0.safety) 가 그대로 적용된다.
    specimen_top_z_m: 판 상면을 **올리기만** 하는 추가값 (max 결합; 실측 판 상면 top_k). 1G 바닥 = 이 판 상면.
    floor_z_m: 3G 광학 원점 바닥(floor_0)을 올리기만 하는 추가값. 1G 에서는 판 상면과 같은 뜻이므로 판 상면에 max 결합된다.
    모든 높이·거리는 base 3D 키포인트에서 계산되며(check_envelope, I2) 센서 원시 z 는 쓰지 않는다.
    목표는 절대 수정하지 않는다: 위반이면 EnvelopeError (I4). T_from/T_to 는 복사되어 호출자의 배열은 건드리지 않는다.
    """
    if phase not in LOCAL_MOVE_PHASES:
        raise ValueError(f"phase must be one of {LOCAL_MOVE_PHASES}, got {phase!r}")
    T0 = _as_transform(T_from, "T_from")
    T1 = _as_transform(T_to, "T_to")
    if not (math.isfinite(speed_mps) and speed_mps > 0.0 and math.isfinite(accel_mps2) and accel_mps2 > 0.0):
        raise ValueError(f"speed_mps/accel_mps2 must be > 0, got {speed_mps!r}/{accel_mps2!r}")
    fr = sensor_frames(job_ref, calib)                         # job_ref.validate() 포함 (방향 규칙 등 JobError)
    orientation = fr.orientation
    top = float(_specimen_top(job_ref, orientation))            # 올려진 판 상면 max(specimen_top_z_m, plate_top_measured_m)
    if specimen_top_z_m is not None:
        top = max(top, float(specimen_top_z_m))
    floor = fr.floor_h
    if floor_z_m is not None:
        if orientation == "vertical_3G":
            floor = max(float(floor), float(floor_z_m))
        else:
            top = max(top, float(floor_z_m))
    clearance = float(job_ref.safe_clearance_m)
    h_upper = float(max(job_ref.height(job_ref.start), job_ref.height(job_ref.end)))
    z_safe = max(top, h_upper) + clearance
    z_safe_opt = z_safe + (float(job_ref.standoff_m) if orientation == "flat_1G" else 0.0)
    rules = ["never_below_table", "never_cross_plate_plane", "transit_above", "laser_tilt", "clamp_clearance"]
    rules += ["direction_3G", "3G_floor"] if orientation == "vertical_3G" else ["direction_1G", "1G_above_specimen_top"]
    wps = [Waypoint("now", "start", T0, 0.0, 0.0, "move_l"),
           Waypoint(str(name), phase, T1, float(speed_mps), float(accel_mps2), "move_l")]
    plan = TransitPlan(waypoints=wps, frames=fr, orientation=orientation, z_safe_m=z_safe, z_safe_opt_m=z_safe_opt,
                       specimen_top_z_m=top, scan_leg_indices=[1] if phase == "scan" else [],
                       notes=[f"local move '{name}' phase={phase} from current flange pose; rules of job_ref "
                              f"{job_ref.job_id or '-'} (plate_top={top:.4f} m, clamp={float(job_ref.clamp_height_m):.3f} m)"],
                       floor_z_m=floor, reference_point=job_ref.reference_point,
                       vertical_direction=job_ref.vertical_direction, travel_sign=fr.travel_sign,
                       s_scan_from_m=fr.s_scan_from, s_scan_to_m=fr.s_scan_to, rules=rules, calib=calib)
    stats = check_envelope(plan, job_ref, calib, sample_step_m=min(float(sample_step_m), MAX_SAMPLE_STEP_M))
    return plan, stats


def local_target_from_correction(T_bf_now, d_base, roll_axis_base=None, roll_deg: float = 0.0,
                                 pivot_base=None, *, calib: Optional[Calib] = None) -> np.ndarray:
    """현재 측정 플랜지 포즈에 상대 보정을 적용한 목표 T_to (I3: 절대 재계산 없음).

    순서: (1) pivot_base 를 지나는 roll_axis_base 둘레 roll_deg 회전, (2) base 벡터 d_base 만큼 평행이동.
          T_to = Trans(d) · [Trans(pivot) · Rot(axis, roll) · Trans(−pivot)] · T_now
    → 회전 후 pivot 은 정확히 pivot + d 로, 툴의 다른 점은 pivot + d + R·(p − pivot) 으로 간다. roll_deg = 0 이면 순수 이동.

    피벗 기본값 (pivot_base=None) = **현재 포즈의 광학 원점** (calib 가 있으면 (T_now·T_flange←opt) 의 원점, 없으면 플랜지 원점).
    이 기본값이 scan_core.localize.apply_correction 과 같은 규약이다: Correction.d_base 는 광학 원점의 **순 이동**(롤 스윙 포함)
    이고 회전은 원점 둘레이므로, 결과 자세가 apply_correction(T_now, calib, corr) 과 일치한다 (1e-9).
    d_base 자리에 localize.Correction 을 넘기면 그 d_base / roll_axis_base / roll_deg 를 쓰고 피벗은 광학 원점으로 둔다
    (corr.pivot_base 는 심 점이며 스윙 회계용이다 — 그 점을 피벗으로 넘기면 스윙이 두 번 적용되므로 쓰지 않는다).
    """
    T_now = _as_transform(T_bf_now, "T_bf_now")
    if isinstance(d_base, Correction):
        corr = d_base
        if pivot_base is not None:
            raise ValueError("pivot_base must not be given with a Correction (its pivot is the optical origin, like apply_correction)")
        d_base, roll_axis_base, roll_deg = corr.d_base, corr.roll_axis_base, corr.roll_deg
    d = np.asarray(d_base, dtype=float).reshape(3)
    if not np.all(np.isfinite(d)):
        raise ValueError(f"d_base must be a finite 3-vector, got {d_base!r}")
    roll = float(roll_deg)
    if not math.isfinite(roll):
        raise ValueError(f"roll_deg must be finite, got {roll_deg!r}")
    if roll != 0.0:
        if roll_axis_base is None:
            raise ValueError("roll_axis_base is required when roll_deg != 0")
        from scipy.spatial.transform import Rotation
        axis = np.asarray(roll_axis_base, dtype=float).reshape(3)
        if not np.all(np.isfinite(axis)) or float(np.linalg.norm(axis)) < 1e-12:
            raise ValueError(f"roll_axis_base must be a finite non-zero 3-vector, got {roll_axis_base!r}")
        if pivot_base is None:
            pivot = (T_now @ calib.T_flange_opt)[:3, 3] if calib is not None else T_now[:3, 3]   # 광학 원점 (apply_correction 규약)
        else:
            pivot = np.asarray(pivot_base, dtype=float).reshape(3)
        if not np.all(np.isfinite(pivot)):
            raise ValueError(f"pivot_base must be a finite 3-vector, got {pivot_base!r}")
        R = Rotation.from_rotvec(unit(axis) * math.radians(roll)).as_matrix()
        T_rot = make_transform(R, pivot - R @ pivot)              # Trans(pivot)·Rot·Trans(−pivot)
        T_to = T_rot @ T_now
    else:
        T_to = T_now
    T_to = np.array(T_to, dtype=float, copy=True)
    T_to[:3, 3] += d
    return T_to


def seam_align_target(T_now, calib: Calib, loc, est, target_standoff_m: float, frac: float = 1.0) -> tuple:
    """S_align 목표 (P2 4.3 절): 레이저 선을 측정 심에 수직으로 돌리고 (판 법선 둘레, 그루브 점 피벗) 측면·거리·롤을 한 이동으로.

    1. G = loc.top_centre_base (지금 자세 드웰이 잰 레이저 평면 위 그루브 점).
    2. T1 = local_target_from_correction(T_now, 0, roll_axis_base=n, roll_deg=frac·δ, pivot_base=G) — G 를 지나는 n 축 회전: G 는 새 레이저
       평면 위에 남고 (x_c 그대로), 모든 툴 점의 판 거리·레이저 기울기 (Z_opt·n)·1G 높이가 불변.
    3. loc_rot = transform_localization(loc, T1·T_now⁻¹) (센서에 붙은 양만 옮긴다; G·판 평면 불변).
    4. corr = relative_correction(loc_rot, target, seam_dir=u) — 측정 심에 수직인 축 w 로만; T_to = apply_correction(T1, calib, corr).
    δ = scan_core.seam_track.yaw_to_seam_rad (|δ| < 90°), n·u = est.n·est.u (u 는 n 평면에 투영). frac: 부분 회전 (사전검사 이분 탐색; 0 =
    회전 없이 수직축 보정만). → (T_to, 기록 dict {delta_deg, applied_deg, frac, pivot, correction})."""
    from scan_core.frames import inv_transform
    from scan_core.localize import apply_correction, relative_correction, transform_localization
    from scan_core.seam_track import seam_frame, yaw_to_seam_rad
    T0 = _as_transform(T_now, "T_now")
    n, u, _w = seam_frame(est.n, est.u)
    R_opt = (T0 @ np.asarray(calib.T_flange_opt, dtype=float))[:3, :3]
    delta = float(yaw_to_seam_rad(R_opt, n, u))
    G = np.asarray(loc.top_centre_base, dtype=float).reshape(3)
    if not (np.all(np.isfinite(G)) and math.isfinite(delta)):
        raise ValueError("seam_align_target: no groove point in the dwell or no laser-line yaw")
    ang = float(frac) * delta
    T1 = local_target_from_correction(T0, np.zeros(3), roll_axis_base=n, roll_deg=math.degrees(ang), pivot_base=G)
    loc_rot = transform_localization(loc, T1 @ inv_transform(T0))
    corr = relative_correction(loc_rot, target_standoff_m=float(target_standoff_m), seam_dir=u)
    T_to = apply_correction(T1, calib, corr)
    return T_to, {"delta_deg": math.degrees(delta), "applied_deg": math.degrees(ang), "frac": float(frac), "pivot": G.tolist(),
                  "axis": n.tolist(), "correction": corr.to_dict(), "T_rot": T1}
