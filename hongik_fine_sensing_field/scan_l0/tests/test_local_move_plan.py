"""로컬 이동 계획(make_local_move_plan) · 상대 보정 목표(local_target_from_correction) · auto_teach 설정 접근자.

모든 높이/거리는 base 3D 키포인트(FK·T_flange<-opt)에서 계산된다 (I2); 목표는 절대 수정되지 않고 위반은 EnvelopeError (I4);
보정은 현재 플랜지 포즈에 대한 작은 델타 (I3).
"""
import math
import os

import numpy as np
import pytest
from conftest import CONFIG, STACK

from scan_core.frames import inv_transform, rot_y, transform_points
from scan_core.localize import Correction, _rot_axis, apply_correction
from scan_core.path_gen import EnvelopeError, sensor_frames, tool_keypoints
from scan_core.types import Calib, JobError, ScanJob
from scan_l0 import config as C
from scan_l0.planning import LOCAL_MOVE_PHASES, local_target_from_correction, make_local_move_plan

CONFIG_KIRO = os.path.join(STACK, "hongik_scan_ros", "config", "hongik_kiro.yaml")   # KIRO 통합: kiro_right.yaml 대체 (같은 auto_teach 값)


@pytest.fixture
def calib() -> Calib:
    return Calib.lab_initial(eoat_yaw_rad=-math.pi / 2, opt_yaw_branch=-1)


@pytest.fixture
def job_ref() -> ScanJob:
    # 거친 1G 작업: 판 상면 z=0.05, 테이블 −0.05, standoff 0.40 (패스 0), 피치 0
    return ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, 0.3, 0.05], surface_normal=[0, 0, 1], standoff_m=0.40,
                   pitch_deg=0.0, table_z_m=-0.05, specimen_top_z_m=0.05, job_id="rough")


def _scan_pose(job, calib, s=0.1):
    return sensor_frames(job, calib).T_base_flange(s)


def _shifted(T, d):
    T2 = np.array(T, dtype=float, copy=True)
    T2[:3, 3] += np.asarray(d, dtype=float)
    return T2


# --------------------------------------------------------------------------------------
# make_local_move_plan
# --------------------------------------------------------------------------------------
def test_local_scan_move_ok_carries_job_rules(job_ref, calib):
    T0 = _scan_pose(job_ref, calib)
    T1 = _shifted(T0, [0.02, 0.0, 0.0])                    # 측면 20 mm (판 위)
    plan, stats = make_local_move_plan(T0, T1, "scan", job_ref, calib, speed_mps=0.02, accel_mps2=0.2, name="lateral")
    assert [w.name for w in plan.waypoints] == ["now", "lateral"]
    assert [w.phase for w in plan.waypoints] == ["start", "scan"]
    assert plan.scan_leg_indices == [1] and plan.orientation == "flat_1G"
    assert set(plan.rules) >= {"never_below_table", "never_cross_plate_plane", "transit_above", "laser_tilt",
                               "clamp_clearance", "direction_1G", "1G_above_specimen_top"}
    assert plan.specimen_top_z_m == pytest.approx(0.05)
    assert plan.z_safe_m == pytest.approx(0.05 + 0.10) and plan.z_safe_opt_m == pytest.approx(0.05 + 0.10 + 0.40)
    assert plan.floor_z_m is None and plan.calib is calib
    assert stats["n_samples"] >= 5 and stats["sample_step_m"] <= 0.005 and stats["max_laser_tilt_deg"] == pytest.approx(0.0, abs=1e-6)
    assert stats["min_opt_height"] == pytest.approx(0.45, abs=1e-9)
    np.testing.assert_allclose(plan.waypoints[1].T_base_flange, T1)
    assert plan.waypoints[1].speed_mps == 0.02 and plan.waypoints[1].motion == "move_l"


def test_local_move_refuses_target_below_floor_and_never_modifies_target(job_ref, calib):
    T0 = _scan_pose(job_ref, calib)
    # 100 mm 하강: 광학 원점 0.35 (≥ standoff_min), 팁 0.182 ≥ 판 0.05 → 거친 작업으로는 통과
    T_down = _shifted(T0, [0.0, 0.0, -0.10])
    make_local_move_plan(T0, T_down, "descend", job_ref, calib, speed_mps=0.02, accel_mps2=0.2)
    # 실측 판 상면 0.20 (specimen_top_z_m 인자로 올림) → 팁이 바닥 아래 → 거부, 목표 배열은 그대로
    T_keep = T_down.copy()
    with pytest.raises(EnvelopeError, match=r"rule=1G_above_specimen_top: torch tip height .* below specimen_top_z=0.2000"):
        make_local_move_plan(T0, T_down, "descend", job_ref, calib, specimen_top_z_m=0.20, speed_mps=0.02, accel_mps2=0.2)
    np.testing.assert_array_equal(T_down, T_keep)
    # 1G: floor_z_m 도 판 상면을 올리는 뜻 → 같은 거부
    with pytest.raises(EnvelopeError, match="rule=1G_above_specimen_top"):
        make_local_move_plan(T0, T_down, "descend", job_ref, calib, floor_z_m=0.20, speed_mps=0.02, accel_mps2=0.2)
    # job_ref 의 plate_top_measured_m 도 같은 효과 (실측 바닥은 모든 키포인트의 하한 → never_below_table 이 먼저 잡는다)
    job_meas = ScanJob.from_dict({**job_ref.to_dict(), "plate_top_measured_m": 0.20})
    with pytest.raises(EnvelopeError, match=r"rule=never_below_table: .*measured plate top=0.2000"):
        make_local_move_plan(T0, T_down, "descend", job_meas, calib, speed_mps=0.02, accel_mps2=0.2)
    # 판 쪽으로 너무 가까운 목표 (300 mm 하강: 광학 원점 판거리 0.10 < standoff_min 0.245, 팁은 테이블 + 한계 아래) → 거부
    with pytest.raises(EnvelopeError, match="rule=never_(cross_plate_plane|below_table)"):
        make_local_move_plan(T0, _shifted(T0, [0.0, 0.0, -0.30]), "descend", job_ref, calib, speed_mps=0.02, accel_mps2=0.2)
    # 테이블 아래까지 (350 mm): 어떤 규칙이든 거부 (여기서는 never_below_table 이 먼저)
    with pytest.raises(EnvelopeError, match="rule=never_below_table"):
        make_local_move_plan(T0, _shifted(T0, [0.0, 0.0, -0.35]), "descend", job_ref, calib, speed_mps=0.02, accel_mps2=0.2)


def test_local_move_clamp_rule_applies(job_ref, calib):
    T0 = _scan_pose(job_ref, calib)
    T1 = _shifted(T0, [0.0, 0.01, 0.0])
    job_clamp = ScanJob.from_dict({**job_ref.to_dict(), "clamp_height_m": 0.25})   # 팁 0.05+0.232 < 0.30
    with pytest.raises(EnvelopeError, match=r"rule=clamp_clearance: torch tip height .*\(torch keypoint\)"):
        make_local_move_plan(T0, T1, "scan", job_clamp, calib, speed_mps=0.02, accel_mps2=0.2)
    job_ok = ScanJob.from_dict({**job_ref.to_dict(), "clamp_height_m": 0.20})
    _, st = make_local_move_plan(T0, T1, "scan", job_ok, calib, speed_mps=0.02, accel_mps2=0.2)
    assert st["clamp_margin_min"] == pytest.approx(0.40 - 0.168 - 0.20, abs=1e-6)


def test_local_scan_move_refuses_laser_tilt_over_max(job_ref, calib):
    T0 = _scan_pose(job_ref, calib)
    # 라인축(y_eoat) 둘레 35° 회전한 목표 → 광학 Z vs −n = 35° > max_laser_tilt_deg 30
    T1 = np.array(T0, dtype=float, copy=True)
    T1[:3, :3] = T0[:3, :3] @ calib.T_flange_eoat[:3, :3] @ rot_y(math.radians(35.0)) @ calib.T_flange_eoat[:3, :3].T
    with pytest.raises(EnvelopeError, match=r"\[probe/scan\] rule=laser_tilt"):
        make_local_move_plan(T0, T1, "scan", job_ref, calib, speed_mps=0.02, accel_mps2=0.2, name="probe")
    # 20° 는 허용 (max 30)
    T2 = np.array(T0, dtype=float, copy=True)
    T2[:3, :3] = T0[:3, :3] @ calib.T_flange_eoat[:3, :3] @ rot_y(math.radians(20.0)) @ calib.T_flange_eoat[:3, :3].T
    _, st = make_local_move_plan(T0, T2, "scan", job_ref, calib, speed_mps=0.02, accel_mps2=0.2)
    assert st["max_laser_tilt_deg"] == pytest.approx(20.0, abs=1e-6)


def test_local_move_transit_phase_and_ascend(job_ref, calib):
    T0 = _scan_pose(job_ref, calib)
    T_up = _shifted(T0, [0.0, 0.0, 0.10])                       # 광학 원점 0.55 = z_safe_opt
    plan, _ = make_local_move_plan(T0, T_up, "ascend", job_ref, calib, speed_mps=0.02, accel_mps2=0.2)
    assert plan.waypoints[1].phase == "ascend"
    make_local_move_plan(T_up, _shifted(T_up, [0.05, 0.0, 0.0]), "transit", job_ref, calib, speed_mps=0.05, accel_mps2=0.2)
    # transit 구간이 z_safe_opt 아래이면 거부
    with pytest.raises(EnvelopeError, match="rule=transit_above"):
        make_local_move_plan(T0, _shifted(T0, [0.05, 0.0, 0.0]), "transit", job_ref, calib, speed_mps=0.05, accel_mps2=0.2)


def test_local_move_argument_validation(job_ref, calib):
    T0 = _scan_pose(job_ref, calib)
    assert "start" not in LOCAL_MOVE_PHASES and "home" not in LOCAL_MOVE_PHASES and "scan" in LOCAL_MOVE_PHASES
    with pytest.raises(ValueError, match="phase"):
        make_local_move_plan(T0, T0, "start", job_ref, calib, speed_mps=0.02, accel_mps2=0.2)
    with pytest.raises(ValueError, match="rigid"):
        make_local_move_plan(T0, np.eye(4) * 2.0, "scan", job_ref, calib, speed_mps=0.02, accel_mps2=0.2)
    with pytest.raises(ValueError, match="speed_mps"):
        make_local_move_plan(T0, T0, "scan", job_ref, calib, speed_mps=0.0, accel_mps2=0.2)
    # 기준 작업 자체가 방향 규칙을 어기면 JobError (sensor_frames 의 validate)
    bad = ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, 0.3, 0.35], surface_normal=[0, -1, 1], orientation="flat_1G")
    with pytest.raises(JobError):
        make_local_move_plan(T0, T0, "scan", bad, calib, speed_mps=0.02, accel_mps2=0.2)


def test_local_move_3g_floor_raise(calib):
    job3 = ScanJob(start=[0.5, 0.7, 0.3], end=[0.5, 0.7, 0.6], surface_normal=[0, -1, 0], specimen_top_z_m=0.7, table_z_m=0.0)
    fr = sensor_frames(job3, calib)
    T0 = fr.T_base_flange(0.15)
    T1 = fr.T_base_flange(0.05)                                 # 아래 점 쪽으로 100 mm (바닥 0.3 위)
    plan, st = make_local_move_plan(T0, T1, "scan", job3, calib, speed_mps=0.02, accel_mps2=0.2)
    assert plan.floor_z_m == pytest.approx(0.3) and st["floor_margin_min"] >= -1e-6
    # 바닥을 0.40 으로 올리면(floor_0 잠금 후 상향) 광학 원점 0.35 < 0.40 → 3G_floor 거부
    with pytest.raises(EnvelopeError, match="rule=3G_floor"):
        make_local_move_plan(T0, T1, "scan", job3, calib, floor_z_m=0.40, speed_mps=0.02, accel_mps2=0.2)
    # 내리는 값은 무시
    plan2, _ = make_local_move_plan(T0, T1, "scan", job3, calib, floor_z_m=0.10, speed_mps=0.02, accel_mps2=0.2)
    assert plan2.floor_z_m == pytest.approx(0.3)


def test_first_leg_may_start_below_a_floor_by_the_arrival_error_but_never_go_lower(calib):
    """첫 구간 (4d 검토): 출발 자세가 도착 오차로 바닥 아래 (도착 허용 안) 면 그 높이까지는 되지만 출발보다 더 낮게는 안 된다 — 예전에는
    첫 구간 전체가 도착 허용 (2 mm) 을 받아 바닥 옆 제자리 회전이 출발보다 낮게 한계 아래를 스쳤다."""
    job3 = ScanJob(start=[0.5, 0.7, 0.3], end=[0.5, 0.7, 0.6], surface_normal=[0, -1, 0], specimen_top_z_m=0.7, table_z_m=0.0,
                   pose_tol_m=0.002)
    T = sensor_frames(job3, calib).T_base_flange(0.05)
    kp = tool_keypoints(T, calib, job3)
    names = ("opt", "tip") if job3.check_torch_tip else ("opt",)
    low = min(float(kp[k][2]) for k in names)
    T0 = _shifted(T, [0.0, 0.0, (0.3 - 0.001) - low])                 # 가장 낮은 바닥 키포인트가 바닥 1 mm 아래 (도착 오차)
    make_local_move_plan(T0, _shifted(T0, [0.01, 0.0, 0.0]), "scan", job3, calib, speed_mps=0.02, accel_mps2=0.2)   # 같은 높이: 됨
    make_local_move_plan(T0, _shifted(T0, [0.0, 0.0, 0.002]), "scan", job3, calib, speed_mps=0.02, accel_mps2=0.2)  # 위로: 됨
    with pytest.raises(EnvelopeError, match="rule=3G_floor"):
        make_local_move_plan(T0, _shifted(T0, [0.0, 0.0, -0.0005]), "scan", job3, calib, speed_mps=0.02, accel_mps2=0.2)


# --------------------------------------------------------------------------------------
# local_target_from_correction (I3: 현재 포즈 + 델타)
# --------------------------------------------------------------------------------------
def test_local_target_pure_translation(job_ref, calib):
    T_now = _scan_pose(job_ref, calib)
    d = np.array([0.012, -0.007, 0.003])
    T_to = local_target_from_correction(T_now, d)
    np.testing.assert_allclose(T_to[:3, :3], T_now[:3, :3], atol=1e-15)
    np.testing.assert_allclose(T_to[:3, 3] - T_now[:3, 3], d, atol=1e-15)
    # FK: 모든 키포인트가 정확히 d 만큼 이동
    k0, k1 = tool_keypoints(T_now, calib, job_ref), tool_keypoints(T_to, calib, job_ref)
    for k in k0:
        np.testing.assert_allclose(k1[k] - k0[k], d, atol=1e-12)


def test_local_target_roll_about_pivot_then_translate(job_ref, calib):
    T_now = _scan_pose(job_ref, calib)
    T_bo = T_now @ calib.T_flange_opt
    X_opt = T_bo[:3, 0]
    seam = job_ref.start + job_ref.unit_dir * 0.1                # 광학 Z 가 향하는 심 점 (피벗)
    d = np.array([0.0, 0.004, 0.0])
    roll = 3.0
    T_to = local_target_from_correction(T_now, d, roll_axis_base=X_opt, roll_deg=roll, pivot_base=seam)
    # 회전: R_to = R(X_opt, 3°) · R_now
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(X_opt / np.linalg.norm(X_opt) * math.radians(roll)).as_matrix()
    np.testing.assert_allclose(T_to[:3, :3], R @ T_now[:3, :3], atol=1e-12)
    # 피벗은 정확히 d 만큼만 이동 (플랜지 좌표로 표현한 피벗을 새 포즈로 변환)
    p_local = transform_points(inv_transform(T_now), seam)
    np.testing.assert_allclose(transform_points(T_to, p_local), seam + d, atol=1e-12)
    # 광학 원점(FK): pivot + d + R (o_now − pivot); 3° 회전 × 0.40 m 레버 ≈ 21 mm 이동
    o_now = tool_keypoints(T_now, calib, job_ref)["opt"]
    o_to = tool_keypoints(T_to, calib, job_ref)["opt"]
    np.testing.assert_allclose(o_to, seam + d + R @ (o_now - seam), atol=1e-12)
    assert np.linalg.norm(o_to - o_now - d) == pytest.approx(2 * 0.40 * math.sin(math.radians(roll) / 2), abs=1e-9)
    # 회전 후 광학 Z 는 여전히 피벗을 향한다 (심 점 중심 호 이동)
    Z_to = (T_to @ calib.T_flange_opt)[:3, 2]
    assert np.linalg.norm(np.cross(Z_to, seam + d - o_to)) < 1e-9
    # 입력 배열은 불변
    T_copy = _scan_pose(job_ref, calib)
    np.testing.assert_array_equal(T_now, T_copy)
    # 델타가 포락선 검사를 통과하는 로컬 이동으로 이어진다
    make_local_move_plan(T_now, T_to, "scan", job_ref, calib, speed_mps=0.02, accel_mps2=0.2, name="correct")


def test_local_target_default_pivot_is_optical_origin_and_matches_apply_correction(job_ref, calib):
    """Minor-5: with no pivot the rotation is about the optical origin of T_now — exactly localize.apply_correction's
    convention (Correction.d_base = net motion of the optical origin, roll about it). Both call forms agree to 1e-9;
    feeding the seam pivot (corr.pivot_base) instead would double-apply the roll swing, which is what the check documents."""
    T_now = _scan_pose(job_ref, calib)
    T_bo = T_now @ calib.T_flange_opt
    o_now, Y_opt = T_bo[:3, 3].copy(), T_bo[:3, 1].copy()
    seam = job_ref.start + job_ref.unit_dir * 0.1
    corr = Correction(d_base=np.array([0.004, -0.003, 0.002]), roll_axis_base=Y_opt, roll_deg=3.0, pivot_base=seam)
    T_ref = apply_correction(T_now, calib, corr)
    # (a) explicit pieces, default pivot (calib -> optical origin)
    T_a = local_target_from_correction(T_now, corr.d_base, corr.roll_axis_base, corr.roll_deg, calib=calib)
    np.testing.assert_allclose(T_a, T_ref, atol=1e-9)
    # (b) the Correction itself
    T_b = local_target_from_correction(T_now, corr, calib=calib)
    np.testing.assert_allclose(T_b, T_ref, atol=1e-9)
    # (c) explicit pivot = optical origin: identical
    T_c = local_target_from_correction(T_now, corr.d_base, corr.roll_axis_base, corr.roll_deg, pivot_base=o_now)
    np.testing.assert_allclose(T_c, T_ref, atol=1e-9)
    # the optical origin moved by exactly d_base and the rotation is Rot(Y_opt, 3 deg) . R_now
    o_to = (T_a @ calib.T_flange_opt)[:3, 3]
    np.testing.assert_allclose(o_to - o_now, corr.d_base, atol=1e-12)
    np.testing.assert_allclose(T_a[:3, :3], _rot_axis(Y_opt, math.radians(3.0)) @ T_now[:3, :3], atol=1e-12)
    # (d) seam pivot with Correction.d_base is NOT the same pose: it moves the origin by the swing (Rot - I)(o - p) on top
    T_d = local_target_from_correction(T_now, corr.d_base, corr.roll_axis_base, corr.roll_deg, pivot_base=seam)
    o_d = (T_d @ calib.T_flange_opt)[:3, 3]
    swing = (_rot_axis(Y_opt, math.radians(3.0)) - np.eye(3)) @ (o_now - seam)
    np.testing.assert_allclose(o_d - o_to, swing, atol=1e-9)
    assert np.linalg.norm(swing) > 5e-3                          # ~ 0.4 m * 3 deg: a real, not numerical, difference
    with pytest.raises(ValueError, match="pivot_base must not be given with a Correction"):
        local_target_from_correction(T_now, corr, pivot_base=seam, calib=calib)
    # no calib and no pivot: flange origin is the pivot (documented fallback)
    T_e = local_target_from_correction(T_now, corr.d_base, corr.roll_axis_base, corr.roll_deg)
    np.testing.assert_allclose(T_e[:3, 3] - T_now[:3, 3], corr.d_base, atol=1e-12)
    # zero roll: pure translation regardless of pivot / calib
    T_z = local_target_from_correction(T_now, Correction(d_base=corr.d_base, roll_axis_base=Y_opt, roll_deg=0.0, pivot_base=seam))
    np.testing.assert_allclose(T_z[:3, :3], T_now[:3, :3], atol=1e-15)
    np.testing.assert_allclose(T_z[:3, 3] - T_now[:3, 3], corr.d_base, atol=1e-15)


def test_local_target_validation(job_ref, calib):
    T_now = _scan_pose(job_ref, calib)
    with pytest.raises(ValueError, match="roll_axis_base is required"):
        local_target_from_correction(T_now, [0, 0, 0], roll_deg=1.0)
    with pytest.raises(ValueError, match="pivot_base"):
        local_target_from_correction(T_now, [0, 0, 0], roll_axis_base=[0, 1, 0], roll_deg=1.0, pivot_base=[0, float("nan"), 0])
    with pytest.raises(ValueError, match="roll_axis_base"):
        local_target_from_correction(T_now, [0, 0, 0], roll_axis_base=[0, 0, 0], roll_deg=1.0, pivot_base=[0, 0, 0])
    with pytest.raises(ValueError, match="d_base"):
        local_target_from_correction(T_now, [0, float("nan"), 0])
    with pytest.raises(ValueError, match="rigid"):
        local_target_from_correction(np.zeros((4, 4)), [0, 0, 0])


# --------------------------------------------------------------------------------------
# auto_teach 설정
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [CONFIG, CONFIG_KIRO])
def test_auto_teach_config_typed_accessors(path):
    s = C.load_settings(path)
    at = s.auto_teach_config()
    assert at.dwell_frames == 50                                     # 거리·측면 탐색은 설정이 아니라 유도 (scan_l0.sensor_view)
    assert at.max_corrections == 3                                   # 2026-10-04 2 → 3 (판 롤 12°)
    assert at.max_passes == (3 if path == CONFIG_KIRO else 4)         # KIRO 1G: 패스 예산 3 (사용자 2026-10-11 밤: 전체 패스 ~3 번)
    # 2026-10-01: 착점 오버스캔·끝 연장·패스 1 연장 비율·충실도 게이트·판 끝 식별 — 모두 설정값 (두 플랫폼 키 동일)
    assert at.accept_inferred_end is False                            # 오버스캔·끝 연장은 유도 (autoteach.overscan_need_m, 설계 길이)
    assert not hasattr(at, "fidelity") and s.judgement_k == 3.0            # 판정 기준은 유도 (scan_core.measure), 명시값은 k 하나
    assert not hasattr(at, "consistency") and not hasattr(at, "tol_fine") and not hasattr(at, "end_detection")   # 판 끝 식별도 유도
    assert at.first_frame_timeout_s == 2.0 and not hasattr(at, "guard")   # 실시간 센서 가드 없음: 센서 시작 대기만
    assert s.sensor["ghost_burst_frames"] == 2 and s.safety["clearance_m"] == 0.05
    assert not hasattr(at, "clamp_height_m")                          # 클램프는 작업 파일의 고정구 기하 (2026-10-08 플랫폼 키 은퇴)
    assert not hasattr(at, "tol_coarse") and not hasattr(at, "min_site_distance_m")   # 드웰 수렴·S·E 거리는 유도 (4d P3/P4)
    d = at.to_dict()
    assert "tolerances" not in d and "guard" not in d
    assert C.AutoTeachConfig.from_params(d) == at                     # dict 왕복


def test_auto_teach_config_null_handling_and_defaults():
    at = C.AutoTeachConfig.from_params({"first_frame_timeout_s": None})
    assert at.first_frame_timeout_s == 2.0
    with pytest.raises(ValueError, match="guard is no longer a setting"):          # 이 패키지에는 실시간 센서 가드가 없다
        C.AutoTeachConfig.from_params({"guard": {"first_frame_timeout_s": 2.0}})
    with pytest.raises(ValueError, match="tolerances is no longer a setting"):
        C.AutoTeachConfig.from_params({"tolerances": {"coarse": {"psi_deg": 0.3}}})
    with pytest.raises(ValueError, match="fidelity is no longer a setting"):
        C.AutoTeachConfig.from_params({"fidelity": {"measurement_noise_max_mm": 0.2}})
    for k in C.AUTO_TEACH_RETIRED_KEYS:                              # 유도로 바뀐 옛 키는 조용히 무시하지 않고 거부
        with pytest.raises(ValueError, match=k):
            C.AutoTeachConfig.from_params({k: 0.4})
    assert C.Settings({}).auto_teach == {}
    with pytest.raises(ValueError, match="job file"):
        C.AutoTeachConfig.from_params({"clamp_height_m": 0.0})


def test_job_from_settings_clamp_comes_only_from_the_job(tmp_path):
    s = C.load_settings(CONFIG)
    G = [0.0, 0.0, -1.0]                   # MoveIt 플랫폼 설정의 중력은 TF 유도 (null) — 작업에는 명시
    job = C.job_from_settings(s, [0.6, -0.2, 0.05], [0.6, 0.3, 0.05], [0, 0, 1], specimen_top_z_m=0.05,
                             gravity_axis_base=G)
    assert job.clamp_height_m == 0.0 and job.plate_top_measured_m is None
    # 클램프 = 시편 고정구 기하: 작업에 명시 (플랫폼 설정 키는 은퇴 — 2026-10-08)
    j2 = C.job_from_settings(s, [0.6, -0.2, 0.05], [0.6, 0.3, 0.05], [0, 0, 1], clamp_height_m=0.03, gravity_axis_base=G)
    assert j2.clamp_height_m == 0.03
    # 작업 파일 왕복에 새 필드 포함
    pj = C.save_job(tmp_path / "j.yaml", ScanJob.from_dict({**j2.to_dict(), "plate_top_measured_m": 0.052}))
    j4, doc = C.load_job(pj)
    assert j4.clamp_height_m == 0.03 and j4.plate_top_measured_m == pytest.approx(0.052)
    assert doc["job"]["plate_top_measured_m"] == pytest.approx(0.052)
