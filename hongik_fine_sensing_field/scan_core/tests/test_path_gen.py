import math

import numpy as np
import pytest

from scan_core.frames import angle_between, rotation_error
from scan_core.path_gen import EnvelopeError, build_transit_plan, sensor_frames, tool_keypoints, check_envelope
from scan_core.types import Calib, JobError, ScanJob


def _keypoint_min_z(T, calib, job):
    return min(float(p[2]) for p in tool_keypoints(T, calib, job).values())


def test_sensor_frames_geometry_1g(job_1g, calib_lab):
    fr = sensor_frames(job_1g, calib_lab)
    assert fr.L == pytest.approx(0.5)
    # 광학 원점이 start + u s + n standoff 에 정확히 놓인다 (피치와 무관)
    for s in (-0.04, 0.0, 0.25, 0.54):
        assert np.allclose(fr.T_base_opt(s)[:3, 3], job_1g.start + fr.u * s + fr.n * 0.32, atol=1e-12)
    # 광학 Z 는 -n 에서 |pitch| 만큼 기울어짐
    assert fr.laser_tilt_deg() == pytest.approx(5.0, abs=1e-9)
    # eoat 축: z_e = -n, x_e = +u (progression_sign=+1) — 피치 회전축 y_e 는 라인축
    R_e = fr.R_base_eoat
    assert np.allclose(R_e[:, 1], np.cross(-fr.n, fr.u), atol=1e-12)
    assert abs(math.degrees(angle_between(R_e[:, 2], -fr.n)) - 5.0) < 1e-9
    # scan frame: x = progression_sign·u, z = n
    assert np.allclose(fr.T_base_scan[:3, 0], fr.u) and np.allclose(fr.T_base_scan[:3, 2], fr.n)


def test_tool_pose_independent_of_optical_yaw_branch(job_1g):
    """profiler_optical_joint 의 ±π/2 분기(Y_opt = ±x_eoat)와 무관하게 플랜지 목표가 같다 (C10)."""
    cm = Calib.lab_initial(opt_yaw_branch=-1)
    cp = Calib.lab_initial(opt_yaw_branch=+1)
    Tm = sensor_frames(job_1g, cm).T_base_flange(0.1)
    Tp = sensor_frames(job_1g, cp).T_base_flange(0.1)
    assert np.allclose(Tm, Tp, atol=1e-12)
    # 두 분기의 광학 Y 축은 서로 반대, Z 는 같다
    Zm = (Tm @ cm.T_flange_opt)[:3, 2]
    Zp = (Tp @ cp.T_flange_opt)[:3, 2]
    Ym = (Tm @ cm.T_flange_opt)[:3, 1]
    Yp = (Tp @ cp.T_flange_opt)[:3, 1]
    assert np.allclose(Zm, Zp) and np.allclose(Ym, -Yp)


def test_progression_sign_flips_x_eoat(job_1g, calib_lab):
    fr_p = sensor_frames(job_1g, calib_lab)
    job_m = ScanJob.from_dict({**job_1g.to_dict(), "progression_sign": -1})
    fr_m = sensor_frames(job_m, calib_lab)
    assert np.allclose(fr_p.R_base_eoat[:, 0] @ fr_p.u, math.cos(math.radians(5.0)))
    assert np.allclose(fr_m.R_base_eoat[:, 0] @ fr_m.u, -math.cos(math.radians(5.0)))
    assert np.allclose(fr_p.T_base_opt(0.2)[:3, 3], fr_m.T_base_opt(0.2)[:3, 3])


def test_plan_1g_structure_and_envelope(job_1g, calib_lab):
    T_home = np.array([[1, 0, 0, 0.3], [0, -1, 0, 0.0], [0, 0, -1, 0.7], [0, 0, 0, 1.0]])
    plan = build_transit_plan(job_1g, calib_lab, T_home_flange=T_home)
    names = [w.name for w in plan.waypoints]
    assert names[:2] == ["home", "home_above"] or names[0] == "home"
    assert "above_lead_in" in names and "lead_in" in names and "lead_out" in names and "above_exit" in names
    assert plan.orientation == "flat_1G"
    assert plan.z_safe_m == pytest.approx(0.05 + 0.10)
    z_req = plan.z_safe_m
    for w in plan.waypoints:
        if w.phase == "transit":
            assert _keypoint_min_z(w.T_base_flange, calib_lab, job_1g) >= z_req
        assert _keypoint_min_z(w.T_base_flange, calib_lab, job_1g) >= job_1g.table_z_m + 0.02
    scan = plan.scan_waypoints()
    assert len(scan) == 1 and scan[0].speed_mps == job_1g.speed_mps and scan[0].s_from_m == -0.04
    assert np.allclose(scan[0].T_base_flange, plan.frames.T_base_flange(0.5 + 0.04))
    # 하강 구간은 -n 방향 (수평 이동 없음)
    i = names.index("lead_in")
    d = plan.waypoints[i].T_base_flange[:3, 3] - plan.waypoints[i - 1].T_base_flange[:3, 3]
    assert abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9 and d[2] < 0
    stats = check_envelope(plan, job_1g, calib_lab)
    assert stats["min_plate_dist_opt"] >= job_1g.standoff_min_m
    assert stats["max_laser_tilt_deg"] == pytest.approx(5.0, abs=1e-6)


def test_plan_1g_bidirectional_exits_from_start_side(job_1g, calib_lab):
    job = ScanJob.from_dict({**job_1g.to_dict(), "bidirectional": True})
    plan = build_transit_plan(job, calib_lab)
    scan = plan.scan_waypoints()
    assert len(scan) == 2 and scan[1].s_to_m == -0.04
    assert np.allclose(plan.waypoints[-1].T_base_flange[:3, :2], plan.waypoints[0].T_base_flange[:3, :2])


def test_1g_specimen_top_above_tool_tip_raises(job_1g, calib_lab):
    # 시편 상단이 스캔 자세의 토치 팁(판 위 0.32-0.168 = 0.152 m)보다 높으면 위반
    job = ScanJob.from_dict({**job_1g.to_dict(), "specimen_top_z_m": 0.05 + 0.20})
    with pytest.raises(EnvelopeError, match="specimen_top"):
        build_transit_plan(job, calib_lab)


def test_1g_table_above_tool_raises(job_1g, calib_lab):
    job = ScanJob.from_dict({**job_1g.to_dict(), "table_z_m": 0.25})
    with pytest.raises(EnvelopeError, match="table"):
        build_transit_plan(job, calib_lab)


def test_standoff_below_torch_raises(job_1g, calib_lab):
    job = ScanJob.from_dict({**job_1g.to_dict(), "standoff_m": 0.20, "standoff_min_m": 0.1})
    with pytest.raises(JobError, match="torch"):
        build_transit_plan(job, calib_lab)


def test_laser_tilt_limit(job_1g, calib_lab):
    job = ScanJob.from_dict({**job_1g.to_dict(), "pitch_deg": -25.0, "max_laser_tilt_deg": 20.0})
    with pytest.raises(JobError, match="pitch"):
        build_transit_plan(job, calib_lab)
    # 판이 1G 로는 너무 기울어진 경우 (법선 25° 기울기 + 피치 → base -Z 대비 30° 초과)
    n = np.array([math.sin(math.radians(28)), 0, math.cos(math.radians(28))])
    u = np.array([0, 1.0, 0])
    job2 = ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, 0.3, 0.05], surface_normal=n, orientation="flat_1G",
                   pitch_deg=-5.0)
    # 피치가 법선 기울기와 같은 평면에서 더해지도록 진행방향 수직 성분은 없음 → 각 ≥ 28°; 허용 30° → 통과
    build_transit_plan(job2, calib_lab)
    job3 = ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, 0.3, 0.05], surface_normal=n, orientation="flat_1G",
                   pitch_deg=-5.0, max_laser_tilt_deg=25.0)
    with pytest.raises(EnvelopeError, match="tilt"):
        build_transit_plan(job3, calib_lab)


def test_plan_3g_vertical_scan(job_3g, calib_lab):
    """3G 기본('down'): start(z 0.3) 가 아래, end(z 0.6) 가 위 → 위→아래 이동, 리드인은 위쪽에만, 리드아웃 없음."""
    plan = build_transit_plan(job_3g, calib_lab)
    assert plan.orientation == "vertical_3G"
    names = [w.name for w in plan.waypoints]
    assert names == ["above_pre_in", "pre_in", "lead_in", "scan_end", "pre_exit", "above_pre_exit"]
    assert plan.travel_sign == -1 and plan.vertical_direction == "down"
    assert plan.s_scan_from_m == pytest.approx(0.3 + 0.04) and plan.s_scan_to_m == pytest.approx(0.0)
    assert plan.floor_z_m == pytest.approx(0.3) and "3G_floor" in plan.rules and "direction_3G" in plan.rules
    n = plan.frames.n
    # transit 높이: 광학 원점 ≥ 판 상단 + clearance (기준점 = 광학 원점)
    for w in plan.waypoints:
        if w.name.startswith("above"):
            assert w.opt_origin(calib_lab)[2] >= 0.7 + 0.10 - 1e-9             # 부동소수 (등호 경계)
    # pre_in → lead_in 은 -n(수평) 방향으로 정확히 clearance 만큼 (높이 불변)
    i = names.index("lead_in")
    d = plan.waypoints[i].T_base_flange[:3, 3] - plan.waypoints[i - 1].T_base_flange[:3, 3]
    assert np.allclose(d, -n * job_3g.safe_clearance_m, atol=1e-12)
    # 리드인 지점의 광학 원점은 위 점(end) 위쪽 0.04, 스캔 종점의 광학 원점은 정확히 아래 점(start) 높이
    assert plan.waypoints[i].opt_origin(calib_lab)[2] == pytest.approx(0.6 + 0.04)
    assert plan.scan_waypoints()[0].opt_origin(calib_lab)[2] == pytest.approx(0.3)
    # 스캔 중 광학 원점은 판에서 standoff, 팁은 로봇 쪽
    stats = check_envelope(plan, job_3g, calib_lab)
    assert stats["min_plate_dist_opt"] >= job_3g.standoff_min_m
    assert stats["min_plate_dist_tip"] >= 0.05
    assert stats["floor_margin_min"] >= -1e-6
    # 광학 Z 는 수평(판 법선 방향)
    T = plan.scan_waypoints()[0].T_base_flange @ calib_lab.T_flange_opt
    assert abs(math.degrees(angle_between(T[:3, 2], -n)) - 5.0) < 1e-9
    # 실제 이동 방향(end→start = -z) 기준 x_e = progression_sign·u_travel
    assert plan.frames.R_base_eoat[:, 0] @ np.array([0, 0, -1.0]) == pytest.approx(math.cos(math.radians(5.0)))
    assert any("no lead-out" in s for s in plan.notes)


def test_3g_floor_also_keeps_the_torch_tip_above_and_takes_the_job_floor(job_3g, calib_lab):
    """2026-10-05 g3_02_dry: 토치가 레이저 아래쪽 (progression 반대) 이면 아래 점에서 토치 팁이 착점보다 ~60 mm 아래 — 3G 바닥 규칙이
    광학 원점만 봐서 놓쳤다. 이제 1G 시편 상단 규칙처럼 토치 팁도 (check_torch_tip). 바닥 = job.floor_z_m (auto-teach: 대략 아래 점
    → 실측 아래 끝) 이 있으면 그 값."""
    plan = build_transit_plan(job_3g, calib_lab)
    stats = check_envelope(plan, job_3g, calib_lab)                      # 템플릿: 토치가 레이저 위쪽 → 통과
    assert stats["min_tip_height"] > 0.3 + 0.05
    flipped = ScanJob.from_dict({**job_3g.to_dict(), "progression_sign": -job_3g.progression_sign})
    kp = tool_keypoints(sensor_frames(flipped, calib_lab).T_base_flange(0.0), calib_lab, flipped)
    assert float(kp["tip"][2]) < float(kp["opt"][2]) - 0.05                # 토치가 아래쪽
    with pytest.raises(EnvelopeError, match=r"rule=3G_floor: torch tip height .* below floor_z=0\.3000 m"):
        build_transit_plan(flipped, calib_lab)                           # 계획이 포락선을 검사한다
    low = ScanJob.from_dict({**flipped.to_dict(), "floor_z_m": 0.15})    # 실측 아래 끝이 150 mm 아래 → 토치 팁도 그 위
    plan_low = build_transit_plan(low, calib_lab)
    assert plan_low.floor_z_m == pytest.approx(0.15)
    check_envelope(plan_low, low, calib_lab)
    with pytest.raises(JobError, match="floor_z_m"):
        ScanJob.from_dict({**job_3g.to_dict(), "floor_z_m": float("nan")}).validate()


def test_plan_3g_horizontal_scan_rejected(calib_lab):
    """3G 는 수직 경로만 허용: 수평 경로는 JobError('3G requires vertical path')."""
    job = ScanJob(start=[0.3, 0.7, 0.4], end=[0.7, 0.7, 0.4], surface_normal=[0, -1, 0], specimen_top_z_m=0.7,
                  table_z_m=0.0)
    assert job.resolved_orientation() == "vertical_3G"
    with pytest.raises(JobError, match="3G requires vertical path"):
        build_transit_plan(job, calib_lab)


def test_3g_normal_away_from_robot_raises(calib_lab):
    job = ScanJob(start=[0.5, 0.7, 0.3], end=[0.5, 0.7, 0.6], surface_normal=[0, 1, 0], specimen_top_z_m=0.7)
    with pytest.raises(EnvelopeError, match="away from the robot"):
        build_transit_plan(job, calib_lab)


def test_3g_requires_specimen_top(calib_lab):
    job = ScanJob(start=[0.5, 0.7, 0.3], end=[0.5, 0.7, 0.6], surface_normal=[0, -1, 0])
    assert job.resolved_orientation() == "vertical_3G"
    with pytest.raises(EnvelopeError, match="specimen_top_z_m"):
        build_transit_plan(job, calib_lab)


def test_3g_table_violation(job_3g, calib_lab):
    # 테이블이 스캔 시작 높이(0.3 - lead)보다 높으면 위반
    job = ScanJob.from_dict({**job_3g.to_dict(), "table_z_m": 0.35})
    with pytest.raises(EnvelopeError, match="table"):
        build_transit_plan(job, calib_lab)


def test_auto_orientation_from_normal():
    assert ScanJob(start=[0, 0, 0], end=[0.1, 0, 0]).resolved_orientation() == "flat_1G"
    assert ScanJob(start=[0, 0, 0], end=[0.1, 0, 0], surface_normal=[0, -1, 0]).resolved_orientation() == "vertical_3G"
    j = ScanJob(start=[0, 0, 0], end=[0.1, 0, 0], surface_normal=[0, -1, 0], orientation="flat_1G")
    with pytest.raises(JobError):
        ScanJob(start=[0, 0, 0], end=[0.1, 0, 0], surface_normal=[0, 0, 1], orientation="vertical_3G").validate()
    assert j.resolved_orientation() == "flat_1G"


def test_job_validation_errors():
    with pytest.raises(JobError, match="< 0.05"):
        ScanJob(start=[0, 0, 0], end=[0.01, 0, 0]).validate()
    with pytest.raises(JobError, match="perpendicular"):
        ScanJob(start=[0, 0, 0], end=[0.1, 0, 0.05]).validate()
    with pytest.raises(JobError, match="progression_sign"):
        ScanJob(start=[0, 0, 0], end=[0.1, 0, 0], progression_sign=2).validate()
