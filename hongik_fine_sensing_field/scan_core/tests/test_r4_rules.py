"""R4 규칙 테스트: 기준점 = 광학 원점, 1G 수평/3G 수직 방향 규칙, 3G 바닥, transit 위쪽 규칙, 토치 키포인트 플래그,
감속 구간 프레임(valid=True, in_cruise=False)."""
import math

import numpy as np
import pytest

from scan_core.frames import interpolate_transforms, rotation_error
from scan_core.path_gen import (EnvelopeError, build_transit_plan, check_envelope, expected_cruise_s_range, ramp_zones_m,
                                scan_s_range, sensor_frames, tool_keypoints)
from scan_core.reconstruct import reconstruct_scan
from scan_core.synth import PlaneScene, SynthConfig, simulate_scan
from scan_core.types import Calib, JobError, ScanJob


def _opt_trajectory(plan, calib, step=0.002, skip_home=True):
    """계획의 모든 구간을 step 으로 보간한 광학 원점 궤적 (N,3) — 바닥 규칙 독립 검증용."""
    pts = []
    wps = plan.waypoints
    for i in range(1, len(wps)):
        if skip_home and not wps[i].floor_rule:
            continue
        T0, T1 = wps[i - 1].T_base_flange, wps[i].T_base_flange
        span = max(float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])), rotation_error(T0[:3, :3], T1[:3, :3]) * 0.5)
        Ts = interpolate_transforms(T0, T1, max(2, int(math.ceil(span / step)) + 1))
        T_be = Ts @ calib.T_flange_eoat
        pts.append(np.einsum("nij,j->ni", T_be[:, :3, :3], calib.p_opt_eoat) + T_be[:, :3, 3])
    return np.concatenate(pts)


def _tilted_job_1g(tilt_deg: float, **kw) -> ScanJob:
    """진행방향이 수평에서 tilt_deg 만큼 기울고 법선은 그에 수직인 1G 작업 (법선 수직성은 유지)."""
    t = math.radians(tilt_deg)
    u = np.array([0.0, math.cos(t), math.sin(t)])
    n = np.array([0.0, -math.sin(t), math.cos(t)])
    start = np.array([0.6, -0.2, 0.05])
    d = dict(start=start, end=start + 0.5 * u, surface_normal=n, orientation="flat_1G", table_z_m=-0.05)
    d.update(kw)
    return ScanJob(**d)


# --------------------------------------------------------------------------------------
# ScanJob 필드 · 방향 규칙
# --------------------------------------------------------------------------------------
def test_scanjob_new_fields_defaults_and_roundtrip(job_1g):
    assert np.allclose(job_1g.gravity_axis_base, [0, 0, -1])
    assert job_1g.vertical_direction == "down" and job_1g.reference_point == "optical"
    assert job_1g.direction_tol_deg == 20.0 and job_1g.check_torch_tip is True     # 자세 입력 확인 (2026-10-04 5 → 20)
    d = job_1g.to_dict()
    assert d["gravity_axis_base"] == [0.0, 0.0, -1.0]
    j2 = ScanJob.from_dict(d)
    assert np.allclose(j2.gravity_axis_base, job_1g.gravity_axis_base) and j2.vertical_direction == "down"
    assert np.allclose(job_1g.height([0.1, 0.2, 0.3]), 0.3)
    with pytest.raises(JobError, match="vertical_direction"):
        ScanJob.from_dict({**d, "vertical_direction": "sideways"}).validate()
    with pytest.raises(JobError, match="reference_point"):
        ScanJob.from_dict({**d, "reference_point": "flange"}).validate()
    with pytest.raises(JobError, match="gravity_axis_base"):
        ScanJob.from_dict({**d, "gravity_axis_base": [0.0, 0.0]})


def test_1g_horizontal_ok_and_tilted_rejected(calib_lab):
    build_transit_plan(_tilted_job_1g(0.0), calib_lab)
    build_transit_plan(_tilted_job_1g(10.0), calib_lab)                     # 허용 20° 이내 (시편은 대체로 수평일 뿐, 2026-10-04)
    job = _tilted_job_1g(30.0)                                               # 자세 입력 오류급
    assert job.direction_angles_deg()["horizontal_dev_deg"] == pytest.approx(30.0)
    with pytest.raises(JobError, match="1G requires horizontal path"):
        job.validate()
    with pytest.raises(JobError, match="1G requires horizontal path"):
        build_transit_plan(job, calib_lab)
    # 허용각을 넓히면 통과
    build_transit_plan(_tilted_job_1g(10.0, direction_tol_deg=15.0), calib_lab)


def test_3g_vertical_ok_and_horizontal_rejected(job_3g, calib_lab):
    build_transit_plan(job_3g, calib_lab)
    # 4° 기울어진 수직 경로 (법선은 여전히 수평·수직에 가까움) → 허용
    t = math.radians(4.0)
    u = np.array([math.sin(t), 0.0, math.cos(t)])
    j_ok = ScanJob(start=[0.5, 0.7, 0.3], end=np.array([0.5, 0.7, 0.3]) + 0.3 * u, surface_normal=[0, -1, 0],
                   specimen_top_z_m=0.7, table_z_m=0.0)
    build_transit_plan(j_ok, calib_lab)
    j_bad = ScanJob(start=[0.3, 0.7, 0.4], end=[0.7, 0.7, 0.4], surface_normal=[0, -1, 0], specimen_top_z_m=0.7)
    assert j_bad.direction_angles_deg()["vertical_dev_deg"] == pytest.approx(90.0)
    with pytest.raises(JobError, match="3G requires vertical path"):
        j_bad.validate()
    # 45° 경사 경로: 1G 로도 3G 로도 거부
    j_45 = ScanJob(start=[0.5, 0.7, 0.3], end=[0.5, 0.9, 0.5], surface_normal=[0, -1, 1], orientation="vertical_3G",
                   specimen_top_z_m=0.7)
    with pytest.raises(JobError, match="3G requires vertical path"):
        j_45.validate()
    j_45b = ScanJob(start=[0.5, 0.7, 0.3], end=[0.5, 0.9, 0.5], surface_normal=[0, -1, 1], orientation="flat_1G")
    with pytest.raises(JobError, match="1G requires horizontal path"):
        j_45b.validate()


def test_gravity_axis_parameter_changes_orientation_and_heights():
    # base y 축이 아래쪽인 좌표계: 법선 +y 는 '위' → flat_1G, 경로는 x 방향(수평)
    j = ScanJob(start=[0.0, 0.5, 0.0], end=[0.3, 0.5, 0.0], surface_normal=[0, -1, 0], gravity_axis_base=[0, 1, 0])
    assert j.resolved_orientation() == "flat_1G"
    j.validate()
    assert j.height([0.0, -0.2, 0.0]) == pytest.approx(0.2)
    # 같은 기하를 기본 중력축으로 보면 3G(법선 수평) 인데 경로가 수평이라 거부
    j2 = ScanJob(start=[0.0, 0.5, 0.0], end=[0.3, 0.5, 0.0], surface_normal=[0, -1, 0])
    assert j2.resolved_orientation() == "vertical_3G"
    with pytest.raises(JobError, match="3G requires vertical path"):
        j2.validate()


# --------------------------------------------------------------------------------------
# 3G 바닥 규칙 (광학 원점) — 양방향, 접근/후퇴/보간 샘플 포함
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", ["down", "up"])
@pytest.mark.parametrize("start_upper", [False, True])
def test_3g_optical_origin_never_below_floor(job_3g, calib_lab, direction, start_upper):
    d = job_3g.to_dict()
    if start_upper:
        d["start"], d["end"] = d["end"], d["start"]
    d["vertical_direction"] = direction

    def tip_up(sign):                              # 3G 바닥은 토치 팁도 본다 (2026-10-05) → 토치가 레이저 위쪽인 진행 부호
        j = ScanJob.from_dict({**d, "progression_sign": sign})
        kp = tool_keypoints(sensor_frames(j, calib_lab).T_base_flange(0.0), calib_lab, j)
        return float(j.height(kp["tip"]) - j.height(kp["opt"]))
    d["progression_sign"] = max((1, -1), key=tip_up)
    job = ScanJob.from_dict(d)
    T_home = np.array([[1, 0, 0, 0.3], [0, -1, 0, 0.0], [0, 0, -1, 0.9], [0, 0, 0, 1.0]])
    plan = build_transit_plan(job, calib_lab, T_home_flange=T_home)
    floor = min(job.start[2], job.end[2])
    assert plan.floor_z_m == pytest.approx(floor) == pytest.approx(0.3)
    # (a) check_envelope 통계
    stats = check_envelope(plan, job, calib_lab)
    assert stats["floor_margin_min"] >= -1e-6
    assert stats["sample_step_m"] <= 0.005
    # (b) 독립 보간(2 mm) 광학 원점 궤적: home 왕복 제외 전 구간에서 z ≥ floor
    O = _opt_trajectory(plan, calib_lab, step=0.002)
    assert O[:, 2].min() >= floor - 1e-6
    # (c) 스캔 구간의 s 범위: 아래 점에서 리드 없음, 위 점 바깥쪽에만 리드
    s_from, s_to, sign = scan_s_range(job)
    lo_s, hi_s = sorted((s_from, s_to))
    s_bot = 0.0 if not start_upper else job.length_m          # 아래 점의 s
    s_top = job.length_m if not start_upper else 0.0
    assert min(abs(lo_s - s_bot), abs(hi_s - s_bot)) < 1e-12                # 한쪽 끝은 정확히 아래 점
    other = hi_s if abs(lo_s - s_bot) < 1e-12 else lo_s
    assert abs(other - s_top) == pytest.approx(0.04)                          # 다른 끝은 위 점 + 리드 (위 점 바깥쪽)
    if direction == "down":
        assert plan.frames.opt_origin(s_from)[2] > plan.frames.opt_origin(s_to)[2]   # 위 → 아래
    else:
        assert plan.frames.opt_origin(s_from)[2] < plan.frames.opt_origin(s_to)[2]   # 아래 → 위
    # (d) 접근/후퇴 구간은 높이 불변 (판 법선 수평 성분)
    names = [w.name for w in plan.waypoints]
    for a, b in (("pre_in", names[names.index("pre_in") + 1]), (names[names.index("pre_exit") - 1], "pre_exit")):
        Ta = plan.waypoints[names.index(a)].T_base_flange
        Tb = plan.waypoints[names.index(b)].T_base_flange
        assert abs(Ta[2, 3] - Tb[2, 3]) < 1e-12
    # (e) 스캔 종점/시점의 광학 원점은 정확히 아래 점 높이 (그 아래로 절대 내려가지 않음)
    scan_wp = plan.scan_waypoints()[0]
    o_end = scan_wp.opt_origin(calib_lab)
    o_start = plan.waypoints[plan.scan_leg_indices[0] - 1].opt_origin(calib_lab)
    assert min(o_end[2], o_start[2]) == pytest.approx(floor)


def test_3g_floor_violation_detected_in_scan_and_retreat(job_3g, calib_lab):
    """법선이 아래로 4° 기울면(경로는 허용각 내 수직) standoff 만큼 떨어진 광학 원점이 아래 점보다 낮아진다 → rule=3G_floor."""
    t = math.radians(4.0)
    n = np.array([0.0, -math.cos(t), -math.sin(t)])          # 로봇 쪽(-y) + 약간 아래
    u = np.array([0.0, -math.sin(t), math.cos(t)])            # n 에 수직, 거의 수직
    d = job_3g.to_dict()
    d.update(start=[0.5, 0.7, 0.3], end=(np.array([0.5, 0.7, 0.3]) + 0.3 * u).tolist(), surface_normal=n.tolist())
    job = ScanJob.from_dict(d)
    with pytest.raises(EnvelopeError, match="rule=3G_floor.*optical origin"):
        build_transit_plan(job, calib_lab)
    # 법선이 위로 기울면(광학 원점이 아래 점보다 높음) 통과
    n2 = np.array([0.0, -math.cos(t), math.sin(t)])
    u2 = np.array([0.0, math.sin(t), math.cos(t)])
    d.update(end=(np.array([0.5, 0.7, 0.3]) + 0.3 * u2).tolist(), surface_normal=n2.tolist())
    plan = build_transit_plan(ScanJob.from_dict(d), calib_lab)
    assert check_envelope(plan, ScanJob.from_dict(d), calib_lab)["floor_margin_min"] > 0.0


def test_3g_floor_applies_to_interpolated_samples_not_only_waypoints(job_3g, calib_lab):
    """웨이포인트는 모두 바닥 위지만 중간 구간이 바닥 아래로 처지는 인위적 계획 → 보간 샘플에서 검출."""
    plan = build_transit_plan(job_3g, calib_lab)
    names = [w.name for w in plan.waypoints]
    i = names.index("pre_exit")
    # pre_exit 를 바닥 아래로 내리고(웨이포인트 자체 위반) 검출되는지, 그리고 웨이포인트를 되돌리고
    # 그 사이에 바닥 아래 경유점을 삽입해도 검출되는지 확인
    wp = plan.waypoints[i]
    T_low = wp.T_base_flange.copy()
    T_low[2, 3] -= 0.02
    from scan_core.path_gen import Waypoint
    plan.waypoints.insert(i, Waypoint("dip", "retreat_normal", T_low, 0.02, 0.2))
    with pytest.raises(EnvelopeError, match=r"\[dip/retreat_normal\] rule=3G_floor"):
        check_envelope(plan, job_3g, calib_lab)


def test_3g_home_below_floor_is_allowed_on_home_legs_only(job_3g, calib_lab):
    """홈이 바닥보다 낮아도 홈 왕복 구간(floor_rule=False)은 예외, 그 외 구간은 규칙 적용."""
    T_home = np.array([[1, 0, 0, 0.3], [0, -1, 0, -0.3], [0, 0, -1, 0.40], [0, 0, 0, 1.0]])   # 광학 원점 z≈0.25 < floor 0.3
    assert tool_keypoints(T_home, calib_lab, job_3g)["opt"][2] < 0.3
    job_3g = ScanJob.from_dict({**job_3g.to_dict(), "safe_clearance_m": 0.05})   # 홈의 가장 낮은 키포인트가 테이블 위 85 mm
    plan = build_transit_plan(job_3g, calib_lab, T_home_flange=T_home)
    home_like = [w for w in plan.waypoints if not w.floor_rule]
    assert {w.name for w in home_like} == {"home", "home_above"}
    assert all(w.floor_rule for w in plan.waypoints if w.name not in ("home", "home_above"))
    O = _opt_trajectory(plan, calib_lab, step=0.002, skip_home=True)
    assert O[:, 2].min() >= 0.3 - 1e-6


# --------------------------------------------------------------------------------------
# transit 위쪽 규칙 (광학 원점) · 토치 키포인트 플래그
# --------------------------------------------------------------------------------------
def test_transit_above_rule_uses_optical_origin_1g(job_1g, calib_lab):
    # specimen_top 이 스캔면보다 높은 경우: z_safe = max(top, 위 점)+clearance, 광학 원점은 + standoff
    job = ScanJob.from_dict({**job_1g.to_dict(), "specimen_top_z_m": 0.15})
    plan = build_transit_plan(job, calib_lab)
    assert plan.z_safe_m == pytest.approx(0.15 + 0.10)
    assert plan.z_safe_opt_m == pytest.approx(0.15 + 0.10 + 0.32)
    for w in plan.waypoints:
        if w.phase == "transit" or w.name.startswith("above"):
            kp = tool_keypoints(w.T_base_flange, calib_lab, job)
            assert kp["opt"][2] >= plan.z_safe_opt_m - 1e-9
            assert kp["tip"][2] >= plan.z_safe_m - 1e-9
    # transit 웨이포인트를 광학 원점 기준으로 2 cm 낮추면 rule=transit_above (optical origin) 위반
    T_home = np.array([[1, 0, 0, 0.3], [0, -1, 0, 0.0], [0, 0, -1, 0.9], [0, 0, 0, 1.0]])
    plan2 = build_transit_plan(job, calib_lab, T_home_flange=T_home)
    names = [w.name for w in plan2.waypoints]
    i = names.index("above_lead_in")
    assert plan2.waypoints[i].phase == "transit"
    plan2.waypoints[i].T_base_flange[2, 3] -= 0.02
    with pytest.raises(EnvelopeError, match="rule=transit_above: transit optical origin"):
        check_envelope(plan2, job, calib_lab)


def test_transit_above_rule_uses_optical_origin_3g(job_3g, calib_lab):
    plan = build_transit_plan(job_3g, calib_lab)
    assert plan.z_safe_m == pytest.approx(0.7 + 0.10) and plan.z_safe_opt_m == pytest.approx(0.7 + 0.10)
    T_home = np.array([[1, 0, 0, 0.3], [0, -1, 0, 0.0], [0, 0, -1, 0.9], [0, 0, 0, 1.0]])
    plan2 = build_transit_plan(job_3g, calib_lab, T_home_flange=T_home)
    names = [w.name for w in plan2.waypoints]
    i = names.index("above_pre_in")
    assert plan2.waypoints[i].phase == "transit"
    o = plan2.waypoints[i].opt_origin(calib_lab)
    assert o[2] >= plan2.z_safe_opt_m
    plan2.waypoints[i].T_base_flange[2, 3] -= 0.03
    with pytest.raises(EnvelopeError, match="rule=transit_above"):
        check_envelope(plan2, job_3g, calib_lab)


def test_torch_keypoint_flag(job_1g, calib_lab):
    """시편 상단이 토치 팁보다 높은 1G: 플래그 ON 이면 '(torch keypoint)' 메시지로 거부, OFF 면 광학 원점만 검사 → 통과."""
    d = {**job_1g.to_dict(), "specimen_top_z_m": 0.05 + 0.20}
    with pytest.raises(EnvelopeError, match=r"rule=1G_above_specimen_top: torch tip .*\(torch keypoint\)"):
        build_transit_plan(ScanJob.from_dict(d), calib_lab)
    job_off = ScanJob.from_dict({**d, "check_torch_tip": False})
    plan = build_transit_plan(job_off, calib_lab)
    stats = check_envelope(plan, job_off, calib_lab)
    assert stats["min_opt_height"] >= 0.25
    assert stats["min_tip_height"] < 0.25            # 팁은 실제로 시편 상단 아래지만 플래그 OFF 라 검사 안 함
    # transit 규칙에서도 플래그 (3G: 홈 자세→스캔 자세 slerp 도중 토치 팁이 흔들려 내려감):
    # OFF 로 만든 계획(광학 원점만 들어올림)을 ON 작업으로 검사하면 '(torch keypoint)' 위반, ON 으로 만들면 양끝이 더 들려 통과
    job3_on = ScanJob(start=[0.5, 0.7, 0.3], end=[0.5, 0.7, 0.6], surface_normal=[0, -1, 0], specimen_top_z_m=0.7,
                      table_z_m=0.0)
    job3_off = ScanJob.from_dict({**job3_on.to_dict(), "check_torch_tip": False})
    T_home = np.array([[1, 0, 0, 0.3], [0, -1, 0, 0.0], [0, 0, -1, 0.9], [0, 0, 0, 1.0]])
    plan_off = build_transit_plan(job3_off, calib_lab, T_home_flange=T_home)
    plan_on = build_transit_plan(job3_on, calib_lab, T_home_flange=T_home)
    st_off = check_envelope(plan_off, job3_off, calib_lab)
    st_on = check_envelope(plan_on, job3_on, calib_lab)
    assert st_off["min_opt_height"] >= 0.3 - 1e-9 and st_on["min_tip_height"] >= 0.0
    # 두 계획 모두 광학 원점 transit 규칙은 만족하지만 OFF 계획의 팁은 z_safe 아래로 흔들린다
    i_off = [w.name for w in plan_off.waypoints].index("above_pre_in")
    i_on = [w.name for w in plan_on.waypoints].index("above_pre_in")
    assert plan_on.waypoints[i_on].T_base_flange[2, 3] > plan_off.waypoints[i_off].T_base_flange[2, 3]
    with pytest.raises(EnvelopeError, match=r"rule=transit_above: transit torch tip .*\(torch keypoint\)"):
        check_envelope(plan_off, job3_on, calib_lab)
    assert any("raised" in n for n in plan_on.notes)


def test_envelope_sample_step_is_capped_at_5mm(job_1g, calib_lab):
    plan = build_transit_plan(job_1g, calib_lab, check=False)
    stats = check_envelope(plan, job_1g, calib_lab, sample_step_m=0.05)
    assert stats["sample_step_m"] == pytest.approx(0.005)
    # 스캔 구간 0.58 m → 최소 117 샘플
    assert stats["n_samples"] > 0.58 / 0.005


def test_plan_summary_mentions_rules_and_optical_origin(job_3g, calib_lab):
    s = build_transit_plan(job_3g, calib_lab).summary()
    assert "3G_floor" in s and "opt=(" in s and "reference=optical" in s


# --------------------------------------------------------------------------------------
# 3G 'down' 리드아웃 없음: 감속 구간 프레임은 valid 이지만 in_cruise=False, 커버리지는 기대 순항 구간 기준
# --------------------------------------------------------------------------------------
def test_3g_down_decel_zone_frames_valid_but_not_cruise(job_3g, calib_lab):
    d = job_3g.to_dict()
    d.update(end=[0.5, 0.7, 0.45], speed_mps=0.05, accel_mps2=0.5, lead_in_m=0.02, lead_out_m=0.02)
    job = ScanJob.from_dict(d)
    s_from, s_to, sign = scan_s_range(job)
    assert (s_from, s_to, sign) == (pytest.approx(0.15 + 0.02), pytest.approx(0.0), -1)
    d_kin = 0.05 ** 2 / (2 * 0.5)                        # 운동학적 감속 거리 2.5 mm
    assert ramp_zones_m(job) == [(0.0, pytest.approx(d_kin))]
    assert expected_cruise_s_range(job) == (pytest.approx(d_kin), pytest.approx(0.15))
    d_ramp = d_kin + 0.05 * 0.15                          # + 속도 측정창(0.3 s)/2 번짐 = 10 mm
    cfg = SynthConfig(n_points=96, sigma_z_mm=0.0, rx_latency_s=0.0, robot_poll_rate_hz=0.0, seed=3)
    syn = simulate_scan(job, calib_lab, PlaneScene(h0=0.0), cfg)
    # 합성 모션이 위(0.17)에서 아래(0.0)로, 아래 점에서 정확히 멈춘다
    assert syn.truth["s_m"].max() == pytest.approx(0.17, abs=1e-9) and syn.truth["s_m"].min() == pytest.approx(0.0, abs=1e-9)
    res = reconstruct_scan(syn.frames, syn.robot, calib_lab, job)
    assert res.meta["travel_sign"] == -1 and res.meta["ramp_zones_m"] == [[0.0, pytest.approx(d_ramp)]]
    in_ramp = (res.s_m >= 0.0) & (res.s_m <= d_ramp)
    assert in_ramp.sum() > 0
    # 감속 구간 프레임: 점은 유효(valid), 하지만 순항 아님
    assert np.all(res.valid[in_ramp].mean(axis=1) > 0.9)
    assert (res.in_cruise & in_ramp & (res.s_m < d_ramp * 0.5)).sum() == 0
    # 리드 쪽(위)은 순항이 시편 상단까지 도달
    assert res.in_cruise[(res.s_m > 0.149) & (res.s_m < 0.15)].any()
    lo, hi = expected_cruise_s_range(job, settle_s=float(res.meta.get("speed_window_s", 0.3)) / 2.0)
    assert (lo, hi) == (pytest.approx(d_ramp), pytest.approx(0.15))
    s_c = res.s_m[res.in_cruise]
    assert s_c.min() <= lo + 2e-3 and s_c.max() >= hi - 2e-3                  # 순항이 기대 구간을 덮는다
    assert (in_ramp & res.in_cruise).sum() < in_ramp.sum()
