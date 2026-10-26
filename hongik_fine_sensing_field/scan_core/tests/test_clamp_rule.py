"""clamp_clearance 규칙 + 센서 실측 판 상면(plate_top_measured_m)으로 올려진 바닥 (설계 패널 2026-09-30 revision #5).

높이는 항상 up·P (base 3D 키포인트, up = −gravity_axis_base) 로 계산되며 센서 원시 z 가 아니다 (I2) — 중력축을 바꾼
작업에서도 같은 규칙이 같은 방식으로 동작하는지 함께 검사한다.
"""
import math

import numpy as np
import pytest

from scan_core.frames import inv_transform, transform_points
from scan_core.path_gen import (RULES, EnvelopeError, Waypoint, build_transit_plan, check_envelope, sensor_frames,
                                tool_keypoints)
from scan_core.types import JobError, ScanJob

def _job(job_1g, **kw) -> ScanJob:
    return ScanJob.from_dict({**job_1g.to_dict(), **kw})


def _tip_above_plate(job: ScanJob, calib) -> float:
    """스캔 자세(s=0.1)에서 토치 팁의 판 위 높이(1G, up·P − plate_top) / 판 평면 거리(3G) — base 3D 키포인트로 계산.
    피치 −5° + 토치 측면 오프셋(−100 mm) 때문에 단순 'standoff − 168 mm'(0.152) 가 아니라 ≈0.1624 m 이다 (I2)."""
    fr = sensor_frames(job, calib)
    kp = tool_keypoints(fr.T_base_flange(0.1), calib, job)
    if fr.orientation == "flat_1G":
        return float(fr.height(kp["tip"])) - float(job.plate_top_z_m())
    return float((kp["tip"] - fr.start) @ fr.n)


# --------------------------------------------------------------------------------------
# ScanJob 필드
# --------------------------------------------------------------------------------------
def test_scanjob_clamp_and_measured_top_fields(job_1g):
    assert job_1g.clamp_height_m == 0.0 and job_1g.plate_top_measured_m is None
    assert job_1g.plate_top_z_m() == pytest.approx(0.05)
    d = job_1g.to_dict()
    assert d["clamp_height_m"] == 0.0 and d["plate_top_measured_m"] is None
    j2 = ScanJob.from_dict({**d, "clamp_height_m": 0.02, "plate_top_measured_m": 0.08})
    assert j2.validate().clamp_height_m == 0.02 and j2.plate_top_measured_m == 0.08
    assert j2.plate_top_z_m() == pytest.approx(0.08)                      # 실측이 입력을 올린다
    assert ScanJob.from_dict({**d, "plate_top_measured_m": 0.01}).plate_top_z_m() == pytest.approx(0.05)   # 내리지는 않는다
    assert ScanJob.from_dict({**d, "specimen_top_z_m": None}).plate_top_z_m() is None
    assert ScanJob.from_dict({**d, "specimen_top_z_m": None, "plate_top_measured_m": 0.07}).plate_top_z_m() == pytest.approx(0.07)
    with pytest.raises(JobError, match="clamp_height_m"):
        ScanJob.from_dict({**d, "clamp_height_m": -0.01}).validate()
    with pytest.raises(JobError, match="clamp_height_m"):
        ScanJob.from_dict({**d, "clamp_height_m": float("nan")}).validate()
    with pytest.raises(JobError, match="plate_top_measured_m"):
        ScanJob.from_dict({**d, "plate_top_measured_m": float("inf")}).validate()


def test_rule_registered_and_listed(job_1g, job_3g, calib_lab):
    assert "clamp_clearance" in RULES
    assert "clamp_clearance" in build_transit_plan(job_1g, calib_lab).rules
    assert "clamp_clearance" in build_transit_plan(job_3g, calib_lab).rules


# --------------------------------------------------------------------------------------
# 1G 클램프: 판 위 구간에서 토치 팁 / 광학 원점 높이 ≥ plate_top + clamp
# --------------------------------------------------------------------------------------
def test_clamp_refuses_in_plane_leg_with_torch_tip_below_plate_top_plus_clamp(job_1g, calib_lab):
    tip_h = _tip_above_plate(job_1g, calib_lab)                 # ≈ 0.1624 (피치·측면 오프셋 반영), 0.152 아님
    assert 0.16 < tip_h < 0.165 and abs(tip_h - (0.32 - 0.168)) > 0.005
    # 스캔 자세: 광학 원점 = 판 + 0.32, 토치 팁 = 판 + tip_h. clamp 0.20 → 팁이 plate_top + clamp = 0.25 아래 → 거부
    job = _job(job_1g, clamp_height_m=0.20)
    with pytest.raises(EnvelopeError, match=r"rule=clamp_clearance: torch tip height .*during in-plane motion \(torch keypoint\)"):
        build_transit_plan(job, calib_lab)
    # clamp = 0 → 통과 (판 상면 자체가 하한; 기존 1G_above_specimen_top 과 동일)
    plan0 = build_transit_plan(_job(job_1g, clamp_height_m=0.0), calib_lab)
    st = check_envelope(plan0, _job(job_1g, clamp_height_m=0.0), calib_lab)
    assert st["clamp_height_m"] == 0.0 and st["clamp_margin_min"] == pytest.approx(tip_h, abs=1e-6)
    # 팁이 아슬아슬하게 위: clamp = tip_h − 2 mm → 통과, clamp_margin_min = 2 mm; clamp = tip_h + 2 mm → 거부
    job_ok = _job(job_1g, clamp_height_m=tip_h - 0.002)
    st = check_envelope(build_transit_plan(job_ok, calib_lab), job_ok, calib_lab)
    assert st["clamp_margin_min"] == pytest.approx(0.002, abs=1e-6)
    with pytest.raises(EnvelopeError, match="rule=clamp_clearance: torch tip height"):
        build_transit_plan(_job(job_1g, clamp_height_m=tip_h + 0.002), calib_lab)
    # 토치 팁 플래그 OFF: 팁은 검사하지 않고 광학 원점(0.37)만 → clamp 0.20 통과, clamp 0.35 (0.40 > 0.37) 는 광학 원점으로 거부
    build_transit_plan(_job(job_1g, clamp_height_m=0.20, check_torch_tip=False), calib_lab)
    with pytest.raises(EnvelopeError, match=r"rule=clamp_clearance: optical origin height"):
        build_transit_plan(_job(job_1g, clamp_height_m=0.35, check_torch_tip=False), calib_lab)


def test_clamp_message_names_the_scan_leg_and_the_plate_top(job_1g, calib_lab):
    job = _job(job_1g, clamp_height_m=0.20)
    with pytest.raises(EnvelopeError) as ei:
        build_transit_plan(job, calib_lab)
    msg = str(ei.value)
    assert "rule=clamp_clearance" in msg and "plate_top+clamp_height=0.0500+0.2000=0.2500" in msg
    assert msg.startswith("[lead_in/descend]") or msg.startswith("[lead_out/scan]")   # 하강 하단 샘플 또는 스캔 구간에서 검출


def test_clamp_checks_working_height_samples_of_descend_but_not_ascend(job_1g, calib_lab):
    """클램프 대역으로 내려가는 descend 는 하단 샘플에서 거부되고, 대역을 벗어나는 ascend 와 현재 자세('start')는 면제."""
    job = _job(job_1g, clamp_height_m=0.20)
    plan = build_transit_plan(_job(job_1g), calib_lab, check=False)      # clamp 0 로 만든 기하를 clamp 0.20 작업으로 검사
    names = [w.name for w in plan.waypoints]
    i_in = names.index("lead_in")
    assert plan.waypoints[i_in].phase == "descend"
    # descend 만 있는 부분 계획 → 하단(작업 높이) 샘플에서 거부
    sub = _subplan(plan, [i_in - 1, i_in])
    with pytest.raises(EnvelopeError, match=r"\[lead_in/descend\] rule=clamp_clearance"):
        check_envelope(sub, job, calib_lab)
    # 같은 두 자세를 반대로(ascend) → 허용
    T_low, T_high = plan.waypoints[i_in].T_base_flange, plan.waypoints[i_in - 1].T_base_flange
    up_plan = _subplan(plan, [i_in, i_in - 1])
    up_plan.waypoints[0] = Waypoint("now", "start", T_low, 0.0, 0.0)
    up_plan.waypoints[1] = Waypoint("above", "ascend", T_high, 0.02, 0.2)
    st = check_envelope(up_plan, job, calib_lab)
    assert st["n_samples"] > 2


def _subplan(plan, idx):
    import copy
    p = copy.copy(plan)
    p.waypoints = [copy.copy(plan.waypoints[i]) for i in idx]
    p.scan_leg_indices = [k for k, i in enumerate(idx) if plan.waypoints[i].phase == "scan"]
    return p


# --------------------------------------------------------------------------------------
# 3G 클램프: 모든 키포인트 판 평면 거리 ≥ clamp
# --------------------------------------------------------------------------------------
def test_clamp_3g_uses_plate_plane_distance(job_3g, calib_lab):
    # 스캔 자세: 광학 원점 판거리 0.32, 토치 팁 ≈ 0.162 (피치·측면 오프셋). clamp 0.20 → 팁 거부, 0.10 → 통과
    tip_d = _tip_above_plate(job_3g, calib_lab)
    assert 0.16 < tip_d < 0.165
    with pytest.raises(EnvelopeError, match=r"rule=clamp_clearance: torch tip distance to plate plane .*\(torch keypoint\)"):
        build_transit_plan(ScanJob.from_dict({**job_3g.to_dict(), "clamp_height_m": 0.20}), calib_lab)
    job_ok = ScanJob.from_dict({**job_3g.to_dict(), "clamp_height_m": 0.10})
    st = check_envelope(build_transit_plan(job_ok, calib_lab), job_ok, calib_lab)
    assert st["clamp_margin_min"] == pytest.approx(tip_d - 0.10, abs=1e-6)
    assert st["min_plate_dist_tip"] == pytest.approx(tip_d, abs=1e-6)
    # 높이(base z)가 아니라 판 평면 거리로 판정: 팁 높이(≥ 0.4) 는 clamp 0.20 보다 훨씬 큰데도 판거리 0.162 로 거부됐다
    assert st["min_tip_height"] > 0.20


# --------------------------------------------------------------------------------------
# 실측 판 상면이 바닥을 올린다 (1G_above_specimen_top · never_below_table · clamp_clearance)
# --------------------------------------------------------------------------------------
def test_measured_plate_top_raises_floor_for_specimen_top_rule(job_1g, calib_lab):
    # specimen_top 0.05 로는 유효한 계획 (팁 ≈ 0.212). 실측 판 상면 0.22 → 팁이 그 아래 → 거부
    # (실측 바닥은 모든 키포인트의 하한이라 never_below_table 이 먼저 잡는다; 1G_above_specimen_top 의 top 도 같은 값으로 올라간다)
    plan_ok = build_transit_plan(job_1g, calib_lab)
    assert plan_ok.specimen_top_z_m == pytest.approx(0.05)
    job_meas = _job(job_1g, plate_top_measured_m=0.22)
    with pytest.raises(EnvelopeError, match=r"rule=never_below_table: tool keypoint height 0.21\d+ m below floor 0.2200 m .*measured plate top=0.2200"):
        build_transit_plan(job_meas, calib_lab)
    # 1G_above_specimen_top 도 올려진 top 을 쓴다: 테이블 규칙이 먼저 잡지 못하게 tip 플래그 ON 인 채 광학 원점만 낮은 자세로 확인
    # → 실측 0.40 (광학 원점 0.37 아래): never_below_table 이 먼저 잡되 메시지의 바닥이 실측값
    plan_nochk = build_transit_plan(job_1g, calib_lab, check=False)
    with pytest.raises(EnvelopeError, match=r"rule=never_below_table: .*measured plate top=0.4000"):
        check_envelope(plan_nochk, _job(job_1g, plate_top_measured_m=0.40), calib_lab)
    # 실측이 plan 생성 후에 올라간 경우: check_envelope 는 max(plan top, job 실측) 을 쓴다 (올리기만)
    st = check_envelope(plan_nochk, _job(job_1g, plate_top_measured_m=0.08), calib_lab)
    assert st["plate_top_m"] == pytest.approx(0.08)
    # 실측 0.10 (팁 위): 계획은 만들어지되 판 상면·z_safe 가 실측값으로 올라간다
    job_10 = _job(job_1g, plate_top_measured_m=0.10)
    plan = build_transit_plan(job_10, calib_lab)
    assert plan.specimen_top_z_m == pytest.approx(0.10) and plan.z_safe_m == pytest.approx(0.20)
    assert plan.z_safe_opt_m == pytest.approx(0.20 + 0.32)
    # 실측이 입력보다 낮으면 무시 (올리기만)
    plan_low = build_transit_plan(_job(job_1g, plate_top_measured_m=0.01), calib_lab)
    assert plan_low.specimen_top_z_m == pytest.approx(0.05)


def test_measured_plate_top_plus_clamp_refuses_plan_valid_against_typed_top(job_1g, calib_lab):
    """specimen_top 0.05 + clamp 0.15 로 유효한 계획(팁 ≈0.212 ≥ 0.20) 이 실측 판 상면 0.10 이 들어오면 (0.25) 거부된다."""
    job_typed = _job(job_1g, clamp_height_m=0.15)
    plan = build_transit_plan(job_typed, calib_lab)
    check_envelope(plan, job_typed, calib_lab)
    job_meas = _job(job_1g, clamp_height_m=0.15, plate_top_measured_m=0.10)
    with pytest.raises(EnvelopeError, match=r"rule=clamp_clearance: torch tip height .*plate_top\+clamp_height=0.1000\+0.1500"):
        build_transit_plan(job_meas, calib_lab)
    # 같은 (검사 없이 만든) 기하를 실측 작업으로 재검사해도 거부 — 포락선은 ScanJob 필드만 본다
    with pytest.raises(EnvelopeError, match="rule=clamp_clearance"):
        check_envelope(build_transit_plan(job_typed, calib_lab, check=False), job_meas, calib_lab)


def test_measured_plate_top_is_a_physical_floor_for_every_keypoint(job_1g, calib_lab):
    """never_below_table: 실측 판 상면은 토치 팁 플래그와 무관하게 모든 키포인트의 하한 (판이 테이블 위에 있으므로 테이블 바닥을 올린다)."""
    job = _job(job_1g, plate_top_measured_m=0.30, check_torch_tip=False)   # 광학 원점 0.37 은 위, 팁 ≈0.212 는 아래
    with pytest.raises(EnvelopeError, match=r"rule=never_below_table: tool keypoint height 0.21\d+ m below floor 0.3000 m .*measured plate top"):
        build_transit_plan(job, calib_lab)
    # 테이블이 없고 실측만 있어도 같은 규칙
    job2 = _job(job_1g, table_z_m=None, plate_top_measured_m=0.30, check_torch_tip=False)
    with pytest.raises(EnvelopeError, match="rule=never_below_table"):
        build_transit_plan(job2, calib_lab)
    st = check_envelope(build_transit_plan(_job(job_1g, plate_top_measured_m=0.10), calib_lab),
                        _job(job_1g, plate_top_measured_m=0.10), calib_lab)
    assert st["plate_top_m"] == pytest.approx(0.10) and st["plate_top_measured_m"] == pytest.approx(0.10)


# --------------------------------------------------------------------------------------
# I2: 높이는 up·P (중력축 기준) — 중력축이 base y 인 좌표계에서도 같은 규칙
# --------------------------------------------------------------------------------------
def test_clamp_uses_gravity_axis_heights_not_base_z(calib_lab):
    # base +y 가 아래쪽: 판은 y = 0.5 평면 (높이 = −0.5), 법선 −y (위), 경로 x 방향 (수평)
    base = dict(start=[0.0, 0.5, 0.0], end=[0.3, 0.5, 0.0], surface_normal=[0, -1, 0], gravity_axis_base=[0, 1, 0],
                specimen_top_z_m=-0.5)
    job0 = ScanJob(**base)
    assert job0.resolved_orientation() == "flat_1G"
    plan = build_transit_plan(job0, calib_lab)
    st = check_envelope(plan, job0, calib_lab)
    tip_h = _tip_above_plate(job0, calib_lab)                                 # ≈ 0.1624, up 축 기준
    assert st["min_tip_height"] == pytest.approx(-0.5 + tip_h, abs=1e-6)
    kp = tool_keypoints(plan.scan_waypoints()[0].T_base_flange, calib_lab, job0)
    assert kp["tip"][1] == pytest.approx(0.5 - tip_h, abs=1e-6)              # base y 로는 '아래'가 +y
    assert abs(kp["tip"][2]) < 0.15                                           # base z 는 높이가 아니다 (측면 오프셋만)
    # clamp 0.20: 팁 높이 ≈ −0.338 < −0.30 → 거부 (base z 를 높이로 쓰면 검출 못 함)
    with pytest.raises(EnvelopeError, match=f"rule=clamp_clearance: torch tip height {-0.5 + tip_h:.4f}"):
        build_transit_plan(ScanJob(**base, clamp_height_m=0.20), calib_lab)
    build_transit_plan(ScanJob(**base, clamp_height_m=0.10), calib_lab)
    # 실측 판 상면도 같은 높이축: −0.30 (팁 −0.338 보다 위) → 거부, 메시지의 바닥은 실측값; −0.40 (팁 아래) 은 통과
    with pytest.raises(EnvelopeError, match=f"rule=never_below_table: tool keypoint height {-0.5 + tip_h:.4f} m below floor -0.3000"):
        build_transit_plan(ScanJob(**base, plate_top_measured_m=-0.30), calib_lab)
    plan_m = build_transit_plan(ScanJob(**base, plate_top_measured_m=-0.40), calib_lab)
    assert plan_m.specimen_top_z_m == pytest.approx(-0.40)


def test_tilted_tool_pose_heights_come_from_base_keypoints(job_1g, calib_lab):
    """레이저 기울기(피치 20°)로 광학 Z 가 중력과 어긋나도 키포인트 높이는 base 3D 점의 up·P 로 계산된다."""
    job = _job(job_1g, pitch_deg=-20.0, clamp_height_m=0.0)
    fr = sensor_frames(job, calib_lab)
    T = fr.T_base_flange(0.1)
    kp = tool_keypoints(T, calib_lab, job)
    # 광학 원점은 여전히 판 + standoff (순수 회전); 토치 팁은 툴이 기울어 '광학 원점 − 168 mm' 가 아니다
    assert kp["opt"][2] == pytest.approx(0.05 + 0.32, abs=1e-9)
    naive_tip_h = 0.05 + 0.32 - 0.168                  # 원시 툴 축 길이를 높이로 쓰면 이 값 (틀림)
    assert abs(kp["tip"][2] - naive_tip_h) > 0.005
    # base 3D 팁 점을 툴 프레임으로 되돌리면 정확히 (torch_lateral, p_opt_z + protrusion): 높이는 그 점의 up·P
    tip_eoat = transform_points(inv_transform(T @ calib_lab.T_flange_eoat), kp["tip"])
    np.testing.assert_allclose(tip_eoat, [-0.100, 0.0, calib_lab.p_opt_eoat[2] + 0.168], atol=1e-12)
    assert kp["tip"][2] == pytest.approx(float(fr.height(kp["tip"])), abs=1e-12)
    # 같은 기울기로 clamp 를 걸면 팁의 '실제' 높이로 판정된다
    lim = kp["tip"][2] - 0.05                          # 팁이 판 위로 떠 있는 실제 높이
    build_transit_plan(_job(job, clamp_height_m=lim - 0.002), calib_lab)
    with pytest.raises(EnvelopeError, match="rule=clamp_clearance: torch tip height"):
        build_transit_plan(_job(job, clamp_height_m=lim + 0.002), calib_lab)
