"""ScanJob.plate_plane_measured (센서 실측 판 평면) 과 check_envelope 의 3G 판 평면 규칙 (리뷰 라운드 3 S8).

정련 작업의 시점/종점은 루트(그루브 바닥) 점이라 (start, surface_normal) 평면은 판 상면보다 그루브 깊이만큼 뒤에 있다. 실측 판
평면이 있으면 never_cross_plate_plane / 3G clamp_clearance 는 두 평면 중 **더 가까운** 거리를 쓴다 (보수적, I1). 채택은 raise-only.
"""
import numpy as np
import pytest

from scan_core.path_gen import EnvelopeError, build_transit_plan, check_envelope, sensor_frames, tool_keypoints
from scan_core.types import JobError, ScanJob, merge_plate_plane


def _tip_dist(job, calib, s=0.1) -> float:
    fr = sensor_frames(job, calib)
    kp = tool_keypoints(fr.T_base_flange(s), calib, job)
    return float((kp["tip"] - fr.start) @ fr.n)


def test_field_normalised_serialised_and_merged_raise_only(job_3g):
    n = job_3g.normal_unit
    j = ScanJob.from_dict({**job_3g.to_dict(), "plate_plane_measured": (job_3g.start + n * 0.01, n * 2.0)})
    assert j.plate_plane_measured == {"point_m": pytest.approx(list(job_3g.start + n * 0.01)), "normal": pytest.approx(list(n))}
    j2 = ScanJob.from_dict(j.to_dict())                                    # dict 로 직렬화·복원
    p, nn = j2.plate_plane()
    np.testing.assert_allclose(p, job_3g.start + n * 0.01)
    np.testing.assert_allclose(nn, n)
    assert job_3g.plate_plane() is None and job_3g.to_dict()["plate_plane_measured"] is None
    with pytest.raises(JobError):
        ScanJob.from_dict({**job_3g.to_dict(), "plate_plane_measured": {"point_m": [0, 0, 0], "normal": [0, 0, 0]}})
    # raise-only: 로봇 쪽(+n) 평면만 채택, 더 먼 평면은 무시
    a = {"point_m": list(job_3g.start), "normal": list(n)}
    closer = merge_plate_plane(a, (job_3g.start + n * 0.005, n))
    farther = merge_plate_plane(closer, (job_3g.start - n * 0.010, n))
    assert closer["point_m"] == pytest.approx(list(job_3g.start + n * 0.005)) and farther == closer
    assert merge_plate_plane(None, (job_3g.start, -n))["normal"] == pytest.approx(list(-n))     # 첫 평면은 그대로
    flipped = merge_plate_plane(a, (job_3g.start + n * 0.002, -n))                              # 반구 정렬
    assert flipped["normal"] == pytest.approx(list(n))


def test_3g_fine_leg_ok_against_the_root_plane_but_too_close_to_the_measured_plate_is_refused(job_3g, calib_lab):
    """루트 직선 평면(start/n) 기준 팁 거리 0.162 m 로 clamp 0.155 를 통과하는 3G 스캔이, 그루브 깊이 10 mm 앞의 실측 판 상면
    기준으로는 0.152 m < 0.155 → clamp_clearance 로 REFUSED. 실측 평면이 더 가까우면 never_cross_plate_plane 도 그 평면으로."""
    n = job_3g.normal_unit
    tip_root = _tip_dist(job_3g, calib_lab)
    assert 0.16 < tip_root < 0.165
    job_ok = ScanJob.from_dict({**job_3g.to_dict(), "clamp_height_m": 0.155})
    st = check_envelope(build_transit_plan(job_ok, calib_lab), job_ok, calib_lab)          # 루트 평면만: 통과
    assert st["min_plate_dist_tip"] == pytest.approx(tip_root, abs=1e-6) and st["plate_plane"] == "start/surface_normal"
    job_meas = ScanJob.from_dict({**job_ok.to_dict(), "plate_plane_measured": (job_3g.start + n * 0.010, n)})
    plan = build_transit_plan(job_ok, calib_lab)                           # 같은 계획(루트 평면에서 만든) 을 실측 평면으로 검사
    with pytest.raises(EnvelopeError, match=r"rule=clamp_clearance: torch tip distance to plate plane 0\.15"):
        check_envelope(plan, job_meas, calib_lab)
    # clamp 없이: 실측 평면이 130 mm 앞이면 광학 원점 0.19 < standoff_min 0.245 (팁 0.032 < 0.05) → never_cross_plate_plane
    job_near = ScanJob.from_dict({**job_3g.to_dict(), "plate_plane_measured": (job_3g.start + n * 0.130, n)})
    with pytest.raises(EnvelopeError, match=r"rule=never_cross_plate_plane: optical origin distance to plate plane 0\.19"):
        check_envelope(build_transit_plan(job_3g, calib_lab), job_near, calib_lab)
    # 더 먼(뒤의) 실측 평면은 규칙을 느슨하게 하지 않는다 (가까운 쪽 = 루트 평면)
    job_far = ScanJob.from_dict({**job_ok.to_dict(), "plate_plane_measured": (job_3g.start - n * 0.050, n)})
    st2 = check_envelope(build_transit_plan(job_ok, calib_lab), job_far, calib_lab)
    assert st2["min_plate_dist_tip"] == pytest.approx(tip_root, abs=1e-6)
    assert "measured plate plane" in st2["plate_plane"]


def test_tilted_measured_plane_is_used_away_from_the_dwell_point(job_3g, calib_lab):
    """실측 법선이 거친 법선과 2° 다르면 판은 멀리서 갈라진다: 위 끝에서 판이 로봇 쪽으로 다가오는 실측 평면이면 그 끝의 샘플이
    거부된다 (거친 평면만으로는 통과)."""
    n = job_3g.normal_unit
    u = job_3g.unit_dir
    ang = np.radians(2.0)
    n_t = np.cos(ang) * n - np.sin(ang) * u                                # 위쪽(+u) 으로 갈수록 판이 로봇 쪽으로
    job_c = ScanJob.from_dict({**job_3g.to_dict(), "clamp_height_m": 0.155})
    plan = build_transit_plan(job_c, calib_lab)
    check_envelope(plan, job_c, calib_lab)                                 # 거친 평면: 통과 (팁 0.162 ≥ 0.155)
    job_t = ScanJob.from_dict({**job_c.to_dict(), "plate_plane_measured": (job_3g.start, n_t)})
    with pytest.raises(EnvelopeError, match="rule=clamp_clearance"):
        check_envelope(plan, job_t, calib_lab)
