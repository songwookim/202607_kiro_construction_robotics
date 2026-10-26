"""입사 기울기 incidence_tilt_deg: 3G 부호 규약(+ = 내려다봄 → 원점이 심 점 위), 바닥 규칙, 1G 는 pitch 와 합성, 검증."""
import math
import warnings

import numpy as np
import pytest

from scan_core.path_gen import (EnvelopeError, build_transit_plan, check_envelope, incidence_tilt_rotation, sensor_frames,
                                tool_keypoints)
from scan_core.reconstruct import reconstruct_scan
from scan_core.synth import PlaneScene, SynthConfig, simulate_scan
from scan_core.types import IncidenceTiltWarning, JobError, ScanJob


def _with(job, **kw):
    return ScanJob.from_dict({**job.to_dict(), **kw})


def _torch_up(job, calib):
    """3G: 툴 진행 부호 중 토치가 레이저 위쪽인 쪽 (auto-teach 가 고르는 쪽; 3G 바닥은 토치 팁도 본다, 2026-10-05)."""
    def tip_up(j):
        kp = tool_keypoints(sensor_frames(j, calib).T_base_flange(0.0), calib, j)
        return float(j.height(kp["tip"]) - j.height(kp["opt"]))
    return max((_with(job, progression_sign=s) for s in (1, -1)), key=tip_up)


def test_scanjob_incidence_tilt_default_roundtrip_and_validation(job_1g):
    assert job_1g.incidence_tilt_deg == 0.0
    d = job_1g.to_dict()
    assert d["incidence_tilt_deg"] == 0.0
    assert ScanJob.from_dict({**d, "incidence_tilt_deg": 7.5}).incidence_tilt_deg == 7.5
    _with(job_1g, incidence_tilt_deg=35.0, pitch_deg=-5.0, max_laser_tilt_deg=40.0).validate()      # 3G 상한 (2026-10-05)
    with pytest.raises(JobError, match="incidence_tilt_deg"):
        _with(job_1g, incidence_tilt_deg=35.1).validate()
    with pytest.raises(JobError, match="incidence_tilt_deg"):
        _with(job_1g, incidence_tilt_deg=float("nan")).validate()
    # 합성 각: |pitch| + |tilt| ≤ max_laser_tilt_deg
    with pytest.raises(JobError, match="compose"):
        _with(job_1g, pitch_deg=-8.0, incidence_tilt_deg=24.0).validate()
    # 정반사 대역 −12°±3° 경고
    with pytest.warns(IncidenceTiltWarning, match="specular"):
        _with(job_1g, incidence_tilt_deg=-11.0).validate()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _with(job_1g, incidence_tilt_deg=-8.0).validate()          # 대역 밖: 경고 없음
        _with(job_1g, incidence_tilt_deg=0.0).validate()


def test_3g_tilt_geometry_origin_above_seam_and_looking_down(job_3g, calib_lab):
    """τ = +10: 광학 원점 = 심 점 + standoff·(n cosτ + up sinτ), Z_opt(pitch 0) 가 정확히 심 점을 향한다. 'down'/'up' 모두."""
    for direction in ("down", "up"):
        job = _with(job_3g, incidence_tilt_deg=10.0, pitch_deg=0.0, vertical_direction=direction)
        fr = sensor_frames(job, calib_lab)
        n, up = fr.n, fr.up
        tau = math.radians(10.0)
        np.testing.assert_allclose(fr.d_opt, 0.32 * (n * math.cos(tau) + up * math.sin(tau)), atol=1e-12)
        assert fr.origin_height_offset_m == pytest.approx(0.32 * math.sin(tau))          # ≈ +55.6 mm 위
        for s in (0.0, 0.15, 0.3):
            T = fr.T_base_opt(s)
            o, Z = T[:3, 3], T[:3, 2]
            np.testing.assert_allclose(o, fr.seam_point(s) + fr.d_opt, atol=1e-12)
            np.testing.assert_allclose(o + 0.32 * Z, fr.seam_point(s), atol=1e-9)         # Z 가 심 점을 향함
            assert float(Z @ up) < 0                                                       # 아래를 내려다봄
            assert fr.s_of(o) == pytest.approx(s, abs=1e-12)                               # s 는 심 점 기준
        assert fr.laser_tilt_deg() == pytest.approx(10.0, abs=1e-9)
        # 음수: 올려다봄, 원점이 심 점 아래
        fr_m = sensor_frames(_with(job, incidence_tilt_deg=-10.0), calib_lab)
        assert fr_m.origin_height_offset_m == pytest.approx(-0.32 * math.sin(tau))
        assert float(fr_m.T_base_opt(0.1)[:3, 2] @ up) > 0
    # pitch 와 합성: pitch −5 + tilt +10 (down, progression +1) → 레이저 기울기 |−(−5)… | = 5° (같은 축 둘레 합성)
    fr_c = sensor_frames(_with(job_3g, incidence_tilt_deg=10.0, pitch_deg=-5.0), calib_lab)
    assert fr_c.laser_tilt_deg() == pytest.approx(5.0, abs=1e-9)
    R, phi = incidence_tilt_rotation(_with(job_3g, incidence_tilt_deg=10.0), fr_c.n, fr_c.u_travel, "vertical_3G")
    assert abs(math.degrees(phi)) == pytest.approx(10.0)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_3g_tilt_plus10_keeps_floor_rule_even_at_lower_endpoint(job_3g, calib_lab):
    for direction in ("down", "up"):
        job = _torch_up(_with(job_3g, incidence_tilt_deg=10.0, vertical_direction=direction), calib_lab)
        plan = build_transit_plan(job, calib_lab)
        stats = check_envelope(plan, job, calib_lab)
        # 아래 점(0.3)에서 광학 원점은 floor + standoff·sin10° ≈ 0.3556
        assert stats["floor_margin_min"] == pytest.approx(0.32 * math.sin(math.radians(10.0)), abs=2e-3)
        assert stats["floor_margin_min"] > 0.05
        o = np.stack([w.opt_origin(calib_lab) for w in plan.waypoints if w.floor_rule])
        assert o[:, 2].min() >= 0.3 + 0.05
        assert any("looks DOWNWARD" in n and "55.6 mm" in n for n in plan.notes)
        assert any("confirmed with the operator" in n for n in plan.notes)


def test_3g_tilt_minus10_violates_floor_at_lower_endpoint(job_3g, calib_lab):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncidenceTiltWarning)
        for direction in ("down", "up"):
            job = _torch_up(_with(job_3g, incidence_tilt_deg=-10.0, vertical_direction=direction), calib_lab)
            with pytest.raises(EnvelopeError, match="rule=3G_floor.*optical origin height 0.244") as ei:
                build_transit_plan(job, calib_lab)
            assert "below floor_z=0.3000" in str(ei.value)
        # 바닥은 start/end 중 낮은 점으로 정의되므로 점을 올려도 바닥이 함께 올라간다 → 음의 틸트는 3G 에서 항상 위반
        # (바닥을 따로 낮추는 파라미터는 없음; 필요하면 아래 점을 실제 용접선 끝보다 낮게 티칭해야 한다)
        job_hi = _with(job_3g, incidence_tilt_deg=-10.0, start=[0.5, 0.7, 0.36])
        with pytest.raises(EnvelopeError, match="rule=3G_floor"):
            build_transit_plan(job_hi, calib_lab)
        plan = build_transit_plan(job_hi, calib_lab, check=False)
        assert any("looks UPWARD (origin below the seam point)" in n and "55.6 mm" in n for n in plan.notes)
        with pytest.raises(EnvelopeError, match="rule=3G_floor"):
            check_envelope(plan, job_hi, calib_lab)


def test_1g_tilt_composes_with_pitch_and_keeps_height(job_1g, calib_lab):
    fr0 = sensor_frames(job_1g, calib_lab)                                   # pitch −5, tilt 0
    fr = sensor_frames(_with(job_1g, incidence_tilt_deg=10.0), calib_lab)   # pitch −5 + tilt 10 → |5|
    assert fr0.laser_tilt_deg() == pytest.approx(5.0, abs=1e-9)
    assert fr.laser_tilt_deg() == pytest.approx(5.0, abs=1e-9)
    assert sensor_frames(_with(job_1g, incidence_tilt_deg=5.0), calib_lab).laser_tilt_deg() == pytest.approx(0.0, abs=1e-9)
    assert sensor_frames(_with(job_1g, incidence_tilt_deg=-5.0), calib_lab).laser_tilt_deg() == pytest.approx(10.0, abs=1e-9)
    # 1G: 원점 높이는 그대로 (n ∥ up), 진행방향으로만 standoff·sin(tilt) 이동; s_of 는 심 점 기준으로 보정
    assert fr.origin_height_offset_m == pytest.approx(0.32 * math.cos(math.radians(10.0)))
    assert abs(float(fr.d_opt @ fr.u)) == pytest.approx(0.32 * math.sin(math.radians(10.0)))
    assert fr.s_of(fr.opt_origin(0.2)) == pytest.approx(0.2, abs=1e-12)
    # 계획은 1G 규칙 그대로 통과 (transit/specimen_top 규칙 영향 없음)
    plan = build_transit_plan(_with(job_1g, incidence_tilt_deg=10.0), calib_lab)
    assert plan.orientation == "flat_1G" and check_envelope(plan, _with(job_1g, incidence_tilt_deg=10.0), calib_lab)["max_laser_tilt_deg"] == pytest.approx(5.0, abs=1e-6)
    assert any("(1G)" in n and "composes with pitch" in n for n in plan.notes)
    # 틸트 0 이면 종전과 동일한 계획
    plan0 = build_transit_plan(job_1g, calib_lab)
    for w0, w in zip(plan0.waypoints, build_transit_plan(_with(job_1g, incidence_tilt_deg=0.0), calib_lab).waypoints):
        np.testing.assert_allclose(w0.T_base_flange, w.T_base_flange, atol=1e-12)


def test_reconstruct_s_uses_seam_point_with_tilt(job_1g, calib_lab):
    """틸트가 있어도 재구성 s_m 은 심 점 기준 → 합성 진실값 s 와 일치."""
    job = _with(job_1g, end=[0.6, -0.12, 0.05], lead_in_m=0.02, lead_out_m=0.02, speed_mps=0.05, accel_mps2=0.5,
                incidence_tilt_deg=8.0, pitch_deg=0.0)
    cfg = SynthConfig(n_points=48, seed=4, robot_poll_rate_hz=0.0)
    syn = simulate_scan(job, calib_lab, PlaneScene(h0=0.0), cfg)
    res = reconstruct_scan(syn.frames, syn.robot, calib_lab, job)
    k = res.result_counter
    np.testing.assert_allclose(res.s_m, syn.truth["s_m"][k], atol=1e-4)
    # 프로파일 중앙 z = standoff (Z 가 심 점을 정확히 향함: 원점 높이 standoff·cos8°, 광선 8° → range = standoff)
    zc = np.nanmedian(np.where(res.valid, res.z_mm, np.nan), axis=1)
    assert abs(np.nanmedian(zc) - 320.0) < 0.5
    # pitch 를 더하면(라인축 둘레 순수 회전) 중앙 광선이 심 점을 벗어나 range 가 standoff·cosτ/cos(τ+pitch) 로 바뀐다
    job_p = _with(job, pitch_deg=-5.0)
    syn_p = simulate_scan(job_p, calib_lab, PlaneScene(h0=0.0), cfg)
    res_p = reconstruct_scan(syn_p.frames, syn_p.robot, calib_lab, job_p)
    zc_p = np.nanmedian(np.where(res_p.valid, res_p.z_mm, np.nan), axis=1)
    assert abs(np.nanmedian(zc_p) - 320.0 * math.cos(math.radians(8.0)) / math.cos(math.radians(3.0))) < 0.5
