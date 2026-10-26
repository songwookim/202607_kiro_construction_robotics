import numpy as np
import pytest

from scan_core.calib.hysteresis import (PassEdge, edge_position_from_result, estimate_dt_hysteresis, fit_dt,
                                        inclined_edge_crossing, step_edge_y_in_profile)
from scan_core.reconstruct import reconstruct_scan
from scan_core.synth import StepScene, SynthConfig, simulate_scan
from scan_core.types import ScanJob


@pytest.mark.parametrize("seed", range(10))
def test_dt_recovery_within_0p3ms(seed):
    """3 속도(20/40/80 mm/s) × 양방향 × 10 반복, 에지 위치 노이즈 σ 0.02 mm, 백래시 c=0.05 mm."""
    rng = np.random.default_rng(seed)
    dt_true = rng.uniform(-0.015, 0.02)
    c_true = 0.05e-3
    s0 = 0.123
    passes = []
    for v in (0.02, 0.04, 0.08):
        for d in (+1, -1):
            for _ in range(10):
                s = s0 + d * (v * dt_true + c_true / 2) + rng.normal(0, 0.02e-3)
                passes.append(PassEdge(v_mps=v, direction=d, s_edge_m=s))
    r = estimate_dt_hysteresis(passes)
    assert abs(r.dt_s - dt_true) < 0.3e-3
    assert abs(r.c_m - c_true) < 0.05e-3
    assert r.sigma_dt_s < 0.3e-3 and r.n_speeds == 3 and r.n_passes == 60
    dt2, c2, sdt, sc = fit_dt([v for v, _ in r.deltas], [dl for _, dl in r.deltas])
    assert abs(dt2 - dt_true) < 0.3e-3


def test_fit_dt_two_speeds_no_sigma():
    dt, c, sdt, sc = fit_dt([0.02, 0.08], [2 * 0.02 * 0.01 + 1e-4, 2 * 0.08 * 0.01 + 1e-4])
    assert dt == pytest.approx(0.01) and c == pytest.approx(1e-4) and np.isnan(sdt)


def test_step_edge_detection_subsample():
    y = np.linspace(-0.05, 0.05, 201)
    h = np.where(y > 0.0123, 0.005, 0.0)
    ye = step_edge_y_in_profile(y, h)
    assert abs(ye - 0.0123) < 0.0006          # 샘플 간격 0.5 mm 의 중간
    s = np.linspace(0, 0.1, 50)
    y_edge = 0.02 + np.tan(np.radians(35)) * (s - 0.04)      # 에지가 y=0.02 를 s=0.04 에서 지남
    sc, m, rms = inclined_edge_crossing(s, y_edge, y_ref=0.02)
    assert sc == pytest.approx(0.04, abs=1e-9)


def test_end_to_end_hysteresis_on_inclined_step(job_1g, calib_lab):
    """기울인 단차(35°, 5 mm) 를 같은 툴 자세로 왕복 스캔(dt 진실 8 ms) → dt=0 으로 재구성 → 히스테리시스로 dt 회수."""
    dt_true = 0.008
    passes = []
    scene = StepScene(h_high=0.005, x0=0.05, angle_deg=35.0)   # 판 프레임(진행 x): 에지 x = 0.05 + tan35·y
    for v in (0.02, 0.04, 0.08):
        d = job_1g.to_dict()
        d.update(dict(end=[0.6, -0.1, 0.05], lead_in_m=0.02, lead_out_m=0.02, speed_mps=v, accel_mps2=0.5,
                      pitch_deg=-3.0, bidirectional=True))
        job = ScanJob.from_dict(d)
        cfg = SynthConfig(n_points=160, sigma_z_mm=0.005, rx_latency_s=dt_true, rx_jitter_scale_s=0.0001,
                          robot_poll_rate_hz=0.0, seed=int(v * 1000), z_grid_mm=(200.0, 450.0, 2.0), dwell_s=0.4)
        syn = simulate_scan(job, calib_lab, scene, cfg)
        calib_lab.dt_s = 0.0
        res = reconstruct_scan(syn.frames, syn.robot, calib_lab, job)
        (t0, d0), (t1, d1) = syn.truth["segments"]
        t_split = 0.5 * (t0 + d0 + t1)
        for direction, mask in ((+1, res.t_rx_s < t_split), (-1, res.t_rx_s >= t_split)):
            e = edge_position_from_result(res, y_ref=0.0, min_step_m=0.002, frame_mask=mask)
            assert e["n_profiles"] > 20 and np.isfinite(e["s_edge_m"]), (v, direction, e["n_profiles"])
            v_meas = float(np.mean(res.speed_mps[res.in_cruise & mask]))
            assert abs(v_meas - v) / v < 0.03
            passes.append(PassEdge(v_mps=v_meas, direction=direction, s_edge_m=e["s_edge_m"]))
    r = estimate_dt_hysteresis(passes)
    assert abs(r.dt_s - dt_true) < 0.3e-3, r.to_dict()
    assert abs(r.c_m) < 0.05e-3, r.to_dict()
    assert abs(r.s0_m - 0.05) < 0.3e-3
