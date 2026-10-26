import numpy as np
import pytest

from scan_core.frames import transform_points, inv_transform
from scan_core.path_gen import sensor_frames
from scan_core.reconstruct import assemble, cruise_mask, measured_speed, profile_to_base, profiles_to_base, reconstruct_scan
from scan_core.synth import PlaneScene, SynthConfig, TrapezoidProfile, simulate_scan
from scan_core.types import ScanJob


def _short_job(job_1g, **kw):
    d = job_1g.to_dict()
    d.update(dict(end=[0.6, -0.1, 0.05], lead_in_m=0.02, lead_out_m=0.02, speed_mps=0.05, accel_mps2=0.5))
    d.update(kw)
    return ScanJob.from_dict(d)


def test_profile_to_base_matches_vectorized(calib_lab):
    rng = np.random.default_rng(1)
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.1, 0.6]
    x = rng.uniform(-70, 70, 50)
    z = rng.uniform(250, 500, 50)
    P1 = profile_to_base(T, calib_lab.T_flange_opt, x, z)
    P2 = profiles_to_base(T[None], calib_lab.T_flange_opt, x[None], z[None])[0]
    assert np.allclose(P1, P2)
    # 손 계산: T·T_fo·[x,0,z]
    Tfo = T @ calib_lab.T_flange_opt
    ref = (np.stack([x, 0 * x, z], -1) * 1e-3) @ Tfo[:3, :3].T + Tfo[:3, 3]
    assert np.allclose(P1, ref)


def test_trapezoid_profile():
    p = TrapezoidProfile(-0.02, 0.12, 0.05, 0.5)
    assert p.s(0) == pytest.approx(-0.02) and p.s(p.duration) == pytest.approx(0.12)
    t = np.linspace(0, p.duration, 200)
    s = p.s(t)
    assert np.all(np.diff(s) >= -1e-12)
    assert p.v_of(p.duration / 2) == pytest.approx(0.05)


def test_measured_speed_and_cruise_mask():
    p = TrapezoidProfile(0.0, 0.3, 0.05, 0.5)
    t = np.arange(0, p.duration, 0.01)
    P = np.zeros((t.size, 3))
    P[:, 1] = p.s(t)
    v = measured_speed(t, P, window_s=0.2)
    m, vref = cruise_mask(v, tol_pct=5.0)
    assert abs(vref - 0.05) < 1e-3
    assert np.all(np.abs(v[m] - 0.05) / 0.05 < 0.05)
    assert m.sum() > 0.5 * t.size


def test_reconstruct_noise_free_plane_error_below_1e5(job_1g, calib_lab):
    job = _short_job(job_1g)
    cfg = SynthConfig(n_points=128, sigma_z_mm=0.0, ts_quant_ms=1.0, rx_latency_s=0.004, rx_jitter_scale_s=0.0,
                      robot_poll_rate_hz=0.0, drift_ppm=0.0, seed=3)
    scene = PlaneScene(h0=0.0)
    syn = simulate_scan(job, calib_lab, scene, cfg)
    calib = calib_lab
    calib.dt_s = cfg.rx_latency_s       # dt 진실값 적용
    res = reconstruct_scan(syn.frames, syn.robot, calib, job)
    assert res.n_profiles == len(syn.frames) and res.n_points == 128
    assert res.meta["clock"]["locked"]
    # 스탬프 오차
    kmap = {int(k): i for i, k in enumerate(syn.truth["k"])}
    t_true = np.array([syn.truth["t_acq_s"][kmap[int(k)]] for k in res.result_counter])
    assert np.max(np.abs(res.t_acq_s - t_true)) < 1e-6
    # 유효점은 모두 판 위 (z=0.05 평면)
    pts, pi, ci, flags = assemble(res, only_cruise=True, only_valid=True)
    assert pts.shape[0] > 1000
    h = pts[:, 2] - 0.05
    assert np.max(np.abs(h)) < 1e-5
    # 순항 프레임 존재, 진행좌표 [0,L] 커버
    assert res.in_cruise.sum() > 0
    s = res.s_m[res.in_cruise]
    assert s.min() < 0.0 + 0.005 and s.max() > job.length_m - 0.005
    assert np.all(~res.unsynced[res.in_cruise])
    # 광학 원점 진실 대비
    T_true = np.array([syn.truth["T_base_opt"][kmap[int(k)]] for k in res.result_counter])
    d = np.linalg.norm(res.T_base_opt[:, :3, 3] - T_true[:, :3, 3], axis=1)
    assert np.max(d[res.in_cruise]) < 1e-6


def test_reconstruct_with_noise_stays_within_budget(job_1g, calib_lab):
    job = _short_job(job_1g)
    cfg = SynthConfig(n_points=128, sigma_z_mm=0.01, ts_quant_ms=1.0, rx_latency_s=0.005, rx_jitter_scale_s=0.0003,
                      robot_poll_rate_hz=500.0, drift_ppm=30.0, dropout_prob=0.02, frame_drop_prob=0.0, seed=5)
    syn = simulate_scan(job, calib_lab, PlaneScene(h0=0.0), cfg)
    calib = calib_lab
    calib.dt_s = cfg.rx_latency_s
    res = reconstruct_scan(syn.frames, syn.robot, calib, job)
    pts, *_ = assemble(res, only_cruise=True, only_valid=True)
    h = pts[:, 2] - 0.05
    assert np.sqrt(np.mean(h ** 2)) < 0.05e-3
    assert 0.9 < res.valid.mean() < 1.0
    assert abs(res.ds_mm - 0.5) < 0.05                       # 50 mm/s @ 100 Hz


def test_frame_drop_and_sentinel_handling(job_1g, calib_lab):
    job = _short_job(job_1g, standoff_m=0.26, safe_clearance_m=0.05)   # 하한 233 근접 → 일부 열 센티널 가능성 (토치 팁 ≥ 한계)
    cfg = SynthConfig(n_points=64, frame_drop_prob=0.05, seed=7, robot_poll_rate_hz=0.0)
    syn = simulate_scan(job, calib_lab, PlaneScene(h0=0.0), cfg)
    res = reconstruct_scan(syn.frames, syn.robot, calib_lab, job)
    from scan_core.clock_sync import counter_gaps
    assert counter_gaps(res.result_counter) > 0
    assert res.meta["clock"]["locked"]
    # 센티널 z=0 은 무효
    z = res.z_mm
    assert np.all(~res.valid[z == 0.0])


def test_reconstruct_survives_a_sensor_stall_and_a_host_receive_backlog(job_1g, calib_lab):
    """실기 at10 (2026-10-03) 재현: 녹화 초반 계획 계산으로 호스트가 0.23 s 수신을 못 하고 (프레임이 한꺼번에 도착) 그 직후 센서가
    28 ms 촬영을 멈춘다. 이전 모델 (k 만, OLS 기울기) 은 멈춘 뒤 모든 프레임을 28 ms 틀린 시각에 두고 (30 mm/s 에서 0.84 mm) 무결성
    플래그로 전부 순항에서 뺐으며, 수신 막힘이 OLS 기울기를 수천 ppm 끌었다. 지금: 멈춤 보정 + 하한 포락선 기울기."""
    job = _short_job(job_1g, end=[0.85, -0.1, 0.05], speed_mps=0.03, accel_mps2=0.1)
    cfg = SynthConfig(n_points=64, sigma_z_mm=0.0, ts_quant_ms=1.0, rx_latency_s=0.004, rx_jitter_scale_s=0.002,
                      robot_poll_rate_hz=0.0, drift_ppm=20.0, seed=11, pre_roll_s=1.0,
                      stalls=((89, 28.0),), rx_stalls_s=((0.66, 0.89),))
    syn = simulate_scan(job, calib_lab, PlaneScene(h0=0.0), cfg)
    calib = calib_lab
    calib.dt_s = cfg.rx_latency_s
    res = reconstruct_scan(syn.frames, syn.robot, calib, job)
    st = res.meta["sensor_stalls"]
    assert st["n_stalls"] >= 1 and abs(st["total_stall_ms"] - 28.0) <= 1.0 and st["n_flagged"] == 1
    assert abs(res.meta["clock"]["slope_ppm"] - 20.0) < 60.0 and res.meta["clock"]["slope_method"] == "envelope"
    assert abs(res.meta["clock"]["slope_ppm_ols"]) > 300.0             # 이 자료에서 OLS 는 크게 틀린다 (진단으로 남음)
    kmap = {int(k): i for i, k in enumerate(syn.truth["k"])}
    t_true = np.array([syn.truth["t_acq_s"][kmap[int(k)]] for k in res.result_counter])
    assert np.max(np.abs(res.t_acq_s - t_true)) < 2.0e-3
    T_true = np.array([syn.truth["T_base_opt"][kmap[int(k)]] for k in res.result_counter])
    d = np.linalg.norm(res.T_base_opt[:, :3, 3] - T_true[:, :3, 3], axis=1)
    assert np.max(d[res.in_cruise]) < 0.06e-3                            # 30 mm/s × 2 ms
    assert res.in_cruise.sum() > 0.8 * np.sum(res.speed_mps > 0.0285)     # 멈춤 뒤 프레임도 순항으로 쓴다
