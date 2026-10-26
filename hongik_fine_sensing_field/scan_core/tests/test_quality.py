import numpy as np

from scan_core.quality import integrity_facts, integrity_report, ransac_plane
from scan_core.reconstruct import assemble, reconstruct_scan
from scan_core.synth import PlaneScene, SynthConfig, simulate_scan
from scan_core.types import ScanJob


def test_ransac_plane():
    rng = np.random.default_rng(0)
    P = np.column_stack([rng.uniform(0, 1, 3000), rng.uniform(0, 1, 3000), rng.normal(0, 0.02e-3, 3000) + 0.1])
    P[:100, 2] += 0.02                            # 아웃라이어
    pl = ransac_plane(P, inlier_mm=0.3)
    assert np.allclose(pl["normal"], [0, 0, 1], atol=1e-3) and pl["rms_mm"] < 0.03 and pl["inlier_ratio"] > 0.95
    pa = ransac_plane(P)                          # 적응 문턱 (LMedS σ, 하한 센서 양자화)
    assert np.allclose(pa["normal"], [0, 0, 1], atol=1e-3) and pa["inlier_ratio"] > 0.95 and np.isfinite(pa["sigma_mm"])


def test_integrity_facts_on_synthetic_scan(job_1g, calib_lab):
    """4d P2: 구간마다 남는 것은 무결성 사실뿐 (판정 없음, verdict 'UNJUDGED'). 재구성 자체의 정확도도 여기서 본다."""
    d = job_1g.to_dict()
    d.update(dict(end=[0.6, -0.05, 0.05], lead_in_m=0.02, lead_out_m=0.02, speed_mps=0.05, accel_mps2=0.5))
    job = ScanJob.from_dict(d)
    cfg = SynthConfig(n_points=96, sigma_z_mm=0.01, rx_latency_s=0.004, rx_jitter_scale_s=0.0002,
                      robot_poll_rate_hz=500.0, seed=11)
    syn = simulate_scan(job, calib_lab, PlaneScene(h0=0.0), cfg)
    calib_lab.dt_s = cfg.rx_latency_s
    res = reconstruct_scan(syn.frames, syn.robot, calib_lab, job)
    m = integrity_facts(res, auto_trigger_error=0)
    assert m["frames_dropped"] == 0 and m["clock_locked"] and m["n_points"] == 96 and not m["n_points_changed"]
    assert m["ts_integrity_violations"] == 0 and m["unsynced_frames"] == 0 and m["n_frames"] == res.n_profiles
    qc = integrity_report(res, auto_trigger_error=2)
    assert qc.verdict == "UNJUDGED" and qc.failures == [] and qc.metrics["auto_trigger_error"] == 2
    v = res.speed_mps[res.in_cruise]
    v = v[np.isfinite(v)]
    assert abs(float(v.mean()) - 0.05) / 0.05 < 0.03
    pts, _, _, _ = assemble(res, only_cruise=True, only_valid=True)
    pl = ransac_plane(pts, inlier_mm=0.3)
    assert pl["rms_mm"] < 0.03 and abs(abs(float(pl["normal"][2])) - 1.0) < 1e-3
