import numpy as np
import pytest

from scan_core.io_npz import load_scan_result, save_scan_result, save_cloud_ply, has_h5py
from scan_core.quality import integrity_report
from scan_core.reconstruct import assemble, reconstruct_scan
from scan_core.synth import (CompositeScene, GaugeBlockScene, PlaneScene, SphereScene, StepScene, SynthConfig,
                             simulate_profile, simulate_scan)
from scan_core.types import Calib, ScanJob
from scan_core.validity import column_monotonic_mask, dX_mm, valid_mask


def test_validity_mask_and_dx():
    x = np.array([0.0, 1.0, np.nan, 2.0, 3.0])
    z = np.array([300.0, 0.0, 300.0, 600.0, 233.1])
    assert valid_mask(x, z).tolist() == [True, False, False, False, False]
    assert dX_mm(250.0) == pytest.approx(0.082) and dX_mm(525.0) == pytest.approx(0.155)
    x2 = np.array([0.0, 0.1, 0.2, 5.0, 0.4])
    z2 = np.full(5, 320.0)
    m = column_monotonic_mask(x2, z2, k=5.0)
    assert m.tolist() == [True, True, False, False, False]


def test_scenes_heights():
    xp = np.array([0.0, 0.03, -0.03, 0.2])
    yp = np.zeros(4)
    assert np.allclose(PlaneScene(h0=0.01, x_range=(-0.1, 0.1)).height(xp, yp)[:3], 0.01)
    assert np.isnan(PlaneScene(x_range=(-0.1, 0.1)).height(xp, yp)[3])
    sp = SphereScene(xc=0.0, yc=0.0, radius=0.02)
    assert sp.height(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.04)
    st = StepScene(h_high=0.005, x0=0.01, angle_deg=0.0)
    assert st.height(xp, yp).tolist() == [0.0, 0.005, 0.0, 0.005]
    gb = GaugeBlockScene(xc=0.0, yc=0.0, sx=0.02, sy=0.02, h=0.01)
    assert gb.height(xp, yp).tolist() == [0.01, 0.0, 0.0, 0.0]
    cs = CompositeScene([PlaneScene(h0=0.0), gb])
    assert cs.height(xp, yp).tolist() == [0.01, 0.0, 0.0, 0.0]


def test_simulate_profile_plane_geometry(calib_lab):
    """수평 판, 센서 320 mm 위, 피치 0 → 모든 열 z ≈ 320, x 는 k·dX(320)."""
    from scan_core.frames import make_transform
    from scan_core.path_gen import plate_frame
    T_plate = plate_frame([0, 0, 0], [0, 1, 0], [0, 0, 1])
    R = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]])          # Z_opt 아래
    T_bo = make_transform(R, (0, 0, 0.32))
    cfg = SynthConfig(n_points=64, seed=1)
    x, z = simulate_profile(T_bo, PlaneScene(h0=0.0), T_plate, cfg)
    assert np.allclose(z, 320.0, atol=1e-6)
    k = (np.arange(64) - 31.5) * (1408 / 64)          # 열 서브샘플링: FOV 유지
    assert np.allclose(x, k * dX_mm(320.0), atol=1e-6)
    # 교정범위 밖 (z=600) → z=0 센티널, x 유효
    x2, z2 = simulate_profile(make_transform(R, (0, 0, 0.6)), PlaneScene(h0=0.0), T_plate, cfg)
    assert np.all(z2 == 0.0) and np.all(np.isfinite(x2))
    assert not valid_mask(x2, z2).any()


def test_save_load_roundtrip(tmp_path, job_1g, calib_lab):
    d = job_1g.to_dict()
    d.update(dict(end=[0.6, -0.12, 0.05], lead_in_m=0.02, lead_out_m=0.02, speed_mps=0.05, accel_mps2=0.5))
    job = ScanJob.from_dict(d)
    syn = simulate_scan(job, calib_lab, PlaneScene(h0=0.0), SynthConfig(n_points=48, seed=2, robot_poll_rate_hz=0.0))
    res = reconstruct_scan(syn.frames, syn.robot, calib_lab, job)
    res.qc = integrity_report(res)
    out = save_scan_result(tmp_path / "scan1", res, extra_meta={"operator": "test"}, h5=True)
    res2 = load_scan_result(out)
    for k, v in res.array_fields().items():
        assert np.array_equal(np.asarray(v), np.asarray(getattr(res2, k)), equal_nan=True), k
    assert res2.calib is not None and np.allclose(res2.calib.T_flange_opt, calib_lab.T_flange_opt)
    assert res2.qc.verdict == res.qc.verdict == "UNJUDGED" and res2.qc.metrics["n_frames"] == res.qc.metrics["n_frames"]
    assert res2.meta["clock"]["locked"]
    if has_h5py():
        assert (out / "raw.h5").exists()
    pts, pi, ci, fl = assemble(res)
    save_cloud_ply(tmp_path / "cloud.ply", pts, pi, ci, fl)
    assert (tmp_path / "cloud.ply").stat().st_size > 100


def test_calib_yaml_roundtrip(tmp_path):
    c = Calib.lab_initial(eoat_yaw_rad=-np.pi / 2, dt_s=0.0071, x_sign=-1, date="2026-09-30")
    p = tmp_path / "eoat_lab_01.yaml"
    c.save_yaml(p)
    c2 = Calib.load_yaml(p)
    assert np.allclose(c2.T_flange_opt, c.T_flange_opt) and c2.dt_s == 0.0071 and c2.x_sign == -1
    assert np.allclose(c2.T_flange_eoat, c.T_flange_eoat)
    assert np.allclose(c2.p_opt_eoat, [0.012294, 0.0, 0.147], atol=1e-9)
