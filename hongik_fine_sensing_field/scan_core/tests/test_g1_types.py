"""g1_types (G1_CORE_SPEC §8.2, WP0): Tolerances.from_qc on the lab 1G job, LegObs save/load round trip (uncompressed npz + json meta),
SeamEstimate.r(s) / to_dict."""
import json
import os
import zipfile

import numpy as np
import pytest
import yaml

from scan_core.g1_types import (EndEst, LegObs, SeamEstimate, SeamPrior, StationEst, THETA_NAMES, Tolerances, taught_frame)
from scan_core.specimen import GrooveSpec

LAB_JOB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "scan_l0", "scan_l0", "jobs", "groove_10T_1G.yaml")


def _lab_qc():
    with open(LAB_JOB, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["qc_targets"]


def test_from_qc_lab_job():
    t = Tolerances.from_qc(_lab_qc())
    assert t.root_position_mm == 1.0
    assert t.end_position_mm == 1.0            # default = root_position (§7 #12)
    assert (t.gap_mm, t.depth_mm, t.bevel_deg, t.opening_mm, t.length_mm) == (0.5, 0.5, 2.0, 0.5, 1.0)
    assert t.for_dim("hi_lo_mm") is None and t.for_dim("root_face_mm") is None


def test_from_qc_end_override_and_required_root():
    qc = dict(_lab_qc(), end_position_mm={"tol": 1.5})
    assert Tolerances.from_qc(qc).end_position_mm == 1.5
    qc.pop("root_position_mm")
    with pytest.raises(ValueError):
        Tolerances.from_qc(qc)
    t = Tolerances.from_qc({"root_position_mm": {"tol": 1.0}})
    assert t.gap_mm is None and t.length_mm is None


def _leg(n=7, seed=0):
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, 3] = rng.normal(size=(n, 3))
    A = rng.normal(size=(n, 10, 10))
    return LegObs(name="01_coarse", cls="coarse", direction=-1, t_acq_s=np.linspace(0, 1, n), T_base_opt=T,
                  v_opt_mps=rng.normal(size=(n, 3)), theta=rng.normal(size=(n, 10)), cov=A @ A.transpose(0, 2, 1),
                  log_bf=rng.normal(size=n), sigma_r_mm=rng.random(n), eps=rng.random(n), n_valid=rng.integers(0, 1408, n),
                  converged=rng.random(n) > 0.5, bevel_x_sign=-1,
                  meta={"speed_mps": 0.03, "pitch_deg": np.float64(-5.0), "files": {"raw": "raw.npz"}, "nan": float("nan")})


def test_legobs_roundtrip(tmp_path):
    leg = _leg()
    p = tmp_path / "legs" / "01_coarse" / "profiles.npz"
    leg.save(p)
    assert sorted(os.listdir(p.parent)) == ["profiles.npz"]          # atomic: no tmp left
    with zipfile.ZipFile(p) as z:                                        # uncompressed
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
    back = LegObs.load(p)
    assert (back.name, back.cls, back.direction, back.bevel_x_sign) == ("01_coarse", "coarse", -1, -1)
    for k in ("t_acq_s", "T_base_opt", "v_opt_mps", "theta", "cov", "log_bf", "sigma_r_mm", "eps", "n_valid", "converged"):
        np.testing.assert_array_equal(getattr(back, k), getattr(leg, k))
    assert back.converged.dtype == bool and back.n_valid.dtype == np.int64
    assert back.meta == {"speed_mps": 0.03, "pitch_deg": -5.0, "files": {"raw": "raw.npz"}, "nan": None}
    assert back.n == 7 and len(THETA_NAMES) == 10


def test_legobs_shape_check():
    leg = _leg()
    with pytest.raises(ValueError):
        LegObs(**{**leg.__dict__, "theta": leg.theta[:, :9]})
    with pytest.raises(ValueError):
        LegObs(**{**leg.__dict__, "cls": "bogus"})
    for c in ("dwell", "transit", "pass"):                       # spec r2 §3.6a classes (+ old aliases coarse / fine)
        assert LegObs(**{**leg.__dict__, "cls": c}).cls == c
    assert LegObs(**{**leg.__dict__, "cls": "dwell", "direction": 0}).direction == 0
    with pytest.raises(ValueError):
        LegObs(**{**leg.__dict__, "cls": "pass", "direction": 0})   # direction 0 only for a dwell


def _prior():
    spec = GrooveSpec()
    return SeamPrior(S0=[0.9, -0.2, 0.05], E0=[0.9, 0.12, 0.06], up=[0, 0, 1], spec=spec, tol=Tolerances.from_qc(_lab_qc()),
                     k=3.0, dt_s=0.0037, dt_sigma_s=0.001, rep_sigma_m=5e-5, he_sigma_m=0.002, sigma_eps=0.0,
                     station_step_m=0.005, station_window_m=0.010)


def test_taught_frame_and_prior():
    pr = _prior()
    f = pr.frame0()
    np.testing.assert_allclose(f["u0"], [0, 1, 0], atol=1e-12)      # E0 − S0 projected ⟂ up
    np.testing.assert_allclose(f["w0"], [-1, 0, 0], atol=1e-12)     # n0 × u0 (left looking S->E)
    with pytest.raises(ValueError):
        taught_frame([0, 0, 0], [0, 0, 1], [0, 0, 1])                 # S0 = E0 in plane
    d = pr.to_dict()
    json.dumps(d)
    assert d["tol"]["root_position_mm"] == 1.0 and d["spec"]["thickness_mm"] == 10.0


def _est(prior, model="line", params=None):
    f = prior.frame0()
    params = np.array([[0.002, 0.01], [-0.010, 0.0]]) if params is None else params
    m = params.shape[1]
    cov = np.diag(np.r_[np.full(m, 1e-8), np.full(m, 4e-8)])
    ends = {"S": EndEst(0.02, 3e-4, np.zeros(3), True, 2, [{"leg": "02_fine"}]),
            "E": EndEst(0.40, 3e-4, np.ones(3), False, 1, [])}
    st = StationEst(s_m=0.02, root_m=np.zeros(3), root_sigma_m=(1e-4, 2e-4), u=f["u0"], n=f["n0"], w=f["w0"],
                    dims={"gap_mm": {"value": 5.0, "sigma": 0.05, "sigma_sys": 0.0, "n": 30}},
                    faces={"bevel_left": {"normal": [0, 0.5, 0.86], "point_m": [0, 0, 0], "line_wn_mm": [0, 0, 1, 0],
                                          "sigma_deg": 0.3, "n": 30}}, n_profiles=30)
    return SeamEstimate(groove_found=True, log_bf_run=1234.0, line_model=model, line_params=params, line_cov=cov,
                        u=f["u0"], n_plate=f["n0"], plate={"roll_deg": 0.1}, ends=ends, dt_s=0.0037, dt_sigma_s=0.001,
                        class_offset_m={"coarse": np.zeros(3), "fine": np.zeros(3)}, leg_eta_m={}, sigma_zeta_m={"fine": 1e-4},
                        tau_end_m=float("inf"), bevel_sign_w=1, stations=[st], ratios={"gap_mm": 0.3}, done=False,
                        binding=["end_S"], unresolved=["end_S"], frame0=f)


def test_estimate_r_and_to_dict():
    pr = _prior()
    f = pr.frame0()
    est = _est(pr)
    s = np.array([0.0, 0.1, 0.38])
    r = est.r(s)
    exp = f["S0"] + s[:, None] * f["u0"] + (0.002 + 0.01 * s)[:, None] * f["w0"] - 0.010 * f["n0"]
    np.testing.assert_allclose(r, exp, atol=1e-15)
    np.testing.assert_allclose(est.r(0.1), exp[1], atol=1e-15)
    t = est.tangent(0.2)
    np.testing.assert_allclose(t, (f["u0"] + 0.01 * f["w0"]) / np.hypot(1, 0.01), atol=1e-15)
    sl, sn = est.r_sigma(np.array([0.0, 1.0]))
    np.testing.assert_allclose(sl, [1e-4, np.sqrt(2e-8)])
    np.testing.assert_allclose(sn, [2e-4, np.sqrt(8e-8)])
    d = est.to_dict()
    txt = json.dumps(d)
    assert d["ends"]["S"]["two_sided"] is True and d["tau_end_m"] is None and d["stations"][0]["dims"]["gap_mm"]["n"] == 30
    assert d["line_params"] == [[0.002, 0.01], [-0.01, 0.0]] and "frame0" in json.loads(txt)


def test_estimate_quadratic_and_needs_frame():
    pr = _prior()
    est = _est(pr, "quadratic", np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]))
    f = pr.frame0()
    np.testing.assert_allclose(est.r(0.2), f["S0"] + 0.2 * f["u0"] + 0.5 * 0.04 * f["w0"], atol=1e-15)
    np.testing.assert_allclose(est.tangent(0.2), (f["u0"] + 0.2 * f["w0"]) / np.hypot(1, 0.2), atol=1e-15)
    est.frame0 = None
    with pytest.raises(ValueError):
        est.r(0.1)
    with pytest.raises(ValueError):
        _est(pr, "cubic")
