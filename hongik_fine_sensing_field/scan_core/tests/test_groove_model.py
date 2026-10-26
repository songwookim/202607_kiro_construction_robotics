"""groove_model (G1_CORE_SPEC §3, WP1): synthetic profiles from scan_core.synth (heightmap renderer, independent of the template
ray cast) with known truth — oblique cut (ψ), pitch (φ, with the ψ·φ shear), plate roll, hi-lo, occlusion, outliers (reflection
band), missing data; evidence λ sign on no-groove views (plate, table, plate end, objects) and on a view partly beyond the plate end;
leg fit (bevel side by Bayes factor, LegObs, base-frame outputs)."""
import math

import numpy as np
import pytest

from scan_core import groove_model as gm
from scan_core.g1_types import LegObs, Tolerances
from scan_core.sensor_model import VC_NANO_3DZ_LARGE
from scan_core.specimen import GrooveSpec
from scan_core.synth import CompositeScene, GaugeBlockScene, GrooveScene, PlaneScene, StepScene, SynthConfig, simulate_profile
from scan_core.validity import valid_mask

TOL = Tolerances(root_position_mm=1.0, end_position_mm=1.0, gap_mm=0.5, depth_mm=0.5, bevel_deg=2.0, opening_mm=0.5,
                 length_mm=1.0)
SPEC = GrooveSpec(plate_width_mm=600.0)
NAMES = gm.VERTEX_NAMES["single_bevel"]


# ---------------------------------------------------------------------------------------------------------------- helpers
def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _sensor(standoff=444.0, psi=0.0, phi=0.0, x=0.0, y=0.0):
    X, Z = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    Rz = _rot([0, 0, 1], math.radians(psi))
    X, Z = Rz @ X, Rz @ Z
    Z = _rot(X, math.radians(phi)) @ Z
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = X, np.cross(Z, X), Z
    T[:3, 3] = np.array([x, y, standoff]) * 1e-3
    return T


def _plate(roll=0.0):
    T = np.eye(4)
    T[:3, :3] = _rot([1, 0, 0], math.radians(roll))
    return T


def _truth(spec, T_bo, T_bp):
    """specimen edge lines (along plate x) ∩ laser plane → sensor (x, z) [mm] in VERTEX_NAMES order."""
    t, G, rf, hl = spec.thickness_mm, spec.root_gap_mm, spec.root_face_mm, spec.hi_lo_mm
    tb = math.tan(math.radians(spec.bevel_angle_deg))
    pts = [(-(G / 2 + (t - rf) * tb), 0.0), (-G / 2, -t + rf), (-G / 2, -t), (G / 2, -t), (G / 2, hl)]
    o, X, Y, Z = T_bo[:3, 3], T_bo[:3, 0], T_bo[:3, 1], T_bo[:3, 2]
    d = T_bp[:3, :3] @ np.array([1.0, 0.0, 0.0])
    out = []
    for lat, h in pts:
        p0 = T_bp[:3, :3] @ np.array([0.0, lat * 1e-3, h * 1e-3]) + T_bp[:3, 3]
        p = p0 - float(Y @ (p0 - o)) / float(Y @ d) * d
        out.append(np.array([float(X @ (p - o)), float(Z @ (p - o))]) * 1e3)
    return np.array(out)


def _render(scene, T_bo, T_bp=None, *, sigma=0.03, seed=0, z_lo=None, z_hi=None, **cfg_kw):
    """synth renderer on a fine local z grid (the default 2 mm bracket misses thin occluding wedges under roll)."""
    T_bp = np.eye(4) if T_bp is None else T_bp
    so = float(T_bo[2, 3]) * 1e3
    cfg = SynthConfig(sigma_z_mm=sigma, seed=seed, z_grid_mm=(z_lo or so - 30.0, z_hi or so + 70.0, 0.05), bisect_iters=20,
                      **cfg_kw)
    x, z = simulate_profile(T_bo, scene, T_bp, cfg, np.random.default_rng(seed))
    x, z = x.astype(float), z.astype(float)
    return x, z, valid_mask(x, z)


def _case(psi=0.0, phi=0.0, roll=0.0, hi_lo=0.0, seed=0, **kw):
    spec = GrooveSpec(plate_width_mm=600.0, hi_lo_mm=hi_lo)
    y = float(np.random.default_rng(1000 + seed).uniform(-2.0, 2.0))         # random sub-column phase
    T_bo, T_bp = _sensor(444.0, psi, phi, y=y), _plate(roll)
    x, z, v = _render(GrooveScene(spec=spec, y_range=(-0.3, 0.3)), T_bo, T_bp, seed=seed, **kw)
    cg = gm.cut_geometry(T_bo, T_bp[:3, 0], T_bp[:3, 2])
    prior = gm.profile_prior(SPEC, TOL, c=cg["c"], phi_rad=cg["phi_rad"], shear=cg["shear"])
    tpl = gm.Template(SPEC, -1, shear=cg["shear"])       # bevel plate on plate −y = sensor −x
    return x, z, v, tpl, prior, _truth(spec, T_bo, T_bp), cg


# ---------------------------------------------------------------------------------------------------------------- sensor law
def test_rayfan_law_matches_sensor_model_and_columns_meet_at_the_emitter():
    fan = gm.RayFan.from_validity()
    assert fan.z0 == pytest.approx(-58.9, abs=0.2)
    z = np.linspace(VC_NANO_3DZ_LARGE.z_nominal_mm[0], VC_NANO_3DZ_LARGE.z_nominal_mm[1], 50)
    assert np.max(np.abs(fan.fov_mm(z) - VC_NANO_3DZ_LARGE.fov_x_mm(z))) <= 1.0          # §3.1: one law, ≤ 1 mm
    k = 300.0                                                                                 # a column index offset
    zz = np.array([250.0, 400.0, 540.0])
    xx = k * fan.dx_mm(zz)
    assert np.allclose(xx / (zz - fan.z0), xx[0] / (zz[0] - fan.z0))                         # straight line through o


@pytest.mark.parametrize("kind", ["single_bevel", "single_v", "square"])
@pytest.mark.parametrize("sign", [1, -1])
def test_template_jacobians_match_finite_differences(kind, sign):
    tpl = gm.Template(GrooveSpec(kind=kind), sign, shear=0.02)
    th = np.array([1.3, 450.0, 0.05, 5.2, 0.52, 0.4, 10.2, 10.9, 0.03, -0.04])
    if kind == "square":
        th[gm.I_BE] = th[gm.I_RF] = 0.0
    g = tpl.geometry(th)
    for i in range(gm.N_THETA):
        h = 1e-6
        tp, tm = th.copy(), th.copy()
        tp[i] += h
        tm[i] -= h
        gp, gn = tpl.geometry(tp), tpl.geometry(tm)
        for key, dkey in (("V", "dV"), ("A", "dA"), ("n", "dn"), ("u", "du")):
            assert np.allclose((gp[key] - gn[key]) / (2 * h), g[dkey][..., i], atol=1e-6), (key, i)
    assert np.all(g["n"][[0, -1], 1] < 0)                   # plate-top normals point to the sensor (−z)


def test_cast_first_hit_occlusion_and_open_gap():
    fan = gm.RayFan.from_validity()
    spec = GrooveSpec(plate_width_mm=600.0)
    th = np.array([0.0, 454.0, 0.0, 5.0, math.radians(30), 0.0, 10.0, 10.0, 0.0, 0.08])   # R plate rolled: overhanging wall
    tpl = gm.Template(spec, -1)
    g = tpl.geometry(th)
    CR = g["V"][NAMES.index("top_corner_R")]
    RR = g["V"][NAMES.index("root_edge_R")]
    assert RR[0] > CR[0]                                    # wall leans over the floor
    pts = np.array([[CR[0] - 0.05, CR[1] + 5.0], [CR[0] + 0.05, CR[1] + 5.0], [0.0, 460.0], [-20.0, 450.0]])
    t, k = tpl.cast(th, fan.rays(pts[:, 0], pts[:, 1]))
    names = [g["names"][i] for i in k]
    assert names == ["gap_bottom", "top_R", "gap_bottom", "top_L"]
    tpl_nb = gm.Template(GrooveSpec(plate_width_mm=600.0, backing=False), -1)
    _, k2 = tpl_nb.cast(th, fan.rays(np.array([0.0]), np.array([460.0])))
    assert k2[0] == -1                                      # open gap: no direct return predicted


def test_cut_geometry_stretch_pitch_and_shear():
    u, n = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    g = gm.cut_geometry(_sensor(psi=20.0), u, n)
    assert g["c"] == pytest.approx(1 / math.cos(math.radians(20)), rel=1e-9) and abs(g["shear"]) < 1e-12
    g = gm.cut_geometry(_sensor(phi=-5.0), u, n)
    assert g["c"] == pytest.approx(1.0) and g["phi_rad"] == pytest.approx(math.radians(5), abs=1e-9) and abs(g["shear"]) < 1e-12
    g = gm.cut_geometry(_sensor(psi=20.0, phi=-5.0), u, n)
    assert abs(g["shear"]) > 0.01                           # ψ and φ together shear the cut (not negligible at 20°/5°)


def test_profile_prior_apparent_design():
    p = gm.profile_prior(SPEC, TOL, c=1.1, sigma_c=0.05, phi_rad=math.radians(5))
    assert p.mu[gm.I_G] == pytest.approx(5.5)
    assert p.sd[gm.I_G] == pytest.approx(math.hypot(1.1 * 0.5, 5.0 * 0.05))
    assert math.tan(p.mu[gm.I_BE]) == pytest.approx(1.1 * math.cos(math.radians(5)) * math.tan(math.radians(30)))
    assert p.mu[gm.I_HL] == pytest.approx(10.0 / math.cos(math.radians(5)))
    assert not np.isfinite(p.sd[gm.I_XR]) and not np.isfinite(p.sd[gm.I_AL]) and not np.isfinite(p.sd[gm.I_GL])
    assert p.lo[gm.I_RF] == 0.0 and p.free[gm.I_RF] and not np.isfinite(p.sd[gm.I_RF])     # no job tolerance → flat (+ BIC)


def test_root_face_design_prior_from_the_job():
    import os
    import yaml
    job = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scan_l0", "scan_l0", "jobs",
                       "groove_10T_1G.yaml")
    with open(job, "r", encoding="utf-8") as f:
        qc = yaml.safe_load(f)["qc_targets"]
    assert gm.root_face_design(qc, SPEC) == (0.0, 0.2)                 # user 2026-10-09: root face 0 ± 0.2 mm
    p = gm.profile_prior(SPEC, TOL, phi_rad=math.radians(5), qc_targets=qc)
    assert p.mu[gm.I_RF] == 0.0 and p.sd[gm.I_RF] == pytest.approx(0.2 / math.cos(math.radians(5)))
    assert p.lo[gm.I_RF] == 0.0 and p.free[gm.I_RF]


# ---------------------------------------------------------------------------------------------------------------- accuracy
CASES = [dict(), dict(psi=20.0, roll=4.0, hi_lo=1.0), dict(phi=-5.0, roll=4.0), dict(psi=20.0, phi=-5.0), dict(psi=10.0, hi_lo=1.0)]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "_".join(f"{k}{v:g}" for k, v in c.items()) or "nominal")
def test_fit_recovers_truth_with_honest_covariance(case):
    errs, zs, dims = [], [], []
    for seed in range(5):
        x, z, v, tpl, prior, TV, cg = _case(seed=seed, **case)
        f = gm.fit_profile(x, z, v, tpl, prior)
        assert f.converged and f.log_bf > 0
        V = tpl.geometry(f.theta)["V"]
        R, Rt = 0.5 * (V[2] + V[3]), 0.5 * (TV[2] + TV[3])
        errs.append(R - Rt)
        zs.append((R - Rt) / np.sqrt(np.diag(f.cov[:2, :2])))
        e1 = (TV[3] - TV[2]) / np.linalg.norm(TV[3] - TV[2])
        D_t = abs(float((TV[0] - TV[2]) @ np.array([e1[1], -e1[0]])))
        dims.append([np.linalg.norm(V[3] - V[2]) - np.linalg.norm(TV[3] - TV[2]),            # G
                     np.linalg.norm(V[4] - V[0]) - np.linalg.norm(TV[4] - TV[0]),            # O
                     f.theta[gm.I_HL] - D_t,                                                  # D
                     math.degrees(f.theta[gm.I_BE] - math.atan(cg["c"] * math.cos(cg["phi_rad"]) * math.tan(math.radians(30))))])
    errs, zs, dims = np.array(errs), np.array(zs), np.array(dims)
    assert np.all(np.abs(errs.mean(axis=0)) < 0.05)
    assert np.all(np.abs(zs) <= 3.0)
    assert np.all(np.abs(dims[:, :3].mean(axis=0)) < 0.05)
    assert abs(dims[:, 3].mean()) < 0.3


def test_outliers_and_missing_columns():
    """reflection band (10–40 columns 1–4 mm too deep, inside the groove) + 20 % dropped columns. A band that forms a coherent
    alternative bevel can win on a single profile (the leg-level Huber estimator handles that profile); most profiles are exact."""
    ok = 0
    for seed in range(5):
        x, z, v, tpl, prior, TV, _ = _case(seed=seed, reflection_prob=1.0, reflection_min_depth_mm=0.5, dropout_prob=0.2)
        f = gm.fit_profile(x, z, v, tpl, prior)
        V = tpl.geometry(f.theta)["V"]
        assert f.n_valid < 1408 and f.log_bf > 0
        zR = (0.5 * (V[2] + V[3]) - 0.5 * (TV[2] + TV[3])) / np.sqrt(np.diag(f.cov[:2, :2]))
        zG = (f.theta[gm.I_G] - np.linalg.norm(TV[3] - TV[2])) / math.sqrt(f.cov[gm.I_G, gm.I_G])
        ok += bool(np.all(np.abs(zR) <= 3.0) and abs(zG) <= 3.0 and f.eps > 0.003)     # error within the reported k·σ
    assert ok >= 4


# ---------------------------------------------------------------------------------------------------------------- evidence
def _lam(scene, T_bo, seed=3):
    x, z, v = _render(scene, T_bo, seed=seed, z_lo=300.0, z_hi=500.0)
    return max(gm.fit_profile(x, z, v, gm.Template(SPEC, s), gm.profile_prior(SPEC, TOL)).log_bf for s in (1, -1))


@pytest.mark.parametrize("label,scene", [
    ("plate", PlaneScene(h0=0.0)),
    ("table", PlaneScene(h0=-0.02)),
    ("plate_end_step_to_table", StepScene(h_high=0.0, h_low=-0.02, angle_deg=90.0)),
    ("block_on_plate", CompositeScene([PlaneScene(h0=0.0), GaugeBlockScene(xc=0.0, yc=0.01, sx=0.05, sy=0.03, h=0.015)])),
    ("bar_on_plate", CompositeScene([PlaneScene(h0=0.0), GaugeBlockScene(xc=0.0, yc=0.01, sx=0.05, sy=0.008, h=0.010)])),
])
def test_log_bf_negative_without_groove(label, scene):
    assert _lam(scene, _sensor(444.0)) < 0, label


def test_log_bf_positive_with_part_of_the_line_beyond_the_plate_end_and_negative_past_it():
    L = SPEC.weld_length_mm * 1e-3
    sc = CompositeScene([GrooveScene(spec=SPEC, x_range=(-L, 0.0), y_range=(-0.3, 0.3)), PlaneScene(h0=-0.03)])
    assert _lam(sc, _sensor(444.0, psi=20.0, x=-15.0)) > 0            # ~27 % of the columns beyond the end, groove in view
    assert _lam(sc, _sensor(444.0, psi=20.0, x=10.0)) < 0             # groove beyond the end, plate sliver + table in view


# ---------------------------------------------------------------------------------------------------------------- leg
class _Res:
    pass


def _leg_result(n=6, side="left"):
    spec = GrooveSpec(plate_width_mm=600.0, bevel_side=side)
    X, Z, V, T, Tr = [], [], [], [], []
    for i in range(n):
        T_bo = _sensor(342.0, 0.0, -5.0, x=-100.0 + 4.0 * i, y=0.7)
        x, z, v = _render(GrooveScene(spec=spec, y_range=(-0.3, 0.3)), T_bo, seed=i)
        X.append(x), Z.append(z), V.append(v), T.append(T_bo)
        # truth root point in base: root line (plate x axis at lateral 0, height −t) ∩ laser plane
        o, Y = T_bo[:3, 3], T_bo[:3, 1]
        p0, d = np.array([0.0, 0.0, -spec.thickness_mm * 1e-3]), np.array([1.0, 0.0, 0.0])
        Tr.append(p0 - float(Y @ (p0 - o)) / float(Y @ d) * d)
    r = _Res()
    r.x_mm, r.z_mm, r.valid, r.T_base_opt = np.array(X), np.array(Z), np.array(V), np.array(T)
    r.t_acq_s = np.arange(n) * 0.13
    r.meta = {}
    return r, np.array(Tr)


def _local_map(fn, chunks, n):
    return [fn(c) for c in chunks]


def test_fit_leg_side_by_bayes_factor_outputs_and_roundtrip(tmp_path):
    res, truth = _leg_result()
    prior = gm.profile_prior(SPEC, TOL, phi_rad=math.radians(5))
    leg = gm.fit_leg(res, SPEC, TOL, cls="fine", direction=1, bevel_x_sign=None, n_children=2, map_fn=_local_map,
                     prior=prior, name="pass_1")
    assert leg.bevel_x_sign == -1 and leg.meta["bevel_side_ln_bf"] < 0      # bevel plate on sensor −x (plate −y)
    assert np.all(leg.log_bf > 0) and np.all(leg.converged) and leg.n == 6
    P, C = gm.root_points_base(leg)
    assert np.all(np.linalg.norm(P - truth, axis=1) < 0.05e-3)
    assert C.shape == (6, 3, 3) and np.all(np.linalg.eigvalsh(C)[:, 0] > -1e-18)
    Vb, Jb = gm.vertices_base(leg)
    assert Vb.shape == (6, 5, 3) and Jb.shape == (6, 5, 3, gm.N_THETA)
    fl = gm.face_lines_base(leg)
    assert {"plate_top_left", "plate_top_right", "bevel_right", "wall_left", "gap_bottom"} == set(fl)   # +x travel: bevel (−y) is right
    Pp, Dd, Cv = fl["plate_top_left"]
    assert np.allclose(np.linalg.norm(Dd, axis=1), 1.0) and Cv.shape == (6, 2, 2)
    leg.save(tmp_path / "profiles.npz")
    back = LegObs.load(tmp_path / "profiles.npz")
    assert back.bevel_x_sign == leg.bevel_x_sign and np.allclose(back.theta, leg.theta)
    sgn, lnbf = gm.choose_bevel_side(res, SPEC, TOL, stride=2, prior=prior)
    assert sgn == -1 and lnbf < 0
    assert gm.bevel_x_sign_for(res.T_base_opt, np.array([0.0, 1.0, 0.0]), -1) == -1    # bevel on −w(= −y): sensor x = +y → −1


# ---------------------------------------------------------------------------------------------------------------- dwell
def test_fit_dwell_oblique_frames_root_standoff_roll_and_no_groove():
    """dwell = F frames of one static profile (oblique ψ 15°, plate roll 3°): per-column median profile → fit; root, stand-off, roll
    with σ; side by Bayes factor; λ < 0 on a table-only dwell."""
    spec = GrooveSpec(plate_width_mm=600.0)
    T_bo, T_bp = _sensor(443.0, 15.0, 0.0, y=1.1), _plate(3.0)
    frames = [_render(GrooveScene(spec=spec, y_range=(-0.3, 0.3)), T_bo, T_bp, seed=200 + k) for k in range(8)]
    X, Z, V = (np.array([f[i] for f in frames]) for i in range(3))
    cg = gm.cut_geometry(T_bo, T_bp[:3, 0], T_bp[:3, 2])
    prior = gm.profile_prior(SPEC, TOL, c=cg["c"], phi_rad=cg["phi_rad"], shear=cg["shear"])
    d = gm.fit_dwell(X, Z, V, SPEC, TOL, prior=prior, T_base_opt=T_bo)
    TV = _truth(spec, T_bo, T_bp)
    R = 0.5 * (TV[2] + TV[3])
    assert d.n_frames == 8 and d.bevel_x_sign == -1 and d.side_ln_bf < 0 and d.log_bf > 0
    assert abs(d.root_x_mm - R[0]) <= 3 * math.sqrt(d.root_cov_mm2[0, 0]) + 1e-3
    assert abs(d.root_z_mm - R[1]) <= 3 * math.sqrt(d.root_cov_mm2[1, 1]) + 1e-3
    top_z = 0.5 * (TV[0][1] + TV[4][1])                                   # plate tops (no hi-lo) at the root: ≈ corners' z
    assert abs(d.standoff_mm - top_z) < 0.3                               # corners lie ±5 mm from the root on a 3° roll
    assert abs(math.degrees(d.roll_rad)) == pytest.approx(3.0 / math.cos(math.radians(15.0)), abs=0.2) or \
        abs(math.degrees(d.roll_rad)) == pytest.approx(3.0, abs=0.3)
    truth_base = T_bo[:3, 3] + 1e-3 * (R[0] * T_bo[:3, 0] + R[1] * T_bo[:3, 2])
    assert np.linalg.norm(d.root_base_m - truth_base) < 0.1e-3
    tbl = [_render(PlaneScene(h0=-0.02), T_bo, seed=300 + k, z_lo=300.0, z_hi=500.0) for k in range(4)]
    dn = gm.fit_dwell(np.array([t[0] for t in tbl]), np.array([t[1] for t in tbl]), np.array([t[2] for t in tbl]), SPEC, TOL)
    assert dn.log_bf < 0


# ---------------------------------------------------------------------------------------------------------------- seam evidence
def test_seam_prior_evidence_rejects_a_groove_shaped_feature_away_from_the_seam():
    """with a prediction (θ0, Σ_R) from the fused state, λ compares 'this seam's groove' with 'not this seam' (alternative or a groove
    shape anywhere in the window): the real groove at the seam stays positive, a groove-shaped table feature 20 mm below goes negative."""
    spec = GrooveSpec(plate_width_mm=600.0)
    T_bo = _sensor(444.0, y=0.6)
    prior = gm.profile_prior(SPEC, TOL)
    x, z, v = _render(GrooveScene(spec=spec, y_range=(-0.3, 0.3)), T_bo, seed=1)
    f0 = gm.fit_profile(x, z, v, gm.Template(SPEC, -1), prior)
    pred = (f0.theta.copy(), np.diag([1.0, 1.0]))
    f = gm.fit_profile(x, z, v, gm.Template(SPEC, -1), prior, start_pred=pred)
    assert f.log_bf > 0 and math.isfinite(f.seam_term) and f.log_bf_free > 0
    assert np.all(f.plate_inliers > 0.8 * f.plate_columns) and np.all(f.plate_columns > 100) and f.beyond_plate == -1
    x, z, v = _render(GrooveScene(spec=spec, y_range=(-0.3, 0.3), h_top=-0.020), T_bo, seed=2, z_lo=300.0, z_hi=520.0)
    f = gm.fit_profile(x, z, v, gm.Template(SPEC, -1), prior, start_pred=pred)
    assert f.log_bf_free > 0 and f.log_bf < 0
    th, c = gm.parse_prediction({"theta": pred[0], "cov": np.eye(10)})
    assert np.allclose(th, pred[0]) and c.shape == (2, 2)
    assert gm.parse_prediction(pred[0])[1] is None and gm.parse_prediction(None) == (None, None)


def test_fit_leg_batched_equals_per_profile_and_carries_seam_and_plate_outputs():
    res, _ = _leg_result(n=6)
    prior = gm.profile_prior(SPEC, TOL, phi_rad=math.radians(5))
    kw = dict(cls="fine", direction=1, bevel_x_sign=-1, prior=prior, warm=False)
    a = gm.fit_leg(res, SPEC, TOL, batched=True, **kw)
    b = gm.fit_leg(res, SPEC, TOL, batched=False, **kw)
    sd = np.sqrt(np.diagonal(b.cov, axis1=1, axis2=2))
    dz = np.where(sd > 0, np.abs(a.theta - b.theta) / np.where(sd > 0, sd, 1.0), np.abs(a.theta - b.theta))   # fixed params: σ = 0
    assert np.max(dz) < 1e-3 and np.allclose(a.log_bf, b.log_bf, atol=1e-3)
    assert np.array(a.meta["plate_columns"]).shape == (6, 2) and a.meta["log_bf_kind"] == "free"
    pred = lambda i: (b.theta[i], np.diag([0.5, 0.5]))
    c = gm.fit_leg(res, SPEC, TOL, cls="fine", direction=1, bevel_x_sign=-1, prior=prior, predict=pred)
    assert c.meta["log_bf_kind"] == "seam" and np.all(c.log_bf > 0) and len(c.meta["log_bf_free"]) == 6


# ---------------------------------------------------------------------------------------------------------------- mirror
def _mirror_theta(th):
    t = np.array(th, dtype=float).copy()
    t[gm.I_XR] = -t[gm.I_XR]
    t[gm.I_AL] = -t[gm.I_AL]
    return t


@pytest.mark.parametrize("case", [dict(hi_lo=2.0, roll=4.0), dict(psi=10.0, hi_lo=1.0), dict(phi=-5.0, roll=-3.0)],
                         ids=["hilo2_roll4", "psi10_hilo1", "phi5_roll-3"])
def test_mirror_equivariance(case):
    """review 10-10 item 6: fit(profile, sign) and fit(mirrored profile, −sign) give mirrored θ (x_R, α → −x_R, −α), the same λ, σ_r,
    ε and per-face inliers, and both match the truth. Equal up to the [R] stopping tolerance (steps ≤ 0.1σ, gain ≤ 0.01 nat): the mirrored
    problem sums in reversed order, so line-search decisions at the tolerance can differ (the 10-10 bug was mm-level)."""
    x, z, v, tpl, prior, TV, cg = _case(seed=7, **case)
    fan = gm.RayFan.from_validity()
    xm, zm, vm = -x[::-1], z[::-1], v[::-1]                                     # column k ↔ −k (the fan is symmetric)
    tpl_m = gm.Template(SPEC, -tpl.bevel_x_sign, shear=-tpl.shear)
    r, rm = gm._rays_of(x, z, v, fan), gm._rays_of(xm, zm, vm, fan)
    th0 = gm.template_search_start(None, None, None, tpl, prior, _rays=r)
    a = gm._fit_template(tpl, prior, r, th0)
    th, sig, eps = a["theta"], a["sigma"], a["eps"]
    # (1) the model itself is exactly equivariant: likelihood and Laplace covariance at mirrored θ (no optimizer)
    ev = []
    for tp, rr, t in ((tpl, r, th), (tpl_m, rm, _mirror_theta(th))):
        C = gm._components(tp, t, rr)
        lt, gam = gm._estep(C, rr, sig, eps)
        H, _ = gm._normal_eq(C, gam / sig ** 2, gam, prior, t)
        ev.append((float(lt.sum()), gm._solve(H, np.zeros(gm.N_THETA), prior.free)[1], C["face"][:C["n_one"]].size))
    assert ev[1][0] == pytest.approx(ev[0][0], abs=1e-6) and ev[1][2] == ev[0][2]
    M = np.diag([-1.0, 1, -1, 1, 1, 1, 1, 1, 1, 1])
    assert np.allclose(M @ ev[1][1] @ M, ev[0][1], rtol=1e-6, atol=1e-15)
    # (2) end to end from an exactly mirrored start: equal up to the stopping tolerance
    b = gm._fit_template(tpl_m, prior, rm, _mirror_theta(th0))
    sd = np.sqrt(np.diag(a["cov"]))
    sdz = np.where(sd > 0, sd, 1.0)
    assert np.max(np.abs(_mirror_theta(b["theta"]) - a["theta"]) / sdz) < 0.3
    assert abs(b["log_post"] - a["log_post"]) < 0.1 and b["sigma"] == pytest.approx(a["sigma"], rel=0.01)
    f = gm.fit_profile(x, z, v, tpl, prior)                                     # full path (own matched-filter starts)
    fm = gm.fit_profile(xm, zm, vm, tpl_m, prior)
    assert np.max(np.abs(_mirror_theta(fm.theta) - f.theta) / sdz) < 0.6
    assert fm.log_bf == pytest.approx(f.log_bf, abs=1.0) and fm.sigma_r == pytest.approx(f.sigma_r, rel=0.02)
    assert np.allclose(fm.face_inliers, f.face_inliers, atol=3.0)
    # residuals of the outer plate tops stay at noise level for both signs (the 10-10 bug gave mm-level residuals for sign +1)
    for tp, ff, rr in ((tpl, f, r), (tpl_m, fm, rm)):
        C = gm._components(tp, ff.theta, rr)
        assert np.percentile(np.abs(C["r"][:C["n_one"]]), 99) < 0.2
    V = tpl.geometry(f.theta)["V"]
    assert np.all(np.abs(0.5 * (V[2] + V[3]) - 0.5 * (TV[2] + TV[3])) <= 3 * np.sqrt(np.diag(f.cov[:2, :2])) + 1e-3)


def test_outer_faces_follow_the_mirror_in_both_paths():
    from scan_core import groove_batch as gb
    fan = gm.RayFan.from_validity()
    prior = gm.profile_prior(SPEC, TOL)
    for sign in (-1, 1):
        tpl = gm.Template(SPEC, sign)
        th = prior.start.copy()
        th[gm.I_XR], th[gm.I_ZR], th[gm.I_AL], th[gm.I_HR], th[gm.I_GR] = 3.0, 444.0, 0.02, th[gm.I_HL] + 2.0, 0.03
        D = gm.column_rays(fan, 1408)
        g = tpl.geometry(th)
        t, face = gm._cast(g["seg0"], g["seg1"], fan.origin, D)
        ok = np.isfinite(t)
        P = fan.origin + t[ok, None] * D[ok]
        rays = fan.rays(P[:, 0], P[:, 1], n_points=1408)
        C = gm._components(tpl, th, rays)
        assert np.all(C["face"][:C["n_one"]] == face[ok][C["col"][:C["n_one"]]])
        assert np.max(np.abs(C["r"][:C["n_one"]])) < 1e-6
        CB = gb._components_b(tpl, th[None], gb.rays_batch([rays]), np.array([0]))
        one = CB["one"][0]
        assert np.all(CB["kC"][0][one] == face[ok][one]) and np.max(np.abs(CB["r1"][0][one])) < 1e-6
