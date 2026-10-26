"""WP2 seam estimator (G1_CORE_SPEC §4, §10 WP2 acceptance) on synthetic LegObs generated from a known seam.

The generator builds the true 3D groove (root line, ends, plate normals, bevel side, G/D/β/hi-lo), moves an optical frame along each
leg (yaw ψ of the laser plane vs the seam, pitch φ, standoff, speed, fps), intersects the true edge lines with the true laser plane
(exact, so ψ and pitch stretch are in the data), turns the in-plane points into θ (spec §3.2 order) and records the pose at the
capture time + δ with the leg's offset removed (measured + o = true). Per-profile noise: θ ~ N(θ_true, Σ_θ) plus an extra root jitter
σ_ζ that is not in Σ_θ (the estimator must find it by REML). Beyond-end profiles get λ < 0 and junk θ.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from scan_core.g1_types import LegObs, Precision, SeamPrior, Tolerances, taught_frame
from scan_core.specimen import GrooveSpec
from scan_core import seam_estimate as se

UP = np.array([0.0, 0.0, 1.0])
# the done test's 1-σ precision targets of these estimator tests (the k·σ ≤ tol they were written for: tol/k)
PRECISION = Precision(end_position_mm=1.0 / 3.0, root_position_mm=1.0 / 3.0, length_mm=0.5 / 3.0)
QC = {"root_position_mm": {"tol": 1.0}, "root_gap_mm": {"tol": 0.5}, "groove_depth_mm": {"tol": 0.5},
      "bevel_angle_deg": {"tol": 2.0}, "top_opening_mm": {"tol": 0.5}, "weld_length_mm": {"tol": 1.0}}


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


class Seam:
    """True groove: root line A + σ·u (σ ∈ [0, L]), plate normal n (roll about u), bevel on +w if bevel_w = +1."""

    def __init__(self, A=(0.6, -0.1, 0.05), yaw_deg=3.0, slope_deg=0.3, L_mm=380.5, roll_deg=0.0, bevel_w=1, G=5.0, D=10.0,
                 beta_deg=30.0, rf=0.0, hilo=0.0, misalign_deg=0.0):
        self.A = np.asarray(A, float)
        u = _rot(UP, math.radians(yaw_deg)) @ np.array([1.0, 0, 0])
        u = _rot(np.cross(UP, u), -math.radians(slope_deg)) @ u
        self.u = u / np.linalg.norm(u)
        n_ = UP - (UP @ self.u) * self.u
        n_ /= np.linalg.norm(n_)
        self.n = _rot(self.u, math.radians(roll_deg)) @ n_
        self.w = np.cross(self.n, self.u)
        self.L = L_mm * 1e-3
        self.bw, self.G, self.D, self.beta, self.rf, self.H = bevel_w, G * 1e-3, D * 1e-3, math.radians(beta_deg), rf * 1e-3, \
            hilo * 1e-3
        self.n_bev = _rot(self.u, math.radians(misalign_deg) * 0.5 * bevel_w) @ self.n
        self.n_sq = _rot(self.u, -math.radians(misalign_deg) * 0.5 * bevel_w) @ self.n

    def lines(self):
        """Named 3D edge lines (point at σ = 0) parallel to u."""
        A, w, n, b = self.A, self.w, self.n, self.bw
        g2 = 0.5 * self.G
        return {"R": A, "B0": A + b * g2 * w, "B1": A + b * g2 * w + self.rf * n,
                "Cb": A + b * (g2 + (self.D - self.rf) * math.tan(self.beta)) * w + self.D * n,
                "Q0": A - b * g2 * w, "Cs": A - b * g2 * w + (self.D + self.H) * n}

    def point(self, sig):
        return self.A + sig * self.u


def _plate_dir2(nplate, eyp, exp_, ezp):
    d3 = np.cross(nplate, eyp)
    return np.array([d3 @ exp_, d3 @ ezp]) / np.linalg.norm([d3 @ exp_, d3 @ ezp])


def _theta_wp1(theta0, bxs, pts, dirs, iters=8):
    """θ (N,10) such that groove_model's Template(spec, bxs) reproduces the given in-plane points (N,2) [mm] and plate-top directions
    (2,) — vectorised Gauss–Newton with WP1's analytic Jacobians (geometry_b)."""
    from scan_core.groove_batch import geometry_b
    from scan_core.groove_model import Template
    tpl = Template(GrooveSpec(), int(bxs), shear=0.0)
    th = np.array(theta0, dtype=float)
    vn, fn = list(tpl.vertex_names), None
    for _ in range(iters):
        g = geometry_b(tpl, th)
        fn = list(g["names"])
        res, jac = [], []
        for nm, P in pts.items():
            i = vn.index(nm)
            res.append(g["V"][:, i] - P)
            jac.append(g["dV"][:, i])
        for nm, d in dirs.items():
            f = fn.index(nm)
            u = g["u"][:, f]
            sg = np.where(u @ d >= 0, 1.0, -1.0)[:, None]
            res.append(1e3 * (u - sg * d))                        # plate-top directions are the truth of γ (1 m lever)
            jac.append(1e3 * g["du"][:, f])
        r = np.concatenate(res, axis=1)                           # (N, K)
        J = np.concatenate(jac, axis=1)                           # (N, K, 10)
        A = np.einsum("nkp,nkq->npq", J, J) + 1e-9 * np.eye(10)
        b = np.einsum("nkp,nk->np", J, r)
        th = th - np.linalg.solve(A, b[..., None])[..., 0]
    return th


def make_leg(seam: Seam, f0: dict, *, name, cls, direction, s_from_mm, s_to_mm, standoff_mm, pitch_deg=0.0, yaw_deg=0.0,
             lateral_mm=0.0, speed_mps=0.03, fps=100.0, dt_true_s=0.0, offset_m=(0, 0, 0), flip_x=False, sig_theta=None,
             jitter_mm=0.05, lam_in=40.0, lam_out=-40.0, lam_sd=10.0, outlier_frac=0.0, rng=None, travel_u=None,
             end_shift_mm=(0.0, 0.0), end_height_mm=5.0, plate_ext_mm=((0.0, 0.0), (0.0, 0.0))):
    """One synthetic leg along the taught u0 between taught s_from → s_to (direction from the order), optical frame:
    ey = travel direction rotated by yaw about up (laser plane normal), ez = down ⟂ ey then pitched about ex, ex = ey × ez."""
    rng = np.random.default_rng(0) if rng is None else rng
    S0, u0, w0, n0 = f0["S0"], f0["u0"], f0["w0"], f0["n0"]
    d_tr = (u0 if travel_u is None else np.asarray(travel_u, float)) * (1.0 if s_to_mm >= s_from_mm else -1.0)
    length = abs(s_to_mm - s_from_mm) * 1e-3
    n = int(length / speed_mps * fps) + 1
    t = np.arange(n) / fps
    ey = _rot(UP, math.radians(yaw_deg)) @ d_tr
    if flip_x:
        ey = -ey
    ez0 = -UP + (UP @ ey) * ey
    ez0 /= np.linalg.norm(ez0)
    ex0 = np.cross(ey, ez0)
    Rp = _rot(ex0, math.radians(pitch_deg))
    eyp, ezp = Rp @ ey, Rp @ ez0
    exp_ = np.cross(eyp, ezp)
    Rm = np.stack([exp_, eyp, ezp], axis=1)
    # path: above the true root at the start s, standoff along −ez
    sig0 = ((S0 + s_from_mm * 1e-3 * u0) - seam.A) @ seam.u
    root0 = seam.point(sig0)
    o0 = root0 - standoff_mm * 1e-3 * ezp + lateral_mm * 1e-3 * w0
    v = speed_mps * d_tr

    def origin(tt):
        return o0 + np.outer(tt, v)

    o_true = origin(t)
    lines = seam.lines()
    pts = {}
    sig_root = None
    for k, X0 in lines.items():
        sg = ((o_true - X0) @ eyp) / (eyp @ seam.u)
        P = X0 + sg[:, None] * seam.u
        rel = P - o_true
        pts[k] = np.stack([rel @ exp_, rel @ ezp], axis=1) * 1e3
        if k == "R":
            sig_root = sg
    B0, Q0, R, B1, Cb, Cs = pts["B0"], pts["Q0"], pts["R"], pts["B1"], pts["Cb"], pts["Cs"]
    bxs = 1 if float(np.median(B0[:, 0] - Q0[:, 0])) > 0 else -1
    e1 = bxs * (B0 - Q0)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    al = np.arctan2(e1[:, 1], e1[:, 0])
    e2 = np.stack([np.sin(al), -np.cos(al)], axis=1)
    G = np.linalg.norm(B0 - Q0, axis=1)
    rf = np.einsum("ij,ij->i", B1 - R, e2)
    hL = np.einsum("ij,ij->i", Cb - R, e2)
    xi_cb = np.einsum("ij,ij->i", Cb - R, e1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.arctan2(bxs * xi_cb - 0.5 * G, hL - rf)
    hR = np.einsum("ij,ij->i", Cs - R, e2)

    def gam(nplate):                    # groove_model convention: angle in the canonical (ξ, η), ξ = m·e1, m = −bevel_x_sign
        d3 = np.cross(nplate, eyp)
        d2 = np.array([d3 @ exp_, d3 @ ezp])
        xi, eta = -bxs * (e1 @ d2), e2 @ d2
        g = np.arctan2(eta, xi)
        return np.where(np.cos(g) < 0, np.arctan2(-eta, -xi), g)

    gL, gR = gam(seam.n_bev), gam(seam.n_sq)
    theta = np.stack([R[:, 0], R[:, 1], al, G, beta, rf, hL, hR, gL, gR], axis=1)
    # exact θ in groove_model's convention (own-plate frames, mirroring): Gauss–Newton through WP1's geometry_b so that the template
    # reproduces the true points and plate-top directions of every cut
    theta = _theta_wp1(theta, bxs, {"top_corner_L": Cb, "root_face_top_L": B1, "root_edge_L": B0, "root_edge_R": Q0,
                                    "top_corner_R": Cs},
                       {"top_L": _plate_dir2(seam.n_bev, eyp, exp_, ezp), "top_R": _plate_dir2(seam.n_sq, eyp, exp_, ezp)})
    sd = np.array([0.02, 0.02, 0.002, 0.03, 0.005, 0.03, 0.02, 0.02, 0.002, 0.002]) if sig_theta is None else np.asarray(sig_theta)
    cov = np.broadcast_to(np.diag(sd ** 2), (n, 10, 10)).copy()
    theta = theta + rng.normal(size=(n, 10)) * sd
    theta[:, 0] += rng.normal(size=n) * jitter_mm
    theta[:, 1] += rng.normal(size=n) * jitter_mm
    # groove evidence changes where the (tilted) laser plane crosses the end face at height h above the root (+ per-reading scatter)
    Xh = seam.A + end_height_mm * 1e-3 * seam.n
    sig_h = ((o_true - Xh) @ eyp) / (eyp @ seam.u)
    inside = (sig_h >= end_shift_mm[0] * 1e-3) & (sig_h <= seam.L + end_shift_mm[1] * 1e-3)
    lam = np.where(inside, lam_in, lam_out) + rng.normal(size=n) * lam_sd
    out = ~inside
    theta[out, 0] += rng.normal(size=out.sum()) * 5.0
    theta[out, 1] += rng.normal(size=out.sum()) * 3.0
    if outlier_frac > 0:
        k = np.flatnonzero(inside & (rng.random(n) < outlier_frac))
        theta[k, 0] += rng.choice([-1, 1], size=k.size) * rng.uniform(2.0, 8.0, size=k.size)
    # plate-top evidence (WP1 meta['face_inliers'], faces top_L … top_R): a plate's top face has inliers while the top-surface line
    # (height D above the root) cuts that plate; plate L = the bevel plate; plate_ext_mm = (S, E) extension beyond the groove ends
    Xt = seam.A + seam.D * seam.n
    sig_t = ((o_true - Xt) @ eyp) / (eyp @ seam.u)
    fi = np.zeros((n, 6))
    for col, (eS_, eE_) in ((0, plate_ext_mm[0]), (5, plate_ext_mm[1])):
        pres = (sig_t >= (end_shift_mm[0] - eS_) * 1e-3) & (sig_t <= seam.L + (end_shift_mm[1] + eE_) * 1e-3)
        fi[:, col] = np.where(pres, 500.0 * np.exp(rng.normal(size=n) * 0.1), rng.poisson(1.0, size=n))
    fi[:, 1:5] = 40.0
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rm
    T[:, :3, 3] = origin(t + dt_true_s) - np.asarray(offset_m, float)
    vv = np.tile(v, (n, 1))
    leg = LegObs(name=name, cls=cls, direction=direction, t_acq_s=t, T_base_opt=T, v_opt_mps=vv, theta=theta, cov=cov,
                 log_bf=lam, sigma_r_mm=np.full(n, 0.05), eps=np.full(n, 0.01), n_valid=np.full(n, 1000),
                 converged=np.ones(n, bool), bevel_x_sign=bxs, meta={"speed": speed_mps, "fps": fps, "n_points": 1408,
                                                                     "face_inliers": fi.tolist(),
                                                                     "face_names": ["top_L", "bevel_L", "root_face_L", "gap_bottom",
                                                                                    "wall_R", "top_R"]})
    leg._true = {"sig_root": sig_root, "inside": inside, "o": o_true, "R": Rm}
    return leg


def render_measured(leg, seam: Seam, faces: dict, *, ncol=320, fov_mm=140.0, noise_mm=0.02, backing_ext_mm=6.0,
                    backing_w_mm=40.0, backing_t_mm=10.0, rng=None, rollover_E_mm=None, tack_mm=None) -> dict:
    """Measured profiles of a synthetic leg: rays along the optical axis of the TRUE pose (orthographic, ncol columns over the FOV)
    against the true solid — the two plates truncated by their end-face lines σ = e0 + κ·h (faces[(X, 'bev'|'sq')] = (e0_m, κ) in
    the seam's σ, h along the seam normal), the backing under the gap reaching backing_ext beyond the plates; first hit, range noise.
    rollover_E_mm = (r, d): both plates' top edges at E roll over — the top drops linearly by d over the last r before the end face
    (shear-cut roll-over). tack_mm = (l, t): tack welds on both end faces below the section (between the backing bar and the plates,
    user 2026-10-11): blocks reaching l beyond each plate end, from the backing top up to t, across both plates. Returns the recon
    arrays (recorded poses = the leg's T_base_opt)."""
    rng = np.random.default_rng(5) if rng is None else rng
    o, R = leg._true["o"], leg._true["R"]
    ex, ez = R[:, 0], R[:, 2]
    xk = np.linspace(-0.5 * fov_mm, 0.5 * fov_mm, ncol) * 1e-3
    A, u, w, n = seam.A, seam.u, seam.w, seam.n
    b, G, D, rf, H, tb = seam.bw, seam.G, seam.D, seam.rf, seam.H, math.tan(seam.beta)
    Q = o[:, None, :] + xk[None, :, None] * ex[None, None, :]
    rel = Q - A
    sq, wq, hq = rel @ u, rel @ w, rel @ n
    sd, wd, hd = float(ez @ u), float(ez @ w), float(ez @ n)
    lo_b = min(faces[("S", "bev")][0], faces[("S", "sq")][0]) - backing_ext_mm * 1e-3
    hi_b = max(faces[("E", "bev")][0], faces[("E", "sq")][0]) + backing_ext_mm * 1e-3

    def e(X, pl, hh):
        return faces[(X, pl)][0] + faces[(X, pl)][1] * hh

    ro = None if rollover_E_mm is None else (rollover_E_mm[0] * 1e-3, rollover_E_mm[1] * 1e-3)
    tk = None if tack_mm is None else (tack_mm[0] * 1e-3, tack_mm[1] * 1e-3)
    tk_spans = [] if tk is None else [(min(faces[("S", "bev")][0], faces[("S", "sq")][0]) - tk[0],
                                       max(faces[("S", "bev")][0], faces[("S", "sq")][0])),
                                      (min(faces[("E", "bev")][0], faces[("E", "sq")][0]),
                                       max(faces[("E", "bev")][0], faces[("E", "sq")][0]) + tk[0])]

    def drop(sg, pl):
        if ro is None:
            return 0.0
        return ro[1] * np.clip((sg - (faces[("E", pl)][0] - ro[0])) / ro[0], 0.0, None)

    def inside(sg, ww, hh):
        omb, oms = b * ww, -b * ww
        bev = (hh >= 0) & (hh <= D - drop(sg, "bev")) & (omb >= G / 2 + np.maximum(hh - rf, 0.0) * tb) & \
            (sg >= e("S", "bev", hh)) & (sg <= e("E", "bev", hh))
        sqp = (hh >= 0) & (hh <= D + H - drop(sg, "sq")) & (oms >= G / 2) & (sg >= e("S", "sq", hh)) & (sg <= e("E", "sq", hh))
        bk = (hh <= 0) & (hh >= -backing_t_mm * 1e-3) & (np.abs(ww) <= backing_w_mm * 1e-3) & (sg >= lo_b) & (sg <= hi_b)
        out = bev | sqp | bk
        if tk is not None:
            for a_, b_ in tk_spans:
                out = out | ((hh >= 0) & (hh <= tk[1]) & (np.abs(ww) <= backing_w_mm * 1e-3) & (sg >= a_) & (sg <= b_))
        return out

    planes = [(0, 0, 1, -D), (0, 0, 1, -(D + H)), (0, 0, 1, 0.0), (0, b, -tb, -G / 2 + rf * tb), (0, b, 0, -G / 2),
              (0, -b, 0, -G / 2), (1, 0, 0, -lo_b), (1, 0, 0, -hi_b)]
    planes += [(1, 0, -faces[k_][1], -faces[k_][0]) for k_ in faces]
    if tk is not None:
        planes += [(0, 0, 1, -tk[1])] + [(1, 0, 0, -v) for sp in tk_spans for v in sp]
    if ro is not None:                                  # the rolled top: h = top − d·(σ − (e − r))/r
        for pl_, top in (("bev", D), ("sq", D + H)):
            e_ = faces[("E", pl_)][0]
            planes.append((ro[1] / ro[0], 0, 1, -top - ro[1] / ro[0] * (e_ - ro[0])))
    best = np.full(sq.shape, np.inf)
    eps = 1e-7
    for cs, cw, ch, c0 in planes:
        v1 = cs * sd + cw * wd + ch * hd
        if abs(v1) < 1e-12:
            continue
        t = -(cs * sq + cw * wq + ch * hq + c0) / v1
        ok = (t > 0) & (t < best)
        ta, tb_ = t + eps, t - eps
        hit = ok & inside(sq + ta * sd, wq + ta * wd, hq + ta * hd) & ~inside(sq + tb_ * sd, wq + tb_ * wd, hq + tb_ * hd)
        best = np.where(hit, t, best)
    valid = np.isfinite(best)
    z = np.where(valid, best * 1e3 + rng.normal(size=best.shape) * noise_mm, 0.0)
    return {"x_mm": np.broadcast_to(xk * 1e3, best.shape).astype(np.float32), "z_mm": z.astype(np.float32), "valid": valid,
            "T_base_opt": np.asarray(leg.T_base_opt), "t_acq_s": np.asarray(leg.t_acq_s)}


def scenario(*, rng=None, seam=None, dt_true_s=0.0, beta_coarse_mm=(1.5, -1.0, 2.0), rep_sigma_m=5e-5, n_fine=2,
             fine_offsets_mm=None, coarse=True, coarse_yaw_deg=4.0, coarse_lateral_mm=12.0, fine_overscan_mm=20.0,
             fine_spans=None, speed_mps=0.03, fps=100.0, jitter_mm=0.05, outlier_frac=0.0, dt_prior=(0.0, 1e-3),
             end_scatter_mm=0.0, lam_in=40.0, end_height_mm=5.0, plate_ext_mm=((0.0, 0.0), (0.0, 0.0)), pitch_deg=-5.0):
    rng = np.random.default_rng(1) if rng is None else rng
    seam = Seam() if seam is None else seam
    S_true = seam.point(0.0)
    E_true = seam.point(seam.L)
    S0 = S_true + 0.030 * seam.u + 0.015 * seam.w + seam.D * seam.n
    E0 = E_true - 0.030 * seam.u - 0.010 * seam.w + seam.D * seam.n
    f0 = taught_frame(S0, E0, UP)
    eS = float((S_true - S0) @ f0["u0"]) * 1e3
    eE = float((E_true - S0) @ f0["u0"]) * 1e3
    legs = []
    true_off = {}
    if coarse:
        L0 = float(np.linalg.norm((E0 - S0)[:2])) * 1e3
        o = np.asarray(beta_coarse_mm) * 1e-3
        legs.append(make_leg(seam, f0, name="00_coarse", cls="coarse", direction=1, s_from_mm=-108.0, s_to_mm=L0 + 108.0,
                             standoff_mm=444.0, yaw_deg=coarse_yaw_deg, lateral_mm=coarse_lateral_mm, speed_mps=speed_mps,
                             fps=fps, dt_true_s=dt_true_s, offset_m=o, jitter_mm=jitter_mm, rng=rng, lam_in=lam_in,
                             outlier_frac=outlier_frac, end_height_mm=end_height_mm))
        true_off["coarse"] = o
    for k in range(n_fine):
        dirn = -1 if k % 2 == 0 else 1
        eta = (np.asarray(fine_offsets_mm[k]) * 1e-3 if fine_offsets_mm is not None else rng.normal(size=3) * rep_sigma_m)
        shift = tuple(rng.normal(size=2) * end_scatter_mm) if end_scatter_mm else (0.0, 0.0)    # spec §4.4 ε per reading
        a, b = (eS - fine_overscan_mm, eE + fine_overscan_mm) if fine_spans is None else fine_spans[k]
        s_from, s_to = (b, a) if dirn < 0 else (a, b)
        legs.append(make_leg(seam, f0, name=f"{k + 1:02d}_fine", cls="fine", direction=dirn, s_from_mm=s_from, s_to_mm=s_to,
                             standoff_mm=352.0, pitch_deg=pitch_deg, speed_mps=speed_mps, fps=fps, dt_true_s=dt_true_s,
                             offset_m=eta, jitter_mm=jitter_mm, rng=rng, flip_x=(k % 2 == 1), lam_in=lam_in,
                             outlier_frac=outlier_frac, travel_u=seam.u, end_shift_mm=shift, end_height_mm=end_height_mm,
                             plate_ext_mm=plate_ext_mm))
        true_off[f"{k + 1:02d}_fine"] = eta
    spec = GrooveSpec(kind="single_bevel", thickness_mm=10.0, bevel_angle_deg=30.0, root_gap_mm=5.0, weld_length_mm=380.0,
                      plate_width_mm=600.0)
    prior = SeamPrior(S0=S0, E0=E0, up=UP, spec=spec, tol=Tolerances.from_qc(QC), k=3.0, dt_s=dt_prior[0],
                      dt_sigma_s=dt_prior[1], rep_sigma_m=rep_sigma_m, he_sigma_m=2e-3, sigma_eps=0.0, station_step_m=0.005,
                      station_window_m=0.010, precision=PRECISION)
    truth = {"seam": seam, "f0": f0, "eS_m": eS * 1e-3, "eE_m": eE * 1e-3, "offsets": true_off, "dt": dt_true_s}
    return legs, prior, truth


def true_line_wn(truth, s_m):
    """True root lateral (w0) / normal (n0) offsets at taught s."""
    f0, seam = truth["f0"], truth["seam"]
    s_m = np.atleast_1d(s_m)
    # point on the true line with (P − S0)·u0 = s
    sig = (s_m - (seam.A - f0["S0"]) @ f0["u0"]) / (seam.u @ f0["u0"])
    P = seam.A + sig[:, None] * seam.u
    return (P - f0["S0"]) @ f0["w0"], (P - f0["S0"]) @ f0["n0"]


# ====================================================================================================================
def test_max_segment_exact():
    rng = np.random.default_rng(3)
    for _ in range(50):
        lam = rng.normal(size=40) * 5
        best, a, b = se.max_segment(lam)
        brute = max([(0.0, -1, -1)] + [(lam[i:j + 1].sum(), i, j) for i in range(40) for j in range(i, 40)], key=lambda t: t[0])
        assert best == pytest.approx(brute[0])
        assert (a, b) == (brute[1], brute[2]) or best == pytest.approx(lam[a:b + 1].sum())


def test_section_quantities_roundtrip_through_wp1_geometry():
    """Noise-free θ (exact in groove_model's convention) → section quantities in the true section = design, with ψ, pitch, roll and
    hi-lo (shear-free cuts: pitch 0 at any yaw, or ψ ≈ 0 at any pitch)."""
    from scan_core.specimen import GrooveSpec as _GS
    for seam, cfg in ((Seam(hilo=1.0, slope_deg=0.0), dict(standoff_mm=444, yaw_deg=12.0)),
                      (Seam(roll_deg=4.0, hilo=1.0, slope_deg=0.0), dict(standoff_mm=352, pitch_deg=-5.0, travel_u="seam"))):
        f0 = taught_frame(seam.point(0.0) + seam.D * seam.n, seam.point(seam.L) + 0.01 * seam.w + seam.D * seam.n, UP)
        if cfg.get("travel_u") == "seam":
            cfg = dict(cfg, travel_u=seam.u)
        lg = make_leg(seam, f0, name="x", cls="fine", direction=1, s_from_mm=-50, s_to_mm=400, sig_theta=np.full(10, 1e-6),
                      jitter_mm=0.0, rng=np.random.default_rng(0), **cfg)
        prof = se._Profiles([lg], f0)
        sel = lg._true["inside"] & prof.ok
        _, n_, w_ = se._section_basis(seam.u, seam.n)
        q, v = se._section_quantities(prof.geometry(_GS()), sel, prof.cov[sel], w_, n_, "single_bevel")
        G, D, beta, O, H, rf = q.T
        tol_mm = 5e-3                                    # residual cut shear of the generator's κ = 0 template
        assert np.allclose(G, 5.0, atol=tol_mm) and np.allclose(D, 10.0, atol=tol_mm) and np.allclose(beta, 30.0, atol=0.05)
        assert np.allclose(O, 5.0 + 10.0 * math.tan(math.radians(30)), atol=tol_mm) and np.allclose(H, 1.0, atol=tol_mm)
        assert np.allclose(rf, 0.0, atol=tol_mm)


def _run(legs, prior):
    t0 = time.perf_counter()
    est = se.estimate(legs, prior)
    return est, time.perf_counter() - t0


def with_points(legs, truth, faces=None, w_x=0.0, z_c_mm=352.0, **kw):
    """Attach rendered measured profiles to the pass legs (end faces of the true seam unless given). w_x [rad]: laser-plane
    rotation about the line axis that the calibration does not know — the recorded optical frames are rotated by it about the point
    at range z_c on the optical axis (the calibration is exact at its working range)."""
    seam = truth["seam"]
    faces = faces or {("S", "bev"): (0.0, 0.0), ("S", "sq"): (0.0, 0.0), ("E", "bev"): (seam.L, 0.0), ("E", "sq"): (seam.L, 0.0)}
    for lg in legs:
        if lg.cls == "fine":
            m = render_measured(lg, seam, faces, **kw)
            if w_x:
                T = np.array(m["T_base_opt"], dtype=float)
                R = T[:, :3, :3]
                Rx = _rot([1.0, 0.0, 0.0], w_x)
                Rr = R @ Rx
                T[:, :3, 3] = T[:, :3, 3] + 1e-3 * z_c_mm * (R[:, :, 2] - Rr[:, :, 2])
                T[:, :3, :3] = Rr
                m["T_base_opt"] = T
            lg._measured_cache = m
    return faces


def apply_edge_response(legs, truth, faces, sigma_l_mm=0.5, theta_deg=26.0, f_det=0.2):
    """The sensor's response at the plate top edges (Curless & Levoy 1995), on the rendered profiles of the pass legs: a column whose
    sheet centre lies a·σ_L inside a plate's top (a ≥ Φ⁻¹(f_det): the lit part is still detected) returns the top with its height
    moved by G·(centroid shift σ_L·λ(a) into the material), G from the true pose (camera on the sensor −y side, angle θ); beyond the
    edge the ideal hit is kept."""
    from scipy.stats import norm as _n
    seam = truth["seam"]
    sl, th = sigma_l_mm * 1e-3, math.radians(theta_deg)
    a_min = float(_n.ppf(f_det))
    for lg in legs:
        m = getattr(lg, "_measured_cache", None)
        if m is None:
            continue
        R = lg._true["R"]
        o = lg._true["o"]
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
        cdir = -math.cos(th) * ez - math.sin(th) * ey
        G = float((seam.u @ ey) * (cdir @ seam.n) / math.sin(th))
        z = np.array(m["z_mm"], dtype=float)
        val = np.array(m["valid"], dtype=bool)
        xk = np.asarray(m["x_mm"], dtype=float)[0] * 1e-3
        Q = o[:, None, :] + xk[None, :, None] * ex[None, None, :]
        rel = Q - seam.A
        for pl, top, sgn_w, lo_w in (("bev", seam.D, seam.bw, 0.5 * seam.G + (seam.D - seam.rf) * math.tan(seam.beta)),
                                     ("sq", seam.D + seam.H, -seam.bw, 0.5 * seam.G)):
            hd = float(ez @ seam.n)
            t = (top - rel @ seam.n) / hd                       # ray → this plate's top plane
            P = rel + t[..., None] * ez
            sg, ww = P @ seam.u, sgn_w * (P @ seam.w)
            on = (ww >= lo_w + 1e-4) & (t > 0)
            eS, eE = faces[("S", pl)][0], faces[("E", pl)][0]
            for X, d, sh in (("S", (sg - eS) / sl, 1.0), ("E", (eE - sg) / sl, -1.0)):
                near = on & (np.abs(d) < 6.0) & (d >= a_min)
                lam = np.exp(_n.logpdf(d) - _n.logcdf(d))
                dh = G * sh * sl * lam
                z = np.where(near, (t + dh / hd) * 1e3, z)
                val = val | near
        m["z_mm"] = z.astype(np.float32)
        m["valid"] = val


def _s_of(truth, sig):
    seam, f0 = truth["seam"], truth["f0"]
    return float((seam.A + sig * seam.u - f0["S0"]) @ f0["u0"])


def test_full_scenario_line_ends_dims_and_done():
    legs, prior, truth = scenario(rng=np.random.default_rng(11), n_fine=3, dt_true_s=0.0037, dt_prior=(0.0037, 1e-3))
    with_points(legs, truth)
    est, dt = _run(legs, prior)
    assert est.groove_found
    k = prior.k
    # line within k·σ at stations
    s = np.array([st.s_m for st in est.stations])
    tw, tn = true_line_wn(truth, s)
    L = se.LineModel.from_estimate(est)
    r = L.r(s)
    ew = (r - truth["f0"]["S0"]) @ truth["f0"]["w0"]
    en = (r - truth["f0"]["S0"]) @ truth["f0"]["n0"]
    sw, sn = L.sigma_wn(s)
    assert np.all(np.abs(ew - tw) <= k * sw + 1e-12), np.max(np.abs(ew - tw) / sw)
    assert np.all(np.abs(en - tn) <= k * sn + 1e-12), np.max(np.abs(en - tn) / sn)
    # ends within k·σ and within a couple of spacings
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        e = est.ends[X]
        assert e.two_sided
        assert abs(e.s_m - e_true) <= k * e.sigma_m + 1e-12, (X, e.s_m - e_true, e.sigma_m)
        assert abs(e.s_m - e_true) < 1e-3
    # dims near design at mid stations (bevel side known: +w)
    mid = est.stations[len(est.stations) // 2]
    assert mid.dims["gap_mm"]["value"] == pytest.approx(5.0, abs=0.05)
    assert mid.dims["depth_mm"]["value"] == pytest.approx(10.0, abs=0.05)
    assert mid.dims["bevel_angle_deg"]["value"] == pytest.approx(30.0, abs=0.3)
    assert est.bevel_sign_w == 1
    assert mid.faces["bevel_left"]["normal"] is not None and mid.faces["wall_right"]["normal"] is not None
    assert est.ratios and set(est.ratios) >= {"root_lateral", "end_S", "gap_mm"}
    assert dt < 20.0


def test_line_nees_coverage_monte_carlo():
    """Line NEES at the mid station over independent runs: mean ≈ 2 (2 dof), |z| ≤ k in ≥ 97 % (k = 3)."""
    z = []
    for i in range(24):
        rng = np.random.default_rng(100 + i)
        seam = Seam(yaw_deg=rng.uniform(-5, 5), roll_deg=rng.uniform(-4, 4))
        legs, prior, truth = scenario(rng=rng, seam=seam, n_fine=2, fps=25.0, outlier_frac=0.02)
        est = se.estimate(legs, prior)
        L = se.LineModel.from_estimate(est)
        for s in (est.ends["S"].s_m + 0.02, 0.5 * (est.ends["S"].s_m + est.ends["E"].s_m), est.ends["E"].s_m - 0.02):
            tw, tn = true_line_wn(truth, s)
            r = L.r(s)
            sw, sn = L.sigma_wn(s)
            z += [float(((r - truth["f0"]["S0"]) @ truth["f0"]["w0"] - tw[0]) / sw),
                  float(((r - truth["f0"]["S0"]) @ truth["f0"]["n0"] - tn[0]) / sn)]
    z = np.asarray(z)
    assert np.mean(np.abs(z) <= 3.0) >= 0.95, z
    assert 0.3 < np.mean(z ** 2) < 2.5, np.mean(z ** 2)


def test_change_point_within_one_spacing():
    legs, prior, truth = scenario(rng=np.random.default_rng(21), n_fine=1, jitter_mm=0.02, end_height_mm=0.0)   # root crossing
    est = se.estimate(legs, prior)
    sp = 0.03 / 100.0
    for cp in est.plate["legs"].values():
        assert cp["segment_m"] is not None
    fine = [r for r in est.plate["groove_ends"]["S"]["readings"] if r["cls"] == "pass" and r["kind"] == "two_sided"]
    assert fine
    # reading in the measured frame of that leg; the leg's η is tiny, δ = 0; σ ≥ the actual s-gap/√12 (gap < nominal spacing)
    lg = [l for l in legs if l.cls == "fine"][0]
    s = np.sort(se.LineModel.from_estimate(est).s_star(lg.T_base_opt[:, :3, 3], lg.T_base_opt[:, :3, 1]))
    gap = float(np.median(np.diff(s)))
    assert gap <= sp
    assert abs(fine[0]["mean_m"] - truth["eS_m"]) <= gap
    assert fine[0]["sigma_m"] >= gap / math.sqrt(12) * (1 - 1e-6)


def direction_scenario(rng, *, b_mm=(0.02, 0.17, 0.25), dt_true_s=0.004, legs_spec=((-1, 0.03), (1, 0.03), (-1, 0.06), (1, 0.06)),
                       eta_mm=(0.01, 0.02, 0.0), zeta_mm=0.02, fps=100.0, points=True, extra=()):
    """Passes with the robot's motion-direction offset (joint friction × compliance, lab kv_first 2026-10-11): a leg moving in
    direction d along the seam is recorded offset by d·b plus its own η (sd per component eta_mm), and stamped δ = dt_true_s late
    (calibrated dt = 0). b [mm] = (w0, n0) parts in the taught frame (the line's offset frame) and the along-track part along the
    motion (the seam: the taught u0 is 4.5° off it here, an along-u0 offset would leak into w0). legs_spec = (direction, speed) per
    pass. points: measured top edges with the sensor's edge response (plate-end readings, calibrated edge response), each leg's
    crossing of each end moved along the seam by its own ζ ~ N(0, zeta_mm²) (robot path / timing at that crossing — the lab's
    along-track scatter is per crossing: kv_first S and E residuals uncorrelated)."""
    legs, prior, truth = scenario(rng=rng, n_fine=0, coarse=False, rep_sigma_m=float("nan"))
    seam, f0 = truth["seam"], truth["f0"]
    eS, eE = truth["eS_m"] * 1e3, truth["eE_m"] * 1e3
    b3 = (b_mm[0] * f0["w0"] + b_mm[1] * f0["n0"] + b_mm[2] * seam.u) * 1e-3
    truth["eta"] = {}
    for k, (d, v) in enumerate(legs_spec):
        eta = (rng.normal(size=3) * np.asarray(eta_mm)) * 1e-3
        truth["eta"][f"{k + 1:02d}_pass"] = eta
        eta3 = eta[0] * f0["w0"] + eta[1] * f0["n0"] + eta[2] * seam.u
        s_from, s_to = (eE + 20.0, eS - 20.0) if d < 0 else (eS - 20.0, eE + 20.0)
        legs.append(make_leg(seam, f0, name=f"{k + 1:02d}_pass", cls="fine", direction=d, s_from_mm=s_from, s_to_mm=s_to,
                             standoff_mm=352.0, pitch_deg=-5.0, speed_mps=v, fps=fps, dt_true_s=dt_true_s, offset_m=d * b3 + eta3,
                             jitter_mm=0.02, rng=rng, flip_x=(k % 2 == 1), travel_u=seam.u, lam_in=200.0))
    legs = list(extra) + legs
    if points:
        for lg in legs:
            if lg.cls != "fine":
                continue
            zS, zE = rng.normal(size=2) * zeta_mm * 1e-3
            faces = {("S", "bev"): (zS, 0.0), ("S", "sq"): (zS, 0.0), ("E", "bev"): (seam.L + zE, 0.0), ("E", "sq"): (seam.L + zE, 0.0)}
            lg._measured_cache = render_measured(lg, seam, faces)
            apply_edge_response([lg], truth, faces, sigma_l_mm=0.5, theta_deg=26.0)
        prior.edge_response = {"sigma_l_m": 0.5e-3, "sigma_l_log_sd": 0.05, "theta_tri_deg": 26.0, "theta_tri_sigma_deg": 1.0,
                               "source": "test calibration"}
    truth["b_m"] = np.asarray(b_mm) * 1e-3
    truth["dt"] = dt_true_s
    return legs, prior, truth


def test_timing_prior_from_frame_and_pose_periods():
    """The time-offset prior is the run's own timing, not a constant: σ_δ² = (T_frame² + T_pose²)/12 (each stamp uniform within its
    period); legs without a separate pose sampling (poses per profile) contribute T_pose = 0."""
    legs, prior, truth = scenario(rng=np.random.default_rng(5), n_fine=2, coarse=False, fps=150.0)
    tp = se.timing_prior(legs, truth["f0"]["u0"])
    assert tp["frame_period_s"] == pytest.approx(1.0 / 150.0, rel=1e-6) and tp["pose_period_s"] is None
    assert tp["sigma_s"] == pytest.approx((1.0 / 150.0) / math.sqrt(12.0), rel=1e-6)
    for lg in legs:
        lg.meta["pose_period_s"] = 0.010
    tp = se.timing_prior(legs, truth["f0"]["u0"])
    assert tp["sigma_s"] == pytest.approx(math.sqrt((1.0 / 150.0) ** 2 + 0.010 ** 2) / math.sqrt(12.0), rel=1e-6)
    est = se.estimate(legs, prior)
    assert est.plate["time_offset"]["prior_sigma_s"] == pytest.approx(tp["sigma_s"], rel=1e-6)
    assert prior.dt_sigma_s != pytest.approx(tp["sigma_s"])          # the prior's (calibration) σ is not what the model uses
    # static grabs pair their frames with one still pose: the robot stamps stored next to them (the virtual arm: two, a dwell
    # apart) are no pose sampling of their frames and do not set T_pose (test_sim_teach1g m1: σ_δ 240 ms from three dwells)
    f0 = truth["f0"]
    dw = [make_dwell(truth["seam"], f0, name=f"dwell_{k}", s_mm=10.0 + k, rng=np.random.default_rng(k)) for k in range(3)]
    for d in dw:
        assert se.motion_sign(d, f0["u0"]) == 0
        d.meta["pose_period_s"] = 1.1
    tp2 = se.timing_prior(dw + legs, f0["u0"])
    assert tp2["pose_period_s"] == pytest.approx(0.010) and tp2["sigma_s"] == pytest.approx(tp["sigma_s"], rel=1e-9)


def test_direction_offset_and_time_offset_recovered_from_two_speeds():
    """Alternating passes at two speeds: the robot's direction offset b (w, n, u) and the time offset δ are separable (the
    along-track shift d·(v·δ − b_u) differs between the speeds) — both recovered within k·σ, δ carried by the data (not the prior);
    the leg offsets' variance components are per component (w, n)."""
    for i in range(2):
        legs, prior, truth = direction_scenario(np.random.default_rng(2100 + i))
        est = se.estimate(legs, prior)
        do = est.plate["direction_offset"]
        assert do["identified"] == [True, True, True], do
        for c in range(3):
            assert abs(do["b_m"][c] - truth["b_m"][c]) <= prior.k * do["sigma_m"][c], (c, do)
        assert do["sigma_m"][1] < 0.05e-3 and do["sigma_m"][2] < 0.1e-3, do
        to = est.plate["time_offset"]
        assert to["separable_from_b_u"] and to["data_share"] > 0.5, to
        assert abs(est.dt_s - truth["dt"]) <= prior.k * est.dt_sigma_s and est.dt_sigma_s < 0.5 * to["prior_sigma_s"], to
        assert len(est.plate["leg_offset_sigma_wn_m"]) == 2
        for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):          # the ends in the static frame
            e = est.ends[X]
            assert abs(e.s_m - e_true) <= max(prior.k * e.sigma_m, 1.5e-4), (X, e.s_m - e_true, e.sigma_m)


def test_one_speed_identifies_the_along_track_direction_term_not_its_split():
    """Alternating passes at one speed: c_v·δ and d·b_u are collinear — the combined along-track direction term A = v·δ − b_u is
    recovered within k·σ and the ends stay honest, but the run does not separate δ from b_u: δ stays at its prior (data share < ½,
    σ ≈ the timing prior's), b_u takes the rest with a σ that says so. b_w, b_n are identified."""
    legs, prior, truth = direction_scenario(np.random.default_rng(2200), legs_spec=((-1, 0.06), (1, 0.06), (-1, 0.06), (1, 0.06)))
    est = se.estimate(legs, prior)
    do = est.plate["direction_offset"]
    assert do["identified"] == [True, True, True], do
    for c in (0, 1):
        assert abs(do["b_m"][c] - truth["b_m"][c]) <= prior.k * do["sigma_m"][c], (c, do)
    to = est.plate["time_offset"]
    assert not to["separable_from_b_u"] and to["data_share"] < 0.5, to
    assert est.dt_sigma_s > 0.7 * to["prior_sigma_s"], to
    atd = to["along_track_direction"]
    A_true = atd["speed_mps"] * truth["dt"] - truth["b_m"][2]
    assert abs(atd["value_m"] - A_true) <= prior.k * atd["sigma_m"], (atd, A_true)
    assert do["sigma_m"][2] >= 0.7 * atd["speed_mps"] * to["prior_sigma_s"], do          # b_u alone is not known better than δ
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        e = est.ends[X]
        assert abs(e.s_m - e_true) <= max(prior.k * e.sigma_m, 1.5e-4), (X, e.s_m - e_true, e.sigma_m)


def test_direction_offset_needs_both_directions_rank_not_count():
    """b enters the model only where the data identify it (rank of the fixed-effect design): dwells (static) + a transit (one
    direction, its own class offset) + one pass → not identified (no value, σ ∞ — no false precision); a second pass in the other
    direction → identified. Passes all in one direction → not identified, and the delivered root is that direction's frame (the
    static frame is not claimed)."""
    rng = np.random.default_rng(2300)
    seam = Seam()
    legs0, prior, truth = scenario(rng=rng, seam=seam, n_fine=0, coarse=False)
    f0 = truth["f0"]
    eS, eE = truth["eS_m"] * 1e3, truth["eE_m"] * 1e3
    tr = make_leg(seam, f0, name="00_transit", cls="transit", direction=1, s_from_mm=eS + 30.0, s_to_mm=eE - 30.0,
                  standoff_mm=444.0, yaw_deg=3.0, rng=rng)
    dS = make_dwell(seam, f0, name="dwell_S", s_mm=eS + 30.0, rng=rng)
    b3 = 0.17e-3 * f0["n0"]
    p1 = make_leg(seam, f0, name="01_pass", cls="pass", direction=-1, s_from_mm=eE + 20.0, s_to_mm=eS - 20.0, standoff_mm=352.0,
                  pitch_deg=-5.0, offset_m=-b3, jitter_mm=0.02, rng=rng, travel_u=seam.u)
    p2 = make_leg(seam, f0, name="02_pass", cls="pass", direction=1, s_from_mm=eS - 20.0, s_to_mm=eE + 20.0, standoff_mm=352.0,
                  pitch_deg=-5.0, offset_m=b3, jitter_mm=0.02, rng=rng, travel_u=seam.u, flip_x=True)
    p3 = make_leg(seam, f0, name="03_pass", cls="pass", direction=-1, s_from_mm=eE + 20.0, s_to_mm=eS - 20.0, standoff_mm=352.0,
                  pitch_deg=-5.0, offset_m=-b3, jitter_mm=0.02, rng=rng, travel_u=seam.u)
    est = se.estimate([dS, tr, p1], prior)
    do = est.plate["direction_offset"]
    assert do["identified"] == [False, False, False] and all(math.isnan(v) for v in do["b_m"]), do
    assert all(math.isinf(v) for v in do["sigma_m"]) and do["leg_direction"] == {"dwell_S": 0, "00_transit": 1, "01_pass": -1}
    est = se.estimate([dS, tr, p1, p2], prior)
    do = est.plate["direction_offset"]
    assert do["identified"][:2] == [True, True], do
    assert abs(do["b_m"][1] - 0.17e-3) <= prior.k * do["sigma_m"][1], do
    # passes in one direction only: no b; the root is where that direction sees it (true − d·b... measured + o = true, o = d·b)
    est = se.estimate([p1, p3], prior)
    do = est.plate["direction_offset"]
    assert do["identified"] == [False, False, False], do
    L = se.LineModel.from_estimate(est)
    s = 0.5 * (est.ends["S"].s_m + est.ends["E"].s_m)
    _, tn = true_line_wn(truth, s)
    _, sn = L.sigma_wn(s)
    en = float((L.r(s) - f0["S0"]) @ f0["n0"])
    assert abs(en - (tn[0] - (-1) * 0.17e-3)) <= prior.k * float(sn) + 2e-5, (en - tn[0], float(sn))
    assert not est.plate["time_offset"]["separable_from_b_u"]


def test_leg_offset_variance_per_component():
    """The robot's leg-to-leg scatter is not isotropic (lab kv_first: normal 0.02–0.03 mm, lateral 0.018 mm about the direction
    means): η has one REML variance component per component, and the line stays honest in both."""
    rng = np.random.default_rng(2400)
    spec = tuple((-1 if k % 2 == 0 else 1, 0.03) for k in range(8))
    zs, ratio = [], []
    for i in range(2):
        legs, prior, truth = direction_scenario(np.random.default_rng(2400 + i), legs_spec=spec, eta_mm=(0.01, 0.12, 0.0),
                                                b_mm=(0.0, 0.15, 0.0), dt_true_s=0.0, fps=25.0, points=False)
        est = se.estimate(legs, prior)
        sw, sn = est.plate["leg_offset_sigma_wn_m"]
        ratio.append(sn / max(sw, 1e-9))
        L = se.LineModel.from_estimate(est)
        s = 0.5 * (est.ends["S"].s_m + est.ends["E"].s_m)
        tw, tn = true_line_wn(truth, s)
        lw, ln_ = L.sigma_wn(s)
        r = L.r(s)
        zs += [float(((r - truth["f0"]["S0"]) @ truth["f0"]["w0"] - tw[0]) / lw),
               float(((r - truth["f0"]["S0"]) @ truth["f0"]["n0"] - tn[0]) / ln_)]
        do = est.plate["direction_offset"]
        assert abs(do["b_m"][1] - 0.15e-3) <= prior.k * do["sigma_m"][1], do
    assert min(ratio) > 3.0, ratio
    assert np.all(np.abs(zs) <= 3.0), zs


def test_end_scatter_recovered_by_reml():
    """Per-reading groove-change scatter 0.6 mm (spec §4.4 ε) over 4 fine legs: the groove ends ("where is the seam") carry τ_e of
    that size and honest z-scores."""
    taus, zs = [], []
    for i in range(12):
        legs, prior, truth = scenario(rng=np.random.default_rng(400 + i), n_fine=4, end_scatter_mm=0.6, coarse=True)
        est = se.estimate(legs, prior)
        ge = est.plate["groove_ends"]
        taus.append(ge["tau_m"])
        for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
            zs.append((ge[X]["s_m"] - e_true) / ge[X]["sigma_m"])
    taus = np.asarray(taus) * 1e3
    assert 0.3 < np.median(taus) < 1.0, taus
    assert np.mean(np.abs(zs) <= 3.0) >= 0.9, zs


def test_one_sided_end_gets_length_prior_posterior():
    legs, prior, truth = scenario(rng=np.random.default_rng(31), n_fine=1, coarse=False,
                                  fine_spans=[(30.0, 1e3 * 0.0 + 380.0 + 60.0)])
    est = se.estimate(legs, prior)
    e = est.ends["S"]
    assert not e.two_sided
    assert est.plate["groove_ends"]["E"]["two_sided"]
    # S lies before the first groove profile; σ is the length prior's (≈ tol_L), not the profile spacing
    assert e.sigma_m > 5e-4 or not math.isfinite(e.sigma_m)
    pq = se.plan_quantities(est, prior)
    assert math.isfinite(pq["sigma_plan_S_m"])


def test_tau_unidentified_means_not_done_but_plannable():
    legs, prior, truth = scenario(rng=np.random.default_rng(41), n_fine=0)
    est = se.estimate(legs, prior)
    # σ_e always finite (coordinator 2026-10-09), but τ_e unidentified → the done test reports the ends unresolved
    assert math.isfinite(est.ends["S"].sigma_m) and not math.isfinite(est.tau_end_m)
    assert not est.done and "end_S" in est.unresolved and est.ratios["end_S"] == math.inf
    pq = se.plan_quantities(est, prior)
    assert not pq["pass_class_measured"]
    assert pq["sigma_plan_S_m"] >= 0.95 * prior.he_sigma_m   # σ_he inside σ_e via the transit class prior (r2)
    assert math.isfinite(pq["sigma_plan_S_m"]) and pq["psi_res_rad"] > 0
    assert pq["depth_m"] == pytest.approx(0.010, abs=2e-4)
    assert pq["e_S_m"] == pytest.approx(truth["eS_m"], abs=3e-3)
    assert est.plate["top_point_m"] is not None and math.isfinite(est.plate["sigma_top_m"])


def test_bevel_side_right_and_mirrored_sensor():
    seam = Seam(bevel_w=-1)
    legs, prior, truth = scenario(rng=np.random.default_rng(51), seam=seam, n_fine=2)
    est = se.estimate(legs, prior)
    assert est.bevel_sign_w == -1
    mid = est.stations[len(est.stations) // 2]
    assert mid.faces["bevel_right"]["normal"] is not None and mid.faces["wall_left"]["normal"] is not None
    assert mid.dims["bevel_angle_deg"]["value"] == pytest.approx(30.0, abs=0.3)


def test_no_groove_on_coarse_leg():
    legs, prior, truth = scenario(rng=np.random.default_rng(61), n_fine=0, lam_in=-40.0)
    est = se.estimate(legs, prior)
    assert not est.groove_found and not est.done


def test_runtime_three_legs_2000_profiles():
    legs, prior, truth = scenario(rng=np.random.default_rng(71), n_fine=2, fine_overscan_mm=110.0)
    assert sum(lg.n for lg in legs) >= 5700, [lg.n for lg in legs]
    t0 = time.perf_counter()
    se.estimate(legs, prior)
    assert time.perf_counter() - t0 <= 2.0


def test_plate_roll_and_misalignment():
    seam = Seam(roll_deg=4.0, misalign_deg=2.0)
    legs, prior, truth = scenario(rng=np.random.default_rng(81), seam=seam, n_fine=2)
    est = se.estimate(legs, prior)
    assert est.plate["roll_deg"] == pytest.approx(4.0, abs=0.2)
    assert est.plate["misalignment_deg"] == pytest.approx(2.0, abs=0.3)


def test_profile_predictor_reproduces_theta():
    """θ0 predicted from the estimate for a new fine leg ≈ the noise-free θ of that leg (WP1 start (2), §3.4)."""
    seam = Seam(roll_deg=3.0)
    legs, prior, truth = scenario(rng=np.random.default_rng(91), seam=seam, n_fine=2)
    est = se.estimate(legs, prior)
    lg = make_leg(seam, truth["f0"], name="new", cls="fine", direction=1, s_from_mm=-60, s_to_mm=400, standoff_mm=352,
                  pitch_deg=-5.0, travel_u=seam.u, sig_theta=np.zeros(10), jitter_mm=0.0, rng=np.random.default_rng(0))
    pred = se.profile_predictor(est, lg.T_base_opt, lg.bevel_x_sign, spec=prior.spec)
    ins = np.flatnonzero(lg._true["inside"])
    got = np.array([pred(i)[0] for i in ins[10:-10:25]])
    ref = lg.theta[ins[10:-10:25]]
    assert np.all(np.abs(got[:, :2] - ref[:, :2]) < 0.05)                     # root point (mm)
    assert np.all(np.abs(got[:, 2] - ref[:, 2]) < 2e-3)                       # α
    assert np.all(np.abs(got[:, 3:8] - ref[:, 3:8]) < np.array([0.05, 0.01, 0.05, 0.05, 0.05]))
    assert np.all(np.abs(got[:, 8:] - ref[:, 8:]) < 2e-3)                     # γ
    # (θ0, Σ_R): also beyond the ends (the extrapolated line); Σ_R positive definite, at least the class jitter
    th_b, cov_b = pred(0)
    assert np.all(np.isfinite(th_b)) and np.all(np.linalg.eigvalsh(cov_b) > 0)
    sz = est.sigma_zeta_m["pass"] * 1e3
    assert np.linalg.eigvalsh(pred(ins[200])[1]).min() >= 0.5 * sz ** 2


def test_coarse_bound_with_unknown_class_offset_is_ignored():
    """A coarse leg that stops inside the ends gives only bounds; with fine data its along-track class offset is free and unread,
    so its bounds must not truncate the ends (they would be off by up to σ_he)."""
    seam = Seam()
    legs, prior, truth = scenario(rng=np.random.default_rng(97), seam=seam, n_fine=2, beta_coarse_mm=(3.0, 0.0, 0.0))
    f0 = truth["f0"]
    eS, eE = truth["eS_m"] * 1e3, truth["eE_m"] * 1e3
    legs[0] = make_leg(seam, f0, name="00_coarse", cls="coarse", direction=1, s_from_mm=eS + 1.0, s_to_mm=eE - 1.0,
                       standoff_mm=444.0, yaw_deg=4.0, offset_m=np.array([3e-3, 0.0, 0.0]), rng=np.random.default_rng(1))
    est = se.estimate(legs, prior)
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        assert abs(est.ends[X].s_m - e_true) <= prior.k * est.ends[X].sigma_m + 1e-12


def test_leg_offsets_by_reml_when_repeatability_slot_empty():
    """rep_sigma_m = NaN (datasheet slot empty): σ_η is estimated by REML from ≥ 2 fine legs; injected leg offsets of 0.3 mm give
    σ_η of that size and an honest line (z within k at mid) instead of a pinned, over-confident one."""
    zs, sig = [], []
    for i in range(8):
        rng = np.random.default_rng(700 + i)
        offs = [tuple(rng.normal(size=3) * 0.3) for _ in range(4)]
        legs, prior, truth = scenario(rng=rng, n_fine=4, fine_offsets_mm=offs, rep_sigma_m=float("nan"), fps=25.0)
        est = se.estimate(legs, prior)
        assert est.plate["leg_offset_sigma_source"] == "reml"
        sig.append(est.plate["leg_offset_sigma_m"])
        L = se.LineModel.from_estimate(est)
        s = 0.5 * (est.ends["S"].s_m + est.ends["E"].s_m)
        tw, tn = true_line_wn(truth, s)
        # truth in the gauge frame: β_fine = 0 ⇒ the line sits at the mean of the true leg offsets (η ~ N(0, σ_η²) prior)
        r = L.r(s)
        sw, sn = L.sigma_wn(s)
        zs += [float(((r - truth["f0"]["S0"]) @ truth["f0"]["w0"] - tw[0]) / sw),
               float(((r - truth["f0"]["S0"]) @ truth["f0"]["n0"] - tn[0]) / sn)]
    assert 0.1e-3 < np.median(sig) < 0.6e-3, sig
    assert np.mean(np.abs(zs) <= 3.0) >= 0.85, zs


def test_groove_like_feature_beyond_end_is_not_the_seam():
    """Lab case (g1_batch1_t08 pass_1): beyond the S end a groove-like structure 20 mm lower (table level) gets λ > 0. The change
    point runs on λ^line (this seam's groove at the line), so the S end stays at the plate end."""
    legs, prior, truth = scenario(rng=np.random.default_rng(131), n_fine=2, fine_overscan_mm=40.0)
    lg = legs[1]
    sr = lg._true["sig_root"]
    fake = np.flatnonzero((sr < -0.010) & (sr > -0.030))           # 10..30 mm beyond S
    lg.log_bf[fake] = 500.0
    lg.theta[fake, 1] += 20.0                                        # 20 mm further from the sensor
    lg.theta[fake, 0] = lg.theta[fake, 0] + np.random.default_rng(0).normal(size=fake.size) * 8.0
    est = se.estimate(legs, prior)
    ge = est.plate["groove_ends"]["S"]
    assert abs(ge["s_m"] - truth["eS_m"]) < 1e-3, (ge["s_m"], truth["eS_m"])
    assert abs(est.ends["S"].s_m - truth["eS_m"]) < 1e-3                       # plate-top evidence is unaffected
    r = [x for x in ge["readings"] if x["leg"] == lg.name][0]
    assert r["kind"] == "two_sided" and abs(r["mean_m"] - truth["eS_m"]) < 1e-3


@pytest.mark.parametrize("h_mm", [0.0, 10.0])
def test_end_face_crossing_height_is_in_the_end_sigma(h_mm):
    """The groove evidence may change anywhere between the root (h = 0) and the plate top (h = D) as the pitched laser plane
    crosses the end face; with all fine passes at the same tilt h is not identified and the end σ carries c_n·D/√12."""
    zs = []
    for i in range(6):
        legs, prior, truth = scenario(rng=np.random.default_rng(900 + i), n_fine=3, end_height_mm=h_mm)
        est = se.estimate(legs, prior)
        for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
            ge = est.plate["groove_ends"][X]                 # the groove (λ) change: crossing height unknown in [0, D]
            zs.append((ge["s_m"] - e_true) / ge["sigma_m"])
            # c_n (pitch −5°) · D/√24: the pitch-0 coarse leg fixes only h_E − h_S (its length is h-free), the mean stays open
            assert ge["sigma_m"] >= 0.08 * 0.010 / math.sqrt(24) * 0.9
            assert est.ends[X].s_m == ge["s_m"]                                  # the seam ends are the groove's
    # h at an edge of its uniform prior sits √3 σ_h from the prior mean (moment-matched Gaussian) → |z| ≲ √3 + noise
    assert np.all(np.abs(zs) <= math.sqrt(3.0) + 1.5), zs


def test_end_coverage_with_crossing_height_uniform():
    rng = np.random.default_rng(950)
    zs = []
    for i in range(10):
        legs, prior, truth = scenario(rng=np.random.default_rng(960 + i), n_fine=3, end_height_mm=float(rng.uniform(0, 10)))
        est = se.estimate(legs, prior)
        zs += [(est.plate["groove_ends"][X]["s_m"] - e) / est.plate["groove_ends"][X]["sigma_m"]
               for X, e in (("S", truth["eS_m"]), ("E", truth["eE_m"]))]
    assert np.mean(np.abs(zs) <= 3.0) >= 0.95, zs
    assert np.sqrt(np.mean(np.square(zs))) < 2.0, zs


# ---------------------------------------------------------------- process frame of 2026-10-09: dwell + transit + passes
def make_dwell(seam, f0, *, name, s_mm, n_frames=30, standoff_mm=342.0, yaw_deg=0.0, lateral_mm=0.0, offset_m=(0, 0, 0),
               jitter_mm=0.05, frame_sd=None, rng=None):
    """Static grab: n_frames of one profile at one pose (common section jitter, independent frame noise)."""
    rng = np.random.default_rng(0) if rng is None else rng
    lg0 = make_leg(seam, f0, name=name, cls="fine", direction=1, s_from_mm=s_mm, s_to_mm=s_mm, standoff_mm=standoff_mm,
                   yaw_deg=yaw_deg, lateral_mm=lateral_mm, sig_theta=np.zeros(10), jitter_mm=0.0, offset_m=offset_m, rng=rng,
                   travel_u=seam.u)
    sd = np.array([0.02, 0.02, 0.002, 0.03, 0.005, 0.03, 0.02, 0.02, 0.002, 0.002]) if frame_sd is None else frame_sd
    th0 = lg0.theta[0].copy()
    th0[:2] += rng.normal(size=2) * jitter_mm
    th = th0 + rng.normal(size=(n_frames, 10)) * sd
    lg = LegObs(name=name, cls="dwell", direction=0, t_acq_s=np.arange(n_frames) * 0.01,
                T_base_opt=np.repeat(lg0.T_base_opt, n_frames, axis=0), v_opt_mps=np.zeros((n_frames, 3)), theta=th,
                cov=np.broadcast_to(np.diag(sd ** 2), (n_frames, 10, 10)).copy(), log_bf=np.full(n_frames, 40.0),
                sigma_r_mm=np.full(n_frames, 0.05), eps=np.full(n_frames, 0.01), n_valid=np.full(n_frames, 1000),
                converged=np.ones(n_frames, bool), bevel_x_sign=lg0.bevel_x_sign, meta={"n_points": 1408, "site": name[-1]})
    lg._truth0 = lg0
    return lg


def process_frame_scenario(rng, seam=None, he_mm=(1.5, -1.0, 2.0), dwell_he_mm=(-1.0, 0.8, -1.2), n_pass=2, dt_true_s=0.0):
    seam = Seam() if seam is None else seam
    legs, prior, truth = scenario(rng=rng, seam=seam, n_fine=n_pass, coarse=False, fine_overscan_mm=20.0, dt_true_s=dt_true_s)
    f0 = truth["f0"]
    eS, eE = truth["eS_m"] * 1e3, truth["eE_m"] * 1e3
    tr = make_leg(seam, f0, name="00_transit", cls="transit", direction=1, s_from_mm=eS + 30.0, s_to_mm=eE - 30.0,
                  standoff_mm=444.0, yaw_deg=3.0, offset_m=np.asarray(he_mm) * 1e-3, rng=rng, dt_true_s=dt_true_s)
    dS = make_dwell(seam, f0, name="dwell_S", s_mm=eS + 30.0, offset_m=np.asarray(dwell_he_mm) * 1e-3, rng=rng)
    dE = make_dwell(seam, f0, name="dwell_E", s_mm=eE - 30.0, offset_m=np.asarray(dwell_he_mm) * 1e-3, rng=rng)
    for lg in legs:
        lg.cls = "pass"
    return [dS, dE, tr] + legs, prior, truth


def test_dwell_transit_pass_fusion():
    zs, zl = [], []
    for i in range(6):
        legs, prior, truth = process_frame_scenario(np.random.default_rng(1000 + i))
        est = se.estimate(legs, prior)
        assert est.groove_found and est.plate["gauge_class"] == "pass"
        assert set(est.class_offset_m) >= {"pass", "transit", "dwell_S", "dwell_E"}
        for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
            assert math.isfinite(est.ends[X].sigma_m)
            zs.append((est.ends[X].s_m - e_true) / est.ends[X].sigma_m)
        L = se.LineModel.from_estimate(est)
        s = 0.5 * (est.ends["S"].s_m + est.ends["E"].s_m)
        tw, tn = true_line_wn(truth, s)
        sw, sn = L.sigma_wn(s)
        r = L.r(s)
        zl += [float(((r - truth["f0"]["S0"]) @ truth["f0"]["w0"] - tw[0]) / sw),
               float(((r - truth["f0"]["S0"]) @ truth["f0"]["n0"] - tn[0]) / sn)]
        # the transit's class offset (w, n parts) is recovered (truth 1.5/−1.0/2.0 mm in base; compare the ⟂-u parts)
        bt = est.class_offset_m["transit"]
        bt_true = np.array([1.5, -1.0, 2.0]) * 1e-3
        for ax in (truth["f0"]["w0"], truth["f0"]["n0"]):
            assert abs(float(bt @ ax) - float(bt_true @ ax)) < 2e-4
    assert np.all(np.abs(zs) <= 3.0), zs
    assert np.mean(np.abs(zl) <= 3.0) >= 0.9, zl


def test_unobserved_ends_have_finite_prior_sigma():
    """Dwells + a transit that stays inside the specimen: no end is seen; σ_e is the model's prior interval (contract + design
    length + bounds), finite, the true end inside it, and the done test reports the ends unresolved."""
    legs, prior, truth = process_frame_scenario(np.random.default_rng(1100), n_pass=0)
    est = se.estimate(legs, prior)
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        e = est.ends[X]
        assert math.isfinite(e.sigma_m) and e.sigma_m > 1e-3 and not e.two_sided
        assert abs(e.s_m - e_true) <= math.sqrt(3) * e.sigma_m + 1e-9           # inside the uniform interval
    assert {"end_S", "end_E"} <= set(est.unresolved)
    pq = se.plan_quantities(est, prior)
    assert math.isfinite(pq["sigma_plan_S_m"]) and not pq["pass_class_measured"]


class _Calib:
    T_flange_opt = np.eye(4)                     # flange = optical frame in these tests


def _dwell_at_pose(seam, f0, T_opt, name="d", rng=None, sd=None):
    """One grab (N = 1) whose optical frame is exactly T_opt (exact θ through WP1's geometry, small fit noise)."""
    o, ex, ey, ez = T_opt[:3, 3], T_opt[:3, 0], T_opt[:3, 1], T_opt[:3, 2]
    pts = {}
    for k, X0 in seam.lines().items():
        sg = float((o - X0) @ ey) / float(ey @ seam.u)
        P = X0 + sg * seam.u - o
        pts[k] = np.array([[P @ ex, P @ ez]]) * 1e3
    bxs = 1 if float(pts["B0"][0, 0] - pts["Q0"][0, 0]) > 0 else -1
    e1 = bxs * (pts["B0"] - pts["Q0"])[0]
    al = math.atan2(e1[1], e1[0])
    th0 = np.array([[pts["R"][0, 0], pts["R"][0, 1], al, 5.0, math.radians(30), 0.0, 10.0, 10.0, 0.0, 0.0]])
    th = _theta_wp1(th0, bxs, {"top_corner_L": pts["Cb"], "root_face_top_L": pts["B1"], "root_edge_L": pts["B0"],
                               "root_edge_R": pts["Q0"], "top_corner_R": pts["Cs"]},
                    {"top_L": _plate_dir2(seam.n_bev, ey, ex, ez), "top_R": _plate_dir2(seam.n_sq, ey, ex, ez)})
    sd = np.array([0.02, 0.02, 0.002, 0.03, 0.005, 0.03, 0.02, 0.02, 0.002, 0.002]) if sd is None else sd
    lg = LegObs(name=name, cls="dwell", direction=0, t_acq_s=np.zeros(1), T_base_opt=T_opt[None].copy(), v_opt_mps=np.zeros((1, 3)),
                theta=th, cov=np.diag(sd ** 2)[None], log_bf=np.array([400.0]), sigma_r_mm=np.array([0.05]), eps=np.array([0.01]),
                n_valid=np.array([1000]), converged=np.array([True]), bevel_x_sign=bxs, meta={"n_points": 1408, "site": "S"})
    return lg


def _pose_over(seam, s_sig, *, lateral_mm, standoff_mm, roll_deg=0.0, yaw_deg=0.0, pitch_deg=0.0):
    root = seam.point(s_sig)
    ez = -seam.n
    ey = seam.u
    R = _rot(np.cross(seam.u, seam.n), math.radians(pitch_deg))             # optical axis tilted along the seam
    R = _rot(seam.u, math.radians(roll_deg)) @ R
    R = _rot(seam.n, math.radians(yaw_deg)) @ R
    ez, ey = R @ ez, R @ ey
    ex = np.cross(ey, ez)
    T = np.eye(4)
    T[:3, :3] = np.stack([ex, ey, ez], axis=1)
    T[:3, 3] = root - standoff_mm * 1e-3 * ez + lateral_mm * 1e-3 * ex
    return T


def test_site_state_target_centres_the_groove_and_stops():
    seam = Seam(roll_deg=3.0)
    legs, prior, truth = scenario(rng=np.random.default_rng(1200), seam=seam, n_fine=0)
    f0 = truth["f0"]
    T0 = _pose_over(seam, 0.05, lateral_mm=7.0, standoff_mm=330.0, roll_deg=-4.0)
    g1 = _dwell_at_pose(seam, f0, T0, "dwell_S1")
    view = {"coarse_plate_m": 0.300}
    prior.sigma_eps = float("nan")                                                          # datasheet slot empty (lab)
    st = se.site_state([g1], None, prior, "S", T_now=T0, calib=_Calib, view=view, goal_tol_m=0.5e-3)
    assert st["ok"] and abs(st["x_R_m"] - (-0.007)) < 2e-4 and st["needs_correction"]          # off-axis, σ_ε unknown → ∞
    assert not math.isfinite(st["sigma_lat_m"])
    T1 = st["T_target"]
    g2 = _dwell_at_pose(seam, f0, T1, "dwell_S2")
    st2 = se.site_state([g1, g2], None, prior, "S", T_now=T1, calib=_Calib, view=view, goal_tol_m=0.5e-3)
    assert abs(st2["x_R_m"]) < 0.5e-3                                                       # centred within e_goal
    assert abs(st2["standoff_err_m"]) < 0.3e-3 and abs(st2["roll_rad"]) < math.radians(0.2)
    assert not st2["needs_correction"]                                                     # D15 stops at T*
    # with a published x-scale linearity the systematic term σ_ε·|x_R| decides: large σ_ε → correct; tiny σ_ε → do not
    prior.sigma_eps = 0.05
    assert se.site_state([g1], None, prior, "S", T_now=T0, calib=_Calib, view=view, goal_tol_m=0.5e-3)["needs_correction"]
    prior.sigma_eps = 1e-5
    assert not se.site_state([g1], None, prior, "S", T_now=T0, calib=_Calib, view=view, goal_tol_m=0.5e-3)["needs_correction"]


def test_site_state_second_site_yaw_to_the_estimate():
    seam = Seam()
    legs, prior, truth = scenario(rng=np.random.default_rng(1250), seam=seam, n_fine=2)
    est = se.estimate(legs, prior)
    T0 = _pose_over(seam, 0.30, lateral_mm=-3.0, standoff_mm=310.0, yaw_deg=6.0)
    g = _dwell_at_pose(seam, truth["f0"], T0, "dwell_E1")
    st = se.site_state([g], est, prior, "E", T_now=T0, calib=_Calib, view={"coarse_plate_m": 0.300}, goal_tol_m=0.5e-3)
    assert st["yaw_rad"] is not None and abs(abs(math.degrees(st["yaw_rad"])) - 6.0) < 0.3
    g2 = _dwell_at_pose(seam, truth["f0"], st["T_target"], "dwell_E2")
    st2 = se.site_state([g, g2], est, prior, "E", T_now=st["T_target"], calib=_Calib, view={"coarse_plate_m": 0.300},
                        goal_tol_m=0.5e-3)
    assert abs(math.degrees(st2["yaw_rad"])) < 0.3 and abs(st2["x_R_m"]) < 0.5e-3


def test_site_state_target_keeps_the_root_in_the_laser_plane_with_a_pitched_axis():
    """Roll and yaw do not remove a pitch of the optical axis along û: T* moves the origin along the rotated optical axis, so the
    root stays in the laser plane and on the axis (moving along n̂ would shift the plane (D + r_t)·tan(pitch) along the seam)."""
    seam = Seam(roll_deg=4.0)
    legs, prior, truth = scenario(rng=np.random.default_rng(1260), seam=seam, n_fine=2)
    est = se.estimate(legs, prior)
    r_t = 0.443
    for site, e in (("S", None), ("E", est)):
        T0 = _pose_over(seam, 0.30, lateral_mm=60.0, standoff_mm=450.0, yaw_deg=20.0, pitch_deg=1.3)
        g = _dwell_at_pose(seam, truth["f0"], T0, f"dwell_{site}1")
        st = se.site_state([g], e, prior, site, T_now=T0, calib=_Calib, view={"coarse_plate_m": r_t}, goal_tol_m=0.5e-3)
        T1 = st["T_target"]
        R, n = np.asarray(st["root_m"]), np.asarray(st["n"])
        o1, X1, Y1, Z1 = T1[:3, 3], T1[:3, 0], T1[:3, 1], T1[:3, 2]
        assert abs(float((R - o1) @ Y1)) < 1e-9                                   # root in the new laser plane
        assert abs(float((R - o1) @ X1)) < 1e-9                                   # on the optical axis (sensor x = 0)
        assert abs(float((o1 - R) @ n) - (st["depth_m"] + r_t)) < 1e-9            # plate top at the view's plate distance
        g2 = _dwell_at_pose(seam, truth["f0"], T1, f"dwell_{site}2")
        st2 = se.site_state([g, g2], e, prior, site, T_now=T1, calib=_Calib, view={"coarse_plate_m": r_t}, goal_tol_m=0.5e-3)
        assert abs(st2["x_R_m"]) < 0.5e-3 and abs(st2["standoff_err_m"]) < 0.3e-3
        assert abs(float((np.asarray(st2["root_m"]) - R) @ seam.u)) < 1e-3              # the same section (not shifted along û)


def test_profile_predictor_from_dict_and_derived_side():
    seam = Seam()
    legs, prior, truth = scenario(rng=np.random.default_rng(1300), seam=seam, n_fine=2)
    est = se.estimate(legs, prior)
    lg = legs[1]
    p_obj = se.profile_predictor(est, lg.T_base_opt, lg.bevel_x_sign, spec=prior.spec)
    p_dict = se.profile_predictor(est.to_dict(), lg.T_base_opt, None, spec=prior.spec)
    i = int(np.flatnonzero(lg._true["inside"])[200])
    assert np.allclose(p_obj(i)[0], p_dict(i)[0], atol=1e-9) and np.allclose(p_obj(i)[1], p_dict(i)[1], rtol=1e-6)


def test_gross_end_reading_is_downweighted():
    """Lab case (transit stopping inside the specimen with λ < 0 on its stationary tail): one reading 28 mm short must not set
    τ_e or move the ends (Huber-robust fusion, τ_e by REML on the weighted readings)."""
    legs, prior, truth = process_frame_scenario(np.random.default_rng(1400))
    tr = [lg for lg in legs if lg.cls == "transit"][0]
    sr = tr._true["sig_root"]
    tail = np.flatnonzero(sr > sr.max() - 0.0004)                    # last 0.4 mm of the transit (stationary-like tail)
    tr.log_bf[tail] = -90.0
    est = se.estimate(legs, prior)
    bad = [r for r in est.plate["groove_ends"]["E"]["readings"] if r["leg"] == tr.name][0]
    assert bad["kind"] == "two_sided" and bad["weight"] < 0.1
    ge = est.plate["groove_ends"]["E"]
    assert abs(ge["s_m"] - truth["eE_m"]) <= 3.0 * ge["sigma_m"] and ge["sigma_m"] < 1e-3
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        assert abs(est.ends[X].s_m - e_true) <= 3.0 * est.ends[X].sigma_m and est.ends[X].sigma_m < 1e-3


def test_per_plate_ends_recorded():
    """Per-plate ends (provisional report until WP1 exports a per-plate presence log-BF): both plates run 4 mm past the groove at S and
    only the bevel plate 3 mm past E — recorded per plate (left/right), the inner one named; the seam ends are the groove's."""
    legs, prior, truth = scenario(rng=np.random.default_rng(1500), n_fine=3, fine_overscan_mm=20.0,
                                  plate_ext_mm=((4.0, 3.0), (4.0, 0.0)))
    est = se.estimate(legs, prior)
    pe = est.plate["plate_ends"]
    eS, eE = truth["eS_m"], truth["eE_m"]
    du = 1e-3 / float(truth["seam"].u @ truth["f0"]["u0"])                    # 1 mm along the seam in taught s
    bev, sq = ("left", "right") if est.bevel_sign_w > 0 else ("right", "left")
    for side in (bev, sq):
        assert abs(pe["S"][side]["s_m"] - (eS - 4 * du)) < 5e-4
    assert abs(pe["E"][bev]["s_m"] - (eE + 3 * du)) < 5e-4 and abs(pe["E"][sq]["s_m"] - eE) < 5e-4
    assert pe["E"]["inner"] == sq
    assert abs(est.ends["S"].s_m - eS) <= 3 * est.ends["S"].sigma_m and abs(est.ends["E"].s_m - eE) <= 3 * est.ends["E"].sigma_m


# ---------------------------------------------------------------- review-fix round (2026-10-10)
def test_stationary_profiles_at_a_leg_edge_have_finite_resolution():
    """at23: a pass that rests at its start right at the end (profiles at one s with mixed evidence) must give a reading with the
    resolution floor (spacing²/12 + class jitter), never σ = 0, and the fusion must not go singular."""
    legs, prior, truth = scenario(rng=np.random.default_rng(1600), n_fine=2, fine_overscan_mm=0.0)
    lg = legs[1]
    sr = lg._true["sig_root"]
    k0 = int(np.argmin(np.abs(sr - 0.0)))                          # the profile at the S end
    n_rest = 40                                                     # the leg rests there: identical poses, mixed evidence
    for name in ("t_acq_s", "T_base_opt", "v_opt_mps", "theta", "cov", "log_bf", "sigma_r_mm", "eps", "n_valid", "converged"):
        a = getattr(lg, name)
        setattr(lg, name, np.concatenate([np.repeat(a[k0:k0 + 1], n_rest, axis=0), a]))
    lg.v_opt_mps[:n_rest] = 0.0
    lg.log_bf[:n_rest] = np.where(np.arange(n_rest) % 2 == 0, 40.0, -40.0)
    lg.meta["face_inliers"] = np.concatenate([np.repeat(np.asarray(lg.meta["face_inliers"])[k0:k0 + 1], n_rest, 0),
                                              np.asarray(lg.meta["face_inliers"])]).tolist()
    est = se.estimate(legs, prior)
    for X in ("S", "E"):
        assert math.isfinite(est.ends[X].sigma_m) and est.ends[X].sigma_m > 0
        for r in est.ends[X].readings:
            if r["kind"] == "two_sided":
                assert r["sigma_m"] >= 0.9 * r["floor_m"] > 0


def test_end_never_crossed_cannot_read_as_resolved():
    """Passes stop inside the E end (never bracketed beyond it): the E end's grid posterior is the bounds × the length prior from
    the S end — wide and not resolved, with the truth inside its spread."""
    legs, prior, truth = scenario(rng=np.random.default_rng(1700), n_fine=3, coarse=False,
                                  fine_spans=[(truth_s - 20.0, truth_e - 15.0) for truth_s, truth_e in [(-28.69, 350.65)] * 3])
    est = se.estimate(legs, prior)
    eE = est.ends["E"]
    assert not eE.two_sided
    assert "end_E" in est.unresolved and est.ratios["end_E"] > 1.0
    assert abs(eE.s_m - truth["eE_m"]) <= 3.0 * eE.sigma_m


def test_bevel_side_from_pooled_evidence_not_the_first_grab():
    """A ghost first grab votes the wrong side weakly; the later legs vote the right side strongly: the pooled ln BF decides, and
    legs fitted with the other mirror are reported and give no section dims."""
    legs, prior, truth = process_frame_scenario(np.random.default_rng(1800))
    f0 = truth["f0"]
    right = 1
    for lg in legs:
        ex = np.median(lg.T_base_opt[:, :3, 0], axis=0)
        sx = 1.0 if float(ex @ f0["w0"]) >= 0 else -1.0
        lg.meta["bevel_side_ln_bf"] = (-30.0 if lg.name == "dwell_S" else 400.0) * right * sx
    est = se.estimate(legs, prior)
    assert est.bevel_sign_w == 1
    assert est.plate["bevel_side"]["pooled_ln_bf_w"] > 0 and est.plate["bevel_side"]["mismatched_legs"] == []
    # the same run with one pass fitted with the other mirror: reported, and its profiles carry no section dims
    lg = [l for l in legs if l.cls == "pass"][0]
    lg.bevel_x_sign = -lg.bevel_x_sign
    est2 = se.estimate(legs, prior)
    assert lg.name in est2.plate["bevel_side"]["mismatched_legs"]
    assert est2.bevel_sign_w == 1


def test_pooled_dims_ignore_stations_without_information():
    st = []
    for j in range(40):
        st.append(se.StationEst(s_m=j * 0.005, root_m=np.zeros(3), root_sigma_m=(1e-4, 1e-4), u=np.array([1.0, 0, 0]),
                                n=np.array([0, 0, 1.0]), w=np.array([0, 1.0, 0]),
                                dims={q: {"value": 10.0 + 0.01 * math.sin(j), "sigma": 0.01, "sigma_sys": 0.0, "n": 50}
                                      for q in se.DIM_NAMES}, faces={}, n_profiles=50))
    st[3].dims["top_opening_mm"] = {"value": 1e9, "sigma": 1e6, "sigma_sys": 0.0, "n": 1}        # an end station gone wild
    st[7].dims["top_opening_mm"] = {"value": 10.0, "sigma": 1e5, "sigma_sys": 0.0, "n": 1}      # no information
    d = se.seam_dims(st)["top_opening_mm"]
    assert abs(d["value"] - 10.0) < 0.01 and d["sigma"] < 0.01 and d["along_seam_sd"] < 0.05


def test_sections_doc_carries_the_pooled_dims_that_gate_done():
    from scan_core import g1_outputs as go
    legs, prior, truth = scenario(rng=np.random.default_rng(1900), n_fine=2)
    est = se.estimate(legs, prior)
    doc = go.sections_doc(est, prior, {})
    sd = doc["seam_dims"]
    assert set(sd) >= {"gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm"}
    assert sd["gap_mm"]["gates_done"] and sd["gap_mm"]["ratio"] == pytest.approx(est.ratios["gap_mm"], rel=1e-6)
    assert not sd["hi_lo_mm"]["gates_done"]


def test_local_root_excursion_stays_this_seam():
    """Lab at17: the transit's root reading steps 0.5–0.8 mm sideways over 60 mm (tack / local fit bias, 6–9 σ_ζ) near the E end. The
    location term of λ^line is the line fit's own residual law (Huber), not its Gaussian core: the excursion stays this seam, the leg
    does not read a false end 90 mm short."""
    legs, prior, truth = process_frame_scenario(np.random.default_rng(1700))
    lg = [x for x in legs if x.cls == "pass"][0]
    L = truth["seam"].L
    sr = lg._true["sig_root"]
    k = np.flatnonzero((sr > 0.68 * L) & (sr < 0.86 * L))
    lg.theta[k, 0] += 0.8
    est = se.estimate(legs, prior)
    r = [x for x in est.plate["groove_ends"]["E"]["readings"] if x["leg"] == lg.name][0]
    assert r["kind"] == "two_sided" and abs(r["mean_m"] - truth["eE_m"]) < 1e-3, (r["mean_m"], truth["eE_m"])
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        assert abs(est.ends[X].s_m - e_true) <= 3.0 * est.ends[X].sigma_m and est.ends[X].sigma_m < 1e-3


def test_stationary_leg_beyond_the_end_is_not_this_seam():
    """Lab at10 pass_3: a pass that rests 20 mm beyond the E end (repeatability slot empty → σ_η by REML) sees a groove-like structure
    21 mm deeper with fits scattered ±15 mm, 15 % of them at the line's lateral place. Its own leg offset must not make it 'this seam'
    (λ^line predictive over η, high-breakdown start of σ_ζ/σ_η), it must not inflate σ_η, τ_e or the ends (lab: τ_e 182 mm)."""
    rng = np.random.default_rng(1800)
    legs, prior, truth = process_frame_scenario(rng)
    prior.rep_sigma_m = float("nan")
    f0, seam = truth["f0"], truth["seam"]
    s_rest = truth["eE_m"] * 1e3 + 20.0
    lg0 = make_leg(seam, f0, name="03_pass", cls="pass", direction=1, s_from_mm=s_rest, s_to_mm=s_rest + 0.5, standoff_mm=352.0,
                   pitch_deg=-5.0, travel_u=seam.u, end_shift_mm=(-1e3, 1e3), rng=rng)
    n = 111
    lg = LegObs(name="03_pass", cls="pass", direction=1, t_acq_s=np.arange(n) * 0.01,
                T_base_opt=np.repeat(lg0.T_base_opt[:1], n, axis=0), v_opt_mps=np.zeros((n, 3)),
                theta=np.repeat(lg0.theta[:1], n, axis=0), cov=np.repeat(lg0.cov[:1], n, axis=0),
                log_bf=34.0 + rng.normal(size=n) * 10.0, sigma_r_mm=np.full(n, 0.05), eps=np.full(n, 0.01),
                n_valid=np.full(n, 1000), converged=np.ones(n, bool), bevel_x_sign=lg0.bevel_x_sign,
                meta={"n_points": 1408})
    lg.theta[:, 1] += 21.0
    lg.theta[:, :2] += rng.normal(size=(n, 2)) * 0.05
    scat = rng.random(n) < 0.85
    lg.theta[scat, 0] += rng.normal(size=int(scat.sum())) * 15.0
    est = se.estimate(legs + [lg], prior)
    ge = est.plate["groove_ends"]                          # the groove change points: where the seam is (no measured points here)
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        assert abs(ge[X]["s_m"] - e_true) <= 3.0 * ge[X]["sigma_m"] and ge[X]["sigma_m"] < 1e-3, (X, ge[X])
        assert est.ends[X].s_m == ge[X]["s_m"]
        for r in ge[X]["readings"]:
            if r["leg"] == lg.name and r["kind"] == "two_sided":
                assert r.get("weight", 1.0) < 0.1
    assert est.plate["leg_offset_sigma_m"] < 1e-3
    assert math.isfinite(ge["tau_m"]) and ge["tau_m"] < 1e-3


def _check_plate_ends(est, truth, faces, k, tol_m):
    ef = est.plate["end_faces"]
    for X in ("S", "E"):
        for pl, key in (("bev", "L"), ("sq", "R")):
            q = ef["plates"].get(f"{X}:{key}")
            if q is None:
                continue
            ref = _s_of(truth, faces[(X, pl)][0])
            assert abs(q["s_m"] - ref) <= max(k * q["sigma_m"], tol_m), (X, key, q["s_m"] - ref, q["sigma_m"])
    tr = {"S": max(faces[("S", "bev")][0], faces[("S", "sq")][0]), "E": min(faces[("E", "bev")][0], faces[("E", "sq")][0])}
    for X, sig in tr.items():
        e = ef["ends"][X]
        assert abs(e["root"]["s_m"] - _s_of(truth, sig)) <= max(k * e["root"]["sigma_m"], tol_m), (X, e["root"], _s_of(truth, sig))
        assert e["top"]["s_m"] == e["root"]["s_m"]                       # square: the top edge ends where the root ends


def test_end_faces_square_ends_ideal_edge():
    """Plate ends from the measured top edges (user 2026-10-10/11): square plate ends (declared joint) — one end per plate at every
    height, inner of the two plates; the plates end at different s (the square plate 0.4 mm beyond at E, 0.3 mm at S). An ideal
    sensor (no sheet thickness): the edge spread estimate goes to the low end of its prior and the ends are read at the last top
    points."""
    legs, prior, truth = scenario(rng=np.random.default_rng(11), n_fine=3)
    _side_evidence(legs, truth)
    seam = truth["seam"]
    L = seam.L
    faces = {("S", "bev"): (0.0, 0.0), ("S", "sq"): (-0.0003, 0.0), ("E", "bev"): (L, 0.0), ("E", "sq"): (L + 0.0004, 0.0)}
    with_points(legs, truth, faces)
    est = se.estimate(legs, prior)
    ef = est.plate["end_faces"]
    assert {r["leg"] for r in ef["readings"]} <= {lg.name for lg in legs if lg.cls == "fine"}
    assert ef["sensor"]["sigma_L_m"] < 2e-4, ef["sensor"]
    _check_plate_ends(est, truth, faces, prior.k, 2.5e-4)
    assert est.ends["S"].two_sided and est.ends["E"].two_sided
    assert math.isfinite(est.tau_end_m) and est.done
    s0, _ = se.end_at(est, "S", 0.0)
    sT, _ = se.end_at(est, "S", 0.010)
    assert s0 == sT == pytest.approx(ef["ends"]["S"]["root_combined"]["s_m"])


@pytest.mark.parametrize("pitch_deg", [-5.0, 5.0])
def test_end_faces_through_the_sensor_edge_response(pitch_deg):
    """The sloped points at a square plate's top edge are the sensor's response (Curless & Levoy 1995: the laser sheet's thickness
    truncated by the edge, read as a height change through the triangulation): a rise at one end, a droop at the other. The edge
    spread σ_L is estimated from the data (shared by every leg, end and plate) and the ends are read at the square edge — no
    geometric chamfer; the λ ends are not used (design length never pulls them)."""
    legs, prior, truth = scenario(rng=np.random.default_rng(7), n_fine=3, pitch_deg=pitch_deg)
    _side_evidence(legs, truth)
    seam = truth["seam"]
    L = seam.L
    faces = {("S", "bev"): (0.0, 0.0), ("S", "sq"): (-0.0003, 0.0), ("E", "bev"): (L, 0.0), ("E", "sq"): (L + 0.0004, 0.0)}
    with_points(legs, truth, faces)
    apply_edge_response(legs, truth, faces, sigma_l_mm=0.5)
    est = se.estimate(legs, prior)
    sen = est.plate["end_faces"]["sensor"]
    assert 0.35e-3 < sen["sigma_L_m"] < 0.7e-3 and abs(sen["theta_tri_deg"] - 26.0) < 4.0, sen
    _check_plate_ends(est, truth, faces, prior.k, 1.5e-4)
    assert est.done


def test_end_faces_ignore_tack_welds_and_the_backing_beyond_the_ends():
    """Tack welds on the end faces between the backing bar and the plates (below the section, user 2026-10-11) and the backing bar
    reaching beyond the plate ends: material outside the declared joint — never plate, never end — so the ends do not move."""
    legs, prior, truth = scenario(rng=np.random.default_rng(11), n_fine=3)
    seam = truth["seam"]
    L = seam.L
    faces = {("S", "bev"): (0.0, 0.0), ("S", "sq"): (0.0, 0.0), ("E", "bev"): (L, 0.0), ("E", "sq"): (L, 0.0)}
    with_points(legs, truth, faces, tack_mm=(4.0, 3.0), backing_ext_mm=12.0)
    apply_edge_response(legs, truth, faces, sigma_l_mm=0.5)
    est = se.estimate(legs, prior)
    _check_plate_ends(est, truth, faces, prior.k, 1.5e-4)


def test_end_faces_calibrated_laser_pitch_sigma_is_carried_not_estimated():
    """A laser-plane pitch error w_x of the calibration (1.5°, σ stated 1.5°) moves the ends read at the top edge by h*·g·w_x: the
    model carries it with the calibration's σ (the readings of one tool orientation cannot identify it — apparent end slants are the
    sensor's edge response, not w_x, lab 2026-10-11), so the ends stay within their σ and w_x keeps its prior σ."""
    legs, prior, truth = scenario(rng=np.random.default_rng(11), n_fine=3)
    seam = truth["seam"]
    L = seam.L
    faces = {("S", "bev"): (0.0, 0.0), ("S", "sq"): (0.0, 0.0), ("E", "bev"): (L, 0.0), ("E", "sq"): (L, 0.0)}
    with_points(legs, truth, faces, w_x=math.radians(1.5))
    prior.laser_pitch_sigma_rad = math.radians(1.5)
    est = se.estimate(legs, prior)
    ef = est.plate["end_faces"]
    assert ef["w_x"]["sigma_rad"] == pytest.approx(math.radians(1.5), rel=0.05)
    _check_plate_ends(est, truth, faces, prior.k, 0.0)


def test_end_faces_calibrated_edge_response_is_not_re_estimated():
    """The sensor's edge response from the calibration (σ_L, θ and their σ; calib g1: edge_response): used as it is — no per-scan
    re-estimation — and its σ enters the fusion through each reading's derivatives (common terms)."""
    legs, prior, truth = scenario(rng=np.random.default_rng(7), n_fine=4)
    _side_evidence(legs, truth)
    seam = truth["seam"]
    L = seam.L
    faces = {("S", "bev"): (0.0, 0.0), ("S", "sq"): (-0.0003, 0.0), ("E", "bev"): (L, 0.0), ("E", "sq"): (L + 0.0004, 0.0)}
    with_points(legs, truth, faces)
    apply_edge_response(legs, truth, faces, sigma_l_mm=0.5, theta_deg=26.0)
    prior.edge_response = {"sigma_l_m": 0.5e-3, "sigma_l_log_sd": 0.05, "theta_tri_deg": 26.0, "theta_tri_sigma_deg": 1.0,
                           "source": "test calibration"}
    est = se.estimate(legs, prior)
    ef = est.plate["end_faces"]
    assert ef["sensor"]["source"] == "test calibration" and ef["sensor"]["sigma_L_m"] == 0.5e-3
    assert all(math.isfinite(r["d_log_sigma_L"]) and math.isfinite(r["d_theta_rad"]) for r in ef["readings"])
    _check_plate_ends(est, truth, faces, prior.k, 1.5e-4)
    assert ef["variance_components"]["established"] and est.done


def test_end_sigma_needs_redundant_crossings():
    """Small samples: the leg-end offsets' variance is estimated from the crossings beyond the plate ends' own unknowns (ν). Two
    crossing passes leave ν = 2 — the posterior of that variance has no finite mean, so the ends' σ is not established and the done
    test cannot pass; four passes establish it."""
    for n_fine, ok in ((2, False), (4, True)):
        legs, prior, truth = scenario(rng=np.random.default_rng(5), n_fine=n_fine)
        _side_evidence(legs, truth)
        faces = with_points(legs, truth)
        apply_edge_response(legs, truth, faces)
        est = se.estimate(legs, prior)
        ef = est.plate["end_faces"]
        assert ef["variance_components"]["established"] == ok, (n_fine, ef["variance_components"])
        for X in ("S", "E"):
            assert math.isfinite(ef["ends"][X]["root"]["sigma_m"]) == ok
            assert math.isfinite(est.ratios["end_" + X]) == ok
        assert est.done == ok


def _synthetic_end_readings(rng, legs=4, eS=-0.0376, eE=0.3423, sd_leg=6e-5, sd_read=3e-5):
    out = []
    for k in range(legs):
        d = 1 if k % 2 else -1
        for X, e in (("S", eS), ("E", eE)):
            z = rng.normal() * sd_leg
            for p in ("L", "R"):
                out.append({"end": X, "plate": p, "leg": f"pass{k + 1}", "cls": "pass", "direction": d, "c_v": 0.03 * d,
                            "h_star_m": 0.01, "pitch_gain": 0.1, "c_u": 1.0, "top_h_m": 0.01, "u1_m": e + z + rng.normal() * sd_read,
                            "u1_var": sd_read ** 2, "region_m": (e - 0.008, e + 0.008)})
    return out


def test_end_fusion_rejects_a_gross_misreading():
    """A reading locked on a ghost 6.8 mm beyond the end (narrow, confident): the fusion's gross-error class takes it — its weight
    goes to ~0, the plate end stays, and the per-reading scatter τ is not inflated by it."""
    from types import SimpleNamespace
    from scan_core import end_faces as ef_
    rng = np.random.default_rng(3)
    rd = _synthetic_end_readings(rng)
    bad = [i for i, r in enumerate(rd) if r["leg"] == "pass2" and r["end"] == "S" and r["plate"] == "R"][0]
    rd[bad]["u1_m"] = -0.0444
    rd[bad]["u1_var"] = (1e-5) ** 2
    prior = SimpleNamespace(dt_sigma_s=0.001, rep_sigma_m=float("nan"), he_sigma_m=0.002, laser_pitch_sigma_rad=float("nan"))
    fz = ef_.fuse_end_faces(rd, prior, {"gauge": "pass", "eta_legs": []}, {}, {})
    w = np.asarray(fz["huber_weights"])
    assert w[bad] < 0.01 and np.all(np.delete(w, bad) > 0.5), w
    q = fz["plates"][("S", "R")]
    assert abs(q["s_m"] + 0.0376) <= 3.0 * math.sqrt(q["var"]) and math.sqrt(q["var"]) < 2e-4, q
    assert fz["tau_e_m"] < 1e-4 and fz["established"]


def test_end_faces_need_a_leg_that_crosses_the_end():
    """A pass that stops short of S gives no S end-face reading (its groove change point at S is a bound): the S end stays the
    groove-evidence end ("where is the seam", for the next pass) and is not resolved."""
    legs, prior, truth = scenario(rng=np.random.default_rng(4), n_fine=1, fine_spans=[(30.0, 420.0)])
    with_points(legs, truth)
    est = se.estimate(legs, prior)
    ef = est.plate["end_faces"]
    assert not any(r["end"] == "S" for r in ef["readings"])
    assert not est.ends["S"].two_sided and not est.done
    assert est.ends["S"].s_m == pytest.approx(est.plate["groove_ends"]["S"]["s_m"])


# ---------------------------------------------------------------- §4.9 model fit + hypotheses (2026-10-11)
def _side_evidence(legs, truth, per_profile=5.0, sign=1):
    """The mirrored-template evidence a real run carries in its first grab / transit (WP1 meta['bevel_side_ln_bf'], sensor-x sign):
    per_profile nats per profile for the true side (sign −1: for the other side) on the coarse leg."""
    f0 = truth["f0"]
    for lg in legs:
        if lg.cls == "coarse":
            ex = np.median(lg.T_base_opt[:, :3, 0], axis=0)
            lg.meta["bevel_side_ln_bf"] = float(sign) * per_profile * lg.n * truth["seam"].bw * (1.0 if ex @ f0["w0"] >= 0 else -1.0)
    return legs


def _blocking(est, kind=None):
    return [b for b in est.plate["model_checks"]["blocking"] if kind is None or b["kind"] == kind]


def test_segment_mixture_accounts_for_multiplicity():
    """H0 (all segments from the predictive): no segment of a family of 400 is a misfit (|z| up to ~3.5 occurs) — ε of the family
    shrinks with n (Scott & Berger); one gross segment gets γ → 1 and the verdict 'misfit'; an uninformative family stays at ε ≈ ½."""
    rng = np.random.default_rng(5)
    z = rng.normal(size=400)
    lnf = -0.5 * z * z - 0.5 * math.log(2 * math.pi * 0.05 ** 2)          # σ 0.05 mm, extent 100 mm
    lo, g, eps = se.segment_mixture(lnf, math.log(100.0))
    verd = [se.segment_verdict(2 * x, 3.0) for x in lo]
    assert verd.count("misfit") == 0 and eps < 0.01 and np.max(np.abs(z)) > 2.5
    z2 = np.r_[z[:50], 40.0]
    lnf2 = -0.5 * z2 * z2 - 0.5 * math.log(2 * math.pi * 0.05 ** 2)
    lo2, g2, _ = se.segment_mixture(lnf2, math.log(100.0))
    assert se.segment_verdict(2 * lo2[-1], 3.0) == "misfit" and g2[-1] > 0.999 and np.all(g2[:50] < 1e-3)
    assert se.segment_verdict(2 * lo2[0], 3.0) == "fits"


def test_clean_runs_fit_and_resolve_every_check():
    """No false alarm: clean synthetic runs (outliers 2 %, random yaw/roll) with the mirrored-template evidence — no done-test quantity
    rests on a misfit or an unresolved hypothesis, every leg shows this seam and every station window fits (end READINGS may be
    down-weighted: the generator's per-reading λ noise makes some end readings scatter by millimetres — a reading's indicator, it
    blocks nothing), the done test is the k·σ ≤ tol test alone."""
    for i in range(6):
        rng = np.random.default_rng(2100 + i)
        seam = Seam(yaw_deg=rng.uniform(-5, 5), roll_deg=rng.uniform(-4, 4))
        legs, prior, truth = scenario(rng=rng, seam=seam, n_fine=3, fps=25.0, outlier_frac=0.02)
        est = se.estimate(_side_evidence(legs, truth), prior)
        mc = est.plate["model_checks"]
        assert not mc["blocking"], mc["summary"]
        assert all(l_["verdict"] == "fits" for l_ in mc["legs"]), mc["legs"]
        assert all(w_["this_seam"] == "supported" and w_.get("verdict") == "fits" for w_ in mc["windows"]["stations"])
        assert all(l_["verdict"] == "this seam" for l_ in mc["legs_this_seam"]) and not mc["summary"]["legs_not_this_seam"]
        assert mc["hypotheses"]["bevel_side"]["resolved"]
        assert est.unresolved == [q for q, r in est.ratios.items() if not r <= 1.0]


def test_bevel_side_is_a_hypothesis_never_an_override():
    """The pooled mirrored-template ln BF is the side's log odds: with the evidence the side is resolved; with a prior side that the
    evidence contradicts (the job's side, SeamPrior.bevel_sign_w) the decided side has 2 ln odds << −k² → every done-test quantity is
    unresolved (never done); with no leg fitted with both mirrors the side is not established either."""
    import dataclasses
    legs, prior, truth = scenario(rng=np.random.default_rng(2200), n_fine=3)
    _side_evidence(legs, truth)
    est = se.estimate(legs, prior)
    h = est.plate["model_checks"]["hypotheses"]["bevel_side"]
    assert h["resolved"] and h["two_ln_odds"] > 9.0 and h["from"] == "evidence" and est.bevel_sign_w == truth["seam"].bw
    wrong = se.estimate(legs, dataclasses.replace(prior, bevel_sign_w=-truth["seam"].bw))
    hw = wrong.plate["model_checks"]["hypotheses"]["bevel_side"]
    assert not hw["resolved"] and hw["two_ln_odds"] < -9.0 and hw["from"] == "prior (job)"
    assert not wrong.done and set(wrong.unresolved) == set(wrong.ratios)
    for lg in legs:
        lg.meta.pop("bevel_side_ln_bf", None)
    none = se.estimate(legs, prior)
    assert not none.plate["model_checks"]["hypotheses"]["bevel_side"]["resolved"] and not none.done


def test_window_off_the_line_is_measured_by_its_own_data():
    """A 30 mm stretch where every pass reads the root 0.6 mm aside (12 σ_ζ per profile — still 'this seam' per profile through the
    Huber tail of λ^line): the windows there do not fit the line + waviness model (2·ln O < −k²). The root's local deviation is a robust
    random effect (Box & Tiao): those stations take the root from their own window (shifted ≈ 0.6 mm, σ ≈ the window's), τ_root is
    estimated without them, the other stations keep the line; nothing blocks — the deviation is measured, not a failure."""
    legs, prior, truth = scenario(rng=np.random.default_rng(2300), n_fine=3, fps=25.0)
    _side_evidence(legs, truth)
    ref = se.estimate(legs, prior)
    for lg in legs:
        if lg.cls == "fine":
            sr = lg._true["sig_root"]
            on = (sr > 0.180) & (sr < 0.210)
            lg.theta[on, 0] += 0.6 * (1.0 if lg.bevel_x_sign > 0 else -1.0)
    est = se.estimate(legs, prior)
    mc = est.plate["model_checks"]
    off = [w for w in mc["windows"]["stations"] if w.get("verdict") == "misfit"]
    assert off and mc["summary"]["windows_root_from_own_data"] == len(off)
    s_lo = truth["eS_m"] + 0.180 / float(truth["seam"].u @ truth["f0"]["u0"])
    assert all(s_lo - 0.012 <= w["s_m"] <= s_lo + 0.042 for w in off), ([w["s_m"] for w in off], s_lo)
    assert not _blocking(est, "misfit") and est.done == ref.done
    by_s = {round(st.s_m, 6): st for st in est.stations}
    for w in off:
        st = by_s[round(w["s_m"], 6)]
        shift = float(np.hypot(*w["root_shift_mm"]))
        assert 0.45 < shift < 0.7 and max(st.root_sigma_m) < 1e-4, (shift, st.root_sigma_m)
    assert mc["windows"]["tau_root_mm"][0] < 0.05                         # the deviation is not waviness of the whole seam


def test_leg_that_is_not_this_seam_is_downweighted_and_the_estimate_stands():
    """Lab g1d_p15_04: a pass whose template fits lock onto other features (its one-step-ahead λ — WP1 against the estimate before it —
    is negative for almost every profile while its free λ is positive and its roots scatter ±14 mm). The leg's 'this seam' indicator
    is decisively negative, the leg is down-weighted (EM), and the estimate is the one without it: the same ends and plate top, its
    readings are no observation of the ends."""
    legs, prior, truth = scenario(rng=np.random.default_rng(2700), n_fine=2, fps=25.0)
    _side_evidence(legs, truth)
    for lg in legs:
        lg.meta["log_bf_kind"] = "free"
    ref = se.estimate(legs[:2], prior)                                     # coarse + pass 1
    bad = legs[2]
    rng = np.random.default_rng(3)
    inside = bad._true["inside"]
    bad.meta["log_bf_free"] = np.where(inside, 60.0, bad.log_bf).tolist()   # groove-like (free λ > 0) …
    bad.log_bf = np.where(inside, -12.0, bad.log_bf)                      # … but not this seam at the predicted root
    bad.meta["log_bf_kind"] = "seam"
    bad.theta[inside, 0] += rng.choice([-14.0, 14.0], size=int(inside.sum()))
    est = se.estimate(legs, prior)
    mc = est.plate["model_checks"]
    sup = {l_["segment"]: l_ for l_ in mc["legs_this_seam"]}
    assert sup["leg:" + bad.name]["verdict"] == "not this seam" and sup["leg:" + bad.name]["gamma_out"] > 0.999, sup
    assert all(v["verdict"] == "this seam" for k_, v in sup.items() if k_ != "leg:" + bad.name)
    assert mc["summary"]["legs_not_this_seam"] == [bad.name] and mc["em_rounds"] == 2
    for X in ("S", "E"):
        assert abs(est.ends[X].s_m - ref.ends[X].s_m) < 0.2 * ref.ends[X].sigma_m + 1e-5, (X, est.ends[X].s_m, ref.ends[X].s_m)
        assert est.ends[X].sigma_m < 1.2 * ref.ends[X].sigma_m
        assert all(r["leg"] != bad.name for r in est.ends[X].readings if r["kind"] == "two_sided" and r.get("weight", 0) > 1e-3)
    assert abs(float((est.plate["top_point_m"] - ref.plate["top_point_m"]) @ truth["f0"]["n0"])) < 2e-4


@pytest.mark.parametrize("leg_w", [0.0, 0.2])
def test_not_this_seam_leg_alone_in_its_direction_identifies_nothing(leg_w):
    """test_sim_teach1g m2 (2026-10-12, w_x calibration): the only pass of one motion direction is not this seam (fitted with the
    wrong-side template; its change points 80 / 10 mm off). Its readings are no observation of the ends — at leg weight 0 or at a
    soft weight with the 'not this seam' decision (the estimate's last EM round): the direction offset b_u is not in the groove
    fusion (the other passes are all one direction), the ends and their σ are those of the fusion without that pass. Before: its
    readings kept b_u in the model; the ends and b_u shared a direction without (or with only the rejected readings') curvature,
    the mode drifted along it (here −0.33 mm, σ 1.05 instead of 0.49 mm; lab-like m2: e_S −229 mm and a negative Laplace
    variance — math domain error in the bound term)."""
    legs, prior, truth = scenario(rng=np.random.default_rng(2700), n_fine=3, fps=25.0)      # coarse +1, fine −1 / +1 / −1
    bad = legs[2]
    line = se.fit_line(legs, prior)
    assert [se.leg_direction(line, lg.name) for lg in legs] == [1, -1, 1, -1]

    def readings(with_bad: bool):
        out = []
        for lg in legs:
            if lg is bad and not with_bad:
                continue
            cp = se.leg_change_points(lg, line)
            for X, shift in (("S", 0.080), ("E", 0.010)):
                r = cp[X]
                if r is None:
                    continue
                r["leg_w"], r["leg_not_this_seam"] = (leg_w, True) if lg is bad else (1.0, False)
                if lg is bad and r["kind"] == "two_sided":
                    r["mean_m"] += shift
                    r["_curve"] = None
            out += [cp["S"], cp["E"]]
        return out
    ref = se.fuse_ends(readings(False), prior, line, length_prior=False)
    rd = readings(True)
    assert sum(1 for r in rd if r is not None and r["leg"] == bad.name and r["kind"] == "two_sided") == 2
    fz = se.fuse_ends(rd, prior, line, length_prior=False)
    assert "b_u" not in fz["names"] and fz["b_u_identified"] is False, fz["names"]
    for X in ("S", "E"):
        e, e0 = fz["ends"][X], ref["ends"][X]
        assert math.isfinite(e["sigma_m"]) and 0.0 < e["sigma_m"] < 1e-3, (X, e["sigma_m"])
        assert abs(e["s_m"] - e0["s_m"]) < 1e-6 and e["sigma_m"] == pytest.approx(e0["sigma_m"], rel=0.01), (X, e, e0)
        assert all(r["weight"] == 0.0 for r in e["readings"] if r["leg"] == bad.name and r["kind"] == "two_sided")
    assert np.all(np.diag(fz["cov"]) >= 0.0)


def test_dims_are_the_passes_a_biased_coarse_leg_does_not_move_them():
    """The section dims are the gauge class's (user 2026-10-12): a coarse leg whose section reads 0.8 mm wider and 3° steeper (its
    own pose-class systematic — lab kv_offA_a, transit 10° off the seam: gap 6.90, bevel 23.4° vs the passes' 5.17 / 30.5) leaves the
    dims of one pass unchanged; with no pass there are no dims."""
    legs, prior, truth = scenario(rng=np.random.default_rng(2900), n_fine=1)
    assert legs[0].cls == "coarse" and legs[1].cls == "fine"
    ref = se.estimate(legs, prior)
    legs[0].theta[:, 3] += 0.8                                           # θ = (x_R, z_R, α, G, β, …): gap
    legs[0].theta[:, 4] += math.radians(3.0)
    est = se.estimate(legs, prior)
    for q in ("gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm"):
        d, d0 = est.plate["seam_dims"][q], ref.plate["seam_dims"][q]
        assert math.isfinite(d["sigma"]) and abs(d["value"] - d0["value"]) < 0.05 * d0["sigma"], (q, d, d0)
    est0 = se.estimate(legs[:1], prior)
    assert all(not math.isfinite(est0.plate["seam_dims"][q]["sigma"]) for q in ("gap_mm", "depth_mm", "bevel_angle_deg")), \
        est0.plate["seam_dims"]


def test_normal_inverse_on_the_identified_subspace():
    """The fusion's Laplace covariance: N = AᵀA with a direction no row informs (e_S, e_E and b_u move together) — functionals in
    that direction have an infinite variance, the identified ones (the length e_E − e_S, e_S − b_u) a finite non-negative one equal to
    the reduced model's; full rank = the plain inverse; the Newton step solves the identified part and leaves the unidentified
    direction where it was (in the unit-free Jacobi-scaled coordinates)."""
    A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])         # two readings of one direction: e_X + b_u
    N = A.T @ np.diag([4.0, 9.0, 1.0]) @ A
    Ni = se._NormalInverse(N)
    assert not Ni.full
    assert math.isinf(Ni.var([1.0, 0.0, 0.0])) and math.isinf(Ni.var([0.0, 0.0, 1.0]))
    v_len = Ni.var([-1.0, 1.0, 0.0])
    assert v_len == pytest.approx(1.0 / 5.0 + 1.0 / 9.0)                       # e_S + b_u from two rows (4 + 1), e_E + b_u from one (9)
    assert Ni.var([1.0, 0.0, 1.0]) == pytest.approx(1.0 / 5.0)
    C = Ni.cov()
    assert np.all(np.isinf(np.diag(C)))
    x0 = np.array([0.1, 0.4, 0.0])
    x1 = Ni.step(N @ np.array([0.2, 0.5, 0.3]), x0)
    assert x1[0] + x1[2] == pytest.approx(0.5) and x1[1] + x1[2] == pytest.approx(0.8)
    null = np.array([1.0, 1.0, -1.0])                                          # N·null = 0
    assert np.allclose(N @ null, 0.0)
    assert float(null @ (np.diag(N) * (x1 - x0))) == pytest.approx(0.0, abs=1e-12)   # scaled: (D⁻¹null)·(D⁻¹Δx) = 0, D⁻² = diag N
    M = np.array([[4.0, 1.0], [1.0, 3.0]])
    Mi = se._NormalInverse(M)
    assert Mi.full and np.allclose(Mi.cov(), np.linalg.inv(M)) and Mi.var([1.0, -1.0]) >= 0.0


@pytest.mark.parametrize("hole_mm,where", [(12.0, "station window"), (30.0, "end region S")])
def test_stretch_that_is_not_this_seam_is_a_misfit(hole_mm, where):
    """A stretch where no profile of any leg shows this seam at the line (ghost / foreign structure: groove evidence absent, junk θ):
    short (12 mm) — the change point bridges it, the stations there are decisively not this seam (2 ln BF < −k²): the line through
    them is not a measurement; long (30 mm) — the single groove segment splits there (a plate split, g3_14) and the S end jumps to the
    stretch while both plates are present to the true end: the end region is a misfit. Never done."""
    legs, prior, truth = scenario(rng=np.random.default_rng(2400), n_fine=2, fps=25.0)
    _side_evidence(legs, truth)
    rng = np.random.default_rng(1)
    for lg in legs:
        sr = lg._true["sig_root"]
        on = (sr > 0.150) & (sr < 0.150 + hole_mm * 1e-3)
        lg.log_bf[on] = -40.0
        lg.theta[on, 0] += rng.normal(size=on.sum()) * 5.0
        lg.theta[on, 1] += 2.5
    est = se.estimate(legs, prior)
    mis = _blocking(est, "misfit")
    assert any(b["segment"].startswith(where) for b in mis), (where, [b["segment"] + ": " + b["why"] for b in mis])
    assert not est.done


@pytest.mark.parametrize("ext_mm,verdict", [(0.0, "fits"), (25.0, "misfit")])
def test_end_region_groove_versus_both_plates(ext_mm, verdict):
    """End region (user 2026-10-09: the seam ends where both plates are present): the groove change point and the plates' own
    presence end are two readings of one end. Plates that run 25 mm past the groove end (a groove-like jig, a ghost stretch or a
    groove stopping short) → decisive misfit on end_E; no extension → fits."""
    legs, prior, truth = scenario(rng=np.random.default_rng(2500), n_fine=3, fine_overscan_mm=40.0,
                                  plate_ext_mm=((0.0, ext_mm), (0.0, ext_mm)))
    _side_evidence(legs, truth)
    est = se.estimate(legs, prior)
    reg = est.plate["model_checks"]["ends"]["E"]["region"]
    assert reg["verdict"] == verdict, reg
    blocked = any(b["kind"] == "misfit" and b["quantities"] == ["end_E"] for b in _blocking(est))
    assert blocked == (verdict == "misfit")


def test_outlying_leg_is_downweighted_by_its_indicator():
    """Repeatability from the datasheet (σ_η 0.05 mm): one of five passes sits 1.5 mm aside (a robot error, 30 σ_η) — its profiles are
    not this seam at the line under the leg-offset predictive, the leg's indicator is decisive, the EM weight takes it out of the line
    (the line follows the others within k·σ), nothing blocks. The outlier has replication in its motion direction (passes 1, 3, 5
    run E→S): a pass that is the only one of its direction is fitted exactly by the robot's direction offset b and cannot be told
    from it (2026-10-11 direction model)."""
    legs, prior, truth = scenario(rng=np.random.default_rng(2600), n_fine=5, rep_sigma_m=5e-5,
                                  fine_offsets_mm=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    _side_evidence(legs, truth)
    est = se.estimate(legs, prior)
    mc = est.plate["model_checks"]
    sup = {l_["segment"]: l_ for l_ in mc["legs_this_seam"]}
    assert sup["leg:03_fine"]["verdict"] == "not this seam" and sup["leg:03_fine"]["gamma_out"] > 0.99, sup
    assert all(sup[f"leg:0{k}_fine"]["verdict"] == "this seam" for k in (1, 2, 4, 5))
    assert mc["summary"]["legs_not_this_seam"] == ["03_fine"] and not mc["blocking"]
    L = se.LineModel.from_estimate(est)
    s = 0.5 * (est.ends["S"].s_m + est.ends["E"].s_m)
    tw, tn = true_line_wn(truth, s)
    sw, sn = L.sigma_wn(s)
    r = L.r(s)
    assert abs((r - truth["f0"]["S0"]) @ truth["f0"]["w0"] - tw[0]) <= 3 * sw + 1e-12
    assert abs((r - truth["f0"]["S0"]) @ truth["f0"]["n0"] - tn[0]) <= 3 * max(float(sn), 2e-5) + 1e-4
    assert est.plate["model_checks"]["em_rounds"] >= 2


def test_station_dims_report_a_foreign_section():
    """Station dims (report only): a stretch of stations with another section (bevel 0° — a square jig) is a misfit of its pooled dim
    family and takes no weight in the pooled dim; a taper does not."""
    from scan_core.g1_types import StationEst
    sts = []
    for j in range(60):
        b = 0.0 if 50 <= j < 56 else 30.0 + 0.01 * j
        dims = {nm: {"value": v, "sigma": 0.03, "sigma_sys": 0.0, "n": 50.0} for nm, v in
                (("gap_mm", 5.0 + 0.005 * j), ("depth_mm", 10.0), ("bevel_angle_deg", b), ("top_opening_mm", 10.8),
                 ("hi_lo_mm", 0.0), ("root_face_mm", 0.0))}
        sts.append(StationEst(s_m=0.005 * j, root_m=np.zeros(3), root_sigma_m=(1e-5, 1e-5), u=np.array([1.0, 0, 0]),
                              n=np.array([0, 0, 1.0]), w=np.array([0, 1.0, 0]), dims=dims, faces={}, n_profiles=50))
    ext = {"gap_mm": 100.0, "top_opening_mm": 100.0, "depth_mm": 300.0, "hi_lo_mm": 300.0, "root_face_mm": 300.0,
           "bevel_angle_deg": 90.0}
    d = se.seam_dims(sts, extents=ext)
    bf = d["bevel_angle_deg"]["station_fit"]
    assert bf["n_misfit"] == 6 and abs(d["bevel_angle_deg"]["value"] - 30.3) < 0.2, (bf, d["bevel_angle_deg"]["value"])
    assert d["gap_mm"]["station_fit"]["n_misfit"] == 0


def test_joint_off_the_design_is_a_warning_never_a_misfit():
    """User rule 2026-10-11: the measured joint 6 mm longer than the design and with a 6.0 mm gap (design 5.0, tol 0.5) is reported as
    a WARNING with the measured values (status unchanged); no segment check uses a design value — nothing blocks, every reading and
    window fits, the measured length and gap are the data's (not pulled to the design)."""
    seam = Seam(L_mm=386.5, G=6.0)
    legs, prior, truth = scenario(rng=np.random.default_rng(2800), seam=seam, n_fine=3, fps=25.0)
    _side_evidence(legs, truth)
    est = se.estimate(legs, prior)
    mc = est.plate["model_checks"]
    assert not mc["blocking"] and mc["summary"]["misfit"] == 0, mc["summary"]
    dd = est.plate["design_deviations"]
    L_true = (truth["eE_m"] - truth["eS_m"]) * float(truth["seam"].u @ truth["f0"]["u0"]) ** -1 * 1e3
    assert dd["weld_length_mm"]["warning"] and abs(dd["weld_length_mm"]["measured"] - L_true) <= 3 * dd["weld_length_mm"]["sigma"]
    assert dd["gap_mm"]["warning"] and abs(dd["gap_mm"]["measured"] - 6.0) < 0.1
    assert any(w.startswith("weld_length_mm measured") for w in est.plate["warnings"])
    assert any(w.startswith("gap_mm measured") for w in est.plate["warnings"])


def test_station_root_follows_a_real_waviness_blup():
    """User 2026-10-11 'deliver the measured values as they are': the root wavers 0.15 mm (period 120 mm) about the straight line in
    every leg's data. The delivered station root = line + the BLUP of its local deviation (Henderson / Robinson 1991, REML τ_root):
    it follows the waviness (|deviation| − |wave| within k·σ at every station) with the conditional σ (< the marginal √(σ_line² + τ²)),
    while the line stays straight (planning)."""
    legs, prior, truth = scenario(rng=np.random.default_rng(2900), n_fine=3, fps=25.0, jitter_mm=0.03)
    _side_evidence(legs, truth)
    amp, per = 0.15, 0.120
    for lg in legs:
        sr = lg._true["sig_root"]
        lg.theta[:, 0] += amp * np.sin(2 * math.pi * sr / per) * (1.0 if lg.bevel_x_sign > 0 else -1.0)
    est = se.estimate(legs, prior)
    L = se.LineModel.from_estimate(est)
    tau = est.plate["model_checks"]["windows"]["tau_root_mm"]
    assert 0.05 < tau[0] < 0.2, tau
    dev, wave, ok = [], [], []
    for st in est.stations:
        d = (np.asarray(st.root_m) - L.r(st.s_m)) @ np.asarray(st.w) * 1e3
        sig_true = (st.s_m - (truth["seam"].A - truth["f0"]["S0"]) @ truth["f0"]["u0"]) / float(truth["seam"].u @ truth["f0"]["u0"])
        dev.append(d)
        wave.append(amp * math.sin(2 * math.pi * sig_true / per))
        sw = st.root_sigma_m[0] * 1e3
        lw = float(L.sigma_wn(st.s_m)[0]) * 1e3
        ok.append(sw < math.hypot(lw, tau[0]))
    dev, wave = np.array(dev), np.array(wave)
    c = float(np.corrcoef(dev, wave)[0, 1])
    sgn = 1.0 if c > 0 else -1.0
    assert abs(c) > 0.95, c
    sig = np.array([st.root_sigma_m[0] * 1e3 for st in est.stations])
    assert np.all(np.abs(sgn * dev - wave) <= 3.0 * sig + 0.01), np.max(np.abs(sgn * dev - wave) / sig)
    assert np.mean(ok) > 0.9


# ---- calibrated variance components (noise priors), structural b_u identification, settling, pose smoothing (g1-3pass) ----
def _noise_priors(sz_m=2.5e-4, tau_m=5e-5, nu0=30.0, v=0.06, eta_m=(1.5e-5, 2.5e-5)):
    """A calib 'g1: noise_priors' block whose population scale at speed v is the given σ (all speed dependence in the timing part
    for the crossing, none for the reading)."""
    return {"end_crossing": {"static_var_m2": 0.0, "timing_var_s2": sz_m ** 2 / v ** 2, "nu0": nu0},
            "end_reading": {"static_var_m2": tau_m ** 2, "timing_var_s2": 0.0, "nu0": nu0},
            "leg_offset_w": {"static_var_m2": eta_m[0] ** 2, "timing_var_s2": 0.0, "nu0": nu0},
            "leg_offset_n": {"static_var_m2": eta_m[1] ** 2, "timing_var_s2": 0.0, "nu0": nu0}}


def test_noise_priors_parse_and_speed_model():
    from scan_core.noise_priors import parse_noise_priors
    np_ = parse_noise_priors(_noise_priors())
    assert set(np_) == {"end_crossing", "end_reading", "leg_offset_w", "leg_offset_n"}
    p = np_["end_crossing"]
    assert p.s2(0.06) == pytest.approx(2.5e-4 ** 2) and p.s2(0.03) == pytest.approx(0.25 * 2.5e-4 ** 2)
    assert parse_noise_priors({"end_crossing": {"nu0": 3.0}}) == {}                    # no scale: not calibrated
    assert parse_noise_priors(None) == {}


def test_calibrated_end_noise_gives_honest_sigma_after_two_passes():
    """Two opposite passes (one crossing per end and direction): without a calibration the crossing scatter is unknown (ν = 0, σ
    infinite); with the calibrated population prior (ν0) the ends' σ is finite and honest — the plate ends' z-scores over many
    runs are N(0, 1)-like (per-crossing σ_ζ 0.25 mm, per-reading τ 0.05 mm, as the 60 mm/s lab runs)."""
    from types import SimpleNamespace
    from scan_core import end_faces as ef_
    prior0 = SimpleNamespace(dt_sigma_s=0.001, rep_sigma_m=float("nan"), he_sigma_m=0.002, laser_pitch_sigma_rad=float("nan"),
                             noise_priors=None)
    rng = np.random.default_rng(77)
    rd = _synthetic_end_readings(rng, legs=2, sd_leg=2.5e-4, sd_read=5e-5)
    fz = ef_.fuse_end_faces(rd, prior0, {"gauge": "pass", "eta_legs": []}, {}, {})
    assert not fz["established"] and not math.isfinite(fz["ends"]["S"]["root"]["sigma_m"])
    prior = SimpleNamespace(**{**vars(prior0), "noise_priors": _noise_priors()})
    zs = []
    for i in range(60):
        rd = _synthetic_end_readings(np.random.default_rng(1000 + i), legs=2, sd_leg=2.5e-4, sd_read=5e-5)
        for r in rd:
            r["c_v"] = 0.06 * r["direction"]
        fz = ef_.fuse_end_faces(rd, prior, {"gauge": "pass", "eta_legs": []}, {}, {})
        assert fz["established"] and fz["nu_posterior"] > 2
        for (X, p), e in ((("S", "L"), -0.0376), (("E", "R"), 0.3423)):
            q = fz["plates"][(X, p)]
            zs.append((q["s_m"] - e) / math.sqrt(q["var"]))
    zs = np.asarray(zs)
    assert 0.7 < float(np.sqrt(np.mean(zs ** 2))) < 1.3, zs
    assert np.mean(np.abs(zs) <= 2.0) > 0.85


def test_calibrated_leg_offsets_establish_the_root_with_two_passes():
    """Two opposite passes: the direction offset b and the gauge intercept take both legs' offsets (ν = 0 per component); with the
    calibrated leg-offset prior the root is established and its σ carries the population's leg scatter."""
    spec = ((-1, 0.06), (1, 0.06))
    legs, prior, truth = direction_scenario(np.random.default_rng(2600), legs_spec=spec, eta_mm=(0.015, 0.025, 0.0),
                                            b_mm=(0.0, 0.15, 0.0), dt_true_s=0.0, fps=25.0, points=False)
    est0 = se.estimate(legs, prior)
    assert est0.plate["root_established"] == [False, False]
    prior.noise_priors = _noise_priors(eta_m=(1.5e-5, 2.5e-5))
    est = se.estimate(legs, prior)
    assert est.plate["root_established"] == [True, True], est.plate.get("leg_offset_nu")
    assert all(src.startswith("posterior") for src in est.plate["leg_offset_sigma_wn_source"])
    sw, sn = est.plate["leg_offset_sigma_wn_m"]
    assert 0.5e-5 < sw < 4e-5 and 1e-5 < sn < 6e-5, (sw, sn)
    s = 0.5 * (est.ends["S"].s_m + est.ends["E"].s_m)
    L = se.LineModel.from_estimate(est)
    lw, ln_ = L.sigma_wn(s)
    assert float(ln_) >= 0.5 * 2.5e-5 / math.sqrt(2.0), float(ln_)          # the leg scatter is in the root σ (two legs)
    assert math.isfinite(est.ratios["root_normal"])


def test_direction_offset_identification_is_structural():
    """Change-point readings of ONE motion direction with per-reading c_u slightly different from 1 (the plane's yaw): b_u is not
    identified (g1c_04 without pass 2 had b_u 7.8 m and ends at -7.8 m … 124 m when the rank followed the c_u values)."""
    line = {"gauge": "pass", "leg_direction": {"p1": -1, "p3": -1}, "offsets": {}}
    two = [{"end": "E", "cls": "pass", "leg": "p1", "c_u": 1.0001, "c_v": -0.03, "c_n": 0.1, "kind": "two_sided"},
           {"end": "S", "cls": "pass", "leg": "p3", "c_u": 1.0003, "c_v": -0.03, "c_n": 0.1, "kind": "two_sided"},
           {"end": "E", "cls": "pass", "leg": "p3", "c_u": 1.0004, "c_v": -0.03, "c_n": 0.1, "kind": "two_sided"}]
    assert not se._b_u_identified(two, line)
    line["leg_direction"]["p3"] = 1                               # both directions at one end: identified
    assert se._b_u_identified(two, line)


def test_settling_time_prior_and_a_measured_transient():
    from scan_core.settling import leg_deviation, settle_time, trapezoid
    assert settle_time([], 0.1)["t_settle_s"] == pytest.approx(0.1)
    assert settle_time([], None)["t_settle_s"] == 0.0
    rng = np.random.default_rng(5)
    devs = []
    v, a, L = 0.06, 0.1, 0.4
    for k in range(6):
        t = np.arange(0.0, 8.0, 0.01)
        s, ta, tc, T = trapezoid(t - 0.3, L, v, a)
        tr = t - 0.3 - ta
        tr_dev = np.where(tr >= 0, 4e-4 * np.exp(-np.maximum(tr, 0) / 0.08) * np.sin(2 * math.pi * 12 * tr), 0.0)
        s = s + tr_dev + rng.normal(size=t.size) * 2e-5
        P = np.c_[s, np.zeros_like(s), np.zeros_like(s)]
        devs.append(leg_deviation(t, P, v, a))
    st = settle_time(devs, 0.02)
    assert 0.08 < st["t_settle_s"] < 0.5, st                      # the transient lasts ln(400/20)·0.08 ≈ 0.24 s
    assert st["n_legs"] == 6


def test_pose_smoother_follows_the_command_and_keeps_a_physical_deviation():
    from scan_core.pose_smoother import smoothed_positions
    from scan_core.settling import trapezoid
    rng = np.random.default_rng(8)
    v, a, L = 0.06, 0.1, 0.3
    t = np.arange(0.0, 6.5, 0.01)
    s, _, _, _ = trapezoid(t - 0.2, L, v, a)
    phys = 2e-4 * np.sin(2 * math.pi * 0.7 * t)                    # a slow physical deviation (kept)
    noise = rng.normal(size=t.size) * 1e-4                         # sample noise (removed)
    P = np.c_[s + phys + noise, np.zeros_like(s), np.zeros_like(s)]
    tq = np.linspace(1.0, 5.5, 200)
    out = smoothed_positions(t, P, tq, v, a)
    s_true = np.interp(tq, t, s + phys)
    err_lin = np.interp(tq, t, P[:, 0]) - s_true
    err_sm = out["p"][:, 0] - s_true
    assert np.std(err_sm) < 0.6 * np.std(err_lin), (np.std(err_sm), np.std(err_lin))


# ====================================================================================================================
# the joint-state source (js_source, user 2026-10-12): a reference echo leaves the physical arm an unknown lag τ_echo ≥ 0 behind
# the reported poses — the time offset δ's prior is the hypotheses' cut mixture
# ====================================================================================================================
def _js_evidence(legs, ln_bf_meas_vs_ref: float) -> None:
    """Joint-state evidence on the legs (as the worker stores it, js_source.leg_evidence), summing to ln_bf_meas_vs_ref."""
    for lg in legs:
        lg.meta["joint_state_evidence"] = {"n": 100, "ln_bf": {"lattice": ln_bf_meas_vs_ref / len(legs), "rest": 0.0},
                                           "lattice": [], "rest": None, "representation": "float32 deg"}


TWO_SPEEDS = ((-1, 0.03), (1, 0.03), (-1, 0.06), (1, 0.06))
ONE_SPEED = ((-1, 0.03), (1, 0.03), (-1, 0.03), (1, 0.03))


def test_echo_lag_two_speeds_the_data_identify_it_under_the_echo_prior():
    """Passes at two speeds, both directions, stamped τ = 25 ms late (a reference echo: the arm trails the reported pose — a lag the
    timing prior, σ_q 2.9 ms, does not know). Joint states resolved as the echo: δ is the data's (τ within k·σ, σ ≪ the lag), the
    direction offset b_u and the ends are right with their measured-source σ. The same data under the measured source's prior: δ is
    held near 0, b_u takes the lag (wrong), the ends stay honest only by widening ×6. Undecided (P ½): the mixture covers both."""
    out = {}
    for name, lnbf in (("ref", -60.0), ("meas", 60.0), ("undecided", 0.0)):
        legs, prior, truth = direction_scenario(np.random.default_rng(2100), dt_true_s=0.025, legs_spec=TWO_SPEEDS)
        _js_evidence(legs, lnbf)
        est = se.estimate(legs, prior)
        out[name] = (est, truth, prior)
    est, truth, prior = out["ref"]
    to = est.plate["time_offset"]
    js = to["joint_state_source"]
    assert js["hypothesis"] == "reference" and js["ln_odds_reference"] == pytest.approx(60.0) and js["warning"].startswith("WARNING"), js
    assert abs(est.dt_s - truth["dt"]) <= prior.k * est.dt_sigma_s and est.dt_sigma_s < 0.002, (est.dt_s, est.dt_sigma_s)
    do = est.plate["direction_offset"]
    assert abs(do["b_m"][2] - truth["b_m"][2]) <= prior.k * do["sigma_m"][2], do
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        e = est.ends[X]
        assert abs(e.s_m - e_true) <= prior.k * e.sigma_m and e.sigma_m < 1e-4, (X, e.s_m - e_true, e.sigma_m)
    em, _, _ = out["meas"]
    assert em.plate["time_offset"]["joint_state_source"]["hypothesis"] == "measured"
    assert abs(em.dt_s - truth["dt"]) > prior.k * em.dt_sigma_s                         # the measured prior holds δ near 0 …
    assert abs(em.plate["direction_offset"]["b_m"][2] - truth["b_m"][2]) > prior.k * do["sigma_m"][2]   # … b_u takes the lag
    assert all(em.ends[X].sigma_m > 3.0 * est.ends[X].sigma_m for X in ("S", "E"))       # and the ends widen
    eu, _, _ = out["undecided"]
    assert eu.plate["time_offset"]["joint_state_source"]["hypothesis"] == "undecided"
    assert abs(eu.dt_s - truth["dt"]) <= prior.k * eu.dt_sigma_s and abs(eu.dt_s) <= prior.k * eu.dt_sigma_s   # covers both
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        assert abs(eu.ends[X].s_m - e_true) <= prior.k * eu.ends[X].sigma_m


def test_echo_lag_one_speed_the_ends_are_unbiased_the_split_is_not_claimed():
    """Passes of one speed, both directions, τ = 25 ms: c_v·δ and d·b_u are collinear, the lag cannot be told from the direction
    offset — under the echo (a flat lag) neither is claimed (σ ∞), the along-track direction term A = v·δ − b_u and the ends are
    the data's, unbiased, with the same σ as without the lag."""
    legs, prior, truth = direction_scenario(np.random.default_rng(2200), dt_true_s=0.025, legs_spec=ONE_SPEED)
    _js_evidence(legs, -60.0)
    est = se.estimate(legs, prior)
    legs0, _, truth0 = direction_scenario(np.random.default_rng(2200), dt_true_s=0.0, legs_spec=ONE_SPEED)
    est0 = se.estimate(legs0, prior)
    to = est.plate["time_offset"]
    assert to["joint_state_source"]["hypothesis"] == "reference"
    assert not math.isfinite(est.dt_sigma_s) and not to["separable_from_b_u"], to
    assert not math.isfinite(est.plate["direction_offset"]["sigma_m"][2])
    atd = to["along_track_direction"]
    A_true = atd["speed_mps"] * truth["dt"] - truth["b_m"][2]
    assert abs(atd["value_m"] - A_true) <= prior.k * atd["sigma_m"], (atd, A_true)
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        e = est.ends[X]
        assert abs(e.s_m - e_true) <= max(prior.k * e.sigma_m, 1.5e-4), (X, e.s_m - e_true, e.sigma_m)
        assert e.sigma_m == pytest.approx(est0.ends[X].sigma_m, rel=0.2)


def _one_pass_process(seed: int, end_speed_ratio: float, ln_bf_meas_vs_ref: float):
    """The 1G process after its first pass: dwells at the two sites and the transit between them (inside the seam: the transit
    never crosses an end, its end readings are one-sided, at rest), one pass E→S at 30 mm/s. end_speed_ratio < 1: the pass is still
    accelerating where it crosses E (its reported velocity ramps from that ratio of the cruise speed at its start to the cruise at
    S — test_m on the fake MoveIt arm, 2026-10-13: 28.86 vs 30.01 mm/s). No lag (the profiles sit where the reported poses put them)."""
    legs, prior, truth = process_frame_scenario(np.random.default_rng(seed), n_pass=1)
    ps = [lg for lg in legs if lg.cls == "pass"][0]
    t = np.asarray(ps.t_acq_s, float)
    ps.v_opt_mps = ps.v_opt_mps * (end_speed_ratio + (1.0 - end_speed_ratio) * (t - t[0]) / (t[-1] - t[0]))[:, None]
    _js_evidence(legs, ln_bf_meas_vs_ref)
    return legs, prior, truth


def test_echo_lag_after_one_pass_the_sites_at_rest_bound_it(monkeypatch):
    """After one pass (one direction, one speed at both ends) a flat echo lag and the ends the pass read are confounded. With the
    two end crossings at slightly different speeds the design length's Cauchy prior identifies δ in the planning fusion, but only
    weakly (σ seconds) — and a flat τ ≥ 0 then put its mean seconds out (test_m: δ 3.6 ± 3.1 s, plan ends ±90–133 mm, the next
    pass out of reach). The one-sided readings at rest bound it: an end cannot lie beyond a site where the groove was seen, so
    δ ≤ (site − end read)/v (the sites here 30 mm inside the ends: 1 s at 30 mm/s). The plan ends are then the data's — finite, the
    truth within k·σ, σ about the sites' distance over √12. Where no Gaussian reading identifies δ (both crossings at one speed) the
    bounds alone do not put the flat lag into the model: δ free as before, bit for bit. The measured source (P(echo) 0 in double
    precision, as the lab's runs): the factors are 1 on its prior's support — bit for bit the estimate without them."""
    from scan_core import js_source as J
    site0 = J.DeltaPrior.site
    legs, prior, truth = _one_pass_process(2300, 0.96, -60.0)
    est = se.estimate(legs, prior)
    assert est.plate["time_offset"]["joint_state_source"]["hypothesis"] == "reference"
    d_site, v = 0.030, 0.03
    pe = se.plan_ends(est)
    for X, e_true in (("S", truth["eS_m"]), ("E", truth["eE_m"])):
        assert pe[X]["sigma_m"] < 0.5 * d_site, (X, pe[X])
        assert abs(pe[X]["s_m"] - e_true) <= prior.k * pe[X]["sigma_m"], (X, pe[X]["s_m"] - e_true, pe[X]["sigma_m"])
    # the same data without the one-sided readings' factors: the weak δ runs out on the half-line
    with monkeypatch.context() as mp:
        mp.setattr(J.DeltaPrior, "site", lambda self, m_L, V_L, factors=(): site0(self, m_L, V_L))
        legs0, _, _ = _one_pass_process(2300, 0.96, -60.0)
        p0 = se.plan_ends(se.estimate(legs0, prior))
    assert all(p0[X]["sigma_m"] > 3.0 * pe[X]["sigma_m"] for X in ("S", "E")), (p0, pe)
    # one speed at both ends: δ not identified — the bounds change nothing (bit for bit)
    for lnbf in (-60.0, 1e4):                            # echo; measured with P(echo) 0 in double precision
        legs1, _, _ = _one_pass_process(2300, 1.0 if lnbf < 0 else 0.96, lnbf)
        e1 = se.estimate(legs1, prior)
        if lnbf > 0:
            assert e1.plate["time_offset"]["joint_state_source"]["p_reference_echo"] == 0.0
        with monkeypatch.context() as mp:
            mp.setattr(J.DeltaPrior, "site", lambda self, m_L, V_L, factors=(): site0(self, m_L, V_L))
            legs2, _, _ = _one_pass_process(2300, 1.0 if lnbf < 0 else 0.96, lnbf)
            e2 = se.estimate(legs2, prior)
        p1, p2 = se.plan_ends(e1), se.plan_ends(e2)
        for X in ("S", "E"):
            assert (e1.ends[X].s_m, e1.ends[X].sigma_m) == (e2.ends[X].s_m, e2.ends[X].sigma_m), (lnbf, X)
            assert (p1[X]["s_m"], p1[X]["sigma_m"]) == (p2[X]["s_m"], p2[X]["sigma_m"]), (lnbf, X)
        assert e1.dt_s == e2.dt_s and e1.dt_sigma_s == e2.dt_sigma_s


def test_without_joint_states_the_time_prior_is_todays():
    """Legs without joint-state samples (poses given per profile) carry no echo to consider: P(echo) 0, δ's prior N(0, σ_q²)."""
    legs, prior, truth = direction_scenario(np.random.default_rng(2100), dt_true_s=0.004, legs_spec=TWO_SPEEDS)
    tp = se.timing_prior(legs, truth["f0"]["u0"])
    assert tp["joint_state"] is None and tp["p_ref"] == 0.0
    line = se.fit_line(legs, prior)
    assert line["delta_prior"].p_ref == 0.0 and line["delta_prior"].prior_precision() == pytest.approx(1.0 / tp["sigma_s"] ** 2)
