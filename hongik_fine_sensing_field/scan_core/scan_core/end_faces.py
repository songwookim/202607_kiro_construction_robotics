"""Seam ends from the measured 3-D points at the plate top edges through the sensor's edge response (1G end model r5, 2026-10-11).

Why. The groove change point (§4.3, λ^line) is evidence of the WHOLE profile; near a plate end the template is misspecified (the plane
sweeps the section, faces are lit or shadowed, ghosts and tack welds appear) and λ moves with the pitch (lab 2026-10-11 C3: −5 → 0°
moved the λ ends +0.6 / −2.1 mm). λ only says where the seam is. The plate ends are read from the measured points instead.

Given (design) information: square plate ends (declared joint; the S face checked with a square), so per plate p ∈ {L, R} (WP1
template plates) and end X ∈ {S, E} one end position e_Xp at every height; the seam ends where both plates are present: e_S = max_p
e_Sp, e_E = min_p e_Ep (Clark 1961 moments of the max/min of two correlated normals). No geometric edge parameter (chamfer, rounding):
the sloped points at a plate's top edge — rising at one end, drooping at the other, changing with the pitch (lab C1) — are the
SENSOR's response to a depth step. Design lengths never enter the reading (the λ region is fused without the length prior).

Sensor model (Curless & Levoy 1995, "Better optical triangulation through spacetime analysis"; Hebert & Krotkov 1992 mixed pixels):
the laser sheet has a Gaussian thickness σ_L along the scan (sheet ⊗ camera blur). With the sheet centre a distance a·σ_L inside the
material, only the fraction Φ(a) falls on the top; its centroid sits σ_L·λ(a) further in (λ = φ/Φ) and the triangulation reads that
shift as a height change G·(shift along û), G = (û·m)(c·n̂)/sin θ with c = −cos θ·z − sin θ·m the ray to the camera (sensor −y
side) and θ the triangulation angle — a rise at the end whose material lies toward the camera's view direction, a droop at the
other; the partial spot's centroid noise is sd0/√Φ(a) (+ the row's pose noise σ_row, shared by its columns) and it is reported
with probability p_j·Φ(a) (it fades with its lit fraction). With the centre beyond the edge the camera may see the lit corner
itself (a glint, on the asymptote ±G·distance; probability π_c·(1 − Φ)) — at a LIT end (the face turned toward the camera) the
whole END FACE (plate + backing, design depth D): a lit spot at depth q reads on the corner line shifted by q·(1 + G·c_n), q uniform
over [0, D] (lab 2026-10-11: the S end of the right plate read 2.7 / 7 mm beyond its true end — glint streaks from its plate and
backing faces crossing the top height, explained by a corner slid out; g1-3pass sr/findings.txt). Beyond the end the sheet lights
the SUPPORT the specimen stands on (a plane D below the top, its height profiled; probability π_s where the sheet at that depth is
beyond the end — seen at its foot only past a lit end; past a shadowed end the plate hides it). Otherwise a column returns NOTHING (ν; beyond a shadowed end the camera sees nothing — evidence too) or OTHER
(uniform over the sensor window): ghost reflections, tack welds, lower structures, debris — never plate, it cannot move the end. A whole row
may be MASKED (1 − ρ; a ghost or glare outshines the spot in every column). The top is one surface across the strip: in a lit row
each column returns it at its own rate p_j (measured on the far rows), so a ghost that crosses the top height in one column of a row
is not plate. σ_L and θ are properties of the sensor: calibration constants (calib g1: edge_response — pooled from square plate
ends, scan_l0.calib_edge_response) with their σ; without them (another sensor) they are estimated online from all readings
(σ_L log-normal prior, θ prior N(26°, 2°), lab C5).

Observation per leg ℓ, end X, plate p — legs whose groove change point brackets X: the strip columns of the plate's top (those that
return the top on most far rows) on every row near the end — a return or none — in the seam frame (s = orthogonal foot on the fused
root line, (w, h) in the section basis, the leg's known (w, n) offsets), each with its row's sheet centre s0 at the top height. Region
of e: the λ end (no length prior) ± (k·σ + |c_n|·D + 4·σ_L,max); the top plane near the end (offset, slope along s) is profiled out
per end position, its across-seam slope, the point noise sd0 and the row noise σ_row come from the far surface. Reading = joint
mode of (e, class weights) over the profile's best separated local maxima, then the posterior mean / variance of e (uniform prior
over the region) and its derivatives in log σ_L and θ.

Fusion (fuse_end_faces): u1 = e_Xp + common terms (δ, calibration Δlog σ_L / Δθ / w_x, class and leg offsets, each with its known
σ) + ζ_(leg, X) + ε; the variance components σ_ζ (leg-end offset, shared by the two plates of a crossing) and τ (per reading) are
integrated over their posterior (small-sample: Student-t-like ends); their prior is the robot + sensor population calibrated from
saved runs (calib g1: noise_priors — scaled-inverse-χ², speed model s0²(v) = a + b·v²; noise_priors module), so the posterior degrees
of freedom are ν0 + the run's redundant crossings and two passes give finite, honest σ; uncalibrated, the scale-invariant prior and
'not established below three redundant crossings'. A gross-error class per reading (MLESAC) takes misreadings.

Numeric settings marked [R] are iteration caps / steps only.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

EM_ITERS = 20              # [R] mixture EM / IRLS steps
ROBUST_ROUNDS = 6          # [R] robust rounds of the fusion
W_FLOOR = 1e-6             # [R] smallest reading weight (keeps the normal equations regular)
SIG_GRID = 30              # [R] log-grid points per variance component (σ_ζ, τ)
SIG_SPAN = (-6.0, 9.0)     # [R] that grid's log range about the readings' own σ (×0.0025 … ×8100)
PRIOR_TAIL = 1e-9          # [R] a calibrated prior's tail probability the coarse grid reaches (then refined to the posterior)
HUBER_C = 1.345            # statistical constant (95 % Gaussian efficiency)
REL_NULL = 1e-12           # [R] relative eigenvalue below which a normal-equation direction is unidentified
# sensor (VC nano 3D-Z) — properties of the instrument, not of the specimen:
THETA_TRI_DEG = 26.0       # angle between the laser plane and the camera ray; lab pitch sweep 2026-10-11 (C5): 26 ± 2° (specular lobe
THETA_TRI_SIGMA_DEG = 2.0  # at job pitch +θ/2 ≈ +13°, camera on the sensor −y side); prior N(26°, 2°), refined by the data
SIGMA_L_PRIOR_M = (3e-5, 1.5e-3)  # search support of the edge spread σ_L (laser-sheet thickness ⊗ camera blur along the scan)
SIGMA_L_MEDIAN_M = 4.5e-4  # its prior: log-normal, median 0.45 mm, log-σ 0.5 (×1.65) — the sensor's spread as the lab scans of
SIGMA_L_LOG_SD = 0.5       # 2026-10-10/11 measured it (0.31–0.65 mm over 7 runs, −15…+5° pitch); estimated online on every scan
N_SIGMA_L = 6              # [R] coarse log grid of σ_L over its prior support (start of the local search)
E_STEP_M = 1e-4            # [R] end-position grid step, refined around the mode
E_FINE = 41                # [R] refined grid size (± two steps)
STRIP_MARGIN_M = 2e-3      # [R] lateral margin of the first-order point preselection beyond the plates' strips
COLS_PER_ROW = 4           # [R] columns per row and plate used (a row's columns share its sheet position and pose)
E_LOCAL_M = 1.5e-3         # [R] half window of the end position about a mode (local re-maximisation, derivatives)
LOGIT_MAX = 12.0           # [R] bound of the class-weight logits (weights within 6e-6 … 1 − 6e-6)
N_CANDIDATES = 3           # [R] local maxima of a reading's profile tried as joint (e, π) modes
DERIV_STEP_LOG = 0.05      # [R] central-difference steps of the readings' derivatives in log σ_L and θ
DERIV_STEP_THETA_DEG = 0.25
LOG_PHI_MIN = math.log(1e-12)  # [R] floor of log Φ (a partial spot below 1e-12 of the sheet is no spot)
PLANE_STEPS = 2            # [R] weighted least-squares steps of the local top plane per end position
PLATES = ("L", "R")
PI_START = np.array([0.9, 0.2, 0.5, 0.2])   # [R] start of the class weights (ρ visible, π_c end face, ν none, π_s support)
EDGE_REACH_M = 4.0 * SIGMA_L_PRIOR_M[1]   # the edge response's reach at the widest spread of the prior: a reading uses the sheet
                                          # positions this far beyond its end region (leg_edge_data; the pass overscan plans for it)


# ====================================================================================================================
# data
# ====================================================================================================================
def load_measured(leg) -> Optional[dict]:
    """The leg's measured profiles (recon.npz next to profiles.npz): x_mm, z_mm, valid, T_base_opt, t_acq_s — None when the leg has
    no record of them (synthetic legs without points)."""
    cached = getattr(leg, "_measured_cache", None)
    if cached is not None:
        return cached
    path = getattr(leg, "recon_path", None)
    if not path:
        return None
    keys = ("x_mm", "z_mm", "valid", "T_base_opt", "t_acq_s")
    with np.load(path, allow_pickle=False) as z:
        if not all(k in z.files for k in keys):
            return None
        out = {k: np.asarray(z[k]) for k in keys}
    try:
        leg._measured_cache = out
    except AttributeError:
        pass
    return out


def _basis(L, s: np.ndarray, f0: dict) -> tuple:
    """Section basis along the fused line at s: t = dr/ds, û, n̂ = n0 ⟂ û, ŵ = n̂ × û (each (…, 3))."""
    t = L.t(s)
    u = t / np.linalg.norm(t, axis=-1, keepdims=True)
    n = f0["n0"] - (u @ f0["n0"])[..., None] * u
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    return t, u, n, np.cross(n, u)


def seam_coords(P: np.ndarray, L, f0: dict, iters: int = 2) -> tuple:
    """Point(s) → (s, w, h): s = the line parameter of the point's orthogonal foot on the fused root line (Gauss–Newton from the u0
    projection, [R] iterations), (w, h) = its offset in the section basis there — an end face perpendicular to the seam has one s
    over the section whatever the taught direction."""
    s = (P - f0["S0"]) @ f0["u0"]
    for _ in range(iters):
        t = L.t(s)
        s = s + np.einsum("...i,...i->...", P - L.r(s), t) / np.einsum("...i,...i->...", t, t)
    _, _, n, w = _basis(L, s, f0)
    d = P - L.r(s)
    return s, np.einsum("...i,...i->...", d, w), np.einsum("...i,...i->...", d, n)


def _wmedian(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & (w > 0)
    if not np.any(ok):
        return float("nan")
    o = np.argsort(x[ok])
    xs, cw = x[ok][o], np.cumsum(w[ok][o])
    return float(xs[np.searchsorted(cw, 0.5 * cw[-1])])


# ====================================================================================================================
# fused section near the ends (per plate polyline in the plate's outward lateral coordinate ω and height h)
# ====================================================================================================================
def section_model(geo: dict, s_star: np.ndarray, line: dict, weights: np.ndarray) -> Optional[dict]:
    """Per plate: polyline root edge → (root-face top) → top corner → top strip (half the measured top opening), its side in w0,
    band [ω_lo, ω_hi], top / bottom height and cross-section area — weighted medians of the template vertices of every
    profile (weights = groove-at-the-line probability), in metres relative to the fused root line."""
    from .seam_estimate import _line_from
    L = _line_from(line)
    f0 = line["frame0"]
    names = list(geo["names_v"])
    V = np.asarray(geo["V"], dtype=float)                        # (N, k, 3)
    if V.ndim != 3 or not V.shape[0]:
        return None
    _, W, H = seam_coords(V, L, f0)
    wt = np.where(np.isfinite(weights), weights, 0.0)
    med = {nm: np.array([_wmedian(W[:, i], wt), _wmedian(H[:, i], wt)]) for i, nm in enumerate(names)}
    if any(not np.all(np.isfinite(v)) for v in med.values()):
        return None
    U = np.asarray(geo["U"], dtype=float)
    fn = list(geo["names_f"])
    O = abs(float(med["top_corner_L"][0] - med["top_corner_R"][0]))
    plates = {}
    for p in PLATES:
        chain = [nm for nm in (f"root_edge_{p}", f"root_face_top_{p}", f"top_corner_{p}") if nm in med]
        side = 1.0 if med[f"top_corner_{p}"][0] >= 0.0 else -1.0
        pts = np.array([[side * med[nm][0], med[nm][1]] for nm in chain])
        f_top = fn.index(f"top_{p}") if f"top_{p}" in fn else None
        if f_top is not None:
            _, _, nb, wb = _basis(L, s_star, f0)
            dw = np.array([_wmedian(np.einsum("ij,ij->i", U[:, f_top], wb), wt), _wmedian(np.einsum("ij,ij->i", U[:, f_top], nb), wt)])
            dw = np.array([side * dw[0], dw[1]])
            dw = dw / np.linalg.norm(dw) if np.all(np.isfinite(dw)) and np.linalg.norm(dw) > 0 else np.array([1.0, 0.0])
            dw = dw if dw[0] >= 0 else -dw
        else:
            dw = np.array([1.0, 0.0])
        top_end = pts[-1] + (0.5 * O / max(dw[0], 1e-9)) * dw
        poly = np.vstack([pts, top_end])
        lo, hi = float(poly[0, 0]), float(top_end[0])
        # surface height as a function of ω (a vertical wall keeps its top)
        o = np.argsort(poly[:, 0], kind="stable")
        xs, ys = poly[o, 0], poly[o, 1]
        ux, inv = np.unique(xs, return_inverse=True)
        uy = np.full(ux.shape, -np.inf)
        np.maximum.at(uy, inv, ys)
        grid = np.linspace(lo, hi, 257)
        area = float(np.trapz(np.maximum(np.interp(grid, ux, uy), 0.0), grid))
        plates[p] = {"side": side, "poly": poly, "omega_lo": lo, "omega_hi": hi, "top_h": float(pts[-1][1]),
                     "bottom_h": float(np.min(pts[:, 1])),
                     "surf_x": ux, "surf_y": uy, "area": max(area, 1e-12)}
    return {"plates": plates, "top_opening_m": O}


# ====================================================================================================================
# helpers
# ====================================================================================================================
def _clark_max(m1, v1, m2, v2, c12) -> tuple:
    """Mean and variance of max(A, B), (A, B) jointly normal (Clark 1961). Infinite inputs → infinite variance."""
    if not (math.isfinite(v1) and math.isfinite(v2)):
        if math.isfinite(v1) or math.isfinite(v2):
            mm = m1 if not math.isfinite(v1) else m2
            return float(mm), math.inf
        return float(max(m1, m2)), math.inf
    th2 = v1 + v2 - 2.0 * c12
    if not th2 > 1e-30:
        return float(max(m1, m2)), float(v1 if m1 >= m2 else v2)
    th = math.sqrt(th2)
    a = (m1 - m2) / th
    P, Q, ph = norm.cdf(a), norm.cdf(-a), norm.pdf(a)
    e1 = m1 * P + m2 * Q + th * ph
    e2 = (m1 * m1 + v1) * P + (m2 * m2 + v2) * Q + (m1 + m2) * th * ph
    return float(e1), float(max(e2 - e1 * e1, 0.0))


def _huber_plane(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple:
    """Huber M-estimate (c = HUBER_C, MAD scale) of y ≈ X·b with prior weights w → (b, robust residual scale)."""
    b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    sc = 0.0
    for _ in range(EM_ITERS):
        r = y - X @ b
        sc = 1.4826 * _wmedian(np.abs(r - _wmedian(r, w)), w)
        if not sc > 0:
            break
        hw = w * np.minimum(1.0, HUBER_C * sc / np.maximum(np.abs(r), 1e-300))
        b_new = np.linalg.lstsq(X * np.sqrt(hw)[:, None], y * np.sqrt(hw), rcond=None)[0]
        if np.allclose(b_new, b, rtol=0.0, atol=1e-9):
            b = b_new
            break
        b = b_new
    return b, float(sc)


# ====================================================================================================================
# one leg, one end, one plate: the top edge through the sensor's edge response
# ====================================================================================================================
def leg_edge_data(leg, meas: dict, line: dict, sec: dict, coarse: dict, prior, split_s: float, sigma_d: float,
                  jitter: float, z_window_m: float, crosses: dict) -> list:
    """Per end this leg crosses (crosses[X]: its groove change point at X has profiles on both sides — λ says where the seam is;
    the leg's halves split at its groove-segment middle) and per plate: the top-strip points near the end as (sheet centre s0 at the
    top height, height deviation dh from the leg's own top surface, height h), with the leg's sensor geometry, noise and kinematics.
    The reading itself needs the sensor's edge spread, which every leg and end share (edge_readings)."""
    from .seam_estimate import _canon_cls, _line_from, leg_offset_wn
    L = _line_from(line)
    f0 = line["frame0"]
    T = np.asarray(meas["T_base_opt"], dtype=float)
    if T.ndim != 3 or not T.shape[0]:
        return []
    o, ex, m, ez = T[:, :3, 3], T[:, :3, 0], T[:, :3, 1], T[:, :3, 2]
    cls = _canon_cls(leg.cls, getattr(leg, "meta", None))
    ow, on = leg_offset_wn(line, leg.name, cls)          # class + motion direction × b + the leg's own η (w0, n0)
    o3 = ow * f0["w0"] + on * f0["n0"]
    # the sheet's crossing of the line from the corrected pose (measured + o = true), like the points below: with the laser plane
    # tilted (pitch), a normal offset moves the sheet along the seam by c_n·o_n (≈ 0.015 mm for b_n 0.17 mm at −5°, direction-signed)
    s_rows_all = L.s_star(o + o3, m)
    t_acq = np.asarray(meas["t_acq_s"], dtype=float)
    ou = o @ f0["u0"]
    v_u = np.gradient(ou, t_acq) if (t_acq.size > 1 and np.all(np.diff(t_acq) > 0)) else np.zeros(ou.shape)
    tb, ub, nb, wb = _basis(L, s_rows_all, f0)
    mt = np.einsum("ij,ij->i", m, tb)
    c_w_rows, c_n_rows = np.einsum("ij,ij->i", m, wb) / mt, np.einsum("ij,ij->i", m, nb) / mt   # s* − s of a point at (w, h)
    gain_rows = np.einsum("ij,ij->i", m, ub) / np.einsum("ij,ij->i", ez, nb)   # ∂s/∂h of a point under a laser-pitch error, /rad
    # triangulation geometry: a spot centroid moved by δ along û reads δ·(û·m)(c·n̂)/sin θ higher, c = −cos θ·z − sin θ·m the ray to
    # the camera (sensor −y side); stored as the three dot products, θ enters in edge_readings
    g_um, g_zn, g_mn = np.einsum("ij,ij->i", m, ub), np.einsum("ij,ij->i", ez, nb), np.einsum("ij,ij->i", m, nb)
    xm, zm, va = meas["x_mm"], meas["z_mm"], np.asarray(meas["valid"], dtype=bool)
    k = float(prior.k)
    # the square end face below the top edge (design): the plate and the backing it stands on — the depth of the lit face the camera
    # sees and of the support surface beyond the end (END FACE / SUPPORT classes)
    sp_ = getattr(prior, "spec", None)
    face_depth = 0.0 if sp_ is None else 1e-3 * (float(sp_.thickness_mm) + (float(sp_.backing_thickness_mm) if sp_.backing else 0.0))
    # the surfaces below the top that the design puts beyond an end: the backing top (plate thickness; where the backing reaches beyond
    # the plates) and the support under the specimen (plate + backing) — the SUPPORT class's starting depths
    support_depths = sorted({1e-3 * float(sp_.thickness_mm), face_depth}) if sp_ is not None else [face_depth]
    reach = EDGE_REACH_M                                 # the edge response's reach at the widest spread of the prior
    out = []
    for X in ("S", "E"):
        if not crosses.get(X):
            continue
        rows = np.flatnonzero(s_rows_all < split_s) if X == "S" else np.flatnonzero(s_rows_all >= split_s)
        c = coarse.get(X) or {}
        e_c, sg_c = float(c.get("s_m", float("nan"))), float(c.get("sigma_m", math.inf))
        if rows.size < 2 or not (math.isfinite(e_c) and math.isfinite(sg_c)):
            continue
        # region of the end: "where is the seam" (λ fusion) ± k·σ, widened by what λ cannot see at an end — the plane's sweep over the
        # section (|c_n|·D) and the edge response's reach (the whole-profile template is misspecified there, lab 2026-10-11 C3)
        cn_x = float(np.median(np.abs(c_n_rows[rows])))
        half = k * sg_c + cn_x * max(float(sec["plates"][p_]["top_h"]) for p_ in PLATES) + reach
        lo, hi = e_c - half, e_c + half
        lit = bool(np.median(c_n_rows[rows]) > 0.0) if X == "S" else bool(np.median(c_n_rows[rows]) < 0.0)
        # only what the reading uses: the rows of the window and of the far surface beside it (plus the plane's sweep), and in them the
        # points near the plates' top strips (first-order lateral offset in the basis at the row's root crossing; the columns' rays
        # are nearly along the section normal there)
        ext = reach + 1.5 * (hi - lo)
        rows = rows[(s_rows_all[rows] >= lo - ext) & (s_rows_all[rows] <= hi + ext)]
        if rows.size < 2:
            continue
        rr, cc = np.nonzero(va[rows])
        ri = rows[rr]
        P = (o[ri] + 1e-3 * (xm[ri, cc][:, None] * ex[ri] + zm[ri, cc][:, None] * ez[ri])) + o3
        r_row = L.r(s_rows_all[rows])
        w1 = np.einsum("ij,ij->i", P - r_row[rr], wb[ri])
        w_out = max(float(sec["plates"][p_]["omega_hi"]) for p_ in PLATES) + STRIP_MARGIN_M
        keep = np.abs(w1) <= w_out
        ri, cc, P = ri[keep], cc[keep], P[keep]
        _, w_p, h_p = seam_coords(P, L, f0)
        n_col = int(va.shape[1])
        cc_all = cc
        idx2d = np.full(va.shape, -1, dtype=np.int64)
        idx2d[ri, cc] = np.arange(ri.size)
        for p in PLATES:
            pl = sec["plates"][p]
            om = pl["side"] * w_p
            corner = pl["poly"][-2]
            strip = (om >= corner[0]) & (om <= pl["omega_hi"])
            if not np.count_nonzero(strip):
                continue
            rs, oms, hs, ws = ri[strip], om[strip], h_p[strip], w_p[strip]
            s0 = s_rows_all[rs] - (c_w_rows[rs] * ws + c_n_rows[rs] * pl["top_h"])   # sheet centre at the top height
            win = (s0 >= lo - reach) & (s0 <= hi + reach)
            if X == "S":
                far = (s0 > hi + reach) & (s0 <= hi + reach + (hi - lo))
            else:
                far = (s0 < lo - reach) & (s0 >= lo - reach - (hi - lo))
            if np.count_nonzero(far) < 3 or np.count_nonzero(win) < 3:
                continue
            # the leg's own top surface far from the end (height over ω and s0), Huber; its residual scale = the point noise
            Xf = np.c_[np.ones(np.count_nonzero(far)), oms[far], s0[far]]
            b, sd0 = _huber_plane(Xf, hs[far], np.ones(np.count_nonzero(far)))
            if not sd0 > 0:
                sd0 = sigma_d
            # one observation per row (the sheet position is the row's; its columns share the pose): the median top-strip height
            # (across-seam slope removed) — robust to a minority of other returns in the row
            yv = hs - b[1] * oms
            ur_f, inv_f = np.unique(rs[far], return_inverse=True)
            med_f = np.array([np.median(yv[far][inv_f == j]) for j in range(ur_f.size)])
            s_f = np.array([np.median(s0[far][inv_f == j]) for j in range(ur_f.size)])
            n_f = np.bincount(inv_f).astype(float)
            # row-to-row height noise beyond the points' own (pose jitter, settling): from the far rows about the far plane
            rr_f = med_f - (b[0] + b[2] * s_f)
            s_med = 1.4826 * float(np.median(np.abs(rr_f - np.median(rr_f))))
            sig_row = math.sqrt(max(s_med ** 2 - (math.pi / 2.0) * sd0 ** 2 / float(np.median(n_f)), 0.0))
            # the plate's top-strip columns: those that return the strip on most profiles far from the end; on every profile near
            # the end each of them is one observation — its return (wherever it lies) or NO return (beyond a shadowed end the camera
            # sees nothing: that absence is evidence too)
            far_rows = np.unique(rs[far])
            cols_far = cc_all[strip][far]
            frac = np.bincount(cols_far, minlength=n_col) / float(far_rows.size)
            J = np.flatnonzero(frac >= 0.5)
            # each column's own rate of returning the top where the sheet lies fully on it (rule of succession over the far rows):
            # the top is one surface across the strip — a lit row shows it in every column at these rates, a ghost does not
            p_top = (np.bincount(cols_far, minlength=n_col) + 1.0) / (float(far_rows.size) + 2.0)
            if J.size < 2:
                continue
            om_col = np.full(n_col, np.nan)
            for j in J:
                om_col[j] = float(np.median(oms[far][cols_far == j]))
            Jsub = J[np.unique(np.linspace(0, J.size - 1, min(COLS_PER_ROW, J.size)).round().astype(int))]
            om_mid = float(np.median(om_col[J]))
            s0_rows = s_rows_all[rows] - (c_w_rows[rows] * pl["side"] * om_mid + c_n_rows[rows] * pl["top_h"])
            wrows = rows[(s0_rows >= lo - reach) & (s0_rows <= hi + reach)]
            if wrows.size < 3:
                continue
            kk = idx2d[np.repeat(wrows, Jsub.size), np.tile(Jsub, wrows.size)]
            rr_o = np.repeat(np.arange(wrows.size), Jsub.size)
            vo = kk >= 0
            p_o = np.tile(p_top[Jsub], wrows.size)
            om_o = np.where(vo, pl["side"] * w_p[np.maximum(kk, 0)], np.tile(om_col[Jsub], wrows.size))
            h_o = np.where(vo, h_p[np.maximum(kk, 0)], 0.0)
            ro_ = wrows[rr_o]
            s0_o = s_rows_all[ro_] - (c_w_rows[ro_] * pl["side"] * om_o + c_n_rows[ro_] * pl["top_h"])
            y_o = np.where(vo, h_o - b[1] * om_o, 0.0)
            s_r = s0_rows[(s0_rows >= lo - reach) & (s0_rows <= hi + reach)]
            s_c = 0.5 * (lo + hi)
            inv = rr_o
            # one weight per distinct sheet position: frames repeated at one pose (dwell) are one observation of the edge
            ds = np.abs(np.diff(np.sort(s_r)))
            step = float(np.median(ds[ds > 0])) if np.any(ds > 0) else float("inf")
            if math.isfinite(step):
                cnt = np.array([np.count_nonzero(np.abs(s_r - v) <= 0.5 * step) for v in s_r])
            else:
                cnt = np.full(s_r.size, float(s_r.size))
            w_row = 1.0 / cnt.astype(float)
            ur = wrows
            iw = np.flatnonzero(vo)
            rin = rows[(s_rows_all[rows] >= lo) & (s_rows_all[rows] <= hi)]
            mv_ = rin[np.abs(v_u[rin]) > 0] if rin.size else rin
            c_v = float(np.median(v_u[mv_])) if mv_.size else 0.0
            near = rin if rin.size else rows
            out.append({"end": X, "plate": p, "leg": leg.name, "cls": cls, "direction": int(leg.direction),
                        "motion_sign": int((line.get("leg_direction") or {}).get(leg.name, 0)), "lit": lit,
                        "s0": s0_o, "y": y_o, "valid": vo, "p_top": p_o, "ds": s0_o - s_c, "row": inv, "w_row": w_row,
                        "n_in_row": np.bincount(inv, minlength=ur.size).astype(float),
                        "n_valid_row": np.bincount(inv, weights=vo.astype(float), minlength=ur.size),
                        "row_start": np.searchsorted(inv, np.arange(ur.size)),
                        "b_far": np.array([b[0] + b[2] * s_c, b[2]]), "sd0": float(sd0), "sig_row": float(sig_row), "top_h": float(pl["top_h"]),
                        "bottom_h": float(pl["bottom_h"]), "c_n": float(np.median(c_n_rows[near])),
                        "g_um": float(np.median(g_um[near])), "g_zn": float(np.median(g_zn[near])),
                        "g_mn": float(np.median(g_mn[near])), "region": (float(lo), float(hi)), "c_v": c_v,
                        "v_row": v_u[ur], "s0_row": s_r,
                        "pitch_gain": float(np.median(gain_rows[near])), "floor": float(jitter), "z_window": float(z_window_m),
                        "n_rows": int(ur.size), "n_points": int(iw.size), "face_depth": face_depth,
                        "support_depths": support_depths})
    return out


def _gain(g: dict, theta: float) -> float:
    """Height read per unit centroid shift along û (signed) at triangulation angle θ."""
    return g["g_um"] * (-math.cos(theta) * g["g_zn"] - math.sin(theta) * g["g_mn"]) / math.sin(theta)


def _parts(g: dict, E: np.ndarray, sen: tuple) -> tuple:
    """Per end position (rows of E) and point: log Φ(a), the TOP height deviation μ and its variance, μ_c, the support gating and
    the gain. TOP — the top surface through the truncated-sheet response (Curless & Levoy 1995): with the sheet centre a·σ_L inside
    the material only the fraction Φ(a) falls on the top; its centroid sits σ_L·λ(a) further in (λ = φ/Φ), which the triangulation
    reads as a height change ±G·σ_L·λ(a); variance sd0²/Φ(a) (the partial spot's centroid noise) + n·σ_row² (the row's pose noise,
    shared by its n columns: their joint weight is that of one row). μ_c: a light source on the end face's top corner (q = 0) read on
    the sheet centre's plane at G·(e − s0) — the END FACE class spreads it over the face's depth (_components). Also (sgn, E, σ_L)
    for the SUPPORT's gating, which needs the support's profiled depth (_components)."""
    sL, G = sen
    sgn = 1.0 if g["end"] == "S" else -1.0
    a = sgn * (g["s0"][None, :] - E[:, None]) / sL
    lP = log_ndtr(a)
    lPhi = np.maximum(lP, LOG_PHI_MIN)
    mu = sgn * G * sL * np.exp(norm.logpdf(a) - lP)
    vr = g["n_in_row"][g["row"]] * g["sig_row"] ** 2
    var = g["sd0"] ** 2 * np.exp(-lPhi) + vr[None, :]
    mu_c = -sgn * G * sL * a
    # the support's gating needs its depth (profiled with the top plane, _components): the end positions and the scale are passed on
    return lPhi, mu, var, mu_c, (sgn, E, sL), G


def _components(g: dict, parts: tuple, B: np.ndarray, HS: Optional[np.ndarray] = None) -> tuple:
    """Class log-densities of every point (rows of B = end positions): (log Φ, TOP, END FACE, SUPPORT, log support gating) given
    the local top plane B and the support height HS (relative to the top plane, per end position).
    END FACE: at a LIT square end (the face turned toward the camera) the camera sees the whole end face — plate and backing, depth
    D below the top (design: plate thickness + backing thickness). A glint or lit spot at depth q on it is a light source off the
    sheet plane, read on the sheet plane at μ_c − q·k (k = 1 + G·c_n: the sheet meets depth q c_n·q further along); q unknown,
    uniform over [0, D] → the corner line smeared over the band [μ_c − k·D, μ_c] (it also holds the face's direct returns while the
    sheet sweeps down it). At a shadowed end the camera sees no face, only its top corner: q = 0 (μ_c).
    SUPPORT: beyond the end the sheet lights a surface below the top that the end uncovers (the support the specimen stands on, a
    backing ledge), a plane parallel to the top at HS (profiled, start −D) — its returns are not plate and do not move with the end;
    the sheet reaches it only where its point at that depth (c_n·(−HS) along) is beyond the end, and the camera sees it at the
    face's foot only at a LIT end (at a shadowed end the plate stands between them)."""
    lPhi, mu, var, mu_c, (sgn, E, sL), G = parts
    dh = g["y"][None, :] - (B[:, :1] + B[:, 1:2] * g["ds"][None, :])
    lt = -0.5 * (dh - mu) ** 2 / var - 0.5 * np.log(2.0 * math.pi * var)
    vc = g["sd0"] ** 2 + g["n_in_row"][g["row"]] * g["sig_row"] ** 2
    D = float(g.get("face_depth", 0.0) or 0.0)
    if g.get("lit") and D > 0.0:
        sc = np.sqrt(vc)[None, :]
        kD = (1.0 + G * float(g["c_n"])) * D
        if kD > 0.0:
            band = ndtr((mu_c - dh) / sc) - ndtr((mu_c - kD - dh) / sc)
            lc = np.log(np.maximum(band, 1e-300)) - math.log(kD)
        else:
            lc = -0.5 * (dh - mu_c) ** 2 / vc[None, :] - 0.5 * np.log(2.0 * math.pi * vc)[None, :]
    else:
        lc = -0.5 * (dh - mu_c) ** 2 / vc[None, :] - 0.5 * np.log(2.0 * math.pi * vc)[None, :]
    hs = (np.full(B.shape[0], -D) if HS is None else HS)[:, None]
    ls = -0.5 * (dh - hs) ** 2 / vc[None, :] - 0.5 * np.log(2.0 * math.pi * vc)[None, :]
    if g.get("lit") and D > 0.0:
        # the sheet meets the support plane (depth −hs below the top) c_n·(−hs) further along than at the top: it lights the support
        # only where that point is beyond the end
        a_s = sgn * (g["s0"][None, :] + float(g["c_n"]) * np.maximum(-hs, 0.0) - E[:, None]) / sL
        l_bs = log_ndtr(-a_s)
    else:                                   # a shadowed end: the plate stands between the camera and the support at its foot
        l_bs = np.full(dh.shape, -np.inf)
    return lPhi, lt, lc, ls, l_bs


def _point_ll(g: dict, comp: tuple, pi) -> tuple:
    """Column log-likelihoods in a VISIBLE row and the TOP / SUPPORT responsibilities. A column returns the top spot with
    probability p_j·Φ(a) — p_j its own rate on the full top (far rows), fading with the lit fraction of the sheet —, the end face
    with π_c·(1 − Φ(a)) — only the sheet beyond the edge can light it —, the support with π_s·(1 − Φ(a_s)) — only where the sheet
    at the support's depth is beyond the end —; otherwise NOTHING with probability ν (beyond a shadowed end the camera sees nothing)
    or an OTHER return: anything else the sensor reports in its window (ghost reflections, tack welds, debris, lower structures),
    uniform over the window. pi = (ρ, π_c, ν, π_s)."""
    lPhi, lt, lc, ls, l_bs = comp
    v = g["valid"][None, :]
    Phi = np.exp(lPhi)
    pt = g["p_top"][None, :] * Phi
    pc = pi[1] * (1.0 - Phi) * (1.0 - pt)
    ps = (pi[3] if len(pi) > 3 else 0.0) * np.exp(l_bs) * np.maximum(1.0 - pt - pc, 0.0)
    lrest = np.log(np.maximum(1.0 - pt - pc - ps, 1e-300))
    l1 = np.log(np.maximum(pt, 1e-300)) + lt
    l3 = np.log(np.maximum(pc, 1e-300)) + lc
    l4 = np.log(np.maximum(ps, 1e-300)) + ls
    l2 = lrest + math.log(max(1.0 - pi[2], 1e-300)) - math.log(g["z_window"])
    tot_v = np.logaddexp(np.logaddexp(l1, l2), np.logaddexp(l3, l4))
    tot = np.where(v, tot_v, lrest + math.log(max(pi[2], 1e-300)))
    return tot, np.where(v, np.exp(l1 - tot_v), 0.0), np.where(v, np.exp(l4 - tot_v), 0.0)


def _row_ll(g: dict, lvis: np.ndarray, pi) -> tuple:
    """Row-level mixture: a row (one sheet position) is VISIBLE with probability ρ, or MASKED — another light path (a ghost
    reflection, glare off an end face) outshines the spot in every column, so each column returns OTHER or nothing. A masked row on
    the plate then costs log(1 − ρ) once instead of evidence of a missing plate in every column. → (Σ_rows w_row·ll_row per end
    position, posterior visibility per row)."""
    rho, nu = pi[0], pi[2]
    S = np.add.reduceat(lvis, g["row_start"], axis=1)          # observations are stored row by row
    nv = g["n_valid_row"]
    Sm = (nv * (math.log(max(1.0 - nu, 1e-300)) - math.log(g["z_window"])) + (g["n_in_row"] - nv) * math.log(max(nu, 1e-300)))
    a1 = math.log(max(rho, 1e-300)) + S
    a2 = math.log(max(1.0 - rho, 1e-300)) + Sm[None, :]
    tot = np.logaddexp(a1, a2)
    return tot @ g["w_row"], np.exp(a1 - tot)


def _group_ll(g: dict, E: np.ndarray, sen: tuple, pi, want_b: bool = False):
    """Log-likelihood of the reading's points for each end position in E, the local top plane and the support height profiled out
    (weighted least squares on the TOP / weighted mean on the SUPPORT responsibilities of visible rows, PLANE_STEPS steps from the
    far-surface fit). The support's depth starts at each surface the design puts below the top beyond an end (support_depths: the
    backing top — where the backing reaches beyond the plates — and the support under the backing) and the better profile counts
    per end position (a start far from the actual surface never finds it: its responsibilities are nil)."""
    parts = _parts(g, E, sen)
    mu, var = parts[1], parts[2]
    X2 = np.c_[np.ones(g["ds"].size), g["ds"]]
    wr = g["w_row"][g["row"]]
    starts = [float(x) for x in (g.get("support_depths") or [g.get("face_depth", 0.0) or 0.0])] if g.get("lit") else [0.0]
    best = None
    for d0 in starts:
        B = np.tile(g["b_far"], (E.size, 1))
        HS = np.full(E.size, -d0)
        for _ in range(PLANE_STEPS):
            lv, rt, rs = _point_ll(g, _components(g, parts, B, HS), pi)
            _, vis = _row_ll(g, lv, pi)
            Wt = rt * vis[:, g["row"]] * wr[None, :] / var
            A = np.einsum("en,ni,nj->eij", Wt, X2, X2) + np.eye(2)[None, :, :] * 1e-12 * (np.sum(Wt, axis=1)[:, None, None] + 1.0)
            rhs = np.einsum("en,ni->ei", Wt * (g["y"][None, :] - mu), X2)
            ok = np.sum(Wt, axis=1) > 0
            if np.any(ok):
                B[ok] = np.linalg.solve(A[ok], rhs[ok][:, :, None])[:, :, 0]
            Ws = rs * vis[:, g["row"]] * wr[None, :]
            sw = np.sum(Ws, axis=1)
            dh = g["y"][None, :] - (B[:, :1] + B[:, 1:2] * g["ds"][None, :])
            HS = np.where(sw > 0, np.sum(Ws * dh, axis=1) / np.where(sw > 0, sw, 1.0), HS)
        lv, _, _ = _point_ll(g, _components(g, parts, B, HS), pi)
        ll, _ = _row_ll(g, lv, pi)
        if best is None:
            best = (ll, B, HS)
        else:
            take = ll > best[0]
            best = (np.where(take, ll, best[0]), np.where(take[:, None], B, best[1]), np.where(take, HS, best[2]))
    ll, B, HS = best
    return (ll, B, HS) if want_b else ll


def _grid(g: dict) -> np.ndarray:
    lo, hi = g["region"]
    n = max(int(math.ceil((hi - lo) / E_STEP_M)), 1)
    return np.linspace(lo, hi, n + 1)


def _local(g: dict, e: float) -> np.ndarray:
    lo, hi = g["region"]
    return np.linspace(max(lo, e - E_LOCAL_M), min(hi, e + E_LOCAL_M), E_FINE)


def _argmax_e(g: dict, E: np.ndarray, sen: tuple, pi) -> tuple:
    """Max over the end positions E (then a parabola through the best node and its neighbours): (e, log-likelihood)."""
    ll = _group_ll(g, E, sen, pi)
    j = int(np.argmax(ll))
    if 0 < j < E.size - 1:
        y0, y1, y2 = ll[j - 1], ll[j], ll[j + 1]
        den = y0 - 2.0 * y1 + y2
        if den < 0:
            e_ = E[j] + 0.5 * (y0 - y2) / den * (E[1] - E[0])
            return float(e_), float(_group_ll(g, np.array([e_]), sen, pi)[0])
    return float(E[j]), float(ll[j])


def _fit_pi(g: dict, e: float, sen: tuple, pi0) -> tuple:
    """(ρ, π_c, ν, π_s) of one reading at its end position: maximum likelihood (bounded logits; L-BFGS-B), the top plane and the
    support height held at their profile. → (π, log-likelihood)."""
    from scipy.optimize import minimize
    E1 = np.array([e])
    _, B, HS = _group_ll(g, E1, sen, pi0, want_b=True)
    comp = _components(g, _parts(g, E1, sen), B, HS)

    def f(x):
        q = 1.0 / (1.0 + np.exp(-x))
        lv, _, _ = _point_ll(g, comp, q)
        return -float(_row_ll(g, lv, q)[0][0])
    p0 = np.clip(np.asarray(pi0, float), 1e-4, 1.0 - 1e-4)
    r = minimize(f, np.log(p0 / (1.0 - p0)), method="L-BFGS-B", bounds=[(-LOGIT_MAX, LOGIT_MAX)] * p0.size,
                 options={"ftol": 1e-10, "gtol": 1e-6, "maxiter": 200})
    return 1.0 / (1.0 + np.exp(-r.x)), -float(r.fun)


def _posterior_e(g: dict, sen: tuple, pi, E: Optional[np.ndarray] = None) -> tuple:
    """Posterior of the reading's end position (uniform prior over its region): (mean, variance). Grid E_STEP_M (or the given
    nodes); where the posterior is narrower than the step it is recomputed on a fine grid around the mode."""
    E = _grid(g) if E is None else E
    st = E[1] - E[0] if E.size > 1 else 0.0
    ll = _group_ll(g, E, sen, pi)
    p = np.exp(ll - np.max(ll))
    p /= p.sum()
    mu = float(np.sum(p * E))
    var = float(np.sum(p * (E - mu) ** 2))
    if var < st * st:
        j = int(np.argmax(p))
        Ef = np.linspace(E[max(j - 2, 0)], E[min(j + 2, E.size - 1)], E_FINE)
        lf = _group_ll(g, Ef, sen, pi)
        pf = np.exp(lf - np.max(lf))
        pf /= pf.sum()
        mu = float(np.sum(pf * Ef))
        var = float(np.sum(pf * (Ef - mu) ** 2)) + (Ef[1] - Ef[0]) ** 2 / 12.0
    return mu, var


def _mode(g: dict, sen: tuple, pi0) -> tuple:
    """Joint maximum of the reading's likelihood over (e, π): the local maxima of the profile at the start weights are candidates
    (an extended ghost beyond the end makes a second mode whose class weights differ); at each the class weights are fitted and e is
    re-maximised nearby — the best joint likelihood wins. → (e, π, log-likelihood, all candidates (e, π, log-likelihood))."""
    E = _grid(g)
    ll = _group_ll(g, E, sen, pi0)
    loc = []                                             # best local maxima, each at least E_LOCAL_M from a better one
    for j in np.argsort(-ll, kind="stable"):
        if (j == 0 or ll[j] >= ll[j - 1]) and (j == E.size - 1 or ll[j] >= ll[j + 1]) and \
                all(abs(E[j] - E[i]) > E_LOCAL_M for i in loc):
            loc.append(int(j))
            if len(loc) == N_CANDIDATES:
                break
    cands = []
    for j in loc:
        e = float(E[j])
        pi, _ = _fit_pi(g, e, sen, pi0)
        e, _ = _argmax_e(g, _local(g, e), sen, pi)
        pi, val = _fit_pi(g, e, sen, pi)
        cands.append((e, pi, val))
    best = max(cands, key=lambda c_: c_[2])
    return best[0], best[1], best[2], cands


def _reading(g: dict, sen: tuple, pi0, steps: tuple) -> dict:
    """One plate-end reading at given sensor parameters: joint mode (e, π), posterior of e (π profiled over the candidate modes'
    weights), and the derivatives of
    the posterior mean with respect to log σ_L and θ (central differences of `steps`; they carry the calibration's uncertainty
    into the fusion as common terms)."""
    sL, th = sen
    sen_g = (sL, _gain(g, th))
    e, pi, _, cands = _mode(g, sen_g, pi0)
    # posterior of e with the class weights profiled over the candidates' fitted weights: where another interpretation of the
    # returns (e.g. masked rows vs. a shorter plate) is nearly as likely, the reading keeps both
    E = _grid(g)
    ll = np.max(np.stack([_group_ll(g, E, sen_g, c_[1]) for c_ in cands]), axis=0)
    p_ = np.exp(ll - np.max(ll))
    p_ /= p_.sum()
    mu = float(np.sum(p_ * E))
    var = float(np.sum(p_ * (E - mu) ** 2))
    st = E[1] - E[0] if E.size > 1 else 0.0
    if var < st * st:                                    # narrower than the grid: the fine posterior at the best mode
        mu, var = _posterior_e(g, sen_g, pi, _local(g, e))
    d = []
    for d_ls, d_th in ((steps[0], 0.0), (0.0, steps[1])):
        Eloc = _local(g, mu)
        mp, _ = _posterior_e(g, (sL * math.exp(d_ls), _gain(g, th + d_th)), pi, Eloc)
        mm_, _ = _posterior_e(g, (sL * math.exp(-d_ls), _gain(g, th - d_th)), pi, Eloc)
        d.append((mp - mm_) / (2.0 * (d_ls or d_th)))
    return {"u1_m": mu, "u1_var": var, "pi": pi, "d_log_sigma_L": d[0], "d_theta_rad": d[1]}


def _search_sensor(groups: list, ls_prior: Optional[tuple], th_prior: tuple, pis: list) -> tuple:
    """Joint mode of the edge spread (log σ_L) and the triangulation angle θ shared by `groups`: Σ (profile log-likelihood of each
    reading) + log priors (ls_prior = (mean, sd) of log σ_L or None = flat over SIGMA_L_PRIOR_M; th_prior = (mean, sd) rad);
    coarse σ_L grid, then alternating bounded searches with the class weights refitted between. → (log σ_L, θ, π list, Laplace sd
    of log σ_L, of θ, log-likelihood)."""
    from scipy.optimize import minimize_scalar
    th0, sth = th_prior
    lsl = (math.log(SIGMA_L_PRIOR_M[0]), math.log(SIGMA_L_PRIOR_M[1]))
    e_near = [None] * len(groups)

    def J(ls, th, local=True):
        sL = math.exp(ls)
        res = []
        for k, g in enumerate(groups):
            sen = (sL, _gain(g, th))
            res.append(_argmax_e(g, _local(g, e_near[k]) if (local and e_near[k] is not None) else _grid(g), sen, pis[k]))
        pr = -0.5 * ((th - th0) / sth) ** 2 + (-0.5 * ((ls - ls_prior[0]) / ls_prior[1]) ** 2 if ls_prior else 0.0)
        return sum(r[1] for r in res) + pr, res

    ls_start = ls_prior[0] if ls_prior else 0.5 * (lsl[0] + lsl[1])
    pis = [_mode(g, (math.exp(ls_start), _gain(g, th0)), pis[k])[1] for k, g in enumerate(groups)]
    grid_ls = np.linspace(lsl[0], lsl[1], N_SIGMA_L)
    vals = [J(v, th0, local=False) for v in grid_ls]
    ib = int(np.argmax([v[0] for v in vals]))
    ls, th = float(grid_ls[ib]), th0
    e_near = [r[0] for r in vals[ib][1]]
    dls = float(grid_ls[1] - grid_ls[0])
    for _round in range(2):
        ls = float(minimize_scalar(lambda v: -J(v, th)[0], bounds=(max(lsl[0], ls - dls), min(lsl[1], ls + dls)), method="bounded",
                                   options={"xatol": 0.01}).x)
        th = float(minimize_scalar(lambda v: -J(ls, v)[0], bounds=(th0 - 3.0 * sth, th0 + 3.0 * sth), method="bounded",
                                   options={"xatol": math.radians(0.1)}).x)
        _, res = J(ls, th)
        e_near = [r[0] for r in res]
        pis = [_fit_pi(g, res[k][0], (math.exp(ls), _gain(g, th)), pis[k])[0] for k, g in enumerate(groups)]
    f0_, _ = J(ls, th)
    sds = []
    for d_ls, d_th in ((0.05, 0.0), (0.0, math.radians(0.25))):
        fp, _ = J(ls + d_ls, th + d_th)
        fm, _ = J(ls - d_ls, th - d_th)
        h_ = d_ls or d_th
        c2 = (fp - 2.0 * f0_ + fm) / (h_ * h_)
        sds.append(math.sqrt(-1.0 / c2) if c2 < 0 else math.inf)
    return ls, th, pis, sds[0], sds[1], float(f0_)


def calibrated_sensor(cal: Optional[dict]) -> Optional[dict]:
    """The calibration's edge response (calib 'g1: edge_response:'): {sigma_l_m, sigma_l_log_sd, theta_tri_deg, theta_tri_sigma_deg}
    → normalised, or None when it does not state all four."""
    if not cal:
        return None
    try:
        out = {"sigma_L_m": float(cal["sigma_l_m"]), "log_sigma_L_sd": float(cal["sigma_l_log_sd"]),
               "theta_tri_deg": float(cal["theta_tri_deg"]), "theta_tri_sd_deg": float(cal["theta_tri_sigma_deg"])}
    except (KeyError, TypeError, ValueError):
        return None
    if not (out["sigma_L_m"] > 0 and all(math.isfinite(v) and v >= 0 for v in out.values())):
        return None
    out["source"] = str(cal.get("source", "calibration"))
    return out


def edge_readings(groups: list, cal: Optional[dict] = None, map_fn=None, n_children: int = 1) -> tuple:
    """Readings of the plate ends from all legs' top-edge data. The sensor's edge spread σ_L and triangulation angle θ are
    calibration constants (`cal`, calibrated_sensor: value and σ — no re-estimation; their σ enters the fusion as common terms through
    each reading's derivatives) or, for a sensor whose calibration does not state them, estimated online from all readings
    (σ_L log-normal prior SIGMA_L_MEDIAN_M × exp(±SIGMA_L_LOG_SD), θ prior N(THETA_TRI_DEG, THETA_TRI_SIGMA_DEG); Laplace σ).
    Readings are independent given the sensor parameters: with map_fn (the worker's fork_map) they are computed in n_children
    processes. → (readings, sensor)."""
    if not groups:
        return [], {}
    pi_start = PI_START
    sen_cal = calibrated_sensor(cal)
    if sen_cal is not None:
        ls, th = math.log(sen_cal["sigma_L_m"]), math.radians(sen_cal["theta_tri_deg"])
        sd_ls, sd_th = sen_cal["log_sigma_L_sd"], math.radians(sen_cal["theta_tri_sd_deg"])
        pis = [pi_start] * len(groups)
        source, ll = sen_cal["source"], float("nan")
    else:
        ls, th, pis, sd_ls, sd_th, ll = _search_sensor(groups, (math.log(SIGMA_L_MEDIAN_M), SIGMA_L_LOG_SD),
                                                       (math.radians(THETA_TRI_DEG), math.radians(THETA_TRI_SIGMA_DEG)),
                                                       [pi_start] * len(groups))
        source = "online (no calibrated edge response)"
    steps = (DERIV_STEP_LOG, math.radians(DERIV_STEP_THETA_DEG))
    sen = (math.exp(ls), th)
    jobs = list(range(len(groups)))

    def one(ks):
        return [_reading(groups[k], sen, pis[k], steps) for k in ks]
    if map_fn is not None and n_children > 1 and len(groups) > 1:
        n = min(int(n_children), len(groups))
        chunks = [jobs[i::n] for i in range(n)]
        parts = map_fn(one, chunks, n)
        res = [None] * len(groups)
        for ch, rr in zip(chunks, parts):
            for k, r in zip(ch, rr):
                res[k] = r
    else:
        res = one(jobs)
    readings = []
    for g, r in zip(groups, res):
        # the along-track speed of the profiles that cross this end (the time offset moves the reading by c_v·δ): the rows whose sheet
        # centre lies within the edge response of the read end — a pass that starts or stops near the end is still slow there (lab and
        # sim legs start ~2 mm beyond the end: the median over the whole region took the standstill rows)
        c_v = g["c_v"]
        if "v_row" in g and np.size(g["v_row"]):
            dsr = np.abs(np.asarray(g["s0_row"]) - r["u1_m"])
            ds_ = np.diff(np.sort(np.asarray(g["s0_row"])))
            step = float(np.median(ds_[ds_ > 0])) if np.any(ds_ > 0) else 0.0
            sel = dsr <= max(2.0 * math.exp(ls), step) + math.sqrt(max(r["u1_var"], 0.0))
            if not np.any(sel):
                sel = dsr <= float(np.min(dsr))
            c_v = float(np.median(np.asarray(g["v_row"])[sel]))
        readings.append({kk: g[kk] for kk in ("end", "plate", "leg", "cls", "direction", "lit", "pitch_gain", "n_rows",
                                                "n_points", "c_n")} | {"c_v": c_v, "motion_sign": int(g.get("motion_sign", 0))}
                        | {"kind": "top_edge", "u1_m": r["u1_m"], "u1_var": r["u1_var"], "h_star_m": g["top_h"],
                           "h_eff_m": g["top_h"], "top_h_m": g["top_h"], "c_u": 1.0, "floor_m": g["floor"], "sd0_m": g["sd0"],
                           "gain": _gain(g, th), "d_log_sigma_L": r["d_log_sigma_L"], "d_theta_rad": r["d_theta_rad"],
                           "rho_visible": float(r["pi"][0]), "pi_corner": float(r["pi"][1]), "nu_none": float(r["pi"][2]),
                           "pi_support": float(r["pi"][3]), "face_depth_m": float(g.get("face_depth", 0.0) or 0.0),
                           "sig_row_m": g["sig_row"], "region_m": g["region"]})
    sensor = {"sigma_L_m": sen_cal["sigma_L_m"] if sen_cal else math.exp(ls), "log_sigma_L_sd": sd_ls,
              "theta_tri_deg": sen_cal["theta_tri_deg"] if sen_cal else math.degrees(th),
              "theta_tri_sd_deg": math.degrees(sd_th), "source": source, "log_likelihood": ll, "n_readings": len(readings)}
    return readings, sensor


def pool_edge_response(runs: dict, th_prior_deg: tuple = (THETA_TRI_DEG, THETA_TRI_SIGMA_DEG)) -> dict:
    """Calibration of the edge response from scans of square plate ends (runs: name → top-edge groups of its estimate): the pooled
    value is the joint mode over all runs (σ_L flat over SIGMA_L_PRIOR_M, θ with the independent lobe prior th_prior_deg); its σ
    from the replication over LEGS (each leg re-estimates log σ_L at the pooled θ and θ at the pooled σ_L): standard error s/√n
    widened to the Student-t predictive variance (ν = n − 1, ν/(ν − 2)). The between-run spread of the leg estimates is reported
    (one-way ANOVA F) to show whether the run, not the leg, is the replication unit."""
    allg = [g for gs in runs.values() for g in gs]
    th0 = (math.radians(th_prior_deg[0]), math.radians(th_prior_deg[1]))
    pis0 = [PI_START] * len(allg)
    ls, th, pis, sd_ls, sd_th, ll = _search_sensor(allg, None, th0, pis0)
    from scipy.optimize import minimize_scalar
    lsl = (math.log(SIGMA_L_PRIOR_M[0]), math.log(SIGMA_L_PRIOR_M[1]))
    at = {}
    for g, p in zip(allg, pis):                         # each reading's mode at the pooled sensor: its window for the leg searches
        e, p2, _, _ = _mode(g, (math.exp(ls), _gain(g, th)), p)
        at[id(g)] = (_local(g, e), p2)
    per_leg = []
    for run, gs in runs.items():
        for leg in sorted({g["leg"] for g in gs}):
            lg = [g for g in gs if g["leg"] == leg]

            def J(ls_, th_):
                return sum(_argmax_e(g, at[id(g)][0], (math.exp(ls_), _gain(g, th_)), at[id(g)][1])[1] for g in lg)
            l_ = float(minimize_scalar(lambda v: -J(v, th), bounds=(max(lsl[0], ls - 1.5), min(lsl[1], ls + 1.5)),
                                       method="bounded", options={"xatol": 0.005}).x)
            t_ = float(minimize_scalar(lambda v: -J(ls, v), bounds=(th - math.radians(10.0), th + math.radians(10.0)),
                                       method="bounded", options={"xatol": math.radians(0.05)}).x)
            per_leg.append({"run": run, "leg": leg, "n_readings": len(lg), "log_sigma_L": l_, "theta_rad": t_})
    out = {"sigma_l_m": math.exp(ls), "theta_tri_deg": math.degrees(th), "laplace_log_sd": sd_ls, "laplace_theta_sd_deg":
           math.degrees(sd_th), "per_leg": per_leg, "runs": list(runs), "theta_prior_deg": list(th_prior_deg)}
    n = len(per_leg)
    for key, out_key, conv in (("log_sigma_L", "sigma_l_log_sd", 1.0), ("theta_rad", "theta_tri_sigma_deg", 180.0 / math.pi)):
        v = np.array([q[key] for q in per_leg])
        s = float(np.std(v, ddof=1)) if n > 1 else math.inf
        nu = n - 1
        out[out_key] = (s / math.sqrt(n) * math.sqrt(nu / (nu - 2.0)) * conv) if nu > 2 else math.inf
        out[out_key + "_leg_sd"] = s * conv
        groups_ = [np.array([q[key] for q in per_leg if q["run"] == r]) for r in runs]
        groups_ = [x for x in groups_ if x.size]
        if len(groups_) > 1 and n > len(groups_):
            from scipy.stats import f_oneway
            F = f_oneway(*groups_) if all(x.size > 1 for x in groups_) else None
            out[out_key + "_between_runs_F"] = None if F is None else {"F": float(F.statistic), "p": float(F.pvalue)}
    out["dof"] = n - 1
    return out


# ====================================================================================================================
# fusion over legs: plate ends, common terms, variance components (σ_ζ, τ) with small-sample posterior, inner plate
# ====================================================================================================================
def _rank(M: np.ndarray) -> int:
    if not M.size:
        return 0
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv > REL_NULL * max(float(sv[0]), 1e-300)))


def echo_branch_vec(dpri, m_L: np.ndarray, V_L: np.ndarray):
    """The echo branch of δ at every variance-grid point of the plate-end fusion (js_source.DeltaPrior, vectorised): the readings'
    own likelihood of δ N(m_L, V_L) (V_L ∞: they carry none) times the echo prior Φ(δ/σ_q) (τ_echo ≥ 0 flat ⊗ quantisation) — times
    the groove fusion's likelihood N(m0, v0) when the prior carries one (ref_lik_mean / ref_lik_var: then proper everywhere, the
    branch where the readings carry nothing being Φ·N(m0, v0)/Z0 itself). → (posterior mean, variance, log-evidence relative to the
    measured branch's N(m_L; 0, V_L + σ_q²)) arrays, or None when the flat lag is improper at some grid point (no earlier likelihood
    and the readings leave δ unidentified there). Each element equals DeltaPrior._branches(m_L, V_L)['ref'] (its ln Z less the
    measured branch's and, with an earlier likelihood, less ln Z0 = ln Φ(m0/√(v0 + σ_q²)): the prior's normaliser)."""
    from .js_source import probit_tilt_vec
    s = float(dpri.sigma_q)
    s2 = s * s
    m_L = np.asarray(m_L, dtype=float)
    V_L = np.asarray(V_L, dtype=float)
    ok = np.isfinite(V_L)
    Vf = np.where(ok, V_L, 1.0)
    mL = np.where(ok, m_L, 0.0)
    lm = -0.5 * np.log(Vf + s2) - 0.5 * mL * mL / (Vf + s2)        # ln N(m_L; 0, V_L + σ_q²) + ½ ln 2π
    if dpri.ref_lik_var is not None and math.isfinite(float(dpri.ref_lik_var)) and float(dpri.ref_lik_var) > 0 and s > 0:
        m0, v0 = float(dpri.ref_lik_mean), float(dpri.ref_lik_var)
        v_c = np.where(ok, 1.0 / (1.0 / Vf + 1.0 / v0), v0)
        m_c = np.where(ok, v_c * (mL / Vf + m0 / v0), m0)
        mu_r, v_r, lZ = probit_tilt_vec(m_c, v_c, s)
        lZ0 = float(log_ndtr(m0 / math.sqrt(v0 + s2)))
        lr = -0.5 * np.log(Vf + v0) - 0.5 * (mL - m0) ** 2 / (Vf + v0) + lZ - lZ0
        return mu_r, v_r, np.where(ok, lr - lm, 0.0)
    if not bool(np.all(ok)) or not s > 0:
        return None
    mu_r, v_r, lZ = probit_tilt_vec(m_L, V_L, s)
    return mu_r, v_r, lZ - lm + 0.5 * math.log(2.0 * math.pi)


def fuse_end_faces(readings: list, prior, line: dict, sigma_zeta: dict, sensor: Optional[dict] = None,
                   vc_out: Optional[dict] = None) -> dict:
    """Fused plate ends (square end faces — the declared joint: one end position per plate and end at every height) and the inner
    ends of the two plates, with small-sample honest σ.

    Reading model: u1 = e_Xp + c_v·δ + h*·g·w_x + d_σ·Δlog σ_L + d_θ·Δθ − c_u·(β_class + d_leg·b_u) + ζ_(leg, X) + ε,
      fixed       e_Xp (flat prior); b_u — the robot's along-track motion-direction offset (joint friction × compliance), times the
                  leg's motion sign d (+1 / −1, line['leg_direction']) — only where the rank of the fixed-effect design (ends, class
                  offsets taken as free) grows with it: readings of both directions; its zero is the midpoint of the two directions
                  (static frame);
      common      δ ~ N(0, σ_δ²) with σ_δ from the run's timing (line['dt_sigma_s']: one frame period and one pose sample period —
                  no tuned constant; with one speed c_v·δ and d·b_u are collinear, the data's share of δ's posterior precision says
                  whether this run separates them), w_x ~ N(0, σ_φ²) (only when the calibration states σ_φ), Δlog σ_L, Δθ ~ the edge-response
                  calibration's σ (sensor) — one value for all readings, so they do not average down over legs; β ~ N(0, σ_he²) per
                  non-gauge class;
      ζ           one along-track offset per leg and end, shared by its two plate readings (robot path / timing at that crossing),
                  N(0, σ_ζ²) — the legs' along-track scatter (the line's lateral / normal leg variances say nothing about it; 10 lab
                  runs of 2026-10-11 at 60 mm/s: a leg's S and E residuals about their direction medians are uncorrelated, r −0.42 …
                  0.75, mean 0.01 — the scatter is per crossing, not a per-leg shift);
      ε           N(0, v + τ²) — the reading's own posterior variance plus a per-reading scatter τ².
    σ_ζ and τ are unknown variance components. Their posterior on a log grid = the restricted likelihood (fixed effects and common
    terms integrated out) × their prior: calibrated (calib g1: noise_priors end_crossing / end_reading — the population of runs,
    σ²(v) = λ·s0²(v), λ ~ Scaled-Inv-χ²(ν0, 1); a crossing or reading at speed v carries s0²(v)/s0²(v_ref) of the reference
    speed's variance) or, uncalibrated, the scale-invariant prior on each total variance, p(σ²) ∝ 1/(v̄ + σ²) (v̄ = the readings'
    median own variance); the grid is refined to where the posterior holds its mass; all quantities are mixtures over that posterior (the small-sample answer: with few crossings the posterior of σ_ζ
    has a heavy tail and so do the ends — a Student-t-like predictive, not a plug-in REML σ). The tail of σ_ζ falls as σ_ζ^−ν with
    ν = rank[X Z] − rank[X] (leg-end crossings beyond the plate ends' own unknowns) plus the calibrated ν0: the ends' variance is
    finite only for ν0 + ν > 2 — otherwise the ends are reported with infinite σ (not established). Robust: a gross-error class per reading
    (uniform over its region; MLESAC responsibilities on leave-one-out predictions, ε by EM) — its weight multiplies the reading's
    precision."""
    from .js_source import DeltaPrior
    from .seam_estimate import _adds_rank, leg_direction
    keys = sorted({(r["end"], r["plate"]) for r in readings})
    gauge = line.get("gauge")
    dt_sig = float(line.get("dt_sigma_s", prior.dt_sigma_s))
    dpri = line.get("delta_prior") or DeltaPrior(max(dt_sig, 0.0), 0.0)
    p_ref = float(dpri.p_ref) if dt_sig > 0 else 0.0
    ref_info = {"p_ref": p_ref, "delta_identified": True}
    if not readings:
        sp = float(getattr(prior, "laser_pitch_sigma_rad", float("nan")))
        nil = {"s_m": float("nan"), "sigma_m": math.inf, "inner_plate": None}
        return {"plates": {}, "ends": {X: {"root": dict(nil), "top_measured": dict(nil), "top": dict(nil)} for X in ("S", "E")},
                "delta_s": 0.0, "delta_sigma_s": (1.0 / math.sqrt(dpri.prior_precision())) if (dt_sig > 0 and
                                                                                       dpri.prior_precision() > 0) else math.inf,
                "delta_prior_sigma_s": dt_sig,
                "delta_data_share": 0.0, "delta_separable": False, "b_u": 0.0, "b_u_sigma": math.inf, "b_u_identified": False,
                "along_track_direction": None, "w_x_rad": 0.0,
                "w_x_sigma_rad": sp if (math.isfinite(sp) and sp > 0) else math.inf, "beta_u": {}, "eta_u": {}, "zeta_u": {},
                "tau": {}, "tau_identified": {}, "tau_e_m": float("nan"), "sigma_leg_end_m": float("nan"), "nu_leg_end": 0,
                "established": False, "huber_weights": [], "gross_fraction": float("nan"), "n_readings": 0}
    he = float(prior.he_sigma_m) if prior.he_sigma_m is not None else float("nan")
    sphi = float(getattr(prior, "laser_pitch_sigma_rad", float("nan")))
    sen = sensor or {}
    sd_ls = float(sen.get("log_sigma_L_sd", float("nan")))
    sd_th = math.radians(float(sen.get("theta_tri_sd_deg", float("nan"))))
    fixed = [f"e0:{X}:{p}" for X, p in keys]
    # the along-track direction offset b_u: in the model where the readings identify it (rank of [ends, class offsets as free] grows)
    dsg = np.array([float(leg_direction(line, r["leg"])) for r in readings])
    cls_ng = sorted({r["cls"] for r in readings} - {gauge})
    XF0 = np.array([[1.0 if (r["end"], r["plate"]) == k_ else 0.0 for k_ in keys]
                    + [-r["c_u"] if r["cls"] == c else 0.0 for c in cls_ng] for r in readings])
    if _adds_rank(XF0, np.array([-r["c_u"] for r in readings]) * dsg):
        fixed.append("b_u")
    common = []                                           # (name, prior σ) — random with a known σ
    if dt_sig > 0:
        common.append(("delta", dt_sig))
    if math.isfinite(sphi) and sphi > 0:
        common.append(("w_x", sphi))
    if math.isfinite(sd_ls) and sd_ls > 0:
        common.append(("cal_log_sigma_L", sd_ls))
    if math.isfinite(sd_th) and sd_th > 0:
        common.append(("cal_theta", sd_th))
    for c in sorted({r["cls"] for r in readings} - {gauge}):
        if math.isfinite(he) and he > 0:
            common.append(("beta:" + c, he))
        else:
            fixed.append("beta:" + c)
    # no along-track leg offset η_u with the line's σ: the line measures the legs' lateral / normal scatter only (one variance
    # component each); the legs' along-track scatter is the end data's own — ζ_(leg, X), its variance estimated below
    zg = sorted({(r["leg"], r["end"]) for r in readings})
    names = fixed + [n_ for n_, _ in common] + [f"zeta:{a}:{b}" for a, b in zg]
    col = {n_: i for i, n_ in enumerate(names)}
    P, nF, nC = len(names), len(fixed), len(common)
    n = len(readings)
    A = np.zeros((n, P))
    y = np.zeros(n)
    v0 = np.zeros(n)
    for i, r in enumerate(readings):
        A[i, col[f"e0:{r['end']}:{r['plate']}"]] = 1.0
        if "delta" in col:
            A[i, col["delta"]] = r["c_v"]
        if "w_x" in col:
            A[i, col["w_x"]] = r["h_star_m"] * r["pitch_gain"]
        if "cal_log_sigma_L" in col:
            A[i, col["cal_log_sigma_L"]] = r.get("d_log_sigma_L", 0.0)
        if "cal_theta" in col:
            A[i, col["cal_theta"]] = r.get("d_theta_rad", 0.0)
        if "beta:" + r["cls"] in col:
            A[i, col["beta:" + r["cls"]]] = -r["c_u"]
        if "eta:" + r["leg"] in col:
            A[i, col["eta:" + r["leg"]]] = -r["c_u"]
        if "b_u" in col:
            A[i, col["b_u"]] = -r["c_u"] * dsg[i]
        A[i, col[f"zeta:{r['leg']}:{r['end']}"]] = 1.0
        y[i] = r["u1_m"]
        v0[i] = r["u1_var"]          # (the profiles' pose jitter is in the reading's own row noise σ_row — not added again)
    lam_c = np.array([1.0 / sd ** 2 for _, sd in common])
    vbar = float(np.median(v0)) if n else 1e-12
    m_z = len(zg)
    XF = A[:, :nF]
    # the variance components' priors: calibrated (calib g1: noise_priors — the robot + sensor population, speed model s0²(v) =
    # a + b·v², λ ~ Scaled-Inv-χ²(ν0, 1); noise_priors module) or, uncalibrated, the scale-invariant prior on each total variance.
    # σ_ζ and τ are reported at the run's reference speed v_ref (median along-track speed of the moving readings); a crossing or a
    # reading at another speed has its variance scaled by s0²(v)/s0²(v_ref)
    from .noise_priors import parse_noise_priors, refine_bounds
    npri = parse_noise_priors(getattr(prior, "noise_priors", None))
    pz, pt = npri.get("end_crossing"), npri.get("end_reading")
    v_rd = np.array([abs(float(r.get("c_v") or 0.0)) for r in readings])
    mov = v_rd[v_rd > 0]
    v_ref = float(np.median(mov)) if mov.size else 0.0
    v_z = np.array([float(np.median(v_rd[[i for i, r in enumerate(readings) if (r["leg"], r["end"]) == g]])) for g in zg]) \
        if zg else np.zeros(0)
    rz = (pz.s2(v_z) / pz.s2(v_ref)) if pz is not None else np.ones(m_z)
    rt = (pt.s2(v_rd) / pt.s2(v_ref)) if pt is not None else np.ones(n)

    def lprior(sig, p_):
        return p_.log_density_log_sigma(sig, v_ref) if p_ is not None else np.log(sig ** 2 / (vbar + sig ** 2))

    def axis0(p_):
        """Coarse ln σ axis: the data's span about the readings' own σ, widened to the calibrated prior's central range."""
        lo, hi = 0.5 * math.log(max(vbar, 1e-30)) + SIG_SPAN[0], 0.5 * math.log(max(vbar, 1e-30)) + SIG_SPAN[1]
        if p_ is not None:
            lo = min(lo, math.log(p_.quantile_sigma(PRIOR_TAIL, v_ref)))
            hi = max(hi, math.log(p_.quantile_sigma(1.0 - PRIOR_TAIL, v_ref)))
        return np.linspace(lo, hi, SIG_GRID)

    def mesh(az, at):
        a_, b_ = np.meshgrid(np.exp(az), np.exp(at), indexing="ij")
        return a_.ravel(), b_.ravel()

    def grid(hw, SZ, TT):
        """Posterior over the (σ_ζ, τ) grid: weights, means and covariances of all unknowns."""
        Wg = hw[None, :] / (v0[None, :] + TT[:, None] ** 2 * rt[None, :])
        lam = np.zeros((SZ.size, P))
        lam[:, nF:nF + nC] = lam_c[None, :]
        lam[:, nF + nC:] = 1.0 / (SZ[:, None] ** 2 * rz[None, :])
        N = np.einsum("gn,ni,nj->gij", Wg, A, A)
        N[:, np.arange(P), np.arange(P)] += lam
        b = np.einsum("gn,ni->gi", Wg * y[None, :], A)
        dS = 1.0 / np.sqrt(np.diagonal(N, axis1=1, axis2=2))        # Jacobi scaling (fixed vs. tight random terms differ by 1e10)
        Ns = dS[:, :, None] * N * dS[:, None, :]
        Lc = np.linalg.cholesky(Ns)
        Nsi = np.linalg.inv(Ns)
        Ninv = dS[:, :, None] * Nsi * dS[:, None, :]
        x = np.einsum("gij,gj->gi", Ninv, b)
        logdet = 2.0 * np.sum(np.log(np.diagonal(Lc, axis1=1, axis2=2)), axis=1) - 2.0 * np.sum(np.log(dS), axis=1)
        r = y[None, :] - x @ A.T                                   # quadratic form as residuals + prior terms (no cancellation)
        quad = np.sum(Wg * r ** 2, axis=1) + np.sum(lam * x ** 2, axis=1)
        ll = 0.5 * np.sum(np.log(Wg), axis=1) - m_z * np.log(SZ) - 0.5 * logdet - 0.5 * quad
        lw = ll + lprior(SZ, pz) + lprior(TT, pt)
        if p_ref > 0.0 and "delta" in col:
            return _with_echo_branch(lw, x, Ninv) + (Wg,)
        w = np.exp(lw - np.max(lw))
        return w / w.sum(), x, Ninv, Wg

    def _with_echo_branch(lw, x, Ninv):
        """The joint-state hypotheses at every grid point (js_source: a cut mixture, weight p_ref on the reference echo). The grid
        is computed under the measured source's δ ~ N(0, σ_q²); δ's prior only reshapes δ's marginal, the other unknowns follow it
        by Gaussian conditioning (regression k = Cov(x, δ)/Var δ). Echo branch: the data's own likelihood of δ, N(m_L, V_L) (the
        measured branch's posterior with N(0, σ_q²) divided out), times the echo's δ prior Φ(δ/σ_q) (τ_echo ≥ 0 flat ⊗
        quantisation) — times the groove fusion's likelihood of δ N(m0, v0) when given (seam_estimate._end_faces; proper: a flat lag
        here is identified only through the speed contrast of readings scattered by σ_ζ — at large σ_ζ the lag's posterior runs off,
        synthetic two-speed run: δ 22 s; a likelihood, not that fusion's posterior: the half-line holds — KIRO stand-in 2026-10-13,
        a Gaussian posterior handed on took δ to −471 ms). Its grid log-weights = the measured ones with the δ evidence factor
        N(m_L; 0, V_L + σ_q²) exchanged for the echo's. A flat lag the readings do not inform (one speed: δ and b_u collinear) is
        improper along that direction: taken given δ = 0 (b_u absorbs the lag for every reading of that speed, so every identified
        quantity — the ends, the along-track direction term — is exact) and δ, b_u are reported unidentified. → per-grid-point
        mixture (w, x, Ninv) of the two branches."""
        jd = col["delta"]
        s2 = dt_sig ** 2
        vm = Ninv[:, jd, jd]
        mm = x[:, jd]
        prec = 1.0 / vm - 1.0 / s2                                  # the readings' own precision on δ
        k = Ninv[:, :, jd] / vm[:, None]
        V_L = np.where(prec > REL_NULL / s2, 1.0 / np.where(prec > 0, prec, 1.0), np.inf)
        m_L = np.where(np.isfinite(V_L), np.where(np.isfinite(V_L), V_L, 0.0) * mm / vm, 0.0)
        br = echo_branch_vec(dpri, m_L, V_L)
        if br is not None:
            mu_r, v_r, dlw = br
            lw_r = lw + dlw
            x_r = x + k * (mu_r - mm)[:, None]
            C_r = Ninv + k[:, :, None] * k[:, None, :] * (v_r - vm)[:, None, None]
        else:                                                       # flat lag, improper along the δ direction: given δ = 0
            ref_info["delta_identified"] = False
            lw_r = lw.copy()
            x_r = x - k * mm[:, None]
            C_r = Ninv - k[:, :, None] * k[:, None, :] * vm[:, None, None]
        e_m, e_r = np.exp(lw - np.max(lw)), np.exp(lw_r - np.max(lw_r))
        a_m = (1.0 - p_ref) * e_m / np.sum(e_m)                     # each branch normalised over its grid, weighted by P(H)
        a_r = p_ref * e_r / np.sum(e_r)
        wg = a_m + a_r
        f_m = np.where(wg > 0, a_m / np.where(wg > 0, wg, 1.0), 0.0)
        f_r = 1.0 - f_m
        xg = f_m[:, None] * x + f_r[:, None] * x_r
        dm, dr = x - xg, x_r - xg
        Cg = (f_m[:, None, None] * (Ninv + dm[:, :, None] * dm[:, None, :])
              + f_r[:, None, None] * (C_r + dr[:, :, None] * dr[:, None, :]))
        return wg / np.sum(wg), xg, Cg

    def posterior(hw):
        """The grid posterior on a coarse ln σ grid, then on a grid refined to where it holds its mass (a calibrated prior can be
        far narrower than the coarse step). → (w, x, Ninv, Wg, SZ, TT)."""
        az, at = axis0(pz), axis0(pt)
        SZ, TT = mesh(az, at)
        w = grid(hw, SZ, TT)[0].reshape(az.size, at.size)
        bz, bt = refine_bounds(az, w.sum(axis=1)), refine_bounds(at, w.sum(axis=0))
        SZ, TT = mesh(np.linspace(bz[0], bz[1], SIG_GRID), np.linspace(bt[0], bt[1], SIG_GRID))
        return grid(hw, SZ, TT) + (SZ, TT)

    # gross misreadings (a reading locked on a ghost or debris): each reading is the model's (prob 1 − ε) or a gross error uniform
    # over its own region (ε); responsibilities on leave-one-out predictions, ε by EM (MLESAC). The first round screens at the
    # smallest per-reading scatter τ (an outlier must not hide in an inflated τ), later rounds at the posterior mode.
    width = np.array([max(float(r["region_m"][1]) - float(r["region_m"][0]), 1e-6) for r in readings])
    hw = np.ones(n)
    eps = 1.0 / (n + 1.0)
    w = x = Ninv = Wg = SZ = TT = None
    for rnd in range(ROBUST_ROUNDS):
        w, x, Ninv, Wg, SZ, TT = posterior(hw)
        if rnd == 0:
            low = TT == np.min(TT)
            g_ = int(np.flatnonzero(low)[np.argmax(w[low])]) if np.any(w[low] > 0) else int(np.argmax(w))
        else:
            g_ = int(np.argmax(w))
        lamg = np.r_[np.zeros(nF), lam_c, 1.0 / (SZ[g_] ** 2 * rz)]
        echo = p_ref > 0.0 and "delta" in col
        if echo:
            lamg[col["delta"]] = 0.0                          # δ's factor: its site given the other readings (below)
        dens_in = np.full(n, np.nan)
        z_loo = np.full(n, np.nan)
        for i in range(n):                                  # leave-one-out at that grid point
            Wd = Wg[g_].copy()
            Wd[i] = 0.0
            fx = np.flatnonzero(A[i, :nF])
            if any(not np.any((A[:, j] != 0) & (Wd > 0)) for j in fx):
                continue                                    # nothing else measures this reading's plate end: it cannot be judged
            Nd = (A.T * Wd) @ A + np.diag(lamg)
            if echo:                                        # the joint-state hypotheses' δ (js_source): EP site from the others
                from .seam_estimate import _NormalInverse
                jd_ = col["delta"]
                bd = A.T @ (Wd * y)
                Ik = _NormalInverse(Nd)
                V_L = Ik.var(np.eye(P)[jd_])
                m_L = float(Ik.step(bd, np.zeros(P))[jd_]) if math.isfinite(V_L) else 0.0
                lam_s, lam_sm = dpri.site(m_L, V_L)
                Nd[jd_, jd_] += lam_s
                bd[jd_] += lam_sm
                Ik = _NormalInverse(Nd)
                va = Ik.var(A[i])
                if not math.isfinite(va):
                    continue                                # the reading touches what the others leave unidentified
                xd = Ik.step(bd, np.zeros(P))
                vp = va + (v0[i] + TT[g_] ** 2 * rt[i])
            else:
                try:
                    xd = np.linalg.solve(Nd, A.T @ (Wd * y))
                    vp = float(A[i] @ np.linalg.solve(Nd, A[i])) + (v0[i] + TT[g_] ** 2 * rt[i])
                except np.linalg.LinAlgError:
                    continue
            if vp > 0 and math.isfinite(vp):
                dens_in[i] = math.exp(-0.5 * (y[i] - float(A[i] @ xd)) ** 2 / vp) / math.sqrt(2.0 * math.pi * vp)
                z_loo[i] = (y[i] - float(A[i] @ xd)) / math.sqrt(vp)
        ok = np.isfinite(dens_in)                         # a reading nothing else predicts cannot be judged: kept
        for _em in range(EM_ITERS):
            a_in = (1.0 - eps) * dens_in[ok]
            resp = a_in / (a_in + eps / width[ok]) if np.any(ok) else np.zeros(0)
            eps_new = (float(np.sum(1.0 - resp)) + 1.0) / (float(resp.size) + 2.0)   # rule of succession: ε stays inside (0, 1)
            if abs(eps_new - eps) < 1e-6:
                eps = eps_new
                break
            eps = eps_new
        hw_new = np.ones(n)
        if np.any(ok):
            a_in = (1.0 - eps) * dens_in[ok]
            hw_new[ok] = np.maximum(a_in / (a_in + eps / width[ok]), W_FLOOR)
        if np.max(np.abs(hw_new - hw)) < 1e-4:
            hw = hw_new
            break
        hw = hw_new
    # each reading's fit (§4.9 segment form: 2 ln odds of 'the model's reading' vs 'gross', k² decides) — for the model checks
    from .seam_estimate import segment_verdict
    k_ = float(getattr(prior, "k", 3.0))
    for i, r in enumerate(readings):
        if np.isfinite(dens_in[i]) and dens_in[i] > 0:
            tlo = 2.0 * (math.log(max(1.0 - eps, 1e-300)) + math.log(dens_in[i]) - math.log(eps / width[i]))
        elif np.isfinite(dens_in[i]):
            tlo = -math.inf
        else:
            tlo = float("nan")                              # nothing else measures its plate end: untestable
        r["fit"] = {"z": float(z_loo[i]), "two_ln_odds": float(tlo), "gamma_out": float(1.0 - hw[i]),
                    "verdict": segment_verdict(tlo, k_)}
    w, x, Ninv, Wg, SZ, TT = posterior(hw)
    if vc_out is not None:                              # the restricted likelihood's ingredients (noise-prior calibration)
        vc_out.update(A=A.copy(), y=y.copy(), v0=v0.copy(), hw=hw.copy(), nF=nF, nC=nC, lam_c=lam_c.copy(), m_z=m_z, rz=rz.copy(),
                      rt=rt.copy(), v_ref=v_ref, nu=None, nu_tau=None)
    inl = hw > 0.5                                      # readings more likely the model's than gross: the redundancy that counts
    nu = _rank(np.c_[XF[inl], A[inl, nF + nC:]]) - _rank(XF[inl])
    nu_tau = int(np.count_nonzero(inl)) - _rank(np.c_[XF[inl], A[inl, nF + nC:]])
    # the posterior degrees of freedom of σ_ζ: the run's redundant crossings + the calibrated population's ν0 (a calibrated σ_ζ is
    # known without redundant crossings: its posterior has a finite second moment for ν0 + ν > 2)
    nu_post = nu + (pz.nu0 if pz is not None else 0.0)
    # the static-frame ends need the along-track direction term: readings of a moving robot in ONE direction leave b_u out of the
    # model (not identified) — the ends are then in that direction's frame, offset by the unknown d·b_u: not established
    moving = bool(np.any(dsg[inl] != 0)) if np.any(inl) else False
    established = nu_post > 2 and ("b_u" in col or not moving)
    if vc_out is not None:
        vc_out.update(nu=int(nu), nu_tau=int(nu_tau))
    if established:
        xbar = np.einsum("g,gi->i", w, x)
        C = np.einsum("g,gij->ij", w, Ninv) + np.einsum("g,gi,gj->ij", w, x, x) - np.outer(xbar, xbar)
    else:                                               # the mixture's moments do not exist: locations at the posterior mode
        g_ = int(np.argmax(w))
        xbar, C = x[g_], Ninv[g_]
    q = np.cumsum(w[np.argsort(SZ, kind="stable")])
    sz_med = float(np.sort(SZ, kind="stable")[np.searchsorted(q, 0.5)])
    q = np.cumsum(w[np.argsort(TT, kind="stable")])
    tt_med = float(np.sort(TT, kind="stable")[np.searchsorted(q, 0.5)])

    def mv(c_):
        m_ = float(c_ @ xbar)
        v_ = float(c_ @ C @ c_)
        return m_, (v_ if established else math.inf)

    cw = np.zeros(P)
    if "w_x" in col:
        cw[col["w_x"]] = 1.0
    plates = {}
    for X, p in keys:
        rs = [r for r in readings if r["end"] == X and r["plate"] == p]
        hT = float(np.median([r["top_h_m"] for r in rs]))
        g0 = float(np.median([r["pitch_gain"] for r in rs]))
        ce0 = np.zeros(P)
        ce0[col[f"e0:{X}:{p}"]] = 1.0
        e0, v0_ = mv(ce0)
        etm, vtm = mv(ce0 + hT * g0 * cw)
        plates[(X, p)] = {"s_m": e0, "var": v0_, "top_h_m": hT, "measured_top_m": etm, "measured_top_var": vtm, "pitch_gain": g0,
                          "n_legs": len({r["leg"] for r in rs}), "c_e0": ce0, "c_top": ce0 + hT * g0 * cw}
    ends = {}
    for X in ("S", "E"):
        pl = [plates.get((X, p)) for p in PLATES]
        res = {}
        for which, ck in (("root", "c_e0"), ("top_measured", "c_top")):
            if any(q_ is None for q_ in pl):
                res[which] = {"s_m": float("nan"), "sigma_m": math.inf, "inner_plate": None}
                continue
            c1, c2 = pl[0][ck], pl[1][ck]
            m1, m2 = float(c1 @ xbar), float(c2 @ xbar)
            v1, v2, c12 = float(c1 @ C @ c1), float(c2 @ C @ c2), float(c1 @ C @ c2)
            if not established:                         # no established σ: the inner plate's location, no Clark smoothing
                mm, vv = (max(m1, m2) if X == "S" else min(m1, m2)), math.inf
            elif X == "S":
                mm, vv = _clark_max(m1, v1, m2, v2, c12)
            else:
                mm, vv = _clark_max(-m1, v1, -m2, v2, c12)
                mm = -mm
            inner = PLATES[0] if ((m1 >= m2) == (X == "S")) else PLATES[1]
            res[which] = {"s_m": mm, "sigma_m": math.sqrt(vv) if (established and math.isfinite(vv)) else math.inf,
                          "inner_plate": inner, "h_m": 0.0 if which == "root" else float(np.mean([q_["top_h_m"] for q_ in pl]))}
        res["top"] = dict(res["root"], h_m=res["top_measured"].get("h_m", float("nan")))
        ends[X] = res

    def comp(nm):
        if nm not in col:
            return 0.0, math.inf
        c_ = np.zeros(P)
        c_[col[nm]] = 1.0
        return float(c_ @ xbar), float(c_ @ C @ c_)
    dlt, v_d = comp("delta")
    bu, v_bu = comp("b_u") if "b_u" in col else (0.0, math.inf)
    if p_ref > 0.0 and not ref_info["delta_identified"]:    # the echo branch is improper along δ: δ and b_u apart are unknown
        v_d, v_bu = math.inf, math.inf
    wx, v_wx = comp("w_x") if "w_x" in col else (0.0, sphi ** 2 if math.isfinite(sphi) else math.inf)
    # the data's share of δ's posterior precision (0 = the prior alone: this run cannot tell latency from the direction offset); the
    # prior as a whole (js_source.DeltaPrior.prior_precision): N(0, σ_q²) for a measured source, flat once an echo has weight
    share = max(0.0, 1.0 - v_d * (dpri.prior_precision() if dt_sig > 0 else 0.0)) if ("delta" in col and math.isfinite(v_d)) \
        else 0.0
    # the combined along-track direction term at the run's speed: a reading in direction d moves by d·A, A = v̄·δ − b_u (c_u ≈ 1)
    vmov = [abs(float(r["c_v"])) for r, d_ in zip(readings, dsg) if d_ != 0 and r.get("c_v")]
    atd = None
    if vmov:
        vbar = float(np.median(vmov))
        cA = np.zeros(P)
        if "delta" in col:
            cA[col["delta"]] = vbar
        if "b_u" in col:
            cA[col["b_u"]] = -1.0
        mA, vA = float(cA @ xbar), float(cA @ C @ cA)
        atd = {"value_m": mA, "sigma_m": math.sqrt(vA) if (established and math.isfinite(vA) and vA >= 0) else math.inf,
               "split_m": 2.0 * mA, "speed_mps": vbar, "b_u_identified": "b_u" in col}
    # posterior of the fixed / common location terms (ends, δ, b_u, w_x) for predictions (leave-one-out checks)
    pn = [nm for nm in names if nm.startswith("e0:") or nm in ("delta", "b_u", "w_x")]
    pj = [col[nm] for nm in pn]
    gkey = f"e:{gauge}"
    return {"plates": plates, "ends": ends, "delta_s": dlt, "delta_sigma_s": math.sqrt(v_d) if math.isfinite(v_d) else math.inf,
            "delta_prior_sigma_s": dt_sig, "delta_data_share": float(share), "delta_separable": bool(share > 0.5),
            "delta_prior": dict(dpri.as_dict(), p_ref_used=p_ref, delta_identified_by_readings=bool(ref_info["delta_identified"])),
            "b_u": bu, "b_u_sigma": math.sqrt(v_bu) if (established and math.isfinite(v_bu)) else math.inf,
            "b_u_identified": "b_u" in col, "along_track_direction": atd, "moving_readings": bool(np.any(dsg != 0)),
            "posterior": {"names": pn, "mean": [float(xbar[j]) for j in pj],
                          "cov": [[float(C[a_, b_]) if established else math.inf for b_ in pj] for a_ in pj]},
            "w_x_rad": wx, "w_x_sigma_rad": math.sqrt(v_wx) if math.isfinite(v_wx) else math.inf,
            "beta_u": {nm.split(":", 1)[1]: float(xbar[j]) for nm, j in col.items() if nm.startswith("beta:")},
            "eta_u": {nm.split(":", 1)[1]: float(xbar[j]) for nm, j in col.items() if nm.startswith("eta:")},
            "zeta_u": {nm.split(":", 1)[1]: float(xbar[j]) for nm, j in col.items() if nm.startswith("zeta:")},
            "tau": {gkey: tt_med}, "tau_identified": {gkey: bool(established)},
            "tau_e_m": tt_med if established else float("nan"),
            "sigma_leg_end_m": sz_med, "nu_leg_end": int(nu), "established": bool(established),
            "nu_reading": int(nu_tau), "nu_posterior": float(nu_post), "speed_ref_mps": v_ref,
            "variance_priors": {"end_crossing": None if pz is None else pz.to_dict(), "end_reading": None if pt is None else pt.to_dict(),
                                "source": "calibration (g1 noise_priors)" if (pz or pt) else "scale-invariant (not calibrated)"},
            "huber_weights": [float(w_) for w_ in hw], "gross_fraction": float(eps), "n_readings": n}


def vc_loglik(design: dict, sig_z: np.ndarray, sig_t: np.ndarray) -> np.ndarray:
    """The plate-end fusion's restricted log-likelihood of (σ_ζ, τ) at the reference speed on the grid sig_z × sig_t (no prior; the
    fixed effects and common terms integrated out; the run's final robust weights) — the run's evidence about its variance
    components, as fuse_end_faces uses it (design = its vc_out). → (len(sig_z), len(sig_t))."""
    A, y, v0, hw = design["A"], design["y"], design["v0"], design["hw"]
    nF, nC, lam_c, m_z, rz, rt = design["nF"], design["nC"], design["lam_c"], design["m_z"], design["rz"], design["rt"]
    P = A.shape[1]
    SZ, TT = np.meshgrid(np.asarray(sig_z, float), np.asarray(sig_t, float), indexing="ij")
    SZ, TT = SZ.ravel(), TT.ravel()
    out = np.empty(SZ.size)
    for c0 in range(0, SZ.size, 2048):                  # [R] chunk size (memory)
        sz, tt = SZ[c0:c0 + 2048], TT[c0:c0 + 2048]
        Wg = hw[None, :] / (v0[None, :] + tt[:, None] ** 2 * rt[None, :])
        lam = np.zeros((sz.size, P))
        lam[:, nF:nF + nC] = lam_c[None, :]
        lam[:, nF + nC:] = 1.0 / (sz[:, None] ** 2 * rz[None, :])
        N = np.einsum("gn,ni,nj->gij", Wg, A, A)
        N[:, np.arange(P), np.arange(P)] += lam
        b = np.einsum("gn,ni->gi", Wg * y[None, :], A)
        dS = 1.0 / np.sqrt(np.diagonal(N, axis1=1, axis2=2))
        Ns = dS[:, :, None] * N * dS[:, None, :]
        Lc = np.linalg.cholesky(Ns)
        x = dS * np.linalg.solve(Ns, (dS * b)[:, :, None])[:, :, 0]
        logdet = 2.0 * np.sum(np.log(np.diagonal(Lc, axis1=1, axis2=2)), axis=1) - 2.0 * np.sum(np.log(dS), axis=1)
        r = y[None, :] - x @ A.T
        quad = np.sum(Wg * r ** 2, axis=1) + np.sum(lam * x ** 2, axis=1)
        out[c0:c0 + sz.size] = (0.5 * np.sum(np.log(np.where(hw[None, :] > 0, Wg, 1.0)), axis=1) - m_z * np.log(sz)
                                - 0.5 * logdet - 0.5 * quad)
    return out.reshape(len(sig_z), len(sig_t))


def summarize(fz: dict, readings: list, prior) -> dict:
    """est.plate['end_faces']: per end the inner plate end (square faces: the same at the root and the top edge) and where the
    calibrated frame puts the top edge, per plate its end, the calibration term w_x, δ, the sensor's edge response and the variance
    components. Plain floats (JSON)."""
    def sd(v_):
        return math.sqrt(v_) if (v_ is not None and math.isfinite(v_) and v_ >= 0) else math.inf

    out = {"ends": {}, "plates": {}}
    for X in ("S", "E"):
        out["ends"][X] = {w: {k_: v for k_, v in d.items()} for w, d in fz["ends"][X].items()}
    for (X, p), q in fz["plates"].items():
        out["plates"][f"{X}:{p}"] = {"s_m": q["s_m"], "sigma_m": sd(q["var"]), "measured_top_s_m": q["measured_top_m"],
                                     "measured_top_sigma_m": sd(q["measured_top_var"]), "top_h_m": q["top_h_m"],
                                     "pitch_gain": q["pitch_gain"], "n_legs": q["n_legs"]}
    w, sw = fz["w_x_rad"], fz["w_x_sigma_rad"]
    sphi = float(getattr(prior, "laser_pitch_sigma_rad", float("nan")))
    out["w_x"] = {"rad": w, "sigma_rad": sw, "deg": math.degrees(w), "sigma_deg": math.degrees(sw) if math.isfinite(sw) else math.inf,
                  "calibration_prior_sigma_deg": math.degrees(sphi) if math.isfinite(sphi) else None,
                  "note": ("rotation of the true optical frame relative to the calibrated one about its laser-line (x) axis "
                           "(right-handed); from the calibration only (apparent end slants are the sensor's edge response, not w_x)")}
    out["sensor"] = fz.get("sensor", {})
    out["delta_s"], out["delta_sigma_s"] = fz["delta_s"], fz["delta_sigma_s"]
    out["time_offset"] = {"delta_s": fz["delta_s"], "sigma_s": fz["delta_sigma_s"], "prior_sigma_s": fz.get("delta_prior_sigma_s"),
                          "prior": fz.get("delta_prior"),
                          "data_share": fz.get("delta_data_share"), "separable_from_b_u": fz.get("delta_separable"),
                          "note": ("residual time offset after the calibrated dt; prior width from the run's frame and pose sample "
                                   "periods; data_share = the data's part of its posterior precision (> 0.5: the run separates it "
                                   "from the direction offset b_u)")}
    out["direction_offset_u"] = {"b_u_m": fz.get("b_u", 0.0), "sigma_m": fz.get("b_u_sigma", math.inf),
                                 "identified": fz.get("b_u_identified", False)}
    out["along_track_direction"] = fz.get("along_track_direction")
    out["posterior"] = fz.get("posterior")
    out["tau"], out["tau_identified"] = fz["tau"], fz["tau_identified"]
    out["variance_components"] = {"sigma_leg_end_m": fz.get("sigma_leg_end_m"), "tau_m": fz.get("tau_e_m"),
                                  "nu_leg_end": fz.get("nu_leg_end"), "nu_reading": fz.get("nu_reading"),
                                  "nu_posterior": fz.get("nu_posterior"), "speed_ref_mps": fz.get("speed_ref_mps"),
                                  "priors": fz.get("variance_priors"), "established": fz.get("established"),
                                  "note": ("posterior medians at the reference speed; the ends' sigma integrates over their posterior "
                                           "(small-sample); not established (sigma infinite) while the posterior degrees of freedom "
                                           "(redundant leg-end crossings + the calibrated prior's nu0) are <= 2")}
    out["readings"] = [{k_: v for k_, v in r.items() if not k_.startswith("_")} for r in readings]
    return out
