"""1G core seam estimator (WP2, G1_CORE_SPEC §4, §8.3) — legs of per-profile template fits → one SeamEstimate.

Model (taught frame F0 = g1_types.taught_frame; s from S0 along u0 in the taught S→E order):
  root line   r(s) = S0 + s·u0 + a(s)·w0 + b(s)·n0, a/b polynomials of degree 1 ('line') or 2 ('quadratic'), chosen by BIC (D10)
  offsets     measured root + o_ℓ = true root, o_ℓ = β_class + d_ℓ·b + η_leg (β per posture class with the gauge β_pass = 0 once
              passes exist; b = the robot's motion-direction offset (joint friction × compliance: the wind-up downstream of the
              motor encoders follows the direction of motion, lab kv_first 2026-10-11: pass normal offsets alternate ±0.17 mm with the
              direction at 30 and 60 mm/s), a FIXED effect times the leg's motion sign d_ℓ (+1 / −1 by its along-track velocity, 0 for a
              static leg — a dwell) — its zero is the midpoint of the two directions: the delivered positions are the static frame;
              b is in the model only where the data identify it (both directions among the legs that inform it: rank of the
              fixed-effect design, not a count); η_leg ~ N(0, diag(σ_η,w², σ_η,n²)) per pass, one variance component per direction
              (REML)). The (w0, n0) parts come from the line, the u0 part from the ends (motion along the line is projected out of the
              line, §4.2).
  time        δ = dt − dt_calib: a profile stamped δ late sits v·δ further along its travel. Along the seam c_v·δ (∝ speed) and d·b_u
              (speed-independent) are collinear when all moving legs share one speed: b_u is a fixed effect where the rank allows,
              δ has the prior N(0, σ_δ²) whose width is the run's own timing (timing_prior: one sensor frame period and one pose
              sample period, each uniform) — no tuned constant; the data's share of δ's posterior precision says whether this run
              separates the two (report).
  ends        where the seam is: one groove segment per leg (exact max-sum change point of λ, §4.3), fused with δ, offsets, the
              Cauchy length prior and τ_e (bounded-influence score, §4.4) — planning and the region of the plate-end readings (fused
              there without the length prior). The ends themselves: the plate top edges from the measured points through the
              sensor's edge response (end_faces), inner of the two plates.
  sections    exact 3D projection of every profile's template points along û onto the station plane (§4.6), first-order covariance,
              Huber/REML station means; plate normals from the plate-top lines (§4.2).
  checks      §4.9 (deep analysis §3.4, R3 §4.3): model fit of every segment that feeds a done-test quantity (legs, station windows,
              end readings, end regions, station dims: a segment-level outlier indicator with its own posterior, EM-weighted) and the
              discrete hypotheses the quantities rest on (bevel side, 'this seam' at each station window) — model_checks().
  done        max_q σ_q / precision_q ≤ 1 (§4.7, D12; the job's 1-σ precision targets, user 2026-10-11 — the design tolerances only
              feed the measured-vs-design warnings) and no done-test quantity rests on a misfit segment or an unresolved hypothesis.

Decisions in this module (all in the spec's forms, one k): D10 line vs quadratic (BIC), D11 groove found (2·max Σλ > k²), the per-profile
groove weight π = logistic(λ) (soft), the exact argmax segment per leg (empty segment = 'no groove in this leg'), §4.9 segment fits /
misfit (2·ln O ≷ ±k²) and hypotheses resolved (2·ln odds > k²), D12 done test.
Numeric settings marked [R] are iteration caps / steps only (spec §7 [R]); Huber 1.345 and the BIC ln n are statistical constants.

Sign conventions (see INTERFACE_NOTES 2026-10-09 WP2): a reading ẽ (measured s of an end) = e + c_v·δ − c·o, where c_v = (v·m)/(m·û)
and c·o = (m·o)/(m·û) for the laser-plane normal m — the same offset sign as the line (§4.2: r_i + o − r(s*)). A reading of a leg
moving in direction d at speed v is moved along the seam by d·(v·δ − b_u) (c_u ≈ 1): the along-track direction term.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr
from scipy.stats import chi2, norm

from . import js_source
from .cross_sections import FACE_NAMES
from .g1_types import (DIM_NAMES, LINE_MODELS, EndEst, LegObs, SeamEstimate, SeamPrior, StationEst, Tolerances,
                       taught_frame)
from .specimen import GrooveSpec
from .validity import Z_VALID_MM_DEFAULT, dX_mm

__all__ = ["estimate", "fit_line", "seam_dims", "leg_change_points", "fuse_ends", "station_sections", "done_test", "plan_quantities",
           "plate_normals", "groove_evidence", "max_segment", "profile_predictor", "site_state", "plate_change_points", "LineModel",
           "end_at", "Q_NAMES", "model_checks", "segment_mixture", "segment_verdict", "design_deviations", "delivered_length", "plan_ends",
           "CLASS_ORDER"]

N_THETA_SE = 10            # len(THETA_NAMES)
HUBER_C = 1.345            # statistical constant (95 % Gaussian efficiency) [L§1]
MAD_SCALE = 1.4826         # statistical constant (MAD → σ for a Gaussian)
GN_PASSES = 2              # [R] Gauss–Newton passes for s* / the line per IRLS round
IRLS_ROUNDS = 12           # [R] cap
IRLS_TOL = 1e-8            # [R] parameter change [m] that ends the IRLS rounds
CAUCHY_ITERS = 50          # [R]
FD_STEP = 1e-5             # [R] finite-difference step on θ (mm / rad)
GOLDEN_ITERS = 32          # [R] vectorised golden-section steps for the station REML
REL_NULL = 1e-12           # [R] relative singular value below which a design direction is unidentified (as end_faces.REL_NULL)
VC_GRID = 30               # [R] log-grid points of a leg-offset variance component's posterior (as end_faces.SIG_GRID)
VC_SPAN = (-6.0, 9.0)      # [R] that grid's log range about the legs' own data σ (as end_faces.SIG_SPAN)
PRIOR_TAIL = 1e-9          # [R] a calibrated prior's tail probability the coarse grid reaches (as end_faces.PRIOR_TAIL)
SENSOR_COLUMNS_DEFAULT = 1408   # VC nano 3D-Z NumberOfPoints; LegObs.meta['n_points'] overrides (sensor fact, not a tuning value)

# observation classes (spec r2 §3.6a, §4.1): attitude classes dwell_S, dwell_E (one per site, meta.site), transit; the pass class is
# the gauge (β_pass ≡ 0). Every non-pass class carries β_c ~ N(0, σ_he² I) always (no gauge switching): before any pass the frame and
# L1's σ come from that prior. 'coarse'/'fine' are accepted as aliases of transit/pass.
CLASS_ALIASES = {"fine": "pass", "coarse": "transit"}
GAUGE_CLASS = "pass"
CLASS_ORDER = ("pass", "transit", "dwell_S", "dwell_E", "dwell")


def _lam_free(leg) -> np.ndarray:
    """The leg's groove-vs-alternative log BF WITHOUT a seam-position prior: WP1 stores it as meta['log_bf_free'] when its fit used a
    predicted seam position (Σ_R); otherwise log_bf is already free. WP2 adds its own location term on the fused line (no double
    counting) and D11 uses the free λ."""
    lf = (getattr(leg, "meta", None) or {}).get("log_bf_free")
    lb = np.asarray(leg.log_bf, dtype=float)
    if lf is not None:
        lf = np.asarray(lf, dtype=float)
        if lf.shape == lb.shape:
            return np.where(np.isfinite(lf), lf, lb)
    return lb


def _lam_seq(leg) -> np.ndarray:
    """The leg's λ WITH the seam-position prior of the estimate that existed before the leg (WP1 fit_profile with Σ_R: 'this seam at
    the predicted position' vs 'no groove, or a groove-like structure anywhere in the window') — a one-step-ahead predictive of a leg
    that was not in that estimate (Dawid 1984, prequential); NaN for the profiles fitted without a prediction."""
    lb = np.asarray(leg.log_bf, dtype=float)
    meta = getattr(leg, "meta", None) or {}
    used = meta.get("log_bf_seam_used")
    if meta.get("log_bf_kind") == "seam":
        return lb.copy()
    if used is not None and len(used) == lb.shape[0]:
        return np.where(np.asarray(used, dtype=bool), lb, np.nan)
    return np.full(lb.shape, np.nan)


def _canon_cls(c: str, meta: Optional[dict] = None) -> str:
    c = CLASS_ALIASES.get(c, c)
    if c == "dwell" and meta and str(meta.get("site", "")).upper() in ("S", "E"):
        return "dwell_" + str(meta["site"]).upper()
    return c


# done-test quantities (§4.7) — names used in ratios / binding / unresolved
Q_NAMES = ("root_lateral", "root_normal", "end_S", "end_E", "gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm")
_THETA_IX = {n: i for i, n in enumerate(("x_R", "z_R", "alpha", "G", "beta", "rf", "h_L", "h_R", "gamma_L", "gamma_R"))}


# ====================================================================================================================
# small helpers
# ====================================================================================================================
def _unit(v, axis=-1):
    v = np.asarray(v, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):       # rows of non-converged profiles stay NaN (weight 0 downstream)
        return v / np.linalg.norm(v, axis=axis, keepdims=True)


def _logistic(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _inv2(V):
    """(…,2,2) inverse and log-determinant (analytic)."""
    a, b, c, d = V[..., 0, 0], V[..., 0, 1], V[..., 1, 0], V[..., 1, 1]
    det = a * d - b * c
    inv = np.empty_like(V)
    inv[..., 0, 0] = d / det
    inv[..., 0, 1] = -b / det
    inv[..., 1, 0] = -c / det
    inv[..., 1, 1] = a / det
    return inv, np.log(det)


def _rank(M: np.ndarray) -> int:
    """Numerical rank of a design (columns scaled to unit norm first: fixed effects in m, rad and s differ by orders of magnitude)."""
    M = np.asarray(M, dtype=float)
    if not M.size:
        return 0
    nrm = np.linalg.norm(M, axis=0)
    M = M[:, nrm > 0] / nrm[nrm > 0]
    if not M.size:
        return 0
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv > REL_NULL * max(float(sv[0]), 1e-300)))


def _adds_rank(F: np.ndarray, col: np.ndarray) -> bool:
    """True when the column is not in the span of the fixed-effect design F (rank decision, no count threshold)."""
    col = np.asarray(col, dtype=float).reshape(-1, 1)
    if not np.any(col != 0):
        return False
    F = np.asarray(F, dtype=float).reshape(col.shape[0], -1)
    return _rank(np.c_[F, col]) > _rank(F)


class _NormalInverse:
    """Inverse of a symmetric positive semi-definite normal (information) matrix N on the subspace the data identify. N is a sum of
    PSD terms (weighted design + priors), so its only non-positive eigenvalues are directions without information: Jacobi-scaled
    eigendecomposition (unit-free: fixed effects in m, s and rad differ by orders of magnitude), a direction with relative eigenvalue
    ≤ REL_NULL is unidentified (the rank rule of _rank / end_faces). N⁻ = F·Fᵀ, F = D·V_id·Λ_id^(-1/2) (D the Jacobi scaling): the
    Moore–Penrose inverse in the scaled coordinates, a generalised inverse of N (N·N⁻·N = N), PSD by construction. A linear
    functional a has a finite posterior variance only when it lies in the identified subspace (a ∈ range N — not _adds_rank(N, a));
    its variance ‖Fᵀa‖² ≥ 0 is then the same for every generalised inverse. With full rank this is the plain inverse."""

    def __init__(self, N: np.ndarray):
        N = np.asarray(N, dtype=float)
        self.N = N
        d = np.diag(N) if N.size else np.zeros(0)
        sc = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 1.0)
        Ns = sc[:, None] * N * sc[None, :]
        lam, V = np.linalg.eigh(0.5 * (Ns + Ns.T)) if N.size else (np.zeros(0), np.zeros((0, 0)))
        top = float(lam[-1]) if lam.size else 0.0
        ok = lam > REL_NULL * top if top > 0 else np.zeros(lam.shape, dtype=bool)
        self.F = (V[:, ok] * sc[:, None]) / np.sqrt(lam[ok])[None, :]
        self.full = bool(np.all(ok))
        self.pinv = self.F @ self.F.T

    def identified(self, a: np.ndarray) -> bool:
        return self.full or not _adds_rank(self.N, np.asarray(a, dtype=float))

    def var(self, a: np.ndarray) -> float:
        """Posterior variance of a·x (∞ for a functional the data do not identify)."""
        a = np.asarray(a, dtype=float)
        if not self.identified(a):
            return math.inf
        return float(np.sum((self.F.T @ a) ** 2))

    def cov(self) -> np.ndarray:
        """Covariance of the unknowns: N⁺ between identified unknowns, ∞ in the rows / columns of an unknown the data do not
        identify on its own."""
        C = self.pinv.copy()
        if not self.full:
            bad = [j for j in range(C.shape[0]) if not self.identified(np.eye(C.shape[0])[j])]
            C[bad, :] = np.inf
            C[:, bad] = np.inf
        return C

    def step(self, rhs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Solution of N·x = rhs nearest x (in the scaled coordinates): the Newton step in the identified subspace — a direction
        without information keeps the value it has (the start / convention)."""
        return x + self.pinv @ (rhs - self.N @ x)


def motion_sign(leg, u0: np.ndarray) -> int:
    """The leg's motion direction along the seam: +1 / −1 = sign of the median along-track velocity of its profiles, 0 = a static leg
    (a dwell grab: one pose, v ≡ 0). The robot's direction offset b multiplies it."""
    v = np.asarray(getattr(getattr(leg, "src", leg), "v_opt_mps", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    if not v.size:
        return 0
    return int(np.sign(np.median(v @ np.asarray(u0, dtype=float))))


def _pose_period_s(leg) -> Optional[float]:
    """The pose sample period of a leg's recording: meta 'pose_period_s' when the leg states it, else the robot samples recorded next
    to its profiles (raw.npz robot_stamp_ns); None = poses given per profile (no separate pose sampling, e.g. synthetic legs)."""
    lg = getattr(leg, "src", leg)
    v = (getattr(lg, "meta", None) or {}).get("pose_period_s")
    if v is not None and math.isfinite(float(v)) and float(v) > 0:
        return float(v)
    rp = getattr(lg, "recon_path", None)
    if not rp:
        return None
    import os
    raw = os.path.join(os.path.dirname(str(rp)), "raw.npz")
    if not os.path.isfile(raw):
        return None
    try:
        with np.load(raw, allow_pickle=False) as z:
            if "robot_stamp_ns" not in z.files:
                return None
            ts = np.asarray(z["robot_stamp_ns"], dtype=float) * 1e-9
    except (OSError, ValueError):
        return None
    d = np.diff(np.sort(ts))
    d = d[d > 0]
    return float(np.median(d)) if d.size else None


_JS_CACHE: dict = {}      # raw.npz path → ((mtime_ns, size), evidence): the worker keeps a leg's joint-state evidence between estimates


def _leg_js_evidence(leg) -> Optional[dict]:
    """The leg's joint-state evidence (js_source.leg_evidence): meta 'joint_state_evidence' when the worker stored it with the leg,
    else from the raw.npz next to its profiles (robot_* samples and motions_json); None = the leg has no joint-state samples
    (synthetic legs: poses given per profile)."""
    lg = getattr(leg, "src", leg)
    ev = (getattr(lg, "meta", None) or {}).get("joint_state_evidence")
    if ev:
        return ev
    rp = getattr(lg, "recon_path", None)
    if not rp:
        return None
    import os
    raw = os.path.join(os.path.dirname(str(rp)), "raw.npz")
    try:
        st = os.stat(raw)
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    hit = _JS_CACHE.get(raw)
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        with np.load(raw, allow_pickle=False) as z:
            if "robot_joint_positions_rad" not in z.files:
                return None
            arrays = {k: z[k] for k in ("robot_joint_positions_rad", "robot_t_rx_ns", "motions_json", "robot_joint_names") if k in z.files}
    except (OSError, ValueError):
        return None
    ev = js_source.raw_evidence(arrays)
    _JS_CACHE[raw] = (key, ev)
    return ev


def timing_prior(legs, u0: np.ndarray) -> dict:
    """Prior of the residual time offset δ = dt − dt_calib from the run's own timing (user 2026-10-11: no tuned constant). Where in its
    exposure / readout cycle a profile is stamped is unknown within one sensor frame period T_f, and where in the controller's cycle the
    pose it is paired with was sampled within one pose sample period T_r: each uniform over its period, so the quantisation
    q ~ N(0, σ_q²) with σ_q² = (T_f² + T_r²)/12. T_f = the median profile interval of the moving legs (all legs when none moves), T_r =
    the median pose sample period of the same legs that record one (0 when the poses are given per profile). Both from the legs whose
    profiles are paired with sampled poses: δ acts through the motion (c_v·δ); a static grab pairs its frames with its one still pose,
    and the robot samples stored next to it are no pose sampling of its frames (the virtual arm streams no samples while still: two
    stamps a dwell apart, 0.55–1.2 s, took the median T_r of a 5-leg + 1-pass run to 0.83 s — σ_δ 240 ms, 9.6 mm on the ends at
    40 mm/s, test_sim_teach1g m1).
    The joint-state source (user 2026-10-12, js_source): the legs' joint-state evidence (ALL legs — dwells hold the rest data) gives
    P(reference echo); the δ prior is the cut mixture js_source.DeltaPrior(σ_q, P) — measured: N(0, σ_q²) as before; echo: δ = τ_echo
    + q, τ_echo ≥ 0 unknown. Legs without joint-state samples (poses given per profile) carry no echo to consider: P = 0.
    → {sigma_s (σ_q), frame_period_s, pose_period_s, joint_state (pooled evidence or None), p_ref}."""
    mv = [lg for lg in legs if motion_sign(lg, u0) != 0] or list(legs)
    tf = []
    for lg in mv:
        t = np.asarray(getattr(getattr(lg, "src", lg), "t_acq_s", np.zeros(0)), dtype=float)
        dt_ = np.diff(t)
        dt_ = dt_[dt_ > 0]
        if dt_.size:
            tf.append(float(np.median(dt_)))
    T_f = float(np.median(tf)) if tf else 0.0
    tr = [p for p in (_pose_period_s(lg) for lg in mv) if p is not None]
    T_r = float(np.median(tr)) if tr else 0.0
    evs, names = [], []
    for lg in legs:
        e = _leg_js_evidence(lg)
        if e and int(e.get("n", 0)) >= 2:
            evs.append(e)
            names.append(str(getattr(getattr(lg, "src", lg), "name", "?")))
    js = js_source.pool(evs, names=names) if evs else None
    return {"sigma_s": math.sqrt((T_f * T_f + T_r * T_r) / 12.0), "frame_period_s": T_f, "pose_period_s": T_r if tr else None,
            "joint_state": js, "p_ref": float(js["p_ref"]) if js else 0.0}


def max_segment(lam: np.ndarray) -> tuple:
    """Exact argmax_{a ≤ b} Σ_{i=a..b} λ_i (Kadane, O(n)), the empty segment (sum 0) included as 'no groove'.
    Returns (best_sum, a, b) with a = b = -1 for the empty segment."""
    lam = np.asarray(lam, dtype=float)
    best, ba, bb = 0.0, -1, -1
    cur, ca = 0.0, 0
    for i, x in enumerate(lam):
        if cur <= 0.0:
            cur, ca = x, i
        else:
            cur += x
        if cur > best:
            best, ba, bb = cur, ca, i
    return best, ba, bb


# ====================================================================================================================
# §4.9 segment checks — decision forms of the deep analysis §3.4 (the same k everywhere): model fit and hypothesis odds
# ====================================================================================================================
EM_ITERS = 50                  # [R] cap of the segment-indicator EM (ε, γ) and of the estimate's reweighting rounds below
EM_TOL = 1e-9                  # [R] γ change that ends the indicator EM
REWEIGHT_ROUNDS = 2            # [R] cap: estimate rounds with the segment weights (EM M-steps on the fused model)
REWEIGHT_TOL = 1e-3            # [R] largest change of a profile's segment weight that still asks for another round


def _equiv_z(q, d: int):
    """Standardised posterior-predictive residual of a d-dimensional Gaussian residual with Mahalanobis q: the normal deviate with the
    same two-sided tail probability (probability integral transform), z = Φ⁻¹(1 − ½·P(χ²_d > q)); d = 1 → |x|/σ."""
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    if d == 1:
        return np.sqrt(q)
    with np.errstate(all="ignore"):
        z = norm.isf(0.5 * chi2.sf(q, d))
    return np.where(np.isfinite(z), z, np.sqrt(q))                 # beyond double precision: the asymptotic deviate


def segment_mixture(lnf, ln_extent, gam0=None, lnf_peak=None) -> tuple:
    """Segment-level outlier indicators of one family of segments (a mixture with its own posterior — Box & Tiao 1968; EM as Aitkin &
    Wilson 1980): H_fit = the segment's aggregated observation follows the fused model's posterior predictive (log density lnf_j),
    H_out = it lies anywhere in its physical extent (uniform, log density −ln_extent_j: the sensor window, the leg's span, the range of
    the quantity). Prior P(H_out) = ε ~ Beta(1, 1), estimated with the family: ε = (Σγ + 1)/(n + 2) (posterior mean, the rule of
    succession) — the multiplicity of n segments enters through ε (Scott & Berger 2010), not through a new threshold.
    lnf_peak_j = the predictive's log density at its own centre: a segment whose predictive is no narrower than its extent
    (lnf_peak + ln_extent ≤ 0 — the likelihood ratio cannot exceed 1 anywhere, e.g. an end that nothing else predicts) is not
    testable: no indicator (γ = 0, ln O = NaN) and it does not count in ε.
    Returns (ln O_j = posterior log odds 'segment j fits', γ_j = posterior P(H_out), ε)."""
    lnf = np.asarray(lnf, dtype=float).reshape(-1)
    ext = np.broadcast_to(np.asarray(ln_extent, dtype=float), lnf.shape)
    n_all = lnf.size
    lo = np.full(n_all, np.nan)
    g = np.zeros(n_all)
    if n_all == 0:
        return lo, g, float("nan")
    on = np.isfinite(lnf) & np.isfinite(ext)
    if lnf_peak is not None:
        on &= (np.broadcast_to(np.asarray(lnf_peak, dtype=float), lnf.shape) + ext) > 0.0
    n = int(np.count_nonzero(on))
    if n == 0:
        return lo, g, float("nan")
    gg = np.zeros(n) if gam0 is None else np.nan_to_num(np.asarray(gam0, dtype=float).reshape(-1)[on])
    lo_on = np.zeros(n)
    eps = 0.5
    for _ in range(EM_ITERS):
        eps = (float(np.sum(gg)) + 1.0) / (n + 2.0)
        lo_on = math.log1p(-eps) - math.log(eps) + lnf[on] + ext[on]
        g_new = _logistic(-lo_on)
        done_ = float(np.max(np.abs(g_new - gg))) < EM_TOL
        gg = g_new
        if done_:
            break
    lo[on], g[on] = lo_on, gg
    return lo, g, eps


def segment_verdict(two_ln_odds: float, k: float) -> str:
    """Decision form 1 on 'this segment fits' (k² on 2·ln odds): 'fits' / 'misfit' (2 ln O < −k²: the segment decisively does not
    fit) / 'ambiguous'; NaN = 'untestable' (segment_mixture)."""
    t = float(two_ln_odds)
    if math.isnan(t):
        return "untestable"
    if not math.isfinite(t):
        return "misfit" if t < 0 else "fits"
    return "fits" if t > k * k else ("misfit" if t < -k * k else "ambiguous")


# ====================================================================================================================
# line model (taught frame + polynomial a(s), b(s))
# ====================================================================================================================
class LineModel:
    """r(s) = S0 + s·u0 + a(s)·w0 + b(s)·n0 (same algebra as SeamEstimate.r)."""

    def __init__(self, frame0: dict, params: np.ndarray, cov: Optional[np.ndarray] = None):
        self.f = {k: np.asarray(frame0[k], dtype=float).reshape(3) for k in ("S0", "u0", "w0", "n0")}
        self.P = np.asarray(params, dtype=float).reshape(2, -1)
        self.d = self.P.shape[1] - 1
        self.cov = None if cov is None else np.asarray(cov, dtype=float)

    @property
    def model(self) -> str:
        return {1: "line", 2: "quadratic"}[self.d]

    def g(self, s):
        s = np.asarray(s, dtype=float)
        return s[..., None] ** np.arange(self.d + 1)

    def dg(self, s):
        s = np.asarray(s, dtype=float)
        k = np.arange(self.d + 1)
        return np.where(k >= 1, k * s[..., None] ** np.maximum(k - 1, 0), 0.0)

    def r(self, s):
        s = np.asarray(s, dtype=float)
        g = self.g(s)
        return (self.f["S0"] + s[..., None] * self.f["u0"] + (g @ self.P[0])[..., None] * self.f["w0"]
                + (g @ self.P[1])[..., None] * self.f["n0"])

    def t(self, s):
        """dr/ds (not unit)."""
        dg = self.dg(s)
        return self.f["u0"] + (dg @ self.P[0])[..., None] * self.f["w0"] + (dg @ self.P[1])[..., None] * self.f["n0"]

    def tangent(self, s):
        return _unit(self.t(s))

    def s_star(self, q, m, s0=None, iters: int = 3):
        """s where the plane (point q, normal m) cuts the line: m·(r(s) − q) = 0 (Newton; exact in one step for a line)."""
        q = np.asarray(q, dtype=float)
        m = np.asarray(m, dtype=float)
        if s0 is None:
            s = np.einsum("...i,...i->...", m, q - self.f["S0"]) / np.einsum("...i,i->...", m, self.f["u0"])
        else:
            s = np.asarray(s0, dtype=float).copy()
        for _ in range(iters):
            f = np.einsum("...i,...i->...", m, self.r(s) - q)
            fp = np.einsum("...i,...i->...", m, self.t(s))
            s = s - f / fp
        return s

    def sigma_wn(self, s):
        """(σ along w0, σ along n0) of r(s) from the line covariance [m]."""
        if self.cov is None:
            z = np.zeros(np.shape(s))
            return z, z
        g = self.g(s)
        m = self.d + 1
        va = np.einsum("...i,ij,...j->...", g, self.cov[:m, :m], g)
        vb = np.einsum("...i,ij,...j->...", g, self.cov[m:, m:], g)
        return np.sqrt(np.maximum(va, 0.0)), np.sqrt(np.maximum(vb, 0.0))

    def sigma_dir(self, s):
        """1-σ of the direction (rad) at s: lateral slope (yaw about n0) and normal slope (pitch)."""
        if self.cov is None:
            z = np.zeros(np.shape(s))
            return z, z
        dg = self.dg(s)
        m = self.d + 1
        va = np.einsum("...i,ij,...j->...", dg, self.cov[:m, :m], dg)
        vb = np.einsum("...i,ij,...j->...", dg, self.cov[m:, m:], dg)
        return np.sqrt(np.maximum(va, 0.0)), np.sqrt(np.maximum(vb, 0.0))

    @classmethod
    def from_estimate(cls, est: SeamEstimate) -> "LineModel":
        return cls(est.frame0, est.line_params, est.line_cov)


# ====================================================================================================================
# per-profile template geometry — groove_model (WP1) is the one owner of the θ convention (own-plate frames, mirroring, shear);
# this module only maps its vectorised geometry (groove_batch.geometry_b) to base and projects it along û
# ====================================================================================================================
_VSEC = {"single_bevel": ("top_corner_L", "root_face_top_L", "root_edge_L", "root_edge_R", "top_corner_R"),
         "single_v": ("top_corner_L", "root_face_top_L", "root_edge_L", "root_edge_R", "root_face_top_R", "top_corner_R"),
         "square": ("top_corner_L", "root_edge_L", "root_edge_R", "top_corner_R")}


def _leg_geometry(view, spec) -> dict:
    """One leg: template vertices / face anchors and directions in base with their θ-Jacobians (exact, WP1 geometry_b)."""
    from .groove_batch import geometry_b
    from .groove_model import Template
    shear = float(((view.meta or {}).get("prior") or {}).get("shear", 0.0) or 0.0)
    tpl = Template(spec, int(view.bevel_x_sign), shear=shear)
    TH = np.asarray(view.theta, dtype=float)
    with np.errstate(all="ignore"):
        g = geometry_b(tpl, TH)
    T = np.asarray(view.T_base_opt, dtype=float)
    X, Z, o = T[:, :3, 0], T[:, :3, 2], T[:, :3, 3]

    def pts(P2):                                         # (N,k,2) mm → (N,k,3) m
        return o[:, None, :] + 1e-3 * (P2[..., 0:1] * X[:, None, :] + P2[..., 1:2] * Z[:, None, :])

    def dirs(D2):                                        # (N,k,2) → (N,k,3)
        return D2[..., 0:1] * X[:, None, :] + D2[..., 1:2] * Z[:, None, :]

    def jac(dP2, scale):                                 # (N,k,2,10) → (N,k,3,10)
        return scale * (dP2[:, :, 0:1, :] * X[:, None, :, None] + dP2[:, :, 1:2, :] * Z[:, None, :, None])

    al = TH[:, 2]
    return {"names_v": tpl.vertex_names, "names_f": g["names"], "V": pts(g["V"]), "dV": jac(g["dV"], 1e-3),
            "A": pts(g["A"]), "dA": jac(g["dA"], 1e-3), "U": dirs(g["u"]), "dU": jac(g["du"], 1.0),
            "E1": np.cos(al)[:, None] * X + np.sin(al)[:, None] * Z, "dE1": -np.sin(al)[:, None] * X + np.cos(al)[:, None] * Z}


def _section_basis(u: np.ndarray, n: np.ndarray) -> tuple:
    """(û, n̂ ⟂ û, ŵ = n̂ × û) — the station frame of §9."""
    u = _unit(u)
    n = _unit(n - float(n @ u) * u)
    w = np.cross(n, u)
    return u, n, w / np.linalg.norm(w)


def _section_dims_from_z(z: np.ndarray, kind: str) -> np.ndarray:
    """Section quantities from section coordinates z (…, 2·nv + 4) [mm]: vertices (_VSEC order), backing direction e1, bevel-plate
    top direction → G, D, β [deg], O, H, rf (DIM_NAMES order). w̃ = backing direction (+w), ñ ⟂ toward +n; β vs the bevel plate's
    own normal (WP1 frames); single_v / square: D on the −w side, H = (+w top) − (−w top) (WP4)."""
    nv = len(_VSEC[kind])
    P = {nm: z[..., 2 * i:2 * i + 2] for i, nm in enumerate(_VSEC[kind])}
    e1 = z[..., 2 * nv:2 * nv + 2]
    uL = z[..., 2 * nv + 2:2 * nv + 4]

    def dot(a, b):
        return np.sum(a * b, axis=-1)

    wt = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
    wt = wt * np.where(wt[..., :1] >= 0, 1.0, -1.0)
    nt = np.stack([-wt[..., 1], wt[..., 0]], axis=-1)
    CL, RL, RR, CR = P["top_corner_L"], P["root_edge_L"], P["root_edge_R"], P["top_corner_R"]
    FL = P.get("root_face_top_L", RL)
    G = np.linalg.norm(RR - RL, axis=-1)
    D = dot(CL - RL, nt)
    ul = uL / np.linalg.norm(uL, axis=-1, keepdims=True)
    nl = np.stack([-ul[..., 1], ul[..., 0]], axis=-1)
    nl = nl * np.where(dot(nl, nt)[..., None] >= 0, 1.0, -1.0)
    bev = CL - FL
    beta = np.degrees(np.arctan2(np.abs(dot(bev, ul)), dot(bev, nl)))
    O = np.abs(dot(CL - CR, wt))
    H = dot(CR - CL, nt)
    rf = np.linalg.norm(FL - RL, axis=-1)
    if kind != "single_bevel":
        L_plus = CL[..., 0] > CR[..., 0]
        D = np.where(L_plus, dot(CR - RR, nt), D)
        H = np.where(L_plus, -H, H)
    return np.stack([G, D, beta, O, H, rf], axis=-1)


def _section_quantities(geo: dict, sel: np.ndarray, cov: np.ndarray, w: np.ndarray, n: np.ndarray, kind: str) -> tuple:
    """Exact section quantities (§4.6) of the selected profiles and their first-order (GUM) variances: differences of template
    points in base projected along û onto the section (coordinates (·ŵ, ·n̂)), Jacobian = ∂F/∂z (finite differences on the
    section coordinates [R]) · ∂z/∂θ (analytic from WP1's dV, dU)."""
    vi = [geo["names_v"].index(nm) for nm in _VSEC[kind]]
    fL = geo["names_f"].index("top_L")
    Wn = np.stack([w, n], axis=0) * 1e3                                       # (2,3) → mm
    V = geo["V"][sel][:, vi]                                                  # (m,nv,3)
    dV = geo["dV"][sel][:, vi]                                                # (m,nv,3,10)
    zV = np.einsum("ki,mvi->mvk", Wn, V - V[:, :1])                           # relative to the first vertex (translation-free)
    dzV = np.einsum("ki,mvip->mvkp", Wn, dV - dV[:, :1])
    E1, dE1 = geo["E1"][sel], geo["dE1"][sel]
    U, dU = geo["U"][sel][:, fL], geo["dU"][sel][:, fL]
    W3 = Wn * 1e-3
    zE = E1 @ W3.T
    dzE = np.zeros(zE.shape + (N_THETA_SE,))
    dzE[..., 2] = dE1 @ W3.T
    zU = U @ W3.T
    dzU = np.einsum("ki,mip->mkp", W3, dU)
    z = np.concatenate([zV.reshape(len(zV), -1), zE, zU], axis=1)
    dz = np.concatenate([dzV.reshape(len(zV), -1, N_THETA_SE), dzE, dzU], axis=1)  # (m, Z, 10)
    q0 = _section_dims_from_z(z, kind)
    Jz = np.zeros(q0.shape + (z.shape[1],))
    for j in range(z.shape[1]):
        dzj = np.zeros(z.shape[1])
        dzj[j] = FD_STEP
        Jz[:, :, j] = (_section_dims_from_z(z + dzj, kind) - _section_dims_from_z(z - dzj, kind)) / (2 * FD_STEP)
    Jt = np.einsum("mqz,mzp->mqp", Jz, dz)
    var = np.einsum("mqp,mpr,mqr->mq", Jt, cov, Jt)
    return q0, var


# ====================================================================================================================
# gather all profiles
# ====================================================================================================================
class _LegView:
    """A leg as the estimator sees it: canonical class (dwell → dwell_S / dwell_E by meta.site); a dwell normally has N = 1 (WP1
    fit_dwell on the column-median profile); if frames are given they are collapsed to one observation (median θ, median cov,
    median λ — static speckle does not average over frames)."""

    def __init__(self, lg):
        self.src = lg
        self.name, self.direction, self.bevel_x_sign, self.meta = lg.name, lg.direction, lg.bevel_x_sign, lg.meta
        self.cls = _canon_cls(lg.cls, lg.meta)
        arr = {k: getattr(lg, k) for k in ("t_acq_s", "T_base_opt", "v_opt_mps", "theta", "cov", "log_bf", "converged")}
        if self.cls.startswith("dwell") and lg.n > 1:
            ok = (np.all(np.isfinite(lg.theta), axis=1) & np.asarray(lg.converged, bool)
                  & np.all(np.isfinite(lg.cov.reshape(lg.n, -1)), axis=1))
            if ok.sum() >= 1:
                ix = np.flatnonzero(ok)
                mid = ix[len(ix) // 2]
                lam = np.where(np.isfinite(lg.log_bf), lg.log_bf, 0.0)
                # static speckle is correlated between frames (Dorsch 1994; WP1 fit_dwell): frames do not average it away
                arr = {"t_acq_s": lg.t_acq_s[mid:mid + 1], "T_base_opt": lg.T_base_opt[mid:mid + 1],
                       "v_opt_mps": np.zeros((1, 3)), "theta": np.median(lg.theta[ok], axis=0)[None],
                       "cov": np.median(lg.cov[ok], axis=0)[None], "log_bf": np.array([float(np.median(lam[ok]))]),
                       "converged": np.array([True])}
        for k, v in arr.items():
            setattr(self, k, v)
        self.n = int(self.theta.shape[0])
        self.n_frames = int(lg.n)


class _Profiles:
    """Struct of arrays over every profile of every leg (base frame, m)."""

    def __init__(self, legs: list, frame0: dict):
        self.legs = [_LegView(lg) for lg in legs if lg.n > 0]
        self.f0 = frame0
        T = np.concatenate([lg.T_base_opt for lg in self.legs]) if self.legs else np.zeros((0, 4, 4))
        th = np.concatenate([lg.theta for lg in self.legs]) if self.legs else np.zeros((0, 10))
        cv = np.concatenate([lg.cov for lg in self.legs]) if self.legs else np.zeros((0, 10, 10))
        self.T, self.theta, self.cov = T, th, cv
        self.o, self.ex, self.m, self.ez = T[:, :3, 3], T[:, :3, 0], T[:, :3, 1], T[:, :3, 2]
        self.lam = np.concatenate([_lam_free(lg) for lg in self.legs]) if self.legs else np.zeros(0)
        # §4.9 the one-step-ahead (prequential) evidence: WP1's λ against the estimate that existed BEFORE this leg (its seam-position
        # prior), where WP1 used one; NaN elsewhere
        self.lam_seq = np.concatenate([_lam_seq(lg) for lg in self.legs]) if self.legs else np.zeros(0)
        self.v = np.concatenate([lg.v_opt_mps for lg in self.legs]) if self.legs else np.zeros((0, 3))
        self.leg = np.concatenate([np.full(lg.n, i) for i, lg in enumerate(self.legs)]) if self.legs else np.zeros(0, int)
        # the legs' motion directions (+1 / −1 along u0, 0 static) and per profile — the direction offset b multiplies them
        self.dir_leg = np.array([motion_sign(lg, frame0["u0"]) for lg in self.legs], dtype=float)
        self.dsign = self.dir_leg[self.leg] if self.legs else np.zeros(0)
        self.cls = np.array([self.legs[i].cls for i in self.leg]) if self.legs else np.zeros(0, dtype="<U6")
        self.bxs = np.concatenate([np.full(lg.n, lg.bevel_x_sign) for lg in self.legs]) if self.legs else np.zeros(0)
        cov2 = cv[:, :2, :2]
        fin = (np.all(np.isfinite(th), axis=1) & np.all(np.isfinite(cv.reshape(len(cv), -1)), axis=1)
               & np.isfinite(self.lam))
        conv = np.concatenate([lg.converged for lg in self.legs]) if self.legs else np.zeros(0, bool)
        with np.errstate(invalid="ignore"):
            det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] * cov2[:, 1, 0]
        self.ok = fin & conv & (det > 0)
        self.lam = np.where(np.isfinite(self.lam), self.lam, 0.0)        # no evidence → neutral (missing at random)
        # root point (base) and its in-plane covariance (m²)
        self.r = self.o + (th[:, 0, None] * self.ex + th[:, 1, None] * self.ez) * 1e-3
        self.S2 = np.where(self.ok[:, None, None], cov2, np.eye(2)) * 1e-6
        npts = np.concatenate([np.full(lg.n, float(lg.meta.get("n_points", SENSOR_COLUMNS_DEFAULT))) for lg in self.legs]) \
            if self.legs else np.zeros(0)
        self.fov_mm = npts * dX_mm(np.where(np.isfinite(th[:, 1]), th[:, 1], 0.0))
        zw = np.concatenate([np.full(lg.n, float(np.diff(lg.meta.get("z_window_mm", Z_VALID_MM_DEFAULT))[0]))
                             for lg in self.legs]) if self.legs else np.zeros(0)
        self.win_mm2 = self.fov_mm * zw                                     # sensor measurement window (x FOV × calibrated z)
        self.r = np.where(self.ok[:, None], self.r, self.o)

    @property
    def N(self) -> int:
        return int(self.T.shape[0])

    def idx(self, i: int) -> np.ndarray:
        return np.flatnonzero(self.leg == i)

    def geometry(self, spec) -> dict:
        """WP1 template geometry of every profile in base (cached): V, dV, A, U, dU, E1, dE1 (+ names)."""
        if getattr(self, "_geo", None) is None:
            parts = [_leg_geometry(lg, spec) for lg in self.legs]
            geo = {"names_v": parts[0]["names_v"], "names_f": parts[0]["names_f"]} if parts else {}
            for k in ("V", "dV", "A", "dA", "U", "dU", "E1", "dE1"):
                geo[k] = np.concatenate([p_[k] for p_ in parts]) if parts else np.zeros((0,))
            # L plate (sensor-x side bevel_x_sign) on +w? per profile, from its own corners (taught frame w0)
            iL, iR = geo["names_v"].index("top_corner_L"), geo["names_v"].index("top_corner_R")
            geo["L_left"] = ((geo["V"][:, iL] - geo["V"][:, iR]) @ self.f0["w0"]) > 0
            self._geo = geo
        return self._geo


def _gauge(legs) -> Optional[str]:
    """The gauge class (β ≡ 0): the passes, when present; before any pass no class is the gauge (all carry the σ_he prior)."""
    return GAUGE_CLASS if any(_canon_cls(lg.cls, lg.meta) == GAUGE_CLASS for lg in legs) else None


def _classes_present(legs) -> list:
    present = {_canon_cls(lg.cls, lg.meta) for lg in legs}
    return [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))


def _eta_legs(legs) -> list:
    """Legs with their own offset η_ℓ ~ N(0, σ_rep² I): the passes, once ≥ 2 exist (one pass is the gauge itself; §4.1 r2)."""
    passes = [lg.name for lg in legs if _canon_cls(lg.cls, lg.meta) == GAUGE_CLASS]
    return passes if len(passes) >= 2 else []


# ====================================================================================================================
# §4.2 line — covariance-weighted robust 3D line
# ====================================================================================================================
def _line_layout(prof: _Profiles, d: int, dir_ok: tuple = (False, False)):
    """Parameter layout: [a_0..a_d, b_0..b_d, (β_w, β_n) per non-gauge class, the direction offset (b_w, b_n) — each component only
    where dir_ok says the data identify it —, (η_w, η_n) per leg of a class with ≥ 2 legs]. dir_col = {component (0 = w, 1 = n): col}."""
    gauge = _gauge(prof.legs)
    cols, names = 2 * (d + 1), []
    cls_col, leg_col, dir_col = {}, {}, {}
    for c in _classes_present(prof.legs):
        if c != gauge:
            cls_col[c] = cols
            names.append(("class", c))
            cols += 2
    for c_ in (0, 1):
        if dir_ok[c_]:
            dir_col[c_] = cols
            names.append(("direction", "wn"[c_]))
            cols += 1
    eta = set(_eta_legs(prof.legs))
    for i, lg in enumerate(prof.legs):
        if lg.name in eta:
            leg_col[i] = cols
            names.append(("leg", lg.name))
            cols += 2
    return cols, cls_col, leg_col, dir_col, names


def _offsets(prof: _Profiles, p: np.ndarray, cls_col: dict, leg_col: dict, dir_col: Optional[dict] = None) -> np.ndarray:
    """Per-profile 3D offset (w0, n0 parts) from the parameter vector: class + motion direction × b + leg."""
    w0, n0 = prof.f0["w0"], prof.f0["n0"]
    off = np.zeros((prof.N, 3))
    for c, j in cls_col.items():
        sel = prof.cls == c
        off[sel] += p[j] * w0 + p[j + 1] * n0
    for c_, j in (dir_col or {}).items():
        off += (prof.dsign * p[j])[:, None] * (w0 if c_ == 0 else n0)[None, :]
    for i, j in leg_col.items():
        sel = prof.leg == i
        off[sel] += p[j] * w0 + p[j + 1] * n0
    return off


def _line_residuals(prof, p, d, cls_col, leg_col, s0, sel, dir_col: Optional[dict] = None):
    """Residuals e (n,2) [m], Jacobian J (n,2,P), s* (n,) for the selected profiles (exact oblique projection along the line)."""
    f0 = prof.f0
    L = LineModel(f0, p[:2 * (d + 1)].reshape(2, d + 1))
    off = _offsets(prof, p, cls_col, leg_col, dir_col)[sel]
    q = prof.r[sel] + off
    m = prof.m[sel]
    s = L.s_star(q, m, s0=s0, iters=GN_PASSES)
    E = np.stack([prof.ex[sel], prof.ez[sel]], axis=1)                     # (n,2,3)
    t = L.t(s)
    mt = np.einsum("ni,ni->n", m, t)
    Et = np.einsum("nki,ni->nk", E, t)
    EQ = E - Et[:, :, None] * m[:, None, :] / mt[:, None, None]             # E·(I − t mᵀ/(m·t))
    dvec = q - L.r(s)
    e = np.einsum("nki,ni->nk", EQ, dvec)
    P = p.size
    J = np.zeros((e.shape[0], 2, P))
    g = L.g(s)
    EQw = EQ @ f0["w0"]
    EQn = EQ @ f0["n0"]
    for k in range(d + 1):
        J[:, :, k] = -EQw * g[:, k, None]
        J[:, :, d + 1 + k] = -EQn * g[:, k, None]
    cl = prof.cls[sel]
    lg = prof.leg[sel]
    for c, j in cls_col.items():
        on = (cl == c)[:, None]
        J[:, :, j] = np.where(on, EQw, 0.0)
        J[:, :, j + 1] = np.where(on, EQn, 0.0)
    for i, j in leg_col.items():
        on = (lg == i)[:, None]
        J[:, :, j] = np.where(on, EQw, 0.0)
        J[:, :, j + 1] = np.where(on, EQn, 0.0)
    ds = prof.dsign[sel][:, None]
    for c_, j in (dir_col or {}).items():
        J[:, :, j] = ds * (EQw if c_ == 0 else EQn)
    return e, J, s


def _direction_identified(prof: _Profiles, d: int, sel: np.ndarray, pi: np.ndarray) -> tuple:
    """Which components of the direction offset b the line's data identify: the rank of the fixed-effect design (line polynomial,
    class offsets taken as free — their σ_he prior is not data —, b) of the profiles that inform the fit (weights √π), with vs.
    without the b column. The random leg offsets η are not part of it (b would always be in their span). One motion direction
    among the legs that inform it, or b equal to a class offset (a transit in one direction, dwells static) → not identified."""
    if not np.any(sel) or not np.any(prof.dsign[sel] != 0):
        return (False, False)
    P, cls_col, leg_col, dir_col, _ = _line_layout(prof, d, (True, True))
    _, J, _ = _line_residuals(prof, np.zeros(P), d, cls_col, {}, None, sel, dir_col)
    w = np.sqrt(np.maximum(np.asarray(pi, dtype=float)[sel], 0.0))
    R = (J * w[:, None, None]).reshape(-1, P)
    fixed = list(range(2 * (d + 1))) + [k for j in cls_col.values() for k in (j, j + 1)]
    F = R[:, fixed]
    return tuple(bool(_adds_rank(F, R[:, dir_col[c_]])) for c_ in (0, 1))


def _V_of(S2, cls_masks, log_sig):
    V = S2.copy()
    for ci, msk in enumerate(cls_masks):
        v = math.exp(2 * log_sig[ci])
        V[msk, 0, 0] += v
        V[msk, 1, 1] += v
    return V


def _eig2(S):
    """Analytic eigen-decomposition of symmetric (…,2,2): eigenvalues (…,2), eigenvectors as rows (…,2,2)."""
    a, b, d = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    m = 0.5 * (a + d)
    r = np.sqrt(np.maximum(0.25 * (a - d) ** 2 + b ** 2, 0.0))
    lam = np.stack([m + r, m - r], axis=-1)
    th = 0.5 * np.arctan2(2 * b, a - d)
    c, s_ = np.cos(th), np.sin(th)
    U = np.stack([np.stack([c, s_], axis=-1), np.stack([-s_, c], axis=-1)], axis=-2)   # rows = eigenvectors
    return np.maximum(lam, 0.0), U


class _LinSys:
    """Linearised line system in the eigen-coordinates of each profile's S2: V_i = U diag(λ + σ_c²) Uᵀ, so a REML evaluation is
    one diagonal-weighted GEMM (no per-profile 2×2 work)."""

    def __init__(self, J, e0, S2, w, cls_masks, Lam, p0, eta_idx=None, eta_scale=None):
        lam, U = _eig2(S2)
        self.R = np.einsum("nki,nip->nkp", U, J).reshape(-1, J.shape[2])         # (2n, P)
        self.Rt = np.ascontiguousarray(self.R.T)                                   # (P, 2n)
        self.rho = np.einsum("nki,ni->nk", U, e0).reshape(-1)
        self.lam = lam.reshape(-1)
        self.w2 = np.repeat(w, 2)
        self.cls_row = np.full(self.lam.size, -1)
        for ci, msk in enumerate(cls_masks):
            self.cls_row[np.repeat(msk, 2)] = ci
        self.Lam, self.p0 = Lam, p0
        # η columns per component: (w0 parts, n0 parts) — one variance component each (the robot's leg-to-leg scatter is not isotropic)
        ei = [] if eta_idx is None else list(eta_idx)
        if ei and not isinstance(ei[0], (list, tuple, np.ndarray)):
            ei = [ei[0::2], ei[1::2]]
        self.eta_idx = [np.asarray(x, dtype=int) for x in ei] if ei else []
        # per leg and component: its variance relative to the reference speed's, s0²(v_leg)/s0²(v_ref) (calibrated speed model; 1
        # when not calibrated or one speed)
        self.eta_scale = [np.ones(ix.size) for ix in self.eta_idx] if eta_scale is None else \
            [np.asarray(sc_, dtype=float).reshape(ix.size) for sc_, ix in zip(eta_scale, self.eta_idx)]

    def with_eta(self, log_sig_eta):
        """Λ with the η blocks set to 1/(σ_η,c²·r_ℓ) per component (σ_η,c at the reference speed, r_ℓ = the leg's speed scale)."""
        if log_sig_eta is None or not self.eta_idx:
            return self.Lam
        Lam = self.Lam.copy()
        ls = np.broadcast_to(np.asarray(log_sig_eta, dtype=float), (len(self.eta_idx),))
        for ix, l_, sc_ in zip(self.eta_idx, ls, self.eta_scale):
            Lam[ix, ix] = math.exp(-2 * float(l_)) / sc_
        return Lam

    def v(self, log_sig):
        return self.lam + np.exp(2 * np.asarray(log_sig, dtype=float))[self.cls_row]

    def solve(self, log_sig, log_sig_eta=None):
        v = self.v(log_sig)
        c = self.w2 / v
        sc = np.sqrt(c)
        X = self.Rt * sc
        Lam = self.with_eta(log_sig_eta)
        A = X @ X.T + Lam
        g = X @ (sc * self.rho) + Lam @ self.p0
        try:
            dp = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            dp = -np.linalg.lstsq(A, g, rcond=None)[0]
        return self.p0 + dp, A, self.rho + self.R @ dp, v, c

    def eta_profile(self, log_sig, log_sig_eta, comp: int, grid) -> tuple:
        """−REML log-likelihood, solution and covariance A⁻¹ on a grid of ln σ_η for one component (the other held): the data part
        of the normal matrix does not depend on σ_η, so the whole grid costs one GEMM. → (nll (G,), p (G,P), C (G,P,P))."""
        v = self.v(log_sig)
        c = self.w2 / v
        sc = np.sqrt(c)
        X = self.Rt * sc
        M = X @ X.T
        g0 = X @ (sc * self.rho)
        r0 = float(np.sum(c * self.rho ** 2))
        base = float(np.sum(self.w2 * np.log(v)))
        nl, ps, Cs = [], [], []
        for x in grid:
            le = np.array(log_sig_eta, dtype=float).copy()
            le[comp] = float(x)
            Lam = self.with_eta(le)
            A = M + Lam
            try:
                Ai = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                Ai = np.linalg.pinv(A)
            dp = -Ai @ (g0 + Lam @ self.p0)
            p = self.p0 + dp
            _, lA = np.linalg.slogdet(A)
            ld = float(sum(2.0 * ix.size * float(l_) for ix, l_ in zip(self.eta_idx, le)))
            nl.append(0.5 * (base + r0 + 2.0 * float(dp @ g0) + float(dp @ M @ dp) + float(p @ Lam @ p) + lA + ld))
            ps.append(p)
            Cs.append(Ai)
        return np.array(nl), np.array(ps), np.array(Cs)

    def nll(self, log_sig, log_sig_eta=None):
        """−REML log-likelihood (π·Huber-weighted profiles; η as random effects N(0, σ_η² I))."""
        p, A, e, v, c = self.solve(log_sig, log_sig_eta)
        Lam = self.with_eta(log_sig_eta)
        _, lA = np.linalg.slogdet(A)
        ld_eta = 0.0                                                                            # −log|Λ_η|
        if log_sig_eta is not None and self.eta_idx:
            ls = np.broadcast_to(np.asarray(log_sig_eta, dtype=float), (len(self.eta_idx),))
            ld_eta = float(sum(2.0 * ix.size * float(l_) for ix, l_ in zip(self.eta_idx, ls)))
        return 0.5 * (float(np.sum(self.w2 * np.log(v))) + float(np.sum(c * e * e)) + float(p @ Lam @ p) + lA + ld_eta)


def _huber_w(dist):
    return np.where(dist <= HUBER_C, 1.0, HUBER_C / np.maximum(dist, 1e-300))


def _fit_line_model(prof: _Profiles, prior: SeamPrior, d: int, pi0: np.ndarray, sel: np.ndarray, sig0: dict) -> dict:
    """IRLS (Huber on the Mahalanobis norm) + REML σ_ζ per class + GN for s*, one polynomial degree. The direction offset b (w, n) is
    a fixed effect where the rank identifies it (_direction_identified); η per leg with one REML variance component per component."""
    dir_ok = _direction_identified(prof, d, sel, pi0)
    P, cls_col, leg_col, dir_col, names = _line_layout(prof, d, dir_ok)
    p = np.zeros(P)
    Lam = np.zeros((P, P))
    rep = float(prior.rep_sigma_m) if prior.rep_sigma_m is not None else float("nan")
    rep_known = math.isfinite(rep) and rep > 0
    eta_idx = [[j for j in leg_col.values()], [j + 1 for j in leg_col.values()]]       # (w0 parts, n0 parts)
    # σ_η,w, σ_η,n: the datasheet repeatability (spec §4.1, isotropic) or, when that slot is empty, two REML variance components (needs
    # ≥ 2 fine legs; one fine leg is the gauge itself → η pinned). The direction offset is a fixed effect, so these are the legs'
    # scatter about their direction's mean
    reml_eta = (not rep_known) and len(leg_col) >= 2
    e0_ = sig0.get("_eta", (1e-4, 1e-4))
    log_sig_eta = np.log(np.maximum(np.broadcast_to(np.asarray(e0_, dtype=float), (2,)), 1e-7)) if reml_eta else None
    # the legs' replication per component: ν_c = (gauge legs that show this seam at the line — mean profile weight π > ½, the
    # redundancy that counts, as the plate ends' inliers) − rank(their leg-level fixed design: the gauge intercept, d·b_c where b_c is
    # in the model). With ν_c ≤ 2 the variance component is not established (its posterior has no finite second moment). With
    # ν_c = 0 the legs' offsets are fully explained by the fixed effects (two passes of opposite directions: intercept and b; one
    # pass) — the REML is flat and would return its search bound: the component is held at the high-breakdown S-estimate of the
    # legs' offsets (_robust_scales: ≈ 0 then — pinned, as one pass always was)
    eta_est, eta_nu = [True, True], [None, None]
    # calibrated priors of the two components (calib g1: noise_priors leg_offset_w / leg_offset_n — the robot + sensor population,
    # σ²(v) = λ·s0²(v), λ ~ Scaled-Inv-χ²(ν0, 1); noise_priors module): the REML becomes a posterior (MAP for the plug-in, the
    # mixture over the posterior for the covariance) with ν0 + ν_c degrees of freedom; σ_η,c is at the reference speed v_ref (median
    # speed of the η legs), a leg at another speed carries s0²(v_leg)/s0²(v_ref) of it
    from .noise_priors import parse_noise_priors, refine_bounds
    npri = parse_noise_priors(getattr(prior, "noise_priors", None))
    pe = (npri.get("leg_offset_w"), npri.get("leg_offset_n"))
    v_leg = np.array([float(np.median(np.abs(prof.v[prof.idx(i)] @ prof.f0["u0"]))) if prof.idx(i).size else 0.0
                      for i in leg_col])
    v_ref = float(np.median(v_leg)) if v_leg.size else 0.0
    eta_scale = [pe[c_].s2(v_leg) / pe[c_].s2(v_ref) if pe[c_] is not None else np.ones(v_leg.size) for c_ in (0, 1)]

    def lp_eta(ce, log_sig_):
        """log prior density of ln σ_η,c (calibrated; 0 = flat when not calibrated — the REML as before)."""
        return pe[ce].log_density_log_sigma(np.exp(log_sig_), v_ref) if pe[ce] is not None else 0.0
    gauge = _gauge(prof.legs)
    if not rep_known and gauge is not None:              # the gauge legs' offsets are unknown without their replication (1 pass: ν = 0)
        pl_ = np.asarray(pi0, dtype=float)
        lix = [i for i, lg in enumerate(prof.legs) if lg.cls == gauge
               and np.any(sel & (prof.leg == i)) and float(np.mean(pl_[sel & (prof.leg == i)])) > 0.5]
        for ce in (0, 1):
            Xl = np.c_[np.ones(len(lix)), prof.dir_leg[lix]] if dir_ok[ce] else np.ones((len(lix), 1))
            eta_nu[ce] = int(len(lix) - _rank(Xl)) if lix else 0
            # the calibrated ν0 counts where the legs' offsets are in the model (η per leg: ≥ 2 gauge legs); one pass has no η
            # column — its own offset is not represented, the root is not established (as before)
            eta_est[ce] = eta_nu[ce] + (pe[ce].nu0 if (pe[ce] is not None and reml_eta) else 0.0) > 2
    rob_ = sig0.get("_eta_robust")
    # ν_c = 0 without a calibrated prior: the REML is flat — held at the S-estimate (≈ 0, pinned); with the prior its MAP is the prior's
    eta_hold = [bool(reml_eta and eta_nu[ce] == 0 and rob_ is not None and pe[ce] is None) for ce in (0, 1)]
    for ce in (0, 1):
        if eta_hold[ce]:
            log_sig_eta[ce] = math.log(max(float(rob_[ce]), 1e-9))
    for i, j in leg_col.items():
        prec = 1.0 / rep ** 2 if rep_known else 1e30
        Lam[j, j] = Lam[j + 1, j + 1] = prec
    he = float(prior.he_sigma_m) if prior.he_sigma_m is not None else float("nan")
    for c, j in cls_col.items():                           # class offsets: hand-eye residual prior (free if the slot is empty)
        Lam[j, j] = Lam[j + 1, j + 1] = 1.0 / he ** 2 if (math.isfinite(he) and he > 0) else 0.0
    # profile jitter σ_ζ per stream class; the dwell sites share one (a site has 1–3 grabs and its own offset — its jitter alone
    # is not identified)
    jit = np.array([("dwell" if c.startswith("dwell") else c) for c in prof.cls[sel]])
    classes = [c for c in ("pass", "transit", "dwell") if np.any(jit == c)] + sorted(set(jit) - {"pass", "transit", "dwell"})
    cls_masks = [jit == c for c in classes]
    log_sig = np.array([math.log(sig0.get(c, 1e-4)) for c in classes])
    S2 = prof.S2[sel]
    h = np.ones(int(sel.sum()))
    pi = pi0[sel]
    s = None
    reml_done = False
    warm = all(c in sig0 for c in classes)
    for rnd in range(IRLS_ROUNDS):
        e0, J, s = _line_residuals(prof, p, d, cls_col, leg_col, s, sel, dir_col)
        w = pi * h
        ls = _LinSys(J, e0, S2, w, cls_masks, Lam, p, eta_idx, eta_scale)
        if not reml_done:
            rms = math.sqrt(float(np.sum(w * np.sum(e0 ** 2, axis=1)) / max(float(np.sum(w)), 1e-12)))
            hi = math.log(max(10.0 * rms, 1e-6))
            old = log_sig.copy()
            if reml_eta:
                for ce in (0, 1):                          # one component at a time (coordinate-wise REML, as the σ_ζ below)
                    if eta_hold[ce]:
                        continue

                    def obj_e(x, ce=ce):
                        le = log_sig_eta.copy()
                        le[ce] = x
                        return ls.nll(log_sig, le) - float(lp_eta(ce, x))
                    log_sig_eta[ce] = float(minimize_scalar(obj_e, bounds=(math.log(1e-7), math.log(1e-2)), method="bounded",
                                                            options={"xatol": 1e-2}).x)
            for ci in range(len(classes)):
                def obj(x, ci=ci):
                    lsig = log_sig.copy()
                    lsig[ci] = x
                    return ls.nll(lsig, log_sig_eta)
                lo_b, hi_b = math.log(1e-7), hi
                if warm:                                   # [R] bracket ±1 in ln σ around the previous value (re-widened at an edge)
                    lo_b, hi_b = max(lo_b, log_sig[ci] - 1.0), min(hi_b, log_sig[ci] + 1.0)
                res = minimize_scalar(obj, bounds=(lo_b, max(hi_b, lo_b + 1e-3)), method="bounded", options={"xatol": 3e-2})
                x = float(res.x)
                if warm and (x - lo_b < 0.02 or hi_b - x < 0.02):
                    x = float(minimize_scalar(obj, bounds=(math.log(1e-7), hi), method="bounded",
                                              options={"xatol": 1e-2}).x)
                log_sig[ci] = x
            reml_done = warm and float(np.max(np.abs(log_sig - old))) < 3e-2
            warm = True
        p_new, A, _, _, _ = ls.solve(log_sig, log_sig_eta)
        Vi, _ = _inv2(_V_of(S2, cls_masks, log_sig))
        e = e0 + J @ (p_new - p)
        dist = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", e, Vi, e), 0.0))
        h = _huber_w(dist)
        step = float(np.max(np.abs(p_new - p))) if P else 0.0
        p = p_new
        if step < IRLS_TOL and rnd >= 2 and reml_done:
            break
    # final linearisation at p̂: covariance (Laplace of the stacked system), likelihood for BIC
    e0, J, s = _line_residuals(prof, p, d, cls_col, leg_col, s, sel, dir_col)
    V = _V_of(S2, cls_masks, log_sig)
    Vi, ld = _inv2(V)
    w = pi * h
    ls_f = _LinSys(J, e0, S2, w, cls_masks, Lam, p, eta_idx, eta_scale)
    _, A, _, _, _ = ls_f.solve(log_sig, log_sig_eta)
    try:
        C = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(A)
    # small samples (as the plate ends, end_faces.fuse_end_faces): each leg-offset variance component is known only through the legs'
    # replication, ν_c = (legs with η) − rank(their leg-level fixed design: the gauge intercept, d·b_c where b_c is in the model). The
    # line's covariance is the mixture over each component's posterior (REML likelihood × the scale-invariant prior p(σ²) ∝ 1/(v̄ + σ²),
    # v̄ = the legs' own data variance), one component at a time. With ν_c ≤ 2 that posterior has no finite second moment: the root
    # component is not established (the plug-in covariance stays for planning; the done test reads 'eta_established').
    if reml_eta:
        Mdat = ls_f.Rt * np.sqrt(ls_f.w2 / ls_f.v(log_sig))
        Mdat = np.einsum("ij,ij->i", Mdat, Mdat)                               # diagonal of the data information
        C_mix = C.copy()
        for ce in (0, 1):
            if not eta_est[ce]:
                continue
            jj = [j + ce for j in leg_col.values()]
            vbar = float(np.median(1.0 / np.maximum(Mdat[jj], 1e-300)))
            lo_g, hi_g = 0.5 * math.log(max(vbar, 1e-30)) + VC_SPAN[0], 0.5 * math.log(max(vbar, 1e-30)) + VC_SPAN[1]
            if pe[ce] is not None:                       # the calibrated prior's central range (then refined to the posterior)
                lo_g = min(lo_g, math.log(pe[ce].quantile_sigma(PRIOR_TAIL, v_ref)))
                hi_g = max(hi_g, math.log(pe[ce].quantile_sigma(1.0 - PRIOR_TAIL, v_ref)))

            def post_w(grid, ce=ce, vbar=vbar):
                nl_g, p_g, C_g = ls_f.eta_profile(log_sig, log_sig_eta, ce, grid)
                sg2 = np.exp(2.0 * grid)
                lw = -nl_g + (lp_eta(ce, grid) if pe[ce] is not None else np.log(sg2 / (vbar + sg2)))
                wg = np.exp(lw - np.max(lw))
                return wg / wg.sum(), p_g, C_g
            grid = np.linspace(lo_g, hi_g, VC_GRID)
            b_ = refine_bounds(grid, post_w(grid)[0])
            grid = np.linspace(b_[0], b_[1], VC_GRID)
            wg, p_g, C_g = post_w(grid)
            dpg = p_g - p[None, :]
            Cc = np.einsum("g,gij->ij", wg, C_g) + np.einsum("g,gi,gj->ij", wg, dpg, dpg)
            C_mix = C_mix + (Cc - C)
        C = 0.5 * (C_mix + C_mix.T)
    dist = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", e0, Vi, e0), 0.0))
    rho = np.where(dist <= HUBER_C, 0.5 * dist ** 2, HUBER_C * dist - 0.5 * HUBER_C ** 2)
    lnL = float(np.sum(pi * (-0.5 * (ld + 2 * math.log(2 * math.pi)) - rho)))
    n_obs = 2.0 * float(np.sum(pi))
    k_line = 2 * (d + 1)
    bic = -2.0 * lnL + k_line * math.log(max(n_obs, 2.0))
    nl = 2 * (d + 1)
    offs = {"class": {}, "leg": {}}
    offs_cov = {"class": {}, "leg": {}}
    for c, j in cls_col.items():
        offs["class"][c] = p[j:j + 2].copy()
        offs_cov["class"][c] = C[j:j + 2, j:j + 2].copy()
    for i, j in leg_col.items():
        offs["leg"][prof.legs[i].name] = p[j:j + 2].copy()
        offs_cov["leg"][prof.legs[i].name] = C[j:j + 2, j:j + 2].copy()
    # direction offset b (w0, n0): value and covariance of the identified components (0 and ∞ where the rank says no)
    ji = [dir_col.get(0), dir_col.get(1)]
    offs["direction"] = np.array([p[j] if j is not None else 0.0 for j in ji])
    cb = np.diag([math.inf if j is None else 0.0 for j in ji])
    for a_, ja in enumerate(ji):
        for b_, jb in enumerate(ji):
            if ja is not None and jb is not None:
                cb[a_, b_] = C[ja, jb]
    offs_cov["direction"] = cb
    offs["direction_identified"] = (ji[0] is not None, ji[1] is not None)
    offs["leg_direction"] = {lg.name: int(prof.dir_leg[i]) for i, lg in enumerate(prof.legs)}
    # moving legs inform the line: without b identified, the line is then the frame of their motion direction, not the static one
    offs["moving_legs"] = bool(np.any((prof.dsign[sel] != 0) & (pi0[sel] > 0)))
    sig = {c: math.exp(log_sig[ci]) for ci, c in enumerate(classes)}
    sig_wn = (rep, rep) if rep_known else ((float(math.exp(log_sig_eta[0])), float(math.exp(log_sig_eta[1]))) if reml_eta
                                           else (0.0, 0.0))
    vc = {"ls": ls_f, "log_sig": log_sig.copy(), "log_sig_eta": None if log_sig_eta is None else np.array(log_sig_eta, dtype=float),
          "nu": tuple(eta_nu), "v_ref": v_ref, "reml": bool(reml_eta)}
    return {"d": d, "params": p[:nl].reshape(2, d + 1).copy(), "cov": C[:nl, :nl].copy(), "p_full": p, "C_full": C, "_vc": vc,
            "eta_v_ref_mps": v_ref, "eta_priors": [None if q_ is None else q_.to_dict() for q_ in pe],
            "sigma_eta": _iso_sigma(sig_wn), "sigma_eta_wn": sig_wn, "eta_established": tuple(eta_est), "eta_nu": tuple(eta_nu),
            "sigma_eta_wn_source": tuple(("S-estimate (not established)" if eta_hold[ce] else
                                          ("posterior (calibrated prior)" if pe[ce] is not None else "reml")) if reml_eta else
                                         ("datasheet" if rep_known else "pinned") for ce in (0, 1)),
            "sigma_eta_source": "datasheet" if rep_known else ("reml" if reml_eta else "pinned"),
            "offsets": offs, "offsets_cov": offs_cov, "sigma_zeta": sig, "bic": bic, "lnL": lnL, "n_obs": n_obs,
            "e": e0, "s": s, "dist": dist, "huber": h, "Vi": Vi, "cls_col": cls_col, "leg_col": leg_col, "dir_col": dir_col,
            "names": names}


def _iso_sigma(sig_wn) -> float:
    """One σ for a leg offset in a direction the line does not see (along the seam: the end fusions' η_u): the isotropic equivalent
    of the two components, √((σ_w² + σ_n²)/2) — what one isotropic variance component of the same legs would be."""
    a, b = float(sig_wn[0]), float(sig_wn[1])
    return math.sqrt(0.5 * (a * a + b * b))


def line_eta_loglik(vc: dict, comp: int, sig_grid: np.ndarray) -> np.ndarray:
    """The line's restricted log-likelihood of the leg-offset scatter σ_η,c (component 0 = w, 1 = n) at the reference speed on
    sig_grid, the other component held at its fit (no prior) — the run's evidence about it (noise-prior calibration; vc = the
    line fit's '_vc', estimate(vc_out=…)['line'])."""
    if not vc or not vc.get("reml") or vc.get("log_sig_eta") is None:
        return np.full(len(sig_grid), np.nan)
    nl, _, _ = vc["ls"].eta_profile(vc["log_sig"], vc["log_sig_eta"], comp, np.log(np.asarray(sig_grid, dtype=float)))
    return -nl


def _robust_scales(prof: _Profiles, F: dict, sel: np.ndarray, pi: np.ndarray) -> dict:
    """High-breakdown start of the variance components for the first λ_line (the S-estimator start of a robust fit): σ_ζ per jitter
    group solves the π-weighted median of eᵢᵀ(S2_i + σ² I)⁻¹eᵢ = median of χ²₂ (= 2 ln 2); σ_η solves the same over the legs' offsets
    η̂_ℓ with their covariance. The first fit weights profiles by shape evidence only (π = logistic λ), so profiles or a whole leg that
    see something else (lab at10: a stationary pass beyond the end, fits scattered ±15 mm) inflate the REML components and would then
    pass as 'this seam' — up to half of them cannot move a median. The second fit's components are REML again (π from this λ_line)."""
    q0 = 2.0 * math.log(2.0)
    e = F["e"]
    S2 = prof.S2[sel]
    w = pi[sel]
    grp = np.array([("dwell" if c.startswith("dwell") else c) for c in prof.cls[sel]])
    sig = dict(F["sigma_zeta"])

    def wmed(x, wt):
        o = np.argsort(x)
        cw = np.cumsum(wt[o])
        return float(x[o][min(int(np.searchsorted(cw, 0.5 * cw[-1])), x.size - 1)])

    def solve(qfun, wt, hi):
        lo_, hi_ = math.log(1e-9), math.log(hi)
        if wmed(qfun(math.exp(lo_)), wt) <= q0:
            return math.exp(lo_)
        for _ in range(GOLDEN_ITERS + 20):              # [R] bisection steps (the median of q decreases with σ)
            mid = 0.5 * (lo_ + hi_)
            if wmed(qfun(math.exp(mid)), wt) > q0:
                lo_ = mid
            else:
                hi_ = mid
        return math.exp(0.5 * (lo_ + hi_))

    V = S2.copy()
    for c in sig:
        msk = (grp == c) & (w > 0)
        if not np.any(msk):
            continue
        ec, Sc, wc = e[msk], S2[msk], w[msk]

        def qf(sg, ec=ec, Sc=Sc):
            Vi_, _ = _inv2(Sc + sg * sg * np.eye(2))
            return np.einsum("ni,nij,nj->n", ec, Vi_, ec)
        hi = 10.0 * float(np.sqrt(np.max(np.sum(ec * ec, axis=1)))) + 1e-6
        sig[c] = solve(qf, wc, hi)
        on = grp == c
        V[on, 0, 0] += sig[c] ** 2
        V[on, 1, 1] += sig[c] ** 2
    out = dict(F, sigma_zeta=sig)
    Vi, _ = _inv2(V)
    out["Vi"] = Vi
    leg_col = F.get("leg_col") or {}
    if len(leg_col) >= 2 and F.get("sigma_eta_source") == "reml":
        names = [prof.legs[i].name for i in leg_col]
        et = np.array([F["offsets"]["leg"][n_] for n_ in names])
        Ce = np.array([F["offsets_cov"]["leg"][n_] for n_ in names])
        q1 = float(chi2.ppf(0.5, 1))                    # one component: the median of χ²₁
        swn = []
        for c in (0, 1):                                # per component (w, n): the legs' offsets about their direction's mean

            def qe(sg, c=c):
                return et[:, c] ** 2 / (Ce[:, c, c] + sg * sg)
            swn.append(_solve_median(qe, np.ones(len(names)), 10.0 * float(np.max(np.abs(et[:, c]))) + 1e-6, q1, wmed))
        out["sigma_eta_wn"] = (swn[0], swn[1])
        out["sigma_eta"] = _iso_sigma(swn)
    return out


def _solve_median(qfun, wt, hi, q0, wmed) -> float:
    """σ with weighted median of qfun(σ) = q0 (q decreases with σ): bisection in ln σ from 1e-9 to hi [R steps]."""
    lo_, hi_ = math.log(1e-9), math.log(hi)
    if wmed(qfun(math.exp(lo_)), wt) <= q0:
        return math.exp(lo_)
    for _ in range(GOLDEN_ITERS + 20):
        mid = 0.5 * (lo_ + hi_)
        if wmed(qfun(math.exp(mid)), wt) > q0:
            lo_ = mid
        else:
            hi_ = mid
    return math.exp(0.5 * (lo_ + hi_))


def fit_line(legs: list, prior: SeamPrior, *, model: str = "auto", _prof: Optional[_Profiles] = None,
             seg_w: Optional[np.ndarray] = None) -> dict:
    """§4.2: covariance-weighted robust 3D root line over all legs. model 'line' | 'quadratic' | 'auto' (BIC, D10).

    Two rounds: round 1 weights π_i = logistic(λ_i); round 2 uses λ_i^line = λ_i + ln N(e_lat; 0, V_lat) − ln(1/FOV_i)
    (groove at the line vs anywhere along the laser line). seg_w (per profile, default 1) = 1 − posterior of the segment-level outlier
    indicators (§4.9: the leg / station window the profile belongs to) — the EM weight of a segment that does not fit the fused model.
    Returns the line dict used by the other functions."""
    f0 = prior.frame0()
    prof = _prof if _prof is not None else _Profiles(legs, f0)
    sel = prof.ok.copy()
    models = {"line": [1], "quadratic": [2], "auto": [1, 2]}[model]
    sw_ = np.ones(prof.N) if seg_w is None else np.asarray(seg_w, dtype=float)
    pi = _logistic(prof.lam) * sw_
    fits = {}
    sig0 = {}
    for rnd in range(2):
        fits = {}
        for d in (models if rnd == 1 else models[:1]):     # round 1 only initialises π (λ_line), lowest degree
            fits[d] = _fit_line_model(prof, prior, d, pi, sel, sig0)
        best = min(fits.values(), key=lambda F: F["bic"])
        sig0 = dict(best["sigma_zeta"])
        if best["sigma_eta_source"] == "reml":
            sig0["_eta"] = tuple(max(float(x), 1e-7) for x in best["sigma_eta_wn"])
        if rnd == 0:
            rob = _robust_scales(prof, best, sel, pi)
            pi_noseg = _logistic(_lam_line(prof, rob, sel))
            pi = pi_noseg * sw_
            if rob.get("sigma_eta_wn") is not None and rob.get("sigma_eta_source") == "reml":
                sig0["_eta_robust"] = tuple(float(x) for x in rob["sigma_eta_wn"])
    best_d = min(fits, key=lambda d: fits[d]["bic"])
    F = fits[best_d]
    lam_line = _lam_line(prof, F, sel)
    tp = timing_prior(prof.legs, f0["u0"])
    L = LineModel(f0, F["params"], F["cov"])
    out = {"model": L.model, "params": F["params"], "cov": F["cov"], "frame0": f0, "offsets": F["offsets"],
           "sigma_eta": F["sigma_eta"], "sigma_eta_wn": tuple(F["sigma_eta_wn"]), "sigma_eta_source": F["sigma_eta_source"],
           "eta_established": tuple(F["eta_established"]), "eta_nu": tuple(F["eta_nu"]),
           "sigma_eta_wn_source": tuple(F.get("sigma_eta_wn_source") or ()),
           "eta_v_ref_mps": F.get("eta_v_ref_mps"), "eta_priors": F.get("eta_priors"),
           "gauge": _gauge(prof.legs),
           # the time-offset prior from the run's own timing (frame and pose sample periods) — the calibration's σ is not used
           "dt_sigma_s": tp["sigma_s"], "timing": tp, "delta_prior": js_source.DeltaPrior(tp["sigma_s"], tp["p_ref"]),
           "leg_direction": dict(F["offsets"]["leg_direction"]),
           "eta_legs": [prof.legs[i].name for i in F["leg_col"]],
           "offsets_cov": F["offsets_cov"], "sigma_zeta": F["sigma_zeta"], "bic": {LineModel(f0, fits[d]["params"]).model:
                                                                                    fits[d]["bic"] for d in fits},
           "pi": pi, "pi_noseg": pi_noseg, "lam": prof.lam, "lam_line": lam_line, "used": sel, "n_used": int(sel.sum()),
           "huber_downweighted": int(np.sum(F["huber"] < 1)), "seg_w": sw_,
           "_prof": prof, "_line": L, "_F": F}
    # s* of every profile on the fused line (recorded planes, no offset): ends and stations start from these
    out["s_star"] = L.s_star(prof.o, prof.m)
    return out


def _lam_line(prof: _Profiles, F: dict, sel: np.ndarray) -> np.ndarray:
    """λ_i^line (§4.2): posterior log odds "THIS seam's groove at the fused line" vs "no groove, or a groove-like structure anywhere
    else in the sensor's measurement window" (same form as WP1's seam term):
        λ^line = λ + ln f(e; Σ_i + σ_ζ² I) − logaddexp(0, λ − ln A_w),   A_w = FOV_i·W_z (uniform location under 'elsewhere').
    f = the line fit's own residual law (normalised 2-D Huber on the Mahalanobis norm, HUBER_C — the law the fit is robust to), not its
    Gaussian core alone: a local excursion of this seam's root reading (tack, gap change, local fit bias; lab at17: 0.5–0.8 mm over
    60 mm = 6–9 σ_ζ) stays this seam, while a groove-like feature millimetres aside still loses (linear tail).
    For a strongly groove-like profile (λ ≫ ln A_w) only the location decides — a table feature 20 mm aside is no longer carried by its
    shape evidence. Profiles without a usable fit have no location: ln f → −ln A_w (as likely here as anywhere).
    The leg offset η is a random effect: the location is predictive over η ~ N(0, diag(σ_η,w², σ_η,n²)) (residual without the leg's own η̂,
    Σ + B·diag(σ_η²)·Bᵀ,
    B = the in-plane image of (w0, n0)), not conditional on η̂ — a leg that sees another feature cannot make it 'this seam' by fitting
    its own offset to it (lab at10: a stationary pass beyond the end, its random fits at η̂ = 21 mm, read 'S = E = 376 mm')."""
    V = np.linalg.inv(F["Vi"]) * 1e6                                                   # mm²
    e = F["e"] * 1e3                                                                    # mm
    leg_col = F.get("leg_col") or {}
    swn = np.asarray(F.get("sigma_eta_wn", (F.get("sigma_eta", 0.0),) * 2), dtype=float)
    if leg_col and float(np.max(swn)) > 0:
        f0 = prof.f0
        L = LineModel(f0, F["params"])
        ss = F["s"]
        E = np.stack([prof.ex[sel], prof.ez[sel]], axis=1)                              # (n,2,3)
        t = L.t(ss)
        m = prof.m[sel]
        EQ = E - np.einsum("nki,ni->nk", E, t)[:, :, None] * m[:, None, :] / np.einsum("ni,ni->n", m, t)[:, None, None]
        B = np.stack([EQ @ f0["w0"], EQ @ f0["n0"]], axis=2)                             # (n,2,2): columns = images of w0, n0
        lg = prof.leg[sel]
        eta = np.zeros((e.shape[0], 2))
        has = np.zeros(e.shape[0], bool)
        for i in leg_col:
            on = lg == i
            eta[on] = F["offsets"]["leg"][prof.legs[i].name] * 1e3
            has |= on
        e = e - np.einsum("nij,nj->ni", B, eta)
        V = V + np.where(has[:, None, None], np.einsum("nij,j,nkj->nik", B, (swn * 1e3) ** 2, B), 0.0)
    Vi, _ = _inv2(V)
    lnA = np.log(prof.win_mm2)
    lnN = -lnA.copy()
    c = HUBER_C
    lnZ = math.log(2 * math.pi) + math.log1p(math.exp(-0.5 * c * c) / (c * c))         # ∫ exp(−ρ(|z|)) d²z
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.einsum("ni,nij,nj->n", e, Vi, e)
        _, ld = _inv2(V)
        r = np.sqrt(np.maximum(q, 0.0))
        rho = np.where(r <= c, 0.5 * q, c * r - 0.5 * c * c)
        lnN[sel] = -rho - ld * 0.5 - lnZ
        lnN = np.where(np.isfinite(lnN), lnN, -lnA)
        out = prof.lam + lnN - np.logaddexp(0.0, prof.lam - lnA)
    return np.where(np.isfinite(out), out, 0.0)


def _line_from(line: dict) -> LineModel:
    if "_line" in line:
        return line["_line"]
    return LineModel(line["frame0"], line["params"], line.get("cov"))


def leg_direction(line: dict, leg_name: str) -> int:
    """The leg's motion sign in the line's model (+1 / −1 along u0, 0 static or unknown)."""
    return int((line.get("leg_direction") or (line.get("offsets") or {}).get("leg_direction") or {}).get(leg_name, 0))


def leg_offset_wn(line: dict, leg_name: str, cls: str) -> np.ndarray:
    """(w0, n0) parts of a leg's offset from the line (measured + o = true): class β + motion direction × b + its own η."""
    off = line.get("offsets") or {}
    o = np.zeros(2)
    if cls in (off.get("class") or {}):
        o += np.asarray(off["class"][cls], dtype=float)[:2]
    if leg_name in (off.get("leg") or {}):
        o += np.asarray(off["leg"][leg_name], dtype=float)[:2]
    b = off.get("direction")
    if b is not None:
        o += leg_direction(line, leg_name) * np.asarray(b, dtype=float)[:2]
    return o


# ====================================================================================================================
# §4.3 ends — exact single-segment change point per leg
# ====================================================================================================================
def _offset_coeffs(m, uhat, f0):
    mu = m @ uhat
    return {"c_w": float(m @ f0["w0"] / mu), "c_n": float(m @ f0["n0"] / mu), "c_u": float(m @ f0["u0"] / mu)}


def _segment_readings(leg, line: dict, ll: np.ndarray, extra: Optional[dict] = None, contain: Optional[tuple] = None) -> dict:
    """Exact single-segment change point on per-profile log-likelihood ratios ll (O(n) max-sum, the empty segment = none) and the
    end readings from the profile-likelihood curve on the gaps between consecutive s* (two-sided when the leg has profiles beyond the
    end, else a one-sided bound at the segment's last profile)."""
    L = _line_from(line)
    f0 = line["frame0"]
    T = leg.T_base_opt
    o, m = T[:, :3, 3], T[:, :3, 1]
    s = L.s_star(o, m)
    order = np.argsort(s, kind="stable")
    ss, ll = s[order], np.asarray(ll, float)[order]
    if contain is None:
        best, a, b = max_segment(ll)
    else:                                                 # segment ⊇ [contain]: S and E changes are independent (exact)
        i0 = int(np.searchsorted(ss, contain[0], side="left"))
        i1 = int(np.searchsorted(ss, contain[1], side="right")) - 1
        if i1 < i0:
            best, a, b = max_segment(ll)
        else:
            Cc = np.concatenate([[0.0], np.cumsum(ll)])
            a = int(np.argmax(Cc[i0] - Cc[:i0 + 1]))     # maximises Σ_{a..i0-1} ll
            b = i1 + int(np.argmax(Cc[i1 + 1:] - Cc[i1 + 1]))
            b = max(b, i1)
            best = float(Cc[b + 1] - Cc[a])
    cls = _canon_cls(leg.cls, leg.meta)
    out = {"leg": leg.name, "cls": cls, "direction": leg.direction, "log_bf_seg": float(best), "n": int(leg.n),
           "S": None, "E": None, "segment_m": None}
    if a < 0 or leg.n == 0:
        return out
    out["segment_m"] = (float(ss[a]), float(ss[b]))
    C = np.concatenate([[0.0], np.cumsum(ll)])                    # C[k] = Σ_{i<k} ll_i
    ex = dict(extra or {})

    def coeffs(i0, i1):
        mm = _unit(m[order[i0]] + m[order[i1]])
        sm = 0.5 * (ss[i0] + ss[i1])
        uh = L.tangent(sm)
        vv = 0.5 * (leg.v_opt_mps[order[i0]] + leg.v_opt_mps[order[i1]])
        c = _offset_coeffs(mm, uh, f0)
        c["c_v"] = float(vv @ mm / (mm @ uh))
        c.update(ex)
        return c

    # per-profile along-seam resolution (floor of every reading, "σ_cp ≥ spacing/√12 by construction"): the leg's sample spacing and
    # the class's profile (pose) jitter σ_ζ (REML, in-plane; taken isotropic) — stationary profiles never make a zero-width reading.
    # The common time offset δ is an unknown of the fusion (c_v·δ), not part of a reading's σ.
    dsp = np.diff(ss)
    spacing = float(np.median(dsp)) if dsp.size else 0.0
    jit = float((line.get("sigma_zeta") or {}).get("dwell" if cls.startswith("dwell") else cls, 0.0) or 0.0)

    def floor2(c):
        return spacing ** 2 / 12.0 + jit ** 2

    # S end: change point in gap (ss[k-1], ss[k]) for k = 1..b (segment k..b); k = 0 = the end lies before the leg's first profile
    out["_ss"] = ss
    span = float(ss[-1] - ss[0]) if len(ss) else 0.0               # where a change point of this leg could be read at all (§4.9 extent)
    kS_max = b if contain is None else min(b, max(int(np.searchsorted(ss, contain[0], side="left")), 1))
    kE_min = a if contain is None else max(a, int(np.searchsorted(ss, contain[1], side="right")) - 1)
    n = len(ss)
    if a >= 1:
        k = np.arange(1, kS_max + 1)
        c_ = coeffs(a - 1, a)
        out["S"] = _gap_reading(C[b + 1] - C[k], ss[k - 1], ss[k], "S", leg, c_, floor2(c_),
                                tail=("below", float(ss[0]), float(C[b + 1] - C[0])))
        out["S"]["_gaps"] = k                             # gap index k = (ss[k-1], ss[k])
        out["S"]["span_m"] = span
    else:
        c_ = coeffs(a, min(a + 1, n - 1))
        out["S"] = {"end": "S", "kind": "bound", "side": "upper", "bound_m": float(ss[a]), "leg": leg.name, "cls": cls,
                    "floor_m": math.sqrt(floor2(c_)), **c_}
    # E end: change point in gap (ss[k], ss[k+1]) for k = a..n-2 (segment a..k); k = n-1 = beyond the leg's last profile
    if b <= n - 2:
        k = np.arange(kE_min, n - 1)
        c_ = coeffs(b, b + 1)
        out["E"] = _gap_reading(C[k + 1] - C[a], ss[k], ss[k + 1], "E", leg, c_, floor2(c_),
                                tail=("above", float(ss[-1]), float(C[n] - C[a])))
        out["E"]["_gaps"] = k + 1                         # same convention: gap index j = (ss[j-1], ss[j])
        out["E"]["span_m"] = span
    else:
        c_ = coeffs(max(b - 1, 0), b)
        out["E"] = {"end": "E", "kind": "bound", "side": "lower", "bound_m": float(ss[b]), "leg": leg.name, "cls": cls,
                    "floor_m": math.sqrt(floor2(c_)), **c_}
    return out


def leg_change_points(leg: LegObs, line: dict) -> dict:
    """Groove segment per leg: (â, b̂) = argmax Σ_{s*∈[a,b]} λ^line (exact, O(n)) and its end readings (_segment_readings).
    λ^line (§4.2) instead of λ: a groove-like feature elsewhere in view beyond an end (lab: table level 20 mm below, λ > 0) is
    not this seam (INTERFACE_NOTES WP2 2026-10-09). The groove segment needs the whole template incl. the backing seen in the gap."""
    lam = np.where(np.isfinite(_lam_free(leg)), _lam_free(leg), 0.0)
    prof = line.get("_prof")
    if prof is not None and "lam_line" in line:          # this seam's groove at the line (§4.2 λ_line), not a groove anywhere
        ix = [i for i, lg in enumerate(prof.legs) if lg is leg or lg.name == leg.name]
        if ix and prof.idx(ix[0]).size == leg.n:
            lam = line["lam_line"][prof.idx(ix[0])]
    return _segment_readings(leg, line, lam)


def plate_change_points(leg, line: dict, groove: dict, depth_m: float, spec=None) -> dict:
    """Per plate (template top_L / top_R) presence segment of a leg and its end readings — the seam ends where BOTH plates are present
    (user 2026-10-09). Per-profile log-likelihood ratio "this plate's top is in view at its place" vs "not":
      count term    y = ln(1 + expected inliers of the plate's top face) (WP1 meta['face_inliers']): present = the leg's own
                    groove-segment distribution of y (Cauchy: median, 1.4826·MAD); absent = the face sits on nothing (y = 0, same
                    scale) or on another surface (y like present), equal odds — a high count alone is no evidence of THIS plate;
      location term the fitted top corner's height above the fused root line (·n̂0) [mm]: present = Cauchy(median, MAD) of the leg's
                    groove segment (heavy tail: a degraded fit at the plate is still the plate), absent = anywhere in the calibrated
                    z window (uniform, 1/W_z) —
    inliers of a face fitted to another surface (table, clamp) beyond the end are not this plate. The change happens where the
    top-surface line (height D above the root) leaves the plate end → readings carry the known crossing height h = D.
    Returns {'L': cp, 'R': cp, 'inner': {S, E}} or {} (no face inliers in the leg)."""
    fi = (leg.meta or {}).get("face_inliers")
    names = list((leg.meta or {}).get("face_names") or [])
    if fi is None or not names or groove.get("segment_m") is None:
        return {}
    fi = np.asarray(fi, dtype=float)
    if fi.ndim != 2 or fi.shape[0] != leg.n:
        return {}
    Lm = _line_from(line)
    f0 = line["frame0"]
    T = leg.T_base_opt
    s = Lm.s_star(T[:, :3, 3], T[:, :3, 1])
    g0, g1 = groove["segment_m"]
    ing = (s >= g0) & (s <= g1)
    view = leg if isinstance(leg, _LegView) else _LegView(leg)
    geo = _leg_geometry(view, spec if spec is not None else GrooveSpec(kind=(leg.meta or {}).get("kind", "single_bevel")))
    rel = None
    win = None
    th = np.asarray(leg.theta, float)
    ok = np.all(np.isfinite(th), axis=1)
    out = {}
    for side, face, vname in (("L", "top_L", "top_corner_L"), ("R", "top_R", "top_corner_R")):
        if face not in names or not np.any(ing & ok):
            continue
        y = np.log1p(np.maximum(fi[:, names.index(face)], 0.0))
        g_ = ing & ok
        m_ = float(np.median(y[g_]))
        sc = MAD_SCALE * float(np.median(np.abs(y[g_] - m_)))
        sc = sc if sc > 0 else max(float(np.std(y[g_])), 1e-6)
        # absent = the face sits on nothing (y ≈ 0) or on another surface (y like present), equal odds; Cauchy densities (heavy
        # tails): a single dropout frame cannot outweigh its neighbours, a run of absent profiles is decisive
        lp_pres = -np.log1p(((y - m_) / sc) ** 2)
        lp_zero = -np.log1p((y / sc) ** 2)
        ll = lp_pres - (np.logaddexp(lp_pres, lp_zero) - math.log(2.0))
        C = geo["V"][:, geo["names_v"].index(vname)]
        qn = ((C - Lm.r(s)) @ f0["n0"]) * 1e3                     # top corner height above the fused root line [mm]
        mq = float(np.median(qn[g_]))
        sq = MAD_SCALE * float(np.median(np.abs(qn[g_] - mq)))
        sq = sq if sq > 0 else max(float(np.std(qn[g_])), 1e-6)
        if win is None:
            win = float(np.diff((leg.meta or {}).get("z_window_mm", Z_VALID_MM_DEFAULT))[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            lnC = -np.log(math.pi * sq * (1.0 + ((qn - mq) / sq) ** 2))   # heavy-tailed: a degraded fit at the plate is still the plate
        loc = np.where(ok & np.isfinite(lnC), lnC + math.log(win), 0.0)
        # the groove (both plates + backing in the gap) lies inside each plate's presence segment (structure of the template)
        out[side] = _segment_readings(leg, line, np.where(ok, ll, 0.0) + loc, {"h_m": depth_m, "plate": side},
                                      contain=groove["segment_m"])
        # §4.9 the same evidence without that structure: where the plate's own evidence ends (a groove-like jig at plate height
        # beyond an end extends the groove segment and, through the containment, the plate segments — not the plate's own evidence)
        out[side + "_free"] = _segment_readings(leg, line, np.where(ok, ll, 0.0) + loc, {"h_m": depth_m, "plate": side})
    if "L" in out and "R" in out:
        out["inner"] = {X: _inner_reading(out["L"][X], out["R"][X], out["L"]["_ss"], X) for X in ("S", "E")}
        out["inner_free"] = {X: _inner_reading(out["L_free"][X], out["R_free"][X], out["L_free"]["_ss"], X) for X in ("S", "E")}
    return out


def _gap_reading(logL, lo, hi, end, leg, coeffs, floor2: float = 0.0, tail: Optional[tuple] = None) -> dict:
    """Change-point reading from the profile-likelihood curve: uniform within each gap, gap weights ∝ L·width; σ² adds the per-profile
    along-seam resolution floor2 (≥ spacing²/12). Gaps of zero width (profiles at one s) carry their L as a point at that s (the floor
    gives it width). The full curve is kept for the grid posterior (§4.4): gaps (lo, hi, logL) and the tail (end beyond the leg's edge:
    ('below'|'above', edge s, logL per unit length)), logs relative to one common maximum."""
    logL = np.asarray(logL, dtype=float)
    wdt = np.maximum(hi - lo, 0.0)
    top = float(np.max(logL)) if logL.size else 0.0
    if tail is not None:
        top = max(top, float(tail[2]))
    lw = logL - top
    p = np.exp(lw) * wdt
    if not np.sum(p) > 0:
        p = np.exp(lw)
    p = p / np.sum(p)
    mid = 0.5 * (lo + hi)
    mean = float(np.sum(p * mid))
    var = float(np.sum(p * ((mid - mean) ** 2 + wdt ** 2 / 12.0))) + float(floor2)
    out = {"end": end, "kind": "two_sided", "mean_m": mean, "sigma_m": math.sqrt(max(var, 0.0)), "leg": leg.name,
           "cls": _canon_cls(leg.cls, leg.meta), "floor_m": math.sqrt(max(float(floor2), 0.0)), "_pmf": p,
           "_curve": {"lo": np.asarray(lo, float), "hi": np.asarray(hi, float), "logL": lw,
                      "tail": None if tail is None else (tail[0], float(tail[1]), float(tail[2]) - top)}, **coeffs}
    return out


def _inner_reading(rL: Optional[dict], rR: Optional[dict], ss: np.ndarray, end: str) -> Optional[dict]:
    """Inner of the two plates' ends in one leg (S: max, E: min) on the leg's own gap grid: P(max ≤ k) = F_L(k)·F_R(k) (min: survival
    product) — exact for independent change locations, and flush plates (same gap) give no bias. A bound (plate present at the leg's
    edge) cannot be the inner end when the other plate's change lies inside the leg; both bounds → a bound."""
    if rL is None or rR is None:
        return None
    two = [r for r in (rL, rR) if r["kind"] == "two_sided"]
    if not two:
        return dict(rL, plate="both")
    if len(two) == 1:
        return dict(two[0], plate="inner")
    ng = len(ss)
    pm = []
    for r in (rL, rR):
        q = np.zeros(ng + 1)
        q[r["_gaps"]] = r["_pmf"]
        pm.append(q)
    if end == "S":
        F = np.cumsum(pm[0]) * np.cumsum(pm[1])
        pmf = np.diff(np.concatenate([[0.0], F]))
    else:
        Sv = (1.0 - np.cumsum(pm[0]) + pm[0]) * (1.0 - np.cumsum(pm[1]) + pm[1])      # P(K ≥ k)
        pmf = Sv - np.concatenate([Sv[1:], [0.0]])
    j = np.flatnonzero(pmf > 0)
    j = j[(j >= 1) & (j <= ng - 1)]
    if not j.size:
        return dict(two[0], plate="inner")
    w = pmf[j] / np.sum(pmf[j])
    lo, hi = ss[j - 1], ss[j]
    mid = 0.5 * (lo + hi)
    mean = float(np.sum(w * mid))
    var = float(np.sum(w * ((mid - mean) ** 2 + (hi - lo) ** 2 / 12.0)))
    out = {k_: v for k_, v in two[0].items() if not k_.startswith("_")}
    out.update(mean_m=mean, sigma_m=math.sqrt(max(var, 0.0)), plate="inner")
    return out


# ====================================================================================================================
# §4.4 end fusion with δ, offsets, Cauchy length prior, τ_e (bounded-influence score)
# ====================================================================================================================
def _end_layout(readings: list, line: dict, with_b_u: bool = False):
    """Unknowns: e_S, e_E, δ, h_S, h_E (end-face crossing heights), β_u per non-gauge class, the along-track direction offset b_u
    (when identified). No along-track leg offset η_u with the line's σ: the line measures the legs' lateral / normal scatter only;
    the legs' along-track scatter is the readings' own (τ_e)."""
    names = ["e_S", "e_E", "delta", "h_S", "h_E"]
    gauge = line.get("gauge")
    for c in CLASS_ORDER + tuple(sorted({r["cls"] for r in readings} - set(CLASS_ORDER))):
        if c != gauge and any(r["cls"] == c for r in readings):
            names.append("beta:" + c)
    if with_b_u:
        names.append("b_u")
    return names, {n_: i for i, n_ in enumerate(names)}


def _known_shift(r: dict, line: dict) -> float:
    """c·o of the (w0, n0) offset parts already known from the line (class + motion direction × b + leg)."""
    ow, on = leg_offset_wn(line, r["leg"], r["cls"])
    return r["c_w"] * ow + r["c_n"] * on


def _reading_row(r: dict, col: dict, P: int, line: Optional[dict] = None) -> np.ndarray:
    a = np.zeros(P)
    a[col["e_" + r["end"]]] = 1.0
    a[col["delta"]] = r["c_v"]
    if r.get("h_m") is None:                            # the change happens where the (tilted) laser plane crosses the end face at h
        a[col["h_" + r["end"]]] = r["c_n"]
    if "beta:" + r["cls"] in col:
        a[col["beta:" + r["cls"]]] = -r["c_u"]
    if "b_u" in col and line is not None:
        a[col["b_u"]] = -r["c_u"] * leg_direction(line, r["leg"])
    if "eta:" + r["leg"] in col:
        a[col["eta:" + r["leg"]]] = -r["c_u"]
    return a


def _b_u_identified(two: list, line: dict) -> bool:
    """The along-track direction offset b_u is in the end model when the rank of the readings' fixed-effect design (the ends, the
    class offsets taken as free) grows with its column −c_u·d — readings of both motion directions at one end, not a count."""
    if not two:
        return False
    names, col = _end_layout(two, line, with_b_u=True)
    # the STRUCTURE of the design (which end, which class, which motion direction): c_u = (m·û)/(m·t) is 1 up to the plane's
    # yaw (1 + O(ψ²) ≈ 1 + 1e-4) and varies a little per reading — with the numerical columns −c_u·d, readings of ONE direction
    # made b_u look identified against the end columns (rank grew by those 1e-4 differences; g1c_04 without pass 2: b_u 7.8 m,
    # ends −7.8 m … 124 m, 26 000 stations); the identification is a property of the directions, so c_u enters as its structural 1
    A = np.zeros((len(two), len(names)))
    for i, r in enumerate(two):
        A[i, col["e_" + r["end"]]] = 1.0
        if "beta:" + r["cls"] in col:
            A[i, col["beta:" + r["cls"]]] = -1.0
        A[i, col["b_u"]] = -float(leg_direction(line, r["leg"]))
    fixed = [col["e_S"], col["e_E"]] + [j for nm, j in col.items() if nm.startswith("beta:")]
    return _adds_rank(A[:, fixed], A[:, col["b_u"]])


def _ref_likelihood(dpri, site_rec: dict) -> Optional[dict]:
    """A fusion's likelihood of δ for the echo branch of a later fusion of the same run (js_source.DeltaPrior ref_lik_mean /
    ref_lik_var): the data's own Gaussian N(m_L, V_L) — exact where its one-sided factors leave the echo branch as it is; where they
    tilt it, the Gaussian N(m, V) whose flat-lag posterior Φ(δ/σ_q)·N(m, V) has the tilted branch's first two moments
    (js_source.flat_lag_fit: the projection that keeps τ_echo ≥ 0). {mean_s, var_s, projected}; None when the echo has no weight or
    the readings do not inform δ (V_L ∞: the flat lag stays unidentified — the factors alone do not put it into the model)."""
    if dpri is None or not dpri.p_ref > 0.0 or "V_L" not in site_rec or not dpri.sigma_q > 0:
        return None
    m_L, V_L = float(site_rec["m_L"]), float(site_rec["V_L"])
    if not (math.isfinite(V_L) and V_L > 0):
        return None
    b, tilted = js_source.DeltaPrior(dpri.sigma_q, 1.0)._branches(m_L, V_L, site_rec.get("factors") or (), ("ref",))
    if "ref" not in tilted:
        return {"mean_s": m_L, "var_s": V_L, "projected": False}
    mu, v, _ = b["ref"]
    if not (math.isfinite(v) and v > 0):
        return None
    m, V = js_source.flat_lag_fit(float(mu), float(v), float(dpri.sigma_q), x0=(m_L, V_L))
    return {"mean_s": float(m), "var_s": float(V), "projected": True}


def fuse_ends(readings: list, prior: SeamPrior, line: dict, *, tau_m: Optional[float] = None, length_prior: bool = True,
              _not_obs: frozenset = frozenset()) -> dict:
    """§4.4: e_S, e_E with δ, class/leg along-track offsets, end-face crossing heights, Cauchy length prior (IRLS), τ_e (bounded-
    influence score on the leave-one-out residuals);
    one-sided readings act as (soft) bounds on the conditional posterior and, through their probit in δ, on δ's posterior where δ is
    in the model (the joint-state hypotheses' prior, js_source.DeltaPrior factors). σ_e is always finite: an end that nothing pins down gets
    the model's prior interval (taught point inside the specimen — action contract —, design length + k·tol_L from the other end,
    the bounds). τ_e unidentified → σ given τ_e = 0 and `tau_identified` False (the done test reports the ends unresolved).

    Reading model: ẽ = e_X + c_v·δ + c_n·h_X − c_w·o_w − c_n·o_n − c_u·(β_u,class + d_leg·b_u) + ε, ε ~ N(0, σ_cp² + τ_e²);
    b_u (the robot's along-track direction offset, fixed) only when the readings' rank identifies it (_b_u_identified), δ with the
    run's timing prior (line['dt_sigma_s']);
    h_X = height above the root where the tilted laser plane crosses the end face when the groove evidence changes (unknown in
    [0, D]: prior N(D/2, D²/12) from the design depth; identified only by readings with different tilt c_n)."""
    L = _line_from(line)
    f0 = line["frame0"]
    rd = [r for r in readings if r is not None]
    two = [r for r in rd if r["kind"] == "two_sided"]
    bounds = [r for r in rd if r["kind"] == "bound"]
    # observations of the ends = the two-sided readings of legs the §4.9 checks keep (leg weight > 0, the leg not decided 'not this
    # seam' — the same readings the ends' 'identified' and the plate-end model (end_faces: no reading from such a leg) count). Any
    # other reading keeps its leave-one-out fit check but has no weight and identifies nothing — no term enters the model (b_u),
    # counts as read or adds degrees of freedom for τ_e through it (2026-10-12, test m2: a pass fitted with the wrong-side template —
    # the only pass of its direction — weight 0 (later 0.2 with 'not this seam'), kept b_u in the model; the ends and b_u then
    # shared a direction without (or with only the rejected readings') curvature, the mode drifted along it and the Laplace
    # covariance there was not positive definite / end σ 6 m)
    obs = np.array([float(r.get("leg_w", 1.0)) > 0.0 and not r.get("leg_not_this_seam", False) and id(r) not in _not_obs
                    for r in two], dtype=bool)
    names, col = _end_layout(rd, line, with_b_u=_b_u_identified([r for r, o in zip(two, obs) if o], line))
    P = len(names)
    dt_sig = float(line.get("dt_sigma_s", prior.dt_sigma_s))     # the run's timing prior (fit_line: frame and pose sample periods)
    # the δ prior of the joint-state hypotheses (js_source.DeltaPrior, fit_line): its factor enters the normal equations as the EP
    # site of the cut mixture (delta_site); its precision as a whole classifies δ as random (measured source alone) or fixed (an
    # echo's flat lag has weight) for the degrees of freedom. No timing (poses given per profile): δ ≡ 0 as before
    dpri = line.get("delta_prior") or js_source.DeltaPrior(max(dt_sig, 0.0), 0.0)
    use_site = dt_sig > 0
    jd = col["delta"]
    he = float(prior.he_sigma_m) if prior.he_sigma_m is not None else float("nan")
    prior_prec = np.zeros(P)
    prior_prec[jd] = dpri.prior_precision() if use_site else 1e30
    for nm, j in col.items():
        if nm.startswith("beta:"):
            prior_prec[j] = 1.0 / he ** 2 if (math.isfinite(he) and he > 0) else 0.0
    # end-face crossing height h_X ∈ [0, D] (root … plate top; design depth): uniform → N(D/2, D²/12); h = D/2 + h'
    D_d = float(prior.spec.groove_depth_mm) * 1e-3
    prior_prec[col["h_S"]] = prior_prec[col["h_E"]] = 12.0 / D_d ** 2 if D_d > 0 else 1e30

    def h_mean_shift(r):
        return r["c_n"] * (0.5 * D_d if r.get("h_m") is None else float(r["h_m"]))

    A = np.array([_reading_row(r, col, P, line) for r in two]).reshape(len(two), P)
    y = np.array([r["mean_m"] + _known_shift(r, line) - h_mean_shift(r) for r in two], dtype=float)
    vcp = np.array([r["sigma_m"] ** 2 for r in two], dtype=float)
    seen = (A[obs] != 0).any(axis=0) if np.any(obs) else np.zeros(P, bool)
    # τ_e identification: dof of the gauge readings over the free unknowns (unknowns with a prior are random effects)
    rand = [j for j in range(P) if prior_prec[j] > 0 and seen[j]]
    fixed = [j for j in range(P) if j not in rand and seen[j]]
    hw = np.ones(len(two))                               # Huber weights of the readings (robust fusion, as the line §4.2)
    # τ_e = per-leg end scatter of the readings that define the ends: the gauge class (aligned passes). Readings of other classes
    # (transit, dwell) reach the ends only through their class offset (σ_he, §12-4) and do not set τ_e.
    gidx = np.array([i for i, r in enumerate(two) if r["cls"] == line.get("gauge") and obs[i]], dtype=int)
    Ag = A[gidx] if gidx.size else np.zeros((0, P))
    seen_g = (Ag != 0).any(axis=0) if gidx.size else np.zeros(P, bool)
    rand_g = [j for j in range(P) if prior_prec[j] > 0 and seen_g[j]]
    fixed_g = [j for j in range(P) if j not in rand_g and seen_g[j]]
    dof = int(gidx.size - (np.linalg.matrix_rank(Ag[:, fixed_g]) if fixed_g else 0))
    if tau_m is not None:
        tau, tau_ident = float(tau_m), True
    else:
        tau, tau_ident = 0.0, dof > 0
    W = hw / (vcp + tau ** 2) if len(two) else np.zeros(0)
    Ld = float(prior.spec.weld_length_mm) * 1e-3
    gam = (prior.tol.length_mm * 1e-3) if (prior.tol.length_mm and length_prior) else None
    lrow = np.zeros(P)
    lrow[col["e_S"]], lrow[col["e_E"]] = -1.0, 1.0
    has = {X: bool(seen[col["e_" + X]]) for X in ("S", "E")}
    use_len = gam is not None and (has["S"] or has["E"])
    # identified unknowns: read, or with a prior, or an end tied to a read end by the length prior
    ident = seen | (prior_prec > 0)
    for X, Y in (("S", "E"), ("E", "S")):
        if not has[X] and has[Y] and use_len:
            ident[col["e_" + X]] = True
    keep = np.flatnonzero(ident)
    rest = np.flatnonzero(~ident)
    # starting point (and the value of unidentified unknowns)
    x = np.zeros(P)
    L0 = float(np.linalg.norm((prior.E0 - prior.S0) - ((prior.E0 - prior.S0) @ f0["n0"]) * f0["n0"]))
    for X in ("S", "E"):
        rs = [r["mean_m"] for r, o in zip(two, obs) if r["end"] == X and o]
        if rs:
            x[col["e_" + X]] = float(np.mean(rs))
    for X, Y, sg in (("S", "E", -1.0), ("E", "S", 1.0)):
        if not has[X]:
            bl = [r["bound_m"] for r in bounds if r["end"] == X]
            if has[Y]:
                x[col["e_" + X]] = x[col["e_" + Y]] + sg * Ld
            elif bl:
                x[col["e_" + X]] = min(bl) if X == "S" else max(bl)
            else:
                x[col["e_" + X]] = 0.0 if X == "S" else L0
    base = np.diag(np.where(np.arange(P) == jd, 0.0 if use_site else prior_prec[jd], prior_prec))
    wL = 0.0
    kpos = {j: i for i, j in enumerate(keep)}
    # one-sided readings (a leg whose groove segment reaches its edge — a dwell grab, the transit between the two sites, a pass that
    # stops inside the seam — bounds an end at that edge) are observations of their row like the two-sided readings: given δ and the
    # Gaussian part of the model each is a probit in δ (js_source.DeltaPrior factors), so they enter δ's posterior with its prior, not
    # only the end's grid at the nuisances' mode (below). The readings at rest (dwells, the transit's first and last profiles) carry
    # no δ themselves: an end cannot lie beyond a site where the groove was seen, so they bound how far a lag can have moved the ends
    # a moving pass read. They refine δ where it is in the model (its Gaussian prior, or readings that identify a flat lag); a flat
    # lag the readings leave unidentified stays free — interval bounds alone leave its location arbitrary (js_source.DeltaPrior).
    # 2026-10-13, test_m on the fake MoveIt arm (reference echo, one pass E→S crossing the ends at 30.0 / 28.9 mm/s): the design
    # length identified δ in the planning fusion only through that 1.2 mm/s difference (σ 4.6 s), the flat τ ≥ 0 took its mean to
    # 3.6 s, the plan ends ±90–133 mm, the next pass out of reach; the sites 17.6 mm inside the S end the pass read bound δ ≤ 0.59 s
    # (plan ends ±4.6 mm). Observations only (leg weight > 0, the leg this seam — as the two-sided readings), the same rows and soft
    # widths as the end's grid.
    bnd = [r for r in bounds if float(r.get("leg_w", 1.0)) > 0.0 and not r.get("leg_not_this_seam", False)
           and not ("beta:" + r["cls"] in col and not ident[col["beta:" + r["cls"]]])]
    B_rows = np.array([_reading_row(r, col, P, line) for r in bnd]).reshape(len(bnd), P)
    # a reading whose row touches an unknown outside the identified set (an end no two-sided reading reaches) carries nothing on δ:
    # that unknown takes it
    B_on = ~np.any(B_rows[:, rest] != 0.0, axis=1) if len(bnd) else np.zeros(0, bool)
    B_rows = B_rows[B_on]
    bnd = [r for r, on_ in zip(bnd, B_on) if on_]
    B_lim = np.array([r["bound_m"] + _known_shift(r, line) - h_mean_shift(r) for r in bnd])
    B_fl2 = np.array([float(r.get("floor_m", 0.0)) ** 2 for r in bnd])
    B_up = np.array([r["side"] == "upper" for r in bnd], dtype=bool)
    B_w = np.array([float(r.get("leg_w", 1.0)) for r in bnd])
    o_idx = np.array([j for j in keep if j != jd], dtype=int)
    B_o = B_rows[:, o_idx]

    def bound_factors(Nm, rhs, x) -> tuple:
        """The one-sided readings' likelihood of δ from the rest of the normal equations (Nm, rhs without δ's factor): given δ the
        other unknowns are Gaussian, a·x has mean α + β·δ and variance v, and an upper bound lim is Φ((lim − α − β·δ)/√(v + floor²))
        (a lower one the mirror) → ([(β, c, s, w)] (js_source.DeltaPrior factors), (m_L, V_L) — the Gaussian likelihood of δ from
        the same block solve (Schur complement), None when δ's block is not separable). A reading whose row the rest leaves
        unidentified given δ carries nothing on δ."""
        if not bnd or not o_idx.size:
            return [], None
        No = Nm[np.ix_(o_idx, o_idx)]
        r0 = rhs[o_idx] - Nm[np.ix_(o_idx, rest)] @ x[rest]
        b_ = Nm[o_idx, jd]
        dg = np.diag(No)
        chol = None
        if np.all(dg > 0):                              # the usual case: every other unknown identified given δ (priors on h, β)
            sc = 1.0 / np.sqrt(dg)
            try:
                chol = cho_factor(sc[:, None] * No * sc[None, :], lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                chol = None
        if chol is not None:                            # Jacobi-scaled Cholesky solve
            sol = sc[:, None] * cho_solve(chol, sc[:, None] * np.c_[r0, -b_, B_o.T], check_finite=False)
            x0, g = sol[:, 0], sol[:, 1]
            v = np.einsum("ij,ji->i", B_o, sol[:, 2:])
            sep = True
        else:                                           # a direction the rest leaves unidentified: the identified-subspace inverse
            Io = _NormalInverse(No)
            x0 = Io.step(r0, x[o_idx])
            g = Io.pinv @ (-b_)
            v = np.array([Io.var(a_) for a_ in B_o])
            sep = Io.identified(b_)                     # δ's coupling to the others lies in what they identify (PSD)
        al = B_o @ x0
        be = B_o @ g + B_rows[:, jd]
        sd = np.sqrt(np.maximum(v, 0.0) + B_fl2)
        fac = [((be[i], B_lim[i] - al[i], sd[i], B_w[i]) if B_up[i] else (-be[i], al[i] - B_lim[i], sd[i], B_w[i]))
               for i in range(len(bnd)) if math.isfinite(v[i]) and sd[i] > 0.0]
        sch = None
        if sep:                                         # δ's own block: N_δδ − N_δo N_oo⁻ N_oδ
            S_ = float(Nm[jd, jd] + b_ @ g)
            r_d = float(rhs[jd] - Nm[jd, rest] @ x[rest])
            sch = ((r_d - float(b_ @ x0)) / S_, 1.0 / S_) if S_ > REL_NULL * float(Nm[jd, jd]) else (0.0, math.inf)
        return fac, sch

    def delta_site(Nm, rhs, x, out: Optional[dict] = None):
        """The δ factor's EP site (λ, λm) from the rest of the normal equations (Nm, rhs without it — the data's own Gaussian
        likelihood of δ, N(m_L, V_L), V_L ∞ when nothing else informs δ) and the one-sided readings' factors on δ (bound_factors):
        the cut mixture's posterior moments (js_source.DeltaPrior). Measured source alone and no one-sided reading that changes its
        posterior: the prior itself, N(0, σ_q²)."""
        if not use_site or jd not in kpos:
            return 0.0, 0.0
        if not dpri.p_ref > 0.0 and not bnd:
            return 1.0 / dt_sig ** 2, 0.0
        fac, sch = bound_factors(Nm, rhs, x)
        if not dpri.p_ref > 0.0 and sch is not None and out is None:
            # the measured source alone: the data's Gaussian on δ from the same block solve; no one-sided reading that changes
            # N(0, σ_q²) × data (the usual case): the prior's own site, exactly as without them
            return dpri.site(sch[0], sch[1], fac)
        Ik = _NormalInverse(Nm[np.ix_(keep, keep)])
        e_ = np.zeros(keep.size)
        e_[kpos[jd]] = 1.0
        V_L = Ik.var(e_)
        m_L = 0.0
        if math.isfinite(V_L):
            m_L = float(Ik.step(rhs[keep] - Nm[np.ix_(keep, rest)] @ x[rest], x[keep])[kpos[jd]])
        lam, lam_m = dpri.site(m_L, V_L, fac)
        if out is not None:
            out.update(m_L=m_L, V_L=V_L, factors=fac)
        return lam, lam_m

    def with_site(Nm, rhs, x):
        lam, lam_m = delta_site(Nm, rhs, x)
        if lam == 0.0 and lam_m == 0.0:
            return Nm, rhs
        Nm, rhs = Nm.copy(), rhs.copy()
        Nm[jd, jd] += lam
        rhs[jd] += lam_m
        return Nm, rhs

    def kappa(x):
        """arc length per unit s between the current ends (Simpson; exact for a straight line)."""
        a_, b_ = float(x[col["e_S"]]), float(x[col["e_E"]])
        tt = np.linalg.norm(L.t(np.array([a_, 0.5 * (a_ + b_), b_])), axis=-1)
        return float((tt[0] + 4 * tt[1] + tt[2]) / 6.0)

    def normal_eq(x, curv=False, W_=None, design=True):
        W_ = W if W_ is None else W_
        lrow[col["e_S"]], lrow[col["e_E"]] = -kappa(x), kappa(x)
        Nm = base.copy()
        rhs = np.zeros(P)
        if len(two):
            Nm += A.T @ (W_[:, None] * A)
            rhs += A.T @ (W_ * y)
        wl = 0.0
        if use_len and design:
            rL = float(lrow @ x) - Ld
            wl = (max(2.0 * (gam ** 2 - rL ** 2) / (gam ** 2 + rL ** 2) ** 2, 0.0) if curv
                  else 2.0 / (gam ** 2 + rL ** 2))
            Nm += wl * np.outer(lrow, lrow)
            rhs += (0.0 if curv else wl) * lrow * Ld
        return Nm, rhs, wl

    def solve_x(x):
        wl_ = 0.0
        for _ in range(CAUCHY_ITERS):                   # IRLS for the Cauchy length prior
            Nm, rhs, wl_ = normal_eq(x)
            Nm, rhs = with_site(Nm, rhs, x)
            x_new = x.copy()
            if keep.size:
                Nk = Nm[np.ix_(keep, keep)]
                rk = rhs[keep] - Nm[np.ix_(keep, rest)] @ x[rest]
                # every reading of a direction down-weighted to 0 (§4.9 indicators): that direction has no information — it keeps
                # its value (the start / convention), the others are solved
                x_new[keep] = _NormalInverse(Nk).step(rk, x[keep])
            done_ = float(np.max(np.abs(x_new - x))) < 1e-12
            x = x_new
            if done_:
                break
        return x, wl_

    # robust rounds: start at τ_e = 0 (gross readings stand out), Huber weights on the leave-one-out standardised residuals, τ_e from
    # the bounded-influence score on the gauge readings' leave-one-out residuals, until the weights settle
    def loo_z(x, tau_=None, hw_=None, design=True):
        """Leave-one-out standardised residual of every two-sided reading (prediction from all other readings and the priors,
        variance a·C₋ᵢ·aᵀ + σ_i² + τ²): a gross reading cannot hide in the soft nuisances (h, β_u) it would otherwise pull.
        Returns (z, v_pred) — v_pred = ∞ for a reading nothing else predicts."""
        tau_ = tau if tau_ is None else tau_
        W_ = (hw if hw_ is None else hw_) / (vcp + tau_ ** 2)
        Nm, rhs, _ = normal_eq(x, W_=W_, design=design)
        z = np.zeros(len(two))
        vp = np.full(len(two), np.inf)
        for i in range(len(two)):
            a = A[i]
            Ni = Nm - W_[i] * np.outer(a, a)
            ri = rhs - W_[i] * a * y[i]
            Ni, ri = with_site(Ni, ri, x)               # the δ factor given the other readings (its site from their likelihood)
            Nk = Ni[np.ix_(keep, keep)]
            Ik = _NormalInverse(Nk)
            v_a = Ik.var(a[keep])                       # ∞: the other readings and the priors do not identify this reading
            if not math.isfinite(v_a):
                continue
            xi = x.copy()
            xi[keep] = Ik.step(ri[keep] - Ni[np.ix_(keep, rest)] @ x[rest], x[keep])
            vpred = v_a + vcp[i] + tau_ ** 2
            if vpred > 0 and math.isfinite(vpred):
                z[i] = (y[i] - float(a @ xi)) / math.sqrt(vpred)
                vp[i] = vpred
        return z, vp

    cH = HUBER_C
    beta_H = (2 * float(norm.cdf(cH)) - 1) - 2 * cH * float(norm.pdf(cH)) + 2 * cH * cH * (1 - float(norm.cdf(cH)))   # E[min(Z², c²)]
    g_rd = np.zeros(len(two))                             # §4.9 segment indicator of each gauge reading (posterior P(outlier))
    rd_chk = {"lo": np.full(len(two), np.nan), "z": np.full(len(two), np.nan), "eps": float("nan")}

    def indicators(z, vp):
        """§4.9 model fit of each reading (the end region of one leg): H_fit = the leave-one-out posterior predictive N(0, v_pred)
        (reading σ, τ_e, the other readings, δ, offsets, end heights, length prior), H_out = the change point anywhere along the leg
        (uniform over its span) — the family's γ (the reading's EM weight is 1 − γ) and ln O."""
        g_ = np.zeros(len(two))
        on = np.array([i for i in range(len(two)) if np.isfinite(vp[i]) and vp[i] > 0 and two[i].get("span_m", 0.0) > 0.0], dtype=int)
        if not on.size:
            return g_, {"lo": np.full(len(two), np.nan), "z": np.full(len(two), np.nan), "eps": float("nan")}
        lnf0 = -0.5 * np.log(2.0 * math.pi * vp[on])
        lnf = -0.5 * z[on] ** 2 + lnf0
        ext = np.log([float(two[i]["span_m"]) for i in on])
        lo_, g_on, eps_ = segment_mixture(lnf, ext, g_rd[on], lnf_peak=lnf0)
        g_[on] = g_on
        lo_all = np.full(len(two), np.nan)
        lo_all[on] = lo_
        z_all = np.full(len(two), np.nan)
        z_all[on] = np.abs(z[on])
        return g_, {"lo": lo_all, "z": z_all, "eps": eps_}

    def tau_score(x, hw_):
        """τ_e (end scatter of the gauge-class readings) from the bounded-influence, precision-weighted score on the leave-one-out
        residuals: Σ_gauge (min(z_i², c²) − E[min(Z², c²)]) / v_pred,i = 0 (the same Huberised variance score as the station dims;
        z_i is exactly N(0, 1) under the model — the leave-one-out already accounts for the fitted unknowns). One gross reading
        adds at most c² and cannot inflate τ_e for all (lab at10: a stationary pass beyond the end read 'S = E = 376 mm' and made a
        REML τ_e of 182 mm). A reading nothing else predicts carries no weight. Returns (τ, identified)."""
        def f(t):
            z_, vp_ = loo_z(x, t, hw_)
            g = np.isfinite(vp_[gidx])
            if not np.any(g):
                return None
            # τ_e is the readings' shared scatter, bounded-influence on ALL readings of legs that show this seam — the readings' own
            # indicators do not shrink it (else the indicators and τ_e feed each other: swamping, one reading left)
            zz, vv, ww_ = z_[gidx][g], vp_[gidx][g], leg_w[gidx][g]
            return float(np.sum(ww_ * (np.minimum(zz * zz, cH * cH) - beta_H) / vv))
        f0 = f(0.0)
        if f0 is None:
            return 0.0, False
        if f0 <= 0.0:
            return 0.0, True
        z0, vp0 = loo_z(x, 0.0, hw_)
        g = np.isfinite(vp0[gidx])
        hi = float(np.max(np.abs(z0[gidx][g]) * np.sqrt(vp0[gidx][g]))) + 1e-9    # largest LOO residual: f(hi) < 0
        lo = 0.0
        for _it in range(GOLDEN_ITERS + 20):            # [R] bisection steps, to 0.1 % of τ
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if fm is not None and fm > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-3 * hi:                    # [R]
                break
        return 0.5 * (lo + hi), True

    # §4.9 the leg's own segment weight (1 − γ of its indicators); 0 for a reading that is no observation (above)
    leg_w = np.array([float(r.get("leg_w", 1.0)) if o else 0.0 for r, o in zip(two, obs)])
    seg_w = leg_w.copy()                                # §4.9 EM weights of the readings (1 − γ), fixed within one robust fit
    tau = 0.0 if tau_m is None else float(tau_m)
    hw_h = np.ones(len(two))
    settled = False
    for _em in range(EM_ITERS):
        # the first robust fit starts at τ_e = 0 (gross readings stand out) and runs to convergence; later M-steps start from the
        # previous fit and take one IRLS round each (ECM, Meng & Rubin 1993: one conditional maximisation per E-step converges to
        # the same fixed point); the EM ends when the weights AND the robust fit have settled
        hw = hw_h * seg_w
        W = hw / (vcp + tau ** 2) if len(two) else np.zeros(0)
        for _ in range(IRLS_ROUNDS if _em == 0 else 1):
            x, wL = solve_x(x)
            if not len(two):
                break
            z, _vp = loo_z(x)
            hw_h = _huber_w(np.abs(z))
            hw_new = hw_h * seg_w
            tau_new = tau
            # τ_e is a variance component of the gauge readings: with no residual degree of freedom (dof 0 — as many fixed unknowns
            # as readings, e.g. one pass) the readings say nothing about their scatter, and the score would take it from what else
            # predicts them — the design length's prior (planning fusion) or δ's prior: the measured-vs-design length mismatch read as
            # reading scatter (2026-10-13: measured runs after pass 1, τ_e 1.48 mm, plan σ 1.5–1.9 mm instead of 0.4). Unidentified,
            # it is not estimated: σ given τ_e = 0, `tau_identified` False (this function's contract; the done test reports such ends
            # unresolved)
            if tau_m is None and gidx.size and dof > 0:
                tau_new, tau_ident = tau_score(x, hw_h * leg_w)
            settled = float(np.max(np.abs(hw_new - hw))) < 1e-6 and abs(tau_new - tau) <= 1e-3 * max(tau, 1e-6)
            hw, tau = hw_new, tau_new
            W = hw / (vcp + tau ** 2)
            if settled:
                break
        x, wL = solve_x(x)
        if not len(two):
            break
        # E-step: each reading's indicator from its leave-one-out residual against the robust fit (the scatter the readings share,
        # τ_e, is not an outlier); M-step: the robust fit with the readings weighted by 1 − γ. The predictive is the OTHER
        # MEASUREMENTS only (their robust Huber weights, not their segment weights — no feedback through the indicators) and never
        # the design length: a reading is judged against measurements; how far the measured joint is from the design is a warning
        # (design_deviations), not a misfit (user rule 2026-10-11). A reading nothing else measures is untestable.
        z2, vp2 = loo_z(x, tau, hw_h * leg_w, design=False)
        g_rd, rd_chk = indicators(z2, vp2)
        new_w = leg_w * (1.0 - g_rd)
        if float(np.max(np.abs(new_w - seg_w))) < REWEIGHT_TOL and settled:
            break
        seg_w = new_w
    # the model's terms are those the KEPT observations identify: a reading the indicators removed outright (weight 0 to double
    # precision, 2 ln O below about −74) is no observation either — when its removal changes what the observations identify (b_u,
    # an end read), the fusion is redone with it as a non-observation (never a Laplace approximation along a direction that no kept
    # reading informs, nor a mode placed there by readings the fit rejected)
    gone = [i for i in range(len(two)) if obs[i] and not hw[i] > 0]
    if gone:
        o2 = obs.copy()
        o2[gone] = False
        seen2 = (A[o2] != 0).any(axis=0) if np.any(o2) else np.zeros(P, bool)
        if _b_u_identified([r for r, o in zip(two, o2) if o], line) != ("b_u" in col) or np.any(seen2 != seen):
            return fuse_ends(readings, prior, line, tau_m=tau_m, length_prior=length_prior,
                             _not_obs=frozenset(_not_obs) | {id(two[i]) for i in gone})
    k_chk = float(prior.k)
    for i, (r, wgt) in enumerate(zip(two, hw)):
        r["weight"] = float(wgt)
        if math.isfinite(rd_chk["z"][i]):
            r["fit"] = {"z": float(rd_chk["z"][i]), "two_ln_odds": 2.0 * float(rd_chk["lo"][i]), "gamma_out": float(g_rd[i]),
                        "verdict": segment_verdict(2.0 * float(rd_chk["lo"][i]), k_chk)}
    H, _, _ = normal_eq(x, curv=True)                   # Laplace precision (Cauchy with its true curvature, ≥ 0)
    site_rec: dict = {}
    Nm_, rhs_, _ = normal_eq(x)
    lam_d, lam_dm = delta_site(Nm_, rhs_, x, out=site_rec)      # the δ factor's site at the solution (as solve_x used it)
    H = H.copy()
    H[jd, jd] += lam_d
    # Laplace covariance on the subspace the weighted readings identify (_NormalInverse): an unknown or a functional with a
    # component along a direction without information has an infinite variance (never a negative one from inverting a singular H)
    C = np.full((P, P), np.inf)
    Hinv = _NormalInverse(H[np.ix_(keep, keep)]) if keep.size else None

    def var_of(a: np.ndarray) -> float:
        """Posterior variance of a·x (a over all P unknowns; ∞ where it touches an unknown outside the model's identified set)."""
        a = np.asarray(a, dtype=float)
        if np.any(a[rest] != 0) or Hinv is None:
            return math.inf if np.any(a != 0) else 0.0
        return Hinv.var(a[keep])
    if keep.size:
        C[np.ix_(keep, keep)] = Hinv.cov()
    var = np.diag(C).copy()
    kap = kappa(x)
    k_ = float(prior.k)
    span = (Ld + k_ * (gam or (prior.tol.end_position_mm or 1.0) * 1e-3)) / kap    # design length + k·tol in s units
    # §4.4 grid posterior of each end given the nuisances at their mode: its own readings (Gaussian gap part ⊕ τ_e and the
    # "beyond the leg's edge" tail; bounds soft by floor ⊕ nuisance σ) × the Cauchy length prior marginalised over the OTHER end's
    # reading likelihood (FFT cross-correlation on a common uniform grid) — an end never bracketed on both sides keeps the width
    # its readings, bounds and the design length leave it (each ≈ uniform over the remaining length before any pass).
    parts = {}
    for X in ("S", "E"):
        j = col["e_" + X]
        terms, pts = [], []
        for i, r in enumerate(two):
            if r["end"] != X or not hw[i] > 0:
                continue
            m_e = float(y[i] - (A[i] @ x - x[j]))
            sd = math.sqrt(vcp[i] + tau ** 2)
            cv = r.get("_curve") or {}
            wdt = np.maximum(np.asarray(cv.get("hi", [])) - np.asarray(cv.get("lo", [])), math.sqrt(12.0) * r.get("floor_m", 0.0))
            lM = float(np.log(np.sum(np.exp(np.asarray(cv.get("logL", [0.0]))) * (wdt if wdt.size else 1.0))))
            tl = cv.get("tail")
            edge = None if tl is None else float(tl[1]) + (m_e - float(r["mean_m"]))
            terms.append(("two", float(hw[i]), m_e, sd, lM, tl, edge, max(r.get("floor_m", 0.0), 1e-9)))
            pts.append((m_e, sd))
        for r in bounds:
            if r["end"] != X or ("beta:" + r["cls"] in col and not ident[col["beta:" + r["cls"]]]):
                continue
            ar = _reading_row(r, col, P, line)
            lim = r["bound_m"] + _known_shift(r, line) - h_mean_shift(r) - (float(ar @ x) - x[j])
            ar[j] = 0.0
            s2 = var_of(ar)                             # the nuisances' share of the bound (0 when it has none)
            if not math.isfinite(s2):
                continue
            sdb = math.sqrt(s2 + r.get("floor_m", 0.0) ** 2) + 1e-9
            terms.append(("bound", lim, sdb, r["side"] == "upper", float(r.get("leg_w", 1.0))))
            pts.append((lim, sdb))
        parts[X] = (terms, pts)
    # domains: the contract (taught point on the specimen) and design length + k·tol from the other end's support, widened to
    # every reading
    supp = {X: ([m_ for m_, _ in parts[X][1]] or [0.0 if X == "S" else L0]) for X in ("S", "E")}
    dom = {"S": [min(supp["E"]) - span, 0.0], "E": [L0, max(supp["S"]) + span]}
    for X in ("S", "E"):
        for m_, sd_ in parts[X][1]:
            dom[X] = [min(dom[X][0], m_ - 10 * sd_), max(dom[X][1], m_ + 10 * sd_)]
    sds = [sd_ for X in ("S", "E") for _, sd_ in parts[X][1]] + ([gam / kap] if gam else [])
    h = min(sds) / 10.0 if sds else (dom["S"][1] - dom["S"][0]) / 4000.0                    # [R] grid step
    widest = max(dom["S"][1] - dom["S"][0], dom["E"][1] - dom["E"][0])
    h = max(h, widest / 2 ** 17)                                                               # [R] grid size
    grids = {X: dom[X][0] + h * np.arange(int(math.ceil((dom[X][1] - dom[X][0]) / h)) + 1) for X in ("S", "E")}

    def own_loglik(X, e):
        lp = np.zeros_like(e)
        for t in parts[X][0]:
            if t[0] == "two":
                _, wgt, m_e, sd, lM, tl, edge, fl = t
                lg = lM + norm.logpdf(e, m_e, sd)
                if tl is not None:
                    side = (edge - e) if tl[0] == "below" else (e - edge)
                    lg = np.logaddexp(lg, float(tl[2]) + log_ndtr(side / fl))
                lp += wgt * lg
            else:
                _, lim, sdb, up, lw = t
                lp += lw * log_ndtr(((lim - e) if up else (e - lim)) / sdb)
        return lp

    lr = {X: own_loglik(X, grids[X]) for X in ("S", "E")}
    post = dict(lr)
    if gam is not None:
        from scipy.signal import fftconvolve
        nS, nE = grids["S"].size, grids["E"].size
        d = (dom["E"][0] - dom["S"][0]) + h * np.arange(-(nS - 1), nE)                         # e_E − e_S for every index pair
        kern = 1.0 / (1.0 + ((kap * d - Ld) / gam) ** 2)
        LS = np.exp(lr["S"] - np.max(lr["S"]))
        LE = np.exp(lr["E"] - np.max(lr["E"]))
        CS = fftconvolve(LE, kern[::-1])[nE - 1:nE - 1 + nS]
        CE = fftconvolve(LS, kern)[nS - 1:nS - 1 + nE]
        post["S"] = lr["S"] + np.log(np.maximum(CS, 1e-300))
        post["E"] = lr["E"] + np.log(np.maximum(CE, 1e-300))
    ends = {}
    for X in ("S", "E"):
        j = col["e_" + X]
        v_marg = float(var[j])
        v_cond = 1.0 / H[j, j] if (ident[j] and H[j, j] > 0) else math.inf
        add = max(v_marg - v_cond, 0.0) if (math.isfinite(v_marg) and math.isfinite(v_cond)) else 0.0
        e = grids[X]
        wpost = np.exp(post[X] - np.max(post[X]))
        Z = float(np.sum(wpost))
        mu = float(np.sum(wpost * e) / Z)
        vg = float(np.sum(wpost * (e - mu) ** 2) / Z) + h * h / 12.0
        sig = math.sqrt(max(vg + add, 0.0))
        rs = [r for r in rd if r["end"] == X]
        # an observation of the end = a two-sided reading that the §4.9 decisions keep (its leg is this seam, the reading fits)
        obs = [r for r in rs if r["kind"] == "two_sided" and (r.get("fit") or {}).get("verdict") != "misfit"
               and not r.get("leg_not_this_seam", False)]
        ends[X] = {"s_m": mu, "sigma_m": sig, "sigma_given_tau_m": sig,
                   "identified": bool(any(t[0] == "two" for t in parts[X][0])) and bool(obs),
                   "two_sided": bool(obs),
                   "n_legs": len({r["leg"] for r in obs}),
                   "readings": rs, "point_m": L.r(mu)}
    dj = col["delta"]
    jb = col.get("b_u")
    i_s, i_e = col["e_S"], col["e_E"]
    a_len = np.zeros(P)
    a_len[i_s], a_len[i_e] = -1.0, 1.0
    vl = var_of(a_len)
    return {"ends": ends, "delta_s": float(x[dj]), "delta_sigma_s": math.sqrt(var[dj]) if math.isfinite(var[dj]) else math.inf,
            "tau_m": tau if tau_ident else float("nan"), "tau_identified": bool(tau_ident), "tau_dof": int(dof), "x": x,
            "cov": C, "names": names,
            "end_height_m": {X: 0.5 * D_d + float(x[col["h_" + X]]) for X in ("S", "E")},
            "end_height_sigma_m": {X: math.sqrt(var[col["h_" + X]]) if math.isfinite(var[col["h_" + X]]) else math.inf
                                   for X in ("S", "E")},
            "beta_u": {nm.split(":", 1)[1]: float(x[j]) for nm, j in col.items() if nm.startswith("beta:")},
            "eta_u": {nm.split(":", 1)[1]: float(x[j]) for nm, j in col.items() if nm.startswith("eta:")},
            "b_u": float(x[jb]) if jb is not None else 0.0,
            "b_u_sigma": (math.sqrt(var[jb]) if math.isfinite(var[jb]) else math.inf) if jb is not None else math.inf,
            "b_u_identified": jb is not None, "delta_prior_sigma_s": dt_sig, "delta_prior": dpri.as_dict(),
            "delta_ref_likelihood": _ref_likelihood(dpri, site_rec),
            "delta_site": {"precision": lam_d, "mean_s": (lam_dm / lam_d) if lam_d != 0.0 else float("nan"),
                           "data_mean_s": site_rec.get("m_L", float("nan")), "data_sigma_s": math.sqrt(site_rec["V_L"])
                           if math.isfinite(site_rec.get("V_L", math.inf)) else math.inf,
                           "one_sided_readings": len(site_rec.get("factors") or ())},
            "moving_readings": bool(any(leg_direction(line, r["leg"]) != 0 for r in two)),
            "length_m": kap * float(ends["E"]["s_m"] - ends["S"]["s_m"]),
            "length_sigma_m": kap * math.sqrt(max(vl, 0.0)) if math.isfinite(vl) else
            kap * math.hypot(ends["S"]["sigma_m"], ends["E"]["sigma_m"]),
            "length_weight": wL, "reading_outlier_eps": rd_chk["eps"]}


# ====================================================================================================================
# robust weighted means with a REML variance component (vectorised over rows)
# ====================================================================================================================
def _reml_rows(y, v, w0, huber_rounds: int = 4):
    """Rows of (y, v = σ_i², w0 = prior weight, 0 = absent): robust weighted mean with weights w0·h/(v + τ²) (Huber h on the
    standardised residuals) and the variance component τ² from the Huberised, precision-weighted ML score
    Σ w0·(min(z², c²) − β')/(v + τ²) = 0, β' = E[min(Z², c²)]·(n−1)/n (Gaussian consistency, REML dof) — bounded influence, and entries
    without information get no weight. Robust start: weighted median and 1.4826·MAD; bracket from the median scale. Returns
    (μ, σ_μ, τ, n_eff)."""
    y = np.where(w0 > 0, y, 0.0)
    v = np.where(w0 > 0, v, 1.0)
    K = y.shape[0]
    c = HUBER_C
    beta = (2 * float(norm.cdf(c)) - 1) - 2 * c * float(norm.pdf(c)) + 2 * c * c * (1 - float(norm.cdf(c)))
    n_eff = (w0 > 0).sum(1).astype(float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ym = np.where(w0 > 0, y, np.nan)
        mu = np.nanmedian(ym, axis=1)
        mad = MAD_SCALE * np.nanmedian(np.abs(ym - mu[:, None]), axis=1)
        sv = np.sqrt(np.nanmedian(np.where(w0 > 0, v, np.nan), axis=1))
    mu = np.where(np.isfinite(mu), mu, 0.0)
    scale = np.where(np.isfinite(mad), mad, 0.0) + np.where(np.isfinite(sv), sv, 0.0)
    scale = np.where(scale > 0, scale, 1.0)
    tau = np.where(np.isfinite(mad), mad, 0.0)

    def zsq(t, m):
        return np.where(w0 > 0, (y - m[:, None]) ** 2 / (v + t[:, None] ** 2), 0.0)

    for _ in range(huber_rounds):
        h = np.where(w0 > 0, _huber_w(np.sqrt(zsq(tau, mu))), 0.0)
        wt = w0 * h / (v + tau[:, None] ** 2)
        sw = wt.sum(1)
        mu = np.where(sw > 0, (wt * y).sum(1) / np.where(sw > 0, sw, 1.0), mu)
        bdof = beta * np.maximum(n_eff - 1.0, 0.0) / np.maximum(n_eff, 1.0)            # β with the REML dof, per entry

        def f(t):
            # Huberised ML score for τ²: Σ w0·(min(z², c²) − β')/(v + τ²) — precision-weighted, so an entry without information
            # (huge v) contributes neither residual nor dof, and no single entry sets the bracket
            return (w0 * (np.minimum(zsq(t, mu), c * c) - bdof[:, None]) / (v + t[:, None] ** 2)).sum(1)

        lo = np.zeros(K)
        hi = scale * 1e3
        f0 = f(lo)
        for _it in range(GOLDEN_ITERS + 20):            # [R] bisection steps
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            lo = np.where(fm > 0, mid, lo)
            hi = np.where(fm > 0, hi, mid)
        tau = np.where((f0 > 0) & (n_eff >= 2), 0.5 * (lo + hi), 0.0)
    h = np.where(w0 > 0, _huber_w(np.sqrt(zsq(tau, mu))), 0.0)
    wt = w0 * h / (v + tau[:, None] ** 2)
    sw = wt.sum(1)
    mu = np.where(sw > 0, (wt * y).sum(1) / np.where(sw > 0, sw, 1.0), np.nan)
    with np.errstate(divide="ignore"):
        var_mu = np.where(sw > 0, 1.0 / np.maximum(sw, 1e-300), np.inf)
    return mu, np.sqrt(var_mu), tau, w0.sum(1)


# ====================================================================================================================
# plate normals (§4.2)
# ====================================================================================================================
def _plate_angles(geo: dict, sel: np.ndarray, cov: np.ndarray, face: str, uhat, n_ref, w_ref) -> tuple:
    """Angle φ (rad) of a plate-top normal about û (from n_ref toward w_ref) per selected profile, and its variance (∂φ/∂d by finite
    differences on the 3-D direction [R] · dU/dθ analytic)."""
    f = geo["names_f"].index(face)
    d, dd = geo["U"][sel][:, f], geo["dU"][sel][:, f]

    def phi(dv):
        nrm = np.cross(dv, uhat)
        nrm = nrm * np.where((nrm @ n_ref) >= 0, 1.0, -1.0)[:, None]
        return np.arctan2(nrm @ w_ref, nrm @ n_ref)

    p0 = phi(d)
    Jd = np.zeros((len(d), 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = FD_STEP
        Jd[:, j] = (phi(d + e) - phi(d - e)) / (2 * FD_STEP)
    Jt = np.einsum("mi,mip->mp", Jd, dd)
    return p0, np.einsum("mp,mpr,mr->m", Jt, cov, Jt)


def plate_normals(prof: _Profiles, line: dict, prior: SeamPrior, pi: np.ndarray) -> dict:
    """Plate-top normals (left/right by geometry: +w = left looking S→E), roll about û vs up, angular misalignment."""
    L = _line_from(line)
    f0 = line["frame0"]
    s_mid = float(np.median(line["s_star"][prof.ok])) if np.any(prof.ok) else 0.0
    uhat = L.tangent(s_mid)
    _, n_ref, w_ref = _section_basis(uhat, f0["n0"])
    sel = prof.ok
    out = {}
    geo = prof.geometry(prior.spec)
    Lleft = geo["L_left"][sel]
    for face, left in (("top_L", Lleft), ("top_R", ~Lleft)):
        phi, vphi = _plate_angles(geo, sel, prof.cov[sel], face, uhat, n_ref, w_ref)
        for name, msk in (("left", left), ("right", ~left)):
            w0 = np.where(msk & np.isfinite(phi) & np.isfinite(vphi), pi[sel], 0.0)
            if np.sum(w0) <= 0 or (name in out and out[name]["n"] >= float(np.sum(w0))):
                continue
            mu, smu, tau, neff = _reml_rows(np.where(w0 > 0, phi, 0.0)[None, :], np.where(w0 > 0, vphi, 1.0)[None, :],
                                            w0[None, :])
            out[name] = {"phi": float(mu[0]), "sigma": float(smu[0]), "tau": float(tau[0]), "n": float(neff[0])}
    res = {}
    nn = {}
    for name in ("left", "right"):
        if name in out:
            ph = out[name]["phi"]
            nn[name] = math.cos(ph) * n_ref + math.sin(ph) * w_ref
    n_hat = _unit(sum(nn.values())) if nn else n_ref
    up = _unit(prior.up)
    up_perp = _unit(up - float(up @ uhat) * uhat)
    roll = math.degrees(math.atan2(float(np.cross(up_perp, n_hat) @ uhat), float(up_perp @ n_hat)))
    sig_roll = math.degrees(math.sqrt(np.mean([out[k]["sigma"] ** 2 for k in out]) / max(len(out), 1))) if out else math.inf
    mis = math.degrees(math.acos(max(-1.0, min(1.0, float(nn["left"] @ nn["right"]))))) if len(nn) == 2 else float("nan")
    sig_mis = math.degrees(math.sqrt(sum(out[k]["sigma"] ** 2 for k in out))) if len(out) == 2 else float("nan")
    res.update({"n_left": nn.get("left"), "n_right": nn.get("right"), "normal": n_hat, "roll_deg": roll,
                "misalignment_deg": mis, "sigma_deg": {"left": math.degrees(out["left"]["sigma"]) if "left" in out else None,
                                                      "right": math.degrees(out["right"]["sigma"]) if "right" in out else None,
                                                      "roll": sig_roll, "misalignment": sig_mis},
                "scatter_deg": {k: math.degrees(out[k]["tau"]) for k in out}, "u": uhat})
    return res


# ====================================================================================================================
# §4.6 stations
# ====================================================================================================================
_FACE_BASE = {"top": "plate_top", "bevel": "bevel", "wall": "wall"}


def _faces_profile(prof: _Profiles, spec, offset3=None) -> dict:
    """Per-profile face geometry from WP1's template: {FACE_NAMES name: (anchor point base (N,3), unit direction (N,3), valid (N,),
    anchor θ-Jacobian (N,3,10))}; L/R → left/right per profile by its own corners (+w = left looking S→E); points corrected by the
    leg/class offsets."""
    geo = prof.geometry(spec)
    off = np.zeros((prof.N, 3)) if offset3 is None else offset3
    Lleft = geo["L_left"]
    faces = {}
    for f, fname in enumerate(geo["names_f"]):
        if fname.startswith("root_face"):
            continue                                      # root face is in the dims block (§4.6)
        P, D, dP = geo["A"][:, f] + off, geo["U"][:, f], geo["dA"][:, f]
        if fname == "gap_bottom":
            faces["gap_bottom"] = (P, D, np.ones(prof.N, bool), dP)
            continue
        base, side = fname.rsplit("_", 1)
        is_left = Lleft if side == "L" else ~Lleft
        for nm, msk in ((_FACE_BASE[base] + "_left", is_left), (_FACE_BASE[base] + "_right", ~is_left)):
            if nm in faces:
                P0, D0, M0, dP0 = faces[nm]
                faces[nm] = (np.where(msk[:, None], P, P0), np.where(msk[:, None], D, D0), M0 | msk,
                             np.where(msk[:, None, None], dP, dP0))
            else:
                faces[nm] = (np.where(msk[:, None], P, np.nan), np.where(msk[:, None], D, np.nan), msk.copy(),
                             np.where(msk[:, None, None], dP, np.nan))
    return faces


def station_sections(legs: list, line: dict, ends: dict, prior: SeamPrior, *, _prof: Optional[_Profiles] = None,
                     s_true: Optional[np.ndarray] = None, n_plate: Optional[np.ndarray] = None,
                     pi: Optional[np.ndarray] = None, offset3: Optional[np.ndarray] = None, pi_check: Optional[np.ndarray] = None,
                     win_w: Optional[np.ndarray] = None, win_out: Optional[dict] = None) -> list:
    """§4.6: stations from e_S to e_E (uniform, step ≤ station_step), window station_window; per station the Huber/REML mean of
    every profile's exact section quantities, face normals/lines, and the root point r(s_j) with its line σ.
    §4.9 inputs/outputs: pi_check = the per-profile 'this seam' weights WITHOUT the segment EM weights (the window statistics are the
    window's own data against a line fitted with the window down-weighted — cross-validated); win_w = per-station weight (1 − γ of the
    window's outlier indicator) in the REML of τ_root; win_out (dict) receives the window means, their σ, τ_root, the profile indices
    and the line σ at the stations."""
    L = _line_from(line)
    f0 = line["frame0"]
    prof = _prof if _prof is not None else (line.get("_prof") or _Profiles(legs, f0))
    s_prof = line["s_star"] if s_true is None else s_true
    pi = line["pi"] if pi is None else pi
    eS = ends["S"]["s_m"] if isinstance(ends["S"], dict) else ends["S"].s_m
    eE = ends["E"]["s_m"] if isinstance(ends["E"], dict) else ends["E"].s_m
    length = max(eE - eS, 0.0)
    step = float(prior.station_step_m)
    # stations at most one step apart along the seam itself, not along the taught coordinate: |dr/ds| >= 1 (s is the taught
    # coordinate; a line that turns off the taught direction is longer than its s span) and, for the polynomial line (d <= 2), it is
    # largest at an end of the span - so the arc between neighbours is <= step (the KIRO waypoint contract, <= 5 mm)
    speed = max(float(np.linalg.norm(L.t(np.asarray(eS, dtype=float)))), float(np.linalg.norm(L.t(np.asarray(eE, dtype=float)))))
    nst = int(math.ceil(length * speed / step - 1e-12)) + 1 if length > 0 else 1
    s_st = np.linspace(eS, eE, nst) if nst > 1 else np.array([eS])
    half = 0.5 * float(prior.station_window_m)
    s_mid = 0.5 * (eS + eE)
    uhat = L.tangent(s_mid)
    nh = n_plate if n_plate is not None else f0["n0"]
    u, n, w = _section_basis(uhat, nh)
    kind = prior.spec.kind
    sel = prof.ok
    K = len(s_st)
    # per-profile section quantities + first-order σ
    Q = np.full((prof.N, 6), np.nan)
    VQ = np.full((prof.N, 6), np.inf)
    if np.any(sel):
        q_, v_ = _section_quantities(prof.geometry(prior.spec), sel, prof.cov[sel], w, n, kind)
        Q[sel], VQ[sel] = q_, v_
    in_win = (np.abs(s_prof[None, :] - s_st[:, None]) <= half) & sel[None, :]          # (K, N)
    idx = [np.flatnonzero(in_win[j]) for j in range(K)]
    M = max((len(ix) for ix in idx), default=0)
    dims_val = np.full((K, 6), np.nan)
    dims_sig = np.full((K, 6), np.inf)
    dims_n = np.zeros((K, 6))
    # the section DIMENSIONS are the gauge class's (the passes): the dims likelihood takes only the gauge-class profiles. The other
    # pose classes (transit, dwells) see the section from another standoff, lateral position, yaw and incidence; their section
    # quantities carry pose-class systematics the model does not hold (lab 2026-10-12: kv_offA_a, rough points ±30 mm aside — gap
    # 6.00 / 5.74 / 5.40 mm and bevel 27.4 / 28.9 / 30.2° after 1 / 2 / 3 passes while the transit dominated; gap drifting 5.31 →
    # 5.21 mm over 1 … 6 passes on the aligned runs). Giving each coarse class its own dims offset instead would leave the gauge
    # value the same — a fixed offset per class absorbs that class's mean, so its profiles inform the gauge dims with nothing but
    # the offset itself — with extra unknowns, and an informative prior on those offsets would be a constant nobody calibrated.
    # The coarse classes keep their role in localisation (line, class offsets, faces / plate tops, ends); before the first pass
    # there are no dims (σ ∞ — the planning falls back to the design, plan_quantities / profile_predictor).
    gauge = line.get("gauge")
    w_dims = np.where(np.asarray(prof.cls) == gauge, pi, 0.0) if gauge is not None else np.zeros(prof.N)
    if M > 0:                                           # one stacked REML call: rows = (station, quantity)
        y = np.zeros((K, 6, M))
        v = np.ones((K, 6, M))
        w0 = np.zeros((K, 6, M))
        for j, ix in enumerate(idx):
            qq, vv = Q[ix].T, VQ[ix].T
            ok = np.isfinite(qq) & np.isfinite(vv)
            y[j, :, :len(ix)] = np.where(ok, qq, 0.0)
            v[j, :, :len(ix)] = np.where(ok, np.maximum(vv, 1e-12), 1.0)
            w0[j, :, :len(ix)] = np.where(ok, w_dims[ix][None, :], 0.0)
        mu, smu, tau, neff = _reml_rows(y.reshape(K * 6, M), v.reshape(K * 6, M), w0.reshape(K * 6, M))
        dims_val, dims_sig, dims_n = mu.reshape(K, 6), smu.reshape(K, 6), neff.reshape(K, 6)
    # σ_ε (sensor x-scale linearity, datasheet): not published → NaN → σ_sys of the lateral dims is NOT reported (NaN)
    sig_eps = float(prior.sigma_eps) if prior.sigma_eps is not None else float("nan")
    sw, sn = L.sigma_wn(s_st)
    # line misfit (seam waviness the line cannot carry): window means of the root residuals about the line, τ_root by REML across
    # the stations (each window mean with its own σ); root σ at a station = line σ ⊕ τ_root  (0 when the data show no structure)
    tau_root = (0.0, 0.0)
    pi_c = pi if pi_check is None else np.asarray(pi_check, dtype=float)
    if win_w is not None:
        ww = np.asarray(win_w, dtype=float)
    elif pi_check is not None:                          # the window's share of segment weight: Σ π·seg_w / Σ π (1 when nothing misfits)
        ww = np.array([float(np.sum(pi[ix])) / float(np.sum(pi_c[ix])) if ix.size and float(np.sum(pi_c[ix])) > 0 else 1.0
                       for ix in idx])
    else:
        ww = np.ones(K)
    mus, sws = np.full((K, 2), np.nan), np.full((K, 2), np.inf)
    if M > 0 and np.any(sel):
        off = np.zeros((prof.N, 3)) if offset3 is None else offset3
        qr = prof.r + off
        sr = L.s_star(qr, prof.m, s0=s_prof)
        dres = qr - L.r(sr)
        res2 = np.stack([dres @ w, dres @ n], axis=1)                                     # (N,2) m
        vr = np.array([prof.S2[:, 0, 0], prof.S2[:, 1, 1]]).T + np.asarray(
            [line["sigma_zeta"].get("dwell" if c.startswith("dwell") else c, 0.0) ** 2 for c in prof.cls])[:, None]
        tr = []
        for comp in range(2):
            y = np.zeros((K, M))
            v = np.ones((K, M))
            w0 = np.zeros((K, M))
            for j, ix in enumerate(idx):
                y[j, :len(ix)] = res2[ix, comp]
                v[j, :len(ix)] = vr[ix, comp]
                w0[j, :len(ix)] = pi_c[ix]
            mu_w, s_w, _, n_w = _reml_rows(y, v, w0)                                    # window means and their σ
            ok = (n_w > 0) & np.isfinite(mu_w) & np.isfinite(s_w)
            mus[:, comp] = np.where(ok, mu_w, np.nan)
            sws[:, comp] = np.where(ok, s_w, np.inf)
            okw = ok & (ww > 0)
            if np.count_nonzero(okw) >= 2:
                _, _, t_, _ = _reml_rows(mu_w[okw][None, :], (s_w[okw] ** 2)[None, :], ww[okw][None, :])
                tr.append(float(t_[0]))
            else:
                tr.append(0.0)
        tau_root = (tr[0], tr[1])
    if win_out is not None:
        win_out.update(s_st=s_st, idx=idx, mu_m=mus, sigma_m=sws, tau_root_m=tau_root, line_sigma_m=np.stack([sw, sn], axis=1),
                       w=w, n=n)
    sw = np.sqrt(sw ** 2 + tau_root[0] ** 2)
    sn = np.sqrt(sn ** 2 + tau_root[1] ** 2)
    sdu = L.sigma_dir(s_mid)
    sig_dir_deg = math.degrees(math.sqrt(float(sdu[0]) ** 2 + float(sdu[1]) ** 2))
    R_st = L.r(s_st)                                                                     # (K,3)
    faces = _faces_profile(prof, prior.spec, offset3=offset3)
    pad = np.full((K, max(M, 1)), -1, dtype=int)
    for j, ix in enumerate(idx):
        pad[j, :len(ix)] = ix
    has = pad >= 0
    padc = np.where(has, pad, 0)
    face_out = {}
    for fname in FACE_NAMES:
        if fname not in faces:
            continue
        Pf, Df, Mf, dPf = faces[fname]
        okf = Mf & np.all(np.isfinite(Df), axis=1) & sel
        if not np.any(okf):
            continue
        Pz = np.where(okf[:, None], Pf, prof.o)
        nrm = _unit(np.cross(np.where(okf[:, None], Df, u), u))
        nrm = nrm * np.where(np.einsum("ij,ij->i", nrm, prof.o - Pz) >= 0, 1.0, -1.0)[:, None]
        Wk = np.where(has, (pi * okf)[padc], 0.0)                                        # (K, M)
        mvec, sdeg, nk = _sphere_huber_rows(nrm[padc], Wk)
        # face point (w, n) per profile and its first-order (GUM) variance from the template anchor's θ-Jacobian; per station the
        # same Huber/REML mean as the section quantities (§4.6) - a profile whose anchor is ill-conditioned (two face lines near
        # parallel: the corner runs off to infinity, g1c_02 2026-10-10) carries its own huge variance and gets no weight
        with np.errstate(all="ignore"):
            pw = np.stack([Pz @ w, Pz @ n], axis=1) * 1e3                                       # (N,2) mm, absolute
            Jp = np.einsum("ki,nip->nkp", np.stack([w, n], axis=0) * 1e3, np.where(okf[:, None, None], dPf, 0.0))
            vp = np.einsum("nkp,npq,nkq->nk", Jp, prof.cov, Jp)                                  # (N,2) mm²
        okp = okf[:, None] & np.isfinite(pw) & np.isfinite(vp)
        y2 = np.where(okp, pw, 0.0)[padc].transpose(0, 2, 1)                                     # (K,2,M)
        v2 = np.where(okp, np.maximum(vp, 1e-12), 1.0)[padc].transpose(0, 2, 1)
        w2 = (np.where(okp, pi[:, None], 0.0)[padc] * has[:, :, None]).transpose(0, 2, 1)
        mu2, smu2, _, _ = _reml_rows(y2.reshape(K * 2, -1), v2.reshape(K * 2, -1), w2.reshape(K * 2, -1))
        has2 = w2.reshape(K * 2, -1).sum(1) > 0
        mu2 = np.where(has2, mu2, np.nan)                                                      # no usable profile: no point
        smu2 = np.where(has2 & np.isfinite(smu2), smu2, np.inf)                                # its σ (window mean, REML) [mm]
        dw = _unit(np.stack([np.where(okf, Df @ w, 1.0), np.where(okf, Df @ n, 0.0)], axis=1))
        ref = dw[np.argmax(pi * okf)]
        dw = (dw * np.where(dw @ ref >= 0, 1.0, -1.0)[:, None])[padc]
        with np.errstate(invalid="ignore", divide="ignore"):
            pm = mu2.reshape(K, 2) - np.stack([R_st @ w, R_st @ n], axis=1) * 1e3
            dm = _unit(np.einsum("km,kmi->ki", Wk, dw))
        face_out[fname] = (mvec, sdeg, nk, pm, dm, smu2.reshape(K, 2))
    stations = []
    for j in range(K):
        sj = float(s_st[j])
        rj = R_st[j]
        dims = {}
        for q, nm in enumerate(DIM_NAMES):
            val = float(dims_val[j, q])
            ss = float(dims_sig[j, q]) if dims_n[j, q] > 0 else math.inf
            sys_ = (abs(val) * sig_eps if math.isfinite(val) else float("nan")) if nm in ("gap_mm", "top_opening_mm") else 0.0
            dims[nm] = {"value": val, "sigma": ss, "sigma_sys": sys_, "n": float(dims_n[j, q])}
        fdict = {}
        for fname in FACE_NAMES:
            if fname not in face_out or face_out[fname][2][j] <= 0:
                fdict[fname] = {"normal": None, "point_m": None, "line_wn_mm": None, "sigma_deg": math.inf, "n": 0,
                                "point_sigma_wn_mm": [math.inf, math.inf]}
                continue
            mvec, sdeg, nk, pm, dm, spm = face_out[fname]
            fdict[fname] = {"normal": mvec[j], "point_m": rj + (pm[j, 0] * w + pm[j, 1] * n) * 1e-3,
                            "line_wn_mm": [float(pm[j, 0]), float(pm[j, 1]), float(dm[j, 0]), float(dm[j, 1])],
                            "sigma_deg": math.hypot(float(sdeg[j]), sig_dir_deg), "n": int(round(float(nk[j]))),
                            "point_sigma_wn_mm": [float(spm[j, 0]), float(spm[j, 1])]}   # the face point about the root (window)
        n_prof = int(round(float(np.sum(pi[idx[j]])))) if idx[j].size else 0
        stations.append(StationEst(s_m=sj, root_m=rj, root_sigma_m=(float(sw[j]), float(sn[j])), u=L.tangent(sj),
                                   n=n, w=w, dims=dims, faces=fdict, n_profiles=n_prof))
    return stations


def _sphere_huber_rows(nrm: np.ndarray, W: np.ndarray, rounds: int = 5) -> tuple:
    """Per row (station) of unit vectors nrm (K,M,3) with weights W (K,M): Huber mean direction on the sphere (robust scale =
    MAD of the angles) → (K,3) unit means, σ_deg of each mean (scatter / √n_eff), Σ weights."""
    sw = W.sum(1)
    z = np.array([0.0, 0.0, 1.0])
    m = _unit(np.einsum("km,kmi->ki", W, nrm) + (sw <= 0)[:, None] * z)
    h = np.ones_like(W)
    for _ in range(rounds):
        ang = np.arccos(np.clip(np.einsum("ki,kmi->km", m, nrm), -1.0, 1.0))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # rows without profiles (σ = ∞ below)
            med = np.nanmedian(np.where(W > 0, ang, np.nan), axis=1)
        scale = np.where(np.isfinite(med) & (med > 0), MAD_SCALE * med, 1e-12)
        h = _huber_w(ang / scale[:, None])
        Wh = W * h
        m = _unit(np.einsum("km,kmi->ki", Wh, nrm) + (Wh.sum(1) <= 0)[:, None] * z)
    ang = np.arccos(np.clip(np.einsum("ki,kmi->km", m, nrm), -1.0, 1.0))
    Wh = W * h
    s1 = Wh.sum(1)
    with np.errstate(all="ignore"):
        neff = s1 ** 2 / np.maximum((Wh ** 2).sum(1), 1e-300)
        sd = np.sqrt((Wh * ang ** 2).sum(1) / np.maximum(s1, 1e-300))
        sdeg = np.degrees(sd / np.sqrt(np.maximum(neff, 1.0)))
    return m, np.where(sw > 0, sdeg, np.inf), sw


# ====================================================================================================================
# §4.5 groove evidence, §4.7 done test, §5.2 plan quantities
# ====================================================================================================================
def groove_evidence(leg: LegObs, line: Optional[dict] = None, prior: Optional[SeamPrior] = None) -> float:
    """max_{a≤b} Σ λ over the leg's profiles sorted by s* (the rough line when no fused line is given)."""
    T = leg.T_base_opt
    if line is not None:
        s = _line_from(line).s_star(T[:, :3, 3], T[:, :3, 1])
    else:
        f0 = prior.frame0()
        s = LineModel(f0, np.zeros((2, 2))).s_star(T[:, :3, 3], T[:, :3, 1])
    lam = _lam_free(leg)                                  # D11 on the seam-free λ
    lam = np.where(np.isfinite(lam), lam, 0.0)[np.argsort(s, kind="stable")]
    return float(max_segment(lam)[0])


def _huber_lnZ1() -> float:
    """ln ∫ exp(−ρ_H(z)) dz (1-D Huber law, HUBER_C) — the normaliser of the law the pooled estimators are robust to."""
    c = HUBER_C
    return math.log(math.sqrt(2.0 * math.pi) * (2.0 * float(norm.cdf(c)) - 1.0) + 2.0 * math.exp(-0.5 * c * c) / c)


def seam_dims(stations: list, sigma_eps: float = float("nan"), extents: Optional[dict] = None, k: Optional[float] = None) -> dict:
    """Seam-level section dims (pooled over the stations): Huber/REML mean of the station values with weights 1/(σ_j² + τ²), τ by
    REML = the along-seam variation of the dim (reported; a taper is specimen deviation, not measurement quality). σ = Laplace of the
    pooled mean; σ_sys = |value|·σ_ε for the lateral dims (NaN when the σ_ε slot is empty → not reported).
    §4.9 (extents = {dim: physical range of the quantity} given): each station window is a segment of the pooled dim — H_fit = the
    pooled model's predictive (the Huber law of this estimator, scale √(σ_j² + τ² + σ_μ²)), H_out = anywhere in the quantity's range;
    the family EM weights the stations by 1 − γ_j (a window that shows another section is not pooled). Report only: per-station dims
    are specimen deviation (user 2026-10-02), the pooled dim never depends on one station."""
    out = {}
    lnZ1 = _huber_lnZ1()
    for nm in DIM_NAMES:
        v = np.array([st.dims[nm]["value"] for st in stations], dtype=float) if stations else np.zeros(0)
        sg = np.array([st.dims[nm]["sigma"] for st in stations], dtype=float) if stations else np.zeros(0)
        ok = np.isfinite(v) & np.isfinite(sg)
        if not ok.any():
            out[nm] = {"value": float("nan"), "sigma": math.inf, "sigma_sys": float("nan"), "along_seam_sd": float("nan"),
                       "n_stations": 0}
            continue
        vv, ss2 = v[ok], np.maximum(sg[ok], 1e-12) ** 2
        wst = np.ones(int(ok.sum()))
        chk = None
        ext = (extents or {}).get(nm)
        for _ in range(REWEIGHT_ROUNDS + 1):
            mu, smu, tau, _ = _reml_rows(vv[None, :], ss2[None, :], wst[None, :])
            if ext is None or not (ext > 0) or vv.size < 2:
                break
            sp = np.sqrt(ss2 + float(tau[0]) ** 2 + float(smu[0]) ** 2)
            zz = (vv - float(mu[0])) / sp
            az = np.abs(zz)
            rho = np.where(az <= HUBER_C, 0.5 * zz * zz, HUBER_C * az - 0.5 * HUBER_C ** 2)
            lo, g, eps = segment_mixture(-rho - np.log(sp) - lnZ1, math.log(float(ext)), None if chk is None else chk["gamma"],
                                         lnf_peak=-np.log(sp) - lnZ1)
            new_w = 1.0 - g
            chk = {"z": zz, "lo": lo, "gamma": g, "eps": eps}
            if float(np.max(np.abs(new_w - wst))) < REWEIGHT_TOL:
                wst = new_w
                break
            wst = new_w
        if chk is not None:                                    # the pooled dim with the final segment weights
            mu, smu, tau, _ = _reml_rows(vv[None, :], ss2[None, :], wst[None, :])
        val = float(mu[0])
        sys_ = (abs(val) * float(sigma_eps)) if nm in ("gap_mm", "top_opening_mm") else 0.0
        out[nm] = {"value": val, "sigma": float(smu[0]), "sigma_sys": sys_ if math.isfinite(sys_) else float("nan"),
                   "along_seam_sd": float(tau[0]), "n_stations": int(ok.sum())}
        if chk is not None:
            s_ok = np.array([st.s_m for st in stations], dtype=float)[ok]
            kk = 3.0 if k is None else float(k)
            verd = [segment_verdict(2.0 * float(x), kk) for x in chk["lo"]]
            out[nm]["station_fit"] = {
                "extent": float(ext), "outlier_eps": float(chk["eps"]), "max_abs_z": float(np.max(np.abs(chk["z"]))),
                "n_fits": verd.count("fits"), "n_ambiguous": verd.count("ambiguous"), "n_misfit": verd.count("misfit"),
                "not_fitting": [{"s_m": float(s_ok[i]), "value": float(vv[i]), "z": float(chk["z"][i]),
                                 "two_ln_odds": 2.0 * float(chk["lo"][i]), "verdict": verd[i]}
                                for i in range(len(verd)) if verd[i] != "fits"]}
    return out


def done_test(est: SeamEstimate, precision, spec=None) -> tuple:
    """D12 (scope per the user's separation "measurement quality = judgement, specimen deviation = report"; user 2026-10-11: the
    requirement is the job's PRECISION targets — 1-σ measurement repeatability — not the design tolerances, which only feed the
    measured-vs-design warnings): done ⇔ max_q σ_q / precision_q ≤ 1 over q = root lateral/normal at every station
    (precision.root_position_mm), both ends (precision.end_position_mm) and the seam-level (pooled) section dims (gap, depth, top
    opening: precision.length_mm; bevel: the angle that length subtends over the bevel face, Precision.for_dim), AND (§4.9,
    est.plate['model_checks']) no q rests on a segment that does not fit the fused model or on an unresolved hypothesis. precision None
    (no targets) → nothing is done. Per-station dims are reported with σ but do not gate. Returns (done, ratios {q: max σ/precision},
    unresolved [q with ratio > 1, then q blocked by §4.9]); binding = argmax q (SeamEstimate.binding)."""
    ratios = {}
    st = est.stations or []

    def worst(vals):
        return max(vals) if vals else math.inf

    if precision is None:
        return False, {q: math.inf for q in Q_NAMES}, list(Q_NAMES)
    t = float(precision.root_position_mm) * 1e-3
    # the root's σ is established only with enough legs to know their offsets' variance (ν > 2 per component, as the ends' τ below)
    r_ok = list((est.plate or {}).get("root_established") or (True, True))
    ratios["root_lateral"] = worst([s.root_sigma_m[0] / t for s in st]) if r_ok[0] else math.inf
    ratios["root_normal"] = worst([s.root_sigma_m[1] / t for s in st]) if r_ok[1] else math.inf
    t = float(precision.end_position_mm) * 1e-3
    tau_ok = math.isfinite(est.tau_end_m)                # per-leg end scatter unidentified → the end σ is not established
    for X in ("S", "E"):
        e = est.ends.get(X)
        ratios["end_" + X] = (e.sigma_m / t) if (e is not None and tau_ok) else math.inf
    sd = (est.plate or {}).get("seam_dims") or seam_dims(st)
    for nm in ("gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm"):
        t = precision.for_dim(nm, spec)
        if not t:
            continue
        d = sd.get(nm, {})
        ss_ = d.get("sigma_sys", 0.0)
        ss_ = ss_ if (ss_ is not None and math.isfinite(ss_)) else 0.0     # σ_sys not reported (σ_ε slot empty) → σ_rand only
        sg = d.get("sigma", math.inf)
        ratios[nm] = math.hypot(sg, ss_) / t if math.isfinite(sg) else math.inf
    unresolved = [q for q, r in ratios.items() if not (r <= 1.0)]
    for b in ((est.plate or {}).get("model_checks") or {}).get("blocking") or []:
        for q in b.get("quantities") or []:
            if q in ratios and q not in unresolved:
                unresolved.append(q)
    done = bool(ratios) and not unresolved
    return done, ratios, unresolved


def plan_ends(est) -> dict:
    """The ends to PLAN the next action with: est.plate['ends_plan'] (measurements + the design length) when the estimate has it, else
    the delivered ends. {X: {'s_m', 'sigma_m', 'point_m'}}."""
    pe = (getattr(est, "plate", None) or {}).get("ends_plan") if not isinstance(est, dict) else (est.get("plate") or {}).get("ends_plan")
    out = {}
    for X in ("S", "E"):
        if pe and X in pe:
            out[X] = {"s_m": float(pe[X]["s_m"]), "sigma_m": float(pe[X]["sigma_m"]), "point_m": np.asarray(pe[X]["point_m"], float)}
        else:
            e = est.ends[X]
            out[X] = {"s_m": float(e.s_m), "sigma_m": float(e.sigma_m), "point_m": np.asarray(e.point_m, float)}
    return out


def plan_quantities(est: SeamEstimate, prior: SeamPrior) -> dict:
    """§5.2 inputs for the next pass (keys per INTERFACE_NOTES WP3): e_S_m / e_E_m (taught s), point_X_m, sigma_e_X_m,
    sigma_plan_X_m (= σ_e: the non-pass class priors already carry σ_he, spec r2 §4.2), psi_res_rad
    (k·σ of the line yaw), pitch_res_rad, depth_m / D_hat_mm (D̂), u, n, pass_class_measured. σ_e is always finite (σ given
    τ_e = 0 when τ_e is unidentified; the done test still reports such ends)."""
    k = float(prior.k)
    measured = any(_canon_cls(r.get("cls", "")) == GAUGE_CLASS for X in ("S", "E") for r in (est.ends[X].readings or []))
    L = LineModel.from_estimate(est)
    out = {"u": est.u, "n": est.n_plate, "pass_class_measured": measured, "fine_class_measured": measured}
    pe = plan_ends(est)                                    # the next action is planned with the design length (user rule 2026-10-11)
    for X in ("S", "E"):                                   # σ_he is inside σ_e already (class priors, r2 §4.2)
        e = pe[X]
        out[f"e_{X}_m"] = e["s_m"]
        out[f"point_{X}_m"] = e["point_m"]
        out[f"sigma_e_{X}_m"] = e["sigma_m"]
        out[f"sigma_plan_{X}_m"] = e["sigma_m"]
    s_mid = 0.5 * (pe["S"]["s_m"] + pe["E"]["s_m"])
    sa, sb = L.sigma_dir(np.array([pe["S"]["s_m"], s_mid, pe["E"]["s_m"]]))
    out["psi_res_rad"] = k * float(np.max(sa))
    out["pitch_res_rad"] = k * float(np.max(sb))
    dep = [s.dims["depth_mm"]["value"] for s in est.stations if math.isfinite(s.dims["depth_mm"]["value"])]
    dsig = [s.dims["depth_mm"]["sigma"] for s in est.stations if math.isfinite(s.dims["depth_mm"]["value"])]
    if dep:
        wgt = 1.0 / np.maximum(np.square(dsig), 1e-12)
        out["D_hat_mm"] = float(np.sum(wgt * np.asarray(dep)) / np.sum(wgt))
    else:
        out["D_hat_mm"] = float(prior.spec.thickness_mm)
    out["depth_m"] = out["D_hat_mm"] * 1e-3
    return out


def _top_point(stations: list, n_hat: np.ndarray) -> dict:
    """S1 reference (§5.3, WP3): highest fused plate-top point along n̂ over the stations that MEASURED it, and its σ along n̂: the
    root's normal σ (line, offsets) ⊕ the face point's own σ about the root (its window mean — the plate top is a localisation
    quantity, read by every pose class; it no longer borrows the depth / hi-lo σ, which are the passes' dims only). A station's
    plate-top point counts only where profiles carry that face (its count n > 0) and its σ is finite: a window whose profiles all
    have weight ~0 still returns a template point with σ ~1e15 mm (test_sim_teach1g m2 after the w_x calibration: that point was
    'the highest', the world's plate top went to 7.8e12 m and the run was refused). None measured → (None, ∞)."""
    best, bs = None, None
    nh = _unit(n_hat)
    for st in stations:
        for f in ("plate_top_left", "plate_top_right"):
            fc = st.faces.get(f, {})
            P = fc.get("point_m")
            if P is None or not int(fc.get("n", 0) or 0) > 0:
                continue
            sp = fc.get("point_sigma_wn_mm") or [math.inf, math.inf]
            v = st.root_sigma_m[1] ** 2 + (float(sp[1]) * 1e-3) ** 2
            if not math.isfinite(v):
                continue
            h = float(np.asarray(P) @ nh)
            if best is None or h > best[0]:
                best, bs = (h, np.asarray(P, float)), math.sqrt(v)
    if best is None:
        return {"top_point_m": None, "sigma_top_m": math.inf}
    return {"top_point_m": best[1], "sigma_top_m": bs}


def _as_estimate(est) -> SeamEstimate:
    """SeamEstimate or its to_dict() form (worker payloads) → SeamEstimate."""
    if isinstance(est, SeamEstimate):
        return est
    from dataclasses import fields as _fields
    d = dict(est)

    def num(v):
        return float("nan") if v is None else v

    ends = {}
    for X, e in (d.get("ends") or {}).items():
        ends[X] = EndEst(s_m=num(e["s_m"]), sigma_m=num(e["sigma_m"]), point_m=np.asarray(e["point_m"], float),
                         two_sided=bool(e["two_sided"]), n_legs=int(e["n_legs"]), readings=list(e.get("readings") or []))
    sts = []
    for st in d.get("stations") or []:
        sts.append(StationEst(s_m=num(st["s_m"]), root_m=np.asarray(st["root_m"], float),
                              root_sigma_m=tuple(num(x) for x in st["root_sigma_m"]), u=np.asarray(st["u"], float),
                              n=np.asarray(st["n"], float), w=np.asarray(st["w"], float), dims=st["dims"], faces=st["faces"],
                              n_profiles=int(st["n_profiles"])))
    kw = {f.name: d.get(f.name) for f in _fields(SeamEstimate)}
    kw.update(ends=ends, stations=sts)
    for k in ("log_bf_run", "dt_s", "dt_sigma_s", "tau_end_m"):
        kw[k] = num(kw[k])
    return SeamEstimate(**kw)


def _rotation(axis, ang: float) -> np.ndarray:
    a = _unit(np.asarray(axis, dtype=float))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _view_plate_m(view, depth_m: float) -> float:
    """Target plate distance `coarse_plate_m` from the view (SensorView.coarse_plate_m(depth), a dict or a number)."""
    f = getattr(view, "coarse_plate_m", None)
    if callable(f):
        return float(f(depth_m))
    if isinstance(view, dict):
        return float(view["coarse_plate_m"])
    return float(view)


def site_state(grabs: list, est, prior: SeamPrior, site: str, *, T_now: np.ndarray, calib, view, goal_tol_m: float) -> dict:
    """§4.8 (r2, D15): state of a site from its grabs (dwell LegObs, N = 1; the latest one is read) and the current estimate.

    root_m (base) with the pose-dependent σ in the laser plane: σ_lat² = σ_fit,x² + (σ_ε·|x_R|)², σ_nor = σ_fit,z (σ_fit from the
    grab's Laplace covariance; σ_ε NaN → the x-scale term is unknown (∞) for |x_R| > e_goal, 0 within). The class offset σ_he is
    pose-independent and excluded. roll_rad (about û: optical axis → −n̂ of the plate), standoff_err_m (plate top vs the view's
    coarse plate distance along n̂), yaw_rad (laser-plane normal → û about n̂; only with an estimate, i.e. at the second site), each
    with σ. T_target = flange pose with the root at sensor x = 0, plate top at `coarse_plate_m`, optical axis ⟂ plate in the laser
    plane (and the laser plane ⟂ û at the second site): rotation about the root (roll about û, then yaw about n̂) applied to T_now —
    the root stays in the rotated laser plane — then the optical origin moves along the rotated optical axis through the root to
    the plate distance (normal distance of the origin above the plate top = coarse_plate_m). Neither rotation removes a pitch of
    the optical axis along û; moving along n̂ instead would shift the laser plane off the root by (D + r_t)·tan(pitch) along the
    seam (≈ 10 mm at 1.3°, past an open plate end next to the site). needs_correction (D15) = k·max(σ_lat, σ_nor)/tol_root > 1 and
    the predicted ratio at T* (|x_R| = e_goal) is smaller."""
    g = grabs[-1]
    v = _LegView(g)
    f0 = prior.frame0()
    th, cv = np.asarray(v.theta[0], float), np.asarray(v.cov[0], float)
    T_bo = np.asarray(v.T_base_opt[0], float)
    k = float(prior.k)
    if not (np.all(np.isfinite(th)) and np.all(np.isfinite(cv))):
        return {"ok": False, "needs_correction": False, "why": "grab has no converged fit"}
    X, Y, Z, o = T_bo[:3, 0], T_bo[:3, 1], T_bo[:3, 2], T_bo[:3, 3]
    if est is not None:
        E = _as_estimate(est)
        Lm = LineModel.from_estimate(E)
        s_d = float(Lm.s_star(o, Y))
        uhat = Lm.tangent(s_d)
        sig_yaw = float(Lm.sigma_dir(s_d)[0])
    else:
        uhat, sig_yaw, s_d = f0["u0"], None, float((o - f0["S0"]) @ f0["u0"])
    geo = _leg_geometry(v, prior.spec)
    R = o + 1e-3 * (th[0] * X + th[1] * Z)
    # plate normal from both plate-top lines (⟂ û), oriented toward the sensor; its angle about û and σ
    _, n_ref, w_ref = _section_basis(uhat, f0["n0"])
    phis, vphis = [], []
    for face in ("top_L", "top_R"):
        ph, vp = _plate_angles(geo, np.array([True]), cv[None], face, uhat, n_ref, w_ref)
        phis.append(float(ph[0]))
        vphis.append(float(vp[0]))
    phi = 0.5 * (phis[0] + phis[1])
    sig_phi = 0.5 * math.sqrt(vphis[0] + vphis[1])
    n_d = _unit(math.cos(phi) * n_ref + math.sin(phi) * w_ref)
    # roll about û: current viewing axis (⟂ û) → −n̂
    a = _unit(Z - float(Z @ uhat) * uhat)
    roll = math.atan2(float(np.cross(a, -n_d) @ uhat), float(a @ -n_d))
    # plate top above the root along n̂ (mean of both top corners) and the stand-off error
    iL, iR = geo["names_v"].index("top_corner_L"), geo["names_v"].index("top_corner_R")
    hts = [float((geo["V"][0, i] - R) @ n_d) for i in (iL, iR)]
    D = 0.5 * (hts[0] + hts[1])
    r_t = _view_plate_m(view, D)
    standoff_now = float((o - (R + n_d * D)) @ n_d)
    # σ of the stand-off: standoff = (o − ½(C_L + C_R))·n̂ (o fixed) → θ-Jacobian −½ n̂·(dC_L + dC_R)
    jS = -0.5 * (n_d @ geo["dV"][0, iL] + n_d @ geo["dV"][0, iR])
    sig_standoff = math.sqrt(max(float(jS @ cv @ jS), 0.0))
    # yaw about n̂: laser-plane normal → ±û (second site, once the transit is fused)
    yaw = None
    if est is not None:
        y = _unit(Y - float(Y @ n_d) * n_d)
        tgt = uhat if float(y @ uhat) >= 0 else -uhat
        tgt = _unit(tgt - float(tgt @ n_d) * n_d)
        yaw = math.atan2(float(np.cross(y, tgt) @ n_d), float(y @ tgt))
    # target pose (flange): rotate about the root (the root stays in the laser plane), then move the optical origin along the
    # rotated optical axis through the root until its normal distance above the plate top is r_t
    Rc = _rotation(uhat, roll)
    if yaw is not None:
        Rc = _rotation(n_d, yaw) @ Rc
    T_now = np.asarray(T_now, dtype=float)
    T_fo = np.asarray(calib.T_flange_opt, dtype=float) if hasattr(calib, "T_flange_opt") else np.asarray(calib, float)
    T_rot = np.eye(4)
    T_rot[:3, :3] = Rc
    T_rot[:3, 3] = R - Rc @ R
    T1 = T_rot @ T_now
    T1o = T1 @ T_fo
    o1, Z1 = T1o[:3, 3], T1o[:3, 2]
    o_star = R - Z1 * ((D + r_t) / float(-Z1 @ n_d))
    T_target = T1.copy()
    T_target[:3, 3] += o_star - o1
    # D15
    sfx, sfz = 1e-3 * math.sqrt(max(cv[0, 0], 0.0)), 1e-3 * math.sqrt(max(cv[1, 1], 0.0))
    xR = 1e-3 * float(th[0])
    eps = float(prior.sigma_eps) if prior.sigma_eps is not None else float("nan")
    e_goal = float(goal_tol_m)

    def s_lat(x):
        if math.isfinite(eps):
            return math.hypot(sfx, eps * abs(x))
        return sfx if abs(x) <= e_goal else math.inf

    tol = float(prior.tol.root_position_mm) * 1e-3
    ratio = k * max(s_lat(xR), sfz) / tol
    ratio_T = k * max(s_lat(e_goal), sfz) / tol
    needs = bool(ratio > 1.0 and ratio_T < ratio)
    why = (f"k·max(σ_lat, σ_nor)/tol_root = {ratio:.2f} at |x_R| = {abs(xR) * 1e3:.2f} mm; at T* (|x_R| = e_goal "
           f"{e_goal * 1e3:.2f} mm) {ratio_T:.2f}" + ("" if math.isfinite(eps) else "; σ_ε not published → x-scale term ∞ off e_goal"))
    return {"ok": True, "site": site, "grab": g.name, "s_m": s_d, "root_m": R, "x_R_m": xR, "sigma_fit_x_m": sfx,
            "sigma_lat_m": s_lat(xR), "sigma_nor_m": sfz, "roll_rad": roll, "sigma_roll_rad": sig_phi,
            "standoff_m": standoff_now, "standoff_target_m": r_t, "standoff_err_m": standoff_now - r_t,
            "sigma_standoff_m": sig_standoff, "yaw_rad": yaw, "sigma_yaw_rad": sig_yaw, "n": n_d, "u": uhat, "depth_m": D,
            "T_target": T_target, "needs_correction": needs, "ratio": ratio, "ratio_at_target": ratio_T, "why": why,
            "log_bf": float(v.log_bf[0])}


def profile_predictor(est, T_base_opt: np.ndarray, bevel_x_sign: Optional[int] = None, spec=None, cls: str = "pass"):
    """Start (2) of §3.4 and the seam-position prior for WP1's fit_profile: callable(i) → (θ0 (10,), Σ_R (2,2) mm²) for profile i of a
    leg whose poses are T_base_opt (N,4,4) — also beyond the ends (the extrapolated line), where the prior rejects groove-like features
    off the seam. `est` may be a SeamEstimate or its to_dict(); bevel_x_sign None → derived from the fused bevel side and the leg's
    laser-line axis; cls = the new leg's class.

    Σ_R = covariance of where this profile's measured root will be, in the sensor plane (x_R, z_R): the fused line at s (its
    polynomial covariance — grows when extrapolated; contains σ_he before any pass), the line misfit τ_root (waviness), the class's
    profile jitter σ_ζ, the leg offset σ_η of a new pass, and σ_he when the leg's class is not in the estimate yet; mapped into the laser
    plane by the exact oblique projection along the seam. Gaussian at 1σ — WP1 should read it with heavy tails (INTERFACE_NOTES).

    Exact geometry: the nearest station's section (root r(s), ŵ, n̂, measured G, D, β, rf, H, plate normals) is turned into edge lines
    parallel to û, intersected with the laser plane, and read in the sensor (x, z) frame (apparent values carry ψ and pitch)."""
    est = _as_estimate(est)
    T = np.asarray(T_base_opt, dtype=float)
    L = LineModel.from_estimate(est)
    st = est.stations
    if bevel_x_sign is None:                              # the leg's sensor-x side of the bevel plate from the fused side
        _, _, w_ = _section_basis(est.u, est.n_plate)
        bevel_x_sign = 1 if float(np.median(T[:, :3, 0], axis=0) @ w_) * est.bevel_sign_w >= 0 else -1
    if not st:
        return lambda i: None
    s_st = np.array([x.s_m for x in st])
    o, ex, m, ez = T[:, :3, 3], T[:, :3, 0], T[:, :3, 1], T[:, :3, 2]
    s_i = L.s_star(o, m)
    b = int(bevel_x_sign)
    bw = int(est.bevel_sign_w)
    nl, nr = est.plate.get("n_left"), est.plate.get("n_right")
    eS, eE = est.ends["S"].s_m, est.ends["E"].s_m

    def dim(x, k, default):
        v = x.dims.get(k, {}).get("value", float("nan"))
        return float(v) if math.isfinite(v) else default

    f0 = est.frame0
    ccls = _canon_cls(cls)
    jkey = "dwell" if ccls.startswith("dwell") else ccls
    sz = float((est.sigma_zeta_m or {}).get(jkey, 0.0) or 0.0)
    swn_ = (est.plate or {}).get("leg_offset_sigma_wn_m")
    se_ = float((est.plate or {}).get("leg_offset_sigma_m", 0.0) or 0.0) if ccls == GAUGE_CLASS else 0.0
    se_w, se_n = ((float(swn_[0]), float(swn_[1])) if (swn_ is not None and ccls == GAUGE_CLASS) else (se_, se_))
    # the new leg's motion direction (its poses' travel along u0) × the direction offset b (w, n) where the estimate identifies it
    do_ = (est.plate or {}).get("direction_offset") or {}
    d_new = int(np.sign(float((o[-1] - o[0]) @ f0["u0"]))) if len(o) > 1 else 0
    b_wn = np.array([float(do_["b_m"][c]) if (do_.get("identified") or [False] * 3)[c] else 0.0 for c in (0, 1)]) \
        if do_ else np.zeros(2)
    vb_wn = np.array([float(do_["sigma_m"][c]) ** 2 if (do_.get("identified") or [False] * 3)[c] else 0.0 for c in (0, 1)]) \
        if do_ else np.zeros(2)
    he = float((est.plate or {}).get("he_sigma_m", float("nan")))
    sb = 0.0 if (ccls in (est.class_offset_m or {}) or not math.isfinite(he)) else he
    st_rs = np.array([x.root_sigma_m for x in st], dtype=float)
    lw_, ln_ = L.sigma_wn(s_st)
    tau_w = float(np.sqrt(np.median(np.maximum(st_rs[:, 0] ** 2 - lw_ ** 2, 0.0))))
    tau_n = float(np.sqrt(np.median(np.maximum(st_rs[:, 1] ** 2 - ln_ ** 2, 0.0))))
    beta_c = np.asarray((est.class_offset_m or {}).get(ccls, np.zeros(3)), dtype=float)

    def cov_root(i, si):
        """Σ_R (x_R, z_R) in mm²."""
        sw_, sn_ = L.sigma_wn(si)
        cw = float(sw_) ** 2 + tau_w ** 2 + sz ** 2 + se_w ** 2 + sb ** 2 + d_new * d_new * vb_wn[0]
        cn = float(sn_) ** 2 + tau_n ** 2 + sz ** 2 + se_n ** 2 + sb ** 2 + d_new * d_new * vb_wn[1]
        t = L.t(si)
        mi = m[i]
        Q = np.eye(3) - np.outer(t, mi) / float(mi @ t)
        E2 = np.stack([ex[i], ez[i]]) @ Q                  # sensor (x, z) ← base displacement of the root line
        Jw, Jn = E2 @ f0["w0"], E2 @ f0["n0"]
        return 1e6 * (cw * np.outer(Jw, Jw) + cn * np.outer(Jn, Jn))

    def predict(i):
        si = float(s_i[i])
        x = st[int(np.argmin(np.abs(s_st - si)))]
        u, n, w = _section_basis(L.tangent(si), x.n)
        r = L.r(si)
        G = dim(x, "gap_mm", spec.root_gap_mm if spec else 5.0) * 1e-3
        D = dim(x, "depth_mm", spec.thickness_mm if spec else 10.0) * 1e-3
        be = math.radians(dim(x, "bevel_angle_deg", spec.bevel_angle_deg if spec else 30.0))
        rf = dim(x, "root_face_mm", 0.0) * 1e-3
        H = dim(x, "hi_lo_mm", 0.0) * 1e-3
        P = {"R": r, "B0": r + bw * 0.5 * G * w, "B1": r + bw * 0.5 * G * w + rf * n,
             "Cb": r + bw * (0.5 * G + (D - rf) * math.tan(be)) * w + D * n, "Q0": r - bw * 0.5 * G * w,
             "Cs": r - bw * 0.5 * G * w + (D + H) * n}
        mi, oi, exi, ezi = m[i], o[i], ex[i], ez[i]
        q = {}
        for k_, X0 in P.items():
            sg = float(mi @ (oi - X0)) / float(mi @ u)
            Pk = X0 + sg * u - oi
            q[k_] = np.array([Pk @ exi, Pk @ ezi]) * 1e3
        e1 = b * (q["B0"] - q["Q0"])
        nrm = float(np.linalg.norm(e1))
        if not nrm > 0:
            return None
        e1 /= nrm
        al = math.atan2(e1[1], e1[0])
        e2 = np.array([math.sin(al), -math.cos(al)])
        Gs = nrm
        rfs = float((q["B1"] - q["R"]) @ e2)
        hL = float((q["Cb"] - q["R"]) @ e2)
        xi = float((q["Cb"] - q["R"]) @ e1)
        bet = math.atan2(b * xi - 0.5 * Gs, hL - rfs) if spec is None or spec.kind != "square" else 0.0
        hR = float((q["Cs"] - q["R"]) @ e2)

        def gam(np_):
            if np_ is None:
                return 0.0
            d3 = np.cross(np.asarray(np_, float), mi)
            d2 = np.array([d3 @ exi, d3 @ ezi])
            xi_c, eta = -b * float(e1 @ d2), float(e2 @ d2)
            g = math.atan2(eta, xi_c)
            return math.atan2(-eta, -xi_c) if math.cos(g) < 0 else g

        b_left = bw > 0
        gL = gam(nl if b_left else nr)
        gR = gam(nr if b_left else nl)
        th0 = np.array([q["R"][0], q["R"][1], al, Gs, bet, rfs, hL, hR, gL, gR])
        # where THIS leg will see the root: measured + β_c + d·b = true (class offset of a known class, its motion direction)
        dR = -beta_c - d_new * (b_wn[0] * f0["w0"] + b_wn[1] * f0["n0"])
        th0[0] += 1e3 * float(dR @ ex[i])
        th0[1] += 1e3 * float(dR @ ez[i])
        return th0, cov_root(i, si)

    return predict


def _end_faces(legs: list, prof: "_Profiles", line: dict, prior: SeamPrior, pi_sec: np.ndarray, per_leg: dict, fz: dict,
               fz_plan: Optional[dict] = None, *, map_fn=None, n_children: int = 1, edge_groups_out: Optional[list] = None,
               vc_out: Optional[dict] = None) -> tuple:
    """Seam ends from the measured points at the plate ends (end_faces): fused section near the ends, per-leg top-edge data (each
    leg's halves split at its groove segment's middle; the region = the λ end of the measurements only — fz, without the design
    length; fz_plan only where fz has no two-sided reading), readings through the sensor's edge response (calibrated; independent per
    reading → map_fn), fusion. Legs without measured profiles give no reading."""
    from . import end_faces as ef
    sec = ef.section_model(prof.geometry(prior.spec), line["s_star"], line, pi_sec)
    measured = []
    for lg in legs:
        cp = per_leg.get(lg.name) or {}
        if sec is None or cp.get("segment_m") is None or not any((cp.get(X) or {}).get("kind") == "two_sided" for X in ("S", "E")):
            continue
        meas = ef.load_measured(lg)
        if meas is not None:
            measured.append((lg, cp, meas))
    coarse = {}
    for X in ("S", "E"):
        ok_ = math.isfinite(float(fz["ends"][X]["sigma_m"])) and fz["ends"][X].get("two_sided")
        src = fz if (ok_ or fz_plan is None) else fz_plan
        coarse[X] = {"s_m": float(src["ends"][X]["s_m"]), "sigma_m": float(src["ends"][X]["sigma_m"])}

    def one_leg(item):
        lg, cp, meas = item
        cls = _canon_cls(lg.cls, lg.meta)
        jit = float((line.get("sigma_zeta") or {}).get("dwell" if cls.startswith("dwell") else cls, 0.0) or 0.0)
        sr = np.asarray(lg.sigma_r_mm, dtype=float)
        sr_m = float(np.nanmedian(sr)) * 1e-3 if np.isfinite(sr).any() else 0.0
        sigma_d = math.hypot(sr_m, jit)
        zw = float(np.diff((lg.meta or {}).get("z_window_mm", Z_VALID_MM_DEFAULT))[0]) * 1e-3
        crosses = {X: bool((cp.get(X) or {}).get("kind") == "two_sided") for X in ("S", "E")}
        return ef.leg_edge_data(lg, meas, line, sec, coarse, prior, 0.5 * float(sum(cp["segment_m"])), sigma_d, jit, zw, crosses)

    todo = measured                                      # legs that cross an end, with measured points
    if map_fn is not None and n_children > 1 and len(todo) > 1:
        per = map_fn(lambda ch: [g for it in ch for g in one_leg(it)], [[it] for it in todo], min(int(n_children), len(todo)))
        groups = [g for part in per for g in part]
    else:
        groups = [g for it in todo for g in one_leg(it)]
    if edge_groups_out is not None:
        edge_groups_out.extend(groups)
    rd, sensor = ef.edge_readings(groups, getattr(prior, "edge_response", None), map_fn=map_fn, n_children=n_children)
    # the echo branch's δ: the flat lag times the groove fusion's likelihood of δ (the run's data — dwells, transit, passes at their
    # speeds — identify the lag; the plate-end readings of one pass speed alone cannot, and a flat lag there has a heavy tail through
    # the crossing scatter σ_ζ) — a likelihood, not that fusion's posterior: the branch keeps τ_echo ≥ 0 (js_source module doc)
    rb = fz.get("delta_ref_likelihood")
    dp = line.get("delta_prior")
    line_fe = line if (rb is None or dp is None) else dict(line, delta_prior=js_source.DeltaPrior(
        dp.sigma_q, dp.p_ref, ref_lik_mean=float(rb["mean_s"]), ref_lik_var=float(rb["var_s"])))
    fe = ef.fuse_end_faces(rd, prior, line_fe, line.get("sigma_zeta") or {}, sensor, vc_out=vc_out)
    fe["sensor"] = sensor
    fe["has_points"] = any(getattr(lg, "_measured_cache", None) is not None or getattr(lg, "recon_path", None) for lg in legs)
    return fe, rd


def _direction_report(line: dict, o_src: dict) -> dict:
    """est.plate['direction_offset']: the robot's motion-direction offset b = (w, n, u) [m] with σ, per component whether the data
    identify it (rank), and the legs' motion signs. Unidentified components are NaN / σ ∞ (no value is claimed)."""
    off = line.get("offsets") or {}
    ok = tuple(off.get("direction_identified") or (False, False))
    b = np.asarray(off.get("direction", np.zeros(2)), dtype=float)
    cb = np.asarray((line.get("offsets_cov") or {}).get("direction", np.diag([math.inf, math.inf])), dtype=float)
    ok_u = bool(o_src.get("b_u_identified", False))
    val = [float(b[0]) if ok[0] else float("nan"), float(b[1]) if ok[1] else float("nan"),
           float(o_src.get("b_u", 0.0)) if ok_u else float("nan")]
    sig = [math.sqrt(max(float(cb[0, 0]), 0.0)) if ok[0] else math.inf, math.sqrt(max(float(cb[1, 1]), 0.0)) if ok[1] else math.inf,
           float(o_src.get("b_u_sigma", math.inf)) if ok_u else math.inf]
    mov = bool(off.get("moving_legs", False))
    mov_u = bool(o_src.get("moving_readings", mov))
    stat = [bool(ok[0]) or not mov, bool(ok[1]) or not mov, ok_u or not mov_u]
    return {"b_m": val, "sigma_m": sig, "identified": [bool(ok[0]), bool(ok[1]), ok_u],
            "static_frame_identified": stat,
            "leg_direction": dict(line.get("leg_direction") or {}),
            "frame": ("static: the midpoint of the two motion directions (a leg moving in direction d is offset by d*b)" if all(stat)
                      else "the components with static_frame_identified false are in the frame of the one motion direction measured "
                           "(b not identified: the static frame is not known there)"),
            "note": ("robot motion-direction offset (joint friction x compliance); a fixed effect where the rank of the design "
                     "identifies it (legs of both directions), otherwise not in the model (NaN)")}


TRACKING_MONITOR_ECHO_WARNING = (
    "WARNING: /joint_states is the robot controller's reference echo, not the measured joint angles: it repeats our command, so a "
    "blocked or lagging arm does not show in it; the physical arm trails the reported pose by an unknown servo lag (fused as the "
    "time offset delta)")


def joint_state_report(timing: dict, dpri, k: float) -> dict:
    """est.plate['time_offset']['joint_state_source']: the hypothesis posterior (js_source, prior ½ / ½), the evidence per feature
    (ln BF measured : reference, summed over the legs, and per leg), the data it rests on, the δ prior used and — when the evidence
    resolves the reference echo (2·ln odds > k², the estimator's one decision form) — the tracking-monitor warning."""
    js = (timing or {}).get("joint_state")
    dp = dpri.as_dict() if dpri is not None else None
    if not js:
        return {"hypothesis": js_source.H_MEAS, "p_reference_echo": 0.0, "evidence": None, "delta_prior": dp, "warning": None,
                "note": "no joint-state samples in the legs (poses given per profile): no echo to consider"}
    v = js_source.verdict(float(js["ln_odds_ref"]), float(k))
    return {"hypothesis": v, "p_reference_echo": float(js["p_ref"]), "ln_odds_reference": float(js["ln_odds_ref"]),
            "prior_p_reference": float(js.get("prior_p_ref", 0.5)),
            "evidence": {"ln_bf_measured_vs_reference": dict(js["ln_bf_meas_vs_ref"]), "per_leg": list(js.get("per_leg") or [])},
            "data": {nm: js.get(nm) for nm in ("legs", "samples", "lattice_values", "rest_pairs", "rest_changes", "representation")},
            "delta_prior": dp, "warning": TRACKING_MONITOR_ECHO_WARNING if v == js_source.H_REF else None,
            "note": ("two-hypothesis comparison from the recorded /joint_states and our commanded motions only (no robot-controller "
                     "access): encoder lattice of the values and hold-vs-dither at rest (js_source); the time-offset prior is the "
                     "hypotheses' cut mixture")}


def _time_report(line: dict, o_src: dict, prior: SeamPrior, source: str) -> dict:
    """est.plate['time_offset']: the residual time offset δ after the calibrated dt, its prior (the run's timing quantisation and
    the joint-state source, js_source.DeltaPrior), the data's share of its posterior precision, whether this run separates it from
    the along-track direction offset b_u, and the joint-state source record (joint_state_report)."""
    tp = line.get("timing") or {}
    dpri = line.get("delta_prior")
    s_pr = float(line.get("dt_sigma_s", float("nan")))
    s_po = float(o_src.get("delta_sigma_s", math.inf))
    share = o_src.get("delta_data_share")
    if share is None:
        prec = dpri.prior_precision() if dpri is not None else (1.0 / s_pr ** 2 if s_pr > 0 else 0.0)
        share = max(0.0, 1.0 - s_po * s_po * prec) if math.isfinite(s_po) else 0.0
    out = {"calibrated_dt_s": float(prior.dt_s), "delta_s": float(o_src.get("delta_s", 0.0)), "sigma_s": s_po,
           "prior_sigma_s": s_pr, "frame_period_s": tp.get("frame_period_s"), "pose_period_s": tp.get("pose_period_s"),
           "data_share": float(share), "separable_from_b_u": bool(float(share) > 0.5), "source": source,
           "joint_state_source": joint_state_report(tp, dpri, float(prior.k)),
           "note": ("prior (1 - p_ref) N(0, sigma) + p_ref [reference echo: delta = tau_echo >= 0 flat + N(0, sigma)], sigma^2 = "
                    "(frame period^2 + pose sample period^2)/12, p_ref = P(reference echo | joint states); data_share = the data's "
                    "part of the posterior precision of delta (> 0.5: the run separates the latency from the direction offset b_u)")}
    atd = o_src.get("along_track_direction")
    if atd:
        out["along_track_direction"] = atd
    return out


def _end_faces_summary(fe: dict, rd: list, prior: SeamPrior) -> dict:
    from . import end_faces as ef
    return ef.summarize(fe, rd, prior)


def end_at(est, X: str, h_m: float) -> tuple:
    """Seam end X ('S' | 'E') for a weld pass at height h_m above the root: (s_m, sigma_m) in the taught frame. The plate end faces
    are square (declared joint), so it is the inner plate end at every height."""
    ef_ = (est.plate or {}).get("end_faces") or {}
    r = ((ef_.get("ends") or {}).get(X) or {}).get("root_combined")
    if not r or not math.isfinite(float(r.get("s_m", float("nan")))):
        return float(est.ends[X].s_m), float(est.ends[X].sigma_m)
    return float(r["s_m"]), float(r["sigma_m"])


def _wn(v, f0) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([float(v @ f0["w0"]), float(v @ f0["n0"])])


# ====================================================================================================================
# estimate()
# ====================================================================================================================
def _leg_side_w(lg, f0) -> int:
    """The w-frame side (+1 = bevel plate on +w) that the leg's template mirror (bevel_x_sign) stands for."""
    ex_med = _unit(np.median(np.asarray(lg.T_base_opt)[:, :3, 0], axis=0))
    return 1 if float(ex_med @ f0["w0"]) * int(lg.bevel_x_sign) >= 0 else -1


def _bevel_sign_w(legs: list, prior: SeamPrior, f0: dict) -> tuple:
    """D10 bevel side in the base frame, re-decided at every estimate from ALL the evidence: the pooled log Bayes factor
    Σ_legs ln BF_w over every leg fitted with both mirrored templates (meta['bevel_side_ln_bf'], sensor-x sign mapped through the leg's
    laser-line axis onto w0) — a ghost first grab cannot outvote the later grabs and the transit. The job's side overrides (prior).
    No leg carries a BF → the side the legs were fitted with (the first leg). Returns (sign, pooled ln BF_w or None, per-leg dict)."""
    per = {}
    for lg in legs:
        bf = (lg.meta or {}).get("bevel_side_ln_bf")
        if lg.n and bf is not None and math.isfinite(float(bf)):
            ex_med = _unit(np.median(np.asarray(lg.T_base_opt)[:, :3, 0], axis=0))
            per[lg.name] = float(bf) * (1.0 if float(ex_med @ f0["w0"]) >= 0 else -1.0)
    pooled = float(sum(per.values())) if per else None
    if prior.bevel_sign_w is not None:
        return int(prior.bevel_sign_w), pooled, per
    if pooled is not None and pooled != 0.0:
        return (1 if pooled > 0 else -1), pooled, per
    cand = [lg for lg in legs if lg.n]
    return (_leg_side_w(cand[0], f0) if cand else 1), pooled, per


HUBER_ROUNDS = 8               # [R] IRLS rounds of a segment's Huber mean (§4.9)


def _wmedian(x, w) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(int(np.searchsorted(cw, 0.5 * cw[-1])), x.size - 1)])


def _leg_start(B: np.ndarray, r: np.ndarray, w: np.ndarray) -> np.ndarray:
    """High-breakdown start of a leg's mean offset (w0, n0): each profile's own offset B_i⁻¹ r_i, component-wise weighted median."""
    with np.errstate(all="ignore"):
        xi = np.linalg.solve(B, r[..., None])[..., 0]
    ok = np.all(np.isfinite(xi), axis=1) & (w > 0)
    if not np.any(ok):
        return np.zeros(2)
    return np.array([_wmedian(xi[ok, c], w[ok]) for c in (0, 1)])


def _dim_extents(prof: _Profiles) -> dict:
    """§4.9 physical range of each section quantity (H_out of a station window): lateral dims within the sensor's x field of view,
    vertical dims within its calibrated z window, the bevel angle within [0°, 90°) (the template's own bound)."""
    ok = prof.ok & np.isfinite(prof.fov_mm) & (prof.fov_mm > 0)
    if not np.any(ok):
        return {}
    fov = float(np.median(prof.fov_mm[ok]))
    wz = float(np.median(prof.win_mm2[ok] / prof.fov_mm[ok]))
    return {"gap_mm": fov, "top_opening_mm": fov, "depth_mm": wz, "hi_lo_mm": wz, "root_face_mm": wz, "bevel_angle_deg": 90.0}


def _mirror_is_hypothesis(spec) -> bool:
    """The bevel side is a discrete hypothesis only when the two mirrored templates are different models (single bevel); a V or a
    square groove is its own mirror image (the side changes the plate labels only)."""
    return str(getattr(spec, "kind", "single_bevel")) == "single_bevel"


def _q_of_station(j: int, K: int) -> list:
    q = ["root_lateral", "root_normal"]
    if j == 0:
        q.append("end_S")
    if j == K - 1:
        q.append("end_E")
    return q


def model_checks(est: SeamEstimate, ctx: dict, prior: SeamPrior, gam_prev: Optional[dict] = None) -> dict:
    """§4.9 the two decision forms the done test lacked (deep analysis §3.4, review R3 §4.3; the same k as k·σ ≤ tol):

    1. Model fit — every segment that feeds a done-test quantity gets a segment-level outlier indicator with its own posterior
       (segment_mixture: H_fit = the fused model's posterior predictive of the segment's aggregated observation, H_out = uniform over
       the physical extent; ε of the family by the rule of succession). 2·ln O > k² 'fits', < −k² 'misfit' (posterior predictive check,
       Gelman, Meng & Stern 1996, with the multiplicity of the family in ε — Scott & Berger 2010), else 'ambiguous'. The EM weight
       1 − γ down-weights the segment in the next round of the estimate (legs, station windows; end readings inside fuse_ends; station
       dims inside seam_dims). Families:
         legs      the raw offset of each leg / class (Huber GLS mean of its groove profiles' residuals about the line without its
                   own offset, from a median start; centred on the passes' median — the gauge frame is the passes' mean) vs
                   N(0, σ_prior² + its data covariance): σ_η for the passes, σ_he for the dwell / transit classes; extent = sensor window;
         windows   the station window's mean root residual (w, n) vs N(0, σ_w² + τ_root² + σ_line²) — only for windows that show this
                   seam (hypothesis below); extent = sensor window. Feeds root_lateral / root_normal at that station;
         end readings  each reading's leave-one-out residual against the converged robust fusion (in fuse_ends); extent = the leg's
                   span. Feeds end_X;
         end regions   the groove change point vs the end of the plates' own presence evidence (no groove containment, no design
                   length) — two readings of the same end (user 2026-10-09: the seam ends where both plates are present); extent = the
                   leg's span. Feeds end_X. 'ambiguous' is reported only (the end-face slant between the root and the plate top is not
                   in this model);
         station dims  in seam_dims (report: the pooled dim never rests on one station).
       A segment whose predictive is no narrower than its extent is 'untestable' (segment_mixture: no indicator).
    2. Discrete hypotheses that the done-test quantities rest on, 2·ln odds > k² or the quantities are unresolved:
         bevel side   the pooled log Bayes factor of the mirrored templates (Σ over profiles of ln[(Z₊ + Z_alt)/(Z₋ + Z_alt)], the
                      marginal of 'groove or none' per profile — a proper log BF for independent profiles), signed by the decided
                      side (a prior side that the data contradict has negative odds). Single bevel only;
         this seam at each station window (incl. the end windows = the end regions): Σ_i ln[P_in,i·((1 − ε)·e^{λ_i} + ε) + 1 − P_in,i]
                      — the window is this seam's groove at the fused line (per-profile λ^line, iid per-profile outliers at the run's
                      rate ε) vs 'not this seam'; P_in = posterior probability that the profile lies between the ends. < −k² is a
                      misfit (the seam model says the seam is there).
    A done-test quantity that rests on a 'misfit' segment is reported in blocking (kind 'misfit' → MODEL_MISFIT after the budget);
    on an unresolved hypothesis or an ambiguous window fit (kind 'unresolved'). Returns the report (JSON-able) + '_seg_w' (per-profile
    weights for the next round) + '_gam'."""
    k = float(prior.k)
    k2 = k * k
    line, fz, win, s_true, prof = ctx["line"], ctx["fz"], ctx["win"], ctx["s_true"], ctx["prof"]
    f0 = line["frame0"]
    L = _line_from(line)
    blocking = []
    hyp = {}
    q_all = [q for q in Q_NAMES if q in ("root_lateral", "root_normal", "end_S", "end_E")
             or (prior.precision is not None and prior.precision.for_dim(q, prior.spec))]
    # ---------------------------------------------------------------- hypothesis: bevel side
    if _mirror_is_hypothesis(prior.spec):
        bs = (est.plate or {}).get("bevel_side") or {}
        bbf = bs.get("pooled_ln_bf_w")
        lo_side = float(est.bevel_sign_w) * float(bbf) if (bbf is not None and math.isfinite(float(bbf))) else 0.0
        hyp["bevel_side"] = {"side_w": int(est.bevel_sign_w), "two_ln_odds": 2.0 * lo_side, "resolved": bool(2.0 * lo_side > k2),
                             "from": "prior (job)" if prior.bevel_sign_w is not None else "evidence",
                             "per_leg_ln_bf_w": bs.get("per_leg_ln_bf_w") or {}}
        if not 2.0 * lo_side > k2:
            blocking.append({"quantities": q_all, "kind": "unresolved", "segment": "hypothesis bevel_side",
                             "why": f"bevel side {'+w' if est.bevel_sign_w > 0 else '-w'} ({hyp['bevel_side']['from']}): 2 ln odds "
                                    f"{2.0 * lo_side:.1f} <= k^2 {k2:g}" + ("" if bbf is not None else " (no leg was fitted with both "
                                                                                                         "mirrored templates)")})
    else:
        hyp["bevel_side"] = {"resolved": True, "note": f"{prior.spec.kind}: the mirrored templates are the same model"}
    # ---------------------------------------------------------------- 'this seam' per profile, per leg (hypothesis) and per window
    s_st = np.asarray(win.get("s_st", np.zeros(0)), dtype=float)
    idx = win.get("idx") or []
    K = len(s_st)
    ok = prof.ok
    lam = np.where(np.isfinite(line["lam_line"]), line["lam_line"], 0.0)
    # where the seam is for 'inside the seam': the delivered ends, or the planning ends while an end's σ is not established
    pe_ = plan_ends(est)
    eS, eE = [(est.ends[X].s_m, est.ends[X].sigma_m) if math.isfinite(float(est.ends[X].sigma_m))
              else (pe_[X]["s_m"], pe_[X]["sigma_m"]) for X in ("S", "E")]
    with np.errstate(all="ignore"):
        Pin = (norm.cdf((s_true - float(eS[0])) / max(float(eS[1]), 1e-9))
               * norm.cdf((float(eE[0]) - s_true) / max(float(eE[1]), 1e-9)))
    Pin = np.where(ok & np.isfinite(Pin), Pin, 0.0)
    pic = _logistic(lam)
    # per-profile log BF 'this seam at the fused line (iid per-profile outliers at the run's rate ε)' vs 'not', for a profile inside
    # the seam with probability P_in. A leg's own indicator ('the leg shows this seam') is the sum over its profiles of the same
    # term with the leg's one-step-ahead λ where WP1 has one (its fit against the estimate before the leg — the leg is not in that
    # predictive, so a leg cannot make itself 'this seam' by dragging the line and its own jitter: lab g1d_p15_04 pass 1 at pitch +15,
    # template fits on the groove's top corners ±14 mm aside, 63 of 1360 profiles this seam), else λ^line. ε is estimated over the
    # legs that show this seam (EM: a leg's indicator takes it out of ε)
    lam_leg = np.where(np.isfinite(prof.lam_seq), prof.lam_seq, lam)
    leg_ix = [prof.idx(i) for i in range(len(prof.legs))]
    g_sup = np.zeros(len(prof.legs))
    lp = np.zeros(prof.N)
    eps_p = float("nan")
    lo_sup = np.full(len(prof.legs), np.nan)
    eps_L = float("nan")
    for _ in range(EM_ITERS):
        wl = np.ones(prof.N)
        for i_l, ix in enumerate(leg_ix):
            wl[ix] = 1.0 - g_sup[i_l]
        eps_p = (float(np.sum(wl * Pin * (1.0 - pic))) + 1.0) / (float(np.sum(wl * Pin)) + 2.0)
        with np.errstate(divide="ignore"):
            lp = np.logaddexp(np.log(Pin) + np.logaddexp(lam + math.log1p(-eps_p), math.log(eps_p)), np.log1p(-Pin))
            lpl = np.logaddexp(np.log(Pin) + np.logaddexp(lam_leg + math.log1p(-eps_p), math.log(eps_p)), np.log1p(-Pin))
        lp = np.where(ok, lp, 0.0)
        lpl = np.where(ok, lpl, 0.0)
        S_leg = np.array([float(np.sum(lpl[ix])) for ix in leg_ix])
        has = np.array([float(np.sum(Pin[ix])) > 0.0 for ix in leg_ix], dtype=bool)
        lnf = np.where(has, S_leg, np.nan)                         # a leg with no profile inside the seam is not testable
        lo_sup, g_new, eps_L = segment_mixture(lnf, 0.0)
        done_ = float(np.max(np.abs(g_new - g_sup), initial=0.0)) < EM_TOL
        g_sup = g_new
        if done_:
            break
    wl = np.ones(prof.N)
    for i_l, ix in enumerate(leg_ix):
        wl[ix] = 1.0 - g_sup[i_l]
    support = np.array([float(np.sum(wl[ix] * lp[ix])) for ix in idx]) if K else np.zeros(0)
    legs_sup = []
    for i_l, lg in enumerate(prof.legs):
        if not math.isfinite(lo_sup[i_l]):
            continue
        v_ = segment_verdict(2.0 * float(lo_sup[i_l]), k)
        legs_sup.append({"segment": "leg:" + lg.name, "this_seam_two_ln_bf": 2.0 * float(np.sum(lpl[leg_ix[i_l]])),
                         "evidence": "one-step-ahead (WP1 against the previous estimate)" if np.any(np.isfinite(
                             prof.lam_seq[leg_ix[i_l]])) else "fused line",
                         "this_seam_profiles": int(np.sum(_logistic(lam_leg[leg_ix[i_l]]) > 0.5)), "profiles": int(leg_ix[i_l].size),
                         "two_ln_odds": 2.0 * float(lo_sup[i_l]), "gamma_out": float(g_sup[i_l]),
                         "verdict": {"fits": "this seam", "misfit": "not this seam", "ambiguous": "unresolved"}[v_]})
    # ---------------------------------------------------------------- station windows: the root's local deviation (robust random effect)
    # The root at a station = the line + a local deviation d_j ~ (1 − ε_w)·N(0, τ_root²) + ε_w·U(sensor window) (the line misfit
    # τ_root of §4.6 with its own segment indicator, Box & Tiao): a window that the line + waviness does not predict is not pulled to
    # the line — its own data measure the root there (posterior mean line + γ·ē, variance (1−γ)(σ_line² + τ²) + γ(σ_line² + σ_w²) +
    # γ(1−γ)ē²); τ_root is re-estimated without it (REML weighted by 1 − γ). Only windows that show this seam (hypothesis above)
    # carry a location; a window that does not cannot be measured or predicted as this seam's root
    mus, sgs = np.asarray(win.get("mu_m", np.zeros((K, 2)))), np.asarray(win.get("sigma_m", np.zeros((K, 2))))
    tau_r = np.asarray(win.get("tau_root_m", (0.0, 0.0)), dtype=float)
    lsig = np.asarray(win.get("line_sigma_m", np.zeros((K, 2))))
    loc_on = np.array([2.0 * support[j] > k2 and bool(np.all(np.isfinite(mus[j]))) and bool(np.all(np.isfinite(sgs[j])))
                       for j in range(K)], dtype=bool) if K else np.zeros(0, bool)
    lo_loc = np.full(K, np.nan)
    g_loc = np.zeros(K)
    z_loc = np.full(K, np.nan)
    eps_w = float("nan")
    if np.any(loc_on):
        jj = np.flatnonzero(loc_on)
        x = mus[jj] * 1e3
        ext = np.log([float(np.median(prof.win_mm2[idx[j]])) for j in jj])
        g_ = np.zeros(jj.size)
        tau_mm = tau_r * 1e3
        for _ in range(EM_ITERS):
            V = sgs[jj] ** 2 * 1e6 + tau_mm[None, :] ** 2 + lsig[jj] ** 2 * 1e6                      # mm²
            q = np.sum(x * x / V, axis=1)
            lnf0 = -math.log(2.0 * math.pi) - 0.5 * np.sum(np.log(V), axis=1)
            lo_, g_new, eps_w = segment_mixture(lnf0 - 0.5 * q, ext, g_, lnf_peak=lnf0)
            g_new = np.nan_to_num(g_new)
            if jj.size >= 2:
                tau_new = np.array([float(_reml_rows(x[:, c][None, :], (sgs[jj, c] * 1e3)[None, :] ** 2,
                                                     np.maximum(1.0 - g_new, 0.0)[None, :])[2][0]) for c in (0, 1)])
            else:
                tau_new = tau_mm
            done_ = float(np.max(np.abs(g_new - g_))) < EM_TOL and float(np.max(np.abs(tau_new - tau_mm))) <= 1e-6
            g_, tau_mm = g_new, tau_new
            if done_:
                break
        lo_loc[jj], g_loc[jj], z_loc[jj] = lo_, g_, _equiv_z(q, 2)
        tau_r = tau_mm * 1e-3
    # station roots (user 2026-10-11: deliver the measured root as it is): root(s_j) = line(s_j) + d_j, d_j the local deviation — a
    # random effect N(0, τ_root²) (REML above) or, for a window the line + waviness does not predict, an outlier deviation (γ_j);
    # delivered = the posterior (Henderson's BLUP, Robinson 1991): E = line + [(1 − γ)·κ + γ]·ē_j, κ = τ²/(τ² + σ_w²) per component,
    # prediction-error variance (1 − γ)·(σ_line² + κ·σ_w²) + γ·(σ_line² + σ_w²) + γ(1 − γ)·((1 − κ)·ē)²  (σ_line² + τ²σ_w²/(τ²+σ_w²) for
    # a window that fits; the line's own σ is kept whole — the window's share of the line is not removed: conservative). A station
    # without this seam's data keeps the line with the marginal σ_line² + τ². The line itself stays the planning model.
    st_shift = np.zeros((K, 2))
    for j, st in enumerate(est.stations[:K]):
        l2 = lsig[j] ** 2
        if loc_on[j] and math.isfinite(g_loc[j]):
            g1 = float(g_loc[j])
            d = mus[j]
            s2 = sgs[j] ** 2
            t2 = tau_r ** 2
            kap = np.where(t2 + s2 > 0, t2 / np.maximum(t2 + s2, 1e-300), 0.0)
            v_fit = l2 + kap * s2
            v_ = (1.0 - g1) * v_fit + g1 * (l2 + s2) + g1 * (1.0 - g1) * ((1.0 - kap) * d) ** 2
            shift = ((1.0 - g1) * kap + g1) * d
            st.root_m = np.asarray(st.root_m, dtype=float) + shift[0] * np.asarray(st.w) + shift[1] * np.asarray(st.n)
            sh = shift * 1e3
            for fc in (st.faces or {}).values():                    # face lines are relative to the station root
                if fc.get("line_wn_mm") is not None:
                    fc["line_wn_mm"] = [fc["line_wn_mm"][0] - float(sh[0]), fc["line_wn_mm"][1] - float(sh[1])] + list(fc["line_wn_mm"][2:])
        else:
            shift = np.zeros(2)
            v_ = l2 + tau_r ** 2
        st.root_sigma_m = (float(math.sqrt(max(v_[0], 0.0))), float(math.sqrt(max(v_[1], 0.0))))
        st_shift[j] = shift
    wins = []
    for j in range(K):
        sv = segment_verdict(2.0 * support[j], k)
        sup = {"fits": "supported", "misfit": "contradicted", "ambiguous": "unresolved"}[sv]
        lv = segment_verdict(2.0 * lo_loc[j], k) if loc_on[j] else None
        rec = {"s_m": float(s_st[j]), "n": int(len(idx[j])), "this_seam_two_ln_bf": 2.0 * float(support[j]), "this_seam": sup}
        if loc_on[j]:
            rec.update(root_mean_mm=[float(v_ * 1e3) for v_ in mus[j]], z=float(z_loc[j]), two_ln_odds=2.0 * float(lo_loc[j]),
                       gamma_out=float(g_loc[j]), verdict=lv, root_shift_mm=[float(v_ * 1e3) for v_ in st_shift[j]])
        wins.append(rec)
        seg = f"station window s={float(s_st[j]) * 1e3:.1f} mm"
        if sup != "supported":
            blocking.append({"quantities": _q_of_station(j, K), "kind": "misfit" if sup == "contradicted" else "unresolved",
                             "segment": seg, "why": f"this seam in the window: 2 ln BF {2.0 * support[j]:.1f} "
                                                   f"{'< -' if sup == 'contradicted' else 'within +-'}k^2 ({len(idx[j])} profiles)"})
    # ---------------------------------------------------------------- legs / classes (offsets)
    F = line.get("_F") or {}
    legs_rep = []
    sw_leg = np.ones(prof.N)
    sel = np.asarray(line.get("used", ok), dtype=bool)
    if F and np.any(sel):
        selix = np.flatnonzero(sel)
        e, Vi, ss_f = np.asarray(F["e"]), np.asarray(F["Vi"]), np.asarray(F["s"])
        Eo = np.stack([prof.ex[sel], prof.ez[sel]], axis=1)
        Lf = LineModel(f0, F["params"])
        t = Lf.t(ss_f)
        mm_ = prof.m[sel]
        EQ = Eo - np.einsum("nki,ni->nk", Eo, t)[:, :, None] * mm_[:, None, :] / np.einsum("ni,ni->n", mm_, t)[:, None, None]
        B = np.stack([EQ @ f0["w0"], EQ @ f0["n0"]], axis=2)                                   # (n,2,2)
        # the leg's own aggregate: profiles weighted by their groove (shape) evidence only — λ^line already holds the location against
        # the current line, the quantity under test — and a Huber M-estimate about the leg's own mean (per-profile outliers are the
        # per-profile mixture's, they must not move the leg's aggregate)
        wv = (_logistic(prof.lam) * wl)[sel]
        he = float(prior.he_sigma_m) if prior.he_sigma_m is not None else float("nan")
        # every profile's own offset (class + leg) in (w0, n0): the raw position of a class / leg relative to the line is its
        # residual with that offset added back
        o_prof = np.zeros((int(sel.sum()), 2))
        cl_s, lg_s = prof.cls[sel], prof.leg[sel]
        for c, (ow, on) in (line.get("offsets") or {}).get("class", {}).items():
            o_prof[cl_s == c] += (ow, on)
        for i_l in F["leg_col"]:
            o_prof[lg_s == i_l] += np.asarray(F["offsets"]["leg"][prof.legs[i_l].name])
        # classes: EVERY pose class, the gauge class too — the gauge (β_pass ≡ 0) is a frame convention, physically each class carries
        # its hand-eye residual σ_he; the family is centred on the classes' median (a pass that reads another feature is the outlier,
        # not the transit and the dwells that agree with each other — lab g1d_p15_04). Legs: within the gauge class (σ_η), centred on
        # the passes' median
        segs = []
        for c in [c_ for c_ in CLASS_ORDER if np.any(cl_s == c_)] + sorted(set(cl_s) - set(CLASS_ORDER)):
            segs.append(("class:" + c, cl_s == c, np.array([he, he]), "class"))
        swn = np.asarray(line.get("sigma_eta_wn", (line["sigma_eta"],) * 2), dtype=float)
        for i_l in F["leg_col"]:
            segs.append(("leg:" + prof.legs[i_l].name, lg_s == i_l, swn, "leg"))      # η per component (w, n)
        rows = []
        for name, mem, sp, fam in segs:
            if not (np.any(mem) and np.all(np.isfinite(sp))):
                continue
            # class: add back the class offset (and the legs' η around it); leg: add back its η only
            o_add = o_prof[mem] if fam == "class" else np.asarray(F["offsets"]["leg"][name.split(":", 1)[1]])[None, :]
            r_ = e[mem] + np.einsum("nij,nj->ni", B[mem], np.broadcast_to(o_add, (int(mem.sum()), 2)))
            BtV = np.einsum("nji,njk->nik", B[mem], Vi[mem])                                    # Bᵀ V⁻¹
            Bm, Vm, w_m = B[mem], Vi[mem], wv[mem]
            if not np.any(w_m > 0):
                continue
            xr = _leg_start(Bm, r_, w_m)
            Pd = None
            for _ in range(HUBER_ROUNDS):                       # [R] Huber IRLS of the leg's mean
                d_ = r_ - np.einsum("nij,j->ni", Bm, xr)
                hh = _huber_w(np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", d_, Vm, d_), 0.0)))
                ww = w_m * hh
                Pd = np.einsum("n,nik,nkj->ij", ww, BtV, Bm)
                bb = np.einsum("n,nik,nk->i", ww, BtV, r_)
                if not (np.all(np.isfinite(Pd)) and np.linalg.det(Pd) > 0):
                    Pd = None
                    break
                xr = np.linalg.solve(Pd, bb)
            if Pd is None:
                continue
            Vp = (np.diag(sp * sp) + np.linalg.inv(Pd)) * 1e6                                     # mm²
            rows.append([name, mem, xr * 1e3, Vp, 0.0, 0.0, math.log(float(np.median(prof.win_mm2[selix[mem]]))), fam])
        # each family centred on its members' median (the high-breakdown start of the EM, as the line's own S-start, _robust_scales):
        # the frame of the current line is itself pulled by an outlying member; the next round's line is fitted with the indicator
        # weights and the median residual is ≈ 0. The members are the segments IN this round's line: a segment the previous round
        # took out outright (every profile at weight 0 — e.g. a leg decided not this seam) defines no part of the frame; it is
        # tested against the others but is not a member of the centre (test_sim_teach1g m2, w_x calibration: two of four passes
        # out after round 1 — the median of four sat halfway between them and the two passes that define the line, and every
        # pass was a misfit in round 2)
        sw_in = ctx.get("seg_w")
        for fam in ("class", "leg"):
            fr = [r_ for r_ in rows if r_[7] == fam]
            if not fr:
                continue
            mem_in = [r_ for r_ in fr if sw_in is None or float(np.max(np.asarray(sw_in)[selix[r_[1]]])) > 0.0] or fr
            c_mm = np.median(np.array([r_[2] for r_ in mem_in]), axis=0)
            for r_ in fr:
                r_[2] = r_[2] - c_mm
                r_[4] = float(r_[2] @ np.linalg.solve(r_[3], r_[2]))
                r_[5] = -0.5 * r_[4] - math.log(2.0 * math.pi) - 0.5 * math.log(np.linalg.det(r_[3]))
            lo_, g_, eps_l = segment_mixture([r_[5] for r_ in fr], [r_[6] for r_ in fr], lnf_peak=[r_[5] + 0.5 * r_[4] for r_ in fr])
            for (name, mem, xm, Vp, qq, _, _, _), lo1, g1 in zip(fr, lo_, g_):
                legs_rep.append({"segment": name, "mean_residual_mm": [float(v_) for v_ in xm],
                                 "sigma_pred_mm": [float(math.sqrt(Vp[0, 0])), float(math.sqrt(Vp[1, 1]))],
                                 "z": float(_equiv_z(qq, 2)), "two_ln_odds": 2.0 * float(lo1), "gamma_out": float(np.nan_to_num(g1)),
                                 "verdict": segment_verdict(2.0 * float(lo1), k)})
                sw_leg[selix[mem]] *= (1.0 - float(np.nan_to_num(g1)))
    # ---------------------------------------------------------------- ends: readings (fuse_ends) and end regions (groove vs plates)
    ends_rep = {}
    pe = (est.plate or {}).get("plate_ends") or {}
    reg = []
    for X in ("S", "E"):
        # the delivered end's readings: the plate-end readings (end_faces; their fit = the fusion's gross-error class on
        # leave-one-out predictions)
        rds = [r for r in (est.ends[X].readings or []) if r.get("fit")]
        verd = [r["fit"]["verdict"] for r in rds]
        ends_rep[X] = {"readings": [{"leg": r.get("leg"), "plate": r.get("plate"), **r["fit"]} for r in rds]}
        two_X = [r for r in (est.ends[X].readings or []) if r.get("kind") in ("two_sided", "top_edge")]
        if verd and len(rds) == len(two_X) and all(v_ == "misfit" for v_ in verd):
            blocking.append({"quantities": ["end_" + X], "kind": "misfit", "segment": f"end {X} readings",
                             "why": "every gauge reading of the end is a misfit: " + ", ".join(
                                 f"{r.get('leg')} z {r['fit']['z']:.1f}" for r in rds)})
        inner = ((pe.get(X) or {}).get("inner_own_evidence") or (pe.get(X) or {}).get("inner_fused")) if pe else None
        g_rd = (((ctx.get("fz_measured") or {}).get("ends") or {}).get(X) or {}).get("readings") or []
        span = max([float(r.get("span_m", 0.0)) for r in g_rd if r.get("kind") == "two_sided"] or [0.0])
        gm = ((ctx.get("fz_measured") or {}).get("ends") or {}).get(X) or {}      # groove end from the measurements only
        if inner and gm and gm.get("two_sided") and inner.get("two_sided") and span > 0 and math.isfinite(float(inner["sigma_m"])) \
                and math.isfinite(float(gm["sigma_m"])):
            d_ = float(gm["s_m"]) - float(inner["s_m"])
            v_ = float(gm["sigma_m"]) ** 2 + float(inner["sigma_m"]) ** 2
            if v_ > 0:
                reg.append((X, d_, v_, -0.5 * d_ * d_ / v_ - 0.5 * math.log(2.0 * math.pi * v_), math.log(span),
                            -0.5 * math.log(2.0 * math.pi * v_)))
    if reg:
        lo_, g_, eps_r = segment_mixture([r_[3] for r_ in reg], [r_[4] for r_ in reg], lnf_peak=[r_[5] for r_ in reg])
        for (X, d_, v_, _, _, _), lo1, g1 in zip(reg, lo_, g_):
            vv = segment_verdict(2.0 * float(lo1), k)
            ends_rep[X]["region"] = {"groove_minus_plates_mm": d_ * 1e3, "sigma_mm": math.sqrt(v_) * 1e3, "z": abs(d_) / math.sqrt(v_),
                                     "two_ln_odds": 2.0 * float(lo1), "gamma_out": float(g1), "verdict": vv,
                                     "note": "the groove change point vs the plates' presence evidence (both whole-profile "
                                             "readings of the joint's end; the delivered end is the plate-end model)"}
            # the declared joint ends where its groove walls AND both plates end: the plate-end model reads the plate tops; the groove
            # change point (walls, backing) and the plates' presence evidence are two whole-profile readings of the same end — when
            # they decisively disagree, something at that end is not the declared joint (a flush jig, a ghost band): blocking
            ends_rep[X]["region"]["blocking"] = bool(vv == "misfit")
            if ends_rep[X]["region"]["blocking"]:
                g_s = float(((ctx.get("fz_measured") or {}).get("ends") or {})[X]["s_m"])
                blocking.append({"quantities": ["end_" + X], "kind": "misfit", "segment": f"end region {X}",
                                 "why": f"groove change point {g_s * 1e3:.1f} mm vs both plates present "
                                        f"{(g_s - d_) * 1e3:.1f} mm: {d_ * 1e3:+.1f} mm, z {abs(d_) / math.sqrt(v_):.1f}"})
    # ---------------------------------------------------------------- station dims (seam_dims, report)
    dims_rep = {}
    for q, d in ((est.plate or {}).get("seam_dims") or {}).items():
        sf = d.get("station_fit")
        if sf:
            dims_rep[q] = {k_: sf[k_] for k_ in ("n_fits", "n_ambiguous", "n_misfit", "max_abs_z", "outlier_eps")}
            dims_rep[q]["not_fitting"] = sf.get("not_fitting", [])[:12]
    # ---------------------------------------------------------------- segment weights for the next round: the legs (a leg feeds
    # every quantity; a window's indicator acts on its own station above, the readings' inside fuse_ends)
    seg_w = sw_leg * wl
    # legs that decisively are not this seam (their own evidence) or sit decisively off it (their class / leg offset is a misfit)
    off_cls = {l_["segment"].split(":", 1)[1] for l_ in legs_rep if l_["verdict"] == "misfit" and l_["segment"].startswith("class:")}
    off_leg = {l_["segment"].split(":", 1)[1] for l_ in legs_rep if l_["verdict"] == "misfit" and l_["segment"].startswith("leg:")}
    not_seam = sorted({l_["segment"].split(":", 1)[1] for l_ in legs_sup if l_["verdict"] == "not this seam"} | off_leg
                      | {lg.name for lg in prof.legs if lg.cls in off_cls})
    n_seg = {"windows": int(np.count_nonzero(loc_on)), "legs": len(legs_rep), "legs_this_seam": len(legs_sup),
             "end_readings": sum(len(v_["readings"]) for v_ in ends_rep.values()), "end_regions": len(reg)}
    verdicts = ([w_.get("verdict") for w_ in wins if w_.get("verdict")] + [l_["verdict"] for l_ in legs_rep]
                + [r_["verdict"] for v_ in ends_rep.values() for r_ in v_["readings"]]
                + [v_["region"]["verdict"] for v_ in ends_rep.values() if "region" in v_])
    summary = {"segments": n_seg, "fits": verdicts.count("fits"), "ambiguous": verdicts.count("ambiguous"),
               "misfit": verdicts.count("misfit"), "untestable": verdicts.count("untestable"),
               "this_seam_unresolved": sum(1 for w_ in wins if w_["this_seam"] == "unresolved"),
               "this_seam_contradicted": sum(1 for w_ in wins if w_["this_seam"] == "contradicted"),
               "hypotheses_unresolved": [h for h, v_ in hyp.items() if not v_.get("resolved", True)],
               "legs_not_this_seam": not_seam,
               "windows_root_from_own_data": sum(1 for w_ in wins if w_.get("verdict") == "misfit"),
               "misfit_blocking": [b["segment"] + ": " + b["why"] for b in blocking if b["kind"] == "misfit"],
               "unresolved_blocking": [b["segment"] + ": " + b["why"] for b in blocking if b["kind"] == "unresolved"]}
    return {"k": k, "summary": summary, "hypotheses": hyp, "blocking": blocking,
            "windows": {"profile_outlier_eps": eps_p, "window_outlier_eps": eps_w, "leg_outlier_eps": eps_L,
                        "tau_root_mm": [float(v_ * 1e3) for v_ in tau_r],
                        "stations": wins},
            "legs": legs_rep, "legs_this_seam": legs_sup, "ends": ends_rep, "dims": dims_rep, "_seg_w": seg_w,
            "_gam": {"legs_not_this_seam": set(not_seam)},
            "_unsupported": [j for j, w_ in enumerate(wins) if w_["this_seam"] != "supported"]}


def delivered_length(est: SeamEstimate) -> tuple:
    """The measured weld length between the delivered ends (m) and its σ: the S→E contract length of sections.json (scan_core.
    seam_frame: (E − S)·u, u = unit(E − S) ⟂ the plate normal) between the root-line points r(s_S), r(s_E), with the ends' σ (along the
    taught axis s) carried through dL/ds = r'(s)·u at each end (independent ends). Not the taught-axis difference s_E − s_S: that is
    shorter by the cosine of the taught line's angle to the seam (kv_offA_a 2026-10-12: 372.99 vs 379.46 mm). (nan, nan) when an end
    is not a finite point."""
    from .seam_frame import SeamFrameError, seam_frame
    eS, eE = (est.ends or {}).get("S"), (est.ends or {}).get("E")
    if eS is None or eE is None:
        return float("nan"), float("nan")
    try:
        fr = seam_frame(est.r(float(eS.s_m)), est.r(float(eE.s_m)), est.n_plate)
    except SeamFrameError:
        return float("nan"), float("nan")
    g = [abs(float(np.asarray(est.dr_ds(float(e.s_m)), dtype=float) @ fr.u)) for e in (eS, eE)]
    return fr.length_m, math.hypot(g[0] * float(eS.sigma_m), g[1] * float(eE.sigma_m))


def design_deviations(est: SeamEstimate, prior: SeamPrior, fz_measured: Optional[dict] = None) -> tuple:
    """(b) of the user rule 2026-10-11: the measured joint vs its DESIGN values — a report, never a judgement (the estimates are
    measurements; design values neither pull nor judge them). Length (the ends fused without the design length) and the pooled section
    dims vs the job's design; WARNING when |measured − design| > the design tolerance (qc_targets), 'established' when even
    |measured − design| − k·σ > tol. Returns ({quantity: record}, [warning lines, English])."""
    k = float(prior.k)
    spec, tol = prior.spec, prior.tol
    rows = {}
    fzm = fz_measured or {}
    measured_both = all(bool(((fzm.get("ends") or {}).get(X) or {}).get("two_sided")) for X in ("S", "E"))
    if measured_both and fzm.get("length_m") is not None and math.isfinite(float(fzm["length_m"])):    # both ends observed
        rows["weld_length_mm"] = (float(fzm["length_m"]) * 1e3, float(fzm.get("length_sigma_m", float("nan"))) * 1e3,
                                  float(spec.weld_length_mm), tol.length_mm)
    sd = (est.plate or {}).get("seam_dims") or {}
    for q, design, t in (("gap_mm", float(spec.root_gap_mm), tol.gap_mm), ("depth_mm", float(spec.groove_depth_mm), tol.depth_mm),
                         ("bevel_angle_deg", float(spec.bevel_angle_deg), tol.bevel_deg),
                         ("top_opening_mm", float(spec.top_opening_mm), tol.opening_mm)):
        d = sd.get(q) or {}
        if math.isfinite(float(d.get("value", float("nan")))):
            rows[q] = (float(d["value"]), float(d.get("sigma", float("nan"))), design, t)
    out, warn = {}, []
    for q, (val, sg, design, t) in rows.items():
        dv = val - design
        rec = {"measured": val, "sigma": sg, "design": design, "deviation": dv, "tol": t,
               "z_vs_design": dv / sg if (math.isfinite(sg) and sg > 0) else None}
        if t:
            rec["warning"] = bool(abs(dv) > float(t))
            rec["established"] = bool(math.isfinite(sg) and abs(dv) - k * sg > float(t))
            if rec["warning"]:
                warn.append(f"{q} measured {val:.2f} (sigma {sg:.2f}) vs design {design:.2f}: {dv:+.2f}, beyond the design tolerance "
                            f"{float(t):g}" + (f" by more than k sigma" if rec["established"] else ""))
        out[q] = rec
    return out, warn


def estimate(legs: list, prior: SeamPrior, *, map_fn=None, n_children: int = 1,
             edge_groups_out: Optional[list] = None, vc_out: Optional[dict] = None) -> SeamEstimate:
    """§4: all legs → SeamEstimate (line, plate, ends, δ, offsets, stations, done test). map_fn / n_children: the worker's fork_map
    for the plate-end readings (independent given the calibrated sensor); edge_groups_out: a list that receives the top-edge data of
    the plate ends (edge-response calibration, end_faces.pool_edge_response); vc_out: a dict that receives the final round's
    variance-component evidence ('end': the plate-end fusion's restricted-likelihood design, end_faces.vc_loglik; 'line': the leg
    offsets' REML profile, line_eta_loglik) — the noise-prior calibration (scan_l0.calib_noise_priors)."""
    f0 = prior.frame0()
    legs = [lg for lg in legs if lg.n > 0]
    k = float(prior.k)
    ev_legs = legs                                     # D11 at a site is WP3's (first grab); here: any leg's best segment
    log_bf_run = max([groove_evidence(lg, prior=prior) for lg in ev_legs], default=0.0)
    found = bool(2.0 * log_bf_run > k ** 2)                                             # D11
    bsw, bbf, bf_per = _bevel_sign_w(legs, prior, f0)
    if not found:
        nanP = np.zeros((2, 2))
        return SeamEstimate(groove_found=False, log_bf_run=log_bf_run, line_model="line", line_params=nanP,
                            line_cov=np.full((4, 4), np.inf), u=f0["u0"], n_plate=f0["n0"], plate={"bevel_side_ln_bf": bbf},
                            ends={}, dt_s=float(prior.dt_s), dt_sigma_s=float(timing_prior(legs, f0["u0"])["sigma_s"]),
                            class_offset_m={},
                            leg_eta_m={}, sigma_zeta_m={}, tau_end_m=float("nan"), bevel_sign_w=bsw, stations=[],
                            ratios={}, done=False, binding=[], unresolved=list(Q_NAMES), frame0=f0)
    prof = _Profiles(legs, f0)
    # §4.9 EM over the segment-level outlier indicators: estimate → segment checks → weights → estimate again while a profile's weight
    # still changes (normal data: the indicators are ≈ 0, one round)
    seg_w, gam_prev, rounds = None, {}, 0
    while True:
        est, ctx = _estimate_round(legs, prior, prof, seg_w, log_bf_run=log_bf_run, bsw=bsw, bbf=bbf, bf_per=bf_per,
                                   not_seam=gam_prev.get("legs_not_this_seam") or set(), map_fn=map_fn, n_children=n_children,
                                   edge_groups_out=edge_groups_out if rounds == 0 else None, vc_out=vc_out)
        chk = model_checks(est, ctx, prior, gam_prev=gam_prev)
        rounds += 1
        new_w = chk.pop("_seg_w")
        gam_prev = chk.pop("_gam")
        old_w = np.ones(prof.N) if seg_w is None else seg_w
        if float(np.max(np.abs(new_w - old_w), initial=0.0)) < REWEIGHT_TOL or rounds > REWEIGHT_ROUNDS:
            break
        seg_w = new_w
    chk["em_rounds"] = rounds
    # a station window that does not show this seam carries no section of this seam (its profiles are explained as 'not this seam'):
    # no dims, no faces — and the pooled dims and the plate-top point are taken without it
    unsup = chk.pop("_unsupported", [])
    if unsup:
        for j in unsup:
            st = est.stations[j]
            st.dims = {nm: {"value": float("nan"), "sigma": math.inf, "sigma_sys": float("nan"), "n": 0.0} for nm in st.dims}
            st.faces = {nm: {"normal": None, "point_m": None, "line_wn_mm": None, "sigma_deg": math.inf, "n": 0} for nm in st.faces}
        sig_eps = float(prior.sigma_eps) if prior.sigma_eps is not None else float("nan")
        est.plate["seam_dims"] = seam_dims(est.stations, sig_eps, extents=_dim_extents(prof), k=k)
    est.plate.update(_top_point(est.stations, est.n_plate))             # with the stations' final root σ (§4.9 local deviations)
    est.plate["model_checks"] = chk
    # the delivered ends' measured length (the plate-end model's ends; the groove change point where the legs carry no measured points):
    # the seam length between them, as sections.json reports it - not their taught-axis difference (delivered_length)
    L_del, sig_del = delivered_length(est)
    fz_del = {"ends": {X: {"two_sided": bool(est.ends[X].two_sided)} for X in ("S", "E")},
              "length_m": L_del, "length_sigma_m": sig_del}
    est.plate["design_deviations"], est.plate["warnings"] = design_deviations(est, prior, fz_del)
    done, ratios, unresolved = done_test(est, prior.precision, prior.spec)
    est.done, est.ratios, est.unresolved = done, ratios, unresolved
    est.binding = [max(ratios, key=lambda q: ratios[q])] if ratios else []
    return est


def _estimate_round(legs: list, prior: SeamPrior, prof: _Profiles, seg_w: Optional[np.ndarray], *, log_bf_run: float, bsw: int,
                    bbf, bf_per: dict, not_seam: Optional[set] = None, map_fn=None, n_children: int = 1,
                    edge_groups_out: Optional[list] = None, vc_out: Optional[dict] = None) -> tuple:
    """One pass of §4.2–4.7 with the segment EM weights seg_w (None = 1) → (SeamEstimate without the done test, context for §4.9).
    not_seam = legs that the previous round decided are not this seam (their readings are no observation of an end)."""
    f0 = prior.frame0()
    k = float(prior.k)
    line = fit_line(legs, prior, _prof=prof, seg_w=seg_w)
    L = line["_line"]
    # §4.9 a leg's segment weight (1 − γ of its indicators; the same for all its profiles) also weights its end readings
    leg_w = {lg.name: (float(np.mean(seg_w[prof.idx(i)])) if (seg_w is not None and prof.idx(i).size) else 1.0)
             for i, lg in enumerate(prof.legs)}
    readings = []
    per_leg = {}
    for lg in legs:
        cp = leg_change_points(lg, line)
        per_leg[lg.name] = cp
        for X in ("S", "E"):
            if cp[X] is not None:
                cp[X]["leg_w"] = leg_w.get(lg.name, 1.0)
                cp[X]["leg_not_this_seam"] = lg.name in (not_seam or ())
        readings += [cp["S"], cp["E"]]
    # user rule 2026-10-11: delivered values and judgements from the measurements only — δ, offsets and the λ ends ("where is the
    # seam": the plate-end readings' region) come from the fusion WITHOUT the design length; the design length (Cauchy prior) only
    # plans the next action (pass extents, the world box before the ends are measured): plate['ends_plan']
    fz = fuse_ends(readings, prior, line, length_prior=False)
    fz_plan = fuse_ends([dict(r) if r is not None else None for r in readings], prior, line)
    # legs fitted with the other template mirror than the decided side measure no section of THIS groove (dims, faces, plate tops);
    # their root points and groove evidence stay in the line and the ends
    mism = [lg.name for lg in prof.legs if _leg_side_w(lg, f0) != bsw]
    pi_sec = np.asarray(line["pi"], dtype=float).copy()
    for i, lg in enumerate(prof.legs):
        if lg.name in mism:
            pi_sec[prof.idx(i)] = 0.0
    # seam ends (user 2026-10-10/11): per plate end from the measured top edge through the sensor's edge response, inner of the two;
    # legs the model checks found not to be this seam give no reading
    legs_ends = [lg for lg in legs if lg.name not in (not_seam or ())]
    vc_end = {} if vc_out is not None else None
    fe, fe_readings = _end_faces(legs_ends, prof, line, prior, pi_sec, per_leg, fz, fz_plan, map_fn=map_fn, n_children=n_children,
                                 edge_groups_out=edge_groups_out, vc_out=vc_end)
    if vc_out is not None:
        vc_out["end"] = vc_end
        vc_out["line"] = line["_F"].get("_vc")
    o_src = fe if fe.get("n_readings", 0) else fz        # δ and along-track offsets: the plate-end fusion once it has readings
    # true-frame s of every profile (δ, the direction offset and along-track offsets removed) for the stations
    delta = o_src["delta_s"]
    b_u = float(o_src.get("b_u", 0.0)) if o_src.get("b_u_identified") else 0.0
    s_true = line["s_star"].copy()
    off3 = np.zeros((prof.N, 3))
    for i, lg in enumerate(prof.legs):
        ix = prof.idx(i)
        m = prof.m[ix]
        uh = L.tangent(s_true[ix])
        mu = np.einsum("ni,ni->n", m, uh)
        c_v = np.einsum("ni,ni->n", prof.v[ix], m) / mu
        ow, on = leg_offset_wn(line, lg.name, lg.cls)        # lg = _LegView (canonical class): class + direction × b + η
        o3 = ow * f0["w0"] + on * f0["n0"]
        o3 = o3 + (o_src["beta_u"].get(lg.cls, 0.0) + leg_direction(line, lg.name) * b_u
                   + o_src["eta_u"].get(lg.name, 0.0)) * f0["u0"]
        s_true[ix] = s_true[ix] - c_v * delta + (m @ o3) / mu
        off3[ix] = o3 + delta * (-prof.v[ix])                     # measured + o − v·δ = true (time shift along travel)
    plate = plate_normals(prof, line, prior, pi_sec)
    plate["bevel_side_ln_bf"] = bbf
    plate["he_sigma_m"] = float(prior.he_sigma_m) if prior.he_sigma_m is not None else float("nan")
    plate["bevel_side"] = {"sign_w": bsw, "pooled_ln_bf_w": bbf, "per_leg_ln_bf_w": bf_per, "mismatched_legs": mism}
    n_hat = plate["normal"]
    # seam ends = where BOTH plates are present (user 2026-10-09): per-plate ends from the plate-top evidence, inner of the two
    D_d = float(prior.spec.groove_depth_mm) * 1e-3
    rd_p = {"L": [], "R": [], "inner": [], "inner_free": []}
    for lg in legs:
        pc = plate_change_points(lg, line, per_leg[lg.name], D_d, prior.spec)
        for side in ("L", "R", "inner", "inner_free"):
            if side in pc and pc[side] is not None:
                src = pc[side]
                for r in (src["S"], src["E"]):
                    if r is not None:
                        r["leg_w"] = leg_w.get(lg.name, 1.0)
                        r["leg_not_this_seam"] = lg.name in (not_seam or ())
                        rd_p[side].append(r)
    geo = prof.geometry(prior.spec)
    L_is_left = bool(np.median(geo["L_left"][prof.ok]) > 0.5) if np.any(prof.ok) else True
    side_name = {"L": "left" if L_is_left else "right", "R": "right" if L_is_left else "left"}
    if rd_p["L"] and rd_p["R"] and rd_p["inner"]:
        fzp = {side: fuse_ends(rd_p[side], prior, line, length_prior=False) for side in ("L", "R")}
        fzi = fuse_ends(rd_p["inner"], prior, line, length_prior=False)     # inner per-leg readings, known h = D (provisional)
        # §4.9: where the plates' own evidence ends — a second reading of each end, so without the design-length prior (that is shared
        # information and would pull both readings toward the design)
        fzf = fuse_ends(rd_p["inner_free"], prior, line, length_prior=False) if rd_p["inner_free"] else None
        plate_ends = {}
        for X in ("S", "E"):
            eL, eR = fzp["L"]["ends"][X], fzp["R"]["ends"][X]
            inner = ("L" if eL["s_m"] >= eR["s_m"] else "R") if X == "S" else ("L" if eL["s_m"] <= eR["s_m"] else "R")
            plate_ends[X] = {side_name[sd_]: {"s_m": fzp[sd_]["ends"][X]["s_m"], "sigma_m": fzp[sd_]["ends"][X]["sigma_m"],
                                              "point_m": L.r(fzp[sd_]["ends"][X]["s_m"]),
                                              "two_sided": fzp[sd_]["ends"][X]["two_sided"]} for sd_ in ("L", "R")}
            plate_ends[X]["inner"] = side_name[inner]
            plate_ends[X]["inner_fused"] = {"s_m": fzi["ends"][X]["s_m"], "sigma_m": fzi["ends"][X]["sigma_m"],
                                            "two_sided": fzi["ends"][X]["two_sided"]}
            if fzf is not None:
                plate_ends[X]["inner_own_evidence"] = {"s_m": fzf["ends"][X]["s_m"], "sigma_m": fzf["ends"][X]["sigma_m"],
                                                       "two_sided": fzf["ends"][X]["two_sided"]}
        plate["plate_ends"] = plate_ends
        plate["plate_end_fusion"] = {side_name[sd_]: {"tau_m": fzp[sd_]["tau_m"], "tau_dof": fzp[sd_]["tau_dof"],
                                                      "length_m": fzp[sd_]["length_m"]} for sd_ in ("L", "R")}
        plate["plate_end_fusion"]["seam"] = {"tau_m": fzi["tau_m"], "tau_dof": fzi["tau_dof"], "length_m": fzi["length_m"],
                                             "length_sigma_m": fzi["length_sigma_m"], "delta_s": fzi["delta_s"],
                                             "delta_sigma_s": fzi["delta_sigma_s"]}
        plate["plate_ends_note"] = ("provisional: per-plate presence from WP1 face-inlier counts + top-corner height; the seam ends "
                                    "use the groove change point until WP1 provides a per-plate presence log-BF")
    else:                                                # no plate-top evidence in the legs (meta['face_inliers'] absent)
        plate["plate_ends"] = None
    # seam ends = the plate-end model (measured top edges through the sensor's edge response, inner of the two plates): the delivered
    # location and σ (∞ while the variance components are not established). An end is IDENTIFIED (two_sided, the identification rule)
    # when both plates' ends there have a reading the fusion keeps as the model's. The λ end only says where the seam is (the readings'
    # region; misspecified at the ends, lab 2026-10-11 C3): the location of an end no plate reading reaches yet (σ ∞, not identified)
    # Legs without measured 3-D points (no recon record: synthetic estimator inputs) give the plate-end model nothing: then the
    # groove change points of the measurements only (fz) are the ends' only observation and are delivered as they are
    ends = {}
    w_r = fe.get("huber_weights") or []
    ends_src = "end_faces" if fe.get("has_points") else "groove change point (no measured points in the legs)"
    for X in ("S", "E"):
        fr, cr = fe["ends"][X]["root"], fz["ends"][X]
        if not fe.get("has_points"):
            ends[X] = EndEst(s_m=float(cr["s_m"]), sigma_m=float(cr["sigma_m"]), point_m=np.asarray(cr["point_m"]),
                             two_sided=bool(cr["two_sided"]), n_legs=int(cr["n_legs"]),
                             readings=[{k_: v for k_, v in r.items() if not k_.startswith("_")} for r in cr["readings"]])
            continue
        rs = [r for r in fe_readings if r["end"] == X]
        kept = {r["plate"] for r, w_ in zip(fe_readings, w_r) if r["end"] == X and w_ > 0.5}
        ident = all(p_ in kept for p_ in ("L", "R"))
        if ident and math.isfinite(float(fr["s_m"])):
            mu_, sg_ = float(fr["s_m"]), float(fr["sigma_m"])
        else:                                            # not identified by a plate reading: where the seam is (σ ∞)
            mu_ = float(cr["s_m"]) if (cr.get("two_sided") and math.isfinite(float(cr["s_m"]))) else float(fz_plan["ends"][X]["s_m"])
            sg_ = math.inf
        ends[X] = EndEst(s_m=float(mu_), sigma_m=float(sg_), point_m=np.asarray(L.r(mu_)), two_sided=bool(ident),
                         n_legs=len({r["leg"] for r in rs}),
                         readings=[{k_: v for k_, v in r.items() if not k_.startswith("_")} for r in rs])
    plate["ends_source"] = ends_src
    plate["groove_ends"] = {X: {"s_m": fz["ends"][X]["s_m"], "sigma_m": fz["ends"][X]["sigma_m"],
                                "two_sided": fz["ends"][X]["two_sided"], "point_m": fz["ends"][X]["point_m"],
                                "readings": [{k_: v for k_, v in r.items() if not k_.startswith("_")}
                                             for r in fz["ends"][X]["readings"]]} for X in ("S", "E")}
    plate["groove_ends"]["tau_m"] = fz["tau_m"]
    plate["groove_ends"]["note"] = ("groove change point (whole template, measurements only): where the seam is - the region of "
                                    "the plate-end readings and the location of an end no plate reading reaches yet")
    ef_doc = _end_faces_summary(fe, fe_readings, prior)
    for X in ("S", "E"):
        top = fe["ends"][X]["top"]
        ef_doc["ends"][X]["top"]["point_m"] = np.asarray(L.r(top["s_m"])) if math.isfinite(top["s_m"]) else None
        ef_doc["ends"][X]["root_combined"] = {"s_m": ends[X].s_m, "sigma_m": ends[X].sigma_m}
    plate["end_faces"] = ef_doc
    fz_ends_for_stations = {X: {"s_m": ends[X].s_m} for X in ("S", "E")}
    # §4.9 window statistics (window means of the root residuals, their σ, τ_root) for the checks
    pi_chk = pi_sec
    win = {}
    stations = station_sections(legs, line, fz_ends_for_stations, prior, _prof=prof, s_true=s_true, n_plate=n_hat, pi=pi_sec,
                                offset3=off3, win_out=win)
    # offsets (3D, base)
    cls_off = {line["gauge"]: np.zeros(3)} if line["gauge"] else {}
    for c, (ow, on) in line["offsets"]["class"].items():
        cls_off[c] = ow * f0["w0"] + on * f0["n0"] + o_src["beta_u"].get(c, 0.0) * f0["u0"]
    leg_eta = {}
    for nm, (ow, on) in line["offsets"]["leg"].items():
        leg_eta[nm] = ow * f0["w0"] + on * f0["n0"] + o_src["eta_u"].get(nm, 0.0) * f0["u0"]
    s_mid = 0.5 * (ends["S"].s_m + ends["E"].s_m)
    est = SeamEstimate(groove_found=True, log_bf_run=log_bf_run, line_model=line["model"], line_params=line["params"],
                       line_cov=line["cov"], u=L.tangent(s_mid), n_plate=n_hat, plate=plate, ends=ends,
                       dt_s=float(prior.dt_s) + delta, dt_sigma_s=float(o_src["delta_sigma_s"]), class_offset_m=cls_off,
                       leg_eta_m=leg_eta, sigma_zeta_m=dict(line["sigma_zeta"]), tau_end_m=float(fe["tau_e_m"]),
                       bevel_sign_w=bsw, stations=stations, ratios={}, done=False, binding=[], unresolved=[], frame0=f0)
    sig_eps = float(prior.sigma_eps) if prior.sigma_eps is not None else float("nan")
    est.plate["seam_dims"] = seam_dims(stations, sig_eps, extents=_dim_extents(prof), k=k)
    est.plate["accuracy"] = ("absolute (calibration covariance given)" if getattr(prior, "calib_cov", None) is not None else
                             "relative (repeatability); absolute accuracy needs a calibration covariance")
    est.plate["sigma_eps_reported"] = bool(math.isfinite(sig_eps))
    est.plate.update(_top_point(stations, n_hat))
    est.plate["line_bic"] = line["bic"]
    est.plate["leg_offset_sigma_m"] = line["sigma_eta"]
    est.plate["leg_offset_sigma_wn_m"] = [float(x) for x in line["sigma_eta_wn"]]
    # the root's σ rests on the legs' replication (ν per component, small-sample posterior mixture); ν ≤ 2: not established
    est.plate["root_established"] = [bool(x) for x in line["eta_established"]]
    est.plate["leg_offset_nu"] = [None if x is None else int(x) for x in line["eta_nu"]]
    est.plate["leg_offset_sigma_wn_source"] = list(line.get("sigma_eta_wn_source") or [])
    est.plate["leg_offset_priors"] = {"w": (line.get("eta_priors") or [None, None])[0], "n": (line.get("eta_priors") or [None, None])[1],
                                      "speed_ref_mps": line.get("eta_v_ref_mps"),
                                      "note": "calibrated priors of the leg-offset scatter (calib g1 noise_priors); None = not calibrated"}
    est.plate["leg_offset_sigma_source"] = line["sigma_eta_source"]
    est.plate["direction_offset"] = _direction_report(line, o_src)
    est.plate["time_offset"] = _time_report(line, o_src, prior, "plate-end readings" if o_src is fe else "groove change points")
    est.plate["gauge_class"] = line["gauge"]
    est.plate["end_fusion"] = {"tau_dof": fz["tau_dof"], "length_m": fz["length_m"], "length_sigma_m": fz["length_sigma_m"],
                               "tau_identified": fz["tau_identified"],
                               "identified": {X: fz["ends"][X]["identified"] for X in ("S", "E")},
                               "end_height_m": fz["end_height_m"], "end_height_sigma_m": fz["end_height_sigma_m"],
                               "beta_u_m": fz["beta_u"], "eta_u_m": fz["eta_u"], "b_u_m": fz.get("b_u", 0.0),
                               "b_u_identified": bool(fz.get("b_u_identified", False)),
                               "sigma_given_tau_m": {X: fz["ends"][X]["sigma_given_tau_m"] for X in ("S", "E")}}
    est.plate["legs"] = {nm: {"log_bf_seg": cp["log_bf_seg"], "segment_m": cp["segment_m"]} for nm, cp in per_leg.items()}
    est.plate["ends_plan"] = {X: {"s_m": float(fz_plan["ends"][X]["s_m"]), "sigma_m": float(fz_plan["ends"][X]["sigma_m"]),
                                  "point_m": np.asarray(fz_plan["ends"][X]["point_m"], dtype=float),
                                  "two_sided": bool(fz_plan["ends"][X]["two_sided"])} for X in ("S", "E")}
    est.plate["ends_plan"]["note"] = ("planning only (next pass extents, world box): the measured ends fused with the design length "
                                      "(Cauchy prior); the delivered ends and the done test use the measurements only")
    ctx = {"line": line, "fz": fz, "fz_measured": fz, "win": win, "s_true": s_true, "pi_chk": pi_chk, "prof": prof, "seg_w": seg_w}
    return est, ctx
