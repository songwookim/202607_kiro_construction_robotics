"""Joint-state source of the pose stream (user 2026-10-12): measured encoder angles (H_meas) or the control box's own joint reference
echoed back (H_ref) — a two-hypothesis Bayesian comparison from the run's ROS-level data, and the time-offset prior it implies.

Why. KIRO's arm runs an RB-provided hardware interface (an rbpodo fork). Depending on its configuration /joint_states carries the
encoder angles (rbpodo 'joint_state_source: measured' = CB jnt_ang, our lab default) or the control box's joint reference (upstream
default: read() puts CB jnt_ref into the position state). Under the echo the physical arm trails the reported pose by the servo's
tracking lag τ_echo ≥ 0 (lab CB 2026-10-12: jnt_ang trails jnt_ref by about one 10–12 ms tick — 0.3 mm along the seam at 30 mm/s) and
its vibration is invisible; the seam model's time offset δ (seam_estimate.timing_prior) then carries τ_echo besides the timing
quantisation.

Data (nothing talks to the control box): the joint-state history the pipeline records with every leg (raw.npz robot_*: the
/joint_states samples) and the motion intervals of the trajectories WE commanded (motions_json t_start_ns / t_stop_ns, on the clock of
robot_t_rx_ns). The features use the ORDER and the VALUES of the samples, never their stamps: a hardware interface stamps a sample at
its own read cycle, up to one control tick off the control box's tick — an exact echo looks rough in time, never in value.

  lattice  An encoder counts. A measured joint's values lie on a lattice c·Δ_j + o_j (c integer counts, Δ_j one encoder step
           through the gear, o_j the encoder's zero), each value then rounded ONCE to the interface's representation (CB float32
           degrees → rad: read from the values themselves, representation()). A reference is a computed trajectory: its values are
           continuous up to the same representation grid. Per distinct value the likelihood ratio H_meas : H_ref is Δ/u where the
           value is the representable value nearest a lattice point (|v − cΔ − o| ≤ u/2), 0 elsewhere (the lattice points are 1/Δ
           apart, the reference's values 1/u per representable value — the values of a sweeping joint are smooth on the scale Δ).
           Distinct values at the unit u (a run of values less than u apart is one value: a reference converging on its target packs
           many floats into one unit, an encoder's counts are Δ ≫ u apart). The ratio is exact for the representation's own grid (a
           lattice Δ = m·u gives ratios 0 or m — no evidence either way). Robust: a fraction ε of off-lattice values, ε ~ U(0, 1)
           (ε = 1 is H_ref itself: a stream without a lattice gives H_ref its Occam odds, ~½ ln n per joint, never more); the phase o
           ~ U[0, Δ) and ε integrated exactly (Beta integrals over the phase circle's pieces); Δ_j unknown, scale-invariant prior
           over [one representation unit, the largest increment] — Monte-Carlo average over that prior plus Laplace terms at the
           lattice periods the smallest increments propose (refined by lattice least squares). The increments add the lattice's
           support: an encoder never moves by less than one count, so Δ ≤ min|d| + u (exact). Increments (not values) would not do:
           a steady motion's increments cluster at v·T — not smooth on the scale Δ — and a lattice put on the cluster gains
           spuriously (lab CB reference streams: +10…+70 nats per joint).
  rest     While our command is stationary (outside every commanded motion interval) both sources first converge (a tail of
           unknown length: the reference generator's tail / the servo's settling creep, changes at a rate p); then the echo of a
           constant command holds bit-exactly, while an encoder of a servoed joint keeps changing (±1 count dither) at an unknown rate
           r. Per pair of consecutive samples: changed (any joint) or not. H_ref is H_meas at r = 0; tail length uniform, p, r ~
           U(0, 1) per segment (analytic Beta integrals). A held segment gives H_ref its Occam odds; a change long after the tail is
           what only an encoder does. (A model without the encoder's own settling tail — changes at one rate throughout — read the
           lab's measured settling bursts as an echo's tail: −70…−115 nats on measured runs.)

Each leg is evaluated with its own nuisances (Δ_j, o_j, ε_j, tail lengths, p, r), so legs combine by adding their log Bayes factors (pool()); the
posterior P(H_ref) uses the prior ½ / ½. Every number below is a statistical constant or an [R] computational setting (grid sizes,
iteration caps) — no threshold decides anything: the hypotheses' likelihoods do.

Known limit (lattice): a joint at exactly constant JOINT velocity sampled at an exactly periodic tick is an arithmetic progression,
and every v·T/m is a lattice of that cruise (the support excludes only periods above the leg's smallest increment). The recorded
legs are Cartesian lines (a joint's velocity changes along the line) and dwells; the real echo streams (lab CB jnt_ref, 324 legs at
30–100 Hz reads; the ROS echo run k30_ref_1) show no lattice. A reference model of the samples as a smooth function of time (it
would predict a cruise's values from their neighbours) would close it.

Not used (on purpose): the deviation of the samples from a smooth joint-space path. On the lab's real CB streams the reference's own
path roughness after a local quadratic fit is 0.03–0.3 µrad with a small speed-proportional part (relative 1e-5…3e-5, the curve fit's
remainder and float rounding), the encoder's 1–100 µrad growing ∝ speed (relative 0.007–0.09: the six joints are not latched at one
instant) — the gap is three decades, but a nested 'extra noise ∝ speed' test with 50 000 samples rejects the smooth model for the
reference too; the lattice carries the same encoder physics without a scale to choose.

δ prior (DeltaPrior; seam_estimate.timing_prior / fuse_ends, end_faces.fuse_end_faces): H_meas δ ~ N(0, σ_q²) (the frame and pose
sample quantisation — unchanged); H_ref δ = τ_echo + q with τ_echo ≥ 0 unknown, flat (no lab value: the run's data identify it — two
speeds, both directions; where they cannot, δ and everything that depends on it stay unidentified, σ ∞), q ~ N(0, σ_q²): prior density
∝ Φ(δ/σ_q). The hypotheses enter as a cut mixture: their weights are P(H | joint states) — the sensor data do not reweight them (with
the improper τ prior their Bayes factor would be arbitrary, Lindley's paradox) — and each fusion's δ posterior is the mixture of the
two hypotheses' posteriors. The Gaussian fusions get its exact first two moments: an EP site on δ (Gaussian × probit for H_ref), the
other unknowns follow δ by Gaussian conditioning, so their first two moments are exact as well. One-sided end readings (a site inside
the seam where the groove was seen at rest bounds how far a lag can have moved an end a moving pass read) are probit factors on δ
given the Gaussian readings (DeltaPrior factors, 2026-10-13): they refine each branch that is proper — the measured prior, or a flat
lag the Gaussian readings identify however weakly (one pass whose two end crossings differ in speed, through the design length: the
flat τ alone took such a δ seconds out on the half-line) — by exact 1-D quadrature; they do not by themselves put an unidentified
flat lag into the model, and a factor that is 1 to double precision over a branch's support leaves it exactly as it was.
A later fusion of the same run (the plate-end fusion) takes an earlier one's (the groove fusion's) δ information as its LIKELIHOOD,
never as a posterior: its echo branch is the flat lag ⊗ quantisation Φ(δ/σ_q) times the earlier data's Gaussian N(m, V) (their
one-sided factors folded in by the projection onto that family, flat_lag_fit), times its own data — so τ_echo ≥ 0 holds in every
fusion. (2026-10-13: a Gaussian of the groove fusion's echo posterior handed on as a prior lost the half-line; the KIRO stand-in's
plate-end readings then took δ to −471 ± 217 ms.)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.special import log_ndtr

__all__ = ["representation", "lattice_evidence", "rest_evidence", "rest_segments", "leg_evidence", "raw_evidence", "pool",
           "DeltaPrior", "probit_tilt", "probit_tilt_vec", "flat_lag_fit", "verdict", "H_MEAS", "H_REF"]

H_MEAS, H_REF = "measured", "reference"
LN2PI = math.log(2.0 * math.pi)
# representations an interface can deliver a joint angle in, coarsest first: (numpy dtype, unit) — the stream's own values decide
# which one holds (representation()); rbpodo: float32 degrees from the control box × DEG2RAD (double)
REPRESENTATIONS = ((np.float32, "deg"), (np.float32, "rad"), (np.float64, "rad"))
LATTICE_SEEDS = 12         # [R] smallest distinct increments that propose a lattice period (each / 1, 2, 3)
LATTICE_DIVISORS = 3       # [R]
LATTICE_LS_ITERS = 30      # [R] cap of the lattice least-squares refinement of a proposed period
LATTICE_FIT_MIN = 16       # [R] values nearest the median that start the lattice least squares (window doubles)
LATTICE_MC = 24            # [R] Monte-Carlo periods (stratified in ln Δ) for the prior average away from the proposed lattices


# ======================================================================================================================
# representation of the values
# ======================================================================================================================
def representation(q: np.ndarray) -> dict:
    """The coarsest representation in REPRESENTATIONS that reproduces every value of the stream (rad, float64 as received) to double
    rounding: {'dtype', 'unit', 'scale' (unit per rad), 'values' (in that unit, exact), 'ulp' (each value's representation unit)}."""
    q = np.asarray(q, dtype=np.float64)
    fin = np.isfinite(q)
    for dt, unit in REPRESENTATIONS:
        sc = 180.0 / math.pi if unit == "deg" else 1.0
        v = q * sc
        r = v.astype(dt).astype(np.float64)
        # exact up to the double rounding of the interface's unit conversion (a few ulp of double)
        if np.all(np.abs(r[fin] / sc - q[fin]) <= 4.0 * np.spacing(np.abs(q[fin]))):
            ulp = np.spacing(np.abs(r).astype(dt)).astype(np.float64)
            return {"dtype": np.dtype(dt).name, "unit": unit, "scale": sc, "values": r, "ulp": ulp}
    raise AssertionError("float64 always represents a float64 stream")


# ======================================================================================================================
# helpers
# ======================================================================================================================
def _logsumexp(a, axis=None):
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else float(np.squeeze(out))


def _refine_period(d: np.ndarray, psi: np.ndarray, D0: float, lo: float, hi: float) -> Optional[tuple]:  # psi: rounding sd
    """Lattice least squares from a proposed period (a search step — the evidence is evaluated on all increments afterwards):
    k_i = round(d_i/Δ), Δ = Σ k d/ψ² / Σ k²/ψ², to a fixed point on the increments up to K counts, K doubling from 2 until every
    increment is in (small counts first: their k is unambiguous while Δ is still rough). → (Δ, σ_Δ) or None (left the prior's range)."""
    a = np.abs(d)
    w = 1.0 / psi ** 2
    D = float(D0)
    sig = math.inf
    K = 2.0
    while True:
        sel = a <= (K + 0.5) * D
        for _ in range(LATTICE_LS_ITERS):
            k = np.round(d[sel] / D)
            ok = k != 0
            if not np.any(ok):
                return None
            sw = float(np.sum(w[sel][ok] * k[ok] ** 2))
            D_new = float(np.sum(w[sel][ok] * k[ok] * d[sel][ok]) / sw)
            sig = 1.0 / math.sqrt(sw)
            if not (lo <= D_new <= hi):
                return None
            done = abs(D_new - D) <= 1e-13 * D          # [R] fixed point
            D = D_new
            if done:
                break
        if bool(np.all(sel)) or not np.any(~sel):
            break
        K *= 2.0
    return D, sig


def _refine_values(v: np.ndarray, u: float, D0: float, lo: float, hi: float) -> Optional[tuple]:
    """Lattice least squares of the values themselves (a search step): v ≈ c·Δ + o with integer counts c = round((v − o)/Δ), (Δ, o) by
    least squares to a fixed point, on the LATTICE_FIT_MIN values nearest the median value first, the window doubling until all are in
    (the counts stay unambiguous while Δ is still rough). → (Δ, σ_Δ, o) or None (left the prior's range)."""
    vr = v - float(np.median(v))
    order = np.argsort(np.abs(vr), kind="stable")
    D, o = float(D0), 0.0
    k = min(LATTICE_FIT_MIN, vr.size)
    while True:
        x = vr[order[:k]]
        for _ in range(LATTICE_LS_ITERS):
            c = np.round((x - o) / D)
            # closed-form least squares of x = Δ·c + o
            cm, xm = float(c.mean()), float(x.mean())
            scc = float(np.sum((c - cm) ** 2))
            if not scc > 0 or np.count_nonzero(c != c[0]) < 2:  # fewer than three distinct counts fit any lattice: no information
                return None
            D_new = float(np.sum((c - cm) * (x - xm)) / scc)
            o_new = xm - D_new * cm
            if not (lo < D_new <= hi):
                return None
            done = abs(D_new - D) <= 1e-13 * D and abs(o_new - o) <= 1e-13 * D      # [R] fixed point
            D, o = D_new, o_new
            if done:
                break
        if k >= vr.size:
            break
        k = min(2 * k, vr.size)
    c = np.round((vr - o) / D)
    sig = (u / math.sqrt(12.0)) / math.sqrt(max(float(np.sum((c - c.mean()) ** 2)), 1e-300))
    return D, sig, o


def _ln_g(m: np.ndarray, n: int, L: float) -> np.ndarray:
    """ln ∫₀¹ ((1 − ε)L + ε)^m ε^(n − m) dε for L > 1 (m values on the lattice with ratio L each, n − m off it with ratio ε; ε ~ U(0, 1)):
    with s = (L − 1)ε/L it is L^(n+1) (L − 1)^−a B(a, b) I_x(a, b), a = n − m + 1, b = m + 1, x = (L − 1)/L (regularised incomplete
    beta) — exact."""
    from scipy.special import betainc, betaln
    m = np.asarray(m, dtype=float)
    a, b = n - m + 1.0, m + 1.0
    with np.errstate(divide="ignore"):
        return (n + 1.0) * math.log(L) - a * math.log(L - 1.0) + betaln(a, b) + np.log(betainc(a, b, (L - 1.0) / L))


def _ln_lattice(v: np.ndarray, u: float, D: float) -> float:
    """ln of the lattice likelihood ratio of the distinct values v at period D: the phase o ~ U[0, D) and ε integrated. A value is on
    the lattice when it is the representable value nearest some c·D + o (|v − cD − o| ≤ u/2): ratio D/u there (the lattice points
    are 1/D apart, the reference's values have density 1/u per representable value), 0 off it; the windows [ρ − u/2, ρ + u/2] of
    the residues ρ = v mod D cut the phase circle into pieces of constant count m (exact sweep)."""
    n = int(v.size)
    rho = np.mod(v - float(np.median(v)), D)
    s = np.mod(rho - 0.5 * u, D)
    e = np.mod(rho + 0.5 * u, D)
    m0 = int(np.count_nonzero(s > e))                     # windows that wrap contain the phase 0
    pos = np.r_[s, e]
    dl = np.r_[np.ones(n), -np.ones(n)]
    order = np.argsort(pos, kind="stable")
    pos, dl = pos[order], dl[order]
    cnt = np.r_[m0, m0 + np.cumsum(dl)]
    seg = np.diff(np.r_[0.0, pos, D])
    ok = seg > 0
    if not np.any(ok):
        return -math.inf
    mu, inv = np.unique(np.clip(cnt[ok], 0, n), return_inverse=True)       # few distinct counts: one Beta integral each
    return float(_logsumexp(np.log(seg[ok]) + _ln_g(mu, n, D / u)[inv]) - math.log(D))


def lattice_evidence(v: np.ndarray, ulp: np.ndarray, seed: int = 0) -> dict:
    """One joint's values in sample order (representation units) and their representation units → ln BF (H_meas : H_ref) of 'the
    distinct values lie on an unknown lattice c·Δ + o, each rounded once (plus a fraction ε off it)' vs 'continuous values on the
    representation grid' (module doc). Distinct values: a held or revisited count is one lattice point, a held reference one value.
    The unit u is the coarsest of the leg's values (one binade almost always; coarser is conservative: a finer value's window is
    wider than its rounding, its ratio smaller). → {'ln_bf', 'n', 'period', 'period_sigma', 'ln_bf_at_period', 'unit_value'}."""
    v_all = np.asarray(v, dtype=float)
    ulp = np.asarray(ulp, dtype=float)
    fin = np.isfinite(v_all)
    d = np.diff(v_all[fin])
    d = d[d != 0.0]
    out = {"ln_bf": 0.0, "n": 0, "period": None, "period_sigma": None, "ln_bf_at_period": None, "unit_value": None}
    if not d.size:
        return out
    u = float(np.max(ulp[fin]))
    # values distinct at the unit u: a run of values each less than u above the previous is one value (a reference converging on its
    # target packs many distinct floats within a unit — near zero, where the representation is finest, thousands; an encoder's counts
    # are Δ ≫ u apart and never merge)
    vs = np.unique(v_all[fin])
    vals = vs[np.r_[True, np.diff(vs) >= u]]
    n = int(vals.size)
    out["n"], out["unit_value"] = n, u
    if n < 2:
        return out
    lo, hi = u, float(np.max(np.abs(d)))            # prior: a lattice no finer than one unit, no coarser than the largest increment
    if not hi > lo:
        return out
    ln_range = math.log(hi / lo)
    # the increments' support: an encoder never moves by less than one count (|d| = |k|Δ ± one unit, k ≠ 0), so the lattices with
    # Δ > min|d| + u have likelihood 0 — exact, no robustness (the lab's measured streams: every increment within its rounding). It
    # also keeps out the one lattice a reference can show: a joint at constant joint velocity sampled at an exactly periodic tick
    # is an arithmetic progression of period v·T — a period no larger than the smallest increment only if the joint never slows
    hi_eff = min(hi, float(np.min(np.abs(d))) + u)
    out["period_support_max"] = hi_eff
    if not hi_eff > lo:                             # every lattice is excluded by an increment below one unit: not an encoder
        out["ln_bf"] = -math.inf
        return out
    ln_eff = math.log(hi_eff / lo)
    ln_frac = math.log(ln_eff / ln_range)           # the supported part's prior mass (scale-invariant: in ln Δ)
    # Monte-Carlo average over the scale-invariant prior on the supported part (stratified in ln Δ): the background away from a
    # lattice, times the prior mass of that part
    rng = np.random.default_rng(seed)
    w = (np.arange(LATTICE_MC) + rng.random(LATTICE_MC)) / LATTICE_MC
    mc = np.array([_ln_lattice(vals, u, lo * math.exp(wi * ln_eff)) for wi in w])
    ln_bg = _logsumexp(mc) - math.log(LATTICE_MC) + ln_frac
    hi = hi_eff
    # lattice periods proposed by the smallest increments (one count, or two, three), refined by lattice least squares on the
    # increments and then on the values; each a Laplace term ∫ L π ≈ L(Δ̂)·√(2π)·σ_Δ/(Δ̂·ln(hi/lo))
    psi = np.full(d.size, u * math.sqrt(2.0 / 12.0))
    props = []
    for s_ in np.unique(np.abs(d))[:LATTICE_SEEDS]:
        for m in range(1, LATTICE_DIVISORS + 1):
            D0 = float(s_) / m
            if not (lo < D0 <= hi):
                continue
            r = _refine_period(d, psi, D0, lo, hi)
            if r is not None and not any(abs(r[0] - p_[0]) <= 3.0 * (r[1] + p_[1]) for p_ in props):   # [R] the same period
                props.append(r)
    cands = []
    for D1, _s in props:
        r = _refine_values(vals, u, D1, lo, hi)
        if r is not None and not any(abs(r[0] - c[0]) <= 1e-9 * r[0] for c in cands):           # [R] the same period
            cands.append(r)
    terms = [ln_bg]
    best = None
    for D, sD, _o in cands:
        lm = _ln_lattice(vals, u, D)
        terms.append(lm + 0.5 * LN2PI + math.log(sD / D) - math.log(ln_range))
        if best is None or lm > best[2]:
            best = (D, sD, lm)
    out["ln_bf"] = float(_logsumexp(np.array(terms)))
    if best is not None:
        out.update(period=best[0], period_sigma=best[1], ln_bf_at_period=float(best[2]))
    return out


# ======================================================================================================================
# rest (hold vs creep / dither)
# ======================================================================================================================
def rest_segments(t_rx_ns: np.ndarray, motions: Sequence) -> list:
    """Index runs of the samples during which our command is stationary: outside every commanded motion interval
    [t_start_ns, t_stop_ns] (same clock as t_rx_ns). A motion without both times makes the leg's command state unknown: no segment."""
    t = np.asarray(t_rx_ns, dtype=np.int64)
    rest = np.ones(t.size, dtype=bool)
    for m in motions or ():
        a, b = (m.get("t_start_ns"), m.get("t_stop_ns")) if isinstance(m, dict) else (m[0], m[1])
        if a is None or b is None:
            return []
        rest &= ~((t >= int(a)) & (t <= int(b)))
    segs, i = [], 0
    while i < t.size:
        if not rest[i]:
            i += 1
            continue
        j = i
        while j < t.size and rest[j]:
            j += 1
        if j - i >= 2:
            segs.append((i, j))
        i = j
    return segs


def rest_evidence(changed: Sequence[np.ndarray]) -> dict:
    """Rest segments' change indicators (one bool array per segment: pair i changed in any joint) → ln BF (H_meas : H_ref).

    Both sources converge after a stop: a segment starts with a tail of unknown length h (uniform over 0 … N_s pairs) in which the
    stream changes at a rate p (the reference generator's tail / the servo's settling creep). After the tail an encoder keeps changing
    at a rate r (±1 count dither of a held joint); an echo of a stationary command holds bit-exactly (r = 0). Rates per segment (the
    tail depends on the motion that ended), p, r ~ U(0, 1) — analytic Beta integrals. H_ref is H_meas at r = 0: a segment that holds
    gives H_ref its Occam odds, a change long after the tail is what only an encoder does."""
    from scipy.special import betaln
    segs = [np.asarray(c, dtype=bool) for c in changed if len(c)]
    N = sum(int(c.size) for c in segs)
    C = sum(int(np.count_nonzero(c)) for c in segs)
    out = {"ln_bf": 0.0, "pairs": N, "changes": C, "segments": len(segs)}
    if N == 0:
        return out
    ln_bf = 0.0
    for c in segs:
        Ns, Cs = int(c.size), int(np.count_nonzero(c))
        h = np.arange(Ns + 1)
        Ch = np.r_[0, np.cumsum(c)]                       # changes in the first h pairs
        tail = betaln(Ch + 1, h - Ch + 1)                 # ∫ p^C(h) (1 − p)^(h − C(h)) dp
        after = betaln(Cs - Ch + 1, (Ns - h) - (Cs - Ch) + 1)
        ln_meas = _logsumexp(tail + after) - math.log(Ns + 1)
        hold = Ch == Cs                                   # H_ref: no change after the tail
        ln_ref = _logsumexp(tail[hold]) - math.log(Ns + 1)
        ln_bf += float(ln_meas - ln_ref)
    out["ln_bf"] = float(ln_bf)
    return out


# ======================================================================================================================
# one leg, several legs
# ======================================================================================================================
def leg_evidence(q: np.ndarray, t_rx_ns: Optional[np.ndarray] = None, motions: Optional[Sequence] = None, *,
                 names: Optional[Sequence[str]] = None) -> dict:
    """One leg's joint-state samples (N, J) [rad] in recording order, their receive times (robot_t_rx_ns) and the motions we
    commanded during the leg → {'ln_bf': {'lattice', 'rest'} (H_meas : H_ref), 'lattice': per joint, 'rest': …, 'n': N,
    'representation': …}."""
    q = np.asarray(q, dtype=float).reshape(len(q), -1) if len(q) else np.zeros((0, 6))
    out = {"n": int(q.shape[0]), "ln_bf": {"lattice": 0.0, "rest": 0.0}, "lattice": [], "rest": None, "representation": None}
    if q.shape[0] < 2:
        return out
    rep = representation(q)
    out["representation"] = f"{rep['dtype']} {rep['unit']}"
    lat = []
    for j in range(q.shape[1]):
        e = lattice_evidence(rep["values"][:, j], rep["ulp"][:, j], seed=j)
        e["joint"] = (list(names)[j] if names is not None and j < len(names) else j)
        e["unit"] = rep["unit"]
        lat.append(e)
    out["lattice"] = lat
    out["ln_bf"]["lattice"] = float(sum(e["ln_bf"] for e in lat))
    if t_rx_ns is not None:
        segs = rest_segments(t_rx_ns, motions or [])
        ch = np.any(np.diff(rep["values"], axis=0) != 0.0, axis=1)
        r = rest_evidence([ch[a:b - 1] for a, b in segs])
        out["rest"] = r
        out["ln_bf"]["rest"] = float(r["ln_bf"])
    return out


def raw_evidence(raw: dict) -> dict:
    """leg_evidence from a leg's raw arrays (raw.npz members, or the worker's raw_arrays + raw_json): robot_joint_positions_rad,
    robot_t_rx_ns, motions_json."""
    q = np.asarray(raw.get("robot_joint_positions_rad", np.zeros((0, 6))), dtype=float)
    t = raw.get("robot_t_rx_ns")
    mj = raw.get("motions_json")
    try:
        motions = json.loads(str(mj)) if mj is not None and not isinstance(mj, (list, tuple)) else list(mj or [])
    except (TypeError, ValueError):
        motions = None
    names = raw.get("robot_joint_names")
    return leg_evidence(q, None if t is None or motions is None else np.asarray(t, dtype=np.int64), motions,
                        names=None if names is None else [str(x) for x in np.asarray(names).ravel()])


def pool(evidences: Sequence[Optional[dict]], names: Optional[Sequence[str]] = None) -> dict:
    """Legs' evidence → the run's: ln BF (H_meas : H_ref) per feature (sum over legs), the log odds of H_ref, P(H_ref) (prior ½ / ½),
    the per-leg ln BFs and the data counts. Legs without evidence (None) contribute nothing."""
    from scipy.special import expit
    pairs = [(e, (list(names)[i] if names is not None and i < len(names) else str(i))) for i, e in enumerate(evidences) if e]
    ev = [e for e, _ in pairs]
    lb = {"lattice": float(sum(e["ln_bf"].get("lattice", 0.0) for e in ev)),
          "rest": float(sum(e["ln_bf"].get("rest", 0.0) for e in ev))}
    ln_odds_ref = -(lb["lattice"] + lb["rest"])
    p_ref = float(expit(ln_odds_ref))                 # prior ½ / ½: the posterior is the logistic of the summed log Bayes factors
    rest = [e.get("rest") or {} for e in ev]
    return {"ln_bf_meas_vs_ref": lb, "ln_odds_ref": float(ln_odds_ref), "p_ref": p_ref, "prior_p_ref": 0.5,
            "legs": len(ev), "samples": int(sum(e.get("n", 0) for e in ev)),
            "lattice_values": int(sum(sum(j.get("n", 0) for j in e.get("lattice") or []) for e in ev)),
            "rest_pairs": int(sum(r.get("pairs", 0) for r in rest)), "rest_changes": int(sum(r.get("changes", 0) for r in rest)),
            "representation": sorted({e["representation"] for e in ev if e.get("representation")}),
            "per_leg": [{"leg": nm, "lattice": round(float(e["ln_bf"].get("lattice", 0.0)), 3),
                         "rest": round(float(e["ln_bf"].get("rest", 0.0)), 3), "samples": int(e.get("n", 0))} for e, nm in pairs]}


def verdict(ln_odds_ref: float, k: float) -> str:
    """The hypothesis the evidence resolves, in the estimator's one decision form (2·ln odds against k², seam_estimate
    segment_verdict): 'reference' / 'measured', or 'undecided' between."""
    if 2.0 * ln_odds_ref > k * k:
        return H_REF
    if -2.0 * ln_odds_ref > k * k:
        return H_MEAS
    return "undecided"


# ======================================================================================================================
# the δ prior the hypotheses imply
# ======================================================================================================================
def probit_tilt(m: float, V: float, s: float) -> tuple:
    """Moments and log normaliser of N(δ; m, V)·Φ(δ/s) (V, s > 0): (mean, var, ln ∫ N·Φ dδ). Gaussian × probit (EP's classic)."""
    S = math.sqrt(V + s * s)
    z = m / S
    lZ = float(log_ndtr(z))
    r = math.exp(-0.5 * z * z - 0.5 * LN2PI - lZ)                 # φ(z)/Φ(z), stable through log Φ
    mu = m + V * r / S
    var = V - (V * V / (V + s * s)) * r * (z + r)
    return mu, max(var, 0.0), lZ


def probit_tilt_vec(m, V, s: float) -> tuple:
    """probit_tilt elementwise over arrays m, V (V > 0 finite, s > 0): (mean, var, ln normaliser) arrays."""
    m = np.asarray(m, dtype=float)
    V = np.asarray(V, dtype=float)
    S = np.sqrt(V + s * s)
    z = m / S
    lZ = log_ndtr(z)
    r = np.exp(-0.5 * z * z - 0.5 * LN2PI - lZ)
    mu = m + V * r / S
    var = np.maximum(V - (V * V / (V + s * s)) * r * (z + r), 0.0)
    return mu, var, lZ


FIT_ITERS = 200            # [R] cap of the moment-matching iterations (flat_lag_fit)


def flat_lag_fit(mu: float, var: float, s: float, x0: Optional[tuple] = None) -> tuple:
    """The member Φ(δ/s)·N(δ; m, V) of the flat-lag posterior family (the echo branch's posterior under Gaussian data) with first two
    moments (mu, var) → (m, V). The family is an exponential family with base measure Φ(δ/s) and statistics (δ, δ²); matching the
    moments is the KL(p‖q) projection of a distribution p onto it (the EP projection keeping the flat lag's half-line exact), the
    minimum of the convex A(η) − η·(mu, var + mu²) over the natural parameters η = (m/V, −1/(2V)) — unique. A log-concave p on the
    half-line has a coefficient of variation ≤ 1, inside the family's moment range. Solved in units of p's own sd (BFGS, analytic
    gradient = the family's moments − the target's); x0 = (m, V) start (default (mu, var))."""
    from scipy.optimize import minimize
    sd = math.sqrt(var)
    ss = s / sd
    t1 = mu / sd
    t2 = 1.0 + t1 * t1
    m0, V0 = x0 if x0 is not None else (mu, var)
    m0, V0 = float(m0) / sd, max(float(V0), 1e-300) / var

    def f(x):
        e1, rho = float(x[0]), float(x[1])
        V = math.exp(-rho)                      # η2 = −1/(2V) = −e^ρ/2
        m = e1 * V
        e2 = -0.5 / V
        a = 0.5 * (LN2PI - rho) + 0.5 * m * m / V + float(log_ndtr(m / math.sqrt(V + ss * ss)))
        mq, vq, _ = probit_tilt(m, V, ss)
        return a - e1 * t1 - e2 * t2, np.array([mq - t1, (vq + mq * mq - t2) * e2])

    res = minimize(f, np.array([m0 / V0, -math.log(V0)]), jac=True, method="BFGS",
                   options={"gtol": 1e-12, "maxiter": FIT_ITERS})
    V = math.exp(-float(res.x[1]))
    return sd * float(res.x[0]) * V, var * V


# one-sided readings' factors on δ (DeltaPrior factors): exact 1-D quadrature of the branch posterior where a factor is not 1
# half-width of a branch's support in its standard deviations: beyond it a Gaussian's density is below double precision's unit
# roundoff 2^-53 of its peak (and its tail mass below that too) — a factor equal to 1 to double precision over that support leaves
# the branch's moments as they are (machine precision, not a setting)
DELTA_GRID_SD = math.sqrt(2.0 * 53.0 * math.log(2.0))
DELTA_GRID_N = 129         # [R] nodes of the grid that finds where a branch holds its mass (and of the factors' activity test)
DELTA_GRID_ROUNDS = 40     # [R] cap of its refinements
DELTA_GRID_PER_SCALE = 4   # [R] quadrature nodes per smallest scale of the integrand (the base's width, a factor's s/|β|)
DELTA_GRID_MAX = 1 << 14   # [R] cap of the quadrature nodes


def _factor_log(factors: Sequence, d: np.ndarray) -> np.ndarray:
    """Σ_i w_i·ln Φ((c_i − β_i·δ)/s_i) at the δ values d (factors (β, c, s, w))."""
    out = np.zeros(np.shape(d))
    for b, c, s, w in factors:
        out = out + float(w) * log_ndtr((float(c) - float(b) * d) / float(s))
    return out


def _factors_active(fs: np.ndarray, lo: float, hi: float, logb) -> np.ndarray:
    """Which factors Φ((c − β·δ)/s)^w (rows (β, c, s, w)) change a branch with support [lo, hi] and log density logb (relative to any
    constant) in double precision. A constant multiplier normalises out, so what counts is a factor's variation: it is monotone in δ,
    so where its values at the two ends of the support agree to double precision it is constant there and changes nothing.
    Otherwise its relative variation from its value at the branch's peak, weighted by the branch's density relative to that peak, on
    the support's grid: below the unit roundoff 2^-53 at every node, every normalised density value there, and so every sum over
    them, is unchanged in double precision."""
    fs = np.asarray(fs, dtype=float).reshape(-1, 4)
    b, c, s, w = fs.T
    ok = (w > 0.0) & (b != 0.0) & (s > 0.0)
    act = np.zeros(fs.shape[0], dtype=bool)
    if not np.any(ok):
        return act
    i = np.flatnonzero(ok)
    l_lo = w[i] * log_ndtr((c[i] - b[i] * lo) / s[i])
    l_hi = w[i] * log_ndtr((c[i] - b[i] * hi) / s[i])
    i = i[np.exp(-np.abs(l_lo - l_hi)) < 1.0]
    if not i.size:
        return act
    d = np.linspace(lo, hi, DELTA_GRID_N)
    lb = np.asarray(logb(d), dtype=float)
    k0 = int(np.argmax(lb))
    L = w[i, None] * log_ndtr((c[i, None] - b[i, None] * d[None, :]) / s[i, None])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        H = np.log(np.abs(np.expm1(L - L[:, k0:k0 + 1]))) + (lb - lb[k0])[None, :]
    H = np.where(np.isnan(H), np.inf, H)                 # a factor 0 at the branch's peak: it changes the branch
    act[i] = np.max(H, axis=1) >= -53.0 * math.log(2.0)
    return act


def _quad_moments(logp, lo: float, hi: float, scale: float, outer: Optional[tuple] = None) -> tuple:
    """Mean, variance and ln ∫ of exp(logp(δ)) for a log-concave logp whose mass lies in [lo, hi] — or, when it reaches an edge of
    that interval, in outer: a uniform grid, refined to the nodes within DELTA_GRID_SD²/2 of its maximum (one contiguous run for a
    log-concave density) until they span a quarter of it; then the trapezoid rule on that interval with DELTA_GRID_PER_SCALE nodes
    per smallest scale of the integrand (scale)."""
    for rnd in range(DELTA_GRID_ROUNDS):
        d = np.linspace(lo, hi, DELTA_GRID_N)
        lp = np.asarray(logp(d), dtype=float)
        top = float(np.max(lp))
        if not math.isfinite(top):                      # no mass the grid can represent: no proper posterior from it
            return float("nan"), math.inf, -math.inf
        on = np.flatnonzero(lp >= top - 0.5 * DELTA_GRID_SD ** 2)
        if rnd == 0 and outer is not None and (on[0] == 0 or on[-1] == DELTA_GRID_N - 1):
            return _quad_moments(logp, outer[0], outer[1], scale)
        i0, i1 = max(int(on[0]) - 1, 0), min(int(on[-1]) + 1, DELTA_GRID_N - 1)
        lo, hi = float(d[i0]), float(d[i1])
        if i1 - i0 >= DELTA_GRID_N // 4:
            break
    n = int(min(max(DELTA_GRID_N, math.ceil(DELTA_GRID_PER_SCALE * (hi - lo) / scale) + 1), DELTA_GRID_MAX)) if scale > 0 \
        else DELTA_GRID_MAX
    d = np.linspace(lo, hi, n)
    lp = np.asarray(logp(d), dtype=float)
    top = float(np.max(lp))
    w = np.exp(lp - top)
    h = float(d[1] - d[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    Z = float(np.sum(w))
    mu = float(np.sum(w * d) / Z)
    var = float(np.sum(w * (d - mu) ** 2) / Z)
    return mu, max(var, 0.0), math.log(Z * h) + top


@dataclass(frozen=True)
class DeltaPrior:
    """The time offset δ's prior under the joint-state hypotheses (module doc): H_meas N(0, σ_q²); H_ref ∝ Φ(δ/σ_q) (τ_echo ≥ 0 flat
    ⊗ the quantisation) — times, given ref_lik_mean / ref_lik_var, the Gaussian N(ref_lik_mean, ref_lik_var): an earlier fusion's
    LIKELIHOOD of δ (the plate-end fusion takes the groove fusion's — seam_estimate._end_faces: the run's data identify the lag, the
    plate ends alone often cannot, and a flat lag there has a heavy tail through the crossing scatter; its one-sided factors folded
    in by flat_lag_fit) — the echo branch stays on the lag's half-line; cut mixture with weight p_ref on H_ref. p_ref 0 = today's
    prior exactly."""

    sigma_q: float
    p_ref: float = 0.0
    ref_lik_mean: Optional[float] = None
    ref_lik_var: Optional[float] = None

    def _ref_lik(self) -> bool:
        return (self.ref_lik_var is not None and math.isfinite(float(self.ref_lik_var)) and float(self.ref_lik_var) > 0.0
                and self.sigma_q > 0)

    def _ref_gauss(self, m_L: float, V_L: float) -> tuple:
        """The echo branch's Gaussian part: the data's N(m_L, V_L) (V_L ∞: none) times the earlier fusion's likelihood N(m0, v0)
        when given → (m, V, ln of the data's Gaussian evidence under the earlier likelihood) — (nan, ∞, 0) when neither is proper."""
        if self._ref_lik():
            m0, v0 = float(self.ref_lik_mean), float(self.ref_lik_var)
            if not math.isfinite(V_L):
                return m0, v0, 0.0
            vc = 1.0 / (1.0 / V_L + 1.0 / v0)
            return vc * (m_L / V_L + m0 / v0), vc, -0.5 * (LN2PI + math.log(V_L + v0)) - 0.5 * (m_L - m0) ** 2 / (V_L + v0)
        if not math.isfinite(V_L):
            return float("nan"), math.inf, 0.0
        return m_L, V_L, 0.0

    def branches(self, m_L: float, V_L: float, factors: Sequence = ()) -> dict:
        """Each hypothesis' posterior of δ given the data's own Gaussian likelihood N(m_L, V_L) of δ (V_L = ∞: the data carry none)
        and the non-Gaussian factors the data put on δ — the one-sided end readings (seam_estimate.fuse_ends: a reading of the groove
        at a leg's edge bounds an end; given δ and the Gaussian readings it is the probit Φ((c − β·δ)/s)^w, factors (β, c, s, w)):
        {'meas': (μ, v, ln Z), 'ref': (μ, v, ln Z)} — ln Z the log normaliser (the flat H_ref's up to its constant height); an
        improper posterior (flat H_ref with V_L = ∞) has μ = nan, v = ∞. The factors refine a branch that is proper — the prior's
        Gaussian or the data's (V_L finite) put δ in the model; they do not put a flat lag the data leave unidentified into it (an
        interval bound alone leaves its location within the interval arbitrary: δ stays free, the ends in the frame of the speed
        measured — the identification rule of seam_estimate, as for the direction offset b_u). A factor that is 1 to double precision
        over the branch's support leaves it exactly as without it; otherwise the branch is the exact 1-D quadrature of prior ×
        N(m_L, V_L) × factors (log-concave)."""
        return self._branches(m_L, V_L, factors)[0]

    def _branches(self, m_L: float, V_L: float, factors: Sequence = (), which: tuple = ("meas", "ref")) -> tuple:
        """branches() for the hypotheses in `which` (the others without the factors) → (branches, [hypotheses the factors changed])."""
        s2 = self.sigma_q ** 2
        R = DELTA_GRID_SD
        if not math.isfinite(V_L):
            meas = (0.0, s2, 0.0)
        else:
            v = 1.0 / (1.0 / V_L + 1.0 / s2) if s2 > 0 else 0.0
            meas = (v * m_L / V_L, v, -0.5 * (LN2PI + math.log(V_L + s2)) - 0.5 * m_L * m_L / (V_L + s2))
        # the echo branch: Φ(δ/σ_q) × its Gaussian part (the data's, times an earlier fusion's likelihood when given)
        m_c, V_c, lnN = self._ref_gauss(m_L, V_L)
        if not math.isfinite(V_c):
            ref = (float("nan"), math.inf, math.inf)
        elif self.sigma_q > 0:
            mu_r, v_r, lZ = probit_tilt(m_c, V_c, self.sigma_q)
            ref = (mu_r, v_r, lZ + lnN)
        else:
            ref = (max(m_c, 0.0), V_c, lnN)
        out = {"meas": meas, "ref": ref}
        tilted: list = []
        if not factors:
            return out, tilted
        fs = np.asarray([[float(x) for x in f] for f in factors], dtype=float).reshape(-1, 4)
        # supports (where each branch without the factors holds its mass) and the base log densities
        sup, logb, scl = {}, {}, {}
        mm, vm = meas[0], meas[1]
        if vm > 0:
            sup["meas"] = (mm - R * math.sqrt(vm), mm + R * math.sqrt(vm))
            logb["meas"] = lambda d, mm=mm, vm=vm: -0.5 * (d - mm) ** 2 / vm
            scl["meas"] = math.sqrt(vm)
        if math.isfinite(V_c) and self.sigma_q <= 0:
            mr, vr = ref[0], ref[1]
            if vr > 0:
                sup["ref"] = (mr - R * math.sqrt(vr), mr + R * math.sqrt(vr))
                logb["ref"] = lambda d, mr=mr, vr=vr: -0.5 * (d - mr) ** 2 / vr
                scl["ref"] = math.sqrt(vr)
        elif self.sigma_q > 0 and math.isfinite(V_c):    # flat τ_echo ≥ 0 ⊗ N(0, σ_q²): Φ(δ/σ_q), times the Gaussian part
            sq = self.sigma_q
            hi_ = max(ref[0] + R * math.sqrt(ref[1]), m_c + R * math.sqrt(V_c))
            sup["ref"] = (min(ref[0] - R * math.sqrt(ref[1]), -R * sq), hi_)
            logb["ref"] = lambda d, m_c=m_c, V_c=V_c, sq=sq: -0.5 * (d - m_c) ** 2 / V_c + log_ndtr(d / sq)
            scl["ref"] = min(sq, math.sqrt(V_c))
        for H in which:
            if H not in sup:
                continue
            lo, hi = sup[H]
            act = [tuple(f) for f in fs[_factors_active(fs, lo, hi, logb[H])]]
            if not act:
                continue
            # the mass lies within the hull of the base's support and the active factors' transitions (log-concave product)
            tr = [(c / b, s / abs(b)) for b, c, s, _ in act]
            lo_q = min([lo] + [t - R * wd for t, wd in tr])
            hi_q = max([hi] + [t + R * wd for t, wd in tr])
            lb = logb[H]
            out[H] = _quad_moments(lambda d, lb=lb, act=act: lb(d) + _factor_log(act, d), lo, hi,
                                   min([scl[H]] + [wd for _, wd in tr]), outer=(lo_q, hi_q))
            tilted.append(H)
        return out, tilted

    def moments(self, m_L: float, V_L: float, factors: Sequence = ()) -> tuple:
        """The cut mixture's δ posterior (μ, v): (1 − p)·post_meas + p·post_ref. (nan, ∞) when the H_ref part is improper."""
        return self._moments(m_L, V_L, factors)[0]

    def _moments(self, m_L: float, V_L: float, factors: Sequence = ()) -> tuple:
        """moments() → ((μ, v), [hypotheses the factors changed]) — only the branches with weight are evaluated."""
        p = float(self.p_ref)
        b, tilted = self._branches(m_L, V_L, factors, ("meas",) if p <= 0.0 else (("ref",) if p >= 1.0 else ("meas", "ref")))
        return self._mix(b, p), tilted

    def _mix(self, b: dict, p: float) -> tuple:
        if p <= 0.0:
            return b["meas"][0], b["meas"][1]
        if p >= 1.0:
            return b["ref"][0], b["ref"][1]
        (mm, vm, _), (mr, vr, _) = b["meas"], b["ref"]
        if not math.isfinite(vr):
            return float("nan"), math.inf
        mu = (1.0 - p) * mm + p * mr
        return mu, (1.0 - p) * (vm + (mm - mu) ** 2) + p * (vr + (mr - mu) ** 2)

    def site(self, m_L: float, V_L: float, factors: Sequence = ()) -> tuple:
        """EP site on δ: (λ, λ·m) such that N(m_L, V_L)·exp(−½λδ² + λm·δ) has the mixture posterior's two moments (λ may be negative
        when the hypotheses' posteriors disagree by more than the data's own width — their sum stays positive). V_L = ∞ and a
        proper mixture: the posterior is the prior's moments; an improper one: (0, 0) — δ free. factors: the one-sided readings'
        probits on δ (branches) — the site then carries their information on δ with the prior's. The measured source alone with no
        factor that changes its branch: the prior's own site (1/σ_q², 0) exactly."""
        (mu, v), tilted = self._moments(m_L, V_L, factors)
        if float(self.p_ref) <= 0.0 and self.sigma_q > 0 and not tilted:
            return 1.0 / self.sigma_q ** 2, 0.0
        if not math.isfinite(v):
            return 0.0, 0.0
        if not math.isfinite(V_L):
            return 1.0 / v, mu / v
        return 1.0 / v - 1.0 / V_L, mu / v - m_L / V_L

    def prior_precision(self) -> float:
        """The precision of the δ prior as a whole (the moment-matched mixture prior): 1/σ_q² under H_meas alone; with weight on a
        flat H_ref 0 — δ is then a fixed effect of the fusions (it counts in their degrees of freedom); with an earlier fusion's
        likelihood on H_ref (proper: Φ(δ/σ_q)·N(m0, v0)) the mixture's 1/variance."""
        if self.sigma_q <= 0:
            return 0.0
        p = float(self.p_ref)
        if p <= 0.0:
            return 1.0 / self.sigma_q ** 2
        if not self._ref_lik():
            return 0.0
        m0, v0, _ = probit_tilt(float(self.ref_lik_mean), float(self.ref_lik_var), self.sigma_q)
        mu = p * m0
        var = (1.0 - p) * (self.sigma_q ** 2 + mu * mu) + p * (v0 + (m0 - mu) ** 2)
        return 1.0 / var

    def as_dict(self) -> dict:
        d = {"sigma_q_s": float(self.sigma_q), "p_ref": float(self.p_ref),
             "form": ("(1 - p_ref) N(0, sigma_q^2) + p_ref [tau_echo >= 0 flat] * N(0, sigma_q^2) (cut mixture; the fusions use the "
                      "posterior's first two moments)")}
        if self._ref_lik():
            d.update(ref_likelihood_mean_s=float(self.ref_lik_mean), ref_likelihood_sigma_s=math.sqrt(float(self.ref_lik_var)),
                     form=("(1 - p_ref) N(0, sigma_q^2) + p_ref [tau_echo >= 0 flat] * N(0, sigma_q^2) * N(ref_likelihood_mean, "
                           "ref_likelihood_sigma^2) (the groove fusion's likelihood of delta in the same run; cut mixture)"))
        return d
