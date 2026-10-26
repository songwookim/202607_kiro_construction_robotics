"""Settling of the robot's along-track motion after the start ramp of a straight leg (1G pass overscan, user 2026-10-11 night).

Why. A pass starts at rest beyond an end, accelerates over v²/2a and should cross the end at steady cruise: the plate-end readings
of the ENTRY crossing carry most gross errors (lab 2026-10-11: the laser reached the end 2–4 mm after the 18 mm ramp). After the
commanded ramp the measured flange still deviates from the commanded motion while the servo settles; the overscan adds the distance
v·t_settle travelled until that transient has decayed to the cruise's own steady level.

Model. Per leg the commanded along-track motion is the planned trapezoid (length L, cruise v, acceleration a; the controller's
time-optimal straight-line retiming) started at an unknown time t0 (start latency + the command-to-measurement lag, one shift);
t0 = least squares of the measured along-track positions. The deviation d(t) = measured − commanded(t − t0) during cruise, with
t measured from the end of the ramp, is pooled over the run's moving legs and modelled as

    E[d²](t) = σ_ss² + A²·exp(−2t/T)      (t ≥ 0)

(an exponentially decaying transient on the steady cruise scatter σ_ss — the servo's settling), fitted by Gaussian maximum
likelihood. t_settle = the time the transient amplitude A·exp(−t/T) takes to fall to σ_ss: T·ln(A/σ_ss) (0 when A ≤ σ_ss).
The prior is the controller's estimated command-to-measurement lag τ̂ (tracking_lag): a delayed follower reaches the cruise τ̂ after
the commanded ramp end; a platform that reports no lag gives the pose sample period of its legs instead (a sampled follower is at
most one controller cycle late). The data alone carry little (their serial correlation): the prior keeps the estimate physical.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = ["trapezoid", "leg_deviation", "settle_time"]

FIT_STEP_S = 2e-3          # [R] coarse grid step of the start time t0 (then refined)


def trapezoid(t, L: float, v: float, a: float) -> tuple:
    """Along-track position of a rest-to-rest straight leg (cruise v, acceleration a; triangular when L < v²/a) at times t [s] from
    its start → (s (…), ramp time t_a, cruise time t_c, total T)."""
    t = np.asarray(t, dtype=float)
    ta = v / a
    da = 0.5 * a * ta * ta
    if 2.0 * da > L:
        ta = math.sqrt(L / a)
        v = a * ta
        da = 0.5 * L
    tc = (L - 2.0 * da) / v
    T = 2.0 * ta + tc
    tt = np.clip(t, 0.0, T)
    s = np.where(tt < ta, 0.5 * a * tt * tt, np.where(tt < ta + tc, da + v * (tt - ta), L - 0.5 * a * (T - tt) ** 2))
    return s, ta, tc, T


def leg_deviation(t_s, p_m, v_mps: float, a_mps2: float) -> Optional[dict]:
    """One leg's measured flange positions p (N, 3) [m] at times t (N,) [s] → its along-track deviation from the commanded
    trapezoid during cruise: {'t_after_ramp_s', 'dev_m', 't0_s', 'ramp_s', 'cruise_s', 'v_mps'}; None for a leg too short or static."""
    t = np.asarray(t_s, dtype=float)
    p = np.asarray(p_m, dtype=float).reshape(-1, 3)
    if t.size < 8 or not (v_mps > 0 and a_mps2 > 0):
        return None
    o = np.argsort(t)
    t, p = t[o], p[o]
    d = p[-1] - p[0]
    L = float(np.linalg.norm(d))
    if not L > 1e-3:                                    # [R] a static leg (dwell)
        return None
    u = d / L
    s = (p - p[0]) @ u

    def sse(t0):
        return float(np.sum((s - trapezoid(t - t0, L, v_mps, a_mps2)[0]) ** 2))
    _, ta, tc, T = trapezoid(0.0, L, v_mps, a_mps2)
    g = np.arange(t[0] - T, t[-1], FIT_STEP_S)
    e = np.array([sse(x) for x in g])
    t0 = float(g[int(np.argmin(e))])
    from scipy.optimize import minimize_scalar
    t0 = float(minimize_scalar(sse, bounds=(t0 - FIT_STEP_S, t0 + FIT_STEP_S), method="bounded", options={"xatol": 1e-5}).x)
    sc, ta, tc, T = trapezoid(t - t0, L, v_mps, a_mps2)
    tr = t - t0 - ta
    cr = (tr >= 0.0) & (tr <= tc)
    if np.count_nonzero(cr) < 4:
        return None
    dt = np.diff(t)
    dt = dt[dt > 0]
    return {"t_after_ramp_s": tr[cr], "dev_m": (s - sc)[cr], "t0_s": t0, "ramp_s": ta, "cruise_s": tc, "v_mps": float(v_mps),
            "sample_period_s": float(np.median(dt)) if dt.size else None}


def _iat(x: np.ndarray) -> float:
    """Integrated autocorrelation time (in samples) of a series: 1 + 2 Σ ρ_k over the initial positive sequence (Geyer 1992)."""
    x = np.asarray(x, dtype=float) - float(np.mean(x))
    n = x.size
    if n < 8 or not float(np.dot(x, x)) > 0:
        return 1.0
    f = np.fft.rfft(np.r_[x, np.zeros(n)])
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    tau = 1.0
    for k in range(1, n // 2):
        if ac[k] <= 0:
            break
        tau += 2.0 * ac[k]
    return float(tau)


def settle_time(devs: list, lag_prior_s: Optional[float] = None, step_s: float = 5e-3) -> dict:
    """Pooled settling time of the run's moving legs (leg_deviation results), Bayesian change point of the deviation's variance:
    E[d²] = σ_tr² for t < t_s after the ramp, σ_ss² after (σ_tr ≥ σ_ss — a transient is larger than the steady scatter; otherwise
    no settling), profile likelihood in t_s (σ's at their ML values), the samples' serial correlation taken into account (the log
    likelihood per sample divided by the integrated autocorrelation time of d²), prior N(τ̂, τ̂²) on t_s ≥ 0 from the controller's
    command-to-measurement lag τ̂ (a delayed follower reaches the cruise τ̂ after the commanded ramp end; σ = τ̂: its order of
    magnitude only). → {'t_settle_s' (posterior mean), 'sigma_s', 'sigma_tr_m', 'sigma_ss_m', 'n_legs', 'source'}. No usable leg:
    the prior (no lag either: 0). step_s: [R] grid step of t_s."""
    devs = [d_ for d_ in devs if d_ is not None and np.size(d_["dev_m"]) >= 8]
    tau = float(lag_prior_s) if (lag_prior_s is not None and math.isfinite(float(lag_prior_s)) and lag_prior_s > 0) else None
    if tau is None:                                     # no lag estimate: one pose sample period (a sampled follower's cycle)
        per = [float(d_["sample_period_s"]) for d_ in devs if d_.get("sample_period_s")]
        tau = float(np.median(per)) if per else None
    if not devs:
        return {"t_settle_s": tau or 0.0, "sigma_s": tau if tau else None, "sigma_tr_m": None, "sigma_ss_m": None, "n_legs": 0,
                "source": "prior: the controller's command-to-measurement lag" if tau else "no leg and no lag estimate (0)"}
    t = np.concatenate([d_["t_after_ramp_s"] for d_ in devs])
    # each leg about its own cruise median (a constant along-track offset — the lag's residual, the direction offset — is not settling)
    r2 = np.concatenate([(d_["dev_m"] - np.median(d_["dev_m"])) ** 2 for d_ in devs])
    iat = float(np.median([_iat((d_["dev_m"] - np.median(d_["dev_m"])) ** 2) for d_ in devs]))
    o = np.argsort(t)
    t, r2 = t[o], r2[o]
    half = 0.5 * float(np.min([d_["cruise_s"] for d_ in devs]))
    grid = np.arange(0.0, half + 0.5 * step_s, step_s)
    c = np.r_[0.0, np.cumsum(r2)]
    n = t.size
    s_all = c[-1] / n
    ll = np.empty(grid.size)
    for j, ts in enumerate(grid):
        k = int(np.searchsorted(t, ts))
        if k < 2 or n - k < 2:
            ll[j] = -0.5 * n * math.log(s_all)
            continue
        v1, v2 = c[k] / k, (c[-1] - c[k]) / (n - k)
        ll[j] = -0.5 * (k * math.log(v1) + (n - k) * math.log(v2)) if v1 > v2 else -0.5 * n * math.log(s_all)
    ll = ll / max(iat, 1.0)
    lp = -0.5 * ((grid - tau) / tau) ** 2 if tau else np.zeros(grid.size)
    w = np.exp(ll + lp - np.max(ll + lp))
    w /= w.sum()
    m = float(np.sum(w * grid))
    sd = math.sqrt(float(np.sum(w * (grid - m) ** 2)))
    k = int(np.searchsorted(t, m))
    return {"t_settle_s": m, "sigma_s": sd, "sigma_tr_m": math.sqrt(c[k] / k) if k > 0 else None,
            "sigma_ss_m": math.sqrt((c[-1] - c[k]) / max(n - k, 1)), "n_legs": len(devs), "iat_samples": iat,
            "lag_prior_s": tau, "source": "measured: the run's legs (encoder along-track deviation after the ramp)"
            + ((" with the controller-lag prior" if (lag_prior_s is not None and lag_prior_s > 0) else " with the pose-sample-period prior")
               if tau else "")}
