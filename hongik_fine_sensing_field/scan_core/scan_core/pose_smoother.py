"""Flange pose at the sensor times of a straight leg: the commanded motion shifted by the controller's lag, plus the measured
deviation from it smoothed by a state-space model (user 2026-10-11 night: instead of linear interpolation of the joint samples only).

Model (along each base axis, positions p [m]):
    p(t)      = p_cmd(t − τ) + x(t)                      the true flange: the command τ later + a physical deviation x
    x(t)      local level: dx = w dt, w white with spectral density q        (servo deviation: a random walk between samples)
    p_meas,k  = p(t_k) + ε_k,  ε_k ~ N(0, R)                                   (the encoder samples' non-physical part: sample-time
                                                                               jitter × speed, quantisation)
p_cmd is the planned straight line with its trapezoid (rest to rest, cruise v, acceleration a); its start time and τ come from the
recorded controller stream (raw.npz ctrl_*: the reference starts moving at the command start; τ = the measured joints' lag behind the
reference, least squares as tracking_lag) or, for a leg without it, from the measured samples (one start shift = command start + τ,
least squares). (q, R) per axis by maximum likelihood (innovations), x by the Rauch–Tung–Striebel smoother, evaluated at the sensor
times (the smoothed state between samples: the conditional mean of a random walk = linear between the smoothed neighbours, with the
sample's own update). The evaluation that decides whether the estimator uses it is offline (fsv1_work/g1_3pass/smoother);
reconstruct keeps linear interpolation until then.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .settling import trapezoid

__all__ = ["command_start_from_ctrl", "smooth_axis", "smoothed_positions"]


def command_start_from_ctrl(ctrl_t_s: np.ndarray, ref_q: np.ndarray, fb_q: np.ndarray) -> tuple:
    """(command start time [s] — the first reference sample that moved —, lag τ [s] of the feedback behind the reference: least
    squares of e = q_ref(t) − q_fb(t) = τ·q̇_ref over the moving samples (tracking_lag's model))."""
    t = np.asarray(ctrl_t_s, dtype=float)
    R = np.asarray(ref_q, dtype=float)
    F = np.asarray(fb_q, dtype=float)
    mv = np.flatnonzero(np.any(np.abs(R - R[0]) > 0.0, axis=1))
    t_start = float(t[max(int(mv[0]) - 1, 0)]) if mv.size else float(t[0])
    v = np.gradient(R, t, axis=0)
    e = R - F
    m = np.any(np.abs(v) > 0, axis=1)
    den = float(np.sum(v[m] * v[m]))
    tau = float(np.sum(e[m] * v[m]) / den) if den > 0 else 0.0
    return t_start, max(tau, 0.0)


def _kalman_ll(y: np.ndarray, dt: np.ndarray, q: float, R: float) -> float:
    """Innovations log-likelihood of a local level (random walk with density q, noise R); diffuse start."""
    x, P = float(y[0]), R
    ll = 0.0
    for k in range(1, y.size):
        P = P + q * dt[k - 1]
        S = P + R
        r = y[k] - x
        ll += -0.5 * (math.log(2 * math.pi * S) + r * r / S)
        K = P / S
        x += K * r
        P = (1 - K) * P
    return ll


def smooth_axis(t: np.ndarray, y: np.ndarray, t_out: np.ndarray) -> tuple:
    """Local-level RTS smoother of y(t) at t_out with ML (q, R) → (x̂(t_out), q, R)."""
    from scipy.optimize import minimize
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = np.diff(t)
    v0 = float(np.var(np.diff(y))) + 1e-24
    dtm = max(float(np.median(dt)), 1e-6)
    lim = (-30.0, 10.0)                                 # [R] search range of (q, R) relative to the data's increment variance

    def unpack(th):
        th = np.clip(th, *lim)
        return v0 / dtm * math.exp(th[0]), v0 * math.exp(th[1])
    best = None
    for f in (0.01, 0.5, 0.99):
        th0 = np.log([f, 0.5 * (1.0 - f)])
        r = minimize(lambda th: -_kalman_ll(y, dt, *unpack(th)), th0, method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
        if best is None or r.fun < best.fun:
            best = r
    q, R = unpack(best.x)
    n = y.size
    xf, Pf, xp, Pp = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    x, P = float(y[0]), R
    xf[0], Pf[0], xp[0], Pp[0] = x, P, x, P
    for k in range(1, n):
        Pk = P + q * dt[k - 1]
        xp[k], Pp[k] = x, Pk
        K = Pk / (Pk + R)
        x = x + K * (y[k] - x)
        P = (1 - K) * Pk
        xf[k], Pf[k] = x, P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] / Pp[k + 1]
        xs[k] = xf[k] + C * (xs[k + 1] - xp[k + 1])
    return np.interp(np.asarray(t_out, dtype=float), t, xs), q, R


def smoothed_positions(t_s, p_m, t_out, v_mps: float, a_mps2: float, *, ctrl: Optional[dict] = None) -> dict:
    """Leg samples (t, p (N,3)) → flange positions at t_out: command (trapezoid along the straight leg, start = ctrl command start +
    τ, else fitted) + smoothed deviation. → {'p': (M,3), 'p_cmd': (M,3), 'shift_s', 'qR': [(q, R)] * 3, 'source'}."""
    from .settling import leg_deviation
    t = np.asarray(t_s, dtype=float)
    p = np.asarray(p_m, dtype=float).reshape(-1, 3)
    o = np.argsort(t)
    t, p = t[o], p[o]
    d = p[-1] - p[0]
    L = float(np.linalg.norm(d))
    u = d / L
    if ctrl is not None:
        t0, tau = command_start_from_ctrl(ctrl["t_s"], ctrl["ref"], ctrl["fb"])
        shift, src = t0 + tau, "controller stream (command start + lag)"
    else:
        dev = leg_deviation(t, p, v_mps, a_mps2)
        shift, src = float(dev["t0_s"]), "fitted (command start + lag in one shift)"

    def cmd(tq):
        s, _, _, _ = trapezoid(np.asarray(tq, dtype=float) - shift, L, v_mps, a_mps2)
        return p[0][None, :] + s[:, None] * u[None, :]
    r = p - cmd(t)
    out, qR = np.zeros((np.size(t_out), 3)), []
    for ax in range(3):
        xs, q, R = smooth_axis(t, r[:, ax], t_out)
        out[:, ax] = xs
        qR.append((q, R))
    pc = cmd(t_out)
    return {"p": pc + out, "p_cmd": pc, "shift_s": shift, "qR": qR, "source": src}
