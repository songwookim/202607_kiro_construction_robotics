"""Joint-state source detection (scan_core.js_source, user 2026-10-12): measured encoder angles vs the control box's reference echo,
from the values and their order only (never the stamps), and the time-offset prior the hypotheses imply."""
from __future__ import annotations

import math

import numpy as np
import pytest

from scipy.special import log_ndtr

from scan_core import js_source as J

D2R = math.pi / 180.0
COUNT_DEG = 360.0 / (16384 * 101)          # an RB20-like encoder count through the gear (the detector does not know it)


def _ros(deg):
    """rbpodo read(): the control box's float32 degrees × DEG2RAD in double."""
    return np.asarray(deg, dtype=np.float32).astype(np.float64) * D2R


def _command(t, q0, q1, t0, t1):
    """A rest-to-rest joint move (smooth cosine ramp) from q0 to q1 [deg] between t0 and t1."""
    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return q0[None, :] + (q1 - q0)[None, :] * (0.5 - 0.5 * np.cos(math.pi * s))[:, None]


def _filtered(t, cmd, tau):
    """The control box's reference: the command through a first-order filter (time constant tau), float32 degrees — it converges on
    the target and then holds bit-exactly."""
    out = np.empty_like(cmd)
    out[0] = cmd[0]
    a = np.exp(-np.diff(t) / tau)
    for i in range(1, len(t)):
        out[i] = a[i - 1] * out[i - 1] + (1.0 - a[i - 1]) * cmd[i]
    return out.astype(np.float32).astype(np.float64)


def _streams(seed=0, cb_hz=100.0, read_hz=50.0, jitter=0.3):
    """One leg as a hardware interface records it: 0.4 s rest, a 6 s move, 0.6 s rest; the control box ticks at cb_hz, read() at a
    jittered read_hz clock takes the latest packet (zero-order hold). → (t_read_ns, echo (rad), measured (rad), motions)."""
    rng = np.random.default_rng(seed)
    T = 7.0
    t_cb = np.arange(0.0, T, 1.0 / cb_hz)
    q0 = np.array([178.07, -45.07, -73.0, -5.58, -60.4, 355.2])
    q1 = q0 + np.array([3.1, 20.5, -35.2, 0.6, 14.6, -3.75])
    cmd = _command(t_cb, q0, q1, 0.4, 6.4)
    ref = _filtered(t_cb, cmd, 0.03)
    # the arm trails the reference by one tick (servo) with a little vibration; the encoder counts it (lattice, own zero), then the
    # control box writes float32 degrees
    lag = np.vstack([ref[:1], ref[:-1]])
    vib = 2e-4 * np.sin(2 * math.pi * 3.5 * t_cb)[:, None] * (t_cb[:, None] < 6.6)
    zero = rng.uniform(0, COUNT_DEG, 6)
    dither = rng.normal(0.0, 0.35 * COUNT_DEG, lag.shape)
    ang = (np.round((lag + vib + dither - zero) / COUNT_DEG) * COUNT_DEG + zero).astype(np.float32).astype(np.float64)
    tr = [0.0]
    while tr[-1] < T - 2.0 / read_hz:
        tr.append(tr[-1] + (1.0 + jitter * (2.0 * rng.random() - 1.0)) / read_hz)
    tr = np.array(tr)
    idx = np.searchsorted(t_cb, tr, side="right") - 1
    t_ns = (tr * 1e9).astype(np.int64) + 1_000_000_000
    motions = [{"t_start_ns": int(0.4e9) + 1_000_000_000, "t_stop_ns": int(6.4e9) + 1_000_000_000}]
    return t_ns, _ros(ref[idx]), _ros(ang[idx]), motions


def test_representation_is_read_from_the_values():
    deg = np.array([178.0712, -45.07, 0.0031, 355.2])
    rep = J.representation(_ros(deg))
    assert rep["dtype"] == "float32" and rep["unit"] == "deg"
    assert np.array_equal(rep["values"], np.asarray(deg, np.float32).astype(np.float64))
    rep = J.representation(np.array([0.1, 0.2 + 1e-12]))
    assert rep["dtype"] == "float64"


def test_beta_integral_is_exact():
    e = np.linspace(0.0, 1.0, 400001)
    for n, m, L in ((10, 7, 5.0), (300, 280, 14.0), (300, 30, 14.0), (50, 0, 2.0)):
        f = m * np.log((1 - e) * L + e) + (n - m) * np.log(np.maximum(e, 1e-300))
        num = float(np.log(np.trapz(np.exp(f - f.max()), e)) + f.max())
        assert J._ln_g(np.array([m]), n, L)[0] == pytest.approx(num, abs=2e-3)


@pytest.mark.parametrize("read_hz", [100.0, 50.0, 30.0])
def test_encoder_and_echo_are_told_apart_through_a_jittered_read_clock(read_hz):
    """KIRO-like read(): 30–100 Hz, ±30 % jitter, zero-order hold. The encoder's values sit on its count lattice (decisive H_meas);
    the reference echo is continuous and holds at rest (H_ref with its Occam odds, ~½ ln n per joint) — the stamps are not used."""
    t_ns, echo, meas, motions = _streams(seed=int(read_hz), read_hz=read_hz)
    em = J.leg_evidence(meas, t_ns, motions)
    er = J.leg_evidence(echo, t_ns, motions)
    assert em["representation"] == er["representation"] == "float32 deg"
    assert all(j["ln_bf"] > 50.0 for j in em["lattice"]), [j["ln_bf"] for j in em["lattice"]]
    # the count (or a sub-lattice of it: every count is also on Δ/2 — both are in the marginal)
    assert all(any(j["period"] == pytest.approx(COUNT_DEG / m, rel=1e-6) for m in (1, 2, 3)) for j in em["lattice"])
    assert all(j["ln_bf"] < 0.0 for j in er["lattice"]), [j["ln_bf"] for j in er["lattice"]]
    pm, pr = J.pool([em]), J.pool([er])
    k = 3.0
    assert J.verdict(pm["ln_odds_ref"], k) == J.H_MEAS and pm["p_ref"] < 1e-20
    assert er["ln_bf"]["rest"] <= 0.0                                   # the echo holds after its tail
    assert J.verdict(pr["ln_odds_ref"], k) == J.H_REF, pr
    # the evidence does not depend on the stamps: a stamp jitter that reorders nothing changes nothing
    t2 = t_ns + np.random.default_rng(1).integers(0, 3_000_000, t_ns.size)
    t2 = np.maximum.accumulate(t2)
    assert J.leg_evidence(echo, t2, motions)["ln_bf"]["lattice"] == pytest.approx(er["ln_bf"]["lattice"])


def test_the_representation_grid_and_a_cartesian_line_are_no_lattice():
    """A reference stream is on the representation's own grid (float32 degrees): the exact rounding model gives the grid's lattices
    (Δ = m·u) ratios 0 or m — no evidence. A straight Cartesian leg (rest to rest, trapezoid along the line) moves a joint
    non-linearly: no lattice of its own either. (Known limit, js_source doc: a joint at exactly constant JOINT velocity sampled at an
    exactly periodic tick is an arithmetic progression — every v·T/m is a lattice of its cruise; our recorded legs are Cartesian
    lines and dwells, and the real echo streams show none.)"""
    t = np.arange(0.0, 10.0, 0.01)                                          # exact 100 Hz tick
    v, ta = 40.0, 1.0                                                       # mm/s along the line, ramp time
    s_ = np.where(t < ta, 0.5 * v / ta * t ** 2, np.where(t < 10.0 - ta, 0.5 * v * ta + v * (t - ta),
                                                           0.5 * v * ta + v * (10.0 - 2 * ta) + 0.5 * v / ta * (ta * ta - (10.0 - t) ** 2)))
    for reach_mm, ang0 in ((700.0, 178.0), (900.0, -45.0), (500.0, 5.6)):
        deg = (ang0 + np.degrees(np.arctan2(s_ - 180.0, reach_mm) - np.arctan2(-180.0, reach_mm))).astype(np.float32).astype(np.float64)
        e = J.lattice_evidence(deg, np.spacing(np.abs(deg).astype(np.float32)).astype(np.float64))
        assert e["ln_bf"] < 0.0, (reach_mm, e)
    slow = (178.0 + 0.002 * t + 0.03 * (1 - np.cos(t))).astype(np.float32).astype(np.float64)    # increments of a few units
    e = J.lattice_evidence(slow, np.spacing(np.abs(slow).astype(np.float32)).astype(np.float64))
    assert e["ln_bf"] <= 0.0, e


def test_rest_hold_vs_dither():
    """After the tail an echo holds (H_ref Occam odds); changes long after it are an encoder's (H_meas)."""
    tail = np.r_[np.ones(15, bool), np.zeros(60, bool)]
    assert J.rest_evidence([tail, tail])["ln_bf"] < -2.0
    rng = np.random.default_rng(0)
    dith = np.r_[np.ones(15, bool), rng.random(60) < 0.15]
    dith[-3] = True
    assert J.rest_evidence([dith, dith])["ln_bf"] > 2.0
    assert J.rest_evidence([])["ln_bf"] == 0.0


def test_delta_prior_measured_is_todays_and_echo_is_gaussian_times_probit():
    s = 0.004
    dm = J.DeltaPrior(s, 0.0)
    lam, lam_m = dm.site(0.02, 0.01 ** 2)
    assert lam == pytest.approx(1.0 / s ** 2, rel=1e-12) and abs(lam_m) < 1e-9
    assert dm.site(0.0, math.inf) == pytest.approx((1.0 / s ** 2, 0.0))
    assert dm.prior_precision() == pytest.approx(1.0 / s ** 2)
    dr = J.DeltaPrior(s, 1.0)
    assert dr.prior_precision() == 0.0
    d = np.linspace(-0.2, 0.3, 500001)
    for m_L, V_L in ((0.025, 0.008 ** 2), (-0.01, 0.006 ** 2), (0.0, 0.05 ** 2)):
        from scipy.special import ndtr
        p = np.exp(-0.5 * (d - m_L) ** 2 / V_L) * ndtr(d / s)
        mu = float(np.trapz(p * d, d) / np.trapz(p, d))
        var = float(np.trapz(p * (d - mu) ** 2, d) / np.trapz(p, d))
        mu_r, v_r = dr.moments(m_L, V_L)
        assert mu_r == pytest.approx(mu, abs=1e-6) and v_r == pytest.approx(var, rel=1e-4)
        lam, lam_m = dr.site(m_L, V_L)                                  # the site reproduces those moments with the data
        assert 1.0 / (1.0 / V_L + lam) == pytest.approx(v_r, rel=1e-9)
        assert (m_L / V_L + lam_m) / (1.0 / V_L + lam) == pytest.approx(mu_r, abs=1e-12)
    # no data on δ: the echo's posterior is improper — δ is free (site 0), whatever the weight
    assert dr.site(0.0, math.inf) == (0.0, 0.0) and J.DeltaPrior(s, 1e-6).site(0.0, math.inf) == (0.0, 0.0)
    # undecided: the mixture's moments cover both hypotheses
    du = J.DeltaPrior(s, 0.5)
    (mm, vm, _), (mr, vr, _) = du.branches(0.025, 0.008 ** 2).values()
    mu, v = du.moments(0.025, 0.008 ** 2)
    assert min(mm, mr) < mu < max(mm, mr) and v >= min(vm, vr)


def _brute(logp, lo, hi, n=2_000_001):
    d = np.linspace(lo, hi, n)
    lp = logp(d)
    w = np.exp(lp - lp.max())
    mu = float(np.sum(w * d) / np.sum(w))
    return mu, float(np.sum(w * (d - mu) ** 2) / np.sum(w))


def test_delta_prior_one_sided_factors_bound_a_flat_lag_and_leave_the_measured_prior_exact():
    """The one-sided end readings' factors on δ, Φ((c − β·δ)/s)^w (seam_estimate.fuse_ends: a site inside the seam read at rest
    bounds how far the lag can have moved the end a moving pass read): with the data's Gaussian on δ — however weak (test_m: the
    design length through a 1.2 mm/s difference of the end speeds, σ 4.6 s) — the flat echo lag's posterior is prior × data ×
    factors, exact 1-D moments (δ between 0 and the bound, not seconds out on the half-line); a flat lag the Gaussian readings leave
    unidentified stays free (the bounds alone leave its location within them arbitrary); a factor that is 1 to double precision
    over a branch's support changes nothing (the measured prior's site stays (1/σ_q², 0) exactly)."""
    from scipy.special import log_ndtr
    s = 0.004
    up = (0.03, 0.015, 0.0006, 1.0)                      # an end read 15 mm before a site at rest, pass at 30 mm/s: δ ≤ 0.5 s
    dr = J.DeltaPrior(s, 1.0)
    assert dr.site(0.0, math.inf) == (0.0, 0.0) and dr.site(0.0, math.inf, [up]) == (0.0, 0.0)
    # the data's weak Gaussian on δ: without the factor the flat lag's mean runs seconds out; with it, between 0 and the bound
    half = (0.03, 0.009, 0.0006, 0.5)                    # a half-weighted second one (§4.9 leg weight)
    m_L, V_L = -0.3, 4.6 ** 2
    assert dr.moments(m_L, V_L)[0] > 3.0
    mu, v = dr.moments(m_L, V_L, [up])
    bm, bv = _brute(lambda d: -0.5 * (d - m_L) ** 2 / V_L + log_ndtr(d / s) + log_ndtr((up[1] - up[0] * d) / up[2]), -0.05, 0.6)
    assert mu == pytest.approx(bm, abs=1e-6) and v == pytest.approx(bv, rel=1e-4) and 0.2 < mu < 0.3, (mu, v)
    lam, lam_m = dr.site(m_L, V_L, [up])
    assert 1.0 / (1.0 / V_L + lam) == pytest.approx(v, rel=1e-9) and (m_L / V_L + lam_m) / (1.0 / V_L + lam) == pytest.approx(mu)
    mu, v = dr.moments(m_L, V_L, [up, half])
    bm, bv = _brute(lambda d: -0.5 * (d - m_L) ** 2 / V_L + log_ndtr(d / s) + log_ndtr((up[1] - up[0] * d) / up[2])
                    + 0.5 * log_ndtr((half[1] - half[0] * d) / half[2]), -0.05, 0.6)
    assert mu == pytest.approx(bm, abs=1e-6) and v == pytest.approx(bv, rel=1e-4), (mu, v, bm, bv)
    low = (-0.03, 0.03, 0.0006, 1.0)                     # δ ≥ −1 s
    # the measured prior: a far factor is 1 on its support — exactly today's site and branch
    dm = J.DeltaPrior(s, 0.0)
    assert dm.site(0.02, 0.01 ** 2, [up, low]) == (1.0 / s ** 2, 0.0) and dm.site(0.0, math.inf, [up]) == (1.0 / s ** 2, 0.0)
    assert dm.branches(0.02, 0.01 ** 2, [up])["meas"] == dm.branches(0.02, 0.01 ** 2)["meas"]
    # … a near one (the end read 0.3 mm before the site) tilts it — exact moments of N(0, σ_q²) × data × factor
    near = (0.03, 0.0003, 0.0006, 1.0)
    mu, v = dm.moments(0.002, 0.003 ** 2, [near])
    vg = 1.0 / (1.0 / s ** 2 + 1.0 / 0.003 ** 2)
    mg = vg * 0.002 / 0.003 ** 2
    bm, bv = _brute(lambda d: -0.5 * (d - mg) ** 2 / vg + log_ndtr((near[1] - near[0] * d) / near[2]), -0.05, 0.05)
    assert mu == pytest.approx(bm, abs=1e-8) and v == pytest.approx(bv, rel=1e-5) and mu < mg, (mu, v, bm, bv)
    assert dm.site(0.002, 0.003 ** 2, [near]) != (1.0 / s ** 2, 0.0)


def test_delta_prior_with_an_earlier_fusions_likelihood_keeps_the_lags_half_line():
    """A later fusion's echo branch takes an earlier fusion's LIKELIHOOD of δ, N(m0, v0): Φ(δ/σ_q)·N(m0, v0) × its own data — exact
    moments, proper also where its own readings carry nothing on δ (then the branch is Φ·N(m0, v0) itself), and δ a random effect
    again (the mixture's precision). The KIRO stand-in r13 pattern (2026-10-13): both fusions' data favour δ ≈ −0.8 … −0.9 s with
    weak precision (speed contrasts of 3 %); the earlier fusion's echo posterior (0.52 ± 0.37 s) handed on as a Gaussian prior took
    the plate-end δ to −0.47 s — outside the flat lag's support. As a likelihood the branch stays where Φ(δ/σ_q) has its mass."""
    s, m0, v0 = 0.0147, -0.79, 1.15 ** 2
    m_L, V_L = -0.90, 0.25 ** 2
    dp = J.DeltaPrior(s, 1.0, ref_lik_mean=m0, ref_lik_var=v0)
    mu, v = dp.moments(m_L, V_L)
    bm, bv = _brute(lambda d: log_ndtr(d / s) - 0.5 * (d - m0) ** 2 / v0 - 0.5 * (d - m_L) ** 2 / V_L, -0.3, 1.5)
    assert mu == pytest.approx(bm, abs=1e-7) and v == pytest.approx(bv, rel=1e-4), (mu, v, bm, bv)
    # on the half-line: the data's Gaussian (−0.90 ± 0.24 s) cut by Φ(δ/σ_q) — an exponential-like branch of scale V/|m| ≈ 66 ms
    assert 0.0 < mu < 0.1 and math.sqrt(v) < 0.1, (mu, v)
    # the Gaussian-posterior hand-off it replaces: N(0.523, 0.368²) × N(m_L, V_L) — far below the support
    g_m, g_v = 0.5228, 0.3677 ** 2
    vg = 1.0 / (1.0 / g_v + 1.0 / V_L)
    assert vg * (g_m / g_v + m_L / V_L) < -0.4
    # no own data: the branch is Φ·N(m0, v0); a proper prior, its moments; the mixture's precision
    mt, vt, _ = J.probit_tilt(m0, v0, s)
    assert dp.moments(0.0, math.inf) == pytest.approx((mt, vt))
    assert dp.site(0.0, math.inf) == pytest.approx((1.0 / vt, mt / vt))
    assert dp.prior_precision() == pytest.approx(1.0 / vt)
    # a proper earlier likelihood far from the half-line: the probit is 1 there — the Gaussian product exactly
    dq = J.DeltaPrior(0.004, 1.0, ref_lik_mean=0.024, ref_lik_var=0.003 ** 2)
    mu, v = dq.moments(0.030, 0.006 ** 2)
    vv = 1.0 / (1.0 / 0.006 ** 2 + 1.0 / 0.003 ** 2)
    assert v == pytest.approx(vv, rel=1e-6) and mu == pytest.approx(vv * (0.030 / 0.006 ** 2 + 0.024 / 0.003 ** 2), rel=1e-6)
    # the measured source is unchanged by it
    assert J.DeltaPrior(s, 0.0, ref_lik_mean=m0, ref_lik_var=v0).site(0.01, 0.002 ** 2)[0] == pytest.approx(1.0 / s ** 2)


def test_flat_lag_fit_is_the_moment_projection_onto_the_flat_lag_family():
    """flat_lag_fit(μ, v, σ_q) → (m, V) with Φ(δ/σ_q)·N(m, V) of moments (μ, v): a family member is returned exactly; a member
    tilted by one-sided factors (an end bound) is projected — its moments matched to double precision."""
    for m, V, s in ((-0.79, 1.15 ** 2, 0.0147), (0.025, 0.003 ** 2, 0.003), (0.004, 0.02 ** 2, 0.0147), (-0.02, 0.005 ** 2, 0.004)):
        mu, v, _ = J.probit_tilt(m, V, s)
        mf, Vf = J.flat_lag_fit(mu, v, s)
        assert mf == pytest.approx(m, abs=1e-7 * math.sqrt(V) + 1e-12) and Vf == pytest.approx(V, rel=1e-6), (m, V, mf, Vf)
    s, m_L, V_L = 0.0147, -0.79, 1.15 ** 2
    up = (1.0, 0.9, 0.05, 1.0)                           # δ ≲ 0.9 s (an end bound)
    dp = J.DeltaPrior(s, 1.0)
    b, tilted = dp._branches(m_L, V_L, [up], ("ref",))
    assert tilted == ["ref"]
    mu, v, _ = b["ref"]
    mf, Vf = J.flat_lag_fit(mu, v, s, x0=(m_L, V_L))
    mq, vq, _ = J.probit_tilt(mf, Vf, s)
    assert mq == pytest.approx(mu, abs=1e-9) and vq == pytest.approx(v, rel=1e-8), (mu, v, mq, vq)


def test_plate_end_echo_branch_vectorised_equals_the_delta_prior():
    """end_faces.echo_branch_vec (the plate-end fusion's echo branch at every variance-grid point) is DeltaPrior's echo branch:
    moments elementwise, its log-evidence = ln Z_ref − ln Z_meas (− ln Z0 of the earlier likelihood's prior); grid points whose
    readings carry nothing on δ take the prior branch; without an earlier likelihood such a point leaves the flat lag improper."""
    from scan_core.end_faces import echo_branch_vec
    s = 0.0147
    m_L = np.array([-0.9, -0.2, 0.0, 0.03, 0.4, 0.0])
    V_L = np.array([0.25, 0.1, 0.02, 0.005, 0.3, np.inf]) ** 2
    for dp in (J.DeltaPrior(s, 1.0, ref_lik_mean=-0.79, ref_lik_var=1.15 ** 2), J.DeltaPrior(s, 1.0, ref_lik_mean=0.02,
                                                                                              ref_lik_var=0.01 ** 2)):
        mu, v, dlw = echo_branch_vec(dp, m_L, V_L)
        lZ0 = float(log_ndtr(dp.ref_lik_mean / math.sqrt(dp.ref_lik_var + s * s)))
        for i in range(m_L.size):
            b = dp.branches(float(m_L[i]), float(V_L[i]))
            assert mu[i] == pytest.approx(b["ref"][0], abs=1e-12) and v[i] == pytest.approx(b["ref"][1], rel=1e-12), i
            want = 0.0 if not np.isfinite(V_L[i]) else b["ref"][2] - lZ0 - b["meas"][2]
            assert dlw[i] == pytest.approx(want, abs=1e-9), (i, dlw[i], want)
    flat = J.DeltaPrior(s, 1.0)
    assert echo_branch_vec(flat, m_L, V_L) is None
    mu, v, dlw = echo_branch_vec(flat, m_L[:-1], V_L[:-1])
    for i in range(m_L.size - 1):
        b = flat.branches(float(m_L[i]), float(V_L[i]))
        assert mu[i] == pytest.approx(b["ref"][0], abs=1e-12) and dlw[i] == pytest.approx(b["ref"][2] - b["meas"][2], abs=1e-9)


def test_groove_fusion_hands_on_its_likelihood_exactly_or_projected():
    """seam_estimate._ref_likelihood: where the one-sided factors leave the echo branch as it is, the groove fusion's own Gaussian
    likelihood of δ exactly; where they tilt it, the Gaussian whose flat-lag posterior has the tilted branch's moments — so the
    plate-end fusion's branch without its own data reproduces the groove fusion's echo posterior; none where the readings leave δ
    unidentified or the echo has no weight."""
    from scan_core.seam_estimate import _ref_likelihood
    s = 0.0147
    dp = J.DeltaPrior(s, 1.0)
    far = (1.0, 30.0, 0.05, 1.0)                         # an end bound 30 s away: 1 on the branch's support
    assert _ref_likelihood(dp, {"m_L": -0.79, "V_L": 1.15 ** 2, "factors": [far]}) == \
        {"mean_s": -0.79, "var_s": 1.15 ** 2, "projected": False}
    up = (1.0, 0.9, 0.05, 1.0)
    rl = _ref_likelihood(dp, {"m_L": -0.79, "V_L": 1.15 ** 2, "factors": [up]})
    assert rl["projected"]
    want = dp.branches(-0.79, 1.15 ** 2, [up])["ref"]
    got = J.DeltaPrior(s, 1.0, ref_lik_mean=rl["mean_s"], ref_lik_var=rl["var_s"]).moments(0.0, math.inf)
    assert got[0] == pytest.approx(want[0], abs=1e-9) and got[1] == pytest.approx(want[1], rel=1e-8)
    assert _ref_likelihood(dp, {"m_L": 0.0, "V_L": math.inf, "factors": [up]}) is None
    assert _ref_likelihood(J.DeltaPrior(s, 0.0), {"m_L": -0.79, "V_L": 1.15 ** 2}) is None


