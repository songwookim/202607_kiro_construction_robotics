"""Noise-prior calibration (scan_l0.calib_noise_priors): the type-II ML over runs integrates each run's likelihood against the
population prior with the prior's exact bin probabilities — a population far narrower than the σ grid step (ν0 in the thousands)
neither spikes at grid nodes nor loses its mass (2026-10-12: the 25-run fit stopped at such spikes)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from scan_l0 import calib_noise_priors as C


@pytest.mark.parametrize("nu", [0.7, 3.0, 40.0, 3.0e3, 1.0e6])
def test_bin_masses_are_the_whole_prior_at_any_width(nu):
    _lf, edges = C._fine_axis()
    for v in (0.015, 0.06):
        m = np.exp(C._log_mass(edges, (1.8e-5) ** 2, 1e-9, 1e-6, nu, v))
        assert np.all(m >= 0.0) and float(np.sum(m)) == pytest.approx(1.0, abs=1e-9)


def _chi2_ll(s_hat2: float, nu_run: float) -> np.ndarray:
    """A run's likelihood of σ on GRID from a χ² statistic: ν ŝ²/σ² ~ χ²_ν."""
    x = C.GRID ** 2
    return -0.5 * nu_run * np.log(x) - 0.5 * nu_run * s_hat2 / x


def test_a_narrow_population_is_the_run_likelihood_at_its_centre_between_grid_nodes():
    """ν0 → ∞: the marginal tends to L(σ0) — also for σ0 midway between two grid nodes (a density sum on the grid gave a spike
    at the nodes and ~nothing between them)."""
    lg = np.log(C.GRID)
    lf, edges = C._fine_axis()
    from scipy.interpolate import CubicSpline
    from scipy.special import logsumexp
    ll = _chi2_ll((2.0e-5) ** 2, 4.0)
    llf = CubicSpline(lg, ll)(lf)
    out = []
    for k in (50, 50.5, 51):                                   # node, midway, node
        s0 = math.exp(lg[0] + k * (lg[1] - lg[0]))
        lm = float(logsumexp(llf + C._log_mass(edges, s0 * s0, 0.0, 0.0, 1.0e6, 0.03)))
        exact = -0.5 * 4.0 * math.log(s0 * s0) - 0.5 * 4.0 * (2.0e-5) ** 2 / (s0 * s0)
        assert lm == pytest.approx(exact, abs=2e-3)
        out.append(lm)
    assert abs(out[1] - 0.5 * (out[0] + out[2])) < 0.05         # smooth in σ0 — no spike at the nodes


def test_fit_recovers_a_narrow_population_off_the_grid():
    rng = np.random.default_rng(7)
    s_true = 1.7e-5                                            # between the grid nodes 1.58e-5 and 1.78e-5
    lls, vs = [], []
    for k in range(24):
        nu_run = 40.0                                          # each speed group: 8 runs × 40 dof → the scale to ~4 %
        s_hat2 = s_true ** 2 * rng.chisquare(nu_run) / nu_run  # population ν0 ≈ ∞: every run at s_true
        lls.append(_chi2_ll(s_hat2, nu_run))
        vs.append((0.02, 0.04, 0.06)[k % 3])
    pars, _lml = C._fit(lls, vs, 1)
    a, b, c, nu0 = pars[0]
    for v in (0.02, 0.04, 0.06):
        s0 = math.sqrt(a + b * v * v + c * v ** 4)
        assert s0 == pytest.approx(s_true, rel=0.12), (v, s0)
    assert nu0 > 20.0                                          # a homogeneous population is found as such
