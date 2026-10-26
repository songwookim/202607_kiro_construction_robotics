"""Calibrated variance components of the 1G estimator (calib file 'g1: noise_priors:') — hierarchical (empirical Bayes) priors.

Why. A run's scatter quantities — the per-crossing along-track offset of the plate-end readings σ_ζ, the per-reading scatter τ, the
legs' lateral / normal offset scatter σ_η,w / σ_η,n — are properties of the robot and the sensor, not of one run. Learned only
within a run they are unknown with few legs (2 passes: no redundant crossing, σ = ∞). Here each run's value is a draw from the
robot + sensor population, calibrated from saved runs:

    σ²_run(v) = λ_run · s0²(v),   λ_run ~ Scaled-Inv-χ²(ν0, 1),   s0²(v) = a + b·v² + c·v⁴

  a [m²]      the speed-independent part (positioning, sensor),
  b [s²]      a time error times the speed (pose ↔ profile timing, servo look-ahead jitter: an along-track offset v·t_err),
  c [s⁴/m²]   an error that grows with v² (the servo's dynamic deviation; the even expansion of σ²(v) one order further),
  ν0          the population's degrees of freedom (how much a new run is like the calibration runs).
The run's own data update it (posterior ν = ν0 + ν_run): the estimator multiplies its restricted likelihood of the component by
this prior. A new speed is handled by the speed model s0²(v), not by a table. The values are only consumed here: how they were
obtained (another robot, another procedure) does not matter to the estimator.

Calib block (all keys optional; a missing component = not calibrated → the estimator's own small-sample treatment):

    g1:
      noise_priors:
        end_crossing:  {static_var_m2: a, timing_var_s2: b, dynamic_var_s4_m2: c, nu0: ν0}   # σ_ζ (end_faces.fuse_end_faces)
        end_reading:   {static_var_m2: …, timing_var_s2: …, dynamic_var_s4_m2: …, nu0: …}   # τ
        leg_offset_w:  {static_var_m2: …, timing_var_s2: …, nu0: …}    # σ_η,w (seam_estimate line)
        leg_offset_n:  {static_var_m2: …, timing_var_s2: …, nu0: …}    # σ_η,n
        speed_range_mps: [lo, hi]      # the measurement speeds the calibration runs were made at (their commanded speeds, run logs)
        speed_range_source: text       # where the range came from
        source / date / runs / method: provenance (not read by the estimator)

Calibrated speed range (user 2026-10-13: the measurement speed is chosen per run, inside the calibrated range): s0²(v) is fitted
over the calibration runs' speeds; a run override of the measurement speed (client --measure-speed, the AutoTeach goal's
measure_speed_mps, a job.speed_mps override) is accepted inside [lo, hi] only (check_measure_speed; the 1G core's gates, before any
motion). The range is data of the calibration (scan_l0.calib_noise_priors writes it from the runs it fitted), not a setting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["VarPrior", "parse_noise_priors", "COMPONENTS", "refine_bounds", "calibrated_speed_range", "check_measure_speed"]

COMPONENTS = ("end_crossing", "end_reading", "leg_offset_w", "leg_offset_n")


@dataclass(frozen=True)
class VarPrior:
    """One calibrated variance component: s0²(v) = a + b·v² + c·v⁴, λ ~ Scaled-Inv-χ²(ν0, 1)."""
    a_m2: float
    b_s2: float
    nu0: float
    c_s4_m2: float = 0.0

    def s2(self, v_mps) -> np.ndarray:
        """The population scale s0²(v) [m²] at the speed(s) v [m/s] (|v|)."""
        v2 = np.abs(np.asarray(v_mps, dtype=float)) ** 2
        return self.a_m2 + self.b_s2 * v2 + self.c_s4_m2 * v2 * v2

    def log_density_log_sigma(self, sigma, v_mps) -> np.ndarray:
        """log prior density of ln σ (up to a constant) for σ² ~ Scaled-Inv-χ²(ν0, s0²(v)): in x = σ²,
        p(x) ∝ x^−(ν0/2+1) exp(−ν0 s0²/(2x)); per ln σ (Jacobian 2x): ∝ x^−ν0/2 exp(−ν0 s0²/(2x))."""
        x = np.asarray(sigma, dtype=float) ** 2
        s2 = self.s2(v_mps)
        with np.errstate(divide="ignore", over="ignore"):
            return -0.5 * self.nu0 * np.log(x) - 0.5 * self.nu0 * s2 / x

    def quantile_sigma(self, p: float, v_mps: float) -> float:
        """σ at the prior probability p (Scaled-Inv-χ²: σ² = ν0 s0² / χ²_ν0)."""
        from scipy.stats import chi2
        return math.sqrt(self.nu0 * float(self.s2(v_mps)) / float(chi2.ppf(1.0 - p, self.nu0)))

    def to_dict(self) -> dict:
        return {"static_var_m2": float(self.a_m2), "timing_var_s2": float(self.b_s2), "dynamic_var_s4_m2": float(self.c_s4_m2),
                "nu0": float(self.nu0)}


def _one(d) -> Optional[VarPrior]:
    if not isinstance(d, dict):
        return None
    try:
        a, b, nu = float(d.get("static_var_m2", 0.0) or 0.0), float(d.get("timing_var_s2", 0.0) or 0.0), float(d["nu0"])
        c = float(d.get("dynamic_var_s4_m2", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    if not (all(math.isfinite(x) and x >= 0.0 for x in (a, b, c)) and math.isfinite(nu) and (a + b + c) > 0.0 and nu > 0.0):
        return None
    return VarPrior(a_m2=a, b_s2=b, nu0=nu, c_s4_m2=c)


def parse_noise_priors(block) -> dict:
    """Calib 'g1: noise_priors:' → {component: VarPrior} (only the components the block states completely)."""
    out = {}
    if not isinstance(block, dict):
        return out
    for c in COMPONENTS:
        p = _one(block.get(c))
        if p is not None:
            out[c] = p
    return out


def refine_bounds(grid: np.ndarray, weights: np.ndarray, rel: float = 1e-12) -> tuple:
    """Bounds of a 1-D grid (uniform in ln σ) that hold the posterior: the nodes whose weight exceeds rel·max ([R] numeric
    resolution of the quadrature), widened by one node on each side (the mass between the coarse nodes)."""
    w = np.asarray(weights, dtype=float)
    on = np.flatnonzero(w > rel * float(np.max(w)))
    i0, i1 = max(int(on[0]) - 1, 0), min(int(on[-1]) + 1, grid.size - 1)
    return float(grid[i0]), float(grid[i1])


def calibrated_speed_range(block) -> Optional[dict]:
    """Calib 'g1: noise_priors:' → {lo_mps, hi_mps, source, n_runs}: the measurement speeds the speed model was calibrated over
    (speed_range_mps, written by scan_l0.calib_noise_priors from the calibration runs' commanded speeds) and where they came from
    (speed_range_source). None = the block states no range. ValueError for a malformed range."""
    if not isinstance(block, dict) or block.get("speed_range_mps") is None:
        return None
    r = block.get("speed_range_mps")
    try:
        lo, hi = (float(x) for x in r)
    except (TypeError, ValueError):
        raise ValueError(f"noise_priors.speed_range_mps must be [lo, hi] in m/s, got {r!r}") from None
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo <= hi):
        raise ValueError(f"noise_priors.speed_range_mps must be 0 < lo <= hi, got {r!r}")
    src = str(block.get("speed_range_source") or "")
    if not src:
        src = f"the calibration runs of {block.get('source') or 'the noise priors'} ({block.get('date') or 'no date'})"
    return {"lo_mps": lo, "hi_mps": hi, "source": src, "n_runs": len(block.get("runs") or [])}


def check_measure_speed(v_mps: float, block, *, what: str = "measurement speed", calib_source: str = "") -> dict:
    """A run's measurement speed [m/s] chosen at run time against the calibrated range of the calib's noise priors → the record
    {speed_mps, lo_mps, hi_mps, source, n_runs, calib}. ValueError outside the range, or when the calibration states no range (the
    speed model would be used beyond what it was fitted on - the job file's own speed is not a run-time choice and is not checked
    here). The bounds are the recorded speeds themselves (equal = inside; float representation only)."""
    v = float(v_mps)
    cal = f" (calibration {calib_source})" if calib_source else ""
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{what} must be a finite speed > 0, got {v_mps!r}")
    rng = calibrated_speed_range(block)
    if rng is None:
        raise ValueError(f"{what} {v * 1e3:g} mm/s: the calibration{cal} states no calibrated speed range (g1 noise_priors "
                         "speed_range_mps, scan_l0.calib_noise_priors) - a run-time measurement speed needs it; without one the "
                         "job file's speed_mps runs as it is")
    lo, hi = rng["lo_mps"], rng["hi_mps"]
    inside = (v >= lo or math.isclose(v, lo, rel_tol=1e-12)) and (v <= hi or math.isclose(v, hi, rel_tol=1e-12))
    if not inside:
        raise ValueError(f"{what} {v * 1e3:g} mm/s is outside the calibrated range {lo * 1e3:g}..{hi * 1e3:g} mm/s of the noise "
                         f"priors{cal} ({rng['source']})")
    return {"speed_mps": v, **rng, "calib": calib_source or None}
