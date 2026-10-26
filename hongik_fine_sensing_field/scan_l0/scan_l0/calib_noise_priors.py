"""Noise-prior calibration of the 1G estimator from saved runs (calib 'g1: noise_priors:'; consumed by scan_core.noise_priors).

Per run (scan_data/hongik_auto_teach/<run>: verification/run_log.json prior, verification/legs/*) the estimator is run with NO noise
priors; its restricted likelihoods of the variance components are taken on a common grid of σ: the plate-end fusion's (σ_ζ the
per-crossing along-track offset, τ the per-reading scatter; end_faces.vc_loglik) and the line's leg-offset scatter (σ_η,w, σ_η,n;
seam_estimate.line_eta_loglik), each at the run's speed. Across runs, each component is a draw from the robot + sensor population

    σ²_run = λ·s0²(v),  s0²(v) = a + b·v² + c·v⁴,  λ ~ Scaled-Inv-χ²(ν0, 1)

(static, timing × speed, dynamics ∝ v²) whose hyperparameters (a, b, c ≥ 0, ν0) maximise the marginal likelihood of all runs (type-II maximum likelihood / empirical Bayes; the
end components jointly). ν0 is then combined with the information the calibration runs carry about the population scale
(ν_eff = 1 / (1/ν0 + 1/Σν_run), Σν_run = the runs' total redundant degrees of freedom): a population that looks homogeneous is
not known better than the pooled runs determine it.

    python3 -m scan_l0.calib_noise_priors RUN_DIR [RUN_DIR ...] [--write CALIB_YAML] [--json OUT]

Prints the 'noise_priors:' block (with provenance: runs, date, method, and the calibrated speed range speed_range_mps = the
lowest / highest commanded measurement speed of the runs, from their run logs - a run-time measurement speed must lie in it,
scan_core.noise_priors.check_measure_speed); --write replaces (or adds) that block under 'g1:' in the calib yaml, keeping the rest
of the file as it is."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import sys

import numpy as np

GRID = np.exp(np.linspace(math.log(1e-6), math.log(1e-2), 81))     # [R] σ grid of the per-run likelihoods (1 µm … 10 mm)
REFINE = 4                 # [R] nodes per GRID step of the interpolated likelihood in the marginal (ln σ step 0.115 / 4 = 0.029)
END_COMPONENTS = ("end_crossing", "end_reading")
LINE_COMPONENTS = ("leg_offset_w", "leg_offset_n")


def run_evidence(run_dir: str, edge_response: dict | None = None) -> dict:
    """One run → its restricted log-likelihoods on GRID and their speeds / degrees of freedom. edge_response: the sensor's edge
    response to read the plate ends with (the calib file's, as the estimator will use it); None = the run's own (run_log prior)."""
    import glob
    from scan_core import end_faces as ef
    from scan_core import seam_estimate as se
    from scan_core.g1_types import LegObs
    from .calib_edge_response import _prior_of
    prior, rl = _prior_of(run_dir)
    er = edge_response if edge_response is not None else (rl.get("prior") or {}).get("edge_response")
    prior.edge_response = er if isinstance(er, dict) else None
    prior.noise_priors = None
    legs = [LegObs.load(d + "profiles.npz") for d in sorted(glob.glob(os.path.join(run_dir, "verification", "legs", "*/")))]
    vc: dict = {}
    se.estimate(legs, prior, vc_out=vc)
    out = {"run": os.path.basename(os.path.normpath(run_dir)), "commanded_speed_mps": run_commanded_speed(run_dir)}
    end = vc.get("end") or {}
    if end.get("A") is not None and end["A"].shape[0]:
        out["end"] = {"ll": ef.vc_loglik(end, GRID, GRID), "v_mps": float(end["v_ref"]), "nu": [int(end["nu"]), int(end["nu_tau"])],
                      "n_readings": int(end["A"].shape[0])}
    lv = vc.get("line") or {}
    if lv.get("reml"):
        out["line"] = {"ll": [se.line_eta_loglik(lv, c, GRID) for c in (0, 1)], "v_mps": float(lv["v_ref"]),
                       "nu": [int(x or 0) for x in lv["nu"]]}
    return out


def run_commanded_speed(run_dir: str) -> float | None:
    """The run's commanded measurement speed [m/s] (its run log: measure_speed.speed_mps, else job_rough.speed_mps); None = none."""
    p = os.path.join(run_dir, "verification", "run_log.json")
    try:
        with open(p, encoding="utf-8") as fh:
            rl = json.load(fh)
    except (OSError, ValueError):
        return None
    for v in (((rl.get("measure_speed") or {}).get("speed_mps")), ((rl.get("job_rough") or {}).get("speed_mps"))):
        if v is not None and math.isfinite(float(v)) and float(v) > 0.0:
            return float(v)
    return None


def speed_range(evidence: list) -> tuple:
    """(lo, hi, source text) of the runs' commanded measurement speeds; (None, None, why) when no run has one."""
    sp = [(float(e["commanded_speed_mps"]), e["run"]) for e in evidence if e.get("commanded_speed_mps") is not None]
    if not sp:
        return None, None, "no run log with a measurement speed"
    lo, hi = min(sp), max(sp)
    n_lo = sum(1 for v, _ in sp if v == lo[0])
    n_hi = sum(1 for v, _ in sp if v == hi[0])
    return lo[0], hi[0], (f"the commanded measurement speeds of the {len(sp)} calibration runs (run_log measure_speed / job_rough "
                          f"speed_mps): lowest {lo[0] * 1e3:g} mm/s ({n_lo} run(s), e.g. {lo[1]}), highest {hi[0] * 1e3:g} mm/s "
                          f"({n_hi} run(s), e.g. {hi[1]})")


def _fine_axis() -> tuple:
    """Fine ln σ nodes (REFINE per GRID step) and the edges of the bin around each node; the end bins take the tails (below the
    grid the per-run likelihood is at its plateau — the component vanishes into the readings' own noise —, above it it is ~0)."""
    lg = np.log(GRID)
    h = float(lg[1] - lg[0]) / REFINE
    lf = lg[0] + h * np.arange((lg.size - 1) * REFINE + 1)
    edges = np.r_[-np.inf, 0.5 * (lf[1:] + lf[:-1]), np.inf]
    return lf, edges


def _log_mass(edges: np.ndarray, a: float, b: float, c: float, nu: float, v: float) -> np.ndarray:
    """log prior probability of each ln σ bin for σ² ~ Scaled-Inv-χ²(ν, s0²(v)), s0² = a + b v² + c v⁴, from the exact CDF
    P(σ ≤ x) = Q(ν/2, ν s0²/(2x²)) (regularised upper incomplete gamma) — the lower / upper tail form where each is accurate."""
    from scipy.special import gammainc, gammaincc
    s2 = a + b * v * v + c * v ** 4
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        t = 0.5 * nu * s2 / np.exp(2.0 * edges)               # edges −∞ … +∞ → t +∞ … 0
        F = gammaincc(0.5 * nu, t)                            # CDF at the edges (accurate where small)
        S = gammainc(0.5 * nu, t)                             # 1 − CDF (accurate where small)
    m = np.where(F[1:] < 0.5, F[1:] - F[:-1], S[:-1] - S[1:])
    with np.errstate(divide="ignore"):
        return np.log(np.where(m > 0.0, m, 0.0))              # a difference of one monotone sequence: ≥ 0 but for round-off


def _fit(lls: list, speeds: list, dims: int) -> tuple:
    """Type-II ML of (a, b, c, ν0) per component (dims = 1: one component; 2: two independent components, joint 2-D likelihoods).
    → (list of (a, b, c, ν0) per component, max log marginal likelihood).

    Each run's marginal ∫ L(σ) p(σ | a, b, c, ν0) is the sum over bins of ln σ of the run's likelihood (cubic interpolation of its
    GRID values to REFINE nodes per step — a restricted likelihood of a variance component is smooth in ln σ) times the prior's
    exact probability of the bin (_log_mass). The population can be far narrower than the grid step (ν0 in the thousands: width
    1/√(2ν0) ≈ 0.01–0.02 in ln σ against a step of 0.115): a Riemann sum of the prior DENSITY on the grid then made the marginal
    spike wherever the population's centre met a grid node, and the optimiser stopped at one of these spikes (2026-10-12: the
    25-run fit's σ_η,n differed between two fits of nearly the same evidence by 2× and neither was the optimum — 8 and 77 log-units
    below it)."""
    from scipy.interpolate import CubicSpline, RectBivariateSpline
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    lg = np.log(GRID)
    lf, edges = _fine_axis()
    if dims == 1:
        llf = [CubicSpline(lg, np.asarray(ll, dtype=float))(lf) for ll in lls]
    else:
        llf = [RectBivariateSpline(lg, lg, np.asarray(ll, dtype=float), kx=3, ky=3)(lf, lf) for ll in lls]

    def unpack(th):
        return [tuple(math.exp(x) for x in th[4 * c:4 * c + 4]) for c in range(dims)]

    def nlml(th):
        pars = unpack(th)
        tot = 0.0
        for ll, v in zip(llf, speeds):
            if dims == 1:
                tot += float(logsumexp(ll + _log_mass(edges, *pars[0], v)))
            else:
                lm0, lm1 = _log_mass(edges, *pars[0], v), _log_mass(edges, *pars[1], v)
                tot += float(logsumexp(ll + lm0[:, None] + lm1[None, :]))
        return -tot
    # starts: each run's likelihood mode as the scale, both speed parts, ν0 from small to large
    best = None
    bnd = []
    for _ in range(dims):
        bnd += [(math.log(1e-16), math.log(1e-4)), (math.log(1e-14), math.log(1e-2)), (math.log(1e-14), math.log(1e1)),
                (math.log(0.5), math.log(1e6))]
    for s_start in (1e-5, 1e-4, 3e-4):
        for nu_start in (2.0, 20.0, 200.0):
            th0 = []
            for _ in range(dims):
                th0 += [math.log(s_start ** 2 / 3), math.log(s_start ** 2 / 3 / 0.05 ** 2), math.log(s_start ** 2 / 3 / 0.05 ** 4),
                        math.log(nu_start)]
            r = minimize(nlml, np.array(th0), method="L-BFGS-B", bounds=bnd)
            if best is None or r.fun < best.fun:
                best = r
    return unpack(best.x), -float(best.fun)


def calibrate(evidence: list) -> dict:
    """Hyperparameters of the four components from the runs' evidence; ν0 combined with the runs' total degrees of freedom."""
    out = {}
    ends = [e for e in evidence if "end" in e]
    if ends:
        pars, lml = _fit([e["end"]["ll"] for e in ends], [e["end"]["v_mps"] for e in ends], 2)
        for c, nm in enumerate(END_COMPONENTS):
            nu_sum = float(sum(e["end"]["nu"][c] for e in ends))
            a, b, cc, nu0 = pars[c]
            out[nm] = {"static_var_m2": a, "timing_var_s2": b, "dynamic_var_s4_m2": cc,
                       "nu0": 1.0 / (1.0 / nu0 + 1.0 / max(nu_sum, 1e-9)), "_nu0_between": nu0, "_nu_runs": nu_sum, "_lml": lml}
    lines = [e for e in evidence if "line" in e]
    for c, nm in enumerate(LINE_COMPONENTS):
        rows = [e for e in lines if np.all(np.isfinite(e["line"]["ll"][c]))]
        if not rows:
            continue
        pars, lml = _fit([e["line"]["ll"][c] for e in rows], [e["line"]["v_mps"] for e in rows], 1)
        nu_sum = float(sum(e["line"]["nu"][c] for e in rows))
        a, b, cc, nu0 = pars[0]
        out[nm] = {"static_var_m2": a, "timing_var_s2": b, "dynamic_var_s4_m2": cc,
                   "nu0": 1.0 / (1.0 / nu0 + 1.0 / max(nu_sum, 1e-9)), "_nu0_between": nu0, "_nu_runs": nu_sum, "_lml": lml}
    return out


def block_text(cal: dict, runs: list, speeds: tuple = (None, None, "")) -> str:
    """The calib yaml block (two-space indent under 'g1:'). speeds = speed_range(evidence): the calibrated speed range."""
    today = datetime.date.today().isoformat()
    L = ["  noise_priors:"]
    for nm in END_COMPONENTS + LINE_COMPONENTS:
        if nm not in cal:
            continue
        c = cal[nm]

        def sg(v, c=c):
            return math.sqrt(c["static_var_m2"] + c["timing_var_s2"] * v * v + c["dynamic_var_s4_m2"] * v ** 4) * 1e3
        L.append(f"    {nm}: {{static_var_m2: {c['static_var_m2']:.4e}, timing_var_s2: {c['timing_var_s2']:.4e}, "
                 f"dynamic_var_s4_m2: {c['dynamic_var_s4_m2']:.4e}, nu0: {c['nu0']:.2f}}}"
                 f"   # sigma at 15/20/30/40/60 mm/s: {'/'.join(f'{sg(v):.4f}' for v in (0.015, 0.02, 0.03, 0.04, 0.06))} mm")
    if speeds[0] is not None:
        L.append(f"    speed_range_mps: [{speeds[0]:.4g}, {speeds[1]:.4g}]")
        L.append(f"    speed_range_source: '{speeds[2]}'")
    L.append(f"    source: 'scan_l0.calib_noise_priors, {len(runs)} runs'")
    L.append(f"    runs: [{', '.join(runs)}]")
    L.append(f"    date: '{today}'")
    L.append("    method: 'type-II maximum likelihood of a scaled-inverse-chi2 population over runs (restricted likelihoods of the "
             "estimator without priors), s0^2(v) = static_var + timing_var*v^2 + dynamic_var*v^4, nu0 = 1/(1/nu_between + "
             "1/sum nu_runs)'")
    return "\n".join(L) + "\n"


def write_block(path: str, text: str) -> None:
    """Replace the 'noise_priors:' block under 'g1:' (or add it at the end of the g1 block); the rest of the file is kept."""
    src = open(path, encoding="utf-8").read().splitlines(keepends=True)
    i_g1 = next((i for i, ln in enumerate(src) if re.match(r"^g1:\s*(#.*)?$", ln)), None)
    if i_g1 is None:
        raise ValueError(f"{path}: no top-level 'g1:' block")
    # the g1 block ends at the first non-blank line that is not indented (a top-level key or comment)
    end = next((i for i in range(i_g1 + 1, len(src)) if src[i].strip() and not src[i].startswith(" ")), len(src))
    while end > i_g1 + 1 and not src[end - 1].strip():
        end -= 1
    i_np = next((i for i in range(i_g1 + 1, end) if re.match(r"^  noise_priors:\s*(#.*)?$", src[i])), None)
    if i_np is not None:
        j = i_np + 1
        while j < end and (src[j].startswith("    ") or not src[j].strip()):
            j += 1
        new = src[:i_np] + [text] + src[j:]
    else:
        new = src[:end] + [text] + src[end:]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("".join(new))
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (scan_data/hongik_auto_teach/<run>)")
    ap.add_argument("--write", default=None, metavar="CALIB_YAML", help="replace / add the g1 noise_priors block in this calib file")
    ap.add_argument("--json", default=None, metavar="OUT", help="also write the per-run evidence summary and the fit as JSON")
    ap.add_argument("--evidence", default=None, metavar="PICKLE",
                    help="cache of the runs' evidence: read the runs found there, add the others, write it back")
    ap.add_argument("--calib", default=None, metavar="CALIB_YAML",
                    help="read the sensor edge response from this calib file (default: the --write file, else each run's own)")
    a = ap.parse_args(argv)
    src = a.calib or a.write
    er = None
    if src:
        import yaml
        er = ((yaml.safe_load(open(src, encoding="utf-8")) or {}).get("g1") or {}).get("edge_response")
    ev = []
    cache = {}
    if a.evidence and os.path.exists(a.evidence):
        import pickle
        with open(a.evidence, "rb") as fh:
            cache = pickle.load(fh)
    for d in a.runs:
        key = os.path.abspath(os.path.normpath(d))
        e = cache.get(key) or run_evidence(d, er if isinstance(er, dict) else None)
        if "commanded_speed_mps" not in e:                            # a cache written before the speed range was recorded
            e = {**e, "commanded_speed_mps": run_commanded_speed(d)}
        cache[key] = e
        ev.append(e)
        lg = np.log(GRID)
        msg = [e["run"]]
        if "end" in e:
            ll = e["end"]["ll"]
            i, j = np.unravel_index(int(np.argmax(ll)), ll.shape)
            msg.append(f"ends: v {e['end']['v_mps'] * 1e3:.0f} mm/s, nu {e['end']['nu']}, REML sigma_zeta {GRID[i] * 1e3:.3f} mm, "
                       f"tau {GRID[j] * 1e3:.3f} mm")
        if "line" in e:
            md = [GRID[int(np.nanargmax(x))] * 1e3 if np.any(np.isfinite(x)) else float("nan") for x in e["line"]["ll"]]
            msg.append(f"legs: v {e['line']['v_mps'] * 1e3:.0f} mm/s, nu {e['line']['nu']}, REML sigma_eta w/n {md[0]:.4f}/{md[1]:.4f} mm")
        del lg
        print("; ".join(msg), file=sys.stderr, flush=True)
    if a.evidence:
        import pickle
        with open(a.evidence, "wb") as fh:
            pickle.dump(cache, fh)
    cal = calibrate(ev)
    runs = [e["run"] for e in ev]
    txt = block_text(cal, runs, speed_range(ev))
    print(txt, end="")
    for nm, c in cal.items():
        print(f"  {nm}: nu0 between runs {c['_nu0_between']:.1f}, runs' total dof {c['_nu_runs']:.0f} -> nu0 {c['nu0']:.2f}",
              file=sys.stderr)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"fit": cal, "runs": [{k: (v if k != "end" and k != "line" else {kk: vv for kk, vv in v.items() if kk != "ll"})
                                             for k, v in e.items()} for e in ev]}, fh, indent=1, default=float)
    if a.write:
        write_block(a.write, txt)
        print(f"wrote the noise_priors block into {a.write}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
