"""Edge-response calibration of the line sensor for the 1G end model (scan_core.end_faces): the laser-sheet spread σ_L along the scan
(sheet thickness ⊗ camera blur) and the triangulation angle θ, from the plate top edges of finished 1G runs on a specimen with square
plate ends (the design; no measured length or reference value enters).

For every run directory (scan_data/hongik_auto_teach/<run>) the run's own prior (verification/run_log.json) and legs are re-estimated
with the edge response NOT calibrated, which yields the top-edge data of every plate end crossed (seam_estimate.estimate
edge_groups_out); end_faces.pool_edge_response pools them: joint mode over all runs, σ from the replication over legs (Student-t
predictive variance), one-way ANOVA of the leg estimates by run.

    python3 -m scan_l0.calib_edge_response RUN_DIR [RUN_DIR ...] [--json OUT]

Prints the calibration block for the calib file ('g1:' → 'edge_response:'); writing the calib file is a calibration change and is done
by the operator after review."""
from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import sys

import numpy as np


def _prior_of(run_dir: str):
    from scan_core.g1_types import SeamPrior, Tolerances
    from scan_core.specimen import GrooveSpec
    rl = json.load(open(os.path.join(run_dir, "verification", "run_log.json"), encoding="utf-8"))
    p = rl["prior"]
    spec = p["spec"]
    spec = ast.literal_eval(spec) if isinstance(spec, str) else spec
    nan = float("nan")

    def f(k):
        v = p.get(k)
        return nan if v is None else float(v)
    return SeamPrior(S0=np.asarray(p["S0"]), E0=np.asarray(p["E0"]), up=np.asarray(p["up"]), spec=GrooveSpec.from_dict(spec),
                     tol=Tolerances(**p["tol"]), k=float(p["k"]), dt_s=f("dt_s"), dt_sigma_s=f("dt_sigma_s"),
                     rep_sigma_m=f("rep_sigma_m"), he_sigma_m=f("he_sigma_m"), sigma_eps=f("sigma_eps"),
                     station_step_m=float(p["station_step_m"]), station_window_m=float(p["station_window_m"]),
                     bevel_sign_w=p.get("bevel_sign_w"), edge_response=None), rl


def run_groups(run_dir: str) -> tuple:
    """(top-edge data of the run's plate ends, job pitch [deg])."""
    from scan_core.g1_types import LegObs
    from scan_core import seam_estimate as se
    prior, rl = _prior_of(run_dir)
    legs = [LegObs.load(d + "profiles.npz") for d in sorted(glob.glob(os.path.join(run_dir, "verification", "legs", "*/")))]
    groups: list = []
    se.estimate(legs, prior, edge_groups_out=groups)
    return groups, (rl.get("job_rough") or {}).get("pitch_deg")


def main(argv=None) -> int:
    from scan_core import end_faces as ef
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    runs, pitch = {}, {}
    for d in a.runs:
        nm = os.path.basename(os.path.normpath(d))
        runs[nm], pitch[nm] = run_groups(d)
        print(f"{nm}: pitch {pitch[nm]} deg, {len(runs[nm])} plate-end readings", file=sys.stderr)
    if len(set(pitch.values())) > 1:
        print(f"note: the runs have different pass pitches {sorted(set(map(str, pitch.values())))}", file=sys.stderr)
    out = ef.pool_edge_response(runs)
    out["pitch_deg"] = pitch
    for q in out["per_leg"]:
        print(f"  {q['run']:12s} {q['leg']:8s} readings {q['n_readings']}  sigma_L {math.exp(q['log_sigma_L']) * 1e3:.3f} mm"
              f"  theta {math.degrees(q['theta_rad']):.2f} deg", file=sys.stderr)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
    print("  edge_response:")
    print(f"    sigma_l_m: {out['sigma_l_m']:.6f}")
    print(f"    sigma_l_log_sd: {out['sigma_l_log_sd']:.4f}")
    print(f"    theta_tri_deg: {out['theta_tri_deg']:.3f}")
    print(f"    theta_tri_sigma_deg: {out['theta_tri_sigma_deg']:.3f}")
    print(f"    source: 'pooled {len(runs)} runs ({', '.join(runs)}), {out['dof'] + 1} legs'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
