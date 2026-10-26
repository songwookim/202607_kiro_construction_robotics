"""Fix the profiler hand-eye translation from measured root-pass wire-tip positions (no robot motion, no sensor).

For each (scan, references) pair: the measured root start / goal TCPs (operator: wire tip on the root corner = where
the 30 deg bevel face meets the bottom plane) give wire tips aim = TCP + L z; Hongik's root aim points of that scan's
estimate are the same corners as the scan sees them. Their World difference, expressed in laser_link (the scan keeps
one orientation), is the error of the calib translation. The mean over all samples is added to the calib xyz and
written as KIRO's calib file (scan_teaching/eoat_kiro_handeye.yaml); the rotation is kept (one scan orientation does
not observe it). Pairs must be measured on the specimen as that scan saw it.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from construct_robot.application.scan_correction import ENDS, load_estimate_detail, load_references, match_references
from construct_robot.core.scan_calib import calib_matrix, handeye_translation_fix, laser_rotation, same_calib
from construct_robot.core.scan_reference import aim_from_tcp
from construct_robot.io.scan_calib import kiro_calib_path, recorded_calib, write_handeye_calib


def handeye_samples(scan_dir, references=None):
    """[(R_world_laser, D_world)] of the root ends of one scan, and its info for the report."""
    detail = load_estimate_detail(scan_dir)
    recorded = recorded_calib(scan_dir)
    if recorded is None or detail["calib"] is None:
        raise ValueError(f"{scan_dir}: the calib of the recording is unknown")
    T = np.load(Path(scan_dir).expanduser() / "scan.npz")["T_world_optical"]
    R_wl, spread = laser_rotation(T, calib_matrix(recorded["xyz"], recorded["rpy"]))
    if spread > 1.0:
        raise ValueError(f"{scan_dir}: the sensor orientation changes by {spread:.1f} deg during the scan")
    p1 = detail["passes"][0]
    matched, notes = match_references(load_references() if references is None else references, detail)
    samples, rows = [], []
    for end in ENDS:
        ref = matched.get((1, end))
        if ref is None:
            raise ValueError(f"{scan_dir}: no root {end} reference")
        D = aim_from_tcp(ref["position"], ref["quat"], p1["L_m"]) - p1[f"aim_{end}"]
        samples.append((R_wl, D))
        rows.append({"scan": Path(scan_dir).name, "end": end, "world_mm": (D * 1e3).round(2).tolist(),
                     "laser_mm": (R_wl.T @ D * 1e3).round(2).tolist()})
    return samples, {"calib": detail["calib"], "rows": rows, "notes": notes, "orientation_spread_deg": spread}


def compute_handeye(pairs, *, write=True, out_path=None, max_std_m=0.003):
    """pairs: [(scan_dir, references or None = the current reference file)]. Returns the report; writes KIRO's
    calib file (base = the calib the estimates used) when write."""
    samples, rows, base = [], [], None
    for scan_dir, references in pairs:
        s, info = handeye_samples(scan_dir, references)
        if base is not None and not same_calib(base, info["calib"], tol=1e-7):
            raise ValueError("the scans' estimates use different calibs: re-run them with one calib first")
        base = info["calib"]
        samples += s
        rows += info["rows"]
    fix = handeye_translation_fix(samples)
    if float(np.max(fix["std_m"])) > max_std_m:
        raise ValueError(f"the samples disagree by {np.round(fix['std_m'] * 1e3, 1).tolist()} mm (std): check that "
                         "each reference set belongs to its scan (specimen not moved)")
    xyz = (np.asarray(base["xyz"], float) + fix["mean_m"]).tolist()
    report = {"base_calib": base, "rows": rows, "fix_laser_mm": (fix["mean_m"] * 1e3).round(2).tolist(),
              "std_laser_mm": (fix["std_m"] * 1e3).round(2).tolist(), "xyz_new_m": xyz, "file": None}
    if write:
        notes = [f"KIRO on-site hand-eye fix ({datetime.now().isoformat(timespec='seconds')}, construct_robot "
                 "application/scan_handeye.py): only xyz differs from the base file below; rpy is not observed.",
                 f"base: {base['file']} xyz {[round(v, 6) for v in base['xyz']]}",
                 f"fix (laser_link) {report['fix_laser_mm']} mm, std {report['std_laser_mm']} mm over {len(rows)} "
                 "root-end samples (measured wire tip on the root corner - Hongik's root aim of the scan):"]
        notes += [f"  {r['scan']} {r['end']}: world {r['world_mm']} mm, laser {r['laser_mm']} mm" for r in rows]
        notes += ["After changing this file: re-run the estimate (it re-bases older recordings) and Compute correction."]
        report["file"] = str(write_handeye_calib(base["file"], xyz, notes, out_path or kiro_calib_path()))
    return report
