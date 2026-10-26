"""KIRO seam estimate of a direct / local scan (no Hongik server, no robot motion).

Root (pass 1) wire tips = the groove's corner 1 (30 deg bevel x bottom, plane fits over the scanned length) at both
plate ends; passes 2.. = pass 1 + Hongik's planned wire-tip layout (template taken from a successful Hongik estimate,
seam frame per end). Written in the estimate layout the rest of the pipeline reads (<scan>/hongik_estimate:
estimate.json, kiro/pass_<k>.yaml, kiro/pass_<k>_info.yaml, calib_used.json), marked "estimator": "kiro".
"""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import yaml

from construct_robot.core.groove_estimate import (DEFAULT_SPEC, DEFAULT_TOL, LENGTH_AMBIGUOUS_MM, GrooveEstimateError,
                                                   estimate_groove, groove_qc)
from construct_robot.core.scan_reference import tcp_from_aim

ESTIMATE_DIR = "hongik_estimate"          # the estimate folder name every consumer reads (both estimators)
ENDS = ("start", "goal")


def _teaching_dir():
    from construct_robot.application.scan_correction import scan_teaching_dir
    return scan_teaching_dir()


def default_job_path():
    """Hongik's job template (scan_l0 jobs/groove_10T_1G.yaml) or None when scan_l0 is not installed."""
    try:
        import scan_l0
    except ImportError:
        return None
    return Path(scan_l0.__file__).resolve().parent / "jobs" / "groove_10T_1G.yaml"


def specimen_spec(job_path=None):
    """The specimen design from the Hongik job YAML ("specimen", "qc_targets"): (spec, tol, job file). spec holds
    thickness_mm / bevel_angle_deg / root_gap_mm / weld_length_mm (+ length_ambiguous_mm, bevel_side); the defaults
    (the same KIRO specimen) when the file is not readable."""
    spec = dict(DEFAULT_SPEC, length_ambiguous_mm=LENGTH_AMBIGUOUS_MM)
    tol = dict(DEFAULT_TOL)
    path = Path(job_path).expanduser() if job_path else default_job_path()
    try:
        doc = yaml.safe_load(path.read_text())
    except (AttributeError, OSError, yaml.YAMLError):
        return spec, tol, None
    specimen, qc = doc.get("specimen") or {}, doc.get("qc_targets") or {}
    for key in DEFAULT_SPEC:
        if specimen.get(key) is not None:
            spec[key] = float(specimen[key])
    if specimen.get("bevel_side"):
        spec["bevel_side"] = str(specimen["bevel_side"])
    for key in tol:
        if isinstance(qc.get(key), dict) and qc[key].get("tol") is not None:
            tol[key] = float(qc[key]["tol"])
    if isinstance(qc.get("weld_length_mm"), dict) and qc["weld_length_mm"].get("ambiguous") is not None:
        spec["length_ambiguous_mm"] = float(qc["weld_length_mm"]["ambiguous"])
    return spec, tol, str(path)


def check_estimate_length(scan_dir, spec=None):
    """Refuse an estimate (either estimator) whose seam length is off the specimen's by more than the ambiguous
    limit: an end was misread, and targets / a correction from it would carry that error. Estimates without both
    ends are not checked. Returns the length [mm] or None."""
    est = json.loads((Path(scan_dir).expanduser() / ESTIMATE_DIR / "estimate.json").read_text())
    ends = est.get("ends") or {}
    if not all(isinstance(ends.get(k), dict) and ends[k].get("point_m") is not None for k in ("S", "E")):
        return None
    spec = spec or specimen_spec()[0]
    length = float(np.linalg.norm(np.subtract(ends["E"]["point_m"], ends["S"]["point_m"]))) * 1e3
    limit = float(spec.get("length_ambiguous_mm", LENGTH_AMBIGUOUS_MM))
    if abs(length - spec["weld_length_mm"]) > limit:
        raise ValueError(f"the estimate's seam is {length:.1f} mm long, the specimen {spec['weld_length_mm']:.0f} mm "
                         f"(more than {limit:.0f} mm off: a plate end was misread) - rescan passing both plate ends")
    return length


def template_path():
    return _teaching_dir() / "hongik_pass_template.yaml"


def _find(doc, key):
    if isinstance(doc, dict):
        if key in doc:
            return doc[key]
        for v in doc.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(doc, list):
        for v in doc:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def build_template_from_hongik(scan_dir, path=None):
    """Hongik's pass layout of a successful Hongik estimate: per pass and end the wire tip relative to pass 1
    (seam frame along / lateral / normal) and its TCP orientation, plus tcp_to_aim."""
    from construct_robot.application.scan_correction import load_estimate_detail
    out_dir = Path(scan_dir).expanduser() / ESTIMATE_DIR
    est = json.loads((out_dir / "estimate.json").read_text())
    if est.get("estimator") == "kiro":
        raise ValueError("the template must come from a Hongik estimate")
    detail = load_estimate_detail(scan_dir)
    frame, p1 = detail["frame"], detail["passes"][0]
    passes = {}
    for row in detail["passes"]:
        passes[int(row["k"])] = {end: {"aim_rel_seam_m": [float(v) for v in frame.to_local(row[f"aim_{end}"] - p1[f"aim_{end}"])],
                                       "orientation_xyzw": [float(v) for v in row[f"quat_{end}"]]} for end in ENDS}
    doc = {"format_version": 1, "source_scan": str(Path(scan_dir).expanduser()),
           "created": datetime.now().isoformat(timespec="seconds"),
           "tcp_to_aim_m": float(detail["passes"][0]["hongik_L_m"]), "bevel_sign_w": int(detail["bevel_sign_w"]),
           "passes": passes}
    out = Path(path or template_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return doc


def load_template(path=None):
    path = Path(path or template_path())
    if not path.is_file():
        raise ValueError(f"no Hongik pass template ({path}): run one Hongik estimate first")
    return yaml.safe_load(path.read_text())


def _pose_doc(tcp, quat):
    return {"position_m": dict(zip("xyz", (float(v) for v in tcp))),
            "orientation_xyzw": dict(zip("xyzw", (float(v) for v in quat)))}


def run_kiro_estimate(scan_dir, *, weld_direction="E->S", calib_path=None, log=print, template=None):
    """Estimate one scan and write the estimate files; returns a report like run_hongik_estimate's RESULT."""
    from construct_robot.core.scan_calib import calib_matrix, rebase_optical, rebase_points, same_calib
    from construct_robot.io.scan_calib import default_calib_path, load_calib_pose, recorded_calib
    scan_dir = Path(scan_dir).expanduser()
    out = scan_dir / ESTIMATE_DIR
    (out / "kiro").mkdir(parents=True, exist_ok=True)
    (out / "estimate.json").unlink(missing_ok=True)
    data = np.load(scan_dir / "scan.npz")
    points = np.asarray(data["points_world_m"], float)
    index = np.asarray(data["points_frame_index"], int)
    T = np.asarray(data["T_world_optical"], float)
    used = load_calib_pose(calib_path or default_calib_path())
    rec = recorded_calib(scan_dir)
    rebased = rec is not None and not same_calib(rec, used)
    if rebased:
        T_new = rebase_optical(T, calib_matrix(rec["xyz"], rec["rpy"]), calib_matrix(used["xyz"], used["rpy"]))
        points = rebase_points(points, index, T, T_new)
        T = T_new
        log(f"recording re-based to the calib {used['file']}")
    (out / "calib_used.json").write_text(json.dumps({"used": used, "recorded": rec, "rebased": bool(rebased)}, indent=1))
    spec, tol, job = specimen_spec()
    log(f"KIRO estimate: {len(points)} points, {len(T)} profiles - specimen {spec['thickness_mm']:.0f}T bevel "
        f"{spec['bevel_angle_deg']:.0f} deg gap {spec['root_gap_mm']:.1f} length {spec['weld_length_mm']:.0f} mm "
        f"({job or 'defaults'}) - fitting the groove...")
    g = estimate_groove(points, index, T[:, :3, 3], spec=spec)
    qc = groove_qc(g, spec, tol)
    log(f"groove: {g['sections']} sections, bevel side {g['bevel_side']}, scan line {g['scan_skew_deg']:.1f} deg off "
        "the seam")
    for row in qc:
        state = "not measured" if row["ok"] is None else "OK" if row["ok"] else "OUT OF TOLERANCE"
        measured = "-" if row["measured"] is None else f"{row['measured']:.2f}"
        log(f"  {row['name']}: {measured} (design {row['design']:.1f} +/- {row['tol']:.1f}) {state}")
    length_off = g["length_mm"] - spec["weld_length_mm"]
    if abs(length_off) > spec["length_ambiguous_mm"]:
        raise GrooveEstimateError(f"seam length {g['length_mm']:.1f} mm, the specimen is {spec['weld_length_mm']:.0f} mm"
                                  ": a plate end was not read - rescan passing both plate ends")
    template = template or load_template()
    L = float(template["tcp_to_aim_m"])
    u, w, n = g["u"], g["w"], g["n"]
    bevel_sign = -1 if g["bevel_side"] == "left" else 1
    # S = the end the scan visited first; KIRO welds E -> S by default (start = E, goal = S)
    ends_point = {"S": g["corner1_S"], "E": g["corner1_E"]}
    start_end, goal_end = ("E", "S") if weld_direction == "E->S" else ("S", "E")
    root = {"start": ends_point[start_end], "goal": ends_point[goal_end]}
    passes_report = []
    for k_str, ends in sorted(template["passes"].items(), key=lambda kv: int(kv[0])):
        k = int(k_str)
        aims, poses = {}, {}
        for end in ENDS:
            rel = np.asarray(ends[end]["aim_rel_seam_m"], float)
            aim = root[end] + rel[0] * u + rel[1] * w + rel[2] * n
            q = ends[end]["orientation_xyzw"]
            aims[end] = [float(v) for v in aim]
            poses[f"current_{end}"] = _pose_doc(tcp_from_aim(aim, q, L), q)
        (out / "kiro" / f"pass_{k}.yaml").write_text(yaml.safe_dump({"pass": k, **poses}, sort_keys=False))
        info = {"derived": {"role": "root" if k == 1 else "fill" if k < len(template["passes"]) else "cap",
                            "estimator": "kiro", "frames": {"tcp_frame": "right_manipulator_ee_point",
                                                            "tcp_to_aim_point_mm": L * 1e3, "aim_point_World_m": aims}}}
        (out / "kiro" / f"pass_{k}_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
        passes_report.append({"pass": k, "aim_start_mm": [v * 1e3 for v in aims["start"]],
                              "aim_goal_mm": [v * 1e3 for v in aims["goal"]]})
    estimate = {"estimator": "kiro", "u": [float(v) for v in u], "n_plate": [float(v) for v in n],
                "bevel_sign_w": bevel_sign, "weld_direction": weld_direction,
                "ends": {name: {"point_m": [float(v) for v in p], "sigma_m": g["end_resolution_mm"] * 1e-3,
                                "two_sided": True} for name, p in ends_point.items()},
                "groove": {k: (v if not isinstance(v, np.ndarray) else [float(x) for x in v]) for k, v in g.items()
                           if k not in ("u", "w", "n") and not k.startswith("corner") and k not in ("S", "E")},
                "corner2": {name: [float(v) for v in g[f"corner2_{name}"]] for name in ("S", "E") if f"corner2_{name}" in g},
                "template": template.get("source_scan"), "specimen": spec, "specimen_job": job, "qc": qc}
    (out / "estimate.json").write_text(json.dumps(estimate, indent=1, default=str))
    return {"status": "OK", "estimator": "kiro", "scan_dir": str(scan_dir), "out_dir": str(out),
            "ends": {name: {"point_mm": [v * 1e3 for v in p], "sigma_mm": g["end_resolution_mm"], "two_sided": True}
                     for name, p in ends_point.items()},
            "groove": estimate["groove"], "qc": qc, "passes": passes_report, "weld_direction": weld_direction}


# ---- centred scan traverse: the laser straight down over the groove centre, the line across the seam -----------

def _mat_quat(R):
    import math
    t = np.trace(R)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        return np.array([(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def centred_scan_poses(scan_dir, margin_m=0.030, standoff_m=None):
    """Scan start / end TCP poses (C / D) from a scan's estimate: optical axis along -plate normal through the groove
    centre (root corner + half the root gap toward the 90 deg face), laser line across the seam (sign kept), the same
    standoff as that scan (or standoff_m above the groove bottom), margin_m before the S end / past the E end.
    Returns {"C": (position, quat), "D": (position, quat), "standoff_m", "length_m"}."""
    from construct_robot.core.scan_reference import quat_matrix
    scan_dir = Path(scan_dir).expanduser()
    est = json.loads((scan_dir / ESTIMATE_DIR / "estimate.json").read_text())
    meta = json.loads((scan_dir / "scan.json").read_text())
    start = meta["scan_start_tcp"]
    start = start if isinstance(start, dict) else __import__("ast").literal_eval(start)
    T = np.asarray(np.load(scan_dir / "scan.npz")["T_world_optical"], float)
    u, n = np.asarray(est["u"], float), np.asarray(est["n_plate"], float)
    w = np.cross(n, u)
    S, E = (np.asarray(est["ends"][k]["point_m"], float) for k in ("S", "E"))
    length = float((E - S) @ u)
    T_tcp = np.eye(4)
    T_tcp[:3, :3] = quat_matrix(start["orientation_xyzw"])
    T_tcp[:3, 3] = start["position_m"]
    T_tcp_opt = np.linalg.inv(T_tcp) @ T[0]                       # the recording starts standing at C
    standoff = float((T[0][:3, 3] - S) @ n) if standoff_m is None else float(standoff_m)
    gap = float((est.get("groove") or {}).get("root_gap_mm") or 5.0) * 1e-3
    toward_vertical = 1.0 if int(est.get("bevel_sign_w", -1)) < 0 else -1.0   # bevel on -w -> 90 deg face on +w
    x = w if T[0][:3, 0] @ w > 0 else -w
    z = -n
    R = np.c_[x, np.cross(z, x), z]
    poses = {}
    for name, a in (("C", -margin_m), ("D", length + margin_m)):
        T_opt = np.eye(4)
        T_opt[:3, :3] = R
        T_opt[:3, 3] = S + u * a + w * (toward_vertical * gap / 2.0) + n * standoff
        T_new = T_opt @ np.linalg.inv(T_tcp_opt)
        poses[name] = (T_new[:3, 3], _mat_quat(T_new[:3, :3]))
    poses.update(standoff_m=standoff, length_m=length)
    return poses
