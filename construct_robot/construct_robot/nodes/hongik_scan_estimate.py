"""KIRO direct / local scan (scan.npz) -> Hongik 1G seam estimate -> weld plan -> KIRO pass start / goal poses.

Runs Hongik's own functions offline (scan_core: fit_leg, seam_estimate.estimate, g1_outputs.write_outputs,
kiro_plan.write_kiro_outputs) on one recorded line. Needs the Hongik Python environment (numpy 1.21 / scipy 1.8):
the GUI starts it as a separate process (see HONGIK_ENV). No robot motion, no sensor access.

    python3 -m construct_robot.nodes.hongik_scan_estimate <scan_dir> [--job JOB] [--calib CALIB] ...

Writes <scan_dir>/hongik_estimate/ (legs/, estimate.json, sections.*, weld_plan.*, kiro/pass_<k>.yaml) and prints
one line "RESULT <json>": ends S / E (point, sigma, two-sided), the pass start / goal TCP poses, or the reason.
Passes are split along the recording's own travel axis. The rough seam points S0 / E0 (Hongik's taught points: origin
and direction of its seam model) come from the scan's groove root at both ends (S = the end visited first);
--rough ab falls back to the A / B teaching only when the scan shows no groove.
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Hongik's packages are built against the system numpy / scipy; /usr/local numpy 2 breaks scipy.
HONGIK_ENV = {"PYTHONNOUSERSITE": "1", "PYTHONPATH_SUFFIX": "/usr/lib/python3/dist-packages"}


def hongik_env(base=None):
    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = HONGIK_ENV["PYTHONNOUSERSITE"]
    # fork_map children without BLAS thread pools (as Hongik's g1_worker): oversubscribed BLAS threads made a
    # 1600-profile fit take > 6 min here (sys time 2x user time).
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + HONGIK_ENV["PYTHONPATH_SUFFIX"]
    return env


def _share(package, *parts):
    from ament_index_python.packages import get_package_share_directory
    return str(Path(get_package_share_directory(package), *parts))


def _defaults():
    from construct_robot.io.scan_calib import default_calib_path
    from construct_robot.io.teaching_yaml import teaching_config_dir
    return {
        "platform_config": _share("hongik_scan_ros", "config", "hongik_kiro.yaml"),
        "calib": default_calib_path() or _share("eoat_description", "calib", "eoat_kiro_01.yaml"),
        "urdf": _share("construct_description", "urdf_0528", "construct_robot_0528_hongik_v4.urdf"),
        "teaching_dir": str(Path(teaching_config_dir()) / "scan_teaching"),
    }


def _matrix(xyz, rpy):
    import numpy as np
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    T = np.eye(4)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]]
    T[:3, 3] = [float(v) for v in xyz]
    return T


def _fixed_chain(urdf_path, child, base):
    import numpy as np
    joints = {j.find("child").get("link"): j for j in ET.parse(urdf_path).getroot().findall("joint")}
    T = np.eye(4)
    while child != base:
        joint = joints[child]
        origin = joint.find("origin")
        T = _matrix(origin.get("xyz", "0 0 0").split(), origin.get("rpy", "0 0 0").split()) @ T
        child = joint.find("parent").get("link")
    return T


def split_passes(t, s, *, min_travel_m=0.02, window_s=0.2):
    """Frames of one recording -> passes [(indices, sign)]: contiguous moving frames with one travel sign along
    the seam axis (s [m] = position on the axis). Standstill (turnaround at the far end) is dropped. The speed is
    taken over window_s: frames arrive in bursts (receive stamps ~2 ms apart), a frame-to-frame slope is noise."""
    import numpy as np
    t, s = np.asarray(t, float), np.asarray(s, float)
    if len(t) < 3:
        return []
    order = np.argsort(t, kind="stable")
    ts, ss = t[order], s[order]
    v = np.empty_like(s)
    v[order] = (np.interp(ts + window_s / 2, ts, ss) - np.interp(ts - window_s / 2, ts, ss)) / window_s
    moving = np.abs(v) > 0.5 * np.percentile(np.abs(v), 90)
    sign = np.where(moving, np.sign(v), 0).astype(int)
    passes, start = [], 0
    for i in range(1, len(t) + 1):
        if i == len(t) or sign[i] != sign[start]:
            idx = np.arange(start, i)
            if sign[start] != 0 and abs(s[idx[-1]] - s[idx[0]]) >= min_travel_m:
                passes.append((idx, int(sign[start])))
            start = i
    return passes


def scan_axis(origins):
    """Principal axis of the sensor positions, oriented along the first motion (from the first frame to the
    farthest one)."""
    import numpy as np
    o = np.asarray(origins, float)
    _u, _s, vt = np.linalg.svd(o - o.mean(axis=0), full_matrices=False)
    axis = vt[0]
    far = o[int(np.argmax(np.linalg.norm(o - o[0], axis=1)))]
    return axis if float((far - o[0]) @ axis) >= 0.0 else -axis


def rough_points_from_legs(legs, origin, u, *, n_end=10, min_profiles=20):
    """S0 / E0 = mean groove root (gap-bottom centre) of the n_end profiles with groove evidence nearest each end
    of the scan axis. Returns (S0, E0, source) or (None, None, why)."""
    import numpy as np
    from scan_core.groove_model import root_points_base
    points = []
    for leg in legs:
        P, _cov = root_points_base(leg)
        ok = (np.asarray(leg.log_bf) > 0.0) & np.asarray(leg.converged, bool) & np.all(np.isfinite(P), axis=1)
        points.append(P[ok])
    P = np.concatenate(points) if points else np.empty((0, 3))
    if len(P) < min_profiles:
        return None, None, f"only {len(P)} profiles with groove evidence"
    order = np.argsort((P - origin) @ u)
    k = min(n_end, len(P) // 4)
    return P[order[:k]].mean(axis=0), P[order[-k:]].mean(axis=0), \
        f"scan: groove root of the {k} first / last groove profiles"


def _rough_point(teaching_dir, which):
    import numpy as np
    import yaml
    with open(Path(teaching_dir) / f"seam_{which}.yaml", encoding="utf-8") as stream:
        doc = yaml.safe_load(stream)
    return np.array(doc["capture_provenance"]["scan_point_world_m"], dtype=float)


WELD_DIRECTIONS = ("E->S", "S->E", "alternate")


def end_diagnostics(document):
    """Keep coarse-boundary detection separate from the delivered precision edge.

    Crossing a specimen boundary does not guarantee a usable top-edge reading;
    the refusal report must not imply that a missing precision edge means the
    scanner failed to travel past the specimen.
    """
    plate = document.get("plate") or {}
    faces = plate.get("end_faces") or {}
    readings = faces.get("readings") or []
    result = {}
    for name in ("S", "E"):
        coarse = ((plate.get("plate_ends") or {}).get(name) or {}).get("inner_own_evidence") or {}
        final = (document.get("ends") or {}).get(name) or {}
        result[name] = {"coarse_boundary_identified": coarse.get("two_sided", False),
                        "coarse_sigma_m": coarse.get("sigma_m"),
                        "precision_edge_readings": sum(r.get("end") == name for r in readings),
                        "precision_plates": sorted(key.split(":", 1)[1] for key in (faces.get("plates") or {})
                                                   if key.startswith(name + ":")),
                        "delivered_identified": final.get("two_sided", False),
                        "delivered_sigma_m": final.get("sigma_m")}
    return result


CALIB_USED_FILE = "calib_used.json"


def rebase_recording_calib(scan_dir, calib_path, T_world_optical, log=print):
    """The recording's optical poses under calib_path: a scan recorded with another calib (e.g. before the KIRO
    hand-eye fix) is re-based, T_new = T_rec @ inv(T_laser_opt_rec) @ T_laser_opt_new (the robot FK is unchanged).
    Returns {"used", "recorded", "rebased", "T_world_optical"}."""
    from construct_robot.core.scan_calib import calib_matrix, rebase_optical, same_calib
    from construct_robot.io.scan_calib import load_calib_pose, recorded_calib
    used, recorded = load_calib_pose(calib_path), recorded_calib(scan_dir)
    rebased = recorded is not None and not same_calib(recorded, used)
    if rebased:
        T_world_optical = rebase_optical(T_world_optical, calib_matrix(recorded["xyz"], recorded["rpy"]),
                                         calib_matrix(used["xyz"], used["rpy"]))
        log(f"recording re-based to the calib {used['file']} (recorded with {recorded['file'] or 'its snapshot'}): "
            f"xyz {[round(v * 1e3, 2) for v in recorded['xyz']]} -> {[round(v * 1e3, 2) for v in used['xyz']]} mm")
    elif recorded is None:
        log(f"the recording names no calib: its poses are taken as recorded (calib {used['file']} for the flange)")
    return {"used": used, "recorded": recorded, "rebased": bool(rebased), "T_world_optical": T_world_optical}


def estimate_scan(scan_dir, *, job, platform_config, calib_path, urdf, teaching_dir, out=None, log=print, rough="scan",
                  weld_direction="E->S"):
    import numpy as np
    import yaml
    from hongik_scan_ros.calib_tf import calib_from_tf
    from scan_core.g1_outputs import write_outputs
    from scan_core.g1_types import Precision, SeamPrior, Tolerances
    from scan_core.groove_model import fit_leg, profile_prior
    from scan_core.kiro_plan import KiroOutputConfig, _offline_nozzle_end, resolve_world_from_base, write_kiro_outputs
    from scan_core.seam_estimate import estimate
    from scan_core.types import ScanResult
    from scan_core.weld_plan import WeldProcess, weld_design_from_doc
    from scan_core.weld_torch import weld_torch_config
    from scan_l0 import config as C
    from scan_l0.autoteach import weld_process_doc

    scan_dir = Path(scan_dir).expanduser()
    out = Path(out).expanduser() if out else scan_dir / "hongik_estimate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "estimate.json").unlink(missing_ok=True)       # never report an earlier run's ends
    data = np.load(scan_dir / "scan.npz")
    report = {"scan_dir": str(scan_dir), "out_dir": str(out), "job": str(job)}

    with open(calib_path, encoding="utf-8") as stream:
        cal = yaml.safe_load(stream)
    T_flange_opt = _fixed_chain(urdf, "hongik_laser_link", "hongik_flange") @ _matrix(cal["xyz"], cal["rpy"])
    T_flange_eoat = _fixed_chain(urdf, "hongik_eoat_base_link", "hongik_flange")
    settings = C.load_settings(platform_config)
    calib = calib_from_tf(C.load_calib(calib_path), T_flange_opt, T_flange_eoat, note="offline: URDF + calib file")
    with open(job, encoding="utf-8") as stream:
        doc = yaml.safe_load(stream)
    # Hongik weld_plan.direction: S = the end the scan visited first. KIRO welds from the far end (2026-10-14).
    if weld_direction:
        doc.setdefault("weld_plan", {})["direction"] = str(weld_direction)
    report["weld_direction"] = (doc.get("weld_plan") or {}).get("direction")
    spec = C.specimen_from_doc(doc)
    tol = Tolerances.from_qc(doc.get("qc_targets") or {})
    process = WeldProcess.from_dict(weld_process_doc(doc, "1G"))

    # ---- the recording as Hongik's reconstruction result (poses already at each frame stamp)
    X, Z = data["x_mm"].astype(np.float32), data["z_mm"].astype(np.float32)
    t = data["stamp_ns"].astype(np.float64) * 1e-9
    T_bo = data["T_world_optical"].astype(np.float64)
    calib_doc = rebase_recording_calib(scan_dir, calib_path, T_bo, log)
    T_bo = calib_doc.pop("T_world_optical")
    report["calib"] = calib_doc
    (out / CALIB_USED_FILE).write_text(json.dumps(calib_doc, indent=1), encoding="utf-8")
    n = len(t)
    if n < 10:
        raise ValueError(f"only {n} recorded profiles")
    z_lo, z_hi = (float(v) for v in settings.z_valid_mm)
    valid = np.isfinite(X) & np.isfinite(Z) & (Z > z_lo) & (Z < z_hi)
    T_bf = T_bo @ np.linalg.inv(T_flange_opt)
    p_f = T_bf[:, :3, 3]
    speed = np.zeros(n)
    speed[1:-1] = np.linalg.norm(p_f[2:] - p_f[:-2], axis=1) / np.maximum(t[2:] - t[:-2], 1e-6)
    origins = T_bo[:, :3, 3]
    u = scan_axis(origins)                                           # the recording's own travel axis, first pass +
    s_axis = (origins - origins[0]) @ u
    passes = split_passes(t, s_axis)
    if not passes:
        raise ValueError("no moving pass in the recording")
    report.update(profiles=n, passes_recorded=[])
    log(f"profiles {n}, {len(passes)} pass(es) along the scan axis {np.round(u, 3).tolist()}")

    # Parallel like the Hongik server (g1_worker: fork_map, n = cpu - 4, at most 16).
    from scan_l0.g1_worker import fork_map
    n_children = max(1, min((os.cpu_count() or 1) - 4, 16))
    legs, bevel = [], None
    for k, (idx, sign) in enumerate(passes, 1):
        res = ScanResult(x_mm=X[idx], z_mm=Z[idx], valid=valid[idx], result_counter=idx.astype(np.int64),
                         ts_ms=(t[idx] * 1e3).astype(np.int64), t_rx_s=t[idx], t_acq_s=t[idx], T_base_flange=T_bf[idx],
                         T_base_opt=T_bo[idx], in_cruise=np.ones(len(idx), bool), unsynced=np.zeros(len(idx), bool),
                         ts_flag=np.zeros(len(idx), bool), speed_mps=speed[idx], s_m=s_axis[idx], base_frame="World",
                         calib=calib, meta={"source": f"KIRO direct scan {scan_dir}", "pass": k})
        name = f"pass{k}"
        log(f"fitting pass {k}/{len(passes)} ({len(idx)} profiles, {'S->E' if sign > 0 else 'E->S'})...")
        # the bevel side found by the first pass holds for the next ones (one mirror template instead of two)
        leg = fit_leg(res, spec, tol, cls="pass", direction=sign, bevel_x_sign=bevel, prior=profile_prior(spec, tol),
                      name=name, n_children=n_children, map_fn=fork_map)
        bevel = int(leg.bevel_x_sign)
        leg_dir = out / "legs" / f"{k:02d}_{name}"
        leg_dir.mkdir(parents=True, exist_ok=True)
        np.savez(leg_dir / "recon.npz", **{key: np.asarray(v) for key, v in res.array_fields().items()})
        leg.save(leg_dir / "profiles.npz")
        leg.recon_path = str(leg_dir / "recon.npz")
        legs.append(leg)
        groove = int((np.asarray(leg.log_bf) > 0).sum())
        info = {"pass": k, "direction": "S->E" if sign > 0 else "E->S", "profiles": len(idx), "groove_profiles": groove,
                "travel_mm": float(abs(s_axis[idx[-1]] - s_axis[idx[0]]) * 1e3), "bevel_x_sign": bevel,
                "sigma_r_mm_median": float(np.nanmedian(leg.sigma_r_mm))}
        report["passes_recorded"].append(info)
        log(f"pass {k} {info['direction']}: {groove}/{len(idx)} profiles with groove evidence over "
            f"{info['travel_mm']:.1f} mm, sigma_r median {info['sigma_r_mm_median']:.3f} mm")

    # Rough seam points (Hongik's taught S0 / E0: the origin and direction of its seam model, the offsets are fitted):
    # the groove root of the first / last profiles with groove evidence along the scan (S = the end visited first).
    S0, E0, rough_source = rough_points_from_legs(legs, origins[0], u)
    if S0 is None and rough == "ab":
        S0, E0, rough_source = _rough_point(teaching_dir, "start"), _rough_point(teaching_dir, "end"), "A / B teaching"
    if S0 is None:
        raise ValueError(f"no groove found in the scan ({rough_source})")
    report["rough_points_mm"] = {"S0": (S0 * 1e3).tolist(), "E0": (E0 * 1e3).tolist(), "source": rough_source}
    log(f"rough seam points ({rough_source}): S0 {np.round(S0 * 1e3, 1).tolist()} E0 {np.round(E0 * 1e3, 1).tolist()} mm")

    prior = SeamPrior(S0=S0, E0=E0, up=np.array([0.0, 0.0, 1.0]), spec=spec, tol=tol, k=float(settings.judgement_k),
                      dt_s=float(calib.dt_s), dt_sigma_s=float("nan"), rep_sigma_m=float("nan"),
                      he_sigma_m=float(calib.he_sigma_m), sigma_eps=float("nan"),
                      station_step_m=float(process.station_step_mm) * 1e-3,
                      station_window_m=float(process.station_window_mm) * 1e-3, bevel_sign_w=None,
                      edge_response=getattr(calib, "edge_response", None), precision=Precision.from_doc(doc.get("precision")),
                      noise_priors=getattr(calib, "noise_priors", None))
    log("estimating the seam ends (S / E) from the passes...")
    est = estimate(legs, prior, map_fn=fork_map, n_children=n_children)
    (out / "estimate.json").write_text(json.dumps(est.to_dict(), indent=1, default=str), encoding="utf-8")
    ends = {}
    for name in ("S", "E"):
        end = (est.ends or {}).get(name)
        ends[name] = None if end is None else {
            "point_mm": (np.asarray(end.point_m) * 1e3).tolist(), "sigma_mm": float(end.sigma_m) * 1e3,
            "two_sided": bool(end.two_sided)}
    report["ends"] = ends

    with open(platform_config, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    params = raw["/**"]["ros__parameters"] if "/**" in raw else raw
    cfg = KiroOutputConfig.from_dict(dict(params.get("kiro_output") or {}), weld_torch_config(params))
    if cfg.torch is not None and cfg.torch.nozzle_end_mm is None:
        cfg = dataclasses.replace(cfg, torch=_offline_nozzle_end(cfg.torch, out, urdf))
    log("ends: " + ", ".join(f"{k_} {'identified' if (e or {}).get('two_sided') else 'NOT identified'}" for k_, e in ends.items())
        + " - building the weld plan and the KIRO pass start / goal poses...")
    outputs = write_outputs(est, prior, process, out, base_frame="World", gravity_axis_base=np.array([0.0, 0.0, -1.0]),
                            k=prior.k, run={"notes": [f"offline from KIRO direct scan {scan_dir}"]},
                            design=weld_design_from_doc(doc, process), torch=cfg.torch)
    plan = outputs["plan"]
    report["plan_summary"] = list(plan.summary_lines())
    world, notes = resolve_world_from_base(cfg, "World")
    kiro = write_kiro_outputs(plan, out, cfg, world_from_base=world, notes=[f"offline from KIRO direct scan {scan_dir}"] + notes)
    report["passes"] = kiro.get("passes")
    report["files"] = [str(f) for f in outputs["files"] + kiro["files"]]
    return report


def main(argv=None):
    defaults = _defaults()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir")
    ap.add_argument("--job", help="Hongik job YAML (default: scan_l0 template groove_10T_1G)")
    ap.add_argument("--platform-config", default=defaults["platform_config"])
    ap.add_argument("--calib", default=defaults["calib"])
    ap.add_argument("--urdf", default=defaults["urdf"])
    ap.add_argument("--teaching-dir", default=defaults["teaching_dir"])
    ap.add_argument("--rough", choices=("scan", "ab"), default="scan",
                    help="rough seam points: from the scan's groove (default), A / B teaching only as a fallback (ab)")
    ap.add_argument("--weld-direction", choices=WELD_DIRECTIONS, default="E->S",
                    help="Hongik weld_plan.direction (S = the end the scan visited first; default E->S)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    job = a.job
    if not job:
        import scan_l0
        job = str(Path(scan_l0.__file__).resolve().parent / "jobs" / "groove_10T_1G.yaml")
    report = {"status": "ERROR"}
    try:
        report = estimate_scan(a.scan_dir, job=job, platform_config=a.platform_config, calib_path=a.calib, urdf=a.urdf,
                               teaching_dir=a.teaching_dir, out=a.out, rough=a.rough, weld_direction=a.weld_direction,
                               log=lambda text: print(text, file=sys.stderr, flush=True))
        report["status"] = "OK"
    except Exception as error:                    # PlanContractError: ends not identified -> no plan (reported, not hidden)
        report.update(status="REFUSED" if type(error).__name__ == "PlanContractError" else "ERROR",
                      message=f"{type(error).__name__}: {error}")
        partial = Path(a.out or Path(a.scan_dir).expanduser() / "hongik_estimate") / "estimate.json"
        if partial.exists():
            partial_doc = json.loads(partial.read_text(encoding="utf-8"))
            report["end_diagnostics"] = end_diagnostics(partial_doc)
            ends = partial_doc.get("ends") or {}
            report["ends"] = {k: {kk: v.get(kk) for kk in ("point_m", "sigma_m", "two_sided")} if v else None
                              for k, v in ends.items()}
    print("RESULT " + json.dumps(report, default=str), flush=True)
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
