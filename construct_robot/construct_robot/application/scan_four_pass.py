"""Four-pass teaching from a corrected scan estimate and taught pass torch orientations (no robot motion).

Positions: the sensed groove (Hongik's aim points = wire tips) as they are when correction is None (default since
2026-10-15: the sensed corner is accurate), else with the reference correction.
Orientations: the operator sets the torch as it must weld and presses a key (Scan tab, orientation mode); only the
orientation is used (the joints are kept as the IK seed). One capture may serve several passes (I = passes 1 and 3).

Per pass k with its orientation q_k (World, TCP = right_manipulator_ee_point):
    start / goal       pass_target_tcp(k, end, q_k): the corrected wire tip of pass k, TCP = wire tip - L z(q_k)
    start / goal wait  the pass-4 wire tip of that end + lift (World +Z), TCP from q_k (the highest pass, so every
                       pass approaches its start from above the finished layers)
    weld_wait          start wait + EXTRA_LIFT_M (World +Z), q_k (approach before the pass)
    weld_finish        goal wait + EXTRA_LIFT_M (World +Z), q_k (retreat after the pass)
Written per scan to ~/ros2_ws/weld_feedback/scan_four_pass_<scan>/ (Welding › 4-PASS: Load 4 references).
Written as construct_robot_pass_teaching_v1 pass_<k>.yaml (requires_ik: the 4-PASS section resolves the joints from
the TCP with MoveIt IK when a pass is applied).
"""

from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import yaml

from construct_robot.application.scan_correction import (
    ENDS, history_dir, load_estimate_detail, pass_target_tcp, scan_teaching_dir,
)
from construct_robot.core.scan_reference import plate_weave_direction, tcp_from_aim, torch_angles

PASSES = (1, 2, 3, 4)
WAIT_LIFT_M = 0.010             # start / goal wait above the pass-4 wire tip
EXTRA_LIFT_M = 0.060            # weld_wait / weld_finish above the start / goal wait (operator 2026-10-15: +30 mm)
# pose name, the wait it sits above (each pass's own orientation)
ADDITIONAL_POSES = (("weld_wait", "start_wait"), ("weld_finish", "goal_wait"))
# passes whose start / goal wait = their own weld start / goal TCP shifted in World (their orientation; operator
# 2026-10-15: pass 2 waits 3 mm -X and 3 mm +Z of its weld points)
WAIT_OFFSET_FROM_WELD_M = {2: (-0.003, 0.0, 0.003)}
ENDPOINT_POSES = (("start_wait", "weld_start_wait"), ("start", "weld_start"),
                  ("goal_wait", "weld_goal_wait"), ("goal", "weld_end"))


def orientation_path():
    return scan_teaching_dir() / "pass_orientations.yaml"


def weld_feedback_dir():
    return Path.home() / "ros2_ws" / "weld_feedback"


def four_pass_dir(scan_dir):
    return weld_feedback_dir() / f"scan_four_pass_{Path(scan_dir).expanduser().name}"


def latest_four_pass_dir():
    """The newest generated four-pass folder in weld_feedback (None when there is none)."""
    folders = [p for p in weld_feedback_dir().glob("scan_four_pass_*") if (p / "pass_1.yaml").is_file()]
    # the files' time: re-writing a folder's passes does not change the folder's own mtime
    return max(folders, key=lambda p: (p / "pass_1.yaml").stat().st_mtime) if folders else None


def save_pass_orientation(passes, quat_xyzw, joints, position, source, path=None):
    """Store the taught torch orientation of passes (e.g. (1, 3)); joints = (names, positions) = the IK seed."""
    passes = tuple(int(k) for k in passes)
    if not passes or any(k not in PASSES for k in passes):
        raise ValueError(f"passes must be within {PASSES}")
    q = np.asarray(quat_xyzw, float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
        raise ValueError("invalid orientation")
    names, positions = joints
    if len(names) != 6 or len(positions) != 6:
        raise ValueError("the IK seed needs the six arm joints")
    path = Path(path or orientation_path())
    doc = (yaml.safe_load(path.read_text()) if path.is_file() else None) or {}
    entries = doc.get("passes") or {}
    entry = {"orientation_xyzw": dict(zip("xyzw", (float(v) for v in q / np.linalg.norm(q)))),
             "captured_at_m": dict(zip("xyz", (float(v) for v in position))),
             "joint_state": {"names": [str(n) for n in names], "positions_rad": [float(v) for v in positions]},
             "captured_for": list(passes), "source": str(source), "saved": datetime.now().isoformat(timespec="seconds")}
    for k in passes:
        entries[k] = entry
    doc = {"format_version": 1, "frame": "World", "tcp_frame": "right_manipulator_ee_point",
           "note": "only the orientation is used; joint_state = IK seed", "passes": dict(sorted(entries.items()))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def load_pass_orientations(path=None):
    """{k: {"quat", "joints" (names, positions), "captured_for", "source", "saved"}} of the taught passes."""
    path = Path(path or orientation_path())
    if not path.is_file():
        return {}
    out = {}
    for k, e in ((yaml.safe_load(path.read_text()) or {}).get("passes") or {}).items():
        out[int(k)] = {"quat": np.array([e["orientation_xyzw"][a] for a in "xyzw"], float),
                       "joints": (tuple(e["joint_state"]["names"]), tuple(e["joint_state"]["positions_rad"])),
                       "captured_for": tuple(e.get("captured_for") or (k,)), "source": e.get("source", ""),
                       "saved": e.get("saved", "")}
    return out


def orientation_angles(scan_dir, orientations, detail=None):
    """{k: (travel, work) deg} of the taught orientations in the estimate's seam frame (Hongik convention, travel
    along this estimate's start -> goal)."""
    detail = detail or load_estimate_detail(scan_dir)
    frame, p1 = detail["frame"], detail["passes"][0]
    travel_sign = 1.0 if float((p1["aim_goal"] - p1["aim_start"]) @ frame.u) >= 0.0 else -1.0
    return {k: torch_angles(o["quat"], frame, travel_sign, detail["bevel_sign_w"]) for k, o in orientations.items()}


def four_pass_targets(scan_dir, correction, orientations, relative="design_layout", lift_m=WAIT_LIFT_M):
    """{k: {endpoint: (tcp xyz, quat xyzw)}} for start_wait / start / goal_wait / goal of passes 1-4."""
    missing = [k for k in PASSES if k not in orientations]
    if missing:
        raise ValueError("teach the torch orientation of pass(es) " + ", ".join(map(str, missing)) +
                         " first (Scan tab, orientation keys: I = passes 1 + 3, O = pass 2, P = pass 4)")
    detail = load_estimate_detail(scan_dir)
    if [r["k"] for r in detail["passes"]] != list(PASSES):
        raise ValueError(f"the estimate has passes {[r['k'] for r in detail['passes']]}, expected 1-4")
    lift = np.array([0.0, 0.0, float(lift_m)])
    top = {}
    for end in ENDS:                                   # the pass-4 wire tip of each end (orientation independent)
        top[end] = pass_target_tcp(scan_dir, 4, end, orientations[4]["quat"], correction, relative, detail)[1]
    out = {}
    for k in PASSES:
        q = orientations[k]["quat"]
        L = next(r["L_m"] for r in detail["passes"] if r["k"] == k)
        row = {}
        for end in ENDS:
            tcp, _aim = pass_target_tcp(scan_dir, k, end, q, correction, relative, detail)
            row[end] = (tcp, q)
            row[f"{end}_wait"] = (tcp_from_aim(top[end] + lift, q, L), q)
        if k in WAIT_OFFSET_FROM_WELD_M:
            for end in ENDS:
                row[f"{end}_wait"] = (row[end][0] + np.asarray(WAIT_OFFSET_FROM_WELD_M[k], float), q)
        for pose_name, wait in ADDITIONAL_POSES:
            row[pose_name] = (row[wait][0] + np.array([0.0, 0.0, EXTRA_LIFT_M]), q)
        # every pass: any pass may weave (4-PASS PARAMS tab), across the seam in the scanned plate plane
        normal = detail["plate_normal_world"]
        transverse = plate_weave_direction(row["start"][0], row["goal"][0], normal)
        row["weave_geometry"] = {"reference": "scan_plate_plane", "normal_world": normal.tolist(),
                                 "transverse_vector_world": transverse.tolist()}
        out[k] = row
    return out


def _pose_doc(tcp, quat):
    return {"position_m": dict(zip("xyz", (float(v) for v in tcp))),
            "orientation_xyzw": dict(zip("xyzw", (float(v) for v in quat)))}


def write_four_pass_teaching(scan_dir, correction, orientations=None, relative="design_layout", out_dir=None,
                             lift_m=WAIT_LIFT_M):
    """Write pass_1..4.yaml (+ a copy in the scan's history); returns (folder, targets)."""
    from construct_robot.application.kiro_estimate import check_estimate_length
    check_estimate_length(scan_dir)              # targets from a misread end run past the plate
    orientations = load_pass_orientations() if orientations is None else orientations
    targets = four_pass_targets(scan_dir, correction, orientations, relative, lift_m)
    folder = Path(out_dir or four_pass_dir(scan_dir))
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    for k, row in targets.items():
        poses = {}
        for endpoint, pose_name in ENDPOINT_POSES + tuple((name, name) for name, *_rest in ADDITIONAL_POSES):
            tcp, q = row[endpoint]
            names, positions = orientations[k]["joints"]
            poses[pose_name] = {
                "planning_group": "right_manipulator",
                "joint_state": {"names": list(names), "positions_rad": [float(v) for v in positions]},
                "tcp_pose_world": _pose_doc(tcp, q),
                "capture_provenance": {"capture_source": "scan_estimate",
                                       "joint_state": "IK seed (pass orientation capture); requires_ik"},
            }
        doc = {"schema": "construct_robot_pass_teaching_v1", "status": "scan_estimate_generated", "timestamp": stamp,
               "pass": int(k), "requires_ik": True, "poses": poses,
               "source_reference": str(Path(scan_dir).expanduser()), "source_reference_kind": "scan_estimate",
               "generated_from": {"scan": str(Path(scan_dir).expanduser()), "relative_model": relative,
                                  "endpoint_adjustments_world_m": (correction or {}).get("endpoint_adjustments_world_m"),
                                  "correction_computed": (correction or {}).get("computed"),
                                  "position_anchor_pass": (correction or {}).get("anchor_pass"),
                                  "reference_set_id": ((correction or {}).get("reference_set") or {}).get("id"),
                                  "calibration_truth_checks": (correction or {}).get("truth_checks"),
                                  "inferred_ends": (correction or {}).get("inferred_ends"),
                                  "anchor_design_m": (correction or {}).get("anchor_design_m"),
                                  "design_layout_m": (correction or {}).get("design_layout_m"),
                                  "calib": ((correction or {}).get("calib") or {}).get("file"),
                                  "orientation": orientations[k]["source"],
                                  "orientation_captured_for": list(orientations[k]["captured_for"]),
                                  "wait": f"pass-4 wire tip of the end + {lift_m * 1e3:.0f} mm World Z",
                                  "weld_wait / weld_finish": f"start / goal wait + {EXTRA_LIFT_M * 1e3:.0f} mm World Z, "
                                                             "this pass's orientation"}}
        if "weave_geometry" in row:
            doc["weave_geometry"] = row["weave_geometry"]
        (folder / f"pass_{k}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    archive = history_dir(scan_dir) / "four_pass"
    if archive.resolve() != folder.resolve():
        shutil.rmtree(archive, ignore_errors=True)
        archive.mkdir(parents=True, exist_ok=True)
        # The result folder also holds this cycle's scans and feedback logs.
        # Teaching history contains only the four teaching files.
        for k in PASSES:
            shutil.copy2(folder / f"pass_{k}.yaml", archive / f"pass_{k}.yaml")
    return folder, targets
