"""KIRO `welding_interfaces/action/PrecisionSensing` <-> our run (interface design r3 §4, r2 §3 / §5) — mapping only, no ROS at import.

  goal   → RunInputs: welding_position 1 → 1G (the 1G core, Teach1G), anything else → InvalidGoal (the server accepts the goal and
           aborts with ERROR_INVALID_GOAL and this text). pose_a / pose_b positions = the rough
           S / E hit points in the robot base frame (the platform base_frame); their orientations are not used (KIRO question Q2).
           The outputs name the ends S / E in one frame: S = the end the run visits first (decided before any motion);
           ends_from_goal() says which goal pose each end came from (npz meta 'ends_from_goal').
           Specimen, tolerances and process come from the job file on this PC (server `job_file`, re-read per goal); the start pose is
           the arm pose at goal acceptance (the job's start_pose is ignored); confirmation = server `kiro_confirm_mode`.
  result → Outcome: our terminal status + category + reason → (error_code, error_message = "<status>: <reason>", succeed / abort /
           canceled) — the table of r3 §4.2 (CATEGORY_CODES, STATUS_DEFAULT_CATEGORY). PASS_UNRESOLVED → SUCCESS + paths (its reason
           lists the unresolved quantities) until KIRO answers Q4; a cancel → canceled with ERROR_INTERNAL (temporary, Q4).
  plan   → PathArrays per pass → WeldingPath[] (welding_paths_msgs): the pass's straight line, the same line as kiro/pass_<k>.yaml
           (scan_core.weld_line.weld_line - the line D14 checked) as the wire-tip (bead aim point) line, in the WeldingPath form KIRO defined: robot
           base frame [m], waypoints every <= the job's weld_plan.station_step_mm (5 mm) in the welding order (weld_plan.direction),
           +Z = the pass's plate normal on the torch side, +X = along the line, +Y = Z x X (constant; last = previous). The first / last
           waypoint = the pass_<k>.yaml start / goal position in the base frame. What WeldingPath cannot carry (its documented frame and
           orientation): the World frame and the torch orientation - ACTION_MAPPING, and the yaml poses ride in the npz (kiro_*).
  npz    → one file out_dir/precision_sensing.npz (precision_npz_arrays / write_precision_npz, compressed, written atomically): meta_json,
           sec_* (the run's section file, prefixed), sec_summary_json, n_paths, p{k}_* (waypoints, our torch frames, process values,
           cap weave, info, dense path), kiro_* (the KIRO-format files of out_dir/kiro/ as text). The existing sections / weld_plan files
           stay as they are.
  KIRO format → out_dir/kiro/pass_<k>.yaml (KIRO's xy_endpoint_review_v1: straight line start -> goal, World frame, KIRO TCP) +
           pass_<k>_info.yaml (assumed KIRO settings + derived values, information - not commands) + README.md, written by the run
           (scan_core.kiro_plan). The action result has NO slot for them (WeldingPath = robot
           base frame waypoints with +Z = plate normal; no endpoint / TCP orientation / process fields): meta 'kiro_outputs' names the
           folder and files, and the npz carries their text (kiro_npz_arrays).
  both directions → out_dir/directions/S_to_E/ and directions/E_to_S/ (user 2026-10-14; scan_core.g1_outputs.DIRECTION_SETS_RULE):
           each a complete set of every pass in one weld direction (weld_plan.json / .npz + kiro/ in KIRO's format), so KIRO can weld
           either way without sensing again. result.paths, the kiro_* arrays and the rest of the npz stay the job's direction (the
           primary set); the npz meta lists the sets: weld_direction.direction_sets (every set: folder, weld_plan files, delivered or
           the reason it is not) and kiro_outputs.directions (each set's kiro/ files and pass poses).
  result file → out_dir/precision_sensing_result.yaml (user 2026-10-13): the full action result message, every field as the server
           returns it (result_yaml_text / write_result_yaml; rosidl_runtime_py), written by the server for every goal whose run made
           a folder and by the operator client hongik_precision_sensing (the same text).

Everything this module returns or writes is English text (user 2026-10-10).
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from scan_core.g1_outputs import DIRECTION_SETS_RULE
from scan_core.g1_types import atomic_write_bytes, jsonable
from scan_l0.autoteach import ABORT_CATEGORIES, STATUS_PASS, STATUS_PASS_UNRESOLVED

__all__ = ["SUCCESS", "ERROR_MOTION_FAILED", "ERROR_SENSOR_FAILED", "ERROR_SENSING_FAILED", "ERROR_WAYPOINT_GENERATION_FAILED",
           "ERROR_INVALID_GOAL", "ERROR_INTERNAL", "CODE_NAMES", "POSITION_NAMES", "SUPPORTED_POSITIONS", "STATUS_PASS",
           "STATUS_PASS_UNRESOLVED", "SUCCESS_STATUSES", "CATEGORIES", "CATEGORY_CODES", "STATUS_DEFAULT_CATEGORY", "CANCEL_CODE",
           "NPZ_NAME", "NPZ_SCHEMA", "InvalidGoal", "RunInputs", "Outcome", "PathArrays", "position_label", "unsupported_position_text",
           "goal_to_run_inputs", "rough_goal", "run_outcome", "refused_outcome", "waypoint_frames", "plan_paths", "welding_paths_msgs",
           "sections_arrays", "npz_meta", "precision_npz_arrays", "write_precision_npz", "load_precision_npz", "SECTION_FILES",
           "ends_from_goal", "weld_direction_of", "kiro_meta", "kiro_npz_arrays", "RESULT_SLOT_NOTE", "ACTION_MAPPING",
           "KIRO_ACTION_TYPE", "RESULT_YAML_NAME", "result_document", "result_yaml_text", "write_result_yaml", "load_result_yaml",
           "result_message"]

# ---- welding_interfaces/action/PrecisionSensing Result codes (the action's constants; tests compare them with the generated type) ----
SUCCESS = 0
ERROR_MOTION_FAILED = 1               # sensing motion: planning or control failed
ERROR_SENSOR_FAILED = 2               # laser profiler data acquisition failed
ERROR_SENSING_FAILED = 3              # seam recognition failed
ERROR_WAYPOINT_GENERATION_FAILED = 4  # weld-path waypoint computation failed
ERROR_INVALID_GOAL = 5
ERROR_INTERNAL = 255
CODE_NAMES = {SUCCESS: "SUCCESS", ERROR_MOTION_FAILED: "ERROR_MOTION_FAILED", ERROR_SENSOR_FAILED: "ERROR_SENSOR_FAILED",
              ERROR_SENSING_FAILED: "ERROR_SENSING_FAILED", ERROR_WAYPOINT_GENERATION_FAILED: "ERROR_WAYPOINT_GENERATION_FAILED",
              ERROR_INVALID_GOAL: "ERROR_INVALID_GOAL", ERROR_INTERNAL: "ERROR_INTERNAL"}

# ---- goal ----
POSITION_NAMES = {1: "1G", 2: "2G", 3: "3G", 4: "4G", 5: "5G", 6: "6G", 11: "1F", 12: "2F", 13: "3F", 14: "4F"}
SUPPORTED_POSITIONS = {1: "1G"}                       # this package: the 1G core only

# ---- our terminal statuses / abort categories (scan_l0.autoteach; the run puts the category on AutoTeachResult.category) ----
SUCCESS_STATUSES = (STATUS_PASS, STATUS_PASS_UNRESOLVED)
CATEGORIES = ABORT_CATEGORIES
CATEGORY_CODES = {"input": ERROR_INVALID_GOAL, "config": ERROR_INTERNAL, "motion": ERROR_MOTION_FAILED, "sensor": ERROR_SENSOR_FAILED,
                  "sensing": ERROR_SENSING_FAILED, "waypoint": ERROR_WAYPOINT_GENERATION_FAILED, "internal": ERROR_INTERNAL}
# no category from the run → by status (r2 §3.2): REFUSED / ABORTED → motion, DATA_FAIL → sensing (3G legacy SPEC_FAIL =
# measurement quality, ALIGN_FAIL = alignment: also sensing), ERROR → internal. DRY_RUN has no mapping (a KIRO goal never dry-runs).
STATUS_DEFAULT_CATEGORY = {"REFUSED": "motion", "ABORTED": "motion", "DATA_FAIL": "sensing", "SPEC_FAIL": "sensing",
                           "ALIGN_FAIL": "sensing", "ERROR": "internal"}
CANCEL_CODE = ERROR_INTERNAL          # temporary until KIRO answers Q4 (no cancel code in the action)

# ---- result files ----
KIRO_ACTION_TYPE = "welding_interfaces/action/PrecisionSensing"
RESULT_YAML_NAME = "precision_sensing_result.yaml"
NPZ_NAME = "precision_sensing.npz"
NPZ_SCHEMA = "hongik.precision_sensing/1"
SECTION_FILES = (("sections.npz", "sections.json"), ("cross_sections.npz", "cross_sections.json"))   # 1G core / legacy runs


class InvalidGoal(ValueError):
    """The goal cannot become a run (→ ERROR_INVALID_GOAL with str(e))."""


@dataclass(frozen=True)
class RunInputs:
    welding_position: int
    position: str                     # '1G'
    start_m: tuple                    # rough S = pose_a.position (robot base frame)
    end_m: tuple                      # rough E = pose_b.position
    notes: tuple = ()


def position_label(n) -> str:
    return POSITION_NAMES.get(int(n), "not a PrecisionSensing position")


def unsupported_position_text(n, g1_pipeline: str) -> str:
    return f"welding_position {int(n)} ({position_label(n)}) not supported: 1G ({str(g1_pipeline or '').strip() or '?'}) only"


def _xyz(p, name: str) -> tuple:
    v = (float(p.x), float(p.y), float(p.z))
    if not all(math.isfinite(x) for x in v):
        raise InvalidGoal(f"{name}.position is not finite: {list(v)}")
    return v


def goal_to_run_inputs(goal, *, g1_pipeline: str) -> RunInputs:
    """PrecisionSensing goal → RunInputs, or InvalidGoal (unsupported welding_position, non-finite position)."""
    n = int(goal.welding_position)
    if n not in SUPPORTED_POSITIONS:
        raise InvalidGoal(unsupported_position_text(n, g1_pipeline))
    S = _xyz(goal.pose_a.position, "pose_a")
    E = _xyz(goal.pose_b.position, "pose_b")
    notes = ("pose_a / pose_b positions = rough S / E hit points in the robot base frame; their orientations are not used (KIRO Q2)",
             "specimen, tolerances and process from the job file on this PC; start pose = the arm pose at goal acceptance "
             "(the job's start_pose is ignored)")
    return RunInputs(welding_position=n, position=SUPPORTED_POSITIONS[n], start_m=S, end_m=E, notes=notes)


def ends_from_goal(s_taught_point: str) -> dict:
    """Which goal pose each output end came from. The run gets pose_a as its taught start and pose_b as its taught end;
    s_taught_point = the run result's AutoTeachResult.s_taught_point ('start' | 'end': the taught point the outputs call S)."""
    sp = str(s_taught_point or "").strip()
    if sp not in ("start", "end"):
        raise ValueError(f"s_taught_point must be 'start' or 'end', got {s_taught_point!r}")
    return {"S": "pose_a", "E": "pose_b"} if sp == "start" else {"S": "pose_b", "E": "pose_a"}


def weld_direction_of(plan) -> Optional[dict]:
    """The plan's weld direction setting (WeldProcess.direction) and the per-pass directions it gave, or None without a plan; a plan
    with direction sets (both weld directions, user 2026-10-14) adds direction_sets = {set: {direction, same_as_primary, delivered,
    reason, folder, files (its weld_plan.*)}} and the rule."""
    if plan is None:
        return None
    out = {"setting": str(plan.process.direction), "passes": {int(p.k): str(p.direction) for p in plan.passes}}
    sets = getattr(plan, "directions", None) or {}
    if sets:
        out["direction_sets"] = jsonable({nm: ds.to_dict() for nm, ds in sets.items()})
        out["direction_sets_rule"] = DIRECTION_SETS_RULE
    return out


def rough_goal(inputs: RunInputs) -> SimpleNamespace:
    """The goal object build_rough_job (auto_teach_server) reads: the position only — every design value comes from the job file
    (0 / '' = 'not given in the goal'); the points are passed to build_rough_job as start= / end=."""
    return SimpleNamespace(position=inputs.position, rough_start=None, rough_end=None, weld_length_mm=0.0, thickness_mm=0.0,
                           root_gap_mm=0.0, bevel_angle_deg=0.0, groove_kind="", bevel_side="")


# --------------------------------------------------------------------------------------
# result codes
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Outcome:
    action: str                       # 'succeed' | 'abort' | 'canceled'
    error_code: int
    error_message: str
    category: Optional[str] = None
    deliver_paths: bool = False       # paths (and their npz part) belong in the result


def run_outcome(status: str, reason: str, *, category: Optional[str] = None, cancelled: bool = False) -> Outcome:
    """Our terminal status (+ category from the run, None = by status) → Outcome (r3 §4.2). ValueError for a status or category
    without a PrecisionSensing mapping (the server reports that as ERROR_INTERNAL)."""
    status = str(status)
    msg = f"{status}: {reason}"
    if cancelled:
        return Outcome("canceled", CANCEL_CODE, msg, category, False)
    if status in SUCCESS_STATUSES:
        return Outcome("succeed", SUCCESS, msg, None, True)
    cat = category if category is not None else STATUS_DEFAULT_CATEGORY.get(status)
    if cat is None:
        raise ValueError(f"run status {status!r} has no PrecisionSensing result mapping")
    if cat not in CATEGORY_CODES:
        raise ValueError(f"run category {cat!r} is not one of {CATEGORIES}")
    return Outcome("abort", CATEGORY_CODES[cat], msg, cat, False)


def refused_outcome(reason: str, category: str) -> Outcome:
    """A goal refused by the server before any run (nothing sent): 'REFUSED: <reason>' with the category's code."""
    return run_outcome("REFUSED", reason, category=category)


# --------------------------------------------------------------------------------------
# plan → WeldingPath
# --------------------------------------------------------------------------------------
@dataclass
class PathArrays:
    k: int
    pos_m: np.ndarray                 # (M, 3) the welding wire-tip TCP (bead aim point) on the pass's straight line, base frame, welding order
    R: np.ndarray                     # (M, 3, 3) waypoint frames, columns [X, Y, Z]
    quat_xyzw: np.ndarray             # (M, 4)
    s_mm: np.ndarray                  # (M,) seam position of each waypoint from the measured S end (linear between the line's ends)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not (math.isfinite(n) and n > 0.0):
        raise ValueError(f"not a direction: {v.tolist()}")
    return v / n


def waypoint_frames(P, Z) -> np.ndarray:
    """KIRO WeldingPath orientation: +Z = Z[i] (unit), +X = (P[i+1] − P[i]) projected onto the plane normal to +Z, +Y = Z × X, last
    waypoint = previous one. ValueError if there are fewer than two waypoints or a step has no component in that plane."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    Z = np.asarray(Z, dtype=float).reshape(-1, 3)
    N = P.shape[0]
    if N < 2 or Z.shape[0] != N:
        raise ValueError(f"waypoint frames need >= 2 waypoints with one normal each (got {N} points, {Z.shape[0]} normals)")
    R = np.zeros((N, 3, 3))
    for i in range(N - 1):
        z = _unit(Z[i])
        d = P[i + 1] - P[i]
        x = d - float(d @ z) * z
        nx = float(np.linalg.norm(x))
        if not (math.isfinite(nx) and nx > 0.0):
            raise ValueError(f"waypoints {i} and {i + 1}: no direction to the next waypoint in the plane normal to +Z")
        x = x / nx
        R[i] = np.column_stack([x, np.cross(z, x), z])
    R[N - 1] = R[N - 2]
    return R


def plan_paths(plan) -> list:
    """WeldPlan (scan_core.weld_plan / g1_outputs.G1WeldPlan) → [PathArrays] in pass order (k = 1 …): per pass the straight line of
    kiro/pass_<k>.yaml (scan_core.weld_line.weld_line: the layer's usable seam start -> goal; D14 checked this line) as KIRO's
    WeldingPath defines it: every
    position = the welding wire-tip TCP = our bead aim point (pass_<k>.yaml gives the torch tip end centre = this point moved back along
    -z_TCP by the torch's nozzle end + stick_out_mm), base frame, waypoints evenly
    every <= weld_plan.station_step_mm, +Z = the pass's plate normal (mean of its sections' normals, the torch side, made normal to the
    line), +X = along the line (waypoint_frames)."""
    from scan_core.weld_line import weld_line
    step_mm = float(plan.process.station_step_mm)
    out = []
    for wp in sorted(plan.passes, key=lambda p: int(p.k)):
        ln = weld_line(plan, wp)                                       # THE delivered line's wire end (D14 checked it; kiro yaml)
        e = ln.ends
        P0, P1 = np.asarray(ln.tip_start, dtype=float), np.asarray(ln.tip_goal, dtype=float)
        L_mm = float(np.linalg.norm(P1 - P0)) * 1e3
        M = max(2, int(math.ceil(L_mm / step_mm - 1e-9)) + 1)
        f = np.linspace(0.0, 1.0, M)
        P = P0[None, :] + f[:, None] * (P1 - P0)[None, :]
        n = _unit(np.mean([np.asarray(st.n, dtype=float) for st in plan.stations], axis=0))
        try:
            R = waypoint_frames(P, np.tile(n, (M, 1)))
        except ValueError as ex:
            raise ValueError(f"pass {wp.k}: {ex}") from None
        q = Rotation.from_matrix(R).as_quat()                                   # x, y, z, w
        s = e["start"]["s_mm"] + f * (e["goal"]["s_mm"] - e["start"]["s_mm"])
        out.append(PathArrays(k=int(wp.k), pos_m=P, R=R, quat_xyzw=np.asarray(q, dtype=float).reshape(-1, 4), s_mm=s))
    return out


def welding_paths_msgs(paths: list) -> list:
    """[PathArrays] → [welding_interfaces/msg/WeldingPath] (imports the message types here: the module itself needs no ROS)."""
    from geometry_msgs.msg import Pose
    from welding_interfaces.msg import WeldingPath
    def pose(p, q):
        ps = Pose()
        ps.position.x, ps.position.y, ps.position.z = (float(v) for v in p)
        ps.orientation.x, ps.orientation.y, ps.orientation.z, ps.orientation.w = (float(v) for v in q)
        return ps
    msgs = []
    for pa in paths:                    # KIRO welds start -> goal with its own taught attitude: only the two ends are sent
        m = WeldingPath()
        m.pass_number = int(pa.k)
        m.start = pose(pa.pos_m[0], pa.quat_xyzw[0])
        m.goal = pose(pa.pos_m[-1], pa.quat_xyzw[-1])
        msgs.append(m)
    return msgs


# --------------------------------------------------------------------------------------
# precision_sensing.npz
# --------------------------------------------------------------------------------------
def sections_arrays(out_dir) -> tuple:
    """The run's section file in out_dir (1G core sections.npz, else legacy cross_sections.npz) → (arrays prefixed 'sec_' + sec_summary_json
    = the matching JSON document without its per-section list, source name) — ({}, None) when the run wrote no sections."""
    out = Path(out_dir)
    for npz_name, json_name in SECTION_FILES:
        p = out / npz_name
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            arr = {f"sec_{k}": np.asarray(z[k]) for k in z.files}
        jp = out / json_name
        if jp.exists():
            doc = json.loads(jp.read_text(encoding="utf-8"))
            arr["sec_summary_json"] = np.array(json.dumps({k: v for k, v in doc.items() if k not in ("stations", "passes")},
                                                          ensure_ascii=False))
        return arr, npz_name
    return {}, None


WAYPOINT_DEFINITION = {
    "position": "the welding wire-tip TCP = the bead aim point on the pass's straight line (KIRO WeldingPath definition); "
                "kiro/pass_<k>.yaml current_start / current_goal = the torch tip end centre = this point - (nozzle_end_mm + "
                "stick_out_mm) x z_TCP (meta kiro_outputs.tcp_to_aim_point_mm; the stick-out is measured from the gas nozzle end); "
                "robot base frame [m]; "
                "welding order = the pass's direction (meta weld_direction: the plan setting weld_plan.direction, default S->E)",
    "orientation": "+Z = the pass's plate normal on the torch side; +X = direction to the next waypoint projected onto the plane "
                   "normal to +Z (= along the line); +Y = Z x X; the last waypoint repeats the previous orientation",
    "spacing": "evenly along the line, at most the job's weld_plan.station_step_mm (default 5 mm)",
    "quaternion": "x, y, z, w",
}
FRAME_DEFINITION = {
    "S": "the measured seam end named S: the end the sensing run visited first; meta ends_from_goal says which goal pose (pose_a / "
         "pose_b) it came from",
    "E": "the other measured seam end",
    "u": "seam axis, unit(E - S) from the measured S end to the measured E end (S->E contract)",
    "n": "plate normal on the torch / sensor side (the side the groove opens to)",
    "w": "n x u (left when looking from S to E)",
    "s": "distance from the measured S end along u (p{k}_s_mm in mm, sec_s_m in m)",
    "R_torch": "our torch frame per waypoint: columns [x = travel, y = z x x, z = torch axis toward the work] (work and travel angles)",
    "tcp": "p{k}_tcp_pos_m / p{k}_tcp_R per section: the weld torch's TCP (torch tip end centre; platform config weld_torch = our "
           "torch) at the sections = p{k}_aim_pos_m - (nozzle_end_mm + stick_out_mm) x z_TCP, orientation = R_torch with the plan's "
           "TCP roll (meta kiro_outputs.tcp_to_aim_point_mm / stick_out_mm / nozzle_end_mm / torch_roll / weld_torch), base frame",
    "weave": "cap weave amplitude / width per waypoint in mm (NaN for fill passes); the dense path p{k}_dense_* contains the weave",
}


RESULT_SLOT_NOTE = ("PrecisionSensing.Result has no field for the pass_<k>.yaml poses as they are (World frame, torch orientation): "
                    "result.paths carries the same straight lines in WeldingPath's documented form (ACTION_MAPPING); the yaml text and "
                    "the World poses ride in this npz (kiro_*), the files are in out_dir/kiro/")
ACTION_MAPPING = {
    "paths": "result.paths[k-1] = pass k = the wire-tip (bead aim point) line of kiro/pass_<k>.yaml in WeldingPath's documented "
             "form: every position = the welding wire-tip TCP, robot base frame [m], waypoints evenly every <= 5 mm (job "
             "weld_plan.station_step_mm) in the welding order, +Z = the pass's plate normal on the torch side, +X = along the line, "
             "+Y = Z x X, the last orientation = the previous one. First / last waypoint = the pass_<k>.yaml start / goal TCP "
             "positions + (nozzle_end_mm + stick_out_mm) x z_TCP (meta kiro_outputs.tcp_to_aim_point_mm), expressed in the base frame "
             "(World <- base = meta "
             "kiro_outputs.world_from_base; the same numbers when the planning frame is World).",
    "not_carried": "the World frame and the constant torch (TCP) orientation of pass_<k>.yaml: WeldingPath has no header / frame_id "
                   "('no header is included by design') and its orientation is defined as the weld-surface frame (+Z = surface "
                   "normal), not the torch orientation",
    "conflict": "WeldingPath (README + msg comments): all waypoints in the robot base frame, about 0.005 m apart, +Z = weld-surface "
                "normal, +X = tangent to the next waypoint. pass_<k>.yaml: World frame, start and goal only, orientation = the TCP "
                "(torch) orientation, constant. Putting the yaml poses into paths as they are would change all three documented "
                "meanings without a frame_id to mark it - not done",
    "yaml_poses_in_npz": "data_file_path (precision_sensing.npz): kiro_pass_<k>_yaml (the file text), kiro_pass_<k>_start_pose / "
                         "kiro_pass_<k>_goal_pose = [x, y, z, qx, qy, qz, qw] in World (the TCP = torch tip end centre)",
    "ask_kiro": [
        "May result.paths carry the pass_<k>.yaml poses directly (World frame, torch orientation, two waypoints start / goal)? That "
        "changes WeldingPath's documented frame, spacing and orientation - it needs your definition / README change.",
        "Or add a result field for them (e.g. geometry_msgs/PoseStamped[] pass_start, pass_goal - header.frame_id 'World' - or a "
        "WeldingPass message with a header and the start / goal poses).",
        "Is your MoveIt planning frame World? Then the base-frame positions in result.paths are already the yaml's World positions "
        "and only the torch orientation is missing (your executor could take it from the npz or from your own travel / work angles).",
    ],
}


def _directions_meta(ko: dict) -> Optional[dict]:
    """kiro_outputs.directions of the meta: per direction set its kiro/ folder and files (+ its weld_plan files, the TCP roll and
    per pass the start / goal poses), or why it is not delivered / its KIRO files not written. None without sets."""
    ds = ko.get("directions") or {}
    if not ds:
        return None
    keep = ("direction", "delivered", "same_as_primary", "folder", "dir", "files", "weld_plan_files", "reason", "kiro_written",
            "kiro_reason", "torch_roll", "passes")
    return {"rule": DIRECTION_SETS_RULE,
            "in_this_npz": "the primary set only (the job's weld_plan.direction): result.paths, kiro_* and p{k}_* are its; the sets "
                           "are the files listed here",
            "sets": {nm: {k: d.get(k) for k in keep if k in d} for nm, d in ds.items()}}


def kiro_meta(kiro_outputs: Optional[dict]) -> Optional[dict]:
    """The run's KIRO-format output summary (AutoTeachResult.kiro_outputs) → npz meta 'kiro_outputs' (no per-pass arrays); with
    direction sets (both weld directions) their listing under 'directions' (_directions_meta)."""
    ko = dict(kiro_outputs or {})
    if not ko:
        return None
    if not ko.get("written"):
        return {"written": False, "reason": ko.get("reason"), "result_slot": RESULT_SLOT_NOTE}
    return {"written": True, "dir": ko.get("dir"), "files": list(ko.get("files") or []), "schema": ko.get("schema"),
            "frame_id": ko.get("frame_id"), "tcp_frame": ko.get("tcp_frame"),
            "world_from_base": {k: (ko.get("world_from_base") or {}).get(k) for k in ("source", "status", "base_frame")},
            "ee_from_tip": {k: (ko.get("ee_from_tip") or {}).get(k) for k in ("source", "status")},
            "weld_torch": ko.get("weld_torch"), "stick_out_mm": ko.get("stick_out_mm"), "stick_out_source": ko.get("stick_out_source"),
            "nozzle_end_mm": ko.get("nozzle_end_mm"), "tcp_to_aim_point_mm": ko.get("tcp_to_aim_point_mm"),
            "tcp_definition": ko.get("tcp_definition"), "torch_roll": ko.get("torch_roll"),
            "info_files": ko.get("info_files"), "notes": list(ko.get("notes") or []), "result_slot": RESULT_SLOT_NOTE,
            "runtime_base_frame": ko.get("runtime_base_frame"), "action_mapping": ACTION_MAPPING,
            "passes": [{k_: p.get(k_) for k_ in ("k", "role", "start_World_m", "goal_World_m", "quat_xyzw", "quat_goal_xyzw",
                                                 "tip_start_World_m", "tip_goal_World_m", "torch_roll_deg",
                                                 "wps_nominal_travel_angle_deg", "wps_nominal_work_angle_deg", "written_travel_angle_deg",
                                                 "written_work_angle_deg", "deviation_from_wps_deg", "clearance_mm",
                                                 "required_clearance_mm", "orientation_choice")}
                       for p in ko.get("passes") or []],
            **({"directions": _directions_meta(ko)} if ko.get("directions") else {})}


def kiro_npz_arrays(kiro_outputs: Optional[dict]) -> dict:
    """out_dir/kiro/ files (as the run listed them) → npz string arrays kiro_pass_<k>_yaml, kiro_pass_<k>_info_yaml, kiro_readme_md, and
    per pass the yaml poses kiro_pass_<k>_start_pose / _goal_pose = [x, y, z, qx, qy, qz, qw] (World, TCP = torch tip end centre)."""
    ko = dict(kiro_outputs or {})
    out = {}
    if not ko.get("written"):
        return out
    for p in ko.get("passes") or []:
        k = int(p["k"])
        out[f"kiro_pass_{k}_start_pose"] = np.concatenate([np.asarray(p["start_World_m"], float), np.asarray(p["quat_xyzw"], float)])
        out[f"kiro_pass_{k}_goal_pose"] = np.concatenate([np.asarray(p["goal_World_m"], float),
                                                          np.asarray(p.get("quat_goal_xyzw") or p["quat_xyzw"], float)])
    for f in ko.get("files") or []:
        p = Path(f)
        if not p.is_file():
            continue
        key = "kiro_readme_md" if p.name == "README.md" else "kiro_" + p.stem + "_yaml"
        out[key] = np.array(p.read_text(encoding="utf-8"))
    return out


def npz_meta(*, outcome: Outcome, status: str, reason: str, welding_position: int, position: str, base_frame: str, k: float,
             out_dir: str, run_id: str, inputs: Optional[dict] = None, sections_source: Optional[str] = None,
             unresolved: Optional[list] = None, n_paths: int = 0, notes: Optional[list] = None,
             ends_from_goal: Optional[dict] = None, weld_direction: Optional[dict] = None,
             kiro_outputs: Optional[dict] = None) -> dict:
    """meta_json of precision_sensing.npz (r2 §5). ends_from_goal = {'S': 'pose_a' | 'pose_b', 'E': …} (kiro_io.ends_from_goal);
    weld_direction = kiro_io.weld_direction_of(plan); kiro_outputs = kiro_meta(AutoTeachResult.kiro_outputs)."""
    return jsonable({
        "schema": NPZ_SCHEMA, "status": str(status), "reason": str(reason), "error_code": int(outcome.error_code),
        "error_code_name": CODE_NAMES.get(int(outcome.error_code), "?"), "error_message": outcome.error_message,
        "category": outcome.category, "welding_position": int(welding_position), "position": str(position),
        "base_frame": str(base_frame), "units": {"position": "m", "dims": "mm", "angles": "deg", "time": "s"}, "k": float(k),
        "frames": FRAME_DEFINITION, "waypoints": WAYPOINT_DEFINITION, "run_id": str(run_id), "out_dir": str(out_dir),
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "inputs": dict(inputs or {}),
        "sections_source": sections_source, "unresolved": list(unresolved or []), "n_paths": int(n_paths), "notes": list(notes or []),
        "ends_from_goal": dict(ends_from_goal) if ends_from_goal else None, "weld_direction": weld_direction,
        "kiro_outputs": kiro_outputs,
    })


def _f1(v, n: int) -> np.ndarray:
    return np.full(n, np.nan) if v is None else np.asarray(v, dtype=float).reshape(n)


def precision_npz_arrays(meta: dict, *, plan=None, paths: Optional[list] = None, sections: Optional[dict] = None,
                         kiro: Optional[dict] = None) -> dict:
    """npz arrays: meta_json, sec_* (sections_arrays), kiro_* (kiro_npz_arrays), n_paths, and per pass k: the result's WeldingPath
    p{k}_wp_pos_m (M,3), p{k}_wp_quat_xyzw (M,4), p{k}_wp_s_mm (M,) (the pass's straight line); the plan per section (N sections, in
    the welding order) p{k}_s_mm, p{k}_aim_pos_m (N,3), p{k}_aim_sigma_m (N,2), p{k}_R_torch (N,3,3), p{k}_speed_mps, p{k}_area_mm2,
    p{k}_layer_to_mm, p{k}_weave_amp_mm, p{k}_weave_width_mm; p{k}_info_json; the dense plan path p{k}_dense_t_s, p{k}_dense_pos_m,
    p{k}_dense_R; with a weld torch model on the plan, its TCP poses at the sections p{k}_tcp_pos_m (N,3), p{k}_tcp_R (N,3,3)."""
    a = {"meta_json": np.array(json.dumps(jsonable(meta), ensure_ascii=False))}
    a.update(sections or {})
    a.update(kiro or {})
    paths = list(paths or [])
    a["n_paths"] = np.array(len(paths), dtype=np.int64)
    if not paths:
        return a
    by_k = {int(p.k): p for p in plan.passes}
    feas = {int(f.get("k")): f for f in (getattr(plan, "feasibility", None) or []) if isinstance(f, dict) and f.get("k") is not None}
    aim_sig = getattr(plan, "aim_sigma_m", None) or {}
    for pa in paths:
        wp, k, M = by_k[pa.k], pa.k, pa.pos_m.shape[0]
        N = int(np.asarray(wp.s_mm).size)
        sig = aim_sig.get(k)
        ds = np.abs(np.diff(pa.s_mm))
        d3 = np.linalg.norm(np.diff(pa.pos_m, axis=0), axis=1) * 1e3
        info = {"k": k, "role": wp.role, "kind": wp.kind, "layer": int(wp.layer), "bead": int(wp.bead), "n_beads": int(wp.n_beads),
                "direction": wp.direction, "duration_s": float(wp.duration_s), "n_waypoints": M, "n_sections": N,
                "waypoints": "the pass's straight line (kiro/pass_<k>.yaml start -> goal) in the base frame",
                "spacing_mm": {"max_along_seam": float(ds.max()), "max_3d": float(d3.max())},
                "n_speed_clamped": int(np.sum(np.asarray(wp.speed_clamped, dtype=bool))),
                "reachability": feas.get(k) if k in feas else (dict(getattr(plan, "reachability", None) or {}) or None),
                "speed_const_mm_s": float(getattr(wp, "speed_const_mps", float("nan"))) * 1e3,
                "area_mean_mm2": float(getattr(wp, "area_mean_mm2", float("nan")))}
        a.update({
            f"p{k}_wp_pos_m": pa.pos_m, f"p{k}_wp_quat_xyzw": pa.quat_xyzw, f"p{k}_wp_s_mm": pa.s_mm,
            f"p{k}_s_mm": np.asarray(wp.s_mm, dtype=float), f"p{k}_aim_pos_m": np.asarray(wp.aim_base_m, dtype=float).reshape(N, 3),
            f"p{k}_aim_sigma_m": np.full((N, 2), np.nan) if sig is None else np.asarray(sig, dtype=float).reshape(N, 2),
            f"p{k}_R_torch": np.asarray(wp.R_base_tcp, dtype=float).reshape(N, 3, 3),
            f"p{k}_speed_mps": _f1(wp.speed_mps, N), f"p{k}_area_mm2": _f1(wp.area_mm2, N), f"p{k}_layer_to_mm": _f1(wp.layer_to_mm, N),
            f"p{k}_weave_amp_mm": _f1(wp.weave_amp_mm, N), f"p{k}_weave_width_mm": _f1(wp.weave_width_mm, N),
            f"p{k}_info_json": np.array(json.dumps(jsonable(info), ensure_ascii=False)),
            f"p{k}_dense_t_s": np.asarray(wp.dense_t_s, dtype=float), f"p{k}_dense_pos_m": np.asarray(wp.dense_pos_m, dtype=float),
            f"p{k}_dense_R": np.asarray(wp.dense_R, dtype=float)})
        torch = getattr(plan, "torch", None)
        if torch is not None:                                   # the weld torch's TCP poses at the sections (weld_torch, stick-out, roll)
            P_tcp, R_tcp = torch.tcp_poses(np.asarray(wp.aim_base_m, dtype=float).reshape(N, 3),
                                           np.asarray(wp.R_base_tcp, dtype=float).reshape(N, 3, 3),
                                           roll_deg=float(getattr(plan, "torch_roll_deg", 0.0) or 0.0))
            a.update({f"p{k}_tcp_pos_m": P_tcp, f"p{k}_tcp_R": R_tcp})
    return a


def write_precision_npz(out_dir, arrays: dict) -> str:
    """out_dir/precision_sensing.npz (compressed; temporary file + fsync + os.replace) → its absolute path."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    p = (Path(out_dir) / NPZ_NAME).resolve()
    atomic_write_bytes(p, buf.getvalue())
    return str(p)


def load_precision_npz(path) -> dict:
    """precision_sensing.npz → {key: array}, with meta_json / *_json decoded into 'meta' / '<key without _json>' dicts."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k in [k for k in d if k.endswith("_json")]:
        d[k[:-5] if k != "meta_json" else "meta"] = json.loads(str(d[k]))
    return d


# --------------------------------------------------------------------------------------
# precision_sensing_result.yaml - the full action result message (user 2026-10-13)
# --------------------------------------------------------------------------------------
def _plain(v):
    """rosidl_runtime_py's OrderedDicts / lists → plain dicts / lists (yaml.safe_dump, field order kept)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def result_document(result) -> dict:
    """A PrecisionSensing Result message → {field: value} with every field of the message definition, nested messages as mappings
    (rosidl_runtime_py.message_to_ordereddict: the generated type's own field list - nothing chosen by hand)."""
    from rosidl_runtime_py import message_to_ordereddict
    return _plain(message_to_ordereddict(result))


def result_yaml_text(result) -> str:
    """The result file's text: two comment lines (the action, the code's name, what the waypoints are) + the message as YAML (floats
    written with repr: the file reads back to the same message). The server and the client write the same text for the same result."""
    import yaml
    doc = result_document(result)
    code = int(doc.get("error_code", -1))
    head = (f"# {KIRO_ACTION_TYPE} result: the full message as the server returned it; error_code {code} = "
            f"{CODE_NAMES.get(code, '?')}\n"
            "# paths[k-1] = pass k (WeldingPath): pass_number, start / goal = the wire tip at the pass's ends (the run's "
            "kiro/pass_<k>.yaml line), robot base frame (the frame of pose_a / pose_b) [m], orientation = quaternion x, y, z, w\n")
    return head + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=4096, allow_unicode=True)


def write_result_yaml(run_dir, result) -> str:
    """run_dir/precision_sensing_result.yaml (temporary file + fsync + os.replace) → its absolute path. The folder must exist (the run's
    folder: nothing is created here)."""
    d = Path(run_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"run folder {d} does not exist")
    p = (d / RESULT_YAML_NAME).resolve()
    atomic_write_bytes(p, result_yaml_text(result).encode("utf-8"))
    return str(p)


def load_result_yaml(path) -> dict:
    """precision_sensing_result.yaml → the message as a mapping (the comment lines dropped)."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a {KIRO_ACTION_TYPE} result document")
    return doc


def result_message(doc: dict, result_type):
    """A result mapping (load_result_yaml) → a result_type message (rosidl_runtime_py.set_message_fields: every field set, an unknown
    field = error)."""
    from rosidl_runtime_py import set_message_fields
    msg = result_type()
    set_message_fields(msg, doc)
    return msg
