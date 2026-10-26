"""The torch orientation of each pass (user 2026-10-13): the WPS angles (travel / work, AWS A3.0) are the target; the plan takes, per
pass, the collision-free and reachable torch orientation CLOSEST to them, with no preset bound on the deviation - the deviation is an
output (nominal, chosen, deviation and WHY - the constraint that rejected the nominal and the closer candidates -, clearance per
pass in weld_plan.json, the KIRO info files and the plan summary), not a failure. Only when no orientation at any deviation is
feasible does the plan fail, with the reason.

Everything here is evaluated on the DELIVERED line of the pass (scan_core.weld_line, user 2026-10-13: checked = delivered =
executed): the one straight line at one orientation that kiro/pass_<k>.yaml carries and result.paths carries (its wire-end
line) - a candidate (travel, work) is the line whose orientation is the rotation mean of the pass's
station torch frames at those angles; its samples (the platform's Cartesian step apart) are checked, each in the section of the
station nearest its wire end. The per-station layer plan (weld_plan.npz dense paths) is information.

  feasible     the tool (the robot's URDF collision geometry rigid with the weld torch's TCP - the sensor guard included, the wire stub
               not: weld_clearance.tool_body_from_urdf) clears the specimen before the pass (weld_scene: both plates with the measured
               groove, the backing, the earlier passes' beads) by the required clearance of each part's role at every station of
               every line of the pass (fill: the centre line; cap: the centre and both weave edge lines) - the process parts (the gas
               nozzle and the contact tip, weld_clearance.tool_roles) by PROCESS_REQUIRED_MM (0: no interpenetration, KIRO 2026-10-13),
               every other part by r - AND the platform's D14 (check_weld_plan - MoveIt: IK, joint limits, the Cartesian path with
               collision checking against the same specimen grown by r, the link carrying the process parts allowed to touch it there
               and checked per role by this exact model; the virtual arm: IK, limits and the same geometric clearance) passes it.
  closest      angular distance = the angle between the candidate's torch axis and the nominal one; candidates on the grid of the
               WPS angle resolution (travel and work angles in steps of ANGLE_STEP_DEG from the nominal, over the whole hemisphere
               |travel|, |work| < 90 deg); equal distance → the larger clearance.
  roll         the two TCP rolls that keep the TCP y across the seam (weld_torch.ROLLS_DEG; KIRO weaves along TCP y), one per job.
               The torch neck bends WITH the travel angle (user 2026-10-13, a process definition): the straight torch body (the
               flange axis, from the robot model: weld_torch flange_from_tcp - our torch 0.5 x_TCP + 0.866 z_TCP) is inclined more than
               the nozzle axis and on the same side along the travel - a push torch has the parts after the bend facing the travel
               direction, a drag torch away from it (KIRO's identified September posture: body -69..-74 vs nozzle -39..-44 deg).
               So a roll serves the travel angles of its side only (neck_side: sign of the body's lean beyond the nozzle along the
               travel at that roll): a pass with a nonzero WPS travel angle is searched in that roll only, and every candidate
               searched in a roll keeps the travel side of that roll (travel 0 fits both); a pass with travel 0 (or a straight torch)
               is searched in both. No silent flip: a pass without a feasible orientation in its roll is refused with the reason.
               Among the rolls: the one at which every pass is feasible with the smaller total deviation (sum over the passes),
               equal → the larger joint-limit margin (D14), then 0 deg.
  required r   (the parts other than the gas nozzle and the contact tip)
               k x the combined sigma (root-sum-square) of the stated uncertainties (clearance_requirement): the estimator's single
               decision constant k (platform judgement.k), the measured surface / aim position (the estimate's root-line and plate-top
               sigmas) and the robot / TCP pose (the posture-dependent pose error documented in the calibration file, he_sigma_m). A
               source without a documented value is listed as missing (never invented) - r is then a lower bound and says so.

The geometric search (weld_clearance, exact for the tool mesh against the convex section prisms) orders and filters the candidates;
the platform's own check decides (MoveIt: the specimen in the planning scene) - a candidate it refuses is skipped and the next closest
one is tried, so the chosen orientation is one MoveIt accepted.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .weld_clearance import (OTHER_ROLE, PROCESS_ROLE, ToolBody, ToolRoles, body_digest, clearance, nearest_station, part_link,
                             pass_geometry, rows_geometry, rows_poses)
from .weld_line import LINE_RULE, line_step_m, rotation_mean, station_rotations, weld_line
from .weld_torch import ROLLS_DEG, plan_weld_torch, roll_matrix

__all__ = ["ANGLE_STEP_DEG", "PROCESS_REQUIRED_MM", "PROCESS_REQUIREMENT_RULE", "ClearanceRequirement", "clearance_requirement",
           "pass_tilt", "torch_axis", "retarget_pass", "pass_clearance", "candidate_grid", "search_pass", "choose_orientation",
           "ORIENTATION_RULE", "LineRows", "line_rows", "refusal_of", "poses_role_clearance", "NECK_RULE", "body_axis_tcp",
           "neck_side", "roll_for_travel", "rule_roll", "axis_travel_deg"]

# The grid of the candidate torch angles: the WPS gives the travel and work angles in whole degrees (KIRO's dataset likewise)
ANGLE_STEP_DEG = 1.0
# search_pass's Lipschitz skipping compares the candidates' torch frames (R0) by their rotation angle; a rotation turns every vector by
# at most its angle, so a candidate whose torch AXIS is farther than the bound from the killer's cannot be in reach - only the others'
# frames are computed (the same function, on demand). The comparison of the axes allows this much more [rad] than the bound: far above
# the floating-point error of either angle (the rotation angle from a trace: ~1e-8 rad near 0; the axis angle: ~1e-15), far below
# the grid step - it only widens the set computed, never the set skipped (the same candidates are skipped as with every frame known).
_AXIS_PREFILTER_RAD = 1e-4
# The process parts' required clearance (user 2026-10-13, KIRO: in the root pass the gas nozzle almost touches the plate): the gas
# nozzle and the contact tip are designed to approach the work - they must only not penetrate the nominal workpiece (KIRO's practice,
# not a tuned value); every other part keeps k x the combined sigma (ClearanceRequirement.required_mm).
PROCESS_REQUIRED_MM = 0.0
PROCESS_REQUIREMENT_RULE = ("process parts (the gas nozzle and the contact tip - weld_clearance.tool_roles): no interpenetration with "
                            "the nominal workpiece (required 0 mm; KIRO 2026-10-13: in the root pass the gas nozzle almost touches the "
                            "plate - KIRO's practice); every other part (sensor, guard, dust plate, bracket, clamp, torch body / neck, "
                            "the wrist link, and in MoveIt the arm links): k x the combined sigma")
ORIENTATION_RULE = ("per pass the collision-free and reachable torch orientation closest to the WPS nominal (angle between the torch "
                    "axes; grid of the WPS angle resolution over the whole hemisphere, no bound on the deviation; equal distance -> the "
                    "larger margin), on the DELIVERED line (one straight line at one orientation - kiro/pass_<k>.yaml = result.paths); "
                    "feasible = the tool (URDF collision geometry rigid with the TCP, no wire stub) clears the "
                    "specimen before the pass (plates, groove, backing, deposited layers) at every sample of the line (and the cap's "
                    "weave edges) by the required "
                    "clearance of each part's role (gas nozzle + contact tip: 0 mm, no interpenetration - KIRO practice; every other "
                    "part: k x the combined sigma), and the platform's D14 (MoveIt: IK, limits, Cartesian path with collision checking) passes it; one TCP "
                    "roll per job, the torch neck bending with the travel angle (NECK_RULE: a nonzero travel angle is searched in its "
                    "roll only, every candidate keeps that roll's travel side), then the smaller total deviation, then the larger "
                    "joint-limit margin")
NECK_RULE = ("the torch neck bends with the travel angle (user 2026-10-13, process definition): the straight torch body (the flange "
             "axis, from the robot model's flange <- TCP) is inclined more than the nozzle axis and on the same side along the travel - "
             "push: the parts after the bend face the travel direction, drag: away from it; equivalently sign(x_TCP . travel) = "
             "sign(travel angle) for a body bent toward +x_TCP. A roll serves the travel angles of its side (0 fits both); a pass with "
             "a nonzero WPS travel angle is searched in that roll only, no flip to the other roll")


# --------------------------------------------------------------------------------------
# the torch neck and the roll (user 2026-10-13)
# --------------------------------------------------------------------------------------
def body_axis_tcp(torch) -> Optional[np.ndarray]:
    """The straight torch body's axis (toward the work: the flange +z) in the TCP frame, from the torch model's flange <- TCP (on this
    arm: the robot's TF) - our torch: (0.5, 0, 0.866). None = no torch model."""
    if torch is None or getattr(torch, "flange_from_tcp", None) is None:
        return None
    return np.asarray(torch.flange_from_tcp.R, dtype=float)[2, :].copy()


def _body_tip(torch, roll_deg: float) -> Optional[np.ndarray]:
    """The body axis in the plan's tip frame [x travel side, y, z torch axis] at TCP roll roll_deg (weld_torch.tcp_poses:
    R_tcp = R_tip Rz(roll) (TCP <- tip)^T)."""
    b = body_axis_tcp(torch)
    if b is None:
        return None
    return roll_matrix(float(roll_deg)) @ np.asarray(torch.tcp_from_tip.R, dtype=float).T @ b


def neck_side(torch, roll_deg: float) -> int:
    """+1 / -1: the body leans beyond the nozzle toward / against the travel at this roll (the sign of the body's tip-frame x - the
    tip frame's x is the travel side); 0 = no bend (a straight torch, to the float representation) or no torch model: both sides."""
    b = _body_tip(torch, roll_deg)
    if b is None or abs(float(b[0])) <= 1e-9 * max(1.0, float(np.linalg.norm(b))):
        return 0
    return 1 if float(b[0]) > 0.0 else -1


def roll_for_travel(torch, travel_deg: float, rolls=ROLLS_DEG) -> list:
    """The rolls that keep the torch neck bending with this travel angle (NECK_RULE): travel 0 or no bend → every roll."""
    t = float(travel_deg)
    if t == 0.0:
        return [float(r) for r in rolls]
    return [float(r) for r in rolls if neck_side(torch, r) in (0, 1 if t > 0.0 else -1)]


def rule_roll(torch, travel_degs, rolls=ROLLS_DEG) -> tuple:
    """The TCP roll the NECK_RULE gives a plan before any D14 search: the roll that keeps the torch neck bending with every pass's
    WPS travel angle (roll_for_travel); when the passes' travel sides disagree (push and drag in one plan) no roll serves them all -
    the roll serving the most passes, the others named (D14, when it runs, refuses them in that roll). When every roll fits (all
    travel angles 0, or a straight torch) the first of rolls. -> (roll_deg, record)."""
    t = [float(v) for v in travel_degs]
    allowed = {float(r): [k + 1 for k, tk in enumerate(t) if float(r) in roll_for_travel(torch, tk, rolls)] for r in rolls}
    best = max((float(r) for r in rolls), key=lambda r: (len(allowed[r]), -list(map(float, rolls)).index(r)))
    left_out = [k + 1 for k in range(len(t)) if (k + 1) not in allowed[best]]
    side = neck_side(torch, best)
    if all(len(v) == len(t) for v in allowed.values()):
        why = (f"roll {best:g} deg: every roll fits these travel angles (NECK_RULE) - "
               + ("no neck bend to orient" if side == 0 else "all WPS travel angles are 0"))
    else:
        why = (f"roll {best:g} deg: the torch neck bends with the WPS travel angle (NECK_RULE: "
               f"{'push' if side > 0 else 'drag'} side; the straight body beyond the nozzle on the same side along the travel)"
               + (f"; pass(es) {left_out} have the other travel side - no single roll serves them" if left_out else ""))
    return best, {"deg": best, "reason": why, "rule": NECK_RULE, "source": "the WPS travel angles (no D14 search yet)",
                  "neck_side": side, "passes_on_side": allowed[best], "passes_off_side": left_out}


def axis_travel_deg(axis, d) -> float:
    """The AWS travel angle of an axis (toward the work) for the travel direction d: + = leaning ahead (push), - = behind (drag)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    dd = np.asarray(d, dtype=float)
    dd = dd / np.linalg.norm(dd)
    return float(math.degrees(math.asin(float(np.clip(a @ dd, -1.0, 1.0)))))


def _neck_record(torch, ln) -> Optional[dict]:
    """The leans along the travel of the delivered line ln (weld_line.WeldLine with the torch): nozzle axis and straight body [deg]."""
    b = body_axis_tcp(torch)
    if b is None or getattr(ln, "R_tcp", None) is None:
        return None
    d = np.asarray(ln.tip_goal, dtype=float) - np.asarray(ln.tip_start, dtype=float)
    if not float(np.linalg.norm(d)) > 0.0:
        return None
    R = np.asarray(ln.R_tcp, dtype=float)
    nz, bd = axis_travel_deg(R[:, 2], d), axis_travel_deg(R @ b, d)
    return {"nozzle_lean_deg": nz, "body_lean_deg": bd,
            "with_travel": bool((nz == 0.0) or (np.sign(bd) == np.sign(nz) and abs(bd) > abs(nz)))}


@dataclass
class ClearanceRequirement:
    k: float
    sources: list                              # [{name, sigma_mm (None = not documented), source}]
    required_mm: float = float("nan")
    sigma_mm: float = float("nan")
    missing: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"k": float(self.k), "sigma_mm": float(self.sigma_mm), "required_mm": float(self.required_mm),
                "rule": "required clearance = k x root-sum-square of the documented sigmas (k = the estimator's decision constant)",
                "sources": [dict(s) for s in self.sources], "missing": list(self.missing),
                "complete": not self.missing,
                "by_role": {"process": {"required_mm": PROCESS_REQUIRED_MM, "parts": "gas nozzle + contact tip"},
                            "other": {"required_mm": float(self.required_mm), "parts": "every other tool part and the arm links"},
                            "rule": PROCESS_REQUIREMENT_RULE}}

    def describe(self) -> str:
        parts = ", ".join(f"{s['name']} {s['sigma_mm']:.2f}" for s in self.sources if s.get("sigma_mm") is not None)
        miss = f"; not documented: {', '.join(self.missing)} (r is a lower bound)" if self.missing else ""
        return (f"{self.required_mm:.2f} mm = {self.k:g} x {self.sigma_mm:.2f} mm ({parts} mm){miss}; the gas nozzle and contact tip "
                f"{PROCESS_REQUIRED_MM:.2f} mm (no interpenetration, KIRO practice)")


def clearance_requirement(k: float, sources: list) -> ClearanceRequirement:
    """sources = [(name, sigma_m or None, source text)] → ClearanceRequirement (sigma = root-sum-square of the documented ones)."""
    rows, miss, v = [], [], 0.0
    for name, sig, src in sources:
        if sig is None or not math.isfinite(float(sig)):
            rows.append({"name": str(name), "sigma_mm": None, "source": str(src)})
            miss.append(str(name))
            continue
        rows.append({"name": str(name), "sigma_mm": float(sig) * 1e3, "source": str(src)})
        v += (float(sig) * 1e3) ** 2
    s = math.sqrt(v)
    return ClearanceRequirement(k=float(k), sources=rows, required_mm=float(k) * s, sigma_mm=s, missing=miss)


# --------------------------------------------------------------------------------------
# pass geometry
# --------------------------------------------------------------------------------------
def pass_tilt(plan, wp) -> float:
    """The w multiplier of the pass's work-angle tilt direction (plan_weld): one bead / cap / middle bead = toward the bevel plate,
    the wall beads of a multi-bead layer toward their wall."""
    bev = 1.0 if str(plan.bevel_side) == "right" else -1.0
    if wp.kind == "cap" or int(wp.n_beads) == 1:
        return bev
    if int(wp.bead) == 1:
        return -1.0
    if int(wp.bead) == int(wp.n_beads):
        return 1.0
    return bev


def _frame(plan, wp) -> tuple:
    st = plan.stations[0]
    d = st.u if wp.direction == "S->E" else -st.u
    return np.asarray(st.n, dtype=float), np.asarray(st.w, dtype=float), np.asarray(d, dtype=float), pass_tilt(plan, wp) * np.asarray(st.w, dtype=float)


def torch_axis(n, d, tilt, travel_deg, work_deg) -> np.ndarray:
    """The torch axis (toward the work) of AWS travel / work angles (weld_plan._tcp_R); arrays broadcast."""
    T, W = np.radians(np.asarray(travel_deg, dtype=float)), np.radians(np.asarray(work_deg, dtype=float))
    p = -np.cos(W)[..., None] * n + np.sin(W)[..., None] * tilt
    return np.sin(T)[..., None] * d + np.cos(T)[..., None] * p


def _nominal(wp) -> tuple:
    t0 = float(getattr(wp, "travel_nominal_deg", float("nan")))
    w0 = float(getattr(wp, "work_nominal_deg", float("nan")))
    return (t0 if math.isfinite(t0) else float(wp.travel_angle_deg), w0 if math.isfinite(w0) else float(wp.work_angle_deg))


def retarget_pass(plan, wp, travel_deg: float, work_deg: float):
    """The pass with the torch at (travel, work) [deg] (work signed: + = toward the pass's tilt side) in each station's own section
    frame (weld_line.station_rotations, as plan_weld): rotations, the cap's weave amplitude in the layer plan (the WPS amplitude along
    the torch y projected on the section) and the dense path rebuilt. Its delivered line (weld_line.weld_line) has the rotation mean
    of exactly these rotations - the rotation the orientation search evaluated for this candidate."""
    from .weld_line import station_rotations                          # noqa: PLC0415
    from .weld_plan import _densify, end_insets_mm                    # noqa: PLC0415
    Rs = station_rotations(plan, wp, travel_deg, work_deg)
    by_s = {float(st.s_mm): st for st in plan.stations}
    amps = ([float(plan.process.weave_amplitude_mm) * abs(float(R[:, 1] @ by_s[float(sv)].w)) for sv, R in zip(wp.s_mm, Rs)]
            if plan.process.weave_amplitude_mm is not None else [])
    t0, w0 = _nominal(wp)
    new = dataclasses.replace(wp, R_base_tcp=Rs, travel_angle_deg=float(travel_deg),
                              work_angle_deg=float(work_deg), travel_nominal_deg=t0, work_nominal_deg=w0,
                              orientation=dict(getattr(wp, "orientation", {}) or {}))
    if wp.kind == "cap" and wp.weave_amp_mm is not None and amps:
        new.weave_amp_mm = np.asarray(amps, dtype=float)
    _densify(new, plan.stations, plan.process, end_insets_mm(plan))
    return new


def _role_bodies(body, required_mm: float) -> list:
    """[(role, ToolBody, required clearance [mm])]: a ToolRoles → its process parts (gas nozzle + contact tip) at PROCESS_REQUIRED_MM
    and its other parts at required_mm (k x sigma); a plain ToolBody → one role 'tool' at required_mm."""
    if isinstance(body, ToolRoles):
        return [(r, b, PROCESS_REQUIRED_MM if r == PROCESS_ROLE else float(required_mm)) for r, b in body.roles()]
    return [("tool", body, float(required_mm))]


def _combine(recs: dict, reqs: dict) -> dict:
    """{role: clearance record} → one record: excess_mm = min over the roles of (clearance - its required clearance) (>= 0 = clears
    every requirement); the limiting role's fields (station, solid, kind, point) at the top; clearance_mm = the other parts'
    clearance (the k x sigma comparison; a plain tool: the tool's); by_role = each role's clearance, requirement and closest point."""
    lim = min(recs, key=lambda r: recs[r]["clearance_mm"] - reqs[r])
    out = dict(recs[lim])
    main = OTHER_ROLE if OTHER_ROLE in recs else lim
    out.update(limiting_role=lim, excess_mm=float(recs[lim]["clearance_mm"] - reqs[lim]),
               below=any(bool(c_.get("below")) for c_ in recs.values()), intersects=any(bool(c_.get("intersects")) for c_ in recs.values()),
               clearance_mm=float(recs[main]["clearance_mm"]),
               by_role={r: {"clearance_mm": float(c_["clearance_mm"]), "required_mm": float(reqs[r]), "below": bool(c_.get("below")),
                            "point_tcp_mm": c_.get("point_tcp_mm"), "solid": c_.get("solid"), "kind": c_.get("kind"),
                            "s_mm": c_.get("s_mm"), "station": c_.get("station"), "line": c_.get("line"),
                            "sample": c_.get("sample"), "line_s_mm": c_.get("line_s_mm")} for r, c_ in recs.items()})
    return out


@dataclass
class LineRows:
    """A pass's delivered-line samples as clearance rows (weld_line module docstring): the wire-end samples (independent of the torch
    orientation), the station each sample is evaluated in (nearest to its wire end) and the geometry with one row per sample."""
    geo: object                    # the pass's PassGeometry (stations)
    rows: object                   # rows_geometry(geo, station)
    station: np.ndarray            # (n,) station index per sample
    tip: np.ndarray                # (n, 3) the wire-end samples [m, base]
    f: np.ndarray                  # (n,) fraction along the line
    s_mm: np.ndarray               # (n,) the line's seam coordinate
    ends: dict                     # weld_line.pass_endpoints
    step_m: float


def line_rows(plan, wp, step_m: float, *, geo=None) -> LineRows:
    """The rows of pass wp's delivered line at the sample step step_m (weld_line.WeldLine.samples)."""
    geo = geo or pass_geometry(plan, wp)
    ln = weld_line(plan, wp)                                         # the wire-end line (no torch: the same for every orientation)
    smp = ln.samples(step_m)
    st = nearest_station(geo, smp["tip"])
    return LineRows(geo=geo, rows=rows_geometry(geo, st), station=st, tip=smp["tip"], f=smp["f"], s_mm=smp["s_mm"], ends=ln.ends,
                    step_m=float(step_m))


def _line_record(ln, n: int, step_m: float) -> dict:
    """The line poses a D14 evaluation checked (base frame): TCP start / goal, rotation, wire end, samples."""
    return {"tcp_start_m": np.asarray(ln.tcp_start, dtype=float).copy(), "tcp_goal_m": np.asarray(ln.tcp_goal, dtype=float).copy(),
            "R_base_tcp": np.asarray(ln.R_tcp, dtype=float).copy(), "tip_start_m": np.asarray(ln.tip_start, dtype=float).copy(),
            "tip_goal_m": np.asarray(ln.tip_goal, dtype=float).copy(), "roll_deg": float(ln.roll_deg), "n_samples": int(n),
            "step_mm": float(step_m) * 1e3, "weave": None if ln.weave is None else {k_: v_ for k_, v_ in ln.weave.items()}}


def _eval_key(roles: list, lr: LineRows, ln) -> str:
    """The digest of every input of the exact per-role values of a delivered line (_eval_line, thr False): each role's tool geometry
    (weld_clearance.body_digest) and required clearance, the rows (the sections the samples are evaluated in: planes, plate ends,
    frames; the samples' stations and seam coordinates), the line's TCP samples, its rotation and the lines the torch runs (the cap's
    weave edges). Those values are a deterministic function of exactly these inputs: two evaluations with the same key are the same
    evaluation - the platform's check reuses the search's record of the candidate it checks instead of repeating it
    (pass_clearance(known=...))."""
    h = hashlib.sha256(b"weld_orient._eval_line exact values v1")

    def put(a, dt=np.float64):
        a = np.ascontiguousarray(a, dtype=dt)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    for role, b, r_role in roles:
        h.update(str(role).encode())
        h.update(body_digest(b).encode())
        put(float(r_role))
    rw = lr.rows
    for nm, pl in zip(rw.names, rw.planes):
        h.update(str(nm).encode())
        for a in pl:
            put(a)
    for a in (rw.ylo, rw.yhi, rw.A, rw.origins, rw.s_mm, lr.s_mm):
        put(a)
    put(lr.station, np.int64)
    put(ln.samples(lr.step_m)["tcp"])
    put(ln.R_tcp)
    for name, off in ln.offsets():
        h.update(str(name).encode())
        put(off)
    return h.hexdigest()


def _eval_line(roles: list, lr: LineRows, ln, *, thr: bool, order: Optional[dict] = None, reach=None) -> dict:
    """The exact per-role clearance of the delivered line ln (weld_line.WeldLine with the torch) at the rows lr: every line the torch
    runs (the centre; the cap's weave edges), every sample in its row's section. thr True = the feasibility test (each role against
    its requirement, stops at the first line / role below; order[role] = the rows to try first, updated), False = the exact values.
    reach (thr only): a sample whose TCP the elbow cannot reach ends the test (kind 'reach'). → the worst line's record (_combine +
    'line', 'n_samples')."""
    smp = ln.samples(lr.step_m)
    P0 = smp["tcp"]
    n = len(P0)
    if n != len(lr.station):
        raise ValueError("the line's samples do not match its rows")
    worst = None
    for name, off in ln.offsets():
        P = P0 + np.asarray(off, dtype=float)[None, :]
        if reach is not None and thr:
            v = reach.violation(np.broadcast_to(ln.R_tcp, (n, 3, 3)), P)
            if v is not None:
                i_bad = int(v["index"])
                return {"clearance_mm": float("nan"), "station": int(lr.station[i_bad]), "s_mm": float(lr.geo.s_mm[lr.station[i_bad]]),
                        "line_s_mm": float(lr.s_mm[i_bad]), "solid": "", "kind": "reach", "point_tcp_mm": None, "below": True,
                        "intersects": False, "n_stations": 0, "line": name, "excess_mm": float("nan"), "by_role": {},
                        "limiting_role": "reach", "reach": v, "n_samples": n}
        R_sec, P_sec = rows_poses(lr.rows, ln.R_tcp, P)
        recs, reqs = {}, {}
        for role, b, r_role in roles:
            c = clearance(b, lr.rows, R_sec, P_sec, threshold=r_role if thr else None, order=None if order is None else order.get(role))
            j = int(c.get("station", -1))                                    # the row (sample) index
            c.update(line=name, sample=j, station=int(lr.station[j]) if j >= 0 else -1,
                     line_s_mm=float(lr.s_mm[j]) if j >= 0 else float("nan"))
            if thr and c["below"] and j >= 0 and order is not None:
                order[role] = np.concatenate([[j], [k for k in range(n) if k != j]])
            recs[role], reqs[role] = c, r_role
            if thr and c["below"]:
                break                                                       # one role below: the candidate is out
        c = _combine(recs, reqs)
        c["n_samples"] = n
        if worst is None or c["excess_mm"] < worst["excess_mm"]:
            worst = c
        if thr and c["below"]:
            break
    return worst


def pass_clearance(plan, wp, body, torch, roll_deg: float, *, geo=None, threshold: Optional[float] = None,
                   required_mm: Optional[float] = None, step_m: Optional[float] = None, rows: Optional[LineRows] = None,
                   known: Optional[dict] = None) -> dict:
    """Clearance [mm] of the pass's DELIVERED line (weld_line.weld_line: the straight line at one orientation that kiro/pass_<k>.yaml
    carries, its rotation the mean of the pass's station torch frames, TCP roll roll_deg; the cap's weave edges) → the worst line's
    record (_combine: by_role, excess_mm, limiting_role; 'line', 'sample', 'line_s_mm', 'n_samples', 'checked_line' = the poses
    evaluated). body = a ToolBody or a ToolRoles (the process parts at PROCESS_REQUIRED_MM, the other parts at required_mm).
    threshold (the other parts' requirement) = the feasibility test (stops at the first line below), None = the exact values.
    step_m = the sample step (default the plan's recorded line_step_mm, else weld_line.line_step_m). The exact values carry 'eval_key'
    (_eval_key: the digest of every input); known = such a record already computed (the orientation search's record of the candidate
    a platform check is given - choose_orientation): when its key is this evaluation's key it IS this evaluation and is returned
    instead of repeating it (any other record is ignored)."""
    st_m = _plan_step_m(plan, step_m)
    lr = rows or line_rows(plan, wp, st_m, geo=geo)
    req = float(required_mm if required_mm is not None else (threshold if threshold is not None else 0.0))
    roles = _role_bodies(body, req)
    ln = weld_line(plan, wp, torch, roll_deg, ends=lr.ends)
    key = _eval_key(roles, lr, ln) if threshold is None else None
    if key is not None and isinstance(known, dict) and known.get("eval_key") == key:
        c = copy.deepcopy(known)                                       # the same evaluation (every input equal): not repeated
    else:
        c = _eval_line(roles, lr, ln, thr=threshold is not None, order={} if threshold is not None else None)
        if key is not None:
            c["eval_key"] = key
    c["checked_line"] = _line_record(ln, len(lr.tip), lr.step_m)
    return c


def poses_role_clearance(roles, geo, R: np.ndarray, P_tcp: np.ndarray, tip_tcp_m, required_mm: float) -> dict:
    """The exact per-role check of TCP poses at one base rotation (R, P_tcp (n, 3)) against a specimen geometry (weld_clearance.
    PassGeometry, e.g. the bare specimen): each pose in the section of the station nearest its wire end
    (as the delivered line's samples, line_rows); process parts (gas nozzle + contact tip) required PROCESS_REQUIRED_MM (0: no
    interpenetration), every other part required_mm. → {ok, by_role {role: {clearance_mm, required_mm, s_mm, solid, kind, link,
    point_tcp_mm, sample}}, limiting_role, excess_mm, n_samples}."""
    from .weld_clearance import poses_clearance                       # noqa: PLC0415
    P = np.asarray(P_tcp, dtype=float).reshape(-1, 3)
    tip = np.asarray(tip_tcp_m, dtype=float)
    st = nearest_station(geo, P + (np.asarray(R, dtype=float) @ tip)[None, :])
    by, excess, lim = {}, math.inf, None
    for role, body, req in _role_bodies(roles, float(required_mm)):
        c = poses_clearance(body, geo, R, P, st)
        link = part_link(body, c.get("point_tcp_mm")) if c.get("point_tcp_mm") is not None else ""
        by[role] = {"clearance_mm": float(c["clearance_mm"]), "required_mm": float(req), "s_mm": c.get("s_mm"),
                    "solid": c.get("solid"), "kind": c.get("kind"), "link": link, "point_tcp_mm": c.get("point_tcp_mm"),
                    "sample": c.get("sample"), "intersects": bool(c.get("intersects"))}
        e = float(c["clearance_mm"]) - float(req)
        if e < excess:
            excess, lim = e, role
    return {"ok": bool(excess >= 0.0), "by_role": by, "limiting_role": lim, "excess_mm": float(excess), "n_samples": int(len(P))}


def _plan_step_m(plan, step_m: Optional[float] = None) -> float:
    """The line sample step [m]: the given one, else the plan's recorded D14 step (plan.line_step_mm), else weld_line.line_step_m."""
    if step_m is not None:
        return float(step_m)
    v = float(getattr(plan, "line_step_mm", float("nan")))
    return v * 1e-3 if math.isfinite(v) and v > 0.0 else line_step_m(plan)[0]


def _lean_at(torch, roll_deg: float, travel_deg: float) -> float:
    """The straight body's lean along the travel [deg] at a travel angle (work 0, in the travel plane): travel + the bend toward the
    tip frame's x at this roll."""
    b = _body_tip(torch, roll_deg)
    return float(travel_deg) + (math.degrees(math.atan2(float(b[0]), float(b[2]))) if b is not None else 0.0)


def _neck_refusal(torch, roll_deg: float, travel_deg: float, rolls) -> str:
    """Why a pass is not searched in this roll (NECK_RULE)."""
    good = [r for r in rolls if float(r) in roll_for_travel(torch, travel_deg, rolls)]
    kind = "push" if travel_deg > 0.0 else "drag"
    return (f"the torch neck must bend with the travel angle (NECK_RULE): the WPS travel {travel_deg:+.0f} deg ({kind}) needs roll "
            + (" or ".join(f"{float(r):g}" for r in good) or "none") + " deg"
            + (f" (the straight body {_lean_at(torch, good[0], travel_deg):+.0f} deg along the travel, beyond the nozzle on its side)"
               if good else "")
            + f"; at roll {float(roll_deg):g} deg the body would lean {_lean_at(torch, roll_deg, travel_deg):+.0f} deg - the neck bent "
              "against the travel angle: not searched in this roll (no flip)")


def candidate_grid(t0: float, w0: float, n, d, tilt, travel_side: int = 0) -> tuple:
    """(travel, work, deviation [deg]) of every grid candidate, closest first: travel = t0 + i·step, work = w0 + j·step inside
    (-90, 90) deg each; deviation = the angle between the candidate's torch axis and the nominal one. travel_side +1 / -1 = only the
    travel angles of that side and 0 (the roll's side, NECK_RULE), 0 = all."""
    st = ANGLE_STEP_DEG
    ti = np.arange(math.ceil((-90.0 - t0) / st + 1e-9), math.floor((90.0 - t0) / st - 1e-9) + 1)
    wj = np.arange(math.ceil((-90.0 - w0) / st + 1e-9), math.floor((90.0 - w0) / st - 1e-9) + 1)
    T, W = np.meshgrid(t0 + ti * st, w0 + wj * st, indexing="ij")
    T, W = T.ravel(), W.ravel()
    if travel_side:
        keep = T * float(travel_side) >= 0.0
        T, W = T[keep], W[keep]
    a = torch_axis(n, d, tilt, T, W)
    k0 = np.flatnonzero((T == t0) & (W == w0))                    # the nominal is a grid point (i = j = 0): its axis from the same
    a0 = a[k0[0]] if len(k0) else torch_axis(n, d, tilt, t0, w0)  # evaluation, so its deviation is exactly 0
    # the angle between two unit vectors as atan2(|a x a0|, a . a0): well conditioned near 0 (arccos of a dot product near 1 turns a
    # rounding error of 1e-16 into ~1e-6 deg - the nominal then did not read as the nominal for some section frames)
    dev = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, a0), axis=-1), a @ a0))
    o = np.lexsort((np.abs(W - w0), np.abs(T - t0), np.round(dev, 9)))
    return T[o], W[o], dev[o]


def _reject_key(c: dict) -> tuple:
    """(constraint, part) of a rejected candidate's record (search_pass)."""
    if c.get("kind") == "reach":
        return "reach", str((c.get("reach") or {}).get("joint") or "")
    return "clearance", str(c.get("limiting_role") or "")


class _Rejections:
    """The candidates the search rejected, by constraint (the 'why' of a deviation from the WPS nominal): counts and the closest example
    of each (constraint, part); the nominal's own rejection. Cumulative, small (no per-candidate list)."""

    def __init__(self, roles: list):
        self.by: dict = {}
        self.nominal: Optional[dict] = None
        self.n = 0
        self._bodies = {r_: b_ for r_, b_, _q in roles}

    def add(self, rec: dict) -> None:
        key = rec["constraint"] + ("" if not rec.get("part") else f" / {rec['part']}")
        e = self.by.setdefault(key, {"constraint": rec["constraint"], "part": rec.get("part") or "", "n": 0, "closest": None})
        e["n"] += 1
        self.n += 1
        if e["closest"] is None:
            e["closest"] = rec
        if rec.get("nominal"):
            self.nominal = rec

    def clearance(self, T, W, dev, c: dict, nominal: bool, exact: bool = False) -> None:
        """A candidate rejected by the clearance or the reach test (c: _eval_line's record). exact False = the feasibility test's
        record (the first sample found below the requirement, not the deepest)."""
        kind, part = _reject_key(c)
        rec = {"travel_deg": float(T), "work_deg": float(W), "deviation_deg": float(dev), "nominal": bool(nominal)}
        if kind == "reach":
            v = c.get("reach") or {}
            rec.update(constraint="reach", part=str(v.get("joint") or ""), line=c.get("line"), line_s_mm=c.get("line_s_mm"),
                       detail=(f"the arm cannot reach it: the wrist centre {v.get('side', '')} of the elbow ({v.get('joint')}) limits "
                               f"({float(v.get('rho_mm', float('nan'))):.0f} mm from the 2nd axis, shell {float(v.get('lo_mm', float('nan'))):.0f}"
                               f" .. {float(v.get('hi_mm', float('nan'))):.0f} mm) at the {c.get('line')} line, s "
                               f"{float(c.get('line_s_mm', float('nan'))):.0f} mm"))
        else:
            role = str(c.get("limiting_role") or "")
            br = (c.get("by_role") or {}).get(role) or {}
            b = self._bodies.get(role)
            link = part_link(b, br.get("point_tcp_mm")) if b is not None and br.get("point_tcp_mm") is not None else ""
            v_, r_ = float(br.get("clearance_mm", float("nan"))), float(br.get("required_mm", float("nan")))
            rec.update(constraint="clearance", part=role, link=link, solid=br.get("solid"), line=br.get("line"),
                       line_s_mm=br.get("line_s_mm"), clearance_mm=br.get("clearance_mm"), required_mm=br.get("required_mm"),
                       point_tcp_mm=br.get("point_tcp_mm"), exact=bool(exact),
                       detail=((f"the tool's {role} parts ({link or '-'}) come within {v_:.2f} mm of the {br.get('solid')} (required "
                                f"{r_:.2f} mm; the least over the line, at the {br.get('line')} line, s "
                                f"{float(br.get('line_s_mm', float('nan'))):.0f} mm)") if exact else
                               (f"the tool's {role} parts ({link or '-'}) below the required {r_:.2f} mm from the {br.get('solid')} "
                                f"(first sample found: {v_:.2f} mm at the {br.get('line')} line, s "
                                f"{float(br.get('line_s_mm', float('nan'))):.0f} mm)")))
        self.add(rec)

    def bound(self, T, W, dev, killer: dict, nominal: bool) -> None:
        """A candidate ruled out without evaluation: a tool vertex of a closer rejected candidate cannot clear its requirement within
        the angle between them (the exact Lipschitz skipping)."""
        self.add({"travel_deg": float(T), "work_deg": float(W), "deviation_deg": float(dev), "nominal": bool(nominal),
                  "constraint": "clearance", "part": killer.get("part", ""), "link": killer.get("link", ""), "solid": killer.get("solid"),
                  "by_bound_from": [killer.get("travel_deg"), killer.get("work_deg")],
                  "detail": (f"the tool's {killer.get('part')} parts ({killer.get('link') or '-'}) cannot clear the {killer.get('solid')} "
                             f"(bound from travel {float(killer.get('travel_deg')):+.0f} work {float(killer.get('work_deg')):+.0f} deg)")})

    def snapshot(self) -> dict:
        return {"n": self.n, "by": {k_: dict(v_) for k_, v_ in self.by.items()}, "nominal": self.nominal}


def search_pass(plan, wp, body, torch, roll_deg: float, required_mm: float, *, geo=None, log=None, reach=None,
                max_deviation_deg: Optional[float] = None, step_m: Optional[float] = None, rows: Optional[LineRows] = None):
    """Generator: the pass's candidates whose DELIVERED line (weld_line: one straight line at one orientation - the rotation mean of
    the pass's station torch frames at the candidate's travel / work angles, TCP roll roll_deg) clears the specimen by the required
    clearance of every tool part's role (body = a ToolRoles: the gas nozzle and contact tip by PROCESS_REQUIRED_MM, the other parts by
    required_mm; a plain ToolBody: all by required_mm) at every sample of every line (the cap's weave edges included), closest to the
    nominal first (equal distance → the larger margin over the requirement) → dicts {travel_deg, work_deg, deviation_deg, clearance
    (record with by_role, excess_mm, limiting_role; checked_line = the poses evaluated; eval_key - _eval_key), rejected (the candidates
    rejected so far, by constraint: _Rejections.snapshot), n_evaluated, pass (retarget_pass of the candidate: the pass the platform
    check is given)}. Ends when the grid is exhausted (the last dict then has 'exhausted': True, the best margin bound seen and
    'rejected').
    Exact skipping: the wire-end samples do not depend on the orientation, and the line's rotation of a candidate is Q . R0(candidate)
    for one rotation Q per pass (R0 = the torch frame in the first station's section frame: the stations' section frames are rotations
    of it, and the rotation mean of Q_i R0 is mean(Q_i) R0), so two candidates' line rotations differ by the angle between their R0.
    When a tool vertex at distance d from the wire end (about which the torch turns) has g = v at a candidate, it has g <= v + theta
    (d + a) at any candidate turned by theta from it (g is 1-Lipschitz; the point moves by at most theta d, a weave edge by at most
    theta a, a = the WPS weave amplitude along y_TCP) - candidates with v + theta (d + a) < required are not feasible either.
    reach (weld_reach.ArmReach, the platform's arm): a candidate whose line puts the wrist centre outside the elbow's shell at any
    sample has no IK solution there - not feasible (a necessary condition of the robot check, exact).
    max_deviation_deg: the search ends ('exhausted' with 'cut_deg') past this deviation - choose_orientation's bound when another roll
    already has every pass feasible at that total deviation: a candidate farther than it cannot make this roll win (exact)."""
    from .weld_plan import _tcp_R                                 # noqa: PLC0415
    lr = rows or line_rows(plan, wp, _plan_step_m(plan, step_m), geo=geo)
    n, w, d, tilt = _frame(plan, wp)
    t0, w0 = _nominal(wp)
    T, W, dev = candidate_grid(t0, w0, n, d, tilt, travel_side=neck_side(torch, roll_deg))   # the roll's travel side (NECK_RULE)
    nom0 = bool(len(T)) and float(T[0]) == float(t0) and float(W[0]) == float(w0)      # the nominal is in this roll's grid
    # R0 per candidate (_tcp_R; (C, 3, 3)), computed on demand by kill_around (the rows never needed stay unknown, NaN)
    Rall = np.full((len(T), 3, 3), np.nan)
    R_have = np.zeros(len(T), dtype=bool)
    ax_all = torch_axis(n, d, tilt, T, W)                          # (C, 3) the candidates' torch axes (kill_around's prefilter)

    def frames(idx):
        """Rall rows idx computed (the same _tcp_R for every candidate)."""
        idx = np.asarray(idx, dtype=np.int64)
        for c_ in idx[~R_have[idx]]:
            Rall[c_] = _tcp_R(n, w, d, tilt, float(T[c_]), float(W[c_]))
        R_have[idx] = True
    dead = np.zeros(len(T), dtype=bool)
    killer = np.full(len(T), -1, dtype=np.int64)
    killed_by: dict = {}
    tip_tcp_mm = (np.asarray(torch.tcp_from_tip.t, dtype=float) * 1e3 if torch is not None   # the aim point (wire end) in the TCP
                  else np.zeros(3))                                                           # frame: nozzle end + stick-out along z
    roles = _role_bodies(body, required_mm)
    rej = _Rejections(roles)
    order = {}
    best_seen = (-math.inf, None)
    n_eval = 0
    n_near = 0
    n_reach = 0
    i = 0
    N = len(T)
    w0_ = weld_line(plan, wp, torch, roll_deg, ends=lr.ends)
    a_lat = (float(w0_.weave["amplitude_mm"]) if (w0_.weave is not None and w0_.weave.get("rotates_with_torch")) else 0.0)

    def kill_around(m, c):
        """Mark the candidates that the vertex results of c at candidate m rule out (docstring) - each role's vertex against its own
        required clearance."""
        if c is None:
            return
        for role, rc in (c.get("by_role") or {}).items():
            if rc.get("kind") != "vertex" or rc.get("point_tcp_mm") is None:
                continue
            v = float(rc["clearance_mm"])
            dist = float(np.linalg.norm(np.asarray(rc["point_tcp_mm"], dtype=float) - tip_tcp_mm)) + a_lat
            if not dist > 0.0:
                continue
            theta_max = (float(rc["required_mm"]) - v) / dist
            if theta_max <= 0.0:
                continue
            # only candidates whose torch axis is within theta_max (+ _AXIS_PREFILTER_RAD) of m's can be within theta_max as frames:
            # their frames computed; the others stay NaN (never hit - as with their frames known: farther than theta_max)
            am = ax_all[m]
            ang_ax = np.arctan2(np.linalg.norm(np.cross(ax_all, am), axis=-1), ax_all @ am)
            frames(np.append(np.flatnonzero((ang_ax <= theta_max + _AXIS_PREFILTER_RAD) & ~dead), m))
            with np.errstate(invalid="ignore"):
                cosang = np.clip((np.einsum("cij,ij->c", Rall, Rall[m]) - 1.0) / 2.0, -1.0, 1.0)
                hit = (np.arccos(cosang) < theta_max - 1e-12) & ~dead
            if hit.any():
                b = rej._bodies.get(role)
                killed_by[m] = {"travel_deg": float(T[m]), "work_deg": float(W[m]), "part": role, "solid": rc.get("solid"),
                                "link": part_link(b, rc.get("point_tcp_mm")) if b is not None else ""}
                killer[hit] = m
                dead[hit] = True

    def line_of(tt, ww):
        """The delivered line of the candidate (travel tt, work ww): weld_line with the rotation mean of the station torch frames."""
        return weld_line(plan, wp, torch, roll_deg, R_tip=rotation_mean(station_rotations(plan, wp, float(tt), float(ww))),
                         ends=lr.ends)

    def feasible(tt, ww, thr):
        """thr True = the feasibility test (each role against its requirement, stops at the first line below), None = exact values."""
        nonlocal n_reach
        ln = line_of(tt, ww)
        c = _eval_line(roles, lr, ln, thr=bool(thr), order=order if thr else None, reach=reach if thr else None)
        if c.get("kind") == "reach":
            n_reach += 1
        if not thr:
            c["eval_key"] = _eval_key(roles, lr, ln)                        # (the platform check reuses this exact record)
        c["checked_line"] = _line_record(ln, len(lr.tip), lr.step_m)
        return c

    cut = None
    while i < N:
        if max_deviation_deg is not None and dev[i] > float(max_deviation_deg) + 1e-9:
            cut = float(max_deviation_deg)                                   # the rest cannot make this roll win (branch and bound)
            break
        j = i
        while j + 1 < N and abs(dev[j + 1] - dev[i]) <= 1e-9:
            j += 1
        group = []
        for m in range(i, j + 1):
            if dead[m]:
                rej.bound(T[m], W[m], dev[m], killed_by.get(int(killer[m]), {}), nominal=(m == 0 and nom0))
                continue
            c = feasible(T[m], W[m], True)
            n_eval += 1
            if c is not None and not c["below"]:
                group.append(m)
            else:
                if c is not None:
                    if m == 0 and nom0 and c.get("kind") != "reach":                # the nominal's own record: the exact clearance
                        cx = feasible(T[m], W[m], None)
                        rej.clearance(T[m], W[m], dev[m], cx if cx.get("excess_mm", 0.0) < 0.0 else c, nominal=True,
                                      exact=cx.get("excess_mm", 0.0) < 0.0)
                    else:
                        rej.clearance(T[m], W[m], dev[m], c, nominal=(m == 0 and nom0))
                if c is not None and c.get("kind") == "vertex" and c["excess_mm"] > best_seen[0]:   # a tool vertex's g: an upper bound
                    best_seen = (c["excess_mm"], (float(T[m]), float(W[m])))
                if c is not None and c.get("kind") not in ("vertex", "reach"):
                    n_near += 1                                                     # inside the grown solid, no vertex in it
                kill_around(m, c)
        if group:
            vals = []
            for m in group:
                c = feasible(T[m], W[m], None)                                      # the exact clearance of the tied candidates
                vals.append((c["excess_mm"], m, c))
            vals.sort(key=lambda v: (-v[0], v[1]))
            for val, m, c in vals:
                if val < -1e-9:
                    rej.clearance(T[m], W[m], dev[m], c, nominal=(m == 0 and nom0), exact=True)
                    continue
                yield {"travel_deg": float(T[m]), "work_deg": float(W[m]), "deviation_deg": float(dev[m]), "clearance": c,
                       "n_evaluated": n_eval, "rejected": rej.snapshot(),
                       "pass": retarget_pass(plan, wp, float(T[m]), float(W[m]))}       # (computed where the search runs)
        i = j + 1
    yield {"exhausted": True, "n_evaluated": n_eval, "best_seen_mm": float(best_seen[0]) if best_seen[1] else float("nan"),
           "best_seen_at": best_seen[1], "n_near": n_near, "n_reach": n_reach, "cut_deg": cut,
           "reach": reach.describe() if reach is not None else None, "rejected": rej.snapshot()}


def _roll_record(torch, roll_deg: float, travel_deg: float, rolls, neck: Optional[dict]) -> dict:
    """The pass's roll and why (NECK_RULE), with the leans of the delivered line's nozzle and body (neck: _neck_record)."""
    side = neck_side(torch, roll_deg)
    t = float(travel_deg)
    if t != 0.0 and float(roll_deg) not in roll_for_travel(torch, t, rolls):
        why = _neck_refusal(torch, roll_deg, t, rolls)
    elif side == 0:
        why = f"roll {float(roll_deg):g} deg: the torch has no neck bend to orient (both rolls fit any travel angle)"
    elif t == 0.0:
        why = (f"roll {float(roll_deg):g} deg: the WPS travel is 0 deg - both rolls fit (NECK_RULE); this roll's side "
               f"{'push' if side > 0 else 'drag'} for any deviation")
    else:
        why = (f"roll {float(roll_deg):g} deg: the torch neck bends with the travel angle (NECK_RULE) - the {'push' if t > 0 else 'drag'} "
               f"travel {t:+.0f} deg needs the straight body beyond the nozzle on the same side")
    if neck is not None:
        why += (f"; on the delivered line nozzle {neck['nozzle_lean_deg']:+.1f} deg, body {neck['body_lean_deg']:+.1f} deg along the "
                "travel")
    return {"deg": float(roll_deg), "neck_side": side, "rule": NECK_RULE, "why": why, **(neck or {})}


def _plan_with(plan, chosen: dict):
    """A copy of the plan whose passes k in chosen are replaced (the others as they are)."""
    passes = [chosen.get(wp.k, wp) for wp in plan.passes]
    return dataclasses.replace(plan, passes=passes)


def refusal_of(row: dict) -> dict:
    """The robot check's refusal of a candidate as {constraint, part, links, detail}: the platform's own record (row['refusal']: the
    MoveIt adapter and the virtual arm give it) or, for a row without one, read from its reason text."""
    rf = dict(row.get("refusal") or {})
    why = str(row.get("reason") or "")
    if rf.get("constraint"):
        rf.setdefault("detail", why)
        return rf
    low = why.lower()
    links = []
    if "(contacts:" in why:
        links = [p.strip() for p in why.split("(contacts:", 1)[1].split(")", 1)[0].split(",") if p.strip()]
    if links:
        world = any(any(s_.startswith("hongik/") for s_ in p.split("-")) for p in links)
        kind = "arm vs world" if world else "self-collision"
    elif "no ik" in low or "did not converge" in low:
        kind = "IK"
    elif "urdf limit" in low or "joint limit" in low:
        kind = "joint limit"
    elif "joint-space jump" in low:
        kind = "joint jump"
    elif "parts come within" in low:
        kind = "clearance (exact model)"
    else:
        kind = "robot check"
    return {"constraint": kind, "part": "", "links": links, "detail": why}


def _deviation_why(t0: float, w0: float, c: Optional[dict], tries: list, chosen_dev: Optional[float], req_txt: str) -> dict:
    """Why the pass deviates from the WPS nominal (user 2026-10-13): the constraint that rejected the nominal and the closer (or equally
    close) candidates - the search's rejections by constraint (clearance by part role and link, the arm's reach by joint) and the robot
    check's refusals (IK, joint limit / jump, self-collision, arm vs world with the contact links) → {nominal, closer {n, by: [...]},
    text}."""
    snap = dict((c or {}).get("rejected") or {})
    by = [dict(v_) for v_ in (snap.get("by") or {}).values()]
    nominal = snap.get("nominal")
    robot: dict = {}
    for t_ in tries:
        rf = refusal_of(t_)
        key = "robot check / " + rf["constraint"]
        e = robot.setdefault(key, {"constraint": "robot check: " + rf["constraint"], "part": ", ".join(rf.get("links") or []),
                                   "n": 0, "closest": None})
        e["n"] += 1
        rec = {"travel_deg": t_["travel_deg"], "work_deg": t_["work_deg"], "deviation_deg": t_["deviation_deg"],
               "constraint": e["constraint"], "links": rf.get("links") or [], "detail": rf.get("detail") or t_.get("reason")}
        if e["closest"] is None:
            e["closest"] = rec
        if float(t_["deviation_deg"]) < 1e-9 and nominal is None:
            nominal = dict(rec, nominal=True)
    groups = by + list(robot.values())
    groups.sort(key=lambda g_: (-int(g_["n"]), float((g_.get("closest") or {}).get("deviation_deg", 0.0))))
    n_all = sum(int(g_["n"]) for g_ in groups)
    if chosen_dev is not None and chosen_dev < 1e-9:
        return {"nominal": None, "closer": {"n": 0, "by": []}, "text": "the WPS nominal"}

    def g_txt(g_):
        cl = g_.get("closest") or {}
        return (f"{g_['n']} by {g_['constraint']}" + (f" ({g_['part']})" if g_.get("part") else "")
                + (f" - closest travel {float(cl['travel_deg']):+.0f} work {float(cl['work_deg']):+.0f} deg: {cl.get('detail')}"
                   if cl else ""))
    nom_txt = (f"the WPS nominal (travel {t0:+.0f} work {w0:+.0f} deg) is out: " + (
        f"{nominal['constraint']}" + (f" ({nominal.get('part')})" if nominal.get("part") else "") + f" - {nominal.get('detail')}"
        if nominal else "not evaluated"))
    text = (nom_txt + (f"; {n_all} orientation(s) at or within the chosen deviation rejected: " + "; ".join(g_txt(g_) for g_ in groups)
                       if groups else "")
            + (f" (requirement: {req_txt})" if any(g_["constraint"] == "clearance" for g_ in groups) else ""))
    return {"nominal": nominal, "closer": {"n": n_all, "by": groups}, "text": text}


def choose_orientation(plan, check: Optional[Callable], *, body: ToolBody, required, torch=None, rolls=ROLLS_DEG,
                       log: Optional[Callable[[str], None]] = None, candidates: Optional[Callable] = None, reach=None,
                       step_m: Optional[float] = None) -> dict:
    """Per pass the closest feasible torch orientation of its DELIVERED line (weld_line: the one straight line at one orientation that
    kiro/pass_<k>.yaml and result.paths carry), one TCP roll for the job (module docstring). check(plan, only=[k...])
    = the platform's D14 for those passes (plan.torch_roll_deg and plan.line_step_mm read by it; rows {k, reachable, reason,
    min_limit_margin_rad, refusal, ...}); None = the geometric search only (no robot check: the rows say so). step_m = the line sample
    step (the platform's Cartesian step; default weld_line.line_step_m) - recorded as plan.line_step_mm. Fills plan.passes (the chosen
    orientations), plan.torch_roll_deg, plan.torch_roll, plan.feasibility (the chosen roll's D14 rows), plan.orientation (the record:
    per pass the nominal, the chosen, the deviation and WHY - the constraint that rejected the nominal and the closer candidates -,
    the clearance by role on the delivered line and the line poses checked) → that record.
    candidates(wp, roll, max_deviation_deg) → an iterator of search_pass's dicts (e.g. the same search in a worker process); default:
    search_pass here. An iterator with ask() (start computing its next dict, no wait) is asked for the next candidates of all the
    passes before they are needed - the first ones together, and while the robot check runs (the next one in case it refuses): the
    searches of the passes then run in parallel. Each search's sequence is its own (the same dicts in the same order whether asked or
    not), so this changes no result. The rolls are searched in turn; once a roll has every pass feasible, a later roll's searches stop past that total
    deviation (search_pass max_deviation_deg: a pass farther than the whole winning total cannot make the later roll win)."""
    req = required if isinstance(required, ClearanceRequirement) else None
    r_mm = float(req.required_mm if req is not None else required)
    roles = isinstance(body, ToolRoles)
    req_txt = (f"the required clearance of each part's role (gas nozzle + contact tip {PROCESS_REQUIRED_MM:.2f} mm, other parts "
               f"{r_mm:.2f} mm)" if roles else f"the required {r_mm:.2f} mm")
    role_body = {r_: b_ for r_, b_, _q in _role_bodies(body, r_mm)}
    wt = plan_weld_torch(plan, torch)
    if wt is None:
        raise ValueError("torch orientation search: no weld torch model")
    st_m, st_src = line_step_m(plan, step_m)
    plan.line_step_mm = st_m * 1e3
    rows = {wp.k: line_rows(plan, wp, st_m) for wp in plan.passes} if candidates is None else {}
    t_start = time.monotonic()
    tm = {"candidates_wait_s": 0.0, "robot_check_s": 0.0, "retarget_s": 0.0, "n_robot_checks": 0}

    def ask(it):                                   # start the iterator's next dict where it runs (candidates with ask())
        f = getattr(it, "ask", None)
        if callable(f):
            f()
    res = {}
    best_total = (math.inf, None)                  # the smallest total deviation of a roll with every pass feasible so far
    neck_refused: dict = {}                        # {roll: {k: why}} - the passes a roll does not serve (NECK_RULE)
    for roll in rolls:
        bound = best_total[0] if math.isfinite(best_total[0]) else None
        off = {}
        for wp in plan.passes:                     # the torch neck bends with the travel angle: a nonzero one is searched in its roll only
            t0_, _w0 = _nominal(wp)
            if float(roll) not in roll_for_travel(wt, t0_, rolls):
                off[wp.k] = _neck_refusal(wt, float(roll), t0_, rolls)
        neck_refused[float(roll)] = off
        gens = {wp.k: (search_pass(plan, wp, body, wt, float(roll), r_mm, rows=rows[wp.k], reach=reach, max_deviation_deg=bound)
                       if candidates is None else iter(candidates(wp, float(roll), bound))) for wp in plan.passes if wp.k not in off}
        cur, done, rows_, tries, failed, last = {}, {}, {}, {wp.k: [] for wp in plan.passes}, {}, {}
        by_k = {wp.k: wp for wp in plan.passes}
        for k_, why_ in off.items():
            failed[k_] = why_
            rows_[k_] = {"k": k_, "reachable": False, "reason": why_, "torch_roll_deg": float(roll), "refusal": {
                "constraint": "torch neck vs travel angle", "part": "", "links": [], "detail": why_}}

        def advance(k):
            t_ = time.monotonic()
            c = next(gens[k])
            tm["candidates_wait_s"] += time.monotonic() - t_
            last[k] = c
            if c.get("exhausted") and c.get("cut_deg") is not None:
                failed[k] = (f"not searched further: no orientation within {c['cut_deg']:.1f} deg of the WPS clears the specimen "
                             f"({req_txt}) - roll {best_total[1]:g} deg has every pass feasible at a total deviation of "
                             f"{best_total[0]:.1f} deg")
                return None
            if c.get("exhausted"):
                seen = c.get("best_seen_at")
                failed[k] = (f"no torch orientation at any deviation is feasible: "
                             + (f"{len(tries[k])} candidate(s) that clear the specimen by {req_txt} refused by the robot check "
                                f"(last: {tries[k][-1]['reason']})" if tries[k] else
                                f"none of the {c['n_evaluated']} orientations evaluated (the others ruled out by the Lipschitz bound) "
                                f"clears the specimen by {req_txt}"
                                + (f"; {c.get('n_near')} of them reach into the grown specimen with no tool vertex inside it (clearance "
                                   f"below the requirement, not resolved)" if c.get("n_near") else "")
                                + (f"; the best tool-vertex bound of the margin over the requirement {c['best_seen_mm']:+.2f} mm at "
                                   f"travel {seen[0]:+.0f} work {seen[1]:+.0f} deg" if seen else "")
                                + (f"; {c['n_reach']} ruled out by the arm's reach ({c.get('reach')})" if c.get("n_reach") else "")))
                return None
            cur[k] = c
            return c
        pending = []
        for k_ in gens:                            # the first candidates of every pass together
            ask(gens[k_])
        for wp in plan.passes:
            if wp.k not in off and advance(wp.k) is not None:
                pending.append(wp.k)
        unchecked = None
        while pending:
            lb = sum(done[k][0]["deviation_deg"] for k in done) + sum(cur[k]["deviation_deg"] for k in pending)
            search_failed = set(failed) - set(off)                   # (passes this roll does not serve do not stop the others)
            beaten = bool(off) and math.isfinite(best_total[0])      # not every pass here, another roll has them all
            if search_failed or beaten or lb > best_total[0] + 1e-9:   # this roll cannot win: stop (the candidates only grow)
                why_cut = ("not searched further: another pass of this roll has no feasible orientation" if search_failed else
                           f"not searched further: {len(off)} pass(es) are not this roll's travel side, roll {best_total[1]:g} deg "
                           f"has every pass feasible" if beaten else
                           f"not searched further: this roll's total deviation is already >= {lb:.1f} deg, roll "
                           f"{best_total[1]:g} deg has every pass feasible at {best_total[0]:.1f} deg")
                for k in pending:
                    failed[k] = why_cut
                    rows_[k] = {"k": k, "reachable": False, "reason": why_cut, "torch_roll_deg": float(roll)}
                pending = []
                break
            t_ = time.monotonic()
            trial_passes = {k: (cur[k]["pass"] if cur[k].get("pass") is not None else      # (search_pass: computed where it ran)
                                retarget_pass(plan, by_k[k], cur[k]["travel_deg"], cur[k]["work_deg"])) for k in pending}
            tm["retarget_s"] += time.monotonic() - t_
            trial = _plan_with(plan, trial_passes)
            trial.torch_roll_deg = float(roll)
            trial.line_step_mm = plan.line_step_mm
            # the search's exact per-role record of each candidate's delivered line: the platform check's own exact evaluation of that
            # line reuses it when every input is the same (pass_clearance(known=...): the same key), else evaluates
            trial.d14_exact = {k: cur[k].get("clearance") for k in pending}
            if check is None:
                out = [{"k": k, "reachable": None, "reason": "no robot check on this platform (geometric clearance only)"}
                       for k in pending]
            else:
                for k in pending:                  # the next candidates computed while the robot check runs (used if it refuses)
                    ask(gens[k])
                t_ = time.monotonic()
                out = list(check(trial, only=list(pending)) or [])
                tm["robot_check_s"] += time.monotonic() - t_
                tm["n_robot_checks"] += 1
            got = {r_.get("k"): r_ for r_ in out}
            nxt = []
            for k in pending:
                row = dict(got.get(k) or {"k": k, "reachable": None, "reason": "no robot-check answer for this pass"},
                           torch_roll_deg=float(roll))
                if row.get("reachable") is True:
                    done[k] = (cur[k], trial_passes[k])
                    rows_[k] = row
                elif row.get("reachable") is None:
                    unchecked = unchecked or row.get("reason")
                    done[k] = (cur[k], trial_passes[k])
                    rows_[k] = row
                else:
                    tries[k].append({"travel_deg": cur[k]["travel_deg"], "work_deg": cur[k]["work_deg"],
                                     "deviation_deg": cur[k]["deviation_deg"], "reason": row.get("reason"),
                                     "refusal": refusal_of(row)})
                    if log:
                        log(f"pass {k} roll {roll:g}: travel {cur[k]['travel_deg']:+.0f} work {cur[k]['work_deg']:+.0f} refused by "
                            f"the robot check ({row.get('reason')}) - next closest")
                    if advance(k) is not None:
                        nxt.append(k)
                    else:
                        rows_[k] = dict(row, reachable=False, reason=failed[k])
            pending = nxt
        res[float(roll)] = {"done": done, "rows": rows_, "tries": tries, "failed": failed, "unchecked": unchecked, "last": last}
        if len(done) == len(plan.passes) and all(rows_[k].get("reachable") is True for k in done):
            tot = sum(done[k][0]["deviation_deg"] for k in done)
            if tot < best_total[0]:
                best_total = (tot, float(roll))
    elapsed = time.monotonic() - t_start

    def summary(rr):
        ok = [k for k in rr["rows"] if rr["rows"][k].get("reachable") is True]
        dev = sum(rr["done"][k][0]["deviation_deg"] for k in rr["done"]) if rr["done"] else math.inf
        m = [float(rr["rows"][k]["min_limit_margin_rad"]) for k in ok if rr["rows"][k].get("min_limit_margin_rad") is not None
             and math.isfinite(float(rr["rows"][k]["min_limit_margin_rad"]))]
        return {"feasible_passes": len(ok), "n_passes": len(plan.passes), "all_feasible": len(ok) == len(plan.passes),
                "total_deviation_deg": float(dev) if len(rr["done"]) == len(plan.passes) else None,
                "min_limit_margin_rad": min(m) if m and len(ok) == len(plan.passes) else None,
                "not_feasible": [f"pass {k}: {rr['failed'][k]}" for k in rr["failed"]]}
    sums = {r: summary(res[r]) for r in res}
    for r in res:
        sums[r]["neck_refused_passes"] = sorted(neck_refused.get(r, {}))
    full = [r for r in res if sums[r]["all_feasible"]]
    if full:
        best = min(full, key=lambda r: (round(sums[r]["total_deviation_deg"], 9),
                                        -(sums[r]["min_limit_margin_rad"] if sums[r]["min_limit_margin_rad"] is not None else -math.inf),
                                        list(res).index(r)))
        why = (f"roll {best:.0f} deg: every pass feasible, total deviation from the WPS {sums[best]['total_deviation_deg']:.1f} deg"
               + (f", joint-limit margin {sums[best]['min_limit_margin_rad']:.2f} rad" if sums[best]['min_limit_margin_rad'] is not None else "")
               + "".join(f"; roll {o:.0f} deg: " + (f"every pass feasible, total deviation {sums[o]['total_deviation_deg']:.1f} deg"
                                                    if sums[o]["all_feasible"] else
                                                    f"{sums[o]['feasible_passes']}/{len(plan.passes)} passes feasible"
                                                    + (f" (pass(es) {sums[o]['neck_refused_passes']}: the torch neck would bend "
                                                       "against the travel angle - NECK_RULE)" if sums[o]["neck_refused_passes"] else ""))
                         for o in res if o != best))
    else:
        best = max(res, key=lambda r: (sums[r]["feasible_passes"], -len(sums[r]["neck_refused_passes"]), -list(res).index(r)))
        why = ("no roll makes every pass feasible: " + "; ".join(f"roll {r:.0f} deg {sums[r]['feasible_passes']}/{len(plan.passes)}"
                                                               + (f" (pass(es) {sums[r]['neck_refused_passes']} not this roll's "
                                                                  "travel side - NECK_RULE)" if sums[r]["neck_refused_passes"] else "")
                                                               for r in res) + f" - roll {best:.0f} deg recorded")
    rr = res[best]
    per = []
    new_passes = []
    for wp in plan.passes:
        t0, w0 = _nominal(wp)
        if wp.k in rr["done"]:
            c, p_new = rr["done"][wp.k]
            cl = c["clearance"]
            lim = cl.get("limiting_role")
            by_role = {r_: dict(v_, link=part_link(role_body[r_], v_.get("point_tcp_mm")) if r_ in role_body else "")
                       for r_, v_ in (cl.get("by_role") or {}).items()}
            dw = _deviation_why(t0, w0, c, rr["tries"][wp.k], float(c["deviation_deg"]), req_txt)
            rec = {"k": wp.k, "nominal": {"travel_deg": t0, "work_deg": w0}, "chosen": {"travel_deg": c["travel_deg"],
                   "work_deg": c["work_deg"]}, "deviation_deg": c["deviation_deg"], "clearance_mm": cl["clearance_mm"],
                   "required_mm": r_mm, "torch_roll_deg": float(best), "closest_part": {
                       "point_tcp_mm": cl.get("point_tcp_mm"), "solid": cl.get("solid"), "line": cl.get("line"),
                       "s_mm": cl.get("s_mm"), "kind": cl.get("kind")},
                   "clearance_by_role": by_role, "excess_mm": cl.get("excess_mm"),
                   "limiting_part": {"role": lim, "link": (by_role.get(lim) or {}).get("link", ""),
                                     "clearance_mm": (by_role.get(lim) or {}).get("clearance_mm"),
                                     "required_mm": (by_role.get(lim) or {}).get("required_mm")},
                   "robot_check": {"reachable": rr["rows"][wp.k].get("reachable"), "reason": rr["rows"][wp.k].get("reason")},
                   "refused_closer": rr["tries"][wp.k], "n_evaluated": c.get("n_evaluated"),
                   "deviation_why": dw, "on": "the delivered line (weld_line: kiro/pass_<k>.yaml = result.paths)",
                   "checked_line": cl.get("checked_line"), "worst_sample": {"line": cl.get("line"), "line_s_mm": cl.get("line_s_mm"),
                                                                           "n_samples": cl.get("n_samples")}}
            neck = _neck_record(wt, weld_line(plan, p_new, wt, float(best)))
            rec["roll"] = _roll_record(wt, float(best), t0, rolls, neck)
            rec["why"] = ("the WPS nominal" if c["deviation_deg"] < 1e-9 else
                          f"the closest orientation that clears the specimen by {req_txt}"
                          + (" and passes the robot check" if check is not None else "")
                          + (f" ({len(rr['tries'][wp.k])} closer one(s) refused by the robot check)" if rr["tries"][wp.k] else "")
                          + f"; {dw['text']}") + f"; {rec['roll']['why']}"
            p_new = dataclasses.replace(p_new, orientation=rec)
            new_passes.append(p_new)
        else:
            dw = _deviation_why(t0, w0, rr["last"].get(wp.k), rr["tries"][wp.k], None, req_txt)
            rec = {"k": wp.k, "nominal": {"travel_deg": t0, "work_deg": w0}, "chosen": None, "deviation_deg": None,
                   "clearance_mm": None, "required_mm": r_mm, "torch_roll_deg": float(best),
                   "why": rr["failed"].get(wp.k, "") + (f"; {dw['text']}" if dw.get("closer", {}).get("n") else ""),
                   "deviation_why": dw, "refused_closer": rr["tries"][wp.k], "roll": _roll_record(wt, float(best), t0, rolls, None)}
            new_passes.append(dataclasses.replace(wp, orientation=rec, travel_nominal_deg=t0, work_nominal_deg=w0))
        per.append(rec)
    plan.passes = new_passes
    plan.torch_roll_deg = float(best)
    plan.feasibility = [rr["rows"].get(wp.k, {"k": wp.k, "reachable": False, "reason": rr["failed"].get(wp.k, "")})
                        for wp in plan.passes]
    plan.torch_roll = {"deg": float(best), "reason": why, "rule": "the two TCP rolls about the torch axis that keep the TCP y across "
                       "the seam (KIRO weaves along the TCP y): 0 deg (TCP x = travel) and 180 deg (TCP x = -travel); "
                       + NECK_RULE + "; then the roll at "
                       "which every pass is feasible with the smaller total deviation from the WPS, then the larger joint-limit "
                       "margin; the same roll for every pass of the plan",
                       "neck": {"body_axis_tcp": None if body_axis_tcp(wt) is None else [float(v) for v in body_axis_tcp(wt)],
                                "source": f"the weld torch's flange <- TCP ({wt.source or wt.tcp_frame}): the flange +z = the straight body",
                                "side_by_roll": {f"{float(r):g}": neck_side(wt, r) for r in rolls},
                                "refused_by_roll": {f"{r:g}": {str(k): v for k, v in neck_refused.get(r, {}).items()} for r in res}},
                       "candidates": [{"deg": float(r), **sums[r]} for r in res]}
    by_roll = {}
    for r_, rr_ in res.items():
        rows_ = []
        for wp in plan.passes:
            t0, w0 = _nominal(wp)
            if wp.k in rr_["done"]:
                c_ = rr_["done"][wp.k][0]
                rows_.append({"k": wp.k, "chosen": {"travel_deg": c_["travel_deg"], "work_deg": c_["work_deg"]},
                              "deviation_deg": c_["deviation_deg"], "clearance_mm": c_["clearance"]["clearance_mm"],
                              "clearance_by_role": {r_: v_["clearance_mm"] for r_, v_ in (c_["clearance"].get("by_role") or {}).items()},
                              "limiting_role": c_["clearance"].get("limiting_role"),
                              "robot_check": rr_["rows"].get(wp.k, {}).get("reachable")})
            else:
                rows_.append({"k": wp.k, "chosen": None, "why": rr_["failed"].get(wp.k, "")})
        by_roll[f"{r_:g}"] = rows_
    record = {"rule": ORIENTATION_RULE, "angle_step_deg": ANGLE_STEP_DEG, "by_roll": by_roll, "line_rule": LINE_RULE,
              "line_step_mm": st_m * 1e3, "line_step_source": st_src,
              "reach_model": reach.describe() if reach is not None else "none (the robot check alone)",
              "required": req.to_dict() if req is not None else
              {"required_mm": r_mm}, "tool": body.describe(), "tool_roles": body.how if roles else "one role (the whole tool)",
              "process_required_mm": PROCESS_REQUIRED_MM if roles else None,
              "passes": per, "torch_roll_deg": float(best), "elapsed_s": elapsed,
              "timing": {**tm, "rest_s": elapsed - tm["candidates_wait_s"] - tm["robot_check_s"] - tm["retarget_s"]},
              "robot_check": "the platform's D14 (check_weld_plan)" if check is not None else "none (geometric clearance only)"}
    if rr["unchecked"]:
        record["unchecked"] = rr["unchecked"]
    plan.orientation = record
    return record
