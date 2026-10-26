"""The delivered weld line of a pass - ONE source for what D14 checks, what is delivered and executed (user 2026-10-13:
"checked = delivered = executed").

KIRO's executor runs each pass as ONE straight line at ONE orientation (kiro/pass_<k>.yaml: the weld torch's TCP from current_start
to current_goal); our layer plan's per-station torch poses (weld_plan.npz dense paths) are information only. This module builds that
line, and nothing else builds it:

  wire end      the bead aim point runs straight from the layer's usable seam start to its goal (pass_endpoints: the measured seam
                ends at the layer height - G1WeldPlan.layer_ends, else the first / last plan sections - moved inside by the weld
                design's end inset; the layer plan's aim (x, z) at those s).
  orientation   one for the whole line: the rotation mean of the pass's station torch frames (station_rotations: each station's
                frame at the pass's travel / work angles in that station's own section frame - what plan_weld and
                weld_orient.retarget_pass build), then the plan's TCP roll about the torch axis; the TCP = the wire end moved back
                along -z_TCP by the torch's nozzle end + stick-out (WeldTorch.tcp_poses).
  weave (cap)   KIRO's executor weaves the TCP along its y axis by the WPS amplitude (weld_plan weave_axis tool_y,
                weave_amplitude_mm): the weave edge lines are the line shifted by +- amplitude x y_TCP (a plan without a WPS
                amplitude - the cap band model - shifts by its widest planned band half width along the plate's lateral axis).
  samples       every pose that is checked is a sample of the line: evenly from start to goal at most step_m apart, both ends
                included (segment_points) - step_m = the platform's Cartesian step (the interpolation MoveIt executes a straight
                line with: D14 samples the poses MoveIt executes), else the plan's dense path step. Each sample is evaluated in
                the section of the plan station nearest to its wire end (weld_clearance.nearest_station); a weave edge sample in the
                section of its line sample.

Users: weld_orient (D14's orientation search and its exact per-role clearance), the platforms' check_weld_plan (D14's robot check:
IK, limits, MoveIt's Cartesian path with collision checking - on this line), kiro_plan.write_kiro_outputs (pass_<k>.yaml),
hongik_scan_ros.kiro_io.plan_paths (result.paths = the wire-end line), g1_outputs (weld_plan.json weld_line), the lab weld path dry
run (it executes pass_<k>.yaml and re-checks the same samples).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["station_rotations", "rotation_mean", "line_rotation", "pass_endpoints", "segment_points", "WeldLine", "weld_line",
           "line_step_m", "LINE_RULE"]

LINE_RULE = ("the delivered line (scan_core.weld_line): the wire end straight from the layer's usable seam start to its goal at ONE "
             "orientation (the rotation mean of the pass's station torch frames at its travel / work angles, then the TCP roll); the cap's "
             "weave edges = the line shifted by +- the WPS amplitude along y_TCP (KIRO's executor weaves along the TCP y); D14 checks "
             "the line's samples (at most the platform's Cartesian step apart, both ends) - the same poses kiro/pass_<k>.yaml "
             "and result.paths carry")


# --------------------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------------------
def _unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def station_rotations(plan, wp, travel_deg: float, work_deg: float) -> np.ndarray:
    """(N, 3, 3) base <- tip frame [x travel, y = z x x, z = torch axis toward the work] of pass wp at (travel, work) [deg] in each of
    its stations' own section frames, in the pass's station order (wp.s_mm) - the AWS A3.0 construction of weld_plan._tcp_R (work
    angle signed: + = toward the pass's tilt side, weld_orient.pass_tilt), vectorized over the stations. weld_orient.retarget_pass
    stores exactly these as the pass's R_base_tcp, so line_rotation of a retargeted pass is the rotation the search evaluated."""
    from .weld_orient import pass_tilt                                     # noqa: PLC0415
    by_s = {float(st.s_mm): st for st in plan.stations}
    sts = [by_s[float(s)] for s in np.asarray(wp.s_mm, dtype=float)]
    n = np.array([np.asarray(st.n, dtype=float) for st in sts])
    w = np.array([np.asarray(st.w, dtype=float) for st in sts])
    u = np.array([np.asarray(st.u, dtype=float) for st in sts])
    d = u if wp.direction == "S->E" else -u
    tilt = pass_tilt(plan, wp) * w
    T, W = math.radians(float(travel_deg)), math.radians(float(work_deg))
    p = _unit_rows(-n * math.cos(W) + tilt * math.sin(W))
    a = _unit_rows(d * math.sin(T) + p * math.cos(T))
    x = _unit_rows(d - np.sum(d * a, axis=1, keepdims=True) * a)
    y = np.cross(a, x)
    return np.stack([x, y, a], axis=-1)


def rotation_mean(R) -> np.ndarray:
    """The rotation mean (the nearest rotation to the sum, SVD) of rotations R (N, 3, 3)."""
    M = np.sum(np.asarray(R, dtype=float), axis=0)
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))]) @ Vt


def line_rotation(wp) -> np.ndarray:
    """The delivered line's torch orientation (base <- tip frame, before the TCP roll): the rotation mean of the pass's station torch
    frames (wp.R_base_tcp)."""
    return rotation_mean(wp.R_base_tcp)


# --------------------------------------------------------------------------------------
# the wire-end line
# --------------------------------------------------------------------------------------
def _stations_sorted(plan) -> list:
    return sorted(plan.stations, key=lambda st: float(st.s_mm))


def _frame_at(stations: list, s_mm: float) -> tuple:
    """The section frame (origin, u, n, w) at s [mm]: linear between sections, beyond the ends extended along the end section's u."""
    S = np.array([float(st.s_mm) for st in stations])
    sc = float(np.clip(s_mm, S[0], S[-1]))

    def lerp(attr):
        A = np.array([np.asarray(getattr(st, attr), dtype=float) for st in stations])
        return np.array([np.interp(sc, S, A[:, j]) for j in range(3)])
    o, u, n, w = (lerp(a) for a in ("origin", "u", "n", "w"))
    u, n, w = (v / np.linalg.norm(v) for v in (u, n, w))
    return o + (float(s_mm) - sc) * 1e-3 * u, u, n, w


def pass_endpoints(plan, wp) -> dict:
    """The pass's usable start and goal (the seam ends at the layer height - G1WeldPlan.layer_ends, else the first / last sections -
    moved inside along the seam by the design's end inset (weld_design.end_inset_mm {start, goal}: start = the end the pass starts
    at; weld_plan.end_insets_mm)) as aim points (section x, z [mm], base position [m]). The aim offsets are the layer plan's at that s
    (beyond the end sections: the end section's). 'measured' = the measured ends before the inset (same aim point rule, sigma),
    'inset_mm' = the insets used. A bare pose list (a pass without the layer plan's aim offsets, a plan without a process): the line
    from its first to its last pose (pass order)."""
    from .weld_plan import end_insets_mm                                   # noqa: PLC0415
    if getattr(wp, "aim_xz_mm", None) is None or getattr(plan, "process", None) is None:
        return _bare_endpoints(wp)
    st = _stations_sorted(plan)
    S = np.array([float(x.s_mm) for x in st])
    le = (getattr(plan, "layer_ends", None) or {}).get(int(wp.k))
    if le and all(math.isfinite(float((le.get(X) or {}).get("s_m", float("nan")))) for X in ("S", "E")):
        sS, sE = float(le["S"]["s_m"]) * 1e3, float(le["E"]["s_m"]) * 1e3
        sig = {X: float((le.get(X) or {}).get("sigma_m", float("nan"))) * 1e3 for X in ("S", "E")}
        src = "the seam ends at the layer height (end-face model, G1WeldPlan.layer_ends)"
    else:
        sS, sE = float(S[0]), float(S[-1])
        ends = getattr(plan, "ends", None) or {}
        sig = {X: float(ends.get(f"sigma_{X}_mm", float("nan"))) for X in ("S", "E")}
        src = "the first and last plan sections (measured S and E ends)"
    i0, i1 = end_insets_mm(plan)
    if wp.direction == "S->E":
        (a, sa), (b, sb) = ("S", sS), ("E", sE)
        s0, s1 = sa + i0, sb - i1
    else:
        (a, sa), (b, sb) = ("E", sE), ("S", sS)
        s0, s1 = sa - i0, sb + i1
    o = np.argsort(np.asarray(wp.s_mm, dtype=float))
    ss = np.asarray(wp.s_mm, dtype=float)[o]
    axz = np.asarray(wp.aim_xz_mm, dtype=float)[o]

    def at(s):
        x, z = float(np.interp(s, ss, axz[:, 0])), float(np.interp(s, ss, axz[:, 1]))
        org, u, n, w = _frame_at(st, s)
        return {"s_mm": float(s), "aim_xz_mm": [x, z], "tip_base_m": org + (x * w + z * n) * 1e-3, "n": n, "w": w, "u": u}
    meas = {"start": {"end": a, **at(sa), "sigma_mm": sig[a]}, "goal": {"end": b, **at(sb), "sigma_mm": sig[b]}}
    return {"start": at(s0), "goal": at(s1), "source": src, "measured": meas, "inset_mm": {"start": i0, "goal": i1}}


def _bare_endpoints(wp) -> dict:
    """pass_endpoints of a bare pose list: its first and last poses (pass order), no aim offsets, no inset."""
    P = np.asarray(wp.aim_base_m, dtype=float)
    S = np.asarray(wp.s_mm, dtype=float)

    def at(i):
        return {"s_mm": float(S[i]), "aim_xz_mm": [float("nan"), float("nan")], "tip_base_m": P[i].copy()}
    e = {"start": at(0), "goal": at(-1)}
    return {**e, "source": "a bare pose list: its first and last poses", "inset_mm": {"start": 0.0, "goal": 0.0},
            "measured": {X: dict(e[X], end=X, sigma_mm=float("nan")) for X in ("start", "goal")}}


def segment_points(P_a, P_b, step_m: float) -> np.ndarray:
    """The straight segment a -> b sampled evenly at most step_m apart, both ends included (n, 3) - the sampling of every checked
    straight move (D14's delivered line)."""
    a, b = np.asarray(P_a, dtype=float), np.asarray(P_b, dtype=float)
    L = float(np.linalg.norm(b - a))
    n = max(2, int(math.ceil(L / float(step_m))) + 1)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


def line_step_m(plan, platform_step_m: Optional[float] = None) -> tuple:
    """(step [m], source) of the line samples: the platform's Cartesian step (the interpolation of its straight moves - MoveIt's
    max_step) when given, else the plan's dense path step (weld_plan.dense_step_mm)."""
    if platform_step_m is not None and math.isfinite(float(platform_step_m)) and float(platform_step_m) > 0.0:
        return float(platform_step_m), "the platform's Cartesian step (the interpolation of its straight moves)"
    pr = getattr(plan, "process", None)
    return float(pr.dense_step_mm) * 1e-3, "the plan's dense path step (weld_plan.dense_step_mm; no platform step given)"


@dataclass(frozen=True, eq=False)
class WeldLine:
    """One pass's delivered line (module docstring). Positions [m] and rotations in the plan's base frame; R_tcp / tcp_* are None
    when built without a weld torch (the wire-end line only)."""
    k: int
    kind: str
    direction: str
    s_start_mm: float
    s_goal_mm: float
    tip_start: np.ndarray                  # (3,) the wire end (bead aim point) at the start
    tip_goal: np.ndarray
    R_tip: np.ndarray                      # (3, 3) base <- tip frame [x travel, y, z torch axis], before the roll
    roll_deg: float
    R_tcp: Optional[np.ndarray]            # (3, 3) base <- TCP (roll and the torch's TCP <- tip axes applied)
    tcp_start: Optional[np.ndarray]        # (3,) the TCP at the start
    tcp_goal: Optional[np.ndarray]
    tip_tcp_m: Optional[np.ndarray]        # (3,) the wire end in the TCP frame (nozzle end + stick-out along z_TCP)
    weave: Optional[dict]                  # cap: {amplitude_mm, axis, offset_m (3,), rule}; None for a fill line
    ends: dict = field(default_factory=dict)          # pass_endpoints

    @property
    def axis(self) -> np.ndarray:
        """The torch axis toward the work (base)."""
        return self.R_tip[:, 2]

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(self.tip_goal - self.tip_start))

    def T_tcp(self, which: str) -> np.ndarray:
        """4x4 base <- TCP at 'start' or 'goal'."""
        if self.R_tcp is None:
            raise ValueError("the line was built without a weld torch (no TCP poses)")
        T = np.eye(4)
        T[:3, :3] = self.R_tcp
        T[:3, 3] = self.tcp_start if which == "start" else self.tcp_goal
        return T

    def offsets(self) -> list:
        """[(name, offset (3,) [m])] of the lines the torch runs: the centre, and for the cap the two weave edges."""
        out = [("centre", np.zeros(3))]
        if self.weave is not None:
            v = np.asarray(self.weave["offset_m"], dtype=float)
            out += [("+edge", v), ("-edge", -v)]
        return out

    def samples(self, step_m: float) -> dict:
        """The line's samples (segment_points of the wire-end line, at most step_m apart, both ends; the TCP at the same fractions):
        {'tip' (n, 3) the wire end, 'tcp' (n, 3) the TCP (None without a torch), 'f' (n,) the fraction along the line, 's_mm' (n,) the
        line's seam coordinate}."""
        tip = segment_points(self.tip_start, self.tip_goal, step_m)       # the count from the wire-end line: the same for every
        n = len(tip)                                                        # orientation (D14's rows do not change between candidates)
        f = np.linspace(0.0, 1.0, n)
        tcp = None if self.tcp_start is None else self.tcp_start + f[:, None] * (self.tcp_goal - self.tcp_start)
        return {"tip": tip, "tcp": tcp, "f": f, "s_mm": self.s_start_mm + f * (self.s_goal_mm - self.s_start_mm)}

    def to_dict(self) -> dict:
        from .g1_types import jsonable                                     # noqa: PLC0415
        return jsonable({"k": self.k, "kind": self.kind, "direction": self.direction, "s_start_mm": self.s_start_mm,
                         "s_goal_mm": self.s_goal_mm, "length_mm": self.length_m * 1e3,
                         "wire_end_base_m": {"start": self.tip_start, "goal": self.tip_goal},
                         "tcp_base_m": None if self.tcp_start is None else {"start": self.tcp_start, "goal": self.tcp_goal},
                         "R_base_tcp": self.R_tcp, "R_base_tip": self.R_tip, "torch_axis_base": self.axis,
                         "roll_deg": self.roll_deg, "wire_end_in_tcp_m": self.tip_tcp_m,
                         "weave": None if self.weave is None else {k_: v for k_, v in self.weave.items()},
                         "rule": LINE_RULE})


def _weave(plan, wp, R_tcp: Optional[np.ndarray], R_tip: np.ndarray) -> Optional[dict]:
    """The cap's weave edge offset (module docstring) or None for a fill line."""
    if wp.kind != "cap" or wp.weave_amp_mm is None:
        return None
    pr = getattr(plan, "process", None)
    y = (R_tcp if R_tcp is not None else R_tip)[:, 1]
    if pr is not None and pr.weave_amplitude_mm is not None:
        A = float(pr.weave_amplitude_mm)
        return {"amplitude_mm": A, "axis": str(pr.weave_axis), "offset_m": y * A * 1e-3, "rotates_with_torch": True,
                "rule": f"KIRO's executor weaves the TCP along its y axis ({pr.weave_axis}) by the WPS amplitude {A:g} mm"}
    A = float(np.max(np.asarray(wp.weave_amp_mm, dtype=float)))
    w = np.mean([np.asarray(st.w, dtype=float) for st in plan.stations], axis=0)
    w = w / np.linalg.norm(w)
    return {"amplitude_mm": A, "axis": "plate lateral", "offset_m": w * A * 1e-3, "rotates_with_torch": False,
            "rule": f"no WPS weave amplitude (the cap band model): the widest planned band half width {A:.2f} mm along the plate's "
                    "lateral axis"}


def weld_line(plan, wp, torch=None, roll_deg: Optional[float] = None, *, R_tip: Optional[np.ndarray] = None,
              ends: Optional[dict] = None) -> WeldLine:
    """Pass wp's delivered line (module docstring). torch = the plan's weld torch (WeldTorch with its nozzle end and stick-out; None =
    the wire-end line only); roll_deg = the TCP roll (default: the plan's torch_roll_deg); R_tip = the line's orientation (base <- tip
    frame, before the roll; default line_rotation(wp) - D14's search passes a candidate's rotation_mean(station_rotations(...)));
    ends = pass_endpoints(plan, wp) when already computed (it does not depend on the orientation)."""
    e = pass_endpoints(plan, wp) if ends is None else ends
    roll = float(getattr(plan, "torch_roll_deg", 0.0) or 0.0) if roll_deg is None else float(roll_deg)
    Rt = line_rotation(wp) if R_tip is None else np.asarray(R_tip, dtype=float)
    P0, P1 = np.asarray(e["start"]["tip_base_m"], dtype=float), np.asarray(e["goal"]["tip_base_m"], dtype=float)
    if torch is not None:                       # the TCP = the wire end moved back along -z_TCP (WeldTorch.tcp_poses)
        Pc, Rc = torch.tcp_poses(np.stack([P0, P1]), Rt, roll_deg=roll)
        c0, c1 = Pc[0].copy(), Pc[1].copy()
        tip_tcp = np.asarray(torch.tcp_from_tip.t, dtype=float)
    else:
        c0 = c1 = Rc = tip_tcp = None
    return WeldLine(k=int(wp.k), kind=str(wp.kind), direction=str(getattr(wp, "direction", "S->E")), s_start_mm=float(e["start"]["s_mm"]),
                    s_goal_mm=float(e["goal"]["s_mm"]), tip_start=P0, tip_goal=P1, R_tip=Rt, roll_deg=roll, R_tcp=Rc,
                    tcp_start=c0, tcp_goal=c1, tip_tcp_m=tip_tcp, weave=_weave(plan, wp, Rc, Rt), ends=e)
