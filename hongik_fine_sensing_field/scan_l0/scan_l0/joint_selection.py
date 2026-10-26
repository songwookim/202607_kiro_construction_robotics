"""Joint-space path selection by planned duration (KIRO field 2026-10-14, review B1) - no ROS, no robot.

A joint-space move (acquisition, start pose, post-weld approach) has several goal configurations (IK solutions and their 2 pi
representations). Its cost is the move's duration: the backend's retime of the dense joint path under the platform's declared
limits (scan_core.trajectory.retime_joint_path) - a physical cost with nothing to tune. Checking a candidate costs a dense
collision sweep in move_group (and an OMPL plan when the straight joint line is blocked), so the candidates are searched best-first
under an exact lower bound of that duration:

  duration_lower_bound  for ANY joint path the backend can time from q0 to q_goal (the straight joint line or an OMPL path):
      joint part  max_j max(0, |dq_j| - h) / v_j  (v_j = the declared joint velocity limit, h = the backend's dense joint-path step).
                  Why: the retime keeps every interior central difference (q_{i+1} - q_{i-1}) / (t_{i+1} - t_{i-1}) <= v_j and only
                  stretches time; the differences at points 1, 3, 5, ... span disjoint time intervals covering [t_0, t_N] (N even) or
                  [t_0, t_{N-1}] (N odd - then one dense step <= h remains), so |dq_j| <= v_j T + h. The rest-to-rest acceleration
                  term (v / a of the continuous trapezoid) is NOT in the bound: the central-difference acceleration check never
                  anchors the odd-indexed velocities at rest, so the discrete retime can be faster than the continuous minimum (it
                  is on short moves) and the bound would not be one.
      TCP part    the trapezoid time (TCP speed v, acceleration a of the move) of max(flange chord, flange rotation x rotation
                  radius): the retime's path variable sums max(translation, rotation x radius) per step, which is >= both (the
                  triangle inequality in R^3 and on SO(3)); its trapezoid duration is nondecreasing in that length and nonincreasing in
                  v and a, and time is only ever stretched.
  DurationSearch  best-first enumeration in nondecreasing planned duration: candidates in bound order (ties keep the supplied
                  order); a checked candidate is yielded once its duration <= the next unchecked bound (no unchecked candidate can
                  be shorter), so the first yield is the true minimum and the search stops as early as the bounds allow. The
                  straight joint line first (check(..., allow_ompl=False); the most predictable path): only when no candidate has an
                  executable straight line, MoveIt OMPL for the blocked ones in bound order - a failed OMPL plan moves on to the
                  next candidate. A stop (the run's cancel / latch) and an unavailable platform (PlatformUnavailable) end the
                  search at once - neither is a property of a candidate.
"""
from __future__ import annotations

import heapq
import math
from typing import Callable, Iterable, Optional

import numpy as np

from scan_core.trajectory import TrapezoidProfile, rotation_vector

__all__ = ["joint_time_bound", "tcp_time_bound", "duration_lower_bound", "DurationSearch", "TIMING_LIMIT_KEYS"]

# the backend's path_timing_limits(): what its retime of a joint path is bound by (the platform's declared limits + its dense step)
TIMING_LIMIT_KEYS = ("joint_velocity_rad_s", "joint_path_step_rad", "tcp_speed_mps", "tcp_accel_mps2", "rotation_radius_m")


def joint_time_bound(q0, q_goal, v_joint, step_rad: float = 0.0) -> float:
    """max_j max(0, |q_goal_j - q0_j| - step_rad) / v_j [s] (module docstring: the joint part). v_joint: one limit for all joints or
    one per joint (> 0)."""
    dq = np.abs(np.asarray(q_goal, dtype=float).reshape(-1) - np.asarray(q0, dtype=float).reshape(-1))
    v = np.broadcast_to(np.asarray(v_joint, dtype=float), dq.shape)
    if not (np.all(np.isfinite(dq)) and np.all(v > 0.0)):
        raise ValueError("joint_time_bound: joints must be finite and the joint velocity limits > 0")
    return float(np.max(np.maximum(0.0, dq - max(0.0, float(step_rad))) / v)) if dq.size else 0.0


def tcp_time_bound(T0, T1, v_mps: float, a_mps2: float, rotation_radius_m: float) -> float:
    """Trapezoid duration of max(|p1 - p0|, angle(R0, R1) x rotation_radius_m) at TCP speed v_mps and acceleration a_mps2 [s] (module
    docstring: the TCP part)."""
    T0 = np.asarray(T0, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    chord = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
    turn = float(np.linalg.norm(rotation_vector(T0[:3, :3].T @ T1[:3, :3]))) * float(rotation_radius_m)
    S = max(chord, turn)
    if not (math.isfinite(S) and v_mps > 0.0 and a_mps2 > 0.0):
        raise ValueError("tcp_time_bound: needs a finite path and a TCP speed / acceleration > 0")
    return float(TrapezoidProfile(S=S, v=float(v_mps), a=float(a_mps2)).duration)


def duration_lower_bound(q0, q_goal, limits: dict, *, fk: Optional[Callable] = None, v_mps: Optional[float] = None,
                         a_mps2: Optional[float] = None) -> float:
    """Lower bound [s] of the planned duration of any joint path from q0 to q_goal on a backend with these timing limits (its
    path_timing_limits(): TIMING_LIMIT_KEYS): the joint part, and the TCP part when the flange FK and the move's TCP speed /
    acceleration are given (each capped by the platform's TCP limits, as the backend caps them)."""
    b = joint_time_bound(q0, q_goal, limits["joint_velocity_rad_s"], limits.get("joint_path_step_rad", 0.0))
    if fk is not None and v_mps is not None and a_mps2 is not None:
        v = min(float(v_mps), float(limits["tcp_speed_mps"]))
        a = min(float(a_mps2), float(limits["tcp_accel_mps2"]))
        b = max(b, tcp_time_bound(fk(np.asarray(q0, dtype=float)), fk(np.asarray(q_goal, dtype=float)), v, a,
                                  float(limits["rotation_radius_m"])))
    return b


def _plan_row(plan: Optional[dict]) -> Optional[dict]:
    """The few plan numbers a candidate row keeps (the full plan stays with the selected move's record)."""
    if not plan:
        return None
    keys = ("route", "duration_s", "flange_translation_m", "flange_rotation_deg", "max_joint", "max_joint_rotation_deg", "estimate_only")
    return {k: plan.get(k) for k in keys if k in plan}


class DurationSearch:
    """Best-first enumeration of joint-space candidates in nondecreasing planned duration (module docstring). Iterating yields
    (index, candidate, check) - check = the backend's check_joint result (its '_checked_plan' is the trajectory move_checked_joint
    sends unchanged).

    cands   candidate dicts ({'q': goal joints, ...} - read only)
    bounds  duration_lower_bound per candidate [s] (0 everywhere = no early stop: still the duration order, every candidate checked)
    check   check(cand, allow_ompl) -> {ok, reason, plan, _checked_plan} (the backend's check_joint)
    stop    stop(where) - raises to end the search (the run's cancel / latch); called before and after every check
    accept  accept(cand, check) -> '' or the reason an executable plan is not usable (e.g. longer than an explicit motion timeout)

    rows: one record per candidate (supplied rank, bound, what was checked, outcome, duration) - the run log's joint_candidates."""

    def __init__(self, cands: Iterable[dict], bounds: Iterable[float], check: Callable, *, stop: Optional[Callable] = None,
                 accept: Optional[Callable] = None):
        self.cands = list(cands)
        self.bounds = [float(b) for b in bounds]
        if len(self.bounds) != len(self.cands) or not all(math.isfinite(b) and b >= 0.0 for b in self.bounds):
            raise ValueError("DurationSearch: one finite lower bound >= 0 per candidate")
        self.check, self.stop, self.accept = check, stop, accept
        n = len(self.cands)
        self._seq = sorted(range(n), key=lambda i: (self.bounds[i], i))
        self._pos = 0
        self._pending: list = []                           # heap (duration, position in the check order, index)
        self._checks: dict = {}
        self._blocked: list = []                           # straight line blocked (estimate only): OMPL-eligible
        self.phase = "straight"
        self.n_straight = self.n_ompl = 0
        self.n_yielded = 0
        self.any_straight = False
        self.rows = [{"rank": i + 1, "q_goal_deg": np.round(np.degrees(np.asarray(c["q"], dtype=float)), 3).tolist(),
                      "bound_s": round(self.bounds[i], 4), "checked": None, "ok": None, "reason": "", "duration_s": None,
                      "selected": False} for i, c in enumerate(self.cands)]

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        while True:
            nxt = self._seq[self._pos] if self._pos < len(self._seq) else None
            if self._pending and (nxt is None or self._pending[0][0] <= self.bounds[nxt]):
                _d, _p, i = heapq.heappop(self._pending)
                self.n_yielded += 1
                self.rows[i]["duration_order"] = self.n_yielded
                return i, self.cands[i], self._checks[i]
            if nxt is not None:
                self._pos += 1
                self._evaluate(nxt, allow_ompl=self.phase == "ompl")
                continue
            if self.phase == "straight" and not self.any_straight and self._blocked:
                self.phase = "ompl"                         # no candidate has an executable straight line: OMPL, bound order
                self._seq = sorted(self._blocked, key=lambda i: (self.bounds[i], i))
                self._pos = 0
                continue
            raise StopIteration

    def _stop(self, where: str) -> None:
        if self.stop is not None:
            self.stop(where)

    def _evaluate(self, i: int, *, allow_ompl: bool) -> None:
        c, row = self.cands[i], self.rows[i]
        what = f"candidate {i + 1}/{len(self.cands)} ({'OMPL' if allow_ompl else 'straight joint line'})"
        self._stop(f"before checking {what}")
        chk = self.check(c, allow_ompl) or {}             # PlatformUnavailable propagates: the search ends here
        self._stop(f"after checking {what}")
        if allow_ompl:
            self.n_ompl += 1
        else:
            self.n_straight += 1
        plan = chk.get("plan") or {}
        row.update(checked="ompl" if allow_ompl else "straight", ok=bool(chk.get("ok")), reason=str(chk.get("reason") or ""),
                   plan=_plan_row(plan), check_order=self.n_straight + self.n_ompl)
        if chk.get("ok"):
            d = plan.get("duration_s")
            why = ""
            if d is None or not math.isfinite(float(d)) or float(d) < 0.0:
                why = f"the backend reported no valid duration ({d!r})"
            elif chk.get("_checked_plan") is None:
                why = "the backend did not keep the checked trajectory (move_checked_joint could not send exactly it)"
            elif self.accept is not None:
                why = str(self.accept(c, chk) or "")
            if why:
                row.update(ok=False, reason=why)
                return
            row["duration_s"] = round(float(d), 4)
            self._checks[i] = chk
            heapq.heappush(self._pending, (float(d), self._pos, i))
            if not allow_ompl:
                self.any_straight = True
            return
        if not allow_ompl and plan.get("estimate_only"):
            row["straight_estimate_s"] = None if plan.get("duration_s") is None else round(float(plan["duration_s"]), 4)
            self._blocked.append(i)

    def finish(self, chosen: Optional[int]) -> list:
        """Mark the chosen candidate and say why every other one was not (rows for the record)."""
        beaten = "its lower bound cannot beat the selected duration"
        for i, row in enumerate(self.rows):
            row["selected"] = i == chosen
            if row["checked"] is None:
                row["reason"] = f"not checked: {beaten}" if chosen is not None else "not checked"
            elif row["ok"] and i != chosen:
                row["reason"] = "not selected: a shorter planned move, or the yaw-band preference"
            elif row["checked"] == "straight" and "straight_estimate_s" in row:
                row["reason"] += ("; OMPL not planned: a candidate has an executable straight joint line" if self.any_straight else
                                  f"; OMPL not planned: {beaten}" if chosen is not None else "")
        return self.rows
