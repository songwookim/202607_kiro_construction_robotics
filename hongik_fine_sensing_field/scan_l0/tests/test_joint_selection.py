"""Joint-space path selection by planned duration (scan_l0.joint_selection, AutoTeach._select_joint / _joint_move / _prefer_yaw_band;
KIRO field 2026-10-14, review B1/B2/B5). No ROS, no robot: the bound against the real retime (scan_core.trajectory) on the lab RB20
URDF, the search against scripted backend checks, and the AutoTeach methods bound to a minimal stand-in object."""
from __future__ import annotations

import io
import math
import os
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest

from scan_core.fk_urdf import chain_from_joints, parse_urdf_joints
from scan_core.trajectory import densify_joint_path, fast_fk, retime_joint_path
from scan_l0 import autoteach as AT
from scan_l0.joint_selection import DurationSearch, duration_lower_bound, joint_time_bound
from scan_l0.motion_types import MotionError, PlatformUnavailable

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rb20_1900es_u_eoat.urdf")
STEP = 0.01                                                    # the backends' dense joint-path step (JOINT_PATH_STEP_RAD)


def _chain():
    ch = chain_from_joints(parse_urdf_joints(io.StringIO(open(URDF, encoding="utf-8").read())), "link0", "tcp")
    return ch, fast_fk(ch), np.array([j.lower for j in ch.movable_joints]), np.array([j.upper for j in ch.movable_joints])


# ---- the bound -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("lim", [(0.06, 0.1, 0.24, 1.0),        # KIRO / lab platform limits (TCP 60 mm/s, joints 0.24 rad/s)
                                 (0.06, 0.3, 1.0, 2.0),         # the virtual arm's
                                 (1.0, 5.0, 0.5, 0.3)])         # joint-limited (time-scaled) moves
def test_bound_is_below_the_retimed_duration_of_any_joint_path(lim):
    """For random goal pairs of every size (tiny to several rad) and random detours (OMPL stand-ins: 1-8 via points), the retimed
    duration of the dense path (the backends' retime and step) is never below duration_lower_bound."""
    _ch, fk, lo, hi = _chain()
    v_tcp, a_tcp, v_j, a_j = lim
    L = dict(joint_velocity_rad_s=v_j, joint_path_step_rad=STEP, tcp_speed_mps=v_tcp, tcp_accel_mps2=a_tcp, rotation_radius_m=0.2)
    rng = np.random.default_rng(20261014)
    for _ in range(150):
        q0 = np.clip(lo + (hi - lo) * rng.random(6), -2.0 * math.pi, 2.0 * math.pi)
        scale = float(rng.choice([1e-3, 0.005, 0.02, 0.1, 0.5, 2.0]))
        q1 = np.clip(q0 + scale * rng.standard_normal(6), lo, hi)
        via = [np.clip(q0 + (q1 - q0) * rng.random() + scale * rng.standard_normal(6), lo, hi) for _v in range(int(rng.choice([0, 1, 3, 8])))]
        rt = retime_joint_path(densify_joint_path([q0, *via, q1], STEP), fk, v_tcp, a_tcp, rotation_radius_m=0.2, max_joint_velocity=v_j,
                               max_joint_accel=a_j)
        assert duration_lower_bound(q0, q1, L, fk=fk, v_mps=v_tcp, a_mps2=a_tcp) <= rt.duration + 1e-12


def test_the_continuous_rest_to_rest_term_is_not_a_bound_of_the_discrete_retime():
    """Why the bound has no v/a term: the retime checks central differences, which never anchor the odd-indexed velocities at rest -
    a short move times below the continuous trapezoid |dq|/v + v/a (or 2 sqrt(|dq|/a))."""
    _ch, fk, _lo, _hi = _chain()
    q0 = np.array([0.3, -0.4, 1.2, 0.1, 0.8, -0.2])
    q1 = q0 + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.015])
    v_j, a_j = 0.5, 0.3
    rt = retime_joint_path(densify_joint_path([q0, q1], STEP), fk, 1.0, 5.0, rotation_radius_m=0.2, max_joint_velocity=v_j,
                           max_joint_accel=a_j)
    assert rt.duration < 2.0 * math.sqrt(0.015 / a_j)
    assert joint_time_bound(q0, q1, v_j, STEP) <= rt.duration


# ---- the search ------------------------------------------------------------------------------------------------------
def _cand(k):
    return {"q": np.full(6, float(k)), "tag": k}


def _ok(d):
    return {"ok": True, "reason": "", "plan": {"duration_s": float(d), "route": "straight"}, "_checked_plan": object()}


def _blocked(d, why="point 3/40 invalid"):
    return {"ok": False, "reason": f"straight joint line blocked: {why}", "plan": {"duration_s": float(d), "estimate_only": True},
            "_checked_plan": None}


def _ompl(d):
    return {"ok": True, "reason": "", "plan": {"duration_s": float(d), "route": "OMPL"}, "_checked_plan": object()}


class Backend:
    """Scripted check_joint: outcome[(tag, allow_ompl)]; records the calls."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def check(self, c, allow_ompl):
        self.calls.append((c["tag"], allow_ompl))
        out = self.outcome[(c["tag"], allow_ompl)]
        if isinstance(out, Exception):
            raise out
        return out


def test_best_first_returns_the_true_minimum_and_stops_early():
    """Durations >= bounds; the supplied order is not the duration order. The first yield is the minimum (9 s, candidate 4); the
    search checks only the candidates whose bound is below 9 s (2 of 6) - the rest cannot be shorter."""
    bounds = [12.0, 5.0, 30.0, 8.0, 9.5, 40.0]
    dur = [13.0, 11.0, 31.0, 9.0, 10.0, 41.0]
    be = Backend({(k, False): _ok(dur[k]) for k in range(6)})
    s = DurationSearch([_cand(k) for k in range(6)], bounds, be.check)
    i, c, chk = next(s)
    assert c["tag"] == 3 and chk is be.outcome[(3, False)]
    assert [t for t, _o in be.calls] == [1, 3] and s.n_straight == 2             # bounds 5, 8 - 9.5, 12, 30, 40 not checked
    assert min(dur) == 9.0
    rows = s.finish(i)
    assert rows[3]["selected"] and not any(r_["selected"] for k, r_ in enumerate(rows) if k != 3)
    assert rows[0]["checked"] is None and rows[0]["reason"].startswith("not checked: its lower bound cannot beat")
    # the full duration order on demand (the yaw band walks it): exact, still lazily checked
    order = [c_["tag"] for _i, c_, _k in s]
    assert order == [4, 1, 0, 2, 5] and sorted(dur)[1:] == [dur[k] for k in order]


def test_straight_lines_first_ompl_only_when_none_is_straight_and_a_failed_plan_moves_on():
    """No candidate has an executable straight line: OMPL in bound order; the first OMPL plan fails -> the next candidate (no
    refusal while candidates remain); the best OMPL duration is kept and the search stops when no unchecked bound can beat it."""
    bounds = [4.0, 6.0, 7.0, 50.0]
    be = Backend({(0, False): _blocked(5.0), (1, False): _blocked(7.0), (2, False): _blocked(8.0), (3, False): _blocked(55.0),
                  (0, True): {"ok": False, "reason": "MoveIt joint-space planning failed: error code -1", "plan": None},
                  (1, True): _ompl(20.0), (2, True): _ompl(9.0), (3, True): _ompl(60.0)})
    s = DurationSearch([_cand(k) for k in range(4)], bounds, be.check)
    i, c, chk = next(s)
    assert c["tag"] == 2 and chk["plan"]["route"] == "OMPL"
    assert be.calls == [(0, False), (1, False), (2, False), (3, False), (0, True), (1, True), (2, True)]   # 3 (bound 50) never planned
    rows = s.finish(i)
    assert rows[0]["checked"] == "ompl" and "planning failed" in rows[0]["reason"]
    assert rows[3]["checked"] == "straight" and "OMPL not planned: its lower bound cannot beat" in rows[3]["reason"]


def test_an_executable_straight_line_excludes_ompl_and_unusable_plans_are_not_selected():
    be = Backend({(0, False): _blocked(1.0), (1, False): {**_ok(3.0), "_checked_plan": None}, (2, False): _ok(30.0), (3, False): _ok(9.0)})
    s = DurationSearch([_cand(k) for k in range(4)], [0.5, 1.0, 2.0, 3.0], be.check,
                       accept=lambda c, chk: "planned 30.0 s >= the explicit motion timeout 20.0 s" if c["tag"] == 2 else "")
    i, c, _chk = next(s)
    assert c["tag"] == 3 and all(not o for _t, o in be.calls)
    rows = s.finish(i)
    assert "did not keep the checked trajectory" in rows[1]["reason"] and "explicit motion timeout" in rows[2]["reason"]
    assert "OMPL not planned: a candidate has an executable straight joint line" in rows[0]["reason"]


def test_platform_unavailable_and_a_stop_end_the_search_at_once():
    be = Backend({(0, False): PlatformUnavailable("move_group /check_state_validity did not answer"), (1, False): _ok(2.0)})
    s = DurationSearch([_cand(0), _cand(1)], [1.0, 1.5], be.check)
    with pytest.raises(PlatformUnavailable):
        next(s)
    assert be.calls == [(0, False)]

    class Stopped(Exception):
        pass

    seen = []

    def stop(where):
        seen.append(where)
        if len(seen) > 2:
            raise Stopped(where)
    be2 = Backend({(k, False): _blocked(1.0) for k in range(3)})
    with pytest.raises(Stopped, match="before checking candidate 2/3"):
        next(DurationSearch([_cand(k) for k in range(3)], [0.0, 0.0, 0.0], be2.check, stop=stop))
    assert be2.calls == [(0, False)]


# ---- AutoTeach: selection, execution without a re-sort, yaw band, tracking --------------------------------------------
def _teach(outcome, *, bounds=None, follow_ok=None, executed=None):
    """A stand-in AutoTeach: the real _select_joint / _joint_move / _prefer_yaw_band bound to it, a scripted backend."""
    calls = []

    def check_joint(q, v, a, *, timeout_s=None, allow_ompl=True):
        tag = int(round(float(q[0])))
        calls.append((tag, allow_ompl))
        out = outcome[(tag, allow_ompl)]
        if isinstance(out, Exception):
            raise out
        return out

    def move_checked_joint(chk, q, v, a, *, timeout_s=None):
        executed.append((chk, int(round(float(q[0])))))
        return NS(ok=True, reason="trajectory succeeded", motion_duration_s=1.0, t_send_ns=0, t_stop_ns=0,
                  as_dict=lambda: {"ok": True})
    be = NS(check_joint=check_joint, move_checked_joint=move_checked_joint, estopped=False)
    decisions, moves = [], []
    t = NS(rs=NS(backend=be, follow_from=follow_ok), transit_speed=0.06, cfg=NS(v_eff_max_mps=0.06), accel=0.1, motion_timeout_s=None,
           res=NS(n_move_l=0), _motion_sent=False, calls=calls, decisions=decisions, moves=moves)
    t._decide = lambda step, msg, **d: decisions.append((step, msg, d))
    t._check_stop = lambda where: None
    t._duration_bounds = (lambda cands, v: list(bounds)) if bounds is not None else (lambda cands, v: [0.0] * len(cands))
    t._T_now = lambda: np.eye(4)
    t._confirm = lambda *a, **k: None
    t._check_start_pose = lambda *a, **k: None
    t._event = lambda *a, **k: None
    t._p = lambda *a, **k: None
    t._jtc_tracking = lambda t0, t1: None
    t._lever_m = lambda: 0.4
    t._yaw_band_rad = lambda: math.radians(3.0)

    def record(name, phase, T_from, T_to, stats, speed, res=None, refused=None):
        d = {"name": name, "dist_mm": 100.0, "rot_deg": 0.0, "refused": refused}
        moves.append(d)
        return d
    t._record_move = record
    for nm in ("_select_joint", "_joint_move", "_prefer_yaw_band"):
        setattr(t, nm, MethodType(getattr(AT.AutoTeach, nm), t))
    return t


def test_resort_regression_the_executed_candidate_is_the_selected_one_with_its_own_trajectory():
    """KIRO's harness (resort_test): both candidates blocked, candidate 1's straight estimate 5 s but its OMPL plan 42 s, candidate 2's
    estimate 10 s and OMPL 12 s. KIRO finalized candidate 1, re-sorted, planned candidate 2 too and executed it while the log marked
    candidate 1. Here: the selection is candidate 2 (12 s), _joint_move sends exactly candidate 2's checked trajectory, plans nothing
    again and marks only candidate 2 selected."""
    executed = []
    final_1, final_2 = _ompl(42.0), _ompl(12.0)
    t = _teach({(1, False): _blocked(5.0), (2, False): _blocked(10.0), (1, True): final_1, (2, True): final_2},
               bounds=[2.0, 3.0], executed=executed)
    cands = [_cand(1), _cand(2)]
    sel = t._select_joint(cands, "acquire_S")
    assert sel["cand"] is cands[1] and sel["check"] is final_2 and sel["duration_s"] == 12.0
    n_checks = len(t.calls)
    chosen = t._joint_move(cands, np.eye(4), "acquire_S", "the dwell", selection=sel)
    assert chosen is cands[1] and executed == [(final_2, 2)] and len(t.calls) == n_checks          # no planning, no re-sort
    rows = t.moves[-1]["joint_candidates"]
    assert [r_["selected"] for r_ in rows] == [False, True] and rows[1]["duration_s"] == 12.0 and rows[0]["duration_s"] == 42.0
    assert t.moves[-1]["joint_selection"]["candidate"] == 2 and t.moves[-1]["joint_selection"]["n_ompl"] == 2


def test_joint_move_without_a_selection_searches_then_sends_that_check():
    executed = []
    first = _ok(32.0)
    t = _teach({(1, False): first, (2, False): _ok(352.5)}, bounds=[1.0, 400.0], executed=executed)
    chosen = t._joint_move([_cand(1), _cand(2)], np.eye(4), "start_pose", "the start pose")
    assert chosen["tag"] == 1 and executed == [(first, 1)] and t.calls == [(1, False)]


def test_platform_unavailable_refuses_the_move_at_once():
    executed = []
    t = _teach({(1, False): PlatformUnavailable("move_group /check_state_validity did not answer (is move_group running?)"),
                (2, False): _ok(1.0)}, executed=executed)
    with pytest.raises(AT._Abort) as e:
        t._joint_move([_cand(1), _cand(2)], np.eye(4), "acquire_S", "the dwell")
    assert e.value.status == "REFUSED" and e.value.reason.startswith("acquire_S: move_group not answering (")
    assert e.value.reason.endswith("nothing sent") and e.value.category == "motion"
    assert t.calls == [(1, False)] and executed == []


def test_no_usable_path_is_refused_with_the_reasons_and_nothing_sent():
    executed = []
    t = _teach({(1, False): {"ok": False, "reason": "joint goal outside the URDF limits", "plan": None},
                (2, False): _blocked(3.0), (2, True): {"ok": False, "reason": "MoveIt joint-space planning failed", "plan": None}},
               executed=executed)
    with pytest.raises(AT._Abort) as e:
        t._joint_move([_cand(1), _cand(2)], np.eye(4), "acquire_S", "the dwell")
    assert e.value.status == "REFUSED" and "URDF limits" in e.value.reason and executed == []
    assert t.moves[-1]["refused"] and [r_["checked"] for r_ in t.moves[-1]["joint_candidates"]] == ["straight", "ompl"]


def test_yaw_band_preference_walks_the_duration_order_not_the_supplied_order():
    """Supplied order 1, 2, 3; durations 30, 10, 20 s. Candidate 1 (supplied first) follows the yaw band, 2 (the shortest) does not,
    3 (second shortest) does -> 3: the preference applies to the duration order (review B1: a promotion before the duration sort
    was lost)."""
    executed = []
    follows = {1: True, 2: False, 3: True}
    t = _teach({(1, False): _ok(30.0), (2, False): _ok(10.0), (3, False): _ok(20.0)}, bounds=[9.0, 9.0, 9.0], executed=executed,
               follow_ok=lambda q, ch, sigma_radius_m: {"ok": follows[int(round(float(q[0])))]})
    cands = [{**_cand(k), "sign": 1} for k in (1, 2, 3)]
    job = NS(normal_unit=np.array([0.0, 0.0, 1.0]), start=np.zeros(3), end=np.array([0.2, 0.0, 0.0]))
    ctx = {1: {"job": job, "T_in": np.eye(4), "T_out": np.eye(4)}}
    sel = t._select_joint(cands, "acquire_S", prefer=lambda s: t._prefer_yaw_band(s, ctx))
    assert sel["cand"]["tag"] == 3 and sel["prefer"]["promoted"] == 1 and "duration rank 2" in sel["prefer"]["text"]
    assert cands[1]["yaw_band_ok"] is False and cands[2]["yaw_band_ok"] is True and "yaw_band_ok" not in cands[0]
    follows[3] = False
    t2 = _teach({(1, False): _ok(30.0), (2, False): _ok(10.0), (3, False): _ok(20.0)}, bounds=[9.0, 9.0, 9.0], executed=executed,
                follow_ok=lambda q, ch, sigma_radius_m: {"ok": follows[int(round(float(q[0])))]})
    cands2 = [{**_cand(k), "sign": 1} for k in (1, 2, 3)]
    sel2 = t2._select_joint(cands2, "acquire_S", prefer=lambda s: t2._prefer_yaw_band(s, ctx))
    assert sel2["cand"]["tag"] == 1 and sel2["prefer"]["promoted"] == 2      # third in duration order (30 s) - within the budget
    follows[1] = False
    t3 = _teach({(1, False): _ok(30.0), (2, False): _ok(10.0), (3, False): _ok(20.0)}, bounds=[9.0, 9.0, 9.0], executed=executed,
                follow_ok=lambda q, ch, sigma_radius_m: {"ok": follows[int(round(float(q[0])))]})
    sel3 = t3._select_joint([{**_cand(k), "sign": 1} for k in (1, 2, 3)], "acquire_S", prefer=lambda s: t3._prefer_yaw_band(s, ctx))
    assert sel3["cand"]["tag"] == 2 and sel3["prefer"]["promoted"] is None and "the shortest kept" in sel3["prefer"]["text"]


def test_jtc_tracking_record_names_the_worst_joint_and_when():
    t = np.array([0, 10, 20, 30], dtype=np.int64) * 1_000_000
    ref = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.05], [0.3, 0.1]])
    fb = np.array([[0.0, 0.0], [0.05, 0.0], [0.04, 0.05], [0.25, 0.1]])
    r = AT.jtc_tracking_record({"ctrl_t_rx_ns": t, "ctrl_joint_names": np.array(["j1", "j2"]), "ctrl_ref_positions_rad": ref,
                                "ctrl_fb_positions_rad": fb})
    assert r["n"] == 4 and r["worst_joint"] == "j1" and r["max_error_rad"] == pytest.approx([0.16, 0.0])
    assert r["t_of_max_s"][0] == pytest.approx(0.02) and r["span_s"] == pytest.approx(0.03)
    assert AT.jtc_tracking_record({})["n"] == 0


def test_motion_error_from_the_checked_send_is_aborted_with_the_tracking_window():
    executed = []
    t = _teach({(1, False): _ok(5.0)}, executed=executed)

    def boom(chk, q, v, a, *, timeout_s=None):
        raise MotionError("checked joint plan start changed: max joint 0.2000 deg - plan and confirm again (nothing sent)")
    t.rs.backend.move_checked_joint = boom
    t._jtc_tracking = lambda t0, t1: {"n": 0, "window": [t0, t1]}
    with pytest.raises(AT._Abort) as e:
        t._joint_move([_cand(1)], np.eye(4), "acquire_S", "the dwell")
    assert e.value.status == "ABORTED" and "start changed" in e.value.reason
    assert t.moves[-1]["motion"]["error"].startswith("checked joint plan start changed") and t.moves[-1]["motion"]["tracking"]["n"] == 0
