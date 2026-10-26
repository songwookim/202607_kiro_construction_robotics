"""weld_orient (user 2026-10-13): per pass the collision-free, reachable torch orientation closest to the WPS nominal (no bound on the
deviation; the deviation is information), one TCP roll per job, the required clearance from the stated uncertainties."""
import math

import numpy as np
import pytest

import scan_core.weld_orient as WO
from scan_core.weld_clearance import tool_body_from_triangles
from scan_core.weld_plan import WeldProcess, _tcp_R, weld_plan_from_features
from scan_core.weld_torch import WeldTorch

from test_weld_plan import SPEC, extent, synth_feats
from test_weld_torch import NOZZLE_TEST_MM, TORCH


@pytest.fixture(scope="module")
def plan():
    f = synth_feats()
    p = weld_plan_from_features(f, extent(f), SPEC, WeldProcess())
    return p


@pytest.fixture(scope="module")
def torch():
    return WeldTorch.from_dict(TORCH).with_nozzle_end(NOZZLE_TEST_MM, "test (the CAD nozzle end)")


def far_body():
    """A small triangle 150 mm up the torch axis behind the TCP: never near the specimen."""
    tri = np.array([[[0.0, 0.0, -0.150], [0.005, 0.0, -0.150], [0.0, 0.005, -0.150]]])
    return tool_body_from_triangles(tri, tcp_frame="torch_tcp", source="test far")


def guard_body():
    """A plate like the sensor guard: 47 mm off the torch axis along the TCP -x, reaching 16 mm beyond the TCP plane."""
    x0, x1, y0, y1, z = -0.060, -0.030, -0.020, 0.020, 0.016
    tri = np.array([[[x0, y0, z], [x1, y0, z], [x1, y1, z]], [[x0, y0, z], [x1, y1, z], [x0, y1, z]]])
    return tool_body_from_triangles(tri, tcp_frame="torch_tcp", source="test guard")


def test_requirement_is_k_times_the_root_sum_square_of_the_documented_sigmas():
    r = WO.clearance_requirement(3.0, [("a", 0.0003, "x"), ("b", 0.0004, "y"), ("c", None, "not documented")])
    assert r.sigma_mm == pytest.approx(0.5) and r.required_mm == pytest.approx(1.5) and r.missing == ["c"]
    d = r.to_dict()
    assert d["complete"] is False and d["sources"][2]["sigma_mm"] is None and "lower bound" in r.describe()


def test_candidate_grid_starts_at_the_nominal_and_grows_by_the_torch_axis_angle(plan):
    wp = plan.passes[0]
    n, w, d, tilt = WO._frame(plan, wp)
    t0, w0 = WO._nominal(wp)
    assert (t0, w0) == (-42.0, 9.0)                                           # the WPS root (KIRO's September welds, bevel left)
    T, W, dev = WO.candidate_grid(t0, w0, n, d, tilt)
    assert (T[0], W[0], dev[0]) == (t0, w0, 0.0) and np.all(np.diff(dev) >= -1e-12)
    assert np.all(np.abs(T) < 90.0) and np.all(np.abs(W) < 90.0) and len(T) == 179 * 179
    a0 = WO.torch_axis(n, d, tilt, t0, w0)
    k = 1234
    assert dev[k] == pytest.approx(math.degrees(math.acos(float(np.clip(WO.torch_axis(n, d, tilt, T[k], W[k]) @ a0, -1, 1)))))


def test_retarget_rebuilds_the_rotations_and_the_dense_path_and_keeps_the_nominal(plan):
    wp = plan.passes[3]
    new = WO.retarget_pass(plan, wp, -20.0, 35.0)
    n, w, d, tilt = WO._frame(plan, wp)
    R = _tcp_R(n, w, d, tilt, -20.0, 35.0)
    assert np.allclose(new.R_base_tcp, R) and np.allclose(new.dense_R, R, atol=1e-12)
    assert (new.travel_angle_deg, new.work_angle_deg, new.travel_nominal_deg, new.work_nominal_deg) == (-20.0, 35.0, -44.0, 7.0)
    assert np.allclose(new.aim_base_m, wp.aim_base_m) and new.weave_amp_mm[0] == pytest.approx(4.0 * abs(float(R[:, 1] @ w)))
    assert wp.R_base_tcp is not new.R_base_tcp and np.allclose(wp.R_base_tcp[0], _tcp_R(n, w, d, tilt, -44.0, 7.0))


def fake_check(margins=None, refuse=()):
    calls = []

    def check(p, only=None):
        calls.append((float(p.torch_roll_deg), list(only)))
        out = []
        for wp in p.passes:
            if only is not None and wp.k not in only:
                continue
            key = (wp.k, float(p.torch_roll_deg), round(float(wp.travel_angle_deg), 6), round(float(wp.work_angle_deg), 6))
            if key in refuse:
                out.append({"k": wp.k, "reachable": False, "reason": "test refusal"})
            else:
                out.append({"k": wp.k, "reachable": True, "reason": "", "min_limit_margin_rad": (margins or {}).get(p.torch_roll_deg, 0.5)})
        return out
    check.calls = calls
    return check


def test_nominal_kept_when_feasible_and_the_roll_by_the_joint_limit_margin(plan, torch):
    import copy
    p = copy.deepcopy(plan)
    rec = WO.choose_orientation(p, fake_check({0.0: 0.3, 180.0: 0.6}), body=far_body(), required=2.0, torch=torch)
    assert p.torch_roll_deg == 180.0 and "total deviation from the WPS 0.0 deg" in p.torch_roll["reason"]
    for o, wp in zip(rec["passes"], p.passes):
        assert o["deviation_deg"] == 0.0 and o["chosen"] == o["nominal"] and o["why"].startswith("the WPS nominal; roll 180 deg: the "
                                                                                                   "torch neck bends with the travel angle")
        assert wp.orientation == o and wp.travel_angle_deg == wp.travel_nominal_deg
    assert all(f["reachable"] is True for f in p.feasibility)
    lines = p.summary_lines()
    assert any("torch pass 1: nominal travel -42 work +9 -> chosen travel -42 work +9 deg (deviation 0.0 deg)" in ln for ln in lines)


def test_a_refused_candidate_gives_the_next_closest(plan, torch):
    import copy
    p = copy.deepcopy(plan)
    refuse = {(1, r) + WO._nominal(p.passes[0]) for r in (0.0, 180.0)}
    WO.choose_orientation(p, fake_check(refuse=refuse), body=far_body(), required=2.0, torch=torch)
    o = p.passes[0].orientation
    assert o["deviation_deg"] > 0.0 and o["deviation_deg"] <= 1.0 + 1e-9 and len(o["refused_closer"]) == 1
    assert "1 closer one(s) refused by the robot check" in o["why"]
    assert all(q.orientation["deviation_deg"] == 0.0 for q in p.passes[1:])


def test_the_guard_forces_the_closest_clear_orientation_exactly(plan, torch):
    """A guard-like plate below the TCP plane collides at the nominal; the search returns a clear orientation and every grid
    candidate closer to the nominal is refused by the geometry (the exact skipping included). The plate is on the TCP -x side, which
    a push torch lowers toward the plates (a drag torch lifts it): the pass's nominal here is a push (travel +14 / work +22 deg, a
    test input - the WPS default is KIRO's drag since 2026-10-13)."""
    import copy
    import dataclasses
    p = copy.deepcopy(plan)
    body = guard_body()
    req = 1.0
    wt = torch.with_stick_out(20.0, "test")
    wp = WO.retarget_pass(p, dataclasses.replace(p.passes[0], travel_nominal_deg=14.0, work_nominal_deg=22.0), 14.0, 22.0)
    p.passes[0] = wp
    assert WO._nominal(wp) == (14.0, 22.0)
    lr = WO.line_rows(p, wp, 0.002)                                           # the delivered line's samples (weld_line)
    c0 = WO.pass_clearance(p, wp, body, wt, 0.0, rows=lr)
    assert c0["clearance_mm"] < req and c0["n_samples"] == len(lr.tip)
    first = next(WO.search_pass(p, wp, body, wt, 0.0, req, rows=lr))
    assert first["deviation_deg"] > 0.0 and first["clearance"]["clearance_mm"] >= req
    rj = first["rejected"]                                                    # why the nominal and the closer ones are out
    assert rj["nominal"]["constraint"] == "clearance" and rj["nominal"]["part"] == "tool" and rj["n"] >= 1
    # branch and bound (choose_orientation's bound from a roll that already has every pass feasible): no candidate past the bound
    cut = next(WO.search_pass(p, wp, body, wt, 0.0, req, rows=lr, max_deviation_deg=0.5 * first["deviation_deg"]))
    assert cut.get("exhausted") and cut["cut_deg"] == pytest.approx(0.5 * first["deviation_deg"])
    same = next(WO.search_pass(p, wp, body, wt, 0.0, req, rows=lr, max_deviation_deg=first["deviation_deg"]))   # the bound is inclusive
    assert not same.get("exhausted") and same["deviation_deg"] == first["deviation_deg"]
    n, w, d, tilt = WO._frame(p, wp)
    T, W, dev = WO.candidate_grid(*WO._nominal(wp), n, d, tilt)
    closer = np.flatnonzero(dev < first["deviation_deg"] - 1e-9)
    for m in closer[:: max(1, len(closer) // 60)]:                           # a sample of the closer ones, all infeasible on the line
        cand = WO.retarget_pass(p, wp, float(T[m]), float(W[m]))
        assert WO.pass_clearance(p, cand, body, wt, 0.0, threshold=req, rows=lr)["below"], (T[m], W[m], dev[m])


def test_no_feasible_orientation_at_any_deviation_fails_the_plan_with_the_reason(plan, torch, monkeypatch):
    import copy
    monkeypatch.setattr(WO, "ANGLE_STEP_DEG", 30.0)                            # a coarse grid: the logic, quickly
    p = copy.deepcopy(plan)
    p.passes = p.passes[:1]
    big = np.array([[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]])   # a 2 m triangle through the TCP plane
    body = tool_body_from_triangles(big, tcp_frame="torch_tcp", source="test")
    WO.choose_orientation(p, fake_check(), body=body, required=1.0, torch=torch)
    f = p.feasibility[0]
    assert f["reachable"] is False and f["reason"].startswith("no torch orientation at any deviation is feasible: none of ")
    assert p.passes[0].orientation["chosen"] is None and "no roll makes every pass feasible" in p.torch_roll["reason"]
    assert any("NO feasible orientation" in ln for ln in p.summary_lines())


def test_curved_seam_each_section_keeps_its_own_frame(plan, torch):
    """A curved seam (each station's u, w turned about n): the plan builds every station's torch frame from that station's section
    frame, retarget_pass does the same; the delivered line has ONE base rotation (their mean) and each of its samples is evaluated in
    the section of its nearest station - the clearance rows' poses are the base-frame poses mapped into that station's own frame."""
    import dataclasses as dc
    from scan_core.weld_plan import plan_weld
    from scipy.spatial.transform import Rotation as Rot
    st0 = plan.stations
    curved = []
    for i, st in enumerate(st0):
        Rz = Rot.from_rotvec(np.asarray(st.n, float) * math.radians(0.04 * i)).as_matrix()     # 3 deg over the seam
        curved.append(dc.replace(st, u=Rz @ st.u, w=Rz @ st.w))
    p = plan_weld(curved, plan.process, spec_design=plan.spec_design)
    wt = torch.with_stick_out(20.0, "test")
    from scan_core.weld_clearance import rows_poses
    from scan_core.weld_line import station_rotations, weld_line
    for wp in (p.passes[0], p.passes[-1]):
        new = WO.retarget_pass(p, wp, 5.0, 12.0)
        n, w, d, tilt = WO._frame(p, wp)
        by_s = {float(st.s_mm): st for st in p.stations}
        for j, sv in enumerate(new.s_mm):                                  # every station's frame in its own section frame
            st = by_s[float(sv)]
            dj = st.u if wp.direction == "S->E" else -st.u
            assert np.allclose(new.R_base_tcp[j], _tcp_R(st.n, st.w, dj, WO.pass_tilt(p, wp) * st.w, 5.0, 12.0), atol=1e-12)
        assert np.array_equal(new.R_base_tcp, station_rotations(p, wp, 5.0, 12.0))
        lr = WO.line_rows(p, new, 0.002)
        ln = weld_line(p, new, wt, 180.0)
        smp = ln.samples(0.002)
        for name, off in ln.offsets():
            P = smp["tcp"] + off
            R_sec, P_sec = rows_poses(lr.rows, ln.R_tcp, P)
            for j in range(0, len(P), 17):
                st = p.stations[int(lr.station[j])]
                A = np.stack([st.w, st.u, st.n])
                assert np.allclose(A @ ln.R_tcp, R_sec[j], atol=1e-12)
                assert np.allclose(A @ (P[j] - st.origin) * 1e3, P_sec[j], atol=1e-9)
        assert len({tuple(np.round(r_, 12).ravel()) for r_ in rows_poses(lr.rows, ln.R_tcp, smp["tcp"])[0]}) > 1   # the sections turn


def _role_tool(mirror: bool = False):
    """A tool like ours in the TCP frame: the gas nozzle tube around the axis (rim 1.2 mm beyond the TCP plane), the contact tip in its
    bore (end face = the TCP plane), a ring on the neck behind (around the axis, outside the nozzle's outline) and a guard-like plate
    beside the axis 16 mm beyond the TCP plane (on -x_TCP; mirror: on +x_TCP - the same tool turned 180 deg about the torch axis, so
    at TCP roll 180 it sits where the plain one sits at roll 0)."""
    from test_weld_torch import _ring
    from scan_core.weld_clearance import tool_body_from_triangles
    nozzle = _ring(0.008, 0.010, -0.070, 0.0012)
    tip = _ring(0.0006, 0.0031, -0.020, 0.0)
    neck_ring = _ring(0.010, 0.014, -0.090, -0.072)
    x0, x1, y0, y1, z = -0.060, -0.030, -0.020, 0.020, 0.016
    if mirror:
        x0, x1, y0, y1 = -x0, -x1, -y0, -y1
    guard = np.array([[[x0, y0, z], [x1, y0, z], [x1, y1, z]], [[x0, y0, z], [x1, y1, z], [x0, y1, z]]])
    tris = [nozzle, tip, neck_ring, guard]
    fl = np.concatenate([np.full(len(a), i, dtype=int) for i, a in enumerate(tris)])
    return tool_body_from_triangles(np.concatenate(tris), tcp_frame="torch_tcp", source="test", links=("noz", "tip", "neck", "guard"),
                                    face_link=fl)


def test_tool_roles_are_the_nozzle_and_the_parts_in_its_bore():
    """weld_clearance.tool_roles (user 2026-10-13): the process parts = the gas nozzle (of the parts the torch axis passes through, the
    one reaching farthest along +z_TCP) and the parts within its outline seen along the axis (the contact tip); a neck ring around the
    axis but outside the nozzle's outline and a plate beside the axis are 'other'."""
    from scan_core.weld_clearance import OTHER_ROLE, PROCESS_ROLE, tool_roles
    r = tool_roles(_role_tool())
    assert [k for k, _b in r.roles()] == [PROCESS_ROLE, OTHER_ROLE]
    assert set(r.process_links) == {"noz", "tip"} and len(r.process.F) == 2 * (24 * 8)
    assert {r.other.links[i] for i in np.unique(r.other.face_link)} == {"neck", "guard"}
    assert r.process.V[:, 2].max() == pytest.approx(0.0012) and "rim 1.20 mm" in r.how
    assert tool_roles(r) is r and tool_roles(_role_tool()) is not r


def test_role_requirements_let_the_nozzle_approach_and_keep_the_others_at_k_sigma(plan, torch):
    """Per role (user 2026-10-13, KIRO: the gas nozzle almost touches the plate in the root): the process parts must only not
    penetrate the specimen (PROCESS_REQUIRED_MM = 0), every other part keeps the required r. The same tool as one body needs r for the
    nozzle too and so leaves the nominal; by role the nominal stays when the nozzle clears 0 and the other parts r - and the record
    carries each role's clearance, its requirement and the limiting role."""
    import copy
    from scan_core.weld_clearance import tool_roles
    from scan_core.weld_orient import PROCESS_REQUIRED_MM
    assert PROCESS_REQUIRED_MM == 0.0
    p = copy.deepcopy(plan)
    p.passes = p.passes[2:3]                                                     # the second fill (room for the nozzle)
    wt = torch.with_stick_out(15.0, "test")                                      # the nozzle rim 15 mm from the aim point
    wp = p.passes[0]
    body = _role_tool(mirror=True)                                               # at roll 180 (the drag's roll, NECK_RULE) the guard
    roles = tool_roles(body)                                                     # sits where the plain tool's sits at roll 0
    c_all = WO.pass_clearance(p, wp, body, wt, 180.0)
    c_roles = WO.pass_clearance(p, wp, roles, wt, 180.0, required_mm=8.0)
    br = c_roles["by_role"]
    assert set(br) == {"process", "other"} and br["process"]["required_mm"] == 0.0 and br["other"]["required_mm"] == 8.0
    assert c_all["clearance_mm"] == pytest.approx(min(br["process"]["clearance_mm"], br["other"]["clearance_mm"]), abs=1e-9)
    assert c_roles["clearance_mm"] == br["other"]["clearance_mm"]                # the k x sigma comparison: the other parts
    assert c_roles["excess_mm"] == pytest.approx(min(br["process"]["clearance_mm"], br["other"]["clearance_mm"] - 8.0))
    req = br["process"]["clearance_mm"] + 0.5                                    # between the nozzle's clearance and the others'
    assert 0.0 < br["process"]["clearance_mm"] < req < br["other"]["clearance_mm"], br
    first_all = next(WO.search_pass(p, wp, body, wt, 180.0, req))                # one body: the nozzle needs req too
    first_roles = next(WO.search_pass(p, wp, roles, wt, 180.0, req))
    assert first_all["deviation_deg"] > 0.0 and first_roles["deviation_deg"] == 0.0
    assert first_roles["clearance"]["by_role"]["process"]["clearance_mm"] >= 0.0
    rec = WO.choose_orientation(p, fake_check(), body=roles, required=req, torch=wt)
    o = rec["passes"][0]
    assert o["deviation_deg"] == 0.0 and o["clearance_by_role"]["process"]["required_mm"] == 0.0
    assert o["clearance_by_role"]["other"]["required_mm"] == req and o["clearance_mm"] >= req
    assert o["limiting_part"]["role"] == "process" and o["limiting_part"]["link"] in ("noz", "tip")
    assert rec["process_required_mm"] == 0.0 and "process parts = the gas nozzle" in rec["tool_roles"]
    assert any("gas nozzle + contact tip" in ln and "limiting: process parts" in ln for ln in p.summary_lines())


# --------------------------------------------------------------------------------------
# the torch neck bends with the travel angle (user 2026-10-13): the roll is defined by the travel angle
# --------------------------------------------------------------------------------------
def _plan_travel(travel):
    f = synth_feats()
    return weld_plan_from_features(f, extent(f), SPEC, WeldProcess(travel_angle_deg=travel))


def test_the_body_axis_and_the_roll_side_come_from_the_robot_models_torch(torch):
    """The straight body = the flange +z in the TCP frame, from the torch model's flange <- TCP (on the arm: TF): our torch
    0.5 x_TCP + 0.866 z_TCP (the 30 deg bend toward +x_TCP); roll 0 bends it toward the travel (push), roll 180 away (drag)."""
    b = WO.body_axis_tcp(torch)
    assert np.allclose(b, [0.5, 0.0, math.sqrt(3.0) / 2.0], atol=1e-5)
    assert (WO.neck_side(torch, 0.0), WO.neck_side(torch, 180.0)) == (1, -1)
    assert WO.roll_for_travel(torch, 10.0) == [0.0] and WO.roll_for_travel(torch, -42.0) == [180.0]
    assert WO.roll_for_travel(torch, 0.0) == [0.0, 180.0] and WO.roll_for_travel(None, -42.0) == [0.0, 180.0]
    import dataclasses as _dc
    from scan_core.rigid import RigidTransform
    straight = _dc.replace(torch, flange_from_tcp=RigidTransform(R=np.eye(3), t=np.zeros(3)))     # no bend: both rolls
    assert WO.neck_side(straight, 0.0) == 0 and WO.roll_for_travel(straight, 10.0) == [0.0, 180.0]
    # the candidate grid of a roll keeps that roll's travel side (0 fits both)
    plan = _plan_travel((10.0, 10.0, 10.0))
    n, w, d, tilt = WO._frame(plan, plan.passes[0])
    T, W, dev = WO.candidate_grid(10.0, -9.0, n, d, tilt, travel_side=1)
    assert T.min() == 0.0 and (T[0], W[0], dev[0]) == (10.0, -9.0, 0.0)
    T2, _W2, dev2 = WO.candidate_grid(10.0, -9.0, n, d, tilt, travel_side=-1)
    assert T2.max() == 0.0 and dev2[0] > 0.0


@pytest.mark.parametrize("travel,roll,nozzle,body", [((10.0, 10.0, 10.0), 0.0, 10.0, 40.0),
                                                      ((-42.0, (-41.0, -44.0), -44.0), 180.0, -42.0, -72.0)])
def test_push_takes_the_roll_with_the_body_ahead_and_drag_the_body_behind(torch, travel, roll, nozzle, body):
    """Push +10: the roll whose straight body leans +40 deg (same side as the nozzle, more inclined - the parts after the bend face
    the travel direction); KIRO's drag -42: roll 180, body -72 (KIRO's identified September posture). The other roll is not searched
    (every pass refused there with the reason); the per-pass 'why' and the record carry the roll's reason and the leans."""
    import copy
    p = copy.deepcopy(_plan_travel(travel))
    rec = WO.choose_orientation(p, fake_check(), body=far_body(), required=2.0, torch=torch)
    assert p.torch_roll_deg == roll and p.torch_roll["neck"]["side_by_roll"] == {"0": 1, "180": -1}
    other = f"{180.0 - roll:g}"
    assert sorted(p.torch_roll["neck"]["refused_by_roll"][other]) == ["1", "2", "3", "4"] and not p.torch_roll["neck"]["refused_by_roll"][f"{roll:g}"]
    assert "the torch neck would bend against the travel angle - NECK_RULE" in p.torch_roll["reason"]
    assert all(r_["chosen"] is None and "NECK_RULE" in r_["why"] for r_ in rec["by_roll"][other])
    o = rec["passes"][0]
    assert o["roll"]["nozzle_lean_deg"] == pytest.approx(nozzle, abs=0.6) and o["roll"]["body_lean_deg"] == pytest.approx(body, abs=0.6)
    assert all(q["roll"]["with_travel"] and abs(q["roll"]["body_lean_deg"]) > abs(q["roll"]["nozzle_lean_deg"])
               and np.sign(q["roll"]["body_lean_deg"]) == np.sign(q["roll"]["nozzle_lean_deg"]) for q in rec["passes"])
    assert all("the torch neck bends with the travel angle (NECK_RULE)" in q["why"] for q in rec["passes"])
    assert NECK_IN(p.torch_roll["rule"])


def NECK_IN(text):
    return "the torch neck bends with the travel angle" in text


def test_no_flip_a_pass_whose_roll_has_no_feasible_orientation_is_refused(torch):
    """Mixed travel signs (root drag, fill / cap push): one roll per job, each pass only in its own roll - the passes of the other
    side are refused with the NECK_RULE reason (never searched in a roll that bends the neck against them); a robot check refusing
    every orientation of a pass in its roll refuses the pass, it does not flip to the other roll."""
    import copy
    p = copy.deepcopy(_plan_travel((-42.0, 10.0, 10.0)))
    rec = WO.choose_orientation(p, fake_check(), body=far_body(), required=2.0, torch=torch)
    assert p.torch_roll_deg == 0.0 and "no roll makes every pass feasible" in p.torch_roll["reason"]
    o1 = rec["passes"][0]
    assert o1["chosen"] is None and "needs roll 180 deg" in o1["why"] and "no flip" in o1["why"]
    assert all(q["chosen"] is not None for q in rec["passes"][1:])
    assert p.feasibility[0]["reachable"] is False and p.feasibility[0]["refusal"]["constraint"] == "torch neck vs travel angle"
    # every candidate of pass 1 in its roll refused by the robot check: refused, the other roll is never used for it
    p2 = copy.deepcopy(_plan_travel((-42.0, (-41.0, -44.0), -44.0)))

    def check(trial, only=None):
        return [{"k": wp.k, "reachable": wp.k != 1, "reason": "test refusal" if wp.k == 1 else "",
                 "min_limit_margin_rad": 0.5} for wp in trial.passes if only is None or wp.k in only]
    calls = []

    def cands(wp, roll, bound):                                    # the search, cut after 2 candidates (a short grid)
        calls.append((wp.k, roll))
        for i, c in enumerate(WO.search_pass(p2, wp, far_body(), torch, roll, 2.0, max_deviation_deg=bound)):
            if c.get("exhausted") or i >= 2:
                yield {"exhausted": True, "n_evaluated": i, "best_seen_mm": float("nan"), "best_seen_at": None,
                       "rejected": c.get("rejected") or {}}
                return
            yield c
    rec2 = WO.choose_orientation(p2, check, body=far_body(), required=2.0, torch=torch, candidates=cands)
    assert p2.torch_roll_deg == 180.0 and rec2["passes"][0]["chosen"] is None
    assert "no torch orientation at any deviation is feasible: 2 candidate(s)" in rec2["passes"][0]["why"]
    assert all(r_["chosen"] is None for r_ in rec2["by_roll"]["0"]) and {r_ for _k, r_ in calls} == {180.0}   # roll 0 never searched


def test_travel_zero_keeps_both_rolls(torch):
    import copy
    p = copy.deepcopy(_plan_travel((0.0, 0.0, 0.0)))
    rec = WO.choose_orientation(p, fake_check({0.0: 0.3, 180.0: 0.6}), body=far_body(), required=2.0, torch=torch)
    assert p.torch_roll_deg == 180.0 and all(r_["chosen"] is not None for r_ in rec["by_roll"]["0"])   # both searched, margin decides
    assert all("the WPS travel is 0 deg - both rolls fit" in q["why"] for q in rec["passes"])
