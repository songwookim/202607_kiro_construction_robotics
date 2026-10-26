"""scan_core.weld_line (user 2026-10-13): D14 verifies exactly what is delivered and executed. The KIRO yaml poses are D14's checked
poses (and the robot check's), result.paths' wire-end line is the same line; the per-row rotations of the clearance model are exact;
a seam whose straight delivered line runs below the per-station layer plan no longer delivers a penetrating pass - D14 picks another
orientation (or refuses) and says why."""
import copy
import dataclasses
import math

import numpy as np
import pytest

import scan_core.weld_orient as WO
from scan_core import kiro_plan as KP
from scan_core.weld_clearance import clearance, pass_geometry, rows_geometry, rows_poses, tool_roles
from scan_core.weld_line import line_rotation, rotation_mean, segment_points, station_rotations, weld_line
from scan_core.weld_plan import WeldProcess, plan_weld, weld_plan_from_features
from scan_core.weld_torch import WeldTorch, roll_matrix

from test_weld_orient import _role_tool, fake_check
from test_weld_plan import SPEC, extent, synth_feats
from test_weld_torch import NOZZLE_TEST_MM, TORCH

WORLD = {"enabled": True, "world_frame": "link0"}


@pytest.fixture(scope="module")
def plan():
    f = synth_feats()
    return weld_plan_from_features(f, extent(f), SPEC, WeldProcess())


def _torch(stick_mm=20.0):
    return WeldTorch.from_dict(TORCH).with_nozzle_end(NOZZLE_TEST_MM, "test (the CAD nozzle end)").with_stick_out(stick_mm, "test")


def test_per_row_rotations_are_exact(plan):
    """clearance() with one rotation per row (a straight move at one base rotation seen in several stations' sections) gives every
    row's own value: the same as one call per row, in both modes - the bounds are widened by the rows' rotation spread, never cut."""
    from scipy.spatial.transform import Rotation as Rot
    roles = tool_roles(_role_tool())
    wp = plan.passes[2]
    geo = pass_geometry(plan, wp)
    rng = np.random.default_rng(3)
    st = np.sort(rng.integers(0, len(geo.s_mm), 9))
    rows = rows_geometry(geo, st)
    ln = weld_line(plan, wp, _torch(15.0), 0.0)
    P = ln.samples(0.04)["tcp"][: len(st)]
    R_sec, P_sec = rows_poses(rows, ln.R_tcp, P)
    R_sec = np.einsum("nij,njk->nik", Rot.from_rotvec(rng.normal(0.0, math.radians(1.5), (len(st), 3))).as_matrix(), R_sec)
    for body in (roles.process, roles.other):
        one = [clearance(body, rows_geometry(geo, st[j:j + 1]), R_sec[j], P_sec[j:j + 1])["clearance_mm"] for j in range(len(st))]
        got = clearance(body, rows, R_sec, P_sec)
        assert got["clearance_mm"] == pytest.approx(min(one), abs=1e-9) and one[got["station"]] == pytest.approx(min(one), abs=1e-9)
        for thr in (min(one) - 0.3, min(one) + 0.3):
            assert clearance(body, rows, R_sec, P_sec, threshold=thr)["below"] == (min(one) < thr)


def test_the_line_rotation_of_a_candidate_is_the_retargeted_pass_line_and_turns_like_the_first_station(plan):
    """A candidate's line rotation (the rotation mean of its station torch frames) is exactly the delivered line's rotation of the
    retargeted pass; on a curved seam two candidates' line rotations differ by the angle between their first-station frames (the
    premise of the search's exact skipping)."""
    from scipy.spatial.transform import Rotation as Rot
    curved = []
    for i, st in enumerate(plan.stations):
        Rz = Rot.from_rotvec(np.asarray(st.n, float) * math.radians(0.04 * i)).as_matrix()     # 3 deg over the seam
        curved.append(dataclasses.replace(st, u=Rz @ st.u, w=Rz @ st.w))
    p = plan_weld(curved, plan.process, spec_design=plan.spec_design)
    wp = p.passes[3]
    for t_, w_ in ((-44.0, -7.0), (-20.0, 35.0)):
        new = WO.retarget_pass(p, wp, t_, w_)
        assert np.array_equal(new.R_base_tcp, station_rotations(p, wp, t_, w_))
        assert np.array_equal(line_rotation(new), rotation_mean(station_rotations(p, wp, t_, w_)))
    n, w, d, tilt = WO._frame(p, wp)
    from scan_core.weld_plan import _tcp_R

    def ang(A, B):
        return math.degrees(Rot.from_matrix(A.T @ B).magnitude())
    pairs = [((-44.0, -7.0), (-30.0, 12.0)), ((-10.0, 40.0), (25.0, -15.0))]
    for (a1, b1), (a2, b2) in pairs:
        L1, L2 = (rotation_mean(station_rotations(p, wp, a_, b_)) for a_, b_ in ((a1, b1), (a2, b2)))
        assert ang(L1, L2) == pytest.approx(ang(_tcp_R(n, w, d, tilt, a1, b1), _tcp_R(n, w, d, tilt, a2, b2)), abs=1e-9)


def test_the_kiro_yaml_poses_are_exactly_d14s_checked_poses(plan, tmp_path):
    """The one source (user 2026-10-13: checked = delivered = executed): D14's orientation search records the line poses it checked;
    the platform's robot check gets the same line; kiro/pass_<k>.yaml carries exactly those TCP poses (World = the base here) and the
    wire-end line is the pass's planned start / goal aim."""
    p = copy.deepcopy(plan)
    wt = _torch(20.0)
    p.torch = wt                                                                # the plan's torch (plan_weld with the torch)
    seen = {}
    base = fake_check()

    def check(trial, only=None):
        for wp in trial.passes:
            if only is None or wp.k in only:
                seen[(wp.k, float(trial.torch_roll_deg), float(wp.travel_angle_deg), float(wp.work_angle_deg))] = \
                    weld_line(trial, wp, wt, trial.torch_roll_deg)
        return base(trial, only=only)
    rec = WO.choose_orientation(p, check, body=tool_roles(_role_tool()), required=1.0, torch=wt, step_m=0.002)
    assert p.line_step_mm == pytest.approx(2.0) and rec["line_step_mm"] == pytest.approx(2.0)
    cfg = KP.KiroOutputConfig.from_dict(WORLD, wt)
    wfb, notes = KP.resolve_world_from_base(cfg, p.base_frame)
    KP.write_kiro_outputs(p, tmp_path, cfg, world_from_base=wfb, notes=notes)
    for o, wp in zip(rec["passes"], p.passes):
        cl = o["checked_line"]
        y = KP.read_pass_yaml(tmp_path / "kiro" / f"pass_{wp.k}.yaml")
        assert np.array_equal(y["start"][0], cl["tcp_start_m"]) and np.array_equal(y["goal"][0], cl["tcp_goal_m"])
        from scan_core.rigid import _quat_to_R
        assert np.allclose(_quat_to_R(y["start"][1], "q"), cl["R_base_tcp"], atol=1e-12)
        assert np.array_equal(_quat_to_R(y["start"][1], "q"), _quat_to_R(y["goal"][1], "q"))
        rob = seen[(wp.k, float(p.torch_roll_deg), float(o["chosen"]["travel_deg"]), float(o["chosen"]["work_deg"]))]
        assert np.array_equal(rob.tcp_start, cl["tcp_start_m"]) and np.array_equal(rob.R_tcp, cl["R_base_tcp"])
        e = KP.pass_endpoints(p, wp)
        assert np.array_equal(cl["tip_start_m"], e["start"]["tip_base_m"]) and np.array_equal(cl["tip_goal_m"], e["goal"]["tip_base_m"])
        assert cl["n_samples"] == len(segment_points(cl["tip_start_m"], cl["tip_goal_m"], 0.002))
        assert o["why"] == "the WPS nominal" or o["deviation_why"]["text"] in o["why"]


def _bowed(plan, bow_mm: float):
    """The plan's seam bowed up along the plate normal by bow_mm in the middle (each station's section moved, its frame kept): the
    per-station layer plan follows it, the delivered straight line between the end aims runs up to bow_mm below it."""
    L = plan.stations[-1].s_mm - plan.stations[0].s_mm
    st = [dataclasses.replace(s, origin=np.asarray(s.origin, float) + np.asarray(s.n, float) * bow_mm * 1e-3
                              * math.sin(math.pi * (s.s_mm - plan.stations[0].s_mm) / L)) for s in plan.stations]
    return plan_weld(st, plan.process, spec_design=plan.spec_design)


def _station_model(p, wp, roles, wt, roll, r_mm):
    """The superseded per-station check (each station's torch frame at that station's aim, in its own section): the layer plan's poses."""
    geo = pass_geometry(p, wp)
    s_idx = {float(s): i for i, s in enumerate(geo.s_mm)}
    order = np.array([s_idx[float(s)] for s in wp.s_mm])
    R_tcp = np.asarray(wp.R_base_tcp, float)[0] @ roll_matrix(roll) @ wt.tcp_from_tip.R.T
    i0 = order[0]
    R_sec = geo.A[i0] @ R_tcp
    aim = np.zeros((len(geo.s_mm), 3))
    aim[order, 0], aim[order, 2] = np.asarray(wp.aim_xz_mm)[:, 0], np.asarray(wp.aim_xz_mm)[:, 1]
    P_sec = aim - (R_sec @ wt.tcp_from_tip.t) * 1e3
    return {r_: clearance(b_, geo, R_sec, P_sec)["clearance_mm"] for r_, b_, _q in WO._role_bodies(roles, r_mm)}


def test_a_straight_line_below_a_bowed_seam_is_not_delivered_into_the_plate(plan, tmp_path):
    """The weldpose finding (2026-10-13): the per-station layer plan cleared, the delivered straight line (one orientation) put the
    gas nozzle into the plate. Here a seam bowed up in the middle: at the WPS nominal the per-station model clears, the delivered line
    does not - D14 (on the line) leaves the nominal, says why (the process parts' clearance), and the KIRO yaml line - re-checked
    per pose (poses_role_clearance on its samples) - clears."""
    roles = tool_roles(_role_tool(mirror=True))                                 # at roll 180 = the plain tool at roll 0
    wt = _torch(15.0)
    p0 = copy.deepcopy(plan)
    p0.passes = p0.passes[2:3]
    c_st = _station_model(p0, p0.passes[0], roles, wt, 180.0, 1.0)["process"]   # roll 180: the drag's roll (NECK_RULE)
    assert c_st > 0.0
    p = _bowed(p0, c_st + 1.5)
    p.passes = [dataclasses.replace(wp, k=1) for wp in p.passes[2:3]]
    wp = p.passes[0]
    assert _station_model(p, wp, roles, wt, 180.0, 1.0)["process"] == pytest.approx(c_st, abs=0.05)   # the layer plan rides the bow
    line0 = WO.pass_clearance(p, wp, roles, wt, 180.0, required_mm=1.0, step_m=0.002)
    assert line0["by_role"]["process"]["clearance_mm"] < 0.0                    # the delivered line at the nominal: into the plate
    p.torch = wt                                                                # the plan's torch (stick-out 15 mm)
    rec = WO.choose_orientation(p, fake_check(), body=roles, required=1.0, torch=wt, step_m=0.002, rolls=(180.0,))
    o = rec["passes"][0]
    if o["chosen"] is None:                                                     # refused with the reason: never delivered
        assert p.feasibility[0]["reachable"] is False and o["why"] and "NECK_RULE" not in o["why"]
        return
    assert o["deviation_deg"] > 0.0 and o["clearance_by_role"]["process"]["clearance_mm"] >= 0.0
    dw = o["deviation_why"]
    assert dw["nominal"]["constraint"] == "clearance" and dw["nominal"]["part"] == "process" and dw["nominal"]["link"] in ("noz", "tip")
    assert "the WPS nominal (travel -44 work +7 deg) is out: clearance (process)" in o["why"]
    cfg = KP.KiroOutputConfig.from_dict(WORLD, wt)
    wfb, notes = KP.resolve_world_from_base(cfg, p.base_frame)
    KP.write_kiro_outputs(p, tmp_path, cfg, world_from_base=wfb, notes=notes)
    y = KP.read_pass_yaml(tmp_path / "kiro" / "pass_1.yaml")
    from scan_core.rigid import _quat_to_R
    R = _quat_to_R(y["start"][1], "q")
    P = segment_points(y["start"][0], y["goal"][0], 0.002)
    chk = WO.poses_role_clearance(roles, pass_geometry(p, p.passes[0]), R, P, wt.tcp_from_tip.t, 1.0)
    assert chk["ok"] and chk["by_role"]["process"]["clearance_mm"] == pytest.approx(o["clearance_by_role"]["process"]["clearance_mm"],
                                                                                   abs=1e-6)
    assert any("on the DELIVERED lines" in ln for ln in p.summary_lines())
