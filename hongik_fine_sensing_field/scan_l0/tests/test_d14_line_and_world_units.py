"""General units of the lab line's 2026-10-13 changes that the KIRO package keeps (kiro-1g-next; ported from the lab-only weld path
dry run's unit tests, without that tool): D14 on the DELIVERED line on the lab run l5_k30_1_a at stick-out 15 (weldpose's pass 2
case), the arm-inside-world start refusal only before the first motion (Teach1G._apply_world) and the follow record that names the
segment and the distance where the IK stops (scan_core.trajectory.follow_candidate). No arm, no ROS."""
from __future__ import annotations

import dataclasses
import gzip
import io
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(STACK, "scan_core", "tests"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from scan_core import kiro_plan as KP  # noqa: E402
from scan_core import weld_orient as WO  # noqa: E402
from scan_core.weld_clearance import pass_geometry, tool_body_from_urdf, tool_roles  # noqa: E402
from scan_core.weld_line import segment_points  # noqa: E402
from scan_core.weld_plan import WeldProcess  # noqa: E402
from scan_core.weld_torch import WeldTorch  # noqa: E402

URDF = os.path.join(HERE, "data", "rb20_1900es_u_eoat.urdf")
LAB_CONFIG = os.path.join(HERE, "data", "hongik_lab_rb20_values.yaml")      # the lab platform's values (test data on kiro-1g)
REAL_L5 = os.path.join(HERE, "data", "real_l5_k30_1_a_final_estimate_2026-10-13.json.gz")
NOZZLE_MM = 1.2
WORLD = {"enabled": True, "world_frame": "World", "world_from_base": {"xyz_m": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                                                                     "source": "lab definition (test)", "status": "lab definition"}}


def _torch() -> WeldTorch:
    """The lab platform's weld torch (the torch turned in its clamp, 2026-10-13) with the CAD nozzle end."""
    from scan_core.weld_torch import weld_torch_config
    p = yaml.safe_load(open(LAB_CONFIG, encoding="utf-8"))["/**"]["ros__parameters"]
    return weld_torch_config(p).with_nozzle_end(NOZZLE_MM, "test (the CAD nozzle end)")


@pytest.fixture(scope="module")
def tool():
    body = tool_body_from_urdf(open(URDF, encoding="utf-8").read(), "torch_tcp")
    return body, tool_roles(body)


def _real_plan(stick_mm: float):
    """The weld plan of the lab run l5_k30_1_a (its final estimate and prior, 2026-10-13) re-made at a stick-out [mm from the nozzle
    end] with the lab torch (turned in its clamp, the CAD nozzle end): the plan's own lateral aims (KIRO's practice, convex beads)."""
    from scan_core.g1_outputs import weld_plan_from_estimate
    from scan_core.g1_types import SeamEstimate, SeamPrior
    from scan_core.weld_plan import weld_design_from_doc
    d = json.load(gzip.open(REAL_L5, "rt", encoding="utf-8"))
    est, prior = SeamEstimate.from_dict(d["estimate"]), SeamPrior.from_dict(d["prior"])
    pr = dataclasses.replace(WeldProcess(), stick_out_mm=float(stick_mm))
    return weld_plan_from_estimate(est, prior.spec, pr, base_frame=d["base_frame"], gravity_axis_base=np.asarray(d["gravity_axis_base"]),
                                   k=float(prior.k), design=weld_design_from_doc({}, pr), torch=_torch())


def _layer_plan_clearance(plan, wp, body, wt, roll):
    """The superseded per-station check: each station's torch frame at that station's aim, in its own section (the layer plan's poses)."""
    from scan_core.weld_clearance import clearance
    from scan_core.weld_torch import roll_matrix
    geo = pass_geometry(plan, wp)
    s_idx = {float(v): i for i, v in enumerate(geo.s_mm)}
    order = np.array([s_idx[float(v)] for v in wp.s_mm])
    R_sec = geo.A[order[0]] @ (np.asarray(wp.R_base_tcp, float)[0] @ roll_matrix(roll) @ wt.tcp_from_tip.R.T)
    aim = np.zeros((len(geo.s_mm), 3))
    aim[order, 0], aim[order, 2] = np.asarray(wp.aim_xz_mm)[:, 0], np.asarray(wp.aim_xz_mm)[:, 1]
    return float(clearance(body, geo, R_sec, aim - (R_sec @ wt.tcp_from_tip.t) * 1e3)["clearance_mm"])


def test_lab_run_stick_out_15_pass_2_is_no_longer_delivered_into_the_plate(tmp_path, tool):
    """weldpose's finding on the lab run l5_k30_1_a at stick-out 15 (the torch turned, roll 180): D14 on the layer plan's per-station
    poses accepted pass 2 at travel -36 / work +17 (nozzle +0.08 mm) while its delivered straight line put the nozzle 0.33 mm into the
    right plate. D14 now searches the delivered line: the chosen orientation clears on the line, the KIRO yaml line is exactly the
    checked line and its per-pose re-check (weld_orient.poses_role_clearance on the line's samples, against the specimen before the
    pass) gives D14's value, and the record says why the pass left the WPS nominal (the gas nozzle / contact tip against the plate)."""
    _body, roles = tool
    plan = _real_plan(15.0)
    wt = plan.torch
    wp = plan.passes[1]
    # weldpose's choice was -36 / +17 under the 09-22 aims; with KIRO's practice and the convex beads (2026-10-13 night) the same case
    # is found near the WPS nominal: an orientation the per-station check accepts while its delivered line is in the plate (the one
    # deepest in the plate on the 2-deg grid around the nominal)
    t0, w0 = WO._nominal(wp)
    cases = []
    for dt in range(-8, 9, 2):
        for dw in range(-10, 11, 2):
            cand = WO.retarget_pass(plan, wp, t0 + dt, w0 + dw)
            if _layer_plan_clearance(plan, cand, roles.process, wt, 180.0) > 0.0:
                ln = WO.pass_clearance(plan, cand, roles, wt, 180.0, required_mm=6.0, step_m=0.002)["by_role"]["process"]["clearance_mm"]
                cases.append((ln, t0 + dt, w0 + dw))
    line_c, t_old, w_old = min(cases)
    old = WO.retarget_pass(plan, wp, t_old, w_old)
    assert _layer_plan_clearance(plan, old, roles.process, wt, 180.0) > 0.0     # the superseded check accepted it ...
    line_old = WO.pass_clearance(plan, old, roles, wt, 180.0, required_mm=6.0, step_m=0.002)
    assert line_old["by_role"]["process"]["clearance_mm"] < 0.0                  # ... its delivered line is in the plate
    p2 = dataclasses.replace(plan, passes=[wp])
    rec = WO.choose_orientation(p2, None, body=roles, required=6.0, torch=wt, step_m=0.002, rolls=(180.0,))
    o = rec["passes"][0]
    assert o["chosen"] is not None and o["deviation_deg"] > 0.0 and o["clearance_by_role"]["process"]["clearance_mm"] >= 0.0
    nom = o["deviation_why"]["nominal"]
    assert nom["constraint"] == "clearance" and nom["part"] == "process" and nom["exact"] and nom["clearance_mm"] < 0.0
    assert nom["link"] == "eoat_base_link" and "come within" in o["why"]
    cfg = KP.KiroOutputConfig.from_dict(WORLD, wt)
    wfb, notes = KP.resolve_world_from_base(cfg, "link0")
    KP.write_kiro_outputs(p2, str(tmp_path), cfg, world_from_base=wfb, notes=notes)
    y = KP.read_pass_yaml(os.path.join(str(tmp_path), "kiro", f"pass_{wp.k}.yaml"))
    assert np.array_equal(y["start"][0], o["checked_line"]["tcp_start_m"]) and np.array_equal(y["goal"][0], o["checked_line"]["tcp_goal_m"])
    from scan_core.rigid import _quat_to_R
    R = _quat_to_R(y["start"][1], "q")
    P = segment_points(y["start"][0], y["goal"][0], 0.002)
    chk = WO.poses_role_clearance(roles, pass_geometry(p2, p2.passes[0]), R, P, wt.tcp_from_tip.t, 6.0)
    assert chk["by_role"]["process"]["clearance_mm"] == pytest.approx(o["clearance_by_role"]["process"]["clearance_mm"], abs=1e-6)
    assert chk["by_role"]["process"]["clearance_mm"] >= 0.0


def test_the_arm_inside_the_world_refuses_only_before_the_first_motion():
    """Teach1G._apply_world (the 1G core, the post-weld scan): the arm inside the run's world is a start refusal only before the first
    motion; after it (e.g. stopped at a weld pose) it is recorded, not refused."""
    from scan_l0.autoteach import _Abort
    from scan_l0.teach_1g import Teach1G
    t = Teach1G.__new__(Teach1G)
    t.world = SimpleNamespace(fixed_bodies_known=True, version=2, moveit_objects=lambda: [{"id": "hongik/specimen"}],
                              to_dict=lambda: {"version": 2}, summary=lambda: "world v2",
                              explain=lambda c: "eoat_base_link is inside hongik/specimen")
    t.sess = SimpleNamespace(robot=SimpleNamespace(set_scene_boxes=lambda objs: True, remove_scene_boxes=lambda ids: None))
    t.cell_ids, t.world_history = [], []
    t._world_to_envelope = lambda: None
    t._event = lambda *a, **k: None
    decs = []
    t._decide = lambda step, msg, **kw: decs.append((step, msg))
    t._arm_contacts = lambda: ([{"link": "eoat_base_link", "object": "hongik/specimen", "kind": "world"}], "")
    t._motion_sent = False
    with pytest.raises(_Abort) as e:
        t._apply_world("prior: before any motion")
    assert e.value.status == "REFUSED" and "the arm at its current pose is inside the cell world v2" in e.value.reason \
        and "(nothing sent)" in e.value.reason
    t._motion_sent = True
    t._apply_world("back to the run's world after pass 1")
    assert decs[-1][0] == "world" and "recorded, not refused" in decs[-1][1] and "eoat_base_link is inside hongik/specimen" in decs[-1][1]


def test_the_follow_names_the_segment_and_the_distance_where_the_ik_stops():
    """trajectory.follow_candidate on the lab chain: a retreat far beyond the arm's reach - the record names the segment (i_fail 2 =
    the retreat after the descent and the line) and the distance along it of the first pose the IK did not reach."""
    from scan_core.fk_urdf import chain_from_joints, parse_urdf_joints
    from scan_core.trajectory import fast_fk, follow_candidate
    chain = chain_from_joints(parse_urdf_joints(io.StringIO(open(URDF, encoding="utf-8").read())), "link0", "tcp")
    fk = fast_fk(chain)
    q0 = np.array([2.754425756423491, 0.0469306991049455, -2.5981306910011384, -0.9163067123076475, 0.4618459288737595,
                   5.142267921429792])
    lo = np.array([j.lower for j in chain.movable_joints])
    hi = np.array([j.upper for j in chain.movable_joints])
    T0 = fk(q0)
    T_in, T_out, T_far = T0.copy(), T0.copy(), T0.copy()
    T_in[2, 3] -= 0.04
    T_out[:3, 3] = T_in[:3, 3] + np.array([0.08, 0.01, 0.0])
    u = T_out[:3, 3] / np.linalg.norm(T_out[:3, 3])
    T_far[:3, 3] = T_out[:3, 3] + 2.0 * u                                 # 2 m away from the base: out of reach
    rec = follow_candidate(chain, fk, q0, [T_in, T_out, T_far], step_m=0.002, lower=lo, upper=hi, max_joint_step_rad=0.05,
                           sigma_radius_m=0.4)
    assert not rec["ok"] and rec["i_fail"] == 2 and rec["fail_segment_m"] == pytest.approx(2.0)
    assert 0.1 < rec["fail_done_m"] < 1.5, rec
    ok = follow_candidate(chain, fk, q0, [T_in, T_out], step_m=0.002, lower=lo, upper=hi, max_joint_step_rad=0.05, sigma_radius_m=0.4)
    assert ok["ok"] and ok["i_fail"] == -1 and ok["fail_done_m"] is None
