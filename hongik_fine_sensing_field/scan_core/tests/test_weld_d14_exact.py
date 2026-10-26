"""D14's speed devices change no result (g1-d14speed, 2026-10-13): the clearance's batched vertex values and its skipped element stages,
the orientation search's frames computed on demand, the platform check's reuse of the search's exact record - each compared with
the plain computation (the same records, bit for bit)."""
import copy
import dataclasses
import math

import numpy as np
import pytest

import scan_core.weld_clearance as WC
import scan_core.weld_orient as WO
from scan_core.weld_plan import WeldProcess, weld_plan_from_features
from scan_core.weld_torch import WeldTorch

from test_weld_orient import _role_tool, guard_body
from test_weld_plan import SPEC, extent, synth_feats
from test_weld_torch import NOZZLE_TEST_MM, TORCH

FIELDS = ("clearance_mm", "station", "s_mm", "solid", "kind", "point_tcp_mm", "below", "intersects", "n_stations")


@pytest.fixture(scope="module")
def plan():
    f = synth_feats()
    return weld_plan_from_features(f, extent(f), SPEC, WeldProcess())


@pytest.fixture(scope="module")
def torch():
    return WeldTorch.from_dict(TORCH).with_nozzle_end(NOZZLE_TEST_MM, "test (the CAD nozzle end)")


def _plain(o):
    """Nested records as plain values (arrays element for element, floats by their bits): equal iff bit-equal."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_plain(v) for v in o.tolist()] if o.ndim else _plain(o.item())
    if isinstance(o, (np.floating, float)):
        return "nan" if np.isnan(o) else float(o).hex()
    if isinstance(o, np.integer):
        return int(o)
    return o


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_rows_g_is_g_row_by_row_bit_for_bit():
    rng = np.random.default_rng(3)
    n, m, k = 7, 40, 5
    Q = rng.normal(scale=20.0, size=(n, m, 3))
    Nrm = rng.normal(size=(n, k, 2))
    Nrm /= np.linalg.norm(Nrm, axis=-1, keepdims=True)
    b = rng.normal(scale=5.0, size=(n, k))
    ylo, yhi = rng.uniform(-200, -100, n), rng.uniform(100, 200, n)
    G = WC._g_rows(Q, Nrm, b, ylo, yhi)
    for i in range(n):
        assert np.array_equal(G[i], WC._g(Q[i], Nrm[i], b[i], float(ylo[i]), float(yhi[i])))


def _lines(p, wt, rng, n_or=4):
    """Delivered lines of every pass at orientations around the nominal, shifted toward / into the specimen and sideways."""
    for wp in p.passes:
        t0, w0 = WO._nominal(wp)
        lr = WO.line_rows(p, wp, 0.002)
        for _ in range(n_or):
            tt, ww = t0 + float(rng.integers(-15, 16)), w0 + float(rng.integers(-15, 16))
            roll = float(rng.choice([0.0, 180.0]))
            ln = WO.weld_line(p, wp, wt, roll, R_tip=WO.rotation_mean(WO.station_rotations(p, wp, tt, ww)), ends=lr.ends)
            P0 = ln.samples(lr.step_m)["tcp"]
            n_pl, w_pl = np.asarray(p.stations[0].n, float), np.asarray(p.stations[0].w, float)
            for sh in (np.zeros(3), -n_pl * float(rng.uniform(0.5, 8.0)) * 1e-3,
                       w_pl * float(rng.uniform(-4, 4)) * 1e-3 - n_pl * float(rng.uniform(0, 3)) * 1e-3):
                for _name, off in ln.offsets():
                    yield lr, ln.R_tcp, P0 + np.asarray(off, float) + sh


def test_skipped_element_stages_give_the_same_records(plan, torch, monkeypatch):
    """clearance() skips the separating-axis and contact-event stages of a (row, solid) whose triangles and edges cannot come below
    what can still matter; with nothing skipped (_elements_lb always -inf) every record - value, place, kind, intersection - is the
    same, in both modes (value; the feasibility test at 0, at a requirement, rows permuted), for every tool role."""
    rng = np.random.default_rng(7)
    p = copy.deepcopy(plan)
    wt = torch.with_stick_out(15.0, "test")
    roles = WC.tool_roles(_role_tool(mirror=True))
    bodies = [b for _r, b in roles.roles()] + [guard_body()]
    cases = []
    for lr, R, P in _lines(p, wt, rng, n_or=2):
        R_sec, P_sec = WC.rows_poses(lr.rows, R, P)
        for b in bodies:
            for thr, order in ((None, None), (0.0, None), (float(rng.uniform(0, 12)), rng.permutation(len(P)))):
                cases.append((b, lr.rows, R_sec, P_sec, thr, order))
    got = [WC.clearance(b, rows, R_sec, P_sec, threshold=thr, order=order) for b, rows, R_sec, P_sec, thr, order in cases]
    monkeypatch.setattr(WC, "_elements_lb", lambda F, g, idx, L, above: np.full(g.shape[0], -np.inf))
    plain = [WC.clearance(b, rows, R_sec, P_sec, threshold=thr, order=order) for b, rows, R_sec, P_sec, thr, order in cases]
    kinds = set()
    for a, c in zip(got, plain):
        assert all(_same(a.get(f), c.get(f)) for f in FIELDS), {f: (a.get(f), c.get(f)) for f in FIELDS}
        kinds.add((a["kind"], a["below"], a["intersects"]))
    assert {"vertex", "triangle"} <= {k for k, _b, _i in kinds} and any(i for _k, _b, i in kinds)   # contacts and intersections seen


def test_frames_on_demand_skip_the_same_candidates(plan, torch, monkeypatch):
    """search_pass computes a candidate's torch frame only when a rejected candidate's bound may reach it (the torch-axis prefilter);
    with every frame computed (prefilter off) the candidates, their order, the rejections and the records are the same."""
    p = copy.deepcopy(plan)
    body = guard_body()
    wt = torch.with_stick_out(20.0, "test")
    wp = WO.retarget_pass(p, dataclasses.replace(p.passes[0], travel_nominal_deg=14.0, work_nominal_deg=22.0), 14.0, 22.0)
    p.passes[0] = wp
    lr = WO.line_rows(p, wp, 0.002)

    def run():
        out = []
        for i, c in enumerate(WO.search_pass(p, wp, body, wt, 0.0, 1.0, rows=lr)):
            c = dict(c)
            c.pop("pass", None)
            out.append(c)
            if c.get("exhausted") or i >= 2:
                return out
        return out
    a = run()
    monkeypatch.setattr(WO, "_AXIS_PREFILTER_RAD", math.inf)
    b = run()
    assert len(a) == len(b) and a[0]["deviation_deg"] > 0.0 and a[0]["rejected"]["n"] > a[0]["n_evaluated"]   # bound-skipped ones
    for x, y in zip(a, b):
        assert _plain(x) == _plain(y)


def test_the_check_reuses_the_searchs_record_only_for_the_same_evaluation(plan, torch, monkeypatch):
    """The platform check's exact per-role values of a pass's delivered line (pass_clearance) are the orientation search's record of
    that candidate when every input is the same (eval_key), not a re-evaluation; any other record (another requirement, another line)
    is ignored and the line evaluated."""
    p = copy.deepcopy(plan)
    p.passes = p.passes[2:3]
    wt = torch.with_stick_out(15.0, "test")
    roles = WC.tool_roles(_role_tool(mirror=True))
    wp = p.passes[0]
    first = next(WO.search_pass(p, wp, roles, wt, 180.0, 3.0))
    trial = first["pass"]
    exact = WO.pass_clearance(p, trial, roles, wt, 180.0, required_mm=3.0)                 # evaluated
    assert exact["eval_key"] == first["clearance"]["eval_key"]
    for k_ in ("clearance_mm", "excess_mm", "limiting_role", "by_role", "line", "sample", "kind", "point_tcp_mm", "checked_line"):
        assert _plain(exact[k_]) == _plain(first["clearance"][k_])
    calls = []
    real = WO._eval_line
    monkeypatch.setattr(WO, "_eval_line", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    reused = WO.pass_clearance(p, trial, roles, wt, 180.0, required_mm=3.0, known=first["clearance"])
    assert not calls and _plain(reused) == _plain(exact)
    other = WO.pass_clearance(p, trial, roles, wt, 180.0, required_mm=4.0, known=first["clearance"])   # another requirement
    assert calls and other["by_role"]["other"]["required_mm"] == 4.0 and other["eval_key"] != first["clearance"]["eval_key"]
    calls.clear()
    WO.pass_clearance(p, trial, roles, wt, 0.0, required_mm=3.0, known=first["clearance"])           # another line (the roll)
    assert calls


def test_choose_orientation_gives_the_check_the_searchs_records(plan, torch):
    """choose_orientation hands each candidate's exact record to the platform check (the trial plan's d14_exact) and the pass the
    search built (search_pass 'pass': retarget_pass of the candidate)."""
    p = copy.deepcopy(plan)
    p.passes = p.passes[:2]
    wt = torch.with_stick_out(15.0, "test")
    roles = WC.tool_roles(_role_tool(mirror=True))
    seen = []

    def check(trial, only=None):                                  # as the platforms' check: the plan's torch, its step
        from scan_core.weld_torch import plan_weld_torch
        out = []
        for wp in trial.passes:
            if only is not None and wp.k not in only:
                continue
            rec = trial.d14_exact[wp.k]
            again = WO.pass_clearance(trial, wp, roles, plan_weld_torch(trial, wt), trial.torch_roll_deg, required_mm=1.0, known=rec)
            seen.append((wp.k, rec["eval_key"], again["eval_key"]))
            out.append({"k": wp.k, "reachable": True, "reason": "", "min_limit_margin_rad": 0.5})
        return out
    WO.choose_orientation(p, check, body=roles, required=1.0, torch=wt, step_m=0.002)
    assert seen and all(a == b for _k, a, b in seen)
