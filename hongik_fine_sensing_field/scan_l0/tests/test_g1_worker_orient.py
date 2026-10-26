"""The compute worker's torch-orientation search (g1_worker orient_open / orient_ask / orient_take / orient_close, g1-d14speed
2026-10-13): each (pass, roll) search in its own forked child - the passes in parallel - gives exactly the in-process search_pass
sequence; a child that dies is replaced by the same generator replayed in the worker (the same sequence); orient_next is ask + take."""
import copy
import os
import signal
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scan_core", "tests"))

import scan_core.weld_orient as WO                                              # noqa: E402
from scan_core.weld_clearance import tool_roles                                 # noqa: E402
from scan_core.weld_plan import WeldProcess, weld_plan_from_features            # noqa: E402
from scan_core.weld_torch import WeldTorch                                      # noqa: E402
from scan_l0 import g1_worker as G                                              # noqa: E402

from test_weld_orient import _role_tool                                         # noqa: E402
from test_weld_plan import SPEC, extent, synth_feats                            # noqa: E402
from test_weld_torch import NOZZLE_TEST_MM, TORCH                               # noqa: E402

N_TAKE = 3


@pytest.fixture(scope="module")
def setup():
    f = synth_feats()
    p = weld_plan_from_features(f, extent(f), SPEC, WeldProcess())
    wt = WeldTorch.from_dict(TORCH).with_nozzle_end(NOZZLE_TEST_MM, "test").with_stick_out(15.0, "test")
    p.torch = wt
    return p, tool_roles(_role_tool(mirror=True)), wt


def _plain(o):
    """A candidate dict as plain nested lists / floats (arrays exactly, element for element; the WeldPass by its rotations and dense
    path) - equal iff bit-equal."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_plain(v) for v in o.tolist()] if o.ndim else _plain(o.item())
    if isinstance(o, (np.floating, float)):
        return ("nan" if np.isnan(o) else float(o).hex())
    if isinstance(o, np.integer):
        return int(o)
    if hasattr(o, "R_base_tcp") and hasattr(o, "dense_pos_m"):
        return {"R": _plain(np.asarray(o.R_base_tcp)), "dense": _plain(np.asarray(o.dense_pos_m)), "k": o.k}
    return o


def _key(c):
    return _plain(c)


def _reference(p, body, wt, k, roll, req):
    wp = {w.k: w for w in p.passes}[k]
    out = []
    for i, c in enumerate(WO.search_pass(p, wp, body, wt, roll, req, rows=WO.line_rows(p, wp, 0.002))):
        out.append(_key(c))
        if c.get("exhausted") or i + 1 >= N_TAKE:
            break
    return out


def _run(p, body, wt, req, ks, kill=None):
    """The worker's searches of passes ks at roll 180: the first candidates asked together, then orient_next; kill = a pass whose
    child is killed after its first answer → ({k: [candidates]}, {k: child pid})."""
    G.run_job("orient_open", {"plan": copy.deepcopy(p), "body": body, "torch": wt, "required_mm": req, "reach": None, "step_m": 0.002})
    try:
        ids = {k: G.run_job("orient_ask", {"k": k, "roll": 180.0, "max_deviation_deg": None})["id"] for k in ks}   # all at once
        pids = {k: G._ORIENT["searches"][(k, 180.0)].pid for k in ks}
        assert all(pids.values()) and len(set(pids.values())) == len(ks)                 # one child per (pass, roll)
        got = {k: [_key(G.run_job("orient_take", {"k": k, "roll": 180.0, "id": ids[k]}))] for k in ks}
        if kill is not None:
            os.kill(pids[kill], signal.SIGKILL)                                          # replayed in the worker from here on
        for _ in range(N_TAKE - 1):
            for k in ks:
                if "'exhausted': True" not in repr(got[k][-1]):
                    got[k].append(_key(G.run_job("orient_next", {"k": k, "roll": 180.0, "max_deviation_deg": None})))
        alive = {k: G._ORIENT["searches"][(k, 180.0)].pid for k in ks}
        if kill is not None:                                                             # an unwanted candidate still computing
            other = next(k for k in ks if k != kill)
            assert G.run_job("orient_ask", {"k": other, "roll": 180.0, "max_deviation_deg": None})["id"] == len(got[other])
    finally:
        closed = G.run_job("orient_close", {})
    assert closed["closed"] == len(ks) and not G._ORIENT["searches"]
    for pid in pids.values():                                                            # every child reaped
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
    return got, pids, alive


def test_parallel_children_give_the_in_process_sequence(setup):
    """The role tool (gas nozzle, contact tip, neck ring, guard): searches with rejections, the Lipschitz skipping, exhaustion."""
    p, roles, wt = setup
    ks = [wp.k for wp in p.passes[:3]]
    got, _pids, _alive = _run(p, roles, wt, 3.0, ks)
    for k in ks:
        assert got[k] == _reference(p, roles, wt, k, 180.0, 3.0), k


def test_a_dead_child_is_replayed_in_the_worker(setup):
    """A child killed between two candidates: its search continues in the worker, the generator replayed past the answers it gave -
    the same sequence; the other child is untouched."""
    from test_weld_orient import far_body
    p, _roles, wt = setup
    body = far_body()                                                                    # every candidate clears: long sequences
    ks = [wp.k for wp in p.passes[1:3]]
    got, pids, alive = _run(p, body, wt, 2.0, ks, kill=ks[0])
    assert alive[ks[0]] is None and alive[ks[1]] == pids[ks[1]]
    for k in ks:
        assert len(got[k]) == N_TAKE and got[k] == _reference(p, body, wt, k, 180.0, 2.0), k


def test_a_take_of_a_request_never_asked_is_an_error(setup):
    p, roles, wt = setup
    G.run_job("orient_open", {"plan": copy.deepcopy(p), "body": roles, "torch": wt, "required_mm": 3.0, "reach": None, "step_m": 0.002})
    try:
        with pytest.raises(ValueError, match="not outstanding"):
            G.run_job("orient_take", {"k": p.passes[0].k, "roll": 180.0, "id": 5})
    finally:
        G.run_job("orient_close", {})
