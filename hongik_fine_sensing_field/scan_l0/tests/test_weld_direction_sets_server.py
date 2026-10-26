"""Both weld directions on the server path (user 2026-10-14; Teach1G._weld_direction_sets / _direction_sets_withdrawn): D14 runs
once per direction set that is not the primary plan (the set that IS the primary plan is saved from the same object, no second
D14); a set that fails D14 goes to verification/directions/<set>/ with its reason in the run's notes and decisions, the run's status
stays the primary's; when the primary is not delivered no set is. The D14 itself (Teach1G._weld_plan_d14 - the same code as the
primary's) is replaced here by a recorder; the full robot check is the virtual-arm suites'."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scan_core" / "tests"))
from g1_synth import synth_estimate, synth_prior  # noqa: E402

from scan_core.g1_outputs import DIRECTIONS_DIR, retract_weld_plan, write_outputs  # noqa: E402
from scan_core.weld_plan import WeldProcess  # noqa: E402
from scan_l0.teach_1g import Teach1G  # noqa: E402


def _fake_run(out, fail_dirs=()):
    pr = synth_prior()
    est = synth_estimate(pr)
    res = write_outputs(est, pr, WeldProcess(), out, k=pr.k, base_frame="link0")
    calls, decisions = [], []

    def d14(est_, plan_, d14_, note_, label=""):
        calls.append((plan_, label))
        bad = [(wp.k, {"reachable": False, "reason": "test: IK"}) for wp in plan_.passes][:1] \
            if {wp.direction for wp in plan_.passes} & set(fail_dirs) else []
        return [(wp.k, {"reachable": True}) for wp in plan_.passes], bad
    me = SimpleNamespace(res=SimpleNamespace(status="PASS", notes=[], files=list(res["files"])), out=Path(out),
                         rs=SimpleNamespace(adapter="test arm"), _weld_plan_d14=d14,
                         _decide=lambda step, msg, **kw: decisions.append((step, msg, kw)))
    return est, res["plan"], me, calls, decisions


def test_d14_once_per_other_set_and_a_failing_set_goes_to_verification(tmp_path):
    out = tmp_path / "out"
    est, plan, me, calls, decisions = _fake_run(out, fail_dirs=("E->S",))
    Teach1G._weld_direction_sets(me, est, plan, True, "")
    es, se = plan.directions["E_to_S"], plan.directions["S_to_E"]
    assert [lb for _p, lb in calls] == ["weld plan direction set E_to_S (every pass E->S)"] and calls[0][0] is es.plan
    assert se.delivered and se.same_as_primary and sorted(os.listdir(out / DIRECTIONS_DIR)) == ["S_to_E"]
    assert (out / DIRECTIONS_DIR / "S_to_E" / "weld_plan.json").read_bytes() == (out / "weld_plan.json").read_bytes()
    assert es.delivered is False and es.reason == "weld plan not verified (D14): pass 1: not reachable (test: IK)"
    assert sorted(os.listdir(out / "verification" / DIRECTIONS_DIR / "E_to_S")) == ["weld_plan.json", "weld_plan.npz"]
    assert me.res.status == "PASS"                                            # the run's status is the primary's
    assert any(n.startswith("weld plan direction set E_to_S (every pass E->S) not delivered: weld plan not verified (D14)")
               for n in me.res.notes)
    assert any("NOT delivered" in m and "verification/directions/E_to_S/" in m for _s, m, _k in decisions)
    assert all(f in me.res.files for f in es.files + se.files)
    assert all("/verification/directions/E_to_S/" in f for f in es.files)


def test_every_set_delivered_when_d14_passes(tmp_path):
    out = tmp_path / "out"
    est, plan, me, calls, decisions = _fake_run(out)
    Teach1G._weld_direction_sets(me, est, plan, True, "")
    assert len(calls) == 1 and all(ds.delivered for ds in plan.directions.values()) and me.res.notes == []
    assert sorted(os.listdir(out / DIRECTIONS_DIR)) == ["E_to_S", "S_to_E"] and not (out / "verification" / DIRECTIONS_DIR).exists()


def test_no_set_without_the_primary(tmp_path):
    out = tmp_path / "out"
    est, plan, me, calls, decisions = _fake_run(out)
    moved = retract_weld_plan(out)                                          # the primary failed D14 (teach_1g._finish)
    Teach1G._direction_sets_withdrawn(me, plan, moved, "the primary plan is not delivered (test)")
    assert calls == [] and not (out / DIRECTIONS_DIR).exists()
    assert all(not ds.delivered and ds.reason == "the primary plan is not delivered (test)" for ds in plan.directions.values())
    assert sorted(os.listdir(out / "verification" / DIRECTIONS_DIR)) == ["E_to_S", "S_to_E"]
    assert all(ds.files and all("/verification/directions/" in f for f in ds.files) for ds in plan.directions.values())
    assert me.res.status == "PASS" and any("not delivered: the primary plan is not delivered (test)" in n for n in me.res.notes)


@pytest.mark.parametrize("missing", [True])
def test_plans_without_sets_are_untouched(tmp_path, missing):
    me = SimpleNamespace(res=SimpleNamespace(status="PASS", notes=[], files=[]), out=tmp_path)
    plan = SimpleNamespace(passes=[])                                       # a plan without direction sets (a stand-in model's)
    Teach1G._weld_direction_sets(me, None, plan, True, "")
    Teach1G._direction_sets_withdrawn(me, plan, [], "x")
    assert me.res.notes == [] and os.listdir(tmp_path) == []
