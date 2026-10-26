"""kiro_io and both weld directions (user 2026-10-14): the npz meta lists the direction sets (weld_direction.direction_sets: every set's
folder and weld_plan files, delivered or why not; kiro_outputs.directions: each set's kiro/ files and poses); result.paths, the
kiro_* arrays and the p{k}_* arrays stay the primary set's (the job's direction) - exactly those of the plan without sets."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STACK = os.path.dirname(PKG)
for p in (PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"), os.path.join(STACK, "scan_core", "tests"),
          os.path.join(STACK, "vc3dz_driver")):
    if p not in sys.path:
        sys.path.insert(0, p)

from g1_synth import synth_estimate, synth_prior  # noqa: E402
from hongik_scan_ros import kiro_io as K  # noqa: E402
from scan_core import kiro_plan as KP  # noqa: E402
from scan_core.g1_outputs import write_outputs  # noqa: E402
from scan_core.weld_plan import WeldProcess  # noqa: E402
from test_kiro_plan import _cfg  # noqa: E402


def test_meta_lists_both_sets_and_the_arrays_stay_the_primary(tmp_path):
    pr = synth_prior()
    est = synth_estimate(pr)
    res = write_outputs(est, pr, WeldProcess(), tmp_path, k=pr.k, base_frame="link0", torch=_cfg().torch)
    plan = res["plan"]
    plan.directions["E_to_S"].delivered = False                          # as after a failed D14 of that set on the robot
    plan.directions["E_to_S"].reason = "weld plan not verified (D14): pass 2: not reachable (test)"
    c = _cfg()
    wt, notes = KP.resolve_world_from_base(c, "link0")
    ko = {"written": True, **KP.write_kiro_outputs(plan, tmp_path, c, world_from_base=wt, notes=notes)}
    wd = K.weld_direction_of(plan)
    assert wd["setting"] == "S->E" and set(wd["passes"].values()) == {"S->E"} and "BOTH directions" in wd["direction_sets_rule"]
    ds = wd["direction_sets"]
    assert ds["S_to_E"]["delivered"] and ds["S_to_E"]["same_as_primary"] and ds["S_to_E"]["folder"] == "directions/S_to_E"
    assert ds["E_to_S"]["delivered"] is False and "pass 2: not reachable (test)" in ds["E_to_S"]["reason"]
    m = K.kiro_meta(ko)
    sets = m["directions"]["sets"]
    assert "primary set only" in m["directions"]["in_this_npz"]
    assert sets["S_to_E"]["kiro_written"] and len(sets["S_to_E"]["files"]) == 9 and [p_["k"] for p_ in sets["S_to_E"]["passes"]] == [1, 2, 3, 4]
    assert sets["E_to_S"]["delivered"] is False and "not reachable (test)" in sets["E_to_S"]["reason"]
    assert m["files"] == ko["files"] and all(os.path.dirname(f) == str(tmp_path / "kiro") for f in m["files"])
    # the arrays: the primary set's only (the same keys and text as the plan alone)
    arr = K.kiro_npz_arrays(ko)
    assert str(arr["kiro_pass_1_yaml"]) == (tmp_path / "kiro" / "pass_1.yaml").read_text(encoding="utf-8")
    paths = K.plan_paths(plan)
    meta = K.npz_meta(outcome=K.run_outcome("PASS", "done"), status="PASS", reason="done", welding_position=1, position="1G",
                      base_frame="link0", k=3.0, out_dir=str(tmp_path), run_id="x", n_paths=len(paths), weld_direction=wd,
                      kiro_outputs=m)
    d = K.load_precision_npz(K.write_precision_npz(tmp_path, K.precision_npz_arrays(meta, plan=plan, paths=paths, kiro=arr)))
    assert d["meta"]["weld_direction"]["direction_sets"]["E_to_S"]["delivered"] is False
    assert d["meta"]["kiro_outputs"]["directions"]["sets"]["S_to_E"]["folder"] == "directions/S_to_E"
    for pa in paths:                                                     # result.paths = the primary's lines (S->E)
        wp = {w.k: w for w in plan.passes}[pa.k]
        assert wp.direction == "S->E" and np.array_equal(d[f"p{pa.k}_wp_pos_m"], pa.pos_m)
