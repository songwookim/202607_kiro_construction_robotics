"""Both weld directions (user 2026-10-14; scan_core.g1_outputs.DIRECTION_SETS_RULE): every run delivers directions/S_to_E/ and
directions/E_to_S/, each a complete set of every pass in ONE direction (weld_plan.* + kiro/), built by the same plan_weld from the
same measured sections - only the direction differs. The job's direction stays the primary plan, its files exactly as without the
sets. Checked here: the geometry of the two sets (aim points identical, start / goal along the seam reversed, the nozzle drag
relative to each set's own travel with the straight torch body beyond it on the same side - the NECK_RULE roll -, the work-angle
tilt identical in the specimen frame), the files and formats (pass yaml = KIRO's schema), the primary unchanged, a set that cannot
be built kept in verification/ with its reason, and the offline paths (write_outputs + write_kiro_outputs, the kiro_plan CLI)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from g1_synth import synth_estimate, synth_prior
from scan_core import g1_outputs as GO
from scan_core import kiro_plan as KP
from scan_core.g1_outputs import (DIRECTION_SETS, DIRECTIONS_DIR, direction_plan, save_weld_plan, weld_plan_from_estimate,
                                  write_outputs)
from scan_core.weld_line import weld_line
from scan_core.weld_orient import _neck_record, rule_roll
from scan_core.weld_plan import WeldProcess

from test_kiro_plan import CFG, SAMPLES, TORCH, _cfg, _sig

PRIMARY_FILES = ["pass_1.yaml", "pass_1_info.yaml", "pass_2.yaml", "pass_2_info.yaml", "pass_3.yaml", "pass_3_info.yaml", "pass_4.yaml",
                 "pass_4_info.yaml"]


def _outputs(out, direction="S->E"):
    pr = synth_prior()
    est = synth_estimate(pr)
    torch = _cfg().torch
    res = write_outputs(est, pr, WeldProcess(direction=direction), out, k=pr.k, base_frame="link0", torch=torch)
    return pr, est, torch, res


def _kiro(plan, out):
    c = _cfg()
    wt, notes = KP.resolve_world_from_base(c, "link0")
    return KP.write_kiro_outputs(plan, out, c, world_from_base=wt, notes=notes)


@pytest.fixture(scope="module")
def run_SE(tmp_path_factory):
    out = tmp_path_factory.mktemp("bidir_SE")
    pr, est, torch, res = _outputs(out)
    ko = _kiro(res["plan"], out)
    return out, pr, est, torch, res, ko


def test_the_two_sets_weld_the_same_layers_each_built_for_its_own_travel(run_SE):
    out, pr, est, torch, res, ko = run_SE
    plan = res["plan"]
    sets = plan.directions
    assert list(sets) == ["S_to_E", "E_to_S"] and [s.direction for s in sets.values()] == ["S->E", "E->S"]
    assert sets["S_to_E"].same_as_primary and sets["S_to_E"].plan is None and direction_plan(plan, sets["S_to_E"]) is plan
    se, es = plan, direction_plan(plan, sets["E_to_S"])
    assert es is not None and not sets["E_to_S"].same_as_primary and es.process.direction == "E->S"
    assert {p.direction for p in se.passes} == {"S->E"} and {p.direction for p in es.passes} == {"E->S"}
    # the roll by the NECK_RULE for the WPS drag angles: the same roll for both travels (the TCP frame's x follows each travel)
    roll = rule_roll(se.torch, [p.travel_nominal_deg for p in se.passes])[0]
    assert se.torch_roll_deg == es.torch_roll_deg == roll
    for a, b in zip(se.passes, es.passes):
        assert (a.k, a.kind, a.role, a.layer, a.bead) == (b.k, b.kind, b.role, b.layer, b.bead)
        # the aim points (the beads, the layer plan) are the same; only the welding order reverses
        np.testing.assert_allclose(a.aim_base_m, b.aim_base_m[::-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.s_mm, b.s_mm[::-1], rtol=0, atol=1e-12)
        la, lb = weld_line(se, a, se.torch, se.torch_roll_deg), weld_line(es, b, es.torch, es.torch_roll_deg)
        # start / goal along the seam reversed (insets 0): the wire-end line runs the other way between the same ends
        np.testing.assert_allclose(la.tip_start, lb.tip_goal, rtol=0, atol=1e-9)
        np.testing.assert_allclose(la.tip_goal, lb.tip_start, rtol=0, atol=1e-9)
        assert la.s_start_mm == pytest.approx(lb.s_goal_mm, abs=1e-9) and la.s_goal_mm == pytest.approx(lb.s_start_mm, abs=1e-9)
        # the nozzle drags relative to EACH set's own travel; the straight torch body leans more, on the same side (NECK_RULE)
        for wp, ln in ((a, la), (b, lb)):
            nk = _neck_record(se.torch, ln)
            assert nk["nozzle_lean_deg"] == pytest.approx(wp.travel_angle_deg, abs=0.5) and wp.travel_angle_deg < 0.0
            assert nk["body_lean_deg"] < nk["nozzle_lean_deg"] and nk["with_travel"] is True
        # the work-angle tilt is the same in the specimen frame (toward the bevel face); only the travel component flips
        st = se.stations[len(se.stations) // 2]
        za, zb = la.R_tcp[:, 2], lb.R_tcp[:, 2]
        assert float(za @ st.w) == pytest.approx(float(zb @ st.w), abs=1e-9)
        assert float(za @ st.n) == pytest.approx(float(zb @ st.n), abs=1e-9)
        assert float(za @ st.u) == pytest.approx(-float(zb @ st.u), abs=1e-9)


def test_files_and_formats_of_both_sets(run_SE):
    out, pr, est, torch, res, ko = run_SE
    ref = _sig(yaml.safe_load((SAMPLES / "pass_1.yaml").read_text(encoding="utf-8")))
    for nm in ("S_to_E", "E_to_S"):
        d = out / DIRECTIONS_DIR / nm
        assert sorted(os.listdir(d)) == ["kiro", "weld_plan.json", "weld_plan.npz"]
        assert sorted(os.listdir(d / "kiro")) == sorted(["README.md"] + PRIMARY_FILES)
        wpj = json.loads((d / "weld_plan.json").read_text(encoding="utf-8"))
        assert wpj["process"]["direction"] == DIRECTION_SETS[nm] and {p["direction"] for p in wpj["passes"]} == {DIRECTION_SETS[nm]}
        for k in (1, 2, 3, 4):
            text = (d / "kiro" / f"pass_{k}.yaml").read_text(encoding="utf-8")
            y = yaml.safe_load(text)
            assert _sig(y) == ref and y["pass"] == k and KP.format_pass_yaml(y) == text            # KIRO's schema, order, types
            info = yaml.safe_load((d / "kiro" / f"pass_{k}_info.yaml").read_text(encoding="utf-8"))
            assert info["pass_file_location"].startswith(f"<dirname(data_file_path)>/{DIRECTIONS_DIR}/{nm}/kiro/pass_{k}.yaml: the {nm} set")
            assert info["derived"]["direction"] == DIRECTION_SETS[nm]
        readme = (d / "kiro" / "README.md").read_text(encoding="utf-8")
        assert f"This is the `{nm}` set of the run" in readme and f"every pass welds {DIRECTION_SETS[nm]}" in readme
        assert "never swap start and goal" in readme and "`../../../kiro/`" in readme
        rec = ko["directions"][nm]
        assert rec["delivered"] is True and rec["kiro_written"] is True and rec["folder"] == f"{DIRECTIONS_DIR}/{nm}"
        assert sorted(Path(f).name for f in rec["files"]) == sorted(["README.md"] + PRIMARY_FILES)
        assert sorted(Path(f).name for f in rec["weld_plan_files"]) == ["weld_plan.json", "weld_plan.npz"]
        assert [p_["k"] for p_ in rec["passes"]] == [1, 2, 3, 4]
    # the set that is the primary plan: the same weld plan and pass files (its info files say where they are)
    se = out / DIRECTIONS_DIR / "S_to_E"
    assert (se / "weld_plan.json").read_bytes() == (out / "weld_plan.json").read_bytes()
    for k in (1, 2, 3, 4):
        assert (se / "kiro" / f"pass_{k}.yaml").read_bytes() == (out / "kiro" / f"pass_{k}.yaml").read_bytes()
    # the other set's poses: the same aim points as the primary's start / goal swapped, its own orientation
    for p_se, p_es in zip(ko["passes"], ko["directions"]["E_to_S"]["passes"]):
        np.testing.assert_allclose(p_se["tip_start_World_m"], p_es["tip_goal_World_m"], atol=1e-9)
        np.testing.assert_allclose(p_se["tip_goal_World_m"], p_es["tip_start_World_m"], atol=1e-9)
        assert p_es["written_travel_angle_deg"] == pytest.approx(p_se["written_travel_angle_deg"], abs=0.5)
        assert p_es["written_work_angle_deg"] == pytest.approx(p_se["written_work_angle_deg"], abs=0.5)
        assert not np.allclose(np.abs(p_se["quat_xyzw"]), np.abs(p_es["quat_xyzw"]), atol=1e-3)
    readme = (out / "kiro" / "README.md").read_text(encoding="utf-8")
    assert "## Both weld directions" in readme and "`../directions/S_to_E/kiro/`" in readme and "`../directions/E_to_S/kiro/`" in readme
    assert "(weld_plan.direction\n  S->E) = the `S_to_E` set" in readme and "Per pass either set may be used" in readme


def test_primary_outputs_unchanged_by_the_sets(run_SE, tmp_path):
    """The primary weld_plan.* and kiro/ pass files are exactly those of the plan alone (no sets); the primary README only gains the
    section naming the sets."""
    out, pr, est, torch, res, ko = run_SE
    alone = weld_plan_from_estimate(est, pr.spec, WeldProcess(), k=pr.k, base_frame="link0", torch=torch)
    save_weld_plan(alone, tmp_path)
    assert (tmp_path / "weld_plan.json").read_bytes() == (out / "weld_plan.json").read_bytes()
    with np.load(tmp_path / "weld_plan.npz") as a, np.load(out / "weld_plan.npz") as b:
        assert sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    assert alone.directions == {}
    ka = _kiro(alone, tmp_path)
    assert "directions" not in ka and not (tmp_path / DIRECTIONS_DIR).exists()
    for f in PRIMARY_FILES:
        assert (tmp_path / "kiro" / f).read_bytes() == (out / "kiro" / f).read_bytes(), f
    ra, rb = ((p / "kiro" / "README.md").read_text(encoding="utf-8") for p in (tmp_path, out))
    i0 = rb.index("\n## Both weld directions\n")
    i1 = rb.index("\n## Notes") if "\n## Notes" in rb else len(rb)
    assert rb[:i0] + rb[i1:] == ra and "## Both weld directions" not in ra
    assert {k_: v for k_, v in ko.items() if k_ not in ("directions", "files", "dir")} == \
        {k_: v for k_, v in ka.items() if k_ not in ("files", "dir")}


def test_job_direction_E_to_S_and_alternate(tmp_path):
    _pr, _est, _torch, res = _outputs(tmp_path / "es", "E->S")
    sets = res["plan"].directions
    assert sets["E_to_S"].same_as_primary and not sets["S_to_E"].same_as_primary and sets["S_to_E"].plan is not None
    assert (tmp_path / "es" / "weld_plan.json").read_bytes() == (tmp_path / "es" / DIRECTIONS_DIR / "E_to_S" / "weld_plan.json").read_bytes()
    _pr, _est, _torch, res = _outputs(tmp_path / "alt", "alternate")
    sets = res["plan"].directions
    assert not any(s.same_as_primary for s in sets.values()) and all(s.plan is not None and s.delivered for s in sets.values())
    ko = _kiro(res["plan"], tmp_path / "alt")
    readme = (tmp_path / "alt" / "kiro" / "README.md").read_text(encoding="utf-8")
    assert "not one of the sets (the passes do not all weld one way)" in readme
    assert set(ko["directions"]) == {"S_to_E", "E_to_S"} and res["directions"]["primary_setting"] == "alternate"


def test_a_set_that_cannot_be_built_is_kept_in_verification_with_its_reason(tmp_path, monkeypatch):
    real = GO.weld_plan_from_estimate

    def fake(est, spec, process, **kw):
        if process.direction == "E->S":
            raise ValueError("test: no plan this way")
        return real(est, spec, process, **kw)
    monkeypatch.setattr(GO, "weld_plan_from_estimate", fake)
    out = tmp_path / "out"
    # an earlier write of the same folder delivered this set (an offline estimator writes into the same folder): removed
    (out / DIRECTIONS_DIR / "E_to_S" / "kiro").mkdir(parents=True)
    (out / DIRECTIONS_DIR / "E_to_S" / "weld_plan.json").write_text("{}", encoding="utf-8")
    (out / DIRECTIONS_DIR / "E_to_S" / "kiro" / "pass_1.yaml").write_text("x", encoding="utf-8")
    _pr, _est, _torch, res = _outputs(out)
    es = res["plan"].directions["E_to_S"]
    assert es.delivered is False and es.plan is None and "ValueError: test: no plan this way" in es.reason
    assert not (out / DIRECTIONS_DIR / "E_to_S").exists() and sorted(os.listdir(out / DIRECTIONS_DIR)) == ["S_to_E"]
    assert res["directions"]["sets"]["E_to_S"]["folder"] == f"verification/{DIRECTIONS_DIR}/E_to_S"
    ko = _kiro(res["plan"], out)
    assert ko["directions"]["E_to_S"]["delivered"] is False and "test: no plan this way" in ko["directions"]["E_to_S"]["reason"]
    assert ko["directions"]["S_to_E"]["delivered"] is True
    readme = (out / "kiro" / "README.md").read_text(encoding="utf-8")
    assert "`E_to_S` (every pass E->S" in readme and "NOT delivered - weld plan not built (ValueError: test: no plan this way)" in readme
    # a refused plan (S->E contract) of a set is kept in verification/directions/<set>/
    from scan_core.weld_plan import PlanContractError

    def refuse(est, spec, process, **kw):
        p = real(est, spec, process, **kw)
        if process.direction == "E->S":
            e = PlanContractError("test: contract")
            e.plan = p
            raise e
        return p
    monkeypatch.setattr(GO, "weld_plan_from_estimate", refuse)
    out2 = tmp_path / "out2"
    _pr, _est, _torch, res = _outputs(out2)
    es = res["plan"].directions["E_to_S"]
    assert es.delivered is False and "PlanContractError: test: contract" in es.reason
    assert sorted(os.listdir(out2 / "verification" / DIRECTIONS_DIR / "E_to_S")) == ["weld_plan.json", "weld_plan.npz"]
    assert sorted(Path(f).name for f in es.files) == ["weld_plan.json", "weld_plan.npz"]


def test_offline_cli_writes_both_sets(tmp_path, capsys):
    pr = synth_prior()
    est = synth_estimate(pr)
    run = tmp_path / "run"
    (run / "verification" / "estimates").mkdir(parents=True)
    (run / "verification" / "estimates" / "est_3.json").write_text(json.dumps(est.to_dict()), encoding="utf-8")
    (run / "verification" / "run_log.json").write_text(json.dumps({"prior": pr.to_dict(), "job_rough": {
        "base_frame": "link0", "gravity_axis_base": [0.0, 0.0, -1.0]}}), encoding="utf-8")
    (run / "weld_plan.json").write_text(json.dumps({"weld_torch": {"nozzle_end_mm": 1.2, "nozzle_end_source": "test"}}), encoding="utf-8")
    cfg = tmp_path / "platform.yaml"
    cfg.write_text(yaml.safe_dump({"/**": {"ros__parameters": {"kiro_output": CFG, "weld_torch": TORCH}}}), encoding="utf-8")
    out = tmp_path / "regen"
    assert KP.main([str(run), "--platform-config", str(cfg), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    for nm in ("S_to_E", "E_to_S"):
        d = out / DIRECTIONS_DIR / nm
        assert sorted(os.listdir(d / "kiro")) == sorted(["README.md"] + PRIMARY_FILES)
        assert f"wrote {d / 'weld_plan.json'}" in text and f"wrote {d / 'kiro' / 'pass_4.yaml'}" in text
    assert (out / DIRECTIONS_DIR / "S_to_E" / "kiro" / "pass_2.yaml").read_bytes() == (out / "kiro" / "pass_2.yaml").read_bytes()
