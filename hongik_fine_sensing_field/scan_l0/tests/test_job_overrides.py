"""Run-time job overrides (scan_l0.job_overrides; user 2026-10-13: "parameters must be adjustable per run - site conditions differ")
and the measurement speed inside the noise priors' calibrated range (scan_core.noise_priors.check_measure_speed)."""
from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scan_core.g1_types import Precision
from scan_core.noise_priors import calibrated_speed_range, check_measure_speed
from scan_core.weld_plan import WeldProcess, weld_design_from_doc
from scan_l0 import config as C
from scan_l0 import job_overrides as J
from scan_l0.autoteach import weld_process_doc

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.dirname(os.path.dirname(HERE))
CALIB = os.path.join(STACK, "eoat_description", "calib", "eoat_lab_01.yaml")
KIRO_CALIB = os.path.join(STACK, "eoat_description", "calib", "eoat_kiro_01.yaml")
LAB = os.path.join(HERE, "data", "hongik_lab_rb20_values.yaml")          # the lab platform's values (test data on kiro-1g)
TEMPLATE = str(C.JOBS_DIR / "groove_10T_1G.yaml")


def _template():
    return C.load_job(TEMPLATE)


def _apply(doc, items, source="goal"):
    return J.apply_overrides(doc, J.parse_set_items(items), source=source, position="1G")


def test_valid_paths_reach_the_parsers_and_the_job_file_is_not_changed():
    job, doc = _template()
    doc0 = copy.deepcopy(doc)
    p0, L0 = doc["precision"]["end_position_mm"], float(doc["specimen"]["weld_length_mm"])      # the template's values
    p1, L1 = 2.0 * p0, 1.5 * L0                                                                    # other values than the template's
    items = [f"precision.end_position_mm={p1}", "weld_plan.stick_out_mm=17.5", "weld_design.end_inset_mm.goal=79.7",
             "weld_plan.travel_angle_deg=[10,10,10]", "weld_plan.speed_mps.1=0.008", f"specimen.weld_length_mm={L1}",
             "qc_targets.root_gap_mm.tol=0.4", "auto_teach.max_passes=3", "job.pitch_deg=-3", "job.profile_fps=100",
             "weld_plan.direction=alternate"]
    out, recs = _apply(doc, items)
    assert doc == doc0                                                          # the input document is not changed
    assert [r["path"] for r in recs] == [i.split("=", 1)[0] for i in items]
    assert all(r["source"] == "goal" for r in recs)
    by = {r["path"]: r for r in recs}
    assert (by["precision.end_position_mm"]["old"], by["precision.end_position_mm"]["new"]) == (p0, p1)
    assert by["precision.end_position_mm"]["old_from"] == "job file"
    assert by["weld_plan.speed_mps.1"]["old"] == pytest.approx(doc["weld_plan"]["speed_mps"][1])
    assert by["auto_teach.max_passes"]["old"] == C.AutoTeachConfig().max_passes                  # not in the template: the default
    assert by["auto_teach.max_passes"]["old_from"] == "default (the per-job auto_teach keys)"
    # the effective document goes through the job file's parsers
    proc = WeldProcess.from_dict(weld_process_doc(out, "1G"))
    v0 = doc["weld_plan"]["speed_mps"]
    assert proc.stick_out_mm == 17.5 and proc.speed_mps == (v0[0], 0.008, v0[2]) and proc.direction == "alternate"
    assert [proc.role_angle("travel_angle_deg", r_) for r_ in ("root", "fill", "cap")] == [10.0, 10.0, 10.0]
    assert weld_design_from_doc(out, proc).end_insets_mm == (0.0, 79.7)
    assert Precision.from_doc(out["precision"]).end_position_mm == p1
    assert C.specimen_from_doc(out).weld_length_mm == L1 and out["qc_targets"]["root_gap_mm"]["tol"] == 0.4
    assert C.AutoTeachConfig.from_params(out["auto_teach"]).max_passes == 3
    j2 = C.job_from_dict(out["job"])
    assert (j2.pitch_deg, j2.profile_fps) == (-3, 100)


def test_keys_absent_from_the_job_are_valid_when_the_parser_or_the_template_knows_them():
    """A job file without the WPS keys (they take WeldProcess's defaults) or without a weld_design block: the override is
    accepted, 'old' is the value the run would have used (the parser's default) and a list index seeds that default list."""
    _job, doc = _template()
    for k in ("travel_angle_deg", "speed_mps", "stick_out_mm"):
        doc["weld_plan"].pop(k)
    doc.pop("weld_design")
    out, recs = _apply(doc, ["weld_plan.speed_mps.2=0.0065", "weld_plan.stick_out_mm=15", "weld_design.end_inset_mm.goal=79.7",
                             "weld_plan.equipment.ctwd_mm=21"])
    dflt = WeldProcess()
    by = {r["path"]: r for r in recs}
    assert by["weld_plan.speed_mps.2"]["old"] == dflt.speed_mps[2] and by["weld_plan.speed_mps.2"]["old_from"] == "default (WeldProcess)"
    assert out["weld_plan"]["speed_mps"] == [dflt.speed_mps[0], dflt.speed_mps[1], 0.0065]
    assert by["weld_plan.stick_out_mm"]["old"] == dflt.stick_out_mm
    assert by["weld_design.end_inset_mm.goal"]["old"] == 0.0 and by["weld_design.end_inset_mm.goal"]["old_from"] == "default (WeldDesign)"
    assert out["weld_design"] == {"end_inset_mm": {"goal": 79.7}}                # only the given key: the rest stays the parser's default
    assert weld_design_from_doc(out, WeldProcess.from_dict(out["weld_plan"])).end_insets_mm == (0.0, 79.7)
    assert by["weld_plan.equipment.ctwd_mm"]["old"] is None and out["weld_plan"]["equipment"] == {"ctwd_mm": 21}   # a free mapping


@pytest.mark.parametrize("item,match", [
    ("weld_plan.stick_out=1", "unknown key 'stick_out' in block 'weld_plan'"),
    ("job.speed_mp=0.1", "unknown key 'speed_mp' in block 'job'"),
    ("precision.end_mm=0.1", "unknown key 'end_mm' in block 'precision'"),
    ("foo.bar=1", "unknown block 'foo'"),
    ("teach.start=[0,0,0]", "block 'teach' cannot be overridden"),
    ("start_pose.joints_deg=[0,0,0,0,0,0]", "block 'start_pose' cannot be overridden"),
    ("job.start=[0,0,0]", "the rough start point comes from the goal"),
    ("job.position=3G", "the welding position comes from the goal"),
    ("job.safe_clearance_m=0.05", "derived each run by the safety policy"),
    ("job.torch_lateral_m=[0,0]", "the torch keypoint comes from the robot model"),
    ("weld_design.end_inset_mm.middle=1", "unknown key 'middle' under 'weld_design.end_inset_mm' (known: start, goal)"),
    ("precision.end_position_mm.sigma=0.1", "'precision.end_position_mm' is a single value"),
    ("weld_plan.speed_mps.5=0.01", "'5' is not an index of the list 'weld_plan.speed_mps' (0..2)"),
    ("weld_plan=3", "a whole block cannot be replaced"),
])
def test_unknown_keys_and_keys_a_run_does_not_take_from_the_job_are_refused_by_name(item, match):
    _job, doc = _template()
    with pytest.raises(J.OverrideError, match=match.replace("(", r"\(").replace(")", r"\)").replace(".", r"\.")) as e:
        _apply(doc, [item])
    assert item.split("=", 1)[0] in str(e.value)                                # the message names the key


@pytest.mark.parametrize("item,match", [
    ("weld_plan.travel_angle_deg=[10,10]", "needs 3 values (root, fill, cap)"),
    ("weld_plan.stick_out_mm=long", "could not convert string to float"),
    ("weld_plan.stick_out_mm=-2", "stick_out_mm must be >= 0"),
    ("weld_plan.direction=uphillish", "direction must be one of"),
    ("precision.end_position_mm=-1", "precision.end_position_mm must be finite and > 0"),
    ("weld_design.end_inset_mm.goal=-5", "must be >= 0"),
    ("specimen.kind=vee", "kind must be one of"),
    ("qc_targets.root_gap_mm.tol=abc", "could not convert string to float"),
    ("auto_teach.max_passes=0", "max_passes >= 1"),
    ("job.speed_mps=fast", "type error - the job value is a number"),
    ("job.bidirectional=1", "type error - the job value is a yes/no"),
    ("job.pitch_deg=[1]", "type error - the job value is a number"),
])
def test_type_errors_surface_through_the_job_parsers(item, match):
    _job, doc = _template()
    with pytest.raises(J.OverrideError, match=match.replace("(", r"\(").replace(")", r"\)")) as e:
        _apply(doc, [item])
    assert item.split("=", 1)[0] in str(e.value)


def test_text_forms_set_items_files_and_the_goal_field():
    assert J.parse_set_items(["a.b=1e-5", "c.d=[10, 10, 10]", "e.f=S->E", "g.h=true", "i.j="]) == {
        "a.b": 1e-5, "c.d": [10, 10, 10], "e.f": "S->E", "g.h": True, "i.j": None}
    assert J.parse_set_items(["weld_design.end_inset_mm={start: 5, goal: 79.7}"]) == {
        "weld_design.end_inset_mm.start": 5, "weld_design.end_inset_mm.goal": 79.7}   # a mapping value merges like a file
    assert J.parse_set_items(["a.b=1", "a.b=2"]) == {"a.b": 2}                 # later items win
    with pytest.raises(J.OverrideError, match="expects KEY=VALUE"):
        J.parse_set_items(["weld_plan.stick_out_mm"])
    with pytest.raises(J.OverrideError, match="bad path"):
        J.parse_set_items(["weld_plan..x=1"])
    nested = "precision:\n  end_position_mm: 0.1\nweld_plan.stick_out_mm: 17.5\nweld_design: {end_inset_mm: {goal: 79.7}}\n"
    flat = {"precision.end_position_mm": 0.1, "weld_plan.stick_out_mm": 17.5, "weld_design.end_inset_mm.goal": 79.7}
    assert J.parse_mapping_text(nested, what="file") == flat
    assert J.parse_mapping_text(json.dumps(flat), what="param") == flat and J.parse_mapping_text("", what="x") == {}
    with pytest.raises(J.OverrideError, match="must be a mapping"):
        J.parse_mapping_text("[1, 2]", what="server parameter job_overrides")
    with pytest.raises(J.OverrideError, match="not YAML / JSON"):
        J.parse_mapping_text("{a: [1,", what="server parameter job_overrides")
    # the goal field: a JSON object (json.dumps writes exponent floats that YAML 1.1 would read as text)
    g = SimpleNamespace(job_overrides=J.to_goal_text({"precision.end_position_mm": 1e-5, "weld_plan.travel_angle_deg": [10, 10, 10]}))
    assert J.goal_overrides(g) == {"precision.end_position_mm": 1e-5, "weld_plan.travel_angle_deg": [10, 10, 10]}
    assert J.goal_overrides(SimpleNamespace()) == {} and J.goal_overrides(SimpleNamespace(job_overrides="")) == {}
    assert J.to_goal_text({}) == ""
    with pytest.raises(J.OverrideError, match="not a JSON object"):
        J.goal_overrides(SimpleNamespace(job_overrides="{precision.end_position_mm: 0.1}"))
    assert J.merge({"a": 1, "b": 2}, {"a": 3}) == {"b": 2, "a": 3}


def test_the_bead_toe_angle_is_a_site_adjustment_and_the_removed_aim_keys_are_refused():
    """The convex bead's one shape value (weld_plan.bead_toe_angle_deg, g1-bead 2026-10-13 night) is an override the client's help
    lists and it reaches WeldProcess (arc height / chord = tan(toe / 2) / 2); the removed 09-22 aim keys are refused by name."""
    _job, doc = _template()
    t0 = float(doc["weld_plan"]["bead_toe_angle_deg"])
    out, recs = _apply(doc, [f"weld_plan.bead_toe_angle_deg={t0 + 10.0}"])
    assert [(r["path"], r["old"], r["old_from"]) for r in recs] == [("weld_plan.bead_toe_angle_deg", t0, "job file")]
    proc = WeldProcess.from_dict(weld_process_doc(out, "1G"))
    assert proc.bead_toe_angle_deg == t0 + 10.0
    assert proc.bead_aspect_ratio == pytest.approx(np.tan(np.radians(t0 + 10.0) / 2.0) / 2.0)
    assert "weld_plan.bead_toe_angle_deg" in [k for k, _ in J.COMMON_KEYS]
    for item in ("weld_plan.aim_lateral_tcp_mm=[[0],[0,0],[0]]", "weld_plan.aim_lateral_axis=[[0],[0,0],[0]]"):
        with pytest.raises(J.OverrideError, match="unknown key 'aim_lateral_") as e:
            _apply(doc, [item])
        assert item.split("=", 1)[0] in str(e.value)


def test_server_parameter_then_goal_layers_and_the_record(tmp_path):
    _job, doc = _template()
    p0, so0 = doc["precision"]["end_position_mm"], doc["weld_plan"]["stick_out_mm"]             # the template's values
    site = J.parse_mapping_text("{precision: {end_position_mm: 0.2}, weld_plan.stick_out_mm: 15}", what="server parameter job_overrides")
    goal = J.parse_set_items(["precision.end_position_mm=0.10", "weld_plan.travel_angle_deg=[10,10,10]"])
    out, recs = J.apply_layers(doc, [("server parameter", site), ("goal", goal)], position="1G")
    assert out["precision"]["end_position_mm"] == 0.1 and out["weld_plan"]["stick_out_mm"] == 15    # the goal on top of the site
    assert [(r["path"], r["old"], r["new"], r["source"], r["old_from"]) for r in recs] == [
        ("precision.end_position_mm", p0, 0.2, "server parameter", "job file"),
        ("weld_plan.stick_out_mm", so0, 15, "server parameter", "job file"),
        ("precision.end_position_mm", 0.2, 0.1, "goal", "server parameter"),
        ("weld_plan.travel_angle_deg", list(WeldProcess().to_dict()["travel_angle_deg"]), [10, 10, 10], "goal", "job file")]
    lines = J.record_lines(recs)
    assert lines[2] == "precision.end_position_mm 0.2 -> 0.1 (goal; old from server parameter)"
    assert J.summary_line(recs, "job.yaml").startswith(f"job overrides (4) on job.yaml: precision.end_position_mm {J._short(p0)} -> 0.2 "
                                                       "(server parameter; old from job file); ")
    assert J.summary_line([], "job.yaml") == "job overrides: none (the job file on job.yaml as it is)"
    with pytest.raises(ValueError, match="source must be one of"):
        J.apply_overrides(doc, goal, source="operator")
    # the effective document carries the records; the run saves it (meta.job_file / meta.job_overrides) as YAML
    out["meta"] = {**dict(out.get("meta") or {}), "job_file": TEMPLATE, "job_overrides": recs}
    assert J.records_of(out) == recs and J.job_file_of(out) == TEMPLATE and J.records_of({}) == [] and J.job_file_of(None) is None
    p = J.write_effective_job(tmp_path / "verification" / J.EFFECTIVE_JOB_NAME, out)
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# the effective job of this run") and yaml.safe_load(text) == json.loads(json.dumps(out, default=float))
    assert not list(p.parent.glob("*.tmp"))


def test_the_measurement_speed_range_comes_from_the_calibration_runs(tmp_path):
    """The noise priors' speed model s0^2(v) was fitted on runs at 15..60 mm/s (eoat_lab_01: the commanded speeds of its 25 runs):
    a run-time measurement speed inside is accepted (the bounds included), outside refused with the range and its source; a
    calibration without a range refuses any run-time speed (the job file's own speed is not a run-time choice)."""
    calib = C.load_calib(CALIB)
    rng = calibrated_speed_range(calib.noise_priors)
    assert (rng["lo_mps"], rng["hi_mps"], rng["n_runs"]) == (0.015, 0.06, 25)
    assert rng["source"].startswith("the commanded measurement speeds of the 25 calibration runs")
    for v in (15 * 1e-3, 0.02, 0.0375, 60 * 1e-3):
        rec = check_measure_speed(v, calib.noise_priors, calib_source=calib.source)
        assert rec["speed_mps"] == v and (rec["lo_mps"], rec["hi_mps"]) == (0.015, 0.06) and rec["calib"] == calib.source
    for v in (0.0149, 0.0601, 0.1):
        with pytest.raises(ValueError, match=r"outside the calibrated range 15\.\.60 mm/s of the noise priors"):
            check_measure_speed(v, calib.noise_priors, calib_source=calib.source)
    no_range = {k: v for k, v in calib.noise_priors.items() if not k.startswith("speed_range")}
    assert calibrated_speed_range(no_range) is None and calibrated_speed_range(None) is None
    with pytest.raises(ValueError, match="states no calibrated speed range"):
        check_measure_speed(0.03, no_range)
    for bad in ([0.06, 0.015], [0.0, 0.06], "fast", [0.015]):
        with pytest.raises(ValueError, match="speed_range_mps"):
            calibrated_speed_range({**no_range, "speed_range_mps": bad})
    with pytest.raises(ValueError, match="finite speed > 0"):
        check_measure_speed(float("nan"), calib.noise_priors)
    # the calibration tool writes the range from the runs' run logs (commanded speeds)
    from scan_l0 import calib_noise_priors as CNP
    for name, rl in (("r15", {"job_rough": {"speed_mps": 0.015}}), ("r60", {"measure_speed": {"speed_mps": 0.06}, "job_rough": {}}),
                     ("rx", {})):
        (tmp_path / name / "verification").mkdir(parents=True)
        (tmp_path / name / "verification" / "run_log.json").write_text(json.dumps(rl), encoding="utf-8")
    assert [CNP.run_commanded_speed(str(tmp_path / n)) for n in ("r15", "r60", "rx", "none")] == [0.015, 0.06, None, None]
    ev = [{"run": n, "commanded_speed_mps": CNP.run_commanded_speed(str(tmp_path / n))} for n in ("r15", "r60", "rx")]
    lo, hi, src = CNP.speed_range(ev)
    assert (lo, hi) == (0.015, 0.06) and "of the 2 calibration runs" in src and "e.g. r15" in src and "e.g. r60" in src
    cal = {"end_crossing": {"static_var_m2": 1e-9, "timing_var_s2": 0.0, "dynamic_var_s4_m2": 1e-3, "nu0": 10.0}}
    blk = yaml.safe_load("g1:\n" + CNP.block_text(cal, ["r15", "r60"], (lo, hi, src)))["g1"]["noise_priors"]
    assert calibrated_speed_range(blk)["lo_mps"] == 0.015 and blk["speed_range_source"] == src
    assert "speed_range_mps" not in CNP.block_text(cal, ["r"], (None, None, "no run log with a measurement speed"))


def test_the_kiro_calibration_carries_the_lab_priors_and_their_speed_range():
    """KIRO's calibration file starts from the lab's (the same sensor, bracket and torch): its noise priors are the lab's 25 runs', so
    the speed range a run-time measurement speed is checked against is the lab file's (values and source; kiro-1g 2026-10-13)."""
    lab, kiro = C.load_calib(CALIB), C.load_calib(KIRO_CALIB)
    comps = ("end_crossing", "end_reading", "leg_offset_w", "leg_offset_n")   # the variance components the estimator reads
    assert all(kiro.noise_priors[c] == lab.noise_priors[c] for c in comps)
    for k in set(kiro.noise_priors) - {"source"}:                              # the KIRO file's source says it was copied
        assert kiro.noise_priors[k] == lab.noise_priors[k], k
    assert calibrated_speed_range(kiro.noise_priors) == calibrated_speed_range(lab.noise_priors) is not None


def test_the_run_records_a_job_speed_override_and_checks_it_like_the_goal_speed(tmp_path):
    """The measurement speed chosen at run time - the goal's measure_speed_mps or a job.speed_mps override - is one thing: the run's
    record names it (the job file's value = the override record's old) and the 1G core's gates check it against the calibrated range
    before any motion."""
    from scan_l0.autoteach import AutoTeach
    from scan_l0.session import ScanSession
    from scan_l0.teach_1g import Teach1G
    job, doc = _template()
    out, recs = _apply(doc, ["job.speed_mps=0.02"])
    out["meta"] = {**dict(out.get("meta") or {}), "job_file": TEMPLATE, "job_overrides": recs}
    job2 = C.job_from_dict(out["job"])
    st = C.load_settings(LAB)
    calib = C.load_calib(CALIB)
    kw = dict(out_dir=str(tmp_path / "o"), session=ScanSession(robot=None, sensor=None, owned=False))
    v_file = float(job.speed_mps)                                               # the template's (the job file's) measurement speed
    at = AutoTeach(st, job2, out, calib, **kw)
    assert at.job_overrides() == recs and at.measure_speed_choice() == "job override job.speed_mps (goal)"
    rec = at.measure_speed_record()
    assert rec == {"speed_mps": 0.02, "job_file_speed_mps": v_file, "override_mps": 0.02, "source": "job override job.speed_mps (goal)"}
    assert at.measure_speed_text() == f"job override job.speed_mps (goal); the job file's speed_mps {v_file * 1e3:.1f} mm/s"
    at0 = AutoTeach(st, job, doc, calib, **kw)
    assert at0.measure_speed_choice() is None and at0.job_overrides() == []
    # the 1G core's check (one path for both run-time choices); the job file's own speed is not checked
    t1 = Teach1G(st, job2, out, calib, **kw)
    t1._check_measure_speed()
    assert t1.measure_speed_check["speed_mps"] == 0.02 and t1.measure_speed_record()["calibrated_range"]["hi_mps"] == 0.06
    t_goal = Teach1G(st, job, doc, calib, measure_speed_mps=0.045, **kw)
    t_goal._check_measure_speed()
    assert t_goal.measure_speed_record()["calibrated_range"]["speed_mps"] == 0.045
    from scan_l0.autoteach import _Abort
    for v in (0.07, 0.01):
        t_bad = Teach1G(st, job, doc, calib, measure_speed_mps=v, **kw)
        with pytest.raises(_Abort, match="outside the calibrated range 15..60 mm/s") as e:
            t_bad._check_measure_speed()
        assert e.value.status == "REFUSED" and "(nothing sent)" in str(e.value.reason)
    out_fast, recs_fast = _apply(doc, ["job.speed_mps=0.08"])
    out_fast["meta"] = {"job_file": TEMPLATE, "job_overrides": recs_fast}
    with pytest.raises(_Abort, match=r"measurement speed \(job override job.speed_mps \(goal\)\) 80 mm/s is outside"):
        Teach1G(st, C.job_from_dict(out_fast["job"]), out_fast, calib, **kw)._check_measure_speed()
    t_file = Teach1G(st, dataclass_speed(job, 0.1), doc, calib, **kw)
    t_file._check_measure_speed()                                               # the job file's own speed: no run-time choice
    assert t_file.measure_speed_check is None
    assert np.isclose(job.speed_mps, v_file)


def dataclass_speed(job, v):
    import dataclasses
    return dataclasses.replace(job, speed_mps=v)
