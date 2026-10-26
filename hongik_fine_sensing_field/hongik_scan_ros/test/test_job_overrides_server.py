"""Run-time job overrides through the client and the server (user 2026-10-13: "parameters must be adjustable per run - site
conditions differ"): client --set / --set-file -> the AutoTeach goal's job_overrides; the server parameter job_overrides (site
level, every goal of both actions - KIRO's way) and the goal's on top in build_rough_job, before anything is planned."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STACK = os.path.dirname(PKG)


def _job_file(tmp_path, name="rough.yaml", **job):
    from scan_l0 import config as C
    with open(C.JOBS_DIR / "groove_10T_1G.yaml", "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc["job"].update(start=[1.07, -0.06, -0.15], end=[1.41, -0.03, -0.15], position="1G", **job)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


def _goal(**kw):
    from geometry_msgs.msg import Point
    g = SimpleNamespace(position="1G", rough_start=Point(x=1.07, y=-0.06, z=-0.15), rough_end=Point(x=1.41, y=-0.03, z=-0.15),
                        weld_length_mm=0.0, thickness_mm=0.0, root_gap_mm=0.0, bevel_angle_deg=0.0, groove_kind="", bevel_side="")
    for k, v in kw.items():
        setattr(g, k, v)
    return g


def test_build_rough_job_applies_the_server_layer_then_the_goal_before_planning(tmp_path):
    from hongik_scan_ros.auto_teach_server import build_rough_job
    from scan_l0 import job_overrides as J
    jp = _job_file(tmp_path)
    site = {"precision.end_position_mm": 0.2, "weld_plan.stick_out_mm": 15.0, "job.pitch_deg": -3.0}
    goal_ov = {"precision.end_position_mm": 0.1, "weld_plan.travel_angle_deg": [10, 10, 10], "job.speed_mps": 0.02,
               "weld_design.end_inset_mm.goal": 79.7}
    job, doc = build_rough_job(None, str(jp), _goal(), base_frame="link0", gravity_axis_base=[0.0, 0.0, -1.0],
                               overrides=[("server parameter", site), ("goal", goal_ov)])
    # the job geometry is the goal's, the overridden job fields reach the ScanJob the run gets
    assert np.allclose(job.start, [1.07, -0.06, -0.15]) and job.base_frame == "link0" and job.position == "1G"
    assert (job.speed_mps, job.pitch_deg) == (0.02, -3.0) and doc["job"]["speed_mps"] == 0.02
    assert doc["precision"]["end_position_mm"] == 0.1 and doc["weld_plan"]["stick_out_mm"] == 15.0
    assert doc["weld_plan"]["travel_angle_deg"] == [10, 10, 10] and doc["weld_design"]["end_inset_mm"]["goal"] == 79.7
    recs = J.records_of(doc)
    assert J.job_file_of(doc) == str(jp) and doc["meta"]["rough"] is True
    assert [(r["path"], r["source"]) for r in recs] == [
        ("precision.end_position_mm", "server parameter"), ("weld_plan.stick_out_mm", "server parameter"),
        ("job.pitch_deg", "server parameter"), ("precision.end_position_mm", "goal"), ("weld_plan.travel_angle_deg", "goal"),
        ("job.speed_mps", "goal"), ("weld_design.end_inset_mm.goal", "goal")]
    v_file = yaml.safe_load(open(jp, encoding="utf-8"))["job"]["speed_mps"]          # the job file's (= the template's) speed
    assert recs[3]["old"] == 0.2 and recs[3]["old_from"] == "server parameter" and recs[5]["old"] == v_file
    # no overrides: the job file as it is, an empty record
    job0, doc0 = build_rough_job(None, str(jp), _goal(), base_frame="link0", gravity_axis_base=[0.0, 0.0, -1.0])
    tpl = yaml.safe_load(open(jp, encoding="utf-8"))                            # the job file (= the template's values)
    assert J.records_of(doc0) == [] and job0.speed_mps == tpl["job"]["speed_mps"] and \
        doc0["precision"]["end_position_mm"] == tpl["precision"]["end_position_mm"]


def test_build_rough_job_refuses_by_name_and_keeps_the_goal_design_fields(tmp_path):
    from hongik_scan_ros.auto_teach_server import build_rough_job
    from scan_l0 import job_overrides as J
    jp = _job_file(tmp_path)
    kw = dict(base_frame="link0", gravity_axis_base=[0.0, 0.0, -1.0])
    for layers, match in (([("goal", {"weld_plan.stick_out": 1})], "unknown key 'stick_out' in block 'weld_plan'"),
                          ([("server parameter", {"job.start": [0, 0, 0]})], "the rough start point comes from the goal"),
                          ([("goal", {"precision.end_position_mm": -1})], "precision.end_position_mm must be finite and > 0")):
        with pytest.raises(J.OverrideError, match=match):
            build_rough_job(None, str(jp), _goal(), overrides=layers, **kw)
    # the goal's own design field and a server-level override of the same key: the goal wins (it is more specific) ...
    job, doc = build_rough_job(None, str(jp), _goal(weld_length_mm=200.0), overrides=[("server parameter", {"specimen.weld_length_mm": 250.0})],
                               **kw)
    assert doc["specimen"]["weld_length_mm"] == 200.0 and J.records_of(doc)[0]["new"] == 250.0
    # ... the goal giving it twice (its field and its job_overrides) is refused
    with pytest.raises(J.OverrideError, match="given twice in the goal"):
        build_rough_job(None, str(jp), _goal(weld_length_mm=200.0), overrides=[("goal", {"specimen.weld_length_mm": 210.0})], **kw)
    job, doc = build_rough_job(None, str(jp), _goal(), overrides=[("goal", {"specimen.weld_length_mm": 210.0})], **kw)
    assert doc["specimen"]["weld_length_mm"] == 210.0


def test_kiro_goal_takes_the_site_overrides_only(tmp_path):
    """KIRO's PrecisionSensing goal has no job_overrides field: the server parameter's layer is applied to its job file."""
    import hongik_scan_ros.kiro_io as K
    from hongik_scan_ros.auto_teach_server import build_rough_job
    from scan_l0 import config as C
    from scan_l0 import job_overrides as J
    goal = SimpleNamespace(welding_position=1, pose_a=SimpleNamespace(position=SimpleNamespace(x=0.8, y=-0.19, z=0.05)),
                           pose_b=SimpleNamespace(position=SimpleNamespace(x=0.8, y=0.19, z=0.05)))
    inp = K.goal_to_run_inputs(goal, g1_pipeline="core")
    rg = K.rough_goal(inp)
    assert J.goal_overrides(rg) == {}
    job, doc = build_rough_job(None, str(C.JOBS_DIR / "groove_10T_1G.yaml"), rg, base_frame="link0", gravity_axis_base=[0.0, 0.0, -1.0],
                               start=np.array(inp.start_m), end=np.array(inp.end_m),
                               overrides=[("server parameter", {"weld_plan.stick_out_mm": 17.5}), ("goal", J.goal_overrides(rg))])
    assert doc["weld_plan"]["stick_out_mm"] == 17.5 and [r["source"] for r in J.records_of(doc)] == ["server parameter"]


def test_server_parameter_is_parsed_and_checked_at_start():
    from hongik_scan_ros.auto_teach_server import AutoTeachServer
    from scan_l0 import config as C
    from scan_l0.ports import PlatformConfigError
    me = SimpleNamespace(job_file=str(C.JOBS_DIR / "groove_10T_1G.yaml"))
    assert AutoTeachServer._site_overrides(me, "") == {}
    assert AutoTeachServer._site_overrides(me, "{weld_plan: {stick_out_mm: 17.5}, precision.end_position_mm: 0.1}") == {
        "weld_plan.stick_out_mm": 17.5, "precision.end_position_mm": 0.1}
    assert AutoTeachServer._site_overrides(me, '{"weld_design.end_inset_mm.goal": 79.7}') == {"weld_design.end_inset_mm.goal": 79.7}
    for bad, match in (("{weld_plan.stick_out: 1}", "unknown key 'stick_out'"), ("[1, 2]", "must be a mapping"),
                       ("{precision.end_position_mm: 0}", "must be finite and > 0")):
        with pytest.raises(PlatformConfigError, match=match) as e:
            AutoTeachServer._site_overrides(me, bad)
        assert str(e.value).startswith("parameter job_overrides: ")


def test_client_set_and_set_file_reach_the_goal_and_are_checked_before_sending(tmp_path):
    from hongik_scan_ros import auto_teach_client as cl
    from scan_l0 import job_overrides as J
    jp = _job_file(tmp_path)
    f = tmp_path / "site.yaml"
    f.write_text("precision:\n  end_position_mm: 0.2\nweld_plan.stick_out_mm: 17.5\n", encoding="utf-8")
    ov = cl.cli_overrides(["precision.end_position_mm=0.10", "weld_plan.travel_angle_deg=[10,10,10]"], str(f))
    assert ov == {"weld_plan.stick_out_mm": 17.5, "precision.end_position_mm": 0.1, "weld_plan.travel_angle_deg": [10, 10, 10]}
    recs = cl.check_overrides(jp, ov)
    assert [(r["path"], r["new"]) for r in recs] == list(ov.items()) and all(r["source"] == "goal" for r in recs)
    g = cl.goal_from_job(jp, job_overrides=ov)
    assert json.loads(g.job_overrides) == ov and J.goal_overrides(g) == ov
    assert cl.goal_from_job(jp).job_overrides == ""
    ap = cl.build_parser()
    a = ap.parse_args(["--job", str(jp), "--set", "job.speed_mps=0.02", "--set", "a.b=1", "--set-file", str(f)])
    assert a.set_items == ["job.speed_mps=0.02", "a.b=1"] and a.set_file == str(f)
    assert "weld_plan.stick_out_mm" in ap.format_help() and "precision.end_position_mm" in ap.format_help()
    # refused before the server is contacted (rc 2): an unknown key, a bad value, the speed given twice, --setup with --set
    said = []
    base = ["--job", str(jp)]
    for extra, match in ((["--set", "weld_plan.stick_out=1"], "REFUSED: job override 'weld_plan.stick_out': unknown key 'stick_out'"),
                         (["--set", "precision.end_position_mm=-1"], "precision.end_position_mm must be finite and > 0"),
                         (["--set", "job.start=[0,0,0]"], "the rough start point comes from the goal"),
                         (["--measure-speed", "30", "--set", "job.speed_mps=0.02"], "the measurement speed is given twice"),
                         (["--measure-speed", "-3"], "--measure-speed must be a speed > 0"),
                         (["--set", "weld_plan.stick_out_mm"], "expects KEY=VALUE"),
                         (["--set-file", str(tmp_path / "missing.yaml")], "--set-file")):
        said.clear()
        assert cl.main(base + extra, say=said.append) == 2, (extra, said)
        assert any(match in s_ for s_ in said), (extra, said)
    said.clear()
    assert cl.main(["--setup", "--job", str(jp), "--set", "weld_plan.stick_out_mm=17.5"], say=said.append) == 2
    assert any("--setup only writes the specimen input" in s_ for s_ in said), said


def test_server_parameter_documented_in_the_help_and_the_launch_files():
    from hongik_scan_ros.node_help import NODE_HELP
    assert "job_overrides (str, '')" in NODE_HELP["hongik_auto_teach_server"]
    for name in ("hongik_moveit.launch.py",):                                    # the KIRO package's one launch
        text = open(os.path.join(PKG, "launch", name), encoding="utf-8").read()
        assert 'DeclareLaunchArgument("job_overrides"' in text, name
        # a string parameter: launch_ros would otherwise read the YAML / JSON mapping text as a mapping and refuse it
        assert '"job_overrides": ParameterValue(LaunchConfiguration("job_overrides").perform(context), value_type=str)' in text, name
    assert "job_overrides" in open(os.path.join(PKG, "config", "hongik_kiro.yaml"), encoding="utf-8").read()
