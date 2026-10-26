"""Post-weld scan units (scan_l0.post_weld, user 2026-10-13) - no arm, no sensor: the line definition, the input-run lookup and
loading (and their refusals), after_pass (a label: folder name, run log; checked against the run's weld plan), the output folder
name, and the cell world restored from a run's world record (the scene its passes used - nothing added for the bead)."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from scan_l0.cell_world import FLOOR_ID, SPECIMEN_ID, CellWorld
from scan_l0.post_weld import (OUT_PREFIX, SPAN_M, InputRunError, after_pass_label, latest_run, load_input_run, out_dir_for,
                               resolve_run_dir, scan_line)

S = np.array([1.05, -0.025, -0.150])
E = np.array([1.428, 0.012, -0.144])
N = np.array([-0.016, -0.0014, 0.9999])
SPEC = {"kind": "single_bevel", "thickness_mm": 10.0, "bevel_angle_deg": 30.0, "root_gap_mm": 5.0, "weld_length_mm": 380.0,
        "plate_widths_mm": [300.0, 300.0], "backing": True, "backing_thickness_mm": 10.0, "derived": {"top_opening_mm": 10.77}}
SENSOR_INIT = {"DataMode": 4, "AutoTriggerFPS": 150.0, "ExposureMode": 0, "ShutterTime": 100, "GainVal": 250}


def test_span_is_20_mm_beyond_each_measured_end():
    assert SPAN_M == 0.020                                             # the user's specification (2026-10-13), not a tuned value


def test_scan_line_spans_the_measured_ends_plus_the_span_along_the_seam():
    ln = scan_line(S, E, N, 0.020)
    u, n = ln["u"], ln["n"]
    L = float((E - S) @ u)
    assert abs(float(u @ n)) < 1e-12 and abs(np.linalg.norm(u) - 1.0) < 1e-12 and abs(np.linalg.norm(n) - 1.0) < 1e-12
    assert float(n @ N) > 0.999 and float(u @ (E - S)) > 0.0
    assert ln["L_m"] == pytest.approx(L) and ln["s_from_m"] == pytest.approx(-0.020) and ln["s_to_m"] == pytest.approx(L + 0.020)
    assert np.allclose(ln["P_from"], S - u * 0.020) and np.allclose(ln["P_to"], S + u * (L + 0.020))
    assert (ln["P_to"] - ln["P_from"]) @ u == pytest.approx(L + 0.040)
    assert scan_line(S, E, N, 0.0)["s_to_m"] == pytest.approx(L)
    with pytest.raises(ValueError, match="span"):
        scan_line(S, E, N, -0.001)
    with pytest.raises(ValueError):
        scan_line(S, S, N)


# ---- the input run ---------------------------------------------------------------------------------------------------------
def _world_record(S_, E_, n_) -> dict:
    w = CellWorld(up=(0, 0, 1), thickness_m=0.01, backing_m=0.01, design_length_m=0.38, clearance_m=0.05, reach_m=0.5,
                  plate_widths_m=(0.3, 0.3), gap_m=0.005)
    w.update_measured(top_point=S_ + n_ * 0.01, sigma_top_m=1e-5, normal=n_, k=3.0, root_lo=S_, root_hi=E_,
                      ends={"S": {"s_m": 0.0}, "E": {"s_m": 0.378}}, why="unit")
    w.set_plate_sides(-1)
    return w.to_dict()


def make_run(root, name="r1", *, pipeline="g1core", plan_where="out", with_pass=True, mtime=None, sensor_info=True) -> str:
    """A minimal finished-run folder (the files load_input_run reads; a 4-pass weld plan when plan_where)."""
    d = root / name
    (d / "verification" / "legs" / "05_pass1").mkdir(parents=True)
    u = (E - S) / np.linalg.norm(E - S)
    n = N - u * float(N @ u)
    n /= np.linalg.norm(n)
    (d / "sections.json").write_text(json.dumps({
        "schema": "g1core.sections/1", "base_frame": "link0", "status": "PASS",
        "frame": {"S_m": S.tolist(), "E_m": E.tolist(), "u": u.tolist(), "n": n.tolist()},
        "ends": {"S": {"two_sided": True}, "E": {"two_sided": False}}, "seam_dims": {"depth_mm": {"value": 10.5}}}), encoding="utf-8")
    legs = [{"name": "transit_S_to_E", "kind": "transit", "dir": "/elsewhere/legs/03_transit_S_to_E"}]
    if with_pass:
        legs.append({"name": "pass1", "kind": "pass", "dir": "/elsewhere/legs/05_pass1"})
    job = {"start": S.tolist(), "end": E.tolist(), "surface_normal": [0, 0, 1], "orientation": "flat_1G", "standoff_m": 0.32,
           "speed_mps": 0.03, "profile_fps": 150.0, "pitch_deg": -5.0, "base_frame": "link0", "gravity_axis_base": [0, 0, -1]}
    (d / "verification" / "run_log.json").write_text(json.dumps({
        "pipeline": pipeline, "status": "PASS", "job_rough": job, "legs": legs, "world": [_world_record(S, E, n)],
        "measure_speed": {"speed_mps": 0.03, "job_file_speed_mps": 0.015}, "prior": {"spec": SPEC}}), encoding="utf-8")
    pj = dict(job, start=E.tolist(), end=S.tolist(), surface_normal=n.tolist(), standoff_m=0.3425, progression_sign=1)
    arrays = {"job_json": np.array(json.dumps(pj))}
    if sensor_info:
        arrays["sensor_info_json"] = np.array(json.dumps({"init_params": SENSOR_INIT, "params_before": {}}))
    np.savez(d / "verification" / "legs" / "05_pass1" / "raw.npz", **arrays)
    if plan_where:
        plan = {"passes": [{"k": k, "kind": "cap" if k == 4 else "fill"} for k in (1, 2, 3, 4)]}
        (d / ("weld_plan.json" if plan_where == "out" else os.path.join("verification", "weld_plan.json"))).write_text(
            json.dumps(plan), encoding="utf-8")
    if mtime is not None:
        os.utime(d / "verification" / "run_log.json", (mtime, mtime))
    return str(d)


def test_load_input_run_takes_the_measured_seam_and_the_runs_scan_settings(tmp_path):
    d = make_run(tmp_path)
    r = load_input_run(d)
    assert np.allclose(r.S, S) and np.allclose(r.E, E) and r.ends_two_sided == {"S": True, "E": False}
    assert r.depth_m == pytest.approx(10.5e-3) and r.run_speed_mps == pytest.approx(0.03) and r.job_file_speed_mps == pytest.approx(0.015)
    assert r.pass_leg == "pass1" and r.pass_job["standoff_m"] == pytest.approx(0.3425) and r.sensor_init == SENSOR_INIT
    assert [lg["name"] for lg in r.stream_legs] == ["transit_S_to_E", "pass1"]
    # the legs' raw.npz are found inside this run folder (the log keeps the run's original absolute paths)
    assert str(r.stream_legs[1]["raw"]).startswith(d) and r.n_plan_passes == 4
    job, doc = r.job_and_doc()
    assert np.allclose(job.start, S) and np.allclose(job.end, E) and job.speed_mps == pytest.approx(0.03)   # the run's speed
    assert job.plate_top_measured_m is None and job.table_z_m is None and doc["specimen"]["weld_length_mm"] == 380.0
    assert job.orientation == "flat_1G" and float(job.normal_unit @ N) > 0.999
    # the weld plan only gives the after_pass range: in verification/ (D14 / sensing DATA_FAIL) or none at all
    assert load_input_run(make_run(tmp_path, "r2", plan_where="verification")).n_plan_passes == 4
    assert load_input_run(make_run(tmp_path, "r3", plan_where=None)).n_plan_passes is None


@pytest.mark.parametrize("how,match", [("legacy", "not a 1G core run"), ("no_pass", "no pass leg"), ("no_raw", "raw.npz"),
                                       ("no_sensor", "no sensor init parameters")])
def test_load_input_run_refusals(tmp_path, how, match):
    kw = {"legacy": {"pipeline": "legacy"}, "no_pass": {"with_pass": False}, "no_raw": {}, "no_sensor": {"sensor_info": False}}[how]
    d = make_run(tmp_path, **kw)
    if how == "no_raw":
        os.remove(os.path.join(d, "verification", "legs", "05_pass1", "raw.npz"))
    with pytest.raises(InputRunError, match=match):
        load_input_run(d)


def test_after_pass_is_a_label_checked_against_the_runs_weld_plan(tmp_path):
    r = load_input_run(make_run(tmp_path))
    assert [r.check_after_pass(k) for k in (0, 1, 4)] == [0, 1, 4]
    for bad, match in ((5, "the run's weld plan has passes 1..4"), (-1, ">= 0")):
        with pytest.raises(ValueError, match=match):
            r.check_after_pass(bad)
    assert load_input_run(make_run(tmp_path, "np", plan_where=None)).check_after_pass(7) == 7        # no plan: no range to check
    assert after_pass_label(0) == "final (after every pass)" and after_pass_label(2, 4) == "after pass 2 of 4"
    assert after_pass_label(3) == "after pass 3"


def test_run_dir_resolution_latest_name_and_absolute(tmp_path):
    root = tmp_path / "out_root"
    root.mkdir()
    with pytest.raises(InputRunError, match="no finished run"):
        latest_run(root)
    t = time.time()
    a = make_run(root, "a_older", mtime=t - 100)
    b = make_run(root, "b_newer", mtime=t)
    (root / "not_a_run").mkdir()
    (root / "dry_only").mkdir()
    (root / "dry_only" / "autoteach_log.json").write_text("{}", encoding="utf-8")
    assert str(latest_run(root)) == b and str(resolve_run_dir("", root)) == b
    assert str(resolve_run_dir("a_older", root)) == a and str(resolve_run_dir(a, root)) == a
    with pytest.raises(InputRunError, match="does not exist"):
        resolve_run_dir("missing", root)
    with pytest.raises(InputRunError, match="not a finished run folder"):
        resolve_run_dir("not_a_run", root)
    inner = Path(b) / "directions" / "E_to_S" / "kiro"                 # a weld-plan set inside a run: the run is named
    inner.mkdir(parents=True)
    with pytest.raises(InputRunError, match=r"is a folder inside the run .*b_newer .*--run .*--direction"):
        resolve_run_dir(str(inner), root)
    with pytest.raises(InputRunError, match="does not exist"):
        latest_run(tmp_path / "nowhere")
    assert issubclass(InputRunError, ValueError)


def test_out_dir_name_after_pass_or_final_and_no_overwrite(tmp_path):
    import datetime as dt
    now = dt.datetime(2026, 10, 13, 9, 5, 7)
    p = out_dir_for(tmp_path, 0, now)
    assert p == tmp_path / f"{OUT_PREFIX}final_20261013-090507" and out_dir_for(tmp_path, now=now) == p
    p.mkdir()
    assert out_dir_for(tmp_path, 0, now) == tmp_path / f"{OUT_PREFIX}final_20261013-090507_2"
    assert out_dir_for(tmp_path, 2, now) == tmp_path / f"{OUT_PREFIX}p2_20261013-090507"
    with pytest.raises(ValueError, match="after_pass"):
        out_dir_for(tmp_path, -1, now)


# ---- the cell world restored from a run's record: the scene its passes used ---------------------------------------------------
def test_world_from_record_is_the_runs_world():
    u = (E - S) / np.linalg.norm(E - S)
    n = N - u * float(N @ u)
    n /= np.linalg.norm(n)
    rec = _world_record(S, E, n)
    w = CellWorld.from_record(rec, thickness_m=0.01, backing_m=0.01, design_length_m=0.38, clearance_m=0.05, reach_m=0.5)
    d = w.to_dict()
    assert np.allclose(d["planes"][0]["point_m"], rec["planes"][0]["point_m"]) and np.allclose(d["planes"][0]["normal"],
                                                                                              rec["planes"][0]["normal"])
    assert np.allclose(d["planes"][0]["extent"]["line_m"], rec["planes"][0]["extent"]["line_m"])
    assert d["envelope"] == pytest.approx(rec["envelope"]) and d["planes"][0]["k_sigma_m"] == pytest.approx(rec["planes"][0]["k_sigma_m"])
    assert {o["id"] for o in d["objects"]} == {SPECIMEN_ID, FLOOR_ID} and w.left_sign_w == -1 and d["ends"] == rec["ends"]
    for a_, b_ in zip(d["objects"], rec["objects"]):
        assert a_["id"] == b_["id"] and np.allclose(a_["size_m"], b_["size_m"]) and np.allclose(a_["pose"]["position_m"],
                                                                                                b_["pose"]["position_m"])
        assert np.allclose(a_["pose"]["quat_xyzw"], b_["pose"]["quat_xyzw"])
    with pytest.raises(ValueError, match="schema"):
        CellWorld.from_record({"schema": "other"}, thickness_m=0.01, backing_m=0.01, design_length_m=0.38, clearance_m=0.05, reach_m=0.5)
    prior = CellWorld.from_rough_points(S, E, hit_face_m=0.01, up=(0, 0, 1), thickness_m=0.01, backing_m=0.01, design_length_m=0.38,
                                        clearance_m=0.05, reach_m=0.5, plate_widths_m=(0.3, 0.3)).to_dict()
    with pytest.raises(ValueError, match="no measured plate top"):
        CellWorld.from_record(prior, thickness_m=0.01, backing_m=0.01, design_length_m=0.38, clearance_m=0.05, reach_m=0.5)
    assert math.isfinite(w.floor_m)


# ---- the scan line's speed chosen at run time: the 1G core's one check against the calibrated range ----------------------------
def test_goal_speed_is_checked_against_the_calibrated_range_before_any_motion(tmp_path):
    """The goal's measure_speed_mps (client --measure-speed: any speed) is a run-time measurement speed: the same check as a run's
    (Teach1G._check_measure_speed -> scan_core.noise_priors.check_measure_speed; KIRO's config, eoat_kiro_01 = the lab priors: 15..60
    mm/s, the bounds included), recorded with its range; outside = REFUSED in the setup gates, nothing sent (no move, no sensor).
    Without a goal speed the scan takes the input run's measurement speed (that run's own; not a run-time choice here - not
    checked)."""
    from scan_l0 import config as C
    from scan_l0.autoteach import _Abort
    from scan_l0.post_weld import PostWeldScan
    from scan_l0.session import ScanSession
    stack = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    st = C.load_settings(os.path.join(stack, "hongik_scan_ros", "config", "hongik_kiro.yaml"))
    calib = C.load_calib(os.path.join(stack, "eoat_description", "calib", "eoat_kiro_01.yaml"))
    run = load_input_run(make_run(tmp_path))

    def pw(v=None, **kw):
        return PostWeldScan(st, run, calib, out_dir=str(tmp_path / "pw"),
                            session=ScanSession(robot=None, sensor=None, owned=False), measure_speed_mps=v, **kw)

    p0 = pw()
    p0._check_measure_speed()
    rec = p0.measure_speed_record()
    assert p0.measure_speed_choice() is None and p0.measure_speed_check is None and "calibrated_range" not in rec
    assert rec["speed_mps"] == pytest.approx(0.03) and rec["source"] == "the input run's measurement speed"
    for v in (0.015, 0.045, 0.06):
        p = pw(v)
        p._check_measure_speed()
        rec = p.measure_speed_record()
        assert rec["speed_mps"] == v and rec["override_mps"] == v and rec["source"] == "goal override (measure_speed_mps)"
        assert (rec["calibrated_range"]["lo_mps"], rec["calibrated_range"]["hi_mps"]) == (0.015, 0.06)
        assert rec["calibrated_range"]["speed_mps"] == v and rec["calibrated_range"]["calib"] == calib.source
    for v in (0.0149, 0.07):
        with pytest.raises(_Abort, match=r"measurement speed \(goal override \(measure_speed_mps\)\) .* mm/s is outside the "
                                         r"calibrated range 15\.\.60 mm/s") as e:
            pw(v)._check_measure_speed()
        assert e.value.status == "REFUSED" and e.value.category == "input" and str(e.value.reason).endswith("(nothing sent)")
    # through the scan's setup (a dry run: the platform-less refusal path): refused in the gates, before the platform is touched (no
    # robot here) - nothing sent, no plan, no sensor, no folder
    p = pw(0.07, dry_run=True)
    res = p.run()
    assert res.status == "REFUSED" and res.category == "input" and "outside the calibrated range 15..60 mm/s" in res.reason, \
        (res.status, res.reason, p.res.notes)
    assert not [m for m in p.res.moves if m.get("executed")] and p.sess.sensor is None and p.plan is None and res.out_dir == ""
    assert not (tmp_path / "pw").exists()
    # a calibration without a range refuses any goal speed (the speed model would be used beyond its fit); none given = runs as is
    no_rng = dict(calib.noise_priors)
    for k in [k for k in no_rng if k.startswith("speed_range")]:
        no_rng.pop(k)
    import dataclasses
    calib_nr = dataclasses.replace(calib, noise_priors=no_rng)
    with pytest.raises(_Abort, match="states no calibrated speed range"):
        PostWeldScan(st, run, calib_nr, out_dir=str(tmp_path / "pw2"),
                     session=ScanSession(robot=None, sensor=None, owned=False), measure_speed_mps=0.03)._check_measure_speed()
    PostWeldScan(st, run, calib_nr, out_dir=str(tmp_path / "pw3"),
                 session=ScanSession(robot=None, sensor=None, owned=False))._check_measure_speed()
