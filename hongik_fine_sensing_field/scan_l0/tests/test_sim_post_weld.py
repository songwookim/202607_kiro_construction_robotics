"""Post-weld data-acquisition scan (scan_l0.post_weld.PostWeldScan, user 2026-10-13) on the virtual arm.

A normal 1G core run (Teach1G, the lab layout, 30 mm/s, positioning at the platform limits), then the weld bead appears (the
welding: a box on the plate top along the seam in the virtual world - the sensor sees it), then the post-weld scan on the run's
folder from where the run left the arm: a dry run as after pass 1 (sends nothing, writes nothing), then the final scan. Checked:
the scan moves like a pass of the run (its world - the same MoveIt objects and envelope, nothing added for the bead -, its recorded
standoff, pitch and tool orientation, the move classes and speeds), the line covers [S - 20 mm, E + 20 mm] of the run's measured
seam at the constant measurement speed (the laser hit on the plate-top line, from the recorded robot poses), raw data only (no
profile checks, the run's sensor settings, every frame recorded), the output folder (raw.npz in the pass legs' format + run_log.json
+ the sensor backup; nothing else, nothing else in the run folder changes).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from groove_world import LAB_E_TRUE, LAB_S_TRUE, BoxPrim, SwitchablePrim
from scan_core.path_gen import laser_plane_line_param
from scan_l0 import config as C
from scan_l0.post_weld import OUT_PREFIX, SCHEMA, SPAN_M, PostWeldScan, load_input_run, out_dir_for
from scan_l0.teach_1g import MOVE_MEASURE, MOVE_POSITION
from sim_harness import CALIB, RUN_KW, executed, rot_between_deg
from test_sim_teach1g import calib_1g, lab_limits_settings, run1g, scene

V_RUN = 0.03                      # the lab measurement speed (user 2026-10-12 night)
BEAD_H_TRUE = 0.0032              # the welded bead in the virtual world (the sensor sees it; the motion adds nothing for it)
BEAD_W_TRUE = 0.014


@pytest.fixture
def calib():
    return C.load_calib(CALIB)


def _files(d) -> dict:
    """{relative path: (size, mtime_ns)} of every file under d."""
    out = {}
    for root, _dirs, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            out[os.path.relpath(p, d)] = (st.st_size, st.st_mtime_ns)
    return out


def _post_weld(st, cell, run_dir, calib, after_pass: int = 0, **kw):
    a = dict(RUN_KW)
    a.update(session=cell.session, release_sensor=True, sensor_ip="127.0.0.1", sensor_port=cell.fs.port, local_speed_mps=None,
             descend_speed_mps=None, transit_speed_mps=None)
    a.pop("dwell_frames", None)
    a.pop("probe_frames", None)
    a.update(kw)
    lines: list = []
    pw = PostWeldScan(st, load_input_run(run_dir), calib_1g(calib), out_dir=str(out_dir_for(run_dir, after_pass)), after_pass=after_pass,
                      log=lines.append, **a)
    return pw, pw.run(), lines


def _hit_s(T_bf, calib, q0, u) -> np.ndarray:
    """Laser hit parameter s [m] on the line q0 + s u for each flange pose (N, 4, 4)."""
    return np.array([float(laser_plane_line_param(np.asarray(T, dtype=float) @ calib.T_flange_opt, q0, u)) for T in T_bf])


def test_post_weld_scan_after_a_1g_run_covers_S_minus_20_to_E_plus_20_at_the_measurement_speed(calib, tmp_path):
    st = lab_limits_settings(max_passes=2)
    L_true = float(np.linalg.norm(np.asarray(LAB_E_TRUE) - np.asarray(LAB_S_TRUE)))
    bead = SwitchablePrim(BoxPrim(xc=0.5 * L_true, yc=0.0, sx=L_true, sy=BEAD_W_TRUE, h_top=BEAD_H_TRUE))
    run_dir = tmp_path / "run"
    with scene(st, calib, tmp_path, speed=V_RUN, job_fields={"profile_fps": 150.0}, lat_S=0.025, lat_E=-0.02,
               extras=[bead]) as (cell, job, doc):
        world = cell.world
        doc = dict(doc)
        doc["precision"] = {"end_position_mm": 0.06, "root_position_mm": 0.05, "length_mm": 0.05}
        at, r = run1g(st, cell, job, doc, calib, run_dir, local_speed_mps=None, descend_speed_mps=None, transit_speed_mps=None)
        assert r.status in ("PASS", "PASS_UNRESOLVED") and r.weld_plan is not None, (r.status, r.reason)
        before = _files(run_dir)
        run_log0 = json.loads((run_dir / "verification" / "run_log.json").read_text(encoding="utf-8"))
        # the welding: the bead appears on the plate top along the seam (the render cache holds the pre-weld frames)
        bead.on = True
        world._cache.clear()
        T_end_of_run = cell.arm.pose.latest_pose().copy()
        # dry run first - as after pass 1 (the scan may follow any pass; the caller says which - a label only): plans the line and
        # the moves, sends nothing, opens no sensor, writes nothing
        pwd, rd, lines_d = _post_weld(st, cell, run_dir, calib, after_pass=1, dry_run=True)
        assert rd.status == "DRY_RUN" and rd.out_dir == "" and rd.after_pass == 1 and not executed(pwd.res), (rd.status, rd.reason,
                                                                                                             lines_d[-5:])
        assert np.allclose(cell.arm.pose.latest_pose(), T_end_of_run) and _files(run_dir) == before
        assert cell.session.sensor is None and cell.arm.boxes == {}                    # no sensor opened, our scene objects removed
        assert any("dry run, not moved - 1) joint-space move" in l_ for l_ in lines_d), lines_d[-6:]
        assert lines_d[-1].startswith("post-weld scan DRY_RUN: run run, after pass 1 of 4, line "), lines_d[-1]
        dry_line = pwd.plan["line"]
        # the scan
        n_streams0 = len(cell.fs.stream_log)
        pw, res, lines = _post_weld(st, cell, run_dir, calib)
        fps_streams = sorted({round(float(row[4]), 3) for row in cell.fs.stream_log[n_streams0:]})
        boxes_left = dict(cell.arm.boxes)
        # right after a weld the torch is at the seam, inside the run's grown world (KIRO field 2026-10-15): the scan lifts it out
        # along the plate normal first, then scans; the direction asked for is the one scanned
        T_e = cell.arm.pose.latest_pose().copy()
        n_up = np.asarray(load_input_run(run_dir).n, dtype=float)
        T_low = T_e.copy()
        T_low[:3, 3] -= n_up * 0.12                       # the torch tip ~30 mm above the plate top (inside C = 50 mm)
        q_low = next(np.asarray(c["q"]) for c in cell.arm.reach_candidates(T_low, avoid_collisions=False))
        cell.arm.pose.place(q_low)
        cell.arm._backend = None                          # as on a fresh server: the scan is its first goal (no motion backend yet)
        pw2, res2, lines2 = _post_weld(st, cell, run_dir, calib, after_pass=2, direction="E->S")
    assert res2.status == "OK", (res2.status, res2.reason, lines2[-8:])
    mv2 = executed(pw2.res)
    assert [m["name"] for m in mv2] == ["post_weld_lift", "post_weld_approach", "post_weld_descend", "post_weld_scan"], \
        [m["name"] for m in mv2]
    assert mv2[0]["phase"] == "ascend" and mv2[0]["rot_deg"] < 1e-3 and mv2[0]["dist_mm"] > 0.0, mv2[0]
    assert pw2.plan["direction"] == "E->S" and pw2.plan["tool_orientation"] == "pass"
    assert res.status == "OK" and res.category is None, (res.status, res.reason, lines[-8:])
    out = run_dir / os.path.basename(res.out_dir)
    assert out.name.startswith(OUT_PREFIX + "final_") and out.is_dir() and res.out_dir == str(out) and res.after_pass == 0
    # the line is the same after any pass
    assert np.allclose(pw.plan["line"]["P_from"], dry_line["P_from"]) and np.allclose(pw.plan["line"]["P_to"], dry_line["P_to"])
    # nothing else produced: the run folder gains exactly the post-weld folder; the post-weld folder holds the raw data, its log and
    # the sensor's parameter backup - no estimate, no sections, no weld plan, no KIRO files
    after = _files(run_dir)
    out2 = os.path.basename(res2.out_dir)                 # the second (lift) scan's folder
    assert {k: v for k, v in after.items() if not k.startswith((out.name + os.sep, out2 + os.sep))} == before
    assert sorted(os.listdir(out)) == ["raw.npz", "run_log.json", "sensor_backup"], sorted(os.listdir(out))
    assert boxes_left == {} and fps_streams == [150.0], (boxes_left, fps_streams)
    assert lines[-1].startswith("post-weld scan OK: run run, final (after every pass), line ") and f"-> {out}" in lines[-1], lines[-1]
    assert f"{res.n_frames} frames in " in lines[-1] and " s of scan, " in lines[-1], lines[-1]               # counts only
    # moves: joint-space approach, descent at the positioning speed, the scan line at the measurement speed
    mv = executed(pw.res)
    assert [m["name"] for m in mv] == ["post_weld_approach", "post_weld_descend", "post_weld_scan"], [m["name"] for m in mv]
    v_p = float(st.params["moveit_jtc"]["limits"]["max_tcp_speed_mps"])
    assert mv[0]["phase"] == "joint" and mv[1]["speed_class"] == MOVE_POSITION and mv[1]["speed_mps"] == pytest.approx(v_p)
    assert mv[2]["speed_class"] == MOVE_MEASURE and mv[2]["speed_mps"] == pytest.approx(V_RUN)
    log = json.loads((out / "run_log.json").read_text(encoding="utf-8"))
    assert log["schema"] == SCHEMA and log["status"] == "OK" and log["input_run"]["run_dir"] == str(run_dir)
    assert log["scan"]["speed_mps"] == pytest.approx(V_RUN) and log["scan"]["measure_speed"]["source"] == "the input run's measurement speed"
    # the run's pass settings: standoff, pitch and tool orientation of its last pass
    last_pass = [lg for lg in run_log0["legs"] if lg["kind"] == "pass"][-1]
    with np.load(os.path.join(last_pass["dir"], "raw.npz"), allow_pickle=False) as z:
        pj = json.loads(str(z["job_json"]))
    assert log["scan"]["standoff_m"] == pytest.approx(float(pj["standoff_m"])) and log["scan"]["pitch_deg"] == pytest.approx(pj["pitch_deg"])
    m_pass = [m for m in run_log0["moves"] if m.get("executed") and m["name"] == last_pass["name"]][0]
    assert rot_between_deg(m_pass["to_R"], mv[2]["to_R"]) < 1.0, rot_between_deg(m_pass["to_R"], mv[2]["to_R"])
    # the run's world, as its passes used it: the same MoveIt objects and envelope (user decision 2026-10-13: nothing for the bead)
    w_run, w_pw = run_log0["world"][-1], log["world"][-1]
    assert log["after_pass"] == 0 and log["after_pass_label"] == "final (after every pass)" and "weld_surface" not in log
    # pytest 6 does not approximately compare list values inside a dict. Check
    # each field with the same tolerances, including vector-valued plate geometry.
    assert set(w_pw["envelope"]) == set(w_run["envelope"])
    for key, value in w_run["envelope"].items():
        assert w_pw["envelope"][key] == pytest.approx(value)
    assert [o["id"] for o in w_pw["objects"]] == [o["id"] for o in w_run["objects"]]
    for a_, b_ in zip(w_pw["objects"], w_run["objects"]):
        assert np.allclose(a_["size_m"], b_["size_m"]) and np.allclose(a_["pose"]["position_m"], b_["pose"]["position_m"]), a_["id"]
    assert log["scan"]["standoff_m"] == float(pj["standoff_m"])                     # the run's recorded standoff, exactly
    # raw.npz: the pass legs' format + the scan definition
    with np.load(out / "raw.npz", allow_pickle=False) as z:
        keys = set(z.files)
        for k in ("sensor_x_mm", "sensor_z_mm", "sensor_t_rx_ns", "robot_t_rx_ns", "robot_T_base_flange", "job_json", "calib_json",
                  "config_json", "motions_json", "status_json", "post_weld_scan_json"):
            assert k in keys, (k, sorted(keys))
        t = np.asarray(z["robot_t_rx_ns"], dtype=float) * 1e-9
        T_bf = np.asarray(z["robot_T_base_flange"], dtype=float)
        n_frames = int(z["sensor_t_rx_ns"].shape[0])
        definition = json.loads(str(z["post_weld_scan_json"]))
        definition_sensor_init = json.loads(str(z["sensor_info_json"]))["init_params"]
        assert "client_stats" in json.loads(str(z["sensor_summary_json"]))
        assert json.loads(str(z["job_json"]))["speed_mps"] == pytest.approx(V_RUN)
    assert n_frames == res.n_frames > 0 and definition["schema"] == SCHEMA and definition["line"]["span_m"] == pytest.approx(SPAN_M)
    # the line: the laser hit on the measured plate-top line covers [S - 20, E + 20] mm while the hit moves at the measurement speed
    run = load_input_run(run_dir)
    q0 = run.S + run.n * run.depth_m
    s = _hit_s(T_bf, calib_1g(calib), q0, run.u)
    o = np.argsort(t)
    t, s = t[o], s[o]
    L = float((run.E - run.S) @ run.u)
    ds = np.gradient(s, t)
    sgn = 1.0 if s[-1] > s[0] else -1.0
    cruise = np.abs(sgn * ds - V_RUN) <= 0.02 * V_RUN
    lo, hi = float(np.min(s[cruise])), float(np.max(s[cruise]))
    step = V_RUN * float(np.median(np.diff(t)))
    assert lo <= -SPAN_M + step and hi >= L + SPAN_M - step, (lo * 1e3, hi * 1e3, L * 1e3, step * 1e3)
    inside = (s >= -SPAN_M) & (s <= L + SPAN_M)
    assert np.all(cruise[inside]), np.round(ds[inside & ~cruise] * 1e3, 2)[:10]     # constant speed over the whole data span
    # the line in the run log: the measured ends -/+ 20 mm in base, and in World (the platform's kiro_output World <- base)
    ln = log["line"]
    assert np.allclose(ln["base_m"]["span_from"], run.S - run.u * SPAN_M) and np.allclose(ln["base_m"]["span_to"],
                                                                                          run.S + run.u * (L + SPAN_M))
    assert ln["world_m"] is not None and ln["world_frame"] == "World" and ln["direction"] in ("S->E", "E->S")
    assert log["timing"]["scan_motion_s"] > 0 and log["timing"]["positioning_motion_s"] > 0 and log["n_frames"] == n_frames
    # the overscan beyond the span = the pass's cruise terms only: the speed ramp v^2/2a and the settling v t_settle at the first end
    ov = log["scan"]["overscan"]
    assert ov["speed_ramp_m"] == pytest.approx(V_RUN ** 2 / (2.0 * float(pw.accel)))
    assert ov["first_end_m"] == pytest.approx(SPAN_M + ov["speed_ramp_m"] + ov["settling_m"]) and \
        ov["last_end_m"] == pytest.approx(SPAN_M + ov["speed_ramp_m"])
    assert log["scan"]["settling"]["n_legs"] >= 2, log["scan"]["settling"]                 # the input run's transit + passes
    # raw data only (user 2026-10-13): nothing evaluates the profiles; the sensor ran with the run's recorded settings
    assert pw.res.estop_record is None and log["profile_checks"].startswith("none") and "sensor_guard" not in log
    run_init = json.loads(str(np.load(os.path.join(last_pass["dir"], "raw.npz"))["sensor_info_json"]))["init_params"]
    assert log["sensor_settings"]["init_params"] == run_init and definition_sensor_init == run_init
    assert "ShutterTime" in run_init and int(run_init["ExposureMode"]) == 0             # fixed exposure (no auto exposure)
    print(f"[post-weld] {lines[-1]}; hit span {lo * 1e3:.1f} .. {hi * 1e3:.1f} mm at constant {V_RUN * 1e3:.0f} mm/s (L {L * 1e3:.1f} mm), "
          f"overscan first {ov['first_end_m'] * 1e3:.2f} / last {ov['last_end_m'] * 1e3:.2f} mm")
    # the default tool orientation = the sensing passes' (the same view): the lift scan's flange orientation = the last pass's
    assert rot_between_deg(m_pass["to_R"], mv2[3]["to_R"]) < 1.0, rot_between_deg(m_pass["to_R"], mv2[3]["to_R"])
    assert ln["tool_orientation"] == "pass"
