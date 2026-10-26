"""KIRO estimate: writes the estimate layout the pipeline reads (P1 = corner 1, passes 2.. = Hongik template)."""

import json

import numpy as np
import pytest
import yaml

from construct_robot.application import kiro_estimate as ke
from construct_robot.application import scan_correction as sc
from construct_robot.application.direct_scan import check_scan_coverage, run_estimate
from test_groove_estimate import synthetic_scan

TEMPLATE = {"format_version": 1, "source_scan": "hongik", "tcp_to_aim_m": 0.0212, "bevel_sign_w": -1,
            "passes": {k: {e: {"aim_rel_seam_m": rel, "orientation_xyzw": [1.0, 0.0, 0.0, 0.0]} for e in ("start", "goal")}
                       for k, rel in ((1, [0, 0, 0]), (2, [0, 0.0049, 0.0041]), (3, [0, 0.0004, 0.0084]), (4, [0, -0.0004, 0.0112]))}}


@pytest.fixture
def scan(tmp_path, monkeypatch):
    P, fi, o = synthetic_scan(skew_deg=0.0)
    T = np.repeat(np.eye(4)[None], len(o), axis=0)
    T[:, :3, 3] = o
    d = tmp_path / "x_scan"
    d.mkdir()
    np.savez(d / "scan.npz", points_world_m=P.astype(np.float32), points_frame_index=fi, T_world_optical=T,
             stamp_ns=np.arange(len(o)), x_mm=np.zeros((len(o), 1)), z_mm=np.zeros((len(o), 1)))
    (d / "scan.json").write_text(json.dumps({"status": "OK"}))
    calib = tmp_path / "calib.yaml"
    calib.write_text("parent_frame: laser_link\nxyz: [0, 0, 0]\nrpy: [0, 0, 0]\n")
    monkeypatch.setattr(sc, "TCP_TO_WIRE_TIP_M", None)
    from construct_robot.application import tcp_calibration
    monkeypatch.setattr(tcp_calibration, "load_offset", lambda path=None: None)
    return d, calib


def test_kiro_estimate_writes_a_pipeline_compatible_estimate(scan):
    d, calib = scan
    report = ke.run_kiro_estimate(d, calib_path=calib, template=TEMPLATE, log=lambda t: None)
    assert report["status"] == "OK" and report["estimator"] == "kiro"
    est = json.loads((d / "hongik_estimate" / "estimate.json").read_text())
    assert est["estimator"] == "kiro" and est["bevel_sign_w"] == -1
    detail = sc.load_estimate_detail(d)
    assert [r["k"] for r in detail["passes"]] == [1, 2, 3, 4]
    p1, p4 = detail["passes"][0], detail["passes"][3]
    # E -> S: start = the E end (along 200), goal = S (along 0); root on corner 1 (lateral -2.5 mm, bottom 0.56 m)
    assert p1["aim_start"][1] == pytest.approx(-1.5 - 0.200, abs=0.002)
    assert p1["aim_goal"][1] == pytest.approx(-1.5, abs=0.002)
    assert p1["aim_start"][0] == pytest.approx(-0.0025, abs=0.0005) and p1["aim_start"][2] == pytest.approx(0.56, abs=0.0005)
    np.testing.assert_allclose(detail["frame"].to_local(p4["aim_goal"] - p1["aim_goal"]), [0, -0.0004, 0.0112], atol=1e-6)


def test_coverage_check_flags_a_scan_that_starts_on_the_plate(tmp_path):
    P, fi, o = synthetic_scan(start=5.0)
    T = np.repeat(np.eye(4)[None], len(o), axis=0)
    T[:, :3, 3] = o
    np.savez(tmp_path / "scan.npz", points_world_m=P, points_frame_index=fi, T_world_optical=T)
    ok, message = check_scan_coverage(tmp_path)
    assert not ok and "starts on the plate" in message


def test_unknown_estimator_is_refused():
    with pytest.raises(ValueError):
        run_estimate("other", "/x")


def test_specimen_spec_reads_the_hongik_job(tmp_path):
    job = tmp_path / "job.yaml"
    job.write_text(yaml.safe_dump({"specimen": {"thickness_mm": 12.0, "bevel_angle_deg": 35.0, "root_gap_mm": 4.0,
                                                "weld_length_mm": 300.0, "bevel_side": "left"},
                                   "qc_targets": {"root_gap_mm": {"tol": 0.4}, "weld_length_mm": {"tol": 1.0, "ambiguous": 6.0}}}))
    spec, tol, path = ke.specimen_spec(job)
    assert spec["thickness_mm"] == 12.0 and spec["weld_length_mm"] == 300.0 and spec["length_ambiguous_mm"] == 6.0
    assert tol["root_gap_mm"] == 0.4 and path == str(job)
    spec, _tol, path = ke.specimen_spec(tmp_path / "missing.yaml")
    assert spec["weld_length_mm"] == 200.0 and path is None


def test_a_seam_shorter_than_the_specimen_is_refused(tmp_path, monkeypatch):
    P, fi, o = synthetic_scan(skew_deg=0.0, plate_len=160.0)
    T = np.repeat(np.eye(4)[None], len(o), axis=0)
    T[:, :3, 3] = o
    d = tmp_path / "short_scan"
    d.mkdir()
    np.savez(d / "scan.npz", points_world_m=P, points_frame_index=fi, T_world_optical=T)
    calib = tmp_path / "calib.yaml"
    calib.write_text("parent_frame: laser_link\nxyz: [0, 0, 0]\nrpy: [0, 0, 0]\n")
    monkeypatch.setattr(ke, "default_job_path", lambda: None)
    with pytest.raises(ke.GrooveEstimateError, match="specimen is 200 mm"):
        ke.run_kiro_estimate(d, calib_path=calib, template=TEMPLATE, log=lambda t: None)


def test_an_estimate_with_a_misread_end_gives_no_correction_or_targets(tmp_path, monkeypatch):
    # Hongik read the S end 37 mm short on 20261015-104624: its correction put the goal 35 mm past the plate
    out = tmp_path / "s_scan" / "hongik_estimate"
    out.mkdir(parents=True)
    ends = {"S": {"point_m": [0.0, -1.574, 0.564]}, "E": {"point_m": [0.0, -1.7365, 0.562]}}
    (out / "estimate.json").write_text(json.dumps({"u": [0, -1, 0], "n_plate": [0, 0, 1], "ends": ends}))
    monkeypatch.setattr(ke, "default_job_path", lambda: None)
    with pytest.raises(ValueError, match="162"):
        sc.compute_correction(tmp_path / "s_scan", references=[])
    from construct_robot.application import scan_four_pass
    with pytest.raises(ValueError, match="misread"):
        scan_four_pass.write_four_pass_teaching(tmp_path / "s_scan", None, orientations={}, out_dir=tmp_path / "o")
    ends["S"]["point_m"] = [0.0, -1.5365, 0.564]
    (out / "estimate.json").write_text(json.dumps({"u": [0, -1, 0], "n_plate": [0, 0, 1], "ends": ends}))
    assert ke.check_estimate_length(tmp_path / "s_scan") == pytest.approx(200.0, abs=0.1)


def test_centred_scan_poses_put_the_laser_straight_down_over_the_groove_centre(scan):
    d, calib = scan
    meta = json.loads((d / "scan.json").read_text())
    meta["scan_start_tcp"] = {"position_m": [0.05, -1.485, 0.9], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    (d / "scan.json").write_text(json.dumps(meta))
    ke.run_kiro_estimate(d, calib_path=calib, template=TEMPLATE, log=lambda t: None)
    poses = ke.centred_scan_poses(d)
    from construct_robot.core.scan_reference import quat_matrix
    for name, along in (("C", -0.030), ("D", 0.200 + 0.030)):
        position, quat = poses[name]
        # bevel on -lateral (corner 1 at x -2.5 mm): the groove centre is half the root gap toward the 90 deg face
        assert position[0] == pytest.approx(0.0, abs=0.0006)
        assert position[1] == pytest.approx(-1.5 - along, abs=0.002)
        assert position[2] == pytest.approx(0.9, abs=0.001)
        np.testing.assert_allclose(quat_matrix(quat)[:, 2], [0.0, 0.0, -1.0], atol=1e-4)   # fitted plate normal
