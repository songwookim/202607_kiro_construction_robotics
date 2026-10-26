"""KIRO hand-eye fix of the profiler calib: math, calib files, re-based estimates, guarded corrections."""

import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from construct_robot.application import scan_correction
from construct_robot.core.scan_calib import (
    calib_matrix, handeye_translation_fix, laser_rotation, rebase_optical, rebase_points, rpy_matrix, same_calib,
)
from construct_robot.io import scan_calib
from construct_robot.nodes.hongik_scan_estimate import rebase_recording_calib
from construct_robot.nodes.scan_recorder import ScanRecorder
from construct_robot.nodes.scanner_live import LASER_FRAME, calib_transform

RPY = [3.11659954, -0.02107031, 1.57079633]
BASE = {"xyz": [0.047463, 0.001382, -0.099097], "rpy": RPY}
FIXED = {"xyz": [0.039454, 0.024934, -0.100543], "rpy": RPY}


def quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def calib_file(tmp_path, pose=BASE, name="calib.yaml"):
    path = tmp_path / name
    path.write_text("# lab calib\nparent_frame: laser_link\nchild_frame: profiler_optical_frame\n"
                    f"xyz: [{', '.join(str(v) for v in pose['xyz'])}]\nrpy: [{', '.join(str(v) for v in pose['rpy'])}]\n"
                    "g1:\n  dt_s: 0.0037\n")
    return path


def test_rpy_matrix_matches_the_live_view_static_tf(tmp_path):
    xyz, q = calib_transform(calib_file(tmp_path))
    np.testing.assert_allclose(rpy_matrix(RPY), quat_matrix(q), atol=1e-12)
    np.testing.assert_allclose(calib_matrix(xyz, RPY)[:3, 3], BASE["xyz"])


def scan_poses(n=5):
    """World <- laser_link of a constant-orientation line scan (laser tilted like the C -> D scan)."""
    T_wl = np.repeat(np.eye(4)[None], n, axis=0)
    T_wl[:, :3, :3] = rpy_matrix([0.1, -0.2, 2.0])
    T_wl[:, :3, 3] = np.c_[np.full(n, 0.03), -1.28 - 0.05 * np.arange(n), np.full(n, 0.77)]
    return T_wl


def test_a_measured_world_offset_becomes_the_calib_translation_fix():
    T_wl = scan_poses()
    T_lo_old, T_lo_true = calib_matrix(BASE["xyz"], RPY), calib_matrix(FIXED["xyz"], RPY)
    recorded, true = T_wl @ T_lo_old, T_wl @ T_lo_true
    point_opt = np.array([0.004, 0.0, 0.29, 1.0])              # the root corner seen on the laser line
    seen, real = (recorded[2] @ point_opt)[:3], (true[2] @ point_opt)[:3]
    R_wl, spread = laser_rotation(recorded, T_lo_old)
    assert spread < 1e-6
    fix = handeye_translation_fix([(R_wl, real - seen)] * 2)
    np.testing.assert_allclose(np.asarray(BASE["xyz"]) + fix["mean_m"], FIXED["xyz"], atol=1e-12)
    rebased = rebase_optical(recorded, T_lo_old, T_lo_true)
    np.testing.assert_allclose(rebased, true, atol=1e-12)
    points = np.array([seen, (recorded[4] @ point_opt)[:3]])
    np.testing.assert_allclose(rebase_points(points, [2, 4], recorded, rebased),
                               [real, (true[4] @ point_opt)[:3]], atol=1e-12)


def test_same_calib_treats_unknown_as_comparable():
    assert same_calib(BASE, dict(BASE)) and same_calib(None, BASE) and not same_calib(BASE, FIXED)


def test_write_handeye_calib_changes_only_xyz(tmp_path):
    base = calib_file(tmp_path)
    out = scan_calib.write_handeye_calib(base, FIXED["xyz"], ["KIRO fix", "base: lab"], tmp_path / "kiro.yaml")
    text = out.read_text()
    assert text.startswith("# KIRO fix\n# base: lab\n#\n# lab calib\n")
    doc = yaml.safe_load(text)
    assert doc["xyz"] == pytest.approx(FIXED["xyz"]) and doc["rpy"] == pytest.approx(RPY)
    assert doc["g1"] == {"dt_s": 0.0037} and doc["parent_frame"] == "laser_link"


def test_default_calib_prefers_the_kiro_file(tmp_path, monkeypatch):
    kiro = tmp_path / scan_calib.KIRO_CALIB_NAME
    monkeypatch.setattr(scan_calib, "kiro_calib_path", lambda: kiro)
    monkeypatch.setattr(scan_calib, "hongik_calib_path", lambda: "/hongik/eoat_kiro_01.yaml")
    assert scan_calib.default_calib_path() == "/hongik/eoat_kiro_01.yaml"
    calib_file(tmp_path, FIXED, kiro.name)
    assert scan_calib.default_calib_path() == str(kiro)


def recording(tmp_path, meta):
    scan = tmp_path / "x_scan"
    scan.mkdir()
    (scan / "scan.json").write_text(json.dumps(meta))
    return scan


def test_recorded_calib_snapshot_first_then_the_named_file(tmp_path):
    base = calib_file(tmp_path)
    old = recording(tmp_path, {"sensor": {"calib_file": str(base)}})
    assert scan_calib.recorded_calib(old)["xyz"] == pytest.approx(BASE["xyz"])
    (old / "scan.json").write_text(json.dumps({"sensor": {"calib_file": str(base)}, "calib": dict(FIXED, file="k")}))
    assert scan_calib.recorded_calib(old)["xyz"] == pytest.approx(FIXED["xyz"])
    (old / "scan.json").write_text(json.dumps({"sensor": {"calib_file": str(tmp_path / "gone.yaml")}}))
    assert scan_calib.recorded_calib(old) is None


def test_estimate_rebases_a_recording_made_with_another_calib(tmp_path):
    scan = recording(tmp_path, {"sensor": {"calib_file": str(calib_file(tmp_path))}})
    T_wl = scan_poses()
    recorded = T_wl @ calib_matrix(BASE["xyz"], RPY)
    doc = rebase_recording_calib(scan, calib_file(tmp_path, FIXED, "kiro.yaml"), recorded, log=lambda _t: None)
    assert doc["rebased"] and doc["recorded"]["xyz"] == pytest.approx(BASE["xyz"])
    np.testing.assert_allclose(doc["T_world_optical"], T_wl @ calib_matrix(FIXED["xyz"], RPY), atol=1e-12)
    same = rebase_recording_calib(scan, calib_file(tmp_path), recorded, log=lambda _t: None)
    assert not same["rebased"] and same["T_world_optical"] is recorded


def test_a_correction_is_refused_on_an_estimate_with_another_calib():
    correction = {"calib": dict(FIXED, file="kiro.yaml")}
    scan_correction.check_correction_calib(correction, {"calib": dict(FIXED, file="kiro.yaml")})
    scan_correction.check_correction_calib({}, {"calib": dict(BASE, file="lab.yaml")})          # legacy file
    with pytest.raises(ValueError, match="Compute correction"):
        scan_correction.check_correction_calib(correction, {"calib": dict(BASE, file="lab.yaml")})


def test_estimate_calib_reads_calib_used_first(tmp_path):
    scan = recording(tmp_path, {"sensor": {"calib_file": str(calib_file(tmp_path))}})
    assert scan_correction.estimate_calib(scan)["xyz"] == pytest.approx(BASE["xyz"])
    (scan / "hongik_estimate").mkdir()
    (scan / "hongik_estimate" / "calib_used.json").write_text(json.dumps({"used": dict(FIXED, file="k"), "rebased": True}))
    assert scan_correction.estimate_calib(scan)["xyz"] == pytest.approx(FIXED["xyz"])


def test_recorder_applies_the_calib_to_the_laser_link_pose(tmp_path):
    from builtin_interfaces.msg import Time as TimeMsg
    from geometry_msgs.msg import Transform
    from rclpy.time import Time
    asked = []

    def lookup(_target, source, time, timeout=None):
        asked.append(source)
        t = Transform()
        t.translation.z, t.rotation.w = 0.5, 1.0
        return SimpleNamespace(transform=t, header=SimpleNamespace(stamp=TimeMsg()))
    node = SimpleNamespace(create_publisher=Mock(return_value=SimpleNamespace(publish=Mock())),
                           tf_buffer=SimpleNamespace(lookup_transform=lookup),
                           get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=1)))
    T_lo = calib_matrix(FIXED["xyz"], RPY)
    recorder = ScanRecorder(node, lambda x, z: np.isfinite(z), out_root=str(tmp_path), laser_to_optical=T_lo)
    recorder.start()
    recorder.add(SimpleNamespace(x=[0.0], z=[300.0]), 1)
    recorder.stop()
    assert set(asked) == {LASER_FRAME}
    T_wl = np.eye(4)
    T_wl[2, 3] = 0.5
    np.testing.assert_allclose(recorder.frames[0][3], T_wl @ T_lo)
    folder = recorder.save({"calib": dict(FIXED, file="kiro.yaml")})
    meta = json.loads(open(f"{folder}/scan.json").read())
    assert meta["calib"]["xyz"] == FIXED["xyz"] and LASER_FRAME in meta["pose_source"]
    assert math.isfinite(recorder.all_points()[0, 2])
