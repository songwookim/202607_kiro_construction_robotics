"""PLY exports reproduce recorded World XYZ and stay linked to their execution without equipment."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from construct_robot.application.scan_weld_cycle import export_post_weld_ply
from construct_robot.io.scan_cloud import load_scan_cloud, write_cloud_ply
from construct_robot.io.scan_cycle_log import ScanCycleLog


def read_ply(path):
    header, body = Path(path).read_bytes().split(b"end_header\n", 1)
    return header.decode("ascii"), np.frombuffer(body, dtype="<f4").reshape(-1, 3)


def test_binary_ply_preserves_every_world_point_and_units(tmp_path):
    points = np.array([[.002, -1.7, .6], [-.005, -1.5, .62]], dtype=np.float64)
    path = write_cloud_ply(tmp_path / "scan.ply", points)
    header, actual = read_ply(path)
    assert "binary_little_endian" in header and "frame World; units metres" in header
    assert "element vertex 2" in header
    np.testing.assert_array_equal(actual, points.astype(np.float32))
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("points", [[[float("nan"), 0, 0]], [[1, 2]], [1, 2, 3]])
def test_invalid_cloud_never_publishes_a_ply(tmp_path, points):
    with pytest.raises(ValueError):
        write_cloud_ply(tmp_path / "scan.ply", points)
    assert not (tmp_path / "scan.ply").exists()


def test_post_scan_export_goes_directly_into_its_four_pass_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("KIRO_SCAN_CYCLE_LOG_ROOT", str(tmp_path / "cycles"))
    journal = ScanCycleLog(tmp_path / "cycles", [], {}, True)
    teaching = tmp_path / "scan_four_pass_example_scan"
    teaching.mkdir()
    journal.event("artifact", kind="four_pass_teaching", pass_number=0, path=str(teaching))
    raw = tmp_path / "post_scan"
    raw.mkdir()
    points = np.array([[.01, -1.7, .58], [.02, -1.5, .59]], np.float32)
    np.savez_compressed(raw / "scan.npz", points_world_m=points)
    before = (raw / "scan.npz").read_bytes()
    host = SimpleNamespace(_scan_cycle_journal=journal, _scan_cycle_log_error=None,
                           post=lambda f, *a: f(*a), error=Mock())
    path = export_post_weld_ply(host, raw, 2)
    assert path == teaching / "npz" / "pass_2_rescan.ply"
    np.testing.assert_array_equal(read_ply(path)[1], points)
    assert (raw / "scan.npz").read_bytes() == before
    journal.finish("COMPLETED", "done")
    doc = json.loads((teaching / "summary.json").read_text())
    assert doc["artifacts"][-1]["kind"] == "post_weld_ply"
    assert doc["artifacts"][-1]["points"] == len(points)
    manifest = json.loads((teaching / "manifest.json").read_text())
    item = next(f for f in manifest["files"] if f["file"].endswith(".ply"))
    assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not (raw / "scan.ply").exists()  # no second full-size PLY copy


def test_standalone_post_scan_exports_beside_its_npz(tmp_path):
    np.savez_compressed(tmp_path / "scan.npz", points_world_m=np.empty((0, 3)))
    host = SimpleNamespace()
    path = export_post_weld_ply(host, tmp_path, 0)
    assert path == tmp_path / "scan.ply"
    assert "element vertex 0" in read_ply(path)[0]
    with pytest.raises(ValueError, match="No scan.npz"):
        load_scan_cloud(tmp_path / "missing")


def test_ply_write_error_does_not_publish_partial_data(tmp_path, monkeypatch):
    from construct_robot.io import scan_cloud
    path = tmp_path / "scan.ply"
    path.write_bytes(b"previous export")
    monkeypatch.setattr(scan_cloud.os, "replace", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        write_cloud_ply(path, [[1, 2, 3]])
    assert path.read_bytes() == b"previous export" and not list(tmp_path.glob("*.tmp"))
