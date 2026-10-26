"""Offline World-frame point clouds and binary PLY export; never acquires data or moves equipment."""

import json
import os
from pathlib import Path
import tempfile

import numpy as np


def load_scan_cloud(folder):
    folder = Path(folder).expanduser()
    local = folder / "scan.npz"
    if local.is_file():
        with np.load(local, allow_pickle=False) as data:
            if "points_world_m" not in data:
                raise ValueError(f"World point cloud is missing from {local}")
            return np.asarray(data["points_world_m"], dtype=np.float32)
    # Hongik PostWeldScan records raw profiles and timestamped robot poses.
    # Reuse its offline reconstruction (including clock/calibration handling).
    raw = folder / "raw.npz"
    if not raw.is_file():
        raise ValueError(f"No scan.npz or Hongik raw.npz for PLY export: {folder}")
    from scan_l0.autoteach import _load_raw_for_refine
    from scan_l0.config import Settings
    from scan_core.reconstruct import assemble, reconstruct_scan
    frames, robot, job, calib = _load_raw_for_refine(raw)
    if job.base_frame != "World":
        raise ValueError(f"PLY export requires World frame, got {job.base_frame!r}")
    with np.load(raw, allow_pickle=False) as data:
        settings = Settings(json.loads(str(data["config_json"])))
    result = reconstruct_scan(frames, robot, calib, job, fps_nominal=job.profile_fps,
                              z_valid_mm=settings.z_valid_mm)
    return assemble(result, only_cruise=False, only_valid=True, frame="base")[0]


def write_cloud_ply(path, points):
    """Save XYZ float32 in World metres, preserving every finite input point."""
    path = Path(path)
    points = np.asarray(points, dtype="<f4")
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("PLY requires finite World XYZ points with shape (N, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(("ply\nformat binary_little_endian 1.0\ncomment frame World; units metres\n"
                          f"element vertex {len(points)}\nproperty float x\nproperty float y\n"
                          "property float z\nend_header\n").encode("ascii"))
            points.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
