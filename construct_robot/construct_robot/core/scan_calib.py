"""Profiler hand-eye (laser_link -> optical frame) math for recorded scans (pure numpy, no ROS).

A recording stores T_world_optical per frame = T_world_laser(robot FK at the frame stamp) @ T_laser_optical(calib file
in use). Changing the calib therefore only replaces the right-hand factor:

    T_world_optical_new = T_world_optical_recorded @ inv(T_laser_optical_recorded) @ T_laser_optical_new

The hand-eye translation fix comes from measured wire-tip positions (robot TCP model) and the same points in the scan:
during one constant-orientation scan, a World offset D between them is the constant laser-frame translation
R_world_laser^T D (the calib xyz is expressed in laser_link).
"""

import math

import numpy as np


def rpy_matrix(rpy):
    """URDF / ROS fixed-axis roll-pitch-yaw -> 3x3 (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def calib_matrix(xyz, rpy):
    """laser_link -> optical 4x4 of a calib file's xyz [m] / rpy [rad]."""
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(rpy)
    T[:3, 3] = [float(v) for v in xyz]
    return T


def same_calib(a, b, tol=1e-9):
    """a, b: {"xyz", "rpy"} (None = unknown -> not comparable -> True)."""
    if a is None or b is None:
        return True
    return all(abs(float(x) - float(y)) <= tol for key in ("xyz", "rpy") for x, y in zip(a[key], b[key]))


def rebase_optical(T_world_optical, T_laser_optical_recorded, T_laser_optical_new):
    """(N, 4, 4) recorded optical poses -> the poses under another calib (same robot FK)."""
    return np.asarray(T_world_optical, float) @ np.linalg.inv(T_laser_optical_recorded) @ T_laser_optical_new


def rebase_points(points, frame_index, T_world_optical_old, T_world_optical_new):
    """World points of a recording (points_world_m, points_frame_index) under the new optical poses."""
    D = np.asarray(T_world_optical_new, float) @ np.linalg.inv(np.asarray(T_world_optical_old, float))
    Di = D[np.asarray(frame_index, int)]
    p = np.asarray(points, float)
    return np.einsum("nij,nj->ni", Di[:, :3, :3], p) + Di[:, :3, 3]


def mean_rotation(rotations):
    """Chordal mean of (N, 3, 3) rotations."""
    U, _s, Vt = np.linalg.svd(np.mean(np.asarray(rotations, float), axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def laser_rotation(T_world_optical, T_laser_optical):
    """Mean World <- laser_link rotation of a recording and its spread [deg] (a constant-orientation scan: ~0)."""
    R_wl = np.asarray(T_world_optical, float)[:, :3, :3] @ np.asarray(T_laser_optical, float)[:3, :3].T
    R = mean_rotation(R_wl)
    cos = np.clip((np.einsum("ij,nij->n", R, R_wl) - 1.0) / 2.0, -1.0, 1.0)
    return R, float(np.degrees(np.arccos(cos)).max()) if len(R_wl) else 0.0


def handeye_translation_fix(samples):
    """samples: [(R_world_laser (3x3), D_world (3,) = measured point - same point in the scan)] -> laser-frame
    translation fix {"per_sample_m", "mean_m", "std_m"} (add mean_m to the calib xyz)."""
    if not samples:
        raise ValueError("no hand-eye samples")
    per = np.array([np.asarray(R, float).T @ np.asarray(D, float) for R, D in samples])
    return {"per_sample_m": per, "mean_m": per.mean(axis=0), "std_m": per.std(axis=0)}
