"""Test helper: a pose written as [x, y, z mm, rx, ry, rz deg] (R = Rz(rz) . Ry(ry) . Rx(rx), the ROS RPY convention) -> 4x4 [m]."""
import numpy as np
from scipy.spatial.transform import Rotation


def from_mm_deg(pose_mm_deg) -> np.ndarray:
    p = np.asarray(pose_mm_deg, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", p[3:], degrees=True).as_matrix()
    T[:3, 3] = p[:3] * 1e-3
    return T
