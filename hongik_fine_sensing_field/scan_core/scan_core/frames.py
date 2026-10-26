"""4x4 동차변환 · RPY 유틸.

규약 (URDF / ROS / Rainbow CB 공통):
    R = Rz(yaw) · Ry(pitch) · Rx(roll)      # roll=rx, pitch=ry, yaw=rz
CB `move_l(pnt[x,y,z,rx,ry,rz])` 의 Euler Z-Y'-X'' 도 같은 행렬이므로
`xyz_rpy_from_transform` 결과를 (m→mm, rad→deg) 만 바꿔 그대로 넣는다
(이 패키지는 로봇 제어기 명령을 만들지 않는다).

scipy 검증: Rotation.from_euler('xyz', [roll, pitch, yaw]) (extrinsic) 와 동일.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

__all__ = [
    "rot_x", "rot_y", "rot_z", "rpy_to_matrix", "matrix_to_rpy",
    "make_transform", "transform_from_xyz_rpy", "xyz_rpy_from_transform",
    "inv_transform", "compose", "transform_points", "transform_vectors",
    "rotation_angle", "rotation_error", "project_to_so3", "unit", "angle_between",
    "slerp_transform", "interpolate_transforms", "is_transform",
]


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw)·Ry(pitch)·Rx(roll)  [rad]."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """역변환 (roll, pitch, yaw) [rad]. pitch ∈ [-π/2, π/2]; 짐벌락에서는 roll=0 으로 둔다."""
    R = np.asarray(R, dtype=float)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, float(sy)))
    pitch = math.asin(sy)
    if abs(math.cos(pitch)) < 1e-9:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def make_transform(R: np.ndarray | None = None, t: Iterable[float] | None = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    if t is not None:
        T[:3, 3] = np.asarray(list(t), dtype=float).reshape(3)
    return T


def transform_from_xyz_rpy(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    """URDF `<origin xyz rpy>` 와 같은 의미 (m, rad)."""
    r, p, y = [float(v) for v in rpy]
    return make_transform(rpy_to_matrix(r, p, y), xyz)


def xyz_rpy_from_transform(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=float)
    return T[:3, 3].copy(), np.array(matrix_to_rpy(T[:3, :3]))


def inv_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def compose(*Ts: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    for T in Ts:
        out = out @ np.asarray(T, dtype=float)
    return out


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """P (...,3) → T·P (...,3)."""
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    return P @ T[:3, :3].T + T[:3, 3]


def transform_vectors(T: np.ndarray, V: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    return np.asarray(V, dtype=float) @ T[:3, :3].T


def rotation_angle(R: np.ndarray) -> float:
    """회전행렬의 회전각 [rad]."""
    tr = float(np.trace(np.asarray(R, dtype=float)[:3, :3]))
    return math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))


def rotation_error(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """두 회전 사이 각 [rad]."""
    Ra = np.asarray(Ra, dtype=float)[:3, :3]
    Rb = np.asarray(Rb, dtype=float)[:3, :3]
    return rotation_angle(Ra.T @ Rb)


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """SVD 로 가장 가까운 회전행렬 (det=+1)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt


def unit(v: Iterable[float]) -> np.ndarray:
    v = np.asarray(list(v) if not isinstance(v, np.ndarray) else v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def angle_between(a: Iterable[float], b: Iterable[float]) -> float:
    """두 벡터 사이 각 [rad]."""
    ua, ub = unit(a), unit(b)
    return math.acos(max(-1.0, min(1.0, float(np.dot(ua, ub)))))


def is_transform(T: np.ndarray, tol: float = 1e-6) -> bool:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    return (np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol
            and np.allclose(T[3], [0, 0, 0, 1], atol=tol))


def slerp_transform(T0: np.ndarray, T1: np.ndarray, w: float) -> np.ndarray:
    """위치 선형 · 회전 slerp 보간 (w ∈ [0,1])."""
    from scipy.spatial.transform import Rotation, Slerp

    T0 = np.asarray(T0, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    w = float(w)
    key = Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    R = Slerp([0.0, 1.0], key)(w).as_matrix()
    t = (1.0 - w) * T0[:3, 3] + w * T1[:3, 3]
    return make_transform(R, t)


def interpolate_transforms(T0: np.ndarray, T1: np.ndarray, n: int) -> np.ndarray:
    """T0→T1 을 n 개(양끝 포함)로 보간. 반환 (n,4,4)."""
    from scipy.spatial.transform import Rotation, Slerp

    n = max(2, int(n))
    T0 = np.asarray(T0, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    ws = np.linspace(0.0, 1.0, n)
    key = Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    Rs = Slerp([0.0, 1.0], key)(ws).as_matrix()
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = Rs
    out[:, :3, 3] = (1.0 - ws)[:, None] * T0[:3, 3] + ws[:, None] * T1[:3, 3]
    return out
