"""설정에서 읽는 강체 변환 (target ← source, 출처·확인 상태 포함) — KIRO 형식 출력의 World ← base (scan_core.kiro_plan) 와 용접 계획의
토치 모형 (scan_core.weld_torch: 플랜지 ← TCP ← 계획의 끝 틀) 이 같은 형식·검사를 쓴다. 수는 모두 설정에서 (코드에 수 없음).

  quat 형식  {xyz_m: [x, y, z], quat_xyzw: [x, y, z, w]}   (단위 쿼터니언, 허용 = 입력 수치 정밀도 1e-6)
  axes 형식  {xyz_m: [x, y, z], <axes key>: {x: [...], y: [...], z: [...]}}   (source 틀의 축을 target 틀에서 — 오른손 직교, 1e-6)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .g1_types import jsonable

__all__ = ["RigidTransform", "compose", "transform_from_quat", "transform_from_axes"]


def _vec(v, n: int, where: str) -> np.ndarray:
    try:
        a = np.asarray([float(x) for x in v], dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be a list of {n} numbers, got {v!r}") from None
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{where} must be a list of {n} finite numbers, got {v!r}")
    return a


def _nearest_rotation(M: np.ndarray, where: str) -> np.ndarray:
    """입력 회전 (쓴 자릿수의 반올림 포함) → 가장 가까운 회전 (SVD). 직교·오른손이 아니면 ValueError — 허용 = 입력 수치 정밀도
    (행렬 성분의 float32 수준 1e-6)."""
    if not np.all(np.isfinite(M)):
        raise ValueError(f"{where}: not finite")
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0 or np.max(np.abs(M.T @ M - np.eye(3))) > 1e-6:
        raise ValueError(f"{where} is not a rotation (orthonormal, right-handed): {np.round(M, 9).tolist()}")
    return R


def _quat_to_R(q, where: str) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    q = _vec(q, 4, where)
    nq = float(np.linalg.norm(q))
    if abs(nq - 1.0) > 1e-6:
        raise ValueError(f"{where} must be a unit quaternion [x, y, z, w], norm {nq:.9f}")
    return Rotation.from_quat(q / nq).as_matrix()


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()                       # x, y, z, w
    return -q if q[3] < 0.0 else q


@dataclass(frozen=True)
class RigidTransform:
    """target ← source: P_target = R · P_source + t [m]. source·status = 출처·확인 상태 (출력 메모)."""
    R: np.ndarray
    t: np.ndarray
    source: str = ""
    status: str = ""
    base_frame: str = ""            # 이 변환을 쓴 실행의 base 프레임 (기록용 — 설정에는 이름이 없다)

    def apply(self, P) -> np.ndarray:
        return np.asarray(P, dtype=float) @ self.R.T + self.t

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T

    def to_dict(self) -> dict:
        return jsonable({"xyz_m": self.t, "quat_xyzw": _R_to_quat(self.R), "R": self.R, "source": self.source, "status": self.status,
                         "base_frame": self.base_frame})


def compose(a: RigidTransform, b: RigidTransform, source: str = "") -> RigidTransform:
    """a ∘ b (P = a.R (b.R P + b.t) + a.t) — 예: World ← 계획 프레임 ∘ 계획 프레임 ← 계획의 base."""
    return RigidTransform(R=a.R @ b.R, t=a.R @ b.t + a.t, source=source or f"{a.source}; {b.source}",
                          status=a.status, base_frame=b.base_frame)


def transform_from_quat(d: dict, where: str, hint: str = "") -> RigidTransform:
    """{xyz_m, quat_xyzw, source, status} → RigidTransform (알 수 없는 키는 ValueError; hint = 그 메시지 꼬리)."""
    d = dict(d or {})
    unknown = sorted(set(d) - {"xyz_m", "quat_xyzw", "source", "status"})
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}{hint}")
    return RigidTransform(R=_quat_to_R(d.get("quat_xyzw"), f"{where}.quat_xyzw"), t=_vec(d.get("xyz_m"), 3, f"{where}.xyz_m"),
                          source=str(d.get("source") or ""), status=str(d.get("status") or ""))


def transform_from_axes(d: dict, where: str, axes_key: str) -> RigidTransform:
    """{xyz_m, <axes_key>: {x, y, z}, source, status} → RigidTransform (축 = source 틀의 x, y, z 를 target 틀에서; 오른손 직교)."""
    d = dict(d or {})
    unknown = sorted(set(d) - {"xyz_m", axes_key, "source", "status"})
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")
    ax = d.get(axes_key) or {}
    M = np.column_stack([_vec(ax.get(c), 3, f"{where}.{axes_key}.{c}") for c in "xyz"])
    return RigidTransform(R=_nearest_rotation(M, f"{where}.{axes_key}"), t=_vec(d.get("xyz_m"), 3, f"{where}.xyz_m"),
                          source=str(d.get("source") or ""), status=str(d.get("status") or ""))
