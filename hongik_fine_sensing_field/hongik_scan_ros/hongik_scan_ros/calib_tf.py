"""EOAT 캘리브 ↔ TF (ROS 없음 — 단위 시험 가능).

캘리브 파일 저장 규약 (설계 §6.2 #3): laser_link → profiler_optical_frame (eoat_yaw 하류) xyz/rpy + eoat_yaw + dt_s + x_sign.

* 정적 TF `<부모> → hongik_profiler_optical_frame` 의 값 (static_tf_from_calib):
    parent_kind 'laser'  : 부모 = 레이저 CAD 프레임 (KIRO laser_link, 연구실 URDF laser_link) → 캘리브 xyz/rpy 그대로.
    parent_kind 'flange' : 부모 = 플랜지 (EOAT 링크가 TF 에 없을 때) →
                           T_flange←opt = Rz(eoat_yaw)·T_eoat←laser(EOAT xacro)·T_laser←opt.
* MoveIt 어댑터 의 Calib (calib_from_tf): T_flange←opt 와 T_flange←eoat 를 **플랫폼 TF** 에서 읽는다 (KIRO URDF 의 eoat_base_link →
  laser_link 와 우리 정적 TF) — 플랫폼마다 다른 eoat_yaw/장착을 코드나 파일에 다시 적지 않는다. dt_s·x_sign·출처는 파일에서.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np

from scan_core.frames import xyz_rpy_from_transform
from scan_core.types import Calib

__all__ = ["static_tf_from_calib", "calib_from_tf", "PARENT_KINDS", "calib_consistency"]

PARENT_KINDS = ("laser", "flange")


def static_tf_from_calib(calib_doc: dict, parent_kind: str = "laser") -> tuple:
    """캘리브 YAML dict → (xyz (3,), rpy (3,)) of parent → optical. parent_kind ∈ PARENT_KINDS."""
    if parent_kind not in PARENT_KINDS:
        raise ValueError(f"parent_kind must be one of {PARENT_KINDS}, got {parent_kind!r}")
    if str(calib_doc.get("parent_frame", "laser_link")) != "laser_link":
        raise ValueError(f"calib file must store laser_link -> optical (got parent {calib_doc.get('parent_frame')!r})")
    xyz = [float(v) for v in calib_doc["xyz"]]
    rpy = [float(v) for v in calib_doc["rpy"]]
    if parent_kind == "laser":
        return np.asarray(xyz, dtype=float), np.asarray(rpy, dtype=float)
    c = Calib.from_laser_link(opt_xyz=tuple(xyz), opt_rpy=tuple(rpy), eoat_yaw_rad=float(calib_doc.get("eoat_yaw", 0.0) or 0.0))
    return xyz_rpy_from_transform(c.T_flange_opt)


def calib_from_tf(base: Calib, T_flange_opt: np.ndarray, T_flange_eoat: np.ndarray, *, note: str = "") -> Calib:
    """파일 캘리브 + 플랫폼 TF (T_flange←opt, T_flange←eoat) → MoveIt 어댑터 의 Calib. TF 가 정하는 것 (두 변환과 eoat_yaw) 과 출처만
    바꾸고, 나머지 필드 (dt_s·x_sign·'g1:' 블록 — 1G 시간 오프셋·σ·edge_response·noise_priors — ·툴 몸체 …) 는 파일 캘리브 그대로
    (dataclasses.replace). 필드를 하나씩 옮겨 적으면 Calib 에 새로 생긴 필드가 실기 경로에서 조용히 빠진다 (2026-10-12 아침 실기:
    g1 noise_priors·edge_response 가 빠져 루트가 확립되지 않고 끝 모형이 센서 응답을 온라인으로 추정했다)."""
    Te = np.asarray(T_flange_eoat, dtype=float)
    yaw = math.atan2(Te[1, 0], Te[0, 0])
    return dataclasses.replace(base, T_flange_opt=np.asarray(T_flange_opt, dtype=float), T_flange_eoat=Te, eoat_yaw_rad=yaw,
                               source=(base.source + "+tf" + (f"({note})" if note else "")).strip("+"))


def calib_consistency(file_calib: Calib, tf_calib: Calib) -> dict:
    """파일 캘리브 (연구실 eoat_yaw·EOAT xacro 상수) 와 TF 캘리브의 차이 (보고용: 다른 플랫폼이면 차이가 정상)."""
    A = np.asarray(file_calib.T_flange_opt, dtype=float)
    B = np.asarray(tf_calib.T_flange_opt, dtype=float)
    dp = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
    R = A[:3, :3].T @ B[:3, :3]
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return {"dp_mm": dp * 1e3, "dr_deg": math.degrees(math.acos(c)),
            "file_eoat_yaw_deg": math.degrees(file_calib.eoat_yaw_rad), "tf_eoat_yaw_deg": math.degrees(tf_calib.eoat_yaw_rad)}


def transform_from_tf_values(xyz, quat_xyzw) -> np.ndarray:
    """TF (translation, quaternion x y z w) → 4x4."""
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([float(v) for v in quat_xyzw]).as_matrix()
    T[:3, 3] = [float(v) for v in xyz]
    return T


def tf_values_from_transform(T) -> tuple:
    """4x4 → (xyz (3,), quat x y z w (4,))."""
    from scipy.spatial.transform import Rotation
    T = np.asarray(T, dtype=float)
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()


def rpy_static_args(xyz, rpy, parent: str, child: str) -> list:
    """tf2_ros static_transform_publisher 인자 (Humble 이름 인자: --x … --roll --pitch --yaw --frame-id --child-frame-id)."""
    return ["--x", repr(float(xyz[0])), "--y", repr(float(xyz[1])), "--z", repr(float(xyz[2])),
            "--roll", repr(float(rpy[0])), "--pitch", repr(float(rpy[1])), "--yaw", repr(float(rpy[2])),
            "--frame-id", str(parent), "--child-frame-id", str(child)]


def check_transform(T: Optional[np.ndarray]) -> bool:
    if T is None:
        return False
    T = np.asarray(T, dtype=float)
    return T.shape == (4, 4) and bool(np.all(np.isfinite(T)))


