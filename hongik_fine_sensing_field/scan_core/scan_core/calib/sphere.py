"""구 기반 핸드아이 T_flange←opt (설계 §5.6.2, Sensors 2024 방식).

프로파일 원호 → 원 피팅 (x_c, z_c, r) → 구 중심 c_opt = (x_c, ±√(R²−r²), z_c)  [mm, 광학 프레임]
정지 자세 i 의 플랜지 포즈 T_i (jnt_ang FK) 에 대해  T_i · T_fo · c_i = C (불변 구 중심)
  1) 선형 LS: 미지수 vec(R_fo) 9 + t_fo 3 + C 3  →  R 을 SVD 로 SO(3) 투영
  2) R 고정 후 (t, C) 선형 LS
  3) scipy least_squares 로 (rotvec, t, C) 비선형 정련
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..frames import make_transform, project_to_so3

__all__ = ["fit_circle", "sphere_center_from_arc", "sphere_center_along_track", "solve_hand_eye", "HandEyeResult"]


def fit_circle(x, z, refine: bool = True, iters: int = 20) -> tuple[float, float, float, float]:
    """2D 원 피팅 (Kåsa 대수 + Gauss–Newton 기하 정련) → (xc, zc, r, rms)."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    ok = np.isfinite(x) & np.isfinite(z)
    x, z = x[ok], z[ok]
    if x.size < 3:
        raise ValueError("need >= 3 points for circle fit")
    A = np.column_stack([2 * x, 2 * z, np.ones_like(x)])
    b = x ** 2 + z ** 2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, zc = float(sol[0]), float(sol[1])
    r = float(np.sqrt(max(sol[2] + xc ** 2 + zc ** 2, 0.0)))
    if refine:
        p = np.array([xc, zc, r])
        for _ in range(iters):
            dx, dz = x - p[0], z - p[1]
            d = np.sqrt(dx ** 2 + dz ** 2)
            d = np.where(d < 1e-12, 1e-12, d)
            res = d - p[2]
            J = np.column_stack([-dx / d, -dz / d, -np.ones_like(d)])
            step, _, _, _ = np.linalg.lstsq(J, -res, rcond=None)
            p = p + step
            if np.linalg.norm(step) < 1e-12:
                break
        xc, zc, r = (float(v) for v in p)
    d = np.sqrt((x - xc) ** 2 + (z - zc) ** 2)
    rms = float(np.sqrt(np.mean((d - r) ** 2)))
    return xc, zc, r, rms


def sphere_center_from_arc(x_mm, z_mm, radius_mm: float, y_sign: int = +1) -> tuple[np.ndarray, dict]:
    """원호 → 구 중심 (x_c, y_sign·√(R²−r²), z_c) [mm, 광학 프레임]. 반환 (c, info)."""
    xc, zc, r, rms = fit_circle(x_mm, z_mm)
    R = float(radius_mm)
    if r > R:
        y = 0.0
        clipped = True
    else:
        y = float(np.sqrt(R * R - r * r))
        clipped = False
    c = np.array([xc, float(np.sign(y_sign)) * y, zc])
    return c, {"xc": xc, "zc": zc, "r": r, "rms": rms, "clipped": clipped}


def sphere_center_along_track(s_m, r_mm, radius_mm: float) -> tuple[float, float]:
    """프로파일별 원호 반지름 r_i 와 진행좌표 s_i → r² = R² − (s − s_c)² 포물선 피팅 → (s_c [m], R_est [mm])."""
    s = np.asarray(s_m, dtype=float) * 1e3
    r = np.asarray(r_mm, dtype=float)
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]
    if s.size < 3:
        return float("nan"), float("nan")
    # r² = -s² + 2 s_c s + (R² − s_c²)  → 선형 in (s_c, K)
    A = np.column_stack([2 * s, np.ones_like(s)])
    b = r ** 2 + s ** 2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    s_c = float(sol[0])
    R2 = float(sol[1]) + s_c ** 2
    return s_c * 1e-3, float(np.sqrt(max(R2, 0.0)))


@dataclass
class HandEyeResult:
    T_flange_opt: np.ndarray
    C_base: np.ndarray            # 구 중심 (base, m)
    rms_m: float
    residuals_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_poses: int = 0
    refined: bool = False

    def to_dict(self) -> dict:
        return {"T_flange_opt": self.T_flange_opt.tolist(), "C_base": self.C_base.tolist(), "rms_m": self.rms_m,
                "n_poses": self.n_poses, "refined": self.refined}


def _residuals(params: np.ndarray, Rs: np.ndarray, ps: np.ndarray, cs: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(params[:3]).as_matrix()
    t = params[3:6]
    C = params[6:9]
    pred = np.einsum("nij,nj->ni", Rs, cs @ R.T + t) + ps
    return (pred - C).reshape(-1)


def solve_hand_eye(T_base_flange: Sequence[np.ndarray], c_opt_m: Sequence[np.ndarray], refine: bool = True,
                   T_init: Optional[np.ndarray] = None) -> HandEyeResult:
    """정지 자세 플랜지 포즈 T_i (m) + 광학프레임 구 중심 c_i (m) → T_flange←opt.
    자세에 회전 변화가 없으면 (t, C) 가 분리되지 않으므로 병진+회전 자세를 섞는다 (≥ 5 자세, 권장 20)."""
    Ts = np.asarray(T_base_flange, dtype=float).reshape(-1, 4, 4)
    cs = np.asarray(c_opt_m, dtype=float).reshape(-1, 3)
    n = Ts.shape[0]
    if n < 5 or cs.shape[0] != n:
        raise ValueError("need >= 5 poses with matching sphere centers")
    Rs, ps = Ts[:, :3, :3], Ts[:, :3, 3]
    I3 = np.eye(3)
    if T_init is None:
        # 1) 선형: R_i (c_iᵀ ⊗ I) r9 + R_i t − C = −p_i
        A = np.zeros((3 * n, 15))
        b = np.zeros(3 * n)
        for i in range(n):
            Kc = np.hstack([cs[i, j] * I3 for j in range(3)])        # (3,9): R c = Kc·vec_col(R)
            A[3 * i: 3 * i + 3, 0:9] = Rs[i] @ Kc
            A[3 * i: 3 * i + 3, 9:12] = Rs[i]
            A[3 * i: 3 * i + 3, 12:15] = -I3
            b[3 * i: 3 * i + 3] = -ps[i]
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        R_lin = sol[:9].reshape(3, 3, order="F")
        R0 = project_to_so3(R_lin)
    else:
        R0 = np.asarray(T_init, dtype=float)[:3, :3]
    # 2) R 고정, (t, C) 선형
    A2 = np.zeros((3 * n, 6))
    b2 = np.zeros(3 * n)
    for i in range(n):
        A2[3 * i: 3 * i + 3, 0:3] = Rs[i]
        A2[3 * i: 3 * i + 3, 3:6] = -I3
        b2[3 * i: 3 * i + 3] = -(Rs[i] @ (R0 @ cs[i]) + ps[i])
    sol2, _, _, _ = np.linalg.lstsq(A2, b2, rcond=None)
    t0, C0 = sol2[:3], sol2[3:]
    refined = False
    R, t, C = R0, t0, C0
    if refine:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        x0 = np.concatenate([Rotation.from_matrix(R0).as_rotvec(), t0, C0])
        res = least_squares(_residuals, x0, args=(Rs, ps, cs), method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14,
                            max_nfev=2000)
        R = Rotation.from_rotvec(res.x[:3]).as_matrix()
        t, C = res.x[3:6], res.x[6:9]
        refined = True
    T = make_transform(R, t)
    r = _residuals(np.concatenate([_rotvec(R), t, C]), Rs, ps, cs).reshape(-1, 3)
    rms = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
    return HandEyeResult(T_flange_opt=T, C_base=np.asarray(C, dtype=float), rms_m=rms,
                         residuals_m=np.linalg.norm(r, axis=1), n_poses=n, refined=refined)


def _rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()
