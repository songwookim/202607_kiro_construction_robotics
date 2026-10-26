"""Wire-tip (TCP) pivot calibration: the wire tip held on one fixed point while the torch orientation changes.

Every capture i (TCP position p_i, rotation R_i of right_manipulator_ee_point) satisfies p_i + R_i t = P with the
unknown wire tip t in the TCP frame and the unknown point P (World). Stacked: [R_i  -I] [t; P] = -p_i, least squares.
One orientation cannot separate t from P; two observe t except along their relative rotation axis; three or more
about different axes observe all of it (the robot pendant's "4-point TCP").
"""

import numpy as np

from construct_robot.core.scan_reference import quat_matrix

MIN_SAMPLES = 3
MIN_SPREAD_DEG = 15.0


def pivot_calibration(samples):
    """samples: [(position xyz, quat xyzw)] -> {"tip_tcp_m", "point_world_m", "residuals_m", "rms_m",
    "singular_values", "spread_deg"}. Raises ValueError when the orientations do not determine the tip."""
    if len(samples) < MIN_SAMPLES:
        raise ValueError(f"capture at least {MIN_SAMPLES} orientations (have {len(samples)})")
    Rs = [quat_matrix(q) for _p, q in samples]
    ps = [np.asarray(p, float) for p, _q in samples]
    A = np.vstack([np.hstack([R, -np.eye(3)]) for R in Rs])
    b = -np.hstack(ps)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    # observability of t: the rotations relative to the first one (P removed)
    D = np.vstack([R - Rs[0] for R in Rs[1:]])
    sv = np.linalg.svd(D, compute_uv=False)
    spread = max(np.degrees(np.arccos(np.clip((np.trace(Rs[0].T @ R) - 1) / 2, -1, 1))) for R in Rs[1:])
    if spread < MIN_SPREAD_DEG:
        raise ValueError(f"the torch orientations differ by only {spread:.1f} deg: tilt at least {MIN_SPREAD_DEG:.0f} deg")
    if sv[-1] < 0.1 * sv[0]:
        raise ValueError("the orientations turn about one axis only: add a tilt in another direction (sideways / twist)")
    t, P = x[:3], x[3:]
    residuals = np.array([np.linalg.norm(p + R @ t - P) for p, R in zip(ps, Rs)])
    return {"tip_tcp_m": t, "point_world_m": P, "residuals_m": residuals,
            "rms_m": float(np.sqrt(np.mean(residuals ** 2))), "singular_values": sv, "spread_deg": float(spread)}


# Quaternion helpers for the touch calibration targets.


def quat_multiply(a, b):
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def axis_quat(axis, degrees):
    half = np.radians(float(degrees)) / 2.0
    v = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]
    return np.array([v[0] * np.sin(half), v[1] * np.sin(half), v[2] * np.sin(half), np.cos(half)])


def tip_vector(quat_xyzw, tip):
    """World TCP -> wire tip for a TCP-frame vector tip, or a length along the torch axis."""
    tip = np.asarray(tip, float)
    R = quat_matrix(quat_xyzw)
    return R @ tip if tip.shape == (3,) else float(tip) * R[:, 2]


# ---- automatic plane-touch calibration: the wire tip touches one flat surface (Fastech DI0) at several spots
# with the start orientation (-> the plane normal) and at several tilts (-> the tip vector):
#     n . (p_i + R_i t) = d        (p_i, R_i: TCP at the DI0 edge; t: wire tip in the TCP frame; plane n, d)
# One plane observes the tip across the torch well (sin tilt) but its length along the torch weakly (1 - cos tilt):
# large tilts about two axes, plus a second set for averaging. The result reports the per-axis sigma.
DEFAULT_TOUCH_TILT_DEG = 35.0


def touch_tilts(max_tilt_deg=DEFAULT_TOUCH_TILT_DEG):
    """Tool x / y tilts at +-max and +-max/2 (8 touches)."""
    big = float(max_tilt_deg)
    if not 5.0 <= big <= 35.0:
        raise ValueError("touch calibration tilt must be 5..35 deg")
    return tuple((axis, sign * value) for value in (big, big / 2.0) for axis in ("x", "y") for sign in (1.0, -1.0))
TOUCH_SPOTS_MM = ((0.0, 0.0), (12.0, 0.0), (0.0, 12.0))


def touch_calibration_plan(start_position, start_quat, tip_model, max_tilt_deg=DEFAULT_TOUCH_TILT_DEG,
                           spots_mm=TOUCH_SPOTS_MM):
    """Touch targets above the start: [{"kind": "plane"|"tilt", "quat", "tip": modelled tip start point,
    "tcp": approach TCP}] - the modelled tip of every approach is at the start tip height (World XY offsets for the
    plane spots, tool-axis tilts at the first spot)."""
    q0 = np.asarray(start_quat, float) / np.linalg.norm(start_quat)
    tip0 = np.asarray(start_position, float) + tip_vector(q0, tip_model)
    plan = []
    for dx, dy in spots_mm:
        tip = tip0 + np.array([dx, dy, 0.0]) * 1e-3
        plan.append({"kind": "plane", "quat": q0, "tip": tip, "tcp": tip - tip_vector(q0, tip_model),
                     "label": f"plane spot ({dx:+.0f}, {dy:+.0f}) mm"})
    for axis, degrees in touch_tilts(max_tilt_deg):
        q = quat_multiply(q0, axis_quat(axis, degrees))
        q = q / np.linalg.norm(q)
        plan.append({"kind": "tilt", "quat": q, "tip": tip0, "tcp": tip0 - tip_vector(q, tip_model),
                     "label": f"tilt tool {axis} {degrees:+.0f} deg"})
    return plan


def plane_touch_calibration(positions, quats, kinds):
    """Wire tip from plane touches. kinds[i] == "plane": same-orientation touches at different spots (their TCPs
    span a plane parallel to the touched one -> normal n). Then linear least squares for t and d over all touches."""
    P = np.asarray(positions, float)
    Rs = [quat_matrix(q) for q in quats]
    plane = P[[i for i, k in enumerate(kinds) if k == "plane"]]
    if len(plane) < 3:
        raise ValueError("touch at least 3 spots with the start orientation (plane normal)")
    c = plane.mean(axis=0)
    _u, sv_plane, vt = np.linalg.svd(plane - c)
    if sv_plane[1] < 1e-3:
        raise ValueError("the plane spots are on one line")
    normal = vt[2] if vt[2][2] >= 0 else -vt[2]
    A = np.array([np.r_[normal @ R, -1.0] for R in Rs])
    b = np.array([-(normal @ p) for p in P])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[-1] < 0.01 * sv[0]:
        raise ValueError("the tilts do not determine the wire tip: tilt about two different axes")
    residuals = A @ x - b
    dof = max(1, len(b) - 4)
    sigma = np.sqrt(max(float(residuals @ residuals) / dof, 1e-12) * np.diag(np.linalg.inv(A.T @ A)))[:3]
    spread = max(np.degrees(np.arccos(np.clip((np.trace(Rs[0].T @ R) - 1) / 2, -1, 1))) for R in Rs[1:])
    return {"tip_tcp_m": x[:3], "tip_sigma_m": sigma, "plane_normal": normal, "plane_d_m": float(x[3]), "residuals_m": residuals,
            "rms_m": float(np.sqrt(np.mean(residuals ** 2))), "singular_values": sv, "spread_deg": float(spread)}
