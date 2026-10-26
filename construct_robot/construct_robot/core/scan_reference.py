"""Correct Hongik-estimated pass endpoints with a measured root-pass reference (pure math, no ROS).

Positions are compared at the wire tip (aim point), not at the TCP: the TCP sits L = tcp_to_aim behind the wire tip
along the torch axis, so two TCPs with different torch orientations differ by up to L even for the same aim point
(Hongik's root pass is tilted ~42 deg; an operator's measured torch may be vertical).

    aim = tcp + L * z_tcp          tcp = aim - L * z_tcp          (z_tcp = torch axis toward the work)

The offsets (reference aim - estimated aim) are expressed in the seam frame of the estimate they were measured on
(u = S -> E, n = plate normal, w = n x u, as Hongik's section frame), one per end (start / goal), so the same
correction can be applied to the passes of this estimate and carried to a later estimate of the same setup.
If the execution keeps the reference torch orientation, the pass-1 TCPs reproduce the measured TCPs exactly, and a
wrong stick-out cancels (it is in both the reference conversion and the execution conversion).
"""

from dataclasses import dataclass
import math

import numpy as np

ENDS = ("start", "goal")


def unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 0.0 or not math.isfinite(n):
        raise ValueError("zero / invalid vector")
    return v / n


def torch_axis(quat_xyzw):
    """z axis of a TCP quaternion (x, y, z, w) = the torch axis toward the work."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    q = math.sqrt(x * x + y * y + z * z + w * w)
    if q < 1e-9:
        raise ValueError("zero quaternion")
    x, y, z, w = x / q, y / q, z / q, w / q
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def quat_matrix(quat_xyzw):
    x, y, z, w = (float(v) for v in quat_xyzw)
    q = math.sqrt(x * x + y * y + z * z + w * w)
    if q < 1e-9:
        raise ValueError("zero quaternion")
    x, y, z, w = x / q, y / q, z / q, w / q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wire_tip_vector(quat_xyzw, L):
    """World vector TCP -> wire tip. L: a length along the torch axis (z), or a TCP-frame 3-vector (calibrated)."""
    if np.ndim(L) == 0:
        return float(L) * torch_axis(quat_xyzw)
    return quat_matrix(quat_xyzw) @ np.asarray(L, float).reshape(3)


def aim_from_tcp(tcp, quat_xyzw, L):
    return np.asarray(tcp, float) + wire_tip_vector(quat_xyzw, L)


def tcp_from_aim(aim, quat_xyzw, L):
    return np.asarray(aim, float) - wire_tip_vector(quat_xyzw, L)


def plate_weave_direction(start, goal, plate_normal):
    """Unit World direction across the actual seam and inside the scanned plate plane, independent of tool roll."""
    direction = unit(np.asarray(goal, float) - np.asarray(start, float))
    normal = unit(plate_normal)
    return unit(np.cross(normal, direction))


@dataclass(frozen=True)
class SeamFrame:
    u: np.ndarray       # along the seam, S -> E
    w: np.ndarray       # lateral, n x u (+ = left looking S -> E)
    n: np.ndarray       # plate normal (torch side)

    @classmethod
    def from_estimate(cls, u, n_plate):
        u = unit(u)
        n = np.asarray(n_plate, float)
        n = unit(n - float(n @ u) * u)        # exactly orthogonal to u
        if n[2] < 0.0:                       # the torch side of a 1G plate is up
            n = -n
        return cls(u=u, w=unit(np.cross(n, u)), n=n)

    def to_local(self, v):
        v = np.asarray(v, float)
        return np.array([v @ self.u, v @ self.w, v @ self.n])

    def to_world(self, local):
        a, b, c = (float(x) for x in local)
        return a * self.u + b * self.w + c * self.n


def torch_angles(quat_xyzw, frame: SeamFrame, travel_sign=1.0, bevel_sign_w=1):
    """(travel, work) [deg] of a torch in the seam frame, Hongik's (AWS A3.0) definitions (kiro_plan._angles):
    travel = asin(a . d), d = the travel direction (+ push / - drag); work = atan2(a . b, -a . n), b = toward the bevel
    face (bevel_sign_w: +1 bevel plate on +w = left looking S -> E, -1 on the right)."""
    a = torch_axis(quat_xyzw)
    d = frame.u * float(travel_sign)
    b = frame.w * (1.0 if int(bevel_sign_w) >= 0 else -1.0)
    travel = math.degrees(math.asin(max(-1.0, min(1.0, float(a @ d)))))
    work = math.degrees(math.atan2(float(a @ b), float(-a @ frame.n)))
    return travel, work


def reference_offsets(reference, pass1, frame: SeamFrame, L):
    """reference: {end: (tcp xyz, quat xyzw)} measured; pass1: {"aim_start", "aim_goal"} estimated aim points.
    Returns {end: {"world_m", "seam_m" (along, lateral, normal), "reference_aim_m"}}."""
    out = {}
    for end in ENDS:
        tcp, quat = reference[end]
        ref_aim = aim_from_tcp(tcp, quat, L)
        delta = ref_aim - np.asarray(pass1[f"aim_{end}"], float)
        out[end] = {"world_m": delta, "seam_m": frame.to_local(delta), "reference_aim_m": ref_aim}
    return out


def corrected_aim(aim, end, seam_offsets, frame: SeamFrame):
    """aim + the end's offset (stored in seam coordinates) expressed in this estimate's seam frame."""
    return np.asarray(aim, float) + frame.to_world(seam_offsets[end])


def offset_report(offsets):
    """Operator text: per end the world and seam-frame offsets [mm], and the start / goal disagreement."""
    lines = []
    for end in ENDS:
        w = offsets[end]["world_m"] * 1e3
        s = offsets[end]["seam_m"] * 1e3
        lines.append(f"{end}: world ({w[0]:+.1f}, {w[1]:+.1f}, {w[2]:+.1f}) mm = along {s[0]:+.1f} / lateral {s[1]:+.1f} "
                     f"/ normal {s[2]:+.1f} mm, |{np.linalg.norm(w):.1f}| mm")
    diff = (offsets["goal"]["seam_m"] - offsets["start"]["seam_m"]) * 1e3
    lines.append(f"goal - start: along {diff[0]:+.1f} / lateral {diff[1]:+.1f} / normal {diff[2]:+.1f} mm "
                 "(lateral / normal: seam direction error; along: end position / length difference)")
    return lines
