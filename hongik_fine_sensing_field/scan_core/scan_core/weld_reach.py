"""A necessary reach condition of the arm for the torch-orientation search (weld_orient) - exact, from the robot's own kinematic
chain (URDF): it only rules out orientations that no IK solution can reach within the joint limits, so the search does not walk
through them one robot check at a time.

For a 6R arm whose last three axes meet in one point (a spherical wrist: the wrist centre W, fixed in the flange frame), whose
2nd and 3rd axes are parallel and whose 1st axis meets the 2nd at right angles in a point S (the RB series), the 3rd joint (the
elbow) is fixed by the distance of W from the 2nd axis alone: rho^2 = |W - S|^2 - d^2 (d = W's constant offset along the 2nd axis),
rho^2 = L1^2 + L2^2 + 2 L1 L2 cos(psi0 + s q3) (L1 = distance between the 2nd and 3rd axes, L2 = distance of W from the 3rd axis,
psi0 / s from the zero pose). The elbow's URDF limits [lo, hi] give the admissible range of cos(psi0 + s q3), hence of rho - every
IK branch has the same rho, so a pose whose W lies outside that shell has no solution within the elbow limits (whatever the other
joints do). Any other chain structure: no model (the search then relies on the robot check alone).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["ArmReach", "arm_reach_from_chain"]

_TOL_M = 1e-6


@dataclass(frozen=True)
class ArmReach:
    S: np.ndarray                 # (3,) base frame: the 1st axis meets the 2nd here
    d: float                      # W's offset along the 2nd axis [m]
    rho2_lo: float                # admissible rho^2 [m^2] (elbow limits)
    rho2_hi: float
    W_f: np.ndarray               # (3,) the wrist centre in the chain's tip frame
    T_tip_tcp: np.ndarray         # (4, 4) chain tip <- weld torch TCP
    joint: str
    source: str

    def _rho2(self, R_tcp: np.ndarray, P_tcp: np.ndarray) -> np.ndarray:
        """The wrist centre's squared distance from the 2nd axis [m^2] at TCP poses (base: R (N, 3, 3), P (N, 3) [m])."""
        R_tcp = np.asarray(R_tcp, dtype=float).reshape(-1, 3, 3)
        P_tcp = np.asarray(P_tcp, dtype=float).reshape(-1, 3)
        T_tcp_tip = np.linalg.inv(self.T_tip_tcp)
        R_tip = R_tcp @ T_tcp_tip[:3, :3]
        p_tip = P_tcp + R_tcp @ T_tcp_tip[:3, 3]
        W = p_tip + R_tip @ self.W_f
        return np.sum((W - self.S) ** 2, axis=1) - self.d ** 2

    def ok(self, R_tcp: np.ndarray, P_tcp: np.ndarray) -> np.ndarray:
        """TCP poses (base: R (N, 3, 3), P (N, 3) [m]) → (N,) bool: the elbow can reach each (necessary condition)."""
        rho2 = self._rho2(R_tcp, P_tcp)
        return (rho2 >= self.rho2_lo - 1e-12) & (rho2 <= self.rho2_hi + 1e-12)

    def violation(self, R_tcp: np.ndarray, P_tcp: np.ndarray) -> Optional[dict]:
        """The first of the TCP poses (as ok) the elbow cannot reach → {index, rho_mm, lo_mm, hi_mm, side ('beyond the outer shell' |
        'inside the inner shell'), joint} (the wrist centre's distance from the 2nd axis against the elbow limits' shell), None if all."""
        rho2 = self._rho2(R_tcp, P_tcp)
        bad = np.flatnonzero((rho2 < self.rho2_lo - 1e-12) | (rho2 > self.rho2_hi + 1e-12))
        if not bad.size:
            return None
        i = int(bad[0])
        return {"index": i, "rho_mm": math.sqrt(max(float(rho2[i]), 0.0)) * 1e3, "lo_mm": math.sqrt(max(self.rho2_lo, 0.0)) * 1e3,
                "hi_mm": math.sqrt(max(self.rho2_hi, 0.0)) * 1e3, "joint": self.joint,
                "side": "beyond the outer shell" if rho2[i] > self.rho2_hi else "inside the inner shell"}

    def describe(self) -> str:
        return (f"{self.joint} limits as a wrist-centre shell: {math.sqrt(max(self.rho2_lo, 0.0)) * 1e3:.0f} .. "
                f"{math.sqrt(self.rho2_hi) * 1e3:.0f} mm from the 2nd axis ({self.source})")


def _line_closest(p1, a1, p2, a2) -> tuple:
    """Closest points of two lines (point, unit direction) → (c1, c2, distance)."""
    w = p1 - p2
    b = float(a1 @ a2)
    den = 1.0 - b * b
    if den < 1e-12:                                          # parallel
        t1 = 0.0
        t2 = float(w @ a2)
    else:
        t1 = (b * float(w @ a2) - float(w @ a1)) / den
        t2 = (float(w @ a2) - b * float(w @ a1)) / den
    c1, c2 = p1 + t1 * a1, p2 + t2 * a2
    return c1, c2, float(np.linalg.norm(c1 - c2))


def _axes(chain) -> list:
    """(point, unit axis) of every movable joint at the zero pose, base frame."""
    T = np.eye(4)
    out = []
    for j in chain.joints:
        T = T @ j.origin()
        if j.movable:
            a = T[:3, :3] @ np.asarray(j.axis, dtype=float)
            out.append((T[:3, 3].copy(), a / np.linalg.norm(a), j))
            T = T @ j.motion(0.0)
    return out


def arm_reach_from_chain(chain, T_tip_tcp, source: str = "") -> Optional[ArmReach]:
    """The model (module docstring) of a kinematic chain (scan_core.fk_urdf.KinematicChain, base → tip) and the weld torch TCP on its
    tip (tip ← TCP); None when the chain is not of that structure."""
    ax = _axes(chain)
    if len(ax) != 6 or any(j.type not in ("revolute", "continuous") for _p, _a, j in ax):
        return None
    (p1, a1, j1), (p2, a2, j2), (p3, a3, j3), (p4, a4, _j4), (p5, a5, _j5), (p6, a6, _j6) = ax
    c45, c54, d45 = _line_closest(p4, a4, p5, a5)
    c56, c65, d56 = _line_closest(p5, a5, p6, a6)
    if d45 > _TOL_M or d56 > _TOL_M or float(np.linalg.norm(c45 - c56)) > _TOL_M:
        return None                                            # no spherical wrist
    W0 = (c45 + c54 + c56 + c65) / 4.0
    if float(np.linalg.norm(np.cross(a2, a3))) > 1e-9 or abs(float(a1 @ a2)) > 1e-9:
        return None
    S, S2, d12 = _line_closest(p1, a1, p2, a2)
    if d12 > _TOL_M:
        return None
    d = float((W0 - S) @ a2)
    v23 = (p3 - p2) - float((p3 - p2) @ a2) * a2              # axis 2 → axis 3, perpendicular
    L1 = float(np.linalg.norm(v23))
    v3w = (W0 - p3) - float((W0 - p3) @ a3) * a3              # axis 3 → W, perpendicular
    L2 = float(np.linalg.norm(v3w))
    if L1 < _TOL_M or L2 < _TOL_M:
        return None
    psi0 = math.atan2(float(np.cross(v23, v3w) @ a3), float(v23 @ v3w))     # the angle from v23 to v3w about a3
    s = 1.0                                                    # q3 turns v3w about a3 (right hand) by +q3
    lo, hi = float(j3.lower), float(j3.upper)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo >= 2.0 * math.pi:
        c_lo, c_hi = -1.0, 1.0
    else:
        t0, t1 = psi0 + s * lo, psi0 + s * hi
        a, b = min(t0, t1), max(t0, t1)
        ends = [math.cos(a), math.cos(b)]
        c_hi = 1.0 if math.floor(b / (2 * math.pi)) >= math.ceil(a / (2 * math.pi)) else max(ends)
        c_lo = -1.0 if math.floor((b - math.pi) / (2 * math.pi)) >= math.ceil((a - math.pi) / (2 * math.pi)) else min(ends)
    T0 = chain.fk(np.zeros(6))
    W_f = (np.linalg.inv(T0) @ np.append(W0, 1.0))[:3]
    return ArmReach(S=S, d=d, rho2_lo=L1 * L1 + L2 * L2 + 2 * L1 * L2 * c_lo, rho2_hi=L1 * L1 + L2 * L2 + 2 * L1 * L2 * c_hi,
                    W_f=W_f, T_tip_tcp=np.asarray(T_tip_tcp, dtype=float), joint=j3.name,
                    source=source or f"{chain.base_frame} -> {chain.tip_frame} (URDF), elbow {j3.name} [{math.degrees(lo):.0f}, "
                                     f"{math.degrees(hi):.0f}] deg")
