"""Pose of the moving chain by forward kinematics of /joint_states (user approval 2026-10-13) — pure model, no rclpy.

Why: KIRO's weld logs (2026-10-13) show TF (World -> right_manipulator_ee_point) reaching their nodes ~0.8 s late while /joint_states
is fresh (<= 10 ms). The MoveIt adapter's pose stream looked TF up at each JointState header.stamp, so at KIRO it would wait, drop
samples (tf_timeout_s) or lag. Now T_base<-flange of a /joint_states message = FK of THAT message's joint values (matched by name)
with the URDF the running system publishes (robot_description: lab = our launch's robot_state_publisher, KIRO = KIRO's main system),
fixed joints included, chain base_frame -> flange_frame (the URDF parent chain = the chain MoveIt plans with). What the joint values
mean (command echo or measured) is not decided here.

The lab record shows why per-message FK is also the exact value: robot_state_publisher publishes a joint state only when its stamp is
>= the last published stamp + 1 / publish_frequency, so the TF of a skipped message was tf2's interpolation between published neighbours
(lab k30_1_a: 1986 of 5142 messages; up to 0.37 mm from that message's own joints on the stepped joint stream).

Static transforms (base <- world / ground, flange <- EOAT / sensor / optical / torch) may still come from TF: read once, verified static
(every TF edge between the two frames is a /tf_static transform — tf_dynamic_edges) and, where the URDF holds the same path with fixed
joints only, equal to the URDF's (fixed_transform); then cached. Nothing per sample from TF.
"""
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from scan_core.fk_urdf import chain_from_joints, parse_urdf_joints_root
from scan_core.trajectory import fast_fk

__all__ = ["METHOD", "FRAME_ROT_TOL_DEG", "FkPoseModel", "tf_dynamic_edges", "pose_difference"]

METHOD = "fk_from_joint_states"
# The rotation bound of "two sources of the same frame agree" (moveit_adapter: FK of the measured joints vs the pose at the plan start —
# the bound that check has always used; path_check_tol_m is its position bound).
FRAME_ROT_TOL_DEG = 0.05
_FK_TYPES = ("fixed", "revolute", "continuous", "prismatic")


def pose_difference(Ta, Tb) -> tuple:
    """(position difference [m], rotation difference [deg]) of two 4x4 transforms. The angle is atan2(sin, cos) of the relative
    rotation — exact near zero (acos of the trace alone bottoms out at ~1e-6 deg in double precision)."""
    Ta = np.asarray(Ta, dtype=float)
    Tb = np.asarray(Tb, dtype=float)
    R = Ta[:3, :3].T @ Tb[:3, :3]
    s = 0.5 * float(np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), math.degrees(math.atan2(s, c))


class FkPoseModel:
    """The robot_description URDF -> T_base<-flange of joint values given BY NAME (any order, extra joints ignored).

    urdf_text: the URDF XML as published; source: where it came from (run log). ValueError when the URDF cannot be parsed, the
    flange is not below the base in the URDF tree, or the chain holds a joint that /joint_states cannot drive (floating / planar)."""

    def __init__(self, urdf_text: str, base_frame: str, flange_frame: str, source: str):
        try:
            root = ET.fromstring(urdf_text)
        except ET.ParseError as e:
            raise ValueError(f"not a URDF ({e})") from None
        self.joints = parse_urdf_joints_root(root)
        self.chain = chain_from_joints(self.joints, base_frame, flange_frame)
        bad = [f"{j.name} ({j.type})" for j in self.chain.joints if j.type not in _FK_TYPES]
        if bad:
            raise ValueError(f"chain {base_frame} -> {flange_frame} has joints /joint_states cannot drive: {', '.join(bad)}")
        self.fk = fast_fk(self.chain)
        self.joint_names = tuple(self.chain.joint_names)
        self.base_frame = str(base_frame)
        self.flange_frame = str(flange_frame)
        self.source = str(source)
        self.sha256 = hashlib.sha256(urdf_text.encode("utf-8")).hexdigest()
        self._parent_joint = {j.child: j for j in self.joints.values()}
        self.links = {el.get("name") for el in root.findall("link")} | set(self._parent_joint) | {j.parent for j in self.joints.values()}
        self._idx: dict = {}                                   # tuple(message names) -> chain index array, or None (joints missing)

    # ---- per message ----
    def _index(self, names) -> Optional[np.ndarray]:
        key = tuple(names)
        idx = self._idx.get(key, False)
        if idx is False:
            pos = {n: i for i, n in enumerate(key)}
            idx = None if any(n not in pos for n in self.joint_names) else np.array([pos[n] for n in self.joint_names], dtype=int)
            if len(self._idx) < 64:                            # the publishers' name orders (a few); never grows without bound
                self._idx[key] = idx
        return idx

    def chain_q(self, names, positions) -> tuple:
        """(chain-order joint values or None, why None): the message's values of the chain joints, matched by name."""
        idx = self._index(names)
        if idx is None:
            return None, "joints missing"
        p = np.asarray(positions, dtype=float).ravel()
        if p.size != len(names):
            return None, "position count differs from the name count"
        q = p[idx]
        if not np.all(np.isfinite(q)):
            return None, "non-finite joint value"
        return q, ""

    def pose(self, names, positions) -> tuple:
        """(T_base<-flange 4x4 or None, why None) of one /joint_states message."""
        q, why = self.chain_q(names, positions)
        return (None, why) if q is None else (self.fk(q), "")

    def missing_joints(self, names) -> list:
        have = set(names)
        return [n for n in self.joint_names if n not in have]

    # ---- static paths ----
    def _up(self, link: str) -> list:
        """Joints from link up to the URDF root (link's own parent joint first)."""
        out, cur, seen = [], link, set()
        while cur in self._parent_joint and cur not in seen:
            seen.add(cur)
            j = self._parent_joint[cur]
            out.append(j)
            cur = j.parent
        return out

    def fixed_transform(self, target: str, source: str) -> tuple:
        """URDF T_target<-source: (4x4, []) when the URDF path between the two links has fixed joints only; (None, [moving joint names])
        when a movable joint lies on it; (None, []) when a frame is not in this URDF."""
        if target not in self.links or source not in self.links:
            return None, []
        up_t, up_s = self._up(target), self._up(source)
        anc_t = [target] + [j.parent for j in up_t]
        anc_s = [source] + [j.parent for j in up_s]
        common = next((a for a in anc_s if a in set(anc_t)), None)
        if common is None:
            return None, []
        jt = up_t[:anc_t.index(common)]
        js = up_s[:anc_s.index(common)]
        moving = [j.name for j in list(reversed(jt)) + list(reversed(js)) if j.movable]      # common ancestor -> down
        if moving:
            return None, moving

        def down(joints):
            T = np.eye(4)
            for j in reversed(joints):
                T = T @ j.origin()
            return T
        return np.linalg.inv(down(jt)) @ down(js), []

    def record(self) -> dict:
        return {"method": METHOD, "urdf_source": self.source, "urdf_sha256": self.sha256, "base_frame": self.base_frame,
                "flange_frame": self.flange_frame, "chain": [f"{j.name} ({j.type})" for j in self.chain.joints],
                "joint_names": list(self.joint_names)}


def tf_dynamic_edges(frames: dict, target: str, source: str) -> tuple:
    """TF tree (tf2 Buffer.all_frames_as_yaml, parsed: {child: {parent, most_recent_transform, ...}}) -> (edges on the path between
    target and source as [(child, parent)], the ones among them that are NOT static). A /tf_static transform is stored by tf2 without
    a time (most_recent_transform 0); anything on /tf has its stamps. ValueError when the two frames are not connected."""
    frames = frames if isinstance(frames, dict) else {}

    def up(f):
        out, cur, seen = [], str(f), set()
        while cur in frames and cur not in seen:
            seen.add(cur)
            p = str((frames[cur] or {}).get("parent", ""))
            out.append((cur, p))
            cur = p
        return out

    up_t, up_s = up(target), up(source)
    anc_t = [str(target)] + [p for _c, p in up_t]
    anc_s = [str(source)] + [p for _c, p in up_s]
    common = next((a for a in anc_s if a in set(anc_t)), None)
    if common is None:
        raise ValueError(f"frames {target!r} and {source!r} are not connected in TF")
    path = up_t[:anc_t.index(common)] + up_s[:anc_s.index(common)]
    dyn = [(c, p) for c, p in path if float((frames.get(c) or {}).get("most_recent_transform", 0.0) or 0.0) != 0.0]
    return path, dyn
