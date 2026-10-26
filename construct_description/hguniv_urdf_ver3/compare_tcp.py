#!/usr/bin/env python3
"""Compare right_manipulator_ee_point FK between the KIRO and Hongik v4 URDFs.

Offline only (no ROS graph, no robot motion). For each joint configuration it
prints the TCP pose in World for both models and their difference.

Usage:
  python3 compare_tcp.py                         # all saved right-arm state YAMLs
  python3 compare_tcp.py path/to/state.yaml ...  # specific state YAMLs
  python3 compare_tcp.py --joints j1 j2 j3 j4 j5 j6   # radians
"""

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DESCRIPTION = HERE.parent
URDFS = {
    "kiro": DESCRIPTION / "urdf_0528" / "construct_robot_0528.urdf",
    "hongik_v4": DESCRIPTION / "urdf_0528" / "construct_robot_0528_hongik_v4.urdf",
}
TIP = "right_manipulator_ee_point"
ROOT = "World"
JOINTS = [f"right_manipulator_joint{i}" for i in range(1, 7)]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis, angle):
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _homogeneous(rotation, translation=(0.0, 0.0, 0.0)):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


class Chain:
    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.by_child = {j.find("child").get("link"): j for j in root.findall("joint")}

    def fk(self, positions, tip=TIP):
        transform = np.eye(4)
        link = tip
        while link != ROOT:
            joint = self.by_child[link]
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            local = _homogeneous(_rpy(*rpy), xyz)
            if joint.get("type") in ("revolute", "continuous"):
                axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
                local = local @ _homogeneous(
                    _axis_angle(axis, positions.get(joint.get("name"), 0.0)))
            transform = local @ transform
            link = joint.find("parent").get("link")
        return transform


def _quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _angle_deg(r_a, r_b):
    cos = (np.trace(r_a.T @ r_b) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _fmt(v):
    return "(" + ", ".join(f"{x:+.4f}" for x in v) + ")"


def report(label, positions, chains, recorded=None):
    poses = {name: chain.fk(positions) for name, chain in chains.items()}
    kiro, v2 = poses["kiro"], poses["hongik_v4"]
    print(f"\n[{label}]")
    for name, pose in poses.items():
        print(f"  {name:<10} pos[m]={_fmt(pose[:3, 3])}  tool_z={_fmt(pose[:3, 2])}")
    delta = v2[:3, 3] - kiro[:3, 3]
    print(f"  delta v4-kiro  pos[mm]={_fmt(delta * 1000)} "
          f"|d|={np.linalg.norm(delta) * 1000:.1f} mm  "
          f"rot={_angle_deg(kiro[:3, :3], v2[:3, :3]):.2f} deg")
    if recorded is not None:
        pos = np.array([recorded["position_m"][k] for k in "xyz"])
        quat = [recorded["orientation_xyzw"][k] for k in "xyzw"]
        err = np.linalg.norm(kiro[:3, 3] - pos) * 1000
        rot = _angle_deg(kiro[:3, :3], _quat_xyzw_to_matrix(quat))
        print(f"  check: kiro FK vs recorded tcp_pose_world  {err:.2f} mm, {rot:.3f} deg")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("yaml_files", nargs="*", type=Path)
    parser.add_argument("--joints", nargs=6, type=float, metavar="RAD")
    args = parser.parse_args(argv)
    chains = {name: Chain(path) for name, path in URDFS.items()}

    if args.joints:
        report("--joints", dict(zip(JOINTS, args.joints)), chains)
        return 0
    files = args.yaml_files or sorted((DESCRIPTION / "config").glob("right_manipulator_*state*.yaml"))
    for path in files:
        data = yaml.safe_load(path.read_text())
        state = (data or {}).get("joint_state")
        if not state:
            print(f"\n[{path.name}] skipped (no joint_state)")
            continue
        positions = dict(zip(state["names"], state["positions_rad"]))
        report(path.name, positions, chains, data.get("tcp_pose_world"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
