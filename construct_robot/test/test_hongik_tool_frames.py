"""Tool-frame offsets in the Hongik robot model, without robot motion."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _zero_joint_pose(joints, link):
    if link not in joints:
        return np.eye(4)
    joint = joints[link]
    origin = joint.find("origin")
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0., 0., 0.]
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0., 0., 0.]
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    transform = np.eye(4)
    transform[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    transform[:3, 3] = xyz
    return _zero_joint_pose(joints, joint.find("parent").get("link")) @ transform


def test_hongik_weld_reference_is_20mm_on_tcp_local_z():
    path = Path(__file__).parents[2] / "construct_description/urdf_0528/construct_robot_0528_hongik_v4.urdf"
    robot = ET.parse(path).getroot()
    joints = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
    tcp = _zero_joint_pose(joints, "right_manipulator_ee_point")
    weld = _zero_joint_pose(joints, "hongik_weld_link")
    tcp_from_weld = np.linalg.solve(tcp, weld)
    expected = np.eye(4)
    expected[2, 3] = 0.020
    np.testing.assert_allclose(tcp_from_weld, expected, atol=1e-12)
    # The offset follows the rotated TCP axis, rather than the world's Z axis.
    np.testing.assert_allclose(weld[:3, 3] - tcp[:3, 3], tcp[:3, 2] * 0.020, atol=1e-12)
    assert joints["hongik_weld_link"].get("type") == "fixed"
