"""Commanded-spline TCP fidelity and the IK branch-change guard."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from construct_robot.core.keyboard_jog_kinematics import ArmChain
from construct_robot.core.path_fidelity import (
    _matrix_quaternion,
    spline_samples,
    tcp_path_deviation,
)
from construct_robot.nodes.robot_model import verify_path_fidelity

URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "construct_description" / "urdf_0528" / "construct_robot_0528_hongik_v4.urdf"
)
Q_RIGHT = np.array([5.1067, 1.0121, 2.0954, 3.5405, 1.1626, 1.4709])
NAMES = [f"right_manipulator_joint{i}" for i in range(1, 7)]


@pytest.fixture(scope="module")
def chain():
    return ArmChain(URDF_PATH.read_text(encoding="utf-8"), "World", "right_manipulator_ee_point")


def trajectory(times, positions, velocities=None, accelerations=None, names=NAMES):
    result = RobotTrajectory()
    result.joint_trajectory.joint_names = list(names)
    for index, (t, q) in enumerate(zip(times, positions)):
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q]
        if velocities is not None:
            point.velocities = [float(v) for v in velocities[index]]
        if accelerations is not None:
            point.accelerations = [float(v) for v in accelerations[index]]
        point.time_from_start = Duration(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))
        result.joint_trajectory.points.append(point)
    return result


def pose(position, rotation):
    result = Pose()
    result.position.x, result.position.y, result.position.z = (float(v) for v in position)
    (result.orientation.x, result.orientation.y,
     result.orientation.z, result.orientation.w) = (float(v) for v in _matrix_quaternion(rotation))
    return result


def straight_line_joints(chain, q, delta, steps):
    """Newton IK onto evenly spaced poses of a straight TCP line, fixed attitude."""
    joints = [np.array(q, dtype=float)]
    start_position, target_rotation = chain.fk(q)
    for step in range(1, steps + 1):
        target = start_position + np.asarray(delta) * step / steps
        current = joints[-1].copy()
        for _ in range(5):
            position, rotation = chain.fk(current)
            error_rotation = target_rotation @ rotation.T
            angle_axis = 0.5 * np.array([
                error_rotation[2, 1] - error_rotation[1, 2],
                error_rotation[0, 2] - error_rotation[2, 0],
                error_rotation[1, 0] - error_rotation[0, 1],
            ])
            twist = np.concatenate([target - position, angle_axis])
            current = current + np.linalg.pinv(chain.jacobian(current)) @ twist
        joints.append(current)
    return np.array(joints)


def test_spline_matches_the_controller_quintic():
    path = trajectory([0.0, 1.0], [[0.0] * 6, [1.0] * 6],
                      velocities=[[0.0] * 6] * 2, accelerations=[[0.0] * 6] * 2)
    times, joints = spline_samples(path, NAMES, samples_per_segment=4)
    assert times.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    # Rest-to-rest quintic: 10s^3 - 15s^4 + 6s^5.
    assert joints[:, 0] == pytest.approx([0.0, 0.103515625, 0.5, 0.896484375, 1.0])


def test_spline_without_derivatives_is_linear_and_holds_skip():
    path = trajectory([0.0, 1.0, 1.0, 2.0], [[0.0] * 6, [1.0] * 6, [1.0] * 6, [3.0] * 6])
    times, joints = spline_samples(path, NAMES, samples_per_segment=2)
    assert joints[:, 0] == pytest.approx([0.0, 0.5, 1.0, 2.0, 3.0])


def test_moveit_like_straight_line_is_on_the_requested_path(chain):
    joints = straight_line_joints(chain, Q_RIGHT, (0.0, 0.10, 0.0), 40)
    knots = joints[::5]
    times = np.linspace(0.0, 4.0, len(knots))
    velocities = np.gradient(knots, times, axis=0)
    accelerations = np.gradient(velocities, times, axis=0)
    path = trajectory(times, knots, velocities, accelerations)
    start, end = chain.fk(joints[0]), chain.fk(joints[-1])
    result = tcp_path_deviation(path, NAMES, chain.fk, [pose(*end)])
    assert result["position_m"] < 1e-4
    assert result["orientation_rad"] < math.radians(0.1)
    assert result["end_position_m"] < 1e-6
    assert verify_path_fidelity(SimpleNamespace(joint_names=NAMES, fk=chain.fk), [pose(*end)], path)
    assert np.allclose(start[0] + (0.0, 0.10, 0.0), end[0], atol=1e-4)


def test_branch_change_with_a_complete_fraction_is_refused(chain):
    # Same requested TCP from start to end, but the commanded joints swing the
    # wrist through a full J4/J6 turn in between (the Humble failure mode).
    flipped = Q_RIGHT + np.array([0.0, 0.0, 0.0, 2 * math.pi, 0.0, -2 * math.pi])
    times = np.linspace(0.0, 2.0, 9)
    knots = np.array([Q_RIGHT + (flipped - Q_RIGHT) * s for s in np.linspace(0, 1, 9)])
    path = trajectory(times, knots)
    stationary = pose(*chain.fk(Q_RIGHT))
    result = tcp_path_deviation(path, NAMES, chain.fk, [stationary])
    assert result["end_position_m"] < 1e-6        # endpoints look perfect ...
    assert result["position_m"] > 0.05 or result["orientation_rad"] > math.radians(90)
    with pytest.raises(RuntimeError, match="IK branch change"):
        verify_path_fidelity(SimpleNamespace(joint_names=NAMES, fk=chain.fk), [stationary], path)


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_rotation_only_segment_compares_attitude_along_its_slerp():
    fk = lambda q: (np.array([0.5, 0.0, 0.3]), rotation_z(q[0]))  # noqa: E731
    target = pose((0.5, 0.0, 0.3), rotation_z(math.pi / 2))
    following = trajectory([0.0, 1.0], [[0.0] * 6, [math.pi / 2] + [0.0] * 5])
    assert tcp_path_deviation(following, NAMES, fk, [target])["orientation_rad"] < 1e-9
    overshoot = trajectory([0.0, 1.0, 2.0], [[0.0] * 6, [math.pi / 2 + 0.5] + [0.0] * 5,
                                             [math.pi / 2] + [0.0] * 5])
    result = tcp_path_deviation(overshoot, NAMES, fk, [target])
    assert result["orientation_rad"] == pytest.approx(0.5, abs=1e-6)
    assert result["position_m"] == 0.0


def test_joint_order_is_resolved_by_name(chain):
    joints = straight_line_joints(chain, Q_RIGHT, (0.0, 0.0, 0.05), 10)
    reversed_names = list(reversed(NAMES))
    path = trajectory([0.0, 1.0], [joints[0][::-1], joints[-1][::-1]], names=reversed_names)
    result = tcp_path_deviation(path, NAMES, chain.fk, [pose(*chain.fk(joints[-1]))])
    assert result["end_position_m"] < 1e-6


def test_missing_joints_short_points_or_non_finite_values_are_rejected():
    with pytest.raises(ValueError, match="each arm joint exactly once"):
        spline_samples(trajectory([0.0], [[0.0] * 5], names=NAMES[:5]), NAMES)
    with pytest.raises(ValueError, match="one position per joint"):
        spline_samples(trajectory([0.0, 1.0], [[0.0] * 6, [0.0] * 5]), NAMES)
    with pytest.raises(ValueError, match="non-finite"):
        spline_samples(trajectory([0.0, 1.0], [[0.0] * 6, [float("nan")] * 6]), NAMES)
