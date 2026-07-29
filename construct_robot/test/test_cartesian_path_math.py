import math

from geometry_msgs.msg import Pose
import numpy as np

from construct_robot.cartesian_path_common import (
    pose_is_valid,
    tip_link_for_group,
)
from construct_robot.cartesian_path_server import (
    interpolate_pose,
    rotate_vector,
)
from construct_robot.viser_utils import (
    merge_joint_positions,
    resolve_ros_resource,
    xyzw_to_wxyz,
)


def make_pose(x=0.0, y=0.0, z=0.0, quaternion=(0.0, 0.0, 0.0, 1.0)):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = quaternion
    return pose


def test_tip_link_for_supported_groups():
    assert tip_link_for_group("left_manipulator") == "left_manipulator_ee_point"
    assert tip_link_for_group("right_manipulator") == "right_manipulator_ee_point"


def test_tip_link_rejects_unknown_group():
    try:
        tip_link_for_group("dual_arm")
    except ValueError as error:
        assert "Unsupported planning group" in str(error)
    else:
        raise AssertionError("Expected an unsupported group to raise ValueError")


def test_pose_validation():
    assert pose_is_valid(make_pose())
    assert not pose_is_valid(make_pose(quaternion=(0.0, 0.0, 0.0, 0.0)))
    assert not pose_is_valid(make_pose(x=math.nan))


def test_interpolation_handles_antipodal_quaternions():
    start = make_pose(quaternion=(0.0, 0.0, 0.0, 1.0))
    goal = make_pose(x=2.0, quaternion=(0.0, 0.0, 0.0, -1.0))
    midpoint = interpolate_pose(start, goal, 0.5)
    assert midpoint.position.x == 1.0
    assert midpoint.orientation.w == 1.0


def test_rotate_vector_quarter_turn_about_z():
    pose = make_pose(
        quaternion=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    )
    rotated = rotate_vector(pose.orientation, (1.0, 0.0, 0.0))
    assert math.isclose(rotated[0], 0.0, abs_tol=1e-9)
    assert math.isclose(rotated[1], 1.0, abs_tol=1e-9)
    assert math.isclose(rotated[2], 0.0, abs_tol=1e-9)


def test_resolve_ros_resource():
    resolved = resolve_ros_resource(
        "package://construct_description/urdf_0528/construct_robot_0528.urdf"
    )
    assert resolved.endswith(
        "/construct_description/urdf_0528/construct_robot_0528.urdf"
    )


def test_joint_state_merge_ignores_unknown_and_non_finite_values():
    merged = merge_joint_positions(
        [0.0, 0.0],
        {"joint_a": 0, "joint_b": 1},
        ["joint_b", "unknown", "joint_a"],
        [1.5, 2.0, math.nan],
    )
    assert merged.tolist() == [0.0, 1.5]


def test_viser_quaternion_order_and_normalization():
    pose = make_pose(quaternion=(0.0, 0.0, 2.0, 2.0))
    wxyz = xyzw_to_wxyz(pose.orientation)
    assert np.allclose(wxyz, [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
