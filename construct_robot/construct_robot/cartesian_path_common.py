import math

from geometry_msgs.msg import Pose


PLANNING_GROUP_TIPS = {
    "left_manipulator": "left_manipulator_ee_point",
    "right_manipulator": "right_manipulator_ee_point",
}


def tip_link_for_group(planning_group: str) -> str:
    """Return the configured TCP link for a supported MoveIt group."""
    try:
        return PLANNING_GROUP_TIPS[planning_group]
    except KeyError as error:
        supported = ", ".join(sorted(PLANNING_GROUP_TIPS))
        raise ValueError(
            f"Unsupported planning group '{planning_group}'; expected {supported}"
        ) from error


def pose_is_valid(pose: Pose) -> bool:
    """Check that a pose is finite and has a non-zero quaternion."""
    values = (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    if not all(math.isfinite(value) for value in values):
        return False
    quaternion_norm_squared = sum(value * value for value in values[3:])
    return quaternion_norm_squared > 1e-12
