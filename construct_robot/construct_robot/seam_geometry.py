"""Pure seam vector helpers shared by GUI and multi-pass correction."""
import math

from construct_robot.cartesian_path_common import pose_is_valid


def _vector_dot(first, second):
    return sum(float(a) * float(b) for a, b in zip(first, second))


def _vector_cross(first, second):
    ax, ay, az = (float(value) for value in first)
    bx, by, bz = (float(value) for value in second)
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


def _unit_vector(vector, description="vector"):
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-9:
        raise ValueError(f"{description} has near-zero length")
    return tuple(value / norm for value in values)

def seam_direction(start, goal, *, xy_only=False):
    """Return a unit START→GOAL direction, optionally projected onto World XY."""
    if not pose_is_valid(start) or not pose_is_valid(goal):
        raise ValueError("seam START/GOAL poses must be valid")
    direction = (
        goal.position.x - start.position.x,
        goal.position.y - start.position.y,
        0.0 if xy_only else goal.position.z - start.position.z,
    )
    return _unit_vector(direction, "seam direction")

def _pose_position_tuple(pose):
    return (pose.position.x, pose.position.y, pose.position.z)


