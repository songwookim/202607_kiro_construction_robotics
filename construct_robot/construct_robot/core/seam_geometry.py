"""Pure seam and approach geometry shared by GUI and multi-pass correction."""
import copy
import math
from dataclasses import dataclass

from geometry_msgs.msg import Pose

from construct_robot.core.cartesian_path_common import pose_is_valid


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


@dataclass
class CorrectedSeamGeometry:
    """Actual two-plane seam line and projected taught endpoints."""

    start: Pose
    goal: Pose
    taught_start: tuple
    taught_goal: tuple
    origin: tuple
    d_teach: tuple
    d_real: tuple
    e_a: tuple
    e_w: tuple
    wall_normal: tuple
    wall_plane_value: float
    floor_normal: tuple
    floor_plane_value: float
    length_before: float
    length_after: float
    direction_dot: float
    safe_start: Pose = None
    lead_start: Pose = None


def compute_surface_plane(normal, touch_points, offset=0.0):
    """Estimate unit-normal plane n·x=c from touch points and a probe hint.

    With START and GOAL contacts, their connecting vector lies in the real
    surface.  Projecting the configured probe-normal hint perpendicular to
    that vector captures workpiece tilt while selecting the otherwise
    ambiguous plane normal.  A single contact retains the hint as fallback.
    """
    normal_hint = _unit_vector(normal, "surface probe-normal hint")
    positions = []
    for point in touch_points:
        if isinstance(point, Pose):
            if not pose_is_valid(point):
                raise ValueError("surface touch pose is invalid")
            positions.append(_pose_position_tuple(point))
        else:
            values = tuple(float(value) for value in point)
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError("surface touch point must be a finite XYZ vector")
            positions.append(values)
    if not positions:
        raise ValueError("surface plane needs at least one touch point")
    normal = normal_hint
    if len(positions) >= 2:
        first, second = max(
            (
                (first, second)
                for index, first in enumerate(positions[:-1])
                for second in positions[index + 1:]
            ),
            key=lambda pair: math.dist(pair[0], pair[1]),
        )
        span = tuple(second[index] - first[index] for index in range(3))
        if math.sqrt(_vector_dot(span, span)) > 1e-6:
            tangent = _unit_vector(span, "surface touch span")
            projection = _vector_dot(normal_hint, tangent)
            normal = _unit_vector(
                tuple(
                    normal_hint[index] - projection * tangent[index]
                    for index in range(3)
                ),
                "probe hint projected onto sensed surface normal",
            )
    plane_value = (
        sum(_vector_dot(normal, point) for point in positions) / len(positions)
        + float(offset)
    )
    if not math.isfinite(plane_value):
        raise ValueError("surface plane value is not finite")
    return normal, plane_value


def compute_real_seam_direction(wall_normal, floor_normal, d_teach):
    """Return the oriented intersection direction of two surface planes."""
    wall_normal = _unit_vector(wall_normal, "wall normal")
    floor_normal = _unit_vector(floor_normal, "floor normal")
    d_teach = _unit_vector(d_teach, "taught seam direction")
    cross = _vector_cross(wall_normal, floor_normal)
    cross_norm = math.sqrt(_vector_dot(cross, cross))
    if cross_norm < 1e-6:
        raise ValueError(
            "wall and floor normals are nearly parallel; actual seam line "
            "cannot be determined"
        )
    d_real = tuple(value / cross_norm for value in cross)
    alignment = _vector_dot(d_real, d_teach)
    if alignment < 0.0:
        d_real = tuple(-value for value in d_real)
        alignment = -alignment
    if alignment < 1e-6:
        raise ValueError(
            "actual seam direction is nearly perpendicular to taught START→GOAL"
        )
    return d_real


def compute_plane_intersection_line(
    wall_normal,
    wall_plane_value,
    floor_normal,
    floor_plane_value,
    direction_reference=None,
):
    """Return the minimum-norm point and direction of two planes' line."""
    wall_normal = _unit_vector(wall_normal, "wall normal")
    floor_normal = _unit_vector(floor_normal, "floor normal")
    reference = (
        direction_reference
        if direction_reference is not None
        else _vector_cross(wall_normal, floor_normal)
    )
    direction = compute_real_seam_direction(
        wall_normal, floor_normal, reference
    )
    normal_dot = _vector_dot(wall_normal, floor_normal)
    denominator = 1.0 - normal_dot * normal_dot
    if denominator < 1e-12:
        raise ValueError(
            "wall and floor normals are nearly parallel; no stable plane "
            "intersection exists"
        )
    wall_coefficient = (
        float(wall_plane_value) - normal_dot * float(floor_plane_value)
    ) / denominator
    floor_coefficient = (
        float(floor_plane_value) - normal_dot * float(wall_plane_value)
    ) / denominator
    origin = tuple(
        wall_coefficient * wall_normal[index]
        + floor_coefficient * floor_normal[index]
        for index in range(3)
    )
    return origin, direction


def project_point_to_line(point, origin, direction):
    """Orthogonally project a Pose/XYZ point onto origin+t*direction."""
    position = _pose_position_tuple(point) if isinstance(point, Pose) else tuple(point)
    if len(position) != 3 or not all(math.isfinite(float(v)) for v in position):
        raise ValueError("point to project must be a finite XYZ vector")
    origin = tuple(float(value) for value in origin)
    direction = _unit_vector(direction, "line direction")
    parameter = _vector_dot(
        direction,
        tuple(float(position[index]) - origin[index] for index in range(3)),
    )
    return tuple(
        origin[index] + parameter * direction[index] for index in range(3)
    )


def compute_seam_local_frame(
    d_real,
    wall_normal,
    floor_normal,
    approach_reference=None,
):
    """Build orthonormal travel/weave/approach axes for the actual seam."""
    d_real = _unit_vector(d_real, "actual seam direction")
    wall_normal = _unit_vector(wall_normal, "wall normal")
    floor_normal = _unit_vector(floor_normal, "floor normal")
    approach = _unit_vector(
        tuple(wall_normal[index] + floor_normal[index] for index in range(3)),
        "wall/floor approach bisector",
    )
    if approach_reference is not None:
        reference = _unit_vector(approach_reference, "taught TCP approach")
        if _vector_dot(approach, reference) < 0.0:
            approach = tuple(-value for value in approach)
    weave = _unit_vector(
        _vector_cross(d_real, approach), "geometry-derived weave direction"
    )
    approach = _unit_vector(
        _vector_cross(weave, d_real), "orthogonalized approach direction"
    )
    return d_real, weave, approach


def compute_corrected_seam_endpoints(taught_start, taught_goal, origin, d_real):
    """Project taught longitudinal endpoints onto the actual seam line."""
    start_xyz = project_point_to_line(taught_start, origin, d_real)
    goal_xyz = project_point_to_line(taught_goal, origin, d_real)
    start = copy.deepcopy(taught_start)
    goal = copy.deepcopy(taught_goal)
    start.position.x, start.position.y, start.position.z = start_xyz
    goal.position.x, goal.position.y, goal.position.z = goal_xyz
    corrected_direction = seam_direction(start, goal)
    direction_dot = _vector_dot(corrected_direction, d_real)
    if direction_dot < 1.0 - 1e-6:
        raise ValueError(
            "projected START→GOAL direction does not match actual seam line "
            f"(dot={direction_dot:.9f})"
        )
    return start, goal, direction_dot


def compute_corrected_seam_geometry(
    taught_start,
    taught_goal,
    wall_plane,
    floor_plane,
    approach_reference=None,
):
    """Compute a two-plane seam line and project taught endpoints onto it."""
    if not pose_is_valid(taught_start) or not pose_is_valid(taught_goal):
        raise ValueError("taught START/GOAL poses must be valid")
    d_teach = seam_direction(taught_start, taught_goal)
    wall_normal = _unit_vector(wall_plane[0], "wall normal")
    floor_normal = _unit_vector(floor_plane[0], "floor normal")
    d_real = compute_real_seam_direction(
        wall_normal, floor_normal, d_teach
    )
    origin, line_direction = compute_plane_intersection_line(
        wall_normal,
        wall_plane[1],
        floor_normal,
        floor_plane[1],
        d_teach,
    )
    # Both helpers use the same sign reference; retain the explicit direction
    # result as a consistency check against future implementation changes.
    if _vector_dot(d_real, line_direction) < 1.0 - 1e-9:
        raise ValueError("inconsistent actual seam directions")
    start, goal, direction_dot = compute_corrected_seam_endpoints(
        taught_start, taught_goal, origin, d_real
    )
    d_real, e_w, e_a = compute_seam_local_frame(
        d_real,
        wall_normal,
        floor_normal,
        approach_reference,
    )
    before = math.dist(
        _pose_position_tuple(taught_start), _pose_position_tuple(taught_goal)
    )
    after = math.dist(_pose_position_tuple(start), _pose_position_tuple(goal))
    return CorrectedSeamGeometry(
        start=start,
        goal=goal,
        taught_start=_pose_position_tuple(taught_start),
        taught_goal=_pose_position_tuple(taught_goal),
        origin=origin,
        d_teach=d_teach,
        d_real=d_real,
        e_a=e_a,
        e_w=e_w,
        wall_normal=wall_normal,
        wall_plane_value=float(wall_plane[1]),
        floor_normal=floor_normal,
        floor_plane_value=float(floor_plane[1]),
        length_before=before,
        length_after=after,
        direction_dot=direction_dot,
    )


def compute_safe_weld_approach(
    corrected_start,
    d_real,
    e_a,
    safe_distance_m,
    lead_distance_m,
):
    """Return fixed-attitude safe and pre-start poses for corner approach."""
    if not pose_is_valid(corrected_start):
        raise ValueError("corrected weld START pose is invalid")
    d_real = _unit_vector(d_real, "actual seam direction")
    e_a = _unit_vector(e_a, "torch approach direction")
    if abs(_vector_dot(d_real, e_a)) > 1e-6:
        raise ValueError("actual seam and torch approach directions are not orthogonal")
    safe_distance_m = float(safe_distance_m)
    lead_distance_m = float(lead_distance_m)
    if not math.isfinite(safe_distance_m) or safe_distance_m <= 0.0:
        raise ValueError("safe approach distance must be positive and finite")
    if not math.isfinite(lead_distance_m) or lead_distance_m < 0.0:
        raise ValueError("pre-start lead distance must be non-negative and finite")

    lead_pose = copy.deepcopy(corrected_start)
    for index, axis in enumerate(("x", "y", "z")):
        setattr(
            lead_pose.position,
            axis,
            getattr(corrected_start.position, axis)
            - lead_distance_m * d_real[index],
        )
    safe_pose = copy.deepcopy(lead_pose)
    for index, axis in enumerate(("x", "y", "z")):
        setattr(
            safe_pose.position,
            axis,
            getattr(lead_pose.position, axis) + safe_distance_m * e_a[index],
        )

    safe_offset = tuple(
        getattr(safe_pose.position, axis) - getattr(lead_pose.position, axis)
        for axis in ("x", "y", "z")
    )
    if _vector_dot(_unit_vector(safe_offset), e_a) < 1.0 - 1e-6:
        raise ValueError("safe approach offset is not aligned with e_a")
    if lead_distance_m > 1e-9:
        lead_vector = tuple(
            getattr(corrected_start.position, axis)
            - getattr(lead_pose.position, axis)
            for axis in ("x", "y", "z")
        )
        if _vector_dot(_unit_vector(lead_vector), d_real) < 1.0 - 1e-6:
            raise ValueError("pre-start lead is not aligned with d_real")
    return safe_pose, lead_pose

# lead 는 weld seam 의 시작점과 끝점을 기준으로, 용접을 시작하기 전과 끝난 후에 로봇이 움직일 수 있는 여유 공간을 제공하는 포즈를 계산하는 함수입니다.
# lead out은 arc off를 하면서 로봇이 움직일 수 있는 여유 공간을 제공하는 포즈를 계산합니다.
def seam_lead_poses(start, goal, lead_in_m=0.0, lead_out_m=0.0):
    """Extend a seam tangentially before START and after GOAL.

    The original START/GOAL remain the usable weld seam.  The returned lead
    poses provide sacrificial run-in/run-out distance so robot acceleration,
    arc establishment, deceleration, and arc extinction occur outside that
    usable seam.  Endpoint orientations are preserved from START/GOAL.
    """
    if not pose_is_valid(start) or not pose_is_valid(goal):
        raise ValueError("seam START/GOAL poses must be valid")
    lead_in_m = float(lead_in_m)
    lead_out_m = float(lead_out_m)
    if (
        not math.isfinite(lead_in_m)
        or not math.isfinite(lead_out_m)
        or lead_in_m < 0.0
        or lead_out_m < 0.0
    ):
        raise ValueError("weld lead-in/out distances must be finite and non-negative")
    tangent = seam_direction(start, goal)
    lead_start = copy.deepcopy(start)
    lead_end = copy.deepcopy(goal)
    for axis, component in zip(("x", "y", "z"), tangent):
        setattr(
            lead_start.position,
            axis,
            getattr(start.position, axis) - component * lead_in_m,
        )
        setattr(
            lead_end.position,
            axis,
            getattr(goal.position, axis) + component * lead_out_m,
        )
    return lead_start, lead_end
