"""Pure seam and approach geometry shared by GUI and multi-pass correction."""
import copy
import math
from dataclasses import dataclass

from geometry_msgs.msg import Pose

from construct_robot.core.cartesian_path_common import (
    linear_pose_waypoints,
    midpoint_pose,
    pose_is_valid,
    pose_with_rpy_offset,
    transform_xyz,
)


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


CORNER_TOUCH_NAMES = (
    "start_floor",
    "start_wall",
    "goal_floor",
    "goal_wall",
)


def corner_seam_from_touches(touches, count):
    """Build a seam between two floor/wall touch-pair midpoints."""
    missing = [name for name in CORNER_TOUCH_NAMES if touches.get(name) is None]
    if missing:
        raise ValueError("missing corner touches: " + ", ".join(missing))
    start = midpoint_pose(touches["start_floor"], touches["start_wall"])
    end = midpoint_pose(touches["goal_floor"], touches["goal_wall"])
    return linear_pose_waypoints(start, end, count)


def corrected_corner_seam_from_four_touches(
    touches,
    seam_axis,
    count,
    wall_offset=0.0,
    floor_offset=0.0,
):
    """Project START/GOAL wall-floor touch pairs onto the corner seam."""
    missing = [name for name in CORNER_TOUCH_NAMES if touches.get(name) is None]
    if missing:
        raise ValueError("missing corner touches: " + ", ".join(missing))
    if seam_axis.lower() != "x":
        raise ValueError("Y/Z touch seam calculation requires World X axis")
    endpoints = []
    for endpoint in ("start", "goal"):
        floor = touches[f"{endpoint}_floor"]
        wall = touches[f"{endpoint}_wall"]
        pose = midpoint_pose(floor, wall)
        # Y/Z probing reconstructs the corner using only these components:
        # X = common probe cross-section (1:1 mean), Y = wall, Z = floor.
        pose.position.x = (wall.position.x + floor.position.x) * 0.5
        pose.position.y = wall.position.y + wall_offset
        pose.position.z = floor.position.z + floor_offset
        endpoints.append(pose)
    return linear_pose_waypoints(endpoints[0], endpoints[1], count)


def corner_endpoint_from_two_touches(
    wall_touch,
    floor_touch,
    orientation_pose,
    seam_axis,
    wall_offset=0.0,
    floor_offset=0.0,
):
    """Reconstruct one seam XYZ from World-axis wall/floor probes.

    Both probes start from the same cross-section and move only along the
    configured World wall axis or World Z.  Their nominal seam-axis coordinate
    is therefore the mean of the two measured TCP coordinates.  The taught
    pose supplies orientation only; none of its XYZ values enter the result.
    """
    for name, pose in (
        ("wall touch", wall_touch),
        ("floor touch", floor_touch),
        ("orientation pose", orientation_pose),
    ):
        if not pose_is_valid(pose):
            raise ValueError(f"{name} pose is invalid")
    if seam_axis.lower() != "x":
        raise ValueError("Y/Z touch seam calculation requires World X axis")
    result = copy.deepcopy(orientation_pose)
    result.position.x = (
        wall_touch.position.x + floor_touch.position.x
    ) * 0.5
    result.position.y = wall_touch.position.y + wall_offset

    # The wall touch measures the lateral wall coordinate; the floor touch
    # measures the floor height.  Orientation is intentionally untouched.
    result.position.z = floor_touch.position.z + floor_offset
    return result


def aligned_wait_pose(wait_pose, seam_point, seam_axis):
    """Align a wait pose to the seam cross-section while retaining stand-off."""
    if not pose_is_valid(wait_pose) or not pose_is_valid(seam_point):
        raise ValueError("wait pose and seam point must be valid")
    result = copy.deepcopy(wait_pose)
    axis = seam_axis.lower()
    if axis == "x":
        result.position.y = seam_point.position.y
    elif axis == "y":
        result.position.x = seam_point.position.x
    else:
        raise ValueError("0°/90° seam axis must be World X or Y")
    result.position.z = seam_point.position.z
    return result


def translated_wait_pose(wait_pose, taught_seam_pose, corrected_seam_pose):
    """Move a taught wait TCP with its corrected seam endpoint.

    The taught wait-to-seam offset is the intentional approach clearance.  A
    wall/floor touch midpoint is a measurement artifact, not a safe wait pose,
    so never replace that clearance with the midpoint coordinates.
    """
    for name, pose in (
        ("wait pose", wait_pose),
        ("taught seam pose", taught_seam_pose),
        ("corrected seam pose", corrected_seam_pose),
    ):
        if not pose_is_valid(pose):
            raise ValueError(f"{name} is invalid")
    result = copy.deepcopy(wait_pose)
    result.position.x += (
        corrected_seam_pose.position.x - taught_seam_pose.position.x
    )
    result.position.y += (
        corrected_seam_pose.position.y - taught_seam_pose.position.y
    )
    result.position.z += (
        corrected_seam_pose.position.z - taught_seam_pose.position.z
    )
    return result


def two_touch_corner_seam(
    wall_touch,
    floor_touch,
    taught_start,
    taught_end,
    seam_axis,
    count,
    wall_offset=0.0,
    floor_offset=0.0,
):
    """Build an orthogonal seam from wall/floor touches and taught endpoints."""
    for name, pose in (
        ("wall touch", wall_touch),
        ("floor touch", floor_touch),
        ("taught start", taught_start),
        ("taught end", taught_end),
    ):
        if not pose_is_valid(pose):
            raise ValueError(f"{name} pose is invalid")
    axis = seam_axis.lower()
    if axis not in ("x", "y"):
        raise ValueError("0°/90° seam axis must be World X or Y")
    start = copy.deepcopy(taught_start)
    end = copy.deepcopy(taught_end)
    if axis == "x":
        start.position.y = wall_touch.position.y + wall_offset
        end.position.y = start.position.y
    else:
        start.position.x = wall_touch.position.x + wall_offset
        end.position.x = start.position.x
    start.position.z = floor_touch.position.z + floor_offset
    end.position.z = start.position.z
    return linear_pose_waypoints(start, end, count)


def _axis_unit_vector(axis):
    axis = str(axis).strip().lower().replace("world ", "")
    vectors = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }
    if axis not in vectors:
        raise ValueError(f"unsupported World probe axis: {axis}")
    return vectors[axis]


def seam_xy_normal(start, goal):
    """Return the +90° World-XY normal of the taught START→GOAL seam."""
    tx, ty, _tz = seam_direction(start, goal, xy_only=True)
    return (-ty, tx, 0.0)


def intersect_three_planes(normal_a, value_a, normal_b, value_b, normal_c, value_c):
    """Return the unique point satisfying n·p=d for three independent planes."""
    normal_a = _unit_vector(normal_a, "plane A normal")
    normal_b = _unit_vector(normal_b, "plane B normal")
    normal_c = _unit_vector(normal_c, "plane C normal")
    b_cross_c = _vector_cross(normal_b, normal_c)
    denominator = _vector_dot(normal_a, b_cross_c)
    if abs(denominator) < 1e-6:
        raise ValueError(
            "probe directions and seam cross-section are not independent; "
            "choose probe directions that measure two different surfaces"
        )
    c_cross_a = _vector_cross(normal_c, normal_a)
    a_cross_b = _vector_cross(normal_a, normal_b)
    numerator = tuple(
        float(value_a) * b_cross_c[index]
        + float(value_b) * c_cross_a[index]
        + float(value_c) * a_cross_b[index]
        for index in range(3)
    )
    return tuple(value / denominator for value in numerator)


def generalized_corner_endpoint_from_two_touches(
    wall_touch,
    floor_touch,
    orientation_pose,
    taught_start,
    taught_goal,
    wall_normal,
    floor_normal,
    wall_offset=0.0,
    floor_offset=0.0,
):
    """Reconstruct a seam endpoint from two touched planes and a seam cross-section.

    The two contact TCP positions define one point on each sensed plane.  The
    configured probe directions are used as those plane normals.  The third
    plane is perpendicular to the taught seam direction; its location is the
    mean longitudinal coordinate of the two contacts.  This is the vector form
    of the old World-X/Y/Z rule (mean X, wall Y, floor Z).
    """
    for name, pose in (
        ("wall touch", wall_touch),
        ("floor touch", floor_touch),
        ("orientation pose", orientation_pose),
        ("taught start", taught_start),
        ("taught goal", taught_goal),
    ):
        if not pose_is_valid(pose):
            raise ValueError(f"{name} pose is invalid")

    wall_normal = _unit_vector(wall_normal, "wall probe direction")
    floor_normal = _unit_vector(floor_normal, "floor probe direction")
    tangent = seam_direction(taught_start, taught_goal)
    wall_position = _pose_position_tuple(wall_touch)
    floor_position = _pose_position_tuple(floor_touch)

    wall_plane = _vector_dot(wall_normal, wall_position) + float(wall_offset)
    floor_plane = _vector_dot(floor_normal, floor_position) + float(floor_offset)
    cross_section = 0.5 * (
        _vector_dot(tangent, wall_position)
        + _vector_dot(tangent, floor_position)
    )
    x, y, z = intersect_three_planes(
        wall_normal, wall_plane,
        floor_normal, floor_plane,
        tangent, cross_section,
    )
    result = copy.deepcopy(orientation_pose)
    result.position.x = x
    result.position.y = y
    result.position.z = z
    return result


def apply_sensed_seam_orientation(
    taught_start,
    taught_goal,
    sensed_start,
    sensed_goal,
    mode,
):
    """Combine sensed XYZ with the orientation policy selected for welding."""
    normalized = str(mode).strip().lower()
    if normalized.startswith("wait"):
        # WAIT XYZ is a probe standby location, not a taught seam endpoint.
        # Its START→GOAL heading must not rotate the fixed welding attitudes.
        start = copy.deepcopy(taught_start)
        goal = copy.deepcopy(taught_goal)
        start.position = copy.deepcopy(sensed_start.position)
        goal.position = copy.deepcopy(sensed_goal.position)
        return start, goal, 0.0, "WAIT + fixed World-XYZ tilt; yaw not applied"
    if normalized.startswith("yaw") or normalized.startswith("follow"):
        start, goal, delta_yaw = yaw_corrected_seam_poses(
            taught_start, taught_goal, sensed_start, sensed_goal
        )
        return start, goal, delta_yaw, "yaw-corrected"
    if normalized.startswith("keep"):
        start = copy.deepcopy(taught_start)
        goal = copy.deepcopy(taught_goal)
        start.position = copy.deepcopy(sensed_start.position)
        goal.position = copy.deepcopy(sensed_goal.position)
        return start, goal, 0.0, "teaching orientation kept"
    raise ValueError(f"unknown seam orientation mode: {mode}")


def seam_yaw(start, goal):
    """Return World-Z seam yaw from two TCP positions."""
    dx = goal.position.x - start.position.x
    dy = goal.position.y - start.position.y
    if math.hypot(dx, dy) < 1e-9:
        raise ValueError("seam START/GOAL have no usable XY direction")
    return math.atan2(dy, dx)


def yaw_corrected_seam_poses(
    taught_start,
    taught_goal,
    sensed_start,
    sensed_goal,
):
    """Apply sensed-vs-taught seam yaw to taught orientations and sensed XYZ."""
    for name, pose in (
        ("taught start", taught_start),
        ("taught goal", taught_goal),
        ("sensed start", sensed_start),
        ("sensed goal", sensed_goal),
    ):
        if not pose_is_valid(pose):
            raise ValueError(f"{name} pose is invalid")
    taught_yaw = seam_yaw(taught_start, taught_goal)
    sensed_yaw = seam_yaw(sensed_start, sensed_goal)
    delta_yaw = math.atan2(
        math.sin(sensed_yaw - taught_yaw),
        math.cos(sensed_yaw - taught_yaw),
    )
    corrected_start = pose_with_rpy_offset(
        taught_start, 0.0, 0.0, delta_yaw, reference="world"
    )
    corrected_goal = pose_with_rpy_offset(
        taught_goal, 0.0, 0.0, delta_yaw, reference="world"
    )
    corrected_start.position = copy.deepcopy(sensed_start.position)
    corrected_goal.position = copy.deepcopy(sensed_goal.position)
    return corrected_start, corrected_goal, delta_yaw


def fixed_tilt_wait_reference_poses(
    start_wait,
    goal_wait,
    tilt_y_deg,
    tilt_x_deg=0.0,
    tilt_z_deg=0.0,
):
    """Create consistent seam attitudes from START/GOAL WAIT teaching.

    The WAIT poses supply the two base orientations. Exactly the same fixed
    World XYZ RPY rotation is then applied at both ends. Their XYZ values are
    kept so the WAIT-to-WAIT vector can also serve as the nominal seam
    direction when no separate weld START/GOAL teaching exists.
    """
    if not pose_is_valid(start_wait) or not pose_is_valid(goal_wait):
        raise ValueError("START/GOAL WAIT poses must be valid")
    tilt_x_deg = float(tilt_x_deg)
    tilt_y_deg = float(tilt_y_deg)
    tilt_z_deg = float(tilt_z_deg)
    tilts = (tilt_x_deg, tilt_y_deg, tilt_z_deg)
    if not all(
        math.isfinite(value) and -180.0 <= value <= 180.0
        for value in tilts
    ):
        raise ValueError("fixed World XYZ angles must each be in -180..180 degrees")
    rpy = tuple(math.radians(value) for value in tilts)
    return (
        pose_with_rpy_offset(start_wait, *rpy, reference="world"),
        pose_with_rpy_offset(goal_wait, *rpy, reference="world"),
    )


def wide_sensing_path_poses(
    start_xyz,
    end_xyz,
    world_from_sensor,
    orientation,
    offset_m=(0.0, 0.0, 0.0),
    reverse=False,
):
    """Convert a sensed metric segment into two World-frame weld poses."""
    start = transform_xyz(world_from_sensor, start_xyz)
    end = transform_xyz(world_from_sensor, end_xyz)
    if reverse:
        start, end = end, start
    offset = tuple(float(value) for value in offset_m)
    if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
        raise ValueError("Wide Sensing World offset must be finite XYZ")
    poses = []
    for xyz in (start, end):
        pose = Pose()
        pose.position.x = xyz[0] + offset[0]
        pose.position.y = xyz[1] + offset[1]
        pose.position.z = xyz[2] + offset[2]
        pose.orientation = copy.deepcopy(orientation)
        if not pose_is_valid(pose):
            raise ValueError("Wide Sensing produced an invalid World pose")
        poses.append(pose)
    if math.dist(start, end) < 1e-5:
        raise ValueError("Wide Sensing weld segment is shorter than 0.01 mm")
    return tuple(poses)


WEAVE_REFERENCES = ("touch_pair", "bisector", "manual")
TOUCH_PAIR_MIN_SEPARATION_M = 0.0005


def touch_pair_weave_direction(touches, d_real, orientation_reference=None):
    """Weave across the joint: from each endpoint's wall touch to its floor touch.

    For every endpoint with both touches, the wall→floor contact vector is
    projected perpendicular to the actual seam ``d_real``; START and GOAL
    results are averaged.  The weave then oscillates in the plane spanned by
    the seam and the line between the two touched points: across a groove's
    two faces (3G) or across a fillet's two legs.  ``orientation_reference``
    (the bisector ``e_w``) only fixes the +/- sign so left/right dwell keep
    their meaning.  Returns None when no endpoint has a usable touch pair.
    """
    d_real = _unit_vector(d_real, "actual seam direction")
    directions = []
    for endpoint in ("start", "goal"):
        wall = touches.get(f"{endpoint}_wall")
        floor = touches.get(f"{endpoint}_floor")
        if wall is None or floor is None:
            continue
        across = tuple(
            b - a for a, b in zip(_pose_position_tuple(wall), _pose_position_tuple(floor))
        )
        along = _vector_dot(across, d_real)
        across = tuple(across[i] - along * d_real[i] for i in range(3))
        if math.sqrt(_vector_dot(across, across)) < TOUCH_PAIR_MIN_SEPARATION_M:
            continue
        directions.append(_unit_vector(across, f"{endpoint} touch-pair direction"))
    if not directions:
        return None
    first = directions[0]
    summed = [0.0, 0.0, 0.0]
    for direction in directions:
        sign = -1.0 if _vector_dot(direction, first) < 0.0 else 1.0
        for index in range(3):
            summed[index] += sign * direction[index]
    along = _vector_dot(summed, d_real)
    weave = _unit_vector(
        tuple(summed[i] - along * d_real[i] for i in range(3)),
        "touch-pair weave direction",
    )
    if orientation_reference is not None and _vector_dot(weave, orientation_reference) < 0.0:
        weave = tuple(-value for value in weave)
    return weave
