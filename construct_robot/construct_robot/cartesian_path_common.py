import math
import copy

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


def _normalize_vector(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return tuple(value / norm for value in vector)


def _quaternion_rotate(quaternion, vector):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("Cannot rotate with a zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    q_vector = (x, y, z)
    cross_1 = (
        q_vector[1] * vector[2] - q_vector[2] * vector[1],
        q_vector[2] * vector[0] - q_vector[0] * vector[2],
        q_vector[0] * vector[1] - q_vector[1] * vector[0],
    )
    cross_2 = (
        q_vector[1] * cross_1[2] - q_vector[2] * cross_1[1],
        q_vector[2] * cross_1[0] - q_vector[0] * cross_1[2],
        q_vector[0] * cross_1[1] - q_vector[1] * cross_1[0],
    )
    return tuple(
        vector[index]
        + 2.0 * (w * cross_1[index] + cross_2[index])
        for index in range(3)
    )


def _quaternion_from_axes(x_axis, y_axis, z_axis):
    """Return XYZW quaternion for a rotation matrix stored by columns."""
    m00, m10, m20 = x_axis
    m01, m11, m21 = y_axis
    m02, m12, m22 = z_axis
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale
    return qx, qy, qz, qw


def straight_waypoints(
    start: Pose,
    distance: float,
    count: int,
    axis="x",
    reference="world",
):
    """Generate equally spaced poses along a World or TCP-local axis."""
    if not pose_is_valid(start):
        raise ValueError("Straight start pose must be finite and valid")
    if not math.isfinite(distance) or abs(distance) < 1e-9:
        raise ValueError("Straight distance must be finite and non-zero")
    if count < 2:
        raise ValueError("A straight path needs at least two points")
    axis_vectors = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }
    if axis not in axis_vectors:
        raise ValueError(f"Unsupported straight axis: {axis}")
    if reference not in ("world", "tool"):
        raise ValueError(f"Unsupported straight reference: {reference}")
    direction = axis_vectors[axis]
    if reference == "tool":
        direction = _quaternion_rotate(start.orientation, direction)

    points = []
    for index in range(count):
        offset = distance * index / (count - 1)
        pose = copy.deepcopy(start)
        pose.position.x += direction[0] * offset
        pose.position.y += direction[1] * offset
        pose.position.z += direction[2] * offset
        points.append(pose)
    return points


def circle_waypoints(
    center: Pose,
    radius: float,
    count: int,
    closed=True,
    face_center=False,
    normal_axis="x",
):
    """Generate a World-frame circle normal to X/Y/Z.

    ``normal_axis`` selects the circle normal: X produces a YZ circle, Y an
    XZ circle, and Z an XY circle.  When ``face_center`` is true, TCP +Z
    points toward the center while TCP +X follows the circle normal.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Circle radius must be positive and finite")
    if count < 4:
        raise ValueError("A circle needs at least four points")
    plane_axes = {
        "x": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        "y": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "z": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    }
    if normal_axis not in plane_axes:
        raise ValueError(f"Unsupported circle normal axis: {normal_axis}")
    radial_u, radial_v, normal = plane_axes[normal_axis]
    points = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        pose = Pose()
        radial = tuple(
            math.cos(angle) * u + math.sin(angle) * v
            for u, v in zip(radial_u, radial_v)
        )
        pose.position.x = center.position.x + radius * radial[0]
        pose.position.y = center.position.y + radius * radial[1]
        pose.position.z = center.position.z + radius * radial[2]
        if face_center:
            z_axis = tuple(-value for value in radial)
            x_axis = normal
            y_axis = (
                z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
                z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
                z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
            )
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ) = _quaternion_from_axes(x_axis, y_axis, z_axis)
        else:
            pose.orientation = copy.deepcopy(center.orientation)
        points.append(pose)
    if closed:
        closing_pose = Pose()
        closing_pose.position = copy.deepcopy(points[0].position)
        closing_pose.orientation = copy.deepcopy(points[0].orientation)
        points.append(closing_pose)
    return points


def slerp_quaternion(first, second, ratio):
    """Shortest-path spherical interpolation of two XYZW quaternions."""
    first_q = _normalize_vector((first.x, first.y, first.z, first.w))
    second_q = _normalize_vector((second.x, second.y, second.z, second.w))
    dot = sum(a * b for a, b in zip(first_q, second_q))
    if dot < 0.0:
        second_q = tuple(-value for value in second_q)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _normalize_vector(tuple(
            a + (b - a) * ratio for a, b in zip(first_q, second_q)
        ))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    first_scale = math.sin((1.0 - ratio) * theta) / sin_theta
    second_scale = math.sin(ratio * theta) / sin_theta
    return tuple(
        first_scale * a + second_scale * b
        for a, b in zip(first_q, second_q)
    )


def _interpolate_pose(first, second, ratio):
    pose = Pose()
    pose.position.x = (
        first.position.x
        + (second.position.x - first.position.x) * ratio
    )
    pose.position.y = (
        first.position.y
        + (second.position.y - first.position.y) * ratio
    )
    pose.position.z = (
        first.position.z
        + (second.position.z - first.position.z) * ratio
    )
    interpolated = slerp_quaternion(
        first.orientation, second.orientation, ratio
    )
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = interpolated
    return pose


def linear_pose_waypoints(start, end, count):
    """Interpolate a straight Cartesian 6D-pose path between two TCP poses."""
    if not pose_is_valid(start) or not pose_is_valid(end):
        raise ValueError("Linear TCP endpoints must be finite and valid")
    if count < 2:
        raise ValueError("A linear TCP path needs at least two points")
    distance = math.sqrt(
        (end.position.x - start.position.x) ** 2
        + (end.position.y - start.position.y) ** 2
        + (end.position.z - start.position.z) ** 2
    )
    if distance < 1e-9:
        raise ValueError("Linear TCP endpoints must have different positions")
    return [
        _interpolate_pose(start, end, index / (count - 1))
        for index in range(count)
    ]


def weaving_from_path(
    source_points,
    amplitude,
    cycles,
    samples_per_cycle,
    transverse_axis="tool_y",
):
    """Resample an existing taught seam and add transverse sinusoidal weave."""
    if len(source_points) < 2:
        raise ValueError("Teach at least two source points before weaving")
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("Weave amplitude must be positive and finite")
    if cycles < 1 or samples_per_cycle < 4:
        raise ValueError("Weave requires cycles >= 1 and samples/cycle >= 4")
    axis_vectors = {
        "tool_x": (1.0, 0.0, 0.0),
        "tool_y": (0.0, 1.0, 0.0),
        "tool_z": (0.0, 0.0, 1.0),
        "world_x": (1.0, 0.0, 0.0),
        "world_y": (0.0, 1.0, 0.0),
        "world_z": (0.0, 0.0, 1.0),
    }
    if transverse_axis not in axis_vectors:
        raise ValueError(f"Unsupported weave axis: {transverse_axis}")

    segment_lengths = []
    for first, second in zip(source_points[:-1], source_points[1:]):
        length = math.sqrt(
            (second.position.x - first.position.x) ** 2
            + (second.position.y - first.position.y) ** 2
            + (second.position.z - first.position.z) ** 2
        )
        segment_lengths.append(length)
    total_length = sum(segment_lengths)
    if total_length < 1e-9:
        raise ValueError("Taught seam has zero length")

    sample_count = cycles * samples_per_cycle
    points = []
    segment_index = 0
    distance_before_segment = 0.0
    for sample_index in range(sample_count + 1):
        ratio = sample_index / sample_count
        target_distance = total_length * ratio
        while (
            segment_index < len(segment_lengths) - 1
            and target_distance
            > distance_before_segment + segment_lengths[segment_index]
        ):
            distance_before_segment += segment_lengths[segment_index]
            segment_index += 1
        segment_length = segment_lengths[segment_index]
        if segment_length < 1e-12:
            local_ratio = 0.0
        else:
            local_ratio = (
                target_distance - distance_before_segment
            ) / segment_length
        first = source_points[segment_index]
        second = source_points[segment_index + 1]
        pose = _interpolate_pose(first, second, local_ratio)
        tangent = _normalize_vector(
            (
                second.position.x - first.position.x,
                second.position.y - first.position.y,
                second.position.z - first.position.z,
            )
        )
        preferred = axis_vectors[transverse_axis]
        if transverse_axis.startswith("tool_"):
            preferred = _quaternion_rotate(pose.orientation, preferred)
        dot = sum(a * b for a, b in zip(preferred, tangent))
        transverse = tuple(
            preferred[index] - dot * tangent[index] for index in range(3)
        )
        try:
            transverse = _normalize_vector(transverse)
        except ValueError:
            fallback = min(
                axis_vectors.values(),
                key=lambda axis: abs(
                    sum(a * b for a, b in zip(axis, tangent))
                ),
            )
            fallback_dot = sum(
                a * b for a, b in zip(fallback, tangent)
            )
            transverse = _normalize_vector(
                tuple(
                    fallback[index] - fallback_dot * tangent[index]
                    for index in range(3)
                )
            )
        offset = amplitude * math.sin(2.0 * math.pi * cycles * ratio)
        pose.position.x += transverse[0] * offset
        pose.position.y += transverse[1] * offset
        pose.position.z += transverse[2] * offset
        points.append(pose)
    return points


def weaving_waypoints(
    start: Pose,
    length: float,
    amplitude: float,
    cycles: int,
    samples_per_cycle: int,
):
    """Generate a World-Y seam with sinusoidal World-Z weaving."""
    values = (length, amplitude)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Weave length and amplitude must be positive")
    if cycles < 1 or samples_per_cycle < 4:
        raise ValueError("Weave requires cycles >= 1 and samples/cycle >= 4")
    sample_count = cycles * samples_per_cycle
    points = []
    for index in range(sample_count + 1):
        ratio = index / sample_count
        pose = Pose()
        pose.position.x = start.position.x
        pose.position.y = start.position.y + length * ratio
        pose.position.z = (
            start.position.z
            + amplitude * math.sin(2.0 * math.pi * cycles * ratio)
        )
        pose.orientation = start.orientation
        points.append(pose)
    return points


def scale_trajectory_speed(trajectory, velocity_scale: float):
    """Scale RobotTrajectory timing, velocity, and acceleration in place."""
    if (
        not math.isfinite(velocity_scale)
        or velocity_scale <= 0.0
        or velocity_scale > 1.0
    ):
        raise ValueError("Velocity scale must be in the range (0.0, 1.0]")
    joint_trajectory = trajectory.joint_trajectory
    for point in joint_trajectory.points:
        duration = (
            float(point.time_from_start.sec)
            + float(point.time_from_start.nanosec) * 1e-9
        )
        scaled_duration = duration / velocity_scale
        point.time_from_start.sec = int(scaled_duration)
        point.time_from_start.nanosec = int(
            round((scaled_duration - int(scaled_duration)) * 1e9)
        )
        if point.time_from_start.nanosec >= 1_000_000_000:
            point.time_from_start.sec += 1
            point.time_from_start.nanosec -= 1_000_000_000
        point.velocities = [
            value * velocity_scale for value in point.velocities
        ]
        point.accelerations = [
            value * velocity_scale * velocity_scale
            for value in point.accelerations
        ]
    return trajectory


def trajectory_duration_seconds(trajectory):
    """Return the final JointTrajectory timestamp in seconds."""
    points = trajectory.joint_trajectory.points
    if not points:
        return 0.0
    stamp = points[-1].time_from_start
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def cartesian_path_length(waypoints):
    """Return translational polyline length in metres."""
    return sum(
        math.sqrt(
            (second.position.x - first.position.x) ** 2
            + (second.position.y - first.position.y) ** 2
            + (second.position.z - first.position.z) ** 2
        )
        for first, second in zip(waypoints, waypoints[1:])
    )


def scale_trajectory_to_tcp_speed(trajectory, waypoints, tcp_speed_m_s):
    """Scale timing to a requested average TCP speed without speeding past limits.

    Returns ``(applied_scale, achieved_speed_m_s)``. A target faster than the
    MoveIt-produced trajectory is capped at the original safe trajectory.
    """
    target = float(tcp_speed_m_s)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("TCP speed must be greater than zero")
    length = cartesian_path_length(waypoints)
    duration = trajectory_duration_seconds(trajectory)
    if length <= 1e-9:
        raise ValueError("TCP speed mode needs a translational path")
    if duration <= 1e-9:
        raise ValueError("MoveIt trajectory has no usable timing")
    desired_duration = length / target
    applied_scale = min(1.0, duration / desired_duration)
    scale_trajectory_speed(trajectory, applied_scale)
    achieved_duration = duration / applied_scale
    return applied_scale, length / achieved_duration
