import math
import copy

from geometry_msgs.msg import Pose
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from shape_msgs.msg import SolidPrimitive


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
    # Weave generation calls this five times per output waypoint, so the
    # generator expressions this used to build (one for the sum, one for the
    # result tuple) were 22% of the time to generate a weave path.  The
    # 3- and 4-element cases are unrolled; both still divide by ``norm``
    # rather than multiplying by its reciprocal, so results are unchanged
    # bit for bit.
    length = len(vector)
    if length == 3:
        x, y, z = vector
        norm = math.sqrt(x * x + y * y + z * z)
        if norm < 1e-12:
            raise ValueError("Cannot normalize a zero-length vector")
        return (x / norm, y / norm, z / norm)
    if length == 4:
        x, y, z, w = vector
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            raise ValueError("Cannot normalize a zero-length vector")
        return (x / norm, y / norm, z / norm, w / norm)
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
    ax, ay, az, aw = _normalize_vector((first.x, first.y, first.z, first.w))
    bx, by, bz, bw = _normalize_vector((second.x, second.y, second.z, second.w))
    dot = ax * bx + ay * by + az * bz + aw * bw
    if dot < 0.0:
        bx, by, bz, bw = -bx, -by, -bz, -bw
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _normalize_vector((
            ax + (bx - ax) * ratio,
            ay + (by - ay) * ratio,
            az + (bz - az) * ratio,
            aw + (bw - aw) * ratio,
        ))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    first_scale = math.sin((1.0 - ratio) * theta) / sin_theta
    second_scale = math.sin(ratio * theta) / sin_theta
    return (
        first_scale * ax + second_scale * bx,
        first_scale * ay + second_scale * by,
        first_scale * az + second_scale * bz,
        first_scale * aw + second_scale * bw,
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
    transverse_vector=None,
    pattern="sine",
):
    """Resample a seam and add sine or forward-bulged crescent weave.

    ``transverse_vector`` is a World-frame direction supplied by a sensed
    geometry pipeline.  When present it takes priority over the generic
    tool/world-axis selector, while manual paths retain the old behavior.
    """
    if len(source_points) < 2:
        raise ValueError("Teach at least two source points before weaving")
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("Weave amplitude must be positive and finite")
    if cycles < 1 or samples_per_cycle < 4:
        raise ValueError("Weave requires cycles >= 1 and samples/cycle >= 4")
    if pattern not in ("sine", "crescent"):
        raise ValueError("Weave pattern must be sine or crescent")
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

    pitch_m = total_length / cycles
    # Two forward-facing half-moons per cycle. The bound keeps seam progress
    # strictly increasing: ds/dphase = pitch/(2*pi) + bulge*sin(2*phase).
    crescent_bulge_m = min(0.5 * amplitude, pitch_m / (4.0 * math.pi))

    sample_count = cycles * samples_per_cycle
    points = []
    segment_index = 0
    distance_before_segment = 0.0
    for sample_index in range(sample_count + 1):
        ratio = sample_index / sample_count
        phase = 2.0 * math.pi * cycles * ratio
        target_distance = total_length * ratio
        if pattern == "crescent":
            target_distance += 0.5 * crescent_bulge_m * (1.0 - math.cos(2.0 * phase))
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
        preferred = transverse_vector
        if preferred is None:
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
        offset = amplitude * math.sin(phase)
        pose.position.x += transverse[0] * offset
        pose.position.y += transverse[1] * offset
        pose.position.z += transverse[2] * offset
        points.append(pose)
    return points


def weave_cycles_for_pitch(length_m, pitch_mm, max_cycles=100):
    """Use whole cycles so the weave starts and ends on its centerline.

    The requested pitch is a maximum. The actual pitch is length / cycles.
    """
    if not math.isfinite(length_m) or length_m <= 0.0:
        raise ValueError("Weave seam length must be positive and finite")
    if not math.isfinite(pitch_mm) or not 0.1 <= pitch_mm <= 100.0:
        raise ValueError("Weave pitch must be in 0.1..100 mm/cycle")
    cycles = max(1, math.ceil(length_m * 1000.0 / pitch_mm - 1e-9))
    if cycles > max_cycles:
        raise ValueError(f"Weave requires {cycles} cycles; increase pitch (max {max_cycles})")
    return cycles


def sine_weaving_with_dwell(
    source_points, amplitude, cycles, left_dwell_s=0.0, right_dwell_s=0.0,
    transverse_axis="tool_y", transverse_vector=None, pattern="sine",
):
    """Generate a smooth ±amplitude sine/crescent with peak holds."""
    if not all(math.isfinite(v) and 0.0 <= v <= 10.0 for v in
               (left_dwell_s, right_dwell_s)):
        raise ValueError("Weave dwell must be in 0..10 seconds")
    # Twelve samples put exact points at the ±A peaks, while retaining
    # intermediate samples to round the turns rather than form a zigzag.
    samples_per_cycle = 12
    points = weaving_from_path(
        source_points, amplitude, cycles, samples_per_cycle,
        transverse_axis, transverse_vector, pattern,
    )
    points[0] = copy.deepcopy(source_points[0])
    points[-1] = copy.deepcopy(source_points[-1])
    holds = [0.0] * len(points)
    for cycle in range(cycles):
        holds[cycle * samples_per_cycle + 3] = left_dwell_s
        holds[cycle * samples_per_cycle + 9] = right_dwell_s
    return points, holds


def circular_weaving_from_path(
    source_points,
    radius,
    cycles,
    samples_per_cycle,
    radial_axis="tool_y",
    radial_vector=None,
):
    """Resample a seam and orbit its centerline with a circular TCP weave.

    The selected radial axis is projected perpendicular to the local path
    tangent; the second radial axis is derived orthogonally.  A one-cycle
    smooth ramp at each end brings the path onto/off the circle at the seam
    centerline, avoiding a discontinuous jump from a straight lead-in/out.
    TCP orientation follows the unmodified source path.
    """
    if len(source_points) < 2:
        raise ValueError("Teach at least two source points before weaving")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Circle weave radius must be positive and finite")
    if cycles < 1 or samples_per_cycle < 8:
        raise ValueError(
            "Circle weave requires cycles >= 1 and samples/cycle >= 8"
        )
    axis_vectors = {
        "tool_x": (1.0, 0.0, 0.0),
        "tool_y": (0.0, 1.0, 0.0),
        "tool_z": (0.0, 0.0, 1.0),
        "world_x": (1.0, 0.0, 0.0),
        "world_y": (0.0, 1.0, 0.0),
        "world_z": (0.0, 0.0, 1.0),
    }
    if radial_axis not in axis_vectors:
        raise ValueError(f"Unsupported circle-weave radial axis: {radial_axis}")

    segment_lengths = []
    for first, second in zip(source_points[:-1], source_points[1:]):
        segment_lengths.append(math.sqrt(
            (second.position.x - first.position.x) ** 2
            + (second.position.y - first.position.y) ** 2
            + (second.position.z - first.position.z) ** 2
        ))
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
        local_ratio = (
            0.0 if segment_length < 1e-12
            else (target_distance - distance_before_segment) / segment_length
        )
        first = source_points[segment_index]
        second = source_points[segment_index + 1]
        pose = _interpolate_pose(first, second, local_ratio)
        tangent = _normalize_vector((
            second.position.x - first.position.x,
            second.position.y - first.position.y,
            second.position.z - first.position.z,
        ))
        preferred = radial_vector
        if preferred is None:
            preferred = axis_vectors[radial_axis]
            if radial_axis.startswith("tool_"):
                preferred = _quaternion_rotate(pose.orientation, preferred)
        projection = sum(a * b for a, b in zip(preferred, tangent))
        primary = tuple(
            preferred[index] - projection * tangent[index]
            for index in range(3)
        )
        try:
            primary = _normalize_vector(primary)
        except ValueError:
            fallback = min(
                axis_vectors.values(),
                key=lambda axis: abs(sum(a * b for a, b in zip(axis, tangent))),
            )
            fallback_projection = sum(
                a * b for a, b in zip(fallback, tangent)
            )
            primary = _normalize_vector(tuple(
                fallback[index] - fallback_projection * tangent[index]
                for index in range(3)
            ))
        secondary = _normalize_vector((
            tangent[1] * primary[2] - tangent[2] * primary[1],
            tangent[2] * primary[0] - tangent[0] * primary[2],
            tangent[0] * primary[1] - tangent[1] * primary[0],
        ))
        phase_cycles = cycles * ratio
        ramp = min(1.0, phase_cycles, cycles - phase_cycles)
        envelope = math.sin(0.5 * math.pi * max(0.0, ramp)) ** 2
        phase = 2.0 * math.pi * phase_cycles
        primary_offset = radius * envelope * math.cos(phase)
        secondary_offset = radius * envelope * math.sin(phase)
        for axis, first_component, second_component in zip(
            ("x", "y", "z"), primary, secondary
        ):
            setattr(
                pose.position,
                axis,
                getattr(pose.position, axis)
                + primary_offset * first_component
                + secondary_offset * second_component,
            )
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


def retime_trajectory_constant_velocity(
    trajectory, ramp_fraction=0.2, ramp_duration_s=None
):
    """Reshape a RobotTrajectory's timing into a constant-velocity trapezoid.

    Positions and point order are untouched; only timing (and the
    velocities/accelerations derived from it) change, in place. This trades
    MoveIt's jerk-limited S-curve (smooth but always accelerating/
    decelerating) for a cruise at constant speed bounded by linear ramps.
    Positions and waypoints remain unchanged.

    By default, ``ramp_fraction`` is the fraction of the total duration spent
    accelerating and, symmetrically, decelerating (each end), while preserving
    the input duration.

    When ``ramp_duration_s`` is provided, the input average path speed becomes
    the constant cruise speed.  A fixed ramp of that duration is added at each
    end; together the two half-speed ramps add one ramp duration to the total
    trajectory time.  This mode is intended for a physical TCP-speed target:
    the target is the steady welding speed, not the whole-path average.
    """
    if not 0.0 < ramp_fraction <= 0.5:
        raise ValueError("ramp_fraction must be in (0.0, 0.5]")
    points = trajectory.joint_trajectory.points
    if len(points) < 2:
        return trajectory

    total_time = trajectory_duration_seconds(trajectory)
    if total_time <= 1e-9:
        return trajectory

    # Cumulative joint-space Euclidean distance as a path-progress metric.
    # This needs no forward kinematics and stays monotonic along the path.
    distances = [0.0]
    for previous, current in zip(points, points[1:]):
        step = math.sqrt(sum(
            (b - a) ** 2
            for a, b in zip(previous.positions, current.positions)
        ))
        distances.append(distances[-1] + step)
    total_distance = distances[-1]
    if total_distance <= 1e-9:
        return trajectory

    if ramp_duration_s is None:
        ramp_time = ramp_fraction * total_time
        cruise_speed = total_distance / (total_time - ramp_time)
    else:
        requested_ramp_time = float(ramp_duration_s)
        if not math.isfinite(requested_ramp_time) or requested_ramp_time <= 0.0:
            raise ValueError("ramp_duration_s must be finite and greater than zero")
        nominal_duration = total_time
        ramp_time = min(requested_ramp_time, nominal_duration)
        cruise_speed = total_distance / nominal_duration
        total_time = nominal_duration + ramp_time
    ramp_distance = 0.5 * cruise_speed * ramp_time

    def time_for_distance(distance):
        if distance <= ramp_distance:
            return math.sqrt(2.0 * ramp_time * distance / cruise_speed)
        if distance >= total_distance - ramp_distance:
            remaining = total_distance - distance
            return total_time - math.sqrt(
                2.0 * ramp_time * remaining / cruise_speed
            )
        return ramp_time + (distance - ramp_distance) / cruise_speed

    new_times = [time_for_distance(distance) for distance in distances]
    for index in range(1, len(new_times)):
        if new_times[index] < new_times[index - 1]:
            new_times[index] = new_times[index - 1]

    for point, t in zip(points, new_times):
        point.time_from_start.sec = int(t)
        point.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
        if point.time_from_start.nanosec >= 1_000_000_000:
            point.time_from_start.sec += 1
            point.time_from_start.nanosec -= 1_000_000_000

    joint_count = len(points[0].positions)
    last = len(points) - 1

    def finite_difference(values, index):
        if index == 0 or index == last:
            return [0.0] * joint_count
        dt = new_times[index + 1] - new_times[index - 1]
        if dt <= 1e-9:
            return [0.0] * joint_count
        return [
            (values[index + 1][joint] - values[index - 1][joint]) / dt
            for joint in range(joint_count)
        ]

    positions = [list(point.positions) for point in points]
    velocities = [
        finite_difference(positions, index) for index in range(len(points))
    ]
    accelerations = [
        finite_difference(velocities, index) for index in range(len(points))
    ]
    for point, velocity, acceleration in zip(points, velocities, accelerations):
        point.velocities = velocity
        point.accelerations = acceleration
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


def midpoint_pose(first, second):
    """Return the 1:1 internal division point, keeping the first TCP attitude."""
    if not pose_is_valid(first) or not pose_is_valid(second):
        raise ValueError("both touch poses must be valid")
    result = copy.deepcopy(first)
    result.position.x = (first.position.x + second.position.x) * 0.5
    result.position.y = (first.position.y + second.position.y) * 0.5
    result.position.z = (first.position.z + second.position.z) * 0.5
    return result


def quaternion_angular_distance(first, second):
    first_q = (first.x, first.y, first.z, first.w)
    second_q = (second.x, second.y, second.z, second.w)
    first_norm = math.sqrt(sum(value * value for value in first_q))
    second_norm = math.sqrt(sum(value * value for value in second_q))
    if first_norm < 1e-12 or second_norm < 1e-12:
        return math.inf
    dot = abs(sum(
        a * b / (first_norm * second_norm)
        for a, b in zip(first_q, second_q)
    ))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def named_tcp_linear_waypoints(start, goal):
    """Sample a named TCP transition with linear XYZ and orientation SLERP."""
    distance = math.sqrt(sum(
        (getattr(goal.position, axis) - getattr(start.position, axis)) ** 2
        for axis in ("x", "y", "z")
    ))
    angle = quaternion_angular_distance(start.orientation, goal.orientation)
    count = max(
        2,
        math.ceil(distance / 0.005) + 1,
        math.ceil(angle / math.radians(2.0)) + 1,
    )
    return linear_pose_waypoints(start, goal, count)


def pose_with_rpy_offset(pose, roll, pitch, yaw, reference="tool"):
    """Apply an RPY orientation offset about either tool or World axes."""
    result = copy.deepcopy(pose)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    offset = (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
    original = (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    reference = str(reference).strip().lower()
    if reference == "tool":
        first, second = original, offset
    elif reference == "world":
        first, second = offset, original
    else:
        raise ValueError("RPY reference must be 'tool' or 'world'")
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    composed = (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )
    norm = math.sqrt(sum(value * value for value in composed))
    if norm < 1e-12:
        raise ValueError("RPY adjustment produced an invalid orientation")
    (
        result.orientation.x,
        result.orientation.y,
        result.orientation.z,
        result.orientation.w,
    ) = (value / norm for value in composed)
    return result


def _quaternion_rotate_vector(orientation, vector):
    """Rotate a 3-vector by a geometry_msgs quaternion."""
    q = (
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
        float(orientation.w),
    )
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-12:
        raise ValueError("orientation quaternion has near-zero length")
    qx, qy, qz, qw = (value / norm for value in q)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def transform_xyz(transform, xyz):
    """Transform one finite XYZ point using geometry_msgs/TransformStamped."""
    values = tuple(float(value) for value in xyz)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("Wide Sensing point must contain three finite values")
    rotated = _quaternion_rotate_vector(
        transform.transform.rotation,
        values,
    )
    translation = transform.transform.translation
    return (
        rotated[0] + float(translation.x),
        rotated[1] + float(translation.y),
        rotated[2] + float(translation.z),
    )


def tcp_position_is_valid(target_pose):
    return target_pose is not None and all(
        math.isfinite(float(getattr(target_pose.position, axis)))
        for axis in ("x", "y", "z")
    )


def position_only_goal_constraints(planning_group, target_pose, tolerance=0.001):
    """Build a World-frame TCP position goal without orientation constraints."""
    if not tcp_position_is_valid(target_pose):
        raise ValueError("position-only target XYZ is invalid")
    if tolerance <= 0.0:
        raise ValueError("position-only tolerance must be positive")
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.SPHERE
    primitive.dimensions = [float(tolerance)]
    center = Pose()
    center.position = copy.deepcopy(target_pose.position)
    center.orientation.w = 1.0
    position = PositionConstraint()
    position.header.frame_id = "World"
    position.link_name = tip_link_for_group(planning_group)
    position.constraint_region.primitives = [primitive]
    position.constraint_region.primitive_poses = [center]
    position.weight = 1.0
    constraints = Constraints()
    constraints.position_constraints.append(position)
    return constraints


def tcp_pose_goal_constraints(
    planning_group,
    target_pose,
    position_tolerance=0.001,
    orientation_tolerance=0.01,
):
    """Use corrected XYZ together with the orientation captured for this pose."""
    if not pose_is_valid(target_pose):
        raise ValueError("complete TCP target pose is invalid")
    constraints = position_only_goal_constraints(
        planning_group, target_pose, position_tolerance
    )
    orientation = OrientationConstraint()
    orientation.header.frame_id = "World"
    orientation.link_name = tip_link_for_group(planning_group)
    orientation.orientation = copy.deepcopy(target_pose.orientation)
    orientation.absolute_x_axis_tolerance = float(orientation_tolerance)
    orientation.absolute_y_axis_tolerance = float(orientation_tolerance)
    orientation.absolute_z_axis_tolerance = float(orientation_tolerance)
    orientation.weight = 1.0
    constraints.orientation_constraints.append(orientation)
    return constraints


def pose_with_local_rpy_offset(pose, roll, pitch, yaw):
    """Backward-compatible helper for a tool-frame RPY adjustment."""
    return pose_with_rpy_offset(pose, roll, pitch, yaw, "tool")


WELD_WEAVE_SAMPLES_PER_CYCLE = 12


def weld_weave_geometry(
    seam_start, seam_goal, pattern, amplitude_mm, pitch_mm, axis,
    left_dwell_s=0.0, right_dwell_s=0.0, transverse_vector=None,
):
    """Derive whole cycles from pitch and build one consistent weld weave."""
    seam_length = math.sqrt(sum(
        (getattr(seam_goal.position, component)
         - getattr(seam_start.position, component)) ** 2
        for component in ("x", "y", "z")
    ))
    cycles = weave_cycles_for_pitch(seam_length, float(pitch_mm))
    amplitude = float(amplitude_mm) * 0.001
    if not math.isfinite(amplitude) or not 0.0001 <= amplitude <= 0.05:
        raise ValueError("Weave one-side amplitude must be in 0.1..50 mm")
    if pattern in ("sine", "crescent"):
        points, holds = sine_weaving_with_dwell(
            (seam_start, seam_goal), amplitude, cycles,
            float(left_dwell_s), float(right_dwell_s), axis,
            transverse_vector, pattern,
        )
    elif pattern == "circle":
        if float(left_dwell_s) != 0.0 or float(right_dwell_s) != 0.0:
            raise ValueError("Circle weave has no left/right peaks; set dwell to 0 or use sine")
        points = circular_weaving_from_path(
            (seam_start, seam_goal), amplitude, cycles,
            WELD_WEAVE_SAMPLES_PER_CYCLE, axis, transverse_vector,
        )
        holds = [0.0] * len(points)
    else:
        raise ValueError("Weld weave pattern must be sine, crescent or circle")
    return points, holds, cycles, seam_length * 1000.0 / cycles


def weave_path_speed_m_s(seam_length_m, path_length_m, travel_mm_s, holds):
    """Nominal TCP speed for requested average seam advance, including dwell."""
    target_s = seam_length_m / (travel_mm_s * 0.001)
    moving_s = target_s - sum(holds)
    if moving_s <= 0.0:
        raise ValueError(
            "Weave dwell exceeds travel-time target; lower dwell or seam speed"
        )
    if path_length_m <= 0.0 or seam_length_m <= 0.0:
        raise ValueError("Weave path has no usable travel distance")
    return path_length_m / moving_s


def validated_seam_speed_factor(value):
    """Accept harmless round-off above one, but reject invalid geometry."""
    factor = float(value)
    if not math.isfinite(factor) or factor <= 0.0 or factor > 1.0 + 1e-9:
        raise ValueError("invalid path/seam speed factor")
    return min(factor, 1.0)
