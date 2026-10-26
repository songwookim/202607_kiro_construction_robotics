"""How far the commanded TCP strays from the requested Cartesian path.

MoveIt's GetCartesianPath (2.5, Humble) only applies a *relative* joint-jump
threshold, which this stack disables; the revolute/prismatic absolute
thresholds of the request are ignored by that version.  An IK branch change
(wrist flip, full joint turn, elbow switch) between two Cartesian samples is
therefore returned as a valid plan, and time parameterization turns it into a
fast joint-space reconfiguration while the torch leaves the seam.

This module evaluates the trajectory exactly as joint_trajectory_controller
2.x interpolates it (quintic with accelerations, cubic with velocities,
otherwise linear), runs forward kinematics on dense samples and measures the
TCP position/orientation deviation from the requested polyline.  The same
numbers quantify ordinary path fidelity (time-parameterization blends and the
spline between resampled points).
"""

import math

import numpy as np


def _seconds(duration):
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def _segment_coefficients(a, b, duration):
    """Polynomial coefficients (in seconds from a) the JTC uses for one segment."""
    p0, p1 = a["positions"], b["positions"]
    if a["velocities"] is None or b["velocities"] is None:
        return [p0, (p1 - p0) / duration]
    v0, v1 = a["velocities"], b["velocities"]
    T = duration
    if a["accelerations"] is None or b["accelerations"] is None:
        return [
            p0,
            v0,
            (-3.0 * p0 + 3.0 * p1 - 2.0 * v0 * T - v1 * T) / T ** 2,
            (2.0 * p0 - 2.0 * p1 + v0 * T + v1 * T) / T ** 3,
        ]
    a0, a1 = a["accelerations"], b["accelerations"]
    return [
        p0,
        v0,
        0.5 * a0,
        (-20.0 * p0 + 20.0 * p1 - 3.0 * a0 * T ** 2 + a1 * T ** 2
         - 12.0 * v0 * T - 8.0 * v1 * T) / (2.0 * T ** 3),
        (30.0 * p0 - 30.0 * p1 + 3.0 * a0 * T ** 2 - 2.0 * a1 * T ** 2
         + 16.0 * v0 * T + 14.0 * v1 * T) / (2.0 * T ** 4),
        (-12.0 * p0 + 12.0 * p1 - a0 * T ** 2 + a1 * T ** 2
         - 6.0 * v0 * T - 6.0 * v1 * T) / (2.0 * T ** 5),
    ]


def spline_samples(trajectory, joint_names, samples_per_segment=8, joint_resolution_rad=None):
    """(times, positions[N, J]) along the controller spline, in `joint_names` order.

    With ``joint_resolution_rad`` a segment gets just enough samples (2 to
    ``samples_per_segment``) for its largest joint change; nearly stationary
    segments cannot hide a deviation between samples.
    """
    joint_trajectory = trajectory.joint_trajectory
    names = list(joint_trajectory.joint_names)
    if len(set(names)) != len(names) or not set(joint_names).issubset(names):
        raise ValueError("trajectory must contain each arm joint exactly once")
    order = [names.index(name) for name in joint_names]

    def field(values):
        if len(values) != len(names):
            return None
        array = np.asarray(values, dtype=float)[order]
        if not np.all(np.isfinite(array)):
            raise ValueError("trajectory contains a non-finite value")
        return array

    knots = []
    for point in joint_trajectory.points:
        positions = field(point.positions)
        if positions is None:
            raise ValueError("every trajectory point needs one position per joint")
        knots.append({
            "time": _seconds(point.time_from_start),
            "positions": positions,
            "velocities": field(point.velocities),
            "accelerations": field(point.accelerations),
        })
    if not knots:
        raise ValueError("trajectory has no points")
    times, positions = [knots[0]["time"]], [knots[0]["positions"]]
    for a, b in zip(knots, knots[1:]):
        duration = b["time"] - a["time"]
        if duration <= 0.0:
            if np.max(np.abs(b["positions"] - a["positions"])) > 1e-9:
                raise ValueError("trajectory moves between points with the same timestamp")
            continue
        coefficients = _segment_coefficients(a, b, duration)
        count = samples_per_segment
        if joint_resolution_rad:
            change = float(np.max(np.abs(b["positions"] - a["positions"])))
            count = max(2, min(samples_per_segment, math.ceil(change / joint_resolution_rad)))
        for index in range(1, count + 1):
            s = duration * index / count
            positions.append(sum(c * s ** power for power, c in enumerate(coefficients)))
            times.append(a["time"] + s)
    return np.asarray(times), np.asarray(positions)


def _quaternion_matrix(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_quaternion(rotation):
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                         (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                     0.25 * s, (m[1, 0] - m[0, 1]) / s])


def _slerp(first, second, ratio):
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(first @ second)
    if dot < 0.0:
        second, dot = -second, -dot
    if dot > 0.9995:
        result = first + (second - first) * ratio
        return result / np.linalg.norm(result)
    theta = math.acos(min(1.0, dot))
    return (math.sin((1.0 - ratio) * theta) * first + math.sin(ratio * theta) * second) / math.sin(theta)


def _rotation_angle(first, second):
    cosine = (np.trace(first.T @ second) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, cosine)))


def _slerp_arc_angle(first, second, rotation):
    """Rotation angle from `rotation` to the nearest attitude on SLERP(first, second)."""
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    if float(first @ second) < 0.0:
        second = -second
    sample = _matrix_quaternion(rotation)
    sample = sample / np.linalg.norm(sample)
    normal = second - float(second @ first) * first
    if np.linalg.norm(normal) < 1e-12:
        return 2.0 * math.acos(min(1.0, abs(float(sample @ first))))
    normal = normal / np.linalg.norm(normal)
    arc = math.atan2(float(second @ normal), float(second @ first))
    best = math.pi
    for sign in (1.0, -1.0):  # q and -q are the same rotation
        phase = math.atan2(sign * float(sample @ normal), sign * float(sample @ first))
        phase = min(max(phase, 0.0), arc)
        nearest = math.cos(phase) * first + math.sin(phase) * normal
        best = min(best, 2.0 * math.acos(min(1.0, abs(float(sample @ nearest)))))
    return best


def tcp_path_deviation(trajectory, joint_names, fk, waypoints, samples_per_segment=8,
                       joint_resolution_rad=0.005):
    """Largest deviation of the commanded TCP from the requested Cartesian path.

    ``fk(q)`` returns (position (3,), rotation (3, 3)) in the waypoint frame.
    Like GetCartesianPath, the requested path starts at the trajectory's
    first TCP and then visits ``waypoints`` (geometry_msgs Pose) in order,
    linearly in position with orientation SLERP.

    Returns a dict: ``position_m`` / ``orientation_rad`` (worst sample),
    ``time_s`` (where the worst position deviation occurs) and
    ``end_position_m`` / ``end_orientation_rad`` (final TCP vs final waypoint).
    """
    times, joints = spline_samples(
        trajectory, joint_names, samples_per_segment, joint_resolution_rad)
    start_position, start_rotation = fk(joints[0])
    vertices = [np.asarray(start_position, dtype=float)]
    quaternions = [_matrix_quaternion(np.asarray(start_rotation))]
    for pose in waypoints:
        vertices.append(np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float))
        o = pose.orientation
        quaternions.append(np.array([o.x, o.y, o.z, o.w], dtype=float))
    vertices = np.asarray(vertices)
    starts, ends = vertices[:-1], vertices[1:]
    if len(starts) == 0:
        starts, ends = vertices, vertices
        quaternions = quaternions + quaternions
    directions = ends - starts
    lengths_squared = np.einsum("ij,ij->i", directions, directions)

    worst_position, worst_orientation, worst_time = 0.0, 0.0, float(times[0])
    for time_s, q in zip(times, joints):
        position, rotation = fk(q)
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        offsets = position - starts
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(
                lengths_squared > 1e-18,
                np.einsum("ij,ij->i", offsets, directions) / lengths_squared,
                0.0,
            )
        ratios = np.clip(ratios, 0.0, 1.0)
        nearest = starts + ratios[:, None] * directions
        distances = np.linalg.norm(position - nearest, axis=1)
        distance = float(distances.min())
        if distance > worst_position:
            worst_position, worst_time = distance, float(time_s)
        orientation = math.inf
        for index in np.flatnonzero(distances <= distance + 1e-4):
            if lengths_squared[index] > 1e-18:
                reference = _quaternion_matrix(*_slerp(
                    quaternions[index], quaternions[index + 1], float(ratios[index])))
                angle = _rotation_angle(reference, rotation)
            else:  # rotation-only segment: any attitude along its SLERP
                angle = _slerp_arc_angle(quaternions[index], quaternions[index + 1], rotation)
            orientation = min(orientation, angle)
        worst_orientation = max(worst_orientation, orientation)

    end_position, end_rotation = fk(joints[-1])
    final = waypoints[-1] if waypoints else None
    if final is None:
        end_position_error, end_orientation_error = 0.0, 0.0
    else:
        end_position_error = float(np.linalg.norm(np.asarray(end_position) - vertices[-1]))
        end_orientation_error = _rotation_angle(
            _quaternion_matrix(*quaternions[-1]), np.asarray(end_rotation))
    return {
        "position_m": worst_position,
        "orientation_rad": worst_orientation,
        "time_s": worst_time,
        "end_position_m": end_position_error,
        "end_orientation_rad": end_orientation_error,
    }
