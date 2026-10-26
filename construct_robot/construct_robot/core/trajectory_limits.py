"""Bound joint derivatives of the quintic splines sent to ros2_control."""

import math

import numpy as np

from construct_robot.core.cartesian_path_common import scale_trajectory_speed


DERIVATIVES = ("velocity", "acceleration", "jerk")


def _polynomial_peak(coefficients):
    derivative = np.polynomial.polynomial.polyder(coefficients)
    roots = np.polynomial.polynomial.polyroots(derivative)
    samples = [0.0, 1.0]
    samples.extend(float(root.real) for root in roots
                   if abs(root.imag) < 1e-8 and 0.0 < root.real < 1.0)
    return max(abs(float(np.polynomial.polynomial.polyval(value, coefficients)))
               for value in samples)


def joint_spline_peaks(trajectory):
    """Exact segment extrema for position/velocity/acceleration quintic interpolation.

    Bounds include both sides of every knot, rather than only the derivatives
    stored at the waypoints. They describe the commanded spline, not measured
    robot tracking or the controller's bridge from its current state.
    """
    joint_trajectory = trajectory.joint_trajectory
    names = list(joint_trajectory.joint_names)
    points = joint_trajectory.points
    if not names or len(set(names)) != len(names) or len(points) < 2:
        raise ValueError("Joint spline needs unique joint names and at least two points")
    times = []
    for point in points:
        for field in (point.positions, point.velocities, point.accelerations):
            if len(field) != len(names) or not all(math.isfinite(value) for value in field):
                raise ValueError("Joint spline needs finite position, velocity and acceleration for every joint")
        stamp = point.time_from_start
        if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
            raise ValueError("Joint spline has an invalid timestamp")
        times.append(float(stamp.sec) + float(stamp.nanosec) * 1e-9)
    if any(second <= first for first, second in zip(times, times[1:])):
        raise ValueError("Joint spline timestamps must be strictly increasing")
    peaks = {name: dict.fromkeys(DERIVATIVES, 0.0) for name in names}
    for first, second, start, end in zip(points, points[1:], times, times[1:]):
        duration = end - start
        for joint_index, name in enumerate(names):
            position = first.positions[joint_index]
            velocity = first.velocities[joint_index] * duration
            acceleration = first.accelerations[joint_index] * duration ** 2 / 2.0
            position_error = second.positions[joint_index] - position - velocity - acceleration
            velocity_error = second.velocities[joint_index] * duration - velocity - 2.0 * acceleration
            acceleration_error = second.accelerations[joint_index] * duration ** 2 - 2.0 * acceleration
            coefficients = np.array([
                position, velocity, acceleration,
                10.0 * position_error - 4.0 * velocity_error + 0.5 * acceleration_error,
                -15.0 * position_error + 7.0 * velocity_error - acceleration_error,
                6.0 * position_error - 3.0 * velocity_error + 0.5 * acceleration_error,
            ])
            for order, derivative in enumerate(DERIVATIVES, 1):
                coefficients = np.polynomial.polynomial.polyder(coefficients)
                peak = _polynomial_peak(coefficients) / duration ** order
                if not math.isfinite(peak):
                    raise ValueError(f"Non-finite {derivative} for {name}")
                peaks[name][derivative] = max(peaks[name][derivative], peak)
    return peaks


def limit_trajectory_derivatives(trajectory, limits):
    """Uniformly stretch time to meet v/a/jerk bounds; return the stretch factor.

    One factor covers the entire path. Positions, derivative continuity and
    the relative TCP speed profile are preserved. Missing or invalid limits
    and malformed trajectories fail rather than bypassing the checks.
    """
    for name in trajectory.joint_trajectory.joint_names:
        for derivative in DERIVATIVES:
            limit = limits.get(name, {}).get(derivative)
            if (isinstance(limit, bool) or not isinstance(limit, (int, float))
                    or not math.isfinite(limit) or limit <= 0.0):
                raise ValueError(f"Missing or invalid {name} {derivative} limit")
    total_stretch = 1.0
    for _attempt in range(3):
        peaks = joint_spline_peaks(trajectory)
        stretch = max(
            [1.0] + [(values[derivative] / limits[name][derivative]) ** (1.0 / order)
                     for name, values in peaks.items()
                     for order, derivative in enumerate(DERIVATIVES, 1)]
        )
        if not math.isfinite(stretch):
            raise ValueError("Joint limits require a non-finite trajectory duration")
        if stretch <= 1.0 + 1e-9:
            return total_stretch
        stretch *= 1.0 + 1e-6
        scale_trajectory_speed(trajectory, 1.0 / stretch)
        total_stretch *= stretch
    raise ValueError("Unable to satisfy joint derivative limits after time scaling")


# smoothstep s(u) = 10u^3 - 15u^4 + 6u^5: peak |s'|, |s''|, |s'''| on u in [0, 1]
_SMOOTHSTEP_PEAKS = (1.875, 10.0 / math.sqrt(3.0), 60.0)


def straight_joint_retime(positions, limits, velocity_scale, collinear_tol_rad=1e-3, dt=0.02):
    """Retime a joint-space line to the requested speed fraction.

    ``positions`` lists the planned waypoints (first = start, last = goal).
    Returns ``(times, positions, velocities, accelerations)`` for one smoothstep
    segment whose v/a/jerk peaks are at most ``velocity_scale`` times ``limits``,
    or ``None`` when the planned path leaves the start-goal line (so the
    planner's own timing must be kept). The geometry is unchanged: only a
    path that is already that line is retimed.
    """
    if not 0.0 < velocity_scale <= 1.0 or len(positions) < 2:
        return None
    path = np.asarray(positions, dtype=float)
    start, goal = path[0], path[-1]
    delta = goal - start
    span = float(np.linalg.norm(delta))
    if span < 1e-6:
        return None
    direction = delta / span
    offsets = path - start
    along = offsets @ direction
    if float(np.max(np.linalg.norm(offsets - np.outer(along, direction), axis=1))) > collinear_tol_rad:
        return None
    if np.any(np.diff(along) < -collinear_tol_rad):
        return None
    duration = 0.0
    for index, joint_delta in enumerate(np.abs(delta)):
        joint = limits[index]
        for order, (peak, key) in enumerate(zip(_SMOOTHSTEP_PEAKS, DERIVATIVES), 1):
            limit = float(joint[key]) * velocity_scale
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError(f"Invalid joint {key} limit")
            duration = max(duration, (peak * joint_delta / limit) ** (1.0 / order))
    count = max(2, int(math.ceil(duration / dt)) + 1)
    times = np.linspace(0.0, duration, count)
    u = times / duration
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
    dds = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
    return (times, start + np.outer(s, delta), np.outer(ds, delta), np.outer(dds, delta))
