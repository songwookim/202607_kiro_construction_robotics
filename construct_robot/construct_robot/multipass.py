"""Cumulative four-pass registration; no GUI or robot side effects."""
import copy
import math

from construct_robot.cartesian_path_common import pose_is_valid
from construct_robot.seam_geometry import (
    _pose_position_tuple,
    _unit_vector,
    _vector_cross,
    _vector_dot,
    seam_direction,
)


def _minimal_direction_rotation(old_direction, new_direction, maximum_degrees=30.0):
    """Return the unique minimal rotation quaternion from old to new direction."""
    old_direction = _unit_vector(old_direction, "old seam direction")
    new_direction = _unit_vector(new_direction, "measured seam direction")
    cosine = max(-1.0, min(1.0, _vector_dot(old_direction, new_direction)))
    angle = math.acos(cosine)
    if angle > math.radians(float(maximum_degrees)):
        raise ValueError(
            f"Seam direction changed {math.degrees(angle):.1f}° "
            f"(limit {float(maximum_degrees):.1f}°)"
        )
    cross = _vector_cross(old_direction, new_direction)
    sine = math.sqrt(_vector_dot(cross, cross))
    if sine <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0), math.degrees(angle)
    axis = tuple(value / sine for value in cross)
    half = angle * 0.5
    scale = math.sin(half)
    return (
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
        math.cos(half),
    ), math.degrees(angle)


def _rotate_vector_by_quaternion(vector, quaternion):
    qx, qy, qz, qw = (float(value) for value in quaternion)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _rotate_pose_orientation_left(pose, rotation):
    """Apply q_rotation * q_pose without introducing seam-axis roll."""
    rx, ry, rz, rw = rotation
    px = float(pose.orientation.x)
    py = float(pose.orientation.y)
    pz = float(pose.orientation.z)
    pw = float(pose.orientation.w)
    values = (
        rw * px + rx * pw + ry * pz - rz * py,
        rw * py - rx * pz + ry * pw + rz * px,
        rw * pz + rx * py - ry * px + rz * pw,
        rw * pw - rx * px - ry * py - rz * pz,
    )
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        raise ValueError("Rotated welding orientation is invalid")
    result = copy.deepcopy(pose)
    (
        result.orientation.x,
        result.orientation.y,
        result.orientation.z,
        result.orientation.w,
    ) = (value / norm for value in values)
    return result


def correct_remaining_passes(
    passes, anchor_pass, measured_start, measured_goal, maximum_degrees=30.0
):
    """Cumulatively register one pass and propagate only to later passes.

    ``passes`` is the current working state, not the immutable source logs.
    Earlier passes remain byte-for-byte independent deep copies. START and
    GOAL offsets use their respective measured anchor, while one minimal
    direction rotation updates offsets and welding attitudes.
    """
    if set(passes) != {1, 2, 3, 4}:
        raise ValueError("Exactly four current pass states are required")
    anchor_pass = int(anchor_pass)
    if anchor_pass not in passes:
        raise ValueError("Anchor pass must be 1, 2, 3, or 4")
    if not all(
        pose is not None and pose_is_valid(pose)
        for pose in (measured_start, measured_goal)
    ):
        raise ValueError("Measured START and GOAL are required")
    current = passes[anchor_pass]
    old_start = current["start"]
    old_goal = current["goal"]
    old_direction = seam_direction(old_start, old_goal)
    new_direction = seam_direction(measured_start, measured_goal)
    rotation, angle_degrees = _minimal_direction_rotation(
        old_direction, new_direction, maximum_degrees
    )
    corrected = copy.deepcopy(passes)
    old_start_xyz = _pose_position_tuple(old_start)
    old_goal_xyz = _pose_position_tuple(old_goal)
    measured_start_xyz = _pose_position_tuple(measured_start)
    measured_goal_xyz = _pose_position_tuple(measured_goal)

    # The selected pass is directly taught, not transformed. Preserve its
    # WAIT poses and use the captured TCP positions AND orientations exactly.
    corrected[anchor_pass]["start"] = copy.deepcopy(measured_start)
    corrected[anchor_pass]["goal"] = copy.deepcopy(measured_goal)
    for number in range(anchor_pass + 1, 5):
        for endpoint, old_anchor_xyz, new_anchor_xyz in (
            ("start_wait", old_start_xyz, measured_start_xyz),
            ("start", old_start_xyz, measured_start_xyz),
            ("goal_wait", old_goal_xyz, measured_goal_xyz),
            ("goal", old_goal_xyz, measured_goal_xyz),
        ):
            current_pose = passes[number][endpoint]
            offset = tuple(
                value - old_anchor_xyz[index]
                for index, value in enumerate(_pose_position_tuple(current_pose))
            )
            rotated_offset = _rotate_vector_by_quaternion(offset, rotation)
            result = _rotate_pose_orientation_left(current_pose, rotation)
            result.position.x, result.position.y, result.position.z = tuple(
                new_anchor_xyz[index] + rotated_offset[index]
                for index in range(3)
            )
            corrected[number][endpoint] = result

    metadata = {
        "anchor_pass": anchor_pass,
        "direction_change_deg": angle_degrees,
        "rotation_xyzw": tuple(float(value) for value in rotation),
        "start_translation_m": tuple(
            measured_start_xyz[index] - old_start_xyz[index]
            for index in range(3)
        ),
        "goal_translation_m": tuple(
            measured_goal_xyz[index] - old_goal_xyz[index]
            for index in range(3)
        ),
        "later_passes_updated": list(range(anchor_pass + 1, 5)),
    }
    return corrected, metadata


def correct_four_pass_references(references, corrected_root_start, corrected_root_goal):
    """Compatibility wrapper for a Pass-1 sequential registration."""
    current = {
        number: {
            endpoint: copy.deepcopy(reference[endpoint])
            for endpoint in ("start_wait", "start", "goal_wait", "goal")
        }
        for number, reference in references.items()
    }
    corrected, metadata = correct_remaining_passes(
        current, 1, corrected_root_start, corrected_root_goal
    )
    return corrected, metadata["direction_change_deg"]


def correct_seam_from_measured_start(predicted_start, predicted_goal, measured_start):
    """Translate a predicted seam by one physically measured START.

    A START-only measurement cannot observe seam yaw or length. Preserve the
    predicted START-to-GOAL vector and both logged welding attitudes, and apply
    only the measured XYZ translation to the complete seam.
    """
    if not all(
        pose is not None and pose_is_valid(pose)
        for pose in (predicted_start, predicted_goal, measured_start)
    ):
        raise ValueError("Predicted START/GOAL and measured START are required")
    if math.dist(
        _pose_position_tuple(predicted_start),
        _pose_position_tuple(predicted_goal),
    ) < 0.001:
        raise ValueError("Predicted seam is shorter than 1 mm")
    delta = tuple(
        measured - predicted
        for measured, predicted in zip(
            _pose_position_tuple(measured_start),
            _pose_position_tuple(predicted_start),
        )
    )
    corrected_start = copy.deepcopy(predicted_start)
    corrected_goal = copy.deepcopy(predicted_goal)
    for index, axis in enumerate(("x", "y", "z")):
        setattr(
            corrected_start.position,
            axis,
            getattr(measured_start.position, axis),
        )
        setattr(
            corrected_goal.position,
            axis,
            getattr(predicted_goal.position, axis) + delta[index],
        )
    return corrected_start, corrected_goal, delta
