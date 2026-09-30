"""Keyboard TCP-jog velocity mapping, independent of Tk and ROS."""

import math

from construct_robot.core.cartesian_path_common import _quaternion_rotate_vector



KEYBOARD_JOG_SELECTIONS = {
    "X": (0,),
    "Y": (1,),
    "Z": (2,),
    "RX": (3,),
    "RY": (4,),
    "RZ": (5,),
    "XY": (0, 1),
    "XZ": (0, 2),
    "YZ": (1, 2),
    "RX/RY": (3, 4),
    "RX/RZ": (3, 5),
    "RY/RZ": (4, 5),
}


def keyboard_jog_velocity(selection, direction, linear_speed, angular_speed):
    """Build a signed 6D keyboard-axis vector using linear/angular magnitudes."""
    axes = KEYBOARD_JOG_SELECTIONS.get(str(selection))
    if axes is None:
        raise ValueError(f"unsupported keyboard jog selection: {selection}")
    direction = str(direction)
    if len(axes) == 1:
        if direction not in ("Left", "Right", "Up", "Down"):
            raise ValueError(f"unsupported keyboard jog direction: {direction}")
        axis = axes[0]
        sign = 1.0 if direction in ("Right", "Up") else -1.0
    else:
        mapping = {
            "Left": (axes[0], -1.0),
            "Right": (axes[0], 1.0),
            "Down": (axes[1], -1.0),
            "Up": (axes[1], 1.0),
        }
        if direction not in mapping:
            raise ValueError(f"unsupported keyboard jog direction: {direction}")
        axis, sign = mapping[direction]
    speed = float(angular_speed if axis >= 3 else linear_speed)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("keyboard jog speed must be positive and finite")
    velocity = [0.0] * 6
    velocity[axis] = sign * speed
    return tuple(velocity)


def keyboard_velocity_vector(
    orientation,
    selection,
    direction,
    linear_speed_m_s,
    angular_speed_rad_s,
    reference,
):
    """Use the selected frame for XYZ; rotations always follow World axes."""
    values = keyboard_jog_velocity(
        selection,
        direction,
        linear_speed_m_s,
        angular_speed_rad_s,
    )
    reference = str(reference).strip().lower()
    if reference == "world":
        world_linear = values[:3]
    elif reference == "tool":
        world_linear = _quaternion_rotate_vector(orientation, values[:3])
    else:
        raise ValueError("keyboard velocity frame must be World or Tool")
    # Previous angular mapping (retained for comparison):
    # if reference == "world":
    #     world_angular = values[3:]
    # elif reference == "tool":
    #     world_angular = _quaternion_rotate_vector(orientation, values[3:])
    # RX/RY/RZ now refer to fixed World axes, independent of TCP attitude
    # and the XYZ frame selector. resolve_keyboard_velocity still converts
    # this World vector into the robot base required by jog_robot_l(mode=1).
    world_angular = values[3:]
    return tuple(world_linear) + tuple(world_angular)


def next_keyboard_speed(current, choices):
    """Return the next discrete teaching speed, wrapping to the first."""
    values = tuple(float(value) for value in choices)
    if not values:
        raise ValueError("keyboard speed choices must not be empty")
    try:
        index = next(
            i for i, value in enumerate(values)
            if math.isclose(float(current), value, abs_tol=1e-9)
        )
    except (StopIteration, TypeError, ValueError):
        return values[0]
    return values[(index + 1) % len(values)]
