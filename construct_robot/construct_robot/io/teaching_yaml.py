"""Shared YAML parsing for named TCP teaching poses."""
import math
from pathlib import Path

import yaml
from geometry_msgs.msg import Pose

ARM_JOINT_NAMES = {
    arm: frozenset(
        f"{arm}_manipulator_joint{index}" for index in range(1, 7)
    )
    for arm in ("left", "right")
}


def _finite_float(value, description):
    """Return a finite float while rejecting YAML booleans and bad values."""
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a number") from error
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _pose_from_yaml_dict(data, description="pose"):
    """Reconstruct a geometry_msgs Pose from a position_m/orientation_xyzw
    mapping, the shape used by both the per-pose teaching YAML files and the
    teaching/touch snapshot embedded in weld feedback logs."""
    if not isinstance(data, dict):
        raise ValueError(f"{description} must be a mapping")
    position = data.get("position_m")
    orientation = data.get("orientation_xyzw")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise ValueError(
            f"{description} position_m and orientation_xyzw must be mappings"
        )
    pose = Pose()
    for field in ("x", "y", "z"):
        setattr(
            pose.position,
            field,
            _finite_float(position.get(field), f"{description} position {field}"),
        )
    for field in ("x", "y", "z", "w"):
        setattr(
            pose.orientation,
            field,
            _finite_float(
                orientation.get(field), f"{description} orientation {field}"
            ),
        )
    norm = math.sqrt(
        pose.orientation.x ** 2
        + pose.orientation.y ** 2
        + pose.orientation.z ** 2
        + pose.orientation.w ** 2
    )
    if norm < 1e-9:
        raise ValueError(f"{description} orientation quaternion must be non-zero")
    return pose


def load_initial_state_yaml(path):
    """Load and validate a TCP teaching YAML file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(document, dict):
        raise ValueError("YAML root must be a mapping")
    if document.get("format_version") != 1:
        raise ValueError("unsupported or missing format_version (expected 1)")

    planning_group = document.get("planning_group")
    if planning_group not in ("left_manipulator", "right_manipulator"):
        raise ValueError(
            "planning_group must be left_manipulator or right_manipulator"
        )
    arm = planning_group.removesuffix("_manipulator")

    joint_state = document.get("joint_state")
    if not isinstance(joint_state, dict):
        raise ValueError("joint_state must be a mapping")
    names = joint_state.get("names")
    positions = joint_state.get("positions_rad")
    if not isinstance(names, list) or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError("joint_state.names must be a list of joint names")
    if set(names) != ARM_JOINT_NAMES[arm] or len(names) != 6:
        raise ValueError(
            f"joint_state.names must contain the six {arm} arm joints"
        )
    if not isinstance(positions, list) or len(positions) != len(names):
        raise ValueError(
            "joint_state.positions_rad must match joint_state.names"
        )
    positions = tuple(
        _finite_float(value, f"position for {name}")
        for name, value in zip(names, positions)
    )

    tcp = _pose_from_yaml_dict(document.get("tcp_pose_world"), "TCP")
    return planning_group, tuple(names), positions, tcp
