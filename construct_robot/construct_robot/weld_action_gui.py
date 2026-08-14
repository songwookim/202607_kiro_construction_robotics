import copy
import math
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import rclpy
import yaml
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Point, Pose, PoseArray
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rbpodo_msgs.msg import SystemState
from rbpodo_msgs.srv import MoveStop, SetDigitalOutput
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from construct_msgs.action import CartesianPath
from construct_robot.cartesian_path_common import (
    circle_waypoints,
    linear_pose_waypoints,
    pose_is_valid,
    scale_trajectory_speed,
    straight_waypoints,
    tip_link_for_group,
    weaving_from_path,
)
from construct_robot.cartesian_path_server import make_weld_visualization
from construct_robot.hicomm_welder import (
    BIT_ARC,
    BIT_FORWARD,
    BIT_GAS,
    BIT_REVERSE,
    BIT_STICK,
    DIAMETER_CODES,
    GAS_CODES,
    HiCommWelderClient,
    MATERIAL_CODES,
    MODE_CODES,
    TxState,
    build_request,
)


MANUAL_IO_CANDIDATES = frozenset((0, 4, 8, 9, 10, 12, 13))
TOUCH_INPUT_PORT = 8
ARM_JOINT_NAMES = {
    arm: frozenset(
        f"{arm}_manipulator_joint{index}" for index in range(1, 7)
    )
    for arm in ("left", "right")
}
HEAD_JOINT_NAMES = frozenset((
    "robot_head_rev_joint1",
    "robot_head_rev_joint2",
))
CONTROLLED_JOINT_NAMES = {
    **ARM_JOINT_NAMES,
    "head": HEAD_JOINT_NAMES,
}
CONTROLLER_NAMES = {
    "left": "left_manipulator_controller",
    "right": "right_manipulator_controller",
    "head": "robot_head_controller",
}

CORNER_TOUCH_NAMES = (
    "start_floor",
    "start_wall",
    "goal_floor",
    "goal_wall",
)

TEACHING_POSES = {
    "robot_start": "1 · Initial pose",
    "weld_wait": "2 · Weld wait pose",
    "weld_start_wait": "3 · Weld start wait pose",
    "weld_start": "4 · Reference TCP 1 / Weld start",
    "weld_goal_wait": "5 · Weld goal wait pose",
    "weld_end": "6 · Reference TCP 2 / Weld goal",
    "weld_finish": "7 · Weld end pose",
}

# Every Named TCP Teaching execution is contact guarded.  Planning remains
# unguarded because it does not command physical motion.
DI8_GUARDED_TEACHING_POSES = frozenset(TEACHING_POSES)

# Corrected seam teaching poses combine sensed/corrected XYZ with the
# orientation originally captured for that individual named pose.
TCP_POSE_TEACHING_POSES = frozenset((
    "weld_start_wait",
    "weld_start",
    "weld_goal_wait",
    "weld_end",
    "weld_finish",
))

DIGITAL_WELD_RECIPE_KEYS = (
    "current_a",
    "voltage_tenths",
    "material",
    "diameter_mm",
    "mode",
    "gas",
    "synergic",
    "correction",
    "pre_gas_s",
    "post_gas_s",
)
DIGITAL_WELD_COMMANDS = frozenset(("set", "on", "off"))

DEFAULT_DIGITAL_WELD_SETTINGS = {
    "current_a": 100,
    "voltage_tenths": 100,
    "voltage": 10.0,
    "material": "FE-SOLID",
    "diameter_mm": 1.2,
    "mode": "LSM",
    "gas": "CO2",
    "synergic": False,
    "correction": 0.0,
    "pre_gas_s": 0.0,
    "post_gas_s": 0.0,
    # Sequence timing metadata. It is intentionally not part of TX Byte1..11.
    "preflow_seconds": 0.0,
}


def digital_weld_recipe(settings):
    """Return only the values encoded into the Hi-COMM welding frame."""
    return {key: settings[key] for key in DIGITAL_WELD_RECIPE_KEYS}


def validate_digital_weld_settings(settings):
    """Normalize and validate GUI/sequence digital-welding settings."""
    normalized = copy.deepcopy(DEFAULT_DIGITAL_WELD_SETTINGS)
    normalized.update(settings)
    normalized["current_a"] = int(round(float(normalized["current_a"])))
    normalized["voltage_tenths"] = int(round(
        float(normalized["voltage_tenths"])
    ))
    normalized["voltage"] = normalized["voltage_tenths"] / 10.0
    normalized["diameter_mm"] = float(normalized["diameter_mm"])
    normalized["synergic"] = bool(normalized["synergic"])
    for key in ("correction", "pre_gas_s", "post_gas_s", "preflow_seconds"):
        normalized[key] = float(normalized[key])
    if not 0.0 <= normalized["preflow_seconds"] <= 10.0:
        raise ValueError("pre-weld gas flow must be in 0..10 seconds")
    # build_request is the protocol's single source of range/enum validation.
    build_request(TxState(**digital_weld_recipe(normalized)))
    return normalized


def midpoint_pose(first, second):
    """Return the 1:1 internal division point, keeping the first TCP attitude."""
    if not pose_is_valid(first) or not pose_is_valid(second):
        raise ValueError("both touch poses must be valid")
    result = copy.deepcopy(first)
    result.position.x = (first.position.x + second.position.x) * 0.5
    result.position.y = (first.position.y + second.position.y) * 0.5
    result.position.z = (first.position.z + second.position.z) * 0.5
    return result


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


def seam_xy_normal(start, goal):
    """Return the +90° World-XY normal of the taught START→GOAL seam."""
    tx, ty, _tz = seam_direction(start, goal, xy_only=True)
    return (-ty, tx, 0.0)


def _pose_position_tuple(pose):
    return (pose.position.x, pose.position.y, pose.position.z)


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
    """Combine sensed XYZ with either yaw-corrected or unchanged taught attitudes."""
    normalized = str(mode).strip().lower()
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


def pose_with_local_rpy_offset(pose, roll, pitch, yaw):
    """Backward-compatible helper for a tool-frame RPY adjustment."""
    return pose_with_rpy_offset(pose, roll, pitch, yaw, "tool")


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


def save_initial_state_yaml(path, planning_group, joint_names, positions, tcp):
    """Atomically save a captured joint state and its TCP pose as YAML."""
    path = Path(path)
    document = {
        "format_version": 1,
        "planning_group": planning_group,
        "joint_state": {
            "names": list(joint_names),
            "positions_rad": [float(value) for value in positions],
        },
        "tcp_pose_world": {
            "position_m": {
                "x": float(tcp.position.x),
                "y": float(tcp.position.y),
                "z": float(tcp.position.z),
            },
            "orientation_xyzw": {
                "x": float(tcp.orientation.x),
                "y": float(tcp.orientation.y),
                "z": float(tcp.orientation.z),
                "w": float(tcp.orientation.w),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            yaml.safe_dump(document, stream, sort_keys=False)
        temporary_path.replace(path)
        # Do not report a successful teaching update unless the final target
        # file can be read back and contains exactly what was requested.
        with path.open("r", encoding="utf-8") as stream:
            persisted = yaml.safe_load(stream)
        if persisted != document:
            raise OSError(f"YAML read-back verification failed: {path}")
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_seam_touch_yaml(
    path,
    planning_group,
    seam_axis,
    touches,
    starts,
    stopped_poses=None,
    probe_configuration=None,
):
    """Atomically save raw DI8 contact and probe-start poses for diagnostics."""
    path = Path(path)

    def pose_document(pose):
        if pose is None:
            return None
        return {
            "position_m": {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
            },
            # Retained only to diagnose TCP/sensor-offset consistency.  Seam
            # geometry intentionally uses position_m only.
            "orientation_xyzw": {
                "x": float(pose.orientation.x),
                "y": float(pose.orientation.y),
                "z": float(pose.orientation.z),
                "w": float(pose.orientation.w),
            },
        }

    records = {}
    stopped_poses = stopped_poses or {}
    for name in CORNER_TOUCH_NAMES:
        contact = touches.get(name)
        start = starts.get(name)
        stopped = stopped_poses.get(name)
        if contact is None and start is None and stopped is None:
            continue
        records[name] = {
            "contact_tcp": pose_document(contact),
            "stopped_tcp": pose_document(stopped),
            "probe_start_tcp": pose_document(start),
        }
    document = {
        "format_version": 1,
        "planning_group": planning_group,
        # Keep seam_axis for compatibility with the existing diagnostic plotter.
        "seam_axis": str(seam_axis).upper(),
        "probe_configuration": copy.deepcopy(probe_configuration),
        "saved_unix_time": time.time(),
        "note": (
            "touch orientation is diagnostic only; seam XYZ comes from sensed "
            "plane intersection and seam orientation is handled separately"
        ),
        "touches": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            yaml.safe_dump(document, stream, sort_keys=False)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_initial_state_yaml(path):
    """Load and validate a TCP teaching YAML file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
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

    tcp_data = document.get("tcp_pose_world")
    if not isinstance(tcp_data, dict):
        raise ValueError("tcp_pose_world must be a mapping")
    position = tcp_data.get("position_m")
    orientation = tcp_data.get("orientation_xyzw")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise ValueError(
            "TCP position_m and orientation_xyzw must be mappings"
        )
    tcp = Pose()
    for field in ("x", "y", "z"):
        setattr(
            tcp.position,
            field,
            _finite_float(position.get(field), f"TCP position {field}"),
        )
    for field in ("x", "y", "z", "w"):
        setattr(
            tcp.orientation,
            field,
            _finite_float(orientation.get(field), f"TCP orientation {field}"),
        )
    norm = math.sqrt(
        tcp.orientation.x ** 2
        + tcp.orientation.y ** 2
        + tcp.orientation.z ** 2
        + tcp.orientation.w ** 2
    )
    if norm < 1e-9:
        raise ValueError("TCP orientation quaternion must be non-zero")
    return planning_group, tuple(names), positions, tcp


def save_seam_teaching_reference_yaml(path, planning_group, references):
    """Persist pre-correction TCP references used for seam-yaw correction."""
    document = {
        "format_version": 1,
        "planning_group": planning_group,
        "poses": {},
    }
    for name, stored in references.items():
        pose = stored[3]
        document["poses"][name] = {
            "position_m": {
                axis: float(getattr(pose.position, axis))
                for axis in ("x", "y", "z")
            },
            "orientation_xyzw": {
                axis: float(getattr(pose.orientation, axis))
                for axis in ("x", "y", "z", "w")
            },
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        yaml.safe_dump(document, stream, sort_keys=False)
    temporary_path.replace(path)


def load_seam_teaching_reference_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    poses = {}
    for name, data in document.get("poses", {}).items():
        pose = Pose()
        for axis in ("x", "y", "z"):
            setattr(pose.position, axis, float(data["position_m"][axis]))
        for axis in ("x", "y", "z", "w"):
            setattr(
                pose.orientation,
                axis,
                float(data["orientation_xyzw"][axis]),
            )
        if not pose_is_valid(pose):
            raise ValueError(f"invalid seam teaching reference: {name}")
        poses[name] = pose
    return document.get("planning_group"), poses


class WeldGuiNode(Node):
    """ROS interface used by the editable weld-path GUI."""

    def __init__(self, ui):
        super().__init__("weld_action_gui")
        self.ui = ui
        self.declare_parameter("expected_execute_motion", True)
        self.declare_parameter("robot_feedback_timeout", 5.0)
        self.declare_parameter("left_robot_ip", "192.168.1.11")
        self.declare_parameter("right_robot_ip", "192.168.1.12")
        self.declare_parameter("use_fake_head_hardware", False)
        self.declare_parameter("hicomm_source_ip", "192.168.1.2")
        self.declare_parameter("hicomm_welder_ip", "192.168.1.10")
        self.declare_parameter("hicomm_port", 60000)
        self.cartesian_motion_client = ActionClient(
            self, CartesianPath, "cartesian_path"
        )
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )
        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )
        self.cartesian_planning_client = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
        )
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.joint_trajectory_clients = {
            "left": ActionClient(
                self,
                FollowJointTrajectory,
                "/left_manipulator_controller/follow_joint_trajectory",
            ),
            "right": ActionClient(
                self,
                FollowJointTrajectory,
                "/right_manipulator_controller/follow_joint_trajectory",
            ),
            "head": ActionClient(
                self,
                FollowJointTrajectory,
                "/robot_head_controller/follow_joint_trajectory",
            ),
        }
        self.joint_trajectory_cancel_clients = {
            device: self.create_client(
                CancelGoal,
                f"/{CONTROLLER_NAMES[device]}/follow_joint_trajectory/"
                "_action/cancel_goal",
            )
            for device in ("left", "right", "head")
        }
        self.digital_output_client = self.create_client(
            SetDigitalOutput,
            "/right_rbpodo_hardware/set_digital_output",
        )
        self.move_stop_clients = {
            arm: self.create_client(
                MoveStop,
                f"/{arm}_rbpodo_hardware/move_stop",
            )
            for arm in ("left", "right")
        }
        self.controller_list_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
        )
        self.controller_switch_client = self.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.active_motion_goal = None
        self.active_touch_probe = None
        self.touch_probe_edge_pose = None
        self.touch_probe_stop_requested = threading.Event()
        self.touch_probe_controller_deactivated = False
        self.touch_stop_lock = threading.Lock()
        self.active_touch_guard = None
        self.touch_guard_stop_lock = threading.Lock()
        self.touch_guard_triggered = threading.Event()
        self.touch_guard_stop_complete = threading.Event()
        self.touch_guard_stop_success = False
        self.node_touch_input_states = {"left": None, "right": None}
        self.initial_planned_trajectory = None
        self.initial_planned_pose_name = None
        self.initial_planned_group = None
        self.latest_rviz_display = None
        self.latest_rviz_display_at = None
        self.request_execution = False
        self.execute_motion_enabled = self.get_parameter(
            "expected_execute_motion"
        ).value
        controlled_devices = ("left", "right", "head")
        self.expect_robot_feedback = {
            device: True for device in controlled_devices
        }
        self.robot_feedback_seen = {
            device: False for device in controlled_devices
        }
        self.robot_ready_reported = {
            device: False for device in controlled_devices
        }
        self.controller_states = {
            device: None for device in controlled_devices
        }
        self.controller_state_future = None
        self.latest_joint_positions = {}
        self.last_robot_feedback_at = {
            device: None for device in controlled_devices
        }
        startup_deadline = time.monotonic() + 90.0
        self.connection_deadline = {
            device: startup_deadline for device in controlled_devices
        }
        self.rviz_goal_refresh_pending = True
        marker_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            "weld_path_markers",
            marker_qos,
        )
        self.pose_publisher = self.create_publisher(
            PoseArray,
            "weld_6d_poses",
            marker_qos,
        )
        self.display_trajectory_publisher = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            marker_qos,
        )
        self.create_subscription(
            DisplayTrajectory,
            "/display_planned_path",
            self._display_trajectory_received,
            marker_qos,
        )
        self.rviz_goal_refresh_publisher = self.create_publisher(
            Empty,
            "/rviz/moveit/update_goal_state",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.create_timer(0.5, self._check_robot_feedback)
        self.create_subscription(
            SystemState,
            "/right_rbpodo_hardware/system_state",
            lambda message: self._system_state(message, "right"),
            10,
        )
        self.create_subscription(
            SystemState,
            "/left_rbpodo_hardware/system_state",
            lambda message: self._system_state(message, "left"),
            10,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state,
            10,
        )
        self.ui.post(
            self.ui.set_execution_configuration,
            self.get_parameter("expected_execute_motion").value,
            self.get_parameter("left_robot_ip").value,
            self.get_parameter("right_robot_ip").value,
            self.get_parameter("use_fake_head_hardware").value,
            self.get_parameter("hicomm_source_ip").value,
            self.get_parameter("hicomm_welder_ip").value,
            self.get_parameter("hicomm_port").value,
        )

    def _system_state(self, message, arm):
        touch_active = bool(message.digital_in[TOUCH_INPUT_PORT])
        previous_touch = self.node_touch_input_states[arm]
        self.node_touch_input_states[arm] = touch_active
        probe = self.active_touch_probe
        if (
            probe is not None
            and probe[0] == arm
            and previous_touch is False
            and touch_active
            and not self.touch_probe_stop_requested.is_set()
        ):
            # Latch before starting the worker.  Contact inputs can bounce
            # OFF/ON while the tool settles; only the first edge belongs to
            # this probe.
            self.touch_probe_stop_requested.set()
            try:
                # Latch the TCP at the DI8 edge.  Waiting for measured
                # standstill before reading TF records braking overshoot as if
                # it were the physical contact point, especially along Y.
                self.touch_probe_edge_pose = self._current_tcp_pose(probe[2])
                self.ui.post(
                    self.ui.apply_touch_edge_capture,
                    copy.deepcopy(self.touch_probe_edge_pose),
                    probe[2],
                    probe[1],
                    copy.deepcopy(probe[3]),
                )
            except TransformException as error:
                self.touch_probe_edge_pose = None
                self.ui.post(
                    self.ui.log,
                    f"DI8 edge TCP latch failed; stopped TCP will be used: {error}",
                )
            self.get_logger().warning(
                f"DI{TOUCH_INPUT_PORT} rising edge · {arm} · "
                f"stopping active probe {probe[1]}"
            )
            threading.Thread(
                target=self.stop_touch_probe_and_capture,
                daemon=True,
            ).start()
        guard = self.active_touch_guard
        if (
            guard is not None
            and guard[0] == arm
            and previous_touch is False
            and touch_active
        ):
            threading.Thread(
                target=self.stop_touch_guarded_motion,
                daemon=True,
            ).start()
        if previous_touch is None or previous_touch != touch_active:
            self.ui.post(
                self.ui.update_touch_input,
                arm,
                touch_active,
            )
        if arm == "right":
            self.ui.post_latest(
                "right_control_box_io",
                self.ui.update_control_box_io,
                tuple(message.digital_in),
                tuple(message.digital_out),
            )
        if not self.expect_robot_feedback[arm]:
            return
        self.last_robot_feedback_at[arm] = time.monotonic()
        self.robot_feedback_seen[arm] = True

    def _joint_state(self, message):
        """Use complete finite measured arm states as connection feedback."""
        positions = dict(zip(message.name, message.position))
        self.latest_joint_positions.update(
            {
                name: position
                for name, position in positions.items()
                if math.isfinite(position)
            }
        )
        received_at = time.monotonic()
        for arm, expected_names in CONTROLLED_JOINT_NAMES.items():
            if expected_names.issubset(positions) and all(
                math.isfinite(positions[name]) for name in expected_names
            ):
                self.last_robot_feedback_at[arm] = received_at
                self.robot_feedback_seen[arm] = True

    def _display_trajectory_received(self, message):
        """Keep the latest non-empty trajectory displayed by MoveIt/RViz."""
        if not message.trajectory:
            return
        if not any(
            trajectory.joint_trajectory.points
            for trajectory in message.trajectory
        ):
            return
        self.latest_rviz_display = copy.deepcopy(message)
        self.latest_rviz_display_at = time.monotonic()

    def latest_rviz_plan(self):
        if self.latest_rviz_display is None:
            return None, None
        age = time.monotonic() - self.latest_rviz_display_at
        return copy.deepcopy(self.latest_rviz_display), age

    def capture_touch_pose(self, planning_group, source):
        try:
            pose = self._current_tcp_pose(planning_group)
        except TransformException as error:
            self.ui.post(
                self.ui.error,
                f"Touch TCP capture failed: {error}",
            )
            return
        self.ui.post(
            self.ui.apply_touch_capture,
            pose,
            planning_group,
            source,
        )

    def set_digital_output(self, port, value):
        if not self.digital_output_client.wait_for_service(timeout_sec=2.0):
            self.ui.post(
                self.ui.digital_output_result,
                port,
                False,
                "/right_rbpodo_hardware/set_digital_output unavailable",
            )
            return
        request = SetDigitalOutput.Request()
        request.port = port
        request.value = value
        future = self.digital_output_client.call_async(request)
        future.add_done_callback(
            lambda result: self._digital_output_result(result, port)
        )

    def _digital_output_result(self, future, port):
        try:
            response = future.result()
            self.ui.post(
                self.ui.digital_output_result,
                port,
                response.success,
                response.message,
            )
        except Exception as error:
            self.ui.post(
                self.ui.digital_output_result,
                port,
                False,
                str(error),
            )

    def _set_digital_output_sync(self, port, value):
        if not self.digital_output_client.wait_for_service(timeout_sec=2.0):
            return False, "RBPodo set_digital_output service unavailable"
        request = SetDigitalOutput.Request()
        request.port = int(port)
        request.value = bool(value)
        event = threading.Event()
        outcome = {}

        def completed(future):
            try:
                response = future.result()
                outcome["value"] = (response.success, response.message)
            except Exception as error:
                outcome["value"] = (False, str(error))
            event.set()

        self.digital_output_client.call_async(request).add_done_callback(completed)
        if not event.wait(timeout=3.0):
            return False, "RBPodo digital output command timed out"
        return outcome["value"]

    @staticmethod
    def _call_service_and_wait(client, request, description):
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f"{description} service unavailable"
        event = threading.Event()
        outcome = {}

        def completed(future):
            try:
                response = future.result()
                outcome["value"] = (response.success, response.message)
            except Exception as error:
                outcome["value"] = (False, str(error))
            event.set()

        client.call_async(request).add_done_callback(completed)
        if not event.wait(timeout=10.0):
            return False, f"{description} timed out"
        return outcome["value"]

    def _send_action_goal_and_wait(
        self,
        client,
        goal,
        description,
        *,
        result_timeout=300.0,
        on_accepted=None,
    ):
        """Submit an action goal and block only the calling worker thread."""
        if not client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError(f"{description} action server unavailable")
        accepted = threading.Event()
        finished = threading.Event()
        outcome = {}

        def result_ready(future):
            try:
                outcome["result"] = future.result().result
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        def goal_ready(future):
            try:
                handle = future.result()
                if not handle.accepted:
                    outcome["error"] = f"{description} goal rejected"
                    finished.set()
                    return
                outcome["handle"] = handle
                self.active_motion_goal = handle
                if on_accepted is not None:
                    on_accepted(handle)
                handle.get_result_async().add_done_callback(result_ready)
            except Exception as error:
                outcome["error"] = str(error)
                finished.set()
            finally:
                accepted.set()

        client.send_goal_async(goal).add_done_callback(goal_ready)
        if not accepted.wait(timeout=5.0):
            raise TimeoutError(f"{description} goal response timed out")
        if not finished.wait(timeout=result_timeout):
            handle = outcome.get("handle")
            if handle is not None:
                try:
                    handle.cancel_goal_async()
                except Exception:
                    pass
            raise TimeoutError(f"{description} timed out")
        handle = outcome.get("handle")
        if handle is self.active_motion_goal:
            self.active_motion_goal = None
        if "error" in outcome:
            raise RuntimeError(outcome["error"])
        return outcome["result"]

    def _check_robot_feedback(self):
        self._request_controller_states()
        feedback_timeout = max(
            1.0,
            float(self.get_parameter("robot_feedback_timeout").value),
        )
        move_group_ready = self.move_group_client.server_is_ready()
        for arm in ("left", "right", "head"):
            last_feedback = self.last_robot_feedback_at[arm]
            deadline = self.connection_deadline[arm]
            feedback_is_fresh = (
                last_feedback is not None
                and time.monotonic() - last_feedback <= feedback_timeout
            )
            controller_ready = (
                not self.execute_motion_enabled
                or (
                    self.controller_states[arm] == "active"
                    and self.joint_trajectory_clients[arm].server_is_ready()
                )
            )
            stack_ready = (
                self.robot_feedback_seen[arm]
                and feedback_is_fresh
                and (arm == "head" or move_group_ready)
                and controller_ready
            )
            if stack_ready and not self.robot_ready_reported[arm]:
                self.robot_ready_reported[arm] = True
                self.connection_deadline[arm] = None
                self.ui.post(self.ui.robot_feedback_connected, arm)
                continue
            if self.robot_ready_reported[arm] and not stack_ready:
                self.robot_ready_reported[arm] = False
                self.rviz_goal_refresh_pending = True
                detail = self._not_ready_detail(
                    arm,
                    feedback_is_fresh,
                    move_group_ready,
                    controller_ready,
                )
                self.ui.post(self.ui.robot_feedback_lost, arm, detail)
                self.get_logger().warning(
                    f"{arm.upper()} CONNECTION X · {detail}"
                )
                continue
            if (
                self.expect_robot_feedback[arm]
                and not self.robot_ready_reported[arm]
                and deadline is not None
                and time.monotonic() > deadline
            ):
                self.connection_deadline[arm] = None
                detail = (
                    "measured joint feedback received, but required controllers "
                    "did not become ready"
                    if self.robot_feedback_seen[arm]
                    else "no fresh complete measured joint state received"
                )
                self.ui.post(self.ui.robot_feedback_lost, arm)
                self.get_logger().error(
                    f"{arm.upper()} CONNECTION X · {detail}"
                )
                continue
            if (
                not self.expect_robot_feedback[arm]
                or not self.robot_feedback_seen[arm]
                or last_feedback is None
                or feedback_is_fresh
            ):
                continue
            self.robot_feedback_seen[arm] = False
            if self.robot_ready_reported[arm]:
                self.robot_ready_reported[arm] = False
                self.rviz_goal_refresh_pending = True
                self.ui.post(self.ui.robot_feedback_lost, arm)

        expected_real_arms = tuple(
            arm
            for arm in ("left", "right")
            if self.expect_robot_feedback[arm]
        )
        all_expected_arms_ready = (
            bool(expected_real_arms)
            and move_group_ready
            and all(
                self.robot_ready_reported[arm]
                for arm in expected_real_arms
            )
        )
        if (
            self.rviz_goal_refresh_pending
            and all_expected_arms_ready
            and self.rviz_goal_refresh_publisher.get_subscription_count() > 0
        ):
            # This invokes RViz's own "Goal State = <current>" callback. It
            # changes only the orange query state and sends no robot command.
            self.rviz_goal_refresh_publisher.publish(Empty())
            self.rviz_goal_refresh_pending = False
            self.get_logger().info(
                "Requested RViz Goal State refresh from current state"
            )

    def _request_controller_states(self):
        if not self.controller_list_client.service_is_ready():
            return
        if (
            self.controller_state_future is not None
            and not self.controller_state_future.done()
        ):
            return
        self.controller_state_future = self.controller_list_client.call_async(
            ListControllers.Request()
        )
        self.controller_state_future.add_done_callback(
            self._controller_states_received
        )

    def _controller_states_received(self, future):
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(
                f"Failed to read controller states: {error}"
            )
            return
        states = {
            controller.name: controller.state
            for controller in response.controller
        }
        for arm, name in CONTROLLER_NAMES.items():
            self.controller_states[arm] = states.get(name)

    def _not_ready_detail(
        self,
        arm,
        feedback_is_fresh,
        move_group_ready,
        controller_ready,
    ):
        if not feedback_is_fresh:
            return "measured joint feedback timeout"
        if arm != "head" and not move_group_ready:
            return "MoveGroup action unavailable"
        if self.controller_states[arm] != "active":
            return (
                f"{CONTROLLER_NAMES[arm]} state="
                f"{self.controller_states[arm] or 'unknown'}"
            )
        if not controller_ready:
            return "FollowJointTrajectory action unavailable"
        return "stack not ready"

    def _current_tcp_pose(self, planning_group):
        transform = self.tf_buffer.lookup_transform(
            "World",
            tip_link_for_group(planning_group),
            rclpy.time.Time(),
            timeout=Duration(seconds=1.0),
        )
        source = transform.transform
        pose = Pose()
        pose.position.x = source.translation.x
        pose.position.y = source.translation.y
        pose.position.z = source.translation.z
        pose.orientation = source.rotation
        return pose

    def resolve_tcp_joint_state(
        self,
        planning_group,
        target_pose,
        expected_joint_names,
        endpoint,
        teaching_name,
    ):
        """Use MoveIt IK to make a named pose's joints match its corrected TCP."""
        while rclpy.ok() and not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning(
                f"Waiting for /compute_ik to resolve corrected {teaching_name} TCP"
            )
        if not rclpy.ok():
            return
        request = GetPositionIK.Request()
        request.ik_request.group_name = planning_group
        request.ik_request.robot_state.is_diff = True
        request.ik_request.ik_link_name = tip_link_for_group(planning_group)
        request.ik_request.pose_stamped.header.frame_id = "World"
        request.ik_request.pose_stamped.header.stamp = (
            self.get_clock().now().to_msg()
        )
        request.ik_request.pose_stamped.pose = copy.deepcopy(target_pose)
        # This call resolves and persists a kinematic seed only.  A sensed weld
        # point is intentionally on the workpiece and can be rejected as a
        # collision if scene geometry is present.  Actual Plan/Execute still
        # performs normal collision checking before any physical movement.
        request.ik_request.avoid_collisions = False
        request.ik_request.timeout = Duration(seconds=3.0).to_msg()
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                outcome["response"] = future.result()
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        self.ik_client.call_async(request).add_done_callback(
            response_ready
        )
        while rclpy.ok() and not finished.wait(timeout=0.2):
            pass
        if not rclpy.ok():
            return
        if "error" in outcome:
            self.ui.post(
                self.ui.corrected_tcp_joint_state_failed,
                endpoint,
                teaching_name,
                outcome["error"],
            )
            return
        response = outcome["response"]
        if response.error_code.val != 1:
            self.ui.post(
                self.ui.corrected_tcp_joint_state_failed,
                endpoint,
                teaching_name,
                f"MoveIt IK error code {response.error_code.val}",
            )
            return
        resolved = dict(zip(
            response.solution.joint_state.name,
            response.solution.joint_state.position,
        ))
        missing = [name for name in expected_joint_names if name not in resolved]
        if missing:
            self.ui.post(
                self.ui.corrected_tcp_joint_state_failed,
                endpoint,
                teaching_name,
                "corrected TCP IK omitted joints: " + ", ".join(missing),
            )
            return
        self.ui.post(
            self.ui.apply_corrected_tcp_joint_state,
            endpoint,
            teaching_name,
            planning_group,
            tuple(expected_joint_names),
            tuple(resolved[name] for name in expected_joint_names),
            copy.deepcopy(target_pose),
        )

    def resolve_tcp_joint_states(self, targets):
        """Resolve related corrected named poses serially for deterministic YAML."""
        for (
            endpoint,
            planning_group,
            target_pose,
            joint_names,
            teaching_name,
        ) in targets:
            self.resolve_tcp_joint_state(
                planning_group,
                target_pose,
                joint_names,
                endpoint,
                teaching_name,
            )

    def publish_points(self, points, visible=True):
        displayed_points = points if visible else []
        markers, pose_array = make_weld_visualization(
            displayed_points,
            "World",
            self.get_clock().now().to_msg(),
        )
        self.marker_publisher.publish(markers)
        self.pose_publisher.publish(pose_array)

    def publish_seam_comparison(self, raw_points, corrected_points, visible=True):
        """Show raw seam opaque and offset-corrected seam translucent."""
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)
        if visible:
            for marker_id, points, color in (
                (100, raw_points, (1.0, 0.05, 0.02, 1.0)),
                (101, corrected_points, (0.0, 0.7, 1.0, 0.38)),
            ):
                line = Marker()
                line.header.frame_id = "World"
                line.header.stamp = stamp
                # moveit.rviz already enables this namespace.
                line.ns = "weld_seam"
                line.id = marker_id
                line.type = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = 0.006 if marker_id == 100 else 0.010
                line.color.r, line.color.g, line.color.b, line.color.a = color
                line.points = [
                    Point(
                        x=pose.position.x,
                        y=pose.position.y,
                        z=pose.position.z,
                    )
                    for pose in points
                ]
                markers.markers.append(line)
        self.marker_publisher.publish(markers)
        pose_array = PoseArray()
        pose_array.header.frame_id = "World"
        pose_array.header.stamp = stamp
        pose_array.poses = list(raw_points)
        self.pose_publisher.publish(pose_array)

    def publish_touch_geometry(self, endpoint, wall, floor, seam_point):
        """Show two DI8 contacts, their midpoint, and reconstructed seam point."""
        endpoint = str(endpoint).strip().lower()
        if endpoint not in ("start", "goal"):
            raise ValueError(f"unknown touch endpoint: {endpoint}")
        if not all(pose_is_valid(pose) for pose in (wall, floor, seam_point)):
            raise ValueError("touch visualization poses must be valid")
        midpoint = midpoint_pose(wall, floor)
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        base_id = 300 if endpoint == "start" else 400
        namespace = "seam_touch_geometry"

        # Delete only this endpoint's old diagnostic markers so the weld path
        # and the other endpoint remain visible.
        for marker_id in range(base_id, base_id + 16):
            marker = Marker()
            marker.header.frame_id = "World"
            marker.header.stamp = stamp
            marker.ns = namespace
            marker.id = marker_id
            marker.action = Marker.DELETE
            markers.markers.append(marker)

        items = (
            ("WALL TOUCH", wall, (1.0, 0.08, 0.05, 1.0)),
            ("FLOOR TOUCH", floor, (0.05, 0.3, 1.0, 1.0)),
            ("SEAM POINT", seam_point, (0.0, 1.0, 0.2, 1.0)),
            ("1:1 MIDPOINT", midpoint, (1.0, 0.8, 0.0, 1.0)),
        )
        for index, (label_text, pose, color) in enumerate(items):
            sphere = Marker()
            sphere.header.frame_id = "World"
            sphere.header.stamp = stamp
            sphere.ns = namespace
            sphere.id = base_id + index * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = copy.deepcopy(pose)
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.014
            (
                sphere.color.r,
                sphere.color.g,
                sphere.color.b,
                sphere.color.a,
            ) = color
            markers.markers.append(sphere)

            label = Marker()
            label.header.frame_id = "World"
            label.header.stamp = stamp
            label.ns = namespace
            label.id = base_id + index * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = pose.position.x
            label.pose.position.y = pose.position.y
            label.pose.position.z = pose.position.z + 0.022
            label.pose.orientation.w = 1.0
            label.scale.z = 0.018
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text = f"{endpoint.upper()} {label_text}"
            markers.markers.append(label)

        connection = Marker()
        connection.header.frame_id = "World"
        connection.header.stamp = stamp
        connection.ns = namespace
        connection.id = base_id + 12
        connection.type = Marker.LINE_STRIP
        connection.action = Marker.ADD
        connection.scale.x = 0.003
        connection.color.r = connection.color.g = connection.color.b = 0.9
        connection.color.a = 0.8
        connection.points = [
            Point(x=wall.position.x, y=wall.position.y, z=wall.position.z),
            Point(
                x=midpoint.position.x,
                y=midpoint.position.y,
                z=midpoint.position.z,
            ),
            Point(x=floor.position.x, y=floor.position.y, z=floor.position.z),
        ]
        markers.markers.append(connection)
        self.marker_publisher.publish(markers)

    def acquire_points(
        self,
        reference,
        axis,
        distance,
        count,
        explicit_position,
        rpy_offset,
        rpy_reference,
        visible,
        planning_group,
    ):
        try:
            tcp = self._current_tcp_pose(planning_group)
            if explicit_position is not None:
                (
                    tcp.position.x,
                    tcp.position.y,
                    tcp.position.z,
                ) = explicit_position
            tcp = pose_with_rpy_offset(
                tcp, *rpy_offset, reference=rpy_reference
            )
            points = straight_waypoints(
                tcp,
                distance,
                count,
                axis,
                reference,
            )
        except (TransformException, ValueError) as error:
            self.ui.post(
                self.ui.error,
                f"Straight path acquisition failed: {error}",
            )
            return
        self.publish_points(points, visible)
        self.ui.post(self.ui.set_new_points, points, "straight")
        start_description = (
            "current TCP"
            if explicit_position is None
            else (
                "World XYZ "
                f"({explicit_position[0]:.3f}, "
                f"{explicit_position[1]:.3f}, "
                f"{explicit_position[2]:.3f})"
            )
        )
        self.ui.post(
            self.ui.log,
            f"Acquired straight seam · start={start_description} · "
            f"{reference} {axis.upper()} · "
            f"distance={distance * 1000.0:.1f} mm · {count} poses",
        )

    def generate_circle(
        self,
        normal_axis,
        radius,
        count,
        closed,
        face_center,
        visible,
        planning_group,
    ):
        try:
            tcp = self._current_tcp_pose(planning_group)
            points = circle_waypoints(
                tcp,
                radius,
                count,
                closed,
                face_center,
                normal_axis,
            )
        except (TransformException, ValueError) as error:
            self.ui.post(self.ui.error, f"Circle generation failed: {error}")
            return
        self.publish_points(points, visible)
        self.ui.post(self.ui.set_new_points, points, "circle")
        description = (
            f"{count} unique points"
            f"{' + closing point' if closed else ''}, radius={radius:.3f} m"
        )
        orientation = (
            "TCP +Z faces center"
            if face_center
            else "fixed TCP orientation"
        )
        self.ui.post(
            self.ui.log,
            f"Generated World-{normal_axis.upper()} normal circle · "
            f"{description} · {orientation}",
        )

    def capture_initial_state(self, planning_group, pose_name="robot_start"):
        arm = "left" if planning_group.startswith("left") else "right"
        joint_names = [
            f"{arm}_manipulator_joint{index}" for index in range(1, 7)
        ]
        try:
            positions = [self.latest_joint_positions[name] for name in joint_names]
            tcp = self._current_tcp_pose(planning_group)
        except KeyError:
            self.ui.post(self.ui.error, "Complete measured joint state is unavailable")
            return
        except TransformException as error:
            self.ui.post(self.ui.error, f"Initial TCP capture failed: {error}")
            return
        self.ui.post(
            self.ui.apply_initial_state,
            pose_name,
            planning_group,
            joint_names,
            positions,
            tcp,
        )

    def execute_touch_probe(
        self,
        planning_group,
        probe_kind,
        direction,
        distance,
        velocity_scale,
        interpolation_step,
    ):
        """Execute a straight World-vector probe path; the GUI cancels on DI8."""
        arm = planning_group.removesuffix("_manipulator")
        try:
            start = self._current_tcp_pose(planning_group)
            self.active_touch_probe = (
                arm,
                probe_kind,
                planning_group,
                copy.deepcopy(start),
                velocity_scale,
                interpolation_step,
            )
            self.touch_probe_edge_pose = None
            self.touch_probe_controller_deactivated = False
            self.touch_probe_stop_requested.clear()
            # MoveIt's GetCartesianPath already interpolates this segment using
            # max_step.  Supplying every 1 mm point here duplicated that work
            # and made a 50 mm probe spend many seconds in PLAN PREVIEW.
            count = 2
            if isinstance(direction, str):
                points = straight_waypoints(
                    start, distance, count, direction.lower(), "world"
                )
            else:
                dx, dy, dz = _unit_vector(direction, "touch probe direction")
                goal = copy.deepcopy(start)
                goal.position.x += float(distance) * dx
                goal.position.y += float(distance) * dy
                goal.position.z += float(distance) * dz
                points = [copy.deepcopy(start), goal]
        except (TransformException, ValueError) as error:
            self.active_touch_probe = None
            self.ui.post(self.ui.touch_probe_failed, str(error))
            return
        self.publish_points(points, True)
        self.submit_cartesian_motion(
            points,
            velocity_scale,
            interpolation_step,
            True,
            True,
            False,
            planning_group,
        )

    def stop_touch_probe_and_capture(self):
        """Stop command streaming, confirm standstill, then capture TCP."""
        if not self.touch_stop_lock.acquire(blocking=False):
            return
        try:
            self._stop_touch_probe_and_capture_locked()
        finally:
            self.touch_stop_lock.release()

    def _stop_touch_probe_and_capture_locked(self):
        """Serialized implementation for a DI8 rising edge."""
        probe = self.active_touch_probe
        if probe is None:
            return
        arm, kind, planning_group, start, speed, interpolation = probe
        stationary, controller_deactivated = self._stop_motion_on_di8(
            arm, f"probe {kind}"
        )
        self.touch_probe_controller_deactivated = controller_deactivated
        if not stationary:
            self.active_touch_probe = None
            if self.touch_probe_controller_deactivated:
                self.switch_arm_controller(arm, True)
                self.touch_probe_controller_deactivated = False
            self.ui.post(
                self.ui.touch_probe_failed,
                "DI8 received, but measured joints did not reach standstill",
            )
            return
        try:
            stopped_pose = self._current_tcp_pose(planning_group)
        except TransformException as error:
            self.active_touch_probe = None
            if self.touch_probe_controller_deactivated:
                self.switch_arm_controller(arm, True)
                self.touch_probe_controller_deactivated = False
            self.ui.post(self.ui.touch_probe_failed, str(error))
            return
        touched = (
            copy.deepcopy(self.touch_probe_edge_pose)
            if self.touch_probe_edge_pose is not None
            else copy.deepcopy(stopped_pose)
        )
        edge_values = (
            touched.position.x, touched.position.y, touched.position.z
        )
        stopped_values = (
            stopped_pose.position.x,
            stopped_pose.position.y,
            stopped_pose.position.z,
        )
        braking_mm = tuple(
            (stopped_values[index] - edge_values[index]) * 1000.0
            for index in range(3)
        )
        self.ui.post(
            self.ui.log,
            f"DI8 CONTACT LATCH · {kind} · edge XYZ="
            f"({edge_values[0]:.6f}, {edge_values[1]:.6f}, "
            f"{edge_values[2]:.6f}) m · braking delta="
            f"({braking_mm[0]:+.3f}, {braking_mm[1]:+.3f}, "
            f"{braking_mm[2]:+.3f}) mm",
        )
        self.touch_probe_edge_pose = None
        self.ui.post(
            self.ui.apply_touch_capture,
            touched,
            planning_group,
            f"automatic probe:{kind}",
            start,
            stopped_pose,
        )

    def _stop_motion_on_di8(self, arm, label):
        """Apply the seam-probe controlled-stop ladder to any guarded move."""
        handle = self.active_motion_goal
        action_finished = threading.Event()
        controller_deactivated = False
        if handle is not None:
            try:
                handle.get_result_async().add_done_callback(
                    lambda _future: action_finished.set()
                )
                handle.cancel_goal_async()
            except Exception as error:
                self.ui.post(
                    self.ui.log,
                    f"DI8 {label} action cancel warning: {error}",
                )

        # Prefer action cancellation while keeping the trajectory controller
        # active.  Deactivate/reactivate mode switches caused a visible kick at
        # contact and before retract.  Escalate only when cancellation cannot
        # establish standstill promptly.
        cancel_success, cancel_message = self.cancel_controller_goals(arm)
        self.ui.post(
            self.ui.log,
            f"DI8 direct trajectory cancel: "
            f"{'OK' if cancel_success else 'FAILED'} · {cancel_message}",
        )
        action_finished.wait(timeout=0.25)
        stationary = (
            cancel_success
            and self.wait_until_arm_stopped(arm, timeout=1.0)
        )
        if cancel_success and stationary:
            self.ui.post(
                self.ui.log,
                "DI8 smooth stop: action canceled · controller kept active",
            )
        else:
            controller_success, controller_message = self.switch_arm_controller(
                arm, False
            )
            controller_deactivated = bool(controller_success)
            direct_stop_success, direct_stop_message = (
                self.request_direct_motion_stop(arm)
            )
            self.ui.post(
                self.ui.log,
                f"DI8 fallback controller stop: "
                f"{'OK' if controller_success else 'FAILED'} · "
                f"{controller_message}",
            )
            self.ui.post(
                self.ui.log,
                f"DI8 fallback RBPodo move_stop: "
                f"{'OK' if direct_stop_success else 'FAILED'} · "
                f"{direct_stop_message}",
            )
            stationary = self.wait_until_arm_stopped(arm)
        if handle is not None and not action_finished.wait(timeout=2.0):
            self.ui.post(
                self.ui.log,
                "DI8 motion is physically stopped; outer action cleanup "
                "is still pending",
            )
        return bool(stationary), controller_deactivated

    def cancel_controller_goals(self, arm):
        """Cancel every active FollowJointTrajectory goal for one arm."""
        client = self.joint_trajectory_cancel_clients.get(arm)
        if client is None or not client.wait_for_service(timeout_sec=0.25):
            return False, f"{arm} trajectory cancel service unavailable"
        request = CancelGoal.Request()
        # Zero UUID + zero timestamp means cancel all goals.
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                response = future.result()
                outcome["code"] = int(response.return_code)
                outcome["count"] = len(response.goals_canceling)
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        client.call_async(request).add_done_callback(response_ready)
        if not finished.wait(timeout=1.0):
            return False, "cancel response timed out"
        if "error" in outcome:
            return False, outcome["error"]
        success = outcome.get("code") == CancelGoal.Response.ERROR_NONE
        return success, (
            f"return_code={outcome.get('code')} · "
            f"goals_canceling={outcome.get('count', 0)}"
        )

    def stop_sequence_equipment(self, devices):
        """Cancel every robot goal and escalate until measured motion stops."""
        results = []
        for device in tuple(dict.fromkeys(devices)):
            canceled, cancel_message = self.cancel_controller_goals(device)
            stationary = self.wait_until_device_stopped(device, timeout=1.5)
            if stationary:
                results.append(
                    f"{device}: stationary · cancel="
                    f"{'OK' if canceled else cancel_message}"
                )
            else:
                deactivated, deactivate_message = self.switch_arm_controller(
                    device, False
                )
                if device in ("left", "right"):
                    direct, direct_message = self.request_direct_motion_stop(device)
                else:
                    direct, direct_message = False, "head has no RBPodo move_stop"
                stopped_after_fallback = self.wait_until_device_stopped(device)
                results.append(
                    f"{device}: fallback stop="
                    f"{'OK' if stopped_after_fallback else 'FAILED'} · "
                    f"controller_off={deactivated} ({deactivate_message}) · "
                    f"move_stop={direct} ({direct_message})"
                )
        self.ui.post(self.ui.sequence_hard_stop_finished, results)

    def stop_touch_guarded_motion(self):
        """Stop a guarded named move using the seam-probe stop ladder."""
        if not self.touch_guard_stop_lock.acquire(blocking=False):
            return
        try:
            guard = self.active_touch_guard
            if guard is None:
                return
            arm, pose_name = guard
            self.touch_guard_triggered.set()
            self.touch_guard_stop_success = False
            stationary, controller_deactivated = self._stop_motion_on_di8(
                arm, f"guarded named pose {pose_name}"
            )
            restored = True
            restore_message = "controller remained active"
            if controller_deactivated:
                restored, restore_message = self.switch_arm_controller(arm, True)
                self.ui.post(
                    self.ui.log,
                    f"DI8 guarded motion controller restore: "
                    f"{'OK' if restored else 'FAILED'} · {restore_message}",
                )
            self.touch_guard_stop_success = bool(stationary and restored)
            if not stationary:
                self.ui.post(
                    self.ui.error,
                    f"DI8 detected during {pose_name}, but standstill was not confirmed",
                )
            elif not restored:
                self.ui.post(
                    self.ui.error,
                    f"DI8 stopped {pose_name}, but controller restore failed: "
                    f"{restore_message}",
                )
        finally:
            # The DI8 edge terminates only this guarded execution.  Drop the
            # guard as soon as stop/recovery finishes so a later command can
            # move again (including a deliberate retraction while DI8 is
            # still high).  A new stop requires DI8 to release and rise again.
            if self.active_touch_guard == guard:
                self.active_touch_guard = None
            self.touch_guard_stop_complete.set()
            self.touch_guard_stop_lock.release()

    def request_direct_motion_stop(self, arm):
        """Request RBPodo move_stop; this is a controlled stop, not E-stop."""
        client = self.move_stop_clients.get(arm)
        if client is None or not client.wait_for_service(timeout_sec=0.25):
            return False, f"/{arm}_rbpodo_hardware/move_stop unavailable"
        request = MoveStop.Request()
        request.timeout = 2.0
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                response = future.result()
                outcome["success"] = bool(response.success)
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        client.call_async(request).add_done_callback(response_ready)
        if not finished.wait(timeout=3.0):
            return False, "service response timed out"
        if "error" in outcome:
            return False, outcome["error"]
        return outcome.get("success", False), "controlled move_stop completed"

    def switch_arm_controller(self, arm, activate):
        """Deactivate to stop command streaming, or reactivate for return."""
        client = self.controller_switch_client
        controller = CONTROLLER_NAMES[arm]
        if not client.wait_for_service(timeout_sec=0.5):
            return False, "/controller_manager/switch_controller unavailable"
        request = SwitchController.Request()
        if activate:
            request.activate_controllers = [controller]
        else:
            request.deactivate_controllers = [controller]
        request.strictness = SwitchController.Request.BEST_EFFORT
        request.activate_asap = True
        request.timeout.sec = 3
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                outcome["success"] = bool(future.result().ok)
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        client.call_async(request).add_done_callback(response_ready)
        if not finished.wait(timeout=4.0):
            return False, f"{controller} switch timed out"
        if "error" in outcome:
            return False, outcome["error"]
        action = "activated" if activate else "deactivated"
        if not outcome.get("success", False):
            return False, f"{controller} failed to become {action}"
        expected_state = "active" if activate else "inactive"
        if not self.wait_for_controller_state(controller, expected_state):
            return False, (
                f"{controller} did not report {expected_state} after switch"
            )
        return True, f"{controller} {action}"

    def wait_for_controller_state(self, controller, expected, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            finished = threading.Event()
            outcome = {}

            def response_ready(future):
                try:
                    outcome["response"] = future.result()
                except Exception as error:
                    outcome["error"] = str(error)
                finished.set()

            self.controller_list_client.call_async(
                ListControllers.Request()
            ).add_done_callback(response_ready)
            if finished.wait(timeout=0.5) and "response" in outcome:
                states = {
                    item.name: item.state
                    for item in outcome["response"].controller
                }
                if states.get(controller) == expected:
                    return True
            time.sleep(0.05)
        return False

    def wait_until_arm_stopped(self, arm, timeout=3.0):
        """Confirm measured joints remain still before capturing the touch."""
        return self.wait_until_device_stopped(arm, timeout)

    def wait_until_device_stopped(self, device, timeout=3.0):
        """Confirm a controlled arm or head remains measurably stationary."""
        names = tuple(sorted(CONTROLLED_JOINT_NAMES[device]))
        deadline = time.monotonic() + timeout
        previous = None
        stable_since = None
        while time.monotonic() < deadline:
            try:
                current = tuple(self.latest_joint_positions[name] for name in names)
            except KeyError:
                time.sleep(0.02)
                continue
            now = time.monotonic()
            if previous is not None:
                maximum_delta = max(
                    abs(value - old)
                    for value, old in zip(current, previous)
                )
                if maximum_delta <= 2e-5:
                    stable_since = stable_since or now
                    if now - stable_since >= 0.30:
                        return True
                else:
                    stable_since = None
            previous = current
            time.sleep(0.02)
        return False

    def return_touch_probe(
        self,
        planning_group,
        touched,
        start,
        speed,
        step,
        probe_kind,
        settle_seconds,
    ):
        """Execute the reverse probe path back to its captured start pose."""
        try:
            arm = planning_group.removesuffix("_manipulator")
            self.ui.post(
                self.ui.log,
                f"DI8 {probe_kind} standstill dwell · "
                f"{settle_seconds:.1f} seconds",
            )
            time.sleep(settle_seconds)
            if self.touch_probe_controller_deactivated:
                activated, activation_message = self.switch_arm_controller(
                    arm, True
                )
                if not activated:
                    raise RuntimeError(activation_message)
                self.touch_probe_controller_deactivated = False
            points = linear_pose_waypoints(touched, start, 2)
            success, message = self.run_sequence_cartesian_motion(
                {
                    "planning_group": planning_group,
                    "interpolation_step": step,
                    "velocity_scale": speed,
                    "points": points,
                },
                True,
            )
        except (RuntimeError, ValueError, TransformException) as error:
            success, message = False, str(error)
        self.active_touch_probe = None
        self.ui.post(
            self.ui.touch_probe_return_finished,
            success,
            message,
            probe_kind,
        )

    def clear_touch_probe(self):
        self.active_touch_probe = None
        self.touch_probe_edge_pose = None
        self.touch_probe_stop_requested.set()

    def stop_auto_motion(self, arm):
        """Stop any auto-seam motion and restore an idle active controller."""
        handle = self.active_motion_goal
        if handle is not None:
            try:
                handle.cancel_goal_async()
            except Exception as error:
                self.ui.post(self.ui.log, f"STOP AUTO cancel warning: {error}")
        stopped, message = self.switch_arm_controller(arm, False)
        if not stopped:
            fallback, fallback_message = self.request_direct_motion_stop(arm)
            message = f"{message}; fallback={fallback}: {fallback_message}"
        stationary = self.wait_until_arm_stopped(arm)
        activated, activation_message = self.switch_arm_controller(arm, True)
        self.active_touch_probe = None
        success = stationary and activated
        self.ui.post(
            self.ui.auto_seam_stop_finished,
            success,
            f"{message}; {activation_message}",
        )

    def plan_initial_state(
        self,
        planning_group,
        joint_names,
        positions,
        velocity_scale,
        pose_name=None,
        target_tcp=None,
    ):
        self.initial_planned_trajectory = None
        self.initial_planned_pose_name = None
        self.initial_planned_group = None
        try:
            current_positions = [
                self.latest_joint_positions[name] for name in joint_names
            ]
        except KeyError:
            self.ui.post(
                self.ui.error,
                "Complete measured joint state is unavailable",
            )
            return
        tcp_target = bool(
            pose_name in TCP_POSE_TEACHING_POSES
            and pose_is_valid(target_tcp)
        )
        if tcp_target:
            try:
                current_tcp = self._current_tcp_pose(planning_group)
            except TransformException as error:
                self.ui.post(self.ui.error, f"Current TCP lookup failed: {error}")
                return
            target_delta = math.sqrt(sum(
                (
                    getattr(current_tcp.position, axis)
                    - getattr(target_tcp.position, axis)
                ) ** 2
                for axis in ("x", "y", "z")
            ))
            orientation_delta = quaternion_angular_distance(
                current_tcp.orientation, target_tcp.orientation
            )
            already_at_target = (
                target_delta <= 0.001 and orientation_delta <= 0.01
            )
        else:
            maximum_delta = max(
                abs(current - target)
                for current, target in zip(current_positions, positions)
            )
            already_at_target = maximum_delta <= 0.002
        if already_at_target:
            self.ui.post(
                self.ui.pipeline_result,
                "Already at selected corrected XYZ + taught orientation"
                if tcp_target
                else "Already at selected taught pose · no plan required",
            )
            return
        if tcp_target:
            self._plan_named_tcp_linear(
                planning_group,
                current_tcp,
                target_tcp,
                tuple(positions),
                velocity_scale,
                pose_name,
            )
            return
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(self.ui.error, "MoveGroup action server unavailable")
            return
        goal = MoveGroup.Goal()
        goal.request.group_name = planning_group
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.start_state.is_diff = True
        goal.request.max_velocity_scaling_factor = velocity_scale
        goal.request.max_acceleration_scaling_factor = velocity_scale
        constraints = Constraints()
        for name, position in zip(joint_names, positions):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = position
            constraint.tolerance_above = 0.001
            constraint.tolerance_below = 0.001
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        self.ui.post(
            self.ui.pipeline_waiting,
            "Planning to selected taught joint angles",
        )
        future = self.move_group_client.send_goal_async(goal)
        target_positions = tuple(positions)
        future.add_done_callback(
            lambda result: self._initial_plan_goal_response(
                result,
                planning_group,
                target_positions,
                velocity_scale,
                pose_name,
            )
        )

    def _plan_named_tcp_linear(
        self,
        planning_group,
        current_tcp,
        target_tcp,
        target_positions,
        velocity_scale,
        pose_name,
    ):
        """Plan current TCP→named TCP as linear XYZ plus quaternion SLERP."""
        if not self.cartesian_planning_client.wait_for_service(timeout_sec=3.0):
            self.ui.post(
                self.ui.error, "/compute_cartesian_path service unavailable"
            )
            return
        try:
            waypoints = named_tcp_linear_waypoints(current_tcp, target_tcp)
        except ValueError as error:
            self.ui.post(self.ui.error, f"Named TCP path failed: {error}")
            return
        request = GetCartesianPath.Request()
        request.header.frame_id = "World"
        request.start_state.is_diff = True
        request.group_name = planning_group
        request.link_name = tip_link_for_group(planning_group)
        # The current state is already the path start.  Send every sampled
        # pose after it so MoveIt follows the explicit SLERP sequence.
        request.waypoints = copy.deepcopy(waypoints[1:])
        request.max_step = 0.005
        request.jump_threshold = 0.0
        request.avoid_collisions = True
        self.ui.post(
            self.ui.pipeline_waiting,
            f"Planning named TCP linear path · {len(waypoints)} SLERP samples",
        )
        finished = threading.Event()
        outcome = {}

        def response_ready(future):
            try:
                outcome["response"] = future.result()
            except Exception as error:
                outcome["error"] = str(error)
            finished.set()

        self.cartesian_planning_client.call_async(request).add_done_callback(
            response_ready
        )
        if not finished.wait(timeout=30.0):
            self.ui.post(self.ui.error, "Named TCP Cartesian planning timed out")
            return
        if "error" in outcome:
            self.ui.post(
                self.ui.error,
                f"Named TCP Cartesian planning failed: {outcome['error']}",
            )
            return
        response = outcome["response"]
        if response.fraction < 0.999:
            self.ui.post(
                self.ui.error,
                f"Named TCP Cartesian path planned only "
                f"{response.fraction:.1%}",
            )
            return
        scale_trajectory_speed(response.solution, velocity_scale)
        display = DisplayTrajectory()
        display.trajectory_start = response.start_state
        display.trajectory.append(response.solution)
        self.initial_planned_trajectory = copy.deepcopy(response.solution)
        self.initial_planned_pose_name = pose_name
        self.initial_planned_group = planning_group
        self.display_trajectory_publisher.publish(display)
        self.ui.post(
            self.ui.initial_position_plan_ready,
            planning_group,
            target_positions,
            velocity_scale,
            "Named TCP linear plan shown in RViz · XYZ linear + orientation SLERP",
        )

    def _initial_plan_goal_response(
        self,
        future,
        planning_group,
        target_positions,
        velocity_scale,
        pose_name,
    ):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Taught-pose plan failed: {error}",
            )
            return
        if not goal_handle.accepted:
            self.ui.post(self.ui.error, "Taught-pose plan was rejected")
            return
        goal_handle.get_result_async().add_done_callback(
            lambda result: self._initial_plan_result(
                result,
                planning_group,
                target_positions,
                velocity_scale,
                pose_name,
            )
        )

    def _initial_plan_result(
        self,
        future,
        planning_group,
        target_positions,
        velocity_scale,
        pose_name,
    ):
        try:
            result = future.result().result
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Taught-pose plan failed: {error}",
            )
            return
        if result.error_code.val != 1:
            self.ui.post(
                self.ui.error,
                "Taught-pose plan failed "
                f"(MoveIt code {result.error_code.val})",
            )
            return
        display = DisplayTrajectory()
        display.trajectory_start = result.trajectory_start
        display.trajectory.append(result.planned_trajectory)
        self.initial_planned_trajectory = copy.deepcopy(
            result.planned_trajectory
        )
        self.initial_planned_pose_name = pose_name
        self.initial_planned_group = planning_group
        self.display_trajectory_publisher.publish(display)
        self.ui.post(
            self.ui.initial_position_plan_ready,
            planning_group,
            target_positions,
            velocity_scale,
            "Taught-pose plan shown in RViz · ready to execute",
        )

    def execute_initial_plan(self):
        if not self.execute_motion_enabled:
            self.ui.post(
                self.ui.error,
                "Taught-pose execution is disabled by launch "
                "configuration",
            )
            return
        trajectory = self.initial_planned_trajectory
        if trajectory is None:
            self.ui.post(self.ui.error, "Plan the selected taught pose first")
            return
        if not self.execute_trajectory_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(
                self.ui.error,
                "ExecuteTrajectory action server unavailable",
            )
            return
        pose_name = self.initial_planned_pose_name
        planning_group = self.initial_planned_group
        self.initial_planned_trajectory = None
        self.initial_planned_pose_name = None
        self.initial_planned_group = None
        touch_guarded = bool(
            pose_name in DI8_GUARDED_TEACHING_POSES and planning_group
        )
        if touch_guarded:
            arm = planning_group.removesuffix("_manipulator")
            if self.node_touch_input_states.get(arm):
                self.ui.post(
                    self.ui.log,
                    "DI8 is already ON; new taught-pose execution is allowed. "
                    "Guard will stop only after DI8 releases and rises again",
                )
            self.touch_guard_triggered.clear()
            self.touch_guard_stop_complete.clear()
            self.touch_guard_stop_success = False
            self.active_touch_guard = (arm, pose_name)
            self.ui.post(
                self.ui.log,
                f"DI8 stop guard armed for approved taught pose: {pose_name}",
            )
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        self.ui.post(
            self.ui.pipeline_waiting,
            "Executing the approved taught-pose plan",
        )
        future = self.execute_trajectory_client.send_goal_async(goal)
        future.add_done_callback(
            lambda result: self._initial_execute_goal_response(
                result, touch_guarded
            )
        )

    def _initial_execute_goal_response(self, future, touch_guarded=False):
        try:
            goal_handle = future.result()
        except Exception as error:
            if touch_guarded:
                self.active_touch_guard = None
            self.ui.post(
                self.ui.error,
                f"Taught-pose execution failed: {error}",
            )
            return
        if not goal_handle.accepted:
            if touch_guarded:
                self.active_touch_guard = None
            self.ui.post(self.ui.error, "Taught-pose execution rejected")
            return
        self.active_motion_goal = goal_handle
        if touch_guarded and self.touch_guard_triggered.is_set():
            goal_handle.cancel_goal_async()
        goal_handle.get_result_async().add_done_callback(
            lambda result: self._initial_execute_result(result, touch_guarded)
        )

    def _initial_execute_result(self, future, touch_guarded=False):
        self.active_motion_goal = None
        try:
            result = future.result().result
        except Exception as error:
            self.ui.post(
                self.ui.error,
                f"Taught-pose execution failed: {error}",
            )
            if touch_guarded:
                self.active_touch_guard = None
            return
        if touch_guarded and self.touch_guard_triggered.is_set():
            stop_complete = self.touch_guard_stop_complete.wait(timeout=5.0)
            stop_success = self.touch_guard_stop_success
            self.active_touch_guard = None
            if not stop_complete:
                self.ui.post(
                    self.ui.error,
                    "Guarded taught-pose action ended, but DI8 stop cleanup timed out",
                )
            elif not stop_success:
                self.ui.post(
                    self.ui.error,
                    "Guarded taught-pose action ended, but controller recovery failed",
                )
            else:
                self.ui.post(
                    self.ui.pipeline_result,
                    "Guarded taught-pose execution stopped by DI8 · "
                    "execution closed · ready for the next command",
                )
            return
        if touch_guarded:
            self.active_touch_guard = None
        if result.error_code.val != 1:
            self.ui.post(
                self.ui.error,
                "Taught-pose execution failed "
                f"(MoveIt code {result.error_code.val})",
            )
            return
        self.ui.post(
            self.ui.initial_position_execution_finished,
            "Robot reached the selected taught pose",
        )

    def generate_weave(
        self,
        source_points,
        amplitude,
        cycles,
        samples_per_cycle,
        transverse_axis,
        visible,
    ):
        try:
            points = weaving_from_path(
                source_points,
                amplitude,
                cycles,
                samples_per_cycle,
                transverse_axis,
            )
        except ValueError as error:
            self.ui.post(self.ui.error, f"Weave generation failed: {error}")
            return
        self.publish_points(points, visible)
        self.ui.post(self.ui.set_new_points, points, "weave")
        self.ui.post(
            self.ui.log,
            f"Applied weave to taught seam · amplitude=±{amplitude:.3f} m, "
            f"cycles={cycles}, axis={transverse_axis}",
        )

    def capture_tcp(self, replace_index, visible, planning_group):
        try:
            pose = self._current_tcp_pose(planning_group)
        except TransformException as error:
            self.ui.post(self.ui.error, f"TCP capture failed: {error}")
            return
        self.ui.post(
            self.ui.apply_captured_tcp,
            pose,
            replace_index,
            visible,
        )

    def capture_linear_tcp(self, endpoint_index, planning_group):
        arm = "left" if planning_group.startswith("left") else "right"
        joint_names = [
            f"{arm}_manipulator_joint{index}" for index in range(1, 7)
        ]
        try:
            pose = self._current_tcp_pose(planning_group)
            positions = [self.latest_joint_positions[name] for name in joint_names]
        except KeyError:
            self.ui.post(
                self.ui.error,
                "Complete measured joint state is unavailable for TCP teaching",
            )
            return
        except TransformException as error:
            self.ui.post(self.ui.error, f"TCP capture failed: {error}")
            return
        self.ui.post(
            self.ui.apply_linear_tcp,
            endpoint_index,
            pose,
            planning_group,
            tuple(joint_names),
            tuple(positions),
        )

    def generate_tcp_line(self, start, end, count, visible):
        try:
            points = linear_pose_waypoints(start, end, count)
        except ValueError as error:
            self.ui.post(self.ui.error, f"TCP line generation failed: {error}")
            return
        distance = math.sqrt(
            (end.position.x - start.position.x) ** 2
            + (end.position.y - start.position.y) ** 2
            + (end.position.z - start.position.z) ** 2
        )
        self.publish_points(points, visible)
        self.ui.post(self.ui.set_new_points, points, "tcp_line")
        self.ui.post(
            self.ui.log,
            f"Generated endpoint-to-endpoint linear 6D path · "
            f"distance={distance * 1000.0:.1f} mm · {count} poses",
        )

    def submit_cartesian_motion(
        self,
        points,
        velocity_scale,
        interpolation_step,
        visualize_path,
        execute_requested,
        reuse_approved_plan,
        planning_group,
    ):
        if not points:
            self.ui.post(self.ui.error, "Create weld points first")
            return
        if not self.cartesian_motion_client.wait_for_server(timeout_sec=3.0):
            self.ui.post(
                self.ui.error,
                "cartesian_path action server unavailable",
            )
            return
        goal = CartesianPath.Goal()
        goal.planning_group = planning_group
        goal.interpolation_step = interpolation_step
        goal.velocity_scale = velocity_scale
        goal.execute_requested = execute_requested
        goal.reuse_approved_plan = reuse_approved_plan
        goal.visualize_path = visualize_path
        goal.waypoints = points
        self.request_execution = execute_requested
        self.ui.post(
            self.ui.begin,
            velocity_scale,
            execute_requested,
        )
        self.active_motion_goal = None
        future = self.cartesian_motion_client.send_goal_async(
            goal,
            feedback_callback=self._cartesian_feedback_received,
        )
        future.add_done_callback(self._cartesian_goal_response)

    def run_sequence_cartesian_motion(self, step, execute_requested):
        """Plan or execute one stored path and block only the worker thread."""
        goal = CartesianPath.Goal()
        goal.planning_group = step["planning_group"]
        goal.interpolation_step = step["interpolation_step"]
        goal.velocity_scale = step["velocity_scale"]
        goal.execute_requested = bool(execute_requested)
        goal.reuse_approved_plan = False
        goal.visualize_path = True
        goal.waypoints = copy.deepcopy(step["points"])
        touch_guarded = bool(execute_requested and step.get("touch_guard", False))
        arm = step["planning_group"].removesuffix("_manipulator")
        guard_name = step.get("path_kind", "Cartesian approach")
        if touch_guarded:
            if self.node_touch_input_states.get(arm):
                if step.get("accept_initial_touch", False):
                    return True, (
                        f"{guard_name} already at START contact (DI8 ON) · "
                        "approach skipped and weld stages will continue"
                    )
                if not step.get("allow_initial_touch_motion", False):
                    return False, (
                        f"DI8 is already ON; {guard_name} was not started"
                    )
                self.ui.post(
                    self.ui.log,
                    f"{guard_name} starts while DI8 is ON · Cartesian "
                    "retraction allowed; guard waits for a new rising edge",
                )
            self.touch_guard_triggered.clear()
            self.touch_guard_stop_complete.clear()
            self.touch_guard_stop_success = False
            self.active_motion_goal = None
            self.active_touch_guard = (arm, guard_name)
            self.ui.post(
                self.ui.log,
                f"DI8 stop guard armed only for {guard_name}",
            )
        try:
            result = self._send_action_goal_and_wait(
                self.cartesian_motion_client,
                goal,
                "Cartesian motion",
                on_accepted=(
                    lambda handle: handle.cancel_goal_async()
                    if touch_guarded and self.touch_guard_triggered.is_set()
                    else None
                ),
            )
            if touch_guarded and self.touch_guard_triggered.is_set():
                if not self.touch_guard_stop_complete.wait(timeout=5.0):
                    return False, f"{guard_name} DI8 stop confirmation timed out"
                if not self.touch_guard_stop_success:
                    return False, f"{guard_name} DI8 standstill was not confirmed"
                continue_after_touch = bool(
                    step.get("continue_after_touch", False)
                )
                return continue_after_touch, (
                    f"{guard_name} stopped by DI8 · "
                    + (
                        "continuing with unguarded weld stages"
                        if continue_after_touch
                        else "sequence stopped"
                    )
                )
            return bool(result.success), result.message
        except (RuntimeError, TimeoutError) as error:
            return False, str(error)
        finally:
            if touch_guarded and self.active_touch_guard == (arm, guard_name):
                self.active_touch_guard = None

    def run_sequence_named_pose(self, step, execute_requested):
        """Plan or plan-and-execute one taught joint pose."""
        tcp_target = bool(
            step.get("pose_name") in TCP_POSE_TEACHING_POSES
            and pose_is_valid(step.get("tcp_pose"))
        )
        if tcp_target:
            try:
                current_tcp = self._current_tcp_pose(step["planning_group"])
                points = named_tcp_linear_waypoints(
                    current_tcp, step["tcp_pose"]
                )
            except (TransformException, ValueError) as error:
                return False, f"Named TCP linear path failed: {error}"
            return self.run_sequence_cartesian_motion(
                {
                    "planning_group": step["planning_group"],
                    "interpolation_step": 0.005,
                    "velocity_scale": step["velocity_scale"],
                    "points": points,
                    "path_kind": f"{step['pose_label']} TCP linear",
                    "touch_guard": bool(step.get("touch_guard", True)),
                    "continue_after_touch": bool(
                        step.get("continue_after_touch", False)
                    ),
                    "allow_initial_touch_motion": True,
                },
                execute_requested,
            )
        goal = MoveGroup.Goal()
        goal.request.group_name = step["planning_group"]
        goal.request.num_planning_attempts = int(
            step.get("planning_attempts", 5)
        )
        goal.request.allowed_planning_time = float(
            step.get("planning_time", 5.0)
        )
        goal.request.start_state.is_diff = True
        goal.request.max_velocity_scaling_factor = step["velocity_scale"]
        goal.request.max_acceleration_scaling_factor = step["velocity_scale"]
        constraints = Constraints()
        for name, position in zip(step["joint_names"], step["positions"]):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = position
            constraint.tolerance_above = 0.001
            constraint.tolerance_below = 0.001
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = not bool(execute_requested)
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        touch_guarded = bool(
            execute_requested
            and (
                step.get("touch_guard", False)
                or step.get("pose_name") in DI8_GUARDED_TEACHING_POSES
            )
        )
        arm = step["planning_group"].removesuffix("_manipulator")
        if touch_guarded:
            if self.node_touch_input_states.get(arm):
                self.ui.post(
                    self.ui.log,
                    f"{step['pose_label']} starts while DI8 is ON · "
                    "allowed for contact retraction; guard waits for a new rising edge",
                )
            self.touch_guard_triggered.clear()
            self.touch_guard_stop_complete.clear()
            self.touch_guard_stop_success = False
            self.active_touch_guard = (arm, step["pose_name"])
            self.ui.post(
                self.ui.log,
                f"DI8 stop guard armed for {step['pose_label']} approach",
            )
        try:
            result = self._send_action_goal_and_wait(
                self.move_group_client,
                goal,
                "MoveGroup",
                on_accepted=(
                    lambda handle: handle.cancel_goal_async()
                    if touch_guarded and self.touch_guard_triggered.is_set()
                    else None
                ),
            )
            if touch_guarded and self.touch_guard_triggered.is_set():
                if not self.touch_guard_stop_complete.wait(timeout=5.0):
                    return False, "DI8 stop confirmation timed out"
                if not self.touch_guard_stop_success:
                    return False, "DI8 standstill was not confirmed"
                continue_after_touch = bool(
                    step.get("continue_after_touch", False)
                )
                return continue_after_touch, (
                    f"{step['pose_label']} stopped by DI8 · "
                    + (
                        "continuing sequence"
                        if continue_after_touch
                        else "sequence stopped"
                    )
                )
            success = result.error_code.val == 1
            if success and not execute_requested:
                display = DisplayTrajectory()
                display.trajectory_start = result.trajectory_start
                display.trajectory.append(result.planned_trajectory)
                self.display_trajectory_publisher.publish(display)
            return success, (
                f"{step['pose_label']} "
                f"{'reached' if execute_requested else 'planned in RViz'} · "
                "stored joint state"
                if success
                else f"MoveIt code {result.error_code.val}"
            )
        except (RuntimeError, TimeoutError) as error:
            return False, str(error)
        finally:
            if touch_guarded and self.active_touch_guard == (
                arm, step["pose_name"]
            ):
                self.active_touch_guard = None

    def run_sequence_planned_trajectory(self, step, execute_requested):
        """Preview or execute trajectories captured from RViz without replanning."""
        display = DisplayTrajectory()
        display.model_id = step.get("model_id", "")
        display.trajectory_start = copy.deepcopy(step["trajectory_start"])
        display.trajectory = copy.deepcopy(step["trajectories"])
        if not execute_requested:
            self.display_trajectory_publisher.publish(display)
            return True, (
                f"stored RViz plan previewed · "
                f"{len(display.trajectory)} trajectory(s)"
            )
        for index, trajectory in enumerate(display.trajectory, start=1):
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = copy.deepcopy(trajectory)
            try:
                result = self._send_action_goal_and_wait(
                    self.execute_trajectory_client,
                    goal,
                    f"RViz trajectory {index}",
                )
            except (RuntimeError, TimeoutError) as error:
                return False, str(error)
            if result.error_code.val != 1:
                return False, (
                    f"RViz trajectory {index} failed · "
                    f"MoveIt code {result.error_code.val}"
                )
        return True, (
            f"executed exact stored RViz plan · "
            f"{len(display.trajectory)} trajectory(s)"
        )

    def _cartesian_feedback_received(self, message):
        feedback = message.feedback
        self.ui.post(
            self.ui.progress,
            feedback.progress,
            feedback.waypoint_index,
            feedback.current_pose,
            feedback.phase,
        )

    def _cartesian_goal_response(self, future):
        try:
            self.active_motion_goal = future.result()
        except Exception as error:
            self.active_motion_goal = None
            self.ui.post(self.ui.error, str(error))
            return
        if not self.active_motion_goal.accepted:
            self.active_motion_goal = None
            self.ui.post(self.ui.error, "Action goal rejected")
            return
        operation = (
            "approved trajectory execution"
            if self.request_execution
            else "MoveIt plan preview"
        )
        self.ui.post(self.ui.log, f"Action accepted · {operation}")
        result = self.active_motion_goal.get_result_async()
        result.add_done_callback(
            lambda completed, handle=self.active_motion_goal: (
                self._cartesian_result_received(completed, handle)
            )
        )

    def _cartesian_result_received(self, future, goal_handle=None):
        if (
            goal_handle is not None
            and goal_handle is not self.active_motion_goal
        ):
            self.ui.post(
                self.ui.log,
                "Ignored completion from a superseded Cartesian goal",
            )
            return
        result = future.result().result
        if goal_handle is self.active_motion_goal:
            self.active_motion_goal = None
        if result.success:
            self.ui.post(
                self.ui.finish,
                f"SUCCESS · {len(result.sampled_path)} samples · "
                f"{result.message}",
                self.request_execution,
            )
        elif self.active_touch_probe is not None:
            self.ui.post(
                self.ui.log,
                f"Expected probe trajectory interruption · {result.message}",
            )
        else:
            self.ui.post(self.ui.error, result.message)

    def cancel_active_motion(self):
        if self.active_motion_goal is not None:
            self.active_motion_goal.cancel_goal_async()
            self.ui.post(self.ui.log, "Cancel requested")


class WeldActionGui:
    """Tk GUI for acquiring, editing, visualizing, and running weld paths."""

    POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")

    def _create_toggle_section(
        self,
        parent,
        key,
        title,
        expanded=False,
    ):
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, pady=2)
        button = ttk.Button(
            container,
            command=lambda selected=key: self.toggle_motion_section(selected),
        )
        button.pack(fill=tk.X)
        body = ttk.Frame(container, padding=(8, 5))
        self.motion_sections[key] = {
            "body": body,
            "button": button,
            "title": title,
            "expanded": bool(expanded),
        }
        if expanded:
            body.pack(fill=tk.X)
        self._refresh_motion_section_button(key)
        return body

    def _refresh_motion_section_button(self, key):
        section = self.motion_sections[key]
        marker = "▼" if section["expanded"] else "▶"
        section["button"].configure(
            text=f"{marker}  {section['title']}",
        )

    def toggle_motion_section(self, key):
        section = self.motion_sections[key]
        section["expanded"] = not section["expanded"]
        if section["expanded"]:
            section["body"].pack(fill=tk.X)
        else:
            section["body"].pack_forget()
        self._refresh_motion_section_button(key)
        self.root.after_idle(self._update_scroll_region)

    @staticmethod
    def _add_labeled_value(parent, pair_index, label, variable, width=8):
        column = pair_index * 2
        ttk.Label(parent, text=label).grid(
            row=0, column=column, padx=(6, 2), pady=3, sticky=tk.E
        )
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=0, column=column + 1, padx=(2, 6), pady=3
        )

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Editable Cartesian Action")
        self.root.geometry("1240x940")
        self._closing = False
        self._ui_queue = queue.SimpleQueue()
        self._latest_ui_updates = {}
        self._latest_ui_updates_lock = threading.Lock()
        self.points = []
        self.weave_source = []
        self.weave_base_paths = {"linear": [], "circle": []}
        self.path_kind = "empty"
        self.execution_allowed = False
        self.robot_connected = {
            "left": False,
            "right": False,
            "head": False,
        }
        self.fake_head_hardware = False
        self.plan_approved = False
        self.linear_tcp_endpoints = [None, None]
        self.initial_joint_state = None
        self.initial_plan_ready = False
        self.teaching_pose_name = tk.StringVar(
            value=TEACHING_POSES["robot_start"]
        )
        self.taught_robot_poses = {name: None for name in TEACHING_POSES}
        self.pose_variables = {
            name: tk.StringVar(value="0.0") for name in self.POSE_FIELDS
        }
        self.radius_mm = tk.DoubleVar(value=20.0)
        self.circle_count = tk.IntVar(value=16)
        self.close_circle = tk.BooleanVar(value=True)
        self.circle_face_center = tk.BooleanVar(value=True)
        self.circle_axis = tk.StringVar(value="X")
        self.nudge_mm = tk.DoubleVar(value=5.0)
        self.velocity_percent = tk.DoubleVar(value=20.0)
        self.interpolation_step_mm = tk.DoubleVar(value=5.0)
        self.show_path = tk.BooleanVar(value=True)
        self.weave_amplitude_mm = tk.DoubleVar(value=3.0)
        self.weave_cycles = tk.IntVar(value=4)
        self.weave_samples = tk.IntVar(value=8)
        self.weave_axis = tk.StringVar(value="tool_y")
        self.weave_base = tk.StringVar(value="linear")
        self.straight_reference = tk.StringVar(value="world")
        self.straight_axis = tk.StringVar(value="+X")
        self.straight_start_mode = tk.StringVar(value="Current TCP")
        self.straight_start_x = tk.DoubleVar(value=0.0)
        self.straight_start_y = tk.DoubleVar(value=0.0)
        self.straight_start_z = tk.DoubleVar(value=0.0)
        self.straight_distance_mm = tk.DoubleVar(value=150.0)
        self.straight_count = tk.IntVar(value=5)
        self.tcp_line_count = tk.IntVar(value=10)
        self.tcp_line_direction = tk.StringVar(value="TCP 1 → TCP 2")
        self.straight_roll_deg = tk.DoubleVar(value=0.0)
        self.straight_pitch_deg = tk.DoubleVar(value=0.0)
        self.straight_yaw_deg = tk.DoubleVar(value=0.0)
        self.straight_rotation_reference = tk.StringVar(value="tool")
        self.planning_group = tk.StringVar(value="right_manipulator")
        self.rbpodo_welder_ready = False
        self.latest_right_system_state = None
        self.hicomm_connected = False
        self.hicomm_client = None
        self.hicomm_source_ip = tk.StringVar(value="192.168.1.2")
        self.hicomm_welder_ip = tk.StringVar(value="192.168.1.10")
        self.hicomm_port = tk.IntVar(value=60000)
        self.hicomm_arc_unlocked = tk.BooleanVar(value=False)
        self.hicomm_gas_enabled = tk.BooleanVar(value=False)
        self.hicomm_inching_direction = None
        self.inching_distance_lock = threading.Lock()
        self.inching_total_mm = 0.0
        self.inching_forward_mm = 0.0
        self.inching_reverse_mm = 0.0
        self.inching_last_status_time = None
        self.hicomm_feedback_last_log_time = 0.0
        self.hicomm_feedback_last_signature = None
        self.hicomm_feedback_log_period_s = 0.2
        self.hicomm_feedback_idle_log_period_s = 1.0
        self.touch_sensing_enabled = tk.BooleanVar(value=False)
        self.corner_touch_target = tk.StringVar(value="start_floor")
        self.corner_touch_count = tk.IntVar(value=10)
        self.corner_touches = {name: None for name in CORNER_TOUCH_NAMES}
        # seam_axis is retained only for backward-compatible touch YAML / legacy
        # helpers.  New DI8 probing uses explicit probe directions below.
        self.seam_axis = tk.StringVar(value="X")
        self.wall_probe_axis = tk.StringVar(value="AUTO ⟂ taught seam (XY)")
        self.floor_probe_axis = tk.StringVar(value="World Z")
        self.seam_orientation_mode = tk.StringVar(
            value="Follow sensed seam yaw"
        )
        self.reference_yaw_status = tk.StringVar(value="Reference yaw: --")
        self.sensed_yaw_status = tk.StringVar(value="Sensed yaw: --")
        self.delta_yaw_status = tk.StringVar(value="ΔYaw: --")
        self.reference_length_status = tk.StringVar(value="Length: --")
        self.quick_teaching_status = tk.StringVar(
            value="Auxiliary teaching: not captured"
        )
        self.wall_probe_sign = tk.StringVar(value="-")
        self.floor_probe_sign = tk.StringVar(value="-")
        self.touch_probe_distance_mm = tk.DoubleVar(value=25.0)
        self.touch_probe_speed_percent = tk.DoubleVar(value=5.0)
        self.touch_settle_seconds = tk.DoubleVar(value=0.7)
        self.seam_wall_offset_mm = tk.DoubleVar(value=0.0)
        self.seam_floor_offset_mm = tk.DoubleVar(value=0.0)
        self.seam_probe_touches = {
            name: None for name in CORNER_TOUCH_NAMES
        }
        self.seam_probe_starts = {
            name: None for name in CORNER_TOUCH_NAMES
        }
        self.seam_probe_stops = {
            name: None for name in CORNER_TOUCH_NAMES
        }
        self.raw_two_touch_seam = []
        self.corrected_two_touch_seam = []
        self.computed_seam_endpoints = {"start": None, "goal": None}
        self.computed_seam_wait_points = {"start": None, "goal": None}
        self.seam_teaching_reference = None
        self.automatic_probe_kind = None
        self.seam_auto_running = False
        self.seam_auto_stage_event = threading.Event()
        self.seam_auto_stage_success = False
        self.seam_auto_expected_kind = None
        self.seam_auto_returned_kinds = set()
        self.sequence_steps = []
        self.sequence_sleep_seconds = tk.DoubleVar(value=1.0)
        self.sequence_parallel_slot = tk.IntVar(value=1)
        self.sequence_duration_seconds = tk.DoubleVar(value=3.0)
        self.sequence_edit_velocity_percent = tk.DoubleVar(value=20.0)
        self.sequence_edit_touch_guard = tk.BooleanVar(value=False)
        self.sequence_edit_continue_after_touch = tk.BooleanVar(value=False)
        self.sequence_running = False
        self.sequence_stop_requested = False
        self.last_action_phase = ""
        self.previous_control_box_io = None
        self.touch_input_states = {"left": None, "right": None}
        self.touch_input_rising_edges = {"left": 0, "right": 0}
        self.last_touch_pose = None
        self.motion_sections = {}
        self.control_box_io_labels = {}
        self.pending_do_ports = set()
        self.unlock_all_do_ports = tk.BooleanVar(value=False)
        # Reproduce the successful v5.2 Rainbow capture byte-for-byte by
        # default.  Operators can apply a different recipe explicitly.
        weld_defaults = DEFAULT_DIGITAL_WELD_SETTINGS
        self.weld_current_raw = tk.IntVar(value=weld_defaults["current_a"])
        self.weld_voltage_raw = tk.IntVar(
            value=weld_defaults["voltage_tenths"]
        )
        self.weld_material = tk.StringVar(value=weld_defaults["material"])
        self.weld_diameter_mm = tk.DoubleVar(
            value=weld_defaults["diameter_mm"]
        )
        self.weld_mode = tk.StringVar(value=weld_defaults["mode"])
        self.weld_gas = tk.StringVar(value=weld_defaults["gas"])
        self.weld_synergic = tk.BooleanVar(value=weld_defaults["synergic"])
        self.weld_correction = tk.DoubleVar(value=weld_defaults["correction"])
        self.weld_pre_gas_s = tk.DoubleVar(value=weld_defaults["pre_gas_s"])
        self.weld_post_gas_s = tk.DoubleVar(value=weld_defaults["post_gas_s"])
        self.weld_preflow_seconds = tk.DoubleVar(
            value=weld_defaults["preflow_seconds"]
        )
        self.robot_ips = {
            "left": "192.168.1.12",
            "right": "192.168.1.19",
        }

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 18, "bold"))
        style.configure("Step.TLabel", font=("Sans", 11, "bold"))

        scroll_container = ttk.Frame(self.root)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        self.content_canvas = tk.Canvas(
            scroll_container,
            highlightthickness=0,
        )
        content_scrollbar = ttk.Scrollbar(
            scroll_container,
            orient=tk.VERTICAL,
            command=self.content_canvas.yview,
        )
        self.content_canvas.configure(yscrollcommand=content_scrollbar.set)
        content_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.content_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        outer = ttk.Frame(self.content_canvas, padding=16)
        self.content_window = self.content_canvas.create_window(
            (0, 0),
            window=outer,
            anchor=tk.NW,
        )
        outer.bind("<Configure>", self._update_scroll_region)
        self.content_canvas.bind("<Configure>", self._resize_scroll_content)
        self.root.bind_all("<MouseWheel>", self._scroll_content)
        self.root.bind_all("<Button-4>", self._scroll_content)
        self.root.bind_all("<Button-5>", self._scroll_content)
        ttk.Label(
            outer,
            text="Welding Interface",
            style="Title.TLabel",
        ).pack(anchor=tk.W)

        # Connection state is deliberately first: no motion or welding control
        # should be interpreted before the operator checks these indicators.
        robot_status = ttk.LabelFrame(outer, text="Connection")
        robot_status.pack(fill=tk.X, pady=(5, 8))
        self.robot_connection_labels = {}
        for arm in ("left", "right"):
            label = tk.Label(
                robot_status,
                text=f"Connect {arm.upper()} (IP): X",
                width=32,
                relief=tk.SOLID,
                borderwidth=1,
                bg="#fce8e6",
                fg="#b3261e",
                font=("Sans", 11, "bold"),
            )
            label.pack(side=tk.LEFT, padx=6, pady=6)
            self.robot_connection_labels[arm] = label
        head_label = tk.Label(
            robot_status,
            text="Connect HEAD (CAN2): X",
            width=26,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#fce8e6",
            fg="#b3261e",
            font=("Sans", 11, "bold"),
        )
        head_label.pack(side=tk.LEFT, padx=6, pady=6)
        self.robot_connection_labels["head"] = head_label
        self.welder_connection_label = tk.Label(
            robot_status,
            text="HICOMM WELDER: X",
            width=22,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#fce8e6",
            fg="#b3261e",
            font=("Sans", 11, "bold"),
        )
        self.welder_connection_label.pack(side=tk.LEFT, padx=6, pady=6)
        tk.Button(
            robot_status,
            text="EMERGENCY STOP (SOFTWARE)\nALL MOTION + WELDER",
            command=self.emergency_stop_all,
            bg="#b3261e",
            fg="white",
            activebackground="#7f1d1d",
            activeforeground="white",
            font=("Sans", 11, "bold"),
            relief=tk.RAISED,
            borderwidth=3,
            padx=12,
            pady=3,
        ).pack(side=tk.RIGHT, padx=8, pady=4)

        arm_selection = ttk.Frame(outer)
        arm_selection.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(
            arm_selection,
            text="Cartesian arm:",
            style="Step.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Combobox(
            arm_selection,
            textvariable=self.planning_group,
            values=("right_manipulator", "left_manipulator"),
            state="readonly",
            width=22,
        ).pack(side=tk.LEFT)
        self.planning_group.trace_add("write", self.arm_changed)

        motion_tests = self._create_toggle_section(
            outer, "motion_test", "Motion Test", expanded=False
        )
        straight = ttk.Frame(motion_tests)
        straight.pack(fill=tk.X, pady=2)
        ttk.Button(
            straight,
            text="Generate linear path",
            command=self.acquire,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(straight, text="reference").pack(side=tk.LEFT)
        ttk.Combobox(
            straight,
            textvariable=self.straight_reference,
            values=("world", "tool"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 7))
        ttk.Label(straight, text="axis").pack(side=tk.LEFT)
        ttk.Combobox(
            straight,
            textvariable=self.straight_axis,
            values=("+X", "-X", "+Y", "-Y", "+Z", "-Z"),
            state="readonly",
            width=4,
        ).pack(side=tk.LEFT, padx=(3, 7))
        ttk.Label(straight, text="distance mm").pack(side=tk.LEFT)
        ttk.Spinbox(
            straight,
            from_=0.1,
            to=5000,
            increment=1,
            textvariable=self.straight_distance_mm,
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 6))
        ttk.Label(straight, text="points").pack(side=tk.LEFT)
        ttk.Spinbox(
            straight,
            from_=2,
            to=200,
            increment=1,
            textvariable=self.straight_count,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 0))

        straight_angles = ttk.Frame(motion_tests)
        straight_angles.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(
            straight_angles,
            text="Angle adjustment",
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(straight_angles, text="reference").pack(side=tk.LEFT)
        ttk.Combobox(
            straight_angles,
            textvariable=self.straight_rotation_reference,
            values=("tool", "world"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 7))
        for label, variable in (
            ("ΔRoll °", self.straight_roll_deg),
            ("ΔPitch °", self.straight_pitch_deg),
            ("ΔYaw °", self.straight_yaw_deg),
        ):
            ttk.Label(straight_angles, text=label).pack(
                side=tk.LEFT, padx=(5, 2)
            )
            ttk.Entry(
                straight_angles, textvariable=variable, width=7
            ).pack(side=tk.LEFT)
        ttk.Label(
            straight_angles,
            text="Applied to every generated path TCP orientation",
            foreground="#5f6368",
        ).pack(side=tk.LEFT, padx=10)

        controls = ttk.Frame(motion_tests)
        controls.pack(fill=tk.X, pady=2)
        ttk.Button(
            controls,
            text="Generate circle",
            command=self.generate_circle,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(controls, text="axis").pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=self.circle_axis,
            values=("X", "Y", "Z"),
            state="readonly",
            width=3,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(controls, text="radius (mm)").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls,
            from_=1,
            to=200,
            increment=1,
            textvariable=self.radius_mm,
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(controls, text="unique points").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls,
            from_=4,
            to=200,
            increment=1,
            textvariable=self.circle_count,
            width=5,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Checkbutton(
            controls,
            text="close path",
            variable=self.close_circle,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            controls,
            text="TCP +Z faces center",
            variable=self.circle_face_center,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            motion_tests,
            text="show planned path",
            variable=self.show_path,
            command=self.toggle_path_visibility,
        ).pack(anchor=tk.W, pady=(3, 0))

        welder = self._create_toggle_section(
            outer, "welder", "Digital Welder · Hi-COMM TCP", expanded=False
        )
        ttk.Label(
            welder,
            text=(
                "Welder controls: ON by default · Robot motion: ROS 2/RBPodo "
                "· welding: direct Hi-COMM TX55/40 ms/RX71"
            ),
            foreground="#b06000",
        ).pack(anchor=tk.W, padx=4, pady=2)

        network = ttk.LabelFrame(welder, text="Hi-COMM network")
        network.pack(fill=tk.X, pady=2)
        for label, variable, width in (
            ("PC source IP", self.hicomm_source_ip, 15),
            ("Hi-COMM IP", self.hicomm_welder_ip, 15),
            ("port", self.hicomm_port, 7),
        ):
            ttk.Label(network, text=label).pack(side=tk.LEFT, padx=(6, 2))
            ttk.Entry(network, textvariable=variable, width=width).pack(
                side=tk.LEFT, padx=(0, 6)
            )
        self.hicomm_connect_button = ttk.Button(
            network, text="Connect", command=self.connect_hicomm
        )
        self.hicomm_connect_button.pack(side=tk.LEFT, padx=3)
        self.hicomm_disconnect_button = ttk.Button(
            network,
            text="Disconnect",
            command=self.disconnect_hicomm,
            state=tk.DISABLED,
        )
        self.hicomm_disconnect_button.pack(side=tk.LEFT, padx=3)

        welder_test = self._create_toggle_section(
            outer,
            "welder_test",
            "Welder Test · Hi-COMM physical outputs",
            expanded=False,
        )
        ttk.Label(
            welder_test,
            text=(
                "Available when Welder controls and Hi-COMM are connected · "
                "disconnect sends ALL OUTPUT OFF"
            ),
            foreground="#b3261e",
        ).pack(anchor=tk.W, padx=4, pady=2)

        wire_test = ttk.LabelFrame(
            welder_test, text="Wire inching / gas test"
        )
        wire_test.pack(fill=tk.X, pady=2)
        self.hicomm_forward_button = ttk.Button(
            wire_test, text="Hold: forward inch", state=tk.DISABLED
        )
        self.hicomm_forward_button.pack(side=tk.LEFT, padx=4, pady=3)
        self.hicomm_reverse_button = ttk.Button(
            wire_test, text="Hold: reverse inch", state=tk.DISABLED
        )
        self.hicomm_reverse_button.pack(side=tk.LEFT, padx=4, pady=3)
        for button, direction in (
            (self.hicomm_forward_button, "forward"),
            (self.hicomm_reverse_button, "reverse"),
        ):
            button.bind(
                "<ButtonPress-1>",
                lambda _event, selected=direction: self.request_hicomm_inching(
                    selected, True
                ),
            )
            button.bind(
                "<ButtonRelease-1>",
                lambda _event, selected=direction: self.request_hicomm_inching(
                    selected, False
                ),
            )
            button.bind(
                "<Leave>",
                lambda _event, selected=direction: self.request_hicomm_inching(
                    selected, False
                ),
            )
        self.hicomm_gas_check = ttk.Checkbutton(
            wire_test,
            text="Gas test",
            variable=self.hicomm_gas_enabled,
            command=self.request_hicomm_gas,
            state=tk.DISABLED,
        )
        self.hicomm_gas_check.pack(side=tk.LEFT, padx=8)
        self.hicomm_all_off_button = ttk.Button(
            wire_test,
            text="ALL OUTPUT OFF",
            command=self.clear_hicomm_test_outputs,
        )
        self.hicomm_all_off_button.pack(side=tk.LEFT, padx=8)
        self.hicomm_test_status = ttk.Label(wire_test, text="test locked")
        self.hicomm_test_status.pack(side=tk.LEFT, padx=8)
        ttk.Button(
            wire_test,
            text="Reset inch length",
            command=self.reset_inching_distance,
        ).pack(side=tk.LEFT, padx=4)

        digital = ttk.LabelFrame(
            welder_test, text="ARC SET / ARC ON / ARC OFF"
        )
        digital.pack(fill=tk.X, pady=2)
        self._add_labeled_value(digital, 0, "current A", self.weld_current_raw)
        self._add_labeled_value(
            digital, 1, "voltage ×0.1 V", self.weld_voltage_raw
        )
        self.hicomm_arc_set_button = ttk.Button(
            digital,
            text="ARC SET (I/V TX)",
            command=self.request_digital_weld_set,
            state=tk.DISABLED,
        )
        self.hicomm_arc_set_button.grid(row=0, column=4, padx=5)
        self.hicomm_arc_unlock_check = ttk.Checkbutton(
            digital,
            text="Unlock ARC ON",
            variable=self.hicomm_arc_unlocked,
            command=self.hicomm_arc_unlock_changed,
            state=tk.DISABLED,
        )
        self.hicomm_arc_unlock_check.grid(row=0, column=5, padx=5)
        self.hicomm_arc_on_button = ttk.Button(
            digital,
            text="ARC ON",
            command=lambda: self.request_digital_arc(True),
            state=tk.DISABLED,
        )
        self.hicomm_arc_on_button.grid(row=0, column=6, padx=3)
        self.hicomm_arc_off_button = ttk.Button(
            digital,
            text="ARC OFF",
            command=lambda: self.request_digital_arc(False),
        )
        self.hicomm_arc_off_button.grid(row=0, column=7, padx=3)
        self.hicomm_weld_status = ttk.Label(
            digital, text="DISCONNECTED · ARC OFF"
        )
        self.hicomm_weld_status.grid(row=0, column=8, padx=8)
        for column, (label, variable, values, width) in enumerate((
            ("material", self.weld_material, tuple(MATERIAL_CODES), 11),
            ("diameter", self.weld_diameter_mm, tuple(DIAMETER_CODES), 5),
            ("mode", self.weld_mode, tuple(MODE_CODES), 5),
            ("gas", self.weld_gas, tuple(GAS_CODES), 14),
        )):
            ttk.Label(digital, text=label).grid(
                row=1, column=column * 2, padx=(3, 2), pady=3
            )
            ttk.Combobox(
                digital,
                textvariable=variable,
                values=values,
                state="readonly",
                width=width,
            ).grid(row=1, column=column * 2 + 1, padx=(0, 4), pady=3)
        ttk.Checkbutton(
            digital, text="synergic", variable=self.weld_synergic
        ).grid(row=2, column=0, columnspan=2, padx=3, sticky=tk.W)
        for offset, (label, variable) in enumerate((
            ("correction", self.weld_correction),
            ("recipe pre-gas s", self.weld_pre_gas_s),
            ("recipe post-gas s", self.weld_post_gas_s),
        )):
            ttk.Label(digital, text=label).grid(
                row=2, column=2 + offset * 2, padx=(3, 2), pady=3
            )
            ttk.Entry(digital, textvariable=variable, width=7).grid(
                row=2, column=3 + offset * 2, padx=(0, 4), pady=3
            )
        ttk.Label(digital, text="pre-weld gas flow s").grid(
            row=3, column=0, padx=(3, 2), pady=3
        )
        ttk.Entry(
            digital, textvariable=self.weld_preflow_seconds, width=7
        ).grid(row=3, column=1, padx=(0, 6), pady=3)
        self.hicomm_rx_bit_status = ttk.Label(
            digital,
            text="RX Byte0 · b5 WCR=0 · b4 STICK=0 · "
            "b3 GAS CHECK=0 · b0 TORCH=0",
            font=("Monospace", 10, "bold"),
        )
        self.hicomm_rx_bit_status.grid(
            row=3, column=2, columnspan=7, padx=8, pady=3, sticky=tk.W
        )

        tcp_teaching = self._create_toggle_section(
            outer, "tcp_teaching", "TCP Teaching · Seam Reference", expanded=True
        )

        reference_row = ttk.Frame(tcp_teaching)
        reference_row.pack(fill=tk.X, pady=2)
        ttk.Label(
            reference_row,
            text="Reference seam",
            font=("Sans", 10, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            reference_row,
            text="Teach TCP 1 / START",
            command=lambda: self.capture_linear_tcp(0),
        ).pack(side=tk.LEFT, padx=(0, 5))
        self.tcp_1_status = ttk.Label(reference_row, text="not saved")
        self.tcp_1_status.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(
            reference_row,
            text="Teach TCP 2 / GOAL",
            command=lambda: self.capture_linear_tcp(1),
        ).pack(side=tk.LEFT, padx=(0, 5))
        self.tcp_2_status = ttk.Label(reference_row, text="not saved")
        self.tcp_2_status.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(
            reference_row,
            text="Load reference YAML",
            command=self.load_seam_reference,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(reference_row, text="preview points").pack(side=tk.LEFT)
        ttk.Spinbox(
            reference_row,
            from_=2,
            to=200,
            increment=1,
            textvariable=self.tcp_line_count,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 5))
        self.generate_tcp_line_button = ttk.Button(
            reference_row,
            text="Preview reference line",
            command=self.acquire_two_tcp,
            state=tk.DISABLED,
        )
        self.generate_tcp_line_button.pack(side=tk.LEFT)

        reference_status = ttk.Frame(tcp_teaching)
        reference_status.pack(fill=tk.X, pady=(0, 3))
        for variable in (
            self.reference_yaw_status,
            self.reference_length_status,
            self.sensed_yaw_status,
            self.delta_yaw_status,
        ):
            ttk.Label(reference_status, textvariable=variable).pack(
                side=tk.LEFT, padx=(0, 16)
            )
        ttk.Label(
            reference_status,
            text="TCP 1/2 are the immutable nominal seam reference used for yaw correction.",
        ).pack(side=tk.LEFT, padx=(4, 0))

        auxiliary_row = ttk.Frame(tcp_teaching)
        auxiliary_row.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(
            auxiliary_row,
            text="Auxiliary poses",
            font=("Sans", 10, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        for text, pose_name in (
            ("Teach Weld WAIT", "weld_wait"),
            ("Teach START WAIT", "weld_start_wait"),
            ("Teach GOAL WAIT", "weld_goal_wait"),
            ("Teach Weld END", "weld_finish"),
        ):
            ttk.Button(
                auxiliary_row,
                text=text,
                command=lambda name=pose_name: self.quick_capture_teaching_pose(name),
            ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(
            auxiliary_row, textvariable=self.quick_teaching_status
        ).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            tcp_teaching,
            text=(
                "Required for seam correction: TCP 1 + TCP 2.  "
                "START/GOAL WAIT are used by automatic sensing; Weld END is optional post-weld teaching."
            ),
        ).pack(anchor=tk.W, pady=(0, 2))

        teaching = ttk.LabelFrame(outer, text="Teaching Detail · Plan / Execute / YAML")
        teaching.pack(fill=tk.X, pady=(7, 0))
        ttk.Label(teaching, text="pose").pack(side=tk.LEFT, padx=(3, 2))
        teaching_pose_box = ttk.Combobox(
            teaching,
            textvariable=self.teaching_pose_name,
            values=tuple(TEACHING_POSES.values()),
            state="readonly",
            width=23,
        )
        teaching_pose_box.pack(side=tk.LEFT, padx=3)
        teaching_pose_box.bind(
            "<<ComboboxSelected>>", self.teaching_pose_changed
        )
        ttk.Button(
            teaching,
            text="Capture current + save YAML",
            command=self.capture_initial_state,
        ).pack(side=tk.LEFT, padx=3)
        self.plan_initial_button = ttk.Button(
            teaching,
            text="1 · Plan selected pose",
            command=self.plan_initial_state,
            state=tk.DISABLED,
        )
        self.plan_initial_button.pack(side=tk.LEFT, padx=3)
        self.execute_initial_button = ttk.Button(
            teaching,
            text="2 · Execute selected plan",
            command=self.execute_initial_plan,
            state=tk.DISABLED,
        )
        self.execute_initial_button.pack(side=tk.LEFT, padx=3)
        ttk.Button(
            teaching,
            text="Load from YAML",
            command=self.load_initial_state,
        ).pack(side=tk.LEFT, padx=3)
        self.initial_state_status = ttk.Label(teaching, text="not captured")
        self.initial_state_status.pack(side=tk.LEFT, padx=(12, 0))
        self.path_summary = ttk.Label(teaching, text="empty path")

        path_tests = self._create_toggle_section(
            outer, "path_test", "Path Generation · Weave", expanded=False
        )
        weaving = ttk.Frame(path_tests)
        weaving.pack(fill=tk.X, pady=2)
        ttk.Button(
            weaving,
            text="Generate weave path",
            command=self.generate_weave,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(weaving, text="base").pack(side=tk.LEFT)
        ttk.Combobox(
            weaving,
            textvariable=self.weave_base,
            values=("linear", "circle"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 8))
        for label, variable, start, end in (
            ("one-side amplitude mm", self.weave_amplitude_mm, 0.1, 50),
            ("weave count", self.weave_cycles, 1, 30),
            ("samples/cycle", self.weave_samples, 4, 30),
        ):
            ttk.Label(weaving, text=label).pack(side=tk.LEFT)
            ttk.Spinbox(
                weaving,
                from_=start,
                to=end,
                textvariable=variable,
                width=6,
            ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(weaving, text="transverse axis").pack(side=tk.LEFT)
        ttk.Combobox(
            weaving,
            textvariable=self.weave_axis,
            values=(
                "tool_x",
                "tool_y",
                "tool_z",
                "world_x",
                "world_y",
                "world_z",
            ),
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(3, 8))
        self.weave_summary = ttk.Label(
            weaving,
            text="Apply after teaching a seam",
        )
        self.weave_summary.pack(side=tk.LEFT, padx=(8, 0))

        touch_corner = self._create_toggle_section(
            outer,
            "touch_corner",
            "Seam Correction · DI8 wall/base probing + seam-yaw orientation",
        )

        geometry_controls = ttk.Frame(touch_corner)
        geometry_controls.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(geometry_controls, text="wall probe direction").pack(side=tk.LEFT)
        ttk.Combobox(
            geometry_controls,
            textvariable=self.wall_probe_axis,
            values=(
                "AUTO ⟂ taught seam (XY)",
                "World X",
                "World Y",
                "World Z",
            ),
            state="readonly",
            width=24,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(geometry_controls, text="sign").pack(side=tk.LEFT)
        ttk.Combobox(
            geometry_controls,
            textvariable=self.wall_probe_sign,
            values=("+", "-"),
            state="readonly",
            width=2,
        ).pack(side=tk.LEFT, padx=(3, 10))

        ttk.Label(geometry_controls, text="base/floor probe direction").pack(side=tk.LEFT)
        ttk.Combobox(
            geometry_controls,
            textvariable=self.floor_probe_axis,
            values=("World X", "World Y", "World Z"),
            state="readonly",
            width=9,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(geometry_controls, text="sign").pack(side=tk.LEFT)
        ttk.Combobox(
            geometry_controls,
            textvariable=self.floor_probe_sign,
            values=("+", "-"),
            state="readonly",
            width=2,
        ).pack(side=tk.LEFT, padx=(3, 10))

        ttk.Label(geometry_controls, text="orientation").pack(side=tk.LEFT)
        ttk.Combobox(
            geometry_controls,
            textvariable=self.seam_orientation_mode,
            values=(
                "Follow sensed seam yaw",
                "Keep reference orientation",
            ),
            state="readonly",
            width=27,
        ).pack(side=tk.LEFT, padx=(4, 8))

        yaw_summary = ttk.Frame(touch_corner)
        yaw_summary.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(yaw_summary, text="Orientation status", font=("Sans", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        for variable in (
            self.reference_yaw_status,
            self.sensed_yaw_status,
            self.delta_yaw_status,
        ):
            ttk.Label(yaw_summary, textvariable=variable).pack(
                side=tk.LEFT, padx=(0, 14)
            )

        offset_controls = ttk.Frame(touch_corner)
        offset_controls.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(offset_controls, text="wall plane offset mm").pack(side=tk.LEFT)
        ttk.Entry(
            offset_controls,
            textvariable=self.seam_wall_offset_mm,
            width=6,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(offset_controls, text="base plane offset mm").pack(side=tk.LEFT)
        ttk.Entry(
            offset_controls,
            textvariable=self.seam_floor_offset_mm,
            width=6,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(
            offset_controls,
            text=(
                "AUTO wall = World-XY normal of taught START→GOAL; "
                "positive offsets follow each configured +normal"
            ),
        ).pack(side=tk.LEFT, padx=(8, 0))

        motion_controls = ttk.Frame(touch_corner)
        motion_controls.pack(fill=tk.X, pady=(0, 3))
        for label, variable, width in (
            ("max travel mm", self.touch_probe_distance_mm, 6),
            ("speed %", self.touch_probe_speed_percent, 5),
        ):
            ttk.Label(motion_controls, text=label).pack(side=tk.LEFT, padx=(0, 2))
            ttk.Entry(
                motion_controls, textvariable=variable, width=width
            ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(motion_controls, text="settle s").pack(side=tk.LEFT)
        ttk.Spinbox(
            motion_controls,
            from_=0.2,
            to=5.0,
            increment=0.1,
            textvariable=self.touch_settle_seconds,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(motion_controls, text="path points").pack(side=tk.LEFT)
        ttk.Spinbox(
            motion_controls,
            from_=2,
            to=200,
            textvariable=self.corner_touch_count,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 10))

        probe_actions = ttk.Frame(touch_corner)
        probe_actions.pack(fill=tk.X, pady=3)
        for label, kind in (
            ("1 · START wall", "start_wall"),
            ("2 · START base", "start_floor"),
            ("3 · GOAL wall", "goal_wall"),
            ("4 · GOAL base", "goal_floor"),
        ):
            ttk.Button(
                probe_actions,
                text=label,
                command=lambda selected=kind: (
                    self.start_automatic_touch_probe(selected)
                ),
            ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            probe_actions,
            text="Compute START",
            command=lambda: self.compute_seam_endpoint("start"),
        ).pack(side=tk.LEFT, padx=(12, 3))
        ttk.Button(
            probe_actions,
            text="Compute GOAL",
            command=lambda: self.compute_seam_endpoint("goal"),
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            probe_actions,
            text="Compute full seam + orientation",
            command=self.compute_two_touch_seam,
        ).pack(side=tk.LEFT, padx=7)
        self.correct_two_touch_seam_button = ttk.Button(
            probe_actions,
            text="Commit corrected seam YAML",
            command=self.correct_two_touch_seam,
            state=tk.DISABLED,
        )
        self.correct_two_touch_seam_button.pack(side=tk.LEFT, padx=7)

        auto_actions = ttk.Frame(touch_corner)
        auto_actions.pack(fill=tk.X, padx=3, pady=3)
        self.auto_start_correction_button = ttk.Button(
            auto_actions,
            text="AUTO START · wait → wall/base → compute",
            command=self.run_automatic_start_correction,
        )
        self.auto_start_correction_button.pack(side=tk.LEFT, padx=(0, 6))
        self.auto_seam_correction_button = ttk.Button(
            auto_actions,
            text=(
                "AUTO FULL · START wall/base → GOAL wall/base → "
                "yaw-correct + save"
            ),
            command=self.run_automatic_seam_correction,
        )
        self.auto_seam_correction_button.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_auto_seam_button = ttk.Button(
            auto_actions,
            text="STOP AUTO",
            command=self.stop_automatic_seam_correction,
            state=tk.DISABLED,
        )
        self.stop_auto_seam_button.pack(side=tk.LEFT, padx=3)
        ttk.Button(
            auto_actions,
            text="RViz seam",
            command=self.show_computed_seam_in_rviz,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            auto_actions,
            text="RViz touches",
            command=self.show_touch_geometry_in_rviz,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            auto_actions,
            text="PyPlot touches",
            command=self.show_touch_geometry_pyplot,
        ).pack(side=tk.LEFT, padx=3)
        self.corner_touch_status = ttk.Label(
            touch_corner,
            text=(
                "Teach rough START/GOAL first · AUTO wall follows the taught seam normal · "
                "touch geometry corrects XYZ; START→GOAL corrects welding yaw"
            ),
        )
        self.corner_touch_status.pack(anchor=tk.W, pady=(3, 0))

        sequence = self._create_toggle_section(
            outer, "sequence", "Sequence Builder", expanded=False
        )
        sequence_buttons = ttk.Frame(sequence)
        sequence_buttons.pack(fill=tk.X, pady=(0, 4))
        for index, (text, command) in enumerate((
            ("Add motion path", self.add_motion_sequence_step),
            ("Add latest RViz plan", self.add_latest_rviz_plan_step),
            ("Add Go To selected pose", self.add_named_pose_sequence_step),
            ("Add D-WELD ON", lambda: self.add_digital_weld_step("on")),
            ("Add D-WELD OFF", lambda: self.add_digital_weld_step("off")),
            ("Add D-WELD SET", lambda: self.add_digital_weld_step("set")),
            ("Build weld scenario (START enough)", self.build_sensed_weld_sequence),
            ("Add Sleep", self.add_sleep_sequence_step),
            ("Delete", self.delete_sequence_step),
            ("↑", lambda: self.move_sequence_step(-1)),
            ("↓", lambda: self.move_sequence_step(1)),
            ("Apply selected values", self.apply_selected_sequence_values),
            ("Edit selected...", self.open_sequence_step_editor),
            ("Plan selected", lambda: self.run_sequence(False, False)),
            ("Plan all", lambda: self.run_sequence(True, False)),
            ("Execute selected", lambda: self.run_sequence(False, True)),
            ("Execute all", lambda: self.run_sequence(True, True)),
            ("STOP NOW · Robot + Welder", self.stop_sequence),
        )):
            ttk.Button(sequence_buttons, text=text, command=command).grid(
                row=index // 7, column=index % 7, padx=2, pady=2, sticky=tk.W
            )
        sleep_editor = ttk.Frame(sequence)
        sleep_editor.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(sleep_editor, text="Parallel slot").pack(side=tk.LEFT)
        ttk.Spinbox(
            sleep_editor,
            from_=1,
            to=999,
            increment=1,
            textvariable=self.sequence_parallel_slot,
            width=5,
        ).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(sleep_editor, text="Step duration seconds").pack(side=tk.LEFT)
        ttk.Spinbox(
            sleep_editor,
            from_=0.0,
            to=3600.0,
            increment=0.1,
            textvariable=self.sequence_duration_seconds,
            width=8,
        ).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(sleep_editor, text="Sleep seconds").pack(side=tk.LEFT)
        ttk.Spinbox(
            sleep_editor,
            from_=0.0,
            to=3600.0,
            increment=0.1,
            textvariable=self.sequence_sleep_seconds,
            width=8,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(sleep_editor, text="Selected motion speed %").pack(
            side=tk.LEFT, padx=(12, 2)
        )
        ttk.Spinbox(
            sleep_editor,
            from_=1.0,
            to=100.0,
            increment=1.0,
            textvariable=self.sequence_edit_velocity_percent,
            width=6,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            sleep_editor,
            text="DI8 guard",
            variable=self.sequence_edit_touch_guard,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(
            sleep_editor,
            text="continue after DI8 stop",
            variable=self.sequence_edit_continue_after_touch,
        ).pack(side=tk.LEFT, padx=6)
        self.sequence_table = ttk.Treeview(
            sequence,
            columns=("order", "type", "detail"),
            show="headings",
            height=5,
            selectmode="browse",
        )
        for name, width in (("order", 60), ("type", 130), ("detail", 850)):
            self.sequence_table.heading(name, text=name.upper())
            self.sequence_table.column(name, width=width, anchor=tk.W)
        self.sequence_table.pack(fill=tk.X)
        self.sequence_table.bind(
            "<<TreeviewSelect>>", self.load_selected_sequence_values
        )
        self.sequence_table.bind(
            "<Double-1>", self.open_sequence_step_editor
        )
        self.sequence_status = ttk.Label(sequence, text="Sequence idle")
        self.sequence_status.pack(anchor=tk.W, pady=(3, 0))

        planned_path = self._create_toggle_section(
            outer, "planned_path", "Planned Path · World frame"
        )
        columns = ("id",) + self.POSE_FIELDS
        self.table = ttk.Treeview(
            planned_path,
            columns=columns,
            show="headings",
            height=4,
            selectmode="browse",
        )
        for name in columns:
            self.table.heading(name, text=name.upper())
            self.table.column(
                name,
                width=48 if name == "id" else 105,
                anchor=tk.CENTER,
            )
        self.table.pack(fill=tk.X)
        ttk.Button(
            planned_path,
            text="Delete All",
            command=self.clear_path,
        ).pack(anchor=tk.E, pady=(5, 0))

        io_monitor = self._create_toggle_section(
            outer,
            "digital_io",
            "Digital I/O · DI8 = TOUCH · ports 0..15",
        )
        for io_row, kind in enumerate(("DI", "DO")):
            ttk.Label(
                io_monitor,
                text=kind,
                font=("Sans", 10, "bold"),
            ).grid(row=io_row, column=0, padx=(6, 4), pady=3)
            for port in range(16):
                candidate = port in MANUAL_IO_CANDIDATES
                label = tk.Label(
                    io_monitor,
                    text=f"{port:02d}\n–",
                    width=4,
                    relief=tk.SOLID,
                    borderwidth=2 if candidate else 1,
                    bg="#dbeafe" if candidate else "#eeeeee",
                    font=("Monospace", 9, "bold" if candidate else "normal"),
                )
                label.grid(row=io_row, column=port + 1, padx=2, pady=3)
                self.control_box_io_labels[(kind, port)] = label
                if kind == "DO":
                    label.configure(cursor="hand2")
                    label.bind(
                        "<Button-1>",
                        lambda _event, selected=port: (
                            self.request_do_toggle(selected)
                        ),
                    )
        self.control_box_io_status = ttk.Label(
            io_monitor,
            text="Waiting for /right_rbpodo_hardware/system_state",
        )
        self.control_box_io_status.grid(
            row=2,
            column=0,
            columnspan=17,
            sticky=tk.W,
            padx=6,
            pady=(2, 5),
        )
        ttk.Checkbutton(
            io_monitor,
            text="Unlock clicking non-candidate DO ports",
            variable=self.unlock_all_do_ports,
            command=self.confirm_all_do_unlock,
        ).grid(
            row=3,
            column=0,
            columnspan=12,
            sticky=tk.W,
            padx=6,
            pady=(0, 5),
        )
        ttk.Button(
            io_monitor,
            text="Candidate DO all OFF",
            command=self.candidate_outputs_off,
        ).grid(
            row=3,
            column=12,
            columnspan=5,
            sticky=tk.E,
            padx=6,
            pady=(0, 5),
        )

        execution = ttk.Frame(outer)
        execution.pack(fill=tk.X, pady=(12, 0))
        self.plan_button = ttk.Button(
            execution,
            text="1 · Plan Preview",
            command=self.plan_preview,
            state=tk.DISABLED,
        )
        self.plan_button.pack(side=tk.LEFT, padx=(0, 8))
        self.execute_button = ttk.Button(
            execution,
            text="2 · Execute Approved Plan",
            command=self.execute_approved,
            state=tk.DISABLED,
        )
        self.execute_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            execution,
            text="Cancel",
            command=self.cancel,
        ).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(
            execution,
            text="velocity scale",
        ).pack(side=tk.LEFT)
        ttk.Scale(
            execution,
            from_=1,
            to=100,
            variable=self.velocity_percent,
            command=self.update_speed_label,
            length=220,
        ).pack(side=tk.LEFT, padx=(7, 5))
        self.speed_label = ttk.Label(execution, text="20%")
        self.speed_label.pack(side=tk.LEFT)

        planning_settings = ttk.Frame(outer)
        planning_settings.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(planning_settings, text="Cartesian interpolation step mm").pack(
            side=tk.LEFT
        )
        ttk.Spinbox(
            planning_settings,
            from_=0.5,
            to=20.0,
            increment=0.5,
            textvariable=self.interpolation_step_mm,
            width=6,
            command=self.invalidate_approved_plan,
        ).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(
            planning_settings,
            text="trajectory profile: TOTG + Ruckig jerk smoothing",
        ).pack(side=tk.LEFT)

        ttk.Label(
            outer,
            text="Action feedback",
            style="Step.TLabel",
        ).pack(anchor=tk.W, pady=(12, 5))
        self.bar = ttk.Progressbar(outer, maximum=100)
        self.bar.pack(fill=tk.X)
        self.feedback_label = ttk.Label(
            outer,
            text="waypoint: –    pose: –",
        )
        self.feedback_label.pack(anchor=tk.W, pady=4)

        ttk.Label(
            outer,
            text="Pipeline status",
            style="Step.TLabel",
        ).pack(anchor=tk.W, pady=(8, 5))
        self.pipeline_status = tk.Label(
            outer,
            text="WAITING · ready",
            anchor=tk.W,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#eeeeee",
            font=("Sans", 10, "bold"),
        )
        self.pipeline_status.pack(fill=tk.X, ipady=5)

        self.node = WeldGuiNode(self)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
        )
        self.executor_thread.start()
        self._auto_load_teaching_states()
        signal.signal(
            signal.SIGINT,
            lambda _signum, _frame: self.root.after(0, self.close),
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.check_ros)

    def post(self, callback, *args):
        self._ui_queue.put((callback, args))

    def post_latest(self, key, callback, *args):
        """Coalesce high-rate telemetry so Tk only renders the newest value."""
        with self._latest_ui_updates_lock:
            self._latest_ui_updates[key] = (callback, args)

    def _drain_ui_queue(self):
        # Never monopolize Tk's event loop.  ROS callbacks can produce work
        # faster than widgets can render it; an unlimited drain starves mouse,
        # scrolling, repainting, and the Hi-COMM cyclic Python thread.
        deadline = time.monotonic() + 0.004
        processed = 0
        while processed < 64 and time.monotonic() < deadline:
            try:
                callback, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            callback(*args)
            processed += 1
        with self._latest_ui_updates_lock:
            latest = tuple(self._latest_ui_updates.values())
            self._latest_ui_updates.clear()
        for callback, args in latest:
            callback(*args)

    def _update_scroll_region(self, _event=None):
        self.content_canvas.configure(
            scrollregion=self.content_canvas.bbox("all")
        )

    def _resize_scroll_content(self, event):
        self.content_canvas.itemconfigure(
            self.content_window,
            width=event.width,
        )

    def _scroll_content(self, event):
        delta = getattr(event, "delta", 0)
        button = getattr(event, "num", None)
        if delta > 0 or button == 4:
            direction = -1
        elif delta < 0 or button == 5:
            direction = 1
        else:
            return
        self.content_canvas.yview_scroll(direction * 2, "units")

    def _set_welder_test_controls(self, enabled):
        active = bool(enabled and self.hicomm_connected)
        state = tk.NORMAL if active else tk.DISABLED
        for widget in (
            self.hicomm_forward_button,
            self.hicomm_reverse_button,
            self.hicomm_gas_check,
            self.hicomm_arc_set_button,
            self.hicomm_arc_unlock_check,
        ):
            widget.configure(state=state)
        self.hicomm_arc_on_button.configure(
            state=(
                tk.NORMAL
                if active and self.hicomm_arc_unlocked.get()
                else tk.DISABLED
            )
        )

    def request_hicomm_inching(self, direction, active):
        client = self.hicomm_client
        if client is None or not client.connected:
            return
        mask = BIT_FORWARD if direction == "forward" else BIT_REVERSE
        try:
            if active:
                client.allow_outputs()
                opposite = BIT_REVERSE if mask == BIT_FORWARD else BIT_FORWARD
                client.set_command_bit(opposite, False)
                client.set_command_bit(mask, True)
                self.hicomm_inching_direction = direction
            else:
                client.set_command_bit(mask, False)
                if self.hicomm_inching_direction == direction:
                    self.hicomm_inching_direction = None
            state = "ON" if active else "OFF"
            self.hicomm_test_status.configure(
                text=f"{direction} inch {state}"
            )
            self.log(f"Hi-COMM {direction} inch {state}")
        except Exception as error:
            client.clear_outputs()
            self.error(f"Hi-COMM inching failed: {error}")

    def request_hicomm_gas(self):
        enabled = bool(self.hicomm_gas_enabled.get())
        client = self.hicomm_client
        if client is None or not client.connected:
            self.hicomm_gas_enabled.set(False)
            return
        try:
            if enabled:
                client.allow_outputs()
            client.set_command_bit(BIT_GAS, enabled)
            self.hicomm_test_status.configure(
                text=f"gas {'ON' if enabled else 'OFF'}"
            )
            self.log(f"Hi-COMM gas {'ON' if enabled else 'OFF'}")
        except Exception as error:
            client.clear_outputs()
            self.hicomm_gas_enabled.set(False)
            self.error(f"Hi-COMM gas test failed: {error}")

    def clear_hicomm_test_outputs(self):
        client = self.hicomm_client
        if client is not None:
            client.clear_outputs()
        self.hicomm_inching_direction = None
        self.hicomm_gas_enabled.set(False)
        self.hicomm_arc_unlocked.set(False)
        if hasattr(self, "hicomm_arc_on_button"):
            self.hicomm_arc_on_button.configure(state=tk.DISABLED)
        if hasattr(self, "hicomm_test_status"):
            self.hicomm_test_status.configure(text="ALL OUTPUTS OFF")

    def connect_hicomm(self):
        if self.hicomm_client is not None and self.hicomm_client.connected:
            return
        try:
            client = HiCommWelderClient(
                self.hicomm_source_ip.get().strip(),
                self.hicomm_welder_ip.get().strip(),
                int(self.hicomm_port.get()),
                connection_callback=lambda connected, detail: self.post(
                    self.hicomm_connection_changed, connected, detail
                ),
                status_callback=self._hicomm_status_received,
                log_callback=lambda message: self.post(self.log, message),
            )
            self.hicomm_client = client
            self.hicomm_connect_button.configure(state=tk.DISABLED)
            self.hicomm_weld_status.configure(text="CONNECTING… · ARC OFF")
            client.start()
        except (ValueError, OSError, tk.TclError) as error:
            self.hicomm_connect_button.configure(state=tk.NORMAL)
            self.error(f"Hi-COMM connection setup failed: {error}")

    def disconnect_hicomm(self):
        client = self.hicomm_client
        if client is not None:
            self.clear_hicomm_test_outputs()
            threading.Thread(target=client.stop, daemon=True).start()

    def hicomm_connection_changed(self, connected, detail):
        self.hicomm_connected = bool(connected)
        self.rbpodo_welder_ready = self.hicomm_connected
        self.hicomm_feedback_last_log_time = 0.0
        self.hicomm_feedback_last_signature = None
        retrying = not connected and detail.startswith("retrying in")
        self.hicomm_connect_button.configure(
            state=tk.DISABLED if connected or retrying else tk.NORMAL
        )
        self.hicomm_disconnect_button.configure(
            state=tk.NORMAL if connected or retrying else tk.DISABLED
        )
        if not connected:
            self.clear_hicomm_test_outputs()
        self._set_welder_test_controls(
            connected
        )
        self.welder_connection_label.configure(
            text=f"HICOMM WELDER: {'O' if connected else 'X'}",
            bg="#e6f4ea" if connected else "#fce8e6",
            fg="#137333" if connected else "#b3261e",
        )
        self.hicomm_weld_status.configure(
            text=(
                "CONNECTED"
                if connected
                else ("RETRYING / 200 ms" if retrying else "DISCONNECTED")
            )
            + " · ARC OFF"
        )
        if not connected:
            self.hicomm_rx_bit_status.configure(
                text=(
                    "RX Byte0 · b5 WCR=? · b4 STICK=? · "
                    "b3 GAS CHECK=? · b0 TORCH=?"
                ),
                foreground="#5f6368",
            )
        if not retrying:
            self.log(
                f"Hi-COMM {'connected' if connected else 'disconnected'} · "
                f"{detail}"
            )

    def hicomm_status_changed(self, status):
        arc_on = bool(status["arc_ack"])
        arc_established = bool(status.get("arc_established"))
        error_code = int(status["welder_error"])
        self.hicomm_weld_status.configure(
            text=(
                f"ARC={'ESTABLISHED' if arc_established else ('ON' if arc_on else 'OFF')} · "
                f"{status.get('sequence_stage', 'unknown')} · "
                f"FB {status['feedback_current_a']}A/"
                f"{status['feedback_voltage_v']:.1f}V · ERR={error_code}"
            )
        )
        self.hicomm_rx_bit_status.configure(
            text=(
                "RX Byte0 · "
                f"b5 WCR={int(bool(status['wcr_detected']))} · "
                f"b4 STICK={int(bool(status['stick_ack']))} · "
                f"b3 GAS CHECK={int(bool(status['gas_ack']))} · "
                f"b0 TORCH={int(bool(status['arc_ack']))}"
            ),
            foreground=(
                "#b3261e"
                if status["torch_collision"] or error_code
                else "#137333"
            ),
        )
        acknowledgements = []
        for key, name in (
            ("wcr_detected", "WCR"),
            ("stick_ack", "STICK"),
            ("forward_ack", "FWD"),
            ("reverse_ack", "REV"),
            ("gas_ack", "GAS"),
            ("arc_ack", "ARC"),
        ):
            if status[key]:
                acknowledgements.append(name)
        if self.hicomm_connected:
            total_mm, forward_mm, reverse_mm = self._inching_distance_snapshot()
            self.hicomm_test_status.configure(
                text=(
                    "RX ACK="
                    + (",".join(acknowledgements) if acknowledgements else "OFF")
                    + f" · WFS={status['wire_feed_m_min']:.1f} m/min"
                    + f" · inch={total_mm:+.1f} mm "
                    + f"(F {forward_mm:.1f}/R {reverse_mm:.1f})"
                    + f" · ERR={error_code}"
                )
            )

    def _hicomm_status_received(self, status):
        """Integrate RX wire-feed speed before forwarding status to Tk."""
        timestamp = float(status.get("timestamp_monotonic", time.monotonic()))
        with self.inching_distance_lock:
            previous = self.inching_last_status_time
            self.inching_last_status_time = timestamp
            if previous is not None:
                dt = max(0.0, min(0.2, timestamp - previous))
                distance_mm = (
                    max(0.0, float(status.get("wire_feed_m_min", 0.0)))
                    * 1000.0 / 60.0 * dt
                )
                if status.get("forward_ack"):
                    self.inching_forward_mm += distance_mm
                    self.inching_total_mm += distance_mm
                elif status.get("reverse_ack"):
                    self.inching_reverse_mm += distance_mm
                    self.inching_total_mm -= distance_mm
        self._log_hicomm_feedback(status, timestamp)
        self.post_latest(
            "hicomm_status", self.hicomm_status_changed, status
        )

    def _log_hicomm_feedback(self, status, timestamp):
        """Continuously expose TX commands and decoded welder RX in ROS logs."""
        client = self.hicomm_client
        if client is None:
            return
        try:
            tx = client.snapshot()
        except Exception:
            return
        command = int(tx.command)
        signature = (
            command,
            tx.base_profile,
            int(status.get("raw0", 0)),
            int(status.get("output_state", -1)),
            int(status.get("welder_error", 0)),
            bool(status.get("db_unavailable")),
            bool(status.get("torch_collision")),
        )
        state_changed = signature != self.hicomm_feedback_last_signature
        active_feedback = bool(
            command
            or int(status.get("raw0", 0))
            or int(status.get("output_state", 0))
            or int(status.get("welder_error", 0))
            or status.get("db_unavailable")
            or status.get("torch_collision")
        )
        log_period = (
            self.hicomm_feedback_log_period_s
            if active_feedback
            else self.hicomm_feedback_idle_log_period_s
        )
        if (
            signature == self.hicomm_feedback_last_signature
            and timestamp - self.hicomm_feedback_last_log_time
            < log_period
        ):
            return
        self.hicomm_feedback_last_signature = signature
        self.hicomm_feedback_last_log_time = timestamp

        if state_changed:
            tx_raw = build_request(tx)
            rx_raw = status.get("raw_frame", b"")
            self.node.get_logger().info(
                f"HICOMM TX RAW [{len(tx_raw)}B] · "
                f"{tx_raw.hex(' ').upper()}"
            )
            if rx_raw:
                self.node.get_logger().info(
                    f"HICOMM RX RAW [{len(rx_raw)}B] · "
                    f"{bytes(rx_raw).hex(' ').upper()}"
                )

        def bit(value, mask):
            return int(bool(value & mask))

        self.node.get_logger().info(
            "HICOMM FEEDBACK · "
            f"PROFILE={tx.base_profile} · TX=0x{command:02X} "
            f"ARC={bit(command, BIT_ARC)} GAS={bit(command, BIT_GAS)} "
            f"FWD={bit(command, BIT_FORWARD)} REV={bit(command, BIT_REVERSE)} "
            f"STICK={bit(command, BIT_STICK)} · "
            f"RX=0x{int(status.get('raw0', 0)):02X} "
            f"ARC={int(bool(status.get('arc_ack')))} "
            f"GAS={int(bool(status.get('gas_ack')))} "
            f"FWD={int(bool(status.get('forward_ack')))} "
            f"REV={int(bool(status.get('reverse_ack')))} "
            f"WCR={int(bool(status.get('wcr_detected')))} "
            f"STICK={int(bool(status.get('stick_ack')))} · "
            f"OUT={status.get('output_state_name', 'unknown')}"
            f"({int(status.get('output_state', -1))}) · "
            f"FB={int(status.get('feedback_current_a', 0))}A/"
            f"{float(status.get('feedback_voltage_v', 0.0)):.1f}V "
            f"WFS={float(status.get('wire_feed_m_min', 0.0)):.1f}m/min · "
            f"SET={int(status.get('set_current_a', 0))}A/"
            f"{float(status.get('set_voltage_v', 0.0)):.1f}V · "
            f"DB={int(bool(status.get('db_unavailable')))} "
            f"COLL={int(bool(status.get('torch_collision')))} "
            f"ERR={int(status.get('welder_error', 0))}"
        )

    def _inching_distance_snapshot(self):
        with self.inching_distance_lock:
            return (
                self.inching_total_mm,
                self.inching_forward_mm,
                self.inching_reverse_mm,
            )

    def reset_inching_distance(self):
        with self.inching_distance_lock:
            self.inching_total_mm = 0.0
            self.inching_forward_mm = 0.0
            self.inching_reverse_mm = 0.0
            self.inching_last_status_time = None
        self.log("Hi-COMM estimated inching length reset to 0 mm")

    def hicomm_arc_unlock_changed(self):
        unlocked = self.hicomm_arc_unlocked.get()
        if unlocked and (
            self.planning_group.get() != "right_manipulator"
            or not self.hicomm_connected
        ):
            self.hicomm_arc_unlocked.set(False)
            self.error(
                "Select the right arm and connect Hi-COMM first"
            )
            return
        if unlocked and not messagebox.askyesno(
            "Unlock digital ARC",
            "This permits a physical ARC ON command through Hi-COMM.\n\n"
            "Confirm the cell is safe and the torch is ready.",
        ):
            self.hicomm_arc_unlocked.set(False)
            unlocked = False
        if not unlocked and self.hicomm_client is not None:
            self.hicomm_client.set_arc(False)
        self.hicomm_arc_on_button.configure(
            state=(
                tk.NORMAL
                if unlocked and self.hicomm_connected
                else tk.DISABLED
            )
        )

    def request_digital_weld_set(self):
        if not self.hicomm_connected or self.hicomm_client is None:
            self.error("Connect Hi-COMM first")
            return
        try:
            settings = self._digital_weld_settings()
        except ValueError as error:
            self.error(str(error))
            return
        self.hicomm_client.arc_set(**digital_weld_recipe(settings))
        self.log(
            f"Hi-COMM SET applied · {settings['current_a']} A / "
            f"{settings['voltage']:.1f} V"
        )

    def request_digital_arc(self, enabled):
        if enabled and not self.hicomm_arc_unlocked.get():
            self.error("Unlock ARC ON first")
            return
        if self.hicomm_client is None or not self.hicomm_connected:
            self.error("Connect Hi-COMM first")
            return
        try:
            settings = self._digital_weld_settings()
        except ValueError as error:
            self.error(str(error))
            return
        if enabled:
            self.hicomm_client.allow_outputs()
        threading.Thread(
            target=self._manual_digital_weld_worker,
            args=(enabled, settings),
            daemon=True,
        ).start()

    def _manual_digital_weld_worker(self, enabled, settings):
        kind = "on" if enabled else "off"
        success, message = self._execute_hicomm_weld(kind, settings)
        self.post(
            self.log,
            f"Hi-COMM ARC {kind.upper()} · "
            f"{'OK' if success else 'FAILED'} · {message}",
        )

    def _digital_weld_settings(self):
        try:
            return validate_digital_weld_settings({
                "current_a": self.weld_current_raw.get(),
                "voltage_tenths": self.weld_voltage_raw.get(),
                "material": self.weld_material.get(),
                "diameter_mm": self.weld_diameter_mm.get(),
                "mode": self.weld_mode.get(),
                "gas": self.weld_gas.get(),
                "synergic": self.weld_synergic.get(),
                "correction": self.weld_correction.get(),
                "pre_gas_s": self.weld_pre_gas_s.get(),
                "post_gas_s": self.weld_post_gas_s.get(),
                "preflow_seconds": self.weld_preflow_seconds.get(),
            })
        except (ValueError, tk.TclError) as error:
            raise ValueError(
                f"digital weld settings are invalid: {error}"
            ) from error

    def _execute_hicomm_weld(self, kind, settings):
        client = self.hicomm_client
        if client is None or not client.connected:
            return False, "Hi-COMM disconnected"
        if kind not in DIGITAL_WELD_COMMANDS:
            return False, f"unsupported D-WELD command: {kind}"
        try:
            if kind == "set":
                client.arc_set(**digital_weld_recipe(settings))
                echo = client.setting_echo()
                return True, f"recipe applied · RX echo={echo}"
            elif kind == "on":
                status = client.arc_on(
                    wait_recognition=True,
                    wait_welding=True,
                    wait_established=True,
                    timeout=5.0,
                )
                return True, (
                    "ARC established (main_weld + WCR + feed) · "
                    f"output={status['output_state_name']} · "
                    f"feedback={status['feedback_current_a']} A/"
                    f"{status['feedback_voltage_v']:.1f} V · "
                    f"WFS={status['wire_feed_m_min']:.1f} m/min"
                )
            status = client.arc_off(
                timeout=max(
                    5.0,
                    (
                        settings.get("post_gas_s", 0.0)
                        if isinstance(settings, dict)
                        else 0.0
                    ) + 2.0,
                ),
                wait_idle=True,
                wait_sequence_clear=True,
            )
            return True, (
                "ARC OFF sequence clear · "
                f"output={status['output_state_name']} · "
                f"stage={status.get('sequence_stage', 'unknown')}"
            )
        except Exception as error:
            # Match v5.2: an ARC feedback timeout/error does not itself alter
            # the already transmitted ARC command.  Only explicit D-WELD OFF,
            # STOP, disconnect, or the sequence failure safety cleanup may
            # clear outputs.
            if kind != "on":
                client.set_arc(False)
            return False, str(error)

    def capture_corner_touch_now(self):
        target = self.corner_touch_target.get()
        self.node.capture_touch_pose(
            self.planning_group.get(), f"manual corner capture:{target}"
        )

    def select_weld_wait_pose(self):
        self.teaching_pose_name.set(TEACHING_POSES["weld_wait"])
        self.teaching_pose_changed()
        if self.initial_joint_state is None:
            self.error("Capture or load Weld wait pose first")
            return
        self.plan_initial_state()

    def run_automatic_seam_correction(self):
        self._run_automatic_seam_correction(start_only=False)

    def run_automatic_start_correction(self):
        self._run_automatic_seam_correction(start_only=True)

    def _run_automatic_seam_correction(self, start_only):
        if self.seam_auto_running:
            self.error("Automatic seam correction is already running")
            return
        if not self.execution_allowed or not self.robot_connected["right"]:
            self.error("Connect the right robot and enable physical execution")
            return
        if self.planning_group.get() != "right_manipulator":
            self.error("Automatic seam correction currently supports right arm")
            return
        required = (
            ("weld_wait", "weld_start_wait", "weld_start")
            if start_only
            else (
                "weld_start_wait",
                "weld_start",
                "weld_goal_wait",
                "weld_end",
            )
        )
        missing = [
            TEACHING_POSES[name]
            for name in required
            if self.taught_robot_poses[name] is None
        ]
        if missing:
            self.error("Capture/load required poses: " + ", ".join(missing))
            return
        wrong_group = [
            TEACHING_POSES[name]
            for name in required
            if self.taught_robot_poses[name][0] != "right_manipulator"
        ]
        if wrong_group:
            self.error(
                "These poses belong to another arm: "
                + ", ".join(wrong_group)
            )
            return
        if self.touch_input_states["right"] is None:
            self.error("DI8 state has not been received yet")
            return
        if self.touch_input_states["right"]:
            self.error("DI8 is already ON; release it before auto correction")
            return
        title = (
            "Automatic START Seam Correction"
            if start_only
            else "Automatic Seam Correction"
        )
        description = (
            "Execute START wall/floor touch probes automatically?\n\n"
            "Weld wait → wall touch/return → floor touch/return\n"
            "The START seam pose will be saved; both wait poses remain "
            "unchanged."
            if start_only
            else
            "Execute four physical touch probes automatically?\n\n"
            "START wait → wall/floor → GOAL wait → wall/floor\n"
            "Each DI8 edge performs controlled move_stop and return.\n"
            "The corrected Weld start/goal TCP YAML files will be updated."
        )
        if not messagebox.askyesno(title, description):
            return
        wait_steps = {}
        wait_names = (
            ("weld_wait",)
            if start_only
            else ("weld_start_wait", "weld_goal_wait")
        )
        for wait_name in wait_names:
            group, names, positions, tcp = self.taught_robot_poses[wait_name]
            wait_steps[wait_name] = {
                "type": "named_pose",
                "pose_name": wait_name,
                "pose_label": TEACHING_POSES[wait_name],
                "planning_group": group,
                "joint_names": tuple(names),
                "positions": tuple(positions),
                "tcp_pose": copy.deepcopy(tcp),
                "velocity_scale": max(
                    0.01,
                    min(1.0, self.velocity_percent.get() / 100.0),
                ),
                # These are already taught joint targets.  Keep automatic seam
                # correction responsive instead of allowing 5 s × 5 attempts.
                "planning_attempts": 1,
                "planning_time": 1.0,
            }
        workflow = [(
            wait_steps["weld_wait" if start_only else "weld_start_wait"],
            ("start_wall", "start_floor"),
        )]
        if not start_only:
            workflow.append((
                wait_steps["weld_goal_wait"],
                ("goal_wall", "goal_floor"),
            ))
        for name in (
            ("start_wall", "start_floor")
            if start_only else CORNER_TOUCH_NAMES
        ):
            self.seam_probe_touches[name] = None
            self.seam_probe_starts[name] = None
            self.seam_probe_stops[name] = None
        self.raw_two_touch_seam = []
        self.corrected_two_touch_seam = []
        self.seam_auto_running = True
        self.seam_auto_returned_kinds.clear()
        self.auto_start_correction_button.configure(state=tk.DISABLED)
        self.auto_seam_correction_button.configure(state=tk.DISABLED)
        self.stop_auto_seam_button.configure(state=tk.NORMAL)
        threading.Thread(
            target=self._automatic_seam_correction_worker,
            args=(tuple(workflow), start_only),
            daemon=True,
        ).start()

    def stop_automatic_seam_correction(self):
        if not self.seam_auto_running and self.automatic_probe_kind is None:
            self.log("STOP AUTO ignored · automatic seam correction is idle")
            return
        self.seam_auto_running = False
        self.seam_auto_expected_kind = None
        self.seam_auto_stage_success = False
        self.seam_auto_stage_event.set()
        self.automatic_probe_kind = None
        self.stop_auto_seam_button.configure(state=tk.DISABLED)
        self.corner_touch_status.configure(
            text="STOP AUTO requested · stopping robot motion"
        )
        threading.Thread(
            target=self.node.stop_auto_motion,
            args=("right",),
            daemon=True,
        ).start()

    def auto_seam_stop_finished(self, success, message):
        self.auto_start_correction_button.configure(state=tk.NORMAL)
        self.auto_seam_correction_button.configure(state=tk.NORMAL)
        self.stop_auto_seam_button.configure(state=tk.DISABLED)
        if success:
            self.pipeline_result("AUTO SEAM STOPPED · robot stationary")
        else:
            self.error(f"STOP AUTO could not confirm safe idle: {message}")

    def show_computed_seam_in_rviz(self):
        if not self.raw_two_touch_seam or not self.corrected_two_touch_seam:
            self.error("Compute the four-touch seam first")
            return
        self.show_path.set(True)
        self.node.publish_seam_comparison(
            self.raw_two_touch_seam,
            self.corrected_two_touch_seam,
            True,
        )
        for endpoint in ("start", "goal"):
            self._publish_touch_geometry_if_ready(
                endpoint, self.computed_seam_endpoints.get(endpoint)
            )
        self.log(
            "Published computed seam to RViz /weld_path_markers · "
            "red=raw, translucent cyan=corrected"
        )

    def _publish_touch_geometry_if_ready(self, endpoint, seam_point=None):
        wall = self.seam_probe_touches.get(f"{endpoint}_wall")
        floor = self.seam_probe_touches.get(f"{endpoint}_floor")
        if wall is None or floor is None:
            return False
        try:
            if seam_point is None:
                teaching_reference = self._ensure_seam_teaching_reference(
                    require_complete=True
                )
                if teaching_reference is None:
                    return False
                pose_name = "weld_start" if endpoint == "start" else "weld_end"
                (
                    _reference,
                    wall_normal,
                    floor_normal,
                    _wall_label,
                    _floor_label,
                ) = self._seam_geometry_settings(require_teaching=True)
                seam_point = generalized_corner_endpoint_from_two_touches(
                    wall,
                    floor,
                    teaching_reference[pose_name][3],
                    teaching_reference["weld_start"][3],
                    teaching_reference["weld_end"][3],
                    wall_normal,
                    floor_normal,
                    float(self.seam_wall_offset_mm.get()) * 0.001,
                    float(self.seam_floor_offset_mm.get()) * 0.001,
                )
            self.node.publish_touch_geometry(
                endpoint, wall, floor, seam_point
            )
        except (ValueError, tk.TclError) as error:
            self.error(f"{endpoint.upper()} touch visualization failed: {error}")
            return False
        return True

    def show_touch_geometry_in_rviz(self):
        published = [
            endpoint
            for endpoint in ("start", "goal")
            if self._publish_touch_geometry_if_ready(
                endpoint, self.computed_seam_endpoints.get(endpoint)
            )
        ]
        if not published:
            self.error("Capture a complete wall/floor touch pair first")
            return
        self.show_path.set(True)
        self.log(
            "Published DI8 touch geometry to RViz /weld_path_markers · "
            f"{', '.join(name.upper() for name in published)} · "
            "red=wall, blue=floor, green=seam, yellow=midpoint"
        )

    def show_touch_geometry_pyplot(self, endpoint=None):
        touch_yaml = self._seam_touch_yaml_path("right_manipulator")
        if not touch_yaml.is_file():
            self.error("No saved DI8 touch YAML exists yet")
            return False
        plot_python = Path.home() / "ros2_ws" / ".venv" / "bin" / "python"
        if not plot_python.is_file():
            self.error(f"PyPlot Python is unavailable: {plot_python}")
            return False
        plot_script = Path(__file__).with_name("seam_touch_plot.py")
        try:
            command = [
                str(plot_python),
                str(plot_script),
                str(touch_yaml),
                "--wall-offset-mm",
                str(float(self.seam_wall_offset_mm.get())),
                "--floor-offset-mm",
                str(float(self.seam_floor_offset_mm.get())),
            ]
            if endpoint in ("start", "goal"):
                command.extend(("--endpoint", endpoint))
            subprocess.Popen(command, start_new_session=True)
        except (OSError, ValueError, tk.TclError) as error:
            self.error(f"Cannot open touch-result PyPlot: {error}")
            return False
        self.log(
            "Opened DI8 touch-result PyPlot · "
            + (endpoint.upper() if endpoint else "START + GOAL")
        )
        return True

    def _automatic_seam_correction_worker(self, workflow, start_only=False):
        probe_index = 0
        group_total = len(workflow)
        probe_total = sum(len(kinds) for _step, kinds in workflow)
        for group_index, (wait_step, probe_kinds) in enumerate(
            workflow, start=1
        ):
            if not self.seam_auto_running:
                return
            self.post(
                self._set_auto_seam_status,
                f"AUTO GROUP {group_index}/{group_total} · moving to "
                f"{wait_step['pose_label']}",
            )
            success, message = False, "not attempted"
            for attempt in (1,):
                success, message = self.node.run_sequence_named_pose(
                    wait_step, True
                )
                if success or not self.seam_auto_running:
                    break
                self.post(
                    self.log,
                    f"AUTO {wait_step['pose_label']} attempt {attempt} "
                    f"failed · {message}",
                )
                time.sleep(0.5)
            if not self.seam_auto_running:
                return
            if not success:
                self.post(
                    self._finish_automatic_seam_correction,
                    False,
                    f"{wait_step['pose_label']} failed: {message}",
                )
                return
            self.post(
                self.log,
                f"AUTO reached {wait_step['pose_label']} · "
                f"starting {probe_kinds[0]} then {probe_kinds[1]}",
            )
            for kind in probe_kinds:
                if not self.seam_auto_running:
                    return
                probe_index += 1
                self.seam_auto_expected_kind = kind
                self.seam_auto_stage_success = False
                self.seam_auto_stage_event.clear()
                self.post(
                    self._set_auto_seam_status,
                    f"AUTO PROBE {probe_index}/{probe_total} · {kind}",
                )
                self.post(self._launch_automatic_seam_stage, kind)
                if not self.seam_auto_stage_event.wait(timeout=180.0):
                    self.post(
                        self._finish_automatic_seam_correction,
                        False,
                        f"{kind} timed out",
                    )
                    return
                if not self.seam_auto_running:
                    return
                touch_saved = self.seam_probe_touches.get(kind) is not None
                returned = kind in self.seam_auto_returned_kinds
                if not (
                    self.seam_auto_stage_success
                    and touch_saved
                    and returned
                ):
                    self.post(
                        self._finish_automatic_seam_correction,
                        False,
                        f"{kind} incomplete: touch_saved={touch_saved}, "
                        f"returned={returned}",
                    )
                    return
                self.post(
                    self.log,
                    f"AUTO CHECKPOINT {probe_index}/{probe_total} · {kind} · "
                    f"touch_saved={touch_saved} · returned={returned}",
                )
                self.post(
                    self._set_auto_seam_status,
                    f"AUTO {kind} returned · waiting for DI8 OFF",
                )
                if not self._wait_for_di8_release(timeout=180.0):
                    self.post(
                        self._finish_automatic_seam_correction,
                        False,
                        f"{kind} completed, but DI8 remained ON",
                    )
                    return
            if not start_only and group_index == 1:
                self.post(
                    self._set_auto_seam_status,
                    "START wall/floor complete · next: moving to "
                    "5 · Weld goal wait pose",
                )
        self.post(
            self._complete_automatic_start_correction
            if start_only
            else self._complete_automatic_seam_correction
        )

    def _complete_automatic_start_correction(self):
        point = self.compute_seam_endpoint("start")
        if point is None:
            self._finish_automatic_seam_correction(
                False, "START two-touch computation failed"
            )
            return
        self.seam_auto_running = False
        self.seam_auto_expected_kind = None
        self.seam_auto_stage_event.set()
        self.auto_start_correction_button.configure(state=tk.NORMAL)
        self.auto_seam_correction_button.configure(state=tk.NORMAL)
        self.stop_auto_seam_button.configure(state=tk.DISABLED)

    def _wait_for_di8_release(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.touch_input_states.get("right", True):
                return True
            time.sleep(0.02)
        return False

    def _launch_automatic_seam_stage(self, kind):
        self.start_automatic_touch_probe(kind, skip_confirmation=True)
        if self.automatic_probe_kind != kind:
            self._signal_auto_seam_stage(False, kind)

    def _signal_auto_seam_stage(self, success, kind=None):
        if not self.seam_auto_running:
            return
        if kind is not None and kind != self.seam_auto_expected_kind:
            self.log(
                f"Ignored stale auto probe result for {kind}; "
                f"waiting for {self.seam_auto_expected_kind}"
            )
            return
        self.seam_auto_stage_success = bool(success)
        self.seam_auto_stage_event.set()

    def _set_auto_seam_status(self, text):
        self.corner_touch_status.configure(text=text)
        self.pipeline_waiting(text)

    def _complete_automatic_seam_correction(self):
        self.compute_two_touch_seam()
        if not self.corrected_two_touch_seam:
            self._finish_automatic_seam_correction(
                False, "four-touch seam computation failed"
            )
            return
        success = self.path_kind == "di8_four_touch_corrected"
        self._finish_automatic_seam_correction(
            success,
            (
                "four touches complete; corrected path and YAML adopted"
                if success
                else "corrected seam could not be adopted"
            ),
        )

    def _finish_automatic_seam_correction(self, success, message):
        self.seam_auto_running = False
        self.seam_auto_expected_kind = None
        self.seam_auto_stage_event.set()
        self.auto_start_correction_button.configure(state=tk.NORMAL)
        self.auto_seam_correction_button.configure(state=tk.NORMAL)
        self.stop_auto_seam_button.configure(state=tk.DISABLED)
        if success:
            self.pipeline_result(f"AUTO SEAM CORRECTION COMPLETE · {message}")
        else:
            self.error(f"Automatic seam correction stopped: {message}")

    def _resolve_probe_direction(self, surface, teaching_reference=None):
        """Resolve a configured positive World probe normal and display label."""
        surface = str(surface).strip().lower()
        if surface == "wall":
            selection = self.wall_probe_axis.get().strip()
            if selection.upper().startswith("AUTO"):
                if teaching_reference is None:
                    teaching_reference = self._ensure_seam_teaching_reference(
                        require_complete=True
                    )
                if teaching_reference is None:
                    raise ValueError("START/GOAL teaching is required for AUTO wall normal")
                direction = seam_xy_normal(
                    teaching_reference["weld_start"][3],
                    teaching_reference["weld_end"][3],
                )
                return direction, (
                    "AUTO seam-normal "
                    f"({direction[0]:+.3f}, {direction[1]:+.3f}, {direction[2]:+.3f})"
                )
            direction = _axis_unit_vector(selection)
            return direction, selection
        if surface == "floor":
            selection = self.floor_probe_axis.get().strip()
            direction = _axis_unit_vector(selection)
            return direction, selection
        raise ValueError(f"unknown probe surface: {surface}")

    def _seam_geometry_settings(self, require_teaching=True):
        teaching_reference = self._ensure_seam_teaching_reference(
            require_complete=require_teaching
        )
        if require_teaching and teaching_reference is None:
            raise ValueError("complete seam teaching reference is unavailable")
        wall_normal, wall_label = self._resolve_probe_direction(
            "wall", teaching_reference
        )
        floor_normal, floor_label = self._resolve_probe_direction(
            "floor", teaching_reference
        )
        return teaching_reference, wall_normal, floor_normal, wall_label, floor_label

    def start_automatic_touch_probe(self, kind, skip_confirmation=False):
        if kind not in CORNER_TOUCH_NAMES:
            self.error(f"Unknown touch probe kind: {kind}")
            return
        if self.automatic_probe_kind is not None:
            self.error("Another DI8 touch probe is already active")
            return
        if self.planning_group.get() != "right_manipulator":
            self.error("Automatic DI8 seam probing currently supports the right arm")
            return
        if not self.execution_allowed or not self.robot_connected["right"]:
            self.error("Connect the right robot and enable physical execution")
            return
        if self.touch_input_states["right"] is None:
            self.error("DI8 state has not been received yet")
            return
        if self.touch_input_states["right"]:
            self.error("DI8 is already ON; release the touch signal before probing")
            return
        try:
            distance = float(self.touch_probe_distance_mm.get()) * 0.001
            speed = float(self.touch_probe_speed_percent.get()) / 100.0
            settle = float(self.touch_settle_seconds.get())
        except (ValueError, tk.TclError):
            self.error("Touch probe distance, speed, or settle time is invalid")
            return
        if not 0.001 <= distance <= 0.200:
            self.error("Touch probe max travel must be in 1..200 mm")
            return
        if not 0.001 <= speed <= 0.10:
            self.error("Touch probe speed must be in 0.1..10%")
            return
        if not 0.2 <= settle <= 5.0:
            self.error("Touch settle time must be in 0.2..5.0 seconds")
            return
        surface = kind.rsplit("_", 1)[1]
        try:
            teaching_reference = self._ensure_seam_teaching_reference(
                require_complete=True
            )
            if teaching_reference is None:
                return
            direction, direction_label = self._resolve_probe_direction(
                surface, teaching_reference
            )
        except ValueError as error:
            self.error(f"Cannot resolve touch probe direction: {error}")
            return
        sign = (
            self.wall_probe_sign.get()
            if surface == "wall"
            else self.floor_probe_sign.get()
        )
        sign_scale = 1.0 if sign == "+" else -1.0
        signed_direction = tuple(sign_scale * value for value in direction)
        if not skip_confirmation:
            if not messagebox.askyesno(
                "Execute DI8 touch probe",
                f"{kind.replace('_', ' ').upper()}\n"
                f"Direction: {direction_label} · sign {sign}\n"
                f"World vector=({signed_direction[0]:+.3f}, "
                f"{signed_direction[1]:+.3f}, {signed_direction[2]:+.3f})\n"
                f"Travel up to {distance * 1000.0:.1f} mm at {speed:.1%}?\n\n"
                "DI8 will stop the motion and return to the current "
                "start pose.",
            ):
                return
        if self.hicomm_client is not None:
            self.hicomm_client.set_arc(False)
        self.automatic_probe_kind = kind
        self.corner_touch_status.configure(
            text=(
                f"PROBING {kind.upper()} · {direction_label} {sign} · "
                f"v=({signed_direction[0]:+.2f}, {signed_direction[1]:+.2f}, "
                f"{signed_direction[2]:+.2f}) · waiting for DI8"
            )
        )
        threading.Thread(
            target=self.node.execute_touch_probe,
            args=(
                self.planning_group.get(),
                kind,
                signed_direction,
                distance,
                speed,
                0.001,
            ),
            daemon=True,
        ).start()

    def touch_probe_failed(self, message):
        kind = self.automatic_probe_kind
        self.automatic_probe_kind = None
        self.node.clear_touch_probe()
        self._signal_auto_seam_stage(False, kind)
        self.error(f"{kind or 'touch'} probe failed: {message}")

    def _ensure_seam_teaching_reference(self, require_complete=False):
        """Return immutable TCP1/TCP2 seam references used for geometry/yaw."""
        names = ("weld_start", "weld_end")
        if self.seam_teaching_reference is None:
            self.seam_teaching_reference = {}
        for name in names:
            if (
                name not in self.seam_teaching_reference
                and self.taught_robot_poses.get(name) is not None
            ):
                self.seam_teaching_reference[name] = copy.deepcopy(
                    self.taught_robot_poses[name]
                )
        missing = [
            TEACHING_POSES[name]
            for name in names
            if name not in self.seam_teaching_reference
        ]
        if missing and require_complete:
            self.error(
                "Capture/load seam teaching references first: "
                + ", ".join(missing)
            )
            return None
        if not missing:
            self.log(
                "Seam reference ready for yaw correction · TCP1 START / TCP2 GOAL"
            )
        return self.seam_teaching_reference

    def compute_seam_endpoint(self, endpoint, update_wait_joints=True):
        """Compute START or GOAL independently from its two DI8 poses."""
        teaching_reference = self._ensure_seam_teaching_reference(
            require_complete=True
        )
        if teaching_reference is None:
            return None
        endpoint = str(endpoint).strip().lower()
        if endpoint not in ("start", "goal"):
            self.error(f"Unknown seam endpoint: {endpoint}")
            return None
        wall = self.seam_probe_touches.get(f"{endpoint}_wall")
        floor = self.seam_probe_touches.get(f"{endpoint}_floor")
        missing = [
            name
            for name, pose in (("wall", wall), ("floor", floor))
            if pose is None
        ]
        if missing:
            self.error(
                f"Complete {endpoint.upper()} two-pose sensing first: "
                + ", ".join(missing)
            )
            return None
        pose_name = "weld_start" if endpoint == "start" else "weld_end"
        wait_name = (
            "weld_start_wait" if endpoint == "start" else "weld_goal_wait"
        )
        endpoint_data = self.taught_robot_poses.get(pose_name)
        wait_data = self.taught_robot_poses.get(wait_name)
        if endpoint_data is None or wait_data is None:
            self.error(
                f"Capture/load {TEACHING_POSES[pose_name]} and "
                f"{TEACHING_POSES[wait_name]} first"
            )
            return None
        if endpoint_data[0] != wait_data[0]:
            self.error(
                f"{TEACHING_POSES[pose_name]} and "
                f"{TEACHING_POSES[wait_name]} belong to different arms"
            )
            return None
        try:
            wall_offset = float(self.seam_wall_offset_mm.get()) * 0.001
            floor_offset = float(self.seam_floor_offset_mm.get()) * 0.001
            (
                _reference,
                wall_normal,
                floor_normal,
                wall_label,
                floor_label,
            ) = self._seam_geometry_settings(require_teaching=True)
            point = generalized_corner_endpoint_from_two_touches(
                wall,
                floor,
                teaching_reference[pose_name][3],
                teaching_reference["weld_start"][3],
                teaching_reference["weld_end"][3],
                wall_normal,
                floor_normal,
                wall_offset,
                floor_offset,
            )
            wait_point = copy.deepcopy(wait_data[3])
        except (ValueError, tk.TclError) as error:
            self.error(f"{endpoint.upper()} two-pose computation failed: {error}")
            return None
        updates = [(pose_name, endpoint_data, point)]
        try:
            saved_paths = []
            for teaching_name, stored, corrected_tcp in updates:
                planning_group, joint_names, positions, _old_tcp = stored
                yaml_path = self._initial_state_yaml_path(
                    planning_group, teaching_name
                )
                save_initial_state_yaml(
                    yaml_path,
                    planning_group,
                    joint_names,
                    positions,
                    corrected_tcp,
                )
                saved_paths.append(yaml_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.error(
                f"{endpoint.upper()} computed, but teaching YAML update "
                f"failed: {error}"
            )
            return None
        self.log(
            f"{endpoint.upper()} TCP YAML WRITE VERIFIED · "
            + " · ".join(str(path) for path in saved_paths)
        )
        for teaching_name, stored, corrected_tcp in updates:
            planning_group, joint_names, positions, _old_tcp = stored
            self.taught_robot_poses[teaching_name] = (
                planning_group,
                joint_names,
                positions,
                copy.deepcopy(corrected_tcp),
            )
        self.computed_seam_endpoints[endpoint] = copy.deepcopy(point)
        self.computed_seam_wait_points[endpoint] = copy.deepcopy(wait_point)
        self._publish_touch_geometry_if_ready(endpoint, point)
        # Make the automatically changed wait pose immediately visible in the
        # Named Robot Pose Teaching panel.  The stored joint seed remains the
        # taught one; the corrected TCP is used by the sensed weld workflow.
        self.teaching_pose_name.set(TEACHING_POSES[wait_name])
        self.teaching_pose_changed()
        yaw_commit_done = False
        if all(self.computed_seam_endpoints.values()):
            try:
                teaching_reference = self._ensure_seam_teaching_reference(
                    require_complete=True
                )
                if teaching_reference is None:
                    return None
                count = int(self.corner_touch_count.get())
                corrected_start, corrected_goal, delta_yaw, orientation_label = (
                    apply_sensed_seam_orientation(
                        teaching_reference["weld_start"][3],
                        teaching_reference["weld_end"][3],
                        self.computed_seam_endpoints["start"],
                        self.computed_seam_endpoints["goal"],
                        self.seam_orientation_mode.get(),
                    )
                )
                self.computed_seam_endpoints = {
                    "start": corrected_start,
                    "goal": corrected_goal,
                }
                self._update_seam_yaw_status(
                    self.computed_seam_endpoints["start"],
                    self.computed_seam_endpoints["goal"],
                )
                # Both wait poses are fixed, manually taught standby poses.
                # Cartesian transitions connect them to yaw-corrected seam
                # endpoints with linear XYZ and orientation SLERP.
                self.computed_seam_wait_points = {
                    "start": copy.deepcopy(
                        self.taught_robot_poses["weld_start_wait"][3]
                    ),
                    "goal": copy.deepcopy(
                        self.taught_robot_poses["weld_goal_wait"][3]
                    ),
                }
                preview = linear_pose_waypoints(
                    corrected_start,
                    corrected_goal,
                    count,
                )
                self.corrected_two_touch_seam = copy.deepcopy(preview)
                self.node.publish_points(preview, self.show_path.get())
                self.correct_two_touch_seam()
                yaw_commit_done = True
                self.log(
                    "Both endpoints ready · "
                    f"orientation={orientation_label} · "
                    f"World Δyaw={math.degrees(delta_yaw):+.3f}° · "
                    "both wait poses kept as taught standby"
                )
            except (ValueError, tk.TclError) as error:
                self.error(f"Endpoint preview failed: {error}")
                return None
        values = self._pose_values(point)
        wait_values = self._pose_values(wait_point)
        wall_values = self._pose_values(wall)
        floor_values = self._pose_values(floor)
        taught_values = self._pose_values(endpoint_data[3])
        correction_mm = tuple(
            (values[index] - taught_values[index]) * 1000.0
            for index in range(3)
        )
        self.log(
            f"{endpoint.upper()} SEAM XYZ INPUT · "
            f"wall=({wall_values[0]:.6f}, {wall_values[1]:.6f}, "
            f"{wall_values[2]:.6f}) · "
            f"floor=({floor_values[0]:.6f}, {floor_values[1]:.6f}, "
            f"{floor_values[2]:.6f}) m"
        )
        self.log(
            f"{endpoint.upper()} SEAM XYZ RESULT · "
            f"computed=({values[0]:.6f}, {values[1]:.6f}, "
            f"{values[2]:.6f}) m · old teaching="
            f"({taught_values[0]:.6f}, {taught_values[1]:.6f}, "
            f"{taught_values[2]:.6f}) m · correction="
            f"({correction_mm[0]:+.3f}, {correction_mm[1]:+.3f}, "
            f"{correction_mm[2]:+.3f}) mm · "
            + (
                "orientation=yaw-corrected after both endpoints"
                if yaw_commit_done
                else "orientation=temporary teaching value"
            )
        )
        self.corner_touch_status.configure(
            text=(
                f"{endpoint.upper()} computed · wall={wall_label} · "
                f"base={floor_label} · "
                f"seam=({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}) · "
                f"fixed wait=({wait_values[0]:.4f}, {wait_values[1]:.4f}, "
                f"{wait_values[2]:.4f})"
            )
        )
        if update_wait_joints and not yaw_commit_done:
            target_labels = " and ".join(
                TEACHING_POSES[teaching_name]
                for teaching_name, _stored, _tcp in updates
            )
            self.pipeline_waiting(
                f"{endpoint.upper()} TCP computed · resolving MoveIt IK for "
                f"{target_labels}"
            )
            ik_targets = tuple(
                (
                    endpoint,
                    stored[0],
                    copy.deepcopy(corrected_tcp),
                    tuple(stored[1]),
                    teaching_name,
                )
                for teaching_name, stored, corrected_tcp in updates
            )
            threading.Thread(
                target=self.node.resolve_tcp_joint_states,
                args=(ik_targets,),
                daemon=True,
            ).start()
        else:
            self.pipeline_result(
                f"{endpoint.upper()} TWO-POSE TCP APPLIED · "
                f"{TEACHING_POSES[pose_name]} TCP="
                f"({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}) · "
                f"{TEACHING_POSES[wait_name]} TCP="
                f"({wait_values[0]:.4f}, {wait_values[1]:.4f}, "
                f"{wait_values[2]:.4f}) · YAML saved"
            )
        if update_wait_joints:
            self.show_touch_geometry_pyplot(endpoint)
        return point

    def apply_corrected_tcp_joint_state(
        self,
        endpoint,
        teaching_name,
        planning_group,
        joint_names,
        positions,
        tcp,
    ):
        try:
            save_initial_state_yaml(
                self._initial_state_yaml_path(planning_group, teaching_name),
                planning_group,
                joint_names,
                positions,
                tcp,
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.corrected_tcp_joint_state_failed(
                endpoint, teaching_name, str(error)
            )
            return
        self.taught_robot_poses[teaching_name] = (
            planning_group,
            tuple(joint_names),
            tuple(positions),
            copy.deepcopy(tcp),
        )
        self.teaching_pose_name.set(TEACHING_POSES[teaching_name])
        self.teaching_pose_changed()
        values = self._pose_values(tcp)
        self.pipeline_result(
            f"{endpoint.upper()} TWO-POSE APPLIED · "
            f"{TEACHING_POSES[teaching_name]} TCP="
            f"({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}) · "
            "MoveIt joint state resolved and YAML saved · computation complete"
        )

    def corrected_tcp_joint_state_failed(self, endpoint, teaching_name, message):
        self.error(
            f"{str(endpoint).upper()} TCP was computed, but "
            f"{TEACHING_POSES.get(teaching_name, teaching_name)} IK update "
            f"failed: {message}"
        )

    def _sensed_motion_step(self, points, label, slot, touch_guard=False):
        return {
            "type": "motion",
            "planning_group": "right_manipulator",
            "points": copy.deepcopy(points),
            "velocity_scale": max(
                0.01, min(1.0, self.velocity_percent.get() / 100.0)
            ),
            "interpolation_step": max(
                0.0005,
                min(0.02, float(self.interpolation_step_mm.get()) * 0.001),
            ),
            "path_kind": label,
            "parallel_slot": slot,
            "duration": 0.0,
            "touch_guard": bool(touch_guard),
        }

    def build_sensed_weld_sequence(self):
        """Append a weld workflow with START sensing and an optional sensed GOAL."""
        if (
            self.computed_seam_endpoints["start"] is None
            or self.computed_seam_wait_points["start"] is None
        ):
            if self.compute_seam_endpoint("start", update_wait_joints=False) is None:
                return

        goal_is_sensed = all(
            self.seam_probe_touches.get(f"goal_{surface}") is not None
            for surface in ("wall", "floor")
        )
        if goal_is_sensed and self.compute_seam_endpoint(
            "goal", update_wait_joints=False
        ) is None:
            return

        goal_data = self.taught_robot_poses.get("weld_end")
        goal_wait_data = self.taught_robot_poses.get("weld_goal_wait")
        finish_data = self.taught_robot_poses.get("weld_finish")
        if goal_data is None or goal_wait_data is None or finish_data is None:
            self.error(
                f"Capture/load {TEACHING_POSES['weld_end']} and "
                f"{TEACHING_POSES['weld_goal_wait']} and "
                f"{TEACHING_POSES['weld_finish']} first"
            )
            return
        if (
            goal_data[0] != "right_manipulator"
            or goal_wait_data[0] != "right_manipulator"
            or finish_data[0] != "right_manipulator"
        ):
            self.error(
                "Weld goal, goal-wait, and end poses must belong to the right arm"
            )
            return
        try:
            settings = self._digital_weld_settings()
            # D-WELD ON drives the welder sequence itself.  Do not emit the
            # separate Gas Check command from an automatically built weld.
            settings["preflow_seconds"] = 0.0
            start_wait_data = self.taught_robot_poses["weld_start_wait"]
            if start_wait_data is None:
                raise ValueError(
                    f"Capture/load {TEACHING_POSES['weld_start_wait']} first"
                )
            start_wait_group = start_wait_data[0]
            if start_wait_group != "right_manipulator":
                raise ValueError("Weld start-wait pose must belong to the right arm")
            start = self.computed_seam_endpoints["start"]
            if goal_is_sensed:
                goal = self.computed_seam_endpoints["goal"]
                goal_source = "sensed GOAL"
                goal_path_name = "sensed_goal"
            else:
                goal = copy.deepcopy(goal_data[3])
                goal_source = TEACHING_POSES["weld_end"]
                goal_path_name = "taught_goal"
            count = int(self.corner_touch_count.get())
            preview = linear_pose_waypoints(start, goal, count)
            self.node.publish_points(preview, self.show_path.get())
            base_slot = int(self.sequence_parallel_slot.get())

            def named_step(name, stored, slot):
                group, names, joints, tcp = stored
                return {
                    "type": "named_pose",
                    "pose_name": name,
                    "pose_label": TEACHING_POSES[name],
                    "planning_group": group,
                    "joint_names": tuple(names),
                    "positions": tuple(joints),
                    "tcp_pose": copy.deepcopy(tcp),
                    "velocity_scale": max(
                        0.01,
                        min(1.0, self.velocity_percent.get() / 100.0),
                    ),
                    "parallel_slot": slot,
                    "duration": 0.0,
                    "touch_guard": True,
                    "continue_after_touch": False,
                }

            approach_start = self._sensed_motion_step(
                (start,),
                "start_wait_to_sensed_start_DI8_guarded",
                base_slot + 1,
                touch_guard=True,
            )
            approach_start["continue_after_touch"] = True
            approach_start["accept_initial_touch"] = True
            steps = [
                named_step("weld_start_wait", start_wait_data, base_slot),
                approach_start,
                {
                    "type": "digital_weld", "command": "on",
                    "settings": copy.deepcopy(settings),
                    "parallel_slot": base_slot + 2, "duration": 0.0,
                },
                self._sensed_motion_step(
                    (start, goal),
                    f"sensed_start_to_{goal_path_name}_weld_DI8_ignored",
                    base_slot + 3,
                ),
                {
                    "type": "digital_weld", "command": "off",
                    "settings": None,
                    "parallel_slot": base_slot + 4, "duration": 0.0,
                },
                named_step(
                    "weld_goal_wait", goal_wait_data, base_slot + 5
                ),
                named_step("weld_finish", finish_data, base_slot + 6),
            ]
        except (ValueError, TypeError, tk.TclError) as error:
            self.error(f"Cannot build sensed weld sequence: {error}")
            return
        self.sequence_steps.extend(steps)
        self.refresh_sequence_table(select_last=True)
        self.log(
            f"Built weld workflow from sensed START to {goal_source} · "
            f"{len(steps)} steps · slots {base_slot}..{base_slot + 6} · "
            "START WAIT → START → D-WELD ON → GOAL → D-WELD OFF → "
            "END WAIT → END · D-WELD ON only after START arrival"
        )

    def compute_two_touch_seam(self):
        missing = [
            name
            for name in CORNER_TOUCH_NAMES
            if self.seam_probe_touches[name] is None
        ]
        if missing:
            self.error(
                "Complete all four DI8 probes first: " + ", ".join(missing)
            )
            return
        teaching_reference = self._ensure_seam_teaching_reference(
            require_complete=True
        )
        if teaching_reference is None:
            return
        start_data = teaching_reference["weld_start"]
        end_data = teaching_reference["weld_end"]
        if start_data is None or end_data is None:
            self.error("Capture/load Weld start and Weld goal poses first")
            return
        if (
            start_data[0] != self.planning_group.get()
            or end_data[0] != self.planning_group.get()
        ):
            self.error("Weld start/goal teaching poses belong to another arm")
            return
        try:
            count = int(self.corner_touch_count.get())
            wall_offset = float(self.seam_wall_offset_mm.get()) * 0.001
            floor_offset = float(self.seam_floor_offset_mm.get()) * 0.001
        except (ValueError, tk.TclError):
            self.error("Seam point count or probe offsets are invalid")
            return
        try:
            raw_points = corner_seam_from_touches(
                self.seam_probe_touches,
                count,
            )
            (
                _reference,
                wall_normal,
                floor_normal,
                wall_label,
                floor_label,
            ) = self._seam_geometry_settings(require_teaching=True)
            sensed_start = generalized_corner_endpoint_from_two_touches(
                self.seam_probe_touches["start_wall"],
                self.seam_probe_touches["start_floor"],
                start_data[3],
                start_data[3],
                end_data[3],
                wall_normal,
                floor_normal,
                wall_offset,
                floor_offset,
            )
            sensed_goal = generalized_corner_endpoint_from_two_touches(
                self.seam_probe_touches["goal_wall"],
                self.seam_probe_touches["goal_floor"],
                end_data[3],
                start_data[3],
                end_data[3],
                wall_normal,
                floor_normal,
                wall_offset,
                floor_offset,
            )
            corrected_points = linear_pose_waypoints(sensed_start, sensed_goal, count)
            # Raw touch geometry is diagnostic.  The adopted seam uses sensed
            # endpoint XYZ and rotates both taught welding orientations by the
            # same sensed-vs-taught World yaw delta.
            raw_points[0].orientation = copy.deepcopy(
                start_data[3].orientation
            )
            raw_points[-1].orientation = copy.deepcopy(
                end_data[3].orientation
            )
            raw_points[:] = linear_pose_waypoints(
                raw_points[0], raw_points[-1], count
            )
            corrected_start, corrected_goal, delta_yaw, orientation_label = (
                apply_sensed_seam_orientation(
                    start_data[3],
                    end_data[3],
                    corrected_points[0],
                    corrected_points[-1],
                    self.seam_orientation_mode.get(),
                )
            )
            corrected_points[:] = linear_pose_waypoints(
                corrected_start, corrected_goal, count
            )
        except ValueError as error:
            self.error(f"Four-touch seam generation failed: {error}")
            return
        self.raw_two_touch_seam = copy.deepcopy(raw_points)
        self.corrected_two_touch_seam = copy.deepcopy(corrected_points)
        self.computed_seam_endpoints = {
            "start": copy.deepcopy(corrected_points[0]),
            "goal": copy.deepcopy(corrected_points[-1]),
        }
        self._update_seam_yaw_status(
            self.computed_seam_endpoints["start"],
            self.computed_seam_endpoints["goal"],
        )
        try:
            self.computed_seam_wait_points = {
                "start": copy.deepcopy(
                    self.taught_robot_poses["weld_start_wait"][3]
                ),
                "goal": copy.deepcopy(
                    self.taught_robot_poses["weld_goal_wait"][3]
                ),
            }
        except (TypeError, ValueError):
            self.computed_seam_wait_points = {"start": None, "goal": None}
        self.path_kind = "di8_four_touch_raw"
        self.weave_source = copy.deepcopy(raw_points)
        self.set_points(raw_points)
        self.node.publish_seam_comparison(
            raw_points,
            corrected_points,
            self.show_path.get(),
        )
        for endpoint in ("start", "goal"):
            self._publish_touch_geometry_if_ready(
                endpoint, self.computed_seam_endpoints[endpoint]
            )
        self.correct_two_touch_seam_button.configure(state=tk.NORMAL)
        self.corner_touch_status.configure(
            text=(
                f"RAW + CORRECTED PREVIEW · {len(raw_points)} points · "
                "opaque=raw, translucent=offset corrected"
            )
        )
        self.log(
            "Computed START→GOAL seam from four DI8 touches · "
            f"wall={wall_label} · base={floor_label} · "
            f"orientation={orientation_label} · "
            f"World Δyaw={math.degrees(delta_yaw):+.3f}° · "
            "both wait poses kept as taught standby"
        )
        self.show_touch_geometry_pyplot()
        # Calculation is the commit point: persist corrected start/goal and
        # wait teaching YAML immediately instead of requiring a second button.
        self.correct_two_touch_seam()

    def correct_two_touch_seam(self):
        if not self.corrected_two_touch_seam:
            self.error("Compute the raw/corrected seam preview first")
            return
        start_data = self.taught_robot_poses["weld_start"]
        end_data = self.taught_robot_poses["weld_end"]
        if start_data is None or end_data is None:
            self.error("Weld start/goal teaching poses are unavailable")
            return
        corrected_start = copy.deepcopy(self.corrected_two_touch_seam[0])
        corrected_end = copy.deepcopy(self.corrected_two_touch_seam[-1])
        updates = [
            ("weld_start", start_data, corrected_start),
            ("weld_end", end_data, corrected_end),
        ]
        try:
            saved_paths = []
            for pose_name, stored, corrected_tcp in updates:
                planning_group, joint_names, positions, _old_tcp = stored
                yaml_path = self._initial_state_yaml_path(
                    planning_group, pose_name
                )
                save_initial_state_yaml(
                    yaml_path,
                    planning_group,
                    joint_names,
                    positions,
                    corrected_tcp,
                )
                saved_paths.append(yaml_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.error(f"Corrected seam YAML update failed: {error}")
            return
        self.log(
            "CORRECTED SEAM TCP YAML SAVED · "
            + " · ".join(path.name for path in saved_paths)
        )
        for pose_name, stored, corrected_tcp in updates:
            planning_group, joint_names, positions, _old_tcp = stored
            self.taught_robot_poses[pose_name] = (
                planning_group,
                joint_names,
                positions,
                corrected_tcp,
            )
        ik_targets = []
        for pose_name, stored, corrected_tcp in updates:
            endpoint = (
                "goal"
                if pose_name in ("weld_end", "weld_goal_wait")
                else "start"
            )
            ik_targets.append((
                endpoint,
                stored[0],
                copy.deepcopy(corrected_tcp),
                tuple(stored[1]),
                pose_name,
            ))
        threading.Thread(
            target=self.node.resolve_tcp_joint_states,
            args=(tuple(ik_targets),),
            daemon=True,
        ).start()
        self.path_kind = "di8_four_touch_corrected"
        self.weave_source = copy.deepcopy(self.corrected_two_touch_seam)
        self.set_points(self.corrected_two_touch_seam)
        self.node.publish_points(
            self.corrected_two_touch_seam, self.show_path.get()
        )
        self.correct_two_touch_seam_button.configure(state=tk.DISABLED)
        self.corner_touch_status.configure(
            text=(
                "CORRECTED SEAM ADOPTED · Weld start/goal YAML saved · "
                "both wait poses kept as manual standby"
            )
        )
        self.log(
            "Adopted corrected seam and updated Weld start/Weld goal YAML · "
            "START/GOAL wait unchanged · sequential MoveIt IK update started"
        )

    def _advance_corner_touch_target(self):
        current = self.corner_touch_target.get()
        index = CORNER_TOUCH_NAMES.index(current)
        if index + 1 < len(CORNER_TOUCH_NAMES):
            self.corner_touch_target.set(CORNER_TOUCH_NAMES[index + 1])

    def _record_corner_touch(self, pose, source):
        target = self.corner_touch_target.get()
        self.corner_touches[target] = copy.deepcopy(pose)
        captured = [name for name in CORNER_TOUCH_NAMES if self.corner_touches[name] is not None]
        self.corner_touch_status.configure(
            text=f"Captured {target} from {source} · {len(captured)}/4: {', '.join(captured)}"
        )
        self.log(f"Corner touch stored · {target} · source={source}")
        self._advance_corner_touch_target()

    def generate_corner_touch_seam(self):
        try:
            points = corner_seam_from_touches(
                self.corner_touches, int(self.corner_touch_count.get())
            )
        except (ValueError, tk.TclError) as error:
            self.error(f"Corner seam generation failed: {error}")
            return
        self.path_kind = "corner_midpoint"
        self.weave_source = copy.deepcopy(points)
        self.set_points(points)
        self.node.publish_points(points, self.show_path.get())
        self.log(
            "Generated 90° corner root seam from two floor/wall 1:1 midpoint pairs"
        )

    def add_motion_sequence_step(self):
        if not self.points:
            self.error("Create or teach a motion path first")
            return
        try:
            interpolation = float(self.interpolation_step_mm.get()) * 0.001
        except (ValueError, tk.TclError):
            self.error("Cartesian interpolation step is invalid")
            return
        try:
            slot, duration = self._sequence_slot_and_duration()
        except ValueError as error:
            self.error(str(error))
            return
        self.sequence_steps.append({
            "type": "motion",
            "planning_group": self.planning_group.get(),
            "points": copy.deepcopy(self.points),
            "velocity_scale": max(0.01, min(1.0, self.velocity_percent.get() / 100.0)),
            "interpolation_step": interpolation,
            "path_kind": self.path_kind,
            "parallel_slot": slot,
            "duration": duration,
            "touch_guard": False,
            "continue_after_touch": False,
        })
        self.refresh_sequence_table(select_last=True)

    def add_latest_rviz_plan_step(self):
        display, age = self.node.latest_rviz_plan()
        if display is None:
            self.error("Plan a path in RViz/MoveIt first")
            return
        try:
            slot, duration = self._sequence_slot_and_duration()
        except ValueError as error:
            self.error(str(error))
            return
        trajectories = [
            copy.deepcopy(trajectory)
            for trajectory in display.trajectory
            if trajectory.joint_trajectory.points
        ]
        joint_names = tuple(
            dict.fromkeys(
                name
                for trajectory in trajectories
                for name in trajectory.joint_trajectory.joint_names
            )
        )
        arms = [
            arm for arm, names in ARM_JOINT_NAMES.items()
            if names.intersection(joint_names)
        ]
        planning_group = (
            f"{arms[0]}_manipulator" if len(arms) == 1 else "unknown"
        )
        point_count = sum(
            len(trajectory.joint_trajectory.points)
            for trajectory in trajectories
        )
        self.sequence_steps.append({
            "type": "planned_trajectory",
            "planning_group": planning_group,
            "required_arms": tuple(arms),
            "trajectory_start": copy.deepcopy(display.trajectory_start),
            "trajectories": trajectories,
            "model_id": display.model_id,
            "joint_names": joint_names,
            "point_count": point_count,
            "captured_age": float(age),
            "parallel_slot": slot,
            "duration": duration,
        })
        self.refresh_sequence_table(select_last=True)
        self.log(
            f"Added latest RViz plan · {len(trajectories)} trajectory(s) · "
            f"{point_count} points · received {age:.1f} s ago"
        )

    def add_named_pose_sequence_step(self):
        pose_name = self._selected_teaching_pose_name()
        stored = self.taught_robot_poses[pose_name]
        if stored is None:
            self.error(
                f"Capture or load {TEACHING_POSES[pose_name]} first"
            )
            return
        try:
            slot, duration = self._sequence_slot_and_duration()
        except ValueError as error:
            self.error(str(error))
            return
        planning_group, joint_names, positions, tcp = stored
        self.sequence_steps.append({
            "type": "named_pose",
            "pose_name": pose_name,
            "pose_label": TEACHING_POSES[pose_name],
            "planning_group": planning_group,
            "joint_names": tuple(joint_names),
            "positions": tuple(positions),
            "tcp_pose": copy.deepcopy(tcp),
            "velocity_scale": max(
                0.01,
                min(1.0, self.velocity_percent.get() / 100.0),
            ),
            "parallel_slot": slot,
            "duration": duration,
            "touch_guard": pose_name in DI8_GUARDED_TEACHING_POSES,
            "continue_after_touch": False,
        })
        self.refresh_sequence_table(select_last=True)

    def add_sleep_sequence_step(self):
        try:
            seconds = float(self.sequence_sleep_seconds.get())
        except (ValueError, tk.TclError):
            self.error("Sleep duration is invalid")
            return
        if not math.isfinite(seconds) or not 0.0 <= seconds <= 3600.0:
            self.error("Sleep duration must be in 0..3600 seconds")
            return
        self.sequence_steps.append({
            "type": "sleep",
            "seconds": seconds,
        })
        self.refresh_sequence_table(select_last=True)

    def add_digital_weld_step(self, command):
        command = str(command).strip().lower()
        if command not in ("on", "off", "set"):
            self.error(f"Unknown D-WELD command: {command}")
            return
        try:
            slot, duration = self._sequence_slot_and_duration()
        except ValueError as error:
            self.error(str(error))
            return
        # ARC OFF has no recipe payload.  Do not prevent adding the safety/off
        # command because an unrelated recipe entry is incomplete or invalid.
        settings = None
        if command != "off":
            try:
                settings = self._digital_weld_settings()
            except ValueError as error:
                self.error(f"Cannot add D-WELD {command.upper()}: {error}")
                return
        self.sequence_steps.append({
            "type": "digital_weld",
            "command": command,
            "settings": settings,
            "parallel_slot": slot,
            "duration": duration,
        })
        self.refresh_sequence_table(select_last=True)
        self.log(
            f"Added D-WELD {command.upper()} to sequence · "
            f"slot {slot} · {duration:.3f} s"
        )

    def add_gas_sequence_step(self, enabled):
        try:
            slot, duration = self._sequence_slot_and_duration()
        except ValueError as error:
            self.error(str(error))
            return
        enabled = bool(enabled)
        self.sequence_steps.append({
            "type": "gas",
            "enabled": enabled,
            "parallel_slot": slot,
            "duration": duration,
        })
        self.refresh_sequence_table(select_last=True)
        self.log(
            f"Added GAS {'ON' if enabled else 'OFF'} to sequence · "
            f"slot {slot} · {duration:.3f} s"
        )

    def _sequence_slot_and_duration(self):
        try:
            slot = int(self.sequence_parallel_slot.get())
            duration = float(self.sequence_duration_seconds.get())
        except (ValueError, tk.TclError) as error:
            raise ValueError("Sequence slot/duration is invalid") from error
        if not 1 <= slot <= 999:
            raise ValueError("Sequence parallel slot must be in 1..999")
        if not math.isfinite(duration) or not 0.0 <= duration <= 3600.0:
            raise ValueError("Sequence duration must be in 0..3600 seconds")
        return slot, duration

    def _selected_sequence_index(self):
        selected = self.sequence_table.selection()
        if not selected:
            return None
        return int(selected[0])

    def refresh_sequence_table(self, select_last=False):
        selected = self._selected_sequence_index() if self.sequence_table.get_children() else None
        self.sequence_table.delete(*self.sequence_table.get_children())
        for index, step in enumerate(self.sequence_steps):
            timing = (
                f"slot {step.get('parallel_slot', index + 1)} · "
                f"{step.get('duration', 0.0):.1f} s"
            )
            if step["type"] == "motion":
                guard_detail = (
                    " · DI8 GUARDED"
                    if step.get("touch_guard", False)
                    else " · DI8 IGNORED"
                )
                detail = (
                    f"{step['planning_group']} · {len(step['points'])} poses · "
                    f"speed {step['velocity_scale']:.0%} · "
                    f"{step['path_kind']}{guard_detail} · {timing}"
                )
                kind = "MOTION"
            elif step["type"] == "planned_trajectory":
                kind = "RVIZ PLAN"
                detail = (
                    f"{step.get('planning_group', 'unknown')} · exact stored "
                    f"trajectory · {len(step['trajectories'])} segment(s) · "
                    f"{step.get('point_count', 0)} points · {timing}"
                )
            elif step["type"] == "named_pose":
                kind = "GO TO POSE"
                detail = (
                    f"{step['pose_label']} · {step['planning_group']} · "
                    f"speed {step['velocity_scale']:.0%} · {timing}"
                )
            elif step["type"] == "sleep":
                kind = "SLEEP"
                detail = f"{step['seconds']:.3f} seconds"
            elif step["type"] == "digital_weld":
                settings = step.get("settings")
                kind = f"D-WELD {step['command'].upper()}"
                if settings is None:
                    detail = f"Hi-COMM · no recipe payload · {timing}"
                else:
                    detail = (
                        f"Hi-COMM · I={settings['current_a']} A "
                        f"V={settings['voltage']:.1f} V · {timing}"
                    )
            elif step["type"] == "gas":
                kind = f"GAS {'ON' if step['enabled'] else 'OFF'}"
                detail = f"Hi-COMM shielding gas · {timing}"
            else:
                kind = f"INCH {step['direction'].upper()}"
                detail = f"Hi-COMM timed wire feed · {timing}"
            self.sequence_table.insert(
                "", tk.END, iid=str(index), values=(index + 1, kind, detail)
            )
        target = len(self.sequence_steps) - 1 if select_last else selected
        if target is not None and 0 <= target < len(self.sequence_steps):
            self.sequence_table.selection_set(str(target))
            self.load_selected_sequence_values()

    def load_selected_sequence_values(self, _event=None):
        """Load the selected row into the Sequence Builder edit controls."""
        index = self._selected_sequence_index()
        if index is None or not 0 <= index < len(self.sequence_steps):
            return
        step = self.sequence_steps[index]
        if step["type"] == "sleep":
            self.sequence_sleep_seconds.set(step.get("seconds", 0.0))
        else:
            self.sequence_parallel_slot.set(step.get("parallel_slot", index + 1))
            self.sequence_duration_seconds.set(step.get("duration", 0.0))
        if step["type"] in ("motion", "named_pose"):
            self.sequence_edit_velocity_percent.set(
                float(step.get("velocity_scale", 0.2)) * 100.0
            )
            self.sequence_edit_touch_guard.set(
                bool(step.get("touch_guard", False))
            )
            self.sequence_edit_continue_after_touch.set(
                bool(step.get("continue_after_touch", False))
            )
        else:
            self.sequence_edit_touch_guard.set(False)
            self.sequence_edit_continue_after_touch.set(False)
        if step["type"] == "digital_weld" and step.get("settings"):
            try:
                settings = validate_digital_weld_settings(step["settings"])
            except ValueError as error:
                self.error(f"Invalid D-WELD sequence settings: {error}")
                return
            step["settings"] = settings
            self.weld_current_raw.set(settings["current_a"])
            self.weld_voltage_raw.set(settings["voltage_tenths"])
            self.weld_material.set(settings["material"])
            self.weld_diameter_mm.set(settings["diameter_mm"])
            self.weld_mode.set(settings["mode"])
            self.weld_gas.set(settings["gas"])
            self.weld_synergic.set(settings["synergic"])
            self.weld_correction.set(settings["correction"])
            self.weld_pre_gas_s.set(settings["pre_gas_s"])
            self.weld_post_gas_s.set(settings["post_gas_s"])
            self.weld_preflow_seconds.set(settings["preflow_seconds"])
        self.sequence_status.configure(
            text=f"Editing sequence #{index + 1} · change values, then Apply selected values"
        )

    def apply_selected_sequence_values(self):
        """Apply editor values to one selected sequence step."""
        index = self._selected_sequence_index()
        if index is None:
            self.error("Select a sequence step to edit")
            return
        step = self.sequence_steps[index]
        try:
            if step["type"] == "sleep":
                seconds = float(self.sequence_sleep_seconds.get())
                if not math.isfinite(seconds) or not 0.0 <= seconds <= 3600.0:
                    raise ValueError("Sleep duration must be in 0..3600 seconds")
                step["seconds"] = seconds
            else:
                slot, duration = self._sequence_slot_and_duration()
                step["parallel_slot"] = slot
                step["duration"] = duration
            if step["type"] in ("motion", "named_pose"):
                speed = float(self.sequence_edit_velocity_percent.get())
                if not math.isfinite(speed) or not 1.0 <= speed <= 100.0:
                    raise ValueError("Selected motion speed must be in 1..100%")
                step["velocity_scale"] = speed / 100.0
                step["touch_guard"] = bool(
                    self.sequence_edit_touch_guard.get()
                )
                step["continue_after_touch"] = bool(
                    self.sequence_edit_continue_after_touch.get()
                )
            if (
                step["type"] == "digital_weld"
                and step.get("command") in ("on", "set")
            ):
                step["settings"] = copy.deepcopy(
                    self._digital_weld_settings()
                )
        except (ValueError, tk.TclError) as error:
            self.error(f"Sequence edit failed: {error}")
            return
        self.refresh_sequence_table()
        self.sequence_table.selection_set(str(index))
        self.load_selected_sequence_values()
        self.log(f"Updated sequence #{index + 1} values")

    def open_sequence_step_editor(self, _event=None):
        """Open a type-aware editor for one generated scenario row."""
        index = self._selected_sequence_index()
        if index is None:
            self.error("Select a sequence step to edit")
            return
        step = self.sequence_steps[index]
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit sequence #{index + 1} · {step['type']}")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        variables = {}
        row = 0

        def entry(name, label, value, width=14):
            nonlocal row
            variable = tk.StringVar(value=str(value))
            variables[name] = variable
            ttk.Label(body, text=label).grid(
                row=row, column=0, padx=4, pady=3, sticky=tk.W
            )
            ttk.Entry(body, textvariable=variable, width=width).grid(
                row=row, column=1, padx=4, pady=3, sticky=tk.W
            )
            row += 1

        def choice(name, label, value, values):
            nonlocal row
            variable = tk.StringVar(value=str(value))
            variables[name] = variable
            ttk.Label(body, text=label).grid(
                row=row, column=0, padx=4, pady=3, sticky=tk.W
            )
            ttk.Combobox(
                body,
                textvariable=variable,
                values=tuple(values),
                state="readonly",
                width=16,
            ).grid(row=row, column=1, padx=4, pady=3, sticky=tk.W)
            row += 1

        def check(name, label, value):
            nonlocal row
            variable = tk.BooleanVar(value=bool(value))
            variables[name] = variable
            ttk.Checkbutton(
                body, text=label, variable=variable
            ).grid(row=row, column=0, columnspan=2, padx=4, pady=3, sticky=tk.W)
            row += 1

        if step["type"] != "sleep":
            entry("parallel_slot", "Parallel slot", step.get("parallel_slot", 1))
            entry("duration", "Duration (s)", step.get("duration", 0.0))
        if step["type"] == "sleep":
            entry("seconds", "Sleep (s)", step.get("seconds", 0.0))
        elif step["type"] in ("motion", "named_pose"):
            entry(
                "velocity_percent",
                "Motion speed (%)",
                float(step.get("velocity_scale", 0.2)) * 100.0,
            )
            if step["type"] == "motion":
                entry(
                    "interpolation_mm",
                    "Interpolation (mm)",
                    float(step.get("interpolation_step", 0.005)) * 1000.0,
                )
            check("touch_guard", "Stop this step on DI8", step.get("touch_guard"))
            check(
                "continue_after_touch",
                "Continue scenario after confirmed DI8 stop",
                step.get("continue_after_touch"),
            )
        elif step["type"] == "digital_weld":
            choice("command", "D-WELD command", step["command"], ("on", "off", "set"))
            settings = step.get("settings") or copy.deepcopy(
                DEFAULT_DIGITAL_WELD_SETTINGS
            )
            entry("current_a", "Current (A)", settings["current_a"])
            entry("voltage", "Voltage (V)", settings["voltage"])
            choice("material", "Wire material", settings["material"], MATERIAL_CODES)
            choice("diameter_mm", "Wire diameter (mm)", settings["diameter_mm"], DIAMETER_CODES)
            choice("mode", "Mode", settings["mode"], MODE_CODES)
            choice("gas", "Gas type", settings["gas"], GAS_CODES)
            check("synergic", "Synergic", settings["synergic"])
            entry("correction", "Correction", settings["correction"])
            entry("pre_gas_s", "Recipe pre-gas (s)", settings["pre_gas_s"])
            entry("post_gas_s", "Recipe post-gas (s)", settings["post_gas_s"])
            entry(
                "preflow_seconds",
                "ARC ON pre-flow (s)",
                settings["preflow_seconds"],
            )
        elif step["type"] == "gas":
            choice(
                "enabled", "Gas command",
                "on" if step["enabled"] else "off", ("on", "off")
            )

        def save():
            try:
                updated = copy.deepcopy(step)
                if updated["type"] == "sleep":
                    seconds = float(variables["seconds"].get())
                    if not math.isfinite(seconds) or not 0.0 <= seconds <= 3600.0:
                        raise ValueError("Sleep must be in 0..3600 seconds")
                    updated["seconds"] = seconds
                else:
                    slot = int(variables["parallel_slot"].get())
                    duration = float(variables["duration"].get())
                    if not 1 <= slot <= 999:
                        raise ValueError("Parallel slot must be in 1..999")
                    if not math.isfinite(duration) or not 0.0 <= duration <= 3600.0:
                        raise ValueError("Duration must be in 0..3600 seconds")
                    updated["parallel_slot"] = slot
                    updated["duration"] = duration
                if updated["type"] in ("motion", "named_pose"):
                    speed = float(variables["velocity_percent"].get())
                    if not 1.0 <= speed <= 100.0:
                        raise ValueError("Motion speed must be in 1..100%")
                    updated["velocity_scale"] = speed / 100.0
                    updated["touch_guard"] = variables["touch_guard"].get()
                    updated["continue_after_touch"] = variables[
                        "continue_after_touch"
                    ].get()
                    if updated["type"] == "motion":
                        interpolation = float(variables["interpolation_mm"].get())
                        if not 0.5 <= interpolation <= 20.0:
                            raise ValueError("Interpolation must be in 0.5..20 mm")
                        updated["interpolation_step"] = interpolation * 0.001
                elif updated["type"] == "digital_weld":
                    updated["command"] = variables["command"].get()
                    if updated["command"] == "off":
                        updated["settings"] = None
                    else:
                        current = int(round(float(variables["current_a"].get())))
                        voltage_tenths = int(round(
                            float(variables["voltage"].get()) * 10.0
                        ))
                        settings = validate_digital_weld_settings({
                            "current_a": current,
                            "voltage_tenths": voltage_tenths,
                            "material": variables["material"].get(),
                            "diameter_mm": variables["diameter_mm"].get(),
                            "mode": variables["mode"].get(),
                            "gas": variables["gas"].get(),
                            "synergic": variables["synergic"].get(),
                            "correction": variables["correction"].get(),
                            "pre_gas_s": variables["pre_gas_s"].get(),
                            "post_gas_s": variables["post_gas_s"].get(),
                            "preflow_seconds": variables[
                                "preflow_seconds"
                            ].get(),
                        })
                        updated["settings"] = settings
                elif updated["type"] == "gas":
                    updated["enabled"] = variables["enabled"].get() == "on"
            except (ValueError, tk.TclError) as error:
                messagebox.showerror("Invalid sequence value", str(error), parent=dialog)
                return
            self.sequence_steps[index] = updated
            self.refresh_sequence_table()
            self.sequence_table.selection_set(str(index))
            self.load_selected_sequence_values()
            self.log(f"Updated scenario step #{index + 1} in editor")
            dialog.destroy()

        buttons = ttk.Frame(body)
        buttons.grid(row=row, column=0, columnspan=2, pady=(10, 0), sticky=tk.E)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(
            side=tk.RIGHT, padx=3
        )
        ttk.Button(buttons, text="Apply", command=save).pack(side=tk.RIGHT, padx=3)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.grab_set()

    def delete_sequence_step(self):
        index = self._selected_sequence_index()
        if index is None:
            self.error("Select a sequence step")
            return
        del self.sequence_steps[index]
        self.refresh_sequence_table()

    def move_sequence_step(self, offset):
        index = self._selected_sequence_index()
        if index is None:
            self.error("Select a sequence step")
            return
        target = index + offset
        if not 0 <= target < len(self.sequence_steps):
            return
        self.sequence_steps[index], self.sequence_steps[target] = (
            self.sequence_steps[target], self.sequence_steps[index]
        )
        self.refresh_sequence_table()
        self.sequence_table.selection_set(str(target))

    def run_sequence(self, run_all, execute_requested):
        if self.sequence_running:
            self.error("A sequence is already running")
            return
        if run_all:
            indices = list(range(len(self.sequence_steps)))
        else:
            selected = self._selected_sequence_index()
            indices = [] if selected is None else [selected]
        if not indices:
            self.error("Add or select a sequence step")
            return
        steps = [copy.deepcopy(self.sequence_steps[index]) for index in indices]
        if execute_requested:
            required_arms = set()
            for step in steps:
                if step["type"] == "planned_trajectory":
                    required_arms.update(step.get("required_arms", ()))
                elif step["type"] in ("motion", "named_pose"):
                    required_arms.add(
                        step["planning_group"].removesuffix("_manipulator")
                    )
            disconnected = [
                arm for arm in sorted(required_arms)
                if not self.robot_connected.get(arm, False)
            ]
            if not self.execution_allowed or disconnected:
                self.error(
                    "Physical execution is unavailable or a required robot is "
                    f"disconnected: {', '.join(disconnected) or 'execution disabled'}"
                )
                return
            contains_weld_command = any(
                step["type"] in ("digital_weld", "gas") for step in steps
            )
            if contains_weld_command and (
                not self.hicomm_connected
            ):
                self.error(
                    "Connect Hi-COMM"
                )
                return
            contains_arc_on = any(
                step["type"] == "digital_weld"
                and step["command"] == "on"
                for step in steps
            )
            if contains_arc_on and not self.hicomm_arc_unlocked.get():
                self.error("Unlock ARC ON before executing this sequence")
                return
            motion_counts = {}
            for local_index, step in enumerate(steps):
                if step["type"] not in (
                    "motion", "named_pose", "planned_trajectory"
                ):
                    continue
                slot = step.get("parallel_slot", local_index + 1)
                motion_counts[slot] = motion_counts.get(slot, 0) + 1
            duplicate_motion_slots = [
                slot for slot, count in motion_counts.items() if count > 1
            ]
            if duplicate_motion_slots:
                self.error(
                    "Only one robot motion is allowed in each parallel slot: "
                    + ", ".join(map(str, duplicate_motion_slots))
                )
                return
            if not messagebox.askyesno(
                "Execute sequence",
                f"Execute {len(steps)} stored step(s) on physical equipment?",
            ):
                return
            if self.hicomm_client is not None:
                self.hicomm_client.allow_outputs()
        self.sequence_running = True
        self.sequence_stop_requested = False
        mode = "EXECUTE" if execute_requested else "PLAN"
        self._set_sequence_status(
            f"{mode} running · {len(steps)} step(s)"
        )
        threading.Thread(
            target=self._sequence_worker,
            args=(steps, indices, execute_requested),
            daemon=True,
        ).start()

    def _interruptible_wait(self, seconds):
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if self.sequence_stop_requested:
                return False
            time.sleep(min(0.05, deadline - time.monotonic()))
        return True

    def _sequence_worker(self, steps, indices, execute_requested):
        success = True
        message = "complete"
        groups = []
        group_lookup = {}
        for local_index, (stored_index, step) in enumerate(zip(indices, steps)):
            key = (
                ("sleep", stored_index)
                if step["type"] == "sleep"
                else ("slot", step.get("parallel_slot", local_index + 1))
            )
            if key not in group_lookup:
                group_lookup[key] = []
                groups.append((key, group_lookup[key]))
            group_lookup[key].append((stored_index, step))

        for group_index, (key, members) in enumerate(groups, start=1):
            if self.sequence_stop_requested:
                success, message = False, "stopped by operator"
                break
            slot_label = key[1] if key[0] == "slot" else "sleep"
            self.post(
                self._set_sequence_status,
                f"Parallel slot {slot_label} · group "
                f"{group_index}/{len(groups)} · {len(members)} task(s)",
            )
            results = {}
            workers = []

            def run_member(result_key, member_step):
                results[result_key] = self._run_sequence_step(
                    member_step, execute_requested
                )

            for stored_index, step in members:
                worker = threading.Thread(
                    target=run_member,
                    args=(stored_index, step),
                    daemon=True,
                )
                workers.append(worker)
                worker.start()
            for worker in workers:
                worker.join()
            for stored_index, _step in members:
                step_success, step_message = results.get(
                    stored_index, (False, "parallel task produced no result")
                )
                self.post(
                    self.log,
                    f"Sequence #{stored_index + 1} · "
                    f"{'OK' if step_success else 'FAILED'} · {step_message}",
                )
                if not step_success:
                    success, message = False, step_message
                    break
            if not success:
                break
        if execute_requested and (not success or self.sequence_stop_requested):
            client = self.hicomm_client
            if client is not None:
                client.clear_outputs()
        self.post(self._sequence_finished, success, message)

    def _run_sequence_step(self, step, execute_requested):
        if step["type"] == "motion":
            return self.node.run_sequence_cartesian_motion(
                step, execute_requested
            )
        if step["type"] == "planned_trajectory":
            return self.node.run_sequence_planned_trajectory(
                step, execute_requested
            )
        if step["type"] == "named_pose":
            return self.node.run_sequence_named_pose(step, execute_requested)
        if step["type"] == "sleep":
            if not execute_requested:
                return True, "sleep planned (no wait)"
            success = self._interruptible_wait(step["seconds"])
            return success, (
                f"slept {step['seconds']:.3f} seconds"
                if success
                else "sleep interrupted"
            )
        if not execute_requested:
            return True, "Hi-COMM timed command planned (no output sent)"
        duration = float(step.get("duration", 0.0))
        if step["type"] == "digital_weld":
            success, message = self._execute_hicomm_weld(
                step["command"], step["settings"]
            )
            if not success:
                return success, message
            if step["command"] == "on" and duration <= 0.0:
                return True, f"{message} · remains ON until D-WELD OFF"
            waited = self._interruptible_wait(duration)
            if step["command"] == "on":
                off_success, off_message = self._execute_hicomm_weld(
                    "off", step["settings"]
                )
                if not off_success:
                    return False, off_message
            return waited, (
                f"{message} · duration {duration:.3f} seconds"
                if waited
                else "D-WELD duration interrupted"
            )
        if step["type"] == "gas":
            client = self.hicomm_client
            if client is None or not client.connected:
                return False, "Hi-COMM disconnected"
            enabled = bool(step["enabled"])
            try:
                client.set_command_bit(BIT_GAS, enabled)
                if not enabled:
                    return True, "GAS OFF sent"
                # A positive duration makes GAS ON a timed pulse.  Duration 0
                # keeps gas on until an explicit GAS OFF sequence step.
                if duration <= 0.0:
                    return True, "GAS ON sent; remains on until GAS OFF"
                waited = self._interruptible_wait(duration)
                client.set_command_bit(BIT_GAS, False)
                return waited, (
                    f"GAS ON for {duration:.3f} seconds, then OFF"
                    if waited
                    else "GAS timer interrupted; GAS OFF sent"
                )
            except Exception as error:
                client.set_command_bit(BIT_GAS, False)
                return False, str(error)
        return False, "unsupported sequence step"

    def _set_sequence_status(self, text):
        self.sequence_status.configure(text=text)
        self.pipeline_waiting(f"SEQUENCE STATUS · {text}")

    def _sequence_finished(self, success, message):
        self.sequence_running = False
        text = (
            f"Sequence {'complete' if success else 'stopped/failed'} · "
            f"{message}"
        )
        self.sequence_status.configure(text=text)
        if success:
            self.pipeline_result(f"SEQUENCE COMPLETE · {message}")
        else:
            self.error(f"SEQUENCE FAILED · {message}")

    def stop_sequence(self):
        self.sequence_stop_requested = True
        if self.hicomm_client is not None:
            self.hicomm_client.inhibit_outputs()
        self.hicomm_inching_direction = None
        self.hicomm_gas_enabled.set(False)
        self.hicomm_arc_unlocked.set(False)
        self.hicomm_arc_on_button.configure(state=tk.DISABLED)
        self.hicomm_test_status.configure(text="STOP NOW · ALL OUTPUTS INHIBITED")
        self.node.cancel_active_motion()
        devices = [
            device for device in ("left", "right", "head")
            if self.robot_connected.get(device, False)
        ]
        threading.Thread(
            target=self.node.stop_sequence_equipment,
            args=(devices,),
            daemon=True,
        ).start()
        self.sequence_status.configure(
            text="STOP NOW · Hi-COMM inhibited · canceling robot controllers"
        )
        self.pipeline_waiting(
            "STOP NOW · ARC/GAS/INCH OFF · canceling all robot motion"
        )

    def emergency_stop_all(self):
        """Stop every GUI-owned workflow, robot goal, and welder output."""
        self.seam_auto_running = False
        self.seam_auto_expected_kind = None
        self.seam_auto_stage_success = False
        self.seam_auto_stage_event.set()
        self.automatic_probe_kind = None
        self.node.clear_touch_probe()
        self.node.active_touch_guard = None
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        self.auto_start_correction_button.configure(state=tk.NORMAL)
        self.auto_seam_correction_button.configure(state=tk.NORMAL)
        self.stop_auto_seam_button.configure(state=tk.DISABLED)
        self.corner_touch_status.configure(
            text="EMERGENCY STOP requested · all GUI motion workflows aborted"
        )
        self.root.bell()
        self.stop_sequence()
        self.pipeline_waiting(
            "EMERGENCY STOP (SOFTWARE) · ALL ROBOT MOTION + WELDER STOP REQUESTED"
        )

    def sequence_hard_stop_finished(self, results):
        message = " · ".join(results) if results else "no connected arm goal"
        self.sequence_status.configure(text=f"STOP NOW complete · {message}")
        self.pipeline_result(
            f"STOP NOW COMPLETE · welder outputs inhibited · {message}"
        )

    def arm_changed(self, *_args):
        if not hasattr(self, "node"):
            return
        group = self.planning_group.get()
        if group != "right_manipulator":
            self.clear_hicomm_test_outputs()
            self._set_welder_test_controls(False)
        else:
            self._set_welder_test_controls(self.hicomm_connected)
        self.linear_tcp_endpoints = [None, None]
        self.tcp_1_status.configure(text="not saved")
        self.tcp_2_status.configure(text="not saved")
        self.reference_yaw_status.set("Reference yaw: --")
        self.reference_length_status.set("Length: --")
        self.sensed_yaw_status.set("Sensed yaw: --")
        self.delta_yaw_status.set("ΔYaw: --")
        self.generate_tcp_line_button.configure(state=tk.DISABLED)
        self.path_kind = "empty"
        self.weave_source = []
        self.weave_base_paths = {"linear": [], "circle": []}
        self.initial_joint_state = None
        self.initial_plan_ready = False
        self.plan_initial_button.configure(state=tk.DISABLED)
        self.execute_initial_button.configure(state=tk.DISABLED)
        self.initial_state_status.configure(text="not captured")
        self.taught_robot_poses = {name: None for name in TEACHING_POSES}
        self.set_points([])
        self.node.publish_points([], self.show_path.get())
        self._auto_load_teaching_states()
        self._refresh_execution_controls()
        self.log(f"Cartesian arm changed to {group} · path cleared")

    def _selected_arm(self):
        return (
            "left"
            if self.planning_group.get() == "left_manipulator"
            else "right"
        )

    def _selected_robot_connected(self):
        return self.robot_connected[self._selected_arm()]

    def _refresh_execution_controls(self):
        selected_arm = self._selected_arm()
        connected = self.robot_connected[selected_arm]
        for arm in ("left", "right"):
            value = self.robot_connected[arm]
            self.robot_connection_labels[arm].configure(
                text=(
                    f"Connect {arm.upper()} ({self.robot_ips[arm]}): "
                    f"{'O' if value else 'X'}"
                ),
                bg="#e6f4ea" if value else "#fce8e6",
                fg="#137333" if value else "#b3261e",
            )
        head_connected = self.robot_connected["head"]
        head_kind = "FAKE" if self.fake_head_hardware else "CAN2"
        self.robot_connection_labels["head"].configure(
            text=(
                f"Connect HEAD ({head_kind}): "
                f"{'O' if head_connected else 'X'}"
            ),
            bg="#e6f4ea" if head_connected else "#fce8e6",
            fg="#137333" if head_connected else "#b3261e",
        )
        self.plan_button.configure(
            state=tk.NORMAL if self.points and connected else tk.DISABLED
        )
        self.execute_button.configure(
            state=(
                tk.NORMAL
                if (
                    self.plan_approved
                    and self.execution_allowed
                    and connected
                )
                else tk.DISABLED
            )
        )
        self._refresh_initial_position_controls()

    def _refresh_initial_position_controls(self):
        if not hasattr(self, "plan_initial_button"):
            return
        can_plan = (
            self.initial_joint_state is not None
            and self._selected_robot_connected()
        )
        self.plan_initial_button.configure(
            state=tk.NORMAL if can_plan else tk.DISABLED
        )
        can_execute = (
            self.initial_plan_ready
            and self.initial_joint_state is not None
            and self.execution_allowed
            and self._selected_robot_connected()
        )
        self.execute_initial_button.configure(
            state=tk.NORMAL if can_execute else tk.DISABLED
        )

    def log(self, text):
        if text.startswith("ERROR") or " · FAILED · " in text:
            message = text.removeprefix("ERROR · ")
            self._set_pipeline_status("ERROR", message)
        elif text.startswith(("SUCCESS", "RESULT")):
            self._set_pipeline_status("RESULT", text)
        else:
            self._set_pipeline_status("WAITING", text)

    def _set_pipeline_status(self, state, message):
        colors = {
            "WAITING": ("#eeeeee", "#202124"),
            "ERROR": ("#fce8e6", "#b3261e"),
            "RESULT": ("#e6f4ea", "#137333"),
        }
        background, foreground = colors[state]
        self.pipeline_status.configure(
            text=f"{state} · {message}",
            bg=background,
            fg=foreground,
        )
        terminal_message = f"PIPELINE {state} · {message}"
        if hasattr(self, "node"):
            logger = self.node.get_logger()
            if state == "ERROR":
                logger.error(terminal_message)
            elif state == "RESULT":
                logger.info(terminal_message)
            else:
                logger.info(terminal_message)
        else:
            print(terminal_message, flush=True)

    def pipeline_waiting(self, message):
        self._set_pipeline_status("WAITING", message)

    def pipeline_result(self, message):
        self._set_pipeline_status("RESULT", message)

    def error(self, text):
        self.log(f"ERROR · {text}")
        self.plan_approved = False
        state = (
            tk.NORMAL
            if self.points and self._selected_robot_connected()
            else tk.DISABLED
        )
        self.plan_button.configure(state=state)
        self.execute_button.configure(state=tk.DISABLED)

    @staticmethod
    def _pose_values(pose):
        p, q = pose.position, pose.orientation
        return (p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    def set_points(self, points, selected_index=0):
        self.invalidate_approved_plan()
        self.points = copy.deepcopy(list(points))
        self.table.delete(*self.table.get_children())
        for index, pose in enumerate(self.points, 1):
            values = tuple(f"{value:.5f}" for value in self._pose_values(pose))
            self.table.insert("", tk.END, values=(index,) + values)
        self.plan_button.configure(
            state=(
                tk.NORMAL
                if self.points and self._selected_robot_connected()
                else tk.DISABLED
            ),
        )
        children = self.table.get_children()
        if children:
            selected_index = min(max(selected_index, 0), len(children) - 1)
            self.table.selection_set(children[selected_index])
            self.table.focus(children[selected_index])
            self.table.see(children[selected_index])
        self.path_summary.configure(
            text=f"{self.path_kind} · {len(self.points)} poses"
        )

    def set_new_points(self, points, kind):
        if kind != "weave":
            self.weave_source = copy.deepcopy(list(points))
        if kind == "circle":
            self.weave_base_paths["circle"] = copy.deepcopy(list(points))
        elif kind == "tcp_line":
            self.weave_base_paths["linear"] = copy.deepcopy(list(points))
        self.path_kind = kind
        self.set_points(points)

    def set_execution_configuration(
        self,
        execute_motion,
        left_ip,
        right_ip,
        use_fake_head_hardware,
        hicomm_source_ip,
        hicomm_welder_ip,
        hicomm_port,
    ):
        self.execution_allowed = execute_motion
        self.fake_head_hardware = bool(use_fake_head_hardware)
        self.robot_ips = {"left": left_ip, "right": right_ip}
        self.hicomm_source_ip.set(hicomm_source_ip)
        self.hicomm_welder_ip.set(hicomm_welder_ip)
        self.hicomm_port.set(int(hicomm_port))
        self.robot_connected = {
            "left": False,
            "right": False,
            "head": False,
        }
        self._refresh_execution_controls()
        head_kind = "FAKE" if self.fake_head_hardware else "CAN2"
        self.log(
            f"Connecting LEFT {left_ip} + RIGHT {right_ip} · "
            f"HEAD {head_kind} · waiting for measured feedback and "
            "controller readiness · Hi-COMM waits for Connect"
        )

    def robot_feedback_connected(self, arm):
        self.robot_connected[arm] = True
        self._refresh_execution_controls()
        description = "head" if arm == "head" else f"{arm}-arm"
        self.log(f"READY · {description} feedback and controller available")

    def robot_feedback_lost(self, arm, detail="measured joint feedback timeout"):
        self.robot_connected[arm] = False
        if self._selected_arm() == arm:
            self.invalidate_approved_plan()
            self.initial_plan_ready = False
        self._refresh_execution_controls()
        if self._selected_arm() == arm:
            self.plan_button.configure(state=tk.DISABLED)
        description = "head" if arm == "head" else f"{arm}-arm"
        self.log(f"ERROR · {description} unavailable · {detail}")

    def invalidate_approved_plan(self):
        self.plan_approved = False
        if hasattr(self, "execute_button"):
            self.execute_button.configure(state=tk.DISABLED)

    def selected_index(self):
        selection = self.table.selection()
        if not selection:
            return None
        return int(self.table.item(selection[0], "values")[0]) - 1

    def load_selected(self, _event=None):
        index = self.selected_index()
        if index is None:
            return
        for name, value in zip(
            self.POSE_FIELDS,
            self._pose_values(self.points[index]),
        ):
            self.pose_variables[name].set(f"{value:.6f}")

    def publish_edits(self, selected_index):
        if self.path_kind != "weave":
            self.weave_source = copy.deepcopy(self.points)
        self.set_points(self.points, selected_index)
        self.node.publish_points(self.points, self.show_path.get())
        self.log(f"Published edited path · {len(self.points)} poses")

    def toggle_path_visibility(self):
        if (
            self.path_kind == "di8_four_touch_raw"
            and self.raw_two_touch_seam
            and self.corrected_two_touch_seam
        ):
            self.node.publish_seam_comparison(
                self.raw_two_touch_seam,
                self.corrected_two_touch_seam,
                self.show_path.get(),
            )
        else:
            self.node.publish_points(self.points, self.show_path.get())
        state = "ON" if self.show_path.get() else "OFF"
        self.log(f"Planned path visualization {state}")

    def apply_selected(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        try:
            values = [
                float(self.pose_variables[name].get())
                for name in self.POSE_FIELDS
            ]
        except ValueError:
            self.error("Pose fields must be numeric")
            return
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = values[:3]
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = values[3:]
        if not pose_is_valid(pose):
            self.error("Pose must be finite with a non-zero quaternion")
            return
        self.points[index] = pose
        self.publish_edits(index)

    def duplicate_selected(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        self.points.insert(index + 1, copy.deepcopy(self.points[index]))
        self.publish_edits(index + 1)

    def delete_selected(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        self.points.pop(index)
        self.publish_edits(max(0, index - 1))

    def move_selected(self, offset):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        destination = index + offset
        if destination < 0 or destination >= len(self.points):
            return
        self.points[index], self.points[destination] = (
            self.points[destination],
            self.points[index],
        )
        self.publish_edits(destination)

    def nudge(self, axis, direction):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        try:
            distance = float(self.nudge_mm.get()) * 0.001 * direction
        except (ValueError, tk.TclError):
            self.error("Nudge distance must be numeric")
            return
        position = self.points[index].position
        setattr(position, axis, getattr(position, axis) + distance)
        self.publish_edits(index)

    # Acquire a straight seam from an axis and a World/tool reference frame.
    def acquire(self):
        try:
            reference = self.straight_reference.get()
            direction = self.straight_axis.get()
            axis = direction[-1].lower()
            sign = -1.0 if direction.startswith("-") else 1.0
            distance = (
                float(self.straight_distance_mm.get()) * 0.001 * sign
            )
            count = int(self.straight_count.get())
            rpy_offset = tuple(
                math.radians(float(variable.get()))
                for variable in (
                    self.straight_roll_deg,
                    self.straight_pitch_deg,
                    self.straight_yaw_deg,
                )
            )
            explicit_position = None
            if self.straight_start_mode.get() == "World XYZ":
                explicit_position = (
                    float(self.straight_start_x.get()),
                    float(self.straight_start_y.get()),
                    float(self.straight_start_z.get()),
                )
        except (ValueError, tk.TclError):
            self.error(
                "Straight position/distance/count/RPY must be numeric"
            )
            return
        self.log(
            f"Reading current {self.planning_group.get()} TCP and generating "
            f"{reference} {direction} straight seam · orientation rotation "
            f"reference={self.straight_rotation_reference.get()}"
        )
        threading.Thread(
            target=self.node.acquire_points,
            args=(
                reference,
                axis,
                distance,
                count,
                explicit_position,
                rpy_offset,
                self.straight_rotation_reference.get(),
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def generate_circle(self):
        try:
            radius = float(self.radius_mm.get()) * 0.001
            count = int(self.circle_count.get())
        except (ValueError, tk.TclError):
            self.error("Circle radius/count must be numeric")
            return
        threading.Thread(
            target=self.node.generate_circle,
            args=(
                self.circle_axis.get().lower(),
                radius,
                count,
                bool(self.close_circle.get()),
                bool(self.circle_face_center.get()),
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def generate_weave(self):
        base_kind = self.weave_base.get()
        source = self.weave_base_paths.get(base_kind, [])
        if len(source) < 2:
            self.error(
                f"Generate a {base_kind} base path before applying weave"
            )
            return
        try:
            amplitude = float(self.weave_amplitude_mm.get()) * 0.001
            cycles = int(self.weave_cycles.get())
            samples = int(self.weave_samples.get())
        except (ValueError, tk.TclError):
            self.error("Weave settings must be numeric")
            return
        self.weave_source = copy.deepcopy(source)
        seam_length = sum(
            (
                (
                    second.position.x - first.position.x
                ) ** 2
                + (
                    second.position.y - first.position.y
                ) ** 2
                + (
                    second.position.z - first.position.z
                ) ** 2
            ) ** 0.5
            for first, second in zip(source[:-1], source[1:])
        )
        pitch_mm = seam_length * 1000.0 / max(cycles, 1)
        self.weave_summary.configure(
            text=(
                f"±{amplitude * 1000.0:.1f} mm · "
                f"pitch≈{pitch_mm:.1f} mm · {cycles} cycles"
            )
        )
        threading.Thread(
            target=self.node.generate_weave,
            args=(
                copy.deepcopy(source),
                amplitude,
                cycles,
                samples,
                self.weave_axis.get(),
                self.show_path.get(),
            ),
            daemon=True,
        ).start()

    def append_tcp(self):
        threading.Thread(
            target=self.node.capture_tcp,
            args=(
                None,
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def capture_initial_state(self):
        pose_name = self._selected_teaching_pose_name()
        self.pipeline_waiting(
            f"Capturing {TEACHING_POSES[pose_name]} and measured joint angles"
        )
        threading.Thread(
            target=self.node.capture_initial_state,
            args=(self.planning_group.get(), pose_name),
            daemon=True,
        ).start()

    def _selected_teaching_pose_name(self):
        selected_label = self.teaching_pose_name.get()
        return next(
            name
            for name, label in TEACHING_POSES.items()
            if label == selected_label
        )

    def teaching_pose_changed(self, _event=None):
        pose_name = self._selected_teaching_pose_name()
        stored = self.taught_robot_poses[pose_name]
        self.initial_plan_ready = False
        self.node.initial_planned_trajectory = None
        if stored is None:
            self.initial_joint_state = None
            self.initial_state_status.configure(
                text=f"{TEACHING_POSES[pose_name]}: not captured"
            )
        else:
            group, names, positions, tcp = stored
            self.initial_joint_state = (group, names, positions)
            angles = ", ".join(
                f"{math.degrees(value):.1f}°" for value in positions
            )
            tcp_values = self._pose_values(tcp)
            self.initial_state_status.configure(
                text=(
                    f"{TEACHING_POSES[pose_name]} · TCP "
                    f"({tcp_values[0]:.4f}, {tcp_values[1]:.4f}, "
                    f"{tcp_values[2]:.4f}) m · joints {angles}"
                )
            )
        self._refresh_initial_position_controls()

# /home/irs/ros2_ws/src/construct_robot_ros2/construct_description/config
    def _initial_state_yaml_path(self, planning_group=None, pose_name=None):
        group = planning_group or self.planning_group.get()
        selected_pose = pose_name or self._selected_teaching_pose_name()
        return (
            Path.home()
            / "ros2_ws"
            / "src"
            / "construct_robot_ros2"
            / "construct_description"
            / "config"
            / f"{group}_{selected_pose}_state.yaml"
        )

    def _seam_reference_yaml_path(self, planning_group=None):
        group = planning_group or self.planning_group.get()
        return (
            Path.home()
            / "ros2_ws"
            / "src"
            / "construct_robot_ros2"
            / "construct_description"
            / "config"
            / f"{group}_seam_teaching_reference.yaml"
        )

    def _seam_touch_yaml_path(self, planning_group=None):
        group = planning_group or self.planning_group.get()
        return (
            Path.home()
            / "ros2_ws"
            / "src"
            / "construct_robot_ros2"
            / "construct_description"
            / "config"
            / f"{group}_seam_touch_points.yaml"
        )

    def _auto_load_teaching_states(self):
        """Load every named teaching pose found at its default YAML path."""
        planning_group = self.planning_group.get()
        selected_pose = self._selected_teaching_pose_name()
        loaded = []
        for pose_name in TEACHING_POSES:
            path = self._initial_state_yaml_path(planning_group, pose_name)
            if not path.is_file():
                continue
            try:
                group, joint_names, positions, tcp = load_initial_state_yaml(
                    path
                )
            except (OSError, ValueError, yaml.YAMLError) as error:
                self.log(
                    f"Skipped invalid teaching YAML {path.name}: {error}"
                )
                continue
            if group != planning_group:
                self.log(
                    f"Skipped teaching YAML {path.name}: expected "
                    f"{planning_group}, got {group}"
                )
                continue
            self.taught_robot_poses[pose_name] = (
                group,
                tuple(joint_names),
                tuple(positions),
                copy.deepcopy(tcp),
            )
            loaded.append(TEACHING_POSES[pose_name])

        reference_path = self._seam_reference_yaml_path(planning_group)
        if reference_path.is_file():
            try:
                reference_group, reference_poses = (
                    load_seam_teaching_reference_yaml(reference_path)
                )
                if reference_group != planning_group:
                    raise ValueError(
                        f"reference group is {reference_group}, expected "
                        f"{planning_group}"
                    )
                self.seam_teaching_reference = {}
                for name, pose in reference_poses.items():
                    stored = self.taught_robot_poses.get(name)
                    if stored is not None:
                        self.seam_teaching_reference[name] = (
                            stored[0], stored[1], stored[2], copy.deepcopy(pose)
                        )
                for index, pose_name in enumerate(("weld_start", "weld_end")):
                    stored_reference = self.seam_teaching_reference.get(pose_name)
                    if stored_reference is not None:
                        pose = copy.deepcopy(stored_reference[3])
                        self.linear_tcp_endpoints[index] = pose
                        label = self.tcp_1_status if index == 0 else self.tcp_2_status
                        label.configure(
                            text=(
                                f"loaded ({pose.position.x:.3f}, {pose.position.y:.3f}, "
                                f"{pose.position.z:.3f})"
                            )
                        )
                if all(item is not None for item in self.linear_tcp_endpoints):
                    self.generate_tcp_line_button.configure(state=tk.NORMAL)
                self._update_seam_yaw_status()
                self.log(f"Loaded seam teaching reference from {reference_path}")
            except (OSError, ValueError, yaml.YAMLError, KeyError) as error:
                self.error(f"Seam teaching reference load failed: {error}")

        self.teaching_pose_name.set(TEACHING_POSES[selected_pose])
        self.teaching_pose_changed()
        if loaded:
            self.log(
                f"Auto-loaded {len(loaded)} teaching YAML pose(s) for "
                f"{planning_group}: {', '.join(loaded)}"
            )

    def load_initial_state(self):
        pose_name = self._selected_teaching_pose_name()
        default_path = self._initial_state_yaml_path(pose_name=pose_name)
        path = filedialog.askopenfilename(
            title="Load TCP teaching state",
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
            filetypes=(("YAML", "*.yaml *.yml"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            planning_group, joint_names, positions, tcp = (
                load_initial_state_yaml(path)
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.error(f"Failed to load initial state YAML: {error}")
            return
        if planning_group != self.planning_group.get():
            self.error(
                f"YAML is for {planning_group}; selected arm is "
                f"{self.planning_group.get()}"
            )
            return
        self.apply_initial_state(
            pose_name,
            planning_group,
            joint_names,
            positions,
            tcp,
            save_to_yaml=False,
        )
        self.log(f"Loaded TCP teaching state from {path}")

    def apply_initial_state(
        self,
        pose_name,
        planning_group,
        joint_names,
        positions,
        tcp,
        save_to_yaml=True,
    ):
        if pose_name not in TEACHING_POSES:
            self.error(f"Unknown teaching pose: {pose_name}")
            return
        self.teaching_pose_name.set(TEACHING_POSES[pose_name])
        self.taught_robot_poses[pose_name] = (
            planning_group,
            tuple(joint_names),
            tuple(positions),
            copy.deepcopy(tcp),
        )
        if pose_name in TCP_POSE_TEACHING_POSES:
            if self.seam_teaching_reference is None:
                self.seam_teaching_reference = {}
            self.seam_teaching_reference[pose_name] = (
                planning_group,
                tuple(joint_names),
                tuple(positions),
                copy.deepcopy(tcp),
            )
            try:
                save_seam_teaching_reference_yaml(
                    self._seam_reference_yaml_path(planning_group),
                    planning_group,
                    self.seam_teaching_reference,
                )
            except (OSError, ValueError, yaml.YAMLError) as error:
                self.error(f"Seam teaching reference save failed: {error}")
        self.initial_joint_state = (
            planning_group,
            tuple(joint_names),
            tuple(positions),
        )
        self.initial_plan_ready = False
        angles = ", ".join(f"{math.degrees(value):.1f}°" for value in positions)
        self.initial_state_status.configure(
            text=f"{TEACHING_POSES[pose_name]}: {angles}"
        )
        self._refresh_initial_position_controls()
        if pose_name in ("weld_wait", "weld_start_wait", "weld_goal_wait", "weld_finish"):
            self.quick_teaching_status.set(
                f"Saved {TEACHING_POSES[pose_name]}"
            )
        values = self._pose_values(tcp)
        saved_message = ""
        save_error = None
        if save_to_yaml:
            path = self._initial_state_yaml_path(planning_group, pose_name)
            try:
                save_initial_state_yaml(
                    path,
                    planning_group,
                    joint_names,
                    positions,
                    tcp,
                )
                saved_message = f" · saved to {path}"
            except (OSError, ValueError, yaml.YAMLError) as error:
                save_error = error
        self.pipeline_result(
            f"{TEACHING_POSES[pose_name]} captured · TCP World XYZ="
            f"({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}) m"
            f"{saved_message}"
        )
        if save_error is not None:
            self.error(
                f"Initial state captured, but YAML save failed: {save_error}"
            )

    def plan_initial_state(self):
        if self.initial_joint_state is None:
            self.error("Capture or load the selected robot pose first")
            return
        if not self._selected_robot_connected():
            self.error("Connect the selected REAL RB robot first")
            return
        pose_name = self._selected_teaching_pose_name()
        group, joint_names, positions = self.initial_joint_state
        stored = self.taught_robot_poses.get(pose_name)
        target_tcp = copy.deepcopy(stored[3]) if stored is not None else None
        if group != self.planning_group.get():
            self.error("Selected taught pose belongs to another arm")
            return
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        threading.Thread(
            target=self.node.plan_initial_state,
            args=(
                group,
                joint_names,
                positions,
                max(0.01, min(1.0, self.velocity_percent.get() / 100.0)),
                pose_name,
                target_tcp,
            ),
            daemon=True,
        ).start()

    def initial_position_plan_ready(
        self,
        planning_group,
        target_positions,
        velocity_scale,
        message,
    ):
        if self.initial_joint_state is None:
            return
        group, _joint_names, positions = self.initial_joint_state
        if (
            group != planning_group
            or tuple(positions) != tuple(target_positions)
            or group != self.planning_group.get()
            or not math.isclose(
                velocity_scale,
                max(0.01, min(1.0, self.velocity_percent.get() / 100.0)),
            )
        ):
            self.log("Discarded stale taught-pose plan")
            return
        self.initial_plan_ready = True
        self._refresh_initial_position_controls()
        self.pipeline_result(message)

    def execute_initial_plan(self):
        if not self.initial_plan_ready:
            self.error(
                "Plan and inspect the selected taught-pose trajectory first"
            )
            return
        if not self.execution_allowed:
            self.error("Robot execution is disabled by launch configuration")
            return
        if not self._selected_robot_connected():
            self.error("Connect the selected REAL RB robot first")
            return
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        threading.Thread(
            target=self.node.execute_initial_plan,
            daemon=True,
        ).start()

    def initial_position_execution_finished(self, message):
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        self.pipeline_result(message)

    def capture_linear_tcp(self, endpoint_index):
        role = "START" if endpoint_index == 0 else "GOAL"
        self.log(
            f"Teaching reference TCP {endpoint_index + 1} / {role} from "
            f"current {self.planning_group.get()} TCP..."
        )
        threading.Thread(
            target=self.node.capture_linear_tcp,
            args=(endpoint_index, self.planning_group.get()),
            daemon=True,
        ).start()

    def apply_linear_tcp(
        self,
        endpoint_index,
        pose,
        planning_group=None,
        joint_names=None,
        positions=None,
    ):
        """Store TCP1/TCP2 as persistent nominal seam reference teaching."""
        planning_group = planning_group or self.planning_group.get()
        pose_name = "weld_start" if endpoint_index == 0 else "weld_end"
        role = "START" if endpoint_index == 0 else "GOAL"
        if joint_names is None or positions is None:
            self.error(
                f"Reference TCP {endpoint_index + 1} needs a measured six-joint seed"
            )
            return
        stored = (
            planning_group,
            tuple(joint_names),
            tuple(positions),
            copy.deepcopy(pose),
        )
        self.linear_tcp_endpoints[endpoint_index] = copy.deepcopy(pose)
        self.taught_robot_poses[pose_name] = copy.deepcopy(stored)
        if self.seam_teaching_reference is None:
            self.seam_teaching_reference = {}
        self.seam_teaching_reference[pose_name] = copy.deepcopy(stored)
        try:
            state_path = self._initial_state_yaml_path(planning_group, pose_name)
            save_initial_state_yaml(
                state_path,
                planning_group,
                joint_names,
                positions,
                pose,
            )
            reference_path = self._seam_reference_yaml_path(planning_group)
            save_seam_teaching_reference_yaml(
                reference_path,
                planning_group,
                self.seam_teaching_reference,
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.error(f"Reference TCP save failed: {error}")
            return
        position = pose.position
        status = (
            f"saved ({position.x:.3f}, {position.y:.3f}, {position.z:.3f})"
        )
        label = self.tcp_1_status if endpoint_index == 0 else self.tcp_2_status
        label.configure(text=status)
        ready = all(item is not None for item in self.linear_tcp_endpoints)
        self.generate_tcp_line_button.configure(
            state=tk.NORMAL if ready else tk.DISABLED
        )
        self._update_seam_yaw_status()
        self.quick_teaching_status.set(
            f"Reference TCP {endpoint_index + 1} / {role} saved"
        )
        self.log(
            f"REFERENCE TCP {endpoint_index + 1} / {role} SAVED · "
            f"World XYZ {status} · {reference_path}"
        )

    def quick_capture_teaching_pose(self, pose_name):
        if pose_name not in TEACHING_POSES:
            self.error(f"Unknown quick teaching pose: {pose_name}")
            return
        self.teaching_pose_name.set(TEACHING_POSES[pose_name])
        self.teaching_pose_changed()
        self.quick_teaching_status.set(
            f"Capturing {TEACHING_POSES[pose_name]}..."
        )
        self.capture_initial_state()

    def _update_seam_yaw_status(self, sensed_start=None, sensed_goal=None):
        ref_start = self.linear_tcp_endpoints[0]
        ref_goal = self.linear_tcp_endpoints[1]
        if ref_start is None or ref_goal is None:
            reference = self.seam_teaching_reference or {}
            if ref_start is None and reference.get("weld_start") is not None:
                ref_start = reference["weld_start"][3]
            if ref_goal is None and reference.get("weld_end") is not None:
                ref_goal = reference["weld_end"][3]
        if ref_start is None or ref_goal is None:
            self.reference_yaw_status.set("Reference yaw: --")
            self.reference_length_status.set("Length: --")
            self.sensed_yaw_status.set("Sensed yaw: --")
            self.delta_yaw_status.set("ΔYaw: --")
            return
        try:
            reference_yaw = seam_yaw(ref_start, ref_goal)
            dx = ref_goal.position.x - ref_start.position.x
            dy = ref_goal.position.y - ref_start.position.y
            dz = ref_goal.position.z - ref_start.position.z
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.reference_yaw_status.set(
                f"Reference yaw: {math.degrees(reference_yaw):+.2f}°"
            )
            self.reference_length_status.set(
                f"Length: {length * 1000.0:.1f} mm"
            )
            if sensed_start is None or sensed_goal is None:
                self.sensed_yaw_status.set("Sensed yaw: --")
                self.delta_yaw_status.set("ΔYaw: --")
                return
            sensed_value = seam_yaw(sensed_start, sensed_goal)
            delta = math.atan2(
                math.sin(sensed_value - reference_yaw),
                math.cos(sensed_value - reference_yaw),
            )
            self.sensed_yaw_status.set(
                f"Sensed yaw: {math.degrees(sensed_value):+.2f}°"
            )
            self.delta_yaw_status.set(
                f"ΔYaw: {math.degrees(delta):+.2f}°"
            )
        except ValueError:
            self.reference_yaw_status.set("Reference yaw: invalid")
            self.reference_length_status.set("Length: --")
            self.sensed_yaw_status.set("Sensed yaw: --")
            self.delta_yaw_status.set("ΔYaw: --")

    def load_seam_reference(self):
        default_path = self._seam_reference_yaml_path()
        path = filedialog.askopenfilename(
            title="Load seam reference TCP 1 / TCP 2",
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
            filetypes=(("YAML", "*.yaml *.yml"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            group, poses = load_seam_teaching_reference_yaml(path)
        except (OSError, ValueError, yaml.YAMLError, KeyError) as error:
            self.error(f"Reference YAML load failed: {error}")
            return
        if group != self.planning_group.get():
            self.error(
                f"Reference YAML is for {group}; selected arm is {self.planning_group.get()}"
            )
            return
        missing = [
            name for name in ("weld_start", "weld_end") if name not in poses
        ]
        if missing:
            self.error(
                "Reference YAML must contain TCP1/TCP2: " + ", ".join(missing)
            )
            return
        if self.seam_teaching_reference is None:
            self.seam_teaching_reference = {}
        for index, pose_name in enumerate(("weld_start", "weld_end")):
            pose = copy.deepcopy(poses[pose_name])
            stored = self.taught_robot_poses.get(pose_name)
            if stored is not None:
                self.seam_teaching_reference[pose_name] = (
                    stored[0], stored[1], stored[2], copy.deepcopy(pose)
                )
            else:
                self.seam_teaching_reference[pose_name] = (
                    group, tuple(), tuple(), copy.deepcopy(pose)
                )
            self.linear_tcp_endpoints[index] = copy.deepcopy(pose)
            label = self.tcp_1_status if index == 0 else self.tcp_2_status
            label.configure(
                text=(
                    f"loaded ({pose.position.x:.3f}, {pose.position.y:.3f}, "
                    f"{pose.position.z:.3f})"
                )
            )
        self.generate_tcp_line_button.configure(state=tk.NORMAL)
        self._update_seam_yaw_status()
        self.log(f"Loaded TCP1/TCP2 seam reference from {path}")

    def acquire_two_tcp(self):
        if any(pose is None for pose in self.linear_tcp_endpoints):
            self.error("Teach or load both reference TCP 1 and TCP 2 first")
            return
        try:
            count = int(self.tcp_line_count.get())
        except (ValueError, tk.TclError):
            self.error("TCP line point count is invalid")
            return
        start, end = self.linear_tcp_endpoints
        self.log("Previewing reference seam · TCP 1 / START → TCP 2 / GOAL")
        threading.Thread(
            target=self.node.generate_tcp_line,
            args=(
                copy.deepcopy(start),
                copy.deepcopy(end),
                count,
                self.show_path.get(),
            ),
            daemon=True,
        ).start()

    def replace_with_tcp(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint to replace")
            return
        threading.Thread(
            target=self.node.capture_tcp,
            args=(
                index,
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def apply_captured_tcp(self, pose, replace_index, visible):
        if replace_index is None:
            self.points.append(copy.deepcopy(pose))
            selected_index = len(self.points) - 1
            action = "Appended"
        else:
            self.points[replace_index] = copy.deepcopy(pose)
            selected_index = replace_index
            action = "Replaced"
        self.path_kind = "taught"
        self.weave_source = copy.deepcopy(self.points)
        self.set_points(self.points, selected_index)
        self.node.publish_points(self.points, visible)
        self.log(
            f"{action} current {self.planning_group.get()} TCP · "
            "World 6D pose"
        )

    def reverse_path(self):
        if len(self.points) < 2:
            self.error("Path needs at least two poses")
            return
        self.points.reverse()
        if self.path_kind != "weave":
            self.weave_source = copy.deepcopy(self.points)
        self.publish_edits(0)
        self.log("Reversed seam direction")

    def restore_weave_source(self):
        if not self.weave_source:
            self.error("No source seam is available")
            return
        self.path_kind = "source"
        self.set_points(self.weave_source)
        self.node.publish_points(self.points, self.show_path.get())
        self.log("Restored the seam used before weaving")

    def clear_path(self):
        self.path_kind = "empty"
        self.weave_source = []
        self.weave_base_paths = {"linear": [], "circle": []}
        self.set_points([])
        self.node.publish_points([], self.show_path.get())
        self.log("Cleared taught path")

    def update_speed_label(self, _value=None):
        self.invalidate_approved_plan()
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        self.speed_label.configure(
            text=f"{self.velocity_percent.get():.0f}%"
        )

    def plan_preview(self):
        if not self._selected_robot_connected():
            self.error("Connect the robot and wait for live /joint_states")
            return
        self._send_path(execute_requested=False)

    def execute_approved(self):
        if not self.plan_approved:
            self.error("Plan Preview is required before execution")
            return
        if not self.execution_allowed:
            self.error("Server execution is disabled by launch configuration")
            return
        if not self._selected_robot_connected():
            self.error("Connect the REAL RB robot first")
            return
        self._send_path(execute_requested=True)

    def _send_path(self, execute_requested):
        speed = max(0.01, min(1.0, self.velocity_percent.get() / 100.0))
        planning_group = self.planning_group.get()
        try:
            interpolation_step = (
                float(self.interpolation_step_mm.get()) * 0.001
            )
        except (ValueError, tk.TclError):
            self.error("Cartesian interpolation step is invalid")
            return
        if not 0.0005 <= interpolation_step <= 0.02:
            self.error("Cartesian interpolation step must be 0.5..20 mm")
            return
        threading.Thread(
            target=self.node.submit_cartesian_motion,
            args=(
                copy.deepcopy(self.points),
                speed,
                interpolation_step,
                self.show_path.get(),
                execute_requested,
                execute_requested,
                planning_group,
            ),
            daemon=True,
        ).start()

    def update_control_box_io(self, digital_in, digital_out):
        current = (tuple(digital_in), tuple(digital_out))
        previous = self.previous_control_box_io
        changes = []
        for kind, values, old_values in (
            ("DI", current[0], previous[0] if previous else None),
            ("DO", current[1], previous[1] if previous else None),
        ):
            for port, value in enumerate(values):
                changed = (
                    old_values is not None and value != old_values[port]
                )
                candidate = port in MANUAL_IO_CANDIDATES
                if old_values is not None and value == old_values[port]:
                    continue
                if value:
                    background = "#81c995"
                elif changed:
                    background = "#fdd663"
                elif candidate:
                    background = "#dbeafe"
                else:
                    background = "#eeeeee"
                self.control_box_io_labels[(kind, port)].configure(
                    text=f"{port:02d}\n{'ON' if value else 'OFF'}",
                    bg=background,
                )
                if changed:
                    changes.append(
                        f"{kind}{port}={'ON' if value else 'OFF'}"
                    )
        self.previous_control_box_io = current
        active_inputs = [
            str(index) for index, value in enumerate(current[0]) if value
        ]
        active_outputs = [
            str(index) for index, value in enumerate(current[1]) if value
        ]
        self.control_box_io_status.configure(
            text=(
                f"Active DI: {', '.join(active_inputs) or 'none'} · "
                f"Active DO: {', '.join(active_outputs) or 'none'}"
            )
        )
        if changes:
            self.log("Rainbow control-box I/O changed · " + ", ".join(changes))

    def update_touch_input(self, arm, active):
        previous = self.touch_input_states[arm]
        active = bool(active)
        self.touch_input_states[arm] = active
        if previous is not None and active and not previous:
            self.touch_input_rising_edges[arm] += 1
            self._handle_touch_event(arm, f"{arm.upper()} DI8")

    def _handle_touch_event(self, arm, source):
        if arm != self._selected_arm():
            return
        planning_group = f"{arm}_manipulator"
        if self.automatic_probe_kind is not None:
            kind = self.automatic_probe_kind
            self.root.bell()
            self.pipeline_waiting(
                f"DI8 TOUCH DETECTED · stopping {kind} probe before capture"
            )
            # WeldGuiNode._system_state owns the stop trigger. Starting a
            # second worker here allowed a bounced DI8 edge to capture and
            # launch the return path twice.
            return
        guard = self.node.active_touch_guard
        if guard is not None and guard[0] == arm:
            self.root.bell()
            self.pipeline_waiting(
                f"DI8 TOUCH DETECTED · stopping guarded {guard[1]} motion"
            )
            return
        if not self.touch_sensing_enabled.get():
            return
        self.root.bell()
        self.pipeline_waiting(
            f"TOUCH DETECTED · source={source} · "
            f"capturing {planning_group} TCP"
        )
        threading.Thread(
            target=self.node.capture_touch_pose,
            args=(planning_group, source),
            daemon=True,
        ).start()

    def _persist_seam_touch_yaml(self, planning_group, event_label):
        touch_yaml = self._seam_touch_yaml_path(planning_group)
        try:
            save_seam_touch_yaml(
                touch_yaml,
                planning_group,
                self.seam_axis.get(),
                self.seam_probe_touches,
                self.seam_probe_starts,
                self.seam_probe_stops,
                probe_configuration={
                    "wall_direction": self.wall_probe_axis.get(),
                    "wall_sign": self.wall_probe_sign.get(),
                    "base_direction": self.floor_probe_axis.get(),
                    "base_sign": self.floor_probe_sign.get(),
                    "orientation_mode": self.seam_orientation_mode.get(),
                },
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.error(f"DI8 touch YAML save failed: {error}")
            return None
        self.log(f"DI8 {event_label} YAML SAVED · {touch_yaml}")
        return touch_yaml

    def apply_touch_edge_capture(
        self,
        pose,
        planning_group,
        kind,
        probe_start,
    ):
        """Persist the DI8-edge pose before controlled-stop completion."""
        if kind not in CORNER_TOUCH_NAMES:
            self.error(f"Unknown DI8 edge capture kind: {kind}")
            return
        self.seam_probe_touches[kind] = copy.deepcopy(pose)
        self.seam_probe_starts[kind] = copy.deepcopy(probe_start)
        self.seam_probe_stops[kind] = None
        self._persist_seam_touch_yaml(planning_group, "EDGE CONTACT")

    def apply_touch_capture(
        self,
        pose,
        planning_group,
        source,
        probe_start=None,
        stopped_pose=None,
    ):
        self.last_touch_pose = copy.deepcopy(pose)
        values = self._pose_values(pose)
        self.pipeline_result(
            f"TOUCH TCP CAPTURED · {planning_group} · "
            f"World XYZ=({values[0]:.6f}, {values[1]:.6f}, "
            f"{values[2]:.6f}) m"
        )
        if source.startswith("automatic probe:"):
            kind = source.split(":", 1)[1]
            self.seam_probe_touches[kind] = copy.deepcopy(pose)
            self.seam_probe_starts[kind] = (
                copy.deepcopy(probe_start) if probe_start is not None else None
            )
            self.seam_probe_stops[kind] = (
                copy.deepcopy(stopped_pose) if stopped_pose is not None else None
            )
            touch_yaml = self._persist_seam_touch_yaml(
                planning_group, "STOPPED-POSE UPDATE"
            )
            if touch_yaml is not None:
                endpoint = kind.split("_", 1)[0]
                wall = self.seam_probe_touches.get(f"{endpoint}_wall")
                floor = self.seam_probe_touches.get(f"{endpoint}_floor")
                if wall is not None and floor is not None:
                    delta_x_mm = (
                        wall.position.x - floor.position.x
                    ) * 1000.0
                    delta_y_mm = (
                        wall.position.y - floor.position.y
                    ) * 1000.0
                    self.log(
                        f"{endpoint.upper()} TOUCH PAIR CHECK · "
                        f"wall-floor ΔX={delta_x_mm:+.3f} mm · "
                        f"ΔY={delta_y_mm:+.3f} mm · "
                        f"seam-axis coordinate uses pair mean"
                    )
                    self._publish_touch_geometry_if_ready(endpoint)
            self.automatic_probe_kind = None
            # Contact is complete.  The following motion is a deliberate
            # retract and must not be treated as the same active touch probe.
            self.node.clear_touch_probe()
            completed = [
                name
                for name, value in self.seam_probe_touches.items()
                if value is not None
            ]
            self.corner_touch_status.configure(
                text=(
                    f"DI8 {kind} touch saved · {len(completed)}/4 · "
                    "returning to probe start"
                )
            )
            self.log(f"Automatic DI8 {kind} touch stored")
            if probe_start is not None:
                try:
                    settle_seconds = max(
                        0.2,
                        min(5.0, float(self.touch_settle_seconds.get())),
                    )
                except (ValueError, tk.TclError):
                    settle_seconds = 0.7
                threading.Thread(
                    target=self.node.return_touch_probe,
                    args=(
                        planning_group,
                        copy.deepcopy(stopped_pose or pose),
                        copy.deepcopy(probe_start),
                        max(
                            0.001,
                            min(
                                0.10,
                                float(self.touch_probe_speed_percent.get())
                                / 100.0,
                            ),
                        ),
                        0.001,
                        kind,
                        settle_seconds,
                    ),
                    daemon=True,
                ).start()
            return
        if self.touch_sensing_enabled.get() or source.startswith(
            "manual corner capture:"
        ):
            self._record_corner_touch(pose, source)

    def touch_probe_return_finished(self, success, message, probe_kind):
        if success:
            self.seam_auto_returned_kinds.add(probe_kind)
            completed = [
                name
                for name, value in self.seam_probe_touches.items()
                if value is not None
            ]
            self.corner_touch_status.configure(
                text=(
                    f"Probe returned · {len(completed)}/4 captured: "
                    f"{', '.join(completed) or 'none'}"
                )
            )
            self.pipeline_result("DI8 touch captured and probe start restored")
            self._signal_auto_seam_stage(True, probe_kind)
        else:
            self._signal_auto_seam_stage(False, probe_kind)
            self.error(f"Touch captured, but probe return failed: {message}")

    def confirm_all_do_unlock(self):
        if not self.unlock_all_do_ports.get():
            return
        if not messagebox.askyesno(
            "Unlock all Rainbow DO ports",
            "Unknown outputs may operate gas, inching, ARC, or another "
            "actuator. Allow clicking every DO0..15 port?",
        ):
            self.unlock_all_do_ports.set(False)

    def request_do_toggle(self, port):
        if self.previous_control_box_io is None:
            self.error("Rainbow control-box state is not available")
            return
        if port in self.pending_do_ports:
            return
        candidate = port in MANUAL_IO_CANDIDATES
        if not candidate and not self.unlock_all_do_ports.get():
            self.error(
                f"DO{port} is locked · enable non-candidate DO clicking first"
            )
            return
        current = bool(self.previous_control_box_io[1][port])
        target = not current
        if not messagebox.askyesno(
            f"Toggle Rainbow DO{port}",
            f"Command control-box DO{port}: "
            f"{'ON' if current else 'OFF'} → {'ON' if target else 'OFF'}?\n\n"
            "This is a physical output and may operate connected equipment.",
        ):
            return
        self.pending_do_ports.add(port)
        label = self.control_box_io_labels[("DO", port)]
        label.configure(bg="#fdd663", text=f"{port:02d}\nWAIT")
        self.log(
            f"Rainbow DO{port} command requested · "
            f"{'ON' if target else 'OFF'}"
        )
        threading.Thread(
            target=self.node.set_digital_output,
            args=(port, target),
            daemon=True,
        ).start()

    def candidate_outputs_off(self):
        if not messagebox.askyesno(
            "Force candidate outputs OFF",
            "Command DO4, DO8, DO9, DO10, DO12, and DO13 to OFF?",
        ):
            return
        for port in sorted(MANUAL_IO_CANDIDATES):
            self.pending_do_ports.add(port)
            threading.Thread(
                target=self.node.set_digital_output,
                args=(port, False),
                daemon=True,
            ).start()
        self.log("Rainbow candidate DO all-OFF requested")

    def digital_output_result(self, port, success, message):
        self.pending_do_ports.discard(port)
        prefix = "OK" if success else "REJECTED"
        self.log(f"Rainbow DO{port} {prefix} · {message}")
        if not success and self.previous_control_box_io is not None:
            value = self.previous_control_box_io[1][port]
            self.control_box_io_labels[("DO", port)].configure(
                text=f"{port:02d}\n{'ON' if value else 'OFF'}",
                bg=(
                    "#81c995"
                    if value
                    else (
                        "#dbeafe"
                        if port in MANUAL_IO_CANDIDATES
                        else "#eeeeee"
                    )
                ),
            )

    def begin(self, velocity_scale, execute_requested):
        self.bar["value"] = 0
        self.last_action_phase = ""
        self.plan_button.configure(state=tk.DISABLED)
        self.execute_button.configure(state=tk.DISABLED)
        operation = (
            "EXECUTE exact approved plan"
            if execute_requested
            else "PLAN PREVIEW for RViz"
        )
        self.pipeline_waiting(
            f"{operation} · "
            f"speed={velocity_scale:.0%}"
        )

    def progress(self, value, waypoint, pose, phase):
        self.bar["value"] = value * 100
        position = pose.position
        self.feedback_label.configure(
            text=(
                f"{phase or 'PATH'} · waypoint: {waypoint + 1} · "
                f"progress: {value:.0%} · "
                f"pose: ({position.x:.3f}, {position.y:.3f}, "
                f"{position.z:.3f})"
            ),
        )
        if phase and phase != self.last_action_phase:
            self.last_action_phase = phase
            self.log(f"Sequence phase · {phase}")

    def finish(self, text, was_execution):
        if was_execution and self.automatic_probe_kind is not None:
            kind = self.automatic_probe_kind
            self.automatic_probe_kind = None
            self.node.clear_touch_probe()
            self._signal_auto_seam_stage(False, kind)
            self.error(
                f"{kind} probe reached maximum travel without a DI8 edge"
            )
            return
        self.bar["value"] = 100
        self.plan_button.configure(
            state=(
                tk.NORMAL
                if self.points and self._selected_robot_connected()
                else tk.DISABLED
            )
        )
        self.plan_approved = not was_execution
        self.execute_button.configure(
            state=(
                tk.NORMAL
                if (
                    self.plan_approved
                    and self.execution_allowed
                    and self._selected_robot_connected()
                )
                else tk.DISABLED
            )
        )
        if self.plan_approved:
            self.pipeline_result(
                f"{text} · plan approved; inspect RViz, then execute"
            )
        else:
            self.pipeline_result(text)

    def cancel(self):
        self.node.cancel_active_motion()

    def close(self):
        if self._closing:
            return
        self._closing = True
        if self.hicomm_client is not None:
            self.hicomm_client.stop()
        self.root.quit()
        self.root.destroy()

    def shutdown_ros(self):
        if rclpy.ok():
            rclpy.shutdown()
        self.executor_thread.join(timeout=1.0)
        self.node.destroy_node()

    def check_ros(self):
        self._drain_ui_queue()
        if not rclpy.ok():
            self.root.destroy()
            return
        self.root.after(50, self.check_ros)

    def mainloop(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    gui = WeldActionGui()
    try:
        gui.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        gui.shutdown_ros()
