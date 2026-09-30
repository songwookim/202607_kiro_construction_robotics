"""Shared YAML parsing and persistence for TCP teaching, touch and pass references."""
import copy
import hashlib
import math
from pathlib import Path
import tempfile
import time

import yaml
from geometry_msgs.msg import Pose

from construct_robot.core.cartesian_path_common import pose_is_valid
from construct_robot.core.seam_geometry import CORNER_TOUCH_NAMES, _pose_position_tuple

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


def save_initial_state_yaml(path, planning_group, joint_names, positions, tcp, provenance=None):
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
    if provenance:
        document["capture_provenance"] = copy.deepcopy(provenance)
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
            persisted = yaml.load(stream, Loader=yaml.CSafeLoader)
        if persisted != document:
            raise OSError(f"YAML read-back verification failed: {path}")
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_pass_teaching_reference(path, expected_pass):
    """Read one independently saved pass teaching YAML as a reference."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Pass teaching reference is missing: {path}")
    raw = path.read_bytes()
    document = yaml.load(raw.decode("utf-8"), Loader=yaml.CSafeLoader) or {}
    if document.get("schema") != "construct_robot_pass_teaching_v1":
        raise ValueError(f"Unsupported pass teaching schema: {path.name}")
    if int(document.get("pass", 0)) != int(expected_pass):
        raise ValueError(
            f"{path.name} contains Pass {document.get('pass')}, "
            f"expected Pass {expected_pass}"
        )
    entries = document.get("poses")
    if not isinstance(entries, dict):
        raise ValueError(f"{path.name} has no poses mapping")
    poses = {}
    joint_states = {}
    for endpoint, pose_name in (
        ("start_wait", "weld_start_wait"),
        ("start", "weld_start"),
        ("goal_wait", "weld_goal_wait"),
        ("goal", "weld_end"),
    ):
        group, names, positions, tcp = parse_teaching_snapshot_entry(
            pose_name, entries.get(pose_name)
        )
        if group != "right_manipulator":
            raise ValueError(f"{path.name} {pose_name} is not a right-arm pose")
        poses[endpoint] = copy.deepcopy(tcp)
        joint_states[endpoint] = (tuple(names), tuple(positions))
    if math.dist(
        _pose_position_tuple(poses["start"]),
        _pose_position_tuple(poses["goal"]),
    ) < 0.001:
        raise ValueError(f"{path.name} seam is shorter than 1 mm")
    additional_pose_entries = {
        pose_name: copy.deepcopy(entries[pose_name])
        for pose_name in ("robot_start", "weld_wait", "weld_finish")
        if pose_name in entries
    }
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "reference_kind": "saved_pass_teaching",
        "source_reference": document.get("source_reference"),
        "source_reference_sha256": document.get("source_reference_sha256"),
        "source_log": document.get("source_log"),
        "source_log_sha256": document.get("source_log_sha256"),
        "requires_ik": bool(document.get("requires_ik", False)),
        "correction_history": copy.deepcopy(
            document.get("correction_history", [])
            if isinstance(document.get("correction_history", []), list)
            else []
        ),
        **poses,
        "joint_states": joint_states,
        "additional_pose_entries": additional_pose_entries,
    }


def save_seam_touch_yaml(
    path,
    planning_group,
    seam_axis,
    touches,
    starts,
    stopped_poses=None,
    probe_configuration=None,
):
    """Atomically save raw Fastech DI0 contact and probe-start poses for diagnostics."""
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


def parse_teaching_snapshot_entry(pose_name, entry):
    """Validate one entry of a weld feedback log's teaching snapshot the way
    ``load_initial_state_yaml`` validates a standalone per-pose YAML file.

    Raises ``ValueError`` on malformed data.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"{pose_name}: entry must be a mapping")
    planning_group = entry.get("planning_group")
    if planning_group not in ("left_manipulator", "right_manipulator"):
        raise ValueError(
            f"{pose_name}: planning_group must be left_manipulator or "
            "right_manipulator"
        )
    arm = planning_group.removesuffix("_manipulator")
    joint_state = entry.get("joint_state")
    if not isinstance(joint_state, dict):
        raise ValueError(f"{pose_name}: joint_state must be a mapping")
    names = joint_state.get("names")
    positions = joint_state.get("positions_rad")
    if not isinstance(names, list) or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError(f"{pose_name}: joint_state.names must be a list")
    if set(names) != ARM_JOINT_NAMES[arm] or len(names) != 6:
        raise ValueError(
            f"{pose_name}: joint_state.names must contain the six {arm} arm joints"
        )
    if not isinstance(positions, list) or len(positions) != len(names):
        raise ValueError(
            f"{pose_name}: joint_state.positions_rad must match joint_state.names"
        )
    positions = tuple(
        _finite_float(value, f"{pose_name} position")
        for value in positions
    )
    tcp = _pose_from_yaml_dict(entry.get("tcp_pose_world"), f"{pose_name} TCP")
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
        document = yaml.load(stream, Loader=yaml.CSafeLoader)
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
