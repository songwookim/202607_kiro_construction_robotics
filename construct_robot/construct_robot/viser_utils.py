import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import numpy as np


def default_urdf_path() -> Path:
    """Return the installed KIRO URDF path."""
    return (
        Path(get_package_share_directory("construct_description"))
        / "urdf_0528"
        / "construct_robot_0528.urdf"
    )


def resolve_ros_resource(fname: str) -> str:
    """Resolve a ROS package URI for mesh loaders outside the ROS ecosystem."""
    prefix = "package://"
    if not fname.startswith(prefix):
        return fname
    package_path = fname[len(prefix):]
    package_name, separator, relative_path = package_path.partition("/")
    if not separator:
        raise ValueError(f"Invalid ROS package URI: {fname}")
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def xyzw_to_wxyz(quaternion) -> np.ndarray:
    """Convert and normalize a geometry_msgs-style quaternion for Viser."""
    wxyz = np.array(
        [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
        dtype=np.float64,
    )
    norm = np.linalg.norm(wxyz)
    if not math.isfinite(norm) or norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return wxyz / norm


def merge_joint_positions(
    base_configuration,
    joint_index,
    names,
    positions,
) -> np.ndarray:
    """Merge a possibly partial ROS joint vector into a full configuration."""
    configuration = np.asarray(base_configuration, dtype=np.float64).copy()
    for name, position in zip(names, positions):
        index = joint_index.get(name)
        if index is not None and math.isfinite(position):
            configuration[index] = position
    return configuration


def trajectory_time(point) -> float:
    """Return a JointTrajectoryPoint-like time_from_start in seconds."""
    return (
        float(point.time_from_start.sec)
        + float(point.time_from_start.nanosec) * 1e-9
    )
