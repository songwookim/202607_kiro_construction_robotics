"""Direct 1G scan motion: lift, joint approach to the taught scan start, constant-speed line to the end, lift.

Pure geometry and step dictionaries (the existing sequence step schema); no ROS calls.
"""

import copy
from dataclasses import dataclass
import math

from construct_robot.core.cartesian_path_common import (
    named_tcp_linear_waypoints,
    pose_is_valid,
    quaternion_angular_distance,
)
from construct_robot.core.precision_sensing_inspection import MAX_1G_SEAM_TILT_DEG, seam_tilt_deg

SCAN_GROUP = "right_manipulator"
MIN_SCAN_LENGTH_M = 0.005
# Taught scan start/end orientations farther apart than this are reported: the line keeps the start orientation.
MAX_SCAN_ORIENTATION_CHANGE_DEG = 5.0


@dataclass(frozen=True)
class ScanPose:
    """A taught scan pose (teaching YAML): joints and the TCP pose in World."""
    joint_names: tuple
    positions: tuple
    tcp: object


@dataclass(frozen=True)
class ScanStep:
    name: str
    step: dict
    record: bool = False


def _position(pose):
    return (float(pose.position.x), float(pose.position.y), float(pose.position.z))


def lifted(pose, clearance_m):
    """Copy of pose moved up along World +Z."""
    result = copy.deepcopy(pose)
    result.position.z = float(result.position.z) + float(clearance_m)
    return result


def scan_line_pose(start, end):
    """The line end pose: the end position with the start orientation (constant sensor attitude)."""
    result = copy.deepcopy(start.tcp)
    result.position.x, result.position.y, result.position.z = _position(end.tcp)
    return result


def check_scan_line(start, end):
    """Validate the taught line; returns (length_m, tilt_deg, orientation_change_deg)."""
    for name, pose in (("start", start), ("end", end)):
        if not pose_is_valid(pose.tcp):
            raise ValueError(f"scan {name} TCP pose is invalid; re-teach it")
    tilt, length = seam_tilt_deg(_position(start.tcp), _position(end.tcp))
    if length < MIN_SCAN_LENGTH_M:
        raise ValueError(f"scan start and end are only {length * 1000.0:.1f} mm apart")
    if tilt > MAX_1G_SEAM_TILT_DEG:
        raise ValueError(f"scan line is {tilt:.0f}° from horizontal (1G limit {MAX_1G_SEAM_TILT_DEG:.0f}°); "
                         "re-teach I/J along the seam at the same height")
    change = math.degrees(quaternion_angular_distance(start.tcp.orientation, end.tcp.orientation))
    return length, tilt, change


def build_scan_steps(current_tcp, start, end, *, clearance_m, tcp_speed_m_s, velocity_scale, return_pass=False):
    """Steps: lift from the current pose, joint move to the scan start, scan line (recorded), lift.
    return_pass: also scan back from the end to the start (a second, opposite pass over both seam ends, as the
    Hongik 1G passes alternate S->E / E->S); the recording spans both lines."""
    if not 0.0 <= float(clearance_m) <= 0.3:
        raise ValueError("lift clearance must be 0..300 mm")
    if not 0.0001 <= float(tcp_speed_m_s) <= 0.5:
        raise ValueError("scan TCP speed must be 0.1..500 mm/s")
    velocity_scale = max(0.01, min(1.0, float(velocity_scale)))
    check_scan_line(start, end)

    def line(name, a, b, *, speed=0.0, record=False):
        return ScanStep(name, {
            "type": "motion",
            "planning_group": SCAN_GROUP,
            "interpolation_step": 0.002,
            "velocity_scale": velocity_scale,
            "tcp_speed_m_s": float(speed),
            "linear_motion_profile": bool(speed),
            "points": named_tcp_linear_waypoints(a, b),
            "path_kind": f"Scan {name}",
            "touch_guard": False,
        }, record)

    steps = []
    if clearance_m > 0.0:
        steps.append(line("lift", current_tcp, lifted(current_tcp, clearance_m)))
    steps.append(ScanStep("approach start", {
        "type": "named_pose",
        "pose_name": "scan_start",
        "pose_label": "Scan start",
        "planning_group": SCAN_GROUP,
        "joint_names": tuple(start.joint_names),
        "positions": tuple(start.positions),
        "tcp_pose": copy.deepcopy(start.tcp),
        "use_joint_planning": True,
        "velocity_scale": velocity_scale,
        "touch_guard": False,
        "continue_after_touch": False,
    }))
    end_pose = scan_line_pose(start, end)
    steps.append(line("line", start.tcp, end_pose, speed=tcp_speed_m_s, record=True))
    last = end_pose
    if return_pass:
        steps.append(line("return line", end_pose, start.tcp, speed=tcp_speed_m_s, record=True))
        last = start.tcp
    if clearance_m > 0.0:
        steps.append(line("retract", last, lifted(last, clearance_m)))
    return steps


POST_WELD_SCAN = "post_weld_scan"


def post_weld_scan_step(start_path, end_path, *, tcp_speed_m_s, velocity_scale, slot, label="Post-weld scan C→D→C"):
    """Sequence row: the local scan (scan start C -> end D -> C, recording) after a weld; files only, no estimate."""
    if not 0.001 <= float(tcp_speed_m_s) <= 0.2:
        raise ValueError("post-weld scan speed must be 1..200 mm/s")
    return {"type": POST_WELD_SCAN, "parallel_slot": int(slot), "duration": 0.0, "planning_group": SCAN_GROUP,
            "pose_label": label, "start_path": str(start_path), "end_path": str(end_path),
            "tcp_speed_m_s": float(tcp_speed_m_s), "velocity_scale": max(0.01, min(1.0, float(velocity_scale)))}


def pose_move_step(current_tcp, target_tcp, *, velocity_scale, label, keep_orientation=True):
    """A Cartesian TCP line from the current pose to a target pose (sequence 'motion' row). The Fastech DI0
    touch guard is armed: contact stops the motion and the sequence (pass poses put the torch at the plate)."""
    if not pose_is_valid(target_tcp):
        raise ValueError(f"{label}: target pose is invalid")
    if keep_orientation:            # position only: the operator set the torch orientation at the start pose
        target_tcp = copy.deepcopy(target_tcp)
        target_tcp.orientation = copy.deepcopy(current_tcp.orientation)
    return {
        "type": "motion", "parallel_slot": 1, "duration": 0.0,
        "planning_group": SCAN_GROUP, "interpolation_step": 0.002,
        "velocity_scale": max(0.01, min(1.0, float(velocity_scale))), "tcp_speed_m_s": 0.0,
        "points": named_tcp_linear_waypoints(current_tcp, target_tcp),
        "path_kind": label, "pose_label": label,
        "touch_guard": True, "continue_after_touch": False, "allow_initial_touch_motion": False,
    }

