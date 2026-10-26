"""Read-only numerical inspection of Hongik wire-tip and TCP poses."""

from dataclasses import dataclass
import math
from types import SimpleNamespace

from construct_robot.core.cartesian_path_common import _quaternion_rotate_vector, quaternion_angular_distance


def finite_values(values, count):
    values = tuple(float(value) for value in values)
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected {count} finite numbers")
    return values


@dataclass(frozen=True)
class SensingPose:
    pass_number: int
    point: str
    kind: str
    frame: str
    position: tuple
    quaternion: tuple

    def __post_init__(self):
        if not self.frame or self.frame.startswith("/") or any(c.isspace() for c in self.frame):
            raise ValueError("A valid TF parent frame is required")
        finite_values(self.position, 3)
        q = finite_values(self.quaternion, 4)
        if abs(math.sqrt(sum(v * v for v in q)) - 1.0) > 0.001:
            raise ValueError("Quaternion must have unit norm (within 0.001)")


def pose_from_mapping(data, *, pass_number, point, kind, frame):
    position_key, orientation_key = (
        ("position", "orientation") if kind == "wire" else
        ("position_m", "orientation_xyzw")
    )
    try:
        return SensingPose(
            pass_number, str(point), kind, frame,
            finite_values((data[position_key][key] for key in "xyz"), 3),
            finite_values((data[orientation_key][key] for key in "xyzw"), 4),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Pass {pass_number} {kind} {point}: {error}") from error


def pose_difference(pose, reference):
    """Compare only like pose conventions in the same parent frame."""
    if pose.frame != reference.frame or pose.kind != reference.kind:
        raise ValueError("Reference must use the same frame and wire/TCP convention")
    distance_mm = math.dist(pose.position, reference.position) * 1000.0
    qa, qb = (SimpleNamespace(**dict(zip("xyzw", p.quaternion))) for p in (pose, reference))
    angle_deg = math.degrees(quaternion_angular_distance(qa, qb))
    return distance_mm, angle_deg


def point_on_tool(position, orientation, offset):
    """Position of a point fixed in a tool frame: position + R(orientation)·offset (all metres)."""
    delta = _quaternion_rotate_vector(orientation, offset)
    return tuple(float(p) + d for p, d in zip(position, delta))


# The Hongik 1G server refuses rough points whose line is steeper than this (World z up).
MAX_1G_SEAM_TILT_DEG = 45.0


def seam_tilt_deg(a, b):
    """Angle of the line A->B from the World horizontal plane [deg] and its length [m]."""
    dx, dy, dz = (float(q) - float(p) for p, q in zip(a, b))
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 0.0:
        raise ValueError("A and B must be distinct")
    return math.degrees(math.atan2(abs(dz), math.hypot(dx, dy))), length


def tcp_wire_error_mm(tcp, wire, offset_mm):
    if tcp.kind != "tcp" or wire.kind != "wire" or tcp.frame != wire.frame:
        raise ValueError("TCP/wire consistency needs matching parent frames")
    offset = finite_values((offset_mm,), 1)[0]
    if offset < 0:
        raise ValueError("TCP-to-wire offset must be non-negative")
    orientation = SimpleNamespace(**dict(zip("xyzw", tcp.quaternion)))
    delta = _quaternion_rotate_vector(orientation, (0.0, 0.0, offset / 1000.0))
    aim = tuple(p + d for p, d in zip(tcp.position, delta))
    return math.dist(aim, wire.position) * 1000.0


def inspect_paths(document, frame):
    """Validate action data, preserving the returned wire-tip convention."""
    code = int(document["error_code"])
    notes = [f"error_code={code}: {document.get('error_message', '')}"]
    if code != 0:
        return (), notes + ["Failed result: no poses published."]
    paths = document.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("Successful result contains no paths")
    poses = []
    for k, path in enumerate(paths, 1):
        if path.get("start") is not None and path.get("goal") is not None:
            waypoints = [path["start"], path["goal"]]      # current interface: the pass's two ends only
        else:
            waypoints = path.get("waypoints", [])          # earlier results: the dense wire-tip line
        if len(waypoints) < 2:
            raise ValueError(f"Pass {k} needs a start and a goal")
        rows = [pose_from_mapping(p, pass_number=k, point=i, kind="wire", frame=frame)
                for i, p in enumerate(waypoints)]
        steps = [math.dist(a.position, b.position) * 1000.0
                 for a, b in zip(rows, rows[1:])]
        notes.append(f"Pass {k}: {len(rows)} points, length={sum(steps):.3f} mm, "
                     f"max step={max(steps):.3f} mm" +
                     (" · CHECK: spacing exceeds 5 mm" if len(rows) > 2 and max(steps) > 5.001 else ""))
        if sum(steps) <= 1e-6:
            raise ValueError(f"Pass {k} has zero length")
        poses.extend(rows)
    return tuple(poses), notes
