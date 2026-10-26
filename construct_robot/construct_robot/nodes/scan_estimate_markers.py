"""RViz view of a direct / local scan and its Hongik estimate (display only, never commands).

MarkerArray /hongik_scanner/estimate_markers (World): seam ends S / E, per-pass START/GOAL points and connecting TCP / wire-tip lines.
Points, lines and TCP z-axis (torch direction) arrows are displayed without labels. PointCloud2
/hongik_scanner/scan_cloud: the scan's World points. Both latched (transient local).

    python3 -m construct_robot.nodes.scan_estimate_markers <scan_dir>     (publishes until Ctrl+C)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from geometry_msgs.msg import Point
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

MARKER_TOPIC = "/hongik_scanner/estimate_markers"
CLOUD_TOPIC = "/hongik_scanner/scan_cloud"
FRAME = "World"
PASS_COLORS = ((1.0, 0.3, 0.2), (0.2, 0.9, 0.3), (0.3, 0.5, 1.0), (1.0, 0.8, 0.1), (0.9, 0.3, 0.9))
MAX_CLOUD_POINTS = 300_000


def _find(doc, key):
    """First value of key anywhere in a nested YAML document."""
    if isinstance(doc, dict):
        if key in doc:
            return doc[key]
        for value in doc.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(doc, list):
        for value in doc:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _z_axis(q):
    x, y, z, w = (float(q[k]) for k in "xyzw")
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def load_estimate(scan_dir):
    """Ends and passes of <scan_dir>/hongik_estimate (estimate.json, kiro/pass_<k>[_info].yaml)."""
    out = Path(scan_dir).expanduser() / "hongik_estimate"
    ends = {}
    if (out / "estimate.json").is_file():
        for name, end in (json.loads((out / "estimate.json").read_text()).get("ends") or {}).items():
            if end:
                ends[name] = {"point": np.asarray(end["point_m"], float), "two_sided": bool(end.get("two_sided")),
                              "sigma_m": end.get("sigma_m")}
    passes = []
    for path in sorted((out / "kiro").glob("pass_[0-9]*.yaml")):
        if path.stem.endswith("_info"):
            continue
        doc = yaml.safe_load(path.read_text())
        info_path = path.with_name(f"{path.stem}_info.yaml")
        info = yaml.safe_load(info_path.read_text()) if info_path.is_file() else {}
        aim = _find(info, "aim_point_World_m") or {}
        row = {"k": int(doc["pass"])}
        for end in ("start", "goal"):
            pose = doc[f"current_{end}"]
            row[end] = np.array([pose["position_m"][a] for a in "xyz"], float)
            row[f"{end}_z"] = _z_axis(pose["orientation_xyzw"])
            row[f"aim_{end}"] = np.asarray(aim[end], float) if end in aim else None
        passes.append(row)
    return ends, passes


def _marker(ns, mid, kind, color, scale, stamp):
    m = Marker()
    m.header.frame_id, m.header.stamp = FRAME, stamp
    m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = scale
    m.color.r, m.color.g, m.color.b, m.color.a = (*color, 1.0)
    return m


def _pt(v):
    return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def build_markers(ends, passes, stamp, corrected_lines=(), reference_points=()):
    """Display estimated geometry only; correction/reference inputs remain API-compatible but hidden."""
    markers = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)
    for i, (name, end) in enumerate(sorted(ends.items())):
        color = (0.1, 0.9, 0.1) if name == "S" else (0.95, 0.1, 0.1)
        sphere = _marker("ends", i, Marker.SPHERE, color, (0.006, 0.006, 0.006), stamp)
        sphere.pose.position = _pt(end["point"])
        markers.markers.append(sphere)
    for row in passes:
        k, color = row["k"], PASS_COLORS[(row["k"] - 1) % len(PASS_COLORS)]
        line = _marker("tcp_lines", k, Marker.LINE_STRIP, color, (0.0015, 0.0, 0.0), stamp)
        line.points = [_pt(row["start"]), _pt(row["goal"])]
        markers.markers.append(line)
        if row["aim_start"] is not None and row["aim_goal"] is not None:
            aim = _marker("wire_tip_lines", k, Marker.LINE_STRIP, tuple(0.5 + 0.5 * c for c in color), (0.0008, 0.0, 0.0), stamp)
            aim.points = [_pt(row["aim_start"]), _pt(row["aim_goal"])]
            markers.markers.append(aim)
        for j, end in enumerate(("start", "goal")):
            point = _marker("tcp_endpoints", 10 * k + j, Marker.SPHERE, color,
                            (0.004, 0.004, 0.004), stamp)
            point.pose.position = _pt(row[end])
            markers.markers.append(point)
            arrow = _marker("tcp_axes", 10 * k + j, Marker.ARROW, color,
                            (0.00075, 0.0015, 0.002), stamp)
            tip = row.get(f"aim_{end}")
            if tip is None:
                tip = row[end] + 0.03 * row[f"{end}_z"]
            arrow.points = [_pt(row[end]), _pt(tip)]
            markers.markers.append(arrow)
    return markers


def scan_cloud(scan_dir, stamp):
    """The scan's World points; re-based like its estimate when that used another calib (calib_used.json)."""
    scan_dir = Path(scan_dir).expanduser()
    data = np.load(scan_dir / "scan.npz")
    points = np.asarray(data["points_world_m"], np.float32).reshape(-1, 3)
    index = np.asarray(data["points_frame_index"]) if "points_frame_index" in data.files else None
    step = int(math.ceil(len(points) / MAX_CLOUD_POINTS)) if len(points) > MAX_CLOUD_POINTS else 1
    points = points[::step]
    used_path = scan_dir / "hongik_estimate" / "calib_used.json"
    if index is not None and used_path.is_file():
        used = json.loads(used_path.read_text())
        if used.get("rebased") and used.get("recorded") and used.get("used"):
            from construct_robot.core.scan_calib import calib_matrix, rebase_optical, rebase_points
            T_old = np.asarray(data["T_world_optical"], float)
            T_new = rebase_optical(T_old, calib_matrix(used["recorded"]["xyz"], used["recorded"]["rpy"]),
                                   calib_matrix(used["used"]["xyz"], used["used"]["rpy"]))
            points = rebase_points(points, index[::step], T_old, T_new).astype(np.float32)
    msg = PointCloud2()
    msg.header.frame_id, msg.header.stamp = FRAME, stamp
    msg.height, msg.width = 1, len(points)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 12, 12 * len(points), True
    msg.data = points.tobytes()
    return msg


class EstimateViz:
    def __init__(self, node):
        self.node = node
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.markers = node.create_publisher(MarkerArray, MARKER_TOPIC, qos)
        self.cloud = node.create_publisher(PointCloud2, CLOUD_TOPIC, qos)

    def show(self, scan_dir, corrected_lines=(), reference_points=()):
        """Publish the scan cloud and the estimate markers; returns (n ends, n passes)."""
        stamp = self.node.get_clock().now().to_msg()
        ends, passes = load_estimate(scan_dir)
        self.cloud.publish(scan_cloud(scan_dir, stamp))
        self.markers.publish(build_markers(ends, passes, stamp, corrected_lines, reference_points))
        return len(ends), len(passes)


def main(argv=None):
    import rclpy
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    rclpy.init()
    node = rclpy.create_node("scan_estimate_markers")
    ends, passes = EstimateViz(node).show(argv[0])
    print(f"published {ends} end(s), {passes} pass(es) on {MARKER_TOPIC} and the scan on {CLOUD_TOPIC} (Ctrl+C to stop)")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
