"""Scan previews show torch direction while hiding reference overlays and labels."""
import numpy as np
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker
from construct_robot.nodes.scan_estimate_markers import build_markers


def test_scan_preview_has_torch_arrows_without_reference_overlays():
    start = np.array([0.1, 0.2, 0.3])
    goal = start + [0.0, 0.1, 0.0]
    result = build_markers(
        {"S": {"point": start}, "E": {"point": goal}},
        [{"k": 1, "start": start, "goal": goal,
          "aim_start": start + [0.002, 0., -0.04], "aim_goal": None,
          "start_z": np.array([0., 0., -1.]),
          "goal_z": np.array([0., 0., -1.])}], Time(),
        corrected_lines=[(1, start, goal)], reference_points=[("truth", start)],
    )
    assert result.markers[0].action == Marker.DELETEALL
    visible = result.markers[1:]
    assert all(m.type in (Marker.SPHERE, Marker.LINE_STRIP, Marker.ARROW) for m in visible)
    arrows = [m for m in visible if m.ns == "tcp_axes"]
    assert len(arrows) == 2
    assert np.isclose(arrows[0].points[1].x, start[0] + 0.002)
    assert np.isclose(arrows[0].points[1].z, start[2] - 0.04)
    assert np.isclose(arrows[1].points[1].z, goal[2] - 0.03)
    endpoints = [m for m in visible if m.ns == "tcp_endpoints"]
    assert len(endpoints) == 2
    assert endpoints[0].pose.position.x == start[0]
    assert endpoints[1].pose.position.y == goal[1]
    assert {m.ns for m in visible} == {
        "ends", "tcp_endpoints", "tcp_lines", "tcp_axes",
    }
