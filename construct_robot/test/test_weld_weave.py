import math
import pytest
from geometry_msgs.msg import Pose
from construct_robot.cartesian_path_common import weave_cycles_for_pitch
from construct_robot.weld_action_gui import (
    weld_weave_geometry, weave_path_speed_m_s, validated_seam_speed_factor,
)


def seam():
    start, goal = Pose(), Pose()
    start.orientation.w = goal.orientation.w = 1.0
    goal.position.x = 0.02
    return start, goal


def test_sine_amplitude_pitch_holds_and_fixed_attitude():
    start, goal = seam()
    points, holds, cycles, actual_pitch = weld_weave_geometry(
        start, goal, "sine", 3.0, 5.0, "world_y", 0.2, 0.3)
    assert cycles == 4 and actual_pitch == pytest.approx(5.0)
    assert len(points) == len(holds) == 49
    assert points[0] == start and points[-1] == goal
    assert [i for i, hold in enumerate(holds) if hold] == [3, 9, 15, 21, 27, 33, 39, 45]
    assert [holds[i] for i in (3, 9, 15, 21)] == [0.2, 0.3, 0.2, 0.3]
    assert max(p.position.y for p in points) == pytest.approx(0.003)
    assert min(p.position.y for p in points) == pytest.approx(-0.003)
    assert all(p.position.z == 0 and p.orientation == start.orientation for p in points)
    # Each cycle begins/ends on the seam at evenly spaced 5 mm intervals.
    assert [points[i].position.x for i in range(0, len(points), 12)] == pytest.approx(
        [0, .005, .010, .015, .020]
    )
    assert [points[i].position.y for i in range(0, len(points), 12)] == pytest.approx([0]*5)
    # The transverse slope approaches zero at the turning points.
    assert points[2].position.y < points[3].position.y
    assert points[4].position.y < points[3].position.y


def test_sensed_direction_and_diagonal_seam():
    start, goal = seam()
    goal.position.y = 0.02
    points, _, _, _ = weld_weave_geometry(start, goal, "sine", 3, 10, "tool_y")
    assert (points[3].position.y - points[3].position.x) / math.sqrt(2) == pytest.approx(0.003)
    points, _, _, _ = weld_weave_geometry(start, goal, "sine", 3, 10, "tool_y", transverse_vector=(0, 0, 1))
    assert points[3].position.z == pytest.approx(0.003)


@pytest.mark.parametrize("amplitude,pitch,left,right", [(0,5,0,0), (3,0,0,0), (3,5,-1,0), (3,5,0,11), (float('nan'),5,0,0)])
def test_invalid_settings(amplitude, pitch, left, right):
    with pytest.raises(ValueError):
        weld_weave_geometry(*seam(), "sine", amplitude, pitch, "tool_y", left, right)


def test_pitch_rounds_to_whole_cycles_without_exceeding_requested_pitch():
    start, goal = seam()
    points, holds, cycles, actual_pitch = weld_weave_geometry(
        start, goal, "sine", 3, 6, "tool_y")
    assert cycles == 4
    assert actual_pitch == pytest.approx(5)
    assert len(points) == len(holds) == 49
    with pytest.raises(ValueError, match="increase pitch"):
        weave_cycles_for_pitch(.02, .1)


def test_circle_uses_pitch_and_rejects_peak_dwell():
    start, goal = seam()
    points, holds, cycles, actual_pitch = weld_weave_geometry(
        start, goal, "circle", 3, 5, "tool_y")
    assert cycles == 4 and actual_pitch == pytest.approx(5)
    assert points[0] == start and points[-1] == goal
    assert not any(holds)
    with pytest.raises(ValueError, match="no left/right peaks"):
        weld_weave_geometry(start, goal, "circle", 3, 5, "tool_y", .2, 0)


def test_dwell_cannot_exceed_target_travel_time():
    with pytest.raises(ValueError, match="dwell exceeds"):
        weave_path_speed_m_s(.02, .04, 10, [1.1, 1.0])


def test_straight_seam_roundoff_does_not_abort_arc_off_watcher():
    assert validated_seam_speed_factor(1.0000000000000002) == 1.0
    assert validated_seam_speed_factor(0.5) == 0.5
    with pytest.raises(ValueError, match="invalid path/seam"):
        validated_seam_speed_factor(1.001)


def test_hold_plan_is_continuous_and_stationary_during_dwell():
    from types import SimpleNamespace
    from construct_msgs.action import CartesianPath
    from moveit_msgs.srv import GetCartesianPath
    from trajectory_msgs.msg import JointTrajectoryPoint
    from construct_robot.cartesian_path_server import CartesianPathActionServer

    states = []

    def plan(request, start_state=None, publish=False):
        assert not request.waypoint_hold_s and not request.linear_motion_profile
        states.append(start_state)
        response = GetCartesianPath.Response()
        response.fraction = 1.0
        response.start_state.joint_state.name = ["joint"]
        response.start_state.joint_state.position = [0.0]
        response.solution.joint_trajectory.joint_names = ["joint"]
        for second, position in ((0, len(states)-1), (1, len(states))):
            point = JointTrajectoryPoint()
            point.positions = [float(position)]
            point.velocities = [0.0]
            point.accelerations = [0.0]
            point.time_from_start.sec = second
            response.solution.joint_trajectory.points.append(point)
        return response

    start, goal = seam()
    request = CartesianPath.Goal()
    request.waypoints = [start, goal, goal]
    request.waypoint_hold_s = [0.0, 0.2, 0.3]
    server = SimpleNamespace(plan_with_moveit=plan)
    result = CartesianPathActionServer.plan_with_holds(server, request, publish=False)
    points = result.solution.joint_trajectory.points
    times = [p.time_from_start.sec + p.time_from_start.nanosec*1e-9 for p in points]
    assert times == pytest.approx([0, 1, 1.2, 2.2, 2.5])
    assert list(states[1].joint_state.position) == [1.0]
    assert points[1].positions == points[2].positions
    assert points[3].positions == points[4].positions
    assert all(not any(p.velocities) for p in points)


def test_builder_edit_regenerates_sine_and_aligns_holds():
    from construct_robot.weld_action_gui import update_weld_scenario_motion_values
    start, goal = seam()
    points, _, _, _ = weld_weave_geometry(start, goal, "sine", 3, 5, "tool_y", .2, .3)
    motion = dict(type="motion", weld_scenario_stage="weld_motion", weld_scenario_id="cap",
                  weld_weave_enabled=True, weld_weave_pattern="sine", points=points,
                  usable_seam_start=start, usable_seam_goal=goal,
                  weld_weave_amplitude_mm=3., weld_weave_pitch_mm=5.,
                  weld_weave_left_dwell_s=.2, weld_weave_right_dwell_s=.3)
    off = dict(type="digital_weld", weld_scenario_stage="arc_off", weld_scenario_id="cap")
    result = update_weld_scenario_motion_values([motion, off], 0,
                                               tcp_speed_mm_s=5, lead_in_mm=0, lead_out_mm=6)
    edited = result[0]
    assert len(edited["points"]) == len(edited["waypoint_hold_s"]) == 50
    assert edited["waypoint_hold_s"][-1] == 0
    assert not edited["linear_motion_profile"]
    assert result[1]["tcp_speed_m_s"] == edited["tcp_speed_m_s"]
    path_length = sum(math.dist(
        (a.position.x, a.position.y, a.position.z),
        (b.position.x, b.position.y, b.position.z))
        for a, b in zip(edited["usable_weld_points"][:-1], edited["usable_weld_points"][1:]))
    total_s = path_length / edited["tcp_speed_m_s"] + sum(edited["waypoint_hold_s"])
    assert .02 / total_s == pytest.approx(.005)


def test_zero_dwell_sine_uses_one_continuous_smoothed_path():
    from construct_robot.weld_action_gui import update_weld_scenario_motion_values
    start, goal = seam()
    points, holds, _, _ = weld_weave_geometry(start, goal, "sine", 3, 5, "tool_y")
    assert not any(holds)
    motion = dict(type="motion", weld_scenario_stage="weld_motion", weld_scenario_id="cap",
                  weld_weave_enabled=True, weld_weave_pattern="sine", points=points,
                  usable_seam_start=start, usable_seam_goal=goal,
                  weld_weave_amplitude_mm=3., weld_weave_pitch_mm=5.,
                  weld_weave_left_dwell_s=0., weld_weave_right_dwell_s=0.)
    off = dict(type="digital_weld", weld_scenario_stage="arc_off", weld_scenario_id="cap")
    updated = update_weld_scenario_motion_values([motion, off], 0, tcp_speed_mm_s=3.5,
                                                  lead_in_mm=0, lead_out_mm=0)
    assert "waypoint_hold_s" not in updated[0]
    assert updated[0]["linear_motion_profile"] is True


def test_sine_speed_means_seam_progress_not_wavy_tcp_distance():
    from construct_robot.cartesian_path_common import weaving_from_path
    from construct_robot.weld_action_gui import update_weld_scenario_motion_values
    start, goal = seam()
    points = weaving_from_path((start, goal), .003, 4, 12, transverse_vector=(0, 1, 0))
    motion = dict(type="motion", weld_scenario_stage="weld_motion", weld_scenario_id="sine",
                  weld_weave_enabled=True, weld_weave_pattern="sine", points=points,
                  usable_weld_points=points, usable_seam_start=start, usable_seam_goal=goal)
    off = dict(type="digital_weld", weld_scenario_stage="arc_off", weld_scenario_id="sine")
    updated = update_weld_scenario_motion_values([motion, off], 0, tcp_speed_mm_s=3.5,
                                                  lead_in_mm=0, lead_out_mm=0)
    factor = updated[0]["path_to_seam_speed_factor"]
    assert 0 < factor < 1
    assert updated[0]["tcp_speed_m_s"] * factor == pytest.approx(.0035)
    assert updated[1]["path_to_seam_speed_factor"] == pytest.approx(factor)


def test_pass_through_lead_is_not_planned_as_a_tiny_separate_leg():
    from types import SimpleNamespace
    from construct_msgs.action import CartesianPath
    from moveit_msgs.srv import GetCartesianPath
    from trajectory_msgs.msg import JointTrajectoryPoint
    from construct_robot.cartesian_path_server import CartesianPathActionServer
    calls = []

    def plan(request, start_state=None, publish=False):
        calls.append(list(request.waypoints))
        assert len(request.waypoints) == 3
        response = GetCartesianPath.Response()
        response.fraction = 1.0
        response.start_state.joint_state.name = ["joint"]
        response.start_state.joint_state.position = [0.0]
        response.solution.joint_trajectory.joint_names = ["joint"]
        for second in (0, 1):
            point = JointTrajectoryPoint()
            point.positions = [float(second)]
            point.time_from_start.sec = second
            response.solution.joint_trajectory.points.append(point)
        return response

    start, goal = seam()
    request = CartesianPath.Goal()
    request.waypoints = [start, start, goal, goal, goal]
    request.waypoint_hold_s = [0.0, 0.0, 0.2, 0.0, 0.0]
    server = SimpleNamespace(plan_with_moveit=plan)
    response = CartesianPathActionServer.plan_with_holds(server, request, publish=False)
    assert len(calls) == 2
    assert len(response.solution.joint_trajectory.points) == 4


# --------------------------------------------------------------------------
# preview / execution agreement on the weave plane
# --------------------------------------------------------------------------
def _sensed_fillet_geometry():
    """A touch-corrected fillet joint: wall plane + floor plane, 4 deg tilt."""
    from construct_robot.weld_action_gui import (
        compute_corrected_seam_geometry, compute_surface_plane,
    )

    taught_start, taught_goal = Pose(), Pose()
    taught_start.orientation.w = taught_goal.orientation.w = 1.0
    taught_goal.position.x = 0.186

    tilt = math.tan(math.radians(4.0))
    wall = compute_surface_plane(
        (0.0, 1.0, 0.0),
        [(0.03, 0.003, 0.02), (0.15, 0.003 + 0.12 * tilt, 0.02)])
    floor = compute_surface_plane(
        (0.0, 0.0, 1.0), [(0.03, 0.03, -0.002), (0.15, 0.05, -0.002)])
    return compute_corrected_seam_geometry(taught_start, taught_goal,
                                           wall, floor)


def _gui_with(geometry, touches):
    from construct_robot.weld_action_gui import WeldActionGui

    gui = object.__new__(WeldActionGui)
    gui.corrected_seam_geometry = geometry
    gui.seam_probe_touches = touches
    return gui


def test_sensed_weave_direction_is_the_geometry_axis_not_a_tool_axis():
    """The weave plane of a touch-corrected seam comes from the sensed planes.

    e_w lies in the wall/floor bisecting plane, which on a fillet joint is 45
    degrees away from any tool/world axis.  Previewing the weave about a
    generic axis and welding it about e_w puts the peaks millimetres apart, so
    both paths have to read this one accessor.
    """
    geometry = _sensed_fillet_geometry()
    gui = _gui_with(geometry, dict.fromkeys(
        ("start_wall", "start_floor", "goal_wall", "goal_floor"), object()))

    assert gui.sensed_weave_transverse_vector() == geometry.e_w

    generic = (0.0, 1.0, 0.0)
    tangent = geometry.d_real
    projected = [generic[i] - sum(a * b for a, b in zip(generic, tangent)) * tangent[i]
                 for i in range(3)]
    norm = math.sqrt(sum(v * v for v in projected))
    projected = [v / norm for v in projected]
    angle = math.degrees(math.acos(
        abs(sum(a * b for a, b in zip(projected, geometry.e_w)))))
    assert angle == pytest.approx(45.0, abs=1.0)


@pytest.mark.parametrize("touches, expected_none", [
    ({"start_wall": 1, "start_floor": 1, "goal_wall": 1, "goal_floor": 1}, False),
    ({"start_wall": 1, "start_floor": 1, "goal_wall": None, "goal_floor": None}, False),
    ({"start_wall": 1, "start_floor": None, "goal_wall": 1, "goal_floor": None}, True),
    ({"start_wall": None, "start_floor": None, "goal_wall": None, "goal_floor": None}, True),
])
def test_sensed_weave_direction_requires_a_complete_endpoint(touches, expected_none):
    """One endpoint needs *both* surfaces before its geometry means anything."""
    gui = _gui_with(_sensed_fillet_geometry(), touches)
    assert (gui.sensed_weave_transverse_vector() is None) is expected_none


def test_sensed_weave_direction_is_dropped_without_corrected_geometry():
    gui = _gui_with(None, dict.fromkeys(
        ("start_wall", "start_floor", "goal_wall", "goal_floor"), object()))
    assert gui.sensed_weave_transverse_vector() is None


def test_weave_preview_accepts_the_sensed_direction():
    """The preview entry point must be able to weave about e_w.

    It previously had no transverse_vector parameter at all, so the preview
    could only ever draw a generic-axis weave while the sequence builder ran
    the sensed one.
    """
    import inspect

    from construct_robot.weld_action_gui import WeldGuiNode

    signature = inspect.signature(WeldGuiNode.generate_weave)
    assert "transverse_vector" in signature.parameters


def test_sensed_and_generic_weaves_actually_differ():
    """Guards the claim above with numbers rather than trust."""
    from construct_robot.cartesian_path_common import (
        linear_pose_waypoints, weaving_from_path,
    )

    geometry = _sensed_fillet_geometry()
    base = linear_pose_waypoints(geometry.start, geometry.goal, 2)
    amplitude = 0.002

    generic = weaving_from_path(base, amplitude, 7, 12, "world_y")
    sensed = weaving_from_path(base, amplitude, 7, 12, "world_y",
                               tuple(geometry.e_w))
    worst = max(
        math.dist(
            (a.position.x, a.position.y, a.position.z),
            (b.position.x, b.position.y, b.position.z),
        )
        for a, b in zip(generic, sensed)
    )
    # Nearly twice the amplitude itself -- not a rounding difference.
    assert worst * 1000 > 3.0
