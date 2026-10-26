"""Joint spline limits and TCP time scaling without robot connections."""

import copy
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
from rcl_interfaces.msg import ParameterValue
from trajectory_msgs.msg import JointTrajectoryPoint

from construct_robot.core.cartesian_path_common import (
    retime_trajectory_constant_velocity, tcp_progress, trajectory_duration_seconds,
)
from construct_robot.core.trajectory_limits import (
    joint_spline_peaks, limit_trajectory_derivatives,
)
from construct_robot.nodes.cartesian_path_server import CartesianPathActionServer


def trajectory(positions, times, names=("joint",)):
    result = RobotTrajectory()
    result.joint_trajectory.joint_names = list(names)
    for values, elapsed in zip(positions, times):
        point = JointTrajectoryPoint()
        point.positions = list(values)
        point.velocities = [0.0] * len(names)
        point.accelerations = [0.0] * len(names)
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(round(elapsed * 1e9), 1_000_000_000)
        result.joint_trajectory.points.append(point)
    return result


def limits(velocity=10.0, acceleration=100.0, jerk=1000.0):
    return {"joint": dict(velocity=velocity, acceleration=acceleration, jerk=jerk)}


def test_quintic_extrema_include_peaks_between_zero_derivative_endpoints():
    path = trajectory([[0.0], [1.0]], [0.0, 1.0])
    peaks = joint_spline_peaks(path)["joint"]
    assert peaks == pytest.approx(dict(velocity=1.875, acceleration=10.0 / math.sqrt(3.0), jerk=60.0))
    assert all(point.velocities[0] == point.accelerations[0] == 0.0
               for point in path.joint_trajectory.points)


def test_spline_bounds_use_nonzero_endpoint_velocities_and_accelerations():
    path = trajectory([[0.0], [1.0]], [0.0, 1.0])
    path.joint_trajectory.points[-1].velocities = [3.0]
    path.joint_trajectory.points[-1].accelerations = [6.0]
    assert joint_spline_peaks(path)["joint"] == pytest.approx(dict(velocity=3.0, acceleration=6.0, jerk=6.0))


@pytest.mark.parametrize("derivative,factor", [("velocity", 2), ("acceleration", 3), ("jerk", 4)])
def test_whole_path_slowdown_enforces_each_derivative_limit(derivative, factor):
    path = trajectory([[0.0], [1.0]], [0.0, 1.0])
    bound = limits()
    peak = joint_spline_peaks(path)["joint"][derivative]
    order = ("velocity", "acceleration", "jerk").index(derivative) + 1
    bound["joint"][derivative] = peak / factor ** order
    positions = [list(point.positions) for point in path.joint_trajectory.points]
    stretch = limit_trajectory_derivatives(path, bound)
    assert stretch == pytest.approx(factor, rel=2e-6)
    assert trajectory_duration_seconds(path) == pytest.approx(stretch, abs=1e-9)
    assert [list(point.positions) for point in path.joint_trajectory.points] == positions
    assert all(value <= bound["joint"][name] * (1.0 + 1e-9)
               for name, value in joint_spline_peaks(path)["joint"].items())


def test_safe_spline_is_unchanged_and_joint_limits_follow_names():
    path = trajectory([[0.0, 0.0], [0.0, 1.0]], [0.0, 1.0], ("still", "moving"))
    before = copy.deepcopy(path)
    bound = {"moving": limits()["joint"], "still": limits(velocity=1e-6)["joint"]}
    assert limit_trajectory_derivatives(path, bound) == 1.0
    assert path == before
    bound["moving"]["velocity"] = 0.9375
    assert limit_trajectory_derivatives(path, bound) == pytest.approx(2.0, rel=2e-6)


@pytest.mark.parametrize("value", [None, 0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_invalid_limits_cannot_bypass_checks(value):
    bound = limits()
    bound["joint"]["acceleration"] = value
    with pytest.raises(ValueError, match="limit"):
        limit_trajectory_derivatives(trajectory([[0.0], [1.0]], [0.0, 1.0]), bound)


@pytest.mark.parametrize("times", [[0.0, 0.0], [1.0, 0.0], [-1.0, 1.0]])
def test_bad_timestamps_are_rejected(times):
    with pytest.raises(ValueError, match="timestamp"):
        joint_spline_peaks(trajectory([[0.0], [1.0]], times))


def test_nonfinite_or_incomplete_derivative_arrays_are_rejected():
    path = trajectory([[0.0], [1.0]], [0.0, 1.0])
    path.joint_trajectory.points[0].accelerations = []
    with pytest.raises(ValueError, match="finite"):
        joint_spline_peaks(path)
    path.joint_trajectory.points[0].accelerations = [float("nan")]
    with pytest.raises(ValueError, match="finite"):
        joint_spline_peaks(path)


def nonlinear_path():
    return trajectory([[1.0 + index / 100.0] for index in range(101)],
                      [index / 10.0 for index in range(101)])


def cruise_speeds(path):
    points = path.joint_trajectory.points
    times = [point.time_from_start.sec + point.time_from_start.nanosec * 1e-9 for point in points]
    return [(points[index + 1].positions[0] ** 2 - points[index].positions[0] ** 2) /
            (times[index + 1] - times[index]) for index in range(30, 70)]


def test_tcp_cruise_stays_constant_after_acceleration_and_jerk_slowdown():
    path = nonlinear_path()
    progress = tcp_progress(path, lambda joints: (joints[0] ** 2, 0.0, 0.0), ["joint"])
    assert progress[-1] == 3.0
    retime_trajectory_constant_velocity(path, ramp_duration_s=1.0, progress=progress, target_speed=0.2)
    assert cruise_speeds(path) == pytest.approx([0.2] * 40, rel=1e-8)
    original = copy.deepcopy(path)
    bound = limits(velocity=0.01, acceleration=0.001, jerk=0.001)
    stretch = limit_trajectory_derivatives(path, bound)
    assert stretch > 1.0
    assert cruise_speeds(path) == pytest.approx([0.2 / stretch] * 40, rel=1e-7)
    for first, second in zip(original.joint_trajectory.points, path.joint_trajectory.points):
        assert first.positions == second.positions
        assert second.velocities == pytest.approx(np.asarray(first.velocities) / stretch)
        assert second.accelerations == pytest.approx(np.asarray(first.accelerations) / stretch ** 2)


def test_tcp_progress_resolves_joint_order_and_rejects_fictitious_translation():
    path = trajectory([[0.0, 1.0], [0.0, 2.0]], [0.0, 1.0], ("other", "joint"))
    assert tcp_progress(path, lambda joints: (joints[0], 0.0, 0.0), ["joint", "other"]) == [0.0, 1.0]
    with pytest.raises(ValueError, match="rotation-only"):
        tcp_progress(path, lambda joints: (0.0, 0.0, 0.0), ["joint", "other"])
    with pytest.raises(ValueError, match="unique joint"):
        tcp_progress(path, lambda joints: (0.0, 0.0, 0.0), ["joint"])


def server_host(bound=None):
    chain = SimpleNamespace(joint_names=["joint"], velocity_limits=[0.5],
                            fk=lambda joints: (np.array([joints[0] ** 2, 0.0, 0.0]), None))
    host = SimpleNamespace(_arm_chain=lambda group: chain, get_logger=Mock(),
                           _joint_motion_limits=lambda group: bound or limits(),
                           _verify_path_fidelity=Mock(return_value=None))
    host._retime_constant_tcp_speed = lambda *args: CartesianPathActionServer._retime_constant_tcp_speed(host, *args)
    return host


def test_server_retimes_copy_and_reports_actual_ramp_after_slowdown():
    host = server_host(limits(velocity=0.01))
    path = nonlinear_path()
    before = copy.deepcopy(path)
    candidate, stretch, ramp = host._retime_constant_tcp_speed("right_manipulator", path, 1.0, 0.2)
    assert stretch > 1.0 and ramp == stretch
    assert path == before
    assert cruise_speeds(candidate) == pytest.approx([0.2 / stretch] * 40, rel=1e-7)


def test_missing_fk_or_bad_progress_aborts_instead_of_joint_space_fallback():
    host = server_host()
    host._arm_chain = Mock(side_effect=RuntimeError("URDF unavailable"))
    with pytest.raises(RuntimeError, match="URDF unavailable"):
        host._retime_constant_tcp_speed("right_manipulator", nonlinear_path(), 1.0, 0.2)
    host._arm_chain = lambda group: SimpleNamespace(joint_names=["joint"], fk=lambda joints: ([0, 0, 0], None))
    with pytest.raises(RuntimeError, match="rotation-only"):
        host._retime_constant_tcp_speed("right_manipulator", nonlinear_path(), 1.0, 0.2)


def test_active_moveit_limits_are_loaded_cached_and_capped_by_urdf():
    host = server_host()
    values = []
    for maximum in (1.0, 0.5, 2.0):
        values.extend([ParameterValue(type=1, bool_value=True), ParameterValue(type=3, double_value=maximum)])
    client = SimpleNamespace(wait_for_service=lambda **kwargs: True, call_async=Mock(return_value="future"))
    host.create_client = Mock(return_value=client)
    host.destroy_client = Mock()
    host._wait_for_future = lambda *args: SimpleNamespace(values=values)
    bound = CartesianPathActionServer._joint_motion_limits(host, "right_manipulator")
    assert bound == {"joint": dict(velocity=0.5, acceleration=0.5, jerk=2.0)}
    request = client.call_async.call_args[0][0]
    assert request.names[-1] == "robot_description_planning.joint_limits.joint.max_jerk"
    assert CartesianPathActionServer._joint_motion_limits(host, "right_manipulator") is bound
    host.create_client.assert_called_once()
    host.destroy_client.assert_called_once_with(client)


def test_missing_jerk_limit_rejects_planning_and_is_not_cached():
    host = server_host()
    values = []
    for maximum in (1.0, 0.5, 2.0):
        values.extend([ParameterValue(type=1, bool_value=True), ParameterValue(type=3, double_value=maximum)])
    values[-2] = ParameterValue()
    host.create_client = lambda *args, **kwargs: SimpleNamespace(
        wait_for_service=lambda **kwargs: True, call_async=lambda request: None)
    host.destroy_client = Mock()
    host._wait_for_future = lambda *args: SimpleNamespace(values=values)
    with pytest.raises(RuntimeError, match="jerk limit"):
        CartesianPathActionServer._joint_motion_limits(host, "right_manipulator")
    assert host._joint_motion_limits_cache == {}


def test_plan_publishes_limited_tcp_trajectory_and_reports_reduced_speed():
    host = server_host(limits(velocity=0.01))
    response = SimpleNamespace(solution=nonlinear_path(), start_state=None, fraction=1.0)
    host._cartesian_client = SimpleNamespace(wait_for_service=lambda **kwargs: True,
                                            call_async=lambda request: None)
    host._wait_for_future = lambda *args: response
    host.get_parameter = lambda name: SimpleNamespace(value="World")
    host.publish_trajectories = Mock()
    first, last = Pose(), Pose()
    first.position.x, last.position.x = 1.0, 4.0
    first.orientation.w = last.orientation.w = 1.0
    request = SimpleNamespace(waypoint_hold_s=[], planning_group="right_manipulator",
                              waypoints=[first, last], interpolation_step=0.005,
                              tcp_speed_m_s=0.2, linear_motion_profile=True, velocity_scale=0.2)
    result = CartesianPathActionServer.plan_with_moveit(host, request)
    speeds = cruise_speeds(result.solution)
    assert max(speeds) < 0.2
    assert max(speeds) - min(speeds) < 1e-8
    host.get_logger().warning.assert_called_once()
    host.publish_trajectories.assert_called_once_with(None, [result.solution])
    assert "planned cruise=" in host.get_logger().info.call_args[0][0]


def test_urdf_lookup_failure_does_not_poison_the_cache():
    client = SimpleNamespace(wait_for_service=lambda **kwargs: False)
    host = SimpleNamespace(create_client=Mock(return_value=client), destroy_client=Mock())
    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="robot_state_publisher"):
            CartesianPathActionServer._arm_chain(host, "right_manipulator")
    assert host._arm_chains == {}
    assert host.create_client.call_count == host.destroy_client.call_count == 2


@pytest.mark.parametrize("progress", [[0.0, float("nan")], [0.0, 0.0], [1.0, 2.0], [0.0, float("inf")]])
def test_bad_tcp_progress_rejected_before_retiming(progress):
    path = trajectory([[0.0], [1.0]], [0.0, 1.0])
    before = copy.deepcopy(path)
    with pytest.raises(ValueError, match="strictly increasing"):
        retime_trajectory_constant_velocity(path, ramp_duration_s=1.0, progress=progress)
    assert path == before


def test_straight_joint_retime_follows_speed_scale_and_rejects_detours():
    from construct_robot.core.trajectory_limits import straight_joint_retime
    limits = [{"velocity": 2.0, "acceleration": 0.5, "jerk": 2.0}] * 2
    line = [[0.0, 0.0], [0.4, 0.2], [0.8, 0.4]]
    slow = straight_joint_retime(line, limits, 0.3)
    fast = straight_joint_retime(line, limits, 1.0)
    assert fast[0][-1] < slow[0][-1]
    assert abs(slow[1][-1][0] - 0.8) < 1e-9 and abs(slow[1][-1][1] - 0.4) < 1e-9
    assert max(abs(v[0]) for v in slow[2]) <= 2.0 * 0.3 + 1e-9
    assert max(abs(a[0]) for a in slow[3]) <= 0.5 * 0.3 + 1e-9
    assert straight_joint_retime([[0.0, 0.0], [0.4, 0.5], [0.8, 0.4]], limits, 1.0) is None
