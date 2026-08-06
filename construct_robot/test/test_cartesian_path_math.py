import math
import struct

from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
import numpy as np
from trajectory_msgs.msg import JointTrajectoryPoint

from construct_robot.cartesian_path_common import (
    circle_waypoints,
    linear_pose_waypoints,
    pose_is_valid,
    scale_trajectory_speed,
    straight_waypoints,
    tip_link_for_group,
    weaving_from_path,
    weaving_waypoints,
)
from construct_robot.cartesian_path_server import (
    CartesianPathActionServer,
    interpolate_pose,
    rotate_vector,
)
from construct_msgs.action import CartesianPath
from construct_robot.h600_modbus_bridge import H600Protocol, H600State


def make_pose(x=0.0, y=0.0, z=0.0, quaternion=(0.0, 0.0, 0.0, 1.0)):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = quaternion
    return pose


def test_tip_link_for_supported_groups():
    assert (
        tip_link_for_group("left_manipulator")
        == "left_manipulator_ee_point"
    )
    assert (
        tip_link_for_group("right_manipulator")
        == "right_manipulator_ee_point"
    )


def test_tip_link_rejects_unknown_group():
    try:
        tip_link_for_group("dual_arm")
    except ValueError as error:
        assert "Unsupported planning group" in str(error)
    else:
        raise AssertionError(
            "Expected an unsupported group to raise ValueError"
        )


def test_pose_validation():
    assert pose_is_valid(make_pose())
    assert not pose_is_valid(make_pose(quaternion=(0.0, 0.0, 0.0, 0.0)))
    assert not pose_is_valid(make_pose(x=math.nan))


def test_interpolation_handles_antipodal_quaternions():
    start = make_pose(quaternion=(0.0, 0.0, 0.0, 1.0))
    goal = make_pose(x=2.0, quaternion=(0.0, 0.0, 0.0, -1.0))
    midpoint = interpolate_pose(start, goal, 0.5)
    assert midpoint.position.x == 1.0
    assert midpoint.orientation.w == 1.0


def test_rotate_vector_quarter_turn_about_z():
    pose = make_pose(
        quaternion=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    )
    rotated = rotate_vector(pose.orientation, (1.0, 0.0, 0.0))
    assert math.isclose(rotated[0], 0.0, abs_tol=1e-9)
    assert math.isclose(rotated[1], 1.0, abs_tol=1e-9)
    assert math.isclose(rotated[2], 0.0, abs_tol=1e-9)


def test_straight_waypoints_world_negative_axis_and_spacing():
    start = make_pose(x=1.0, y=2.0, z=3.0)
    points = straight_waypoints(
        start,
        distance=-0.2,
        count=5,
        axis="y",
        reference="world",
    )
    assert len(points) == 5
    assert math.isclose(points[0].position.y, 2.0)
    assert math.isclose(points[2].position.y, 1.9)
    assert math.isclose(points[-1].position.y, 1.8)
    assert points[-1].orientation == start.orientation


def test_straight_waypoints_tool_axis_uses_tcp_orientation():
    start = make_pose(
        quaternion=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    )
    points = straight_waypoints(
        start,
        distance=0.2,
        count=3,
        axis="x",
        reference="tool",
    )
    assert math.isclose(points[-1].position.x, 0.0, abs_tol=1e-9)
    assert math.isclose(points[-1].position.y, 0.2, abs_tol=1e-9)


def test_straight_waypoints_rejects_invalid_inputs():
    try:
        straight_waypoints(make_pose(), 0.0, 2)
    except ValueError as error:
        assert "distance" in str(error)
    else:
        raise AssertionError("Expected zero straight distance to be rejected")


def test_linear_pose_waypoints_interpolates_two_tcp_6d_poses():
    start = make_pose(
        x=0.1,
        y=0.2,
        z=0.3,
        quaternion=(0.0, 0.0, 0.0, 1.0),
    )
    end = make_pose(
        x=0.3,
        y=0.4,
        z=0.5,
        quaternion=(0.0, 0.0, 1.0, 0.0),
    )
    points = linear_pose_waypoints(start, end, 3)
    assert len(points) == 3
    assert points[0] == start
    assert points[-1] == end
    assert math.isclose(points[1].position.x, 0.2)
    assert math.isclose(points[1].position.y, 0.3)
    assert math.isclose(points[1].position.z, 0.4)
    assert math.isclose(points[1].orientation.z, math.sqrt(0.5))
    assert math.isclose(points[1].orientation.w, math.sqrt(0.5))


def test_linear_pose_waypoints_rejects_same_position():
    start = make_pose()
    end = make_pose(quaternion=(0.0, 0.0, 1.0, 0.0))
    try:
        linear_pose_waypoints(start, end, 2)
    except ValueError as error:
        assert "different positions" in str(error)
    else:
        raise AssertionError("Expected equal TCP positions to be rejected")


def test_closed_four_point_circle_has_four_unique_points_and_return():
    center = make_pose(1.0, 2.0, 3.0)
    points = circle_waypoints(center, radius=0.1, count=4, closed=True)
    assert len(points) == 5
    assert math.isclose(points[0].position.y, 2.1)
    assert math.isclose(points[1].position.z, 3.1)
    assert points[-1] == points[0]


def test_circle_6d_poses_point_tcp_positive_z_toward_center():
    center = make_pose(x=0.4, y=0.2, z=0.7)
    points = circle_waypoints(
        center,
        radius=0.1,
        count=8,
        closed=False,
        face_center=True,
    )
    for pose in points:
        quaternion = pose.orientation
        local_z = np.array([0.0, 0.0, 1.0])
        q_xyz = np.array(
            [quaternion.x, quaternion.y, quaternion.z]
        )
        rotated_z = (
            local_z
            + 2.0
            * (
                quaternion.w * np.cross(q_xyz, local_z)
                + np.cross(q_xyz, np.cross(q_xyz, local_z))
            )
        )
        inward = np.array(
            [
                center.position.x - pose.position.x,
                center.position.y - pose.position.y,
                center.position.z - pose.position.z,
            ]
        )
        inward /= np.linalg.norm(inward)
        assert np.allclose(rotated_z, inward, atol=1e-7)


def test_circle_normal_axis_selects_world_plane():
    center = make_pose(1.0, 2.0, 3.0)
    expected_constant_coordinate = {"x": 1.0, "y": 2.0, "z": 3.0}
    for axis, expected in expected_constant_coordinate.items():
        points = circle_waypoints(
            center,
            radius=0.1,
            count=8,
            closed=False,
            normal_axis=axis,
        )
        assert all(
            math.isclose(getattr(point.position, axis), expected)
            for point in points
        )


def test_circle_rejects_unknown_normal_axis():
    try:
        circle_waypoints(make_pose(), 0.1, 8, normal_axis="bad")
    except ValueError as error:
        assert "normal axis" in str(error)
    else:
        raise AssertionError("Expected unknown normal axis to be rejected")


def test_trajectory_velocity_scaling_changes_time_velocity_acceleration():
    trajectory = RobotTrajectory()
    point = JointTrajectoryPoint()
    point.time_from_start.sec = 1
    point.time_from_start.nanosec = 500_000_000
    point.velocities = [2.0]
    point.accelerations = [4.0]
    trajectory.joint_trajectory.points = [point]
    scale_trajectory_speed(trajectory, 0.5)
    scaled = trajectory.joint_trajectory.points[0]
    assert scaled.time_from_start.sec == 3
    assert scaled.time_from_start.nanosec == 0
    assert list(scaled.velocities) == [1.0]
    assert list(scaled.accelerations) == [1.0]


def test_weaving_path_starts_and_ends_on_centerline():
    start = make_pose(1.0, 2.0, 3.0)
    points = weaving_waypoints(
        start,
        length=0.08,
        amplitude=0.003,
        cycles=2,
        samples_per_cycle=8,
    )
    assert len(points) == 17
    assert math.isclose(points[0].position.z, 3.0)
    assert math.isclose(points[-1].position.y, 2.08)
    assert math.isclose(points[-1].position.z, 3.0, abs_tol=1e-12)
    assert math.isclose(max(p.position.z for p in points), 3.003)
    assert math.isclose(min(p.position.z for p in points), 2.997)


def test_weaving_is_applied_to_existing_taught_line():
    source = [
        make_pose(x=0.1, y=0.2, z=0.3),
        make_pose(x=0.2, y=0.2, z=0.3),
    ]
    points = weaving_from_path(
        source,
        amplitude=0.004,
        cycles=1,
        samples_per_cycle=4,
        transverse_axis="tool_y",
    )
    assert len(points) == 5
    assert math.isclose(points[0].position.x, 0.1)
    assert math.isclose(points[-1].position.x, 0.2)
    assert math.isclose(points[1].position.y, 0.204)
    assert math.isclose(points[3].position.y, 0.196)
    assert math.isclose(points[0].position.y, 0.2)
    assert math.isclose(points[-1].position.y, 0.2)


def test_approved_plan_signature_changes_with_path_or_speed():
    goal = CartesianPath.Goal()
    goal.planning_group = "right_manipulator"
    goal.interpolation_step = 0.005
    goal.velocity_scale = 0.2
    goal.waypoints = [make_pose(x=0.1), make_pose(x=0.2)]
    original = CartesianPathActionServer.plan_signature(goal)

    goal.enable_arc = True
    assert CartesianPathActionServer.plan_signature(goal) != original
    goal.enable_arc = False
    goal.velocity_scale = 0.1
    assert CartesianPathActionServer.plan_signature(goal) != original
    goal.velocity_scale = 0.2
    goal.waypoints[1].position.y = 0.001
    assert CartesianPathActionServer.plan_signature(goal) != original


class _TestLogger:
    def warning(self, _message):
        pass


def test_h600_modbus_command_read_and_feedback_write():
    state = H600State(
        robot_ready=True,
        command_robot_error=True,
        command_touch=True,
        gas=True,
        reverse_inching=True,
        inching=True,
        arc=True,
        current_raw=120,
        voltage_raw=240,
    )
    protocol = H600Protocol(state, _TestLogger())
    response = protocol.process_pdu(struct.pack(">BHH", 0x03, 201, 10))
    values = struct.unpack(">10H", response[2:])
    assert values[0] == 1
    assert values[1] == 0x009F
    assert values[3:5] == (120, 240)

    state.command_robot_error = False
    state.command_touch = False
    state.gas = False
    state.reverse_inching = False
    state.inching = True
    state.arc = False
    assert state.command_registers()[1] == 0x0002
    state.reverse_inching = True
    state.inching = False
    assert state.command_registers()[1] == 0x0004

    write = (
        struct.pack(">BHHB", 0x10, 211, 3, 6)
        + struct.pack(">3H", 0x0020, 111, 222)
    )
    assert protocol.process_pdu(write) == struct.pack(">BHH", 0x10, 211, 3)
    assert state.registers[211] == 0x0020
    assert state.registers[212] == 111
    assert state.registers[213] == 222


class _WeldSequenceFake:
    def __init__(self):
        self.events = []

    def require_h600_connection(self):
        self.events.append(("connected",))
        return "192.168.1.2:50200"

    def publish_phase(self, _goal_handle, _request, phase, _progress):
        self.events.append(("phase", phase))

    def set_welder(self, _request, ready, gas, arc, setpoints):
        self.events.append(("command", ready, gas, arc, setpoints))

    def wait_for_welding_feedback(self, expected):
        self.events.append(("feedback", expected))


def test_h600_tcp1_to_tcp2_weld_command_sequence():
    fake = _WeldSequenceFake()
    goal = CartesianPath.Goal()
    goal.waypoints = [make_pose(), make_pose(x=0.1)]
    goal.weld_preflow_seconds = 0.0
    goal.weld_postflow_seconds = 0.0
    goal.require_welding_feedback = True

    CartesianPathActionServer.start_welding(fake, object(), goal)
    CartesianPathActionServer.stop_welding(fake, object(), goal)

    commands = [event for event in fake.events if event[0] == "command"]
    assert commands == [
        ("command", True, True, False, True),
        ("command", True, True, True, True),
        ("command", True, True, False, False),
        ("command", False, False, False, False),
    ]
    feedback = [event for event in fake.events if event[0] == "feedback"]
    assert feedback == [("feedback", True), ("feedback", False)]
