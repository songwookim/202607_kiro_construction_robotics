import math
import struct
import threading
import time

from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
import numpy as np
from rclpy.action import CancelResponse
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
from construct_robot.hicomm_welder import (
    BIT_ARC,
    BIT_FORWARD,
    BIT_REVERSE,
    CAPTURED_IDLE_REQUEST,
    HiCommWelderClient,
    OUTPUT_STATE_MAIN_WELD,
    PROFILE_INCHING,
    PROFILE_WELDING,
    TxState,
    build_request,
    decode_response,
)
from construct_robot.weld_action_gui import (
    aligned_wait_pose,
    corner_seam_from_touches,
    corner_endpoint_from_two_touches,
    corrected_corner_seam_from_four_touches,
    pose_with_local_rpy_offset,
    pose_with_rpy_offset,
    touch_midpoint_wait_pose,
    two_touch_corner_seam,
)


def test_corner_endpoint_and_wait_alignment_for_world_x_seam():
    wall = make_pose(0.4, 0.2, 0.5)
    floor = make_pose(0.4, 0.6, 0.1)
    taught = make_pose(0.7, 0.0, 0.0)
    wait = make_pose(0.6, -0.1, 0.3)
    endpoint = corner_endpoint_from_two_touches(
        wall, floor, taught, "X", wall_offset=0.002, floor_offset=0.003
    )
    aligned = aligned_wait_pose(wait, endpoint, "X")
    assert math.isclose(endpoint.position.x, 0.7)
    assert math.isclose(endpoint.position.y, 0.202)
    assert math.isclose(endpoint.position.z, 0.103)
    assert math.isclose(aligned.position.x, 0.6)
    assert math.isclose(aligned.position.y, endpoint.position.y)
    assert math.isclose(aligned.position.z, endpoint.position.z)


def test_wait_pose_uses_touch_midpoint_for_world_x_seam():
    wait = make_pose(0.8, 9.0, 9.0)
    wall = make_pose(0.4, 0.2, 0.5)
    floor = make_pose(0.4, 0.6, 0.1)
    midpoint = touch_midpoint_wait_pose(wait, wall, floor, "X")
    assert math.isclose(midpoint.position.x, 0.4)
    assert math.isclose(midpoint.position.y, 0.4)
    assert math.isclose(midpoint.position.z, 0.3)


def test_goal_wait_uses_touch_midpoint_for_world_y_seam():
    wait = make_pose(9.0, 0.8, 9.0)
    wall = make_pose(0.2, 1.0, 0.5)
    floor = make_pose(0.6, 1.0, 0.1)
    midpoint = touch_midpoint_wait_pose(wait, wall, floor, "Y")
    assert math.isclose(midpoint.position.x, 0.4)
    assert math.isclose(midpoint.position.y, 1.0)
    assert math.isclose(midpoint.position.z, 0.3)


def test_corner_endpoint_midpoint_normal_intersects_xy_plane():
    wall = make_pose(0.4, 0.2, 0.4)
    floor = make_pose(0.4, 0.6, 0.0)
    taught = make_pose(0.7, 9.0, 9.0)
    endpoint = corner_endpoint_from_two_touches(
        wall, floor, taught, "X"
    )
    assert math.isclose(endpoint.position.x, 0.7)
    assert math.isclose(endpoint.position.y, 0.2)
    assert math.isclose(endpoint.position.z, 0.0)


def test_corner_endpoint_rejects_far_normal_intersection():
    wall = make_pose(0.4, 0.2, 0.4)
    floor = make_pose(0.4, 0.201, 0.0)
    taught = make_pose(0.7, 0.0, 0.0)
    endpoint = corner_endpoint_from_two_touches(
        wall, floor, taught, "X", wall_offset=0.002, floor_offset=0.003
    )
    assert math.isclose(endpoint.position.x, 0.7)
    assert math.isclose(endpoint.position.y, 0.202)
    assert math.isclose(endpoint.position.z, 0.003)


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


def test_corner_touch_pairs_create_midpoint_seam():
    touches = {
        "start_floor": make_pose(0.0, 0.0, 0.0),
        "start_wall": make_pose(0.0, 0.2, 0.2),
        "goal_floor": make_pose(1.0, 0.0, 0.0),
        "goal_wall": make_pose(1.0, 0.2, 0.2),
    }
    points = corner_seam_from_touches(touches, 3)
    assert len(points) == 3
    assert points[0].position.x == 0.0
    assert points[0].position.y == 0.1
    assert points[0].position.z == 0.1
    assert points[-1].position.x == 1.0
    assert points[-1].position.y == 0.1
    assert points[-1].position.z == 0.1


def test_four_touch_correction_projects_each_endpoint_to_corner():
    touches = {
        "start_floor": make_pose(0.0, 0.01, 0.10),
        "start_wall": make_pose(0.02, 0.30, 0.20),
        "goal_floor": make_pose(1.0, 0.02, 0.11),
        "goal_wall": make_pose(1.02, 0.31, 0.21),
    }
    points = corrected_corner_seam_from_four_touches(
        touches, "X", 3, wall_offset=0.01, floor_offset=-0.01
    )
    assert len(points) == 3
    assert math.isclose(points[0].position.x, 0.01)
    assert math.isclose(points[0].position.y, 0.31)
    assert math.isclose(points[0].position.z, 0.09)
    assert math.isclose(points[-1].position.x, 1.01)
    assert math.isclose(points[-1].position.y, 0.32)
    assert math.isclose(points[-1].position.z, 0.10)


def test_two_touch_corner_seam_uses_taught_x_and_touched_yz():
    wall = make_pose(0.2, 0.45, 0.3)
    floor = make_pose(0.4, 0.2, 0.12)
    taught_start = make_pose(1.0, 9.0, 8.0)
    taught_end = make_pose(2.0, 7.0, 6.0)
    points = two_touch_corner_seam(
        wall,
        floor,
        taught_start,
        taught_end,
        "X",
        3,
        wall_offset=0.01,
        floor_offset=-0.02,
    )
    assert len(points) == 3
    assert points[0].position.x == 1.0
    assert points[-1].position.x == 2.0
    assert all(math.isclose(point.position.y, 0.46) for point in points)
    assert all(math.isclose(point.position.z, 0.10) for point in points)


def test_linear_tcp_local_yaw_adjustment_preserves_position():
    pose = make_pose(1.0, 2.0, 3.0)
    adjusted = pose_with_local_rpy_offset(
        pose, 0.0, 0.0, math.radians(90.0)
    )
    assert adjusted.position.x == 1.0
    assert adjusted.position.y == 2.0
    assert adjusted.position.z == 3.0
    assert math.isclose(adjusted.orientation.z, math.sqrt(0.5))
    assert math.isclose(adjusted.orientation.w, math.sqrt(0.5))


def test_linear_tcp_world_rotation_uses_world_axes():
    pose = make_pose(
        quaternion=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    )
    world_adjusted = pose_with_rpy_offset(
        pose, 0.0, 0.0, math.radians(90.0), "world"
    )
    tool_adjusted = pose_with_rpy_offset(
        pose, 0.0, 0.0, math.radians(90.0), "tool"
    )
    assert math.isclose(world_adjusted.orientation.x, 0.5)
    assert math.isclose(world_adjusted.orientation.y, 0.5)
    assert math.isclose(world_adjusted.orientation.z, 0.5)
    assert math.isclose(world_adjusted.orientation.w, 0.5)
    assert math.isclose(tool_adjusted.orientation.y, -0.5)


def test_hicomm_request_preserves_capture_and_sets_documented_fields():
    assert build_request(TxState()) == CAPTURED_IDLE_REQUEST
    assert CAPTURED_IDLE_REQUEST == bytes.fromhex(
        "00 0C 00 64 00 64 00 32 00 00 00 00 33 33 00 00 "
        "0F 00 00 32 32 32 32 32 32 32 32 32 32 32 32 32 "
        "32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 "
        "32 09 00 00 00 00 00"
    )
    frame = build_request(
        TxState(command=BIT_ARC, current_a=123, voltage_tenths=234)
    )
    assert len(frame) == 55
    assert frame[0] == BIT_ARC
    assert int.from_bytes(frame[3:5], "little") == 123
    assert int.from_bytes(frame[5:7], "little") == 234
    assert frame[53:55] == b"\x00\x00"


def test_hicomm_v4_recipe_encodes_documented_bytes_1_through_11():
    frame = build_request(TxState(
        current_a=150,
        voltage_tenths=205,
        material="STS-SOLID",
        diameter_mm=1.2,
        mode="DPM",
        gas="AR80+CO2 20%",
        synergic=True,
        correction=-1.5,
        pre_gas_s=0.5,
        post_gas_s=1.2,
    ))
    assert frame[1] == (2 << 5) | (3 << 2) | 2
    assert frame[2] == 0x81
    assert int.from_bytes(frame[3:5], "little") == 150
    assert int.from_bytes(frame[5:7], "little") == 205
    assert frame[7] == 35
    assert int.from_bytes(frame[8:10], "little") == 50
    assert int.from_bytes(frame[10:12], "little") == 120


def test_hicomm_response_decodes_arc_feedback_and_error():
    frame = bytearray(71)
    frame[0] = BIT_ARC | 0x20 | 0x10 | 0x08
    frame[2:4] = (121).to_bytes(2, "little")
    frame[4:6] = (219).to_bytes(2, "little")
    frame[6] = 32
    frame[1] = OUTPUT_STATE_MAIN_WELD
    frame[9] = 7
    frame[10:12] = (120).to_bytes(2, "little")
    frame[12:14] = (220).to_bytes(2, "little")
    decoded = decode_response(bytes(frame))
    assert decoded["raw_frame"] == bytes(frame)
    assert decoded["arc_ack"] is True
    assert decoded["wcr_detected"] is True
    assert decoded["stick_ack"] is True
    assert decoded["gas_ack"] is True
    assert decoded["arc_established"] is True
    assert decoded["sequence_stage"] == "welding_feedback"
    assert decoded["feedback_current_a"] == 121
    assert decoded["feedback_voltage_v"] == 21.9
    assert decoded["welder_error"] == 7


def test_hicomm_inching_directions_are_mutually_exclusive():
    client = HiCommWelderClient("127.0.0.1", "127.0.0.1")
    client._connected = True
    client.set_command_bit(BIT_FORWARD, True)
    assert client.snapshot().command & BIT_FORWARD
    client.set_command_bit(BIT_REVERSE, True)
    assert client.snapshot().command & BIT_REVERSE
    assert not client.snapshot().command & BIT_FORWARD


def test_hicomm_inching_profile_is_temporary():
    client = HiCommWelderClient("127.0.0.1", "127.0.0.1")
    client._connected = True
    client.arc_set(current_a=150, voltage_tenths=205)
    client.set_command_bit(BIT_FORWARD, True)
    during = client.snapshot()
    assert during.base_profile == PROFILE_INCHING
    assert during.current_a == 100
    assert during.voltage_tenths == 200
    client.set_command_bit(BIT_FORWARD, False)
    restored = client.snapshot()
    assert restored.base_profile == PROFILE_WELDING
    assert restored.current_a == 150
    assert restored.voltage_tenths == 205


def test_hicomm_arc_switches_back_to_successful_welding_profile():
    client = HiCommWelderClient("127.0.0.1", "127.0.0.1")
    client._connected = True
    client.set_command_bit(BIT_FORWARD, True)
    assert client.snapshot().base_profile == PROFILE_INCHING
    client.set_arc(True)
    state = client.snapshot()
    assert state.base_profile == PROFILE_WELDING
    assert state.command == BIT_ARC


def test_hicomm_rejects_manual_test_command_during_arc():
    client = HiCommWelderClient("127.0.0.1", "127.0.0.1")
    client._connected = True
    client.set_arc(True)
    try:
        client.set_command_bit(BIT_FORWARD, True)
    except RuntimeError as error:
        assert "while ARC is ON" in str(error)
    else:
        raise AssertionError("inching must be rejected while ARC is ON")


def test_hicomm_arc_off_cancels_establishment_wait_immediately():
    client = HiCommWelderClient("127.0.0.1", "127.0.0.1")
    client._connected = True
    client.set_arc(True)
    outcome = []

    def wait_for_establishment():
        try:
            client.wait_arc_established(timeout=2.0)
        except Exception as error:
            outcome.append(error)

    waiter = threading.Thread(target=wait_for_establishment)
    waiter.start()
    client.set_arc(False)
    waiter.join(timeout=0.25)

    assert not waiter.is_alive()
    assert len(outcome) == 1
    assert str(outcome[0]) == "ARC OFF during establishment"


def test_hicomm_status_callback_is_coalesced_on_callback_thread():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    received = []

    def slow_callback(status):
        callback_entered.set()
        release_callback.wait(timeout=1.0)
        received.append(status["sequence"])

    client = HiCommWelderClient(
        "127.0.0.1", "127.0.0.1", status_callback=slow_callback
    )
    client._callback_stop.clear()
    callback_thread = threading.Thread(target=client._run_callbacks)
    callback_thread.start()
    client._dispatch_status({"sequence": 1})
    assert callback_entered.wait(timeout=0.25)

    # These calls must return while the user callback is still blocked. Only
    # the newest pending telemetry sample should be delivered afterward.
    client._dispatch_status({"sequence": 2})
    client._dispatch_status({"sequence": 3})
    release_callback.set()
    deadline = time.monotonic() + 0.5
    while len(received) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    client._callback_stop.set()
    with client._callback_condition:
        client._callback_condition.notify_all()
    callback_thread.join(timeout=0.25)

    assert received == [1, 3]


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


def test_cartesian_cancel_is_forwarded_to_active_moveit_execution():
    class ExecuteHandle:
        canceled = False

        def cancel_goal_async(self):
            self.canceled = True

    execute_handle = ExecuteHandle()
    fake_server = type("FakeServer", (), {})()
    fake_server._execute_handle_lock = threading.Lock()
    fake_server._active_execute_handle = execute_handle
    fake_server.get_logger = lambda: _TestLogger()

    response = CartesianPathActionServer.cancel_callback(fake_server, None)

    assert response == CancelResponse.ACCEPT
    assert execute_handle.canceled


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
        self._arc_on_client = object()
        self._arc_off_client = object()

    def require_rbpodo_welder(self):
        self.events.append(("connected",))

    def publish_phase(self, _goal_handle, _request, phase, _progress):
        self.events.append(("phase", phase))

    def call_arc_service(self, _client, command, description):
        self.events.append(("command", description, command))

    def wait_for_welding_feedback(self, expected):
        self.events.append(("feedback", expected))


def test_rbpodo_tcp1_to_tcp2_weld_command_sequence():
    fake = _WeldSequenceFake()
    goal = CartesianPath.Goal()
    goal.waypoints = [make_pose(), make_pose(x=0.1)]
    goal.weld_initial_wait = 0.0
    goal.weld_finish_wait = 0.0
    goal.require_welding_feedback = True

    CartesianPathActionServer.start_welding(fake, object(), goal)
    CartesianPathActionServer.stop_welding(fake, object(), goal)

    commands = [event[1] for event in fake.events if event[0] == "command"]
    assert commands == ["RBPodo arc_on", "RBPodo arc_off"]
    feedback = [event for event in fake.events if event[0] == "feedback"]
    assert feedback == [("feedback", True), ("feedback", False)]
