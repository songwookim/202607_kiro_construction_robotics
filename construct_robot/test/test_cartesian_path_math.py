import math
import threading
import time
from types import SimpleNamespace

from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
import numpy as np
import yaml
from rclpy.action import CancelResponse
from trajectory_msgs.msg import JointTrajectoryPoint

from construct_robot.cartesian_path_common import (
    circle_waypoints,
    linear_pose_waypoints,
    pose_is_valid,
    scale_trajectory_speed,
    scale_trajectory_to_tcp_speed,
    slerp_quaternion,
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
import construct_robot.hicomm_welder as hicomm_welder_module
from construct_robot.hicomm_welder import (
    BIT_ARC,
    BIT_FORWARD,
    BIT_REVERSE,
    CAPTURED_IDLE_REQUEST,
    CAPTURED_WELDING_BASE_REQUEST_0102,
    HiCommWelderClient,
    OUTPUT_STATE_MAIN_WELD,
    PROFILE_INCHING,
    PROFILE_WELDING,
    PROTOCOL_PROFILE_VERSION,
    TxState,
    build_request,
    decode_response,
)
from construct_robot.weld_action_gui import (
    DEFAULT_DIGITAL_WELD_SETTINGS,
    DI8_GUARDED_TEACHING_POSES,
    WeldGuiNode,
    aligned_wait_pose,
    corner_seam_from_touches,
    corner_endpoint_from_two_touches,
    corrected_corner_seam_from_four_touches,
    digital_weld_recipe,
    next_sequential_slot,
    pose_with_local_rpy_offset,
    pose_with_rpy_offset,
    named_tcp_linear_waypoints,
    position_only_goal_constraints,
    tcp_pose_goal_constraints,
    save_seam_touch_yaml,
    save_weld_feedback_log,
    seam_yaw,
    translated_wait_pose,
    two_touch_corner_seam,
    validate_digital_weld_settings,
    validate_managed_weld_sequence,
    yaw_corrected_seam_poses,
)
from construct_robot.weld_feedback_plot import parse_weld_feedback_log


def managed_weld_steps(base_slot=1):
    scenario_id = "test-weld"
    stages = (
        ("start_wait", "named_pose", None, True, False),
        ("start_contact", "motion", None, True, True),
        ("touch_output_off", "digital_output", None, False, False),
        ("arc_on", "digital_weld", "on", False, False),
        ("weld_motion", "motion", None, False, False),
        ("arc_off", "digital_weld", "off", False, False),
        ("goal_wait", "named_pose", None, True, False),
        ("finish", "named_pose", None, True, False),
    )
    stage_slot_offsets = {
        "start_wait": 0,
        "start_contact": 1,
        "touch_output_off": 2,
        "arc_on": 3,
        "weld_motion": 3,
        "arc_off": 4,
        "goal_wait": 5,
        "finish": 6,
    }
    result = []
    for stage, kind, command, guard, continue_after in stages:
        step = {
            "type": kind,
            "parallel_slot": base_slot + stage_slot_offsets[stage],
            "duration": 0.0,
            "touch_guard": guard,
            "continue_after_touch": continue_after,
            "weld_scenario_id": scenario_id,
            "weld_scenario_stage": stage,
        }
        if command is not None:
            step["command"] = command
        if stage == "start_contact":
            step["accept_initial_touch"] = False
        elif stage == "touch_output_off":
            step["port"] = 4
            step["value"] = False
        result.append(step)
    return result


def test_weld_scenario_uses_slots_after_existing_sequence():
    existing = [
        {"type": "motion", "parallel_slot": 2},
        {"type": "digital_weld", "parallel_slot": 7},
        {"type": "sleep", "seconds": 1.0},
    ]
    assert next_sequential_slot(existing, requested=3) == 8
    assert next_sequential_slot([], requested=4) == 4


def test_weld_feedback_log_is_persisted_atomically(tmp_path):
    path = tmp_path / "latest_weld_feedback.log"
    document = {
        "format_version": 1,
        "result": "completed",
        "started": "2026-08-18 12:00:00",
        "ended": "2026-08-18 12:00:01",
        "elapsed_seconds": 1.0,
        "commanded": {"current_a": 200, "voltage": 25.0},
        "rx_setting_echo": {"current_a": 200, "voltage_v": 25.0},
        "feedback": {
            "rx_samples": 1,
            "welding_samples": 1,
            "wcr_seen": True,
            "current_a": {"min": 197.0, "average": 197.5, "max": 198.0},
            "voltage_v": {"min": 24.8, "average": 25.0, "max": 25.2},
            "wire_feed_m_min": {"min": 7.4, "average": 7.5, "max": 7.6},
        },
        "samples": [{
            "elapsed_s": 0.5,
            "raw0": 0x2B,
            "output_state_name": "main_weld",
            "arc_ack": True,
            "gas_ack": True,
            "forward_ack": True,
            "wcr_detected": True,
            "feedback_current_a": 198,
            "feedback_voltage_v": 25.2,
            "wire_feed_m_min": 7.6,
            "set_current_a": 200,
            "set_voltage_v": 25.0,
            "welder_error": 0,
            "db_unavailable": False,
            "torch_collision": False,
        }],
    }
    save_weld_feedback_log(path, document)
    content = path.read_text(encoding="utf-8")
    assert "result=completed" in content
    assert "OPERATOR OVERVIEW" in content
    assert "REQUESTED : 200 A / 25.0 V" in content
    assert "current_a.average=197.5" in content
    assert "0x2B main_weld 1 1 1 1 198 25.2 7.6" in content
    sections, samples = parse_weld_feedback_log(path)
    assert sections["commanded"]["current_a"] == "200"
    assert samples[0]["current_a"] == 198.0
    assert samples[0]["voltage_v"] == 25.2


def test_managed_weld_scenario_pairs_arc_on_with_weld_motion():
    steps = managed_weld_steps()
    assert validate_managed_weld_sequence(steps, require_complete=True)
    assert steps[3]["parallel_slot"] == steps[4]["parallel_slot"]

    unsafe = managed_weld_steps()
    unsafe[4]["parallel_slot"] += 1
    try:
        validate_managed_weld_sequence(unsafe, require_complete=True)
    except ValueError as error:
        assert "share" in str(error) or "slot" in str(error)
    else:
        raise AssertionError("Separated generated ARC ON/motion was accepted")


def test_manual_arc_on_cannot_share_a_robot_motion_slot():
    unsafe = [
        {
            "type": "digital_weld",
            "command": "on",
            "parallel_slot": 3,
            "duration": 0.0,
        },
        {"type": "motion", "parallel_slot": 3},
    ]
    try:
        validate_managed_weld_sequence(unsafe)
    except ValueError as error:
        assert "cannot share slot 3" in str(error)
    else:
        raise AssertionError("Manual ARC ON/motion slot overlap was accepted")


def test_managed_weld_scenario_requires_fresh_start_touch_and_explicit_off():
    stale_touch = managed_weld_steps()
    stale_touch[1]["accept_initial_touch"] = True
    try:
        validate_managed_weld_sequence(stale_touch, require_complete=True)
    except ValueError as error:
        assert "new DI8 edge" in str(error)
    else:
        raise AssertionError("stale DI8 was accepted for generated START")

    timed_arc = managed_weld_steps()
    timed_arc[3]["duration"] = 3.0
    try:
        validate_managed_weld_sequence(timed_arc, require_complete=True)
    except ValueError as error:
        assert "duration must be 0" in str(error)
    else:
        raise AssertionError("timed generated ARC ON was accepted")

    touch_output_on = managed_weld_steps()
    touch_output_on[2]["value"] = True
    try:
        validate_managed_weld_sequence(
            touch_output_on, require_complete=True
        )
    except ValueError as error:
        assert "DO4 OFF before ARC ON" in str(error)
    else:
        raise AssertionError("Generated ARC accepted with DO4 ON")


def test_di8_guarded_named_teaching_poses():
    assert DI8_GUARDED_TEACHING_POSES == {
        "robot_start",
        "weld_wait",
        "weld_start_wait",
        "weld_start",
        "weld_goal_wait",
        "weld_end",
        "weld_finish",
    }


def test_position_only_goal_ignores_yaml_orientation():
    target = make_pose(
        0.1,
        -0.2,
        0.3,
        quaternion=(0.7, -0.2, 0.1, 0.67),
    )
    constraints = position_only_goal_constraints(
        "right_manipulator", target, tolerance=0.0005
    )
    assert constraints.orientation_constraints == []
    assert constraints.joint_constraints == []
    position = constraints.position_constraints[0]
    center = position.constraint_region.primitive_poses[0]
    assert position.header.frame_id == "World"
    assert position.link_name == "right_manipulator_ee_point"
    assert center.position == target.position
    assert center.orientation.w == 1.0
    assert list(position.constraint_region.primitives[0].dimensions) == [0.0005]


def test_tcp_pose_goal_combines_position_and_captured_orientation():
    target = make_pose(
        0.1,
        -0.2,
        0.3,
        quaternion=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
    )
    constraints = tcp_pose_goal_constraints("right_manipulator", target)
    center = (
        constraints.position_constraints[0]
        .constraint_region.primitive_poses[0]
    )
    orientation = constraints.orientation_constraints[0]
    assert center.position == target.position
    assert orientation.orientation == target.orientation
    assert orientation.link_name == "right_manipulator_ee_point"


def test_named_tcp_transition_is_linear_xyz_and_slerp_orientation():
    start = make_pose(quaternion=(0.0, 0.0, 0.0, 1.0))
    goal = make_pose(
        x=0.010,
        quaternion=(0.0, 0.0, 1.0, 0.0),
    )
    points = named_tcp_linear_waypoints(start, goal)
    assert points[0] == start
    assert points[-1] == goal
    middle = points[len(points) // 2]
    assert math.isclose(middle.position.x, 0.005, abs_tol=1e-9)
    assert math.isclose(middle.orientation.z, math.sqrt(0.5), abs_tol=1e-9)
    assert math.isclose(middle.orientation.w, math.sqrt(0.5), abs_tol=1e-9)


def test_sensed_seam_yaw_rotates_taught_orientations_and_uses_sensed_xyz():
    taught_start = make_pose(0.0, 0.0, 0.2)
    taught_goal = make_pose(1.0, 0.0, 0.2)
    sensed_start = make_pose(0.3, -0.4, 0.5)
    sensed_goal = make_pose(0.3, 0.6, 0.5)
    corrected_start, corrected_goal, delta = yaw_corrected_seam_poses(
        taught_start, taught_goal, sensed_start, sensed_goal
    )
    assert math.isclose(seam_yaw(taught_start, taught_goal), 0.0)
    assert math.isclose(delta, math.pi / 2.0)
    assert corrected_start.position == sensed_start.position
    assert corrected_goal.position == sensed_goal.position
    assert math.isclose(corrected_start.orientation.z, math.sqrt(0.5))
    assert math.isclose(corrected_start.orientation.w, math.sqrt(0.5))


def test_corner_endpoint_and_wait_alignment_for_world_x_seam():
    wall = make_pose(0.4, 0.2, 0.5)
    floor = make_pose(0.4, 0.6, 0.1)
    taught = make_pose(0.7, 0.0, 0.0)
    wait = make_pose(0.6, -0.1, 0.3)
    endpoint = corner_endpoint_from_two_touches(
        wall, floor, taught, "X", wall_offset=0.002, floor_offset=0.003
    )
    aligned = aligned_wait_pose(wait, endpoint, "X")
    assert math.isclose(endpoint.position.x, 0.4)
    assert math.isclose(endpoint.position.y, 0.202)
    assert math.isclose(endpoint.position.z, 0.103)
    assert math.isclose(aligned.position.x, 0.6)
    assert math.isclose(aligned.position.y, endpoint.position.y)
    assert math.isclose(aligned.position.z, endpoint.position.z)


def test_wait_pose_preserves_taught_clearance_for_world_x_seam():
    taught_seam = make_pose(0.7, 0.1, 0.2)
    wait = make_pose(0.6, -0.1, 0.4)
    corrected_seam = make_pose(0.7, 0.2, 0.1)
    corrected_wait = translated_wait_pose(wait, taught_seam, corrected_seam)
    assert math.isclose(corrected_wait.position.x, 0.6)
    assert math.isclose(corrected_wait.position.y, 0.0)
    assert math.isclose(corrected_wait.position.z, 0.3)
    assert math.isclose(
        corrected_wait.position.x - corrected_seam.position.x,
        wait.position.x - taught_seam.position.x,
    )


def test_goal_wait_translation_preserves_orientation():
    taught_seam = make_pose(0.2, 0.8, 0.1)
    wait = make_pose(0.1, 0.7, 0.3, quaternion=(0.0, 0.0, 0.5, 0.5))
    corrected_seam = make_pose(0.3, 1.0, 0.15)
    corrected_wait = translated_wait_pose(wait, taught_seam, corrected_seam)
    assert math.isclose(corrected_wait.position.x, 0.2)
    assert math.isclose(corrected_wait.position.y, 0.9)
    assert math.isclose(corrected_wait.position.z, 0.35)
    assert math.isclose(corrected_wait.orientation.z, 0.5)


def test_raw_seam_touch_yaml_records_contact_and_probe_start(tmp_path):
    path = tmp_path / "right_manipulator_seam_touch_points.yaml"
    touches = {
        "start_wall": make_pose(0.4, 0.2, 0.5),
        "start_floor": make_pose(0.402, 0.6, 0.1),
    }
    starts = {
        "start_wall": make_pose(0.401, 0.4, 0.4),
        "start_floor": make_pose(0.401, 0.4, 0.4),
    }
    stops = {
        "start_wall": make_pose(0.4, 0.198, 0.5),
    }
    save_seam_touch_yaml(
        path, "right_manipulator", "X", touches, starts, stops
    )
    document = yaml.safe_load(path.read_text())
    assert document["seam_axis"] == "X"
    assert document["touches"]["start_wall"]["contact_tcp"][
        "position_m"
    ]["x"] == 0.4
    assert document["touches"]["start_floor"]["probe_start_tcp"][
        "position_m"
    ]["x"] == 0.401
    assert document["touches"]["start_wall"]["stopped_tcp"][
        "position_m"
    ]["y"] == 0.198


def test_corner_endpoint_uses_touched_xyz_not_taught_position():
    wall = make_pose(0.4, 0.2, 0.4)
    floor = make_pose(0.4, 0.6, 0.0)
    taught = make_pose(0.7, 9.0, 9.0)
    endpoint = corner_endpoint_from_two_touches(
        wall, floor, taught, "X"
    )
    assert math.isclose(endpoint.position.x, 0.4)
    assert math.isclose(endpoint.position.y, 0.2)
    assert math.isclose(endpoint.position.z, 0.0)


def test_corner_endpoint_applies_offsets_to_touched_xyz():
    wall = make_pose(0.4, 0.2, 0.4)
    floor = make_pose(0.4, 0.201, 0.0)
    taught = make_pose(0.7, 0.0, 0.0)
    endpoint = corner_endpoint_from_two_touches(
        wall, floor, taught, "X", wall_offset=0.002, floor_offset=0.003
    )
    assert math.isclose(endpoint.position.x, 0.4)
    assert math.isclose(endpoint.position.y, 0.202)
    assert math.isclose(endpoint.position.z, 0.003)


def test_yz_touch_endpoint_rejects_non_world_x_axis():
    try:
        corner_endpoint_from_two_touches(
            make_pose(0.4, 0.2, 0.4),
            make_pose(0.4, 0.6, 0.0),
            make_pose(0.7, 0.0, 0.0),
            "Y",
        )
    except ValueError as error:
        assert "requires World X" in str(error)
    else:
        raise AssertionError("Y/Z probing must reject a World Y seam axis")


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
        "00 0C 00 64 00 64 00 32 00 00 00 00 32 32 00 00 "
        "0F 00 00 32 32 32 32 32 32 32 32 32 32 32 32 32 "
        "32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 "
        "32 09 00 00 00 00 00"
    )
    assert PROTOCOL_PROFILE_VERSION == "v5.3-1120"
    assert CAPTURED_WELDING_BASE_REQUEST_0102[12:14] == b"\x33\x33"
    assert CAPTURED_IDLE_REQUEST[12:14] == b"\x32\x32"
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


def test_digital_weld_defaults_match_captured_welding_profile():
    settings = validate_digital_weld_settings(
        DEFAULT_DIGITAL_WELD_SETTINGS
    )
    assert settings["current_a"] == 100
    assert settings["voltage_tenths"] == 100
    assert settings["voltage"] == 10.0
    assert build_request(TxState(**digital_weld_recipe(settings))) == (
        CAPTURED_IDLE_REQUEST
    )


def test_digital_weld_recipe_excludes_gui_timing_metadata():
    settings = validate_digital_weld_settings({
        "current_a": "150",
        "voltage_tenths": "205",
        "preflow_seconds": "1.5",
    })
    recipe = digital_weld_recipe(settings)
    assert settings["voltage"] == 20.5
    assert settings["preflow_seconds"] == 1.5
    assert "voltage" not in recipe
    assert "preflow_seconds" not in recipe


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


def test_hicomm_v53_rx_drain_returns_after_one_complete_frame(monkeypatch):
    client = HiCommWelderClient("127.0.0.1", "127.0.0.1")
    received = []

    class FakeSocket:
        calls = 0

        def recv(self, _size):
            self.calls += 1
            return bytes(71)

    sock = FakeSocket()
    monkeypatch.setattr(
        hicomm_welder_module.select,
        "select",
        lambda *_args, **_kwargs: ([sock], [], []),
    )
    client._dispatch_status = lambda status: received.append(status)
    client._drain_rx(sock, bytearray(), time.monotonic() + 1.0)

    assert sock.calls == 1
    assert len(received) == 1


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
    assert str(outcome[0]) == (
        "ARC OFF while waiting for ARC ESTABLISHED (WCR + feed)"
    )


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


def test_weld_gui_action_waiter_returns_result_and_clears_active_goal():
    expected_result = object()

    class CompletedFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

        def add_done_callback(self, callback):
            callback(self)

    class GoalHandle:
        accepted = True

        @staticmethod
        def get_result_async():
            return CompletedFuture(SimpleNamespace(result=expected_result))

    class ActionClient:
        @staticmethod
        def wait_for_server(timeout_sec):
            return timeout_sec == 3.0

        @staticmethod
        def send_goal_async(_goal):
            return CompletedFuture(GoalHandle())

    node = SimpleNamespace(active_motion_goal=None)
    result = WeldGuiNode._send_action_goal_and_wait(
        node,
        ActionClient(),
        object(),
        "test motion",
    )
    assert result is expected_result
    assert node.active_motion_goal is None


def test_interpolation_handles_antipodal_quaternions():
    start = make_pose(quaternion=(0.0, 0.0, 0.0, 1.0))
    goal = make_pose(x=2.0, quaternion=(0.0, 0.0, 0.0, -1.0))
    midpoint = interpolate_pose(start, goal, 0.5)
    assert midpoint.position.x == 1.0
    assert midpoint.orientation.w == 1.0


def test_quaternion_slerp_has_constant_angular_progress():
    start = make_pose(quaternion=(0.0, 0.0, 0.0, 1.0))
    goal = make_pose(quaternion=(0.0, 0.0, 1.0, 0.0))
    quarter = slerp_quaternion(
        start.orientation, goal.orientation, 0.25
    )
    assert math.isclose(quarter[2], math.sin(math.pi / 8.0), abs_tol=1e-9)
    assert math.isclose(quarter[3], math.cos(math.pi / 8.0), abs_tol=1e-9)


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
    goal.tcp_speed_m_s = 0.0
    goal.waypoints = [make_pose(x=0.1), make_pose(x=0.2)]
    original = CartesianPathActionServer.plan_signature(goal)

    goal.velocity_scale = 0.1
    assert CartesianPathActionServer.plan_signature(goal) != original
    goal.velocity_scale = 0.2
    goal.tcp_speed_m_s = 0.01
    assert CartesianPathActionServer.plan_signature(goal) != original
    goal.tcp_speed_m_s = 0.0
    goal.waypoints[1].position.y = 0.001
    assert CartesianPathActionServer.plan_signature(goal) != original


def test_tcp_speed_mode_sets_average_duration_without_exceeding_plan():
    trajectory = RobotTrajectory()
    point = JointTrajectoryPoint()
    point.time_from_start.sec = 2
    point.velocities = [1.0]
    point.accelerations = [1.0]
    trajectory.joint_trajectory.points.append(point)
    waypoints = [make_pose(x=0.0), make_pose(x=0.1)]

    scale, achieved = scale_trajectory_to_tcp_speed(
        trajectory, waypoints, 0.025
    )
    assert math.isclose(scale, 0.5)
    assert trajectory.joint_trajectory.points[-1].time_from_start.sec == 4
    assert math.isclose(achieved, 0.025)

    fast_trajectory = RobotTrajectory()
    fast_point = JointTrajectoryPoint()
    fast_point.time_from_start.sec = 2
    fast_trajectory.joint_trajectory.points.append(fast_point)
    scale, achieved = scale_trajectory_to_tcp_speed(
        fast_trajectory, waypoints, 0.1
    )
    assert math.isclose(scale, 1.0)
    assert math.isclose(achieved, 0.05)


def test_cartesian_action_is_motion_only():
    goal = CartesianPath.Goal()
    assert not hasattr(goal, "enable_arc")
    assert not hasattr(goal, "weld_current_a")
    assert not hasattr(goal, "require_welding_feedback")


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
