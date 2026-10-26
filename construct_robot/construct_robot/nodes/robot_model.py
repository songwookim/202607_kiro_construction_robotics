"""Kinematic chains from the running robot_state_publisher, for plan checks."""

import math

from rcl_interfaces.srv import GetParameters

from construct_robot.core.cartesian_path_common import tip_link_for_group
from construct_robot.core.path_fidelity import tcp_path_deviation
from construct_robot.nodes.action_call import wait_for_future


# Commanded TCP deviation from the requested Cartesian path above which a
# plan is treated as an IK branch change and refused.  Real Humble plans of
# this cell stay below 0.5 mm / 4 deg (weave without dwell: 0.44 mm, single
# 150 deg re-orientation step: 3.7 deg); a branch change moves the TCP by tens
# to thousands of millimetres.
PATH_DEVIATION_LIMIT_MM = 3.0
PATH_ORIENTATION_LIMIT_DEG = 10.0


def fetch_robot_description(node, timeout=5.0, callback_group=None):
    """URDF string of /robot_state_publisher; RuntimeError when unavailable.

    Blocks the calling thread: the node must be spun by another executor
    thread (pass a reentrant callback group when called from a callback).
    """
    client = node.create_client(
        GetParameters, "/robot_state_publisher/get_parameters",
        callback_group=callback_group,
    )
    try:
        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("robot_state_publisher parameters unavailable")
        reply = wait_for_future(
            client.call_async(GetParameters.Request(names=["robot_description"])),
            timeout, "robot_description",
        )
    finally:
        node.destroy_client(client)
    urdf = reply.values[0].string_value if reply and reply.values else ""
    if not urdf:
        raise RuntimeError("empty robot_description")
    return urdf


def planning_frame_chain(urdf, planning_frame, planning_group):
    """FK chain from the planning frame to the group's TCP (fixed joints only between)."""
    from construct_robot.core.keyboard_jog_kinematics import ArmChain

    return ArmChain(urdf, planning_frame, tip_link_for_group(planning_group))


def verify_path_fidelity(chain, waypoints, trajectory,
                         position_limit_mm=PATH_DEVIATION_LIMIT_MM,
                         orientation_limit_deg=PATH_ORIENTATION_LIMIT_DEG):
    """Measure the commanded TCP path; RuntimeError above the limits."""
    try:
        deviation = tcp_path_deviation(trajectory, chain.joint_names, chain.fk, list(waypoints))
    except ValueError as error:
        raise RuntimeError(f"commanded TCP path could not be verified: {error}") from error
    position_limit_m = float(position_limit_mm) * 0.001
    orientation_limit_rad = math.radians(float(orientation_limit_deg))
    if deviation["position_m"] > position_limit_m or deviation["orientation_rad"] > orientation_limit_rad:
        raise RuntimeError(
            "Planned trajectory leaves the requested Cartesian path by "
            f"{deviation['position_m'] * 1000.0:.1f} mm / "
            f"{math.degrees(deviation['orientation_rad']):.1f} deg at t={deviation['time_s']:.2f} s "
            f"(limit {position_limit_m * 1000.0:.1f} mm / {math.degrees(orientation_limit_rad):.1f} deg): "
            "IK branch change (wrist flip / joint turn) suspected; execution inhibited"
        )
    return deviation


def describe_fidelity(deviation):
    return (
        f"TCP path deviation max {deviation['position_m'] * 1000.0:.3f} mm / "
        f"{math.degrees(deviation['orientation_rad']):.2f} deg"
    )
