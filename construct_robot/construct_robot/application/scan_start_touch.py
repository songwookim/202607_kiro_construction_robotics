"""Bounded, fresh-edge start-height probing for the scan/multipass cycle.

Only the execution snapshot changes. Stored scan truth and later passes retain
their own geometry; each pass measures its own height before ARC preplanning.
"""

import copy
import math
import time
import threading

from construct_robot.core.cartesian_path_common import pose_is_valid


# Go all the way to the pass's reference TCP (its START), then search World -Z slowly from there (operator
# 2026-10-19). The search starts only at the reference TCP: a contact on the way there stops (ARC blocked).
START_TOUCH = dict(speed_m_s=0.0005, max_descent_m=0.010)


def configure_start_touch(steps, policy):
    motion = next(s for s in steps if s.get("weld_scenario_stage") == "weld_motion")
    contact = next(s for s in steps if s.get("weld_scenario_stage") == "start_contact")
    nominal = copy.deepcopy(motion["usable_seam_start"])
    nominal.orientation = copy.deepcopy(motion["points"][0].orientation)
    bottom = copy.deepcopy(nominal)
    bottom.position.z -= policy["max_descent_m"]
    approach = float(contact.get("tcp_speed_m_s") or 0.0)     # the approach speed of the taught START approach
    contact.update(start_touch_probe=copy.deepcopy(policy), start_touch_nominal=nominal,
                   start_touch_approach_speed_m_s=approach,
                   points=(copy.deepcopy(nominal), bottom), path_kind="scan_start_touch_world_z",
                   tcp_speed_m_s=policy["speed_m_s"], touch_guard=True,
                   continue_after_touch=True, accept_initial_touch=False,
                   allow_initial_touch_motion=False, collision_checking=True)


def seek_start_touch(host, step, execute_requested):
    """Move straight to the reference TCP (START) as the taught approach does, then search World -Z slowly from
    there until contact. A contact before the reference TCP is reached stops the pass (ARC blocked). Standstill
    required, DO0 always off."""
    node = host.node
    policy, nominal = step["start_touch_probe"], step["start_touch_nominal"]
    speed, depth = (float(policy[k]) for k in ("speed_m_s", "max_descent_m"))
    approach = float(step.get("start_touch_approach_speed_m_s") or 0.0)
    if (not all(math.isfinite(v) for v in (speed, depth, approach))
            or not 0 < speed <= .0005 or not 0 < depth <= .010 or approach < 0
            or step.get("planning_group") != "right_manipulator" or not pose_is_valid(nominal)):
        return False, "invalid bounded start-touch policy / nominal pose"
    current = node._current_tcp_pose("right_manipulator")
    if current is None or not pose_is_valid(current):
        return False, "start touch has no valid measured TCP"
    bottom = copy.deepcopy(nominal)
    bottom.position.z -= depth
    probe = copy.deepcopy(step)
    probe.pop("start_touch_probe", None)
    probe.update(touch_guard=True, accept_initial_touch=False, allow_initial_touch_motion=False,
                 collision_checking=True, reuse_approved_plan=False)
    if not execute_requested:
        probe["points"] = (copy.deepcopy(nominal), bottom)
        return node.run_sequence_cartesian_motion(probe, False)
    if host.sequence_stop_requested:
        return False, "start touch canceled before motion"
    if not host.fastech_connected:
        return False, "start touch requires connected Fastech"
    client = host.hicomm_client
    if client is None or not client.connected:
        return False, "start touch requires welder feedback confirming ARC OFF"
    status = client.latest_status()
    if (not isinstance(status, dict) or status.get("arc_ack") is not False
            or status.get("wcr_detected") is not False
            or time.monotonic() - float(status.get("timestamp_monotonic", 0)) > 1.0
            or float(status.get("feedback_current_a", 0)) > 0
            or float(status.get("wire_feed_m_min", 0)) > 0):
        return False, "start touch requires fresh feedback confirming ARC OFF and no wire feed"
    success, message, registration = False, "start touch failed", None

    def guarded_move(motion):
        done, lost = threading.Event(), threading.Event()

        def watch_input():
            while not done.wait(.02 if not lost.is_set() else .1):
                if (host.sequence_stop_requested or not host.fastech_connected
                        or node.node_touch_input_states.get("right") is None):
                    lost.set()
                    node.cancel_active_motion()

        watcher = threading.Thread(target=watch_input, daemon=True)
        watcher.start()
        try:
            outcome = node.run_sequence_cartesian_motion(motion, True)
        finally:
            done.set()
            watcher.join(timeout=.5)
        if (lost.is_set() or host.sequence_stop_requested or not host.fastech_connected
                or node.node_touch_input_states.get("right") is None):
            raise RuntimeError("start touch stopped: STOP or touch input connection lost; ARC blocked")
        return outcome

    try:
        ok, text = host._set_fastech_output_sync(0, True)
        if not ok:
            raise RuntimeError(f"touch enable failed: {text}")
        if not host._interruptible_wait(.1) or host.sequence_stop_requested:
            raise RuntimeError("start touch stopped while enabling sensor")
        if node.node_touch_input_states.get("right") is not False:
            raise RuntimeError("start touch requires known touch input LOW before a fresh contact")
        # 1) all the way to the reference TCP (the taught approach speed, touch-guarded)
        probe.update(points=(copy.deepcopy(nominal),), tcp_speed_m_s=approach,
                     path_kind="start_touch_move_to_reference_tcp", continue_after_touch=False)
        node.touch_guard_edge_pose = None
        ok, text = guarded_move(probe)
        if getattr(node, "touch_guard_edge_pose", None) is not None or node.touch_guard_triggered.is_set():
            raise RuntimeError("contact before the reference TCP was reached; the search must start there - ARC "
                               "blocked (check this pass's START height)")
        if not ok or host.sequence_stop_requested:
            raise RuntimeError(f"start touch move to the reference TCP stopped: {text}")
        if node.node_touch_input_states.get("right") is not False:
            raise RuntimeError("touch input must be LOW at the reference TCP before the slow descent")
        # 2) from the reference TCP: World -Z, slow, until the fresh touch edge
        probe.update(points=(bottom,), tcp_speed_m_s=speed, continue_after_touch=True,
                     path_kind="start_touch_slow_world_z_descent")
        # The runtime resets these when arming its fresh-edge guard as well.
        node.touch_guard_edge_pose = None
        ok, text = guarded_move(probe)
        edge = getattr(node, "touch_guard_edge_pose", None)
        if not ok or host.sequence_stop_requested:
            raise RuntimeError(f"start touch descent stopped: {text}")
        if (not node.touch_guard_triggered.is_set() or not node.touch_guard_stop_complete.is_set()
                or not node.touch_guard_stop_success or edge is None or not pose_is_valid(edge)):
            raise RuntimeError("no confirmed fresh touch within the bounded descent; ARC blocked")
        if (math.hypot(edge.position.x - nominal.position.x, edge.position.y - nominal.position.y) > .0005
                or not nominal.position.z - depth - .0001 <= edge.position.z <= nominal.position.z + .0001):
            raise RuntimeError("touch position is outside the vertical search; ARC blocked")
        dz = float(edge.position.z - nominal.position.z)
        registration = dict(delta_world_m=[0., 0., dz],
                            nominal_start_world_m=[getattr(nominal.position, a) for a in "xyz"],
                            touch_tcp_world_m=[getattr(edge.position, a) for a in "xyz"],
                            speed_m_s=speed, max_descent_m=depth,
                            scope="this pass START/GOAL and weld path", source="production touch rising edge; standstill confirmed")
        success, message = True, f"start touch confirmed; this pass World Z {dz * 1000:+.3f} mm"
    except Exception as error:
        message = str(error)
    finally:
        try:
            off, text = host._set_fastech_output_sync(0, False)
        except Exception as error:
            off, text = False, str(error)
        if not off:
            success, message = False, f"start touch output OFF failed; ARC blocked: {text}"
    if success and not host.sequence_stop_requested:
        step["start_touch_result"] = registration
        return True, message
    return False, message


def register_touch_height(steps, contact):
    """Translate only this scenario's remaining geometry; preserve direction/width and all source poses."""
    if contact.get("start_touch_applied"):
        raise ValueError("start touch already applied to this execution")
    result = contact.get("start_touch_result")
    if not result:
        raise ValueError("no confirmed start-touch result")
    dz = float(result["delta_world_m"][2])
    if not math.isfinite(dz):
        raise ValueError("invalid touch displacement")

    def shifted(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return type(value)(shifted(v) for v in value)
        if not pose_is_valid(value):
            raise ValueError("invalid pose in touched weld path")
        pose = copy.deepcopy(value)
        pose.position.z += dz
        return pose

    scenario = contact["weld_scenario_id"]
    past_contact, patches = False, []
    pose_keys = ("points", "tcp_pose", "lead_start", "lead_end", "usable_seam_start", "usable_seam_goal",
                 "usable_weld_points", "safe_approach", "approach_lead", "start_wait_tcp",
                 "expected_start_tcp", "endpoint")
    for step in steps:
        if step is contact:
            past_contact = True
            continue
        if not past_contact or step.get("weld_scenario_id") != scenario:
            continue
        updates = {key: shifted(step[key]) for key in pose_keys if key in step}
        updates["weld_start_touch"] = copy.deepcopy(result)
        if step.get("type") == "named_pose":
            updates["resolve_target_tcp_ik"] = True
        updates["reuse_approved_plan"] = False
        patches.append((step, updates))
    if not patches:
        raise ValueError("no remaining weld stages for start-touch registration")
    for step, updates in patches:
        step.update(updates)
    contact["start_touch_applied"] = True
    return result
