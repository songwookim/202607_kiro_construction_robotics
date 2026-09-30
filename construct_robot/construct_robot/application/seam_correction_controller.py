"""Seam-correction and Fastech touch-probe workflow, independent of Tk.

Moved from ``WeldActionGui`` without changing semantics: the automatic
four-probe correction (WAIT move -> wall/base probes -> checkpoint -> touch
release -> seam computation -> optional END move), manual touch probes, contact
capture/return callbacks, probe-direction resolution, touch-corrected seam
geometry and START/GOAL endpoint computation with YAML persistence.

``SeamCorrectionController`` holds no state.  The workflow state stays on the
host (today ``WeldActionGui``) because the ROS node callbacks, STOP, the
sequence builder and the torch-cleaner panel share it:
``seam_auto_running``, ``seam_auto_stage_event`` / ``_success`` /
``_expected_kind``, ``seam_auto_returned_kinds``, ``automatic_probe_kind``,
``seam_probe_touches`` / ``_starts`` / ``_stops``, ``computed_seam_endpoints``,
``computed_seam_wait_points``, ``corrected_seam_geometry`` and
``taught_robot_poses``.

Widgets and dialogs are reached only through host hooks called at the
original points (``_set_corner_touch_status``, ``_set_auto_seam_button_enabled``,
``_set_stop_auto_seam_button_enabled``, ``_confirm_automatic_seam_correction``,
``_confirm_touch_probe``, ``_show_teaching_pose``).  Operator settings are still
read from the host's variables with ``.get()``; their UI exception types are
injected as ``ui_error_types``.  Sibling operations go through the host so
node callbacks and host-level overrides keep their entry points.
"""

import copy
import math
import threading
import time

import yaml

from construct_robot.core.cartesian_path_common import (
    _quaternion_rotate_vector,
    linear_pose_waypoints,
    pose_is_valid,
    quaternion_angular_distance,
)
from construct_robot.core.seam_geometry import (
    CORNER_TOUCH_NAMES,
    _axis_unit_vector,
    _pose_position_tuple,
    _unit_vector,
    _vector_dot,
    apply_sensed_seam_orientation,
    compute_corrected_seam_geometry,
    compute_seam_local_frame,
    compute_surface_plane,
    fixed_tilt_wait_reference_poses,
    seam_xy_normal,
)
from construct_robot.core.task_teaching_model import TEACHING_POSES
from construct_robot.io.teaching_yaml import save_initial_state_yaml


class SeamCorrectionController:
    """Seam-correction / touch-probe orchestration against the host's state."""

    def __init__(self, host, *, touch_output_port, ui_error_types=()):
        self.host = host
        self.touch_output_port = touch_output_port
        self.ui_error_types = tuple(ui_error_types)

    def run_automatic_correction(self):
        host = self.host
        if host.seam_auto_running:
            host.error("Automatic seam correction is already running")
            return
        host.pass_probe_touch_yaml_target = None
        if host.keyboard_velocity_arm is not None or host.keyboard_velocity_switching:
            host.error("Disable Keyboard Teaching before automatic seam correction")
            return
        if not host.execution_allowed or not host.robot_connected["right"]:
            host.error("Connect the right robot and enable physical execution")
            return
        if host.planning_group.get() != "right_manipulator":
            host.error("Automatic seam correction currently supports right arm")
            return
        fixed_tilt_mode = host._wait_fixed_tilt_mode_enabled()
        required = (
            ("weld_start_wait", "weld_goal_wait")
            if fixed_tilt_mode
            else ("weld_start_wait", "weld_start", "weld_goal_wait", "weld_end")
        )
        if host.auto_seam_move_to_end_pose.get():
            required += ("weld_finish",)
        missing = [
            TEACHING_POSES[name]
            for name in required
            if host.taught_robot_poses[name] is None
        ]
        if missing:
            host.error("Capture/load required poses: " + ", ".join(missing))
            return
        wrong_group = [
            TEACHING_POSES[name]
            for name in required
            if host.taught_robot_poses[name][0] != "right_manipulator"
        ]
        if wrong_group:
            host.error(
                "These poses belong to another arm: "
                + ", ".join(wrong_group)
            )
            return
        if host.touch_input_states["right"] is None:
            host.error("Fastech DI4 state has not been received yet")
            return
        if host.touch_input_states["right"]:
            host.error("Fastech DI4 is already ON; release it before auto correction")
            return
        if not host._confirm_automatic_seam_correction(fixed_tilt_mode):
            return
        wait_steps = {}
        for wait_name in ("weld_start_wait", "weld_goal_wait"):
            group, names, positions, tcp = host.taught_robot_poses[wait_name]
            endpoint_name = (
                "weld_start" if wait_name == "weld_start_wait" else "weld_end"
            )
            if fixed_tilt_mode:
                endpoint_tcp = fixed_tilt_wait_reference_poses(
                    host.taught_robot_poses["weld_start_wait"][3],
                    host.taught_robot_poses["weld_goal_wait"][3],
                    float(host.weld_fixed_tilt_y_deg.get()),
                    tilt_x_deg=float(host.weld_fixed_tilt_x_deg.get()),
                    tilt_z_deg=float(host.weld_fixed_tilt_z_deg.get()),
                )[
                    0 if wait_name == "weld_start_wait" else 1
                ]
                orientation_source = (
                    f"{TEACHING_POSES[wait_name]} + fixed World XYZ "
                    f"({float(host.weld_fixed_tilt_x_deg.get()):+.1f}°, "
                    f"{float(host.weld_fixed_tilt_y_deg.get()):+.1f}°, "
                    f"{float(host.weld_fixed_tilt_z_deg.get()):+.1f}°)"
                )
            else:
                endpoint_tcp = host.taught_robot_poses[endpoint_name][3]
                orientation_source = TEACHING_POSES[endpoint_name]

            probe_wait_tcp = copy.deepcopy(tcp)
            probe_wait_tcp.orientation = copy.deepcopy(endpoint_tcp.orientation)
            orientation_delta = quaternion_angular_distance(
                tcp.orientation, probe_wait_tcp.orientation
            )
            host.log(
                f"AUTO probe orientation · {TEACHING_POSES[wait_name]} XYZ kept · "
                f"orientation aligned to {orientation_source} · "
                f"Δattitude={math.degrees(orientation_delta):.2f}°"
            )
            wait_steps[wait_name] = {
                "type": "named_pose",
                "pose_name": wait_name,
                "pose_label": TEACHING_POSES[wait_name],
                "planning_group": group,
                "joint_names": tuple(names),
                "positions": tuple(positions),
                "tcp_pose": probe_wait_tcp,
                "velocity_scale": max(
                    0.01,
                    min(1.0, host.velocity_percent.get() / 100.0),
                ),
                "probe_orientation_source": (
                    orientation_source
                ),
                # These are already taught TCP targets. Keep automatic seam
                # correction responsive instead of allowing 5 s × 5 attempts.
                "planning_attempts": 1,
                "planning_time": 1.0,
            }
        workflow = [
            (
                wait_steps["weld_start_wait"],
                ("start_wall", "start_floor"),
            ),
            (
                wait_steps["weld_goal_wait"],
                ("goal_wall", "goal_floor"),
            ),
        ]
        # Every AUTO run is a new measurement session.  Clear *all* derived
        # seam state, including computed endpoints from the previous run.
        host._invalidate_seam_correction_runtime(
            "new AUTO seam-correction session", clear_touches=True
        )
        host.seam_auto_move_to_end_requested = bool(
            host.auto_seam_move_to_end_pose.get()
        )
        host.seam_auto_running = True
        host._set_auto_seam_button_enabled(False)
        host._set_stop_auto_seam_button_enabled(True)
        threading.Thread(
            target=host._automatic_seam_correction_worker,
            args=(tuple(workflow),),
            daemon=True,
        ).start()

    def stop_automatic_correction(self):
        host = self.host
        if not host.seam_auto_running and host.automatic_probe_kind is None:
            host.log("STOP AUTO ignored · automatic seam correction is idle")
            return
        host.seam_auto_running = False
        host.seam_auto_move_to_end_requested = False
        host.seam_auto_expected_kind = None
        host.seam_auto_stage_success = False
        host.seam_auto_stage_event.set()
        host.automatic_probe_kind = None
        host._set_stop_auto_seam_button_enabled(False)
        host._set_corner_touch_status(
            text="STOP AUTO requested · stopping robot motion"
        )
        threading.Thread(
            target=host.node.stop_auto_motion,
            args=("right",),
            daemon=True,
        ).start()

    def automatic_correction_worker(self, workflow):
        host = self.host
        probe_index = 0
        group_total = len(workflow)
        probe_total = sum(len(kinds) for _step, kinds in workflow)
        for group_index, (wait_step, probe_kinds) in enumerate(
            workflow, start=1
        ):
            if not host.seam_auto_running:
                return
            host.post(
                host._set_auto_seam_status,
                f"AUTO GROUP {group_index}/{group_total} · moving to "
                f"{wait_step['pose_label']}",
            )
            success, message = False, "not attempted"
            for attempt in (1,):
                success, message = host.node.run_sequence_named_pose(
                    wait_step, True
                )
                if success or not host.seam_auto_running:
                    break
                host.post(
                    host.log,
                    f"AUTO {wait_step['pose_label']} attempt {attempt} "
                    f"failed · {message}",
                )
                time.sleep(0.5)
            if not host.seam_auto_running:
                return
            if not success:
                host.post(
                    host._finish_automatic_seam_correction,
                    False,
                    f"{wait_step['pose_label']} failed: {message}",
                )
                return
            host.post(
                host.log,
                f"AUTO reached {wait_step['pose_label']} · "
                f"starting {probe_kinds[0]} then {probe_kinds[1]}",
            )
            for kind in probe_kinds:
                if not host.seam_auto_running:
                    return
                probe_index += 1
                host.seam_auto_expected_kind = kind
                host.seam_auto_stage_success = False
                host.seam_auto_stage_event.clear()
                host.post(
                    host._set_auto_seam_status,
                    f"AUTO PROBE {probe_index}/{probe_total} · {kind}",
                )
                host.post(host._launch_automatic_seam_stage, kind)
                if not host.seam_auto_stage_event.wait(timeout=180.0):
                    host.post(
                        host._finish_automatic_seam_correction,
                        False,
                        f"{kind} timed out",
                    )
                    return
                if not host.seam_auto_running:
                    return
                touch_saved = host.seam_probe_touches.get(kind) is not None
                returned = kind in host.seam_auto_returned_kinds
                if not (
                    host.seam_auto_stage_success
                    and touch_saved
                    and returned
                ):
                    host.post(
                        host._finish_automatic_seam_correction,
                        False,
                        f"{kind} incomplete: touch_saved={touch_saved}, "
                        f"returned={returned}",
                    )
                    return
                host.post(
                    host.log,
                    f"AUTO CHECKPOINT {probe_index}/{probe_total} · {kind} · "
                    f"touch_saved={touch_saved} · returned={returned}",
                )
                host.post(
                    host._set_auto_seam_status,
                    f"AUTO {kind} returned · waiting for Fastech DI0 OFF",
                )
                if not host._wait_for_touch_release(timeout=180.0):
                    host.post(
                        host._finish_automatic_seam_correction,
                        False,
                        f"{kind} completed, but Fastech DI0 remained ON",
                    )
                    return
            if group_index == 1 and group_total > 1:
                host.post(
                    host._set_auto_seam_status,
                    "START wall/base complete · next: moving to "
                    "5 · Weld goal wait pose",
                )
        host.post(host._complete_automatic_seam_correction)

    def wait_for_touch_release(self, timeout):
        host = self.host
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not host.touch_input_states.get("right", True):
                return True
            time.sleep(0.02)
        return False

    def launch_automatic_stage(self, kind):
        host = self.host
        host.start_automatic_touch_probe(kind, skip_confirmation=True)
        if host.automatic_probe_kind != kind:
            host._signal_auto_seam_stage(False, kind)

    def signal_automatic_stage(self, success, kind=None):
        host = self.host
        if not host.seam_auto_running:
            return
        if kind is not None and kind != host.seam_auto_expected_kind:
            host.log(
                f"Ignored stale auto probe result for {kind}; "
                f"waiting for {host.seam_auto_expected_kind}"
            )
            return
        host.seam_auto_stage_success = bool(success)
        host.seam_auto_stage_event.set()

    def complete_automatic_correction(self):
        host = self.host
        host.compute_two_touch_seam()
        if not host.corrected_two_touch_seam:
            host._finish_automatic_seam_correction(
                False, "four-touch seam computation failed"
            )
            return
        success = host.path_kind == "di8_four_touch_corrected"
        if not success:
            host._finish_automatic_seam_correction(
                False, "corrected seam could not be adopted"
            )
            return
        if not host.seam_auto_move_to_end_requested:
            host._finish_automatic_seam_correction(
                True,
                "four touches complete; corrected path/YAML adopted; "
                "remaining at GOAL WAIT (automatic END move disabled)",
            )
            return
        finish_data = host.taught_robot_poses.get("weld_finish")
        if finish_data is None or finish_data[0] != "right_manipulator":
            host._finish_automatic_seam_correction(
                False, "7 · Weld end pose is unavailable"
            )
            return
        group, names, joints, tcp = finish_data
        finish_step = {
            "type": "named_pose",
            "pose_name": "weld_finish",
            "pose_label": TEACHING_POSES["weld_finish"],
            "planning_group": group,
            "joint_names": tuple(names),
            "positions": tuple(joints),
            "tcp_pose": copy.deepcopy(tcp),
            "velocity_scale": max(
                0.01, min(1.0, host.velocity_percent.get() / 100.0)
            ),
            "tcp_speed_m_s": 0.0,
            "touch_guard": True,
            "continue_after_touch": False,
            "planning_attempts": 5,
            "planning_time": 5.0,
        }
        host._set_auto_seam_status(
            "CORRECTION SAVED · moving to 7 · Weld end pose"
        )
        threading.Thread(
            target=host._automatic_seam_end_pose_worker,
            args=(finish_step,),
            daemon=True,
        ).start()

    def automatic_end_pose_worker(self, finish_step):
        host = self.host
        if not host.seam_auto_running:
            return
        success, message = host.node.run_sequence_named_pose(
            finish_step, True
        )
        if not host.seam_auto_running:
            return
        host.post(
            host._finish_automatic_seam_correction,
            success,
            (
                "four touches complete; corrected path/YAML adopted; "
                "reached 7 · Weld end pose"
                if success
                else f"correction saved, but Weld end move failed: {message}"
            ),
        )

    def finish_automatic_correction(self, success, message):
        host = self.host
        host.seam_auto_running = False
        host.seam_auto_move_to_end_requested = False
        host.seam_auto_expected_kind = None
        host.seam_auto_stage_event.set()
        host._set_auto_seam_button_enabled(True)
        host._set_stop_auto_seam_button_enabled(False)
        if success:
            host.pipeline_result(f"AUTO SEAM CORRECTION COMPLETE · {message}")
        else:
            host.error(f"Automatic seam correction stopped: {message}")

    def resolve_probe_direction(self, surface, teaching_reference=None):
        """Resolve a configured positive World probe normal and display label."""
        host = self.host
        surface = str(surface).strip().lower()
        if surface == "wall":
            selection = host.wall_probe_axis.get().strip()
            if selection.upper().startswith("AUTO"):
                if teaching_reference is None:
                    teaching_reference = host._ensure_seam_teaching_reference(
                        require_complete=True
                    )
                if teaching_reference is None:
                    raise ValueError("START/GOAL teaching is required for AUTO wall normal")
                direction = seam_xy_normal(
                    teaching_reference["weld_start"][3],
                    teaching_reference["weld_end"][3],
                )
                return direction, (
                    "AUTO seam-normal "
                    f"({direction[0]:+.3f}, {direction[1]:+.3f}, {direction[2]:+.3f})"
                )
            direction = _axis_unit_vector(selection)
            return direction, selection
        if surface == "floor":
            selection = host.floor_probe_axis.get().strip()
            direction = _axis_unit_vector(selection)
            return direction, selection
        raise ValueError(f"unknown probe surface: {surface}")

    def seam_geometry_settings(self, require_teaching=True):
        host = self.host
        teaching_reference = host._ensure_seam_teaching_reference(
            require_complete=require_teaching
        )
        if require_teaching and teaching_reference is None:
            raise ValueError("complete seam teaching reference is unavailable")
        wall_normal, wall_label = host._resolve_probe_direction(
            "wall", teaching_reference
        )
        floor_normal, floor_label = host._resolve_probe_direction(
            "floor", teaching_reference
        )
        return teaching_reference, wall_normal, floor_normal, wall_label, floor_label

    def compute_touch_corrected_geometry(
        self,
        teaching_reference,
        wall_normal,
        floor_normal,
        wall_offset,
        floor_offset,
        *,
        log_debug=False,
    ):
        """Estimate common surface planes and their projected seam segment."""
        host = self.host
        wall_touches = [
            host.seam_probe_touches[name]
            for name in ("start_wall", "goal_wall")
            if host.seam_probe_touches.get(name) is not None
        ]
        floor_touches = [
            host.seam_probe_touches[name]
            for name in ("start_floor", "goal_floor")
            if host.seam_probe_touches.get(name) is not None
        ]
        if not wall_touches or not floor_touches:
            raise ValueError(
                "actual seam geometry needs at least one wall and one floor touch"
            )
        taught_start = teaching_reference["weld_start"][3]
        taught_goal = teaching_reference["weld_end"][3]
        wall_plane = compute_surface_plane(
            wall_normal, wall_touches, wall_offset
        )
        floor_plane = compute_surface_plane(
            floor_normal, floor_touches, floor_offset
        )
        start_tool_z = _quaternion_rotate_vector(
            taught_start.orientation, (0.0, 0.0, 1.0)
        )
        goal_tool_z = _quaternion_rotate_vector(
            taught_goal.orientation, (0.0, 0.0, 1.0)
        )
        approach_reference = tuple(
            start_tool_z[index] + goal_tool_z[index] for index in range(3)
        )
        try:
            approach_reference = _unit_vector(
                approach_reference, "mean taught Tool +Z approach"
            )
        except ValueError:
            approach_reference = _unit_vector(
                start_tool_z, "taught START Tool +Z approach"
            )
        geometry = compute_corrected_seam_geometry(
            taught_start,
            taught_goal,
            wall_plane,
            floor_plane,
            approach_reference,
        )
        start_wait_data = host.taught_robot_poses.get("weld_start_wait")
        if start_wait_data is not None and pose_is_valid(start_wait_data[3]):
            wait_outward = tuple(
                getattr(start_wait_data[3].position, axis)
                - getattr(geometry.start.position, axis)
                for axis in ("x", "y", "z")
            )
            # The taught WAIT is the strongest available sign convention for
            # "away from workpiece".  Only its sign is used; e_a remains the
            # orthogonal wall/floor-normal bisector.
            if math.sqrt(_vector_dot(wait_outward, wait_outward)) > 1e-4:
                geometry.d_real, geometry.e_w, geometry.e_a = (
                    compute_seam_local_frame(
                        geometry.d_real,
                        geometry.wall_normal,
                        geometry.floor_normal,
                        wait_outward,
                    )
                )
        host.corrected_seam_geometry = geometry
        if log_debug:
            host._log_corrected_seam_geometry(geometry)
        return geometry

    def log_corrected_geometry(self, geometry):
        host = self.host
        vector = lambda values: "(" + ", ".join(
            f"{float(value):+.6f}" for value in values
        ) + ")"
        position = lambda pose: vector(_pose_position_tuple(pose))
        angle_deg = math.degrees(math.acos(max(
            -1.0, min(1.0, _vector_dot(geometry.d_teach, geometry.d_real))
        )))
        host.log(
            "SEAM GEOMETRY DEBUG · "
            f"d_teach={vector(geometry.d_teach)} · "
            f"wall_normal={vector(geometry.wall_normal)}, "
            f"c_w={geometry.wall_plane_value:+.6f} · "
            f"floor_normal={vector(geometry.floor_normal)}, "
            f"c_f={geometry.floor_plane_value:+.6f} · "
            f"d_real={vector(geometry.d_real)} · P0={vector(geometry.origin)}"
        )
        host.log(
            "SEAM PROJECTION DEBUG · "
            f"P_start_teach={vector(geometry.taught_start)} · "
            f"P_start_corrected={position(geometry.start)} · "
            f"P_goal_teach={vector(geometry.taught_goal)} · "
            f"P_goal_corrected={position(geometry.goal)}"
        )
        host.log(
            "SEAM FRAME DEBUG · "
            f"e_a={vector(geometry.e_a)} · e_w={vector(geometry.e_w)} · "
            f"length_before={geometry.length_before:.6f} m · "
            f"length_after={geometry.length_after:.6f} m · "
            f"angle(d_teach,d_real)={angle_deg:.4f} deg · "
            f"dot(d_real,e_a)={_vector_dot(geometry.d_real, geometry.e_a):+.3e} · "
            f"dot(d_real,e_w)={_vector_dot(geometry.d_real, geometry.e_w):+.3e} · "
            f"dot(e_a,e_w)={_vector_dot(geometry.e_a, geometry.e_w):+.3e}"
        )

    def start_touch_probe(self, kind, skip_confirmation=False):
        host = self.host
        if not host.seam_auto_running:
            host.pass_probe_touch_yaml_target = None
        if kind not in CORNER_TOUCH_NAMES:
            host.error(f"Unknown touch probe kind: {kind}")
            return
        if host.automatic_probe_kind is not None:
            host.error("Another Fastech DI4 touch probe is already active")
            return
        if host.keyboard_velocity_arm is not None or host.keyboard_velocity_switching:
            host.error("Disable Keyboard Teaching before touch probing")
            return
        if host.planning_group.get() != "right_manipulator":
            host.error("Automatic Fastech DI4 seam probing currently supports the right arm")
            return
        if not host.execution_allowed or not host.robot_connected["right"]:
            host.error("Connect the right robot and enable physical execution")
            return
        if host.touch_input_states["right"] is None:
            host.error("Fastech DI4 state has not been received yet")
            return
        if host.touch_input_states["right"]:
            host.error("Fastech DI4 is already ON; release the touch signal before probing")
            return
        try:
            distance = float(host.touch_probe_distance_mm.get()) * 0.001
            speed = float(host.touch_probe_speed_percent.get()) / 100.0
            settle = float(host.touch_settle_seconds.get())
        except (ValueError, *self.ui_error_types):
            host.error("Touch probe distance, speed, or settle time is invalid")
            return
        if not 0.001 <= distance <= 0.200:
            host.error("Touch probe max travel must be in 1..200 mm")
            return
        if not 0.001 <= speed <= 0.10:
            host.error("Touch probe speed must be in 0.1..10%")
            return
        if not 0.2 <= settle <= 5.0:
            host.error("Touch settle time must be in 0.2..5.0 seconds")
            return
        surface = kind.rsplit("_", 1)[1]
        try:
            teaching_reference = host._ensure_seam_teaching_reference(
                require_complete=True
            )
            if teaching_reference is None:
                return
            direction, direction_label = host._resolve_probe_direction(
                surface, teaching_reference
            )
        except ValueError as error:
            host.error(f"Cannot resolve touch probe direction: {error}")
            return
        sign = (
            host.wall_probe_sign.get()
            if surface == "wall"
            else host.floor_probe_sign.get()
        )
        sign_scale = 1.0 if sign == "+" else -1.0
        signed_direction = tuple(sign_scale * value for value in direction)
        if not skip_confirmation:
            if not host._confirm_touch_probe(
                kind, direction_label, sign, signed_direction, distance, speed
            ):
                return
        touch_enabled, touch_message = host._set_fastech_output_sync(
            self.touch_output_port, True
        )
        if not touch_enabled:
            host.error(
                "Cannot enable touch sensing on Fastech DO0: "
                f"{touch_message}"
            )
            return
        if host.node.node_touch_input_states.get("right"):
            host.error(
                "Fastech DI0 became ON while enabling touch sensing; "
                "probe motion was not started"
            )
            return
        host.log("Fastech DO0 touch sensing enabled · readback confirmed")
        if host.hicomm_client is not None:
            host.hicomm_client.set_arc(False)
        host.automatic_probe_kind = kind
        host._set_corner_touch_status(
            text=(
                f"PROBING {kind.upper()} · {direction_label} {sign} · "
                f"v=({signed_direction[0]:+.2f}, {signed_direction[1]:+.2f}, "
                f"{signed_direction[2]:+.2f}) · waiting for Fastech DI0"
            )
        )
        threading.Thread(
            target=host.node.execute_touch_probe,
            args=(
                host.planning_group.get(),
                kind,
                signed_direction,
                distance,
                speed,
                0.001,
            ),
            daemon=True,
        ).start()

    def touch_probe_failed(self, message):
        host = self.host
        kind = host.automatic_probe_kind
        host.automatic_probe_kind = None
        host.node.clear_touch_probe()
        host._signal_auto_seam_stage(False, kind)
        host.error(f"{kind or 'touch'} probe failed: {message}")

    def compute_seam_endpoint(self, endpoint, update_wait_joints=True):
        """Compute START or GOAL independently from its two Fastech DI0 poses."""
        host = self.host
        teaching_reference = host._ensure_seam_teaching_reference(
            require_complete=True
        )
        if teaching_reference is None:
            return None
        endpoint = str(endpoint).strip().lower()
        if endpoint not in ("start", "goal"):
            host.error(f"Unknown seam endpoint: {endpoint}")
            return None
        wall = host.seam_probe_touches.get(f"{endpoint}_wall")
        floor = host.seam_probe_touches.get(f"{endpoint}_floor")
        missing = [
            name
            for name, pose in (("wall", wall), ("floor", floor))
            if pose is None
        ]
        if missing:
            host.error(
                f"Complete {endpoint.upper()} two-pose sensing first: "
                + ", ".join(missing)
            )
            return None
        pose_name = "weld_start" if endpoint == "start" else "weld_end"
        wait_name = (
            "weld_start_wait" if endpoint == "start" else "weld_goal_wait"
        )
        endpoint_data = host.taught_robot_poses.get(pose_name)
        wait_data = host.taught_robot_poses.get(wait_name)
        if endpoint_data is None and host._wait_fixed_tilt_mode_enabled():
            # The corrected endpoint will immediately be resolved through IK;
            # the WAIT joints are only its initial seed/storage scaffold.
            endpoint_data = copy.deepcopy(wait_data)
        if endpoint_data is None or wait_data is None:
            host.error(
                f"Capture/load {TEACHING_POSES[pose_name]} and "
                f"{TEACHING_POSES[wait_name]} first"
            )
            return None
        if endpoint_data[0] != wait_data[0]:
            host.error(
                f"{TEACHING_POSES[pose_name]} and "
                f"{TEACHING_POSES[wait_name]} belong to different arms"
            )
            return None
        try:
            wall_offset = 0.0
            floor_offset = 0.0
            (
                _reference,
                wall_normal,
                floor_normal,
                wall_label,
                floor_label,
            ) = host._seam_geometry_settings(require_teaching=True)
            geometry = host._compute_touch_corrected_seam_geometry(
                teaching_reference,
                wall_normal,
                floor_normal,
                wall_offset,
                floor_offset,
                log_debug=all(
                    host.seam_probe_touches.get(name) is not None
                    for name in CORNER_TOUCH_NAMES
                ),
            )
            point = copy.deepcopy(
                geometry.start if endpoint == "start" else geometry.goal
            )
            wait_point = copy.deepcopy(wait_data[3])
        except (ValueError, *self.ui_error_types) as error:
            host.error(f"{endpoint.upper()} two-pose computation failed: {error}")
            return None
        updates = [(pose_name, endpoint_data, point)]
        try:
            saved_paths = []
            for teaching_name, stored, corrected_tcp in updates:
                planning_group, joint_names, positions, _old_tcp = stored
                yaml_path = host._initial_state_yaml_path(
                    planning_group, teaching_name
                )
                save_initial_state_yaml(
                    yaml_path,
                    planning_group,
                    joint_names,
                    positions,
                    corrected_tcp,
                )
                saved_paths.append(yaml_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            host.error(
                f"{endpoint.upper()} computed, but teaching YAML update "
                f"failed: {error}"
            )
            return None
        host.log(
            f"{endpoint.upper()} TCP YAML WRITE VERIFIED · "
            + " · ".join(str(path) for path in saved_paths)
        )
        for teaching_name, stored, corrected_tcp in updates:
            planning_group, joint_names, positions, _old_tcp = stored
            host.taught_robot_poses[teaching_name] = (
                planning_group,
                joint_names,
                positions,
                copy.deepcopy(corrected_tcp),
            )
        host.computed_seam_endpoints[endpoint] = copy.deepcopy(point)
        host.computed_seam_wait_points[endpoint] = copy.deepcopy(wait_point)
        host._publish_touch_geometry_if_ready(endpoint, point)
        # Make the automatically changed wait pose immediately visible in the
        # Named Robot Pose Teaching panel.  The stored joint seed remains the
        # taught one; the corrected TCP is used by the sensed weld workflow.
        host._show_teaching_pose(TEACHING_POSES[wait_name])
        yaw_commit_done = False
        if all(host.computed_seam_endpoints.values()):
            try:
                teaching_reference = host._ensure_seam_teaching_reference(
                    require_complete=True
                )
                if teaching_reference is None:
                    return None
                count = int(host.corner_touch_count.get())
                corrected_start, corrected_goal, delta_yaw, orientation_label = (
                    apply_sensed_seam_orientation(
                        teaching_reference["weld_start"][3],
                        teaching_reference["weld_end"][3],
                        host.computed_seam_endpoints["start"],
                        host.computed_seam_endpoints["goal"],
                        host.seam_orientation_mode.get(),
                    )
                )
                host.computed_seam_endpoints = {
                    "start": corrected_start,
                    "goal": corrected_goal,
                }
                if host.corrected_seam_geometry is not None:
                    host.corrected_seam_geometry.start = copy.deepcopy(
                        corrected_start
                    )
                    host.corrected_seam_geometry.goal = copy.deepcopy(
                        corrected_goal
                    )
                host._update_seam_yaw_status(
                    host.computed_seam_endpoints["start"],
                    host.computed_seam_endpoints["goal"],
                )
                # Both wait poses are fixed, manually taught standby poses.
                # Cartesian transitions connect them to yaw-corrected seam
                # endpoints with linear XYZ and orientation SLERP.
                host.computed_seam_wait_points = {
                    "start": copy.deepcopy(
                        host.taught_robot_poses["weld_start_wait"][3]
                    ),
                    "goal": copy.deepcopy(
                        host.taught_robot_poses["weld_goal_wait"][3]
                    ),
                }
                preview = linear_pose_waypoints(
                    corrected_start,
                    corrected_goal,
                    count,
                )
                host.corrected_two_touch_seam = copy.deepcopy(preview)
                host.node.publish_points(preview, host.show_path.get())
                host.correct_two_touch_seam()
                yaw_commit_done = True
                host.log(
                    "Both endpoints ready · "
                    f"orientation={orientation_label} · "
                    f"World Δyaw={math.degrees(delta_yaw):+.3f}° · "
                    "both wait poses kept as taught standby"
                )
            except (ValueError, *self.ui_error_types) as error:
                host.error(f"Endpoint preview failed: {error}")
                return None
        values = host._pose_values(point)
        wait_values = host._pose_values(wait_point)
        wall_values = host._pose_values(wall)
        floor_values = host._pose_values(floor)
        taught_values = host._pose_values(endpoint_data[3])
        correction_mm = tuple(
            (values[index] - taught_values[index]) * 1000.0
            for index in range(3)
        )
        host.log(
            f"{endpoint.upper()} SEAM XYZ INPUT · "
            f"wall=({wall_values[0]:.6f}, {wall_values[1]:.6f}, "
            f"{wall_values[2]:.6f}) · "
            f"floor=({floor_values[0]:.6f}, {floor_values[1]:.6f}, "
            f"{floor_values[2]:.6f}) m"
        )
        host.log(
            f"{endpoint.upper()} SEAM XYZ RESULT · "
            f"computed=({values[0]:.6f}, {values[1]:.6f}, "
            f"{values[2]:.6f}) m · old teaching="
            f"({taught_values[0]:.6f}, {taught_values[1]:.6f}, "
            f"{taught_values[2]:.6f}) m · correction="
            f"({correction_mm[0]:+.3f}, {correction_mm[1]:+.3f}, "
            f"{correction_mm[2]:+.3f}) mm · "
            + (
                "orientation=yaw-corrected after both endpoints"
                if yaw_commit_done
                else "orientation=temporary teaching value"
            )
        )
        host._set_corner_touch_status(
            text=(
                f"{endpoint.upper()} computed · wall={wall_label} · "
                f"base={floor_label} · "
                f"seam=({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}) · "
                f"fixed wait=({wait_values[0]:.4f}, {wait_values[1]:.4f}, "
                f"{wait_values[2]:.4f})"
            )
        )
        if update_wait_joints and not yaw_commit_done:
            target_labels = " and ".join(
                TEACHING_POSES[teaching_name]
                for teaching_name, _stored, _tcp in updates
            )
            host.pipeline_waiting(
                f"{endpoint.upper()} TCP computed · resolving MoveIt IK for "
                f"{target_labels}"
            )
            ik_targets = tuple(
                (
                    endpoint,
                    stored[0],
                    copy.deepcopy(corrected_tcp),
                    tuple(stored[1]),
                    teaching_name,
                )
                for teaching_name, stored, corrected_tcp in updates
            )
            threading.Thread(
                target=host.node.resolve_tcp_joint_states,
                args=(ik_targets,),
                daemon=True,
            ).start()
        else:
            host.pipeline_result(
                f"{endpoint.upper()} TWO-POSE TCP APPLIED · "
                f"{TEACHING_POSES[pose_name]} TCP="
                f"({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}) · "
                f"{TEACHING_POSES[wait_name]} TCP="
                f"({wait_values[0]:.4f}, {wait_values[1]:.4f}, "
                f"{wait_values[2]:.4f}) · YAML saved"
            )
        # Geometry plots are opened only by the explicit GUI button.
        return point

    def publish_touch_geometry_if_ready(self, endpoint, seam_point=None):
        host = self.host
        wall = host.seam_probe_touches.get(f"{endpoint}_wall")
        floor = host.seam_probe_touches.get(f"{endpoint}_floor")
        if wall is None or floor is None:
            return False
        try:
            if seam_point is None:
                teaching_reference = host._ensure_seam_teaching_reference(
                    require_complete=True
                )
                if teaching_reference is None:
                    return False
                (
                    _reference,
                    wall_normal,
                    floor_normal,
                    _wall_label,
                    _floor_label,
                ) = host._seam_geometry_settings(require_teaching=True)
                geometry = host._compute_touch_corrected_seam_geometry(
                    teaching_reference,
                    wall_normal,
                    floor_normal,
                    0.0,
                    0.0,
                )
                seam_point = (
                    geometry.start if endpoint == "start" else geometry.goal
                )
            host.node.publish_touch_geometry(
                endpoint, wall, floor, seam_point
            )
        except (ValueError, *self.ui_error_types) as error:
            host.error(f"{endpoint.upper()} touch visualization failed: {error}")
            return False
        return True

    def apply_touch_edge_capture(
        self,
        pose,
        planning_group,
        kind,
        probe_start,
    ):
        """Persist the Fastech DI0-edge pose before controlled-stop completion."""
        host = self.host
        if kind not in CORNER_TOUCH_NAMES:
            host.error(f"Unknown Fastech DI0 edge capture kind: {kind}")
            return
        host.seam_probe_touches[kind] = copy.deepcopy(pose)
        host.seam_probe_starts[kind] = copy.deepcopy(probe_start)
        host.seam_probe_stops[kind] = None
        host._persist_seam_touch_yaml(planning_group, "EDGE CONTACT")

    def apply_touch_capture(
        self,
        pose,
        planning_group,
        source,
        probe_start=None,
        stopped_pose=None,
        cancel_event=None,
    ):
        host = self.host
        host.last_touch_pose = copy.deepcopy(pose)
        values = host._pose_values(pose)
        host.pipeline_result(
            f"TOUCH TCP CAPTURED · {planning_group} · "
            f"World XYZ=({values[0]:.6f}, {values[1]:.6f}, "
            f"{values[2]:.6f}) m"
        )
        if source.startswith("automatic probe:"):
            kind = source.split(":", 1)[1]
            host.seam_probe_touches[kind] = copy.deepcopy(pose)
            host.seam_probe_starts[kind] = (
                copy.deepcopy(probe_start) if probe_start is not None else None
            )
            host.seam_probe_stops[kind] = (
                copy.deepcopy(stopped_pose) if stopped_pose is not None else None
            )
            touch_yaml = host._persist_seam_touch_yaml(
                planning_group, "STOPPED-POSE UPDATE"
            )
            if touch_yaml is not None:
                endpoint = kind.split("_", 1)[0]
                wall = host.seam_probe_touches.get(f"{endpoint}_wall")
                floor = host.seam_probe_touches.get(f"{endpoint}_floor")
                if wall is not None and floor is not None:
                    delta_x_mm = (
                        wall.position.x - floor.position.x
                    ) * 1000.0
                    delta_y_mm = (
                        wall.position.y - floor.position.y
                    ) * 1000.0
                    host.log(
                        f"{endpoint.upper()} TOUCH PAIR CHECK · "
                        f"wall-floor ΔX={delta_x_mm:+.3f} mm · "
                        f"ΔY={delta_y_mm:+.3f} mm · "
                        f"seam-axis coordinate uses pair mean"
                    )
                    host._publish_touch_geometry_if_ready(endpoint)
            host.automatic_probe_kind = None
            # Contact is complete.  The following motion is a deliberate
            # retract and must not be treated as the same active touch probe.
            host.node.clear_touch_probe(cancel_return=False)
            completed = [
                name
                for name, value in host.seam_probe_touches.items()
                if value is not None
            ]
            host._set_corner_touch_status(
                text=(
                    f"Fastech DI0 {kind} touch saved · {len(completed)}/4 · "
                    "returning to probe start"
                )
            )
            host.log(f"Automatic Fastech DI0 {kind} touch stored")
            if probe_start is not None:
                try:
                    settle_seconds = max(
                        0.2,
                        min(5.0, float(host.touch_settle_seconds.get())),
                    )
                except (ValueError, *self.ui_error_types):
                    settle_seconds = 0.7
                threading.Thread(
                    target=host.node.return_touch_probe,
                    args=(
                        planning_group,
                        copy.deepcopy(stopped_pose or pose),
                        copy.deepcopy(probe_start),
                        max(
                            0.001,
                            min(
                                0.10,
                                float(host.touch_probe_speed_percent.get())
                                / 100.0,
                            ),
                        ),
                        0.001,
                        kind,
                        settle_seconds,
                        cancel_event,
                    ),
                    daemon=True,
                ).start()
            return
        if host.touch_sensing_enabled.get() or source.startswith(
            "manual corner capture:"
        ):
            host._record_corner_touch(pose, source)

    def touch_probe_return_finished(self, success, message, probe_kind):
        host = self.host
        if success:
            host.seam_auto_returned_kinds.add(probe_kind)
            completed = [
                name
                for name, value in host.seam_probe_touches.items()
                if value is not None
            ]
            host._set_corner_touch_status(
                text=(
                    f"Probe returned · {len(completed)}/4 captured: "
                    f"{', '.join(completed) or 'none'}"
                )
            )
            host.pipeline_result("Fastech DI0 touch captured and probe start restored")
            host._signal_auto_seam_stage(True, probe_kind)
        else:
            host._signal_auto_seam_stage(False, probe_kind)
            host.error(f"Touch captured, but probe return failed: {message}")
