"""Sequence Builder execution orchestration, independent of Tk.

``SequenceExecutor`` runs a frozen ``SequenceModel`` execution snapshot: it
groups rows into parallel slots, runs each slot's members on their own
threads, dispatches each step to the runtime, and performs the failure/STOP
cleanup.  It was moved from ``WeldActionGui`` without changing semantics.

Safety-relevant state stays on the *host* (today ``WeldActionGui``) because
the host's ARC, TCP-recording, torch-cleaner and STOP code share it: the
``sequence_stop_requested`` flag, the ARC/motion ``threading.Event`` objects
and success flags, the weld feedback record and its lock, the Hi-COMM client
and the ROS node.  The executor reaches all of it through the host, and calls
its sibling operations (``_sequence_worker_body``, ``_run_sequence_step``,
``_interruptible_wait``) through the host too, so existing host-level
overrides keep working.  The executor holds no state of its own.

Host interface used here (duck-typed; see ``SequenceRuntimeHost``).
"""

import copy
from collections import Counter, defaultdict
import threading
import time
import math
from pathlib import Path
from typing import Protocol

from construct_robot.core.cartesian_path_common import pose_is_valid
from construct_robot.core.sequence_model import SequenceModel, validate_managed_weld_sequence
from construct_robot.core.weld_config import validate_digital_weld_settings
from construct_robot.io.hicomm_welder import BIT_GAS, BIT_FORWARD, BIT_REVERSE
from construct_robot.io.scan_cycle_log import ScanCycleLog, record_cycle_event

# Largest joint error between where the robot stopped after the contact
# approach and the start the early weld plan assumed (~0.7 mm at the TCP).
EARLY_PREPLAN_START_TOLERANCE_RAD = 0.001
EARLY_PREPLAN_STAGES = ("weld_wait", "start_wait", "start_safe", "start_contact")


class SequenceRuntimeHost(Protocol):
    """Everything :class:`SequenceExecutor` reads, writes or calls on its host."""

    # Shared execution state (owned by the host; read/written here).
    sequence_model: SequenceModel
    sequence_running: bool           # host view of sequence_model.running
    sequence_stop_requested: bool
    weld_motion_done_event: threading.Event
    weld_motion_success: bool
    weld_arc_established_event: threading.Event
    weld_arc_on_done_event: threading.Event
    weld_arc_on_success: bool
    weld_feedback_lock: object
    _weld_feedback_stopped: bool
    _sequence_fake_arc_snapshot: bool
    fake_arc_enabled: object          # ``.get()`` -> live FAKE ARC toggle
    fastech_connected: bool
    robot_connected: dict
    execution_allowed: bool
    hicomm_connected: bool
    hicomm_client: object             # Hi-COMM client or None
    node: object                      # WeldGuiNode (motion / legacy DO / cancel)

    # Operator feedback (marshalled to the UI thread by ``post``).

    def post(self, callback, *args): ...

    def log(self, message): ...

    def error(self, message): ...

    def _set_sequence_status(self, text): ...

    def _sequence_finished(self, success, message): ...

    # Execution entry points that delegate back to this executor.

    def _sequence_worker(self, steps, indices, execute_requested): ...

    def _sequence_worker_body(self, steps, indices, execute_requested): ...

    def _run_sequence_step(self, step, execute_requested): ...

    def _interruptible_wait(self, seconds): ...

    # Weld feedback, ARC, crater and Fastech runtime operations.

    def _begin_weld_feedback_record(self, settings, execution_conditions): ...

    def _finish_weld_feedback_record(self, reason, final_status=None): ...

    def _record_actual_tcp_until_motion_done(self, weld_motion_step): ...

    def _pending_weld_final_status(self): ...

    def _mark_weld_motion_timing(self, event): ...

    def _execute_hicomm_weld(self, command, settings, execution_conditions=None): ...

    def _execute_triggered_arc_off(self, step): ...

    def _execute_software_crater(self, step): ...

    def _software_crater_restore(self, settings): ...

    def _execute_custom_hot_start(self, step): ...

    def _set_fastech_output_sync(self, port, enabled): ...


def pose_execution_conditions(pose):
    if pose is None or not pose_is_valid(pose):
        return None
    return {
        "position_m": {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
        },
        "orientation_xyzw": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
    }


def record_step_conditions(steps, indices):
    """Per-step part of the execution-conditions snapshot (weld feedback log)."""
    recorded_steps = []
    for stored_index, step in zip(indices, steps):
        condition = {
            "sequence_number": int(stored_index) + 1,
            "type": step.get("type"),
        }
        for key in (
            "parallel_slot",
            "duration",
            "planning_group",
            "velocity_scale",
            "tcp_speed_m_s",
            "interpolation_step",
            "path_kind",
            "pose_name",
            "pose_label",
            "touch_guard",
            "continue_after_touch",
            "accept_initial_touch",
            "start_touch_probe",
            "weld_start_touch",
            "allow_initial_touch_motion",
            "command",
            "port",
            "value",
            "enabled",
            "direction",
            "seconds",
            "weld_scenario_id",
            "weld_scenario_stage",
            "lead_in_mm",
            "lead_out_mm",
            "linear_motion_profile",
            "path_to_seam_speed_factor",
            "trigger_before_goal",
            "arc_off_delay_s",
            "waypoint_hold_s",
            "weld_weave_enabled",
            "weld_weave_pattern",
            "weld_weave_amplitude_mm",
            "weld_weave_pitch_mm",
            "weld_weave_cycles",
            "weld_weave_actual_pitch_mm",
            "weld_weave_crescent_bulge_mm",
            "weld_weave_left_dwell_s",
            "weld_weave_right_dwell_s",
            "weld_weave_axis",
            "weld_weave_reference",
            "weld_weave_transverse_vector",
            "weld_plate_normal_world",
            "joint1_rad",
            "joint2_rad",
            "cleaner_source_index",
            "resolve_target_tcp_ik",
            "work_cycle_id",
            "work_cycle_number",
            "fake_arc_required",
            "scan_weld_cycle_id",
            "scan_cycle_phase",
            "scan_cycle_pass",
            "scan_cycle_run_dir",
            "weld_tcp_speed_mm_s",
            "spray_kind",
            "distance_m",
            "radius_m",
            "unique_points",
            "closed",
            "face_center",
        ):
            if key in step:
                condition[key] = step[key]
        if step.get("settings") is not None:
            condition["settings"] = copy.deepcopy(step["settings"])
        if step.get("scan") is not None:
            condition["scan"] = copy.deepcopy(step["scan"])
        if step.get("type") == "motion":
            points = step.get("points", ())
            condition["waypoint_count"] = len(points)
            condition["waypoints"] = [
                pose_execution_conditions(pose) for pose in points
            ]
            for pose_key in (
                "lead_start",
                "usable_seam_start",
                "usable_seam_goal",
                "lead_end",
            ):
                if pose_key in step:
                    condition[pose_key] = pose_execution_conditions(
                        step[pose_key]
                    )
        elif step.get("type") == "named_pose":
            condition["target_tcp"] = pose_execution_conditions(
                step.get("tcp_pose")
            )
            condition["joint_names"] = list(step.get("joint_names", ()))
            condition["joint_positions_rad"] = [
                float(value) for value in step.get("positions", ())
            ]
        elif step.get("type") in ("dual_arm_pose", "spray_motion"):
            condition["joint_names"] = list(step["joint_names"])
            condition["joint_positions_rad"] = list(step["positions"])
        elif step.get("type") == "planned_trajectory":
            condition["trajectory_segments"] = len(
                step.get("trajectories", ())
            )
            condition["trajectory_point_count"] = int(
                step.get("point_count", 0)
            )
            condition["required_arms"] = list(
                step.get("required_arms", ())
            )
        recorded_steps.append(condition)
    return recorded_steps


def is_work_cycle(steps):
    return any(step.get("work_cycle_id") for step in steps)


def attach_execution_conditions(steps, execution_conditions):
    """Give every ARC ON row its own copy of the Execute-time snapshot."""
    for step in steps:
        if (
            (step.get("type") == "digital_weld" and step.get("command") == "on")
            or step.get("type") == "scan_weld_pass"
        ):
            step["execution_conditions"] = copy.deepcopy(
                execution_conditions
            )
            if step.get("scan_weld_cycle_id") and step.get("type") == "digital_weld":
                effective = step["execution_conditions"]
                effective["scan_cycle_pass"] = step.get("scan_cycle_pass")
                motion = next((row for row in steps
                               if row.get("weld_scenario_id") == step.get("weld_scenario_id")
                               and row.get("weld_scenario_stage") == "weld_motion"), {})
                for key, value in motion.items():
                    if key.startswith(("weld_", "fixed_world_")) or key in (
                        "seam_orientation_mode", "lead_in_mm", "lead_out_mm", "safe_approach_mm", "pre_start_lead_mm",
                    ):
                        effective[key] = copy.deepcopy(value)


def contains_weld_command(steps):
    return any(
        step["type"] in ("digital_weld", "gas", "software_crater", "custom_hot_start", "scan_weld_pass") for step in steps
    )


class SequenceExecutor:
    """Run Sequence Builder snapshots against the host's runtime operations."""

    def __init__(self, host, *, touch_io_backend):
        self.host = host
        self.touch_io_backend = touch_io_backend

    def execution_preflight_error(self, steps, work_cycle):
        """Return the first reason physical execution must not start, or None."""
        host = self.host
        requires_fastech = any(
            step.get("touch_guard", False)
            or step.get("io_backend") == self.touch_io_backend
            for step in steps
        )
        if requires_fastech and not host.fastech_connected:
            return (
                "Connect Fastech Ethernet before executing touch-sensing "
                "or Fastech output steps"
            )
        required_arms = set()
        for step in steps:
            if step["type"] == "planned_trajectory":
                required_arms.update(step.get("required_arms", ()))
            elif step["type"] in ("motion", "named_pose", "spray_motion", "software_crater", "custom_hot_start", "scan_weld_pass"):
                required_arms.add(
                    step["planning_group"].removesuffix("_manipulator")
                )
            elif step["type"] in ("precision_sensing_scan", "hongik_post_weld_scan", "post_weld_scan"):
                required_arms.add("right")
            elif step["type"] == "dual_arm_pose":
                required_arms.update(("left", "right"))
            elif (
                step["type"] == "digital_output"
                and step.get("io_backend") != self.touch_io_backend
            ):
                required_arms.add("right")
        disconnected = [
            arm for arm in sorted(required_arms)
            if not host.robot_connected.get(arm, False)
        ]
        if not host.execution_allowed or disconnected:
            return (
                "Physical execution is unavailable or a required robot is "
                f"disconnected: {', '.join(disconnected) or 'execution disabled'}"
            )
        if any(step["type"] == "wire_feed" for step in steps) and not host.hicomm_connected:
            return "Connect Hi-COMM before wire feed"
        if contains_weld_command(steps) and not work_cycle and (
            not host.hicomm_connected
        ):
            return "Connect Hi-COMM"
        # Retained from the original pre-flight: it also rejects a malformed
        # D-WELD row (missing "command") before anything is started.
        any(
            step["type"] == "digital_weld"
            and step["command"] == "on"
            for step in steps
        )
        motion_counts = Counter(
            step.get("parallel_slot", local_index + 1)
            for local_index, step in enumerate(steps)
            if step["type"] in (
                "motion", "named_pose", "planned_trajectory", "dual_arm_pose", "spray_motion",
                "precision_sensing_scan", "hongik_post_weld_scan", "post_weld_scan", "scan_weld_pass",
            )
        )
        duplicate_motion_slots = [
            slot for slot, count in motion_counts.items() if count > 1
        ]
        if duplicate_motion_slots:
            return (
                "Only one robot motion is allowed in each parallel slot: "
                + ", ".join(map(str, duplicate_motion_slots))
            )
        return None

    def begin(self, steps, indices, execute_requested, fake_arc_snapshot):
        """Freeze run state on the host and start the sequence worker thread."""
        host = self.host
        host._sequence_fake_arc_snapshot = fake_arc_snapshot
        with host.weld_feedback_lock:
            host._weld_feedback_stopped = False
        host.sequence_model.start(indices, execute_requested)
        host.sequence_stop_requested = False
        mode = "EXECUTE" if execute_requested else "PLAN"
        host._set_sequence_status(
            f"{mode} running · {len(steps)} step(s)"
        )
        threading.Thread(
            target=host._sequence_worker,
            args=(steps, indices, execute_requested),
            daemon=True,
        ).start()

    def interruptible_wait(self, seconds):
        host = self.host
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if host.sequence_stop_requested:
                return False
            time.sleep(min(0.05, deadline - time.monotonic()))
        return True

    def fake_arc(self):
        host = self.host
        # Tk variable reads from workers wait for the GUI event loop. Freeze
        # this operator setting at Execute rather than at the ARC boundary.
        if getattr(host, "sequence_running", False) and hasattr(host, "_sequence_fake_arc_snapshot"):
            return host._sequence_fake_arc_snapshot
        return bool(host.fake_arc_enabled.get())

    def run_worker(self, steps, indices, execute_requested):
        host = self.host
        host._scan_cycle_journal = None
        host._scan_cycle_log_error = None
        try:
            if any(step.get("type") in ("precision_sensing_scan", "scan_weld_pass") for step in steps):
                from construct_robot.application.scan_weld_cycle import validate_scan_weld_cycle
                validate_scan_weld_cycle(steps)
                conditions = next((step.get("execution_conditions") for step in steps
                                   if step.get("execution_conditions")), {})
                initial_stage_type = steps[0]["type"]
                if execute_requested:
                    directory = (host._weld_feedback_directory() if hasattr(host, "_weld_feedback_directory")
                                 else Path.home() / "ros2_ws" / "weld_feedback")
                    host._scan_cycle_journal = ScanCycleLog(
                        Path(directory), record_step_conditions(steps, indices),
                        conditions, self.fake_arc(),
                    )
                    conditions = dict(conditions, scan_cycle_log_directory=str(host._scan_cycle_journal.directory))
                    host.post(host.log, f"SCAN + 4 PASSES LOG · {host._scan_cycle_journal.directory}")
                    if not record_cycle_event(host, "step_start", sequence_number=1, phase="precision", pass_number=0,
                                              type=initial_stage_type):
                        raise RuntimeError(host._scan_cycle_log_error)
                scan_started = time.monotonic()
                try:
                    steps = host._prepare_scan_weld_cycle(steps, execute_requested)
                except Exception as error:
                    record_cycle_event(host, "step_end", sequence_number=1, phase="precision", pass_number=0,
                                       type=initial_stage_type, success=False,
                                       duration_s=time.monotonic() - scan_started, message=str(error))
                    raise
                if not record_cycle_event(host, "step_end", sequence_number=1, phase="precision", pass_number=0,
                                          type=initial_stage_type, success=True,
                                          duration_s=time.monotonic() - scan_started, message="scan and path generation completed"):
                    raise RuntimeError(host._scan_cycle_log_error)
                indices = list(range(len(steps)))
                validate_managed_weld_sequence(steps, require_complete=True)
                if execute_requested:
                    error = self.execution_preflight_error(steps, False)
                    if error:
                        raise RuntimeError(error)
                # Record the actual measured geometry and each pass recipe,
                # rather than the placeholders visible before scanning.
                conditions = dict(conditions, steps=record_step_conditions(steps, indices))
                if host._scan_cycle_journal is not None:
                    conditions["scan_cycle_log_directory"] = str(host._scan_cycle_journal.directory)
                    host._scan_cycle_journal.prepared(conditions["steps"])
                attach_execution_conditions(steps, conditions)
            if execute_requested:
                arc_step = next((s for s in steps if s.get("type") == "digital_weld"
                                 and s.get("command") == "on"), None)
                if arc_step is not None:
                    host._finish_weld_feedback_record("closed before new sequence")
                    if host._scan_cycle_journal is None:
                        host._begin_weld_feedback_record(
                            arc_step["settings"],
                            {**(arc_step.get("execution_conditions") or {}),
                             "fake_arc": self.fake_arc()}
                        )
            host._sequence_worker_body(steps, indices, execute_requested)
        except Exception as error:
            # Otherwise a worker traceback leaves sequence_running latched and
            # every later Plan/Execute reports "A sequence is already running".
            stopped = host.sequence_stop_requested
            host.weld_motion_done_event.set()
            host.sequence_stop_requested = True
            client = host.hicomm_client
            if execute_requested:
                if client is not None and not self.fake_arc():
                    try:
                        client.inhibit_outputs()
                    except Exception:
                        pass
                try:
                    host.node.cancel_active_motion()
                except Exception:
                    pass
                try:
                    host._finish_weld_feedback_record(
                        f"failed: sequence worker exception: {error}",
                        client.latest_status() if client is not None else None,
                    )
                except Exception as feedback_error:
                    host.post(host.error, f"Weld feedback cleanup failed: {feedback_error}")
            record_cycle_event(host, "exception", message=str(error))
            self._finish_cycle_log("STOPPED" if stopped else "FAILED", str(error))
            host.post(host._sequence_finished, False, f"internal sequence error: {error}")
        finally:
            # Close every execution session, including fake runs.
            if execute_requested:
                host._finish_weld_feedback_record(
                    "stopped" if host.sequence_stop_requested else "sequence ended",
                    host.hicomm_client.latest_status() if host.hicomm_client is not None else None,
                )

    def _finish_cycle_log(self, status, message):
        journal = getattr(self.host, "_scan_cycle_journal", None)
        if journal is None:
            return True
        try:
            journal.finish(status, message)
            return True
        except (OSError, TypeError, ValueError) as error:
            self.host._scan_cycle_log_error = f"Scan + 4 Passes log finalization failed: {error}"
            self.host.post(self.host.error, self.host._scan_cycle_log_error)
            return False

    @staticmethod
    def _weld_motion_for(steps, scenario_id):
        return next((
            step for step in steps
            if step.get("type") == "motion"
            and step.get("weld_scenario_id") == scenario_id
            and step.get("weld_scenario_stage") == "weld_motion"
        ), None) if scenario_id else None

    @staticmethod
    def _path_key(step):
        return tuple(
            (p.position.x, p.position.y, p.position.z,
             p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
            for p in step.get("points", ())
        )

    def start_early_weld_preplan(self, steps, trigger_step):
        """Plan the weld path while the robot is still approaching START.

        Started at the scenario's first approach stage (weld/start WAIT), so a
        long weaving plan overlaps the approach motions instead of parking the
        torch at START before ARC ON.  It plans from the joints IK gives for
        the path start, seeded with the START WAIT joints (else the current
        ones). preplan_weld_motion uses it only if the robot really stopped
        there and the path did not change; otherwise it plans again from the
        current state as before.
        """
        scenario_id = trigger_step.get("weld_scenario_id")
        preplans = self.__dict__.setdefault("_early_preplans", {})
        if scenario_id in preplans:
            return
        weld_motion = self._weld_motion_for(steps, scenario_id)
        node = self.host.node
        if weld_motion is None or not hasattr(node, "path_start_joints"):
            return
        start_wait = next((
            step for step in steps
            if step.get("weld_scenario_id") == scenario_id
            and step.get("weld_scenario_stage") == "start_wait"
            and step.get("joint_names") and step.get("positions")
        ), None)
        seed = (dict(zip(start_wait["joint_names"], map(float, start_wait["positions"])))
                if start_wait is not None else None)
        early = {"done": threading.Event(), "ok": False, "detail": "",
                 "path": self._path_key(weld_motion), "started": time.monotonic()}
        preplans[scenario_id] = early

        def plan():
            try:
                early["names"], early["positions"] = node.path_start_joints(weld_motion, seed)
                candidate = dict(weld_motion, start_joint_positions=early["positions"],
                                 reuse_approved_plan=False)
                early["ok"], early["detail"] = node.run_sequence_cartesian_motion(
                    candidate, False, track_goal=False)
            except Exception as error:
                early["ok"], early["detail"] = False, f"{type(error).__name__}: {error}"
            finally:
                early["finished"] = time.monotonic()
                early["done"].set()

        threading.Thread(target=plan, daemon=True).start()

    def _use_early_preplan(self, weld_motion, scenario_id):
        early = self.__dict__.get("_early_preplans", {}).pop(scenario_id, None)
        if early is None:
            return False
        waited = time.monotonic()
        if not early["done"].wait(timeout=30.0):
            reason = "still planning after 30 s"
        elif not early["ok"]:
            reason = early["detail"]
        elif self._path_key(weld_motion) != early["path"]:
            reason = "weld path changed after the approach started"
        elif not self.host.node.joints_within(
                early["names"], early["positions"], EARLY_PREPLAN_START_TOLERANCE_RAD):
            reason = "robot did not stop at the planned start joints"
        else:
            weld_motion["start_joint_positions"] = list(early["positions"])
            weld_motion["reuse_approved_plan"] = True
            self.host.post(
                self.host.log,
                f"WELD PATH PRE-PLANNED during approach · "
                f"{early['finished'] - early['started']:.2f} s planning · "
                f"waited {time.monotonic() - waited:.2f} s at start · {early['detail']}",
            )
            return True
        self.host.post(self.host.log, f"Early weld plan not used ({reason}) · planning from the current state")
        return False

    def preplan_weld_motion(self, steps, arc_on_step):
        """Approve the scenario's weld path before its ARC ON is sent.

        The weld motion then executes the approved trajectory directly
        (``reuse_approved_plan``) as soon as ARC is established.  Returns an
        error message when planning fails, so ARC ON is never commanded.
        """
        host = self.host
        scenario_id = arc_on_step.get("weld_scenario_id")
        if not scenario_id:
            return None
        weld_motion = next((
            step for step in steps
            if step.get("type") == "motion"
            and step.get("weld_scenario_id") == scenario_id
            and step.get("weld_scenario_stage") == "weld_motion"
        ), None)
        if weld_motion is None:
            return None
        if self._use_early_preplan(weld_motion, scenario_id):
            return None
        weld_motion.pop("start_joint_positions", None)
        started = time.monotonic()
        planned, detail = host.node.run_sequence_cartesian_motion(weld_motion, False)
        if not planned:
            return f"weld path planning failed before ARC ON: {detail}"
        weld_motion["reuse_approved_plan"] = True
        host.post(
            host.log,
            f"WELD PATH PRE-PLANNED before ARC ON · "
            f"{time.monotonic() - started:.2f} s · {detail}",
        )
        return None

    def run_groups(self, steps, indices, execute_requested):
        host = self.host
        # The worker may also be exercised without the GUI constructor by
        # tests; keep its application state lazy like sequence_steps.
        if not hasattr(host, "sequence_model"):
            host.sequence_model = SequenceModel()
        success = True
        message = "complete"
        active_scenario_id = None
        # Never reuse an early plan left by an aborted earlier run.
        self.__dict__["_early_preplans"] = {}
        groups = defaultdict(list)
        for local_index, (stored_index, step) in enumerate(zip(indices, steps)):
            key = (
                ("sleep", stored_index)
                if step["type"] == "sleep"
                else ("slot", step.get("parallel_slot", local_index + 1))
            )
            groups[key].append((stored_index, step))

        for group_index, (key, members) in enumerate(groups.items(), start=1):
            if getattr(host, "_scan_cycle_log_error", None):
                success, message = False, host._scan_cycle_log_error
                break
            if host.sequence_stop_requested:
                success, message = False, "stopped by operator"
                break
            slot_label = key[1] if key[0] == "slot" else "sleep"
            host.sequence_model.set_progress(
                slot_label, group_index, len(groups),
                tuple(index for index, _step in members),
            )
            host.post(
                host._set_sequence_status,
                f"Parallel slot {slot_label} · group "
                f"{group_index}/{len(groups)} · {len(members)} task(s)",
            )
            results = {}
            workers = []
            weld_motion_group = any(
                step.get("weld_scenario_stage") == "weld_motion"
                for _stored_index, step in members
            )
            if weld_motion_group:
                host.weld_motion_done_event.clear()
                host.weld_motion_success = False
            if any(step.get("weld_scenario_stage") == "arc_on"
                   for _stored_index, step in members):
                active_scenario_id = next(step.get("weld_scenario_id") for _stored_index, step in members
                                          if step.get("weld_scenario_stage") == "arc_on")
                host.weld_arc_established_event.clear()
                host.weld_arc_on_done_event.clear()
                host.weld_arc_on_success = False
            if execute_requested:
                arc_on_step = next((
                    step for _stored_index, step in members
                    if step.get("type") == "digital_weld"
                    and step.get("command") == "on"
                ), None)
                if arc_on_step is not None:
                    # Plan the weld path before ARC ON is commanded; planning
                    # after ARC establishment left the torch parked on a live
                    # arc for the whole planning time (1-4 s with weaving).
                    preplan_error = self.preplan_weld_motion(steps, arc_on_step)
                    if preplan_error is not None:
                        record_cycle_event(host, "preplan_failed", pass_number=arc_on_step.get("scan_cycle_pass"), message=preplan_error)
                        success, message = False, preplan_error
                        break
                    if host.sequence_stop_requested:
                        success, message = False, "stopped by operator"
                        break
                    # Establish one common monotonic time base before motion and
                    # HICOMM workers race each other to their first callback.
                    host._begin_weld_feedback_record(
                        arc_on_step.get("settings"),
                        {**(arc_on_step.get("execution_conditions") or {}),
                         "fake_arc": self.fake_arc()},
                    )
            tcp_recorder = None
            if execute_requested and weld_motion_group:
                weld_motion_step = next(
                    step for _stored_index, step in members
                    if step.get("weld_scenario_stage") == "weld_motion"
                )
                tcp_recorder = threading.Thread(
                    target=host._record_actual_tcp_until_motion_done,
                    args=(weld_motion_step,),
                    daemon=True,
                )
                tcp_recorder.start()

            def run_member(result_key, member_step):
                member_result = (False, "sequence task did not run")
                task_started = time.monotonic()
                log_fields = dict(sequence_number=result_key + 2, type=member_step["type"],
                                  phase=member_step.get("scan_cycle_phase"), pass_number=member_step.get("scan_cycle_pass"),
                                  stage=member_step.get("weld_scenario_stage"), parallel_slot=member_step.get("parallel_slot"))
                try:
                    if not record_cycle_event(host, "step_start", **log_fields):
                        member_result = False, host._scan_cycle_log_error
                    else:
                        member_result = host._run_sequence_step(member_step, execute_requested)
                except Exception as error:
                    member_result = False, f"{type(error).__name__}: {error}"
                finally:
                    if not record_cycle_event(host, "step_end", **log_fields, success=bool(member_result[0]),
                                              message=str(member_result[1]), duration_s=time.monotonic() - task_started):
                        member_result = False, host._scan_cycle_log_error
                    results[result_key] = member_result
                    if execute_requested and not member_result[0]:
                        client = host.hicomm_client
                        if client is not None and not self.fake_arc():
                            client.inhibit_outputs()
                        host.node.cancel_active_motion()
                    host.post(host.log,
                        f"SEQUENCE STEP TIMING · #{result_key + 1} · "
                        f"stage={member_step.get('weld_scenario_stage', member_step['type'])} · "
                        f"duration={time.monotonic() - task_started:.3f}s · "
                        f"success={bool(member_result[0])}"
                    )
                    if member_step.get("weld_scenario_stage") == "weld_motion":
                        host.weld_motion_success = bool(member_result[0])
                        host.weld_motion_done_event.set()

            if execute_requested:
                for _stored_index, step in members:
                    if step.get("weld_scenario_stage") in EARLY_PREPLAN_STAGES:
                        self.start_early_weld_preplan(steps, step)
            for stored_index, step in members:
                worker = threading.Thread(
                    target=run_member,
                    args=(stored_index, step),
                    daemon=True,
                )
                workers.append(worker)
                worker.start()
            for worker in workers:
                worker.join()
            if tcp_recorder is not None:
                tcp_recorder.join(timeout=2.0)
            for stored_index, _step in members:
                step_success, step_message = results.get(
                    stored_index, (False, "parallel task produced no result")
                )
                host.post(
                    host.log,
                    f"Sequence #{stored_index + 1} · "
                    f"{'OK' if step_success else 'FAILED'} · {step_message}",
                )
                if not step_success:
                    success, message = False, step_message
                    break
            if not success:
                break
            if execute_requested:
                for _index, contact in members:
                    if not contact.get("start_touch_probe") or not contact.get("start_touch_result"):
                        continue
                    try:
                        from construct_robot.application.scan_start_touch import register_touch_height
                        registration = register_touch_height(steps, contact)
                        for arc in steps:
                            if arc.get("weld_scenario_id") == contact.get("weld_scenario_id") and arc.get("weld_scenario_stage") == "arc_on":
                                conditions = arc.setdefault("execution_conditions", {})
                                conditions["weld_start_touch"] = copy.deepcopy(registration)
                                conditions["steps"] = record_step_conditions(steps, indices)
                        if not record_cycle_event(host, "artifact", kind="start_touch_registration",
                                                  pass_number=contact.get("scan_cycle_pass"), **registration):
                            raise RuntimeError(host._scan_cycle_log_error)
                        host.post(host.log, f"START TOUCH · pass {contact.get('scan_cycle_pass')} · World Z "
                                  f"{registration['delta_world_m'][2] * 1000:+.3f} mm · START/GOAL updated before ARC")
                    except Exception as error:
                        success, message = False, f"start touch registration failed: {error}"
                        break
                if not success:
                    break
            if execute_requested and any(
                step.get("weld_scenario_stage") == "arc_off"
                and step.get("trigger_before_goal", False)
                for _stored_index, step in members
            ):
                if not self.fake_arc():
                    final_status = host._pending_weld_final_status()
                    host._finish_weld_feedback_record("completed", final_status)
            if (execute_requested and self.fake_arc() and getattr(host, "_scan_cycle_journal", None) is not None
                    and any(step.get("weld_scenario_stage") == "arc_off" for _index, step in members)):
                host._finish_weld_feedback_record("completed (fake ARC)")
        if execute_requested and success and not host.sequence_stop_requested and self.fake_arc():
            host._finish_weld_feedback_record("completed")
        if execute_requested and (not success or host.sequence_stop_requested):
            client = host.hicomm_client
            if client is not None and not self.fake_arc():
                client.clear_outputs()
            software_step = next((step for step in steps if step.get("weld_scenario_stage") == "software_crater"
                                  and step.get("weld_scenario_id") == active_scenario_id), None)
            if software_step is not None and client is not None:
                try:
                    host._software_crater_restore(validate_digital_weld_settings(software_step["settings"]))
                except Exception as restore_error:
                    host.post(host.error, f"Software crater failure cleanup restore failed: {restore_error}")
            # Keep touch-enable unchanged on STOP or failure. Explicit
            # scenario/GUI DO0 commands remain responsible for this output.
            host._finish_weld_feedback_record(
                "stopped" if host.sequence_stop_requested else f"failed: {message}",
                client.latest_status() if client is not None else None,
            )
        # Imported cleaner tasks must never leave a cutter/cleaner latched ON,
        # including when a later motion fails. Do not change touch-enable DO0.
        if execute_requested:
            cleaner_ports = {
                int(step["port"]) for step in steps
                if step.get("task_cleaner_output") and step.get("port") in (5, 6, 7)
            }
            for port in sorted(cleaner_ports):
                try:
                    off_ok, off_message = host._set_fastech_output_sync(port, False)
                except Exception as error:
                    off_ok, off_message = False, str(error)
                record_cycle_event(host, "cleaner_cleanup", port=port, value=False,
                                   success=bool(off_ok), message=str(off_message))
                if not off_ok:
                    success, message = False, f"Cleaner DO{port} cleanup OFF failed: {off_message}"
                    host.post(host.error, message)
        if getattr(host, "_scan_cycle_log_error", None):
            success, message = False, host._scan_cycle_log_error
        if not self._finish_cycle_log("STOPPED" if host.sequence_stop_requested else "COMPLETED" if success else "FAILED", message):
            success, message = False, host._scan_cycle_log_error
        host.post(host._sequence_finished, success, message)

    def run_step(self, step, execute_requested):
        host = self.host
        if step["type"] == "motion":
            if step.get("start_touch_probe"):
                from construct_robot.application.scan_start_touch import seek_start_touch
                return seek_start_touch(host, step, execute_requested)
            if (
                execute_requested
                and step.get("weld_scenario_stage") == "weld_motion"
            ):
                # The legacy path shares ARC ON and motion in one slot; custom
                # hot start uses preceding ARC ON/hold slots. Either way, the
                # robot cannot leave the motion start before establishment.
                deadline = time.monotonic() + 6.0
                while not host.weld_arc_established_event.is_set():
                    if host.sequence_stop_requested:
                        return False, "weld motion interrupted before ARC established"
                    if (
                        host.weld_arc_on_done_event.is_set()
                        and not host.weld_arc_on_success
                    ):
                        return False, "weld motion aborted: ARC ON failed"
                    if time.monotonic() >= deadline:
                        return False, "weld motion timed out waiting for ARC established"
                    time.sleep(0.01)

                host._mark_weld_motion_timing("start")
                try:
                    return host.node.run_sequence_cartesian_motion(
                        step, execute_requested
                    )
                finally:
                    host._mark_weld_motion_timing("complete")
            return host.node.run_sequence_cartesian_motion(step, execute_requested)
        # Only these whitelisted motion types share the runtime call signature.
        if step["type"] in (
            "planned_trajectory", "named_pose", "dual_arm_pose", "spray_motion", "head_motion",
        ):
            return getattr(host.node, f"run_sequence_{step['type']}")(step, execute_requested)
        if step["type"] == "post_weld_scan":
            return host._run_post_weld_scan(step, execute_requested)
        if step["type"] == "hongik_post_weld_scan":
            return host._run_hongik_post_weld_scan(step, execute_requested)
        if step["type"] == "sleep":
            if not execute_requested:
                return True, "sleep planned (no wait)"
            success = host._interruptible_wait(step["seconds"])
            return success, (
                f"slept {step['seconds']:.3f} seconds"
                if success
                else "sleep interrupted"
            )
        if step["type"] == "software_crater":
            if not execute_requested:
                return True, "software_crater planned (no setpoint sent)"
            return host._execute_software_crater(step)
        if step["type"] == "custom_hot_start":
            if not execute_requested:
                return True, "Custom Hot Start planned (no hold)"
            return host._execute_custom_hot_start(step)
        if not execute_requested:
            return True, "Equipment output command planned (no output sent)"
        duration = float(step.get("duration", 0.0))
        if step["type"] == "digital_weld":
            if step.get("command") == "off" and step.get(
                "trigger_before_goal", False
            ):
                return host._execute_triggered_arc_off(step)
            success, message = host._execute_hicomm_weld(
                step["command"],
                step["settings"],
                step.get("execution_conditions"),
            )
            if not success:
                return success, message
            if step["command"] == "on" and duration <= 0.0:
                return True, f"{message} · remains ON until D-WELD OFF"
            waited = host._interruptible_wait(duration)
            if step["command"] == "on":
                off_success, off_message = host._execute_hicomm_weld(
                    "off", step["settings"]
                )
                if not off_success:
                    return False, off_message
            return waited, (
                f"{message} · duration {duration:.3f} seconds"
                if waited
                else "D-WELD duration interrupted"
            )
        if step["type"] == "wire_feed":
            client = host.hicomm_client
            if client is None or not client.connected:
                return False, "Hi-COMM disconnected"
            if not math.isfinite(duration) or not 0 < duration <= 30:
                return False, "Wire feed duration must be 0..30 seconds"
            try:
                client.allow_outputs()
                client.set_command_bit(BIT_REVERSE, False)
                client.set_command_bit(BIT_FORWARD, True)
                waited = host._interruptible_wait(duration)
                return waited, "Wire feed complete" if waited else "Wire feed interrupted"
            finally:
                client.set_command_bit(BIT_FORWARD, False)
        if step["type"] == "gas":
            client = host.hicomm_client
            if client is None or not client.connected:
                return False, "Hi-COMM disconnected"
            enabled = bool(step["enabled"])
            try:
                client.set_command_bit(BIT_GAS, enabled)
                if not enabled:
                    return True, "GAS OFF sent"
                # A positive duration makes GAS ON a timed pulse.  Duration 0
                # keeps gas on until an explicit GAS OFF sequence step.
                if duration <= 0.0:
                    return True, "GAS ON sent; remains on until GAS OFF"
                waited = host._interruptible_wait(duration)
                client.set_command_bit(BIT_GAS, False)
                return waited, (
                    f"GAS ON for {duration:.3f} seconds, then OFF"
                    if waited
                    else "GAS timer interrupted; GAS OFF sent"
                )
            except Exception as error:
                client.set_command_bit(BIT_GAS, False)
                return False, str(error)
        if step["type"] == "digital_output":
            port = int(step["port"])
            enabled = bool(step["value"])
            backend = step.get("io_backend", "rainbow_legacy")
            if backend == self.touch_io_backend:
                set_output = host._set_fastech_output_sync
                output_name = "Fastech DO"
            else:
                set_output = host.node._set_legacy_digital_output_sync
                output_name = "Legacy Rainbow DO"
            success, message = set_output(port, enabled)
            if not success:
                return False, f"{output_name}{port} command failed: {message}"
            if not enabled or duration <= 0.0:
                return True, (
                    f"{output_name}{port} "
                    f"{'ON' if enabled else 'OFF'} confirmed"
                )
            waited = host._interruptible_wait(duration)
            off_success, off_message = set_output(port, False)
            if not off_success:
                return False, (
                    f"{output_name}{port} timed OFF failed: {off_message}"
                )
            return waited, (
                f"{output_name}{port} ON for {duration:.3f} seconds, then OFF"
                if waited
                else f"{output_name}{port} duration interrupted; OFF confirmed"
            )
        return False, "unsupported sequence step"
