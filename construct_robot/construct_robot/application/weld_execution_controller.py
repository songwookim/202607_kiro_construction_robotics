"""Weld ARC runtime orchestration and weld-feedback recording, independent of Tk.

Moved from ``WeldActionGui`` without changing semantics:

* :class:`WeldFeedbackRecorder` owns the weld feedback *session operations*
  (begin, Hi-COMM RX/TX samples, TCP trajectory, ARC-OFF control marks,
  hot-start/crater records, finish + log save + plot launch).
* :class:`WeldExecutionController` owns the ARC *operations*: fake ARC,
  D-WELD SET/ON/OFF, the pre-GOAL triggered ARC OFF watcher, Custom Hot
  Start and software crater.

Neither class holds state.  The synchronization objects stay on the host
(today ``WeldActionGui``) because the sequence executor, the STOP path and
Hi-COMM callbacks share them: ``weld_arc_established_event``,
``weld_arc_on_done_event``, ``weld_arc_on_success``, ``weld_motion_done_event``,
``weld_motion_success``, ``sequence_stop_requested``, ``weld_feedback_lock``,
``active_weld_feedback_session`` and ``_weld_feedback_stopped``.  Sibling
operations are called through the host (``host._execute_hicomm_weld``,
``host._mark_arc_off_control`` ...) so host-level overrides keep working.
The FAKE ARC decision is injected as ``fake_arc`` (the sequence executor's).
"""

import copy
import math
from pathlib import Path
import subprocess
import sys
import time

from geometry_msgs.msg import Pose
from tf2_ros import TransformException

from construct_robot.core.cartesian_path_common import (
    pose_is_valid,
    validated_seam_speed_factor,
)
from construct_robot.core.seam_geometry import _pose_position_tuple
from construct_robot.core.weld_config import (
    DIGITAL_WELD_COMMANDS,
    digital_weld_recipe,
    validate_digital_weld_settings,
    weld_current_profile,
)
from construct_robot.core.weld_quality_metrics import analyze_weld_quality
from construct_robot.io.hicomm_welder import BIT_ARC, PERIOD_SECONDS
from construct_robot.io.weld_logging import (
    calculate_weld_production_metrics,
    save_weld_feedback_log,
)


def weld_status_snapshot(status):
    keys = (
        "raw0",
        "output_state",
        "output_state_name",
        "sequence_stage",
        "arc_ack",
        "wcr_detected",
        "forward_ack",
        "gas_ack",
        "feedback_current_a",
        "feedback_voltage_v",
        "wire_feed_m_min",
        "set_current_a",
        "set_voltage_v",
        "hot_start_current_a",
        "hot_start_hold_adjustment",
        "hot_start_rx_raw_hex",
        "hot_start_arc_voltage_adjustment",
        "welder_error",
        "db_unavailable",
        "torch_collision",
    )
    return {key: status.get(key) for key in keys}


class WeldFeedbackRecorder:
    """Weld feedback session operations on the host's shared session/lock."""

    def __init__(self, host, *, tcp_sample_period_s):
        self.host = host
        self.tcp_sample_period_s = tcp_sample_period_s

    def record_software_crater(self, **values):
        host = self.host
        with host.weld_feedback_lock:
            session = host.active_weld_feedback_session
            if session is not None:
                session.setdefault("software_crater_control", {}).update(values)

    def record_custom_hot_start(self, **values):
        host = self.host
        with host.weld_feedback_lock:
            session = host.active_weld_feedback_session
            if session is not None:
                session.setdefault("custom_hot_start", {}).update(values)

    def begin(self, settings, execution_conditions=None):
        host = self.host
        with host.weld_feedback_lock:
            if getattr(host, "_weld_feedback_stopped", False):
                return
            if host.active_weld_feedback_session is not None:
                return
        conditions = copy.deepcopy(
            execution_conditions or {"mode": "manual_arc"}
        )
        conditions["hicomm_cyclic_period_ms"] = PERIOD_SECONDS * 1000.0
        conditions["arc_wait_recognition"] = True
        conditions["arc_wait_main_welding"] = True
        conditions["arc_wait_established"] = True
        conditions["arc_establishment_timeout_s"] = 5.0
        custom_planned = bool(settings.get("custom_hot_start_enabled", False)) and any(
            step.get("weld_scenario_stage") == "custom_hot_start"
            for step in conditions.get("steps", ())
        )
        with host.weld_feedback_lock:
            if getattr(host, "_weld_feedback_stopped", False):
                return
            if host.active_weld_feedback_session is not None:
                return
            host.active_weld_feedback_session = {
                "started_unix_time": time.time(),
                "started_monotonic": time.monotonic(),
                "commanded": copy.deepcopy(settings),
                "execution_conditions": conditions,
                "teaching_snapshot": host._teaching_snapshot_document(),
                "touch_snapshot": host._touch_snapshot_document(),
                "rx_samples": 0,
                "welding_samples": 0,
                "wcr_seen": False,
                "values": {
                    "current_a": [],
                    "voltage_v": [],
                    "wire_feed_m_min": [],
                },
                "setting_echo": None,
                "welding_setting_echo": None,
                "last_welding_status": None,
                "latest_measurement": None,
                "samples": [],
                "tx_frames": [],
                "tcp_samples": [],
                "latest_tcp_speed_m_s": 0.0,
                "arc_off_control": {},
                "custom_hot_start": {
                    "enabled": custom_planned,
                    "requested_hold_s": float(settings.get("custom_hot_start_hold_s", 0.15)),
                    "status": ("ARC_NOT_ESTABLISHED"
                               if custom_planned
                               else "DISABLED"),
                },
                "software_crater_control": {"enabled": bool(settings.get("software_crater_enabled", False))},
                "weld_motion_timing": {},
                "pending_final_status": None,
            }

    def record_tx_frame(self, frame, unix_time, monotonic):
        """Capture the exact bytes prepared for socket.send, without I/O work."""
        host = self.host
        with host.weld_feedback_lock:
            session = host.active_weld_feedback_session
            if session is None:
                return
            session["tx_frames"].append({
                "elapsed_s": max(0.0, monotonic-float(session["started_monotonic"])),
                "unix_time": float(unix_time),
                "raw_hex": bytes(frame).hex(" ").upper(),
            })

    def mark_weld_motion_timing(self, event):
        """Record weld-path start/completion on the feedback session clock."""
        host = self.host
        if event not in ("start", "complete"):
            raise ValueError(f"unsupported weld motion timing event: {event}")
        unix_time = time.time()
        monotonic = time.monotonic()
        with host.weld_feedback_lock:
            session = host.active_weld_feedback_session
            if session is None:
                return
            elapsed = max(
                0.0, monotonic - float(session["started_monotonic"])
            )
            timing = session.setdefault("weld_motion_timing", {})
            timing[f"{event}_elapsed_s"] = elapsed
            timing[f"{event}_unix_time"] = unix_time
            timing[f"{event}_wall_time"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(unix_time)
            ) + f".{int((unix_time % 1.0) * 1000.0):03d}"
        host.post(
            host.log,
            f"WELD MOTION {event.upper()} · "
            f"elapsed={elapsed:.3f} s · "
            f"wall={timing[f'{event}_wall_time']}",
        )

    def record_feedback_sample(self, status):
        host = self.host
        with host.weld_feedback_lock:
            session = host.active_weld_feedback_session
            if session is None:
                return
            session["rx_samples"] += 1
            session["setting_echo"] = {
                "current_a": int(status.get("set_current_a", 0)),
                "voltage_v": float(status.get("set_voltage_v", 0.0)),
                "hot_start_current_a": int(
                    status.get("hot_start_current_a", 0)
                ),
                "hot_start_hold_adjustment": int(
                    status.get("hot_start_hold_adjustment", 0)
                ),
                "hot_start_rx_raw_hex": status.get("hot_start_rx_raw_hex"),
            }
            sample = host._weld_status_snapshot(status)
            sample["elapsed_s"] = max(
                0.0, time.monotonic() - float(session["started_monotonic"])
            )
            session["samples"].append(sample)
            active = bool(
                status.get("arc_ack")
                or int(status.get("output_state", 0))
                or status.get("wcr_detected")
                or float(status.get("wire_feed_m_min", 0.0)) > 0.0
                or int(status.get("feedback_current_a", 0)) > 0
                or float(status.get("feedback_voltage_v", 0.0)) > 0.0
            )
            if not active:
                return
            session["welding_setting_echo"] = copy.deepcopy(
                session["setting_echo"]
            )
            session["wcr_seen"] = bool(
                session["wcr_seen"] or status.get("wcr_detected")
            )
            session["last_welding_status"] = host._weld_status_snapshot(status)
            measurement = bool(
                status.get("wcr_detected")
                or float(status.get("wire_feed_m_min", 0.0)) > 0.0
                or int(status.get("feedback_current_a", 0)) > 0
                or float(status.get("feedback_voltage_v", 0.0)) > 0.0
            )
            if not measurement:
                return
            session["welding_samples"] += 1
            current = float(status.get("feedback_current_a", 0.0))
            voltage = float(status.get("feedback_voltage_v", 0.0))
            wire_feed = float(status.get("wire_feed_m_min", 0.0))
            session["latest_measurement"] = {
                "elapsed_s": float(sample["elapsed_s"]),
                "current_a": current,
                "voltage_v": voltage,
                "wire_feed_m_min": wire_feed,
                "wcr_detected": bool(status.get("wcr_detected")),
                "arc_ack": bool(status.get("arc_ack")),
            }
            if current > 0.0:
                session["values"]["current_a"].append(current)
            if voltage > 0.0:
                session["values"]["voltage_v"].append(voltage)
            if wire_feed > 0.0:
                session["values"]["wire_feed_m_min"].append(wire_feed)

    def record_tcp_sample(
        self,
        pose,
        *,
        progress=0.0,
        waypoint_index=-1,
        phase="unknown",
        tf_stamp_s=None,
        along_mm=None,
        remaining_mm=None,
        cross_track_mm=None,
    ):
        """Record one unique physical TCP pose on the weld time base."""
        host = self.host
        if pose is None:
            return
        now = time.monotonic()
        with host.weld_feedback_lock:
            session = host.active_weld_feedback_session
            if session is None:
                return
            elapsed = max(0.0, now - float(session["started_monotonic"]))
            sample = {
                "elapsed_s": elapsed,
                "x_m": float(pose.position.x),
                "y_m": float(pose.position.y),
                "z_m": float(pose.position.z),
                "qx": float(pose.orientation.x),
                "qy": float(pose.orientation.y),
                "qz": float(pose.orientation.z),
                "qw": float(pose.orientation.w),
                "tf_stamp_s": (
                    None if tf_stamp_s is None else float(tf_stamp_s)
                ),
                "along_mm": None if along_mm is None else float(along_mm),
                "remaining_mm": (
                    None if remaining_mm is None else float(remaining_mm)
                ),
                "cross_track_mm": (
                    None if cross_track_mm is None else float(cross_track_mm)
                ),
                "progress": float(progress),
                "waypoint_index": int(waypoint_index),
                "phase": str(phase),
            }
            previous = session["tcp_samples"][-1] if session["tcp_samples"] else None
            same_pose = previous is not None and all(
                math.isclose(sample[key], float(previous[key]), abs_tol=1e-12)
                for key in ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
            )
            # Preserve fresh stationary TF updates for measured dwell/endpoint
            # settling; discard repeats of the very same TF timestamp.
            if same_pose and (
                tf_stamp_s is None
                or previous.get("tf_stamp_s") is None
                or float(tf_stamp_s) <= float(previous["tf_stamp_s"])
            ):
                return
            raw_speed = 0.0
            if previous is not None:
                previous_tf_stamp = previous.get("tf_stamp_s")
                tf_dt = (
                    float(tf_stamp_s) - float(previous_tf_stamp)
                    if tf_stamp_s is not None and previous_tf_stamp is not None
                    else 0.0
                )
                dt = (
                    tf_dt
                    if tf_dt > 1e-4
                    else elapsed - float(previous["elapsed_s"])
                )
                if dt > 1e-4:
                    dx = sample["x_m"] - float(previous["x_m"])
                    dy = sample["y_m"] - float(previous["y_m"])
                    dz = sample["z_m"] - float(previous["z_m"])
                    raw_speed = math.sqrt(dx * dx + dy * dy + dz * dz) / dt
            previous_filtered = float(session.get("latest_tcp_speed_m_s", 0.0))
            filtered = (
                raw_speed
                if previous is None or previous_filtered <= 0.0
                else 0.30 * raw_speed + 0.70 * previous_filtered
            )
            filtered = max(0.0, min(2.0, filtered))
            sample["raw_speed_m_s"] = raw_speed
            sample["speed_m_s"] = filtered
            session["latest_tcp_speed_m_s"] = filtered
            session["tcp_samples"].append(sample)

    def record_actual_tcp_until_motion_done(self, step):
        """Record stamped TF poses through the complete weld lead-out."""
        host = self.host
        group = step.get("planning_group", "right_manipulator")
        seam_start = step.get("usable_seam_start")
        seam_goal = step.get("usable_seam_goal")
        geometry_valid = pose_is_valid(seam_start) and pose_is_valid(seam_goal)
        if geometry_valid:
            sx, sy, sz = _pose_position_tuple(seam_start)
            gx, gy, gz = _pose_position_tuple(seam_goal)
            vx, vy, vz = gx - sx, gy - sy, gz - sz
            seam_length = math.sqrt(vx * vx + vy * vy + vz * vz)
            geometry_valid = seam_length > 1e-9
            if geometry_valid:
                tx, ty, tz = vx / seam_length, vy / seam_length, vz / seam_length

        if geometry_valid:
            with host.weld_feedback_lock:
                session = host.active_weld_feedback_session
                if session is not None:
                    session["execution_conditions"]["seam_start_xyz"] = (sx, sy, sz)
                    session["execution_conditions"]["seam_goal_xyz"] = (gx, gy, gz)
                    session["execution_conditions"]["planned_weave_waypoints_xyz"] = [
                        _pose_position_tuple(pose) for pose in step.get("points", ())
                    ]

        stopped_observing_at = None

        while True:
            try:
                transform = host.node._current_tcp_transform(group)
                source = transform.transform
                pose = Pose()
                pose.position.x = source.translation.x
                pose.position.y = source.translation.y
                pose.position.z = source.translation.z
                pose.orientation = source.rotation
                stamp = transform.header.stamp
                tf_stamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
                along_mm = remaining_mm = cross_track_mm = None
                if geometry_valid:
                    dx = float(pose.position.x) - sx
                    dy = float(pose.position.y) - sy
                    dz = float(pose.position.z) - sz
                    along_m = dx * tx + dy * ty + dz * tz
                    px = dx - along_m * tx
                    py = dy - along_m * ty
                    pz = dz - along_m * tz
                    along_mm = along_m * 1000.0
                    remaining_mm = (seam_length - along_m) * 1000.0
                    cross_track_mm = math.sqrt(px * px + py * py + pz * pz) * 1000.0
                host.record_weld_tcp_sample(
                    pose,
                    progress=0.0,
                    waypoint_index=-1,
                    phase="ACTUAL_TF",
                    tf_stamp_s=tf_stamp_s,
                    along_mm=along_mm,
                    remaining_mm=remaining_mm,
                    cross_track_mm=cross_track_mm,
                )
            except TransformException:
                pass
            if host.weld_motion_done_event.is_set():
                if stopped_observing_at is None:
                    stopped_observing_at = time.monotonic()
                elif time.monotonic() - stopped_observing_at >= 0.25:
                    return
            time.sleep(self.tcp_sample_period_s)

    def latest_tcp_state(self):
        host = self.host
        with host.weld_feedback_lock:
            session = host.active_weld_feedback_session
            if session is None or not session.get("tcp_samples"):
                return None
            return copy.deepcopy(session["tcp_samples"][-1])

    def mark_arc_off_control(self, **values):
        host = self.host
        with host.weld_feedback_lock:
            session = host.active_weld_feedback_session
            if session is None:
                return
            control = session.setdefault("arc_off_control", {})
            control.update(values)
            control.setdefault(
                "command_elapsed_s",
                max(0.0, time.monotonic() - float(session["started_monotonic"])),
            )

    def pending_final_status(self):
        host = self.host
        with host.weld_feedback_lock:
            session = host.active_weld_feedback_session
            if session is None:
                return None
            return copy.deepcopy(session.get("pending_final_status"))

    def finish(self, result, final_status=None):
        host = self.host
        with host.weld_feedback_lock:
            session = host.active_weld_feedback_session
            host.active_weld_feedback_session = None
        if session is None:
            return None

        def statistics(values):
            if not values:
                return {"min": None, "average": None, "max": None}
            return {
                "min": float(min(values)),
                "average": float(sum(values) / len(values)),
                "max": float(max(values)),
            }

        # Derive ARC extinction latency from electrical feedback after the
        # actual OFF command.  WCR is retained separately because some power
        # sources clear that bit noticeably later than welding current.
        arc_off_control = copy.deepcopy(session.get("arc_off_control", {}))
        command_elapsed = arc_off_control.get("command_elapsed_s")
        if command_elapsed is not None:
            command_elapsed = float(command_elapsed)
            post_off = [
                sample for sample in session.get("samples", ())
                if float(sample.get("elapsed_s", 0.0)) >= command_elapsed
            ]
            current_threshold_a = 10.0
            arc_off_control["extinction_current_threshold_a"] = current_threshold_a
            extinction_elapsed = None
            for index in range(max(0, len(post_off) - 1)):
                first = post_off[index]
                second = post_off[index + 1]
                if (
                    float(first.get("feedback_current_a", 0.0) or 0.0)
                    <= current_threshold_a
                    and float(second.get("feedback_current_a", 0.0) or 0.0)
                    <= current_threshold_a
                ):
                    extinction_elapsed = float(first.get("elapsed_s", 0.0))
                    break
            if extinction_elapsed is not None:
                extinction_delay = max(0.0, extinction_elapsed - command_elapsed)
                arc_off_control["feedback_current_extinguished_elapsed_s"] = (
                    extinction_elapsed
                )
                arc_off_control["feedback_extinction_delay_s"] = extinction_delay
                speed = float(arc_off_control.get("trigger_speed_m_s", 0.0) or 0.0)
                if speed > 0.0:
                    arc_off_control["feedback_recommended_pre_off_distance_m"] = (
                        speed * extinction_delay
                    )
            wcr_clear_elapsed = next((
                float(sample.get("elapsed_s", 0.0))
                for sample in post_off
                if not bool(sample.get("wcr_detected"))
            ), None)
            if wcr_clear_elapsed is not None:
                arc_off_control["wcr_clear_elapsed_s"] = wcr_clear_elapsed
                arc_off_control["wcr_clear_delay_s"] = max(
                    0.0, wcr_clear_elapsed - command_elapsed
                )

        ended = time.time()
        motion_timing = copy.deepcopy(session.get("weld_motion_timing", {}))
        production_metrics = calculate_weld_production_metrics(
            session.get("samples", ()),
            weld_motion_start_elapsed_s=motion_timing.get("start_elapsed_s"),
            weld_motion_complete_elapsed_s=motion_timing.get(
                "complete_elapsed_s"
            ),
            wire_consumable_alpha_mm=session.get("commanded", {}).get(
                "wire_consumable_alpha_mm", 0.0
            ),
        )
        production_metrics.update({
            "weld_motion_start_unix_time": motion_timing.get("start_unix_time"),
            "weld_motion_start_wall_time": motion_timing.get("start_wall_time"),
            "weld_motion_complete_unix_time": motion_timing.get(
                "complete_unix_time"
            ),
            "weld_motion_complete_wall_time": motion_timing.get(
                "complete_wall_time"
            ),
        })
        document = {
            "format_version": 1,
            "result": str(result),
            "started": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(float(session["started_unix_time"])),
            ),
            "ended": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(ended)
            ),
            "started_unix_time": float(session["started_unix_time"]),
            "ended_unix_time": ended,
            "elapsed_seconds": max(
                0.0, time.monotonic() - float(session["started_monotonic"])
            ),
            "commanded": session["commanded"],
            "rx_setting_echo": session["setting_echo"],
            "rx_welding_setting_echo": session["welding_setting_echo"],
            "execution_conditions": session["execution_conditions"],
            "feedback": {
                "rx_samples": int(session["rx_samples"]),
                "welding_samples": int(session["welding_samples"]),
                "wcr_seen": bool(session["wcr_seen"]),
                "current_a": statistics(session["values"]["current_a"]),
                "voltage_v": statistics(session["values"]["voltage_v"]),
                "wire_feed_m_min": statistics(
                    session["values"]["wire_feed_m_min"]
                ),
                "last_welding_status": session["last_welding_status"],
                "final_status": (
                    host._weld_status_snapshot(final_status)
                    if final_status is not None
                    else None
                ),
            },
            "samples": session["samples"],
            "tx_frames": session.get("tx_frames", []),
            "tcp_trajectory": session.get("tcp_samples", []),
            "arc_off_control": arc_off_control,
            "custom_hot_start": copy.deepcopy(session.get("custom_hot_start", {})),
            "software_crater_control": session.get("software_crater_control", {}),
            "production_metrics": production_metrics,
            "teaching_snapshot": session.get("teaching_snapshot", {}),
            "touch_snapshot": session.get("touch_snapshot", {}),
        }
        try:
            document["quality_metrics"] = analyze_weld_quality(document)
            custom = document["custom_hot_start"]
            custom_metrics = document["quality_metrics"].get("custom_hot_start", {})
            custom["current_a"] = custom_metrics.get("current_a")
            custom["voltage_v"] = custom_metrics.get("voltage_v")
            custom["rx_sample_count"] = custom_metrics.get("sample_count", 0)
            timeline = document["quality_metrics"].get("timeline", {})
            recognized = (timeline.get("ARC_RECOGNIZED") or {}).get("elapsed_s")
            begin = custom.get("hold_start_elapsed_s")
            end = custom.get("hold_end_elapsed_s")
            motion_start = motion_timing.get("start_elapsed_s")
            custom["arc_recognized_to_begin_s"] = (
                max(0.0, begin - recognized)
                if begin is not None and recognized is not None else None
            )
            custom["end_to_motion_start_s"] = (
                max(0.0, motion_start - end)
                if motion_start is not None and end is not None else None
            )
        except (ArithmeticError, KeyError, TypeError, ValueError) as error:
            # A malformed or missing analysis field must never discard the raw
            # feedback, TX frames, or TCP trajectory captured during a weld.
            document["quality_metrics"] = {"error": str(error)}
            host.post(host.error, f"Weld quality analysis unavailable: {error}")
        directory = host._weld_feedback_directory()
        timestamp = (
            time.strftime("%Y%m%d_%H%M%S", time.localtime(ended))
            + f"_{int((ended % 1.0) * 1000.0):03d}"
        )
        history_path = directory / f"weld_feedback_{timestamp}.log"
        latest_path = directory / "latest_weld_feedback.log"
        try:
            save_weld_feedback_log(history_path, document)
            save_weld_feedback_log(latest_path, document)
        except (KeyError, OSError, TypeError, ValueError) as error:
            host.post(host.error, f"Weld feedback save failed: {error}")
            return None
        workspace_python = Path.home() / "ros2_ws" / ".venv" / "bin" / "python"
        python = str(
            workspace_python if workspace_python.is_file() else sys.executable
        )
        try:
            subprocess.Popen((
                python,
                "-m",
                "construct_robot.io.weld_feedback_plot",
                str(history_path),
                str(latest_path),
                "--no-show",
            ))
        except OSError as error:
            host.post(host.error, f"Weld feedback plot launch failed: {error}")
        host.post(
            host.log,
            f"WELD FEEDBACK SAVED · {history_path} · "
            f"feedback + trajectory_3d PNG generation requested · result={result}",
        )
        return history_path


class WeldExecutionController:
    """ARC runtime operations against the host's Hi-COMM client and events."""

    def __init__(self, host, *, fake_arc):
        self.host = host
        self.fake_arc = fake_arc

    def execute_fake_arc(self, kind):
        """Simulate a D-WELD command without touching the welder.

        Used when "Fake ARC" is enabled so a sequence can be run to check
        motion only. The weld-motion thread still waits on the ARC
        established/done events before leaving LEAD START, so those are set
        exactly as the real ARC ON handshake would.
        """
        host = self.host
        if kind not in DIGITAL_WELD_COMMANDS:
            return False, f"unsupported D-WELD command: {kind}"
        if kind == "on":
            host.weld_arc_on_success = True
            host.weld_arc_established_event.set()
            host.weld_arc_on_done_event.set()
            return True, "FAKE ARC ON · no command sent to welder"
        if kind == "off":
            return True, "FAKE ARC OFF · no command sent to welder"
        return True, "FAKE ARC SET · no command sent to welder"

    def execute_custom_hot_start(self, step):
        """Dwell at the existing start pose after the ARC ON handshake."""
        host = self.host
        settings = validate_digital_weld_settings(step["settings"])
        hold_s = settings["custom_hot_start_hold_s"]
        client = host.hicomm_client
        fake = self.fake_arc()
        if not host.weld_arc_established_event.is_set() or not host.weld_arc_on_success:
            host._custom_hot_start_record(status="ARC_NOT_ESTABLISHED")
            return False, "Custom Hot Start blocked: ARC was not established"
        if not fake and (client is None or not client.comm_alive()):
            host._custom_hot_start_record(status="ABORTED", failure="Hi-COMM feedback unavailable")
            return False, "Custom Hot Start blocked: Hi-COMM feedback unavailable"
        begin = None
        started = None
        max_drift_mm = 0.0
        end_xyz = None
        boosted = False
        target_current = round(settings["current_a"] * (1 + settings["custom_hot_start_percent"] / 100))
        try:
            if host.sequence_stop_requested:
                raise RuntimeError("sequence stopped before Custom Hot Start")
            if not fake:
                boosted = True
                client.update_setpoints(target_current, settings["voltage_tenths"])
                host._custom_hot_start_record(
                    requested_boost_percent=settings["custom_hot_start_percent"],
                    target_current_a=target_current, main_current_a=settings["current_a"],
                    target_voltage_v=settings["voltage_tenths"] / 10.0,
                    setpoint_tx="SENT",
                )
                deadline = time.monotonic() + 2.0
                while True:
                    if host.sequence_stop_requested or not client.comm_alive():
                        raise RuntimeError("Custom boost confirmation interrupted")
                    status = client.latest_status() or {}
                    if status.get("arc_established") and abs(float(status.get("feedback_current_a", 0)) - target_current) <= max(10, target_current * 0.10):
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Custom boost feedback did not reach target within 2 s")
                    time.sleep(0.02)
            reference = host.node._current_tcp_pose(step["planning_group"])
            start_xyz = _pose_position_tuple(reference)
            expected = step.get("expected_start_tcp")
            expected_error_mm = (
                1000.0 * math.dist(start_xyz, _pose_position_tuple(expected))
                if expected is not None and pose_is_valid(expected) else None
            )
            with host.weld_feedback_lock:
                session = host.active_weld_feedback_session
                started = session["started_monotonic"] if session is not None else None
                established = ((session.get("arc_off_control") or {}).get(
                    "arc_established_elapsed_s") if session is not None else None)
            begin = time.monotonic()
            host._custom_hot_start_record(
                arc_established_elapsed_s=established,
                hold_start_elapsed_s=begin - started if started is not None else None,
                start_tcp_xyz=start_xyz, max_tcp_drift_mm=0.0,
                expected_start_error_mm=expected_error_mm,
                start_pose_role=step.get("start_pose_role", "weld_motion_start"),
                status="ABORTED", simulated=fake,
            )
            host.post(
                host.log,
                f"CUSTOM HOT START BEGIN · arc established · holding {hold_s:.3f} s "
                f"at {step.get('start_pose_role', 'weld_motion_start')} · "
                f"target error={expected_error_mm if expected_error_mm is not None else float('nan'):.3f} mm",
            )
            deadline = begin + hold_s
            end_xyz = start_xyz
            while True:
                if host.sequence_stop_requested:
                    raise RuntimeError("sequence stopped during Custom Hot Start")
                if not fake:
                    if client is None or not client.comm_alive():
                        raise RuntimeError("Hi-COMM communication lost during Custom Hot Start")
                    status = client.latest_status()
                    if not status or not status.get("arc_established"):
                        raise RuntimeError("ARC lost during Custom Hot Start")
                current_xyz = _pose_position_tuple(
                    host.node._current_tcp_pose(step["planning_group"])
                )
                end_xyz = current_xyz
                max_drift_mm = max(max_drift_mm, 1000.0 * math.dist(start_xyz, current_xyz))
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(0.02, remaining))
            end = time.monotonic()
            if not fake:
                client.update_setpoints(settings["current_a"], settings["voltage_tenths"])
                boosted = False
                host._custom_hot_start_record(main_restored=True)
            host._custom_hot_start_record(
                hold_end_elapsed_s=end - started if started is not None else None,
                actual_hold_s=end - begin,
                end_tcp_xyz=end_xyz,
                max_tcp_drift_mm=max_drift_mm,
                status="COMPLETED",
            )
            host.post(host.log, f"CUSTOM HOT START END · actual={end - begin:.3f} s · TCP drift={max_drift_mm:.3f} mm")
            return True, f"Custom Hot Start completed · hold={end - begin:.3f} s · drift={max_drift_mm:.3f} mm"
        except Exception as error:
            if not fake and client is not None:
                client.inhibit_outputs()
                if boosted:
                    try:
                        client.update_setpoints(settings["current_a"], settings["voltage_tenths"])
                        host._custom_hot_start_record(main_restored=True)
                    except Exception as restore_error:
                        host._custom_hot_start_record(restore_error=str(restore_error))
            partial = {"status": "ABORTED", "failure": str(error),
                       "max_tcp_drift_mm": max_drift_mm}
            if begin is not None:
                ended = time.monotonic()
                partial["actual_hold_s"] = ended - begin
                partial["hold_end_elapsed_s"] = (
                    ended - started if started is not None else None
                )
                partial["end_tcp_xyz"] = end_xyz
            host._custom_hot_start_record(**partial)
            return False, f"Custom Hot Start aborted: {error}"

    def software_crater_restore(self, settings):
        host = self.host
        client = host.hicomm_client
        if client is not None:
            client.update_setpoints(settings["current_a"], settings["voltage_tenths"])
            host._software_crater_record(main_restored=True)

    def execute_software_crater(self, step):
        """Hold at the measured endpoint with reduced ARC-ON setpoints."""
        host = self.host
        settings = validate_digital_weld_settings(step["settings"])
        if self.fake_arc():
            return True, "FAKE software_crater · no setpoint or ARC command sent"
        client = host.hicomm_client
        target = round(settings["current_a"] * settings["software_crater_ratio_percent"] / 100.0)
        voltage = round(settings["software_crater_voltage_v"] * 10.0)
        hold_s = settings["software_crater_hold_s"]
        host._software_crater_record(
            enabled=True, main_current_a=settings["current_a"],
            main_voltage_v=settings["voltage"], target_current_a=target,
            target_voltage_v=voltage / 10.0,
            ratio_percent=settings["software_crater_ratio_percent"],
            requested_hold_s=hold_s, status="NOT_OBSERVED",
        )
        if settings.get("expect_native_crater", False):
            host.post(host.log, "SOFTWARE CRATER · native crater is also expected; disable it on the welder panel for a software-only test")
        try:
            if client is None or not client.connected or not client.comm_alive():
                raise RuntimeError("Hi-COMM feedback unavailable before software_crater")
            if not client.snapshot().command & BIT_ARC:
                raise RuntimeError("ARC is not ON before software_crater")
            if host.sequence_stop_requested or not host.weld_motion_done_event.is_set() or not host.weld_motion_success:
                raise RuntimeError("weld motion did not complete before software_crater")
            endpoint = step.get("endpoint")
            if not pose_is_valid(endpoint):
                raise RuntimeError("software_crater endpoint is invalid")
            group = step.get("planning_group", "right_manipulator")
            arm = "left" if group.startswith("left") else "right"
            if not host.node.wait_until_arm_stopped(arm, timeout=2.0):
                raise RuntimeError("robot did not settle at crater endpoint")
            actual = host.node._current_tcp_transform(group).transform.translation
            error_mm = math.sqrt(sum((float(getattr(actual, axis)) - float(getattr(endpoint.position, axis))) ** 2
                                     for axis in ("x", "y", "z"))) * 1000.0
            host._software_crater_record(endpoint_error_mm=error_mm)
            if error_mm > 2.0:
                raise RuntimeError(f"TCP is {error_mm:.2f} mm from crater endpoint (>2 mm)")
            with host.weld_feedback_lock:
                session = host.active_weld_feedback_session
                sample_count = len(session["samples"]) if session else 0
                frame_count = len(session["tx_frames"]) if session else 0
            client.update_setpoints(target, voltage)
            host.post(host.log, f"SOFTWARE CRATER · endpoint HOLD · TX requested {target} A / {voltage / 10:.1f} V")
            host._software_crater_record(command_elapsed_s=time.monotonic() - session["started_monotonic"] if session else None)
            deadline = time.monotonic() + 1.5
            tx_seen = False
            tx_elapsed = None
            consecutive = 0
            echo = None
            tolerance = max(10.0, target * 0.20)
            while time.monotonic() < deadline:
                if host.sequence_stop_requested or not client.comm_alive():
                    raise RuntimeError("software_crater interrupted or Hi-COMM disconnected")
                with host.weld_feedback_lock:
                    live = host.active_weld_feedback_session
                    frames = list(live["tx_frames"][frame_count:]) if live else []
                    samples = list(live["samples"][sample_count:]) if live else []
                    sample_count += len(samples)
                for frame in frames:
                    raw = bytes.fromhex(frame["raw_hex"])
                    if len(raw) == 55 and raw[0] & BIT_ARC and int.from_bytes(raw[3:5], "little") == target and int.from_bytes(raw[5:7], "little") == voltage:
                        tx_seen = True
                        tx_elapsed = float(frame["elapsed_s"])
                        host._software_crater_record(tx_status="SENT", tx_elapsed_s=frame["elapsed_s"])
                        break
                frame_count += len(frames)
                for sample in samples:
                    if not tx_seen or float(sample.get("elapsed_s", -1)) < tx_elapsed:
                        continue
                    echo = {"current_a": sample.get("set_current_a"), "voltage_v": sample.get("set_voltage_v")}
                    current = float(sample.get("feedback_current_a", 0) or 0)
                    if bool(sample.get("wcr_detected")) and abs(current - target) <= tolerance:
                        consecutive += 1
                    else:
                        consecutive = 0
                    if consecutive >= 2:
                        confirmed = time.monotonic()
                        host._software_crater_record(rx_echo=echo, feedback_confirmed=True,
                                                     hold_start_elapsed_s=confirmed - live["started_monotonic"])
                        host.post(host.log, f"SOFTWARE CRATER · actual current confirmed near {target} A · hold timer started")
                        hold_deadline = confirmed + hold_s
                        while time.monotonic() < hold_deadline:
                            if host.sequence_stop_requested or not client.comm_alive():
                                raise RuntimeError("software_crater HOLD interrupted or feedback lost")
                            time.sleep(min(0.02, hold_deadline - time.monotonic()))
                        ended = time.monotonic()
                        host._software_crater_record(hold_end_elapsed_s=ended - live["started_monotonic"],
                                                     actual_hold_s=ended - confirmed)
                        return True, f"software_crater HOLD complete · {target} A / {voltage / 10:.1f} V · {ended - confirmed:.3f} s"
                time.sleep(0.02)
            host._software_crater_record(rx_echo=echo, feedback_confirmed=False)
            raise RuntimeError("software_crater setpoint feedback timeout (1.5 s)")
        except Exception as error:
            host._software_crater_record(status="FAILED", failure=str(error))
            if client is not None:
                try:
                    host._mark_arc_off_control(fallback="software_crater_failure")
                    client.arc_off(timeout=1.0, wait_idle=False, wait_sequence_clear=False)
                except Exception:
                    client.set_arc(False)
                finally:
                    try:
                        host._software_crater_restore(settings)
                    except Exception as restore_error:
                        host.post(host.error, f"Software crater main setpoint restore failed: {restore_error}")
            return False, f"software_crater failed: {error}"

    def execute_hicomm_weld(
        self, kind, settings, execution_conditions=None, *, finalize_feedback=True,
        apply_crater=True,
    ):
        host = self.host
        if self.fake_arc():
            return host._execute_fake_arc(kind)
        client = host.hicomm_client
        if client is None or not client.connected:
            if kind == "on":
                host.weld_arc_on_success = False
                host.weld_arc_on_done_event.set()
            return False, "Hi-COMM disconnected"
        if kind not in DIGITAL_WELD_COMMANDS:
            return False, f"unsupported D-WELD command: {kind}"
        try:
            settings = validate_digital_weld_settings(settings or {})
            if kind == "set":
                client.arc_set(**digital_weld_recipe(settings))
                echo = client.setting_echo()
                return True, f"recipe applied · RX echo={echo}"
            elif kind == "on":
                # The generated scenario owns a frozen recipe snapshot. Apply
                # that exact snapshot immediately before ARC ON so execution
                # never depends on whichever manual SET happened previously.
                current_profile = weld_current_profile(settings)
                # Hot Start has its own official TX field (Byte14-15). Keep
                # Byte3-4 at the nominal main-weld current; changing the main
                # setpoint for 0.5 s would only imitate, and can conflict with,
                # the power source's native start sequence.
                client.arc_set(**digital_weld_recipe(settings))
                host._begin_weld_feedback_record(
                    settings, execution_conditions
                )

                # The pre-GOAL ARC-OFF watcher is allowed to run in parallel
                # with motion, but it must remain DISARMED until this exact
                # establishment handshake succeeds.
                host.weld_arc_on_success = False
                status = client.arc_on(
                    wait_recognition=True,
                    wait_welding=True,
                    wait_established=True,
                    timeout=5.0,
                )
                host.weld_arc_on_success = True
                with host.weld_feedback_lock:
                    session = host.active_weld_feedback_session
                    if session is not None:
                        session.setdefault("arc_off_control", {})[
                            "arc_established_elapsed_s"
                        ] = max(
                            0.0,
                            time.monotonic() - float(session["started_monotonic"]),
                        )
                host.weld_arc_established_event.set()
                host.post(
                    host.log,
                    "HOT START NATIVE · "
                    f"TX Byte14-15={settings['hot_start_current_a']} A · "
                    f"TX Byte16 hold adjustment="
                    f"{settings['hot_start_hold_adjustment']:+d}",
                )
                host.weld_arc_on_done_event.set()
                return True, (
                    "ARC established (main_weld + WCR + feed) · "
                    f"native hot={settings['hot_start_current_a']} A → "
                    f"nominal={current_profile['nominal']} A · "
                    f"output={status['output_state_name']} · "
                    f"feedback={status['feedback_current_a']} A/"
                    f"{status['feedback_voltage_v']:.1f} V · "
                    f"WFS={status['wire_feed_m_min']:.1f} m/min"
                )
            if apply_crater and settings.get("expect_native_crater", True):
                # The available TX protocol has no crater-current/time field.
                # Clearing ARC starts the welder-panel crater sequence; RX
                # output_state=2 records when that native sequence is active.
                host.post(
                    host.log,
                    "CRATER NATIVE · ARC OFF will use welder-panel settings · "
                    f"panel reference={settings['crater_panel_current_ref_a']:.1f}A/"
                    f"{settings['crater_panel_voltage_ref_v']:.1f}V/"
                    f"{settings['crater_panel_time_ref_s']:.2f}s",
                )
            host._mark_arc_off_control()
            status = client.arc_off(
                timeout=max(
                    5.0,
                    (
                        0.0
                    ) + 2.0,
                ),
                wait_idle=True,
                wait_sequence_clear=True,
            )
            with host.weld_feedback_lock:
                session = host.active_weld_feedback_session
                if session is not None:
                    session.setdefault("arc_off_control", {})[
                        "sequence_clear_elapsed_s"
                    ] = max(0.0, time.monotonic() - float(session["started_monotonic"]))
            if settings.get("software_crater_enabled", False):
                host._software_crater_restore(settings)
            if not finalize_feedback:
                with host.weld_feedback_lock:
                    session = host.active_weld_feedback_session
                    if session is not None:
                        session["pending_final_status"] = host._weld_status_snapshot(status)
                return True, (
                    "ARC OFF sequence clear while lead-out motion continues · "
                    f"output={status['output_state_name']} · "
                    f"stage={status.get('sequence_stage', 'unknown')}"
                )
            feedback_path = host._finish_weld_feedback_record(
                "completed", status
            )
            return True, (
                "ARC OFF sequence clear · "
                f"output={status['output_state_name']} · "
                f"stage={status.get('sequence_stage', 'unknown')} · "
                f"feedback={feedback_path or 'not recorded'}"
            )
        except Exception as error:
            # Match v5.2: an ARC feedback timeout/error does not itself alter
            # the already transmitted ARC command.  Only explicit D-WELD OFF,
            # STOP, disconnect, or the sequence failure safety cleanup may
            # clear outputs.
            if kind == "on":
                host.weld_arc_on_success = False
                # Release a waiting watcher so it can fail closed instead of
                # waiting for TCP geometry and issuing a racing ARC-OFF.
                host.weld_arc_on_done_event.set()
            else:
                client.set_arc(False)
                if kind == "off" and isinstance(settings, dict) and settings.get("software_crater_enabled", False):
                    try:
                        host._software_crater_restore(settings)
                    except Exception as restore_error:
                        host.post(host.error, f"Software crater main setpoint restore failed: {restore_error}")
                host._finish_weld_feedback_record(
                    f"ARC {kind.upper()} failed: {error}",
                    client.latest_status(),
                )
            return False, str(error)
        finally:
            if kind == "off" and isinstance(settings, dict) and settings.get("software_crater_enabled", False):
                try:
                    host._software_crater_restore(settings)
                except Exception as restore_error:
                    host.post(host.error, f"Software crater main setpoint restore failed: {restore_error}")

    def execute_triggered_arc_off(self, step):
        """Turn ARC off before GOAL without breaking the continuous TCP motion."""
        host = self.host
        if self.fake_arc():
            # Dry runs finish at motion completion; no geometric pre-OFF or
            # welder-feedback timing is needed. Call the fake handler directly
            # so changing the checkbox cannot send a real OFF from this branch.
            while not host.weld_motion_done_event.wait(timeout=0.05):
                if host.sequence_stop_requested:
                    host._execute_fake_arc("off")
                    return False, "FAKE ARC OFF · sequence interrupted"
                if (host.weld_arc_on_done_event.is_set()
                        and not host.weld_arc_on_success):
                    host._execute_fake_arc("off")
                    return False, "FAKE ARC OFF · ARC ON failed"
            success, message = host._execute_fake_arc("off")
            if host.sequence_stop_requested or not host.weld_motion_success:
                return False, message + " · weld motion failed or interrupted"
            return success, message + " · weld motion completed"

        start = step.get("usable_seam_start")
        goal = step.get("usable_seam_goal")
        if not pose_is_valid(start) or not pose_is_valid(goal):
            return False, "ARC OFF watcher has invalid START/GOAL geometry"
        try:
            delay_s = max(0.0, float(step.get("arc_off_delay_s", 0.0)))
            configured_speed = max(0.0, float(step.get("tcp_speed_m_s", 0.0)))
            path_to_seam_speed_factor = validated_seam_speed_factor(
                step.get("path_to_seam_speed_factor", 1.0)
            )
            settings = validate_digital_weld_settings(step.get("settings") or {})
            expect_native_crater = bool(settings.get("expect_native_crater", True))
        except (TypeError, ValueError):
            return False, "ARC OFF watcher timing is invalid"

        sx, sy, sz = _pose_position_tuple(start)
        gx, gy, gz = _pose_position_tuple(goal)
        vx, vy, vz = gx - sx, gy - sy, gz - sz
        seam_length = math.sqrt(vx * vx + vy * vy + vz * vz)
        if seam_length < 1e-6:
            return False, "ARC OFF watcher seam length is zero"
        tx, ty, tz = vx / seam_length, vy / seam_length, vz / seam_length
        started = time.monotonic()
        last_log = 0.0

        # CRITICAL synchronization gate: motion and ARC ON may start in the
        # same parallel slot, but ARC OFF must not be evaluated until ARC ON
        # has actually reached main_weld + WCR + feed.
        while not host.weld_arc_established_event.is_set():
            if host.sequence_stop_requested:
                host._mark_arc_off_control(fallback="sequence_stop_before_arc_established")
                return False, "ARC OFF watcher interrupted before ARC established"
            if host.weld_arc_on_done_event.is_set() and not host.weld_arc_on_success:
                host._mark_arc_off_control(fallback="arc_on_failed_before_establishment")
                return False, "ARC OFF watcher aborted: ARC ON failed before establishment"
            if host.weld_motion_done_event.is_set():
                host._mark_arc_off_control(
                    fallback="weld_motion_completed_before_arc_established"
                )
                return False, (
                    "ARC OFF watcher aborted: weld motion completed before "
                    "ARC establishment"
                )
            if time.monotonic() - started > 6.0:
                host._mark_arc_off_control(fallback="arc_establishment_gate_timeout")
                return False, "ARC OFF watcher timed out waiting for ARC establishment"
            time.sleep(0.01)

        host.post(
            host.log,
            "ARC OFF watcher ARMED · ARC established confirmed; "
            "actual-TCP monitoring started",
        )

        # Control must use the physically measured TF pose, not CartesianPath
        # PLAN_PREVIEW feedback.  The independent weld-motion recorder logs
        # stamped TF updates through lead-out; this watcher only owns ARC OFF.
        previous_pose = None
        previous_time = None
        filtered_speed = 0.0
        valid_speed_samples = 0
        seen_inside_seam = False

        while not host.sequence_stop_requested:
            try:
                pose = host.node._current_tcp_pose("right_manipulator")
            except TransformException:
                time.sleep(0.02)
                continue

            now = time.monotonic()
            measured_speed = 0.0
            if previous_pose is not None and previous_time is not None:
                dt = now - previous_time
                if dt > 1e-4:
                    ddx = float(pose.position.x) - float(previous_pose.position.x)
                    ddy = float(pose.position.y) - float(previous_pose.position.y)
                    ddz = float(pose.position.z) - float(previous_pose.position.z)
                    raw_speed = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) / dt
                    # Ignore implausible TF/timestamp spikes instead of clipping
                    # them to a huge value that would make ARC OFF immediate.
                    if 0.0 <= raw_speed <= 0.50:
                        if valid_speed_samples == 0:
                            filtered_speed = raw_speed
                        else:
                            filtered_speed = 0.25 * raw_speed + 0.75 * filtered_speed
                        valid_speed_samples += 1
            previous_pose = copy.deepcopy(pose)
            previous_time = now
            measured_speed = max(0.0, filtered_speed)

            dx = float(pose.position.x) - sx
            dy = float(pose.position.y) - sy
            dz = float(pose.position.z) - sz
            along = dx * tx + dy * ty + dz * tz
            remaining = seam_length - along

            # Do not permit a trigger until the actual TCP has at least entered
            # the taught START→GOAL seam interval.
            if 0.0 <= along <= seam_length:
                seen_inside_seam = True

            if configured_speed > 1e-6:
                trigger_speed = configured_speed * path_to_seam_speed_factor
                speed_source = "tcp_setpoint_projected_along_seam"
                speed_ready = True
            else:
                trigger_speed = measured_speed * path_to_seam_speed_factor
                speed_source = "actual_tf_speed_projected_along_seam"
                # Require several real samples so one timestamp jump cannot arm
                # the pre-OFF calculation.
                speed_ready = valid_speed_samples >= 3 and trigger_speed > 1e-4

            if speed_ready:
                arc_off_lead_distance = trigger_speed * delay_s
                # Safety bound: compensation can never exceed 20 mm nor 25% of
                # the usable seam.  A bad speed estimate therefore cannot turn
                # ARC OFF near START.
                max_comp = min(0.020, 0.25 * seam_length)
                arc_off_lead_distance = max(
                    0.0, min(arc_off_lead_distance, max_comp)
                )
                trigger_along = seam_length - arc_off_lead_distance
            else:
                arc_off_lead_distance = 0.0
                trigger_along = seam_length

            if seen_inside_seam and speed_ready and along >= trigger_along:
                host._mark_arc_off_control(
                    delay_s=delay_s,
                    speed_source=speed_source,
                    trigger_speed_m_s=trigger_speed,
                    calculated_pre_off_distance_m=arc_off_lead_distance,
                    crater_source=(
                        "welder_panel_native" if expect_native_crater else "not_expected"
                    ),
                    seam_length_m=seam_length,
                    trigger_along_m=trigger_along,
                    actual_along_m=along,
                    actual_remaining_to_goal_m=remaining,
                    tcp_x_m=float(pose.position.x),
                    tcp_y_m=float(pose.position.y),
                    tcp_z_m=float(pose.position.z),
                )
                crater_message = (
                    "native welder-panel crater after ARC OFF"
                    if expect_native_crater else "native crater not expected; panel reference only"
                )
                success, message = host._execute_hicomm_weld(
                    "off", step.get("settings"), finalize_feedback=False,
                    apply_crater=False,
                )
                return success, (
                    f"pre-GOAL ARC OFF · lead={arc_off_lead_distance * 1000.0:.2f} mm · "
                    f"v={trigger_speed * 1000.0:.2f} mm/s ({speed_source}) · "
                    f"delay={delay_s * 1000.0:.0f} ms · {crater_message} · {message}"
                )

            if now - last_log >= 0.5:
                last_log = now
                host.post(
                    host.log,
                    f"ARC OFF watcher · actual remaining={remaining * 1000.0:.2f} mm · "
                    f"ARC-OFF lead={arc_off_lead_distance * 1000.0:.2f} mm · "
                    f"crater={'expected' if expect_native_crater else 'not expected'} · "
                    f"v={trigger_speed * 1000.0:.2f} mm/s · "
                    f"speed_ready={int(speed_ready)}",
                )

            if host.weld_motion_done_event.is_set():
                # Never leave the arc on if feedback/trigger geometry missed GOAL.
                host._mark_arc_off_control(
                    fallback="weld_motion_completed_before_trigger",
                    delay_s=delay_s,
                )
                success, message = host._execute_hicomm_weld(
                    "off", step.get("settings"), finalize_feedback=False
                )
                return success, "ARC OFF late fallback after motion completion · " + message
            if time.monotonic() - started > 300.0:
                host._mark_arc_off_control(fallback="watch_timeout", delay_s=delay_s)
                success, message = host._execute_hicomm_weld(
                    "off", step.get("settings"), finalize_feedback=False
                )
                return False, "ARC OFF watcher timed out; forced OFF · " + message
            time.sleep(0.01)

        host._mark_arc_off_control(fallback="sequence_stop", delay_s=delay_s)
        success, message = host._execute_hicomm_weld(
            "off", None, finalize_feedback=False
        )
        return False, "ARC OFF watcher interrupted; forced OFF · " + message
