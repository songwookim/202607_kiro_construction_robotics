"""Sequential multi-pass (four-pass) seam registration workflow, independent of Tk.

Moved from ``WeldActionGui`` without changing semantics: loading pass
references (pass_N.yaml overrides / N.log fallbacks) and restoring the newest
cumulative correction, source-hash validation, the guided registration
(START WAIT -> keyboard I capture -> retract -> GOAL WAIT -> J capture ->
``core.multipass.correct_remaining_passes`` -> save -> Weld end), corrected
endpoint verification moves, and selected-pass teaching save/load/apply.

``MultipassController`` holds no state.  ``MultiPassState`` (exposed by the
host's ``four_pass_*`` / ``multi_pass_registration`` properties) remains the
canonical model; keyboard-teaching state (``keyboard_velocity_arm``,
``keyboard_velocity_switching``, ``keyboard_teaching_capture_in_progress``),
teaching poses and the ROS node stay on the host.  Dialogs and widget state
are reached through host hooks called at the original points
(``_confirm_multi_pass_registration``, ``_confirm_corrected_pass_endpoint``,
``_set_keyboard_jog_controls_enabled``, ``_focus_keyboard_teaching``); operator
variables are still read/set through the host.  Sibling operations go through
the host so existing entry points and host-level overrides keep working.
"""

import copy
import hashlib
import math
from pathlib import Path
import threading
import time

from tf2_ros import TransformException
import yaml

from construct_robot.core.cartesian_path_common import (
    named_tcp_linear_waypoints,
    pose_is_valid,
)
from construct_robot.core.multipass import correct_remaining_passes
from construct_robot.core.seam_geometry import _pose_position_tuple
from construct_robot.core.task_teaching_model import TEACHING_POSES
from construct_robot.io.teaching_yaml import (
    _pose_from_yaml_dict,
    atomic_yaml,
    parse_teaching_snapshot_entry,
    read_pass_teaching_reference,
    save_initial_state_yaml,
    save_seam_teaching_reference_yaml,
)
from construct_robot.io.weld_logging import read_weld_pass_reference


class MultipassController:
    """Four-pass registration orchestration against the host's MultiPassState."""

    def __init__(self, host):
        self.host = host

    def load_references(self):
        host = self.host
        folder = Path(host.four_pass_folder.get()).expanduser().resolve()
        try:
            if not folder.is_dir():
                raise ValueError(f"4-pass work folder does not exist: {folder}")

            # Feedback logs are immutable execution evidence.  Editable pass
            # teaching is stored directly in the browsed folder as
            # pass_N.yaml.  The former pass_teaching/pass_N_teaching.yaml
            # layout remains read-only compatible during migration.
            if folder.name == "pass_teaching":
                log_folder = folder.parent
            else:
                log_folder = folder

            references = {}
            yaml_count = 0
            log_count = 0
            missing = []
            for number in range(1, 5):
                teaching_path = folder / f"pass_{number}.yaml"
                legacy_paths = (
                    folder / f"pass_{number}_teaching.yaml",
                    folder / "pass_teaching" / f"pass_{number}_teaching.yaml",
                )
                log_path = log_folder / f"{number}.log"
                if teaching_path.is_file():
                    references[number] = read_pass_teaching_reference(
                        teaching_path, number
                    )
                    yaml_count += 1
                elif any(path.is_file() for path in legacy_paths):
                    legacy_path = next(
                        path for path in legacy_paths if path.is_file()
                    )
                    references[number] = read_pass_teaching_reference(
                        legacy_path, number
                    )
                    yaml_count += 1
                elif log_path.is_file():
                    references[number] = read_weld_pass_reference(log_path)
                    log_count += 1
                else:
                    missing.append(
                        f"Pass {number}: {teaching_path} or {log_path}"
                    )
            if missing:
                raise ValueError(
                    "No teaching YAML or fallback weld log for "
                    + " · ".join(missing)
                )
            if yaml_count == 4:
                reference_set_kind = "4 pass-teaching YAML overrides"
            elif log_count == 4:
                reference_set_kind = "4 completed weld logs"
            else:
                reference_set_kind = (
                    f"{yaml_count} pass-teaching YAML override(s) + "
                    f"{log_count} weld-log fallback(s)"
                )
        except (OSError, ValueError, yaml.YAMLError) as error:
            host.four_pass_references = {}
            host.four_pass_loaded_folder = None
            host.error(f"Cannot load 4-pass references: {error}")
            return False
        host.four_pass_references = references
        host.four_pass_loaded_folder = folder
        host.four_pass_corrected = {
            number: {
                endpoint: copy.deepcopy(reference[endpoint])
                for endpoint in ("start_wait", "start", "goal_wait", "goal")
            }
            for number, reference in references.items()
        }
        restored = host._load_latest_sequential_four_pass_state(
            folder, references
        )
        if restored is None:
            host.four_pass_output_folder = None
            histories = [
                reference.get("correction_history", [])
                for reference in references.values()
            ]
            host.four_pass_history = copy.deepcopy(
                max(histories, key=len, default=[])
            )
        else:
            (
                host.four_pass_corrected,
                host.four_pass_output_folder,
                host.four_pass_history,
            ) = restored
        host.multi_pass_registration = None
        lengths = [
            1000.0 * math.dist(
                _pose_position_tuple(references[number]["start"]),
                _pose_position_tuple(references[number]["goal"]),
            ) for number in range(1, 5)
        ]
        restore_note = (
            f" · resumed corrections from {host.four_pass_output_folder.name}"
            if host.four_pass_output_folder is not None
            else " · no saved cumulative correction; using loaded reference set"
        )
        host._set_four_pass_status(
            f"Loaded {reference_set_kind} for pass 1–4 · seam lengths "
            + "/".join(f"{length:.1f}" for length in lengths)
            + f" mm{restore_note}"
        )
        host.log(
            f"4-PASS REFERENCES LOADED · {folder} · {reference_set_kind} · "
            "logs immutable / teaching editable in YAML · "
            f"lengths={lengths} mm"
        )
        return True

    def load_latest_sequential_state(self, folder, references):
        """Restore the newest valid cumulative correction for these source logs."""
        host = self.host
        # Canonical pass YAMLs already contain the current corrected/manual
        # teaching. Source ancestry hashes are not evidence that an older
        # manifest should overwrite those newly saved poses.
        if any(
            reference.get("reference_kind") == "saved_pass_teaching"
            for reference in references.values()
        ):
            return None
        candidates = sorted(
            [
                path
                for path in (
                    folder / "manifest.yaml",
                    *folder.glob("sequential_corrected_*/manifest.yaml"),
                )
                if path.is_file()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for manifest_path in candidates:
            try:
                manifest = yaml.load(
                    manifest_path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader
                ) or {}
                schema = manifest.get("schema")
                if schema not in (
                    "construct_robot_sequential_four_pass_correction_v2",
                    "construct_robot_sequential_four_pass_correction_v3",
                ):
                    continue
                entries = {
                    int(entry["pass"]): entry["file"]
                    for entry in manifest.get("passes", ())
                }
                if set(entries) != {1, 2, 3, 4}:
                    raise ValueError("manifest does not list exactly Pass 1..4")
                records = {}
                for number in range(1, 5):
                    record = yaml.load(
                        (manifest_path.parent / entries[number]).read_text(
                            encoding="utf-8"
                        ), Loader=yaml.CSafeLoader
                    ) or {}
                    accepted_hashes = {references[number]["sha256"]}
                    for key in ("source_reference_sha256", "source_log_sha256"):
                        source_hash = references[number].get(key)
                        if source_hash:
                            accepted_hashes.add(source_hash)
                    record_source_hash = record.get("source_log_sha256") or record.get(
                        "source_reference_sha256"
                    )
                    if record_source_hash not in accepted_hashes:
                        raise ValueError(
                            f"Pass {number} source hash no longer matches"
                        )
                    records[number] = record
                history = manifest.get("history", [])
                if not isinstance(history, list):
                    raise ValueError("manifest history is not a list")
                if schema.endswith("_v3"):
                    if all(
                        f"current_{endpoint}" in records[number]
                        for number in range(1, 5)
                        for endpoint in ("start_wait", "start", "goal_wait", "goal")
                    ):
                        corrected = {
                            number: {
                                endpoint: _pose_from_yaml_dict(
                                    records[number][f"current_{endpoint}"],
                                    f"Pass {number} current {endpoint}",
                                )
                                for endpoint in (
                                    "start_wait", "start", "goal_wait", "goal"
                                )
                            }
                            for number in range(1, 5)
                        }
                    else:
                        corrected = {
                            number: {
                                endpoint: read_pass_teaching_reference(
                                    manifest_path.parent / entries[number], number
                                )[endpoint]
                                for endpoint in (
                                    "start_wait", "start", "goal_wait", "goal"
                                )
                            }
                            for number in range(1, 5)
                        }
                else:
                    # v2 stored corrected START/GOAL only. Replay its measured
                    # anchor events on today's full log references so the
                    # pass-specific WAIT poses receive identical transforms.
                    corrected = {
                        number: {
                            endpoint: copy.deepcopy(references[number][endpoint])
                            for endpoint in (
                                "start_wait", "start", "goal_wait", "goal"
                            )
                        }
                        for number in range(1, 5)
                    }
                    for event in history:
                        anchor = int(event["selected_pass"])
                        measured_start = _pose_from_yaml_dict(
                            event["measured_start"],
                            f"Pass {anchor} v2 measured START",
                        )
                        measured_goal = _pose_from_yaml_dict(
                            event["measured_goal"],
                            f"Pass {anchor} v2 measured GOAL",
                        )
                        corrected, _transform = correct_remaining_passes(
                            corrected, anchor, measured_start, measured_goal
                        )
            except (
                KeyError, OSError, TypeError, ValueError, yaml.YAMLError
            ) as error:
                host.log(
                    f"Skipped invalid cumulative correction {manifest_path}: {error}"
                )
                continue
            host.log(
                f"RESTORED CUMULATIVE 4-PASS CORRECTION · {manifest_path.parent} · "
                f"events={len(history)}"
            )
            return corrected, manifest_path.parent, copy.deepcopy(history)
        return None

    def run_correction(self):
        host = self.host
        if host.multi_pass_registration is not None:
            host.error("A multi-pass registration is already in progress")
            return
        if host.sequence_running or host.node.active_motion_goal is not None:
            host.error("Wait for the current robot motion to finish")
            return
        if host.keyboard_velocity_arm is not None or host.keyboard_velocity_switching:
            host.error("Disable Keyboard Teaching before moving to START WAIT")
            return
        if (
            not host.four_pass_references
            or Path(host.four_pass_folder.get()).expanduser().resolve()
            != host.four_pass_loaded_folder
        ) and not host.load_four_pass_references():
            return
        try:
            number = host._selected_pass()
            if number not in (1, 2, 3, 4):
                raise ValueError("Select Pass 1, 2, 3, or 4")
            host._validate_four_pass_source_hashes()
            waits = {
                endpoint: host._multi_pass_translated_wait(number, endpoint)
                for endpoint in ("start", "goal")
            }
            end_entry = host.four_pass_references[number].get(
                "additional_pose_entries", {}
            ).get("weld_finish")
            if end_entry is None:
                raise ValueError("Save this pass's Weld end pose before correction")
            end_group, _, _, end_pose = parse_teaching_snapshot_entry(
                "weld_finish", end_entry
            )
            if end_group != "right_manipulator":
                raise ValueError("Weld end pose must belong to the right arm")
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            host.error(f"Cannot start multi-pass correction: {error}")
            return
        if not host.execution_allowed or not host.robot_connected.get("right", False):
            host.error("Connect the right robot and enable physical execution")
            return
        if not host._confirm_multi_pass_registration(number):
            return
        host.multi_pass_registration = {
            "pass": number,
            "phase": "moving_start_wait",
            "previous": copy.deepcopy(host.four_pass_corrected),
            "waits": waits,
            "end_pose": copy.deepcopy(end_pose),
            "velocity_scale": max(
                0.01, min(1.0, float(host.velocity_percent.get()) / 100.0)
            ),
            "measured_start": None,
            "measured_goal": None,
        }
        host._set_four_pass_status(
            f"Pass {number} correction · moving to corrected logged START WAIT"
        )
        threading.Thread(
            target=host._multi_pass_start_wait_worker,
            args=(number, copy.deepcopy(waits["start"])),
            daemon=True,
        ).start()

    def validate_source_hashes(self):
        host = self.host
        if set(host.four_pass_references) != {1, 2, 3, 4}:
            raise ValueError("Load all four pass references first")
        for number, reference in host.four_pass_references.items():
            if hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest() != reference["sha256"]:
                raise ValueError(
                    f"Pass {number} reference changed after loading; reload references"
                )

    def translated_wait(self, number, endpoint):
        host = self.host
        if set(host.four_pass_corrected) != {1, 2, 3, 4}:
            raise ValueError("Current four-pass prediction is unavailable")
        wait_endpoint = f"{endpoint}_wait"
        selected = host.four_pass_corrected[number][endpoint]
        translated = copy.deepcopy(host.four_pass_corrected[number][wait_endpoint])
        selected_separation = math.dist(
            _pose_position_tuple(translated), _pose_position_tuple(selected)
        )
        if selected_separation < 0.001:
            raise ValueError(
                f"Pass {number} logged/corrected {wait_endpoint.upper()} "
                f"is indistinguishable from {endpoint.upper()} · separation "
                f"{selected_separation * 1000.0:.1f} mm"
            )
        if selected_separation < 0.020:
            host.log(
                f"4-PASS WAIT CLEARANCE WARNING · Pass {number} "
                f"{wait_endpoint.upper()} is only "
                f"{selected_separation * 1000.0:.1f} mm from "
                f"{endpoint.upper()} · using the pass log value as requested · "
                "verify the collision scene and keep STOP accessible"
            )
        return translated

    def run_tcp_move(
        self, target, label, velocity_scale, touch_guard=False
    ):
        host = self.host
        try:
            current = host.node._path_start_tcp_pose("right_manipulator")
            points = named_tcp_linear_waypoints(current, target)
        except (TransformException, ValueError) as error:
            return False, f"{label} path failed: {error}"
        return host.node.run_sequence_cartesian_motion({
            "planning_group": "right_manipulator",
            "interpolation_step": 0.005,
            "velocity_scale": float(velocity_scale),
            "tcp_speed_m_s": 0.0,
            "points": points,
            "path_kind": label,
            "touch_guard": bool(touch_guard),
            "continue_after_touch": False,
            "allow_initial_touch_motion": False,
        }, True)

    def start_wait_worker(self, number, target):
        host = self.host
        session = host.multi_pass_registration
        if session is None or session["pass"] != number:
            return
        success, message = host._run_multi_pass_tcp_move(
            target,
            f"Pass {number} corrected logged START WAIT",
            session["velocity_scale"],
        )
        host.post(host._multi_pass_start_wait_finished, number, success, message)

    def start_wait_finished(self, number, success, message):
        host = self.host
        session = host.multi_pass_registration
        if session is None or session["pass"] != number:
            return
        if not success:
            host.multi_pass_registration = None
            host.error(f"Pass {number} START WAIT move failed: {message}")
            return
        session["phase"] = "waiting_start_capture"
        host._set_four_pass_status(
            f"Pass {number} correction · waiting for START capture (I) · "
            "enable Keyboard Teaching and jog to the real START"
        )
        host.pipeline_result(
            f"Pass {number} corrected logged START WAIT reached · "
            "no welding command sent"
        )
        host._enable_multi_pass_keyboard_teaching(number, "i")

    def enable_keyboard_teaching(self, number, expected_key):
        """Enable the existing keyboard controller for the next I/J capture."""
        host = self.host
        session = host.multi_pass_registration
        if session is None or session["pass"] != number:
            return
        expected_key = str(expected_key).lower()
        if expected_key not in ("i", "j"):
            raise ValueError("Multi-pass capture key must be I or J")
        if host.keyboard_velocity_arm == "right" and not host.keyboard_velocity_switching:
            host._focus_keyboard_teaching()
            host.keyboard_jog_enabled.set(True)
            host.keyboard_jog_status.set(
                f"READY RIGHT · jog then press {expected_key.upper()} to capture"
            )
            return
        if host.keyboard_velocity_switching:
            host._set_four_pass_status(
                f"Pass {number} correction · waiting for Keyboard Teaching · "
                f"then press {expected_key.upper()}"
            )
            return
        host.keyboard_jog_enabled.set(True)
        host._set_four_pass_status(
            f"Pass {number} correction · enabling Keyboard Teaching for "
            f"{expected_key.upper()} capture"
        )
        host.keyboard_jog_status.set(
            f"AUTO ENABLE · preparing {expected_key.upper()} capture..."
        )
        host.keyboard_jog_enable_changed()

    def finish_keyboard_capture(self, key, captured, error):
        host = self.host
        host.keyboard_teaching_capture_in_progress = False
        session = host.multi_pass_registration
        if session is None:
            host.error("Multi-pass capture arrived after the session ended")
            return
        number = session["pass"]
        expected_key = "i" if session["phase"] == "waiting_start_capture" else "j"
        if session["phase"] not in ("waiting_start_capture", "waiting_goal_capture"):
            host.error("Wait for automatic multi-pass motion to finish before capturing")
            return
        if key != expected_key:
            host.error(
                f"Pass {number} expects {expected_key.upper()} capture, not {key.upper()}"
            )
            return
        if error is not None:
            host.keyboard_jog_status.set(f"{key.upper()} · capture rejected")
            host._set_four_pass_status(
                f"Pass {number} correction · {key.upper()} capture FAILED · retry"
            )
            host.error(f"Pass {number} {key.upper()} capture rejected: {error}")
            return
        _joint_names, _positions, pose, provenance = captured
        if key == "i":
            session["measured_start"] = copy.deepcopy(pose)
            session["start_capture_provenance"] = copy.deepcopy(provenance)
            session["phase"] = "moving_goal_wait"
            host.keyboard_velocity_switching = True
            host._set_keyboard_jog_controls_enabled(False)
            host._set_four_pass_status(
                f"Pass {number} correction · I accepted / START captured · "
                "restoring trajectory controller and moving to GOAL WAIT"
            )
            host.keyboard_jog_status.set(
                f"I COMPLETE · Pass {number} START saved · moving to GOAL WAIT"
            )
            threading.Thread(
                target=host._multi_pass_goal_wait_worker,
                args=(number, copy.deepcopy(session["waits"]["goal"])),
                daemon=True,
            ).start()
            return
        session["measured_goal"] = copy.deepcopy(pose)
        session["goal_capture_provenance"] = copy.deepcopy(provenance)
        host._set_four_pass_status(
            f"Pass {number} correction · J accepted / GOAL captured · "
            "calculating cumulative correction"
        )
        host.keyboard_jog_status.set(
            f"J COMPLETE · Pass {number} GOAL saved"
        )
        host._complete_multi_pass_registration()

    def goal_wait_worker(self, number, target):
        host = self.host
        session = host.multi_pass_registration
        if session is None or session["pass"] != number:
            return
        host.node.clear_keyboard_velocity()
        time.sleep(0.10)
        switched, switch_message = host.node.set_keyboard_velocity_controller_enabled(
            "right", False
        )
        if not switched:
            host.post(
                host._multi_pass_goal_wait_finished,
                number, False,
                f"trajectory controller restore failed: {switch_message}",
            )
            return
        if host.multi_pass_registration is not session:
            return
        # Leave the workpiece along the pass log's corrected START-WAIT route before
        # traversing to the far GOAL WAIT.  A direct real-START -> GOAL-WAIT
        # Cartesian segment can cut through the groove or an existing bead.
        success, message = host._run_multi_pass_tcp_move(
            copy.deepcopy(session["waits"]["start"]),
            f"Pass {number} retract to corrected logged START WAIT",
            session["velocity_scale"],
        )
        if not success:
            host.post(
                host._multi_pass_goal_wait_finished,
                number, False, f"START WAIT retract failed: {message}",
            )
            return
        if host.multi_pass_registration is not session:
            return
        success, message = host._run_multi_pass_tcp_move(
            target,
            f"Pass {number} corrected logged GOAL WAIT",
            session["velocity_scale"],
            touch_guard=True,
        )
        host.post(host._multi_pass_goal_wait_finished, number, success, message)

    def goal_wait_finished(self, number, success, message):
        host = self.host
        host.keyboard_velocity_switching = False
        host.keyboard_velocity_arm = None
        host.keyboard_jog_enabled.set(False)
        host._set_keyboard_jog_controls_enabled(True)
        host.keyboard_jog_status.set("Keyboard teaching locked")
        session = host.multi_pass_registration
        if session is None or session["pass"] != number:
            return
        if not success:
            host.multi_pass_registration = None
            host.error(f"Pass {number} GOAL WAIT move failed: {message}")
            return
        session["phase"] = "waiting_goal_capture"
        host._set_four_pass_status(
            f"Pass {number} correction · START captured · "
            "waiting for GOAL capture (J) · enable Keyboard Teaching"
        )
        host.pipeline_result(
            f"Pass {number} corrected logged GOAL WAIT reached · "
            "no welding command sent"
        )
        host._enable_multi_pass_keyboard_teaching(number, "j")

    def complete_registration(self):
        host = self.host
        session = host.multi_pass_registration
        if session is None:
            return
        number = session["pass"]
        try:
            host._validate_four_pass_source_hashes()
            corrected, transform = correct_remaining_passes(
                session["previous"],
                number,
                session["measured_start"],
                session["measured_goal"],
            )
            host._save_sequential_four_pass_state(
                corrected, session, transform
            )
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            host.error(f"Pass {number} correction failed: {error}")
            return
        host.four_pass_corrected = corrected
        if "end_pose" in session:
            session["phase"] = "moving_end"
            host.keyboard_velocity_switching = True
            host._set_keyboard_jog_controls_enabled(False)
            host._set_four_pass_status(
                f"Pass {number} correction saved · moving to Weld end"
            )
            threading.Thread(
                target=host._multi_pass_end_worker, args=(session,), daemon=True
            ).start()
            return
        host.multi_pass_registration = None
        later = transform["later_passes_updated"]
        later_text = "/".join(str(value) for value in later) or "none"
        host._set_four_pass_status(
            f"Pass {number} corrected · later predictions updated: {later_text} · "
            "verify corrected START/GOAL before welding"
        )
        host.pipeline_result(
            f"SEQUENTIAL PASS {number} REGISTRATION COMPLETE · "
            f"direction change={transform['direction_change_deg']:+.3f}° · "
            f"updated later passes={later_text} · ARC/WELD not started"
        )
        host.keyboard_jog_status.set(
            f"PASS {number} CORRECTION COMPLETE · verify corrected START/GOAL"
        )

    def end_worker(self, session):
        host = self.host
        try:
            success, message = host.node.set_keyboard_velocity_controller_enabled("right", False)
            if success and host.multi_pass_registration is session:
                success, message = host._run_multi_pass_tcp_move(
                    session["end_pose"], f"Pass {session['pass']} Weld end",
                    session["velocity_scale"], touch_guard=False,
                )
        except Exception as error:
            success, message = False, str(error)
        host.post(host._multi_pass_end_finished, session, success, message)

    def end_finished(self, session, success, message):
        host = self.host
        host.keyboard_velocity_switching = False
        host._set_keyboard_jog_controls_enabled(True)
        host.keyboard_jog_enabled.set(False)
        host.keyboard_velocity_arm = None
        if host.multi_pass_registration is not session:
            return
        host.multi_pass_registration = None
        text = (
            f"Pass {session['pass']} correction saved · Weld end reached"
            if success else f"Correction saved, but Weld end move stopped/failed: {message}"
        )
        host._set_four_pass_status(text)
        host.keyboard_jog_status.set("Keyboard teaching locked")
        (host.pipeline_result if success else host.error)(text)

    def save_sequential_state(self, corrected, session, transform):
        host = self.host
        folder = host.four_pass_loaded_folder
        if folder is None:
            raise ValueError("Four-pass source folder is unavailable")
        output = folder
        number = session["pass"]
        previous = session["previous"]
        pose_dict = host._pose_execution_conditions
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        event = {
            "timestamp": timestamp,
            "status": "measured_anchor_applied",
            "selected_pass": number,
            "source_log": host.four_pass_references[number]["path"],
            "source_log_sha256": host.four_pass_references[number]["sha256"],
            "previous_predicted_start_wait": pose_dict(
                previous[number]["start_wait"]
            ),
            "previous_predicted_start": pose_dict(previous[number]["start"]),
            "previous_predicted_goal_wait": pose_dict(
                previous[number]["goal_wait"]
            ),
            "previous_predicted_goal": pose_dict(previous[number]["goal"]),
            "measured_start": pose_dict(session["measured_start"]),
            "measured_goal": pose_dict(session["measured_goal"]),
            "direction_change_deg": transform["direction_change_deg"],
            "start_translation_mm": [
                value * 1000.0 for value in transform["start_translation_m"]
            ],
            "goal_translation_mm": [
                value * 1000.0 for value in transform["goal_translation_m"]
            ],
            "rotation_xyzw": list(transform["rotation_xyzw"]),
            "later_passes_updated": list(transform["later_passes_updated"]),
            "orientation_policy": (
                "q_new = q_minimal_direction_rotation * q_current; "
                "no additional seam-axis roll"
            ),
            "wait_orientation_policy": (
                "pass-specific WAIT from source log; corrected with the same "
                "minimal seam rotation"
            ),
            "start_capture_provenance": session.get("start_capture_provenance"),
            "goal_capture_provenance": session.get("goal_capture_provenance"),
        }
        history = [*host.four_pass_history, event]
        manifest = {
            "schema": "construct_robot_sequential_four_pass_correction_v3",
            "status": "sequential_pass_registration",
            "planning_group": "right_manipulator",
            "source_folder": str(folder),
            "source_logs_immutable": True,
            "current_anchor_pass": number,
            "wait_pose_source": "pass-specific teaching snapshot in each N.log",
            "history": history,
            "passes": [],
        }

        for pass_number in range(1, 5):
            reference = host.four_pass_references[pass_number]
            record = {
                "schema": "construct_robot_sequential_pass_v3",
                "pass": pass_number,
                "status": (
                    "measured_anchor"
                    if pass_number == number
                    else (
                        f"propagated_from_pass_{number}"
                        if pass_number > number
                        else "previously_registered_unchanged"
                    )
                ),
                "source_log": reference["path"],
                "source_log_sha256": reference["sha256"],
                "source_start_wait": pose_dict(reference["start_wait"]),
                "source_start": pose_dict(reference["start"]),
                "source_goal_wait": pose_dict(reference["goal_wait"]),
                "source_goal": pose_dict(reference["goal"]),
                "current_start_wait": pose_dict(
                    corrected[pass_number]["start_wait"]
                ),
                "current_start": pose_dict(corrected[pass_number]["start"]),
                "current_goal_wait": pose_dict(
                    corrected[pass_number]["goal_wait"]
                ),
                "current_goal": pose_dict(corrected[pass_number]["goal"]),
                "last_registration": event if pass_number >= number else None,
            }
            file_name = f"pass_{pass_number}.yaml"
            atomic_yaml(output / file_name, record)
            manifest["passes"].append({"pass": pass_number, "file": file_name})

            # The selected work folder is the operational source of truth.
            # Rewrite all four pass files after every correction so Pass 1
            # registration immediately propagates to Passes 2..4 without an
            # extra export/load step.  Joint snapshots are retained only as
            # IK seeds; corrected Cartesian poses must be solved again.
            joint_states = reference.get("joint_states", {})
            pose_entries = {}
            for endpoint, pose_name in (
                ("start_wait", "weld_start_wait"),
                ("start", "weld_start"),
                ("goal_wait", "weld_goal_wait"),
                ("goal", "weld_end"),
            ):
                names, positions = joint_states[endpoint]
                pose_entries[pose_name] = {
                    "planning_group": "right_manipulator",
                    "joint_state": {
                        "names": list(names),
                        "positions_rad": [float(value) for value in positions],
                    },
                    "tcp_pose_world": pose_dict(
                        corrected[pass_number][endpoint]
                    ),
                }
            canonical = {
                "schema": "construct_robot_pass_teaching_v1",
                "status": record["status"],
                "timestamp": timestamp,
                "pass": pass_number,
                "requires_ik": True,
                "correction_anchor_pass": number,
                "correction_history": copy.deepcopy(history),
                "poses": pose_entries,
            }
            if pass_number == number:
                additional_pose_entries = copy.deepcopy(
                    reference.get("additional_pose_entries", {})
                )
                for pose_name in ("robot_start", "weld_wait", "weld_finish"):
                    stored = host.taught_robot_poses.get(pose_name)
                    if stored is None or stored[0] != "right_manipulator":
                        continue
                    group, names, positions, tcp = stored
                    if len(names) != 6 or len(positions) != 6 or not pose_is_valid(tcp):
                        continue
                    additional_pose_entries[pose_name] = {
                        "planning_group": group,
                        "joint_state": {
                            "names": list(names),
                            "positions_rad": [float(value) for value in positions],
                        },
                        "tcp_pose_world": pose_dict(tcp),
                    }
            else:
                additional_pose_entries = reference.get(
                    "additional_pose_entries", {}
                )
            canonical["poses"].update(copy.deepcopy(additional_pose_entries))
            canonical_path = folder / f"pass_{pass_number}.yaml"
            if Path(reference["path"]).resolve() != canonical_path.resolve():
                canonical.update({
                    "source_reference": reference["path"],
                    "source_reference_kind": reference.get(
                        "reference_kind", "unknown"
                    ),
                    "source_reference_sha256": reference["sha256"],
                })
            atomic_yaml(canonical_path, canonical)
        atomic_yaml(output / "manifest.yaml", manifest)
        host.four_pass_references = {
            pass_number: read_pass_teaching_reference(
                folder / f"pass_{pass_number}.yaml", pass_number
            )
            for pass_number in range(1, 5)
        }
        host.four_pass_output_folder = output
        host.four_pass_history = history

    def go_to_corrected_endpoint(self, endpoint):
        host = self.host
        endpoint = str(endpoint).strip().lower()
        try:
            number = host._selected_pass()
            if endpoint not in ("start", "goal"):
                raise ValueError("Endpoint must be START or GOAL")
            target = copy.deepcopy(host.four_pass_corrected[number][endpoint])
            host._validate_four_pass_source_hashes()
        except (KeyError, OSError, TypeError, ValueError) as error:
            host.error(f"Cannot move to corrected endpoint: {error}")
            return
        if host.multi_pass_registration is not None:
            host.error("Finish the active multi-pass registration first")
            return
        if host.keyboard_velocity_arm is not None or host.keyboard_velocity_switching:
            host.error("Disable Keyboard Teaching before corrected-pose motion")
            return
        if host.sequence_running or host.node.active_motion_goal is not None:
            host.error("Another robot motion is active")
            return
        if not host.execution_allowed or not host.robot_connected.get("right", False):
            host.error("Connect the right robot and enable physical execution")
            return
        if not host._confirm_corrected_pass_endpoint(number, endpoint):
            return
        speed = max(0.01, min(1.0, float(host.velocity_percent.get()) / 100.0))
        threading.Thread(
            target=host._go_to_corrected_pass_endpoint_worker,
            args=(number, endpoint, target, speed),
            daemon=True,
        ).start()

    def corrected_endpoint_worker(
        self, number, endpoint, target, velocity_scale
    ):
        host = self.host
        success, message = host._run_multi_pass_tcp_move(
            target,
            f"Pass {number} corrected {endpoint.upper()} verification",
            velocity_scale,
            touch_guard=True,
        )
        host.post(
            host._go_to_corrected_pass_endpoint_finished,
            number, endpoint, success, message,
        )

    def corrected_endpoint_finished(
        self, number, endpoint, success, message
    ):
        host = self.host
        if success:
            host.pipeline_result(
                f"Pass {number} corrected {endpoint.upper()} reached · "
                "visual verification only · no welding command sent"
            )
        else:
            host.error(
                f"Pass {number} corrected {endpoint.upper()} move failed: {message}"
            )

    def selected_pass_teaching_path(self, number):
        host = self.host
        folder_field = getattr(host, "four_pass_folder", None)
        if folder_field is not None:
            text = folder_field.get().strip()
            if not text:
                raise ValueError("Select a pass folder before saving")
            folder = Path(text).expanduser().resolve()
        else:
            folder = host.four_pass_loaded_folder
        return folder / f"pass_{int(number)}.yaml"

    def load_saved_pass_teaching(self, number):
        """Return a selected pass's independent manual teaching, if present."""
        host = self.host
        path = host._selected_pass_teaching_path(number)
        if not path.is_file():
            return None
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader) or {}
        if document.get("schema") != "construct_robot_pass_teaching_v1":
            raise ValueError(f"Unsupported pass teaching schema: {path}")
        if int(document.get("pass", 0)) != int(number):
            raise ValueError(f"Saved teaching pass does not match Pass {number}")
        pose_entries = document.get("poses")
        if not isinstance(pose_entries, dict):
            raise ValueError(f"Pass {number} saved teaching has no poses mapping")
        current = {}
        joint_states = {}
        for pose_name in ("robot_start", "weld_wait", "weld_finish"):
            host.taught_robot_poses[pose_name] = None
        for endpoint, pose_name in (
            ("start_wait", "weld_start_wait"),
            ("start", "weld_start"),
            ("goal_wait", "weld_goal_wait"),
            ("goal", "weld_end"),
        ):
            group, names, positions, tcp = parse_teaching_snapshot_entry(
                pose_name, pose_entries.get(pose_name)
            )
            if group != "right_manipulator":
                raise ValueError(f"{pose_name} is not a right-arm teaching pose")
            current[endpoint] = copy.deepcopy(tcp)
            joint_states[endpoint] = (tuple(names), tuple(positions))
        for pose_name in ("robot_start", "weld_wait", "weld_finish"):
            entry = pose_entries.get(pose_name)
            if entry is None:
                continue
            group, names, positions, tcp = parse_teaching_snapshot_entry(
                pose_name, entry
            )
            if group != "right_manipulator":
                raise ValueError(f"{pose_name} is not a right-arm teaching pose")
            host.taught_robot_poses[pose_name] = (
                group, tuple(names), tuple(positions), copy.deepcopy(tcp)
            )
        host.four_pass_corrected[number] = copy.deepcopy(current)
        return current, joint_states, path

    def save_teaching_to_selected_pass(self):
        """Save current Teaching Detail poses without modifying N.log sources."""
        host = self.host
        try:
            number = host._selected_pass()
            if number not in (1, 2, 3, 4):
                raise ValueError("Select Pass 1, 2, 3, or 4")
            pose_records = {}
            current = {}
            for endpoint, pose_name in (
                ("start_wait", "weld_start_wait"),
                ("start", "weld_start"),
                ("goal_wait", "weld_goal_wait"),
                ("goal", "weld_end"),
            ):
                stored = host.taught_robot_poses.get(pose_name)
                if stored is None or stored[0] != "right_manipulator":
                    raise ValueError(
                        f"Teaching Detail has no right-arm {TEACHING_POSES[pose_name]}"
                    )
                group, names, positions, tcp = stored
                if len(names) != 6 or len(positions) != 6 or not pose_is_valid(tcp):
                    raise ValueError(
                        f"Teaching Detail {TEACHING_POSES[pose_name]} is incomplete"
                    )
                pose_records[pose_name] = {
                    "planning_group": group,
                    "joint_state": {
                        "names": list(names),
                        "positions_rad": [float(value) for value in positions],
                    },
                    "tcp_pose_world": host._pose_execution_conditions(tcp),
                }
                provenance = getattr(
                    self, "teaching_capture_provenance", {}
                ).get(pose_name)
                if provenance:
                    pose_records[pose_name]["capture_provenance"] = copy.deepcopy(
                        provenance
                    )
                current[endpoint] = copy.deepcopy(tcp)
            for pose_name in ("robot_start", "weld_wait", "weld_finish"):
                stored = host.taught_robot_poses.get(pose_name)
                if stored is None or stored[0] != "right_manipulator":
                    continue
                group, names, positions, tcp = stored
                if len(names) != 6 or len(positions) != 6 or not pose_is_valid(tcp):
                    continue
                pose_records[pose_name] = {
                    "planning_group": group,
                    "joint_state": {
                        "names": list(names),
                        "positions_rad": [float(value) for value in positions],
                    },
                    "tcp_pose_world": host._pose_execution_conditions(tcp),
                }
                provenance = getattr(
                    self, "teaching_capture_provenance", {}
                ).get(pose_name)
                if provenance:
                    pose_records[pose_name]["capture_provenance"] = copy.deepcopy(
                        provenance
                    )
            reference = host.four_pass_references.get(number)
            document = {
                "schema": "construct_robot_pass_teaching_v1",
                "status": "manual_teaching_saved",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "pass": number,
                "requires_ik": False,
                "poses": pose_records,
            }
            if reference is not None:
                document.update({
                    "source_reference": reference["path"],
                    "source_reference_kind": reference.get(
                        "reference_kind", "unknown"
                    ),
                    "source_reference_sha256": reference["sha256"],
                    # Retain the v1 provenance keys for existing files and
                    # correction manifests.  They do not make saving depend
                    # on a log being loaded.
                    "source_log": reference["path"],
                    "source_log_sha256": reference["sha256"],
                })
            path = host._selected_pass_teaching_path(number)
            atomic_yaml(path, document)
            persisted = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader) or {}
            if persisted != document:
                raise OSError(f"Pass teaching YAML read-back failed: {path}")
            replaces_loaded_reference = (
                reference is not None
                and reference.get("reference_kind") == "saved_pass_teaching"
                and Path(reference["path"]).resolve() == path.resolve()
            )
            if replaces_loaded_reference:
                host.four_pass_references[number] = read_pass_teaching_reference(
                    path, number
                )
        except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
            host.error(f"Cannot save selected-pass teaching: {error}")
            return
        host.four_pass_corrected[number] = current
        host._set_four_pass_status(
            f"Pass {number} Teaching Detail saved · {path}"
        )
        host.pipeline_result(
            f"PASS {number} TEACHING SAVED · WAIT/START/GOAL WAIT/GOAL · "
            f"{path} · "
            + (
                "loaded teaching reference updated"
                if replaces_loaded_reference
                else "saved independently of pass-reference loading"
            )
        )

    def apply_selected_pass_correction(self):
        """Load the selected pass from the cumulative seam-correction state."""
        host = self.host
        host._apply_selected_pass_teaching(use_saved_teaching=False)

    def load_saved_teaching_for_selected_pass(self):
        """Load the selected pass's explicit manual-teaching override."""
        host = self.host
        host._apply_selected_pass_teaching(use_saved_teaching=True)

    def apply_selected_pass_teaching(self, use_saved_teaching):
        """Apply either cumulative correction or explicit saved teaching."""
        host = self.host
        try:
            number = host._selected_pass()
            if number not in (1, 2, 3, 4):
                raise ValueError("Select Pass 1, 2, 3, or 4")
            if not use_saved_teaching:
                host._validate_four_pass_source_hashes()
                current = host.four_pass_corrected[number]
                reference = host.four_pass_references[number]
                joint_states = reference.get(
                    "joint_states", {}
                )
                teaching_source = "latest cumulative seam correction"
                resolve_corrected_ik = (
                    bool(reference.get("requires_ik"))
                    if reference.get("reference_kind") == "saved_pass_teaching"
                    else True
                )
            else:
                saved = host._load_saved_pass_teaching(number)
                if saved is None:
                    raise ValueError(
                        f"Pass {number} has no separately saved teaching file"
                    )
                current, joint_states, saved_path = saved
                reference = read_pass_teaching_reference(saved_path, number)
                teaching_source = f"saved pass teaching {saved_path}"
                resolve_corrected_ik = bool(reference.get("requires_ik"))
            additional_poses = {}
            for pose_name in ("robot_start", "weld_wait", "weld_finish"):
                entry = reference.get("additional_pose_entries", {}).get(pose_name)
                if entry is None:
                    additional_poses[pose_name] = None
                    continue
                stored = parse_teaching_snapshot_entry(pose_name, entry)
                if stored[0] != "right_manipulator":
                    raise ValueError(f"Pass {number} {pose_name} is not a right-arm pose")
                additional_poses[pose_name] = copy.deepcopy(stored)
            required_endpoints = {"start_wait", "start", "goal_wait", "goal"}
            if not required_endpoints.issubset(joint_states):
                raise ValueError(
                    f"{number}.log has no complete WAIT/START/GOAL WAIT/GOAL "
                    "joint snapshots"
                )
        except (KeyError, OSError, TypeError, ValueError) as error:
            host.error(f"Cannot apply selected pass: {error}")
            return
        host._invalidate_seam_correction_runtime(
            f"applying cumulative Pass {number} correction", clear_touches=True
        )
        # Replace the whole pass-specific teaching context.  Missing optional
        # poses must not inherit the previously selected pass's teaching.
        for pose_name, stored in additional_poses.items():
            host.taught_robot_poses[pose_name] = stored
        provenance = getattr(host, "teaching_capture_provenance", {})
        for pose_name in (
            "robot_start", "weld_wait", "weld_finish", "weld_start_wait",
            "weld_start", "weld_goal_wait", "weld_end",
        ):
            provenance.pop(pose_name, None)
        for endpoint, pose_name in (
            ("start_wait", "weld_start_wait"),
            ("start", "weld_start"),
            ("goal_wait", "weld_goal_wait"),
            ("goal", "weld_end"),
        ):
            names, positions = joint_states[endpoint]
            host.taught_robot_poses[pose_name] = (
                "right_manipulator",
                tuple(names),
                tuple(positions),
                copy.deepcopy(current[endpoint]),
            )
            if endpoint in ("start", "goal"):
                host.linear_tcp_endpoints[0 if endpoint == "start" else 1] = (
                    copy.deepcopy(current[endpoint])
                )
        host.seam_teaching_reference = {
            name: copy.deepcopy(host.taught_robot_poses[name])
            for name in ("weld_start", "weld_end")
        }
        try:
            save_seam_teaching_reference_yaml(
                host._seam_reference_yaml_path("right_manipulator"),
                "right_manipulator",
                host.seam_teaching_reference,
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            host.error(f"Selected pass seam reference save failed: {error}")
            return
        host.teaching_pose_changed()
        if resolve_corrected_ik:
            ik_targets = tuple(
                (
                    "goal" if endpoint.startswith("goal") else "start",
                    "right_manipulator",
                    copy.deepcopy(current[endpoint]),
                    tuple(joint_states[endpoint][0]),
                    pose_name,
                )
                for endpoint, pose_name in (
                    ("start_wait", "weld_start_wait"),
                    ("start", "weld_start"),
                    ("goal_wait", "weld_goal_wait"),
                    ("goal", "weld_end"),
                )
            )
            threading.Thread(
                target=host.node.resolve_tcp_joint_states,
                args=(ik_targets,),
                daemon=True,
            ).start()
            load_completion = "resolving corrected joint states"
        else:
            try:
                for endpoint, pose_name in (
                    ("start_wait", "weld_start_wait"),
                    ("start", "weld_start"),
                    ("goal_wait", "weld_goal_wait"),
                    ("goal", "weld_end"),
                ):
                    names, positions = joint_states[endpoint]
                    save_initial_state_yaml(
                        host._initial_state_yaml_path(
                            "right_manipulator", pose_name
                        ),
                        "right_manipulator",
                        names,
                        positions,
                        current[endpoint],
                    )
            except (OSError, ValueError, yaml.YAMLError) as error:
                host.error(f"Saved pass teaching restore failed: {error}")
                return
            load_completion = "saved joint/TCP pairs restored exactly"
        host._set_four_pass_status(
            f"Pass {number} corrected WAIT/START/GOAL WAIT/GOAL loaded into "
            f"Teaching Detail · {teaching_source} · {load_completion}"
        )
        host.pipeline_result(
            f"PASS {number} CUMULATIVE CORRECTION APPLIED · "
            f"four teaching poses loaded · {load_completion} · "
            "no welding started"
        )
