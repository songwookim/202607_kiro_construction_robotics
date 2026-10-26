"""Independent welding task settings and working state for 1G and 3G."""

import copy
from pathlib import Path
import tkinter as tk

import yaml

from construct_robot.io.teaching_yaml import atomic_yaml, teaching_config_dir
from construct_robot.io.weld_logging import read_last_execution_settings
from construct_robot.gui.weld_parameter_panel import SETTING_VARIABLES


def task_config_directory(position):
    if position not in ("1G", "3G"):
        raise ValueError(f"Unsupported welding task: {position}")
    directory = teaching_config_dir()
    return directory if position == "1G" else directory / "welding_3g"


def task_feedback_directory(position):
    if position not in ("1G", "3G"):
        raise ValueError(f"Unsupported welding task: {position}")
    directory = Path.home() / "ros2_ws" / "weld_feedback"
    return directory if position == "1G" else directory / "3g"


TASK_VARIABLES = (
    "teaching_pose_name", "velocity_percent", "speed_mode", "tcp_speed_mm_s",
    "interpolation_step_mm", "linear_motion_profile", "weave_base",
    "seam_axis", "wall_probe_axis", "floor_probe_axis", "seam_orientation_mode",
    "wall_probe_sign", "floor_probe_sign", "touch_probe_distance_mm",
    "touch_probe_speed_percent", "touch_settle_seconds", "corner_touch_target",
    "corner_touch_count", "four_pass_folder", "selected_pass_number",
    "auto_seam_move_to_end_pose", "work_cycle_repeats",
)

STATUS_VARIABLES = (
    "reference_yaw_status", "sensed_yaw_status", "delta_yaw_status",
    "reference_length_status", "quick_teaching_status", "four_pass_status",
)

WORKING_STATE = (
    "teaching_state", "_multipass_state", "sequence_model",
    "linear_tcp_endpoints", "initial_joint_state", "seam_teaching_reference",
    "corner_touches", "seam_probe_touches", "seam_probe_starts", "seam_probe_stops",
    "raw_two_touch_seam", "corrected_two_touch_seam", "corrected_seam_geometry",
    "computed_seam_endpoints", "computed_seam_wait_points",
    "pass_probe_touch_yaml_target", "points", "weave_source", "weave_base_paths",
    "path_kind",
)


class WeldingTaskController:
    def __init__(self, host):
        self.host = host
        self.states = {}
        self.variables = {
            name: variable for name, variable in vars(host).items()
            if isinstance(variable, tk.Variable) and (
                name in TASK_VARIABLES or name.startswith(("weld_", "weave_"))
            )
        }
        self.empty_state = copy.deepcopy(self.capture())
        self.load_error = None
        try:
            self.load_settings("1G")
        except (OSError, ValueError, tk.TclError, yaml.YAMLError) as error:
            for name, value in self.empty_state["values"].items():
                self.variables[name].set(value)
            self.load_error = str(error)

    def capture(self):
        host = self.host
        return {
            "working": {name: getattr(host, name) for name in WORKING_STATE
                        if hasattr(host, name)},
            "values": self.values(),
            "status": {name: getattr(host, name).get() for name in STATUS_VARIABLES
                       if hasattr(host, name)},
        }

    def values(self):
        host = self.host
        values = {name: variable.get() for name, variable in self.variables.items()}
        if getattr(host, "_active_teaching_task", "welding") != "welding":
            values["velocity_percent"] = host._teaching_task_settings["welding"][4]
        return values

    def load_settings(self, position):
        path = task_config_directory(position) / "task_settings.yaml"
        if path.is_file():
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            if (not isinstance(document, dict) or document.get("format_version") != 1
                    or document.get("welding_position") != position
                    or not isinstance(document.get("values"), dict)):
                raise ValueError(f"Invalid {position} task settings: {path}")
            for name, value in document["values"].items():
                if name in self.variables:
                    self.variables[name].set(value)
                    self.variables[name].get()
            return
        if position == "3G":
            latest = task_feedback_directory(position) / "latest_weld_feedback.log"
            saved = read_last_execution_settings(latest)
            settings = dict(saved.get("settings", {}))
            settings.update(saved.get("motion", {}))
            for key, (name, scale) in SETTING_VARIABLES.items():
                if key in settings and name in self.variables:
                    value = settings[key]
                    self.variables[name].set(value / scale if scale else value)

    def save_settings(self):
        position = self.host.welding_position
        path = task_config_directory(position) / "task_settings.yaml"
        atomic_yaml(path, {
            "format_version": 1,
            "welding_position": position,
            "values": self.values(),
        })
        return path

    def switch(self, position):
        host = self.host
        if position == host.welding_position:
            return False
        task_config_directory(position)
        sensing = getattr(host, "precision_sensing_test_panel", None)
        node = host.node
        if (host.sequence_running or node.active_motion_goal is not None
                or getattr(node, "_motion_action_calls", None)
                or host.active_weld_feedback_session is not None
                or host.keyboard_teaching_capture_in_progress or host.seam_auto_running
                or getattr(host, "multi_pass_registration", None) is not None
                or host.keyboard_velocity_switching or host.keyboard_wire_active_key is not None
                or host.hicomm_inching_direction is not None or host.robot_power_busy
                or (sensing is not None and (
                    sensing.busy or sensing.estimating or sensing.bridge.busy
                    or (sensing.direct is not None and sensing.direct.busy)))):
            raise ValueError("Stop the active operation before changing welding tasks")
        self.save_settings()
        self.states[host.welding_position] = self.capture()
        state = self.states.get(position)
        fresh = state is None
        if fresh:
            state = copy.deepcopy(self.empty_state)
            state["values"]["four_pass_folder"] = str(task_feedback_directory(position))
        previous = host.welding_position
        host.welding_position = position
        try:
            for name, value in state["working"].items():
                setattr(host, name, value)
            for name, value in state["values"].items():
                self.variables[name].set(value)
            for name, value in state["status"].items():
                getattr(host, name).set(value)
            if fresh:
                self.load_settings(position)
                host._auto_load_teaching_states()
        except (OSError, ValueError, tk.TclError, yaml.YAMLError):
            host.welding_position = previous
            original = self.states[previous]
            for name, value in original["working"].items():
                setattr(host, name, value)
            for name, value in original["values"].items():
                self.variables[name].set(value)
            for name, value in original["status"].items():
                getattr(host, name).set(value)
            raise
        host.initial_plan_ready = False
        host.invalidate_approved_plan()
        host.teaching_pose_changed()
        host.set_points(host.points)
        node.publish_points(host.points, host.show_path.get())
        node.publish_seam_comparison(
            host.raw_two_touch_seam, host.corrected_two_touch_seam, host.show_path.get()
        )
        host.refresh_sequence_table()
        host._update_seam_yaw_status()
        host.update_speed_label()
        host._update_motion_profile_label()
        host.weld_parameter_panel.latest_log_path = host._latest_weld_feedback_path()
        host.weld_parameter_panel.refresh()
        host.log(f"Selected {position} welding task · save folder {host._weld_feedback_directory()}")
        return True
