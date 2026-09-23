"""Thin same-thread port to the existing Tk production application.

The Tk application remains the sole ROS/MoveIt/Hi-COMM/Fastech owner.  This
module contains no robot, welder, touch, or controller execution algorithm.
"""

import threading
from types import SimpleNamespace
from pathlib import Path

from .task_teaching_model import TEACHING_POSES
from .weld_config import validate_digital_weld_settings


_RECIPE_VARIABLES = {
    "current_a": "weld_current_raw",
    "voltage_tenths": "weld_voltage_raw",
    "material": "weld_material",
    "diameter_mm": "weld_diameter_mm",
    "mode": "weld_mode",
    "gas": "weld_gas",
    "synergic": "weld_synergic",
    "correction": "weld_correction",
    "hot_start_enabled": "weld_hot_start_enabled",
    "hot_start_percent": "weld_hot_start_percent",
    "hot_start_hold_adjustment": "weld_hot_start_hold_adjustment",
    "custom_hot_start_enabled": "weld_custom_hot_start_enabled",
    "custom_hot_start_hold_s": "weld_custom_hot_start_hold_s",
    "custom_hot_start_percent": "weld_custom_hot_start_percent",
    "expect_native_crater": "weld_expect_native_crater",
    "crater_panel_current_ref_a": "weld_crater_panel_current_ref_a",
    "crater_panel_voltage_ref_v": "weld_crater_panel_voltage_ref_v",
    "crater_panel_time_ref_s": "weld_crater_panel_time_ref_s",
    "software_crater_enabled": "weld_software_crater_enabled",
    "software_crater_ratio_percent": "weld_software_crater_ratio_percent",
    "software_crater_voltage_v": "weld_software_crater_voltage_v",
    "software_crater_hold_s": "weld_software_crater_hold_s",
    "wire_consumable_alpha_mm": "weld_wire_consumable_alpha_mm",
}

_MOTION_VARIABLES = {
    "weld_tcp_speed_mm_s": "weld_tcp_speed_mm_s",
    "weld_weave_enabled": "weld_weave_enabled",
    "weld_weave_pattern": "weave_pattern",
    "weld_weave_amplitude_mm": "weave_amplitude_mm",
    "weld_weave_pitch_mm": "weave_pitch_mm",
    "weld_weave_left_dwell_s": "weave_left_dwell_s",
    "weld_weave_right_dwell_s": "weave_right_dwell_s",
    "weld_weave_axis": "weave_axis",
}


class ProductionRuntimePort:
    """Delegates Qt actions to one already initialized WeldActionGui instance."""

    def __init__(self, gui):
        self.gui = gui
        self.sequence_state = gui.sequence_model
        self.teaching_state = gui.teaching_state
        self.multipass_state = gui.multipass_state
        self._owner_thread = threading.get_ident()

    def _same_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Production Tk runtime must be called on its GUI thread")

    def subscribe_status(self, callback):
        """Sequence events may originate on workers; Qt queues the callback."""
        return self.sequence_state.subscribe(lambda _event: callback())

    def _sync_builder(self):
        """Discard stale Tk row edits before executing the shared Qt rows."""
        self._same_thread()
        table = self.gui.sequence_table
        selected = table.selection()
        if selected:
            table.selection_remove(*selected)
        self.gui.refresh_sequence_table()

    def _raise_confirmation_host(self):
        root = getattr(self.gui, "root", None)
        if root is not None:
            root.lift()

    def plan_sequence(self):
        self._sync_builder()
        return self.gui.run_sequence(True, False)

    def execute_sequence(self):
        self._sync_builder()
        self._raise_confirmation_host()
        # Existing run_sequence owns all connection, ARC unlock and operator
        # confirmation checks.  Do not call its worker directly.
        return self.gui.run_sequence(True, True)

    def stop_sequence(self):
        self._same_thread()
        return self.gui.stop_sequence()

    def capture_teaching_pose(self, name, planning_group):
        self._same_thread()
        if name not in TEACHING_POSES:
            raise ValueError("Unknown named teaching pose")
        if planning_group not in ("left_manipulator", "right_manipulator"):
            raise ValueError("Select a supported robot arm")
        if self.gui.planning_group.get() != planning_group:
            raise ValueError("Select this arm in the production runtime first")
        if self.sequence_state.running or self.gui.node.active_motion_goal is not None:
            raise ValueError("Finish active motion before teaching capture")
        self.gui.teaching_pose_name.set(TEACHING_POSES[name])
        self.gui.teaching_pose_changed()
        return self.gui.capture_initial_state()

    def load_teaching_pose(self, name, planning_group, path):
        self._same_thread()
        if name not in TEACHING_POSES:
            raise ValueError("Unknown named teaching pose")
        if self.sequence_state.running or self.gui.node.active_motion_goal is not None:
            raise ValueError("Finish active motion before loading teaching")
        if self.gui.planning_group.get() != planning_group:
            raise ValueError("Select this arm in the production runtime first")
        # The existing method validates YAML, invalidates stale seam state,
        # updates reference/initial state and schedules FK verification.
        return self.gui.load_initial_state_from_path(path, name)

    def begin_multi_pass_registration(self, number):
        self._same_thread()
        self.multipass_state.select(number)
        self.gui.selected_pass_number.set(int(number))
        self._raise_confirmation_host()
        # Existing method includes source hash, clearance, connection, motion
        # and confirmation checks; its worker handles registration phases.
        return self.gui.run_four_pass_correction()

    def multi_pass_folder(self):
        self._same_thread()
        return self.gui.four_pass_folder.get()

    def load_multi_pass_references(self, folder):
        self._same_thread()
        if self.sequence_state.running or self.gui.multi_pass_registration is not None:
            raise ValueError("Finish the active sequence/registration first")
        chosen = Path(folder).expanduser().resolve()
        if not chosen.is_dir():
            raise ValueError(f"4-pass work folder does not exist: {chosen}")
        self.gui.four_pass_folder.set(str(chosen))
        return self.gui.load_four_pass_references()

    def load_selected_pass(self, number):
        self._same_thread()
        self.multipass_state.select(number)
        self.gui.selected_pass_number.set(int(number))
        return self.gui.apply_selected_pass_correction()

    def capture_multi_pass_start(self):
        return self._capture_multi_pass_key("i", "waiting_start_capture")

    def capture_multi_pass_goal(self):
        return self._capture_multi_pass_key("j", "waiting_goal_capture")

    def _capture_multi_pass_key(self, key, expected_phase):
        self._same_thread()
        session = self.gui.multi_pass_registration
        if session is None or session.get("phase") != expected_phase:
            raise ValueError(f"Wait for {expected_phase} before {key.upper()} capture")
        self._raise_confirmation_host()
        root = getattr(self.gui, "root", None)
        if root is not None:
            root.focus_force()
        event = SimpleNamespace(keysym=key)
        # This is the *same* keyboard path, including focus, velocity-mode,
        # motion-active and duplicate-capture guards.  Never call its worker.
        result = self.gui.keyboard_teaching_shortcut_key(event)
        if result == "break":
            self.gui.keyboard_teaching_shortcut_release(event)
        return result

    def stop_multi_pass_registration(self):
        self._same_thread()
        return self.gui.stop_multi_pass_correction()

    def build_weld_scenario(self, state=None):
        self._same_thread()
        self._sync_builder()
        if state is not None:
            self.apply_weld_configuration(state)
        return self.gui.build_sensed_weld_sequence()

    def plan_cleaner(self):
        self._sync_builder()
        steps = [step for step in self.sequence_state.steps
                 if step.get("torch_clean_scenario")]
        if not steps:
            raise ValueError("Build Torch Cleaner in the shared Sequence Builder first")
        return self.gui.run_sequence(True, False, steps_override=steps)

    def execute_cleaner(self):
        self._sync_builder()
        steps = [step for step in self.sequence_state.steps
                 if step.get("torch_clean_scenario")]
        if not steps:
            raise ValueError("Build Torch Cleaner in the shared Sequence Builder first")
        return self.gui.run_sequence(True, True, steps_override=steps)

    def weld_configuration(self):
        """Read current Tk builder defaults, not any active welder setpoints."""
        self._same_thread()
        recipe = self.gui._digital_weld_settings()
        motion = {key: getattr(self.gui, attr).get()
                  for key, attr in _MOTION_VARIABLES.items()}
        return recipe, motion

    def apply_weld_configuration(self, state):
        """Set future Tk builder defaults only; no Hi-COMM call or row rewrite."""
        self._same_thread()
        if self.sequence_state.running:
            raise ValueError("Finish the active sequence before changing builder defaults")
        settings = validate_digital_weld_settings(state.recipe)
        if set(_MOTION_VARIABLES) - set(state.motion):
            raise ValueError("Weld motion configuration is incomplete")
        for key, attr in _RECIPE_VARIABLES.items():
            getattr(self.gui, attr).set(settings[key])
        for key, attr in _MOTION_VARIABLES.items():
            getattr(self.gui, attr).set(state.motion[key])
        return "Builder defaults updated; rebuild the weld scenario to change stored rows"

    def robot_status(self):
        self._same_thread()
        return dict(self.gui.robot_connected)

    def selected_planning_group(self):
        self._same_thread()
        return self.gui.planning_group.get()

    def runtime_status(self):
        self._same_thread()
        status = None
        client = self.gui.hicomm_client
        if client is not None and self.gui.hicomm_connected:
            status = client.latest_status()
        return {
            "active_motion": self.gui.node.active_motion_goal is not None,
            "welding": (
                status.get("sequence_stage") or status.get("output_state_name")
                if status else "Disconnected"
            ),
            "error": getattr(self.gui, "latest_pipeline_error", None),
        }

    def io_status(self):
        self._same_thread()
        state = self.gui.fastech_previous_state
        connected = bool(self.gui.fastech_connected)
        touch = (bool(state.digital_in[4]) if connected and state is not None
                 and len(state.digital_in) > 4 else None)
        enabled = (bool(state.digital_out[0]) if connected and state is not None
                   and state.digital_out else None)
        control = self.gui.previous_control_box_io
        client = self.gui.hicomm_client
        welding = client.latest_status() if client is not None and self.gui.hicomm_connected else None
        return {
            "fastech_connected": connected,
            "touch_input": touch,
            "touch_enabled": enabled,
            "hicomm_connected": bool(self.gui.hicomm_connected),
            "welder_output_state": (
                welding.get("output_state_name", welding.get("output_state"))
                if welding else None
            ),
            "control_box_io": (
                f"DI={tuple(control[0])}, DO={tuple(control[1])}"
                if control is not None else None
            ),
        }
