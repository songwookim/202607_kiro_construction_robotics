import copy
import math
from pathlib import Path
import queue
import signal
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import rclpy
import yaml
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import TransformException

from construct_robot.core.cartesian_path_common import (
    PLANNING_GROUP_TIPS,
    WELD_WEAVE_SAMPLES_PER_CYCLE,
    _quaternion_rotate_vector,
    linear_pose_waypoints,
    midpoint_pose,
    named_tcp_linear_waypoints,
    pose_is_valid,
    pose_with_local_rpy_offset,
    pose_with_rpy_offset,
    position_only_goal_constraints,
    quaternion_angular_distance,
    tcp_pose_goal_constraints,
    tcp_position_is_valid,
    trajectory_duration_seconds,
    tip_link_for_group,
    transform_xyz,
    validated_seam_speed_factor,
    weave_cycles_for_pitch,
    weave_path_speed_m_s,
    weld_weave_geometry,
)
from construct_robot.io.hicomm_welder import (
    BIT_FORWARD,
    BIT_GAS,
    BIT_REVERSE,
    BIT_STICK,
    DIAMETER_CODES,
    GAS_CODES,
    HiCommWelderClient,
    MATERIAL_CODES,
    MODE_CODES,
    TxState,
    build_request,
)
from construct_robot.core.weld_quality_metrics import format_quality_summary
from construct_robot.core.sequence_model import (
    SequenceModel,
    WELD_SCENARIO_STAGE_ORDER,
    next_sequential_slot,
    taught_wait_approach_steps,
    update_weld_scenario_motion_values,
    validate_managed_weld_sequence,
)
from construct_robot.core.work_cycle import assemble_work_cycle, load_work_cycle
from construct_robot.core.seam_geometry import (
    CORNER_TOUCH_NAMES,
    CorrectedSeamGeometry,
    _axis_unit_vector,
    _pose_position_tuple,
    _unit_vector,
    _vector_dot,
    compute_corrected_seam_endpoints,
    compute_corrected_seam_geometry,
    compute_plane_intersection_line,
    compute_real_seam_direction,
    compute_safe_weld_approach,
    compute_seam_local_frame,
    compute_surface_plane,
    project_point_to_line,
    seam_lead_poses,
    aligned_wait_pose,
    apply_sensed_seam_orientation,
    corner_endpoint_from_two_touches,
    corner_seam_from_touches,
    corrected_corner_seam_from_four_touches,
    fixed_tilt_wait_reference_poses,
    generalized_corner_endpoint_from_two_touches,
    intersect_three_planes,
    seam_xy_normal,
    seam_yaw,
    translated_wait_pose,
    two_touch_corner_seam,
    wide_sensing_path_poses,
    yaw_corrected_seam_poses,
)
from construct_robot.core.keyboard_jog import (
    KEYBOARD_JOG_SELECTIONS,
    keyboard_jog_velocity,
    keyboard_velocity_vector,
    next_keyboard_speed,
)
from construct_robot.core.multipass import (
    MultiPassState,
    correct_four_pass_references,
    correct_remaining_passes,
    correct_seam_from_measured_start,
)
from construct_robot.io.weld_logging import (
    calculate_weld_production_metrics,
    format_weld_feedback_log,
    read_last_execution_settings,
    read_teaching_and_touch_snapshot,
    read_weld_pass_reference,
    save_weld_feedback_log,
    weld_weave_settings_text,
)
from construct_robot.core.task_teaching_model import (
    JOINT_RECALL_TEACHING_POSES,
    SEAM_REFERENCE_TEACHING_POSES,
    TCP_POSE_TEACHING_POSES,
    TEACHING_POSES,
    TOUCH_GUARDED_TEACHING_POSES,
    TeachingState,
)
from construct_robot.io.teaching_yaml import (
    ARM_JOINT_NAMES, _pose_from_yaml_dict,
    load_initial_state_yaml,
    load_seam_teaching_reference_yaml,
    parse_teaching_snapshot_entry,
    read_pass_teaching_reference,
    save_initial_state_yaml,
    save_seam_teaching_reference_yaml,
    save_seam_touch_yaml,
)
from construct_robot.core.weld_config import (
    DEFAULT_DIGITAL_WELD_SETTINGS,
    DIGITAL_WELD_RECIPE_KEYS, digital_weld_recipe,
    validate_digital_weld_settings, weld_current_profile,
)
from construct_robot.application.weld_sequence_builder import (
    WeldScenarioInput,
    WeldStepMotionInput,
    allocate_weld_scenario_slots,
    build_weld_scenario_steps,
    new_weld_scenario_id,
    plan_weld_approach,
    plan_weld_path,
    resolve_weld_endpoints,
    validate_required_weld_poses,
)
from construct_robot.application.multipass_controller import MultipassController
from construct_robot.application.seam_correction_controller import (
    SeamCorrectionController,
)
from construct_robot.application.weld_execution_controller import (
    WeldExecutionController,
    WeldFeedbackRecorder,
    weld_status_snapshot,
)
from construct_robot.application.sequence_executor import (
    SequenceExecutor,
    attach_execution_conditions,
    contains_weld_command,
    is_work_cycle,
    pose_execution_conditions,
    record_step_conditions,
)
# Helpers formerly defined in this module now live in core/, io/ and nodes/.
# Unused-looking imports above and below are deliberate re-exports so existing
# ``weld_action_gui.<name>`` imports keep working.
from construct_robot.nodes.weld_runtime_node import (
    CONTROLLED_JOINT_NAMES,
    CONTROLLER_NAMES,
    FASTECH_TOUCH_INPUT_PORT,
    FASTECH_TOUCH_OUTPUT_PORT,
    HEAD_JOINT_NAME_ORDER,
    HEAD_JOINT_NAMES,
    KEYBOARD_TF_LOOKUP_TIMEOUT_S,
    KEYBOARD_VELOCITY_CONTROLLER_NAMES,
    KEYBOARD_VELOCITY_DEADMAN_TIMEOUT_S,
    KEYBOARD_VELOCITY_INITIAL_DEADMAN_TIMEOUT_S,
    KEYBOARD_ZERO_BURST_COUNT,
    LEGACY_RAINBOW_TOUCH_INPUT_PORT,
    LEGACY_RAINBOW_TOUCH_OUTPUT_PORT,
    WeldGuiNode,
)

MANUAL_IO_CANDIDATES = frozenset((0, 4, 8, 9, 10, 12, 13))
FASTECH_GUI_CHANNELS = {
    0: "Touch sensing",
    5: "Torch cleaner 3",
    6: "Torch cleaner 2",
    7: "Torch cleaner 1",
}
# Retired test channels are hidden from manual control, but the all-off
# safety operation still clears them if an earlier session left them ON.
FASTECH_ALL_OFF_CHANNELS = (0, 3, 4, 5, 6, 7)
FASTECH_TOUCH_BACKEND = "fastech_ethernet"

KEYBOARD_LINEAR_SPEEDS_MM_S = (5.0, 15.0, 45.0)
KEYBOARD_ANGULAR_SPEEDS_DEG_S = (3.0, 7.0, 10.0)
TCP_FEEDBACK_SAMPLE_PERIOD_S = 0.01  # 50 Hz logging poll; unique TF rate is measured separately.
KEYBOARD_TEACHING_POSE_SHORTCUTS = {
    "o": "weld_start_wait",
    "k": "weld_goal_wait",
    "p": "weld_wait",
    "l": "weld_finish",
    "m": "robot_start",
}

WAIT_FIXED_TILT_ORIENTATION_MODE = "Wait poses + fixed World-XYZ tilt"
LEGACY_WAIT_FIXED_TILT_ORIENTATION_MODE = "Wait poses + fixed Tool-XYZ tilt"


def _sequence_executor_for(host):
    """Executor bound to a GUI (or a lightweight test double of one)."""
    return SequenceExecutor(host, touch_io_backend=FASTECH_TOUCH_BACKEND)


def _weld_controller_for(host):
    """ARC runtime operations bound to a GUI (or a test double of one)."""
    return WeldExecutionController(
        host, fake_arc=lambda: WeldActionGui._execution_fake_arc(host)
    )


def _multipass_for(host):
    """Four-pass registration workflow bound to a GUI (or a test double)."""
    return MultipassController(host)


def _seam_correction_for(host):
    """Seam-correction / touch-probe workflow bound to a GUI (or a test double)."""
    return SeamCorrectionController(
        host,
        touch_output_port=FASTECH_TOUCH_OUTPUT_PORT,
        ui_error_types=(tk.TclError,),
    )


def _weld_feedback_for(host):
    """Weld feedback session operations bound to a GUI (or a test double)."""
    return WeldFeedbackRecorder(
        host, tcp_sample_period_s=TCP_FEEDBACK_SAMPLE_PERIOD_S
    )


class WeldActionGui:
    """Tk GUI for acquiring, editing, visualizing, and running weld paths."""

    POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")

    @property
    def multipass_state(self):
        if not hasattr(self, "_multipass_state"):
            self._multipass_state = MultiPassState()
        return self._multipass_state

    @property
    def four_pass_references(self):
        return self.multipass_state.references

    @four_pass_references.setter
    def four_pass_references(self, value):
        self.multipass_state.references = value

    @property
    def four_pass_loaded_folder(self):
        return self.multipass_state.loaded_folder

    @four_pass_loaded_folder.setter
    def four_pass_loaded_folder(self, value):
        self.multipass_state.loaded_folder = value

    @property
    def four_pass_corrected(self):
        return self.multipass_state.corrected

    @four_pass_corrected.setter
    def four_pass_corrected(self, value):
        self.multipass_state.corrected = value

    @property
    def four_pass_output_folder(self):
        return self.multipass_state.output_folder

    @four_pass_output_folder.setter
    def four_pass_output_folder(self, value):
        self.multipass_state.output_folder = value

    @property
    def four_pass_history(self):
        return self.multipass_state.history

    @four_pass_history.setter
    def four_pass_history(self, value):
        self.multipass_state.history = value

    @property
    def multi_pass_registration(self):
        return self.multipass_state.registration

    @multi_pass_registration.setter
    def multi_pass_registration(self, value):
        self.multipass_state.registration = value

    @property
    def taught_robot_poses(self):
        if not hasattr(self, "teaching_state"):
            self.teaching_state = TeachingState(TEACHING_POSES)
        return self.teaching_state.poses

    @taught_robot_poses.setter
    def taught_robot_poses(self, poses):
        if not hasattr(self, "teaching_state"):
            self.teaching_state = TeachingState(TEACHING_POSES)
        self.teaching_state.poses = poses

    @property
    def teaching_capture_provenance(self):
        if not hasattr(self, "teaching_state"):
            self.teaching_state = TeachingState(TEACHING_POSES)
        return self.teaching_state.provenance

    @teaching_capture_provenance.setter
    def teaching_capture_provenance(self, provenance):
        if not hasattr(self, "teaching_state"):
            self.teaching_state = TeachingState(TEACHING_POSES)
        self.teaching_state.provenance = provenance

    @property
    def sequence_steps(self):
        # Laziness keeps lightweight GUI test doubles compatible with __new__.
        if not hasattr(self, "sequence_model"):
            self.sequence_model = SequenceModel()
        return self.sequence_model.steps

    @sequence_steps.setter
    def sequence_steps(self, steps):
        if not hasattr(self, "sequence_model"):
            self.sequence_model = SequenceModel()
        self.sequence_model.replace(steps)

    @property
    def sequence_running(self):
        if not hasattr(self, "sequence_model"):
            self.sequence_model = SequenceModel()
        return self.sequence_model.running

    @sequence_running.setter
    def sequence_running(self, running):
        if not hasattr(self, "sequence_model"):
            self.sequence_model = SequenceModel()
        self.sequence_model.running = bool(running)

    def _create_toggle_section(
        self,
        parent,
        key,
        title,
        expanded=False,
    ):
        container = ttk.Frame(parent)
        # Keep backing widgets alive for shared callbacks, but retire these
        # panels from the operator UI.
        visible = key not in {"path_test", "planned_path", "digital_io"}
        if visible:
            container.pack(fill=tk.X, pady=2)
        section_styles = (
            "SectionBlue.TButton",
            "SectionGreen.TButton",
            "SectionAmber.TButton",
            "SectionViolet.TButton",
        )
        button = ttk.Button(
            container,
            command=lambda selected=key: self.toggle_motion_section(selected),
            style=section_styles[len(self.motion_sections) % len(section_styles)],
        )
        button.pack(fill=tk.X)
        body = ttk.Frame(container, padding=(8, 5))
        self.motion_sections[key] = {
            "body": body,
            "button": button,
            "title": title,
            "number": 1 + sum(
                section.get("visible", True)
                for section in self.motion_sections.values()
            ),
            "visible": visible,
            "expanded": bool(expanded),
        }
        if expanded:
            body.pack(fill=tk.X)
        self._refresh_motion_section_button(key)
        return body

    def _refresh_motion_section_button(self, key):
        section = self.motion_sections[key]
        marker = "▼" if section["expanded"] else "▶"
        section["button"].configure(
            text=f"{marker}  {section['number']}. {section['title']}",
        )

    def toggle_motion_section(self, key):
        section = self.motion_sections[key]
        section["expanded"] = not section["expanded"]
        if section["expanded"]:
            section["body"].pack(fill=tk.X)
        else:
            section["body"].pack_forget()
        self._refresh_motion_section_button(key)
        self.root.after_idle(self._update_scroll_region)

    @staticmethod
    def _add_labeled_value(parent, pair_index, label, variable, width=8):
        column = pair_index * 2
        ttk.Label(parent, text=label).grid(
            row=0, column=column, padx=(6, 2), pady=3, sticky=tk.E
        )
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=0, column=column + 1, padx=(2, 6), pady=3
        )

    def _create_operator_scroll_page(self, notebook, title):
        page = ttk.Frame(notebook)
        notebook.add(page, text=title)
        canvas = tk.Canvas(page, highlightthickness=0)
        scrollbar = ttk.Scrollbar(page, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        content = ttk.Frame(canvas, padding=16)
        window = canvas.create_window((0, 0), window=content, anchor=tk.NW)
        content.bind(
            "<Configure>",
            lambda _event, active=canvas: active.configure(
                scrollregion=active.bbox("all")
            ),
        )
        canvas.bind(
            "<Configure>",
            lambda event, active=canvas, item=window: active.itemconfigure(
                item, width=event.width
            ),
        )
        self._operator_page_canvases[str(page)] = (canvas, window)
        return content

    def _operator_page_changed(self, _event=None):
        if self.keyboard_velocity_active_key is not None:
            self._stop_keyboard_jog_command("STOPPED · task changed")
        selected = self.operator_notebook.select()
        if selected == str(self.task_notebook.master):
            selected = self.task_notebook.select()
            task = "cleaner" if selected == self.task_notebook.tabs()[1] else "welding"
            previous = getattr(self, "_active_teaching_task", task)
            if task != previous:
                settings = self._teaching_task_settings
                settings[previous] = (
                    self.keyboard_jog_selection.get(), self.keyboard_jog_frame.get(),
                    self.keyboard_jog_linear_speed.get(), self.keyboard_jog_angular_speed.get(),
                    self.velocity_percent.get(),
                )
                selection, frame, linear, angular, scale = settings[task]
                self.keyboard_jog_selection.set(selection)
                self.keyboard_jog_frame.set(frame)
                self.keyboard_jog_linear_speed.set(linear)
                self.keyboard_jog_angular_speed.set(angular)
                self.velocity_percent.set(scale)
                self.update_speed_label()
                self._active_teaching_task = task
            if task == "cleaner" and self.keyboard_velocity_arm == "left":
                self._disable_keyboard_velocity_async()
        current = self._operator_page_canvases.get(selected)
        if current is not None:
            self.content_canvas, self.content_window = current
            self.root.after_idle(self._update_scroll_region)

    def _build_connection_header(self, connection_page):
        # Connection state is deliberately first: no motion or welding control
        # should be interpreted before the operator checks these indicators.
        robot_status = ttk.LabelFrame(connection_page, text="Connection")
        robot_status.pack(fill=tk.X, pady=(5, 8))
        self.robot_connection_labels = {}
        for arm in ("left", "right"):
            label = tk.Label(
                robot_status,
                text=f"Connect {arm.upper()} (IP): X",
                width=32,
                relief=tk.SOLID,
                borderwidth=1,
                bg="#fce8e6",
                fg="#b3261e",
                font=("Sans", 11, "bold"),
            )
            label.pack(side=tk.LEFT, padx=6, pady=6)
            self.robot_connection_labels[arm] = label
        head_label = tk.Label(
            robot_status,
            text="Connect HEAD (CAN2): X",
            width=26,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#fce8e6",
            fg="#b3261e",
            font=("Sans", 11, "bold"),
        )
        head_label.pack(side=tk.LEFT, padx=6, pady=6)
        self.robot_connection_labels["head"] = head_label
        self.welder_connection_label = tk.Label(
            robot_status,
            text="HICOMM WELDER: X",
            width=22,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#fce8e6",
            fg="#b3261e",
            font=("Sans", 11, "bold"),
        )
        self.welder_connection_label.pack(side=tk.LEFT, padx=6, pady=6)
        tk.Button(
            robot_status,
            text="EMERGENCY STOP (SOFTWARE)\nALL MOTION + WELDER",
            command=self.emergency_stop_all,
            bg="#b3261e",
            fg="white",
            activebackground="#7f1d1d",
            activeforeground="white",
            font=("Sans", 11, "bold"),
            relief=tk.RAISED,
            borderwidth=3,
            padx=12,
            pady=3,
        ).pack(side=tk.RIGHT, padx=8, pady=4)

        robot_power = ttk.Frame(connection_page)
        robot_power.pack(fill=tk.X, pady=(0, 7))
        self.robot_activate_both_button = ttk.Button(
            robot_power,
            text="ACTIVATE BOTH · Real mode",
            command=lambda: self.request_both_robot_power(True),
        )
        self.robot_activate_both_button.pack(side=tk.LEFT, padx=(0, 6))
        self.robot_shutdown_both_button = ttk.Button(
            robot_power,
            text="SHUTDOWN BOTH",
            command=lambda: self.request_both_robot_power(False),
        )
        self.robot_shutdown_both_button.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            robot_power,
            textvariable=self.robot_power_status,
        ).pack(side=tk.LEFT)

    def _build_keyboard_teaching_controls(self, task_page, *, cleaner=False):
        if not cleaner:
            arm_selection = ttk.Frame(task_page)
            arm_selection.pack(fill=tk.X, pady=(0, 7))
            ttk.Label(
                arm_selection,
                text="Cartesian arm:",
                style="Step.TLabel",
            ).pack(side=tk.LEFT, padx=(0, 6))
            ttk.Combobox(
                arm_selection,
                textvariable=self.planning_group,
                values=("right_manipulator", "left_manipulator"),
                state="readonly",
                width=22,
            ).pack(side=tk.LEFT)
            self.planning_group.trace_add("write", self.arm_changed)
        else:
            ttk.Label(task_page, text="Torch Cleaner · right arm", style="Step.TLabel").pack(anchor=tk.W)

        keyboard_jog = self._create_toggle_section(
            task_page,
            "keyboard_jog_cleaner" if cleaner else "keyboard_jog",
            "Keyboard Teaching · SpaceMouse-style hold-to-run velocity",
            expanded=False,
        )
        jog_row = ttk.Frame(keyboard_jog)
        jog_row.pack(fill=tk.X, pady=2)
        enable_button = ttk.Checkbutton(
            jog_row,
            text="Enable keyboard teaching",
            variable=self.keyboard_jog_enabled,
            command=(self._cleaner_keyboard_jog_enable_changed if cleaner
                     else self.keyboard_jog_enable_changed),
        )
        enable_button.pack(side=tk.LEFT, padx=(0, 8))
        self.keyboard_jog_enable_buttons.append(enable_button)
        if not cleaner:
            self.keyboard_jog_enable_button = enable_button
        ttk.Label(jog_row, text="axis/plane").pack(side=tk.LEFT)
        ttk.Combobox(
            jog_row,
            textvariable=self.keyboard_jog_selection,
            values=tuple(KEYBOARD_JOG_SELECTIONS),
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(jog_row, text="XYZ frame").pack(side=tk.LEFT)
        ttk.Combobox(
            jog_row,
            textvariable=self.keyboard_jog_frame,
            values=("World", "Tool"),
            state="readonly",
            width=6,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(jog_row, text="XYZ mm/s").pack(side=tk.LEFT)
        ttk.Spinbox(
            jog_row,
            from_=0.1,
            to=25.0,
            increment=0.5,
            textvariable=self.keyboard_jog_linear_speed,
            width=6,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(jog_row, text="RPY deg/s").pack(side=tk.LEFT)
        ttk.Spinbox(
            jog_row,
            from_=0.1,
            to=10.0,
            increment=0.5,
            textvariable=self.keyboard_jog_angular_speed,
            width=6,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(
            jog_row,
            textvariable=self.keyboard_jog_status,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(
            keyboard_jog,
            text=(
                "Keys: 1=X  2=Y  3=Z  4=RX  5=RY  6=RZ  ·  "
                "7=XY  8=XZ  9=YZ  A=RX/RY  S=RX/RZ  D=RY/RZ"
            ),
        ).pack(anchor=tk.W)
        ttk.Label(
            keyboard_jog,
            text=(
                "Speed: V=XYZ 5/15/25 mm/s · X=rotation 3/7/10 deg/s · "
                +
                ("Save: selected cleaner pose via Teaching Detail below"
                 if cleaner else
                 "Save: I/J=TCP1/2 · O/K=START/GOAL WAIT · "
                 "P/L=Weld WAIT(init)/END · M=Initial pose")
            ),
            foreground="#174ea6",
        ).pack(anchor=tk.W)
        ttk.Label(
            keyboard_jog,
            text=(
                "Arrows: single axis Left/Down=-, Right/Up=+; plane "
                "Left/Right=first axis, Down/Up=second axis. "
                "Hold=move, release=zero velocity. "
                "Rotation RX/RY/RZ: always Global (World) axes."
            ),
            foreground="#137333",
        ).pack(anchor=tk.W)
        if not cleaner:
            ttk.Label(
                keyboard_jog,
                text="Wire (right arm, Hi-COMM connected): hold F=feed forward, R=reverse; release=OFF",
                foreground="#b3261e",
            ).pack(anchor=tk.W)
        if cleaner:
            return
        for key_name in (
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "s", "d"
        ):
            self.root.bind(
                f"<KeyPress-{key_name}>",
                self.keyboard_jog_selection_key,
                add="+",
            )
        for key_name in ("v", "x", "i", "j", "o", "k", "p", "l", "m"):
            self.root.bind(
                f"<KeyPress-{key_name}>",
                self.keyboard_teaching_shortcut_key,
                add="+",
            )
            self.root.bind(
                f"<KeyRelease-{key_name}>",
                self.keyboard_teaching_shortcut_release,
                add="+",
            )
        for key_name in ("Left", "Right", "Up", "Down"):
            self.root.bind(
                f"<KeyPress-{key_name}>",
                self.keyboard_jog_key_press,
                add="+",
            )
            self.root.bind(
                f"<KeyRelease-{key_name}>",
                self.keyboard_jog_key_release,
                add="+",
            )
        for key_name in ("f", "F", "r", "R"):
            self.root.bind(
                f"<KeyPress-{key_name}>", self.keyboard_wire_key_press, add="+"
            )
            self.root.bind(
                f"<KeyRelease-{key_name}>", self.keyboard_wire_key_release, add="+"
            )
        self.root.bind("<FocusOut>", self.keyboard_jog_focus_out, add="+")

    def _build_welder_controls(self, connection_page, welding_page):
        welder = self._create_toggle_section(
            connection_page, "welder", "Digital Welder · Hi-COMM TCP", expanded=False
        )
        ttk.Label(
            welder,
            text=(
                "Welder controls: ON by default · Robot motion: ROS 2/RBPodo "
                "· welding: direct Hi-COMM TX55/40 ms/RX71"
            ),
            foreground="#b06000",
        ).pack(anchor=tk.W, padx=4, pady=2)

        network = ttk.LabelFrame(welder, text="Hi-COMM network")
        network.pack(fill=tk.X, pady=2)
        for label, variable, width in (
            ("PC source IP", self.hicomm_source_ip, 15),
            ("Hi-COMM IP", self.hicomm_welder_ip, 15),
            ("port", self.hicomm_port, 7),
        ):
            ttk.Label(network, text=label).pack(side=tk.LEFT, padx=(6, 2))
            ttk.Entry(network, textvariable=variable, width=width).pack(
                side=tk.LEFT, padx=(0, 6)
            )
        self.hicomm_connect_button = ttk.Button(
            network, text="Connect", command=self.connect_hicomm
        )
        self.hicomm_connect_button.pack(side=tk.LEFT, padx=3)
        self.hicomm_disconnect_button = ttk.Button(
            network,
            text="Disconnect",
            command=self.disconnect_hicomm,
            state=tk.DISABLED,
        )
        self.hicomm_disconnect_button.pack(side=tk.LEFT, padx=3)

        feedback_tools = ttk.LabelFrame(
            welder, text="Weld feedback log / graph"
        )
        feedback_tools.pack(fill=tk.X, pady=2)
        ttk.Button(
            feedback_tools,
            text="Load teaching/touch from log...",
            command=self.load_teaching_and_touch_from_log,
        ).pack(side=tk.LEFT, padx=5, pady=3)
        ttk.Label(
            feedback_tools,
            text="saves feedback PNG + complete trajectory_3d PNG beside the log",
        ).pack(side=tk.LEFT, padx=8)

        weld_parameters = self._create_toggle_section(
            welding_page, "weld_parameters",
            "Weld Parameters / Welder Test · Recipe / Hot Start / Crater",
            expanded=True,
        )
        welder_test = ttk.Frame(weld_parameters)
        welder_test.pack(fill=tk.X)
        ttk.Label(
            welder_test,
            text=(
                "Available when Welder controls and Hi-COMM are connected · "
                "disconnect sends ALL OUTPUT OFF"
            ),
            foreground="#b3261e",
        ).pack(anchor=tk.W, padx=4, pady=2)

        wire_test = ttk.LabelFrame(
            welder_test, text="Wire inching / gas test"
        )
        wire_test.pack(fill=tk.X, pady=2)
        self.hicomm_forward_button = ttk.Button(
            wire_test, text="Hold: forward inch", state=tk.DISABLED
        )
        self.hicomm_forward_button.pack(side=tk.LEFT, padx=4, pady=3)
        self.hicomm_reverse_button = ttk.Button(
            wire_test, text="Hold: reverse inch", state=tk.DISABLED
        )
        self.hicomm_reverse_button.pack(side=tk.LEFT, padx=4, pady=3)
        for button, direction in (
            (self.hicomm_forward_button, "forward"),
            (self.hicomm_reverse_button, "reverse"),
        ):
            button.bind(
                "<ButtonPress-1>",
                lambda _event, selected=direction: self.request_hicomm_inching(
                    selected, True
                ),
            )
            button.bind(
                "<ButtonRelease-1>",
                lambda _event, selected=direction: self.request_hicomm_inching(
                    selected, False
                ),
            )
            button.bind(
                "<Leave>",
                lambda _event, selected=direction: self.request_hicomm_inching(
                    selected, False
                ),
            )
        self.hicomm_gas_check = ttk.Checkbutton(
            wire_test,
            text="Gas test",
            variable=self.hicomm_gas_enabled,
            command=self.request_hicomm_gas,
            state=tk.DISABLED,
        )
        self.hicomm_gas_check.pack(side=tk.LEFT, padx=8)
        self.hicomm_test_status = ttk.Label(wire_test, text="test locked")
        self.hicomm_test_status.pack(side=tk.LEFT, padx=8)

        digital_test = ttk.LabelFrame(
            welder_test, text="ARC ON / ARC OFF · uses recipe below"
        )
        digital_test.pack(fill=tk.X, pady=2)
        self.hicomm_arc_on_button = ttk.Button(
            digital_test,
            text="ARC ON",
            command=lambda: self.request_digital_arc(True),
            state=tk.DISABLED,
        )
        self.hicomm_arc_on_button.grid(row=0, column=6, padx=3)
        self.hicomm_arc_off_button = ttk.Button(
            digital_test,
            text="ARC OFF",
            command=lambda: self.request_digital_arc(False),
        )
        self.hicomm_arc_off_button.grid(row=0, column=7, padx=3)
        self.hicomm_weld_status = ttk.Label(
            digital_test, text="DISCONNECTED · ARC OFF"
        )
        self.hicomm_weld_status.grid(row=0, column=8, padx=8)
        self.fake_arc_check = ttk.Checkbutton(
            digital_test,
            text="Fake ARC (motion only, no real welding)",
            variable=self.fake_arc_enabled,
            command=self.fake_arc_changed,
        )
        self.fake_arc_check.grid(
            row=0, column=9, padx=(12, 3), sticky=tk.W
        )

        digital = ttk.Frame(weld_parameters)
        digital.pack(fill=tk.X)
        self._add_labeled_value(digital, 0, "current A", self.weld_current_raw)
        self._add_labeled_value(digital, 1, "voltage ×0.1 V", self.weld_voltage_raw)
        for column, (label, variable, values, width) in enumerate((
            ("material", self.weld_material, tuple(MATERIAL_CODES), 11),
            ("diameter", self.weld_diameter_mm, tuple(DIAMETER_CODES), 5),
            ("mode", self.weld_mode, tuple(MODE_CODES), 5),
            ("gas", self.weld_gas, tuple(GAS_CODES), 14),
        )):
            ttk.Label(digital, text=label).grid(
                row=1, column=column * 2, padx=(3, 2), pady=3
            )
            ttk.Combobox(
                digital,
                textvariable=variable,
                values=values,
                state="readonly",
                width=width,
            ).grid(row=1, column=column * 2 + 1, padx=(0, 4), pady=3)
        ttk.Checkbutton(
            digital, text="synergic", variable=self.weld_synergic
        ).grid(row=2, column=0, columnspan=2, padx=3, sticky=tk.W)
        ttk.Label(digital, text="correction").grid(
            row=2, column=2, padx=(3, 2), pady=3
        )
        ttk.Entry(digital, textvariable=self.weld_correction, width=7).grid(
            row=2, column=3, padx=(0, 4), pady=3
        )
        ttk.Checkbutton(
            digital, text="Hot start (native)", variable=self.weld_hot_start_enabled
        ).grid(row=2, column=4, padx=(8, 2), pady=3, sticky=tk.W)
        ttk.Label(digital, text="Hot current boost %").grid(row=2, column=5, padx=(2, 1))
        ttk.Spinbox(
            digital, from_=0.0, to=100.0, increment=1.0,
            textvariable=self.weld_hot_start_percent, width=5,
        ).grid(row=2, column=6, padx=(1, 3))
        ttk.Label(digital, text="hold adj").grid(row=2, column=7, padx=(2, 1))
        ttk.Spinbox(
            digital, from_=-15, to=15, increment=1,
            textvariable=self.weld_hot_start_hold_adjustment, width=5,
        ).grid(row=2, column=8, padx=(1, 3))
        ttk.Checkbutton(
            digital, text="Custom Hot Start (Current boost + Hold)",
            variable=self.weld_custom_hot_start_enabled,
        ).grid(row=3, column=0, padx=(3, 2), pady=3, sticky=tk.W)
        ttk.Label(digital, text="Hold Time s").grid(row=3, column=1, padx=(2, 1))
        ttk.Spinbox(
            digital, from_=0.01, to=5.0, increment=0.05,
            textvariable=self.weld_custom_hot_start_hold_s, width=5,
        ).grid(row=3, column=2, padx=(1, 3))
        ttk.Label(digital, text="Custom boost %").grid(row=3, column=3)
        ttk.Spinbox(
            digital, from_=0, to=100, increment=1,
            textvariable=self.weld_custom_hot_start_percent, width=5,
        ).grid(row=3, column=4)
        ttk.Checkbutton(
            digital, text="Observe panel native crater (RX only)", variable=self.weld_expect_native_crater
        ).grid(row=4, column=0, padx=(3, 2), pady=3, sticky=tk.W)
        ttk.Label(digital, text="Panel Current Ref A").grid(row=4, column=1, padx=(2, 1))
        ttk.Spinbox(
            digital, from_=0.0, to=600.0, increment=5.0,
            textvariable=self.weld_crater_panel_current_ref_a, width=5,
        ).grid(row=4, column=2, padx=(1, 3))
        ttk.Label(digital, text="Panel Voltage Ref V").grid(row=4, column=3, padx=(2, 1))
        ttk.Spinbox(
            digital, from_=3.0, to=80.0, increment=0.1,
            textvariable=self.weld_crater_panel_voltage_ref_v, width=5,
        ).grid(row=4, column=4, padx=(1, 3))
        ttk.Label(digital, text="Panel Time Ref s").grid(row=4, column=5, padx=(2, 1))
        ttk.Spinbox(
            digital, from_=0.0, to=30.0, increment=0.1,
            textvariable=self.weld_crater_panel_time_ref_s, width=5,
        ).grid(row=4, column=6, padx=(1, 3))
        ttk.Checkbutton(digital, text="Software Crater Enabled",
                        variable=self.weld_software_crater_enabled).grid(row=5, column=0, sticky=tk.W)
        ttk.Label(digital, text="Current Ratio %").grid(row=5, column=1)
        ttk.Spinbox(digital, from_=20.0, to=40.0, increment=1.0,
                    textvariable=self.weld_software_crater_ratio_percent, width=5).grid(row=5, column=2)
        ttk.Label(digital, text="Crater Voltage V").grid(row=5, column=3)
        ttk.Spinbox(digital, from_=10.0, to=40.0, increment=0.1,
                    textvariable=self.weld_software_crater_voltage_v, width=5).grid(row=5, column=4)
        ttk.Label(digital, text="Hold s").grid(row=5, column=5)
        ttk.Spinbox(digital, from_=0.1, to=5.0, increment=0.1,
                    textvariable=self.weld_software_crater_hold_s, width=5).grid(row=5, column=6)
        ttk.Label(digital, text="Wire alpha mm").grid(
            row=6, column=0, padx=(8, 2), pady=3
        )
        ttk.Entry(
            digital, textvariable=self.weld_wire_consumable_alpha_mm, width=7
        ).grid(row=6, column=1, padx=(0, 4), pady=3)
        self.hicomm_rx_bit_status = ttk.Label(
            digital_test,
            text="RX Byte0 · b5 WCR=0 · b4 STICK=0 · "
            "b3 GAS CHECK=0 · b0 TORCH=0",
            font=("Monospace", 10, "bold"),
        )
        self.hicomm_rx_bit_status.grid(
            row=1, column=0, columnspan=9, padx=8, pady=3, sticky=tk.W
        )

    def _build_seam_correction_controls(self, welding_page):
        touch_corner = self._create_toggle_section(
            welding_page,
            "touch_corner",
            "Seam Correction · Fastech DI4 wall/base probing + seam-yaw orientation",
        )

        geometry_controls = ttk.Frame(touch_corner)
        geometry_controls.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(geometry_controls, text="wall probe direction").pack(side=tk.LEFT)
        ttk.Combobox(
            geometry_controls,
            textvariable=self.wall_probe_axis,
            values=(
                "AUTO ⟂ taught seam (XY)",
                "World X",
                "World Y",
                "World Z",
            ),
            state="readonly",
            width=24,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(geometry_controls, text="sign").pack(side=tk.LEFT)
        ttk.Combobox(
            geometry_controls,
            textvariable=self.wall_probe_sign,
            values=("+", "-"),
            state="readonly",
            width=2,
        ).pack(side=tk.LEFT, padx=(3, 10))

        ttk.Label(geometry_controls, text="base/floor probe direction").pack(side=tk.LEFT)
        ttk.Combobox(
            geometry_controls,
            textvariable=self.floor_probe_axis,
            values=("World X", "World Y", "World Z"),
            state="readonly",
            width=9,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(geometry_controls, text="sign").pack(side=tk.LEFT)
        ttk.Combobox(
            geometry_controls,
            textvariable=self.floor_probe_sign,
            values=("+", "-"),
            state="readonly",
            width=2,
        ).pack(side=tk.LEFT, padx=(3, 10))

        ttk.Label(geometry_controls, text="orientation").pack(side=tk.LEFT)
        ttk.Combobox(
            geometry_controls,
            textvariable=self.seam_orientation_mode,
            values=(
                "Follow sensed seam yaw",
                "Keep reference orientation",
                WAIT_FIXED_TILT_ORIENTATION_MODE,
            ),
            state="readonly",
            width=27,
        ).pack(side=tk.LEFT, padx=(4, 8))
        for axis, variable in (
            ("X", self.weld_fixed_tilt_x_deg),
            ("Y", self.weld_fixed_tilt_y_deg),
            ("Z", self.weld_fixed_tilt_z_deg),
        ):
            ttk.Label(geometry_controls, text=f"World-{axis} °").pack(
                side=tk.LEFT
            )
            ttk.Spinbox(
                geometry_controls,
                from_=-180.0,
                to=180.0,
                increment=0.5,
                textvariable=variable,
                width=6,
            ).pack(side=tk.LEFT, padx=(3, 6))

        yaw_summary = ttk.Frame(touch_corner)
        yaw_summary.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(yaw_summary, text="Orientation status", font=("Sans", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        for variable in (
            self.reference_yaw_status,
            self.sensed_yaw_status,
            self.delta_yaw_status,
        ):
            ttk.Label(yaw_summary, textvariable=variable).pack(
                side=tk.LEFT, padx=(0, 14)
            )

        safe_approach_controls = ttk.Frame(touch_corner)
        safe_approach_controls.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(safe_approach_controls, text="Approach mode").pack(side=tk.LEFT)
        ttk.Combobox(
            safe_approach_controls, textvariable=self.weld_approach_mode,
            values=("taught_wait", "corner_geometry"), state="readonly", width=18,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(
            safe_approach_controls, text="Safe approach distance mm"
        ).pack(side=tk.LEFT)
        ttk.Spinbox(
            safe_approach_controls,
            from_=1.0,
            to=200.0,
            increment=1.0,
            textvariable=self.weld_safe_approach_mm,
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(
            safe_approach_controls, text="Pre-start lead distance mm"
        ).pack(side=tk.LEFT)
        ttk.Spinbox(
            safe_approach_controls,
            from_=0.0,
            to=100.0,
            increment=1.0,
            textvariable=self.weld_pre_start_lead_mm,
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(
            safe_approach_controls,
            text="taught_wait: taught clearance + weld attitude; corner_geometry: e_a clearance",
        ).pack(side=tk.LEFT, padx=(8, 0))

        weld_lead_controls = ttk.Frame(touch_corner)
        weld_lead_controls.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(
            weld_lead_controls,
            text="weld lead-in mm",
        ).pack(side=tk.LEFT)
        ttk.Spinbox(
            weld_lead_controls,
            from_=0.0,
            to=100.0,
            increment=1.0,
            textvariable=self.weld_lead_in_mm,
            width=6,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(
            weld_lead_controls,
            text="weld lead-out mm",
        ).pack(side=tk.LEFT)
        ttk.Spinbox(
            weld_lead_controls,
            from_=0.0,
            to=100.0,
            increment=1.0,
            textvariable=self.weld_lead_out_mm,
            width=6,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(weld_lead_controls, text="avg seam travel mm/s").pack(side=tk.LEFT)
        ttk.Spinbox(
            weld_lead_controls,
            from_=0.1,
            to=100.0,
            increment=0.1,
            textvariable=self.weld_tcp_speed_mm_s,
            width=6,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(weld_lead_controls, text="ARC OFF lead ms").pack(side=tk.LEFT)
        ttk.Spinbox(
            weld_lead_controls,
            from_=0.0,
            to=2000.0,
            increment=10.0,
            textvariable=self.weld_arc_off_delay_ms,
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(
            weld_lead_controls,
            text=(
                "weld stroke uses fixed TCP target; global scale remains for approach/return"
            ),
        ).pack(side=tk.LEFT, padx=(8, 0))

        weld_weave_controls = ttk.Frame(touch_corner)
        weld_weave_controls.pack(fill=tk.X, pady=(0, 3))
        ttk.Checkbutton(
            weld_weave_controls,
            text="use weave in weld scenario",
            variable=self.weld_weave_enabled,
        ).pack(side=tk.LEFT)
        ttk.Label(weld_weave_controls, text="pattern").pack(
            side=tk.LEFT, padx=(10, 2)
        )
        ttk.Combobox(
            weld_weave_controls,
            textvariable=self.weave_pattern,
            values=("sine", "crescent", "circle"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(weld_weave_controls, text="A mm").pack(
            side=tk.LEFT
        )
        ttk.Spinbox(
            weld_weave_controls, from_=0.1, to=50.0, increment=0.1,
            textvariable=self.weave_amplitude_mm, width=6,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(weld_weave_controls, text="pitch mm/cycle").pack(side=tk.LEFT)
        ttk.Spinbox(
            weld_weave_controls, from_=0.1, to=100.0, increment=0.1,
            textvariable=self.weave_pitch_mm, width=6,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(weld_weave_controls, text="radial/transverse axis").pack(
            side=tk.LEFT
        )
        ttk.Combobox(
            weld_weave_controls,
            textvariable=self.weave_axis,
            values=(
                "tool_x", "tool_y", "tool_z",
                "world_x", "world_y", "world_z",
            ),
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(
            touch_corner,
            text=(
                "A: sine = centerline ±A (full width 2A); "
                "circle = orbit radius A (diameter 2A, ramped at ends)."
            ),
            foreground="#174ea6",
        ).pack(anchor=tk.W, pady=(0, 3))
        dwell_controls = ttk.Frame(touch_corner)
        dwell_controls.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(dwell_controls, text="Sine peak dwell (travel stops)").pack(side=tk.LEFT)
        for label, variable in (
            ("Left s", self.weave_left_dwell_s),
            ("Right s", self.weave_right_dwell_s),
        ):
            ttk.Label(dwell_controls, text=label).pack(side=tk.LEFT, padx=(10, 2))
            ttk.Spinbox(
                dwell_controls, from_=0.0, to=10.0, increment=0.1,
                textvariable=variable, width=5,
            ).pack(side=tk.LEFT)
        ttk.Label(dwell_controls, text="Circle requires both 0").pack(
            side=tk.LEFT, padx=(12, 0)
        )

        motion_controls = ttk.Frame(touch_corner)
        motion_controls.pack(fill=tk.X, pady=(0, 3))
        for label, variable, width in (
            ("max travel mm", self.touch_probe_distance_mm, 6),
            ("speed %", self.touch_probe_speed_percent, 5),
        ):
            ttk.Label(motion_controls, text=label).pack(side=tk.LEFT, padx=(0, 2))
            ttk.Entry(
                motion_controls, textvariable=variable, width=width
            ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(motion_controls, text="settle s").pack(side=tk.LEFT)
        ttk.Spinbox(
            motion_controls,
            from_=0.2,
            to=5.0,
            increment=0.1,
            textvariable=self.touch_settle_seconds,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(motion_controls, text="path points").pack(side=tk.LEFT)
        ttk.Spinbox(
            motion_controls,
            from_=2,
            to=200,
            textvariable=self.corner_touch_count,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 10))

        auto_actions = ttk.Frame(touch_corner)
        auto_actions.pack(fill=tk.X, padx=3, pady=(5, 3))
        self.auto_seam_correction_button = ttk.Button(
            auto_actions,
            text=(
                "AUTO ALL · START WAIT → WALL/BASE → GOAL WAIT → "
                "WALL/BASE → COMPUTE/SAVE"
            ),
            command=self.run_automatic_seam_correction,
        )
        self.auto_seam_correction_button.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_auto_seam_button = ttk.Button(
            auto_actions,
            text="STOP AUTO",
            command=self.stop_automatic_seam_correction,
            state=tk.DISABLED,
        )
        self.stop_auto_seam_button.pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(
            auto_actions,
            text="Move to Weld end after correction",
            variable=self.auto_seam_move_to_end_pose,
        ).pack(side=tk.LEFT, padx=(12, 0))

        # Multi-pass alignment is a separate production workflow.  Keeping it
        # inside the single-seam touch panel made the root correction and the
        # per-pass verification controls look like part of one operation.
        multi_pass = self._create_toggle_section(
            welding_page,
            "multi_pass_correction",
            "Multi-pass Seam Correction · 1G root/pass alignment",
            expanded=False,
        )
        four_pass = ttk.LabelFrame(
            multi_pass, text="4-pass reference set · cumulative selected-pass anchor"
        )
        four_pass.pack(fill=tk.X, pady=3)
        four_pass_row = ttk.Frame(four_pass)
        four_pass_row.pack(fill=tk.X, pady=2)
        ttk.Entry(
            four_pass_row, textvariable=self.four_pass_folder, width=48
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            four_pass_row, text="Browse...", command=self.browse_four_pass_folder
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            four_pass_row, text="Load 4 references", command=self.load_four_pass_references
        ).pack(side=tk.LEFT, padx=3)
        ttk.Label(
            four_pass,
            text=(
                "Browse the work folder: pass_N.yaml overrides N.log "
                "for each pass"
            ),
        ).pack(anchor=tk.W, padx=3, pady=(0, 2))
        pass_probe_row = ttk.Frame(four_pass)
        pass_probe_row.pack(fill=tk.X, pady=2)
        ttk.Label(pass_probe_row, text="Pass").pack(side=tk.LEFT, padx=3)
        pass_selection = ttk.Combobox(
            pass_probe_row, textvariable=self.selected_pass_number,
            values=(1, 2, 3, 4), width=4, state="readonly",
        )
        pass_selection.pack(side=tk.LEFT, padx=3)
        pass_selection.bind(
            "<<ComboboxSelected>>", lambda _event: self._selected_pass()
        )
        ttk.Button(
            pass_probe_row, text="Load Selected Pass",
            command=self.apply_selected_pass_correction,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            pass_probe_row, text="Save Pass Teaching YAML",
            command=self.save_teaching_to_selected_pass,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            pass_probe_row, text="Multi-pass Seam Correction",
            command=self.run_four_pass_correction,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            pass_probe_row, text="STOP MULTI-PASS (ALL MOTION)",
            command=self.stop_multi_pass_correction,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Label(four_pass, textvariable=self.four_pass_status).pack(
            anchor=tk.W, padx=3
        )
        ttk.Label(
            four_pass,
            text=(
                "Each N.log supplies that pass's START WAIT, START, GOAL WAIT and GOAL. "
                "Select Pass N, move from its corrected START WAIT, jog to the real START "
                "and press I; then use its corrected GOAL WAIT, jog to GOAL and press J. "
                "Selected pass: captured TCP1/2, WAIT unchanged. Transform only N+1..4. No welding starts."
            ),
            foreground="#b3261e",
        ).pack(anchor=tk.W, padx=3)

        self.corner_touch_status = ttk.Label(
            touch_corner,
            text=(
                "Teach rough START/GOAL first · AUTO wall follows the taught seam normal · "
                "touch geometry corrects XYZ; START→GOAL corrects welding yaw"
            ),
        )
        self.corner_touch_status.pack(anchor=tk.W, pady=(3, 0))

    def _build_sequence_and_sensing_controls(self, sequence_page, shared_task_controls):
        sequence = self._create_toggle_section(
            shared_task_controls, "sequence", "Sequence Builder", expanded=True
        )
        sequence_buttons = ttk.Frame(sequence)
        sequence_buttons.pack(fill=tk.X, pady=(0, 4))
        for text, command in (
            ("Build Weld Scenario", self.build_sensed_weld_sequence),
            ("Build Torch Clean", self.build_torch_clean_sequence),
            ("Build Full Work Cycle", self.build_combined_work_cycle),
            ("Execute", lambda: self.run_sequence(True, True)),
            ("Delete", self.delete_sequence_step),
            ("Delete All", self.delete_all_sequence_steps),
            ("STOP", self.stop_sequence),
        ):
            ttk.Button(sequence_buttons, text=text, command=command).pack(
                side=tk.LEFT, padx=2, pady=2
            )
        ttk.Label(sequence_buttons, text="Repeat").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(sequence_buttons, from_=1, to=20, width=3,
                    textvariable=self.work_cycle_repeats).pack(side=tk.LEFT)
        self.sequence_table = ttk.Treeview(
            sequence,
            columns=("order", "type", "detail"),
            show="headings",
            height=5,
            selectmode="browse",
        )
        for name, width in (("order", 60), ("type", 130), ("detail", 850)):
            self.sequence_table.heading(name, text=name.upper())
            self.sequence_table.column(name, width=width, anchor=tk.W)
        self.sequence_table.pack(fill=tk.X)
        self.sequence_table.bind(
            "<<TreeviewSelect>>", self.load_selected_sequence_values
        )
        self.sequence_table.bind(
            "<Double-1>", self.open_sequence_step_editor
        )
        self.sequence_status = ttk.Label(
            sequence, text="Sequence idle · Build → Execute all rows"
        )
        self.sequence_status.pack(anchor=tk.W, pady=(3, 0))

        planned_path = self._create_toggle_section(
            sequence_page, "planned_path", "Planned Path · World frame"
        )
        columns = ("id",) + self.POSE_FIELDS
        self.table = ttk.Treeview(
            planned_path,
            columns=columns,
            show="headings",
            height=4,
            selectmode="browse",
        )
        for name in columns:
            self.table.heading(name, text=name.upper())
            self.table.column(
                name,
                width=48 if name == "id" else 105,
                anchor=tk.CENTER,
            )
        self.table.pack(fill=tk.X)
        ttk.Button(
            planned_path,
            text="Delete All",
            command=self.clear_path,
        ).pack(anchor=tk.E, pady=(5, 0))

        wide_sensing = self._create_toggle_section(
            sequence_page,
            "wide_sensing",
            "Wide Sensing · detected weld segment → World planned path",
            expanded=False,
        )
        wide_row = ttk.Frame(wide_sensing)
        wide_row.pack(fill=tk.X, pady=2)
        ttk.Label(wide_row, text="segment").pack(side=tk.LEFT)
        self.wide_sensing_segment_box = ttk.Combobox(
            wide_row,
            textvariable=self.wide_sensing_segment_id,
            values=(),
            state="readonly",
            width=24,
        )
        self.wide_sensing_segment_box.pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(wide_row, text="source frame").pack(side=tk.LEFT)
        ttk.Entry(
            wide_row,
            textvariable=self.wide_sensing_source_frame,
            width=14,
        ).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Checkbutton(
            wide_row,
            text="reverse START/END",
            variable=self.wide_sensing_reverse,
        ).pack(side=tk.LEFT, padx=(0, 10))
        for axis, variable in (
            ("World offset X mm", self.wide_sensing_offset_x_mm),
            ("Y", self.wide_sensing_offset_y_mm),
            ("Z", self.wide_sensing_offset_z_mm),
        ):
            ttk.Label(wide_row, text=axis).pack(side=tk.LEFT)
            ttk.Entry(
                wide_row,
                textvariable=variable,
                width=6,
            ).pack(side=tk.LEFT, padx=(3, 7))
        self.wide_sensing_load_button = ttk.Button(
            wide_row,
            text="Load segment as planned path",
            command=self.load_wide_sensing_segment,
            state=tk.DISABLED,
        )
        self.wide_sensing_load_button.pack(side=tk.LEFT, padx=(8, 3))
        wide_actions = ttk.Frame(wide_sensing)
        wide_actions.pack(fill=tk.X, pady=(2, 0))
        self.wide_sensing_plan_button = ttk.Button(
            wide_actions,
            text="Load + Plan Preview",
            command=lambda: self.load_wide_sensing_segment(True),
            state=tk.DISABLED,
        )
        self.wide_sensing_plan_button.pack(side=tk.LEFT, padx=3)
        self.wide_sensing_execute_button = ttk.Button(
            wide_actions,
            text="Execute Approved",
            command=self.execute_approved,
            state=tk.DISABLED,
        )
        self.wide_sensing_execute_button.pack(side=tk.LEFT, padx=3)
        ttk.Label(
            wide_sensing,
            textvariable=self.wide_sensing_status,
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(
            wide_sensing,
            text=(
                "Input XYZ is metre in helios_link. TF converts it to World; "
                "START/END preserve the selected robot's current TCP orientation. "
                "Use Plan Preview before Execute."
            ),
        ).pack(anchor=tk.W, pady=(1, 2))

    def _build_fastech_controls(self, connection_page):
        fastech_io = self._create_toggle_section(
            connection_page,
            "fastech_ethernet",
            "Fastech ROS I/O · 0 Touch · 5/6/7 Torch cleaner",
        )
        ttk.Label(fastech_io, text="IP").grid(
            row=0, column=0, padx=(6, 2), pady=4, sticky=tk.E
        )
        ttk.Entry(
            fastech_io,
            textvariable=self.fastech_ip,
            width=15,
            state="readonly",
        ).grid(row=0, column=1, padx=(2, 8), pady=4)
        ttk.Label(fastech_io, text="Board ID").grid(
            row=0, column=2, padx=(2, 2), pady=4, sticky=tk.E
        )
        ttk.Spinbox(
            fastech_io,
            from_=0,
            to=255,
            textvariable=self.fastech_board_id,
            width=5,
            state="readonly",
        ).grid(row=0, column=3, padx=(2, 8), pady=4)
        self.fastech_connect_button = ttk.Button(
            fastech_io,
            text="Connect Fastech",
            command=self.connect_fastech_ethernet,
        )
        self.fastech_connect_button.grid(row=0, column=4, padx=3, pady=4)
        self.fastech_disconnect_button = ttk.Button(
            fastech_io,
            text="Disconnect",
            command=self.disconnect_fastech_ethernet,
            state=tk.DISABLED,
        )
        self.fastech_disconnect_button.grid(row=0, column=5, padx=3, pady=4)
        self.fastech_all_off_button = ttk.Button(
            fastech_io,
            text="Exposed DO all OFF",
            command=self.fastech_outputs_all_off,
            state=tk.DISABLED,
        )
        self.fastech_all_off_button.grid(
            row=0, column=6, padx=(12, 3), pady=4
        )
        self.fastech_io_status = ttk.Label(
            fastech_io,
            text="Starting · auto-connect to 192.168.0.3",
        )
        self.fastech_io_status.grid(
            row=0, column=7, columnspan=2, padx=(10, 6), pady=4, sticky=tk.W
        )

        for column, heading in enumerate(
            ("Channel", "Function", "DI state", "DO state", "DO ON", "DO OFF")
        ):
            ttk.Label(
                fastech_io, text=heading, font=("Sans", 9, "bold")
            ).grid(row=1, column=column, padx=6, pady=(4, 2), sticky=tk.W)
        for row, (channel, description) in enumerate(
            FASTECH_GUI_CHANNELS.items(), start=2
        ):
            ttk.Label(fastech_io, text=str(channel)).grid(
                row=row, column=0, padx=6, pady=3, sticky=tk.W
            )
            ttk.Label(fastech_io, text=description).grid(
                row=row, column=1, padx=6, pady=3, sticky=tk.W
            )
            for column, kind in ((2, "DI"), (3, "DO")):
                label = tk.Label(
                    fastech_io,
                    text=f"{kind}{channel} –",
                    width=10,
                    relief=tk.SOLID,
                    bg="#eeeeee",
                    font=("Monospace", 9, "bold"),
                )
                label.grid(row=row, column=column, padx=6, pady=3)
                self.fastech_io_labels[(kind, channel)] = label
            on_button = ttk.Button(
                fastech_io,
                text="ON",
                state=tk.DISABLED,
                command=lambda selected=channel: self.request_fastech_output(
                    selected, True
                ),
            )
            off_button = ttk.Button(
                fastech_io,
                text="OFF",
                state=tk.DISABLED,
                command=lambda selected=channel: self.request_fastech_output(
                    selected, False
                ),
            )
            on_button.grid(row=row, column=4, padx=3, pady=3)
            off_button.grid(row=row, column=5, padx=3, pady=3)
            self.fastech_output_buttons.extend((on_button, off_button))

    def _build_cleaner_controls(self, cleaner_page):
        from .torch_cleaner_panel import TorchCleanerPanel
        self.torch_cleaner_panel = TorchCleanerPanel(
            self, cleaner_page, save_initial_state_yaml, load_initial_state_yaml
        )

    def _build_legacy_io_controls(self, connection_page):
        io_monitor = self._create_toggle_section(
            connection_page,
            "digital_io",
            "Legacy Rainbow Controller Digital I/O test · ports 0..15",
        )
        for io_row, kind in enumerate(("DI", "DO")):
            ttk.Label(
                io_monitor,
                text=kind,
                font=("Sans", 10, "bold"),
            ).grid(row=io_row, column=0, padx=(6, 4), pady=3)
            for port in range(16):
                candidate = port in MANUAL_IO_CANDIDATES
                label = tk.Label(
                    io_monitor,
                    text=f"{port:02d}\n–",
                    width=4,
                    relief=tk.SOLID,
                    borderwidth=2 if candidate else 1,
                    bg="#dbeafe" if candidate else "#eeeeee",
                    font=("Monospace", 9, "bold" if candidate else "normal"),
                )
                label.grid(row=io_row, column=port + 1, padx=2, pady=3)
                self.control_box_io_labels[(kind, port)] = label
                if kind == "DO":
                    label.configure(cursor="hand2")
                    label.bind(
                        "<Button-1>",
                        lambda _event, selected=port: (
                            self.request_do_toggle(selected)
                        ),
                    )
        self.control_box_io_status = ttk.Label(
            io_monitor,
            text="Waiting for /right_rbpodo_hardware/system_state",
        )
        self.control_box_io_status.grid(
            row=2,
            column=0,
            columnspan=17,
            sticky=tk.W,
            padx=6,
            pady=(2, 5),
        )
        ttk.Checkbutton(
            io_monitor,
            text="Unlock clicking non-candidate DO ports",
            variable=self.unlock_all_do_ports,
            command=self.confirm_all_do_unlock,
        ).grid(
            row=3,
            column=0,
            columnspan=12,
            sticky=tk.W,
            padx=6,
            pady=(0, 5),
        )
        ttk.Button(
            io_monitor,
            text="Candidate DO all OFF",
            command=self.candidate_outputs_off,
        ).grid(
            row=3,
            column=12,
            columnspan=5,
            sticky=tk.E,
            padx=6,
            pady=(0, 5),
        )

    def _build_sequence_execution_controls(self, sequence_page):
        execution = ttk.Frame(sequence_page)
        execution.pack(fill=tk.X, pady=(12, 0))
        self.plan_button = ttk.Button(
            execution,
            text="1 · Plan Preview",
            command=self.plan_preview,
            state=tk.DISABLED,
        )
        self.plan_button.pack(side=tk.LEFT, padx=(0, 8))
        self.execute_button = ttk.Button(
            execution,
            text="2 · Execute Approved Plan",
            command=self.execute_approved,
            state=tk.DISABLED,
        )
        self.execute_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            execution,
            text="Cancel",
            command=self.cancel,
        ).pack(side=tk.LEFT, padx=(0, 18))

        planning_settings = ttk.Frame(sequence_page)
        planning_settings.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(planning_settings, text="Speed mode").pack(side=tk.LEFT)
        ttk.Radiobutton(
            planning_settings,
            text="Velocity scale (%)",
            variable=self.speed_mode,
            value="scale",
            command=self.speed_mode_changed,
        ).pack(side=tk.LEFT, padx=(4, 6))
        ttk.Radiobutton(
            planning_settings,
            text="TCP average speed",
            variable=self.speed_mode,
            value="tcp",
            command=self.speed_mode_changed,
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.tcp_speed_spinbox = ttk.Spinbox(
            planning_settings,
            from_=0.1,
            to=500.0,
            increment=0.5,
            textvariable=self.tcp_speed_mm_s,
            width=7,
            command=self.speed_mode_changed,
        )
        self.tcp_speed_spinbox.pack(side=tk.LEFT)
        self.tcp_speed_spinbox.bind(
            "<FocusOut>", lambda _event: self.speed_mode_changed()
        )
        self.tcp_speed_spinbox.bind(
            "<Return>", lambda _event: self.speed_mode_changed()
        )
        ttk.Label(planning_settings, text="mm/s").pack(
            side=tk.LEFT, padx=(2, 14)
        )
        ttk.Label(planning_settings, text="Cartesian interpolation step mm").pack(
            side=tk.LEFT
        )
        ttk.Spinbox(
            planning_settings,
            from_=0.5,
            to=20.0,
            increment=0.5,
            textvariable=self.interpolation_step_mm,
            width=6,
            command=self.invalidate_approved_plan,
        ).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Checkbutton(
            planning_settings,
            text="Linear (constant velocity)",
            variable=self.linear_motion_profile,
            command=self._motion_profile_toggled,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.motion_profile_label = ttk.Label(planning_settings, text="")
        self.motion_profile_label.pack(side=tk.LEFT)
        self._update_motion_profile_label()

        ttk.Label(
            sequence_page,
            text="Action feedback",
            style="Step.TLabel",
        ).pack(anchor=tk.W, pady=(12, 5))
        self.bar = ttk.Progressbar(sequence_page, maximum=100)
        self.bar.pack(fill=tk.X)
        self.feedback_label = ttk.Label(
            sequence_page,
            text="waypoint: –    pose: –",
        )
        self.feedback_label.pack(anchor=tk.W, pady=4)

        ttk.Label(
            sequence_page,
            text="Pipeline status",
            style="Step.TLabel",
        ).pack(anchor=tk.W, pady=(8, 5))
        self.pipeline_status = tk.Label(
            sequence_page,
            text="WAITING · ready",
            anchor=tk.W,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#eeeeee",
            font=("Sans", 10, "bold"),
        )
        self.pipeline_status.pack(fill=tk.X, ipady=5)

    def _build_operator_layout(self):
        """Construct Connection, Task, and Sequence pages from shared Tk state."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 18, "bold"))
        style.configure("Step.TLabel", font=("Sans", 11, "bold"))
        section_colors = (
            ("SectionBlue.TButton", "#dbeafe", "#bfdbfe"),
            ("SectionGreen.TButton", "#dcfce7", "#bbf7d0"),
            ("SectionAmber.TButton", "#fef3c7", "#fde68a"),
            ("SectionViolet.TButton", "#ede9fe", "#ddd6fe"),
        )
        for name, normal, active in section_colors:
            style.configure(
                name,
                background=normal,
                foreground="#172033",
                font=("Sans", 10, "bold"),
                padding=(8, 6),
                anchor=tk.W,
            )
            style.map(name, background=[("active", active)])

        self.operator_notebook = ttk.Notebook(self.root)
        self.operator_notebook.pack(fill=tk.BOTH, expand=True)
        self._operator_page_canvases = {}
        connection_page = self._create_operator_scroll_page(
            self.operator_notebook, "1. CONNECTION"
        )
        task_page = ttk.Frame(self.operator_notebook, padding=8)
        self.operator_notebook.add(task_page, text="2. TASK")
        shared_task_controls = ttk.Frame(task_page)
        shared_task_controls.pack(fill=tk.X)
        self.task_notebook = ttk.Notebook(task_page)
        self.task_notebook.pack(fill=tk.BOTH, expand=True)
        welding_page = self._create_operator_scroll_page(
            self.task_notebook, "Welding"
        )
        cleaner_page = self._create_operator_scroll_page(
            self.task_notebook, "Torch Cleaner"
        )
        sequence_page = self._create_operator_scroll_page(
            self.operator_notebook, "3. etc"
        )
        self.content_canvas, self.content_window = self._operator_page_canvases[
            self.operator_notebook.tabs()[0]
        ]
        self.operator_notebook.bind("<<NotebookTabChanged>>", self._operator_page_changed)
        self.task_notebook.bind("<<NotebookTabChanged>>", self._operator_page_changed)
        self.root.bind_all("<MouseWheel>", self._scroll_content)
        self.root.bind_all("<Button-4>", self._scroll_content)
        self.root.bind_all("<Button-5>", self._scroll_content)
        # Tk processes a widget's class binding before bind_all. Spinbox,
        # Combobox and Scale can therefore change their values on a wheel
        # event even though the page also scrolls. Override only those class
        # wheel bindings so the wheel scrolls the page without editing values.
        for widget_class in (
            "TSpinbox", "Spinbox", "TCombobox", "Combobox", "TScale", "Scale",
        ):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.root.bind_class(
                    widget_class, sequence, self._scroll_value_control
                )
        ttk.Label(connection_page, text="Connection & diagnostics", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(welding_page, text="Welding task", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(cleaner_page, text="Torch Cleaner task", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(sequence_page, text="Etc · Diagnostics / Wide Sensing", style="Title.TLabel").pack(anchor=tk.W)

        self._build_connection_header(connection_page)

        self.keyboard_jog_enable_buttons = []
        teaching_defaults = (
            self.keyboard_jog_selection.get(), self.keyboard_jog_frame.get(),
            self.keyboard_jog_linear_speed.get(), self.keyboard_jog_angular_speed.get(),
            self.velocity_percent.get(),
        )
        self._teaching_task_settings = {
            "welding": teaching_defaults, "cleaner": teaching_defaults,
        }
        self._active_teaching_task = "welding"
        self._build_keyboard_teaching_controls(welding_page)
        self._build_keyboard_teaching_controls(cleaner_page, cleaner=True)

        motion_tests = self._create_toggle_section(
            sequence_page, "motion_test", "Motion Test", expanded=False
        )
        straight = ttk.Frame(motion_tests)
        straight.pack(fill=tk.X, pady=2)
        ttk.Button(
            straight,
            text="Generate linear path",
            command=self.acquire,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(straight, text="reference").pack(side=tk.LEFT)
        ttk.Combobox(
            straight,
            textvariable=self.straight_reference,
            values=("world", "tool"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 7))
        ttk.Label(straight, text="axis").pack(side=tk.LEFT)
        ttk.Combobox(
            straight,
            textvariable=self.straight_axis,
            values=("+X", "-X", "+Y", "-Y", "+Z", "-Z"),
            state="readonly",
            width=4,
        ).pack(side=tk.LEFT, padx=(3, 7))
        ttk.Label(straight, text="distance mm").pack(side=tk.LEFT)
        ttk.Spinbox(
            straight,
            from_=0.1,
            to=5000,
            increment=1,
            textvariable=self.straight_distance_mm,
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 6))
        ttk.Label(straight, text="points").pack(side=tk.LEFT)
        ttk.Spinbox(
            straight,
            from_=2,
            to=200,
            increment=1,
            textvariable=self.straight_count,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 0))

        straight_angles = ttk.Frame(motion_tests)
        straight_angles.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(
            straight_angles,
            text="Angle adjustment",
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(straight_angles, text="reference").pack(side=tk.LEFT)
        ttk.Combobox(
            straight_angles,
            textvariable=self.straight_rotation_reference,
            values=("tool", "world"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 7))
        for label, variable in (
            ("ΔRoll °", self.straight_roll_deg),
            ("ΔPitch °", self.straight_pitch_deg),
            ("ΔYaw °", self.straight_yaw_deg),
        ):
            ttk.Label(straight_angles, text=label).pack(
                side=tk.LEFT, padx=(5, 2)
            )
            ttk.Entry(
                straight_angles, textvariable=variable, width=7
            ).pack(side=tk.LEFT)
        ttk.Label(
            straight_angles,
            text="Applied to every generated path TCP orientation",
            foreground="#5f6368",
        ).pack(side=tk.LEFT, padx=10)

        controls = ttk.Frame(motion_tests)
        controls.pack(fill=tk.X, pady=2)
        ttk.Button(
            controls,
            text="Generate circle",
            command=self.generate_circle,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(controls, text="axis").pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=self.circle_axis,
            values=("X", "Y", "Z"),
            state="readonly",
            width=3,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(controls, text="radius (mm)").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls,
            from_=1,
            to=200,
            increment=1,
            textvariable=self.radius_mm,
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(controls, text="unique points").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls,
            from_=4,
            to=200,
            increment=1,
            textvariable=self.circle_count,
            width=5,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Checkbutton(
            controls,
            text="close path",
            variable=self.close_circle,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            controls,
            text="TCP +Z faces center",
            variable=self.circle_face_center,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            motion_tests,
            text="show planned path",
            variable=self.show_path,
            command=self.toggle_path_visibility,
        ).pack(anchor=tk.W, pady=(3, 0))

        self._build_welder_controls(connection_page, welding_page)

        teaching = ttk.LabelFrame(welding_page, text="Teaching Detail · Welding poses / Plan / Execute / YAML")
        teaching.pack(fill=tk.X, pady=(7, 0))
        scale_row = ttk.Frame(teaching)
        scale_row.pack(fill=tk.X, side=tk.BOTTOM, pady=3)
        ttk.Label(scale_row, text="Build / Teaching speed scale %").pack(side=tk.LEFT)
        ttk.Scale(
            scale_row, from_=1, to=100, variable=self.velocity_percent,
            command=self.update_speed_label, length=220,
        ).pack(side=tk.LEFT, padx=7)
        self.speed_label = ttk.Label(scale_row, text=f"{self.velocity_percent.get():.1f}%")
        self.speed_label.pack(side=tk.LEFT)
        ttk.Label(teaching, text="pose").pack(side=tk.LEFT, padx=(3, 2))
        teaching_pose_box = ttk.Combobox(
            teaching,
            textvariable=self.teaching_pose_name,
            values=tuple(TEACHING_POSES.values()),
            state="readonly",
            width=23,
        )
        teaching_pose_box.pack(side=tk.LEFT, padx=3)
        teaching_pose_box.bind(
            "<<ComboboxSelected>>", self.teaching_pose_changed
        )
        ttk.Button(
            teaching,
            text="Capture current + save YAML",
            command=self.capture_initial_state,
        ).pack(side=tk.LEFT, padx=3)
        self.plan_initial_button = ttk.Button(
            teaching,
            text="1 · Plan selected pose",
            command=self.plan_initial_state,
            state=tk.DISABLED,
        )
        self.plan_initial_button.pack(side=tk.LEFT, padx=3)
        self.execute_initial_button = ttk.Button(
            teaching,
            text="2 · Execute selected plan",
            command=self.execute_initial_plan,
            state=tk.DISABLED,
        )
        self.execute_initial_button.pack(side=tk.LEFT, padx=3)
        ttk.Button(
            teaching,
            text="Load from YAML",
            command=self.load_initial_state,
        ).pack(side=tk.LEFT, padx=3)
        self.initial_state_status = ttk.Label(teaching, text="not captured")
        self.initial_state_status.pack(side=tk.LEFT, padx=(12, 0))
        self.path_summary = ttk.Label(teaching, text="empty path")
        cleaner_teaching = ttk.LabelFrame(cleaner_page, text="Teaching Detail · Cleaner positions / YAML")
        cleaner_teaching.pack(fill=tk.X, pady=(7, 0))
        cleaner_scale = ttk.Frame(cleaner_teaching)
        cleaner_scale.pack(fill=tk.X, pady=3)
        ttk.Label(cleaner_scale, text="Build / Teaching speed scale %").pack(side=tk.LEFT)
        ttk.Scale(
            cleaner_scale, from_=1, to=100, variable=self.velocity_percent,
            command=self.update_speed_label, length=220,
        ).pack(side=tk.LEFT, padx=7)
        self.cleaner_speed_label = ttk.Label(
            cleaner_scale, text=f"{self.velocity_percent.get():.1f}%"
        )
        self.cleaner_speed_label.pack(side=tk.LEFT)

        path_tests = self._create_toggle_section(
            sequence_page, "path_test", "Path Generation · Weave", expanded=False
        )
        weaving = ttk.Frame(path_tests)
        weaving.pack(fill=tk.X, pady=2)
        ttk.Button(
            weaving,
            text="Generate weave path",
            command=self.generate_weave,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(weaving, text="pattern").pack(side=tk.LEFT)
        ttk.Combobox(
            weaving,
            textvariable=self.weave_pattern,
            values=("sine", "crescent", "circle"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(weaving, text="base").pack(side=tk.LEFT)
        ttk.Combobox(
            weaving,
            textvariable=self.weave_base,
            values=("linear", "circle"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 8))
        for label, variable, start, end in (
            ("A mm (sine ±A / circle R)", self.weave_amplitude_mm, 0.1, 50),
            ("pitch mm/cycle", self.weave_pitch_mm, 0.1, 100),
        ):
            ttk.Label(weaving, text=label).pack(side=tk.LEFT)
            ttk.Spinbox(
                weaving,
                from_=start,
                to=end,
                textvariable=variable,
                width=6,
            ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(weaving, text="transverse axis").pack(side=tk.LEFT)
        ttk.Combobox(
            weaving,
            textvariable=self.weave_axis,
            values=(
                "tool_x",
                "tool_y",
                "tool_z",
                "world_x",
                "world_y",
                "world_z",
            ),
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(3, 8))
        self.weave_summary = ttk.Label(
            weaving,
            text="Apply after teaching a seam",
        )
        self.weave_summary.pack(side=tk.LEFT, padx=(8, 0))

        self._build_seam_correction_controls(welding_page)

        self._build_sequence_and_sensing_controls(sequence_page, shared_task_controls)

        self._build_fastech_controls(connection_page)

        self._build_cleaner_controls(cleaner_teaching)

        self._build_legacy_io_controls(connection_page)

        self._build_sequence_execution_controls(sequence_page)

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Editable Cartesian Action")
        self.root.geometry("1240x940")
        self._closing = False
        self._ui_queue = queue.SimpleQueue()
        self._latest_ui_updates = {}
        self._latest_ui_updates_lock = threading.Lock()
        # Start every GUI parameter (recipe I/V/material and motion
        # speed/lead/ARC timing) from whatever the last saved weld feedback
        # log actually ran, not a hard-coded fallback. A field only falls
        # back to its hard-coded default when the log has never recorded it.
        self._last_execution_settings = read_last_execution_settings(
            Path.home() / "ros2_ws" / "weld_feedback" / "latest_weld_feedback.log"
        )
        last_execution_motion = self._last_execution_settings.get("motion", {})
        self.points = []
        self.weave_source = []
        self.weave_base_paths = {"linear": [], "circle": [], "corrected": []}
        self.path_kind = "empty"
        self.execution_allowed = False
        self.robot_connected = {
            "left": False,
            "right": False,
            "head": False,
        }
        self.robot_power_busy = False
        self.robot_power_status = tk.StringVar(
            value="Arm power: use ACTIVATE BOTH before physical motion"
        )
        self.keyboard_jog_enabled = tk.BooleanVar(value=False)
        self.keyboard_jog_selection = tk.StringVar(value="XY")
        self.keyboard_jog_frame = tk.StringVar(value="World")
        self.keyboard_jog_linear_speed = tk.DoubleVar(value=5.0)
        self.keyboard_jog_angular_speed = tk.DoubleVar(value=3.0)
        self.keyboard_jog_status = tk.StringVar(
            value="Keyboard teaching locked"
        )
        self.keyboard_velocity_arm = None
        self.keyboard_velocity_switching = False
        self.keyboard_velocity_active_key = None
        self.keyboard_release_after_id = None
        self.keyboard_stop_generation = 0
        self.keyboard_ros_input_last_at = 0.0
        self.keyboard_ros_physical_key = None
        self.keyboard_ros_physical_mask = 0
        self.keyboard_ros_zero_seen = False
        self.keyboard_ros_dispatching = False
        self.keyboard_shortcut_active_keys = set()
        self.keyboard_shortcut_release_ids = {}
        self.keyboard_teaching_capture_in_progress = False
        self.fake_head_hardware = False
        self.plan_approved = False
        self.linear_tcp_endpoints = [None, None]
        self.initial_joint_state = None
        self.initial_plan_ready = False
        self.teaching_pose_name = tk.StringVar(
            value=TEACHING_POSES["robot_start"]
        )
        self.taught_robot_poses = {name: None for name in TEACHING_POSES}
        self.teaching_capture_provenance = {}
        self.pose_variables = {
            name: tk.StringVar(value="0.0") for name in self.POSE_FIELDS
        }
        self.radius_mm = tk.DoubleVar(value=20.0)
        self.circle_count = tk.IntVar(value=16)
        self.close_circle = tk.BooleanVar(value=True)
        self.circle_face_center = tk.BooleanVar(value=True)
        self.circle_axis = tk.StringVar(value="X")
        self.nudge_mm = tk.DoubleVar(value=5.0)
        self.velocity_percent = tk.DoubleVar(
            value=last_execution_motion.get("gui_velocity_percent", 20.0)
        )
        self.speed_mode = tk.StringVar(
            value=last_execution_motion.get("gui_speed_mode", "scale")
        )
        self.tcp_speed_mm_s = tk.DoubleVar(
            value=last_execution_motion.get("gui_tcp_speed_mm_s", 10.0)
        )
        self.interpolation_step_mm = tk.DoubleVar(value=5.0)
        self.linear_motion_profile = tk.BooleanVar(value=True)
        self.show_path = tk.BooleanVar(value=True)
        self.weave_amplitude_mm = tk.DoubleVar(
            value=(
                float(last_execution_motion.get("capping_width_mm", 6.0)) * 0.5
                if last_execution_motion.get("weld_weave_pattern") == "capping"
                else last_execution_motion.get("weld_weave_amplitude_mm", 3.0)
            )
        )
        self.weave_pitch_mm = tk.DoubleVar(
            value=last_execution_motion.get(
                "weld_weave_pitch_mm",
                last_execution_motion.get("capping_pitch_mm", 5.0),
            )
        )
        self.weave_left_dwell_s = tk.DoubleVar(
            value=last_execution_motion.get(
                "weld_weave_left_dwell_s",
                last_execution_motion.get("capping_left_dwell_s", 0.0),
            )
        )
        self.weave_right_dwell_s = tk.DoubleVar(
            value=last_execution_motion.get(
                "weld_weave_right_dwell_s",
                last_execution_motion.get("capping_right_dwell_s", 0.0),
            )
        )
        self.weave_axis = tk.StringVar(
            value=last_execution_motion.get("weld_weave_axis", "tool_y")
        )
        self.weave_base = tk.StringVar(value="linear")
        self.weave_pattern = tk.StringVar(
            value=(
                "sine" if last_execution_motion.get("weld_weave_pattern") == "capping"
                else last_execution_motion.get("weld_weave_pattern", "sine")
            )
        )
        self.weld_weave_enabled = tk.BooleanVar(
            value=last_execution_motion.get("weld_weave_enabled", False)
        )
        self.straight_reference = tk.StringVar(value="world")
        self.straight_axis = tk.StringVar(value="+X")
        self.straight_start_mode = tk.StringVar(value="Current TCP")
        self.straight_start_x = tk.DoubleVar(value=0.0)
        self.straight_start_y = tk.DoubleVar(value=0.0)
        self.straight_start_z = tk.DoubleVar(value=0.0)
        self.straight_distance_mm = tk.DoubleVar(value=150.0)
        self.straight_count = tk.IntVar(value=5)
        self.tcp_line_direction = tk.StringVar(value="TCP 1 → TCP 2")
        self.straight_roll_deg = tk.DoubleVar(value=0.0)
        self.straight_pitch_deg = tk.DoubleVar(value=0.0)
        self.straight_yaw_deg = tk.DoubleVar(value=0.0)
        self.straight_rotation_reference = tk.StringVar(value="tool")
        self.planning_group = tk.StringVar(value="right_manipulator")
        self.rbpodo_welder_ready = False
        self.latest_right_system_state = None
        self.hicomm_connected = False
        self.hicomm_client = None
        self.hicomm_source_ip = tk.StringVar(value="192.168.1.2")
        self.hicomm_welder_ip = tk.StringVar(value="192.168.1.10")
        self.hicomm_port = tk.IntVar(value=60000)
        self.hicomm_gas_enabled = tk.BooleanVar(value=False)
        # When enabled, D-WELD ARC SET/ON/OFF commands are simulated instead
        # of being sent to the welder over Hi-COMM, so a sequence can be run
        # to exercise motion only (no real welding output).
        self.fake_arc_enabled = tk.BooleanVar(value=False)
        self.work_cycle_repeats = tk.IntVar(value=1)
        self.hicomm_inching_direction = None
        self.keyboard_wire_active_key = None
        self.keyboard_wire_release_after_id = None
        self.inching_distance_lock = threading.Lock()
        self.inching_total_mm = 0.0
        self.inching_forward_mm = 0.0
        self.inching_reverse_mm = 0.0
        self.inching_last_status_time = None
        self.hicomm_feedback_last_log_time = 0.0
        self.hicomm_feedback_last_signature = None
        self.hicomm_feedback_log_period_s = 0.2
        self.hicomm_feedback_idle_log_period_s = 1.0
        self.weld_feedback_lock = threading.Lock()
        self.active_weld_feedback_session = None
        self.weld_motion_done_event = threading.Event()
        self.weld_motion_success = False
        # Sequence welding synchronization: ARC-OFF must never race ahead of
        # the ARC-ON establishment handshake.  These events are reset for
        # every generated weld slot before its parallel workers are started.
        self.weld_arc_established_event = threading.Event()
        self.weld_arc_on_done_event = threading.Event()
        self.weld_arc_on_success = False
        self.touch_sensing_enabled = tk.BooleanVar(value=False)
        self.corner_touch_target = tk.StringVar(value="start_floor")
        self.corner_touch_count = tk.IntVar(value=10)
        self.corner_touches = {name: None for name in CORNER_TOUCH_NAMES}
        # seam_axis is retained only for backward-compatible touch YAML / legacy
        # helpers.  New Fastech DI0 probing uses explicit probe directions below.
        self.seam_axis = tk.StringVar(value="X")
        self.wall_probe_axis = tk.StringVar(value="AUTO ⟂ taught seam (XY)")
        self.floor_probe_axis = tk.StringVar(value="World Z")
        self.seam_orientation_mode = tk.StringVar(
            value=WAIT_FIXED_TILT_ORIENTATION_MODE
        )
        self.weld_fixed_tilt_x_deg = tk.DoubleVar(
            value=last_execution_motion.get("weld_fixed_tilt_x_deg", 0.0)
        )
        self.weld_fixed_tilt_y_deg = tk.DoubleVar(
            value=last_execution_motion.get("weld_fixed_tilt_y_deg", -15.0)
        )
        self.weld_fixed_tilt_z_deg = tk.DoubleVar(
            value=last_execution_motion.get("weld_fixed_tilt_z_deg", 0.0)
        )
        self.reference_yaw_status = tk.StringVar(value="Reference yaw: --")
        self.sensed_yaw_status = tk.StringVar(value="Sensed yaw: --")
        self.delta_yaw_status = tk.StringVar(value="ΔYaw: --")
        self.reference_length_status = tk.StringVar(value="Length: --")
        self.quick_teaching_status = tk.StringVar(
            value="Auxiliary teaching: not captured"
        )
        self.wall_probe_sign = tk.StringVar(value="-")
        self.floor_probe_sign = tk.StringVar(value="-")
        self.touch_probe_distance_mm = tk.DoubleVar(value=25.0)
        self.touch_probe_speed_percent = tk.DoubleVar(value=5.0)
        self.touch_settle_seconds = tk.DoubleVar(value=0.7)
        self.weld_approach_mode = tk.StringVar(
            value=last_execution_motion.get("weld_approach_mode", "corner_geometry")
        )
        self.weld_safe_approach_mm = tk.DoubleVar(
            value=last_execution_motion.get("weld_safe_approach_mm", 30.0)
        )
        self.weld_pre_start_lead_mm = tk.DoubleVar(
            value=last_execution_motion.get("weld_pre_start_lead_mm", 10.0)
        )
        # Current qualified starting values. They are copied into a generated
        # weld scenario at Build time, so the generated sequence is immutable
        # even if the GUI is edited afterwards.
        # Prefer the value that actually ran in the loaded/latest feedback log.
        # Ten millimetres remains the fallback for a fresh installation.
        self.weld_lead_in_mm = tk.DoubleVar(
            value=last_execution_motion.get("weld_lead_in_mm", 10.0)
        )
        self.weld_lead_out_mm = tk.DoubleVar(
            value=last_execution_motion.get("weld_lead_out_mm", 5.0)
        )
        self.weld_arc_off_delay_ms = tk.DoubleVar(
            value=last_execution_motion.get("weld_arc_off_delay_ms", 500.0)
        )
        # Welding travel uses a physical TCP-speed target independent of the
        # global velocity-scale slider. This prevents seam length from changing
        # the cruise speed (e.g. a long 150 mm seam reaching a much higher
        # MoveIt scale plateau than a short 50 mm seam at the same percentage).
        self.weld_tcp_speed_mm_s = tk.DoubleVar(
            value=last_execution_motion.get("weld_tcp_speed_mm_s", 3.0)
        )
        self.seam_probe_touches = {
            name: None for name in CORNER_TOUCH_NAMES
        }
        self.seam_probe_starts = {
            name: None for name in CORNER_TOUCH_NAMES
        }
        self.seam_probe_stops = {
            name: None for name in CORNER_TOUCH_NAMES
        }
        self.raw_two_touch_seam = []
        self.corrected_two_touch_seam = []
        self.corrected_seam_geometry = None
        self.computed_seam_endpoints = {"start": None, "goal": None}
        self.computed_seam_wait_points = {"start": None, "goal": None}
        self.four_pass_folder = tk.StringVar(
            value=str(self._weld_feedback_directory() / "test_shimen_gth")
        )
        self.four_pass_status = tk.StringVar(
            value="Load 1.log..4.log · WAIT/START/GOAL WAIT/GOAL come from each log"
        )
        self.multipass_state.status = self.four_pass_status.get()
        self.four_pass_references = {}
        self.four_pass_loaded_folder = None
        self.four_pass_corrected = {}
        self.four_pass_output_folder = None
        self.four_pass_history = []
        self.selected_pass_number = tk.IntVar(value=1)
        self.multi_pass_registration = None
        self.pass_probe_touch_yaml_target = None
        self.seam_teaching_reference = None
        self.automatic_probe_kind = None
        self.seam_auto_running = False
        self.auto_seam_move_to_end_pose = tk.BooleanVar(value=False)
        self.seam_auto_move_to_end_requested = False
        self.seam_auto_stage_event = threading.Event()
        self.seam_auto_stage_success = False
        self.seam_auto_expected_kind = None
        self.seam_auto_returned_kinds = set()
        self.sequence_steps = []
        self.sequence_sleep_seconds = tk.DoubleVar(value=1.0)
        self.sequence_parallel_slot = tk.IntVar(value=1)
        self.sequence_duration_seconds = tk.DoubleVar(value=0.0)
        self.sequence_edit_velocity_percent = tk.DoubleVar(value=20.0)
        self.sequence_edit_tcp_speed_mm_s = tk.DoubleVar(value=0.0)
        self.sequence_edit_touch_guard = tk.BooleanVar(value=False)
        self.sequence_edit_continue_after_touch = tk.BooleanVar(value=False)
        self.sequence_head_joint1_deg = tk.DoubleVar(value=0.0)
        self.sequence_head_joint2_deg = tk.DoubleVar(value=0.0)
        self.sequence_running = False
        self.sequence_stop_requested = False
        self.last_action_phase = ""
        self.previous_control_box_io = None
        self.touch_input_states = {"left": None, "right": None}
        self.touch_input_rising_edges = {"left": 0, "right": 0}
        self.last_touch_pose = None
        self.motion_sections = {}
        self.control_box_io_labels = {}
        self.pending_do_ports = set()
        self.unlock_all_do_ports = tk.BooleanVar(value=False)
        self.fastech_ip = tk.StringVar(value="192.168.0.3")
        self.fastech_board_id = tk.IntVar(value=0)
        self.fastech_poll_rate_hz = 100.0
        self.fastech_connected = False
        self.fastech_connecting = False
        self.fastech_previous_state = None
        self.fastech_pending_outputs = set()
        self.fastech_io_labels = {}
        self.fastech_output_buttons = []
        self.latest_wide_sensing_result = None
        self.wide_sensing_segments = {}
        self.wide_sensing_segment_id = tk.StringVar(value="")
        self.wide_sensing_source_frame = tk.StringVar(value="helios_link")
        self.wide_sensing_reverse = tk.BooleanVar(value=False)
        self.wide_sensing_offset_x_mm = tk.DoubleVar(value=0.0)
        self.wide_sensing_offset_y_mm = tk.DoubleVar(value=0.0)
        self.wide_sensing_offset_z_mm = tk.DoubleVar(value=0.0)
        self.wide_sensing_status = tk.StringVar(
            value="Waiting for /wide_sensing/output/result"
        )
        # Reproduce the successful v5.2 Rainbow capture byte-for-byte by
        # default, unless the last saved weld feedback log recorded a
        # different recipe -- then start from exactly what last ran.
        weld_defaults = dict(DEFAULT_DIGITAL_WELD_SETTINGS)
        weld_defaults.update(self._last_execution_settings.get("settings", {}))
        # RX observation is diagnostic, not a transmitted crater command.
        # Always start by checking whether the panel reports native crater,
        # even when the previous saved run had this observation unchecked.
        weld_defaults["expect_native_crater"] = True
        self.weld_current_raw = tk.IntVar(value=weld_defaults["current_a"])
        self.weld_voltage_raw = tk.IntVar(
            value=weld_defaults["voltage_tenths"]
        )
        self.weld_material = tk.StringVar(value=weld_defaults["material"])
        self.weld_diameter_mm = tk.DoubleVar(
            value=weld_defaults["diameter_mm"]
        )
        self.weld_mode = tk.StringVar(value=weld_defaults["mode"])
        self.weld_gas = tk.StringVar(value=weld_defaults["gas"])
        self.weld_synergic = tk.BooleanVar(value=weld_defaults["synergic"])
        self.weld_correction = tk.DoubleVar(value=weld_defaults["correction"])
        self.weld_hot_start_enabled = tk.BooleanVar(
            value=weld_defaults["hot_start_enabled"]
        )
        self.weld_hot_start_percent = tk.DoubleVar(
            value=weld_defaults["hot_start_percent"]
        )
        self.weld_hot_start_hold_adjustment = tk.IntVar(
            value=weld_defaults["hot_start_hold_adjustment"]
        )
        self.weld_custom_hot_start_enabled = tk.BooleanVar(
            value=weld_defaults["custom_hot_start_enabled"]
        )
        self.weld_custom_hot_start_hold_s = tk.DoubleVar(
            value=weld_defaults["custom_hot_start_hold_s"]
        )
        self.weld_custom_hot_start_percent = tk.DoubleVar(
            value=weld_defaults["custom_hot_start_percent"]
        )
        self.weld_expect_native_crater = tk.BooleanVar(
            value=weld_defaults["expect_native_crater"]
        )
        self.weld_crater_panel_current_ref_a = tk.DoubleVar(
            value=weld_defaults["crater_panel_current_ref_a"]
        )
        self.weld_crater_panel_voltage_ref_v = tk.DoubleVar(
            value=weld_defaults["crater_panel_voltage_ref_v"]
        )
        self.weld_crater_panel_time_ref_s = tk.DoubleVar(
            value=weld_defaults["crater_panel_time_ref_s"]
        )
        self.weld_software_crater_enabled = tk.BooleanVar(value=weld_defaults["software_crater_enabled"])
        self.weld_software_crater_ratio_percent = tk.DoubleVar(value=weld_defaults["software_crater_ratio_percent"])
        self.weld_software_crater_voltage_v = tk.DoubleVar(value=weld_defaults["software_crater_voltage_v"])
        self.weld_software_crater_hold_s = tk.DoubleVar(value=weld_defaults["software_crater_hold_s"])
        self.weld_wire_consumable_alpha_mm = tk.DoubleVar(
            value=weld_defaults["wire_consumable_alpha_mm"]
        )
        self.robot_ips = {
            "left": "192.168.1.11",
            "right": "192.168.1.12",
        }

        self._build_operator_layout()

        self.node = WeldGuiNode(self)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
        )
        self.executor_thread.start()
        self._auto_load_teaching_states()
        loaded_settings = self._last_execution_settings.get("settings", {})
        loaded_motion = self._last_execution_settings.get("motion", {})
        if loaded_settings or loaded_motion:
            self.log(
                "Loaded GUI defaults from last weld feedback log · "
                f"recipe fields={sorted(loaded_settings)} · "
                f"motion fields={sorted(loaded_motion)}"
            )
        signal.signal(
            signal.SIGINT,
            lambda _signum, _frame: self.root.after(0, self.close),
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.check_ros)

    def post(self, callback, *args):
        self._ui_queue.put((callback, args))

    def post_latest(self, key, callback, *args):
        """Coalesce high-rate telemetry so Tk only renders the newest value."""
        with self._latest_ui_updates_lock:
            self._latest_ui_updates[key] = (callback, args)

    def _drain_ui_queue(self):
        # Never monopolize Tk's event loop.  ROS callbacks can produce work
        # faster than widgets can render it; an unlimited drain starves mouse,
        # scrolling, repainting, and the Hi-COMM cyclic Python thread.
        deadline = time.monotonic() + 0.004
        processed = 0
        while processed < 64 and time.monotonic() < deadline:
            try:
                callback, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            callback(*args)
            processed += 1
        with self._latest_ui_updates_lock:
            latest = tuple(self._latest_ui_updates.values())
            self._latest_ui_updates.clear()
        for callback, args in latest:
            callback(*args)

    def _update_scroll_region(self, _event=None):
        self.content_canvas.configure(
            scrollregion=self.content_canvas.bbox("all")
        )

    def _resize_scroll_content(self, event):
        self.content_canvas.itemconfigure(
            self.content_window,
            width=event.width,
        )

    def _scroll_content(self, event):
        delta = getattr(event, "delta", 0)
        button = getattr(event, "num", None)
        if delta > 0 or button == 4:
            direction = -1
        elif delta < 0 or button == 5:
            direction = 1
        else:
            return
        self.content_canvas.yview_scroll(direction * 2, "units")

    def _scroll_value_control(self, event):
        self._scroll_content(event)
        return "break"

    def _set_welder_test_controls(self, enabled):
        active = bool(enabled and self.hicomm_connected)
        state = tk.NORMAL if active else tk.DISABLED
        for widget in (
            self.hicomm_forward_button,
            self.hicomm_reverse_button,
            self.hicomm_gas_check,
        ):
            widget.configure(state=state)
        self.hicomm_arc_on_button.configure(
            state=(
                tk.NORMAL
                if active
                else tk.DISABLED
            )
        )

    def request_hicomm_inching(self, direction, active):
        client = self.hicomm_client
        if client is None or not client.connected:
            return False
        mask = BIT_FORWARD if direction == "forward" else BIT_REVERSE
        try:
            if active:
                client.allow_outputs()
                opposite = BIT_REVERSE if mask == BIT_FORWARD else BIT_FORWARD
                client.set_command_bit(opposite, False)
                client.set_command_bit(mask, True)
                self.hicomm_inching_direction = direction
            else:
                client.set_command_bit(mask, False)
                if self.hicomm_inching_direction == direction:
                    self.hicomm_inching_direction = None
            state = "ON" if active else "OFF"
            self.hicomm_test_status.configure(
                text=f"{direction} inch {state}"
            )
            self.log(f"Hi-COMM {direction} inch {state}")
            return True
        except Exception as error:
            client.clear_outputs()
            self.hicomm_inching_direction = None
            self.error(f"Hi-COMM inching failed: {error}")
            return False

    def request_hicomm_gas(self):
        enabled = bool(self.hicomm_gas_enabled.get())
        client = self.hicomm_client
        if client is None or not client.connected:
            self.hicomm_gas_enabled.set(False)
            return
        try:
            if enabled:
                client.allow_outputs()
            client.set_command_bit(BIT_GAS, enabled)
            self.hicomm_test_status.configure(
                text=f"gas {'ON' if enabled else 'OFF'}"
            )
            self.log(f"Hi-COMM gas {'ON' if enabled else 'OFF'}")
        except Exception as error:
            client.clear_outputs()
            self.hicomm_gas_enabled.set(False)
            self.error(f"Hi-COMM gas test failed: {error}")

    def clear_hicomm_test_outputs(self):
        self._stop_keyboard_wire()
        client = self.hicomm_client
        if client is not None:
            client.clear_outputs()
        self.hicomm_inching_direction = None
        self.hicomm_gas_enabled.set(False)
        if hasattr(self, "hicomm_arc_on_button"):
            self.hicomm_arc_on_button.configure(state=tk.DISABLED)
        if hasattr(self, "hicomm_test_status"):
            self.hicomm_test_status.configure(text="ALL OUTPUTS OFF")

    def connect_hicomm(self):
        if self.hicomm_client is not None and self.hicomm_client.connected:
            return
        try:
            client = HiCommWelderClient(
                self.hicomm_source_ip.get().strip(),
                self.hicomm_welder_ip.get().strip(),
                int(self.hicomm_port.get()),
                connection_callback=lambda connected, detail: self.post(
                    self.hicomm_connection_changed, connected, detail
                ),
                status_callback=self._hicomm_status_received,
                log_callback=lambda message: self.post(self.log, message),
                tx_frame_callback=self._record_hicomm_tx_frame,
            )
            self.hicomm_client = client
            self.hicomm_connect_button.configure(state=tk.DISABLED)
            self.hicomm_weld_status.configure(text="CONNECTING… · ARC OFF")
            client.start()
        except (ValueError, OSError, tk.TclError) as error:
            self.hicomm_connect_button.configure(state=tk.NORMAL)
            self.error(f"Hi-COMM connection setup failed: {error}")

    def disconnect_hicomm(self):
        client = self.hicomm_client
        if client is not None:
            self._finish_weld_feedback_record(
                "disconnected", client.latest_status()
            )
            self.clear_hicomm_test_outputs()
            threading.Thread(target=client.stop, daemon=True).start()

    def hicomm_connection_changed(self, connected, detail):
        self.hicomm_connected = bool(connected)
        self.rbpodo_welder_ready = self.hicomm_connected
        self.hicomm_feedback_last_log_time = 0.0
        self.hicomm_feedback_last_signature = None
        retrying = not connected and detail.startswith("retrying in")
        self.hicomm_connect_button.configure(
            state=tk.DISABLED if connected or retrying else tk.NORMAL
        )
        self.hicomm_disconnect_button.configure(
            state=tk.NORMAL if connected or retrying else tk.DISABLED
        )
        if not connected:
            if self.hicomm_client is not None:
                self._finish_weld_feedback_record(
                    "connection lost", self.hicomm_client.latest_status()
                )
            self.clear_hicomm_test_outputs()
        self._set_welder_test_controls(
            connected
        )
        self.welder_connection_label.configure(
            text=f"HICOMM WELDER: {'O' if connected else 'X'}",
            bg="#e6f4ea" if connected else "#fce8e6",
            fg="#137333" if connected else "#b3261e",
        )
        self.hicomm_weld_status.configure(
            text=(
                "CONNECTED"
                if connected
                else ("RETRYING / 200 ms" if retrying else "DISCONNECTED")
            )
            + " · ARC OFF"
        )
        if not connected:
            self.hicomm_rx_bit_status.configure(
                text=(
                    "RX Byte0 · b5 WCR=? · b4 STICK=? · "
                    "b3 GAS CHECK=? · b0 TORCH=?"
                ),
                foreground="#5f6368",
            )
        if not retrying:
            self.log(
                f"Hi-COMM {'connected' if connected else 'disconnected'} · "
                f"{detail}"
            )

    def hicomm_status_changed(self, status):
        arc_on = bool(status["arc_ack"])
        arc_established = bool(status.get("arc_established"))
        error_code = int(status["welder_error"])
        self.hicomm_weld_status.configure(
            text=(
                f"ARC={'ESTABLISHED' if arc_established else ('ON' if arc_on else 'OFF')} · "
                f"{status.get('sequence_stage', 'unknown')} · "
                f"FB {status['feedback_current_a']}A/"
                f"{status['feedback_voltage_v']:.1f}V · ERR={error_code}"
            )
        )
        self.hicomm_rx_bit_status.configure(
            text=(
                "RX Byte0 · "
                f"b5 WCR={int(bool(status['wcr_detected']))} · "
                f"b4 STICK={int(bool(status['stick_ack']))} · "
                f"b3 GAS CHECK={int(bool(status['gas_ack']))} · "
                f"b0 TORCH={int(bool(status['arc_ack']))}"
            ),
            foreground=(
                "#b3261e"
                if status["torch_collision"] or error_code
                else "#137333"
            ),
        )
        acknowledgements = []
        for key, name in (
            ("wcr_detected", "WCR"),
            ("stick_ack", "STICK"),
            ("forward_ack", "FWD"),
            ("reverse_ack", "REV"),
            ("gas_ack", "GAS"),
            ("arc_ack", "ARC"),
        ):
            if status[key]:
                acknowledgements.append(name)
        if self.hicomm_connected:
            total_mm, forward_mm, reverse_mm = self._inching_distance_snapshot()
            self.hicomm_test_status.configure(
                text=(
                    "RX ACK="
                    + (",".join(acknowledgements) if acknowledgements else "OFF")
                    + f" · WFS={status['wire_feed_m_min']:.1f} m/min"
                    + f" · inch={total_mm:+.1f} mm "
                    + f"(F {forward_mm:.1f}/R {reverse_mm:.1f})"
                    + f" · ERR={error_code}"
                )
            )

    _weld_status_snapshot = staticmethod(weld_status_snapshot)

    def _teaching_snapshot_document(self):
        """Every currently taught robot pose, in the same shape the per-pose
        teaching YAML files use, so a weld feedback log can be replayed with
        ``load_teaching_and_touch_from_log``."""
        poses = {}
        for pose_name in TEACHING_POSES:
            stored = self.taught_robot_poses.get(pose_name)
            if stored is None:
                continue
            group, joint_names, positions, tcp = stored
            tcp_condition = (
                self._pose_execution_conditions(tcp) if tcp is not None else None
            )
            if tcp_condition is None:
                continue
            poses[pose_name] = {
                "planning_group": group,
                "joint_state": {
                    "names": list(joint_names),
                    "positions_rad": [float(value) for value in positions],
                },
                "tcp_pose_world": tcp_condition,
            }
            provenance = self.teaching_capture_provenance.get(pose_name)
            if provenance:
                poses[pose_name]["capture_provenance"] = copy.deepcopy(provenance)
        return poses

    def _touch_snapshot_document(self):
        """Every currently captured seam probe touch point, keyed by
        ``CORNER_TOUCH_NAMES``."""
        touches = {}
        for name in CORNER_TOUCH_NAMES:
            pose = self.seam_probe_touches.get(name)
            if pose is None:
                continue
            condition = self._pose_execution_conditions(pose)
            if condition is not None:
                touches[name] = condition
        return touches

    def _begin_weld_feedback_record(self, settings, execution_conditions=None):
        return _weld_feedback_for(self).begin(settings, execution_conditions)

    def _record_hicomm_tx_frame(self, frame, unix_time, monotonic):
        return _weld_feedback_for(self).record_tx_frame(frame, unix_time, monotonic)

    def _mark_weld_motion_timing(self, event):
        return _weld_feedback_for(self).mark_weld_motion_timing(event)

    def _record_weld_feedback_sample(self, status):
        return _weld_feedback_for(self).record_feedback_sample(status)

    def record_weld_tcp_sample(
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
        return _weld_feedback_for(self).record_tcp_sample(pose, progress=progress, waypoint_index=waypoint_index, phase=phase, tf_stamp_s=tf_stamp_s, along_mm=along_mm, remaining_mm=remaining_mm, cross_track_mm=cross_track_mm)

    def _record_actual_tcp_until_motion_done(self, step):
        return _weld_feedback_for(self).record_actual_tcp_until_motion_done(step)

    def _latest_weld_tcp_state(self):
        return _weld_feedback_for(self).latest_tcp_state()

    def _mark_arc_off_control(self, **values):
        return _weld_feedback_for(self).mark_arc_off_control(**values)

    def _pending_weld_final_status(self):
        return _weld_feedback_for(self).pending_final_status()

    def _weld_feedback_directory(self):
        return Path.home() / "ros2_ws" / "weld_feedback"

    def _latest_weld_feedback_path(self):
        return self._weld_feedback_directory() / "latest_weld_feedback.log"



    def load_teaching_and_touch_from_log(self):
        """Restore taught poses -- and, after confirmation, touch points --
        from a previously saved weld feedback log's embedded snapshot.

        Every weld feedback log now embeds the taught robot poses and seam
        probe touch points that were active for that run (see
        ``_teaching_snapshot_document``/``_touch_snapshot_document``). This
        lets an old log be replayed to reproduce exactly what was on screen
        for that weld, e.g. while debugging why a specific run behaved
        differently.
        """
        default_dir = self._weld_feedback_directory()
        path = filedialog.askopenfilename(
            title="Load teaching/touch from weld feedback log",
            initialdir=(
                str(default_dir) if default_dir.is_dir() else str(Path.home())
            ),
            filetypes=(("Weld feedback log", "*.log"), ("All files", "*.*")),
        )
        if not path:
            return
        execution_defaults = read_last_execution_settings(path).get("motion", {})
        applied_defaults, invalid_defaults = (
            self._apply_loaded_weld_motion_defaults(execution_defaults)
        )
        teaching_raw, touch_raw = read_teaching_and_touch_snapshot(path)
        if not teaching_raw and not touch_raw and not applied_defaults:
            self.error(
                f"No teaching/touch snapshot or reusable weld motion settings "
                f"found in {Path(path).name}"
            )
            return

        planning_group = self.planning_group.get()
        applied_poses = []
        skipped = []
        for pose_name, entry in teaching_raw.items():
            if pose_name not in TEACHING_POSES:
                continue
            try:
                group, joint_names, positions, tcp = parse_teaching_snapshot_entry(
                    pose_name, entry
                )
            except ValueError as error:
                skipped.append(f"{pose_name} ({error})")
                continue
            if group != planning_group:
                skipped.append(
                    f"{pose_name} (arm {group}, selected {planning_group})"
                )
                continue
            self.taught_robot_poses[pose_name] = (
                group, tuple(joint_names), tuple(positions), copy.deepcopy(tcp)
            )
            provenance = entry.get("capture_provenance")
            if isinstance(provenance, dict):
                self.teaching_capture_provenance[pose_name] = copy.deepcopy(provenance)
            else:
                self.teaching_capture_provenance.pop(pose_name, None)
            applied_poses.append(TEACHING_POSES[pose_name])
        if applied_poses:
            self._verify_loaded_teaching_poses_async({
                name: copy.deepcopy(self.taught_robot_poses[name])
                for name in teaching_raw
                if name in TEACHING_POSES and self.taught_robot_poses[name] is not None
            })

        # Touch points are physical contact points on the real workpiece --
        # unlike joint teaching, they cannot be trusted blindly since the
        # fixture may have moved since the log was written.  Apply them only
        # after an explicit confirmation.
        applied_touches = []
        if touch_raw:
            parsed_touches = {}
            for name, entry in touch_raw.items():
                if name not in CORNER_TOUCH_NAMES:
                    continue
                try:
                    parsed_touches[name] = _pose_from_yaml_dict(
                        entry, f"{name} touch"
                    )
                except ValueError as error:
                    skipped.append(f"{name} touch ({error})")
            if parsed_touches and messagebox.askyesno(
                "Restore touch points",
                f"{len(parsed_touches)} seam probe touch point(s) were "
                "captured during that logged run. These are physical "
                "contact points on the real workpiece and have NOT been "
                "re-probed just now -- the fixture may have moved since "
                "then.\n\nApply them anyway?",
                parent=self.root,
            ):
                for name, pose in parsed_touches.items():
                    self.seam_probe_touches[name] = pose
                    self.seam_probe_starts[name] = None
                    self.seam_probe_stops[name] = None
                applied_touches = list(parsed_touches)

        selected_pose = self._selected_teaching_pose_name()
        self.teaching_pose_name.set(TEACHING_POSES[selected_pose])
        self.teaching_pose_changed()

        summary = (
            f"Loaded from {Path(path).name}: "
            f"{len(applied_poses)} teaching pose(s)"
            + (
                f", {len(applied_touches)} touch point(s)"
                if applied_touches
                else ""
            )
            + (
                f", defaults={', '.join(applied_defaults)}"
                if applied_defaults else ""
            )
        )
        skipped.extend(invalid_defaults)
        if skipped:
            summary += f" · skipped: {', '.join(skipped)}"
        self.log(summary)

    def _apply_loaded_weld_motion_defaults(self, motion):
        """Apply persisted seam-motion values to the next Build defaults."""
        specifications = (
            ("weld_fixed_tilt_x_deg", self.weld_fixed_tilt_x_deg, -180.0, 180.0),
            ("weld_fixed_tilt_y_deg", self.weld_fixed_tilt_y_deg, -180.0, 180.0),
            ("weld_fixed_tilt_z_deg", self.weld_fixed_tilt_z_deg, -180.0, 180.0),
            ("weld_lead_in_mm", self.weld_lead_in_mm, 0.0, 100.0),
            ("weld_lead_out_mm", self.weld_lead_out_mm, 0.0, 100.0),
            (
                "weld_safe_approach_mm",
                self.weld_safe_approach_mm,
                1.0,
                200.0,
            ),
            (
                "weld_pre_start_lead_mm",
                self.weld_pre_start_lead_mm,
                0.0,
                100.0,
            ),
            ("weld_tcp_speed_mm_s", self.weld_tcp_speed_mm_s, 0.1, 100.0),
        )
        applied = []
        invalid = []
        for key, variable, minimum, maximum in specifications:
            if key not in motion:
                continue
            try:
                value = float(motion[key])
            except (TypeError, ValueError):
                invalid.append(f"{key} (not numeric)")
                continue
            if not math.isfinite(value) or not minimum <= value <= maximum:
                invalid.append(f"{key} (outside {minimum:g}..{maximum:g})")
                continue
            variable.set(value)
            applied.append(key)
        if any(key.startswith("weld_fixed_tilt_") for key in applied):
            self.seam_orientation_mode.set(WAIT_FIXED_TILT_ORIENTATION_MODE)
            self._update_seam_yaw_status()
        if motion.get("weld_weave_pattern") == "capping":
            motion["weld_weave_pattern"] = "sine"
            motion["weld_weave_amplitude_mm"] = float(motion.get("capping_width_mm", 6.0)) * 0.5
            motion["weld_weave_pitch_mm"] = motion.get("capping_pitch_mm", 5.0)
            motion["weld_weave_left_dwell_s"] = motion.get("capping_left_dwell_s", 0.0)
            motion["weld_weave_right_dwell_s"] = motion.get("capping_right_dwell_s", 0.0)
        for key, variable, allowed in (
            ("weld_approach_mode", self.weld_approach_mode,
             {"taught_wait", "corner_geometry"}),
            ("weld_weave_pattern", self.weave_pattern, {"sine", "crescent", "circle"}),
            (
                "weld_weave_axis",
                self.weave_axis,
                {"tool_x", "tool_y", "tool_z", "world_x", "world_y", "world_z"},
            ),
        ):
            if key not in motion:
                continue
            value = str(motion[key]).strip().lower()
            if value not in allowed:
                invalid.append(f"{key} (unsupported {value})")
                continue
            variable.set(value)
            applied.append(key)
        for key, variable, minimum, maximum in (
            ("weld_weave_amplitude_mm", self.weave_amplitude_mm, 0.1, 50.0),
            ("weld_weave_pitch_mm", self.weave_pitch_mm, 0.1, 100),
            ("weld_weave_left_dwell_s", self.weave_left_dwell_s, 0, 10),
            ("weld_weave_right_dwell_s", self.weave_right_dwell_s, 0, 10),
        ):
            if key not in motion:
                continue
            try:
                value = float(motion[key])
            except (TypeError, ValueError):
                invalid.append(f"{key} (not numeric)")
                continue
            if not math.isfinite(value) or not minimum <= value <= maximum:
                invalid.append(f"{key} (outside {minimum:g}..{maximum:g})")
                continue
            variable.set(int(value) if isinstance(variable, tk.IntVar) else value)
            applied.append(key)
        if "weld_weave_enabled" in motion:
            self.weld_weave_enabled.set(bool(motion["weld_weave_enabled"]))
            applied.append("weld_weave_enabled")
        return applied, invalid

    def _finish_weld_feedback_record(self, result, final_status=None):
        return _weld_feedback_for(self).finish(result, final_status)

    def _hicomm_status_received(self, status):
        """Integrate RX wire-feed speed before forwarding status to Tk."""
        timestamp = float(status.get("timestamp_monotonic", time.monotonic()))
        with self.inching_distance_lock:
            previous = self.inching_last_status_time
            self.inching_last_status_time = timestamp
            if previous is not None:
                dt = max(0.0, min(0.2, timestamp - previous))
                distance_mm = (
                    max(0.0, float(status.get("wire_feed_m_min", 0.0)))
                    * 1000.0 / 60.0 * dt
                )
                if status.get("forward_ack"):
                    self.inching_forward_mm += distance_mm
                    self.inching_total_mm += distance_mm
                elif status.get("reverse_ack"):
                    self.inching_reverse_mm += distance_mm
                    self.inching_total_mm -= distance_mm
        self._record_weld_feedback_sample(status)
        self._log_hicomm_feedback(status, timestamp)
        self.post_latest(
            "hicomm_status", self.hicomm_status_changed, status
        )

    def _log_hicomm_feedback(self, status, timestamp):
        """Continuously expose TX commands and decoded welder RX in ROS logs."""
        client = self.hicomm_client
        if client is None:
            return
        try:
            tx = client.snapshot()
        except Exception:
            return
        command = int(tx.command)
        signature = (
            command,
            tx.base_profile,
            int(status.get("raw0", 0)),
            int(status.get("output_state", -1)),
            int(status.get("welder_error", 0)),
            bool(status.get("db_unavailable")),
            bool(status.get("torch_collision")),
        )
        state_changed = signature != self.hicomm_feedback_last_signature
        active_feedback = bool(
            command
            or int(status.get("raw0", 0))
            or int(status.get("output_state", 0))
            or int(status.get("welder_error", 0))
            or status.get("db_unavailable")
            or status.get("torch_collision")
        )
        log_period = (
            self.hicomm_feedback_log_period_s
            if active_feedback
            else self.hicomm_feedback_idle_log_period_s
        )
        if (
            signature == self.hicomm_feedback_last_signature
            and timestamp - self.hicomm_feedback_last_log_time
            < log_period
        ):
            return
        self.hicomm_feedback_last_signature = signature
        self.hicomm_feedback_last_log_time = timestamp

        # if state_changed:
        #     tx_raw = build_request(tx)
        #     rx_raw = status.get("raw_frame", b"")
        #     self.node.get_logger().info(
        #         f"HICOMM TX RAW [{len(tx_raw)}B] · "
        #         f"{tx_raw.hex(' ').upper()}"
        #     )
            # if rx_raw:
            #     self.node.get_logger().info(
            #         f"HICOMM RX RAW [{len(rx_raw)}B] · "
            #         f"{bytes(rx_raw).hex(' ').upper()}"
            #     )

        def bit(value, mask):
            return int(bool(value & mask))

        # self.node.get_logger().info(
        #     "HICOMM FEEDBACK · "
        #     f"PROFILE={tx.base_profile} · TX=0x{command:02X} "
        #     f"ARC={bit(command, BIT_ARC)} GAS={bit(command, BIT_GAS)} "
        #     f"FWD={bit(command, BIT_FORWARD)} REV={bit(command, BIT_REVERSE)} "
        #     f"STICK={bit(command, BIT_STICK)} · "
        #     f"RX=0x{int(status.get('raw0', 0)):02X} "
        #     f"ARC={int(bool(status.get('arc_ack')))} "
        #     f"GAS={int(bool(status.get('gas_ack')))} "
        #     f"FWD={int(bool(status.get('forward_ack')))} "
        #     f"REV={int(bool(status.get('reverse_ack')))} "
        #     f"WCR={int(bool(status.get('wcr_detected')))} "
        #     f"STICK={int(bool(status.get('stick_ack')))} · "
        #     f"OUT={status.get('output_state_name', 'unknown')}"
        #     f"({int(status.get('output_state', -1))}) · "
        #     f"FB={int(status.get('feedback_current_a', 0))}A/"
        #     f"{float(status.get('feedback_voltage_v', 0.0)):.1f}V "
        #     f"WFS={float(status.get('wire_feed_m_min', 0.0)):.1f}m/min · "
        #     f"SET={int(status.get('set_current_a', 0))}A/"
        #     f"{float(status.get('set_voltage_v', 0.0)):.1f}V · "
        #     f"DB={int(bool(status.get('db_unavailable')))} "
        #     f"COLL={int(bool(status.get('torch_collision')))} "
        #     f"ERR={int(status.get('welder_error', 0))}"
        # )

    def _inching_distance_snapshot(self):
        with self.inching_distance_lock:
            return (
                self.inching_total_mm,
                self.inching_forward_mm,
                self.inching_reverse_mm,
            )


    def fake_arc_changed(self):
        enabled = self.fake_arc_enabled.get()
        self.log(
            "FAKE ARC enabled · sequence execution will run motion only, "
            "no D-WELD command will reach the welder"
            if enabled
            else "FAKE ARC disabled · D-WELD commands go to the welder again"
        )

    def request_digital_arc(self, enabled):
        if enabled:
            if self.planning_group.get() != "right_manipulator":
                self.error("Select the right arm first")
                return
            if not messagebox.askyesno(
                "Physical ARC ON",
                "Start welding with the current recipe? Confirm the cell is safe and the torch is ready.",
            ):
                return
        if self.hicomm_client is None or not self.hicomm_connected:
            self.error("Connect Hi-COMM first")
            return
        try:
            settings = self._digital_weld_settings()
        except ValueError as error:
            self.error(str(error))
            return
        if enabled:
            self.hicomm_client.allow_outputs()
            with self.weld_feedback_lock:
                self._weld_feedback_stopped = False
        execution_conditions = (
            {
                "mode": "manual_arc_button",
                "hicomm_source_ip": self.hicomm_source_ip.get().strip(),
                "hicomm_welder_ip": self.hicomm_welder_ip.get().strip(),
                "hicomm_port": int(self.hicomm_port.get()),
            }
            if enabled else None
        )
        threading.Thread(
            target=self._manual_digital_weld_worker,
            args=(enabled, settings, execution_conditions),
            daemon=True,
        ).start()

    def _manual_digital_weld_worker(
        self, enabled, settings, execution_conditions
    ):
        kind = "on" if enabled else "off"
        success, message = self._execute_hicomm_weld(
            kind,
            settings,
            execution_conditions,
        )
        self.post(
            self.log,
            f"Hi-COMM ARC {kind.upper()} · "
            f"{'OK' if success else 'FAILED'} · {message}",
        )

    def _digital_weld_settings(self):
        try:
            return validate_digital_weld_settings({
                "current_a": self.weld_current_raw.get(),
                "voltage_tenths": self.weld_voltage_raw.get(),
                "material": self.weld_material.get(),
                "diameter_mm": self.weld_diameter_mm.get(),
                "mode": self.weld_mode.get(),
                "gas": self.weld_gas.get(),
                "synergic": self.weld_synergic.get(),
                "correction": self.weld_correction.get(),
                "hot_start_enabled": self.weld_hot_start_enabled.get(),
                "hot_start_percent": self.weld_hot_start_percent.get(),
                "hot_start_hold_adjustment": (
                    self.weld_hot_start_hold_adjustment.get()
                ),
                "custom_hot_start_enabled": self.weld_custom_hot_start_enabled.get(),
                "custom_hot_start_hold_s": self.weld_custom_hot_start_hold_s.get(),
                "custom_hot_start_percent": self.weld_custom_hot_start_percent.get(),
                "expect_native_crater": self.weld_expect_native_crater.get(),
                "crater_panel_current_ref_a": self.weld_crater_panel_current_ref_a.get(),
                "crater_panel_voltage_ref_v": self.weld_crater_panel_voltage_ref_v.get(),
                "crater_panel_time_ref_s": self.weld_crater_panel_time_ref_s.get(),
                "software_crater_enabled": self.weld_software_crater_enabled.get(),
                "software_crater_ratio_percent": self.weld_software_crater_ratio_percent.get(),
                "software_crater_voltage_v": self.weld_software_crater_voltage_v.get(),
                "software_crater_hold_s": self.weld_software_crater_hold_s.get(),
                "wire_consumable_alpha_mm": (
                    self.weld_wire_consumable_alpha_mm.get()
                ),
            })
        except (ValueError, tk.TclError) as error:
            raise ValueError(
                f"digital weld settings are invalid: {error}"
            ) from error

    def _execute_fake_arc(self, kind):
        return _weld_controller_for(self).execute_fake_arc(kind)

    def _software_crater_record(self, **values):
        return _weld_feedback_for(self).record_software_crater(**values)

    def _custom_hot_start_record(self, **values):
        return _weld_feedback_for(self).record_custom_hot_start(**values)

    def _execute_custom_hot_start(self, step):
        return _weld_controller_for(self).execute_custom_hot_start(step)

    def _software_crater_restore(self, settings):
        return _weld_controller_for(self).software_crater_restore(settings)

    def _execute_software_crater(self, step):
        return _weld_controller_for(self).execute_software_crater(step)

    def _execute_hicomm_weld(
        self, kind, settings, execution_conditions=None, *, finalize_feedback=True,
        apply_crater=True,
    ):
        return _weld_controller_for(self).execute_hicomm_weld(kind, settings, execution_conditions, finalize_feedback=finalize_feedback, apply_crater=apply_crater)

    def _execute_triggered_arc_off(self, step):
        return _weld_controller_for(self).execute_triggered_arc_off(step)

    def capture_corner_touch_now(self):
        target = self.corner_touch_target.get()
        self.node.capture_touch_pose(
            self.planning_group.get(), f"manual corner capture:{target}"
        )

    def select_weld_wait_pose(self):
        self.teaching_pose_name.set(TEACHING_POSES["weld_wait"])
        self.teaching_pose_changed()
        if self.initial_joint_state is None:
            self.error("Capture or load Weld wait pose first")
            return
        self.plan_initial_state()

    def _invalidate_seam_correction_runtime(self, reason=None, clear_touches=True):
        """Invalidate sensed/corrected seam data that depends on teaching.

        Sequence rows are snapshots and are intentionally left alone; this only
        clears the live sensing/correction cache so a later Build cannot reuse
        geometry from an older teaching/correction session.
        """
        if clear_touches:
            for name in CORNER_TOUCH_NAMES:
                self.seam_probe_touches[name] = None
                self.seam_probe_starts[name] = None
                self.seam_probe_stops[name] = None
        self.raw_two_touch_seam = []
        self.corrected_two_touch_seam = []
        self.corrected_seam_geometry = None
        # Drop the corrected weave base with it, or a later "Generate weave"
        # would preview a seam whose geometry has already been invalidated.
        self.weave_base_paths["corrected"] = []
        self.computed_seam_endpoints = {"start": None, "goal": None}
        self.computed_seam_wait_points = {"start": None, "goal": None}
        self.seam_auto_returned_kinds.clear()

        # A displayed Fastech DI0 seam is also stale once its teaching changes.
        if str(self.path_kind).startswith("di8_four_touch"):
            self.path_kind = "empty"
            self.weave_source = []
            self.set_points([])
            self.node.publish_points([], self.show_path.get())

        if reason:
            self.log(f"Seam correction cache cleared · {reason}")

    def browse_four_pass_folder(self):
        folder = filedialog.askdirectory(
            title="Select 4-pass work folder (logs + pass_teaching YAML)",
            initialdir=self.four_pass_folder.get(),
            parent=self.root,
        )
        if folder:
            self.four_pass_folder.set(folder)
            self.four_pass_references = {}
            self.four_pass_loaded_folder = None
            self.four_pass_output_folder = None
            self.four_pass_corrected = {}
            self.four_pass_history = []
            self.multi_pass_registration = None
            self._set_four_pass_status("Folder changed · load four references")


    # Tk-only hooks used by application.multipass_controller at the exact
    # points where the multi-pass workflow previously touched widgets/dialogs.
    def _confirm_multi_pass_registration(self, number):
        return messagebox.askyesno(
            "Sequential multi-pass registration",
            f"Register Pass {number} START and GOAL?\n\n"
            "The robot uses this pass's corrected WAIT poses loaded from its log. "
            "Keyboard Teaching enables automatically after START WAIT; jog to the real "
            "START and press I. It will then move to GOAL WAIT; jog to the real "
            "GOAL and press J; save correction, then move to Weld end.\n"
            "START WAIT to GOAL WAIT stops on touch contact.\n\n"
            f"Pass {number}: save captured TCP1/2, keep WAIT. Transform later passes only. No arc or welding "
            "command will be sent.",
            parent=self.root,
        )

    def _confirm_corrected_pass_endpoint(self, number, endpoint):
        return messagebox.askyesno(
            "Verify corrected pass endpoint",
            f"Move to corrected Pass {number} {endpoint.upper()}?\n\n"
            "This is a robot motion only. ARC and welding outputs remain OFF.",
            parent=self.root,
        )

    def _set_keyboard_jog_controls_enabled(self, enabled):
        self._set_keyboard_jog_enable_state(tk.NORMAL if enabled else tk.DISABLED)

    def _focus_keyboard_teaching(self):
        self.root.focus_set()

    def load_four_pass_references(self):
        return _multipass_for(self).load_references()

    def _load_latest_sequential_four_pass_state(self, folder, references):
        return _multipass_for(self).load_latest_sequential_state(folder, references)

    def _set_four_pass_status(self, text):
        self.multipass_state.status = text
        self.four_pass_status.set(text)

    def _selected_pass(self):
        """Copy the operator's pass choice into the application working state."""
        return self.multipass_state.select(self.selected_pass_number.get())

    def run_four_pass_correction(self):
        return _multipass_for(self).run_correction()

    def _validate_four_pass_source_hashes(self):
        return _multipass_for(self).validate_source_hashes()

    def _multi_pass_translated_wait(self, number, endpoint):
        return _multipass_for(self).translated_wait(number, endpoint)

    def _run_multi_pass_tcp_move(
        self, target, label, velocity_scale, touch_guard=False
    ):
        return _multipass_for(self).run_tcp_move(target, label, velocity_scale, touch_guard)

    def stop_multi_pass_correction(self):
        """Invalidate registration and use the existing all-motion stop path."""
        self.emergency_stop_all()
        self._set_four_pass_status(
            "Multi-pass STOP requested · all robot motion stopping · restart correction to continue"
        )

    def _multi_pass_start_wait_worker(self, number, target):
        return _multipass_for(self).start_wait_worker(number, target)

    def _multi_pass_start_wait_finished(self, number, success, message):
        return _multipass_for(self).start_wait_finished(number, success, message)

    def _enable_multi_pass_keyboard_teaching(self, number, expected_key):
        return _multipass_for(self).enable_keyboard_teaching(number, expected_key)

    def _finish_multi_pass_keyboard_capture(self, key, captured, error):
        return _multipass_for(self).finish_keyboard_capture(key, captured, error)

    def _multi_pass_goal_wait_worker(self, number, target):
        return _multipass_for(self).goal_wait_worker(number, target)

    def _multi_pass_goal_wait_finished(self, number, success, message):
        return _multipass_for(self).goal_wait_finished(number, success, message)

    def _complete_multi_pass_registration(self):
        return _multipass_for(self).complete_registration()

    def _multi_pass_end_worker(self, session):
        return _multipass_for(self).end_worker(session)

    def _multi_pass_end_finished(self, session, success, message):
        return _multipass_for(self).end_finished(session, success, message)

    def _save_sequential_four_pass_state(self, corrected, session, transform):
        return _multipass_for(self).save_sequential_state(corrected, session, transform)

    def go_to_corrected_pass_endpoint(self, endpoint):
        return _multipass_for(self).go_to_corrected_endpoint(endpoint)

    def _go_to_corrected_pass_endpoint_worker(
        self, number, endpoint, target, velocity_scale
    ):
        return _multipass_for(self).corrected_endpoint_worker(number, endpoint, target, velocity_scale)

    def _go_to_corrected_pass_endpoint_finished(
        self, number, endpoint, success, message
    ):
        return _multipass_for(self).corrected_endpoint_finished(number, endpoint, success, message)

    def _selected_pass_teaching_path(self, number):
        return _multipass_for(self).selected_pass_teaching_path(number)

    def _load_saved_pass_teaching(self, number):
        return _multipass_for(self).load_saved_pass_teaching(number)

    def save_teaching_to_selected_pass(self):
        return _multipass_for(self).save_teaching_to_selected_pass()

    def apply_selected_pass_correction(self):
        return _multipass_for(self).apply_selected_pass_correction()

    def load_saved_teaching_for_selected_pass(self):
        return _multipass_for(self).load_saved_teaching_for_selected_pass()

    def _apply_selected_pass_teaching(self, use_saved_teaching):
        return _multipass_for(self).apply_selected_pass_teaching(use_saved_teaching)


    # Tk-only hooks used by application.seam_correction_controller at the
    # exact points where the workflow previously touched widgets/dialogs.
    def _set_corner_touch_status(self, text):
        self.corner_touch_status.configure(text=text)

    def _set_auto_seam_button_enabled(self, enabled):
        self.auto_seam_correction_button.configure(
            state=tk.NORMAL if enabled else tk.DISABLED
        )

    def _set_stop_auto_seam_button_enabled(self, enabled):
        self.stop_auto_seam_button.configure(
            state=tk.NORMAL if enabled else tk.DISABLED
        )

    def _show_teaching_pose(self, label):
        self.teaching_pose_name.set(label)
        self.teaching_pose_changed()

    def _confirm_automatic_seam_correction(self, fixed_tilt_mode):
        orientation_note = (
            "START/GOAL orientation = each WAIT orientation + fixed World XYZ "
            f"({float(self.weld_fixed_tilt_x_deg.get()):+.1f}°, "
            f"{float(self.weld_fixed_tilt_y_deg.get()):+.1f}°, "
            f"{float(self.weld_fixed_tilt_z_deg.get()):+.1f}°)\n"
            "Separate Weld START/GOAL teaching is not required."
            if fixed_tilt_mode
            else "START/GOAL orientation = existing Weld START/GOAL teaching."
        )
        return messagebox.askyesno(
            "Automatic Seam Correction",
            "Execute the complete four-probe correction?\n\n"
            "START wait → wall/base → GOAL wait → wall/base\n"
            "→ compute seam geometry → save START/GOAL YAML\n"
            + (
                "→ move to 7 · Weld end pose\n\n"
                if self.auto_seam_move_to_end_pose.get()
                else "→ remain at GOAL WAIT (no automatic END move)\n\n"
            )
            + (
                "Fixed-tilt mode keeps the WAIT-based weld orientations; "
                "sensed seam yaw is diagnostic only.\n"
                if fixed_tilt_mode else ""
            )
            + f"{orientation_note}\n\n"
            "Each Fastech DI4 edge stops the probe and returns to its probe start.\n"
            "The taught START/GOAL wait poses remain unchanged.",
        )

    def _confirm_touch_probe(
        self, kind, direction_label, sign, signed_direction, distance, speed
    ):
        return messagebox.askyesno(
            "Execute Fastech DI0 touch probe",
            f"{kind.replace('_', ' ').upper()}\n"
            f"Direction: {direction_label} · sign {sign}\n"
            f"World vector=({signed_direction[0]:+.3f}, "
            f"{signed_direction[1]:+.3f}, {signed_direction[2]:+.3f})\n"
            f"Travel up to {distance * 1000.0:.1f} mm at {speed:.1%}?\n\n"
            "Fastech DI0 will stop the motion and return to the current "
            "start pose.",
        )

    def run_automatic_seam_correction(self):
        return _seam_correction_for(self).run_automatic_correction()

    def stop_automatic_seam_correction(self):
        return _seam_correction_for(self).stop_automatic_correction()

    def auto_seam_stop_finished(self, success, message):
        self.auto_seam_correction_button.configure(state=tk.NORMAL)
        self.stop_auto_seam_button.configure(state=tk.DISABLED)
        if success:
            self.pipeline_result("AUTO SEAM STOPPED · robot stationary")
        else:
            self.error(f"STOP AUTO could not confirm safe idle: {message}")

    def show_computed_seam_in_rviz(self):
        if not self.raw_two_touch_seam or not self.corrected_two_touch_seam:
            self.error("Compute the four-touch seam first")
            return
        self.show_path.set(True)
        self.node.publish_seam_comparison(
            self.raw_two_touch_seam,
            self.corrected_two_touch_seam,
            True,
        )
        for endpoint in ("start", "goal"):
            self._publish_touch_geometry_if_ready(
                endpoint, self.computed_seam_endpoints.get(endpoint)
            )
        self.log(
            "Published computed seam to RViz /weld_path_markers · "
            "red=raw, translucent cyan=corrected"
        )

    def _publish_touch_geometry_if_ready(self, endpoint, seam_point=None):
        return _seam_correction_for(self).publish_touch_geometry_if_ready(endpoint, seam_point)

    def show_touch_geometry_in_rviz(self):
        published = [
            endpoint
            for endpoint in ("start", "goal")
            if self._publish_touch_geometry_if_ready(
                endpoint, self.computed_seam_endpoints.get(endpoint)
            )
        ]
        if not published:
            self.error("Capture a complete wall/floor touch pair first")
            return
        self.show_path.set(True)
        self.log(
            "Published Fastech DI0 touch geometry to RViz /weld_path_markers · "
            f"{', '.join(name.upper() for name in published)} · "
            "red=wall, blue=floor, green=seam, yellow=midpoint"
        )


    def _automatic_seam_correction_worker(self, workflow):
        return _seam_correction_for(self).automatic_correction_worker(workflow)

    def _wait_for_touch_release(self, timeout):
        return _seam_correction_for(self).wait_for_touch_release(timeout)

    def _launch_automatic_seam_stage(self, kind):
        return _seam_correction_for(self).launch_automatic_stage(kind)

    def _signal_auto_seam_stage(self, success, kind=None):
        return _seam_correction_for(self).signal_automatic_stage(success, kind)

    def _set_auto_seam_status(self, text):
        self.corner_touch_status.configure(text=text)
        self.pipeline_waiting(text)

    def _complete_automatic_seam_correction(self):
        return _seam_correction_for(self).complete_automatic_correction()

    def _automatic_seam_end_pose_worker(self, finish_step):
        return _seam_correction_for(self).automatic_end_pose_worker(finish_step)

    def _finish_automatic_seam_correction(self, success, message):
        return _seam_correction_for(self).finish_automatic_correction(success, message)

    def _resolve_probe_direction(self, surface, teaching_reference=None):
        return _seam_correction_for(self).resolve_probe_direction(surface, teaching_reference)

    def _seam_geometry_settings(self, require_teaching=True):
        return _seam_correction_for(self).seam_geometry_settings(require_teaching)

    def _compute_touch_corrected_seam_geometry(
        self,
        teaching_reference,
        wall_normal,
        floor_normal,
        wall_offset,
        floor_offset,
        *,
        log_debug=False,
    ):
        return _seam_correction_for(self).compute_touch_corrected_geometry(teaching_reference, wall_normal, floor_normal, wall_offset, floor_offset, log_debug=log_debug)

    def _log_corrected_seam_geometry(self, geometry):
        return _seam_correction_for(self).log_corrected_geometry(geometry)

    def start_automatic_touch_probe(self, kind, skip_confirmation=False):
        return _seam_correction_for(self).start_touch_probe(kind, skip_confirmation)

    def touch_probe_failed(self, message):
        return _seam_correction_for(self).touch_probe_failed(message)

    def _wait_fixed_tilt_mode_enabled(self):
        return self.seam_orientation_mode.get().strip() in (
            WAIT_FIXED_TILT_ORIENTATION_MODE,
            LEGACY_WAIT_FIXED_TILT_ORIENTATION_MODE,
        )

    def _wait_fixed_tilt_seam_reference(self, require_complete=False):
        """Return virtual START/GOAL references derived only from WAIT poses."""
        start_wait = self.taught_robot_poses.get("weld_start_wait")
        goal_wait = self.taught_robot_poses.get("weld_goal_wait")
        missing = []
        if start_wait is None:
            missing.append(TEACHING_POSES["weld_start_wait"])
        if goal_wait is None:
            missing.append(TEACHING_POSES["weld_goal_wait"])
        if missing:
            if require_complete:
                self.error(
                    "Fixed-tilt mode needs only START/GOAL WAIT teaching: "
                    + ", ".join(missing)
                )
            return None
        if start_wait[0] != goal_wait[0]:
            if require_complete:
                self.error("START/GOAL WAIT poses belong to different arms")
            return None
        try:
            start_pose, goal_pose = fixed_tilt_wait_reference_poses(
                start_wait[3],
                goal_wait[3],
                float(self.weld_fixed_tilt_y_deg.get()),
                tilt_x_deg=float(self.weld_fixed_tilt_x_deg.get()),
                tilt_z_deg=float(self.weld_fixed_tilt_z_deg.get()),
            )
        except (ValueError, tk.TclError) as error:
            if require_complete:
                self.error(str(error))
            return None
        return {
            "weld_start": (
                start_wait[0], tuple(start_wait[1]), tuple(start_wait[2]), start_pose
            ),
            "weld_end": (
                goal_wait[0], tuple(goal_wait[1]), tuple(goal_wait[2]), goal_pose
            ),
        }

    def _ensure_seam_teaching_reference(self, require_complete=False):
        """Return immutable TCP1/TCP2 seam references used for geometry/yaw."""
        if self._wait_fixed_tilt_mode_enabled():
            reference = self._wait_fixed_tilt_seam_reference(require_complete)
            if reference is not None:
                self.log(
                    "Seam reference ready from START/GOAL WAIT + fixed World XYZ "
                    f"({float(self.weld_fixed_tilt_x_deg.get()):+.1f}°, "
                    f"{float(self.weld_fixed_tilt_y_deg.get()):+.1f}°, "
                    f"{float(self.weld_fixed_tilt_z_deg.get()):+.1f}°)"
                )
            return reference
        names = ("weld_start", "weld_end")
        if self.seam_teaching_reference is None:
            self.seam_teaching_reference = {}
        for name in names:
            if (
                name not in self.seam_teaching_reference
                and self.taught_robot_poses.get(name) is not None
            ):
                self.seam_teaching_reference[name] = copy.deepcopy(
                    self.taught_robot_poses[name]
                )
        missing = [
            TEACHING_POSES[name]
            for name in names
            if name not in self.seam_teaching_reference
        ]
        if missing and require_complete:
            self.error(
                "Capture/load seam teaching references first: "
                + ", ".join(missing)
            )
            return None
        if not missing:
            self.log(
                "Seam reference ready for yaw correction · TCP1 START / TCP2 GOAL"
            )
        return self.seam_teaching_reference

    def compute_seam_endpoint(self, endpoint, update_wait_joints=True):
        return _seam_correction_for(self).compute_seam_endpoint(endpoint, update_wait_joints)

    def apply_corrected_tcp_joint_state(
        self,
        endpoint,
        teaching_name,
        planning_group,
        joint_names,
        positions,
        tcp,
    ):
        try:
            save_initial_state_yaml(
                self._initial_state_yaml_path(planning_group, teaching_name),
                planning_group,
                joint_names,
                positions,
                tcp,
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.corrected_tcp_joint_state_failed(
                endpoint, teaching_name, str(error)
            )
            return
        self.taught_robot_poses[teaching_name] = (
            planning_group,
            tuple(joint_names),
            tuple(positions),
            copy.deepcopy(tcp),
        )
        self.teaching_pose_name.set(TEACHING_POSES[teaching_name])
        self.teaching_pose_changed()
        values = self._pose_values(tcp)
        self.pipeline_result(
            f"{endpoint.upper()} TWO-POSE APPLIED · "
            f"{TEACHING_POSES[teaching_name]} TCP="
            f"({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}) · "
            "MoveIt joint state resolved and YAML saved · computation complete"
        )

    def corrected_tcp_joint_state_failed(self, endpoint, teaching_name, message):
        self.error(
            f"{str(endpoint).upper()} TCP was computed, but "
            f"{TEACHING_POSES.get(teaching_name, teaching_name)} IK update "
            f"failed: {message}"
        )

    def endpoint_is_sensed(self, endpoint):
        """True when both surfaces of one seam endpoint have been touched."""
        return all(
            self.seam_probe_touches.get(f"{endpoint}_{surface}") is not None
            for surface in ("wall", "floor")
        )

    def sensed_weave_transverse_vector(self):
        """The weave direction the executed weld will use, or None.

        Preview and execution must agree on this.  ``e_w`` is derived from the
        sensed wall and floor planes and lies in their bisecting plane, which
        on a fillet joint is **45 degrees** away from any generic tool/world
        axis: at a 2 mm amplitude that puts the weave peaks 3.7 mm from where a
        generic-axis preview draws them.  Deciding it in one place is what
        keeps "what the operator approved in RViz" and "what the robot runs"
        the same path.

        Returns None when the seam has not been touch-corrected, which is the
        signal to fall back to the operator's tool/world axis selector.
        """
        if self.corrected_seam_geometry is None:
            return None
        if not (self.endpoint_is_sensed("start")
                or self.endpoint_is_sensed("goal")):
            return None
        return self.corrected_seam_geometry.e_w

    def build_sensed_weld_sequence(
        self, *, teaching_poses=None, force_unsensed=False,
        weave_override=None, append=True,
    ):
        """Append a weld workflow. START/GOAL each use touch-sensed geometry
        when wall+floor touches are available, otherwise the plain taught
        weld_start/weld_end pose -- touch probing is optional, not required.

        Step generation lives in ``application.weld_sequence_builder``; this
        method snapshots Tk state, keeps the operator-visible side effects in
        their original order, and updates the SequenceModel.
        """
        poses = self.taught_robot_poses if teaching_poses is None else teaching_poses
        start_is_sensed = not force_unsensed and self.endpoint_is_sensed("start")
        if start_is_sensed:
            # Never trust a cached START endpoint here.  Build is a snapshot of
            # the *current* touch pair + current teaching, so recompute START
            # exactly like GOAL on every build.
            if self.compute_seam_endpoint("start", update_wait_joints=False) is None:
                return

        goal_is_sensed = not force_unsensed and self.endpoint_is_sensed("goal")
        if goal_is_sensed and self.compute_seam_endpoint(
            "goal", update_wait_joints=False
        ) is None:
            return

        try:
            validate_required_weld_poses(poses)
        except ValueError as error:
            self.error(str(error))
            return
        goal_wait_data = poses.get("weld_goal_wait")
        finish_data = poses.get("weld_finish")
        try:
            # Freeze one recipe snapshot at Build time. ARC ON and the paired
            # ARC OFF both carry this same snapshot so selecting/editing either
            # step never falls back to unrelated defaults. ARC OFF does not
            # retransmit I/V, but retaining the snapshot also preserves post-gas
            # timing and makes the generated scenario self-describing.
            settings = copy.deepcopy(self._digital_weld_settings())
            endpoints = resolve_weld_endpoints(
                poses,
                start_is_sensed=start_is_sensed,
                goal_is_sensed=goal_is_sensed,
                sensed_endpoints=self.computed_seam_endpoints,
            )
            count = int(self.corner_touch_count.get())
            lead_in_mm = float(self.weld_lead_in_mm.get())
            lead_out_mm = float(self.weld_lead_out_mm.get())
            safe_approach_mm = float(self.weld_safe_approach_mm.get())
            pre_start_lead_mm = float(self.weld_pre_start_lead_mm.get())
            arc_off_delay_ms = float(self.weld_arc_off_delay_ms.get())
            weld_tcp_speed_mm_s = float(self.weld_tcp_speed_mm_s.get())
            weave_enabled = (bool(self.weld_weave_enabled.get())
                             if weave_override is None else bool(weave_override["enabled"]))
            weave_pattern = self.weave_pattern.get().strip().lower()
            weave_amplitude_mm = float(self.weave_amplitude_mm.get())
            weave_pitch_mm = float(self.weave_pitch_mm.get())
            weave_left_dwell_s = float(self.weave_left_dwell_s.get())
            weave_right_dwell_s = float(self.weave_right_dwell_s.get())
            weave_axis = (self.weave_axis.get().strip().lower()
                          if weave_override is None else weave_override["transverse_axis"])
            approach_mode = "taught_wait" if force_unsensed else self.weld_approach_mode.get()
            scenario = WeldScenarioInput(
                endpoints=endpoints,
                goal_wait=goal_wait_data,
                finish=finish_data,
                settings=settings,
                corner_touch_count=count,
                lead_in_mm=lead_in_mm,
                lead_out_mm=lead_out_mm,
                safe_approach_mm=safe_approach_mm,
                pre_start_lead_mm=pre_start_lead_mm,
                arc_off_delay_ms=arc_off_delay_ms,
                weld_tcp_speed_mm_s=weld_tcp_speed_mm_s,
                weave_enabled=weave_enabled,
                weave_pattern=weave_pattern,
                weave_amplitude_mm=weave_amplitude_mm,
                weave_pitch_mm=weave_pitch_mm,
                weave_left_dwell_s=weave_left_dwell_s,
                weave_right_dwell_s=weave_right_dwell_s,
                weave_axis=weave_axis,
                approach_mode=approach_mode,
                touch_io_backend=FASTECH_TOUCH_BACKEND,
                touch_output_port=FASTECH_TOUCH_OUTPUT_PORT,
                seam_geometry=self.corrected_seam_geometry,
                weave_transverse_vector=(
                    None if force_unsensed or not weave_enabled
                    else self.sensed_weave_transverse_vector()
                ),
                start_wait_tcp_override=self.computed_seam_wait_points.get("start"),
            )
            approach = plan_weld_approach(scenario)
            safe_approach = approach.safe_approach
            if safe_approach is not None:
                start = endpoints.start
                approach_lead = approach.approach_lead
                self.corrected_seam_geometry.safe_start = copy.deepcopy(
                    safe_approach
                )
                self.corrected_seam_geometry.lead_start = copy.deepcopy(
                    approach_lead
                )
                fmt = lambda pose: "(" + ", ".join(
                    f"{getattr(pose.position, axis):+.6f}"
                    for axis in ("x", "y", "z")
                ) + ")"
                fmt_vector = lambda values: "(" + ", ".join(
                    f"{float(value):+.6f}" for value in values
                ) + ")"
                self.log(
                    "SAFE WELD APPROACH DEBUG · "
                    f"P_start={fmt(start)} · "
                    f"d_real={fmt_vector(self.corrected_seam_geometry.d_real)} · "
                    f"e_a={fmt_vector(self.corrected_seam_geometry.e_a)} · "
                    f"safe={safe_approach_mm:.1f} mm · "
                    f"pre-start lead={pre_start_lead_mm:.1f} mm · "
                    f"P_safe={fmt(safe_approach)} · "
                    f"P_lead={fmt(approach_lead)} · "
                    f"align(approach,-e_a)={approach.approach_alignment:.9f} · "
                    f"align(lead,d_real)={approach.lead_alignment:.9f}"
                )
            path = plan_weld_path(scenario, approach)
            geometry_weave_direction = path.weave_direction
            if geometry_weave_direction is not None:
                self.log(
                    "Touch-corrected weave uses geometry-derived e_w="
                    f"({geometry_weave_direction[0]:+.6f}, "
                    f"{geometry_weave_direction[1]:+.6f}, "
                    f"{geometry_weave_direction[2]:+.6f})"
                )
            if append:
                self.node.publish_points(path.preview, self.show_path.get())
            base_slot = (next_sequential_slot(
                self.sequence_steps,
                int(self.sequence_parallel_slot.get()),
            ) if append else 1)
            slots = allocate_weld_scenario_slots(
                base_slot,
                has_safe_approach=safe_approach is not None,
                has_lead_in=approach.has_lead_in,
                settings=settings,
            )
            final_slot = slots.final
            scenario_id = new_weld_scenario_id()
            motion = WeldStepMotionInput(
                velocity_percent=self.velocity_percent.get(),
                tcp_speed_m_s=self._selected_tcp_speed_m_s(),
                interpolation_step_mm=float(self.interpolation_step_mm.get()),
                seam_orientation_mode=self.seam_orientation_mode.get(),
                fixed_world_x_tilt_deg=float(self.weld_fixed_tilt_x_deg.get()),
                fixed_world_y_tilt_deg=float(self.weld_fixed_tilt_y_deg.get()),
                fixed_world_z_tilt_deg=float(self.weld_fixed_tilt_z_deg.get()),
            )
            steps = build_weld_scenario_steps(
                scenario, approach, path, slots, motion, scenario_id
            )
        except (ValueError, TypeError, tk.TclError) as error:
            self.error(f"Cannot build sensed weld sequence: {error}")
            return
        if append:
            self.sequence_model.extend(steps)
            self.refresh_sequence_table(select_last=True)
        start_source = endpoints.start_source
        goal_source = endpoints.goal_source
        has_lead_in = approach.has_lead_in
        weave_cycles = path.weave_cycles
        weave_actual_pitch_mm = path.weave_actual_pitch_mm
        weave_crescent_bulge_mm = path.weave_crescent_bulge_mm
        custom_hot_start_enabled = bool(settings["custom_hot_start_enabled"])
        software_crater_enabled = bool(settings["software_crater_enabled"])
        self.log(
            f"Built weld workflow from {start_source} to {goal_source} · "
            f"approach={approach_mode} · "
            f"{len(steps)} steps · slots {base_slot}..{final_slot} · "
            f"lead-in={lead_in_mm:.1f} mm / lead-out={lead_out_mm:.1f} mm · "
            f"safe approach={safe_approach_mm:.1f} mm / "
            f"pre-start lead={pre_start_lead_mm:.1f} mm · "
            f"seam travel target={weld_tcp_speed_mm_s:.2f} mm/s · "
            + (
                f"weave={weave_pattern} ±A={weave_amplitude_mm:.1f} mm · "
                f"pitch≤{weave_pitch_mm:.1f} mm/cycle "
                f"(actual {weave_actual_pitch_mm:.2f}, {weave_cycles} cycles) · "
                f"dwell L/R={weave_left_dwell_s:.2f}/{weave_right_dwell_s:.2f} s · "
                + (f"crescent forward bulge={weave_crescent_bulge_mm:.2f} mm · "
                   if weave_pattern == "crescent" else "")
                + f"axis={weave_axis} · "
                if weave_enabled else "weave=OFF · "
            )
            + ("ARC-OFF at endpoint after software_crater · " if software_crater_enabled
               else f"ARC-OFF lead={arc_off_delay_ms:.0f} ms · ")
            + (
                f"hot start=+{settings['hot_start_percent']:.1f}%/"
                f"hold adj {settings['hot_start_hold_adjustment']:+d} · "
                if settings["hot_start_enabled"] else "hot start=OFF · "
            )
            + (
                f"custom motion hold={settings['custom_hot_start_hold_s']:.3f}s "
                f"at {'lead start' if has_lead_in else 'sensed START'} · "
                if custom_hot_start_enabled else "custom motion hold=OFF · "
            )
            + (
                f"native crater panel ref (RX observation only)={settings['crater_panel_current_ref_a']:.1f}A/"
                f"{settings['crater_panel_voltage_ref_v']:.1f}V/"
                f"{settings['crater_panel_time_ref_s']:.2f}s · "
                if settings["expect_native_crater"] else "native crater observation=OFF · "
            )
            + f"orientation={self.seam_orientation_mode.get()} · "
            "fixed World XYZ tilt="
            f"({float(self.weld_fixed_tilt_x_deg.get()):+.1f}°, "
            f"{float(self.weld_fixed_tilt_y_deg.get()):+.1f}°, "
            f"{float(self.weld_fixed_tilt_z_deg.get()):+.1f}°) · "
            + (
                "START WAIT → align weld attitude → weld lead START → Fastech DO0 OFF → "
                if approach_mode == "taught_wait" else
                "START WAIT → SAFE(weld attitude) → PRE-START → START/Fastech DI0 → Fastech DO0 OFF → "
                if safe_approach is not None
                else "START WAIT → START/Fastech DI0(teaching attitude) → Fastech DO0 OFF → "
            )
            + (
                (
                    "e_a SAFE corridor → WELD LEAD-IN → "
                    if safe_approach is not None
                    else "WAIT-XYZ retract (keep teaching attitude) → LEAD-IN → "
                )
                if has_lead_in and approach_mode != "taught_wait"
                else ""
            )
            + (f"[D-WELD ON/ARC established → "
               + (f"custom hold {settings['custom_hot_start_hold_s']:.3f}s → "
                  if custom_hot_start_enabled else "")
               + f"weld motion → endpoint HOLD software_crater "
               f"{settings['software_crater_ratio_percent']:.0f}%/"
               f"{settings['software_crater_voltage_v']:.1f}V/"
               f"{settings['software_crater_hold_s']:.2f}s → ARC OFF → restore main] "
               if software_crater_enabled else
               "[D-WELD ON/ARC established → "
               + (f"custom hold {settings['custom_hot_start_hold_s']:.3f}s → "
                  if custom_hot_start_enabled else "")
               + "endpoint-only LEAD→LEAD motion "
                 "(START/GOAL are logical ARC landmarks) + pre-GOAL ARC-OFF watcher] ")
            + "→ GOAL WAIT → END → Fastech DO0 ON"
        )
        return steps

    def build_initial_scenario(self):
        """Append INITIAL POSE -> HEAD SWEEP -> WELD WAIT -> WELD FINISH."""
        required = ("robot_start", "weld_wait", "weld_finish")
        missing = [
            TEACHING_POSES[name] for name in required
            if self.taught_robot_poses.get(name) is None
        ]
        if missing:
            self.error("Capture/load first: " + ", ".join(missing))
            return
        for name in required:
            if self.taught_robot_poses[name][0] != "right_manipulator":
                self.error(f"{TEACHING_POSES[name]} must belong to the right arm")
                return
        try:
            tcp_speed_m_s = self._selected_tcp_speed_m_s()
        except ValueError as error:
            self.error(str(error))
            return
        velocity_scale = max(0.01, min(1.0, self.velocity_percent.get() / 100.0))
        base_slot = next_sequential_slot(
            self.sequence_steps, int(self.sequence_parallel_slot.get())
        )

        def named_step(pose_name, slot):
            group, joint_names, positions, tcp = self.taught_robot_poses[pose_name]
            return {
                "type": "named_pose",
                "pose_name": pose_name,
                "pose_label": TEACHING_POSES[pose_name],
                "planning_group": group,
                "joint_names": tuple(joint_names),
                "positions": tuple(positions),
                "tcp_pose": copy.deepcopy(tcp),
                "velocity_scale": velocity_scale,
                "tcp_speed_m_s": tcp_speed_m_s,
                "parallel_slot": slot,
                "duration": 0.0,
                "touch_guard": pose_name in TOUCH_GUARDED_TEACHING_POSES,
                "continue_after_touch": False,
            }

        def head_step(joint1_deg, slot):
            return {
                "type": "head_motion",
                "joint1_rad": math.radians(joint1_deg),
                "joint2_rad": math.radians(45.0),
                "parallel_slot": slot,
                "duration": 2.0,
            }

        steps = [named_step("robot_start", base_slot)]
        steps.extend(
            head_step(angle, base_slot + 1 + index)
            for index, angle in enumerate((0.0, -20.0, 20.0, 0.0))
        )
        steps.append(named_step("weld_wait", base_slot + 5))
        steps.append(named_step("weld_finish", base_slot + 6))

        self.sequence_model.extend(steps)
        self.refresh_sequence_table(select_last=True)
        self.log(
            f"Built initial scenario · {len(steps)} steps · "
            f"slots {base_slot}..{base_slot + 6} · "
            "INITIAL POSE -> HEAD SWEEP J1=(0/-20/20/0 deg), J2=45 deg -> "
            "WELD WAIT -> WELD FINISH"
        )

    def compute_two_touch_seam(self):
        missing = [
            name
            for name in CORNER_TOUCH_NAMES
            if self.seam_probe_touches[name] is None
        ]
        if missing:
            self.error(
                "Complete all four Fastech DI0 probes first: " + ", ".join(missing)
            )
            return
        teaching_reference = self._ensure_seam_teaching_reference(
            require_complete=True
        )
        if teaching_reference is None:
            return
        start_data = teaching_reference["weld_start"]
        end_data = teaching_reference["weld_end"]
        if start_data is None or end_data is None:
            self.error("Capture/load Weld start and Weld goal poses first")
            return
        if (
            start_data[0] != self.planning_group.get()
            or end_data[0] != self.planning_group.get()
        ):
            self.error("Weld start/goal teaching poses belong to another arm")
            return
        try:
            count = int(self.corner_touch_count.get())
            wall_offset = 0.0
            floor_offset = 0.0
        except (ValueError, tk.TclError):
            self.error("Seam point count or probe offsets are invalid")
            return
        try:
            raw_points = corner_seam_from_touches(
                self.seam_probe_touches,
                count,
            )
            (
                _reference,
                wall_normal,
                floor_normal,
                wall_label,
                floor_label,
            ) = self._seam_geometry_settings(require_teaching=True)
            geometry = self._compute_touch_corrected_seam_geometry(
                teaching_reference,
                wall_normal,
                floor_normal,
                wall_offset,
                floor_offset,
                log_debug=True,
            )
            sensed_start = copy.deepcopy(geometry.start)
            sensed_goal = copy.deepcopy(geometry.goal)
            corrected_points = linear_pose_waypoints(sensed_start, sensed_goal, count)
            # Raw touch geometry is diagnostic.  The adopted seam uses sensed
            # endpoint XYZ and rotates both taught welding orientations by the
            # same sensed-vs-taught World yaw delta.
            raw_points[0].orientation = copy.deepcopy(
                start_data[3].orientation
            )
            raw_points[-1].orientation = copy.deepcopy(
                end_data[3].orientation
            )
            raw_points[:] = linear_pose_waypoints(
                raw_points[0], raw_points[-1], count
            )
            corrected_start, corrected_goal, delta_yaw, orientation_label = (
                apply_sensed_seam_orientation(
                    start_data[3],
                    end_data[3],
                    corrected_points[0],
                    corrected_points[-1],
                    self.seam_orientation_mode.get(),
                )
            )
            corrected_points[:] = linear_pose_waypoints(
                corrected_start, corrected_goal, count
            )
            geometry.start = copy.deepcopy(corrected_start)
            geometry.goal = copy.deepcopy(corrected_goal)
        except ValueError as error:
            self.error(f"Four-touch seam generation failed: {error}")
            return
        self.raw_two_touch_seam = copy.deepcopy(raw_points)
        self.corrected_two_touch_seam = copy.deepcopy(corrected_points)
        self.computed_seam_endpoints = {
            "start": copy.deepcopy(corrected_points[0]),
            "goal": copy.deepcopy(corrected_points[-1]),
        }
        self._update_seam_yaw_status(
            self.computed_seam_endpoints["start"],
            self.computed_seam_endpoints["goal"],
        )
        try:
            self.computed_seam_wait_points = {
                "start": copy.deepcopy(
                    self.taught_robot_poses["weld_start_wait"][3]
                ),
                "goal": copy.deepcopy(
                    self.taught_robot_poses["weld_goal_wait"][3]
                ),
            }
        except (TypeError, ValueError):
            self.computed_seam_wait_points = {"start": None, "goal": None}
        self.path_kind = "di8_four_touch_raw"
        self.weave_source = copy.deepcopy(raw_points)
        self.set_points(raw_points)
        self.node.publish_seam_comparison(
            raw_points,
            corrected_points,
            self.show_path.get(),
        )
        for endpoint in ("start", "goal"):
            self._publish_touch_geometry_if_ready(
                endpoint, self.computed_seam_endpoints[endpoint]
            )
        self.corner_touch_status.configure(
            text=(
                f"RAW + CORRECTED PREVIEW · {len(raw_points)} points · "
                "opaque=raw, translucent=offset corrected"
            )
        )
        self.log(
            "Computed START→GOAL seam from four Fastech DI0 touches · "
            f"wall={wall_label} · base={floor_label} · "
            f"orientation={orientation_label} · "
            f"World Δyaw={math.degrees(delta_yaw):+.3f}° · "
            "both wait poses kept as taught standby"
        )
        # Do not open a plot window automatically after seam correction.
        # Calculation is the commit point: persist corrected start/goal and
        # wait teaching YAML immediately instead of requiring a second button.
        self.correct_two_touch_seam()

    def correct_two_touch_seam(self):
        if not self.corrected_two_touch_seam:
            self.error("Compute the raw/corrected seam preview first")
            return
        start_data = self.taught_robot_poses["weld_start"]
        end_data = self.taught_robot_poses["weld_end"]
        if self._wait_fixed_tilt_mode_enabled():
            start_data = start_data or copy.deepcopy(
                self.taught_robot_poses.get("weld_start_wait")
            )
            end_data = end_data or copy.deepcopy(
                self.taught_robot_poses.get("weld_goal_wait")
            )
        if start_data is None or end_data is None:
            self.error(
                "Weld start/goal storage seeds are unavailable; capture "
                "START/GOAL WAIT first"
            )
            return
        corrected_start = copy.deepcopy(self.corrected_two_touch_seam[0])
        corrected_end = copy.deepcopy(self.corrected_two_touch_seam[-1])
        updates = [
            ("weld_start", start_data, corrected_start),
            ("weld_end", end_data, corrected_end),
        ]
        try:
            saved_paths = []
            for pose_name, stored, corrected_tcp in updates:
                planning_group, joint_names, positions, _old_tcp = stored
                yaml_path = self._initial_state_yaml_path(
                    planning_group, pose_name
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
            self.error(f"Corrected seam YAML update failed: {error}")
            return
        self.log(
            "CORRECTED SEAM TCP YAML SAVED · "
            + " · ".join(path.name for path in saved_paths)
        )
        for pose_name, stored, corrected_tcp in updates:
            planning_group, joint_names, positions, _old_tcp = stored
            self.taught_robot_poses[pose_name] = (
                planning_group,
                joint_names,
                positions,
                corrected_tcp,
            )
        ik_targets = []
        for pose_name, stored, corrected_tcp in updates:
            endpoint = (
                "goal"
                if pose_name in ("weld_end", "weld_goal_wait")
                else "start"
            )
            ik_targets.append((
                endpoint,
                stored[0],
                copy.deepcopy(corrected_tcp),
                tuple(stored[1]),
                pose_name,
            ))
        threading.Thread(
            target=self.node.resolve_tcp_joint_states,
            args=(tuple(ik_targets),),
            daemon=True,
        ).start()
        self.path_kind = "di8_four_touch_corrected"
        self.weave_source = copy.deepcopy(self.corrected_two_touch_seam)
        # Register the adopted seam as a weave base so "Generate weave" weaves
        # the seam that will be welded rather than the pre-touch straight line.
        self.weave_base_paths["corrected"] = copy.deepcopy(
            self.corrected_two_touch_seam)
        self.set_points(self.corrected_two_touch_seam)
        self.node.publish_points(
            self.corrected_two_touch_seam, self.show_path.get()
        )
        self.corner_touch_status.configure(
            text=(
                "CORRECTED SEAM ADOPTED · Weld start/goal YAML saved · "
                "both wait poses kept as manual standby"
            )
        )
        self.log(
            "Adopted corrected seam and updated Weld start/Weld goal YAML · "
            "START/GOAL wait unchanged · sequential MoveIt IK update started"
        )

    def _advance_corner_touch_target(self):
        current = self.corner_touch_target.get()
        index = CORNER_TOUCH_NAMES.index(current)
        if index + 1 < len(CORNER_TOUCH_NAMES):
            self.corner_touch_target.set(CORNER_TOUCH_NAMES[index + 1])

    def _record_corner_touch(self, pose, source):
        target = self.corner_touch_target.get()
        self.corner_touches[target] = copy.deepcopy(pose)
        captured = [name for name in CORNER_TOUCH_NAMES if self.corner_touches[name] is not None]
        self.corner_touch_status.configure(
            text=f"Captured {target} from {source} · {len(captured)}/4: {', '.join(captured)}"
        )
        self.log(f"Corner touch stored · {target} · source={source}")
        self._advance_corner_touch_target()

    def generate_corner_touch_seam(self):
        try:
            points = corner_seam_from_touches(
                self.corner_touches, int(self.corner_touch_count.get())
            )
        except (ValueError, tk.TclError) as error:
            self.error(f"Corner seam generation failed: {error}")
            return
        self.path_kind = "corner_midpoint"
        self.weave_source = copy.deepcopy(points)
        self.set_points(points)
        self.node.publish_points(points, self.show_path.get())
        self.log(
            "Generated 90° corner root seam from two floor/wall 1:1 midpoint pairs"
        )

    def add_motion_sequence_step(self):
        if not self.points:
            self.error("Create or teach a motion path first")
            return
        try:
            interpolation = float(self.interpolation_step_mm.get()) * 0.001
            tcp_speed_m_s = self._selected_tcp_speed_m_s()
        except (ValueError, tk.TclError) as error:
            self.error(str(error))
            return
        try:
            slot, duration = self._sequence_slot_and_duration()
        except ValueError as error:
            self.error(str(error))
            return
        self.sequence_model.add({
            "type": "motion",
            "planning_group": self.planning_group.get(),
            "points": copy.deepcopy(self.points),
            "velocity_scale": max(0.01, min(1.0, self.velocity_percent.get() / 100.0)),
            "tcp_speed_m_s": tcp_speed_m_s,
            "interpolation_step": interpolation,
            "path_kind": self.path_kind,
            "parallel_slot": slot,
            "duration": duration,
            "touch_guard": False,
            "continue_after_touch": False,
        })
        self.refresh_sequence_table(select_last=True)

    def add_latest_rviz_plan_step(self):
        display, age = self.node.latest_rviz_plan()
        if display is None:
            self.error("Plan a path in RViz/MoveIt first")
            return
        try:
            slot, duration = self._sequence_slot_and_duration()
        except ValueError as error:
            self.error(str(error))
            return
        trajectories = [
            copy.deepcopy(trajectory)
            for trajectory in display.trajectory
            if trajectory.joint_trajectory.points
        ]
        joint_names = tuple(
            dict.fromkeys(
                name
                for trajectory in trajectories
                for name in trajectory.joint_trajectory.joint_names
            )
        )
        arms = [
            arm for arm, names in ARM_JOINT_NAMES.items()
            if names.intersection(joint_names)
        ]
        planning_group = (
            f"{arms[0]}_manipulator" if len(arms) == 1 else "unknown"
        )
        point_count = sum(
            len(trajectory.joint_trajectory.points)
            for trajectory in trajectories
        )
        self.sequence_model.add({
            "type": "planned_trajectory",
            "planning_group": planning_group,
            "required_arms": tuple(arms),
            "trajectory_start": copy.deepcopy(display.trajectory_start),
            "trajectories": trajectories,
            "model_id": display.model_id,
            "joint_names": joint_names,
            "point_count": point_count,
            "captured_age": float(age),
            "parallel_slot": slot,
            "duration": duration,
        })
        self.refresh_sequence_table(select_last=True)
        self.log(
            f"Added latest RViz plan · {len(trajectories)} trajectory(s) · "
            f"{point_count} points · received {age:.1f} s ago"
        )

    def add_named_pose_sequence_step(self):
        pose_name = self._selected_teaching_pose_name()
        stored = self.taught_robot_poses[pose_name]
        if stored is None:
            self.error(
                f"Capture or load {TEACHING_POSES[pose_name]} first"
            )
            return
        try:
            slot, duration = self._sequence_slot_and_duration()
            tcp_speed_m_s = self._selected_tcp_speed_m_s()
        except ValueError as error:
            self.error(str(error))
            return
        planning_group, joint_names, positions, tcp = stored
        self.sequence_model.add({
            "type": "named_pose",
            "pose_name": pose_name,
            "pose_label": TEACHING_POSES[pose_name],
            "planning_group": planning_group,
            "joint_names": tuple(joint_names),
            "positions": tuple(positions),
            "tcp_pose": copy.deepcopy(tcp),
            "velocity_scale": max(
                0.01,
                min(1.0, self.velocity_percent.get() / 100.0),
            ),
            "tcp_speed_m_s": tcp_speed_m_s,
            "parallel_slot": slot,
            "duration": duration,
            "touch_guard": pose_name in TOUCH_GUARDED_TEACHING_POSES,
            "continue_after_touch": False,
        })
        self.refresh_sequence_table(select_last=True)

    def add_sleep_sequence_step(self):
        try:
            seconds = float(self.sequence_sleep_seconds.get())
        except (ValueError, tk.TclError):
            self.error("Sleep duration is invalid")
            return
        if not math.isfinite(seconds) or not 0.0 <= seconds <= 3600.0:
            self.error("Sleep duration must be in 0..3600 seconds")
            return
        self.sequence_model.add({
            "type": "sleep",
            "seconds": seconds,
        })
        self.refresh_sequence_table(select_last=True)

    def add_head_motion_sequence_step(self):
        try:
            joint1_deg = float(self.sequence_head_joint1_deg.get())
            joint2_deg = float(self.sequence_head_joint2_deg.get())
        except (ValueError, tk.TclError):
            self.error("Head target angle is invalid")
            return
        if not all(math.isfinite(v) and -180.0 <= v <= 180.0 for v in (joint1_deg, joint2_deg)):
            self.error("Head joint targets must be in -180..180 degrees")
            return
        try:
            slot, duration = self._sequence_slot_and_duration()
        except ValueError as error:
            self.error(str(error))
            return
        if duration <= 0.0:
            self.error("Head move duration (Output duration s) must be > 0 seconds")
            return
        self.sequence_model.add({
            "type": "head_motion",
            "joint1_rad": math.radians(joint1_deg),
            "joint2_rad": math.radians(joint2_deg),
            "parallel_slot": slot,
            "duration": duration,
        })
        self.refresh_sequence_table(select_last=True)
        self.log(
            f"Added HEAD MOVE J to sequence · slot {slot} · "
            f"J1={joint1_deg:.1f}° J2={joint2_deg:.1f}° · {duration:.1f} s"
        )

    def add_digital_weld_step(self, command):
        command = str(command).strip().lower()
        if command not in ("on", "off", "set"):
            self.error(f"Unknown D-WELD command: {command}")
            return
        try:
            slot, duration = self._sequence_slot_and_duration()
        except ValueError as error:
            self.error(str(error))
            return
        # Snapshot the current recipe for every D-WELD row, including OFF.
        # OFF does not transmit I/V, but keeping the snapshot prevents the
        # sequence editor from showing stale 100 A / 10 V defaults and keeps
        # ON/OFF metadata consistent.
        try:
            settings = copy.deepcopy(self._digital_weld_settings())
        except ValueError as error:
            if command == "off":
                # Safety OFF must remain addable even if a recipe field is
                # temporarily invalid. Use the current validated defaults only
                # as metadata; execution still issues an unconditional ARC OFF.
                settings = copy.deepcopy(DEFAULT_DIGITAL_WELD_SETTINGS)
            else:
                self.error(f"Cannot add D-WELD {command.upper()}: {error}")
                return
        self.sequence_model.add({
            "type": "digital_weld",
            "command": command,
            "settings": settings,
            "parallel_slot": slot,
            "duration": duration,
        })
        self.refresh_sequence_table(select_last=True)
        self.log(
            f"Added D-WELD {command.upper()} to sequence · "
            f"slot {slot} · {duration:.3f} s"
        )

    def add_gas_sequence_step(self, enabled):
        try:
            slot, duration = self._sequence_slot_and_duration()
        except ValueError as error:
            self.error(str(error))
            return
        enabled = bool(enabled)
        self.sequence_model.add({
            "type": "gas",
            "enabled": enabled,
            "parallel_slot": slot,
            "duration": duration,
        })
        self.refresh_sequence_table(select_last=True)
        self.log(
            f"Added GAS {'ON' if enabled else 'OFF'} to sequence · "
            f"slot {slot} · {duration:.3f} s"
        )

    def _sequence_slot_and_duration(self):
        try:
            slot = int(self.sequence_parallel_slot.get())
            duration = float(self.sequence_duration_seconds.get())
        except (ValueError, tk.TclError) as error:
            raise ValueError("Sequence slot/duration is invalid") from error
        if not 1 <= slot <= 999:
            raise ValueError("Sequence parallel slot must be in 1..999")
        if not math.isfinite(duration) or not 0.0 <= duration <= 3600.0:
            raise ValueError("Sequence duration must be in 0..3600 seconds")
        return slot, duration

    def _selected_sequence_index(self):
        selected = self.sequence_table.selection()
        if not selected:
            self.sequence_model.select(None)
            return None
        return self.sequence_model.select(int(selected[0]))

    def refresh_sequence_table(self, select_last=False):
        selected = self._selected_sequence_index() if self.sequence_table.get_children() else None
        self.sequence_table.delete(*self.sequence_table.get_children())
        for index, step in enumerate(self.sequence_steps):
            timing = (
                f"slot {step.get('parallel_slot', index + 1)} · "
                f"{step.get('duration', 0.0):.1f} s"
            )
            if step["type"] == "motion":
                guard_detail = (
                    " · Fastech DI0 GUARDED"
                    if step.get("touch_guard", False)
                    else " · Fastech DI0 IGNORED"
                )
                tcp_speed = float(step.get("tcp_speed_m_s", 0.0))
                speed_detail = (
                    f"TCP {tcp_speed * 1000.0:.2f} mm/s"
                    if tcp_speed > 0.0
                    else f"speed {step['velocity_scale']:.1%}"
                )
                weave_detail = (
                    f" · {step.get('weld_weave_pattern', 'sine')} weave "
                    f"{float(step.get('weld_weave_amplitude_mm', 0.0)):.1f} mm"
                    if step.get("weld_weave_enabled", False)
                    else ""
                )
                if step.get("weld_weave_enabled"):
                    speed_detail = f"seam travel target {step['weld_tcp_speed_mm_s']:.2f} mm/s"
                if step.get("weld_weave_enabled"):
                    weave_detail += (
                        f" · pitch≤{step.get('weld_weave_pitch_mm', 5.0):.1f} mm/cycle"
                        f" · dwell L/R {step.get('weld_weave_left_dwell_s', 0.0):.2f}/"
                        f"{step.get('weld_weave_right_dwell_s', 0.0):.2f} s"
                    )
                lead_detail = (
                    f" · lead {float(step.get('lead_in_mm', 0.0)):.1f}/"
                    f"{float(step.get('lead_out_mm', 0.0)):.1f} mm"
                    if step.get("weld_scenario_stage") == "weld_motion"
                    else ""
                )
                detail = (
                    f"{step['planning_group']} · {len(step['points'])} poses · "
                    f"{speed_detail}{lead_detail}{weave_detail} · "
                    f"{step['path_kind']}{guard_detail} · {timing}"
                )
                kind = "MOTION"
            elif step["type"] == "planned_trajectory":
                kind = "RVIZ PLAN"
                detail = (
                    f"{step.get('planning_group', 'unknown')} · exact stored "
                    f"trajectory · {len(step['trajectories'])} segment(s) · "
                    f"{step.get('point_count', 0)} points · {timing}"
                )
            elif step["type"] == "named_pose":
                kind = "GO TO POSE"
                tcp_speed = float(step.get("tcp_speed_m_s", 0.0))
                speed_detail = (
                    f"TCP {tcp_speed * 1000.0:.2f} mm/s"
                    if tcp_speed > 0.0
                    else f"speed {step['velocity_scale']:.1%}"
                )
                detail = (
                    f"{step['pose_label']} · {step['planning_group']} · "
                    f"{speed_detail} · {timing}"
                )
            elif step["type"] == "dual_arm_pose":
                kind = "DUAL ARM"
                detail = f"{step['pose_label']} · both arms · speed {step['velocity_scale']:.0%} · {timing}"
            elif step["type"] == "spray_motion":
                kind = "SPRAY PATH"
                detail = f"{step['path_kind']} · left arm · speed {step['velocity_scale']:.0%} · {timing}"
            elif step["type"] == "head_motion":
                kind = "HEAD MOVE J"
                detail = (
                    f"J1={math.degrees(step['joint1_rad']):.1f}° · "
                    f"J2={math.degrees(step['joint2_rad']):.1f}° · "
                    f"{timing}"
                )
            elif step["type"] == "sleep":
                kind = "SLEEP"
                detail = f"{step['seconds']:.3f} seconds"
            elif step["type"] == "digital_weld":
                settings = step.get("settings")
                kind = f"D-WELD {step['command'].upper()}"
                if settings is None:
                    detail = f"Hi-COMM · no recipe payload · {timing}"
                else:
                    detail = (
                        f"Hi-COMM · I={settings['current_a']} A "
                        f"V={settings['voltage']:.1f} V · {timing}"
                    )
                if step.get("trigger_before_goal", False):
                    detail += (
                        f" · ARC OFF lead="
                        f"{float(step.get('arc_off_delay_s', 0.0)) * 1000.0:.0f} ms"
                    )
            elif step["type"] == "software_crater":
                kind = "SOFTWARE CRATER"
                settings = step["settings"]
                detail = (f"endpoint HOLD · {settings['software_crater_ratio_percent']:.1f}% / "
                          f"{settings['software_crater_voltage_v']:.1f} V / "
                          f"{settings['software_crater_hold_s']:.2f} s · {timing}")
            elif step["type"] == "custom_hot_start":
                kind = "CUSTOM HOT START"
                detail = (f"motion hold after ARC established · "
                          f"{step['settings']['custom_hot_start_hold_s']:.3f} s · {timing}")
            elif step["type"] == "gas":
                kind = f"GAS {'ON' if step['enabled'] else 'OFF'}"
                detail = f"Hi-COMM shielding gas · {timing}"
            elif step["type"] == "digital_output":
                backend = step.get("io_backend", "rainbow_legacy")
                source = (
                    "Fastech Ethernet"
                    if backend == FASTECH_TOUCH_BACKEND
                    else "Legacy Rainbow control-box"
                )
                kind = (
                    f"{source} DO{int(step['port'])} "
                    f"{'ON' if step['value'] else 'OFF'}"
                )
                detail = f"{source} output · {timing}"
            else:
                kind = f"INCH {step['direction'].upper()}"
                detail = f"Hi-COMM timed wire feed · {timing}"
            self.sequence_table.insert(
                "", tk.END, iid=str(index), values=(index + 1, kind, detail)
            )
        target = len(self.sequence_steps) - 1 if select_last else selected
        if target is not None and 0 <= target < len(self.sequence_steps):
            self.sequence_table.selection_set(str(target))
            self.load_selected_sequence_values()

    def load_selected_sequence_values(self, _event=None):
        """Load the selected row into the Sequence Builder edit controls."""
        index = self._selected_sequence_index()
        if index is None or not 0 <= index < len(self.sequence_steps):
            return
        step = self.sequence_steps[index]
        if step["type"] == "sleep":
            self.sequence_sleep_seconds.set(step.get("seconds", 0.0))
        else:
            self.sequence_parallel_slot.set(step.get("parallel_slot", index + 1))
            self.sequence_duration_seconds.set(step.get("duration", 0.0))
        if step["type"] in ("motion", "named_pose"):
            self.sequence_edit_velocity_percent.set(
                float(step.get("velocity_scale", 0.2)) * 100.0
            )
            self.sequence_edit_tcp_speed_mm_s.set(
                float(step.get("weld_tcp_speed_mm_s", float(step.get("tcp_speed_m_s", 0.0))*1000.0))
            )
            self.sequence_edit_touch_guard.set(
                bool(step.get("touch_guard", False))
            )
            self.sequence_edit_continue_after_touch.set(
                bool(step.get("continue_after_touch", False))
            )
        else:
            self.sequence_edit_touch_guard.set(False)
            self.sequence_edit_continue_after_touch.set(False)
        if step["type"] == "head_motion":
            self.sequence_head_joint1_deg.set(
                math.degrees(step.get("joint1_rad", 0.0))
            )
            self.sequence_head_joint2_deg.set(
                math.degrees(step.get("joint2_rad", 0.0))
            )
        if step["type"] in ("digital_weld", "custom_hot_start") and step.get("settings"):
            try:
                settings = validate_digital_weld_settings(step["settings"])
            except ValueError as error:
                self.error(f"Invalid D-WELD sequence settings: {error}")
                return
            step["settings"] = settings
            self.weld_current_raw.set(settings["current_a"])
            self.weld_voltage_raw.set(settings["voltage_tenths"])
            self.weld_material.set(settings["material"])
            self.weld_diameter_mm.set(settings["diameter_mm"])
            self.weld_mode.set(settings["mode"])
            self.weld_gas.set(settings["gas"])
            self.weld_synergic.set(settings["synergic"])
            self.weld_correction.set(settings["correction"])
            self.weld_hot_start_enabled.set(settings["hot_start_enabled"])
            self.weld_hot_start_percent.set(settings["hot_start_percent"])
            self.weld_hot_start_hold_adjustment.set(
                settings["hot_start_hold_adjustment"]
            )
            self.weld_custom_hot_start_enabled.set(settings["custom_hot_start_enabled"])
            self.weld_custom_hot_start_hold_s.set(settings["custom_hot_start_hold_s"])
            self.weld_custom_hot_start_percent.set(settings["custom_hot_start_percent"])
            self.weld_expect_native_crater.set(settings["expect_native_crater"])
            self.weld_crater_panel_current_ref_a.set(settings["crater_panel_current_ref_a"])
            self.weld_crater_panel_voltage_ref_v.set(settings["crater_panel_voltage_ref_v"])
            self.weld_crater_panel_time_ref_s.set(settings["crater_panel_time_ref_s"])
            self.weld_software_crater_enabled.set(settings["software_crater_enabled"])
            self.weld_software_crater_ratio_percent.set(settings["software_crater_ratio_percent"])
            self.weld_software_crater_voltage_v.set(settings["software_crater_voltage_v"])
            self.weld_software_crater_hold_s.set(settings["software_crater_hold_s"])
            self.weld_wire_consumable_alpha_mm.set(
                settings["wire_consumable_alpha_mm"]
            )
        self.sequence_status.configure(
            text=(
                f"Sequence #{index + 1} selected · double-click the row to edit"
            )
        )

    def _commit_selected_sequence_step_edits(self, index):
        """Write the Sequence Builder edit-panel values into
        ``self.sequence_steps[index]``.

        Plan/Execute always uses whatever is currently shown in the editor for
        the selected row. Returns ``(True, None)`` on success or
        ``(False, error_message)`` on a validation failure, leaving the step
        unchanged in the failure case.
        """
        step = self.sequence_steps[index]
        original_steps = copy.deepcopy(self.sequence_steps)
        try:
            if step["type"] == "sleep":
                seconds = float(self.sequence_sleep_seconds.get())
                if not math.isfinite(seconds) or not 0.0 <= seconds <= 3600.0:
                    raise ValueError("Sleep duration must be in 0..3600 seconds")
                step["seconds"] = seconds
            else:
                slot, duration = self._sequence_slot_and_duration()
                step["parallel_slot"] = slot
                step["duration"] = duration
            if step["type"] in ("motion", "named_pose"):
                speed = float(self.sequence_edit_velocity_percent.get())
                if not math.isfinite(speed) or not 1.0 <= speed <= 100.0:
                    raise ValueError("Selected motion speed must be in 1..100%")
                step["velocity_scale"] = speed / 100.0
                tcp_speed = float(self.sequence_edit_tcp_speed_mm_s.get())
                if not math.isfinite(tcp_speed) or not 0.0 <= tcp_speed <= 500.0:
                    raise ValueError("Selected TCP speed must be 0..500 mm/s")
                step["tcp_speed_m_s"] = tcp_speed * 0.001
                step["touch_guard"] = bool(
                    self.sequence_edit_touch_guard.get()
                )
                step["continue_after_touch"] = bool(
                    self.sequence_edit_continue_after_touch.get()
                )
                if step.get("weld_scenario_stage") == "weld_motion":
                    self.sequence_steps = update_weld_scenario_motion_values(
                        self.sequence_steps,
                        index,
                        tcp_speed_mm_s=tcp_speed,
                        lead_in_mm=float(step.get("lead_in_mm", 0.0)),
                        lead_out_mm=float(step.get("lead_out_mm", 0.0)),
                    )
                    step = self.sequence_steps[index]
            if step["type"] == "head_motion":
                joint1_deg = float(self.sequence_head_joint1_deg.get())
                joint2_deg = float(self.sequence_head_joint2_deg.get())
                if not all(
                    math.isfinite(v) and -180.0 <= v <= 180.0
                    for v in (joint1_deg, joint2_deg)
                ):
                    raise ValueError(
                        "Head joint targets must be in -180..180 degrees"
                    )
                step["joint1_rad"] = math.radians(joint1_deg)
                step["joint2_rad"] = math.radians(joint2_deg)
            if (
                step["type"] == "digital_weld"
                and step.get("command") in ("on", "set")
            ):
                settings = self._digital_weld_settings()
                scenario_id = step.get("weld_scenario_id")
                if scenario_id:
                    has_stage = any(row.get("weld_scenario_id") == scenario_id
                                    and row.get("weld_scenario_stage") == "software_crater"
                                    for row in self.sequence_steps)
                    if bool(settings["software_crater_enabled"]) != has_stage:
                        raise ValueError("Software Crater Enabled changes sequence structure; rebuild the scenario")
                    has_custom = any(row.get("weld_scenario_id") == scenario_id
                                     and row.get("weld_scenario_stage") == "custom_hot_start"
                                     for row in self.sequence_steps)
                    if bool(settings["custom_hot_start_enabled"]) != has_custom:
                        raise ValueError("Custom Hot Start Enabled changes sequence structure; rebuild the scenario")
                    for row in self.sequence_steps:
                        if row.get("weld_scenario_id") == scenario_id and row.get("type") in ("digital_weld", "software_crater", "custom_hot_start"):
                            row["settings"] = copy.deepcopy(settings)
                else:
                    step["settings"] = copy.deepcopy(settings)
            elif step["type"] == "custom_hot_start":
                if not self.weld_custom_hot_start_enabled.get():
                    raise ValueError("Custom Hot Start Enabled changes sequence structure; rebuild the scenario")
                hold_s = float(self.weld_custom_hot_start_hold_s.get())
                if not 0.01 <= hold_s <= 5.0:
                    raise ValueError("Custom Hot Start hold must be in 0.01..5.0 seconds")
                scenario_id = step.get("weld_scenario_id")
                for row in self.sequence_steps:
                    if row.get("weld_scenario_id") == scenario_id and row.get("type") in (
                        "digital_weld", "software_crater", "custom_hot_start"
                    ):
                        row["settings"]["custom_hot_start_hold_s"] = hold_s
                        row["settings"]["custom_hot_start_percent"] = float(self.weld_custom_hot_start_percent.get())
                        row["settings"] = validate_digital_weld_settings(row["settings"])
            validate_managed_weld_sequence(
                self.sequence_steps, require_complete=True
            )
        except (ValueError, tk.TclError) as error:
            self.sequence_steps = original_steps
            return False, str(error)
        return True, None

    def open_sequence_step_editor(self, _event=None):
        """Open a type-aware editor for one generated scenario row."""
        index = self._selected_sequence_index()
        if index is None:
            self.error("Select a sequence step to edit")
            return
        step = self.sequence_steps[index]
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit sequence #{index + 1} · {step['type']}")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        variables = {}
        row = 0

        def entry(name, label, value, width=14):
            nonlocal row
            variable = tk.StringVar(value=str(value))
            variables[name] = variable
            ttk.Label(body, text=label).grid(
                row=row, column=0, padx=4, pady=3, sticky=tk.W
            )
            ttk.Entry(body, textvariable=variable, width=width).grid(
                row=row, column=1, padx=4, pady=3, sticky=tk.W
            )
            row += 1

        def choice(name, label, value, values):
            nonlocal row
            variable = tk.StringVar(value=str(value))
            variables[name] = variable
            ttk.Label(body, text=label).grid(
                row=row, column=0, padx=4, pady=3, sticky=tk.W
            )
            ttk.Combobox(
                body,
                textvariable=variable,
                values=tuple(values),
                state="readonly",
                width=16,
            ).grid(row=row, column=1, padx=4, pady=3, sticky=tk.W)
            row += 1

        def check(name, label, value):
            nonlocal row
            variable = tk.BooleanVar(value=bool(value))
            variables[name] = variable
            ttk.Checkbutton(
                body, text=label, variable=variable
            ).grid(row=row, column=0, columnspan=2, padx=4, pady=3, sticky=tk.W)
            row += 1

        if step["type"] != "sleep":
            entry("parallel_slot", "Parallel slot", step.get("parallel_slot", 1))
            entry("duration", "Duration (s)", step.get("duration", 0.0))
        if step["type"] == "sleep":
            entry("seconds", "Sleep (s)", step.get("seconds", 0.0))
        elif step["type"] in ("motion", "named_pose"):
            entry(
                "velocity_percent",
                "Motion speed (%)",
                float(step.get("velocity_scale", 0.2)) * 100.0,
            )
            entry(
                "tcp_speed_mm_s",
                "Average seam travel (mm/s, 0=scale)",
                float(step.get("weld_tcp_speed_mm_s", float(step.get("tcp_speed_m_s", 0.0))*1000.0)),
            )
            if step["type"] == "motion":
                entry(
                    "interpolation_mm",
                    "Interpolation (mm)",
                    float(step.get("interpolation_step", 0.005)) * 1000.0,
                )
                check(
                    "linear_motion_profile",
                    "Linear (constant velocity) instead of S-curve",
                    step.get("linear_motion_profile", False),
                )
                if "usable_seam_start" in step and "usable_seam_goal" in step:
                    entry(
                        "lead_in_mm",
                        "Weld lead-in (mm)",
                        float(step.get("lead_in_mm", 0.0)),
                    )
                    entry(
                        "lead_out_mm",
                        "Weld lead-out (mm)",
                        float(step.get("lead_out_mm", 0.0)),
                    )
                    if step.get("weld_weave_enabled"):
                        for key, label, default in (
                            ("weld_weave_amplitude_mm", "One-side amplitude ±A (mm)", 3.0),
                            ("weld_weave_pitch_mm", "Pitch (mm/cycle)", 5.0),
                            ("weld_weave_left_dwell_s", "Sine left dwell (s)", 0.0),
                            ("weld_weave_right_dwell_s", "Sine right dwell (s)", 0.0),
                        ):
                            entry(key, label, step.get(key, default))
            check("touch_guard", "Stop this step on Fastech DI0", step.get("touch_guard"))
            check(
                "continue_after_touch",
                "Continue scenario after confirmed Fastech DI0 stop",
                step.get("continue_after_touch"),
            )
        elif step["type"] == "head_motion":
            entry(
                "head_joint1_deg",
                "Head J1 target (deg)",
                math.degrees(step.get("joint1_rad", 0.0)),
            )
            entry(
                "head_joint2_deg",
                "Head J2 target (deg)",
                math.degrees(step.get("joint2_rad", 0.0)),
            )
        elif step["type"] == "digital_weld":
            choice("command", "D-WELD command", step["command"], ("on", "off", "set"))
            if step.get("settings"):
                settings = copy.deepcopy(step["settings"])
            else:
                try:
                    settings = copy.deepcopy(self._digital_weld_settings())
                except ValueError:
                    settings = copy.deepcopy(DEFAULT_DIGITAL_WELD_SETTINGS)
            entry("current_a", "Current (A)", settings["current_a"])
            entry("voltage", "Voltage (V)", settings["voltage"])
            choice("material", "Wire material", settings["material"], MATERIAL_CODES)
            choice("diameter_mm", "Wire diameter (mm)", settings["diameter_mm"], DIAMETER_CODES)
            choice("mode", "Mode", settings["mode"], MODE_CODES)
            choice("gas", "Gas type", settings["gas"], GAS_CODES)
            check("synergic", "Synergic", settings["synergic"])
            entry("correction", "Correction", settings["correction"])
            check("hot_start_enabled", "Hot start enabled", settings["hot_start_enabled"])
            entry("hot_start_percent", "Hot start boost (%)", settings["hot_start_percent"])
            entry(
                "hot_start_hold_adjustment",
                "Hot start hold adjustment (-15..+15)",
                settings["hot_start_hold_adjustment"],
            )
            check("custom_hot_start_enabled", "Custom Hot Start (Motion Hold)",
                  settings["custom_hot_start_enabled"])
            entry("custom_hot_start_hold_s", "Custom hold after ARC established (s)",
                  settings["custom_hot_start_hold_s"])
            check("expect_native_crater", "Observe panel native crater (RX only)", settings["expect_native_crater"])
            entry("crater_panel_current_ref_a", "Panel Current Ref (A)", settings["crater_panel_current_ref_a"])
            entry("crater_panel_voltage_ref_v", "Panel Voltage Ref (V)", settings["crater_panel_voltage_ref_v"])
            entry("crater_panel_time_ref_s", "Panel Time Ref (s)", settings["crater_panel_time_ref_s"])
            check("software_crater_enabled", "Software Crater Enabled", settings["software_crater_enabled"])
            entry("software_crater_ratio_percent", "Crater Current Ratio (20..40 %)", settings["software_crater_ratio_percent"])
            entry("software_crater_voltage_v", "Software Crater Voltage (V)", settings["software_crater_voltage_v"])
            entry("software_crater_hold_s", "Software Crater Hold (s)", settings["software_crater_hold_s"])
            entry(
                "wire_consumable_alpha_mm",
                "Wire consumable alpha (mm)",
                settings["wire_consumable_alpha_mm"],
            )
            if step.get("trigger_before_goal", False):
                entry(
                    "arc_off_delay_ms",
                    "ARC OFF lead (ms)",
                    float(step.get("arc_off_delay_s", 0.0)) * 1000.0,
                )
        elif step["type"] == "custom_hot_start":
            entry("custom_hot_start_hold_s", "Hold after ARC established (s)",
                  step["settings"]["custom_hot_start_hold_s"])
        elif step["type"] == "gas":
            choice(
                "enabled", "Gas command",
                "on" if step["enabled"] else "off", ("on", "off")
            )
        elif step["type"] == "digital_output":
            backend = step.get("io_backend", "rainbow_legacy")
            entry(
                "port",
                (
                    "Fastech physical DO channel"
                    if backend == FASTECH_TOUCH_BACKEND
                    else "Legacy Rainbow DO port"
                ),
                step["port"],
            )
            choice(
                "value",
                "Output command",
                "on" if step["value"] else "off",
                ("on", "off"),
            )

        def save():
            try:
                updated = copy.deepcopy(step)
                if updated["type"] == "sleep":
                    seconds = float(variables["seconds"].get())
                    if not math.isfinite(seconds) or not 0.0 <= seconds <= 3600.0:
                        raise ValueError("Sleep must be in 0..3600 seconds")
                    updated["seconds"] = seconds
                else:
                    slot = int(variables["parallel_slot"].get())
                    duration = float(variables["duration"].get())
                    if not 1 <= slot <= 999:
                        raise ValueError("Parallel slot must be in 1..999")
                    if not math.isfinite(duration) or not 0.0 <= duration <= 3600.0:
                        raise ValueError("Duration must be in 0..3600 seconds")
                    updated["parallel_slot"] = slot
                    updated["duration"] = duration
                if updated["type"] in ("motion", "named_pose"):
                    speed = float(variables["velocity_percent"].get())
                    if not 1.0 <= speed <= 100.0:
                        raise ValueError("Motion speed must be in 1..100%")
                    updated["velocity_scale"] = speed / 100.0
                    tcp_speed = float(variables["tcp_speed_mm_s"].get())
                    if not 0.0 <= tcp_speed <= 500.0:
                        raise ValueError("TCP speed must be in 0..500 mm/s")
                    updated["tcp_speed_m_s"] = tcp_speed * 0.001
                    updated["touch_guard"] = variables["touch_guard"].get()
                    updated["continue_after_touch"] = variables[
                        "continue_after_touch"
                    ].get()
                    if updated["type"] == "motion":
                        interpolation = float(variables["interpolation_mm"].get())
                        if not 0.5 <= interpolation <= 20.0:
                            raise ValueError("Interpolation must be in 0.5..20 mm")
                        updated["interpolation_step"] = interpolation * 0.001
                        updated["linear_motion_profile"] = variables[
                            "linear_motion_profile"
                        ].get()
                        for key in ("weld_weave_amplitude_mm", "weld_weave_pitch_mm", "weld_weave_left_dwell_s", "weld_weave_right_dwell_s"):
                            if key in variables:
                                updated[key] = float(variables[key].get())
                        if (
                            "usable_seam_start" in updated
                            and "usable_seam_goal" in updated
                        ):
                            lead_in_mm = float(variables["lead_in_mm"].get())
                            lead_out_mm = float(variables["lead_out_mm"].get())
                            if not 0.0 <= lead_in_mm <= 100.0:
                                raise ValueError(
                                    "Weld lead-in must be in 0..100 mm"
                                )
                            if not 0.0 <= lead_out_mm <= 100.0:
                                raise ValueError(
                                    "Weld lead-out must be in 0..100 mm"
                                )
                            seam_start = updated["usable_seam_start"]
                            seam_goal = updated["usable_seam_goal"]
                            lead_start, lead_end = seam_lead_poses(
                                seam_start,
                                seam_goal,
                                lead_in_mm * 0.001,
                                lead_out_mm * 0.001,
                            )
                            motion_start = (
                                lead_start if lead_in_mm > 1e-6
                                else copy.deepcopy(seam_start)
                            )
                            motion_end = (
                                lead_end if lead_out_mm > 1e-6
                                else copy.deepcopy(seam_goal)
                            )
                            updated["points"] = (motion_start, motion_end)
                            updated["lead_start"] = copy.deepcopy(lead_start)
                            updated["lead_end"] = copy.deepcopy(lead_end)
                            updated["lead_in_mm"] = lead_in_mm
                            updated["lead_out_mm"] = lead_out_mm
                elif updated["type"] == "head_motion":
                    joint1_deg = float(variables["head_joint1_deg"].get())
                    joint2_deg = float(variables["head_joint2_deg"].get())
                    if not all(
                        -180.0 <= v <= 180.0 for v in (joint1_deg, joint2_deg)
                    ):
                        raise ValueError(
                            "Head joint targets must be in -180..180 degrees"
                        )
                    updated["joint1_rad"] = math.radians(joint1_deg)
                    updated["joint2_rad"] = math.radians(joint2_deg)
                elif updated["type"] == "digital_weld":
                    updated["command"] = variables["command"].get()
                    if updated["command"] == "off":
                        # Keep a recipe snapshot as metadata even though ARC OFF
                        # does not retransmit current/voltage.
                        current = int(round(float(variables["current_a"].get())))
                        voltage_tenths = int(round(
                            float(variables["voltage"].get()) * 10.0
                        ))
                        updated["settings"] = validate_digital_weld_settings({
                            "current_a": current,
                            "voltage_tenths": voltage_tenths,
                            "material": variables["material"].get(),
                            "diameter_mm": variables["diameter_mm"].get(),
                            "mode": variables["mode"].get(),
                            "gas": variables["gas"].get(),
                            "synergic": variables["synergic"].get(),
                            "correction": variables["correction"].get(),
                            "hot_start_enabled": variables["hot_start_enabled"].get(),
                            "hot_start_percent": variables["hot_start_percent"].get(),
                            "hot_start_hold_adjustment": variables[
                                "hot_start_hold_adjustment"
                            ].get(),
                            "custom_hot_start_enabled": variables["custom_hot_start_enabled"].get(),
                            "custom_hot_start_hold_s": variables["custom_hot_start_hold_s"].get(),
                            "expect_native_crater": variables["expect_native_crater"].get(),
                            "crater_panel_current_ref_a": variables["crater_panel_current_ref_a"].get(),
                            "crater_panel_voltage_ref_v": variables["crater_panel_voltage_ref_v"].get(),
                            "crater_panel_time_ref_s": variables["crater_panel_time_ref_s"].get(),
                            "software_crater_enabled": variables["software_crater_enabled"].get(),
                            "software_crater_ratio_percent": variables["software_crater_ratio_percent"].get(),
                            "software_crater_voltage_v": variables["software_crater_voltage_v"].get(),
                            "software_crater_hold_s": variables["software_crater_hold_s"].get(),
                            "wire_consumable_alpha_mm": variables[
                                "wire_consumable_alpha_mm"
                            ].get(),
                        })
                    else:
                        current = int(round(float(variables["current_a"].get())))
                        voltage_tenths = int(round(
                            float(variables["voltage"].get()) * 10.0
                        ))
                        settings = validate_digital_weld_settings({
                            "current_a": current,
                            "voltage_tenths": voltage_tenths,
                            "material": variables["material"].get(),
                            "diameter_mm": variables["diameter_mm"].get(),
                            "mode": variables["mode"].get(),
                            "gas": variables["gas"].get(),
                            "synergic": variables["synergic"].get(),
                            "correction": variables["correction"].get(),
                            "hot_start_enabled": variables["hot_start_enabled"].get(),
                            "hot_start_percent": variables["hot_start_percent"].get(),
                            "hot_start_hold_adjustment": variables[
                                "hot_start_hold_adjustment"
                            ].get(),
                            "custom_hot_start_enabled": variables["custom_hot_start_enabled"].get(),
                            "custom_hot_start_hold_s": variables["custom_hot_start_hold_s"].get(),
                            "expect_native_crater": variables["expect_native_crater"].get(),
                            "crater_panel_current_ref_a": variables["crater_panel_current_ref_a"].get(),
                            "crater_panel_voltage_ref_v": variables["crater_panel_voltage_ref_v"].get(),
                            "crater_panel_time_ref_s": variables["crater_panel_time_ref_s"].get(),
                            "software_crater_enabled": variables["software_crater_enabled"].get(),
                            "software_crater_ratio_percent": variables["software_crater_ratio_percent"].get(),
                            "software_crater_voltage_v": variables["software_crater_voltage_v"].get(),
                            "software_crater_hold_s": variables["software_crater_hold_s"].get(),
                            "wire_consumable_alpha_mm": variables[
                                "wire_consumable_alpha_mm"
                            ].get(),
                        })
                        updated["settings"] = settings
                    if "arc_off_delay_ms" in variables:
                        arc_off_delay_ms = float(
                            variables["arc_off_delay_ms"].get()
                        )
                        if not 0.0 <= arc_off_delay_ms <= 2000.0:
                            raise ValueError(
                                "ARC OFF lead time must be in 0..2000 ms"
                            )
                        updated["arc_off_delay_s"] = arc_off_delay_ms * 0.001
                elif updated["type"] == "custom_hot_start":
                    hold_s = float(variables["custom_hot_start_hold_s"].get())
                    if not 0.01 <= hold_s <= 5.0:
                        raise ValueError("Custom Hot Start hold must be in 0.01..5.0 seconds")
                    updated["settings"]["custom_hot_start_hold_s"] = hold_s
                elif updated["type"] == "gas":
                    updated["enabled"] = variables["enabled"].get() == "on"
                elif updated["type"] == "digital_output":
                    port = int(variables["port"].get())
                    backend = updated.get("io_backend", "rainbow_legacy")
                    maximum = 7 if backend == FASTECH_TOUCH_BACKEND else 15
                    if not 0 <= port <= maximum:
                        raise ValueError(
                            f"{'Fastech' if backend == FASTECH_TOUCH_BACKEND else 'Rainbow'} "
                            f"DO port must be in 0..{maximum}"
                        )
                    updated["port"] = port
                    updated["value"] = variables["value"].get() == "on"
                candidate_steps = copy.deepcopy(self.sequence_steps)
                candidate_steps[index] = updated
                if updated.get("type") == "digital_weld" and updated.get("weld_scenario_id"):
                    scenario_id = updated["weld_scenario_id"]
                    has_stage = any(row.get("weld_scenario_id") == scenario_id
                                    and row.get("weld_scenario_stage") == "software_crater"
                                    for row in candidate_steps)
                    if bool(updated["settings"]["software_crater_enabled"]) != has_stage:
                        raise ValueError("Software Crater Enabled changes sequence structure; rebuild the scenario")
                    has_custom = any(row.get("weld_scenario_id") == scenario_id
                                     and row.get("weld_scenario_stage") == "custom_hot_start"
                                     for row in candidate_steps)
                    if bool(updated["settings"]["custom_hot_start_enabled"]) != has_custom:
                        raise ValueError("Custom Hot Start Enabled changes sequence structure; rebuild the scenario")
                    for row in candidate_steps:
                        if row.get("weld_scenario_id") == scenario_id and row.get("type") in ("digital_weld", "software_crater", "custom_hot_start"):
                            row["settings"] = copy.deepcopy(updated["settings"])
                if updated.get("type") == "custom_hot_start" and updated.get("weld_scenario_id"):
                    scenario_id = updated["weld_scenario_id"]
                    for row in candidate_steps:
                        if row.get("weld_scenario_id") == scenario_id and row.get("type") in (
                            "digital_weld", "software_crater", "custom_hot_start"
                        ):
                            row["settings"]["custom_hot_start_hold_s"] = updated["settings"]["custom_hot_start_hold_s"]
                if updated.get("weld_scenario_stage") == "weld_motion":
                    candidate_steps = update_weld_scenario_motion_values(
                        candidate_steps,
                        index,
                        tcp_speed_mm_s=(
                            float(updated.get("weld_tcp_speed_mm_s", float(updated.get("tcp_speed_m_s", 0.0)) * 1000.0))
                        ),
                        lead_in_mm=float(updated.get("lead_in_mm", 0.0)),
                        lead_out_mm=float(updated.get("lead_out_mm", 0.0)),
                    )
                validate_managed_weld_sequence(
                    candidate_steps, require_complete=True
                )
            except (ValueError, tk.TclError) as error:
                messagebox.showerror("Invalid sequence value", str(error), parent=dialog)
                return
            self.sequence_steps = candidate_steps
            self.refresh_sequence_table()
            self.sequence_table.selection_set(str(index))
            self.load_selected_sequence_values()
            self.log(f"Updated scenario step #{index + 1} in editor")
            dialog.destroy()

        buttons = ttk.Frame(body)
        buttons.grid(row=row, column=0, columnspan=2, pady=(10, 0), sticky=tk.E)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(
            side=tk.RIGHT, padx=3
        )
        ttk.Button(buttons, text="Save", command=save).pack(side=tk.RIGHT, padx=3)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())

        def grab_when_viewable():
            try:
                if dialog.winfo_exists() and dialog.winfo_viewable():
                    dialog.grab_set()
                elif dialog.winfo_exists():
                    dialog.after(20, grab_when_viewable)
            except tk.TclError:
                # The editor may have been closed before the idle callback.
                return

        dialog.after_idle(grab_when_viewable)

    def delete_sequence_step(self):
        index = self._selected_sequence_index()
        if index is None:
            self.error("Select a sequence step")
            return
        self.sequence_model.delete(index)
        if not self.sequence_steps:
            self.sequence_parallel_slot.set(1)
        self.refresh_sequence_table()

    def build_torch_clean_sequence(self):
        """Refresh cleaner rows from config YAML without executing equipment."""
        if self.sequence_running:
            self.error("Cannot build Torch Clean while a sequence is running")
            return False
        try:
            steps = self.torch_cleaner_panel.build_sequence_steps()
            replacement = self.sequence_model.with_replaced_cleaner(steps)
            validate_managed_weld_sequence(replacement)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            self.error(f"Cannot build Torch Clean: {error}")
            return False
        self.sequence_steps = replacement
        self.refresh_sequence_table(select_last=True)
        self.torch_cleaner_panel.status.set(
            f"Torch Clean: {len(steps)} steps added to Sequence Builder"
        )
        self.log(f"Built Torch Clean from {self.torch_cleaner_panel.folder.get()} · {len(steps)} steps")
        return True

    def build_combined_work_cycle(self):
        """Build the bounded Pass-4/cleaner/spray routine; never move on Build."""
        if self.sequence_running:
            self.error("Cannot build work cycle while a sequence is running")
            return False
        try:
            from construct_robot.teaching_paths import teaching_config_dir
            config = load_work_cycle(teaching_config_dir() / "combined_work_cycle.yaml")
            repeats = int(self.work_cycle_repeats.get())
            if set(self.four_pass_references) != {1, 2, 3, 4}:
                if not self.load_four_pass_references():
                    raise ValueError("Load the four pass references before building the work cycle")
            self._validate_four_pass_source_hashes()
            reference = self.four_pass_references[4]
            corrected = self.four_pass_corrected[4]
            poses = {}
            for endpoint, name in (
                ("start_wait", "weld_start_wait"), ("start", "weld_start"),
                ("goal_wait", "weld_goal_wait"), ("goal", "weld_end"),
            ):
                names, positions = reference["joint_states"][endpoint]
                poses[name] = (
                    "right_manipulator", tuple(names), tuple(positions),
                    copy.deepcopy(corrected[endpoint]),
                )
            finish = reference.get("additional_pose_entries", {}).get("weld_finish")
            if finish is None:
                raise ValueError("Pass 4 must include the weld_finish teaching pose")
            poses["weld_finish"] = parse_teaching_snapshot_entry("weld_finish", finish)
            weld_steps = self.build_sensed_weld_sequence(
                teaching_poses=poses, force_unsensed=True,
                weave_override=config["weave"], append=False,
            )
            if not weld_steps:
                raise ValueError("Pass 4 weld scenario could not be built")
            if reference.get("requires_ik"):
                for step in weld_steps:
                    if step.get("type") == "named_pose":
                        step["resolve_target_tcp_ik"] = True
            cleaner_steps = self.torch_cleaner_panel.build_sequence_steps()
            velocity_scale = max(0.01, min(1.0, float(self.velocity_percent.get()) / 100.0))
            steps = assemble_work_cycle(
                config, weld_steps, cleaner_steps, repeats, velocity_scale,
            )
            validate_managed_weld_sequence(steps, require_complete=True)
        except (OSError, KeyError, TypeError, ValueError, tk.TclError, yaml.YAMLError) as error:
            self.error(f"Cannot build full work cycle: {error}")
            return False
        self.fake_arc_enabled.set(True)
        self.sequence_steps = steps
        self.refresh_sequence_table()
        self.log(
            f"Built full work cycle · {repeats} repeat(s) · {len(steps)} steps · "
            "Pass 4 FAKE ARC/tool-Y weave → torch clean → World-X spray → initial"
        )
        return True

    def delete_all_sequence_steps(self):
        if self.sequence_running:
            self.error("Cannot delete the sequence while Plan/Execute is running")
            return
        if not self.sequence_steps:
            return
        count = len(self.sequence_steps)
        model = getattr(self, "sequence_model", None)
        if model is None:  # compatibility with lightweight non-GUI callers
            self.sequence_steps.clear()
        else:
            model.clear()
        self.sequence_parallel_slot.set(1)
        self.refresh_sequence_table()
        self.sequence_status.configure(text="Sequence empty")
        self.log(f"Deleted all {count} Sequence Builder rows")

    def move_sequence_step(self, offset):
        index = self._selected_sequence_index()
        if index is None:
            self.error("Select a sequence step")
            return
        target = self.sequence_model.move(index, offset)
        if target is None:
            return
        self.refresh_sequence_table()
        self.sequence_table.selection_set(str(target))

    _pose_execution_conditions = staticmethod(pose_execution_conditions)

    def _sequence_execution_conditions(
        self, steps, indices, execute_requested, run_all
    ):
        recorded_steps = record_step_conditions(steps, indices)
        effective_weld_motion = next((
            step for step in steps
            if step.get("weld_scenario_stage") == "weld_motion"
        ), None)
        effective_arc_off = next((
            step for step in steps
            if step.get("weld_scenario_stage") == "arc_off"
        ), None)

        def effective_motion_value(step_key, gui_value):
            if effective_weld_motion is not None and step_key in effective_weld_motion:
                return effective_weld_motion[step_key]
            return gui_value

        return {
            "mode": "sequence_execute" if execute_requested else "sequence_plan",
            "run_all": bool(run_all),
            "touch_io_backend": FASTECH_TOUCH_BACKEND,
            "touch_input_port": FASTECH_TOUCH_INPUT_PORT,
            "touch_sensing_output_port": FASTECH_TOUCH_OUTPUT_PORT,
            "fastech_ip": self.fastech_ip.get().strip(),
            "fastech_poll_target_hz": self.fastech_poll_rate_hz,
            "hicomm_source_ip": self.hicomm_source_ip.get().strip(),
            "hicomm_welder_ip": self.hicomm_welder_ip.get().strip(),
            "hicomm_port": int(self.hicomm_port.get()),
            "gui_velocity_percent": float(self.velocity_percent.get()),
            "gui_speed_mode": self.speed_mode.get(),
            "gui_tcp_speed_mm_s": float(self.tcp_speed_mm_s.get()),
            "gui_interpolation_step_mm": float(
                self.interpolation_step_mm.get()
            ),
            "seam_orientation_mode": (
                effective_weld_motion.get("seam_orientation_mode")
                if effective_weld_motion is not None
                else self.seam_orientation_mode.get()
            ),
            "weld_fixed_tilt_x_deg": float(
                effective_weld_motion.get(
                    "fixed_world_x_tilt_deg",
                    effective_weld_motion.get(
                        "fixed_tool_x_tilt_deg",
                        self.weld_fixed_tilt_x_deg.get(),
                    ),
                )
                if effective_weld_motion is not None
                else self.weld_fixed_tilt_x_deg.get()
            ),
            "weld_fixed_tilt_y_deg": float(
                effective_weld_motion.get(
                    "fixed_world_y_tilt_deg",
                    effective_weld_motion.get(
                        "fixed_tool_y_tilt_deg",
                        self.weld_fixed_tilt_y_deg.get(),
                    ),
                )
                if effective_weld_motion is not None
                else self.weld_fixed_tilt_y_deg.get()
            ),
            "weld_fixed_tilt_z_deg": float(
                effective_weld_motion.get(
                    "fixed_world_z_tilt_deg",
                    effective_weld_motion.get(
                        "fixed_tool_z_tilt_deg",
                        self.weld_fixed_tilt_z_deg.get(),
                    ),
                )
                if effective_weld_motion is not None
                else self.weld_fixed_tilt_z_deg.get()
            ),
            "weld_weave_enabled": bool(effective_motion_value(
                "weld_weave_enabled", self.weld_weave_enabled.get()
            )),
            "weld_weave_pattern": str(effective_motion_value(
                "weld_weave_pattern", self.weave_pattern.get()
            )),
            **{key: float(effective_motion_value(key, variable.get())) for key, variable in (
                ("weld_weave_pitch_mm", self.weave_pitch_mm),
                ("weld_weave_left_dwell_s", self.weave_left_dwell_s),
                ("weld_weave_right_dwell_s", self.weave_right_dwell_s),
            )},
            "weld_weave_amplitude_mm": float(effective_motion_value(
                "weld_weave_amplitude_mm", self.weave_amplitude_mm.get()
            )),
            "weld_weave_cycles": int(effective_motion_value(
                "weld_weave_cycles", 0
            )),
            "weld_weave_actual_pitch_mm": float(effective_motion_value(
                "weld_weave_actual_pitch_mm", 0.0
            )),
            "weld_weave_crescent_bulge_mm": float(effective_motion_value(
                "weld_weave_crescent_bulge_mm", 0.0
            )),
            "weld_weave_amplitude_definition": (
                "centerline +/- A mm; full width = 2A"
                if effective_motion_value("weld_weave_pattern", "sine") in ("sine", "crescent")
                else "orbit radius = A mm; diameter = 2A"
            ),
            "weld_weave_samples_per_cycle": WELD_WEAVE_SAMPLES_PER_CYCLE,
            "weld_weave_axis": str(effective_motion_value(
                "weld_weave_axis", self.weave_axis.get()
            )),
            # These are effective scenario values, not live Seam Correction
            # widgets.  The per-step snapshot below and this summary therefore
            # cannot disagree after a Builder edit.
            "weld_lead_in_mm": float(effective_motion_value(
                "lead_in_mm", self.weld_lead_in_mm.get()
            )),
            "weld_lead_out_mm": float(effective_motion_value(
                "lead_out_mm", self.weld_lead_out_mm.get()
            )),
            "weld_safe_approach_mm": float(effective_motion_value(
                "safe_approach_mm", self.weld_safe_approach_mm.get()
            )),
            "weld_approach_mode": effective_motion_value(
                "weld_approach_mode", self.weld_approach_mode.get()
            ),
            "weld_pre_start_lead_mm": float(effective_motion_value(
                "pre_start_lead_mm", self.weld_pre_start_lead_mm.get()
            )),
            "weld_tcp_speed_mm_s": float(effective_motion_value(
                "weld_tcp_speed_mm_s", self.weld_tcp_speed_mm_s.get()
            )),
            "weld_arc_off_delay_ms": (
                float(effective_arc_off.get("arc_off_delay_s", 0.0)) * 1000.0
                if effective_arc_off is not None
                else float(self.weld_arc_off_delay_ms.get())
            ),
            "tcp_tracking.parent_frame": "World",
            "tcp_tracking.child_frame": tip_link_for_group(
                effective_weld_motion.get("planning_group", "right_manipulator")
                if effective_weld_motion is not None
                else "right_manipulator"
            ),
            "tcp_tracking.timestamp_source": "TF_header_stamp",
            "initial_fastech_di0": bool(
                self.node.node_touch_input_states.get("right", False)
            ),
            "initial_fastech_do0": (
                None
                if self.fastech_previous_state is None
                or len(self.fastech_previous_state.digital_out)
                <= FASTECH_TOUCH_OUTPUT_PORT
                else bool(
                    self.fastech_previous_state.digital_out[
                        FASTECH_TOUCH_OUTPUT_PORT
                    ]
                )
            ),
            "robot_connected": copy.deepcopy(self.robot_connected),
            "hicomm_connected": bool(self.hicomm_connected),
            "execution_allowed": bool(self.execution_allowed),
            "steps": recorded_steps,
        }

    def run_sequence(self, run_all, execute_requested, steps_override=None):
        if self.sequence_running:
            self.error("A sequence is already running")
            return
        # Keyboard teaching keeps the JTC active (MoveIt Servo), so trajectory
        # execution must not share the arm with it.
        if execute_requested and (
            self.keyboard_velocity_arm is not None
            or self.keyboard_velocity_switching
        ):
            self.error("Disable Keyboard Teaching before executing a sequence")
            return
        if steps_override is not None:
            indices, steps = self.sequence_model.execution_snapshot(
                run_all, steps_override
            )
        else:
            # Generated rows are editable by double-click. Capture the selected
            # row's current values before running the visible Builder sequence.
            selected_index = self._selected_sequence_index()
            if (selected_index is not None and 0 <= selected_index < len(self.sequence_steps)
                    and not any(row.get("work_cycle_id") for row in self.sequence_steps)):
                success, error = self._commit_selected_sequence_step_edits(selected_index)
                if not success:
                    self.error(f"Sequence edit failed: {error}")
                    return
                self.refresh_sequence_table()
            if not run_all:
                self._selected_sequence_index()
            indices, steps = self.sequence_model.execution_snapshot(run_all)
        if not indices:
            self.error("Add or select a sequence step")
            return
        work_cycle = is_work_cycle(steps)
        if work_cycle and (not run_all or not all(step.get("fake_arc_required") for step in steps)):
            self.error("Full work cycle must run all rows as a fake-ARC-only sequence")
            return
        if execute_requested and work_cycle and not self.fake_arc_enabled.get():
            self.error("Full work cycle requires FAKE ARC enabled; rebuild or enable it")
            return
        try:
            execution_conditions = self._sequence_execution_conditions(
                steps, indices, execute_requested, run_all
            )
        except (TypeError, ValueError, tk.TclError) as error:
            self.error(f"Cannot capture sequence execution conditions: {error}")
            return
        attach_execution_conditions(steps, execution_conditions)
        try:
            validate_managed_weld_sequence(
                steps, require_complete=bool(run_all)
            )
        except (TypeError, ValueError) as error:
            self.error(f"Unsafe generated weld scenario: {error}")
            return
        executor = _sequence_executor_for(self)
        if execute_requested:
            preflight_error = executor.execution_preflight_error(steps, work_cycle)
            if preflight_error is not None:
                self.error(preflight_error)
                return
            if not messagebox.askyesno(
                "Execute sequence",
                f"Execute {len(steps)} stored step(s) on physical equipment?",
            ):
                return
            if self.hicomm_client is not None and not work_cycle and (
                steps_override is None or contains_weld_command(steps)
            ):
                self.hicomm_client.allow_outputs()
        executor.begin(
            steps, indices, execute_requested,
            bool(work_cycle or self.fake_arc_enabled.get()),
        )

    # Sequence execution lives in application.sequence_executor.  These
    # entry points keep their names so callers and per-instance overrides
    # (tests, diagnostics) keep working; the executor calls back through them.
    def _interruptible_wait(self, seconds):
        return _sequence_executor_for(self).interruptible_wait(seconds)

    def _execution_fake_arc(self):
        return _sequence_executor_for(self).fake_arc()

    def _sequence_worker(self, steps, indices, execute_requested):
        return _sequence_executor_for(self).run_worker(
            steps, indices, execute_requested
        )

    def _sequence_worker_body(self, steps, indices, execute_requested):
        return _sequence_executor_for(self).run_groups(
            steps, indices, execute_requested
        )

    def _run_sequence_step(self, step, execute_requested):
        return _sequence_executor_for(self).run_step(step, execute_requested)

    def _set_sequence_status(self, text):
        self.sequence_model.status = text
        self.sequence_status.configure(text=text)
        self.pipeline_waiting(f"SEQUENCE STATUS · {text}")

    def _sequence_finished(self, success, message):
        self.sequence_model.finish(success, message)
        text = (
            f"Sequence {'complete' if success else 'stopped/failed'} · "
            f"{message}"
        )
        self.sequence_status.configure(text=text)
        if success:
            self.pipeline_result(f"SEQUENCE COMPLETE · {message}")
        else:
            self.error(f"SEQUENCE FAILED · {message}")

    def stop_sequence(self):
        with self.weld_feedback_lock:
            self._weld_feedback_stopped = True
        if hasattr(self, "torch_cleaner_panel"):
            self.torch_cleaner_panel.abort()
        self._stop_keyboard_wire()
        self.sequence_stop_requested = True
        # STOP NOW also invalidates any pending touch dwell/retract.
        self.node.clear_touch_probe()
        if self.hicomm_client is not None:
            self.hicomm_client.inhibit_outputs()
        self._finish_weld_feedback_record(
            "operator stop",
            self.hicomm_client.latest_status() if self.hicomm_client is not None else None,
        )
        self.hicomm_inching_direction = None
        self.hicomm_gas_enabled.set(False)
        self.hicomm_arc_on_button.configure(state=tk.DISABLED)
        self.hicomm_test_status.configure(text="STOP NOW · ALL OUTPUTS INHIBITED")
        self.node.cancel_active_motion()
        devices = [
            device for device in ("left", "right", "head")
            if self.robot_connected.get(device, False)
        ]
        threading.Thread(
            target=self.node.stop_sequence_equipment,
            args=(devices,),
            daemon=True,
        ).start()
        self.sequence_status.configure(
            text="STOP NOW · Hi-COMM inhibited · canceling robot controllers"
        )
        self.pipeline_waiting(
            "STOP NOW · ARC/GAS/INCH OFF · canceling all robot motion"
        )

    def request_both_robot_power(self, enable):
        if self.robot_power_busy:
            return
        if not enable and not messagebox.askyesno(
            "Shutdown both robot arms",
            "Stop motion and power down BOTH Rainbow robot arms?",
        ):
            return
        self.robot_power_busy = True
        self.robot_activate_both_button.configure(state=tk.DISABLED)
        self.robot_shutdown_both_button.configure(state=tk.DISABLED)
        action = "activating" if enable else "shutting down"
        self.robot_power_status.set(f"BOTH arms {action}...")
        if not enable:
            # The shutdown worker restores a possible velocity controller
            # synchronously before deactivating both trajectory controllers.
            self.emergency_stop_all(restore_keyboard_controller=False)
        threading.Thread(
            target=self._both_robot_power_worker,
            args=(bool(enable),),
            daemon=True,
        ).start()

    def _both_robot_power_worker(self, enable):
        controller_results = {}
        if not enable:
            velocity_arm = self.keyboard_velocity_arm
            if velocity_arm is not None:
                self.node.clear_keyboard_velocity()
                time.sleep(0.05)
                controller_results[velocity_arm] = (
                    self.node.set_keyboard_velocity_controller_enabled(
                        velocity_arm, False
                    )
                )
            for arm in ("left", "right"):
                controller_results[arm] = self.node.switch_arm_controller(
                    arm, False
                )
        power_results = self.node.set_both_robot_power_sync(enable)
        if enable:
            for arm in ("left", "right"):
                if power_results.get(arm, (False, ""))[0]:
                    controller_results[arm] = self.node.switch_arm_controller(
                        arm, True
                    )
        elif self.keyboard_velocity_arm is not None:
            self.keyboard_velocity_arm = None
        self.post(
            self._both_robot_power_result,
            enable,
            power_results,
            controller_results,
        )

    def _both_robot_power_result(
        self, enable, power_results, controller_results
    ):
        self.robot_power_busy = False
        self.robot_activate_both_button.configure(state=tk.NORMAL)
        self.robot_shutdown_both_button.configure(state=tk.NORMAL)
        details = []
        all_ok = True
        for arm in ("left", "right"):
            power_ok, power_message = power_results.get(
                arm, (False, "no power response")
            )
            controller_ok, controller_message = controller_results.get(
                arm, (False, "controller not switched")
            )
            arm_ok = power_ok and controller_ok
            all_ok = all_ok and arm_ok
            details.append(
                f"{arm.upper()} power={'OK' if power_ok else 'FAIL'} "
                f"controller={'OK' if controller_ok else 'FAIL'} "
                f"({power_message}; {controller_message})"
            )
        action = "ACTIVATE" if enable else "SHUTDOWN"
        summary = " · ".join(details)
        self.robot_power_status.set(
            f"{action} BOTH {'OK' if all_ok else 'FAILED'}"
        )
        if all_ok:
            self.pipeline_result(f"{action} BOTH COMPLETE · {summary}")
        else:
            self.error(f"{action} BOTH · {summary}")

    @staticmethod
    def _keyboard_focus_accepts_arrows(widget):
        return isinstance(
            widget,
            (tk.Entry, tk.Listbox, tk.Text, ttk.Entry, ttk.Spinbox, ttk.Combobox),
        )

    def _keyboard_focus_allows_jog(self):
        """Return False for text widgets, external focus, and Tk modal windows."""
        try:
            widget = self.root.focus_get()
        except (KeyError, tk.TclError):
            # Native Tk dialogs such as .__tk__messagebox are not registered
            # in root.children, so focus_get() can raise while resolving them.
            return False
        return (
            widget is not None
            and not self._keyboard_focus_accepts_arrows(widget)
        )

    def _cleaner_task_selected(self):
        return (
            hasattr(self, "task_notebook")
            and self.operator_notebook.select() == str(self.task_notebook.master)
            and self.task_notebook.select() == self.task_notebook.tabs()[1]
        )

    def _cleaner_keyboard_jog_enable_changed(self):
        if self.keyboard_jog_enabled.get() and self.planning_group.get() != "right_manipulator":
            if self.keyboard_velocity_arm is not None or self.keyboard_velocity_switching:
                self.keyboard_jog_enabled.set(False)
                self.error("Disable the other arm's keyboard teaching first")
                return
            self.planning_group.set("right_manipulator")
        self.keyboard_jog_enable_changed()

    def _set_keyboard_jog_enable_state(self, state):
        buttons = getattr(self, "keyboard_jog_enable_buttons", None)
        if buttons is None:
            buttons = (self.keyboard_jog_enable_button,)
        for button in buttons:
            button.configure(state=state)

    def keyboard_jog_enable_changed(self):
        enable = bool(self.keyboard_jog_enabled.get())
        if not enable:
            self._stop_keyboard_wire()
        if self.keyboard_velocity_switching:
            self.keyboard_jog_enabled.set(self.keyboard_velocity_arm is not None)
            return
        if enable:
            arm = self._selected_arm()
            if self.sequence_running or self.node.active_motion_goal is not None:
                self.keyboard_jog_enabled.set(False)
                self.error("Keyboard velocity mode is unavailable during motion")
                return
            if not self.robot_connected.get(arm, False):
                self.keyboard_jog_enabled.set(False)
                self.error(f"Activate and connect the {arm.upper()} robot first")
                return
            if not self.node.keyboard_velocity_controller_ready(arm):
                self.keyboard_jog_enabled.set(False)
                self.error(
                    f"{arm.upper()} Cartesian velocity command publisher is unavailable"
                )
                return
        else:
            arm = self.keyboard_velocity_arm
            self._stop_keyboard_jog_command()
            if arm is None:
                self.keyboard_jog_status.set("Keyboard teaching locked")
                return
        self.keyboard_velocity_switching = True
        self._set_keyboard_jog_enable_state(tk.DISABLED)
        uses_servo = self.node.keyboard_teaching_uses_servo()
        self.keyboard_jog_status.set(
            ("STARTING MoveIt Servo keyboard teaching..." if uses_servo
             else "SWITCHING to native Cartesian velocity...")
            if enable
            else ("ZERO command · stopping MoveIt Servo..." if uses_servo
                  else "ZERO command · restoring trajectory controller...")
        )
        threading.Thread(
            target=self._keyboard_velocity_mode_worker,
            args=(arm, enable),
            daemon=True,
        ).start()

    def _keyboard_velocity_mode_worker(self, arm, enable):
        if enable:
            # The velocity controller starts with a zero command. Do not send
            # a motion until the operator presses a direction key.
            self.node.set_keyboard_velocity(None, (0.0,) * 6)
            if not self.node.wait_for_keyboard_velocity_feedback(arm):
                self.post(
                    self._keyboard_velocity_mode_result,
                    arm,
                    enable,
                    False,
                    "fresh measured joint feedback is unavailable",
                )
                return
            canceled, cancel_message = self.node.cancel_controller_goals(arm)
            if not canceled:
                self.post(
                    self._keyboard_velocity_mode_result,
                    arm,
                    enable,
                    False,
                    f"cannot establish exclusive teaching control: {cancel_message}",
                )
                return
            if not self.node.wait_until_arm_stopped(arm, timeout=1.5):
                self.post(
                    self._keyboard_velocity_mode_result,
                    arm,
                    enable,
                    False,
                    "arm did not reach standstill before controller exchange",
                )
                return
            # MoveIt Servo keeps RB in its Servo-J hold on purpose, so RB
            # Idle is only expected for the native jog exchange.
            if (
                not self.node.keyboard_teaching_uses_servo()
                and not self.node.wait_for_robot_idle(arm, timeout=1.0)
            ):
                # Measured standstill above is the hard safety condition. Some
                # RB firmware keeps reporting Moving briefly after the final
                # servo sample; do not turn that status lag into a permanent
                # keyboard-mode lockout.
                self.node.get_logger().warning(
                    f"{arm.upper()} RB motion state did not settle to Idle; "
                    "continuing atomic controller exchange after confirmed standstill"
                )
        else:
            self.node.clear_keyboard_velocity()
            # Let jog_robot_l consume one explicit stop before ownership is
            # returned to the trajectory controller.
            time.sleep(0.10)
        success, message = self.node.set_keyboard_velocity_controller_enabled(
            arm, enable
        )
        self.post(
            self._keyboard_velocity_mode_result,
            arm,
            enable,
            success,
            message,
        )

    def _keyboard_velocity_mode_result(
        self, arm, enable, success, message
    ):
        self.keyboard_velocity_switching = False
        self._set_keyboard_jog_enable_state(tk.NORMAL)
        if success and enable:
            # Jogging invalidates any trajectory preview made from the old pose.
            self.initial_plan_ready = False
            self.node.initial_planned_trajectory = None
            self._refresh_initial_position_controls()
            self.keyboard_velocity_arm = arm
            # Arrow keys are motion controls while teaching is enabled. Move
            # focus away from a speed Spinbox/Combobox so their class binding
            # cannot consume the first key event.
            self.root.focus_set()
            registration = self.multi_pass_registration
            expected_key = None
            if registration is not None and arm == "right":
                expected_key = {
                    "waiting_start_capture": "I",
                    "waiting_goal_capture": "J",
                }.get(registration.get("phase"))
            if expected_key is not None:
                number = registration["pass"]
                self.keyboard_jog_status.set(
                    f"READY RIGHT · jog then press {expected_key} to capture"
                )
                self._set_four_pass_status(
                    f"Pass {number} correction · Keyboard Teaching READY · "
                    f"waiting for {expected_key} capture"
                )
                self.pipeline_result(
                    f"Pass {number} Keyboard Teaching enabled automatically · "
                    f"jog to the real endpoint and press {expected_key}"
                )
            else:
                self.keyboard_jog_status.set(
                    f"READY {arm.upper()} · hold arrow to move"
                )
            backend = (
                "MoveIt Servo" if self.node.keyboard_teaching_uses_servo()
                else "native Cartesian velocity"
            )
            self.log(f"Keyboard {backend} enabled · {message}")
            return
        if success:
            self.keyboard_velocity_arm = None
            self.node.set_keyboard_velocity(None, (0.0,) * 6)
            self.keyboard_jog_enabled.set(False)
            self.keyboard_jog_status.set("Keyboard teaching locked")
            self.log(f"Keyboard trajectory controller restored · {message}")
            return
        self.keyboard_velocity_arm = None
        self.keyboard_jog_enabled.set(False)
        self.keyboard_jog_status.set(f"VELOCITY MODE FAILED · {message}")
        if self.multi_pass_registration is not None:
            number = self.multi_pass_registration["pass"]
            self._set_four_pass_status(
                f"Pass {number} correction · automatic Keyboard Teaching enable "
                "FAILED · use Enable Keyboard Teaching to retry"
            )
        self.error(f"Keyboard controller exchange failed · {message}")

    def _cancel_keyboard_release_timer(self):
        if self.keyboard_release_after_id is None:
            return
        try:
            self.root.after_cancel(self.keyboard_release_after_id)
        except tk.TclError:
            pass
        self.keyboard_release_after_id = None

    def _stop_keyboard_jog_command(self, status=None):
        """Publish zero now and cancel every Tk-side continuation."""
        self._cancel_keyboard_release_timer()
        active_key = self.keyboard_velocity_active_key
        arm = self.keyboard_velocity_arm
        self.keyboard_velocity_active_key = None
        self.keyboard_stop_generation += 1
        generation = self.keyboard_stop_generation
        self.node.clear_keyboard_velocity()
        if status is not None and active_key is not None:
            self.keyboard_jog_status.set(status)
        if active_key is not None and arm is not None:
            threading.Thread(
                target=self._verify_keyboard_jog_stop_worker,
                args=(arm, generation),
                daemon=True,
            ).start()

    def _keyboard_jog_stop_timeout_s(self):
        # Servo-J follows the Servo command ~0.2-0.3 s late, so the arm is
        # still settling at the native jog's 0.22 s check.  A false fallback
        # move_stop would drop RB to Idle and bring back the start kick.
        return 0.6 if self.node.keyboard_teaching_uses_servo() else 0.22

    def _verify_keyboard_jog_stop_worker(self, arm, generation):
        stopped = self.node.wait_until_arm_stopped(
            arm,
            timeout=self._keyboard_jog_stop_timeout_s(),
            stable_duration_s=0.08,
        )
        self.post(
            self._keyboard_jog_stop_verified,
            arm, generation, stopped,
        )

    def _keyboard_jog_stop_verified(self, arm, generation, stopped):
        if (
            generation != self.keyboard_stop_generation
            or self.keyboard_velocity_active_key is not None
            or arm != self.keyboard_velocity_arm
            or not self.keyboard_jog_enabled.get()
        ):
            return
        if stopped:
            self.log(f"Keyboard jog STOP CONFIRMED · {arm.upper()} measured standstill")
            return
        self.keyboard_jog_status.set(
            f"STOP FALLBACK · {arm.upper()} controlled move_stop"
        )
        self.log(
            f"Keyboard jog zero not stationary within "
            f"{self._keyboard_jog_stop_timeout_s():.2f} s · "
            f"requesting {arm.upper()} controlled move_stop"
        )
        threading.Thread(
            target=self._keyboard_jog_direct_stop_worker,
            args=(arm, generation),
            daemon=True,
        ).start()

    def _keyboard_jog_direct_stop_worker(self, arm, generation):
        success, message = self.node.request_direct_motion_stop(arm)
        self.post(
            self._keyboard_jog_direct_stop_result,
            arm, generation, success, message,
        )

    def _keyboard_jog_direct_stop_result(
        self, arm, generation, success, message
    ):
        if generation != self.keyboard_stop_generation:
            return
        callback = self.log if success else self.error
        callback(
            f"Keyboard jog STOP FALLBACK · {arm.upper()} · "
            f"{'OK' if success else 'FAILED'} · {message}"
        )

    def keyboard_velocity_deadman_stopped(self, arm):
        """Synchronize UI state after the ROS-thread deadman sent zero."""
        if arm != self.keyboard_velocity_arm:
            return
        key = self.keyboard_velocity_active_key
        self._stop_keyboard_jog_command()
        self.keyboard_jog_status.set(
            f"STOPPED · {arm.upper()} deadman zero · press direction again"
        )
        self.log(
            f"Keyboard jog DEADMAN STOP · {arm.upper()} · "
            f"stale key={key or 'none'}"
        )

    def _stop_keyboard_wire(self):
        timer = self.keyboard_wire_release_after_id
        self.keyboard_wire_release_after_id = None
        if timer is not None:
            try:
                self.root.after_cancel(timer)
            except tk.TclError:
                pass
        key = self.keyboard_wire_active_key
        self.keyboard_wire_active_key = None
        if key is not None:
            self.request_hicomm_inching(
                "forward" if key == "f" else "reverse", False
            )
            self.keyboard_jog_status.set("Wire inch OFF")

    def keyboard_wire_key_press(self, event):
        if not self.keyboard_jog_enabled.get():
            return None
        if not self._keyboard_focus_allows_jog():
            return None
        key = str(event.keysym).lower()
        if key not in ("f", "r"):
            return None
        timer = self.keyboard_wire_release_after_id
        self.keyboard_wire_release_after_id = None
        if timer is not None:
            self.root.after_cancel(timer)
        if self.keyboard_wire_active_key == key:
            return "break"
        if self.keyboard_wire_active_key is not None:
            self._stop_keyboard_wire()
        if (
            self.keyboard_velocity_switching
            # Arm names here are "left"/"right", the values _selected_arm()
            # returns and the ones keyboard_velocity_arm is assigned from.
            # Comparing against the planning-group name instead made both
            # tests below always true, so wire inching could never start.
            or self.keyboard_velocity_arm != "right"
            or self._selected_arm() != "right"
            or self.sequence_running
            or self.node.active_motion_goal is not None
        ):
            self.error("Wire inching requires idle right-arm keyboard teaching")
            return "break"
        if not self.hicomm_connected:
            self.error("Connect Hi-COMM before keyboard wire inching")
            return "break"
        direction = "forward" if key == "f" else "reverse"
        if self.request_hicomm_inching(direction, True):
            self.keyboard_wire_active_key = key
            self.keyboard_jog_status.set(f"WIRE {direction.upper()} · release to stop")
        return "break"

    def keyboard_wire_key_release(self, event):
        key = str(event.keysym).lower()
        if key != self.keyboard_wire_active_key:
            return None
        if self.keyboard_wire_release_after_id is not None:
            self.root.after_cancel(self.keyboard_wire_release_after_id)
        # X11 key repeat may synthesize release/press pairs. The next press
        # cancels this timer; the final physical release turns the output off.
        self.keyboard_wire_release_after_id = self.root.after(
            35, lambda selected=key: self._finish_keyboard_wire_release(selected)
        )
        return "break"

    def _finish_keyboard_wire_release(self, key):
        self.keyboard_wire_release_after_id = None
        if key == self.keyboard_wire_active_key:
            self._stop_keyboard_wire()

    def keyboard_wire_focus_out(self, _event):
        self._stop_keyboard_wire()

    def keyboard_jog_focus_out(self, event):
        # Losing application focus can also lose KeyRelease.  The latched RB
        # jog must be stopped immediately, independently of that event.
        self.keyboard_wire_focus_out(event)
        if self.keyboard_velocity_active_key is not None:
            arm = self.keyboard_velocity_arm
            self._stop_keyboard_jog_command(
                f"STOPPED · {(arm or 'robot').upper()} window focus lost"
            )
            self.log(
                f"Keyboard jog STOP · {(arm or 'robot').upper()} focus lost"
            )

    def _keyboard_ros_input_online(self):
        return time.monotonic() - self.keyboard_ros_input_last_at < 0.35

    def keyboard_arrow_state_received(self, mask):
        """Consume unambiguous physical arrow state from the ROS2 X11 node."""
        mask = int(mask) & 0x0F
        previous_key = self.keyboard_ros_physical_key
        self.keyboard_ros_input_last_at = time.monotonic()
        self.keyboard_ros_physical_mask = mask
        if mask == 0:
            self.keyboard_ros_zero_seen = True
        key = {
            0x01: "Left",
            0x02: "Right",
            0x04: "Up",
            0x08: "Down",
        }.get(mask)
        self.keyboard_ros_physical_key = key

        # Multiple arrows are treated as STOP. The selected teaching planes
        # already map one arrow to a deterministic Cartesian vector.
        has_focus = self._keyboard_focus_allows_jog()
        usable = (
            self.keyboard_ros_zero_seen
            and has_focus
            and self.keyboard_jog_enabled.get()
        )
        if not usable:
            if self.keyboard_velocity_active_key is not None:
                self._stop_keyboard_jog_command("STOPPED · keyboard input inactive")
            return
        if key == previous_key:
            if key is not None and key == self.keyboard_velocity_active_key:
                self.node.refresh_keyboard_velocity(self.keyboard_velocity_arm)
            return

        # A physical release reaches this path directly; unlike Tk auto-repeat,
        # it is not delayed to guess whether a synthetic release will be
        # followed by another press.
        if self.keyboard_velocity_active_key is not None:
            arm = self.keyboard_velocity_arm
            self._stop_keyboard_jog_command()
            self.log(
                f"Keyboard jog PHYSICAL RELEASE · "
                f"{(arm or 'robot').upper()} velocity zero"
            )
        if key is None:
            return
        self.keyboard_ros_dispatching = True
        try:
            event = type("PhysicalKeyEvent", (), {"keysym": key})()
            self.keyboard_jog_key_press(event)
        finally:
            self.keyboard_ros_dispatching = False

    def keyboard_teaching_shortcut_key(self, event):
        """Handle speed cycling and current-pose saves in teaching mode."""
        key = str(event.keysym).lower()
        if key not in ("v", "x") and self._cleaner_task_selected():
            self.keyboard_jog_status.set(
                "Cleaner pose: select a Teaching index and use Save current right-arm pose"
            )
            return "break"
        registration = self.multi_pass_registration
        if not self.keyboard_jog_enabled.get():
            if registration is not None and key in ("i", "j"):
                self._set_four_pass_status(
                    f"Pass {registration['pass']} correction · {key.upper()} received, "
                    "but Keyboard Teaching is not ready"
                )
                self.error(
                    f"{key.upper()} capture not started · wait for automatic "
                    "Keyboard Teaching enable or enable it manually"
                )
                return "break"
            return None
        if not self._keyboard_focus_allows_jog():
            if registration is not None and key in ("i", "j"):
                self._set_four_pass_status(
                    f"Pass {registration['pass']} correction · {key.upper()} received, "
                    "but keyboard focus is in an input field"
                )
                self.error(
                    f"{key.upper()} capture not started · click the main GUI background "
                    "and press the key again"
                )
                return "break"
            return None
        pending_release = self.keyboard_shortcut_release_ids.pop(key, None)
        if pending_release is not None:
            try:
                self.root.after_cancel(pending_release)
            except tk.TclError:
                pass
        if key in self.keyboard_shortcut_active_keys:
            return "break"
        self.keyboard_shortcut_active_keys.add(key)

        if self.keyboard_velocity_active_key is not None:
            self._stop_keyboard_jog_command()

        if key == "v":
            speed = next_keyboard_speed(
                self.keyboard_jog_linear_speed.get(), KEYBOARD_LINEAR_SPEEDS_MM_S
            )
            self.keyboard_jog_linear_speed.set(speed)
            self.keyboard_jog_status.set(f"XYZ speed {speed:g} mm/s")
            self.log(f"Keyboard XYZ speed selected · {speed:g} mm/s")
            return "break"
        if key == "x":
            speed = next_keyboard_speed(
                self.keyboard_jog_angular_speed.get(),
                KEYBOARD_ANGULAR_SPEEDS_DEG_S,
            )
            self.keyboard_jog_angular_speed.set(speed)
            self.keyboard_jog_status.set(f"Rotation speed {speed:g} deg/s")
            self.log(f"Keyboard rotation speed selected · {speed:g} deg/s")
            return "break"

        if registration is not None and key in ("i", "j"):
            expected = (
                "i"
                if registration.get("phase") == "waiting_start_capture"
                else "j"
                if registration.get("phase") == "waiting_goal_capture"
                else None
            )
            if key != expected:
                self.error(
                    f"Pass {registration['pass']} registration is in "
                    f"{registration.get('phase')} state; "
                    f"{(expected or 'no').upper()} capture is expected"
                )
                return "break"

        if self.sequence_running or self.node.active_motion_goal is not None:
            self.error("Cannot save a teaching pose during another motion")
            return "break"
        arm = self._selected_arm()
        if arm != self.keyboard_velocity_arm or self.keyboard_velocity_switching:
            self.error("Enable keyboard velocity mode for the selected arm first")
            return "break"
        if self.keyboard_teaching_capture_in_progress:
            self.error("Wait for the current keyboard teaching capture to finish")
            return "break"
        self.keyboard_teaching_capture_in_progress = True
        if registration is not None and key in ("i", "j"):
            endpoint = "START" if key == "i" else "GOAL"
            self._set_four_pass_status(
                f"Pass {registration['pass']} correction · {key.upper()} received · "
                f"capturing {endpoint}..."
            )
            self.pipeline_waiting(
                f"Pass {registration['pass']} {key.upper()} CAPTURE IN PROGRESS · "
                f"stopping and measuring {endpoint}"
            )
        self.keyboard_jog_status.set(
            f"{key.upper()} RECEIVED · stopping before pose capture..."
        )
        threading.Thread(
            target=self._keyboard_teaching_capture_worker,
            args=(arm, key),
            daemon=True,
        ).start()
        return "break"

    def keyboard_teaching_shortcut_release(self, event):
        key = str(event.keysym).lower()
        if key not in self.keyboard_shortcut_active_keys:
            return None
        old_timer = self.keyboard_shortcut_release_ids.pop(key, None)
        if old_timer is not None:
            try:
                self.root.after_cancel(old_timer)
            except tk.TclError:
                pass
        # X11 autorepeat emits synthetic release/press pairs. Delay removal so
        # holding a shortcut cannot cycle speeds or save repeatedly.
        self.keyboard_shortcut_release_ids[key] = self.root.after(
            50, lambda selected=key: self._finish_keyboard_shortcut_release(selected)
        )
        return "break"

    def _finish_keyboard_shortcut_release(self, key):
        self.keyboard_shortcut_release_ids.pop(key, None)
        self.keyboard_shortcut_active_keys.discard(key)

    def _keyboard_teaching_capture_worker(self, arm, key):
        registration = self.multi_pass_registration
        if registration is not None and key in ("i", "j"):
            planning_group = f"{arm}_manipulator"
            try:
                captured = self.node.capture_measured_teaching_snapshot(
                    planning_group,
                    f"multi_pass_{registration['pass']}_{'start' if key == 'i' else 'goal'}",
                )
            except Exception as error:
                self.post(
                    self._finish_multi_pass_keyboard_capture,
                    key, None, str(error),
                )
                return
            self.post(
                self._finish_multi_pass_keyboard_capture,
                key, captured, None,
            )
            return
        pose_name = KEYBOARD_TEACHING_POSE_SHORTCUTS.get(key)
        if pose_name is not None:
            planning_group = f"{arm}_manipulator"
            try:
                captured = self.node.capture_measured_teaching_snapshot(
                    planning_group, pose_name
                )
            except Exception as error:
                self.post(
                    self._finish_keyboard_named_pose_capture,
                    key, pose_name, planning_group, None, str(error),
                )
                return
            self.post(
                self._finish_keyboard_named_pose_capture,
                key, pose_name, planning_group, captured, None,
            )
            return
        if not self.node.wait_until_arm_stopped(arm, timeout=2.0):
            self.post(setattr, self, "keyboard_teaching_capture_in_progress", False)
            self.post(
                self.error,
                f"{key.upper()} teaching capture blocked: arm did not reach standstill",
            )
            return
        self.post(self._capture_keyboard_teaching_shortcut, arm, key)

    def _finish_keyboard_named_pose_capture(
        self, key, pose_name, planning_group, captured, error
    ):
        self.keyboard_teaching_capture_in_progress = False
        if error is not None:
            self.keyboard_jog_status.set(f"{key.upper()} · capture rejected")
            self.error(
                f"Keyboard {TEACHING_POSES[pose_name]} capture rejected: {error}"
            )
            return
        joint_names, positions, tcp, provenance = captured
        self.apply_initial_state(
            pose_name,
            planning_group,
            joint_names,
            positions,
            tcp,
            save_to_yaml=True,
            provenance=provenance,
        )
        self.keyboard_jog_status.set(
            f"{key.upper()} · SAVED {TEACHING_POSES[pose_name]}"
        )
        self.log(
            f"Keyboard teaching shortcut {key.upper()} · "
            f"SAVED {TEACHING_POSES[pose_name]}"
        )

    def _capture_keyboard_teaching_shortcut(self, arm, key):
        self.keyboard_teaching_capture_in_progress = False
        if (
            not self.keyboard_jog_enabled.get()
            or arm != self.keyboard_velocity_arm
            or arm != self._selected_arm()
            or self.sequence_running
            or self.node.active_motion_goal is not None
        ):
            self.error(f"{key.upper()} teaching capture canceled because state changed")
            return
        if key == "i":
            self.capture_linear_tcp(0)
            description = "Reference TCP 1"
        elif key == "j":
            self.capture_linear_tcp(1)
            description = "Reference TCP 2"
        else:
            return
        self.keyboard_jog_status.set(f"{key.upper()} · saving {description}")
        self.log(f"Keyboard teaching shortcut {key.upper()} · {description}")

    def keyboard_jog_selection_key(self, event):
        if not self.keyboard_jog_enabled.get():
            return None
        if not self._keyboard_focus_allows_jog():
            return None
        selection = {
            "1": "X",
            "2": "Y",
            "3": "Z",
            "4": "RX",
            "5": "RY",
            "6": "RZ",
            "7": "XY",
            "8": "XZ",
            "9": "YZ",
            "a": "RX/RY",
            "s": "RX/RZ",
            "d": "RY/RZ",
        }.get(str(event.keysym).lower())
        if selection is None:
            return None
        if self.keyboard_velocity_active_key is not None:
            self._stop_keyboard_jog_command()
        self.keyboard_jog_selection.set(selection)
        self.keyboard_jog_status.set(f"Selected {selection}")
        return "break"

    def keyboard_jog_key_press(self, event):
        pressed_at = time.monotonic()
        if self._keyboard_ros_input_online() and not self.keyboard_ros_dispatching:
            return "break"
        if not self.keyboard_jog_enabled.get():
            return None
        if event.keysym not in ("Left", "Right", "Up", "Down"):
            return None
        if self.sequence_running or self.node.active_motion_goal is not None:
            self.error("Keyboard teaching is unavailable during another motion")
            return "break"
        if self.keyboard_teaching_capture_in_progress:
            self.error("Keyboard motion is locked until pose capture finishes")
            return "break"
        arm = self._selected_arm()
        if arm != self.keyboard_velocity_arm or self.keyboard_velocity_switching:
            self.error("Enable keyboard velocity mode for the selected arm first")
            return "break"
        self._cancel_keyboard_release_timer()
        if self.keyboard_velocity_active_key == event.keysym:
            # X11 autorepeat renews the same deadman lease without sending a
            # new RB jog command.
            self.node.refresh_keyboard_velocity(arm)
            return "break"
        try:
            linear_speed_m_s = (
                float(self.keyboard_jog_linear_speed.get()) * 0.001
            )
            angular_speed_rad_s = math.radians(
                float(self.keyboard_jog_angular_speed.get())
            )
            velocity = self.node.resolve_keyboard_velocity(
                self.planning_group.get(),
                self.keyboard_jog_selection.get(),
                event.keysym,
                linear_speed_m_s,
                angular_speed_rad_s,
                self.keyboard_jog_frame.get(),
            )
        except (ValueError, tk.TclError) as error:
            self.error(str(error))
            return "break"
        except Exception as error:
            self.error(f"Keyboard velocity TF failed · {error}")
            return "break"
        self.keyboard_stop_generation += 1
        self.node.set_keyboard_velocity(arm, velocity)
        self.keyboard_velocity_active_key = event.keysym
        resolve_ms = (time.monotonic() - pressed_at) * 1000.0
        self.log(
            f"Keyboard jog START · {arm.upper()} "
            f"{self.keyboard_jog_selection.get()} {event.keysym} · "
            f"robot-base velocity=[{', '.join(f'{value:.6f}' for value in velocity)}] · "
            f"input-to-command={resolve_ms:.1f} ms"
        )
        self.keyboard_jog_status.set(
            f"MOVING {self.keyboard_jog_selection.get()} {event.keysym} · "
            "release to stop"
        )
        return "break"

    def keyboard_jog_key_release(self, event):
        if self._keyboard_ros_input_online() and not self.keyboard_ros_dispatching:
            return "break"
        if event.keysym != self.keyboard_velocity_active_key:
            return None
        self._cancel_keyboard_release_timer()
        # X11 key repeat can emit a synthetic release/press pair.  A repeated
        # press cancels this short timer; the final physical release does not.
        self.keyboard_release_after_id = self.root.after(
            35,
            lambda selected=event.keysym: self._finish_keyboard_key_release(
                selected
            ),
        )
        return "break"

    def _finish_keyboard_key_release(self, key_name):
        self.keyboard_release_after_id = None
        if key_name != self.keyboard_velocity_active_key:
            return
        arm = self.keyboard_velocity_arm
        self._stop_keyboard_jog_command()
        self.log(f"Keyboard jog STOP · {(arm or 'robot').upper()} velocity zero")
        self.keyboard_jog_status.set(
            f"STOPPED · {(arm or 'robot').upper()} velocity zero"
        )

    def _disable_keyboard_velocity_async(self):
        self._stop_keyboard_wire()
        arm = self.keyboard_velocity_arm
        self._cancel_keyboard_release_timer()
        for timer in self.keyboard_shortcut_release_ids.values():
            try:
                self.root.after_cancel(timer)
            except tk.TclError:
                pass
        self.keyboard_shortcut_release_ids.clear()
        self.keyboard_shortcut_active_keys.clear()
        self._stop_keyboard_jog_command()
        self.keyboard_jog_enabled.set(False)
        if arm is None or self.keyboard_velocity_switching:
            return
        self.keyboard_velocity_switching = True
        self._set_keyboard_jog_enable_state(tk.DISABLED)
        threading.Thread(
            target=self._keyboard_velocity_mode_worker,
            args=(arm, False),
            daemon=True,
        ).start()

    def emergency_stop_all(self, restore_keyboard_controller=True):
        """Stop every GUI-owned workflow, robot goal, and welder output."""
        self.multi_pass_registration = None
        self._stop_keyboard_wire()
        self._stop_keyboard_jog_command()
        self.keyboard_jog_enabled.set(False)
        self.keyboard_jog_status.set("Keyboard velocity ZERO sent")
        if restore_keyboard_controller:
            self._disable_keyboard_velocity_async()
        self.seam_auto_running = False
        self.seam_auto_expected_kind = None
        self.seam_auto_stage_success = False
        self.seam_auto_stage_event.set()
        self.automatic_probe_kind = None
        self.node.clear_touch_probe()
        self.node.active_touch_guard = None
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        self.auto_seam_correction_button.configure(state=tk.NORMAL)
        self.stop_auto_seam_button.configure(state=tk.DISABLED)
        self.corner_touch_status.configure(
            text="EMERGENCY STOP requested · all GUI motion workflows aborted"
        )
        self.root.bell()
        self.stop_sequence()
        self.pipeline_waiting(
            "EMERGENCY STOP (SOFTWARE) · ALL ROBOT MOTION + WELDER STOP REQUESTED"
        )

    def sequence_hard_stop_finished(self, results):
        message = " · ".join(results) if results else "no connected arm goal"
        self.sequence_status.configure(text=f"STOP NOW complete · {message}")
        self.pipeline_result(
            f"STOP NOW COMPLETE · welder outputs inhibited · {message}"
        )

    def arm_changed(self, *_args):
        if not hasattr(self, "node"):
            return
        if self.keyboard_velocity_arm is not None:
            self._disable_keyboard_velocity_async()
        group = self.planning_group.get()
        if group != "right_manipulator":
            self.clear_hicomm_test_outputs()
            self._set_welder_test_controls(False)
        else:
            self._set_welder_test_controls(self.hicomm_connected)
        self.linear_tcp_endpoints = [None, None]
        self.reference_yaw_status.set("Reference yaw: --")
        self.reference_length_status.set("Length: --")
        self.sensed_yaw_status.set("Sensed yaw: --")
        self.delta_yaw_status.set("ΔYaw: --")
        self.path_kind = "empty"
        self.weave_source = []
        self.weave_base_paths = {"linear": [], "circle": [], "corrected": []}
        self.initial_joint_state = None
        self.initial_plan_ready = False
        self.plan_initial_button.configure(state=tk.DISABLED)
        self.execute_initial_button.configure(state=tk.DISABLED)
        self.initial_state_status.configure(text="not captured")
        self.taught_robot_poses = {name: None for name in TEACHING_POSES}
        self.teaching_capture_provenance = {}
        self.set_points([])
        self.node.publish_points([], self.show_path.get())
        self._auto_load_teaching_states()
        self._refresh_execution_controls()
        self.log(f"Cartesian arm changed to {group} · path cleared")

    def _selected_arm(self):
        return (
            "left"
            if self.planning_group.get() == "left_manipulator"
            else "right"
        )

    def _selected_robot_connected(self):
        return self.robot_connected[self._selected_arm()]

    def _refresh_execution_controls(self):
        selected_arm = self._selected_arm()
        connected = self.robot_connected[selected_arm]
        for arm in ("left", "right"):
            value = self.robot_connected[arm]
            self.robot_connection_labels[arm].configure(
                text=(
                    f"Connect {arm.upper()} ({self.robot_ips[arm]}): "
                    f"{'O' if value else 'X'}"
                ),
                bg="#e6f4ea" if value else "#fce8e6",
                fg="#137333" if value else "#b3261e",
            )
        head_connected = self.robot_connected["head"]
        head_kind = "FAKE" if self.fake_head_hardware else "CAN2"
        self.robot_connection_labels["head"].configure(
            text=(
                f"Connect HEAD ({head_kind}): "
                f"{'O' if head_connected else 'X'}"
            ),
            bg="#e6f4ea" if head_connected else "#fce8e6",
            fg="#137333" if head_connected else "#b3261e",
        )
        self.plan_button.configure(
            state=tk.NORMAL if self.points and connected else tk.DISABLED
        )
        self.execute_button.configure(
            state=(
                tk.NORMAL
                if (
                    self.plan_approved
                    and self.execution_allowed
                    and connected
                )
                else tk.DISABLED
            )
        )
        self._refresh_initial_position_controls()
        self._refresh_wide_sensing_controls()

    def _refresh_wide_sensing_controls(self):
        if not hasattr(self, "wide_sensing_plan_button"):
            return
        usable = bool(
            self.latest_wide_sensing_result is not None
            and self.latest_wide_sensing_result.success
            and self.wide_sensing_segments
        )
        connected = self._selected_robot_connected()
        self.wide_sensing_load_button.configure(
            state=tk.NORMAL if usable else tk.DISABLED
        )
        self.wide_sensing_plan_button.configure(
            state=tk.NORMAL if usable and connected else tk.DISABLED
        )
        can_execute = bool(
            str(self.path_kind).startswith("wide_sensing:")
            and self.plan_approved
            and self.execution_allowed
            and connected
        )
        self.wide_sensing_execute_button.configure(
            state=tk.NORMAL if can_execute else tk.DISABLED
        )

    def _refresh_initial_position_controls(self):
        if not hasattr(self, "plan_initial_button"):
            return
        can_plan = (
            self.initial_joint_state is not None
            and self._selected_robot_connected()
        )
        self.plan_initial_button.configure(
            state=tk.NORMAL if can_plan else tk.DISABLED
        )
        can_execute = (
            self.initial_plan_ready
            and self.initial_joint_state is not None
            and self.execution_allowed
            and self._selected_robot_connected()
        )
        self.execute_initial_button.configure(
            state=tk.NORMAL if can_execute else tk.DISABLED
        )

    def log(self, text):
        if text.startswith("ERROR") or " · FAILED · " in text:
            message = text.removeprefix("ERROR · ")
            self._set_pipeline_status("ERROR", message)
        elif text.startswith(("SUCCESS", "RESULT")):
            self._set_pipeline_status("RESULT", text)
        else:
            self._set_pipeline_status("WAITING", text)

    def _set_pipeline_status(self, state, message):
        if state == "ERROR":
            self.latest_pipeline_error = str(message)
        colors = {
            "WAITING": ("#eeeeee", "#202124"),
            "ERROR": ("#fce8e6", "#b3261e"),
            "RESULT": ("#e6f4ea", "#137333"),
        }
        background, foreground = colors[state]
        self.pipeline_status.configure(
            text=f"{state} · {message}",
            bg=background,
            fg=foreground,
        )
        terminal_message = f"PIPELINE {state} · {message}"
        if hasattr(self, "node"):
            logger = self.node.get_logger()
            if state == "ERROR":
                logger.error(terminal_message)
            elif state == "RESULT":
                logger.info(terminal_message)
            else:
                logger.info(terminal_message)
        else:
            print(terminal_message, flush=True)

    def pipeline_waiting(self, message):
        self._set_pipeline_status("WAITING", message)

    def pipeline_result(self, message):
        self._set_pipeline_status("RESULT", message)

    def error(self, text):
        self.log(f"ERROR · {text}")
        self.plan_approved = False
        state = (
            tk.NORMAL
            if self.points and self._selected_robot_connected()
            else tk.DISABLED
        )
        self.plan_button.configure(state=state)
        self.execute_button.configure(state=tk.DISABLED)
        if hasattr(self, "wide_sensing_execute_button"):
            self.wide_sensing_execute_button.configure(state=tk.DISABLED)

    @staticmethod
    def _pose_values(pose):
        p, q = pose.position, pose.orientation
        return (p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    def set_points(self, points, selected_index=0):
        self.invalidate_approved_plan()
        self.points = copy.deepcopy(list(points))
        self.table.delete(*self.table.get_children())
        for index, pose in enumerate(self.points, 1):
            values = tuple(f"{value:.5f}" for value in self._pose_values(pose))
            self.table.insert("", tk.END, values=(index,) + values)
        self.plan_button.configure(
            state=(
                tk.NORMAL
                if self.points and self._selected_robot_connected()
                else tk.DISABLED
            ),
        )
        children = self.table.get_children()
        if children:
            selected_index = min(max(selected_index, 0), len(children) - 1)
            self.table.selection_set(children[selected_index])
            self.table.focus(children[selected_index])
            self.table.see(children[selected_index])
        self.path_summary.configure(
            text=f"{self.path_kind} · {len(self.points)} poses"
        )

    def set_new_points(self, points, kind):
        if kind != "weave":
            self.weave_source = copy.deepcopy(list(points))
        if kind == "circle":
            self.weave_base_paths["circle"] = copy.deepcopy(list(points))
        elif kind == "tcp_line":
            self.weave_base_paths["linear"] = copy.deepcopy(list(points))
        self.path_kind = kind
        self.set_points(points)

    def update_wide_sensing_result(self, message):
        """Display the newest detected weld segments without moving a robot."""
        self.latest_wide_sensing_result = message
        if str(message.frame_id).strip():
            self.wide_sensing_source_frame.set(str(message.frame_id).strip())
        segments = {}
        for index, segment in enumerate(message.weld_segments, 1):
            segment_id = str(segment.id).strip() or f"segment_{index}"
            unique_id = segment_id
            duplicate = 2
            while unique_id in segments:
                unique_id = f"{segment_id}#{duplicate}"
                duplicate += 1
            segments[unique_id] = copy.deepcopy(segment)
        self.wide_sensing_segments = segments
        ids = tuple(segments)
        self.wide_sensing_segment_box.configure(values=ids)
        if ids and self.wide_sensing_segment_id.get() not in segments:
            self.wide_sensing_segment_id.set(ids[0])
        if not ids:
            self.wide_sensing_segment_id.set("")
        self._refresh_wide_sensing_controls()
        self.wide_sensing_status.set(
            f"{message.status or 'result'} · success={bool(message.success)} · "
            f"segments={len(ids)} · {message.message}"
            + (
                " · selected robot not ready: ACTIVATE BOTH first"
                if ids and not self._selected_robot_connected()
                else ""
            )
        )
        self.log(
            "Wide Sensing result received · "
            f"success={bool(message.success)} · segments={len(ids)} · "
            f"{message.message}"
        )

    def load_wide_sensing_segment(self, plan_after_load=False):
        """Resolve one sensed segment to World and load it for normal planning."""
        segment_id = self.wide_sensing_segment_id.get()
        segment = self.wide_sensing_segments.get(segment_id)
        if segment is None:
            self.error("Select a valid Wide Sensing weld segment")
            return
        if self.sequence_running:
            self.error("Cannot replace the path while a sequence is running")
            return
        planning_group = self.planning_group.get()
        if planning_group not in PLANNING_GROUP_TIPS:
            self.error("Wide Sensing requires a left or right manipulator")
            return
        try:
            offset_m = tuple(
                float(variable.get()) * 0.001
                for variable in (
                    self.wide_sensing_offset_x_mm,
                    self.wide_sensing_offset_y_mm,
                    self.wide_sensing_offset_z_mm,
                )
            )
        except (ValueError, tk.TclError):
            self.error("Wide Sensing World offset must be numeric")
            return
        self.wide_sensing_load_button.configure(state=tk.DISABLED)
        self.wide_sensing_plan_button.configure(state=tk.DISABLED)
        self.wide_sensing_status.set(
            f"Resolving {self.wide_sensing_source_frame.get()} → World TF..."
        )
        threading.Thread(
            target=self._wide_sensing_segment_worker,
            args=(
                segment_id,
                copy.deepcopy(segment),
                self.wide_sensing_source_frame.get(),
                planning_group,
                offset_m,
                bool(self.wide_sensing_reverse.get()),
                bool(plan_after_load),
            ),
            daemon=True,
        ).start()

    def _wide_sensing_segment_worker(
        self,
        segment_id,
        segment,
        source_frame,
        planning_group,
        offset_m,
        reverse,
        plan_after_load,
    ):
        try:
            poses = self.node.resolve_wide_sensing_segment(
                segment,
                source_frame,
                planning_group,
                offset_m,
                reverse,
            )
        except Exception as error:
            self.post(
                self._wide_sensing_segment_result,
                segment_id,
                None,
                str(error),
                plan_after_load,
            )
            return
        self.post(
            self._wide_sensing_segment_result,
            segment_id,
            poses,
            "",
            plan_after_load,
        )

    def _wide_sensing_segment_result(
        self, segment_id, poses, error, plan_after_load=False
    ):
        self._refresh_wide_sensing_controls()
        if poses is None:
            self.wide_sensing_status.set(f"TF/path conversion failed · {error}")
            self.error(f"Wide Sensing segment conversion failed: {error}")
            return
        self.set_new_points(poses, f"wide_sensing:{segment_id}")
        start, end = poses
        length_mm = math.dist(
            (start.position.x, start.position.y, start.position.z),
            (end.position.x, end.position.y, end.position.z),
        ) * 1000.0
        self.wide_sensing_status.set(
            f"Loaded {segment_id} in World · length={length_mm:.2f} mm · "
            "current TCP orientation preserved · Plan Preview required"
        )
        self.log(
            f"Wide Sensing segment loaded · {segment_id} · "
            f"START=({start.position.x:.6f}, {start.position.y:.6f}, "
            f"{start.position.z:.6f}) m · "
            f"END=({end.position.x:.6f}, {end.position.y:.6f}, "
            f"{end.position.z:.6f}) m · length={length_mm:.2f} mm"
        )
        self._refresh_wide_sensing_controls()
        if plan_after_load:
            self.plan_preview()

    def set_execution_configuration(
        self,
        execute_motion,
        left_ip,
        right_ip,
        use_fake_head_hardware,
        hicomm_source_ip,
        hicomm_welder_ip,
        hicomm_port,
    ):
        self.execution_allowed = execute_motion
        self.fake_head_hardware = bool(use_fake_head_hardware)
        self.robot_ips = {"left": left_ip, "right": right_ip}
        self.hicomm_source_ip.set(hicomm_source_ip)
        self.hicomm_welder_ip.set(hicomm_welder_ip)
        self.hicomm_port.set(int(hicomm_port))
        self.robot_connected = {
            "left": False,
            "right": False,
            "head": False,
        }
        self._refresh_execution_controls()
        head_kind = "FAKE" if self.fake_head_hardware else "CAN2"
        self.log(
            f"Connecting LEFT {left_ip} + RIGHT {right_ip} · "
            f"HEAD {head_kind} · waiting for measured feedback and "
            "controller readiness · Hi-COMM waits for Connect"
        )

    def set_fastech_configuration(self, ip_address, board_id, poll_period_s):
        """Display the owner-node launch configuration as read-only GUI state."""
        self.fastech_ip.set(str(ip_address))
        self.fastech_board_id.set(int(board_id))
        period = max(0.001, float(poll_period_s))
        self.fastech_poll_rate_hz = 1.0 / period
        self.fastech_io_status.configure(
            text=(
                f"Waiting for /fastech/io_state · {ip_address} · board "
                f"{int(board_id)} · {self.fastech_poll_rate_hz:.0f} Hz"
            )
        )

    def robot_feedback_connected(self, arm):
        self.robot_connected[arm] = True
        self._refresh_execution_controls()
        description = "head" if arm == "head" else f"{arm}-arm"
        self.log(f"READY · {description} feedback and controller available")

    def robot_feedback_lost(self, arm, detail="measured joint feedback timeout"):
        self.robot_connected[arm] = False
        if self._selected_arm() == arm:
            self.invalidate_approved_plan()
            self.initial_plan_ready = False
        self._refresh_execution_controls()
        if self._selected_arm() == arm:
            self.plan_button.configure(state=tk.DISABLED)
        description = "head" if arm == "head" else f"{arm}-arm"
        self.log(f"ERROR · {description} unavailable · {detail}")

    def invalidate_approved_plan(self):
        self.plan_approved = False
        if hasattr(self, "execute_button"):
            self.execute_button.configure(state=tk.DISABLED)
        if hasattr(self, "wide_sensing_execute_button"):
            self.wide_sensing_execute_button.configure(state=tk.DISABLED)

    def selected_index(self):
        selection = self.table.selection()
        if not selection:
            return None
        return int(self.table.item(selection[0], "values")[0]) - 1

    def load_selected(self, _event=None):
        index = self.selected_index()
        if index is None:
            return
        for name, value in zip(
            self.POSE_FIELDS,
            self._pose_values(self.points[index]),
        ):
            self.pose_variables[name].set(f"{value:.6f}")

    def publish_edits(self, selected_index):
        if self.path_kind != "weave":
            self.weave_source = copy.deepcopy(self.points)
        self.set_points(self.points, selected_index)
        self.node.publish_points(self.points, self.show_path.get())
        self.log(f"Published edited path · {len(self.points)} poses")

    def toggle_path_visibility(self):
        if (
            self.path_kind == "di8_four_touch_raw"
            and self.raw_two_touch_seam
            and self.corrected_two_touch_seam
        ):
            self.node.publish_seam_comparison(
                self.raw_two_touch_seam,
                self.corrected_two_touch_seam,
                self.show_path.get(),
            )
        else:
            self.node.publish_points(self.points, self.show_path.get())
        state = "ON" if self.show_path.get() else "OFF"
        self.log(f"Planned path visualization {state}")

    def apply_selected(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        try:
            values = [
                float(self.pose_variables[name].get())
                for name in self.POSE_FIELDS
            ]
        except ValueError:
            self.error("Pose fields must be numeric")
            return
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = values[:3]
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = values[3:]
        if not pose_is_valid(pose):
            self.error("Pose must be finite with a non-zero quaternion")
            return
        self.points[index] = pose
        self.publish_edits(index)

    def duplicate_selected(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        self.points.insert(index + 1, copy.deepcopy(self.points[index]))
        self.publish_edits(index + 1)

    def delete_selected(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        self.points.pop(index)
        self.publish_edits(max(0, index - 1))

    def move_selected(self, offset):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        destination = index + offset
        if destination < 0 or destination >= len(self.points):
            return
        self.points[index], self.points[destination] = (
            self.points[destination],
            self.points[index],
        )
        self.publish_edits(destination)

    def nudge(self, axis, direction):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint first")
            return
        try:
            distance = float(self.nudge_mm.get()) * 0.001 * direction
        except (ValueError, tk.TclError):
            self.error("Nudge distance must be numeric")
            return
        position = self.points[index].position
        setattr(position, axis, getattr(position, axis) + distance)
        self.publish_edits(index)

    # Acquire a straight seam from an axis and a World/tool reference frame.
    def acquire(self):
        try:
            reference = self.straight_reference.get()
            direction = self.straight_axis.get()
            axis = direction[-1].lower()
            sign = -1.0 if direction.startswith("-") else 1.0
            distance = (
                float(self.straight_distance_mm.get()) * 0.001 * sign
            )
            count = int(self.straight_count.get())
            rpy_offset = tuple(
                math.radians(float(variable.get()))
                for variable in (
                    self.straight_roll_deg,
                    self.straight_pitch_deg,
                    self.straight_yaw_deg,
                )
            )
            explicit_position = None
            if self.straight_start_mode.get() == "World XYZ":
                explicit_position = (
                    float(self.straight_start_x.get()),
                    float(self.straight_start_y.get()),
                    float(self.straight_start_z.get()),
                )
        except (ValueError, tk.TclError):
            self.error(
                "Straight position/distance/count/RPY must be numeric"
            )
            return
        self.log(
            f"Reading current {self.planning_group.get()} TCP and generating "
            f"{reference} {direction} straight seam · orientation rotation "
            f"reference={self.straight_rotation_reference.get()}"
        )
        threading.Thread(
            target=self.node.acquire_points,
            args=(
                reference,
                axis,
                distance,
                count,
                explicit_position,
                rpy_offset,
                self.straight_rotation_reference.get(),
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def generate_circle(self):
        try:
            radius = float(self.radius_mm.get()) * 0.001
            count = int(self.circle_count.get())
        except (ValueError, tk.TclError):
            self.error("Circle radius/count must be numeric")
            return
        threading.Thread(
            target=self.node.generate_circle,
            args=(
                self.circle_axis.get().lower(),
                radius,
                count,
                bool(self.close_circle.get()),
                bool(self.circle_face_center.get()),
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def generate_weave(self):
        base_kind = self.weave_base.get()
        source = self.weave_base_paths.get(base_kind, [])
        transverse_vector = self.sensed_weave_transverse_vector()
        if transverse_vector is not None and base_kind == "linear":
            # Weave the seam that will actually be welded.  Adopting a
            # touch-corrected seam publishes it to the path table but does not
            # register it as a weave base, so without this the preview weaves
            # whatever straight line was generated before the touch probing --
            # the taught seam, not the corrected one.
            corrected = self.weave_base_paths.get("corrected") or []
            if len(corrected) >= 2:
                source = corrected
        if len(source) < 2:
            self.error(
                f"Generate a {base_kind} base path before applying weave"
            )
            return
        try:
            amplitude = float(self.weave_amplitude_mm.get()) * 0.001
            pitch_mm = float(self.weave_pitch_mm.get())
            samples = WELD_WEAVE_SAMPLES_PER_CYCLE
        except (ValueError, tk.TclError):
            self.error("Weave settings must be numeric")
            return
        self.weave_source = copy.deepcopy(source)
        seam_length = sum(
            (
                (
                    second.position.x - first.position.x
                ) ** 2
                + (
                    second.position.y - first.position.y
                ) ** 2
                + (
                    second.position.z - first.position.z
                ) ** 2
            ) ** 0.5
            for first, second in zip(source[:-1], source[1:])
        )
        try:
            cycles = weave_cycles_for_pitch(seam_length, pitch_mm)
        except ValueError as error:
            self.error(str(error))
            return
        actual_pitch_mm = seam_length * 1000.0 / cycles
        # Say which plane the weave is about.  Reading "sensed e_w" here is how
        # the operator knows the preview is the touch-corrected weld and not a
        # generic-axis stand-in for it.
        plane = (
            f"axis {self.weave_axis.get()}"
            if transverse_vector is None
            else "sensed e_w (touch-corrected)"
        )
        self.weave_summary.configure(
            text=(
                f"{self.weave_pattern.get()} "
                f"{'±' if self.weave_pattern.get() in ('sine', 'crescent') else 'R='}"
                f"{amplitude * 1000.0:.1f} mm · "
                f"pitch≤{pitch_mm:.1f} mm (actual {actual_pitch_mm:.2f}) · "
                f"{cycles} cycles · {plane}"
            )
        )
        threading.Thread(
            target=self.node.generate_weave,
            args=(
                copy.deepcopy(source),
                amplitude,
                cycles,
                samples,
                self.weave_axis.get(),
                self.weave_pattern.get(),
                self.show_path.get(),
                transverse_vector,
            ),
            daemon=True,
        ).start()

    def append_tcp(self):
        threading.Thread(
            target=self.node.capture_tcp,
            args=(
                None,
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def capture_initial_state(self):
        pose_name = self._selected_teaching_pose_name()
        self.pipeline_waiting(
            f"Capturing {TEACHING_POSES[pose_name]} and measured joint angles"
        )
        threading.Thread(
            target=self.node.capture_initial_state,
            args=(self.planning_group.get(), pose_name),
            daemon=True,
        ).start()

    def _selected_teaching_pose_name(self):
        selected_label = self.teaching_pose_name.get()
        name = next(
            name
            for name, label in TEACHING_POSES.items()
            if label == selected_label
        )
        return self.teaching_state.select(name)

    def teaching_pose_changed(self, _event=None):
        pose_name = self._selected_teaching_pose_name()
        stored = self.taught_robot_poses[pose_name]
        self.initial_plan_ready = False
        self.node.initial_planned_trajectory = None
        if stored is None:
            self.initial_joint_state = None
            self.initial_state_status.configure(
                text=f"{TEACHING_POSES[pose_name]}: not captured"
            )
        else:
            group, names, positions, tcp = stored
            self.initial_joint_state = (group, names, positions)
            angles = ", ".join(
                f"{math.degrees(value):.1f}°" for value in positions
            )
            tcp_values = self._pose_values(tcp)
            self.initial_state_status.configure(
                text=(
                    f"{TEACHING_POSES[pose_name]} · TCP "
                    f"({tcp_values[0]:.4f}, {tcp_values[1]:.4f}, "
                    f"{tcp_values[2]:.4f}) m · joints {angles}"
                )
            )
        self._refresh_initial_position_controls()

    def _initial_state_yaml_path(self, planning_group=None, pose_name=None):
        from construct_robot.teaching_paths import teaching_config_dir
        group = planning_group or self.planning_group.get()
        selected_pose = pose_name or self._selected_teaching_pose_name()
        return teaching_config_dir() / f"{group}_{selected_pose}_state.yaml"

    def _seam_reference_yaml_path(self, planning_group=None):
        from construct_robot.teaching_paths import teaching_config_dir
        group = planning_group or self.planning_group.get()
        return teaching_config_dir() / f"{group}_seam_teaching_reference.yaml"

    def _seam_touch_yaml_path(self, planning_group=None):
        from construct_robot.teaching_paths import teaching_config_dir
        group = planning_group or self.planning_group.get()
        return teaching_config_dir() / f"{group}_seam_touch_points.yaml"

    def _auto_load_teaching_states(self):
        """Load every named teaching pose found at its default YAML path."""
        planning_group = self.planning_group.get()
        selected_pose = self._selected_teaching_pose_name()
        loaded = []
        for pose_name in TEACHING_POSES:
            path = self._initial_state_yaml_path(planning_group, pose_name)
            if not path.is_file():
                continue
            try:
                group, joint_names, positions, tcp = load_initial_state_yaml(
                    path
                )
            except (OSError, ValueError, yaml.YAMLError) as error:
                self.log(
                    f"Skipped invalid teaching YAML {path.name}: {error}"
                )
                continue
            if group != planning_group:
                self.log(
                    f"Skipped teaching YAML {path.name}: expected "
                    f"{planning_group}, got {group}"
                )
                continue
            self.taught_robot_poses[pose_name] = (
                group,
                tuple(joint_names),
                tuple(positions),
                copy.deepcopy(tcp),
            )
            try:
                provenance = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader).get(
                    "capture_provenance")
                if isinstance(provenance, dict):
                    self.teaching_capture_provenance[pose_name] = provenance
            except (OSError, yaml.YAMLError, AttributeError):
                pass
            loaded.append(TEACHING_POSES[pose_name])

        reference_path = self._seam_reference_yaml_path(planning_group)
        if reference_path.is_file():
            try:
                reference_group, reference_poses = (
                    load_seam_teaching_reference_yaml(reference_path)
                )
                if reference_group != planning_group:
                    raise ValueError(
                        f"reference group is {reference_group}, expected "
                        f"{planning_group}"
                    )
                self.seam_teaching_reference = {}
                for name, pose in reference_poses.items():
                    stored = self.taught_robot_poses.get(name)
                    if stored is not None:
                        self.seam_teaching_reference[name] = (
                            stored[0], stored[1], stored[2], copy.deepcopy(pose)
                        )
                for index, pose_name in enumerate(("weld_start", "weld_end")):
                    stored_reference = self.seam_teaching_reference.get(pose_name)
                    if stored_reference is not None:
                        pose = copy.deepcopy(stored_reference[3])
                        self.linear_tcp_endpoints[index] = pose
                self._update_seam_yaw_status()
                self.log(f"Loaded seam teaching reference from {reference_path}")
            except (OSError, ValueError, yaml.YAMLError, KeyError) as error:
                self.error(f"Seam teaching reference load failed: {error}")

        self.teaching_pose_name.set(TEACHING_POSES[selected_pose])
        self.teaching_pose_changed()
        if loaded:
            self.log(
                f"Auto-loaded {len(loaded)} teaching YAML pose(s) for "
                f"{planning_group}: {', '.join(loaded)}"
            )
            self._verify_loaded_teaching_poses_async(
                {name: copy.deepcopy(self.taught_robot_poses[name])
                 for name in TEACHING_POSES
                 if self.taught_robot_poses[name] is not None})

    def _verify_loaded_teaching_poses_async(self, poses):
        """Check legacy YAML q/TCP pairs after ROS services become available."""
        def verify():
            if not self.node.fk_client.wait_for_service(timeout_sec=15.0):
                self.post(self.log, "Teaching YAML FK verification deferred: /compute_fk unavailable")
                return
            for name, stored in poses.items():
                try:
                    self.node.validate_named_pose_recall(name, *stored)
                except (RuntimeError, ValueError, TransformException) as error:
                    self.post(self.error,
                              f"Loaded {name} is inconsistent and cannot be recalled: {error}")

        threading.Thread(target=verify, daemon=True).start()

    def load_initial_state(self):
        pose_name = self._selected_teaching_pose_name()
        default_path = self._initial_state_yaml_path(pose_name=pose_name)
        path = filedialog.askopenfilename(
            title="Load TCP teaching state",
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
            filetypes=(("YAML", "*.yaml *.yml"), ("All files", "*.*")),
        )
        if not path:
            return
        self.load_initial_state_from_path(path, pose_name)

    def load_initial_state_from_path(self, path, pose_name):
        """Apply a selected teaching YAML through the production invalidation path."""
        try:
            planning_group, joint_names, positions, tcp = (
                load_initial_state_yaml(path)
            )
            loaded_document = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=yaml.CSafeLoader) or {}
            provenance = loaded_document.get("capture_provenance")
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.error(f"Failed to load initial state YAML: {error}")
            return
        if planning_group != self.planning_group.get():
            self.error(
                f"YAML is for {planning_group}; selected arm is "
                f"{self.planning_group.get()}"
            )
            return
        self.apply_initial_state(
            pose_name,
            planning_group,
            joint_names,
            positions,
            tcp,
            save_to_yaml=False,
            provenance=provenance if isinstance(provenance, dict) else None,
        )
        self._verify_loaded_teaching_poses_async({
            pose_name: copy.deepcopy(self.taught_robot_poses[pose_name])})
        self.log(f"Loaded TCP teaching state from {path}")

    def apply_initial_state(
        self,
        pose_name,
        planning_group,
        joint_names,
        positions,
        tcp,
        save_to_yaml=True,
        provenance=None,
    ):
        if pose_name not in TEACHING_POSES:
            self.error(f"Unknown teaching pose: {pose_name}")
            return
        if pose_name in (
            "weld_start_wait", "weld_start", "weld_goal_wait", "weld_end"
        ):
            self._invalidate_seam_correction_runtime(
                f"re-taught {TEACHING_POSES[pose_name]}", clear_touches=True
            )
        self.teaching_pose_name.set(TEACHING_POSES[pose_name])
        self.taught_robot_poses[pose_name] = (
            planning_group,
            tuple(joint_names),
            tuple(positions),
            copy.deepcopy(tcp),
        )
        if provenance is not None:
            self.teaching_capture_provenance[pose_name] = copy.deepcopy(provenance)
        else:
            self.teaching_capture_provenance.pop(pose_name, None)
        if pose_name in SEAM_REFERENCE_TEACHING_POSES:
            if self.seam_teaching_reference is None:
                self.seam_teaching_reference = {}
            self.seam_teaching_reference[pose_name] = (
                planning_group,
                tuple(joint_names),
                tuple(positions),
                copy.deepcopy(tcp),
            )
            try:
                save_seam_teaching_reference_yaml(
                    self._seam_reference_yaml_path(planning_group),
                    planning_group,
                    self.seam_teaching_reference,
                )
            except (OSError, ValueError, yaml.YAMLError) as error:
                self.error(f"Seam teaching reference save failed: {error}")
        self.initial_joint_state = (
            planning_group,
            tuple(joint_names),
            tuple(positions),
        )
        self.initial_plan_ready = False
        angles = ", ".join(f"{math.degrees(value):.1f}°" for value in positions)
        self.initial_state_status.configure(
            text=f"{TEACHING_POSES[pose_name]}: {angles}"
        )
        self._refresh_initial_position_controls()
        if pose_name in ("weld_wait", "weld_start_wait", "weld_goal_wait", "weld_finish"):
            self.quick_teaching_status.set(
                f"Saved {TEACHING_POSES[pose_name]}"
            )
        values = self._pose_values(tcp)
        saved_message = ""
        save_error = None
        if save_to_yaml:
            path = self._initial_state_yaml_path(planning_group, pose_name)
            try:
                save_initial_state_yaml(
                    path,
                    planning_group,
                    joint_names,
                    positions,
                    tcp,
                    provenance,
                )
                saved_message = f" · saved to {path}"
            except (OSError, ValueError, yaml.YAMLError) as error:
                save_error = error
        self.pipeline_result(
            f"{TEACHING_POSES[pose_name]} captured · TCP World XYZ="
            f"({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}) m"
            f"{saved_message}"
        )
        if save_error is not None:
            self.error(
                f"Initial state captured, but YAML save failed: {save_error}"
            )

    def plan_initial_state(self):
        if self.keyboard_velocity_arm is not None or self.keyboard_velocity_switching:
            self.error("Disable Keyboard Teaching and wait for controller handover before planning")
            return
        if self.initial_joint_state is None:
            self.error("Capture or load the selected robot pose first")
            return
        if not self._selected_robot_connected():
            self.error("Connect the selected REAL RB robot first")
            return
        pose_name = self._selected_teaching_pose_name()
        group, joint_names, positions = self.initial_joint_state
        stored = self.taught_robot_poses.get(pose_name)
        target_tcp = copy.deepcopy(stored[3]) if stored is not None else None
        if group != self.planning_group.get():
            self.error("Selected taught pose belongs to another arm")
            return
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        threading.Thread(
            target=self.node.plan_initial_state,
            args=(
                group,
                joint_names,
                positions,
                max(0.01, min(1.0, self.velocity_percent.get() / 100.0)),
                pose_name,
                target_tcp,
            ),
            daemon=True,
        ).start()

    def initial_position_plan_ready(
        self,
        planning_group,
        target_positions,
        velocity_scale,
        message,
    ):
        if self.initial_joint_state is None:
            return
        group, _joint_names, positions = self.initial_joint_state
        if (
            group != planning_group
            or tuple(positions) != tuple(target_positions)
            or group != self.planning_group.get()
            or not math.isclose(
                velocity_scale,
                max(0.01, min(1.0, self.velocity_percent.get() / 100.0)),
            )
        ):
            self.log("Discarded stale taught-pose plan")
            return
        self.initial_plan_ready = True
        self._refresh_initial_position_controls()
        self.pipeline_result(message)

    def execute_initial_plan(self):
        if self.keyboard_velocity_arm is not None or self.keyboard_velocity_switching:
            self.error("Disable Keyboard Teaching and replan before executing the taught pose")
            return
        if not self.initial_plan_ready:
            self.error(
                "Plan and inspect the selected taught-pose trajectory first"
            )
            return
        if not self.execution_allowed:
            self.error("Robot execution is disabled by launch configuration")
            return
        if not self._selected_robot_connected():
            self.error("Connect the selected REAL RB robot first")
            return
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        threading.Thread(
            target=self.node.execute_initial_plan,
            daemon=True,
        ).start()

    def initial_position_execution_finished(self, message):
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        self.pipeline_result(message)

    def capture_linear_tcp(self, endpoint_index):
        role = "START" if endpoint_index == 0 else "GOAL"
        self.log(
            f"Teaching reference TCP {endpoint_index + 1} / {role} from "
            f"current {self.planning_group.get()} TCP..."
        )
        threading.Thread(
            target=self.node.capture_linear_tcp,
            args=(endpoint_index, self.planning_group.get()),
            daemon=True,
        ).start()

    def apply_linear_tcp(
        self,
        endpoint_index,
        pose,
        planning_group=None,
        joint_names=None,
        positions=None,
        provenance=None,
    ):
        """Store TCP1/TCP2 as persistent nominal seam reference teaching."""
        planning_group = planning_group or self.planning_group.get()
        pose_name = "weld_start" if endpoint_index == 0 else "weld_end"
        role = "START" if endpoint_index == 0 else "GOAL"
        if joint_names is None or positions is None:
            self.error(
                f"Reference TCP {endpoint_index + 1} needs a measured six-joint seed"
            )
            return
        stored = (
            planning_group,
            tuple(joint_names),
            tuple(positions),
            copy.deepcopy(pose),
        )
        self._invalidate_seam_correction_runtime(
            f"re-taught reference TCP {endpoint_index + 1} / {role}",
            clear_touches=True,
        )
        self.linear_tcp_endpoints[endpoint_index] = copy.deepcopy(pose)
        self.taught_robot_poses[pose_name] = copy.deepcopy(stored)
        if provenance is not None:
            self.teaching_capture_provenance[pose_name] = copy.deepcopy(provenance)
        if self.seam_teaching_reference is None:
            self.seam_teaching_reference = {}
        self.seam_teaching_reference[pose_name] = copy.deepcopy(stored)
        try:
            state_path = self._initial_state_yaml_path(planning_group, pose_name)
            save_initial_state_yaml(
                state_path,
                planning_group,
                joint_names,
                positions,
                pose,
                provenance,
            )
            reference_path = self._seam_reference_yaml_path(planning_group)
            save_seam_teaching_reference_yaml(
                reference_path,
                planning_group,
                self.seam_teaching_reference,
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.error(f"Reference TCP save failed: {error}")
            return
        position = pose.position
        status = (
            f"saved ({position.x:.3f}, {position.y:.3f}, {position.z:.3f})"
        )
        self._update_seam_yaw_status()
        self.quick_teaching_status.set(
            f"Reference TCP {endpoint_index + 1} / {role} saved"
        )
        self.log(
            f"REFERENCE TCP {endpoint_index + 1} / {role} SAVED · "
            f"World XYZ {status} · {reference_path}"
        )


    def _update_seam_yaw_status(self, sensed_start=None, sensed_goal=None):
        ref_start = self.linear_tcp_endpoints[0]
        ref_goal = self.linear_tcp_endpoints[1]
        if self._wait_fixed_tilt_mode_enabled():
            wait_reference = self._wait_fixed_tilt_seam_reference(False)
            if wait_reference is not None:
                ref_start = wait_reference["weld_start"][3]
                ref_goal = wait_reference["weld_end"][3]
        if ref_start is None or ref_goal is None:
            reference = self.seam_teaching_reference or {}
            if ref_start is None and reference.get("weld_start") is not None:
                ref_start = reference["weld_start"][3]
            if ref_goal is None and reference.get("weld_end") is not None:
                ref_goal = reference["weld_end"][3]
        if ref_start is None or ref_goal is None:
            self.reference_yaw_status.set("Reference yaw: --")
            self.reference_length_status.set("Length: --")
            self.sensed_yaw_status.set("Sensed yaw: --")
            self.delta_yaw_status.set("ΔYaw: --")
            return
        try:
            reference_yaw = seam_yaw(ref_start, ref_goal)
            dx = ref_goal.position.x - ref_start.position.x
            dy = ref_goal.position.y - ref_start.position.y
            dz = ref_goal.position.z - ref_start.position.z
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.reference_yaw_status.set(
                f"Reference yaw: {math.degrees(reference_yaw):+.2f}°"
            )
            self.reference_length_status.set(
                f"Length: {length * 1000.0:.1f} mm"
            )
            if sensed_start is None or sensed_goal is None:
                self.sensed_yaw_status.set("Sensed yaw: --")
                self.delta_yaw_status.set("ΔYaw: --")
                return
            sensed_value = seam_yaw(sensed_start, sensed_goal)
            delta = math.atan2(
                math.sin(sensed_value - reference_yaw),
                math.cos(sensed_value - reference_yaw),
            )
            self.sensed_yaw_status.set(
                f"Sensed yaw: {math.degrees(sensed_value):+.2f}°"
            )
            self.delta_yaw_status.set(
                (
                    f"Geometric ΔYaw (not applied): {math.degrees(delta):+.2f}°"
                    if self._wait_fixed_tilt_mode_enabled()
                    else f"ΔYaw: {math.degrees(delta):+.2f}°"
                )
            )
        except ValueError:
            self.reference_yaw_status.set("Reference yaw: invalid")
            self.reference_length_status.set("Length: --")
            self.sensed_yaw_status.set("Sensed yaw: --")
            self.delta_yaw_status.set("ΔYaw: --")



    def replace_with_tcp(self):
        index = self.selected_index()
        if index is None:
            self.error("Select a waypoint to replace")
            return
        threading.Thread(
            target=self.node.capture_tcp,
            args=(
                index,
                self.show_path.get(),
                self.planning_group.get(),
            ),
            daemon=True,
        ).start()

    def apply_captured_tcp(self, pose, replace_index, visible):
        if replace_index is None:
            self.points.append(copy.deepcopy(pose))
            selected_index = len(self.points) - 1
            action = "Appended"
        else:
            self.points[replace_index] = copy.deepcopy(pose)
            selected_index = replace_index
            action = "Replaced"
        self.path_kind = "taught"
        self.weave_source = copy.deepcopy(self.points)
        self.set_points(self.points, selected_index)
        self.node.publish_points(self.points, visible)
        self.log(
            f"{action} current {self.planning_group.get()} TCP · "
            "World 6D pose"
        )

    def reverse_path(self):
        if len(self.points) < 2:
            self.error("Path needs at least two poses")
            return
        self.points.reverse()
        if self.path_kind != "weave":
            self.weave_source = copy.deepcopy(self.points)
        self.publish_edits(0)
        self.log("Reversed seam direction")

    def restore_weave_source(self):
        if not self.weave_source:
            self.error("No source seam is available")
            return
        self.path_kind = "source"
        self.set_points(self.weave_source)
        self.node.publish_points(self.points, self.show_path.get())
        self.log("Restored the seam used before weaving")

    def clear_path(self):
        self.path_kind = "empty"
        self.weave_source = []
        self.weave_base_paths = {"linear": [], "circle": [], "corrected": []}
        self.set_points([])
        self.node.publish_points([], self.show_path.get())
        self.log("Cleared taught path")

    def update_speed_label(self, _value=None):
        self.invalidate_approved_plan()
        self.initial_plan_ready = False
        self._refresh_initial_position_controls()
        self.speed_label.configure(text=f"{self.velocity_percent.get():.1f}%")
        if hasattr(self, "cleaner_speed_label"):
            self.cleaner_speed_label.configure(text=f"{self.velocity_percent.get():.1f}%")

    def speed_mode_changed(self):
        self.update_speed_label()

    def _update_motion_profile_label(self):
        if self.linear_motion_profile.get():
            text = "linear: constant-velocity cruise with ramped in/out"
        else:
            text = "S-curve: TOTG + Ruckig jerk smoothing"
        self.motion_profile_label.configure(text=text)

    def _motion_profile_toggled(self):
        self._update_motion_profile_label()
        self.invalidate_approved_plan()

    def _selected_tcp_speed_m_s(self):
        if self.speed_mode.get() != "tcp":
            return 0.0
        try:
            speed_mm_s = float(self.tcp_speed_mm_s.get())
        except (ValueError, tk.TclError) as error:
            raise ValueError("TCP speed is invalid") from error
        if not math.isfinite(speed_mm_s) or not 0.1 <= speed_mm_s <= 500.0:
            raise ValueError("TCP speed must be in 0.1..500.0 mm/s")
        return speed_mm_s * 0.001

    def plan_preview(self):
        if self.keyboard_velocity_arm is not None:
            self.error("Disable Keyboard Teaching before planning")
            return
        if not self._selected_robot_connected():
            self.error("Connect the robot and wait for live /joint_states")
            return
        self._send_path(execute_requested=False)

    def execute_approved(self):
        if self.keyboard_velocity_arm is not None:
            self.error("Disable Keyboard Teaching before execution")
            return
        if not self.plan_approved:
            self.error("Plan Preview is required before execution")
            return
        if not self.execution_allowed:
            self.error("Server execution is disabled by launch configuration")
            return
        if not self._selected_robot_connected():
            self.error("Connect the REAL RB robot first")
            return
        self._send_path(execute_requested=True)

    def _send_path(self, execute_requested):
        speed = max(0.01, min(1.0, self.velocity_percent.get() / 100.0))
        planning_group = self.planning_group.get()
        try:
            tcp_speed_m_s = self._selected_tcp_speed_m_s()
            interpolation_step = (
                float(self.interpolation_step_mm.get()) * 0.001
            )
        except (ValueError, tk.TclError):
            self.error("Cartesian interpolation step is invalid")
            return
        if not 0.0005 <= interpolation_step <= 0.02:
            self.error("Cartesian interpolation step must be 0.5..20 mm")
            return
        threading.Thread(
            target=self.node.submit_cartesian_motion,
            args=(
                copy.deepcopy(self.points),
                speed,
                interpolation_step,
                self.show_path.get(),
                execute_requested,
                execute_requested,
                planning_group,
                tcp_speed_m_s,
                self.linear_motion_profile.get(),
            ),
            daemon=True,
        ).start()

    def _set_fastech_output_buttons(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self.fastech_output_buttons:
            button.configure(state=state)

    def connect_fastech_ethernet(self):
        if self.fastech_connected or self.fastech_connecting:
            return
        self.fastech_connecting = True
        self.fastech_connect_button.configure(state=tk.DISABLED)
        self.fastech_io_status.configure(
            text="Requesting /fastech/connect..."
        )
        threading.Thread(
            target=self._fastech_connection_service_worker,
            args=(True,),
            daemon=True,
        ).start()

    def _fastech_connection_service_worker(self, connect):
        success, message = self.node.set_fastech_connection_sync(connect)
        self.post(
            self._fastech_connection_service_result,
            success,
            message,
        )

    def _fastech_connection_service_result(self, success, message):
        self.fastech_connecting = False
        if not success:
            self.fastech_connect_button.configure(state=tk.NORMAL)
            self.fastech_io_status.configure(text=message)
            self.error(message)
            return
        self.log(message)

    def disconnect_fastech_ethernet(self):
        if self.fastech_connecting:
            return
        self.fastech_connecting = True
        self.fastech_disconnect_button.configure(state=tk.DISABLED)
        self.fastech_io_status.configure(
            text="Requesting /fastech/disconnect..."
        )
        threading.Thread(
            target=self._fastech_connection_service_worker,
            args=(False,),
            daemon=True,
        ).start()

    def update_fastech_io(self, state_message):
        was_connected = self.fastech_connected
        self.fastech_connected = bool(state_message.connected)
        if not self.fastech_connected:
            self.fastech_previous_state = None
            self.fastech_pending_outputs.clear()
            self.fastech_connect_button.configure(state=tk.NORMAL)
            self.fastech_disconnect_button.configure(state=tk.DISABLED)
            self.fastech_all_off_button.configure(state=tk.DISABLED)
            self._set_fastech_output_buttons(False)
            self.touch_input_states["right"] = None
            for (kind, channel), label in self.fastech_io_labels.items():
                label.configure(text=f"{kind}{channel} –", bg="#eeeeee")
            self.fastech_io_status.configure(
                text=f"Disconnected · {state_message.detail}"
            )
            if was_connected:
                self.log(
                    f"Fastech ROS node reports disconnected · {state_message.detail}"
                )
            return

        self.fastech_connecting = False
        self.fastech_connect_button.configure(state=tk.DISABLED)
        self.fastech_disconnect_button.configure(state=tk.NORMAL)
        self.fastech_all_off_button.configure(state=tk.NORMAL)
        self._set_fastech_output_buttons(True)
        previous = self.fastech_previous_state
        changes = []
        for channel in FASTECH_GUI_CHANNELS:
            for kind, values, old_values in (
                (
                    "DI",
                    state_message.digital_in,
                    previous.digital_in if previous is not None else None,
                ),
                (
                    "DO",
                    state_message.digital_out,
                    previous.digital_out if previous is not None else None,
                ),
            ):
                if channel >= len(values):
                    continue
                value = bool(values[channel])
                old_value = (
                    bool(old_values[channel]) if old_values is not None else None
                )
                if old_value is not None and value == old_value:
                    continue
                label = self.fastech_io_labels[(kind, channel)]
                label.configure(
                    text=f"{kind}{channel} {'ON' if value else 'OFF'}",
                    bg="#81c995" if value else "#dbeafe",
                )
                if old_value is not None:
                    changes.append(
                        f"{kind}{channel}={'ON' if value else 'OFF'}"
                    )
        self.fastech_previous_state = state_message
        self.fastech_io_status.configure(
            text=(
                f"Connected {state_message.ip_address} · board "
                f"{state_message.board_id} · {state_message.poll_rate_hz:.0f} Hz · "
                f"raw DI=0x{state_message.raw_input:08X} · "
                f"raw DO=0x{state_message.raw_output:08X}"
            )
        )
        if not was_connected:
            self.log(
                f"Fastech ROS I/O connected · {state_message.ip_address} · "
                f"board {state_message.board_id} · {state_message.detail}"
            )
        elif changes:
            self.log("Fastech I/O changed · " + ", ".join(changes))

    def _set_fastech_output_sync(self, channel, enabled):
        """Command the Fastech owner node from a sequence worker."""
        if not self.fastech_connected:
            return False, "Fastech Ethernet is disconnected"
        return self.node.set_fastech_output_sync(channel, enabled)

    def request_fastech_output(self, channel, enabled):
        channel = int(channel)
        if channel not in FASTECH_GUI_CHANNELS:
            self.error(f"Fastech DO{channel} is not exposed in this GUI")
            return
        if not self.fastech_connected:
            self.error("Connect Fastech before commanding an output")
            return
        if channel in self.fastech_pending_outputs:
            return
        action = "ON" if enabled else "OFF"
        if not messagebox.askyesno(
            f"Fastech DO{channel} {action}",
            f"Command Fastech physical DO{channel} ({FASTECH_GUI_CHANNELS[channel]}) "
            f"to {action}?\n\n"
            "This output may operate connected physical equipment.",
        ):
            return
        self.fastech_pending_outputs.add(channel)
        self.fastech_io_labels[("DO", channel)].configure(
            text=f"DO{channel} WAIT", bg="#fdd663"
        )
        threading.Thread(
            target=self._fastech_output_worker,
            args=({channel: bool(enabled)},),
            daemon=True,
        ).start()

    def fastech_outputs_all_off(self):
        if not self.fastech_connected:
            self.error("Connect Fastech before commanding outputs")
            return
        if not messagebox.askyesno(
            "Fastech exposed outputs all OFF",
            "Command physical Fastech DO0, DO3, DO4, DO5, DO6, and DO7 to OFF? "
            "DO3/DO4 are retired test channels and have no ON controls.",
        ):
            return
        values = {channel: False for channel in FASTECH_ALL_OFF_CHANNELS}
        self.fastech_pending_outputs.update(values)
        for channel in values:
            label = self.fastech_io_labels.get(("DO", channel))
            if label is not None:
                label.configure(text=f"DO{channel} WAIT", bg="#fdd663")
        threading.Thread(
            target=self._fastech_output_worker,
            args=(values,),
            daemon=True,
        ).start()

    def _fastech_output_worker(self, values):
        failures = []
        messages = []
        for channel, enabled in values.items():
            success, message = self.node.set_fastech_output_sync(
                channel, enabled
            )
            messages.append(message)
            if not success:
                failures.append(channel)
        self.post(
            self._fastech_output_result,
            tuple(values),
            tuple(failures),
            "; ".join(messages),
        )

    def _fastech_output_result(self, channels, failures, message):
        self.fastech_pending_outputs.difference_update(channels)
        if failures:
            self.log(
                "Fastech output command rejected · "
                + ", ".join(f"DO{channel}" for channel in failures)
                + f" · {message}"
            )
            if self.fastech_previous_state is not None:
                for channel in failures:
                    if channel >= len(
                        self.fastech_previous_state.digital_out
                    ):
                        continue
                    value = bool(
                        self.fastech_previous_state.digital_out[channel]
                    )
                    label = self.fastech_io_labels.get(("DO", channel))
                    if label is not None:
                        label.configure(
                            text=f"DO{channel} {'ON' if value else 'OFF'}",
                            bg="#81c995" if value else "#dbeafe",
                        )
            return
        self.log(
            "Fastech output command OK · "
            + ", ".join(f"DO{channel}" for channel in channels)
            + f" · {message}"
        )

    def update_control_box_io(self, digital_in, digital_out):
        current = (tuple(digital_in), tuple(digital_out))
        previous = self.previous_control_box_io
        changes = []
        for kind, values, old_values in (
            ("DI", current[0], previous[0] if previous else None),
            ("DO", current[1], previous[1] if previous else None),
        ):
            for port, value in enumerate(values):
                changed = (
                    old_values is not None and value != old_values[port]
                )
                candidate = port in MANUAL_IO_CANDIDATES
                if old_values is not None and value == old_values[port]:
                    continue
                if value:
                    background = "#81c995"
                elif changed:
                    background = "#fdd663"
                elif candidate:
                    background = "#dbeafe"
                else:
                    background = "#eeeeee"
                self.control_box_io_labels[(kind, port)].configure(
                    text=f"{port:02d}\n{'ON' if value else 'OFF'}",
                    bg=background,
                )
                if changed:
                    changes.append(
                        f"{kind}{port}={'ON' if value else 'OFF'}"
                    )
        self.previous_control_box_io = current
        active_inputs = [
            str(index) for index, value in enumerate(current[0]) if value
        ]
        active_outputs = [
            str(index) for index, value in enumerate(current[1]) if value
        ]
        self.control_box_io_status.configure(
            text=(
                f"Active DI: {', '.join(active_inputs) or 'none'} · "
                f"Active DO: {', '.join(active_outputs) or 'none'}"
            )
        )
        if changes:
            self.log("Rainbow control-box I/O changed · " + ", ".join(changes))

    def update_touch_input(self, arm, active):
        previous = self.touch_input_states[arm]
        active = bool(active)
        self.touch_input_states[arm] = active
        if previous is not None and active and not previous:
            self.touch_input_rising_edges[arm] += 1
            self._handle_touch_event(arm, f"{arm.upper()} Fastech DI0")

    def _handle_touch_event(self, arm, source):
        if arm != self._selected_arm():
            return
        planning_group = f"{arm}_manipulator"
        if self.automatic_probe_kind is not None:
            kind = self.automatic_probe_kind
            self.root.bell()
            self.pipeline_waiting(
                f"Fastech DI0 TOUCH DETECTED · stopping {kind} probe before capture"
            )
            # WeldGuiNode._system_state owns the stop trigger. Starting a
            # second worker here allowed a bounced Fastech DI0 edge to capture and
            # launch the return path twice.
            return
        guard = self.node.active_touch_guard
        if guard is not None and guard[0] == arm:
            self.root.bell()
            self.pipeline_waiting(
                f"Fastech DI0 TOUCH DETECTED · stopping guarded {guard[1]} motion"
            )
            return
        if not self.touch_sensing_enabled.get():
            return
        self.root.bell()
        self.pipeline_waiting(
            f"TOUCH DETECTED · source={source} · "
            f"capturing {planning_group} TCP"
        )
        threading.Thread(
            target=self.node.capture_touch_pose,
            args=(planning_group, source),
            daemon=True,
        ).start()

    def _persist_seam_touch_yaml(self, planning_group, event_label):
        touch_yaml = (
            self.pass_probe_touch_yaml_target
            or self._seam_touch_yaml_path(planning_group)
        )
        try:
            save_seam_touch_yaml(
                touch_yaml,
                planning_group,
                self.seam_axis.get(),
                self.seam_probe_touches,
                self.seam_probe_starts,
                self.seam_probe_stops,
                probe_configuration={
                    "wall_direction": self.wall_probe_axis.get(),
                    "wall_sign": self.wall_probe_sign.get(),
                    "base_direction": self.floor_probe_axis.get(),
                    "base_sign": self.floor_probe_sign.get(),
                    "orientation_mode": self.seam_orientation_mode.get(),
                    "fixed_world_x_tilt_deg": float(
                        self.weld_fixed_tilt_x_deg.get()
                    ),
                    "fixed_world_y_tilt_deg": float(
                        self.weld_fixed_tilt_y_deg.get()
                    ),
                    "fixed_world_z_tilt_deg": float(
                        self.weld_fixed_tilt_z_deg.get()
                    ),
                    "weld_lead_in_mm": float(self.weld_lead_in_mm.get()),
                    "weld_lead_out_mm": float(self.weld_lead_out_mm.get()),
                    "weld_safe_approach_mm": float(
                        self.weld_safe_approach_mm.get()
                    ),
                    "weld_pre_start_lead_mm": float(
                        self.weld_pre_start_lead_mm.get()
                    ),
                    "weld_tcp_speed_mm_s": float(
                        self.weld_tcp_speed_mm_s.get()
                    ),
                },
            )
        except (OSError, ValueError, yaml.YAMLError, tk.TclError) as error:
            self.error(f"Fastech DI0 touch YAML save failed: {error}")
            return None
        self.log(f"Fastech DI0 {event_label} YAML SAVED · {touch_yaml}")
        return touch_yaml

    def apply_touch_edge_capture(
        self,
        pose,
        planning_group,
        kind,
        probe_start,
    ):
        return _seam_correction_for(self).apply_touch_edge_capture(pose, planning_group, kind, probe_start)

    def apply_touch_capture(
        self,
        pose,
        planning_group,
        source,
        probe_start=None,
        stopped_pose=None,
        cancel_event=None,
    ):
        return _seam_correction_for(self).apply_touch_capture(pose, planning_group, source, probe_start, stopped_pose, cancel_event)

    def touch_probe_return_finished(self, success, message, probe_kind):
        return _seam_correction_for(self).touch_probe_return_finished(success, message, probe_kind)

    def confirm_all_do_unlock(self):
        if not self.unlock_all_do_ports.get():
            return
        if not messagebox.askyesno(
            "Unlock all Rainbow DO ports",
            "Unknown outputs may operate gas, inching, ARC, or another "
            "actuator. Allow clicking every DO0..15 port?",
        ):
            self.unlock_all_do_ports.set(False)

    def request_do_toggle(self, port):
        if self.previous_control_box_io is None:
            self.error("Rainbow control-box state is not available")
            return
        if port in self.pending_do_ports:
            return
        candidate = port in MANUAL_IO_CANDIDATES
        if not candidate and not self.unlock_all_do_ports.get():
            self.error(
                f"DO{port} is locked · enable non-candidate DO clicking first"
            )
            return
        current = bool(self.previous_control_box_io[1][port])
        target = not current
        if not messagebox.askyesno(
            f"Toggle Rainbow DO{port}",
            f"Command control-box DO{port}: "
            f"{'ON' if current else 'OFF'} → {'ON' if target else 'OFF'}?\n\n"
            "This is a physical output and may operate connected equipment.",
        ):
            return
        self.pending_do_ports.add(port)
        label = self.control_box_io_labels[("DO", port)]
        label.configure(bg="#fdd663", text=f"{port:02d}\nWAIT")
        self.log(
            f"Rainbow DO{port} command requested · "
            f"{'ON' if target else 'OFF'}"
        )
        threading.Thread(
            target=self.node.set_digital_output,
            args=(port, target),
            daemon=True,
        ).start()

    def candidate_outputs_off(self):
        if not messagebox.askyesno(
            "Force candidate outputs OFF",
            "Command DO4, DO8, DO9, DO10, DO12, and DO13 to OFF?",
        ):
            return
        for port in sorted(MANUAL_IO_CANDIDATES):
            self.pending_do_ports.add(port)
            threading.Thread(
                target=self.node.set_digital_output,
                args=(port, False),
                daemon=True,
            ).start()
        self.log("Rainbow candidate DO all-OFF requested")

    def digital_output_result(self, port, success, message):
        self.pending_do_ports.discard(port)
        prefix = "OK" if success else "REJECTED"
        self.log(f"Rainbow DO{port} {prefix} · {message}")
        if not success and self.previous_control_box_io is not None:
            value = self.previous_control_box_io[1][port]
            self.control_box_io_labels[("DO", port)].configure(
                text=f"{port:02d}\n{'ON' if value else 'OFF'}",
                bg=(
                    "#81c995"
                    if value
                    else (
                        "#dbeafe"
                        if port in MANUAL_IO_CANDIDATES
                        else "#eeeeee"
                    )
                ),
            )

    def begin(self, velocity_scale, execute_requested, tcp_speed_m_s=0.0):
        self.bar["value"] = 0
        self.last_action_phase = ""
        self.plan_button.configure(state=tk.DISABLED)
        self.execute_button.configure(state=tk.DISABLED)
        operation = (
            "EXECUTE exact approved plan"
            if execute_requested
            else "PLAN PREVIEW for RViz"
        )
        self.pipeline_waiting(
            f"{operation} · "
            + (
                f"TCP average target={tcp_speed_m_s * 1000.0:.2f} mm/s"
                if tcp_speed_m_s > 0.0
                else f"velocity scale={velocity_scale:.1%}"
            )
        )

    def progress(self, value, waypoint, pose, phase):
        self.bar["value"] = value * 100
        position = pose.position
        self.feedback_label.configure(
            text=(
                f"{phase or 'PATH'} · waypoint: {waypoint + 1} · "
                f"progress: {value:.0%} · "
                f"pose: ({position.x:.3f}, {position.y:.3f}, "
                f"{position.z:.3f})"
            ),
        )
        if phase and phase != self.last_action_phase:
            self.last_action_phase = phase
            self.log(f"Sequence phase · {phase}")

    def finish(self, text, was_execution):
        if was_execution and self.automatic_probe_kind is not None:
            kind = self.automatic_probe_kind
            self.automatic_probe_kind = None
            self.node.clear_touch_probe()
            self._signal_auto_seam_stage(False, kind)
            self.error(
                f"{kind} probe reached maximum travel without a Fastech DI4 edge"
            )
            return
        self.bar["value"] = 100
        self.plan_button.configure(
            state=(
                tk.NORMAL
                if self.points and self._selected_robot_connected()
                else tk.DISABLED
            )
        )
        self.plan_approved = not was_execution
        self.execute_button.configure(
            state=(
                tk.NORMAL
                if (
                    self.plan_approved
                    and self.execution_allowed
                    and self._selected_robot_connected()
                )
                else tk.DISABLED
            )
        )
        self._refresh_wide_sensing_controls()
        if self.plan_approved:
            self.pipeline_result(
                f"{text} · plan approved; inspect RViz, then execute"
            )
        else:
            self.pipeline_result(text)

    def cancel(self):
        self.node.cancel_active_motion()

    def close(self):
        if self._closing:
            return
        self._closing = True
        self._stop_keyboard_wire()
        self._stop_keyboard_jog_command()
        if self.keyboard_velocity_arm is not None:
            time.sleep(0.30)
            self.node.set_keyboard_velocity_controller_enabled(
                self.keyboard_velocity_arm, False
            )
            self.keyboard_velocity_arm = None
        if self.hicomm_client is not None:
            self.hicomm_client.stop()
        self.root.quit()
        self.root.destroy()

    def shutdown_ros(self):
        if rclpy.ok():
            rclpy.shutdown()
        self.executor_thread.join(timeout=1.0)
        self.node.destroy_node()

    def check_ros(self):
        self._drain_ui_queue()
        if not rclpy.ok():
            self.root.destroy()
            return
        # Physical keyboard edges arrive on the ROS executor.  A 50 ms Tk
        # bridge interval added directly to jog start latency (plus one
        # ros2_control cycle).  High-rate telemetry is already coalesced and
        # _drain_ui_queue() has a strict time/item budget, so a 10 ms bridge is
        # responsive without allowing ROS callbacks to starve Tk.
        self.root.after(10, self.check_ros)

    def mainloop(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    gui = WeldActionGui()
    try:
        gui.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        gui.shutdown_ros()
