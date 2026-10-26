"""TASK › Scan: Hongik scans and result view, independent of production pass state."""

import math
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import rclpy
from rclpy.duration import Duration
import yaml

from construct_robot.core.precision_sensing_inspection import (
    MAX_1G_SEAM_TILT_DEG, SensingPose, finite_values, point_on_tool, pose_difference, seam_tilt_deg,
)
from construct_robot.application.direct_scan import (
    DirectScanController, ESTIMATORS, check_scan_coverage, estimate_poses, estimate_summary, latest_scan_dir,
    run_estimate,
    latest_pass_file, load_pass_file, load_scan_pose, pass_motion_steps,
)
from construct_robot.application import scan_correction, scan_four_pass, tcp_calibration
from construct_robot.application.scan_weld_cycle import export_post_weld_ply
from construct_robot.core.scan_motion import check_scan_line, pose_move_step, post_weld_scan_step
from construct_robot.core.sequence_model import next_sequential_slot
from construct_robot.io.scan_paths import post_weld_scan_root
from construct_robot.io.precision_sensing_results import SensingInspection, load_result
from construct_robot.io.scan_cycle_log import record_cycle_event
from construct_robot.io.teaching_yaml import save_initial_state_yaml, teaching_config_dir

# Scan tab keyboard shortcuts: key -> scan point.
# A/B (I/J): scan start/end for the Hongik action and the direct scan.
# C/D (O/K): local scan start/end poses, read only by the local scan.
SCAN_KEY_POINTS = {"i": "start", "j": "end", "o": "local_start", "k": "local_end"}
SCAN_POINT_LABELS = {"start": "A", "end": "B", "local_start": "C", "local_end": "D"}
# Reference key mode (default): the same keys save measured correct TCPs of the passes instead of A-D.
REFERENCE_KEYS = {"i": (1, "start"), "o": (2, "start"), "p": (3, "start"), "j": (4, "start"),
                  "k": (1, "goal"), "l": (2, "goal"), "m": (3, "goal"), "u": (4, "goal")}
# Optional orientation-only mode; reference mode captures each pass independently.
ORIENTATION_KEYS = {"i": (1, 3), "o": (2,), "p": (4,)}
SCAN_KEY_MODES = {"orientation": "I = P1 + P3 · O = P2 · P = P4 torch orientation",
                  "reference": "I O P J = P1-P4 start · K L M U = P1-P4 goal correct position",
                  "scan_poses": "I / J / O / K = scan poses A / B / C / D",
                  "tcp_cal": "I = TCP calibration sample (wire tip on one point)"}
TCP_CAL_KEYS = {"i": "sample"}
SEAM_POINT_LABELS = {"seam_start": "A", "seam_end": "B"}
SCAN_TCP_FRAME = "right_manipulator_ee_point"
# Wire-end reference of the Hongik v4 model (TCP + 20 mm along the TCP z axis).
SCAN_POINT_FRAME = "hongik_weld_link"


class PrecisionSensingTestPanel:
    def __init__(self, host, parent):
        self.host = host
        self.inspection = None
        self.busy = False
        self._tab_states = {}
        self._generation = 0
        self.frame = tk.StringVar(value="World")
        self.action = tk.StringVar(value="/hongik_fine_sensing/precision_sensing")
        self.status_service = tk.StringVar(value="/hongik_fine_sensing/hongik_auto_teach_server/status")
        self.log_topic = tk.StringVar(value="/hongik_fine_sensing/hongik_auto_teach_server/log")
        self.a = [tk.StringVar() for _ in range(3)]
        self.b = [tk.StringVar() for _ in range(3)]
        self.expected = [tk.StringVar() for _ in range(7)]
        self.position_tol = tk.StringVar(value="1.0")
        self.angle_tol = tk.StringVar(value="1.0")
        self.status = tk.StringVar(value="Load a delivered result, or explicitly start a 1G scan.")
        self.selected_text = tk.StringVar(value="Select a returned pose to publish its TF.")
        self.comparison = tk.StringVar(value="Reference values must use the selected pose's frame and convention.")
        self.scan_folder = Path(teaching_config_dir()) / "scan_teaching"
        self.scan_poses_text = tk.StringVar(value="Scan poses: not taught")
        self.capturing = False
        ttk.Label(parent, text="Full scan moves the RIGHT robot through Hongik's MoveIt/JTC server. "
                  "Result inspection only; returned paths are not loaded into the welding task.",
                  wraplength=1100).pack(anchor=tk.W, pady=6)
        config = ttk.LabelFrame(parent, text="1. Server and rough seam points", padding=8)
        config.pack(fill=tk.X, pady=5)
        for row, (label, var) in enumerate((("Action", self.action), ("Status service", self.status_service),
                                           ("Log topic", self.log_topic), ("Parent frame", self.frame))):
            ttk.Label(config, text=label).grid(row=row, column=0, sticky=tk.W)
            ttk.Entry(config, textvariable=var, width=76).grid(row=row, column=1, columnspan=5, sticky=tk.EW, pady=2)
        for row, (label, variables, which) in enumerate((("A · seam point", self.a, "seam_start"),
                                                         ("B · seam point", self.b, "seam_end")), 4):
            ttk.Label(config, text=f"{label} [mm]").grid(row=row, column=0, sticky=tk.W)
            for i, (axis, var) in enumerate(zip("xyz", variables)):
                cell = ttk.Frame(config)
                cell.grid(row=row, column=i + 1, sticky=tk.W)
                ttk.Label(cell, text=axis).pack(side=tk.LEFT)
                ttk.Entry(cell, textvariable=var, width=12).pack(side=tk.LEFT, padx=3)
            ttk.Button(config, text=f"Save wire tip → {SEAM_POINT_LABELS[which]}",
                       command=lambda w=which: self.capture_scan_point(w)).grid(row=row, column=4, padx=5)
        ttk.Label(config, text="Teach with Keyboard Teaching (right arm): move to the scan start pose and press I, "
                  "to the scan end pose and press J. Joints and TCP are saved to "
                  f"{self.scan_folder}/scan_<start|end>.yaml (used by the direct scan). "
                  "Hongik action A/B are separate seam points: place the wire tip on the seam and use Save wire tip. "
                  f"They use {SCAN_POINT_FRAME} in World mm (sent in m) and are saved as seam_<start|end>.yaml.",
                  wraplength=1060).grid(
                      row=6, column=0, columnspan=6, sticky=tk.W, pady=5)
        self._build_direct_scan(parent)
        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, pady=4)
        self.scan_button = ttk.Button(buttons, text="Start full 1G scan (ROBOT MOVES)", command=self.start_scan)
        self.scan_button.pack(side=tk.LEFT, padx=(0, 6))
        self.cancel_button = ttk.Button(buttons, text="Cancel scan", command=self.cancel_scan, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=6)
        self.load_button = ttk.Button(buttons, text="Load precision_sensing_result.yaml", command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Clear result / RViz", command=self.clear).pack(side=tk.LEFT, padx=6)
        ttk.Label(parent, textvariable=self.status, wraplength=1100).pack(anchor=tk.W, pady=5)
        self._build_post_weld(parent)
        self._build_pass_motion(parent)
        # "2. Returned pose values" / "3. Compare ..." are hidden (operator 2026-10-20); their widgets stay
        # unpacked because show_result / select_pose / compare still fill them.
        results = ttk.LabelFrame(parent, text="2. Returned pose values", padding=8)
        ttk.Label(results, text="WIRE: position = wire tip, axes = surface frame (+Z plate normal, +X travel). "
                  "TCP: contact-tip pose / torch orientation from kiro/pass_<k>.yaml. Quaternion order: xyzw.",
                  wraplength=1080).pack(anchor=tk.W, pady=(0, 5))
        table = ttk.Frame(results)
        table.pack(fill=tk.BOTH, expand=True)
        columns = ("pass", "kind", "point", "frame", "x", "y", "z", "qx", "qy", "qz", "qw")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=11, selectmode="browse")
        for column in columns:
            self.tree.heading(column, text=column + (" [mm]" if column in "xyz" else ""))
            self.tree.column(column, width=86 if column not in ("pass", "kind", "point") else 58, stretch=True)
        scroll = ttk.Scrollbar(table, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.select_pose)
        ttk.Label(results, textvariable=self.selected_text, wraplength=1080).pack(anchor=tk.W, pady=5)
        expected = ttk.LabelFrame(parent, text="3. Compare selected pose with measured/reference values", padding=8)
        for col, (name, var) in enumerate(zip(("x mm", "y mm", "z mm", "qx", "qy", "qz", "qw"), self.expected)):
            ttk.Label(expected, text=name).grid(row=0, column=col)
            ttk.Entry(expected, textvariable=var, width=12).grid(row=1, column=col, padx=3)
        ttk.Label(expected, text="Position tolerance [mm]").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Entry(expected, textvariable=self.position_tol, width=9).grid(row=2, column=2)
        ttk.Label(expected, text="Angle tolerance [deg]").grid(row=2, column=3, columnspan=2)
        ttk.Entry(expected, textvariable=self.angle_tol, width=9).grid(row=2, column=5)
        ttk.Button(expected, text="Compare", command=self.compare).grid(row=2, column=6)
        ttk.Label(expected, textvariable=self.comparison, wraplength=1060).grid(row=3, column=0, columnspan=7, sticky=tk.W)
        ttk.Label(parent, text="RViz: Fixed Frame = result frame; add TF and MarkerArray /hongik_test/path_markers. "
                  "TF children: hongik_test/pass_<k>_wire_start/goal, pass_<k>_tcp_start/goal, selected_wire/tcp. "
                  "These are result snapshots, not live robot feedback.", wraplength=1100).pack(anchor=tk.W, pady=6)
        self.details = tk.Text(parent, height=14, wrap=tk.WORD)
        self.details.pack(fill=tk.BOTH, expand=True, pady=5)
        self.details.configure(state=tk.DISABLED)
        self.load_scan_points()
        try:                                          # the latest estimate's poses, if any (files only)
            self.estimate_scan_dir = latest_scan_dir()
            self.estimated_poses = estimate_poses(self.estimate_scan_dir)
            self.pose_combo.configure(values=[label for label, _pose in self.estimated_poses])
            if self.estimated_poses:
                self.pose_choice.set(self.estimated_poses[0][0])
        except Exception:
            self.estimated_poses = []

    @property
    def bridge(self):
        return self.host.node.precision_sensing_test

    def _build_pass_motion(self, parent):
        self.kiro_pass = None
        self.pass_file = tk.StringVar(value=latest_pass_file())
        self.pass_mode = tk.StringVar(value="START → GOAL")
        self.pass_speed = tk.StringVar(value="5.0")
        self.pass_status = tk.StringVar(value="Load a kiro/pass_<k>.yaml to inspect or move its original TCP poses.")
        box = ttk.LabelFrame(parent, text="Original KIRO pass · motion only (no ARC)", padding=8)
        box.pack(fill=tk.X, pady=5)
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self.pass_file, width=78).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=self.browse_pass_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Load pass YAML", command=self.load_pass_motion).pack(side=tk.LEFT)
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=5)
        ttk.Combobox(row, textvariable=self.pass_mode, values=("START", "GOAL", "START → GOAL"),
                     state="readonly", width=18).pack(side=tk.LEFT)
        ttk.Label(row, text="Line speed [mm/s]").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.pass_speed, width=8).pack(side=tk.LEFT)
        ttk.Button(row, text="Plan", command=lambda: self.run_pass_motion(False)).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Execute (ROBOT MOVES)", command=lambda: self.run_pass_motion(True)).pack(side=tk.LEFT)
        ttk.Button(row, text="STOP", command=self.stop_pass_motion).pack(side=tk.LEFT, padx=5)
        ttk.Label(box, text="World · right_manipulator_ee_point · original XYZ and quaternion. "
                  "START → GOAL first approaches START, then follows the straight pass line. "
                  "No correction, extra wire offset or weave; collision and touch-stop checks remain active.",
                  wraplength=1080).pack(anchor=tk.W)
        ttk.Label(box, textvariable=self.pass_status, wraplength=1080).pack(anchor=tk.W, pady=4)

    def browse_pass_file(self):
        path = filedialog.askopenfilename(title="Original KIRO pass YAML", initialdir=str(Path.home() / "hongik" / "runs"),
                                          filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            self.pass_file.set(path)
            self.load_pass_motion()

    def load_pass_motion(self):
        if self.busy or self.capturing:
            return
        self.kiro_pass = None
        path = self.pass_file.get().strip()
        self.pass_status.set("Loading original TCP poses...")
        self._worker(lambda: load_pass_file(path), self._show_pass_motion)

    def _show_pass_motion(self, loaded):
        if loaded.source != str(Path(self.pass_file.get()).expanduser().resolve()):
            return
        self.kiro_pass = loaded
        start, goal = loaded.start.position, loaded.goal.position
        self.pass_status.set(f"P{loaded.number} · START {tuple(round(v * 1000, 3) for v in start)} mm → "
                             f"GOAL {tuple(round(v * 1000, 3) for v in goal)} mm · original orientation · {loaded.source}")
        self.show_result(SensingInspection({"paths": [], "error_message": "Original TCP endpoint file"}, "World",
                         (loaded.start, loaded.goal), ("Original TCP poses; no correction or extra offset applied.",),
                         {}, loaded.source))

    def run_pass_motion(self, execute):
        blocker = self._motion_blocker(moves_robot=execute, check_live=False)
        if self.busy or self.capturing or blocker:
            self.pass_status.set(f"ERROR · {blocker or 'Wait for the active scan / capture'}")
            return
        try:
            loaded = self.kiro_pass
            if loaded is None or loaded.source != str(Path(self.pass_file.get()).expanduser().resolve()):
                raise ValueError("Load the selected pass YAML first")
            mode = self.pass_mode.get()
            speed = float(self.pass_speed.get()) / 1000.0
            scale = float(self.host.velocity_percent.get()) / 100.0
            service = self.status_service.get().strip()
        except (TypeError, ValueError) as error:
            self.pass_status.set(f"ERROR · {error}")
            return
        self.capturing = True
        generation = self._generation
        self.pass_status.set("Checking source file, server state and current TCP...")

        def prepare():
            try:
                loaded.require_unchanged()
                success, message, fields = self.bridge.server_status(service)
                if not success or fields.get("running") != "False" or fields.get("awaiting_confirm") == "True":
                    raise RuntimeError(f"Wait for Hongik motion to finish: {message}")
                current = self.host.node._path_start_tcp_pose("right_manipulator")
                steps = pass_motion_steps(loaded, current, mode, velocity_scale=scale, tcp_speed_m_s=speed)
                self.host.post(deliver, steps, None)
            except Exception as error:
                self.host.post(deliver, None, str(error))

        def deliver(steps, error):
            self.capturing = False
            if getattr(self.host, "_closing", False) or generation != self._generation:
                return
            blocker = self._motion_blocker(moves_robot=execute, check_live=False)
            if error or blocker or self.busy:
                self.pass_status.set(f"ERROR · {error or blocker or 'Scan is active'}")
                return
            if execute:
                preflight_error = self.host.sequence_execution_preflight_error(steps)
                if preflight_error:
                    self.pass_status.set(f"ERROR · {preflight_error}")
                    self.host.error(f"Original KIRO pass · {preflight_error}")
                    return
            self.pass_status.set(f"{'EXECUTE' if execute else 'PLAN'} · P{loaded.number} {mode} · original TCP poses · no ARC")
            self.host.log(f"Original KIRO pass · {loaded.source} · SHA256 {loaded.sha256} · {mode} · no ARC")
            self.host.run_sequence(True, execute, steps_override=steps)

        threading.Thread(target=prepare, daemon=True).start()

    def stop_pass_motion(self):
        self._generation += 1  # STOP also invalidates an in-flight motion preparation.
        self.host.stop_sequence()
        self.pass_status.set("STOP requested · pending pass preparation canceled")

    def _worker(self, operation, success):
        generation = self._generation
        def deliver(callback, value):
            if generation == self._generation and not getattr(self.host, "_closing", False):
                callback(value)
        def run():
            try:
                value = operation()
            except Exception as error:
                self.host.post(deliver, self.show_error, str(error))
            else:
                self.host.post(deliver, success, value)
        threading.Thread(target=run, daemon=True).start()

    def scan_point_path(self, which):
        if which in SEAM_POINT_LABELS:
            return self.scan_folder / f"{which}.yaml"
        return self.scan_folder / f"scan_{which}.yaml"

    def load_scan_points(self):
        """Load action seam points only; sensor scanning poses cannot populate A/B."""
        for which, variables in (("seam_start", self.a), ("seam_end", self.b)):
            try:
                with self.scan_point_path(which).open(encoding="utf-8") as stream:
                    provenance = (yaml.safe_load(stream) or {}).get("capture_provenance") or {}
                point = finite_values(provenance.get("scan_point_world_m") or (), 3)
            except (OSError, ValueError, yaml.YAMLError):
                continue
            for var, value in zip(variables, point):
                var.set(f"{value * 1000.0:.3f}")
        self.refresh_scan_poses()

    def refresh_scan_poses(self):
        """Show the taught TCP positions (World mm): A/B (direct scan) and C/D (local scan)."""
        positions = {}
        for which in SCAN_POINT_LABELS:
            try:
                with self.scan_point_path(which).open(encoding="utf-8") as stream:
                    tcp = (yaml.safe_load(stream) or {})["tcp_pose_world"]["position_m"]
                positions[which] = finite_values((tcp["x"], tcp["y"], tcp["z"]), 3)
            except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
                positions[which] = None
        lines = []
        for first, second in (("start", "end"), ("local_start", "local_end")):
            parts = [f"{SCAN_POINT_LABELS[w]} TCP " + ("not taught" if positions[w] is None else
                     "(" + ", ".join(f"{v * 1000.0:.1f}" for v in positions[w]) + ") mm") for w in (first, second)]
            if positions[first] and positions[second]:
                parts.append(f"line {math.dist(positions[first], positions[second]) * 1000.0:.1f} mm")
            lines.append(" · ".join(parts))
        self.scan_poses_text.set("Scan poses (World): " + lines[0] + "\nLocal scan poses (World): " + lines[1])

    def capture_scan_point(self, which):
        """Capture separate sensor poses (I/J/O/K) or action seam points (buttons)."""
        if which not in SCAN_KEY_POINTS.values() and which not in SEAM_POINT_LABELS:
            raise ValueError(f"unknown scan point {which!r}")
        if self.busy or self.capturing:
            self.show_error("Wait for the active scan / capture to finish")
            return
        if self.frame.get().strip() != "World":
            self.show_error("Scan teaching saves World positions; set Parent frame to World")
            return
        node, path = self.host.node, self.scan_point_path(which)
        variables = {"seam_start": self.a, "seam_end": self.b}.get(which, ())
        label = SEAM_POINT_LABELS.get(which) or SCAN_POINT_LABELS[which]
        self.capturing = True
        self.status.set(f"Capturing scan {which} from measured joints...")

        def work():
            names, positions, tcp, provenance = node.capture_measured_teaching_snapshot(
                "right_manipulator", f"scan_{which}")
            offset = node.tf_buffer.lookup_transform(
                SCAN_TCP_FRAME, SCAN_POINT_FRAME, rclpy.time.Time(), timeout=Duration(seconds=1.0),
            ).transform.translation
            point = point_on_tool((tcp.position.x, tcp.position.y, tcp.position.z), tcp.orientation,
                                  (offset.x, offset.y, offset.z))
            provenance = dict(provenance or {}, scan_point=which, scan_point_frame=SCAN_POINT_FRAME,
                              scan_point_parent="World", scan_point_world_m=[float(v) for v in point],
                              teaching_kind="hongik_seam_point" if which in SEAM_POINT_LABELS else "sensor_scan_pose")
            save_initial_state_yaml(path, "right_manipulator", names, positions, tcp, provenance)
            return point

        def done(point):
            self.capturing = False
            for var, value in zip(variables, point):
                var.set(f"{value * 1000.0:.3f}")
            self.status.set(f"Saved {label} ({which}) · TCP + joints, {SCAN_POINT_FRAME} = "
                            f"({', '.join(f'{v * 1000.0:.1f}' for v in point)}) mm World · {path}")
            self.host.log(f"Scan teaching · {which} saved → {path}")
            self.refresh_scan_poses()

        def failed(text):
            self.capturing = False
            self.show_error(f"Scan {which} capture failed: {text}")

        def run():
            try:
                point = work()
            except Exception as error:
                self.host.post(failed, str(error))
            else:
                self.host.post(done, point)
        threading.Thread(target=run, daemon=True).start()

    def _build_direct_scan(self, parent):
        self.direct_clearance = tk.StringVar(value="50")
        self.direct_status = tk.StringVar(value="Teach I = scan start pose and J = scan end pose (sensor over the seam), then start.")
        self.direct = None
        box = ttk.LabelFrame(parent, text="Direct 1G scan · KIRO motion, no Hongik server (ROBOT MOVES)", padding=8)
        box.pack(fill=tk.X, pady=5)
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Lift clearance mm").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.direct_clearance, width=6).pack(side=tk.LEFT, padx=(3, 12))
        # Hongik's measurement speed (job speed_mps 0.03; calibrated noise priors cover 15..60 mm/s).
        self.scan_speed_mm_s = tk.StringVar(value="30")
        ttk.Label(row, text="Scan speed mm/s").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.scan_speed_mm_s, width=6).pack(side=tk.LEFT, padx=(3, 12))
        self.direct_button = ttk.Button(row, text="Start direct 1G scan (ROBOT MOVES)", command=self.start_direct_scan)
        self.direct_button.pack(side=tk.LEFT, padx=(0, 6))
        self.direct_cancel_button = ttk.Button(row, text="Cancel", command=self.cancel_direct_scan, state=tk.DISABLED)
        self.direct_cancel_button.pack(side=tk.LEFT)
        local_row = ttk.Frame(box)
        local_row.pack(fill=tk.X, pady=(4, 0))
        for which in ("local_start", "local_end"):
            key = next(k for k, name in SCAN_KEY_POINTS.items() if name == which)
            ttk.Button(local_row, text=f"Save current → {SCAN_POINT_LABELS[which]} ({key.upper()})",
                       command=lambda w=which: self.capture_scan_point(w)).pack(side=tk.LEFT, padx=(0, 6))
        self.local_button = ttk.Button(local_row, text="Local 1G scan · C → D → C (ROBOT MOVES)",
                                       command=lambda: self.start_direct_scan(local=True))
        self.local_button.pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(local_row, text="Centred C / D from latest scan", command=self.plan_centred_scan_poses).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(local_row, text="Add post-weld scan (C → D → C) → Sequence Builder",
                   command=self.add_post_weld_scan_to_sequence).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(local_row, text="No lift: joint move to C, recorded line C → D and back D → C (two opposite "
                  "passes, as Hongik's 1G passes; teach C and D beyond both plate ends).",
                  foreground="#5f6368").pack(side=tk.LEFT)
        ttk.Label(box, textvariable=self.scan_poses_text, wraplength=1060).pack(anchor=tk.W, pady=(4, 0))
        self.estimate_job = tk.StringVar(value="")
        self.weld_direction = tk.StringVar(value="E->S")
        self.estimator = tk.StringVar(value=ESTIMATORS[0])     # kiro (default) | hongik
        self.estimate_text = tk.StringVar(value="Estimate: runs after each scan using the recorded groove "
                                                "(KIRO plane fit by default, Hongik optional). Scan past both plate ends.")
        self.estimating = False
        est_row = ttk.Frame(box)
        est_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(est_row, text="Estimator").pack(side=tk.LEFT)
        ttk.Combobox(est_row, textvariable=self.estimator, values=ESTIMATORS, state="readonly", width=7).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(est_row, text="Hongik job YAML ('' = groove_10T_1G template)").pack(side=tk.LEFT)
        ttk.Entry(est_row, textvariable=self.estimate_job, width=46).pack(side=tk.LEFT, padx=(3, 3))
        ttk.Button(est_row, text="Browse", command=self.browse_estimate_job).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(est_row, text="Weld direction (S = end scanned first)").pack(side=tk.LEFT)
        ttk.Combobox(est_row, textvariable=self.weld_direction, values=("E->S", "S->E", "alternate"),
                     state="readonly", width=9).pack(side=tk.LEFT, padx=(3, 8))
        self.estimate_button = ttk.Button(est_row, text="S/E + start/goal from latest scan",
                                          command=lambda: self.start_estimate(None))
        self.estimate_button.pack(side=tk.LEFT)
        ttk.Button(est_row, text="Show latest in RViz", command=lambda: self.show_estimate(None)).pack(side=tk.LEFT, padx=6)
        ttk.Label(box, textvariable=self.estimate_text, wraplength=1060, justify=tk.LEFT).pack(anchor=tk.W, pady=(3, 0))
        self.pose_choice = tk.StringVar()
        self.estimated_poses = []
        self.estimate_scan_dir = None
        move_row = ttk.Frame(box)
        move_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(move_row, text="Estimated pose").pack(side=tk.LEFT)
        self.pose_combo = ttk.Combobox(move_row, textvariable=self.pose_choice, state="readonly", width=18)
        self.pose_combo.pack(side=tk.LEFT, padx=(3, 3))
        ttk.Button(move_row, text="Refresh", command=lambda: self.refresh_estimated_poses(None)).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(move_row, text="Plan", command=lambda: self.move_to_estimated_pose(False)).pack(side=tk.LEFT)
        ttk.Button(move_row, text="Execute (ROBOT MOVES)",
                   command=lambda: self.move_to_estimated_pose(True)).pack(side=tk.LEFT, padx=6)
        ttk.Label(move_row, text="TCP line to the selected pass end with the CURRENT torch orientation: TCP = wire-tip aim "
                  "point - L x torch axis (Build/Teaching speed scale; Fastech DI0 touch stops it; plan first).",
                  foreground="#5f6368", wraplength=700).pack(side=tk.LEFT, padx=6)
        self._build_reference_correction(box)
        ttk.Label(box, text="Sequence: lift +Z from the current pose → joint move to the scan start (I) → straight line "
                  "to the scan end (J) at the scan speed (default 30 mm/s = Hongik's measurement speed) with the scan start orientation, recording every profile with "
                  "the robot pose → lift +Z. Sensor settings come from 1. CONNECTION (calib YAML required); the laser "
                  "is switched on for the scan. Output: ~/ros2_ws/scan_data/kiro_scan/<time>_scan/, RViz PointCloud2 "
                  "/hongik_scanner/scan_cloud (World).", wraplength=1060, foreground="#5f6368").pack(anchor=tk.W, pady=3)
        ttk.Label(box, textvariable=self.direct_status, wraplength=1060).pack(anchor=tk.W)

    def start_direct_scan(self, local=False):
        """Direct scan; local=True skips the lift / retract (start pose -> end pose only)."""
        if self.busy or self.capturing:
            return
        host = self.host
        blocker = self._motion_blocker(check_live=False)
        if blocker:
            self.direct_status.set(f"ERROR · {blocker}")
            return
        connection = host.scanner_connection_panel
        try:
            clearance_m = 0.0 if local else finite_values((self.direct_clearance.get(),), 1)[0] / 1000.0
            tcp_speed_m_s = finite_values((self.scan_speed_mm_s.get(),), 1)[0] / 1000.0
            velocity_scale = finite_values((host.velocity_percent.get(),), 1)[0] / 100.0
            sensor = dict(ip=connection.ip.get().strip(), port=int(connection.port.get()),
                          calib_file=connection.calib.get().strip(), gain=int(connection.gain.get()),
                          backup_dir="~/scan_data/sensor_backup",
                          on_status=connection.sensor_status.set, on_stopped=connection._stopped)
            if not sensor["calib_file"]:
                raise ValueError("Set the calib YAML in 1. CONNECTION (sensor pose on the tool)")
            names = ("local_start", "local_end") if local else ("start", "end")
            paths = [self.scan_point_path(which) for which in names]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise ValueError(f"Teach the scan poses first ({'O / K' if local else 'I / J'}): "
                                 f"missing {', '.join(missing)}")
        except (ValueError, tk.TclError) as error:
            self.direct_status.set(f"ERROR · {error}")
            return
        if self.direct is None:
            from construct_robot.nodes.scan_record_process import RecordSession
            self.direct = DirectScanController(host, RecordSession)
        self._set_busy(True, direct=True)
        connection._set_busy(True)
        self.direct_status.set(f"Starting {'local' if local else 'direct'} scan (ROBOT MOVES)...")

        def finished(result):
            self._set_busy(False)
            connection._set_busy(False)
            folder = f" · saved {result['folder']}" if result.get("folder") else ""
            self.direct_status.set(f"{result['status']} · {result['message']}{folder}")
            (host.log if result["status"] == "OK" else host.error)(f"Direct scan · {result['message']}{folder}")
            if result["status"] == "OK" and result.get("folder"):
                self.start_estimate(result["folder"])
        threading.Thread(target=self.direct.run, daemon=True, kwargs=dict(
            start_path=paths[0], end_path=paths[1], clearance_m=clearance_m, tcp_speed_m_s=tcp_speed_m_s,
            velocity_scale=velocity_scale, sensor=sensor, on_progress=self.direct_status.set,
            on_finished=finished, return_pass=local)).start()

    def plan_centred_scan_poses(self):
        """New C / D: laser straight down over the groove centre, line across the seam, 30 mm past both plate ends
        (from the latest scan's estimate); joints by IK near the current C; old C / D backed up. No motion."""
        host = self.host
        if self.busy or self.capturing:
            return

        def run():
            try:
                import copy
                import shutil
                from datetime import datetime
                from geometry_msgs.msg import Pose
                from construct_robot.application.kiro_estimate import centred_scan_poses
                scan_dir = latest_scan_dir()
                poses = centred_scan_poses(scan_dir)
                old_c = load_scan_pose(self.scan_point_path("local_start"))
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                for which, key in (("local_start", "C"), ("local_end", "D")):
                    position, quat = poses[key]
                    target = Pose()
                    target.position.x, target.position.y, target.position.z = (float(v) for v in position)
                    target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = (float(v) for v in quat)
                    step = {"planning_group": "right_manipulator", "joint_names": list(old_c.joint_names),
                            "positions": tuple(old_c.positions)}
                    joints = host.node._closest_corrected_joint_target(step, target)
                    path = self.scan_point_path(which)
                    if path.is_file():
                        shutil.copy2(path, path.with_name(f"{path.stem}_before_centred_{stamp}.yaml"))
                    save_initial_state_yaml(path, "right_manipulator", list(old_c.joint_names), list(joints), copy.deepcopy(target),
                                            {"capture_source": "centred scan pose (KIRO estimate of " + Path(scan_dir).name + ")"})
                message = (f"Centred C / D saved (from {Path(scan_dir).name}): laser straight down over the groove centre, "
                           f"{poses['length_m'] * 1e3:.0f} mm seam + 30 mm margins, standoff {poses['standoff_m'] * 1e3:.0f} mm · "
                           "check with Plan, run the Local 1G scan, then Compute correction on it (new sensor pose)")
            except Exception as error:
                message = f"ERROR · centred scan poses: {error}"
            host.post(done, message)

        def done(message):
            self.direct_status.set(message)
            self.refresh_scan_poses()
            (host.error if message.startswith("ERROR") else host.log)(message)
        threading.Thread(target=run, daemon=True).start()

    def _post_weld_sensor(self):
        connection = self.host.scanner_connection_panel
        sensor = dict(ip=connection.ip.get().strip(), port=int(connection.port.get()),
                      calib_file=connection.calib.get().strip(), gain=int(connection.gain.get()),
                      backup_dir="~/scan_data/sensor_backup", out_root=str(post_weld_scan_root()))
        if not sensor["calib_file"]:
            raise ValueError("Set the calib YAML in 1. CONNECTION (sensor pose on the tool)")
        return sensor

    def add_post_weld_scan_to_sequence(self):
        """Append the post-weld scan row (local scan C -> D -> C, files only) to the Sequence Builder; no motion."""
        host = self.host
        try:
            if host.sequence_running:
                raise ValueError("a sequence is running")
            paths = [self.scan_point_path(which) for which in ("local_start", "local_end")]
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                raise ValueError(f"teach the scan poses C / D first (O / K, scan pose mode): missing {', '.join(missing)}")
            check_scan_line(load_scan_pose(paths[0]), load_scan_pose(paths[1]))
            step = post_weld_scan_step(
                paths[0], paths[1], tcp_speed_m_s=finite_values((self.scan_speed_mm_s.get(),), 1)[0] / 1000.0,
                velocity_scale=finite_values((host.velocity_percent.get(),), 1)[0] / 100.0,
                slot=next_sequential_slot(host.sequence_steps))
        except (ValueError, OSError, tk.TclError) as error:
            host.error(f"Post-weld scan: {error}")
            return
        host.sequence_model.extend([step])
        host.refresh_sequence_table(select_last=True)
        host.log(f"Post-weld scan added to the Sequence Builder · slot {step['parallel_slot']} · "
                 f"{step['tcp_speed_m_s'] * 1000.0:.0f} mm/s · saved under {post_weld_scan_root()}")

    def run_post_weld_scan(self, step, execute_requested):
        """Sequence worker thread: Plan = check the poses; Execute = C -> D -> C recording, files only."""
        try:
            start, end = load_scan_pose(step["start_path"]), load_scan_pose(step["end_path"])
            length, _tilt, _change = check_scan_line(start, end)
        except (OSError, ValueError) as error:
            return False, f"post-weld scan: {error}"
        if not execute_requested:
            return True, f"post-weld scan planned · C → D → C over {length * 1000.0:.0f} mm (no motion)"
        if self.busy or self.estimating:
            return False, "post-weld scan: the Scan tab is busy (scan / estimate running)"
        try:
            sensor = self._post_weld_sensor()
        except (ValueError, tk.TclError) as error:
            return False, f"post-weld scan: {error}"
        if self.direct is None:
            from construct_robot.nodes.scan_record_process import RecordSession
            self.direct = DirectScanController(self.host, RecordSession)
        done, outcome = threading.Event(), {}

        def finished(result):
            outcome.update(result)
            done.set()

        def progress(text):
            self.direct_status.set(f"Post-weld scan · {text}")
        self.direct.run(start_path=step["start_path"], end_path=step["end_path"], clearance_m=0.0,
                        tcp_speed_m_s=float(step["tcp_speed_m_s"]), velocity_scale=float(step["velocity_scale"]),
                        sensor=sensor, on_progress=progress, on_finished=finished, return_pass=True,
                        extra_meta={"purpose": "post_weld_scan", "estimate": "none (files only)"})
        done.wait()
        folder = outcome.get("folder")
        ok = outcome.get("status") == "OK"
        if folder and not record_cycle_event(self.host, "artifact", kind="post_weld_scan",
                                             pass_number=step.get("after_pass", 0), path=str(folder),
                                             status=outcome.get("status"), message=outcome.get("message")):
            return False, self.host._scan_cycle_log_error
        if ok and folder:
            try:
                ply = export_post_weld_ply(self.host, folder, step.get("after_pass", 0))
            except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
                return False, f"post-weld scan saved {folder}, PLY export failed: {error}"
            self.host.post(self.host.log, f"Post-weld PLY saved · {ply}")
        message = f"post-weld scan {outcome.get('status')} · {outcome.get('message')}" + (f" · saved {folder}" if folder else "")
        self.host.post(self.direct_status.set, message)
        return ok, message

    def local_cycle_scan_config(self):
        """Freeze local scan settings; Scan + 4 Passes always uses the saved correction and its learned layout."""
        host = self.host
        paths = [self.scan_point_path(which) for which in ("local_start", "local_end")]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise ValueError(f"teach the local scan poses C / D first (O / K, scan pose mode): missing {', '.join(missing)}")
        check_scan_line(load_scan_pose(paths[0]), load_scan_pose(paths[1]))
        speed = finite_values((self.scan_speed_mm_s.get(),), 1)[0] / 1000.0
        scale = max(0.01, min(1.0, finite_values((host.velocity_percent.get(),), 1)[0] / 100.0))
        common = dict(source="local_1g", start_path=str(paths[0]), end_path=str(paths[1]),
                      tcp_speed_m_s=speed, velocity_scale=scale)
        correction = scan_correction.load_correction()
        if correction is None:
            raise ValueError("No saved scan correction: save the P1-P4 truth correction before Scan + 4 Passes")
        scan_correction.check_reference_set_applied(correction)
        relative = (correction.get("reference_set") or {}).get("relative_model")
        if not relative:
            relative = "measured_layout" if correction.get("measured_layout_m") else scan_correction.RELATIVE_MODELS[0]
        if relative not in scan_correction.RELATIVE_MODELS:
            raise ValueError(f"Unknown saved correction layout: {relative}")
        self.apply_correction.set(True)
        self.relative_model.set(relative)
        self.show_correction()
        scan = dict(common, relative=relative, apply_correction=True,
                    plate_normal_world=finite_values(correction.get("plate_normal_world") or
                                                     (correction.get("seam_frame") or {}).get("n", ()), 3),
                    job=self.estimate_job.get().strip(), weld_direction=self.weld_direction.get(),
                    estimator=self.estimator.get())
        return scan, dict(common)

    def run_cycle_scan(self, cfg, cancel_requested=lambda: False):
        """Sequence worker: local 1G scan C -> D -> C, offline estimate, four-pass targets (written to
        weld_feedback/scan_four_pass_<scan> too). Returns {"scan_dir", "targets"}; raises on any failure (no weld)."""
        if cfg.get("apply_correction"):
            scan_correction.check_reference_set_applied(scan_correction.load_correction())
        try:
            sensor = self._post_weld_sensor()
        except (ValueError, tk.TclError) as error:
            raise RuntimeError(f"local scan: {error}") from error
        sensor.pop("out_root", None)                      # the regular scan folder (scan_data/kiro_scan)
        if self.direct is None:
            from construct_robot.nodes.scan_record_process import RecordSession
            self.direct = DirectScanController(self.host, RecordSession)
        done, outcome = threading.Event(), {}

        def finished(result):
            outcome.update(result)
            done.set()
        self.direct.run(start_path=cfg["start_path"], end_path=cfg["end_path"], clearance_m=0.0,
                        tcp_speed_m_s=float(cfg["tcp_speed_m_s"]), velocity_scale=float(cfg["velocity_scale"]),
                        sensor=sensor, on_progress=lambda text: self.host.post(self.direct_status.set, f"Cycle scan · {text}"),
                        on_finished=finished, return_pass=True, extra_meta={"purpose": "scan_four_pass_cycle"})
        done.wait()
        if outcome.get("folder") and not record_cycle_event(self.host, "artifact", kind="initial_scan", pass_number=0,
                                                            path=str(outcome["folder"]), status=outcome.get("status"),
                                                            message=outcome.get("message")):
            raise RuntimeError(self.host._scan_cycle_log_error)
        if outcome.get("status") != "OK" or not outcome.get("folder"):
            raise RuntimeError(f"local scan failed: {outcome.get('message')}")
        scan_dir = outcome["folder"]
        if cancel_requested():
            raise RuntimeError("stopped by operator after the local scan")
        covered, detail = check_scan_coverage(scan_dir)
        if not covered:
            raise RuntimeError(f"scan coverage: {detail}")
        estimator = cfg.get("estimator") or ESTIMATORS[0]
        self.host.post(self.host.log, f"Scan + 4 Passes · scan saved {scan_dir} · {estimator} estimate...")
        result = run_estimate(estimator, scan_dir, calib_file=sensor["calib_file"] or None, job=cfg.get("job") or None,
                              weld_direction=cfg.get("weld_direction"),
                              on_line=lambda line: self.host.post(self.estimate_text.set, f"Cycle estimate · {line}"))
        if not record_cycle_event(self.host, "estimate_result", status=result.get("status"),
                                  message=result.get("message"), ends=result.get("ends"),
                                  end_diagnostics=result.get("end_diagnostics"),
                                  scan_dir=str(scan_dir), out_dir=result.get("out_dir"),
                                  log=result.get("log", [])):
            raise RuntimeError(self.host._scan_cycle_log_error)
        if result.get("status") != "OK":
            missing = [name for name, end in (result.get("ends") or {}).items()
                       if end is None or end.get("two_sided") is False or end.get("sigma_m") is None]
            hint = (f" · unidentified seam end(s): {', '.join(missing)}; check precision edge fitting and sensor visibility"
                    if missing else "")
            coarse_only = [name for name in missing if ((result.get("end_diagnostics") or {}).get(name) or {}).get("coarse_boundary_identified")]
            if coarse_only:
                hint += f" · coarse boundary detected at {', '.join(coarse_only)}; precision edge unavailable"
            raise RuntimeError(f"estimate failed: {result.get('message')}{hint}")
        try:
            scan_correction.archive_to_history(scan_dir, note="Scan + 4 Passes estimate")
        except Exception as error:
            self.host.post(self.host.error, f"Scan estimate archive: {error}")
        correction = scan_correction.load_correction() if cfg.get("apply_correction") else None
        if correction is not None and not record_cycle_event(
                self.host, "scan_correction", computed=correction.get("computed"),
                source_scan=correction.get("source_scan"), anchor_pass=correction.get("anchor_pass", 1),
                reference_set_id=(correction.get("reference_set") or {}).get("id"),
                endpoint_adjustments_world_m=correction.get("endpoint_adjustments_world_m"),
                inferred_ends=correction.get("inferred_ends")):
            raise RuntimeError(self.host._scan_cycle_log_error)
        folder, targets = scan_four_pass.write_four_pass_teaching(scan_dir, correction, relative=cfg.get("relative") or scan_correction.RELATIVE_MODELS[0])
        if not record_cycle_event(self.host, "artifact", kind="four_pass_teaching", pass_number=0,
                                  path=str(folder), source_scan=str(scan_dir),
                                  weave_geometry=targets[4].get("weave_geometry")):
            raise RuntimeError(self.host._scan_cycle_log_error)
        self.host.post(self.host.log, f"Scan + 4 Passes · four-pass targets from {scan_dir} "
                                      f"({'corrected' if correction is not None else 'uncorrected'}) → {folder}")
        self.host.post(self.show_estimate, scan_dir)
        return {"scan_dir": scan_dir, "targets": targets}

    def browse_estimate_job(self):
        path = filedialog.askopenfilename(title="Hongik job YAML", filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            self.estimate_job.set(path)

    def start_estimate(self, scan_dir):
        """Hongik 1G estimate (ends S / E, pass start / goal TCP poses) of a scan; no robot motion."""
        if self.estimating:
            return
        try:
            scan_dir = scan_dir or latest_scan_dir()
        except ValueError as error:
            self.estimate_text.set(f"Hongik estimate: ERROR · {error}")
            return
        connection = getattr(self.host, "scanner_connection_panel", None)
        calib = connection.calib.get().strip() if connection is not None else ""
        job = self.estimate_job.get().strip()
        direction = self.weld_direction.get()
        self.estimating = True
        self.estimate_button.configure(state=tk.DISABLED)
        self.estimate_text.set(f"Hongik estimate: CALCULATING on {scan_dir} (profile fit + seam estimate, a few minutes)...")
        self.host.log(f"Hongik estimate · CALCULATING · {scan_dir}")

        def done(result):
            self.estimating = False
            self.estimate_button.configure(state=tk.NORMAL)
            text = estimate_summary(result)
            self.estimate_text.set(text)
            if result.get("status") == "OK":
                try:
                    dest = scan_correction.archive_to_history(scan_dir, note="Hongik estimate")
                    self.host.log(f"Scan estimate archived → {dest}")
                except Exception as error:
                    self.host.error(f"Scan estimate archive: {error}")
            (self.host.log if result.get("status") == "OK" else self.host.error)(text.splitlines()[0])
            self.show_estimate(scan_dir)
            self.refresh_estimated_poses(scan_dir)
            if result.get("status") == "OK":
                # every successful estimate gets its own weld_feedback/scan_four_pass_<scan> folder, loaded in 4-PASS
                self.estimate_scan_dir = scan_dir
                self.write_four_pass()

        estimator = self.estimator.get()

        def progress(line):
            self.estimate_text.set(f"{estimator} estimate: CALCULATING · {line}")
            self.host.log(f"{estimator} estimate · {line}")

        def run():
            try:
                covered, detail = check_scan_coverage(scan_dir)
                if not covered:
                    result = {"status": "REFUSED", "estimator": estimator, "message": detail}
                else:
                    result = run_estimate(estimator, scan_dir, calib_file=calib or None, job=job or None,
                                          weld_direction=direction, on_line=lambda line: self.host.post(progress, line))
                    if result.get("status") == "OK" and estimator == "hongik":
                        try:                                    # keep the pass layout template current
                            from construct_robot.application.kiro_estimate import build_template_from_hongik
                            build_template_from_hongik(scan_dir)
                        except Exception as error:
                            self.host.post(self.host.error, f"Hongik pass template update: {error}")
            except Exception as error:
                result = {"status": "ERROR", "message": str(error)}
            self.host.post(done, result)
        threading.Thread(target=run, daemon=True).start()

    def _build_reference_correction(self, box):
        # Restore the saved position calibration; bead centres use the operator's root-relative design layout.
        self.apply_correction = tk.BooleanVar(value=scan_correction.correction_path().is_file())
        saved_correction = scan_correction.load_correction() or {}
        saved_model = (saved_correction.get("reference_set") or {}).get("relative_model")
        self.relative_model = tk.StringVar(value=saved_model or scan_correction.RELATIVE_MODELS[0])
        self.correction_text = tk.StringVar(value="")
        self.reference_pass = tk.StringVar(value="1")
        self.correction_anchor = tk.StringVar(value=str(saved_correction.get("anchor_pass", 1)))
        self.scan_key_mode = tk.StringVar(value="reference")
        self.four_pass_text = tk.StringVar(value="")
        ref = ttk.LabelFrame(box, text="Position calibration · P1-P4 truth: one measured point per pass",
                             padding=6)
        ref.pack(fill=tk.X, pady=(6, 0))
        row = ttk.Frame(ref)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Pass").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.reference_pass, values=("1", "2", "3", "4"), state="readonly",
                     width=3).pack(side=tk.LEFT, padx=(3, 6))
        for end in ("start", "goal"):
            ttk.Button(row, text=f"Capture current TCP → {end} ref",
                       command=lambda e=end: self.capture_reference(e)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Checkbutton(row, text="Apply correction to Plan / Execute / RViz", variable=self.apply_correction,
                        command=self.show_correction).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Write corrected KIRO files", command=self.write_corrected).pack(side=tk.LEFT, padx=6)
        row2 = ttk.Frame(ref)
        row2.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(row2, text="Position anchor pass:").pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.correction_anchor, values=("1", "2"), state="readonly",
                     width=3).pack(side=tk.LEFT, padx=(3, 6))
        ttk.Button(row2, text="Compute correction", command=self.compute_correction).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(row2, text="P2: root = measured P2 wire tip minus design offset; other saved XYZ ignored.",
                  foreground="#5f6368").pack(side=tk.LEFT)
        truth_row = ttk.Frame(ref)
        truth_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(truth_row, text="New P1-P4 reference set (current scan)",
                   command=self.begin_four_pass_references).pack(side=tk.LEFT)
        ttk.Label(truth_row, text="Same specimen placement as this scan. START: I/O/P/J; GOAL: K/L/M/U. "
                  "One end = translation at the other; both ends = separate corrections. Compute, then Write 4-pass.",
                  foreground="#5f6368", wraplength=770).pack(side=tk.LEFT, padx=8)
        row2 = ttk.Frame(ref)
        row2.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(row2, text="Passes 2-4 relative to the root:").pack(side=tk.LEFT)
        relative = ttk.Combobox(row2, textvariable=self.relative_model, values=scan_correction.RELATIVE_MODELS,
                                state="readonly", width=12)
        relative.pack(side=tk.LEFT, padx=(3, 8))
        relative.bind("<<ComboboxSelected>>", lambda _e: (self.show_correction(), self.show_estimate(None)))
        ttk.Label(row2, text="design_layout = the operator's bead layout (pass_design_layout.yaml); "
                  "hongik_tcp = Hongik's TCP displacement from its root TCP (its per-pass torch angles inside); "
                  "hongik_aim = Hongik's wire-tip displacement, TCP from the execution orientation.",
                  foreground="#5f6368", wraplength=820).pack(side=tk.LEFT)
        keys = ttk.Frame(ref)
        keys.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(keys, text="Keyboard keys (Scan tab, right-arm Keyboard Teaching) save:").pack(side=tk.LEFT)
        for mode, text in SCAN_KEY_MODES.items():
            ttk.Radiobutton(keys, text=text, value=mode, variable=self.scan_key_mode).pack(side=tk.LEFT, padx=(6, 0))
        four = ttk.LabelFrame(ref, text="Four-pass teaching · corrected positions + taught pass orientations · "
                              f"waits = pass-4 wire tip + {scan_four_pass.WAIT_LIFT_M * 1e3:.0f} mm World Z", padding=4)
        four.pack(fill=tk.X, pady=(4, 0))
        row3 = ttk.Frame(four)
        row3.pack(fill=tk.X)
        for passes in ORIENTATION_KEYS.values():
            name = " + ".join(f"P{k}" for k in passes)
            ttk.Button(row3, text=f"Capture torch → {name} orientation",
                       command=lambda p=passes: self.capture_orientation(p)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row3, text="Write 4-pass teaching → load in Welding 4-PASS",
                   command=self.write_four_pass).pack(side=tk.LEFT, padx=(10, 6))
        ttk.Label(four, textvariable=self.four_pass_text, wraplength=1060, justify=tk.LEFT,
                  font=("TkFixedFont", 9)).pack(anchor=tk.W)
        self.tcp_cal_text = tk.StringVar(value="")
        cal = ttk.LabelFrame(ref, text="TCP calibration · wire tip on ONE fixed point, change only the torch orientation "
                             "(tilt forward, sideways, twist; 4+ captures)", padding=4)
        cal.pack(fill=tk.X, pady=(4, 0))
        row4 = ttk.Frame(cal)
        row4.pack(fill=tk.X)
        ttk.Button(row4, text="Add sample (current TCP)", command=self.capture_tcp_sample).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row4, text="Compute + save wire tip", command=self.compute_tcp_calibration).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row4, text="Clear samples", command=self.clear_tcp_samples).pack(side=tk.LEFT, padx=(0, 6))
        row5 = ttk.Frame(cal)
        row5.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(row5, text="Automatic touch (flat surface, Fastech DI0):").pack(side=tk.LEFT)
        self.touch_tilt_deg = tk.StringVar(value="35")
        ttk.Label(row5, text="max tilt °").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Spinbox(row5, from_=10, to=35, increment=5, width=4, textvariable=self.touch_tilt_deg).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Button(row5, text="Touch calibration: start (ROBOT MOVES)", command=self.start_touch_calibration).pack(side=tk.LEFT, padx=(4, 6))
        ttk.Button(row5, text="Stop", command=self.stop_touch_calibration).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(cal, text="Put the wire tip 5-15 mm above a flat area (about 40 x 40 mm free, e.g. the plate top), torch "
                  "roughly as for welding, press Start. The robot touches 3 spots with this orientation (plane) and 8 tilts "
                  "(± max and ± max/2 about two tool axes; smaller tilts = less accurate wire length), slowly (4 mm/s, max 30 mm down), latching the TCP at each DI0 edge; the wire "
                  "tip is then computed, saved and the correction refreshed. Re-run after a wire cut / change.",
                  foreground="#5f6368", wraplength=1060).pack(anchor=tk.W)
        ttk.Label(cal, textvariable=self.tcp_cal_text, wraplength=1060, justify=tk.LEFT).pack(anchor=tk.W)
        self.show_tcp_calibration()
        ttk.Label(ref, text="Put the wire where pass k must start / end with the torch orientation you will weld with, "
                  "capture, then Compute. The end a reference belongs to is decided by its position (the weld direction "
                  "may swap start / goal). Anchor P1: root wire-tip offsets plus measured pass remainders. "
                  "Anchor P2: infer the root from P2's design position, then use design_layout for all passes; "
                  "old P1 / P3 / P4 XYZ are ignored. Position calibration is for a new cycle, not correction during welding.",
                  foreground="#5f6368", wraplength=1060).pack(anchor=tk.W, pady=(3, 0))
        ttk.Label(ref, textvariable=self.correction_text, wraplength=1060, justify=tk.LEFT).pack(anchor=tk.W)
        self.show_correction()

    def active_correction(self):
        if not self.apply_correction.get():
            return None
        correction = scan_correction.load_correction()
        scan_correction.check_reference_set_applied(correction)
        return correction

    def show_correction(self):
        lines = []
        try:
            refs = scan_correction.load_references()
            capture_set = scan_correction.load_reference_set()
            if capture_set:
                lines.append(f"P1-P4 truth set {capture_set['id']} · scan {capture_set['source_scan']}")
                lines.append("Captured: " + " · ".join(
                    f"P{k} " + ("/".join(r["label"] for r in refs if r["pass"] == k) or "MISSING")
                    for k in range(1, 5)))
            lines.append("References (World mm, as saved): " + (" · ".join(
                f"P{r['pass']} {r['label']} ({', '.join(f'{v * 1e3:.1f}' for v in r['position'])})" for r in refs)
                or "none"))
        except (OSError, ValueError) as error:
            lines.append(f"References: {error}")
        doc = scan_correction.load_correction()
        if doc is not None:
            ra = doc.get("reference_torch_angles_deg", {}).get("start", {})
            ha = doc.get("hongik_root_torch_angles_deg", {})
            lines.append(f"Correction from {doc.get('source_scan')} ({doc.get('computed')}) · "
                         f"{'APPLIED' if self.apply_correction.get() else 'not applied'} · "
                         f"position anchor P{doc.get('anchor_pass', 1)}")
            try:
                scan_correction.check_reference_set_applied(doc)
            except ValueError as error:
                lines.append(f"  PENDING · {error}")
            if doc.get("anchor_pass") == 2:
                lines.append("  P2 design from root (along / lateral toward 90° face / up, mm): "
                             + str([round(v * 1e3, 3) for v in doc["anchor_design_m"]]))
            lines += [f"  {line}" for line in doc.get("report", [])]
            lines += [f"  note: {line}" for line in doc.get("notes", [])]
            for k, ends in doc.get("inferred_ends", {}).items():
                if ends:
                    lines.append(f"  P{k} {', '.join(ends)}: inferred from the measured end; not measured")
            if ra and ha:
                lines.append(f"  torch angles (travel / work, deg): reference {ra.get('travel', 0):+.1f} / {ra.get('work', 0):+.1f}"
                             f" · Hongik root {ha.get('travel', 0):+.1f} / {ha.get('work', 0):+.1f}")
            lines += self._targets_lines(doc)
        self.correction_text.set("\n".join(lines))
        self.show_four_pass()

    def _targets_lines(self, correction):
        """Target TCPs (reference orientation) of both relative models, against Hongik's TCPs."""
        try:
            scan_dir = self.estimate_scan_dir or latest_scan_dir()
            rows = scan_correction.targets_table(scan_dir, None, correction) if self.apply_correction.get() else \
                scan_correction.targets_table(scan_dir, scan_correction.reference_quat(correction, "start"), None)
        except Exception as error:
            return [f"  targets: {error}"]

        def mm(v):
            return "(" + ", ".join(f"{x * 1e3:.1f}" for x in v) + ")"
        lines = [f"  Target TCP with the reference orientation (World mm) · selected: {self.relative_model.get()}"]
        selected = self.relative_model.get()
        for r in rows:
            models = " | ".join(f"{'*' if m == selected else ' '}{m} {mm(r[m])}" for m in scan_correction.RELATIVE_MODELS)
            lines.append(f"   P{r['k']} {r['end']:<5} Hongik raw {mm(r['raw_tcp'])} | {models}")
        return lines

    def show_four_pass(self):
        """Taught orientations (torch angles in this estimate's seam frame) and the four-pass targets."""
        if not hasattr(self, "four_pass_text"):
            return
        orientations = scan_four_pass.load_pass_orientations()
        lines = []
        try:
            scan_dir = self.estimate_scan_dir or latest_scan_dir()
            angles = scan_four_pass.orientation_angles(scan_dir, orientations) if orientations else {}
        except Exception as error:
            scan_dir, angles = None, {}
            lines.append(f"angles: {error}")
        lines.append("Orientations: " + " · ".join(
            (f"P{k} travel {angles[k][0]:+.1f} / work {angles[k][1]:+.1f} deg ({orientations[k]['saved']})"
             if k in angles else f"P{k} " + ("taught" if k in orientations else "NOT taught"))
            for k in scan_four_pass.PASSES))
        if scan_dir is not None and len(orientations) == len(scan_four_pass.PASSES):
            try:
                targets = scan_four_pass.four_pass_targets(scan_dir, self.active_correction(), orientations,
                                                           self.relative_model.get())
                for k, row in targets.items():
                    lines.append(f"P{k} " + " | ".join(
                        f"{endpoint} ({', '.join(f'{v * 1e3:.1f}' for v in row[endpoint][0])})"
                        for endpoint, _name in scan_four_pass.ENDPOINT_POSES))
            except Exception as error:
                lines.append(f"targets: {error}")
        self.four_pass_text.set("\n".join(lines))

    def show_tcp_calibration(self, message=""):
        samples = tcp_calibration.load_samples()
        offset = tcp_calibration.load_offset()
        lines = [message] if message else []
        lines.append(f"{len(samples)} sample(s) · wire tip in use: " + (
            f"calibrated {np.round(offset * 1e3, 2).tolist()} mm (TCP frame)" if offset is not None else
            "Hongik 21.2 mm along the torch axis (not calibrated)"))
        self.tcp_cal_text.set("\n".join(lines))

    def capture_tcp_sample(self):
        """One calibration capture (measured joints FK); no motion."""
        if self.busy or self.capturing:
            return
        node = self.host.node
        self.capturing = True

        def run():
            try:
                _names, _positions, tcp, _prov = node.capture_measured_teaching_snapshot("right_manipulator", "tcp_calibration")
                n = tcp_calibration.add_sample((tcp.position.x, tcp.position.y, tcp.position.z),
                                               (tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w),
                                               "GUI capture (measured joints FK)")
                message = f"TCP calibration sample {n} saved"
            except Exception as error:
                message = f"ERROR · TCP calibration capture failed: {error}"
            self.host.post(done, message)

        def done(message):
            self.capturing = False
            self.show_tcp_calibration(message)
            (self.host.error if message.startswith("ERROR") else self.host.log)(f"TCP calibration · {message}")
        threading.Thread(target=run, daemon=True).start()

    def start_touch_calibration(self):
        """Automatic plane-touch wire-tip calibration on a worker thread (ROBOT MOVES, touch guarded)."""
        host = self.host
        if self.busy or self.capturing or getattr(self, "touch_run", None) is not None:
            return
        blocker = self._motion_blocker(moves_robot=True, check_live=False)
        if blocker:
            self.show_tcp_calibration(f"ERROR · {blocker}")
            return
        try:
            tilt = float(self.touch_tilt_deg.get())
            if not 5.0 <= tilt <= 35.0:
                raise ValueError
        except (ValueError, tk.TclError):
            self.show_tcp_calibration("ERROR · max tilt must be 5..35 deg")
            return
        run = tcp_calibration.TouchCalibrationRun(host)
        self.touch_run = run
        scale = min(0.15, float(host.velocity_percent.get()) / 100.0)
        model = scan_correction.wire_tip_model(0.0212)
        host.log("TCP calibration · automatic touch calibration started · ROBOT MOVES")

        def worker():
            try:
                result = run.run(model, scale, lambda text: host.post(self.show_tcp_calibration, f"Touch calibration · {text}"),
                                 max_tilt_deg=tilt)
                message = (f"touch calibration done · wire tip {np.round(result['tip_tcp_m'] * 1e3, 2).tolist()} mm "
                           f"(TCP frame) ± {np.round(result['tip_sigma_m'] * 1e3, 2).tolist()} mm (1σ), rms "
                           f"{result['rms_m'] * 1e3:.2f} mm, tilt spread {result['spread_deg']:.0f} deg"
                           + (" · correction refreshed" if result.get("correction_refreshed") else "")
                           + (f" · correction refresh failed: {result['correction_error']}" if result.get("correction_error") else ""))
            except Exception as error:
                message = f"ERROR · touch calibration: {error}"
            host.post(done, message)

        def done(message):
            self.touch_run = None
            self.show_tcp_calibration(message)
            (host.error if message.startswith("ERROR") else host.log)(f"TCP calibration · {message}")
        threading.Thread(target=worker, daemon=True).start()

    def stop_touch_calibration(self):
        run = getattr(self, "touch_run", None)
        if run is not None:
            run.cancel()

    def compute_tcp_calibration(self):
        try:
            result = tcp_calibration.compute_and_save()
        except Exception as error:
            self.show_tcp_calibration(f"ERROR · {error}")
            return
        message = (f"wire tip {np.round(result['tip_tcp_m'] * 1e3, 2).tolist()} mm (TCP frame), rms "
                   f"{result['rms_m'] * 1e3:.2f} mm over {len(result['residuals_m'])} samples, tilt spread "
                   f"{result['spread_deg']:.0f} deg → {result['file']} · press Compute correction (and re-run "
                   "Write 4-pass teaching)")
        self.host.log(f"TCP calibration · {message}")
        self.show_tcp_calibration(message)

    def clear_tcp_samples(self):
        tcp_calibration.clear_samples()
        self.show_tcp_calibration("samples cleared (archived in history)")

    def capture_orientation(self, passes):
        """Store the current torch orientation (measured joints FK) for passes; position is not used."""
        if self.busy or self.capturing:
            return
        node = self.host.node
        passes = tuple(passes)
        name = "+".join(f"P{k}" for k in passes)
        self.capturing = True
        self.four_pass_text.set(f"Capturing {name} torch orientation from measured joints...")

        def run():
            try:
                names, positions, tcp, _prov = node.capture_measured_teaching_snapshot(
                    "right_manipulator", f"pass{'_'.join(map(str, passes))}_orientation")
                path = scan_four_pass.save_pass_orientation(
                    passes, (tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w),
                    (names, positions), (tcp.position.x, tcp.position.y, tcp.position.z),
                    "GUI capture (measured joints FK)")
                message = f"{name} torch orientation saved → {path}"
            except Exception as error:
                message = f"ERROR · {name} orientation capture failed: {error}"
            self.host.post(done, message)

        def done(message):
            self.capturing = False
            self.show_four_pass()
            self.four_pass_text.set(message + "\n" + self.four_pass_text.get())
            (self.host.error if message.startswith("ERROR") else self.host.log)(f"Four-pass · {message}")
        threading.Thread(target=run, daemon=True).start()

    def write_four_pass(self):
        """pass_1..4.yaml from the corrected estimate + taught orientations, then load them in Welding › 4-PASS."""
        host = self.host
        try:
            folder, _targets = scan_four_pass.write_four_pass_teaching(
                self.estimate_scan_dir or latest_scan_dir(), self.active_correction(),
                relative=self.relative_model.get())
        except Exception as error:
            self.four_pass_text.set(f"ERROR · four-pass teaching: {error}")
            host.error(f"Four-pass teaching: {error}")
            return
        host.log(f"Four-pass · wrote pass_1..4.yaml → {folder}")
        loaded = False
        if getattr(host, "four_pass_folder", None) is not None and hasattr(host, "load_four_pass_references"):
            host.four_pass_folder.set(str(folder))
            loaded = bool(host.load_four_pass_references())
        self.show_four_pass()
        self.four_pass_text.set(f"Wrote {folder}/pass_1..4.yaml" + (" · loaded in Welding › 4-PASS (select a pass and "
                                "apply it there; joints are resolved by IK)" if loaded else "") + "\n"
                                + self.four_pass_text.get())

    def begin_four_pass_references(self):
        if self.busy or self.capturing or getattr(self.host, "sequence_running", False):
            return
        try:
            scan = self.estimate_scan_dir or latest_scan_dir()
            metadata = scan_correction.begin_reference_set(scan, "measured_layout", baseline_relative=self.relative_model.get())
        except Exception as error:
            self.correction_text.set(f"ERROR · new reference set: {error}")
            return
        self.estimate_scan_dir = metadata["source_scan"]
        self.correction_anchor.set("1")
        self.relative_model.set("measured_layout")
        self.scan_key_mode.set("reference")
        self.show_correction()
        self.host.log(f"P1-P4 truth capture · scan {metadata['source_scan']} · previous values saved in {metadata['backup']}")

    def capture_reference(self, end, pass_k=None):
        if self.busy or self.capturing or getattr(self.host, "sequence_running", False):
            return
        node = self.host.node
        pass_k = int(pass_k if pass_k is not None else self.reference_pass.get())
        self.capturing = True
        self.correction_text.set(f"Capturing P{pass_k} {end} reference from measured joints...")

        def run():
            try:
                names, positions, tcp, _prov = node.capture_measured_teaching_snapshot(
                    "right_manipulator", f"pass{pass_k}_ref_{end}")
                extra = {"joints": (names, positions)} if scan_correction.load_reference_set() else {}
                path = scan_correction.save_reference(
                    pass_k, end, (tcp.position.x, tcp.position.y, tcp.position.z),
                    (tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w),
                    "GUI capture (measured joints FK)", **extra)
                message = f"P{pass_k} {end} reference saved → {path}"
            except Exception as error:
                message = f"ERROR · P{pass_k} {end} reference capture failed: {error}"
            self.host.post(done, message)

        def done(message):
            self.capturing = False
            self.show_correction()
            self.correction_text.set(message + "\n" + self.correction_text.get())
            (self.host.error if message.startswith("ERROR") else self.host.log)(f"Scan correction · {message}")
        threading.Thread(target=run, daemon=True).start()

    def compute_correction(self):
        if self.busy or self.capturing or getattr(self.host, "sequence_running", False):
            return
        try:
            capture_set = scan_correction.load_reference_set()
            if capture_set:
                doc = scan_correction.compute_reference_set_correction()
                scan_dir = doc["source_scan"]
                self.estimate_scan_dir = scan_dir
                self.correction_anchor.set("1")
                self.relative_model.set(capture_set["relative_model"])
            else:
                scan_dir = self.estimate_scan_dir or latest_scan_dir()
                doc = scan_correction.compute_correction(scan_dir, anchor_pass=int(self.correction_anchor.get()))
        except Exception as error:
            self.correction_text.set(f"ERROR · correction: {error}")
            return
        self.apply_correction.set(True)
        if doc.get("anchor_pass") == 2:
            self.relative_model.set("design_layout")
        self.show_correction()
        self.host.log("Scan correction · " + " | ".join(doc["report"][:2]))
        self.show_estimate(scan_dir)

    def write_corrected(self):
        doc = scan_correction.load_correction()
        if doc is None:
            self.correction_text.set("ERROR · compute the correction first")
            return
        try:
            scan_correction.check_reference_set_applied(doc)
            files = scan_correction.write_corrected_kiro(self.estimate_scan_dir or latest_scan_dir(), doc,
                                                         relative=self.relative_model.get())
        except Exception as error:
            self.correction_text.set(f"ERROR · {error}")
            return
        self.host.log(f"Scan correction · wrote {len(files)} corrected pass files → {Path(files[0]).parent}")
        self.correction_text.set(f"Wrote {Path(files[0]).parent} (reference torch orientation)\n" + self.correction_text.get())

    def refresh_estimated_poses(self, scan_dir):
        try:
            scan_dir = scan_dir or latest_scan_dir()
            self.estimate_scan_dir = scan_dir
            self.estimated_poses = estimate_poses(scan_dir)
        except Exception as error:
            self.estimated_poses = []
            self.host.error(f"Estimated poses: {error}")
        labels = [label for label, _pose in self.estimated_poses]
        self.pose_combo.configure(values=labels)
        if labels and self.pose_choice.get() not in labels:
            self.pose_choice.set(labels[0])
        elif not labels:
            self.pose_choice.set("")

    def move_to_estimated_pose(self, execute):
        """Plan or execute (explicit button) a TCP line to the selected estimated pose via the sequence runner."""
        host = self.host
        blocker = self._motion_blocker(moves_robot=execute, check_live=False)
        if self.busy or self.capturing or blocker:
            self.estimate_text.set(f"ERROR · {blocker or 'wait for the active scan / capture'}")
            return
        label = self.pose_choice.get()
        if label not in dict(self.estimated_poses):
            self.estimate_text.set("ERROR · select an estimated pose (Refresh after an estimate)")
            return
        try:
            correction = self.active_correction()
            current = host.node._current_tcp_pose("right_manipulator")
            target, info = scan_correction.estimated_move_target(
                self.estimate_scan_dir or latest_scan_dir(), label, current, correction, self.relative_model.get(),
                scan_four_pass.load_pass_orientations())
            step = pose_move_step(current, target, velocity_scale=float(host.velocity_percent.get()) / 100.0,
                                  label=f"Scan estimate {label}", keep_orientation=not info["pass_orientation"])
        except Exception as error:
            self.estimate_text.set(f"ERROR · {error}")
            return
        note = ("corrected" if info["corrected"] else "uncorrected") + f", {info.get('relative', 'root')}"
        if info.get("angle_to_reference_deg") is not None and info["angle_to_reference_deg"] > 3.0:
            note += (f"; the torch is {info['angle_to_reference_deg']:.1f} deg from the reference torch - "
                     "a stick-out error no longer cancels")
        p = target.position
        host.log(f"Scan estimate · {'EXECUTE' if execute else 'PLAN'} → {label} TCP ({p.x * 1e3:.1f}, {p.y * 1e3:.1f}, "
                 f"{p.z * 1e3:.1f}) mm, {'taught pass orientation' if info['pass_orientation'] else 'current orientation'}, "
                 f"{note}")
        self.estimate_text.set(f"{'EXECUTE' if execute else 'PLAN'} {label}: TCP ({p.x * 1e3:.1f}, {p.y * 1e3:.1f}, "
                               f"{p.z * 1e3:.1f}) mm · {note}")
        host.run_sequence(True, execute, steps_override=[step])

    def show_estimate(self, scan_dir):
        """RViz: scan cloud + estimate markers (display only)."""
        try:
            scan_dir = scan_dir or latest_scan_dir()
            if getattr(self, "viz", None) is None:
                from construct_robot.nodes.scan_estimate_markers import EstimateViz
                self.viz = EstimateViz(self.host.node)
            corrected, refs = [], []
            correction = self.active_correction()
            if correction is not None:                # corrected TCP lines (reference orientation, selected model)
                by_pass = {}
                for r in scan_correction.targets_table(scan_dir, None, correction):
                    by_pass.setdefault(r["k"], {})[r["end"]] = r[self.relative_model.get()]
                corrected = [(k, ends["start"], ends["goal"]) for k, ends in sorted(by_pass.items())]
                refs = [(f"P{r['pass']} ref", r["position"]) for r in scan_correction.load_references()]
            ends, passes = self.viz.show(scan_dir, corrected, refs)
        except Exception as error:
            self.host.error(f"RViz estimate view: {error}")
            return
        self.host.log(f"RViz · {scan_dir}: {ends} end(s), {passes} pass(es) on /hongik_scanner/estimate_markers, "
                      "scan on /hongik_scanner/scan_cloud")

    def cancel_direct_scan(self):
        if self.direct is not None:
            self.direct.cancel()
            self.direct_status.set("Cancel requested · stopping the robot")

    def _set_busy(self, busy, direct=False, lock_tabs=True):
        self.busy = busy
        if hasattr(self, "direct_button"):
            self.direct_button.configure(state=tk.DISABLED if busy else tk.NORMAL)
            self.local_button.configure(state=tk.DISABLED if busy else tk.NORMAL)
            self.direct_cancel_button.configure(state=tk.NORMAL if busy and direct else tk.DISABLED)
        for button in (self.scan_button, self.load_button, self.post_weld_button):
            button.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.cancel_button.configure(state=tk.NORMAL if busy else tk.DISABLED)
        self.post_weld_cancel_button.configure(state=tk.NORMAL if busy else tk.DISABLED)
        if not busy:
            self.set_awaiting_confirmation(False)
        # Lock the other operator pages and the other TASK sub-pages while the robot scans.
        notebooks = (self.host.operator_notebook, self.host.task_notebook)
        if busy and lock_tabs:
            self._tab_states = {}
            for notebook in notebooks:
                selected = notebook.select()
                for tab in notebook.tabs():
                    if tab != selected:
                        self._tab_states[(notebook, tab)] = notebook.tab(tab, "state")
                        notebook.tab(tab, state=tk.DISABLED)
        elif not busy:
            for (notebook, tab), state in self._tab_states.items():
                notebook.tab(tab, state=state)
            self._tab_states = {}

    def _build_post_weld(self, parent):
        self.post_weld_action = tk.StringVar(value="/hongik_fine_sensing/post_weld_scan")
        self.confirm_service = tk.StringVar(value="/hongik_fine_sensing/hongik_auto_teach_server/confirm_move")
        self.post_weld_run = tk.StringVar(value="")
        self.post_weld_pass = tk.StringVar(value="0")
        self.post_weld_speed = tk.StringVar(value="0")
        self.post_weld_confirm = tk.StringVar(value="each")
        self.post_weld_direction = tk.StringVar(value="Auto")
        self.post_weld_orientation = tk.StringVar(value="Sensing pass")
        self.post_weld_dry_run = tk.BooleanVar(value=True)
        self.post_weld_status = tk.StringVar(value="Raw-data scan of a finished 1G run (no estimation, no weld-plan change).")
        box = ttk.LabelFrame(parent, text="Post-weld raw scan · /post_weld_scan (ROBOT MOVES unless dry run)", padding=8)
        box.pack(fill=tk.X, pady=5)
        for row, (label, var) in enumerate((("Action", self.post_weld_action), ("Confirm service", self.confirm_service),
                                           ("Run dir ('' = latest)", self.post_weld_run))):
            ttk.Label(box, text=label).grid(row=row, column=0, sticky=tk.W)
            ttk.Entry(box, textvariable=var, width=76).grid(row=row, column=1, columnspan=7, sticky=tk.EW, pady=2)
        options = ttk.Frame(box)
        options.grid(row=3, column=0, columnspan=8, sticky=tk.W, pady=3)
        ttk.Label(options, text="After pass (0 = final)").pack(side=tk.LEFT)
        ttk.Spinbox(options, from_=0, to=20, textvariable=self.post_weld_pass, width=4).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(options, text="Speed mm/s (0 = run's)").pack(side=tk.LEFT)
        ttk.Entry(options, textvariable=self.post_weld_speed, width=7).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(options, text="Confirm").pack(side=tk.LEFT)
        ttk.Combobox(options, textvariable=self.post_weld_confirm, values=("each", "auto"),
                     state="readonly", width=6).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Checkbutton(options, text="Dry run (plan only, no motion)", variable=self.post_weld_dry_run).pack(side=tk.LEFT)
        direction_row = ttk.Frame(box)
        direction_row.grid(row=4, column=0, columnspan=8, sticky=tk.W, pady=3)
        ttk.Label(direction_row, text="Scan direction").pack(side=tk.LEFT)
        ttk.Combobox(direction_row, textvariable=self.post_weld_direction, values=("Auto", "S->E", "E->S"),
                     state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(direction_row, text="Tool orientation").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Combobox(direction_row, textvariable=self.post_weld_orientation, values=("Sensing pass", "Least rotation"),
                     state="readonly", width=16).pack(side=tk.LEFT, padx=5)
        ttk.Label(box, text="Auto starts at the nearer end. Sensing pass = the run's sensing view (default); "
                  "Least rotation = the nearer torch side. From the weld pose: lift along the measured plate normal first. "
                  "S/E are the run's measured ends.", wraplength=1060).grid(row=5, column=0, columnspan=8, sticky=tk.W)
        buttons = ttk.Frame(box)
        buttons.grid(row=6, column=0, columnspan=8, sticky=tk.W, pady=3)
        self.post_weld_button = ttk.Button(buttons, text="Start post-weld scan", command=self.start_post_weld)
        self.post_weld_button.pack(side=tk.LEFT, padx=(0, 6))
        self.post_weld_cancel_button = ttk.Button(buttons, text="Cancel scan", command=self.cancel_scan, state=tk.DISABLED)
        self.post_weld_cancel_button.pack(side=tk.LEFT, padx=6)
        self.confirm_yes_button = ttk.Button(buttons, text="CONFIRM move", state=tk.DISABLED,
                                             command=lambda: self.answer_confirmation(True))
        self.confirm_yes_button.pack(side=tk.LEFT, padx=(18, 6))
        self.confirm_no_button = ttk.Button(buttons, text="Reject move", state=tk.DISABLED,
                                            command=lambda: self.answer_confirmation(False))
        self.confirm_no_button.pack(side=tk.LEFT, padx=6)
        ttk.Label(box, textvariable=self.post_weld_status, wraplength=1060).grid(
            row=7, column=0, columnspan=8, sticky=tk.W)

    def start_post_weld(self):
        if self.busy:
            return
        dry_run = bool(self.post_weld_dry_run.get())
        blocker = self._motion_blocker(moves_robot=not dry_run)
        if blocker:
            self.post_weld_status.set(f"ERROR · {blocker}")
            return
        try:
            speed_mps = finite_values((self.post_weld_speed.get(),), 1)[0] / 1000.0
            after_pass = int(self.post_weld_pass.get())
            orientation = {"Sensing pass": "pass", "Least rotation": "least_rotation"}.get(self.post_weld_orientation.get())
            if orientation is None:
                raise ValueError("Select Sensing pass or Least rotation")
        except ValueError as error:
            self.post_weld_status.set(f"ERROR · {error}")
            return
        args = dict(run_dir=self.post_weld_run.get(), after_pass=after_pass, measure_speed_mps=speed_mps,
                    confirm_mode=self.post_weld_confirm.get(), dry_run=dry_run,
                    direction="" if self.post_weld_direction.get() == "Auto" else self.post_weld_direction.get(),
                    tool_orientation=orientation,
                    action_name=self.post_weld_action.get().strip(),
                    status_service=self.status_service.get().strip(), log_topic=self.log_topic.get().strip(),
                    on_progress=self.post_weld_status.set, on_confirm=self.set_awaiting_confirmation,
                    on_result=self.show_post_weld_result,
                    on_error=lambda text: self.post_weld_status.set(f"ERROR · {text}"),
                    on_finished=lambda: self._set_busy(False))
        self._set_busy(True)
        self.post_weld_status.set("Dry run: planning only..." if dry_run
                                  else "Connecting to Hongik server; post-weld scan includes robot motion...")
        threading.Thread(target=self.bridge.post_weld_scan, kwargs=args, daemon=True).start()

    def show_post_weld_result(self, result):
        self.post_weld_status.set(f"{result.get('status', '')} · {result.get('message', '')} · "
                                  f"out_dir={result.get('out_dir', '') or '-'}")

    def set_awaiting_confirmation(self, awaiting):
        state = tk.NORMAL if awaiting and self.busy else tk.DISABLED
        self.confirm_yes_button.configure(state=state)
        self.confirm_no_button.configure(state=state)

    def answer_confirmation(self, accept):
        service = self.confirm_service.get().strip()
        self.set_awaiting_confirmation(False)
        self._worker(lambda: self.bridge.confirm_move(service, accept),
                     lambda reply: self.post_weld_status.set(
                         f"{'Confirmed' if accept else 'Rejected'} · {reply[1] or reply[0]}"))

    def _motion_blocker(self, moves_robot=True, check_live=True):
        """Reason a scan may not start now, or None."""
        host = self.host
        if host.sequence_running or host.seam_auto_running or host.node.active_motion_goal is not None:
            return "Stop the current motion / sequence before starting a scan"
        if host.keyboard_velocity_arm is not None or host.keyboard_velocity_switching or host.keyboard_jog_enabled.get():
            return "Disable Keyboard Teaching before scanning"
        live = getattr(host.node, "scanner_live", None)
        if check_live and live is not None and live.active:
            return "Stop scanner live view (1. CONNECTION) first: the sensor accepts one client"
        if moves_robot and (not host.execution_allowed or not host.robot_connected.get("right", False)):
            return "Right robot must be connected and motion execution enabled"
        return None

    def start_scan(self):
        if self.busy:
            return
        blocker = self._motion_blocker()
        if blocker:
            self.show_error(blocker)
            return
        try:
            a = tuple(v / 1000.0 for v in finite_values((var.get() for var in self.a), 3))
            b = tuple(v / 1000.0 for v in finite_values((var.get() for var in self.b), 3))
            if a == b:
                raise ValueError("A and B must be distinct")
            frame = self.frame.get().strip()
            SensingPose(1, "A", "wire", frame, a, (0.0, 0.0, 0.0, 1.0))
            tilt, length = seam_tilt_deg(a, b)
            if frame == "World" and tilt > MAX_1G_SEAM_TILT_DEG:
                raise ValueError(
                    f"A→B is {tilt:.0f}° from horizontal ({length * 1000.0:.0f} mm); a 1G seam must be "
                    f"within {MAX_1G_SEAM_TILT_DEG:.0f}°. Re-teach I/J at both seam ends along the seam "
                    "at the same height (nothing sent)")
        except ValueError as error:
            self.show_error(str(error))
            return
        self.clear()
        args = dict(a=a, b=b, frame=frame, action_name=self.action.get().strip(),
                    status_service=self.status_service.get().strip(), log_topic=self.log_topic.get().strip(),
                    on_progress=self.status.set, on_result=self.show_result, on_error=self.show_error,
                    on_finished=lambda: self._set_busy(False))
        self._set_busy(True)
        self.status.set("Connecting to Hongik server; full scan includes robot motion...")
        threading.Thread(target=self.bridge.scan, kwargs=args, daemon=True).start()

    def cancel_scan(self):
        self.bridge.cancel()
        self.status.set("Cancel requested; waiting for the sensing server to stop and return its result.")

    def load_file(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(title="Hongik PrecisionSensing result", filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            frame = self.frame.get().strip()
            self.clear()
            self._worker(lambda: load_result(path, frame), self.show_result)

    def show_error(self, text):
        self.status.set(f"ERROR · {text}")

    def show_result(self, inspection):
        self.inspection = inspection
        self.tree.delete(*self.tree.get_children())
        for i, pose in enumerate(inspection.poses):
            self.tree.insert("", tk.END, iid=str(i), values=(pose.pass_number, pose.kind.upper(), pose.point, pose.frame,
                             *(f"{v * 1000.0:.6f}" for v in pose.position), *(f"{v:.8f}" for v in pose.quaternion)))
        self.status.set(f"{inspection.source} · {inspection.document.get('error_message', '')}")
        section_fields = ("accuracy", "status", "done", "length_mm", "ends", "seam_dims", "design_deviations", "warnings")
        sections = {key: inspection.sections[key] for key in section_fields if key in inspection.sections}
        self.details.configure(state=tk.NORMAL)
        self.details.delete("1.0", tk.END)
        self.details.insert(tk.END, f"Source: {inspection.source}\nParent frame: {inspection.frame}\n"
                            f"Data file: {inspection.document.get('data_file_path', '')}\n\n" +
                            "\n".join(inspection.notes) + "\n\nSection measurements:\n" +
                            yaml.safe_dump(sections, allow_unicode=True, sort_keys=False))
        self.details.configure(state=tk.DISABLED)
        self.bridge.visualize(inspection)
        if inspection.poses:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.select_pose()

    def selected_pose(self):
        selected = self.tree.selection()
        if self.inspection is None or not selected:
            raise ValueError("Select a returned pose first")
        return self.inspection.poses[int(selected[0])]

    def select_pose(self, _event=None):
        try:
            pose = self.selected_pose()
        except ValueError:
            return
        self.selected_text.set(f"Pass {pose.pass_number} {pose.kind.upper()} {pose.point} · parent={pose.frame} · "
                               f"TF=hongik_test/selected_{pose.kind}")
        self.comparison.set("Enter reference values in this frame and pose convention, then Compare.")
        self.bridge.visualize(self.inspection, pose)

    def compare(self):
        try:
            pose = self.selected_pose()
            values = finite_values((var.get() for var in self.expected), 7)
            reference = SensingPose(pose.pass_number, "reference", pose.kind, pose.frame,
                                   tuple(v / 1000.0 for v in values[:3]), values[3:])
            position_tol, angle_tol = finite_values((self.position_tol.get(), self.angle_tol.get()), 2)
            if min(position_tol, angle_tol) < 0:
                raise ValueError("Tolerances must be non-negative")
            distance, angle = pose_difference(pose, reference)
            matches = distance <= position_tol and angle <= angle_tol
            self.comparison.set(f"{'WITHIN TOLERANCE' if matches else 'OUTSIDE TOLERANCE'} · "
                                f"position error={distance:.4f} mm, orientation error={angle:.4f} deg "
                                f"(limits {position_tol:g} mm / {angle_tol:g} deg)")
        except ValueError as error:
            self.comparison.set(f"ERROR · {error}")

    def clear(self):
        if self.busy:
            return
        self._generation += 1
        if hasattr(self, "kiro_pass"):
            self.kiro_pass = None
            self.pass_status.set("Load a pass YAML before original TCP motion")
        self.inspection = None
        self.tree.delete(*self.tree.get_children())
        self.bridge.clear()
        self.selected_text.set("No returned pose selected")
        self.comparison.set("No comparison performed")
        self.status.set("Result / RViz cleared")
        self.details.configure(state=tk.NORMAL)
        self.details.delete("1.0", tk.END)
        self.details.configure(state=tk.DISABLED)
