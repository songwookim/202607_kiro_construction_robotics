"""Operator-confirmed cleaner teaching and motion, using existing GUI motion APIs."""
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from construct_robot.application.weld_sequence_builder import build_cleaner_sequence_steps
from construct_robot.core.task_teaching_model import CleanerTeachingState
from construct_robot.io.teaching_yaml import (
    cleaner_pose_path,
    load_cleaner_order,
    teaching_config_dir,
)

# Torch Cleaner tab keyboard shortcuts: key -> cleaner teaching pose.
CLEANER_KEY_POSES = {
    "u": "start",
    "i": "clear1_top",
    "j": "clear1_inside",
    "o": "clear2_top",
    "k": "clear2_inside",
    "p": "clear3_top",
    "l": "clear3_inside",
}


class TorchCleanerPanel:
    def __init__(self, gui, parent, save_pose, load_pose):
        self.gui, self.save_pose, self.load_pose = gui, save_pose, load_pose
        self.busy = False
        self.folder = tk.StringVar(value=str(teaching_config_dir() / "torch_cleaner_teaching"))
        self.state = CleanerTeachingState(Path(self.folder.get()))
        self.selected = tk.StringVar(value="start")
        self.selected_label = tk.StringVar()
        self.position_names = []
        self.order = tk.StringVar(value="start, clear1_top, clear1_inside, WIRE_FORWARD:0.75, DO7:ON, DO7:OFF, clear1_top, clear2_top, clear2_inside, DO6:2, clear2_top, clear3_top, clear3_inside, DO5:1, clear3_top, start")
        self.status = tk.StringVar(value="Select a numbered teaching pose, or build the cleaner sequence")
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        ttk.Label(row, textvariable=self.folder).pack(side=tk.LEFT)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Teaching index").pack(side=tk.LEFT)
        self.positions = ttk.Combobox(row, textvariable=self.selected_label, state="readonly", width=30)
        self.positions.pack(side=tk.LEFT)
        self.positions.bind("<<ComboboxSelected>>", self.select_position)
        ttk.Button(row, text="Save current right-arm pose", command=self.capture).pack(side=tk.LEFT)
        ttk.Button(row, text="Build → Sequence Builder", command=self.send_to_sequence).pack(side=tk.LEFT)
        ttk.Button(row, text="Plan", command=lambda: self.plan_or_execute(False)).pack(side=tk.LEFT)
        ttk.Button(row, text="Execute", command=lambda: self.plan_or_execute(True)).pack(side=tk.LEFT)
        ttk.Label(parent, textvariable=self.status, wraplength=850).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text=("Keyboard Teaching saves the current right-arm pose: "
                  + " · ".join(f"{key.upper()}={name}" for key, name in CLEANER_KEY_POSES.items())
                  + ". The sequence returns to start at the end."),
            wraplength=850,
        ).pack(anchor=tk.W)
        ttk.Label(parent, text="Cleaner 1 = DO7 · Cleaner 2 = DO6 · Cleaner 3 = DO5. Right arm only; left arm/head are not commanded.").pack(anchor=tk.W)
        self.refresh_teaching_index()

    def select_position(self, _event=None):
        index = self.positions.current()
        if 0 <= index < len(self.position_names):
            self.selected.set(self.position_names[index])
            self.state.selected = self.position_names[index]

    def refresh_teaching_index(self):
        folder = Path(self.folder.get())
        selected = self.selected.get()
        ordered = []
        order_path = folder / "sequence.yaml"
        if order_path.is_file():
            tokens = load_cleaner_order(folder)
            self.order.set(", ".join(tokens))
            if hasattr(self, "state"):
                self.state.set_order(tokens)
            ordered = list(dict.fromkeys(token for token in tokens if ":" not in token))
        extra = sorted(path.stem for path in folder.glob("*.yaml")
                       if path.stem not in ordered and path.stem != "sequence")
        self.position_names = ordered + extra
        self.positions.configure(values=[f"{index:02d}. {name}" for index, name
                                         in enumerate(self.position_names, 1)])
        if self.position_names:
            selected = selected if selected in self.position_names else self.position_names[0]
            self.selected.set(selected)
            if hasattr(self, "state"):
                self.state.selected = selected
            self.positions.current(self.position_names.index(selected))

    def send_to_sequence(self):
        self.gui.build_torch_clean_sequence()

    def plan_or_execute(self, execute):
        try:
            self.idle()
            name = self.selected.get().strip()
            if name not in self.position_names:
                raise ValueError("Select a saved cleaner teaching pose first")
            steps = [next(step for step in self.build_sequence_steps()
                          if step.get("pose_label") == f"Cleaner {name}")]
        except Exception as error:
            self.gui.error(f"Cleaner: {error}")
            return
        self.gui.run_sequence(True, execute, steps_override=steps)

    def build_sequence_steps(self):
        """Read latest teaching YAML and delegate sequence generation."""
        self.refresh_teaching_index()
        order_path = Path(self.folder.get()) / "sequence.yaml"
        if not order_path.is_file():
            raise ValueError(f"Cleaner order is missing: {order_path}")
        tokens = [name.strip() for name in self.order.get().split(",") if name.strip()]
        if hasattr(self, "state"):
            self.state.set_order(tokens)
            tokens = self.state.tokens
        return build_cleaner_sequence_steps(
            self.folder.get(), tokens, self.gui.velocity_percent.get(), self.load_pose
        )

    def idle(self):
        g = self.gui
        if (self.busy or g.sequence_running or g.seam_auto_running
                or g.multi_pass_registration is not None
                or g.node.active_motion_goal is not None):
            raise ValueError("Wait for active motion/workflow to finish")

    def path(self, name):
        return cleaner_pose_path(self.folder.get(), name)

    def capture(self, name=None):
        """Save the measured right-arm pose as ``name`` (default: selected)."""
        try:
            self.idle()
            path = self.path(name or self.selected.get().strip())
            group = "right_manipulator"
            self.busy = True
            self.status.set("Capturing measured TCP and joint positions...")
            def work():
                try:
                    names, positions, tcp, provenance = self.gui.node.capture_measured_teaching_snapshot(group, "torch_cleaner")
                    self.save_pose(path, group, names, positions, tcp, provenance)
                    message = f"Saved {path}"
                except Exception as error:
                    message = f"Capture failed: {error}"
                self.gui.post(self.finished_capture, message)
            threading.Thread(target=work, daemon=True).start()
        except Exception as error:
            self.gui.error(f"Cleaner: {error}")

    def finished_capture(self, message):
        self.busy = False
        self.status.set(message)
        self.gui.keyboard_jog_status.set(message)
        (self.gui.error if message.startswith("Capture failed") else self.gui.log)(
            f"Cleaner · {message}")
        self.refresh_teaching_index()

    def outputs_off(self):
        for channel in (5, 6, 7):
            try:
                ok, message = self.gui.node.set_fastech_output_sync(channel, False)
                if not ok:
                    self.gui.post(self.gui.error, f"Cleaner DO{channel} OFF failed: {message}")
            except Exception as error:
                self.gui.post(self.gui.error, f"Cleaner DO{channel} OFF failed: {error}")

    def abort(self):
        # SequenceExecutor owns motion/output steps. Preserve the panel's
        # cleanup while a teaching capture is in progress.
        if not self.busy:
            return
        self.status.set("Stopped")
        threading.Thread(target=self.outputs_off, daemon=True).start()
