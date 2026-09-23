"""Task/teaching library. Execution remains owned by the existing Sequence Builder."""
import copy
import re
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml

from .teaching_paths import teaching_config_dir
from .task_teaching_model import (
    atomic_yaml,
    build_task_path_steps,
    decode,
    encode,
    TaskOrderState,
    validate_task_group,
    validated_task_speed,
)


TASK_GROUPS = {
    "Right · Welding": "right_manipulator",
    "Right · Torch cleaner": "right_manipulator",
    "Left · Spray path": "left_manipulator",
}


class TaskTeachingPanel:
    def __init__(self, gui, parent, save_pose, load_pose):
        self.gui, self.save_pose, self.load_pose = gui, save_pose, load_pose
        self.busy = False
        self.order_state = TaskOrderState()
        self.folder = tk.StringVar(value=str(teaching_config_dir() / "robot_tasks"))
        self.category = tk.StringVar(value="Left · Spray path")
        self.name = tk.StringVar(value="task_1")
        self.pose_name = tk.StringVar(value="point_1")
        self.speed = tk.StringVar(value="5.0")
        self.status = tk.StringVar(value="Library only: loading/saving never moves a robot")
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self.folder, width=48).pack(side=tk.LEFT)
        ttk.Button(row, text="Browse", command=lambda: self.call(self.browse)).pack(side=tk.LEFT)
        categories = ttk.Combobox(row, textvariable=self.category, values=list(TASK_GROUPS), state="readonly", width=23)
        categories.pack(side=tk.LEFT)
        categories.bind("<<ComboboxSelected>>", lambda _event: self.call(self.refresh))
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Task").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.name, width=18).pack(side=tk.LEFT)
        for label, callback in (("Save Builder → YAML", self.save_task),
                                ("Load YAML → Builder", self.load_task),
                                ("Import cleaner order", self.import_cleaner)):
            ttk.Button(row, text=label, command=lambda c=callback: self.call(c)).pack(side=tk.LEFT)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        self.poses = ttk.Combobox(row, textvariable=self.pose_name, width=22)
        self.poses.pack(side=tk.LEFT)
        ttk.Button(row, text="Save current → selected pose", command=lambda: self.call(self.capture)).pack(side=tk.LEFT)
        ttk.Button(row, text="Add pose to visit order", command=lambda: self.call(self.add_pose)).pack(side=tk.LEFT)
        ttk.Label(row, text="TCP mm/s").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.speed, width=6).pack(side=tk.LEFT)
        self.order = tk.Listbox(parent, height=5, exportselection=False)
        self.order.pack(fill=tk.X)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        for label, callback in (("↑", lambda: self.move(-1)), ("↓", lambda: self.move(1)),
                                ("Remove", self.remove), ("Build continuous path", self.build_path),
                                ("Plan Builder", lambda: self.run_builder(False)),
                                ("Execute Builder…", lambda: self.run_builder(True))):
            ttk.Button(row, text=label, command=lambda c=callback: self.call(c)).pack(side=tk.LEFT)
        ttk.Button(row, text="STOP ALL", command=gui.emergency_stop_all).pack(side=tk.LEFT)
        ttk.Label(parent, textvariable=self.status, wraplength=900).pack(anchor=tk.W)
        ttk.Label(parent, text="Right weld: build in Sequence Builder, select Right · Welding here, then Plan/Execute Builder.\n"
                  "Right cleaner: use its own Plan/Execute panel. Left: first pose joint approach, then one continuous TCP path.\n"
                  "No automatic spray output. Separate arm tasks; execution is sequential. Inspect planned paths before execution.").pack(anchor=tk.W)
        self.refresh()

    def idle(self):
        g = self.gui
        if (self.busy or g.sequence_running or g.seam_auto_running or g.multi_pass_registration is not None
                or g.node.active_motion_goal is not None or g.keyboard_velocity_arm or g.keyboard_velocity_switching):
            raise ValueError("Finish motion/workflow and disable keyboard teaching first")

    def call(self, callback):
        try:
            self.idle()
            callback()
        except Exception as error:
            self.gui.error(f"Task library: {error}")

    def base(self):
        return Path(self.folder.get()).expanduser().resolve() / self.category.get().split(" · ")[0].lower() / {
            "Right · Welding": "welding", "Right · Torch cleaner": "cleaner", "Left · Spray path": "spray"
        }[self.category.get()]

    @staticmethod
    def safe_name(name):
        if not re.fullmatch(r"[\w-]+", name):
            raise ValueError("Name must contain only letters, digits, _ or -")
        return name

    def browse(self):
        folder = filedialog.askdirectory(parent=self.gui.root, initialdir=self.folder.get())
        if folder:
            self.folder.set(folder)
            self.refresh()

    def refresh(self):
        self.poses.configure(values=sorted(p.stem for p in (self.base() / "poses").glob("*.yaml")))
        self.order_state.replace(())
        self.order.delete(0, tk.END)
        self.status.set(f"{TASK_GROUPS[self.category.get()]} · {self.base()}")

    def capture(self):
        group = TASK_GROUPS[self.category.get()]
        path = self.base() / "poses" / (self.safe_name(self.pose_name.get().strip()) + ".yaml")
        if path.exists() and not messagebox.askyesno("Replace teaching", f"Overwrite {path}?", parent=self.gui.root):
            return
        self.busy = True
        self.status.set(f"Capturing stationary measured pose: {group}")
        def work():
            try:
                names, positions, tcp, provenance = self.gui.node.capture_measured_teaching_snapshot(group, "task_library")
                self.save_pose(path, group, names, positions, tcp, provenance)
                result = f"Saved {path}"
            except Exception as error:
                result = f"Capture failed: {error}"
            self.gui.post(self.captured, result)
        threading.Thread(target=work, daemon=True).start()

    def captured(self, result):
        self.busy = False
        self.poses.configure(values=sorted(p.stem for p in (self.base() / "poses").glob("*.yaml")))
        self.status.set(result)

    def add_pose(self):
        name = self.safe_name(self.pose_name.get().strip())
        group, *_ = self.load_pose(self.base() / "poses" / f"{name}.yaml")
        if group != TASK_GROUPS[self.category.get()]:
            raise ValueError("Pose belongs to another arm")
        self.order_state.add(name)
        self.order.insert(tk.END, name)

    def move(self, direction):
        selection = self.order.curselection()
        if selection and 0 <= selection[0] + direction < self.order.size():
            index = selection[0]
            value = self.order_state.names[index]
            self.order_state.move(index, direction)
            self.order.delete(index)
            self.order.insert(index + direction, value)
            self.order.selection_set(index + direction)

    def remove(self):
        if self.order.curselection():
            index = self.order.curselection()[0]
            self.order_state.remove(index)
            self.order.delete(index)

    def replace_builder(self, steps):
        validate_task_group(steps, TASK_GROUPS[self.category.get()])
        if self.gui.sequence_steps and not messagebox.askyesno("Replace Builder", "Replace current Builder rows? Unsaved edits will be lost.", parent=self.gui.root):
            return False
        self.gui.sequence_steps = copy.deepcopy(steps)
        self.gui.refresh_sequence_table(select_last=True)
        self.status.set(f"Loaded {len(steps)} Builder steps; no motion sent")
        return True

    def run_builder(self, execute):
        validate_task_group(self.gui.sequence_steps, TASK_GROUPS[self.category.get()])
        self.gui.run_sequence(True, execute)

    def import_cleaner(self):
        if self.category.get() != "Right · Torch cleaner":
            raise ValueError("Select Right · Torch cleaner first")
        cleaner = self.gui.torch_cleaner_panel
        cleaner.idle()
        steps = cleaner.build_sequence_steps()
        if self.replace_builder(steps):
            self.status.set("Cleaner imported to Sequence Builder; verify the plan before execution.")

    def build_path(self):
        speed = validated_task_speed(self.speed.get())
        names = (tuple(self.order_state.names) if hasattr(self, "order_state")
                 else self.order.get(0, tk.END))
        if not names:
            raise ValueError("Add taught poses to visit order")
        group = TASK_GROUPS[self.category.get()]
        stored = [self.load_pose(self.base() / "poses" / f"{self.safe_name(name)}.yaml") for name in names]
        self.replace_builder(build_task_path_steps(names, stored, group, speed))

    def save_task(self):
        g = self.gui
        selected = g._selected_sequence_index()
        if selected is not None:
            success, error = g._commit_selected_sequence_step_edits(selected)
            if not success:
                raise ValueError(error)
        group = TASK_GROUPS[self.category.get()]
        validate_task_group(g.sequence_steps, group)
        path = self.base() / (self.safe_name(self.name.get().strip()) + ".yaml")
        if path.exists() and not messagebox.askyesno("Replace task", f"Overwrite {path}?", parent=g.root):
            return
        atomic_yaml(path, {"schema": "robot_task_v1", "category": self.category.get(),
                          "visit_order": list(self.order_state.names),
                          "steps": encode(g.sequence_steps)})
        self.status.set(f"Saved latest Builder values: {path}")

    def load_task(self):
        path = self.base() / (self.safe_name(self.name.get().strip()) + ".yaml")
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
        if not isinstance(document, dict) or document.get("schema") != "robot_task_v1" or document.get("category") != self.category.get():
            raise ValueError("Task schema/category mismatch")
        steps = decode(document["steps"])
        if self.replace_builder(steps):
            names = [self.safe_name(name) for name in document.get("visit_order", [])]
            self.order_state.replace(names)
            self.order.delete(0, tk.END)
            for name in names:
                self.order.insert(tk.END, name)
