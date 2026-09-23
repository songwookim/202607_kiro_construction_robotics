"""Task/teaching library. Execution remains owned by the existing Sequence Builder."""
import copy
import math
import os
import re
import tempfile
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml

from .teaching_paths import teaching_config_dir


TASK_GROUPS = {
    "Right · Welding": "right_manipulator",
    "Right · Torch cleaner": "right_manipulator",
    "Left · Spray path": "left_manipulator",
}


def encode(value):
    """Serialize only data and explicitly supported ROS message types, never pickle."""
    if hasattr(value, "get_fields_and_field_types"):
        from rosidl_runtime_py.convert import message_to_ordereddict
        name = type(value).__name__
        if name not in ("Pose", "RobotTrajectory"):
            raise ValueError(f"Unsupported saved message: {name}")
        return {"ros_type": name, "fields": encode(dict(message_to_ordereddict(value)))}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite value in task")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported saved data: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if "ros_type" in value:
            from geometry_msgs.msg import Pose
            from moveit_msgs.msg import RobotTrajectory
            from rosidl_runtime_py.set_message import set_message_fields
            classes = {"Pose": Pose, "RobotTrajectory": RobotTrajectory}
            if value["ros_type"] not in classes:
                raise ValueError("Unsupported ROS message in task")
            message = classes[value["ros_type"]]()
            set_message_fields(message, value["fields"])
            return message
        return {k: decode(v) for k, v in value.items()}
    return value


def atomic_yaml(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            temporary = stream.name
            yaml.safe_dump(document, stream, sort_keys=False, allow_unicode=True)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def validate_task_group(steps, group):
    """Reject cross-arm motion and welding/output steps in the left spray library."""
    if not steps:
        raise ValueError("Task is empty")
    for step in steps:
        kind = step.get("type")
        if kind == "planned_trajectory":
            raise ValueError("Save taught targets, not cached RViz trajectories")
        if step.get("planning_group", group) != group or kind == "head_motion":
            raise ValueError("Task contains another arm/head; split it into separate tasks")
        if group == "left_manipulator" and kind not in ("named_pose", "motion", "sleep"):
            raise ValueError("Left spray task currently supports motion/wait only; no process output mapping")


class TaskTeachingPanel:
    def __init__(self, gui, parent, save_pose, load_pose):
        self.gui, self.save_pose, self.load_pose = gui, save_pose, load_pose
        self.busy = False
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
        self.order.insert(tk.END, name)

    def move(self, direction):
        selection = self.order.curselection()
        if selection and 0 <= selection[0] + direction < self.order.size():
            index = selection[0]
            value = self.order.get(index)
            self.order.delete(index)
            self.order.insert(index + direction, value)
            self.order.selection_set(index + direction)

    def remove(self):
        if self.order.curselection():
            self.order.delete(self.order.curselection()[0])

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
        speed = float(self.speed.get())
        if not math.isfinite(speed) or not 0 < speed <= 50:
            raise ValueError("TCP speed must be >0 and <=50 mm/s")
        names = self.order.get(0, tk.END)
        if not names:
            raise ValueError("Add taught poses to visit order")
        group = TASK_GROUPS[self.category.get()]
        stored = [self.load_pose(self.base() / "poses" / f"{self.safe_name(name)}.yaml") for name in names]
        if any(p[0] != group for p in stored):
            raise ValueError("Visit order contains another arm")
        _, joints, positions, tcp = stored[0]
        steps = [{"type": "named_pose", "pose_name": "task_start", "pose_label": names[0],
                  "planning_group": group, "joint_names": joints, "positions": positions,
                  "tcp_pose": tcp, "use_joint_planning": True, "velocity_scale": 0.05,
                  "tcp_speed_m_s": speed / 1000, "parallel_slot": 1, "duration": 0.0,
                  "touch_guard": False, "continue_after_touch": False}]
        if len(stored) > 1:
            steps.append({"type": "motion", "planning_group": group, "points": [p[3] for p in stored],
                          "velocity_scale": 0.05, "tcp_speed_m_s": speed / 1000,
                          "interpolation_step": 0.005, "path_kind": "taught continuous path",
                          "parallel_slot": 2, "duration": 0.0, "touch_guard": False,
                          "continue_after_touch": False})
        self.replace_builder(steps)

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
                          "visit_order": list(self.order.get(0, tk.END)),
                          "steps": encode(g.sequence_steps)})
        self.status.set(f"Saved latest Builder values: {path}")

    def load_task(self):
        path = self.base() / (self.safe_name(self.name.get().strip()) + ".yaml")
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
        if not isinstance(document, dict) or document.get("schema") != "robot_task_v1" or document.get("category") != self.category.get():
            raise ValueError("Task schema/category mismatch")
        steps = decode(document["steps"])
        if self.replace_builder(steps):
            self.order.delete(0, tk.END)
            for name in document.get("visit_order", []):
                self.order.insert(tk.END, self.safe_name(name))
