"""Operator-confirmed cleaner teaching and motion, using existing GUI motion APIs."""
import threading
import math
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import yaml


class TorchCleanerPanel:
    def __init__(self, gui, parent, save_pose, load_pose):
        self.gui, self.save_pose, self.load_pose = gui, save_pose, load_pose
        self.busy = False
        self.active = False
        self.steps = []
        self.index = 0
        self.folder = tk.StringVar(value=str(Path.cwd() / "torch_cleaner_teaching"))
        self.selected = tk.StringVar(value="point_1")
        self.order = tk.StringVar(value="start, cleaner1_top, cleaner1_inside, DO7:ON, DO7:OFF, cleaner1_top, cleaner2_top, cleaner2_entry, DO6:2, cleaner2_top, cleaner1_return_top, cleaner1_return_inside, DO7:1, cleaner1_return_top, end")
        self.status = tk.StringVar(value="Select a position name and save current position; edit visit order below")
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self.folder, width=55).pack(side=tk.LEFT)
        ttk.Button(row, text="Browse folder", command=self.browse).pack(side=tk.LEFT)
        ttk.Button(row, text="Create supplied teaching YAMLs", command=self.seed).pack(side=tk.LEFT)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        self.positions = ttk.Combobox(row, textvariable=self.selected, values=[f"point_{i}" for i in range(1, 9)], width=20)
        self.positions.pack(side=tk.LEFT)
        ttk.Button(row, text="Save current position", command=self.capture).pack(side=tk.LEFT)
        ttk.Label(parent, text="Order: position names / DO7:ON / DO7:OFF / DO6:2 (2-second pulse). Each step requires Next.").pack(anchor=tk.W)
        ttk.Entry(parent, textvariable=self.order, width=80).pack(fill=tk.X)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Save order YAML", command=self.save_order).pack(side=tk.LEFT)
        ttk.Button(row, text="Prepare sequence", command=self.prepare).pack(side=tk.LEFT)
        ttk.Button(row, text="Confirm / Next motion", command=self.next).pack(side=tk.LEFT)
        ttk.Button(row, text="STOP", command=self.stop).pack(side=tk.LEFT)
        ttk.Label(parent, textvariable=self.status, wraplength=850).pack(anchor=tk.W)
        ttk.Label(parent, text="Cleaner 1 = DO7 · Cleaner 2 = DO6 · Cleaner 3 = DO5. Right arm only; left arm/head are not commanded.").pack(anchor=tk.W)

    def seed(self):
        try:
            self.idle()
            from .torch_cleaner_teaching import JOINT_POSITIONS
            for name, positions in JOINT_POSITIONS.items():
                path = self.path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    continue
                with path.open("x", encoding="utf-8") as stream:
                    yaml.safe_dump({"schema": "torch_cleaner_joints_v1", "planning_group": "right_manipulator",
                                    "joint_state": {"names": [f"right_manipulator_joint{i}" for i in range(1, 7)],
                                                    "positions_rad": positions}}, stream)
            self.positions.configure(values=list(JOINT_POSITIONS))
            self.selected.set("start")
            self.save_order()
        except Exception as error:
            self.gui.error(f"Cleaner: {error}")

    def output_step(self, name):
        channel, value = name.split(":", 1)
        if channel not in ("DO5", "DO6", "DO7"):
            raise ValueError("Cleaner output must be DO5/6/7")
        if value not in ("ON", "OFF"):
            duration = float(value)
            if not math.isfinite(duration) or not 0 < duration <= 30:
                raise ValueError("Pulse duration must be 0..30 seconds")
        return int(channel[2:]), value

    def idle(self, next_step=False):
        g = self.gui
        if self.busy or (self.active and not next_step) or (g.sequence_running and not (self.active and next_step)) or g.seam_auto_running or g.multi_pass_registration is not None or g.node.active_motion_goal is not None:
            raise ValueError("Wait for active motion/workflow to finish")

    def browse(self):
        try:
            self.idle()
            folder = filedialog.askdirectory(parent=self.gui.root, initialdir=self.folder.get())
            if not folder:
                return
            self.folder.set(folder)
            self.steps = []
            names = [p.stem for p in Path(folder).glob("*.yaml") if p.name != "sequence.yaml"]
            self.positions.configure(values=sorted(names))
            path = Path(folder) / "sequence.yaml"
            if path.exists():
                data = yaml.safe_load(path.read_text())
                if not isinstance(data, dict) or not isinstance(data.get("positions"), list):
                    raise ValueError("Invalid cleaner sequence YAML")
                positions = data["positions"]
                if data.get("schema") == "torch_cleaner_sequence_v1":
                    positions = [
                        ("DO7:" + value[4:] if value.startswith("DO5:") else
                         "DO5:" + value[4:] if value.startswith("DO7:") else value)
                        for value in positions
                    ]
                    self.gui.log("Cleaner sequence v1 loaded: swapped DO5/DO7; Save order YAML to persist v2")
                self.order.set(", ".join(positions))
            self.status.set(f"Loaded folder: {folder}")
        except Exception as error:
            self.gui.error(f"Cleaner: {error}")

    def path(self, name):
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) or name == "sequence":
            raise ValueError("Use a position name containing letters, digits, _ or -")
        return Path(self.folder.get()).expanduser().resolve() / f"{name}.yaml"

    def capture(self):
        try:
            self.idle()
            path = self.path(self.selected.get().strip())
            group = self.gui.planning_group.get()
            if group not in ("left_manipulator", "right_manipulator"):
                raise ValueError("Select one robot arm")
            self.busy = True
            self.steps = []
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

    def save_order(self):
        try:
            self.idle()
            import os
            import tempfile
            names = [name.strip() for name in self.order.get().split(",") if name.strip()]
            if not names:
                raise ValueError("Enter position names")
            for name in names:
                self.output_step(name) if ":" in name else self.path(name)
            folder = self.path("start").parent
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=folder, delete=False) as stream:
                temporary = stream.name
                yaml.safe_dump({"schema": "torch_cleaner_sequence_v2", "positions": names}, stream)
            os.replace(temporary, folder / "sequence.yaml")
            self.status.set(f"Saved order: {folder / 'sequence.yaml'}")
        except Exception as error:
            self.gui.error(f"Cleaner: {error}")

    def prepare(self):
        try:
            self.idle()
            self.steps = []
            names = [name.strip() for name in self.order.get().split(",") if name.strip()]
            steps = []
            for name in names:
                if ":" in name:
                    self.output_step(name)
                    steps.append((name, None))
                    continue
                path = self.path(name)
                data = yaml.safe_load(path.read_text())
                if data.get("schema") == "torch_cleaner_joints_v1":
                    joints = data["joint_state"]
                    positions = tuple(float(v) for v in joints["positions_rad"])
                    expected = [f"right_manipulator_joint{i}" for i in range(1, 7)]
                    if joints["names"] != expected or len(positions) != 6 or not all(map(math.isfinite, positions)):
                        raise ValueError(f"Invalid joints: {name}")
                    stored = ("right_manipulator", tuple(expected), positions, None)
                else:
                    stored = self.load_pose(path)
                steps.append((name, stored))
            if not steps:
                raise ValueError("Enter position names")
            if len({pose[0] for _, pose in steps if pose is not None}) != 1:
                raise ValueError("All cleaner positions must belong to the same arm")
            self.steps, self.index = steps, 0
            self.status.set(f"Ready: 1/{len(steps)} → {steps[0][0]}; confirm Next")
        except Exception as error:
            self.gui.error(f"Cleaner: {error}")

    def next(self):
        try:
            self.idle(next_step=True)
            g = self.gui
            if not self.steps or self.index >= len(self.steps):
                raise ValueError("Prepare the sequence first")
            name, stored = self.steps[self.index]
            group, names, positions, tcp = stored or ("right_manipulator", (), (), None)
            if not g.execution_allowed or not g.robot_connected.get(group.removesuffix("_manipulator"), False):
                raise ValueError("Connect the robot and enable physical execution")
            if g.keyboard_velocity_arm is not None or g.keyboard_velocity_switching:
                raise ValueError("Disable keyboard teaching before moving")
            speed = float(g.velocity_percent.get()) / 100.0
            if not 0 < speed <= 1:
                raise ValueError("Invalid velocity scale")
            self.busy = True
            self.active = True
            g.sequence_running = True
            g.sequence_stop_requested = False
            self.status.set(f"Moving {self.index + 1}/{len(self.steps)} → {name}")
            step = dict(pose_name="weld_start", pose_label=f"Cleaner {name}", planning_group=group,
                        joint_names=names, positions=positions, tcp_pose=tcp,
                        velocity_scale=speed, touch_guard=False)
            def work():
                try:
                    if g.sequence_stop_requested:
                        raise RuntimeError("Cleaner stopped")
                    if stored is None:
                        channel, value = self.output_step(name)
                        success, message = g.node.set_fastech_output_sync(channel, value != "OFF")
                        if success and value not in ("ON", "OFF"):
                            try:
                                deadline = time.monotonic() + float(value)
                                while time.monotonic() < deadline and not g.sequence_stop_requested:
                                    time.sleep(0.02)
                            finally:
                                success, message = g.node.set_fastech_output_sync(channel, False)
                    else:
                        if step["tcp_pose"] is None:
                            step["tcp_pose"] = g.node._fk_pose_for_joints(group, names, positions)
                        if g.sequence_stop_requested:
                            raise RuntimeError("Cleaner stopped")
                        if name in ("start", "end"):
                            step["use_joint_planning"] = True
                        success, message = g.node.run_sequence_named_pose(step, True)
                except Exception as error:
                    success, message = False, str(error)
                if not success or g.sequence_stop_requested:
                    self.outputs_off()
                g.post(self.finished_motion, success, message)
            threading.Thread(target=work, daemon=True).start()
        except Exception as error:
            self.gui.error(f"Cleaner: {error}")

    def finished_motion(self, success, message):
        self.busy = False
        if not success or self.gui.sequence_stop_requested:
            self.active = False
            self.gui.sequence_running = False
            self.steps = []
            self.status.set(f"Stopped: {message}; prepare again")
            return
        self.index += 1
        if self.index == len(self.steps):
            self.active = False
            self.gui.sequence_running = False
            threading.Thread(target=self.outputs_off, daemon=True).start()
        self.status.set("Cleaner sequence complete" if self.index == len(self.steps) else
                        f"Arrived. Inspect, then confirm Next → {self.steps[self.index][0]} ({self.index + 1}/{len(self.steps)})")

    def stop(self):
        self.steps = []
        self.gui.emergency_stop_all()
        self.status.set("STOP requested; prepare again")

    def outputs_off(self):
        for channel in (5, 6, 7):
            try:
                ok, message = self.gui.node.set_fastech_output_sync(channel, False)
                if not ok:
                    self.gui.post(self.gui.error, f"Cleaner DO{channel} OFF failed: {message}")
            except Exception as error:
                self.gui.post(self.gui.error, f"Cleaner DO{channel} OFF failed: {error}")

    def abort(self):
        if not self.active and not self.busy:
            self.steps = []
            return
        self.active = False
        if not self.busy:
            self.gui.sequence_running = False
        self.steps = []
        self.status.set("Stopped; prepare sequence again")
        threading.Thread(target=self.outputs_off, daemon=True).start()
