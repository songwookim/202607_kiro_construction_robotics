"""Task-teaching serialization, validation, and path data without Tkinter."""
import math
import os
import re
import tempfile
from pathlib import Path

import yaml


TASK_GROUPS = {
    "Right · Welding": "right_manipulator",
    "Right · Torch cleaner": "right_manipulator",
    "Left · Spray path": "left_manipulator",
}

TEACHING_POSES = {
    "robot_start": "1 · Initial pose",
    "weld_wait": "2 · Weld wait pose",
    "weld_start_wait": "3 · Weld start wait pose",
    "weld_start": "4 · Reference TCP 1 / Weld start",
    "weld_goal_wait": "5 · Weld goal wait pose",
    "weld_end": "6 · Reference TCP 2 / Weld goal",
    "weld_finish": "7 · Weld end pose",
}


def task_base_path(folder, category):
    if category not in TASK_GROUPS:
        raise ValueError("Unknown task category")
    suffix = {"Right · Welding": "welding", "Right · Torch cleaner": "cleaner",
              "Left · Spray path": "spray"}[category]
    return Path(folder).expanduser().resolve() / category.split(" · ")[0].lower() / suffix


def safe_task_name(name):
    if not re.fullmatch(r"[\w-]+", name):
        raise ValueError("Name must contain only letters, digits, _ or -")
    return name


class TeachingState:
    """Named TCP/joint snapshots and provenance, independent of widgets."""

    def __init__(self, names):
        self.poses = {name: None for name in names}
        self.provenance = {}
        self.selected_name = next(iter(self.poses), None)

    def select(self, name):
        if name not in self.poses:
            raise ValueError(f"Unknown teaching pose: {name}")
        self.selected_name = name
        return name

    def store(self, name, snapshot, provenance=None):
        if name not in self.poses:
            raise ValueError(f"Unknown teaching pose: {name}")
        self.poses[name] = snapshot
        if provenance is None:
            self.provenance.pop(name, None)
        else:
            self.provenance[name] = provenance


class TaskOrderState:
    """Ordered taught-pose names used by Task Teaching, without a Listbox."""

    def __init__(self, names=()):
        self.names = list(names)

    def replace(self, names):
        self.names = list(names)

    def add(self, name):
        self.names.append(name)

    def move(self, index, direction):
        target = index + direction
        if not 0 <= target < len(self.names):
            return None
        self.names[index], self.names[target] = self.names[target], self.names[index]
        return target

    def remove(self, index):
        return self.names.pop(index)


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


def load_task_file(path, expected_category=None):
    """Read existing task YAML without widgets, motion, or output commands."""
    document = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
    if (not isinstance(document, dict) or document.get("schema") != "robot_task_v1"
            or document.get("category") not in TASK_GROUPS
            or (expected_category is not None and document["category"] != expected_category)):
        raise ValueError("Task schema/category mismatch")
    steps = decode(document["steps"])
    validate_task_group(steps, TASK_GROUPS[document["category"]])
    return document, steps

def validated_task_speed(speed):
    speed = float(speed)
    if not math.isfinite(speed) or not 0 < speed <= 50:
        raise ValueError("TCP speed must be >0 and <=50 mm/s")
    return speed


def build_task_path_steps(names, stored, group, speed):
    """Build the unchanged joint approach and continuous TCP path rows."""
    speed = validated_task_speed(speed)
    if not names:
        raise ValueError("Add taught poses to visit order")
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
    return steps
