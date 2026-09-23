"""Operator supplied right-arm joint teaching, ordered joint1..joint6, radians."""
from dataclasses import dataclass, field
import math
from pathlib import Path

import yaml


@dataclass
class CleanerTeachingState:
    """Cleaner teaching folder, ordered tokens, and selected pose."""

    folder: Path
    tokens: list = field(default_factory=list)
    selected: str = "start"

    def set_order(self, tokens):
        self.tokens = list(tokens)


def cleaner_pose_path(folder, name):
    if (not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
            or name == "sequence"):
        raise ValueError("Use a position name containing letters, digits, _ or -")
    return Path(folder).expanduser().resolve() / f"{name}.yaml"


def cleaner_output_step(name):
    channel, value = name.split(":", 1)
    if channel not in ("DO5", "DO6", "DO7"):
        raise ValueError("Cleaner output must be DO5/6/7")
    if value not in ("ON", "OFF"):
        duration = float(value)
        if not math.isfinite(duration) or not 0 < duration <= 30:
            raise ValueError("Pulse duration must be 0..30 seconds")
    return int(channel[2:]), value


def build_cleaner_sequence_steps(folder, tokens, velocity_percent, load_pose):
    """Build existing cleaner rows from YAML without Tk or robot commands."""
    tokens = [name.strip() for name in tokens if name.strip()]
    if not tokens:
        raise ValueError("Cleaner sequence is empty")
    pulse_reference = {}
    for token in tokens:
        if ":" in token:
            port, value = cleaner_output_step(token)
            if value not in ("ON", "OFF"):
                pulse_reference[port] = float(value)
    steps = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        common = {"parallel_slot": len(steps) + 1, "duration": 0.0,
                  "torch_clean_scenario": True}
        if ":" in token:
            port, value = cleaner_output_step(token)
            if value == "ON":
                if index + 1 >= len(tokens) or tokens[index + 1] != f"DO{port}:OFF":
                    raise ValueError(f"DO{port}:ON needs an adjacent DO{port}:OFF for automatic execution")
                duration = pulse_reference.get(port, 1.0)
                index += 2
            else:
                duration = float(value) if value != "OFF" else 0.0
                index += 1
            steps.append(dict(common, type="digital_output", port=port,
                              value=value != "OFF", io_backend="fastech_ethernet",
                              task_cleaner_output=True, duration=duration))
            continue
        path = cleaner_pose_path(folder, token)
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
        if not isinstance(document, dict):
            raise ValueError(f"Invalid cleaner pose: {path}")
        if document.get("schema") == "torch_cleaner_joints_v1":
            group = document.get("planning_group")
            joint_state = document.get("joint_state", {})
            names = tuple(joint_state.get("names", ()))
            positions = tuple(float(value) for value in joint_state.get("positions_rad", ()))
            tcp = None
        else:
            group, names, positions, tcp = load_pose(path)
        expected = {f"right_manipulator_joint{i}" for i in range(1, 7)}
        if (group != "right_manipulator" or len(names) != 6 or set(names) != expected
                or len(positions) != 6 or not all(math.isfinite(value) for value in positions)):
            raise ValueError(f"Cleaner pose must contain six right-arm joints: {path}")
        joint_approach = token in ("start", "end")
        steps.append(dict(common, type="named_pose", pose_name=(
            "cleaner_joint" if joint_approach else "weld_start"),
            pose_label=f"Cleaner {token}", planning_group=group,
            joint_names=names, positions=positions, tcp_pose=tcp,
            resolve_tcp_from_joints=tcp is None, use_joint_planning=joint_approach,
            velocity_scale=max(0.01, min(1.0, float(velocity_percent) / 100.0)),
            touch_guard=False, continue_after_touch=False))
        index += 1
    return steps


JOINT_POSITIONS = {
    "start": [5.107377847988664, 0.735175472612407, 1.8645081076203565, 4.980960786835911, 0.19203616863457204, 0.02270595332882586],
    "cleaner1_top": [5.364418166919842, 1.096667166604392, 1.0770633715042095, 5.737482827165848, 0.6060325329259031, -0.5748441202419305],
    "cleaner1_inside": [5.3644144384943155, 1.1386124198263305, 1.0667621312497286, 5.712574814122555, 0.5791613041060308, -0.5448973065476667],
    "cleaner2_top": [5.4174656726731305, 1.0220944616034564, 1.2518536246439715, 5.628976589608313, 0.5469362892572586, -0.40710861411930227],
    "cleaner2_entry": [5.417461944247604, 1.0959381262559826, 1.2322194695073623, 5.56996040721994, 0.5048960943424585, -0.33890722242075866],
    "cleaner1_return_top": [5.486312647909423, 0.9265130108692672, 1.4417975972814205, 5.504173404076148, 0.5086893012620857, -0.20843153705623818],
    "cleaner1_return_inside": [5.486320104760475, 0.9867597068393122, 1.428979004006542, 5.440091889294443, 0.4760542261844685, -0.1357610947324326],
    "end": [5.107377847988664, 0.735175472612407, 1.8645081076203565, 4.980960786835911, 0.19203616863457204, 0.02270595332882586],
}
