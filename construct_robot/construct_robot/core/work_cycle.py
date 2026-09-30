"""Validate the operator's combined-cycle joint teaching and stage settings."""
import math
import copy
from pathlib import Path

import yaml


def load_work_cycle(path):
    document = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
    if not isinstance(document, dict) or document.get("schema") != "construct_robot_combined_work_cycle_v1":
        raise ValueError("Unsupported combined work cycle YAML")
    for phase in ("initial", "designated"):
        poses = document.get(phase)
        if not isinstance(poses, dict):
            raise ValueError(f"{phase} must contain arm joint poses")
        for arm in ("left", "right"):
            pose = poses.get(arm, {})
            names = pose.get("joint_names", ())
            values = pose.get("positions_rad", ())
            expected = [f"{arm}_manipulator_joint{index}" for index in range(1, 7)]
            if (names != expected or len(values) != 6
                    or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                               and math.isfinite(value) for value in values)):
                raise ValueError(f"{phase}.{arm} must contain six finite ordered joint positions")
    head = document["initial"].get("head", {})
    if (head.get("joint_names") != ["robot_head_rev_joint1", "robot_head_rev_joint2"]
            or len(head.get("positions_rad", ())) != 2
            or not all(math.isfinite(float(value)) for value in head["positions_rad"])):
        raise ValueError("initial.head must contain the two head joints")
    sweep = document.get("head_sweep_joint1_deg", ())
    if sweep != [-15.0, 15.0, 0.0]:
        raise ValueError("Head sweep must be -15°, +15°, 0°")
    if document.get("pass_number") != 4 or document.get("fake_arc_required") is not True:
        raise ValueError("Combined cycle requires Pass 4 with fake ARC")
    weave = document.get("weave", {})
    if weave != {"enabled": True, "transverse_axis": "tool_y"}:
        raise ValueError("Combined cycle requires tool-Y weaving")
    spray = document.get("spray_motion", {})
    required = {
        "positive_x_mm": 650.0, "positive_x_velocity_scale": 0.30,
        "negative_x_mm": -650.0, "negative_x_velocity_scale": 0.40,
        "circle_axis": "x", "circle_radius_mm": 100.0,
        "circle_unique_points": 4, "circle_closed": True,
        "circle_face_center": False, "circle_velocity_scale": 0.30,
        "output": "none",
    }
    if spray != required:
        raise ValueError("Combined cycle spray motion must match its approved geometry and speeds")
    return document


def dual_arm_step(poses, label, slot, velocity_scale):
    """One dual_arm MoveIt goal keeps both arms in one collision-checked plan."""
    names = poses["left"]["joint_names"] + poses["right"]["joint_names"]
    positions = poses["left"]["positions_rad"] + poses["right"]["positions_rad"]
    return {
        "type": "dual_arm_pose", "pose_label": label,
        "planning_group": "dual_arm", "joint_names": tuple(names),
        "positions": tuple(positions), "velocity_scale": float(velocity_scale),
        "parallel_slot": int(slot), "duration": 0.0,
    }


def assemble_work_cycle(config, weld_steps, cleaner_steps, repeats, travel_scale):
    """Build bounded, ordered slots; each simultaneous arm move is one dual-arm goal."""
    if not 1 <= repeats <= 20:
        raise ValueError("Repeat count must be 1..20")
    if not weld_steps or not cleaner_steps:
        raise ValueError("Weld and cleaner stages must both be present")
    result = []
    slot = 1
    left = config["designated"]["left"]

    def add(step, cycle):
        nonlocal slot
        row = copy.deepcopy(step)
        row["parallel_slot"] = slot
        row["work_cycle_id"] = "combined_pass4_spray_v1"
        row["work_cycle_number"] = cycle
        row["fake_arc_required"] = True
        result.append(row)
        slot += 1

    def arm_pose(poses, arm, label):
        pose = poses[arm]
        return {
            "type": "named_pose", "pose_name": "work_cycle_joint",
            "pose_label": label, "planning_group": f"{arm}_manipulator",
            "joint_names": tuple(pose["joint_names"]),
            "positions": tuple(pose["positions_rad"]),
            "tcp_pose": None, "resolve_tcp_from_joints": True,
            "use_joint_planning": True, "touch_guard": False,
            "velocity_scale": travel_scale, "duration": 0.0,
        }

    spray = config["spray_motion"]
    for cycle in range(1, repeats + 1):
        add(dual_arm_step(config["initial"], "Both arms initial", slot, travel_scale), cycle)
        add({
            "type": "head_motion",
            "joint1_rad": config["initial"]["head"]["positions_rad"][0],
            "joint2_rad": config["initial"]["head"]["positions_rad"][1],
            "pose_label": "Head initial", "duration": 2.0,
        }, cycle)
        for angle in config["head_sweep_joint1_deg"]:
            add({
                "type": "head_motion", "joint1_rad": math.radians(angle),
                "joint2_rad": config["initial"]["head"]["positions_rad"][1],
                "pose_label": f"Head J1 {angle:+g} deg", "duration": 2.0,
            }, cycle)
        for template in (weld_steps, cleaner_steps):
            old_slots = {}
            for index, step in enumerate(template):
                old_slot = step.get("parallel_slot", index + 1)
                if old_slot not in old_slots:
                    old_slots[old_slot] = slot
                    slot += 1
                row = copy.deepcopy(step)
                row["parallel_slot"] = old_slots[old_slot]
                row["work_cycle_id"] = "combined_pass4_spray_v1"
                row["work_cycle_number"] = cycle
                row["fake_arc_required"] = True
                if row.get("weld_scenario_id"):
                    row["weld_scenario_id"] = f"cycle_{cycle}_{row['weld_scenario_id']}"
                result.append(row)
        add(arm_pose(config["initial"], "right", "Right arm initial"), cycle)
        add(dual_arm_step(config["designated"], "Both arms designated", slot, travel_scale), cycle)
        for distance_key, scale_key, label in (
            ("positive_x_mm", "positive_x_velocity_scale", "Spray World +X"),
            ("negative_x_mm", "negative_x_velocity_scale", "Spray World -X"),
        ):
            add({
                "type": "spray_motion", "spray_kind": "line", "pose_label": label,
                "planning_group": "left_manipulator", "joint_names": tuple(left["joint_names"]),
                "positions": tuple(left["positions_rad"]),
                "distance_m": spray[distance_key] * 0.001,
                "velocity_scale": spray[scale_key], "path_kind": "linear",
                "touch_guard": False, "duration": 0.0,
            }, cycle)
            add(arm_pose(config["designated"], "left", "Left arm designated return"), cycle)
        add({
            "type": "spray_motion", "spray_kind": "circle", "pose_label": "Spray X-axis circle",
            "planning_group": "left_manipulator", "joint_names": tuple(left["joint_names"]),
            "positions": tuple(left["positions_rad"]),
            "radius_m": spray["circle_radius_mm"] * 0.001,
            "unique_points": spray["circle_unique_points"], "closed": spray["circle_closed"],
            "face_center": spray["circle_face_center"],
            "velocity_scale": spray["circle_velocity_scale"], "path_kind": "circle",
            "touch_guard": False, "duration": 0.0,
        }, cycle)
        add(dual_arm_step(config["initial"], "Both arms initial return", slot, travel_scale), cycle)
    if slot > 1000:
        raise ValueError("Combined work cycle exceeds 999 sequence slots")
    return result
