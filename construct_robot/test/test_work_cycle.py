from pathlib import Path

from construct_robot.core.work_cycle import assemble_work_cycle, load_work_cycle


CONFIG = Path(__file__).resolve().parents[2] / "construct_description/config/combined_work_cycle.yaml"


def test_work_cycle_joint_mapping_and_settings():
    config = load_work_cycle(CONFIG)
    assert config["initial"]["left"]["positions_rad"][0] == 2.4356046275763275
    assert config["initial"]["right"]["positions_rad"][0] == 5.1492129129205
    assert config["designated"]["left"]["positions_rad"][0] == 1.5680966803978833
    assert config["designated"]["right"]["positions_rad"][0] == 4.112291168789974
    assert config["weave"] == {"enabled": True, "transverse_axis": "tool_y"}


def test_assembly_order_repeat_and_bounded_fake_arc():
    config = load_work_cycle(CONFIG)
    weld = [
        {"type": "digital_weld", "command": "on", "parallel_slot": 2,
         "weld_scenario_id": "pass4", "weld_scenario_stage": "arc_on"},
        {"type": "motion", "parallel_slot": 2, "planning_group": "right_manipulator",
         "weld_scenario_id": "pass4", "weld_scenario_stage": "weld_motion"},
        {"type": "digital_weld", "command": "off", "parallel_slot": 3,
         "weld_scenario_id": "pass4", "weld_scenario_stage": "arc_off"},
    ]
    cleaner = [{"type": "digital_output", "parallel_slot": 1, "port": 7}]
    rows = assemble_work_cycle(config, weld, cleaner, 2, 0.2)
    assert all(row["fake_arc_required"] for row in rows)
    assert [row["joint1_rad"] for row in rows if row.get("pose_label", "").startswith("Head J1")] == [
        -0.2617993877991494, 0.2617993877991494, 0.0,
        -0.2617993877991494, 0.2617993877991494, 0.0,
    ]
    assert len([row for row in rows if row.get("pose_label") == "Head initial"]) == 2
    assert rows[0]["type"] == "dual_arm_pose"
    assert rows[-1]["type"] == "dual_arm_pose"
    assert rows[-1]["pose_label"] == "Both arms initial return"
    assert len([row for row in rows if row["type"] == "dual_arm_pose"]) == 6
    assert [row["distance_m"] for row in rows if row.get("spray_kind") == "line"] == [
        0.65, -0.65, 0.65, -0.65,
    ]
    assert [row["velocity_scale"] for row in rows if row.get("spray_kind") == "line"] == [
        0.3, 0.4, 0.3, 0.4,
    ]
    assert all(row["unique_points"] == 4 and row["closed"] and not row["face_center"]
               for row in rows if row.get("spray_kind") == "circle")
    arc_on = [row for row in rows if row.get("weld_scenario_stage") == "arc_on"]
    motion = [row for row in rows if row.get("weld_scenario_stage") == "weld_motion"]
    assert [row["parallel_slot"] for row in arc_on] == [row["parallel_slot"] for row in motion]
    assert arc_on[0]["weld_scenario_id"] != arc_on[1]["weld_scenario_id"]
