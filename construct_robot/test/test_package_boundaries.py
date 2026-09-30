"""Legacy imports and public ROS entry points survive the package split."""


def test_legacy_imports_share_implementations():
    # Top-level shims kept only for the frozen gui_qt package must keep
    # returning the canonical implementations.
    import construct_robot.weld_action_gui as legacy_gui
    from construct_robot.gui import weld_action_gui as gui
    from construct_robot.sequence_model import SequenceModel
    from construct_robot.core.sequence_model import SequenceModel as CoreModel
    import construct_robot.hicomm_welder as legacy_welder
    from construct_robot.io import hicomm_welder as welder

    assert legacy_gui is gui
    assert SequenceModel is CoreModel
    assert legacy_welder is welder


def test_weld_runtime_node_is_reexported_and_independent_of_gui():
    import ast
    from pathlib import Path

    import construct_robot.weld_action_gui as legacy_gui
    from construct_robot.gui import weld_action_gui as gui
    from construct_robot.nodes import weld_runtime_node

    assert gui.WeldGuiNode is weld_runtime_node.WeldGuiNode
    assert legacy_gui.WeldGuiNode is weld_runtime_node.WeldGuiNode

    tree = ast.parse(Path(weld_runtime_node.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name and ("gui" in name or name == "tkinter") for name in imported)


def test_core_and_io_do_not_depend_on_gui_or_nodes():
    import ast
    from pathlib import Path

    import construct_robot

    root = Path(construct_robot.__file__).parent
    for path in sorted((root / "core").glob("*.py")) + sorted((root / "io").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        } | {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        }
        for name in imported:
            assert not (name or "").startswith(
                ("construct_robot.gui", "construct_robot.nodes", "tkinter")
            ), f"{path.name} imports {name}"


def test_moved_gui_helpers_remain_importable_from_legacy_paths():
    import construct_robot.weld_action_gui as legacy_gui
    from construct_robot.core import (
        cartesian_path_common, keyboard_jog, seam_geometry, sequence_model,
        task_teaching_model,
    )
    from construct_robot.io import teaching_yaml, weld_logging

    moved = {
        cartesian_path_common: ("midpoint_pose", "pose_with_rpy_offset", "weld_weave_geometry",
                                "position_only_goal_constraints", "transform_xyz"),
        seam_geometry: ("CORNER_TOUCH_NAMES", "corner_seam_from_touches", "seam_yaw",
                        "wide_sensing_path_poses", "fixed_tilt_wait_reference_poses"),
        keyboard_jog: ("KEYBOARD_JOG_SELECTIONS", "keyboard_jog_velocity",
                       "keyboard_velocity_vector", "next_keyboard_speed"),
        sequence_model: ("update_weld_scenario_motion_values", "taught_wait_approach_steps"),
        task_teaching_model: ("TOUCH_GUARDED_TEACHING_POSES", "TCP_POSE_TEACHING_POSES",
                              "JOINT_RECALL_TEACHING_POSES", "SEAM_REFERENCE_TEACHING_POSES"),
        teaching_yaml: ("save_initial_state_yaml", "read_pass_teaching_reference",
                        "save_seam_touch_yaml", "load_seam_teaching_reference_yaml"),
        weld_logging: ("read_last_execution_settings", "read_teaching_and_touch_snapshot",
                       "read_weld_pass_reference"),
    }
    for module, names in moved.items():
        for name in names:
            assert getattr(legacy_gui, name) is getattr(module, name), name
