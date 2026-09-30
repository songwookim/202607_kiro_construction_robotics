"""Legacy imports and public ROS entry points survive the package split."""


def test_legacy_imports_share_implementations():
    import construct_robot.weld_action_gui as legacy_gui
    from construct_robot.gui import weld_action_gui as gui
    from construct_robot.cartesian_path_server import CartesianPathActionServer
    from construct_robot.nodes.cartesian_path_server import CartesianPathActionServer as NodeClass
    from construct_robot.sequence_model import SequenceModel
    from construct_robot.core.sequence_model import SequenceModel as CoreModel
    from construct_robot.seam_geometry import compute_safe_weld_approach
    from construct_robot.core.seam_geometry import compute_safe_weld_approach as CoreApproach
    import construct_robot.hicomm_welder as legacy_welder
    from construct_robot.io import hicomm_welder as welder

    assert legacy_gui is gui
    assert CartesianPathActionServer is NodeClass
    assert SequenceModel is CoreModel
    assert compute_safe_weld_approach is CoreApproach
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
