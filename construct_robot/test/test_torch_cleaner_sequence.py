from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from geometry_msgs.msg import Pose

from construct_robot.teaching_paths import teaching_config_dir
from construct_robot.torch_cleaner_panel import TorchCleanerPanel
from construct_robot.weld_action_gui import WeldActionGui, WeldGuiNode
from construct_robot.core.torch_cleaner_teaching import (
    build_cleaner_sequence_steps, save_cleaner_correction,
)


class Selection:
    def configure(self, **_kwargs):
        pass

    def current(self, index):
        self.index = index


def test_cleaner_yaml_builds_ordered_right_arm_pulses(tmp_path):
    names = [f"right_manipulator_joint{i}" for i in range(1, 7)]
    for name in ("start", "top", "inside", "end"):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump({
            "schema": "torch_cleaner_joints_v1",
            "planning_group": "right_manipulator",
            "joint_state": {"names": names, "positions_rad": [0.0] * 6},
        }))
    (tmp_path / "sequence.yaml").write_text(yaml.safe_dump({
        "schema": "torch_cleaner_sequence_v2",
        "positions": ["start", "top", "inside", "DO7:ON", "DO7:OFF",
                      "top", "DO6:2", "end"],
    }))
    panel = object.__new__(TorchCleanerPanel)
    panel.folder = SimpleNamespace(get=lambda: str(tmp_path))
    panel.selected = SimpleNamespace(get=lambda: "start", set=lambda _value: None)
    panel.positions = Selection()
    panel.order = SimpleNamespace(set=lambda value: setattr(panel, "order_text", value),
                                  get=lambda: panel.order_text)
    panel.gui = SimpleNamespace(velocity_percent=SimpleNamespace(get=lambda: 5.0))
    panel.load_pose = Mock()

    steps = panel.build_sequence_steps()

    assert [step["parallel_slot"] for step in steps] == list(range(1, 8))
    assert [step["type"] for step in steps] == [
        "named_pose", "named_pose", "named_pose", "digital_output",
        "named_pose", "digital_output", "named_pose",
    ]
    assert [step["port"] for step in steps if step["type"] == "digital_output"] == [7, 6]
    assert [step["duration"] for step in steps if step["type"] == "digital_output"] == [1.0, 2.0]
    assert steps[0]["use_joint_planning"] and steps[-1]["use_joint_planning"]
    assert not steps[1]["use_joint_planning"]
    assert steps[1]["resolve_tcp_from_joints"]
    assert all(step["velocity_scale"] == 0.05
               for step in steps if step["type"] == "named_pose")
    assert all(step.get("planning_group", "right_manipulator") == "right_manipulator" for step in steps)
    assert panel.position_names == ["start", "top", "inside", "end"]
    panel.load_pose.assert_not_called()
    panel.idle = Mock()
    panel.gui.run_sequence = Mock()
    panel.plan_or_execute(False)
    panel.gui.run_sequence.assert_called_once()
    selected_steps = panel.gui.run_sequence.call_args.kwargs["steps_override"]
    assert len(selected_steps) == 1
    assert selected_steps[0]["type"] == "named_pose"
    assert selected_steps[0]["pose_label"] == "Cleaner start"


def test_joint_only_cleaner_execution_conditions_record_missing_tcp():
    assert WeldActionGui._pose_execution_conditions(None) is None


def test_sequence_builder_refreshes_cleaner_rows_without_erasing_weld():
    gui = object.__new__(WeldActionGui)
    gui.sequence_running = False
    gui.sequence_steps = [{"type": "sleep", "seconds": 0.1, "parallel_slot": 1}]
    gui.torch_cleaner_panel = SimpleNamespace(
        build_sequence_steps=lambda: [{"type": "named_pose", "planning_group": "right_manipulator",
                                       "torch_clean_scenario": True, "parallel_slot": 1}],
        status=SimpleNamespace(set=Mock()), folder=SimpleNamespace(get=lambda: "cleaner"),
    )
    gui.refresh_sequence_table = Mock()
    gui.error = Mock()
    gui.log = Mock()
    assert gui.build_torch_clean_sequence() is True
    assert [step["parallel_slot"] for step in gui.sequence_steps] == [1, 2]
    assert gui.build_torch_clean_sequence() is True
    assert len(gui.sequence_steps) == 2
    gui.error.assert_not_called()


def test_cleaner_execute_uses_only_cleaner_steps():
    cleaner_step = {"type": "named_pose", "planning_group": "right_manipulator",
                    "pose_label": "Cleaner start"}
    panel = object.__new__(TorchCleanerPanel)
    panel.idle = Mock()
    panel.build_sequence_steps = Mock(return_value=[cleaner_step])
    panel.selected = SimpleNamespace(get=lambda: "start")
    panel.position_names = ["start"]
    panel.gui = SimpleNamespace(run_sequence=Mock(), error=Mock())
    panel.plan_or_execute(True)
    panel.gui.run_sequence.assert_called_once_with(
        True, True, steps_override=[cleaner_step]
    )


def test_cleaner_offset_applies_from_anchor_occurrence_only(tmp_path):
    names = [f"right_manipulator_joint{i}" for i in range(1, 7)]
    for name in ("top", "inside", "end"):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump({
            "schema": "torch_cleaner_joints_v1", "planning_group": "right_manipulator",
            "joint_state": {"names": names, "positions_rad": [0.0] * 6},
        }))
    tokens = ["top", "inside", "DO7:1", "top", "end"]
    save_cleaner_correction(tmp_path, tokens, "inside", (0.0, 0.0, 0.01))
    steps = build_cleaner_sequence_steps(tmp_path, tokens, 5.0, Mock())
    assert "cleaner_world_offset_m" not in steps[0]
    assert steps[1]["cleaner_world_offset_m"] == {"x": 0.0, "y": 0.0, "z": 0.01}
    assert steps[2]["type"] == "digital_output"
    assert steps[3]["cleaner_world_offset_m"]["z"] == 0.01
    assert steps[4]["cleaner_world_offset_m"]["z"] == 0.01
    (tmp_path / "end.yaml").write_text("changed teaching")
    with pytest.raises(ValueError, match="teaching changed"):
        build_cleaner_sequence_steps(tmp_path, tokens, 5.0, Mock())


def test_corrected_cleaner_pose_moves_tcp_target_and_preserves_original_step():
    original = Pose()
    original.position.x = 0.2
    original.position.z = 0.3
    original.orientation.w = 1.0
    current = Pose()
    current.orientation.w = 1.0
    node = SimpleNamespace(
        _fk_pose_for_joints=Mock(return_value=original),
        validate_named_pose_recall=Mock(),
        _cleaner_corrected_joint_target=Mock(return_value=(0.1,) * 6),
        _current_tcp_pose=Mock(return_value=current),
        run_sequence_cartesian_motion=Mock(return_value=(True, "planned")),
        ui=SimpleNamespace(post=Mock(), log=Mock()),
    )
    step = {
        "type": "named_pose", "planning_group": "right_manipulator",
        "pose_name": "weld_start", "pose_label": "Cleaner inside",
        "joint_names": tuple(f"right_manipulator_joint{i}" for i in range(1, 7)),
        "positions": (0.0,) * 6, "tcp_pose": None,
        "resolve_tcp_from_joints": True, "use_joint_planning": False,
        "velocity_scale": 0.05, "touch_guard": False,
        "cleaner_world_offset_m": {"x": 0.0, "y": 0.0, "z": 0.01},
    }
    assert WeldGuiNode.run_sequence_named_pose(node, step, False) == (True, "planned")
    corrected_step = node._cleaner_corrected_joint_target.call_args.args
    assert corrected_step[1].position.z == pytest.approx(0.31)
    assert corrected_step[1].orientation.w == 1.0
    assert original.position.z == 0.3
    assert step["positions"] == (0.0,) * 6


def test_delete_all_has_no_confirmation(monkeypatch):
    gui = object.__new__(WeldActionGui)
    gui.sequence_running = False
    gui.sequence_steps = [{"type": "sleep"}]
    gui.sequence_parallel_slot = SimpleNamespace(set=Mock())
    gui.sequence_status = SimpleNamespace(configure=Mock())
    gui.refresh_sequence_table = Mock()
    gui.log = Mock()
    monkeypatch.setattr("construct_robot.weld_action_gui.messagebox.askyesno",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prompt")))
    gui.delete_all_sequence_steps()
    assert gui.sequence_steps == []


def test_named_teaching_paths_share_description_config():
    gui = object.__new__(WeldActionGui)
    config = teaching_config_dir()
    assert gui._initial_state_yaml_path("right_manipulator", "weld_start").parent == config
    assert gui._seam_reference_yaml_path("right_manipulator").parent == config
    assert gui._seam_touch_yaml_path("right_manipulator").parent == config
