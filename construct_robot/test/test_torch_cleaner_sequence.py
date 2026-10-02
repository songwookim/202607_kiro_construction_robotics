from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from geometry_msgs.msg import Pose

from construct_robot.io.teaching_yaml import (
    teaching_config_dir, load_cleaner_order, load_cleaner_pose,
)
from construct_robot.gui.torch_cleaner_panel import TorchCleanerPanel
from construct_robot.gui.weld_action_gui import WeldActionGui, WeldGuiNode
from construct_robot.application.weld_sequence_builder import build_cleaner_sequence_steps


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


def test_delete_all_has_no_confirmation(monkeypatch):
    gui = object.__new__(WeldActionGui)
    gui.sequence_running = False
    gui.sequence_steps = [{"type": "sleep"}]
    gui.sequence_parallel_slot = SimpleNamespace(set=Mock())
    gui.sequence_status = SimpleNamespace(configure=Mock())
    gui.refresh_sequence_table = Mock()
    gui.log = Mock()
    monkeypatch.setattr("construct_robot.gui.weld_action_gui.messagebox.askyesno",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prompt")))
    gui.delete_all_sequence_steps()
    assert gui.sequence_steps == []


def test_named_teaching_paths_share_description_config():
    gui = object.__new__(WeldActionGui)
    config = teaching_config_dir()
    assert gui._initial_state_yaml_path("right_manipulator", "weld_start").parent == config
    assert gui._seam_reference_yaml_path("right_manipulator").parent == config
    assert gui._seam_touch_yaml_path("right_manipulator").parent == config


def test_legacy_cleaner_order_keeps_do5_do7_migration(tmp_path):
    (tmp_path / "sequence.yaml").write_text(yaml.safe_dump({
        "schema": "torch_cleaner_sequence_v1",
        "positions": ["start", "DO5:ON", "DO5:OFF", "DO6:2", "DO7:1", "end"],
    }))
    assert load_cleaner_order(tmp_path) == [
        "start", "DO7:ON", "DO7:OFF", "DO6:2", "DO5:1", "end",
    ]


@pytest.mark.parametrize("tokens", [["DO7:ON"], ["DO7:ON", "DO6:OFF"], ["DO0:1"]])
def test_cleaner_builder_rejects_unpaired_on_and_touch_output(tmp_path, tokens):
    with pytest.raises(ValueError):
        build_cleaner_sequence_steps(tmp_path, tokens, 5.0, Mock())


def test_cleaner_loader_preserves_named_joint_order_and_current_tcp(tmp_path):
    from construct_robot.io.teaching_yaml import save_initial_state_yaml

    names = tuple(f"right_manipulator_joint{i}" for i in range(6, 0, -1))
    positions = tuple(i / 10.0 for i in range(6))
    tcp = Pose()
    tcp.orientation.w = 1.0
    tcp.position.z = 0.3
    path = tmp_path / "inside.yaml"
    save_initial_state_yaml(path, "right_manipulator", names, positions, tcp)
    assert load_cleaner_pose(path) == ("right_manipulator", names, positions, tcp)


@pytest.mark.parametrize("bad_field", ["arm", "names", "positions"])
def test_cleaner_loader_rejects_invalid_joints(tmp_path, bad_field):
    document = {
        "schema": "torch_cleaner_joints_v1", "planning_group": "right_manipulator",
        "joint_state": {"names": [f"right_manipulator_joint{i}" for i in range(1, 7)],
                        "positions_rad": [0.0] * 6},
    }
    if bad_field == "arm":
        document["planning_group"] = "left_manipulator"
    elif bad_field == "names":
        document["joint_state"]["names"][0] = "right_manipulator_joint2"
    else:
        document["joint_state"]["positions_rad"][0] = float("nan")
    path = tmp_path / "inside.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="six right-arm joints"):
        load_cleaner_pose(path)


@pytest.mark.parametrize("busy", [False, True])
def test_cleaner_abort_preserves_capture_cleanup_and_sequence_ownership(monkeypatch, busy):
    panel = object.__new__(TorchCleanerPanel)
    panel.busy = busy
    panel.status = SimpleNamespace(set=Mock())
    panel.gui = SimpleNamespace(sequence_running=True)
    panel.outputs_off = Mock()
    worker = Mock()
    monkeypatch.setattr("construct_robot.gui.torch_cleaner_panel.threading.Thread", worker)

    panel.abort()

    assert panel.gui.sequence_running is True  # Only common executor may finish it.
    assert panel.busy is busy  # The capture/correction completion owns this state.
    if busy:
        worker.assert_called_once_with(target=panel.outputs_off, daemon=True)
        worker.return_value.start.assert_called_once_with()
    else:
        worker.assert_not_called()


def test_cleaner_outputs_off_attempts_every_owned_channel_after_failure():
    panel = object.__new__(TorchCleanerPanel)
    output = Mock(side_effect=[RuntimeError("disconnected"), (False, "timeout"), (True, "OFF")])
    panel.gui = SimpleNamespace(
        node=SimpleNamespace(set_fastech_output_sync=output), post=Mock(), error=Mock(),
    )
    panel.outputs_off()
    assert [call.args for call in output.call_args_list] == [(5, False), (6, False), (7, False)]
    assert panel.gui.post.call_count == 2


@pytest.mark.parametrize("blocker", ["busy", "sequence", "seam", "multipass", "motion"])
def test_cleaner_teaching_keeps_workflow_interlocks(blocker):
    panel = object.__new__(TorchCleanerPanel)
    panel.busy = blocker == "busy"
    panel.gui = SimpleNamespace(
        sequence_running=blocker == "sequence", seam_auto_running=blocker == "seam",
        multi_pass_registration={} if blocker == "multipass" else None,
        node=SimpleNamespace(active_motion_goal=object() if blocker == "motion" else None),
    )
    with pytest.raises(ValueError, match="Wait for active motion"):
        panel.idle()


def test_cleaner_wire_feed_follows_inside_pose(tmp_path):
    names = [f"right_manipulator_joint{i}" for i in range(1, 7)]
    (tmp_path / "clear1_inside.yaml").write_text(yaml.safe_dump({
        "schema": "torch_cleaner_joints_v1",
        "planning_group": "right_manipulator",
        "joint_state": {"names": names, "positions_rad": [0.0] * 6},
    }))
    steps = build_cleaner_sequence_steps(
        tmp_path, ["clear1_inside", "WIRE_FORWARD:0.75", "DO7:1"], 5.0, Mock())
    assert [step["type"] for step in steps] == ["named_pose", "wire_feed", "digital_output"]
    assert steps[1]["duration"] == 0.75
    assert [step["parallel_slot"] for step in steps] == [1, 2, 3]


def test_saved_cleaner_sequence_uses_key_poses_returns_to_start_and_feeds_075s():
    from construct_robot.gui.torch_cleaner_panel import CLEANER_KEY_POSES
    folder = teaching_config_dir() / "torch_cleaner_teaching"
    tokens = load_cleaner_order(folder)
    poses = [token for token in tokens if ":" not in token]
    assert tokens[0] == tokens[-1] == "start"
    assert "end" not in tokens
    assert set(poses) <= set(CLEANER_KEY_POSES.values())
    assert [t for t in tokens if t.startswith("WIRE_FORWARD:")] == ["WIRE_FORWARD:0.75"]
    for name in set(poses):
        assert (folder / f"{name}.yaml").is_file(), name
    from construct_robot.io.teaching_yaml import load_initial_state_yaml
    steps = build_cleaner_sequence_steps(folder, tokens, 5.0, load_initial_state_yaml)
    assert [s["duration"] for s in steps if s["type"] == "wire_feed"] == [0.75]
    assert steps[0]["positions"] == steps[-1]["positions"]


def test_cleaner_key_map():
    from construct_robot.gui.torch_cleaner_panel import CLEANER_KEY_POSES
    assert CLEANER_KEY_POSES == {
        "u": "start", "i": "clear1_top", "j": "clear1_inside", "o": "clear2_top",
        "k": "clear2_inside", "p": "clear3_top", "l": "clear3_inside",
    }


def keyboard_gui(cleaner=True, arm="right"):
    gui = object.__new__(WeldActionGui)
    gui._cleaner_task_selected = lambda: cleaner
    gui.multi_pass_registration = None
    gui.keyboard_jog_enabled = SimpleNamespace(get=lambda: True)
    gui._keyboard_focus_allows_jog = lambda: True
    gui.keyboard_shortcut_release_ids = {}
    gui.keyboard_shortcut_active_keys = set()
    gui.keyboard_velocity_active_key = None
    gui.keyboard_velocity_arm = arm
    gui.keyboard_velocity_switching = False
    gui.sequence_running = False
    gui.node = SimpleNamespace(active_motion_goal=None)
    gui.keyboard_jog_status = SimpleNamespace(set=Mock())
    gui.torch_cleaner_panel = SimpleNamespace(capture=Mock())
    gui.log = Mock()
    gui.error = Mock()
    return gui


@pytest.mark.parametrize("key,name", [("u", "start"), ("j", "clear1_inside"), ("l", "clear3_inside")])
def test_cleaner_tab_keys_save_their_pose(key, name):
    gui = keyboard_gui()
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym=key)) == "break"
    gui.torch_cleaner_panel.capture.assert_called_once_with(name)


def test_cleaner_keys_need_right_arm_keyboard_teaching():
    gui = keyboard_gui(arm=None)
    gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="i"))
    gui.torch_cleaner_panel.capture.assert_not_called()
    gui.error.assert_called_once()


def test_u_is_ignored_outside_the_cleaner_tab():
    gui = keyboard_gui(cleaner=False)
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="u")) is None
    gui.torch_cleaner_panel.capture.assert_not_called()


def test_sequence_table_displays_all_cleaner_rows_after_wire_feed():
    gui = object.__new__(WeldActionGui)
    gui.sequence_table = Mock()
    gui.sequence_table.get_children.return_value = ()
    gui._selected_sequence_index = Mock(return_value=None)
    gui.load_selected_sequence_values = Mock()
    pose = {"type": "named_pose", "pose_label": "Cleaner inside",
            "planning_group": "right_manipulator", "velocity_scale": 0.05}
    gui.sequence_steps = [dict(pose) for _ in range(15)]
    gui.sequence_steps[3] = {"type": "wire_feed", "duration": 0.25, "parallel_slot": 4}
    gui.refresh_sequence_table(select_last=True)
    assert gui.sequence_table.insert.call_count == 15
    calls = gui.sequence_table.insert.call_args_list
    assert calls[3].kwargs["values"] == (
        4, "INCH FORWARD", "Hi-COMM timed wire feed · slot 4 · 0.25 s")
    assert calls[-1].kwargs["iid"] == "14"
    gui.sequence_table.selection_set.assert_called_once_with("14")
