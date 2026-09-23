import pytest
import yaml

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from construct_robot.gui_qt.main_window import SequenceMainWindow
from construct_robot.multipass import MultiPassState
from construct_robot.sequence_model import SequenceModel
from construct_robot.task_teaching_model import TEACHING_POSES, TeachingState
from construct_robot.torch_cleaner_teaching import CleanerTeachingState
from construct_robot.weld_config import WeldConfigurationState


def teaching_yaml(group="left_manipulator"):
    return {
        "format_version": 1, "planning_group": group,
        "joint_state": {"names": [f"{group}_joint{i}" for i in range(1, 7)],
                        "positions_rad": [0.0] * 6},
        "tcp_pose_world": {"position_m": {"x": 0.0, "y": 0.0, "z": 0.0},
                           "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
    }


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_teaching_panel_reads_same_state_and_selects_without_motion(app):
    teaching = TeachingState(TEACHING_POSES)
    teaching.store("weld_start", ("right_manipulator", ("j1",), (0.1,), object()),
                   {"source": "saved YAML"})
    window = SequenceMainWindow(teaching_state=teaching)
    window.teaching_panel.refresh()
    assert window.teaching_panel.table.item(3, 1).text() == "Available"
    assert window.teaching_panel.table.item(3, 2).text() == "Available"
    assert window.teaching_panel.table.item(3, 3).text() == "saved YAML"
    window.teaching_panel.table.selectRow(5)
    app.processEvents()
    assert teaching.selected_name == "weld_end"
    assert not window.teaching_panel.capture_button.isEnabled()
    window.close()


def test_teaching_yaml_load_updates_canonical_state(app, tmp_path):
    path = tmp_path / "start.yaml"
    path.write_text(yaml.safe_dump(teaching_yaml("right_manipulator")), encoding="utf-8")
    state = TeachingState(TEACHING_POSES)
    window = SequenceMainWindow(teaching_state=state)
    state.select("weld_start")
    window.teaching_panel.load_selected_file(path)
    assert state.poses["weld_start"][0] == "right_manipulator"
    assert window.teaching_panel.table.item(3, 2).text() == "Available"
    assert not window.execute_button.isEnabled()
    window.close()


def test_multipass_panel_selects_canonical_state_and_refreshes(app):
    state = MultiPassState(references={1: {"start": object()}},
                           corrected={2: {"start": object(), "goal": object()}},
                           registration={"phase": "waiting_goal_capture", "measured_start": object()})
    window = SequenceMainWindow(multipass_state=state)
    window.multipass_panel.pass_combo.setCurrentIndex(1)
    assert state.selected_pass == 2
    assert window.multipass_panel.table.item(1, 2).text() == "Available"
    state.corrected[3] = {"start": object()}
    state.status = "corrected"
    window.multipass_panel.refresh()
    assert window.multipass_panel.table.item(2, 2).text() == "Available"
    assert "corrected" in window.multipass_panel.status_label.text()
    assert not window.multipass_panel.correct_button.isEnabled()
    window.close()


def test_welding_editor_validates_without_sending(app):
    state = WeldConfigurationState()
    window = SequenceMainWindow(weld_state=state)
    panel = window.welding_panel
    panel.editors["current_a"].setValue(220)
    panel.editors["current_a"].editingFinished.emit()
    assert state.recipe["current_a"] == 220
    panel.editors["weld_tcp_speed_mm_s"].setValue(5.5)
    panel.editors["weld_tcp_speed_mm_s"].editingFinished.emit()
    assert state.motion["weld_tcp_speed_mm_s"] == 5.5
    old = state.recipe["current_a"]
    panel._commit("current_a", 500, False)
    assert state.recipe["current_a"] == old
    assert panel.editors["current_a"].value() == old
    assert window.status_panel.labels["error"].text() != "—"
    assert not window.execute_button.isEnabled()
    window.close()


def test_cleaner_build_uses_same_sequence_model(app, tmp_path):
    folder = tmp_path / "cleaner"
    folder.mkdir()
    joints = [f"right_manipulator_joint{i}" for i in range(1, 7)]
    for name in ("start", "end"):
        (folder / f"{name}.yaml").write_text(yaml.safe_dump({
            "schema": "torch_cleaner_joints_v1", "planning_group": "right_manipulator",
            "joint_state": {"names": joints, "positions_rad": [0.0] * 6},
        }), encoding="utf-8")
    (folder / "sequence.yaml").write_text(yaml.safe_dump({
        "schema": "torch_cleaner_sequence_v2", "positions": ["start", "DO7:1", "end"],
    }), encoding="utf-8")
    state = SequenceModel([{"type": "sleep", "seconds": 1.0}])
    cleaner = CleanerTeachingState(folder)
    window = SequenceMainWindow(state, cleaner_state=cleaner)
    preview = window.cleaner_panel.preview()
    assert len(preview) == 3
    assert len(state.steps) == 1
    assert "DO7" in window.cleaner_panel.preview_label.text()
    steps = window.cleaner_panel.build()
    app.processEvents()
    assert cleaner.tokens == ["start", "DO7:1", "end"]
    assert len(steps) == 3
    assert window.table_model.rowCount() == 4
    assert state.steps[2]["port"] == 7
    assert not window.execute_button.isEnabled()
    window.close()


def test_task_order_save_load_and_shared_builder(app, tmp_path):
    state = SequenceModel([{"type": "sleep", "seconds": 1.0}])
    window = SequenceMainWindow(state)
    panel = window.task_panel
    panel.folder_edit.setText(str(tmp_path))
    panel.refresh()
    poses = panel.base() / "poses"
    poses.mkdir(parents=True)
    for name in ("one", "two"):
        (poses / f"{name}.yaml").write_text(yaml.safe_dump(teaching_yaml()), encoding="utf-8")
    panel.refresh()
    panel.poses.setCurrentRow(0)
    panel.add_selected_pose()
    panel.poses.setCurrentRow(1)
    panel.add_selected_pose()
    assert panel.order_state.names == ["one", "two"]
    panel.order.setCurrentRow(1)
    panel.move_selected(-1)
    assert panel.order_state.names == ["two", "one"]
    target = panel.save_task()
    assert target.is_file()
    state.clear()
    panel.order_state.replace(())
    panel.load_task()
    app.processEvents()
    assert state.steps[0]["type"] == "sleep"
    assert panel.order_state.names == ["two", "one"]
    assert window.table_model.rowCount() == 1
    assert panel.build_button.isEnabled()
    built = panel.build_path()
    app.processEvents()
    assert [step["type"] for step in built] == ["named_pose", "motion"]
    assert window.table_model.rowCount() == 2
    window.close()


def test_touch_io_status_comes_from_bridge_signal(app):
    class ReadOnlyRuntime:
        def io_status(self):
            return {"fastech_connected": True, "touch_input": False,
                    "hicomm_connected": True}

    window = SequenceMainWindow(runtime=ReadOnlyRuntime())
    assert not window.execute_button.isEnabled()
    window.runtime_bridge.refresh()
    assert window.io_panel.labels["fastech_connected"].text() == "True"
    assert window.io_panel.labels["touch_input"].text() == "False"
    assert "Unknown" in window.io_panel.labels["control_box_io"].text()
    window.close()
