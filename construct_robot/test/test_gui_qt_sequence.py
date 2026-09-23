import pytest
import yaml

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from construct_robot.gui_qt.main_window import SequenceMainWindow
from construct_robot.gui_qt.sequence_model_qt import SequenceTableModel
from construct_robot.sequence_model import SequenceModel


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_qt_table_renders_and_mutates_canonical_sequence(app):
    state = SequenceModel([{"type": "sleep", "seconds": 1.0, "parallel_slot": 1},
                           {"type": "named_pose", "pose_label": "Start",
                            "parallel_slot": 2, "torch_clean_scenario": True}])
    table = SequenceTableModel(state)
    assert table.rowCount() == 2
    assert table.data(table.index(1, 2), Qt.DisplayRole) == "Start"
    assert table.data(table.index(1, 3), Qt.DisplayRole) == 2
    table.select_row(0)
    table.update_selected_field("seconds", 0.5)
    app.processEvents()
    assert state.steps[0]["seconds"] == 0.5
    table.duplicate_selected()
    app.processEvents()
    assert table.rowCount() == 3 and state.selected_index == 1
    table.move_selected(1)
    app.processEvents()
    assert state.selected_index == 2
    table.delete_selected()
    app.processEvents()
    assert table.rowCount() == 2
    table.close()


def test_qt_selection_and_external_model_refresh(app):
    state = SequenceModel([{"type": "sleep", "seconds": 1.0}])
    window = SequenceMainWindow(state)
    assert not window.plan_button.isEnabled()
    assert not window.execute_button.isEnabled()
    window.table.selectRow(0)
    app.processEvents()
    assert state.selected_index == 0
    state.replace([{"type": "sleep", "seconds": 2.0},
                   {"type": "digital_output", "port": 7,
                    "torch_clean_scenario": True, "parallel_slot": 2}])
    app.processEvents()
    assert window.table_model.rowCount() == 2
    assert window.table_model.data(window.table_model.index(1, 1)) == "digital_output"
    window.close()


def test_qt_property_editor_validation_and_progress(app):
    state = SequenceModel([{"type": "sleep", "seconds": 1.0, "parallel_slot": 1}])
    window = SequenceMainWindow(state)
    window.table.selectRow(0)
    app.processEvents()
    editor = window.property_panel.findChild(QDoubleSpinBox)
    assert editor is not None
    editor.setValue(0.25)
    editor.editingFinished.emit()
    app.processEvents()
    assert state.steps[0]["seconds"] == 0.25
    with pytest.raises(ValueError, match="duration"):
        window.table_model.update_selected_field("seconds", -1.0)
    assert state.steps[0]["seconds"] == 0.25
    assert "duration" in window.status_panel.labels["error"].text()
    state.start([0], False)
    state.set_progress("sleep", 1, 1, (0,))
    app.processEvents()
    window.runtime_bridge.refresh()
    assert window.table_model.data(window.table_model.index(0, 4)) == "CURRENT"
    assert window.status_panel.labels["current"].text() == "1"
    state.finish(True, "complete")
    app.processEvents()
    window.close()


def test_external_invalid_sequence_shows_validation_error(app):
    state = SequenceModel()
    window = SequenceMainWindow(state)
    state.replace([{"type": "motion", "parallel_slot": 2},
                   {"type": "digital_weld", "command": "on", "parallel_slot": 2}])
    app.processEvents()
    assert "cannot share slot" in window.status_panel.labels["error"].text()
    window.close()


def test_qt_runtime_port_is_explicit_and_disabled_by_default(app):
    class FakeRuntime:
        def __init__(self, sequence_state):
            self.calls = []
            self.sequence_state = sequence_state

        def plan_sequence(self):
            self.calls.append("plan")

        def execute_sequence(self):
            self.calls.append("execute")

        def stop_sequence(self):
            self.calls.append("stop")

        def robot_status(self):
            return {"left": True, "right": False}

    state = SequenceModel()
    runtime = FakeRuntime(state)
    window = SequenceMainWindow(state, runtime)
    assert window.plan_button.isEnabled()
    window.plan_button.click()
    window.stop_button.click()
    window.runtime_bridge.refresh()
    assert runtime.calls == ["plan", "stop"]
    assert window.status_panel.labels["left"].text() == "Connected"
    assert window.status_panel.labels["right"].text() == "Disconnected"
    window.close()

    mismatch = SequenceMainWindow(SequenceModel(), runtime)
    assert not mismatch.plan_button.isEnabled()
    assert not mismatch.execute_button.isEnabled()
    mismatch.close()


def test_qt_imports_existing_task_and_cleaner_rows_without_execution(app, tmp_path):
    path = tmp_path / "cleaner_task.yaml"
    path.write_text(yaml.safe_dump({
        "schema": "robot_task_v1", "category": "Right · Torch cleaner",
        "visit_order": [],
        "steps": [{"type": "digital_output", "planning_group": "right_manipulator",
                   "port": 7, "value": True, "torch_clean_scenario": True,
                   "parallel_slot": 1}],
    }))
    window = SequenceMainWindow()
    window.load_task_path(path)
    app.processEvents()
    assert window.table_model.rowCount() == 1
    assert window.sequence_state.steps[0]["port"] == 7
    assert window.task_panel.category.currentText() == "Right · Torch cleaner"
    assert window.task_panel.task_name.text() == "cleaner_task"
    assert not window.execute_button.isEnabled()
    window.close()
