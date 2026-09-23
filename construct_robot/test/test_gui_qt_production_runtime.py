import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from construct_robot.gui_qt.main_window import SequenceMainWindow
from construct_robot.multipass import MultiPassState
from construct_robot.production_runtime import (
    ProductionRuntimePort, _MOTION_VARIABLES, _RECIPE_VARIABLES,
)
from construct_robot.sequence_model import SequenceModel
from construct_robot.task_teaching_model import TEACHING_POSES, TeachingState
from construct_robot.weld_config import WeldConfigurationState


class Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def production():
    gui = SimpleNamespace()
    gui.sequence_model = SequenceModel([{"type": "sleep", "seconds": 1.0}])
    gui.teaching_state = TeachingState(TEACHING_POSES)
    gui.multipass_state = MultiPassState()
    gui.sequence_table = SimpleNamespace(selection=Mock(return_value=()),
                                         selection_remove=Mock())
    gui.refresh_sequence_table = Mock()
    gui.run_sequence = Mock()
    gui.stop_sequence = Mock()
    gui.capture_initial_state = Mock()
    gui.load_initial_state_from_path = Mock()
    gui.teaching_pose_changed = Mock()
    gui.run_four_pass_correction = Mock()
    gui.four_pass_folder = Variable("/tmp")
    gui.load_four_pass_references = Mock(return_value=True)
    gui.multi_pass_registration = None
    gui.keyboard_teaching_shortcut_key = Mock(return_value="break")
    gui.keyboard_teaching_shortcut_release = Mock()
    gui.apply_selected_pass_correction = Mock()
    gui.stop_multi_pass_correction = Mock()
    gui.build_sensed_weld_sequence = Mock()
    gui.selected_pass_number = Variable(1)
    gui.planning_group = Variable("right_manipulator")
    gui.teaching_pose_name = Variable(TEACHING_POSES["robot_start"])
    gui.node = SimpleNamespace(active_motion_goal=None)
    gui.robot_connected = {"left": True, "right": False, "head": False}
    gui.fastech_connected = False
    gui.fastech_previous_state = None
    gui.previous_control_box_io = None
    gui.hicomm_connected = False
    gui.hicomm_client = None
    gui.pipeline_status = SimpleNamespace(cget=lambda _name: "WAITING · ready")
    defaults = WeldConfigurationState()
    gui._digital_weld_settings = Mock(return_value=defaults.recipe)
    for key, attr in _RECIPE_VARIABLES.items():
        setattr(gui, attr, Variable(defaults.recipe[key]))
    for key, attr in _MOTION_VARIABLES.items():
        setattr(gui, attr, Variable(defaults.motion[key]))
    return gui, ProductionRuntimePort(gui)


def test_port_delegates_plan_execute_stop_to_existing_path(production):
    gui, port = production
    gui.sequence_table.selection.return_value = ("0",)
    port.plan_sequence()
    port.execute_sequence()
    port.stop_sequence()
    assert gui.sequence_table.selection_remove.call_count == 2
    assert gui.refresh_sequence_table.call_count == 2
    assert gui.run_sequence.call_args_list[0].args == (True, False)
    assert gui.run_sequence.call_args_list[1].args == (True, True)
    gui.stop_sequence.assert_called_once_with()


def test_port_capture_and_multipass_use_existing_workflow(production, tmp_path):
    gui, port = production
    port.capture_teaching_pose("weld_start", "right_manipulator")
    assert gui.teaching_pose_name.get() == TEACHING_POSES["weld_start"]
    gui.capture_initial_state.assert_called_once_with()
    port.load_teaching_pose("weld_start", "right_manipulator", tmp_path / "pose.yaml")
    gui.load_initial_state_from_path.assert_called_once_with(tmp_path / "pose.yaml", "weld_start")
    with pytest.raises(ValueError, match="Select this arm"):
        port.capture_teaching_pose("weld_start", "left_manipulator")
    port.begin_multi_pass_registration(3)
    assert gui.selected_pass_number.get() == 3
    assert port.multipass_state.selected_pass == 3
    gui.run_four_pass_correction.assert_called_once_with()
    port.load_multi_pass_references(tmp_path)
    assert gui.four_pass_folder.get() == str(tmp_path)
    gui.load_four_pass_references.assert_called_once_with()
    port.load_selected_pass(2)
    gui.apply_selected_pass_correction.assert_called_once_with()
    port.stop_multi_pass_registration()
    gui.stop_multi_pass_correction.assert_called_once_with()
    gui.multi_pass_registration = {"phase": "waiting_start_capture"}
    port.capture_multi_pass_start()
    assert gui.keyboard_teaching_shortcut_key.call_args.args[0].keysym == "i"
    gui.multi_pass_registration = {"phase": "waiting_goal_capture"}
    port.capture_multi_pass_goal()
    assert gui.keyboard_teaching_shortcut_key.call_args.args[0].keysym == "j"
    assert gui.keyboard_teaching_shortcut_release.call_count == 2
    with pytest.raises(ValueError, match="waiting_start_capture"):
        port.capture_multi_pass_start()


def test_port_weld_defaults_do_not_command_welder(production):
    gui, port = production
    state = WeldConfigurationState()
    state.set_recipe("current_a", 220)
    state.set_motion("weld_tcp_speed_mm_s", 4.0)
    assert gui.weld_current_raw.get() == 200
    assert gui.weld_tcp_speed_mm_s.get() == 3.0
    port.apply_weld_configuration(state)
    assert gui.weld_current_raw.get() == 220
    assert gui.weld_tcp_speed_mm_s.get() == 4.0
    assert gui.run_sequence.call_count == 0
    port.build_weld_scenario(state)
    gui.build_sensed_weld_sequence.assert_called_once_with()
    assert gui.run_sequence.call_count == 0


def test_cleaner_runtime_uses_existing_sequence_path_and_only_cleaner_rows(production):
    gui, port = production
    cleaner = {"type": "digital_output", "port": 7, "torch_clean_scenario": True}
    port.sequence_state.replace([{"type": "sleep", "seconds": 1.0}, cleaner])
    port.plan_cleaner()
    port.execute_cleaner()
    assert gui.run_sequence.call_args_list[0].args == (True, False)
    assert gui.run_sequence.call_args_list[0].kwargs["steps_override"] == [cleaner]
    assert gui.run_sequence.call_args_list[1].args == (True, True)
    assert gui.run_sequence.call_args_list[1].kwargs["steps_override"] == [cleaner]


def test_qt_actions_share_models_and_startup_has_no_commands(app, production):
    gui, port = production
    window = SequenceMainWindow(port.sequence_state, port,
                                teaching_state=port.teaching_state,
                                multipass_state=port.multipass_state)
    app.processEvents()
    assert gui.run_sequence.call_count == 0
    assert gui.stop_sequence.call_count == 0
    assert gui.capture_initial_state.call_count == 0
    assert gui.load_initial_state_from_path.call_count == 0
    assert gui.build_sensed_weld_sequence.call_count == 0
    window.welding_panel.editors["current_a"].setValue(210)
    window.welding_panel.editors["current_a"].editingFinished.emit()
    assert gui.weld_current_raw.get() == 200
    assert gui.run_sequence.call_count == 0
    window.welding_panel.apply_button.click()
    assert gui.weld_current_raw.get() == 210
    assert gui.run_sequence.call_count == 0
    assert window.plan_button.isEnabled()
    assert window.teaching_panel.capture_button.isEnabled()
    window.plan_button.click()
    window.execute_button.click()
    window.stop_button.click()
    window.teaching_panel.table.selectRow(3)
    window.teaching_panel.capture_button.click()
    window.teaching_panel.load_selected_file("/tmp/teaching.yaml")
    window.multipass_panel.pass_combo.setCurrentIndex(2)
    window.multipass_panel.correct_button.click()
    port.multipass_state.corrected[3] = {"start": object()}
    window.multipass_panel.refresh()
    window.multipass_panel.load_button.click()
    assert gui.run_sequence.call_count == 2
    gui.stop_sequence.assert_called_once_with()
    gui.capture_initial_state.assert_called_once_with()
    gui.load_initial_state_from_path.assert_called_once_with("/tmp/teaching.yaml", "weld_start")
    gui.run_four_pass_correction.assert_called_once_with()
    gui.apply_selected_pass_correction.assert_called_once_with()
    window.close()


def test_worker_status_is_queued_through_qt_signal(app, production):
    _gui, port = production
    window = SequenceMainWindow(port.sequence_state, port,
                                teaching_state=port.teaching_state,
                                multipass_state=port.multipass_state)
    port.sequence_state.start([0], False)
    worker = threading.Thread(target=lambda: port.sequence_state.set_progress("slot", 1, 2, (0,)))
    worker.start()
    worker.join()
    app.processEvents()
    app.processEvents()
    assert window.status_panel.labels["progress"].text() == "1/2"
    assert window.status_panel.labels["current"].text() == "1"
    port.sequence_state.finish(True, "done")
    window.close()


def test_read_only_io_projection_uses_existing_snapshots(production):
    gui, port = production
    gui.fastech_connected = True
    gui.fastech_previous_state = SimpleNamespace(
        digital_in=[False, False, False, False, True], digital_out=[True])
    gui.previous_control_box_io = ((False, True), (True, False))
    gui.hicomm_connected = True
    gui.hicomm_client = SimpleNamespace(latest_status=lambda: {"output_state_name": "Main welding"})
    assert port.io_status()["touch_input"] is True
    assert port.io_status()["touch_enabled"] is True
    assert port.io_status()["welder_output_state"] == "Main welding"
    assert "DI=" in port.io_status()["control_box_io"]


def test_production_launcher_startup_only_constructs_existing_runtime(app, production, monkeypatch):
    import rclpy
    from construct_robot import weld_action_gui
    from construct_robot.gui_qt import production_launch

    gui, _port = production
    gui.root = SimpleNamespace(title=Mock(), update=Mock())
    gui.close = Mock()
    gui.shutdown_ros = Mock()
    monkeypatch.setattr(weld_action_gui, "WeldActionGui", lambda: gui)
    monkeypatch.setattr(rclpy, "init", Mock())
    monkeypatch.setattr(QApplication, "exec", lambda _self: 0)
    assert production_launch.main() == 0
    assert gui.run_sequence.call_count == 0
    assert gui.capture_initial_state.call_count == 0
    assert gui.build_sensed_weld_sequence.call_count == 0
    gui.close.assert_called_once_with()
    gui.shutdown_ros.assert_called_once_with()
