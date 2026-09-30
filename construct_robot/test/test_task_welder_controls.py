from types import SimpleNamespace
from unittest.mock import Mock

from construct_robot.gui import weld_action_gui as module


def test_manual_arc_requires_confirmation_without_unlock_checkbox(monkeypatch):
    gui = object.__new__(module.WeldActionGui)
    gui.planning_group = SimpleNamespace(get=lambda: "right_manipulator")
    gui.hicomm_client = Mock()
    gui.error = Mock()
    confirm = Mock(return_value=False)
    monkeypatch.setattr(module.messagebox, "askyesno", confirm)
    gui.request_digital_arc(True)
    confirm.assert_called_once()
    gui.hicomm_client.allow_outputs.assert_not_called()


def test_manual_arc_still_rejects_left_arm(monkeypatch):
    gui = object.__new__(module.WeldActionGui)
    gui.planning_group = SimpleNamespace(get=lambda: "left_manipulator")
    gui.error = Mock()
    confirm = Mock()
    monkeypatch.setattr(module.messagebox, "askyesno", confirm)
    gui.request_digital_arc(True)
    gui.error.assert_called_once()
    confirm.assert_not_called()


def test_connected_controls_do_not_require_removed_unlock_widget():
    gui = object.__new__(module.WeldActionGui)
    gui.hicomm_connected = True
    widgets = [Mock() for _ in range(4)]
    (gui.hicomm_forward_button, gui.hicomm_reverse_button,
     gui.hicomm_gas_check, gui.hicomm_arc_on_button) = widgets
    gui._set_welder_test_controls(True)
    for widget in widgets:
        widget.configure.assert_called_with(state=module.tk.NORMAL)
    gui._set_welder_test_controls(False)
    for widget in widgets:
        widget.configure.assert_called_with(state=module.tk.DISABLED)
