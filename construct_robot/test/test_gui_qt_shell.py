"""Presentation-only shell checks; no equipment runtime is constructed."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QStackedWidget

from construct_robot.gui_qt.main_window import SequenceMainWindow
from construct_robot.sequence_model import SequenceModel


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_navigation_keeps_shared_models_and_selection(app):
    sequence = SequenceModel([{"type": "sleep", "seconds": 1.0}])
    window = SequenceMainWindow(sequence)
    assert isinstance(window.pages, QStackedWidget)
    assert window.pages.currentWidget() is not None
    window.table.selectRow(0)
    app.processEvents()
    assert sequence.selected_index == 0
    original_panels = (window.teaching_panel, window.multipass_panel, window.welding_panel)
    assert window.teaching_panel.state is window.teaching_state
    assert window.multipass_panel.state is window.multipass_state
    assert window.welding_panel.state is window.weld_state
    for title, panel in zip(("Teaching", "Multi-pass", "Welding"), original_panels):
        window.nav_buttons[title].click()
        assert window.pages.currentWidget() is panel
        assert window.sequence_state is sequence
        assert sequence.selected_index == 0
    window.nav_buttons["Sequence"].click()
    assert window.pages.currentIndex() == 0
    assert window.property_panel.heading.text().startswith("Step 1")
    window.close()


def test_persistent_status_and_execution_controls_use_existing_bridge(app):
    class Runtime:
        def __init__(self, state):
            self.sequence_state = state
            self.calls = []

        def plan_sequence(self):
            self.calls.append("plan")

        def execute_sequence(self):
            self.calls.append("execute")

        def stop_sequence(self):
            self.calls.append("stop")

        def robot_status(self):
            return {"left": True, "right": False}

        def io_status(self):
            return {"hicomm_connected": True, "fastech_connected": False}

    state = SequenceModel([{"type": "sleep", "seconds": 1.0}])
    runtime = Runtime(state)
    window = SequenceMainWindow(state, runtime)
    assert runtime.calls == []
    window.show()
    app.processEvents()
    window.nav_buttons["Touch / IO"].click()
    assert window.status_panel.isVisibleTo(window)
    assert window.top_status.chips["left"].text().endswith("CONNECTED")
    assert window.top_status.chips["right"].text().endswith("DISCONNECTED")
    assert window.top_status.chips["welder"].text().endswith("CONNECTED")
    assert window.top_status.chips["touch"].text().endswith("DISCONNECTED")
    assert window.io_panel.labels["hicomm_connected"].text() == "True"
    window.plan_button.click()
    window.stop_button.click()
    assert runtime.calls == ["plan", "stop"]
    window.close()
