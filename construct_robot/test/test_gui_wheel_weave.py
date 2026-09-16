from types import SimpleNamespace
from unittest.mock import Mock

from construct_robot.weld_action_gui import (
    WeldActionGui, weld_weave_settings_text,
)


def test_wheel_over_value_control_scrolls_page_and_stops_widget_binding():
    gui = object.__new__(WeldActionGui)
    gui.content_canvas = SimpleNamespace(yview_scroll=Mock())
    assert gui._scroll_value_control(SimpleNamespace(delta=-120, num=None)) == "break"
    gui.content_canvas.yview_scroll.assert_called_once_with(2, "units")


def test_weld_weave_log_explains_sine_and_circle_amplitude():
    settings = {
        "weld_weave_enabled": True,
        "weld_weave_pattern": "sine",
        "weld_weave_amplitude_mm": 2.7,
        "weld_weave_pitch_mm": 8.0,
        "weld_weave_actual_pitch_mm": 7.5,
        "weld_weave_cycles": 4,
        "weld_weave_axis": "tool_y",
        "weld_weave_left_dwell_s": 0.2,
        "weld_weave_right_dwell_s": 0.3,
    }
    sine = weld_weave_settings_text(settings)
    assert "centerline ±2.70 mm" in sine
    assert "full width 5.40 mm" in sine
    assert "actual pitch 7.50 mm/cycle" in sine
    assert "dwell L/R 0.20/0.30 s" in sine

    settings["weld_weave_pattern"] = "circle"
    circle = weld_weave_settings_text(settings)
    assert "radius 2.70 mm" in circle
    assert "diameter 5.40 mm" in circle
    assert weld_weave_settings_text({"weld_weave_enabled": False}) == "OFF"
