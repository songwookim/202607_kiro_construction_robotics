from types import SimpleNamespace
from unittest.mock import Mock

from construct_robot.gui.weld_action_gui import WeldActionGui


class FakeRoot:
    def __init__(self):
        self.callbacks = {}
        self.next_id = 0
        self.focused_widget = object()

    def focus_get(self):
        return self.focused_widget

    def after(self, _delay, callback):
        self.next_id += 1
        self.callbacks[self.next_id] = callback
        return self.next_id

    def after_cancel(self, timer):
        self.callbacks.pop(timer, None)


def wire_gui():
    gui = object.__new__(WeldActionGui)
    gui.root = FakeRoot()
    gui.keyboard_jog_enabled = SimpleNamespace(get=lambda: True)
    gui.keyboard_velocity_switching = False
    # "right", not "right_manipulator": this is what the real _selected_arm()
    # returns and what keyboard_velocity_arm is assigned from.  The fixture
    # used to fake the planning-group name here, which made the guard in
    # keyboard_wire_key_press pass in tests while it could never pass live.
    gui.keyboard_velocity_arm = "right"
    gui.keyboard_wire_active_key = None
    gui.keyboard_wire_release_after_id = None
    gui.keyboard_jog_status = SimpleNamespace(set=Mock())
    gui.planning_group = SimpleNamespace(get=lambda: "right_manipulator")
    gui.node = SimpleNamespace(active_motion_goal=None)
    gui.sequence_running = False
    gui.hicomm_connected = True
    gui.error = Mock()
    gui.request_hicomm_inching = Mock(return_value=True)
    return gui


def test_keyboard_wire_hold_repeat_release_and_focus_loss():
    gui = wire_gui()
    forward = SimpleNamespace(keysym="f")
    reverse = SimpleNamespace(keysym="r")

    assert gui.keyboard_wire_key_press(forward) == "break"
    gui.request_hicomm_inching.assert_called_once_with("forward", True)
    gui.keyboard_wire_key_release(forward)
    assert gui.keyboard_wire_key_press(forward) == "break"
    assert not gui.root.callbacks
    assert gui.request_hicomm_inching.call_count == 1

    gui.keyboard_wire_key_release(forward)
    next(iter(gui.root.callbacks.values()))()
    gui.request_hicomm_inching.assert_called_with("forward", False)
    assert gui.keyboard_wire_active_key is None

    gui.keyboard_wire_key_press(reverse)
    gui.keyboard_wire_focus_out(None)
    gui.request_hicomm_inching.assert_called_with("reverse", False)


def test_keyboard_wire_requires_connected_idle_right_arm():
    gui = wire_gui()
    gui.hicomm_connected = False
    assert gui.keyboard_wire_key_press(SimpleNamespace(keysym="f")) == "break"
    gui.request_hicomm_inching.assert_not_called()

    gui.hicomm_connected = True
    gui.sequence_running = True
    assert gui.keyboard_wire_key_press(SimpleNamespace(keysym="r")) == "break"
    gui.request_hicomm_inching.assert_not_called()


def test_keyboard_jog_focus_rejects_native_tk_messagebox():
    gui = wire_gui()
    gui.root.focus_get = Mock(side_effect=KeyError(".__tk__messagebox"))

    assert gui._keyboard_focus_allows_jog() is False


# ----------------------------------------------------- dropped key presses
import time as _time
import tkinter as _tk
from tkinter import ttk as _ttk


def arrow_gui(focus_widget):
    gui = object.__new__(WeldActionGui)
    gui.root = SimpleNamespace(focus_get=lambda: focus_widget, focus_set=Mock())
    gui.keyboard_jog_enabled = SimpleNamespace(get=lambda: True)
    gui.keyboard_ros_physical_key = None
    gui.keyboard_ros_physical_mask = 0
    gui.keyboard_ros_zero_seen = True
    gui.keyboard_ros_pending_key = None
    gui.keyboard_ros_input_last_at = _time.monotonic()
    gui.keyboard_ros_dispatching = False
    gui.keyboard_velocity_active_key = None
    gui.keyboard_velocity_arm = "right"
    gui.keyboard_jog_status = SimpleNamespace(set=Mock())
    gui.node = SimpleNamespace(refresh_keyboard_velocity=Mock())
    gui.keyboard_jog_key_press = Mock(side_effect=lambda event: setattr(
        gui, "keyboard_velocity_active_key", event.keysym))
    gui._stop_keyboard_jog_command = Mock()
    gui.log = Mock()
    return gui


class FakeSpinbox(_ttk.Spinbox):
    def __init__(self):  # no Tk interpreter needed for isinstance checks
        pass


class FakeCombobox(_ttk.Combobox):
    def __init__(self, state):
        self._state = state

    def cget(self, _name):
        return self._state


def test_readonly_selectors_no_longer_block_teaching_keys():
    assert WeldActionGui._keyboard_focus_accepts_arrows(FakeCombobox("readonly")) is False
    assert WeldActionGui._keyboard_focus_accepts_arrows(FakeCombobox("normal")) is True
    assert WeldActionGui._keyboard_focus_accepts_arrows(FakeSpinbox()) is True


def test_press_dropped_for_input_focus_starts_once_focus_returns():
    spinbox = FakeSpinbox()
    gui = arrow_gui(spinbox)
    gui.keyboard_arrow_state_received(0x04)  # Up pressed, focus in a spinbox
    gui.keyboard_jog_key_press.assert_not_called()
    assert gui.keyboard_ros_pending_key == "Up"
    gui.root.focus_get = lambda: object()  # focus back on the main window
    gui.keyboard_arrow_state_received(0x04)  # heartbeat, still held
    gui.keyboard_jog_key_press.assert_called_once()
    assert gui.keyboard_velocity_active_key == "Up"


def test_release_clears_a_pending_press():
    gui = arrow_gui(FakeSpinbox())
    gui.keyboard_arrow_state_received(0x04)
    gui.keyboard_arrow_state_received(0x00)
    gui.root.focus_get = lambda: object()
    gui.keyboard_arrow_state_received(0x00)
    gui.keyboard_jog_key_press.assert_not_called()


def test_press_while_another_app_has_focus_is_not_queued():
    gui = arrow_gui(None)  # e.g. RViz focused
    gui.keyboard_arrow_state_received(0x04)
    assert gui.keyboard_ros_pending_key is None
    gui.root.focus_get = lambda: object()
    gui.keyboard_arrow_state_received(0x04)
    gui.keyboard_jog_key_press.assert_not_called()


def test_deliberate_stop_still_needs_a_new_press():
    gui = arrow_gui(object())
    gui.keyboard_arrow_state_received(0x04)
    assert gui.keyboard_velocity_active_key == "Up"
    gui.keyboard_velocity_active_key = None  # deadman / stop fallback
    gui.keyboard_arrow_state_received(0x04)
    assert gui.keyboard_jog_key_press.call_count == 1


def test_arrow_in_a_widget_jogs_instead_of_editing_it_in_teaching_mode():
    gui = arrow_gui(FakeSpinbox())
    gui.root.after_idle = Mock()
    assert gui._teaching_arrow_in_widget(SimpleNamespace(keysym="Up")) == "break"
    gui.root.focus_set.assert_called_once()
    gui.keyboard_jog_enabled = SimpleNamespace(get=lambda: False)
    assert gui._teaching_arrow_in_widget(SimpleNamespace(keysym="Up")) is None
