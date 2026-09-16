from types import SimpleNamespace
from unittest.mock import Mock

from construct_robot.weld_action_gui import WeldActionGui


class FakeRoot:
    def __init__(self):
        self.callbacks = {}
        self.next_id = 0

    def focus_get(self):
        return None

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
    gui.keyboard_velocity_arm = "right_manipulator"
    gui.keyboard_wire_active_key = None
    gui.keyboard_wire_release_after_id = None
    gui.keyboard_jog_status = SimpleNamespace(set=Mock())
    gui._selected_arm = lambda: "right_manipulator"
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
