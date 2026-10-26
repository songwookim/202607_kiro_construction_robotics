"""Physical arrow presses start the jog in the ROS thread; Tk keeps the interlocks."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from construct_robot.nodes.weld_runtime_node import (
    KEYBOARD_FAST_CONTEXT_MAX_AGE_S,
    KEYBOARD_FAST_REFRESH_MAX_AGE_S,
    WeldGuiNode,
)

UP = (0.0, 0.0, 0.005, 0.0, 0.0, 0.0)


def context(**changes):
    values = dict(
        armed=True, arm="right", planning_group="right_manipulator",
        selection="XZ", frame="World", linear_m_s=0.005, angular_rad_s=0.05,
        updated_monotonic=time.monotonic(),
    )
    values.update(changes)
    return values


def fast_node(**ui_changes):
    node = object.__new__(WeldGuiNode)
    node.keyboard_servo = Mock()
    node.keyboard_velocity_publishers = {"left": Mock(), "right": Mock()}
    node.keyboard_velocity_lock = threading.Lock()
    node.keyboard_velocity_command = {
        "arm": None, "values": (0.0,) * 6, "refreshed_monotonic": time.monotonic(),
        "deadman_timeout_s": 0.8, "zero_burst_remaining": 0,
    }
    node.keyboard_fast_lock = threading.Lock()
    node.keyboard_fast_context = context()
    node.keyboard_command_key = None
    node.keyboard_physical_key = None
    node._keyboard_last_mask = 0
    node.active_motion_goal = None
    ui = dict(
        post=Mock(), keyboard_arrow_state_received=Mock(), sequence_running=False,
        post_latest=Mock(), keyboard_speed_selected=Mock(),
        keyboard_teaching_capture_in_progress=False, keyboard_velocity_switching=False,
        keyboard_velocity_arm="right",
    )
    ui.update(ui_changes)
    node.ui = SimpleNamespace(**ui)
    node.get_logger = lambda: Mock()
    node.resolve_keyboard_velocity = Mock(return_value=UP)
    return node


def arrow(node, mask):
    WeldGuiNode._keyboard_arrow_state(node, SimpleNamespace(data=mask))


def test_press_starts_the_jog_without_the_tk_queue():
    node = fast_node()
    arrow(node, 0x04)
    assert node.keyboard_velocity_command["values"] == UP
    assert node.keyboard_command_key_for("right") == "Up"
    node.keyboard_servo.publish_twist.assert_called_with("right", UP)
    node.resolve_keyboard_velocity.assert_called_once_with(
        "right_manipulator", "XZ", "Up", 0.005, 0.05, "World")
    node.ui.post.assert_called_once_with(node.ui.keyboard_arrow_state_received, 0x04)


def test_held_key_renews_the_deadman_lease_and_release_stops():
    node = fast_node()
    arrow(node, 0x04)
    node.keyboard_velocity_command["refreshed_monotonic"] -= 0.2
    arrow(node, 0x04)  # 10 Hz heartbeat while held
    assert time.monotonic() - node.keyboard_velocity_command["refreshed_monotonic"] < 0.05
    assert node.keyboard_velocity_command["deadman_timeout_s"] == 0.25
    arrow(node, 0x00)
    assert node.keyboard_velocity_command["values"] == (0.0,) * 6
    assert node.keyboard_command_key_for("right") is None


@pytest.mark.parametrize("change", [
    dict(sequence_running=True),
    dict(keyboard_teaching_capture_in_progress=True),
    dict(keyboard_velocity_switching=True),
    dict(keyboard_velocity_arm="left"),
])
def test_interlocks_are_checked_at_the_press(change):
    node = fast_node(**change)
    arrow(node, 0x04)
    node.resolve_keyboard_velocity.assert_not_called()
    assert node.keyboard_velocity_command["values"] == (0.0,) * 6


def test_running_goal_or_disarmed_context_never_starts():
    node = fast_node()
    node.active_motion_goal = object()
    arrow(node, 0x04)
    node.keyboard_fast_context = context(armed=False)
    node.active_motion_goal = None
    arrow(node, 0x00)
    arrow(node, 0x04)
    node.resolve_keyboard_velocity.assert_not_called()


def test_stale_context_leaves_the_start_to_tk_but_still_renews_a_held_jog():
    node = fast_node()
    node.keyboard_fast_context = context(
        updated_monotonic=time.monotonic() - KEYBOARD_FAST_CONTEXT_MAX_AGE_S - 0.05)
    arrow(node, 0x04)
    node.resolve_keyboard_velocity.assert_not_called()
    # Tk started it later; a GUI hiccup must not stop the held jog.
    WeldGuiNode.set_keyboard_velocity(node, "right", UP, key="Up")
    node.keyboard_velocity_command["refreshed_monotonic"] -= 0.2
    arrow(node, 0x04)
    assert time.monotonic() - node.keyboard_velocity_command["refreshed_monotonic"] < 0.05
    node.keyboard_fast_context = context(
        updated_monotonic=time.monotonic() - KEYBOARD_FAST_REFRESH_MAX_AGE_S - 0.05)
    node.keyboard_velocity_command["refreshed_monotonic"] -= 0.2
    arrow(node, 0x04)
    assert time.monotonic() - node.keyboard_velocity_command["refreshed_monotonic"] >= 0.2


def test_key_still_held_after_a_deliberate_stop_needs_a_new_press():
    node = fast_node()
    arrow(node, 0x04)
    WeldGuiNode.clear_keyboard_velocity(node)  # e.g. speed key or task change
    arrow(node, 0x04)
    assert node.keyboard_velocity_command["values"] == (0.0,) * 6
    assert node.resolve_keyboard_velocity.call_count == 1


def test_two_arrows_stop_and_the_remaining_one_is_a_new_press():
    node = fast_node()
    arrow(node, 0x01)
    arrow(node, 0x05)  # Left + Up together: STOP
    assert node.keyboard_velocity_command["values"] == (0.0,) * 6
    arrow(node, 0x04)  # Left released: Up is a new press
    assert node.keyboard_command_key_for("right") == "Up"


# ------------------------------------------------------------------- Tk side
class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def tk_gui(node):
    from construct_robot.gui.weld_action_gui import WeldActionGui

    gui = object.__new__(WeldActionGui)
    gui.root = SimpleNamespace(focus_get=lambda: object(), after_cancel=Mock())
    gui.node = node
    gui.keyboard_jog_enabled = Var(True)
    gui.keyboard_jog_selection = Var("XZ")
    gui.keyboard_jog_frame = Var("World")
    gui.keyboard_jog_linear_speed = Var(5.0)
    gui.keyboard_jog_angular_speed = Var(3.0)
    gui.planning_group = Var("right_manipulator")
    gui.keyboard_jog_status = Var("")
    gui.keyboard_ros_physical_key = None
    gui.keyboard_ros_physical_mask = 0
    gui.keyboard_ros_zero_seen = True
    gui.keyboard_ros_pending_key = None
    gui.keyboard_ros_input_last_at = time.monotonic()
    gui.keyboard_ros_dispatching = False
    gui.keyboard_velocity_active_key = None
    gui.keyboard_velocity_arm = "right"
    gui.keyboard_velocity_switching = False
    gui.keyboard_teaching_capture_in_progress = False
    gui.keyboard_release_after_id = None
    gui.keyboard_stop_generation = 0
    gui.sequence_running = False
    gui.log = Mock()
    gui.keyboard_jog_key_press = Mock()
    return gui


def test_tk_adopts_the_ros_thread_jog_instead_of_starting_it_again():
    node = fast_node()
    gui = tk_gui(node)
    arrow(node, 0x04)
    gui.keyboard_arrow_state_received(0x04)
    gui.keyboard_jog_key_press.assert_not_called()
    assert gui.keyboard_velocity_active_key == "Up"
    assert node.keyboard_velocity_command["values"] == UP
    assert gui.keyboard_jog_status.get().startswith("MOVING XZ Up")


def test_queued_release_does_not_clear_a_newer_ros_thread_press():
    node = fast_node()
    gui = tk_gui(node)
    arrow(node, 0x01)
    gui.keyboard_arrow_state_received(0x01)  # Left adopted
    arrow(node, 0x00)
    arrow(node, 0x04)                        # Up pressed before Tk caught up
    gui._stop_keyboard_jog_command = Mock()
    gui.keyboard_arrow_state_received(0x00)  # queued Left release
    gui._stop_keyboard_jog_command.assert_not_called()
    assert node.keyboard_command_key_for("right") == "Up"
    gui.keyboard_arrow_state_received(0x04)
    assert gui.keyboard_velocity_active_key == "Up"


def test_a_press_released_before_tk_ran_is_never_replayed():
    node = fast_node()
    node.keyboard_fast_context = context(armed=False)  # e.g. busy GUI
    gui = tk_gui(node)
    arrow(node, 0x04)
    arrow(node, 0x00)
    gui.keyboard_arrow_state_received(0x04)
    gui.keyboard_arrow_state_received(0x00)
    gui.keyboard_jog_key_press.assert_not_called()


def test_tk_falls_back_to_its_own_start_while_the_key_is_held():
    node = fast_node()
    node.keyboard_fast_context = context(armed=False)
    gui = tk_gui(node)
    arrow(node, 0x04)
    gui.keyboard_arrow_state_received(0x04)
    gui.keyboard_jog_key_press.assert_called_once()


def test_input_focus_stops_a_ros_thread_jog_tk_has_not_adopted():
    node = fast_node()
    gui = tk_gui(node)
    arrow(node, 0x04)
    gui._keyboard_focus_allows_jog = lambda: False
    gui._keyboard_focus_in_input_field = lambda: True
    gui._stop_keyboard_jog_command = Mock(side_effect=lambda *a: node.clear_keyboard_velocity())
    gui.keyboard_arrow_state_received(0x04)
    gui._stop_keyboard_jog_command.assert_called_once()
    assert node.keyboard_velocity_command["values"] == (0.0,) * 6


def test_context_is_armed_only_when_every_tk_interlock_allows_the_jog():
    gui = tk_gui(fast_node())
    gui._keyboard_focus_allows_jog = lambda: True
    assert gui._keyboard_fast_context()["armed"]
    for name, value in (
        ("keyboard_velocity_switching", True), ("sequence_running", True),
        ("keyboard_teaching_capture_in_progress", True), ("keyboard_velocity_arm", "left"),
        ("keyboard_ros_zero_seen", False),
    ):
        blocked = tk_gui(fast_node())
        blocked._keyboard_focus_allows_jog = lambda: True
        setattr(blocked, name, value)
        assert not blocked._keyboard_fast_context()["armed"], name
    gui._keyboard_focus_allows_jog = lambda: False
    assert not gui._keyboard_fast_context()["armed"]
    gui.keyboard_jog_enabled = Var(False)
    assert gui._keyboard_fast_context() == {
        "armed": False, "updated_monotonic": pytest.approx(time.monotonic(), abs=1.0)}


def test_axis_key_sees_a_ros_thread_jog_that_tk_has_not_adopted_yet():
    node = fast_node()
    node.keyboard_teaching_uses_servo = lambda: True
    gui = tk_gui(node)
    gui._keyboard_focus_allows_jog = lambda: True
    gui.keyboard_ros_physical_key = "Up"
    gui._cancel_keyboard_release_timer = Mock()
    arrow(node, 0x04)
    assert gui.keyboard_jog_selection_key(SimpleNamespace(keysym="3")) == "break"
    # Re-aimed at the new axis while held (servo backend), not dropped.
    gui.keyboard_jog_key_press.assert_called_once()
    assert gui.keyboard_jog_selection.get() == "Z"


def test_queued_release_does_not_clear_a_newer_press_of_the_same_arrow():
    node = fast_node()
    gui = tk_gui(node)
    gui._stop_keyboard_jog_command = Mock(side_effect=lambda *a: node.clear_keyboard_velocity())
    arrow(node, 0x04)
    gui.keyboard_arrow_state_received(0x04)  # adopted
    arrow(node, 0x00)
    arrow(node, 0x04)                        # tapped again before Tk caught up
    gui.keyboard_arrow_state_received(0x00)  # stale release
    gui.keyboard_arrow_state_received(0x04)
    gui._stop_keyboard_jog_command.assert_not_called()
    assert node.keyboard_velocity_command["values"] == UP
    assert gui.keyboard_active_token == node.keyboard_command_token("right")
    arrow(node, 0x00)
    gui.keyboard_arrow_state_received(0x00)  # the real release
    gui._stop_keyboard_jog_command.assert_called_once()
    assert node.keyboard_velocity_command["values"] == (0.0,) * 6


def test_tk_started_jog_is_tracked_by_its_command_generation():
    node = fast_node()
    node.keyboard_fast_context = context(armed=False)
    gui = tk_gui(node)
    gui.keyboard_jog_key_press = lambda event: (
        node.set_keyboard_velocity("right", UP, key=event.keysym),
        setattr(gui, "keyboard_velocity_active_key", event.keysym),
        setattr(gui, "keyboard_active_token", gui._keyboard_command_token()),
    )
    arrow(node, 0x08)
    gui.keyboard_arrow_state_received(0x08)
    token = node.keyboard_command_token("right")
    assert token[0] == "Down" and gui.keyboard_active_token == token
    node.keyboard_velocity_command["refreshed_monotonic"] -= 0.2
    gui.keyboard_arrow_state_received(0x08)  # heartbeat renews the lease
    assert time.monotonic() - node.keyboard_velocity_command["refreshed_monotonic"] < 0.05


def selection(node, char):
    WeldGuiNode._keyboard_selection_key(node, SimpleNamespace(data=ord(char)))


def test_axis_key_reaims_a_held_jog_in_the_ros_thread():
    node = fast_node()
    node.resolve_keyboard_velocity = Mock(side_effect=lambda g, sel, key, *a: (
        (0.005, 0, 0, 0, 0, 0) if sel == "X" else UP))
    arrow(node, 0x04)
    selection(node, "1")  # X while Up is held
    assert node.resolve_keyboard_velocity.call_args.args[1:3] == ("X", "Up")
    assert node.keyboard_velocity_command["values"] == (0.005, 0, 0, 0, 0, 0)
    assert node.keyboard_command_key_for("right") == "Up"


def test_axis_choice_survives_a_lagging_tk_context_until_tk_catches_up():
    node = fast_node()
    selection(node, "3")  # nothing held: only the selection changes
    node.resolve_keyboard_velocity.assert_not_called()
    WeldGuiNode.set_keyboard_fast_context(node, context(selection="XZ"))  # Tk not caught up yet
    assert node.keyboard_fast_context["selection"] == "Z"
    arrow(node, 0x04)
    assert node.resolve_keyboard_velocity.call_args.args[1] == "Z"
    WeldGuiNode.set_keyboard_fast_context(node, context(selection="Z"))  # Tk caught up
    assert node.keyboard_selection_override is None
    WeldGuiNode.set_keyboard_fast_context(node, context(selection="Y"))  # later Tk change wins
    assert node.keyboard_fast_context["selection"] == "Y"


def test_axis_key_does_nothing_for_native_jog_or_disarmed_context():
    node = fast_node()
    arrow(node, 0x04)
    calls = node.resolve_keyboard_velocity.call_count
    node.keyboard_servo = None  # native backend: Tk stops the jog on an axis change
    selection(node, "1")
    node.keyboard_fast_context = context(armed=False)
    selection(node, "2")
    assert node.resolve_keyboard_velocity.call_count == calls


def test_v_updates_speed_and_stops_a_held_jog_before_tk_runs():
    node = fast_node()
    arrow(node, 0x04)
    selection(node, "v")
    assert node.keyboard_fast_context["linear_m_s"] == pytest.approx(0.015)
    assert node.keyboard_velocity_command["values"] == (0.0,) * 6
    node.ui.post_latest.assert_called_once_with(
        "keyboard_speed", node.ui.keyboard_speed_selected, "right", 15.0, 1)
    arrow(node, 0x04)
    assert node.keyboard_command_key_for("right") is None
    arrow(node, 0x00)
    arrow(node, 0x04)
    assert node.resolve_keyboard_velocity.call_args.args[3] == pytest.approx(0.015)


def test_rapid_v_presses_use_the_latest_speed_until_gui_acknowledges():
    node = fast_node()
    for expected in (0.015, 0.045, 0.005, 0.015):
        selection(node, "v")
        node.set_keyboard_fast_context(context())
        assert node.keyboard_fast_context["linear_m_s"] == pytest.approx(expected)
    node.set_keyboard_fast_context(context(linear_m_s=0.015, speed_generation=4))
    assert node.keyboard_speed_override is None
    node.set_keyboard_fast_context(context(linear_m_s=0.045, speed_generation=4))
    assert node.keyboard_fast_context["linear_m_s"] == pytest.approx(0.045)


@pytest.mark.parametrize("change", [dict(armed=False), dict(updated_monotonic=0.0)])
def test_v_does_not_change_speed_from_a_disabled_or_stale_context(change):
    node = fast_node()
    node.keyboard_fast_context = context(**change)
    selection(node, "v")
    assert node.keyboard_fast_context["linear_m_s"] == pytest.approx(0.005)
    node.ui.post_latest.assert_not_called()


@pytest.mark.parametrize("change", [dict(sequence_running=True), dict(keyboard_velocity_switching=True),
                                    dict(keyboard_teaching_capture_in_progress=True),
                                    dict(keyboard_velocity_arm="left")])
def test_v_respects_live_motion_interlocks(change):
    node = fast_node(**change)
    selection(node, "v")
    node.ui.post_latest.assert_not_called()


def test_tk_does_not_cycle_v_again_when_physical_keyboard_is_online():
    from construct_robot.gui.weld_action_gui import WeldActionGui

    gui = object.__new__(WeldActionGui)
    gui._keyboard_ros_input_online = lambda: True
    gui.keyboard_jog_enabled = Var(True)
    gui._keyboard_focus_allows_jog = lambda: True
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="v")) == "break"
    gui._keyboard_focus_allows_jog = lambda: False
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="v")) is None
    gui.keyboard_jog_enabled.set(False)
    gui._keyboard_focus_allows_jog = lambda: True
    assert gui.keyboard_teaching_shortcut_key(SimpleNamespace(keysym="v")) is None


def test_delayed_speed_display_does_not_stop_a_newer_arrow_press():
    node = fast_node()
    gui = tk_gui(node)
    gui._refresh_keyboard_fast_context = Mock()
    gui._stop_keyboard_jog_command = Mock()
    selection(node, "v")
    arrow(node, 0x04)
    gui.keyboard_speed_selected("right", 15.0, 1)
    gui._stop_keyboard_jog_command.assert_not_called()
    assert gui.keyboard_jog_linear_speed.get() == 15.0
    assert node.keyboard_command_key_for("right") == "Up"
    gui.keyboard_speed_selected("right", 45.0, 0)
    assert gui.keyboard_jog_linear_speed.get() == 15.0


def test_speed_display_confirms_stop_without_publishing_a_late_zero():
    node = fast_node()
    gui = tk_gui(node)
    gui._refresh_keyboard_fast_context = Mock()
    gui._stop_keyboard_jog_command = Mock()
    selection(node, "v")
    gui.keyboard_speed_selected("right", 15.0, 1)
    gui._stop_keyboard_jog_command.assert_called_once_with(publish_zero=False)


def test_old_stop_check_cannot_restart_a_new_jog_not_yet_displayed():
    node = fast_node()
    gui = tk_gui(node)
    node.keyboard_teaching_uses_servo = Mock()
    arrow(node, 0x04)
    gui._keyboard_jog_stop_verified("right", gui.keyboard_stop_generation, False)
    node.keyboard_teaching_uses_servo.assert_not_called()
