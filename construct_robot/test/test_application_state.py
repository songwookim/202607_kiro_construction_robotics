from pathlib import Path

import pytest

from construct_robot.multipass import MultiPassState
from construct_robot.task_teaching_model import TaskOrderState, TeachingState
from construct_robot.torch_cleaner_teaching import CleanerTeachingState
from construct_robot.weld_action_gui import TEACHING_POSES, WeldActionGui


def test_named_teaching_data_is_owned_outside_tk():
    gui = object.__new__(WeldActionGui)
    gui.taught_robot_poses = {name: None for name in TEACHING_POSES}
    gui.teaching_state.store("weld_start_wait", ("right_manipulator", (), (), None),
                             {"source": "test"})
    assert gui.taught_robot_poses["weld_start_wait"][0] == "right_manipulator"
    assert gui.teaching_capture_provenance["weld_start_wait"] == {"source": "test"}
    assert gui.teaching_state.select("weld_start_wait") == "weld_start_wait"
    with pytest.raises(ValueError, match="Unknown teaching pose"):
        gui.teaching_state.select("unknown")
    assert isinstance(gui.teaching_state, TeachingState)


def test_multipass_working_set_and_selection_are_tk_independent():
    gui = object.__new__(WeldActionGui)
    gui.four_pass_references = {1: {"source": "immutable log"}}
    gui.four_pass_corrected = {1: {"start": "working value"}}
    gui.four_pass_history = [{"anchor": 1}]
    gui.multi_pass_registration = {"pass": 1, "phase": "start"}
    assert isinstance(gui.multipass_state, MultiPassState)
    assert gui.multipass_state.references is gui.four_pass_references
    assert gui.multipass_state.corrected is gui.four_pass_corrected
    assert gui.multipass_state.registration is gui.multi_pass_registration
    assert gui.multipass_state.select(4) == 4
    with pytest.raises(ValueError, match="Select Pass"):
        gui.multipass_state.select(5)


def test_task_and_cleaner_order_models():
    task = TaskOrderState(["a", "b"])
    task.add("c")
    assert task.move(2, -1) == 1
    assert task.names == ["a", "c", "b"]
    assert task.remove(1) == "c"
    cleaner = CleanerTeachingState(Path("/tmp/cleaner"))
    cleaner.set_order(["start", "DO7:ON", "DO7:OFF"])
    assert cleaner.tokens == ["start", "DO7:ON", "DO7:OFF"]
