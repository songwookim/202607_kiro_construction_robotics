import pytest

from construct_robot.core.sequence_model import (
    SequenceModel,
    next_sequential_slot,
    validate_managed_weld_sequence,
)


def test_sequence_rows_are_owned_without_widgets():
    model = SequenceModel([{"type": "sleep", "parallel_slot": 1}])
    model.extend([{"type": "named_pose", "parallel_slot": 2}])
    model.select(1)
    assert model.delete(0)["type"] == "sleep"
    assert model.selected_index == 0
    model.replace([{"type": "sleep", "seconds": 0.5}])
    assert model.steps[0]["seconds"] == 0.5
    assert model.clear() == 1
    assert model.steps == []
    assert model.selected_index is None


def test_removing_selected_row_clears_selection():
    model = SequenceModel([{"type": "sleep", "seconds": 1.0}])
    model.select(0)
    model.delete(0)
    assert model.selected_index is None
    assert model.execution_snapshot(False) == ([], [])


def test_cleaner_replacement_preserves_other_rows_and_slots():
    weld = {"type": "motion", "parallel_slot": 3}
    stale = {"type": "named_pose", "torch_clean_scenario": True, "parallel_slot": 4}
    input_step = {"type": "named_pose", "torch_clean_scenario": True, "parallel_slot": 1}
    model = SequenceModel([weld, stale])
    result = model.with_replaced_cleaner([input_step])
    assert result == [weld, {**input_step, "parallel_slot": 4}]
    assert model.steps == [weld, stale]
    assert input_step["parallel_slot"] == 1
    with pytest.raises(ValueError, match="maximum parallel slot"):
        SequenceModel([{"type": "motion", "parallel_slot": 999}]).with_replaced_cleaner([input_step])


def test_validation_and_slot_assignment_remain_pure():
    steps = [{"type": "sleep", "parallel_slot": 50},
             {"type": "motion", "parallel_slot": 3}]
    assert next_sequential_slot(steps, 1) == 4
    assert validate_managed_weld_sequence(steps)
    with pytest.raises(ValueError, match="cannot share slot"):
        validate_managed_weld_sequence([
            {"type": "motion", "parallel_slot": 2},
            {"type": "digital_weld", "command": "on", "parallel_slot": 2},
        ])


def test_selected_snapshot_and_execution_progress_are_not_widget_state():
    model = SequenceModel([{"type": "sleep", "seconds": 1.0},
                           {"type": "sleep", "seconds": 2.0}])
    assert model.select(1) == 1
    indices, steps = model.execution_snapshot(False)
    assert indices == [1]
    steps[0]["seconds"] = 99.0
    assert model.steps[1]["seconds"] == 2.0
    model.start(indices, True)
    model.set_progress(4, 1, 3)
    assert model.running and model.current_indices == (1,)
    assert model.current_slot == 4 and model.group_progress == (1, 3)
    model.finish(True, "done")
    assert not model.running and model.status == "complete"
    assert model.current_indices == ()
