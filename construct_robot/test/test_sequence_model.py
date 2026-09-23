import pytest

from construct_robot.sequence_model import (
    SequenceModel,
    next_sequential_slot,
    validate_managed_weld_sequence,
)


def test_sequence_rows_are_owned_without_widgets():
    model = SequenceModel([{"type": "sleep", "parallel_slot": 1}])
    index = model.add({"type": "named_pose", "parallel_slot": 2})
    assert index == 1
    duplicate = model.duplicate(index)
    assert duplicate == 2
    assert model.steps[duplicate] is not model.steps[index]
    assert model.move(duplicate, -1) == 1
    model.edit(1, {"type": "sleep", "seconds": 0.5})
    assert model.delete(1)["seconds"] == 0.5
    assert model.clear() == 2
    assert model.steps == []


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
