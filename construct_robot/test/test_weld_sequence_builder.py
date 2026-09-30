"""Weld-scenario generation without Tk (application.weld_sequence_builder)."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

from geometry_msgs.msg import Pose
import pytest

from construct_robot.application import weld_sequence_builder as builder
from construct_robot.application.weld_sequence_builder import (
    WeldScenarioInput,
    WeldStepMotionInput,
    allocate_weld_scenario_slots,
    build_weld_scenario,
    plan_weld_approach,
    resolve_weld_endpoints,
    validate_required_weld_poses,
)
from construct_robot.core.cartesian_path_common import weave_path_speed_m_s
from construct_robot.core.sequence_model import validate_managed_weld_sequence
from construct_robot.core.weld_config import (
    DEFAULT_DIGITAL_WELD_SETTINGS,
    validate_digital_weld_settings,
)

Q = (0.0, 0.9238795, 0.0, 0.3826834)
JOINTS = tuple(f"right_manipulator_joint{i}" for i in range(1, 7))


def pose(x, y, z, q=Q):
    result = Pose()
    result.position.x, result.position.y, result.position.z = x, y, z
    (result.orientation.x, result.orientation.y,
     result.orientation.z, result.orientation.w) = q
    return result


def stored(p, group="right_manipulator"):
    return (group, JOINTS, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), p)


def teaching():
    return {
        "weld_start_wait": stored(pose(0.50, 0.20, 0.30)),
        "weld_start": stored(pose(0.50, 0.25, 0.10)),
        "weld_goal_wait": stored(pose(0.70, 0.20, 0.30)),
        "weld_end": stored(pose(0.70, 0.25, 0.10)),
        "weld_finish": stored(pose(0.75, 0.10, 0.40)),
    }


GEOMETRY = SimpleNamespace(
    d_real=(1.0, 0.0, 0.0),
    e_a=(0.0, -0.70710678, 0.70710678),
    e_w=(0.0, 0.70710678, 0.70710678),
)


def scenario(sensed=False, settings=None, **overrides):
    poses = teaching()
    sensed_endpoints = {
        "start": pose(0.501, 0.252, 0.101),
        "goal": pose(0.702, 0.249, 0.099),
    }
    recipe = dict(DEFAULT_DIGITAL_WELD_SETTINGS)
    recipe.update(settings or {})
    values = dict(
        endpoints=resolve_weld_endpoints(
            poses, start_is_sensed=sensed, goal_is_sensed=sensed,
            sensed_endpoints=sensed_endpoints,
        ),
        goal_wait=poses["weld_goal_wait"],
        finish=poses["weld_finish"],
        settings=validate_digital_weld_settings(recipe),
        corner_touch_count=5,
        lead_in_mm=0.0,
        lead_out_mm=0.0,
        safe_approach_mm=20.0,
        pre_start_lead_mm=0.0,
        arc_off_delay_ms=120.0,
        weld_tcp_speed_mm_s=6.0,
        weave_enabled=False,
        weave_pattern="sine",
        weave_amplitude_mm=2.0,
        weave_pitch_mm=4.0,
        weave_left_dwell_s=0.0,
        weave_right_dwell_s=0.0,
        weave_axis="tool_y",
        approach_mode="corner_geometry",
        touch_io_backend="fastech_ethernet",
        touch_output_port=0,
        seam_geometry=GEOMETRY if sensed else None,
    )
    values.update(overrides)
    return WeldScenarioInput(**values)


MOTION = WeldStepMotionInput(
    velocity_percent=35.0,
    tcp_speed_m_s=0.0,
    interpolation_step_mm=2.0,
    seam_orientation_mode="Wait poses + fixed World-XYZ tilt",
    fixed_world_x_tilt_deg=0.0,
    fixed_world_y_tilt_deg=15.0,
    fixed_world_z_tilt_deg=-5.0,
)


def stages(steps):
    return [step["weld_scenario_stage"] for step in steps]


def test_builder_imports_no_tkinter_or_gui():
    tree = ast.parse(Path(builder.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    for name in imported:
        assert not (name or "").startswith(
            ("tkinter", "construct_robot.gui", "construct_robot.nodes")
        ), name


def test_taught_endpoints_build_complete_scenario_in_slot_order():
    plain = dict(custom_hot_start_enabled=False, software_crater_enabled=False)
    result = build_weld_scenario(
        scenario(settings=plain), MOTION, base_slot=3, scenario_id="weld-1",
    )
    steps = result.steps
    assert stages(steps) == [
        "start_wait", "start_contact", "touch_output_off", "arc_on",
        "weld_motion", "arc_off", "goal_wait", "finish", "touch_output_on",
    ]
    assert [step["parallel_slot"] for step in steps] == [3, 4, 5, 6, 6, 6, 7, 8, 9]
    assert all(step["weld_scenario_id"] == "weld-1" for step in steps)
    assert all(step["weld_approach_mode"] == "corner_geometry" for step in steps)
    assert validate_managed_weld_sequence(steps, require_complete=True)
    motion = steps[4]
    assert motion["path_kind"] == (
        "continuous_taught_start_to_taught_goal_weld_fastech_di0_ignored"
    )
    assert len(motion["points"]) == 2
    assert motion["tcp_speed_m_s"] == pytest.approx(0.006)
    assert motion["fixed_world_y_tilt_deg"] == 15.0
    assert steps[2] == {
        "type": "digital_output", "io_backend": "fastech_ethernet", "port": 0,
        "value": False, "parallel_slot": 5, "duration": 0.0,
        "weld_scenario_id": "weld-1", "weld_scenario_stage": "touch_output_off",
        "weld_approach_mode": "corner_geometry",
    }
    # Approach/exit rows use scale mode; only the weld keeps its seam speed.
    for step in steps:
        if step["type"] in ("motion", "named_pose"):
            assert step["velocity_scale"] == pytest.approx(0.35)
            if step["weld_scenario_stage"] != "weld_motion":
                assert step["tcp_speed_m_s"] == 0.0


def test_sensed_corner_geometry_adds_safe_approach_and_safe_lead_in():
    result = build_weld_scenario(
        scenario(sensed=True, lead_in_mm=5.0, lead_out_mm=3.0, pre_start_lead_mm=2.0),
        MOTION, scenario_id="weld-2",
    )
    by_stage = {step["weld_scenario_stage"]: step for step in result.steps
                if "weld_scenario_stage" in step}
    assert result.approach.safe_approach is not None
    assert by_stage["start_safe"]["role"] == "safe_approach"
    assert by_stage["start_contact"]["points"][0] is not result.approach.approach_lead
    lead_in = next(step for step in result.steps if step.get("role") == "lead_in")
    assert lead_in["safe_retract_geometry"] is True
    assert lead_in["safe_approach_direction"] == GEOMETRY.e_a
    assert lead_in["related_weld_scenario_id"] == "weld-2"
    assert len(result.path.preview) == len(result.path.usable_weld_points) + 2


def test_weave_with_dwell_uses_hold_profile_and_path_speed():
    result = build_weld_scenario(
        scenario(weave_enabled=True, weave_left_dwell_s=0.2, weave_right_dwell_s=0.1),
        MOTION, scenario_id="weld-3",
    )
    motion = next(step for step in result.steps
                  if step["weld_scenario_stage"] == "weld_motion")
    assert motion["linear_motion_profile"] is False
    assert motion["waypoint_hold_s"] == result.path.weave_holds
    assert motion["tcp_speed_m_s"] == pytest.approx(weave_path_speed_m_s(
        result.path.seam_distance_m, result.path.usable_path_distance_m,
        6.0, result.path.weave_holds,
    ))
    arc_off = next(step for step in result.steps
                   if step["weld_scenario_stage"] == "arc_off")
    assert arc_off["tcp_speed_m_s"] == motion["tcp_speed_m_s"]


def test_hot_start_and_software_crater_shift_weld_and_arc_off_slots():
    slots = allocate_weld_scenario_slots(
        1, has_safe_approach=True, has_lead_in=True,
        settings={"custom_hot_start_enabled": True, "software_crater_enabled": True},
    )
    assert (slots.contact, slots.lead_in, slots.arc_on, slots.weld,
            slots.arc_off, slots.final) == (3, 5, 6, 8, 10, 13)
    with pytest.raises(ValueError, match="Not enough free sequence slots"):
        allocate_weld_scenario_slots(
            995, has_safe_approach=False, has_lead_in=False,
            settings={"custom_hot_start_enabled": False, "software_crater_enabled": False},
        )


def test_taught_wait_mode_rewrites_approach():
    steps = build_weld_scenario(
        scenario(approach_mode="taught_wait"), MOTION, scenario_id="weld-4",
    ).steps
    assert stages(steps)[:3] == ["start_wait", "start_safe", "start_contact"]
    assert all(step["weld_approach_mode"] == "taught_wait" for step in steps)


@pytest.mark.parametrize("overrides,message", [
    (dict(lead_in_mm=101.0), "weld lead-in must be in 0..100 mm"),
    (dict(weave_pattern="zigzag"), "weld weave pattern must be sine, crescent or circle"),
    (dict(approach_mode="other"), "Unsupported weld approach mode"),
])
def test_invalid_values_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        plan_weld_approach(scenario(**overrides))


def test_sensed_corner_start_requires_seam_frame():
    with pytest.raises(ValueError, match="no computed seam local frame"):
        plan_weld_approach(scenario(sensed=True, seam_geometry=None))


def test_required_pose_validation_and_endpoint_selection():
    poses = teaching()
    del poses["weld_end"]
    with pytest.raises(ValueError, match="first"):
        validate_required_weld_poses(poses)
    poses = teaching()
    poses["weld_finish"] = stored(pose(0.7, 0.1, 0.4), "left_manipulator")
    with pytest.raises(ValueError, match="must belong to the right arm"):
        validate_required_weld_poses(poses)

    sensed = {"start": pose(0.5, 0.25, 0.1), "goal": pose(0.7, 0.25, 0.1)}
    endpoints = resolve_weld_endpoints(
        teaching(), start_is_sensed=True, goal_is_sensed=False,
        sensed_endpoints=sensed,
    )
    assert endpoints.start is sensed["start"]
    assert (endpoints.start_path_name, endpoints.goal_path_name) == (
        "sensed_start", "taught_goal",
    )


def test_gui_build_delegates_and_updates_sequence_model():
    """The Tk method only snapshots inputs, publishes, and extends the model."""
    from construct_robot.core.sequence_model import SequenceModel
    from construct_robot.gui.weld_action_gui import WeldActionGui

    class Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    gui = object.__new__(WeldActionGui)
    gui.sequence_model = SequenceModel()
    gui.taught_robot_poses = teaching()
    gui.seam_probe_touches = {}
    gui.computed_seam_endpoints = {}
    gui.computed_seam_wait_points = {}
    gui.corrected_seam_geometry = None
    gui._digital_weld_settings = lambda: validate_digital_weld_settings(
        dict(DEFAULT_DIGITAL_WELD_SETTINGS)
    )
    values = dict(
        corner_touch_count=5, weld_lead_in_mm=0.0, weld_lead_out_mm=0.0,
        weld_safe_approach_mm=20.0, weld_pre_start_lead_mm=0.0,
        weld_arc_off_delay_ms=120.0, weld_tcp_speed_mm_s=6.0,
        weld_weave_enabled=False, weave_pattern="sine", weave_amplitude_mm=2.0,
        weave_pitch_mm=4.0, weave_left_dwell_s=0.0, weave_right_dwell_s=0.0,
        weave_axis="tool_y", weld_approach_mode="corner_geometry", show_path=True,
        sequence_parallel_slot=1, velocity_percent=35.0, speed_mode="scale",
        tcp_speed_mm_s=12.0, interpolation_step_mm=2.0,
        seam_orientation_mode=MOTION.seam_orientation_mode,
        weld_fixed_tilt_x_deg=0.0, weld_fixed_tilt_y_deg=15.0,
        weld_fixed_tilt_z_deg=-5.0,
    )
    for name, value in values.items():
        setattr(gui, name, Var(value))
    published, errors, logs = [], [], []
    gui.node = SimpleNamespace(publish_points=lambda points, visible: published.append(points))
    gui.refresh_sequence_table = lambda **_kwargs: None
    gui.error = errors.append
    gui.log = logs.append

    steps = gui.build_sensed_weld_sequence()

    assert not errors
    assert gui.sequence_model.steps == steps
    assert len(published) == 1
    expected = build_weld_scenario(
        scenario(), MOTION, scenario_id=steps[0]["weld_scenario_id"],
    ).steps
    assert copy.deepcopy(steps) == expected
    assert logs and logs[-1].startswith("Built weld workflow from")
