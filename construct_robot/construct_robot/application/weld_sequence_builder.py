"""Build the managed weld-scenario Sequence Builder steps without Tk.

The Tk GUI snapshots its variables into :class:`WeldScenarioInput` and
:class:`WeldStepMotionInput`; this module turns them into the existing
sequence-step dictionaries (schema unchanged) for ``SequenceModel``.

Generation is staged so the caller can keep its operator-visible side effects
(debug logs, seam-geometry annotation, RViz preview) exactly where they
happened before the extraction:

1. :func:`validate_required_weld_poses` / :func:`resolve_weld_endpoints`
2. :func:`plan_weld_approach`   -- value validation, lead and safe approach
3. :func:`plan_weld_path`       -- centerline / weave, speed factor, preview
4. :func:`allocate_weld_scenario_slots`
5. :func:`build_weld_scenario_steps`

:func:`build_weld_scenario` runs all stages for callers without side effects.
"""

import copy
from dataclasses import dataclass
import math
import time

from construct_robot.core.cartesian_path_common import (
    WELD_WEAVE_SAMPLES_PER_CYCLE,
    linear_pose_waypoints,
    validated_seam_speed_factor,
    weave_path_speed_m_s,
    weld_weave_geometry,
)
from construct_robot.core.seam_geometry import (
    _unit_vector,
    _vector_dot,
    compute_safe_weld_approach,
    seam_lead_poses,
)
from construct_robot.core.sequence_model import (
    taught_wait_approach_steps,
    validate_managed_weld_sequence,
)
from construct_robot.core.task_teaching_model import TEACHING_POSES


@dataclass(frozen=True)
class WeldEndpoints:
    """Teaching entries and the seam START/GOAL chosen for one build.

    Teaching entries keep the stored ``(group, joint_names, positions, tcp)``
    tuple shape.  A sensed endpoint is the caller's computed pose object
    itself (not a copy), as before the extraction.
    """

    start_wait: tuple
    start_is_sensed: bool
    start: object
    start_path_name: str
    start_source: str
    goal: object
    goal_path_name: str
    goal_source: str


@dataclass(frozen=True)
class WeldScenarioInput:
    """Snapshot of every Build input except the per-step motion settings."""

    endpoints: WeldEndpoints
    goal_wait: tuple
    finish: tuple
    settings: dict
    corner_touch_count: int
    lead_in_mm: float
    lead_out_mm: float
    safe_approach_mm: float
    pre_start_lead_mm: float
    arc_off_delay_ms: float
    weld_tcp_speed_mm_s: float
    weave_enabled: bool
    weave_pattern: str
    weave_amplitude_mm: float
    weave_pitch_mm: float
    weave_left_dwell_s: float
    weave_right_dwell_s: float
    weave_axis: str
    approach_mode: str
    touch_io_backend: str
    touch_output_port: int
    # Touch-corrected seam frame (needs ``d_real`` and ``e_a``) or None.
    seam_geometry: object = None
    # Geometry-derived weave direction ``e_w`` when weaving a sensed seam.
    weave_transverse_vector: tuple = None
    # Corrected START WAIT TCP, used by the non-sensed lead-in retract.
    start_wait_tcp_override: object = None


@dataclass(frozen=True)
class WeldStepMotionInput:
    """Shared motion settings stamped onto generated steps at Build time."""

    velocity_percent: float
    tcp_speed_m_s: float
    interpolation_step_mm: float
    seam_orientation_mode: str
    fixed_world_x_tilt_deg: float
    fixed_world_y_tilt_deg: float
    fixed_world_z_tilt_deg: float


@dataclass(frozen=True)
class WeldApproachPlan:
    lead_start: object
    lead_end: object
    has_lead_in: bool
    has_lead_out: bool
    safe_approach: object
    approach_lead: object
    approach_alignment: float = None
    lead_alignment: float = None


@dataclass(frozen=True)
class WeldPathPlan:
    usable_weld_points: object
    weave_holds: list
    weave_cycles: int
    weave_actual_pitch_mm: float
    weave_crescent_bulge_mm: float
    weave_direction: tuple
    seam_distance_m: float
    usable_path_distance_m: float
    path_to_seam_speed_factor: float
    preview: list


@dataclass(frozen=True)
class WeldScenarioSlots:
    base: int
    contact: int
    touch_output_off: int
    lead_in: int
    arc_on: int
    weld: int
    arc_off: int
    goal_wait: int
    finish: int
    final: int


@dataclass(frozen=True)
class WeldScenarioBuild:
    approach: WeldApproachPlan
    path: WeldPathPlan
    slots: WeldScenarioSlots
    steps: list


def validate_required_weld_poses(poses):
    """Require right-arm weld goal, goal-wait and end teaching."""
    goal_data = poses.get("weld_end")
    goal_wait_data = poses.get("weld_goal_wait")
    finish_data = poses.get("weld_finish")
    if goal_data is None or goal_wait_data is None or finish_data is None:
        raise ValueError(
            f"Capture/load {TEACHING_POSES['weld_end']} and "
            f"{TEACHING_POSES['weld_goal_wait']} and "
            f"{TEACHING_POSES['weld_finish']} first"
        )
    if (
        goal_data[0] != "right_manipulator"
        or goal_wait_data[0] != "right_manipulator"
        or finish_data[0] != "right_manipulator"
    ):
        raise ValueError(
            "Weld goal, goal-wait, and end poses must belong to the right arm"
        )


def resolve_weld_endpoints(
    poses, *, start_is_sensed, goal_is_sensed, sensed_endpoints,
):
    """Select touch-sensed or taught START/GOAL and validate START WAIT."""
    goal_data = poses.get("weld_end")
    start_wait_data = poses["weld_start_wait"]
    if start_wait_data is None:
        raise ValueError(
            f"Capture/load {TEACHING_POSES['weld_start_wait']} first"
        )
    start_wait_group = start_wait_data[0]
    if start_wait_group != "right_manipulator":
        raise ValueError("Weld start-wait pose must belong to the right arm")
    if start_is_sensed:
        start = sensed_endpoints["start"]
        start_path_name = "sensed_start"
        start_source = "sensed START"
    else:
        start_data = poses.get("weld_start")
        if start_data is None:
            raise ValueError(
                f"Capture/load {TEACHING_POSES['weld_start']} first"
            )
        if start_data[0] != "right_manipulator":
            raise ValueError("Weld start pose must belong to the right arm")
        start = copy.deepcopy(start_data[3])
        start_path_name = "taught_start"
        start_source = TEACHING_POSES["weld_start"]
    if goal_is_sensed:
        goal = sensed_endpoints["goal"]
        goal_source = "sensed GOAL"
        goal_path_name = "sensed_goal"
    else:
        goal = copy.deepcopy(goal_data[3])
        goal_source = TEACHING_POSES["weld_end"]
        goal_path_name = "taught_goal"
    return WeldEndpoints(
        start_wait=start_wait_data,
        start_is_sensed=bool(start_is_sensed),
        start=start,
        start_path_name=start_path_name,
        start_source=start_source,
        goal=goal,
        goal_path_name=goal_path_name,
        goal_source=goal_source,
    )


def validate_weld_scenario_values(inp):
    """Range-check the operator's weld, lead, approach and weave values."""
    weld_tcp_speed_mm_s = inp.weld_tcp_speed_mm_s
    lead_in_mm = inp.lead_in_mm
    lead_out_mm = inp.lead_out_mm
    safe_approach_mm = inp.safe_approach_mm
    pre_start_lead_mm = inp.pre_start_lead_mm
    arc_off_delay_ms = inp.arc_off_delay_ms
    weave_pattern = inp.weave_pattern
    weave_amplitude_mm = inp.weave_amplitude_mm
    weave_pitch_mm = inp.weave_pitch_mm
    weave_left_dwell_s = inp.weave_left_dwell_s
    weave_right_dwell_s = inp.weave_right_dwell_s
    if (
        not math.isfinite(weld_tcp_speed_mm_s)
        or not 0.1 <= weld_tcp_speed_mm_s <= 100.0
    ):
        raise ValueError("weld TCP speed must be in 0.1..100 mm/s")
    if not 0.0 <= lead_in_mm <= 100.0:
        raise ValueError("weld lead-in must be in 0..100 mm")
    if not 0.0 <= lead_out_mm <= 100.0:
        raise ValueError("weld lead-out must be in 0..100 mm")
    if not 1.0 <= safe_approach_mm <= 200.0:
        raise ValueError("safe approach distance must be in 1..200 mm")
    if not 0.0 <= pre_start_lead_mm <= 100.0:
        raise ValueError("pre-start lead distance must be in 0..100 mm")
    if not 0.0 <= arc_off_delay_ms <= 2000.0:
        raise ValueError("ARC OFF lead time must be in 0..2000 ms")
    if weave_pattern not in ("sine", "crescent", "circle"):
        raise ValueError("weld weave pattern must be sine, crescent or circle")
    if not 0.1 <= weave_amplitude_mm <= 50.0:
        raise ValueError("weld weave amplitude/radius must be 0.1..50 mm")
    if not math.isfinite(weave_pitch_mm) or not 0.1 <= weave_pitch_mm <= 100.0:
        raise ValueError("weld weave pitch must be in 0.1..100 mm/cycle")
    if not all(math.isfinite(v) and 0.0 <= v <= 10.0 for v in
               (weave_left_dwell_s, weave_right_dwell_s)):
        raise ValueError("weave left/right dwell must be in 0..10 s")


def plan_weld_approach(inp):
    """Validate values, then derive lead poses and the corner safe approach."""
    validate_weld_scenario_values(inp)
    start = inp.endpoints.start
    goal = inp.endpoints.goal
    lead_in_mm = inp.lead_in_mm
    lead_out_mm = inp.lead_out_mm
    safe_approach_mm = inp.safe_approach_mm
    pre_start_lead_mm = inp.pre_start_lead_mm
    # Welding orientation is already finalized by seam correction.
    # In fixed-tilt mode it comes from WAIT + one World-XYZ RPY offset; in the
    # legacy modes it comes from the endpoint teaching.  Never apply a
    # second offset here, because probing and welding must use exactly
    # the same tool attitude.
    # lead는 weld motion의 시작과 끝에서 ARC를 켜고 끄는 지점을 결정하는데 사용됩니다.
    lead_start, lead_end = seam_lead_poses(
        start,
        goal,
        lead_in_mm * 0.001,
        lead_out_mm * 0.001,
    )
    has_lead_in = lead_in_mm > 1e-6
    has_lead_out = lead_out_mm > 1e-6
    safe_approach = None
    approach_lead = None
    approach_alignment = None
    lead_alignment = None
    approach_mode = inp.approach_mode
    start_is_sensed = inp.endpoints.start_is_sensed
    if approach_mode not in ("taught_wait", "corner_geometry"):
        raise ValueError("Unsupported weld approach mode")
    if start_is_sensed and approach_mode == "corner_geometry":
        if inp.seam_geometry is None:
            raise ValueError(
                "touch-corrected START has no computed seam local frame"
            )
        safe_approach, approach_lead = compute_safe_weld_approach(
            start,
            inp.seam_geometry.d_real,
            inp.seam_geometry.e_a,
            safe_approach_mm * 0.001,
            pre_start_lead_mm * 0.001,
        )
        approach_vector = _unit_vector(tuple(
            getattr(approach_lead.position, axis)
            - getattr(safe_approach.position, axis)
            for axis in ("x", "y", "z")
        ))
        approach_alignment = _vector_dot(
            approach_vector,
            tuple(-value for value in inp.seam_geometry.e_a),
        )
        lead_alignment = 1.0
        if pre_start_lead_mm > 1e-6:
            lead_vector = _unit_vector(tuple(
                getattr(start.position, axis)
                - getattr(approach_lead.position, axis)
                for axis in ("x", "y", "z")
            ))
            lead_alignment = _vector_dot(
                lead_vector, inp.seam_geometry.d_real
            )
    return WeldApproachPlan(
        lead_start=lead_start,
        lead_end=lead_end,
        has_lead_in=has_lead_in,
        has_lead_out=has_lead_out,
        safe_approach=safe_approach,
        approach_lead=approach_lead,
        approach_alignment=approach_alignment,
        lead_alignment=lead_alignment,
    )


def plan_weld_path(inp, approach):
    """Build the usable weld path (centerline or weave) and RViz preview."""
    start = inp.endpoints.start
    goal = inp.endpoints.goal
    weave_enabled = inp.weave_enabled
    weave_pattern = inp.weave_pattern
    weave_amplitude_mm = inp.weave_amplitude_mm
    seam_centerline = linear_pose_waypoints(start, goal, inp.corner_touch_count)
    weave_holds = []
    weave_cycles = 0
    weave_actual_pitch_mm = 0.0
    weave_crescent_bulge_mm = 0.0
    geometry_weave_direction = None
    if weave_enabled:
        # Same helper the weave preview calls, so an approved preview
        # and the executed weld cannot disagree about the weave plane.
        geometry_weave_direction = inp.weave_transverse_vector
        (usable_weld_points, weave_holds, weave_cycles,
         weave_actual_pitch_mm) = weld_weave_geometry(
            start, goal, weave_pattern, weave_amplitude_mm,
            inp.weave_pitch_mm, inp.weave_axis, inp.weave_left_dwell_s,
            inp.weave_right_dwell_s, geometry_weave_direction,
        )
        if weave_pattern == "crescent":
            weave_crescent_bulge_mm = min(
                0.5 * weave_amplitude_mm,
                weave_actual_pitch_mm / (4.0 * math.pi),
            )
    else:
        usable_weld_points = seam_centerline
    seam_distance_m = math.sqrt(sum(
        (getattr(goal.position, axis) - getattr(start.position, axis)) ** 2
        for axis in ("x", "y", "z")
    ))
    usable_path_distance_m = sum(
        math.sqrt(sum(
            (getattr(second.position, axis) - getattr(first.position, axis)) ** 2
            for axis in ("x", "y", "z")
        ))
        for first, second in zip(
            usable_weld_points[:-1], usable_weld_points[1:]
        )
    )
    path_to_seam_speed_factor = (
        validated_seam_speed_factor(
            seam_distance_m / usable_path_distance_m
        ) if weave_enabled else 1.0
    )
    preview = []
    if approach.has_lead_in:
        preview.append(copy.deepcopy(approach.lead_start))
    preview.extend(copy.deepcopy(usable_weld_points))
    if approach.has_lead_out:
        preview.append(copy.deepcopy(approach.lead_end))
    return WeldPathPlan(
        usable_weld_points=usable_weld_points,
        weave_holds=weave_holds,
        weave_cycles=weave_cycles,
        weave_actual_pitch_mm=weave_actual_pitch_mm,
        weave_crescent_bulge_mm=weave_crescent_bulge_mm,
        weave_direction=geometry_weave_direction,
        seam_distance_m=seam_distance_m,
        usable_path_distance_m=usable_path_distance_m,
        path_to_seam_speed_factor=path_to_seam_speed_factor,
        preview=preview,
    )


def allocate_weld_scenario_slots(base_slot, *, has_safe_approach, has_lead_in, settings):
    """Assign the parallel slots of one weld scenario starting at base_slot."""
    safe_slot_count = 1 if has_safe_approach else 0
    contact_slot = base_slot + 1 + safe_slot_count
    touch_output_off_slot = contact_slot + 1
    lead_in_slot = touch_output_off_slot + 1
    arc_on_slot = touch_output_off_slot + 1 + (1 if has_lead_in else 0)
    custom_hot_start_enabled = bool(settings["custom_hot_start_enabled"])
    weld_slot = arc_on_slot + (2 if custom_hot_start_enabled else 0)
    software_crater_enabled = bool(settings["software_crater_enabled"])
    arc_off_slot = weld_slot + 2 if software_crater_enabled else weld_slot
    goal_wait_slot = arc_off_slot + 1
    finish_slot = goal_wait_slot + 1
    final_slot = finish_slot + 1
    if final_slot > 999:
        raise ValueError(
            "Not enough free sequence slots for weld scenario"
        )
    return WeldScenarioSlots(
        base=base_slot,
        contact=contact_slot,
        touch_output_off=touch_output_off_slot,
        lead_in=lead_in_slot,
        arc_on=arc_on_slot,
        weld=weld_slot,
        arc_off=arc_off_slot,
        goal_wait=goal_wait_slot,
        finish=finish_slot,
        final=final_slot,
    )


def new_weld_scenario_id():
    return f"weld-{time.monotonic_ns()}"


def sensed_motion_step(points, label, slot, motion, touch_guard=False):
    """One right-arm Cartesian motion row stamped with the Build motion settings."""
    return {
        "type": "motion",
        "planning_group": "right_manipulator",
        "points": copy.deepcopy(points),
        "velocity_scale": max(
            0.01, min(1.0, motion.velocity_percent / 100.0)
        ),
        "tcp_speed_m_s": motion.tcp_speed_m_s,
        "interpolation_step": max(
            0.0005,
            min(0.02, motion.interpolation_step_mm * 0.001),
        ),
        "path_kind": label,
        "parallel_slot": slot,
        "duration": 0.0,
        "touch_guard": bool(touch_guard),
    }


def build_weld_scenario_steps(inp, approach, path, slots, motion, scenario_id):
    """Generate the managed weld-scenario steps and validate the result."""
    settings = inp.settings
    start_wait_data = inp.endpoints.start_wait
    goal_wait_data = inp.goal_wait
    finish_data = inp.finish
    start = inp.endpoints.start
    goal = inp.endpoints.goal
    start_path_name = inp.endpoints.start_path_name
    goal_path_name = inp.endpoints.goal_path_name
    lead_in_mm = inp.lead_in_mm
    lead_out_mm = inp.lead_out_mm
    safe_approach_mm = inp.safe_approach_mm
    pre_start_lead_mm = inp.pre_start_lead_mm
    arc_off_delay_ms = inp.arc_off_delay_ms
    weld_tcp_speed_mm_s = inp.weld_tcp_speed_mm_s
    weave_enabled = inp.weave_enabled
    weave_pattern = inp.weave_pattern
    approach_mode = inp.approach_mode
    lead_start = approach.lead_start
    lead_end = approach.lead_end
    has_lead_in = approach.has_lead_in
    has_lead_out = approach.has_lead_out
    safe_approach = approach.safe_approach
    approach_lead = approach.approach_lead
    usable_weld_points = path.usable_weld_points
    weave_holds = path.weave_holds
    path_to_seam_speed_factor = path.path_to_seam_speed_factor
    base_slot = slots.base
    custom_hot_start_enabled = bool(settings["custom_hot_start_enabled"])
    software_crater_enabled = bool(settings["software_crater_enabled"])

    def managed(step, stage):
        step["weld_scenario_id"] = scenario_id
        step["weld_scenario_stage"] = stage
        return step

    def named_step(name, stored, slot, stage):
        group, names, joints, tcp = stored
        return managed({
            "type": "named_pose",
            "pose_name": name,
            "pose_label": TEACHING_POSES[name],
            "planning_group": group,
            "joint_names": tuple(names),
            "positions": tuple(joints),
            "tcp_pose": copy.deepcopy(tcp),
            "velocity_scale": max(
                0.01,
                min(1.0, motion.velocity_percent / 100.0),
            ),
            "tcp_speed_m_s": motion.tcp_speed_m_s,
            "parallel_slot": slot,
            "duration": 0.0,
            "touch_guard": False,
            "continue_after_touch": False,
        }, stage)

    near_approach_points = (
        (approach_lead, start)
        if approach_lead is not None and pre_start_lead_mm > 1e-6
        else (start,)
    )
    approach_start = sensed_motion_step(
        near_approach_points,
        f"safe_to_pre_start_to_{start_path_name}_fastech_di0"
        if safe_approach is not None
        else f"start_wait_to_{start_path_name}_fastech_di0",
        slots.contact,
        motion,
        touch_guard=False,
    )
    approach_start["continue_after_touch"] = True
    # A stale/high Fastech DI0 at START WAIT must never skip directly to ARC.
    # Require a fresh rising edge, confirm standstill, then continue.
    approach_start["accept_initial_touch"] = False
    approach_start.update({
        "role": "approach",
    })
    approach_start.update({
        "safe_approach_mm": safe_approach_mm,
        "pre_start_lead_mm": pre_start_lead_mm,
        "safe_approach": copy.deepcopy(safe_approach),
        "approach_lead": copy.deepcopy(approach_lead),
        "collision_checking": True,
    })
    steps = [named_step(
        "weld_start_wait", start_wait_data, base_slot, "start_wait"
    )]
    if safe_approach is not None:
        safe_motion = sensed_motion_step(
            (safe_approach,),
            f"{start_path_name}_safe_approach_collision_checked",
            base_slot + 1,
            motion,
            touch_guard=False,
        )
        safe_motion.update({
            "role": "safe_approach",
            "safe_approach_mm": safe_approach_mm,
            "pre_start_lead_mm": pre_start_lead_mm,
            "safe_approach": copy.deepcopy(safe_approach),
            "approach_lead": copy.deepcopy(approach_lead),
            "collision_checking": True,
        })
        steps.append(managed(safe_motion, "start_safe"))
    steps.extend([
        managed(approach_start, "start_contact"),
        managed({
            "type": "digital_output",
            "io_backend": inp.touch_io_backend,
            "port": inp.touch_output_port,
            "value": False,
            "parallel_slot": slots.touch_output_off,
            "duration": 0.0,
        }, "touch_output_off"),
    ])

    if has_lead_in:
        if safe_approach is not None:
            safe_over_start, _unused_start = compute_safe_weld_approach(
                start,
                inp.seam_geometry.d_real,
                inp.seam_geometry.e_a,
                safe_approach_mm * 0.001,
                0.0,
            )
            safe_over_weld_lead, computed_weld_lead = (
                compute_safe_weld_approach(
                    start,
                    inp.seam_geometry.d_real,
                    inp.seam_geometry.e_a,
                    safe_approach_mm * 0.001,
                    lead_in_mm * 0.001,
                )
            )
            lead_position_points = (
                safe_over_start,
                safe_over_weld_lead,
                computed_weld_lead,
            )
            lead_position_label = (
                f"{start_path_name}_lift_translate_descend_to_"
                "weld_lead_in_arc_off"
            )
            retraction_reference = safe_over_start
        else:
            # Legacy/non-sensed path retains the taught WAIT clearance.
            start_wait_tcp = copy.deepcopy(
                inp.start_wait_tcp_override
                or start_wait_data[3]
            )
            start_wait_tcp.orientation = copy.deepcopy(start.orientation)
            lead_position_points = (start_wait_tcp, lead_start)
            lead_position_label = (
                f"{start_path_name}_retract_via_wait_to_"
                "weld_lead_in_arc_off"
            )
            retraction_reference = start_wait_tcp
        lead_position = sensed_motion_step(
            lead_position_points,
            lead_position_label,
            slots.lead_in,
            motion,
        )
        lead_position["lead_in_mm"] = lead_in_mm
        lead_position["lead_out_mm"] = lead_out_mm
        lead_position["lead_start"] = copy.deepcopy(lead_start)
        lead_position.update({
            "role": "lead_in",
            "related_weld_scenario_id": scenario_id,
            "start_wait_tcp": copy.deepcopy(retraction_reference),
            "collision_checking": True,
            "safe_retract_geometry": safe_approach is not None,
            "safe_approach_mm": safe_approach_mm,
            "safe_approach_direction": (
                tuple(inp.seam_geometry.e_a)
                if safe_approach is not None
                else None
            ),
        })
        steps.append(lead_position)

    if weave_enabled:
        weld_points = []
        if has_lead_in:
            weld_points.append(copy.deepcopy(lead_start))
        weld_points.extend(copy.deepcopy(usable_weld_points))
        if has_lead_out:
            weld_points.append(copy.deepcopy(lead_end))
        weld_points = tuple(weld_points)
    else:
        # A non-weaving weld stays one endpoint-to-endpoint segment;
        # START/GOAL are logical ARC landmarks, not timing waypoints.
        motion_start = copy.deepcopy(
            lead_start if has_lead_in else start
        )
        motion_end = copy.deepcopy(
            lead_end if has_lead_out else goal
        )
        weld_points = (motion_start, motion_end)

    weld_motion = sensed_motion_step(
        weld_points,
        (
            f"continuous_{weave_pattern}_weave_over_{goal_path_name}_fastech_di0_ignored"
            if weave_enabled
            else f"continuous_lead_to_lead_over_{goal_path_name}_fastech_di0_ignored"
            if has_lead_in or has_lead_out
            else f"continuous_{start_path_name}_to_{goal_path_name}_weld_fastech_di0_ignored"
        ),
        slots.weld,
        motion,
    )
    weld_motion.update({
        "lead_in_mm": lead_in_mm,
        "lead_out_mm": lead_out_mm,
        "record_tcp_trajectory": True,
        "lead_start": copy.deepcopy(lead_start),
        "usable_seam_start": copy.deepcopy(start),
        "usable_seam_goal": copy.deepcopy(goal),
        "lead_end": copy.deepcopy(lead_end),
        # Convert the operator's seam-axis travel target to TCP path
        # speed for weaving below. The action server still enforces
        # MoveIt's trajectory limits and lead-in/out ramps.
        "tcp_speed_m_s": weld_tcp_speed_mm_s * 0.001,
        "linear_motion_profile": True,
        "weld_tcp_speed_mm_s": weld_tcp_speed_mm_s,
        "weld_weave_enabled": weave_enabled,
        "weld_weave_pattern": weave_pattern,
        "weld_weave_amplitude_mm": inp.weave_amplitude_mm,
        "weld_weave_pitch_mm": inp.weave_pitch_mm,
        "weld_weave_cycles": path.weave_cycles,
        "weld_weave_actual_pitch_mm": path.weave_actual_pitch_mm,
        "weld_weave_crescent_bulge_mm": path.weave_crescent_bulge_mm,
        "weld_weave_samples_per_cycle": WELD_WEAVE_SAMPLES_PER_CYCLE,
        "weld_weave_left_dwell_s": inp.weave_left_dwell_s,
        "weld_weave_right_dwell_s": inp.weave_right_dwell_s,
        "weld_weave_axis": inp.weave_axis,
        "usable_weld_points": copy.deepcopy(usable_weld_points),
        "weld_weave_transverse_vector": path.weave_direction if weave_enabled else None,
        "role": "weld_motion",
        "seam_orientation_mode": motion.seam_orientation_mode,
        "fixed_world_x_tilt_deg": motion.fixed_world_x_tilt_deg,
        "fixed_world_y_tilt_deg": motion.fixed_world_y_tilt_deg,
        "fixed_world_z_tilt_deg": motion.fixed_world_z_tilt_deg,
        "safe_approach_mm": safe_approach_mm,
        "pre_start_lead_mm": pre_start_lead_mm,
    })

    if weave_enabled:
        weld_motion["path_to_seam_speed_factor"] = path_to_seam_speed_factor
        weld_motion["tcp_speed_m_s"] = weave_path_speed_m_s(
            path.seam_distance_m, path.usable_path_distance_m,
            weld_tcp_speed_mm_s, weave_holds,
        )
    if any(weave_holds):
        weld_motion["waypoint_hold_s"] = (
            ([0.0] if has_lead_in else []) + weave_holds
            + ([0.0] if has_lead_out else [])
        )
        weld_motion["linear_motion_profile"] = False

    weld_steps = [
        managed({
            "type": "digital_weld", "command": "on",
            "settings": copy.deepcopy(settings),
            "parallel_slot": slots.arc_on, "duration": 0.0,
        }, "arc_on"),
    ]
    if custom_hot_start_enabled:
        weld_steps.append(managed({
            "type": "custom_hot_start",
            "settings": copy.deepcopy(settings),
            "planning_group": weld_motion["planning_group"],
            "expected_start_tcp": copy.deepcopy(weld_points[0]),
            "start_pose_role": ("lead_start" if has_lead_in else "sensed_start"),
            "parallel_slot": slots.arc_on + 1,
            "duration": 0.0,
        }, "custom_hot_start"))
    weld_steps.append(managed(weld_motion, "weld_motion"))
    if software_crater_enabled:
        weld_steps.append(managed({
            "type": "software_crater",
            "settings": copy.deepcopy(settings),
            "endpoint": copy.deepcopy(weld_points[-1]),
            "planning_group": weld_motion.get("planning_group", "right_manipulator"),
            "parallel_slot": slots.weld + 1,
        }, "software_crater"))
    weld_steps.extend([
        managed({
            "type": "digital_weld", "command": "off",
            "settings": copy.deepcopy(settings),
            "parallel_slot": slots.arc_off, "duration": 0.0,
            "trigger_before_goal": not software_crater_enabled,
            "usable_seam_start": copy.deepcopy(start),
            "usable_seam_goal": copy.deepcopy(goal),
            "arc_off_delay_s": arc_off_delay_ms * 0.001,
            "tcp_speed_m_s": float(weld_motion.get("tcp_speed_m_s", 0.0)),
            "velocity_scale": float(weld_motion.get("velocity_scale", 0.0)),
            "path_to_seam_speed_factor": path_to_seam_speed_factor,
        }, "arc_off"),
        named_step(
            "weld_goal_wait",
            goal_wait_data,
            slots.goal_wait,
            "goal_wait",
        ),
        named_step(
            "weld_finish", finish_data, slots.finish, "finish"
        ),
        managed({
            "type": "digital_output",
            "io_backend": inp.touch_io_backend,
            "port": inp.touch_output_port,
            "value": True,
            "parallel_slot": slots.final,
            "duration": 0.0,
        }, "touch_output_on"),
    ])
    steps.extend(weld_steps)
    if approach_mode == "taught_wait":
        steps = taught_wait_approach_steps(steps, start_wait_data[3], goal_wait_data[3])
    for step in steps:
        step["weld_approach_mode"] = approach_mode
        if step.get("type") in ("motion", "named_pose"):
            # Snapshot the shared slider at Build. Welding retains its
            # physical seam-travel target; approach/exit use scale mode.
            step["velocity_scale"] = max(
                0.01, min(1.0, float(motion.velocity_percent) / 100.0)
            )
            if step.get("weld_scenario_stage") != "weld_motion":
                step["tcp_speed_m_s"] = 0.0
    validate_managed_weld_sequence(steps, require_complete=True)
    return steps


def build_weld_scenario(inp, motion, *, base_slot=1, scenario_id=None):
    """Run every stage for callers that need no intermediate side effects."""
    approach = plan_weld_approach(inp)
    path = plan_weld_path(inp, approach)
    slots = allocate_weld_scenario_slots(
        base_slot,
        has_safe_approach=approach.safe_approach is not None,
        has_lead_in=approach.has_lead_in,
        settings=inp.settings,
    )
    if scenario_id is None:
        scenario_id = new_weld_scenario_id()
    steps = build_weld_scenario_steps(inp, approach, path, slots, motion, scenario_id)
    return WeldScenarioBuild(approach=approach, path=path, slots=slots, steps=steps)
