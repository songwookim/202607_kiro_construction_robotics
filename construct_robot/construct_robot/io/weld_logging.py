"""Welding-feedback log parsing, report formatting, and atomic persistence."""
import copy
import hashlib
import math
from pathlib import Path

import yaml

from construct_robot.core.cartesian_path_common import pose_is_valid
from construct_robot.core.seam_geometry import _pose_position_tuple
from construct_robot.core.weld_quality_metrics import (
    _number, analyze_weld_quality, format_quality_summary,
)
from construct_robot.io.teaching_yaml import _pose_from_yaml_dict, atomic_text_writer


def weld_weave_settings_text(conditions):
    """Human-readable meaning of the effective scenario weave settings."""
    if not conditions.get("weld_weave_enabled", False):
        return "OFF"
    pattern = str(conditions.get("weld_weave_pattern", "sine"))
    amplitude = float(conditions.get("weld_weave_amplitude_mm", 0.0))
    pitch = float(conditions.get("weld_weave_pitch_mm", 0.0))
    cycles = int(conditions.get("weld_weave_cycles", 0))
    actual_pitch = float(conditions.get("weld_weave_actual_pitch_mm", 0.0))
    if pattern == "circle":
        amplitude_text = (
            f"radius {amplitude:.2f} mm (diameter {2 * amplitude:.2f} mm)"
        )
    else:
        amplitude_text = (
            f"centerline ±{amplitude:.2f} mm "
            f"(full width {2 * amplitude:.2f} mm)"
        )
    crescent_text = (
        f" · forward bulge {float(conditions.get('weld_weave_crescent_bulge_mm') or 0.0):.2f} mm"
        if pattern == "crescent" else ""
    )
    return (
        f"{pattern} · {amplitude_text} · requested pitch {pitch:.2f} mm/cycle "
        f"· actual pitch {actual_pitch:.2f} mm/cycle · {cycles} cycles{crescent_text} · "
        f"axis {conditions.get('weld_weave_axis', 'tool_y')} · "
        f"dwell L/R {float(conditions.get('weld_weave_left_dwell_s', 0.0)):.2f}/"
        f"{float(conditions.get('weld_weave_right_dwell_s', 0.0)):.2f} s"
    )


def format_weld_feedback_log(document):
    """Return a readable, line-oriented weld report including every RX sample."""
    lines = [
        "WELD FEEDBACK LOG",
        f"result={document['result']}",
        f"started={document['started']}",
        f"ended={document['ended']}",
        f"elapsed_seconds={document['elapsed_seconds']:.3f}",
    ]
    lines.extend(("", *format_quality_summary(document)))
    commanded = document["commanded"]
    welding_echo = (
        document.get("rx_welding_setting_echo")
        or document.get("rx_setting_echo")
        or {}
    )
    feedback = document["feedback"]
    production = document.get("production_metrics", {}) or {}

    def statistic_text(values):
        if values.get("average") is None:
            return "no positive feedback"
        return (
            f"avg {values['average']:.2f} · min {values['min']:.2f} · "
            f"max {values['max']:.2f}"
        )

    requested_current = int(commanded.get("current_a", 0))
    requested_voltage = float(commanded.get("voltage", 0.0))
    echo_current = int(welding_echo.get("current_a", 0))
    echo_voltage = float(welding_echo.get("voltage_v", 0.0))
    echo_match = (
        requested_current == echo_current
        and math.isclose(requested_voltage, echo_voltage, abs_tol=0.05)
    )
    lines.extend((
        "",
        "================ OPERATOR OVERVIEW ================",
        f"REQUESTED : {requested_current} A / {requested_voltage:.1f} V · "
        f"{commanded.get('material')} {commanded.get('diameter_mm')} mm · "
        f"{commanded.get('mode')} · {commanded.get('gas')}",
        f"RX ECHO   : {echo_current} A / {echo_voltage:.1f} V · "
        f"MATCH={'YES' if echo_match else 'NO'}",
        f"CURRENT FB: {statistic_text(feedback['current_a'])} A",
        f"VOLTAGE FB: {statistic_text(feedback['voltage_v'])} V",
        f"WIRE FEED : {statistic_text(feedback['wire_feed_m_min'])} m/min",
        f"ARC ON    : {float(production.get('arc_on_time_s', 0.0)):.3f} s",
        f"NET WELD  : {float(production.get('net_weld_arc_time_s', 0.0)):.3f} s",
        f"WIRE USED : {float(production.get('wire_consumable_mm', 0.0)):.3f} mm "
        f"(base {float(production.get('wire_consumable_base_mm', 0.0)):.3f} "
        f"+ alpha {float(production.get('wire_consumable_alpha_mm', 0.0)):.3f})",
        f"WEAVE     : {weld_weave_settings_text(document.get('execution_conditions', {}))}",
        f"WCR       : {'DETECTED' if feedback['wcr_seen'] else 'NOT DETECTED'}",
        f"SAMPLES   : RX {feedback['rx_samples']} / "
        f"welding {feedback['welding_samples']} / "
        f"TCP {len(document.get('tcp_trajectory', ())) }",
        "===================================================",
        "",
        "[commanded]",
    ))
    for key, value in document["commanded"].items():
        lines.append(f"{key}={value}")

    echo = document.get("rx_setting_echo") or {}
    lines.extend(("", "[rx_setting_echo]"))
    for key, value in echo.items():
        lines.append(f"{key}={value}")
    lines.extend(("", "[rx_welding_setting_echo]"))
    for key, value in (
        document.get("rx_welding_setting_echo") or {}
    ).items():
        lines.append(f"{key}={value}")

    def flattened(prefix, value):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                child_prefix = f"{prefix}.{child_key}" if prefix else str(child_key)
                yield from flattened(child_prefix, child_value)
        elif isinstance(value, (list, tuple)):
            for index, child_value in enumerate(value):
                yield from flattened(f"{prefix}[{index}]", child_value)
        else:
            yield prefix, value

    lines.extend(("", "[execution_conditions]"))
    for key, value in flattened("", document.get("execution_conditions", {})):
        lines.append(f"{key}={value}")

    lines.extend((
        "",
        "[summary]",
        f"rx_samples={feedback['rx_samples']}",
        f"welding_samples={feedback['welding_samples']}",
        f"wcr_seen={int(feedback['wcr_seen'])}",
    ))
    for name in ("current_a", "voltage_v", "wire_feed_m_min"):
        values = feedback[name]
        for statistic in ("min", "average", "max"):
            lines.append(f"{name}.{statistic}={values[statistic]}")

    lines.extend(("", "[production_metrics]"))
    for key, value in production.items():
        lines.append(f"{key}={value}")

    lines.extend(("", "[arc_off_control]"))
    for key, value in flattened("", document.get("arc_off_control", {})):
        lines.append(f"{key}={value}")

    lines.extend(("", "[custom_hot_start]"))
    for key, value in flattened("", document.get("custom_hot_start", {})):
        lines.append(f"{key}={value}")

    lines.extend(("", "[software_crater_control]"))
    for key, value in flattened("", document.get("software_crater_control", {})):
        lines.append(f"{key}={value}")

    lines.extend(("", "[quality_metrics]"))
    for key, value in flattened("", document.get("quality_metrics", {})):
        lines.append(f"{key}={value if value is not None else 'N/A'}")
    lines.extend(("", "[event_timeline]"))
    for name, event in (document.get("quality_metrics", {}).get("timeline", {}) or {}).items():
        for field in ("elapsed_s", "unix_time", "wall_time"):
            value = event.get(field)
            lines.append(f"{name}.{field}={value if value is not None else 'N/A'}")
    lines.extend(("", "[tx_frames]", "elapsed_s unix_time raw_hex"))
    for frame in document.get("tx_frames", ()):
        lines.append(
            f"{float(frame['elapsed_s']):.6f} {float(frame['unix_time']):.6f} "
            f"{frame['raw_hex']}"
        )

    # Embedded YAML snapshots of the taught poses/touch points active for
    # this run, so "Load teaching/touch from log" can restore exactly what
    # was on screen when this weld happened -- reusing the same
    # position_m/orientation_xyzw shape as the per-pose teaching YAML files.
    lines.extend(("", "[teaching_snapshot_yaml]"))
    teaching_yaml = yaml.safe_dump(
        document.get("teaching_snapshot", {}) or {},
        sort_keys=False,
        default_flow_style=False,
    ).rstrip("\n")
    lines.extend(teaching_yaml.splitlines() or ["{}"])

    lines.extend(("", "[touch_snapshot_yaml]"))
    touch_yaml = yaml.safe_dump(
        document.get("touch_snapshot", {}) or {},
        sort_keys=False,
        default_flow_style=False,
    ).rstrip("\n")
    lines.extend(touch_yaml.splitlines() or ["{}"])

    lines.extend((
        "",
        "[samples]",
        "elapsed_s raw0 state arc gas fwd wcr current_a voltage_v "
        "wire_feed_m_min set_current_a set_voltage_v error db collision",
    ))
    for sample in document.get("samples", ()):
        lines.append(
            f"{sample['elapsed_s']:.3f} "
            f"0x{int(sample.get('raw0') or 0):02X} "
            f"{sample.get('output_state_name') or 'unknown'} "
            f"{int(bool(sample.get('arc_ack')))} "
            f"{int(bool(sample.get('gas_ack')))} "
            f"{int(bool(sample.get('forward_ack')))} "
            f"{int(bool(sample.get('wcr_detected')))} "
            f"{sample.get('feedback_current_a') or 0} "
            f"{sample.get('feedback_voltage_v') or 0.0} "
            f"{sample.get('wire_feed_m_min') or 0.0} "
            f"{sample.get('set_current_a') or 0} "
            f"{sample.get('set_voltage_v') or 0.0} "
            f"{sample.get('welder_error') or 0} "
            f"{int(bool(sample.get('db_unavailable')))} "
            f"{int(bool(sample.get('torch_collision')))}"
        )

    lines.extend((
        "",
        "[tcp_trajectory]",
        "elapsed_s x_m y_m z_m qx qy qz qw speed_m_s tf_stamp_s "
        "along_mm remaining_mm cross_track_mm signed_weave_offset_mm "
        "weave_tracking_error_mm raw_speed_m_s progress waypoint phase",
    ))
    for sample in document.get("tcp_trajectory", ()):
        phase = str(sample.get("phase", "unknown")).replace(" ", "_")

        def tcp_value(name, digits):
            value = sample.get(name)
            return "nan" if value is None else f"{float(value):.{digits}f}"

        lines.append(
            f"{float(sample.get('elapsed_s', 0.0)):.4f} "
            f"{float(sample.get('x_m', 0.0)):.7f} "
            f"{float(sample.get('y_m', 0.0)):.7f} "
            f"{float(sample.get('z_m', 0.0)):.7f} "
            f"{float(sample.get('qx', 0.0)):.8f} "
            f"{float(sample.get('qy', 0.0)):.8f} "
            f"{float(sample.get('qz', 0.0)):.8f} "
            f"{float(sample.get('qw', 1.0)):.8f} "
            f"{float(sample.get('speed_m_s', 0.0)):.6f} "
            f"{tcp_value('tf_stamp_s', 9)} "
            f"{tcp_value('along_mm', 3)} "
            f"{tcp_value('remaining_mm', 3)} "
            f"{tcp_value('cross_track_mm', 3)} "
            f"{tcp_value('signed_weave_offset_mm', 3)} "
            f"{tcp_value('weave_tracking_error_mm', 3)} "
            f"{tcp_value('raw_speed_m_s', 6)} "
            f"{float(sample.get('progress', 0.0)):.5f} "
            f"{int(sample.get('waypoint_index', -1))} {phase}"
        )
    return "\n".join(lines) + "\n"


def save_weld_feedback_log(path, document):
    """Atomically save one completed welding-feedback report as text."""
    with atomic_text_writer(path) as stream:
        stream.write(format_weld_feedback_log(document))


def read_weld_log_sections(path):
    """Return ``{section: {key: raw string}}`` for a saved weld feedback log.

    Missing or unreadable files give ``{}``.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    # Lines before the first [section] (result=, started=, ...) go to "header".
    sections = {"header": {}}
    section = "header"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            sections[section] = {}
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        sections[section][key] = value
    return sections


def read_last_execution_settings(path):
    """Recover GUI-parameter defaults from a previously saved weld feedback
    log's ``[commanded]``/``[execution_conditions]`` sections.

    Lets a new GUI session start from exactly what last actually ran (recipe
    I/V/material and motion speed/lead/ARC timing) instead of hard-coded
    fallbacks. Returns ``{}`` (or a partial dict) if the log is missing or a
    field was never recorded -- callers must fall back to their own default
    for anything absent.
    """
    if not Path(path).is_file():
        return {}
    sections = read_weld_log_sections(path)

    def cast(section_name, key, converter):
        raw = sections.get(section_name, {}).get(key)
        if raw is None:
            return None
        try:
            return converter(raw)
        except (TypeError, ValueError):
            return None

    def cast_bool(section_name, key):
        raw = sections.get(section_name, {}).get(key)
        if raw is None:
            return None
        return raw.strip().lower() in ("1", "true", "yes")

    settings = {}
    for key, converter in (
        ("current_a", lambda v: int(round(float(v)))),
        ("voltage_tenths", lambda v: int(round(float(v)))),
        ("material", str),
        ("diameter_mm", float),
        ("mode", str),
        ("gas", str),
        ("correction", float),
        ("hot_start_percent", float),
        ("hot_start_hold_adjustment", lambda v: int(round(float(v)))),
        ("custom_hot_start_hold_s", float),
        ("custom_hot_start_percent", float),
        ("crater_panel_current_ref_a", float),
        ("crater_panel_voltage_ref_v", float),
        ("crater_panel_time_ref_s", float),
        ("software_crater_ratio_percent", float),
        ("software_crater_voltage_v", float),
        ("software_crater_hold_s", float),
        ("wire_consumable_alpha_mm", float),
    ):
        value = cast("commanded", key, converter)
        if value is not None:
            settings[key] = value
    synergic = cast_bool("commanded", "synergic")
    if synergic is not None:
        settings["synergic"] = synergic
    for key in ("hot_start_enabled", "custom_hot_start_enabled",
                "expect_native_crater", "software_crater_enabled"):
        value = cast_bool("commanded", key)
        if value is not None:
            settings[key] = value
    for old, new, converter in (
        ("crater_enabled", "expect_native_crater", lambda raw: raw.strip().lower() in ("1", "true", "yes")),
        ("crater_current_a", "crater_panel_current_ref_a", float),
        ("crater_voltage_v", "crater_panel_voltage_ref_v", float),
        ("crater_seconds", "crater_panel_time_ref_s", float),
    ):
        if new not in settings:
            value = cast("commanded", old, converter)
            if value is not None:
                settings[new] = value

    motion = {}

    for key, converter in (
        ("gui_velocity_percent", float),
        ("gui_speed_mode", str),
        ("gui_tcp_speed_mm_s", float),
        ("weld_lead_in_mm", float),
        ("weld_lead_out_mm", float),
        ("weld_safe_approach_mm", float),
        ("weld_approach_mode", str),
        ("weld_pre_start_lead_mm", float),
        ("weld_arc_off_delay_ms", float),
        ("weld_tcp_speed_mm_s", float),
        ("weld_fixed_tilt_x_deg", float),
        ("weld_fixed_tilt_y_deg", float),
        ("weld_fixed_tilt_z_deg", float),
        ("weld_weave_pattern", str),
        ("capping_width_mm", float),
        ("capping_pitch_mm", float),
        ("capping_left_dwell_s", float),
        ("capping_right_dwell_s", float),
        ("weld_weave_amplitude_mm", float),
        ("weld_weave_pitch_mm", float),
        ("weld_weave_left_dwell_s", float),
        ("weld_weave_right_dwell_s", float),
        ("weld_weave_cycles", lambda value: int(float(value))),
        ("weld_weave_samples_per_cycle", lambda value: int(float(value))),
        ("weld_weave_axis", str),
        ("weld_weave_reference", str),
    ):
        value = cast("execution_conditions", key, converter)
        if value is not None:
            motion[key] = value

    weave_enabled = cast_bool(
        "execution_conditions", "weld_weave_enabled"
    )
    if weave_enabled is not None:
        motion["weld_weave_enabled"] = weave_enabled

    return {"settings": settings, "motion": motion}


def read_teaching_and_touch_snapshot(path):
    """Parse the ``[teaching_snapshot_yaml]``/``[touch_snapshot_yaml]``
    sections a weld feedback log embeds (see ``format_weld_feedback_log``).

    Returns ``(teaching_raw, touch_raw)`` -- plain dicts as they appear in
    the log, not yet validated against ``ARM_JOINT_NAMES`` etc. Older logs
    written before this feature existed have neither section, so both come
    back empty rather than raising.
    """
    path = Path(path)
    if not path.is_file():
        return {}, {}
    section = None
    blocks = {"teaching_snapshot_yaml": [], "touch_snapshot_yaml": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        if section in blocks:
            blocks[section].append(line)

    def load_block(name):
        text = "\n".join(blocks[name]).strip()
        if not text:
            return {}
        try:
            loaded = yaml.load(text, Loader=yaml.CSafeLoader)
        except yaml.YAMLError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    return load_block("teaching_snapshot_yaml"), load_block("touch_snapshot_yaml")


def read_weld_pass_reference(path):
    """Read one pass's WAIT/START/GOAL WAIT/GOAL set from a completed log."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Pass reference log is missing: {path}")
    raw = path.read_bytes()
    if not any(line == "result=completed" for line in
               raw.decode("utf-8").splitlines()[:8]):
        raise ValueError(f"Pass reference must be a completed weld: {path.name}")
    teaching, _touches = read_teaching_and_touch_snapshot(path)
    poses = {}
    joint_states = {}
    for endpoint, name in (
        ("start_wait", "weld_start_wait"),
        ("start", "weld_start"),
        ("goal_wait", "weld_goal_wait"),
        ("goal", "weld_end"),
    ):
        entry = teaching.get(name)
        if not isinstance(entry, dict) or entry.get("planning_group") != "right_manipulator":
            raise ValueError(f"{path.name} has no right-arm {name} reference")
        pose = _pose_from_yaml_dict(entry.get("tcp_pose_world"), name)
        if not pose_is_valid(pose):
            raise ValueError(f"{path.name} has an invalid {name} TCP pose")
        poses[endpoint] = pose
        joint_state = entry.get("joint_state")
        if isinstance(joint_state, dict):
            names = tuple(joint_state.get("names", ()))
            positions = tuple(float(value) for value in
                              joint_state.get("positions_rad", ()))
            if (
                len(names) == 6
                and len(positions) == 6
                and all(math.isfinite(value) for value in positions)
            ):
                joint_states[endpoint] = (names, positions)
    if math.dist(_pose_position_tuple(poses["start"]),
                 _pose_position_tuple(poses["goal"])) < 0.001:
        raise ValueError(f"{path.name} seam is shorter than 1 mm")
    additional_pose_entries = {}
    for name in ("robot_start", "weld_wait", "weld_finish"):
        entry = teaching.get(name)
        if (
            isinstance(entry, dict)
            and entry.get("planning_group") == "right_manipulator"
        ):
            try:
                _pose_from_yaml_dict(entry.get("tcp_pose_world"), name)
            except (TypeError, ValueError):
                continue
            additional_pose_entries[name] = copy.deepcopy(entry)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "reference_kind": "completed_weld_log",
        **poses,
        "joint_states": joint_states,
        "additional_pose_entries": additional_pose_entries,
    }


def analyze_saved_weld_log(path):
    """Calculate the new metrics from an old log without modifying that log."""
    import datetime
    import re
    from construct_robot.io.weld_feedback_plot import (
        parse_weld_feedback_log, parse_weld_trajectory_log,
    )

    sections, raw_samples = parse_weld_feedback_log(path)
    trajectory = parse_weld_trajectory_log(path)["actual"]
    conditions = dict(sections.get("execution_conditions", {}))
    commanded = dict(sections.get("commanded", {}))
    production = dict(sections.get("production_metrics", {}))
    control = dict(sections.get("arc_off_control", {}))
    custom = dict(sections.get("custom_hot_start", {}))
    echo = dict(sections.get("rx_welding_setting_echo", {})
                or sections.get("rx_setting_echo", {}))
    for mapping in (conditions, commanded, production, control, custom, echo):
        for key, value in list(mapping.items()):
            numeric = _number(value)
            if numeric is not None:
                mapping[key] = numeric
            elif value in ("True", "False"):
                mapping[key] = value == "True"
    stages = [(int(match.group(1)), value) for key, value in conditions.items()
              if (match := re.fullmatch(r"steps\[(\d+)\]\.weld_scenario_stage", key))]
    motion_index = next((index for index, stage in stages if stage == "weld_motion"), None)
    if motion_index is not None:
        prefix = f"steps[{motion_index}]"
        for name, target in (("usable_seam_start", "seam_start_xyz"),
                             ("usable_seam_goal", "seam_goal_xyz")):
            values = [_number(conditions.get(f"{prefix}.{name}.position_m.{axis}"))
                      for axis in "xyz"]
            if all(value is not None for value in values):
                conditions[target] = tuple(values)
        waypoints = {}
        expression = re.compile(rf"{re.escape(prefix)}\.waypoints\[(\d+)\]\.position_m\.([xyz])")
        for key, value in conditions.items():
            match = expression.fullmatch(key)
            if match and _number(value) is not None:
                waypoints.setdefault(int(match.group(1)), {})[match.group(2)] = float(value)
        conditions["planned_weave_waypoints_xyz"] = [
            tuple(waypoints[index][axis] for axis in "xyz")
            for index in sorted(waypoints)
            if all(axis in waypoints[index] for axis in "xyz")
        ]
    samples = []
    state_codes = {"idle": 0, "main_weld": 1, "crater": 2, "weld_end": 3}
    for raw in raw_samples:
        samples.append({
            "elapsed_s": raw["elapsed_s"],
            "output_state": state_codes.get(raw["state"], 0),
            "arc_ack": bool(raw["arc"]), "gas_ack": bool(raw["gas"]),
            "forward_ack": bool(raw["fwd"]), "wcr_detected": bool(raw["wcr"]),
            "feedback_current_a": raw["current_a"],
            "feedback_voltage_v": raw["voltage_v"],
            "wire_feed_m_min": raw["wire_feed_m_min"],
        })
    started = sections.get("header", {}).get("started")
    try:
        started_unix = datetime.datetime.fromisoformat(started).timestamp()
    except (TypeError, ValueError):
        started_unix = None
    return analyze_weld_quality({
        "started_unix_time": started_unix,
        "samples": samples, "tcp_trajectory": trajectory,
        "execution_conditions": conditions,
        "commanded": commanded, "rx_welding_setting_echo": echo,
        "production_metrics": production, "arc_off_control": control,
        "custom_hot_start": custom,
    })
