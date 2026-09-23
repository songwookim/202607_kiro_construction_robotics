"""Pure welding-feedback calculations, report text, and atomic save."""
import math
from pathlib import Path
import tempfile

import yaml

from construct_robot.weld_quality_metrics import format_quality_summary


def calculate_weld_production_metrics(
    samples,
    *,
    weld_motion_start_elapsed_s=None,
    weld_motion_complete_elapsed_s=None,
    wire_consumable_alpha_mm=0.0,
):
    """Integrate actual ARC time and wire feed on the feedback time base.

    Each RX value owns the interval until the following RX sample.  An arc is
    considered physically active from WCR/current/output state, rather than
    merely from the outbound ARC command bit.  This excludes pre-gas and ARC
    recognition delay from production time.
    """
    ordered = sorted(
        (sample for sample in samples if sample.get("elapsed_s") is not None),
        key=lambda sample: float(sample["elapsed_s"]),
    )
    arc_on_time_s = 0.0
    main_weld_arc_time_s = 0.0
    crater_arc_time_s = 0.0
    net_weld_arc_time_s = 0.0
    wire_consumable_base_mm = 0.0
    arc_started_elapsed_s = None
    arc_completed_elapsed_s = None
    start = (
        None if weld_motion_start_elapsed_s is None
        else float(weld_motion_start_elapsed_s)
    )
    complete = (
        None if weld_motion_complete_elapsed_s is None
        else float(weld_motion_complete_elapsed_s)
    )
    for first, second in zip(ordered, ordered[1:]):
        interval_start = float(first["elapsed_s"])
        interval_end = float(second["elapsed_s"])
        dt = max(0.0, min(0.25, interval_end - interval_start))
        effective_end = interval_start + dt
        output_state = int(first.get("output_state", 0) or 0)
        active = bool(
            first.get("wcr_detected")
            or float(first.get("feedback_current_a", 0.0) or 0.0) > 10.0
            or output_state in (1, 2)
        )
        if not active or dt <= 0.0:
            continue
        if arc_started_elapsed_s is None:
            arc_started_elapsed_s = interval_start
        arc_completed_elapsed_s = effective_end
        arc_on_time_s += dt
        if output_state == 1:
            main_weld_arc_time_s += dt
        elif output_state == 2:
            crater_arc_time_s += dt
        wire_feed = max(
            0.0, float(first.get("wire_feed_m_min", 0.0) or 0.0)
        )
        wire_consumable_base_mm += wire_feed * 1000.0 / 60.0 * dt
        if start is not None and complete is not None and complete >= start:
            overlap = max(
                0.0,
                min(effective_end, complete) - max(interval_start, start),
            )
            net_weld_arc_time_s += overlap
    average_wire_feed = (
        wire_consumable_base_mm * 60.0 / (1000.0 * arc_on_time_s)
        if arc_on_time_s > 0.0 else 0.0
    )
    alpha = float(wire_consumable_alpha_mm)
    motion_duration = (
        max(0.0, complete - start)
        if start is not None and complete is not None else None
    )
    return {
        "arc_started_elapsed_s": arc_started_elapsed_s,
        "arc_completed_elapsed_s": arc_completed_elapsed_s,
        "arc_on_time_s": arc_on_time_s,
        "main_weld_arc_time_s": main_weld_arc_time_s,
        "crater_arc_time_s": crater_arc_time_s,
        "weld_motion_start_elapsed_s": start,
        "weld_motion_complete_elapsed_s": complete,
        "weld_motion_duration_s": motion_duration,
        "net_weld_arc_time_s": net_weld_arc_time_s,
        "wire_feed_average_m_min": average_wire_feed,
        "wire_consumable_base_mm": wire_consumable_base_mm,
        "wire_consumable_alpha_mm": alpha,
        "wire_consumable_mm": wire_consumable_base_mm + alpha,
        "wire_consumable_formula": "integral(WFS*1000/60*dt)+alpha",
    }


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(format_weld_feedback_log(document))
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

