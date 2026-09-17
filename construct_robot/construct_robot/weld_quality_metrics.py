"""Read-only welding quality metrics from RX and measured TCP samples.

No control decision depends on these values.  Missing geometry or evidence is
reported as None, never filled with a guessed physical quantity.
"""

import math
import statistics


STARTUP_EXCLUSION_S = 0.50
MAIN_CURRENT_STABILITY_RATIO = 0.80
EXTINCTION_EXCLUSION_S = 0.20
MAX_RX_INTERVAL_S = 0.25
MAX_TCP_INTERVAL_S = 0.25
DWELL_AMPLITUDE_TOLERANCE_MM = 0.50
DWELL_TCP_SPEED_THRESHOLD_MM_S = 2.0
MIN_EXTREME_SEPARATION_MM = 0.5
ENDPOINT_SETTLE_S = 0.20
ENDPOINT_MIN_SAMPLES = 3
HOT_START_EARLY_WINDOW_S = 1.0
HOT_START_MIN_PLATEAU_S = 0.12


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _stats(values):
    numbers = [float(value) for value in values if _number(value) is not None]
    if not numbers:
        return {"average": None, "min": None, "max": None, "std": None}
    return {
        "average": statistics.fmean(numbers), "min": min(numbers),
        "max": max(numbers), "std": statistics.pstdev(numbers),
    }


def _active(sample):
    return bool(
        sample.get("wcr_detected")
        or (_number(sample.get("feedback_current_a")) or 0.0) > 10.0
        or int(sample.get("output_state") or 0) in (1, 2)
    )


def _xyz(sample):
    return tuple(_number(sample.get(key)) for key in ("x_m", "y_m", "z_m"))


def _distance_mm(first, second):
    a, b = _xyz(first), _xyz(second)
    if any(value is None for value in a + b):
        return None
    return math.dist(a, b) * 1000.0


def _event(elapsed, started_unix_time):
    if elapsed is None:
        return {"elapsed_s": None, "unix_time": None, "wall_time": None}
    elapsed = float(elapsed)
    unix_time = (None if started_unix_time is None
                 else float(started_unix_time) + elapsed)
    if unix_time is None:
        wall = None
    else:
        import datetime
        wall = datetime.datetime.fromtimestamp(unix_time).isoformat(timespec="milliseconds")
    return {"elapsed_s": elapsed, "unix_time": unix_time, "wall_time": wall}


def event_timeline(samples, *, started_unix_time=None, tx_frames=(),
                   motion_start=None, motion_end=None, arc_off_control=None):
    """First observed RX edges; commands use actual send timestamps when available."""
    ordered = sorted(samples, key=lambda row: float(row.get("elapsed_s", 0.0)))
    control = arc_off_control or {}
    times = {name: None for name in (
        "ARC_ON_CMD", "ARC_RECOGNIZED", "GAS_ON", "FWD_ON", "WCR_ON",
        "WELD_MOTION_START", "CRATER_ENTER", "ARC_OFF_CMD",
        "CURRENT_EXTINCT", "WCR_OFF", "WELD_MOTION_END",
    )}
    previous_tx_arc = False
    for frame in sorted(tx_frames, key=lambda row: float(row.get("elapsed_s", 0.0))):
        raw = frame.get("raw_hex", "")
        try:
            command = int(raw.split()[0], 16)
        except (IndexError, ValueError, AttributeError):
            continue
        arc = bool(command & 1)
        if arc and not previous_tx_arc and times["ARC_ON_CMD"] is None:
            times["ARC_ON_CMD"] = _number(frame.get("elapsed_s"))
        if previous_tx_arc and not arc and times["ARC_OFF_CMD"] is None:
            times["ARC_OFF_CMD"] = _number(frame.get("elapsed_s"))
        previous_tx_arc = arc
    times["ARC_OFF_CMD"] = times["ARC_OFF_CMD"] or _number(control.get("command_elapsed_s"))
    times["WELD_MOTION_START"] = _number(motion_start)
    times["WELD_MOTION_END"] = _number(motion_end)
    checks = {
        "ARC_RECOGNIZED": lambda row: bool(row.get("arc_ack")),
        "GAS_ON": lambda row: bool(row.get("gas_ack")),
        "FWD_ON": lambda row: bool(row.get("forward_ack")),
        "WCR_ON": lambda row: bool(row.get("wcr_detected")),
        "CRATER_ENTER": lambda row: int(row.get("output_state") or 0) == 2,
    }
    for name, predicate in checks.items():
        times[name] = next((float(row["elapsed_s"]) for row in ordered
                            if predicate(row)), None)
    off = times["ARC_OFF_CMD"]
    if off is not None:
        post = [row for row in ordered if float(row["elapsed_s"]) >= off]
        for first, second in zip(post, post[1:]):
            if (times["CURRENT_EXTINCT"] is None
                and (_number(first.get("feedback_current_a")) or 0) <= 10
                and (_number(second.get("feedback_current_a")) or 0) <= 10):
                times["CURRENT_EXTINCT"] = float(first["elapsed_s"])
        times["WCR_OFF"] = next((float(row["elapsed_s"]) for row in post
                                 if not row.get("wcr_detected")), None)
    return {name: _event(value, started_unix_time) for name, value in times.items()}


def _unit(vector):
    length = math.sqrt(sum(value * value for value in vector))
    return None if length < 1e-9 else tuple(value / length for value in vector)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _rotate(q, vector):
    x, y, z, w = q
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if norm < 1e-9:
        return None
    u = (x/norm, y/norm, z/norm)
    v = tuple(float(value) for value in vector)
    cross = _cross(u, v)
    cross2 = _cross(u, cross)
    return tuple(v[i] + 2*w/norm*cross[i] + 2*cross2[i] for i in range(3))


def _transverse_axis(tcp, seam_direction, axis):
    selection = str(axis or "").lower()
    unit_axis = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0),
                 "z": (0.0, 0.0, 1.0)}.get(selection[-1:])
    if unit_axis is None or not selection.startswith(("world_", "tool_")):
        return None
    if selection.startswith("tool_"):
        quaternion = tuple(_number(tcp.get(key)) for key in ("qx", "qy", "qz", "qw"))
        if any(value is None for value in quaternion):
            return None
        unit_axis = _rotate(quaternion, unit_axis)
    dot = sum(a*b for a, b in zip(unit_axis, seam_direction))
    return _unit(tuple(a-dot*b for a, b in zip(unit_axis, seam_direction)))


def _extremes(tcp, amplitude_mm):
    """Return side transitions used for cycles, pitch, period and dwell."""
    if amplitude_mm is None or amplitude_mm <= 0:
        return []
    threshold = max(MIN_EXTREME_SEPARATION_MM, amplitude_mm*0.65)
    result = []
    last_side = 0
    for row in tcp:
        offset = _number(row.get("signed_weave_offset_mm"))
        if offset is None:
            continue
        side = 1 if offset >= threshold else -1 if offset <= -threshold else 0
        if side and side != last_side:
            result.append((side, row))
            last_side = side
    return result


def analyze_weld_quality(document):
    """Augment a live or reconstructed report without changing any commands."""
    samples = sorted(document.get("samples", ()), key=lambda row: float(row.get("elapsed_s", 0.0)))
    tcp = sorted(document.get("tcp_trajectory", ()), key=lambda row: float(row.get("elapsed_s", 0.0)))
    conditions = document.get("execution_conditions", {}) or {}
    commanded = document.get("commanded", {}) or {}
    echo = document.get("rx_welding_setting_echo") or document.get("rx_setting_echo") or {}
    production = document.get("production_metrics", {}) or {}
    control = document.get("arc_off_control", {}) or {}
    motion_start = _number(production.get("weld_motion_start_elapsed_s"))
    motion_end = _number(production.get("weld_motion_complete_elapsed_s"))
    events = event_timeline(
        samples, started_unix_time=document.get("started_unix_time"),
        tx_frames=document.get("tx_frames", ()), motion_start=motion_start,
        motion_end=motion_end, arc_off_control=control,
    )
    established = events["WCR_ON"]["elapsed_s"]
    off = events["ARC_OFF_CMD"]["elapsed_s"]
    steady_begin = None if established is None else established + STARTUP_EXCLUSION_S
    requested_main = _number(commanded.get("current_a"))
    if steady_begin is not None and requested_main is not None:
        settled_start = next((float(row["elapsed_s"]) for row in samples
                              if float(row["elapsed_s"]) >= established
                              and int(row.get("output_state") or 0) == 1
                              and (_number(row.get("feedback_current_a")) or 0)
                              >= MAIN_CURRENT_STABILITY_RATIO*requested_main), None)
        if settled_start is not None:
            steady_begin = max(steady_begin, settled_start)
    steady_end = None if off is None else off - EXTINCTION_EXCLUSION_S
    arc_rows = [row for row in samples
                if (_number(row.get("feedback_current_a")) or 0.0) > 10.0]
    steady_rows = [row for row in arc_rows if int(row.get("output_state") or 0) == 1
                   and steady_begin is not None and steady_end is not None
                   and steady_begin <= float(row["elapsed_s"]) <= steady_end]

    electrical = {}
    for key, label in (("feedback_current_a", "current_a"),
                       ("feedback_voltage_v", "voltage_v")):
        electrical[label] = {
            "requested": _number(commanded.get("current_a" if label == "current_a" else "voltage",
                                               (float(commanded.get("voltage_tenths", 0))/10
                                                if label == "voltage_v" else None))),
            "rx_echo": _number(echo.get(label)),
            "arc_active": _stats(row.get(key) for row in arc_rows),
            "steady_main": _stats(row.get(key) for row in steady_rows),
        }
    electrical["wfs_m_min"] = _stats(row.get("wire_feed_m_min") for row in arc_rows)
    electrical["steady_window_start_s"] = steady_begin
    electrical["steady_window_end_s"] = steady_end

    arc_energy = 0.0
    wire_arc = 0.0
    wire_motion = 0.0
    crater_rows = []
    crater_duration = 0.0
    for first, second in zip(samples, samples[1:]):
        t0, t1 = float(first["elapsed_s"]), float(second["elapsed_s"])
        dt = max(0.0, min(MAX_RX_INTERVAL_S, t1-t0))
        if not _active(first) or dt <= 0:
            continue
        current = max(0.0, _number(first.get("feedback_current_a")) or 0.0)
        voltage = max(0.0, _number(first.get("feedback_voltage_v")) or 0.0)
        wfs = max(0.0, _number(first.get("wire_feed_m_min")) or 0.0)
        arc_energy += current * voltage * dt
        wire_arc += wfs * 1000.0 / 60.0 * dt
        if motion_start is not None and motion_end is not None:
            overlap = max(0.0, min(t0+dt, motion_end)-max(t0, motion_start))
            wire_motion += wfs * 1000.0 / 60.0 * overlap
        if int(first.get("output_state") or 0) == 2:
            crater_duration += dt
            crater_rows.append(first)

    seam_speed = []
    tcp_speed = []
    seam_alongs = []
    for first, second in zip(tcp, tcp[1:]):
        t0, t1 = float(first["elapsed_s"]), float(second["elapsed_s"])
        dt = t1-t0
        if not 1e-4 < dt <= MAX_TCP_INTERVAL_S:
            continue
        if motion_start is not None and t0 < motion_start:
            continue
        if motion_end is not None and t1 > motion_end:
            continue
        a, b = _number(first.get("along_mm")), _number(second.get("along_mm"))
        if a is not None and b is not None and b >= a:
            seam_speed.append((b-a)/dt)
            seam_alongs.extend((a, b))
        distance = _distance_mm(first, second)
        if distance is not None:
            tcp_speed.append(distance/dt)
    seam_length = (max(seam_alongs)-min(seam_alongs)) if seam_alongs else None

    # Signed offset needs a known weave axis and a taught seam segment.  The
    # existing cross_track_mm is an unsigned offset, never a tracking error.
    start_xyz = conditions.get("seam_start_xyz")
    goal_xyz = conditions.get("seam_goal_xyz")
    if start_xyz is not None and goal_xyz is not None:
        start_xyz = tuple(float(value) for value in start_xyz)
        goal_xyz = tuple(float(value) for value in goal_xyz)
        direction = _unit(tuple(b-a for a, b in zip(start_xyz, goal_xyz)))
    else:
        direction = None
    transverse = _transverse_axis(tcp[0], direction, conditions.get("weld_weave_axis")) if tcp and direction else None
    if transverse is not None:
        for row in tcp:
            xyz = _xyz(row)
            along = _number(row.get("along_mm"))
            if None in xyz or along is None:
                row["signed_weave_offset_mm"] = None
                continue
            center = tuple(start_xyz[i]+along*0.001*direction[i] for i in range(3))
            row["signed_weave_offset_mm"] = 1000.0*sum(
                (xyz[i]-center[i])*transverse[i] for i in range(3))
    else:
        for row in tcp:
            row["signed_weave_offset_mm"] = None
    planned = []
    if direction is not None:
        for value in conditions.get("planned_weave_waypoints_xyz", ()) or ():
            try:
                point = tuple(float(component) for component in value)
                along = 1000.0*sum((point[i]-start_xyz[i])*direction[i] for i in range(3))
                planned.append((along, point))
            except (TypeError, ValueError, IndexError):
                continue
    planned.sort(key=lambda item: item[0])
    tracking_errors = []
    for row in tcp:
        row["weave_tracking_error_mm"] = None
        if motion_start is not None and float(row["elapsed_s"]) < motion_start:
            continue
        if motion_end is not None and float(row["elapsed_s"]) > motion_end:
            continue
        along = _number(row.get("along_mm"))
        point = _xyz(row)
        if along is None or None in point or len(planned) < 2:
            continue
        for (a, p), (b, q) in zip(planned, planned[1:]):
            if a <= along <= b and b-a > 1e-6:
                fraction = (along-a)/(b-a)
                target = tuple(p[i]+fraction*(q[i]-p[i]) for i in range(3))
                error = 1000.0*math.dist(point, target)
                row["weave_tracking_error_mm"] = error
                tracking_errors.append(error)
                break

    signed = [_number(row.get("signed_weave_offset_mm")) for row in tcp]
    left = [-value for value in signed if value is not None and value < 0]
    right = [value for value in signed if value is not None and value > 0]
    amplitude = _number(conditions.get("weld_weave_amplitude_mm"))
    extreme_rows = _extremes(tcp, amplitude)
    same_side_pairs = [(first, second) for first, second in zip(extreme_rows, extreme_rows[2:])
                       if first[0] == second[0]]
    periods = [float(b["elapsed_s"])-float(a["elapsed_s"])
               for (_side, a), (_, b) in same_side_pairs]
    pitches = [(_number(b.get("along_mm")) or 0)-(_number(a.get("along_mm")) or 0)
               for (_side, a), (_, b) in same_side_pairs
               if _number(a.get("along_mm")) is not None and _number(b.get("along_mm")) is not None]
    measured_dwell = {"left": [], "right": []}
    dwell_side = None
    dwell_seconds = 0.0
    # Only count contiguous low-speed residence at an actual extreme.
    for first, second in zip(tcp, tcp[1:]):
        dt = float(second["elapsed_s"])-float(first["elapsed_s"])
        if not 0 < dt <= MAX_TCP_INTERVAL_S:
            continue
        offset = _number(first.get("signed_weave_offset_mm"))
        speed = _number(first.get("raw_speed_m_s", first.get("speed_m_s")))
        if offset is None or speed is None or amplitude is None:
            continue
        side = ("right" if offset > 0 else "left") if (
            abs(abs(offset)-amplitude) <= DWELL_AMPLITUDE_TOLERANCE_MM
            and speed*1000 <= DWELL_TCP_SPEED_THRESHOLD_MM_S
        ) else None
        if side != dwell_side and dwell_side is not None and dwell_seconds > 0:
            measured_dwell[dwell_side].append(dwell_seconds)
            dwell_seconds = 0.0
        dwell_side = side
        if side is not None:
            dwell_seconds += dt
    if dwell_side is not None and dwell_seconds > 0:
        measured_dwell[dwell_side].append(dwell_seconds)
    weave = {
        "pattern": conditions.get("weld_weave_pattern") if conditions.get("weld_weave_enabled") else None,
        "requested_amplitude_mm": amplitude,
        "requested_pitch_mm": _number(conditions.get("weld_weave_pitch_mm")),
        "planned_pitch_mm": _number(conditions.get("weld_weave_actual_pitch_mm")),
        "planned_cycle_count": conditions.get("weld_weave_cycles"),
        "axis": conditions.get("weld_weave_axis"),
        "measured_left_amplitude_mm": max(left) if left else None,
        "measured_right_amplitude_mm": max(right) if right else None,
        "measured_full_width_mm": max(left)+max(right) if left and right else None,
        "measured_pitch_avg_mm": _stats(pitches)["average"],
        "measured_pitch_std_mm": _stats(pitches)["std"],
        "measured_cycle_count": len(extreme_rows)//2 if extreme_rows else None,
        "measured_period_avg_s": _stats(periods)["average"],
        "measured_frequency_hz": 1/_stats(periods)["average"] if periods and _stats(periods)["average"] > 0 else None,
        "left_dwell_requested_s": _number(conditions.get("weld_weave_left_dwell_s")),
        "left_dwell_measured_avg_s": _stats(measured_dwell["left"])["average"],
        "right_dwell_requested_s": _number(conditions.get("weld_weave_right_dwell_s")),
        "right_dwell_measured_avg_s": _stats(measured_dwell["right"])["average"],
        "weave_tracking_error_mm": {
            "average": _stats(tracking_errors)["average"],
            "max": _stats(tracking_errors)["max"],
        },
    }

    tx_frames = document.get("tx_frames", ())
    on_tx = next((row for row in tx_frames if row.get("raw_hex", "").startswith("01 ")), None)
    raw = bytes.fromhex(on_tx["raw_hex"]) if on_tx else b""
    tx_current = int.from_bytes(raw[14:16], "little") if len(raw) == 55 else None
    tx_hold = raw[16] if len(raw) == 55 else None
    requested_hot = _number(commanded.get("hot_start_current_a"))
    requested_hold = _number(commanded.get("hot_start_hold_adjustment"))
    rx_hot = _number(echo.get("hot_start_current_a"))
    rx_hold = _number(echo.get("hot_start_hold_adjustment"))
    early = [row for row in samples if established is not None
             and established <= float(row["elapsed_s"]) <= established+HOT_START_EARLY_WINDOW_S]
    early_peak = max((_number(row.get("feedback_current_a")) or 0 for row in early), default=None)
    detected_duration = None
    if requested_hot is not None and requested_main is not None and requested_hot > requested_main:
        threshold = max(requested_hot*0.95,
                        requested_main + 0.5*(requested_hot-requested_main))
        run_start = None
        best_duration = 0.0
        for row in early:
            current = _number(row.get("feedback_current_a")) or 0.0
            t = float(row["elapsed_s"])
            if current >= threshold:
                if run_start is None:
                    run_start = t
                best_duration = max(best_duration, t-run_start)
            else:
                run_start = None
        if best_duration >= HOT_START_MIN_PLATEAU_S:
            detected_duration = best_duration
    tx_mismatch = (
        (tx_current is not None and requested_hot is not None and abs(tx_current-requested_hot) > 1)
        or (tx_hold is not None and requested_hold is not None and tx_hold != requested_hold+15)
    )
    rx_mismatch = (requested_hot and rx_hot is not None
                   and (abs(rx_hot-requested_hot) > 1 or rx_hold != requested_hold))
    if tx_mismatch or rx_mismatch:
        hot_status = "MISMATCH"
    elif tx_current is not None and rx_hot is not None and detected_duration is not None:
        hot_status = "CONFIRMED"
    else:
        hot_status = "UNCONFIRMED"
    hot = {
        "requested_boost_percent": _number(commanded.get("hot_start_percent")),
        "requested_current_a": requested_hot,
        "requested_hold_adjustment": requested_hold,
        "encoded_tx_current_a": tx_current,
        "encoded_tx_hold_raw": tx_hold,
        "encoded_tx_hot_raw_hex": raw[14:17].hex(" ").upper() if len(raw) == 55 else None,
        "rx_current_a": rx_hot,
        "rx_hold_adjustment": rx_hold,
        "rx_hot_raw_hex": echo.get("hot_start_rx_raw_hex"),
        "feedback_peak_near_establishment_a": early_peak,
        "detected_duration_s": detected_duration,
        "status": hot_status,
    }

    settled = [row for row in tcp if motion_end is not None
               and motion_end <= float(row["elapsed_s"]) <= motion_end+ENDPOINT_SETTLE_S]
    remaining_values = [_number(row.get("remaining_mm")) for row in settled[-3:]]
    remaining_values = [value for value in remaining_values if value is not None]
    final_target = planned[-1][1] if planned else goal_xyz
    if len(settled) >= ENDPOINT_MIN_SAMPLES and final_target is not None:
        endpoint_error = statistics.median(math.dist(_xyz(row), final_target)*1000 for row in settled[-3:])
        final_remaining = statistics.median(remaining_values) if remaining_values else None
    else:
        endpoint_error = final_remaining = None

    efficiency = _number(conditions.get("arc_efficiency"))
    energy_per_mm = arc_energy/seam_length if seam_length and seam_length > 0 else None
    geometry = {"work_angle_deg": None, "travel_angle_deg": None, "ctwd_mm": None}
    torch_axis_tool = conditions.get("torch_axis_tool_xyz")
    if direction is not None and torch_axis_tool is not None and tcp:
        try:
            tool_axis = _unit(tuple(float(value) for value in torch_axis_tool))
            mid = tcp[len(tcp)//2]
            quaternion = tuple(float(mid[key]) for key in ("qx", "qy", "qz", "qw"))
            torch_axis = _unit(_rotate(quaternion, tool_axis)) if tool_axis else None
            if torch_axis is not None:
                alignment = min(1.0, abs(sum(a*b for a, b in zip(torch_axis, direction))))
                geometry["travel_angle_deg"] = math.degrees(math.acos(alignment))
                plate_normal = conditions.get("plate_normal_xyz")
                if plate_normal is not None:
                    normal = _unit(tuple(float(value) for value in plate_normal))
                    if normal:
                        alignment = min(1.0, abs(sum(a*b for a, b in zip(torch_axis, normal))))
                        geometry["work_angle_deg"] = math.degrees(math.acos(alignment))
                        tip_offset = conditions.get("contact_tip_offset_tool_xyz_m")
                        surface_point = conditions.get("work_surface_point_xyz_m")
                        if tip_offset is not None and surface_point is not None:
                            rotated = _rotate(quaternion, tuple(float(value) for value in tip_offset))
                            tip = tuple(_xyz(mid)[i]+rotated[i] for i in range(3))
                            denominator = sum(a*b for a, b in zip(normal, torch_axis))
                            if abs(denominator) > 1e-6:
                                distance = sum(normal[i]*(float(surface_point[i])-tip[i])
                                               for i in range(3))/denominator
                                if distance >= 0:
                                    geometry["ctwd_mm"] = distance*1000.0
        except (TypeError, ValueError, IndexError, KeyError):
            pass
    return {
        "electrical": electrical,
        "timeline": events,
        "motion": {
            "requested_seam_speed_mm_s": _number(conditions.get("weld_tcp_speed_mm_s")),
            "actual_seam_speed": _stats(seam_speed),
            "actual_tcp_path_speed": _stats(tcp_speed),
            "actual_seam_length_mm": seam_length,
            "final_endpoint_error_mm": endpoint_error,
            "final_remaining_mm": final_remaining,
        },
        "weave": weave,
        "production": {
            "arc_energy_J": arc_energy if arc_rows else None,
            "arc_energy_J_per_mm": energy_per_mm,
            "heat_input_J_per_mm": (efficiency*energy_per_mm if efficiency is not None and energy_per_mm is not None else None),
            "wire_consumed_arc_active_mm": wire_arc if arc_rows else None,
            "wire_consumed_weld_motion_mm": wire_motion if motion_start is not None and motion_end is not None else None,
        },
        "hot_start": hot,
        "crater": {
            "control_source": "external_panel", "detected": bool(crater_rows),
            "duration_s": crater_duration if crater_rows else 0.0,
            "current": _stats(row.get("feedback_current_a") for row in crater_rows),
            "voltage": _stats(row.get("feedback_voltage_v") for row in crater_rows),
            "wfs_avg_m_min": _stats(row.get("wire_feed_m_min") for row in crater_rows)["average"],
        },
        "geometry": geometry,
    }


def format_quality_summary(document):
    """Compact operator view; detailed metric keys follow in the log."""
    quality = document.get("quality_metrics", {})
    if not quality:
        return []
    if "error" in quality:
        return ["================= WELD SUMMARY =================",
                f"Quality analysis unavailable: {quality['error']}"]
    electrical = quality["electrical"]
    motion = quality["motion"]
    weave = quality["weave"]
    production = quality["production"]
    hot = quality["hot_start"]
    crater = quality["crater"]
    geometry = quality["geometry"]

    def value(number, digits=2):
        return "N/A" if number is None else f"{float(number):.{digits}f}"

    def profile(entry, digits=2):
        steady = entry["steady_main"]
        active = entry["arc_active"]
        return (f"req {value(entry['requested'], digits)} / "
                f"echo {value(entry['rx_echo'], digits)} / "
                f"arc avg {value(active['average'], digits)} "
                f"[{value(active['min'], digits)}, {value(active['max'], digits)}] "
                f"std {value(active['std'], digits)} / "
                f"main avg {value(steady['average'], digits)} "
                f"[{value(steady['min'], digits)}, {value(steady['max'], digits)}] "
                f"std {value(steady['std'], digits)}")

    commanded = document.get("commanded", {})
    original = document.get("production_metrics", {})
    lines = [
        "================= WELD SUMMARY =================",
        "ELECTRICAL",
        f"Current      : {profile(electrical['current_a'])} A",
        f"Voltage      : {profile(electrical['voltage_v'])} V",
        f"WFS          : {value(electrical['wfs_m_min']['average'])} m/min",
        f"Transfer     : {commanded.get('mode', 'N/A')}",
        "MOTION",
        f"Seam speed   : req {value(motion['requested_seam_speed_mm_s'])} / "
        f"actual {value(motion['actual_seam_speed']['average'])} mm/s",
        f"TCP speed    : avg {value(motion['actual_tcp_path_speed']['average'])} / "
        f"max {value(motion['actual_tcp_path_speed']['max'])} mm/s",
        f"Seam length  : {value(motion['actual_seam_length_mm'])} mm",
        f"Work angle   : {value(geometry['work_angle_deg'])} deg",
        f"Travel angle : {value(geometry['travel_angle_deg'])} deg",
        f"CTWD         : {value(geometry['ctwd_mm'])} mm",
        "WEAVE",
        f"Pattern      : {weave['pattern'] or 'OFF'}",
        f"Amplitude    : req +/-{value(weave['requested_amplitude_mm'])} / "
        f"actual L {value(weave['measured_left_amplitude_mm'])}, "
        f"R {value(weave['measured_right_amplitude_mm'])} mm",
        f"Pitch        : req {value(weave['requested_pitch_mm'])} / "
        f"planned {value(weave['planned_pitch_mm'])} / "
        f"measured {value(weave['measured_pitch_avg_mm'])} mm",
        f"Cycles       : planned {weave['planned_cycle_count'] or 'N/A'} / "
        f"measured {weave['measured_cycle_count'] if weave['measured_cycle_count'] is not None else 'N/A'}",
        f"Dwell L/R    : req {value(weave['left_dwell_requested_s'])}/"
        f"{value(weave['right_dwell_requested_s'])} / measured "
        f"{value(weave['left_dwell_measured_avg_s'])}/"
        f"{value(weave['right_dwell_measured_avg_s'])} s",
        "START / END",
        f"Hot Start    : requested +{value(hot['requested_boost_percent'], 1)}%, "
        f"{value(hot['requested_current_a'], 0)} A, "
        f"hold {value(hot['requested_hold_adjustment'], 0)} / "
        f"TX {value(hot['encoded_tx_current_a'], 0)} A, raw "
        f"{value(hot['encoded_tx_hold_raw'], 0)} / "
        f"RX {value(hot['rx_current_a'], 0)} A, hold "
        f"{value(hot['rx_hold_adjustment'], 0)} / {hot['status']}",
        f"Crater       : external panel / detected {crater['detected']} / "
        f"{value(crater['duration_s'], 3)} s / "
        f"I {value(crater['current']['average'])} A, "
        f"V {value(crater['voltage']['average'])} V",
        "PRODUCTION",
        f"Arc On       : {value(original.get('arc_on_time_s'), 3)} s",
        f"Net Weld     : {value(original.get('net_weld_arc_time_s'), 3)} s",
        f"Wire Used    : {value(original.get('wire_consumable_mm'), 2)} mm",
        f"Arc Energy   : {value(production['arc_energy_J_per_mm'], 3)} J/mm "
        "(not heat input without efficiency)",
        f"Heat Input   : {value(production['heat_input_J_per_mm'], 3)} J/mm",
        "TRACKING",
        f"Weave error  : avg {value(weave['weave_tracking_error_mm']['average'])} / "
        f"max {value(weave['weave_tracking_error_mm']['max'])} mm",
        f"Endpoint err : {value(motion['final_endpoint_error_mm'], 3)} mm",
        "=================================================",
    ]
    return lines


def analyze_saved_weld_log(path):
    """Calculate the new metrics from an old log without modifying that log."""
    import datetime
    import re
    from construct_robot.weld_feedback_plot import (
        parse_weld_feedback_log, parse_weld_trajectory_log,
    )

    sections, raw_samples = parse_weld_feedback_log(path)
    trajectory = parse_weld_trajectory_log(path)["actual"]
    conditions = dict(sections.get("execution_conditions", {}))
    commanded = dict(sections.get("commanded", {}))
    production = dict(sections.get("production_metrics", {}))
    control = dict(sections.get("arc_off_control", {}))
    echo = dict(sections.get("rx_welding_setting_echo", {})
                or sections.get("rx_setting_echo", {}))
    for mapping in (conditions, commanded, production, control, echo):
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
    })
