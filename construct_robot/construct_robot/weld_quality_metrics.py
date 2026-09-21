"""Read-only welding quality metrics from RX and measured TCP samples.

No control decision depends on these values.  Missing geometry or evidence is
reported as None, never filled with a guessed physical quantity.
"""

import math
import statistics


STEADY_MAIN_STABILIZATION_MARGIN_S = 0.30
EXTINCTION_EXCLUSION_S = 0.20
MAX_RX_INTERVAL_S = 0.25
MAX_TCP_INTERVAL_S = 0.25
DWELL_TCP_SPEED_THRESHOLD_MM_S = 2.0
DWELL_PEAK_SUPPORT_WINDOW_S = 0.10
DWELL_OFFSET_TOLERANCE_MM = 0.20
DWELL_SEAM_PROGRESS_TOLERANCE_MM = 0.30
DWELL_PLATEAU_WINDOW_S = 0.25
DWELL_MAX_SAMPLE_GAP_S = 0.06
DWELL_MIN_VALID_SAMPLES = 4
DWELL_REQUEST_MATCH_LOWER_RATIO = 0.70
DWELL_REQUEST_MATCH_UPPER_RATIO = 1.30
MIN_PEAK_SEPARATION_S = 0.04
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
                   motion_start=None, motion_end=None, arc_off_control=None,
                   custom_hot_start=None):
    """First observed RX edges; commands use actual send timestamps when available."""
    ordered = sorted(samples, key=lambda row: float(row.get("elapsed_s", 0.0)))
    control = arc_off_control or {}
    times = {name: None for name in (
        "ARC_ON_CMD", "ARC_RECOGNIZED", "GAS_ON", "FWD_ON", "WCR_ON",
        "CUSTOM_HOT_START_BEGIN", "CUSTOM_HOT_START_END",
        "WELD_MOTION_START", "ARC_OFF_CMD", "CRATER_ENTER", "CRATER_EXIT",
        "CURRENT_EXTINCT", "WCR_OFF", "SEQUENCE_CLEAR", "WELD_MOTION_END",
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
    custom = custom_hot_start or {}
    times["CUSTOM_HOT_START_BEGIN"] = _number(custom.get("hold_start_elapsed_s"))
    times["CUSTOM_HOT_START_END"] = _number(custom.get("hold_end_elapsed_s"))
    checks = {
        "ARC_RECOGNIZED": lambda row: bool(row.get("arc_ack")),
        "GAS_ON": lambda row: bool(row.get("gas_ack")),
        "FWD_ON": lambda row: bool(row.get("forward_ack")),
        "WCR_ON": lambda row: bool(row.get("wcr_detected")),
    }
    for name, predicate in checks.items():
        times[name] = next((float(row["elapsed_s"]) for row in ordered
                            if predicate(row)), None)
    off = times["ARC_OFF_CMD"]
    if off is not None:
        post = [row for row in ordered if float(row["elapsed_s"]) >= off]
        for index, row in enumerate(ordered):
            previous_state = (int(ordered[index-1].get("output_state") or 0)
                              if index else None)
            if (float(row["elapsed_s"]) >= off
                    and int(row.get("output_state") or 0) == 2
                    and previous_state != 2):
                times["CRATER_ENTER"] = float(row["elapsed_s"])
                times["CRATER_EXIT"] = next((float(later["elapsed_s"])
                    for later in ordered[index+1:] if int(later.get("output_state") or 0) != 2), None)
                break
        extinction_rows = [row for row in post if times["CRATER_EXIT"] is None
                           or float(row["elapsed_s"]) >= times["CRATER_EXIT"]]
        for first, second in zip(extinction_rows, extinction_rows[1:]):
            if (times["CURRENT_EXTINCT"] is None
                and (_number(first.get("feedback_current_a")) or 0) <= 10
                and (_number(second.get("feedback_current_a")) or 0) <= 10):
                times["CURRENT_EXTINCT"] = float(first["elapsed_s"])
        times["WCR_OFF"] = next((float(row["elapsed_s"]) for row in post
                                 if not row.get("wcr_detected")), None)
    times["SEQUENCE_CLEAR"] = _number(control.get("sequence_clear_elapsed_s"))
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


def _weave_peaks(tcp, planned_pitch_mm=None, planned_peak_alongs=()):
    """Measure actual extrema; commanded amplitude is never a threshold."""
    points = [row for row in tcp
              if _number(row.get("signed_weave_offset_mm")) is not None
              and _number(row.get("along_mm")) is not None]
    if len(points) < 3:
        return []
    if planned_peak_alongs and planned_pitch_mm and planned_pitch_mm > 0:
        # Planned extrema identify *where* to look, never *how far* the robot
        # must reach. Search every TF sample so a flat hold still has a peak.
        half_window = 0.24 * planned_pitch_mm
        guided = []
        used = set()
        for expected_along, side in planned_peak_alongs:
            nearby = [row for row in points
                      if abs(float(row["along_mm"])-expected_along) <= half_window
                      and id(row) not in used]
            if not nearby:
                continue
            score = lambda row: (_number(row["signed_weave_offset_mm"])
                                 if side == "right" else -_number(row["signed_weave_offset_mm"]),
                                 -abs(float(row["along_mm"])-expected_along))
            row = max(nearby, key=score)
            if ((side == "right" and row["signed_weave_offset_mm"] <= 0)
                    or (side == "left" and row["signed_weave_offset_mm"] >= 0)):
                continue
            guided.append((side, row))
            used.add(id(row))
        if guided:
            return guided

    # Fallback for logs without planned waypoint geometry: a three-sample
    # median suppresses isolated TF noise before derivative sign changes.
    offsets = [_number(row["signed_weave_offset_mm"]) for row in points]
    smooth = [statistics.median(offsets[max(0, i-1):i+2])
              for i in range(len(points))]
    noise_floor = max(0.05, (max(smooth)-min(smooth))*0.05)
    candidates = []
    for index in range(1, len(points)-1):
        before, value, after = smooth[index-1:index+2]
        if value >= before and value > after and value > noise_floor:
            candidates.append(("right", points[index]))
        elif value <= before and value < after and value < -noise_floor:
            candidates.append(("left", points[index]))
    result = []
    for side, row in candidates:
        if result and side == result[-1][0]:
            old = result[-1][1]
            if abs(row["signed_weave_offset_mm"]) > abs(old["signed_weave_offset_mm"]):
                result[-1] = (side, row)
        elif not result or float(row["elapsed_s"])-float(result[-1][1]["elapsed_s"]) >= MIN_PEAK_SEPARATION_S:
            result.append((side, row))
    return result


def _low_speed_at_peak(row, peak, tolerance=DWELL_OFFSET_TOLERANCE_MM):
    offset = _number(row.get("signed_weave_offset_mm"))
    speed = _number(row.get("raw_speed_m_s", row.get("speed_m_s")))
    return (offset is not None and abs(offset - peak) <= tolerance
            and speed is not None and speed * 1000.0 <= DWELL_TCP_SPEED_THRESHOLD_MM_S)


def _contiguous_dwell(tcp, peak_row):
    """Measure a contiguous offset/progress plateau at an actual peak.

    World-frame TCP speed is deliberately not a gate: TF position noise can
    make a stationary torch appear to move several mm/s.
    """
    peak = _number(peak_row.get("signed_weave_offset_mm"))
    peak_progress = _number(peak_row.get("along_mm"))
    peak_time = _number(peak_row.get("elapsed_s"))
    detail = {
        "peak_time": peak_time, "peak_offset_mm": peak,
        "dwell_start_time": None, "dwell_end_time": None,
        "dwell_duration_s": None, "sample_count": 0,
        "offset_range_mm": None, "seam_progress_range_mm": None,
        "valid": False,
    }
    if peak is None or peak_progress is None or peak_time is None:
        return detail
    peak_index = next((index for index, row in enumerate(tcp)
                       if row is peak_row), None)
    if peak_index is None:
        return detail

    def at_plateau(row):
        offset = _number(row.get("signed_weave_offset_mm"))
        progress = _number(row.get("along_mm"))
        timestamp = _number(row.get("elapsed_s"))
        return (offset is not None and progress is not None and timestamp is not None
                and abs(timestamp - peak_time) <= DWELL_PLATEAU_WINDOW_S
                and abs(offset - peak) <= DWELL_OFFSET_TOLERANCE_MM
                and abs(progress - peak_progress) <= DWELL_SEAM_PROGRESS_TOLERANCE_MM)

    begin = end = peak_index
    while (begin > 0 and at_plateau(tcp[begin - 1])
           and 0 < float(tcp[begin]["elapsed_s"]) - float(tcp[begin - 1]["elapsed_s"])
           <= DWELL_MAX_SAMPLE_GAP_S):
        begin -= 1
    while (end + 1 < len(tcp) and at_plateau(tcp[end + 1])
           and 0 < float(tcp[end + 1]["elapsed_s"]) - float(tcp[end]["elapsed_s"])
           <= DWELL_MAX_SAMPLE_GAP_S):
        end += 1
    selected = tcp[begin:end + 1]
    offsets = [float(row["signed_weave_offset_mm"]) for row in selected]
    progress = [float(row["along_mm"]) for row in selected]
    start_time = float(selected[0]["elapsed_s"])
    end_time = float(selected[-1]["elapsed_s"])
    detail.update(
        dwell_start_time=start_time, dwell_end_time=end_time,
        sample_count=len(selected), offset_range_mm=max(offsets) - min(offsets),
        seam_progress_range_mm=max(progress) - min(progress),
    )
    if len(selected) >= DWELL_MIN_VALID_SAMPLES and end_time > start_time:
        detail["dwell_duration_s"] = end_time - start_time
        detail["valid"] = True
    return detail


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
        custom_hot_start=document.get("custom_hot_start"),
    )
    established = events["WCR_ON"]["elapsed_s"]
    off = events["ARC_OFF_CMD"]["elapsed_s"]
    # A WCR-based window can include the stationary custom Hot Start hold.
    # Only samples after weld motion has actually begun describe steady main welding.
    steady_begin = (None if motion_start is None else
                    motion_start + STEADY_MAIN_STABILIZATION_MARGIN_S)
    requested_main = _number(commanded.get("current_a"))
    steady_end = None if off is None else off - EXTINCTION_EXCLUSION_S
    software_command = _number((document.get("software_crater_control") or {}).get("command_elapsed_s"))
    if software_command is not None:
        steady_end = min(steady_end, software_command) if steady_end is not None else software_command
    if motion_end is not None:
        steady_end = min(steady_end, motion_end) if steady_end is not None else motion_end
    arc_rows = [row for row in samples
                if (_number(row.get("feedback_current_a")) or 0.0) > 10.0]
    steady_rows = [row for row in arc_rows if int(row.get("output_state") or 0) == 1
                   and steady_begin is not None and steady_end is not None
                   and steady_begin <= float(row["elapsed_s"]) < steady_end]

    custom = document.get("custom_hot_start") or {}
    custom_begin = events["CUSTOM_HOT_START_BEGIN"]["elapsed_s"]
    custom_end = events["CUSTOM_HOT_START_END"]["elapsed_s"]
    custom_rows = [row for row in samples
                   if custom_begin is not None and custom_end is not None
                   and custom_begin <= float(row["elapsed_s"]) <= custom_end]
    custom_summary = {
        "enabled": bool(custom.get("enabled", False)),
        "requested_hold_s": _number(custom.get("requested_hold_s")),
        "actual_hold_s": _number(custom.get("actual_hold_s")),
        "max_tcp_drift_mm": _number(custom.get("max_tcp_drift_mm")),
        "current_a": _stats(row.get("feedback_current_a") for row in custom_rows),
        "voltage_v": _stats(row.get("feedback_voltage_v") for row in custom_rows),
        "sample_count": len(custom_rows),
        "status": custom.get("status", "DISABLED"),
        "window_start_s": custom_begin,
        "window_end_s": custom_end,
    }

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
    crater_enter = events["CRATER_ENTER"]["elapsed_s"]
    crater_exit = events["CRATER_EXIT"]["elapsed_s"]
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
        if (crater_enter is not None and crater_enter <= t0
                and (crater_exit is None or t0 < crater_exit)
                and int(first.get("output_state") or 0) == 2):
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
    planned_peak_alongs = []
    if transverse is not None and planned:
        planned_offsets = []
        for along, point in planned:
            center = tuple(start_xyz[i]+along*0.001*direction[i] for i in range(3))
            offset = 1000.0*sum((point[i]-center[i])*transverse[i] for i in range(3))
            planned_offsets.append((along, offset))
        planned_span = max(abs(offset) for _along, offset in planned_offsets)
        for along, offset in planned_offsets:
            if planned_span <= 1e-6 or abs(offset) < planned_span*0.65:
                continue
            side = "right" if offset > 0 else "left"
            if planned_peak_alongs and planned_peak_alongs[-1][1] == side:
                # A hold inserts repeated waypoints at the same extremum.
                old_along, _ = planned_peak_alongs[-1]
                planned_peak_alongs[-1] = ((old_along+along)*0.5, side)
            else:
                planned_peak_alongs.append((along, side))
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

    amplitude = _number(conditions.get("weld_weave_amplitude_mm"))
    planned_pitch = _number(conditions.get("weld_weave_actual_pitch_mm"))
    weave_tcp = [row for row in tcp
                 if (motion_start is None or float(row["elapsed_s"]) >= motion_start)
                 and (motion_end is None or float(row["elapsed_s"]) <= motion_end)]
    peak_rows = _weave_peaks(weave_tcp, planned_pitch, planned_peak_alongs)
    left_peaks = [-_number(row["signed_weave_offset_mm"]) for side, row in peak_rows
                  if side == "left"]
    right_peaks = [_number(row["signed_weave_offset_mm"]) for side, row in peak_rows
                   if side == "right"]
    same_side_pairs = [(first, second) for first, second in zip(peak_rows, peak_rows[2:])
                       if first[0] == second[0]]
    detected_cycles = []
    pending = None
    for side, row in peak_rows:
        if pending is None or pending[0] == side:
            pending = (side, row)
            continue
        pair = {pending[0]: pending[1], side: row}
        detected_cycles.append(pair)
        pending = None
    periods = [float(b["elapsed_s"])-float(a["elapsed_s"])
               for (_side, a), (_, b) in same_side_pairs]
    pitches = [(_number(b.get("along_mm")) or 0)-(_number(a.get("along_mm")) or 0)
               for (_side, a), (_, b) in same_side_pairs
               if _number(a.get("along_mm")) is not None and _number(b.get("along_mm")) is not None]
    cycle_metrics = []
    for index, cycle in enumerate(detected_cycles):
        left_peak = _number(cycle["left"].get("signed_weave_offset_mm"))
        right_peak = _number(cycle["right"].get("signed_weave_offset_mm"))
        next_cycle = detected_cycles[index + 1] if index + 1 < len(detected_cycles) else None
        cycle_metrics.append({
            "left_peak_mm": left_peak,
            "right_peak_mm": right_peak,
            "full_width_mm": (right_peak - left_peak
                              if left_peak is not None and right_peak is not None else None),
            "pitch_mm": (_number(next_cycle["left"].get("along_mm"))
                         - _number(cycle["left"].get("along_mm"))
                         if next_cycle is not None
                         and _number(next_cycle["left"].get("along_mm")) is not None
                         and _number(cycle["left"].get("along_mm")) is not None else None),
            "period_s": (float(next_cycle["left"].get("elapsed_s"))
                         - float(cycle["left"].get("elapsed_s"))
                         if next_cycle is not None else None),
        })
    measured_dwell = {"left": [], "right": []}
    dwell_samples = {"left": [], "right": []}
    peak_support_samples = {"left": [], "right": []}
    low_speed_samples = {"left": [], "right": []}
    dwell_details = []
    for side, row in peak_rows:
        detail = _contiguous_dwell(weave_tcp, row)
        detail["side"] = side
        dwell_details.append(detail)
        peak_t = float(row["elapsed_s"])
        peak_support_samples[side].append(sum(
            abs(float(sample["elapsed_s"])-peak_t) <= DWELL_PEAK_SUPPORT_WINDOW_S
            for sample in weave_tcp))
        low_speed_samples[side].append(sum(
            abs(float(sample["elapsed_s"])-peak_t) <= DWELL_PEAK_SUPPORT_WINDOW_S
            and _low_speed_at_peak(sample, float(row["signed_weave_offset_mm"]))
            for sample in weave_tcp))
        if detail["valid"]:
            measured_dwell[side].append(detail["dwell_duration_s"])
            dwell_samples[side].append(detail["sample_count"])
    motion_tcp = weave_tcp
    motion_intervals = [float(second["elapsed_s"])-float(first["elapsed_s"])
                        for first, second in zip(motion_tcp, motion_tcp[1:])
                        if float(second["elapsed_s"])-float(first["elapsed_s"]) > 0]
    interval_avg = statistics.fmean(motion_intervals) if motion_intervals else None
    interval_max = max(motion_intervals) if motion_intervals else None
    planned_cycles = _number(conditions.get("weld_weave_cycles"))
    weave_enabled = bool(conditions.get("weld_weave_enabled"))
    insufficient_samples = len(motion_tcp) < 3 or interval_avg is None
    cycle_insufficient = weave_enabled and (insufficient_samples or len(detected_cycles) == 0)
    requested_dwell = min((value for value in (
        _number(conditions.get("weld_weave_left_dwell_s")),
        _number(conditions.get("weld_weave_right_dwell_s")))
        if value is not None and value > 0), default=None)
    expected_dwell_peaks = {
        side: int(planned_cycles) if planned_cycles and planned_cycles > 0 else sum(
            peak_side == side for peak_side, _row in peak_rows)
        for side in ("left", "right")
    }
    valid_dwell_peaks = {side: len(measured_dwell[side]) for side in ("left", "right")}
    dwell_sides = [side for side in ("left", "right")
                   if (_number(conditions.get(f"weld_weave_{side}_dwell_s")) or 0.0) > 0.0]
    dwell_ratios = [valid_dwell_peaks[side] / expected_dwell_peaks[side]
                    for side in dwell_sides if expected_dwell_peaks[side] > 0]
    dwell_near_request = all(
        (mean := _stats(measured_dwell[side])["average"]) is not None
        and DWELL_REQUEST_MATCH_LOWER_RATIO * float(conditions[f"weld_weave_{side}_dwell_s"]) <= mean
        <= DWELL_REQUEST_MATCH_UPPER_RATIO * float(conditions[f"weld_weave_{side}_dwell_s"])
        for side in dwell_sides
    )
    if not dwell_sides:
        dwell_confidence = "NOT_APPLICABLE"
    elif not dwell_ratios or max(dwell_ratios) < 0.4:
        dwell_confidence = "LOW"
    elif min(dwell_ratios) >= 0.75 and dwell_near_request:
        dwell_confidence = "HIGH"
    else:
        dwell_confidence = "MEDIUM"
    reasons = []
    if cycle_insufficient:
        reasons.append("insufficient_cycle_samples")
    elif weave_enabled and planned_cycles and len(detected_cycles) < planned_cycles:
        reasons.append("fewer_peaks_than_planned")
    if weave_enabled and dwell_sides and not any(valid_dwell_peaks.values()) and any(
        detail["sample_count"] < DWELL_MIN_VALID_SAMPLES for detail in dwell_details
    ):
        reasons.append("insufficient_dwell_samples")
    measurement_confidence = (
        "NOT_APPLICABLE" if not weave_enabled else
        "INSUFFICIENT_SAMPLES" if insufficient_samples else
        "LOW" if reasons else
        "HIGH" if interval_avg <= 0.025 and (interval_max or 0.0) <= 0.060 else
        "MEDIUM"
    )
    if dwell_confidence == "LOW" and measurement_confidence in ("HIGH", "MEDIUM"):
        measurement_confidence = "LOW"
    elif dwell_confidence == "MEDIUM" and measurement_confidence == "HIGH":
        measurement_confidence = "MEDIUM"
    weave = {
        "pattern": conditions.get("weld_weave_pattern") if conditions.get("weld_weave_enabled") else None,
        "requested_amplitude_mm": amplitude,
        "requested_pitch_mm": _number(conditions.get("weld_weave_pitch_mm")),
        "planned_pitch_mm": _number(conditions.get("weld_weave_actual_pitch_mm")),
        "crescent_forward_bulge_mm": _number(conditions.get("weld_weave_crescent_bulge_mm")) if conditions.get("weld_weave_pattern") == "crescent" else None,
        "planned_cycle_count": conditions.get("weld_weave_cycles"),
        "axis": conditions.get("weld_weave_axis"),
        "measured_left_amplitude_mm": max(left_peaks) if left_peaks else None,
        "measured_right_amplitude_mm": max(right_peaks) if right_peaks else None,
        "measured_left_amplitude_avg_mm": _stats(left_peaks)["average"],
        "measured_left_amplitude_max_mm": _stats(left_peaks)["max"],
        "measured_right_amplitude_avg_mm": _stats(right_peaks)["average"],
        "measured_right_amplitude_max_mm": _stats(right_peaks)["max"],
        "measured_full_width_mm": _stats(cycle["full_width_mm"] for cycle in cycle_metrics)["average"],
        "measured_full_width_avg_mm": _stats(cycle["full_width_mm"] for cycle in cycle_metrics)["average"],
        "measured_pitch_avg_mm": _stats(pitches)["average"],
        "measured_pitch_std_mm": _stats(pitches)["std"],
        "measured_cycle_count": (len(detected_cycles) if weave_enabled and not cycle_insufficient else None),
        "measurement_reason": ", ".join(reasons) if reasons else None,
        "detected_cycles": cycle_metrics,
        "measured_period_avg_s": _stats(periods)["average"],
        "measured_frequency_hz": 1/_stats(periods)["average"] if periods and _stats(periods)["average"] > 0 else None,
        "left_dwell_requested_s": _number(conditions.get("weld_weave_left_dwell_s")),
        "left_dwell_measured_avg_s": _stats(measured_dwell["left"])["average"],
        "left_dwell_measured_min_s": _stats(measured_dwell["left"])["min"],
        "left_dwell_measured_max_s": _stats(measured_dwell["left"])["max"],
        "left_dwell_measured_std_s": _stats(measured_dwell["left"])["std"],
        "left_dwell_duration_s": measured_dwell["left"],
        "right_dwell_requested_s": _number(conditions.get("weld_weave_right_dwell_s")),
        "right_dwell_measured_avg_s": _stats(measured_dwell["right"])["average"],
        "right_dwell_measured_min_s": _stats(measured_dwell["right"])["min"],
        "right_dwell_measured_max_s": _stats(measured_dwell["right"])["max"],
        "right_dwell_measured_std_s": _stats(measured_dwell["right"])["std"],
        "right_dwell_duration_s": measured_dwell["right"],
        "valid_dwell_peaks": valid_dwell_peaks,
        "expected_dwell_peaks": expected_dwell_peaks,
        "dwell_detection_method": "peak_offset_plateau",
        "dwell_measurement_confidence": dwell_confidence,
        "dwell_peak_details": dwell_details,
        "dwell_thresholds": {
            "offset_tolerance_mm": DWELL_OFFSET_TOLERANCE_MM,
            "seam_progress_tolerance_mm": DWELL_SEAM_PROGRESS_TOLERANCE_MM,
            "max_sample_gap_s": DWELL_MAX_SAMPLE_GAP_S,
            "minimum_valid_samples": DWELL_MIN_VALID_SAMPLES,
            "request_match_ratio": (DWELL_REQUEST_MATCH_LOWER_RATIO,
                                    DWELL_REQUEST_MATCH_UPPER_RATIO),
            "speed_diagnostic_threshold_mm_s": DWELL_TCP_SPEED_THRESHOLD_MM_S,
        },
        "samples_per_detected_dwell": {
            "left": _stats(dwell_samples["left"])["average"],
            "right": _stats(dwell_samples["right"])["average"],
        },
        "peak_support_samples_per_peak": {
            "left": _stats(peak_support_samples["left"])["average"],
            "right": _stats(peak_support_samples["right"])["average"],
        },
        "low_speed_samples_per_peak": {
            "left": _stats(low_speed_samples["left"])["average"],
            "right": _stats(low_speed_samples["right"])["average"],
        },
        "tcp_sample_count": len(motion_tcp),
        "tcp_sample_rate_hz": (1.0 / interval_avg if interval_avg else None),
        "tcp_sample_target_hz": 50.0,
        "tcp_sample_interval_avg_ms": interval_avg * 1000.0 if interval_avg else None,
        "tcp_sample_interval_max_ms": interval_max * 1000.0 if interval_max else None,
        "measurement_confidence": measurement_confidence,
        "detected_peaks": [
            {"side": side, "elapsed_s": float(row["elapsed_s"]),
             "offset_mm": _number(row.get("signed_weave_offset_mm"))}
            for side, row in peak_rows
        ],
        "weave_tracking_error_mm": {
            "average": _stats(tracking_errors)["average"],
            "max": _stats(tracking_errors)["max"],
        },
    }

    tx_frames = document.get("tx_frames", ())
    on_tx = next((row for row in tx_frames
                  if row.get("raw_hex", "") and
                  (int(row["raw_hex"].split()[0], 16) & 0x01)), None)
    raw = bytes.fromhex(on_tx["raw_hex"]) if on_tx else b""
    tx_current = int.from_bytes(raw[14:16], "little") if len(raw) == 55 else None
    tx_hold = raw[16] if len(raw) == 55 else None
    hot_enabled = bool(commanded.get("hot_start_enabled", True))
    requested_hot = (_number(commanded.get("hot_start_current_a"))
                     if hot_enabled else None)
    requested_hold = _number(commanded.get("hot_start_hold_adjustment"))
    rx_hot = _number(echo.get("hot_start_current_a"))
    rx_hold = _number(echo.get("hot_start_hold_adjustment"))
    early = [row for row in samples if hot_enabled and established is not None
             and established <= float(row["elapsed_s"]) <= established+HOT_START_EARLY_WINDOW_S]
    early_peak = max((_number(row.get("feedback_current_a")) or 0 for row in early), default=None)
    steady_current = electrical["current_a"]["steady_main"]
    steady_value = steady_current["average"]
    plateau_rows = []
    detected_duration = None
    if requested_hot is not None and steady_value is not None and requested_hot > steady_value:
        tolerance = max(3.0, requested_hot * 0.03)
        run_start = None
        best_duration = 0.0
        for row in early:
            current = _number(row.get("feedback_current_a")) or 0.0
            t = float(row["elapsed_s"])
            if abs(current-requested_hot) <= tolerance:
                plateau_rows.append(row)
                if run_start is None:
                    run_start = t
                best_duration = max(best_duration, t-run_start)
            else:
                run_start = None
        if best_duration >= HOT_START_MIN_PLATEAU_S:
            detected_duration = best_duration
    tx_status = "SENT" if tx_current is not None and tx_hold is not None else "UNAVAILABLE"
    rx_status = "ECHOED" if (rx_hot is not None and rx_hold is not None
                              and requested_hot is not None
                              and abs(rx_hot-requested_hot) <= 1
                              and requested_hold is not None and rx_hold == requested_hold) else (
        "NOT_ECHOED" if rx_hot is not None or rx_hold is not None else "UNAVAILABLE")
    feedback_status = ("UNCONFIRMED" if not early or requested_hot is None
                       or steady_value is None or requested_hot <= steady_value else
                       "OBSERVED" if detected_duration is not None else "NOT_OBSERVED")
    feedback_detail = ("plateau" if detected_duration is not None else
                       "peak_only" if early_peak is not None and requested_hot is not None
                       and early_peak >= requested_hot * 0.95 else "none")
    hot = {
        "enabled": hot_enabled,
        "requested_boost_percent": (_number(commanded.get("hot_start_percent"))
                                     if hot_enabled else None),
        "requested_current_a": requested_hot,
        "requested_hold_adjustment": requested_hold,
        "encoded_tx_current_a": tx_current,
        "encoded_tx_hold_raw": tx_hold,
        "encoded_tx_hot_raw_hex": raw[14:17].hex(" ").upper() if len(raw) == 55 else None,
        "tx_arc_on_raw_hex": on_tx.get("raw_hex") if on_tx else None,
        "rx_current_a": rx_hot if hot_enabled else None,
        "rx_hold_adjustment": rx_hold if hot_enabled else None,
        "rx_hot_raw_hex": echo.get("hot_start_rx_raw_hex"),
        "feedback_peak_near_establishment_a": early_peak if hot_enabled else None,
        "detected_duration_s": detected_duration,
        "tx_status": tx_status if hot_enabled else "DISABLED",
        "rx_status": rx_status if hot_enabled else "NOT_APPLICABLE",
        "feedback_status": feedback_status if hot_enabled else "NOT_APPLICABLE",
        "feedback_plateau_duration_s": detected_duration,
        "feedback_peak_a": early_peak,
        "startup_peak_current_a": early_peak,
        "startup_peak_voltage_v": _stats(row.get("feedback_voltage_v") for row in early)["max"],
        "startup_plateau_current_avg_a": _stats(row.get("feedback_current_a") for row in plateau_rows)["average"],
        "startup_plateau_duration_s": detected_duration,
        "steady_main_current_avg_a": steady_value,
        "delta_from_main_a": (early_peak - steady_value if early_peak is not None and steady_value is not None else None),
        "requested_delta_a": (requested_hot - requested_main if requested_hot is not None and requested_main is not None else None),
        "feedback_effect": feedback_status if hot_enabled else "NOT_APPLICABLE",
        "feedback_detail": feedback_detail,
        "requested_current_a": requested_hot,
        "requested_boost_percent": (_number(commanded.get("hot_start_percent"))
                                     if hot_enabled else None),
        "tx_current_a": tx_current if hot_enabled else 0,
        "tx_hold_raw": tx_hold,
        "rx_echo_current_a": rx_hot if hot_enabled else None,
        "rx_echo_hold_adjustment": rx_hold if hot_enabled else None,
        "status": feedback_status if hot_enabled else "DISABLED",
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
    software = document.get("software_crater_control") or {}
    software_enabled = bool(commanded.get("software_crater_enabled", False))
    hold_start = _number(software.get("hold_start_elapsed_s"))
    hold_end = _number(software.get("hold_end_elapsed_s"))
    hold_rows = [row for row in samples if hold_start is not None and hold_end is not None
                 and hold_start <= float(row.get("elapsed_s", -1)) <= hold_end]
    target = _number(software.get("target_current_a"))
    tolerance = max(10.0, target * 0.20) if target is not None else None
    near_count = sum(1 for row in hold_rows if target is not None
                     and _number(row.get("feedback_current_a")) is not None
                     and abs(float(row["feedback_current_a"]) - target) <= tolerance)
    plateau_start = None
    plateau_end = None
    longest_plateau = 0.0
    for row in hold_rows:
        current = _number(row.get("feedback_current_a"))
        elapsed = float(row["elapsed_s"])
        if current is not None and target is not None and abs(current - target) <= tolerance:
            if plateau_start is None:
                plateau_start = elapsed
            plateau_end = elapsed
            longest_plateau = max(longest_plateau, plateau_end - plateau_start)
        else:
            plateau_start = plateau_end = None
    actual_hold = _number(software.get("actual_hold_s"))
    observed = (software_enabled and software.get("tx_status") == "SENT"
                and len(hold_rows) >= 3 and near_count >= max(3, math.ceil(len(hold_rows) * 0.6))
                and longest_plateau >= max(0.1, float(commanded.get("software_crater_hold_s", 0.5)) * 0.5)
                and actual_hold is not None and actual_hold >= max(0.0, float(commanded.get("software_crater_hold_s", 0.5)) - 0.05))
    software_summary = {
        "enabled": software_enabled,
        "main_current_a": _number(software.get("main_current_a")),
        "main_voltage_v": _number(software.get("main_voltage_v")),
        "target_current_a": target,
        "target_voltage_v": _number(software.get("target_voltage_v")),
        "ratio_percent": _number(software.get("ratio_percent")),
        "requested_hold_s": _number(software.get("requested_hold_s")),
        "tx_status": software.get("tx_status", "NOT_SENT"),
        "rx_echo": software.get("rx_echo"),
        "actual_current": _stats(row.get("feedback_current_a") for row in hold_rows),
        "actual_voltage": _stats(row.get("feedback_voltage_v") for row in hold_rows),
        "actual_hold_s": actual_hold,
        "hold_sample_count": len(hold_rows),
        "feedback_plateau_duration_s": longest_plateau if hold_rows else None,
        "arc_off_delay_s": (off - hold_end if off is not None and hold_end is not None else None),
        "main_restored": bool(software.get("main_restored")),
        "failure": software.get("failure"),
        "status": ("DISABLED" if not software_enabled else "FAILED" if software.get("status") == "FAILED"
                   else "OBSERVED" if observed else "NOT_OBSERVED"),
    }
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
        "custom_hot_start": custom_summary,
        "crater": {
            "control_source": "external_panel",
            "expected": bool(commanded.get("expect_native_crater", commanded.get("crater_enabled", True))),
            "panel_current_ref_a": _number(commanded.get("crater_panel_current_ref_a", commanded.get("crater_current_a"))),
            "panel_voltage_ref_v": _number(commanded.get("crater_panel_voltage_ref_v", commanded.get("crater_voltage_v"))),
            "panel_time_ref_s": _number(commanded.get("crater_panel_time_ref_s", commanded.get("crater_seconds"))),
            "state2_seen": crater_enter is not None,
            "detected": crater_enter is not None,
            "status": ("NOT_REQUESTED" if not bool(commanded.get(
                "expect_native_crater", commanded.get("crater_enabled", True)))
                       else "OBSERVED" if crater_enter is not None else "NOT_OBSERVED"),
            "enter_delay_s": (crater_enter-off if crater_enter is not None and off is not None else None),
            "duration_s": (crater_exit-crater_enter if crater_enter is not None and crater_exit is not None else None),
            "current": _stats(row.get("feedback_current_a") for row in crater_rows),
            "voltage": _stats(row.get("feedback_voltage_v") for row in crater_rows),
            "wfs_avg_m_min": _stats(row.get("wire_feed_m_min") for row in crater_rows)["average"],
            "arc_off_trigger_remaining_mm": (_number(control.get("actual_remaining_to_goal_m"))*1000
                if _number(control.get("actual_remaining_to_goal_m")) is not None else None),
            "arc_off_to_crater_enter_s": (crater_enter-off if crater_enter is not None and off is not None else None),
            "crater_exit_to_current_extinct_s": (
                events["CURRENT_EXTINCT"]["elapsed_s"]-crater_exit
                if crater_exit is not None and events["CURRENT_EXTINCT"]["elapsed_s"] is not None else None),
            "arc_off_to_wcr_off_s": (
                events["WCR_OFF"]["elapsed_s"]-off
                if off is not None and events["WCR_OFF"]["elapsed_s"] is not None else None),
        },
        "software_crater": software_summary,
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
    software = quality["software_crater"]
    custom = (quality.get("custom_hot_start") or
              document.get("custom_hot_start") or {})
    custom_timing = document.get("custom_hot_start", {}) or {}
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
        *([f"Forward bulge: {value(weave['crescent_forward_bulge_mm'])} mm"]
          if weave['pattern'] == "crescent" else []),
        f"Amplitude req: +/-{value(weave['requested_amplitude_mm'])} mm",
        f"Peak L       : avg {value(weave['measured_left_amplitude_avg_mm'])} / "
        f"max {value(weave['measured_left_amplitude_max_mm'])} mm",
        f"Peak R       : avg {value(weave['measured_right_amplitude_avg_mm'])} / "
        f"max {value(weave['measured_right_amplitude_max_mm'])} mm",
        f"Full width   : avg {value(weave['measured_full_width_avg_mm'])} mm",
        f"Pitch        : req {value(weave['requested_pitch_mm'])} / "
        f"planned {value(weave['planned_pitch_mm'])} / "
        f"measured {value(weave['measured_pitch_avg_mm'])} mm",
        f"Cycles       : planned {weave['planned_cycle_count'] or 'N/A'} / "
        f"measured {weave['measured_cycle_count'] if weave['measured_cycle_count'] is not None else 'N/A'}",
        f"Dwell req L/R: {value(weave['left_dwell_requested_s'])}/"
        f"{value(weave['right_dwell_requested_s'])} s",
        f"Dwell avg L/R: {value(weave['left_dwell_measured_avg_s'], 3)}/"
        f"{value(weave['right_dwell_measured_avg_s'], 3)} s",
        f"Dwell min/max L: {value(weave['left_dwell_measured_min_s'], 3)}/"
        f"{value(weave['left_dwell_measured_max_s'], 3)} s",
        f"Dwell min/max R: {value(weave['right_dwell_measured_min_s'], 3)}/"
        f"{value(weave['right_dwell_measured_max_s'], 3)} s",
        f"Valid dwell peaks: L {weave['valid_dwell_peaks']['left']}/"
        f"{weave['expected_dwell_peaks']['left']}, R {weave['valid_dwell_peaks']['right']}/"
        f"{weave['expected_dwell_peaks']['right']}",
        f"Detection method: {weave['dwell_detection_method']}",
        f"Dwell confidence: {weave['dwell_measurement_confidence']}",
        f"Dwell samples: L {value(weave['samples_per_detected_dwell']['left'], 1)} / "
        f"R {value(weave['samples_per_detected_dwell']['right'], 1)} per peak",
        f"TCP sampling : {weave['tcp_sample_count']} samples / "
        f"{value(weave['tcp_sample_rate_hz'])} Hz / "
        f"avg {value(weave['tcp_sample_interval_avg_ms'])} ms / "
        f"max {value(weave['tcp_sample_interval_max_ms'])} ms",
        f"Confidence   : {weave['measurement_confidence']}"
        + (f" ({weave['measurement_reason']})" if weave['measurement_reason'] else ""),
        "HOT START",
        f"Enabled      : {hot['enabled']}",
        (f"Requested    : +{value(hot['requested_boost_percent'], 1)}% / "
         f"{value(hot['requested_current_a'], 0)} A"
         if hot['enabled'] else "Requested    : NOT_REQUESTED"),
        (f"TX           : {value(hot['tx_current_a'], 0)} A / "
         f"raw {value(hot['tx_hold_raw'], 0)} / {hot['tx_status']}"
         if hot['enabled'] else "TX Current   : 0 A"),
        (f"RX Echo      : {value(hot['rx_echo_current_a'], 0)} A / "
         f"{value(hot['rx_echo_hold_adjustment'], 0)} / {hot['rx_status']}"
         if hot['enabled'] else "RX Echo      : N/A"),
        (f"Feedback     : {hot['feedback_effect']} ({hot['feedback_detail']}) / "
         f"plateau {value(hot['startup_plateau_current_avg_a'], 1)} A for "
         f"{value(hot['startup_plateau_duration_s'], 3)} s"
         if hot['enabled'] else "Feedback     : NOT_APPLICABLE"),
        f"Status       : {hot['status']}",
        "CUSTOM HOT START (MOTION HOLD)",
        f"Enabled      : {custom.get('enabled', False)}",
        f"Hold req     : {value(custom.get('requested_hold_s'), 3)} s",
        f"Hold actual  : {value(custom.get('actual_hold_s'), 3)} s",
        f"TCP drift    : {value(custom.get('max_tcp_drift_mm'), 3)} mm",
        f"Current      : avg/min/max {value((custom.get('current_a') or {}).get('average'))}/"
        f"{value((custom.get('current_a') or {}).get('min'))}/"
        f"{value((custom.get('current_a') or {}).get('max'))} A",
        f"Voltage      : avg/min/max {value((custom.get('voltage_v') or {}).get('average'))}/"
        f"{value((custom.get('voltage_v') or {}).get('min'))}/"
        f"{value((custom.get('voltage_v') or {}).get('max'))} V",
        f"RX samples   : {custom.get('sample_count', 0)}",
        f"ARC→begin    : {value(custom_timing.get('arc_recognized_to_begin_s'), 3)} s",
        f"End→motion   : {value(custom_timing.get('end_to_motion_start_s'), 3)} s",
        f"Status       : {custom.get('status', 'DISABLED')}",
        "CRATER",
        "Control source: welder external panel",
        f"Expected     : {crater['expected']}",
        f"State2 seen  : {crater['state2_seen']}",
        f"Panel ref    : {value(crater['panel_current_ref_a'])} A / "
        f"{value(crater['panel_voltage_ref_v'])} V / "
        f"{value(crater['panel_time_ref_s'], 2)} s",
        f"Status       : {crater['status']}",
        f"Enter delay  : {value(crater['enter_delay_s'], 3)} s",
        f"RX duration  : {value(crater['duration_s'], 3)} s",
        f"Actual I     : avg/min/max {value(crater['current']['average'])}/"
        f"{value(crater['current']['min'])}/{value(crater['current']['max'])} A",
        f"Actual V     : avg/min/max {value(crater['voltage']['average'])}/"
        f"{value(crater['voltage']['min'])}/{value(crater['voltage']['max'])} V",
        "SOFTWARE CRATER",
        f"Enabled      : {software['enabled']}",
        f"Main setpoint: {value(software['main_current_a'], 0)} A / {value(software['main_voltage_v'], 1)} V",
        f"Crater cmd   : {value(software['target_current_a'], 0)} A / {value(software['target_voltage_v'], 1)} V",
        f"Ratio        : {value(software['ratio_percent'], 1)} %",
        f"Requested hold: {value(software['requested_hold_s'], 3)} s",
        f"Setpoint TX  : {software['tx_status']}",
        f"RX echo      : {software['rx_echo'] or 'N/A'}",
        f"Actual I     : avg/min/max {value(software['actual_current']['average'])}/"
        f"{value(software['actual_current']['min'])}/{value(software['actual_current']['max'])} A",
        f"Actual V avg : {value(software['actual_voltage']['average'])} V",
        f"Actual hold  : {value(software['actual_hold_s'], 3)} s",
        f"I plateau    : {value(software['feedback_plateau_duration_s'], 3)} s",
        f"ARC OFF delay: {value(software['arc_off_delay_s'], 3)} s",
        f"Status       : {software['status']}",
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
