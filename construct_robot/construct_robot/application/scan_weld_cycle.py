"""One measured four-pass cycle; Build/Plan never submits a scanning goal.

The visible pass rows freeze teaching, recipe and weave inputs. Execute expands
them only after this execution's PrecisionSensing result has passed validation.
The existing managed weld executor owns every ARC, crater and cleaner operation.
"""

import copy
from dataclasses import replace
import math
from pathlib import Path

import yaml

from construct_robot.application.weld_sequence_builder import build_weld_scenario
from construct_robot.application.scan_start_touch import configure_start_touch
from construct_robot.core.precision_sensing_inspection import finite_values, seam_tilt_deg, MAX_1G_SEAM_TILT_DEG
from construct_robot.core.scan_reference import plate_weave_direction, tcp_from_aim
from construct_robot.core.sequence_model import validate_managed_weld_sequence
from construct_robot.core.weld_config import validate_digital_weld_settings


CYCLE_ID = "scan_four_pass_clean_v1"
SINGLE_SCAN_CYCLE_ID = "scan_four_pass_single_scan_v1"
NO_SCAN_CYCLE_ID = "scan_four_pass_saved_v1"
LOCAL_1G = "local_1g"        # scan source: KIRO local 1G scan (C -> D -> C) + offline estimate + correction
PASS_PROFILES = ((190, 24.5, 5.0), (230, 26.5, 7.0),
                 (230, 26.5, 7.0), (220, 26.0, 5.0))
PASS4_WEAVE = dict(weave_pattern="sine", weave_amplitude_mm=4.0, weave_pitch_mm=4.0,
                   weave_left_dwell_s=0.2, weave_right_dwell_s=0.2)


def export_post_weld_ply(host, folder, number):
    """Export after recording, directly into this execution's collected result folder when available."""
    from construct_robot.io.scan_cloud import load_scan_cloud, write_cloud_ply
    from construct_robot.io.scan_cycle_log import record_cycle_event
    journal = getattr(host, "_scan_cycle_journal", None)
    bundle = getattr(journal, "bundle_directory", None)
    output = (Path(bundle) / "npz" / f"pass_{int(number)}_rescan.ply" if bundle is not None
              else Path(folder) / "scan.ply")
    points = load_scan_cloud(folder)
    write_cloud_ply(output, points)
    if not record_cycle_event(host, "artifact", kind="post_weld_ply", pass_number=int(number), path=str(output),
                              source_scan=str(folder), frame="World", units="m", points=len(points)):
        raise RuntimeError(host._scan_cycle_log_error)
    return output


def build_scan_weld_cycle(scan, post_scan, pass_inputs, cleaner_steps, *, rescan_after_passes=True,
                          traverse_enabled=True, params=None, start_pass=1):
    """Freeze one cycle, preserving every recipe field except main I/V.
    params: scan_cycle_params (per-pass I / V / speed / weave / start touch); None = the fixed defaults.
    start_pass: No Scan only - resume at pass 2 / 3 / 4 (the earlier passes are already welded)."""
    from construct_robot.application.scan_cycle_params import arc_settings, touch_policy, validate_params
    params = validate_params(params)
    touch = touch_policy(params)
    if len(pass_inputs) != 4 or not cleaner_steps:
        raise ValueError("Four pass teachings and a complete Torch Clean sequence are required")
    saved = scan.get("source") == "saved_scan"
    start_pass = int(start_pass)
    if start_pass not in (1, 2, 3, 4):
        raise ValueError("start pass must be 1..4")
    if start_pass != 1 and not saved:
        raise ValueError("starting at a later pass is only for No Scan (a scan cannot see the welded groove)")
    if saved:
        scan = dict(scan, start_pass=start_pass)
        if rescan_after_passes:
            raise ValueError("No Scan cycle cannot contain rescans")
        validate_saved_scan(scan)
    elif scan.get("source") == LOCAL_1G:
        for key in ("start_path", "end_path"):
            if not Path(scan[key]).is_file():
                raise ValueError(f"Teach the local scan poses C / D first: missing {scan[key]}")
    else:
        a, b = finite_values(scan["a"], 3), finite_values(scan["b"], 3)
        if scan["frame"] != "World" or seam_tilt_deg(a, b)[0] > MAX_1G_SEAM_TILT_DEG:
            raise ValueError("PrecisionSensing requires World-frame horizontal 1G seam points")
    rows = []
    cycle_id = NO_SCAN_CYCLE_ID if saved else (CYCLE_ID if rescan_after_passes else SINGLE_SCAN_CYCLE_ID)

    def add_templates(templates, phase, number):
        slot_map = {}
        for index, template in enumerate(templates):
            old_slot = template.get("parallel_slot", index + 1)
            if old_slot not in slot_map:
                slot_map[old_slot] = 1 + max((r["parallel_slot"] for r in rows), default=0)
            row = copy.deepcopy(template)
            row.update(scan_weld_cycle_id=cycle_id, scan_cycle_phase=phase,
                       scan_cycle_pass=number, parallel_slot=slot_map[old_slot])
            rows.append(row)

    add_templates([dict(type="saved_scan_load" if saved else "precision_sensing_scan", scan=scan)], "precision", 0)
    # All passes share the Build-time hot-start/crater recipe (taken from pass 1, so selecting another Builder row
    # later changes nothing) unless the 4-PASS PARAMS per-pass hot start / crater table is switched on.
    recipe = copy.deepcopy(pass_inputs[0][0].settings)
    for k, (scenario, motion) in enumerate(pass_inputs, 1):
        if k < start_pass:
            continue
        p = params["passes"][k]
        current, voltage, speed, weave = p["current_a"], p["voltage_v"], p["speed_mm_s"], p["weave"]
        settings = validate_digital_weld_settings(dict(recipe, **arc_settings(params, k), current_a=current,
                                                       voltage_tenths=round(voltage * 10)))
        scenario = replace(copy.deepcopy(scenario), settings=settings,
                           weld_tcp_speed_mm_s=speed, weave_enabled=weave)
        if weave:
            scenario = replace(scenario, **{name: p[name] for name in PASS4_WEAVE})
        if weave and (scan.get("source") == LOCAL_1G or scan.get("plate_normal_world") is not None):
            normal = scan.get("plate_normal_world")
            if normal is None:
                raise ValueError(f"Scan + 4 Passes needs the scanned plate normal for the pass-{k} weave")
            direction = plate_weave_direction(
                [getattr(scenario.endpoints.start.position, a) for a in "xyz"],
                [getattr(scenario.endpoints.goal.position, a) for a in "xyz"], normal)
            scenario = replace(scenario, weave_transverse_vector=tuple(float(v) for v in direction))
        motion = copy.deepcopy(motion)
        build_weld_scenario(scenario, motion, base_slot=1, scenario_id=f"check_pass_{k}")
        add_templates([dict(type="scan_weld_pass", planning_group="right_manipulator",
                            io_backend=scenario.touch_io_backend, scenario=scenario,
                            motion=motion, settings=settings, weld_tcp_speed_mm_s=speed,
                            start_touch_probe=(copy.deepcopy(touch) if traverse_enabled and p["start_touch"]
                                               else None),
                            weld_weave_enabled=weave)], "weld", k)
        if k < 4:
            add_templates(cleaner_steps, "clean", k)
        if rescan_after_passes:
            add_templates([dict(type="hongik_post_weld_scan", scan=post_scan)], "raw", k)
    validate_scan_weld_cycle(rows)
    return rows


def validate_scan_weld_cycle(rows):
    """Reject missing/reordered stages, selected-row execution and shared slots."""
    if (not rows or rows[0].get("scan_weld_cycle_id") not in (CYCLE_ID, SINGLE_SCAN_CYCLE_ID, NO_SCAN_CYCLE_ID)
            or any(row.get("scan_weld_cycle_id") != rows[0]["scan_weld_cycle_id"] for row in rows)):
        raise ValueError("Run the entire Scan + 4 Passes sequence")
    rescans = rows[0]["scan_weld_cycle_id"] == CYCLE_ID
    initial_type = "saved_scan_load" if rows[0]["scan_weld_cycle_id"] == NO_SCAN_CYCLE_ID else "precision_sensing_scan"
    first = int((rows[0].get("scan") or {}).get("start_pass", 1)) if initial_type == "saved_scan_load" else 1
    if first not in (1, 2, 3, 4):
        raise ValueError("Invalid No Scan start pass")
    phases = []
    last_slot = 0
    previous = None
    for row in rows:
        phase = (row.get("scan_cycle_phase"), row.get("scan_cycle_pass"))
        slot = int(row["parallel_slot"])
        if slot < last_slot or (slot == last_slot and not (phase[0] == "clean" and phase == previous)):
            raise ValueError("Scan/weld stages must run in separate ordered slots")
        if phase != previous:
            phases.append(phase)
        previous, last_slot = phase, slot
        expected_type = {"precision": initial_type, "weld": "scan_weld_pass",
                         "raw": "hongik_post_weld_scan"}.get(phase[0])
        if expected_type and row["type"] != expected_type:
            raise ValueError("Scan cycle stage type was changed; rebuild the sequence")
        if phase[0] == "clean" and not row.get("torch_clean_scenario"):
            raise ValueError("Scan cycle requires the taught Torch Clean steps")
    expected = [("precision", 0)]
    for k in range(first, 5):
        expected.append(("weld", k))
        if k < 4:
            expected.append(("clean", k))
        if rescans:
            expected.append(("raw", k))
    if phases != expected or sum(r["type"] == "scan_weld_pass" for r in rows) != 5 - first or sum(r["type"] == "hongik_post_weld_scan" for r in rows) != (4 if rescans else 0) or sum(r["type"] == initial_type for r in rows) != 1:
        if first != 1:
            raise ValueError(f"Required order: saved targets → P{first} → clean → ... → P4")
        raise ValueError("Required order: scan → P1 → clean → P2 → clean → P3 → clean → P4" +
                         (" with one raw scan after each pass" if rescans else " (initial scan only)"))


def saved_scan_config(folder):
    """Freeze the exact saved teaching files; never fit or apply correction twice."""
    import hashlib
    folder = Path(folder).expanduser().resolve()
    hashes = {}
    for k in range(1, 5):
        path = folder / f"pass_{k}.yaml"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(source="saved_scan", saved_folder=str(folder), saved_hashes=hashes)


def validate_saved_scan(scan):
    current = saved_scan_config(scan["saved_folder"])
    if current["saved_hashes"] != scan.get("saved_hashes"):
        raise ValueError("Saved scan teaching changed after Build; rebuild before execution")


def result_run_dir(inspection):
    document = inspection.document
    if int(document.get("error_code", -1)) != 0 or "UNRESOLVED" in str(document.get("error_message", "")).upper():
        raise ValueError(f"PrecisionSensing did not resolve the passes: {document.get('error_message', '')}")
    if inspection.frame != "World" or len(document.get("paths", ())) != 4:
        raise ValueError("PrecisionSensing must return four World-frame paths")
    data_file = Path(str(document.get("data_file_path", ""))).expanduser()
    if not data_file.is_file():
        raise ValueError("PrecisionSensing result files are unavailable; no welding started")
    return data_file.parent


def _shift_entry(entry, old_end, new_end):
    if entry is None:
        return None
    group, names, joints, pose = copy.deepcopy(entry)
    for axis in "xyz":
        setattr(pose.position, axis, getattr(pose.position, axis)
                + getattr(new_end.position, axis) - getattr(old_end.position, axis))
    return group, names, joints, pose


def measured_pass_input(scenario, inspection, run_dir, number):
    """New measured wire tips, frozen taught torch attitudes and wait offsets.

    The action waypoints describe the wire tips, not the arm TCP. Use the
    delivered contact-tip-to-wire length exactly once; never add weld_link's
    20 mm again. Wait/finish offsets come from this pass's current teaching.
    """
    info_path = run_dir / "kiro" / f"pass_{number}_info.yaml"
    info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    frames = info["derived"]["frames"]
    if frames.get("tcp_frame") != "right_manipulator_ee_point":
        raise ValueError(f"Pass {number}: unexpected TCP frame")
    length = float(frames["tcp_to_aim_point_mm"]) / 1000.0
    if not math.isfinite(length) or length <= 0:
        raise ValueError(f"Pass {number}: invalid TCP-to-wire offset")
    wire = [p for p in inspection.poses if p.pass_number == number and p.kind == "wire"]
    if len(wire) < 2:
        raise ValueError(f"Pass {number}: incomplete measured path")
    endpoints = scenario.endpoints
    start, goal = copy.deepcopy(endpoints.start), copy.deepcopy(endpoints.goal)
    taught_direction = [getattr(goal.position, a) - getattr(start.position, a) for a in "xyz"]
    measured_direction = [b - a for a, b in zip(wire[0].position, wire[-1].position)]
    if sum(a * b for a, b in zip(taught_direction, measured_direction)) <= 0:
        raise ValueError("Scan direction differs from pass teaching; check A/B and the selected pass folder")
    for pose, aim in ((start, wire[0].position), (goal, wire[-1].position)):
        q = [getattr(pose.orientation, a) for a in "xyzw"]
        values = finite_values(tcp_from_aim(aim, q, length), 3)
        pose.position.x, pose.position.y, pose.position.z = values
    transverse = scenario.weave_transverse_vector
    if scenario.weave_enabled:
        normal = inspection.sections.get("frame", {}).get("n")
        if normal is None:
            raise ValueError(f"Hongik result has no measured plate normal for the pass-{number} weave")
        transverse = tuple(float(v) for v in plate_weave_direction(
            [getattr(start.position, a) for a in "xyz"], [getattr(goal.position, a) for a in "xyz"], normal))
    return replace(
        scenario,
        endpoints=replace(endpoints, start=start, goal=goal,
                          start_wait=_shift_entry(endpoints.start_wait, endpoints.start, start),
                          start_source=f"PrecisionSensing pass {number}", goal_source=f"PrecisionSensing pass {number}"),
        goal_wait=_shift_entry(scenario.goal_wait, endpoints.goal, goal),
        finish=_shift_entry(scenario.finish, endpoints.goal, goal),
        weld_wait=_shift_entry(scenario.weld_wait, endpoints.start, start),
        start_wait_tcp_override=None,
        weave_transverse_vector=transverse,
    )


def _pose_at(template, tcp, quat):
    pose = copy.deepcopy(template)
    pose.position.x, pose.position.y, pose.position.z = finite_values(tcp, 3)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(v) for v in quat)
    return pose


def _entry_at(entry, tcp, quat):
    """Teaching entry moved to a new TCP pose; its joints stay as the IK seed (resolve_target_tcp_ik)."""
    if entry is None:
        return None
    group, names, joints, pose = copy.deepcopy(entry)
    return group, names, joints, _pose_at(pose, tcp, quat)


def local_pass_input(scenario, targets, number):
    """Pass inputs from the local scan's four-pass targets (scan_four_pass.four_pass_targets: corrected wire tips,
    taught pass orientations, waits): start / goal and all four wait poses are replaced."""
    endpoints = scenario.endpoints
    start = _pose_at(endpoints.start, *targets["start"])
    goal = _pose_at(endpoints.goal, *targets["goal"])
    transverse = scenario.weave_transverse_vector
    if scenario.weave_enabled:
        geometry = targets.get("weave_geometry") or {}
        if geometry.get("normal_world") is None:
            raise ValueError(f"fresh scan has no plate normal for the pass-{number} weave")
        transverse = tuple(float(v) for v in plate_weave_direction(
            [getattr(start.position, a) for a in "xyz"], [getattr(goal.position, a) for a in "xyz"],
            geometry["normal_world"]))
    return replace(
        scenario,
        endpoints=replace(endpoints, start=start, goal=goal,
                          start_wait=_entry_at(endpoints.start_wait, *targets["start_wait"]),
                          start_source=f"local 1G scan pass {number}", goal_source=f"local 1G scan pass {number}"),
        goal_wait=_entry_at(scenario.goal_wait, *targets["goal_wait"]),
        finish=_entry_at(scenario.finish, *targets["weld_finish"]),
        weld_wait=_entry_at(scenario.weld_wait, *targets["weld_wait"]),
        start_wait_tcp_override=None,
        weave_transverse_vector=transverse,
    )


def expand_scan_weld_cycle(rows, inspection=None, local=None):
    """Plan uses frozen teaching; Execute requires the fresh action inspection (Hongik server) or the local scan's
    targets (local = {"scan_dir", "targets": {k: four_pass_targets row}})."""
    validate_scan_weld_cycle(rows)
    run_dir = result_run_dir(inspection) if inspection is not None else None
    if local is not None:
        run_dir = Path(local["scan_dir"])
    expanded = []
    for row in rows[1:]:
        if row["type"] == "scan_weld_pass":
            k = row["scan_cycle_pass"]
            scenario = copy.deepcopy(row["scenario"])
            if local is not None:
                scenario = local_pass_input(scenario, local["targets"][k], k)
            elif inspection is not None:
                scenario = measured_pass_input(scenario, inspection, run_dir, k)
            steps = build_weld_scenario(scenario, row["motion"], base_slot=1,
                                        scenario_id=f"scan_cycle_pass_{k}").steps
            if row.get("start_touch_probe"):
                configure_start_touch(steps, row["start_touch_probe"])
            for step in steps:
                if step["type"] == "named_pose":
                    step["resolve_target_tcp_ik"] = True
                step.update(scan_cycle_phase="weld", scan_cycle_pass=k)
                weaving = row.get("weld_weave_enabled", k == 4)
                if weaving and rows[0]["scan"].get("source") in (LOCAL_1G, "saved_scan"):
                    geometry = (local["targets"][k].get("weave_geometry") or {}) if local is not None else {}
                    step["weld_weave_reference"] = "scan_plate_plane"
                    step["weld_plate_normal_world"] = geometry.get("normal_world", rows[0]["scan"].get("plate_normal_world"))
                elif weaving and inspection is not None:
                    step["weld_weave_reference"] = "scan_plate_plane"
                    step["weld_plate_normal_world"] = inspection.sections["frame"]["n"]
        else:
            steps = [copy.deepcopy(row)]
            if row["type"] == "hongik_post_weld_scan":
                steps[0]["scan"] = dict(row["scan"], run_dir=str(run_dir or ""),
                                         after_pass=row["scan_cycle_pass"], dry_run=False)
        # Retain only intentional parallelism *within* each generated template.
        # Cleaner rows are already globally ordered, so each keeps its own slot.
        slot_map = {}
        for step in steps:
            old = step.get("parallel_slot", 1)
            if old not in slot_map:
                slot_map[old] = 1 + max((r["parallel_slot"] for r in expanded), default=0)
            step["parallel_slot"] = slot_map[old]
            step["scan_weld_cycle_id"] = rows[0]["scan_weld_cycle_id"]
            step["scan_cycle_run_dir"] = str(run_dir or "teaching preview")
            expanded.append(step)
    validate_managed_weld_sequence(expanded, require_complete=True)
    return expanded
