"""Direct 1G scan (no Hongik server): lift, joint move to the taught scan start, constant TCP-speed line
to the taught scan end while recording sensor profiles with the robot pose, lift.

Runs on a worker thread of the GUI host. Motions use the existing runtime calls
(run_sequence_cartesian_motion / run_sequence_named_pose), so STOP / EMERGENCY STOP cancel them as usual.
Never touches the welder.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import threading

from geometry_msgs.msg import Pose
import yaml

from construct_robot.core.precision_sensing_inspection import pose_from_mapping
from construct_robot.core.scan_motion import (
    MAX_SCAN_ORIENTATION_CHANGE_DEG,
    SCAN_GROUP,
    ScanPose,
    build_scan_steps,
    check_scan_line,
    pose_move_step,
)
from construct_robot.io.teaching_yaml import load_initial_state_yaml


def load_scan_pose(path):
    group, names, positions, tcp = load_initial_state_yaml(path)
    if group != SCAN_GROUP:
        raise ValueError(f"{path}: scan poses must be right-arm poses (got {group})")
    if tcp is None:
        raise ValueError(f"{path}: no TCP pose; re-teach it")
    return ScanPose(tuple(names), tuple(positions), tcp)


class DirectScanController:
    def __init__(self, host, session_factory):
        self.host = host
        # (sensor, meta) -> session with start() (sensor recording, returns when ready) and stop() -> result dict.
        # Production: nodes.scan_record_process.RecordSession (a separate process: the busy GUI process loses frames / TF).
        self.session_factory = session_factory
        self._lock = threading.Lock()
        self._busy = False
        self._cancelled = threading.Event()

    @property
    def busy(self):
        return self._busy

    def cancel(self):
        self._cancelled.set()
        self.host.node.cancel_active_motion()

    def run(self, *, start_path, end_path, clearance_m, tcp_speed_m_s, velocity_scale, sensor,
            on_progress, on_finished, return_pass=False, extra_meta=None):
        """Worker thread. sensor: ip, port, calib_file, gain, backup_dir (1. CONNECTION settings)."""
        with self._lock:
            if self._busy:
                raise RuntimeError("A direct scan is already running")
            self._busy = True
            self._cancelled.clear()
        node = self.host.node
        live = node.scanner_live
        post = self.host.post
        result = {"status": "ERROR", "message": "", "folder": None, "stats": {}}
        session = None
        try:
            start, end = load_scan_pose(start_path), load_scan_pose(end_path)
            length, tilt, change = check_scan_line(start, end)
            if change > MAX_SCAN_ORIENTATION_CHANGE_DEG:
                post(on_progress, f"Note: scan end orientation differs by {change:.1f}° · "
                                  "the line keeps the scan start orientation")
            sensing = getattr(node, "precision_sensing_test", None)
            if sensing is not None and sensing.busy:
                raise RuntimeError("A Hongik scan owns the sensor")
            if live.streaming:                       # the sensor accepts one client: the recorder process takes it
                post(on_progress, "Stopping the RViz live view (the scan recorder takes the sensor)...")
                live.stop()
            current = node._path_start_tcp_pose(SCAN_GROUP)
            steps = build_scan_steps(current, start, end, clearance_m=clearance_m, tcp_speed_m_s=tcp_speed_m_s,
                                     velocity_scale=velocity_scale, return_pass=return_pass)
            meta = self._meta(start, end, length, tilt, change, tcp_speed_m_s, sensor)
            meta["passes"] = 2 if return_pass else 1
            meta.update(extra_meta or {})
            session = None
            for index, scan_step in enumerate(steps, 1):
                if self._cancelled.is_set():
                    raise RuntimeError("scan canceled")
                # One recording spans consecutive recorded lines (forward + return pass).
                last_record = scan_step.record and (index == len(steps) or not steps[index].record)
                if scan_step.record and session is None:
                    post(on_progress, f"[{index}/{len(steps)}] starting the scan recorder (laser ON)...")
                    session = self.session_factory(sensor, meta)
                    session.start()
                post(on_progress, f"[{index}/{len(steps)}] {scan_step.name}"
                                  + (f" · recording at {tcp_speed_m_s * 1000.0:.1f} mm/s over "
                                     f"{length * 1000.0:.0f} mm" if scan_step.record else "") + " (ROBOT MOVES)")
                ok = False
                try:
                    runner = (node.run_sequence_cartesian_motion if scan_step.step["type"] == "motion"
                              else node.run_sequence_named_pose)
                    ok, message = runner(scan_step.step, True)
                finally:
                    if session is not None and (last_record or not ok):
                        post(on_progress, f"[{index}/{len(steps)}] saving the scan (sensor parameters restored)...")
                        recorded = session.stop()
                        result["stats"] = dict(recorded.get("stats") or {})
                        result["folder"] = recorded.get("folder")
                        result["sensor"] = recorded.get("sensor")
                        if recorded.get("status") != "OK":
                            result["message"] = f"recorder: {recorded.get('message')}"
                        stopped = session
                        session = None
                    else:
                        stopped = None
                if not ok:
                    partial = " (partial data saved)" if result["folder"] else ""
                    raise RuntimeError(f"{scan_step.name}: {message}{partial}")
                if stopped is not None and result["folder"] is None:
                    raise RuntimeError(f"scan not recorded: {result['message'] or 'no frames'}")
            stats = result["stats"]
            result.update(status="OK", message=(
                f"Scan done · {stats.get('frames', 0)} frames, {stats.get('points', 0)} points, "
                f"TF failures {stats.get('tf_failures', 0)}, sensor drops "
                f"{((result.get('sensor') or {}).get('driver_stats') or {}).get('queue_drops', 0)}"))
        except Exception as error:
            result["message"] = str(error)
            if session is not None:                   # e.g. canceled between the forward and the return pass
                recorded = session.stop()
                result["folder"] = recorded.get("folder") or result["folder"]
                result["stats"] = dict(recorded.get("stats") or result["stats"])
        finally:
            with self._lock:
                self._busy = False
            post(on_finished, result)

    @staticmethod
    def _meta(start, end, length, tilt, change, speed, sensor):
        def pose(p):
            return {"position_m": [p.tcp.position.x, p.tcp.position.y, p.tcp.position.z],
                    "orientation_xyzw": [p.tcp.orientation.x, p.tcp.orientation.y,
                                         p.tcp.orientation.z, p.tcp.orientation.w],
                    "joints_rad": dict(zip(p.joint_names, p.positions))}
        return {"scan_start_tcp": pose(start), "scan_end_tcp": pose(end),
                "line_length_m": length, "line_tilt_deg": tilt, "end_orientation_change_deg": change,
                "tcp_speed_m_s": speed, "line_orientation": "scan start orientation (constant)",
                "sensor": {k: sensor.get(k) for k in ("ip", "port", "calib_file", "gain")},
                "stamp_source": "sensor frame receive time (no latency correction)",
                "recorder": "separate process (construct_robot.nodes.scan_record_process)"}


def latest_scan_dir(root=None):
    from pathlib import Path
    from construct_robot.io.scan_paths import scan_results_root
    root = Path(root).expanduser() if root else scan_results_root()
    folders = sorted(p for p in root.glob("*_scan") if (p / "scan.npz").is_file())
    if not folders:
        raise ValueError(f"no recorded scan in {root}")
    return folders[-1]


ESTIMATORS = ("kiro", "hongik")       # first = default (KIRO groove estimator; Hongik's 1G estimate optional)


def check_scan_coverage(scan_dir):
    """Fast (~1 s) check that the recording passed both plate ends; (True, margins) or (False, operator reason)."""
    import numpy as np
    from construct_robot.core.groove_estimate import GrooveEstimateError, check_scan_covers_ends
    data = np.load(Path(scan_dir).expanduser() / "scan.npz")
    try:
        return True, check_scan_covers_ends(data["points_world_m"], data["points_frame_index"],
                                            data["T_world_optical"][:, :3, 3])
    except GrooveEstimateError as error:
        return False, str(error)


def run_estimate(estimator, scan_dir, *, calib_file=None, job=None, on_line=None, weld_direction=None):
    """The selected estimator in its own process; both return the same RESULT dict shape and write the same files."""
    if estimator == "hongik":
        return run_hongik_estimate(scan_dir, calib_file=calib_file, job=job, on_line=on_line, weld_direction=weld_direction)
    if estimator != "kiro":
        raise ValueError(f"unknown estimator {estimator!r}")
    cmd = ["-m", "construct_robot.nodes.kiro_scan_estimate", str(scan_dir)]
    if calib_file:
        cmd += ["--calib", str(calib_file)]
    if weld_direction:
        cmd += ["--weld-direction", weld_direction if weld_direction in ("E->S", "S->E") else "E->S"]
    return _run_estimator_process(cmd, None, on_line, 600.0)


def run_hongik_estimate(scan_dir, *, calib_file=None, job=None, timeout_s=1800.0, on_line=None, weld_direction=None):
    """Hongik 1G estimate of one recorded scan in a separate process (Hongik's numpy / scipy environment).
    on_line(text): each progress line of the tool as it runs. Returns the tool's RESULT dict
    (status OK | REFUSED | ERROR, ends, passes start / goal, out_dir)."""
    import sys
    from construct_robot.nodes.hongik_scan_estimate import hongik_env
    cmd = [sys.executable, "-m", "construct_robot.nodes.hongik_scan_estimate", str(scan_dir)]
    if calib_file:
        cmd += ["--calib", str(calib_file)]
    if job:
        cmd += ["--job", str(job)]
    if weld_direction:
        cmd += ["--weld-direction", str(weld_direction)]
    return _run_estimator_process(cmd[1:], hongik_env(), on_line, timeout_s)


def _run_estimator_process(args, env, on_line, timeout_s):
    import json
    import subprocess
    import sys
    import threading
    cmd = [sys.executable] + list(args)
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    progress = []

    def read_progress():
        for line in process.stderr:
            line = line.rstrip()
            if not line or line.lstrip().startswith(("Warning", "warnings.", "a = np.", "b = np.")) or "RuntimeWarning" in line:
                continue
            progress.append(line)
            if on_line is not None:
                on_line(line)
    reader = threading.Thread(target=read_progress, daemon=True)
    reader.start()
    try:
        stdout, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        return {"status": "ERROR", "message": f"estimator timed out after {timeout_s:.0f} s"}
    reader.join(timeout=2.0)
    result = next((json.loads(line[7:]) for line in reversed(stdout.splitlines()) if line.startswith("RESULT ")), None)
    if result is None:
        return {"status": "ERROR", "message": "estimator gave no result: " + " | ".join(progress[-3:])}
    result["log"] = progress[-8:]
    return result


def estimate_summary(result):
    """Operator text: ends and every pass start / goal TCP position (World mm)."""
    def mm(values, scale=1.0):
        return "(" + ", ".join(f"{float(v) * scale:.1f}" for v in values) + ")"
    who = "KIRO estimate" if result.get("estimator") == "kiro" else "Hongik estimate"
    lines = [f"{who}: {result.get('status')}" + (f" · {result['message']}" if result.get("message") else "")]
    groove = result.get("groove") or {}
    if groove:
        lines.append(f"  groove: length {groove.get('length_mm', float('nan')):.1f} mm · root gap "
                     f"{groove.get('root_gap_mm', float('nan')):.2f} mm · depth {groove.get('depth_mm', float('nan')):.2f} mm · "
                     f"bevel {groove.get('bevel_deg_from_vertical', float('nan')):.1f} deg · {groove.get('sections')} sections")
    qc = result.get("qc") or []
    if qc:
        bad = [r for r in qc if r.get("ok") is False]
        lines.append("  specimen check: " + ("all within tolerance" if not bad else "OUT OF TOLERANCE " + ", ".join(
            f"{r['name']} {r['measured']:.2f} (design {r['design']:.1f} ± {r['tol']:.1f})" for r in bad)))
    for name, end in (result.get("ends") or {}).items():
        if not end:
            lines.append(f"  end {name}: not found")
            continue
        point = end.get("point_mm") or [v * 1000.0 for v in (end.get("point_m") or [])]
        sigma = end.get("sigma_mm", None if end.get("sigma_m") is None else end["sigma_m"] * 1000.0)
        lines.append(f"  end {name}: {mm(point)} mm · sigma {'unknown' if sigma is None else f'{sigma:.2f} mm'}"
                     f" · {'two-sided' if end.get('two_sided') else 'NOT identified (one-sided)'}")
    for row in result.get("passes") or []:
        if "aim_start_mm" in row:
            lines.append(f"  pass {row['pass']}: start wire tip {mm(row['aim_start_mm'])} → goal wire tip {mm(row['aim_goal_mm'])} mm World")
            continue
        lines.append(f"  pass {row.get('k')} ({row.get('role')}): start TCP {mm(row['start_World_m'], 1000.0)} → "
                     f"goal TCP {mm(row['goal_World_m'], 1000.0)} mm World")
    if result.get("out_dir"):
        lines.append(f"  files: {result['out_dir']} (kiro/pass_<k>.yaml = start / goal TCP poses)")
    return "\n".join(lines)


def estimate_poses(scan_dir):
    """Pass start / goal TCP poses of <scan_dir>/hongik_estimate/kiro/pass_<k>.yaml -> [(label, Pose)]."""
    from pathlib import Path
    import yaml
    from geometry_msgs.msg import Pose
    poses = []
    for path in sorted((Path(scan_dir).expanduser() / "hongik_estimate" / "kiro").glob("pass_[0-9]*.yaml")):
        if path.stem.endswith("_info"):
            continue
        doc = yaml.safe_load(path.read_text())
        for end in ("start", "goal"):
            values = doc[f"current_{end}"]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(values["position_m"][a]) for a in "xyz")
            (pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w) = (float(values["orientation_xyzw"][a]) for a in "xyzw")
            poses.append((f"Pass {doc['pass']} {end}", pose))
    return poses


@dataclass(frozen=True)
class HongikPass:
    source: str
    sha256: str
    number: int
    start: object
    goal: object

    def require_unchanged(self):
        if hashlib.sha256(Path(self.source).read_bytes()).hexdigest() != self.sha256:
            raise ValueError("Pass file changed after loading; load it again before planning or execution")


def latest_pass_file(root=None):
    root = Path(root) if root else Path.home() / "hongik" / "runs"
    return next((str(p.resolve()) for p in sorted(root.glob("*/kiro/pass_1.yaml"), reverse=True)), "")


def load_pass_file(path):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    if not isinstance(doc, dict) or doc.get("schema") != "xy_endpoint_review_v1":
        raise ValueError("Expected a Hongik xy_endpoint_review_v1 pass YAML")
    if doc.get("frame_id") != "World":
        raise ValueError("Pass motion requires World coordinates; no implicit frame conversion")
    number = int(doc.get("pass", 0))
    if number < 1:
        raise ValueError("Pass number must be positive")
    start, goal = [pose_from_mapping(doc.get(f"current_{end}"), pass_number=number,
                                    point=end.upper(), kind="tcp", frame="World") for end in ("start", "goal")]
    return HongikPass(str(path), hashlib.sha256(raw).hexdigest(), number, start, goal)


def _ros_pose(value):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = value.position
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = value.quaternion
    return pose


def pass_motion_steps(loaded, current, mode, *, velocity_scale, tcp_speed_m_s):
    """Preserve file endpoints and orientations; no correction, weave, offsets or welder rows."""
    if mode not in ("START", "GOAL", "START → GOAL"):
        raise ValueError("Select START, GOAL or START → GOAL")
    if not math.isfinite(tcp_speed_m_s) or not 0.0001 <= tcp_speed_m_s <= 0.5:
        raise ValueError("TCP speed must be 0.1..500 mm/s")
    if not math.isfinite(velocity_scale) or not 0 < velocity_scale <= 1:
        raise ValueError("Velocity scale must be greater than 0 and at most 1")
    start, goal = _ros_pose(loaded.start), _ros_pose(loaded.goal)
    target = goal if mode == "GOAL" else start
    steps = [pose_move_step(current, target, velocity_scale=velocity_scale,
                            label=f"Hongik P{loaded.number} original {mode.split(' → ')[0]}", keep_orientation=False)]
    if mode == "START → GOAL":
        line = pose_move_step(start, goal, velocity_scale=velocity_scale,
                              label=f"Hongik P{loaded.number} original START → GOAL", keep_orientation=False)
        line.update(tcp_speed_m_s=tcp_speed_m_s, linear_motion_profile=True)
        steps.append(line)
    # pose_move_step defaults to slot 1 for a single move. These rows must be
    # sequential: approach START completely before traversing START -> GOAL.
    for slot, step in enumerate(steps, 1):
        step["parallel_slot"] = slot
    return steps
