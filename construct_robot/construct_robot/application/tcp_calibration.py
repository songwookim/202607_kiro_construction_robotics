"""Wire-tip calibration workflow (Scan tab): collect TCP captures with the wire tip on one fixed point at different
torch orientations, solve the wire tip in the TCP frame, store it for the scan targets (no robot motion; the URDF and
the weld teaching are not changed).

scan_teaching/tcp_calibration_samples.yaml   the captures of the current calibration
scan_teaching/wire_tip_offset.yaml           the result: wire tip in right_manipulator_ee_point [m]
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from construct_robot.core.tcp_calibration import pivot_calibration


def _teaching_dir():
    from construct_robot.application.scan_correction import scan_teaching_dir
    return scan_teaching_dir()


def samples_path():
    return _teaching_dir() / "tcp_calibration_samples.yaml"


def offset_path():
    return _teaching_dir() / "wire_tip_offset.yaml"


def load_samples(path=None):
    path = Path(path or samples_path())
    doc = (yaml.safe_load(path.read_text()) if path.is_file() else None) or {}
    return [(np.asarray(s["position_m"], float), np.asarray(s["orientation_xyzw"], float)) for s in doc.get("samples") or []]


def add_sample(position, quat_xyzw, source="", path=None):
    path = Path(path or samples_path())
    doc = (yaml.safe_load(path.read_text()) if path.is_file() else None) or {}
    samples = doc.get("samples") or []
    samples.append({"position_m": [float(v) for v in position], "orientation_xyzw": [float(v) for v in quat_xyzw],
                    "source": str(source), "saved": datetime.now().isoformat(timespec="seconds")})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"note": "wire tip held on ONE fixed point; only the torch orientation changes",
                                    "samples": samples}, sort_keys=False))
    return len(samples)


def clear_samples(path=None):
    path = Path(path or samples_path())
    if path.is_file():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive = _teaching_dir() / "history" / f"tcp_calibration_samples_{stamp}.yaml"
        archive.parent.mkdir(parents=True, exist_ok=True)
        path.replace(archive)


def compute_and_save(path=None, out_path=None):
    """Solve the wire tip from the samples and store it; returns the result (raises when not determined)."""
    samples = load_samples(path)
    result = pivot_calibration(samples)
    doc = {"format_version": 1, "frame": "right_manipulator_ee_point",
           "tip_tcp_m": [float(v) for v in result["tip_tcp_m"]],
           "computed": datetime.now().isoformat(timespec="seconds"), "samples": len(samples),
           "rms_mm": result["rms_m"] * 1e3, "residuals_mm": [float(v) * 1e3 for v in result["residuals_m"]],
           "spread_deg": result["spread_deg"], "point_world_m": [float(v) for v in result["point_world_m"]],
           "note": "used by the scan targets instead of Hongik's tcp_to_aim along the torch axis; recompute the "
                   "reference correction after changing it"}
    out = Path(out_path or offset_path())
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    result["file"] = str(out)
    if out_path is None:
        # the correction holds wire-tip offsets of the old model: recompute it on its own scan (same references)
        from construct_robot.application.scan_correction import refresh_correction_for_wire_tip
        try:
            refreshed = refresh_correction_for_wire_tip()
        except (OSError, ValueError) as error:
            refreshed, result["correction_error"] = None, str(error)
        result["correction_refreshed"] = None if refreshed is None else refreshed.get("source_scan")
    return result


def load_offset(path=None):
    """The calibrated wire tip in the TCP frame [m] (np.array 3) or None."""
    path = Path(path or offset_path())
    if not path.is_file():
        return None
    return np.asarray(yaml.safe_load(path.read_text())["tip_tcp_m"], float)


# ---- automatic plane-touch calibration (Scan tab: robot touches a flat surface with the wire, Fastech DI0)

def save_touch_result(result, touches, out_path=None):
    """Store the plane-touch wire tip like compute_and_save (wire_tip_offset.yaml) and refresh the correction."""
    doc = {"format_version": 1, "frame": "right_manipulator_ee_point", "method": "automatic plane touch (Fastech DI0)",
           "tip_tcp_m": [float(v) for v in result["tip_tcp_m"]],
           "computed": datetime.now().isoformat(timespec="seconds"), "samples": len(touches),
           "rms_mm": result["rms_m"] * 1e3, "residuals_mm": [float(v) * 1e3 for v in result["residuals_m"]],
           "spread_deg": result["spread_deg"], "tip_sigma_mm": [float(v) * 1e3 for v in result["tip_sigma_m"]],
           "plane_normal": [float(v) for v in result["plane_normal"]],
           "plane_d_m": result["plane_d_m"],
           "touches": [{"kind": k, "label": label, "position_m": [float(v) for v in p], "orientation_xyzw": [float(v) for v in q]}
                       for k, label, p, q in touches],
           "note": "used by the scan targets; the reference correction is refreshed automatically"}
    out = Path(out_path or offset_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        history = _teaching_dir() / "history"
        history.mkdir(parents=True, exist_ok=True)
        (history / f"wire_tip_offset_{datetime.now().strftime('%Y%m%d-%H%M%S')}.yaml").write_text(out.read_text())
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    result = dict(result, file=str(out))
    if out_path is None:
        from construct_robot.application.scan_correction import refresh_correction_for_wire_tip
        try:
            refreshed = refresh_correction_for_wire_tip()
            result["correction_refreshed"] = None if refreshed is None else refreshed.get("source_scan")
        except (OSError, ValueError) as error:
            result["correction_error"] = str(error)
    return result


class TouchCalibrationRun:
    """Worker-thread routine: per target lift, move to the approach pose, descend (World -Z, touch guarded, slow)
    until Fastech DI0, latch the TCP at the edge, lift. Touch sensing (DO0) is on only during the run."""

    LIFT_M = 0.015
    MAX_DESCENT_M = 0.030
    DESCENT_SPEED_M_S = 0.004

    def __init__(self, host):
        self.host = host
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        self.host.node.cancel_active_motion()

    def run(self, tip_model, velocity_scale, on_progress, max_tilt_deg=35.0):
        import copy
        from construct_robot.core.scan_motion import pose_move_step
        from construct_robot.core.tcp_calibration import plane_touch_calibration, touch_calibration_plan
        from construct_robot.nodes.weld_runtime_node import FASTECH_TOUCH_OUTPUT_PORT
        host, node = self.host, self.host.node
        group = "right_manipulator"
        if node.node_touch_input_states.get("right") is None:
            raise RuntimeError("Fastech DI0 state is unknown (Fastech connected?)")
        if node.node_touch_input_states.get("right"):
            raise RuntimeError("Fastech DI0 is already ON: lift the wire off the surface first")
        start = node._current_tcp_pose(group)
        q_start = (start.orientation.x, start.orientation.y, start.orientation.z, start.orientation.w)
        plan = touch_calibration_plan((start.position.x, start.position.y, start.position.z), q_start, tip_model,
                                      max_tilt_deg=max_tilt_deg)
        ok, message = host._set_fastech_output_sync(FASTECH_TOUCH_OUTPUT_PORT, True)
        if not ok:
            raise RuntimeError(f"cannot enable touch sensing (DO{FASTECH_TOUCH_OUTPUT_PORT}): {message}")
        if getattr(host, "hicomm_client", None) is not None:
            host.hicomm_client.set_arc(False)
        touches = []

        def move(target, label, **flags):
            current = node._current_tcp_pose(group)
            step = pose_move_step(current, target, velocity_scale=velocity_scale, label=label, keep_orientation=False)
            step.update(flags)
            done, text = node.run_sequence_cartesian_motion(step, True)
            if self.cancelled:
                raise RuntimeError("touch calibration stopped")
            return done, text

        def lifted(pose):
            p = copy.deepcopy(pose)
            p.position.z += self.LIFT_M
            return p
        try:
            for index, target in enumerate(plan, 1):
                on_progress(f"[{index}/{len(plan)}] {target['label']} · lift and move (ROBOT MOVES)")
                ok, text = move(lifted(node._current_tcp_pose(group)), "touch calibration lift",
                                touch_guard=False, allow_initial_touch_motion=True)
                if not ok:
                    raise RuntimeError(f"lift: {text}")
                approach = copy.deepcopy(start)
                approach.position.x, approach.position.y, approach.position.z = (float(v) for v in target["tcp"])
                approach.orientation.x, approach.orientation.y, approach.orientation.z, approach.orientation.w = (
                    float(v) for v in target["quat"])
                ok, text = move(approach, "touch calibration approach", touch_guard=False)
                if not ok:
                    raise RuntimeError(f"approach: {text}")
                bottom = copy.deepcopy(approach)
                bottom.position.z -= self.MAX_DESCENT_M
                on_progress(f"[{index}/{len(plan)}] {target['label']} · descending until the wire touches (DI0)")
                ok, text = move(bottom, "touch calibration descent", touch_guard=True, continue_after_touch=True,
                                tcp_speed_m_s=self.DESCENT_SPEED_M_S)
                edge = getattr(node, "touch_guard_edge_pose", None)
                if not ok or edge is None or not node.node_touch_input_states.get("right"):
                    raise RuntimeError(f"no wire contact within {self.MAX_DESCENT_M * 1e3:.0f} mm ({text}); "
                                       "start 5-15 mm above a flat surface")
                touches.append((target["kind"], target["label"],
                                (edge.position.x, edge.position.y, edge.position.z),
                                (edge.orientation.x, edge.orientation.y, edge.orientation.z, edge.orientation.w)))
                on_progress(f"[{index}/{len(plan)}] contact at z {edge.position.z * 1e3:.2f} mm")
                ok, text = move(lifted(node._current_tcp_pose(group)), "touch calibration retract",
                                touch_guard=False, allow_initial_touch_motion=True)
                if not ok:
                    raise RuntimeError(f"retract: {text}")
            ok, text = move(lifted(start), "touch calibration return", touch_guard=False)
        finally:
            host._set_fastech_output_sync(FASTECH_TOUCH_OUTPUT_PORT, False)
        result = plane_touch_calibration([t[2] for t in touches], [t[3] for t in touches], [t[0] for t in touches])
        return save_touch_result(result, touches)
