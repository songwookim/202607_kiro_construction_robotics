"""Load delivered Hongik results without touching production teaching data."""

from dataclasses import dataclass
import json
from pathlib import Path

import yaml

from construct_robot.core.precision_sensing_inspection import inspect_paths, pose_from_mapping, tcp_wire_error_mm


@dataclass(frozen=True)
class SensingInspection:
    document: dict
    frame: str
    poses: tuple
    notes: tuple
    sections: dict
    source: str


def inspect_document(document, frame, *, run_dir=None, source="action result"):
    sections = {}
    notes = []
    if run_dir is not None:
        sections_file = Path(run_dir) / "sections.json"
        if sections_file.is_file():
            sections = json.loads(sections_file.read_text(encoding="utf-8"))
            actual_frame = sections.get("base_frame")
            if actual_frame and actual_frame != frame:
                raise ValueError(f"Result frame {actual_frame!r} differs from selected frame {frame!r}")
    poses, path_notes = inspect_paths(document, frame)
    poses = list(poses)
    notes.extend(path_notes)
    if run_dir is not None and poses:
        for k in range(1, len(document["paths"]) + 1):
            path = Path(run_dir) / "kiro" / f"pass_{k}.yaml"
            if not path.is_file():
                notes.append(f"Pass {k}: TCP file unavailable; wire surface poses only")
                continue
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if data.get("schema") != "xy_endpoint_review_v1" or int(data.get("pass", 0)) != k:
                raise ValueError(f"Wrong schema/pass number: {path}")
            info_file = path.with_name(f"pass_{k}_info.yaml")
            frames = {}
            if info_file.is_file():
                info = yaml.safe_load(info_file.read_text(encoding="utf-8"))
                frames = info.get("derived", {}).get("frames", {})
            wire_rows = [p for p in poses if p.pass_number == k and p.kind == "wire"]
            for point, key in (("START", "current_start"), ("GOAL", "current_goal")):
                tcp = pose_from_mapping(data[key], pass_number=k, point=point, kind="tcp", frame=data["frame_id"])
                poses.append(tcp)
                if tcp.frame == frame and "tcp_to_aim_point_mm" in frames:
                    error = tcp_wire_error_mm(tcp, wire_rows[0 if point == "START" else -1],
                                              frames["tcp_to_aim_point_mm"])
                    notes.append(f"Pass {k} {point}: TCP + offset → wire error={error:.6f} mm" +
                                 (" · CHECK: files disagree (>0.01 mm)" if error > 0.01 else ""))
                else:
                    notes.append(f"Pass {k} {point}: TCP/wire cross-check unavailable (frame/offset metadata)")
    if run_dir is None:
        notes.append("Run files unavailable: TCP poses / section measurements not loaded")
    if "UNRESOLVED" in str(document.get("error_message", "")):
        notes.append("CHECK: PASS_UNRESOLVED is returned as code 0; precision targets remain unresolved")
    notes.append("Finite values / quaternion / spacing checks do not establish physical measurement accuracy.")
    return SensingInspection(document, frame, tuple(poses), tuple(notes), sections, source)


def load_result(path, frame):
    path = Path(path).expanduser()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Expected a PrecisionSensing result YAML mapping")
    return inspect_document(document, frame, run_dir=path.parent, source=str(path))
