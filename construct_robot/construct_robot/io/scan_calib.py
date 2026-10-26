"""Profiler calibration files (Hongik EOAT calib YAML, laser_link -> optical) and what a recording used.

Hongik's delivered file: eoat_description/calib/eoat_kiro_01.yaml (the lab hand-eye, "verify on KIRO's robot").
KIRO's on-site file: construct_description/config/scan_teaching/eoat_kiro_handeye.yaml = the same file with the
hand-eye fixed from measured wire-tip positions; it is the default once it exists.
"""

import json
import os
import re
from pathlib import Path

import yaml

KIRO_CALIB_NAME = "eoat_kiro_handeye.yaml"


def hongik_calib_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory("eoat_description"), "calib", "eoat_kiro_01.yaml")
    except Exception:
        return ""
    return path if os.path.isfile(path) else ""


def kiro_calib_path():
    from construct_robot.io.teaching_yaml import teaching_config_dir
    return Path(teaching_config_dir()) / "scan_teaching" / KIRO_CALIB_NAME


def default_calib_path():
    """KIRO's hand-eye file when it exists, else Hongik's delivered one ("" when neither is found)."""
    kiro = kiro_calib_path()
    return str(kiro) if kiro.is_file() else hongik_calib_path()


def load_calib_pose(path):
    """{"file", "xyz" [m], "rpy" [rad]} of a calib YAML (laser_link -> optical)."""
    doc = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if doc.get("parent_frame") != "laser_link":
        raise ValueError(f"{path} must store laser_link -> optical (got {doc.get('parent_frame')!r})")
    return {"file": str(path), "xyz": [float(v) for v in doc["xyz"]], "rpy": [float(v) for v in doc["rpy"]]}


def recorded_calib(scan_dir):
    """The calib a recording was made with: the snapshot in scan.json (calib), else the file it names (older
    recordings; that file must not have changed since). None when unknown."""
    path = Path(scan_dir).expanduser() / "scan.json"
    if not path.is_file():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    snapshot = meta.get("calib")
    if isinstance(snapshot, dict) and "xyz" in snapshot and "rpy" in snapshot:
        return {"file": snapshot.get("file", ""), "xyz": [float(v) for v in snapshot["xyz"]],
                "rpy": [float(v) for v in snapshot["rpy"]]}
    sensor = meta.get("sensor")
    if isinstance(sensor, str):                     # some recordings stored the dict as its repr
        try:
            import ast
            sensor = ast.literal_eval(sensor)
        except (ValueError, SyntaxError):
            sensor = None
    calib_file = (sensor or {}).get("calib_file") if isinstance(sensor, dict) else None
    if calib_file and Path(calib_file).is_file():
        return load_calib_pose(calib_file)
    return None


def write_handeye_calib(base_path, xyz, note_lines, out_path=None):
    """Copy base_path (comments, g1 block) with xyz replaced, a provenance header on top; returns the path."""
    text = Path(base_path).read_text(encoding="utf-8")
    new_line = "xyz: [" + ", ".join(f"{float(v):.6f}" for v in xyz) + "]"
    text, count = re.subn(r"^xyz:\s*\[[^\]]*\]", new_line, text, count=1, flags=re.M)
    if count != 1:
        raise ValueError(f"{base_path}: no 'xyz: [...]' line")
    header = "".join(f"# {line}\n" for line in note_lines)
    out = Path(out_path or kiro_calib_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "#\n" + text, encoding="utf-8")
    return out
