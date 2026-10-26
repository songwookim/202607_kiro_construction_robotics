"""Operator parameters of the Scan + 4 Passes cycle (4-PASS PARAMS tab).

The cycle used to freeze fixed values (PASS_PROFILES, PASS4_WEAVE, START_TOUCH). These are still the defaults;
the tab edits, validates and saves them (config/scan_four_pass_parameters.yaml) and the cycle builder takes them.
Touch-probe bounds are the runtime's own (scan_start_touch.seek_start_touch refuses anything outside them).
"""

import copy
from datetime import datetime
import math
from pathlib import Path

import yaml

from construct_robot.io.teaching_yaml import teaching_config_dir

PASSES = (1, 2, 3, 4)
WEAVE_PATTERNS = ("sine", "crescent", "circle")
CLEANER_MODES = ("cut_only", "full")
PASS_KEYS = ("current_a", "voltage_v", "speed_mm_s", "weave", "weave_pattern", "weave_amplitude_mm",
             "weave_pitch_mm", "weave_left_dwell_s", "weave_right_dwell_s", "start_touch")
# START_TOUCH in m / m/s; the tab edits mm / mm/s
TOUCH_LIMITS_MM = {"speed_mm_s": (0.0, 0.5), "max_descent_mm": (0.0, 10.0)}
# hot start / crater per pass: the Welding tab's recipe keys and ranges (weld_config.validate_digital_weld_settings);
# None range = on/off. Used only when params["arc"]["use_table"]; otherwise the Build-time Welding tab recipe.
ARC_LIMITS = {
    "hot_start_enabled": None, "hot_start_percent": (0.0, 100.0), "hot_start_hold_adjustment": (-15.0, 15.0),
    "custom_hot_start_enabled": None, "custom_hot_start_percent": (0.0, 100.0),
    "custom_hot_start_hold_s": (0.01, 5.0),
    "software_crater_enabled": None, "software_crater_ratio_percent": (20.0, 50.0),
    "software_crater_voltage_v": (10.0, 40.0), "software_crater_hold_s": (0.0, 5.0),
}


def default_params():
    """The values the cycle used before the tab existed (PASS_PROFILES / PASS4_WEAVE / START_TOUCH)."""
    from construct_robot.application.scan_start_touch import START_TOUCH
    from construct_robot.application.scan_weld_cycle import PASS4_WEAVE, PASS_PROFILES
    passes = {}
    for k, (current, voltage, speed) in zip(PASSES, PASS_PROFILES):
        passes[k] = {"current_a": float(current), "voltage_v": float(voltage), "speed_mm_s": float(speed),
                     "weave": k == 4, "weave_pattern": PASS4_WEAVE["weave_pattern"],
                     "weave_amplitude_mm": PASS4_WEAVE["weave_amplitude_mm"],
                     "weave_pitch_mm": PASS4_WEAVE["weave_pitch_mm"],
                     "weave_left_dwell_s": PASS4_WEAVE["weave_left_dwell_s"],
                     "weave_right_dwell_s": PASS4_WEAVE["weave_right_dwell_s"], "start_touch": True}
    from construct_robot.core.weld_config import DEFAULT_DIGITAL_WELD_SETTINGS
    arc = {key: DEFAULT_DIGITAL_WELD_SETTINGS[key] for key in ARC_LIMITS}
    return {"passes": passes,
            "start_touch": {"speed_mm_s": START_TOUCH["speed_m_s"] * 1e3,
                            "max_descent_mm": START_TOUCH["max_descent_m"] * 1e3},
            "arc": {"use_table": False, "passes": {k: dict(arc) for k in PASSES}},
            "cleaner": "cut_only"}


def _number(value, name, low, high, *, low_open=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not a number ({value!r})") from None
    if not math.isfinite(value) or value > high or (value <= low if low_open else value < low):
        raise ValueError(f"{name} must be in {'(' if low_open else '['}{low:g}..{high:g}]")
    return value


def validate_params(params):
    """Normalized copy (all passes and keys present, numbers checked) or ValueError naming the field."""
    base = default_params()
    params = params or {}
    out = {"passes": {}, "start_touch": {}, "cleaner": str(params.get("cleaner", base["cleaner"]))}
    given = {int(k): v for k, v in (params.get("passes") or {}).items()}
    for k in PASSES:
        row = dict(base["passes"][k], **(given.get(k) or {}))
        p = f"P{k} "
        out["passes"][k] = {
            "current_a": _number(row["current_a"], p + "current [A]", 30, 400),
            "voltage_v": _number(row["voltage_v"], p + "voltage [V]", 10, 45),
            "speed_mm_s": _number(row["speed_mm_s"], p + "speed [mm/s]", 0.1, 100),
            "weave": _flag(row["weave"]),
            "weave_pattern": str(row["weave_pattern"]),
            "weave_amplitude_mm": _number(row["weave_amplitude_mm"], p + "weave amplitude [mm]", 0.1, 50),
            "weave_pitch_mm": _number(row["weave_pitch_mm"], p + "weave pitch [mm]", 0.1, 100),
            "weave_left_dwell_s": _number(row["weave_left_dwell_s"], p + "left dwell [s]", 0, 10),
            "weave_right_dwell_s": _number(row["weave_right_dwell_s"], p + "right dwell [s]", 0, 10),
            "start_touch": _flag(row["start_touch"]),
        }
        if out["passes"][k]["weave_pattern"] not in WEAVE_PATTERNS:
            raise ValueError(f"{p}weave pattern must be one of {', '.join(WEAVE_PATTERNS)}")
    touch = dict(base["start_touch"], **(params.get("start_touch") or {}))
    for key, (low, high) in TOUCH_LIMITS_MM.items():
        out["start_touch"][key] = _number(touch[key], f"touch {key}", low, high, low_open=True)
    if out["cleaner"] not in CLEANER_MODES:
        raise ValueError(f"cleaner must be one of {', '.join(CLEANER_MODES)}")
    arc_in = params.get("arc") or {}
    arc_given = {int(k): v for k, v in (arc_in.get("passes") or {}).items()}
    out["arc"] = {"use_table": _flag(arc_in.get("use_table", base["arc"]["use_table"])), "passes": {}}
    for k in PASSES:
        row = dict(base["arc"]["passes"][k], **(arc_given.get(k) or {}))
        clean = {}
        for key, limits in ARC_LIMITS.items():
            if limits is None:
                clean[key] = _flag(row[key])
            else:
                clean[key] = _number(row[key], f"P{k} {key}", *limits, low_open=key == "software_crater_hold_s")
        clean["hot_start_hold_adjustment"] = int(round(clean["hot_start_hold_adjustment"]))
        out["arc"]["passes"][k] = clean
    return out


def _flag(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def arc_settings(params, k):
    """Hot start / crater recipe keys for pass k ({} = keep the Build-time Welding tab recipe)."""
    params = validate_params(params)
    return dict(params["arc"]["passes"][int(k)]) if params["arc"]["use_table"] else {}


def touch_policy(params):
    """START_TOUCH-shaped policy (m, m/s) from the validated params."""
    t = validate_params(params)["start_touch"]
    return {"speed_m_s": t["speed_mm_s"] * 1e-3, "max_descent_m": t["max_descent_mm"] * 1e-3}


def params_path():
    return teaching_config_dir() / "scan_four_pass_parameters.yaml"


def load_params(path=None):
    """Saved params (validated) or the defaults when nothing is saved."""
    path = Path(path or params_path())
    if not path.is_file():
        return default_params()
    return validate_params(yaml.safe_load(path.read_text()) or {})


def save_params(params, path=None):
    params = validate_params(params)
    doc = copy.deepcopy(params)
    doc["passes"] = {int(k): v for k, v in doc["passes"].items()}
    doc["arc"]["passes"] = {int(k): v for k, v in doc["arc"]["passes"].items()}
    doc = {"schema": "construct_robot_scan_four_pass_parameters_v1",
           "saved": datetime.now().astimezone().isoformat(timespec="seconds"), **doc}
    path = Path(path or params_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def summary(params):
    """One-line operator text, e.g. '190A/24.5V/5 → ... · P4 sine ±4/4mm · touch P1-P4 0.5mm/s'."""
    params = validate_params(params)
    rows = []
    for k in PASSES:
        r = params["passes"][k]
        text = f"P{k} {r['current_a']:g}A/{r['voltage_v']:g}V/{r['speed_mm_s']:g}mm/s"
        if r["weave"]:
            text += (f" {r['weave_pattern']} ±{r['weave_amplitude_mm']:g}/{r['weave_pitch_mm']:g}mm "
                     f"dwell {r['weave_left_dwell_s']:g}/{r['weave_right_dwell_s']:g}s")
        rows.append(text)
    if params["arc"]["use_table"]:
        for k, a in params["arc"]["passes"].items():
            hot = (f"hot {a['hot_start_percent']:g}%" if a["hot_start_enabled"] else "hot off") + \
                (f"+custom {a['custom_hot_start_percent']:g}%/{a['custom_hot_start_hold_s']:g}s"
                 if a["custom_hot_start_enabled"] else "")
            crater = (f"crater {a['software_crater_ratio_percent']:g}%/{a['software_crater_voltage_v']:g}V/"
                      f"{a['software_crater_hold_s']:g}s" if a["software_crater_enabled"] else "crater off")
            rows[k - 1] += f" [{hot}, {crater}]"
    touched = [f"P{k}" for k in PASSES if params["passes"][k]["start_touch"]]
    t = params["start_touch"]
    return (" → ".join(rows) + f" · touch {','.join(touched) or 'none'} {t['speed_mm_s']:g}mm/s "
            f"≤{t['max_descent_mm']:g}mm · cleaner {params['cleaner']} · hot start/crater "
            f"{'per-pass table' if params['arc']['use_table'] else 'Welding tab recipe'}")
