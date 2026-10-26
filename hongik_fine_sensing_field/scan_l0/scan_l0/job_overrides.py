"""Run-time job overrides (user 2026-10-13: "parameters must be adjustable per run - site conditions differ").

An override is a dotted path into the job document (the job YAML) and a value (a YAML value), e.g.
    precision.end_position_mm = 0.10            weld_plan.stick_out_mm = 17.5
    weld_design.end_inset_mm.goal = 79.7        weld_plan.travel_angle_deg = [10, 10, 10]
    weld_plan.speed_mps.1 = 0.008               (a list entry by its index)
A nested mapping is the same thing written as a tree ({weld_design: {end_inset_mm: {goal: 79.7}}}): mappings merge key by key,
anything else (numbers, text, lists) replaces the value at its path.

Layers - applied in this order right after the job file is loaded (scan_l0 / the server's build_rough_job), before anything is planned:
  1. 'server parameter'  the auto-teach server's ROS parameter job_overrides (a YAML / JSON mapping): site-level overrides for every
                         goal of either action - the way KIRO's site adjusts (KIRO's PrecisionSensing goal is KIRO's and has no field),
  2. 'goal'              the AutoTeach goal's job_overrides (a JSON object; client --set KEY=VALUE / --set-file FILE) on top.

Validation - from the job schema and the existing values only (no new limits):
  path   the first segment is a block the run reads as input (BLOCKS); the key must exist in the job document, in the welding
         position's job template (jobs/groove_10T_<pos>.yaml) or among the keys the block's parser accepts (ScanJob fields, GrooveSpec
         fields, the Precision targets, WeldProcess fields, the WeldDesign keys, the per-job auto_teach keys); a deeper segment must
         exist in the value it indexes (a mapping key, a list index). Keys a run does not take from the job file are refused with
         the reason (JOB_NOT_INPUTS: the goal's rough points and position, the platform's frames / gravity / safety values, the
         robot model's torch, the values the run measures).
  value  the effective block goes through the parser the job file goes through (WeldProcess.from_dict, weld_design_from_doc,
         Precision.from_doc, GrooveSpec.from_dict, Tolerances.from_qc (1G), AutoTeachConfig.from_params) and its message names
         the key. ScanJob fields have no parse-time check (a job file's are checked by ScanJob.validate in the run's gates, after
         the safety policy): an override of one keeps the kind of the existing value (number / yes-no / text / a list of the same
         length).
  speed  job.speed_mps (and the AutoTeach goal's measure_speed_mps, client --measure-speed) is the run's measurement speed: the 1G
         core accepts it inside the noise priors' calibrated speed range only (scan_core.noise_priors.check_measure_speed, in its
         gates before any motion).
Record: one entry per override {path, old, new, source, old_from} (old = the value the run would have used: the job file's, an
earlier layer's, or the parser's default). The effective document carries them in meta.job_overrides with the loaded file in
meta.job_file; the run writes both to its run log and saves the document as job_effective.yaml (write_effective_job).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import yaml

__all__ = ["BLOCKS", "NOT_OVERRIDABLE", "JOB_NOT_INPUTS", "SOURCES", "OverrideError", "parse_mapping_text", "flatten", "yaml_value",
           "parse_set_items", "merge", "to_goal_text", "goal_overrides", "apply_overrides", "apply_layers", "known_keys",
           "record_lines", "summary_line", "records_of", "job_file_of", "write_effective_job", "EFFECTIVE_JOB_NAME",
           "COMMON_KEYS"]

SOURCES = ("server parameter", "goal")
BLOCKS = ("job", "specimen", "qc_targets", "precision", "weld_plan", "weld_design", "auto_teach")
NOT_OVERRIDABLE = {
    "teach": "the teaching record (the two taught points and how they were taught), not a run input",
    "meta": "the job file's record (notes, saved time), not a run input",
    "start_pose": "the taught start posture (joint angles) - teach it again (hongik_auto_teach_client --teach)",
    "experiments": "the post-run experiments block of a lab job file - edit a copy of the job file",
}
# ScanJob fields a run does not take from the job file (an override would be replaced silently): refused with the reason
JOB_NOT_INPUTS = {
    "start": "the rough start point comes from the goal (the taught / KIRO point)",
    "end": "the rough end point comes from the goal (the taught / KIRO point)",
    "position": "the welding position comes from the goal",
    "orientation": "the welding position comes from the goal",
    "base_frame": "the frame is the platform's base frame",
    "gravity_axis_base": "the gravity axis comes from the platform (TF)",
    "surface_normal": "the plate normal is derived from the rough points and gravity",
    "specimen_top_z_m": "the plate top is measured by the run",
    "table_z_m": "the support is measured by the run",
    "plate_top_measured_m": "measured by the run",
    "plate_plane_measured": "measured by the run",
    "floor_z_m": "measured by the run",
    "safe_clearance_m": "derived each run by the safety policy from the platform config (scan_l0.safety)",
    "standoff_min_m": "derived each run by the safety policy from the platform config (scan_l0.safety)",
    "pose_tol_m": "derived each run by the safety policy from the platform config (scan_l0.safety)",
    "max_laser_tilt_deg": "the platform config's safety value",
    "torch_protrusion_m": "the torch keypoint comes from the robot model / the platform config's safety value",
    "torch_lateral_m": "the torch keypoint comes from the robot model",
}
# mapping-valued fields whose keys are free (equipment settings, provenance texts)
FREE_FORM = {("weld_plan", "equipment"), ("weld_plan", "provenance"), ("weld_design", "provenance")}
EFFECTIVE_JOB_NAME = "job_effective.yaml"
# the keys the client's --help lists (the common site adjustments; every key the schema knows is accepted)
COMMON_KEYS = (
    ("precision.end_position_mm", "1-sigma target of the weld line ends [mm] (done test)"),
    ("precision.root_position_mm", "1-sigma target of the root position per section [mm]"),
    ("precision.length_mm", "1-sigma target of the length-type section dimensions [mm]"),
    ("job.speed_mps", "measurement speed [m/s] (= --measure-speed; inside the calibrated range)"),
    ("job.profile_fps", "profiler frame rate [1/s]"),
    ("weld_plan.stick_out_mm", "wire stick-out from the gas nozzle end [mm]"),
    ("weld_plan.travel_angle_deg", "[root, fill, cap] deg, + push / - drag (a role may list its layers: [10, [10, 8], 5])"),
    ("weld_plan.work_angle_deg", "[root, fill, cap] deg, + toward the bevel face"),
    ("weld_plan.bead_toe_angle_deg", "the convex bead's toe angle [deg] (its one shape value: arc height / chord = tan(toe/2)/2)"),
    ("weld_plan.speed_mps", "[root, fill, cap] travel speeds [m/s] (also current_a, voltage_v, wfs_mpm)"),
    ("weld_plan.direction", "S->E | E->S | alternate"),
    ("weld_design.end_inset_mm.start", "weld lines start this far inside the measured start end [mm]"),
    ("weld_design.end_inset_mm.goal", "weld lines end this far inside the measured goal end [mm]"),
    ("weld_design.cap_reinforcement_mm.nominal", "cap height above the plate top [mm] (also .tol)"),
    ("specimen.weld_length_mm", "design weld length [mm] (also root_gap_mm, bevel_angle_deg, thickness_mm, plate_widths_mm)"),
    ("qc_targets.<key>.tol", "design tolerances (root_gap_mm, bevel_angle_deg, weld_length_mm, ...)"),
    ("auto_teach.max_passes", "pass budget of the run"),
)


class OverrideError(ValueError):
    """An override that cannot be applied (unknown key, a key the run does not take from the job, a value the parser refuses)."""


class _Loader(yaml.SafeLoader):
    """yaml.SafeLoader (YAML 1.1) + the YAML 1.2 / JSON float forms without a dot (1e-05, 2E3): PyYAML would read them as text
    (json.dumps writes 1e-05)."""


_Loader.add_implicit_resolver("tag:yaml.org,2002:float",
                              re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"), list("-+0123456789."))


def yaml_value(text: str):
    """One YAML value (the --set VALUE, a mapping text) with the exponent floats of YAML 1.2 / JSON."""
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass


# ---- text forms -------------------------------------------------------------------------------------------------------------
def parse_mapping_text(text, *, what: str, json_only: bool = False) -> dict:
    """A YAML / JSON mapping (the server parameter, an overrides file's text; json_only: the goal field, a JSON object) → flat
    {path: value}. '' = {}."""
    s = str(text or "").strip()
    if not s:
        return {}
    try:
        m = json.loads(s)
    except ValueError as e:
        if json_only:
            raise OverrideError(f"{what}: not a JSON object ({e})") from None
        try:
            m = yaml_value(s)
        except yaml.YAMLError as e2:
            raise OverrideError(f"{what}: not YAML / JSON ({type(e2).__name__}: {str(e2).splitlines()[0]})") from None
    if m is None:
        return {}
    if not isinstance(m, dict):
        raise OverrideError(f"{what}: must be a mapping {{path: value}}, got {type(m).__name__} {m!r}")
    return flatten(m, what=what)


def flatten(mapping: dict, prefix: str = "", *, what: str = "overrides") -> dict:
    """Nested mapping → {dotted path: value}: mappings merge key by key (an empty mapping is a value), the rest are values."""
    out: dict = {}
    for k, v in mapping.items():
        if not isinstance(k, (str, int)) or isinstance(k, bool):
            raise OverrideError(f"{what}: keys must be text, got {k!r}")
        key = f"{prefix}.{k}" if prefix else str(k)
        if any(seg.strip() == "" or seg != seg.strip() for seg in key.split(".")):
            raise OverrideError(f"{what}: bad path {key!r} (dotted names without spaces, e.g. weld_plan.stick_out_mm)")
        if isinstance(v, dict) and v:
            out.update(flatten(v, key, what=what))
        else:
            out[key] = v
    return out


def parse_set_items(items: Iterable[str], *, what: str = "--set") -> dict:
    """Client --set KEY=VALUE items → flat {path: value} (VALUE read as YAML: 0.1, [10, 10, 10], true, S->E; a mapping value merges
    under KEY). Later items win."""
    out: dict = {}
    for it in items or ():
        s = str(it)
        if "=" not in s:
            raise OverrideError(f"{what} expects KEY=VALUE, got {s!r}")
        k, v = s.split("=", 1)
        k = k.strip()
        try:
            val = yaml_value(v) if v.strip() != "" else None
        except yaml.YAMLError as e:
            raise OverrideError(f"{what} {k}: the value {v!r} is not a YAML value ({str(e).splitlines()[0]})") from None
        out.update(flatten({k: val}, what=what))
    return out


def merge(*layers: dict) -> dict:
    """Flat override mappings, later ones win (a path given again replaces the earlier value)."""
    out: dict = {}
    for m in layers:
        for k, v in (m or {}).items():
            out.pop(k, None)
            out[k] = v
    return out


def to_goal_text(overrides: dict) -> str:
    """Flat overrides → the AutoTeach goal's job_overrides (a JSON object; '' = none)."""
    return json.dumps(dict(overrides), ensure_ascii=False, default=str) if overrides else ""


def goal_overrides(goal) -> dict:
    """The AutoTeach goal's job_overrides (JSON object text) → flat {path: value}; a goal without the field (KIRO's rough goal,
    other callers) → {}. OverrideError for text that is not an object."""
    return parse_mapping_text(getattr(goal, "job_overrides", "") or "", what="goal job_overrides", json_only=True)


# ---- schema -----------------------------------------------------------------------------------------------------------------
def _template_doc(position: Optional[str]) -> dict:
    from . import config as C
    pos = str(position or "1G").strip() or "1G"
    p = C.JOBS_DIR / f"groove_10T_{pos}.yaml"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parser_name(block: str) -> str:
    return {"job": "ScanJob", "specimen": "GrooveSpec", "qc_targets": "Tolerances", "precision": "Precision",
            "weld_plan": "WeldProcess", "weld_design": "WeldDesign", "auto_teach": "the per-job auto_teach keys"}[block]


def _parser_defaults(block: str) -> dict:
    """{key: default value in the job file's form} of the block's parser (keys without a default: absent)."""
    from scan_core.specimen import GrooveSpec
    from scan_core.types import ScanJob
    from scan_core.weld_plan import WeldDesign, WeldProcess
    from . import config as C

    def fields_defaults(cls) -> dict:
        out = {}
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                out[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:          # type: ignore[misc]
                out[f.name] = f.default_factory()                          # type: ignore[misc]
        return _plain(out)
    if block == "job":
        return {k: v for k, v in fields_defaults(ScanJob).items()}
    if block == "specimen":
        d = fields_defaults(GrooveSpec)
        d.pop("notes", None)
        return d
    if block == "weld_plan":
        return _plain(WeldProcess().to_dict())
    if block == "weld_design":
        w = WeldDesign()
        return {"cap_reinforcement_mm": {"nominal": w.cap_reinforcement_mm, "tol": w.cap_reinforcement_tol_mm},
                "end_inset_mm": {"start": 0.0, "goal": 0.0}, "source": w.source, "provenance": dict(w.provenance)}
    if block == "auto_teach":
        a = C.AutoTeachConfig()
        return {k: getattr(a, k) for k in C.JOB_AUTO_TEACH_KEYS}
    return {}


def known_keys(block: str, doc: Optional[dict] = None, position: Optional[str] = None) -> list:
    """The keys of a block an override may name: the parser's keys, the job template's and the document's."""
    from . import config as C
    keys: set = set()
    if block == "job":
        keys |= set(C._JOB_FIELDS) - set(JOB_NOT_INPUTS)
    elif block == "specimen":
        keys |= set(_parser_defaults("specimen"))
    elif block == "precision":
        keys |= {"end_position_mm", "root_position_mm", "length_mm"}
    elif block == "weld_plan":
        keys |= set(_parser_defaults("weld_plan"))
    elif block == "weld_design":
        keys |= {"cap_reinforcement_mm", "end_inset_mm", "source", "provenance"}
    elif block == "auto_teach":
        keys |= set(C.JOB_AUTO_TEACH_KEYS)
    elif block == "qc_targets":
        keys |= set(C.QC_DESIGN_KEYS) | {"end_position_mm", "root_face_mm"}
    for src in ((doc or {}).get(block), _template_doc(position).get(block)):
        if isinstance(src, dict):
            keys |= set(str(k) for k in src)
    if block == "job":
        keys -= set(JOB_NOT_INPUTS)
    return sorted(keys)


# ---- apply ------------------------------------------------------------------------------------------------------------------
_ABSENT = object()


def _kind(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bool):
        return "yes/no"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "text"
    if isinstance(v, (list, tuple)):
        return "list"
    if isinstance(v, dict):
        return "mapping"
    return type(v).__name__


def _get(node, seg):
    if isinstance(node, dict):
        return node.get(seg, _ABSENT)
    if isinstance(node, (list, tuple)):
        i = int(seg)
        return node[i] if 0 <= i < len(node) else _ABSENT
    return _ABSENT


def _resolve(doc: dict, path: str, *, position: Optional[str], tpl: dict) -> tuple:
    """Check one path against the schema → (segments, old value or _ABSENT, old_from text, reference value at the leaf)."""
    segs = path.split(".")
    if any(s_ == "" for s_ in segs):
        raise OverrideError(f"job override {path!r}: empty path segment")
    block = segs[0]
    if block in NOT_OVERRIDABLE:
        raise OverrideError(f"job override {path!r}: block '{block}' cannot be overridden at run time ({NOT_OVERRIDABLE[block]})")
    if block not in BLOCKS:
        raise OverrideError(f"job override {path!r}: unknown block '{block}' (run-time overridable blocks: {', '.join(BLOCKS)})")
    if len(segs) < 2:
        raise OverrideError(f"job override {path!r}: a whole block cannot be replaced - give its keys ({block}.<key>)")
    key = segs[1]
    if block == "job" and key in JOB_NOT_INPUTS:
        raise OverrideError(f"job override {path!r}: not a run-time parameter - {JOB_NOT_INPUTS[key]}")
    kk = known_keys(block, doc, position)
    if key not in kk:
        raise OverrideError(f"job override {path!r}: unknown key '{key}' in block '{block}' (not in the job file, the job template "
                            f"groove_10T_{position or '1G'}.yaml or the {_parser_name(block)} keys; known: {', '.join(kk)})")
    # the value the run would use (the document's, else the parser's default) and the shape to check deeper segments against
    # (that value, else the template's)
    blk = doc.get(block) if isinstance(doc.get(block), dict) else {}
    eff = blk.get(key, _ABSENT)
    old_from = "job file"
    if eff is _ABSENT:
        dflt = _parser_defaults(block)
        if key in dflt:
            eff, old_from = copy.deepcopy(dflt[key]), f"default ({_parser_name(block)})"
        else:
            old_from = "absent"
    shape = eff if eff is not _ABSENT else (tpl.get(block) or {}).get(key, _ABSENT)
    free = (block, key) in FREE_FORM
    for i, seg in enumerate(segs[2:], start=2):
        here = ".".join(segs[:i])
        if isinstance(shape, dict) or (free and shape is _ABSENT):
            if not free and seg not in shape:
                raise OverrideError(f"job override {path!r}: unknown key '{seg}' under '{here}' (known: "
                                    f"{', '.join(str(k) for k in shape) or 'none'})")
        elif isinstance(shape, (list, tuple)):
            if not (seg.lstrip("-").isdigit() and 0 <= int(seg) < len(shape)):
                raise OverrideError(f"job override {path!r}: '{seg}' is not an index of the list '{here}' (0..{len(shape) - 1})")
        else:
            raise OverrideError(f"job override {path!r}: '{here}' is a single value ({shape!r}), not a mapping or a list - set "
                                f"'{here}' itself")
        shape = _get(shape, seg) if shape is not _ABSENT else _ABSENT
        eff = _get(eff, seg) if eff is not _ABSENT else _ABSENT
    return segs, eff, old_from, shape


def _set(doc: dict, segs: list, value, *, block_ref) -> None:
    """Set the value at the path, creating the mappings on the way; a list that the document lacks is first copied from the value
    the run would have used (the parser's default) so an index sets one entry of it."""
    node = doc
    ref = block_ref
    for i, seg in enumerate(segs[:-1]):
        nxt = segs[i + 1]
        if isinstance(node, dict):
            cur = node.get(seg)
            r = _get(ref, seg) if ref is not _ABSENT else _ABSENT
            if not isinstance(cur, (dict, list)):
                if nxt.lstrip("-").isdigit() and isinstance(r, (list, tuple)):
                    cur = copy.deepcopy(list(r))
                else:
                    cur = {}
                node[seg] = cur
            node, ref = cur, r
        else:                                                      # a list (index checked in _resolve)
            j = int(seg)
            if isinstance(node[j], tuple):
                node[j] = list(node[j])
            r = _get(ref, seg) if ref is not _ABSENT else _ABSENT
            if not isinstance(node[j], (dict, list)):
                node[j] = {} if not nxt.lstrip("-").isdigit() else copy.deepcopy(list(r) if isinstance(r, (list, tuple)) else [])
            node, ref = node[j], r
    last = segs[-1]
    if isinstance(node, dict):
        node[last] = copy.deepcopy(value)
    else:
        node[int(last)] = copy.deepcopy(value)


def _check_job_kind(path: str, ref, new) -> None:
    """ScanJob fields have no parse-time type check: an override keeps the kind of the value the run would have used."""
    kr, kn = _kind(ref), _kind(new)
    if kr is None or ref is _ABSENT:
        if kn not in (None, "number", "text", "yes/no", "list"):
            raise OverrideError(f"job override {path!r}: a {kn} is not a job value")
        return
    if kr != kn:
        raise OverrideError(f"job override {path!r}: type error - the job value is a {kr} ({ref!r}), got a {kn} ({new!r})")
    if kr == "number" and not math.isfinite(float(new)):
        raise OverrideError(f"job override {path!r}: must be a finite number, got {new!r}")
    if kr == "list":
        if len(new) != len(ref):
            raise OverrideError(f"job override {path!r}: type error - the job value is a list of {len(ref)} ({list(ref)!r}), got "
                                f"{len(new)} ({new!r})")
        for a, b in zip(ref, new):
            if _kind(a) != _kind(b):
                raise OverrideError(f"job override {path!r}: type error - entries are {_kind(a)}s ({list(ref)!r}), got {new!r}")


def _parse_blocks(doc: dict, blocks: set, *, position: Optional[str]) -> None:
    """The effective blocks through the parsers the job file goes through (ValueError / TypeError with the parser's message)."""
    from scan_core.g1_types import Precision, Tolerances
    from scan_core.weld_plan import WeldProcess, weld_design_from_doc
    from . import config as C
    from .autoteach import weld_process_doc
    pos = str(position or "").strip()
    for b in sorted(blocks):
        if b == "job":
            C.job_from_dict(dict(doc.get("job") or {}))
        elif b == "specimen":
            C.specimen_from_doc(doc)
        elif b == "qc_targets" and pos != "3G":                      # the 1G core reads them through Tolerances (3G: as given)
            Tolerances.from_qc(doc.get("qc_targets") or {})
        elif b == "precision":
            Precision.from_doc(doc.get("precision"))
        elif b in ("weld_plan", "weld_design"):
            proc = WeldProcess.from_dict(weld_process_doc(doc, pos or "1G"))
            weld_design_from_doc(doc, proc)
        elif b == "auto_teach":
            C.AutoTeachConfig.from_params(dict(doc.get("auto_teach") or {}))


def apply_overrides(doc: dict, overrides: dict, *, source: str, position: Optional[str] = None,
                    earlier: Sequence[dict] = ()) -> tuple:
    """One layer of overrides on a job document → (new document, records). The document is not changed. earlier = the records of
    the layers already applied (their paths name the source of 'old'). OverrideError names the path and the reason."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    out = copy.deepcopy(dict(doc or {}))
    if not overrides:
        return out, []
    tpl = _template_doc(position)
    recs, touched = [], set()
    for path, new in dict(overrides).items():
        path = str(path).strip()
        segs, old, old_from, _shape = _resolve(out, path, position=position, tpl=tpl)
        block, key = segs[0], segs[1]
        if block == "job":
            ref = old
            if ref is _ABSENT and len(segs) == 2:
                ref = _parser_defaults("job").get(key, _ABSENT)
            _check_job_kind(path, ref, new)
        for e in reversed(list(earlier)):                          # an earlier layer set this value (or a part / a parent of it)
            p = str(e.get("path", ""))
            if p == path or path.startswith(p + ".") or p.startswith(path + "."):
                old_from = str(e.get("source"))
                break
        blk_ref = _parser_defaults(block) if not isinstance(out.get(block), dict) or key not in out.get(block, {}) else _ABSENT
        if not isinstance(out.get(block), dict):
            out[block] = {}
        _set(out, segs, new, block_ref={block: blk_ref} if blk_ref is not _ABSENT else _ABSENT)
        recs.append({"path": path, "old": None if old is _ABSENT else _plain(old), "new": _plain(new), "source": source,
                     "old_from": old_from})
        touched.add(block)
    try:
        _parse_blocks(out, touched, position=position)
    except (TypeError, ValueError) as e:
        names = ", ".join(f"{r['path']}={r['new']!r}" for r in recs if r["path"].split(".")[0] in touched)
        raise OverrideError(f"job override ({source}) {names}: the job parser refuses the result: {e}") from None
    return out, recs


def apply_layers(doc: dict, layers: Sequence[tuple], *, position: Optional[str] = None) -> tuple:
    """Layers [(source, flat overrides), ...] in order (server parameter, then goal) → (new document, all records)."""
    recs: list = []
    for source, ov in layers:
        doc, r = apply_overrides(doc, ov or {}, source=source, position=position, earlier=recs)
        recs.extend(r)
    return doc, recs


# ---- record -----------------------------------------------------------------------------------------------------------------
def records_of(doc: Optional[dict]) -> list:
    """The override records an effective document carries (meta.job_overrides; [] = none)."""
    return list((((doc or {}).get("meta") or {}).get("job_overrides")) or [])


def job_file_of(doc: Optional[dict]) -> Optional[str]:
    """The job file an effective document was loaded from (meta.job_file)."""
    v = ((doc or {}).get("meta") or {}).get("job_file")
    return None if v is None else str(v)


def _short(v) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= 80 else s[:77] + "..."


def record_lines(records: Sequence[dict]) -> list:
    """'<path> <old> -> <new> (<source>; old from <old_from>)' per record."""
    return [f"{r['path']} {_short(r['old'])} -> {_short(r['new'])} ({r['source']}; old from {r['old_from']})" for r in records]


def summary_line(records: Sequence[dict], job_file: Optional[str] = None) -> str:
    """The one server line at the start of a run."""
    where = f" on {job_file}" if job_file else ""
    if not records:
        return f"job overrides: none (the job file{where} as it is)"
    return f"job overrides ({len(records)}){where}: " + "; ".join(record_lines(records))


def write_effective_job(path, doc: dict) -> Path:
    """The run's effective job document (the job file + the overrides + the goal's points; meta.job_file / meta.job_overrides)
    as YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    head = ("# the effective job of this run: the job file meta.job_file + the run-time overrides meta.job_overrides (path, old, new, "
            "source) + the goal's rough points (scan_l0.job_overrides)\n")
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(head)
        yaml.safe_dump(_plain(doc), f, sort_keys=False, allow_unicode=True)
    tmp.replace(p)
    return p


def _plain(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    return o
