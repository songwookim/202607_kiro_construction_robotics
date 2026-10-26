"""Post-weld data acquisition scan (user 2026-10-13): after the welding - or after any pass of it (the caller decides when: in the
lab the operator, at KIRO their main system between passes) - ONE straight scan of the whole weld along the measured seam of a
finished 1G core run. Data acquisition only - no estimation, no decision, no re-planning, no change to the weld plan. Called only
explicitly (the action / the CLI) - nothing in the 1G run starts it.

Input = a finished run folder (the server's out_root/<run>; default the latest): its measured seam (sections.json: the S / E ends,
the section frame u / n, the measured groove depth) and its scan settings (verification/run_log.json: the job, the measurement speed,
the final cell world, the specimen design; the last pass leg's raw.npz: its recorded job - standoff, pitch, incidence, progression
sign = the sensor orientation of the run's passes - and the sensor init parameters the run applied). The scan line's speed = the
run's measurement speed (its passes), or the goal's measure_speed_mps: a speed chosen at run time, checked like a run's run-time
measurement speed (the 1G core's one check, Teach1G._check_measure_speed: inside the calibration's noise-prior speed range,
scan_core.noise_priors.check_measure_speed; REFUSED before any motion otherwise). after_pass (the pass just completed, 0 = final)
only names the output folder and goes into the run log; the scan is the same after any pass.

The scan moves exactly like a sensing pass of that run (user decision 2026-10-13: the bead does not stand meaningfully above the plate
top - nothing is added for it): the run's final world (the same MoveIt scene objects and envelope its passes used; restored from the
run log, cell_world.CellWorld.from_record), the run's recorded pass standoff, pitch, incidence and tool orientation on the measured
line, the measurement speed, the positioning moves at the platform limits and the standard checks of every scan move (envelope D3,
MoveIt D4, confirmation, cancel). The line: from SPAN_M beyond the measured S end to SPAN_M beyond the measured E end along the
measured seam axis ([S - 20 mm, E + 20 mm], the user's data span) - the laser hit on the plate-top line covers that span at the
measurement speed; beyond it the pass's own cruise terms only (path_gen.pass_overscan with Teach1G._cruise_terms: the speed ramp
v^2/2a at both ends, the servo settling v t_settle at the first end - t_settle from the input run's legs with the controller-lag
prior); no other constant. The direction (which end first) is the acquisition's rule: both directions' scan starts, a retreat along
the plate normal (job safe_clearance_m), every arm configuration that follows retreat -> start -> end (platform reach_candidates),
pooled and ranked by the least joint motion among well-conditioned configurations (acquisition.rank_reach_candidates; wrist2 as the
platform config requires), the straight legs checked by the envelope first. Moves (the MoveIt adapter, from wherever the arm is):
joint-space move to the retreat pose, the descent onto the scan start (positioning speed), the scan line (measurement speed, sensor
recording). The arm stays at the end of the line.

RAW DATA ONLY (user 2026-10-13: the bead is rough and hot, it may glow or reflect): nothing looks at the profiles - no feature
extraction, no validity / quality check, no stop because frames look bad or empty. Every frame is recorded as it comes
(the sensor's error / checksum / counters with it) and written. The sensor runs with the run's own settings (the init parameters
the run applied and recorded - its fixed exposure and gain, its frame rate; no auto exposure, nothing new) and is restored at the
end. The only stops: an action cancel, a motion planning / execution failure, a sensor device / connection error. The summary
reports counts only.

Output: <run_dir>/post_weld_scan_p<k>_<YYYYmmdd-HHMMSS>/ (after pass k) or post_weld_scan_final_<...>/ = raw.npz (the pass legs'
format: sensor profiles, robot poses / stamps, controller states, the job / calib / config / sensor JSON + post_weld_scan_json = the
scan definition), run_log.json (input run, after_pass, line endpoints in base and World, speed, standoff, timing, status),
sensor_backup/ (the sensor's own parameters before this scan - the restore record). Dry run: the plan only (the run's world is
applied to the MoveIt scene for the checks and removed again), nothing is sent, the sensor is not opened, no folder is written.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from scan_core.path_gen import EnvelopeError, Waypoint, pass_overscan, sensor_frames, tool_keypoints
from scan_core.tool_body import body_points_base
from scan_core.seam_estimate import _section_basis
from scan_core.types import JobError, ScanJob, _jsonable
from scan_core.g1_types import jsonable

from . import config as C
from . import g1_worker as _gw
from .acquisition import rank_reach_candidates
from .autoteach import AutoTeach, _Abort, _rot_angle_deg, wrist2_branch_problem, move_plan_text
from .cell_world import CellWorld
from .config import attach_tool_body, job_to_dict
from .executor import (STATUS_ABORTED, STATUS_ERROR, STATUS_OK, STATUS_REFUSED, ScanRunResult, StartPoseMismatch, execute_leg,
                       gate_and_arm, raw_payload)
from .planning import make_local_move_plan
from .ports import MotionError
from .ports import PlatformNotReady, ScanRefused
from .teach_1g import MOVE_MEASURE, MOVE_POSITION, Teach1G

__all__ = ["SPAN_M", "OUT_PREFIX", "SCHEMA", "PHASES", "STATUS_DRY_RUN", "InputRun", "InputRunError", "PostWeldResult", "PostWeldScan",
           "latest_run", "resolve_run_dir", "load_input_run", "scan_line", "out_dir_for", "after_pass_label"]

SPAN_M = 0.020                       # the data span beyond each measured end (user specification 2026-10-13: [S - 20 mm, E + 20 mm])
OUT_PREFIX = "post_weld_scan_"
SCHEMA = "hongik.post_weld_scan/1"
PHASES = ("setup", "plan", "move", "scan", "save")
STATUS_DRY_RUN = "DRY_RUN"
SCAN_MOVE = "post_weld_scan"         # the scan line's move name (run log, raw.npz motions)


class InputRunError(ValueError):
    """The input run cannot be used (missing / unreadable file, not a 1G core run, no measured world) - refused before anything is
    sent."""


# --------------------------------------------------------------------------------------
# the input run
# --------------------------------------------------------------------------------------
@dataclass
class InputRun:
    """What the post-weld scan takes from a finished 1G core run (module docstring). Points / vectors in the run's base frame [m]."""
    run_dir: Path
    status: str
    base_frame: str
    S: np.ndarray                       # measured S end (root line), the end the run visited first
    E: np.ndarray
    u: np.ndarray                       # the measured seam axis S -> E (sections frame)
    n: np.ndarray                       # the measured plate normal (section frame, toward the sensor)
    ends_two_sided: dict                # {'S': bool, 'E': bool} - an end the run read from one side only is a bound
    depth_m: float                      # measured groove depth (root -> plate top along n)
    job: dict                           # the run's job (run log job_rough)
    job_file_speed_mps: float           # the job file's measurement speed (run log measure_speed)
    run_speed_mps: float                # the run's measurement speed (with its override)
    pass_job: dict                      # the run's last pass leg: its recorded job (raw.npz job_json)
    pass_leg: str
    stream_legs: list                   # [{name, kind, raw}] the run's recorded moving legs (transit, passes): settling evidence
    world: dict                         # the run's final cell world record
    spec: dict                          # the specimen design (run log prior.spec)
    sensor_init: dict                   # the sensor init parameters the run applied (its last pass's sensor_info)
    n_plan_passes: Optional[int] = None  # passes of the run's weld plan (weld_plan.json, if any): the after_pass label / range
    files: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def n_passes(self) -> Optional[int]:
        return self.n_plan_passes

    def check_after_pass(self, after_pass: int) -> int:
        """after_pass (the pass just completed; 0 = final) → itself; ValueError for a negative number or a pass beyond the run's weld
        plan (when the run has one)."""
        k = int(after_pass or 0)
        if k < 0:
            raise ValueError(f"after_pass must be >= 0 (0 = final), got {k}")
        if self.n_plan_passes is not None and k > int(self.n_plan_passes):
            raise ValueError(f"after_pass {k}: the run's weld plan has passes 1..{self.n_plan_passes} (0 = final)")
        return k

    def job_and_doc(self) -> tuple:
        """→ (ScanJob, job document) of the post-weld scan: the run's job on the measured ends and plate normal at the run's
        measurement speed (its passes' speed; no rough-point inputs, no earlier measurements in the job), the specimen design."""
        d = dict(self.job)
        d.update(start=[float(v) for v in self.S], end=[float(v) for v in self.E], surface_normal=[float(v) for v in self.n],
                 speed_mps=float(self.run_speed_mps), table_z_m=None, specimen_top_z_m=None, plate_top_measured_m=None,
                 plate_plane_measured=None, floor_z_m=None, base_frame=self.base_frame, job_id=f"{self.name}_post_weld")
        job = C.job_from_dict(d)
        doc = {"specimen": dict(self.spec), "job": job_to_dict(job),
               "meta": {"post_weld_scan": True, "input_run": str(self.run_dir),
                        "note": "post-weld scan job from the input run (measured ends, the run's scan settings)"}}
        return job, doc

    def to_dict(self) -> dict:
        return _jsonable({"run_dir": str(self.run_dir), "status": self.status, "base_frame": self.base_frame,
                          "S_end_m": self.S, "E_end_m": self.E, "u": self.u, "n": self.n, "ends_two_sided": self.ends_two_sided,
                          "depth_mm": self.depth_m * 1e3, "job_file_speed_mps": self.job_file_speed_mps,
                          "run_speed_mps": self.run_speed_mps, "pass_leg": self.pass_leg,
                          "pass_job": {k: self.pass_job.get(k) for k in ("standoff_m", "pitch_deg", "incidence_tilt_deg",
                                                                         "progression_sign", "profile_fps", "speed_mps")},
                          "stream_legs": [lg["name"] for lg in self.stream_legs], "world_version": self.world.get("version"),
                          "n_plan_passes": self.n_plan_passes, "sensor_init": self.sensor_init, "files": self.files})


def _read_json(p: Path, what: str) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise InputRunError(f"{what} not found: {p}")
    except (OSError, ValueError) as e:
        raise InputRunError(f"{what} unreadable ({p}): {type(e).__name__}: {e}")


def _is_run_dir(d: Path) -> bool:
    return (d / "sections.json").is_file() and (d / "verification" / "run_log.json").is_file()


def latest_run(out_root) -> Path:
    """The latest finished run under out_root: the folder whose verification/run_log.json was written last among those with a
    sections.json (a 1G core run that measured the seam). InputRunError when there is none."""
    root = Path(out_root).expanduser()
    if not root.is_dir():
        raise InputRunError(f"output root {root} does not exist (no run to scan after)")
    cands = [d for d in root.iterdir() if d.is_dir() and _is_run_dir(d)]
    if not cands:
        raise InputRunError(f"no finished run (sections.json + verification/run_log.json) under {root}")
    return max(cands, key=lambda d: (d / "verification" / "run_log.json").stat().st_mtime)


def resolve_run_dir(run_dir: str, out_root) -> Path:
    """The goal's run_dir → the run folder: '' = the latest run under out_root, an absolute path as it is, otherwise a folder name
    under out_root. InputRunError when it is not a finished run folder."""
    txt = str(run_dir or "").strip()
    if not txt:
        return latest_run(out_root)
    p = Path(txt).expanduser()
    if not p.is_absolute():
        p = Path(out_root).expanduser() / p
    if not p.is_dir():
        raise InputRunError(f"run folder {p} does not exist")
    if not _is_run_dir(p):
        up = next((a for a in p.parents if _is_run_dir(a)), None)
        if up is not None:                                   # e.g. <run>/directions/E_to_S/kiro: a weld-plan set inside the run
            raise InputRunError(f"{p} is a folder inside the run {up} (a weld-plan output, not a run): give the run folder itself "
                                f"(--run {up}); the scan direction is chosen with --direction S->E|E->S (default: the least TCP motion)")
        raise InputRunError(f"{p} is not a finished run folder (needs sections.json and verification/run_log.json)")
    return p


def _leg_raw(run_dir: Path, leg: dict) -> Path:
    """A run-log leg's raw.npz inside this run folder (the log keeps the absolute path of the run's own location; the folder may have
    been moved since)."""
    return run_dir / "verification" / "legs" / Path(str(leg.get("dir") or "")).name / "raw.npz"


def load_input_run(run_dir) -> InputRun:
    """A finished 1G core run folder → InputRun (module docstring). InputRunError with the reason when something is missing."""
    rd = Path(run_dir)
    sec = _read_json(rd / "sections.json", "sections.json")
    log = _read_json(rd / "verification" / "run_log.json", "verification/run_log.json")
    if str(log.get("pipeline")) != "g1core":
        raise InputRunError(f"{rd.name}: not a 1G core run (pipeline {log.get('pipeline')!r}) - the post-weld scan follows a 1G core run")
    plan_p = next((p for p in (rd / "weld_plan.json", rd / "verification" / "weld_plan.json") if p.is_file()), None)
    n_plan = len(_read_json(plan_p, "weld_plan.json").get("passes") or []) if plan_p is not None else None
    try:
        fr = sec["frame"]
        S, E = np.asarray(fr["S_m"], dtype=float).reshape(3), np.asarray(fr["E_m"], dtype=float).reshape(3)
        u_s, n_s = np.asarray(fr["u"], dtype=float).reshape(3), np.asarray(fr["n"], dtype=float).reshape(3)
        ends = sec.get("ends") or {}
        two = {X: bool((ends.get(X) or {}).get("two_sided", False)) for X in ("S", "E")}
        depth_mm = float(((sec.get("seam_dims") or {}).get("depth_mm") or {}).get("value"))
    except (KeyError, TypeError, ValueError) as e:
        raise InputRunError(f"{rd.name}: sections.json has no measured seam frame / ends / depth ({type(e).__name__}: {e})")
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(E)) and float(np.linalg.norm(E - S)) > 0.0 and math.isfinite(depth_mm)):
        raise InputRunError(f"{rd.name}: sections.json ends / depth are not finite")
    u, n, _w = _section_basis(E - S, n_s)
    if float(u @ u_s) <= 0.0:
        raise InputRunError(f"{rd.name}: sections.json frame u does not point from S to E")
    legs = [dict(lg) for lg in (log.get("legs") or [])]
    stream = [{"name": str(lg.get("name")), "kind": str(lg.get("kind")), "raw": _leg_raw(rd, lg)} for lg in legs
              if lg.get("kind") in ("transit", "pass")]
    passes = [lg for lg in stream if lg["kind"] == "pass"]
    if not passes:
        raise InputRunError(f"{rd.name}: the run log has no pass leg (the scan settings are the run's pass settings)")
    last = passes[-1]
    try:
        with np.load(last["raw"], allow_pickle=False) as z:
            pass_job = json.loads(str(z["job_json"]))
            sensor_init = dict((json.loads(str(z["sensor_info_json"])) if "sensor_info_json" in z.files else {}).get("init_params")
                               or {})
    except FileNotFoundError:
        raise InputRunError(f"{rd.name}: the last pass's raw.npz ({last['raw']}) not found - its recorded job is the run's scan setting")
    except (OSError, KeyError, ValueError) as e:
        raise InputRunError(f"{rd.name}: the last pass's recorded job not readable ({last['raw']}): {type(e).__name__}: {e}")
    if not sensor_init:
        raise InputRunError(f"{rd.name}: the last pass's raw.npz has no sensor init parameters (sensor_info_json init_params) - the "
                            "scan keeps the run's own sensor settings")
    worlds = list(log.get("world") or [])
    if not worlds:
        raise InputRunError(f"{rd.name}: the run log has no cell world")
    ms = dict(log.get("measure_speed") or {})
    job = dict(log.get("job_rough") or {})
    if not job:
        raise InputRunError(f"{rd.name}: the run log has no job (job_rough)")
    v_job = ms.get("job_file_speed_mps", job.get("speed_mps"))
    v_run = ms.get("speed_mps", job.get("speed_mps"))
    spec = dict(((log.get("prior") or {}).get("spec")) or {})
    if not spec:
        raise InputRunError(f"{rd.name}: the run log has no specimen design (prior.spec)")
    try:
        v_job, v_run = float(v_job), float(v_run)
    except (TypeError, ValueError) as e:
        raise InputRunError(f"{rd.name}: the run's measurement speed is missing ({type(e).__name__}: {e})")
    return InputRun(run_dir=rd, status=str(log.get("status") or sec.get("status") or ""), base_frame=str(sec.get("base_frame") or
                    job.get("base_frame") or "link0"), S=S, E=E, u=u, n=n, ends_two_sided=two, depth_m=depth_mm * 1e-3, job=job,
                    job_file_speed_mps=v_job, run_speed_mps=v_run, pass_job=pass_job, pass_leg=last["name"], stream_legs=stream,
                    world=dict(worlds[-1]), spec=spec, sensor_init=sensor_init, n_plan_passes=n_plan,
                    files={"sections": str(rd / "sections.json"), "run_log": str(rd / "verification" / "run_log.json"),
                           "weld_plan": str(plan_p) if plan_p is not None else None, "pass_raw": str(last["raw"])})


def after_pass_label(after_pass: int, n_passes: Optional[int] = None) -> str:
    """'after pass 2 of 4' / 'final (after every pass)'."""
    k = int(after_pass or 0)
    if k <= 0:
        return "final (after every pass)"
    return f"after pass {k}" + (f" of {int(n_passes)}" if n_passes else "")


def scan_line(S, E, n, span_m: float = SPAN_M) -> dict:
    """The data span on the measured seam: u = the S -> E axis (made perpendicular to n), s from S; the span runs from s = -span to
    s = L + span (the user's [S - span, E + span]). → {u, n, L_m, s_from_m, s_to_m, P_from, P_to, span_m}."""
    S, E = np.asarray(S, dtype=float).reshape(3), np.asarray(E, dtype=float).reshape(3)
    span = float(span_m)
    if not (math.isfinite(span) and span >= 0.0):
        raise ValueError(f"span must be >= 0 m, got {span_m!r}")
    u, n_l, _w = _section_basis(E - S, np.asarray(n, dtype=float))
    L = float((E - S) @ u)
    if not L > 0.0:
        raise ValueError("the measured ends coincide along the seam")
    return {"u": u, "n": n_l, "L_m": L, "s_from_m": -span, "s_to_m": L + span, "P_from": S - u * span, "P_to": S + u * (L + span),
            "span_m": span}


def out_dir_for(run_dir, after_pass: int = 0, now: Optional[_dt.datetime] = None) -> Path:
    """<run_dir>/post_weld_scan_p<k>_<YYYYmmdd-HHMMSS> after pass k, post_weld_scan_final_<…> for 0 (final / unspecified); a second
    scan in the same second gets _2, _3 …"""
    stamp = (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    k = int(after_pass or 0)
    if k < 0:
        raise ValueError(f"after_pass must be >= 0 (0 = final), got {after_pass!r}")
    base = Path(run_dir) / f"{OUT_PREFIX}{f'p{k}' if k > 0 else 'final'}_{stamp}"
    p, i = base, 1
    while p.exists():
        i += 1
        p = base.with_name(f"{base.name}_{i}")
    return p


def _motion_s(move: dict) -> float:
    """An executed move record's motion time [s] (its MotionResult: motion_duration_s, else elapsed_s)."""
    mo = move.get("motion") or {}
    v = mo.get("motion_duration_s") or mo.get("elapsed_s")
    return float(v) if v is not None else 0.0


@dataclass
class PostWeldResult:
    status: str = STATUS_ERROR
    reason: str = ""
    out_dir: str = ""
    run_dir: str = ""
    after_pass: int = 0
    raw_path: str = ""
    n_frames: int = 0
    category: Optional[str] = None
    elapsed_s: float = 0.0
    line: dict = field(default_factory=dict)
    summary: str = ""


# --------------------------------------------------------------------------------------
# the scan
# --------------------------------------------------------------------------------------
class PostWeldScan(Teach1G):
    """One post-weld data-acquisition scan (module docstring). The motion primitives (world, envelope D3, MoveIt check D4,
    confirmation, joint move, straight legs with sensor recording, cancel, abort path) are the 1G core's."""

    def __init__(self, settings, run: InputRun, calib, *, out_dir, after_pass: int = 0, direction: str = "",
                 tool_orientation: str = "pass", **kw):
        job0, doc = run.job_and_doc()
        kw.pop("max_passes", None)
        super().__init__(settings, job0, doc, calib, out_dir=out_dir, **kw)
        self.run_in = run
        self.direction = str(direction or "").strip().replace(" ", "")
        if self.direction not in ("", "S->E", "E->S"):
            raise ValueError(f"direction must be 'S->E', 'E->S' or '' (the nearer end), got {direction!r}")
        self.tool_orientation = str(tool_orientation or "pass").strip().lower().replace("-", "_")
        if self.tool_orientation not in ("pass", "least_rotation"):
            raise ValueError(f"tool_orientation must be 'pass' or 'least_rotation', got {tool_orientation!r}")
        self._start_inside: Optional[dict] = None
        self.after_pass = run.check_after_pass(after_pass)
        self.vdir = self.out                          # no verification/: the sensor backup goes next to raw.npz
        self.pw = PostWeldResult(run_dir=str(run.run_dir), after_pass=self.after_pass)
        self.plan: Optional[dict] = None
        self.world_transform: dict = {}

    # ---- phases / names -----------------------------------------------------------------------------------------------
    def measure_speed_choice(self) -> Optional[str]:
        """'goal override (measure_speed_mps)' when the goal chose the scan line's speed (a run-time choice: the 1G core's
        _check_measure_speed checks it against the calibrated range), None = the input run's measurement speed (that run's own)."""
        return "goal override (measure_speed_mps)" if self.measure_speed_override_mps is not None else None

    def measure_speed_record(self) -> dict:
        """The scan line's speed and where it came from: the input run's measurement speed (its passes; the run's job speed with its
        own override) unless the goal overrides it; with the calibrated range it was checked against (calibrated_range) when it did."""
        ov = self.measure_speed_override_mps
        rec = {"speed_mps": float(self.job0.speed_mps), "run_speed_mps": float(self.run_in.run_speed_mps),
               "run_job_file_speed_mps": float(self.run_in.job_file_speed_mps), "override_mps": None if ov is None else float(ov),
               "source": self.measure_speed_choice() or "the input run's measurement speed"}
        if self.measure_speed_check is not None:
            rec["calibrated_range"] = dict(self.measure_speed_check)
        return rec

    def measure_speed_text(self) -> str:
        if self.measure_speed_override_mps is None:
            return "the input run's measurement speed"
        return f"goal override; the input run measured at {self.run_in.run_speed_mps * 1e3:.1f} mm/s"

    def _set_phase(self, ph: str) -> None:
        assert ph in PHASES, ph
        self._phase = ph
        self._event("phase", phase=ph)

    # ---- setup: the run's job on the measured line + the run's world ---------------------------------------------------------
    def _derive_jobs(self, job0: ScanJob) -> None:
        """The run's coarse / reference jobs (AutoTeach._derive_jobs) and, once, the world = the input run's final measured world -
        the world its passes used (instead of the rough-point prior of a run)."""
        AutoTeach._derive_jobs(self, job0)
        self.job_c = self._at_speed(self.job_c, MOVE_POSITION)
        if self.world is None:
            dm = self._spec_dims_m()
            try:
                w = CellWorld.from_record(self.run_in.world, thickness_m=dm["t"], backing_m=dm["b"], design_length_m=dm["L"],
                                          clearance_m=float(self.policy.clearance_m),
                                          reach_m=float(self.job_c.standoff_m) + float(job0.safe_clearance_m),
                                          why=f"run {self.run_in.name}: its final measured world v{self.run_in.world.get('version')}")
            except ValueError as e:
                raise _Abort(STATUS_REFUSED, f"input run {self.run_in.name}: {e} (nothing sent)", category="input")
            self.world = w
            self.world_history.append(w.to_dict())
        self._world_to_envelope()

    def _setup(self) -> None:
        """Gates before any motion: position 1G, tool body, specimen design, motion config, safety policy, the job, the world (the
        run's) in MoveIt (the arm must not be inside it), platform gates."""
        if not self.dry_run:
            self.out.mkdir(parents=True, exist_ok=True)
        self._check_position()
        if self.res.position != "1G":
            raise _Abort(STATUS_REFUSED, f"position {self.res.position}: the post-weld scan follows a 1G core run (nothing sent)",
                         category="input")
        try:
            self.calib = attach_tool_body(self.calib, self.settings)
            self.calib0 = attach_tool_body(self.calib0, self.settings)
        except ValueError as e:
            raise _Abort(STATUS_REFUSED, f"tool body: {e} (nothing sent)", category="config")
        if self.spec is None or not (self.spec.weld_length_mm > 0.0):
            raise _Abort(STATUS_REFUSED, "the input run has no specimen design with a weld length (nothing sent)", category="input")
        self._check_measure_speed()                       # the goal's speed: inside the calibrated range (the 1G core's check)
        self._motion_config()
        job0 = self._apply_safety_policy(self.job0)
        try:
            job0.validate()
        except (JobError, ValueError) as e:
            raise _Abort(STATUS_REFUSED, f"post-weld job invalid: {e} (nothing sent)", category="input")
        self.orientation = job0.resolved_orientation()
        self.up = job0.up_unit
        self._derive_jobs(job0)
        run = self.run_in
        one = [X for X, ok in run.ends_two_sided.items() if not ok]
        self._decide("setup", f"{after_pass_label(self.after_pass, run.n_passes)} - input run {run.name} ({run.status}): measured "
                              f"ends S {np.round(run.S, 4).tolist()}, E {np.round(run.E, 4).tolist()} m ({run.base_frame}), seam "
                              f"{float(np.linalg.norm(run.E - run.S)) * 1e3:.2f} mm, groove depth {run.depth_m * 1e3:.2f} mm; scan settings from {run.pass_leg} (standoff "
                              f"{float(run.pass_job.get('standoff_m', float('nan'))) * 1e3:.1f} mm, pitch "
                              f"{float(run.pass_job.get('pitch_deg', float('nan'))):+.1f} deg, {float(run.job.get('profile_fps', 0)):g} fps)"
                              + (f"; end(s) {', '.join(one)} were read from one side only in that run (a bound) - scanned as given"
                                 if one else ""), input_run=run.to_dict())
        self.res.notes.extend(f"input run end {X} was read from one side only (a bound)" for X in one)
        v_m, v_p = self._scan_speed(), self._class_speed(MOVE_POSITION)
        lim = (dict(self.settings.params or {}).get("moveit_jtc") or {}).get("limits") or {}
        qd = lim.get("max_joint_velocity_rad_s")
        self._decide("setup", f"speeds: scan line {v_m * 1e3:.1f} mm/s ({self.measure_speed_text()}; spacing "
                              f"{v_m / float(self.job0.profile_fps) * 1e3:.2f} mm at {float(self.job0.profile_fps):g} fps); positioning "
                              f"{v_p * 1e3:.1f} mm/s (platform TCP limit {float(self.cfg.v_eff_max_mps) * 1e3:.1f} mm/s"
                              + (f", joints {float(qd):g} rad/s" if qd is not None else "") + ")",
                     measure_speed={**self.measure_speed_record(), "commanded_mps": v_m}, speed_classes={MOVE_MEASURE: v_m,
                                                                                                          MOVE_POSITION: v_p})
        try:
            init = gate_and_arm(self.sess, self.cfg, log=self.log)
        except PlatformNotReady as e:
            raise _Abort(STATUS_REFUSED, f"{e} (nothing sent)", category="motion")
        except ScanRefused as e:
            raise _Abort(STATUS_REFUSED, str(e), category="config")
        self.default_speed = float(init.default_speed)
        self._latch_at_start = self._latch_last()
        # the platform's joint-space planning and collision scene: checked once the session has its motion backend (KIRO field
        # 2026-10-15: checked before it, a post-weld scan as the first goal of a fresh server was refused - the backend did not exist yet)
        self._universal_ready()
        self._apply_world(f"the run {run.name}'s world (its passes' scene)", inside_ok=True)

    # ---- the sensor: the run's settings, raw recording only -----------------------------------------------------------------
    def _sensor(self) -> None:
        """Connect and apply the run's own sensor settings (its recorded init parameters: fixed exposure / gain, frame rate) - nothing
        new, no auto exposure - with the parameter backup next to raw.npz; restored at the end. A connection / initialisation failure
        is a device error (REFUSED before any motion)."""
        if self.sensor_ready:
            return
        from .executor import ensure_sensor
        init = dict(self.run_in.sensor_init)
        try:
            ensure_sensor(self.sess, self.settings, sensor_ip=self.conn["sensor_ip"], sensor_port=self.conn["sensor_port"],
                          sensor_init=init, backup_dir=str(self.vdir / "sensor_backup"), log=self.log)
        except _Abort:
            raise
        except Exception as e:  # noqa: BLE001 - any connection / initialisation failure of the sensor (device error)
            raise _Abort(STATUS_ERROR if self._motion_sent else STATUS_REFUSED,
                         f"sensor connection/initialisation failed {'after motion' if self._motion_sent else 'before any motion (nothing sent)'}"
                         f": {type(e).__name__}: {e}", category="sensor")
        self.sensor_ready = True
        keys = ("AutoTriggerFPS", "ExposureMode", "ShutterTime", "GainVal", "DataMode")
        shown = {k: init[k] for k in keys if k in init}
        self._decide("move", f"sensor: the run's settings ({len(init)} init parameters{', ' + str(shown) if shown else ''}); no profile "
                             "checks - every frame is recorded as it comes", sensor_init=init)

    # ---- the plan: line, scan job, overscan, direction ---------------------------------------------------------------------
    def _input_settle_devs(self) -> list:
        """The input run's moving legs (transit, passes): their encoder along-track deviation after the start ramp (the settling
        evidence _settle pools; scan_core.settling.leg_deviation) - read from their raw.npz."""
        from scan_core.settling import leg_deviation
        devs = []
        for lg in self.run_in.stream_legs:
            try:
                with np.load(lg["raw"], allow_pickle=False) as z:
                    t = np.asarray(z["robot_t_rx_ns"], dtype=float) * 1e-9
                    P = np.asarray(z["robot_T_base_flange"], dtype=float)[:, :3, 3]
                    v = float(json.loads(str(z["job_json"]))["speed_mps"])
                devs.append(leg_deviation(t, P, v, float(self.accel)))
            except Exception as e:  # noqa: BLE001 - evidence only: the settling then rests on the other legs / the lag prior
                self.res.notes.append(f"settling evidence of the input leg {lg['name']} not read: {type(e).__name__}: {e}")
        return devs

    def _scan_job(self, line: dict, direction: str, sign: Optional[int] = None) -> tuple:
        """The pass job on the measured line in the travel direction: the run's pass standoff / pitch / incidence, the progression
        sign whose tool orientation is the run's pass orientation (the recorded pass job's flange rotation with this calib), the
        measurement speed → (job, rotation from the run's pass orientation [deg]). The standoff is the recorded one as it is (it
        already carries the run's plate-clearance raise, if any)."""
        rec = self.run_in.pass_job
        P_S = np.asarray(self.run_in.S, dtype=float)
        P_E = P_S + line["u"] * line["L_m"]
        a, b = (P_E, P_S) if direction == "E->S" else (P_S, P_E)
        kw = job_to_dict(self.job0)
        kw["torch_lateral_m"] = tuple(kw["torch_lateral_m"])
        kw.update(start=a.tolist(), end=b.tolist(), surface_normal=line["n"].tolist(), orientation=self.orientation,
                  standoff_m=float(rec["standoff_m"]), pitch_deg=float(rec["pitch_deg"]),
                  incidence_tilt_deg=float(rec.get("incidence_tilt_deg", 0.0)), lead_in_m=0.001, lead_out_m=0.001,
                  bidirectional=False, plate_top_measured_m=self.plate_top, plate_plane_measured=self.job_ref.plate_plane_measured,
                  job_id=f"{self.job0.job_id or 'job'}_{direction.replace('->', '_')}")
        try:
            ref = C.job_from_dict(dict(rec))
            R_ref = sensor_frames(ref, self.calib).T_base_flange(0.0)[:3, :3]
        except (JobError, ValueError, KeyError) as e:
            raise _Abort(STATUS_REFUSED, f"the run's pass job ({self.run_in.pass_leg}) is not usable: {e} (nothing sent)", category="input")
        best = None
        for sign in ((1, -1) if sign is None else (int(sign),)):
            kw["progression_sign"] = sign
            try:
                job = ScanJob(**kw)
                job.validate()
            except (JobError, ValueError) as e:
                raise _Abort(STATUS_REFUSED, f"post-weld scan job invalid: {e} (nothing sent)", category="motion")
            rot = _rot_angle_deg(R_ref, sensor_frames(job, self.calib).T_base_flange(0.0)[:3, :3])
            if best is None or rot < best[0]:
                best = (rot, job)
        rot, job = best
        return self._at_speed(job, MOVE_MEASURE), rot

    def _plan(self) -> dict:
        """Both directions: scan job, laser-hit overscan (span + cruise terms), retreat pose, envelope of the descent and the scan,
        arm configurations that follow retreat -> start -> end; pooled ranking (least joint motion among well-conditioned
        configurations) → the plan of the chosen direction. REFUSED when no direction can be planned."""
        run = self.run_in
        line = scan_line(run.S, run.E, run.n, SPAN_M)
        D = float(run.depth_m)
        v = self._scan_speed()
        self._settle_devs = self._input_settle_devs()
        ramp, settle, st = self._cruise_terms(v)
        retreat = float(self.job0.safe_clearance_m)
        q_now = self._q_now()
        req = self._wrist2_required()
        T_now = self._T_now()
        rows, pool, ctx = [], [], {}
        dirs = (self.direction,) if self.direction else ("S->E", "E->S")
        if self.tool_orientation == "pass":           # the sensing passes' orientation: per direction its progression sign
            options = [(d_, int(self._scan_job(line, d_)[0].progression_sign)) for d_ in dirs]
        else:                                          # both torch sides
            options = [(d_, sg) for d_ in dirs for sg in (1, -1)]
        for direction, sign in options:
            key = f"{direction} {sign:+d}"
            job, rot = self._scan_job(line, direction, sign)
            first = "S" if direction == "S->E" else "E"
            ov_first, ov_last = line["span_m"] + ramp + settle, line["span_m"] + ramp
            try:
                ovs = pass_overscan(job, self.calib, 0.0, top_offset_m=D, overscan_first_m=ov_first, overscan_last_m=ov_last)
            except ValueError as e:
                rows.append({"direction": direction, "error": f"overscan: {e}"})
                continue
            T_from, T_to = np.asarray(ovs["T_from"], dtype=float), np.asarray(ovs["T_to"], dtype=float)
            T_pre = np.array(T_from, dtype=float, copy=True)
            T_pre[:3, 3] += line["n"] * retreat
            env_why = self._legs_pass_envelope(T_pre, [(T_from, "descend", "post_weld_descend", ""), (T_to, "scan", SCAN_MOVE, "")],
                                               why=True)
            tcp_m, rot_now = self._tcp_motion(T_now, T_pre)
            row = {"direction": direction, "key": key, "first_end": first, "progression_sign": int(job.progression_sign),
                   "rotation_from_run_pass_deg": rot, "tool_rot_from_now_deg": rot_now, "tcp_move_mm": tcp_m * 1e3}
            if env_why:
                row["error"] = f"envelope (D3): {env_why}"
                rows.append(row)
                continue
            cands = self.rs.reach_candidates(T_pre, follow=[T_from, T_to], q_now=q_now, sigma_radius_m=self._lever_m())
            if req != "any":
                for c in cands:
                    why = wrist2_branch_problem(c.get("names") or [], np.asarray(c["q"]), req)
                    if why:
                        c["ok"], c["reason"] = False, f"wrist2_branch ({self.res.position}): {why}"
            for c in cands:
                c["direction"], c["key"] = direction, key
            pool.extend(cands)
            row.update(n_ik=len(cands), n_ok=sum(1 for c in cands if c.get("ok")),
                       fails=sorted({str(c.get("reason", ""))[:80] for c in cands if not c.get("ok")}))
            rows.append(row)
            ctx[key] = {"job": job, "ovs": ovs, "T_pre": T_pre, "T_from": T_from, "T_to": T_to, "first": first, "direction": direction,
                        "ov": {"first_m": ov_first, "last_m": ov_last}, "tcp_m": tcp_m, "rot": rot_now}
        ranked = rank_reach_candidates(pool)
        for r_ in rows:
            mine = [c for c in ranked if c.get("key") == r_["key"]]
            if mine:
                r_["least_joint_motion_deg"] = None if mine[0].get("dist_rad") is None else math.degrees(float(mine[0]["dist_rad"]))
        txt = "; ".join(f"{r_['key']}: " + (r_["error"] if "error" in r_ else
                        f"TCP motion: rotation {r_['tool_rot_from_now_deg']:.0f} deg, torch tip {r_['tcp_move_mm']:.0f} mm; "
                        f"{r_['n_ok']}/{r_['n_ik']} arm configurations follow it"
                        + (f" ({', '.join(r_['fails'][:2])})" if r_.get("fails") else "")) for r_ in rows)
        # the tool pose and the direction (user 2026-10-15): tool orientation 'pass' (default) = the sensing passes' (one option per
        # direction - the same view as the passes), 'least_rotation' = the torch side nearer the current pose (the acquisition's
        # rule); then the nearer end (the torch tip's distance); the joint path of that pose by duration
        # the physical tool yaw (the torch's side) of an option: its progression sign in the S->E frame - both directions offer the
        # same two sides, their pitch tilt only flips; the side with the smaller rotation first, then the torch tip's distance (which
        # end first), so a few degrees of pitch never send the scan to the far end
        side = {k_: int(ctx[k_]["job"].progression_sign) * (1 if ctx[k_]["direction"] == "S->E" else -1) for k_ in ctx}
        side_rot = {}
        for k_, sd in side.items():
            side_rot[sd] = min(side_rot.get(sd, math.inf), ctx[k_]["rot"])
        rank_side = {sd: i for i, sd in enumerate(sorted(side_rot, key=lambda sd: side_rot[sd]))}
        usable = sorted((k_ for k_ in ctx if any(c.get("key") == k_ for c in ranked)),
                        key=lambda k_: (rank_side[side[k_]], ctx[k_]["tcp_m"], ctx[k_]["rot"]))
        if not usable:
            self._decide("plan", f"no direction can be planned: {txt}", options=_jsonable(rows))
            raise _Abort(STATUS_REFUSED, f"post-weld scan: no arm configuration reaches the scan start and follows the line "
                                         f"({txt}) (nothing sent)", category="motion")
        key = usable[0]
        cx = ctx[key]
        direction = cx["direction"]
        same = [c for c in ranked if c.get("key") == key]
        job, ovs = cx["job"], cx["ovs"]
        plan = {"direction": direction, "tool_orientation": self.tool_orientation, "first_end": cx["first"], "ranked": same,
                "T_pre": cx["T_pre"], "T_from": cx["T_from"],
                "T_to": cx["T_to"], "job": job, "line": line, "depth_m": D, "retreat_m": retreat,
                "overscan": {"span_m": line["span_m"], "speed_ramp_m": ramp, "settling_m": settle, "first_end_m": cx["ov"]["first_m"],
                             "last_end_m": cx["ov"]["last_m"], "hit_offset_m": float(ovs["hit_offset_m"]),
                             "s_from_m": float(ovs.get("s_from", float("nan"))), "s_to_m": float(ovs.get("s_to", float("nan")))},
                "settling": _jsonable(st), "options": _jsonable(rows)}
        L_leg = float(np.linalg.norm(cx["T_to"][:3, 3] - cx["T_from"][:3, 3]))
        plan["leg_m"] = L_leg
        self.plan = plan
        span_txt = (f"[S - {line['span_m'] * 1e3:.0f}, E + {line['span_m'] * 1e3:.0f}] mm = {(line['s_to_m'] - line['s_from_m']) * 1e3:.1f} "
                    f"mm along the measured seam")
        self._decide("plan", f"line {direction} (first end {cx['first']}): data span {span_txt}; laser-hit overscan beyond the span "
                             f"{cx['ov']['first_m'] * 1e3 - line['span_m'] * 1e3:.2f} mm at {cx['first']} (speed ramp v^2/2a "
                             f"{ramp * 1e3:.2f} + settling v t_settle {settle * 1e3:.2f}: t_settle {float(st['t_settle_s']) * 1e3:.0f} ms "
                             f"from {st['n_legs']} leg(s) of the run, {st['source']}), {ramp * 1e3:.2f} mm at the other end (speed ramp); "
                             f"leg {L_leg * 1e3:.1f} mm at {float(job.speed_mps) * 1e3:.1f} mm/s, standoff {float(job.standoff_m) * 1e3:.1f} "
                             f"mm, pitch {float(job.pitch_deg):+.1f} deg, progression {int(job.progression_sign):+d}; "
                             f"orientation choice: {txt} -> {key} ("
                             + ("tool orientation of the sensing passes" if self.tool_orientation == "pass" else
                                f"the torch side with the smaller rotation ({cx['rot']:.0f} deg)")
                             + f", then the nearer end: torch tip {cx['tcp_m'] * 1e3:.0f} mm"
                             + (f"; direction {self.direction} as asked" if self.direction else "; both directions")
                             + f"{'' if self._wrist2_required() == 'any' else ', wrist2 ' + self._wrist2_required()})",
                     plan=_jsonable({k_: v_ for k_, v_ in plan.items() if k_ not in ("ranked", "job", "line")}),
                     line=_jsonable(line))
        return plan

    def _lift_clear(self) -> None:
        """Start where the welding left the arm (KIRO field 2026-10-15: the post-weld scan follows a weld at once - the torch is at the
        seam, inside our grown specimen objects): one straight move along the measured plate normal, the tool orientation unchanged,
        until the tool's lowest point (keypoints and tool body) is at the plate top plus the safety clearance the envelope keeps
        (job safe_clearance_m = the platform's clearance + the stopping distance). Every tool point moves straight away from the plate
        (a groove opens upward), so the plate-distance rule holds along the line by construction; the end pose passes the full
        envelope; MoveIt checks the line against the robot itself and everything that is not ours, with our grown objects out of the
        scene for this one line only (they are put back right after, and the arm must then be clear of them)."""
        n = np.asarray(self.world.plate_normal, dtype=float)
        n = n / np.linalg.norm(n)
        if float(n @ np.asarray(self.up, dtype=float)) < 0.0:
            n = -n
        P0 = np.asarray(self.world.plate_point, dtype=float)
        T0 = self._T_now()
        kp = tool_keypoints(T0, self.calib, self.job_ref)
        pts = [np.asarray(v, dtype=float).reshape(-1, 3) for v in kp.values()]
        body = body_points_base(T0 @ self.calib.T_flange_eoat, getattr(self.calib, "body_points_eoat", None))
        if body is not None:
            pts.append(np.asarray(body, dtype=float).reshape(-1, 3))
        low = float(min(((p - P0) @ n).min() for p in pts))
        need = float(self.job_ref.safe_clearance_m)
        h_opt = float((np.asarray(kp["opt"], dtype=float).reshape(-1, 3)[0] - P0) @ n)
        need_opt = float(self.job_ref.standoff_min_m)
        deep = float((self._start_inside or {}).get("deep_m", 0.0))
        # the envelope's plate-distance rule at the end pose: tool body / torch >= the clearance, optical origin >= standoff_min
        L = max(need - low, need_opt - h_opt, deep, 0.0)
        T1 = np.array(T0, dtype=float, copy=True)
        T1[:3, 3] += n * L
        name = "post_weld_lift"
        v = self._class_speed(MOVE_POSITION)
        try:                                                     # the end pose: the whole envelope (a resting pose there is a normal start)
            T1b = np.array(T1, dtype=float, copy=True)
            T1b[:3, 3] += n * 1e-4                               # a pose check: the end pose as the start of a tiny ascend
            make_local_move_plan(T1, T1b, "ascend", self.job_ref, self.calib, specimen_top_z_m=self.plate_top, speed_mps=v,
                                 accel_mps2=self.accel, name=name)
        except (EnvelopeError, JobError, ValueError) as e:
            raise _Abort(STATUS_REFUSED, f"post-weld scan: the pose {L * 1e3:.0f} mm above the current one along the plate normal does "
                                         f"not pass the envelope: {e} - move the torch out of the groove and start again (nothing sent)",
                         category="motion")
        d = self._record_move(name, "ascend", T0, T1, {"lift_m": L, "lowest_tool_point_m": low, "clearance_m": need,
                                                     "optical_origin_m": h_opt, "standoff_min_m": need_opt}, v)
        self._decide("move", f"{name} (ascend): {L * 1e3:.1f} mm straight up along the plate normal, tool orientation kept - the tool's "
                             f"lowest point {low * 1e3:+.1f} -> {(low + L) * 1e3:.1f} mm from the plate top (clearance {need * 1e3:.0f} mm), "
                             f"optical origin {h_opt * 1e3:.1f} -> {(h_opt + L) * 1e3:.1f} mm (standoff_min {need_opt * 1e3:.0f} mm); our "
                             "grown objects out of the MoveIt scene for this line only")
        self._confirm(name, L, extra="(out of the weld pose, straight up)", rot_deg=0.0, kind="local")
        ours = list(self.cell_ids)
        self.rs.remove_scene_boxes(ours)
        self._motion_sent = True
        try:
            w = Waypoint(name, "ascend", T1, float(v), float(self.accel), "move_l")
            res = execute_leg(self.sess, w, v, timeout_s=self.motion_timeout_s, expected_start=T0,
                              start_tol=(self.policy.goal_tol_m, self.policy.goal_tol_deg))
        except StartPoseMismatch as e:
            d["refused"] = str(e)
            raise _Abort(STATUS_REFUSED, f"move '{name}': {e} (nothing sent)", category="motion")
        except MotionError as e:
            d["motion"] = {"error": str(e)}
            raise _Abort(STATUS_ABORTED, f"move '{name}': {e}", safety=True, category="motion")
        finally:
            self.rs.set_scene_boxes(self.world.moveit_objects())
        self.res.n_move_l += 1
        d["motion"], d["executed"] = res.as_dict(), bool(res.ok)
        if not res.ok:
            raise _Abort(STATUS_ABORTED, f"move '{name}': {res.reason}", safety=True, category="motion")
        self._p(f"      -> OK {res.reason} v_cmd={res.v_cmd:.1f} mm/s duration={res.motion_duration_s:.2f} s")
        contacts, reason = self._arm_contacts()
        if contacts or reason:
            raise _Abort(STATUS_REFUSED, f"post-weld scan: after the lift the arm still touches the cell world "
                                         f"({self.world.explain(contacts) or reason}) - nothing more sent", category="motion")
        self._start_inside = None

    # ---- World endpoints -------------------------------------------------------------------------------------------------
    def _world_from_base(self) -> tuple:
        """World ← base for the run log (the KIRO-format outputs' rule: the runtime MoveIt planning frame, the platform TF, the config
        kiro_output.world_from_base) → (RigidTransform | None, notes)."""
        from scan_core import kiro_plan as _kp
        from scan_core.rigid import RigidTransform, compose
        notes = []
        try:
            cfg = _kp.kiro_output_config(self.settings.params)
        except (TypeError, ValueError) as e:
            return None, [f"platform config kiro_output: {e}"]
        if cfg is None:
            return None, ["no kiro_output section in the platform config (no World frame)"]
        base = str(self.job0.base_frame)
        tf = getattr(self.rs, "frame_transform", None)
        pf_fn = getattr(self.rs, "planning_frame", None)
        rt, rt_src = pf_fn() if callable(pf_fn) else (None, f"the platform ({self.rs.adapter}) has no MoveIt planning frame")
        if rt is None:
            notes.append(f"planning frame not read ({rt_src}): the base frame {base} is used")
            rt = base
        try:
            T_rp = None
            if rt != base:
                if not callable(tf):
                    return None, notes + [f"the planning frame {rt} is not the base frame {base} and the platform has no TF"]
                M = np.asarray(tf(rt, base), dtype=float)
                T_rp = RigidTransform(R=M[:3, :3], t=M[:3, 3].copy(), source=f"TF {rt} <- {base}", base_frame=base)
            wt, n2 = _kp.resolve_world_from_base(cfg, rt, (lambda a, b: tf(a, b)) if callable(tf) else None)
            notes += n2
            if wt is not None and T_rp is not None:
                wt = compose(wt, T_rp, source=f"{wt.source} x TF {rt} <- {base}")
        except Exception as e:  # noqa: BLE001 - the World record never breaks the scan
            return None, notes + [f"World <- {base} not resolved: {type(e).__name__}: {e}"]
        return wt, notes

    def _line_record(self) -> dict:
        p = self.plan or {}
        line = p.get("line") or scan_line(self.run_in.S, self.run_in.E, self.run_in.n, SPAN_M)
        wt, notes = self._world_from_base()
        pts = {"S_end": self.run_in.S, "E_end": self.run_in.E, "span_from": line["P_from"], "span_to": line["P_to"]}
        if p.get("direction") == "E->S":
            order = ("span_to", "span_from")
        else:
            order = ("span_from", "span_to")
        rec = {"base_frame": self.run_in.base_frame, "direction": p.get("direction"), "tool_orientation": p.get("tool_orientation"),
               "span_m": line["span_m"],
               "data_span_s_m": [line["s_from_m"], line["s_to_m"]], "measured_length_m": line["L_m"],
               "s": "s = (P - S).u from the measured S end along the measured seam axis u (S->E); the root line",
               "base_m": {k: _jsonable(np.asarray(v, dtype=float)) for k, v in pts.items()},
               "scan_start_m": _jsonable(np.asarray(pts[order[0]], dtype=float)) if p else None,
               "scan_end_m": _jsonable(np.asarray(pts[order[1]], dtype=float)) if p else None,
               "u": _jsonable(line["u"]), "n": _jsonable(line["n"])}
        if wt is not None:
            rec["world_frame"] = "World"
            rec["world_m"] = {k: _jsonable(wt.apply(np.asarray(v, dtype=float))) for k, v in pts.items()}
            rec["world_from_base"] = wt.to_dict()
        else:
            rec["world_m"] = None
        rec["world_notes"] = notes
        return rec

    # ---- moves -----------------------------------------------------------------------------------------------------------
    def _dry_run_moves(self, plan: dict) -> None:
        """Dry run: the joint-space move's MoveIt plan (check only, the first arm configuration that plans) and the two straight legs
        (their envelope was checked in the plan) - printed, nothing sent."""
        be = self.rs.backend
        v = min(float(self.transit_speed), self.cfg.v_eff_max_mps)
        pick, fails = None, []
        for c in plan["ranked"]:
            chk = be.check_joint(c["q"], v, self.accel, timeout_s=self.motion_timeout_s)
            if chk.get("ok"):
                pick = (c, chk)
                break
            fails.append(str(chk.get("reason")))
        if pick is None:
            raise _Abort(STATUS_REFUSED, f"post-weld scan (dry run): no collision-free joint-space path to any of the {len(plan['ranked'])} "
                                         f"arm configurations of the retreat pose: {fails[:2]} (nothing sent)", category="motion")
        c, chk = pick
        T_now = self._T_now()
        self._decide("move", f"dry run, not moved - 1) joint-space move to {plan['retreat_m'] * 1e3:.0f} mm above the scan start: "
                             f"{move_plan_text(chk.get('plan') or {})}; arm configuration {np.round(np.degrees(c['q']), 1).tolist()} deg "
                             f"(from {np.round(T_now[:3, 3], 4).tolist()} to {np.round(plan['T_pre'][:3, 3], 4).tolist()} m)")
        d_desc = float(np.linalg.norm(plan["T_from"][:3, 3] - plan["T_pre"][:3, 3]))
        self._decide("move", f"dry run, not moved - 2) descent {d_desc * 1e3:.1f} mm onto the scan start "
                             f"{np.round(plan['T_from'][:3, 3], 4).tolist()} m at {self._class_speed(MOVE_POSITION) * 1e3:.1f} mm/s; "
                             f"3) scan line {plan['leg_m'] * 1e3:.1f} mm to {np.round(plan['T_to'][:3, 3], 4).tolist()} m at "
                             f"{float(plan['job'].speed_mps) * 1e3:.1f} mm/s (sensor recording)")

    def _save_raw(self, res, records: list, samples: list, *, status: str = STATUS_OK, reason: str = "post-weld scan") -> Path:
        """raw.npz (the pass legs' format: raw_payload + controller states) + post_weld_scan_json (the scan definition)."""
        job = self.plan["job"]
        run = ScanRunResult(status=status, reason=reason, out_dir=str(self.out), records=records, robot_samples=samples,
                            motions=[(SCAN_MOVE, res)] if res is not None else [], scan_motion=res,
                            sensor_info=self.ss.info.as_dict() if (self.sess.sensor is not None) else {},
                            sensor_summary=({"client_stats": self.ss.stats} if self.sess.sensor is not None else {}),
                            safety_init=self.rs.backend.safety_init_result if self.rs.backend is not None else None)
        run.t_sensor_start_ns = getattr(self.ss, "t_start_ns", 0) if self.sess.sensor is not None else 0
        run.t_sensor_stop_ns = getattr(self.ss, "t_stop_ns", 0) if self.sess.sensor is not None else 0
        arrays, js = raw_payload(run, job, self.calib, self.settings, self.cfg, self.rs.pose)
        if samples:
            arrays.update(self._controller_record(samples))
        js["post_weld_scan_json"] = json.dumps(jsonable(self._definition()), default=str)
        path = self.out / "raw.npz"
        _gw.atomic_savez(path, **arrays, **{k_: np.array(v_) for k_, v_ in js.items()})
        self.res.files.append(str(path))
        return path

    def _definition(self) -> dict:
        """The scan definition (raw.npz post_weld_scan_json and run_log.json 'scan')."""
        p = self.plan or {}
        job = p.get("job")
        return {"schema": SCHEMA, "input_run": str(self.run_in.run_dir), "line": self._line_record(),
                "scan": None if job is None else {
                    "speed_mps": float(job.speed_mps), "measure_speed": {**self.measure_speed_record(), "commanded_mps": float(job.speed_mps)},
                    "profile_fps": float(job.profile_fps), "spacing_m": float(job.speed_mps) / float(job.profile_fps),
                    "standoff_m": float(job.standoff_m), "pitch_deg": float(job.pitch_deg),
                    "incidence_tilt_deg": float(job.incidence_tilt_deg), "progression_sign": int(job.progression_sign),
                    "depth_m": p.get("depth_m"), "overscan": p.get("overscan"), "settling": p.get("settling"),
                    "leg_m": p.get("leg_m"), "flange_from": _jsonable(p.get("T_from")), "flange_to": _jsonable(p.get("T_to")),
                    "retreat_m": p.get("retreat_m"), "direction_options": p.get("options"),
                    "job": job_to_dict(job)},
                "after_pass": self.after_pass, "after_pass_label": after_pass_label(self.after_pass, self.run_in.n_passes),
                "world": self.world.to_dict() if self.world is not None else None}

    # ---- run -------------------------------------------------------------------------------------------------------------
    def run(self) -> PostWeldResult:
        r, pw = self.res, self.pw
        t_scan = {}
        try:
            self._set_phase("setup")
            self._setup()
            sensor_open = False
            if self._start_inside is not None and not self.dry_run:
                self._set_phase("move")
                self._sensor()                                    # before the first motion (a failure has sent nothing)
                sensor_open = True
                self._lift_clear()
            self._set_phase("plan")
            plan = self._plan()
            if self.dry_run:
                if self._start_inside is not None:
                    self.res.notes.append("dry run: the arm starts inside the run's grown world - a real scan lifts it out along the plate "
                                          "normal first (post_weld_lift); the plan above is from the current pose")
                self._dry_run_moves(plan)
                r.status = STATUS_DRY_RUN
                r.reason = (f"dry run: line {plan['direction']} over [S - {SPAN_M * 1e3:.0f}, E + {SPAN_M * 1e3:.0f}] mm, "
                            f"leg {plan['leg_m'] * 1e3:.1f} mm at {float(plan['job'].speed_mps) * 1e3:.1f} mm/s - nothing sent, "
                            "no sensor, no folder")
                return pw
            self._set_phase("move")
            if not sensor_open:
                self._sensor()                                    # before the first motion (a failure has sent nothing)
            self._joint_move(plan["ranked"], plan["T_pre"], "post_weld_approach",
                             f"{plan['retreat_m'] * 1e3:.0f} mm above the post-weld scan start (first end {plan['first_end']})")
            self._leg(plan["T_from"], "descend", "post_weld_descend", speed_class=MOVE_POSITION,
                      extra=f"(onto the post-weld scan start, {plan['retreat_m'] * 1e3:.0f} mm toward the plate)")
            self._set_phase("scan")
            t0 = time.perf_counter()
            try:
                res, records, samples = self._leg(plan["T_to"], "scan", SCAN_MOVE, speed_class=MOVE_MEASURE, streaming=True,
                                                  extra=f"(post-weld scan {plan['direction']}, sensor recording)")
            except _Abort as e:
                if self._last_stream is not None and self._last_stream[0]:
                    recs_p, samp_p, res_p = self._last_stream
                    try:
                        pth = self._save_raw(res_p, recs_p, samp_p, status=e.status, reason=e.reason)
                        self._decide("scan", f"{SCAN_MOVE}: stopped ({e.status}); partial data saved ({len(recs_p)} frames) -> {pth}")
                        pw.raw_path, pw.n_frames = str(pth), len(recs_p)
                    except Exception as ex:  # noqa: BLE001
                        r.notes.append(f"partial save failed: {type(ex).__name__}: {ex}")
                raise
            t_scan["scan_motion_wall_s"] = time.perf_counter() - t0
            self._set_phase("save")
            t1 = time.perf_counter()
            path = self._save_raw(res, records, samples)
            t_scan["save_s"] = time.perf_counter() - t1
            pw.raw_path, pw.n_frames = str(path), len(records)
            r.status, r.reason = STATUS_OK, (f"{len(records)} frames over {plan['leg_m'] * 1e3:.1f} mm ({plan['direction']}, data span "
                                             f"[S - {SPAN_M * 1e3:.0f}, E + {SPAN_M * 1e3:.0f}] mm) -> {path}")
            self._decide("save", f"raw data: {path} ({len(records)} sensor frames, {len(samples)} robot samples)")
            return pw
        except _Abort as e:
            status = STATUS_ABORTED if self._cancelled and e.status in (STATUS_ABORTED, STATUS_ERROR) else e.status
            cause = self._external_stop() or "cancel requested by the operator"      # an action cancel / the server shutting down
            reason = e.reason if status == e.status else f"{cause} (D7): {e.reason}"
            self._abort_path(status, reason, safety=e.safety, category=e.category if status == e.status else None)
            return pw
        except Exception as e:  # noqa: BLE001 - after a cancel / server shutdown: ABORTED (_unexpected_failure)
            st_, why_, cat_ = self._unexpected_failure(e, " (D7)")
            self._abort_path(st_, why_, category=cat_)
            r.notes.append(traceback.format_exc(limit=8))
            return pw
        except BaseException as e:  # noqa: BLE001
            self._abort_path(STATUS_ABORTED, f"interrupted: {type(e).__name__}")
            raise
        finally:
            r.elapsed_s = self._clock.monotonic() - self.t0
            self._clear_cell()
            try:
                if r.estop_record is None:
                    r.estop_record = self._latch_record_of_this_run()
            except Exception as e:  # noqa: BLE001
                r.notes.append(f"stop record of this run not read: {type(e).__name__}: {e}")
            if self.sess is not None and self.release_sensor and self.sess.sensor is not None:
                try:
                    r.sensor_restore = self.sess.restore_sensor()
                except Exception as e:  # noqa: BLE001
                    r.notes.append(f"sensor restore failed: {e}")
            pw.status, pw.reason, pw.category, pw.elapsed_s = r.status, r.reason, r.category, float(r.elapsed_s)
            wrote = (not self.dry_run) and self.out.is_dir()
            pw.out_dir = str(self.out) if wrote else ""
            try:
                pw.line = self._line_record() if self.plan is not None else {}
            except Exception as e:  # noqa: BLE001
                r.notes.append(f"line record failed: {type(e).__name__}: {e}")
            if wrote:
                self._write_log(t_scan)
            if self.sess is not None and self.release_sensor and self.sess.sensor is not None:
                try:
                    self.sess.sensor.close()
                except Exception as e:  # noqa: BLE001
                    r.notes.append(f"sensor close failed: {e}")
                self.sess.sensor = None
                self.sess.sensor_ready = False
            fds, self._cancel_r, self._cancel_w = (self._cancel_r, self._cancel_w), -1, -1
            for fd in fds:
                try:
                    os.close(fd)
                except OSError as e:
                    r.notes.append(f"cancel pipe close failed: {e}")
            pw.summary = self._summary_line()
            self._p(pw.summary)

    def _summary_line(self) -> str:
        """The one English console line of the scan."""
        pw, p = self.pw, self.plan or {}
        job = p.get("job")
        what = (f"line {p.get('direction')} [S - {SPAN_M * 1e3:.0f}, E + {SPAN_M * 1e3:.0f}] mm, leg {float(p.get('leg_m', 0.0)) * 1e3:.1f} mm "
                f"at {float(job.speed_mps) * 1e3:.0f} mm/s, standoff {float(job.standoff_m) * 1e3:.1f} mm" if job is not None else "no line")
        out = f" -> {pw.out_dir}" if pw.out_dir else ""
        t_scan = next((_motion_s(m) for m in self.res.moves if m.get("name") == SCAN_MOVE and m.get("executed")), None)
        return (f"post-weld scan {pw.status}: run {self.run_in.name}, {after_pass_label(self.after_pass, self.run_in.n_passes)}, {what}, "
                f"{pw.n_frames} frames" + (f" in {t_scan:.1f} s of scan" if t_scan else "") + f", {pw.elapsed_s:.1f} s in all{out}"
                + (f" ({pw.reason})" if pw.status not in (STATUS_OK,) else ""))

    def _write_log(self, t_scan: dict) -> None:
        r = self.res
        moves = [m for m in r.moves if m.get("executed")]
        dur = _motion_s
        doc = {"schema": SCHEMA, "status": r.status, "reason": r.reason, "category": r.category, "input_run": self.run_in.to_dict(),
               **{k: v for k, v in self._definition().items() if k in ("after_pass", "after_pass_label", "line", "scan")},
               "world": self.world_history, "raw": self.pw.raw_path or None, "n_frames": self.pw.n_frames,
               "timing": {"elapsed_s": float(r.elapsed_s), "positioning_motion_s": sum(dur(m) for m in moves if m["name"] != SCAN_MOVE),
                          "scan_motion_s": sum(dur(m) for m in moves if m["name"] == SCAN_MOVE), **t_scan},
               "moves": r.moves, "decisions": r.decisions, "notes": r.notes, "files": r.files, "estop_record": r.estop_record,
               "profile_checks": "none - raw data only: nothing evaluates the profiles (user 2026-10-13)",
               "sensor_settings": {"init_params": self.run_in.sensor_init, "source": f"the input run {self.run_in.pass_leg}'s recorded init"},
               "sensor_restore": r.sensor_restore, "n_moves": r.n_move_l,
               "config": {"platform": self.settings.path, "calib": self.calib.source, "confirm": self.confirm_mode}}
        try:
            _gw.atomic_write_text(self.out / "run_log.json", json.dumps(jsonable(doc), indent=1, ensure_ascii=False, default=str))
        except Exception as e:  # noqa: BLE001
            r.notes.append(f"run log save failed: {type(e).__name__}: {e}")
