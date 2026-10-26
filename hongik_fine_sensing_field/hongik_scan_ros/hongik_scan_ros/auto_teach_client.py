"""hongik_auto_teach_client - the specimen input on this PC and a console client of the /hongik_fine_sensing/auto_teach action.

  --setup   specimen input for the server's job file (job_file:=): if --job exists, show it and ask "Reuse the previous inputs?
            [u]se / [d]o not use"; do not use (or no file yet) = enter the specimen (root gap, bevel angle, weld length; the other values
            come from the template groove_10T_1G.yaml). No motion. On the KIRO PC the PrecisionSensing goal then brings the two points
            (pose_a / pose_b) and the start pose is the arm pose at goal acceptance.
  (default) run a job file whose two points (job.start / job.end, base frame) are set, through the auto_teach action: prints every
            progress line and, at the end, the result (status, ends, length, output folder), the cross sections (sections.json) and
            the multi-pass weld paths (weld_plan.json). Ctrl-C = cancel the goal = the server stops our trajectory and ends the run
            (nothing to reset afterwards).

  --measure-speed V   this run's measurement speed [mm/s] (scan passes, precise corrections at the dwell sites) instead of
            the job file's speed_mps; any speed inside the range the noise priors were calibrated over (the calibration file's
            g1 noise_priors speed_range_mps: 15..60 mm/s) - the server refuses others before any motion; every other move stays at
            the platform's positioning limits. (A KIRO PrecisionSensing goal has no speed: it measures at the job file's speed_mps,
            or at a job.speed_mps of the server parameter job_overrides.)
  --set KEY=VALUE   (repeatable) a run-time job override: KEY = a dotted path into the job document, VALUE = a YAML value
            (0.10, [10, 10, 10], S->E, true); --set-file FILE = a YAML mapping of such paths (nested mappings merge the same
            way), the --set items on top. The job file is not changed: the server applies them on top of its site overrides
            (server parameter job_overrides), refuses an unknown key or a value the job parser refuses (nothing sent), and records
            them (verification/run_log.json job_overrides, verification/job_effective.yaml). The client checks them on the job
            file first and prints old -> new.

  ros2 run hongik_scan_ros hongik_auto_teach_client --setup --job ~/scan_data/kiro/job_current.yaml
  ros2 run hongik_scan_ros hongik_auto_teach_client --job job_rough.yaml --out run01
  ros2 run hongik_scan_ros hongik_auto_teach_client --job job_rough.yaml --out run02 --measure-speed 30 \
      --set weld_plan.stick_out_mm=17.5 --set precision.end_position_mm=0.2 --set "weld_plan.travel_angle_deg=[-40,-40,-40]"
"""
# (한국어 설명, 코드 주석) 터미널에서 /hongik_fine_sensing/auto_teach 액션을 실행한다 (MoveIt 경로, 2026-10-03). 인터페이스 r3 §5·§8
# (2026-10-10): 재사용 질문 (use / do not use — 사용자 2026-10-10 '일부' 없앰), 시편 입력 (자세·루트 갭·개선각·용접 길이; 나머지는
# 템플릿 값을 보여 주고 --edit-all 에서만 고침), --setup (KIRO: 시편만), 확인 기본 auto, 진행 줄은 서버 로그 토픽 (있으면) 또는 피드백
# 에서 하나도 빠짐없이, PASS_UNRESOLVED 표시. 운전자에게 보이는 문구는 모두 영어 (사용자 2026-10-10).
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

__all__ = ["goal_from_job", "measure_speed_mps", "measure_speed_text", "result_lines", "rejection_lines",
           "cli_overrides", "check_overrides", "overrides_help",
           "sections_lines", "main", "build_parser", "load_input", "input_summary", "ask_reuse",
           "ask_specimen", "write_input_job", "has_taught_points", "ProgressPrinter", "log_topic_candidates", "REUSE_PROMPT"]

ACTION = "/hongik_fine_sensing/auto_teach"


def goal_from_job(job_path, *, out_name: str = "", confirm_mode: str = "none", max_passes: int = 0,
                  dry_run: bool = False, position: Optional[str] = None, measure_speed_mps: float = 0.0,
                  job_overrides: Optional[dict] = None):
    """job YAML → AutoTeach.Goal (점은 base 프레임 — job 의 base_frame 이 플랫폼 base 와 같다고 본다; frame_id '' ).
    measure_speed_mps: 이 실행의 측정 속도 [m/s] (패스·드웰 정밀 보정; 0 = 작업 파일의 speed_mps) — 서버가 이 실행의 작업 속도로 쓴다.
    job_overrides: run-time job overrides {dotted path: value} (cli_overrides) → the goal's job_overrides (JSON; '' = none)."""
    from hongik_scan_msgs.action import AutoTeach
    from scan_l0 import config as C
    v_ms = float(measure_speed_mps or 0.0)
    if not (math.isfinite(v_ms) and v_ms >= 0.0):
        raise ValueError(f"measure_speed_mps must be 0 (the job's speed_mps) or a speed > 0 m/s, got {measure_speed_mps!r}")
    job, doc = C.load_job(str(job_path))
    pos = position or job.position
    if pos not in POSITIONS:
        raise ValueError(f"{job_path}: welding position {pos!r} not supported - this package runs 1G jobs (job.position 1G or --position 1G)")
    g = AutoTeach.Goal()
    g.frame_id = ""
    g.rough_start.x, g.rough_start.y, g.rough_start.z = (float(v) for v in job.start)
    g.rough_end.x, g.rough_end.y, g.rough_end.z = (float(v) for v in job.end)
    g.position = str(pos)
    g.job_file = str(Path(job_path).resolve())
    g.confirm_mode = str(confirm_mode)
    g.max_passes = int(max_passes)
    g.measure_speed_mps = v_ms
    g.dry_run = bool(dry_run)
    g.out_name = str(out_name)
    from scan_l0.job_overrides import to_goal_text
    g.job_overrides = to_goal_text(job_overrides or {})
    return g


def measure_speed_mps(mm_s: Optional[float]) -> float:
    """--measure-speed [mm/s] (None = not given) → the goal's measure_speed_mps [m/s] (0 = the job file's speed_mps). ValueError for
    a speed that is not finite and > 0 (the calibrated range is the server's check: the calibration file is on the server)."""
    if mm_s is None:
        return 0.0
    v = float(mm_s)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"--measure-speed must be a speed > 0 mm/s, got {mm_s!r}")
    return v * 1e-3


def cli_overrides(set_items=(), set_file: Optional[str] = None) -> dict:
    """--set-file FILE (a YAML mapping; nested mappings merge) then the --set KEY=VALUE items on top → flat {path: value}.
    scan_l0.job_overrides.OverrideError for a malformed item / file."""
    from scan_l0 import job_overrides as J
    base = {}
    if set_file:
        p = Path(set_file).expanduser()
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            raise J.OverrideError(f"--set-file {p}: {e}") from None
        base = J.parse_mapping_text(text, what=f"--set-file {p}")
    return J.merge(base, J.parse_set_items(set_items or ()))


def check_overrides(job_path, overrides: dict, *, position: Optional[str] = None) -> list:
    """The overrides on the job file the goal names (the same check as the server's, without the server's site overrides) → the
    records (old -> new). OverrideError = refused before anything is sent."""
    from scan_l0 import config as C
    from scan_l0 import job_overrides as J
    if not overrides:
        return []
    job, doc = C.load_job(str(job_path))
    pos = position or str(job.position or "") or None
    if pos in POSITIONS and str(job.position or "") not in ("", pos):
        job, doc = C.load_job(str(C.JOBS_DIR / f"groove_10T_{pos}.yaml"))       # like the server: the goal's position's template
    _doc, recs = J.apply_overrides(doc, overrides, source="goal", position=pos)
    return recs


def overrides_help() -> str:
    """The --help epilog: the common override keys."""
    from scan_l0 import job_overrides as J
    w = max(len(k) for k, _ in J.COMMON_KEYS)
    return ("common --set keys (any key of the job blocks " + ", ".join(J.BLOCKS) + " that the job file, the job template or the "
            "block's parser knows is accepted; the server refuses the others by name):\n"
            + "\n".join(f"  {k:<{w}}  {d}" for k, d in J.COMMON_KEYS))


def measure_speed_text(v_mps: float) -> str:
    """The goal line's measurement speed: 'measurement speed 30 mm/s (run override)' / '... the job file's speed_mps'."""
    v = float(v_mps or 0.0)
    return f"measurement speed {v * 1e3:g} mm/s (run override)" if v > 0.0 else "measurement speed: the job file's speed_mps"


# ---- 입력 흐름 (r3 §5): 재사용 질문 + 시편 입력 — ask/say 를 넘겨 시험에서 입력을 주입한다 ------------------------------------------
REUSE_PROMPT = "Reuse the previous inputs? [u]se / [d]o not use: "
POSITIONS = ("1G",)                                         # this package: the 1G core only
# 늘 묻는 시편 값 (사용자 2026-10-10): (specimen 키, 문구, 단위)
SPEC_ASKED = (("root_gap_mm", "root gap", "mm"), ("bevel_angle_deg", "bevel angle", "deg"), ("weld_length_mm", "weld length", "mm"))
# 나머지 템플릿 값 — 보여 주기만, --edit-all (개발) 에서만 묻는다: (블록, 키, 문구, 단위)
SPEC_SHOWN = (("specimen", "thickness_mm", "thickness", "mm"), ("specimen", "root_face_mm", "root face", "mm"),
              ("qc_targets", "root_face_mm.tol", "root face tolerance +-", "mm"), ("specimen", "backing_thickness_mm", "backing", "mm"),
              ("specimen", "plate_widths_mm.0", "left plate width (each plate, without the gap)", "mm"),
              ("specimen", "plate_widths_mm.1", "right plate width (each plate, without the gap)", "mm"),
              ("specimen", "hi_lo_mm", "hi-lo", "mm"))
TOL_SHOWN = (("root_gap_mm", "root gap", "mm"), ("bevel_angle_deg", "bevel angle", "deg"), ("groove_depth_mm", "groove depth", "mm"),
             ("top_opening_mm", "top opening", "mm"), ("weld_length_mm", "weld length", "mm"), ("root_position_mm", "root position", "mm"),
             ("end_position_mm", "end position", "mm"))


def template_path(position: str) -> Path:
    from scan_l0 import config as C
    return Path(C.JOBS_DIR) / f"groove_10T_{position}.yaml"


def load_input(path) -> Optional[dict]:
    """이전 입력 (job 파일) → 문서, 없거나 못 읽으면 None (못 읽으면 그 사유를 문서 대신 {'_error': …})."""
    from scan_l0 import config as C
    p = Path(path).expanduser()
    if not p.exists():
        return None
    try:
        job, doc = C.load_job(str(p))
        C.specimen_from_doc(doc)
        return {**doc, "_position": job.position}
    except Exception as e:  # noqa: BLE001 - an unreadable file is shown, not reused
        return {"_error": f"{type(e).__name__}: {e}"}


def has_taught_points(doc: Optional[dict]) -> bool:
    """두 점이 교시됐나 (교시 도구가 쓴 teach.start / teach.end; 템플릿의 자리표시 점은 아니다)."""
    t = (doc or {}).get("teach") or {}
    return isinstance(t, dict) and bool(t.get("start")) and bool(t.get("end"))


def _f(v, fmt: str = ".1f") -> str:
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "-"


def _qc_tol(doc: dict, key: str):
    return ((doc.get("qc_targets") or {}).get(key) or {}).get("tol")


def _plate_widths(sp: dict) -> list:
    """Each plate's width [left, right] mm from a specimen block (plate_widths_mm; the old total plate_width_mm as two halves)."""
    w = sp.get("plate_widths_mm")
    if w is None:
        t = sp.get("plate_width_mm")
        return [None, None] if t is None else [0.5 * float(t)] * 2
    w = list(w) if isinstance(w, (list, tuple)) else [w]
    return (w * 2)[:2] if len(w) == 1 else w[:2]


def specimen_line(doc: dict) -> str:
    sp = doc.get("specimen") or {}
    return (f"{doc.get('_position') or (doc.get('job') or {}).get('position') or '?'} | root gap {_f(sp.get('root_gap_mm'))} mm | "
            f"bevel angle {_f(sp.get('bevel_angle_deg'))} deg | weld length {_f(sp.get('weld_length_mm'))} mm")


def template_lines(doc: dict) -> list:
    """시편 입력에서 묻지 않는 값 (템플릿 값) 두 줄."""
    sp = doc.get("specimen") or {}
    rf = (doc.get("qc_targets") or {}).get("root_face_mm") or {}
    a = (f"thickness {_f(sp.get('thickness_mm'))} mm, groove {sp.get('kind', '-')} (bevel side {sp.get('bevel_side', '-')}), root face "
         f"{_f(rf.get('design', sp.get('root_face_mm')))} +- {_f(rf.get('tol'))} mm, backing "
         + (f"{_f(sp.get('backing_thickness_mm'))} mm" if sp.get("backing") else "none")
         + f", plate widths left {_f(_plate_widths(sp)[0])} / right {_f(_plate_widths(sp)[1])} mm, hi-lo {_f(sp.get('hi_lo_mm'))} mm")
    b = "tolerances: " + ", ".join(f"{lab} +-{_f(_qc_tol(doc, k))} {u}" for k, lab, u in TOL_SHOWN if _qc_tol(doc, k) is not None)
    return [a, b]


def input_summary(path, doc: dict, *, setup: bool = False) -> list:
    """재사용 질문 앞에 보이는 이전 입력. setup (KIRO): 시편만 (두 점은 목표가, 시작 자세는 수락 때 팔 자세가 정한다)."""
    p = Path(path).expanduser()
    if "_error" in doc:
        return [f"previous input {p} is not readable ({doc['_error']}) - it is not reused"]
    saved = (doc.get("meta") or {}).get("saved") or "-"
    out = [f"previous inputs ({p}, saved {saved}):", f"  specimen    {specimen_line(doc)}"]
    out += [f"              {ln}" for ln in template_lines(doc)]
    if setup:
        return out
    sp = doc.get("start_pose") or {}
    out.append(f"  start pose  {sp.get('joints_deg')} deg" if sp.get("joints_deg") else
               "  start pose  none (the run starts from the current arm pose)")
    if has_taught_points(doc):
        j = doc.get("job") or {}
        out.append(f"  two points  S {[round(float(v), 4) for v in j.get('start', [])]}  E {[round(float(v), 4) for v in j.get('end', [])]} m")
    else:
        out.append("  two points  not taught (teaching follows)")
    return out


def ask_reuse(ask, say) -> bool:
    """'Reuse the previous inputs? [u]se / [d]o not use' → True (use) / False (do not use). 그 밖의 답은 다시 묻는다."""
    while True:
        a = str(ask(REUSE_PROMPT)).strip().lower()
        if a in ("u", "use"):
            return True
        if a in ("d", "do not use", "dont use", "don't use"):
            return False
        say("  answer u (use the previous inputs) or d (do not use them)")


def _ask_value(ask, say, label: str, unit: str, default, check) -> float:
    while True:
        a = str(ask(f"  {label} [{unit}] [{_f(default, 'g')}]: ")).strip()
        if a == "":
            return float(default)
        try:
            v = float(a)
        except ValueError:
            say(f"  not a number: {a!r}")
            continue
        why = check(v)
        if why:
            say(f"  {why}")
            continue
        return v


def _set_dotted(block: dict, key: str, v) -> None:
    k, _, sub = key.partition(".")
    if sub.isdigit():                                    # a list entry (plate_widths_mm.0 / .1)
        cur = _plate_widths(block) if k == "plate_widths_mm" else list(block.get(k) or [])
        cur = list(cur) + [None] * max(0, int(sub) + 1 - len(cur))
        cur[int(sub)] = v
        block[k] = cur
        if k == "plate_widths_mm":
            block.pop("plate_width_mm", None)              # the old total is superseded by the widths
    elif sub:
        block[k] = {**dict(block.get(k) or {}), sub: v}
    else:
        block[k] = v


def _get_dotted(block: dict, key: str):
    k, _, sub = key.partition(".")
    if sub.isdigit():
        cur = _plate_widths(block or {}) if k == "plate_widths_mm" else list((block or {}).get(k) or [])
        return cur[int(sub)] if int(sub) < len(cur) else None
    v = (block or {}).get(k)
    return (v or {}).get(sub) if sub else v


def ask_specimen(ask, say, *, position_default: Optional[str] = None, edit_all: bool = False) -> dict:
    """시편 입력: 용접 자세, 루트 갭, 개선각, 용접 길이 (Enter = 보이는 값). 기본값과 나머지 값은 그 자세의 템플릿
    (jobs/groove_10T_<pos>.yaml). 나머지 값은 보여 주기만 하고 edit_all (개발 옵션) 에서만 묻는다. 반환 = 템플릿 문서 + 입력값
    (값 검사는 GrooveSpec 그대로)."""
    import copy
    from scan_l0 import config as C
    say("specimen input (Enter = keep the value in brackets):")
    pos_d = position_default if position_default in POSITIONS else "1G"
    while True:
        a = str(ask(f"  welding position ({'|'.join(POSITIONS)}) [{pos_d}]: ")).strip().upper() or pos_d
        if a in POSITIONS:
            break
        say(f"  welding position must be {' or '.join(POSITIONS)}, got {a!r}")
    tpl = template_path(a)
    _job, doc = C.load_job(str(tpl))
    doc = copy.deepcopy(doc)
    sp = dict(doc.get("specimen") or {})
    qc = dict(doc.get("qc_targets") or {})

    def check(key):
        def f(v):
            try:
                tmp = dict(sp)
                _set_dotted(tmp, key, v)
                C.specimen_from_doc({"specimen": tmp})
            except ValueError as e:
                return str(e)
            return ""
        return f
    for key, lab, unit in SPEC_ASKED:
        sp[key] = _ask_value(ask, say, lab, unit, sp.get(key), check(key))
    if edit_all:
        say("  other values (--edit-all):")
        for blk, key, lab, unit in SPEC_SHOWN:
            tgt = sp if blk == "specimen" else qc
            cur = _get_dotted(tgt, key)
            if cur is None:
                continue
            v = _ask_value(ask, say, lab, unit, cur, check(key) if blk == "specimen" else (lambda x: "" if x >= 0.0 else "must be >= 0"))
            _set_dotted(tgt, key, v)
        for key, lab, unit in TOL_SHOWN:
            if _qc_tol({"qc_targets": qc}, key) is None:
                continue
            v = _ask_value(ask, say, f"{lab} tolerance +-", unit, _qc_tol({"qc_targets": qc}, key),
                           lambda x: "" if x > 0.0 else "a tolerance must be > 0")
            qc[key] = {**dict(qc.get(key) or {}), "tol": v}
        if "root_face_mm" in qc and "root_face_mm" in sp:
            qc["root_face_mm"] = {**dict(qc["root_face_mm"]), "design": float(sp["root_face_mm"])}
    doc["specimen"], doc["qc_targets"] = sp, qc
    doc["_position"], doc["_template"] = a, str(tpl)
    if not edit_all:
        say(f"  other values from the template {tpl.name} (change them with --edit-all):")
    for ln in template_lines(doc):
        say(f"    {ln}")
    return doc


def write_input_job(path, doc: dict, *, note: str) -> Path:
    """입력 job 파일: 템플릿의 job 블록 (자리표시 점) + 시편·허용치 + 템플릿의 정밀도 목표·용접 절차·용접 설계 블록, 교시·시작 자세 없음
    (교시 도구가 그 위에 다시 쓴다; KIRO 서버는 점을 목표에서 받는다). 정밀도 목표 (precision: 1G core 의 종료 판정) 가 빠지면 KIRO 의
    실행이 템플릿과 다른 종료 기준으로 돈다 (2026-10-12 KIRO stand-in 에서 발견)."""
    from scan_l0 import config as C
    job = C.job_from_dict(dict(doc.get("job") or {}))
    meta = {"input": note, "template": doc.get("_template"), "points": "placeholders until taught"}
    return C.save_job(Path(path).expanduser(), job, {}, meta, specimen=C.specimen_from_doc(doc), qc_targets=doc.get("qc_targets"),
                      precision=doc.get("precision"), weld_plan=doc.get("weld_plan"), weld_design=doc.get("weld_design"))


# ---- 진행 줄 (r3 §8-1): 하나도 빠짐없이, 한 번씩 ---------------------------------------------------------------------------------------
def log_topic_candidates(action: str) -> list:
    """서버 로그 토픽 후보 (액션 이름의 네임스페이스 기준): 서버 노드의 ~/log, 그다음 <ns>/log."""
    ns = action.rsplit("/", 1)[0]
    return [f"{ns}/hongik_auto_teach_server/log", f"{ns}/log"]


class ProgressPrinter:
    """진행 줄을 하나도 빼지 않고 한 번씩 찍는다. 서버 로그 토픽이 있으면 (use_log) 그 줄을 모두 — 피드백 progress_text 는 같은 줄의 사본이고
    확인 안내 (' -> in another terminal: …') 도 서버가 로그 줄로 내므로 찍지 않는다. 없으면 피드백 progress_text 를 (바로 앞과 같은 줄의
    되풀이만 빼고) 모두."""

    def __init__(self, out=None, *, use_log: bool = False):
        self.out = out or (lambda s: print(s, flush=True))
        self.use_log = bool(use_log)
        self._last_fb: Optional[str] = None
        self.last_log: Optional[str] = None
        self.n_log = 0
        self._lock = threading.Lock()

    def on_log(self, text: str) -> None:
        with self._lock:
            self.n_log += 1
            self.last_log = str(text)
            self.out(str(text))

    def on_feedback(self, text: str, awaiting_confirmation: bool = False) -> None:
        text = str(text or "")
        with self._lock:
            if self.use_log:
                return
            if text and text != self._last_fb:
                self._last_fb = text
                self.out(text)


def _num(v, fmt: str = ".2f", scale: float = 1.0) -> str:
    try:
        x = float(v) * scale
    except (TypeError, ValueError):
        return "-"
    return format(x, fmt) if x == x else "-"


def sections_lines(doc: dict, path=None) -> list:
    """1G core 의 단면 형상 (sections.json, G1_CORE_SPEC §9) → 짧은 요약 줄: 판정 (σ ≤ 정밀도 목표), 양끝 (σ, 양쪽 관측), 길이 vs 설계,
    단면 수와 치수 중앙값, 베벨 쪽·판 롤. 없는 키는 '-'."""
    d = doc or {}
    done = d.get("done") or {}
    out = [f"sections ({d.get('schema', '?')}): status {d.get('status', '-')}, resolved {done.get('resolved', '-')}, max sigma/precision "
           f"{_num(done.get('max_ratio'))}" + (f", unresolved {done.get('unresolved')}" if done.get("unresolved") else "")]
    ends = d.get("ends") or {}
    if ends:
        out.append("  ends: " + "; ".join(f"{X} {[round(float(v), 4) for v in (e.get('point_m') or [])]} m +- "
                                          f"{_num(e.get('sigma_m'), '.2f', 1e3)} mm ({'two-sided' if e.get('two_sided') else 'one-sided'})"
                                          for X, e in ends.items()))
    ln = d.get("length_mm") or {}
    if ln:
        out.append(f"  length {_num(ln.get('value'))} +- {_num(ln.get('sigma'))} mm (design {_num(ln.get('design'), '.1f')} mm)")
    st = d.get("stations") or []
    if st:
        meds = []
        for q in ("gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm"):
            vals = [float(x[q]["value"]) for x in st if isinstance(x.get(q), dict) and x[q].get("value") is not None]
            if vals:
                vals.sort()
                meds.append(f"{q} {vals[len(vals) // 2]:.2f}")
        out.append(f"  {len(st)} stations" + (": median " + ", ".join(meds) if meds else ""))
    pl = d.get("plate") or {}
    if d.get("bevel_side") or pl:
        out.append(f"  bevel side {d.get('bevel_side', '-')}, plate roll {_num(pl.get('roll_deg'))} deg, misalignment "
                   f"{_num(pl.get('misalignment_deg'))} deg")
    if path is not None:
        out.append(f"  cross sections: {path} + {Path(path).with_suffix('.npz')}")
    return out


DELIVERED = ("PASS", "PASS_UNRESOLVED", "DRY_RUN")          # 클라이언트 종료 코드 0 (PASS_UNRESOLVED: 출력은 썼다 — 미해결 줄과 함께)


def result_lines(res) -> list:
    """액션 결과 (+ 출력 폴더의 sections.json · weld_plan.json · specimen_report.json) → 출력 줄. 용접 전 스캔의 결과물 = 단면 + 다층 용접
    경로. 쓰지 않는 결과 필드 (root_centreline; 1G 의 빈 refined_job_path) 는 찍지 않는다 (r3 §8-4). PASS_UNRESOLVED (r3 §8-6) 는 미해결
    목록을 따로 한 줄."""
    out = [f"RESULT: {res.status} - {res.reason}"]
    if str(res.status) == "ABORTED":
        out.append("  The run ended early (action cancel or a failed motion - see the reason); nothing more was sent in this run.")
        out.append("  What to do: check the cell and the reason above, then send the next goal - no reset is needed; the server starts "
                   "it when MoveIt, the trajectory controller, /joint_states and TF are available.")
    sec_doc = None
    sec = Path(res.out_dir) / "sections.json" if res.out_dir else None
    if sec is not None and sec.exists():
        try:
            sec_doc = json.loads(sec.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            out.append(f"  sections.json unreadable: {e}")
    if str(res.status) == "PASS_UNRESOLVED":
        unres = ((sec_doc or {}).get("done") or {}).get("unresolved")
        if not unres and "unresolved: " in str(res.reason):
            unres = str(res.reason).split("unresolved: ", 1)[1].split(" (", 1)[0]
        out.append(f"  UNRESOLVED after the pass budget: {unres if unres else '?'} - the outputs are written; these quantities are "
                   "still above their 1-sigma precision target: check them before welding")
    if res.out_dir:
        out.append(f"  out_dir {res.out_dir}" + (f" | refined job {res.refined_job_path}" if res.refined_job_path else ""))
    if res.length_mm == res.length_mm:                         # not NaN
        out.append(f"  ends observed {res.ends_observed} | length {res.length_mm:.2f} mm | scan passes {res.n_passes} | moves "
                   f"{res.n_moves} | {res.elapsed_s:.1f} s | converged {res.converged or '-'}")
    if sec_doc is not None:                                  # 1G core 의 결과물 1: 단면 형상
        out.extend(sections_lines(sec_doc, sec))
    plan = Path(res.out_dir) / "weld_plan.json" if res.out_dir else None
    if plan is not None and plan.exists():
        from scan_core.weld_plan import format_plan_summary
        out.extend(format_plan_summary(json.loads(plan.read_text(encoding="utf-8"))))
        out.append(f"  weld paths: {plan} + {plan.with_suffix('.npz')} (p<k>_t_s, _pos_m, _R per pass)")
    rep = Path(res.out_dir) / "specimen_report.json" if res.out_dir else None
    if rep is not None and rep.exists():
        from scan_l0.autoteach import format_specimen_table
        out.extend(format_specimen_table(json.loads(rep.read_text(encoding="utf-8"))))
    return out


def rejection_lines(status_text: Optional[str]) -> list:
    """목표 거절 → 사유 줄 (서버 ~/status 응답에서). 서버는 다른 실행 중·플랫폼 준비 안 됨·자세 누락일 때 거절한다. 정지 래치는 없다 —
    준비가 안 된 항목과 할 일을 보여 준다."""
    if not status_text:
        return ["REFUSED: goal rejected by the server (no status answer; see the server log in terminal 1)"]
    if "ready=False" in status_text:
        nr = status_text.split("not_ready=", 1)[1].split(" run_stopped=", 1)[0] if "not_ready=" in status_text else "?"
        return [f"REFUSED: the platform is not ready: {nr.strip(chr(39))}",
                "  What to do: fix these items (start move_group / the trajectory controller / robot_state_publisher of the robot), "
                "then send the goal again - there is nothing to reset on our side."]
    if "running=True" in status_text:
        return ["REFUSED: another auto-teach run is active on the server (wait for it or cancel it)"]
    return ["REFUSED: goal rejected by the server (check the position: 1G; see the server log)", f"  server status: {status_text}"]


CANCEL_ANSWER_S = 10.0          # Ctrl-C 뒤 서버의 취소 응답 (그리고 목표 수락 응답) 대기
CANCEL_RESULT_S = 30.0          # 취소 뒤 결과 대기 (즉시 정지 → 실행이 끝나고 결과를 낸다)


def _wait(fut, timeout_s: float, dt: float = 0.05) -> bool:
    t_end = time.monotonic() + timeout_s
    while not fut.done() and time.monotonic() < t_end:
        time.sleep(dt)
    return fut.done()


def _cancel_goal(gh, res_fut) -> bool:
    """Ctrl-C → 목표 취소 = 서버의 즉시 정지 (그 실행만). 결과가 오면 True. (컨텍스트가 살아 있어야 한다: rclpy.init 의 SIGINT 처리기 끔)"""
    print("\nCtrl-C: cancelling the goal (the server stops our trajectory and ends the run; no reset needed)", flush=True)
    cf = gh.cancel_goal_async()
    if not _wait(cf, CANCEL_ANSWER_S):
        print(f"  no answer to the cancel request within {CANCEL_ANSWER_S:.0f} s - stop the robot with the robot's own stop", flush=True)
    elif cf.result() is not None and not cf.result().goals_canceling:
        print("  the server did not accept the cancel (goal already finished?)", flush=True)
    return _wait(res_fut, CANCEL_RESULT_S, 0.1)


LOG_DRAIN_S = 2.0               # 결과 뒤 런의 마지막 로그 줄 기다림 (로그 토픽과 결과는 따로 온다)
LOG_DISCOVERY_S = 1.0           # 액션 서버를 찾은 뒤 로그 토픽 발행자 찾기


def _input_flow(a, ask, say) -> tuple:
    """--setup 의 입력 단계 → (rc, 용접 자세)."""
    job = Path(a.job).expanduser()
    prev = load_input(job)
    reuse = False
    if prev is not None:
        for ln in input_summary(job, prev, setup=a.setup):
            say(ln)
        if "_error" not in prev:
            reuse = ask_reuse(ask, say)
    if reuse:
        pos = str(prev.get("_position") or "")
        if a.position and a.position != pos:
            say(f"REFUSED: --position {a.position} differs from the previous input's welding position {pos or '?'} - answer d (do not "
                "use) to enter the specimen again, or drop --position")
            return 2, pos
        if pos not in POSITIONS:
            say(f"REFUSED: the previous input's welding position {pos or '?'} is not supported (this package runs 1G jobs) - answer d "
                "(do not use) to enter the specimen again")
            return 2, pos
        say(f"kept {job} unchanged")
        return 0, pos
    doc = ask_specimen(ask, say, position_default=a.position or (prev or {}).get("_position"), edit_all=a.edit_all)
    write_input_job(job, doc, note="hongik_auto_teach_client --setup")
    say(f"wrote {job}: {specimen_line(doc)}")
    return 0, doc["_position"]


def build_parser() -> argparse.ArgumentParser:
    """명령줄 인자 (main 과 시험이 같은 것을 쓴다)."""
    ap = argparse.ArgumentParser(prog="hongik_auto_teach_client", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=overrides_help())
    ap.add_argument("--job", required=True, help="job YAML: the input file (specimen; for a run also the two points job.start / job.end)")
    ap.add_argument("--out", default="", help="output folder name under the server's out_root ('' = timestamp)")
    ap.add_argument("--confirm", default="auto", choices=["each", "large", "scan", "auto", "none"],
                    help="per-move confirmation (~/confirm_move): each = every move, large = moves beyond the clearance, scan = only the "
                         "scan moves across the specimen (S->E transit, passes) and joint-space moves; auto (default) = only joint-space "
                         "moves that are not a straight joint line (MoveIt OMPL detour); none = never; fine moves always run "
                         "automatically except in each")
    ap.add_argument("--max-passes", type=int, default=0)
    ap.add_argument("--measure-speed", type=float, default=None, metavar="MM_S",
                    help="measurement speed for this run [mm/s]: the scan passes and the precise corrections at the dwell sites "
                         "(positioning moves stay at the platform limits); any speed inside the calibration's noise-prior speed range "
                         "(g1 noise_priors speed_range_mps; the server refuses others before any motion); default = the job file's speed_mps")
    ap.add_argument("--set", dest="set_items", action="append", default=[], metavar="KEY=VALUE",
                    help="run-time job override (repeatable): KEY = dotted path into the job document (e.g. "
                         "precision.end_position_mm=0.2, weld_plan.stick_out_mm=17.5, weld_design.end_inset_mm.goal=79.7, "
                         "'weld_plan.travel_angle_deg=[-40,-40,-40]'), VALUE = a YAML value; the job file is not changed")
    ap.add_argument("--set-file", default=None, metavar="FILE",
                    help="run-time job overrides from a YAML mapping (nested or dotted keys); the --set items go on top")
    ap.add_argument("--position", default=None, choices=list(POSITIONS),
                    help="welding position: overrides job.position for a run; with --setup the default of the specimen input")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--action", default=ACTION)
    ap.add_argument("--server-timeout", type=float, default=30.0)
    ap.add_argument("--log-topic", default="",
                    help="server log topic (std_msgs/String, one message per progress line); '' = find it (<ns>/hongik_auto_teach_server/log "
                         "or <ns>/log), 'none' = use the action feedback only")
    ap.add_argument("--setup", action="store_true",
                    help="specimen input only (reuse question + specimen; no motion, no run) - the server reads this job file (job_file:=)")
    ap.add_argument("--edit-all", action="store_true",
                    help="development: also edit the other template values (thickness, root face, backing, plate width, tolerances)")
    return ap


def main(argv=None, *, ask=None, say=None) -> int:
    """ask / say: 입력·출력 함수 (기본 input / print) — 시험이 입력 흐름에 답을 주입한다."""
    ask = ask or input
    say = say or (lambda s_: print(s_, flush=True))
    a = build_parser().parse_args(argv)
    from scan_l0 import job_overrides as J
    try:
        overrides = cli_overrides(a.set_items, a.set_file)
        v_meas = measure_speed_mps(a.measure_speed)
    except ValueError as e:                                    # J.OverrideError is a ValueError
        say(f"REFUSED: {e}")
        return 2
    if overrides and a.setup:
        say("REFUSED: --set / --set-file are run-time overrides of a run; --setup only writes the specimen input (no run)")
        return 2
    if v_meas > 0.0 and "job.speed_mps" in overrides:
        say(f"REFUSED: the measurement speed is given twice (--measure-speed {a.measure_speed:g} and --set job.speed_mps="
            f"{overrides['job.speed_mps']!r}) - give it once")
        return 2
    if a.setup:
        try:
            rc_i, _position = _input_flow(a, ask, say)
        except (EOFError, KeyboardInterrupt):
            say("\nnot written: the input was not completed")
            return 1
        return rc_i
    position = a.position

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String
    from hongik_scan_msgs.action import AutoTeach

    if overrides:
        try:
            recs = check_overrides(Path(a.job).expanduser(), overrides, position=position)
        except ValueError as e:                                # J.OverrideError: the same check the server makes
            say(f"REFUSED: {e} (nothing sent)")
            return 2
        say(f"job overrides for this run ({len(recs)}; on {a.job}; the server applies its site overrides first): "
            + "; ".join(J.record_lines(recs)))
    goal = goal_from_job(Path(a.job).expanduser(), out_name=a.out, confirm_mode=a.confirm,
                         max_passes=a.max_passes, dry_run=a.dry_run, position=position, measure_speed_mps=v_meas,
                         job_overrides=overrides)
    if not rclpy.ok():
        # 기본 SIGINT 처리기는 Ctrl-C 에 컨텍스트를 내린다 → 스핀 스레드가 죽어 취소가 서버에 가지 않았다 (2026-10-08 확인).
        # NO: Ctrl-C = KeyboardInterrupt, 컨텍스트는 살아 있다.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("hongik_auto_teach_client")
    client = ActionClient(node, AutoTeach, a.action)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    rc = 1
    try:
        if not client.wait_for_server(timeout_sec=a.server_timeout):
            say(f"REFUSED: action server {a.action} not available (is hongik_moveit.launch.py running?)")
            return 1
        # 진행 줄 (r3 §8-1): 서버 로그 토픽 (줄마다 하나, depth 1000) 이 있으면 그것을 모두, 없으면 피드백 문구를 모두
        topic = None
        if str(a.log_topic).strip().lower() != "none":
            cands = [a.log_topic] if a.log_topic else log_topic_candidates(a.action)
            t_end = time.monotonic() + LOG_DISCOVERY_S
            while topic is None:
                topic = next((t for t in cands if node.count_publishers(t) > 0), None)
                if topic is not None or time.monotonic() >= t_end:
                    break
                time.sleep(0.05)
        pr = ProgressPrinter(say, use_log=topic is not None)
        if topic is not None:
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1000, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
            node.create_subscription(String, topic, lambda m: pr.on_log(m.data), qos)

        def on_fb(fb):
            f = fb.feedback
            pr.on_feedback(f.progress_text, bool(f.awaiting_confirmation))

        say(f"goal: {a.job} rough S [{goal.rough_start.x:.4f}, {goal.rough_start.y:.4f}, {goal.rough_start.z:.4f}] E "
            f"[{goal.rough_end.x:.4f}, {goal.rough_end.y:.4f}, {goal.rough_end.z:.4f}] m, position {goal.position}, "
            f"confirm {goal.confirm_mode}, out '{goal.out_name}', {measure_speed_text(goal.measure_speed_mps)} "
            "(Ctrl-C = cancel = stop the run); progress "
            + (f"from {topic}" if topic is not None else "from the action feedback"))
        fut = client.send_goal_async(goal, feedback_callback=on_fb)
        try:
            while not fut.done():
                time.sleep(0.05)
        except KeyboardInterrupt:
            # 목표를 보내는 중의 Ctrl-C: 수락되면 바로 취소한다 (서버가 이미 움직이기 시작했을 수 있다)
            if _wait(fut, CANCEL_ANSWER_S) and fut.result() is not None and fut.result().accepted:
                gh = fut.result()
                _cancel_goal(gh, gh.get_result_async())
            else:
                say("\nCtrl-C: the goal was not accepted (nothing to cancel)")
            return 1
        gh = fut.result()
        if gh is None or not gh.accepted:
            status_text = None
            try:
                from std_srvs.srv import Trigger
                cli = node.create_client(Trigger, a.action.rsplit("/", 1)[0] + "/hongik_auto_teach_server/status")
                if cli.wait_for_service(timeout_sec=2.0):
                    sf = cli.call_async(Trigger.Request())
                    t_end = time.monotonic() + 3.0
                    while not sf.done() and time.monotonic() < t_end:
                        time.sleep(0.05)
                    if sf.done() and sf.result() is not None:
                        status_text = str(sf.result().message)
            except Exception:  # noqa: BLE001 - the hint is best effort
                status_text = None
            say("\n".join(rejection_lines(status_text)))
            return 1
        res_fut = gh.get_result_async()
        try:
            while not res_fut.done():
                time.sleep(0.1)
        except KeyboardInterrupt:
            if not _cancel_goal(gh, res_fut):
                return 1
        if pr.use_log:
            # 로그 토픽과 결과는 따로 온다: 런의 마지막 줄 ('auto-teach <상태>: …') 이 오기 전이면 잠깐 기다린다 (줄을 잃지 않게)
            t_end = time.monotonic() + LOG_DRAIN_S
            while not str(pr.last_log or "").startswith("auto-teach ") and time.monotonic() < t_end:
                time.sleep(0.05)
        res = res_fut.result().result
        for line in result_lines(res):
            say(line)
        rc = 0 if str(res.status) in DELIVERED else 1
    finally:
        # 실행기를 먼저 멈추고 스핀 스레드를 기다린 뒤 노드·rclpy 를 내린다 (돌고 있는 스핀 스레드 위에서 끄면 C++ 쪽이 terminate)
        executor.shutdown()
        spin.join(timeout=5.0)
        client.destroy()
        node.destroy_node()
        rclpy.try_shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
