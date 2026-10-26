"""hongik_auto_teach_server - the action server of the Hongik 1G precision-sensing package (namespace /hongik_fine_sensing).

Actions (one goal at a time over both):
  precision_sensing  KIRO welding_interfaces/action/PrecisionSensing: welding_position 1 (1G) + pose_a / pose_b (approximate seam
                     points, positions in the base frame) -> error_code / error_message, WeldingPath[] (4-pass weld plan), data_file_path
                     (precision_sensing.npz on this PC). Goals are always accepted; a goal that cannot run (unsupported position, another
                     run active, platform not ready, sensor busy) is aborted at once with the reason. Feedback: elapsed_time [s] at 1 Hz
                     and on every log line. Mapping in hongik_scan_ros.kiro_io.
  auto_teach         hongik_scan_msgs/action/AutoTeach (our console / test interface, hongik_auto_teach_client): rough S / E + position
                     1G + optional per-move confirmation (~/confirm_move).
Both run the same path: the 1G core (scan_l0.teach_1g.Teach1G) through the MoveIt adapter (MoveItPlatform: the platform's move_group +
JointTrajectoryController + /joint_states + TF). The start pose is the arm pose at goal acceptance. Specimen, tolerances and process come
from the job file on this PC (parameter job_file, re-read per goal). Calib T_flange<-opt / T_flange<-eoat and the gravity axis are read
from TF (gravity.frame). Every progress / decision line is also published on ~/log (std_msgs/String, one line per message, reliable,
depth 1000).

Motion: every straight leg is checked before it is sent by our envelope (plate plane, support, clamp, laser tilt, transit height) and by
MoveIt (Cartesian path with collision checking against the planning scene, which holds our cell-world objects 'hongik/specimen' and
'hongik/floor'); joint-space moves are planned and checked by MoveIt. A cancel of the running goal is the immediate stop of our own
trajectory on the JTC and ends the run (ABORTED with the reason); there is no latch and nothing to reset - the next goal starts when
MoveIt / JTC / joint states / TF are available. KIRO's main system sends the first command (the goal) and the run goes through the
platform's MoveIt and JTC only: this package has no robot-controller concepts (no controller mode, no arm activation, no controller
status, no tool / payload / collision / speed settings) - those belong to KIRO's controller.

The sensor is owned by this server during a run (it asks hongik_profiler_node to release it first when that node runs). Results in
out_root/<run>/: sections.json/.npz (cross sections), weld_plan.json/.npz (multi-pass weld paths), precision_sensing.npz (KIRO), and
verification/. Parameters: platform_config, calib_file, job_file, out_root, confirm_mode, kiro_confirm_mode (default none),
sensor_ip / sensor_port, speeds. All text the server prints, publishes or returns is English.
The full result message of every PrecisionSensing goal whose run made a folder also goes into that folder as
precision_sensing_result.yaml (user 2026-10-13; kiro_io.write_result_yaml), with the line 'precision_sensing: result file <path>'. The
operator's terminal client for this action is hongik_precision_sensing (precision_sensing_client: the two points by coordinates);
~/status names the base frame the goal points are read in (base_frame=).

Run-time job overrides (user 2026-10-13: "parameters must be adjustable per run - site conditions differ"; scan_l0.job_overrides):
the server parameter job_overrides (a YAML or JSON mapping {"<dotted path into the job document>": value}, e.g.
'{precision.end_position_mm: 0.2, weld_plan.stick_out_mm: 17.5}'; default '' = none; read and checked at start) = site-level
overrides applied to every goal of both actions (KIRO's PrecisionSensing goal cannot carry any: this is how KIRO's site adjusts);
the AutoTeach goal's job_overrides (JSON object; client --set / --set-file) on top. Both are applied in build_rough_job right after
the job file is loaded, before anything is planned; an unknown key, a key the run does not take from the job file or a value the
job parser refuses ends the goal REFUSED (PrecisionSensing: the input error) with nothing sent. Recorded: one server line per run
('job overrides: ...'), verification/run_log.json job_file / job_overrides [{path, old, new, source, old_from}] / job_effective,
the effective job document verification/job_effective.yaml, and (KIRO) the precision_sensing.npz meta inputs. A measurement speed
chosen at run time (goal measure_speed_mps, a job.speed_mps override) must lie inside the noise priors' calibrated speed range
(the 1G core's gates; scan_core.noise_priors.check_measure_speed).

Post-weld data acquisition (user 2026-10-13): a third action /hongik_fine_sensing/post_weld_scan (hongik_scan_msgs/action/PostWeldScan)
on the same platform, sensor and run slot - after the welding, one straight scan of the whole weld along the measured seam of a finished
1G run (goal run_dir, '' = the latest under out_root) at that run's pass standoff / orientation and the measurement speed - after
the last pass or after any pass (goal after_pass; the caller decides when); output <run_dir>/post_weld_scan_p<k>|final_<stamp>/
(raw.npz + run_log.json). Data acquisition only (scan_l0.post_weld.PostWeldScan). Goals are always accepted; a goal that cannot run
(another run active, bad input run, platform not ready, sensor busy) ends REFUSED with the reason in the result. KIRO's
PrecisionSensing interface is unchanged.

Start and shutdown (KIRO field 2026-10-14). At start: SciPy must load against the loaded NumPy (numeric_env, before any numeric
import: else one line with the fix, exit 1); one record line of the site parameters (platform_config, job_file, calib_file, out_root,
confirm_mode, kiro_confirm_mode) and one of the numeric packages loaded (numpy / scipy version and file); objects of ours left in the
MoveIt scene by a run that did not end cleanly ('hongik/': the cell world and D14's weld scene) are removed - again before every run.
Every service wait of the platform ends on a stop of the run or the server's shutdown (ros_util.call_service; the planning-scene
clients excepted). SIGINT / SIGTERM: no new run; the running goal is stopped as a cancel stops it and the server waits for its execute
callback to return - the run reports ABORTED 'server shutting down' and removes its MoveIt objects while the platform node still exists
- then removes any object of ours still in the scene and closes the platform. A second signal stops the waiting.
"""
from __future__ import annotations

from .numeric_env import numeric_record_text, require_numeric_stack

require_numeric_stack("hongik_auto_teach_server")     # before the numeric imports below: SystemExit with the fix (KIRO NumPy 2)

import datetime as _dt  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import Future, InvalidStateError  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402

import rclpy  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import SetBool, Trigger  # noqa: E402

from hongik_scan_msgs.action import AutoTeach as AutoTeachAction  # noqa: E402
from hongik_scan_msgs.action import PostWeldScan as PostWeldScanAction  # noqa: E402
from hongik_scan_msgs.msg import ProfileRaw  # noqa: E402
from scan_l0 import config as C  # noqa: E402
from scan_l0 import job_overrides as J  # noqa: E402
from scan_l0.autoteach import CONFIRM_MODES, AutoTeach  # noqa: E402
from scan_l0.teach_1g import G1_PIPELINES, select_teach_class  # noqa: E402
from scan_l0.ports import MotionError, PlatformConfigError  # noqa: E402
from scan_l0.acquisition import rough_normal  # noqa: E402
from scan_l0.cell_world import NAMESPACE as OUR_SCENE_PREFIX  # noqa: E402  ('hongik/': cell world + D14 weld scene 'hongik/weld/')
from scan_l0.executor import CANCEL_OPERATOR, CANCEL_SERVER_SHUTDOWN  # noqa: E402
from scan_l0.session import ScanSession  # noqa: E402

from . import kiro_io as K  # noqa: E402
from .profiler_node import profile_msg  # noqa: E402
from .ros_util import SPIN_PERIOD_S, call_service, point_msg  # noqa: E402

__all__ = ["AutoTeachServer", "main", "build_rough_job", "safe_out_name", "rough_normal", "kiro_action_type", "post_weld_inputs",
           "POST_WELD_OK", "scene_object_ids", "claim_refusal"]

KIRO_ACTION = "welding_interfaces/action/PrecisionSensing"


def kiro_action_type():
    """The KIRO action type with its C type support loaded → (type, None), or (None, one English message with the fix). The server needs
    it to start (it offers the KIRO action on every platform)."""
    try:
        from welding_interfaces.action import PrecisionSensing
        from rclpy.type_support import check_for_type_support
        check_for_type_support(PrecisionSensing)
        return PrecisionSensing, None
    except Exception as e:  # noqa: BLE001 - ImportError, missing type support library, ...
        return None, (f"hongik_auto_teach_server: cannot load the KIRO action type {KIRO_ACTION} ({type(e).__name__}: {e}). "
                      "Fix: build KIRO's welding_interfaces into a workspace and source its setup.bash before starting the server - "
                      "INSTALL.md step 2 (its package.xml declares no build type: run tools/colcon_meta_ament_cmake.py in the "
                      f"workspace before colcon build); check with 'ros2 interface show {KIRO_ACTION}'.")

DEFAULT_OUT_ROOT = os.path.expanduser("~/scan_data/hongik_auto_teach")


def safe_out_name(name) -> str:
    """목표의 out_name → 폴더 이름 하나 (경로 구분자·'..' 금지; 허용 문자 밖은 '_'). '' = 이름 없음."""
    import re
    n = re.sub(r"[^A-Za-z0-9._-]", "_", str(name or "").strip())
    return "" if n in ("", ".", "..") else n.strip(".") or ""


GOAL_SPECIMEN_FIELDS = (("weld_length_mm", "weld_length_mm"), ("thickness_mm", "thickness_mm"), ("root_gap_mm", "root_gap_mm"),
                        ("bevel_angle_deg", "bevel_angle_deg"), ("groove_kind", "kind"), ("bevel_side", "bevel_side"))


def _goal_specimen_paths(goal) -> dict:
    """The specimen keys the AutoTeach goal's own design fields set (> 0 / non-empty) → {path: value}."""
    out = {}
    for f_, key in GOAL_SPECIMEN_FIELDS:
        v = getattr(goal, f_, None)
        if isinstance(v, str):
            if v:
                out[f"specimen.{key}"] = str(v)
        elif v and float(v) > 0.0:
            out[f"specimen.{key}"] = float(v)
    return out


def build_rough_job(settings, job_file: str, goal, *, base_frame: str, gravity_axis_base=None, start=None, end=None,
                    overrides=()) -> tuple:
    """목표 (rough S/E, 자세, 설계값) + job 파일 (설계·QC·공정 블록) → (ScanJob, doc). 점은 base 프레임 (호출자가 변환). meta.rough = true.
    범용 시작 (사용자 2026-10-07): 작업의 기하는 착점 좌표·자세·중력에서만 — 판 법선 = rough_normal (템플릿 축이 아니다), 판 상단·
    테이블은 비운다 (auto-teach 가 착점·설계 길이·센서 측정으로 정한다; 예전 3G '위쪽 점 + 0.10 m' 같은 고정 여유 없음).
    중력축: gravity_axis_base 가 주어지면 (MoveIt 어댑터: TF 유도) job 에 그대로 넣는다.
    overrides: run-time job overrides [(source, {path: value}), ...] in order ('server parameter', then 'goal'; scan_l0.job_overrides)
    applied right after the job file is loaded: the server layer first, then the goal's own design fields (weld_length_mm, ...),
    then the goal layer (a path the goal sets both ways = OverrideError). The document's meta carries job_file (the file loaded) and
    job_overrides (the records). OverrideError (a ValueError) for an override that cannot be applied - nothing is planned."""
    job, doc = C.load_job(job_file)
    pos = str(goal.position or "").strip()
    notes = []
    loaded = str(job_file)
    if pos in ("1G", "3G") and str(job.position or "") not in ("", pos):
        # 서버 기본 job 은 1G — 3G 목표면 3G 템플릿의 설계·공정 블록을 쓴다 (기하는 아래에서 착점으로)
        tmpl = str(C.JOBS_DIR / f"groove_10T_{pos}.yaml")
        notes.append(f"job file {job_file} is {job.position}; the {pos} goal uses {tmpl}")
        sp = doc.get("start_pose")                                  # 손으로 지정한 시작 자세는 job 파일의 것 (템플릿에는 없다)
        job, doc = C.load_job(tmpl)
        loaded = tmpl
        if sp:
            doc = {**doc, "start_pose": sp}
    layers = [(str(src), dict(ov or {})) for src, ov in (overrides or ())]
    pos_ov = pos or str(job.position or "") or None
    goal_spec = _goal_specimen_paths(goal)

    def with_goal_spec(dc: dict) -> dict:                           # the goal's own design fields (specimen)
        dc = dict(dc)
        dc["specimen"] = {**dict(dc.get("specimen", {}) or {}), **{p_.split(".", 1)[1]: v_ for p_, v_ in goal_spec.items()}}
        return dc
    recs: list = []
    spec_done = False
    for src, ov in layers:
        if src == "goal":
            both = sorted(set(ov) & set(goal_spec))
            if both:
                raise J.OverrideError(f"job override {', '.join(both)}: given twice in the goal (its own design field and "
                                      "job_overrides) - give it once")
            doc, spec_done = with_goal_spec(doc), True
        doc, r = J.apply_overrides(doc, ov, source=src, position=pos_ov, earlier=recs)
        recs.extend(r)
    if not spec_done:
        doc = with_goal_spec(doc)
    job = C.job_from_dict(dict(doc.get("job") or {}))
    d = C.job_to_dict(job)
    S = np.asarray(start if start is not None else [goal.rough_start.x, goal.rough_start.y, goal.rough_start.z], dtype=float)
    E = np.asarray(end if end is not None else [goal.rough_end.x, goal.rough_end.y, goal.rough_end.z], dtype=float)
    g = gravity_axis_base if gravity_axis_base is not None else job.gravity_axis_base
    n = rough_normal(S, E, pos or str(job.position or ""), g)                 # scan_l0.acquisition
    d.update(start=S.tolist(), end=E.tolist(), base_frame=base_frame, job_id=f"{job.job_id or 'job'}_goal",
             surface_normal=[float(v) for v in n], specimen_top_z_m=None, table_z_m=None)
    if gravity_axis_base is not None:
        d["gravity_axis_base"] = [float(v) for v in np.asarray(gravity_axis_base, dtype=float).reshape(3)]
    if pos:
        d["position"] = pos
        d.pop("orientation", None)
    job = C.job_from_dict(d)
    doc = dict(doc)
    doc["job"] = C.job_to_dict(job)
    meta = dict(doc.get("meta", {}) or {})
    meta.update(rough=True, note="rough job from the AutoTeach action goal (points given in the base frame)", job_file=loaded,
                job_overrides=recs)
    if notes:
        meta["goal_notes"] = notes
    doc["meta"] = meta
    return job, doc


def goal_measure_speed(goal) -> Optional[float]:
    """The AutoTeach goal's measure_speed_mps → this run's measurement speed [m/s] (the job's speed_mps for the run: the passes and
    the precise corrections at the dwell sites), None = 0 = the job file's speed_mps. ValueError for a negative or non-finite value
    (the goal is refused before anything is sent). Goals without the field (other callers) read as 0."""
    v = float(getattr(goal, "measure_speed_mps", 0.0) or 0.0)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"measure_speed_mps must be 0 (the job's speed_mps) or a speed > 0 m/s, got {v!r}")
    return v if v > 0.0 else None


POST_WELD_OK = ("OK", "DRY_RUN")                 # PostWeldScan statuses that succeed the goal


def claim_refusal(busy: str) -> str:
    """AutoTeachServer._claim's answer (the goal holding the slot, or CANCEL_SERVER_SHUTDOWN) → the refusal text."""
    if busy == CANCEL_SERVER_SHUTDOWN:
        return f"{CANCEL_SERVER_SHUTDOWN}: no new run (nothing sent)"
    return f"busy: another sensing run is active ({busy}; nothing sent)"


def scene_object_ids(platform, timeout_s: float = 5.0) -> Optional[list]:
    """The ids of the world objects in move_group's planning scene (GetPlanningScene WORLD_OBJECT_NAMES on the platform's scene
    client; timeout = the adapter's wait for that service, MoveItPlatform.planning_frame) → [id, ...], None when move_group does not
    offer the service; PlatformUnavailable when it does not answer."""
    from moveit_msgs.msg import PlanningSceneComponents
    from moveit_msgs.srv import GetPlanningScene
    req = GetPlanningScene.Request()
    req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
    resp = call_service(platform.get_scene_client, req, timeout_s)
    if resp is None:
        return None
    return [str(co.id) for co in resp.scene.world.collision_objects]


def post_weld_inputs(goal, *, out_root: str, confirm_default: str) -> dict:
    """PostWeldScan goal → the scan's inputs {run_dir (Path), measure_speed_mps (None = the input run's measurement speed), confirm,
    dry_run, after_pass (0 = final)}. ValueError (scan_l0.post_weld.InputRunError included) with the reason - the goal
    is refused before anything is sent. (after_pass beyond the run's plan is refused when the run is read: InputRun.check_after_pass.)"""
    from scan_l0.post_weld import resolve_run_dir
    v = goal_measure_speed(goal)
    confirm = str(getattr(goal, "confirm_mode", "") or confirm_default).strip()
    if confirm not in CONFIRM_MODES:
        raise ValueError(f"confirm_mode must be one of {CONFIRM_MODES}, got {confirm!r}")
    run_dir = resolve_run_dir(str(getattr(goal, "run_dir", "") or ""), out_root)
    k = int(getattr(goal, "after_pass", 0) or 0)
    if k < 0:
        raise ValueError(f"after_pass must be >= 0 (0 = final), got {k}")
    d = str(getattr(goal, "direction", "") or "").strip().replace(" ", "")
    if d not in ("", "S->E", "E->S"):
        raise ValueError(f"direction must be 'S->E', 'E->S' or '' (the least TCP motion), got {d!r}")
    to = str(getattr(goal, "tool_orientation", "") or "").strip().lower().replace("-", "_") or "pass"
    if to not in ("pass", "least_rotation"):
        raise ValueError(f"tool_orientation must be 'pass' (default) or 'least_rotation', got {to!r}")
    return {"run_dir": run_dir, "measure_speed_mps": v, "confirm": confirm,
            "dry_run": bool(getattr(goal, "dry_run", False)), "after_pass": k, "direction": d, "tool_orientation": to}


class AutoTeachServer(Node):
    def __init__(self):
        super().__init__("hongik_auto_teach_server")
        P = self.declare_parameter
        self.platform_config = P("platform_config", "").value
        self.calib_file = P("calib_file", "").value
        self.job_file = P("job_file", str(C.JOBS_DIR / "groove_10T_1G.yaml")).value
        self.out_root = P("out_root", DEFAULT_OUT_ROOT).value
        self.confirm_mode = P("confirm_mode", "each").value
        # KIRO PrecisionSensing goals have no confirmation field: this mode (r3 §4.1: none). 'each'/'large'/'scan'/'auto' would wait on
        # ~/confirm_move like the AutoTeach goals
        self.kiro_confirm_mode = str(P("kiro_confirm_mode", "none").value).strip()
        if self.kiro_confirm_mode not in CONFIRM_MODES:
            raise PlatformConfigError(f"parameter kiro_confirm_mode must be one of {CONFIRM_MODES}, got {self.kiro_confirm_mode!r}")
        self.sensor_ip = P("sensor_ip", "").value
        self.sensor_port = int(P("sensor_port", 0).value)
        self.sensor_init_overrides = P("sensor_init_overrides", "").value       # 'Name=value,Name=value' (시험용 AutoTriggerFPS 등)
        self.local_speed = float(P("local_speed_mps", 0.0).value)              # 0 = 플랫폼 TCP 한계 (moveit_jtc.limits.max_tcp_speed_mps)
        self.transit_speed = float(P("transit_speed_mps", 0.0).value)
        self.descend_speed = float(P("descend_speed_mps", 0.0).value)          # 0 = 플랫폼 TCP 한계
        self.dwell_frames = int(P("dwell_frames", 0).value)
        self.probe_frames = int(P("probe_frames", 30).value)
        self.motion_timeout = float(P("motion_timeout_s", 0.0).value)
        # 1G core (G1_CORE_SPEC §8.6): 'core' = scan_l0.teach_1g.Teach1G (이 패키지의 유일한 경로)
        self.g1_pipeline = str(P("g1_pipeline", "core").value).strip()
        if self.g1_pipeline not in G1_PIPELINES:
            raise PlatformConfigError(f"parameter g1_pipeline must be one of {G1_PIPELINES}, got {self.g1_pipeline!r}")
        # 센서는 한 명만 잡는다: 목표 수행 전에 hongik_profiler_node 가 센서를 잡고 있으면 놓게 한다 (이 이름은 서버 네임스페이스 기준;
        # '' = 끔). 서비스가 없으면 (프로파일러 노드 미실행) 그대로 진행, 있는데 실패하면 목표 거부.
        self.profiler_release_service = P("profiler_release_service", "hongik_profiler_node/stop").value
        # site-level run-time job overrides (user 2026-10-13; scan_l0.job_overrides): a YAML / JSON mapping {"<dotted path>": value}
        # applied to every goal of both actions right after the job file is loaded (the AutoTeach goal's job_overrides on top) -
        # KIRO's site adjusts through this (its PrecisionSensing goal has no field). Read and checked at start ('' = none)
        self.job_overrides_text = str(P("job_overrides", "").value or "")
        hz = float(P("profile_publish_hz", 20.0).value)               # 시각화 ~/profile 상한 (0 = 내지 않음; 녹화는 모든 프레임)
        self._profile_period_ns = int(1e9 / hz) if hz > 0.0 else None
        self._profile_last_ns = 0
        self._kiro_type, why = kiro_action_type()
        if self._kiro_type is None:
            raise RuntimeError(why)
        if not self.platform_config:
            raise PlatformConfigError("parameter platform_config (platform YAML path) is required")
        if not self.calib_file:
            raise PlatformConfigError("parameter calib_file (EOAT calibration YAML) is required")
        self.site_overrides = self._site_overrides(self.job_overrides_text)
        self.settings = C.load_settings(self.platform_config)
        self.adapter = self.settings.adapter
        self.cb = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._running: Optional[AutoTeach] = None
        self._active: Optional[str] = None             # one goal at a time (both actions), claimed in execute
        self._goal_kind: Optional[str] = None
        self._t_goal = 0.0                             # monotonic time of the active goal's acceptance (PrecisionSensing elapsed_time)
        self._fb_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._confirm_q: list = []
        self._awaiting_confirm = False
        self._cancel = threading.Event()
        self._cancel_fut: Optional[Future] = None       # 지금 목표의 취소 Future (S4): 취소·종료가 끝낸다 → 실행의 모든 작업 프로세스 기다림이 바로 끝난다
        self._shutdown = threading.Event()              # the server is shutting down (request_shutdown): no new run, service waits end
        self._threads_cv = threading.Condition()
        self._n_goal_threads = 0                        # execute callbacks running (destroy waits for them before closing the platform)
        self._goal_handle = None
        self._fb = {"phase": "idle", "pass_index": 0, "text": "", "t_pub": 0.0}
        from .moveit_adapter import MoveItPlatform
        self.platform = MoveItPlatform(self.settings)
        self.platform.latch.end_run()                                  # 실행 사이: 정지 요청은 기록만 (목표마다 _claim 이 실행을 연다)
        # every service wait of the platform (MoveIt planning / checking) ends on a stop of the run or the server's shutdown; the
        # planning-scene clients are exempt: a stopped run still removes its objects (ros_util.call_service)
        self.platform.ctx.add_stop_source(self._service_stop)
        self.platform.ctx.exempt_from_stop(self.platform.scene_client, self.platform.get_scene_client)
        self.base_frame = self.platform.mj.base_frame
        self.optical_frame = self.platform.mj.optical_frame
        self.pub_profile = self.create_publisher(ProfileRaw, "~/profile", 50)
        # every progress / decision line, one per message (r2 §4-1: no 0.2 s overwrite) — the console client and KIRO read this
        log_qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.pub_log = self.create_publisher(String, "~/log", log_qos)
        self.srv_confirm = self.create_service(SetBool, "~/confirm_move", self._on_confirm, callback_group=self.cb)
        self.srv_status = self.create_service(Trigger, "~/status", self._on_status, callback_group=self.cb)
        self._release_cli = (self.create_client(Trigger, self.profiler_release_service, callback_group=self.cb)
                             if self.profiler_release_service else None)
        self.action = ActionServer(self, AutoTeachAction, "auto_teach", execute_callback=self._goal_thread(self._execute),
                                   goal_callback=self._on_goal, cancel_callback=self._on_cancel, callback_group=self.cb)
        # KIRO (r3 §4): always accept; a goal that cannot run is aborted at once with the reason in the result
        self.kiro_action = ActionServer(self, self._kiro_type, "precision_sensing", execute_callback=self._goal_thread(self._execute_kiro),
                                        goal_callback=lambda _goal: GoalResponse.ACCEPT, cancel_callback=self._on_cancel,
                                        callback_group=self.cb)
        # post-weld data acquisition (user 2026-10-13): always accept; a goal that cannot run ends REFUSED with the reason
        self.post_weld_action = ActionServer(self, PostWeldScanAction, "post_weld_scan",
                                             execute_callback=self._goal_thread(self._execute_post_weld),
                                             goal_callback=lambda _goal: GoalResponse.ACCEPT, cancel_callback=self._on_cancel,
                                             callback_group=self.cb)
        self.platform.latch.add_listener(self._on_stop_event)
        # start-up record (KIRO 2026-10-14: which site files and which numeric packages this server really runs with)
        self._say(f"site parameters: platform_config {self.platform_config}, job_file {self.job_file}, calib_file {self.calib_file}, "
                  f"out_root {self.out_root}, confirm_mode {self.confirm_mode}, kiro_confirm_mode {self.kiro_confirm_mode}")
        self._say(f"numeric packages: {numeric_record_text()} (python {sys.executable})")
        self._clear_leftover_scene("server start")
        self._say("site job overrides (server parameter job_overrides, every goal): "
                  + ("; ".join(f"{k_} = {J._short(v_)}" for k_, v_ in self.site_overrides.items()) if self.site_overrides else "none"))
        self._say(f"hongik_auto_teach_server ready: adapter {self.adapter}, config {self.platform_config}, actions "
                  f"{self.resolve_action_name()}, {self.resolve_action_name('precision_sensing')} and "
                  f"{self.resolve_action_name('post_weld_scan')}, log {self.pub_log.topic_name}")

    def _effective_job_path(self, position: str, out_dir: Path) -> Path:
        """Where the run saves its effective job: the 1G core in verification/, the legacy auto-teach next to its log."""
        core = select_teach_class(str(position), self.g1_pipeline).__name__ == "Teach1G"
        return (out_dir / "verification" / J.EFFECTIVE_JOB_NAME) if core else (out_dir / J.EFFECTIVE_JOB_NAME)

    def _site_overrides(self, text: str) -> dict:
        """The server parameter job_overrides → flat {path: value}, its paths and values checked at start on the server's job file
        (the goal's job file is checked again per goal). PlatformConfigError = the server does not start (a site typo is seen at once)."""
        try:
            ov = J.parse_mapping_text(text, what="server parameter job_overrides")
            if ov:
                job, doc = C.load_job(self.job_file)
                J.apply_overrides(doc, ov, source="server parameter", position=str(job.position or "") or None)
        except (J.OverrideError, OSError, ValueError) as e:
            raise PlatformConfigError(f"parameter job_overrides: {e}") from None
        return ov

    def resolve_action_name(self, name: str = "auto_teach") -> str:
        ns = self.get_namespace().rstrip("/")
        return f"{ns}/{name}"

    # ---- one goal at a time (both actions) ----
    def _claim(self, kind: str, goal_handle) -> Optional[str]:
        """Take the run slot for this goal → None, or the description of the goal that holds it / CANCEL_SERVER_SHUTDOWN (the caller
        refuses: claim_refusal)."""
        with self._lock:
            if self._shutdown.is_set():
                return CANCEL_SERVER_SHUTDOWN
            if self._active is not None:
                return f"{self._active} goal running, phase {self._fb['phase']}"
            self._active, self._goal_kind, self._goal_handle = kind, kind, goal_handle
            self._t_goal = time.monotonic()
            self.platform.latch.begin_run()                            # a new run scope: no stop of an earlier run carries over
            # this goal's cancel state from the claim on: a cancel that arrives before the run object exists is kept (the run sees
            # the completed Future before its first motion)
            self._cancel_fut = Future()
            self._cancel.clear()
            self._confirm_q.clear()
            self._awaiting_confirm = False
            return None

    def _release_slot(self) -> None:
        with self._lock:
            self._active, self._goal_kind, self._goal_handle, self._running = None, None, None, None
            self.platform.latch.end_run()                              # the run is over: its stop (if any) ends with it
        self._awaiting_confirm = False

    def _on_stop_event(self, event: str, rec) -> None:
        """Platform stop events on the node log + ~/log (the operator / KIRO sees every stop and what was sent)."""
        if rec is None:
            return
        if event == "stop":
            steps = ", ".join(f"{st}: {str(r).split(' (')[0]}" for st, r in rec.steps)
            self._say(f"STOP ({rec.source}): {rec.reason} - stop sequence: {steps}", "warn")
        elif event == "idle_request":
            self._say(f"stop request ({rec.source}) with no run in progress: {rec.reason} - nothing of ours is moving, nothing sent")

    def _not_ready(self) -> list:
        """MoveItPlatform.readiness() → what keeps the platform from starting a run (empty = ready)."""
        try:
            return list(self.platform.readiness())
        except Exception as e:  # noqa: BLE001 - readiness unknown = not ready
            return [f"readiness check failed: {type(e).__name__}: {e}"]

    # ---- ~/log ----
    def _publish_lines(self, text: str) -> None:
        """One std_msgs/String per line (multi-line text split), every line — nothing is dropped or merged."""
        with self._log_lock:
            for line in (str(text).splitlines() or [""]):
                try:
                    self.pub_log.publish(String(data=line))
                except Exception:  # noqa: BLE001 - shutting down
                    return

    def _say(self, line: str, level: str = "info") -> None:
        """A server line (goal received, refused, result): the node log + ~/log. One logger call site per severity (rclpy refuses a
        call site whose severity changes between calls)."""
        lg = self.get_logger()
        if level == "error":
            lg.error(str(line))
        elif level == "warn":
            lg.warn(str(line))
        else:
            lg.info(str(line))
        self._publish_lines(line)

    # ---- 서비스 ----
    def _on_confirm(self, req, res):
        if not self._awaiting_confirm:
            res.success, res.message = False, "no move is waiting for confirmation"
            return res
        self._confirm_q.append("y" if req.data else "n")
        res.success, res.message = True, "confirmed: execute" if req.data else "declined: abort"
        return res

    def _on_status(self, _req, res):
        at = self._running
        res.success = True
        not_ready = self._not_ready() if at is None else []
        lt = self.platform.latch
        res.message = (f"adapter={self.adapter} base_frame={self.base_frame} running={at is not None} goal={self._active or ''} "
                       f"phase={self._fb['phase']} "
                       f"pass={self._fb['pass_index']} ready={not not_ready} not_ready={'; '.join(not_ready)!r} "
                       f"run_stopped={lt.is_set} stop_reason={lt.reason if lt.is_set else ''!r} "
                       f"awaiting_confirm={self._awaiting_confirm} last={self._fb['text'][:200]}")
        return res

    def _confirm_source(self):
        if self._cancel.is_set():
            return "n"
        self._awaiting_confirm = True
        if self._confirm_q:
            self._awaiting_confirm = False
            return self._confirm_q.pop(0)
        return None

    # ---- 액션 ----
    def _on_goal(self, goal) -> GoalResponse:
        if self._shutdown.is_set():
            self._say(f"auto_teach goal rejected: {CANCEL_SERVER_SHUTDOWN}", "warn")
            return GoalResponse.REJECT
        with self._lock:
            active = self._active
        if active is not None:
            self._say(f"auto_teach goal rejected: another run is active ({active})", "warn")
            return GoalResponse.REJECT
        if str(goal.position).strip() != "1G":
            self._say(f"auto_teach goal rejected: position must be '1G' (this package runs the 1G core only; got {goal.position!r})", "warn")
            return GoalResponse.REJECT
        not_ready = self._not_ready()
        if not_ready:
            self._say(f"auto_teach goal rejected: platform not ready: {'; '.join(not_ready)}", "warn")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        """Cancel of the running goal (either action) = the immediate stop of this run (platform stop sequence; nothing more is sent in
        the run) + the run's cancel Future (S4). No latch: the next goal starts when the platform is ready. A cancel of a goal that is not
        running (a refused PrecisionSensing goal on its way to abort) touches nothing."""
        with self._lock:
            kind = self._goal_kind if (goal_handle is not None and goal_handle is self._goal_handle) else None
        if kind is None:
            self._say("cancel request for a goal that is not running: nothing to stop")
            return CancelResponse.ACCEPT
        self._cancel.set()
        why = f"{kind} action cancel requested"
        self._say(f"{kind} cancel requested: stopping the run", "warn")
        self.platform.latch.fire(why, "action_cancel")
        self._complete_cancel(why)                                   # S4: 실행의 취소 Future (1G core 는 이것으로 request_cancel)
        return CancelResponse.ACCEPT

    def _complete_cancel(self, why: str) -> None:
        """지금 목표의 취소 Future 를 끝낸다 (S4: 정지 명령은 팔을 멈추고, 이것은 실행의 기다림 — 패스 뒤 처리·뒤 저장·용접 경로·단면 — 을 끝낸다)."""
        fut = self._cancel_fut
        if fut is not None and not fut.done():
            try:
                fut.set_result(why)
            except InvalidStateError:                      # 다른 취소 경로가 먼저 끝냈다
                pass

    def _feedback(self, force: bool = True) -> None:
        """Feedback of the active goal — AutoTeach: phase / pass / this line / residuals; PrecisionSensing: elapsed_time. Sent for every
        line (r2 §4-1: the 0.2 s overwrite lost lines); `force` is kept for the callers' wording."""
        with self._lock:
            gh, kind, t0 = self._goal_handle, self._goal_kind, self._t_goal
        if gh is None:
            return
        if kind == "precision_sensing":
            self._kiro_feedback(gh, t0)
            return
        if kind == "post_weld_scan":
            pf = PostWeldScanAction.Feedback()
            pf.phase = str(self._fb["phase"])
            pf.progress_text = str(self._fb["text"])[:2000]
            pf.awaiting_confirmation = bool(self._awaiting_confirm)
            try:
                with self._fb_lock:
                    gh.publish_feedback(pf)
            except Exception:  # noqa: BLE001 - the goal finished meanwhile
                pass
            return
        fb = AutoTeachAction.Feedback()
        fb.phase = str(self._fb["phase"])
        fb.pass_index = int(self._fb["pass_index"])
        fb.progress_text = str(self._fb["text"])[:2000]
        fb.awaiting_confirmation = bool(self._awaiting_confirm)
        at = self._running
        try:
            if at is not None and at.res.residual_history:
                last = at.res.residual_history[-1]
                vals = last.get("residuals", last) if isinstance(last, dict) else {}
                names, nums = [], []
                for k, v in (vals.items() if isinstance(vals, dict) else []):
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                        names.append(str(k))
                        nums.append(float(v))
                fb.residual_names, fb.residuals = names, nums
        except Exception:  # noqa: BLE001
            pass
        try:
            with self._fb_lock:
                gh.publish_feedback(fb)
        except Exception:  # noqa: BLE001
            pass

    def _kiro_feedback(self, gh, t0: float) -> None:
        fb = self._kiro_type.Feedback()
        fb.elapsed_time = float(time.monotonic() - t0)
        try:
            with self._fb_lock:
                gh.publish_feedback(fb)
        except Exception:  # noqa: BLE001 - the goal finished meanwhile
            pass

    def _log(self, msg: str) -> None:
        text = str(msg)
        self.get_logger().info(text)
        self._publish_lines(text)
        line = text.splitlines()[0] if text else ""
        if line.startswith("[") and "]" in line:
            self._fb["phase"] = line[1:line.index("]")]
        if line.startswith("CONFIRM"):
            self._fb["phase"] = "confirm"
            self._awaiting_confirm = True
            srv = str(getattr(getattr(self, "srv_confirm", None), "srv_name", "") or "~/confirm_move")
            # 전체 서비스 이름과 그대로 붙여 쓸 명령 (2026-10-06: '~/confirm_move' 만 보고 /hongik_fine_sensing/confirm_move 를 불러 멈춰 있었다)
            hint = (f" -> in another terminal: ros2 service call {srv} std_srvs/srv/SetBool \"{{data: true}}\" (false = abort)")
            self._fb["text"] = text + hint
            self._publish_lines(hint.strip())
            self._feedback(force=True)
            return
        self._fb["text"] = text
        self._feedback()

    def _on_event(self, evt: str, data: dict) -> None:
        if evt in ("pass_cloud", "outputs"):                            # visualization events: not published by this package
            return
        if evt == "phase":                                              # 1G core: setup|acquire|coarse|estimate|fine|outputs
            self._fb["phase"] = str(data.get("phase", ""))
            self._feedback(force=True)
            return
        if evt == "pass_started":
            self._fb["pass_index"] = int(data.get("k", 0))
            self._fb["phase"] = "pass"
        elif evt in ("move", "transit_started", "descend_started", "probe_done") and not hasattr(self._running, "request_cancel"):
            self._fb["phase"] = evt if evt != "move" else f"move:{data.get('phase', '')}"
        if evt != "confirm":
            self._awaiting_confirm = False
        at = self._running
        ss = getattr(getattr(at, "sess", None), "sensor", None) if at is not None else None
        if ss is not None and not getattr(ss, "_hongik_tap", False):
            ss.add_tap(self._tap_profile)
            ss._hongik_tap = True
        self._feedback(force=True)

    def _tap_profile(self, rec) -> None:
        """센서 녹화 스레드에서 프레임마다 불린다 — 시각화용 ~/profile 을 profile_publish_hz 이하로만 낸다 (0 = 내지 않음).
        실기 at10 (2026-10-03): 100 fps 전부를 list 변환으로 내던 이 탭이 녹화 스레드의 주된 부하였고, 계획 계산과 겹치자 녹화가
        밀려 G3 거짓 정지. 녹화·가드는 모든 프레임을 그대로 받는다 (이 토픽은 조립기의 점구름 시각화에만 쓰인다)."""
        if self._profile_period_ns is None:
            return
        now = time.monotonic_ns()
        if now - self._profile_last_ns < self._profile_period_ns:
            return
        self._profile_last_ns = now
        try:
            if self.pub_profile.get_subscription_count() == 0:             # 보는 노드 (조립기·RViz) 가 없으면 만들지 않는다
                return
            age_ns = now - int(rec.t_rx_ns)
            self.pub_profile.publish(profile_msg(rec, self.optical_frame,
                                                 (self.get_clock().now() - Duration(nanoseconds=max(0, age_ns))).to_msg()))
        except Exception:  # noqa: BLE001 - visualization only
            pass

    def _goal_points_base(self, goal) -> tuple:
        S = np.array([goal.rough_start.x, goal.rough_start.y, goal.rough_start.z], dtype=float)
        E = np.array([goal.rough_end.x, goal.rough_end.y, goal.rough_end.z], dtype=float)
        fid = str(goal.frame_id or "").strip()
        if fid and fid != self.base_frame:
            T = self.platform.pose.static_transform(self.base_frame, fid, 2.0)
            S = (T @ np.append(S, 1.0))[:3]
            E = (T @ np.append(E, 1.0))[:3]
        return S, E

    def _sensor_init(self) -> Optional[dict]:
        """The sensor's initial parameters: the driver standard set, the platform config's sensor.init_params on top (the same set the
        profiler node sends — 2026-10-11: until then the server sent the driver standard only, so a config change such as
        AutoTriggerFPS did not reach the server's runs), then sensor_init_overrides. None = neither config nor overrides."""
        txt = str(self.sensor_init_overrides or "").strip()
        cfg = dict((getattr(self.settings, "sensor", None) or {}).get("init_params") or {})
        if not txt and not cfg:
            return None
        from vc3dz_driver import params as VP
        d = dict(VP.INIT_STANDARD)
        d.update(cfg)
        for it in txt.split(","):
            if "=" in it:
                k, v = it.split("=", 1)
                try:
                    d[k.strip()] = float(v) if "." in v else int(v)
                except ValueError:
                    d[k.strip()] = v.strip()
        return d

    def _new_out_dir(self, tag: str, name: str = "") -> Path:
        stamp_dir = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(self.out_root) / f"{stamp_dir}_{tag}"
        name = safe_out_name(name)
        if name:
            out_dir = Path(self.out_root) / name
            if out_dir.exists() and any(out_dir.iterdir()):
                out_dir = Path(self.out_root) / f"{name}_{stamp_dir}"
        return out_dir

    def _run_teach(self, *, position: str, S, E, goal_like, job_file: str, out_dir: Path, confirm: str, dry_run: bool, max_passes,
                   ignore_start_pose: bool, measure_speed_mps: Optional[float] = None, goal_overrides: Optional[dict] = None):
        """The one run path of both actions: rough job (build_rough_job, with the run-time job overrides: the server parameter's,
        then the goal's) → Teach1G (select_teach_class) → run. Exceptions propagate (the caller reports them; J.OverrideError =
        REFUSED, nothing planned). The cancel state was reset when the goal claimed the slot (_claim)."""
        self._fb.update(phase="setup", pass_index=0, text="")
        self._run_overrides = []
        self._clear_leftover_scene("before the run")
        calib = self.platform.calib_for_platform(C.load_calib(self.calib_file))
        grav = self.platform.gravity_axis_base()
        session = ScanSession(robot=self.platform, sensor=None, owned=False)
        job0, doc = build_rough_job(self.settings, job_file, goal_like, base_frame=self.base_frame, gravity_axis_base=grav, start=S, end=E,
                                    overrides=[("server parameter", self.site_overrides), ("goal", dict(goal_overrides or {}))])
        self._run_overrides = J.records_of(doc)
        self._say(f"{J.summary_line(self._run_overrides, J.job_file_of(doc))} - the run saves the effective job as "
                  f"{self._effective_job_path(position, out_dir)}")
        if ignore_start_pose and doc.get("start_pose"):
            # KIRO: the start pose is the arm pose at goal acceptance (r3 §4.1) — the job's start_pose is not used
            doc = {k: v for k, v in doc.items() if k != "start_pose"}
            meta = dict(doc.get("meta", {}) or {})
            note = "job start_pose ignored: the run starts from the arm pose at goal acceptance"
            meta["goal_notes"] = list(meta.get("goal_notes", []) or []) + [note]
            doc["meta"] = meta
        kw = dict(out_dir=str(out_dir), confirm=confirm, dry_run=bool(dry_run), max_passes=max_passes,
                  measure_speed_mps=measure_speed_mps,
                  confirm_source=self._confirm_source, log=self._log, on_event=self._on_event,
                  local_speed_mps=self.local_speed if self.local_speed > 0 else None,
                  descend_speed_mps=self.descend_speed if self.descend_speed > 0 else None,
                  transit_speed_mps=self.transit_speed if self.transit_speed > 0 else None,
                  dwell_frames=self.dwell_frames if self.dwell_frames > 0 else None, probe_frames=self.probe_frames,
                  motion_timeout_s=self.motion_timeout if self.motion_timeout > 0 else None,
                  sensor_ip=self.sensor_ip or None, sensor_port=self.sensor_port or None, sensor_init=self._sensor_init(),
                  session=session, release_sensor=True, pass_cloud_wanted=None,
                  cancel=self._cancel_fut)
        cls = select_teach_class(position, self.g1_pipeline)    # §8.6: 1G + 'core' → Teach1G (그 밖은 ValueError)
        self._say(f"run: position {position} -> {cls.__name__} (g1_pipeline {self.g1_pipeline}), job {job_file}, confirm {confirm}, "
                  f"out_dir {out_dir}, measurement speed "
                  + (f"{measure_speed_mps * 1e3:g} mm/s (run override)" if measure_speed_mps is not None else "the job's speed_mps"))
        at = cls(self.settings, job0, doc, calib, **kw)
        with self._lock:
            self._running = at
        try:
            return at.run()
        finally:
            with self._lock:
                self._running = None
            self._awaiting_confirm = False

    def _execute(self, goal_handle):
        goal = goal_handle.request
        result = AutoTeachAction.Result()
        busy = self._claim("auto_teach", goal_handle)
        if busy is not None:                                       # the goal callback saw a free slot, another goal took it since
            result.status, result.reason = "REFUSED", claim_refusal(busy)
            self._say(f"auto_teach goal: {result.status}: {result.reason}", "warn")
            goal_handle.abort()
            return result
        try:
            return self._execute_auto_teach(goal_handle, goal, result)
        finally:
            self._release_slot()

    def _execute_auto_teach(self, goal_handle, goal, result):
        out_dir = self._new_out_dir(str(goal.position).strip() or "pos", getattr(goal, "out_name", ""))
        result.out_dir = str(out_dir)
        if goal_handle.is_cancel_requested:                         # cancelled between acceptance and the run slot
            result.status, result.reason = "ABORTED", "cancel requested before the run started (nothing sent)"
            self._say(f"auto_teach result: {result.status}: {result.reason}")
            goal_handle.canceled()
            return result
        try:
            measure_speed = goal_measure_speed(goal)
            goal_ov = J.goal_overrides(goal)
            if measure_speed is not None and "job.speed_mps" in goal_ov:
                raise J.OverrideError(f"the measurement speed is given twice in the goal: measure_speed_mps {measure_speed:g} m/s and "
                                      f"job_overrides job.speed_mps {goal_ov['job.speed_mps']!r} - give it once")
        except ValueError as e:
            result.status, result.reason = "REFUSED", f"{e} (nothing sent)"
            self._say(f"auto_teach result: {result.status}: {result.reason}", "warn")
            goal_handle.abort()
            return result
        ok, why = self._release_profiler()
        if not ok:
            self._say(why, "error")
            result.status, result.reason = "REFUSED", why
            goal_handle.abort()
            return result
        try:
            S, E = self._goal_points_base(goal)
            r = self._run_teach(position=str(goal.position).strip(), S=S, E=E, goal_like=goal, job_file=goal.job_file or self.job_file,
                                out_dir=out_dir, confirm=str(goal.confirm_mode or self.confirm_mode),
                                dry_run=bool(goal.dry_run),
                                max_passes=int(goal.max_passes) if goal.max_passes else None, ignore_start_pose=False,
                                measure_speed_mps=measure_speed, goal_overrides=goal_ov)
        except J.OverrideError as e:                                # an override that cannot be applied: nothing was planned or sent
            result.status, result.reason = "REFUSED", f"{e} (nothing sent)"
            self._say(f"auto_teach result: {result.status}: {result.reason}", "warn")
            goal_handle.abort()
            return result
        except Exception as e:  # noqa: BLE001 - after a cancel / the shutdown: ABORTED (_failure)
            result.status, result.reason, _cat = self._failure(goal_handle, e)
            self._say(f"auto_teach failed before/while running: {result.status}: {result.reason}", "error")
            goal_handle.abort()
            return result
        result.status, result.reason = str(r.status), str(r.reason)
        result.length_mm = float(r.extent.get("length_mm", float("nan"))) if r.extent else float("nan")
        if r.job_final is not None:
            # 방문 순서 (1G 로봇에 가까운 점부터, 3G 위 점부터) 가 내부 S/E 를 바꿔도 납품 작업은 교시 S→E (AutoTeach._final_job) — 그대로
            result.seam_start = point_msg(r.job_final.start)
            result.seam_end = point_msg(r.job_final.end)
        try:
            result.ends_observed = bool(r.extent and r.extent["S"].get("observed") and r.extent["E"].get("observed"))
        except Exception:  # noqa: BLE001
            result.ends_observed = False
        result.converged = str(r.converged or "")
        result.refined_job_path = str(r.job_path or "")
        result.n_passes = len(r.passes)
        result.n_moves = int(r.n_move_l)
        result.elapsed_s = float(r.elapsed_s)
        if self._cancel.is_set() and goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif r.status in (*K.SUCCESS_STATUSES, "DRY_RUN"):            # PASS_UNRESOLVED = PASS with unresolved quantities (r2 §4-6)
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self._say(f"auto_teach result: {result.status}: {result.reason}")
        return result

    # ---- KIRO PrecisionSensing (r3 §4) ----
    def _execute_kiro(self, goal_handle):
        """Accepted always. Elapsed-time feedback at 1 Hz (plus one per log line) from acceptance to the result. The full result message
        goes into the run folder as precision_sensing_result.yaml (every goal whose run made a folder - KIRO's main system's goals and
        the operator client's alike) before the result is returned."""
        t0 = time.monotonic()
        stop = threading.Event()
        ticker = threading.Thread(target=self._kiro_ticker, args=(goal_handle, t0, stop), name="precision_sensing_feedback", daemon=True)
        ticker.start()
        run = {}                                                     # 'dir': this goal's run folder once chosen (_kiro_goal)
        try:
            try:
                oc, paths, data_file = self._kiro_goal(goal_handle, t0, run)
            except Exception as e:  # noqa: BLE001 - a server-side failure outside the run (after a cancel / the shutdown: ABORTED)
                st_, why_, cat_ = self._failure(goal_handle, e)
                oc, paths, data_file = K.run_outcome(st_, why_, category=cat_), [], ""
        finally:
            stop.set()
            ticker.join(timeout=5.0)
        result = self._kiro_type.Result()
        result.error_code = int(oc.error_code)
        result.error_message = str(oc.error_message)
        result.paths = K.welding_paths_msgs(paths) if oc.deliver_paths else []
        result.data_file_path = str(data_file or "")
        self._kiro_result_file(result, run.get("dir"))
        self._say(f"precision_sensing result: {K.CODE_NAMES.get(oc.error_code, '?')} ({oc.error_code}), {len(result.paths)} path(s), "
                  f"data_file_path '{result.data_file_path}' - {oc.error_message}", "info" if oc.action == "succeed" else "warn")
        if oc.action == "canceled" and goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif oc.action == "succeed":
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _kiro_result_file(self, result, run_dir: Optional[Path]) -> None:
        """<run folder>/precision_sensing_result.yaml = the result message (kiro_io.write_result_yaml) and the line 'precision_sensing:
        result file <path>' (the operator client finds the run folder by it). A goal that ended before its run made a folder (refused,
        cancelled before the run) has none: nothing is written. A failed write is reported; the result itself is unchanged."""
        if run_dir is None or not Path(run_dir).is_dir():
            return
        try:
            self._say(f"precision_sensing: result file {K.write_result_yaml(run_dir, result)}")
        except Exception as e:  # noqa: BLE001 - the result goes out regardless
            self._say(f"precision_sensing: result file not written in {run_dir}: {type(e).__name__}: {e}", "warn")

    def _kiro_ticker(self, goal_handle, t0: float, stop: threading.Event) -> None:
        while not stop.wait(1.0):
            self._kiro_feedback(goal_handle, t0)

    def _kiro_goal(self, goal_handle, t0: float, run: dict) -> tuple:
        """→ (kiro_io.Outcome, [PathArrays], data_file_path); run['dir'] = the run folder once chosen (also when the run raises)."""
        goal = goal_handle.request
        n = int(goal.welding_position)
        self._say(f"precision_sensing goal: welding_position {n} ({K.position_label(n)}), pose_a "
                  f"[{goal.pose_a.position.x:.4f}, {goal.pose_a.position.y:.4f}, {goal.pose_a.position.z:.4f}], pose_b "
                  f"[{goal.pose_b.position.x:.4f}, {goal.pose_b.position.y:.4f}, {goal.pose_b.position.z:.4f}] ({self.base_frame})")
        try:
            inp = K.goal_to_run_inputs(goal, g1_pipeline=self.g1_pipeline)
        except K.InvalidGoal as e:
            return K.refused_outcome(f"{e} (nothing sent)", "input"), [], ""
        busy = self._claim("precision_sensing", goal_handle)
        if busy is not None:
            return K.refused_outcome(claim_refusal(busy), "internal"), [], ""
        try:
            with self._lock:
                self._t_goal = t0                                  # elapsed_time counts from acceptance, not from the claim
            if goal_handle.is_cancel_requested:
                return K.run_outcome("ABORTED", "cancel requested before the run started (nothing sent)", cancelled=True), [], ""
            not_ready = self._not_ready()
            if not_ready:
                return K.refused_outcome(f"platform not ready: {'; '.join(not_ready)} (nothing sent)", "motion"), [], ""
            ok, why = self._release_profiler()
            if not ok:
                return K.refused_outcome(why, "sensor"), [], ""
            out_dir = self._new_out_dir(f"kiro_{inp.position}")
            run["dir"] = out_dir
            start = self._start_pose_record()
            for note in inp.notes:
                self._say(f"precision_sensing: {note}")
            try:
                r = self._run_teach(position=inp.position, S=np.asarray(inp.start_m), E=np.asarray(inp.end_m),
                                    goal_like=K.rough_goal(inp), job_file=self.job_file, out_dir=out_dir, confirm=self.kiro_confirm_mode,
                                    dry_run=False, max_passes=None, ignore_start_pose=True,
                                    measure_speed_mps=None)   # PrecisionSensing has no speed: the job file's speed_mps (+ job_overrides)
            except J.OverrideError as e:                            # the site's job_overrides on this goal's job: nothing sent
                return K.refused_outcome(f"{e} (nothing sent)", "input"), [], ""
            return self._kiro_deliver(goal_handle, inp, r, out_dir, start)
        finally:
            self._release_slot()

    def _kiro_deliver(self, goal_handle, inp, r, out_dir: Path, start: dict) -> tuple:
        """Run result → outcome + WeldingPath arrays + precision_sensing.npz (when the run left sections or paths in out_dir)."""
        cancelled = self._cancel.is_set() and goal_handle.is_cancel_requested
        oc = K.run_outcome(r.status, r.reason, category=r.category, cancelled=cancelled)   # category: _Abort(category=...) / D14
        paths = []
        if oc.deliver_paths:
            try:
                if r.weld_plan is None:
                    raise ValueError("the run ended with a success status but without a weld plan")
                paths = K.plan_paths(r.weld_plan)
            except ValueError as e:
                oc = K.run_outcome("DATA_FAIL", f"welding paths not built from the weld plan: {e} (scan: {r.status}: {r.reason})",
                                   category="waypoint")
                paths = []
        sections, src = K.sections_arrays(out_dir)
        if not (paths or src):
            return oc, paths, ""
        try:
            unresolved = list((r.converged_detail or {}).get("unresolved") or [])
            inputs = {"pose_a_m": list(inp.start_m), "pose_b_m": list(inp.end_m), "job_file": str(self.job_file),
                      "confirm_mode": self.kiro_confirm_mode, "start_pose_at_acceptance": start,
                      "pipeline": select_teach_class(inp.position, self.g1_pipeline).__name__,
                      # run-time job overrides (server parameter job_overrides) and the effective job the run used
                      "job_overrides": list(getattr(self, "_run_overrides", []) or []),
                      "job_effective": str(self._effective_job_path(inp.position, out_dir))}
            ends = K.ends_from_goal(getattr(r, "s_taught_point", "start"))
            wd = K.weld_direction_of(r.weld_plan)
            self._say(f"precision_sensing: output ends S = {ends['S']}, E = {ends['E']} (S = the end the run visited first); weld direction {wd['setting'] if wd else '-'} (paths in that order)")
            kiro = getattr(r, "kiro_outputs", None) if paths else None          # KIRO-format files only with delivered paths
            meta = K.npz_meta(outcome=oc, status=r.status, reason=r.reason, welding_position=inp.welding_position, position=inp.position,
                              base_frame=self.base_frame, k=float(self.settings.judgement_k), out_dir=str(out_dir), run_id=out_dir.name,
                              inputs=inputs, sections_source=src, unresolved=unresolved, n_paths=len(paths), notes=list(inp.notes),
                              ends_from_goal=ends, weld_direction=wd, kiro_outputs=K.kiro_meta(kiro))
            arrays = K.precision_npz_arrays(meta, plan=r.weld_plan, paths=paths, sections=sections, kiro=K.kiro_npz_arrays(kiro))
            km = meta.get("kiro_outputs")
            if km:
                where = f"in {km.get('dir')}" if km.get("written") else f"not written: {km.get('reason')}"
                self._say(f"precision_sensing: KIRO-format files {where} - not in the action result (no slot); npz meta "
                          "'kiro_outputs' and kiro_* arrays", "info")
            return oc, paths, K.write_precision_npz(out_dir, arrays)
        except Exception as e:  # noqa: BLE001 - the result file is part of the result: no file, no success
            return K.run_outcome("ERROR", f"{K.NPZ_NAME} not written: {type(e).__name__}: {e} (scan: {r.status}: {r.reason})",
                                 category="internal"), [], ""

    # ---- post-weld data acquisition (user 2026-10-13) ----
    def _execute_post_weld(self, goal_handle):
        """Accepted always: one goal at a time over all actions (busy = REFUSED with the goal that holds the slot)."""
        result = PostWeldScanAction.Result()
        busy = self._claim("post_weld_scan", goal_handle)
        if busy is not None:
            result.status, result.message = "REFUSED", claim_refusal(busy)
            self._say(f"post_weld_scan result: {result.status}: {result.message}", "warn")
            goal_handle.abort()
            return result
        try:
            return self._execute_post_weld_claimed(goal_handle, result)
        finally:
            self._release_slot()

    def _execute_post_weld_claimed(self, goal_handle, result):
        goal = goal_handle.request

        def finish(status: str, message: str, action: str, level: str = "info"):
            result.status, result.message = str(status), str(message)
            self._say(f"post_weld_scan result: {result.status}: {result.message}", level)
            getattr(goal_handle, action)()
            return result
        self._say(f"post_weld_scan goal: run_dir '{goal.run_dir or '(latest)'}', measurement speed "
                  + (f"{float(goal.measure_speed_mps) * 1e3:g} mm/s (override)" if float(goal.measure_speed_mps or 0.0) > 0.0
                     else "the run's measurement speed") + f", confirm '{goal.confirm_mode or self.confirm_mode}', "
                  + (f"after pass {int(goal.after_pass)}" if int(goal.after_pass or 0) > 0 else "final (after every pass)")
                  + f", direction {str(getattr(goal, 'direction', '') or '') or 'the nearer end'}"
                  + f", tool orientation {str(getattr(goal, 'tool_orientation', '') or '') or 'pass'}"
                  + f", {'dry run' if goal.dry_run else 'scan'}")
        if goal_handle.is_cancel_requested:
            return finish("ABORTED", "cancel requested before the scan started (nothing sent)", "canceled")
        try:
            inp = post_weld_inputs(goal, out_root=self.out_root, confirm_default=self.confirm_mode)
            from scan_l0.post_weld import load_input_run
            run = load_input_run(inp["run_dir"])
            run.check_after_pass(inp["after_pass"])                 # a pass the run's weld plan does not have: refused here
        except ValueError as e:
            return finish("REFUSED", f"{e} (nothing sent)", "abort", "warn")
        except Exception as e:  # noqa: BLE001 - an unexpected failure reading the input run: nothing was sent
            return finish("ERROR", f"input run not read: {type(e).__name__}: {e} (nothing sent)", "abort", "error")
        result.run_dir = str(run.run_dir)
        not_ready = self._not_ready()
        if not_ready:
            return finish("REFUSED", f"platform not ready: {'; '.join(not_ready)} (nothing sent)", "abort", "warn")
        if not inp["dry_run"]:
            ok, why = self._release_profiler()
            if not ok:
                return finish("REFUSED", why, "abort", "warn")
        try:
            r = self._run_post_weld(run, inp)
        except Exception as e:  # noqa: BLE001 - a server-side failure outside the scan (after a cancel / the shutdown: ABORTED)
            st_, why_, _cat = self._failure(goal_handle, e)
            return finish(st_, why_, "abort", "error")
        result.out_dir = str(r.out_dir or "")
        if self._cancel.is_set() and goal_handle.is_cancel_requested:
            return finish(r.status, r.summary, "canceled", "warn")
        return finish(r.status, r.summary, "succeed" if r.status in POST_WELD_OK else "abort",
                      "info" if r.status in POST_WELD_OK else "warn")

    def _run_post_weld(self, run, inp: dict):
        """The post-weld scan on the server's platform and sensor (scan_l0.post_weld.PostWeldScan) → PostWeldResult."""
        from scan_l0.post_weld import PostWeldScan, out_dir_for
        self._fb.update(phase="setup", pass_index=0, text="")
        self._clear_leftover_scene("before the scan")
        calib = self.platform.calib_for_platform(C.load_calib(self.calib_file))
        session = ScanSession(robot=self.platform, sensor=None, owned=False)
        out_dir = out_dir_for(run.run_dir, inp["after_pass"])
        pw = PostWeldScan(self.settings, run, calib, out_dir=str(out_dir), after_pass=inp["after_pass"],
                          direction=inp.get("direction", ""), tool_orientation=inp.get("tool_orientation", "pass"), confirm=inp["confirm"],
                          dry_run=bool(inp["dry_run"]), measure_speed_mps=inp["measure_speed_mps"], confirm_source=self._confirm_source,
                          log=self._log, on_event=self._on_event,
                          local_speed_mps=self.local_speed if self.local_speed > 0 else None,
                          descend_speed_mps=self.descend_speed if self.descend_speed > 0 else None,
                          transit_speed_mps=self.transit_speed if self.transit_speed > 0 else None,
                          motion_timeout_s=self.motion_timeout if self.motion_timeout > 0 else None,
                          sensor_ip=self.sensor_ip or None, sensor_port=self.sensor_port or None, sensor_init=self._sensor_init(),
                          session=session, release_sensor=True, cancel=self._cancel_fut)
        self._say(f"post-weld scan: input run {run.run_dir} ({run.status}), out_dir {out_dir if not inp['dry_run'] else '(dry run: none)'}")
        with self._lock:
            self._running = pw
        try:
            return pw.run()
        finally:
            with self._lock:
                self._running = None
            self._awaiting_confirm = False

    def _start_pose_record(self) -> dict:
        """The arm pose at goal acceptance (the run starts from it): joints + flange pose in the base frame, for the npz meta."""
        try:
            snap = self.platform.pose.snapshot()
            rec = {k: snap.record[k] for k in ("joint_names", "joint_positions_rad") if k in snap.record}
            rec["T_base_flange"] = np.asarray(snap.T, dtype=float).tolist()
            return rec
        except Exception as e:  # noqa: BLE001 - informational record only
            return {"unavailable": f"{type(e).__name__}: {e}"}

    def _release_profiler(self) -> tuple:
        """목표 전에 hongik_profiler_node 의 센서를 넘겨받는다 → (진행 가능?, 사유). 서비스 없음 = 노드 미실행 = 진행."""
        if self._release_cli is None:
            return True, "profiler release disabled"
        try:
            resp = call_service(self._release_cli, Trigger.Request(), timeout_s=5.0, wait_service_s=0.5)
        except MotionError as e:                                      # offered but silent: it may still hold the sensor
            return False, f"sensor busy: {e} - goal refused"
        if resp is None:
            return True, "no profiler node"
        if not resp.success:
            return False, f"sensor busy: {self._release_cli.srv_name} did not release the sensor ({resp.message}) - goal refused"
        self.get_logger().info(f"profiler node released the sensor: {resp.message}")
        return True, resp.message

    # ---- shutdown (KIRO 2026-10-14: the platform node was destroyed under a running goal - InvalidHandle, ERROR_INTERNAL and
    #      our boxes left in KIRO's MoveIt scene) ----
    def _goal_thread(self, execute):
        """An execute callback that counts as a running goal thread until it returns (destroy waits for them)."""
        def run(goal_handle):
            with self._threads_cv:
                self._n_goal_threads += 1
            try:
                return execute(goal_handle)
            finally:
                with self._threads_cv:
                    self._n_goal_threads -= 1
                    self._threads_cv.notify_all()
        run.__name__ = getattr(execute, "__name__", "execute")
        return run

    def _service_stop(self) -> Optional[str]:
        """The stop source of the platform's service waits (RosContext.add_stop_source): the server shutting down, or this run stopped
        (the latch: an action cancel / the shutdown's stop) → the reason; else None."""
        if self._shutdown.is_set():
            return CANCEL_SERVER_SHUTDOWN
        lt = self.platform.latch
        return f"run stopped: {lt.reason}" if lt.is_set else None

    def _failure(self, goal_handle, e: BaseException) -> tuple:
        """A server-side exception around a run → (status, reason, category): after the server's shutdown or a cancel of this goal it is
        their consequence → ABORTED '<cause>: <exception>' (no category, as a cancel); else ERROR (category internal)."""
        text = f"{type(e).__name__}: {e}"
        if self._shutdown.is_set():
            return "ABORTED", f"{CANCEL_SERVER_SHUTDOWN}: {text}", None
        if self._cancel.is_set() or bool(getattr(goal_handle, "is_cancel_requested", False)):
            return "ABORTED", f"{CANCEL_OPERATOR}: {text}", None
        return "ERROR", text, "internal"

    def _clear_leftover_scene(self, when: str) -> list:
        """Remove every object of ours (OUR_SCENE_PREFIX 'hongik/': the cell world hongik/specimen|floor|under and D14's weld scene
        hongik/weld/*) from move_group's planning scene → the ids removed. Called when no run of ours is active (server start,
        before every run, shutdown), so any such object was left by a run that did not end cleanly; objects of others are never
        touched. A scene that cannot be read is said in one line (the run then meets move_group itself)."""
        try:
            ids = scene_object_ids(self.platform)
        except MotionError as e:
            self._say(f"MoveIt scene ({when}): object names not read ({e}) - objects left by an earlier run, if any, stay", "warn")
            return []
        if ids is None:
            self._say(f"MoveIt scene ({when}): {self.platform.mj.get_scene_service} not offered - objects left by an earlier run, if "
                      "any, are removed once move_group answers")
            return []
        ours = sorted(i for i in ids if i.startswith(OUR_SCENE_PREFIX))
        if not ours:
            return []
        try:
            ok = bool(self.platform.remove_scene_boxes(ours))
        except MotionError as e:
            ok = False
            self._say(f"MoveIt scene ({when}): removal not confirmed: {e}", "warn")
        self._say(f"MoveIt scene ({when}): removed {len(ours)} object(s) of ours left by an earlier run: {', '.join(ours)}"
                  + ("" if ok else " - move_group did not confirm the removal"), "warn")
        return ours

    def request_shutdown(self) -> None:
        """Shutdown, step 1 (a signal, destroy): no new run; the running goal is stopped as a cancel stops it - the platform stop
        sequence (latch: our motion decelerates to rest; every service wait of the platform ends) and the run's cancel Future with
        CANCEL_SERVER_SHUTDOWN (its worker waits end; it reports ABORTED 'server shutting down'); a move waiting for confirmation is
        declined. Idempotent."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self._cancel.set()
        with self._lock:
            active = self._active
        if active is not None:
            self._say(f"{CANCEL_SERVER_SHUTDOWN}: stopping the running {active} goal", "warn")
            try:
                self.platform.latch.fire(CANCEL_SERVER_SHUTDOWN, "server_shutdown")
            except Exception as e:  # noqa: BLE001 - the cancel Future below still ends the run's waits
                self._say(f"{CANCEL_SERVER_SHUTDOWN}: stop sequence failed: {type(e).__name__}: {e}", "error")
        else:
            self._say(f"{CANCEL_SERVER_SHUTDOWN}: no run active")
        self._complete_cancel(CANCEL_SERVER_SHUTDOWN)                 # 실행 중인 목표의 기다림을 끝낸다 (S4)

    def goal_threads(self) -> int:
        with self._threads_cv:
            return self._n_goal_threads

    def destroy(self, *, spin=None, give_up=None) -> None:
        """Shutdown: request_shutdown, then wait until every execute callback has returned - the run ends after the stop and removes
        its MoveIt objects while the platform node still exists - then remove any object of ours still in the scene and close the
        platform. spin() = one turn of the server's executor while waiting (cancel requests and the goals' results still go out);
        give_up() → True = stop waiting (the operator's second signal; launch's SIGTERM). The wait has no bound of its own: after the
        stop the run's waits end within one SPIN_PERIOD_S (service waits) or by their existing timeouts (motion result, worker)."""
        self.request_shutdown()
        while True:
            with self._threads_cv:
                n = self._n_goal_threads
                if n == 0:
                    break
                if spin is None:
                    self._threads_cv.wait(SPIN_PERIOD_S)
            if give_up is not None and give_up():
                self._say(f"{CANCEL_SERVER_SHUTDOWN}: not waiting for {n} running goal(s) any more (second signal) - closing the "
                          "platform", "warn")
                break
            if spin is not None:
                spin()
        if self.platform is not None:
            self._clear_leftover_scene("server shutdown")
            try:
                self.platform.close()
            except Exception:  # noqa: BLE001
                pass


def _spin_until_quiet(ex, give_up) -> None:
    """Executor turns until one whole SPIN_PERIOD_S passes with nothing to do: the goals that just ended have sent their results."""
    while rclpy.ok() and not give_up():
        t0 = time.monotonic()
        ex.spin_once(timeout_sec=SPIN_PERIOD_S)
        if time.monotonic() - t0 >= SPIN_PERIOD_S:
            return


def main(args=None) -> None:
    from rclpy.signals import SignalHandlerOptions
    from .node_help import print_node_help
    if print_node_help("hongik_auto_teach_server", args):      # --help (English) before rclpy: no node, no connection
        return
    _t, why = kiro_action_type()
    if _t is None:                                             # one clear line, no traceback, non-zero exit (before rclpy / platform)
        print(why, file=sys.stderr, flush=True)
        sys.exit(2)
    # rclpy's own SIGINT handler shuts the context down at once (every node of this process, the platform's included): the running goal
    # could neither stop cleanly nor remove its MoveIt objects. Our handler only records the signal; the loop below shuts down in order
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signals: list = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, _frame: signals.append(signum))
    node = AutoTeachServer()
    ex = MultiThreadedExecutor(num_threads=8)
    ex.add_node(node)
    done = False
    try:
        while rclpy.ok() and not signals:
            ex.spin_once(timeout_sec=SPIN_PERIOD_S)
        if signals:
            node._say(f"{CANCEL_SERVER_SHUTDOWN} ({signal.Signals(signals[0]).name})")
        node.destroy(spin=lambda: ex.spin_once(timeout_sec=SPIN_PERIOD_S), give_up=lambda: len(signals) > 1)
        done = True
        _spin_until_quiet(ex, give_up=lambda: len(signals) > 1)
    finally:
        if not done:
            node.destroy(give_up=lambda: True)                 # an error in the loop: stop the run and close without waiting
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
