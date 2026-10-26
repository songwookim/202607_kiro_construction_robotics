"""자동 교시 오케스트레이터 (scan_l0.autoteach / teach_1g) 가 쓰는 구간 실행·센서·저장 도우미 — 플랫폼 포트 (scan_l0.ports) 에만 의존한다.

모든 이동은 MoveIt 을 거친다 (hongik_scan_ros.moveit_adapter; 시험은 가상 팔 scan_l0/tests/sim_arm.py).

  gate_and_arm      우리 모션 백엔드 객체 생성 → session_init (MoveIt/JTC 준비 확인)
  ensure_sensor     센서 접속·파라미터 백업·초기화
  prepare_leg / execute_leg   구간 계획 (센서를 켜기 전에) / 실행 (시작 자세 게이트: 전송 직전 실측 자세가 계획 시작과 다르면 거부)
  start_sensor_recording / stop_sensor_recording   첫 프레임을 받은 뒤에야 녹화가 시작된 것으로 본다 (오지 않으면 SensorNotStarted →
                    그 구간은 보내지 않고 REFUSED); 정지는 RESET
  save_raw / raw_payload / reconstruct_and_qc   원자료 저장·재구성·QC
"""
from __future__ import annotations

import datetime as _dt
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from scan_core.validity import Z_VALID_MM_DEFAULT
from scan_core.io_npz import save_scan_result
from scan_core.quality import integrity_report
from scan_core.reconstruct import reconstruct_scan
from scan_core.types import (Calib, ProfileFrame, QCReport, ScanJob, ScanResult,
                             _jsonable)


from .config import Settings, job_to_dict
from .ports import MotionBackend, MotionError, MotionResult, PoseSource, RobotPlatform, ScanRefused
from .session import ScanSession, SensorNotStarted, SensorSession, records_to_arrays

__all__ = ["ScanRunResult", "ScanRefused", "save_raw", "reconstruct_and_qc", "frames_from_records", "ensure_sensor", "gate_and_arm",
           "execute_leg", "prepare_leg", "check_leg", "start_sensor_recording", "stop_sensor_recording", "StartPoseMismatch",
           "start_pose_mismatch", "motion_dict", "FIRST_FRAME_TIMEOUT_S", "SensorNotStarted",
           "motion_timeout_shortfall", "leg_pose_error", "LEG_POSE_WARN_M", "LEG_POSE_WARN_DEG", "CANCEL_OPERATOR",
           "CANCEL_SERVER_SHUTDOWN"]

STATUS_OK = "OK"
STATUS_ABORTED = "ABORTED"
STATUS_REFUSED = "REFUSED"
STATUS_ERROR = "ERROR"

# the causes of a run stopped from outside (the run's cancel Future carries the cause as its result): the hosting server shutting
# down (its result is exactly CANCEL_SERVER_SHUTDOWN) or, any other result, an action cancel (D7)
CANCEL_OPERATOR = "cancel requested by the operator"
CANCEL_SERVER_SHUTDOWN = "server shutting down"


@dataclass
class ScanRunResult:
    status: str = STATUS_ERROR
    reason: str = ""
    out_dir: Optional[str] = None
    raw_path: Optional[str] = None
    recon_dir: Optional[str] = None
    qc_path: Optional[str] = None
    qc: Optional[QCReport] = None
    scan_result: Optional[ScanResult] = None
    plan_dict: dict = field(default_factory=dict)
    plan_text: str = ""
    envelope_stats: dict = field(default_factory=dict)
    safety_init: Optional[object] = None               # 플랫폼 게이트 session_init 기록 (as_dict(); MoveIt 어댑터: MoveItInitRecord)
    motions: list = field(default_factory=list)        # (waypoint name, MotionResult)
    scan_motion: Optional[MotionResult] = None
    records: list = field(default_factory=list)        # vc3dz ProfileRecord
    robot_samples: list = field(default_factory=list)  # PoseSource.history() 샘플 (PoseSample)
    sensor_info: dict = field(default_factory=dict)
    sensor_summary: dict = field(default_factory=dict)
    speed_verification: dict = field(default_factory=dict)
    estop_record: Optional[dict] = None
    notes: list = field(default_factory=list)
    leg_pose_check: list = field(default_factory=list)   # S4: 구간별 |실측 도착 − 계획 목표| (프레임 일관성 진단)
    sensor_restore: Optional[dict] = None                 # S6: 소유 세션의 센서 파라미터 복원 결과 (summary.json 전에)
    t_sensor_start_ns: int = 0
    timing: dict = field(default_factory=dict)           # reconstruct_and_qc 단계별 시간 [s] (패스 처리 시간 기록)
    t_sensor_stop_ns: int = 0
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def summary_dict(self) -> dict:
        return {
            "status": self.status, "reason": self.reason, "out_dir": self.out_dir, "raw_path": self.raw_path,
            "recon_dir": self.recon_dir, "qc_path": self.qc_path,
            "qc_verdict": self.qc.verdict if self.qc else None, "qc_failures": list(self.qc.failures) if self.qc else [],
            "n_frames": len(self.records), "n_robot_samples": len(self.robot_samples),
            "motions": [motion_dict(n, r) for n, r in self.motions],
            "scan_motion": self.scan_motion.as_dict() if self.scan_motion else None,
            "safety_init": self.safety_init.as_dict() if self.safety_init else None,
            "speed_verification": self.speed_verification, "envelope_stats": _jsonable(self.envelope_stats),
            "sensor_summary": _jsonable(self.sensor_summary), "estop_record": self.estop_record, "notes": list(self.notes),
            "leg_pose_check": _jsonable(self.leg_pose_check), "sensor_restore": _jsonable(self.sensor_restore),
            "t_sensor_start_ns": self.t_sensor_start_ns, "t_sensor_stop_ns": self.t_sensor_stop_ns,
            "elapsed_s": self.elapsed_s,
        }


# --------------------------------------------------------------------------------------
# 변환 헬퍼
# --------------------------------------------------------------------------------------
def frames_from_records(recs) -> list:
    return [ProfileFrame(x_mm=r.x, z_mm=r.z, ts_ms=int(r.ts_ms), result_counter=int(r.result_counter),
                         t_rx_s=r.t_rx_ns * 1e-9, global_counter=int(r.global_counter), line_counter=r.line_counter)
            for r in recs]


# --------------------------------------------------------------------------------------
# 저장
# --------------------------------------------------------------------------------------
def save_raw(out_dir, run: ScanRunResult, job: ScanJob, calib: Calib, settings: Optional[Settings], cfg,
             pose: Optional[PoseSource] = None) -> Path:
    """raw.npz: 센서 프레임 + 로봇 샘플 (pose.sample_arrays) + job/calib/config/plan/안전초기화/모션/속도검증 (JSON 문자열)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays, js = raw_payload(run, job, calib, settings, cfg, pose)
    path = out / "raw.npz"
    np.savez_compressed(path, **arrays, **{k: np.array(v) for k, v in js.items()})
    run.raw_path = str(path)
    return path


def raw_payload(run: ScanRunResult, job: ScanJob, calib: Calib, settings: Optional[Settings], cfg,
                pose: Optional[PoseSource] = None) -> tuple:
    """raw.npz 내용 (배열 dict, JSON 문자열 dict) — 쓰기는 save_raw 또는 패스 뒤 처리 작업 프로세스 (post_scan.compute_pass)."""
    arrays = records_to_arrays(run.records)
    if run.robot_samples and pose is not None:
        arrays.update(pose.sample_arrays(run.robot_samples))
    js = {
        "job_json": json.dumps(job_to_dict(job)),
        "calib_json": json.dumps(calib.to_dict()),
        "config_json": json.dumps(_jsonable(settings.as_dict()) if settings is not None else {}),
        "motion_config_json": json.dumps(_jsonable(dict(cfg.__dict__)) if cfg is not None else {}, default=str),
        "plan_json": json.dumps(_jsonable(run.plan_dict), default=str),
        "safety_init_json": json.dumps(run.safety_init.as_dict() if run.safety_init else {}, default=str),
        "motions_json": json.dumps([motion_dict(n, r) for n, r in run.motions], default=str),
        "speed_verification_json": json.dumps(_jsonable(run.speed_verification), default=str),
        "sensor_info_json": json.dumps(_jsonable(run.sensor_info), default=str),
        "sensor_summary_json": json.dumps(_jsonable(run.sensor_summary), default=str),
        "status_json": json.dumps({"status": run.status, "reason": run.reason, "notes": run.notes,
                                   "t_sensor_start_ns": run.t_sensor_start_ns, "t_sensor_stop_ns": run.t_sensor_stop_ns,
                                   "saved": _dt.datetime.now().isoformat(timespec="seconds")}),
    }
    return arrays, js




def reconstruct_and_qc(run: ScanRunResult, job: ScanJob, calib: Calib, pose: PoseSource, z_valid_mm=tuple(Z_VALID_MM_DEFAULT),
                       out_dir=None, ts_tol_ms: Optional[float] = None) -> tuple:
    """녹화된 프레임/로봇 샘플 → ScanResult + QCReport (무결성 사실만, verdict 'UNJUDGED' — 판정 없음; 저장 옵션). 실패 시 (None,
    QCReport: failures = 재구성 예외).
    pose: 그 샘플을 기록한 PoseSource (robot_samples 로 RobotSample 변환; 시각축 = 호스트 monotonic, 프레임 t_rx 와 같다).
    ts_tol_ms: 촬영 멈춤 표시 문턱 (기본 clock_sync.stall_tol_ms(fps) = 프레임 주기 − 타임스탬프 양자 둘)."""
    t_ = time.perf_counter()
    frames = frames_from_records(run.records)
    robot = pose.robot_samples(run.robot_samples)
    run.timing["convert_s"] = time.perf_counter() - t_
    try:
        if len(frames) < 2 or len(robot) < 2:
            raise ValueError(f"not enough data for reconstruction: frames={len(frames)} robot_samples={len(robot)}")
        t_ = time.perf_counter()
        res = reconstruct_scan(frames, robot, calib, job, fps_nominal=job.profile_fps, z_valid_mm=z_valid_mm,
                               ts_tol_ms=ts_tol_ms)
        run.timing["reconstruct_s"] = time.perf_counter() - t_
        t_ = time.perf_counter()
        ate = int(run.sensor_info.get("params_after", {}).get("AutoTriggerError", 0) or 0) if run.sensor_info else 0
        qc = integrity_report(res, auto_trigger_error=ate)
        res.qc = qc
        run.timing["qc_s"] = time.perf_counter() - t_
    except Exception as e:  # noqa: BLE001 - QC must never mask the raw save
        qc = QCReport(verdict="UNJUDGED", failures=[f"reconstruction failed: {e}"], notes=[traceback.format_exc(limit=3)])
        res = None
    run.qc = qc
    run.scan_result = res
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if res is not None:
            t_ = time.perf_counter()
            save_scan_result(out / "recon", res, extra_meta={"status": run.status, "reason": run.reason})
            run.timing["save_recon_s"] = time.perf_counter() - t_
            run.recon_dir = str(out / "recon")
        qp = out / "qc.json"
        with open(qp, "w", encoding="utf-8") as f:
            json.dump(qc.to_dict(), f, indent=1, ensure_ascii=False, default=str)
        run.qc_path = str(qp)
    return res, qc


# --------------------------------------------------------------------------------------
# 재사용 가능한 단계 헬퍼 (auto-teach 오케스트레이터가 개별 구간을 실행할 때도 쓴다)
# --------------------------------------------------------------------------------------


def ensure_sensor(sess: ScanSession, settings: Settings, *, sensor_ip: Optional[str] = None,
                  sensor_port: Optional[int] = None, sensor_init: Optional[dict] = None, backup_dir=None,
                  force_init: bool = False, log: Optional[Callable[[str], None]] = None) -> SensorSession:
    """세션의 SensorSession 을 (없으면 접속해) 준비한다. init 은 처음 한 번 또는 force_init / sensor_init 지정 시."""
    p = log or (lambda s: None)
    if sess.sensor is None:
        sess.sensor = SensorSession(sensor_ip or settings.sensor_ip, sensor_port or settings.sensor_port).open()
        sess.sensor_ready = False
    ss = sess.sensor
    if force_init or sensor_init is not None or not sess.sensor_ready:
        mism = ss.init(sensor_init, backup_dir=backup_dir)
        sess.sensor_ready = True
        if mism:
            p(f"  sensor init mismatch (kept): {mism}")
        for w_ in ss.warnings:
            p("WARNING: " + w_)
        p(f"sensor ready: MainVersion={ss.info.params_after.get('MainVersion')} DataMode={ss.info.params_after.get('DataMode')} "
          f"AutoTriggerFPS={ss.info.params_after.get('AutoTriggerFPS')} ROI_DX_PIX={ss.info.params_after.get('ROI_DX_PIX')}")
    return ss


def gate_and_arm(sess: ScanSession, cfg, *, log: Optional[Callable[[str], None]] = None):
    """MotionBackend 객체 생성 (arm: 소프트웨어만) → 세션 확인 (플랫폼 훅: MoveIt/JTC 준비).

    이미 arm 된 공유 세션은 백엔드를 재사용한다 (cfg 는 그 백엔드의 것을 유지; 호출자가 같은 객체를 넘긴다).
    거부는 ScanRefused (아무것도 보내기 전). 반환 = 게이트 기록."""
    p = log or (lambda s: None)
    rs: RobotPlatform = sess.robot
    if not rs.armed:
        rs.arm(cfg)
    assert rs.backend is not None
    return rs.gates.session_init(rs, cfg, p)


class StartPoseMismatch(MotionError):
    """전송 직전 실측 자세가 계획의 시작 자세와 다르다 (프롬프트 동안 조그 등) — 스크립트를 보내지 않았다 (REFUSED 로 처리)."""


def start_pose_mismatch(T_expected, T_measured, tol_m: float, tol_deg: float) -> Optional[str]:
    """계획 시작 자세 T_expected 와 실측 T_measured 의 차이가 허용치 (플랫폼 도착 허용 goal_pose_tol; SafetyPolicy.goal_tol_*) 를
    넘으면 사유 문자열, 아니면 None."""
    if T_expected is None:
        return None
    A = np.asarray(T_expected, dtype=float)
    B = np.asarray(T_measured, dtype=float)
    dp = float(np.linalg.norm(B[:3, 3] - A[:3, 3]))
    R = A[:3, :3].T @ B[:3, :3]
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    dr = float(np.degrees(np.arccos(c)))
    if dp > tol_m or dr > tol_deg:
        return (f"pose changed since the envelope check ({dp * 1e3:.1f} mm, {dr:.2f} deg > {tol_m * 1e3:.1f} mm / {tol_deg:g} deg): "
                "the executed segment would not be the checked one (jogged during the prompt?) - re-plan from the current pose")
    return None


LEG_POSE_WARN_M = 1.0e-3        # S4: 구간 도착 실측 vs 계획 목표 — 프레임 일관성 경고 (거부 아님)
LEG_POSE_WARN_DEG = 0.2


def leg_pose_error(T_planned, T_measured) -> tuple:
    """(|Δp| [m], |Δr| [deg]) 계획 목표 vs 실측 도착 자세."""
    A = np.asarray(T_planned, dtype=float)
    B = np.asarray(T_measured, dtype=float)
    dp = float(np.linalg.norm(B[:3, 3] - A[:3, 3]))
    R = A[:3, :3].T @ B[:3, :3]
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return dp, float(np.degrees(np.arccos(c)))


def motion_timeout_shortfall(timeout_s: Optional[float], dist_m: float, v_eff_mps: float, name: str = "leg",
                             margin_s: float = 10.0, factor: float = 2.0) -> Optional[str]:
    """F-C9 fail-early: 명시 타임아웃(--motion-timeout) 이 factor·거리/v_eff + margin 보다 짧으면 사유 문자열 (아니면 None).
    timeout_s None (타임아웃을 계획에서 유도) 이면 항상 None — RBMotion 이 같은 식으로 유도한다."""
    if timeout_s is None:
        return None
    need = factor * float(dist_m) / max(float(v_eff_mps), 1e-6) + margin_s
    if float(timeout_s) < need:
        return (f"leg '{name}': --motion-timeout {float(timeout_s):.1f} s is shorter than the leg needs "
                f"({factor:.0f} x {float(dist_m) * 1e3:.0f} mm / {float(v_eff_mps) * 1e3:.2f} mm/s + {margin_s:.0f} s = {need:.1f} s) "
                "at the effective speed of this session: the motion loop would end the leg early - "
                "raise --motion-timeout or omit it (the timeout is then derived from the leg)")
    return None


def execute_leg(sess: ScanSession, w, v_target: float, *, timeout_s: Optional[float] = None,
                expected_start=None, start_tol: tuple = (0.0, 0.0)) -> MotionResult:
    """웨이포인트 1개를 MotionBackend.move_linear 로 실행 (백엔드의 게이트 그대로). 예외 MotionError 는 호출자가 abort 한다.
    expected_start: 계획이 검사한 시작 자세 (4x4). 주어지면 전송 직전 실측 자세와 비교해 start_tol (m, deg; 플랫폼 도착 허용) 을
                    넘으면 StartPoseMismatch (스크립트 전송 없음; 호출자는 REFUSED 로 보고한다) — 리뷰 C8.

    w.T_base_flange 는 측정 프레임의 목표이며 포락선 검사는 그 값에 대해 끝났다 (MoveIt 어댑터: 측정 프레임 = 명령 프레임)."""
    be: Optional[MotionBackend] = sess.robot.backend
    assert be is not None, "session not armed (gate_and_arm first)"
    if expected_start is not None:
        why = start_pose_mismatch(expected_start, sess.robot.pose.latest_pose(), float(start_tol[0]), float(start_tol[1]))
        if why:
            raise StartPoseMismatch(f"leg {getattr(w, 'name', '?')}: {why}")
    return be.move_linear(w.T_base_flange, v_target, float(w.accel_mps2), timeout_s=timeout_s)


def prepare_leg(sess: ScanSession, w, v_target: float, *, timeout_s: Optional[float] = None) -> Optional[dict]:
    """센서 녹화를 켜기 **전에** 이 구간의 계획을 미리 한다 — 백엔드가 지원할 때만 (MoveIt 어댑터 MoveItJtcBackend.prepare_linear).
    이어지는 execute_leg 가 같은 인자로 move_linear 를 부르면 그 계획을 그대로 보낸다 (팔이 그동안 움직였으면 다시 계획). 실기
    at09/at10 (2026-10-03): 녹화 중의 계획 계산 (420 mm 조밀 IK ~0.9 s) 이 같은 프로세스의 녹화 스레드를 밀었다. 실패는 조용히 None
    (move_linear 가 다시 계획하며 거부 사유를 보고한다)."""
    be = sess.robot.backend if sess.robot is not None else None
    prep = getattr(be, "prepare_linear", None)
    if prep is None:
        return None
    try:
        return prep(w.T_base_flange, v_target, float(w.accel_mps2), timeout_s=timeout_s)
    except Exception:  # noqa: BLE001
        return None


def check_leg(sess: ScanSession, w, v_target: float, *, timeout_s: Optional[float] = None) -> Optional[dict]:
    """이 구간을 **보내지도 저장하지도 않고** 지금 측정 자세에서 계획해 본다 (확인 프롬프트의 문구용 — 실행할 계획은 prepare_leg 가
    프롬프트 뒤, 센서를 켜기 직전의 자세에서 따로 만든다). 반환은 구조로 구분한다: None = 백엔드가 보내기 전에 계획하는 기능이 없다
    (check_linear 없음), {ok, reason, plan} = 백엔드의 결과 (ok False = 지금 자세에서 계획이
    거부·불가, reason 은 그 사유)."""
    be = sess.robot.backend if sess.robot is not None else None
    chk = getattr(be, "check_linear", None)
    if chk is None:
        return None
    try:
        r = chk(w.T_base_flange, v_target, float(w.accel_mps2), timeout_s=timeout_s)
    except Exception as e:  # noqa: BLE001 - the prompt states the failure; move_linear plans again and reports its own refusal
        return {"ok": False, "reason": f"{type(e).__name__}: {e}", "plan": None}
    r = dict(r or {})
    return {"ok": bool(r.get("ok")), "reason": str(r.get("reason") or ""), "plan": r.get("plan")}


def motion_dict(name: str, res: MotionResult) -> dict:
    """MotionResult → 기록 dict."""
    return {"waypoint": name, **res.as_dict()}


FIRST_FRAME_TIMEOUT_S = 2.0     # S1: 센서 시작 → 첫 프레임 대기 상한 (실기 수백 ms; auto_teach.first_frame_timeout_s)


def start_sensor_recording(sess: ScanSession, first_frame_timeout_s: Optional[float] = FIRST_FRAME_TIMEOUT_S) -> int:
    """녹화 시작 → **첫 프레임을 받을 때까지** 기다린다 (리뷰 S1: 실기는 시작 뒤 첫 프레임까지 수백 ms). first_frame_timeout_s 안에
    프레임이 없으면 녹화를 멈추고 (RESET) SensorNotStarted — 호출자는 그 구간을 보내지 않고 REFUSED 로 보고한다. None 이면 기다리지
    않는다 (시험 전용)."""
    assert sess.sensor is not None
    ss = sess.sensor
    t0 = ss.start_recording()
    if first_frame_timeout_s is not None and not ss.wait_first_frame(float(first_frame_timeout_s)):
        try:
            ss.stop_recording(settle_s=0.0)
        except Exception:  # noqa: BLE001 - the refusal must be reported even if the RESET fails
            pass
        ss.records = []
        raise SensorNotStarted(f"sensor did not start: no profile frame within {float(first_frame_timeout_s):.1f} s after the "
                               "start command (check the sensor link/laser; nothing was sent for this leg)")
    return t0


def stop_sensor_recording(sess: ScanSession, settle_s: float = 0.1) -> list:
    assert sess.sensor is not None
    return sess.sensor.stop_recording(settle_s=settle_s)


# --------------------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------------------

