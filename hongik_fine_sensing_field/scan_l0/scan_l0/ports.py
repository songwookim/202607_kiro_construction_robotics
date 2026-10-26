"""플랫폼 독립 포트 (KIRO 통합 설계 docs/kiro_integration_design_2026-10-01.md §3, M1).

auto-teach 오케스트레이터(autoteach)·실행기(executor)는 **이 모듈의 포트에만** 의존한다. 로봇에 따라 바뀌는 것은 포트 뒤의
어댑터이며 플랫폼 YAML 이 고른다 (select_adapter):

    pose_source     motion_backend   어댑터 (설정값)
    ros_tf          moveit_jtc       MoveIt 어댑터 ('moveit_jtc')     — hongik_scan_ros.moveit_adapter; move_group + JTC + /joint_states
                                                                        (자세 = robot_description FK, 정적 변환만 TF; 이름 'ros_tf' 는 역사적)

용어 (2026-10-05, 사용자: 'A/B' 대신 무엇이 경로를 만드는지로 부른다):
  MoveIt 어댑터      경로 (직교 직선·IK → 관절 궤적) 는 PC 의 MoveIt 이 만들고, ros2_control 의 JTC 가 100 Hz 관절 지령을 하드웨어 층
                     (로봇 제어기는 플랫폼의 것) 으로 스트리밍한다. 자세 = /joint_states 의 관절을 robot_description URDF 로 푼 FK
                     (2026-10-13; TF 는 정적 변환만).
                     연구실 실기·KIRO 의 유일한 경로 (사용자 2026-10-03: 모든 이동은 MoveIt).
  (RB 제어박스 어댑터 — 제어박스 move_l 직접 — 는 2026-10-07 사용자 결정으로 없앴다. 시험은 같은 포트의 가상 팔
                     scan_l0/tests/sim_arm.py 로 돈다.)

포트
  PoseSource     base←플랜지 실측 자세: 최신값(latest_pose), 정지 스냅숏(snapshot: 자세 + 플랫폼 기록), 시각 이력(history,
                 호스트 monotonic 시각축 — 프로파일 프레임 t_rx 와 같은 축) → 재구성용 RobotSample (robot_samples) / raw npz 배열
                 (sample_arrays), 이력의 한 시각의 자세 (pose_at — 프레임을 촬영 시각의 자세와 짝짓는다).
                 MoveIt 어댑터: /joint_states 수신시각 + 그 관절의 robot_description FK (TF 를 샘플마다 조회하지 않는다).
  MotionBackend  블로킹 직선 이동 move_linear(T_base←flange 목표, 실효 속도, 가속, timeout) → MotionResult (측정 샘플 포함),
                 진행 표시 (set_progress), 중단 (abort). 목표는 **측정 프레임**의 플랜지 자세다.
                 관절 공간 이동 move_joint / 계획 확인 check_joint·check_linear / 미리 계획 prepare_linear 는 MoveIt 어댑터의 확장.
                 Joint-space candidate search (KIRO field 2026-10-14; scan_l0.joint_selection): check_joint(..., allow_ompl) →
                 {ok, reason, plan, _checked_plan} (allow_ompl False: a blocked straight line is an estimate only; PlatformUnavailable
                 is raised, not returned), move_checked_joint(check, q_goal, v, a, timeout_s) = exactly that checked trajectory, once
                 (start, scene and the joints not moved checked again right before sending), path_timing_limits() = what its retime is
                 bound by (joint_selection.TIMING_LIMIT_KEYS). AutoTeach needs move_checked_joint (_universal_ready).
  StopLatch      액션 취소의 정지 + **실행 범위** 정지 표시 (이름은 역사적 — 실행 사이에 남는 래치가 아니다): fire(reason, source) / is_set /
                 reason / last (기록, as_dict) / wait(timeout) / begin_run() / end_run(). fire = 정지 명령 (MoveIt 어댑터: 감속 정지 궤적 →
                 안 되면 우리 목표 + JTC 목표 전부 취소 → /trajectory_execution_event 'stop') + 지금 실행에 '정지됨' 표시 → 그 실행은 더
                 아무것도 보내지 않고 ABORTED 로 끝난다. 해제할 것이 없다: 다음 실행은 begin_run() 으로 깨끗이 시작한다. 실행이 없을 때의
                 정지 요청 = 기록만. 이 패키지에서 fire 를 부르는 것은 액션 취소뿐이다 (실행 중 감시 정지 없음, 사용자 2026-10-11).
  PlatformGates  플랫폼 전용 게이트 훅 (모션 설정, 세션 확인, 상태 요약, 자세 기록).
  RobotPlatform  ScanSession.robot — 위 포트의 묶음 (pose, backend, latch, gates).

공용 결과·예외 타입 (MotionResult, MotionError, MotionConfig) 은 scan_l0.motion_types 의 플랫폼 무관 클래스를 다시 내보낸다 (MoveIt
어댑터도 같은 타입을 돌려준다; 명령 속도 = 실효 속도, v_eff_max_mps = 플랫폼 선언 TCP 속도 한계).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from scan_core.clock_sync import slerp_transform
from scan_core.types import RobotSample

from .motion_types import MotionConfig, MotionError, MotionResult  # noqa: F401 - 공용 결과/예외/설정 타입

__all__ = ["PoseSource", "MotionBackend", "StopLatch", "PlatformGates", "RobotPlatform", "PoseSnapshot",
           "PoseSample", "LatchRecord", "ScanRefused", "PlatformNotReady", "PlatformConfigError", "MotionError", "MotionResult", "MotionConfig",
           "ADAPTERS", "POSE_SOURCE_KINDS", "MOTION_BACKEND_KINDS", "select_adapter",
           "pose_samples_to_robot_samples", "pose_samples_to_arrays", "pose_at_in"]

POSE_SOURCE_KINDS = ("ros_tf",)
MOTION_BACKEND_KINDS = ("moveit_jtc",)
# 어댑터 이름 → (pose_source, motion_backend) — 두 키는 짝으로만 허용
ADAPTERS = {"moveit_jtc": ("ros_tf", "moveit_jtc")}


class ScanRefused(Exception):
    """스크립트/목표를 보내기 전에 거부 (모드 불일치·킬 스위치·세션 초기화·계획 포락선·플랫폼 게이트). 메시지 = 사유."""


class PlatformNotReady(ScanRefused):
    """플랫폼이 실행을 시작할 준비가 아니다 (제어기·하드웨어 비활성, 관절 상태 끊김, 로봇 제어기의 보호 정지 …) — 아무것도 보내기 전.
    메시지 = 준비가 안 된 항목들."""


class PlatformConfigError(ValueError):
    """플랫폼 YAML 이 어댑터를 고를 수 없거나 필수 키가 없다 (시작 거부, 설계 §6.4: 플랫폼 의존 값은 기본값 없음)."""


def select_adapter(params: dict) -> str:
    """플랫폼 YAML(ros__parameters) → 어댑터 이름 ('moveit_jtc'). pose_source ∈ POSE_SOURCE_KINDS, motion_backend ∈
    MOTION_BACKEND_KINDS, 짝은 ADAPTERS 의 것만. 없거나 알 수 없는 값 (예전 RB 제어박스 어댑터 cb5001_fk/cb_move_l 포함) 은
    PlatformConfigError."""
    ps = params.get("pose_source")
    mb = params.get("motion_backend")
    ps = None if ps in (None, "", "null") else str(ps)
    mb = None if mb in (None, "", "null") else str(mb)
    if ps not in POSE_SOURCE_KINDS:
        raise PlatformConfigError(f"platform config: pose_source must be one of {POSE_SOURCE_KINDS}, got {ps!r}")
    if mb not in MOTION_BACKEND_KINDS:
        raise PlatformConfigError(f"platform config: motion_backend must be one of {MOTION_BACKEND_KINDS}, got {mb!r}")
    for name, pair in ADAPTERS.items():
        if pair == (ps, mb):
            return name
    raise PlatformConfigError(f"platform config: pose_source {ps!r} cannot be combined with motion_backend {mb!r} "
                              f"(allowed pairs: {ADAPTERS}) - the pose and the commands must live in the same frame")


# --------------------------------------------------------------------------------------
# 자세 (PoseSource)
# --------------------------------------------------------------------------------------
@dataclass
class PoseSnapshot:
    """정지 측정 한 번: T_base←flange (4x4, m), 호스트 monotonic 시각 [ns], 플랫폼 기록 (JSON 가능 dict), npz 배열 (드웰 grab 저장용),
    raw = 플랫폼 원본 (MoveIt 어댑터: JointState 사본) — 게이트 훅만 해석한다."""

    T: np.ndarray
    t_ns: int = 0
    record: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)
    raw: Any = None


@dataclass(frozen=True)
class PoseSample:
    """플랫폼 무관 자세 샘플 (MoveIt 어댑터 의 이력·모션 측정 샘플). t_rx_ns = 호스트 monotonic 수신 시각 (프로파일 t_rx 와 같은 축),
    stamp_ns = 플랫폼 시각 (ROS header.stamp, sim time 일 수 있음), T_base_flange (4x4, m), joints (rad)."""

    t_rx_ns: int
    T_base_flange: np.ndarray
    joints: tuple = ()
    joint_names: tuple = ()
    stamp_ns: int = 0
    changed: bool = True


def pose_samples_to_robot_samples(samples) -> list:
    """PoseSample 목록 → scan_core RobotSample (재구성 입력; t_rx_s = 호스트 monotonic)."""
    out = []
    for s in samples:
        q = np.asarray(s.joints, dtype=float) if len(s.joints) else None
        out.append(RobotSample(t_rx_s=s.t_rx_ns * 1e-9, T_base_flange=np.asarray(s.T_base_flange, dtype=float), jnt_ang_rad=q))
    return out


def pose_samples_to_arrays(samples) -> dict:
    """PoseSample 목록 → raw npz 배열 (robot_* 접두; 재구성 오프라인 재현 refine 이 읽는 키)."""
    M = len(samples)
    T = np.zeros((M, 4, 4))
    nj = max((len(s.joints) for s in samples), default=0)
    Q = np.full((M, nj), np.nan)
    for i, s in enumerate(samples):
        T[i] = np.asarray(s.T_base_flange, dtype=float)
        if len(s.joints):
            Q[i, : len(s.joints)] = np.asarray(s.joints, dtype=float)
    names = list(samples[0].joint_names) if samples else []
    return {
        "robot_t_rx_ns": np.array([s.t_rx_ns for s in samples], np.int64),
        "robot_stamp_ns": np.array([s.stamp_ns for s in samples], np.int64),
        "robot_changed": np.array([s.changed for s in samples], bool),
        "robot_joint_positions_rad": Q,
        "robot_joint_names": np.array(names),
        "robot_T_base_flange": T,
    }


def pose_at_in(samples, t_s: float) -> Optional[np.ndarray]:
    """시각 오름차순 PoseSample 들 (t_rx_ns) → 시각 t_s [s] 의 T_base←flange: 앞뒤 샘플 사이 보간 (위치 선형·회전 측지선 — 재구성의 TransformBuffer
    와 같은 값, scan_core.clock_sync.slerp_transform), 외삽 없음. None: t_s 가 가장 새 샘플보다 뒤 (아직 오지 않았다). LookupError: 가장 오래된
    샘플보다 앞 (이 이력에 없다). 찾기는 뒤에서부터 (묻는 시각은 최근이다)."""
    if not samples:
        return None
    t_ns = float(t_s) * 1e9
    if t_ns > samples[-1].t_rx_ns:
        return None
    later = None
    for smp in reversed(samples):
        if smp.t_rx_ns <= t_ns:
            if later is None or smp.t_rx_ns == t_ns:
                return np.array(smp.T_base_flange, dtype=float, copy=True)
            return slerp_transform(smp.T_base_flange, later.T_base_flange, (t_ns - smp.t_rx_ns) / (later.t_rx_ns - smp.t_rx_ns))
        later = smp
    raise LookupError(f"t = {t_s:.6f} s is before the pose history ({samples[0].t_rx_ns * 1e-9:.6f} s)")


class PoseSource(abc.ABC):
    """base←플랜지 **실측** 자세의 출처 (I2/I3: 계획·모델값이 아니라 측정값)."""

    kind: str = ""

    @abc.abstractmethod
    def latest_pose(self) -> np.ndarray:
        """최신 실측 T_base←flange (4x4, m)."""

    @abc.abstractmethod
    def snapshot(self) -> PoseSnapshot:
        """정지 측정 (드웰 grab 직후): 자세 + 플랫폼 기록."""

    @abc.abstractmethod
    def history(self, since_ns: int) -> list:
        """since_ns (호스트 monotonic) 이후 자세 샘플 (플랫폼 원본 형식; robot_samples / sample_arrays 로 변환)."""

    @abc.abstractmethod
    def robot_samples(self, samples) -> list:
        """history() 결과 → scan_core RobotSample 목록 (재구성)."""

    @abc.abstractmethod
    def sample_arrays(self, samples) -> dict:
        """history() 결과 → raw npz 배열 (robot_* 접두, robot_T_base_flange · robot_t_rx_ns 필수)."""

    @abc.abstractmethod
    def pose_at(self, t_s: float) -> Optional[np.ndarray]:
        """실측 이력의 시각 t_s [s] (호스트 monotonic — 프로파일 t_rx 와 같은 축) 의 T_base←flange (pose_at_in 의 규칙: 보간, 외삽 없음).
        None = 그 시각의 자세가 아직 오지 않았다 (기다리면 온다), LookupError = 이력에 없다. 다른 스레드가 불러도 막지 않는다."""

    def provider(self) -> Callable[[], np.ndarray]:
        """`() -> T_base←flange` — 다른 스레드가 불러도 된다."""
        return self.latest_pose


# --------------------------------------------------------------------------------------
# 정지 (StopLatch: 액션 취소)
# --------------------------------------------------------------------------------------
@dataclass
class LatchRecord:
    """플랫폼 무관 정지 기록 (StopLatch.fire 한 번). steps = 정지 순서의 단계별 결과 [(step, 'ok' | 'error: …' | 'skipped …')]."""

    reason: str
    source: str = ""
    t_trigger_ns: int = 0
    t_done_ns: int = 0
    steps: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def n_sent(self) -> int:
        return sum(1 for _s, r in self.steps if r == "ok")

    def as_dict(self) -> dict:
        return {"reason": self.reason, "source": self.source, "t_trigger_ns": self.t_trigger_ns, "t_done_ns": self.t_done_ns,
                "latency_ms": (self.t_done_ns - self.t_trigger_ns) * 1e-6 if self.t_done_ns else -1.0,
                "steps": [list(s) for s in self.steps], "errors": list(self.errors), "n_sent": self.n_sent}


class StopLatch(abc.ABC):
    """즉시 정지 + 실행 범위 정지 표시 (모듈 docstring). fire() 는 어느 스레드에서나 부를 수 있다. 실행이 열려 있으면 (run_open)
    정지 명령을 보내고 is_set = True — 그 실행이 끝날 때까지 MotionBackend 는 모든 이동을 거부한다. begin_run() / end_run() 이 표시를
    지운다 (새 실행은 앞 실행의 정지를 물려받지 않는다). 실행이 닫혀 있으면 fire() 는 기록만 남긴다 (우리가 움직이는 것이 없다).
    플랫폼은 만들어질 때 실행이 열린 상태다 (도구·시험은 프로세스·팔 하나가 실행 하나); 실행을 여럿 받는 서버는 end_run() 으로 닫아 두고
    목표마다 begin_run()."""

    run_open: bool = True

    @property
    @abc.abstractmethod
    def is_set(self) -> bool: ...

    @abc.abstractmethod
    def fire(self, reason: str, source: str = "", t_trigger_ns: Optional[int] = None): ...

    @abc.abstractmethod
    def wait(self, timeout_s: float) -> bool: ...

    @abc.abstractmethod
    def begin_run(self) -> None:
        """새 실행: 정지 표시와 그 사유를 지우고 실행을 연다."""

    @abc.abstractmethod
    def end_run(self) -> None:
        """실행 끝: 정지 표시를 지우고 실행을 닫는다 (그 뒤의 정지 요청은 기록만)."""


# --------------------------------------------------------------------------------------
# 모션 (MotionBackend)
# --------------------------------------------------------------------------------------
class MotionBackend(abc.ABC):
    """블로킹 직선 이동. 목표 T 는 측정 프레임(PoseSource 와 같은 프레임) 의 base←플랜지 자세이며 포락선 검사는 호출자가 이미
    끝냈다 (I4) — 백엔드는 목표를 고치지 않는다 (직선이 아니게 되면 거부)."""

    kind: str = ""
    cfg: MotionConfig

    @property
    @abc.abstractmethod
    def latch(self) -> Optional[StopLatch]: ...

    @property
    def estopped(self) -> bool:
        lt = self.latch
        return bool(lt is not None and lt.is_set)

    @abc.abstractmethod
    def move_linear(self, T_target: np.ndarray, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> MotionResult:
        """현재 자세 → T_target 직선 (실효 TCP 속도 v_mps, 가속 a_mps2). 거부·실패는 MotionError 또는 res.ok False."""

    @abc.abstractmethod
    def abort(self) -> None:
        """진행 중인 move_linear 를 끝내라는 요청 (정지 표시 없이)."""

    @property
    def safety_init_result(self):
        return None

    def set_progress(self, period_s: float, cb: Optional[Callable[[str], None]]) -> None:
        self.cfg.progress_s = float(period_s)
        self.cfg.progress_cb = cb


# --------------------------------------------------------------------------------------
# 플랫폼 게이트 훅 · 묶음
# --------------------------------------------------------------------------------------
class PlatformGates(abc.ABC):
    """플랫폼 전용 게이트. executor/autoteach 는 플랫폼 종류를 묻지 않고 이 훅만 부른다."""

    name: str = ""

    @abc.abstractmethod
    def motion_config(self, settings, **overrides) -> MotionConfig: ...

    @abc.abstractmethod
    def session_init(self, platform: "RobotPlatform", cfg: MotionConfig, log: Callable[[str], None]):
        """모션 전 세션 확인 (MoveIt 어댑터: move_group 서비스·JTC·/joint_states·TF 준비). 거부는 ScanRefused. 반환 = 기록 (as_dict())."""

    @abc.abstractmethod
    def status(self, platform: "RobotPlatform") -> dict:
        """플랫폼 상태 요약 (JSON 가능)."""

    @abc.abstractmethod
    def default_speed(self, platform: "RobotPlatform") -> float:
        """속도 정책의 배율 (MoveIt 어댑터: 1.0 — 명령 속도 = 실효 속도)."""

    @abc.abstractmethod
    def effective_speed(self, platform: "RobotPlatform", cfg: MotionConfig, v_target: float) -> float:
        """목표 실효 속도 v_target 을 이 플랫폼에 보냈을 때의 기대 실효 TCP 속도 (클램프 포함)."""

    @abc.abstractmethod
    def frame_offset_record(self, platform: "RobotPlatform", snap: PoseSnapshot) -> dict:
        """드웰 기록: 그 자세의 플랫폼 기록 (관절 이름·값·시각)."""

    def scan_leg_record(self, res: MotionResult) -> dict:
        """스캔 구간 속도 기록 (summary.json speed_verification.scan_leg)."""
        return {"v_cmd_mm_s": res.v_cmd, "v_eff_expected_mps": res.v_eff_expected_mps, "v_measured_mps": res.v_measured_mps,
                "default_speed": res.default_speed, "policy": res.policy}


class RobotPlatform(abc.ABC):
    """ScanSession.robot: 포트 묶음. adapter = 'moveit_jtc' (시험: 가상 팔 'sim_jtc'). pose_source = 출력에 기록되는 자세 출처 이름
    (MoveIt 어댑터: 'ros_tf')."""

    adapter: str = ""
    pose_source: str = ""

    @property
    @abc.abstractmethod
    def pose(self) -> PoseSource: ...

    @property
    @abc.abstractmethod
    def backend(self) -> Optional[MotionBackend]:
        """arm() 뒤의 MotionBackend (그 전에는 None)."""

    @property
    @abc.abstractmethod
    def latch(self) -> Optional[StopLatch]: ...

    @property
    @abc.abstractmethod
    def gates(self) -> PlatformGates: ...

    @abc.abstractmethod
    def arm(self, cfg: MotionConfig):
        """우리 MotionBackend 객체를 만든다 (소프트웨어만; 로봇에는 아무것도 보내지 않는다). 세션 확인은 gates.session_init."""

    @abc.abstractmethod
    def close(self) -> None: ...

    # ---- 편의 (포트 위임) ----
    @property
    def armed(self) -> bool:
        return self.backend is not None

    @property
    def estopped(self) -> bool:
        lt = self.latch
        return bool(lt is not None and lt.is_set)

    def current_flange_pose(self) -> np.ndarray:
        return self.pose.latest_pose()

    def pose_provider(self) -> Callable[[], np.ndarray]:
        return self.pose.provider()
