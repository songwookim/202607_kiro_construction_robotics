"""시험용 가상 팔 (ROS 없이) — MoveIt 어댑터 (hongik_scan_ros.moveit_adapter) 와 같은 포트 (scan_l0.ports) 와 같은 계획 규칙.

2026-10-07 사용자 결정: RB 제어박스 어댑터를 없애고 모든 이동은 MoveIt 을 거친다. 빠른 합성 시험 (자동 교시 종단 등) 은 가상
제어박스 대신 이 가상 팔로 돈다:

* 기구학: 연구실 RB20-1900ES-U + EOAT URDF (tests/data, xacro 결과) 의 link0 → tcp 체인 (scan_core.trajectory.fast_fk), 관절 한계 그대로
* 직선 이동: scan_core.trajectory.track_straight_line (MoveIt 어댑터와 같은 조밀 IK·관절 점프·한계 거부) + retime_straight_path
* 관절 공간 이동: 관절 직선을 충돌 장면 (상자) 으로 검사, 막히면 중간 자세 하나를 거치는 표본 계획 (OMPL 대용). The MoveIt adapter's
  candidate-search interface too: check_joint(allow_ompl=False) = a blocked straight line is an estimate only, every check keeps its
  own plan (_checked_plan), move_checked_joint sends exactly that plan once (start and scene checked again right before),
  path_timing_limits (the duration bound's limits)
* 도달 후보: 관절 한계 안 무작위 시드 수치 IK (감쇠 최소제곱) → track_polyline 추종·한계 여유·야코비안 최소 특이값 (MoveIt 어댑터와 같은 형식)
* 충돌 장면: 상자 (id, size, T) — 팔 링크 선분 표본점 + 툴 몸체 점 (캘리브 body_points_eoat; 없으면 키포인트) 이 상자 안이면 충돌
* 실행: 계획한 관절 궤적을 시계 (vc3dz_driver.clock — 기본 벽시계 실시간, 시험의 가상 시계면 시뮬 시간) 로 8 ms 틱마다 재생 (가짜 센서도 같은
  시계로 프레임을 낸다), 자세 이력 (재구성 입력), 정지 래치 (재생 즉시 멈춤). 재생 자세는 순수 함수 playback_q (선계산과 같은 식)
* 관절 상태 (/joint_states, 2026-10-12 사용자 — scan_core.js_source): JointStateModel. 'measured' (기본) = 물리 팔의 엔코더 각
  (카운트 격자, 제어박스 float32 도 × DEG2RAD; 선택: 서보 지연·진동·떨림), 'reference' = 제어박스의 관절 기준값 메아리 (명령이 1차
  look-ahead 필터를 지난 값, float32 도 — 수렴 뒤 비트까지 그대로), 물리 팔은 그 뒤를 echo_lag_s 만큼 따라간다. 센서는 물리 팔을 본다
  (physical_pose); 자세 이력·latest_pose 는 보고된 관절 (/joint_states).
"""
from __future__ import annotations

import io
import math
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from scan_core.fk_urdf import chain_from_joints, parse_urdf_joints
from scan_l0.motion_types import MotionConfig, MotionError, MotionResult
from scan_core.trajectory import (_jacobian, _pose_error, densify_joint_path, fast_fk, follow_candidate, joint_path_stats,
                                  nearest_turn_equivalent, retime_joint_path, retime_straight_path, track_polyline,
                                  track_straight_line)
from scan_l0.ports import (LatchRecord, MotionBackend, PlatformGates, PoseSample, PoseSnapshot, PoseSource, RobotPlatform,
                           StopLatch, pose_at_in, pose_samples_to_arrays, pose_samples_to_robot_samples)
from vc3dz_driver.clock import get_clock

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "data", "rb20_1900es_u_eoat.urdf")
BASE, FLANGE, TORCH = "link0", "tcp", "torch_tcp"
FOLDED_Q = (0.0, 0.0, 2.5, 0.0, 1.57, 0.0)            # 운송용으로 접은 자세 (가짜 팔 MoveIt 시험과 같다)


def _now_ns(clock=None) -> int:
    return (clock if clock is not None else get_clock()).monotonic_ns()


def playback_q(T: np.ndarray, Q: np.ndarray, t_s: float) -> np.ndarray:
    """재생 시각 t_s [s] 의 관절 (계획 시각 T 에서 관절별 선형 보간) — SimBackend._run 과 선계산이 같은 식을 쓴다."""
    return np.array([np.interp(t_s, T, Q[:, j]) for j in range(Q.shape[1])])


def playback_t(t0_ns: int, now_ns: int, time_scale: float) -> float:
    """재생 시작 t0_ns 에서 now_ns 까지의 재생 시각 [s] (SimBackend._run 과 선계산이 같은 식)."""
    return (int(now_ns) - int(t0_ns)) * 1e-9 / time_scale


def _rot_deg(Ra, Rb) -> float:
    c = (float(np.trace(np.asarray(Ra).T @ np.asarray(Rb))) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


DEG = 180.0 / math.pi
RB20_COUNT_RAD = 2.0 * math.pi / (16384 * 101)      # an RB20 joint's encoder count through its gear (lab CB jnt_ang steps)


def _f32deg(q) -> np.ndarray:
    """rbpodo read(): the control box's float32 degrees × DEG2RAD (double)."""
    return (np.asarray(q, dtype=float) * DEG).astype(np.float32).astype(np.float64) / DEG


@dataclass
class JointStateModel:
    """What the virtual arm's /joint_states carry (user 2026-10-12; scan_core.js_source tells them apart).

    measured   the encoder angles of the physical arm: physical = the command servo_lag_s late (+ vibration), reported = its encoder
               count (count_rad, the encoder's own zero) read with dither (sd dither_counts) → float32 degrees. Default: no lag, no
               vibration, no dither — the arm as before (the sensor sees the command), its joint states quantised like the lab's.
    reference  the control box's joint reference echoed back: reported = the command through a first-order look-ahead filter
               (lookahead_s) → float32 degrees (converges, then holds bit-exactly); the physical arm trails the reported reference
               by echo_lag_s (servo) — a lag the time-offset prior does not know."""

    source: str = "measured"
    count_rad: float = RB20_COUNT_RAD
    servo_lag_s: float = 0.0
    dither_counts: float = 0.0
    vib_amp_rad: float = 0.0
    vib_hz: float = 3.5
    lookahead_s: float = 0.03
    echo_lag_s: float = 0.0
    seed: int = 0


class _JointStateState:
    """The model's state: last command, filter output, the delay line of the signal the physical arm follows."""

    def __init__(self, m: JointStateModel, q0, t_ns: int):
        if m.source not in ("measured", "reference"):
            raise ValueError(f"JointStateModel.source must be measured | reference, got {m.source!r}")
        self.m = m
        self.rng = np.random.default_rng(m.seed)
        q0 = np.asarray(q0, dtype=float)
        self.zero = self.rng.uniform(0.0, m.count_rad, q0.size)
        self.reset(q0, t_ns)

    def reset(self, q, t_ns: int) -> None:
        q = np.asarray(q, dtype=float).copy()
        self.cmd, self.y, self.t_ns = q.copy(), q.copy(), int(t_ns)
        self.line: deque = deque([(int(t_ns), q.copy())])           # (t, value the physical arm follows)
        self.t_cmd_end_ns = int(t_ns)

    def _delayed(self, t_ns: int, lag_s: float) -> np.ndarray:
        tq = t_ns - int(round(lag_s * 1e9))
        h = self.line
        if tq <= h[0][0]:
            return h[0][1].copy()
        for (ta, qa), (tb, qb) in zip(reversed(list(h)[:-1]), reversed(list(h)[1:])):
            if ta <= tq <= tb:
                w = (tq - ta) / max(tb - ta, 1)
                return qa + w * (qb - qa)
        return h[-1][1].copy()

    def step(self, cmd, t_ns: int) -> tuple:
        """Advance to t_ns with the command cmd → (reported, physical)."""
        m = self.m
        cmd = np.asarray(cmd, dtype=float)
        if np.any(cmd != self.cmd):
            self.t_cmd_end_ns = int(t_ns)
        self.cmd = cmd.copy()
        dt = max((int(t_ns) - self.t_ns) * 1e-9, 0.0)
        self.t_ns = int(t_ns)
        if m.source == "reference":
            a = math.exp(-dt / m.lookahead_s) if m.lookahead_s > 0 else 0.0
            self.y = a * self.y + (1.0 - a) * cmd
            rep = _f32deg(self.y)
            follow, lag = rep, m.echo_lag_s
        else:
            follow, lag = cmd, m.servo_lag_s
        self.line.append((int(t_ns), np.asarray(follow, dtype=float).copy()))
        keep = int(round((lag + 1.0) * 1e9))
        while len(self.line) > 2 and self.line[1][0] < int(t_ns) - keep:
            self.line.popleft()
        phys = self._delayed(int(t_ns), lag) if lag > 0 else np.asarray(follow, dtype=float).copy()
        if m.source == "measured":
            if m.vib_amp_rad > 0:
                phys = phys + m.vib_amp_rad * math.sin(2 * math.pi * m.vib_hz * t_ns * 1e-9) * np.ones_like(phys)
            noise = self.rng.normal(0.0, m.dither_counts, phys.size) * m.count_rad if m.dither_counts > 0 else 0.0
            rep = _f32deg(np.round((phys + noise - self.zero) / m.count_rad) * m.count_rad + self.zero)
        return rep, phys

    def settled(self, t_ns: int) -> bool:
        """After the command ended: the reference has converged on it (float32 degrees, bit-exact) and the physical arm has caught up
        with what it follows."""
        m = self.m
        lag = m.echo_lag_s if m.source == "reference" else m.servo_lag_s
        if m.source == "reference" and not np.array_equal(_f32deg(self.y), _f32deg(self.cmd)):
            return False
        return int(t_ns) - self.t_cmd_end_ns >= int(round(lag * 1e9))


@dataclass
class SimArmConfig:
    """플랫폼 한계 (MoveIt 어댑터 moveit_jtc 섹션과 같은 뜻)."""

    max_tcp_speed_mps: float = 0.06
    max_tcp_accel_mps2: float = 0.3
    max_joint_velocity_rad_s: float = 1.0
    max_joint_accel_rad_s2: float = 2.0
    cartesian_step_m: float = 0.002
    max_joint_step_rad: float = 0.05
    rotation_radius_m: float = 0.2
    goal_pose_tol_m: float = 1e-3
    goal_pose_tol_deg: float = 0.2
    rate_hz: float = 125.0                     # 재생·자세 이력 주기
    time_scale: float = 1.0                    # 1 = 벽시계 실시간
    joint_states: JointStateModel = field(default_factory=JointStateModel)    # /joint_states: measured (기본) | reference

    @classmethod
    def from_settings(cls, settings) -> "SimArmConfig":
        mj = dict((settings.params.get("moveit_jtc") or {}) if settings is not None else {})
        lim = dict(mj.get("limits") or {})
        kw = {}
        for k in ("max_tcp_speed_mps", "max_tcp_accel_mps2", "max_joint_velocity_rad_s", "max_joint_accel_rad_s2"):
            if lim.get(k) is not None:
                kw[k] = float(lim[k])
        for k in ("cartesian_step_m", "max_joint_step_rad", "rotation_radius_m", "goal_pose_tol_m", "goal_pose_tol_deg"):
            if mj.get(k) is not None:
                kw[k] = float(mj[k])
        return cls(**kw)


# --------------------------------------------------------------------------------------
# 자세 출처
# --------------------------------------------------------------------------------------
class SimPoseSource(PoseSource):
    def __init__(self, arm: "SimArmPlatform", history_s: float = 900.0):
        self.arm = arm
        self._lock = threading.Lock()
        self._q = np.asarray(FOLDED_Q, dtype=float)
        self._T = arm.fk()(self._q)
        self._qp = self._q.copy()                        # the physical arm (what the sensor sees)
        self._Tp = self._T.copy()
        self._hist: deque = deque(maxlen=int(history_s * arm.cfg.rate_hz))
        self._t_last_ns = _now_ns(arm.clock)
        self._js = _JointStateState(arm.cfg.joint_states, self._q, self._t_last_ns)

    def set_q(self, q, record: bool = True) -> None:
        """The command at 'now' (a playback tick): the joint-state model gives the reported joints (/joint_states, recorded) and
        the physical arm."""
        q = np.asarray(q, dtype=float).copy()
        t = _now_ns(self.arm.clock)
        rep, phys = self._js.step(q, t)
        self._store(rep, phys, t, record)

    def place(self, q) -> None:
        """Test setup / start: the arm stands at q (no motion, no lag): reported = the model's reading of q, physical = q."""
        q = np.asarray(q, dtype=float).copy()
        t = _now_ns(self.arm.clock)
        self._js.reset(q, t)
        rep, phys = self._js.step(q, t)
        self._store(rep, phys, t, True)

    def settled(self) -> bool:
        return self._js.settled(_now_ns(self.arm.clock))

    def _store(self, rep, phys, t: int, record: bool) -> None:
        fk = self.arm.fk()
        T = fk(np.asarray(rep, dtype=float))
        Tp = fk(np.asarray(phys, dtype=float))
        with self._lock:
            self._q, self._T, self._t_last_ns = np.asarray(rep, dtype=float), T, t
            self._qp, self._Tp = np.asarray(phys, dtype=float), Tp
            if record:
                self._hist.append(PoseSample(t_rx_ns=t, T_base_flange=T.copy(), joints=tuple(float(v) for v in rep),
                                             joint_names=tuple(self.arm.joint_names), stamp_ns=t))

    def physical_pose(self) -> np.ndarray:
        """T_base←flange of the physical arm (the sensor's pose; equal to latest_pose but for the joint-state model's quantisation
        and lags)."""
        with self._lock:
            return self._Tp.copy()

    @property
    def q_physical(self) -> np.ndarray:
        with self._lock:
            return self._qp.copy()

    @property
    def q(self) -> np.ndarray:
        with self._lock:
            return self._q.copy()

    def latest_pose(self) -> np.ndarray:
        with self._lock:
            return self._T.copy()

    def latest_joint_state(self):
        with self._lock:
            return self._t_last_ns, (list(self.arm.joint_names), self._q.copy())

    def joint_states_age_s(self) -> float:
        return 0.0

    def snapshot(self) -> PoseSnapshot:
        with self._lock:
            T, q, t = self._T.copy(), self._q.copy(), self._t_last_ns
        names = list(self.arm.joint_names)
        rec = {"pose_source": "sim", "base_frame": BASE, "flange_frame": FLANGE, "joint_names": names,
               "joint_positions_rad": [float(v) for v in q], "stamp_ns": t}
        return PoseSnapshot(T=T, t_ns=_now_ns(self.arm.clock), record=rec,
                            arrays={"joint_positions_rad": q.copy(), "joint_names": np.array(names)}, raw=(names, q))

    def history(self, since_ns: int) -> list:
        self.set_q(self._js.cmd)                            # 정지 중에도 '지금' 표본 하나 (재구성의 끝 보간; 같은 명령)
        with self._lock:
            h = list(self._hist)
        i = next((j for j, s_ in enumerate(h) if s_.t_rx_ns >= int(since_ns)), len(h))
        # 가상 팔은 움직일 때만 표본을 남긴다 — 실기 /joint_states 는 정지 중에도 계속 온다 (g3_25/27: 녹화 시작 전 표본 1–2 개, 100 Hz).
        # since 직전 표본 하나를 함께: 그 뒤 첫 이동 표본까지 팔은 그 자세로 서 있었다 (움직이기 전 프레임도 자세를 갖는다)
        return h[max(0, i - 1):]

    def robot_samples(self, samples) -> list:
        return pose_samples_to_robot_samples(samples)

    def sample_arrays(self, samples) -> dict:
        return pose_samples_to_arrays(samples)

    def pose_at(self, t_s: float):
        """이력의 시각 t_s 자세 (ports.pose_at_in). 가상 팔은 자세가 바뀔 때마다 (set_q) 표본을 남기므로 마지막 표본 뒤로는 지금 자세 그대로다."""
        with self._lock:
            if float(t_s) * 1e9 >= self._t_last_ns:
                return self._T.copy()
            return pose_at_in(self._hist, t_s)


# --------------------------------------------------------------------------------------
# 정지 (실행 범위 — ports.StopLatch; 래치 없음, 사용자 2026-10-10)
# --------------------------------------------------------------------------------------
class SimLatch(StopLatch):
    def __init__(self, clock=None):
        self.clock = clock if clock is not None else get_clock()
        self._evt = threading.Event()
        self.reason = ""
        self.last: Optional[LatchRecord] = None
        self.run_open = True
        self.history: list = []
        self._listeners: list = []
        self._lock = threading.Lock()

    @property
    def is_set(self) -> bool:
        return self._evt.is_set()

    def add_listener(self, fn) -> None:
        self._listeners.append(fn)

    def fire(self, reason: str, source: str = "", t_trigger_ns: Optional[int] = None) -> LatchRecord:
        with self._lock:
            if self._evt.is_set() and self.last is not None:
                return self.last
            rec = LatchRecord(reason=str(reason), source=str(source), t_trigger_ns=int(t_trigger_ns or _now_ns(self.clock)))
            self.history.append(rec)
            if not self.run_open:                        # 실행이 없다: 기록만 (보낼 정지 명령이 없다)
                rec.steps.append(("no_run", "no run in progress: no stop command sent"))
                rec.t_done_ns = _now_ns(self.clock)
                return rec
            rec.steps.append(("halt playback", "ok"))
            rec.t_done_ns = _now_ns(self.clock)
            self.reason, self.last = str(reason), rec
            self._evt.set()
        for fn in list(self._listeners):
            try:
                fn("stop", rec)
            except Exception:  # noqa: BLE001
                pass
        return rec

    def wait(self, timeout_s: float) -> bool:
        return self.clock.wait(self._evt, timeout_s)

    def wait_fired(self, timeout_s: float):
        return self.last if self.clock.wait(self._evt, timeout_s) else None

    def begin_run(self) -> None:
        with self._lock:
            self._evt.clear()
            self.reason, self.last, self.run_open = "", None, True

    def end_run(self) -> None:
        with self._lock:
            self._evt.clear()
            self.reason, self.run_open = "", False


# --------------------------------------------------------------------------------------
# 이동 백엔드
# --------------------------------------------------------------------------------------
class SimBackend(MotionBackend):
    kind = "sim_jtc"

    def __init__(self, arm: "SimArmPlatform", cfg: MotionConfig):
        self.arm = arm
        self.cfg = cfg
        self._busy = threading.Lock()
        self._abort = threading.Event()
        self._prepared: Optional[dict] = None
        self.last_plan: dict = {}
        self.n_moves = 0

    @property
    def latch(self) -> StopLatch:
        return self.arm.latch

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def abort(self) -> None:
        self._abort.set()

    # ---- 계획 ----
    def _v_a(self, v_mps: float, a_mps2: float) -> tuple:
        c = self.arm.cfg
        return (min(float(v_mps), float(self.cfg.v_eff_max_mps), c.max_tcp_speed_mps), min(float(a_mps2), c.max_tcp_accel_mps2))

    def _plan_linear(self, T_target: np.ndarray, v_mps: float, a_mps2: float) -> dict:
        arm, c = self.arm, self.arm.cfg
        q0 = arm.pose.q
        fk = arm.fk()
        T0 = fk(q0)
        T_target = np.asarray(T_target, dtype=float)
        try:
            Qd, _fr = track_straight_line(fk, q0, T0, T_target, c.cartesian_step_m, lower=arm.lower, upper=arm.upper,
                                          max_joint_step_rad=c.max_joint_step_rad)
        except ValueError as e:
            raise MotionError(str(e)) from e
        hit = arm.path_collision(Qd)
        if hit:
            raise MotionError(f"straight line collides with the scene: {hit} - refusing")
        v, a = self._v_a(v_mps, a_mps2)
        L = float(np.linalg.norm(T_target[:3, 3] - T0[:3, 3]))
        rot = math.radians(_rot_deg(T0[:3, :3], T_target[:3, :3]))
        rt = retime_straight_path(Qd, L, rot, v, a, rotation_radius_m=c.rotation_radius_m,
                                  max_joint_velocity=c.max_joint_velocity_rad_s, max_joint_accel=c.max_joint_accel_rad_s2,
                                  max_joint_step_rad=c.max_joint_step_rad)
        return {"rt": rt, "q0": q0, "T_target": T_target, "kind": "cartesian",
                "script": f"sim_jtc cartesian -> {Qd.shape[0]} pts, L {L * 1e3:.1f} mm, T {rt.duration:.2f} s",
                "last_plan": {"kind": "cartesian", "n_points": int(Qd.shape[0]), "path_length_m": L, "length_m": L,
                              "rotation_deg": math.degrees(rot), "duration_s": rt.duration, "v_cruise_mps": rt.v_cruise,
                              "time_scale": rt.time_scale}}

    JOINT_PATH_STEP_RAD = 0.01           # dense joint-path step (as the MoveIt adapter's)

    def path_timing_limits(self) -> dict:
        """The MoveIt adapter's path_timing_limits: what the retime of a joint path is bound by."""
        c = self.arm.cfg
        return {"joint_velocity_rad_s": float(c.max_joint_velocity_rad_s), "joint_path_step_rad": self.JOINT_PATH_STEP_RAD,
                "tcp_speed_mps": min(float(self.cfg.v_eff_max_mps), float(c.max_tcp_speed_mps)),
                "tcp_accel_mps2": float(c.max_tcp_accel_mps2), "rotation_radius_m": float(c.rotation_radius_m)}

    def _plan_joint(self, q_goal, v_mps: float, a_mps2: float, *, allow_ompl: bool = True) -> dict:
        arm, c = self.arm, self.arm.cfg
        q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
        if np.any(q_goal < arm.lower - 1e-9) or np.any(q_goal > arm.upper + 1e-9):
            raise MotionError(f"joint goal outside the URDF limits: {np.round(np.degrees(q_goal), 2).tolist()} deg - refusing")
        if arm.config_collision(q_goal):
            raise MotionError("joint goal collides with the scene - refusing")
        q0 = arm.pose.q
        blocked = None
        if allow_ompl:
            W, how = arm.plan_joint_path(q0, q_goal)
        else:                                            # the straight line only: blocked = a ranking estimate (never executable)
            W, how = np.array([q0, q_goal]), "joint-space straight line"
            blocked = arm.path_collision(densify_joint_path(W, self.JOINT_PATH_STEP_RAD))
        Wd = densify_joint_path(W, self.JOINT_PATH_STEP_RAD)
        v, a = self._v_a(v_mps, a_mps2)
        rt = retime_joint_path(Wd, arm.fk(), v, a, rotation_radius_m=c.rotation_radius_m,
                               max_joint_velocity=c.max_joint_velocity_rad_s, max_joint_accel=c.max_joint_accel_rad_s2)
        # MoveIt 어댑터와 같은 계획 요약 (확인 문구: 경로 종류, 플랜지 병진·회전 따로, 가장 많이 도는 관절)
        st = joint_path_stats(rt.positions, arm.fk(), arm.joint_names)
        straight = W.shape[0] == 2
        return {"rt": rt, "q0": q0, "T_target": arm.fk()(q_goal), "kind": "joint", "q_goal": q_goal,
                "script": f"sim_jtc joint ({how}) -> {Wd.shape[0]} pts, flange {st['flange_translation_m'] * 1e3:.0f} mm / "
                          f"{st['flange_rotation_deg']:.0f} deg, T {rt.duration:.1f} s",
                "last_plan": {"kind": "joint", "route": "straight" if straight else "OMPL",
                              "route_text": how if straight else f"sampled via-point path ({W.shape[0]} waypoints)",
                              "n_waypoints": int(W.shape[0]), "how": how, "n_points": int(Wd.shape[0]), "path_length_m": rt.S,
                              "duration_s": rt.duration, "q_goal_deg": np.degrees(q_goal).tolist(), **st,
                              "straight_blocked_reason": blocked, "estimate_only": blocked is not None}}

    def check_linear(self, T_target, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> dict:
        try:
            p = self._plan_linear(T_target, v_mps, a_mps2)
            return {"ok": True, "reason": "", "plan": dict(p["last_plan"])}
        except MotionError as e:
            return {"ok": False, "reason": str(e), "plan": None}

    def prepare_linear(self, T_target, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> Optional[dict]:
        try:
            p = self._plan_linear(T_target, v_mps, a_mps2)
        except MotionError:
            self._prepared = None
            return None
        p["key"] = (np.asarray(T_target, dtype=float).copy(), float(v_mps), float(a_mps2))
        self._prepared = p
        return dict(p["last_plan"])

    def check_joint(self, q_goal, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None, allow_ompl: bool = True) -> dict:
        self.n_checks = getattr(self, "n_checks", 0) + 1
        try:
            p = self._plan_joint(q_goal, v_mps, a_mps2, allow_ompl=allow_ompl)
            p["key"] = (np.asarray(q_goal, dtype=float).copy(), float(v_mps), float(a_mps2))
            p["timeout_s"] = timeout_s
            if p["last_plan"]["estimate_only"]:
                return {"ok": False, "reason": f"straight joint line blocked: {p['last_plan']['straight_blocked_reason']}",
                        "plan": dict(p["last_plan"]), "_checked_plan": None}
            self._prepared = p
            return {"ok": True, "reason": "", "plan": dict(p["last_plan"]), "_checked_plan": p}
        except MotionError as e:
            return {"ok": False, "reason": str(e), "plan": None, "_checked_plan": None}

    def move_checked_joint(self, check: dict, q_goal, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> MotionResult:
        """The MoveIt adapter's move_checked_joint: exactly the checked plan, once - same goal / speed / timeout, the start unchanged,
        the path still collision-free in the current scene right before playing it; else MotionError (nothing played)."""
        p = (check or {}).get("_checked_plan")
        if not (check or {}).get("ok") or p is None or p.get("checked_consumed"):
            raise MotionError("the checked joint plan is unavailable or already used - nothing sent")
        kg, kv, ka = p["key"]
        if not (np.array_equal(kg, np.asarray(q_goal, dtype=float)) and kv == float(v_mps) and ka == float(a_mps2)
                and p.get("timeout_s") == timeout_s):
            raise MotionError("the goal / speed / timeout differ from the checked joint plan - nothing sent")
        p["checked_consumed"] = True
        self._prepared = None
        if float(np.max(np.abs(self.arm.pose.q - p["q0"]))) > 1e-6:
            raise MotionError("checked joint plan start changed - plan and confirm again (nothing sent)")
        hit = self.arm.path_collision(p["rt"].positions)
        if hit:
            raise MotionError(f"the checked joint trajectory is no longer collision-free: {hit} - nothing sent")
        self.executed_checks = getattr(self, "executed_checks", []) + [check]
        return self._run(p, v_mps, a_mps2, note="checked joint trajectory sent unchanged")

    def _take(self, key_goal, v_mps, a_mps2) -> Optional[dict]:
        p, self._prepared = self._prepared, None
        if p is None:
            return None
        kg, kv, ka = p["key"]
        if not (np.allclose(kg, key_goal, rtol=0.0, atol=1e-12) and kv == float(v_mps) and ka == float(a_mps2)):
            return None
        if float(np.max(np.abs(self.arm.pose.q - p["q0"]))) > 1e-6:
            return None
        return p

    def move_linear(self, T_target, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> MotionResult:
        T_target = np.asarray(T_target, dtype=float)
        p = self._take(T_target, v_mps, a_mps2)
        if p is None:
            p = self._plan_linear(T_target, v_mps, a_mps2)
            used = "planned"
        else:
            used = "prepared plan used"
        return self._run(p, v_mps, a_mps2, note=used)

    def move_joint(self, q_goal, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> MotionResult:
        q_goal = np.asarray(q_goal, dtype=float)
        p = self._take(q_goal, v_mps, a_mps2)
        if p is None:
            p = self._plan_joint(q_goal, v_mps, a_mps2)
        return self._run(p, v_mps, a_mps2, note="joint")

    # ---- 실행 (시계로 재생: 8 ms 틱) ----
    def _run(self, p: dict, v_mps: float, a_mps2: float, note: str = "") -> MotionResult:
        arm = self.arm
        clock = arm.clock
        res = MotionResult(script=p["script"], policy="cap", default_speed=1.0)
        if not self._busy.acquire(blocking=False):
            raise MotionError("another motion is in progress")
        try:
            if arm.latch.is_set:
                raise MotionError(f"this run was stopped ({arm.latch.reason}): no further motion in this run")
            self._abort.clear()
            rt = p["rt"]
            self.last_plan = dict(p["last_plan"])
            res.v_cmd = float(min(v_mps, self.cfg.v_eff_max_mps)) * 1e3
            res.a_cmd = float(a_mps2)
            res.v_eff_expected_mps = float(rt.v_cruise)
            res.monitor_events.append(note)
            T, Q = np.asarray(rt.times, dtype=float), np.asarray(rt.positions, dtype=float)
            res.t_send_ns = res.t_ack_ns = res.t_start_ns = _now_ns(clock)
            res.started = True
            self.n_moves += 1
            if clock.virtual and arm.cfg.time_scale != 1.0:
                raise MotionError("the virtual clock replays at time_scale 1 only (sim time is already free)")
            t0_ns = clock.monotonic_ns()
            period = 1.0 / arm.cfg.rate_hz
            ticks: list = []
            n_settle = 0
            for fn in list(arm.trajectory_listeners):
                fn(t0_ns, T, Q, period)
            while True:
                if arm.latch.is_set or self._abort.is_set():
                    res.aborted = True
                    res.reason = f"stopped: {arm.latch.reason}" if arm.latch.is_set else "aborted"
                    break
                now_ns = clock.monotonic_ns()
                ticks.append(now_ns)
                t = playback_t(t0_ns, now_ns, arm.cfg.time_scale)
                if t >= T[-1]:
                    arm.pose.set_q(Q[-1])
                    # the joint-state model after the command's end: the reference echo converges, the physical arm catches up
                    # (the JTC waits for its goal on the reported joints) — nothing to wait for with the default model
                    while not arm.pose.settled() and n_settle < int(5.0 * arm.cfg.rate_hz):     # [R] cap 5 s
                        clock.sleep(period)
                        ticks.append(clock.monotonic_ns())
                        arm.pose.set_q(Q[-1])
                        n_settle += 1
                    break
                arm.pose.set_q(playback_q(T, Q, t))
                clock.sleep(period)
            res.t_stop_ns = _now_ns(clock)
            res.elapsed_s = (clock.monotonic_ns() - t0_ns) * 1e-9
            arm.move_log.append({"t0_ns": t0_ns, "ticks": ticks, "T_end_s": float(T[-1]), "elapsed_s": res.elapsed_s,
                                 "aborted": bool(res.aborted), "period_s": period, "settle_s": n_settle * period})
            if not res.aborted:
                Tm = arm.pose.latest_pose()
                dp = float(np.linalg.norm(Tm[:3, 3] - p["T_target"][:3, 3]))
                dr = _rot_deg(Tm[:3, :3], p["T_target"][:3, :3])
                res.monitor_events.append(f"final measured pose error {dp * 1e3:.3f} mm / {dr:.4f} deg")
                if dp > arm.cfg.goal_pose_tol_m or dr > arm.cfg.goal_pose_tol_deg:
                    res.reason = f"trajectory finished but the measured pose is {dp * 1e3:.2f} mm / {dr:.3f} deg from the target"
                else:
                    res.finished = True
                    res.reason = "trajectory succeeded"
            return res
        finally:
            self._busy.release()


# --------------------------------------------------------------------------------------
# 게이트
# --------------------------------------------------------------------------------------
@dataclass
class SimInitRecord:
    default_speed: float = 1.0
    limits: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"backend": "sim_jtc", "default_speed": self.default_speed, "limits": dict(self.limits)}


class SimGates(PlatformGates):
    name = "sim_jtc"

    def __init__(self, arm: "SimArmPlatform"):
        self.arm = arm

    def motion_config(self, settings, **overrides) -> MotionConfig:
        v = self.arm.cfg.max_tcp_speed_mps
        kw = dict(v_eff_max_mps=v, a_eff_max_mps2=self.arm.cfg.max_tcp_accel_mps2)
        kw.update(overrides)
        return MotionConfig(**kw)

    def session_init(self, platform, cfg, log):
        return SimInitRecord(limits={"max_tcp_speed_mps": self.arm.cfg.max_tcp_speed_mps})

    def status(self, platform) -> dict:
        return {"adapter": "sim_jtc", "run_stopped": platform.latch.is_set, "stop_reason": platform.latch.reason,
                "flange_pose_m": platform.pose.latest_pose()[:3, 3].tolist()}

    def default_speed(self, platform) -> float:
        return 1.0

    def effective_speed(self, platform, cfg, v_target: float) -> float:
        return float(min(float(v_target), cfg.v_eff_max_mps, self.arm.cfg.max_tcp_speed_mps))

    def frame_offset_record(self, platform, snap: PoseSnapshot) -> dict:
        d = {"note": "sim_jtc: targets in the simulated base frame; no command-frame offset", "drift_checked": False}
        d.update({k: v for k, v in snap.record.items() if k in ("joint_names", "joint_positions_rad", "stamp_ns")})
        return d


# --------------------------------------------------------------------------------------
# 플랫폼
# --------------------------------------------------------------------------------------
class SimArmPlatform(RobotPlatform):
    """가상 팔 RobotPlatform (MoveIt 어댑터와 같은 포트·메서드: chain/fk/ik_solutions/ik_seeds/reach_candidates/set_scene_boxes/
    remove_scene_boxes/state_collision/check_weld_plan, backend.move_linear/move_joint/check_*/prepare_linear)."""

    adapter = "sim_jtc"
    pose_source = "sim"

    def __init__(self, settings=None, calib=None, *, cfg: Optional[SimArmConfig] = None, q0=FOLDED_Q, urdf_path: str = URDF):
        self.settings = settings
        self.calib = calib
        self.cfg = cfg or SimArmConfig.from_settings(settings)
        self.clock = get_clock()                  # 만들 때의 시계 (기본 REAL; sim_cell 은 가상 시계를 먼저 설치한다)
        self.trajectory_listeners: list = []      # fn(t0_ns, T, Q, period_s) — 재생 시작 (선계산)
        self.move_log: list = []                  # 이동마다 {t0_ns, ticks, T_end_s, elapsed_s, aborted} (시간 불변식 검사)
        self.urdf_path = urdf_path
        txt = open(urdf_path, "r", encoding="utf-8").read()
        joints = parse_urdf_joints(io.StringIO(txt))
        self._chain = chain_from_joints(joints, BASE, FLANGE)
        self._fk = fast_fk(self._chain)
        self.joint_names = list(self._chain.joint_names)
        self.lower = np.array([j.lower for j in self._chain.movable_joints])
        self.upper = np.array([j.upper for j in self._chain.movable_joints])
        tch = chain_from_joints(joints, FLANGE, TORCH)
        self.T_flange_torch = _fixed_chain(tch)
        self.boxes: dict = {}
        self._pose = SimPoseSource(self)
        self._pose.place(q0)
        self._latch = SimLatch(self.clock)
        self._gates = SimGates(self)
        self._backend: Optional[SimBackend] = None
        self._rng = np.random.default_rng(1)

    # ---- 포트 ----
    @property
    def pose(self) -> SimPoseSource:
        return self._pose

    @property
    def backend(self) -> Optional[SimBackend]:
        return self._backend

    @property
    def latch(self) -> SimLatch:
        return self._latch

    @property
    def gates(self) -> SimGates:
        return self._gates

    @property
    def busy(self) -> bool:
        return bool(self._backend is not None and self._backend.busy)

    def arm(self, cfg: MotionConfig):
        if self._backend is None:
            self._backend = SimBackend(self, cfg)
        else:
            self._backend.cfg = cfg
        return self._backend

    def close(self) -> None:
        pass

    # ---- 기구학 ----
    def chain(self):
        return self._chain

    def fk(self):
        return self._fk

    def set_joints(self, q) -> None:
        """시험 준비: 관절을 바로 놓는다 (이동 없이)."""
        self._pose.place(np.asarray(q, dtype=float))

    def _chain_q(self, q) -> Optional[np.ndarray]:
        if q is None:
            return None
        if isinstance(q, dict):
            try:
                return np.array([float(q[n]) for n in self.joint_names])
            except KeyError:
                return None
        q = np.asarray(q, dtype=float).ravel()
        return q if q.size == len(self.joint_names) else None

    def ik_seeds(self, q_now=None, n_random: int = 32) -> list:
        qc = self._chain_q(q_now)
        out = [[float(v) for v in qc]] if qc is not None else []
        lo = np.maximum(self.lower, -math.pi)
        hi = np.minimum(self.upper, math.pi)
        rng = np.random.default_rng(20261007)
        for _ in range(int(n_random)):
            out.append([float(v) for v in lo + (hi - lo) * rng.random(lo.size)])
        return out

    def _ik(self, T_goal, seed, iters: int = 150) -> Optional[np.ndarray]:
        q = np.asarray(seed, dtype=float).copy()
        fk = self._fk
        for _ in range(iters):
            T = fk(q)
            e = _pose_error(T, T_goal)
            if np.linalg.norm(e[:3]) <= 1e-7 and np.linalg.norm(e[3:]) <= 1e-7:
                if np.all(q >= self.lower - 1e-9) and np.all(q <= self.upper + 1e-9):
                    return q
                return None
            J = _jacobian(fk, q, T)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            n = float(np.linalg.norm(dq))
            if n > 0.3:
                dq *= 0.3 / n
            q = q + dq
        return None

    def ik_solutions(self, T_base_flange, seeds, *, avoid_collisions: bool = True, dedupe_rad: float = 1e-3,
                     timeout_s: float = 1.0) -> list:
        out: list = []
        for s in seeds:
            q = self._ik(np.asarray(T_base_flange, dtype=float), s)
            if q is None or (avoid_collisions and self.config_collision(q)):
                continue
            if all(float(np.max(np.abs(q - o))) > dedupe_rad for o in out):
                out.append(q)
        return out

    def reach_candidates(self, T_goal, follow=(), *, q_now=None, avoid_collisions: bool = True, ik_timeout_s: float = 0.1,
                         sigma_radius_m: float = 0.2) -> list:
        """MoveIt 어댑터 reach_candidates 와 같은 형식."""
        qc = self._chain_q(q_now)
        sols = self.ik_solutions(T_goal, self.ik_seeds(q_now), avoid_collisions=avoid_collisions)
        lo, hi = self.lower, self.upper
        variants: list = []
        for q in sols:
            # 같은 플랜지 자세의 관절 표현 (q_j ± 2π, 한계 안) 중 지금 관절에 가장 가까운 것도 후보로 (IK 는 표현을 고르지 않는다 — 2026-10-08
            # 실기: 손목1 +180° 를 받아 232.5° 회전 48.6 s, 손목2 +290° (= −70°) 로 진행 +1 이 멀어 보여 −1 (센서 뒤집기) 을 골랐다)
            for v in ([q, nearest_turn_equivalent(q, qc, lo, hi)] if qc is not None else [q]):
                v = np.asarray(v, dtype=float)
                if all(float(np.max(np.abs(v - o))) > 1e-3 for o in variants):
                    variants.append(v)
        return [{"names": list(self.joint_names), **self._follow(q, follow, sigma_radius_m, qc)} for q in variants]

    def _follow(self, q, follow, sigma_radius_m: float, q_ref=None) -> dict:
        rec = follow_candidate(self._chain, self._fk, q, follow, step_m=self.cfg.cartesian_step_m, lower=self.lower, upper=self.upper,
                               max_joint_step_rad=self.cfg.max_joint_step_rad, sigma_radius_m=sigma_radius_m, q_ref=q_ref)
        rec.pop("q_end", None)
        return rec

    def follow_from(self, q, follow, *, sigma_radius_m: float = 0.2) -> dict:
        """MoveIt 어댑터 follow_from 과 같은 형식 (IK 없이 그 관절에서 추종)."""
        qc = self._chain_q(q)
        if qc is None:
            raise ValueError("follow_from: joint values do not cover the chain")
        return {"names": list(self.joint_names), **self._follow(qc, follow, sigma_radius_m)}

    # ---- 충돌 장면 ----
    def set_scene_boxes(self, boxes: list) -> bool:
        for b in boxes:
            self.boxes[str(b["id"])] = {"size": np.asarray(b["size"], dtype=float), "T": np.asarray(b["T"], dtype=float)}
        return True

    def remove_scene_boxes(self, ids) -> bool:
        for i in ids:
            self.boxes.pop(str(i), None)
        return True

    def _arm_points(self, q) -> np.ndarray:
        """링크 선분 표본점 (관절 원점 사이 3 cm 간격; base 링크 제외) + 툴 몸체 점 (캘리브 body_points_eoat, 없으면 키포인트)."""
        return self._arm_points_labelled(q)[0]

    def _arm_points_labelled(self, q) -> tuple:
        """_arm_points + 점마다 링크 이름 (관절 k 와 k+1 사이 선분 = 관절 k 의 자식 링크, 툴 몸체 = 'eoat_base_link') — MoveIt 접촉 보고처럼
        '어느 링크가 어느 물체 안' 을 말하려고."""
        P = _joint_origins(self._chain, q)
        links = [j.child for j in self._chain.movable_joints]
        pts, lab = [], []
        for i, (a_, b_) in enumerate(zip(P[1:-1], P[2:])):
            n = max(1, int(math.ceil(float(np.linalg.norm(b_ - a_)) / 0.03)))
            for k in range(n + 1):
                pts.append(a_ + (b_ - a_) * (k / n))
                lab.append(links[min(i, len(links) - 1)])
        Tf = self._fk(np.asarray(q, dtype=float))
        if self.calib is not None:
            from scan_core.path_gen import body_points_base
            B = getattr(self.calib, "body_points_eoat", None)
            if B is not None:
                tb = list(body_points_base((Tf @ self.calib.T_flange_eoat)[None], B)[0])
            else:
                tb = [(Tf @ Tk)[:3, 3] for Tk in (self.calib.T_flange_eoat, self.calib.T_flange_opt, self.T_flange_torch)]
            pts.extend(tb)
            lab.extend(["eoat_base_link"] * len(tb))
        return np.asarray(pts, dtype=float), lab

    def config_contacts(self, q) -> list:
        """[{link, object, depth_m, kind 'world'}] — 장면 상자 안에 든 링크마다 (깊이 = 상자 면까지 가장 깊은 점의 거리)."""
        if not self.boxes:
            return []
        P, lab = self._arm_points_labelled(q)
        out = []
        for bid, b in self.boxes.items():
            Tb = b["T"]
            loc = (P - Tb[:3, 3]) @ Tb[:3, :3]
            depth = np.min(0.5 * b["size"][None, :] - np.abs(loc), axis=1)
            inside = depth >= 0.0
            for link in dict.fromkeys(l_ for l_, k in zip(lab, inside) if k):
                d = max(float(depth[i]) for i, l_ in enumerate(lab) if l_ == link and inside[i])
                out.append({"link": link, "object": bid, "depth_m": d, "kind": "world"})
        return out

    def config_collision(self, q) -> Optional[str]:
        if not self.boxes:
            return None
        P, lab = self._arm_points_labelled(q)
        for bid, b in self.boxes.items():
            Tb = b["T"]
            loc = (P - Tb[:3, 3]) @ Tb[:3, :3]
            inside = np.all(np.abs(loc) <= 0.5 * b["size"][None, :], axis=1)
            if np.any(inside):
                return f"{lab[int(np.argmax(inside))]}-{bid}"
        return None

    def path_collision(self, Q, stride_rad: float = 0.05) -> Optional[str]:
        if not self.boxes:
            return None
        last = None
        for i, q in enumerate(np.asarray(Q, dtype=float)):
            if last is not None and i < len(Q) - 1 and float(np.max(np.abs(q - last))) < stride_rad:
                continue
            last = q
            hit = self.config_collision(q)
            if hit:
                return f"point {i}/{len(Q) - 1} invalid (contacts ['{hit}'])"
        return None

    def state_collision(self) -> Optional[str]:
        return self.config_collision(self._pose.q)

    def state_contacts(self) -> dict:
        """MoveIt 어댑터 state_contacts 와 같은 형식: {valid, contacts [{link, object, depth_m, kind}], why}."""
        c = self.config_contacts(self._pose.q)
        return {"valid": not c, "contacts": c, "why": "contact" if c else ""}

    def arm_reach_m(self) -> float:
        """MoveIt 어댑터 arm_reach_m 과 같은 규칙: 체인 상한 (관절 원점 거리 합) + 플랜지에서 툴 (EOAT·광학·토치) 의 가장 먼 점."""
        ext = [0.0]
        if self.calib is not None:
            for Tk in (self.calib.T_flange_eoat, self.calib.T_flange_opt, self.T_flange_torch):
                ext.append(float(np.linalg.norm(np.asarray(Tk, dtype=float)[:3, 3])))
        return float(self._chain.reach_bound_m()) + max(ext)

    def fixed_bodies(self) -> list:
        """MoveIt 어댑터 fixed_bodies 와 같은 것 (같은 URDF 의 base 고정 링크 충돌 형상) — 가상 팔의 충돌 모형은 base 링크를 빼지만 세계의
        바닥 범위는 실기와 같게."""
        from scan_core.fk_urdf import fixed_link_boxes
        return fixed_link_boxes(self.urdf_path, BASE)

    def plan_joint_path(self, q0, q1) -> tuple:
        """관절 직선 (상태 검사), 막히면 중간 자세 하나를 거치는 표본 계획 (관절 거리 짧은 순) — OMPL 대용."""
        line = densify_joint_path([q0, q1], 0.01)
        blocked = self.path_collision(line)
        if blocked is None:
            return np.array([q0, q1]), "joint-space straight line"
        lo = np.maximum(self.lower, -math.pi)
        hi = np.minimum(self.upper, math.pi)
        best = None
        for _ in range(400):
            m = lo + (hi - lo) * self._rng.random(lo.size)
            d = float(np.sum(np.abs(m - q0)) + np.sum(np.abs(q1 - m)))
            if best is not None and d >= best[0]:
                continue
            if self.config_collision(m):
                continue
            if self.path_collision(densify_joint_path([q0, m], 0.01)) or self.path_collision(densify_joint_path([m, q1], 0.01)):
                continue
            best = (d, m)
        if best is None:
            raise MotionError(f"no joint-space path (the straight line is blocked: {blocked}) - refusing")
        return np.array([q0, best[1], q1]), f"sampled via-point path (the joint-space straight line is blocked: {blocked})"

    # ---- 용접 경로 도달성 ----
    def weld_tool_body(self):
        """The weld torch's tool for the weld-pose clearance (scan_core.weld_clearance): this arm's URDF collision geometry rigid with
        its torch TCP (TORCH - the lab EOAT; the weld_torch TCP name when the URDF has it). Read once."""
        tb = getattr(self, "_weld_tool_body", None)
        if tb is None:
            from scan_core.weld_clearance import tool_body_from_urdf
            from scan_core.weld_torch import weld_torch_config
            wt = weld_torch_config(self.settings.params if self.settings is not None else None)
            text = open(self.urdf_path, "r", encoding="utf-8").read()
            if wt is not None and f'name="{wt.tcp_frame}"' in text:
                tb = tool_body_from_urdf(text, wt.tcp_frame)
            elif wt is not None:                                          # a TCP its URDF does not carry: the tool on the flange
                tb = tool_body_from_urdf(text, wt.tcp_frame, root_frame=wt.flange_frame, T_root_tcp=wt.flange_from_tcp.matrix())
            else:
                tb = tool_body_from_urdf(text, TORCH)
            self._weld_tool_body = tb
        return tb

    def weld_reach_model(self):
        """The arm's necessary reach condition (scan_core.weld_reach) from this arm's URDF chain and its weld torch TCP (the URDF torch
        TCP, else the config's flange <- TCP)."""
        from scan_core.weld_reach import arm_reach_from_chain
        from scan_core.weld_torch import weld_torch_config
        wt = weld_torch_config(self.settings.params if self.settings is not None else None)
        if wt is None:
            return None
        if wt.tcp_frame == TORCH:                                           # as check_weld_plan's lookup (flange <- TCP)
            T = self.T_flange_torch
        elif wt.flange_frame == FLANGE:
            T = wt.flange_from_tcp.matrix()
        else:
            return None
        return arm_reach_from_chain(self._chain, T, source=f"virtual arm URDF {BASE} -> {FLANGE}")

    def weld_line_step_m(self) -> float:
        """The sample step of the delivered weld lines D14 checks: this arm's Cartesian step (as the MoveIt adapter's)."""
        return float(self.cfg.cartesian_step_m)

    def check_weld_plan(self, plan, *, approach_m: float = 0.03, only=None) -> list:
        """The MoveIt adapter's check_weld_plan in form (no motion) on the DELIVERED lines (scan_core.weld_line.weld_line: one straight
        TCP line at one orientation per pass - kiro/pass_<k>.yaml's; the cap also its weave edge lines): IK of the start pose
        approach_m back along the torch axis and dense IK along the line (joint limits, joint jump). Tool = scan_core.weld_torch.
        reach_tool (the plan's torch with its stick-out, the plan's TCP roll; on this arm only, otherwise ValueError). Flange <- the weld
        torch TCP like TF from this arm's URDF (torch_tcp - the lab EOAT); a TCP name the URDF lacks (KIRO's right_manipulator_ee_point)
        from the config. Workpiece (2026-10-13): the virtual arm has no MoveIt - the tool's clearance (weld_tool_body: URDF collision
        geometry) from the specimen before the pass (scan_core.weld_scene) on the same line samples (scan_core.weld_orient.
        pass_clearance), per part role (the gas nozzle and contact tip 0 mm, every other part the required plan.clearance) - no
        arm-vs-world or self-collision check here (the rows say so). only = the passes to check. Refusals carry {constraint, links,
        detail} (weld_orient.refusal_of)."""
        from scan_core.weld_line import weld_line
        from scan_core.weld_torch import plan_weld_torch, reach_tool, torch_with_nozzle_end, weld_torch_config
        wt_cfg = weld_torch_config(self.settings.params if self.settings is not None else None)
        if wt_cfg is not None and wt_cfg.on_this_arm:                                   # the stick-out's reference: the nozzle end of
            wt_cfg = torch_with_nozzle_end(wt_cfg, self.weld_tool_body)                 # this arm's torch, from its URDF (2026-10-13)
        wtorch = getattr(plan, "torch", None) or plan_weld_torch(plan, wt_cfg)

        def lookup(target, source):
            if target == FLANGE and source == TORCH:                                    # this arm's URDF torch TCP (as TF)
                return self.T_flange_torch.copy()
            if wtorch is not None and target == FLANGE == wtorch.flange_frame and source == wtorch.tcp_frame:
                return wtorch.flange_from_tcp.matrix()
            raise LookupError(f"the virtual arm has no transform {target} <- {source}")
        roll = float(getattr(plan, "torch_roll_deg", 0.0) or 0.0)
        _T_ft, tool = reach_tool(wtorch, lookup, FLANGE, roll_deg=roll)
        T_cf = np.linalg.inv(np.asarray(lookup(FLANGE, wtorch.tcp_frame), dtype=float))
        from scan_core.weld_orient import pass_clearance
        r_mm = float((getattr(plan, "clearance", None) or {}).get("required_mm", 0.0) or 0.0)
        specimen = getattr(plan, "spec_design", None) is not None and getattr(plan, "levels_mm", None) is not None
        from scan_core.weld_clearance import part_link, tool_roles
        body = tool_roles(self.weld_tool_body()) if specimen else None
        out = []
        seed = self._pose.q
        for wp in plan.passes:
            if only is not None and wp.k not in set(only):
                continue
            ln = weld_line(plan, wp, wtorch, roll)
            P = np.asarray(wp.dense_pos_m, dtype=float)
            dt = np.diff(np.asarray(wp.dense_t_s, dtype=float))
            v = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(dt, 1e-9)
            vmax = float(np.max(v)) * 1e3 if v.size else float("nan")
            r = {"k": wp.k, "kind": wp.kind, "max_tcp_speed_mm_s": vmax,
                 "speed_over_limit": bool(vmax > self.cfg.max_tcp_speed_mps * 1e3 + 1e-6), "reachable": True, "fraction": 1.0,
                 "max_joint_step_rad": 0.0, "max_step_joint": "", "max_step_s_mm": float("nan"), "min_limit_margin_rad": float("inf"),
                 "lines": [], "reason": "", "tool": tool, "refusal": None,
                 "robot_model": "virtual arm: IK, joint limits and joint jumps along the line; no arm-vs-world or self-collision check",
                 "line": {"tcp_start_m": ln.tcp_start, "tcp_goal_m": ln.tcp_goal, "R_base_tcp": ln.R_tcp, "roll_deg": roll,
                          "length_mm": ln.length_m * 1e3, "edges": [n_ for n_, _o in ln.offsets()[1:]]}}
            if not specimen:
                r["workpiece"] = "not checked: the plan has no specimen model (a bare pose list)"
            c = (pass_clearance(plan, wp, body, wtorch, roll, required_mm=r_mm, known=(getattr(plan, "d14_exact", None) or {}).get(wp.k))
                 if specimen else None)          # (known: the orientation search's record of this line - reused when the same evaluation)
            if c is not None:
                br = c["by_role"]
                r["workpiece"] = ("geometric clearance per part role on the delivered line's samples (the virtual arm has no MoveIt): "
                                  + ", ".join(f"{k_} {v_['clearance_mm']:.2f} mm (required {v_['required_mm']:.2f})" for k_, v_ in br.items())
                                  + f" (the specimen before this pass; {body.describe()})")
                r["clearance_mm"] = float(c["clearance_mm"])
                r["clearance_by_role"] = {k_: float(v_["clearance_mm"]) for k_, v_ in br.items()}
                r["line"].update(n_samples=c.get("n_samples"), step_mm=(c.get("checked_line") or {}).get("step_mm"))
            if c is not None and c["excess_mm"] < 0.0:
                lim = c["by_role"][c["limiting_role"]]
                pt = lim.get("point_tcp_mm") or [float("nan")] * 3
                link = part_link(dict(body.roles())[c["limiting_role"]], lim["point_tcp_mm"]) if lim.get("point_tcp_mm") is not None else ""
                why = (f"the tool's {c['limiting_role']} parts come within {lim['clearance_mm']:.2f} mm of the specimen ({lim['solid']}, "
                       f"{lim['line']} line, s {float(lim.get('line_s_mm') or float('nan')):.0f} mm; tool point ({pt[0]:+.1f}, {pt[1]:+.1f}, "
                       f"{pt[2]:+.1f}) mm in the TCP frame) - required {lim['required_mm']:.2f} mm")
                r.update(reachable=False, fraction=0.0, reason=why,
                         refusal={"constraint": "clearance (exact model)", "part": c["limiting_role"], "links": [link] if link else [],
                                  "detail": why})
                out.append(r)
                continue
            for name, off in ln.offsets():
                T_tcp = [ln.T_tcp("start"), ln.T_tcp("goal")]
                for T in T_tcp:
                    T[:3, 3] += np.asarray(off, dtype=float)
                A = T_tcp[0].copy()
                A[:3, 3] -= ln.axis * float(approach_m)
                follow = [T @ T_cf for T in T_tcp]
                sols = self.ik_solutions(A @ T_cf, [seed] + self.ik_seeds(None, 8), avoid_collisions=False)
                tr = None
                if sols:                                                         # first the solution nearest the current joints
                    q0 = min(sols, key=lambda s_: float(np.sum(np.abs(s_ - seed))))
                    tr = track_polyline(self._chain, q0, follow, self.cfg.cartesian_step_m, self.lower, self.upper, self.cfg.max_joint_step_rad)
                cand_fails = []
                if tr is None or not tr["ok"]:                                   # else by reach (the MoveIt adapter's rule)
                    from scan_l0.acquisition import rank_reach_candidates
                    cands = self.reach_candidates(A @ T_cf, follow, q_now=seed, avoid_collisions=False)
                    cand_fails = sorted({str(c_.get("reason", ""))[:160] for c_ in cands if not c_.get("ok")})
                    ranked = rank_reach_candidates(cands)
                    if ranked:
                        q0 = np.asarray(ranked[0]["q"], dtype=float)
                        tr = track_polyline(self._chain, q0, follow, self.cfg.cartesian_step_m, self.lower, self.upper,
                                            self.cfg.max_joint_step_rad)
                if tr is None:
                    why = f"no IK for the {name} line start: no IK solution within the joint limits (the arm's reach / joint limits)"
                    r.update(reachable=False, fraction=0.0, reason=why, refusal={"constraint": "IK", "links": [], "detail": why})
                    r["lines"].append({"line": name, "reachable": False, "reason": why})
                    break
                seed = q0
                Q = tr["Q"]
                dQ = np.abs(np.diff(Q, axis=0)) if Q.shape[0] > 1 else np.zeros((0, len(self.joint_names)))
                step = float(dQ.max()) if dQ.size else 0.0
                margin = float(min(np.min(Q - self.lower), np.min(self.upper - Q)))
                ok = bool(tr["ok"])
                leg = "the descent onto the line start" if int(tr.get("i_fail", -1)) <= 0 else "the line"
                why = "" if ok else (f"{name} line, {leg}: {tr['reason']}"
                                     + (f" (follow candidates: {'; '.join(cand_fails[:2])})" if cand_fails else ""))
                r["lines"].append({"line": name, "reachable": ok, "fraction": 1.0 if ok else 0.0, "max_joint_step_rad": step,
                                   "max_step_joint": "", "min_limit_margin_rad": margin, "n_waypoints": len(T_tcp), "reason": why})
                r["max_joint_step_rad"] = max(r["max_joint_step_rad"], step)
                r["min_limit_margin_rad"] = min(r["min_limit_margin_rad"], margin)
                if not ok:
                    low_ = tr["reason"].lower()
                    r.update(reachable=False, fraction=0.0, reason=why,
                             refusal={"constraint": "joint limit" if "urdf limit" in low_ else "joint jump" if "jump" in low_ else "IK",
                                      "links": [], "detail": why})
                    break
            out.append(r)
        return out


def _fixed_chain(ch) -> np.ndarray:
    T = np.eye(4)
    for j in ch.joints:
        T = T @ j.origin()
    return T


def _joint_origins(chain, q) -> list:
    """체인 관절 원점 (base) 목록 — 첫 원점 = base, 끝 = 플랜지."""
    q = np.asarray(q, dtype=float).ravel()
    T = np.eye(4)
    out = [T[:3, 3].copy()]
    k = 0
    for j in chain.joints:
        T = T @ j.origin()
        if j.movable:
            out.append(T[:3, 3].copy())
            a = np.asarray(j.axis, dtype=float)
            a = a / np.linalg.norm(a)
            K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
            R = np.eye(3) + math.sin(q[k]) * K + (1.0 - math.cos(q[k])) * (K @ K)
            M = np.eye(4)
            M[:3, :3] = R
            T = T @ M
            k += 1
    out.append(T[:3, 3].copy())
    return out


def open_sim_session(settings, calib, *, q0=FOLDED_Q, cfg: Optional[SimArmConfig] = None):
    """가상 팔 ScanSession (센서는 AutoTeach 가 연다: release_sensor=True 로 넘기면 끝에서 닫는다)."""
    from scan_l0.session import ScanSession
    arm = SimArmPlatform(settings, calib, cfg=cfg, q0=q0)
    return ScanSession(robot=arm, sensor=None, owned=False), arm
