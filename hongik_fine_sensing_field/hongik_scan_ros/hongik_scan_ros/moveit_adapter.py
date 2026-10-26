"""MoveIt 어댑터 'moveit_jtc' — KIRO 호환 MoveIt/ros2_control 경로 (설계 docs/kiro_integration_design_2026-10-01.md §3, §7).

KIRO 는 팔 제조사 드라이버·제어박스를 쓰지 않고 MoveIt 으로만 움직인다: move_group (그룹 right_manipulator, 계획 프레임
robot_base_link) + JointTrajectoryController FollowJointTrajectory 액션 + /joint_states + robot_state_publisher TF. 이 어댑터는
그 통신 패턴만 쓴다 (KIRO URDF 를 쓰지 않는다; 연구실에서는 RB20 MoveIt 가짜 하드웨어로 같은 패턴을 시험).

    포트            구현
    PoseSource      RosPoseSource   : /joint_states 수신 (호스트 monotonic 수신시각 = 프로파일 t_rx 와 같은 시간축) → 그 메시지의 관절
                                      값 (이름으로 맞춤) 을 실행 중인 시스템의 URDF (robot_description 토픽) 로 푼 FK = base_frame ←
                                      flange_frame (pose_fk; 2026-10-13 — TF 를 샘플마다 조회하지 않는다: KIRO 의 TF 는 ~0.8 s 늦게
                                      온다) → 이력. 재구성은 이 샘플들 사이를 시각 보간한다 (scan_core.reconstruct). TF 는 정적 변환만
                                      (한 번 읽고 정적·URDF 대조 확인 뒤 캐시) + 세션마다 한 번 FK ↔ TF 모형 대조 (tf_cross_check).
    MotionBackend   MoveItJtcBackend: 1) /compute_cartesian_path (eef = flange_frame, max_step = cartesian_step_m, 시작 상태 = 실측
                                      관절) 는 **가부 판정 전용**: fraction < min_fraction(0.999) 이면 거부 (부분 직선 실행 금지),
                                      충돌·관절 점프·IK 분기 검사는 move_group 이 한다. 응답 궤적은 TOTG 재표본(예: 50 mm 에 7 점)이라
                                      그대로 보간하면 직선이 아니므로 실행에 쓰지 않는다.
                                      2) 실행 궤적 = 같은 직선을 robot_description 체인으로 조밀하게 다시 푼 관절 경로
                                      (trajectory.track_straight_line, cartesian_step_m 간격, 관절 한계·스텝 한계). 시작 FK 가 TF 자세와
                                      다르면 거부, move_group 의 점들이 이 조밀 경로 위에 있는지 대조 (path_on_moveit_tol_rad 초과 =
                                      다른 IK 분기 → 거부), 끝점을 move_group /compute_fk 로 목표와 대조 (goal_pose_tol 초과 → 거부).
                                      3) 사다리꼴 시간 매개화 (등속 구간, 관절 한계면 균등 감속; trajectory.retime_straight_path) →
                                      FollowJointTrajectory 실행·진행 표시 → 도착 확인 → 측정 샘플 (이력) 포함 MotionResult.
    StopLatch       RosStopLatch    : 액션 취소의 정지 (이 패키지의 유일한 정지; 각 단계 실패해도 다음 단계): 0) 지금 지령에서 감속 정지
                                      궤적 — 되면 1) 생략; 1) 우리 목표 취소 + JTC cancel_goal(전부: 0 id/0 stamp) 2)
                                      /trajectory_execution_event "stop". 정지 표시는 **실행 범위** (ports.StopLatch): 래치 토픽·해제 없음.
                                      로봇 제어기의 설정·컨트롤러 전환은 하지 않는다. 실행 중 감시 정지는 없다 (사용자 2026-10-11): JTC 가
                                      목표를 거부·중단하면 그 이동의 실패로 보고할 뿐이다.
    PlatformGates   MoveItGates     : 준비 확인 (MoveItPlatform.readiness: move_group 서비스·JTC
                                      액션·/joint_states·TF base←flange, 우리 이동 없음), 한계 선언 확인 (없으면 시작 거부).

이동 계획의 검사 (보내기 전): 우리 포락선 (I1/I4: 모든 직선 이동 사전 거부 — 판 평면·지지면·클램프·레이저 기울기·transit; scan_l0.planning,
scan_core.path_gen) 과 MoveIt 의 모형 기반 충돌 검사 (avoid_collisions, 플래닝 씬 — 셀 세계의 'hongik/' 물체 포함). 속도는 플랫폼 설정의
**선언** 한계 (moveit_jtc.limits: TCP 속도·가속, 관절 속도·가속; 없으면 시작 거부) 로 시간 매개화 자체를 제한 (명령 속도 = 실효 속도) +
이동마다 측정 속도 기록. 계획은 요청 뒤에 발행된 /joint_states 에서 시작하고, 보내기 직전에도 계획 뒤에 발행된 상태를 받는다 (없으면 보내지
않는다). 실행 중에는 JTC 결과만 본다 (감시 정지 없음; 정지는 액션 취소뿐). 명령 프레임 = 측정 프레임 (같은 URDF: 자세 = robot_description
FK): 세션 시작에 그 FK 를 TF 와 같은 스탬프에서 대조해 (> path_check_tol) 다르면 준비 안 됨. 로봇 제어기와는 이야기하지 않는다 — 플랫폼의
JTC 와 move_group 만 쓴다 (KIRO 주 시스템이 첫 명령을 보내면 그것으로 실행).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import yaml

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.msg import CollisionObject, Constraints, JointConstraint, MoveItErrorCodes, PlanningScene
from moveit_msgs.msg import AllowedCollisionEntry, PlanningSceneComponents
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath, GetMotionPlan, GetPlanningScene, GetPositionFK, GetPositionIK,
                             GetStateValidity)
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from geometry_msgs.msg import Point
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros

from scan_core.frames import unit
from scan_core.types import Calib
from scan_l0.motion_types import PlatformUnavailable, SpeedMeasurement
from scan_l0.ports import (LatchRecord, MotionBackend, MotionConfig, MotionError, MotionResult, PlatformGates,
                           PlatformNotReady, PoseSample, PoseSnapshot, PoseSource, RobotPlatform, StopLatch,
                           pose_at_in, pose_samples_to_arrays, pose_samples_to_robot_samples)

from .calib_tf import calib_from_tf
from .platform_config import MoveItJtcConfig
from .pose_fk import FRAME_ROT_TOL_DEG, METHOD as POSE_METHOD, FkPoseModel, pose_difference, tf_dynamic_edges
from .ros_util import (RosContext, StampToMonotonic, call_service, matrix_to_pose, pose_to_matrix, stamp_to_ns, transform_to_matrix,
                       wait_future)
from scan_core.trajectory import (densify_joint_path, duration_parts, fast_fk, joint_path_stats, on_path_distance, retime_joint_path,
                                  retime_straight_path, track_straight_line)

__all__ = ["MoveItPlatform", "MoveItGates", "MoveItJtcBackend", "RosPoseSource", "RosStopLatch", "MoveItInitRecord"]


def _now_ns() -> int:
    return time.monotonic_ns()


def _rot_deg(Ra, Rb) -> float:
    R = np.asarray(Ra, dtype=float).T @ np.asarray(Rb, dtype=float)
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return math.degrees(math.acos(c))


JOINT_PATH_STEP_RAD = 0.01        # dense joint-path step of every joint-space plan (densify_joint_path; the duration bound's h)


def _fill_robot_state(rs_msg, measured, override=None) -> None:
    """moveit_msgs/RobotState for a check: the FULL measured joint state (every joint /joint_states carries - KIRO's other arm, a
    head) with the given joints overridden ({name: value}), is_diff = True - MoveIt then keeps the scene's attached bodies (with
    is_diff false robotStateMsgToRobotState clears them) and checks against the real state of the joints we do not move (KIRO field
    2026-10-14). measured = a JointState or None (then only the overridden joints)."""
    state = dict(zip(measured.name, measured.position)) if measured is not None else {}
    state.update({str(n): float(v) for n, v in (override or {}).items()})
    rs_msg.joint_state.name = list(state)
    rs_msg.joint_state.position = [float(v) for v in state.values()]
    rs_msg.is_diff = True


# --------------------------------------------------------------------------------------
# PoseSource
# --------------------------------------------------------------------------------------
class RosPoseSource(PoseSource):
    """/joint_states -> T_base<-flange by FK of each message's joint values (matched by name) with the running system's URDF
    (robot_description topic, transient local; pose_fk) — nothing per sample from TF (user 2026-10-13: KIRO's TF reaches nodes ~0.8 s
    late while /joint_states is fresh). Sample time = header.stamp moved onto the host monotonic axis (ros_util.StampToMonotonic: the
    delivery-latency lower bound — samples stay evenly spaced when the reception bunches under load); with use_sim_time the host receive
    time. A message without the chain joints (another arm's publisher, a partial message) or before the URDF arrived gives no sample
    (counted in drop_reasons; readiness names what is missing — no TF fallback).
    Static transforms from TF only through static_transform (read once, verified static and against the URDF, cached)."""

    kind = "ros_tf"            # the adapter's pose-source name in the platform YAML (scan_l0.ports.ADAPTERS); the method is METHOD
    method = POSE_METHOD

    def __init__(self, ctx: RosContext, mj: MoveItJtcConfig, cb_group):
        self.ctx = ctx
        self.mj = mj
        node = ctx.node
        # TF is read for static transforms only (static_transform) and once per session for the model cross-check (tf_cross_check)
        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.listener = tf2_ros.TransformListener(self.buffer, node, spin_thread=False)
        self._lock = threading.Lock()
        self._hist: deque = deque()
        self._last_js: Optional[tuple] = None             # (t_rx_ns, JointState) — every message
        self._last_sample_js: Optional[tuple] = None      # (t_rx_ns, JointState) — the message of the newest pose sample
        self._stamp_map = None if bool(getattr(mj, "use_sim_time", False)) else StampToMonotonic()
        self._robot_description: Optional[str] = None
        self._model: Optional[FkPoseModel] = None
        self._model_error: Optional[str] = None
        self._rd_event = threading.Event()
        self._model_listeners: list = []
        self._static: dict = {}                           # (target, source) -> 4x4, read once
        self._static_records: dict = {}
        self.n_js = 0
        self.n_resolved = 0
        self.n_dropped = 0
        self.drop_reasons: dict = {}
        qos_latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._rd_sub = node.create_subscription(String, mj.robot_description_topic, self._on_rd, qos_latched, callback_group=cb_group)
        qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
        self._sub = node.create_subscription(JointState, mj.joint_states_topic, self._on_js, qos, callback_group=cb_group)

    # ---- the model (robot_description) ----
    def _on_rd(self, msg) -> None:
        text = str(msg.data)
        mj = self.mj
        try:
            model, err = FkPoseModel(text, mj.base_frame, mj.flange_frame, f"topic {mj.robot_description_topic}"), None
        except Exception as e:  # noqa: BLE001 - any parse / chain error is the readiness reason
            model, err = None, f"robot_description on {mj.robot_description_topic}: chain {mj.base_frame} -> {mj.flange_frame}: {e}"
        with self._lock:
            self._robot_description = text
            self._model, self._model_error = model, err
            self._static.clear()                          # a new robot model: static transforms are read and verified again
            self._static_records.clear()
        self._rd_event.set()
        for fn in list(self._model_listeners):
            fn(model)

    def add_model_listener(self, fn: Callable) -> None:
        """fn(model or None) after every robot_description (a new URDF replaces the model)."""
        self._model_listeners.append(fn)

    @property
    def robot_description(self) -> Optional[str]:
        return self._robot_description

    def wait_robot_description(self, timeout_s: float) -> Optional[str]:
        if self._robot_description is None:
            self._rd_event.wait(max(0.0, timeout_s))
        return self._robot_description

    def model(self, timeout_s: float = 0.0) -> FkPoseModel:
        """The FK model (robot_description chain base_frame -> flange_frame). MotionError with the reason when there is none."""
        self.wait_robot_description(timeout_s)
        m = self._model
        if m is None:
            raise MotionError(self._model_error or f"no robot_description on {self.mj.robot_description_topic}: cannot compute "
                              f"{self.mj.base_frame} <- {self.mj.flange_frame} from /joint_states")
        return m

    # ---- 수신 ----
    def _on_js(self, msg: JointState) -> None:
        t = _now_ns()
        st = stamp_to_ns(msg.header.stamp)
        if self._stamp_map is not None:
            t = self._stamp_map.map(st, t)
        model = self._model
        if model is None:
            T, why = None, "no robot model (robot_description)"
        else:
            T, why = model.pose(msg.name, msg.position)
        with self._lock:
            self._last_js = (t, msg)
            self.n_js += 1
            if T is None:
                self.n_dropped += 1
                self.drop_reasons[why] = self.drop_reasons.get(why, 0) + 1
                return
            self._hist.append(PoseSample(t_rx_ns=t, T_base_flange=T, joints=tuple(float(v) for v in msg.position),
                                         joint_names=tuple(msg.name), stamp_ns=st))
            self._last_sample_js = (t, msg)
            self.n_resolved += 1
            t_min = _now_ns() - int(self.mj.history_s * 1e9)
            while self._hist and self._hist[0].t_rx_ns < t_min:
                self._hist.popleft()

    def fk_problem(self) -> Optional[str]:
        """None when the newest /joint_states message gave a pose (FK with robot_description), else what is missing (no age threshold:
        a plan starts from a message published after its request, MoveItJtcBackend)."""
        mj = self.mj
        with self._lock:
            model, err, last, last_s = self._model, self._model_error, self._last_js, self._last_sample_js
            drops = dict(self.drop_reasons)
        if model is None:
            return err or (f"no robot_description on {mj.robot_description_topic} ({mj.base_frame} <- {mj.flange_frame} is FK of "
                           "/joint_states with that URDF)")
        if last is None:
            return f"no /joint_states on {mj.joint_states_topic}"
        if last_s is None or last_s[0] < last[0]:
            miss = model.missing_joints(last[1].name)
            if miss:
                return (f"/joint_states on {mj.joint_states_topic} lack the chain joints {miss} of {mj.base_frame} -> "
                        f"{mj.flange_frame} (message names {list(last[1].name)})")
            return (f"the newest /joint_states on {mj.joint_states_topic} gave no FK pose (messages without usable chain joint values: "
                    f"{drops})")
        return None

    # ---- 포트 ----
    def static_transform(self, target: str, source: str, timeout_s: float = 0.5) -> np.ndarray:
        """TF target <- source (4x4) of a STATIC transform: read once, verified, cached — nothing per sample. Verified = every TF edge
        between the two frames is a /tf_static transform (a moving frame is FK of /joint_states, never read here) and, where the
        robot_description holds the same path with fixed joints only, the TF value equals the URDF's (path_check_tol_m /
        FRAME_ROT_TOL_DEG — the TF tree and our URDF are the same robot model). MotionError otherwise or when TF lacks the frames."""
        key = (str(target), str(source))
        with self._lock:
            hit = self._static.get(key)
        if hit is not None:
            return hit.copy()
        t_end = time.monotonic() + max(0.0, timeout_s)
        last = None
        while True:
            try:
                tf = self.buffer.lookup_transform(key[0], key[1], Time())
                break
            except Exception as e:  # noqa: BLE001
                last = e
            if time.monotonic() >= t_end:
                raise MotionError(f"TF {key[0]} <- {key[1]} not available: {last}")
            time.sleep(0.01)
        try:
            frames = yaml.safe_load(self.buffer.all_frames_as_yaml()) or {}
            path, moving = tf_dynamic_edges(frames, key[0], key[1])
        except (ValueError, yaml.YAMLError) as e:
            raise MotionError(f"TF {key[0]} <- {key[1]}: {e}") from None
        if moving:
            raise MotionError(f"TF {key[0]} <- {key[1]} is not static: {', '.join(f'{p} -> {c}' for c, p in moving)} is published on /tf "
                              "(a moving frame is computed by FK of /joint_states, not read from TF)")
        T = transform_to_matrix(tf)
        rec: dict = {"source": "TF /tf_static (read once)", "tf_path": [f"{p} -> {c}" for c, p in path]}
        model = self._model
        if model is not None:
            Tu, mov = model.fixed_transform(key[0], key[1])
            if mov:
                raise MotionError(f"TF {key[0]} <- {key[1]} is static but robot_description moves it (joints {mov}): the TF tree and "
                                  f"{self.mj.robot_description_topic} are not the same robot model")
            if Tu is not None:
                dp, dr = pose_difference(Tu, T)
                rec.update(urdf_checked=True, urdf_position_diff_m=dp, urdf_rotation_diff_deg=dr)
                if dp > self.mj.path_check_tol_m or dr > FRAME_ROT_TOL_DEG:
                    raise MotionError(f"TF {key[0]} <- {key[1]} differs from robot_description by {dp * 1e3:.3f} mm / {dr:.4f} deg "
                                      f"(> {self.mj.path_check_tol_m * 1e3:.2f} mm / {FRAME_ROT_TOL_DEG} deg): the TF tree and "
                                      f"{self.mj.robot_description_topic} are not the same robot model")
            else:
                rec["urdf_checked"] = False
        with self._lock:
            self._static[key] = T
            self._static_records[f"{key[0]} <- {key[1]}"] = rec
        return T.copy()

    def tf_cross_check(self) -> dict:
        """One look at the platform's TF base <- flange (per session, not per sample): TF's newest stamp vs the FK sample of the
        /joint_states message with that same header stamp — the URDF we compute with is the one behind the robot's TF. However late
        TF is, the stamp it has already covers is compared; when TF is ahead of our own reception (robot_state_publisher got that message
        first), our stream is waited for up to tf_timeout_s. {"checked": False, "why"} when TF has no base <- flange or no
        sample with that stamp is in the history."""
        mj = self.mj
        try:
            t_tf = self.buffer.get_latest_common_time(mj.base_frame, mj.flange_frame)
            st = int(t_tf.nanoseconds)
        except Exception as e:  # noqa: BLE001
            return {"checked": False, "why": f"TF {mj.base_frame} <- {mj.flange_frame} not available ({e})"}
        t_end = time.monotonic() + mj.tf_timeout_s
        while True:
            with self._lock:
                newest = self._hist[-1].stamp_ns if self._hist else None
                smp = next((s for s in reversed(self._hist) if s.stamp_ns == st), None)
            if smp is not None or (newest is not None and newest > st) or time.monotonic() >= t_end:
                break
            time.sleep(0.005)
        if st == 0 or smp is None:
            return {"checked": False, "why": f"no /joint_states sample with TF {mj.base_frame} <- {mj.flange_frame}'s newest stamp "
                                             f"({st} ns) in the pose history"}
        try:
            T_tf = transform_to_matrix(self.buffer.lookup_transform(mj.base_frame, mj.flange_frame, t_tf))
        except Exception as e:  # noqa: BLE001
            return {"checked": False, "why": f"TF {mj.base_frame} <- {mj.flange_frame} at {st} ns: {e}"}
        dp, dr = pose_difference(smp.T_base_flange, T_tf)
        return {"checked": True, "ok": bool(dp <= mj.path_check_tol_m and dr <= FRAME_ROT_TOL_DEG), "stamp_ns": st,
                "position_diff_m": dp, "rotation_diff_deg": dr, "tol_m": mj.path_check_tol_m, "tol_deg": FRAME_ROT_TOL_DEG,
                "tf_behind_joint_states_s": (int(newest) - st) * 1e-9 if newest else None}

    def record(self) -> dict:
        """Run-log record of the pose source: method, URDF source (+ sha256, chain), counts, static transforms read."""
        m = self._model
        with self._lock:
            d = {"pose_source": self.kind, "method": POSE_METHOD, "joint_states_topic": self.mj.joint_states_topic,
                 "n_joint_states": self.n_js, "n_pose_samples": self.n_resolved, "n_dropped": self.n_dropped,
                 "drop_reasons": dict(self.drop_reasons), "static_transforms": {k: dict(v) for k, v in self._static_records.items()}}
        if m is not None:
            d.update(m.record())
        else:
            d.update(urdf_source=f"topic {self.mj.robot_description_topic}", model_error=self._model_error or "no robot_description")
        return d

    def latest_pose(self) -> np.ndarray:
        with self._lock:
            if self._hist:
                return np.array(self._hist[-1].T_base_flange, copy=True)
        raise MotionError(f"no measured pose: {self.fk_problem() or 'no pose sample yet'}")

    def latest_joint_state(self) -> Optional[tuple]:
        with self._lock:
            return self._last_js

    def joint_states_age_s(self) -> float:
        js = self.latest_joint_state()
        return float("inf") if js is None else (_now_ns() - js[0]) * 1e-9

    def joint_state_since(self, t_ns: int, timeout_s: float) -> Optional[JointState]:
        """시각 t_ns (이력 시간축: 발행 시각, use_sim_time 이면 수신 시각) 이후의 가장 새 /joint_states — 없으면 timeout_s 까지 기다린다
        (밀린 콜백의 옛 메시지는 받지 않는다). None = 그런 메시지가 오지 않았다."""
        t_end = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            js = self.latest_joint_state()
            if js is not None and js[0] >= int(t_ns):
                return js[1]
            if time.monotonic() >= t_end:
                return None
            time.sleep(0.002)

    def snapshot(self) -> PoseSnapshot:
        """Pose + the /joint_states message it was computed from (one message: the joints and the pose belong together)."""
        with self._lock:
            smp = self._hist[-1] if self._hist else None
            js = self._last_sample_js
        if smp is None or js is None:
            raise MotionError(f"no measured pose: {self.fk_problem() or 'no pose sample yet'}")
        rec: dict = {"pose_source": self.kind, "pose_method": POSE_METHOD, "base_frame": self.mj.base_frame,
                     "flange_frame": self.mj.flange_frame}
        rec.update(joint_names=list(js[1].name), joint_positions_rad=[float(v) for v in js[1].position],
                   stamp_ns=stamp_to_ns(js[1].header.stamp))
        arrays = {"joint_positions_rad": np.array(js[1].position, dtype=float), "joint_names": np.array(list(js[1].name))}
        return PoseSnapshot(T=np.array(smp.T_base_flange, copy=True), t_ns=_now_ns(), record=rec, arrays=arrays, raw=js)

    def history(self, since_ns: int) -> list:
        with self._lock:
            return [s for s in self._hist if s.t_rx_ns >= int(since_ns)]

    def robot_samples(self, samples) -> list:
        return pose_samples_to_robot_samples(samples)

    def sample_arrays(self, samples) -> dict:
        return pose_samples_to_arrays(samples)

    def pose_at(self, t_s: float) -> Optional[np.ndarray]:
        """이력 (재구성과 같은 샘플: 발행 시각 축) 의 시각 t_s 자세 (ports.pose_at_in) — 프레임의 촬영 시각으로 부른다. 그 시각이 가장 새
        샘플보다 뒤면 None (부른 쪽이 다음에 다시)."""
        with self._lock:
            return pose_at_in(self._hist, t_s)


# --------------------------------------------------------------------------------------
# StopLatch (§7.1)
# --------------------------------------------------------------------------------------
class RosStopLatch(StopLatch):
    """즉시 정지 + 실행 범위 정지 표시 (ports.StopLatch). 실행이 열려 있을 때의 fire = 정지 순서 (모듈 docstring) + estopped 표시 (이 실행은
    더 보내지 않는다). 실행이 닫혀 있을 때의 fire = 기록만 (우리 이동이 없다 — KIRO 의 JTC 목표를 건드리지 않는다). 래치 토픽·해제
    서비스는 없다: 정지 상태의 권한은 로봇 제어기."""

    def __init__(self, ctx: RosContext, mj: MoveItJtcConfig, cb_group):
        self.ctx = ctx
        self.mj = mj
        node = ctx.node
        self.estopped = threading.Event()                 # 지금 실행이 정지됐다 (begin_run / end_run 이 지운다)
        self.run_open = True
        self.reason = ""
        self.last: Optional[LatchRecord] = None
        self.history: list = []
        self._lock = threading.Lock()
        self._listeners: list = []
        self.busy_getter: Optional[Callable[[], bool]] = None         # 플랫폼: 우리 이동이 진행 중인가 (실행이 닫혀 있어도 그러면 정지)
        self.own_goal_cancel: Optional[Callable[[], None]] = None      # 백엔드가 자기 목표 취소를 건다 (가장 빠른 경로)
        # 백엔드의 감속 정지 (→ (ok, 설명)): 지금 지령 위치·속도에서 감속해 멈추는 궤적을 JTC 에 보낸다. 받아들여지면 취소를 하지 않는다
        self.controlled_stop: Optional[Callable[[], tuple]] = None
        self._cancel_client = node.create_client(CancelGoal, mj.jtc_action.rstrip("/") + "/_action/cancel_goal",
                                                 callback_group=cb_group)
        self._event_pub = node.create_publisher(String, mj.trajectory_execution_event_topic, 10)

    @property
    def is_set(self) -> bool:
        return self.estopped.is_set()

    def add_listener(self, fn: Callable[[str, Optional[LatchRecord]], None]) -> None:
        """fn(event, record): 'stop' (정지 명령을 보냈다), 'idle_request' (실행이 없어 기록만), 'begin_run', 'end_run'."""
        if fn not in self._listeners:
            self._listeners.append(fn)

    def _notify(self, event: str, rec: Optional[LatchRecord]) -> None:
        for fn in list(self._listeners):
            try:
                fn(event, rec)
            except Exception:  # noqa: BLE001 - a listener must never break the stop path
                pass

    def _ours_moving(self) -> bool:
        try:
            return bool(self.busy_getter()) if self.busy_getter is not None else False
        except Exception:  # noqa: BLE001 - unknown: treat as moving (stop)
            return True

    def begin_run(self) -> None:
        with self._lock:
            self.estopped.clear()
            self.reason = ""
            self.last = None
            self.run_open = True
        self._notify("begin_run", None)

    def end_run(self) -> None:
        with self._lock:
            self.estopped.clear()
            self.reason = ""
            self.run_open = False
        self._notify("end_run", None)

    def fire(self, reason: str, source: str = "", t_trigger_ns: Optional[int] = None) -> LatchRecord:
        rec = LatchRecord(reason=reason, source=source, t_trigger_ns=t_trigger_ns or _now_ns())
        with self._lock:
            active = self.run_open or self._ours_moving()
            self.history.append(rec)
            if active:
                self.reason = reason
                self.last = rec
                self.estopped.set()                           # 표시 먼저: 모션 루프가 그 사이 새 목표를 보내지 않게
        if not active:
            rec.steps.append(("no_run", "no run in progress and none of our motions active: no stop command sent"))
            rec.t_done_ns = _now_ns()
            self._notify("idle_request", rec)
            return rec
        # 0. 감속 정지 (2026-10-03 at15): JTC 의 취소·유지는 "지금의 실측 자세" 를 붙드는데 실측은 지령보다 ~0.1 s 늦다 → 지령이 그만큼
        #    뒤로 튀어 빨리 돌던 관절이 순간 역회전 (펜던트 '외부 충돌 의심'). 대신 지금 지령 위치·속도에서 일정 감속으로 멈추는 궤적을
        #    보내 (JTC 가 진행 중 목표를 이것으로 대체) 경로를 따라 매끄럽게 선다. 안 되면 (상태 없음·정지 중·거부) 아래 취소로.
        stopped = False
        if self.controlled_stop is not None:
            try:
                ok, msg = self.controlled_stop()
                rec.steps.append(("controlled_stop", ("ok: " if ok else "not used: ") + str(msg)))
                stopped = bool(ok)
            except Exception as e:  # noqa: BLE001
                rec.steps.append(("controlled_stop", f"error: {e}"))
        if stopped:
            rec.steps.append(("cancel_own_goal", "skipped (replaced by the controlled stop)"))
            rec.steps.append(("cancel_all_jtc_goals", "skipped (a cancel would hold the lagging measured pose)"))
        else:
            self._cancel_goals(rec)
        # 2. MoveIt TrajectoryExecutionManager 가 관리하는 실행 정지
        try:
            self._event_pub.publish(String(data="stop"))
            rec.steps.append(("trajectory_execution_event_stop", "ok"))
        except Exception as e:  # noqa: BLE001
            rec.steps.append(("trajectory_execution_event_stop", f"error: {e}"))
        rec.errors = [f"{s}: {r}" for s, r in rec.steps if str(r).startswith("error")]
        rec.t_done_ns = _now_ns()
        self._notify("stop", rec)
        return rec

    def _cancel_goals(self, rec: LatchRecord) -> None:
        """1. 진행 중인 FollowJointTrajectory 목표 취소 (우리 목표 → 서버의 전부) — 감속 정지를 못 쓸 때."""
        if self.own_goal_cancel is not None:
            try:
                self.own_goal_cancel()
                rec.steps.append(("cancel_own_goal", "ok"))
            except Exception as e:  # noqa: BLE001
                rec.steps.append(("cancel_own_goal", f"error: {e}"))
        try:
            if self._cancel_client.service_is_ready():
                req = CancelGoal.Request()                    # goal_id 0 + stamp 0 = 그 액션 서버의 모든 목표
                fut = self._cancel_client.call_async(req)
                if not wait_future(fut, 0.5) or fut.result() is None:
                    rec.steps.append(("cancel_all_jtc_goals", "error: no response within 0.5 s"))
                elif fut.result().return_code == CancelGoal.Response.ERROR_REJECTED:
                    rec.steps.append(("cancel_all_jtc_goals", "error: cancel rejected by the JTC"))
                else:                                         # NONE / UNKNOWN_GOAL_ID (nothing active) / GOAL_TERMINATED
                    rec.steps.append(("cancel_all_jtc_goals", "ok"))
            else:
                rec.steps.append(("cancel_all_jtc_goals", "error: cancel service not ready"))
        except Exception as e:  # noqa: BLE001
            rec.steps.append(("cancel_all_jtc_goals", f"error: {e}"))

    def wait(self, timeout_s: float) -> bool:
        return self.estopped.wait(max(0.0, float(timeout_s)))

    def wait_fired(self, timeout_s: float):
        return self.last if self.wait(timeout_s) else None


# --------------------------------------------------------------------------------------
# MotionBackend
# --------------------------------------------------------------------------------------
class MoveItJtcBackend(MotionBackend):
    kind = "moveit_jtc"

    def __init__(self, platform: "MoveItPlatform", cfg: MotionConfig):
        self.p = platform
        self.cfg = cfg
        self._action = platform.jtc_client               # 플랫폼이 시작 때 한 번 만든 클라이언트 (DDS 짝 맺기 시간 확보)
        self._cart = platform.cart_client
        self._fk = platform.fk_client
        self._busy = threading.Lock()
        self._abort = threading.Event()
        self._goal_handle = None
        self._gh_lock = threading.Lock()
        self.last_result: Optional[MotionResult] = None
        self.last_plan: dict = {}
        self._prepared: Optional[dict] = None             # prepare_linear 의 계획 (다음 move_linear 한 번)
        self._prepared_joint: Optional[dict] = None       # check_joint 가 통과시킨 관절 계획 (같은 목표의 다음 move_joint 한 번)
        platform.latch.own_goal_cancel = self._cancel_own_goal
        platform.latch.controlled_stop = self._controlled_stop
        self._stop_goal_handle = None

    @property
    def latch(self) -> StopLatch:
        return self.p.latch

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def abort(self) -> None:
        self._abort.set()

    def _controlled_stop(self) -> tuple:
        """액션 취소의 감속 정지: JTC 의 지금 지령 (controller_state.reference 위치 q0·속도 v0, 그 시각 stamp) 에서 일정 감속으로 멈추는 두 점
        궤적 — q(t) = q0 + v0 t − v0 t² / (2T), T = 관절·TCP 감속도 한계 중 느린 쪽 (stop_decel_*), 끝 q0 + v0 T/2 (관절 공간 직선 =
        경로 접선, 정지 거리 수 mm). header.stamp = 그 상태의 시각이라 JTC 는 받는 순간부터 이 궤적을 그 시점에 맞춰 이어 간다 (실측에서
        다시 출발하지 않음 → 뒤로 튀지 않음). JTC 가 받으면 진행 중 목표는 대체 (CANCELED). 반환 (ok, 설명)."""
        p = self.p
        mj = p.mj
        ref = p.controller_reference()
        if ref is None:
            return False, "no controller state received"
        names, q0, v0, stamp, t_rx = ref
        age = (_now_ns() - t_rx) * 1e-9
        if age > 0.1:
            return False, f"controller state {age:.2f} s old"
        vmax = float(np.max(np.abs(v0))) if v0.size else 0.0
        if vmax < 1e-4:
            return False, "not moving"
        T = vmax / max(1e-3, mj.stop_decel_joint_rad_s2)
        v_tcp = float("nan")
        try:                                                   # TCP 속도 (우리 FK) → TCP 감속도 한계
            chain = p.chain()
            idx = [names.index(n) for n in chain.joint_names]
            fk = p.fk()
            dt = 1e-3
            v_tcp = float(np.linalg.norm(fk(q0[idx] + v0[idx] * dt)[:3, 3] - fk(q0[idx])[:3, 3]) / dt)
            T = max(T, v_tcp / max(1e-4, mj.stop_decel_tcp_mps2))
        except Exception:  # noqa: BLE001 - 관절 감속도만으로도 정지는 매끄럽다
            pass
        T = max(T, 0.05)
        q1 = q0 + v0 * (T / 2.0)
        traj = JointTrajectory()
        traj.header.stamp = stamp
        traj.joint_names = list(names)
        for q, v, ts in ((q0, v0, 0.0), (q1, np.zeros_like(v0), T)):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in q]
            pt.velocities = [float(x) for x in v]
            pt.time_from_start.sec, pt.time_from_start.nanosec = duration_parts(ts)
            traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        fut = self._action.send_goal_async(goal)
        if not wait_future(fut, 0.3) or fut.result() is None:
            return False, "stop trajectory: no acceptance within 0.3 s"
        if not fut.result().accepted:
            return False, "stop trajectory rejected by the JTC"
        self._stop_goal_handle = fut.result()
        return True, (f"decelerating to rest in {T:.2f} s (joint {vmax:.3f} rad/s, TCP {v_tcp * 1e3:.1f} mm/s, "
                      f"{vmax * T / 2:.4f} rad / {v_tcp * T / 2 * 1e3:.1f} mm further)")

    def _fresh_joint_state(self, what: str) -> JointState:
        """요청 뒤에 발행된 /joint_states (RosPoseSource.joint_state_since; 기다림 = tf_timeout_s) — 계획은 그 상태에서 시작하고, 보내기
        직전에도 계획 뒤의 상태를 받아야 보낸다. 오지 않으면 MotionError (아무것도 보내지 않는다)."""
        p = self.p
        js = p.pose.joint_state_since(_now_ns(), p.mj.tf_timeout_s)
        if js is None:                                   # the platform, not the move: a candidate search stops here (PlatformUnavailable)
            raise PlatformUnavailable(f"no /joint_states on {p.mj.joint_states_topic} within {p.mj.tf_timeout_s} s {what}: nothing sent")
        return js

    def _cancel_own_goal(self) -> None:
        with self._gh_lock:
            gh = self._goal_handle
        if gh is not None:
            gh.cancel_goal_async()

    # ---- 계획 ----
    def _plan(self, T0: np.ndarray, js: JointState, T_target: np.ndarray) -> tuple:
        mj = self.p.mj
        req = GetCartesianPath.Request()
        req.header.frame_id = mj.base_frame
        req.start_state.joint_state.name = list(js.name)
        req.start_state.joint_state.position = [float(v) for v in js.position]
        req.start_state.is_diff = True
        req.group_name = mj.move_group_name
        req.link_name = mj.flange_frame
        req.waypoints = [matrix_to_pose(T_target)]
        req.max_step = float(mj.cartesian_step_m)
        req.jump_threshold = float(mj.jump_threshold)
        req.avoid_collisions = bool(mj.avoid_collisions)
        resp = call_service(self._cart, req, 10.0, wait_service_s=2.0, resend=False)   # a planning request: sent once
        if resp is None:
            raise MotionError(f"move_group {mj.cartesian_path_service} did not answer (is move_group running?)")
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            raise MotionError(f"compute_cartesian_path failed: MoveIt error code {resp.error_code.val}")
        frac = float(resp.fraction)
        jt = resp.solution.joint_trajectory
        if frac < mj.min_fraction:
            raise MotionError(f"Cartesian path fraction {frac:.4f} < {mj.min_fraction} (IK/collision/jump along the straight "
                              "line): refusing - a partial straight line is never executed")
        if len(jt.points) < 1:
            raise MotionError("Cartesian path is empty: nothing to verify against - refusing")
        return jt, frac

    def _fk_positions(self, names, positions_list) -> list:
        mj = self.p.mj
        out = []
        for q in positions_list:
            req = GetPositionFK.Request()
            req.header.frame_id = mj.base_frame
            req.fk_link_names = [mj.flange_frame]
            req.robot_state.joint_state.name = list(names)
            req.robot_state.joint_state.position = [float(v) for v in q]
            resp = call_service(self._fk, req, 5.0, wait_service_s=2.0)
            if resp is None or resp.error_code.val != MoveItErrorCodes.SUCCESS or not resp.pose_stamped:
                raise MotionError(f"compute_fk failed ({None if resp is None else resp.error_code.val}): cannot verify the "
                                  "straight line - refusing")
            out.append(pose_to_matrix(resp.pose_stamped[0].pose))
        return out

    # ---- 계획 (이동 전, 센서 녹화 전에 미리 할 수 있다) ----
    def _build_plan(self, T_target: np.ndarray, v_mps: float, a_mps2: float, timeout_s: Optional[float], js: JointState,
                    T0: np.ndarray) -> dict:
        """측정 관절 js·자세 T0 에서 T_target 까지의 직선 계획 → JTC 목표 (전송 전 모든 검사 포함). MotionError = 거부."""
        p = self.p
        mj = p.mj
        cfg = self.cfg
        L = float(np.linalg.norm(T_target[:3, 3] - T0[:3, 3]))
        rot = math.radians(_rot_deg(T0[:3, :3], T_target[:3, :3]))
        v = min(float(v_mps), float(cfg.v_eff_max_mps), mj.max_tcp_speed_mps)
        a = min(float(a_mps2), mj.max_tcp_accel_mps2)
        # 1. move_group: 같은 직선의 타당성 게이트 (fraction ≥ min_fraction, 충돌·점프, IK 분기)
        jt, frac = self._plan(T0, js, T_target)
        # 2. 같은 직선을 조밀하게 다시 푼다 (응답 궤적은 TOTG 0.1 s 재표본이라 그대로 보간하면 직선이 아니다; trajectory 모듈 문서)
        chain = p.chain()
        names = list(chain.joint_names)
        if set(names) != set(jt.joint_names):
            raise MotionError(f"base_frame {mj.base_frame!r} -> flange_frame {mj.flange_frame!r} chain joints {names} differ from the "
                              f"move group's joints {list(jt.joint_names)} - refusing")
        js_map = dict(zip(js.name, js.position))
        try:
            q0 = np.array([float(js_map[n]) for n in names])
        except KeyError as e:
            raise MotionError(f"/joint_states has no joint {e}") from e
        fk = p.fk()
        T0_fk = fk(q0)
        d0 = float(np.linalg.norm(T0_fk[:3, 3] - T0[:3, 3]))
        if d0 > mj.path_check_tol_m or _rot_deg(T0_fk[:3, :3], T0[:3, :3]) > FRAME_ROT_TOL_DEG:
            raise MotionError(f"FK of the planning start joints differs from the measured start pose by {d0 * 1e3:.2f} mm: the joint "
                              f"state and the pose are not of the same robot state ({mj.base_frame} <- {mj.flange_frame}) - refusing")
        lo = np.array([j.lower for j in chain.movable_joints])
        hi = np.array([j.upper for j in chain.movable_joints])
        try:
            Qd, _fr = track_straight_line(fk, q0, T0_fk, T_target, mj.cartesian_step_m, lower=lo, upper=hi,
                                          max_joint_step_rad=mj.max_joint_step_rad)
        except ValueError as e:
            raise MotionError(str(e)) from e
        # move_group 의 점들이 이 조밀 경로 위에 있는가 (같은 IK 분기·같은 직선 — 그쪽 충돌 검사가 이 경로에 유효)
        idx = [list(jt.joint_names).index(n) for n in names]
        dev = max(on_path_distance(Qd, np.asarray(pt.positions, dtype=float)[idx]) for pt in jt.points)
        if dev > mj.path_on_moveit_tol_rad:
            raise MotionError(f"the dense straight-line IK leaves move_group's checked path by {dev:.4f} rad "
                              f"(> {mj.path_on_moveit_tol_rad} rad, different IK branch) - refusing")
        # 3. 끝점: 우리 FK = 목표 (구성상), move_group FK 와도 대조 (모델 일치)
        T_end = self._fk_positions(names, [Qd[-1]])[0]
        de = float(np.linalg.norm(T_end[:3, 3] - T_target[:3, 3]))
        re_ = _rot_deg(T_end[:3, :3], T_target[:3, :3])
        if de > mj.goal_pose_tol_m or re_ > mj.goal_pose_tol_deg:
            raise MotionError(f"straight-line end (move_group FK) differs from the target by {de * 1e3:.2f} mm / {re_:.3f} deg "
                              "- refusing")
        Q = Qd
        N = Q.shape[0] - 1
        try:
            rt = retime_straight_path(Q, L, rot, v, a, rotation_radius_m=mj.rotation_radius_m,
                                      max_joint_velocity=mj.max_joint_velocity_rad_s, max_joint_accel=mj.max_joint_accel_rad_s2,
                                      max_joint_step_rad=mj.max_joint_step_rad)
        except ValueError as e:
            raise MotionError(str(e)) from e
        last_plan = {"kind": "cartesian", "n_points": N + 1, "moveit_points": len(jt.points), "fraction": frac, "length_m": L,
                     "rotation_deg": math.degrees(rot), "moveit_path_deviation_rad": dev, "end_error_mm": de * 1e3,
                     "duration_s": rt.duration, "v_cruise_mps": rt.v_cruise, "time_scale": rt.time_scale,
                     "notes": list(rt.notes)}
        script = (f"moveit_jtc cartesian {L * 1e3:.1f} mm rot {math.degrees(rot):.2f} deg -> {N + 1} pts "
                  f"(fraction {frac:.4f}, step {mj.cartesian_step_m * 1e3:.1f} mm) v {rt.v_cruise * 1e3:.1f} mm/s "
                  f"a {rt.a:.3f} m/s^2 T {rt.duration:.2f} s target {np.round(T_target[:3, 3], 4).tolist()}")
        timeout = float(timeout_s) if timeout_s is not None else max(cfg.default_timeout_s,
                                                                      rt.duration * 1.5 + mj.goal_time_margin_s)
        # ---- JTC 목표 (전송은 move_linear) ----
        traj = JointTrajectory()
        traj.joint_names = names
        for i in range(N + 1):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in rt.positions[i]]
            pt.velocities = [float(x) for x in rt.velocities[i]]
            ts = float(rt.times[i])
            pt.time_from_start.sec, pt.time_from_start.nanosec = duration_parts(ts)
            traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance.sec = int(mj.goal_time_margin_s)
        return {"key": (np.array(T_target, dtype=float, copy=True), float(v_mps), float(a_mps2), timeout_s), "names": names,
                "q0": np.asarray(q0, dtype=float).copy(), "rt": rt, "goal": goal, "script": script, "timeout_s": timeout,
                "last_plan": last_plan, "t_planned": time.monotonic()}

    # ---- 관절 공간 이동 (2026-10-07: 임의 초기 자세 — 운송용 접힌 자세 — 에서 획득 자세로) ----
    def _first_invalid(self, names, Q: np.ndarray, stride_rad: float = 0.05, *, joint_state=None) -> Optional[str]:
        """조밀 관절 경로를 stride_rad 간격으로 move_group /check_state_validity (자기·장면 충돌, 관절 한계) — 처음 막힌 곳 설명 또는 None.
        Each state = the full measured joint state (joint_state; default the latest /joint_states) with the arm joints of the path,
        is_diff (_fill_robot_state). The run's stop (latch) is checked at every sample: MotionError('stopped ...'). No answer =
        PlatformUnavailable (the platform, not the path)."""
        p = self.p
        mj = p.mj
        if joint_state is None:
            latest = p.pose.latest_joint_state()
            joint_state = latest[1] if latest is not None else None
        last = None
        for i, q in enumerate(Q):
            if last is not None and i < len(Q) - 1 and float(np.max(np.abs(q - last))) < stride_rad:
                continue
            last = q
            if p.latch.is_set:
                raise MotionError(f"stopped while checking a joint path ({p.latch.reason}) - nothing sent")
            req = GetStateValidity.Request()
            req.group_name = mj.move_group_name
            _fill_robot_state(req.robot_state, joint_state, dict(zip(names, (float(v) for v in q))))
            r = call_service(self.p.validity_client, req, 5.0, wait_service_s=2.0)
            if r is None:
                raise PlatformUnavailable(f"move_group {mj.state_validity_service} did not answer (is move_group running?)")
            if not r.valid:
                pairs = [f"{c.contact_body_1}-{c.contact_body_2}" for c in r.contacts[:3]]
                return f"point {i}/{len(Q) - 1} invalid (contacts {pairs or 'none reported'})"
        return None

    def _ompl_waypoints(self, js: JointState, names, q_goal: np.ndarray) -> np.ndarray:
        mj = self.p.mj
        req = GetMotionPlan.Request()
        r = req.motion_plan_request
        r.group_name = mj.move_group_name
        r.start_state.joint_state.name = list(js.name)
        r.start_state.joint_state.position = [float(v) for v in js.position]
        c = Constraints()
        for n, q in zip(names, q_goal):
            jc = JointConstraint()
            jc.joint_name = str(n)
            jc.position = float(q)
            jc.tolerance_above = 1e-3
            jc.tolerance_below = 1e-3
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        r.goal_constraints = [c]
        r.num_planning_attempts = 4
        r.allowed_planning_time = 5.0
        r.max_velocity_scaling_factor = 0.1
        r.max_acceleration_scaling_factor = 0.1
        resp = call_service(self.p.plan_client, req, 30.0, wait_service_s=2.0, resend=False)
        if resp is None:
            raise MotionError(f"move_group {mj.plan_service} did not answer (is move_group running?)")
        mr = resp.motion_plan_response
        if mr.error_code.val != MoveItErrorCodes.SUCCESS:
            raise MotionError(f"MoveIt joint-space planning failed: error code {mr.error_code.val}")
        jt = mr.trajectory.joint_trajectory
        idx = [list(jt.joint_names).index(n) for n in names]
        W = np.array([[float(pt.positions[i]) for i in idx] for pt in jt.points])
        if W.shape[0] < 2 or float(np.max(np.abs(W[-1] - q_goal))) > 2e-3:
            raise MotionError("MoveIt joint-space plan does not end at the goal - refusing")
        W[-1] = q_goal                                   # 목표 제약 허용 (±1e-3 rad → 플랜지 ~1 mm) 안의 끝점을 정확한 목표로 (도착 검사 1 mm)
        return W

    def _build_joint_plan(self, q_goal: np.ndarray, v_mps: float, a_mps2: float, timeout_s: Optional[float], js: JointState,
                          T0: np.ndarray, *, allow_ompl: bool = True) -> dict:
        """측정 관절 → q_goal 관절 공간 이동 계획 (전송 전 모든 검사): 관절 공간 직선 보간이 move_group 상태 검사 (충돌·한계) 를 통과하면
        그것 (가장 예측 가능한 경로), 막히면 MoveIt OMPL 계획 (같은 상태 검사로 다시 확인). TCP 속도 v·가속 a 와 관절 한계로 재시간.
        allow_ompl False (a candidate search ranks the straight lines first): a blocked straight line is still retimed and returned
        with last_plan estimate_only True + straight_blocked_reason - a ranking estimate, never executable (check_joint says not ok).
        The run's stop is checked between OMPL attempts (MotionError 'stopped ...')."""
        p = self.p
        mj = p.mj
        cfg = self.cfg
        chain = p.chain()
        names = list(chain.joint_names)
        js_map = dict(zip(js.name, js.position))
        try:
            q0 = np.array([float(js_map[n]) for n in names])
        except KeyError as e:
            raise MotionError(f"/joint_states has no joint {e}") from e
        q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
        if q_goal.size != q0.size or not np.all(np.isfinite(q_goal)):
            raise MotionError(f"joint goal must have {q0.size} finite values")
        lo = np.array([j.lower for j in chain.movable_joints])
        hi = np.array([j.upper for j in chain.movable_joints])
        if np.any(q_goal < lo - 1e-9) or np.any(q_goal > hi + 1e-9):
            raise MotionError(f"joint goal outside the URDF limits: {'; '.join(chain.limit_problems(q_goal))} (goal "
                              f"{np.round(np.degrees(q_goal), 2).tolist()} deg) - refusing")
        fk = p.fk()
        T0_fk = fk(q0)
        d0 = float(np.linalg.norm(T0_fk[:3, 3] - T0[:3, 3]))
        if d0 > mj.path_check_tol_m or _rot_deg(T0_fk[:3, :3], T0[:3, :3]) > FRAME_ROT_TOL_DEG:
            raise MotionError(f"FK of the planning start joints differs from the measured start pose by {d0 * 1e3:.2f} mm - refusing")
        Wd = densify_joint_path([q0, q_goal], JOINT_PATH_STEP_RAD)
        blocked = self._first_invalid(names, Wd, joint_state=js)
        how = route_text = "joint-space straight line"
        route, n_wp = "straight", 2
        if blocked is not None and allow_ompl:
            # OMPL 은 확률적이고 자기 검사 간격이 이 조밀 검사보다 성기다 — 조밀 검사에 걸리면 몇 번 다시 계획한다 (통과한 것만)
            bad = None
            for attempt in range(1, 4):
                if p.latch.is_set:
                    raise MotionError(f"stopped while planning a joint path ({p.latch.reason}) - nothing sent")
                W = self._ompl_waypoints(js, names, q_goal)
                W[0] = q0
                Wd = densify_joint_path(W, JOINT_PATH_STEP_RAD)
                bad = self._first_invalid(names, Wd, joint_state=js)
                if bad is None:
                    break
            if bad is not None:
                raise MotionError(f"MoveIt joint-space plan has an invalid state in {attempt} attempts: {bad} - refusing")
            route, n_wp = "OMPL", int(W.shape[0])
            route_text = f"MoveIt OMPL ({W.shape[0]} waypoints{'' if attempt == 1 else f', attempt {attempt}'})"   # 확인 문구용 짧은 설명
            how = (f"MoveIt OMPL ({W.shape[0]} waypoints{'' if attempt == 1 else f', attempt {attempt}'}; the joint-space straight line "
                   f"is blocked: {blocked})")
        if np.any(Wd < lo[None, :] - 1e-9) or np.any(Wd > hi[None, :] + 1e-9):
            raise MotionError("joint-space path leaves the URDF limits - refusing")
        v = min(float(v_mps), float(cfg.v_eff_max_mps), mj.max_tcp_speed_mps)
        a = min(float(a_mps2), mj.max_tcp_accel_mps2)
        try:
            rt = retime_joint_path(Wd, fk, v, a, rotation_radius_m=mj.rotation_radius_m, max_joint_velocity=mj.max_joint_velocity_rad_s,
                                   max_joint_accel=mj.max_joint_accel_rad_s2)
        except ValueError as e:
            raise MotionError(str(e)) from e
        N = Wd.shape[0] - 1
        # 확인·기록용: 계획한 플랜지 병진과 회전을 따로, 가장 많이 도는 관절 (path_length_m = 재시간 경로 변수 S — 병진과 회전 × 반경 중
        # 큰 것의 누적이라 이동 거리로 읽으면 안 된다; 2026-10-08 g1_b)
        st = joint_path_stats(rt.positions, fk, names)
        last_plan = {"kind": "joint", "route": route, "route_text": route_text, "n_waypoints": n_wp, "how": how, "n_points": N + 1,
                     "path_length_m": rt.S, "duration_s": rt.duration, "v_cruise_mps": rt.v_cruise, "time_scale": rt.time_scale,
                     "q_goal_deg": np.degrees(q_goal).tolist(), "notes": list(rt.notes), **st,
                     "straight_blocked_reason": blocked, "estimate_only": bool(blocked is not None and not allow_ompl)}
        script = (f"moveit_jtc joint ({how}) -> {N + 1} pts, flange {st['flange_translation_m'] * 1e3:.0f} mm / "
                  f"{st['flange_rotation_deg']:.0f} deg, T {rt.duration:.1f} s, goal {np.round(np.degrees(q_goal), 1).tolist()} deg")
        timeout = float(timeout_s) if timeout_s is not None else max(cfg.default_timeout_s, rt.duration * 1.5 + mj.goal_time_margin_s)
        traj = JointTrajectory()
        traj.joint_names = names
        for i in range(N + 1):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in rt.positions[i]]
            pt.velocities = [float(x) for x in rt.velocities[i]]
            pt.time_from_start.sec, pt.time_from_start.nanosec = duration_parts(float(rt.times[i]))
            traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance.sec = int(mj.goal_time_margin_s)
        return {"key": None, "names": names, "q0": q0.copy(), "rt": rt, "goal": goal, "script": script, "timeout_s": timeout,
                "last_plan": last_plan, "t_planned": time.monotonic()}

    def path_timing_limits(self) -> dict:
        """What this backend's retime of a joint path is bound by (scan_l0.joint_selection.TIMING_LIMIT_KEYS): the platform's declared
        joint velocity and TCP speed / acceleration limits, the rotation radius of the path variable and the dense joint-path step -
        the candidate search's exact duration lower bound."""
        mj = self.p.mj
        return {"joint_velocity_rad_s": float(mj.max_joint_velocity_rad_s), "joint_path_step_rad": JOINT_PATH_STEP_RAD,
                "tcp_speed_mps": min(float(self.cfg.v_eff_max_mps), float(mj.max_tcp_speed_mps)),
                "tcp_accel_mps2": float(mj.max_tcp_accel_mps2), "rotation_radius_m": float(mj.rotation_radius_m)}

    def check_joint(self, q_goal, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None,
                    allow_ompl: bool = True) -> dict:
        """move_joint 와 같은 계획·검사를 보내지 않고 → {ok, reason, plan, _checked_plan}. _checked_plan = this candidate's own checked
        trajectory (move_checked_joint sends exactly it; every candidate keeps its handle). allow_ompl False: a blocked straight line is
        an estimate only (ok False, plan.estimate_only, no handle). PlatformUnavailable is raised, not returned: it is not a property
        of this goal."""
        p = self.p
        if not self._busy.acquire(blocking=False):
            return {"ok": False, "reason": "motion backend busy", "plan": None, "_checked_plan": None}
        try:
            js = self._fresh_joint_state("before planning")
            plan = self._build_joint_plan(np.asarray(q_goal, dtype=float), v_mps, a_mps2, timeout_s, js, p.pose.latest_pose(),
                                          allow_ompl=allow_ompl)
            # 이 계획을 move_joint 가 그대로 쓴다 (같은 목표·속도, 팔이 그 뒤 움직이지 않았으면) — OMPL 은 확률적이라 다시 계획하면 다른 경로
            plan["key"] = (np.array(q_goal, dtype=float), float(v_mps), float(a_mps2), timeout_s)
            plan["t_planned"] = time.monotonic()
            if plan["last_plan"].get("estimate_only"):
                self._prepared_joint = None
                return {"ok": False, "reason": f"straight joint line blocked: {plan['last_plan']['straight_blocked_reason']}",
                        "plan": dict(plan["last_plan"]), "_checked_plan": None}
            self._prepared_joint = plan
            return {"ok": True, "reason": "", "plan": dict(plan["last_plan"]), "_checked_plan": plan}
        except PlatformUnavailable:
            raise
        except MotionError as e:
            return {"ok": False, "reason": str(e), "plan": None, "_checked_plan": None}
        finally:
            self._busy.release()

    def _validate_checked_joint_start(self, plan: dict, js) -> None:
        """The measured joints must still be the checked plan's start: by raw joint value (a 2 pi / branch change FK cannot see) and
        within move_group's own start-state tolerance (allowed_start_tolerance: the rule move_group applies to every trajectory it
        executes; KIRO field 2026-10-15: the former 0.05 deg / 0.1 mm - a reuse threshold - refused a 0.054 deg holding difference).
        The flange difference is reported. Else MotionError (nothing sent)."""
        m = dict(zip(js.name, js.position))
        try:
            q = np.array([float(m[n]) for n in plan["names"]])
        except KeyError as e:
            raise MotionError(f"checked joint plan: no measured joint {e} - nothing sent") from e
        if not np.all(np.isfinite(q)):
            raise MotionError("checked joint plan: non-finite measured joints - nothing sent")
        dq_deg = float(np.max(np.abs(np.degrees(q - plan["q0"])))) if q.size else 0.0
        fk = self.p.fk()
        Ta, Tb = fk(plan["q0"]), fk(q)
        dp_mm = float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3])) * 1e3
        dr_deg = _rot_deg(Ta[:3, :3], Tb[:3, :3])
        tol, src = self.p.allowed_start_tolerance()
        if not math.radians(dq_deg) <= tol:
            raise MotionError(f"checked joint plan start changed: max joint {dq_deg:.4f} deg > {math.degrees(tol):.3f} deg ({src}); flange "
                              f"{dp_mm:.3f} mm / {dr_deg:.4f} deg - plan and confirm again (nothing sent)")

    def move_checked_joint(self, check: dict, q_goal, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> MotionResult:
        """Execute exactly the trajectory check_joint checked for this goal (check['_checked_plan']), once: never planned again. The
        same goal / speed / acceleration / timeout as the check, the measured start still the plan's start; right before sending
        (_move) the whole trajectory is checked again against the current scene with a fresh full joint state, the start again, the
        joints we do not move unchanged meanwhile, and the run's stop - any failure: MotionError, nothing sent (KIRO field 2026-10-14)."""
        plan = (check or {}).get("_checked_plan")
        q_goal = np.asarray(q_goal, dtype=float)
        if not (check or {}).get("ok") or plan is None or plan.get("checked_consumed"):
            raise MotionError("the checked joint plan is unavailable or already used - nothing sent")
        kq, kv, ka, kto = plan["key"]
        if not (np.array_equal(kq, q_goal) and kv == float(v_mps) and ka == float(a_mps2) and kto == timeout_s):
            raise MotionError("the goal / speed / timeout differ from the checked joint plan - nothing sent")
        T_target = self.p.fk()(q_goal)

        def planner(js, T0, res):
            self._validate_checked_joint_start(plan, js)
            plan["used_prepared"] = True
            plan["validate_before_send"] = lambda fresh: self._validate_checked_joint_start(plan, fresh)
            res.monitor_events.append("the checked joint trajectory sent unchanged (no planning again)")
            return plan

        plan["checked_consumed"] = True                  # once, even if the move returns at the goal or refuses
        self._prepared_joint = None
        return self._move(T_target, planner, joint_goal=q_goal)

    def move_joint(self, q_goal, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> MotionResult:
        """관절 공간 이동 (MoveIt 상태 검사 / OMPL 계획 → 같은 JTC 전송·감시·래치 경로)."""
        q_goal = np.asarray(q_goal, dtype=float)
        T_target = self.p.fk()(q_goal)

        def planner(js, T0, res):
            self._prepared = None
            prep, self._prepared_joint = getattr(self, "_prepared_joint", None), None
            if prep is not None:
                kq, kv, ka, kto = prep["key"]
                same = np.allclose(kq, q_goal, rtol=0.0, atol=1e-12) and kv == float(v_mps) and ka == float(a_mps2) and kto == timeout_s
                if same:
                    self._prepared = {**prep, "key": (self.p.fk()(q_goal), float(v_mps), float(a_mps2), timeout_s)}
                    plan = self._take_prepared(T_target, v_mps, a_mps2, timeout_s, js, res)
                    if plan is not None:
                        return plan
            return self._build_joint_plan(q_goal, v_mps, a_mps2, timeout_s, js, T0)
        return self._move(T_target, planner, joint_goal=q_goal)

    def prepare_linear(self, T_target, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> Optional[dict]:
        """move_linear 의 계획을 **미리** 한다 (센서 녹화를 켜기 전에) — 다음 move_linear 가 같은 목표·속도이고 팔이 그동안
        움직이지 않았으면 (플랜지 prepared_start_tol_mm/deg 이내) 그 계획을 그대로 보낸다. 실기 at09/at10 (2026-10-03): 420 mm 직선의
        조밀 IK 계획 ~0.9 s 가 녹화 중에 같은 프로세스의 GIL 을 잡아 녹화가 밀리고 G3 거짓 정지. 실패·거부는 조용히 None (move_linear
        가 다시 계획하며 그때 같은 오류로 거부한다). 반환: 계획 요약 (last_plan) 또는 None."""
        p = self.p
        self._prepared = None
        T_target = np.asarray(T_target, dtype=float)
        if p.latch.is_set or not self._busy.acquire(blocking=False):
            return None
        try:
            js = self._fresh_joint_state("before planning")
            T0 = p.pose.latest_pose()
            if float(np.linalg.norm(T_target[:3, 3] - T0[:3, 3])) < 1e-5 and _rot_deg(T0[:3, :3], T_target[:3, :3]) < math.degrees(1e-4):
                return None
            plan = self._build_plan(T_target, v_mps, a_mps2, timeout_s, js, T0)
        except Exception:  # noqa: BLE001 - move_linear plans again and reports the refusal itself
            return None
        finally:
            self._busy.release()
        self._prepared = plan
        return dict(plan["last_plan"])

    def check_linear(self, T_target, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> dict:
        """move_linear 와 같은 계획·검사 (MoveIt 직선 fraction, 조밀 IK 점프·관절 한계, 재시간) 를 **보내지 않고** 현재 측정 자세에서 —
        auto-teach 가 센서를 켜기 전에 (dry run 포함) 첫 접근 구간을 미리 본다. 실기 at15 (2026-10-03): dry run 은 통과했는데 실행의
        첫 구간이 시작 자세의 손목 특이점 근처에서 0.054 rad 점프로 거부됐다. 반환 {ok, reason, plan}."""
        p = self.p
        T_target = np.asarray(T_target, dtype=float)
        if not self._busy.acquire(blocking=False):
            return {"ok": False, "reason": "motion backend busy", "plan": None}
        try:
            js = self._fresh_joint_state("before planning")
            T0 = p.pose.latest_pose()
            plan = self._build_plan(T_target, v_mps, a_mps2, timeout_s, js, T0)
            return {"ok": True, "reason": "", "plan": dict(plan["last_plan"])}
        except MotionError as e:
            return {"ok": False, "reason": str(e), "plan": None}
        finally:
            self._busy.release()

    def _take_prepared(self, T_target: np.ndarray, v_mps: float, a_mps2: float, timeout_s: Optional[float], js: JointState,
                       res: MotionResult) -> Optional[dict]:
        """미리 한 계획이 이 이동의 것이고 팔이 그 뒤로 움직이지 않았으면 그 계획 (한 번만 쓴다)."""
        prep, self._prepared = self._prepared, None
        if prep is None:
            return None
        kT, kv, ka, kto = prep["key"]
        if not (np.allclose(kT, T_target, rtol=0.0, atol=1e-12) and kv == float(v_mps) and ka == float(a_mps2) and kto == timeout_s):
            res.monitor_events.append("prepared plan not used (different target or speed) - planned again")
            return None
        js_map = dict(zip(js.name, js.position))
        try:
            q = np.array([float(js_map[n]) for n in prep["names"]])
        except KeyError:
            return None
        # 팔이 계획 뒤로 움직였는가: 관절 차를 플랜지 자세 차로 본다 (실기 정지 중 측정 관절 떨림 wrist3 ~3e-4 rad = 0.02° 는 통과)
        fk = self.p.fk()
        Ta, Tb = fk(prep["q0"]), fk(q)
        dp_mm = float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3])) * 1e3
        dr_deg = _rot_deg(Ta[:3, :3], Tb[:3, :3])
        mj = self.p.mj
        if dp_mm > mj.prepared_start_tol_mm or dr_deg > mj.prepared_start_tol_deg:
            res.monitor_events.append(f"prepared plan not used: the arm moved {dp_mm:.3f} mm / {dr_deg:.4f} deg since planning "
                                      f"(> {mj.prepared_start_tol_mm} mm / {mj.prepared_start_tol_deg} deg) - planned again")
            return None
        res.monitor_events.append(f"prepared plan used: planned {time.monotonic() - prep['t_planned']:.2f} s before sending, "
                                  f"start within {dp_mm:.3f} mm / {dr_deg:.4f} deg")
        prep["used_prepared"] = True
        return prep

    # ---- 실행 ----
    def move_linear(self, T_target, v_mps: float, a_mps2: float, *, timeout_s: Optional[float] = None) -> MotionResult:
        T_target = np.asarray(T_target, dtype=float)

        def planner(js, T0, res):
            plan = self._take_prepared(T_target, v_mps, a_mps2, timeout_s, js, res)
            return plan if plan is not None else self._build_plan(T_target, v_mps, a_mps2, timeout_s, js, T0)
        return self._move(T_target, planner)

    def _move(self, T_target: np.ndarray, planner, *, joint_goal: Optional[np.ndarray] = None) -> MotionResult:
        """계획 (planner(js, T0, res) → 계획 dict) → 전송 → JTC 결과 → 도착 확인. 직선·관절 이동이 같은 경로를 쓴다."""
        p = self.p
        mj = p.mj
        cfg = self.cfg
        res = MotionResult(script="", policy="moveit_jtc", default_speed=1.0)
        if not self._busy.acquire(blocking=False):
            raise MotionError("another motion is in progress")
        t0 = time.monotonic()
        t_send = 0
        try:
            if p.latch.is_set:
                raise MotionError(f"this run was stopped ({p.latch.reason}): no further motion in this run")
            self._abort.clear()
            js = self._fresh_joint_state("before planning")
            T0 = p.pose.latest_pose()
            L = float(np.linalg.norm(T_target[:3, 3] - T0[:3, 3]))
            rot = math.radians(_rot_deg(T0[:3, :3], T_target[:3, :3]))
            if joint_goal is None:
                at_goal = L < 1e-5 and rot < 1e-4
            else:
                m_ = dict(zip(js.name, js.position))
                names_ = list(p.chain().joint_names)
                at_goal = all(n in m_ for n in names_) and float(np.max(np.abs(np.array([float(m_[n]) for n in names_]) - joint_goal))) < 1e-4
            if at_goal:
                self._prepared = None
                res.finished, res.reason = True, "already at target (measured pose == target)"
                res.script = "moveit_jtc: zero-length move"
                return res
            plan = planner(js, T0, res)
            rt = plan["rt"]
            self.last_plan = dict(plan["last_plan"], prepared=bool(plan.get("used_prepared", False)))
            res.v_cmd = rt.v_cruise * 1e3
            res.a_cmd = rt.a * 1e3
            res.v_eff_expected_mps = rt.v_cruise
            res.script = plan["script"]
            res.monitor_events.extend(rt.notes)
            res.timeout_s = plan["timeout_s"]
            goal = plan["goal"]
            # 계획 계산 (조밀 IK·경로 대조, 이 프로세스의 파이썬) 동안 같은 프로세스의 콜백이 밀려 /joint_states 처리가 늦어질 수 있다
            # (실기 at17 2026-10-03: 손목 179° 회전 직선 448 점 계획 뒤 0.56 s 밀림). 보내기 전에 계획 뒤에 발행된 상태를 받는다.
            t_w0 = time.monotonic()
            self._fresh_joint_state("after planning")
            res.monitor_events.append(f"/joint_states published after planning received after {time.monotonic() - t_w0:.3f} s")
            if not self._action.wait_for_server(timeout_sec=2.0):
                raise MotionError(f"JTC action {mj.jtc_action} not available")
            if p.latch.is_set:
                raise MotionError(f"this run was stopped ({p.latch.reason}) before the trajectory was sent: not sent")
            if self._abort.is_set():
                res.aborted, res.reason = True, "aborted before start"
                return res
            validator = plan.get("validate_before_send")
            if validator is not None:
                # a checked plan (move_checked_joint): after confirmation and the sensor start, right before sending - the start, the
                # whole trajectory against the current scene with a fresh full state, the start again, the other joints unchanged
                fresh = self._fresh_joint_state("before sending the checked joint trajectory")
                validator(fresh)
                invalid = self._first_invalid(plan["names"], rt.positions, joint_state=fresh)
                if invalid is not None:
                    raise MotionError(f"the checked joint trajectory is no longer collision-free: {invalid} - nothing sent")
                after = self._fresh_joint_state("after checking the joint trajectory again")
                validator(after)
                a_map = dict(zip(after.name, after.position))
                for jn, v0 in zip(fresh.name, fresh.position):
                    if jn in plan["names"]:
                        continue
                    v1 = a_map.get(jn)
                    if v1 is None or not math.isfinite(float(v1)) or abs(float(v1) - float(v0)) > p.allowed_start_tolerance()[0]:
                        raise MotionError(f"joint {jn} (not ours) moved while the trajectory was checked again - nothing sent")
                if p.latch.is_set or self._abort.is_set():
                    raise MotionError("stopped while the checked joint trajectory was checked again - nothing sent")
            t_send = _now_ns()
            res.t_send_ns = t_send
            fut = self._action.send_goal_async(goal)
            if not wait_future(fut, 5.0) or fut.result() is None:
                res.reason = "JTC goal: no acceptance response within 5 s"
                return res
            gh = fut.result()
            res.t_ack_ns = _now_ns()
            if not gh.accepted:
                why = "JTC goal REJECTED by the controller"
                res.monitor_events.append(why)
                res.reason = why
                return res
            with self._gh_lock:
                self._goal_handle = gh
            res.started = True
            res.t_start_ns = res.t_ack_ns
            rfut = gh.get_result_async()
            prog = cfg.progress_s if (cfg.progress_s > 0 and cfg.progress_cb is not None) else 0.0
            t_m0 = time.monotonic()
            next_prog = t_m0 + prog
            last_p, last_t = T0[:3, 3].copy(), t_m0
            deadline = t_m0 + res.timeout_s
            while True:
                if p.latch.is_set:                                # 액션 취소의 정지 (래치가 이미 감속 정지를 보냈다)
                    res.aborted, res.reason = True, f"stopped: {p.latch.reason}"
                    wait_future(rfut, 2.0)
                    break
                if self._abort.is_set():
                    gh.cancel_goal_async()
                    wait_future(rfut, 2.0)
                    res.aborted, res.reason = True, "aborted by request"
                    break
                if rfut.done():
                    r = rfut.result()
                    status = r.status if r is not None else GoalStatus.STATUS_UNKNOWN
                    ec = r.result.error_code if r is not None else None
                    if status == GoalStatus.STATUS_SUCCEEDED and (ec is None or ec == 0):
                        res.finished = True
                        res.reason = "trajectory succeeded"
                    else:
                        why = (f"JTC goal ended with status {status} error_code {ec} "
                               f"({getattr(r.result, 'error_string', '') if r is not None else ''})")
                        res.monitor_events.append(why)
                        res.aborted, res.reason = True, why               # JTC 가 끝낸 목표: 이동 실패 (정지 명령 없음)
                    break
                if time.monotonic() >= deadline:                      # JTC 결과가 오지 않는다: 우리 목표만 끝낸다
                    why = f"no JTC result within {res.timeout_s:.1f} s: our goal cancelled"
                    gh.cancel_goal_async()
                    res.aborted, res.reason = True, why
                    wait_future(rfut, 2.0)
                    break
                if prog and time.monotonic() >= next_prog:
                    try:
                        Tn = p.pose.latest_pose()
                        tm = time.monotonic()
                        moved = float(np.linalg.norm(Tn[:3, 3] - T0[:3, 3]))
                        vi = float(np.linalg.norm(Tn[:3, 3] - last_p) / max(tm - last_t, 1e-6))
                        last_p, last_t = Tn[:3, 3].copy(), tm
                        msg = f"  moving: t={tm - t_m0:5.1f} s  moved {moved * 1e3:6.1f} / {L * 1e3:.1f} mm  v~{vi * 1e3:.2f} mm/s"
                        if vi > 1e-5:
                            msg += f"  ETA ~{max(0.0, L - moved) / vi:.0f} s"
                        cfg.progress_cb(msg)
                    except Exception:  # noqa: BLE001 - feedback must never affect the motion
                        pass
                    next_prog += prog
                time.sleep(0.005)
            res.t_stop_ns = _now_ns()
            # ---- 도착 확인 (실측) ----
            if res.finished:
                t_end = time.monotonic() + 2.0
                while True:
                    Tm = p.pose.latest_pose()
                    dp = float(np.linalg.norm(Tm[:3, 3] - T_target[:3, 3]))
                    dr = _rot_deg(Tm[:3, :3], T_target[:3, :3])
                    if (dp <= mj.goal_pose_tol_m and dr <= mj.goal_pose_tol_deg) or time.monotonic() >= t_end:
                        break
                    time.sleep(0.01)
                res.monitor_events.append(f"final measured pose error {dp * 1e3:.3f} mm / {dr:.4f} deg")
                if dp > mj.goal_pose_tol_m or dr > mj.goal_pose_tol_deg:
                    res.finished = False
                    res.reason = (f"trajectory finished but the measured pose is {dp * 1e3:.2f} mm / {dr:.3f} deg from the target "
                                  f"(> {mj.goal_pose_tol_m * 1e3:.1f} mm / {mj.goal_pose_tol_deg} deg)")
                res.t_stop_ns = _now_ns()
            return res
        finally:
            with self._gh_lock:
                self._goal_handle = None
            res.elapsed_s = time.monotonic() - t0
            if t_send:
                try:
                    res.samples = p.pose.history(t_send)
                    meas = _measure_speed(res.samples)
                    res.speed_measurement = meas
                    res.v_measured_mps = meas.v_measured_mps
                except Exception as e:  # noqa: BLE001
                    res.monitor_events.append(f"speed measurement failed: {e}")
            self.last_result = res
            self._busy.release()


def _measure_speed(samples, min_dt_s: float = 0.04) -> SpeedMeasurement:
    """PoseSample 목록 → TCP(플랜지) 속도 (구간 속도의 크루즈 중앙값: 최대의 60 % 이상인 구간 속도의 중앙값)."""
    m = SpeedMeasurement(source="ros_tf")
    if not samples:
        return m
    P = np.array([np.asarray(s.T_base_flange)[:3, 3] for s in samples])
    t = np.array([s.t_rx_ns * 1e-9 for s in samples])
    m.n_samples = len(samples)
    if len(samples) < 2:
        return m
    m.path_length_m = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    m.duration_s = float(t[-1] - t[0])
    if m.duration_s > 0:
        m.v_mean_mps = m.path_length_m / m.duration_s
    speeds = []
    j = 0
    for i in range(len(t)):
        while j < len(t) and t[j] - t[i] < min_dt_s:
            j += 1
        if j >= len(t):
            break
        speeds.append(float(np.linalg.norm(P[j] - P[i]) / (t[j] - t[i])))
    if speeds:
        v = np.asarray(speeds)
        m.v_max_mps = float(v.max())
        cruise = v[v >= 0.6 * v.max()] if v.max() > 0 else v
        m.v_measured_mps = float(np.median(cruise)) if cruise.size else None
    return m


# --------------------------------------------------------------------------------------
# 게이트
# --------------------------------------------------------------------------------------
@dataclass
class MoveItInitRecord:
    """MoveItGates.session_init 기록 (준비 확인·선언 한계)."""

    default_speed: float = 1.0
    limits: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    pose: dict = field(default_factory=dict)          # RosPoseSource.record (method fk_from_joint_states, URDF source) + TF cross-check

    def as_dict(self) -> dict:
        return {"backend": "moveit_jtc", "default_speed": self.default_speed, "limits": dict(self.limits), "checks": dict(self.checks),
                "warnings": list(self.warnings), "pose": dict(self.pose)}


class MoveItGates(PlatformGates):
    name = "moveit_jtc"

    def __init__(self, settings, platform: Optional["MoveItPlatform"] = None):
        self.settings = settings
        self.mj = MoveItJtcConfig.from_params(settings.params)      # 한계 선언이 없으면 여기서 PlatformConfigError (시작 거부)
        self.platform = platform

    def motion_config(self, settings, **overrides) -> MotionConfig:
        mj = self.mj
        v_cap = float(mj.max_tcp_speed_mps)                          # 속도 상한은 하나: 플랫폼 선언 TCP 한계 (포락선 여유의 정지 거리도 이 값)
        kw = dict(v_eff_max_mps=v_cap, a_eff_max_mps2=float(mj.max_tcp_accel_mps2))
        kw.update(overrides)
        return MotionConfig(**kw)

    def session_init(self, platform, cfg, log):
        mj = self.mj
        rec = MoveItInitRecord(limits=mj.limits_dict())
        missing = platform.wait_ready(timeout_s=10.0)
        if not missing:
            missing = platform.readiness()
        # the pose stream is FK of /joint_states with robot_description; one look at TF base <- flange (at a stamp TF already covers,
        # however late) shows that URDF is the one behind the robot's TF — a different model is not ready (no per-sample TF)
        xc = platform.pose.tf_cross_check()
        if xc.get("checked") and not xc.get("ok"):
            missing = list(missing) + [f"robot_description FK {mj.base_frame} <- {mj.flange_frame} differs from the platform's TF at the "
                                       f"same stamp by {xc['position_diff_m'] * 1e3:.3f} mm / {xc['rotation_diff_deg']:.4f} deg "
                                       f"(> {mj.path_check_tol_m * 1e3:.2f} mm / {FRAME_ROT_TOL_DEG} deg): not the same robot model"]
        elif not xc.get("checked"):
            rec.warnings.append(f"pose model not cross-checked against TF: {xc.get('why')}")
        rec.pose = dict(platform.pose.record(), tf_cross_check=xc)
        rec.checks = {"missing": list(missing), "joint_states_age_s": platform.pose.joint_states_age_s()}
        if missing:
            raise PlatformNotReady("moveit_jtc platform not ready: " + "; ".join(missing))
        log(f"session_init OK (moveit_jtc): group {mj.move_group_name} eef {mj.flange_frame} in {mj.base_frame}, JTC {mj.jtc_action}, "
            f"pose {POSE_METHOD} ({rec.pose.get('urdf_source')}), limits TCP {mj.max_tcp_speed_mps * 1e3:.0f} mm/s "
            f"{mj.max_tcp_accel_mps2} m/s^2, joints {mj.max_joint_velocity_rad_s} rad/s {mj.max_joint_accel_rad_s2} rad/s^2")
        return rec

    def status(self, platform) -> dict:
        d = {"adapter": "moveit_jtc", "joint_states_age_s": platform.pose.joint_states_age_s(),
             "n_joint_states": platform.pose.n_js, "n_pose_samples_dropped": platform.pose.n_dropped, "pose_method": POSE_METHOD,
             "run_stopped": platform.latch.is_set, "stop_reason": platform.latch.reason}
        try:
            d["flange_pose_m"] = platform.pose.latest_pose()[:3, 3].tolist()
        except Exception as e:  # noqa: BLE001
            d["flange_pose_error"] = str(e)
        return d

    def default_speed(self, platform) -> float:
        return 1.0

    def effective_speed(self, platform, cfg, v_target: float) -> float:
        return float(min(float(v_target), cfg.v_eff_max_mps, self.mj.max_tcp_speed_mps))

    def frame_offset_record(self, platform, snap: PoseSnapshot) -> dict:
        d = {"note": "moveit_jtc: targets are sent in the measured frame (base_frame <- flange_frame = FK of /joint_states with "
                     "robot_description); no command-frame offset",
             "drift_checked": False}
        d.update({k: v for k, v in snap.record.items() if k in ("joint_names", "joint_positions_rad", "stamp_ns")})
        return d

    def scan_leg_record(self, res) -> dict:
        d = super().scan_leg_record(res)
        d["backend"] = "moveit_jtc"
        return d


# --------------------------------------------------------------------------------------
# 플랫폼 묶음
# --------------------------------------------------------------------------------------
def dense_tracks(arglists: list, *, fn=None, workers="auto") -> tuple:
    """조밀 IK 추종 (scan_core.trajectory.track_polyline; fn 으로 다른 추종 — 도달 후보는 trajectory.follow_candidate_job, MoveItPlatform.
    _follow) 을 줄마다 spawn 프로세스 풀 (seam_features.worker_pool — 서버 프로세스의 GIL 경합 없이) 에서 → (tracks, unchecked, note):
    arglists 항목마다 결과 (None 항목은 None), unchecked = {i: 까닭}, note = 풀이 수명 규칙 없이
    돌았으면 그 까닭 (OwnedPool.rule_note; 아니면 None). 풀 규칙 (scan_core.proc_pool): 죽은
    자식·풀 오류 → 그 줄은 여기서 같은 계산 (OwnedPool: 죽음은 바로 BrokenProcessPool); 한계 (RESULT_TIMEOUT_S — 패스 뒤 처리와 같은 값,
    전체에 한 번) 안에 답이 없으면 (살았지만 멈춘 풀 — 그 자식을 끝내고 풀을 내린다) 그 줄은 확인하지 않음. 이 호출은 한계보다 오래 막히지
    않는다 (예전: 한계 없는 f.result() — 결과를 쓰다 죽은 자식·멈춘 자식이면 영원히)."""
    from scan_core import proc_pool
    from scan_core.seam_features import shutdown_feature_pool, worker_pool
    from scan_core.trajectory import track_polyline
    fn = track_polyline if fn is None else fn
    tracks: list = [None] * len(arglists)
    unchecked: dict = {}
    pool, futs = None, [None] * len(arglists)
    try:
        pool = worker_pool(workers)
        for i, a in enumerate(arglists):
            if a is not None:
                futs[i] = pool.submit(fn, *a)
    except Exception:  # noqa: BLE001 - the lines not submitted are computed here
        pass
    _done, stalled = proc_pool.wait_within(futs, proc_pool.RESULT_TIMEOUT_S)
    if stalled:
        shutdown_feature_pool(pool, kill=True)
    for i, a in enumerate(arglists):
        if a is None:
            continue
        f = futs[i]
        if f is not None and f in stalled:
            unchecked[i] = (f"no answer from the process pool within {proc_pool.RESULT_TIMEOUT_S:.0f} s (alive but stalled) - "
                            f"not checked")
        elif f is not None and f.exception() is None:
            tracks[i] = f.result()
        else:
            tracks[i] = fn(*a)                          # 죽은 자식·풀 오류·보내지 못함: 같은 계산을 여기서
    return tracks, unchecked, (pool.rule_note if pool is not None else None)


class MoveItPlatform(RobotPlatform):
    """MoveIt 어댑터 의 RobotPlatform. 전용 rclpy 노드 (/hongik_fine_sensing/hongik_moveit_adapter) 와 executor 스레드를 가진다."""

    adapter = "moveit_jtc"
    pose_source = "ros_tf"

    def __init__(self, settings, *, ctx: Optional[RosContext] = None, node_name: str = "hongik_moveit_adapter"):
        self.settings = settings
        self._gates = MoveItGates(settings, self)
        self.mj: MoveItJtcConfig = self._gates.mj
        self.ctx = ctx or RosContext(node_name, use_sim_time=self.mj.use_sim_time)
        self.cb = ReentrantCallbackGroup()
        node = self.ctx.node
        self._pose = RosPoseSource(self.ctx, self.mj, self.cb)
        self._latch = RosStopLatch(self.ctx, self.mj, self.cb)
        self._backend: Optional[MoveItJtcBackend] = None
        self._chain = None
        self._fk = None
        self._pose.add_model_listener(self._on_model)   # the planning chain = the pose stream's model (a new URDF replaces both)
        self._ctrl_ref: Optional[tuple] = None          # (joint_names, 지령 위치, 지령 속도, stamp, 호스트 수신 ns) — 감속 정지용
        # controller_state 이력 (history_s): 구간마다 지령 (reference) 과 실측 (feedback) 관절을 raw.npz 에 남긴다 — 자세 평활 (지연만큼
        # 민 지령 궤적 + 실측 관절) 의 자료 (사용자 2026-10-11 밤)
        self._cs_hist: deque = deque()
        self._cs_lock = threading.Lock()
        self.n_ctrl_record_errors = 0
        self._cs_sub = node.create_subscription(JointTrajectoryControllerState, self.mj.controller_state, self._on_cs, 10,
                                                callback_group=self.cb)
        self.cart_client = node.create_client(GetCartesianPath, self.mj.cartesian_path_service, callback_group=self.cb)
        self.fk_client = node.create_client(GetPositionFK, self.mj.fk_service, callback_group=self.cb)
        self.ik_client = node.create_client(GetPositionIK, self.mj.ik_service, callback_group=self.cb)
        self.plan_client = node.create_client(GetMotionPlan, self.mj.plan_service, callback_group=self.cb)
        self.validity_client = node.create_client(GetStateValidity, self.mj.state_validity_service, callback_group=self.cb)
        self.scene_client = node.create_client(ApplyPlanningScene, self.mj.apply_scene_service, callback_group=self.cb)
        self.get_scene_client = node.create_client(GetPlanningScene, self.mj.get_scene_service, callback_group=self.cb)
        self.jtc_client = ActionClient(node, FollowJointTrajectory, self.mj.jtc_action, callback_group=self.cb)
        self._latch.busy_getter = lambda: self.busy

    # ---- 구독 ----
    def _on_model(self, _model) -> None:
        self._chain = None
        self._fk = None

    def _on_cs(self, msg) -> None:
        self._cs_record(msg)
        ref = getattr(msg, "reference", None)
        if ref is not None and len(ref.positions) and len(ref.velocities) == len(ref.positions):
            self._ctrl_ref = (list(msg.joint_names), np.asarray(ref.positions, dtype=float), np.asarray(ref.velocities, dtype=float),
                              msg.header.stamp, _now_ns())

    def _cs_record(self, msg) -> None:
        """One controller state into the history: (host receive ns, header stamp ns, names, reference q / qd, feedback q)."""
        try:
            ref = getattr(msg, "reference", None)
            if ref is None or not len(ref.positions):
                ref = getattr(msg, "desired", None)
            fb = getattr(msg, "feedback", None)
            if fb is None or not len(fb.positions):
                fb = getattr(msg, "actual", None)
            if ref is None or fb is None or not len(ref.positions) or len(ref.positions) != len(fb.positions):
                return
            n = len(ref.positions)
            qd = list(ref.velocities) if len(ref.velocities) == n else [float("nan")] * n
            row = (_now_ns(), stamp_to_ns(msg.header.stamp), tuple(msg.joint_names), tuple(float(v) for v in ref.positions),
                   tuple(float(v) for v in qd), tuple(float(v) for v in fb.positions))
            t_min = row[0] - int(self.mj.history_s * 1e9)
            with self._cs_lock:
                self._cs_hist.append(row)
                while self._cs_hist and self._cs_hist[0][0] < t_min:
                    self._cs_hist.popleft()
        except Exception:  # noqa: BLE001 - the callback must survive; the leg then has no controller record
            self.n_ctrl_record_errors += 1

    def controller_history(self, since_ns: int, until_ns: Optional[int] = None) -> dict:
        """The controller states received in [since_ns, until_ns] (host monotonic) as raw.npz arrays (ctrl_* keys): header stamp
        (the controller's clock, as robot_stamp_ns), host receive time, joint names, reference (command) positions / velocities and
        feedback (measured) positions [rad]. {} when none (no controller_state stream)."""
        with self._cs_lock:
            rows = [r for r in self._cs_hist if r[0] >= int(since_ns) and (until_ns is None or r[0] <= int(until_ns))]
        if not rows:
            return {}
        names = rows[-1][2]
        rows = [r for r in rows if r[2] == names]
        return {"ctrl_t_rx_ns": np.array([r[0] for r in rows], np.int64), "ctrl_stamp_ns": np.array([r[1] for r in rows], np.int64),
                "ctrl_joint_names": np.array(list(names)), "ctrl_ref_positions_rad": np.array([r[3] for r in rows], float),
                "ctrl_ref_velocities_rad_s": np.array([r[4] for r in rows], float),
                "ctrl_fb_positions_rad": np.array([r[5] for r in rows], float)}

    def controller_reference(self) -> Optional[tuple]:
        """JTC 가 지금 지령하는 위치·속도 (controller_state.reference) — (names, q, qd, stamp, 호스트 수신 ns) 또는 None."""
        return self._ctrl_ref

    def chain(self):
        """robot_description (URDF) 의 base_frame → flange_frame 직렬 체인 (FK·IK 추종용) = 자세 흐름이 FK 로 쓰는 바로 그 모형
        (RosPoseSource.model; 새 robot_description 이 오면 둘 다 바뀐다). 없으면 MotionError (사유)."""
        if self._chain is None:
            self._chain = self._pose.model(timeout_s=5.0).chain
        return self._chain

    def fk(self):
        """체인의 빠른 FK (trajectory.fast_fk, 처음 한 번)."""
        if self._fk is None:
            self._fk = fast_fk(self.chain())
        return self._fk

    def ik_solutions(self, T_base_flange, seeds, *, avoid_collisions: bool = True, dedupe_rad: float = 1e-3,
                     timeout_s: float = 1.0) -> list:
        """플랜지 자세의 IK 해들 — 시드마다 move_group /compute_ik (충돌 검사 포함), 같은 해는 하나로 (관절 값 그대로: 손목 ±360° 는
        다른 해). 관절 순서 = 체인 순서. 해가 없으면 빈 목록 (2026-10-07: 접힌 자세에서 획득 자세로 — 손목 형태는 도달성으로 고른다)."""
        mj = self.mj
        names = list(self.chain().joint_names)
        out: list = []
        for seed in seeds:
            q = GetPositionIK.Request()
            q.ik_request.group_name = mj.move_group_name
            q.ik_request.ik_link_name = mj.flange_frame
            q.ik_request.pose_stamped.header.frame_id = mj.base_frame
            q.ik_request.pose_stamped.pose = matrix_to_pose(np.asarray(T_base_flange, dtype=float))
            q.ik_request.robot_state.joint_state.name = names
            q.ik_request.robot_state.joint_state.position = [float(v) for v in seed]
            q.ik_request.avoid_collisions = bool(avoid_collisions)
            q.ik_request.timeout.sec = int(timeout_s)
            q.ik_request.timeout.nanosec = int(round((float(timeout_s) - int(timeout_s)) * 1e9))
            r = call_service(self.ik_client, q, 10.0, wait_service_s=2.0)
            if r is None or r.error_code.val != MoveItErrorCodes.SUCCESS:
                continue
            m = dict(zip(r.solution.joint_state.name, r.solution.joint_state.position))
            sol = np.array([float(m[n]) for n in names])
            if all(float(np.max(np.abs(sol - o))) > dedupe_rad for o in out):
                out.append(sol)
        return out

    def _chain_q(self, q) -> Optional[np.ndarray]:
        """관절 값 → 체인 순서 배열: dict {이름: 값} (/joint_states 순서와 무관) 또는 체인 순서 시퀀스. 이름이 모자라면 None."""
        if q is None:
            return None
        names = list(self.chain().joint_names)
        if isinstance(q, dict):
            try:
                return np.array([float(q[n]) for n in names])
            except KeyError:
                return None
        q = np.asarray(q, dtype=float).ravel()
        return q if q.size == len(names) else None

    def ik_seeds(self, q_now=None, n_random: int = 32) -> list:
        """IK 시드 — 로봇 모델·영점 방향을 가정하지 않는다: 지금 관절 (dict 또는 체인 순서, 있으면 맨 앞) + URDF 관절 한계 안 균일 표본
        n_random 개 (한 바퀴 넘는 관절은 ±π 안; 고정 난수 시드 — 같은 입력이면 같은 후보)."""
        chain = self.chain()
        qc = self._chain_q(q_now)
        out = [[float(v) for v in qc]] if qc is not None else []
        lo = np.array([max(j.lower, -math.pi) for j in chain.movable_joints])
        hi = np.array([min(j.upper, math.pi) for j in chain.movable_joints])
        rng = np.random.default_rng(20261007)
        for _ in range(int(n_random)):
            out.append([float(v) for v in lo + (hi - lo) * rng.random(lo.size)])
        return out

    def state_collision(self) -> Optional[str]:
        """지금 관절 상태가 move_group 충돌 장면 (로봇 자신·장면 상자) 과 충돌하면 그 사유, 아니면 None (범용 시작: 장면 상자가 지금 자세와
        겹치지 않게 깎는 데)."""
        js = self.pose.latest_joint_state()
        if js is None:
            return "no /joint_states"
        names = list(self.chain().joint_names)
        m = dict(zip(js[1].name, js[1].position))
        try:
            q = np.array([[float(m[n]) for n in names]])
        except KeyError as e:
            return f"/joint_states has no joint {e}"
        return self.backend._first_invalid(names, q, joint_state=js[1])

    def state_contacts(self) -> dict:
        """지금 관절 상태의 move_group 상태 검사 (check_state_validity, 모션 없음) → {valid, contacts [{link, object, depth_m, kind}], why}.
        kind 'world' = 로봇 링크 ↔ 장면 물체 (link 가 로봇 쪽), 'self' = 로봇 링크끼리, 'attached' = 붙인 물체. 1G core 의 셀 세계가 지금
        팔이 세계 안이면 어느 링크가 어느 물체 안인지 이름으로 거부한다 (세계를 줄이지 않는다)."""
        js = self.pose.latest_joint_state()
        if js is None:
            return {"valid": False, "contacts": [], "why": "no /joint_states"}
        names = list(self.chain().joint_names)
        m = dict(zip(js[1].name, js[1].position))
        try:
            q = [float(m[n]) for n in names]
        except KeyError as e:
            return {"valid": False, "contacts": [], "why": f"/joint_states has no joint {e}"}
        req = GetStateValidity.Request()
        req.group_name = self.mj.move_group_name
        _fill_robot_state(req.robot_state, js[1], dict(zip(names, q)))       # full measured state, is_diff (attached bodies kept)
        r = call_service(self.validity_client, req, 5.0, wait_service_s=2.0)
        if r is None:
            return {"valid": False, "contacts": [], "why": f"move_group {self.mj.state_validity_service} did not answer"}
        by_pair: dict = {}
        for c in r.contacts:
            t1, t2 = int(c.body_type_1), int(c.body_type_2)           # moveit_msgs/ContactInformation: 0 link, 1 world object, 2 attached
            if t1 == 1 and t2 != 1:
                link, obj, kind = c.contact_body_2, c.contact_body_1, "world"
            elif t2 == 1 and t1 != 1:
                link, obj, kind = c.contact_body_1, c.contact_body_2, "world"
            else:
                link, obj, kind = c.contact_body_1, c.contact_body_2, ("self" if t1 == 0 and t2 == 0 else "attached")
            k = (str(link), str(obj))
            d = float(c.depth)
            if k not in by_pair or d > by_pair[k]["depth_m"]:
                by_pair[k] = {"link": k[0], "object": k[1], "depth_m": d, "kind": kind}
        out = list(by_pair.values())
        why = "" if r.valid else ("contact" if out else "invalid without a reported contact (a joint outside its limits or a constraint)")
        return {"valid": bool(r.valid), "contacts": out, "why": why}

    def fixed_bodies(self) -> list:
        """로봇의 고정 몸체: base_frame 에 고정 관절로만 붙은 링크들의 충돌 형상 (robot_description, base frame 축 정렬 상자 —
        scan_core.fk_urdf.fixed_link_boxes; 메시는 ament 색인으로 찾는다). 1G core 의 셀 세계 바닥이 로봇 자신을 덮지 않게 (그 C 앞에서
        멈춘다). 처음 한 번 읽는다."""
        fb = getattr(self, "_fixed_bodies_cache", None)
        if fb is None:
            import io
            from scan_core.fk_urdf import fixed_link_boxes
            self.chain()                                                  # robot_description 이 있어야 한다 (없으면 MotionError)
            fb = fixed_link_boxes(io.StringIO(self.robot_description), self.mj.base_frame)
            self._fixed_bodies_cache = fb
        return fb

    def reach_candidates(self, T_goal, follow=(), *, q_now=None, avoid_collisions: bool = True, ik_timeout_s: float = 0.1,
                         sigma_radius_m: float = 0.2) -> list:
        """획득 자세 후보 (2026-10-07 범용 시작): T_goal (플랜지) 의 IK 해마다 follow 자세들을 직선으로 차례로 IK 추종 (trajectory.
        track_polyline — 실행 백엔드와 같은 단계·관절 점프·URDF 한계 규칙) 해 본다. 반환 [{names (체인 순서), q, ok, reason, limit_margin_rad
        (추종 경로의 최소 관절 한계 여유), min_sigma (야코비안 최소 특이값, 각속도 행 × sigma_radius_m — 손목 특이점에서 0), dist_rad
        (지금 관절에서 Σ|Δq|; q_now 없으면 None), n_track}]. q_now: dict {이름: 값} 또는 체인 순서. IK 해마다 그 해와, 지금 관절에 가장
        가까운 2π 동치 표현 (trajectory.nearest_turn_equivalent — 같은 링크 자세, 한계 안) 을 둘 다 후보로 (같으면 하나).
        손목 형태는 호출자가 이 목록에서 한 번 고르고 (scan_l0.acquisition.rank_reach_candidates) 그 뒤 직선 이동이 이어간다."""
        from scan_core.trajectory import nearest_turn_equivalent
        chain = self.chain()
        names = list(chain.joint_names)
        lo = np.array([j.lower for j in chain.movable_joints])
        hi = np.array([j.upper for j in chain.movable_joints])
        qc = self._chain_q(q_now)
        sols = self.ik_solutions(T_goal, self.ik_seeds(q_now), avoid_collisions=avoid_collisions, timeout_s=ik_timeout_s)
        variants: list = []
        for q in sols:
            # 같은 플랜지 자세의 관절 표현 (q_j ± 2π, 한계 안) 중 지금 관절에 가장 가까운 것도 후보로 (IK 는 표현을 고르지 않는다 — 2026-10-08
            # 실기: 손목1 +180° 를 받아 232.5° 회전 48.6 s, 손목2 +290° (= −70°) 로 진행 +1 이 멀어 보여 −1 (센서 뒤집기) 을 골랐다)
            for v in ([q, nearest_turn_equivalent(q, qc, lo, hi)] if qc is not None else [q]):
                v = np.asarray(v, dtype=float)
                if all(float(np.max(np.abs(v - o))) > 1e-3 for o in variants):
                    variants.append(v)
        return [{"names": names, **rec} for rec in self._follow(variants, follow, sigma_radius_m, qc)]

    def _follow(self, qs, follow, sigma_radius_m: float, q_ref=None) -> list:
        """관절 해 qs 마다 follow 추종 기록 (trajectory.follow_candidate, qs 순서). 후보마다 독립인 순수 계산이라 spawn 프로세스 풀
        (dense_tracks — 서버 프로세스의 ROS 실행기 스레드와 GIL 경합 없이; 실기 2026-10-10 g1c_04: 획득 후보 55 개 추종이 서버 스레드에서
        30 s) 에서 같은 계산을 한다. 풀이 답하지 못한 후보 (한계 안에 답이 없는 멈춘 풀) 는 여기서 같은 계산 — 모든 후보를 같은 규칙으로 본다."""
        from scan_core.trajectory import follow_candidate_job
        chain = self.chain()
        mj = self.mj
        lo = np.array([j.lower for j in chain.movable_joints])
        hi = np.array([j.upper for j in chain.movable_joints])
        kw = dict(step_m=mj.cartesian_step_m, lower=lo, upper=hi, max_joint_step_rad=mj.max_joint_step_rad, sigma_radius_m=sigma_radius_m,
                  q_ref=q_ref)
        poses = [np.asarray(T, dtype=float) for T in follow]
        args = [(chain, np.asarray(q, dtype=float), poses, kw) for q in qs]
        recs, unchecked, _note = dense_tracks(args, fn=follow_candidate_job)
        for i in unchecked:
            recs[i] = follow_candidate_job(*args[i])
        for rec in recs:
            rec.pop("q_end", None)
        return recs

    def follow_from(self, q, follow, *, sigma_radius_m: float = 0.2) -> dict:
        """이미 아는 관절 q (dict 또는 체인 순서) 에서 follow 자세들을 직선으로 차례로 추종 (reach_candidates 와 같은 규칙, IK 호출 없음) →
        {names, q, ok, reason, limit_margin_rad, min_sigma, n_track}. P2: 지금 형태가 회전 정렬·트랜싯 사슬을 따라가나 (사전검사), 획득 후보가
        판 법선 둘레 ±ψ_b 회전 자세도 따라가나 (yaw 띠) — 후보마다 IK 를 다시 풀지 않는다 (실기 IK 계획 시간)."""
        qc = self._chain_q(q)
        if qc is None:
            raise ValueError("follow_from: joint values do not cover the chain")
        return {"names": list(self.chain().joint_names), **self._follow([qc], follow, sigma_radius_m)[0]}

    def set_scene_boxes(self, boxes: list) -> bool:
        """작업 공간 모델 상자 → move_group planning scene (world, 같은 id 는 덮어쓴다). boxes = [{id, size (3,) [m], T (4×4, base)}]."""
        mj = self.mj
        ps = PlanningScene()
        ps.is_diff = True
        for b in boxes:
            co = CollisionObject()
            co.header.frame_id = mj.base_frame
            co.id = str(b["id"])
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [float(x) for x in b["size"]]
            co.primitives = [prim]
            co.primitive_poses = [matrix_to_pose(np.asarray(b["T"], dtype=float))]
            co.operation = CollisionObject.ADD
            ps.world.collision_objects.append(co)
        req = ApplyPlanningScene.Request()
        req.scene = ps
        r = call_service(self.scene_client, req, 10.0, wait_service_s=2.0)
        ok = bool(r is not None and r.success)
        allow = [str(b["id"]) for b in boxes if b.get("allow_fixed_links")]
        if ok and allow:                 # the robot's fixed links (e.g. a test bed under the specimen) may lie in these solids: they
            links = [str(f.get("link")) for f in (self.fixed_bodies() or []) if f.get("link")]     # cannot move - allow exactly those
            base = self._acm_read() if links else None
            if base is not None and not self._acm_set(self._acm_with(base, links, allow)):
                self.ctx.node.get_logger().warn(f"allowed collisions fixed links <-> {allow} not set: a fixed link inside them reads as a "
                                            "contact (the state check then names it)")
        return ok

    def arm_reach_m(self) -> float:
        """How far from its base origin any part of the arm and tool can reach at most (cell world 'under' solid): the chain's bound
        (base -> flange: the summed joint offsets) + the tool's extent from the flange (weld torch TCP, profiler optical frame)."""
        r = float(self.chain().reach_bound_m())
        mj = self.mj
        ext = []
        if mj.weld_torch is not None and getattr(mj.weld_torch, "flange_from_tcp", None) is not None:
            ext.append(float(np.linalg.norm(np.asarray(mj.weld_torch.flange_from_tcp.matrix(), dtype=float)[:3, 3])))
        try:
            ext.append(float(np.linalg.norm(self._pose.static_transform(mj.flange_frame, mj.optical_frame, 2.0)[:3, 3])))
        except Exception:  # noqa: BLE001 - the optical frame not in TF yet: the torch extent alone
            pass
        return r + (max(ext) if ext else 0.0)

    def set_scene_meshes(self, objs: list) -> bool:
        """Mesh objects → move_group planning scene (world, the same id is replaced). objs = [{id, vertices_m (V, 3) base, triangles
        (T, 3)}] (scan_core.weld_scene.scene_meshes - the weld-pose specimen)."""
        ps = PlanningScene()
        ps.is_diff = True
        for o in objs:
            co = CollisionObject()
            co.header.frame_id = self.mj.base_frame
            co.id = str(o["id"])
            m = Mesh()
            m.vertices = [Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in np.asarray(o["vertices_m"], dtype=float)]
            m.triangles = [MeshTriangle(vertex_indices=[int(a), int(b), int(c)]) for a, b, c in np.asarray(o["triangles"])]
            co.meshes = [m]
            co.mesh_poses = [matrix_to_pose(np.eye(4))]
            co.operation = CollisionObject.ADD
            ps.world.collision_objects.append(co)
        req = ApplyPlanningScene.Request()
        req.scene = ps
        r = call_service(self.scene_client, req, 10.0, wait_service_s=2.0)
        return bool(r is not None and r.success)

    def weld_tool_body(self):
        """The weld torch's tool for the weld-pose clearance (scan_core.weld_clearance): this robot's URDF (robot_description) collision
        geometry rigid with the weld torch's TCP - what MoveIt checks. Read once."""
        tb = getattr(self, "_weld_tool_body", None)
        if tb is None:
            from scan_core.weld_clearance import tool_body_from_urdf
            wt = self.mj.weld_torch
            if wt is None:
                raise ValueError("no weld_torch section in the platform config")
            self.chain()                                                  # robot_description 이 있어야 한다
            if f'name="{wt.tcp_frame}"' in (self.robot_description or ""):
                tb = tool_body_from_urdf(self.robot_description, wt.tcp_frame)
            else:                                                         # the TCP is not in the URDF: the tool on the flange, via TF
                tb = tool_body_from_urdf(self.robot_description, wt.tcp_frame, root_frame=wt.flange_frame,
                                         T_root_tcp=self._pose.static_transform(wt.flange_frame, wt.tcp_frame, 2.0))
            self._weld_tool_body = tb
        return tb

    def weld_reach_model(self):
        """The arm's necessary reach condition for the torch-orientation search (scan_core.weld_reach): this robot's chain (base →
        the motion flange) with the weld torch TCP on it from TF; None when the chain is not of that structure."""
        from scan_core.weld_reach import arm_reach_from_chain
        wt = self.mj.weld_torch
        if wt is None:
            return None
        return arm_reach_from_chain(self.chain(), self._pose.static_transform(self.mj.flange_frame, wt.tcp_frame, 2.0),
                                    source=f"robot_description {self.mj.base_frame} -> {self.mj.flange_frame}")

    def _contacts_at(self, T_base_flange: np.ndarray, seed_d: dict) -> str:
        """The contacts of the robot at a flange pose (IK without collision avoidance, then check_state_validity) → text ('' if none or
        no IK) — names what MoveIt's Cartesian path stopped on."""
        return self._contact_pairs_at(T_base_flange, seed_d)[0]

    def _contact_pairs_at(self, T_base_flange: np.ndarray, seed_d: dict) -> tuple:
        """(_contacts_at's text, [{a, b, kind: 'self' | 'world' | 'attached'}], ik: True when an IK solution without collision
        avoidance exists) at a flange pose."""
        latest = self.pose.latest_joint_state()
        measured = latest[1] if latest is not None else None
        q = GetPositionIK.Request()
        q.ik_request.group_name = self.mj.move_group_name
        q.ik_request.ik_link_name = self.mj.flange_frame
        q.ik_request.pose_stamped.header.frame_id = self.mj.base_frame
        q.ik_request.pose_stamped.pose = matrix_to_pose(T_base_flange)
        _fill_robot_state(q.ik_request.robot_state, measured, seed_d)          # full measured state, is_diff (attached bodies kept)
        q.ik_request.avoid_collisions = False
        q.ik_request.timeout.sec = 1
        ir = call_service(self.ik_client, q, 10.0, wait_service_s=2.0)
        if ir is None or ir.error_code.val != MoveItErrorCodes.SUCCESS:
            return "", [], False
        req = GetStateValidity.Request()
        req.group_name = self.mj.move_group_name
        _fill_robot_state(req.robot_state, measured, dict(zip(ir.solution.joint_state.name, ir.solution.joint_state.position)))
        r = call_service(self.validity_client, req, 5.0, wait_service_s=2.0)
        if r is None or r.valid:
            return "", [], True
        kinds = {}
        for c in r.contacts:                                         # moveit_msgs/ContactInformation: 0 link, 1 world object, 2 attached
            t1, t2 = int(c.body_type_1), int(c.body_type_2)
            kinds[(str(c.contact_body_1), str(c.contact_body_2))] = ("world" if 1 in (t1, t2) else
                                                                     "self" if t1 == 0 and t2 == 0 else "attached")
        pairs = sorted(kinds)
        return (", ".join(f"{a}-{b}" for a, b in pairs), [{"a": a, "b": b, "kind": kinds[(a, b)]} for a, b in pairs], True)

    def _acm_read(self):
        """move_group's allowed collision matrix (moveit_msgs AllowedCollisionMatrix), None if it could not be read."""
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        resp = call_service(self.get_scene_client, req, 5.0, wait_service_s=2.0)
        if resp is None or not list(resp.scene.allowed_collision_matrix.entry_names):
            return None
        return resp.scene.allowed_collision_matrix

    @staticmethod
    def _acm_with(base, links, object_ids):
        """A copy of the matrix base with contacts allowed between the robot links and the world objects (the weld-pose check: the link
        carrying the gas nozzle and contact tip may touch the grown specimen in MoveIt - those parts are checked per role by the exact
        model). MoveIt replaces the whole matrix with a scene diff's, so the copy carries every other entry as it was."""
        import copy
        names = list(base.entry_names)
        M = [list(e.enabled) for e in base.entry_values]
        for nm in list(links) + list(object_ids):
            if nm not in names:
                names.append(nm)
                for row in M:
                    row.append(False)
                M.append([False] * len(names))
        for a in links:
            for b in object_ids:
                i, j = names.index(a), names.index(b)
                M[i][j] = M[j][i] = True
        acm = copy.deepcopy(base)
        acm.entry_names = names
        acm.entry_values = [AllowedCollisionEntry(enabled=row) for row in M]
        return acm

    def _acm_set(self, acm) -> bool:
        ps = PlanningScene()
        ps.is_diff = True
        ps.allowed_collision_matrix = acm
        req = ApplyPlanningScene.Request()
        req.scene = ps
        r = call_service(self.scene_client, req, 10.0, wait_service_s=2.0)
        return bool(r is not None and r.success)

    def remove_scene_boxes(self, ids) -> bool:
        ps = PlanningScene()
        ps.is_diff = True
        for i in ids:
            co = CollisionObject()
            co.header.frame_id = self.mj.base_frame
            co.id = str(i)
            co.operation = CollisionObject.REMOVE
            ps.world.collision_objects.append(co)
        req = ApplyPlanningScene.Request()
        req.scene = ps
        r = call_service(self.scene_client, req, 10.0, wait_service_s=2.0)
        return bool(r is not None and r.success)

    def weld_line_step_m(self) -> float:
        """The sample step of the delivered weld lines D14 checks (scan_core.weld_line): this platform's Cartesian step - the
        interpolation MoveIt executes a straight move with (moveit_jtc.cartesian_step_m)."""
        return float(self.mj.cartesian_step_m)

    def _refusal(self, constraint: str, detail: str, pairs=None, links=None) -> dict:
        """A robot-check refusal record (weld_orient.refusal_of): constraint, the contact links and their kind, the text."""
        pairs = list(pairs or [])
        if pairs and constraint in ("Cartesian path", "IK"):
            constraint = ("arm vs world" if any(p_["kind"] == "world" for p_ in pairs) else
                          "self-collision" if all(p_["kind"] == "self" for p_ in pairs) else "collision (attached)")
        return {"constraint": constraint, "links": list(links or []) + [f"{p_['a']}-{p_['b']}" for p_ in pairs],
                "contacts": pairs, "detail": detail}

    def check_weld_plan(self, plan, *, approach_m: float = 0.03, only=None) -> list:
        """Can this robot follow the weld plan's DELIVERED lines - **no motion** (D14; user 2026-10-13: D14 checks exactly what is
        delivered and executed). Per pass the line of scan_core.weld_line.weld_line (the one straight TCP line at one orientation that
        kiro/pass_<k>.yaml carries: the plan's weld torch with its nozzle end and stick-out, the plan's TCP roll; the cap also its two
        weave edge lines, +- the WPS amplitude along y_TCP) moved to the flange with this robot's own TF flange <- TCP (weld torch on
        this arm only - scan_core.weld_torch.reach_tool; otherwise ValueError, the caller skips D14 by d14_on_this_arm), followed by
        move_group /compute_cartesian_path (IK, joint limits, self-collision and the world, max_step = cartesian_step_m) from the IK
        solution of the start pose approach_m back along the torch axis (seeded by the current joints; the next line / pass by the
        previous solution - the same IK branch; else the reach candidates' best: scan_l0.acquisition.rank_reach_candidates), then the
        dense IK along the same lines (cartesian_step_m; joint jump and limits). Workpiece (2026-10-13): per pass the specimen before
        the pass (scan_core.weld_scene: plates, groove, backing, plan.deposits' earlier beads) grown by the required clearance
        (plan.clearance required_mm) in the planning scene (hongik/weld/*; the caller cleared the scan scene); the link carrying the
        gas nozzle + contact tip may touch it in MoveIt (allowed collision matrix, restored) and the tool's parts are checked per role
        by the exact model on the same line samples (scan_core.weld_orient.pass_clearance, the plan's line_step_mm). The objects are
        removed at the end. → per pass {k, kind, reachable, fraction, max_joint_step_rad, max_step_joint, max_step_s_mm,
        min_limit_margin_rad, max_tcp_speed_mm_s (the layer plan's dense path - information), speed_over_limit, lines, elapsed_s,
        reason, refusal (constraint: IK | joint limit | joint jump | self-collision | arm vs world | clearance (exact model) | ...,
        links, detail), workpiece, tool, line (the checked line's poses)}. only = the passes to check (default all)."""
        mj = self.mj
        from scan_core.trajectory import nearest_turn_equivalent
        from scan_core.weld_line import weld_line
        from scan_core.weld_torch import plan_weld_torch, reach_tool, torch_with_nozzle_end
        wt_cfg = mj.weld_torch
        if wt_cfg is not None and wt_cfg.on_this_arm:                   # the stick-out's reference: this arm's torch nozzle end (URDF)
            wt_cfg = torch_with_nozzle_end(wt_cfg, self.weld_tool_body)
        wt_plan = getattr(plan, "torch", None) or plan_weld_torch(plan, wt_cfg)
        roll = float(getattr(plan, "torch_roll_deg", 0.0) or 0.0)

        def lookup(a, b):
            return self._pose.static_transform(a, b, 2.0)
        _T_ft, tool = reach_tool(wt_plan, lookup, mj.flange_frame, roll_deg=roll)    # this arm's torch only (and the tool's description)
        T_cf = np.linalg.inv(np.asarray(lookup(mj.flange_frame, wt_plan.tcp_frame), dtype=float))   # (flange <- TCP)^-1, the robot's TF
        chain = self.chain()
        names = list(chain.joint_names)
        lo = np.array([j.lower for j in chain.movable_joints])
        hi = np.array([j.upper for j in chain.movable_joints])
        js = self._pose.latest_joint_state()[1]
        seed = dict(zip(js.name, (float(v) for v in js.position)))

        def ik(T_base_flange, seed_d):
            q = GetPositionIK.Request()
            q.ik_request.group_name = mj.move_group_name
            q.ik_request.ik_link_name = mj.flange_frame
            q.ik_request.pose_stamped.header.frame_id = mj.base_frame
            q.ik_request.pose_stamped.pose = matrix_to_pose(T_base_flange)
            q.ik_request.robot_state.joint_state.name = list(seed_d)
            q.ik_request.robot_state.joint_state.position = list(seed_d.values())
            q.ik_request.avoid_collisions = bool(mj.avoid_collisions)
            q.ik_request.timeout.sec = 1
            return call_service(self.ik_client, q, 10.0, wait_service_s=2.0)

        from scan_core.weld_scene import pass_solids, scene_meshes
        from scan_core.weld_clearance import tool_roles
        from scan_core.weld_orient import pass_clearance
        r_mm = float((getattr(plan, "clearance", None) or {}).get("required_mm", 0.0) or 0.0)
        weld_ids: set = set()
        out, jobs = [], []
        acm_base = None                                 # move_group's matrix before the check (restored at the end)
        acm_ok = False
        roles = None
        try:
            for wp in plan.passes:
                if only is not None and wp.k not in set(only):
                    continue
                t0 = time.monotonic()
                ln = weld_line(plan, wp, wt_plan, roll)                     # THE delivered line (kiro/pass_<k>.yaml, result.paths)
                specimen = getattr(plan, "spec_design", None) is not None and getattr(plan, "levels_mm", None) is not None
                objs = scene_meshes(plan, pass_solids(plan, wp), r_mm) if specimen else []
                stale = [i_ for i_ in weld_ids if i_ not in {o["id"] for o in objs}]
                if stale:
                    self.remove_scene_boxes(stale)
                scene_ok = self.set_scene_meshes(objs) if objs else True
                weld_ids = {o["id"] for o in objs}
                # the tool parts by role (user 2026-10-13): the gas nozzle and contact tip need only not penetrate the nominal workpiece,
                # every other part k x sigma. MoveIt checks one grown specimen for every link: the link carrying the nozzle may touch it
                # there (allowed collision matrix) and that link's parts are checked per role by the exact model (weld_clearance - the
                # same URDF geometry) on the line's samples; every other link (the arm) keeps MoveIt's check at the grown specimen.
                exact = None
                if objs and scene_ok:
                    roles = roles or tool_roles(self.weld_tool_body())
                    # (known: the orientation search's exact record of this line - reused when it is the same evaluation, weld_orient)
                    exact = pass_clearance(plan, wp, roles, wt_plan, roll, required_mm=r_mm,
                                           known=(getattr(plan, "d14_exact", None) or {}).get(wp.k))
                    if roles.process_links:
                        acm_base = acm_base if acm_base is not None else self._acm_read()
                        acm_ok = acm_base is not None and self._acm_set(self._acm_with(acm_base, list(roles.process_links),
                                                                                         sorted(weld_ids)))
                P = np.asarray(wp.dense_pos_m, dtype=float)
                dt = np.diff(np.asarray(wp.dense_t_s, dtype=float))
                v = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(dt, 1e-9)
                vmax = float(np.max(v)) * 1e3 if v.size else float("nan")
                r = {"k": wp.k, "kind": wp.kind, "max_tcp_speed_mm_s": vmax, "speed_over_limit": bool(vmax > mj.max_tcp_speed_mps * 1e3 + 1e-6),
                     "reachable": True, "fraction": 1.0, "max_joint_step_rad": 0.0, "max_step_joint": "", "max_step_s_mm": float("nan"),
                     "min_limit_margin_rad": float("inf"), "lines": [], "reason": "", "tool": tool, "refusal": None,
                     "line": {"tcp_start_m": ln.tcp_start, "tcp_goal_m": ln.tcp_goal, "R_base_tcp": ln.R_tcp, "roll_deg": roll,
                              "length_mm": ln.length_m * 1e3, "edges": [n_ for n_, _o in ln.offsets()[1:]]},
                     "workpiece": ((f"MoveIt planning scene: the specimen before this pass ({', '.join(o['id'] for o in objs)}) grown by "
                                    f"the required clearance {r_mm:.2f} mm; Cartesian path of the delivered line with collision checking"
                                    + ("" if bool(mj.avoid_collisions) else " - avoid_collisions is OFF in the platform config: NOT checked")
                                    + (f"; {', '.join(roles.process_links)} (carries the gas nozzle + contact tip) allowed to touch the "
                                       "grown specimen in MoveIt and checked per part role by the exact model on the line's samples: "
                                       + ", ".join(f"{k_} {v_['clearance_mm']:.2f} mm (required {v_['required_mm']:.2f})"
                                                   for k_, v_ in exact["by_role"].items())
                                       if exact is not None and roles is not None and acm_ok else
                                       ("; exact model per part role on the line's samples: " + ", ".join(
                                           f"{k_} {v_['clearance_mm']:.2f} mm (required {v_['required_mm']:.2f})" for k_, v_ in exact["by_role"].items())
                                        + " (the allowed collision matrix could not be set: MoveIt checks the nozzle's link at the grown "
                                        "specimen too - conservative)" if exact is not None else "")))
                                   if objs else "not checked: the plan has no specimen model (a bare pose list)")}
                if exact is not None:
                    r["clearance_mm"] = float(exact["clearance_mm"])
                    r["clearance_by_role"] = {k_: float(v_["clearance_mm"]) for k_, v_ in exact["by_role"].items()}
                    r["line"].update(n_samples=exact.get("n_samples"), step_mm=(exact.get("checked_line") or {}).get("step_mm"))
                if not scene_ok:
                    r.update(reachable=None, reason="move_group did not take the weld-pose specimen (apply_planning_scene) - not checked")
                    r["elapsed_s"] = time.monotonic() - t0
                    out.append(r)
                    continue
                if exact is not None and exact["excess_mm"] < 0.0:
                    lim = exact["by_role"][exact["limiting_role"]]
                    link = ""
                    if roles is not None and lim.get("point_tcp_mm") is not None:
                        from scan_core.weld_clearance import part_link
                        link = part_link(dict(roles.roles())[exact["limiting_role"]], lim["point_tcp_mm"])
                    why = (f"the tool's {exact['limiting_role']} parts come within {lim['clearance_mm']:.2f} mm of the specimen "
                           f"({lim['solid']}, {lim['line']} line, s {float(lim.get('line_s_mm') or float('nan')):.0f} mm) - required "
                           f"{lim['required_mm']:.2f} mm")
                    r.update(reachable=False, fraction=0.0, reason=why,
                             refusal=self._refusal("clearance (exact model)", why, links=[link] if link else []))
                    r["refusal"]["part"] = exact["limiting_role"]
                    r["elapsed_s"] = time.monotonic() - t0
                    out.append(r)
                    continue
                for name, off in ln.offsets():
                    T_tcp = [ln.T_tcp("start"), ln.T_tcp("goal")]
                    for T in T_tcp:
                        T[:3, 3] += np.asarray(off, dtype=float)
                    A = T_tcp[0].copy()
                    A[:3, 3] -= ln.axis * float(approach_m)                     # back along the torch axis (away from the work)
                    # the start's IK solution: first seeded by the current joints (fast; the next line / pass by the previous solution - the
                    # same IK branch). If that solution cannot follow the line, the reach rule picks again (the acquisition's rule: of the
                    # solutions that follow the whole line by dense IK the one with the best singularity / limit margin, then the nearest to
                    # the current joints - scan_l0.acquisition.rank_reach_candidates).
                    Fl = [T @ T_cf for T in T_tcp]
                    waypoints = [matrix_to_pose(F) for F in Fl]

                    def cartesian(start_d):
                        req = GetCartesianPath.Request()
                        req.header.frame_id = mj.base_frame
                        req.start_state.joint_state.name = list(start_d)
                        req.start_state.joint_state.position = list(start_d.values())
                        req.start_state.is_diff = True
                        req.group_name = mj.move_group_name
                        req.link_name = mj.flange_frame
                        req.waypoints = waypoints
                        req.max_step = float(mj.cartesian_step_m)
                        req.jump_threshold = 0.0
                        req.avoid_collisions = bool(mj.avoid_collisions)
                        return call_service(self.cart_client, req, 60.0, wait_service_s=2.0, resend=False)

                    def verdict(resp_):
                        f_ = float(resp_.fraction)
                        if resp_.error_code.val != MoveItErrorCodes.SUCCESS:
                            return f_, f"MoveIt error {resp_.error_code.val}", None
                        if f_ < mj.min_fraction:
                            # where it stopped: the path runs approach -> start -> goal; the stop pose on that polyline
                            L_a, L_l = float(approach_m), ln.length_m
                            d_ = f_ * (L_a + L_l)
                            T_stop = (A + (T_tcp[0] - A) * min(d_ / max(L_a, 1e-12), 1.0) if d_ < L_a else
                                      T_tcp[0] + (T_tcp[1] - T_tcp[0]) * min((d_ - L_a) / max(L_l, 1e-12), 1.0))
                            T_stop[:3, :3] = T_tcp[0][:3, :3]
                            who, pairs, _ok = self._contact_pairs_at(T_stop @ T_cf, seed)
                            at = ("on the descent onto the line start" if d_ < L_a else
                                  f"at s {ln.s_start_mm + (ln.s_goal_mm - ln.s_start_mm) * (d_ - L_a) / max(L_l, 1e-12):.0f} mm")
                            why_ = (f"{name} line fraction {f_:.4f} < {mj.min_fraction} {at}" + (f" (contacts: {who})" if who else ""))
                            return f_, why_, self._refusal("Cartesian path", why_, pairs)
                        return f_, "", None
                    start, resp, frac, why, rf = None, None, 0.0, "", None
                    ir = ik(A @ T_cf, seed)
                    if ir is not None and ir.error_code.val == MoveItErrorCodes.SUCCESS:
                        start = dict(seed)
                        start.update(zip(ir.solution.joint_state.name, (float(x) for x in ir.solution.joint_state.position)))
                        # the IK picks no joint representation (a joint whose range spans more than a turn has q and q +- 2 pi at the same
                        # link pose): the line starts in the one nearest the seed - the rule of the acquisition candidates (KIRO stand-in
                        # 2026-10-13: wrist1 returned at +2 pi - 0.004 rad ran into its +6.283 limit along a line it follows with margin at 0)
                        q_near = nearest_turn_equivalent(self._chain_q(start), self._chain_q(seed), lo, hi)
                        start.update(zip(names, (float(x) for x in q_near)))
                        resp = cartesian(start)
                        if resp is not None:
                            frac, why, rf = verdict(resp)
                    cand_fails = []
                    if start is None or resp is None or why:
                        try:
                            from scan_l0.acquisition import rank_reach_candidates
                            cands = self.reach_candidates(A @ T_cf, Fl, q_now=seed, avoid_collisions=bool(mj.avoid_collisions))
                            cand_fails = sorted({str(c_.get("reason", ""))[:160] for c_ in cands if not c_.get("ok")})
                            ranked = rank_reach_candidates(cands)
                        except Exception:  # noqa: BLE001 - keep the first attempt's verdict
                            ranked = []
                        if ranked:
                            alt = dict(seed)
                            alt.update(zip(ranked[0]["names"], (float(x) for x in ranked[0]["q"])))
                            resp_alt = cartesian(alt)
                            if resp_alt is not None:
                                start, resp = alt, resp_alt
                                frac, why, rf = verdict(resp)
                    if start is None:
                        if ir is None:
                            why = f"move_group {mj.ik_service} did not answer"
                            rf = self._refusal("not answered", why)
                        else:
                            who, pairs, ik_free = self._contact_pairs_at(A @ T_cf, seed)
                            why = (f"no IK for the {name} line start (torch {approach_m * 1e3:.0f} mm above the pass start): "
                                   + (f"its IK solutions are in collision (contacts: {who})" if who else
                                      "no IK solution within the joint limits (the arm's reach / joint limits)" if not ik_free else
                                      "IK without collision avoidance found a valid pose, with collision avoidance none")
                                   + (f"; follow candidates: {'; '.join(cand_fails[:2])}" if cand_fails else ""))
                            rf = (self._refusal("IK", why, pairs) if pairs else
                                  self._refusal("joint limit" if any("URDF limit" in f_ for f_ in cand_fails) else "IK", why))
                        r.update(reachable=False, fraction=0.0, reason=why, refusal=rf)
                        r["lines"].append({"line": name, "reachable": False, "reason": why})
                        break
                    if resp is None:
                        why = f"move_group {mj.cartesian_path_service} did not answer"
                        r.update(reachable=False, fraction=0.0, reason=why, refusal=self._refusal("not answered", why))
                        r["lines"].append({"line": name, "reachable": False, "reason": why})
                        break
                    seed = start
                    # breaks and limits: the dense IK (cartesian_step_m) of the moves (move_linear) along the same straight legs - the
                    # move_group answer is resampled at 0.1 s, so a break cannot be judged from neighbouring points. All lines in the
                    # process pool together (below)
                    q0 = np.array([start[n_] for n_ in names])
                    jobs.append((r, name, frac, why, q0, Fl, np.array([ln.s_start_mm, ln.s_goal_mm]), rf))
                    if why:
                        r.update(reachable=False, reason=why, refusal=rf)
                        break
                r["elapsed_s"] = time.monotonic() - t0
                out.append(r)
        finally:                                        # the weld-pose specimen never stays in the scene, nor its allowed contacts
            if acm_base is not None:
                self._acm_set(acm_base)
            if weld_ids:
                self.remove_scene_boxes(sorted(weld_ids))
        # dense IK - spawn process pool (no GIL contention with the server process; every line independent). It does not block the run:
        # a line without an answer is not checked (below)
        t1 = time.monotonic()
        tracks, unchecked, pool_note = dense_tracks([(chain, q0, poses, mj.cartesian_step_m, lo, hi, mj.max_joint_step_rad)
                                                     if not why else None for (r, name, frac, why, q0, poses, S, rf) in jobs])
        if pool_note:                                   # the process pool ran without part of its lifetime rule (scan_core.proc_pool)
            for r in out:
                r["pool_note"] = pool_note
        for i, ((r, name, frac, why, q0, poses, S, rf), tr) in enumerate(zip(jobs, tracks)):
            if i in unchecked:                          # a line not checked: reachable None (a checked False wins)
                r["lines"].append({"line": name, "reachable": None, "fraction": frac, "n_waypoints": len(poses), "reason": unchecked[i]})
                if r["reachable"] is not False:
                    r.update(reachable=None, reason=f"reachability check not done: {unchecked[i]}")
                continue
            at_j, at_s, step, margin = "", float("nan"), float("nan"), float("nan")
            if tr is not None:
                Q = tr["Q"]
                dQ = np.abs(np.diff(Q, axis=0)) if Q.shape[0] > 1 else np.zeros((0, len(names)))
                step = float(dQ.max()) if dQ.size else float("nan")
                margin = float(min(np.min(Q - lo), np.min(hi - Q))) if Q.size else float("nan")
                if dQ.size:
                    i_pt, j = np.unravel_index(int(np.argmax(dQ)), dQ.shape)
                    at_j = names[int(j)]
                    at_s = float(S[tr["seg"][int(i_pt)]]) if int(i_pt) < len(tr["seg"]) else float("nan")
                if not tr["ok"]:
                    leg = "the descent onto the line start" if int(tr["i_fail"]) <= 0 else "the line"
                    why = f"{name} line, {leg}: {tr['reason']}"
                    low_ = tr["reason"].lower()
                    rf = self._refusal("joint limit" if "urdf limit" in low_ else "joint jump" if "jump" in low_ else "IK", why)
            ok = not why
            r["lines"].append({"line": name, "reachable": ok, "fraction": frac, "max_joint_step_rad": step, "max_step_joint": at_j,
                               "min_limit_margin_rad": margin, "n_waypoints": len(poses), "reason": why})
            r["fraction"] = min(r["fraction"], frac)
            if np.isfinite(step) and step >= r["max_joint_step_rad"]:
                r.update(max_joint_step_rad=step, max_step_joint=at_j, max_step_s_mm=at_s)
            if np.isfinite(margin):
                r["min_limit_margin_rad"] = min(r["min_limit_margin_rad"], margin)
            if not ok:
                r.update(reachable=False, reason=why, refusal=rf)
        for r in out:
            r["elapsed_s"] = float(r.get("elapsed_s", 0.0)) + (time.monotonic() - t1) / max(1, len(out))
        return out


    # ---- 포트 ----
    @property
    def pose(self) -> RosPoseSource:
        return self._pose

    @property
    def backend(self) -> Optional[MoveItJtcBackend]:
        return self._backend

    @property
    def latch(self) -> RosStopLatch:
        return self._latch

    @property
    def gates(self) -> MoveItGates:
        return self._gates

    @property
    def busy(self) -> bool:
        return bool(self._backend is not None and self._backend.busy)

    def arm(self, cfg: MotionConfig):
        if self._backend is None:
            self._backend = MoveItJtcBackend(self, cfg)
        else:
            self._backend.cfg = cfg
        return self._backend

    def close(self) -> None:
        self.ctx.close()

    # ---- 상태 · 준비 ----
    @property
    def robot_description(self) -> Optional[str]:
        return self._pose.robot_description

    def wait_ready(self, timeout_s: float = 10.0) -> list:
        """move_group 서비스·JTC 액션·/joint_states·자세 (robot_description 체인 FK — 체인 관절이 든 새 /joint_states) 준비 → 빠진 것 목록
        (빈 목록 = 준비됨). TF base←flange 는 보지 않는다 (자세는 TF 를 기다리지 않는다, 2026-10-13)."""
        mj = self.mj
        t_end = time.monotonic() + timeout_s
        missing: list = []
        checks = [
            lambda: None if self.cart_client.service_is_ready() else f"move_group service {mj.cartesian_path_service}",
            lambda: None if self.fk_client.service_is_ready() else f"move_group service {mj.fk_service}",
            lambda: None if self.jtc_client.server_is_ready() else f"JTC action {mj.jtc_action}",
            self._pose.fk_problem,
        ]
        while True:
            missing = [m for m in (f() for f in checks) if m]
            if not missing or time.monotonic() >= t_end:
                return missing
            time.sleep(0.05)

    def readiness(self, timeout_s: float = 2.0) -> list:
        """지금 실행을 시작할 수 있는가 → 준비가 안 된 항목 목록 (빈 목록 = 준비됨): MoveIt/JTC 를 쓸 수 있는가만 — move_group 서비스·
        JTC 액션·/joint_states (받은 적 있음)·TF base←flange (wait_ready 와 같은 항목) 와 우리 이동 없음. 로봇 제어기의 상태는
        읽지 않는다 (제어기·안전 계층은 플랫폼의 것)."""
        out = list(self.wait_ready(timeout_s=0.0))
        if self.busy:
            out.append("a motion of ours is still in progress")
        return out

    # ---- 캘리브 · 중력 (TF 에서 유도) ----
    def calib_for_platform(self, file_calib: Calib, timeout_s: float = 5.0) -> Calib:
        """T_flange←opt (정적 TF 포함) 와 T_flange←eoat 를 이 플랫폼의 TF 에서 읽은 Calib (dt_s·x_sign 은 파일) — 정적 변환
        (RosPoseSource.static_transform: 한 번 읽고 정적·URDF 대조 확인 뒤 캐시) + 툴 형상을 이 팔의 로봇 모형에서 (tool_geometry): 몸체 =
        robot_description 에서 flange_frame 에 달린 충돌 형상 (MoveIt 이 검사하는 그 형상) + 설정 safety.tool_body_boxes, 토치 = 용접 토치가
        이 팔에 있으면 TF 의 그 TCP 와 그 노즐 끝. 모형에 없으면 설정 값 (attach_tool_body, 작업 torch_lateral_m · safety.torch_protrusion_m)."""
        mj = self.mj
        T_fo = self._pose.static_transform(mj.flange_frame, mj.optical_frame, timeout_s)
        T_fe = self._pose.static_transform(mj.flange_frame, mj.eoat_frame, timeout_s)
        calib = calib_from_tf(file_calib, T_fo, T_fe, note=f"{mj.flange_frame}<-{mj.optical_frame}")
        return self.tool_geometry(calib, timeout_s)

    def tool_geometry(self, calib: Calib, timeout_s: float = 5.0) -> Calib:
        """The tool's geometry for the motion envelope from this arm's own robot model (one source with MoveIt's collision model and
        D14), in the eoat frame of ``calib``:
          body       the collision geometry mounted on flange_frame in robot_description (fk_urdf.mounted_collision_points: the
                     fixed-joint subtree below the flange, hull points) + the configured parts the model does not hold
                     (safety.tool_body_boxes); none in the model = the configured mesh (attach_tool_body, safety.tool_body_mesh);
          torch      when the weld torch is on this arm (weld_torch.on_this_arm): its TCP frame with the tip axes (TF flange_frame <-
                     tcp_frame . tip_in_tcp axes) and its gas nozzle end (this robot's URDF collision geometry, the D14 tool:
                     weld_torch.torch_with_nozzle_end) - the safety policy puts the wire end = TCP + (nozzle end + the job's process
                     stick-out) along the torch axis into the envelope's torch keypoint, the point the weld plan aims and D14 checks;
                     otherwise the job's torch_lateral_m and safety.torch_protrusion_m.
        Both are recorded in the calib (body_source, torch_tip_source)."""
        import io
        from scan_core.fk_urdf import mounted_collision_points
        from scan_core.weld_torch import d14_on_this_arm, torch_with_nozzle_end
        from scan_l0.config import tool_body_box_points
        mj = self.mj
        T_ef = np.linalg.inv(np.asarray(calib.T_flange_eoat, dtype=float))
        if calib.body_points_eoat is None:
            key = (mj.flange_frame, hash(self.robot_description or ""))
            cache = getattr(self, "_mounted_cache", None)
            if cache is None or cache[0] != key:
                self.chain()                                              # robot_description must be here (else MotionError)
                cache = (key, mounted_collision_points(io.StringIO(self.robot_description), mj.flange_frame))
                self._mounted_cache = cache
            m = cache[1]
            if m["points"] is not None:
                B = m["points"] @ T_ef[:3, :3].T + T_ef[:3, 3]
                extra = tool_body_box_points(self.settings)
                calib.body_points_eoat = B if extra is None else np.vstack([B, extra])
                calib.body_source = (f"robot model: collision geometry mounted on {mj.flange_frame} ({', '.join(m['links'])}) hull "
                                     + (f"+ {len(extra) // 8} configured box(es) " if extra is not None else "")
                                     + f"{calib.body_points_eoat.shape[0]} pts" + (f"; not read: {'; '.join(m['why'])}" if m["why"] else ""))
        on, _why = d14_on_this_arm(mj.weld_torch)
        if on:
            wt = torch_with_nozzle_end(mj.weld_torch, self.weld_tool_body)   # ValueError when the model gives no nozzle end
            T_ft = self._pose.static_transform(mj.flange_frame, wt.tcp_frame, timeout_s) @ wt.tcp_from_tip.matrix()
            calib.torch_tcp_eoat = T_ef @ T_ft
            calib.torch_nozzle_end_mm = float(wt.nozzle_end_mm)
            calib.torch_tip_source = (f"the weld torch: TF {mj.flange_frame} <- {wt.tcp_frame} . weld_torch.tip_in_tcp axes "
                                      f"({wt.tcp_from_tip.status or wt.tcp_from_tip.source}); nozzle end {wt.nozzle_end_source}")
        return calib

    def allowed_start_tolerance(self, timeout_s: float = 2.0) -> tuple:
        """move_group's start-state tolerance [rad] - trajectory_execution.allowed_start_tolerance of the move_group node (in the
        namespace of the planning service), the rule move_group itself applies before executing a trajectory; read once. Not readable,
        or 0 (move_group's own check disabled) -> MoveIt's documented default 0.01 rad. -> (value, source text)."""
        hit = getattr(self, "_start_tol", None)
        if hit is not None:
            return hit
        from rcl_interfaces.srv import GetParameters
        name = "trajectory_execution.allowed_start_tolerance"
        srv = self.mj.plan_service.rsplit("/", 1)[0] + "/move_group/get_parameters"
        val, why = None, ""
        try:
            cl = self.ctx.node.create_client(GetParameters, srv, callback_group=self.cb)
            req = GetParameters.Request()
            req.names = [name]
            resp = call_service(cl, req, timeout_s, wait_service_s=min(1.0, timeout_s))
            if resp is None or not resp.values:
                why = f"{srv} not available"
            elif resp.values[0].type == 3 and float(resp.values[0].double_value) > 0.0:     # PARAMETER_DOUBLE
                val = float(resp.values[0].double_value)
            else:
                why = f"{name} not set or 0 on move_group"
        except Exception as e:  # noqa: BLE001 - the documented default stands in, and says so
            why = f"{type(e).__name__}: {e}"
        self._start_tol = (val, f"move_group {name}") if val is not None else \
            (0.01, f"MoveIt default allowed_start_tolerance 0.01 rad ({why})")
        return self._start_tol

    def weld_torch_model(self, torch, timeout_s: float = 2.0):
        """The weld torch as the running robot model has it (the lab torch turned in its clamp, 2026-10-13: the URDF is the tool): for a
        torch on this arm, flange_from_tcp = TF flange_frame <- tcp_frame (static, checked against robot_description) - the record the
        run writes (weld_plan.json weld_torch, kiro/ info and README) is the torch that is mounted. The platform config's
        flange_from_tcp is a record of the same frame: when it differs from the robot model by more than the same-frame tolerance
        (path_check_tol_m / FRAME_ROT_TOL_DEG, as TF vs URDF) → ValueError (the run is refused: config and robot model disagree on the
        torch). No torch / not on this arm → the torch as it is; a TCP frame the TF lacks → MotionError."""
        import dataclasses as _dc
        from scan_core.rigid import RigidTransform
        if torch is None or not getattr(torch, "on_this_arm", False):
            return torch
        T = self._pose.static_transform(torch.flange_frame, torch.tcp_frame, timeout_s)
        cfg = torch.flange_from_tcp
        dp, dr = pose_difference(cfg.matrix(), T)
        if dp > self.mj.path_check_tol_m or dr > FRAME_ROT_TOL_DEG:
            a, b = cfg.t * 1e3, T[:3, 3] * 1e3
            raise ValueError(f"the platform config's weld_torch.flange_from_tcp ({a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f}) mm is not the robot "
                             f"model's {torch.flange_frame} <- {torch.tcp_frame} ({b[0]:+.2f}, {b[1]:+.2f}, {b[2]:+.2f}) mm ({dp * 1e3:.2f} mm "
                             f"/ {dr:.3f} deg apart, same-frame tolerance {self.mj.path_check_tol_m * 1e3:.2f} mm / {FRAME_ROT_TOL_DEG} deg): "
                             "the config record and the URDF describe different torches - update the one that is not the mounted tool")
        sha = ""
        try:
            import hashlib
            sha = hashlib.sha256((self.robot_description or "").encode("utf-8")).hexdigest()[:12]
        except Exception:  # noqa: BLE001 - the source text only
            pass
        src = (f"the running robot model: TF {torch.flange_frame} <- {torch.tcp_frame} (robot_description sha256 {sha}); the platform "
               f"config's record ({cfg.source or 'flange_from_tcp'}) within {dp * 1e3:.3f} mm / {dr:.4f} deg")
        rt = RigidTransform(R=np.asarray(T[:3, :3], dtype=float), t=np.asarray(T[:3, 3], dtype=float).copy(), source=src,
                            status="the robot model of this launch (nominal CAD; KIRO re-calibrates its TCP)")
        return _dc.replace(torch, flange_from_tcp=rt)

    def planning_frame(self, timeout_s: float = 5.0) -> tuple:
        """MoveIt 의 계획 프레임 (move_group 의 모형 프레임 — 가상 관절의 부모, 없으면 URDF 루트) 을 실행 때 읽는다: /get_planning_scene
        (로봇 상태 + 고정 변환) 의 프레임. → (이름 | None, 출처 또는 못 읽은 이유)."""
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ROBOT_STATE | PlanningSceneComponents.TRANSFORMS
        resp = call_service(self.get_scene_client, req, timeout_s, wait_service_s=min(2.0, timeout_s))
        if resp is None:
            return None, f"no answer from {self.mj.get_scene_service}"
        sc = resp.scene
        for fr, where in ((sc.robot_state.joint_state.header.frame_id, "robot_state"),
                          (sc.robot_state.multi_dof_joint_state.header.frame_id, "multi_dof_joint_state"),
                          (sc.fixed_frame_transforms[0].header.frame_id if len(sc.fixed_frame_transforms) else "", "fixed_frame_transforms")):
            if str(fr).strip():
                return str(fr).strip().lstrip("/"), f"MoveIt planning frame ({self.mj.get_scene_service} {where})"
        return None, f"{self.mj.get_scene_service} gave no frame"

    def frame_transform(self, target: str, source: str, timeout_s: Optional[float] = None) -> np.ndarray:
        """TF target ← source (4×4) 정적 변환 — 출력의 프레임 사슬 (KIRO 형식 출력의 World ← base; RosPoseSource.static_transform:
        한 번 읽고 확인·캐시). 프레임이 없거나 정적이 아니면 예외 (호출자가 설정 값으로 대신한다)."""
        return self._pose.static_transform(str(target), str(source), float(self.mj.tf_timeout_s if timeout_s is None else timeout_s))

    def gravity_axis_base(self, timeout_s: float = 5.0) -> np.ndarray:
        """중력 방향 (base 프레임) = R_base←ground · gravity.axis (지면 프레임을 KIRO 가 TF 로 준다; 손으로 적지 않는다)."""
        mj = self.mj
        T = self._pose.static_transform(mj.base_frame, mj.gravity_frame, timeout_s)
        return unit(T[:3, :3] @ np.asarray(mj.gravity_axis, dtype=float))
