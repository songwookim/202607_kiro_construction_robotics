"""직선 Cartesian 경로: 조밀한 관절 경로 (IK 추종) + 시간 매개화 (MoveIt 어댑터, ROS 없음 — 단위 시험 가능).

move_group 의 /compute_cartesian_path 는 직선을 max_step 간격의 IK 점으로 풀지만 (MoveIt cartesian_interpolator: 비율 i/N, slerp +
선형 병진), 응답의 궤적은 TOTG 로 0.1 s 간격 **재표본** 된 몇 개의 점뿐이다 (50 mm → 7 점). 그 점들 사이를 JTC 가 관절 공간에서
보간하면 직선에서 벗어난다. 그래서 어댑터는 move_group 응답을 **타당성 게이트** (fraction ≥ 0.999, 충돌·점프 검사, IK 분기) 로만 쓰고,
실행 궤적은 여기서 같은 직선을 같은 규칙으로 조밀하게 다시 푼다:

  1. track_straight_line: robot_description (URDF) 의 base→flange 체인 FK 와 감쇠 최소제곱 IK 를 이전 해에서 시작해 단계마다
     수렴 (위치 1e-6 m, 자세 1e-6 rad) — MoveIt 과 같은 단계 수 (max(병진/step, 회전/(3.5·step)) + 1) 와 비율 i/N.
     관절 점프 (max_joint_step_rad) 와 URDF 관절 한계를 넘으면 거부. 어댑터는 move_group 의 점들이 이 경로 위에 있는지 (같은 IK 분기)
     확인한다 (on_path_distance).
  2. retime_straight_path: s_i = (i/N)·S, S = max(병진 L, 회전 Θ · rotation_radius_m) → 사다리꼴 속도 (가속 a → 등속 v → 감속) 로
     t(s). 관절 속도·가속 한계 (플랫폼 선언값) 를 넘으면 시간 전체를 균등하게 늘린다 (k ≥ 1). 등속 구간에서 TCP 속도 = v
     (사용자 허용: 간격이 고르지 않을 수 있음 — 재구성은 실측 자세를 쓴다).
  3. 점별 관절 속도 = 중앙 차분 (처음·끝 0) → JTC 3차 스플라인.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = ["densify_joint_path", "retime_joint_path", "TrapezoidProfile", "RetimedPath", "retime_straight_path", "straight_line_model", "path_check", "track_straight_line", "LineTrackError",
           "on_path_distance", "rotation_vector", "interpolate_pose", "cartesian_steps", "fast_fk", "joint_path_stats",
           "follow_candidate", "follow_candidate_job", "nearest_turn_equivalent"]


@dataclass(frozen=True)
class TrapezoidProfile:
    """경로 길이 S [m], 속도 v, 가속 a → t(s). 삼각형이면 v_peak < v."""

    S: float
    v: float
    a: float

    @property
    def triangular(self) -> bool:
        return self.S < self.v * self.v / self.a

    @property
    def v_peak(self) -> float:
        return math.sqrt(self.S * self.a) if self.triangular else self.v

    @property
    def t_acc(self) -> float:
        return self.v_peak / self.a

    @property
    def s_acc(self) -> float:
        return 0.5 * self.a * self.t_acc ** 2

    @property
    def duration(self) -> float:
        if self.S <= 0.0:
            return 0.0
        if self.triangular:
            return 2.0 * self.t_acc
        return 2.0 * self.t_acc + (self.S - 2.0 * self.s_acc) / self.v

    def time_at(self, s: float) -> float:
        s = min(max(float(s), 0.0), self.S)
        vp, ta, sa = self.v_peak, self.t_acc, self.s_acc
        if s <= sa:
            return math.sqrt(2.0 * s / self.a)
        if s <= self.S - sa:
            return ta + (s - sa) / vp
        rem = self.S - s
        return self.duration - math.sqrt(max(0.0, 2.0 * rem / self.a))

    def speed_at(self, s: float) -> float:
        t = self.time_at(s)
        if t <= self.t_acc:
            return self.a * t
        if t >= self.duration - self.t_acc:
            return max(0.0, self.a * (self.duration - t))
        return self.v_peak


@dataclass
class RetimedPath:
    times: np.ndarray                 # (N+1,) [s] (0 … duration)
    positions: np.ndarray             # (N+1, J) [rad]
    velocities: np.ndarray            # (N+1, J) [rad/s]
    s: np.ndarray                     # (N+1,) 경로 매개변수 [m]
    S: float
    v_cruise: float                   # 시간 늘림 뒤 실제 등속 구간 TCP(경로) 속도 [m/s]
    a: float
    time_scale: float = 1.0           # ≥ 1 (관절 한계로 늘린 배율)
    max_joint_velocity: float = 0.0
    max_joint_accel: float = 0.0
    max_joint_step: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.times[-1]) if self.times.size else 0.0


def _central_diff(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = q.shape[0]
    v = np.zeros_like(q)
    if n >= 3:
        dt = (t[2:] - t[:-2])[:, None]
        v[1:-1] = (q[2:] - q[:-2]) / np.where(dt > 1e-9, dt, 1e-9)
    return v


def duration_parts(t_s: float) -> tuple:
    """초 → (sec, nanosec) — 전체 나노초로 반올림한 뒤 나눈다. 예전 식 (sec = int(t), nanosec = round(소수부·1e9) % 1e9) 은 정수 초 바로 아래
    (예 2.9999999997 s) 에서 nanosec 이 1e9 → 0 으로 감겨 2.000 s 가 되어 시각이 뒤로 갔다 → JTC 가 'not strictly increasing' 으로 목표를
    거부하고 정지 래치 (2026-10-06 가짜 팔 박스 캘리브)."""
    ns = int(round(float(t_s) * 1e9))
    return ns // 1_000_000_000, ns % 1_000_000_000


def retime_straight_path(positions: Sequence[Sequence[float]], length_m: float, rotation_rad: float, v_mps: float, a_mps2: float, *,
                         rotation_radius_m: float = 0.2, max_joint_velocity: Optional[Sequence[float]] = None,
                         max_joint_accel: Optional[Sequence[float]] = None, max_joint_step_rad: float = 0.1,
                         min_dt_s: float = 1e-3) -> RetimedPath:
    """MoveIt Cartesian 웨이포인트 (균등 비율) → 사다리꼴 시간 매개화. 관절 점프·잘못된 입력은 ValueError (거부)."""
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2:
        raise ValueError(f"need at least 2 waypoints, got shape {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("non-finite joint positions in the Cartesian path")
    if not (math.isfinite(v_mps) and v_mps > 0 and math.isfinite(a_mps2) and a_mps2 > 0):
        raise ValueError(f"speed/accel must be > 0, got {v_mps}/{a_mps2}")
    N = q.shape[0] - 1
    J = q.shape[1]
    steps = np.abs(np.diff(q, axis=0))
    max_step = float(steps.max()) if steps.size else 0.0
    if max_step > float(max_joint_step_rad):
        i = int(np.unravel_index(int(np.argmax(steps)), steps.shape)[0])
        raise ValueError(f"joint-space jump {max_step:.3f} rad between waypoints {i} and {i + 1} (> {max_joint_step_rad} rad): "
                         "configuration flip along the straight line - refusing")
    S = max(float(length_m), float(rotation_rad) * float(rotation_radius_m), 1e-9)
    prof = TrapezoidProfile(S=S, v=float(v_mps), a=float(a_mps2))
    s = np.linspace(0.0, S, N + 1)
    t = np.array([prof.time_at(x) for x in s], dtype=float)
    # 수치: 엄격 증가 (JTC 는 같은 시각의 점을 거부)
    for i in range(1, t.size):
        if t[i] <= t[i - 1] + min_dt_s * 1e-3:
            t[i] = t[i - 1] + min_dt_s * 1e-3
    vel = _central_diff(q, t)
    acc = _central_diff(vel, t)
    notes: list = []
    k = 1.0
    vmax = None if max_joint_velocity is None else np.broadcast_to(np.asarray(max_joint_velocity, dtype=float), (J,))
    amax = None if max_joint_accel is None else np.broadcast_to(np.asarray(max_joint_accel, dtype=float), (J,))
    if vmax is not None:
        r = float(np.max(np.abs(vel) / vmax[None, :]))
        if r > 1.0:
            k = max(k, r)
    if amax is not None:
        r = float(np.max(np.abs(acc) / amax[None, :]))
        if r > 1.0:
            k = max(k, math.sqrt(r))
    if k > 1.0 + 1e-9:
        k *= 1.02                                   # 여유 2 %
        t = t * k
        vel = vel / k
        acc = acc / (k * k)
        notes.append(f"time-scaled x{k:.3f} to respect the declared joint velocity/acceleration limits "
                     f"(TCP cruise {prof.v_peak / k * 1e3:.2f} mm/s instead of {prof.v_peak * 1e3:.2f} mm/s)")
    vel[0] = 0.0
    vel[-1] = 0.0
    return RetimedPath(times=t, positions=q, velocities=vel, s=s, S=S, v_cruise=prof.v_peak / k, a=prof.a / (k * k), time_scale=k,
                       max_joint_velocity=float(np.max(np.abs(vel))) if vel.size else 0.0,
                       max_joint_accel=float(np.max(np.abs(acc))) if acc.size else 0.0, max_joint_step=max_step, notes=notes)


def densify_joint_path(waypoints: Sequence[Sequence[float]], max_step_rad: float = 0.01) -> np.ndarray:
    """관절 웨이포인트 (MoveIt 계획 또는 시작·끝 두 점) → 이웃 점 사이 관절 변화가 max_step_rad 이하가 되게 선형 보간한 (N, J)."""
    W = np.asarray(waypoints, dtype=float)
    if W.ndim != 2 or W.shape[0] < 2:
        raise ValueError(f"need at least 2 joint waypoints, got shape {W.shape}")
    out = [W[0]]
    for a_, b_ in zip(W[:-1], W[1:]):
        n = max(1, int(math.ceil(float(np.max(np.abs(b_ - a_))) / float(max_step_rad))))
        for k in range(1, n + 1):
            out.append(a_ + (b_ - a_) * (k / n))
    return np.asarray(out)


def retime_joint_path(Q: Sequence[Sequence[float]], fk, v_mps: float, a_mps2: float, *, rotation_radius_m: float = 0.2,
                      max_joint_velocity: Optional[Sequence[float]] = None, max_joint_accel: Optional[Sequence[float]] = None,
                      min_dt_s: float = 1e-3) -> RetimedPath:
    """조밀한 관절 경로 (densify_joint_path) → 사다리꼴 시간 매개화 (관절 공간 이동, 2026-10-07: 접힌 자세에서 획득 자세로). 경로 변수
    s = 플랜지 이동 거리와 회전 × rotation_radius 중 큰 것의 누적 (점마다 fk) — TCP 속도 v·가속 a 를 넘지 않고, 그 위에 선언된 관절 속도·가속
    한계로 시간을 늘린다 (retime_straight_path 와 같은 규칙)."""
    q = np.asarray(Q, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2 or not np.all(np.isfinite(q)):
        raise ValueError(f"need at least 2 finite joint points, got shape {q.shape}")
    if not (math.isfinite(v_mps) and v_mps > 0 and math.isfinite(a_mps2) and a_mps2 > 0):
        raise ValueError(f"speed/accel must be > 0, got {v_mps}/{a_mps2}")
    J = q.shape[1]
    T = [fk(x) for x in q]
    ds = [0.0]
    for Ta, Tb in zip(T[:-1], T[1:]):
        dp = float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3]))
        dr = float(np.linalg.norm(rotation_vector(Ta[:3, :3].T @ Tb[:3, :3]))) * float(rotation_radius_m)
        ds.append(max(dp, dr, 1e-9))
    s = np.cumsum(ds)
    S = float(s[-1])
    prof = TrapezoidProfile(S=S, v=float(v_mps), a=float(a_mps2))
    t = np.array([prof.time_at(x) for x in s], dtype=float)
    for i in range(1, t.size):
        if t[i] <= t[i - 1] + min_dt_s * 1e-3:
            t[i] = t[i - 1] + min_dt_s * 1e-3
    vel = _central_diff(q, t)
    acc = _central_diff(vel, t)
    notes: list = []
    k = 1.0
    vmax = None if max_joint_velocity is None else np.broadcast_to(np.asarray(max_joint_velocity, dtype=float), (J,))
    amax = None if max_joint_accel is None else np.broadcast_to(np.asarray(max_joint_accel, dtype=float), (J,))
    if vmax is not None:
        r = float(np.max(np.abs(vel) / vmax[None, :]))
        if r > 1.0:
            k = max(k, r)
    if amax is not None:
        r = float(np.max(np.abs(acc) / amax[None, :]))
        if r > 1.0:
            k = max(k, math.sqrt(r))
    if k > 1.0 + 1e-9:
        k *= 1.02
        t = t * k
        vel = vel / k
        acc = acc / (k * k)
        notes.append(f"time-scaled x{k:.3f} to respect the declared joint velocity/acceleration limits "
                     f"(TCP cruise {prof.v_peak / k * 1e3:.2f} mm/s instead of {prof.v_peak * 1e3:.2f} mm/s)")
    vel[0] = 0.0
    vel[-1] = 0.0
    steps = np.abs(np.diff(q, axis=0))
    return RetimedPath(times=t, positions=q, velocities=vel, s=s, S=S, v_cruise=prof.v_peak / k, a=prof.a / (k * k), time_scale=k,
                       max_joint_velocity=float(np.max(np.abs(vel))) if vel.size else 0.0,
                       max_joint_accel=float(np.max(np.abs(acc))) if acc.size else 0.0,
                       max_joint_step=float(steps.max()) if steps.size else 0.0, notes=notes)


def straight_line_model(T0: np.ndarray, T1: np.ndarray, f: float) -> np.ndarray:
    """균등 비율 모델의 비율 f 위치 (병진 선형 보간; 회전은 검사하지 않는다 — 위치만 직선성 검사)."""
    T0 = np.asarray(T0, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    return T0[:3, 3] + float(f) * (T1[:3, 3] - T0[:3, 3])


def path_check(T0: np.ndarray, T1: np.ndarray, samples: Sequence[tuple], tol_m: float) -> tuple:
    """samples = [(f, p_fk (3,))] → (최대 편차 [m], 사유 | None). 편차 > tol 이면 '직선이 아님' 사유."""
    worst = 0.0
    worst_f = 0.0
    for f, p in samples:
        d = float(np.linalg.norm(np.asarray(p, dtype=float) - straight_line_model(T0, T1, f)))
        if d > worst:
            worst, worst_f = d, float(f)
    if worst > float(tol_m):
        return worst, (f"Cartesian path deviates {worst * 1e3:.2f} mm from the checked straight line at fraction {worst_f:.2f} "
                       f"(> {float(tol_m) * 1e3:.2f} mm): the executed segment would not be the envelope-checked one - refusing")
    return worst, None


# --------------------------------------------------------------------------------------
# 조밀한 직선 IK 추종 (URDF FK + 감쇠 최소제곱)
# --------------------------------------------------------------------------------------
def rotation_vector(R: np.ndarray) -> np.ndarray:
    """회전 행렬 → 회전 벡터 (축·각, rad). 각 θ < 179° 는 닫힌 식 (θ = atan2(|v|/2, (tr − 1)/2), 축 = v/|v|, v = 반대칭 부분 ×2 — IK
    안쪽 반복에서 scipy 보다 ~20 배 빠름), 180° 근처는 scipy (반대칭 부분이 0 으로 가 축이 불안정)."""
    R = np.asarray(R, dtype=float)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s2 = math.sqrt(float(v @ v))                                   # 2 sin θ
    c = 0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1.0)                  # cos θ
    th = math.atan2(0.5 * s2, c)
    if th < math.radians(179.0):
        return v * (th / s2) if s2 > 1e-12 else 0.5 * v
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, f: float) -> np.ndarray:
    """MoveIt 규칙: 자세 slerp(f), 병진 선형 (f)."""
    from scipy.spatial.transform import Rotation, Slerp
    T0 = np.asarray(T0, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    sl = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]])))
    T = np.eye(4)
    T[:3, :3] = sl([float(f)]).as_matrix()[0]
    T[:3, 3] = (1.0 - f) * T0[:3, 3] + f * T1[:3, 3]
    return T


def cartesian_steps(T0: np.ndarray, T1: np.ndarray, step_m: float) -> int:
    """MoveIt MaxEEFStep(step) 와 같은 단계 수: max(floor(L/step), floor(Θ/(3.5·step))) + 1."""
    L = float(np.linalg.norm(np.asarray(T1)[:3, 3] - np.asarray(T0)[:3, 3]))
    th = float(np.linalg.norm(rotation_vector(np.asarray(T0)[:3, :3].T @ np.asarray(T1)[:3, :3])))
    return max(int(math.floor(L / step_m)), int(math.floor(th / (3.5 * step_m)))) + 1


def _pose_error(T: np.ndarray, T_goal: np.ndarray) -> np.ndarray:
    e = np.zeros(6)
    e[:3] = T_goal[:3, 3] - T[:3, 3]
    e[3:] = rotation_vector(T_goal[:3, :3] @ T[:3, :3].T)
    return e


def _vee(M: np.ndarray) -> np.ndarray:
    """작은 회전 R ≈ I + [w]× → w (반대칭 부분)."""
    return 0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


def _jacobian(fk, q: np.ndarray, T: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """기하 야코비안 (선속도·각속도, base 프레임). fast_fk 의 fk 는 해석적 (fk.jacobian, FK 한 번), 아니면 수치 미분 (FK 관절 수 + 1 번)."""
    jac = getattr(fk, "jacobian", None)
    if jac is not None:
        return jac(q)
    J = np.zeros((6, q.size))
    for k in range(q.size):
        dq = q.copy()
        dq[k] += eps
        Tk = fk(dq)
        J[:3, k] = (Tk[:3, 3] - T[:3, 3]) / eps
        J[3:, k] = _vee(Tk[:3, :3] @ T[:3, :3].T) / eps
    return J


def fast_fk(chain):
    """scan_core.fk_urdf.KinematicChain → 빠른 fk(q) (관절 원점 행렬·축을 미리 계산; 같은 결과)."""
    steps = []
    for j in chain.joints:
        O = j.origin()
        if j.movable:
            a = np.asarray(j.axis, dtype=float)
            a = a / np.linalg.norm(a)
            K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
            steps.append((O, j.type, a, K, K @ K))
        else:
            steps.append((O, "fixed", None, None, None))
    # 연속한 고정 관절은 하나로 합친다
    merged = []
    acc = np.eye(4)
    for O, typ, a, K, KK in steps:
        acc = acc @ O
        if typ != "fixed":
            merged.append((acc, typ, a, K, KK))
            acc = np.eye(4)
    tail = acc

    I3 = np.eye(3)
    steps3 = [(O[:3, :3].copy(), O[:3, 3].copy(), typ, a, K, KK) for O, typ, a, K, KK in merged]
    tR, tp = tail[:3, :3].copy(), tail[:3, 3].copy()

    def _pass(q, joints: Optional[list] = None) -> np.ndarray:
        q = np.asarray(q, dtype=float).ravel()
        R = I3
        p = np.zeros(3)
        for k, (OR, Op, typ, a, K, KK) in enumerate(steps3):
            p = p + R @ Op
            R = R @ OR
            if joints is not None:
                joints.append((typ, R @ a, p))                  # 관절 축 (base) 과 원점 — 움직이기 전 프레임
            if typ == "prismatic":
                p = p + R @ (a * q[k])
            else:
                R = R @ (I3 + math.sin(q[k]) * K + (1.0 - math.cos(q[k])) * KK)
        T = np.eye(4)
        T[:3, :3] = R @ tR
        T[:3, 3] = p + R @ tp
        return T

    def fk(q):
        return _pass(q)

    def jacobian(q) -> np.ndarray:
        """해석적 기하 야코비안: 회전 관절 [w × (p_끝 − p_j); w], 직동 관절 [w; 0] (_jacobian 의 수치 미분과 같은 정의)."""
        js: list = []
        T = _pass(q, js)
        W = np.array([w for _t, w, _p in js])
        P = np.array([p_ for _t, _w, p_ in js])
        J = np.empty((6, len(js)))
        J[:3, :] = np.cross(W, T[:3, 3] - P).T
        J[3:, :] = W.T
        for k, (typ, w, _p) in enumerate(js):
            if typ == "prismatic":
                J[:3, k] = w
                J[3:, k] = 0.0
        return J

    fk.jacobian = jacobian
    return fk


class LineTrackError(ValueError):
    """track_straight_line's refusal (a ValueError) with the fraction of the straight line where it stopped."""

    def __init__(self, msg: str, fraction: float):
        super().__init__(msg)
        self.fraction = float(fraction)


def track_straight_line(fk, q0: Sequence[float], T_start: np.ndarray, T_target: np.ndarray, step_m: float, *,
                        lower: Optional[Sequence[float]] = None, upper: Optional[Sequence[float]] = None,
                        max_joint_step_rad: float = 0.05, tol_p: float = 1e-6, tol_r: float = 1e-6, max_iter: int = 30,
                        damping: float = 1e-6) -> tuple:
    """T_start → T_target 직선을 MoveIt 과 같은 단계로 IK 추종 → (Q (N+1, J), fractions (N+1,)). 실패는 ValueError (거부)."""
    q = np.asarray(q0, dtype=float).copy()
    T_start = np.asarray(T_start, dtype=float)
    T_target = np.asarray(T_target, dtype=float)
    n = cartesian_steps(T_start, T_target, step_m)
    lo = None if lower is None else np.asarray(lower, dtype=float)
    hi = None if upper is None else np.asarray(upper, dtype=float)
    Q = [q.copy()]
    fr = [0.0]
    from scipy.spatial.transform import Rotation, Slerp
    sl = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([T_start[:3, :3], T_target[:3, :3]])))
    Rs = sl(np.arange(n + 1) / n).as_matrix()
    for i in range(1, n + 1):
        f = i / n
        Tg = np.eye(4)
        Tg[:3, :3] = Rs[i]
        Tg[:3, 3] = (1.0 - f) * T_start[:3, 3] + f * T_target[:3, 3]
        q_prev = q.copy()
        if len(Q) >= 2:
            q = q + (Q[-1] - Q[-2])                     # 선형 예측 (이전 단계의 증분) → 보통 1 회 반복으로 수렴
        ok = False
        for _ in range(max_iter):
            T = fk(q)
            e = _pose_error(T, Tg)
            if np.linalg.norm(e[:3]) <= tol_p and np.linalg.norm(e[3:]) <= tol_r:
                ok = True
                break
            J = _jacobian(fk, q, T)
            JJt = J @ J.T + damping * np.eye(6)
            q = q + J.T @ np.linalg.solve(JJt, e)
        if not ok:
            raise LineTrackError(f"IK along the straight line did not converge at fraction {f:.3f} (singularity / unreachable) - "
                                 "refusing", f)
        dq = float(np.max(np.abs(q - q_prev)))
        if dq > max_joint_step_rad:
            raise LineTrackError(f"joint-space jump {dq:.3f} rad at fraction {f:.3f} (> {max_joint_step_rad} rad): configuration "
                                 "change along the straight line - refusing", f)
        if lo is not None and np.any(q < lo - 1e-9):
            k = int(np.argmin(q - lo))
            raise LineTrackError(f"joint {k} below its URDF limit at fraction {f:.3f} ({q[k]:.4f} < {lo[k]:.4f}) - refusing", f)
        if hi is not None and np.any(q > hi + 1e-9):
            k = int(np.argmax(q - hi))
            raise LineTrackError(f"joint {k} above its URDF limit at fraction {f:.3f} ({q[k]:.4f} > {hi[k]:.4f}) - refusing", f)
        Q.append(q.copy())
        fr.append(f)
    return np.asarray(Q), np.asarray(fr)


def on_path_distance(Q_dense: np.ndarray, q: Sequence[float]) -> float:
    """관절 점 q 와 조밀한 관절 경로 (꺾은선) 사이 최소 거리 [rad] — move_group 의 점이 같은 IK 분기·같은 경로인지."""
    Qd = np.asarray(Q_dense, dtype=float)
    q = np.asarray(q, dtype=float)
    best = float("inf")
    for i in range(Qd.shape[0] - 1):
        a, b = Qd[i], Qd[i + 1]
        d = b - a
        L2 = float(d @ d)
        t = 0.0 if L2 < 1e-18 else min(1.0, max(0.0, float((q - a) @ d) / L2))
        best = min(best, float(np.linalg.norm(q - (a + t * d))))
    if Qd.shape[0] == 1:
        best = float(np.linalg.norm(q - Qd[0]))
    return best


def track_polyline(chain, q0, poses, step_m: float, lower, upper, max_joint_step_rad: float) -> dict:
    """조밀 IK 로 플랜지 자세들을 차례로 (이웃 사이 직선, step_m) 따라간다 — 용접 경로 도달성 확인 (프로세스 풀에서도: 입력이 모두 피클 가능).
    반환 {ok, reason, i_fail, Q (M, J)}; 실패 (한계·끊김·수렴) 는 ok False."""
    fk = fast_fk(chain)
    q = np.asarray(q0, dtype=float)
    Qs = [q[None, :]]
    seg = []
    for i, T in enumerate(poses):
        try:
            Qd, _fr = track_straight_line(fk, q, fk(q), np.asarray(T, dtype=float), step_m, lower=lower, upper=upper,
                                          max_joint_step_rad=max_joint_step_rad)
        except ValueError as e:
            return {"ok": False, "reason": str(e), "i_fail": i, "fraction": getattr(e, "fraction", None),
                    "Q": np.concatenate(Qs, axis=0), "seg": seg}
        Qs.append(Qd[1:])
        seg.extend([i] * (Qd.shape[0] - 1))
        q = Qd[-1]
    return {"ok": True, "reason": "", "i_fail": -1, "Q": np.concatenate(Qs, axis=0), "seg": seg}


def follow_candidate(chain, fk, q, follow, *, step_m: float, lower, upper, max_joint_step_rad: float, sigma_radius_m: float,
                     q_ref=None) -> dict:
    """관절 해 q 에서 플랜지 자세들 follow 를 차례로 직선 추종 (track_polyline) → 도달 후보 기록 {q, ok, reason, limit_margin_rad (추종 경로의
    최소 관절 한계 여유), min_sigma (야코비안 최소 특이값, 각속도 행 × sigma_radius_m), n_track, dist_rad (q_ref 에서 Σ|Δq|; 없으면 None)}.
    플랫폼 reach_candidates (IK 해마다) 와 follow_from (이미 아는 관절에서 — IK 호출 없이: 회전 사전검사·yaw 띠 검사) 이 같은 규칙으로 쓴다."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    if follow:
        tr = track_polyline(chain, q, [np.asarray(T, dtype=float) for T in follow], float(step_m), lo, hi, float(max_joint_step_rad))
    else:
        tr = {"ok": True, "reason": "", "Q": q[None, :]}
    Q = np.asarray(tr["Q"], dtype=float)
    margin = float(np.min(np.minimum(Q - lo, hi - Q)))
    sig = float("inf")
    for qq in list(Q[:: max(1, Q.shape[0] // 25)]) + [Q[-1]]:
        J = np.array(fk.jacobian(np.asarray(qq, dtype=float)), dtype=float)
        J[3:, :] *= float(sigma_radius_m)
        sig = min(sig, float(np.linalg.svd(J, compute_uv=False).min()))
    qr = None if q_ref is None else np.asarray(q_ref, dtype=float).reshape(-1)
    # where the follow stopped (a failure): the segment index (0 = q's pose -> follow[0], ...), the distance along it of the first
    # pose the IK did not reach (fraction x length) and the segment's length [m]
    i_fail = int(tr.get("i_fail", -1)) if not tr["ok"] else -1
    done = seg_len = None
    if i_fail >= 0:
        P_a = (fk(q) if i_fail == 0 else np.asarray(follow[i_fail - 1], dtype=float))[:3, 3]
        P_b = np.asarray(follow[i_fail], dtype=float)[:3, 3]
        seg_len = float(np.linalg.norm(P_b - P_a))
        fr = tr.get("fraction")
        done = None if fr is None else float(fr) * seg_len                # the first pose of the segment the IK did not reach
    return {"q": [float(v) for v in q], "ok": bool(tr["ok"]), "reason": str(tr.get("reason", "")), "limit_margin_rad": margin,
            "min_sigma": sig, "n_track": int(Q.shape[0]), "q_end": [float(v) for v in Q[-1]],
            "dist_rad": None if qr is None else float(np.sum(np.abs(q - qr))), "i_fail": i_fail, "fail_done_m": done,
            "fail_segment_m": seg_len}


def follow_candidate_job(chain, q, follow, kw: dict) -> dict:
    """follow_candidate 를 프로세스 풀에서 (입력이 모두 피클 가능 — track_polyline 처럼 fk 는 체인에서 다시 만든다: fast_fk 는 결정적이라 같은
    결과). kw = follow_candidate 의 키워드 인자."""
    return follow_candidate(chain, fast_fk(chain), q, follow, **kw)


def nearest_turn_equivalent(q, q_ref, lower, upper) -> np.ndarray:
    """같은 플랜지 자세의 관절 표현 중 q_ref 에 가장 가까운 것: 관절마다 q_j + 2πk (관절 한계 안) 중 |· − q_ref_j| 가 가장 작은 값.
    한 바퀴 넘게 도는 관절 (한계 폭 > 2π) 만 바뀐다 — 회전 관절의 q 와 q ± 2π 는 같은 링크 자세다. IK 는 표현을 고르지 않으므로 (MoveIt
    /compute_ik: 시드에 따라 손목1 +180° 또는 −180°) 이것 없이 관절 거리를 비교하면 같은 자세가 한 바퀴만큼 멀어 보인다 (2026-10-08 실기:
    acquire_S 가 손목1 +180° 를 받아 232.5° 회전 48.6 s, −180° 였으면 127.5°)."""
    q = np.asarray(q, dtype=float).reshape(-1).copy()
    r = np.asarray(q_ref, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    two_pi = 2.0 * math.pi
    for j in range(q.size):
        k0 = round((r[j] - q[j]) / two_pi)
        best = q[j]
        for k in (k0 - 1, k0, k0 + 1):
            v = q[j] + k * two_pi
            if lo[j] - 1e-9 <= v <= hi[j] + 1e-9 and abs(v - r[j]) < abs(best - r[j]) - 1e-12:
                best = v
        q[j] = best
    return q


def joint_path_stats(Q, fk, joint_names: Optional[Sequence[str]] = None) -> dict:
    """조밀 관절 경로 → 확인·기록용 통계: 플랜지 병진 [m] 과 회전 [deg] (각각 누적 — max(병진, 회전 × 반경) 로 섞지 않는다), 관절별 누적
    회전 [deg], 가장 많이 도는 관절 (이름·순번·각). 2026-10-08 g1_b: 확인 프롬프트의 'move 1893.0 mm' (현 929.2 mm + 툴 레버 × 회전) 는
    이동 거리가 아니었다. 계획 기록의 'path 950 mm' 는 retime_joint_path 의 섞은 경로 변수 S (점마다 max(병진, 회전 × 반경) 의 누적 ≥
    병진) — g1_b 시작 자세 이동은 모든 점에서 병진이 커 S = 플랜지 경로 949.6 mm 였지만 (기록한 시작·목표 관절로 다시 푼 값), 회전이 큰
    경로에서는 S 가 병진보다 길다. 그래서 병진과 회전을 따로 낸다."""
    Q = np.asarray(Q, dtype=float)
    T = [fk(q) for q in Q]
    tr = sum(float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3])) for Ta, Tb in zip(T[:-1], T[1:]))
    rot = sum(float(np.linalg.norm(rotation_vector(Ta[:3, :3].T @ Tb[:3, :3]))) for Ta, Tb in zip(T[:-1], T[1:]))
    per = np.sum(np.abs(np.diff(Q, axis=0)), axis=0) if Q.shape[0] > 1 else np.zeros(Q.shape[1])
    net = Q[-1] - Q[0]
    k = int(np.argmax(per)) if per.size else 0
    names = list(joint_names) if joint_names is not None else list(getattr(fk, "joint_names", []) or [])
    return {"flange_translation_m": tr, "flange_rotation_deg": math.degrees(rot), "joint_rotation_deg": np.degrees(per).tolist(),
            "max_joint": names[k] if k < len(names) else str(k), "max_joint_index": k,
            "max_joint_rotation_deg": float(math.degrees(per[k])) if per.size else 0.0,
            "max_joint_net_deg": float(math.degrees(net[k])) if net.size else 0.0}
