"""MoveIt 어댑터 (moveit_jtc) 플랫폼 설정 읽기·검증 (ROS 없음 — 단위 시험 가능).

플랫폼 YAML (hongik_kiro.yaml) 의 최상위 프레임 키 + `moveit_jtc:` 섹션 + `gravity:` 섹션을 읽는다.
설계 §6.4: 플랫폼 의존 값은 코드에 기본값을 두지 않는다 → 프레임·그룹·액션 이름과 **속도/가속 한계** 가 없으면
PlatformConfigError (시작 거부): 궤적의 시간 매개화가 이 선언 한계를 쓴다. `moveit_jtc:` 의 키는 이 클래스가 읽는 것뿐 — 그 밖의 키는
조용히 무시하지 않고 거부한다 (SECTION_KEYS, REQUIRED_LIMITS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from scan_l0.config import unnull
from scan_l0.ports import PlatformConfigError

__all__ = ["MoveItJtcConfig", "REQUIRED_LIMITS", "REQUIRED_NAMES", "SECTION_KEYS"]

REQUIRED_NAMES = ("move_group_name", "jtc_action", "cartesian_path_service", "joint_states_topic")
REQUIRED_LIMITS = ("max_tcp_speed_mps", "max_tcp_accel_mps2", "max_joint_velocity_rad_s", "max_joint_accel_rad_s2")
# the keys of the moveit_jtc section that this class reads (limits: REQUIRED_LIMITS) - any other key is refused, never ignored silently
SECTION_KEYS = ("move_group_name", "jtc_action", "cartesian_path_service", "fk_service", "ik_service", "plan_service",
                "state_validity_service", "apply_scene_service", "get_scene_service", "joint_states_topic", "controller_name", "controller_state_topic",
                "trajectory_execution_event_topic", "robot_description_topic", "cartesian_step_m", "jump_threshold",
                "max_joint_step_rad", "min_fraction", "avoid_collisions", "rotation_radius_m", "path_check_tol_m",
                "path_on_moveit_tol_rad", "goal_pose_tol_m", "goal_pose_tol_deg", "prepared_start_tol_mm", "prepared_start_tol_deg",
                "limits", "stop_decel_tcp_mps2", "stop_decel_joint_rad_s2", "goal_time_margin_s", "tf_timeout_s", "history_s")


def _req_str(d: dict, key: str, where: str) -> str:
    v = unnull(d.get(key))
    if v is None or not str(v).strip():
        raise PlatformConfigError(f"platform config: {where}{key} is required (no code default for platform-dependent values)")
    return str(v).strip()


def _opt_str(d: dict, key: str, default: str) -> str:
    v = unnull(d.get(key))
    return default if v is None else str(v)


def _pos_float(d: dict, key: str, where: str, *, required: bool = True, default: Optional[float] = None) -> Optional[float]:
    v = unnull(d.get(key))
    if v is None:
        if required:
            raise PlatformConfigError(f"platform config: {where}{key} is required (declared motion limit of this platform: the "
                                      "time parameterisation of every trajectory uses it)")
        return default
    f = float(v)
    if not (math.isfinite(f) and f > 0.0):
        raise PlatformConfigError(f"platform config: {where}{key} must be > 0, got {v!r}")
    return f


@dataclass(frozen=True)
class MoveItJtcConfig:
    # 프레임 (최상위 키)
    base_frame: str
    flange_frame: str                       # 직선 이동의 eef 링크 (MoveIt 그룹 체인의 끝 링크 또는 그 자식) — T_base←flange = /joint_states FK (robot_description)
    eoat_frame: str                         # 툴 목표 자세 구성 기준 (T_flange←eoat 를 TF 로)
    laser_frame: str                        # 광학 프레임의 부모 (EOAT 캘리브 저장 규약 laser_link → optical)
    optical_frame: str                      # 정적 TF 자식 (hongik_profiler_optical_frame)
    torch_frame: str                        # 이 팔에 달린 토치의 TCP (연구실: eoat_description torch_tcp = 우리 토치의 CAD 컨택트 팁 끝) —
                                            # 용접 계획의 토치 모형·D14 도구는 weld_torch (같은 토치; 2026-10-12)
    gravity_frame: str                      # 중력 방향이 알려진 지면 프레임 (TF 로 base 중력축을 유도)
    gravity_axis: tuple                     # gravity_frame 에서의 중력 방향 (보통 [0, 0, -1])
    use_sim_time: bool
    # moveit_jtc 섹션
    move_group_name: str
    jtc_action: str
    cartesian_path_service: str
    fk_service: str
    ik_service: str
    plan_service: str                   # move_group /plan_kinematic_path (관절 공간 이동: 막히면 OMPL 계획)
    state_validity_service: str         # move_group /check_state_validity (관절 공간 직선 보간의 충돌 검사)
    apply_scene_service: str            # move_group /apply_planning_scene (작업 공간 모델 상자)
    get_scene_service: str              # move_group /get_planning_scene (the planning frame name at run time - World <- base of the
                                        # KIRO-format output)
    joint_states_topic: str
    controller_name: str
    controller_state_topic: str
    trajectory_execution_event_topic: str
    robot_description_topic: str
    cartesian_step_m: float
    jump_threshold: float
    max_joint_step_rad: float
    min_fraction: float
    avoid_collisions: bool
    rotation_radius_m: float
    path_check_tol_m: float
    path_on_moveit_tol_rad: float
    goal_pose_tol_m: float
    goal_pose_tol_deg: float
    prepared_start_tol_mm: float
    prepared_start_tol_deg: float
    max_tcp_speed_mps: float
    max_tcp_accel_mps2: float
    max_joint_velocity_rad_s: float
    max_joint_accel_rad_s2: float
    stop_decel_tcp_mps2: float          # 액션 취소의 정지 = 지금 지령 위치·속도에서 이 감속도로 멈추는 궤적 (TCP·관절 중 느린 쪽); 포락선 정지 거리
    stop_decel_joint_rad_s2: float
    goal_time_margin_s: float
    tf_timeout_s: float
    history_s: float
    weld_torch: Optional[object] = field(default=None, compare=False)   # the weld plan's torch model (config weld_torch,
                                                                         # scan_core.weld_torch.WeldTorch; D14 only when that torch is
                                                                         # on this arm, tool = reach_tool)
    raw: dict = field(default_factory=dict, compare=False)

    @classmethod
    def from_params(cls, params: dict) -> "MoveItJtcConfig":
        p = dict(params or {})
        mj = dict(p.get("moveit_jtc") or {})
        if not mj:
            raise PlatformConfigError("platform config: motion_backend moveit_jtc needs a 'moveit_jtc:' section")
        lim = dict(mj.get("limits") or {})
        unknown = [f"moveit_jtc.{k}" for k in mj if k not in SECTION_KEYS] + [f"moveit_jtc.limits.{k}" for k in lim if k not in REQUIRED_LIMITS]
        if unknown:
            raise PlatformConfigError(f"platform config: {', '.join(sorted(unknown))}: not a setting of this package - remove it "
                                      "(the settings are those of hongik_kiro.yaml)")
        grav = dict(p.get("gravity") or {})
        axis = grav.get("axis")
        if axis is None or len(list(axis)) != 3:
            raise PlatformConfigError("platform config: gravity.axis must be a 3-vector in gravity.frame (e.g. [0, 0, -1])")
        ax = tuple(float(v) for v in axis)
        if math.sqrt(sum(v * v for v in ax)) < 1e-6:
            raise PlatformConfigError("platform config: gravity.axis must be non-zero")
        from scan_core.weld_torch import weld_torch_config
        try:
            wtorch = weld_torch_config(p)
        except (TypeError, ValueError) as e:
            raise PlatformConfigError(f"platform config: {e}") from None
        ust = unnull(p.get("use_sim_time"))
        if ust is None:
            raise PlatformConfigError("platform config: use_sim_time is required (true with Isaac Sim /clock, false on hardware)")
        if isinstance(ust, str):
            ust = ust.strip().lower() in ("1", "true", "yes", "on")
        cfg = cls(
            base_frame=_req_str(p, "base_frame", ""), flange_frame=_req_str(p, "flange_frame", ""),
            eoat_frame=_req_str(p, "eoat_frame", ""), laser_frame=_req_str(p, "laser_frame", ""),
            optical_frame=_req_str(p, "tool_frame", ""), torch_frame=_opt_str(p, "torch_frame", "torch_tcp"), gravity_frame=_req_str(grav, "frame", "gravity."), gravity_axis=ax,
            use_sim_time=bool(ust),
            move_group_name=_req_str(mj, "move_group_name", "moveit_jtc."), jtc_action=_req_str(mj, "jtc_action", "moveit_jtc."),
            cartesian_path_service=_req_str(mj, "cartesian_path_service", "moveit_jtc."),
            fk_service=_opt_str(mj, "fk_service", "/compute_fk"), ik_service=_opt_str(mj, "ik_service", "/compute_ik"),
            plan_service=_opt_str(mj, "plan_service", "/plan_kinematic_path"),
            state_validity_service=_opt_str(mj, "state_validity_service", "/check_state_validity"),
            apply_scene_service=_opt_str(mj, "apply_scene_service", "/apply_planning_scene"),
            get_scene_service=_opt_str(mj, "get_scene_service", "/get_planning_scene"),
            joint_states_topic=_req_str(mj, "joint_states_topic", "moveit_jtc."),
            controller_name=_req_str(mj, "controller_name", "moveit_jtc."),
            controller_state_topic=_opt_str(mj, "controller_state_topic", ""),
            trajectory_execution_event_topic=_opt_str(mj, "trajectory_execution_event_topic", "/trajectory_execution_event"),
            robot_description_topic=_opt_str(mj, "robot_description_topic", "/robot_description"),
            cartesian_step_m=float(_pos_float(mj, "cartesian_step_m", "moveit_jtc.")),
            jump_threshold=float(unnull(mj.get("jump_threshold")) or 0.0),
            max_joint_step_rad=float(_pos_float(mj, "max_joint_step_rad", "moveit_jtc.", required=False, default=0.1)),
            min_fraction=float(unnull(mj.get("min_fraction")) or 0.999),
            avoid_collisions=bool(unnull(mj.get("avoid_collisions")) if unnull(mj.get("avoid_collisions")) is not None else True),
            rotation_radius_m=float(_pos_float(mj, "rotation_radius_m", "moveit_jtc.", required=False, default=0.2)),
            path_check_tol_m=float(_pos_float(mj, "path_check_tol_m", "moveit_jtc.", required=False, default=0.5e-3)),
            path_on_moveit_tol_rad=float(_pos_float(mj, "path_on_moveit_tol_rad", "moveit_jtc.", required=False, default=0.02)),
            goal_pose_tol_m=float(_pos_float(mj, "goal_pose_tol_m", "moveit_jtc.", required=False, default=1.0e-3)),
            goal_pose_tol_deg=float(_pos_float(mj, "goal_pose_tol_deg", "moveit_jtc.", required=False, default=0.2)),
            prepared_start_tol_mm=float(_pos_float(mj, "prepared_start_tol_mm", "moveit_jtc.", required=False, default=0.1)),
            prepared_start_tol_deg=float(_pos_float(mj, "prepared_start_tol_deg", "moveit_jtc.", required=False, default=0.05)),
            max_tcp_speed_mps=float(_pos_float(lim, "max_tcp_speed_mps", "moveit_jtc.limits.")),
            max_tcp_accel_mps2=float(_pos_float(lim, "max_tcp_accel_mps2", "moveit_jtc.limits.")),
            max_joint_velocity_rad_s=float(_pos_float(lim, "max_joint_velocity_rad_s", "moveit_jtc.limits.")),
            max_joint_accel_rad_s2=float(_pos_float(lim, "max_joint_accel_rad_s2", "moveit_jtc.limits.")),
            stop_decel_tcp_mps2=float(_pos_float(mj, "stop_decel_tcp_mps2", "moveit_jtc.", required=False, default=0.2)),
            stop_decel_joint_rad_s2=float(_pos_float(mj, "stop_decel_joint_rad_s2", "moveit_jtc.", required=False, default=2.0)),
            goal_time_margin_s=float(_pos_float(mj, "goal_time_margin_s", "moveit_jtc.", required=False, default=5.0)),
            tf_timeout_s=float(_pos_float(mj, "tf_timeout_s", "moveit_jtc.", required=False, default=0.5)),
            history_s=float(_pos_float(mj, "history_s", "moveit_jtc.", required=False, default=600.0)),
            weld_torch=wtorch,
            raw=mj,
        )
        if not (0.0 < cfg.min_fraction <= 1.0) or cfg.min_fraction < 0.999:
            raise PlatformConfigError(f"platform config: moveit_jtc.min_fraction must be in [0.999, 1] (never execute a partial "
                                      f"straight line), got {cfg.min_fraction}")
        if cfg.cartesian_step_m > 0.01:
            raise PlatformConfigError(f"platform config: moveit_jtc.cartesian_step_m {cfg.cartesian_step_m} m is too coarse "
                                      "(<= 0.01 m; the envelope was checked on the straight line)")
        return cfg

    @property
    def controller_state(self) -> str:
        return self.controller_state_topic or f"/{self.controller_name}/controller_state"

    def limits_dict(self) -> dict:
        return {k: getattr(self, k) for k in REQUIRED_LIMITS}
