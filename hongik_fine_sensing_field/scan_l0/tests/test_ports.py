"""KIRO 통합 M1: 포트 (scan_l0.ports) — 어댑터 선택 (MoveIt 만; RB 제어박스 어댑터는 2026-10-07 제거), executor/autoteach 가
포트에만 의존함, 순수 Python 가짜 RobotPlatform 위에서 execute_leg·gate_and_arm 이 그대로 도는지. 실로봇·ROS 없음."""
import ast
import os
from types import SimpleNamespace

import numpy as np
import pytest

from poses import from_mm_deg
from scan_core.path_gen import Waypoint
from scan_l0 import executor as E
from scan_l0.motion_types import MotionConfig, MotionResult
from scan_l0.ports import (ADAPTERS, MotionBackend, PlatformConfigError, PlatformGates, PoseSample, PoseSnapshot, PoseSource,
                           RobotPlatform, StopLatch, pose_samples_to_arrays, pose_samples_to_robot_samples,
                           select_adapter)
from scan_l0.session import ScanSession

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _module_imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for node in tree.body:                       # 모듈 최상위 import 만 (함수 안의 지연 import 는 호환 별칭)
        if isinstance(node, ast.ImportFrom):
            out.append(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
    return out


@pytest.mark.parametrize("mod", ["executor.py", "autoteach.py"])
def test_orchestrator_modules_depend_only_on_the_ports(mod):
    imps = _module_imports(os.path.join(PKG, "scan_l0", mod))
    assert not [m for m in imps if m.startswith("rb_cb_client")], imps          # 제어박스 클라이언트는 이 패키지에 없다
    assert ".ports" in imps


def test_select_adapter():
    assert select_adapter({"pose_source": "ros_tf", "motion_backend": "moveit_jtc"}) == "moveit_jtc"
    for bad in ({"pose_source": "cb5001_fk", "motion_backend": "cb_move_l"}, {"motion_backend": "cb_move_l"}, {},
                {"pose_source": "ros_tf", "motion_backend": "cb_move_l"}):
        with pytest.raises(PlatformConfigError):
            select_adapter(bad)                                                    # 제어박스 어댑터는 없다 (2026-10-07)
    assert ADAPTERS == {"moveit_jtc": ("ros_tf", "moveit_jtc")}


def test_pose_sample_conversions():
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    S = [PoseSample(t_rx_ns=1_000_000_000 + i, T_base_flange=T, joints=(0.1 * i, 0.2), joint_names=("a", "b"), stamp_ns=5 + i)
         for i in range(3)]
    rs = pose_samples_to_robot_samples(S)
    assert len(rs) == 3 and rs[1].t_rx_s == pytest.approx(1.000000001) and np.allclose(rs[2].T_base_flange, T)
    arr = pose_samples_to_arrays(S)
    assert arr["robot_T_base_flange"].shape == (3, 4, 4) and arr["robot_joint_positions_rad"].shape == (3, 2)
    assert list(arr["robot_joint_names"]) == ["a", "b"] and arr["robot_stamp_ns"].shape == (3,)


# --------------------------------------------------------------------------------------
# 순수 Python 가짜 플랫폼 (CB 도 ROS 도 아님) 위에서 실행기 헬퍼
# --------------------------------------------------------------------------------------
class _Latch(StopLatch):
    def __init__(self):
        self._set, self.reason, self.last = False, "", None

    @property
    def is_set(self):
        return self._set

    def fire(self, reason, source="", t_trigger_ns=None):
        self._set, self.reason = True, reason
        self.last = SimpleNamespace(as_dict=lambda: {"reason": reason, "source": source})
        return self.last

    def wait(self, timeout_s):
        return self._set

    def begin_run(self):
        self._set, self.reason, self.run_open = False, "", True

    def end_run(self):
        self._set, self.reason, self.run_open = False, "", False


class _Pose(PoseSource):
    kind = "fake"

    def __init__(self):
        self.T = from_mm_deg([600.0, 0.0, 500.0, 180.0, 0.0, 0.0])

    def latest_pose(self):
        return self.T.copy()

    def snapshot(self):
        return PoseSnapshot(T=self.T.copy())

    def history(self, since_ns):
        return []

    def robot_samples(self, samples):
        return []

    def sample_arrays(self, samples):
        return {}

    def pose_at(self, t_s):
        return self.T.copy()                                            # 서 있는 가짜: 언제나 같은 자세


class _Backend(MotionBackend):
    kind = "fake"

    def __init__(self, plat, cfg):
        self.p, self.cfg, self.sent = plat, cfg, []

    @property
    def latch(self):
        return self.p.latch

    def move_linear(self, T_target, v_mps, a_mps2, *, timeout_s=None):
        if self.estopped:
            raise E.MotionError("this run was stopped")
        self.sent.append(np.array(T_target))
        self.p.pose.T = np.array(T_target)
        return MotionResult(script="fake", started=True, finished=True, reason="done")

    def abort(self):
        pass


class _Gates(PlatformGates):
    name = "fake"

    def motion_config(self, settings, **overrides):
        return MotionConfig(**overrides)

    def session_init(self, platform, cfg, log):
        log("session_init OK (fake)")
        return SimpleNamespace(default_speed=1.0, as_dict=lambda: {"backend": "fake"})

    def status(self, platform):
        return {"fake": True}

    def default_speed(self, platform):
        return 1.0

    def effective_speed(self, platform, cfg, v_target):
        return v_target

    def frame_offset_record(self, platform, snap):
        return {"drift_checked": False}


class _Platform(RobotPlatform):
    adapter = "fake"
    pose_source = "fake"

    def __init__(self):
        self._pose, self._latch, self._gates, self._be = _Pose(), _Latch(), _Gates(), None

    @property
    def pose(self):
        return self._pose

    @property
    def backend(self):
        return self._be

    @property
    def latch(self):
        return self._latch

    @property
    def gates(self):
        return self._gates

    def arm(self, cfg):
        self._be = self._be or _Backend(self, cfg)
        return self._be

    def close(self):
        pass


def test_executor_helpers_run_on_a_non_cb_platform():
    plat = _Platform()
    sess = ScanSession(robot=plat)
    lines = []
    assert not sess.armed
    init = E.gate_and_arm(sess, plat.gates.motion_config(None), log=lines.append)
    assert sess.armed and init.as_dict() == {"backend": "fake"} and "session_init OK (fake)" in lines
    T0 = plat.pose.latest_pose()
    T1 = T0.copy()
    T1[0, 3] += 0.01
    w = Waypoint("probe_X", "scan", T1, 0.02, 0.2, "move_l")
    res = E.execute_leg(sess, w, 0.02, expected_start=T0)
    assert res.ok and np.allclose(plat.backend.sent[-1], T1)                        # 측정 프레임 목표 그대로 (CB δ 없음)
    with pytest.raises(E.StartPoseMismatch):
        E.execute_leg(sess, w, 0.02, expected_start=T0)                           # 이미 움직였다 → 시작 자세 게이트
    plat.latch.fire("test", "test")
    with pytest.raises(E.MotionError, match="this run was stopped"):
        E.execute_leg(sess, Waypoint("x", "scan", T0, 0.02, 0.2, "move_l"), 0.02, expected_start=T1)
    plat.latch.begin_run()                                                         # 다음 실행: 앞 실행의 정지가 남지 않는다 (래치 없음)
    assert E.execute_leg(sess, Waypoint("x", "scan", T0, 0.02, 0.2, "move_l"), 0.02, expected_start=T1).ok
