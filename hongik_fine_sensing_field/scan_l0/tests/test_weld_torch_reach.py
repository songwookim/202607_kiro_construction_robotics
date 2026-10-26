"""D14 (용접 계획의 로봇 확인, 모션 없음) 의 도구 — 가상 팔 (MoveIt 어댑터와 같은 규칙, scan_core.weld_torch.reach_tool): 용접 계획의
토치 = 우리 1차 브라켓 토치 (설정 weld_torch, 사용자·KIRO 합의 2026-10-12) — 연구실·가짜 팔에도 달려 있다 (on_this_arm true). 계획의
끝 자세 (와이어 끝) 를 이 팔 URDF 의 플랜지 ← torch_tcp (TF 와 같은 값) · TCP ← 끝 (+20 mm 토치 축) 으로 플랜지 자세로 옮겨 확인한다.
URDF 에 없는 TCP 이름 (KIRO 의 right_manipulator_ee_point) 이면 설정의 플랜지 ← TCP. 설정이 이 팔에 없다고 하면 확인하지 않는다."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from scan_l0 import config as C
from scan_core.weld_torch import weld_torch_config
from sim_arm import SimArmPlatform
from sim_harness import CALIB, sim_settings


def _plan():
    """패스 하나 (충전, 단면 둘): 토치 축 아래 (−z), 진행 +x."""
    R = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    P = np.array([[0.70, 0.0, 0.05], [0.71, 0.0, 0.05]])
    wp = SimpleNamespace(k=1, kind="fill", s_mm=np.array([0.0, 10.0]), R_base_tcp=np.array([R, R]), aim_base_m=P, weave_amp_mm=None,
                         dense_pos_m=P, dense_t_s=np.array([0.0, 2.0]))
    st = [SimpleNamespace(s_mm=0.0, w=np.array([0.0, 1.0, 0.0])), SimpleNamespace(s_mm=10.0, w=np.array([0.0, 1.0, 0.0]))]
    return SimpleNamespace(passes=[wp], stations=st), R, P


def _first_ik_target(settings, monkeypatch):
    arm = SimArmPlatform(settings, C.load_calib(CALIB))
    asked = []

    def ik_solutions(T, seeds, **kw):
        asked.append(np.asarray(T, dtype=float).copy())
        return []
    monkeypatch.setattr(arm, "ik_solutions", ik_solutions)
    monkeypatch.setattr(arm, "reach_candidates", lambda *a, **k: [])
    plan, R, P = _plan()
    rows = arm.check_weld_plan(plan, approach_m=0.03)
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = R, P[0] - R[:, 2] * 0.03
    return arm, rows, asked[0], A


def test_virtual_arm_has_no_d14_tool_when_the_config_says_the_weld_torch_is_not_on_it(monkeypatch):
    """일반 기제: 설정이 용접 토치가 이 팔에 없다고 하면 (on_this_arm false) check_weld_plan 은 도구가 없다 (ValueError — 부르는 쪽이
    D14 를 건너뛴다), 다른 토치로 IK 를 묻지 않는다."""
    st = sim_settings()
    p = dict(st.params)
    p["weld_torch"] = dict(p["weld_torch"], on_this_arm=False)
    arm = SimArmPlatform(C.Settings(p, st.path), C.load_calib(CALIB))
    asked = []
    monkeypatch.setattr(arm, "ik_solutions", lambda T, seeds, **kw: asked.append(T) or [])
    plan, _R, _P = _plan()
    with pytest.raises(ValueError, match=r"reachability \(D14\) not checked: the platform config says the weld torch is not mounted"):
        arm.check_weld_plan(plan)
    assert asked == []


def test_virtual_arm_checks_the_tip_poses_with_its_urdf_torch_tcp(monkeypatch):
    """가짜 팔 설정 (연구실과 같이 우리 토치가 이 팔에): 첫 IK 목표 = 끝 자세 · (플랜지 ← 끝)⁻¹, 플랜지 ← 끝 = URDF 의 플랜지 ← torch_tcp ·
    TCP ← 끝 (끝 틀; 이 맨 자세 목록에는 공정이 없어 돌출을 놓지 않는다) = 설정의 기록 (flange_from_tcp) 과 같다; 도구 설명에 그 TCP 위치."""
    st = sim_settings()
    wt = weld_torch_config(st.params)
    assert wt.on_this_arm is True and wt.tcp_frame == "torch_tcp" and os.path.basename(st.path) == "hongik_rb20_fake_moveit.yaml"
    arm, rows, T, A = _first_ik_target(st, monkeypatch)
    assert np.allclose(arm.T_flange_torch, wt.flange_from_tcp.matrix(), atol=1e-9)       # the URDF torch_tcp = the config's record
    assert np.allclose(T, A @ np.linalg.inv(arm.T_flange_torch @ wt.tcp_from_tip.matrix()), atol=1e-12)
    from torch_fixtures import tcp_text                                                 # the lab model (the torch turned in its clamp)
    assert rows[0]["tool"] == f"the weld torch torch_tcp (weld_torch, on this arm: tcp <- torch_tcp from the robot's TF: {tcp_text()})"


def test_virtual_arm_mounts_a_tcp_name_its_urdf_does_not_carry_from_the_config(monkeypatch):
    """KIRO 의 TCP 이름 (right_manipulator_ee_point) 은 연구실 URDF 에 없다 — 가상 팔은 설정의 플랜지 ← TCP 로 단다 (KIRO 는 그 로봇의
    TF)."""
    st = sim_settings()
    p = dict(st.params)
    p["weld_torch"] = dict(p["weld_torch"], tcp_frame="right_manipulator_ee_point")
    st = C.Settings(p, st.path)
    wt = weld_torch_config(st.params)
    arm, rows, T, A = _first_ik_target(st, monkeypatch)
    assert np.allclose(T, A @ np.linalg.inv(wt.flange_from_tip()), atol=1e-12)
    t = wt.flange_from_tcp.t * 1e3                                                      # the config's record (= the lab model)
    assert rows[0]["tool"].startswith("the weld torch right_manipulator_ee_point (weld_torch, on this arm: tcp <- "
                                      f"right_manipulator_ee_point from the robot's TF: ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) mm)")
