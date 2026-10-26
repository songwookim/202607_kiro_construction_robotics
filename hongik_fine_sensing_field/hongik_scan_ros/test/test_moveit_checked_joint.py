"""MoveIt adapter joint-space checks and the checked send (KIRO field 2026-10-14, review 4.1 / B1 / 4): every validity request carries
the full measured joint state with is_diff, the run's stop reaches the sampling loop, an unanswered service is PlatformUnavailable (it
ends a candidate search instead of rejecting a candidate), a blocked straight line under allow_ompl False is an estimate only, and
move_checked_joint sends exactly the checked trajectory once - start, scene and the joints we do not move are checked again right
before sending. No ROS graph: the backend on stand-ins (the message types are real)."""
import math
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hongik_scan_ros import moveit_adapter as MA
from scan_l0.motion_types import MotionError, PlatformUnavailable

NAMES = ["right_joint1", "right_joint2"]


def _js(right=(0.0, 0.0), left=0.7):
    return NS(name=["left_joint1", *NAMES], position=[left, *right])


def _backend(js=None):
    js = js or _js()
    mj = NS(move_group_name="right_manipulator", state_validity_service="/check_state_validity", prepared_start_tol_deg=0.05,
            prepared_start_tol_mm=0.1, jtc_action="fake_jtc", max_joint_velocity_rad_s=0.24, max_tcp_speed_mps=0.06,
            max_tcp_accel_mps2=0.1, rotation_radius_m=0.2, goal_pose_tol_m=1e-3, goal_pose_tol_deg=0.2)

    def fk(q):
        T = np.eye(4)
        T[:2, 3] = np.asarray(q, dtype=float)[:2] * 0.1            # a planar stand-in: the flange follows the joints
        return T
    latch = NS(is_set=False, reason="")
    p = NS(mj=mj, latch=latch, fk=lambda: fk, validity_client=object(), chain=lambda: NS(joint_names=list(NAMES)),
           allowed_start_tolerance=lambda *a, **k: (0.01, "MoveIt default allowed_start_tolerance 0.01 rad (test)"),
           pose=NS(latest_joint_state=lambda: (0, js), latest_pose=lambda: np.eye(4), history=lambda t: []))
    be = object.__new__(MA.MoveItJtcBackend)
    be.p, be.cfg = p, NS(v_eff_max_mps=0.06, progress_s=0.0, progress_cb=None, default_timeout_s=120.0)
    be._busy, be._abort, be._gh_lock = threading.Lock(), threading.Event(), threading.Lock()
    be._goal_handle, be._prepared, be._prepared_joint, be.last_plan, be.last_result = None, None, None, {}, None
    be._fresh_joint_state = lambda what: js
    sent = []
    be._action = NS(wait_for_server=lambda timeout_sec: True,
                    send_goal_async=lambda goal: sent.append(goal) or NS(result=lambda: NS(accepted=False)))
    return be, sent


@pytest.fixture
def validity(monkeypatch):
    """call_service stand-in for /check_state_validity: records the requests; answer() decides valid / None."""
    reqs = []
    ans = {"fn": lambda req: NS(valid=True, contacts=[])}
    monkeypatch.setattr(MA, "call_service", lambda client, req, *a, **k: reqs.append(req) or ans["fn"](req))
    monkeypatch.setattr(MA, "wait_future", lambda fut, t: True)
    return reqs, ans


def test_validity_requests_carry_the_full_measured_state_with_is_diff(validity):
    reqs, _ans = validity
    be, _sent = _backend()
    assert be._first_invalid(NAMES, np.array([[0.0, 0.0], [0.5, -0.2]])) is None
    for r in reqs:
        m = dict(zip(r.robot_state.joint_state.name, r.robot_state.joint_state.position))
        assert r.robot_state.is_diff and m["left_joint1"] == pytest.approx(0.7)       # the other arm as measured
    assert dict(zip(reqs[-1].robot_state.joint_state.name, reqs[-1].robot_state.joint_state.position))["right_joint1"] == 0.5


def test_the_stop_reaches_the_sampling_loop_and_no_answer_is_platform_unavailable(validity):
    reqs, ans = validity
    be, _sent = _backend()
    be.p.latch.is_set, be.p.latch.reason = True, "action cancel requested"
    with pytest.raises(MotionError, match="stopped while checking a joint path"):
        be._first_invalid(NAMES, np.array([[0.0, 0.0], [0.5, 0.0]]))
    assert reqs == []
    be.p.latch.is_set = False
    ans["fn"] = lambda req: None
    with pytest.raises(PlatformUnavailable, match="did not answer"):
        be._first_invalid(NAMES, np.array([[0.0, 0.0]]))


def _plan(be, q_goal=(1.0, 0.0), *, estimate=False, blocked=None):
    rt = NS(positions=np.array([[0.0, 0.0], list(q_goal)]), v_cruise=0.06, a=0.1, notes=[])
    return {"key": None, "names": list(NAMES), "q0": np.zeros(2), "rt": rt, "goal": object(), "script": "checked", "timeout_s": 30.0,
            "last_plan": {"duration_s": 4.2, "route": "straight", "estimate_only": estimate, "straight_blocked_reason": blocked}}


def test_check_joint_estimate_only_platform_unavailable_and_own_handles(validity):
    be, _sent = _backend()
    be._build_joint_plan = lambda *a, **k: _plan(be, estimate=not k["allow_ompl"], blocked="point 3/40 invalid")
    est = be.check_joint([1.0, 0.0], 0.06, 0.1, allow_ompl=False)
    assert est["ok"] is False and est["_checked_plan"] is None and est["plan"]["estimate_only"] and "point 3/40" in est["reason"]
    be._build_joint_plan = lambda *a, **k: _plan(be)
    a, b = be.check_joint([1.0, 0.0], 0.06, 0.1), be.check_joint([1.0, 0.0], 0.06, 0.1)
    assert a["ok"] and a["_checked_plan"] is not b["_checked_plan"]                       # every candidate keeps its own trajectory

    def gone(*a, **k):
        raise PlatformUnavailable("move_group /plan_kinematic_path did not answer")
    be._build_joint_plan = gone
    with pytest.raises(PlatformUnavailable):
        be.check_joint([1.0, 0.0], 0.06, 0.1)
    assert not be._busy.locked()


def _checked(be):
    be._build_joint_plan = lambda *a, **k: _plan(be)
    return be.check_joint([1.0, 0.0], 0.06, 0.1)


def test_move_checked_joint_sends_exactly_the_checked_trajectory_once(validity):
    be, sent = _backend()
    chk = _checked(be)
    other = _checked(be)                                         # a later check of another candidate does not replace it
    be.move_checked_joint(chk, [1.0, 0.0], 0.06, 0.1)
    assert sent == [chk["_checked_plan"]["goal"]] and other["_checked_plan"]["goal"] is not sent[0]
    with pytest.raises(MotionError, match="already used"):
        be.move_checked_joint(chk, [1.0, 0.0], 0.06, 0.1)
    with pytest.raises(MotionError, match="differ from the checked joint plan"):
        be.move_checked_joint(other, [1.0, 0.0], 0.05, 0.1)
    assert len(sent) == 1


@pytest.mark.parametrize("case", ["turn", "other_arm", "collision", "stop"])
def test_nothing_is_sent_when_the_check_right_before_sending_fails(validity, case):
    reqs, ans = validity
    be, sent = _backend()
    chk = _checked(be)
    if case == "turn":                                            # a 2 pi change of a joint: same flange, another joint value
        be._fresh_joint_state = lambda what: _js(right=(2.0 * math.pi, 0.0))
        match = "start changed"
    elif case == "other_arm":                                     # the other arm moves while the trajectory is checked again
        states = iter([_js(), _js(), _js(), _js(left=0.8)])     # before planning, after planning, before sending, after re-check
        be._fresh_joint_state = lambda what: next(states)
        match = "left_joint1 .* moved"
    elif case == "collision":
        ans["fn"] = lambda req: NS(valid=False, contacts=[NS(contact_body_1="right_link6", contact_body_2="hongik/specimen")])
        match = "no longer collision-free"
    else:
        ans["fn"] = lambda req: setattr(be.p.latch, "is_set", True) or NS(valid=True, contacts=[])
        match = "stopped"
    with pytest.raises(MotionError, match=match):
        be.move_checked_joint(chk, [1.0, 0.0], 0.06, 0.1)
    assert sent == []


def test_path_timing_limits_are_the_declared_limits_and_the_dense_step():
    be, _sent = _backend()
    assert be.path_timing_limits() == {"joint_velocity_rad_s": 0.24, "joint_path_step_rad": MA.JOINT_PATH_STEP_RAD, "tcp_speed_mps": 0.06,
                                       "tcp_accel_mps2": 0.1, "rotation_radius_m": 0.2}


def test_a_holding_difference_within_move_groups_start_tolerance_is_sent(validity):
    """KIRO field 2026-10-15 run 002753: the arm held 0.054 deg off the checked start (the former 0.05 deg reuse threshold refused
    the run) - within move_group's allowed_start_tolerance the checked trajectory is sent; beyond it nothing is sent."""
    be, sent = _backend()
    chk = _checked(be)
    be._fresh_joint_state = lambda what: _js(right=(math.radians(0.054), 0.0))
    be.move_checked_joint(chk, [1.0, 0.0], 0.06, 0.1)
    assert sent == [chk["_checked_plan"]["goal"]]
    be2, sent2 = _backend()
    chk2 = _checked(be2)
    be2._fresh_joint_state = lambda what: _js(right=(0.02, 0.0))
    with pytest.raises(MotionError, match=r"start changed: max joint 1\.1459 deg > 0\.573 deg \(MoveIt default"):
        be2.move_checked_joint(chk2, [1.0, 0.0], 0.06, 0.1)
    assert sent2 == []
