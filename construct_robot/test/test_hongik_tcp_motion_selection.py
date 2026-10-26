"""Field TCP posture selection against scripted IK/path checks; no ROS or hardware."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def teach_class(monkeypatch):
    stack = Path(__file__).resolve().parents[2] / "hongik_fine_sensing_field"
    for name in ("scan_core", "scan_l0", "vc3dz_driver"):
        monkeypatch.syspath_prepend(str(stack / name))
    from scan_l0.autoteach import AutoTeach
    return AutoTeach


def pose(degrees, distance):
    a = math.radians(degrees)
    T = np.eye(4)
    T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    T[0, 3] = distance
    return T


def planner(teach_class, *, rotations=(115, 75), distances=(0.226, 0.418), blocked=()):
    teach = teach_class.__new__(teach_class)
    teach.job0 = SimpleNamespace(safe_clearance_m=0.05)
    teach.res = SimpleNamespace(visit_order={})
    teach.weld_torch = SimpleNamespace(on_this_arm=True, tcp_frame="right_manipulator_ee_point",
                                        flange_from_tcp=SimpleNamespace(t=np.zeros(3)))
    teach._q_now = lambda: np.zeros(6)
    teach._T_now = lambda: np.eye(4)
    teach._wrist2_required = lambda: "any"
    teach._progression_signs = lambda: [-1, 1]
    teach._lever_m = lambda: 0.2
    teach._site_label = lambda name: name
    teach._reach_rows = lambda candidates: []
    teach._decide = lambda *args, **kwargs: None
    poses = {sg: pose(rotation, distance) for sg, rotation, distance in zip((-1, 1), rotations, distances)}
    teach._acquisition_poses = lambda sg: (teach.job0, poses[sg], poses[sg], poses[sg])
    # The faster, closer joint solution is deliberately the more-rotating tool pose.
    def reach(T, **kwargs):
        duration = 7.3 if np.allclose(T, poses[-1]) else 12.0
        return [dict(q=np.full(6, duration), ok=True, dist_rad=duration, reason="")]
    teach.sess = SimpleNamespace(robot=SimpleNamespace(reach_candidates=reach))
    attempted = []
    def select(candidates, name, **kwargs):
        sg = candidates[0]["sign"]
        attempted.append(sg)
        if sg in blocked:
            return dict(cand=None, rows=[dict(checked=True, ok=False, reason=f"side {sg}: collision")])
        cand = min(candidates, key=lambda c: c["q"][0])
        return dict(cand=cand, check={"plan": {}}, duration_s=cand["q"][0], bound_s=0,
                    route="straight", rows=[], prefer={})
    teach._select_joint = select
    return teach, attempted


def test_smaller_rotation_wins_over_shorter_distance_and_faster_joints(teach_class):
    teach, attempted = planner(teach_class)
    best = teach._plan_acquisition("S")
    assert best["sign"] == 1 and attempted == [1]
    assert best["duration_s"] == 12.0
    assert teach.res.visit_order["tool_rotation_deg"] == pytest.approx(75)
    assert teach.res.visit_order["tcp_move_mm"] == pytest.approx(418)
    assert teach.res.visit_order["tcp"] == "torch tip right_manipulator_ee_point"


def test_blocked_preferred_orientation_uses_other_collision_checked_path(teach_class):
    teach, attempted = planner(teach_class, blocked=(1,))
    assert teach._plan_acquisition("S")["sign"] == -1
    assert attempted == [1, -1]


def test_equal_rotations_choose_shorter_tcp_distance(teach_class):
    teach, attempted = planner(teach_class, rotations=(75, 75))
    assert teach._plan_acquisition("S")["sign"] == -1
    assert attempted == [-1]


def test_both_blocked_refuse_with_both_reasons(teach_class):
    teach, attempted = planner(teach_class, blocked=(-1, 1))
    assert teach._plan_acquisition("S") is None
    assert attempted == [1, -1]
    assert "side 1: collision" in teach._acq_fail and "side -1: collision" in teach._acq_fail


def test_tcp_distance_includes_torch_offset_under_rotation(teach_class):
    teach = teach_class.__new__(teach_class)
    teach.weld_torch = SimpleNamespace(on_this_arm=True, tcp_frame="tip",
                                        flange_from_tcp=SimpleNamespace(t=[0.2, 0, 0]))
    distance, rotation = teach._tcp_motion(np.eye(4), pose(90, 0))
    assert distance == pytest.approx(0.2 * math.sqrt(2))
    assert rotation == pytest.approx(90)
