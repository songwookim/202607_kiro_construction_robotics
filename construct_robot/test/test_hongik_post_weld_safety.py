"""The lift's scene exemption is scoped and restored, including a failed move; no ROS graph."""

from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    stack = Path(__file__).resolve().parents[2] / "hongik_fine_sensing_field"
    for name in ("scan_core", "scan_l0", "vc3dz_driver"):
        monkeypatch.syspath_prepend(str(stack / name))
    from scan_l0 import post_weld
    return post_weld


def lift(module, monkeypatch):
    boxes = [{"id": "hongik/specimen"}]
    p = NS(world=NS(plate_normal=np.array([0., 0., 1.]), plate_point=np.zeros(3), moveit_objects=lambda: boxes),
           up=np.array([0., 0., 1.]), _T_now=lambda: np.eye(4),
           calib=NS(T_flange_eoat=np.eye(4), body_points_eoat=None),
           job_ref=NS(safe_clearance_m=.05, standoff_min_m=.2), _start_inside={"deep_m": .02},
           _class_speed=lambda kind: .01, plate_top=0., accel=.1,
           _record_move=lambda *args: {}, _decide=Mock(), _confirm=Mock(), _p=Mock(),
           cell_ids=["hongik/specimen"], rs=NS(remove_scene_boxes=Mock(), set_scene_boxes=Mock()),
           sess=object(), motion_timeout_s=10., policy=NS(goal_tol_m=.001, goal_tol_deg=.2),
           res=NS(n_move_l=0), _arm_contacts=lambda: ([], ""))
    monkeypatch.setattr(module, "tool_keypoints", lambda *args: {"foot": np.array([[0., 0., .03]]),
                                                               "opt": np.array([[0., 0., .1]])})
    monkeypatch.setattr(module, "body_points_base", lambda *args: None)
    monkeypatch.setattr(module, "make_local_move_plan", Mock())
    return p, boxes


def test_failed_lift_restores_hongik_objects_before_propagating_failure(module, monkeypatch):
    p, boxes = lift(module, monkeypatch)
    def blocked(*args, **kwargs):
        raise module.MotionError("fixed-body collision")
    monkeypatch.setattr(module, "execute_leg", blocked)
    with pytest.raises(module._Abort, match="fixed-body collision"):
        module.PostWeldScan._lift_clear(p)
    p.rs.remove_scene_boxes.assert_called_once_with(["hongik/specimen"])
    p.rs.set_scene_boxes.assert_called_once_with(boxes)
    assert p.res.n_move_l == 0 and p._start_inside is not None


def test_refused_end_envelope_never_removes_objects_or_sends_motion(module, monkeypatch):
    p, _boxes = lift(module, monkeypatch)
    def refuse(*args, **kwargs):
        raise module.EnvelopeError("end pose too close")
    monkeypatch.setattr(module, "make_local_move_plan", refuse)
    execute = Mock()
    monkeypatch.setattr(module, "execute_leg", execute)
    with pytest.raises(module._Abort, match="does not pass the envelope"):
        module.PostWeldScan._lift_clear(p)
    p.rs.remove_scene_boxes.assert_not_called()
    execute.assert_not_called()


def test_contacts_after_lift_refuse_further_motion_with_scene_restored(module, monkeypatch):
    p, boxes = lift(module, monkeypatch)
    monkeypatch.setattr(module, "execute_leg", lambda *args, **kwargs: NS(
        ok=True, as_dict=lambda: {}, reason="done", v_cmd=10., motion_duration_s=1.))
    p._arm_contacts = lambda: ([{"object": "fixed_fixture"}], "fixed fixture")
    p.world.explain = lambda contacts: "fixed fixture"
    with pytest.raises(module._Abort, match="still touches"):
        module.PostWeldScan._lift_clear(p)
    p.rs.set_scene_boxes.assert_called_once_with(boxes)
    assert p._start_inside is not None
