"""Wire-tip pivot calibration and its use in the scan targets."""

import numpy as np
import pytest

from construct_robot.application import scan_correction as sc
from construct_robot.application import tcp_calibration as tc
from construct_robot.core.scan_reference import aim_from_tcp, quat_matrix, tcp_from_aim
from construct_robot.core.tcp_calibration import pivot_calibration
from test_scan_reference import DOWN, compose, quat_axis_angle, write_estimate

@pytest.fixture(autouse=True)
def isolated_teaching(tmp_path, monkeypatch):
    """Never touch the cell's scan_teaching (compute_and_save refreshes the correction there)."""
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "isolated_teaching")
    monkeypatch.setattr(tc, "_teaching_dir", lambda: tmp_path / "isolated_teaching")


TIP = np.array([-0.0001, -0.0224, 0.0312])            # off the torch axis, like the 2026-10-15 estimate
POINT = np.array([0.01, -1.74, 0.565])


def captures(rotations):
    out = []
    for axis, deg in rotations:
        q = compose(quat_axis_angle(axis, deg), DOWN)
        out.append((POINT - quat_matrix(q) @ TIP, q))
    return out


def test_pivot_calibration_recovers_the_wire_tip():
    samples = captures([((1, 0, 0), 0), ((1, 0, 0), 30), ((0, 1, 0), 25), ((0, 0, 1), 40)])
    result = pivot_calibration(samples)
    np.testing.assert_allclose(result["tip_tcp_m"], TIP, atol=1e-9)
    np.testing.assert_allclose(result["point_world_m"], POINT, atol=1e-9)
    assert result["rms_m"] < 1e-9


@pytest.mark.parametrize("rotations,match", [
    ([((1, 0, 0), 0), ((1, 0, 0), 30), ((1, 0, 0), -30)], "one axis"),
    ([((1, 0, 0), 0), ((1, 0, 0), 5), ((0, 1, 0), 5)], "only"),
    ([((1, 0, 0), 0), ((0, 1, 0), 30)], "at least 3"),
])
def test_undetermined_tips_are_refused(rotations, match):
    with pytest.raises(ValueError, match=match):
        pivot_calibration(captures(rotations))


def test_samples_round_trip_and_the_offset_is_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "_teaching_dir", lambda: tmp_path)
    for p, q in captures([((1, 0, 0), 0), ((1, 0, 0), 30), ((0, 1, 0), 25), ((0, 0, 1), 40)]):
        tc.add_sample(p, q, "test")
    result = tc.compute_and_save()
    np.testing.assert_allclose(tc.load_offset(), TIP, atol=1e-9)
    assert result["file"] == str(tmp_path / "wire_tip_offset.yaml")
    tc.clear_samples()
    assert tc.load_samples() == [] and list((tmp_path / "history").glob("tcp_calibration_samples_*.yaml"))


def test_a_calibrated_tip_keeps_the_wire_on_the_target_for_any_orientation(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "TCP_TO_WIRE_TIP_M", None)
    monkeypatch.setattr(tc, "load_offset", lambda path=None: TIP)
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    zero = {"offsets_seam_m": {e: np.zeros(3) for e in ("start", "goal")}, "pass_deltas_seam_m": {}}
    for q in (DOWN, compose(quat_axis_angle((1, 0, 0), 40.0), DOWN), compose(quat_axis_angle((0, 1, 0), -20.0), DOWN)):
        tcp, aim = sc.pass_target_tcp(scan, 1, "start", q, zero)
        np.testing.assert_allclose(tcp + quat_matrix(q) @ TIP, (0.015, -1.539, 0.566), atol=1e-12)
        np.testing.assert_allclose(aim, (0.015, -1.539, 0.566), atol=1e-12)


def test_a_correction_from_another_wire_tip_model_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    monkeypatch.setattr(sc, "TCP_TO_WIRE_TIP_M", None)
    monkeypatch.setattr(tc, "load_offset", lambda path=None: None)
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    for end, aim in (("start", (0.015, -1.539, 0.566)), ("goal", (0.005, -1.738, 0.563))):
        sc.save_reference(1, end, tcp_from_aim(np.array(aim), DOWN, 0.0212), DOWN, "t")
    sc.compute_correction(scan)
    correction = sc.load_correction()
    sc.pass_target_tcp(scan, 1, "start", DOWN, correction)
    monkeypatch.setattr(tc, "load_offset", lambda path=None: TIP)
    with pytest.raises(ValueError, match="TCP calibration changed"):
        sc.pass_target_tcp(scan, 1, "start", DOWN, correction)


def test_a_new_wire_tip_refreshes_the_correction_on_its_own_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path / "teaching")
    monkeypatch.setattr(sc, "TCP_TO_WIRE_TIP_M", None)
    model = {"tip": None}
    monkeypatch.setattr(tc, "load_offset", lambda path=None: model["tip"])
    scan = write_estimate(tmp_path, (0.015, -1.539, 0.566), (0.005, -1.738, 0.563))
    for end, aim in (("start", (0.015, -1.539, 0.566)), ("goal", (0.005, -1.738, 0.563))):
        sc.save_reference(1, end, tcp_from_aim(np.array(aim), DOWN, 0.0212), DOWN, "t")
    sc.compute_correction(scan)
    assert sc.refresh_correction_for_wire_tip() is None               # model unchanged: nothing to do
    model["tip"] = TIP
    doc = sc.refresh_correction_for_wire_tip()
    assert doc is not None and doc["source_scan"] == str(scan)
    correction = sc.load_correction()
    for end in ("start", "goal"):                                     # the references are reproduced again
        ref = next(r for r in sc.load_references() if r["label"] == end)
        np.testing.assert_allclose(sc.pass_target_tcp(scan, 1, end, DOWN, correction)[0], ref["position"], atol=1e-9)


def plane_touches(tip=TIP, normal=(0.01, -0.015, 1.0), d=0.565, noise=0.0, seed=0, max_tilt_deg=35.0):
    """Synthetic touches: for every planned target, slide the TCP down World -Z until the true tip is on the plane."""
    from construct_robot.core.tcp_calibration import touch_calibration_plan
    rng = np.random.default_rng(seed)
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    q0 = compose(quat_axis_angle((1, 0, 0), 8), DOWN)
    start = np.array([0.01, -1.6, 0.6])
    plan = touch_calibration_plan(start, q0, 0.0212, max_tilt_deg=max_tilt_deg)   # planned with the (wrong) axis model
    out = []
    for target in plan:
        R = quat_matrix(target["quat"])
        p = np.asarray(target["tcp"], float)
        s = (n @ (p + R @ tip) - d) / n[2]                            # descent distance until contact
        out.append((target["kind"], p - np.array([0, 0, s]) + rng.normal(0, noise, 3), target["quat"]))
    return out


def test_plane_touch_calibration_recovers_the_wire_tip_and_the_plane():
    from construct_robot.core.tcp_calibration import plane_touch_calibration
    touches = plane_touches()
    result = plane_touch_calibration([t[1] for t in touches], [t[2] for t in touches], [t[0] for t in touches])
    np.testing.assert_allclose(result["tip_tcp_m"], TIP, atol=1e-9)
    np.testing.assert_allclose(result["plane_normal"], np.array([0.01, -0.015, 1.0]) / np.linalg.norm([0.01, -0.015, 1.0]), atol=1e-9)
    assert result["rms_m"] < 1e-9 and len(touches) == 11


def test_plane_touch_calibration_tolerates_touch_noise():
    from construct_robot.core.tcp_calibration import plane_touch_calibration
    errors = []
    for seed in range(20):                                             # 0.05 mm touch repeatability, 25 deg tilts
        touches = plane_touches(noise=0.00005, seed=seed)
        result = plane_touch_calibration([t[1] for t in touches], [t[2] for t in touches], [t[0] for t in touches])
        errors.append(np.linalg.norm(result["tip_tcp_m"] - TIP))
        assert np.all(result["tip_sigma_m"] < 0.003)
    assert np.median(errors) < 0.0015


def test_touch_tilts_are_limited():
    from construct_robot.core.tcp_calibration import touch_tilts
    assert max(abs(d) for _a, d in touch_tilts(25.0)) == 25.0 and len(touch_tilts(25.0)) == 8
    with pytest.raises(ValueError):
        touch_tilts(45.0)


def test_plane_touch_calibration_needs_spots_and_two_tilt_axes():
    from construct_robot.core.tcp_calibration import plane_touch_calibration
    touches = plane_touches()
    with pytest.raises(ValueError, match="3 spots"):
        plane_touch_calibration([t[1] for t in touches[1:]], [t[2] for t in touches[1:]], [t[0] for t in touches[1:]])
    one_axis = touches[:5] + touches[7:9]                              # 3 spots + tilts about tool x only
    with pytest.raises(ValueError, match="two different axes"):
        plane_touch_calibration([t[1] for t in one_axis], [t[2] for t in one_axis], [t[0] for t in one_axis])


def test_touch_run_touches_each_target_and_saves(tmp_path, monkeypatch):
    """The routine against a simulated robot: DI0 rises when the true tip reaches the plane during a guarded descent."""
    import copy
    from types import SimpleNamespace
    from geometry_msgs.msg import Pose
    n, d = np.array([0.0, 0.0, 1.0]), 0.565
    q0 = compose(quat_axis_angle((1, 0, 0), 8), DOWN)
    state = {"pose": Pose(), "touch": False}
    p0 = np.array([0.01, -1.6, 0.6])
    state["pose"].position.x, state["pose"].position.y, state["pose"].position.z = p0
    state["pose"].orientation.x, state["pose"].orientation.y, state["pose"].orientation.z, state["pose"].orientation.w = q0
    outputs = []
    node = SimpleNamespace(node_touch_input_states={"right": False}, touch_guard_edge_pose=None,
                           cancel_active_motion=lambda: None)
    node._current_tcp_pose = lambda group: copy.deepcopy(state["pose"])

    def run(step, execute):
        target = step["points"][-1]
        q = (target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w)
        tip = lambda p: np.asarray(p) + quat_matrix(q) @ TIP
        goal = np.array([target.position.x, target.position.y, target.position.z])
        if step.get("touch_guard"):
            start = np.array([state["pose"].position.x, state["pose"].position.y, state["pose"].position.z])
            s = (n @ tip(start) - d)                                    # World -Z descent until contact
            if s <= (start - goal)[2]:
                contact = copy.deepcopy(target)
                contact.position.x, contact.position.y, contact.position.z = start - np.array([0, 0, s])
                state["pose"], state["touch"] = contact, True
                node.touch_guard_edge_pose = copy.deepcopy(contact)
                node.node_touch_input_states["right"] = True
                return True, "stopped by Fastech DI0 · continuing"
        state["pose"] = copy.deepcopy(target)
        node.node_touch_input_states["right"] = False
        return True, "reached"
    node.run_sequence_cartesian_motion = run
    host = SimpleNamespace(node=node, hicomm_client=None,
                           _set_fastech_output_sync=lambda port, on: (outputs.append((port, on)), (True, "ok"))[1])
    monkeypatch.setattr(tc, "_teaching_dir", lambda: tmp_path)
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: tmp_path)
    result = tc.TouchCalibrationRun(host).run(0.0212, 0.1, lambda text: None)
    np.testing.assert_allclose(result["tip_tcp_m"], TIP, atol=1e-6)
    assert outputs[0] == (0, True) and outputs[-1] == (0, False)       # touch sensing only during the run
    np.testing.assert_allclose(tc.load_offset(), TIP, atol=1e-6)
