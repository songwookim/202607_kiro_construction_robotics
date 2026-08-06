from geometry_msgs.msg import Pose

from construct_robot.welding_scenario_gui import WeldingScenarioModel


def make_pose(x):
    pose = Pose()
    pose.position.x = x
    pose.orientation.w = 1.0
    return pose


def test_touch_capture_can_record_without_changing_tcp_endpoints():
    model = WeldingScenarioModel()
    pose = make_pose(0.1)
    assert model.record_touch(pose, save_endpoint=False) is None
    assert model.last_touch_pose == pose
    assert model.tcp_endpoints == [None, None]


def test_touch_capture_fills_tcp1_then_tcp2():
    model = WeldingScenarioModel()
    assert model.record_touch(make_pose(0.1), save_endpoint=True) == 0
    assert model.record_touch(make_pose(0.2), save_endpoint=True) == 1
    assert model.tcp_endpoints[0].position.x == 0.1
    assert model.tcp_endpoints[1].position.x == 0.2


def test_linear_welding_path_contains_both_tcp_endpoints():
    model = WeldingScenarioModel()
    model.set_endpoint(0, make_pose(0.1))
    model.set_endpoint(1, make_pose(0.3))
    points = model.path(5)
    assert len(points) == 5
    assert points[0].position.x == 0.1
    assert points[-1].position.x == 0.3


def test_path_or_setting_change_invalidates_signature():
    model = WeldingScenarioModel()
    model.set_endpoint(0, make_pose(0.1))
    model.set_endpoint(1, make_pose(0.3))
    first = model.signature((0.1, 0.002, 100, 200))
    assert first != model.signature((0.2, 0.002, 100, 200))
    model.approved_signature = first
    model.set_endpoint(1, make_pose(0.4))
    assert model.approved_signature is None
    assert first != model.signature((0.1, 0.002, 100, 200))


def test_group_reset_clears_scenario_data():
    model = WeldingScenarioModel()
    model.set_initial(("joint1",), (0.1,), make_pose(0.0))
    model.set_endpoint(0, make_pose(0.1))
    model.reset_for_group("left_manipulator")
    assert model.planning_group == "left_manipulator"
    assert model.initial_joint_names == ()
    assert model.tcp_endpoints == [None, None]
