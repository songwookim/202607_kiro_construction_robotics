"""Delayed ROS responses are simulated with futures; no ROS graph is started."""

from concurrent.futures import Future
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from construct_robot.nodes.action_call import ActionCall, wait_for_future
from construct_robot.nodes.weld_runtime_node import WeldGuiNode


def pending_action():
    accepted, result = Future(), Future()
    handle = SimpleNamespace(accepted=True, cancel_goal_async=Mock(),
                             get_result_async=lambda: result)
    client = SimpleNamespace(wait_for_server=lambda **kw: True,
                             send_goal_async=Mock(return_value=accepted))
    return client, accepted, result, handle


def test_late_acceptance_after_timeout_is_canceled_without_reclaiming_motion():
    client, accepted, _result, handle = pending_action()
    owner = Mock()
    call = ActionCall("motion", owner)
    call.send(client, object())
    with pytest.raises(TimeoutError, match="goal response"):
        call.wait(0, 0)
    assert not accepted.cancelled()  # handle response is still required
    accepted.set_result(handle)
    handle.cancel_goal_async.assert_called_once()
    owner.assert_not_called()


def test_execution_timeout_cancels_and_ignores_late_result():
    client, accepted, result, handle = pending_action()
    call = ActionCall("motion")
    call.send(client, object())
    accepted.set_result(handle)
    with pytest.raises(TimeoutError):
        call.wait(0, 0)
    result.set_result(SimpleNamespace(result="late success"))
    handle.cancel_goal_async.assert_called_once()


def test_stop_during_server_discovery_prevents_submission():
    client, *_ = pending_action()
    call = ActionCall("motion")
    call.cancel()
    with pytest.raises(RuntimeError, match="canceled before"):
        call.send(client, object())
    client.send_goal_async.assert_not_called()


def test_stop_before_acceptance_cancels_pending_runtime_motion():
    client, accepted, result, handle = pending_action()
    submitted = threading.Event()
    client.send_goal_async.side_effect = lambda *a, **k: (submitted.set(), accepted)[1]
    node = SimpleNamespace(active_motion_goal=None, ui=SimpleNamespace(post=Mock(), log=Mock()))
    completed = []
    worker = threading.Thread(target=lambda: completed.append(
        WeldGuiNode._send_action_goal_and_wait(node, client, object(), "motion")))
    worker.start()
    try:
        assert submitted.wait(1)
        WeldGuiNode.cancel_active_motion(node)
        accepted.set_result(handle)
        handle.cancel_goal_async.assert_called_once()
        assert node.active_motion_goal is None
        result.set_result(SimpleNamespace(result="canceled"))
        worker.join(1)
        assert completed == ["canceled"]
        assert not node._motion_action_calls
    finally:
        if not result.done():
            result.set_result(SimpleNamespace(result="cleanup"))
        worker.join(1)


def test_future_wait_wakes_on_callback_and_propagates_error():
    future = Future()
    future.set_result(42)
    assert wait_for_future(future, 0, "RPC") == 42
    failed = Future()
    failed.set_exception(ValueError("bad response"))
    with pytest.raises(ValueError, match="bad response"):
        wait_for_future(failed, 0, "RPC")
    with pytest.raises(RuntimeError, match="timed out"):
        wait_for_future(Future(), 0, "RPC")


def test_cartesian_cancel_before_execute_registration_prevents_moveit_send():
    from moveit_msgs.msg import RobotTrajectory
    from construct_robot.nodes.cartesian_path_server import CartesianPathActionServer
    client, *_ = pending_action()
    parent = SimpleNamespace(goal_id=SimpleNamespace(uuid=[1] * 16), is_cancel_requested=False)
    server = SimpleNamespace(
        _execute_handle_lock=threading.Lock(), _execute_calls={},
        _canceled_execute_goals=set(), _execute_client=client,
        get_logger=lambda: Mock())
    # The cancel callback runs before rclpy updates the parent goal's state.
    CartesianPathActionServer.cancel_callback(server, parent)
    with pytest.raises(RuntimeError, match="canceled before"):
        CartesianPathActionServer.execute_moveit_trajectory(server, RobotTrajectory(), parent)
    client.send_goal_async.assert_not_called()
    assert not server._execute_calls


def test_cartesian_cancel_is_scoped_to_its_parent_goal():
    from construct_robot.nodes.cartesian_path_server import CartesianPathActionServer
    calls = {bytes([i] * 16): ActionCall(f"goal {i}") for i in (1, 2)}
    for call in calls.values():
        call.handle = SimpleNamespace(cancel_goal_async=Mock())
    server = SimpleNamespace(
        _execute_handle_lock=threading.Lock(), _execute_calls=calls,
        _canceled_execute_goals=set(), get_logger=lambda: Mock())
    parent = SimpleNamespace(goal_id=SimpleNamespace(uuid=[1] * 16))
    CartesianPathActionServer.cancel_callback(server, parent)
    calls[bytes([1] * 16)].handle.cancel_goal_async.assert_called_once()
    calls[bytes([2] * 16)].handle.cancel_goal_async.assert_not_called()


@pytest.mark.parametrize("approved", [True, False])
def test_executing_the_approved_weld_plan_skips_the_preview(approved):
    """Markers and PLAN PREVIEW feedback were published when the plan was approved before ARC ON; repeating
    them for a 600-waypoint weave parked the torch ~0.8 s at the weld start (2026-10-15)."""
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import RobotTrajectory
    from construct_robot.nodes.cartesian_path_server import CartesianPathActionServer
    waypoints = []
    for i in range(50):
        p = Pose()
        p.position.y, p.orientation.w = 0.004 * i, 1.0
        waypoints.append(p)
    request = SimpleNamespace(waypoints=waypoints, visualize_path=True, execute_requested=True, planning_group="right_manipulator",
                              reuse_approved_plan=approved, interpolation_step=0.001, tcp_speed_m_s=0.0,
                              velocity_scale=0.3, linear_motion_profile=False)
    plan = SimpleNamespace(fraction=1.0, solution=RobotTrajectory())
    handle = SimpleNamespace(request=request, is_cancel_requested=False, publish_feedback=Mock(),
                             succeed=Mock(), abort=Mock(), canceled=Mock())
    params = {"use_moveit": True, "execute_motion": True}
    server = SimpleNamespace(
        publish_weld_markers=Mock(), get_parameter=lambda name: SimpleNamespace(value=params[name]),
        approved_plan=Mock(return_value=plan), plan_with_moveit=Mock(return_value=plan),
        consume_approved_plan=Mock(), approve_plan=Mock(), get_logger=lambda: Mock(),
        _requested_start_state=lambda request: None,
        execute_moveit_trajectory=Mock(return_value=SimpleNamespace(error_code=SimpleNamespace(val=1))))
    result = CartesianPathActionServer._execute_callback(server, handle)
    assert result.success
    server.execute_moveit_trajectory.assert_called_once()
    assert server.publish_weld_markers.called is (not approved)
    assert handle.publish_feedback.called is (not approved)
    assert len(result.sampled_path) >= len(waypoints) - 1                 # the result keeps the sampled path
