"""Synchronous service wrappers preserve their contracts without a ROS graph."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from construct_robot.nodes import action_call
from construct_robot.nodes.weld_runtime_node import WeldGuiNode


@pytest.fixture(params=("generic", "equipment", "legacy", "cancel", "stop", "switch"))
def service_case(request):
    response = SimpleNamespace(success=True, message="ack", ok=True,
                               return_code=0, goals_canceling=[object()])
    future = Future()
    future.set_result(response)
    client = SimpleNamespace(wait_for_service=Mock(return_value=True),
                             call_async=Mock(return_value=future))
    node = SimpleNamespace(
        legacy_digital_output_client=client, legacy_node_digital_outputs={"right": [True]},
        joint_trajectory_cancel_clients={"right": client}, move_stop_clients={"right": client},
        controller_switch_client=client, wait_for_controller_state=Mock(return_value=True),
    )
    cases = {
        "generic": (lambda: WeldGuiNode._call_service_sync(client, object(), "/test"),
                    "ack", "ROS service timeout: /test", 1.0, 3.0),
        "equipment": (lambda: WeldGuiNode._call_service_and_wait(client, object(), "equipment"),
                      "ack", "equipment timed out", 2.0, 10.0),
        "legacy": (lambda: WeldGuiNode._set_legacy_digital_output_sync(node, 0, True),
                   "ack · system_state confirmed", "RBPodo digital output command timed out", 2.0, 3.0),
        "cancel": (lambda: WeldGuiNode.cancel_controller_goals(node, "right"),
                   "return_code=0 · goals_canceling=1", "cancel response timed out", 0.25, 1.0),
        "stop": (lambda: WeldGuiNode.request_direct_motion_stop(node, "right"),
                 "controlled move_stop completed", "service response timed out", 0.25, 3.0),
        "switch": (lambda: WeldGuiNode.switch_arm_controller(node, "right", True),
                   "right_manipulator_controller activated",
                   "right_manipulator_controller switch timed out", 0.5, 4.0),
    }
    return client, node, response, cases[request.param]


def test_service_response_contract_and_discovery_timeout(service_case):
    client, _node, _response, (invoke, success, _timeout, discovery_s, _) = service_case
    assert invoke() == (True, success)
    client.wait_for_service.assert_called_once_with(timeout_sec=discovery_s)
    client.call_async.assert_called_once()


def test_service_future_exception_is_reported(service_case):
    client, node, _response, (invoke, *_) = service_case
    failed = Future()
    failed.set_exception(ValueError("bad reply"))
    client.call_async.return_value = failed
    assert invoke() == (False, "bad reply")
    node.wait_for_controller_state.assert_not_called()


def test_service_deadline_and_message_are_preserved(service_case, monkeypatch):
    client, node, _response, (invoke, _, timeout_message, _, response_s) = service_case
    client.call_async.return_value = Future()
    completed = SimpleNamespace(set=Mock(), wait=Mock(return_value=False))
    monkeypatch.setattr(action_call.threading, "Event", lambda: completed)
    assert invoke() == (False, timeout_message)
    completed.wait.assert_called_once_with(response_s)
    node.wait_for_controller_state.assert_not_called()


def test_unavailable_service_never_submits_a_request(service_case):
    client, _node, _response, (invoke, *_) = service_case
    client.wait_for_service.return_value = False
    success, message = invoke()
    assert not success and "unavailable" in message
    client.call_async.assert_not_called()


def test_controller_switch_still_requires_readback():
    future = Future()
    future.set_result(SimpleNamespace(ok=True))
    client = SimpleNamespace(wait_for_service=lambda **_: True,
                             call_async=Mock(return_value=future))
    node = SimpleNamespace(controller_switch_client=client,
                           wait_for_controller_state=Mock(return_value=False))
    assert WeldGuiNode.switch_arm_controller(node, "right", False) == (
        False, "right_manipulator_controller did not report inactive after switch",
    )
    node.wait_for_controller_state.assert_called_once_with(
        "right_manipulator_controller", "inactive",
    )
