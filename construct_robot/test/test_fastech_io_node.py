from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from construct_robot.io.fastech_ethernet import FastechIoSnapshot
from construct_robot.nodes.fastech_io_node import (
    FastechConnectionManager,
    FastechIONode,
    TOUCH_INPUT_CHANNEL,
    TOUCH_OUTPUT_CHANNEL,
)


class FakeProtocolClient:
    instances = []

    def __init__(self, ip_address, board_id):
        self.ip = ip_address
        self.board_id = board_id
        self.input_count = 8
        self.output_count = 8
        self.output_offset = 8
        self.connected = False
        self.outputs = [False] * 8
        self.set_calls = []
        self.__class__.instances.append(self)

    def connect(self):
        self.connected = True
        return 155, "fake I8O8"

    def close(self):
        self.connected = False

    def read_io(self):
        if not self.connected:
            raise RuntimeError("not connected")
        return FastechIoSnapshot(
            inputs=(True,) + (False,) * 7,
            outputs=tuple(self.outputs),
            raw_input=1,
            raw_output=sum(
                (1 << (8 + channel))
                for channel, value in enumerate(self.outputs)
                if value
            ),
            latch=0,
            trigger_status=0,
        )

    def set_output(self, channel, value):
        self.set_calls.append((int(channel), bool(value)))
        self.outputs[int(channel)] = bool(value)
        return self.read_io()


def test_connection_manager_is_the_single_protocol_client_owner():
    FakeProtocolClient.instances.clear()
    manager = FastechConnectionManager(
        "192.168.0.3",
        0,
        client_factory=FakeProtocolClient,
    )

    first = manager.connect()
    second = manager.connect()

    assert len(FakeProtocolClient.instances) == 1
    assert first.inputs[0] is True
    assert second.inputs[0] is True
    assert manager.connected is True

    manager.disconnect()
    assert manager.connected is False
    assert FakeProtocolClient.instances[0].connected is False


def test_connection_manager_routes_physical_output_and_returns_readback():
    FakeProtocolClient.instances.clear()
    manager = FastechConnectionManager(
        "192.168.0.3",
        0,
        client_factory=FakeProtocolClient,
    )
    manager.connect()

    snapshot = manager.set_output(6, True)

    client = FakeProtocolClient.instances[0]
    assert client.set_calls == [(6, True)]
    assert snapshot.outputs[6] is True
    assert snapshot.inputs[0] is True


def test_touch_semantic_interface_maps_to_configured_physical_channel():
    assert TOUCH_INPUT_CHANNEL == 4
    assert TOUCH_OUTPUT_CHANNEL == 0


def test_gui_does_not_own_or_poll_the_fastech_protocol_adapter():
    gui_source = (
        Path(__file__).parents[1] / "construct_robot" / "gui" / "weld_action_gui.py"
    ).read_text(encoding="utf-8")

    assert "FastechEthernetClient" not in gui_source
    assert "def _fastech_connect_worker" not in gui_source
    assert "def _fastech_poll_worker" not in gui_source
    assert ".read_io(" not in gui_source


def test_poll_publication_cannot_be_overtaken_by_output_readback():
    node = object.__new__(FastechIONode)
    node._operation_lock = threading.RLock()
    node._connection_requested = threading.Event()
    node._connection_requested.set()
    node._stop_event = threading.Event()
    entered, release, output_read = threading.Event(), threading.Event(), threading.Event()
    published = []

    def publish(snapshot):
        if snapshot == "old":
            entered.set()
            assert release.wait(2)
        published.append(snapshot)

    node._publish_snapshot = publish
    node._manager = SimpleNamespace(
        read_io=lambda: "old",
        set_output=lambda *a: (output_read.set(), "new")[1])
    poll = threading.Thread(target=node._poll_once)
    output = threading.Thread(target=lambda: node._command_output(0, True))
    poll.start()
    try:
        assert entered.wait(1)
        output.start()
        assert not output_read.wait(0.03)
    finally:
        release.set()
        poll.join(1)
        output.join(1)
    assert published == ["old", "new"]


def test_disconnected_poll_and_connect_attempt_do_not_republish_old_state():
    node = object.__new__(FastechIONode)
    node._operation_lock = threading.RLock()
    node._connection_requested = threading.Event()
    node._stop_event = threading.Event()
    node._manager = Mock()
    node._publish_snapshot = Mock()
    assert node._poll_once() is None
    assert node._attempt_connect()[0] is False
    node._manager.read_io.assert_not_called()
    node._manager.connect.assert_not_called()
    node._publish_snapshot.assert_not_called()


def publishing_node():
    from construct_robot.nodes import fastech_io_node as module

    node = object.__new__(FastechIONode)
    node.touch_input_channel = 0
    node.ip_address, node.board_id, node.poll_period_s = "192.168.0.3", 0, 0.01
    node._manager = SimpleNamespace(device_detail="fake I8O8", connected=True)
    node._last_raw_contact = None
    node._published_state = None
    node._published_contact = None
    node._published_at = 0.0
    node._state_publisher = Mock()
    node._contact_publisher = Mock()
    node.get_logger = lambda: Mock()
    return node, module


def snapshot(contact):
    return FastechIoSnapshot(inputs=(contact,) + (False,) * 7, outputs=(False,) * 8,
                             raw_input=int(contact), raw_output=0, latch=0, trigger_status=0)


def test_unchanged_polls_are_not_republished_but_edges_are_immediate(monkeypatch):
    node, module = publishing_node()
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    for _ in range(10):  # 100 ms of unchanged 100 Hz polls
        node._publish_snapshot(snapshot(False))
        clock[0] += 0.01
    assert node._contact_publisher.publish.call_count == 1
    assert node._state_publisher.publish.call_count == 1
    node._publish_snapshot(snapshot(True))  # touch edge: same poll
    assert node._contact_publisher.publish.call_args.args[0].data is True
    assert node._state_publisher.publish.call_count == 2
    clock[0] += module.STATE_HEARTBEAT_S
    node._publish_snapshot(snapshot(True))  # heartbeat keeps the stream alive
    assert node._contact_publisher.publish.call_count == 3


def test_reconnect_republishes_the_contact_even_if_unchanged():
    node, _module = publishing_node()
    node._publish_snapshot(snapshot(False))
    node._publish_disconnected("cable")
    node._publish_snapshot(snapshot(False))
    assert node._contact_publisher.publish.call_count == 2
