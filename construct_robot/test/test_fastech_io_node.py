from pathlib import Path

from construct_robot.fastech_ethernet import FastechIoSnapshot
from construct_robot.fastech_io_node import (
    FastechConnectionManager,
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


def test_touch_semantic_interface_maps_to_physical_channel_zero():
    assert TOUCH_INPUT_CHANNEL == 0
    assert TOUCH_OUTPUT_CHANNEL == 0


def test_gui_does_not_own_or_poll_the_fastech_protocol_adapter():
    gui_source = (
        Path(__file__).parents[1] / "construct_robot" / "weld_action_gui.py"
    ).read_text(encoding="utf-8")

    assert "FastechEthernetClient" not in gui_source
    assert "def _fastech_connect_worker" not in gui_source
    assert "def _fastech_poll_worker" not in gui_source
    assert ".read_io(" not in gui_source
