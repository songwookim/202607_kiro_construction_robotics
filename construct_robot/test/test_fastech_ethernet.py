from construct_robot.fastech_ethernet import (
    EZI_IO_I8O8,
    FastechEthernetClient,
)


class FakeFastechApi:
    def __init__(self):
        self.raw_input = 0
        self.raw_output = 0
        self.connected = False
        self.last_set_output = None

    def FAS_ConnectTCP(self, first, second, third, fourth, board_id):
        self.connected = True
        self.connect_args = (first, second, third, fourth, board_id)
        return 1

    def FAS_GetSlaveInfo(self, board_id):
        return 0, EZI_IO_I8O8, "fake I8O8"

    def FAS_GetInput(self, board_id):
        return 0, self.raw_input, 0

    def FAS_GetOutput(self, board_id):
        return 0, self.raw_output, 0

    def FAS_SetOutput(self, board_id, set_mask, clear_mask):
        self.last_set_output = (board_id, set_mask, clear_mask)
        self.raw_output |= set_mask
        self.raw_output &= ~clear_mask
        return 0

    def FAS_Close(self, board_id):
        self.connected = False


def test_i8o8_uses_physical_channels_with_output_bit_offset():
    api = FakeFastechApi()
    api.raw_input = (1 << 0) | (1 << 7)
    api.raw_output = (1 << (8 + 0)) | (1 << (8 + 6))
    client = FastechEthernetClient("192.168.0.3", 0, api=api, ok_code=0)
    device_type, version = client.connect()
    assert device_type == EZI_IO_I8O8
    assert version == "fake I8O8"
    assert api.connect_args == (192, 168, 0, 3, 0)
    assert client.input_count == 8
    assert client.output_count == 8
    assert client.output_offset == 8

    snapshot = client.read_io()
    assert snapshot.inputs[0] is True
    assert snapshot.inputs[4] is False
    assert snapshot.inputs[5] is False
    assert snapshot.inputs[7] is True
    assert snapshot.outputs[0] is True
    assert snapshot.outputs[4] is False
    assert snapshot.outputs[5] is False
    assert snapshot.outputs[6] is True


def test_set_output_only_changes_requested_physical_i8o8_bit():
    api = FakeFastechApi()
    # Preserve an unrelated physical DO1 while operating physical DO5.
    api.raw_output = 1 << (8 + 1)
    client = FastechEthernetClient("192.168.0.3", 0, api=api, ok_code=0)
    client.connect()

    snapshot = client.set_output(5, True)
    assert api.last_set_output == (0, 1 << (8 + 5), 0)
    assert snapshot.outputs[1] is True
    assert snapshot.outputs[5] is True

    snapshot = client.set_output(5, False)
    assert api.last_set_output == (0, 0, 1 << (8 + 5))
    assert snapshot.outputs[1] is True
    assert snapshot.outputs[5] is False


def test_output_channel_range_is_checked_before_vendor_write():
    api = FakeFastechApi()
    client = FastechEthernetClient("192.168.0.3", 0, api=api, ok_code=0)
    client.connect()
    try:
        client.set_output(8, True)
    except ValueError as error:
        assert "0..7" in str(error)
    else:
        raise AssertionError("invalid physical I8O8 output was accepted")
    assert api.last_set_output is None


def test_default_fastech_address_is_dot_three():
    api = FakeFastechApi()
    client = FastechEthernetClient(api=api, ok_code=0)
    client.connect()
    assert api.connect_args == (192, 168, 0, 3, 0)
