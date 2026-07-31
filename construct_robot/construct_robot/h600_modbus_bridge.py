import socket
import struct
import threading
from collections import defaultdict
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node

from construct_msgs.msg import ModbusTrace, WelderStatus
from construct_msgs.srv import SetWelderCommand


COMMAND_BASE = 201
COMMAND_COUNT = 10
STATUS_ADDRESS = 211
CURRENT_FEEDBACK_ADDRESS = 212
VOLTAGE_FEEDBACK_ADDRESS = 213


@dataclass
class H600State:
    """Thread-safe H600 command and feedback register state."""

    registers: dict = field(default_factory=lambda: defaultdict(int))
    robot_ready: bool = False
    gas: bool = False
    arc: bool = False
    current_raw: int = 0
    voltage_raw: int = 0
    v_offset_raw: int = 0
    client_connected: bool = False
    client_address: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock)

    def command_registers(self):
        """Return register values for H600 addresses 201 through 210."""
        values = [0] * COMMAND_COUNT
        with self.lock:
            values[0] = int(self.robot_ready)
            values[1] = (int(self.gas) << 3) | int(self.arc)
            values[3] = self.current_raw
            values[4] = self.voltage_raw
            values[5] = self.v_offset_raw
        return values

    def clear_outputs(self):
        """Force all weld-producing outputs and setpoints to safe values."""
        with self.lock:
            self.robot_ready = False
            self.gas = False
            self.arc = False
            self.current_raw = 0
            self.voltage_raw = 0
            self.v_offset_raw = 0


class H600Protocol:
    """Minimal Modbus TCP PDU implementation based on ~/test.py."""

    def __init__(self, state, logger):
        self.state = state
        self.logger = logger

    @staticmethod
    def exception(function_code, exception_code):
        return bytes([function_code | 0x80, exception_code])

    def process_pdu(self, pdu):
        """Process FC03, FC06, or FC16 and return a response PDU."""
        if not pdu:
            return b""
        function_code = pdu[0]
        if function_code == 0x03:
            if len(pdu) != 5:
                return self.exception(function_code, 0x03)
            start, quantity = struct.unpack(">HH", pdu[1:5])
            if not 1 <= quantity <= 125:
                return self.exception(function_code, 0x03)
            if start == COMMAND_BASE and quantity == COMMAND_COUNT:
                values = self.state.command_registers()
            else:
                with self.state.lock:
                    values = [
                        self.state.registers[start + index] & 0xFFFF
                        for index in range(quantity)
                    ]
            payload = b"".join(struct.pack(">H", value) for value in values)
            return bytes([function_code, len(payload)]) + payload

        if function_code == 0x06:
            if len(pdu) != 5:
                return self.exception(function_code, 0x03)
            address, value = struct.unpack(">HH", pdu[1:5])
            with self.state.lock:
                self.state.registers[address] = value
            return pdu

        if function_code == 0x10:
            if len(pdu) < 6:
                return self.exception(function_code, 0x03)
            start, quantity = struct.unpack(">HH", pdu[1:5])
            byte_count = pdu[5]
            payload = pdu[6:]
            if (
                quantity < 1
                or quantity > 123
                or byte_count != quantity * 2
                or len(payload) != byte_count
            ):
                return self.exception(function_code, 0x03)
            values = struct.unpack(f">{quantity}H", payload)
            with self.state.lock:
                for index, value in enumerate(values):
                    self.state.registers[start + index] = value
            return struct.pack(">BHH", function_code, start, quantity)

        self.logger.warning(
            f"Unsupported H600 Modbus function 0x{function_code:02X}"
        )
        return self.exception(function_code, 0x01)


class H600ModbusBridge(Node):
    """Expose the H600 Modbus server state through ROS services/topics."""

    def __init__(self):
        super().__init__("h600_modbus_bridge")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 502)
        self.declare_parameter("allow_arc_output", False)
        self.declare_parameter("allow_nonzero_setpoints", False)
        self._state = H600State()
        self._protocol = H600Protocol(self._state, self.get_logger())
        self._stop = threading.Event()
        self._server_socket = None
        self._status_publisher = self.create_publisher(
            WelderStatus,
            "/h600/status",
            10,
        )
        self._trace_publisher = self.create_publisher(
            ModbusTrace,
            "/h600/traffic",
            100,
        )
        self._command_service = self.create_service(
            SetWelderCommand,
            "/h600/set_command",
            self._set_command,
        )
        self.create_timer(0.2, self._publish_status)
        self._thread = threading.Thread(
            target=self._serve,
            daemon=True,
        )
        self._thread.start()

    def _set_command(self, request, response):
        allow_arc = self.get_parameter("allow_arc_output").value
        allow_nonzero = self.get_parameter(
            "allow_nonzero_setpoints"
        ).value
        has_nonzero = bool(
            request.current_raw
            or request.voltage_raw
            or request.v_offset_raw
        )
        with self._state.lock:
            client_connected = self._state.client_connected
        if request.arc and not allow_arc:
            response.success = False
            response.message = (
                "ARC blocked: launch with allow_arc_output:=true"
            )
            return response
        if request.arc and not client_connected:
            response.success = False
            response.message = "ARC blocked: H600 Modbus client is disconnected"
            return response
        if request.arc and not request.robot_ready:
            response.success = False
            response.message = "ARC blocked: robot_ready must be true"
            return response
        if (
            has_nonzero
            and (
                not allow_nonzero
                or not request.allow_nonzero_setpoints
            )
        ):
            response.success = False
            response.message = (
                "Nonzero weld values blocked by both safety locks"
            )
            return response
        with self._state.lock:
            self._state.robot_ready = request.robot_ready
            self._state.gas = request.gas
            self._state.arc = request.arc
            self._state.current_raw = request.current_raw
            self._state.voltage_raw = request.voltage_raw
            self._state.v_offset_raw = request.v_offset_raw
        response.success = True
        response.message = (
            f"H600 command ready={request.robot_ready}, "
            f"gas={request.gas}, arc={request.arc}, "
            f"current={request.current_raw}, voltage={request.voltage_raw}"
        )
        self.get_logger().info(response.message)
        return response

    @staticmethod
    def _recv_exact(connection, size):
        data = bytearray()
        while len(data) < size:
            chunk = connection.recv(size - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def _serve(self):
        host = self.get_parameter("host").value
        port = self.get_parameter("port").value
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                self._server_socket = server
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((host, port))
                server.listen(2)
                server.settimeout(0.5)
                self.get_logger().info(
                    f"H600 Modbus server listening on {host}:{port}; "
                    f"ARC allowed={self.get_parameter('allow_arc_output').value}"
                )
                while not self._stop.is_set():
                    try:
                        connection, address = server.accept()
                    except socket.timeout:
                        continue
                    self._handle_client(connection, address)
        except OSError as error:
            if not self._stop.is_set():
                self.get_logger().error(
                    f"H600 Modbus server failed on {host}:{port}: {error}"
                )
        finally:
            self._server_socket = None
            self._state.clear_outputs()

    def _handle_client(self, connection, address):
        address_text = f"{address[0]}:{address[1]}"
        with self._state.lock:
            self._state.client_connected = True
            self._state.client_address = address_text
        self.get_logger().info(f"H600 connected: {address_text}")
        connection.settimeout(0.5)
        with connection:
            while not self._stop.is_set():
                try:
                    header = self._recv_exact(connection, 7)
                    if header is None:
                        break
                    transaction, protocol, length, unit = struct.unpack(
                        ">HHHB",
                        header,
                    )
                    if protocol != 0 or length < 2 or length > 254:
                        break
                    pdu = self._recv_exact(connection, length - 1)
                    if pdu is None:
                        break
                    request_frame = header + pdu
                    register_address, register_count = (
                        self._request_register_range(pdu)
                    )
                    self._publish_trace(
                        "RX",
                        address_text,
                        transaction,
                        unit,
                        pdu,
                        request_frame,
                        register_address,
                        register_count,
                    )
                    response_pdu = self._protocol.process_pdu(pdu)
                    response_header = struct.pack(
                        ">HHHB",
                        transaction,
                        0,
                        len(response_pdu) + 1,
                        unit,
                    )
                    response_frame = response_header + response_pdu
                    connection.sendall(response_frame)
                    self._publish_trace(
                        "TX",
                        address_text,
                        transaction,
                        unit,
                        response_pdu,
                        response_frame,
                        register_address,
                        register_count,
                    )
                except socket.timeout:
                    continue
                except (ConnectionError, OSError):
                    break
        with self._state.lock:
            self._state.client_connected = False
            self._state.client_address = ""
        self._state.clear_outputs()
        self.get_logger().warning(
            "H600 disconnected; ARC/gas/ready forced OFF"
        )

    @staticmethod
    def _request_register_range(pdu):
        if len(pdu) >= 5 and pdu[0] in (0x03, 0x06, 0x10):
            address, value = struct.unpack(">HH", pdu[1:5])
            count = 1 if pdu[0] == 0x06 else value
            return address, count
        return 0, 0

    def _publish_trace(
        self,
        direction,
        client_address,
        transaction,
        unit,
        pdu,
        frame,
        register_address,
        register_count,
    ):
        message = ModbusTrace()
        message.header.stamp = self.get_clock().now().to_msg()
        message.direction = direction
        message.client_address = client_address
        message.transaction_id = transaction
        message.unit_id = unit
        message.function_code = pdu[0] if pdu else 0
        message.register_address = register_address
        message.register_count = register_count
        message.raw_hex = frame.hex(" ").upper()
        function_text = (
            f"EXCEPTION 0x{message.function_code:02X}"
            if message.function_code & 0x80
            else f"FC{message.function_code:02d}"
        )
        register_end = register_address + max(register_count - 1, 0)
        message.summary = (
            f"{function_text} registers "
            f"{register_address}..{register_end}"
        )
        self._trace_publisher.publish(message)

    def _publish_status(self):
        message = WelderStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        with self._state.lock:
            raw = self._state.registers[STATUS_ADDRESS] & 0xFFFF
            message.client_connected = self._state.client_connected
            message.client_address = self._state.client_address
            message.robot_ready = self._state.robot_ready
            message.gas = self._state.gas
            message.arc = self._state.arc
            message.current_command_raw = self._state.current_raw
            message.voltage_command_raw = self._state.voltage_raw
            message.v_offset_command_raw = self._state.v_offset_raw
            message.status_raw = raw
            message.welder_error = bool(raw & (1 << 7))
            message.welding = bool(raw & (1 << 5))
            message.touch_detect = bool(raw & (1 << 4))
            message.current_feedback_raw = (
                self._state.registers[CURRENT_FEEDBACK_ADDRESS] & 0xFFFF
            )
            message.voltage_feedback_raw = (
                self._state.registers[VOLTAGE_FEEDBACK_ADDRESS] & 0xFFFF
            )
        self._status_publisher.publish(message)

    def destroy_node(self):
        self._state.clear_outputs()
        self._stop.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        self._thread.join(timeout=1.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = H600ModbusBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
