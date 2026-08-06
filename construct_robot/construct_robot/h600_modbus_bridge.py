import socket
import struct
import threading
from collections import defaultdict
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from construct_msgs.msg import ModbusTrace, WelderStatus
from construct_msgs.srv import (
    GetModbusRegisters,
    SetModbusServer,
    SetWelderCommand,
)


COMMAND_BASE = 201
COMMAND_COUNT = 10
H600_PORT = 502
STATUS_ADDRESS = 211
CURRENT_FEEDBACK_ADDRESS = 212
VOLTAGE_FEEDBACK_ADDRESS = 213
TRACE_QOS = QoSProfile(
    depth=200,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


@dataclass
class H600State:
    """Thread-safe H600 command and feedback register state."""

    registers: dict = field(default_factory=lambda: defaultdict(int))
    robot_ready: bool = False
    command_robot_error: bool = False
    command_touch: bool = False
    gas: bool = False
    reverse_inching: bool = False
    inching: bool = False
    arc: bool = False
    current_raw: int = 0
    voltage_raw: int = 0
    v_offset_raw: int = 0
    client_connected: bool = False
    client_address: str = ""
    server_running: bool = False
    bind_address: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock)

    def command_registers(self):
        """Return register values for H600 addresses 201 through 210."""
        values = [0] * COMMAND_COUNT
        with self.lock:
            values[0] = int(self.robot_ready)
            values[1] = (
                (int(self.command_robot_error) << 7)
                | (int(self.command_touch) << 4)
                | (int(self.gas) << 3)
                | (int(self.reverse_inching) << 2)
                | (int(self.inching) << 1)
                | int(self.arc)
            )
            values[3] = self.current_raw
            values[4] = self.voltage_raw
            values[5] = self.v_offset_raw
        return values

    def clear_outputs(self):
        """Force all weld-producing outputs and setpoints to safe values."""
        with self.lock:
            self.robot_ready = False
            self.command_robot_error = False
            self.command_touch = False
            self.gas = False
            self.reverse_inching = False
            self.inching = False
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
        self.declare_parameter("allow_arc_output", False)
        self.declare_parameter("allow_nonzero_setpoints", False)
        self.declare_parameter("auto_start", True)
        self._state = H600State()
        self._protocol = H600Protocol(self._state, self.get_logger())
        self._shutdown = threading.Event()
        self._server_stop = None
        self._server_socket = None
        self._client_socket = None
        self._thread = None
        self._server_lock = threading.RLock()
        self._status_publisher = self.create_publisher(
            WelderStatus,
            "/h600/status",
            10,
        )
        self._trace_publisher = self.create_publisher(
            ModbusTrace,
            "/h600/traffic",
            TRACE_QOS,
        )
        self._command_service = self.create_service(
            SetWelderCommand,
            "/h600/set_command",
            self._set_command,
        )
        self._server_service = self.create_service(
            SetModbusServer,
            "/h600/set_server",
            self._set_server,
        )
        self._register_service = self.create_service(
            GetModbusRegisters,
            "/h600/get_registers",
            self._get_registers,
        )
        self.create_timer(0.2, self._publish_status)
        if self.get_parameter("auto_start").value:
            self._start_server(
                self.get_parameter("host").value,
                H600_PORT,
            )

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
            self._state.command_robot_error = request.robot_error
            self._state.command_touch = request.touch
            self._state.gas = request.gas
            self._state.reverse_inching = request.reverse_inching
            self._state.inching = request.inching
            self._state.arc = request.arc
            self._state.current_raw = request.current_raw
            self._state.voltage_raw = request.voltage_raw
            self._state.v_offset_raw = request.v_offset_raw
        control_word = self._state.command_registers()[1]
        response.success = True
        response.message = (
            f"H600 command ready={request.robot_ready}, "
            f"gas={request.gas}, arc={request.arc}, "
            f"202=0x{control_word:04X}, "
            f"current={request.current_raw}, voltage={request.voltage_raw}"
        )
        self.get_logger().info(response.message)
        return response

    def _set_server(self, request, response):
        if request.start:
            success, message = self._start_server(request.host, request.port)
        else:
            success, message = self._stop_server()
        response.success = success
        response.message = message
        return response

    def _get_registers(self, request, response):
        count = int(request.count)
        start = int(request.start_address)
        if count < 1 or count > 1000 or start + count > 65536:
            response.success = False
            response.message = "Range must contain 1..1000 uint16 registers"
            return response
        commands = self._state.command_registers()
        with self._state.lock:
            response.values = [
                commands[address - COMMAND_BASE]
                if COMMAND_BASE <= address < COMMAND_BASE + COMMAND_COUNT
                else self._state.registers[address] & 0xFFFF
                for address in range(start, start + count)
            ]
        response.success = True
        response.message = f"Read registers {start}..{start + count - 1}"
        return response

    def _start_server(self, host, port):
        host = str(host).strip() or "0.0.0.0"
        port = int(port)
        if port != H600_PORT:
            return False, "H600 Modbus TCP port is fixed to 502"
        with self._server_lock:
            if self._thread is not None and self._thread.is_alive():
                return False, "Modbus server is already running"
            stop_event = threading.Event()
            startup_event = threading.Event()
            startup_result = {}
            self._server_stop = stop_event
            self._thread = threading.Thread(
                target=self._serve,
                args=(host, port, stop_event, startup_event, startup_result),
                daemon=True,
            )
            self._thread.start()
        if not startup_event.wait(timeout=1.0):
            self._stop_server()
            return False, f"Timed out while binding {host}:{port}"
        error = startup_result.get("error")
        if error:
            return False, f"Cannot bind {host}:{port}: {error}"
        return True, f"Modbus TCP server listening on {host}:{port}"

    def _stop_server(self):
        with self._server_lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                return True, "Modbus server is already stopped"
            self._server_stop.set()
            for sock in (self._client_socket, self._server_socket):
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    try:
                        sock.close()
                    except OSError:
                        pass
        thread.join(timeout=1.0)
        self._state.clear_outputs()
        return True, "Modbus TCP server stopped; outputs forced OFF"

    @staticmethod
    def _recv_exact(connection, size):
        data = bytearray()
        while len(data) < size:
            chunk = connection.recv(size - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def _serve(
        self,
        host,
        port,
        stop_event,
        startup_event,
        startup_result,
    ):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                self._server_socket = server
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((host, port))
                server.listen(2)
                server.settimeout(0.5)
                with self._state.lock:
                    self._state.server_running = True
                    self._state.bind_address = f"{host}:{port}"
                startup_event.set()
                self.get_logger().info(
                    f"H600 Modbus server listening on {host}:{port}; "
                    f"ARC allowed={self.get_parameter('allow_arc_output').value}"
                )
                while not stop_event.is_set() and not self._shutdown.is_set():
                    try:
                        connection, address = server.accept()
                    except socket.timeout:
                        continue
                    self._handle_client(connection, address, stop_event)
        except OSError as error:
            startup_result["error"] = str(error)
            startup_event.set()
            if not stop_event.is_set() and not self._shutdown.is_set():
                self.get_logger().error(
                    f"H600 Modbus server failed on {host}:{port}: {error}"
                )
        finally:
            startup_event.set()
            self._server_socket = None
            self._client_socket = None
            with self._state.lock:
                self._state.server_running = False
                self._state.bind_address = ""
                self._state.client_connected = False
                self._state.client_address = ""
            self._state.clear_outputs()

    def _handle_client(self, connection, address, stop_event):
        self._client_socket = connection
        address_text = f"{address[0]}:{address[1]}"
        with self._state.lock:
            self._state.client_connected = True
            self._state.client_address = address_text
        self.get_logger().info(f"H600 connected: {address_text}")
        connection.settimeout(0.5)
        with connection:
            while not stop_event.is_set() and not self._shutdown.is_set():
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
        self._client_socket = None
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
            message.server_running = self._state.server_running
            message.bind_address = self._state.bind_address
            message.client_connected = self._state.client_connected
            message.client_address = self._state.client_address
            message.robot_ready = self._state.robot_ready
            message.command_robot_error = self._state.command_robot_error
            message.command_touch = self._state.command_touch
            message.gas = self._state.gas
            message.reverse_inching = self._state.reverse_inching
            message.inching = self._state.inching
            message.arc = self._state.arc
            message.current_command_raw = self._state.current_raw
            message.voltage_command_raw = self._state.voltage_raw
            message.v_offset_command_raw = self._state.v_offset_raw
            message.status_raw = raw
            message.heartbeat = (raw >> 8) & 0xFF
            message.welder_info = raw & 0x03
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
        self._shutdown.set()
        self._stop_server()
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
