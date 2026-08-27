"""ROS 2 owner node for the Fastech Ezi-IO Ethernet connection."""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

from construct_msgs.msg import DigitalIoState
from construct_msgs.srv import SetDigitalOutput
from construct_robot.fastech_ethernet import FastechEthernetClient


TOUCH_INPUT_CHANNEL = 0
TOUCH_OUTPUT_CHANNEL = 0


class FastechConnectionManager:
    """Serialize ownership and lifecycle of one protocol-adapter instance."""

    def __init__(self, ip_address, board_id, client_factory=None):
        self.ip_address = str(ip_address)
        self.board_id = int(board_id)
        self._client_factory = client_factory or FastechEthernetClient
        self._client = None
        self._lock = threading.RLock()
        self.device_detail = ""

    @property
    def connected(self):
        with self._lock:
            return self._client is not None

    def connect(self):
        with self._lock:
            if self._client is not None:
                return self._client.read_io()
            client = self._client_factory(self.ip_address, self.board_id)
            try:
                device_type, version = client.connect()
                snapshot = client.read_io()
            except Exception:
                client.close()
                raise
            self._client = client
            self.device_detail = (
                f"type {device_type} · {version} · "
                f"I{client.input_count}O{client.output_count} · "
                f"DO mask offset={client.output_offset}"
            )
            return snapshot

    def disconnect(self):
        with self._lock:
            client = self._client
            self._client = None
            self.device_detail = ""
            if client is not None:
                client.close()

    def read_io(self):
        with self._lock:
            if self._client is None:
                raise RuntimeError("Fastech Ethernet is disconnected")
            return self._client.read_io()

    def set_output(self, channel, value):
        with self._lock:
            if self._client is None:
                raise RuntimeError("Fastech Ethernet is disconnected")
            return self._client.set_output(int(channel), bool(value))


class FastechIONode(Node):
    """Expose one Fastech Ezi-IO connection as semantic ROS interfaces."""

    def __init__(self):
        super().__init__("fastech_io_node")
        self.declare_parameter("ip_address", "192.168.0.3")
        self.declare_parameter("board_id", 0)
        self.declare_parameter("poll_period_s", 0.01)
        self.declare_parameter("reconnect_period_s", 1.0)
        self.declare_parameter("auto_connect", True)

        self.ip_address = str(self.get_parameter("ip_address").value)
        self.board_id = int(self.get_parameter("board_id").value)
        self.poll_period_s = max(
            0.001, float(self.get_parameter("poll_period_s").value)
        )
        self.reconnect_period_s = max(
            self.poll_period_s,
            float(self.get_parameter("reconnect_period_s").value),
        )
        self._manager = FastechConnectionManager(
            self.ip_address,
            self.board_id,
        )
        self._connection_requested = threading.Event()
        if bool(self.get_parameter("auto_connect").value):
            self._connection_requested.set()
        self._stop_event = threading.Event()
        self._last_error = ""

        state_qos = QoSProfile(depth=1)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._contact_publisher = self.create_publisher(
            Bool, "/touch/contact", state_qos
        )
        self._state_publisher = self.create_publisher(
            DigitalIoState, "/fastech/io_state", state_qos
        )
        self.create_service(SetBool, "/touch/enable", self._touch_enable)
        self.create_service(
            SetDigitalOutput,
            "/fastech/set_output",
            self._set_output,
        )
        self.create_service(Trigger, "/fastech/connect", self._connect)
        self.create_service(Trigger, "/fastech/disconnect", self._disconnect)

        self._publish_disconnected("starting")
        self._worker = threading.Thread(
            target=self._poll_worker,
            name="fastech-io-poll",
            daemon=True,
        )
        self._worker.start()

    def _state_message(self, snapshot=None, connected=None, detail=""):
        message = DigitalIoState()
        message.connected = (
            self._manager.connected if connected is None else bool(connected)
        )
        message.ip_address = self.ip_address
        message.board_id = self.board_id
        message.poll_rate_hz = float(1.0 / self.poll_period_s)
        if snapshot is not None:
            message.raw_input = int(snapshot.raw_input)
            message.raw_output = int(snapshot.raw_output)
            message.digital_in = list(snapshot.inputs)
            message.digital_out = list(snapshot.outputs)
        message.detail = str(detail)
        return message

    def _publish_snapshot(self, snapshot, detail=""):
        if len(snapshot.inputs) <= TOUCH_INPUT_CHANNEL:
            raise RuntimeError("Fastech device does not expose physical DI0")
        self._state_publisher.publish(
            self._state_message(
                snapshot,
                connected=True,
                detail=detail or self._manager.device_detail,
            )
        )
        contact = Bool()
        contact.data = bool(snapshot.inputs[TOUCH_INPUT_CHANNEL])
        self._contact_publisher.publish(contact)

    def _publish_disconnected(self, detail):
        self._state_publisher.publish(
            self._state_message(connected=False, detail=detail)
        )

    def _attempt_connect(self):
        try:
            snapshot = self._manager.connect()
            self._last_error = ""
            self._publish_snapshot(snapshot)
            self.get_logger().info(
                f"Fastech connected · {self.ip_address} · board "
                f"{self.board_id} · {self._manager.device_detail} · "
                f"poll target={1.0 / self.poll_period_s:.0f} Hz"
            )
            return True, self._manager.device_detail
        except Exception as error:
            self._manager.disconnect()
            message = str(error)
            if message != self._last_error:
                self.get_logger().warning(f"Fastech connection failed · {message}")
                self._last_error = message
            self._publish_disconnected(message)
            return False, message

    def _poll_worker(self):
        next_connect_at = 0.0
        while not self._stop_event.is_set():
            if not self._connection_requested.is_set():
                self._stop_event.wait(0.1)
                continue
            if not self._manager.connected:
                now = time.monotonic()
                if now < next_connect_at:
                    self._stop_event.wait(
                        min(0.1, max(0.0, next_connect_at - now))
                    )
                    continue
                connected, _ = self._attempt_connect()
                if not connected:
                    next_connect_at = time.monotonic() + self.reconnect_period_s
                    continue
            cycle_started = time.monotonic()
            try:
                snapshot = self._manager.read_io()
                self._publish_snapshot(snapshot)
            except Exception as error:
                message = str(error)
                self.get_logger().warning(
                    f"Fastech connection lost; reconnecting · {message}"
                )
                self._manager.disconnect()
                self._publish_disconnected(message)
                next_connect_at = time.monotonic() + self.reconnect_period_s
                continue
            remaining = self.poll_period_s - (time.monotonic() - cycle_started)
            if remaining > 0.0:
                self._stop_event.wait(remaining)

    def _connect(self, _request, response):
        self._connection_requested.set()
        success, message = self._attempt_connect()
        response.success = success
        response.message = (
            f"Fastech connected: {message}"
            if success
            else f"Fastech connection requested; retry active: {message}"
        )
        return response

    def _disconnect(self, _request, response):
        self._connection_requested.clear()
        self._manager.disconnect()
        self._publish_disconnected("disconnected by service request")
        response.success = True
        response.message = "Fastech disconnected; outputs were not changed"
        return response

    def _command_output(self, channel, value):
        try:
            snapshot = self._manager.set_output(channel, value)
            self._publish_snapshot(snapshot)
            return True, f"Fastech DO{channel} readback verified"
        except Exception as error:
            return False, str(error)

    def _touch_enable(self, request, response):
        response.success, response.message = self._command_output(
            TOUCH_OUTPUT_CHANNEL, request.data
        )
        return response

    def _set_output(self, request, response):
        response.success, response.message = self._command_output(
            request.channel, request.value
        )
        return response

    def destroy_node(self):
        self._connection_requested.clear()
        self._stop_event.set()
        if hasattr(self, "_worker"):
            self._worker.join(timeout=2.0)
        self._manager.disconnect()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FastechIONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
