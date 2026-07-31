import ipaddress
import os
import signal
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node

from construct_msgs.srv import SetRobotConnection


class WeldStackSupervisor(Node):
    """Own the real-RB weld stack while the connection GUI stays alive."""

    def __init__(self):
        super().__init__("weld_stack_supervisor")
        self.declare_parameter("initial_connected", False)
        self.declare_parameter("right_robot_ip", "192.168.1.10")
        self.declare_parameter("execute_motion", True)
        self.declare_parameter("use_rviz", True)
        self.declare_parameter("use_viser", True)
        self.declare_parameter("use_h600_gui", False)
        self.declare_parameter("h600_port", 1502)
        self._initial_connected = self.get_parameter(
            "initial_connected"
        ).value
        self._right_robot_ip = self.get_parameter("right_robot_ip").value
        self._process = None
        self._process_lock = threading.Lock()
        self._restart_in_progress = False
        self._shutting_down = False
        self._service = self.create_service(
            SetRobotConnection,
            "/weld_stack/set_robot_connection",
            self._connection_request,
        )
        self._initial_timer = self.create_timer(
            0.5,
            self._start_initial_stack,
        )

    @staticmethod
    def _bool_text(value):
        return "true" if value else "false"

    def _launch_command(self):
        execute_motion = self.get_parameter("execute_motion").value
        use_rviz = self.get_parameter("use_rviz").value
        use_viser = self.get_parameter("use_viser").value
        use_h600_gui = self.get_parameter("use_h600_gui").value
        h600_port = self.get_parameter("h600_port").value
        return [
            "ros2",
            "launch",
            "construct_robot",
            "weld_action_gui.launch.py",
            "use_gui:=false",
            f"right_robot_ip:={self._right_robot_ip}",
            f"execute_motion:={self._bool_text(execute_motion)}",
            f"use_rviz:={self._bool_text(use_rviz)}",
            f"use_viser:={self._bool_text(use_viser)}",
            f"use_h600_gui:={self._bool_text(use_h600_gui)}",
            f"h600_port:={h600_port}",
            "allow_arc_output:=false",
            "allow_nonzero_setpoints:=false",
        ]

    def _start_initial_stack(self):
        self._initial_timer.cancel()
        if self._initial_connected:
            self._start_stack()
        else:
            self.get_logger().info(
                "Robot disconnected; waiting for Robot Connect"
            )

    def _start_stack(self):
        if self._shutting_down:
            return
        command = self._launch_command()
        self.get_logger().info(
            f"Connecting REAL right-arm weld stack "
            f"(IP={self._right_robot_ip}, ARC locked OFF)"
        )
        with self._process_lock:
            self._process = subprocess.Popen(
                command,
                start_new_session=True,
            )

    def _stop_stack(self):
        with self._process_lock:
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            return
        self.get_logger().info("Stopping owned weld launch...")
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=20.0)
            return
        except subprocess.TimeoutExpired:
            self.get_logger().warning(
                "Weld launch did not stop after SIGINT; sending SIGTERM"
            )
        except ProcessLookupError:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.get_logger().error(
                "Weld launch did not stop after SIGTERM; sending SIGKILL"
            )
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)
        except ProcessLookupError:
            pass

    def _connection_request(self, request, response):
        if request.connect:
            try:
                ipaddress.ip_address(request.right_robot_ip)
            except ValueError:
                response.accepted = False
                response.message = (
                    f"Invalid right robot IP: {request.right_robot_ip}"
                )
                return response
        with self._process_lock:
            if self._restart_in_progress:
                response.accepted = False
                response.message = "A robot connection change is already running"
                return response
            self._restart_in_progress = True
        response.accepted = True
        operation = "connect" if request.connect else "disconnect"
        response.message = f"Accepted Robot {operation}"
        threading.Thread(
            target=self._connection_worker,
            args=(
                request.connect,
                request.right_robot_ip,
            ),
            daemon=True,
        ).start()
        return response

    def _connection_worker(self, connect, right_robot_ip):
        try:
            time.sleep(0.8)
            self._stop_stack()
            if connect:
                self._right_robot_ip = right_robot_ip
                self._start_stack()
            else:
                self.get_logger().info("REAL right-arm robot disconnected")
        finally:
            with self._process_lock:
                self._restart_in_progress = False

    def destroy_node(self):
        self._shutting_down = True
        self._stop_stack()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WeldStackSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
