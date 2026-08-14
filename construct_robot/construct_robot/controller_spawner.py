import argparse
import sys

import rclpy
from controller_manager_msgs.srv import (
    ConfigureController,
    LoadController,
    SwitchController,
)
from rclpy.utilities import remove_ros_args


class ControllerSpawner:
    def __init__(self, node, manager, response_timeout):
        self.node = node
        self.response_timeout = response_timeout
        prefix = manager.rstrip("/")
        self.load_client = node.create_client(
            LoadController, f"{prefix}/load_controller"
        )
        self.configure_client = node.create_client(
            ConfigureController, f"{prefix}/configure_controller"
        )
        self.switch_client = node.create_client(
            SwitchController, f"{prefix}/switch_controller"
        )

    def _call(self, client, request, operation):
        if not client.wait_for_service(timeout_sec=self.response_timeout):
            raise RuntimeError(f"{operation} service unavailable")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=self.response_timeout,
        )
        if not future.done():
            raise TimeoutError(
                f"{operation} response exceeded {self.response_timeout:.0f}s"
            )
        error = future.exception()
        if error is not None:
            raise RuntimeError(f"{operation} failed: {error}")
        return future.result()

    def activate(self, controller_name):
        load_request = LoadController.Request()
        load_request.name = controller_name
        loaded = self._call(
            self.load_client,
            load_request,
            f"load {controller_name}",
        )
        if not loaded.ok:
            raise RuntimeError(f"controller load rejected: {controller_name}")

        configure_request = ConfigureController.Request()
        configure_request.name = controller_name
        configured = self._call(
            self.configure_client,
            configure_request,
            f"configure {controller_name}",
        )
        if not configured.ok:
            raise RuntimeError(
                f"controller configure rejected: {controller_name}"
            )

        switch_request = SwitchController.Request()
        switch_request.activate_controllers = [controller_name]
        switch_request.strictness = SwitchController.Request.STRICT
        switch_request.activate_asap = True
        switch_request.timeout.sec = int(self.response_timeout)
        activated = self._call(
            self.switch_client,
            switch_request,
            f"activate {controller_name}",
        )
        if not activated.ok:
            raise RuntimeError(
                f"controller activation rejected: {controller_name}"
            )
        self.node.get_logger().info(
            f"Configured and activated {controller_name}"
        )


def main(args=None):
    raw_args = sys.argv if args is None else args
    parser = argparse.ArgumentParser()
    parser.add_argument("controllers", nargs="+")
    parser.add_argument("--controller-manager", default="/controller_manager")
    parser.add_argument("--response-timeout", type=float, default=90.0)
    parsed = parser.parse_args(remove_ros_args(args=raw_args)[1:])

    rclpy.init(args=raw_args)
    node = rclpy.create_node("controller_spawner")
    spawner = ControllerSpawner(
        node,
        parsed.controller_manager,
        parsed.response_timeout,
    )
    exit_code = 0
    try:
        for controller_name in parsed.controllers:
            spawner.activate(controller_name)
    except (RuntimeError, TimeoutError) as error:
        node.get_logger().error(str(error))
        exit_code = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
