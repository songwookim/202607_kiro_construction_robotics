"""Publish physical arrow-key state without X11 auto-repeat ambiguity."""

import ctypes
import ctypes.util
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8


ARROW_KEYSYMS = {
    0xFF51: 1 << 0,  # Left
    0xFF53: 1 << 1,  # Right
    0xFF52: 1 << 2,  # Up
    0xFF54: 1 << 3,  # Down
}


class X11KeyboardState:
    """Small ctypes wrapper around XQueryKeymap; no extra Python package."""

    def __init__(self):
        library = ctypes.util.find_library("X11") or "libX11.so.6"
        self._x11 = ctypes.CDLL(library)
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._x11.XQueryKeymap.restype = ctypes.c_int
        self._x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._x11.XKeysymToKeycode.restype = ctypes.c_uint
        display_name = os.environ.get("DISPLAY")
        self._display = self._x11.XOpenDisplay(
            display_name.encode() if display_name else None
        )
        if not self._display:
            raise RuntimeError(
                f"cannot open X11 display {display_name or '(default)'}"
            )
        self._keycodes = {
            self._x11.XKeysymToKeycode(self._display, keysym): bit
            for keysym, bit in ARROW_KEYSYMS.items()
        }

    def read_mask(self):
        keys = (ctypes.c_ubyte * 32)()
        if not self._x11.XQueryKeymap(self._display, keys):
            raise RuntimeError("XQueryKeymap failed")
        mask = 0
        for keycode, bit in self._keycodes.items():
            if keycode and keys[keycode >> 3] & (1 << (keycode & 7)):
                mask |= bit
        return mask

    def close(self):
        if self._display:
            self._x11.XCloseDisplay(self._display)
            self._display = None


class KeyboardTeachingNode(Node):
    def __init__(self):
        super().__init__("keyboard_teaching_node")
        self.declare_parameter("poll_period_s", 0.01)
        self._keyboard = X11KeyboardState()
        self._publisher = self.create_publisher(
            UInt8, "/keyboard_teaching/arrow_state", 10
        )
        self._last_mask = None
        self._last_publish_at = 0.0
        period = max(0.005, float(self.get_parameter("poll_period_s").value))
        self.create_timer(period, self._poll)
        self.get_logger().info(
            f"Physical arrow-key state polling active at {1.0 / period:.1f} Hz"
        )

    def _poll(self):
        mask = self._keyboard.read_mask()
        now = time.monotonic()
        # Publish every physical edge immediately. A low-rate unchanged-state
        # heartbeat lets the GUI detect the node without flooding its Tk queue.
        if mask == self._last_mask and now - self._last_publish_at < 0.1:
            return
        message = UInt8()
        message.data = mask
        self._publisher.publish(message)
        self._last_mask = mask
        self._last_publish_at = now

    def destroy_node(self):
        self._keyboard.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = KeyboardTeachingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
