"""Publish physical arrow-key state without X11 auto-repeat ambiguity."""

import ctypes
import ctypes.util
import os
import threading
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
# Keyboard-jog axis selection keys (GUI keyboard_jog_selection_key), reported
# as their character code on each press so the GUI node can re-aim a held jog
# without waiting for the Tk event loop.
SELECTION_KEYSYMS = {ord(c): ord(c) for c in "123456789asdv"}


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
        self._selection_keycodes = {
            self._x11.XKeysymToKeycode(self._display, keysym): code
            for keysym, code in SELECTION_KEYSYMS.items()
        }
        self._keys = (ctypes.c_ubyte * 32)()

    def _query(self):
        if not self._x11.XQueryKeymap(self._display, self._keys):
            raise RuntimeError("XQueryKeymap failed")
        return self._keys

    @staticmethod
    def _down(keys, keycode):
        return keycode and keys[keycode >> 3] & (1 << (keycode & 7))

    def read_mask(self):
        return self.read_state()[0]

    def read_state(self):
        """(arrow bit mask, set of selection-key codes held) from one query."""
        keys = self._query()
        mask = 0
        for keycode, bit in self._keycodes.items():
            if self._down(keys, keycode):
                mask |= bit
        held = {code for keycode, code in self._selection_keycodes.items()
                if self._down(keys, keycode)}
        return mask, held

    def close(self):
        if self._display:
            self._x11.XCloseDisplay(self._display)
            self._display = None


class KeyboardTeachingNode(Node):
    """Poll the X11 keymap in a plain thread and publish arrow-state edges.

    An rclpy timer pays the executor's dispatch cost on every poll (18% of a
    core at 200 Hz on the Jetson); this loop only queries X11 and compares a
    byte, so it polls at 500 Hz for less.  Publishing is thread-safe.
    """

    def __init__(self):
        super().__init__("keyboard_teaching_node")
        self.declare_parameter("poll_period_s", 0.002)
        self._keyboard = X11KeyboardState()
        self._publisher = self.create_publisher(
            UInt8, "/keyboard_teaching/arrow_state", 10
        )
        self._selection_publisher = self.create_publisher(
            UInt8, "/keyboard_teaching/selection_key", 10
        )
        self._period = max(0.001, float(self.get_parameter("poll_period_s").value))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop, name="x11-arrow-poll", daemon=True
        )
        self._thread.start()
        self.get_logger().info(
            f"Physical arrow-key state polling active at {1.0 / self._period:.1f} Hz"
        )

    def _poll_loop(self):
        last_mask = None
        last_publish_at = 0.0
        held_selection = set()
        next_poll = time.monotonic()
        while not self._stop.is_set() and rclpy.ok():
            read_state = getattr(self._keyboard, "read_state", None)
            if read_state is not None:
                mask, selection = read_state()
            else:
                mask, selection = self._keyboard.read_mask(), set()
            for code in sorted(selection - held_selection):
                self._selection_publisher.publish(UInt8(data=code))
            held_selection = selection
            now = time.monotonic()
            # Publish every physical edge immediately. A low-rate unchanged-state
            # heartbeat lets the GUI detect the node without flooding its Tk queue.
            if mask != last_mask or now - last_publish_at >= 0.1:
                message = UInt8()
                message.data = mask
                self._publisher.publish(message)
                last_mask = mask
                last_publish_at = now
            next_poll += self._period
            delay = next_poll - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                next_poll = time.monotonic()

    def destroy_node(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
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
