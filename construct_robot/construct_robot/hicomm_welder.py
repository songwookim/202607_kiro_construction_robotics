"""Production Hi-COMM digital-welder TCP protocol used by the weld GUI.

The wire format follows the successful Rainbow ARC capture used by
``test_hicomm_control_v5_2.py``: TX55 every 40 ms and RX71.  This module
deliberately does not depend on ROS or a GUI toolkit.
"""

from dataclasses import dataclass, replace
from collections import deque
import select
import socket
import threading
import time


PERIOD_SECONDS = 0.040
CONNECT_RETRY_SECONDS = 0.200
TX_SIZE = 55
RX_SIZE = 71

BIT_ARC = 0x01
BIT_FORWARD = 0x02
BIT_REVERSE = 0x04
BIT_GAS = 0x08
BIT_STICK = 0x10
COMMAND_BITS = BIT_ARC | BIT_FORWARD | BIT_REVERSE | BIT_GAS | BIT_STICK

OUTPUT_STATE_IDLE = 0
OUTPUT_STATE_MAIN_WELD = 1
OUTPUT_STATE_CRATER = 2
OUTPUT_STATE_END = 3
OUTPUT_STATE_NAMES = {
    OUTPUT_STATE_IDLE: "idle",
    OUTPUT_STATE_MAIN_WELD: "main_weld",
    OUTPUT_STATE_CRATER: "crater",
    OUTPUT_STATE_END: "weld_end",
}

MATERIAL_CODES = {
    "FE-SOLID": 0, "FE-CORED": 1, "STS-SOLID": 2,
    "STS-CORED": 3, "AL-SOFT": 4, "AL-HARD": 5,
    "CUSI": 6, "CUMG": 7,
}
DIAMETER_CODES = {0.8: 0, 0.9: 1, 1.0: 2, 1.2: 3, 1.4: 4, 1.6: 5}
MODE_CODES = {"LSM": 0, "DCM": 1, "DPM": 2, "PM": 3}
GAS_CODES = {
    "CO2": 0, "CO2 100%": 0, "AR80+CO2 20%": 1,
    "AR80_CO2_20": 1, "AR98+O2 2%": 2,
    "AR98_O2_2": 2, "AR 100%": 3, "AR100": 3,
}

PROFILE_WELDING = "welding_success"
PROFILE_INCHING = "inching_capture"

CAPTURED_WELDING_BASE_REQUEST = bytes.fromhex(
    "00 0C 00 64 00 64 00 32 "
    "00 00 00 00 33 33 00 00 "
    "0F 00 00 32 32 32 32 32 "
    "32 32 32 32 32 32 32 32 "
    "32 32 32 32 32 32 32 32 "
    "32 32 32 32 32 32 32 32 "
    "32 09 00 00 00 00 00"
)

CAPTURED_INCHING_BASE_REQUEST = bytes.fromhex(
    "00 08 00 64 00 C8 00 32 "
    "00 00 00 00 32 32 64 00 "
    "0F 64 00 32 32 32 32 32 "
    "32 32 32 32 32 32 32 32 "
    "32 32 32 32 32 32 32 32 "
    "32 32 32 32 32 32 32 32 "
    "32 14 0A 00 64 00 00"
)

BASE_PROFILE_REQUESTS = {
    PROFILE_WELDING: CAPTURED_WELDING_BASE_REQUEST,
    PROFILE_INCHING: CAPTURED_INCHING_BASE_REQUEST,
}

# Compatibility alias retained for existing imports/tests.  IDLE now means
# the ARC-success welding profile with Byte0 cleared.
CAPTURED_IDLE_REQUEST = CAPTURED_WELDING_BASE_REQUEST

for _golden in BASE_PROFILE_REQUESTS.values():
    assert len(_golden) == TX_SIZE
    assert _golden[53:55] == b"\x00\x00"


@dataclass
class TxState:
    command: int = 0
    current_a: int = 100
    voltage_tenths: int = 100
    material: str = "FE-SOLID"
    diameter_mm: float = 1.2
    mode: str = "LSM"
    gas: str = "CO2"
    synergic: bool = False
    correction: float = 0.0
    pre_gas_s: float = 0.0
    post_gas_s: float = 0.0
    base_profile: str = PROFILE_WELDING


def _enum_code(value, mapping, name):
    if isinstance(value, int):
        if value in mapping.values():
            return value
        raise ValueError(f"{name} code is invalid: {value}")
    key = str(value).strip().upper()
    if key not in mapping:
        raise ValueError(f"unknown {name}: {value}")
    return mapping[key]


def _diameter_code(value):
    diameter = round(float(value), 1)
    if diameter not in DIAMETER_CODES:
        raise ValueError(f"unsupported wire diameter: {value}")
    return DIAMETER_CODES[diameter]


def _put_u16le(frame, offset, value):
    frame[offset:offset + 2] = int(value).to_bytes(2, "little")


def build_request(state):
    """Return one captured-compatible 55-byte request."""
    if not 30 <= int(state.current_a) <= 400:
        raise ValueError("digital welding current must be in 30..400 A")
    if not 100 <= int(state.voltage_tenths) <= 400:
        raise ValueError("digital welding voltage must be in 10.0..40.0 V")
    if not -5.0 <= float(state.correction) <= 5.0:
        raise ValueError("synergic correction must be in -5.0..5.0")
    if not 0.0 <= float(state.pre_gas_s) <= 10.0:
        raise ValueError("pre-gas must be in 0..10 seconds")
    if not 0.0 <= float(state.post_gas_s) <= 10.0:
        raise ValueError("post-gas must be in 0..10 seconds")
    try:
        base_request = BASE_PROFILE_REQUESTS[state.base_profile]
    except KeyError as error:
        raise ValueError(
            f"unknown Hi-COMM base profile: {state.base_profile}"
        ) from error
    frame = bytearray(base_request)
    frame[0] = int(state.command) & COMMAND_BITS
    frame[1] = (
        (_enum_code(state.material, MATERIAL_CODES, "material") << 5)
        | (_diameter_code(state.diameter_mm) << 2)
        | _enum_code(state.mode, MODE_CODES, "welding mode")
    )
    frame[2] = (
        (0x80 if state.synergic else 0)
        | _enum_code(state.gas, GAS_CODES, "shielding gas")
    )
    _put_u16le(frame, 3, state.current_a)
    _put_u16le(frame, 5, state.voltage_tenths)
    frame[7] = int(round((float(state.correction) + 5.0) * 10.0))
    _put_u16le(frame, 8, round(float(state.pre_gas_s) * 100.0))
    _put_u16le(frame, 10, round(float(state.post_gas_s) * 100.0))
    frame[53:55] = b"\x00\x00"
    request = bytes(frame)
    if len(request) != TX_SIZE:
        raise AssertionError(f"Hi-COMM TX length changed: {len(request)}")
    return request


assert build_request(TxState()) == CAPTURED_WELDING_BASE_REQUEST


def _u16le(frame, offset):
    return frame[offset] | (frame[offset + 1] << 8)


def decode_response(frame):
    """Decode the documented prefix of one captured 71-byte response."""
    if len(frame) != RX_SIZE:
        raise ValueError(f"Hi-COMM RX length {len(frame)} != {RX_SIZE}")
    status = frame[0]
    output_state = frame[1] & 0x03
    wire_feed_m_min = frame[6] / 10.0
    feedback_current_a = _u16le(frame, 2)
    feedback_voltage_v = _u16le(frame, 4) / 10.0
    arc_ack = bool(status & BIT_ARC)
    wcr_detected = bool(status & 0x20)
    arc_established = bool(
        arc_ack
        and output_state == OUTPUT_STATE_MAIN_WELD
        and wcr_detected
        and wire_feed_m_min > 0.0
    )
    feedback_active = bool(
        feedback_current_a > 0 or feedback_voltage_v > 0.0
    )
    if arc_established and feedback_active:
        sequence_stage = "welding_feedback"
    elif arc_established:
        sequence_stage = "arc_established"
    elif arc_ack and status & BIT_FORWARD:
        sequence_stage = "arc_forward"
    elif arc_ack and status & BIT_GAS:
        sequence_stage = "arc_gas"
    elif arc_ack:
        sequence_stage = "arc_recognized"
    elif status & (0x20 | COMMAND_BITS) or wire_feed_m_min > 0.0:
        sequence_stage = "arc_off_tail"
    else:
        sequence_stage = "idle"
    return {
        "timestamp_monotonic": time.monotonic(),
        "raw_frame": bytes(frame),
        "raw0": status,
        "db_unavailable": bool(status & 0x80),
        "torch_collision": bool(status & 0x40),
        "wcr_detected": wcr_detected,
        "arc_ack": arc_ack,
        "forward_ack": bool(status & BIT_FORWARD),
        "reverse_ack": bool(status & BIT_REVERSE),
        "gas_ack": bool(status & BIT_GAS),
        "stick_ack": bool(status & BIT_STICK),
        "output_state": output_state,
        "output_state_name": OUTPUT_STATE_NAMES.get(
            output_state, f"unknown({output_state})"
        ),
        "feedback_current_a": feedback_current_a,
        "feedback_voltage_v": feedback_voltage_v,
        "wire_feed_m_min": wire_feed_m_min,
        "feedback_active": feedback_active,
        "arc_established": arc_established,
        "sequence_stage": sequence_stage,
        "welder_error": frame[9],
        "material_code": (frame[7] >> 5) & 0x07,
        "diameter_code": (frame[7] >> 2) & 0x07,
        "mode_code": frame[7] & 0x03,
        "synergic": bool(frame[8] & 0x80),
        "gas_code": frame[8] & 0x03,
        "set_current_a": _u16le(frame, 10),
        "set_voltage_v": _u16le(frame, 12) / 10.0,
        "correction_raw": frame[14],
        "pre_gas_raw": _u16le(frame, 15),
        "post_gas_raw": _u16le(frame, 17),
        "extra7": frame[64:71].hex(" ").upper(),
    }


class HiCommWelderClient:
    """Maintain the required 40 ms TCP exchange in a background thread."""

    def __init__(
        self,
        source_ip,
        welder_ip,
        port=60000,
        connection_callback=None,
        status_callback=None,
        log_callback=None,
    ):
        self.source_ip = source_ip
        self.welder_ip = welder_ip
        self.port = int(port)
        self.connection_callback = connection_callback or (lambda *_args: None)
        self.status_callback = status_callback or (lambda *_args: None)
        self.log_callback = log_callback or (lambda *_args: None)
        self._lock = threading.RLock()
        self._status_condition = threading.Condition(self._lock)
        self._state = TxState()
        self._latest_status = None
        self._connected = False
        self._stop = threading.Event()
        self._outputs_inhibited = threading.Event()
        self._thread = None
        self._callback_thread = None
        self._callback_stop = threading.Event()
        self._callback_condition = threading.Condition()
        self._callback_events = deque()
        self._pending_callback_status = None
        self._socket = None
        self._state_before_inching = None
        self._last_tx_monotonic = None
        self._last_cadence_warning_monotonic = 0.0
        self._arc_command_generation = 0

    @property
    def connected(self):
        with self._lock:
            return self._connected

    def snapshot(self):
        with self._lock:
            return replace(self._state)

    def latest_status(self):
        with self._lock:
            return None if self._latest_status is None else dict(self._latest_status)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        socket.inet_aton(self.source_ip)
        socket.inet_aton(self.welder_ip)
        if not 1 <= self.port <= 65535:
            raise ValueError("Hi-COMM port must be in 1..65535")
        # Validate the initial setpoints before starting the thread.
        build_request(self.snapshot())
        self._stop.clear()
        self._callback_stop.clear()
        self._callback_thread = threading.Thread(
            target=self._run_callbacks,
            name="HiCommCallbacks",
            daemon=True,
        )
        self._callback_thread.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=2.5):
        self.clear_outputs()
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._callback_stop.set()
        with self._callback_condition:
            self._callback_condition.notify_all()
        callback_thread = self._callback_thread
        if (
            callback_thread is not None
            and callback_thread is not threading.current_thread()
        ):
            callback_thread.join(timeout=timeout)

    def update_setpoints(self, current_a, voltage_tenths):
        candidate = replace(
            self.snapshot(),
            current_a=int(round(current_a)),
            voltage_tenths=int(round(voltage_tenths)),
        )
        build_request(candidate)
        with self._lock:
            self._state.current_a = candidate.current_a
            self._state.voltage_tenths = candidate.voltage_tenths

    def arc_set(self, **recipe):
        """Update Byte1..11 recipe fields while preserving the golden frame."""
        with self._lock:
            if self._state.command & BIT_ARC:
                raise RuntimeError("arc_set rejected while ARC is ON")
            if self._state.command & (BIT_FORWARD | BIT_REVERSE):
                raise RuntimeError("arc_set rejected while inching is ON")
            candidate = replace(
                self._state, base_profile=PROFILE_WELDING, **recipe
            )
            build_request(candidate)
            self._state = candidate
        self.log_callback(
            f"Hi-COMM ARC SET · {candidate.current_a} A / "
            f"{candidate.voltage_tenths / 10.0:.1f} V · "
            f"{candidate.material} {candidate.diameter_mm:.1f} mm · "
            f"{candidate.mode} / {candidate.gas} · "
            f"synergic={candidate.synergic} · "
            f"pre={candidate.pre_gas_s:.2f}s "
            f"post={candidate.post_gas_s:.2f}s"
        )
        return candidate

    def comm_alive(self, max_age=0.30):
        status = self.latest_status()
        return bool(
            self.connected
            and status is not None
            and time.monotonic() - status["timestamp_monotonic"] <= max_age
        )

    def readiness(self):
        reasons = []
        status = self.latest_status()
        if not self.connected:
            reasons.append("TCP disconnected")
        if not self.comm_alive():
            reasons.append("recent cyclic RX unavailable")
        if status is not None:
            if status["db_unavailable"]:
                reasons.append("Hi-COMM DB welding unavailable")
            if status["torch_collision"]:
                reasons.append("torch collision")
            if status["welder_error"]:
                reasons.append(
                    f"welder error code {status['welder_error']}"
                )
        return not reasons, reasons

    def wait_for_status(self, predicate, timeout, description):
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._status_condition:
            while True:
                status = self._latest_status
                if status is not None and predicate(status):
                    return dict(status)
                if not self.connected:
                    raise RuntimeError(
                        f"disconnected while waiting for {description}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timeout waiting for {description}")
                self._status_condition.wait(timeout=min(0.05, remaining))

    def wait_comm_alive(self, timeout=1.0):
        return self.wait_for_status(
            lambda _status: self.comm_alive(), timeout, "cyclic RX"
        )

    def wait_arc_established(self, timeout=5.0, command_generation=None):
        """Wait for WCR + main-weld + nonzero wire feed."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._status_condition:
            if command_generation is None:
                command_generation = self._arc_command_generation
            while True:
                if (
                    command_generation != self._arc_command_generation
                    or not self._state.command & BIT_ARC
                ):
                    raise RuntimeError("ARC OFF during establishment")
                status = self._latest_status
                if status is not None and status["arc_established"]:
                    return dict(status)
                if not self._connected:
                    raise RuntimeError(
                        "disconnected while waiting for ARC ESTABLISHED "
                        "(WCR + feed)"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timeout waiting for ARC ESTABLISHED (WCR + feed)"
                    )
                self._status_condition.wait(timeout=min(0.05, remaining))

    def arc_on(
        self,
        *,
        wait_recognition=True,
        wait_welding=False,
        wait_established=False,
        timeout=5.0,
        force=False,
    ):
        """Set only ARC Byte0.Bit0, then optionally wait for each RX stage."""
        if self._outputs_inhibited.is_set():
            raise RuntimeError("Hi-COMM outputs are inhibited by STOP NOW")
        ready, reasons = self.readiness()
        if not ready and not force:
            raise RuntimeError("ARC ON precheck failed: " + "; ".join(reasons))
        with self._status_condition:
            if self._outputs_inhibited.is_set():
                raise RuntimeError("ARC ON aborted by STOP NOW")
            self._leave_inching_profile_locked()
            self._state.base_profile = PROFILE_WELDING
            self._state.command &= ~(
                BIT_FORWARD | BIT_REVERSE | BIT_GAS | BIT_STICK
            )
            self._state.command |= BIT_ARC
            self._arc_command_generation += 1
            command_generation = self._arc_command_generation
            self._status_condition.notify_all()
        self.log_callback("ARC ON command -> TX Byte0.Bit0 = 1")

        status = None
        if wait_recognition:
            status = self.wait_for_status(
                lambda value: value["arc_ack"],
                timeout,
                "ARC/Torch recognition",
            )
            self.log_callback(
                "ARC ON recognized by Hi-COMM (RX Byte0.Bit0=1)"
            )
        if wait_welding:
            status = self.wait_for_status(
                lambda value: (
                    value["output_state"] == OUTPUT_STATE_MAIN_WELD
                ),
                timeout,
                "main welding output state",
            )
            self.log_callback("MAIN-WELD STATE (RX Byte1=01)")
        if wait_established:
            status = self.wait_arc_established(timeout, command_generation)
            self.log_callback(
                "ARC ESTABLISHED -> WCR=1, main-weld, "
                f"feed={status['wire_feed_m_min']:.1f}m/min"
            )
        return status

    def arc_off(
        self,
        timeout=5.0,
        wait_idle=True,
        wait_sequence_clear=True,
    ):
        self.log_callback("Hi-COMM COMMAND · ARC OFF via arc_off()")
        with self._status_condition:
            self._arc_command_generation += 1
            self._state.command &= ~BIT_ARC
            self._status_condition.notify_all()
        status = self.wait_for_status(
            lambda value: not value["arc_ack"],
            timeout,
            "ARC recognition OFF",
        )
        if wait_idle:
            status = self.wait_for_status(
                lambda value: value["output_state"] == OUTPUT_STATE_IDLE,
                timeout,
                "welder output idle",
            )
        if wait_sequence_clear:
            status = self.wait_for_status(
                lambda value: (
                    not value["arc_ack"]
                    and not value["forward_ack"]
                    and not value["reverse_ack"]
                    and not value["gas_ack"]
                    and not value["stick_ack"]
                    and not value["wcr_detected"]
                    and value["wire_feed_m_min"] <= 0.0
                ),
                timeout,
                "ARC OFF post-sequence clear",
            )
        return status

    def setting_echo(self):
        requested = self.snapshot()
        status = self.latest_status()
        if status is None:
            return {"available": False, "reason": "no RX status"}
        checks = {
            "current": status["set_current_a"] == requested.current_a,
            "voltage": abs(
                status["set_voltage_v"]
                - requested.voltage_tenths / 10.0
            ) <= 0.05,
        }
        return {"available": True, "all_match": all(checks.values()),
                "checks": checks}

    def set_arc(self, enabled):
        """Compatibility helper; production welding should use arc_on/off."""
        self.log_callback(
            f"Hi-COMM COMMAND · ARC {'ON' if enabled else 'OFF'} "
            "via set_arc()"
        )
        self.set_command_bit(BIT_ARC, enabled)

    def set_command_bit(self, mask, enabled):
        """Set one documented command bit in the periodic TX frame."""
        if mask not in (BIT_ARC, BIT_FORWARD, BIT_REVERSE, BIT_GAS, BIT_STICK):
            raise ValueError(f"unsupported Hi-COMM command mask: 0x{mask:02X}")
        if enabled and not self.connected:
            raise RuntimeError("Hi-COMM is not connected")
        if enabled and self._outputs_inhibited.is_set():
            raise RuntimeError("Hi-COMM outputs are inhibited by STOP NOW")
        with self._status_condition:
            if enabled:
                if mask in (
                    BIT_FORWARD, BIT_REVERSE, BIT_GAS, BIT_STICK
                ) and self._state.command & BIT_ARC:
                    raise RuntimeError(
                        "manual welder test rejected while ARC is ON"
                    )
                # Forward and reverse wire feed must never be asserted together.
                if mask == BIT_FORWARD:
                    self._enter_inching_profile_locked()
                    self._state.command &= ~BIT_REVERSE
                elif mask == BIT_REVERSE:
                    self._enter_inching_profile_locked()
                    self._state.command &= ~BIT_FORWARD
                elif mask == BIT_ARC:
                    self._leave_inching_profile_locked()
                    self._state.base_profile = PROFILE_WELDING
                    self._state.command &= ~(
                        BIT_FORWARD | BIT_REVERSE | BIT_GAS | BIT_STICK
                    )
                    self._arc_command_generation += 1
                self._state.command |= mask
            else:
                self._state.command &= ~mask
                if mask == BIT_ARC:
                    self._arc_command_generation += 1
                if (
                    mask in (BIT_FORWARD, BIT_REVERSE)
                    and not self._state.command & (BIT_FORWARD | BIT_REVERSE)
                ):
                    self._leave_inching_profile_locked()
            self._status_condition.notify_all()

    def _enter_inching_profile_locked(self):
        if self._state_before_inching is None:
            self._state_before_inching = replace(self._state)
        command = self._state.command
        self._state = TxState(
            command=command,
            current_a=100,
            voltage_tenths=200,
            material="FE-SOLID",
            diameter_mm=1.0,
            mode="LSM",
            gas="CO2",
            synergic=False,
            correction=0.0,
            pre_gas_s=0.0,
            post_gas_s=0.0,
            base_profile=PROFILE_INCHING,
        )

    def _leave_inching_profile_locked(self):
        if self._state_before_inching is None:
            return
        command = self._state.command
        self._state = replace(self._state_before_inching, command=command)
        self._state_before_inching = None

    def clear_outputs(self):
        with self._status_condition:
            previous = self._state.command
            self._state.command = 0
            self._arc_command_generation += 1
            self._leave_inching_profile_locked()
            self._status_condition.notify_all()
        if previous:
            self.log_callback(
                f"Hi-COMM COMMAND · ALL OFF via clear_outputs() · "
                f"previous=0x{previous:02X}"
            )

    def inhibit_outputs(self):
        """Latch all physical commands OFF until explicitly re-armed."""
        self._outputs_inhibited.set()
        self.clear_outputs()

    def allow_outputs(self):
        self._outputs_inhibited.clear()

    def wait_for_arc_ack(self, expected, timeout):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            status = self.latest_status()
            if status is not None and bool(status["arc_ack"]) == bool(expected):
                return True
            if not self.connected:
                return False
            time.sleep(0.02)
        return False

    @staticmethod
    def _send_full(sock, payload):
        view = memoryview(payload)
        deadline = time.monotonic() + 0.010
        while view:
            try:
                sent = sock.send(view)
                if sent <= 0:
                    raise ConnectionError("Hi-COMM socket send returned zero")
                view = view[sent:]
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Hi-COMM TX timeout")
                select.select([], [sock], [], min(0.002, remaining))

    def _drain_rx(self, sock, rx_buffer, deadline):
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [sock], [], [], min(0.002, max(0.0, deadline - time.monotonic()))
            )
            if not ready:
                continue
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("Hi-COMM closed the TCP connection")
            rx_buffer.extend(chunk)
            while len(rx_buffer) >= RX_SIZE:
                frame = bytes(rx_buffer[:RX_SIZE])
                del rx_buffer[:RX_SIZE]
                status = decode_response(frame)
                with self._status_condition:
                    self._latest_status = status
                    self._status_condition.notify_all()
                self._dispatch_status(status)

    def _dispatch_status(self, status):
        """Coalesce telemetry without running user code in the I/O thread."""
        with self._callback_condition:
            self._pending_callback_status = dict(status)
            self._callback_condition.notify()

    def _dispatch_callback(self, callback, *args):
        with self._callback_condition:
            self._callback_events.append((callback, args))
            self._callback_condition.notify()

    def _run_callbacks(self):
        """Deliver GUI/ROS callbacks independently from the 40 ms I/O loop."""
        while True:
            with self._callback_condition:
                self._callback_condition.wait_for(
                    lambda: (
                        self._callback_stop.is_set()
                        or self._callback_events
                        or self._pending_callback_status is not None
                    ),
                    timeout=0.20,
                )
                if (
                    self._callback_stop.is_set()
                    and not self._callback_events
                    and self._pending_callback_status is None
                ):
                    return
                events = tuple(self._callback_events)
                self._callback_events.clear()
                status = self._pending_callback_status
                self._pending_callback_status = None
            for callback, args in events:
                try:
                    callback(*args)
                except Exception:
                    # A GUI/logger failure must never stop cyclic Hi-COMM I/O.
                    pass
            if status is not None:
                try:
                    self.status_callback(status)
                except Exception:
                    pass

    def _set_connected(self, connected, detail):
        with self._status_condition:
            self._connected = bool(connected)
            if not connected:
                self._latest_status = None
            self._status_condition.notify_all()
        self._dispatch_callback(
            self.connection_callback, bool(connected), detail
        )

    def _run(self):
        attempt = 0
        while not self._stop.is_set():
            attempt += 1
            try:
                self._run_connected_session()
            except Exception as error:
                self.clear_outputs()
                detail = (
                    f"retrying in {CONNECT_RETRY_SECONDS:.1f} s · "
                    f"{type(error).__name__}: {error}"
                )
                self._set_connected(False, detail)
                if attempt == 1 or attempt % 25 == 0:
                    self._dispatch_callback(
                        self.log_callback,
                        f"Hi-COMM connect attempt {attempt} failed · {detail}"
                    )
            if not self._stop.wait(CONNECT_RETRY_SECONDS):
                continue
            break
        self.clear_outputs()
        self._socket = None
        self._set_connected(False, "stopped")

    def _run_connected_session(self):
        rx_buffer = bytearray()
        cycles = 0
        self._last_tx_monotonic = None
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            self._socket = sock
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.settimeout(CONNECT_RETRY_SECONDS)
            sock.bind((self.source_ip, 0))
            sock.connect((self.welder_ip, self.port))
            sock.setblocking(False)
            self._set_connected(
                True,
                f"{sock.getsockname()} → {sock.getpeername()} · TX55/RX71",
            )
            next_tick = time.monotonic()
            while not self._stop.is_set():
                state = self.snapshot()
                tx_started = time.monotonic()
                previous_tx = self._last_tx_monotonic
                self._last_tx_monotonic = tx_started
                if previous_tx is not None:
                    interval = tx_started - previous_tx
                    if (
                        interval > 0.060
                        and tx_started - self._last_cadence_warning_monotonic
                        >= 1.0
                    ):
                        self._last_cadence_warning_monotonic = tx_started
                        self._dispatch_callback(
                            self.log_callback,
                            "Hi-COMM CADENCE WARNING · "
                            f"TX interval={interval * 1000.0:.1f} ms "
                            f"(target={PERIOD_SECONDS * 1000.0:.1f} ms)"
                        )
                self._send_full(sock, build_request(state))
                cycles += 1
                # if cycles == 1 or cycles % 25 == 0:
                #     self.log_callback(
                #         f"Hi-COMM cycle={cycles} TX0=0x{state.command:02X} "
                #         f"I={state.current_a}A V={state.voltage_tenths / 10:.1f}V"
                #     )
                next_tick += PERIOD_SECONDS
                self._drain_rx(sock, rx_buffer, next_tick)
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()

            # Explicit safe OFF frames before closing the connection.
            off_state = self.snapshot()
            off_state.command = 0
            for _ in range(5):
                try:
                    self._send_full(sock, build_request(off_state))
                except Exception:
                    break
                time.sleep(PERIOD_SECONDS)
