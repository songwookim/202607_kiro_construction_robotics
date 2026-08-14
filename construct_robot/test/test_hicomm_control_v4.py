#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hi-COMM / Rainbow welding controller v4.

이 파일의 목적
=============
PC가 Rainbow Robotics 제어기의 Hi-COMM Ethernet TCP client 역할을 대신하면서
실제 용접에서 필요한 ``arc_set() -> arc_on() -> arc_off()`` 흐름을 하나의
``HiCommWelder`` 클래스로 제공한다.

프로토콜 근거
=============
1) 공식 Rainbow <-> Hi-COMM 프로토콜
   - Hi-COMM: TCP Server
   - Rainbow: TCP Client
   - Rainbow 송신 Byte0
       bit7 : Robot Error status
       bit6 : Torch Collision status
       bit5 : Reserved
       bit4 : Stick Check
       bit3 : Gas Check
       bit2 : Reverse Inching
       bit1 : Forward Inching
       bit0 : Torch / ARC ON
   - Byte1    : wire material / diameter / welding mode
   - Byte2    : individual/synergic + shielding gas
   - Byte3-4  : main welding current [A], little endian
   - Byte5-6  : main welding voltage [0.1 V], little endian
   - Byte7    : synergic voltage correction, raw 0..100 => -5.0..+5.0
   - Byte8-9  : pre-gas, raw 0..1000 => 0..10 s (10 ms/unit)
   - Byte10-11: post-gas, raw 0..1000 => 0..10 s (10 ms/unit)
   - Byte12-52: Hot Start / short / neck / pulse / burn-back / anti-stick 등

2) 실제 Rainbow <-> Hi-COMM 캡처(hicomm_inching_capture.pcapng)
   이 장비/펌웨어 조합에서는 공식 문서의 53-byte TX보다 실제 wire에서
   55-byte TX가 관찰되었다. Byte53-54는 0x00 0x00 이었다.

   관찰값:
     Rainbow TX : 55 bytes
     TX period  : 약 40.05 ms
     Hi-COMM RX : 71 bytes
     idle       : Byte0 = 0x00
     forward    : Byte0 = 0x02
     reverse    : Byte0 = 0x04

   따라서 v4는 문서보다 '실제 캡처된 wire format'을 우선하여
   TX55 / ~40ms / RX71을 사용한다.

핵심 설계
=========
* 소켓은 용접 중 계속 유지한다.
* 별도의 "ARC SET 패킷", "ARC ON 패킷"을 보내는 구조가 아니다.
* 하나의 55-byte cyclic frame을 계속 송신하고 상태만 갱신한다.

  arc_set(recipe)
      -> Byte1..11의 문서화된 용접 조건을 변경
      -> Byte12..52는 검증된 golden frame 값을 그대로 유지

  arc_on()
      -> Byte0.Bit0 = 1
      -> 이후 모든 40 ms cyclic frame에서 계속 1 유지
      -> RX Byte0.Bit0(토치 명령 인식) / RX Byte1(용접 출력 상태) 확인 가능

  arc_off()
      -> Byte0.Bit0 = 0
      -> TCP를 끊지 않고 OFF frame을 계속 송신
      -> 종료/대기 상태를 확인한 뒤 다음 동작 또는 disconnect

실제 용접 코드 예시
===================

    recipe = WeldingRecipe(
        current_a=150,
        voltage_v=20.0,
        material="FE-Solid",
        diameter_mm=1.0,
        mode="LSM",
        gas="CO2",
        synergic=False,
        correction=0.0,
        pre_gas_s=0.5,
        post_gas_s=1.0,
    )

    with HiCommWelder("192.168.1.19", "192.168.1.10", 60000) as welder:
        welder.arc_set(recipe)

        # 현재 펌웨어의 RX71에서 문서상 echo 위치가 실제 장비와 일치하는지
        # 먼저 검증한 뒤 strict 확인을 활성화하는 것을 권장한다.
        print(welder.setting_echo())

        welder.arc_on(wait_recognition=True, wait_welding=True, timeout=3.0)

        # robot.execute_weld_path()

        welder.arc_off(wait_idle=True, timeout=5.0)

ROS2에서의 권장 사용
===================
이 클래스에는 PyQt 의존성이 없다. 따라서 ROS2 Action Server / node 쪽에서
직접 import하여 다음 세 API 중심으로 사용하면 된다.

    welder.arc_set(...)
    welder.arc_on(...)
    welder.arc_off(...)

GUI는 이 파일을 직접 실행했을 때만 선택적으로 사용된다.
"""

from __future__ import annotations

import argparse
import select
import socket
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Optional


# =============================================================================
# 1. Network / captured wire profile
# =============================================================================
DEFAULT_SOURCE_IP = "192.168.1.19"
DEFAULT_HICOMM_IP = "192.168.1.10"
DEFAULT_PORT = 60000

# 실제 Rainbow 캡처에서 관찰된 값.
PERIOD_SECONDS = 0.040
TX_SIZE = 55
RX_SIZE = 71

# 공식 문서 범위. 현재 장비 wire format은 각각 +2, +7 bytes가 더 관찰됨.
DOCUMENTED_TX_SIZE = 53
DOCUMENTED_RX_SIZE = 64

# 실제 Rainbow IDLE frame. Byte12..52의 고급 welding parameters는
# 검증되지 않은 임의값으로 다시 생성하지 않고 이 golden frame을 유지한다.
CAPTURED_IDLE_REQUEST = bytes.fromhex(
    "00 08 00 64 00 C8 00 32 "
    "00 00 00 00 32 32 64 00 "
    "0F 64 00 32 32 32 32 32 "
    "32 32 32 32 32 32 32 32 "
    "32 32 32 32 32 32 32 32 "
    "32 32 32 32 32 32 32 32 "
    "32 14 0A 00 64 00 00"
)
assert len(CAPTURED_IDLE_REQUEST) == TX_SIZE
assert CAPTURED_IDLE_REQUEST[53:55] == b"\x00\x00"


# =============================================================================
# 2. Rainbow TX Byte0 bit definitions
# =============================================================================
BIT_ARC = 0x01
BIT_FORWARD = 0x02
BIT_REVERSE = 0x04
BIT_GAS = 0x08
BIT_STICK = 0x10
BIT_RESERVED5 = 0x20
BIT_TORCH_COLLISION = 0x40
BIT_ROBOT_ERROR = 0x80

COMMAND_BITS = BIT_ARC | BIT_FORWARD | BIT_REVERSE | BIT_GAS | BIT_STICK
ROBOT_STATUS_BITS = BIT_TORCH_COLLISION | BIT_ROBOT_ERROR


# =============================================================================
# 3. Documented enums / recipe encoding
# =============================================================================
MATERIAL_CODES = {
    "FE-SOLID": 0,
    "FE-CORED": 1,
    "STS-SOLID": 2,
    "STS-CORED": 3,
    "AL-SOFT": 4,
    "AL-HARD": 5,
    "CUSI": 6,
    "CUMG": 7,
}

DIAMETER_CODES = {
    0.8: 0,
    0.9: 1,
    1.0: 2,
    1.2: 3,
    1.4: 4,
    1.6: 5,
}

MODE_CODES = {
    "LSM": 0,
    "DCM": 1,
    "DPM": 2,
    "PM": 3,
}

GAS_CODES = {
    "CO2": 0,
    "CO2 100%": 0,
    "AR80+CO2 20%": 1,
    "AR80_CO2_20": 1,
    "AR98+O2 2%": 2,
    "AR98_O2_2": 2,
    "AR 100%": 3,
    "AR100": 3,
}

OUTPUT_STATE_IDLE = 0
OUTPUT_STATE_MAIN_WELD = 1
OUTPUT_STATE_CRATER = 2
OUTPUT_STATE_END = 3

OUTPUT_STATE_NAMES = {
    OUTPUT_STATE_IDLE: "대기",
    OUTPUT_STATE_MAIN_WELD: "본 용접",
    OUTPUT_STATE_CRATER: "크레이터",
    OUTPUT_STATE_END: "용접 종료",
}


def _norm_key(value: str) -> str:
    return value.strip().upper()


def _encode_enum(value: str | int, mapping: dict[str, int], name: str) -> int:
    if isinstance(value, int):
        if value in mapping.values():
            return value
        raise ValueError(f"{name} code out of range: {value}")
    key = _norm_key(value)
    if key not in mapping:
        raise ValueError(f"unknown {name}: {value!r}; choices={list(mapping)}")
    return mapping[key]


def _encode_diameter(value: float | int) -> int:
    f = round(float(value), 1)
    if f not in DIAMETER_CODES:
        raise ValueError(f"unsupported diameter: {value}; choices={list(DIAMETER_CODES)}")
    return DIAMETER_CODES[f]


def _u16le(data: bytes, offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8)


def _put_u16le(frame: bytearray, offset: int, value: int) -> None:
    frame[offset : offset + 2] = int(value).to_bytes(2, "little", signed=False)


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


@dataclass(frozen=True)
class WeldingRecipe:
    """ARC SET에 사용되는 최소 실용 welding recipe.

    고급 파형/단락/Neck/Pulse/Burn-Back/Anti-Stick 설정(Byte12..52)은
    이 dataclass에서 임의로 다시 만들지 않는다. 현재는 실제 Rainbow에서 캡처한
    golden frame의 값을 유지하는 것이 목적이다.
    """

    current_a: int = 100
    voltage_v: float = 20.0
    material: str | int = "FE-Solid"
    diameter_mm: float = 1.0
    mode: str | int = "LSM"
    gas: str | int = "CO2"
    synergic: bool = False
    correction: float = 0.0
    pre_gas_s: float = 0.0
    post_gas_s: float = 0.0

    def validate(self) -> None:
        if not 30 <= int(self.current_a) <= 400:
            raise ValueError("main welding current must be 30..400 A")
        if not 10.0 <= float(self.voltage_v) <= 40.0:
            raise ValueError("main welding voltage must be 10.0..40.0 V")
        if not -5.0 <= float(self.correction) <= 5.0:
            raise ValueError("synergic correction must be -5.0..+5.0")
        if not 0.0 <= float(self.pre_gas_s) <= 10.0:
            raise ValueError("pre-gas must be 0..10 s")
        if not 0.0 <= float(self.post_gas_s) <= 10.0:
            raise ValueError("post-gas must be 0..10 s")
        _encode_enum(self.material, MATERIAL_CODES, "material")
        _encode_diameter(self.diameter_mm)
        _encode_enum(self.mode, MODE_CODES, "mode")
        _encode_enum(self.gas, GAS_CODES, "gas")

    @property
    def material_code(self) -> int:
        return _encode_enum(self.material, MATERIAL_CODES, "material")

    @property
    def diameter_code(self) -> int:
        return _encode_diameter(self.diameter_mm)

    @property
    def mode_code(self) -> int:
        return _encode_enum(self.mode, MODE_CODES, "mode")

    @property
    def gas_code(self) -> int:
        return _encode_enum(self.gas, GAS_CODES, "gas")

    @property
    def voltage_tenths(self) -> int:
        return int(round(float(self.voltage_v) * 10.0))

    @property
    def correction_raw(self) -> int:
        # raw 0..100 maps linearly to -5.0..+5.0 => raw50 = 0.0
        return int(round((float(self.correction) + 5.0) * 10.0))

    @property
    def pre_gas_raw(self) -> int:
        # 10 ms/unit => 100 raw = 1.00 s
        return int(round(float(self.pre_gas_s) * 100.0))

    @property
    def post_gas_raw(self) -> int:
        return int(round(float(self.post_gas_s) * 100.0))


@dataclass
class TxState:
    """Cyclic TX 상태.

    command에는 Byte0의 low command bits만 저장한다.
    robot_error / torch_collision은 Byte0의 robot-side status bits이다.
    """

    recipe: WeldingRecipe = WeldingRecipe()
    command: int = 0
    robot_error: bool = False
    torch_collision: bool = False


def build_request(state: TxState) -> bytes:
    """현재 state로 55-byte Rainbow wire frame을 만든다.

    변경하는 필드
    ------------
    Byte0    : robot status + ARC/Gas/Inching/Stick command
    Byte1    : material / diameter / weld mode
    Byte2    : individual/synergic + gas type
    Byte3-4  : current
    Byte5-6  : voltage
    Byte7    : correction
    Byte8-9  : pre-gas
    Byte10-11: post-gas

    유지하는 필드
    ------------
    Byte12-52: 실제 Rainbow 캡처값 유지
    Byte53-54: 실제 wire에서 관찰된 00 00 유지
    """
    r = state.recipe
    r.validate()

    frame = bytearray(CAPTURED_IDLE_REQUEST)

    byte0 = state.command & COMMAND_BITS
    if state.torch_collision:
        byte0 |= BIT_TORCH_COLLISION
    if state.robot_error:
        byte0 |= BIT_ROBOT_ERROR
    # Byte0.Bit5 reserved: intentionally always zero.
    frame[0] = byte0 & ~BIT_RESERVED5

    frame[1] = (
        ((r.material_code & 0x07) << 5)
        | ((r.diameter_code & 0x07) << 2)
        | (r.mode_code & 0x03)
    )
    frame[2] = (0x80 if r.synergic else 0x00) | (r.gas_code & 0x03)

    _put_u16le(frame, 3, int(r.current_a))
    _put_u16le(frame, 5, r.voltage_tenths)
    frame[7] = r.correction_raw
    _put_u16le(frame, 8, r.pre_gas_raw)
    _put_u16le(frame, 10, r.post_gas_raw)

    # Captured undocumented trailer.
    frame[53] = 0x00
    frame[54] = 0x00

    if len(frame) != TX_SIZE:
        raise AssertionError(f"TX frame length changed unexpectedly: {len(frame)}")
    return bytes(frame)


@dataclass(frozen=True)
class WelderStatus:
    """Hi-COMM RX 상태.

    Byte0..63은 공식 문서 위치를 기준으로 decode한다.
    Byte64..70은 현재 firmware에서 관찰된 7-byte extension이며 의미를 추정하지 않는다.
    """

    timestamp_monotonic: float
    raw: bytes

    db_unavailable: bool
    torch_collision: bool
    wcr_detected: bool
    stick_ack: bool
    gas_ack: bool
    reverse_ack: bool
    forward_ack: bool
    arc_ack: bool

    output_state: int
    feedback_current_a: int
    feedback_voltage_v: float
    wire_feed_m_min: float

    material_code: int
    diameter_code: int
    mode_code: int
    synergic: bool
    gas_code: int

    welder_error: int
    set_current_a: int
    set_voltage_v: float
    correction_raw: int
    pre_gas_raw: int
    post_gas_raw: int

    extra7: bytes

    @property
    def db_available(self) -> bool:
        return not self.db_unavailable

    @property
    def output_state_name(self) -> str:
        return OUTPUT_STATE_NAMES.get(self.output_state, f"unknown({self.output_state})")

    @property
    def pre_gas_s(self) -> float:
        return self.pre_gas_raw / 100.0

    @property
    def post_gas_s(self) -> float:
        return self.post_gas_raw / 100.0


def decode_response(frame: bytes) -> WelderStatus:
    """Decode captured 71-byte Hi-COMM response.

    주의:
    현재 장비는 RX71이지만 공식 문서는 Byte0..63(64B)만 정의한다.
    따라서 아래 앞 64B field mapping은 '문서 정의'이고, 실제 firmware에서
    각 echo field가 원하는 값과 일치하는지는 현장 검증을 계속해야 한다.
    """
    if len(frame) != RX_SIZE:
        raise ValueError(f"RX length {len(frame)} != expected captured {RX_SIZE}")

    p = frame[:DOCUMENTED_RX_SIZE]
    b0 = p[0]
    b7 = p[7]
    b8 = p[8]

    return WelderStatus(
        timestamp_monotonic=time.monotonic(),
        raw=frame,
        db_unavailable=bool(b0 & 0x80),
        torch_collision=bool(b0 & 0x40),
        wcr_detected=bool(b0 & 0x20),
        stick_ack=bool(b0 & 0x10),
        gas_ack=bool(b0 & 0x08),
        reverse_ack=bool(b0 & 0x04),
        forward_ack=bool(b0 & 0x02),
        arc_ack=bool(b0 & 0x01),
        output_state=p[1] & 0x03,
        feedback_current_a=_u16le(p, 2),
        feedback_voltage_v=_u16le(p, 4) / 10.0,
        wire_feed_m_min=p[6] / 10.0,
        material_code=(b7 >> 5) & 0x07,
        diameter_code=(b7 >> 2) & 0x07,
        mode_code=b7 & 0x03,
        synergic=bool(b8 & 0x80),
        gas_code=b8 & 0x03,
        welder_error=p[9],
        set_current_a=_u16le(p, 10),
        set_voltage_v=_u16le(p, 12) / 10.0,
        correction_raw=p[14],
        pre_gas_raw=_u16le(p, 15),
        post_gas_raw=_u16le(p, 17),
        extra7=frame[64:71],
    )


# =============================================================================
# 4. Production-oriented persistent cyclic client
# =============================================================================
class HiCommError(RuntimeError):
    pass


class HiCommTimeout(TimeoutError):
    pass


class HiCommWelder:
    """Persistent Rainbow-compatible Hi-COMM TCP client.

    이 클래스가 처리하는 것
    -----------------------
    * TCP 연결 유지
    * 실제 캡처 기준 TX55 / ~40 ms cyclic transmission
    * TCP stream RX buffering / RX71 framing
    * 최신 상태 저장
    * arc_set / arc_on / arc_off
    * inching / gas / stick helper
    * disconnect 전에 command OFF frame 전송

    외부(robot/ROS) 코드는 cyclic socket 세부사항을 알 필요가 없다.
    """

    def __init__(
        self,
        source_ip: str = DEFAULT_SOURCE_IP,
        hicomm_ip: str = DEFAULT_HICOMM_IP,
        port: int = DEFAULT_PORT,
        *,
        period_s: float = PERIOD_SECONDS,
        logger: Optional[Callable[[str], None]] = print,
    ) -> None:
        self.source_ip = source_ip
        self.hicomm_ip = hicomm_ip
        self.port = int(port)
        self.period_s = float(period_s)
        self._logger = logger

        self._state_lock = threading.RLock()
        self._status_cv = threading.Condition(threading.RLock())
        self._state = TxState()
        self._latest_status: Optional[WelderStatus] = None
        self._last_error: Optional[BaseException] = None

        self._sock: Optional[socket.socket] = None
        self._io_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._rx_buffer = bytearray()

        self._cycle = 0
        self._rx_frames = 0
        self._last_tx_time: Optional[float] = None
        self._last_rx_time: Optional[float] = None
        self._connected = False

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "HiCommWelder":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # logging / state snapshots
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger(f"[{_stamp()}] {message}")

    @property
    def connected(self) -> bool:
        return self._connected and self._io_thread is not None and self._io_thread.is_alive()

    def tx_state(self) -> TxState:
        with self._state_lock:
            return TxState(
                recipe=replace(self._state.recipe),
                command=self._state.command,
                robot_error=self._state.robot_error,
                torch_collision=self._state.torch_collision,
            )

    def latest_status(self) -> Optional[WelderStatus]:
        with self._status_cv:
            return self._latest_status

    def comm_alive(self, max_age_s: float = 0.30) -> bool:
        if not self.connected:
            return False
        status = self.latest_status()
        if status is None:
            return False
        return (time.monotonic() - status.timestamp_monotonic) <= max_age_s

    # ------------------------------------------------------------------
    # connection lifecycle
    # ------------------------------------------------------------------
    def connect(self, timeout: float = 3.0) -> None:
        if self.connected:
            return

        self._last_error = None
        self._stop_evt.clear()
        self._rx_buffer.clear()
        self._cycle = 0
        self._rx_frames = 0
        self._latest_status = None
        self._last_tx_time = None
        self._last_rx_time = None

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.settimeout(timeout)
        sock.bind((self.source_ip, 0))
        self._log(f"TCP connect {self.source_ip}:ephemeral -> {self.hicomm_ip}:{self.port}")
        sock.connect((self.hicomm_ip, self.port))
        sock.setblocking(False)
        self._sock = sock
        self._connected = True

        # New session always starts with command outputs OFF.
        with self._state_lock:
            self._state.command = 0

        self._io_thread = threading.Thread(
            target=self._io_loop,
            name="HiCommWelderIO",
            daemon=True,
        )
        self._io_thread.start()
        self._log(
            f"CONNECTED local={sock.getsockname()} peer={sock.getpeername()} "
            f"TX{TX_SIZE}/{self.period_s*1000:.0f}ms/RX{RX_SIZE}"
        )

    def disconnect(self, off_cycles: int = 5) -> None:
        # 먼저 command를 OFF로 만들고 cyclic thread가 몇 번 송신할 시간을 준다.
        self.all_outputs_off()

        if self.connected and off_cycles > 0:
            deadline = time.monotonic() + off_cycles * self.period_s + 0.1
            start_cycle = self._cycle
            while self.connected and self._cycle - start_cycle < off_cycles:
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(0.01, self.period_s / 2))

        self._stop_evt.set()
        thread = self._io_thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

        sock = self._sock
        self._sock = None
        self._connected = False
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self._log("DISCONNECTED")

    # ------------------------------------------------------------------
    # cyclic IO implementation
    # ------------------------------------------------------------------
    @staticmethod
    def _send_full_nonblocking(sock: socket.socket, payload: bytes) -> None:
        view = memoryview(payload)
        deadline = time.monotonic() + 0.010
        while view:
            try:
                sent = sock.send(view)
                if sent <= 0:
                    raise ConnectionError("socket send returned 0")
                view = view[sent:]
            except BlockingIOError:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    raise TimeoutError("socket TX not writable for 10 ms")
                _, writable, _ = select.select([], [sock], [], min(0.002, remain))
                if not writable and time.monotonic() >= deadline:
                    raise TimeoutError("socket TX timeout")

    def _publish_status(self, status: WelderStatus) -> None:
        with self._status_cv:
            self._latest_status = status
            self._last_rx_time = status.timestamp_monotonic
            self._rx_frames += 1
            self._status_cv.notify_all()

    def _drain_rx_until(self, sock: socket.socket, deadline: float) -> None:
        """RX를 읽되 다음 TX deadline을 막지 않는다.

        TCP에는 message boundary가 없으므로 recv(71)==71을 가정하지 않고
        byte buffer에 누적한 뒤 71-byte 단위로 잘라 decode한다.
        """
        while time.monotonic() < deadline and not self._stop_evt.is_set():
            remain = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([sock], [], [], min(0.002, remain))
            if not ready:
                continue

            try:
                chunk = sock.recv(4096)
            except BlockingIOError:
                continue
            if not chunk:
                raise ConnectionError("Hi-COMM closed TCP connection")

            self._rx_buffer.extend(chunk)
            while len(self._rx_buffer) >= RX_SIZE:
                frame = bytes(self._rx_buffer[:RX_SIZE])
                del self._rx_buffer[:RX_SIZE]
                self._publish_status(decode_response(frame))

    def _io_loop(self) -> None:
        sock = self._sock
        if sock is None:
            return

        next_tick = time.monotonic()
        try:
            while not self._stop_evt.is_set():
                with self._state_lock:
                    state = TxState(
                        recipe=replace(self._state.recipe),
                        command=self._state.command,
                        robot_error=self._state.robot_error,
                        torch_collision=self._state.torch_collision,
                    )
                frame = build_request(state)

                now = time.monotonic()
                self._last_tx_time = now
                self._send_full_nonblocking(sock, frame)
                self._cycle += 1

                if self._cycle == 1 or self._cycle % 25 == 0:
                    age = None if self._last_rx_time is None else now - self._last_rx_time
                    self._log(
                        f"cycle={self._cycle} TX0=0x{frame[0]:02X} "
                        f"I={state.recipe.current_a}A V={state.recipe.voltage_v:.1f}V "
                        f"RX_age={'none' if age is None else f'{age*1000:.1f}ms'}"
                    )

                next_tick += self.period_s
                self._drain_rx_until(sock, next_tick)

                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    self._log(f"WARN IO overrun {-delay*1000:.2f} ms; cadence resync")
                    next_tick = time.monotonic()

        except BaseException as exc:
            self._last_error = exc
            self._log(f"IO FAULT {type(exc).__name__}: {exc}")
        finally:
            self._connected = False
            with self._status_cv:
                self._status_cv.notify_all()

    # ------------------------------------------------------------------
    # wait helpers
    # ------------------------------------------------------------------
    def wait_for_status(
        self,
        predicate: Callable[[WelderStatus], bool],
        timeout: float,
        description: str,
    ) -> WelderStatus:
        deadline = time.monotonic() + timeout
        with self._status_cv:
            while True:
                status = self._latest_status
                if status is not None and predicate(status):
                    return status
                if self._last_error is not None:
                    raise HiCommError(f"communication fault while waiting for {description}: {self._last_error}")
                if not self.connected:
                    raise HiCommError(f"disconnected while waiting for {description}")
                remain = deadline - time.monotonic()
                if remain <= 0:
                    raise HiCommTimeout(f"timeout waiting for {description}")
                self._status_cv.wait(timeout=min(0.05, remain))

    def wait_comm_alive(self, timeout: float = 1.0) -> WelderStatus:
        return self.wait_for_status(lambda _: self.comm_alive(), timeout, "cyclic RX")

    # ------------------------------------------------------------------
    # ARC SET
    # ------------------------------------------------------------------
    def arc_set(self, recipe: Optional[WeldingRecipe] = None, **kwargs) -> WeldingRecipe:
        """Update documented welding parameters in the cyclic TX frame.

        사용법 1:
            welder.arc_set(WeldingRecipe(current_a=150, voltage_v=20.0, ...))

        사용법 2:
            welder.arc_set(current_a=150, voltage_v=20.0, pre_gas_s=0.5)

        ARC가 이미 ON인 상태에서 recipe를 바꾸는 것은 여기서는 금지한다.
        용접 중 동적 parameter tuning이 필요하면 별도 검증 후 정책을 추가하는 편이 낫다.
        """
        with self._state_lock:
            if self._state.command & BIT_ARC:
                raise HiCommError("arc_set() rejected while ARC command is ON")

            if recipe is not None and kwargs:
                raise ValueError("pass either recipe or keyword fields, not both")
            if recipe is None:
                recipe = replace(self._state.recipe, **kwargs)
            recipe.validate()
            self._state.recipe = recipe

        self._log(
            "ARC SET -> "
            f"{recipe.current_a}A / {recipe.voltage_v:.1f}V / "
            f"material={recipe.material} / dia={recipe.diameter_mm:.1f}mm / "
            f"mode={recipe.mode} / gas={recipe.gas} / "
            f"synergic={recipe.synergic} / corr={recipe.correction:+.1f} / "
            f"pre={recipe.pre_gas_s:.2f}s / post={recipe.post_gas_s:.2f}s"
        )
        return recipe

    def setting_echo(self, recipe: Optional[WeldingRecipe] = None) -> dict[str, object]:
        """Compare latest documented RX setting echo with the requested recipe.

        현재 firmware는 문서의 RX64보다 실제 wire에서 RX71이므로, 이 결과는
        현장 검증용이다. 처음부터 ARC ON의 절대 interlock으로 사용하지 않는다.
        """
        if recipe is None:
            recipe = self.tx_state().recipe
        status = self.latest_status()
        if status is None:
            return {"available": False, "reason": "no RX status yet"}

        expected_b1 = (
            (recipe.material_code << 5)
            | (recipe.diameter_code << 2)
            | recipe.mode_code
        )
        expected_b2 = (0x80 if recipe.synergic else 0) | recipe.gas_code
        actual_b1 = (status.material_code << 5) | (status.diameter_code << 2) | status.mode_code
        actual_b2 = (0x80 if status.synergic else 0) | status.gas_code

        checks = {
            "byte1_recipe": actual_b1 == expected_b1,
            "byte2_recipe": actual_b2 == expected_b2,
            "current": status.set_current_a == recipe.current_a,
            "voltage": abs(status.set_voltage_v - recipe.voltage_v) <= 0.05,
            "correction": status.correction_raw == recipe.correction_raw,
            "pre_gas": status.pre_gas_raw == recipe.pre_gas_raw,
            "post_gas": status.post_gas_raw == recipe.post_gas_raw,
        }
        return {
            "available": True,
            "all_match": all(checks.values()),
            "checks": checks,
            "requested": recipe,
            "rx_current_a": status.set_current_a,
            "rx_voltage_v": status.set_voltage_v,
            "rx_correction_raw": status.correction_raw,
            "rx_pre_gas_s": status.pre_gas_s,
            "rx_post_gas_s": status.post_gas_s,
        }

    def wait_setting_applied(
        self,
        timeout: float = 1.0,
        *,
        current_voltage_only: bool = True,
    ) -> WelderStatus:
        """Wait until documented RX setpoint echo matches TX request.

        ``current_voltage_only=True``가 기본인 이유:
        현재 wire RX71과 문서 RX64 사이 firmware 차이가 있으므로 먼저 가장 중요한
        current/voltage echo부터 현장 검증하는 편이 안전하다.
        """
        recipe = self.tx_state().recipe

        if current_voltage_only:
            pred = lambda s: (
                s.set_current_a == recipe.current_a
                and abs(s.set_voltage_v - recipe.voltage_v) <= 0.05
            )
        else:
            def pred(_: WelderStatus) -> bool:
                e = self.setting_echo(recipe)
                return bool(e.get("all_match", False))

        return self.wait_for_status(pred, timeout, "welding setting echo")

    # ------------------------------------------------------------------
    # readiness / ARC ON / ARC OFF
    # ------------------------------------------------------------------
    def readiness(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        state = self.tx_state()
        status = self.latest_status()

        if not self.connected:
            reasons.append("TCP/cyclic thread disconnected")
        if not self.comm_alive():
            reasons.append("recent RX not available")
        if state.robot_error:
            reasons.append("robot_error TX status is ON")
        if state.torch_collision:
            reasons.append("robot torch_collision TX status is ON")

        if status is None:
            reasons.append("no Hi-COMM RX status")
        else:
            if status.db_unavailable:
                reasons.append("Hi-COMM DB welding unavailable")
            if status.welder_error != 0:
                reasons.append(f"welder error code={status.welder_error}")
            if status.torch_collision:
                reasons.append("Hi-COMM reports torch collision")

        return (not reasons, reasons)

    def arc_on(
        self,
        *,
        wait_recognition: bool = True,
        wait_welding: bool = False,
        timeout: float = 3.0,
        force: bool = False,
    ) -> Optional[WelderStatus]:
        """Set Byte0.Bit0 and keep it ON in every cyclic frame.

        기본 precheck:
          - cyclic RX alive
          - DB welding available
          - welder error == 0
          - robot/torch collision status clear

        ``force=True``는 bring-up/diagnostics용으로 precheck를 우회한다.

        wait_recognition=True:
          RX Byte0.Bit0 == 1까지 기다림.

        wait_welding=True:
          그 이후 RX Byte1 == 1(본 용접)까지 기다림.
          실제 arc establishment 조건/가스 예출 등에 따라 timeout을 조정할 것.
        """
        ready, reasons = self.readiness()
        if not ready and not force:
            raise HiCommError("ARC ON precheck failed: " + "; ".join(reasons))

        # ARC와 수동 inch/gas/stick command가 동시에 나가지 않게 정리.
        with self._state_lock:
            self._state.command &= ~(BIT_FORWARD | BIT_REVERSE | BIT_GAS | BIT_STICK)
            self._state.command |= BIT_ARC

        self._log("ARC ON command -> TX Byte0.Bit0 = 1")

        status: Optional[WelderStatus] = None
        if wait_recognition:
            status = self.wait_for_status(lambda s: s.arc_ack, timeout, "ARC/Torch recognition")
            self._log("ARC ON recognized by Hi-COMM (RX Byte0.Bit0=1)")

        if wait_welding:
            status = self.wait_for_status(
                lambda s: s.output_state == OUTPUT_STATE_MAIN_WELD,
                timeout,
                "main welding output state",
            )
            self._log("WELDING ACTIVE (RX Byte1=main welding)")
        return status

    def arc_off(
        self,
        *,
        wait_recognition_off: bool = True,
        wait_idle: bool = True,
        timeout: float = 5.0,
    ) -> Optional[WelderStatus]:
        """Clear Byte0.Bit0 but keep cyclic TCP communication alive.

        ARC OFF 직후 소켓을 닫지 않는다. OFF frame이 계속 송신되는 동안 용접기의
        종료/크레이터/post-gas sequence가 진행될 수 있기 때문이다.
        """
        with self._state_lock:
            self._state.command &= ~BIT_ARC

        self._log("ARC OFF command -> TX Byte0.Bit0 = 0; cyclic communication continues")

        status: Optional[WelderStatus] = None
        if wait_recognition_off:
            status = self.wait_for_status(lambda s: not s.arc_ack, timeout, "ARC/Torch recognition OFF")
            self._log("ARC OFF recognized by Hi-COMM")

        if wait_idle:
            status = self.wait_for_status(
                lambda s: s.output_state == OUTPUT_STATE_IDLE,
                timeout,
                "welder output idle",
            )
            self._log("WELDER OUTPUT IDLE")
        return status

    # ------------------------------------------------------------------
    # support commands
    # ------------------------------------------------------------------
    def _set_command_bit(self, mask: int, on: bool) -> None:
        with self._state_lock:
            if on:
                self._state.command |= mask
            else:
                self._state.command &= ~mask

    def gas_check(self, on: bool) -> None:
        if on and (self.tx_state().command & BIT_ARC):
            raise HiCommError("manual Gas Check rejected while ARC is ON")
        self._set_command_bit(BIT_GAS, on)
        self._log(f"Gas Check {'ON' if on else 'OFF'}")

    def stick_check(self, on: bool) -> None:
        if on and (self.tx_state().command & BIT_ARC):
            raise HiCommError("Stick Check rejected while ARC is ON")
        self._set_command_bit(BIT_STICK, on)
        self._log(f"Stick Check {'ON' if on else 'OFF'}")

    def inching(self, direction: str, seconds: float = 1.0) -> None:
        if self.tx_state().command & BIT_ARC:
            raise HiCommError("inching rejected while ARC is ON")
        key = direction.strip().lower()
        if key not in ("forward", "reverse"):
            raise ValueError("direction must be 'forward' or 'reverse'")
        mask = BIT_FORWARD if key == "forward" else BIT_REVERSE
        ack_name = "forward_ack" if key == "forward" else "reverse_ack"

        # 정/역인칭 mutual exclusion.
        started = time.monotonic()
        with self._state_lock:
            self._state.command &= ~(BIT_FORWARD | BIT_REVERSE)
            self._state.command |= mask
        self._log(f"{key} inching ON for {seconds:.2f}s")

        try:
            # ACK는 진단에 유용하지만 물리 동작을 보장하는 것은 아님.
            # ACK 대기시간까지 포함하여 전체 ON 시간을 seconds에 맞춘다.
            try:
                self.wait_for_status(
                    lambda s: bool(getattr(s, ack_name)),
                    min(1.0, max(0.2, seconds)),
                    f"{key} inch ACK",
                )
            except HiCommTimeout:
                self._log(f"WARN {key} inch ACK timeout; command remains ON until requested duration ends")
            remain = float(seconds) - (time.monotonic() - started)
            if remain > 0:
                time.sleep(remain)
        finally:
            self._set_command_bit(mask, False)
            self._log(f"{key} inching OFF")

    def set_robot_status(self, *, robot_error: bool, torch_collision: bool) -> None:
        """Set robot-side status bits sent to Hi-COMM in Byte0.Bit7/Bit6."""
        with self._state_lock:
            self._state.robot_error = bool(robot_error)
            self._state.torch_collision = bool(torch_collision)

    def all_outputs_off(self) -> None:
        with self._state_lock:
            self._state.command = 0

    # ------------------------------------------------------------------
    # useful runtime data for robot/GUI logging
    # ------------------------------------------------------------------
    def telemetry(self) -> dict[str, object]:
        s = self.latest_status()
        tx = self.tx_state()
        if s is None:
            return {
                "connected": self.connected,
                "comm_alive": False,
                "tx_command": tx.command,
                "status": None,
            }
        return {
            "connected": self.connected,
            "comm_alive": self.comm_alive(),
            "tx_command": tx.command,
            "arc_command": bool(tx.command & BIT_ARC),
            "arc_recognized": s.arc_ack,
            "output_state": s.output_state_name,
            "feedback_current_a": s.feedback_current_a,
            "feedback_voltage_v": s.feedback_voltage_v,
            "wire_feed_m_min": s.wire_feed_m_min,
            "welder_error": s.welder_error,
            "db_available": s.db_available,
            "rx_extra7": s.extra7.hex(" ").upper(),
        }


# =============================================================================
# 5. CLI diagnostics / examples
# =============================================================================
def run_check(src: str, dst: str, port: int, seconds: float) -> int:
    """Connect and transmit captured-style IDLE cyclic frame only."""
    with HiCommWelder(src, dst, port) as welder:
        welder.wait_comm_alive(1.5)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            print(welder.telemetry())
            time.sleep(1.0)
    return 0


def run_inching_test(src: str, dst: str, port: int, direction: str, seconds: float) -> int:
    with HiCommWelder(src, dst, port) as welder:
        welder.wait_comm_alive(1.5)
        welder.inching(direction, seconds)
        time.sleep(0.5)
        print(welder.telemetry())
    return 0


def run_arc_dry_sequence(src: str, dst: str, port: int) -> int:
    """ARC Bit0을 켜지 않고 실제 welding API 순서만 확인한다."""
    recipe = WeldingRecipe(
        current_a=100,
        voltage_v=20.0,
        material="FE-Solid",
        diameter_mm=1.0,
        mode="LSM",
        gas="CO2",
        synergic=False,
        correction=0.0,
        pre_gas_s=0.5,
        post_gas_s=1.0,
    )
    with HiCommWelder(src, dst, port) as welder:
        welder.wait_comm_alive(1.5)
        welder.arc_set(recipe)
        time.sleep(0.5)
        print("setting echo:", welder.setting_echo())
        print("readiness:", welder.readiness())
        print("telemetry:", welder.telemetry())
        # intentionally NO arc_on()
    return 0


def run_arc_test(
    src: str,
    dst: str,
    port: int,
    seconds: float,
    current_a: int,
    voltage_v: float,
) -> int:
    """Explicit CLI ARC test. Only called when --enable-arc is also supplied."""
    recipe = WeldingRecipe(
        current_a=current_a,
        voltage_v=voltage_v,
        material="FE-Solid",
        diameter_mm=1.0,
        mode="LSM",
        gas="CO2",
        synergic=False,
        correction=0.0,
        pre_gas_s=0.5,
        post_gas_s=1.0,
    )

    with HiCommWelder(src, dst, port) as welder:
        welder.wait_comm_alive(1.5)
        welder.arc_set(recipe)
        time.sleep(0.5)
        print("setting echo:", welder.setting_echo())
        print("readiness:", welder.readiness())

        # wait_welding=False by default here because actual arc establishment
        # depends on the physical welding setup. Recognition is still checked.
        welder.arc_on(wait_recognition=True, wait_welding=False, timeout=3.0)
        try:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                print(welder.telemetry())
                time.sleep(0.2)
        finally:
            welder.arc_off(wait_recognition_off=True, wait_idle=True, timeout=5.0)
    return 0


def dump_frame() -> int:
    r = WeldingRecipe()
    frame = build_request(TxState(recipe=r))
    print(f"TX_SIZE={len(frame)} PERIOD={PERIOD_SECONDS*1000:.1f}ms RX_SIZE={RX_SIZE}")
    print(frame.hex(" ").upper())
    for i, value in enumerate(frame):
        label = ""
        if i == 0:
            label = " Byte0 command/status"
        elif i == 1:
            label = " material/diameter/mode"
        elif i == 2:
            label = " synergic/gas"
        elif i in (3, 4):
            label = " current"
        elif i in (5, 6):
            label = " voltage"
        elif i == 7:
            label = " correction"
        elif i in (8, 9):
            label = " pre-gas"
        elif i in (10, 11):
            label = " post-gas"
        elif 12 <= i <= 52:
            label = " golden advanced welding parameter"
        elif i >= 53:
            label = " captured undocumented trailer"
        print(f"Byte{i:02d}=0x{value:02X}{label}")
    return 0


# =============================================================================
# 6. Optional PyQt GUI
# =============================================================================
def run_gui(src: str, dst: str, port: int) -> int:
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        print("PyQt5가 없어 GUI를 실행할 수 없습니다:", exc, file=sys.stderr)
        print("core/ROS2 사용에는 PyQt5가 필요하지 않습니다.", file=sys.stderr)
        return 2

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Hi-COMM welding v4 · arc_set / arc_on / arc_off")
            self.resize(1250, 950)
            self.welder: Optional[HiCommWelder] = None

            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            # Network ---------------------------------------------------------
            net = QGroupBox("Network / captured wire profile")
            n = QGridLayout(net)
            self.src = QLineEdit(src)
            self.dst = QLineEdit(dst)
            self.port = QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(port)
            self.btn_connect = QPushButton("CONNECT")
            self.btn_disconnect = QPushButton("DISCONNECT")
            self.conn = QLabel("DISCONNECTED")
            n.addWidget(QLabel("PC/Rainbow IP"), 0, 0); n.addWidget(self.src, 0, 1)
            n.addWidget(QLabel("Hi-COMM IP"), 0, 2); n.addWidget(self.dst, 0, 3)
            n.addWidget(QLabel("Port"), 0, 4); n.addWidget(self.port, 0, 5)
            n.addWidget(self.btn_connect, 0, 6); n.addWidget(self.btn_disconnect, 0, 7)
            n.addWidget(self.conn, 0, 8)
            n.addWidget(QLabel("Wire"), 1, 0)
            n.addWidget(QLabel("captured TX55 / ~40ms / RX71 · documented TX0..52, RX0..63"), 1, 1, 1, 8)
            layout.addWidget(net)

            # ARC SET recipe --------------------------------------------------
            rg = QGroupBox("ARC SET · documented recipe fields")
            f = QGridLayout(rg)
            self.current = QSpinBox(); self.current.setRange(30, 400); self.current.setValue(100); self.current.setSuffix(" A")
            self.voltage = QDoubleSpinBox(); self.voltage.setRange(10, 40); self.voltage.setDecimals(1); self.voltage.setValue(20.0); self.voltage.setSuffix(" V")
            self.material = QComboBox(); self.material.addItems(["FE-Solid", "FE-Cored", "STS-Solid", "STS-Cored", "AL-Soft", "AL-Hard", "CuSi", "CuMg"])
            self.diameter = QComboBox(); self.diameter.addItems(["0.8", "0.9", "1.0", "1.2", "1.4", "1.6"]); self.diameter.setCurrentText("1.0")
            self.mode = QComboBox(); self.mode.addItems(["LSM", "DCM", "DPM", "PM"])
            self.gas_type = QComboBox(); self.gas_type.addItems(["CO2", "Ar80+CO2 20%", "Ar98+O2 2%", "Ar 100%"])
            self.synergic = QCheckBox("Synergic")
            self.correction = QDoubleSpinBox(); self.correction.setRange(-5, 5); self.correction.setDecimals(1); self.correction.setValue(0)
            self.pre = QDoubleSpinBox(); self.pre.setRange(0, 10); self.pre.setDecimals(2); self.pre.setValue(0.5); self.pre.setSuffix(" s")
            self.post = QDoubleSpinBox(); self.post.setRange(0, 10); self.post.setDecimals(2); self.post.setValue(1.0); self.post.setSuffix(" s")
            self.btn_arc_set = QPushButton("ARC SET")
            widgets = [
                ("Current", self.current), ("Voltage", self.voltage),
                ("Material", self.material), ("Diameter", self.diameter),
                ("Mode", self.mode), ("Gas", self.gas_type),
                ("Correction", self.correction), ("Pre-gas", self.pre),
                ("Post-gas", self.post),
            ]
            for i, (name, w) in enumerate(widgets):
                row, col = divmod(i, 3)
                f.addWidget(QLabel(name), row, col * 2)
                f.addWidget(w, row, col * 2 + 1)
            f.addWidget(self.synergic, 3, 0, 1, 2)
            f.addWidget(self.btn_arc_set, 3, 4, 1, 2)
            layout.addWidget(rg)

            # Commands --------------------------------------------------------
            cg = QGroupBox("Commands / actual welding sequence")
            c = QGridLayout(cg)
            self.forward = QPushButton("정인칭 1s")
            self.reverse = QPushButton("역인칭 1s")
            self.gas = QCheckBox("Gas Check")
            self.stick = QCheckBox("Stick Check")
            self.arc_unlock = QCheckBox("ARC command unlock")
            self.btn_arc_on = QPushButton("ARC ON")
            self.btn_arc_off = QPushButton("ARC OFF")
            self.btn_all_off = QPushButton("ALL OUTPUT OFF")
            c.addWidget(self.forward, 0, 0); c.addWidget(self.reverse, 0, 1)
            c.addWidget(self.gas, 0, 2); c.addWidget(self.stick, 0, 3)
            c.addWidget(self.arc_unlock, 1, 0)
            c.addWidget(self.btn_arc_on, 1, 1); c.addWidget(self.btn_arc_off, 1, 2)
            c.addWidget(self.btn_all_off, 1, 3)
            layout.addWidget(cg)

            # Status ----------------------------------------------------------
            sg = QGroupBox("Hi-COMM RX / welding monitor")
            sf = QFormLayout(sg)
            self.st_comm = QLabel("-")
            self.st_ready = QLabel("-")
            self.st_cmd = QLabel("-")
            self.st_ack = QLabel("-")
            self.st_output = QLabel("-")
            self.st_fb = QLabel("-")
            self.st_error = QLabel("-")
            self.st_echo = QLabel("-")
            self.st_extra = QLabel("-")
            for title, w in [
                ("Cyclic comm", self.st_comm),
                ("ARC readiness", self.st_ready),
                ("TX command", self.st_cmd),
                ("RX recognition", self.st_ack),
                ("Output state", self.st_output),
                ("Feedback I/V/feed", self.st_fb),
                ("Welder error / DB", self.st_error),
                ("Setting echo", self.st_echo),
                ("RX extra7", self.st_extra),
            ]:
                sf.addRow(title, w)
            layout.addWidget(sg)

            self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2000)
            layout.addWidget(self.log, 1)

            # callbacks -------------------------------------------------------
            self.btn_connect.clicked.connect(self.do_connect)
            self.btn_disconnect.clicked.connect(self.do_disconnect)
            self.btn_arc_set.clicked.connect(self.do_arc_set)
            self.forward.clicked.connect(lambda: self.do_inch("forward"))
            self.reverse.clicked.connect(lambda: self.do_inch("reverse"))
            self.gas.toggled.connect(self.do_gas)
            self.stick.toggled.connect(self.do_stick)
            self.btn_arc_on.clicked.connect(self.do_arc_on)
            self.btn_arc_off.clicked.connect(self.do_arc_off)
            self.btn_all_off.clicked.connect(self.do_all_off)

            self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.refresh); self.timer.start()
            self.set_controls(False)

        def append(self, msg: str) -> None:
            self.log.appendPlainText(msg)

        def set_controls(self, enabled: bool) -> None:
            for w in [self.btn_disconnect, self.btn_arc_set, self.forward, self.reverse,
                      self.gas, self.stick, self.arc_unlock, self.btn_arc_on,
                      self.btn_arc_off, self.btn_all_off]:
                w.setEnabled(enabled)
            self.btn_connect.setEnabled(not enabled)

        def make_recipe(self) -> WeldingRecipe:
            return WeldingRecipe(
                current_a=self.current.value(),
                voltage_v=self.voltage.value(),
                material=self.material.currentText(),
                diameter_mm=float(self.diameter.currentText()),
                mode=self.mode.currentText(),
                gas=self.gas_type.currentText(),
                synergic=self.synergic.isChecked(),
                correction=self.correction.value(),
                pre_gas_s=self.pre.value(),
                post_gas_s=self.post.value(),
            )

        def do_connect(self) -> None:
            try:
                self.welder = HiCommWelder(
                    self.src.text().strip(), self.dst.text().strip(), self.port.value(),
                    logger=self.append,
                )
                self.welder.connect()
                self.set_controls(True)
                self.conn.setText("CONNECTED")
            except Exception as exc:
                QMessageBox.critical(self, "Connect", str(exc))
                self.welder = None

        def do_disconnect(self) -> None:
            if self.welder:
                self.welder.disconnect()
            self.welder = None
            self.set_controls(False)
            self.conn.setText("DISCONNECTED")

        def do_arc_set(self) -> None:
            if not self.welder: return
            try:
                self.welder.arc_set(self.make_recipe())
            except Exception as exc:
                QMessageBox.critical(self, "ARC SET", str(exc))

        def do_inch(self, direction: str) -> None:
            if not self.welder: return
            # GUI thread를 막지 않게 worker thread 사용.
            threading.Thread(target=self._do_inch_worker, args=(direction,), daemon=True).start()

        def _do_inch_worker(self, direction: str) -> None:
            try:
                assert self.welder is not None
                self.welder.inching(direction, 1.0)
            except Exception as exc:
                self.append(f"INCH ERROR: {exc}")

        def do_gas(self, on: bool) -> None:
            if not self.welder: return
            try: self.welder.gas_check(on)
            except Exception as exc: self.append(f"GAS ERROR: {exc}")

        def do_stick(self, on: bool) -> None:
            if not self.welder: return
            try: self.welder.stick_check(on)
            except Exception as exc: self.append(f"STICK ERROR: {exc}")

        def do_arc_on(self) -> None:
            if not self.welder: return
            if not self.arc_unlock.isChecked():
                QMessageBox.warning(self, "ARC", "ARC command unlock을 먼저 체크하세요.")
                return
            answer = QMessageBox.question(
                self, "ARC ON", "Byte0.Bit0 ARC/Torch ON 명령을 실제로 송신합니다. 계속합니까?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            threading.Thread(target=self._arc_on_worker, daemon=True).start()

        def _arc_on_worker(self) -> None:
            try:
                assert self.welder is not None
                self.welder.arc_on(wait_recognition=True, wait_welding=False, timeout=3.0)
            except Exception as exc:
                self.append(f"ARC ON ERROR: {exc}")

        def do_arc_off(self) -> None:
            if not self.welder: return
            threading.Thread(target=self._arc_off_worker, daemon=True).start()

        def _arc_off_worker(self) -> None:
            try:
                assert self.welder is not None
                self.welder.arc_off(wait_recognition_off=True, wait_idle=True, timeout=5.0)
            except Exception as exc:
                self.append(f"ARC OFF ERROR: {exc}")

        def do_all_off(self) -> None:
            if self.welder:
                self.welder.all_outputs_off()
            self.gas.blockSignals(True); self.stick.blockSignals(True)
            self.gas.setChecked(False); self.stick.setChecked(False)
            self.gas.blockSignals(False); self.stick.blockSignals(False)

        def refresh(self) -> None:
            w = self.welder
            if not w:
                return
            tx = w.tx_state()
            status = w.latest_status()
            self.st_comm.setText("ALIVE" if w.comm_alive() else "NO RECENT RX")
            ready, reasons = w.readiness()
            self.st_ready.setText("READY" if ready else "BLOCK: " + "; ".join(reasons))
            self.st_cmd.setText(f"0x{build_request(tx)[0]:02X}")
            if status:
                ack = []
                if status.arc_ack: ack.append("ARC")
                if status.forward_ack: ack.append("FWD")
                if status.reverse_ack: ack.append("REV")
                if status.gas_ack: ack.append("GAS")
                if status.stick_ack: ack.append("STICK")
                self.st_ack.setText(", ".join(ack) if ack else "none")
                self.st_output.setText(status.output_state_name)
                self.st_fb.setText(f"{status.feedback_current_a}A / {status.feedback_voltage_v:.1f}V / {status.wire_feed_m_min:.1f}m/min")
                self.st_error.setText(f"err={status.welder_error}, DB={'OK' if status.db_available else 'UNAVAILABLE'}")
                e = w.setting_echo()
                self.st_echo.setText(str(e.get("checks", e)))
                self.st_extra.setText(status.extra7.hex(" ").upper())

        def closeEvent(self, event) -> None:  # noqa: N802
            self.do_disconnect()
            event.accept()

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


# =============================================================================
# 7. main
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Hi-COMM Rainbow welding control v4")
    p.add_argument("--source-ip", default=DEFAULT_SOURCE_IP)
    p.add_argument("--welder-ip", default=DEFAULT_HICOMM_IP)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--current", type=int, default=100)
    p.add_argument("--voltage", type=float, default=20.0)

    p.add_argument("--check", action="store_true", help="IDLE cyclic TX/RX only")
    p.add_argument("--inching-test", choices=("forward", "reverse"))
    p.add_argument("--arc-dry-sequence", action="store_true", help="arc_set/readiness만 수행; ARC ON 안 함")
    p.add_argument("--arc-test", action="store_true", help="실제 ARC ON/OFF CLI 시험")
    p.add_argument("--enable-arc", action="store_true", help="--arc-test를 실제 실행하기 위한 명시적 enable")
    p.add_argument("--dump-frame", action="store_true")
    p.add_argument("--no-gui", action="store_true", help="아무 진단 옵션이 없을 때 GUI를 열지 않음")
    args = p.parse_args()

    if args.dump_frame:
        raise SystemExit(dump_frame())
    if args.check:
        raise SystemExit(run_check(args.source_ip, args.welder_ip, args.port, args.seconds))
    if args.inching_test:
        raise SystemExit(run_inching_test(args.source_ip, args.welder_ip, args.port, args.inching_test, args.seconds))
    if args.arc_dry_sequence:
        raise SystemExit(run_arc_dry_sequence(args.source_ip, args.welder_ip, args.port))
    if args.arc_test:
        if not args.enable_arc:
            print("--arc-test requires explicit --enable-arc", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(
            run_arc_test(
                args.source_ip, args.welder_ip, args.port,
                args.seconds, args.current, args.voltage,
            )
        )
    if args.no_gui:
        return

    raise SystemExit(run_gui(args.source_ip, args.welder_ip, args.port))


if __name__ == "__main__":
    main()
