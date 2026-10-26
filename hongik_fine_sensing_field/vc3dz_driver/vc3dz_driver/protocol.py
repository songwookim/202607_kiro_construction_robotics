"""VC nano 3D-Z TCP(1096) 프로토콜: 상수·struct·메시지 조립/파싱.

메시지 = [Global Header 32 B][Meta][Sub][Payload], 전부 little-endian.
근거: Laser Scanner Manual 1.8.8 복원본 + SmartShape V5.2 exe 정적분석 + 2026-07-24 기록 데이터
(설계서 §3.1). DataMode 4 서브헤더는 4 B(NumberOfPoints)만이며 LineCounter 는 와이어에 없다
(1.8.8 §12 산문의 "8 Bytes"는 문서 오류로 취급) — 단, extra == 4 인 경우 LineCounter 변형으로 방어 파싱.

이 모듈은 소켓을 만지지 않는다(순수 함수 + dataclass) → 테스트·pcap 재생·fake sensor 가 공유.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
SYNC = 0xAA0F55F0                       # exe 에 존재하는 유일한 동기어
SYNC_BYTES = struct.pack('<I', SYNC)    # 와이어 바이트열: F0 55 0F AA

GH = struct.Struct('<IIIIiiiI')   # 32 B Global Header: sync, cks_hdr, cks_data, counter, data_wait_ms, reserved, error, data_length
CMD_TX = struct.Struct('<IIII')   # 16 B Host→Sensor STRING meta: cmd_type(0), param_id, host_cmd_id, str_len
CMD_RX = struct.Struct('<IIIiI')  # 20 B Sensor→Host STRING meta: cmd_type(0), param_id, host_cmd_id_reply, error, str_len
RES = struct.Struct('<IiIQ')      # 20 B Sensor→Host RESULT meta: cmd_type(100), data_mode, result_counter, ts_ms(u64, since boot)
SUB4 = struct.Struct('<I')        # 4 B DataMode 4/5 sub header: number_of_points (LineCounter 없음)
SUB4_LC = struct.Struct('<II')    # 8 B 변형(다른 펌웨어): number_of_points, line_counter

GH_SIZE = GH.size            # 32
CMD_TX_SIZE = CMD_TX.size    # 16
CMD_RX_SIZE = CMD_RX.size    # 20
RES_SIZE = RES.size          # 20
SUB4_SIZE = SUB4.size        # 4
MODE4_PAYLOAD_OFFSET = RES_SIZE + SUB4_SIZE   # 24 = data 오프셋 (메시지 오프셋 56)

CMD_TYPE_STRING = 0
CMD_TYPE_RESULT = 100

# Parameter IDs (STRING command)
PARAM_ID_RESET = 0                 # 정지(NbrLines=0 과 동등)
PARAM_ID_SAVE_SENSOR_FILE = 33     # 플래시 저장 — 금지
PARAM_ID_SAVE_PRODUCT_FILE = 45    # 플래시 저장 — 금지 (제품 파라미터)
PARAM_ID_LOAD_SENSOR_FILE = 42
PARAM_ID_WRITE_PARAMS = 43         # RECEIVE_SENSOR_DATA (host→sensor 설정)
PARAM_ID_READ_PARAMS = 44          # SEND_SENSOR_DATA  (str_len 0 으로 요청 → 현 설정 전문)
FORBIDDEN_PARAM_IDS = frozenset({PARAM_ID_SAVE_SENSOR_FILE, PARAM_ID_SAVE_PRODUCT_FILE})

# DataMode
DATA_MODE_STANDBY = 0
DATA_MODE_TAKE_PIC = 1
DATA_MODE_2D_POINTLIST = 2
DATA_MODE_2D_POINTLIST_AND_LASER_IMG = 3
DATA_MODE_2D_POINTLIST_MM = 4          # x0 z0 x1 z1 ...
DATA_MODE_2D_POINTLIST_MM_SPLIT = 5    # x0 x1 ... z0 z1 ...
DATA_MODE_BINARY_PIC = 6
DATA_MODE_PRODUCT_RESULTS = 8
DATA_MODE_4_AND_8 = 9

# Sensor error codes (Global Header error / STRING reply error)
ERR_NONE = 0
ERR_TCP_IP = -3001
ERR_PARM = -3003
ERR_COMMAND = -3004
ERR_CMD_STRG_LEN = -3010
ERR_CMD_TYPE = -3011
ERR_CMD_DATA_LEN = -3012
ERR_MEM_PROD_ANGLE = -3018
ERROR_NAMES = {
    ERR_NONE: 'ERR_NONE', ERR_TCP_IP: 'ERR_TCP_IP', ERR_PARM: 'ERR_PARM',
    ERR_COMMAND: 'ERR_COMMAND', ERR_CMD_STRG_LEN: 'ERR_CMD_STRG_LEN',
    ERR_CMD_TYPE: 'ERR_CMD_TYPE', ERR_CMD_DATA_LEN: 'ERR_CMD_DATA_LEN',
    ERR_MEM_PROD_ANGLE: 'ERR_MEM_PROD_ANGLE',
}

DEFAULT_WAIT_MS = 1000   # SmartShape 가 보내는 data_wait_ms
U32_MASK = 0xFFFFFFFF


class ProtocolError(ValueError):
    """프레임 형식 불일치. 메시지에 첫 64 B hexdump 포함."""


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def hexdump(b: bytes, n: int = 64, width: int = 16) -> str:
    """첫 n 바이트를 오프셋 + hex + ASCII 로 덤프."""
    b = bytes(b[:n])
    lines = []
    for off in range(0, len(b), width):
        chunk = b[off:off + width]
        hx = ' '.join(f'{c:02x}' for c in chunk)
        asc = ''.join(chr(c) if 32 <= c < 127 else '.' for c in chunk)
        lines.append(f'{off:04x}: {hx:<{width * 3 - 1}}  {asc}')
    return '\n'.join(lines)


def xor_checksum(b: bytes) -> int:
    """바이트 XOR 체크섬(0..255). 빈 입력은 0.

    매뉴얼: "Checksum is an easy XOR combination of all bytes". 32-bit 워드 XOR 변형 가능성은
    P0 Wireshark 대조로 확정(설계 §3.1) — 수신측은 불일치를 경고만 한다.
    """
    if not b:
        return 0
    return int(np.bitwise_xor.reduce(np.frombuffer(bytes(b), dtype=np.uint8)))


def xor_checksum_u32(b: bytes) -> int:
    """대안 후보: 32-bit little-endian 워드 XOR (길이가 4의 배수가 아니면 0 패딩). 대조용."""
    if not b:
        return 0
    b = bytes(b)
    pad = (-len(b)) % 4
    if pad:
        b = b + b'\x00' * pad
    return int(np.bitwise_xor.reduce(np.frombuffer(b, dtype='<u4')))


# ---------------------------------------------------------------------------
# Global Header
# ---------------------------------------------------------------------------
@dataclass
class GlobalHeader:
    sync: int
    cks_hdr: int
    cks_data: int
    counter: int
    data_wait_ms: int
    reserved: int
    error: int
    data_length: int

    @property
    def sync_ok(self) -> bool:
        return self.sync == SYNC

    def pack(self) -> bytes:
        return GH.pack(self.sync, self.cks_hdr, self.cks_data, self.counter,
                       self.data_wait_ms, self.reserved, self.error, self.data_length)


def header_checksum(header_bytes: bytes) -> int:
    """Global Header 체크섬 = 바이트 12..31 (Counter ~ DataLength) XOR."""
    return xor_checksum(header_bytes[12:32])


def parse_global_header(buf: bytes, offset: int = 0, check_sync: bool = True) -> GlobalHeader:
    if len(buf) - offset < GH_SIZE:
        raise ProtocolError(f'global header needs {GH_SIZE} B, got {len(buf) - offset}\n'
                            + hexdump(buf[offset:]))
    h = GlobalHeader(*GH.unpack_from(buf, offset))
    if check_sync and not h.sync_ok:
        raise ProtocolError(f'bad sync 0x{h.sync:08X} (expected 0x{SYNC:08X})\n'
                            + hexdump(buf[offset:offset + 64]))
    return h


def verify_checksums(header_bytes: bytes, data: bytes) -> Tuple[bool, bool]:
    """(header_ok, data_ok). 바이트 XOR 기준."""
    h = GlobalHeader(*GH.unpack_from(header_bytes, 0))
    return (h.cks_hdr == header_checksum(header_bytes), h.cks_data == xor_checksum(data))


def build_message(data: bytes, counter: int, wait_ms: int = DEFAULT_WAIT_MS,
                  error: int = 0, reserved: int = 0) -> bytes:
    """Global Header(체크섬 계산 포함) + data."""
    tail = struct.pack('<IiiiI', counter & U32_MASK, wait_ms, reserved, error, len(data))
    hdr = struct.pack('<III', SYNC, xor_checksum(tail), xor_checksum(data)) + tail
    assert len(hdr) == GH_SIZE
    return hdr + data


# ---------------------------------------------------------------------------
# Host → Sensor STRING command
# ---------------------------------------------------------------------------
def build_string_command(param_id: int, text: str = '', counter: int = 0, host_cmd_id: int = 1,
                         wait_ms: int = DEFAULT_WAIT_MS) -> bytes:
    """STRING 명령 1개 (Global Header 32 B + Meta 16 B + ASCII).

    text: 'Name = value\\n' 항목 연결 (LF 종단). ID 44 는 빈 문자열(str_len 0).
    ID 33/45(플래시 저장)는 절대 조립하지 않는다.
    """
    if param_id in FORBIDDEN_PARAM_IDS:
        raise ValueError(f'param_id {param_id} (flash save) is forbidden by policy')
    body = text.encode('ascii')
    meta = CMD_TX.pack(CMD_TYPE_STRING, int(param_id), host_cmd_id & U32_MASK, len(body))
    return build_message(meta + body, counter, wait_ms)


@dataclass
class StringCommand:
    """호스트→센서 STRING 명령 파싱 결과 (pcap 재생·fake sensor 용)."""
    kind: str
    param_id: int
    host_cmd_id: int
    text: str
    cmd_type: int = CMD_TYPE_STRING


# ---------------------------------------------------------------------------
# Sensor → Host messages
# ---------------------------------------------------------------------------
@dataclass
class StringReply:
    kind: str                # 'string_reply'
    param_id: int
    host_cmd_id: int
    error: int
    text: str
    cmd_type: int = CMD_TYPE_STRING

    @property
    def ok(self) -> bool:
        return self.error == 0

    @property
    def error_name(self) -> str:
        return ERROR_NAMES.get(self.error, f'ERR_{self.error}')


@dataclass
class ResultFrame:
    kind: str                # 'result'
    data_mode: int
    result_counter: int
    ts_ms: int
    n_points: int
    x: np.ndarray            # float32 [mm]
    z: np.ndarray            # float32 [mm]
    has_line_counter: bool = False
    line_counter: Optional[int] = None
    cmd_type: int = CMD_TYPE_RESULT
    raw_payload: Optional[bytes] = None   # 비지원 data_mode 일 때만 채움


Message = Union[StringReply, ResultFrame, StringCommand]


def parse_message(data: bytes, data_length: Optional[int] = None) -> Message:
    """Global Header 이후의 data(= data_length 바이트)를 파싱.

    cmd_type 0  → StringReply (20 B meta) — 길이가 16 B meta 로만 맞으면 StringCommand(호스트 방향)
    cmd_type 100 → ResultFrame; data_mode 4: (x,z) 쌍, 5: x[N] 다음 z[N]
    DataMode 4/5 서브헤더 4 B 가 1차 경로, extra == 4 이면 LineCounter 변형(8 B) 폴백,
    그 외 길이 불일치는 ProtocolError(hexdump 포함).
    """
    data = bytes(data)
    if data_length is None:
        data_length = len(data)
    if data_length != len(data):
        raise ProtocolError(f'data_length {data_length} != len(data) {len(data)}\n' + hexdump(data))
    if data_length < 4:
        raise ProtocolError(f'data too short ({data_length} B)\n' + hexdump(data))
    cmd_type = struct.unpack_from('<I', data, 0)[0]

    if cmd_type == CMD_TYPE_STRING:
        # 센서→호스트 응답(20 B meta) 우선, 호스트→센서 명령(16 B meta)은 길이로 판별
        if data_length >= CMD_RX_SIZE:
            _, pid, hid, err, slen = CMD_RX.unpack_from(data, 0)
            if CMD_RX_SIZE + slen == data_length:
                text = data[CMD_RX_SIZE:CMD_RX_SIZE + slen].decode('ascii', 'replace')
                return StringReply('string_reply', pid, hid, err, text)
        if data_length >= CMD_TX_SIZE:
            _, pid, hid, slen = CMD_TX.unpack_from(data, 0)
            if CMD_TX_SIZE + slen == data_length:
                text = data[CMD_TX_SIZE:CMD_TX_SIZE + slen].decode('ascii', 'replace')
                return StringCommand('string_command', pid, hid, text)
        # 길이가 어느 쪽에도 안 맞음 — 20 B meta 로 관대하게 해석
        if data_length >= CMD_RX_SIZE:
            _, pid, hid, err, slen = CMD_RX.unpack_from(data, 0)
            text = data[CMD_RX_SIZE:].decode('ascii', 'replace')
            return StringReply('string_reply', pid, hid, err, text)
        raise ProtocolError(f'STRING message too short ({data_length} B)\n' + hexdump(data))

    if cmd_type == CMD_TYPE_RESULT:
        if data_length < RES_SIZE:
            raise ProtocolError(f'RESULT meta needs {RES_SIZE} B, got {data_length}\n' + hexdump(data))
        _, mode, rcnt, ts = RES.unpack_from(data, 0)
        if mode in (DATA_MODE_2D_POINTLIST_MM, DATA_MODE_2D_POINTLIST_MM_SPLIT):
            if data_length < RES_SIZE + SUB4_SIZE:
                raise ProtocolError(f'DataMode {mode} sub header missing (len {data_length})\n' + hexdump(data))
            n = SUB4.unpack_from(data, RES_SIZE)[0]
            extra = data_length - MODE4_PAYLOAD_OFFSET - 8 * n
            if extra == 0:
                off, line, has_lc = MODE4_PAYLOAD_OFFSET, None, False
            elif extra == 4:
                line = struct.unpack_from('<I', data, MODE4_PAYLOAD_OFFSET)[0]
                off, has_lc = MODE4_PAYLOAD_OFFSET + 4, True
            else:
                raise ProtocolError(
                    f'DataMode {mode} length mismatch: data_length={data_length} n={n} extra={extra} '
                    f'(expected 24+8N or 28+8N)\n' + hexdump(data))
            flat = np.frombuffer(data, dtype='<f4', count=2 * n, offset=off)
            if mode == DATA_MODE_2D_POINTLIST_MM:
                xz = flat.reshape(n, 2)
                x, z = xz[:, 0].copy(), xz[:, 1].copy()
            else:
                x, z = flat[:n].copy(), flat[n:].copy()
            return ResultFrame('result', mode, rcnt, ts, n, x, z, has_lc, line)
        # 지원하지 않는 모드: 메타만 + raw payload
        return ResultFrame('result', mode, rcnt, ts, 0, np.zeros(0, np.float32), np.zeros(0, np.float32),
                           False, None, raw_payload=data[RES_SIZE:])

    raise ProtocolError(f'unknown cmd_type {cmd_type}\n' + hexdump(data))


def build_result_message(x, z, result_counter: int, ts_ms: int, counter: int,
                         data_mode: int = DATA_MODE_2D_POINTLIST_MM, line_counter: Optional[int] = None,
                         wait_ms: int = 0, error: int = 0) -> bytes:
    """센서→호스트 RESULT 메시지 조립(테스트·fake sensor·pcap 대조용).

    line_counter 를 주면 8 B 서브헤더 변형(extra == 4)으로 조립한다.
    """
    x = np.asarray(x, dtype='<f4').ravel()
    z = np.asarray(z, dtype='<f4').ravel()
    if x.shape != z.shape:
        raise ValueError('x and z must have the same length')
    n = int(x.size)
    meta = RES.pack(CMD_TYPE_RESULT, int(data_mode), result_counter & U32_MASK, int(ts_ms))
    if line_counter is None:
        sub = SUB4.pack(n)
    else:
        sub = SUB4_LC.pack(n, line_counter & U32_MASK)
    if data_mode == DATA_MODE_2D_POINTLIST_MM:
        payload = np.column_stack([x, z]).astype('<f4').tobytes()
    elif data_mode == DATA_MODE_2D_POINTLIST_MM_SPLIT:
        payload = x.tobytes() + z.tobytes()
    else:
        raise ValueError(f'unsupported data_mode {data_mode}')
    return build_message(meta + sub + payload, counter, wait_ms, error=error)


def build_string_reply(param_id: int, host_cmd_id: int, text: str = '', error: int = 0,
                       counter: int = 0, wait_ms: int = 0) -> bytes:
    """센서→호스트 STRING 응답 조립(fake sensor 용)."""
    body = text.encode('ascii')
    meta = CMD_RX.pack(CMD_TYPE_STRING, int(param_id), host_cmd_id & U32_MASK, int(error), len(body))
    return build_message(meta + body, counter, wait_ms)


# ---------------------------------------------------------------------------
# 스트림 파서 (sync 재동기)
# ---------------------------------------------------------------------------
@dataclass
class ParsedMessage:
    offset: int              # 스트림 내 Global Header 시작 오프셋
    header: GlobalHeader
    header_ok: bool          # 헤더 체크섬 일치(바이트 XOR)
    data_ok: bool            # 데이터 체크섬 일치(바이트 XOR)
    message: Optional[Message]
    error: Optional[str] = None   # parse_message 실패 사유 (message None)
    raw_data: Optional[bytes] = None


@dataclass
class StreamStats:
    messages: int = 0
    results: int = 0
    string_replies: int = 0
    string_commands: int = 0
    resyncs: int = 0            # sync 불일치로 건너뛴 횟수(위치)
    skipped_bytes: int = 0
    checksum_mismatch_hdr: int = 0
    checksum_mismatch_data: int = 0
    format_errors: int = 0
    counter_gaps: int = 0       # Global counter 불연속
    error_nonzero: int = 0
    last_counter: Optional[int] = None


class StreamParser:
    """바이트 스트림 → 메시지. feed() 로 조각을 넣고 완성된 메시지를 돌려받는다.

    sync 불일치 시 1 바이트 슬라이딩 재동기. 체크섬 불일치는 경고 카운트만(폐기 없음).
    형식 오류(길이 불일치 등)는 해당 메시지만 폐기(ParsedMessage.message None, error 문자열).
    """

    def __init__(self, max_data_length: int = 64 * 1024 * 1024):
        self.buf = bytearray()
        self.base_offset = 0            # buf[0] 의 스트림 절대 오프셋
        self.stats = StreamStats()
        self.max_data_length = max_data_length

    def feed(self, chunk: bytes) -> List[ParsedMessage]:
        self.buf += chunk
        out: List[ParsedMessage] = []
        while True:
            # 1) sync 검색
            idx = self.buf.find(SYNC_BYTES)
            if idx < 0:
                keep = len(self.buf) - 3 if len(self.buf) > 3 else 0
                if keep > 0:
                    self.stats.skipped_bytes += keep
                    self.stats.resyncs += 1
                    del self.buf[:keep]
                    self.base_offset += keep
                break
            if idx > 0:
                self.stats.skipped_bytes += idx
                self.stats.resyncs += 1
                del self.buf[:idx]
                self.base_offset += idx
            if len(self.buf) < GH_SIZE:
                break
            hdr_bytes = bytes(self.buf[:GH_SIZE])
            h = GlobalHeader(*GH.unpack(hdr_bytes))
            if h.data_length > self.max_data_length:
                # 말이 안 되는 길이 → 가짜 sync 로 보고 1 B 건너뜀
                del self.buf[:1]
                self.base_offset += 1
                self.stats.resyncs += 1
                self.stats.skipped_bytes += 1
                continue
            total = GH_SIZE + h.data_length
            if len(self.buf) < total:
                break
            data = bytes(self.buf[GH_SIZE:total])
            hdr_ok = h.cks_hdr == header_checksum(hdr_bytes)
            data_ok = h.cks_data == xor_checksum(data)
            pm = ParsedMessage(self.base_offset, h, hdr_ok, data_ok, None, raw_data=data)
            try:
                pm.message = parse_message(data, h.data_length)
            except ProtocolError as e:
                pm.error = str(e)
                self.stats.format_errors += 1
            self._account(pm)
            out.append(pm)
            del self.buf[:total]
            self.base_offset += total
        return out

    def _account(self, pm: ParsedMessage) -> None:
        s = self.stats
        s.messages += 1
        if not pm.header_ok:
            s.checksum_mismatch_hdr += 1
        if not pm.data_ok:
            s.checksum_mismatch_data += 1
        if pm.header.error != 0:
            s.error_nonzero += 1
        c = pm.header.counter
        if s.last_counter is not None and c != 0 and ((s.last_counter + 1) & U32_MASK) != c:
            s.counter_gaps += 1
        s.last_counter = c
        m = pm.message
        if isinstance(m, ResultFrame):
            s.results += 1
        elif isinstance(m, StringReply):
            s.string_replies += 1
        elif isinstance(m, StringCommand):
            s.string_commands += 1


def parse_stream(buf: bytes) -> Iterator[ParsedMessage]:
    """바이트열 전체를 한 번에 파싱(재동기 포함). 완성되지 않은 꼬리는 버린다."""
    p = StreamParser()
    for pm in p.feed(bytes(buf)):
        yield pm


def message_offsets_table(data: bytes) -> List[Tuple[int, str, str]]:
    """검증표: (메시지 오프셋, 필드, 값). Wireshark 대조용(설계 §3.6 체크리스트)."""
    rows: List[Tuple[int, str, str]] = []
    h = parse_global_header(data, 0, check_sync=False)
    rows += [(0, 'sync', f'0x{h.sync:08X} ' + ('OK' if h.sync_ok else 'BAD')),
             (4, 'cks_hdr', f'{h.cks_hdr} (calc byte-xor {header_checksum(data[:32])}, u32-xor {xor_checksum_u32(data[12:32])})'),
             (8, 'cks_data', f'{h.cks_data} (calc byte-xor {xor_checksum(data[32:32 + h.data_length])}, '
                             f'u32-xor {xor_checksum_u32(data[32:32 + h.data_length])})'),
             (12, 'counter', str(h.counter)), (16, 'data_wait_ms', str(h.data_wait_ms)),
             (20, 'reserved', str(h.reserved)), (24, 'error', str(h.error)),
             (28, 'data_length', str(h.data_length))]
    d = data[32:32 + h.data_length]
    if len(d) >= 4:
        ct = struct.unpack_from('<I', d, 0)[0]
        rows.append((32, 'cmd_type', str(ct)))
        if ct == CMD_TYPE_RESULT and len(d) >= RES_SIZE:
            _, mode, rc, ts = RES.unpack_from(d, 0)
            rows += [(36, 'data_mode', str(mode)), (40, 'result_counter', str(rc)), (44, 'ts_ms(u64)', str(ts))]
            if len(d) >= RES_SIZE + 4:
                n = SUB4.unpack_from(d, RES_SIZE)[0]
                extra = h.data_length - 24 - 8 * n
                rows.append((52, 'number_of_points', f'{n} (extra={extra}: '
                             + ('OK 4B sub' if extra == 0 else 'LineCounter variant' if extra == 4 else 'MISMATCH') + ')'))
                if extra == 4:
                    rows.append((56, 'line_counter', str(struct.unpack_from('<I', d, 24)[0])))
                off = 24 + (4 if extra == 4 else 0)
                if n and len(d) >= off + 16:
                    f = np.frombuffer(d, '<f4', 4, off)
                    rows.append((32 + off, 'first floats', f'x0={f[0]:.3f} z0={f[1]:.3f} x1={f[2]:.3f} z1={f[3]:.3f}'))
        elif ct == CMD_TYPE_STRING:
            if len(d) >= CMD_RX_SIZE:
                _, pid, hid, err, slen = CMD_RX.unpack_from(d, 0)
                rows += [(36, 'param_id', str(pid)), (40, 'host_cmd_id', str(hid)),
                         (44, 'error(i32) / str_len if 16B meta', str(err)), (48, 'str_len', str(slen))]
            txt = d[16:].decode('ascii', 'replace')
            rows.append((48, 'text(16B meta view)', repr(txt[:80])))
    return rows
