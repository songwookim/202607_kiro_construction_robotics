"""VC nano 3D-Z TCP 클라이언트 (설계 §3.1–3.2, §10.2).

- 연결 직후 counter = 0 (상대 카운터 리셋), 이후 1 씩 증가.
- 전용 수신 스레드: recv_exact(32) → sync 검사(불일치 시 1 B 슬라이딩 재동기) → recv_exact(data_length)
  → 수신 완료 직후 t_rx_ns = time.monotonic_ns() → cmd_type 으로 STRING 응답 큐 / 결과 큐 분기.
  STRING 과 RUN 이 같은 소켓이므로 스트리밍 중 ID 44 를 읽어도 오파싱하지 않는다.
- ID 33/45(플래시 저장)는 절대 보내지 않는다(ValueError).
- 무결성 모니터: counter 불연속, error≠0, 체크섬 불일치, 형식 오류, N 변화 → stats 카운트(폐기는 형식 오류만).
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import queue
import socket
import threading
import warnings
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from . import params as P
from .clock import NULL_LINK, get_clock
from .protocol import (CMD_TYPE_RESULT, CMD_TYPE_STRING, DEFAULT_WAIT_MS, FORBIDDEN_PARAM_IDS, GH, GH_SIZE,
                       PARAM_ID_READ_PARAMS, PARAM_ID_RESET, PARAM_ID_WRITE_PARAMS, SYNC, SYNC_BYTES, U32_MASK,
                       GlobalHeader, ProtocolError, ResultFrame, StringReply, build_string_command,
                       header_checksum, parse_message, xor_checksum)

log = logging.getLogger('vc3dz_driver.client')


class SensorError(RuntimeError):
    """센서가 error != 0 으로 응답."""

    def __init__(self, reply: StringReply, what: str = ''):
        self.reply = reply
        super().__init__(f'{what} sensor error {reply.error} ({reply.error_name}) param_id={reply.param_id}: {reply.text[:120]!r}')


@dataclass
class ProfileRecord:
    """결과 프레임 1개 + 수신 메타."""
    result_counter: int        # RES.result_counter (시각 모델의 k 축)
    ts_ms: int                 # 센서 U64 ms since boot (무결성 검사 전용)
    t_rx_ns: int               # time.monotonic_ns() 수신 완료 직후
    x: np.ndarray              # float32 [mm]
    z: np.ndarray              # float32 [mm]
    n_points: int
    data_mode: int
    global_counter: int        # Global Header counter
    error: int                 # Global Header error
    checksum_ok: bool
    line_counter: Optional[int] = None

    @property
    def t_rx_s(self) -> float:
        return self.t_rx_ns * 1e-9


def _new_stats() -> Dict[str, int]:
    return dict(frames=0, string_replies=0, bytes_rx=0, counter_gaps=0, result_counter_gaps=0,
                error_nonzero=0, checksum_mismatch_hdr=0, checksum_mismatch_data=0, format_errors=0,
                resyncs=0, n_points_changes=0, queue_drops=0, unmatched_replies=0, unknown_cmd_type=0)


class VC3DZClient:
    """센서 TCP 클라이언트. `with VC3DZClient(ip) as c:` 또는 connect()/close()."""

    def __init__(self, ip: str = '192.168.3.15', port: int = 1096, wait_ms: int = DEFAULT_WAIT_MS,
                 recv_timeout: float = 5.0, quickack: Optional[bool] = None, rcvbuf: int = 4 * 1024 * 1024,
                 frame_queue_size: int = 5000, connect_timeout: float = 5.0):
        self.ip, self.port = ip, int(port)
        self.wait_ms = int(wait_ms)
        self.recv_timeout = float(recv_timeout)
        self.connect_timeout = float(connect_timeout)
        # Linux: 매 recv 후 TCP_QUICKACK (지연 ACK 스톨로 fps ~25 고착 방지). None = 속성 있으면 자동.
        self.quickack = (hasattr(socket, 'TCP_QUICKACK') if quickack is None else bool(quickack))
        self.rcvbuf = int(rcvbuf)
        self.sock: Optional[socket.socket] = None
        self.counter = 0
        self.host_cmd_id = 0
        self.n_points: Optional[int] = None
        self.frames: 'queue.Queue[ProfileRecord]' = queue.Queue(frame_queue_size)
        self.last_rx_ns: int = 0           # 마지막 결과 프레임 수신 시각 (monotonic_ns; 수신 스레드) — 소비 지연 진단용 (센서 가드 G3)
        self.replies: 'queue.Queue[StringReply]' = queue.Queue(200)
        self.stats: Dict[str, int] = _new_stats()
        self.last_error: Optional[str] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._running = False
        self._send_lock = threading.Lock()
        self._last_global_counter: Optional[int] = None
        self._last_result_counter: Optional[int] = None
        self.streaming = False
        self.param_backup_text: Optional[str] = None
        self.param_backup: Dict[str, P.Value] = {}
        self._clock = get_clock()          # 시계 이음매 (기본 REAL = 실시계 그 자체; 시험만 가상 시계)
        self._link = NULL_LINK

    # ------------------------------------------------------------------ 연결
    def connect(self) -> 'VC3DZClient':
        if self.sock is not None:
            return self
        self._link = self._clock.link_to(self.ip, self.port)   # 가상 시계는 루프백이 아니면 여기서 거부 (패킷 없음)
        s = socket.create_connection((self.ip, self.port), timeout=self.connect_timeout)
        self._link.attach(s)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
        except OSError:
            pass
        s.settimeout(self.recv_timeout)
        self.sock = s
        self.counter = 0            # 첫 명령 counter=0 = 상대 카운터 리셋
        self.host_cmd_id = 0
        self._last_global_counter = None
        self._last_result_counter = None
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, name='vc3dz-rx', daemon=True)
        self._rx_thread.start()
        log.info('connected %s:%d (quickack=%s)', self.ip, self.port, self.quickack)
        return self

    def close(self) -> None:
        self._running = False
        s, self.sock = self.sock, None
        if s is not None:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()
            self._link.close()
        t = self._rx_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._rx_thread = None

    def __enter__(self) -> 'VC3DZClient':
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        return self.sock is not None and self._running

    # ------------------------------------------------------------------ 송신
    def _send_raw(self, msg: bytes) -> None:
        if self.sock is None:
            raise ConnectionError('not connected')
        with self._send_lock:
            self._link.sent()
            self.sock.sendall(msg)

    def send_string(self, param_id: int, text: str = '') -> int:
        """STRING 명령 전송, host_cmd_id 반환. ID 33/45 는 ValueError."""
        if param_id in FORBIDDEN_PARAM_IDS:
            raise ValueError(f'param_id {param_id} (flash save) is forbidden — never sent by this driver')
        with self._send_lock:
            self.host_cmd_id = (self.host_cmd_id + 1) & U32_MASK or 1
            hid = self.host_cmd_id
            msg = build_string_command(param_id, text, self.counter, hid, self.wait_ms)
            self.counter = (self.counter + 1) & U32_MASK
            if self.sock is None:
                raise ConnectionError('not connected')
            self._link.sent()
            self.sock.sendall(msg)
        log.debug('TX param_id=%d hid=%d len=%d %r', param_id, hid, len(text), text[:60])
        return hid

    def wait_reply(self, host_cmd_id: int, param_id: Optional[int] = None, timeout: float = 2.0) -> StringReply:
        """host_cmd_id 가 일치하는 응답 대기. 불일치 응답은 보관; 타임아웃 시 같은 param_id 응답이 있으면 경고 후 반환."""
        deadline = self._clock.monotonic() + timeout
        candidate: Optional[StringReply] = None
        while True:
            remain = deadline - self._clock.monotonic()
            if remain <= 0:
                break
            try:
                r = self._clock.get(self.replies, remain)
            except queue.Empty:
                break
            if r.host_cmd_id == host_cmd_id:
                return r
            if param_id is not None and r.param_id == param_id:
                candidate = r
            self.stats['unmatched_replies'] += 1
        if candidate is not None:
            warnings.warn(f'reply host_cmd_id mismatch (want {host_cmd_id}, got {candidate.host_cmd_id}); using by param_id')
            return candidate
        raise TimeoutError(f'no reply for host_cmd_id={host_cmd_id} param_id={param_id} within {timeout}s')

    def command(self, param_id: int, text: str = '', timeout: float = 2.0, check: bool = True) -> StringReply:
        hid = self.send_string(param_id, text)
        r = self.wait_reply(hid, param_id, timeout)
        if check and r.error != 0:
            raise SensorError(r, f'param_id={param_id}')
        return r

    # ------------------------------------------------------------------ 고수준
    def read_params(self, timeout: float = 2.0) -> Tuple[str, Dict[str, P.Value]]:
        """ID 44 (str_len 0) → (원문, dict)."""
        r = self.command(PARAM_ID_READ_PARAMS, '', timeout)
        return r.text, P.parse_params_text(r.text)

    def set_params(self, kv: Dict[str, P.Value], timeout: float = 2.0, check: bool = True,
                   strict_whitelist: bool = False) -> StringReply:
        """ID 43 'Name = value\\n' 항목 일괄 전송. error≠0 이면 SensorError(check=True)."""
        unk = P.check_writable(kv, strict=strict_whitelist)
        if unk:
            warnings.warn(f'parameters not in whitelist (sent anyway): {unk}')
        return self.command(PARAM_ID_WRITE_PARAMS, P.format_params(kv), timeout, check)

    def set_params_individually(self, kv: Dict[str, P.Value], timeout: float = 2.0) -> Dict[str, StringReply]:
        """ERR_PARM 시 거부 항목 식별용: 항목별로 ID 43 전송, {name: reply}."""
        out: Dict[str, StringReply] = {}
        for k, v in kv.items():
            out[k] = self.set_params({k: v}, timeout, check=False)
        return out

    def start(self, timeout: float = 2.0) -> StringReply:
        """NbrLines = -1 → RESET 까지 연속 스트리밍."""
        r = self.set_params({'NbrLines': -1}, timeout, check=True)
        self.streaming = True
        return r

    def stop(self, timeout: float = 1.0) -> Optional[StringReply]:
        """ID 0 RESET. 응답이 없어도 예외 없이 None."""
        self.streaming = False
        try:
            hid = self.send_string(PARAM_ID_RESET, '')
        except (OSError, ConnectionError):
            return None
        try:
            return self.wait_reply(hid, PARAM_ID_RESET, timeout)
        except TimeoutError:
            log.warning('RESET: no STRING reply within %.1fs (stream stop assumed)', timeout)
            return None

    def safe_stop(self, timeout: float = 2.0) -> None:
        """§3.2: 안전 정지 상태로 시작 — ID 43 'DataMode = 0, NbrLines = 0'."""
        self.set_params(dict(P.SAFE_STOP), timeout, check=False)
        self.streaming = False

    def backup_params_to_file(self, path: Optional[str] = None, directory: str = '.', timeout: float = 2.0) -> str:
        """ID 44 전문을 backup_<YYYYmmdd-HHMMSS>.txt 로 저장, 경로 반환."""
        text, kv = self.read_params(timeout)
        self.param_backup_text, self.param_backup = text, kv
        if path is None:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, 'backup_' + _dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '.txt')
        with open(path, 'w', encoding='ascii', errors='replace', newline='\n') as f:
            f.write(text if text.endswith('\n') else text + '\n')
        log.info('parameter backup written: %s (%d items)', path, len(kv))
        return path

    def init_session(self, init: Optional[Dict[str, P.Value]] = None, backup_dir: Optional[str] = None,
                     timeout: float = 2.0) -> Dict[str, Tuple[P.Value, P.Value]]:
        """§3.2 시퀀스: 백업 → 안전정지 → 초기 파라미터 → ID 44 재읽기로 반영값 확인. 불일치 dict 반환."""
        if backup_dir is not None:
            self.backup_params_to_file(directory=backup_dir, timeout=timeout)
        else:
            self.param_backup_text, self.param_backup = self.read_params(timeout)
        self.safe_stop(timeout)
        init = dict(P.INIT_STANDARD if init is None else init)
        init.pop('NbrLines', None)
        try:
            self.set_params(init, timeout, check=True)
        except SensorError as e:
            if e.reply.error == -3003:
                per = self.set_params_individually(init, timeout)
                rejected = {k: r.error for k, r in per.items() if r.error != 0}
                warnings.warn(f'ERR_PARM: rejected items {rejected}')
            else:
                raise
        _, actual = self.read_params(timeout)
        return P.diff_params(init, actual)

    # ------------------------------------------------------------------ 프레임 소비
    def get_frame(self, timeout: Optional[float] = 1.0) -> ProfileRecord:
        return self._clock.get(self.frames, timeout)

    def iter_frames(self, timeout: float = 1.0) -> Iterator[ProfileRecord]:
        while self._running:
            try:
                yield self._clock.get(self.frames, timeout)
            except queue.Empty:
                continue

    def drain(self) -> List[ProfileRecord]:
        out: List[ProfileRecord] = []
        while True:
            try:
                out.append(self.frames.get_nowait())
            except queue.Empty:
                return out

    # ------------------------------------------------------------------ 수신 스레드
    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        s = self.sock
        while len(buf) < n:
            if s is None or not self._running:
                raise ConnectionError('closed')
            c = s.recv(n - len(buf))
            if not c:
                raise ConnectionError('peer closed')
            buf += c
            if self.quickack:
                try:
                    s.setsockopt(socket.IPPROTO_TCP, getattr(socket, 'TCP_QUICKACK', 12), 1)
                except OSError:
                    self.quickack = False
        self.stats['bytes_rx'] += n
        return bytes(buf)

    def _rx_loop(self) -> None:
        try:
            while self._running:
                try:
                    hdr = bytearray(self._recv_exact(GH_SIZE))
                except socket.timeout:
                    continue
                # sync 재동기(1 바이트 슬라이딩)
                slid = 0
                while bytes(hdr[:4]) != SYNC_BYTES:
                    del hdr[0]
                    hdr += self._recv_exact(1)
                    slid += 1
                if slid:
                    self.stats['resyncs'] += 1
                    log.warning('resync: skipped %d bytes', slid)
                h = GlobalHeader(*GH.unpack(bytes(hdr)))
                data = self._recv_exact(h.data_length) if h.data_length else b''
                t_rx_ns = self._clock.monotonic_ns()
                hdr_ok = h.cks_hdr == header_checksum(bytes(hdr))
                data_ok = h.cks_data == xor_checksum(data)
                if not hdr_ok:
                    self.stats['checksum_mismatch_hdr'] += 1
                if not data_ok:
                    self.stats['checksum_mismatch_data'] += 1
                if h.error != 0:
                    self.stats['error_nonzero'] += 1
                    log.warning('global header error=%d', h.error)
                if self._last_global_counter is not None and h.counter != 0 and \
                        ((self._last_global_counter + 1) & U32_MASK) != h.counter:
                    self.stats['counter_gaps'] += 1
                self._last_global_counter = h.counter
                try:
                    msg = parse_message(data, h.data_length)
                except ProtocolError as e:
                    self.stats['format_errors'] += 1
                    self.last_error = str(e)
                    log.warning('frame dropped: %s', str(e).splitlines()[0])
                    self._link.received()
                    continue
                if isinstance(msg, ResultFrame):
                    self._on_result(msg, h, t_rx_ns, hdr_ok and data_ok)
                elif isinstance(msg, StringReply):
                    self.stats['string_replies'] += 1
                    try:
                        self.replies.put_nowait(msg)
                    except queue.Full:
                        self.stats['queue_drops'] += 1
                else:
                    self.stats['unknown_cmd_type'] += 1
                self._link.received()                  # 메시지 하나를 다 처리한 뒤 (큐에 넣은 뒤) 정확히 한 번
        except (ConnectionError, OSError) as e:
            if self._running:
                self.last_error = f'rx loop ended: {e}'
                log.error('rx loop ended: %s', e)
        finally:
            self._running = False
            self._link.close()

    def _on_result(self, m: ResultFrame, h: GlobalHeader, t_rx_ns: int, ok: bool) -> None:
        if m.raw_payload is not None:      # 비지원 data_mode
            self.stats['unknown_cmd_type'] += 1
            return
        if self.n_points is None:
            self.n_points = m.n_points
            log.info('NumberOfPoints = %d', m.n_points)
        elif m.n_points != self.n_points:
            self.stats['n_points_changes'] += 1
            if self.stats['n_points_changes'] == 1:
                # 모드 0(유효점만)에서는 매 프레임 N 이 달라지므로 첫 번째만 경고, 이후는 카운트만
                warnings.warn(f'NumberOfPoints changed {self.n_points}->{m.n_points} '
                              f'(TransferAllPos_MM=0 stream, or ROI_DX_PIX/SubSample changed); further changes counted only')
            self.n_points = m.n_points
        if self._last_result_counter is not None and \
                ((self._last_result_counter + 1) & U32_MASK) != m.result_counter:
            self.stats['result_counter_gaps'] += 1
        self._last_result_counter = m.result_counter
        rec = ProfileRecord(m.result_counter, m.ts_ms, t_rx_ns, m.x, m.z, m.n_points, m.data_mode,
                            h.counter, h.error, ok, m.line_counter)
        self.stats['frames'] += 1
        self.last_rx_ns = int(t_rx_ns)
        try:
            self.frames.put_nowait(rec)
        except queue.Full:
            self.stats['queue_drops'] += 1
