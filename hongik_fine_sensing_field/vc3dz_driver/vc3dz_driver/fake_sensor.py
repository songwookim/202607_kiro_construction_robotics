"""센서 에뮬레이터(TCP 서버). 하드웨어 없이 클라이언트·노드 종단간 시험용.

동작:
- STRING 명령 수신(Global Header + 16 B meta + ASCII). 모든 명령에 STRING 응답(20 B meta) 을 돌려준다.
- ID 44 → 현재 파라미터 전문('Name = value;' 줄, VC3DPar.txt 형식). ID 43 → 파라미터 갱신(미지 이름은 ERR_PARM -3003,
  단 strict=False 면 수용). NbrLines = -1 → DataMode 4/5 프레임 스트리밍 시작, NbrLines = 0 / ID 0 RESET → 정지.
  ID 33/45 → ERR_COMMAND(-3004) (드라이버는 원래 보내지 않음).
- 스트리밍: AutoTriggerFPS 주기로 TXT 프레임을 순환 송신. ts_ms 는 1000/fps 씩 누적(정수 ms), result_counter·
  Global counter 는 1 씩 증가. `line_counter_variant=True` 면 8 B 서브헤더(extra==4) 로 보낸다.
- `corrupt_every=n` 이면 n 프레임마다 앞에 쓰레기 바이트를 끼워 재동기 경로를 시험한다.
- `frame_fn(i) -> (x_mm, z_mm)` 를 주면 고정 프레임 목록 대신 **연결별 프레임 인덱스 i**(0부터, 스트리밍 정지/재개에도
  계속 증가) 마다 그 함수가 만든 프로파일을 보낸다 (합성 세계: 현재 가짜 CB 자세로 렌더링 등). 카운터·타임스탬프·
  프로토콜은 고정 목록 경로와 완전히 같고, frame_fn 이 None 이면 기존 고정 목록 순환 그대로다.
- 실기 타이밍 재현 (auto-teach 리뷰 S1): `start_delay_s` = NbrLines -1 응답 뒤 첫 프레임까지 지연 (실기: 수백 ms),
  `reset_delay_s` = ID 0 RESET 에서 스트림은 즉시 멈추지만 STRING 응답은 이만큼 늦게 (실기: 최대 ~1 s; client.stop 은 1.0 s 대기).
  둘 다 기본 0 (기존 동작).
- `stall_late_ms` (기본 None = 기존 동작): 스트림 스레드가 예정 촬영 시각보다 이만큼 넘게 늦으면 실기 센서처럼 **촬영 멈춤**으로
  보낸다 — ts_ms 가 실제 촬영 (frame_fn 호출) 시각만큼 계단으로 뛰고 그 뒤 격자도 그만큼 밀린다 (따라잡기 몰아 보내기 없음; 카운터는
  1 씩 그대로). 기본 동작은 늦어도 ts 를 이상 격자에 찍고 밀린 프레임을 몰아 보내므로, 현재 자세로 렌더링하는 frame_fn 에서는
  CPU 부하 때 프레임 내용 (늦은 자세) 과 시각 (격자) 이 어긋난다 (호스트 stall_model 은 ts 로만 멈춤을 안다).
- `latency_ms` (기본 0 = 기존 동작: 렌더가 끝나는 대로 송신): 촬영 (frame_fn 호출) 에서 송신까지 고정 지연 (실기 센서의 처리 지연,
  호스트는 calib dt_s 로 보정). frame_fn 렌더 시간은 자세 캐시 적중 (정지 ≈ 0) 과 이동 (수십 ms, CPU 부하에 따라 더) 으로 갈려,
  지연이 그대로 송신에 실리면 호스트 카운터 시계의 수신 하한 포락선이 창마다 달라져 기울기가 틀어진다. 지연은 파이프라인이다
  (2026-10-11): 촬영은 주기대로 이어지고 각 프레임이 제 촬영 + 지연에 나간다 — 지연이 주기보다 커도 (150 fps, 40 ms) 프레임율은 그대로.
- `timed_frame_fn` (default False = as before: frame_fn(i) renders whatever pose it reads when it is called): frame_fn(i,
  t_capture_ns) gets the frame's CAPTURE instant on this sensor's clock — the scheduled capture of the stream grid (after a reported
  capture stall, the stalled one) — the instant its ts_ms stamps, its latency counts from (a frame leaves latency_ms after it), and the
  instant a real sensor exposes. A frame_fn that renders the scene at the pose of that instant (from a stamped pose history; real
  time: a late render still shows its capture instant) makes the stream's content and its stamps agree exactly, whatever the render
  time or the CPU load — with a pose read at the call, a real-clock stream shows the pose of the render (late by the render time,
  the pose source's sampling and its delivery: 2026-10-13 fake-arm e2e, 20–40 ms, varying per frame).

CLI: python -m vc3dz_driver.fake_sensor --port 1096 --txt profile.txt --fps 100
"""
from __future__ import annotations

import argparse
import collections
import logging
import socket
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import params as P
from .clock import NULL_LINK, get_clock
from .protocol import (CMD_TX, CMD_TX_SIZE, ERR_COMMAND, ERR_PARM, FORBIDDEN_PARAM_IDS, GH, GH_SIZE,
                       PARAM_ID_READ_PARAMS, PARAM_ID_RESET, PARAM_ID_WRITE_PARAMS, SYNC_BYTES, U32_MASK,
                       build_result_message, build_string_reply)

log = logging.getLogger('vc3dz_driver.fake_sensor')

DEFAULT_SENSOR_PARAMS: Dict[str, P.Value] = {
    'MainVersion': '1.9.3', 'LibVersion': '1.0.0', 'CameraSNR': 4090054,
    'DataMode': 0, 'TransferAllPos_MM': 0, 'NbrLines': 0, 'ShutterTime': 10, 'GainVal': 100,
    'LaserMode': 1, 'RlcThresh': 35, 'LaserSelect': 0, 'OptWidth': 8, 'MinWidth': 1, 'MaxWidth': 40,
    'SpeckleFilterDx': 5, 'SpeckleFilterDy': 3, 'LaserMaskFilter': 0, 'ReflexionFilter': 0,
    'ROI_X_PIX': 0, 'ROI_Y_PIX': 0, 'ROI_DX_PIX': 1408, 'ROI_DY_PIX': 1080, 'SubSample': 1,
    'TriggerMode': 0, 'TriggerDelayNs': 0, 'ExposureMode': 0, 'AutoShutterVal': 100, 'AutoShutterFilter': 0,
    'AutoShutterMin': 5, 'AutoShutterMax': 100, 'DoubleShutter1': 5, 'DoubleShutter2': 100,
    'EthernetPackNr': 1, 'AutoTriggerFPS': 50.0, 'AutoTriggerError': 0, 'EthernetSendNoWait': 1,
    'MedianFilter': 3, 'SensorLUT': 0,
}


def synthetic_frames(n_frames: int = 20, n_points: int = 1408, z0: float = 320.0, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """평판 + 작은 노이즈 합성 프로파일(모드 1 형태: N 열 전부, 양끝 몇 열은 NaN)."""
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_frames):
        dx = 0.082 + (z0 - 233.1) / (544.0 - 233.1) * (0.155 - 0.082)
        x = (np.arange(n_points) - n_points / 2) * dx
        z = z0 + 0.5 * np.sin(x / 40.0 + 0.01 * k) + rng.normal(0, 0.01, n_points)
        x = x.astype(np.float32); z = z.astype(np.float32)
        x[:3] = np.nan; z[:3] = np.nan          # 무효 열(NaN)
        z[-2:] = 0.0                            # 무효 열(z=0 센티널, x finite)
        frames.append((x, z))
    return frames


class FakeSensor:
    def __init__(self, host: str = '127.0.0.1', port: int = 0,
                 frames: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
                 params: Optional[Dict[str, P.Value]] = None, strict_params: bool = False,
                 line_counter_variant: bool = False, corrupt_every: int = 0, ts0_ms: int = 5_000_000,
                 realtime: bool = True,
                 frame_fn: Optional[Callable[[int], Tuple[np.ndarray, np.ndarray]]] = None,
                 start_delay_s: float = 0.0, reset_delay_s: float = 0.0, stall_late_ms: Optional[float] = None,
                 latency_ms: float = 0.0, timed_frame_fn: bool = False):
        self.host, self._port = host, int(port)
        self.frames = list(frames) if frames is not None else synthetic_frames()
        self.frame_fn = frame_fn
        self.timed_frame_fn = bool(timed_frame_fn)           # frame_fn(i, t_capture_ns) (module doc)
        self.params: Dict[str, P.Value] = dict(DEFAULT_SENSOR_PARAMS)
        if params:
            self.params.update(params)
        self.strict_params = strict_params
        self.line_counter_variant = line_counter_variant
        self.corrupt_every = int(corrupt_every)
        self.ts0_ms = int(ts0_ms)
        self.realtime = realtime
        self.start_delay_s = max(0.0, float(start_delay_s))     # NbrLines -1 → 첫 프레임 지연 (실기 수백 ms)
        self.reset_delay_s = max(0.0, float(reset_delay_s))     # RESET 응답 지연 (실기 최대 ~1 s; 스트림은 즉시 정지)
        self.stall_late_ms = None if stall_late_ms is None else max(0.0, float(stall_late_ms))   # 늦은 촬영 → 멈춤 보고 (모듈 문서)
        self.latency_ms = max(0.0, float(latency_ms))           # 촬영 → 송신 고정 지연 (모듈 문서; 0 = 렌더가 끝나는 대로)
        self._srv: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._clients: List[socket.socket] = []
        self.stats = dict(commands=0, frames_sent=0, connections=0, rejected=0, stalls=0, stall_ms=0.0)   # stalls: stall_late_ms 보고
        self.command_log: List[Tuple[int, int, str]] = []   # (param_id, host_cmd_id, text)
        self._clock = get_clock()                           # 시계 (기본 REAL; 가상 시계 시험에서는 촬영 박자·지연이 시뮬 시간)
        self._serve_threads: List[threading.Thread] = []
        self._states: list = []
        # 진단 (시험의 시간 불변식): 보낸 프레임마다 (연결, 스트림, result_counter, ts_ms, 촬영 ns, 송신 ns), 스트림마다 (연결, 스트림, 시작 ns,
        # 멈춤 ns 또는 None, fps)
        self.capture_log: List[Tuple[int, int, int, int, int, int]] = []
        self.stream_log: List[list] = []
        self.stream_listeners: list = []                    # fn('start', c0_ns, period_ns) / fn('stop') — 촬영 격자 (시험의 렌더 선계산)

    # ------------------------------------------------------------ lifecycle
    @property
    def port(self) -> int:
        return self._port

    def start(self) -> 'FakeSensor':
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self._port))
        srv.listen(2)
        srv.settimeout(0.2)
        self._port = srv.getsockname()[1]
        self._srv = srv
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, name='fake-sensor', daemon=True)
        self._thread.start()
        log.info('fake sensor listening on %s:%d', self.host, self._port)
        return self

    def stop(self) -> None:
        self._running = False
        with self._clock.acting():                 # 스트림을 먼저 멈춘다 (가상 시계: 스트림 스레드가 시뮬 시간으로 끝난다)
            for st in list(self._states):
                st.stop_stream()
        for c in list(self._clients):
            try:
                c.close()
            except OSError:
                pass
        if self._srv is not None:
            self._srv.close()
            self._srv = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for th in list(self._serve_threads):        # 연결 스레드도 끝까지 (스트림은 위에서 멈췄다 — 소켓 정리만)
            if th is not threading.current_thread():
                th.join(timeout=2.0)
        self._serve_threads = [t for t in self._serve_threads if t.is_alive()]

    def __enter__(self) -> 'FakeSensor':
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _accept_loop(self) -> None:
        assert self._srv is not None
        while self._running:
            try:
                conn, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.stats['connections'] += 1
            link = self._clock.link_to(*addr[:2])
            link.attach(conn)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._clients.append(conn)
            th = threading.Thread(target=self._serve, args=(conn, link, self.stats['connections']), daemon=True)
            self._serve_threads.append(th)
            th.start()

    # ------------------------------------------------------------ per-connection
    def _serve(self, conn: socket.socket, link=NULL_LINK, conn_no: int = 0) -> None:
        state = _ConnState(self, conn, link, conn_no)
        self._states.append(state)
        try:
            state.run()
        except (ConnectionError, OSError) as e:
            log.debug('client gone: %s', e)
        finally:
            with self._clock.acting():
                state.stop_stream()
            try:
                conn.close()
            except OSError:
                pass
            link.close()
            if conn in self._clients:
                self._clients.remove(conn)

    def frame_at(self, idx: int, t_capture_ns: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """스트리밍 프레임 idx(연결별 0부터)의 (x_mm, z_mm). frame_fn 이 있으면 그 결과(float32 로 캐스팅), 없으면 고정 목록 순환.
        timed_frame_fn: frame_fn(idx, t_capture_ns) — the frame's capture instant on this sensor's clock (module doc)."""
        if self.frame_fn is not None:
            x, z = self.frame_fn(int(idx), t_capture_ns) if self.timed_frame_fn else self.frame_fn(int(idx))
            x = np.asarray(x, dtype=np.float32).ravel()
            z = np.asarray(z, dtype=np.float32).ravel()
            if x.shape != z.shape:
                raise ValueError(f'frame_fn({idx}) returned x/z of different length: {x.shape} vs {z.shape}')
            return x, z
        return self.frames[idx % len(self.frames)]

    def text_params(self) -> str:
        lines = ['#', '#  nano3D_Z Sensor parameter', '#']
        for k, v in self.params.items():
            lines.append(f'{k:>20s} = {P.format_value(v)};')
        lines.append('# End of File')
        return '\n'.join(lines) + '\n'


class _ConnState:
    def __init__(self, sensor: FakeSensor, conn: socket.socket, link=NULL_LINK, conn_no: int = 0):
        self.s = sensor
        self.conn = conn
        self.link = link
        self.conn_no = int(conn_no)
        self.clock = sensor._clock
        self.n_streams = 0
        self.counter = 0            # 센서→호스트 Global counter
        self.result_counter = 0
        self.frame_idx = 0
        self.ts_acc = float(sensor.ts0_ms)
        self._stream_thread: Optional[threading.Thread] = None
        self._streaming = threading.Event()
        self._send_lock = threading.Lock()

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            c = self.conn.recv(n - len(buf))
            if not c:
                raise ConnectionError('closed')
            buf += c
        return bytes(buf)

    def _send(self, b: bytes) -> None:
        with self._send_lock:
            self.link.sent()
            self.conn.sendall(b)

    def run(self) -> None:
        while self.s._running:
            hdr = bytearray(self._recv_exact(GH_SIZE))
            while bytes(hdr[:4]) != SYNC_BYTES:
                del hdr[0]
                hdr += self._recv_exact(1)
            h = GH.unpack(bytes(hdr))
            data = self._recv_exact(h[7]) if h[7] else b''
            with self.link.receiving():                 # 받음 계수 + 처리하는 동안 시계 참여자 (RESET 의 join·지연이 시뮬 시간으로)
                self.handle_command(data)

    def handle_command(self, data: bytes) -> None:
        self.s.stats['commands'] += 1
        if len(data) < CMD_TX_SIZE:
            self._reply(0, 0, '', -3012)
            return
        cmd_type, pid, hid, slen = CMD_TX.unpack_from(data, 0)
        if cmd_type != 0:
            self._reply(pid, hid, '', -3011)
            return
        if CMD_TX_SIZE + slen != len(data):
            self._reply(pid, hid, '', -3010)
            return
        text = data[CMD_TX_SIZE:CMD_TX_SIZE + slen].decode('ascii', 'replace')
        self.s.command_log.append((pid, hid, text))
        if pid == PARAM_ID_RESET:
            self.stop_stream()
            self.s.params['NbrLines'] = 0
            if self.s.reset_delay_s > 0.0:
                self.clock.sleep(self.s.reset_delay_s)        # 실기: RESET 응답이 최대 ~1 s 늦다 (프레임은 이미 멈춤)
            self._reply(pid, hid, '', 0)
        elif pid == PARAM_ID_READ_PARAMS:
            self._reply(pid, hid, self.s.text_params(), 0)
        elif pid == PARAM_ID_WRITE_PARAMS:
            kv = P.parse_params_text(text)
            unk = P.unknown_params(kv)
            if unk and self.s.strict_params:
                self.s.stats['rejected'] += 1
                self._reply(pid, hid, '', ERR_PARM)
                return
            self.s.params.update(kv)
            self._reply(pid, hid, '', 0)
            nl = kv.get('NbrLines')
            if nl == -1:
                self.start_stream()
            elif nl == 0:
                self.stop_stream()
        elif pid in FORBIDDEN_PARAM_IDS:
            self.s.stats['rejected'] += 1
            self._reply(pid, hid, '', ERR_COMMAND)
        else:
            self._reply(pid, hid, '', 0)

    def _reply(self, pid: int, hid: int, text: str, err: int) -> None:
        self._send(build_string_reply(pid, hid, text, err, self.counter))
        self.counter = (self.counter + 1) & U32_MASK

    # ------------------------------------------------------------ streaming
    def start_stream(self) -> None:
        if self._streaming.is_set():
            return
        self._streaming.set()
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.clock.register(self._stream_thread).start()

    def stop_stream(self) -> None:
        self._streaming.clear()
        t = self._stream_thread
        if t is not None and t is not threading.current_thread():
            self.clock.join(t, 1.0)
        self._stream_thread = None

    def _stream_loop(self) -> None:
        """촬영 박자는 정수 ns 일정 (가상 시계에서 촬영 시각이 격자에 정확히 떨어진다; 실시계에서는 이전과 같은 규칙)."""
        clock = self.clock
        fps = float(self.s.params.get('AutoTriggerFPS', 50.0)) or 50.0
        mode = int(self.s.params.get('DataMode', 4))
        if mode not in (4, 5):
            mode = 4
        period_ns = int(round(1e9 / fps))
        self.n_streams += 1
        sno = self.n_streams
        slog = [self.conn_no, sno, clock.monotonic_ns(), None, fps]
        self.s.stream_log.append(slog)
        if self.s.start_delay_s > 0.0:
            # 실기: 시작 응답 뒤 첫 프레임까지 수백 ms — 그 사이 RESET 이 오면 한 프레임도 보내지 않는다
            t_end_ns = clock.monotonic_ns() + int(round(self.s.start_delay_s * 1e9))
            while self._streaming.is_set() and self.s._running and clock.monotonic_ns() < t_end_ns:
                clock.sleep(0.005)
        t_next_ns = clock.monotonic_ns()       # 다음 프레임의 예정 촬영 시각
        for fn in list(self.s.stream_listeners):
            fn('start', t_next_ns, period_ns)
        late_ns = None if self.s.stall_late_ms is None else int(round(self.s.stall_late_ms * 1e6))
        lat_ns = int(round(self.s.latency_ms * 1e6))
        pending = collections.deque()          # (send time, message, result counter, ts_ms, capture time) in processing
        k = 0
        try:
            while self._streaming.is_set() and self.s._running:
                if self.s.realtime and late_ns is not None:
                    late = clock.monotonic_ns() - t_next_ns
                    if late > late_ns:                 # 실기처럼 촬영 멈춤: ts 계단 = 실제 촬영 시각, 격자도 밀림 (따라잡지 않음)
                        self.ts_acc += late * 1e-6
                        t_next_ns += late
                        self.s.stats['stalls'] += 1
                        self.s.stats['stall_ms'] += late * 1e-6
                t_exp_ns = clock.monotonic_ns()        # 촬영 (frame_fn 이 자세를 읽는) 시각
                if self.s.timed_frame_fn and self.s.realtime:
                    # the capture instant = the stream grid's (after a reported stall, the stalled one): what ts_ms stamps — the
                    # frame shows that instant however late this thread runs (module doc, timed_frame_fn)
                    t_exp_ns = t_next_ns
                try:
                    x, z = self.s.frame_at(self.frame_idx, t_exp_ns)
                except Exception:                      # frame_fn 오류: 스트림만 멈추고 연결/명령 처리는 유지
                    log.exception('frame_fn failed at frame %d; stopping stream', self.frame_idx)
                    break
                self.frame_idx += 1
                lc = self.result_counter if self.s.line_counter_variant else None
                ts_ms = int(self.ts_acc)
                msg = build_result_message(x, z, self.result_counter, ts_ms, self.counter, mode, lc)
                k += 1
                if self.s.corrupt_every and k % self.s.corrupt_every == 0:
                    msg = b'\xde\xad\xbe\xef\x00' + msg
                if self.s.realtime and lat_ns > 0:
                    # the sensor's processing latency is a pipeline (2026-10-11): the frame leaves lat after its capture while the
                    # next captures go on at the frame rate — a blocking wait here capped the stream at 1/lat (25 fps at 40 ms)
                    pending.append((t_exp_ns + lat_ns, msg, self.result_counter, ts_ms, t_exp_ns))
                else:
                    self._send(msg)
                    self.s.capture_log.append((self.conn_no, sno, self.result_counter, ts_ms, t_exp_ns, clock.monotonic_ns()))
                    self.s.stats['frames_sent'] += 1
                self.result_counter = (self.result_counter + 1) & U32_MASK
                self.counter = (self.counter + 1) & U32_MASK
                self.ts_acc += 1000.0 / fps
                if self.s.realtime:
                    t_next_ns += period_ns
                    while True:                        # send what is due, sleep to the next send or the next capture
                        now = clock.monotonic_ns()
                        while pending and pending[0][0] <= now:
                            _due, m_, rc_, ts_, te_ = pending.popleft()
                            self._send(m_)
                            self.s.capture_log.append((self.conn_no, sno, rc_, ts_, te_, clock.monotonic_ns()))
                            self.s.stats['frames_sent'] += 1
                            now = clock.monotonic_ns()
                        if not self._streaming.is_set():
                            break
                        wake = min(t_next_ns, pending[0][0]) if pending else t_next_ns
                        dt = wake - now
                        if dt > 0:
                            clock.sleep(dt * 1e-9)
                        elif dt < -1_000_000_000:
                            t_now_ns = clock.monotonic_ns()
                            if self.s.timed_frame_fn:          # the grid jumps: a capture stall in the stamps too (content = stamps)
                                self.ts_acc += (t_now_ns - t_next_ns) * 1e-6
                            t_next_ns = t_now_ns
                        if clock.monotonic_ns() >= t_next_ns:
                            break
                    if not self._streaming.is_set():
                        break                          # RESET: the stream stops at once (frames still in processing are not sent)
        except (ConnectionError, OSError):
            pass
        finally:
            self._streaming.clear()
            slog[3] = clock.monotonic_ns()
            for fn in list(self.s.stream_listeners):
                fn('stop')


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='VC nano 3D-Z fake sensor (TCP server)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=1096)
    ap.add_argument('--txt', help='SmartShape profile_mm_*.txt to replay (default: synthetic plate)')
    ap.add_argument('--max-frames', type=int, default=200)
    ap.add_argument('--fps', type=float, default=None, help='override AutoTriggerFPS default')
    ap.add_argument('--line-counter', action='store_true', help='send 8 B sub header variant (extra==4)')
    ap.add_argument('--corrupt-every', type=int, default=0)
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    frames = None
    if a.txt:
        from .replay import load_smartshape_txt
        frames = [(f.x, f.z) for f in load_smartshape_txt(a.txt, max_frames=a.max_frames)]
    params = {'AutoTriggerFPS': a.fps} if a.fps else None
    with FakeSensor(a.host, a.port, frames, params, line_counter_variant=a.line_counter,
                    corrupt_every=a.corrupt_every) as fs:
        print(f'fake sensor on {fs.host}:{fs.port}  frames={len(fs.frames)}  Ctrl-C to stop')
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
