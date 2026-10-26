"""센서 세션 (연결·녹화) 과 세션 쌍 (ScanSession = 로봇 플랫폼 + 센서).

SensorSession: VC3DZClient 래핑. init_session(INIT_STANDARD) → start() 후 백그라운드 스레드가 프레임을 리스트로 모은다.
              파라미터 백업(ID 44) 은 세션의 **첫** init 에서 한 번만 params_original 로 잡고 (두 번째 init 이 INIT_STANDARD
              상태를 '원본' 으로 덮어쓰지 않게), restore_backup 은 그 원본을 되돌린다 (리뷰 S6). 원본이 이미 INIT 값과 같으면
              직전 세션이 복원하지 않은 것 → 경고 (파일 백업에서 복원하라는 명령 포함).
              실기 타이밍 (리뷰 S1): 시작 뒤 첫 프레임까지 수백 ms, RESET 응답은 최대 ~1 s — start_recording 뒤
              wait_first_frame() 으로 첫 프레임을 확인한다.
ScanSession: 열려 있는 RobotPlatform (MoveIt 어댑터 hongik_scan_ros.moveit_adapter.MoveItPlatform; 시험은 가상 팔) + SensorSession.
2026-10-07 (사용자): RB 제어박스 어댑터 (CB 5000/5001 직접 세션) 를 없앴다 — 모든 이동은 MoveIt 을 거친다.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from vc3dz_driver import params as VP
from vc3dz_driver.clock import get_clock
from vc3dz_driver.client import ProfileRecord, VC3DZClient

from .ports import RobotPlatform

# 원본 파라미터가 이미 INIT 값이면 (직전 세션이 복원하지 않았다) 첫 사용 전에 만든 파일 백업에서 되돌린다 (리뷰 S6)
SENSOR_FILE_BACKUP_HINT = "<the sensor's original parameters: backup_<date>.txt made with 'vc3dz --ip <sensor ip> backup' before the first use>"


class SensorNotStarted(RuntimeError):
    """센서 시작(NbrLines −1) 뒤 first_frame_timeout_s 안에 프레임이 오지 않았다 — 구간을 보내기 전에 거부 (리뷰 S1)."""


# --------------------------------------------------------------------------------------
# 5001 상태 → 포즈 / 모드 검사
# --------------------------------------------------------------------------------------


def sensor_restore_text(rr: Optional[dict]) -> str:
    """센서 파라미터 복원 결과 한 줄 (운영자용, S6)."""
    if rr is None:
        return "sensor parameters: not restored (no init/backup in this session)"
    if rr.get("restored"):
        return f"sensor parameters restored to the session's original values ({rr.get('n')} writable items verified)"
    return (f"WARNING: sensor parameter restore incomplete: error={rr.get('error')} mismatch={rr.get('mismatch')} - restore from the "
            f"file backup: python -m vc3dz_driver.cli --ip <sensor ip> restore --file {SENSOR_FILE_BACKUP_HINT}")


# --------------------------------------------------------------------------------------
# Robot session
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Sensor session
# --------------------------------------------------------------------------------------
class _Recorder(threading.Thread):
    def __init__(self, client: VC3DZClient, sink: list, tap: Optional[Callable[[ProfileRecord], None]] = None):
        super().__init__(name="vc3dz-recorder", daemon=True)
        self.client = client
        self.sink = sink
        self.tap = tap                      # 프레임을 소비하지 않고 구경만 하는 콜백 (스트림 작업·서버 ~/profile)
        self._stop_evt = threading.Event()  # NOT `_stop`: threading.Thread._stop() is a private method used by join() on Python 3.10-3.12

    def stop(self) -> None:
        self._stop_evt.set()

    def _push(self, rec: ProfileRecord) -> None:
        self.sink.append(rec)
        if self.tap is not None:
            self.tap(rec)

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._push(self.client.get_frame(timeout=0.1))
            except queue.Empty:
                continue
            except Exception:  # noqa: BLE001 - connection closed etc.: leave the loop to the stop path
                if not self.client.connected:
                    break
        for rec in self.client.drain():
            self._push(rec)


@dataclass
class SensorInfo:
    params_before_text: str = ""
    params_before: dict = field(default_factory=dict)
    params_after_text: str = ""
    params_after: dict = field(default_factory=dict)
    init_mismatch: dict = field(default_factory=dict)
    init_params: dict = field(default_factory=dict)
    backup_path: Optional[str] = None

    def as_dict(self) -> dict:
        return {"params_before": self.params_before, "params_after": self.params_after,
                "init_mismatch": {k: list(v) for k, v in self.init_mismatch.items()}, "init_params": self.init_params,
                "backup_path": self.backup_path, "params_before_text": self.params_before_text,
                "params_after_text": self.params_after_text}


class SensorUnreachable(ConnectionError):
    """센서에 TCP 로 접속하지 못했다 (2026-10-05 실기: 센서 전원이 꺼져 'TimeoutError: timed out' 만 나와 원인을 알기 어려웠다)."""


def sensor_unreachable_message(ip: str, port: int, err: BaseException) -> str:
    """접속 실패 → 원인별 안내 한 줄."""
    if isinstance(err, ConnectionRefusedError):
        why = ("connection refused: the sensor answers but its port is not open "
               "(sensor still booting, or another program such as SmartShape holds it)")
    elif isinstance(err, (TimeoutError, OSError)) and ("timed out" in str(err) or isinstance(err, TimeoutError)):
        why = "no answer (timed out): sensor powered off, or the cable / PC sensor network port is down"
    else:
        why = f"{type(err).__name__}: {err}"
    return (f"sensor {ip}:{port} not reachable - {why}. Check the sensor power and its cable to the PC sensor port "
            f"(ip -4 -br addr: the PC needs an address on the sensor subnet), then: ping {ip}")


class SensorSession:
    """VC nano 3D-Z 클라이언트 + 녹화. `with SensorSession(ip, port) as ss: ss.init(); ss.start_recording(); ...`"""

    def __init__(self, ip: str, port: int = 1096, recv_timeout: float = 3.0, restore_on_close: bool = False):
        self.ip, self.port = ip, int(port)
        self.recv_timeout = float(recv_timeout)
        self.client: Optional[VC3DZClient] = None
        self.info = SensorInfo()
        self.records: list = []
        self._rec: Optional[_Recorder] = None
        self.t_start_ns: int = 0
        self.t_stop_ns: int = 0
        self._taps: list = []               # 녹화/grab 프레임 구독자 (소비하지 않음)
        self._tap_lock = threading.Lock()
        self.backup_taken = False           # init() 이 ID 44 백업을 찍었다 (restore_backup 가능)
        self.restore_result: Optional[dict] = None
        self.params_original: Optional[dict] = None       # 세션 첫 init 의 ID 44 백업 (이후 init 은 덮어쓰지 않는다, S6)
        self.params_original_text: str = ""
        self.warnings: list = []                          # 운영자에게 보여 줄 경고 (예: 직전 세션 미복원)
        self.restore_on_close = bool(restore_on_close)    # close() 가 원본을 되돌린다 (teach/localize 컨텍스트)
        self._first_frame = threading.Event()             # start_recording/grab 뒤 첫 프레임 (S1)
        self.stream_no = 0                                # 스트림 (녹화·grab) 번호 — 시작마다 +1 (센서 가드의 촬영 시각 대응은 스트림마다 새로, S2)
        self._clock = get_clock()                         # 시계 이음매 (기본 REAL = 실시계 그 자체; 시험만 가상 시계)

    # ---- record taps (subscribers that see every streamed record without consuming it) ----
    def add_tap(self, fn: Callable[[ProfileRecord], None]) -> None:
        """녹화(start_recording)·grab 중 수신되는 **모든** 프레임을 fn(rec) 으로도 전달한다. 프레임은 그대로 records 에 남는다."""
        with self._tap_lock:
            if fn not in self._taps:
                self._taps.append(fn)

    def remove_tap(self, fn: Callable[[ProfileRecord], None]) -> None:
        with self._tap_lock:
            if fn in self._taps:
                self._taps.remove(fn)

    def _notify_taps(self, rec: ProfileRecord) -> None:
        self._first_frame.set()
        with self._tap_lock:
            taps = list(self._taps)
        for fn in taps:
            try:
                fn(rec)
            except Exception:  # noqa: BLE001 - a subscriber must never break recording
                pass

    @property
    def recording(self) -> bool:
        return self._rec is not None

    def open(self) -> "SensorSession":
        if not self.ip:
            raise ValueError("sensor ip is empty (set sensor.ip in the config or pass --sensor-ip)")
        try:
            self.client = VC3DZClient(self.ip, self.port, recv_timeout=self.recv_timeout).connect()
        except OSError as e:                              # TimeoutError·ConnectionRefusedError 포함
            raise SensorUnreachable(sensor_unreachable_message(self.ip, self.port, e)) from e
        return self

    def close(self) -> None:
        if self.client is not None:
            try:
                if self._rec is not None or self.client.streaming:
                    self.stop_recording()
            except Exception:  # noqa: BLE001
                pass
            if self.restore_on_close and self.backup_taken and self.restore_result is None:
                self.restore_result = self.restore_backup()
            self.client.close()
            self.client = None

    def __enter__(self) -> "SensorSession":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- parameters ----
    def read_params(self) -> tuple:
        assert self.client is not None
        return self.client.read_params()

    def init(self, init_params: Optional[dict] = None, backup_dir: Optional[str] = None) -> dict:
        """§3.2: ID 44 백업 → 안전정지 → INIT_STANDARD(또는 지정) → ID 44 재읽기. 불일치 dict 반환.

        원본 백업은 세션의 첫 init 에서만 잡는다 (params_original; 파일 백업도 그때만) — 두 번째 init(공유 세션의 sensor_init
        재적용 등) 의 ID 44 는 이미 INIT 상태라 '원본' 이 아니다 (리뷰 S6). 원본이 INIT 값과 같으면 경고를 남긴다."""
        assert self.client is not None
        init = dict(VP.INIT_STANDARD if init_params is None else init_params)
        first = self.params_original is None
        mism = self.client.init_session(init, backup_dir=backup_dir if first else None)
        if first:
            self.params_original = dict(self.client.param_backup)
            self.params_original_text = self.client.param_backup_text or ""
            same = {k: v for k, v in init.items() if k != "NbrLines"}
            if same and self.params_original and not VP.diff_params(same, self.params_original):
                self.warnings.append(
                    "sensor parameters before this session already equal the INIT values on every INIT key: the previous "
                    "session did not restore; restore from the file backup: python -m vc3dz_driver.cli --ip "
                    f"{self.ip} restore --file {SENSOR_FILE_BACKUP_HINT}")
        self.info.params_before_text = self.params_original_text
        self.info.params_before = dict(self.params_original or {})
        text, kv = self.client.read_params()
        self.info.params_after_text, self.info.params_after = text, kv
        self.info.init_mismatch = dict(mism)
        self.info.init_params = init
        self.backup_taken = True
        return mism

    def restore_backup(self, timeout: float = 2.0) -> dict:
        """init() 이 찍은 ID 44 백업의 쓰기 가능 항목을 ID 43 으로 되돌리고 재읽기로 검증 (리뷰 M-12: 세션이 센서를
        DataMode 4 / 표준 게인 / 고정 노출로 남겨 두지 않도록). 반환 {restored, n, mismatch, error}."""
        out: dict = {"restored": False, "n": 0, "mismatch": {}, "error": None,
                     "source": "params_original (first init of this session)"}
        c = self.client
        orig = self.params_original if self.params_original is not None else (dict(c.param_backup) if c is not None else {})
        if c is None or not getattr(self, "backup_taken", False) or not orig:
            out["error"] = "no backup"
            return out
        try:
            if self._rec is not None or c.streaming:
                self.stop_recording(settle_s=0.1)
            _vals, meta = VP.parse_params_text_with_meta(self.params_original_text or c.param_backup_text or "")
            restore = VP.writable_params(dict(orig), meta or None)
            out["n"] = len(restore)
            if not restore:
                out["error"] = "backup has no writable parameters"
                return out
            c.set_params(restore, timeout, check=False)
            _text, after = c.read_params(timeout)
            out["mismatch"] = {k: list(v) for k, v in VP.diff_params(restore, after).items()}
            out["restored"] = not out["mismatch"]
            self.info.params_after_text, self.info.params_after = _text, after
        except Exception as e:  # noqa: BLE001 - restoring must never raise at session close
            out["error"] = str(e)
        return out

    # ---- recording ----
    def start_recording(self) -> int:
        assert self.client is not None
        self.records = []
        self.client.drain()
        self._first_frame.clear()
        self._rec = _Recorder(self.client, self.records, tap=self._notify_taps)
        self._clock.register(self._rec).start()
        self.t_start_ns = self._clock.monotonic_ns()
        self.stream_no += 1
        self.client.start()
        return self.t_start_ns

    def wait_first_frame(self, timeout_s: float) -> bool:
        """start_recording 뒤 첫 프레임이 timeout_s 안에 왔는가 (실기: 시작 뒤 첫 프레임까지 수백 ms, S1)."""
        return self._clock.wait(self._first_frame, max(0.0, float(timeout_s)))

    def stop_recording(self, settle_s: float = 0.3) -> list:
        """센서 RESET(ID 0) 후 settle 동안 남은 프레임을 받고 녹화 스레드를 멈춘다."""
        assert self.client is not None
        self.client.stop()
        self.t_stop_ns = self._clock.monotonic_ns()
        self._clock.sleep(max(0.0, settle_s))
        if self._rec is not None:
            self._rec.stop()
            self._clock.join(self._rec, 2.0)
            self._rec = None
        return self.records

    def grab(self, n_frames: int = 20, timeout_s: float = 5.0, *, on_first_frame: Optional[Callable[[], None]] = None,
             pre_stop: Optional[Callable[[], None]] = None) -> list:
        """짧은 스트리밍으로 n 프레임 수집 (티칭·상태 확인·드웰).

        on_first_frame(): 첫 프레임을 받은 **뒤** (탭에 넘기기 전) 한 번 (선택 콜백, S1).
        pre_stop(): RESET(client.stop, 응답 최대 ~1 s) **전에** 한 번 (선택 콜백)."""
        assert self.client is not None
        self.client.drain()
        self._first_frame.clear()
        self.stream_no += 1
        self.client.start()
        out: list = []
        deadline = self._clock.monotonic() + timeout_s
        try:
            while len(out) < n_frames and self._clock.monotonic() < deadline:
                try:
                    rec = self.client.get_frame(timeout=0.5)
                except queue.Empty:
                    continue
                if not out and on_first_frame is not None:
                    on_first_frame()
                out.append(rec)
                self._notify_taps(rec)
        finally:
            if pre_stop is not None:
                try:
                    pre_stop()
                except Exception:  # noqa: BLE001 - never skip the RESET
                    pass
            self.client.stop()
        return out

    @property
    def stats(self) -> dict:
        return dict(self.client.stats) if self.client is not None else {}


# --------------------------------------------------------------------------------------
# Shared session pair (robot platform + sensor) — the auto-teach orchestrator reuses one open pair
# --------------------------------------------------------------------------------------
@dataclass
class ScanSession:
    """열려 있는 RobotPlatform (MoveIt 어댑터 hongik_scan_ros.moveit_adapter.MoveItPlatform; 시험은 가상 팔) + (선택) SensorSession 쌍.
    AutoTeach(session=...) 으로 넘기면 재접속 없이 재사용한다. owned=True 면 끝에서 닫는다. sensor_ready: SensorSession.init() 이
    끝났음 (공유 세션의 두 번째 실행은 sensor_init 을 안 주면 재초기화하지 않는다)."""

    robot: RobotPlatform
    sensor: Optional[SensorSession] = None
    owned: bool = False
    sensor_ready: bool = False
    restore_sensor_params: bool = True      # close(): init() 이 찍은 센서 파라미터 백업을 되돌린다 (M-12)

    @property
    def armed(self) -> bool:
        return self.robot.armed

    def restore_sensor(self) -> Optional[dict]:
        """센서 파라미터 원본 복원 (한 번만; 이미 했으면 그 결과). 출력(summary.json / autoteach_log.json) 을 쓰기 **전에**
        불러 결과를 거기에 담는다 (리뷰 S6). init 이 없었으면 None."""
        ss = self.sensor
        if ss is None or not self.restore_sensor_params or not ss.backup_taken:
            return None
        if ss.restore_result is None:
            try:
                ss.restore_result = ss.restore_backup()
            except Exception as e:  # noqa: BLE001
                ss.restore_result = {"restored": False, "error": str(e)}
        return ss.restore_result

    def close(self) -> None:
        if self.sensor is not None:
            self.restore_sensor()
            try:
                self.sensor.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.robot.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "ScanSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def records_to_arrays(recs: list) -> dict:
    """ProfileRecord 목록 → raw npz 배열 (sensor_* 접두)."""
    n = np.array([r.n_points for r in recs], dtype=np.int64) if recs else np.zeros(0, np.int64)
    nmax = int(n.max()) if n.size else 0
    X = np.full((len(recs), nmax), np.nan, np.float32)
    Z = np.full((len(recs), nmax), np.nan, np.float32)
    for i, r in enumerate(recs):
        X[i, : r.n_points] = r.x
        Z[i, : r.n_points] = r.z
    return {
        "sensor_x_mm": X, "sensor_z_mm": Z, "sensor_n_points": n,
        "sensor_ts_ms": np.array([r.ts_ms for r in recs], np.int64),
        "sensor_result_counter": np.array([r.result_counter for r in recs], np.int64),
        "sensor_t_rx_ns": np.array([r.t_rx_ns for r in recs], np.int64),
        "sensor_global_counter": np.array([r.global_counter for r in recs], np.int64),
        "sensor_error": np.array([r.error for r in recs], np.int32),
        "sensor_checksum_ok": np.array([r.checksum_ok for r in recs], bool),
    }

