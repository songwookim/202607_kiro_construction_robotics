"""센서 파라미터: 초기값 세트(설계 §3.3), 알려진 이름 화이트리스트, 'Name = value' 텍스트 조립/파싱.

출처 태그
  [M188] Laser Scanner Manual 1.8.8 복원본 (appendix_sensor_protocol/laser_scanner_manual_188_decoded_v2.txt)
  [PB]   Press brake manual VC3DPar.txt 예시 (press_brake_manual.txt §10)
  [EXE]  3DSmartShape_V52.exe 문자열 (연구실 펌웨어 1.9.x 에서만 존재하는 항목 포함)
  [I]    추정 — 1차 접속에서 ID 44 응답으로 확인할 것
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple, Union

Value = Union[int, float, str]

# ---------------------------------------------------------------------------
# 초기값 세트 (설계 §3.3). NbrLines 는 포함하지 않는다 — start()/stop() 이 따로 보낸다.
# 백업(ID 44)값을 유지할 항목(MedianFilter, SpeckleFilter*, Min/Max/OptWidth, RlcThresh, ReflexionFilter,
# LaserSelect, SubSample)은 여기에 넣지 않는다.
# ---------------------------------------------------------------------------
INIT_STANDARD: Dict[str, Value] = {
    'DataMode': 4,              # mm (x,z) 쌍 — 유일한 형상 모드
    'TransferAllPos_MM': 1,     # 전열 전송(고정 열 인덱스). 거부(ERR_PARM) 시 0 폴백(§3.7)
    'EthernetPackNr': 1,        # 매 스캔 즉시 송신
    'EthernetSendNoWait': 1,
    'TriggerMode': -1,          # 내부 자동트리거
    'AutoTriggerFPS': 100.0,    # 풀프레임 한계 174 fps 의 57 %
    'AutoTriggerError': 0,      # 리셋; >0 이면 과속
    # 노출: 고정(0). 2026-09-30 실기(펌웨어 1.9.7, 흑피 강판 @ ~430 mm)에서 자동노출(1)은 첫 프레임만 전체 유효 후
    # 셔터를 AutoShutterMin 쪽으로 줄여 신호를 잃고 복구하지 못했다. 고정 100 µs(레이저 펄스 상한) + GainVal 300 에서
    # 1408/1408 유효, 열별 시간 σ 0.016 mm.
    # 게인 250 (2026-10-01 실기, 그루브 위 거의 수직 입사 @ ~409 mm, 각 500 프레임): 300 은 판 앞 z 314–349 mm 에 유령점
    # (프레임의 ~33 %, 열의 최대 13 %, 버스트 ≤ 2 프레임) → 센서 가드 G4 거짓 정지. 250: 유령 0, 유효 99.8 %, 베벨면 보임
    # (32.5° vs 설계 30°). 200: 유령 0, 유효 100 % 이나 베벨면 소실. 100: 97 %. (09-30 @ ~430 mm: 200 → 85 %.)
    # 자세·거리에 따라 다시 볼 것 — teach-live 가 유효율·유령 수를 실시간으로 보여 주고 --sensor-param 으로 바꿔 볼 수 있다.
    'ExposureMode': 0,
    'ShutterTime': 100,         # µs (레이저 펄스 ≤100 µs, 초과 시 레이저 OFF)
    'GainVal': 250,             # 0~480. 300 → 유령점 (2026-10-01), 250 → 유령 0·유효 99.8 %, 200 → 베벨면 소실 (위 주석)
    'AutoShutterMin': 5,        # (자동노출 모드용 참고값)
    'AutoShutterMax': 100,      # µs (초과 시 레이저 OFF)
    'ROI_X_PIX': 0,
    'ROI_Y_PIX': 0,
    'ROI_DX_PIX': 1408,         # N(점 수) 결정
    'ROI_DY_PIX': 1080,         # 전체 Z 범위(233–544 mm)
    'LaserMode': 1,             # 레이저 ON
    'TriggerDelayNs': 0,
}

# 사전스캔(§5.2 prescan 행): 스탠드오프 470 mm, 100 mm/s, Δs 1 mm. 센서 파라미터는 표준과 동일
# (fps 100, 풀 ROI). 스탠드오프가 크므로 노출 상한만 여유(자동노출이 알아서 조정, 상한 100 µs 유지).
INIT_PRESCAN: Dict[str, Value] = dict(INIT_STANDARD)

# HW 트리거(P4): 펄스당 1 프레임 = +1, TriggerDelayNs 적용, result_counter = 펄스 번호.
# 첫 에지 후 AutoTriggerFPS 연속(-2)은 현장에서 SmartShape 로 값을 읽어 확정 후 사용(§3.3).
INIT_HW_TRIGGER: Dict[str, Value] = dict(INIT_STANDARD, TriggerMode=1, TriggerDelayNs=0)

# 자동노출 실험용(비권장, 실기에서 신호 소실 관측): 필요 시 AutoGainMax 를 300 이상으로 두고 시험
INIT_AUTO_EXPOSURE: Dict[str, Value] = dict(INIT_STANDARD, ExposureMode=1, AutoShutterMin=20, AutoShutterMax=100,
                                            AutoGainMin=100, AutoGainMax=400)

INIT_SETS = {'standard': INIT_STANDARD, 'prescan': INIT_PRESCAN, 'hw_trigger': INIT_HW_TRIGGER,
             'auto_exposure': INIT_AUTO_EXPOSURE}

# 안전 정지 상태로 시작(§3.2 두 번째 명령)
SAFE_STOP: Dict[str, Value] = {'DataMode': 0, 'NbrLines': 0}

# ---------------------------------------------------------------------------
# 화이트리스트: name -> (type, description, source)
# ---------------------------------------------------------------------------
KNOWN_PARAMS: Dict[str, Tuple[type, str, str]] = {
    # 읽기 전용/정보
    'MainVersion': (str, 'server program version (first item of the ID 44 reply; >= 1.9.3 is recorded)', 'M188'),
    'Mainversion': (str, 'MainVersion spelling variant', 'I'),
    'LibVersion': (str, 'libvclinux version', 'EXE'),
    'CameraSNR': (int, 'camera serial number', 'M188/EXE'),
    'ProductVersion': (int, 'product parameter version', 'PB'),
    # 동작 모드
    'DataMode': (int, '-1 no transfer / 0 standby / 1 image / 2,3 pixels / 4 (x,z) mm pairs / 5 x[],z[] / 6 binary / 8 product / 9 4+8', 'M188'),
    'NbrLines': (int, '-1 endless (until RESET) / 0 stop / N times / -2 follow the other port', 'M188'),
    'TransferAllPos_MM': (int, '0 valid points only / 1 all columns (invalid = x and z NaN; confirmed on firmware 1.9.7, 2026-09-30)', 'M188/LAB197'),
    'TriggerMode': (int, '0 free-run / +1 external HW trigger / -1 automatic (AutoTriggerFPS) / -2 automatic after an external edge -- firmware 1.9.7 range Min=-2 Max=1 (no falling/PULSE values)', 'M188/EXE/LAB197'),
    'TriggerDelayNs': (int, 'delay after the HW trigger [ns]', 'M188'),
    'AutoTriggerFPS': (float, 'auto trigger fps (10-400; > 174 needs a smaller ROI)', 'M188'),
    'AutoTriggerError': (int, 'scans over the jitter limit (> 0 = too fast; reset with 0)', 'M188'),
    'EthernetPackNr': (int, '0 batch after memory / 1 immediately (recommended) / n bundles', 'M188'),
    'EthernetSendNoWait': (int, '0 wait for the transfer / 1 no wait (recommended)', 'M188'),
    # 노출
    'ExposureMode': (int, '0 fixed (ShutterTime) / 1 automatic / 2 double / 3 triple (assumed)', 'M188'),
    'ShutterTime': (int, 'fixed exposure [us] (> 100 us laser OFF)', 'M188'),
    'GainVal': (int, 'gain', 'M188'),
    'AutoShutterVal': (int, 'auto exposure target brightness', 'M188'),
    'AutoShutterFilter': (int, 'auto exposure brightness statistic 0..6', 'M188'),
    'AutoShutterMin': (int, 'auto exposure minimum [us]', 'M188'),
    'AutoShutterMax': (int, 'auto exposure maximum [us]', 'M188'),
    'AutoGainMin': (int, 'auto gain minimum', 'PB'),
    'AutoGainMax': (int, 'auto gain maximum', 'PB'),
    'DoubleShutter1': (int, 'double exposure 1 [us]', 'M188'),
    'DoubleShutter2': (int, 'double exposure 2 [us]', 'M188'),
    'TripleShutter3': (int, 'triple exposure 3 [us]', 'PB'),
    # 레이저/라인 검출
    'LaserMode': (int, '1 laser ON / 0 OFF', 'M188'),
    'LaserSelect': (int, '0 nearest OptWidth / 1 by ReflexionFilter / 2 brightest', 'M188'),
    'ReflexionFilter': (int, '0 none / > 0 n-th line from the top / < 0 n-th line from the bottom', 'M188'),
    'OptWidth': (int, 'optimal line width [px]', 'M188'),
    'MinWidth': (int, 'minimum line width [px]', 'M188'),
    'MaxWidth': (int, 'maximum line width [px]', 'M188'),
    'RlcThresh': (int, 'line detection threshold', 'M188'),
    'LaserMaskFilter': (int, '0 none / > 0 background mask', 'M188'),
    'MedianFilter': (int, 'median filter 3/5/7/9 x1', 'M188'),
    'SpeckleFilterDx': (int, 'speckle low-pass filter dx', 'M188'),
    'SpeckleFilterDy': (int, 'speckle low-pass filter dy', 'M188'),
    'SpeckleFilter': (int, 'GUI 5.2 notation (combined)', 'EXE'),
    'SensorLUT': (int, '10→8 bit LUT 0 linear / 1 sqrt / 2 log', 'M188'),
    'Algorithm': (int, 'line extraction algorithm (Center Pixel / Barycenter / Filtered Barycenter, assumed)', 'EXE/I'),
    'Methode': (int, 'algorithm detail (assumed)', 'EXE/I'),
    'SubSample': (int, 'column subsampling (> 1 gives N < ROI_DX_PIX) - 1.9.x only', 'EXE'),
    # ROI
    'ROI_X_PIX': (int, 'ROI start x [px]', 'M188'),
    'ROI_Y_PIX': (int, 'ROI start y [px]', 'M188'),
    'ROI_DX_PIX': (int, 'ROI width [px] -> number of points N', 'M188'),
    'ROI_DY_PIX': (int, 'ROI height [px] -> Z range and fps', 'M188'),
    'ROI_X_MM': (float, 'ROI x start [mm] (read)', 'M188/EXE'),
    'ROI_Y_MM': (float, 'ROI y start [mm] (read)', 'M188'),
    'ROI_Z_MM': (float, 'ROI z start [mm]', 'EXE'),
    'ROI_DX_MM': (float, 'ROI width [mm]', 'M188/EXE'),
    'ROI_DY_MM': (float, 'ROI height [mm]', 'M188'),
    'ROI_DZ_MM': (float, 'ROI z width [mm]', 'EXE'),
    'AdjustSensorRoi': (int, 'product mode automatic ROI adjustment', 'PB/EXE'),
    'AdjustSenHmax': (int, 'product mode', 'PB'),
}

# 펌웨어 1.9.7 ID 44 응답에서 실측된 추가 항목 (2026-09-30, CameraSNR 4040760)
KNOWN_PARAMS.update({
    'CameraVK': (str, 'camera order number (VK000675)', 'LAB197'),
    'LibConflictErr': (int, 'library conflict error counter (0 = OK)', 'LAB197'),
    'SenSizeDx': (int, 'sensor columns (1408, read only)', 'LAB197'),
    'SenSizeDy': (int, 'sensor rows (1080, read only)', 'LAB197'),
    'MIN_X_MM': (float, 'calibrated X lower limit [mm] (measured -109.78)', 'LAB197'),
    'MAX_X_MM': (float, 'calibrated X upper limit [mm] (measured 115.51)', 'LAB197'),
    'MIN_Z_MM': (float, 'calibrated Z lower limit [mm] (measured 232.60)', 'LAB197'),
    'MAX_Z_MM': (float, 'calibrated Z upper limit [mm] (measured 546.46)', 'LAB197'),
    'ROI_X_MM': (float, 'mm ROI X start (measured -110)', 'LAB197/EXE'),
    'ROI_Z_MM': (float, 'mm ROI Z start (measured 233)', 'LAB197/EXE'),
    'ROI_DX_MM': (float, 'mm ROI X width (measured 225)', 'LAB197/EXE'),
    'ROI_DZ_MM': (float, 'mm ROI Z depth (measured 314)', 'LAB197/EXE'),
})

# ID 43 으로 절대 보내지 않을 항목(읽기 전용/정보)
READ_ONLY_PARAMS = frozenset({'MainVersion', 'Mainversion', 'LibVersion', 'CameraSNR', 'ProductVersion',
                              # 펌웨어 1.9.7 ID 44 응답에서 Min/Max/Step 없이 오는 정보 항목 (2026-09-30 실기)
                              'CameraVK', 'LibConflictErr', 'SenSizeDx', 'SenSizeDy',
                              'MIN_X_MM', 'MAX_X_MM', 'MIN_Z_MM', 'MAX_Z_MM',
                              'ROI_X_MM', 'ROI_Z_MM', 'ROI_DX_MM', 'ROI_DZ_MM'})  # AutoTriggerError 는 0 으로 리셋 가능(쓰기 가능)


def writable_params(kv, meta=None):
    """복원/재전송용 쓰기 가능 항목만 추출. meta(Min/Max/Step)가 있으면 그 존재로 판단(1.9.7),
    없으면 READ_ONLY_PARAMS 제외. NbrLines 는 항상 제외(스트리밍 시작/정지는 별도 명령)."""
    out = {}
    for k, v in kv.items():
        if k == 'NbrLines' or k in READ_ONLY_PARAMS:
            continue
        if meta and k not in meta:
            continue
        out[k] = v
    return out

_PRODUCT_RE = re.compile(r'^J\d\d_[A-Za-z0-9_]+$')   # J00_ProductType 등 제품 파라미터(사용 안 함)


def is_known_param(name: str) -> bool:
    return name in KNOWN_PARAMS or bool(_PRODUCT_RE.match(name))


def unknown_params(kv: Iterable[str]) -> List[str]:
    return [k for k in kv if not is_known_param(k)]


def check_writable(kv: Dict[str, Value], strict: bool = False) -> List[str]:
    """ID 43 전송 전 검사. 읽기 전용 항목은 ValueError; 미지 항목은 strict 면 ValueError, 아니면 반환."""
    ro = [k for k in kv if k in READ_ONLY_PARAMS]
    if ro:
        raise ValueError(f'read-only parameters cannot be written: {ro}')
    unk = unknown_params(kv)
    if unk and strict:
        raise ValueError(f'unknown parameters (not in whitelist): {unk}')
    return unk


# ---------------------------------------------------------------------------
# 텍스트 조립/파싱
# ---------------------------------------------------------------------------
def format_value(v: Value) -> str:
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float):
        s = f'{v:.6f}'.rstrip('0')
        return s + '0' if s.endswith('.') else s
    return str(v)


def format_params(kv: Dict[str, Value], terminator: str = '\n') -> str:
    """호스트→센서 ID 43 본문: 'Name = value\\n' 연결(LF 종단). 순서 유지."""
    return ''.join(f'{k} = {format_value(v)}{terminator}' for k, v in kv.items())


def parse_value(s: str) -> Value:
    s = s.strip().rstrip(';').strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()
    if re.fullmatch(r'[+-]?\d+', s):
        return int(s)
    if re.fullmatch(r'[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?', s):
        return float(s)
    return s


_LINE_RE = re.compile(r'^\s*[|>]?\s*([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(.*?)\s*\|?\s*$')


_META_RE = re.compile(r'\s+(Min|Max|Step)\s*=\s*([+-]?[0-9.eE+-]+)')


def split_value_meta(val: str) -> Tuple[str, Dict[str, Value]]:
    """1.9.7 ID 44 응답 'value Min=<m> Max=<M> Step=<s>' -> ('value', {'Min':..,'Max':..,'Step':..}).

    (2026-09-30 실기: MainVersion=1.9.7 은 쓰기 가능한 항목마다 Min/Max/Step 을 붙여 돌려주며 'Min= 0.1' 처럼
    등호 뒤 공백도 있음. 값 토큰만 비교해야 diff_params 오경보가 없다.)"""
    meta: Dict[str, Value] = {}
    m = _META_RE.search(val)
    if not m:
        return val.strip(), meta
    head = val[:m.start()].strip()
    for k, v in _META_RE.findall(val):
        meta[k] = parse_value(v)
    return head, meta


def parse_params_text_with_meta(text: str) -> Tuple[Dict[str, Value], Dict[str, Dict[str, Value]]]:
    """parse_params_text 와 같되 (values, meta) 를 함께 반환. meta[name] = {'Min','Max','Step'} (있을 때만)."""
    out: Dict[str, Value] = {}
    meta: Dict[str, Dict[str, Value]] = {}
    for raw in re.split(r'[\r\n\x00]+', text):
        line = raw.strip()
        if not line or line.startswith('#') or line.startswith('+--'):
            continue
        # 'Name = value; Name2 = value2;' 한 줄 여러 항목 (1.9.7 의 'Name=value Min=.. Max=..' 는 ';' 가 없어 한 항목)
        multi = ('=' in line and line.count('=') > 1 and ';' in line)
        parts = [q for q in line.split(';') if q.strip()] if multi else [line]
        for q in parts:
            m = _LINE_RE.match(q)
            if not m:
                continue
            name, val = m.group(1), m.group(2)
            val, md = split_value_meta(val)
            if val.strip().rstrip(';').strip() in ('GO!', ''):
                continue
            out[name] = parse_value(val)
            if md:
                meta[name] = md
    return out, meta


def parse_params_text(text: str) -> Dict[str, Value]:
    """센서가 돌려주는 설정 전문을 dict 로.

    허용 형식: 'Name = value;' (VC3DPar.txt), 'Name = value', 'Name=value Min=.. Max=.. Step=..' (펌웨어 1.9.7 ID 44 응답),
    '| Name: [ value ]|' (메뉴 표), '#' 주석·빈 줄·NULL/CR 무시. 같은 이름이 반복되면 마지막 값.
    """
    return parse_params_text_with_meta(text)[0]


def diff_params(requested: Dict[str, Value], actual: Dict[str, Value], rel_tol: float = 1e-6) -> Dict[str, Tuple[Value, Value]]:
    """요청값 vs ID 44 재읽기 값 비교 → {name: (requested, actual)} (불일치·부재만)."""
    d: Dict[str, Tuple[Value, Value]] = {}
    for k, v in requested.items():
        if k not in actual:
            d[k] = (v, None)   # type: ignore[assignment]
            continue
        a = actual[k]
        try:
            if abs(float(v) - float(a)) > rel_tol * max(1.0, abs(float(v))):
                d[k] = (v, a)
        except (TypeError, ValueError):
            if str(v) != str(a):
                d[k] = (v, a)
    return d
