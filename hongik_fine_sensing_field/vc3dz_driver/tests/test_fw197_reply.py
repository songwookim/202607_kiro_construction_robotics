"""펌웨어 1.9.7 실기 ID 44 응답(2026-09-30, CameraSNR 4040760) 회귀 테스트."""
import os

import numpy as np

from vc3dz_driver import params as P

HERE = os.path.dirname(__file__)


def _text():
    with open(os.path.join(HERE, 'data', 'id44_reply_fw197.txt'), encoding='utf-8', errors='replace') as f:
        return f.read()


def test_parse_values_strip_min_max_step():
    vals, meta = P.parse_params_text_with_meta(_text())
    assert vals['MainVersion'] == '1.9.7'
    assert vals['DataMode'] == 0 and vals['ROI_DX_PIX'] == 1408
    assert vals['AutoTriggerFPS'] == 50.0  # 'AutoTriggerFPS=50.0 Min= 0.1 ...' (등호 뒤 공백 포함 메타)
    assert meta['AutoTriggerFPS']['Min'] == 0.1
    assert meta['TriggerMode'] == {'Min': -2, 'Max': 1, 'Step': 1}
    assert vals['MIN_Z_MM'] == 232.60 and vals['MAX_X_MM'] == 115.51
    assert isinstance(vals['ShutterTime'], int)
    assert len(vals) >= 50


def test_diff_params_no_false_positive():
    vals = P.parse_params_text(_text())
    req = {'ROI_DX_PIX': 1408, 'ROI_DY_PIX': 1080, 'EthernetPackNr': 1, 'EthernetSendNoWait': 1, 'AutoShutterMax': 100}
    assert P.diff_params(req, vals) == {}
    assert P.diff_params({'DataMode': 4}, vals) == {'DataMode': (4, 0)}


def test_fw197_names_known():
    vals = P.parse_params_text(_text())
    checker = getattr(P, 'is_known_param', None) or getattr(P, 'is_known', None)
    if checker is None:
        unknown = [k for k in vals if k not in P.KNOWN_PARAMS]
    else:
        unknown = [k for k in vals if not checker(k)]
    assert unknown == []


def test_cadence_anchor_is_robust_to_initial_short_interval():
    from vc3dz_driver.cli import summarize
    from vc3dz_driver.client import ProfileRecord
    n = 200
    d = np.full(n, 10)
    d[1] = 5  # 첫 증분만 5 ms, 이후 10 ms (실기에서 관측된 패턴)
    ts = np.cumsum(d) + 1470000
    recs = []
    for i in range(n):
        recs.append(ProfileRecord(ts_ms=int(ts[i]), result_counter=5 + i, t_rx_ns=int(1e9 + i * 1e7),
                                  global_counter=i, error=0, checksum_ok=True, n_points=4,
                                  x=np.zeros(4, np.float32), z=np.zeros(4, np.float32),
                                  line_counter=None, data_mode=4))
    out = summarize(recs, fps_nominal=100.0)
    assert out['ts_cadence_violations'] == 0
    assert out['ts_cadence_max_dev_ms'] <= 5.0 + 1e-6
