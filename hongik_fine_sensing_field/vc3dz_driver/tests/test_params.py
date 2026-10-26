import pytest

from vc3dz_driver import params as P

VC3DPAR_SAMPLE = """#
#  nano3D_Z Sensor parameter
#
#   Configuration file generated automatically on 2018-02-22 11:06:37
#
          DataMode = 0;
          NbrLines = 0;
       ShutterTime = 100;
        ROI_DX_PIX = 1384;
       TriggerMode = 0;
    AutoTriggerFPS = 50.000000;
  AutoTriggerError = 0;
EthernetSendNoWait = 1;
 TransferAllPos_MM = 0;
# End of File
"""

MENU_SAMPLE = """  +--------------------------------------------+
  |  nano3D_Z Sensor parameter                 |
  +--------------------------------------------+
  >            MainVersion:              1.8.7 |
  |              CameraSNR:            4090054 |
  |               DataMode:     [            0]|
  |         AutoTriggerFPS:     [    50.000000]|
  |  Save sensor parameter:     [          GO!]|
"""


def test_init_sets_match_design_3_3():
    s = P.INIT_STANDARD
    assert s['DataMode'] == 4 and s['TransferAllPos_MM'] == 1 and s['EthernetPackNr'] == 1
    assert s['EthernetSendNoWait'] == 1 and s['TriggerMode'] == -1 and s['AutoTriggerFPS'] == 100.0
    assert s['AutoTriggerError'] == 0 and s['ExposureMode'] == 0 and s['ShutterTime'] == 100 and s['GainVal'] == 250
    assert s['AutoShutterMin'] == 5 and s['AutoShutterMax'] == 100
    assert (s['ROI_X_PIX'], s['ROI_Y_PIX'], s['ROI_DX_PIX'], s['ROI_DY_PIX']) == (0, 0, 1408, 1080)
    assert s['LaserMode'] == 1 and s['TriggerDelayNs'] == 0
    assert 'NbrLines' not in s                       # start()/stop() 이 따로 보냄
    assert P.INIT_HW_TRIGGER['TriggerMode'] == 1 and P.INIT_HW_TRIGGER['TriggerDelayNs'] == 0
    assert P.INIT_PRESCAN['AutoTriggerFPS'] == 100.0
    for name in P.INIT_STANDARD:
        assert name in P.KNOWN_PARAMS, name


def test_format_params_lf_terminated_and_float_format():
    txt = P.format_params({'DataMode': 4, 'AutoTriggerFPS': 100.0, 'TriggerMode': -1})
    assert txt == 'DataMode = 4\nAutoTriggerFPS = 100.0\nTriggerMode = -1\n'
    assert P.format_value(50.5) == '50.5' and P.format_value(True) == '1'


def test_parse_vc3dpar_text():
    kv = P.parse_params_text(VC3DPAR_SAMPLE)
    assert kv['DataMode'] == 0 and isinstance(kv['DataMode'], int)
    assert kv['AutoTriggerFPS'] == 50.0 and isinstance(kv['AutoTriggerFPS'], float)
    assert kv['ROI_DX_PIX'] == 1384 and kv['TransferAllPos_MM'] == 0
    assert len(kv) == 9


def test_parse_menu_table_and_null_terminated():
    kv = P.parse_params_text(MENU_SAMPLE)
    assert kv['MainVersion'] == '1.8.7' and kv['CameraSNR'] == 4090054 and kv['DataMode'] == 0
    assert kv['AutoTriggerFPS'] == 50.0 and 'Save sensor parameter' not in kv
    kv2 = P.parse_params_text('DataMode = 4\x00NbrLines = -1\x00')
    assert kv2 == {'DataMode': 4, 'NbrLines': -1}
    kv3 = P.parse_params_text('DataMode = 4; NbrLines = -1;')
    assert kv3 == {'DataMode': 4, 'NbrLines': -1}


def test_roundtrip_format_parse():
    d = dict(P.INIT_STANDARD)
    assert P.parse_params_text(P.format_params(d)) == d


def test_whitelist_and_writable_checks():
    assert P.unknown_params(['DataMode', 'Foo', 'J00_ProductType', 'SubSample']) == ['Foo']
    with pytest.raises(ValueError):
        P.check_writable({'MainVersion': '1.9.3'})
    with pytest.raises(ValueError):
        P.check_writable({'Bogus': 1}, strict=True)
    assert P.check_writable({'Bogus': 1}) == ['Bogus']


def test_diff_params():
    req = {'DataMode': 4, 'AutoTriggerFPS': 100.0, 'TransferAllPos_MM': 1}
    act = {'DataMode': 4, 'AutoTriggerFPS': 100.0, 'TransferAllPos_MM': 0}
    assert P.diff_params(req, act) == {'TransferAllPos_MM': (1, 0)}
    assert P.diff_params({'X': 1}, {}) == {'X': (1, None)}
