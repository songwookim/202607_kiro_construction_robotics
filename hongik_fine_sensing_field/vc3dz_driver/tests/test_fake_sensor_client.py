import time
import warnings

import numpy as np
import pytest

from vc3dz_driver import params as P
from vc3dz_driver.client import SensorError, VC3DZClient
from vc3dz_driver.fake_sensor import FakeSensor, synthetic_frames
from vc3dz_driver.protocol import PARAM_ID_READ_PARAMS, PARAM_ID_RESET, PARAM_ID_WRITE_PARAMS
from vc3dz_driver.valid import valid_mask


def _collect(c, n, timeout=10.0):
    out = []
    t_end = time.monotonic() + timeout
    while len(out) < n and time.monotonic() < t_end:
        try:
            out.append(c.get_frame(timeout=0.5))
        except Exception:
            pass
    return out


def test_end_to_end_start_receive_stop_readparams(sample_frames):
    frames = [(f.x, f.z) for f in sample_frames]
    with FakeSensor(frames=frames, params={'AutoTriggerFPS': 200.0}) as fs:
        with VC3DZClient('127.0.0.1', fs.port, recv_timeout=2.0) as c:
            text, kv = c.read_params()
            assert kv['DataMode'] == 0 and kv['MainVersion'] == '1.9.3' and 'NbrLines' in kv
            assert 'AutoTriggerFPS = 200.0;' in text
            mism = c.init_session(P.INIT_STANDARD)
            assert mism == {}
            _, kv2 = c.read_params()
            assert kv2['DataMode'] == 4 and kv2['TransferAllPos_MM'] == 1 and kv2['NbrLines'] == 0
            c.start()
            recs = _collect(c, 60)
            assert len(recs) >= 50
            # 스트리밍 중 STRING 읽기가 오파싱 없이 동작
            _, kv3 = c.read_params()
            assert kv3['NbrLines'] == -1
            rep = c.stop()
            assert rep is not None and rep.error == 0 and rep.param_id == PARAM_ID_RESET
            time.sleep(0.1)
            n_after = fs.stats['frames_sent']
            time.sleep(0.15)
            assert fs.stats['frames_sent'] == n_after            # 정지 확인
            # 레코드 내용
            rc = np.array([r.result_counter for r in recs])
            assert (np.diff(rc) == 1).all()
            ts = np.array([r.ts_ms for r in recs], dtype=np.int64)
            assert set(np.diff(ts).tolist()) <= {10}               # INIT_STANDARD AutoTriggerFPS=100 → 10 ms (200 fps 기본값을 덮어씀)
            assert all(r.t_rx_ns > 0 for r in recs)
            assert all(r.checksum_ok for r in recs)
            assert recs[0].n_points == sample_frames[0].x.size and c.n_points is not None
            assert recs[0].x.dtype == np.float32
            np.testing.assert_array_equal(recs[0].x, sample_frames[0].x)
            np.testing.assert_array_equal(recs[1].z, sample_frames[1].z)
            assert valid_mask(recs[0].x, recs[0].z).mean() > 0.95
            assert c.stats['frames'] >= 50 and c.stats['format_errors'] == 0 and c.stats['checksum_mismatch_data'] == 0
            # NumberOfPoints 가 프레임마다 다른 모드-0 데이터 → 변화 카운트 > 0 (경고만, 폐기 없음)
            assert c.stats['n_points_changes'] > 0
        cmds = [pid for pid, _, _ in fs.command_log]
        assert cmds[0] == PARAM_ID_READ_PARAMS
        assert PARAM_ID_WRITE_PARAMS in cmds and cmds[-1] == PARAM_ID_RESET
        # 첫 명령 counter=0 확인은 fake 가 검사하지 않으므로 클라이언트 상태로 확인
        assert 33 not in cmds and 45 not in cmds


def test_counter_starts_at_zero_and_increments():
    with FakeSensor() as fs:
        c = VC3DZClient('127.0.0.1', fs.port)
        c.connect()
        assert c.counter == 0
        c.read_params()
        assert c.counter == 1
        c.read_params()
        assert c.counter == 2
        c.close()


def test_forbidden_ids_raise_without_sending():
    with FakeSensor() as fs:
        with VC3DZClient('127.0.0.1', fs.port) as c:
            with pytest.raises(ValueError):
                c.send_string(33)
            with pytest.raises(ValueError):
                c.command(45)
        assert fs.stats['commands'] == 0


def test_line_counter_variant_and_mode5():
    frames = synthetic_frames(5, 300)
    with FakeSensor(frames=frames, params={'AutoTriggerFPS': 300.0, 'DataMode': 5}, line_counter_variant=True) as fs:
        with VC3DZClient('127.0.0.1', fs.port) as c:
            c.start()
            recs = _collect(c, 20)
            c.stop()
    assert len(recs) >= 20
    assert all(r.line_counter == r.result_counter for r in recs)
    assert all(r.data_mode == 5 for r in recs)
    np.testing.assert_array_equal(recs[0].x, frames[0][0])
    np.testing.assert_array_equal(recs[0].z, frames[0][1])


def test_resync_after_garbage():
    frames = synthetic_frames(3, 100)
    with FakeSensor(frames=frames, params={'AutoTriggerFPS': 300.0}, corrupt_every=5) as fs:
        with VC3DZClient('127.0.0.1', fs.port) as c:
            c.start()
            recs = _collect(c, 40)
            c.stop()
    assert len(recs) >= 40
    assert c.stats['resyncs'] >= 5 and c.stats['format_errors'] == 0
    rc = [r.result_counter for r in recs]
    assert rc == list(range(rc[0], rc[0] + len(rc)))


def test_err_parm_strict_fake_raises_and_identifies():
    with FakeSensor(strict_params=True) as fs:
        with VC3DZClient('127.0.0.1', fs.port) as c:
            with pytest.raises(SensorError) as ei:
                c.set_params({'DataMode': 4, 'Bogus_Param': 1})
            assert ei.value.reply.error == -3003
            per = c.set_params_individually({'DataMode': 4, 'Bogus_Param': 1})
            assert per['DataMode'].error == 0 and per['Bogus_Param'].error == -3003


def test_backup_params_to_file(tmp_path):
    with FakeSensor() as fs:
        with VC3DZClient('127.0.0.1', fs.port) as c:
            p = c.backup_params_to_file(directory=str(tmp_path))
    txt = open(p, encoding='ascii').read()
    assert 'DataMode = 0;' in txt and p.endswith('.txt') and 'backup_' in p
    assert P.parse_params_text(txt)['ROI_DX_PIX'] == 1408
