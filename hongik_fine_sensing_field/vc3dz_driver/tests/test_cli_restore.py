"""`cli restore --file` : 백업 전문의 쓰기 가능 항목만 재전송하고 ID 44 재읽기로 검증한다 (2026-09-30 실기 사고 회귀 테스트)."""
import os

from vc3dz_driver import params as P
from vc3dz_driver.cli import main
from vc3dz_driver.client import VC3DZClient
from vc3dz_driver.fake_sensor import FakeSensor


def test_restore_from_backup_file(tmp_path):
    with FakeSensor(params={'ShutterTime': 100, 'GainVal': 0, 'LaserMode': 0, 'DataMode': 0}) as fs:
        # 1) 원본 상태 백업
        with VC3DZClient('127.0.0.1', fs.port, recv_timeout=2.0) as c:
            path = c.backup_params_to_file(directory=str(tmp_path))
            # 2) 다른 클라이언트(SmartShape 흉내)가 설정을 바꿈
            c.set_params({'ShutterTime': 3000, 'GainVal': 128, 'LaserMode': 1, 'DataMode': 4}, check=False)
            _, changed = c.read_params()
            assert changed['ShutterTime'] == 3000 and changed['LaserMode'] == 1
        # 3) dry-run 은 전송하지 않음
        rc = main(['--ip', '127.0.0.1', '--port', str(fs.port), 'restore', '--file', path, '--dry-run'])
        assert rc == 0
        with VC3DZClient('127.0.0.1', fs.port, recv_timeout=2.0) as c:
            _, still = c.read_params()
            assert still['ShutterTime'] == 3000
        # 4) 실제 복원 → 원본과 일치, 종료코드 0
        rc = main(['--ip', '127.0.0.1', '--port', str(fs.port), 'restore', '--file', path])
        assert rc == 0
        with VC3DZClient('127.0.0.1', fs.port, recv_timeout=2.0) as c:
            _, after = c.read_params()
        assert after['ShutterTime'] == 100 and after['GainVal'] == 0 and after['LaserMode'] == 0 and after['DataMode'] == 0
        assert os.path.exists(path)


def test_writable_params_excludes_readonly_and_nbrlines():
    text = ('MainVersion=1.9.7\nCameraSNR=4040760 Min=0 Max=9999999 Step=1\nDataMode=0 Min=-1 Max=9 Step=1\n'
            'NbrLines=0 Min=-2 Max=1000000 Step=1\nShutterTime=100 Min=1 Max=5000000 Step=1\nMIN_Z_MM=232.60\n')
    vals, meta = P.parse_params_text_with_meta(text)
    w = P.writable_params(vals, meta)
    assert set(w) == {'DataMode', 'ShutterTime'}
