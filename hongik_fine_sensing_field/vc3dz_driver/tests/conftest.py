import os
import sys

import pytest

# 시험만 가상 시계 (vc3dz_driver.virtual_clock) 를 설치할 수 있다 — clock.install 의 허가 (실기·launch 는 이 변수를 두지 않는다)
os.environ['SCAN_SIM_CLOCK_ALLOW'] = '1'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, 'tests', 'data')


@pytest.fixture(scope='session')
def sample_txt_path():
    return os.path.join(DATA_DIR, 'smartshape_sample.txt')


@pytest.fixture(scope='session')
def sample_frames(sample_txt_path):
    from vc3dz_driver.replay import load_smartshape_txt
    return load_smartshape_txt(sample_txt_path)
