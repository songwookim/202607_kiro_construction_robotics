import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scan_core.types import Calib, ScanJob  # noqa: E402


@pytest.fixture
def calib_lab() -> Calib:
    return Calib.lab_initial(eoat_yaw_rad=-np.pi / 2, opt_yaw_branch=-1)


@pytest.fixture
def job_1g() -> ScanJob:
    # 판 상면 z=0.05 (base), 테이블 z=-0.05, 시편 상단 = 판 상면
    return ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, 0.3, 0.05], surface_normal=[0, 0, 1],
                   table_z_m=-0.05, specimen_top_z_m=0.05, job_id="t1g")


@pytest.fixture
def job_3g() -> ScanJob:
    # 수직 판 y=0.7, 법선 -Y (로봇 쪽), 수직 스캔 z 0.3→0.6, 판 상단 0.7
    return ScanJob(start=[0.5, 0.7, 0.3], end=[0.5, 0.7, 0.6], surface_normal=[0, -1, 0],
                   specimen_top_z_m=0.7, table_z_m=0.0, job_id="t3g")
