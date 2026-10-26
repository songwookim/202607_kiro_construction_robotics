"""pytest 공용 헬퍼 (ROS 없는 Windows/Ubuntu 어디서든 실행 가능).

실제 ament_index_python 이 없을 때만 `_no_ros_shim/` 을 sys.path 에 넣어 pip xacro 의 `$(find pkg)` 가
워크스페이스 src/ 의 package.xml 을 찾아가게 한다. Ubuntu(ROS 2)에서는 아무것도 하지 않는다.
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM_DIR = os.path.join(HERE, "_no_ros_shim")


def ament_shim_dir_if_needed():
    """실제 ament_index_python 이 없으면 셤 디렉터리 경로를, 있으면 None 을 돌려준다."""
    try:
        importlib.import_module("ament_index_python.packages")
        return None
    except ImportError:
        return SHIM_DIR


# 최초 1회 판정 결과를 고정 (셤을 sys.path 에 넣은 뒤 다시 부르면 import 가 성공해 None 이 되므로)
AMENT_SHIM_DIR = ament_shim_dir_if_needed()
if AMENT_SHIM_DIR and AMENT_SHIM_DIR not in sys.path:
    sys.path.insert(0, AMENT_SHIM_DIR)

# 시험은 실기 자세–품질 데이터셋 (~/scan_data/scan_quality_dataset.jsonl) 을 읽지도 쓰지도 않는다 — 2026-10-06: e2e_safety 의
# 가짜 실행이 기록을 남겼고, 남은 기록은 다음 시험의 자세 선택 (scan_quality.phi_prior) 을 바꾼다.
# 데이터셋 시험은 경로를 직접 준다. ROS 서버 하위 프로세스도 이 환경을 물려받는다.
os.environ["SCAN_QUALITY_DATASET"] = "off"
os.environ["POSTURE_OFFSETS"] = "off"                     # 3G 자세 오프셋 자료: 씨앗만 (4d P8)
