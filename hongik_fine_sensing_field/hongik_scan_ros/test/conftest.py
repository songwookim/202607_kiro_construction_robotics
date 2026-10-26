"""hongik_scan_ros 시험 공용: 실기 자세–품질 데이터셋 (~/scan_data/scan_quality_dataset.jsonl) 을 읽지도 쓰지도 않는다 — 이 패키지의
setup.cfg 때문에 시험 뿌리가 여기라 scan_stack/conftest.py 의 같은 설정이 닿지 않을 수 있다 (2026-10-06). 서버 하위 프로세스도 물려받는다."""
import os

os.environ["SCAN_QUALITY_DATASET"] = "off"
os.environ["POSTURE_OFFSETS"] = "off"                     # 3G 자세 오프셋 자료: 씨앗만 (4d P8)
