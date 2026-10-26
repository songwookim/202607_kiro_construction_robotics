"""scan_l0 테스트 공용: 형제 패키지 (scan_core, vc3dz_driver) 를 sys.path 에.
실로봇/실센서 접속 없음 — 가상 팔 (sim_arm) + FakeSensor(1096 에뮬레이터) 만 사용."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)                       # .../scan_l0
STACK = os.path.dirname(PKG_ROOT)                      # .../scan_stack
for p in (PKG_ROOT, os.path.join(STACK, "scan_core"), os.path.join(STACK, "vc3dz_driver")):
    if p not in sys.path:
        sys.path.insert(0, p)

# 시험은 실기 자세–품질 데이터셋을 읽지도 쓰지도 않는다 — scan_l0/pyproject.toml 때문에 시험 뿌리가 scan_l0 라 scan_stack/conftest.py 의 같은
# 설정이 여기에는 닿지 않았다 (2026-10-06: e2e_safety 의 가짜 실행 6 건이 실기 데이터셋에 남고, 그 데이터셋의 실기 φ −15°
# 기록이 test_posture_3g_e2e (지금 test_sim_autoteach_3g) 의 자세 선택을 바꿨다). ROS 서버 하위 프로세스도 이 환경을 물려받는다.
os.environ["SCAN_QUALITY_DATASET"] = "off"
os.environ["POSTURE_OFFSETS"] = "off"                     # 3G 자세 오프셋 자료: 씨앗만 (4d P8)
# 시험만 가상 시계 (vc3dz_driver.virtual_clock) 를 설치할 수 있다 — clock.install 의 허가 (실기·launch 는 이 변수를 두지 않는다)
os.environ["SCAN_SIM_CLOCK_ALLOW"] = "1"

CONFIG = os.path.join(HERE, "data", "hongik_rb20_fake_moveit.yaml")   # 가상 팔 플랫폼 (RB20 시험 설정; 시험 자료)
CALIB = os.path.join(STACK, "eoat_description", "calib", "eoat_lab_01.yaml")
JOBS = os.path.join(PKG_ROOT, "scan_l0", "jobs")

# 시험 자세 [x, y, z mm, rx, ry, rz deg] (poses.from_mm_deg): 플랜지 z 가 아래(rx=180) → 광학 Z = -z_base, 광학 원점 = 플랜지 - 0.147 m
POSE_A = [600.0, 0.0, 500.0, 180.0, 0.0, 0.0]


@pytest.fixture(scope="session")
def calib_path() -> str:
    return CALIB


@pytest.fixture
def fake_sensor():
    from vc3dz_driver.fake_sensor import FakeSensor, synthetic_frames

    fs = FakeSensor(frames=synthetic_frames(40, 1408, z0=320.0)).start()
    yield fs
    fs.stop()


def pytest_configure(config):
    config.addinivalue_line("markers", "clock_real: run this sim test on the real clock (canary: load-dependent concurrency)")


@pytest.fixture(autouse=True)
def _sim_clock_mode(request):
    """시뮬레이션 시계 (sim_clock): clock_real 표지가 있으면 이 시험은 실시계. SCAN_AB_DIR 이 있으면 AutoTeach 결과 요약을 남긴다 (E4)."""
    import sim_clock
    sim_clock.set_test_marker(request.node.get_closest_marker("clock_real") is not None)
    restore = None
    if os.environ.get("SCAN_AB_DIR"):
        from scan_l0.autoteach import AutoTeach
        orig = AutoTeach.run
        n = [0]

        def run(self, *a, **k):
            r = orig(self, *a, **k)
            try:
                n[0] += 1
                sim_clock.ab_record(request.node.nodeid, n[0], sim_clock.ab_summary(r))
            except Exception as e:  # noqa: BLE001 - recording must never change a test
                print(f"A/B record failed: {e}")
            return r
        AutoTeach.run = run
        restore = (AutoTeach, orig)
    yield
    sim_clock.set_test_marker(False)
    if restore is not None:
        restore[0].run = restore[1]
