"""안전 정책 (scan_l0.safety, 사용자 2026-10-07): 명시값은 플랫폼 safety.clearance_m 하나, 정지 거리·자세 허용·가까운 한계는 유도."""
import copy
import math
import os

import pytest
from conftest import CALIB, CONFIG, STACK

from scan_core.types import ScanJob
from scan_l0 import config as C
from scan_l0.safety import PolicyError, derive_policy, tool_lever_m

LAB = os.path.join(STACK, "scan_l0", "tests", "data", "hongik_lab_rb20_values.yaml")   # 연구실 플랫폼 값 (시험 자료)


@pytest.fixture(scope="module")
def calib():
    return C.attach_tool_body(C.load_calib(CALIB), C.load_settings(LAB))


def _job(fps=100.0):
    return ScanJob(start=[0.6, -0.19, 0.05], end=[0.6, 0.19, 0.05], profile_fps=fps)


def test_lab_policy_is_the_clearance_plus_a_derived_stopping_distance(calib):
    """연구실: 30 mm/s, 100 fps, 지연 0 (calib dt_s), 유령점 버스트 2 → 확인 3 프레임, 가드 100 Hz, 정지 감속 0.2 m/s²:
    s = 0.03·(3/100 + 0 + 1/100) + 0.03²/(2·0.2) = 3.45 mm → 한계 53.45 mm; standoff_min = 233.1 + 3.45 mm."""
    s = C.load_settings(LAB)
    pol = derive_policy(s.params, calib, _job(), v_max_mps=0.03, guard_rate_hz=100.0)
    assert pol.clearance_m == 0.05 and pol.confirm_frames == 3
    assert pol.stopping_m == pytest.approx(0.03 * 0.04 + 0.03 ** 2 / 0.4, abs=1e-12)
    assert pol.limit_m == pytest.approx(0.05 + pol.stopping_m) and pol.standoff_min_m == pytest.approx(0.2331 + pol.stopping_m)
    assert pol.z_min_mm == pytest.approx(pol.standoff_min_m * 1e3)
    lever = tool_lever_m(calib, _job())
    assert 0.25 < lever < 0.6
    assert pol.pose_tol_m == pytest.approx(0.001 + lever * math.radians(0.2))
    assert "clearance 50 mm" in pol.summary() and pol.as_dict()["limit_m"] == pytest.approx(pol.limit_m)


def test_faster_or_slower_sensing_changes_only_the_derived_part(calib):
    s = C.load_settings(CONFIG)
    slow = derive_policy(s.params, calib, _job(fps=20.0), v_max_mps=0.06, guard_rate_hz=100.0)
    fast = derive_policy(s.params, calib, _job(fps=100.0), v_max_mps=0.03, guard_rate_hz=100.0)
    assert slow.clearance_m == fast.clearance_m == 0.05
    assert slow.stopping_m > 3 * fast.stopping_m and slow.limit_m > fast.limit_m


def test_apply_fills_the_job_safety_fields_from_the_platform(calib):
    s = C.load_settings(LAB)
    pol = derive_policy(s.params, calib, _job(), v_max_mps=0.03, guard_rate_hz=100.0)
    j = pol.apply(_job(), max_laser_tilt_deg=40.0, torch_protrusion_m=0.168)
    assert j.safe_clearance_m == pytest.approx(pol.limit_m) and j.standoff_min_m == pytest.approx(pol.standoff_min_m)
    assert j.pose_tol_m == pytest.approx(pol.pose_tol_m) and j.max_laser_tilt_deg == 40.0       # 플랫폼 값이 작업에 닿는다
    j2 = ScanJob.from_dict(C.job_to_dict(j))                                                    # 파생 작업도 이어받는다
    assert j2.safe_clearance_m == j.safe_clearance_m and j2.pose_tol_m == j.pose_tol_m


@pytest.mark.parametrize("path,key", [(("safety",), "clearance_m"), (("moveit_jtc",), "stop_decel_tcp_mps2"),
                                      (("moveit_jtc",), "goal_pose_tol_m"), (("sensor",), "ghost_burst_frames")])
def test_missing_platform_values_refuse(calib, path, key):
    p = copy.deepcopy(C.load_settings(LAB).params)
    del p[path[0]][key]
    with pytest.raises(PolicyError, match=key):
        derive_policy(p, calib, _job(), v_max_mps=0.03, guard_rate_hz=100.0)
