"""3G 손목 형태 검사 (autoteach.wrist2_branch_problem, auto_teach.wrist2_branch_3g): 같은 툴 자세의 두 손목 해는 3G 자세에서 심 위치가
3–4 mm 달랐다 (2026-10-06 g3_24/25 손목2 + ↔ g3_13/17 −, 형태마다 실행 간 0.1 mm 재현) → 연구실 MoveIt 설정은 손목2 − 를 요구한다."""
import math
import os

import pytest

from scan_l0 import config as C
from scan_l0.autoteach import wrist2_branch_problem

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.dirname(os.path.dirname(HERE))
NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


def q_deg(*v):
    return [math.radians(x) for x in v]


def test_wrist2_branch_problem():
    g3_17 = q_deg(168.1, -28.7, -126.2, 124.4, -31.2, 117.3)                 # 손목2 − (g3_13/17 실기)
    g3_25 = q_deg(168.1, -28.8, -126.6, -54.9, 31.8, -63.1)                  # 손목2 + (g3_24/25 실기)
    assert wrist2_branch_problem(NAMES, g3_17, "neg") is None
    why = wrist2_branch_problem(NAMES, g3_25, "neg")
    assert why and "other wrist configuration" in why and "+31.8" in why and "direct teaching" in why
    assert wrist2_branch_problem(NAMES, g3_25, "pos") is None
    assert wrist2_branch_problem(NAMES, g3_25, "any") is None
    assert "singularity" in wrist2_branch_problem(NAMES, q_deg(168, -28, -126, 90, 2.0, 0), "neg")
    assert "no wrist2 joint" in wrist2_branch_problem([], g3_17, "neg")      # 이름 없으면 관절 순서를 추측하지 않는다 (2026-10-08)
    # 2π 동치 표현은 같은 형태 (P2 2026-10-09: IK 가 손목2 +290° 를 돌려준 실기 후보 = −70°, 지금 관절에 가까운 표현으로 바꿔도 형태는 같다)
    assert wrist2_branch_problem(NAMES, q_deg(176.1, -40.6, -69.6, 0.0, 290.2, 270.5), "neg") is None
    assert "other wrist configuration" in wrist2_branch_problem(NAMES, q_deg(176.1, -40.6, -69.6, 0.0, -290.2, 270.5), "neg")


def test_platform_wrist_configurations():
    """3G: 2026-10-07 사용자 — 음수 손목 선호는 일단 뺀다 (손목 형태는 도달성으로; 검사 자체는 남는다). 1G: 2026-10-09 사용자 — 연구실은
    손목2 음수로 제한 (캘리브 후보를 맞춘 팔 형태); 시험·KIRO 설정은 기본 (제한 없음)."""
    for f, branch_1g in (("scan_l0/tests/data/hongik_lab_rb20_values.yaml", "neg"),
                         ("scan_l0/tests/data/hongik_rb20_fake_moveit.yaml", "any"), ("hongik_scan_ros/config/hongik_kiro.yaml", "any")):
        at = C.load_settings(os.path.join(STACK, f)).auto_teach_config()
        assert at.wrist2_branch_3g == "any" and at.wrist2_branch_1g == branch_1g, f
    with pytest.raises(ValueError, match="wrist2_branch_3g"):
        C.AutoTeachConfig.from_params({"wrist2_branch_3g": "left"})
    with pytest.raises(ValueError, match="wrist2_branch_1g"):
        C.AutoTeachConfig.from_params({"wrist2_branch_1g": "up"})
    assert C.AutoTeachConfig.from_params({"wrist2_branch_1g": "neg"}).wrist2_branch_1g == "neg"
