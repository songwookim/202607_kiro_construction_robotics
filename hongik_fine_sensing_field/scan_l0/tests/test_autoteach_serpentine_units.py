"""auto-teach 2026-10-01 흐름의 순수 단위 테스트 (로봇·센서 없음):

  * 패스 계획: plan_first_pass (보정된 E → S_ext = S − R/2, 착점 오버스캔은 S 쪽만), plan_next_pass (S 관측 → 전체 길이 S + L,
    S 미관측 → 대비 경로 s_min → s_min + L, 이후 일반 규칙: 관측 끝 ± L, 대칭 연장, 직전 범위 밖 상한).
  * 연속 패스 일관성 (진행 방향 무관), 시간 오프셋 추정 dt = (s_fwd − s_rev)/(2v).
  * 용접 자세 입력: job.position (단일 출처) ↔ orientation, 충돌·이전 파일, save_job 은 position 만 쓴다, --position 덮어쓰기,
    auto-teach 는 자세가 없으면 / 배치와 맞지 않으면 아무 연결 없이 REFUSED.
  * make_refined_job 의 착점 기하 리드, S→E 정렬; _leg_phase 의 3G 태그.
"""
import dataclasses
import math
import os

import numpy as np
import pytest
import yaml
from conftest import CALIB, CONFIG, JOBS

from scan_core.frames import unit
from scan_core.localize import PassFit
from scan_core.path_gen import pass_overscan
from scan_core.specimen import GrooveSpec
from scan_core.types import Calib, JobError, ScanJob
from scan_l0 import config as C
from scan_l0.session import ScanSession
from scan_l0.autoteach import (AutoTeach, PassPlan, SeamLine, _Abort, direction_consistency, line_consistency, make_coarse_job,
                               make_refined_job,
                               plan_first_pass, plan_next_pass, time_offset_estimate)

L = 0.380
OV = 0.020


def _line(s_S=0.0, s_E=0.320, **kw) -> SeamLine:
    d = dict(c=np.array([0.6, -0.16, 0.04]), u=np.array([0.0, 1.0, 0.0]), n=np.array([0.0, 0.0, 1.0]), s_S=s_S, s_E=s_E)
    d.update(kw)
    return SeamLine(**d)


# --------------------------------------------------------------------------------------
# 패스 계획
# --------------------------------------------------------------------------------------
def test_plan_first_pass_extends_S_by_the_residual_and_overscans_S_only():
    p = plan_first_pass(_line(0.0, 0.320), design_length_m=L, overscan_m=OV)
    assert p.direction == "E->S" and p.first == "E" and p.last == "S"
    assert p.s_E == pytest.approx(0.320) and p.ov_E == 0.0                     # 보정된 E: 판 안쪽, 오버스캔 없음
    assert p.s_S == pytest.approx(-0.060) and p.ov_S == OV                    # R = 60 mm: S 끝이 있을 수 있는 가장 먼 곳 + 착점 오버스캔
    p0 = plan_first_pass(_line(0.0, 0.380), design_length_m=L, overscan_m=OV)
    assert p0.s_S == pytest.approx(0.0) and p0.ov_S == OV                     # R ≈ 0: S 에서 20 mm 오버스캔만
    pc = plan_first_pass(_line(0.0, 0.100), design_length_m=L, overscan_m=OV)
    assert pc.s_S == pytest.approx(-0.28)                 # R = 280 mm 그대로 (상한 없음: 기하)
    pn = plan_first_pass(_line(0.0, 0.300), design_length_m=None, overscan_m=OV)
    assert pn.s_S == 0.0 and pn.ov_S == OV                                    # 설계 길이 없음 → 연장 없음


def test_plan_next_pass_after_pass1_S_observed_is_the_full_length_scan():
    # 패스 1 (E→S) 이 S 끝을 −0.030 에서 관측, E 쪽은 0.320 (판 안쪽에서 시작) 까지 봤다
    cur = _line(-0.030, 0.320, obs_S=True, obs_E=False, hit_range=(-0.0498, 0.320))
    p = plan_next_pass(1, "E->S", cur, -0.030, None, design_length_m=L, hit_range=cur.hit_range,
                       overscan_m=OV)
    assert p.direction == "S->E" and p.rule == "design_length"
    assert p.s_S == pytest.approx(-0.030) and p.s_E == pytest.approx(0.350) and p.ov_S == OV and p.ov_E == OV
    # 설계 길이로 정한 E 가 본 구간보다 짧으면 최소 오버스캔만큼은 본 끝 밖으로
    cur2 = _line(-0.030, 0.360, obs_S=True, hit_range=(-0.05, 0.36))
    p2 = plan_next_pass(1, "E->S", cur2, -0.030, None, design_length_m=0.300, hit_range=cur2.hit_range,
                        overscan_m=OV)
    assert p2.s_E == pytest.approx(0.380)
    # 상한 없음: 관측 S + 설계 길이 그대로 (도달은 포락선이 본다)
    cur3 = _line(-0.030, 0.150, obs_S=True, hit_range=(-0.05, 0.150))
    p3 = plan_next_pass(1, "E->S", cur3, -0.030, None, design_length_m=L, hit_range=cur3.hit_range,
                        overscan_m=OV)
    assert p3.s_E == pytest.approx(0.350)


def test_plan_next_pass_fallback_when_pass1_missed_the_S_end():
    """패스 1 이 S 끝을 못 봤다 (시편이 설계보다 길거나 끝이 가려짐): 그루브는 적어도 패스 1 경계 s_min 까지 → E 끝 ≤ s_min + L.
    패스 2 = s_min (판 안쪽, 오버스캔 없음) → E_ext = s_min + L (+ 오버스캔); 패스 3 은 관측된 E − L 까지 전체 길이."""
    cur = _line(0.0075, 0.375, obs_S=False, obs_E=False, hit_range=(0.0075, 0.375))
    p = plan_next_pass(1, "E->S", cur, None, None, design_length_m=L, hit_range=cur.hit_range, overscan_m=OV)
    assert p.rule == "fallback" and p.direction == "S->E"
    assert p.s_S == pytest.approx(0.0075) and p.ov_S == 0.0 and p.s_E == pytest.approx(0.3875) and p.ov_E == OV
    # 패스 2 가 E 를 0.380 에서 관측 → 패스 3 (E→S) = 전체 길이 E − L
    cur2 = _line(0.0075, 0.380, obs_S=False, obs_E=True, hit_range=(0.0075, 0.4075))
    p3 = plan_next_pass(2, "S->E", cur2, None, 0.380, design_length_m=L, hit_range=cur2.hit_range,
                        overscan_m=OV)
    assert p3.direction == "E->S" and p3.rule == "design_length" and p3.s_S == pytest.approx(0.0) and p3.s_E == pytest.approx(0.380)
    assert p3.ov_S == OV and p3.ov_E == OV


def test_plan_next_pass_full_length_and_symmetric_rules():
    cur = _line(0.0, 0.380, obs_S=True, obs_E=True, hit_range=(-0.02, 0.40))
    p = plan_next_pass(2, "S->E", cur, 0.0, 0.380, design_length_m=L, hit_range=cur.hit_range, overscan_m=OV)
    assert p.rule == "full_length" and p.direction == "E->S" and (p.s_S, p.s_E) == (0.0, 0.380) and p.ov_S == p.ov_E == OV
    cur2 = _line(0.0, 0.300, hit_range=(0.0, 0.300))
    p2 = plan_next_pass(2, "S->E", cur2, None, None, design_length_m=L, hit_range=cur2.hit_range,
                        overscan_m=OV)
    assert p2.rule == "symmetric" and p2.s_S == pytest.approx(-0.040) and p2.s_E == pytest.approx(0.340)
    cur3 = _line(0.0, 0.200, hit_range=(0.0, 0.200))
    p3 = plan_next_pass(3, "E->S", cur3, None, None, design_length_m=L, hit_range=cur3.hit_range,
                        overscan_m=OV)
    assert p3.s_S == pytest.approx(-0.09) and p3.s_E == pytest.approx(0.29)   # 대칭 연장 (L − 200)/2


# --------------------------------------------------------------------------------------
# 일관성 · 시간 오프셋
# --------------------------------------------------------------------------------------
def test_line_consistency_floor_is_the_weld_aim_tolerance():
    """2026-10-08 1G 실기 g1_b: 패스끼리 직선 0.63 mm, 끝 0.37–0.60 mm 안에서 같았는데 측정 3σ (0.36–0.58 mm) 만으로 '불일치' 라 패스 4
    까지 돌았다. 용접 겨냥 허용치 (root_position_mm.tol 1 mm) 가 바닥이면 수렴; 바닥보다 크게 다르면 여전히 불일치."""
    r = 0.1e-3
    a = _line(0.0, 0.380, obs_S=True, obs_E=True, spacing_m=0.0003, line_rms_m=r, plane_rms_m=r)
    b = _line(0.0006, 0.380, obs_S=True, obs_E=True, spacing_m=0.0003, line_rms_m=r, plane_rms_m=r)
    ok, why, m = line_consistency(a, b)
    assert not ok and "|dS| 0.60 mm" in why                                             # 측정 3σ 만: 0.37 mm < 0.6 mm
    ok, why, m = line_consistency(a, b, aim_tol_m=1.0e-3)
    assert ok and m["end_tol_S_mm"] == pytest.approx(1.0) and m["aim_tol_mm"] == pytest.approx(1.0), why
    assert "weld aim tolerance 1.00 mm" in why
    ok, why, _ = line_consistency(a, _line(0.0015, 0.380, obs_S=True, obs_E=True, spacing_m=0.0003, line_rms_m=r, plane_rms_m=r),
                                  aim_tol_m=1.0e-3)
    assert not ok and "|dS| 1.50 mm" in why


def test_line_consistency_is_direction_agnostic_and_uses_the_measured_sigma():
    """수렴 판정은 두 패스가 잰 잡음에서 (k = 3): 위치 k·√(σ₁²+σ₂²) (σ = 루트 직선 잔차 0.1 mm → 0.42 mm), 방향 atan(그 / 길이)
    (380 mm → 0.064°), 끝 k·√(σ_끝₁² + σ_끝₂²) (σ_끝 = max(패스가 잰 끝 σ, 착점 간격/√12); 0.3 mm 간격 → 0.37 mm, 4.5 mm → 5.5 mm;
    σ_끝을 모르면 끝은 비교하지 않는다)."""
    r = 0.1e-3
    a = _line(0.0, 0.380, obs_S=True, obs_E=False, spacing_m=0.0003, line_rms_m=r, plane_rms_m=r)
    b = _line(0.0003, 0.380, obs_S=True, obs_E=True, spacing_m=0.0003, line_rms_m=r, plane_rms_m=r)
    ok, why, m = line_consistency(a, b)
    assert ok and m["dS_mm"] == pytest.approx(0.3) and "dE_mm" not in m, why          # E 는 a 에서 미관측 → 비교 안 함
    assert m["tol_pos_mm"] == pytest.approx(3 * math.sqrt(2) * 0.1) and m["end_tol_S_mm"] == pytest.approx(3 * math.sqrt(2) * 0.3 / math.sqrt(12))
    ok, why, _ = line_consistency(a, _line(0.0008, 0.380, obs_S=True, spacing_m=0.0003, line_rms_m=r, plane_rms_m=r))
    assert not ok and "|dS| 0.80 mm" in why                                              # 0.37 mm < 0.8 mm
    ok, _, m = line_consistency(_line(obs_S=True, spacing_m=0.0045, line_rms_m=r), _line(0.004, obs_S=True, spacing_m=0.0045, line_rms_m=r))
    assert ok and m["end_tol_S_mm"] == pytest.approx(3 * math.sqrt(2) * 4.5 / math.sqrt(12))
    # 패스가 잰 끝 σ 가 간격/√12 보다 크면 그것으로 (0.3 mm 간격이지만 끝 σ 0.2 mm → 0.85 mm 허용: 0.8 mm 차이는 일관)
    a2 = dataclasses.replace(a, sigma_S_m=0.2e-3)
    ok, why, m = line_consistency(a2, _line(0.0008, 0.380, obs_S=True, spacing_m=0.0003, line_rms_m=r, plane_rms_m=r, sigma_S_m=0.2e-3))
    assert ok and m["end_tol_S_mm"] == pytest.approx(3 * math.sqrt(2) * 0.2), why
    # 끝 σ 를 모르면 (간격 NaN, 끝 σ 없음) 끝 비교를 하지 않는다 (예전: 허용 0 → 늘 불일치)
    ok, why, m = line_consistency(_line(obs_S=True, line_rms_m=r, plane_rms_m=r), _line(0.004, obs_S=True, line_rms_m=r, plane_rms_m=r))
    assert ok and "not compared" in why and math.isnan(m["end_tol_S_mm"])
    tilted = _line(u=unit([math.sin(math.radians(0.2)), 1.0, 0.0]), line_rms_m=r, plane_rms_m=r)
    ok, why, m = line_consistency(_line(line_rms_m=r, plane_rms_m=r), tilted)
    assert not ok and m["du_deg"] > m["tol_u_deg"] and m["tol_u_deg"] == pytest.approx(math.degrees(math.atan2(3 * math.sqrt(2) * r, 0.32)))
    shifted = _line(c=np.array([0.6012, -0.16, 0.04]), line_rms_m=r, plane_rms_m=r)
    ok, _, m = line_consistency(_line(line_rms_m=r, plane_rms_m=r), shifted)
    assert not ok and m["dpos_mm"] == pytest.approx(1.2)
    noisy = 0.5e-3                                                                        # 잡음이 크면 허용도 커진다 (같은 1.2 mm 가 일관)
    ok, _, m = line_consistency(_line(line_rms_m=noisy, plane_rms_m=r), dataclasses.replace(shifted, line_rms_m=noisy))
    assert ok and m["tol_pos_mm"] == pytest.approx(3 * math.sqrt(2) * 0.5)


# 1G 실기 2026-10-08 g1_batch3_t03 의 패스 직선 (autoteach_log.json line_history 1..4): 같은 방향 끼리 S 끝 0.24 / 0.01 mm, 반대 방향
# 끼리 1.28 / 1.51 / 1.50 mm — 연속 비교로는 수렴할 수 없었다 (패스 3·4, 34 s)
B3T03 = [
    dict(k=1, direction="E->S", c=[1.2639730293240734, -0.0038860758862403837, -0.1471121296787312],
         u=[-0.9949345569183529, -0.09920528436388748, -0.01623388443845516], n=[-0.01660225400899716, 0.0028887272407750017, 0.9998580001263926],
         s_S=-0.17225775963651802, s_E=0.15392195745080053, obs_S=True, obs_E=False, spacing_m=0.0002981013533648047,
         line_rms_m=0.00013887135957428874, plane_rms_m=0.0001389162388873576, sigma_S_m=8.605444830548087e-05,
         sigma_E_m=8.605444830548087e-05, range_mm=332.6750754465237),
    dict(k=2, direction="S->E", c=[1.24436721633228, -0.005898972879993089, -0.14708151315788176],
         u=[-0.9949599767460562, -0.09891838456773663, -0.016425524892578907], n=[-0.016832277720529448, 0.0032787950665995958, 0.9998529511531434],
         s_S=-0.1906839266812138, s_E=0.1870578533346585, obs_S=True, obs_E=True, spacing_m=0.00029723551440898836,
         line_rms_m=0.00013024204328220855, plane_rms_m=0.00013695289154024994, sigma_S_m=8.580450212837316e-05,
         sigma_E_m=8.580450212837316e-05, range_mm=332.98568420798904),
    dict(k=3, direction="E->S", c=[1.2450532947206683, -0.005748155417162705, -0.1472807799722935],
         u=[-0.9949364733349874, -0.0991901360792377, -0.016208976905416624], n=[-0.016622573481457802, 0.0033446663812634627, 0.9998562412955433],
         s_S=-0.19150360902858082, s_E=0.18777744293563758, obs_S=True, obs_E=True, spacing_m=0.0002977048531828119,
         line_rms_m=0.00014867402401077127, plane_rms_m=0.00014120241667922955, sigma_S_m=8.593998856207724e-05,
         sigma_E_m=8.593998856207724e-05, range_mm=333.4239742597276),
    dict(k=4, direction="S->E", c=[1.2444952049582827, -0.005886348823618576, -0.14707065275145742],
         u=[-0.9949598132063935, -0.0989327061269842, -0.01634899882838118], n=[-0.016761069444286486, 0.0033353914249875977, 0.9998539601937505],
         s_S=-0.1905679959232504, s_E=0.18741924758041828, obs_S=True, obs_E=True, spacing_m=0.00029824653196323936,
         line_rms_m=0.00013887748850562184, plane_rms_m=0.00013700310715130068, sigma_S_m=8.609635775692429e-05,
         sigma_E_m=8.609635775692429e-05, range_mm=333.1732556102315),
]


def _lab_line(d) -> SeamLine:
    return SeamLine(**{**d, "c": np.array(d["c"]), "u": np.array(d["u"]), "n": np.array(d["n"])})


def test_direction_consistency_lab_b3_t03_converges_at_pass_3_with_the_bias_free_ends():
    """P8: b3 t03 — 패스 2 (첫 반대 방향 쌍) 는 같은 방향 반복도가 아직 없어 한 패스 더 (확인 먼저); 패스 3 은 E→S 패스 1·3 이 반복
    (|dS| 0.23 mm ≤ 1 mm) 되고 S 끝 방향 편향 +1.40 mm (S→E 빼기 E→S) 의 절반 0.70 ≤ 끝 허용 1.00 mm → 수렴, 납품 S = 두 방향
    평균. 패스 4 (17 s) 가 필요 없다. 연속 비교 (예전 규칙) 는 여전히 불일치."""
    lines = [_lab_line(d) for d in B3T03]
    ok, why, _ = line_consistency(lines[1], lines[2], aim_tol_m=1e-3)
    assert not ok and "|dS| 1.51 mm" in why
    d2 = direction_consistency(lines[:2], aim_tol_m=1e-3)
    assert d2["state"] == "no_same_direction_pass" and not d2["ok"]
    d3 = direction_consistency(lines[:3], aim_tol_m=1e-3)
    assert d3["state"] == "converged" and d3["ok"], d3["why"]
    eS, eE = d3["ends"]["S"], d3["ends"]["E"]
    assert eS["bias_m"] * 1e3 == pytest.approx(1.40, abs=0.01) and eS["tol_m"] == pytest.approx(1e-3)
    assert eS["passes_SE"] == [2] and eS["passes_ES"] == [1, 3] and eE["passes_SE"] == [2] and eE["passes_ES"] == [3]
    cur = lines[2]
    s_SE = cur.s_of(lines[1].S)
    s_ES = 0.5 * (cur.s_of(lines[0].S) + cur.s_of(lines[2].S))
    assert eS["s_combined_m"] == pytest.approx(0.5 * (s_SE + s_ES)) and eS["shift_m"] == pytest.approx(0.5 * (s_SE + s_ES) - cur.s_S)
    assert abs(eE["bias_m"]) < 0.1e-3 and abs(eE["shift_m"]) < 0.05e-3
    m = d3["metrics"]
    assert m["same_direction_pass"] == 1 and m["opposite_pass"] == 2 and m["repeat_ok"] and m["opposite_line_ok"]
    assert m["ends"]["S"]["bias_mm"] == pytest.approx(eS["bias_m"] * 1e3) and "bias-free mean" in d3["why"]
    # 패스 4 에서도 (S→E 2·4 반복) 같은 결론 — 일반 규칙
    d4 = direction_consistency(lines, aim_tol_m=1e-3)
    assert d4["state"] == "converged" and d4["metrics"]["same_direction_pass"] == 2
    assert d4["metrics"]["other_direction_repeat"]["passes"] == [1, 3] and d4["metrics"]["other_direction_repeat"]["ok"]


def _dir_line(k, direction, s_S, s_E, *, obs_S=True, obs_E=True, **kw) -> SeamLine:
    r = 0.1e-3
    d = dict(obs_S=obs_S, obs_E=obs_E, k=k, direction=direction, spacing_m=0.0003, line_rms_m=r, plane_rms_m=r, sigma_S_m=0.09e-3,
             sigma_E_m=0.09e-3)
    d.update(kw)
    return _line(s_S, s_E, **d)


def test_direction_consistency_states_follow_repeatability_bias_and_posture():
    """같은 방향 반복이 안 되면 실제 불일치 ('not_repeatable', b1 t08 처럼 E→S 1·3 이 1.6 mm); 편향이 허용의 두 배보다 크면 한 방향만
    반복됐을 때 'bias_unconfirmed' (한 패스 더), 두 방향 모두 반복되면 'bias_exceeds_tol' (멈춤); 자세가 다르면 'posture_differs';
    반대 방향 패스의 직선이 다르면 'line_differs'. 끝 σ 가 크면 허용도 k·σ 로 커진다."""
    aim = 1e-3
    # 반복 안 됨 (E→S 1·3 이 S 끝 1.6 mm 다름)
    h = [_dir_line(1, "E->S", 0.0, 0.380, obs_E=False), _dir_line(2, "S->E", 0.0016, 0.380), _dir_line(3, "E->S", 0.0016, 0.3812)]
    d = direction_consistency(h, aim_tol_m=aim)
    assert d["state"] == "not_repeatable" and not d["ok"] and "|dS| 1.60 mm" in d["why"]
    # 반복되는 편향 3 mm (S→E 가 S 끝을 +3 mm): 절반 1.5 > 1 mm
    big = [_dir_line(1, "E->S", 0.0, 0.380), _dir_line(2, "S->E", 0.003, 0.380), _dir_line(3, "E->S", 0.0001, 0.380)]
    d = direction_consistency(big, aim_tol_m=aim)
    assert d["state"] == "bias_unconfirmed" and not d["ok"] and d["ends"]["S"]["bias_m"] == pytest.approx(0.00295)
    d = direction_consistency(big + [_dir_line(4, "S->E", 0.0031, 0.380)], aim_tol_m=aim)
    assert d["state"] == "bias_exceeds_tol" and "more passes would repeat the same bias" in d["why"]
    assert d["metrics"]["other_direction_repeat"]["passes"] == [1, 3]
    # 같은 3 mm 편향도 끝 σ 가 크면 (착점 간격 4.5 mm → σ 1.3 mm, 허용 5.5 mm) 설명된다
    coarse = [dataclasses.replace(L, spacing_m=0.0045) for L in big]
    d = direction_consistency(coarse, aim_tol_m=aim)
    assert d["state"] == "converged" and d["ends"]["S"]["tol_m"] == pytest.approx(3 * math.sqrt(2) * 0.0045 / math.sqrt(12))
    # 자세가 다르면 방향 편향으로 보지 않는다 (자세 오프셋)
    ok3 = [_dir_line(1, "E->S", 0.0, 0.380), _dir_line(2, "S->E", 0.0014, 0.380), _dir_line(3, "E->S", 0.0002, 0.380)]
    assert direction_consistency(ok3, aim_tol_m=aim)["state"] == "converged"
    d = direction_consistency(ok3, aim_tol_m=aim, postures=[(0.0, 0.0, None), (35.0, 0.0, True), (0.0, 0.0, None)])
    assert d["state"] == "posture_differs" and not d["ok"]
    # 반대 방향 패스의 직선이 1.2 mm 옆 → 끝 편향 모형 밖
    shifted = dataclasses.replace(ok3[1], c=np.array([0.6012, -0.16, 0.04]))
    d = direction_consistency([ok3[0], shifted, ok3[2]], aim_tol_m=aim)
    assert d["state"] == "line_differs" and not d["ok"]
    # 같은 방향 이전 패스가 없으면 (패스 2) 판단하지 않는다
    assert direction_consistency(ok3[:2], aim_tol_m=aim)["state"] == "no_same_direction_pass"


def test_time_offset_estimate_from_forward_and_reverse_end_observations():
    line = _line(0.0, 0.380)
    v, dt = 0.03, 0.010
    S_true = line.S
    ends_f = {"S": {"observed": True, "point": S_true + line.u * v * dt}, "E": {"observed": False, "point": line.E}}
    ends_r = {"S": {"observed": True, "point": S_true - line.u * v * dt}, "E": {"observed": True, "point": line.E}}
    r = time_offset_estimate([{"k": 1, "direction": "E->S", "speed_mps": v, "ends": ends_r},
                              {"k": 2, "direction": "S->E", "speed_mps": v, "ends": ends_f}], line)
    assert r["n_ends"] == 1 and r["dt_s"] == pytest.approx(dt) and set(r["per_end"]) == {"S"}
    r2 = time_offset_estimate([{"k": 1, "direction": "E->S", "speed_mps": v, "ends": ends_r}], line)
    assert r2["n_ends"] == 0 and math.isnan(r2["dt_s"])
    # 자세가 다른 정·역 패스 (3G 끝 패스 vs 주 패스) 는 짝짓지 않는다 — 자세 오프셋이 시간처럼 보인다 (A-27)
    main, endp = (35.0, 0.0, True), (40.0, 0.0, False)
    r3 = time_offset_estimate([{"k": 1, "direction": "E->S", "speed_mps": v, "ends": ends_r, "posture": endp},
                               {"k": 2, "direction": "S->E", "speed_mps": v, "ends": ends_f, "posture": main}], line)
    assert r3["n_ends"] == 0 and math.isnan(r3["dt_s"])
    r4 = time_offset_estimate([{"k": 1, "direction": "E->S", "speed_mps": v, "ends": ends_r, "posture": main},
                               {"k": 2, "direction": "E->S", "speed_mps": v, "ends": ends_r, "posture": endp},
                               {"k": 3, "direction": "S->E", "speed_mps": v, "ends": ends_f, "posture": main}], line)
    assert r4["n_ends"] == 1 and r4["dt_s"] == pytest.approx(dt) and r4["per_end"]["S"]["k_rev"] == 1


# --------------------------------------------------------------------------------------
# 용접 자세 입력
# --------------------------------------------------------------------------------------
def test_job_position_is_the_single_source_and_legacy_orientation_loads(tmp_path):
    j, doc = C.load_job(os.path.join(JOBS, "groove_10T_1G.yaml"))
    assert doc["job"]["position"] == "1G" and "orientation" not in doc["job"] and j.orientation == "flat_1G" and j.position == "1G"
    j3, doc3 = C.load_job(os.path.join(JOBS, "groove_10T_3G.yaml"))
    assert doc3["job"]["position"] == "3G" and j3.orientation == "vertical_3G"
    for name, pos in (("example_1G.yaml", "1G"), ("example_3G.yaml", "3G")):
        assert C.load_job(os.path.join(JOBS, name))[0].position == pos
    # 이전 파일: 명시 orientation 은 그대로 읽힌다; 'auto'/없음 = 미지정
    d = C.job_to_dict(j)
    assert C.job_from_dict({**d, "orientation": "vertical_3G"}).position == "3G"
    assert C.job_from_dict({**d, "orientation": "auto"}).position is None
    assert C.job_from_dict({k: v for k, v in d.items() if k != "orientation"}).position is None
    # position 우선, 충돌은 거부
    assert C.job_from_dict({**d, "orientation": "auto", "position": "3G"}).orientation == "vertical_3G"
    with pytest.raises(JobError, match="conflicts with the legacy orientation"):
        C.job_from_dict({**d, "orientation": "flat_1G", "position": "3G"})
    with pytest.raises(JobError, match="give position"):
        C.job_from_dict({**d, "position": "2G"})
    # save_job: position 만 (orientation 키 없음), 왕복
    p = C.save_job(tmp_path / "j.yaml", j, {}, {"rough": True})
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert raw["job"]["position"] == "1G" and "orientation" not in raw["job"]
    assert C.load_job(p)[0].orientation == "flat_1G"
    ja = C.job_from_dict({**d, "orientation": "auto"})
    raw2 = yaml.safe_load(C.save_job(tmp_path / "a.yaml", ja).read_text(encoding="utf-8"))
    assert raw2["job"]["orientation"] == "auto" and "position" not in raw2["job"]
    # --position 덮어쓰기
    assert C.with_position(ja, "1G").orientation == "flat_1G" and C.with_position(j, None) is j


def _at(job, doc, tmp_path, **kw):
    settings = C.load_settings(CONFIG)
    calib = C.load_calib(CALIB)
    # 로봇 플랫폼은 쓰이지 않는다 (자세 검사가 이동 전에 거부)
    return AutoTeach(settings, job, doc, calib, out_dir=str(tmp_path), log=lambda s: None,
                     session=ScanSession(robot=None, sensor=None, owned=False), **kw)


def test_autoteach_refuses_a_job_without_the_groove_design_before_any_connection(tmp_path):
    """설계 정보 (specimen: 길이·두께·베벨·갭) 는 필수 입력 — 없으면 잔여 연장 R = L_design − L_SE 와 길이 검사를 할 수 없다.
    2026-10-02: 실기 job_rough.yaml 이 teach 로만 만들어져 specimen 블록이 없었다 → 조용히 약화되지 않고 거부."""
    j, doc = C.load_job(os.path.join(JOBS, "groove_10T_1G.yaml"))
    doc_no_spec = {k: v for k, v in doc.items() if k != "specimen"}
    r = _at(j, doc_no_spec, tmp_path / "a").run()
    assert r.status == "REFUSED" and "no specimen (groove design) block" in r.reason and r.n_move_l == 0, r.reason
    assert "groove_10T_1G.yaml" in r.reason
    doc_zero = {**doc, "specimen": {**doc["specimen"], "weld_length_mm": 0.0}}
    r = _at(j, doc_zero, tmp_path / "b").run()
    assert r.status == "REFUSED" and "weld_length_mm > 0" in r.reason


def test_autoteach_refuses_a_missing_or_inconsistent_position_before_any_connection(tmp_path):
    j, doc = C.load_job(os.path.join(JOBS, "groove_10T_1G.yaml"))
    ja = C.job_from_dict({**C.job_to_dict(j), "orientation": "auto"})
    r = _at(ja, doc, tmp_path / "a").run()
    assert r.status == "REFUSED" and "give position: 1G|3G" in r.reason and r.n_move_l == 0
    assert (tmp_path / "a" / "autoteach_log.json").exists()
    # --position 3G 를 수평 배치에 주면 범용 입력 일관성 검사가 거부 (주어진 자세 ↔ 착점 ↔ 플랫폼 중력)
    r = _at(ja, doc, tmp_path / "b", position="3G").run()
    assert r.status == "REFUSED" and "3G" in r.reason and "position and hit points disagree" in r.reason and r.n_move_l == 0, r.reason
    # 수직 배치 + 1G
    j3, doc3 = C.load_job(os.path.join(JOBS, "groove_10T_3G.yaml"))
    r = _at(j3, doc3, tmp_path / "c", position="1G").run()
    assert r.status == "REFUSED" and "1G" in r.reason and "position and hit points disagree" in r.reason and r.n_move_l == 0, r.reason


def test_leg_phase_3g_normal_moves_are_approach_and_retreat_normal():
    ph = AutoTeach._leg_phase
    assert ph(-0.02, 0.0, orientation="vertical_3G") == "approach_normal"
    assert ph(+0.02, 0.0, orientation="vertical_3G") == "retreat_normal"
    assert ph(-0.02, 0.01, orientation="vertical_3G") == "scan"
    assert ph(-0.02, 0.0) == "descend" and ph(+0.02, 0.0) == "ascend"


# --------------------------------------------------------------------------------------
# 납품 작업: 착점 오버스캔 리드, S→E 정렬
# --------------------------------------------------------------------------------------
def test_make_refined_job_leads_from_the_laser_hit_geometry_and_S_to_E_order():
    calib = C.load_calib(CALIB)
    j0, _doc = C.load_job(os.path.join(JOBS, "groove_10T_1G.yaml"))
    S, E = np.array([0.6, -0.19, 0.04]), np.array([0.6, 0.19, 0.04])
    c = 0.5 * (S + E)
    u_rev = unit(S - E)                                                        # 마지막 패스가 E→S 였다
    fit = PassFit(c=c, u=u_rev, n=np.array([0.0, 0.0, 1.0]), s_a=-0.19, s_b=0.19)
    fit.feature_medians = {"depth_mm": 10.0}
    fit.diag["ends"] = {"first": {"observed": True, "point": E.tolist()}, "last": {"observed": True, "point": S.tolist()}}
    job, info = make_refined_job(j0, fit, standoff_plate_m=0.34, pitch_deg=-5.0, calib=calib, overscan_m=OV, u_SE=E - S,
                                 spec=GrooveSpec(bevel_side="auto"))
    np.testing.assert_allclose(job.start, S, atol=1e-12)
    np.testing.assert_allclose(job.end, E, atol=1e-12)
    assert info["ends"] == {"start": "observed", "end": "observed"} and job.standoff_m == pytest.approx(0.35)
    ov = pass_overscan(job, calib, OV, top_offset_m=0.010)                     # 리드가 착점 오버스캔 20 mm 를 준다
    assert job.lead_in_m == pytest.approx(max(-ov["s_from"], 0.001)) and job.lead_out_m == pytest.approx(max(ov["s_to"] - job.length_m, 0.001))
    # the laser plane's tilt = the -5 deg pitch plus the calibration's own rotation about the laser line (w_x, eoat_lab_01 2026-10-12)
    R = Calib.lab_initial(eoat_yaw_rad=calib.eoat_yaw_rad, opt_yaw_branch=+1).T_flange_opt[:3, :3].T @ calib.T_flange_opt[:3, :3]
    w = math.degrees(math.atan2(R[2, 1], R[1, 1]))
    assert max(job.lead_in_m, job.lead_out_m) == pytest.approx(OV + 0.34 * math.tan(math.radians(5.0 - w)), abs=3e-4)
    assert min(job.lead_in_m, job.lead_out_m) <= 0.001 + 1e-12               # 피치 오프셋이 오버스캔보다 커서 한쪽은 리드 없이도 덮는다
    assert job.position == "1G"


def test_3g_lower_end_not_reachable_under_the_floor_rule_is_refused(tmp_path):
    """3G: 아래 끝 착점 오버스캔은 광학 원점 ≥ 아래 주어진 점 (바닥) 안에서. 아래 끝 목표가 바닥 근처면 계획대로; 바닥보다 한참
    아래면 (거친 아래 점을 판 끝보다 훨씬 위에 티칭) 오버스캔이 끝 판정이 요구하는 길이 미만 → REFUSED 'lower end not
    reachable under the 3G floor rule' (아무것도 보내지 않음)."""
    j3, doc3 = C.load_job(os.path.join(JOBS, "groove_10T_3G.yaml"))          # 위 0.68, 아래 0.30 (= 바닥)
    at = _at(j3, doc3, tmp_path)
    at.orientation, at.up = "vertical_3G", j3.up_unit
    at.job_c = make_coarse_job(j3, at._sv(), depth_m=at._depth_spec_m())
    at.job_ref = ScanJob.from_dict(C.job_to_dict(at.job_c))
    at.floor_h, at.floor_src = 0.30, "the lower rough point"                 # _gates 와 같이 (첫 패스 전 바닥 = 대략 아래 점)
    n = j3.normal_unit
    depth = 0.010

    def pass_job(lower_z: float) -> ScanJob:                                 # 루트 직선 (상면 − 깊이), 위 → 아래
        return ScanJob(start=np.array([0.5, 0.7, 0.68]) - n * depth, end=np.array([0.5, 0.7, lower_z]) - n * depth,
                       surface_normal=n, orientation="vertical_3G", standoff_m=at.pass_plate_m + depth, pitch_deg=-5.0,
                       incidence_tilt_deg=10.0, vertical_direction="down", progression_sign=1, lead_in_m=0.001,   # 토치 위쪽
                       lead_out_m=0.001, specimen_top_z_m=0.68, table_z_m=0.0)

    plan = PassPlan(direction="S->E", s_S=0.0, s_E=0.38, ov_S=OV, ov_E=OV)   # S = 위, E = 아래
    _job, ov = at._pass_overscan(pass_job(0.30), plan, 2, depth)
    assert ov["floor_limited"] == [] and ov["overscan_last_m"] == pytest.approx(OV)
    # 아래 끝을 아직 재지 않았으면 (바닥 = 대략 아래 점) 패스 번호와 무관하게 바닥이 허락하는 데까지만 간다 (4d P9: 상태로 판단)
    _job, ov2 = at._pass_overscan(pass_job(0.20), plan, 2, depth)
    assert ov2["floor_blocked"]
    at.floor_src = "the measured lower specimen end"                         # 아래 끝을 잰 뒤에는 거부
    with pytest.raises(_Abort, match="lower end not reachable under the 3G floor rule") as ei:
        at._pass_overscan(pass_job(0.20), plan, 2, depth)                          # 아래 끝 목표가 바닥보다 100 mm 아래
    assert ei.value.status == "REFUSED" and "nothing sent" in ei.value.reason
