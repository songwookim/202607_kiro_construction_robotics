"""auto-teach 리뷰 라운드 4 단위 테스트 (로봇·센서 없음):

  F1  3G 피치 부호: 툴 자세를 유지하면 아무것도 피치 부호를 정하지 않아 템플릿 배치 (S 위, 'down', progression +1) 에서 −5° 가 +10° 입사의
      절반을 상쇄했다. _pass_job 이 패스 1 에서 한 번 입사에 더해지는 쪽 (+5°) 을 고르고 다음 패스·정련 작업에 유지; make_refined_job 은
      3G 아래 끝 오버스캔을 pass_overscan(…, floor) 로 계산하고 부족하면 short_overscan 으로 알린다 (조용한 1 mm 리드 없음).
  F4  make_coarse_job 3G vertical_direction = S0/E0 높이에서 (S0 가 아래 점이면 'up' — 접근이 S0 로).
  F6  끝 식별 문턱은 설정이 아니라 유도 (2026-10-07 4b) — EndParams 는 k·단면 창·지연·유령점 버스트·판 최소 면 길이뿐.
  F7  판 끝 아웃라이어 문턱 = k × max(내부 잔차 강건 σ, 센서 양자화) (2026-10-07: 하한·상한 상수 없음): 1.2 mm 끝 아웃라이어는 빠진다.
"""
import math
import os

import numpy as np
import pytest
from conftest import CALIB, CONFIG, JOBS, STACK

from scan_core.frames import unit
from scan_core.localize import EndParams, PassFit
from scan_core.path_gen import laser_plane_line_param, sensor_frames
from scan_core.specimen import GrooveSpec
from scan_core.types import ScanJob
from scan_l0 import config as C
from scan_l0.session import ScanSession
from scan_l0.autoteach import (AutoTeach, PassPlan, SeamLine, make_coarse_job, make_refined_job, plate_edge_outliers)

CONFIG_KIRO = os.path.join(STACK, "hongik_scan_ros", "config", "hongik_kiro.yaml")   # KIRO 통합: kiro_right.yaml 대체 (같은 auto_teach 값)
DEPTH = 0.010


def _origin_above_hit_m(job: ScanJob, calib, depth_m: float = DEPTH) -> float:
    fr = sensor_frames(job, calib)
    q0 = fr.start + fr.n * depth_m
    d = float(laser_plane_line_param(fr.T_base_opt(0.0), q0, fr.u))
    return float(fr.up @ (fr.opt_origin(0.0) - (q0 + fr.u * d)))


def _rot_deg(Ra, Rb) -> float:
    R = np.asarray(Ra).T @ np.asarray(Rb)
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))


def _at_3g(tmp_path, job0: ScanJob, doc) -> AutoTeach:
    """로봇 플랫폼 없이 패스 작업만 만드는 AutoTeach (_gates 의 일부를 손으로)."""
    settings = C.load_settings(CONFIG)
    calib = C.load_calib(CALIB)
    at = AutoTeach(settings, job0, doc, calib, out_dir=str(tmp_path), log=lambda s: None,
                   session=ScanSession(robot=None, sensor=None, owned=False))
    at.orientation, at.up = job0.resolved_orientation(), job0.up_unit
    at.job_c = make_coarse_job(job0, at._sv(), depth_m=at._depth_spec_m())
    at.job_ref = ScanJob.from_dict(C.job_to_dict(at.job_c))
    at.u0, at.n0 = at.job_c.unit_dir, at.job_c.normal_unit
    return at


# --------------------------------------------------------------------------------------
# F1: 3G 피치 부호
# --------------------------------------------------------------------------------------
def test_f1_3g_pass_pitch_sign_is_chosen_once_to_add_to_the_incidence(tmp_path):
    """템플릿 배치 (groove_10T_3G.yaml: S = 위 점, 'down', progression +1, 피치 −5°, 입사 +10°): 패스 1 (E→S = 아래 → 위) 은 툴 자세에 가까운
    progression_sign (−1) 에서 피치 +5° 를 고른다 — 광학 원점이 착점보다 ≈ 90 mm 위 (−5° 면 ≈ 29 mm). 패스 2 (S→E) 는 같은 피치,
    progression_sign 만 뒤집혀 같은 물리 자세 (회전 < 0.1°)."""
    j3, doc3 = C.load_job(os.path.join(JOBS, "groove_10T_3G.yaml"))
    assert j3.vertical_direction == "down" and j3.progression_sign == 1 and j3.pitch_deg == -5.0 and j3.incidence_tilt_deg == 10.0
    at = _at_3g(tmp_path, j3, doc3)
    calib = at.calib
    fr_c = sensor_frames(at.job_c, calib)
    T_E = fr_c.T_base_flange(at.job_c.length_m)                              # 거친 E (아래 점) 의 거친 자세
    at._T_now = lambda: T_E
    n = j3.normal_unit
    S_root, E_root = j3.start - n * DEPTH, j3.end - n * DEPTH
    line = SeamLine(c=S_root, u=unit(E_root - S_root), n=n, s_S=0.020, s_E=0.350)
    plan1 = PassPlan(direction="E->S", s_S=0.0, s_E=0.350, ov_S=0.020, ov_E=0.0)
    job1 = at._pass_job(line, plan1, 1, DEPTH)
    assert job1.pitch_deg == 5.0 and at.pass_pitch_deg == 5.0 and job1.progression_sign == -1
    alt = ScanJob.from_dict({**C.job_to_dict(job1), "pitch_deg": -5.0})
    # 고른 부호는 피치를 입사에 더한다: 광학 원점이 착점보다 ~2·H·tan 5° 더 높다 (레이저 선 축 회전이 없는 캘리브 90 / 29 mm; eoat_lab_01
    # 2026-10-12 의 w_x −1.43° 는 이 토치 쪽에서 레이저 기울기를 1.43° 줄여 81 / 21 mm — 리뷰 재현값 29.3 mm 는 앞의 캘리브)
    above1, above_alt = _origin_above_hit_m(job1, calib), _origin_above_hit_m(alt, calib)
    assert above1 - above_alt > 0.05 and above1 > 0.070 and above_alt < 0.035
    dec = next(d for d in at.res.decisions if "pitch sign chosen once" in d["msg"])
    assert dec["pitch_deg"] == 5.0 and dec["alternative"]["pitch_deg"] == -5.0
    assert _rot_deg(T_E[:3, :3], sensor_frames(job1, calib).T_base_flange(0.0)[:3, :3]) < 6.0   # 피치 5° 만 (손목 180° 없음)
    # 패스 2: 같은 피치 (다시 고르지 않는다), progression 반전, 같은 물리 자세
    T_1 = sensor_frames(job1, calib).T_base_flange(job1.length_m)
    at._T_now = lambda: T_1
    plan2 = PassPlan(direction="S->E", s_S=0.0, s_E=0.380, ov_S=0.020, ov_E=0.020)
    job2 = at._pass_job(line, plan2, 2, DEPTH)
    assert job2.pitch_deg == 5.0 and job2.progression_sign == +1
    assert _rot_deg(T_1[:3, :3], sensor_frames(job2, calib).T_base_flange(0.0)[:3, :3]) < 0.1
    assert abs(_origin_above_hit_m(job2, calib) - _origin_above_hit_m(job1, calib)) < 1e-3
    assert sum("pitch sign chosen once" in d["msg"] for d in at.res.decisions) == 1


def test_f1_1g_pitch_is_unchanged(tmp_path):
    j1, doc1 = C.load_job(os.path.join(JOBS, "groove_10T_1G.yaml"))
    at = _at_3g(tmp_path, j1, doc1)
    at._T_now = lambda: sensor_frames(at.job_c, at.calib).T_base_flange(at.job_c.length_m)
    n = j1.normal_unit
    line = SeamLine(c=j1.start - n * DEPTH, u=j1.unit_dir, n=n, s_S=0.0, s_E=0.320)
    job = at._pass_job(line, PassPlan(direction="E->S", s_S=-0.03, s_E=0.32, ov_S=0.02, ov_E=0.0), 1, DEPTH)
    assert job.pitch_deg == -5.0 and at.pass_pitch_deg is None


def _fit_3g(j3: ScanJob) -> PassFit:
    n = j3.normal_unit
    S, E = j3.start - n * DEPTH, j3.end - n * DEPTH                           # 위, 아래 (루트)
    fit = PassFit(c=0.5 * (S + E), u=unit(E - S), n=n, s_a=-0.19, s_b=0.19)
    fit.feature_medians = {"depth_mm": 10.0}
    fit.diag["ends"] = {"first": {"observed": True, "point": S.tolist()}, "last": {"observed": True, "point": E.tolist()}}
    return fit


@pytest.mark.parametrize("pitch,short", [(5.0, False), (-5.0, True)])
def test_f1_make_refined_job_3g_lower_end_overscan_is_computed_and_flagged(pitch, short):
    """정련 작업 (3G S 위 → E 아래, 'down', progression +1): 아래 끝에서는 광학 원점이 아래 점에서 멈춘다 (R4) — 아래 끝 착점 오버스캔은
    구조적 값. 입사에 더해지는 피치 (+5°) 면 착점이 아래 끝을 지나 (≥ 계획, 표시 없음), 상쇄하는 피치 (−5°) 면 착점이 아래 끝 위에서
    멈춘다 → info['short_overscan'] (예전에는 조용한 1 mm 리드). 구조적 값: 레이저 선 축 회전이 없는 캘리브 +29 / −31.5 mm, eoat_lab_01
    2026-10-12 (w_x −1.43°, 피벗 334 mm 로 광학 원점이 진행 방향 8.3 mm) +20.0 / −39.9 mm — 계획은 그 아래 15 mm."""
    calib = C.load_calib(CALIB)
    j3, _doc = C.load_job(os.path.join(JOBS, "groove_10T_3G.yaml"))
    job, info = make_refined_job(j3, _fit_3g(j3), standoff_plate_m=0.34, pitch_deg=pitch, calib=calib, overscan_m=0.015,
                                 u_SE=j3.end - j3.start, spec=GrooveSpec(bevel_side="auto"))
    assert job.vertical_direction == "down" and job.pitch_deg == pitch and job.progression_sign == 1
    low = info["lead"]["lower_end"]
    assert low["end"] == "end" and low["planned_m"] == 0.015 and low["floor_h"] == pytest.approx(float(job.height(job.end)))
    if short:
        assert low["hit_overscan_m"] < -0.025 and low["lead_needed_m"] > 0.04
        assert info["short_overscan"][0]["end"] == "end" and info["short_overscan"][0]["hit_overscan_m"] < 0.0
    else:
        assert low["hit_overscan_m"] >= 0.015 and low["lead_needed_m"] == 0.0 and "short_overscan" not in info
    assert job.lead_out_m == 0.001                                            # 아래 점 쪽 리드는 쓰이지 않는다 (3G R4)
    # 위 끝: 착점 오버스캔 리드 (+5°: 착점이 원점보다 아래(진행 쪽) 라 위 끝 앞 20 mm 를 위해 리드가 필요; −5°: 착점이 이미 31.5 mm 위)
    assert (job.lead_in_m > 0.001) if not short else (job.lead_in_m == 0.001)


# --------------------------------------------------------------------------------------
# F4: 3G 거친 작업의 진행 방향
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("s0_lower,vd_file,vd_expected", [(True, "down", "up"), (False, "up", "down"), (False, "down", "down"),
                                                        (True, "up", "up")])
def test_f4_make_coarse_job_3g_vertical_direction_from_the_rough_heights(s0_lower, vd_file, vd_expected):
    j3, _doc = C.load_job(os.path.join(JOBS, "groove_10T_3G.yaml"))
    hi, lo = np.array(j3.start), np.array(j3.end)
    S0, E0 = (lo, hi) if s0_lower else (hi, lo)
    job0 = ScanJob.from_dict({**C.job_to_dict(j3), "start": S0.tolist(), "end": E0.tolist(), "vertical_direction": vd_file})
    from scan_l0.sensor_view import sensor_view
    jc = make_coarse_job(job0, sensor_view(C.load_settings(CONFIG).z_valid_mm), depth_m=0.010)
    assert jc.vertical_direction == vd_expected
    fr = sensor_frames(jc, C.load_calib(CALIB))
    assert fr.travel_sign == 1 and fr.s_scan_from <= 0.0                       # 진행은 S0 → E0, 스캔 시작 = S0 (접근 대상)
    np.testing.assert_allclose(fr.seam_point(0.0), S0, atol=1e-9)


# --------------------------------------------------------------------------------------
# F6: 설정 (두 플랫폼 키 동일)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [CONFIG, CONFIG_KIRO])
def test_f6_end_detection_is_derived_not_configured(path):
    """판 끝 식별·RANSAC 문턱은 설정이 아니다 (사용자 2026-10-07 4b): 남은 설정은 흐름 수치뿐, 예전 키는 거부한다. 끝 식별 입력
    (EndParams) 은 다른 곳에서 정해진 값 — k (judgement.k), 단면 창 (weld_plan), 지연, 유령점 버스트 (sensor), 판 최소 면 길이."""
    s = C.load_settings(path)
    at = s.auto_teach_config()
    assert not hasattr(at, "end_detection") and not hasattr(at, "plate_plane_ransac_thr_m") and not hasattr(at, "min_site_distance_m")
    assert set(EndParams.__dataclass_fields__) == {"k", "window_m", "lag_m", "ghost_burst_frames", "plate_min_len_m"}
    assert s.ghost_burst_frames == 2 and EndParams().confirm_profiles == 3
    p = EndParams.from_dict({"k": 3.0, "window_m": 0.010, "lag_m": 0.007, "ghost_burst_frames": 2, "plate_min_len_m": 0.005})
    assert p.look_m == pytest.approx(0.017) and p.to_dict()["ghost_burst_frames"] == 2
    for key in ("end_detection", "plate_plane_ransac_thr_m"):
        with pytest.raises(ValueError, match=key):
            C.AutoTeachConfig.from_params({key: {} if key == "end_detection" else 0.0005})


# --------------------------------------------------------------------------------------
# F7: 적응 판 끝 아웃라이어 문턱
# --------------------------------------------------------------------------------------
def _noisy_line(sigma_m: float, n: int = 201, seed: int = 7):
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 0.38, n)
    P = np.column_stack([np.full(n, 0.6), s, np.full(n, 0.04)])
    P[:, 0] += rng.normal(0.0, sigma_m, n)                                    # 측면 잡음
    P[:, 2] += rng.normal(0.0, sigma_m, n)                                    # 판 법선 잡음
    fit = PassFit(c=np.array([0.6, 0.0, 0.04]), u=np.array([0.0, 1.0, 0.0]), n=np.array([0.0, 0.0, 1.0]), s_a=0.0, s_b=0.38)
    return P, fit


def test_f7_edge_outlier_threshold_is_adaptive():
    """내부 잔차 σ = 0.05 mm (2 성분): 문턱 = k · max(σ̂, 양자화 하한) = 3 · 0.05 = 0.15 mm (원칙 1, 2026-10-07 — 예전 5σ·하한 0.3 mm·
    상한 2 mm 는 없앴다). 끝 구역의 1.2 mm 아웃라이어는 빠지고 가운데의 큰 이탈 (실제 형상) 은 남는다. σ = 0.1 mm 면 문턱 ≈ 0.3 mm,
    σ = 1 mm 면 ≈ 3 mm (상한 없음 — 측정이 그만큼 흩어졌다), 내부 점이 없으면 전체 점의 σ."""
    sig = 0.05e-3
    P, fit = _noisy_line(sig)
    P[0, 0] = 0.6 + 1.2e-3                                                     # 판 끝 (첫 프로파일) 1.2 mm 아웃라이어
    P[100, 0] = 0.6 + 4.0e-3                                                   # 가운데 4 mm (판 끝 아님: 절대 빼지 않는다)
    info = {}
    m = plate_edge_outliers(P, fit, edge_m=0.020, floor_m=1e-6, diag=info)
    assert m[0] and not m[100]
    assert info["rule"] == "adaptive" and info["thr_m"] == pytest.approx(3.0 * info["sigma_m"])
    assert 0.7 * sig < info["sigma_m"] < 1.3 * sig and info["n_interior"] > 150
    assert not plate_edge_outliers(P, fit, thr_m=2.0e-3)[0]                   # 예전 고정 2 mm 문턱은 1.2 mm 를 남겼다
    P2, fit2 = _noisy_line(0.1e-3)
    P2[1, 0], P2[2, 0] = 0.6 + 0.2e-3, 0.6 + 0.6e-3
    info2 = {}
    m2 = plate_edge_outliers(P2, fit2, edge_m=0.020, floor_m=1e-6, diag=info2)
    assert 0.2e-3 < info2["thr_m"] < 0.4e-3 and not m2[1] and m2[2]
    P3, fit3 = _noisy_line(1.0e-3)
    info3 = {}
    plate_edge_outliers(P3, fit3, edge_m=0.020, floor_m=1e-6, diag=info3)
    assert info3["thr_m"] > 2.0e-3                                            # 상한 없음
    P4, fit4 = _noisy_line(sig, n=5)
    info4 = {}
    plate_edge_outliers(P4, fit4, edge_m=0.20, floor_m=1e-6, diag=info4)     # 내부 점 없음 → 전체 점의 σ
    assert info4["rule"].startswith("adaptive (all points") and np.isfinite(info4["thr_m"])
    info5 = {}
    plate_edge_outliers(P, fit, edge_m=0.020, floor_m=1.0e-3, k=2.0, diag=info5)   # 하한 (센서 양자화) 이 σ 보다 크면 하한
    assert info5["thr_m"] == pytest.approx(2.0e-3)
