"""2026-10-01 auto-teach 재구성의 scan_core 부분 (사용자 결정 1~3차):

  * 착점(laser hit point) 오버스캔 기하 (path_gen.laser_plane_line_param / pass_overscan): 피치 −5° 0.34 m 에서 착점은 광학 원점의
    심 점에서 ≈ 0.34·tan5° = 29.7 mm 떨어지고, 툴 자세를 유지한 채 진행 방향을 뒤집으면 (progression_sign 반전) 그 부호가 바뀐다 —
    두 방향 모두 착점이 각 끝을 정확히 20 mm 지나도록 원점 시작/끝이 정해진다. 3G (+10° 입사) 아래 끝은 바닥 규칙으로 줄어든다.
  * identify_seam_ends: 모델 기반 판 끝 (on-specimen 판정, 가운데에서 키우는 연속 구간, 판 모서리 괄호) — 유한 시편 합성 스캔에서
    진짜 끝의 ± 간격/2 안 (피치 −5° 의 상면/바닥 착점 차에 편향 없음); 스캔이 끝 너머로 충분히 가지 않으면 미관측.
  * fidelity_gate('fidelity'): 충실도 우선 (양끝·길이·치수·바닥·루트 모서리·일관성), 센서 배치는 진단 전용.
  * 용접 자세 입력: position '1G'|'3G' ↔ orientation, 방향 검사 메시지가 주어진 자세와 플랫폼 중력축을 쓴다.
"""
import dataclasses
import math

import numpy as np
import pytest

from scan_core.frames import make_transform, unit
from scan_core.localize import (EndParams, fidelity_gate, identify_seam_ends, plate_points_from_scan,
                                polyline_from_scan_features, refine_from_pass)
from scan_core.path_gen import laser_plane_line_param, pass_overscan, sensor_frames
from scan_core.reconstruct import reconstruct_scan
from scan_core.seam_features import SeamScanFeatures, extract_seam_features_along_scan
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, simulate_scan
from scan_core.types import JobError, ScanJob, orientation_from_position, position_from_orientation

DEPTH = 0.010
STANDOFF_ROOT = 0.34 + DEPTH          # 패스 job standoff = 판 상면 0.34 + 그루브 깊이 (시점/종점 = 루트 점)


def _job_1g(direction: int, prog: int, **kw) -> ScanJob:
    S = np.array([0.6, -0.19, 0.04])
    E = np.array([0.6, 0.19, 0.04])
    a, b = (S, E) if direction > 0 else (E, S)
    d = dict(start=a, end=b, surface_normal=[0, 0, 1], orientation="flat_1G", standoff_m=STANDOFF_ROOT, pitch_deg=-5.0,
             progression_sign=prog, lead_in_m=0.001, lead_out_m=0.001, table_z_m=-0.05)
    d.update(kw)
    return ScanJob(**d)


def _job_3g(direction: int, prog: int, pitch_deg: float = -5.0) -> ScanJob:
    lo = np.array([0.5, 0.70, 0.30])
    hi = np.array([0.5, 0.70, 0.68])
    a, b = (lo, hi) if direction > 0 else (hi, lo)
    return ScanJob(start=a, end=b, surface_normal=[0, -1, 0], orientation="vertical_3G", standoff_m=STANDOFF_ROOT, pitch_deg=pitch_deg,
                   incidence_tilt_deg=10.0, progression_sign=prog, vertical_direction="up" if direction > 0 else "down",
                   lead_in_m=0.001, lead_out_m=0.001, specimen_top_z_m=0.68, table_z_m=0.0)


def _top_hit(T_base_opt, job: ScanJob) -> float:
    """독립 계산: 레이저 평면 (원점, 법선 Y_opt) ∩ 판 상면 심 직선 (start + n·깊이 + u·t) → t."""
    T = np.asarray(T_base_opt, dtype=float)
    o, Y = T[:3, 3], T[:3, 1]
    q0 = job.start + job.normal_unit * DEPTH
    u = job.unit_dir
    return float((o - q0) @ Y / (u @ Y))


# --------------------------------------------------------------------------------------
# 착점 기하
# --------------------------------------------------------------------------------------
def test_laser_plane_line_param_matches_a_hand_computed_hit():
    # 광학 자세: 원점 (0, 0, 0.34), Z_opt 아래, 라인축 X_opt = base x, 라인 평면 법선 Y_opt = base y 를 피치 5° 기울임
    a = math.radians(5.0)
    R = np.column_stack([[1.0, 0.0, 0.0], [0.0, math.cos(a), math.sin(a)], [0.0, math.sin(a), -math.cos(a)]])
    T = make_transform(R, [0.0, 0.0, 0.34])
    t = laser_plane_line_param(T, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])         # 판 상면 위 y 축 직선
    assert t == pytest.approx(0.34 * math.tan(a), abs=1e-12)                 # 착점 = 원점 발에서 standoff·tan(pitch)
    Ts = np.stack([T, make_transform(R, [0.0, 0.05, 0.34])])
    np.testing.assert_allclose(laser_plane_line_param(Ts, [0, 0, 0], [0, 1, 0]), [t, t + 0.05], atol=1e-12)
    assert np.isnan(laser_plane_line_param(T, [0, 0, 0], [1.0, 0.0, 0.0]))   # 직선 ∥ 레이저 평면


def test_pass_overscan_1g_pitch5_hit_point_both_directions_with_the_tool_kept(calib_lab):
    """1G, 피치 −5°, 판 상면 위 0.34 m: 착점 오프셋 |δ| = 0.34·tan5° = 29.75 mm. 툴 자세 유지 (정방향 prog +1 ↔ 역방향 prog −1) 면
    회전 행렬이 같고 δ 의 부호가 바뀐다; 두 방향 모두 착점이 시작 끝 20 mm 앞에서 끝 끝 20 mm 뒤까지 간다 (독립 교점 계산으로 확인)."""
    fwd = pass_overscan(_job_1g(+1, +1, ), calib_lab, 0.020, top_offset_m=DEPTH)
    rev = pass_overscan(_job_1g(-1, -1), calib_lab, 0.020, top_offset_m=DEPTH)
    d0 = 0.34 * math.tan(math.radians(5.0))
    assert abs(fwd["hit_offset_m"]) == pytest.approx(d0, abs=2e-4) and fwd["hit_offset_m"] == pytest.approx(-rev["hit_offset_m"], abs=1e-9)
    np.testing.assert_allclose(fwd["T_from"][:3, :3], rev["T_from"][:3, :3], atol=1e-12)          # 같은 툴 자세 (손목 180° 없음)
    for r, job in ((fwd, _job_1g(+1, +1)), (rev, _job_1g(-1, -1))):
        fr = sensor_frames(job, calib_lab)
        assert r["hit_from"] == pytest.approx(-0.020, abs=1e-9) and r["hit_to"] == pytest.approx(fr.L + 0.020, abs=1e-9)
        assert _top_hit(fr.T_base_opt(r["s_from"]), job) == pytest.approx(-0.020, abs=1e-9)
        assert _top_hit(fr.T_base_opt(r["s_to"]), job) == pytest.approx(fr.L + 0.020, abs=1e-9)
        # 광학 원점은 착점이 아니다: 원점의 심 점은 δ 만큼 반대쪽 (원점 기준이었다면 오버스캔이 20 − 29.7 < 0 인 끝이 생긴다)
        assert r["s_from"] == pytest.approx(-0.020 - r["hit_offset_m"], abs=1e-12)
        np.testing.assert_allclose(r["hit_point_from"], job.start + job.normal_unit * DEPTH - job.unit_dir * 0.020, atol=1e-9)
    # 같은 방향에서 툴을 180° 돌리면 (prog −1) 착점이 반대쪽: 원점 범위가 그만큼 옮겨진다
    alt = pass_overscan(_job_1g(+1, -1), calib_lab, 0.020, top_offset_m=DEPTH)
    assert alt["hit_offset_m"] == pytest.approx(-fwd["hit_offset_m"], abs=1e-9)
    # 끝마다 다른 오버스캔 (0 = 판 안쪽 점에서 정확히 시작)
    r0 = pass_overscan(_job_1g(-1, -1), calib_lab, 0.020, top_offset_m=DEPTH, overscan_first_m=0.0)
    assert r0["hit_from"] == pytest.approx(0.0, abs=1e-12) and r0["overscan_first_m"] == 0.0 and r0["overscan_last_m"] == 0.020


def test_pass_overscan_3g_incidence_and_the_floor_rule(calib_lab):
    """3G (+10° 내려다봄, 피치 −5°): 툴 자세 유지 (위→아래 prog −1 ↔ 아래→위 prog +1… 같은 회전) 에서 위·아래 끝 모두 20 mm. 광학
    원점은 아래 끝 착점보다 위에 있다 (입사 기울기). 바닥(아래 주어진 점) 을 올리면 아래 끝 오버스캔이 줄고, 필요한 길이
    (min_overscan_m — auto-teach 는 끝 판정이 요구하는 길이; 없으면 계획 그대로) 미만이면 floor_blocked (호출자가 'lower end not
    reachable under the 3G floor rule' 로 거부)."""
    # 토치가 레이저 위쪽 + 피치가 입사에 더해지는 자세 (auto-teach 가 고르는 자세; 3G 바닥은 토치 팁도 본다, 2026-10-05)
    down = pass_overscan(_job_3g(-1, +1, 5.0), calib_lab, 0.020, top_offset_m=DEPTH, floor_h=0.30)
    up = pass_overscan(_job_3g(+1, -1, 5.0), calib_lab, 0.020, top_offset_m=DEPTH, floor_h=0.30)
    np.testing.assert_allclose(down["T_from"][:3, :3], up["T_from"][:3, :3], atol=1e-12)
    for r in (down, up):
        assert r["floor_limited"] == [] and r["overscan_first_m"] == pytest.approx(0.020) and r["overscan_last_m"] == pytest.approx(0.020)
    o_low = down["origin_to"]                                                  # 아래 끝의 광학 원점
    assert float(o_low[2]) > 0.30 + 0.050                                      # 착점(0.28) 보다 ≈ 0.09 m 위 → 바닥 0.30 위
    # 바닥을 아래 끝보다 70 mm 위로: 줄어들지만 필요한 10 mm 이상이면 허용 (limited), 90 mm 위면 막힘 (blocked)
    lim = pass_overscan(dataclasses.replace(_job_3g(-1, +1, 5.0), pose_tol_m=0.002), calib_lab, 0.020, top_offset_m=DEPTH, floor_h=0.37,
                        min_overscan_m=0.010)
    assert lim["floor_limited"] == ["last"] and 0.010 <= lim["overscan_last_m"] < 0.020 and lim["floor_blocked"] == []
    strict = pass_overscan(dataclasses.replace(_job_3g(-1, +1, 5.0), pose_tol_m=0.002), calib_lab, 0.020, top_offset_m=DEPTH, floor_h=0.37)
    assert strict["floor_blocked"] == ["last"]                                 # 필요한 길이를 안 주면 계획 그대로가 필요
    assert float(lim["origin_to"][2]) == pytest.approx(0.37 + 0.002, abs=1e-9)        # 바닥 + 도착 허용 (job.pose_tol_m)
    blk = pass_overscan(_job_3g(-1, +1, 5.0), calib_lab, 0.020, top_offset_m=DEPTH, floor_h=0.39, min_overscan_m=0.010)
    below = pass_overscan(_job_3g(-1, -1, -5.0), calib_lab, 0.020, top_offset_m=DEPTH, floor_h=0.30)     # 토치가 아래쪽:
    assert below["floor_keypoint_below_origin_m"] > 0.05 and below["floor_limited"] == ["last"]          # 팁이 바닥을 먼저 만난다
    assert blk["floor_blocked"] == ["last"] and blk["overscan_last_m"] < 0.010
    with pytest.raises(ValueError, match="travel_sign"):
        bad = _job_3g(-1, -1)
        bad.vertical_direction = "up"                                          # 위 점에서 시작하는데 'up' → 역방향 진행
        pass_overscan(bad, calib_lab, 0.020)


# --------------------------------------------------------------------------------------
# identify_seam_ends: 모델 기반 양끝 (유한 시편 합성 스캔, 피치 −5°)
# --------------------------------------------------------------------------------------
def test_identify_seam_ends_finds_both_plate_ends_without_pitch_bias(calib_lab):
    spec = GrooveSpec(bevel_side="left", weld_length_mm=100.0)
    scene = GrooveScene.from_spec(spec, centered=False)                        # x ∈ [0, 0.100] (판 프레임, 원점 = job.start)
    job = ScanJob(start=[0.6, -0.05, 0.05], end=[0.6, 0.05, 0.05], surface_normal=[0, 0, 1], table_z_m=-0.05, specimen_top_z_m=0.05,
                  lead_in_m=0.07, lead_out_m=0.07, speed_mps=0.04, accel_mps2=0.5, profile_fps=20.0, pitch_deg=-5.0, standoff_m=0.34,
                  orientation="flat_1G", job_id="extent_syn")
    syn = simulate_scan(job, calib_lab, scene, SynthConfig(n_points=704, seed=3, sigma_z_mm=0.02, dropout_prob=0.01,
                                                           robot_poll_rate_hz=0.0))
    res = reconstruct_scan(syn.frames, syn.robot, calib_lab, job, fps_nominal=20.0)
    spec_a = GrooveSpec(bevel_side="auto", weld_length_mm=100.0)
    sf = extract_seam_features_along_scan(res, spec_a, centre="root", keep_features=True)
    P, gf = polyline_from_scan_features(res, sf)
    Q = plate_points_from_scan(res, sf.features, sf.indices)
    fit = refine_from_pass(P, Q, gf, sf, None, u_prior=job.unit_dir, n_prior=[0, 0, 1])
    assert fit.ok and "plate_plane_n" in fit.diag
    identify_seam_ends(res, sf, fit, P, gf, spec_a, EndParams())
    ends = fit.diag["ends"]
    assert ends["first"]["observed"] and ends["last"]["observed"] and not any(f.startswith("end_not_observed") for f in fit.flags)
    sp = fit.diag["spacing_m"]
    assert 0.0015 < sp < 0.0025                                                 # 0.04 m/s @ 20 fps
    S_true = job.start - np.array([0, 0, spec.thickness_mm * 1e-3])            # 루트 직선 위 진짜 끝 (끝면은 수직: 상면 끝과 같은 u)
    E_true = job.end - np.array([0, 0, spec.thickness_mm * 1e-3])
    tol = sp / 2 + 0.2e-3                                                       # 괄호의 중점 ± 간격/2 (피치 착점 차는 괄호가 흡수)
    assert np.linalg.norm(np.asarray(ends["first"]["point"]) - S_true) < tol
    assert np.linalg.norm(np.asarray(ends["last"]["point"]) - E_true) < tol
    assert abs((fit.s_b - fit.s_a) - 0.100) < 2 * tol
    assert not ends["last"]["occluded"] and ends["last"]["off_specimen_span_m"] >= 0.005 and "plates_staggered" not in fit.flags
    assert ends["last"]["alt_point"] is not None                                 # 설계 길이로 본 대안 끝 (보고)
    # 착점이 아니라 광학 원점 s 로 끝을 정했다면 피치 오프셋 (0.34·tan5 = 30 mm) 만큼 어긋났을 것이다
    i_last = ends["last"]["profile_inside"]
    s_origin = float(res.s_m[sf.indices[i_last]])
    assert abs(s_origin - 0.100) > 0.020
    # 스캔이 끝 너머까지 가지 않으면 '미관측' (구간 밖 프로파일 < 3 또는 < 5 mm)
    M = P.shape[0]
    cut = ends["last"]["profile_inside"] + 2
    sf2 = dataclasses.replace(sf, indices=sf.indices[:cut], features=sf.features[:cut], ok=sf.ok[:cut],
                              root_gap_mm=sf.root_gap_mm[:cut], depth_mm=sf.depth_mm[:cut], top_opening_mm=sf.top_opening_mm[:cut])
    fit2 = refine_from_pass(P[:cut], Q, gf[:cut], None, None, u_prior=job.unit_dir, n_prior=[0, 0, 1])
    identify_seam_ends(res, sf2, fit2, P[:cut], gf[:cut], spec_a, EndParams())
    assert "end_not_observed_last" in fit2.flags and not fit2.diag["ends"]["last"]["observed"]
    assert fit2.diag["ends"]["first"]["observed"] and cut < M


# --------------------------------------------------------------------------------------
# 충실도 우선 게이트 (판정 기준 = scan_core.measure 원칙: k·σ, 설계 ±50 %, 용접 단면 충분성 — 사용자 2026-10-07)
# --------------------------------------------------------------------------------------
GOOD = {"root_gap_mm": 5.05, "top_opening_mm": 10.8, "depth_mm": 10.02, "bevel_angle_deg": 29.8, "hi_lo_mm": 0.1,
        "root_gap_robust_std_mm": 0.05, "depth_robust_std_mm": 0.03, "wall_shear_k": float("nan"), "fov_margin_mm": 60.0,
        "sensor_z_mm": 341.0}
EXT = {"first_observed": True, "last_observed": True, "length_mm": 380.4, "spacing_mm": 0.3}
EXT0 = dict(EXT, length_mm=380.0)                                               # 설계 길이 그대로 (길이 허용치 없는 시험)
BAD_PLACEMENT = {"psi_deg": 4.0, "theta_deg": 3.0, "roll_deg": 2.5, "lateral_m": 0.012, "standoff_m": 0.009}
QC = {"root_gap_mm": {"tol": 0.5}, "groove_depth_mm": {"tol": 0.5}, "top_opening_mm": {"tol": 0.5}, "bevel_angle_deg": {"tol": 2.0},
      "hi_lo_mm": {"target_max": 1.0}, "weld_length_mm": {"tol": 1.0}}        # 작업 파일의 설계 허용치 (설계 정보)
VALID = {"z_valid_mm": (233.1, 544.0), "pose_tol_mm": 2.1, "plate_min_mm": 5.0}


def _cov(ok=True, n=30, kinds=("groove", "floor", "root_edges")):
    return {kk: {"n_stations": 77, "min_count": n if ok else 2, "median_count": n, "short": [] if ok else [(0.12, 2)], "ok": ok,
                 "need": 5} for kk in kinds}


def test_fidelity_gate_pass_needs_shape_and_length_not_placement():
    spec = GrooveSpec(bevel_side="auto")
    kw = dict(extent=EXT, qc_targets=QC, coverage=_cov(), validity=VALID)
    v = fidelity_gate(GOOD, BAD_PLACEMENT, spec, "fidelity", **kw)
    assert v.status == "PASS", v.summary()
    for k in ("placement_psi_deg", "placement_roll_deg", "placement_lateral_m", "placement_standoff_m", "wall_shear_k"):
        assert v.checks[k][2] is None                                          # 진단 전용
    assert v.checks["placement_psi_deg"][0] == 4.0 and v.checks["weld_length_mm"][2] is True
    assert fidelity_gate(dict(GOOD, wall_shear_k=0.2), BAD_PLACEMENT, spec, "fidelity", **kw).status == "PASS"
    with pytest.raises(ValueError, match="only stage 'fidelity'"):
        fidelity_gate(GOOD, {}, spec, "fine", extent=EXT)
    with pytest.raises(ValueError, match="specimen design"):
        fidelity_gate(GOOD, {}, None, "fidelity", extent=EXT)
    # 끝 미관측 → DATA_FAIL (길이는 평가하지 않는다)
    v = fidelity_gate(GOOD, {}, spec, "fidelity", **dict(kw, extent=dict(EXT, last_observed=False)))
    assert v.status == "DATA_FAIL" and v.failed == ["end_last_observed"] and v.checks["weld_length_mm"][2] is None
    assert fidelity_gate(GOOD, {}, spec, "fidelity", **dict(kw, extent=None)).status == "DATA_FAIL"     # extent 없음
    # 길이: 설계 허용치 (작업 1 mm) 안 = PASS, 그 위 k·√2·σ_끝 (σ_끝 = 착점 간격/√12) 까지 보고, 넘으면 'end_ambiguous' + 대안 끝
    for sp in (0.3, 2.0):
        amb = 1.0 + 3.0 * math.sqrt(2.0) * sp / math.sqrt(12.0)
        ext = dict(EXT, spacing_mm=sp, first_alt=[0.6, -0.19, 0.04], last_alt=[0.6, 0.19, 0.04])
        v = fidelity_gate(GOOD, {}, spec, "fidelity", **dict(kw, extent=dict(ext, length_mm=380.0 + amb - 0.02)))
        assert v.status == "PASS" and v.checks["weld_length_mm"][2] is None and "end_ambiguous" not in v.checks, (sp, v.summary())
        assert any("reported, measured ends kept" in n for n in v.notes)
        v = fidelity_gate(GOOD, {}, spec, "fidelity", **dict(kw, extent=dict(ext, length_mm=380.0 + amb + 0.02)))
        assert v.status == "SPEC_FAIL" and "end_ambiguous" in v.failed and any("alternative ends" in n for n in v.notes)
    v = fidelity_gate(GOOD, {}, spec, "fidelity", **dict(kw, qc_targets={}, extent=dict(EXT, length_mm=380.3)))
    assert v.status == "PASS" and v.checks["weld_length_mm"][2] is None             # 허용치 없음: 끝 판독 산포 안이면 보고만
    # 보고 대역 (사용자 2026-10-06, qc_targets weld_length_mm.ambiguous 5 mm): 2026-10-08 1G 실기 — 판 끝 엇갈림 2.5 mm, 판 길이 381.44 mm
    # (tol + k·σ = 1.36 mm 를 넘어 SPEC_FAIL 로 멈췄다) → 보고만, 5 mm 넘으면 end_ambiguous
    qc5 = dict(QC, weld_length_mm={"tol": 1.0, "ambiguous": 5.0})
    for L, ok in ((381.44, True), (378.6, True), (384.9, True), (385.2, False), (374.0, False)):
        v = fidelity_gate(GOOD, {}, spec, "fidelity", **dict(kw, qc_targets=qc5, extent=dict(EXT, length_mm=L, first_alt=[0, 0, 0],
                                                                                              last_alt=[0, 0, 0])))
        if ok:
            assert v.status == "PASS" and "end_ambiguous" not in v.checks, (L, v.summary())
            assert any("report band qc_targets weld_length_mm.ambiguous 5 mm" in n for n in v.notes), v.notes
        else:
            assert v.status == "SPEC_FAIL" and "end_ambiguous" in v.failed, (L, v.summary())
            assert any("max of the report band 5 mm" in n for n in v.notes)
    # 판 위 물체가 끝을 가림: 기본은 PASS 아님 (DATA_FAIL), accept_inferred_end 면 추정 끝으로
    occ = dict(EXT, last_observed=False, occluded=["last"], last_inferred=[0.6, 0.19, 0.04], length_mm=380.0)
    v = fidelity_gate(GOOD, {}, spec, "fidelity", **dict(kw, extent=occ))
    assert v.status == "DATA_FAIL" and "end_occluded_last" in v.failed and any("end_occluded" in n for n in v.notes)
    v = fidelity_gate(GOOD, {}, spec, "fidelity", **dict(kw, extent=occ), accept_inferred_end=True)
    assert v.status == "PASS" and v.checks["weld_length_mm"][2] is None and any("inferred" in n for n in v.notes)
    # 시편 대비 설계 (작업 허용치) 는 보고만, 설계 공칭 ±50 % 안이면 측정을 믿는다 (예전 1 mm 과도 편차 정지 없음)
    for k, val in (("depth_mm", 11.2), ("root_gap_mm", 6.2), ("bevel_angle_deg", 32.5), ("top_opening_mm", 12.5), ("hi_lo_mm", 1.2)):
        v = fidelity_gate(dict(GOOD, **{k: val}), {}, spec, "fidelity", **kw)
        assert v.status == "PASS" and v.specimen[k]["within"] is False, (k, v.summary())
        assert any("specimen deviates from the design (reported, not a scan failure)" in n and k in n for n in v.notes), v.notes
    v = fidelity_gate(GOOD, {}, spec, "fidelity", **kw)
    assert not any("specimen deviates" in n for n in v.notes)
    g = v.specimen["root_gap_mm"]
    assert (g["median"], g["design"], g["tol"], g["within"]) == (5.05, 5.0, 0.5, True) and "along_within" not in g
    assert v.checks["root_gap_mm"] == (5.05, (2.5, 7.5), True) and v.checks["bevel_angle_deg"][2] is None
    # 설계 공칭 ±50 % 밖 = 특징 오인식 → SPEC_FAIL
    for k, val in (("depth_mm", 15.2), ("root_gap_mm", 2.4), ("top_opening_mm", 16.3)):
        v = fidelity_gate(dict(GOOD, **{k: val}), {}, spec, "fidelity", **kw)
        assert v.status == "SPEC_FAIL" and v.failed == [k] and any("misidentified feature" in n for n in v.notes), (k, v.summary())
    # 단면 충분성: 용접 경로 단면 창에 유효 프로파일이 모자라면 DATA_FAIL (모자란 단면을 노트에)
    v = fidelity_gate(GOOD, {}, spec, "fidelity", **dict(kw, coverage=_cov(False)))
    assert v.status == "DATA_FAIL" and set(v.failed) == {"coverage_groove", "coverage_floor", "coverage_root_edges"}
    assert any("weld-plan sections have fewer than 5 valid profiles" in n for n in v.notes)
    v = fidelity_gate(dict(GOOD, depth_robust_std_mm=0.4), {}, spec, "fidelity", **kw)
    assert v.status == "PASS" and v.checks["depth_robust_std_mm"] == (0.4, None, None)   # 심 방향 강건 표준편차는 보고만
    # 측정이 설계 허용치를 구분하나: k × 단면 중앙값 불확도 (√(π/2)·σ/√n, n = 단면 최소 30) ≤ 허용치 0.5
    lim = 0.5 / (3.0 * math.sqrt(math.pi / 2.0) / math.sqrt(30))
    for k in ("root_gap_noise_mm", "depth_noise_mm", "top_opening_noise_mm"):
        v = fidelity_gate(dict(GOOD, **{k: lim * 1.02}), {}, spec, "fidelity", **kw)
        assert v.status == "SPEC_FAIL" and v.failed == [k], (k, v.summary())
        assert fidelity_gate(dict(GOOD, **{k: lim * 0.98}), {}, spec, "fidelity", **kw).status == "PASS"
        assert fidelity_gate(dict(GOOD, **{k: lim * 1.02}), {}, spec, "fidelity", **dict(kw, qc_targets={}, extent=EXT0)).status == "PASS"
    assert v.checks["depth_noise_mm"][2] is None                               # 없으면 미평가
    # 데이터 유효성: 판 맞춤에 필요한 판 길이, 센서 창 (도착 허용 안쪽, 바닥은 깊이만큼 더 멀다)
    assert fidelity_gate(dict(GOOD, fov_margin_mm=4.0), {}, spec, "fidelity", **kw).status == "DATA_FAIL"
    assert fidelity_gate(dict(GOOD, sensor_z_mm=533.0), {}, spec, "fidelity", **kw).status == "DATA_FAIL"   # 544 − 2.1 − 10 = 531.9
    assert fidelity_gate(dict(GOOD, sensor_z_mm=530.0), {}, spec, "fidelity", **kw).status == "PASS"
    assert fidelity_gate(dict(GOOD, fov_margin_mm=4.0), {}, spec, "fidelity", **dict(kw, validity=None)).status == "PASS"   # 미평가


def test_fidelity_gate_square_groove_and_no_backing():
    sq = GrooveSpec(kind="square", bevel_side="auto")
    fm = dict(GOOD, top_opening_mm=5.0, bevel_angle_deg=float("nan"))
    v = fidelity_gate(fm, {}, sq, "fidelity", extent=EXT0, coverage=_cov())
    assert v.status == "PASS" and v.checks["bevel_angle_deg"][2] is None
    nb = GrooveSpec(bevel_side="auto", backing=False)
    v = fidelity_gate(GOOD, {}, nb, "fidelity", extent=EXT0, coverage=dict(_cov(), floor=_cov(False)["floor"]))
    assert v.status == "PASS" and v.checks["coverage_floor"][2] is None          # 뒷댐재 없음: 바닥 충분성 미평가


def test_seam_scan_features_root_edges_fraction_and_robust_std():
    M = 10
    cr = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], bool)
    gap = np.full(M, 5.0)
    gap[4] = 9.0                                                               # 이상치 하나 (강건 표준편차는 무시)
    sf = SeamScanFeatures(n_profiles=M, indices=np.arange(M), ok=np.ones(M, bool), in_cruise=cr, root_gap_mm=gap,
                          depth_mm=np.linspace(9.9, 10.1, M), top_opening_mm=np.full(M, 10.8), bevel_angle_deg=np.full(M, 30.0),
                          hi_lo_mm=np.zeros(M), tilt_deg=np.zeros(M), wall_shear_k=np.full(M, np.nan), wall_points_n=np.zeros(M),
                          floor_visible=np.ones(M), root_edges_resolved=np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0], float))
    m = sf.medians()
    assert m["root_edges_fraction"] == pytest.approx(7 / 8)                    # ok ∧ cruise 8 개 중 7
    assert m["root_gap_robust_std_mm"] == pytest.approx(0.0, abs=1e-12)
    assert 0.05 < m["depth_robust_std_mm"] < 0.15


def _sf_with_trend(M=1200, L=0.38, sigma=0.10, seed=3):
    """갭이 S→E 로 5.0 → 5.4 mm 벌어지는 시편 (시공 오차) + 프로파일 간 측정 잡음 sigma."""
    rng = np.random.default_rng(seed)
    s_m = np.linspace(0.0, L, M)
    gap = 5.0 + 0.4 * s_m / L + rng.normal(0.0, sigma, M)
    z = np.zeros(M)
    return SeamScanFeatures(n_profiles=M, indices=np.arange(M), ok=np.ones(M, bool), in_cruise=np.ones(M, bool), root_gap_mm=gap,
                            depth_mm=10.0 + rng.normal(0.0, 0.03, M), top_opening_mm=10.8 + 0.4 * s_m / L + rng.normal(0.0, sigma, M),
                            bevel_angle_deg=np.full(M, 31.0), hi_lo_mm=z, tilt_deg=z, wall_shear_k=np.full(M, np.nan),
                            wall_points_n=z, floor_visible=np.ones(M), root_edges_resolved=np.ones(M), s_m=s_m), s_m


def test_measurement_noise_removes_the_real_along_seam_change():
    """2026-10-02 판정 분리: 측정 잡음 = 10 mm 이동 중앙값 (시편의 실제 변화) 을 뺀 프로파일 간 흔들림의 강건 σ. 갭이 0.4 mm 벌어지는
    시편에서도 잡음은 주입한 σ 0.10 근처이고, 전체 강건 표준편차 (보고용) 는 실제 변화를 포함해 더 크다."""
    sf, _ = _sf_with_trend()
    m = sf.medians()
    assert 0.085 < m["root_gap_noise_mm"] < 0.115, m["root_gap_noise_mm"]
    assert 0.085 < m["top_opening_noise_mm"] < 0.115
    assert 0.02 < m["depth_noise_mm"] < 0.04
    assert m["root_gap_robust_std_mm"] > m["root_gap_noise_mm"] + 0.03
    # 프로파일이 너무 적으면 미평가
    few = dataclasses.replace(sf, ok=np.r_[np.ones(8, bool), np.zeros(sf.n_profiles - 8, bool)])
    assert math.isnan(few.medians()["root_gap_noise_mm"])


def test_along_seam_bins_report_the_construction_deviation_per_position():
    sf, s_m = _sf_with_trend(sigma=0.05)
    bins = sf.along_seam_bins(0.020, origin=0.0)
    assert len(bins) == 19 and bins[0]["s_from_m"] == pytest.approx(0.0) and bins[-1]["s_to_m"] == pytest.approx(0.38)
    mids = np.array([(b["s_from_m"] + b["s_to_m"]) / 2 for b in bins])
    g = np.array([b["root_gap_mm"] for b in bins])
    assert np.max(np.abs(g - (5.0 + 0.4 * mids / 0.38))) < 0.03                   # 구간 중앙값 = 그 위치의 실제 갭
    assert all(b["n"] >= 5 and set(b) >= {"depth_mm", "top_opening_mm", "hi_lo_mm", "bevel_angle_deg"} for b in bins)
    # 다른 좌표 (정련 심 직선 t, 반대 방향) 와 경계 정렬
    t = 0.05 - s_m
    b2 = sf.along_seam_bins(0.020, s=t, origin=0.05 - 0.38)
    assert len(b2) == 19 and b2[0]["s_from_m"] == pytest.approx(0.05 - 0.38) and b2[0]["root_gap_mm"] > b2[-1]["root_gap_mm"]
    # 프로파일 < 5 인 구간은 뺀다
    sparse = dataclasses.replace(sf, ok=(np.arange(sf.n_profiles) % 200) == 0)
    assert sparse.along_seam_bins(0.020) == []


# --------------------------------------------------------------------------------------
# 용접 자세 입력
# --------------------------------------------------------------------------------------
def test_position_mapping_and_the_direction_consistency_message():
    assert orientation_from_position("1G") == "flat_1G" and orientation_from_position("3g") == "vertical_3G"
    assert position_from_orientation("vertical_3G") == "3G" and position_from_orientation("auto") is None
    with pytest.raises(JobError, match="give position: 1G\\|3G"):
        orientation_from_position("2G")
    vertical = ScanJob(start=[0.5, 0.7, 0.3], end=[0.5, 0.7, 0.68], surface_normal=[0, -1, 0], orientation="flat_1G",
                       specimen_top_z_m=0.68)
    assert vertical.position == "1G"
    with pytest.raises(JobError) as ei:
        vertical.validate()
    msg = str(ei.value)
    assert "given position 1G" in msg and "gravity_axis_base [0.0, 0.0, -1.0]" in msg and "1G requires horizontal path" in msg
    flat = ScanJob(start=[0.6, -0.19, 0.05], end=[0.6, 0.19, 0.05], surface_normal=[-1, 0, 0], orientation="vertical_3G",
                   specimen_top_z_m=0.05)
    with pytest.raises(JobError, match="given position 3G .*gravity_axis_base"):
        flat.validate()
    assert ScanJob(start=[0, 0, 0], end=[0.1, 0, 0]).position is None                                  # 기본 'auto' = 미지정


def test_fidelity_gate_distance_criteria_with_bevel_report_only():
    """사용자 2026-10-02: 거리 세 개 (갭·깊이·개구) 의 설계 허용치 (작업 파일, 예 ± 0.2 mm) 는 시편 보고의 기준, 베벨각은 보고만
    (qc_targets tol: null). 거리로 계산한 베벨각 (atan((개구 − 갭)/깊이)) 도 보고. 측정 판정은 '단면 중앙값이 허용치를 구분하나'."""
    spec = GrooveSpec(bevel_side="auto")
    q = {"root_gap_mm": {"tol": 0.2}, "groove_depth_mm": {"tol": 0.2}, "top_opening_mm": {"tol": 0.2}, "bevel_angle_deg": {"tol": None}}
    cov = _cov(n=30)
    v = fidelity_gate(dict(GOOD, bevel_angle_deg=33.0), {}, spec, "fidelity", extent=EXT0, qc_targets=q, coverage=cov)
    assert v.status == "PASS", v.summary()
    assert v.checks["bevel_angle_deg"] == (33.0, None, None)                     # 보고만
    b = v.checks["bevel_from_distances_deg"][0]
    assert abs(b - math.degrees(math.atan((GOOD["top_opening_mm"] - GOOD["root_gap_mm"]) / GOOD["depth_mm"]))) < 1e-9
    for k, val in (("root_gap_mm", 5.25), ("depth_mm", 10.25), ("top_opening_mm", 11.0), ("root_gap_robust_std_mm", 0.25)):
        v = fidelity_gate(dict(GOOD, **{k: val}), {}, spec, "fidelity", extent=EXT0, qc_targets=q, coverage=cov)
        assert v.status == "PASS" and not v.failed, (k, v.summary())
        if k in v.specimen:
            assert v.specimen[k]["within"] is False and v.specimen[k]["tol"] == 0.2
    # at07 실측 잡음 0.14 mm: 단면 30 프로파일 중앙값의 k·불확도 = 3·1.25·0.14/√30 = 0.096 ≤ 0.2 → PASS; 단면 2 프로파일이면 못 구분
    assert fidelity_gate(dict(GOOD, root_gap_noise_mm=0.14), {}, spec, "fidelity", extent=EXT0, qc_targets=q, coverage=cov).status == "PASS"
    v = fidelity_gate(dict(GOOD, root_gap_noise_mm=0.14), {}, spec, "fidelity", extent=EXT0, qc_targets=q, coverage=_cov(n=2))
    assert v.failed == ["root_gap_noise_mm"] and any("cannot resolve the tolerance" in n for n in v.notes)


def test_fidelity_gate_reports_the_along_seam_range_of_a_construction_deviation():
    """at07 (2026-10-02) 실측: 갭 중앙값 5.01 (설계 5 ± 0.2 안) 이지만 E 끝 쪽 구간 중앙값이 5.42 까지 벌어짐 (시공 오차). 구간 범위가
    설계 ± tol 밖이면 along_within False + 노트, 판정은 PASS."""
    spec = GrooveSpec(bevel_side="auto")
    q = {"root_gap_mm": {"tol": 0.2}, "groove_depth_mm": {"tol": 0.2}, "top_opening_mm": {"tol": 0.2}, "bevel_angle_deg": {"tol": None}}
    fm = dict(GOOD, root_gap_mm=5.01, root_gap_along_min_mm=4.90, root_gap_along_max_mm=5.42, depth_along_min_mm=9.95,
              depth_along_max_mm=10.10, root_gap_noise_mm=0.14, depth_noise_mm=0.04, top_opening_noise_mm=0.13)
    v = fidelity_gate(fm, {}, spec, "fidelity", extent=EXT0, qc_targets=q, coverage=_cov(n=30))
    assert v.status == "PASS", v.summary()
    g = v.specimen["root_gap_mm"]
    assert g["within"] is True and g["along_within"] is False and (g["along_min"], g["along_max"]) == (4.90, 5.42)
    assert v.specimen["depth_mm"]["along_within"] is True
    assert any("root_gap_mm median 5.010 along 4.900..5.420 (design 5.00 +- 0.20)" in n for n in v.notes), v.notes
    assert "depth_mm median" not in " ".join(v.notes)
    d = v.to_dict()
    assert d["specimen"]["root_gap_mm"]["along_max"] == 5.42 and d["checks"]["root_gap_noise_mm"]["ok"] is True


def test_root_position_precision_is_judged_against_the_job_tolerance():
    """루트 위치 (용접 경로 겨냥점) 의 프로파일 간 잡음도 치수와 같은 규칙 (사용자 2026-10-07: 작업 qc_targets root_position_mm 1 mm):
    k × 단면 중앙값 불확도 √(π/2)·σ/√n ≤ 허용치. 단면당 30 프로파일이면 허용 σ = 1 / (3 · 1.2533 / √30) ≈ 1.46 mm."""
    spec = GrooveSpec(bevel_side="auto")
    qc = dict(QC, root_position_mm={"tol": 1.0})
    kw = dict(extent=EXT, qc_targets=qc, coverage=_cov(n=30), validity=VALID)
    lim = 1.0 / (3.0 * math.sqrt(math.pi / 2.0) / math.sqrt(30))
    v = fidelity_gate(dict(GOOD, root_lateral_noise_mm=0.1, root_normal_noise_mm=0.05), {}, spec, "fidelity", **kw)
    assert v.status == "PASS" and v.checks["root_lateral_noise_mm"][1] == pytest.approx(lim), v.summary()
    v = fidelity_gate(dict(GOOD, root_lateral_noise_mm=1.7, root_normal_noise_mm=0.05), {}, spec, "fidelity", **kw)
    assert v.status == "SPEC_FAIL" and "root_lateral_noise_mm" in v.failed and any("aim point" in n for n in v.notes), v.summary()
    v = fidelity_gate(dict(GOOD, root_lateral_noise_mm=1.7), {}, spec, "fidelity", extent=EXT, qc_targets=QC, coverage=_cov(n=30),
                      validity=VALID)
    assert v.checks["root_lateral_noise_mm"][2] is None                       # 허용치가 없으면 보고만
