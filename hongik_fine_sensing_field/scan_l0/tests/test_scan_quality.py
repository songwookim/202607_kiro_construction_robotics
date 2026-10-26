"""scan_quality: 프로파일 판독 품질, 구간, 패스 병합, 입사 가중, 데이터셋 (적응형 스캔, 2026-10-06)."""
import json

import numpy as np
import pytest

from scan_core.specimen import GrooveSpec
from scan_l0 import scan_quality as SQ


def _feats(n=200, s0=0.0, s1=0.38, bad=None, gap=5.0):
    s = np.linspace(s0, s1, n)
    P = np.column_stack([np.zeros(n), np.zeros(n), s])                     # 루트 직선 = z 축
    f = {"ok": np.ones(n, dtype=bool), "top_opening_mm": np.full(n, 11.0), "depth_mm": np.full(n, 10.0),
         "root_gap_mm": np.full(n, gap), "bevel_angle_deg": np.full(n, 30.0), "hi_lo_mm": np.zeros(n),
         "root_polyline_base_m": P, "s_line_m": s.copy()}
    if bad is not None:
        m = (s >= bad[0]) & (s < bad[1])
        f["bevel_angle_deg"][m] = np.nan
        f["root_gap_mm"][m] = np.nan
    return f


SPEC = GrooveSpec(kind="single_bevel", thickness_mm=10.0, bevel_angle_deg=30.0, bevel_side="left", root_gap_mm=5.0)


def test_profile_quality_and_bins():
    f = _feats(bad=(0.10, 0.20))
    q = SQ.profile_quality(f, SPEC)
    assert q["good"].sum() < q["groove"].sum() and not q["gap"][np.argmin(np.abs(f["s_line_m"] - 0.15))]
    b = SQ.quality_bins(SQ.line_coordinate(f, [0, 0, 0], [0, 0, 1]), q, 0.0, 0.38)
    assert len(b) == 19
    assert b[5]["frac_good"] == pytest.approx(0.0) and b[1]["frac_good"] == pytest.approx(1.0)
    q_sq = SQ.profile_quality(f, GrooveSpec(kind="square", bevel_angle_deg=0.0))          # 베벨 없는 설계: 베벨 항목 무관
    assert q_sq["bevel"].all()


def test_posture_weight_from_the_measured_incidence_bias():
    assert SQ.posture_weight(0.0) == 1.0 and SQ.posture_weight(20.0) == 1.0
    assert SQ.posture_weight(28.0) == 0.8 and SQ.posture_weight(40.0) == 0.5 and SQ.posture_weight(float("nan")) == 0.5


def test_merge_takes_each_section_from_the_best_pass():
    """패스 1 (40°, 가운데 나쁨, 갭 6.3 편향) + 패스 2 (20°, 아래 100 mm 못 감, 갭 5.0) → 구간별: 위·가운데 = 패스 2, 아래 = 패스 1."""
    f1 = _feats(bad=(0.15, 0.25), gap=6.3)
    f2 = _feats(s0=0.10, s1=0.38, gap=5.0)
    e = [{"k": 1, "feats": f1, "q": SQ.profile_quality(f1, SPEC), "posture": {"incidence_deg": 40.0}},
         {"k": 2, "feats": f2, "q": SQ.profile_quality(f2, SPEC), "posture": {"incidence_deg": 20.0}}]
    merged, choice = SQ.merge_feature_arrays(e, [0, 0, 0], [0, 0, 0.38])                # 관측 끝 S, E (base 점)
    by = {round(c["s_from_m"], 2): c["pass"] for c in choice}
    assert by[0.0] == 1 and by[0.06] == 1 and by[0.2] == 2 and by[0.36] == 2
    assert set(np.unique(merged["source_pass"])) == {1, 2}
    assert np.all(np.diff(merged["s_line_m"]) >= 0)
    mm = SQ.merged_medians(merged)
    assert mm["root_gap_mm"] == pytest.approx(5.0)                          # 편향된 40° 갭은 아래 구간에만
    assert mm["depth_mm"] == pytest.approx(10.0)


def test_merge_coordinates_follow_the_S_to_E_contract():
    """병합 좌표 = 기준 패스의 관측 끝 S, E (S→E 계약): s = S 에서 unit(E − S) 로 잰 거리, 구간은 S 에서. 루트가 +z 로 놓여도 S 가 위 끝
    (z 0.38) 이면 축은 −z, 첫 구간은 S 쪽 — 패스 정련 직선의 방향 (진행 방향) 은 들어오지 않는다."""
    f1 = _feats(bad=(0.30, 0.39))                                          # z 0.30–0.38 (S 쪽 80 mm) 이 나쁨
    f2 = _feats(gap=5.2)
    e = [{"k": 1, "feats": f1, "q": SQ.profile_quality(f1, SPEC), "posture": {"incidence_deg": 10.0}},
         {"k": 2, "feats": f2, "q": SQ.profile_quality(f2, SPEC), "posture": {"incidence_deg": 10.0}}]
    merged, choice = SQ.merge_feature_arrays(e, [0, 0, 0.38], [0, 0, 0])
    assert merged["u_base"] == pytest.approx([0.0, 0.0, -1.0]) and merged["line_c_base_m"] == pytest.approx([0.0, 0.0, 0.38])
    assert float(merged["s_a_m"]) == 0.0 and float(merged["s_b_m"]) == pytest.approx(0.38)
    assert choice[0]["s_from_m"] == pytest.approx(0.0) and choice[-1]["s_to_m"] == pytest.approx(0.38)
    s = merged["s_line_m"]
    assert np.all(np.diff(s) >= 0) and s[0] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(s, 0.38 - merged["root_polyline_base_m"][:, 2], atol=1e-12)
    assert choice[0]["scores"][1] == 0.0 and choice[0]["pass"] == 2       # S 쪽 첫 구간 (z 0.38 → 0.36) 은 패스 1 이 나쁘다
    assert choice[-1]["scores"][1] == pytest.approx(1.0)                   # E 쪽 (z 0) 은 둘 다 좋다 (같으면 나중 패스)


def test_plausibility_rejects_a_readable_but_wrong_groove():
    """g3_12 패스 2 (입사 20°, 토치 위): 반사로 단면이 무너져도 값은 다 읽혔다 (개구 20, 갭 15, 깊이 6.7 mm) — 판독만 세면 '좋음'.
    타당성 (설계 ± 25 %, 갭 ± 2.5 mm) 으로 나쁨. 이웃과 어긋난 한 프로파일은 일관성으로 나쁨."""
    f = _feats(n=100)
    f["top_opening_mm"][:] = 20.0
    f["root_gap_mm"][:] = 15.0
    f["depth_mm"][:] = 6.7
    q = SQ.profile_quality(f, SPEC)
    assert q["gap"].all() and q["bevel"].all() and not q["good"].any()
    g = _feats(n=100)
    g["top_opening_mm"][50] = 13.2                                          # 설계 (10.77) ± 25 % 안이지만 이웃 (11.0) 과 2.2 mm 다름
    q2 = SQ.profile_quality(g, SPEC)
    assert q2["plausible"][50] and not q2["consistent"][50] and q2["good"].sum() == 99


def test_phi_prior_model(tmp_path, monkeypatch):
    """φ 예상 (4d P6): 씨앗 점 (실기 재생 v3: +20° 0.00 g3_12 붕괴, +40° 0.82, −5° 0.95, −15° 0.98, −35° 0.77) 은 같은 칸이면 그 값,
    격자 한 칸 (5°) 안으로 이웃한 두 점 사이만 보간, 그 밖은 모른다 (중간점 0.5 — 측정한 적 없는 자세로 옮겨 가지 않게). 정반사 대역은
    자료와 무관하게 0. 데이터셋 (낮은 버전은 무시) 과 이번 실행 측정이 같은 칸의 점을 대신한다."""
    p = tmp_path / "ds.jsonl"
    g, why = SQ.phi_prior("3G", 20.0, p)
    assert g == pytest.approx(0.0) and "static" in why
    g, why = SQ.phi_prior("3G", -7.5, p, measured=[(-10.0, 0.90)])          # −10 (이번 실행) ↔ −5 (5° = 격자 한 칸): 보간
    assert 0.90 < g < 0.95 and "interpolated" in why
    g, why = SQ.phi_prior("3G", 30.0, p)                                    # +20 ↔ +35 (15° 틈): 모름
    assert g == pytest.approx(SQ.PHI_UNKNOWN) == pytest.approx(0.5) and "no measurement" in why
    assert SQ.phi_prior("3G", -25.0, p)[0] == pytest.approx(0.5)            # 자료 사이 큰 틈
    assert SQ.phi_prior("3G", 12.0, p)[0] == 0.0 and "specular" in SQ.phi_prior("3G", 12.0, p)[1]
    # 데이터셋: 낮은 버전은 무시, 현재 버전은 그 칸을 대신
    p.write_text(json.dumps({"position": "3G", "posture": {"phi_deg": -20.0}, "quality": {"frac_good": 0.9}}) + "\n")
    assert SQ.phi_prior("3G", -20.0, p)[1].startswith("no measurement")
    for ph, gg in ((-20.0, 0.30), (-19.0, 0.40)):
        assert SQ.append_dataset({"position": "3G", "posture": {"phi_deg": ph}, "quality": {"frac_good": gg}}, p)
    SQ.append_dataset({"position": "1G", "posture": {"phi_deg": -20.0}, "quality": {"frac_good": 0.99}}, p)
    g, why = SQ.phi_prior("3G", -20.0, p)
    assert g == pytest.approx(0.35) and why.startswith("dataset n=2")
    assert json.loads(p.read_text().splitlines()[1])["quality_version"] == SQ.QUALITY_VERSION
    # 이번 실행 측정이 데이터셋보다 우선
    g, why = SQ.phi_prior("3G", -20.0, p, measured=[(-20.4, 0.05)])
    assert g == pytest.approx(0.05) and why.startswith("this run")
    pred, g, _ = SQ.predicted_score({"phi_deg": -20.0, "incidence_deg": 40.0}, "3G", p, measured=[(-20.0, 0.8)])
    assert pred == pytest.approx(0.4)                                        # 입사 가중 0.5 × 0.8
    # 환경 변수 'off' (시험 conftest): 경로 없이 부르면 읽지도 쓰지도 않는다 (씨앗은 읽는다)
    monkeypatch.setenv("SCAN_QUALITY_DATASET", "off")
    assert SQ.dataset_path() is None and SQ.append_dataset({"position": "3G"}) is None
    assert SQ.phi_prior("3G", 20.0)[0] == pytest.approx(0.0)


def test_phi_sign_flips_with_the_torch_side():
    """3G: 토치 위 φ = +입사, 진행 부호를 바꿔 토치를 광축 둘레로 뒤집으면 φ = −입사 (같은 입사 각도); 1G 피치 −5° 는 φ −5°."""
    from scan_core.types import ScanJob
    from scan_l0 import config as C
    from conftest import CALIB
    calib = C.load_calib(CALIB)
    job, _doc = C.load_job(C.JOBS_DIR / "groove_10T_3G.yaml")
    d = C.job_to_dict(job)
    d.update(start=[1.25, 0.088, 0.167], end=[1.254, 0.063, -0.155], surface_normal=[-0.9226, -0.3854, 0.0179], incidence_tilt_deg=20.0,
             pitch_deg=0.0, torch_lateral_m=tuple(d["torch_lateral_m"]), vertical_direction="down")
    out = {}
    for sign in (1, -1):
        j = ScanJob.from_dict({**d, "progression_sign": sign})
        out[sign] = SQ.pass_posture(j, calib)
    above = [v for v in out.values() if v["torch_above"]]
    below = [v for v in out.values() if not v["torch_above"]]
    assert len(above) == 1 and len(below) == 1
    # the calibration's own rotation of the laser plane about the laser line, relative to the tool (eoat_lab_01 2026-10-12: w_x
    # -1.43 deg) adds to phi on both torch sides and to the 1G pitch: the torch-side flip mirrors the commanded incidence, not phi
    import math
    from scan_core.types import Calib
    R = Calib.lab_initial(eoat_yaw_rad=calib.eoat_yaw_rad, opt_yaw_branch=+1).T_flange_opt[:3, :3].T @ calib.T_flange_opt[:3, :3]
    w = math.degrees(math.atan2(R[2, 1], R[1, 1]))
    assert abs(w) < 3.0
    assert above[0]["phi_deg"] == pytest.approx(20.0 + w, abs=0.2) and below[0]["phi_deg"] == pytest.approx(-20.0 + w, abs=0.2)
    assert 0.5 * (above[0]["incidence_deg"] + below[0]["incidence_deg"]) == pytest.approx(20.0, abs=0.2)
    job1, _ = C.load_job(C.JOBS_DIR / "groove_10T_1G.yaml") if (C.JOBS_DIR / "groove_10T_1G.yaml").exists() else (None, None)
    if job1 is not None and job1.resolved_orientation() == "flat_1G":
        d1 = {**C.job_to_dict(job1), "pitch_deg": -5.0, "incidence_tilt_deg": 0.0, "torch_lateral_m": tuple(job1.torch_lateral_m)}
        for sign in (1, -1):
            assert SQ.pass_posture(ScanJob.from_dict({**d1, "progression_sign": sign}), calib)["phi_deg"] == pytest.approx(-5.0 + w, abs=0.2)


def test_seed_file_is_the_v3_replay_and_matches_the_quality_version():
    """4d P6: 품질 사전값은 패키지 데이터 파일 (scan_l0/data/scan_quality_seed.jsonl) — 실기 패스를 v3 규칙으로 재생해 만든 것
    (scripts/reseed_quality_*.py; 손으로 고치지 않는다), 씨앗 버전 = QUALITY_VERSION. SCAN_QUALITY_DATASET=off 여도 읽힌다."""
    from scan_l0 import scan_quality as Q
    seed = Q._load_seed()
    assert int(seed["meta"]["quality_version"]) == Q.QUALITY_VERSION == 3 and seed["meta"]["n_passes"] >= 50
    pts = dict(Q.STATIC_PHI_PRIOR)
    assert pts[-5.0] > 0.9 and pts[-15.0] > 0.9 and pts[20.0] < 0.05 and 0.75 < pts[40.0] < 0.9    # 1G·토치 아래 좋음, +20° 붕괴
    assert 12.0 not in pts                                                     # 손으로 넣던 정반사 점은 자료가 아니다 (센서 모형 규칙)
    assert Q._WEIGHT_TABLE == ((25.0, 1.0), (30.0, 0.8), (90.0, 0.5))
    assert Q.PHI_UNKNOWN == 0.5 and Q.GOOD_FRAC_MIN == 0.5
