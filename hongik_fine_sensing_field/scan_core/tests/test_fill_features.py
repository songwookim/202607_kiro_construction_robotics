"""fill_features: 부분 충전 그루브 특징 (2026-10-03, 다층 멀티패스 2단계) — bead_fill 의 같은 정의로 만든 합성 데이터의 진실값과 비교,
실기 빈 그루브 (at07 패스 2) 에서 기존 추출기와 비교."""
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from scan_core.bead_fill import FilledGrooveScene, FillSegment, print_layout
from scan_core.fill_features import extract_fill_features, extract_fill_features_along_scan, ransac_quadratic
from scan_core.frames import make_transform, rot_x
from scan_core.path_gen import plate_frame
from scan_core.reconstruct import reconstruct_scan
from scan_core.specimen import GrooveSpec
from scan_core.synth import SynthConfig, simulate_profile, simulate_scan
from scan_core.types import Calib, ScanJob

SPEC = GrooveSpec(bevel_side="left")
T_PLATE = plate_frame([0, 0, 0], [1, 0, 0], [0, 0, 1])
R0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)       # X_s = 판 y (가로), Y_s = 판 x (진행), Z_s = −판 z
REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_groove_at07_pass2_empty_2026-10-02.npz")


def _layout_scene():
    lay = print_layout(SPEC)
    lay.segments.insert(0, FillSegment(-40, 0, None, "0_empty"))
    sc = FilledGrooveScene.from_layout(lay)
    sc.x_range = (sc.x_range[0] - 0.04, sc.x_range[1])
    sc.s_origin_m = sc.x_range[0] + 0.04
    return lay, sc


def _profile(sc, s_mm, *, y_off=0.0, roll_deg=0.0, seed=0, **cfg):
    T_bo = make_transform(rot_x(math.radians(roll_deg)) @ R0, (sc.s_origin_m + s_mm * 1e-3, y_off, 0.34))
    return simulate_profile(T_bo, sc, T_PLATE, SynthConfig(n_points=1408, seed=seed, sigma_z_mm=0.03, **cfg))


def _crown_truth(lay, s_mm, toes):
    (xl, zl), (xr, zr) = toes
    x = np.linspace(xl, xr, 2001)[1:-1]                                         # 토우 끝점은 벽 (판 높이) 쪽 반개구간
    dev = lay.height_mm(x, np.full_like(x, s_mm)) - (zl + (x - xl) * (zr - zl) / (xr - xl))
    return float(dev[np.argmax(np.abs(dev))])


def test_ransac_quadratic_ignores_a_reflection_band():
    rng = np.random.default_rng(1)
    x = np.linspace(-4, 4, 200)
    h = 0.1 * x * x - 0.3 * x + 2.0 + rng.normal(0, 0.02, x.size)
    h[60:90] -= 3.0                                                            # 반사 띠
    c, inl = ransac_quadratic(x, h, 0.2, 200, rng)
    assert c == pytest.approx([2.0, -0.3, 0.1], abs=0.02) and not inl[60:90].any() and inl.sum() > 160


@pytest.mark.parametrize("y_off, roll_deg, tol_area", [(0.0, 0.0, 0.6), (0.03, 0.0, 1.0), (-0.03, 6.0, 1.2)])
def test_every_fill_state_against_the_truth(y_off, roll_deg, tol_area):
    """빈 그루브 → 패스 1·2·3 (볼록·오목·경사·비대칭·기운 비드) → 캡 2 종: 상태, 남은 단면적, 토우 깊이·폭, 크라운, 캡 보강.
    센서를 그루브 위 / 옆 30 mm / 옆 30 mm + 롤 6° 에 두고 노이즈 0.03 mm. 옆에서 보면 수직벽 옆 바닥이 판 모서리에 가려 (그늘 ≈ 깊이 ×
    30/400 mm) 벽 위치를 광선 모델로 추정하므로 허용치가 조금 넓다."""
    lay, sc = _layout_scene()
    for seg in lay.segments:
        for k in range(3):
            s = seg.s0_mm + 6.0 + 15.0 * k
            x, z = _profile(sc, s, y_off=y_off, roll_deg=roll_deg, seed=k)
            f = extract_fill_features(x, z, bevel_side_hint="left", bevel_angle_hint_deg=30.0)
            tr = lay.truth_at(s)
            want = "cap" if tr["kind"] == "cap" else "groove"
            assert f.ok and f.state == want, (seg.label, s, f.report())
            assert abs(f.remaining_area_mm2 - tr["remaining_area_mm2"]) < tol_area, (seg.label, s, f.report(), tr)
            assert abs(f.reinforcement_area_mm2 - tr["reinforcement_area_mm2"]) < 0.6, (seg.label, s, f.report(), tr)
            if want == "groove" and tr["toes_mm"] is not None:
                (xl, zl), (xr, zr) = tr["toes_mm"]
                t = SPEC.thickness_mm
                assert abs(f.toe_depth_left_mm - (t - zl)) < 0.12 and abs(f.toe_depth_right_mm - (t - zr)) < 0.12, (seg.label, f.report())
                assert abs(f.toe_width_mm - (xr - xl)) < 0.3, (seg.label, f.report())
                assert abs(f.crown_mm - _crown_truth(lay, s, tr["toes_mm"])) < 0.2, (seg.label, f.report())
            if tr["kind"] == "empty":
                assert abs(f.toe_width_mm - SPEC.root_gap_mm) < 0.15 and abs(f.depth_center_mm - 10.0) < 0.1
            if want == "cap":
                srf = seg.surface
                assert abs(f.cap_height_mm - srf.reinforcement_mm) < 0.25 and f.remaining_area_mm2 < 0.2, f.report()


def test_cap_and_groove_survive_reflection_bands_in_most_profiles():
    """실기 같은 그루브 안 반사 띠 (프로파일의 50 %, 1–4 mm 아래): 판 끝을 덮은 띠는 프로파일 하나로 구별할 수 없으므로 여기서는 대부분이
    맞는지만 본다 — 걸러 내기는 심 방향 일관성 (아래 전체 스캔 시험)."""
    lay, sc = _layout_scene()
    good = total = 0
    for seg in lay.segments:
        for k in range(4):
            s = seg.s0_mm + 5.0 + 11.0 * k
            x, z = _profile(sc, s, seed=10 + k, reflection_prob=0.5)
            f = extract_fill_features(x, z, bevel_side_hint="left", bevel_angle_hint_deg=30.0)
            total += 1
            good += bool(f.ok and abs(f.remaining_area_mm2 - lay.truth_at(s)["remaining_area_mm2"]) < 1.0)
    assert good >= 0.8 * total, (good, total)


def test_full_synthetic_scan_bins_match_the_truth_with_reflections_and_pitch():
    """로봇 이동 + 합성 센서 (반사 30 %, 피치 −5°) + 재구성 → 프로파일별 충전 특징 → 비스듬한 단면 보정·심 방향 일관성 → 상태별
    남은 단면적·캡 보강 (진실값 = 레이아웃, 위치 = 표면 가운데 점의 심 방향 투영)."""
    calib = Calib.lab_initial(eoat_yaw_rad=-np.pi / 2, opt_yaw_branch=-1)
    spec = GrooveSpec(bevel_side="left", weld_length_mm=170.0)
    lay = print_layout(spec, segment_mm=20.0)
    scene = FilledGrooveScene.from_layout(lay, centered=False)
    job = ScanJob(start=[0.6, -0.085, 0.05], end=[0.6, 0.085, 0.05], surface_normal=[0, 0, 1], table_z_m=-0.05,
                  specimen_top_z_m=0.05, lead_in_m=0.03, lead_out_m=0.03, speed_mps=0.02, accel_mps2=0.5, profile_fps=20.0,
                  pitch_deg=-5.0, standoff_m=0.34, orientation="flat_1G", job_id="fill_syn")
    syn = simulate_scan(job, calib, scene, SynthConfig(n_points=704, seed=3, sigma_z_mm=0.03, dropout_prob=0.01, robot_poll_rate_hz=0.0,
                                                       reflection_prob=0.3))
    res = reconstruct_scan(syn.frames, syn.robot, calib, job, fps_nominal=20.0)
    u = job.unit_dir
    ff0 = extract_fill_features_along_scan(res)
    s_proj = (ff0.centre_base_m - job.start) @ u
    ff = extract_fill_features_along_scan(res, s_along=s_proj, seam_u=u, plate_n=[0, 0, 1])
    assert np.median(ff.obliquity_area[np.isfinite(s_proj)]) == pytest.approx(math.cos(math.radians(5.0)), abs=2e-3)
    err = {}
    for j in np.flatnonzero(ff.ok):
        s = s_proj[j] * 1e3
        seg = lay.segment_at(s)
        if seg is None or not (seg.s0_mm + 1.5 < s < seg.s1_mm - 1.5):
            continue
        tr = lay.truth_at(s)
        assert ff.state[j] == ("cap" if tr["kind"] == "cap" else "groove")
        err.setdefault(seg.label, []).append((ff.values["remaining_area_mm2"][j] - tr["remaining_area_mm2"],
                                              ff.values["reinforcement_area_mm2"][j] - tr["reinforcement_area_mm2"]))
    assert len(err) == 8 and all(len(v) >= 8 for v in err.values()), {k: len(v) for k, v in err.items()}
    for label, e in err.items():
        E = np.asarray(e)
        assert abs(float(np.median(E[:, 0]))) < 0.4 and abs(float(np.median(E[:, 1]))) < 0.4, (label, np.median(E, axis=0))
        assert float(np.max(np.abs(E))) < 2.2, (label, float(np.max(np.abs(E))))   # 한 프로파일 상한 ≈ 일관성 허용치 (2 mm²)
    rej = [r for r in ff.rejected if r]
    assert rej and any(r.startswith("opening_mm") for r in rej)                  # 판 끝을 덮은 반사 띠 → 개구가 튀어 걸러진다
    # 구간 보고 (20 mm, 판 끝 기준)
    bins = ff.along_seam_bins(0.020, s=s_proj, origin=0.0)
    assert [b["state"] for b in bins][:8] == ["groove"] * 6 + ["cap"] * 2


@pytest.mark.skipif(not os.path.exists(REAL), reason="real data excerpt missing")
def test_real_empty_groove_at07_matches_the_existing_extractor_and_is_quiet():
    """실기 at07 패스 2 (빈 그루브, 2026-10-02) 발췌: 기존 추출기 기준 (개구 11.31, 갭 5.01, 깊이 10.02 — 그 패스 전체 중앙값) 과 맞고,
    수직벽 평활화 뒤 프로파일 간 잡음이 작다. 수직벽 쪽 판 끝 처짐·벽 중간의 유령점 무리 (상호 반사) 가 있는 데이터."""
    d = np.load(REAL)
    res = SimpleNamespace(x_mm=d["x_mm"], z_mm=d["z_mm"], valid=None, T_base_opt=d["T_base_opt"], s_m=d["s_m"])
    ff = extract_fill_features_along_scan(res, seam_u=d["seam_u"], plate_n=d["plate_n"], bevel_side_hint="right",
                                          bevel_angle_hint_deg=31.6)
    assert set(ff.state.tolist()) == {"groove"} and int(ff.ok.sum()) >= 0.85 * ff.n_profiles
    m = ff.medians()
    assert m["opening_mm"] == pytest.approx(11.31, abs=0.15) and m["toe_width_mm"] == pytest.approx(5.05, abs=0.15)
    assert m["depth_center_mm"] == pytest.approx(10.0, abs=0.1)
    assert 79.0 < m["remaining_area_mm2"] < 85.0                               # 깊이 × (갭 + 개구)/2 ≈ 81.5 (± 형상)
    s_line = (ff.toe_left_base_m - d["seam_c"]) @ d["seam_u"]
    for key, lim in (("opening_mm", 0.1), ("toe_width_mm", 0.15), ("remaining_area_mm2", 0.8)):
        s_, v = ff.series(key, s=s_line)
        r = v - np.array([np.median(v[a:b]) for a, b in zip(np.searchsorted(s_, s_ - 0.005), np.searchsorted(s_, s_ + 0.005, "right"))])
        assert 1.4826 * np.median(np.abs(r - np.median(r))) < lim, key
    assert np.nanmax(np.abs(ff.wall_shift_mm["left"])) > 0.05                  # 수직벽 (센서 왼쪽) 을 실제로 평활화했다
