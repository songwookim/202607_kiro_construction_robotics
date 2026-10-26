"""모델 기반 판 끝 식별 (scan_core.localize.identify_seam_ends, 사용자 2026-10-01) — GrooveWorld 에 판 끝 너머 물체를 놓고 패스 하나를
직접 렌더링 (synthetic_pass, 가짜 CB 없음) → fit_pass (auto-teach 와 같은 함수) → 끝·플래그·충실도 게이트.

시편: 그루브 10T 일면개선 (lab_spec) 을 짧게 (L = 120 mm, spec.weld_length_mm = 120) — 한 패스가 양끝을 모두 지나가도록. 레이저 착점
간격: 시편 가운데 2 mm, 각 끝 ±20 mm 안 0.5 mm. 피치 −5° (1G) / 입사 +10°·피치 −5° (3G). 진짜 끝은 시편 끝면 (판 프레임 x = 0, L).

사례 (각각 측정 끝이 진짜 끝의 0.5 mm 안, 플래그를 기대하는 경우 제외):
  (i)   판 상면 20 mm 아래 평평한 테이블
  (ii)  테이블 위 30 mm 높이 클램프 블록, 끝에서 10 mm 너머, 심과 나란히
  (iii) 끝 너머 수직 펜스 판
  (iv)  끝 너머 15° 기운 평면
  (v)   그루브 없는 동일 평면 판이 끝에 붙어 있음 → 형상 연속성 (on-specimen 판정) 만으로 그루브가 끝나는 곳이 끝 (엔드 탭·뒷댐재
        연장 자체의 처리는 범위 밖 — 사용자 2026-10-01; 그루브 같은 틈이 이어지는 지그 사례 (vi) 는 같은 이유로 뺐다. 설계 길이 교차
        검사는 일반 일관성 플래그 'end_ambiguous' 로 남는다: scan_core 게이트 단위 시험)
  (vii) 판 위 클램프가 마지막 15 mm 를 덮음 → 'end_occluded', 기본은 게이트 PASS 아님 (accept_inferred_end 면 추정 끝으로 PASS)
  (viii) 3G: 아래 끝 밑의 지그
  (ix)  끝 너머 빈 공간 (센서 범위 안에 아무것도 없음: 전부 NaN 프로파일) — 양끝 0.5 mm 안·관측, 그리고 그 위를 지나는 면내 패스
        구간에서 가드 규칙이 하나도 걸리지 않는다 (라운드 4 F8)
라운드 4 F2: 실제 착점 간격 0.3 mm (0.03 m/s @ 100 fps), 레이저 기울기 5° / 15°, 양 진행 방향 (툴 자세 유지와 툴 반전) — 양끝 0.25 mm,
길이 0.3 mm 안 (판 모서리 괄호의 바닥 증거는 바닥 점을 직접 본다: 상면이 먼저 사라지는 끝에서 '그루브가 사라진 첫 프로파일' 은 바닥이
아직 있어 상한이 아니었다).
"""
import dataclasses
import math

import numpy as np
import pytest
from conftest import CALIB, CONFIG
from groove_world import BoxPrim, PlanePrim, lab_spec, make_world, make_world_3g, synthetic_pass

from scan_core.localize import fidelity_gate
from scan_core.path_gen import pass_overscan, sensor_frames
from scan_core.types import ScanJob
from scan_l0 import config as C
from scan_l0.autoteach import fit_pass, pass_extent_summary

L_SPEC = 0.120
DEPTH = 0.010
TABLE = 0.020


@pytest.fixture(scope="module")
def calib():
    return C.load_calib(CALIB)


@pytest.fixture(scope="module")
def spec():
    return dataclasses.replace(lab_spec("auto"), weld_length_mm=L_SPEC * 1e3)


def _world(calib, spec, *, vertical=False, **kw):
    if vertical:
        S = np.array([0.50, 0.70, 0.40])
        return make_world_3g(calib, spec=spec, S_true=S, E_true=S + np.array([0, 0, L_SPEC]), n_points=704, **kw)
    S = np.array([0.60, -0.06, 0.05])
    return make_world(calib, spec=spec, S_true=S, E_true=S + np.array([0, L_SPEC, 0]), n_points=704, **kw)


def _scan(world, calib, spec, *, vertical=False, beyond=0.025):
    """S→E (1G) / 위→아래 E→S (3G) 한 패스: 착점이 양끝 너머 beyond 까지. 반환 (fit, extent, verdict 함수, job, direction)."""
    root = -world.n * DEPTH
    if vertical:
        a, b = world.E + root, world.S + root                              # 위 → 아래 (3G 'down')
        kw = dict(orientation="vertical_3G", incidence_tilt_deg=10.0, vertical_direction="down", specimen_top_z_m=float(world.E[2]))
        direction, u_SE = "E->S", world.u
    else:
        a, b = world.S + root, world.E + root
        kw = dict(orientation="flat_1G", incidence_tilt_deg=0.0)
        direction, u_SE = "S->E", world.u
    job = ScanJob(start=a, end=b, surface_normal=world.n, standoff_m=0.34 + DEPTH, pitch_deg=-5.0, progression_sign=1,
                  lead_in_m=0.001, lead_out_m=0.001, **kw)
    d = pass_overscan(job, calib, 0.0, top_offset_m=DEPTH)["hit_offset_m"]
    L = job.length_m
    hits = np.unique(np.round(np.concatenate([np.arange(-beyond, 0.020, 0.0005), np.arange(0.020, L - 0.020, 0.002),
                                              np.arange(L - 0.020, L + beyond, 0.0005)]), 6))
    res = synthetic_pass(world, job, calib, hits - d)
    fr = sensor_frames(job, calib)
    sf, fit, P_root, P_top, rp = fit_pass(res, spec, u_prior=job.unit_dir, n_prior=job.normal_unit, stage="fine", gravity_axis_base=job.gravity_axis_base,
                                          u_travel=fr.u_travel, target_standoff_m=0.34, end_params=None)
    ext = pass_extent_summary(fit, direction, u_SE)

    def gate(**kw_):
        return fidelity_gate(fit.feature_medians, rp, spec, "fidelity", extent=ext, **kw_)

    return fit, ext, gate


def _err_mm(world, ext, which: str) -> float:
    """측정 끝 (S/E) − 진짜 끝 (루트 직선 위) 의 심 방향 거리 [mm]."""
    true = (world.S if which == "S" else world.E) - world.n * DEPTH
    return float((np.asarray(ext[which]["point"]) - true) @ world.u) * 1e3


def _assert_clean(world, ext, gate, tag):
    for X in ("S", "E"):
        assert ext[X]["observed"] and not ext[X]["occluded"], (tag, X, ext[X])
        assert abs(_err_mm(world, ext, X)) < 0.5, (tag, X, _err_mm(world, ext, X))
    assert not ext["staggered"] and abs(ext["length_mm"] - L_SPEC * 1e3) < 1.0, (tag, ext["length_mm"])
    v = gate()
    assert v.status == "PASS", (tag, v.summary())


def test_i_flat_table_20mm_below(calib, spec):
    w = _world(calib, spec, table_drop_m=TABLE)
    fit, ext, gate = _scan(w, calib, spec)
    _assert_clean(w, ext, gate, "table")
    assert fit.diag["off_specimen_reasons"].get("plates_off_plane_or_missing", 0) > 10   # 테이블은 판 평면이 아니다


def test_ii_clamp_block_on_the_table_beyond_the_end(calib, spec):
    # 테이블 (판 상면 −20 mm) 위 30 mm 높이 = 판 상면 +10 mm, 끝 10 mm 너머부터 60 mm, 심과 나란히 (판 대역 위)
    clamp = BoxPrim(xc=L_SPEC + 0.010 + 0.030, yc=0.040, sx=0.060, sy=0.030, h_top=-TABLE + 0.030)
    w = _world(calib, spec, table_drop_m=TABLE, extras=[clamp])
    fit, ext, gate = _scan(w, calib, spec, beyond=0.045)
    _assert_clean(w, ext, gate, "clamp beyond")


def test_iii_perpendicular_fence_beyond_the_end(calib, spec):
    fence = BoxPrim(xc=L_SPEC + 0.010 + 0.003, yc=0.0, sx=0.006, sy=0.30, h_top=0.030)
    w = _world(calib, spec, table_drop_m=TABLE, extras=[fence])
    fit, ext, gate = _scan(w, calib, spec)
    _assert_clean(w, ext, gate, "fence")


def test_iv_plane_tilted_15deg_beyond_the_end(calib, spec):
    ramp = PlanePrim(x0=L_SPEC + 0.003, x1=L_SPEC + 0.08, y0=-0.15, y1=0.15, h0=-0.004, slope_x=math.tan(math.radians(15.0)))
    w = _world(calib, spec, extras=[ramp])
    fit, ext, gate = _scan(w, calib, spec)
    _assert_clean(w, ext, gate, "tilted plane")


def test_v_coplanar_plate_without_groove_beyond_the_end(calib, spec):
    """판 상면이 그대로 이어지지만 그루브가 없다 → 끝 = 그루브가 끝나는 곳 (형상 연속성; 판 평면 점만으로는 끝을 정하지 않는다)."""
    tab = PlanePrim(x0=L_SPEC, x1=L_SPEC + 0.05, y0=-0.15, y1=0.15, h0=0.0)
    w = _world(calib, spec, table_drop_m=TABLE, extras=[tab])
    fit, ext, gate = _scan(w, calib, spec)
    _assert_clean(w, ext, gate, "run-off tab")
    assert ext["E"]["observed"] and "no_groove" in str(fit.diag["ends"]["last"]["next_reason"])


def test_vii_clamp_on_top_covering_the_last_15mm_is_occluded(calib, spec):
    clamp = BoxPrim(xc=L_SPEC - 0.015 + 0.030, yc=0.0, sx=0.060, sy=0.10, h_top=0.025)
    w = _world(calib, spec, extras=[clamp])
    fit, ext, gate = _scan(w, calib, spec, beyond=0.040)
    assert ext["E"]["occluded"] and not ext["E"]["observed"] and "end_occluded_last" in fit.flags
    assert ext["S"]["observed"] and abs(_err_mm(w, ext, "S")) < 0.5
    v = gate()
    assert v.status != "PASS" and "end_occluded_last" in v.failed, v.summary()
    inferred = np.asarray(ext["last_inferred"])                                   # 추정 끝 = S + 설계 길이 = 진짜 끝
    assert abs(float((inferred - (w.E - w.n * DEPTH)) @ w.u)) * 1e3 < 0.5
    v2 = gate(accept_inferred_end=True)
    assert v2.status == "PASS" and any("inferred" in n for n in v2.notes), v2.summary()


def test_viii_3g_jig_below_the_lower_end(calib, spec):
    """3G (세운 판, 측정 수직 판 평면): 아래 끝 10 mm 밑에 판면보다 10 mm 튀어나온 지그. 위→아래 패스."""
    jig = BoxPrim(xc=-0.010 - 0.025, yc=0.0, sx=0.050, sy=0.30, h_top=0.010)
    w = _world(calib, spec, vertical=True, extras=[jig])
    fit, ext, gate = _scan(w, calib, spec, vertical=True)
    _assert_clean(w, ext, gate, "3G jig")


# --------------------------------------------------------------------------------------
# (ix) 라운드 4 F8: 끝 너머 빈 공간 (전부 NaN 프로파일)
# --------------------------------------------------------------------------------------
def test_ix_empty_space_beyond_the_ends_all_nan_profiles(calib, spec):
    """끝 너머에 센서 범위 안의 표면이 전혀 없다 (테이블 없음): 끝 너머 프로파일은 전부 NaN. 양끝이 0.5 mm 안에서 '관측' 된다."""
    w = _world(calib, spec)                                                     # table_drop_m=None: 판 밖은 무반환
    fit, ext, gate = _scan(w, calib, spec)
    _assert_clean(w, ext, gate, "empty space")
    for which in ("first", "last"):
        e = fit.diag["ends"][which]
        assert e["observed"] and not e["occluded"] and e["off_specimen_span_m"] >= 0.005, (which, e)
    # 끝 너머 프로파일이 정말 전부 NaN 인지 (시나리오 확인) — 같은 패스 기하로 직접 렌더
    root = -w.n * DEPTH
    job = ScanJob(start=w.S + root, end=w.E + root, surface_normal=w.n, standoff_m=0.34 + DEPTH, pitch_deg=-5.0,
                  progression_sign=1, lead_in_m=0.001, lead_out_m=0.001, orientation="flat_1G")
    d = pass_overscan(job, calib, 0.0, top_offset_m=DEPTH)["hit_offset_m"]
    hits = np.concatenate([np.arange(-0.025, -0.004, 0.002), np.arange(0.010, L_SPEC - 0.010, 0.010),
                           np.arange(L_SPEC + 0.004, L_SPEC + 0.025, 0.002)])
    res = synthetic_pass(w, job, calib, hits - d)
    beyond = (hits < 0.0) | (hits > L_SPEC)
    assert np.all(~np.isfinite(res.z_mm[beyond])), "profiles beyond the ends must be all-NaN (nothing in range)"
    assert np.all(np.isfinite(res.z_mm[~beyond]).sum(axis=1) > 300)


# --------------------------------------------------------------------------------------
# 라운드 4 F2: 실제 착점 간격 0.3 mm 에서 판 모서리 괄호 (레이저 기울기 5° / 15°, 양 끝, 양 진행 방향)
# --------------------------------------------------------------------------------------
SPACING_REAL = 0.0003      # 0.03 m/s @ 100 fps


@pytest.mark.parametrize("tilt_deg", [5.0, 15.0])
@pytest.mark.parametrize("direction,prog,phase", [(+1, +1, 0.0), (-1, -1, 0.00015), (-1, +1, 0.0)],
                         ids=["S_to_E", "E_to_S_tool_kept", "E_to_S_tool_turned"])
def test_f2_plate_end_bracket_at_the_real_spacing(calib, spec, tilt_deg, direction, prog, phase):
    """레이저 면이 진행 방향으로 tilt 만큼 기울면 (1G 피치) 상면 착점선과 바닥 착점선이 깊이·tan(tilt) (0.9 / 2.7 mm) 어긋난다 — 한 끝은
    상면이, 다른 끝은 바닥이 먼저 판을 벗어난다 (툴 자세를 유지한 역방향은 같은 끝, 툴을 돌린 역방향은 반대 끝). 끝 착점 간격 0.3 mm
    (가운데는 4 mm 로 줄여 렌더 시간 절약). 양끝 관측, 각 끝 0.25 mm, 길이 0.3 mm 안 (master 의 괄호는 15° 에서 끝 0.6 mm, 길이 0.5 mm
    까지 안쪽으로 치우쳤다). 렌더러 광선 격자는 0.25 mm (기본 2 mm 격자는 판 끝 근처 바닥 착점을 2·tan(tilt) mm 까지 잃는다 — 합성
    세계의 한계이지 끝 식별의 것이 아니다)."""
    w = _world(calib, spec)
    w.cfg = dataclasses.replace(w.cfg, z_grid_mm=(280.0, 420.0, 0.25))
    root = -w.n * DEPTH
    a, b = (w.S + root, w.E + root) if direction > 0 else (w.E + root, w.S + root)
    job = ScanJob(start=a, end=b, surface_normal=w.n, standoff_m=0.34 + DEPTH, pitch_deg=-float(tilt_deg), progression_sign=prog,
                  lead_in_m=0.001, lead_out_m=0.001, orientation="flat_1G")
    d = pass_overscan(job, calib, 0.0, top_offset_m=DEPTH)["hit_offset_m"]
    L = job.length_m
    dense, beyond = 0.008, 0.012
    hits = np.unique(np.round(np.concatenate([np.arange(-beyond, dense, SPACING_REAL), np.arange(dense, L - dense, 0.004),
                                              np.arange(L - dense, L + beyond, SPACING_REAL)]), 7))
    res = synthetic_pass(w, job, calib, hits + phase - d)
    fr = sensor_frames(job, calib)
    sf, fit, P_root, P_top, rp = fit_pass(res, spec, u_prior=job.unit_dir, n_prior=job.normal_unit, stage="fine", gravity_axis_base=job.gravity_axis_base,
                                          u_travel=fr.u_travel, target_standoff_m=0.34, end_params=None)
    ext = pass_extent_summary(fit, "S->E" if direction > 0 else "E->S", w.u)
    errs = {X: _err_mm(w, ext, X) for X in ("S", "E")}
    for X in ("S", "E"):
        assert ext[X]["observed"] and not ext[X]["occluded"], (X, ext[X])
        assert abs(errs[X]) < 0.25, (tilt_deg, direction, prog, errs)
    assert abs(ext["length_mm"] - L_SPEC * 1e3) < 0.3, (ext["length_mm"], errs)
    assert 0.00025 < fit.diag["spacing_m"] < 0.00035
    # 바닥 증거가 쓰였다 (뒷댐재 바닥이 보이는 시편): 양끝 모두 바닥 점 규칙. 모순된 상한은 한 면에 최대 1 개 — 2026-10-04 연구실 캘리브
    # (레이저 평면이 광학 y 둘레 1.3° 회전, 실제 센서) 에서 15°·E→S 툴 유지 끝 하나가 하한보다 ~0.01 mm 낮은 상한 하나를 버린다 (위 끝
    # 정확도 0.25 mm·길이 0.3 mm 는 그대로 통과; 공칭 캘리브에서는 0 개)
    for which in ("first", "last"):
        for side in fit.diag["ends"][which]["sides"].values():
            assert side["floor_rule"] == "floor_points" and side["UB_dropped"] <= 1, (which, side)


@pytest.mark.parametrize("phase_mm", [0.0, 0.75, 1.5, 2.25, 3.0, 3.75])
def test_vii_b_clamp_edge_profile_showing_the_groove_floor_is_not_a_lower_surface(calib, spec, phase_mm):
    """시야보다 넓은 클램프가 끝을 덮고 착점 간격이 e2e 처럼 4.5 mm 일 때, 샘플링 위상과 무관하게 가림으로 판정 (일반 범위 시험).
    주의: 2026-10-02 e2e 가림 끝 시험의 간헐 실패 (그루브 바닥을 '아래 면' 으로 오판) 는 이 합성 장면으로는 재현되지 않았다 — 그 수정
    (아래 면 증거는 판 대역 안의 점만) 은 실패한 e2e 기록 두 개를 오프라인 재생해 확인했다 (docs/lab_2026-10-01_findings.md §8)."""
    clamp = BoxPrim(xc=L_SPEC - 0.015 + 0.120, yc=0.0, sx=0.240, sy=0.24, h_top=0.025)     # 시야보다 넓은 클램프가 끝 15 mm 를 덮음
    w = _world(calib, spec, extras=[clamp])
    root = -w.n * DEPTH
    job = ScanJob(start=w.S + root, end=w.E + root, surface_normal=w.n, standoff_m=0.34 + DEPTH, pitch_deg=-5.0, progression_sign=1,
                  lead_in_m=0.001, lead_out_m=0.001, orientation="flat_1G", incidence_tilt_deg=0.0)
    d = pass_overscan(job, calib, 0.0, top_offset_m=DEPTH)["hit_offset_m"]
    hits = np.arange(-0.025, job.length_m + 0.040, 0.0045) + phase_mm * 1e-3
    res = synthetic_pass(w, job, calib, hits - d)
    fr = sensor_frames(job, calib)
    sf, fit, P_root, P_top, rp = fit_pass(res, spec, u_prior=job.unit_dir, n_prior=job.normal_unit, stage="fine", gravity_axis_base=job.gravity_axis_base,
                                          u_travel=fr.u_travel, target_standoff_m=0.34, end_params=None)
    ext = pass_extent_summary(fit, "S->E", w.u)
    assert ext["E"]["occluded"] and not ext["E"]["observed"] and "end_occluded_last" in fit.flags, (phase_mm, ext["E"])
    assert ext["S"]["observed"] and not ext["S"]["occluded"], (phase_mm, ext["S"])


# --------------------------------------------------------------------------------------
# 2026-10-02 at06: 판 끝 근처 개선면의 반짝이는 열변색부 반사 — 루트가 몇 mm 동안 튀고, 판 위로 산발 유령점
# --------------------------------------------------------------------------------------
def _scan_modified(world, calib, spec, *, modify_result=None, shift_root=None, monkeypatch=None, beyond=0.025, spacing=0.002):
    """_scan 과 같은 S→E 1G 패스 (착점 간격 spacing) — modify_result(res, hits) 로 프로파일을 고치거나, shift_root=(s0, s1, d) 로
    착점 s ∈ [s0, s1] 프로파일의 루트 점을 측면으로 d 만큼 옮긴다 (반사로 루트가 튀는 구간 모사)."""
    import scan_l0.autoteach as AT
    root = -world.n * DEPTH
    job = ScanJob(start=world.S + root, end=world.E + root, surface_normal=world.n, standoff_m=0.34 + DEPTH, pitch_deg=-5.0,
                  progression_sign=1, lead_in_m=0.001, lead_out_m=0.001, orientation="flat_1G", incidence_tilt_deg=0.0)
    d = pass_overscan(job, calib, 0.0, top_offset_m=DEPTH)["hit_offset_m"]
    hits = np.round(np.arange(-beyond, job.length_m + beyond, spacing), 6)
    res = synthetic_pass(world, job, calib, hits - d)
    if modify_result is not None:
        modify_result(res, hits)
    if shift_root is not None:
        s0, s1, dlat = shift_root
        orig = AT.polyline_from_scan_features

        def shifted(result, sf, centre="root"):
            P, g = orig(result, sf, centre=centre)
            if centre == "root":
                P = np.array(P, dtype=float, copy=True)
                s = (P - world.S) @ world.u
                m = np.isfinite(s) & (s >= s0) & (s <= s1)
                P[m] = P[m] + world.lateral * dlat
            return P, g

        monkeypatch.setattr(AT, "polyline_from_scan_features", shifted)
    fr = sensor_frames(job, calib)
    sf, fit, P_root, P_top, rp = fit_pass(res, spec, u_prior=job.unit_dir, n_prior=job.normal_unit, stage="fine", gravity_axis_base=job.gravity_axis_base,
                                          u_travel=fr.u_travel, target_standoff_m=0.34,
                                          end_params={"k": C.load_settings(CONFIG).judgement_k,
                                                      "ghost_burst_frames": C.load_settings(CONFIG).ghost_burst_frames})
    return fit, pass_extent_summary(fit, "S->E", world.u)


def test_reflection_stretch_inside_the_plate_is_bridged_but_not_too_long(calib, spec, monkeypatch):
    w = _world(calib, spec, table_drop_m=TABLE)
    # 4 mm 동안 루트가 3 mm 옆으로 튐 (판은 계속 보임) → 구간이 끊기지 않고 양끝 관측, 길이 = 설계
    s0 = 0.35 * L_SPEC                                                     # 가운데에서 벗어난 내부 위치
    fit, ext = _scan_modified(w, calib, spec, shift_root=(s0, s0 + 0.004, 0.003), monkeypatch=monkeypatch)
    assert ext["S"]["observed"] and ext["E"]["observed"], (ext["S"], ext["E"])
    assert abs(ext["length_mm"] - L_SPEC * 1e3) < 1.0, ext["length_mm"]
    br = fit.diag["bridged_gaps"]
    assert len(br) == 1 and "root_off_line" in br[0]["reasons"], br
    monkeypatch.undo()
    # 14 mm (> bridge_plates_present_m 10 mm) 는 잇지 않는다: 구간이 그 앞에서 끊긴다 (판 끝처럼 보이는 긴 이상은 넘지 않음)
    fit2, ext2 = _scan_modified(w, calib, spec, shift_root=(s0, s0 + 0.014, 0.003), monkeypatch=monkeypatch)
    assert ext2["length_mm"] < L_SPEC * 1e3 - 20.0 and not fit2.diag["bridged_gaps"], (ext2["length_mm"], fit2.diag["bridged_gaps"])


def test_sporadic_ghost_points_beyond_a_real_end_do_not_make_it_occluded(calib, spec):
    """판 끝 너머 (테이블 20 mm 아래) 의 처음 두 프로파일에 판 위 5 mm 유령점 — 이전 규칙 (위 점이 한 프로파일이라도 먼저 나오면 가림)
    은 끝을 'end_occluded' 로 만들었다 (이 시험은 그 규칙에서 실패함을 확인했다). 이제 연속 occlusion_min_profiles (3) 개가 필요하고
    셋째 프로파일이 테이블 (아래 면) 을 보이므로 끝은 관측."""
    w = _world(calib, spec, table_drop_m=TABLE)

    def ghost(res, hits):
        L = float(np.linalg.norm(w.E - w.S))
        for k in np.flatnonzero(hits > L)[:2]:                                # E 끝 너머 처음 두 프로파일
            x, z = res.x_mm[k], res.z_mm[k]
            cols = np.flatnonzero(np.isfinite(z) & (np.abs(x) < 20.0))
            assert cols.size > 20
            z[cols[: cols.size // 2]] = np.nanmedian(z[cols]) - (TABLE * 1e3 + 5.0)  # 테이블 높이 - 25 mm = 판보다 5 mm 위

    fit, ext = _scan_modified(w, calib, spec, modify_result=ghost)
    assert ext["E"]["observed"] and not ext["E"]["occluded"] and "end_occluded_last" not in fit.flags, ext["E"]
    assert abs(ext["length_mm"] - L_SPEC * 1e3) < 1.0


def test_a_few_above_points_next_to_a_visible_table_do_not_make_the_end_occluded(calib, spec):
    """2026-10-02 at07 패스 3: 판 끝 너머 프로파일에 판 위 점 5–7 개 (모서리 돌출·반사) 와 테이블 점 수십 개가 함께 — 위 점이 먼저 세어져
    3 연속이면 가림이 됐다. 이제 한 프로파일에서 많은 쪽 증거를 따른다: 테이블 (아래 면) 이 우세 → 끝 관측."""
    w = _world(calib, spec, table_drop_m=TABLE)

    def few_up(res, hits):
        L = float(np.linalg.norm(w.E - w.S))
        for k in np.flatnonzero(hits > L)[:3]:                                # 끝 너머 처음 세 프로파일
            x, z = res.x_mm[k], res.z_mm[k]
            cols = np.flatnonzero(np.isfinite(z) & (np.abs(x) < 20.0))
            assert cols.size > 40
            z[cols[:6]] = np.nanmedian(z[cols]) - (TABLE * 1e3 + 5.0)       # 6 점만 판보다 5 mm 위

    fit, ext = _scan_modified(w, calib, spec, modify_result=few_up)
    assert ext["E"]["observed"] and not ext["E"]["occluded"], ext["E"]
    look = fit.diag["ends"]["last"]["occlusion_look"]
    assert look and look[0][1] >= 5 and look[0][2] > look[0][1], look[:3]     # 위 점 ≥ 5 이지만 아래 면이 더 많다


@pytest.mark.parametrize("spacing_mm", [5.5, 6.0])
def test_vii_c_sparse_profiles_and_a_neutral_clamp_edge_profile_are_still_occluded(calib, spec, spacing_mm):
    """2026-10-04 e2e 가림 끝 시험 (CPU 부하로 착점 간격 5.7–6 mm): 시야보다 넓은 클램프가 끝을 덮고, 끝 너머 20 mm 창에 프로파일이 3 개뿐인데
    첫 것이 클램프 벽에 걸쳐 판 높이 점 몇 개뿐 (위·아래 증거 없음) — 이전 규칙은 연속 요구 개수를 창 안 프로파일 수 (3) 로 줄일 뿐이라
    남은 클램프 2 개로는 가림이 안 됐고, 클램프 모서리를 판 끝으로 '관측' 했다 (길이 363 mm, end_ambiguous). 이제 끝 바로 너머의 증거 없는
    프로파일은 그 상한에서 뺀다 (이 시험은 이전 규칙에서 실패함을 확인했다)."""
    clamp = BoxPrim(xc=L_SPEC - 0.015 + 0.120, yc=0.0, sx=0.240, sy=0.24, h_top=0.025)     # 시야보다 넓은 클램프가 끝 15 mm 를 덮음
    w = _world(calib, spec, extras=[clamp])
    seen = {}

    def neutral_edge(res, hits):
        mid = int(np.argmin(np.abs(hits - 0.5 * L_SPEC)))
        x0, z0 = res.x_mm[mid], res.z_mm[mid]
        plate = np.isfinite(z0) & (np.abs(x0) > 25.0)
        z_plate = float(np.nanmedian(z0[plate]))                              # 판 상면 높이 (센서 z)
        for k in np.argsort(hits):
            x, z = res.x_mm[k], res.z_mm[k]
            if hits[k] > 0.5 * L_SPEC and np.sum(np.isfinite(z) & (z < z_plate - 5.0)) > 100:   # 클램프를 보는 첫 프로파일
                keep = np.isfinite(x0) & np.isfinite(z0) & (np.abs(x0) >= 30.0) & (np.abs(x0) <= 45.0)
                z[:] = np.nan
                x[:] = np.nan
                x[keep], z[keep] = x0[keep], z0[keep]                         # 판 높이 점만 (모서리 걸침·그늘) — 가운데 프로파일의 실제 판 점
                seen["k"] = int(k)
                return

    fit, ext = _scan_modified(w, calib, spec, modify_result=neutral_edge, spacing=spacing_mm * 1e-3)
    look = fit.diag["ends"]["last"]["occlusion_look"]
    assert "k" in seen and len(look) == 3 and look[0][1:] == [0, 0] and min(n[1] for n in look[1:]) > 100, look
    assert ext["E"]["occluded"] and not ext["E"]["observed"] and "end_occluded_last" in fit.flags, (look, ext["E"])
    assert ext["S"]["observed"] and not ext["S"]["occluded"], ext["S"]
