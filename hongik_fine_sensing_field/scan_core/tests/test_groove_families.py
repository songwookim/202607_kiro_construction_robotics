"""그루브 종류 일반화 (사용자 2026-10-04: 로직을 먼저 세우고 데이터로 검증 — 우리 시편만이 아니라 V형·루트면·두꺼운 판·넓은 그루브에 같은
로직, 자세·형상은 사전 정보). 같은 단면 모델 (벽 = 루트면 rf + 개선면 β) 과 같은 계획 모델 (비드 = 질량 보존, 층 = 등면적·폭 조건,
캡 = 위빙 폭) 이 모든 종류에서 성립하는지를 정답을 아는 합성 데이터로 본다:

  추출기: 광선 추적 프로파일 (롤 0/±4° — 운용 게이트 거친 1.5°·정밀 0.5° 의 여유, 노이즈 0.02 mm, 드롭아웃 2 %) 에서 갭·개선각·깊이·
          개구, 루트면을 넣은 기하 항등식. 한계 (2026-10-04 측정): 좁은 갭 (3 mm) 단일 베벨에 롤 ±8° 면 직각 벽 그늘로 갭 오차가
          ~0.45 mm 까지 (거친 게이트 밖이라 운용에서 생기지 않는다), 25T/45° 단일 베벨은 롤 0 에서도 갭 +0.16–0.24 mm, 깊이 −0.23–0.30 mm (국소 판 직선이 개선면 시작을 포함 — seam_features 주석).
  계획기: 질량 보존 (단면마다 Σ 패스 단면적 = 그루브 + 보강), 층 비드 수 = 폭을 덮는 최소 개수, 층 안 비드 등면적, 속도 창,
          겨냥 순서·옆 비드 토치 기울기, 두께에 따른 패스 수 단조성, 비드 모델 교체, KIRO 공정이 특수해로 나오는지 (일관성 — 독립 검증은
          KIRO 데이터셋으로 따로)."""
import math

import numpy as np
import pytest

from scan_core.bead_model import MassBalanceBeadModel, PassParams
from scan_core.frames import make_transform, rot_x
from scan_core.path_gen import plate_frame
from scan_core.seam_features import extract_cross_section_features
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, simulate_profile
from scan_core.weld_plan import GrooveStation, WeldProcess, plan_topology, plan_weld

T_PLATE = plate_frame([0, 0, 0], [1, 0, 0], [0, 0, 1])
R0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)       # X_s = plate y (라인이 그루브를 가로지름), Z_s = -plate z

FAMILIES = {
    "bevel30_10T_g3_rf2_L": GrooveSpec(kind="single_bevel", thickness_mm=10, bevel_angle_deg=30, root_gap_mm=3, root_face_mm=2,
                                       bevel_side="left"),
    "bevel30_10T_g3_rf2_R": GrooveSpec(kind="single_bevel", thickness_mm=10, bevel_angle_deg=30, root_gap_mm=3, root_face_mm=2,
                                       bevel_side="right"),
    "V30_12T_g2_rf2": GrooveSpec(kind="single_v", thickness_mm=12, bevel_angle_deg=30, root_gap_mm=2, root_face_mm=2),
    "V30_20T_g3_rf2": GrooveSpec(kind="single_v", thickness_mm=20, bevel_angle_deg=30, root_gap_mm=3, root_face_mm=2),
    "V25_16T_g1.5_rf3": GrooveSpec(kind="single_v", thickness_mm=16, bevel_angle_deg=25, root_gap_mm=1.5, root_face_mm=3),
    "bevel45_25T_g6_R": GrooveSpec(kind="single_bevel", thickness_mm=25, bevel_angle_deg=45, root_gap_mm=6, root_face_mm=0,
                                   bevel_side="right"),
    "V30_25T_g4_rf2": GrooveSpec(kind="single_v", thickness_mm=25, bevel_angle_deg=30, root_gap_mm=4, root_face_mm=2),
}
# KIRO 데이터셋 (2026-09-15/16 weld_testspecimen_logs): 패스 1–3 뒤 그루브 단면적 비율과 층 윗면 — 실측 심 속도의 질량 보존 단면적
# (30.0 / 27.6 / 27.6 mm², η 0.95) 을 우리 10T 시편의 잰 그루브 (k20_2_a 기준 단면, 85.5 mm²) 에 쌓은 값 (fsv1_work/kiro_dataset)
KIRO_REF = GrooveSpec(kind="single_bevel", thickness_mm=10.514106537212687, bevel_angle_deg=30.49794880937114, bevel_side="left",
                      root_gap_mm=5.199548388145472, root_face_mm=0.2804609679147437, hi_lo_mm=-0.6942210186257858)
KIRO_FRACTIONS = (30.02 / 85.51, 57.60 / 85.51, 85.24 / 85.51)
# the convex beads (bead_geometry, 2026-10-13 night; toe angle 30 deg) of those deposits on that section: level toe chords and crowns
# [mm] (the flat layers of the old model were at 4.72 / 7.91 / 10.58 mm - between a bead's toes and its crown)
KIRO_TOES_MM = (4.03, 7.04, 9.56)
KIRO_CROWNS_MM = (5.03, 8.27, 10.99)


def _profile(spec: GrooveSpec, roll_deg: float, seed: int = 1):
    """판 상면 위 ~430 mm 에서 그루브를 가운데 두고 (auto-teach 가 센서를 그루브 위에 둔다) 롤 roll_deg 로 본 프로파일."""
    z = 0.43 + spec.thickness_mm * 1e-3
    y_off = -z * math.tan(math.radians(roll_deg))                   # 광축이 루트갭 중심을 지나도록
    T = make_transform(rot_x(math.radians(roll_deg)) @ R0, (0.0, y_off, z))
    return simulate_profile(T, GrooveScene.from_spec(spec), T_PLATE,
                            SynthConfig(n_points=1408, seed=seed, sigma_z_mm=0.02, dropout_prob=0.02))


# --------------------------------------------------------------------------------------
# 추출기: 단면 모델 (루트면 + 개선면)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("roll", [0.0, 4.0, -4.0])
@pytest.mark.parametrize("name", list(FAMILIES))
def test_extractor_recovers_every_groove_family(name, roll):
    sp = FAMILIES[name]
    x, z = _profile(sp, roll)
    f = extract_cross_section_features(x, z, sp)
    assert f.ok and f.groove_found, (name, f.flags, f.notes)
    assert f.tilt_deg == pytest.approx(roll, abs=0.15)
    assert abs(f.root_gap_mm - sp.root_gap_mm) <= 0.25, f.report()
    assert abs(f.bevel_angle_deg - sp.bevel_angle_deg) <= 0.5
    assert abs(f.depth_mm - sp.thickness_mm) <= 0.35                     # 25T/45° 단일 베벨 최대 −0.30 mm (국소 판 직선 한계, seam_features)
    assert abs(f.top_opening_mm - sp.top_opening_mm) <= 0.25
    assert f.bevel_side == (sp.bevel_side if sp.kind == "single_bevel" else "both")
    # 기하 항등식 (루트면 포함): 개구 − 갭 − n_b·(깊이 − rf)·tanβ ≈ 0
    ident = f.top_opening_mm - f.root_gap_mm - sp.n_bevels * (f.depth_mm - sp.root_face_mm) * math.tan(math.radians(f.bevel_angle_deg))
    assert abs(ident) <= 0.15, ident


@pytest.mark.parametrize("name", ["bevel30_10T_g3_rf2_L", "V30_12T_g2_rf2", "V25_16T_g1.5_rf3"])
def test_root_edge_needs_the_design_root_face(name):
    """루트 모서리 = 개선면 ∩ (바닥 + rf). rf 를 모르고 (0) 개선면을 바닥까지 외삽하면 갭이 모델대로 n_b·rf·tanβ 만큼 작아진다
    (2026-10-04 이전 동작: 12T V rf 2 에서 갭 2 → −0.3 mm). 그래서 rf 는 사전 정보 (job specimen 블록) 로 받는다."""
    sp = FAMILIES[name]
    x, z = _profile(sp, 2.0)
    good = extract_cross_section_features(x, z, sp)
    no_rf = GrooveSpec(**{**{k: getattr(sp, k) for k in ("kind", "thickness_mm", "bevel_angle_deg", "bevel_side", "root_gap_mm")},
                          "root_face_mm": 0.0})
    bad = extract_cross_section_features(x, z, no_rf)
    bias = sp.n_bevels * sp.root_face_mm * math.tan(math.radians(sp.bevel_angle_deg))
    assert abs(good.root_gap_mm - sp.root_gap_mm) <= 0.25
    assert bad.root_gap_mm == pytest.approx(good.root_gap_mm - bias, abs=0.1)


# --------------------------------------------------------------------------------------
# 계획기: 층·비드 배치
# --------------------------------------------------------------------------------------
def _stations(sp: GrooveSpec, n: int = 9, step: float = 10.0, gap_ramp: float = 0.0) -> list:
    """곧은 심 (x 방향), 판 법선 +z. gap_ramp: 끝에서 갭이 이만큼 커진다 (단면마다 다른 단면적)."""
    u, nn = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    out = []
    for i in range(n):
        s = i * step
        spi = GrooveSpec(**{**{k: getattr(sp, k) for k in ("kind", "thickness_mm", "bevel_angle_deg", "bevel_side", "root_face_mm",
                                                           "backing")},
                            "root_gap_mm": sp.root_gap_mm + gap_ramp * i / max(n - 1, 1)})
        out.append(GrooveStation(s_mm=s, origin=np.array([0.5 + s * 1e-3, 0.0, 0.1]), u=u, n=nn, w=np.cross(nn, u), spec=spi, n_profiles=9))
    return out


def test_kiro_procedure_is_a_special_case_of_the_model():
    """우리 10T 시편의 잰 기준 단면 (k20_2_a) + 기본 WPS (= KIRO 데이터셋: root 8.4 / fill 10.8 / cap 10.2 m/min, 5 / 7 / 6 mm/s, 와이어
    1.2 mm, η 0.95) → 한 비드 충전 3 층 + 위빙 캡 1, 층까지 단면적 비율 ≈ KIRO 패스의 질량 보존 값 (±0.01), 볼록 비드의 토우·크라운
    (±0.02 mm), KIRO 속도 그대로 (편차 < 2 %). 일관성 확인 — KIRO 의 그루브 도면이 없어 같은 10T 설계로 가정한다 (WPS_PROVENANCE)."""
    tp = plan_topology(KIRO_REF, WeldProcess())
    assert [ly.n_beads for ly in tp.layers] == [1, 1, 1] and tp.n_cap == 1 and tp.feasible
    assert [ly.role for ly in tp.layers] == ["root", "fill", "fill"]
    assert [ly.frac_top for ly in tp.layers] == pytest.approx(KIRO_FRACTIONS, abs=0.01)
    assert [ly.z_top_ref_mm for ly in tp.layers] == pytest.approx(KIRO_TOES_MM, abs=0.02)
    assert [ly.z_crown_ref_mm for ly in tp.layers] == pytest.approx(KIRO_CROWNS_MM, abs=0.02)
    assert abs(math.log(tp.deposit_scale)) < 0.02 and abs(math.log(tp.cap_scale)) < 0.02          # WPS 그대로 (속도 편차 < 2 %)


@pytest.mark.parametrize("name", list(FAMILIES) + ["V30_40T_g4_rf2"])
def test_layer_plan_conserves_area_and_respects_widths_and_speeds(name):
    sp = FAMILIES.get(name) or GrooveSpec(kind="single_v", thickness_mm=40, bevel_angle_deg=30, root_gap_mm=4, root_face_mm=2)
    proc = WeldProcess(speed_limits=(0.6, 1.6))                    # 적응 속도 창 (계획 모형의 일반성; KIRO 설정 고정은 test_weld_plan)
    bm = proc.bead_model()
    st = _stations(sp, gap_ramp=0.6)
    plan = plan_weld(st, proc)
    tp = plan.topology
    assert tp.feasible and not any("speed window" in n_ for n_ in plan.notes), plan.notes
    lo, hi = proc.speed_limits
    # 기준 단면: 층 비드 수 = one bead when its toe chord is within its free width, else the smallest n whose layer pool is no
    # wider than n bead widths; 비드 단면적은 속도 창 안
    for ly in tp.layers:
        if ly.n_beads == 1:
            assert ly.width_top_ref_mm <= ly.bead_width_mm + 1e-9
        else:
            assert (ly.n_beads - 1) * ly.bead_width_mm < ly.width_top_ref_mm <= ly.n_beads * ly.bead_width_mm + 1e-9
        assert 1.0 / hi - 1e-9 <= ly.bead_area_ref_mm2 / ly.bead_area_nom_mm2 <= 1.0 / lo + 1e-9
    assert 1.0 / hi - 1e-9 <= tp.cap_scale <= 1.0 / lo + 1e-9
    assert tp.cap_width_ref_mm / tp.n_cap <= proc.cap_weave_width_max_mm + 1e-9
    assert len(plan.passes) == tp.n_passes and [p.k for p in plan.passes] == list(range(1, tp.n_passes + 1))
    # 단면마다 질량 보존: each pass's convex bead holds its deposit; Σ over the passes = the groove + the reinforcement area (mass
    # balance: the deposits beyond the groove); the layers rise (toe levels) and stay within the groove's top edges
    from scan_core.bead_geometry import groove_edges_mm
    for i, s in enumerate(st):
        spi = s.spec
        js = {p.k: int(np.flatnonzero(p.s_mm == s.s_mm)[0]) for p in plan.passes}
        for p in plan.passes:
            assert p.beads[js[p.k]].area_mm2 == pytest.approx(float(p.area_mm2[js[p.k]]), rel=1e-6, abs=1e-9)
        if tp.layers:
            assert np.all(np.diff(plan.levels_mm[i]) > 0)
            assert plan.levels_mm[i][-1] <= max(groove_edges_mm(spi)[1], groove_edges_mm(spi)[3]) + 1e-9
        caps = [p for p in plan.passes if p.kind == "cap"]
        assert float(caps[-1].reinforcement_mm[js[caps[-1].k]]) > 0.0             # the caps cover the groove
    # 속도: WFS 고정의 역산 (창 밖이면 자르고 표시)
    for p in plan.passes:
        prm = proc.params(p.role)
        ratio = bm.area_mm2(prm) / p.area_mm2
        assert p.speed_mps == pytest.approx(prm.speed_mps * np.clip(ratio, lo, hi))
        assert np.array_equal(p.speed_clamped, (ratio < lo - 1e-9) | (ratio > hi + 1e-9))


@pytest.mark.parametrize("name", ["V30_20T_g3_rf2", "bevel45_25T_g6_R", "V30_25T_g4_rf2"])
def test_multi_bead_layers_split_the_layer_into_equal_areas(name):
    """Thick plates / wide grooves: multi-bead layers (the geometric rule - KIRO's practice covers single-bead layers): the layer's pool
    cut into equal-area bands, one bead per band, aimed at the band centres left to right inside the walls, each bead (convex) inside
    its band holding its area; the wall beads lean toward their wall (side_work_angle), the middle ones upright; caps weave."""
    sp = FAMILIES[name]
    proc = WeldProcess()
    plan = plan_weld(_stations(sp), proc)
    multi = [ly for ly in plan.topology.layers if ly.n_beads > 1]
    assert multi, [ly.n_beads for ly in plan.topology.layers]                   # 두꺼운 판·넓은 그루브 → 다비드 층
    st = plan.stations[4]
    for j, ly in enumerate(plan.topology.layers):
        beads = [p for p in plan.passes if p.layer == j + 1]
        assert [p.bead for p in beads] == list(range(1, ly.n_beads + 1)) and all(p.kind == "fill" for p in beads)
        jj = int(np.flatnonzero(beads[0].s_mm == st.s_mm)[0])
        xs = [float(p.aim_xz_mm[jj, 0]) for p in beads]
        areas = [float(p.area_mm2[jj]) for p in beads]
        assert areas == pytest.approx([areas[0]] * len(areas), rel=1e-12)      # equal beads
        if ly.n_beads > 1:
            assert np.all(np.diff(xs) > 0)                                       # 단면 −x 벽 → +x 벽 순서
            b = [p.beads[jj] for p in beads]
            assert all(bb.area_mm2 == pytest.approx(a, rel=1e-6) for bb, a in zip(b, areas))
            assert all(b[q].toe_r[0] <= b[q + 1].toe_l[0] + 1e-9 for q in range(len(b) - 1))   # side by side, no overlap
            assert all(bb.toe_l[0] <= x <= bb.toe_r[0] for bb, x in zip(b, xs))              # each aimed inside its own bead
            assert all(p.aim_lateral_rule == "geometric" for p in beads)
            # 옆 비드는 제 벽 쪽으로 side_work_angle 만큼 (AWS 작업각: 토치 축의 단면 가로 성분 = ±cos(진행각)·sin(작업각); 진행각 = 그 층의
            # 값 — WPS 의 충전 진행각은 층 순서별), 가운데 비드는 기울지 않음
            for p in beads:
                s_side = math.cos(math.radians(p.travel_angle_deg)) * math.sin(math.radians(proc.side_work_angle_deg))
                lat = float(p.R_base_tcp[jj][:, 2] @ st.w)
                want = -s_side if p.bead == 1 else s_side if p.bead == ly.n_beads else 0.0
                assert lat == pytest.approx(want, abs=1e-9), (p.k, p.bead)
    caps = [p for p in plan.passes if p.kind == "cap"]
    assert len(caps) == plan.topology.n_cap and all(np.all(p.weave_amp_mm > 0) for p in caps)


def test_pass_count_grows_with_plate_thickness():
    proc = WeldProcess()
    n = [plan_topology(GrooveSpec(kind="single_v", thickness_mm=t, bevel_angle_deg=30, root_gap_mm=3, root_face_mm=2), proc).n_passes
         for t in (8, 12, 16, 20, 25, 30, 40)]
    assert n == sorted(n) and n[0] <= 3 and n[-1] >= 30, n


def test_bead_model_is_replaceable():
    """계획기는 BeadModel 인터페이스만 쓴다 (KIRO 데이터 기반 모델 자리). 비드가 두 배 크면 패스가 줄고, 모델은 자세·장비 설정을 받는다."""
    seen = []

    class Doubled(MassBalanceBeadModel):
        def area_mm2(self, p: PassParams) -> float:
            seen.append((p.role, p.position, dict(p.settings)))
            return 2.0 * super().area_mm2(p)

        def speed_for_area(self, p: PassParams, area_mm2: float) -> float:
            return 2.0 * super().speed_for_area(p, area_mm2)
    sp = FAMILIES["V30_20T_g3_rf2"]
    proc = WeldProcess(equipment={"voltage_V": [22.0, 24.0, 21.0], "ctwd_mm": 15.0})
    base = plan_weld(_stations(sp), proc, position="3G")
    big = plan_weld(_stations(sp), proc, position="3G", bead_model=Doubled())
    assert len(big.passes) < len(base.passes)
    assert ("fill", "3G", {"voltage_V": 24.0, "ctwd_mm": 15.0}) in seen and ("cap", "3G", {"voltage_V": 21.0, "ctwd_mm": 15.0}) in seen
    assert big.bead_model["model"] == "mass_balance" and base.to_dict()["process"]["equipment"]["ctwd_mm"] == 15.0


def test_wps_that_cannot_fill_the_groove_is_flagged():
    """3T 맞대기 (갭 1) 에 기본 WPS (비드 ~23 mm²) — 어떤 배치도 속도 창 안에 못 든다: 가장 가까운 배치 + 표시."""
    tp = plan_topology(GrooveSpec(kind="square", thickness_mm=3, bevel_angle_deg=0, root_gap_mm=1), WeldProcess())
    assert not tp.target_met and any("speed window" in n_ for n_ in tp.notes) and tp.n_passes == 1
