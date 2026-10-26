"""weld_plan: 용접 전 스캔 → 다층 용접 경로 (2026-10-03; 2026-10-04 모델 기반 층·비드 배치). 정답을 아는 합성 특징 (갭·개구가 심
방향으로 바뀌는 단면 베벨 그루브, 우리 시편 10T/30°/갭 5) 으로 단면 복원·층 높이 (bead_fill 과 같은 단면적 정의)·겨냥점·속도·위빙·
토치 자세·진행 방향·저장 형식을 확인한다. 다른 그루브 종류 (V형·루트면·두꺼운 판) 는 test_groove_families.py."""
import json
import math

import numpy as np
import pytest

from scan_core.bead_fill import groove_area_below_mm2, wall_x_mm
from scan_core.specimen import GrooveSpec
from scan_core.weld_plan import (PlanContractError, WeldProcess, check_plan_contract, format_plan_summary, plan_frame, plan_topology,
                                 plan_weld, stations_from_features, weld_plan_from_features)

L_MM = 380.0
U = np.array([0.9953, 0.0969, 0.0]) / np.linalg.norm([0.9953, 0.0969, 0.0])
N = np.array([-0.0149, -0.0195, 0.9997]) / np.linalg.norm([-0.0149, -0.0195, 0.9997])
C0 = np.array([1.24, -0.01, -0.17])


def gap_true(s):
    return 5.0 + 0.4 * np.asarray(s) / L_MM                   # 5.0 → 5.4


def open_true(s):
    return 11.0 + 0.5 * np.sin(np.asarray(s) / L_MM * math.pi)  # 11.0 → 11.5 → 11.0


def synth_feats(*, depth=10.0, hi_lo=-0.15, sensor_flip=False, spacing=0.3, missing=None, s0_line=-0.19, U=U, N=N, C0=C0):
    """프로파일 특징 사전 (features.npz 와 같은 키). 단면 베벨은 단면 좌표 −x (w = n × u 의 반대쪽) — at12 와 같은 배치.
    U·N·C0 를 주면 다른 배치 (예: 3G 세운 판)."""
    U, N, C0 = (np.asarray(v, dtype=float) for v in (U, N, C0))
    u = U - (U @ N) * N
    u /= np.linalg.norm(u)
    w = np.cross(N, u)
    s = np.arange(-15.0, L_MM + 15.0, spacing)                    # 판 끝 밖 프로파일도 (NaN 이 아닌 값; 계획은 S..E 만 쓴다)
    g, o = gap_true(s), open_true(s)
    run = o - g
    P_root = C0 + np.outer(s0_line + s * 1e-3, u)
    P_top = P_root + np.outer(-0.5 * run * 1e-3, w) + depth * 1e-3 * N
    ok = np.ones(s.size, dtype=bool)
    if missing is not None:
        ok &= ~((s >= missing[0]) & (s <= missing[1]))
    return {"line_c_base_m": C0, "u_base": u, "n_base": N, "root_polyline_base_m": P_root, "top_polyline_base_m": P_top,
            "root_gap_mm": g, "depth_mm": np.full(s.size, depth), "top_opening_mm": o,
            "hi_lo_mm": np.full(s.size, -hi_lo if sensor_flip else hi_lo), "ok": ok,
            "bevel_side": np.array(["right" if sensor_flip else "left"] * s.size)}


def ends(f, s0_line=-0.19):
    """합성 특징의 관측 끝 S, E (base 점) — 루트 직선 위 s = 0, L."""
    c, u = np.asarray(f["line_c_base_m"], dtype=float), np.asarray(f["u_base"], dtype=float)
    return c + s0_line * u, c + (s0_line + L_MM * 1e-3) * u


def extent(f, s0_line=-0.19, sigma_m=0.1e-3):
    """패스 extent 와 같은 모양 (autoteach.pass_extent_summary): 끝 = base 점 + 그 σ, 착점 간격."""
    S, E = ends(f, s0_line)
    return {"S": {"point": S.tolist(), "sigma_m": sigma_m, "observed": True},
            "E": {"point": E.tolist(), "sigma_m": sigma_m, "observed": True}, "spacing_mm": 0.3}


def frame(f, s0_line=-0.19):
    return plan_frame(f, *ends(f, s0_line))


SPEC = GrooveSpec(bevel_side="left")


def test_sections_recover_the_groove_and_its_frame():
    f0 = synth_feats()
    st = stations_from_features(f0, SPEC, frame(f0))
    assert len(st) == 77 and st[0].s_mm == 0.0 and st[-1].s_mm == pytest.approx(L_MM)
    S = np.array([x.s_mm for x in st])
    G = np.array([x.spec.root_gap_mm for x in st])
    O = np.array([x.measured["top_opening_mm"] for x in st])
    assert np.max(np.abs(G - gap_true(S))) < 0.01 and np.max(np.abs(O - open_true(S))) < 0.01
    for x in st:
        assert x.spec.bevel_side == "left" and x.spec.thickness_mm == pytest.approx(10.0) and x.spec.hi_lo_mm == pytest.approx(-0.15)
        assert x.spec.bevel_angle_deg == pytest.approx(math.degrees(math.atan((x.measured["top_opening_mm"] - x.spec.root_gap_mm) / 10.0)))
        # 원점 = 루트갭 중심 (정련 직선 위, S 끝에서 s)
        assert np.linalg.norm(x.origin - (C0 + (-0.19 + x.s_mm * 1e-3) * x.u)) < 1e-9
        assert abs(x.u @ x.n) < 1e-12 and np.allclose(x.w, np.cross(x.n, x.u))
    # 센서가 반대로 놓여도 (센서 x = −w) 베벨 쪽·hi-lo 부호는 단면 좌표로 같다
    f2 = synth_feats(sensor_flip=True)
    st2 = stations_from_features(f2, SPEC, frame(f2))
    assert all(x.spec.bevel_side == "left" and x.spec.hi_lo_mm == pytest.approx(-0.15) for x in st2)
    # 유효 프로파일이 없는 구간은 이웃에서 보간하고 표시한다
    f3 = synth_feats(missing=(100.0, 130.0))
    st3 = stations_from_features(f3, SPEC, frame(f3))
    flagged = [x.s_mm for x in st3 if x.interpolated]
    assert flagged and min(flagged) >= 100.0 and max(flagged) <= 130.0
    assert all(abs(x.spec.root_gap_mm - gap_true(x.s_mm)) < 0.02 for x in st3)


def test_four_pass_plan_layers_aims_speeds_and_torch():
    """계획 모형 (적응 속도 창 0.6..1.6 — 층까지의 용착 = 같은 그루브 단면적 비율, 단면마다 속도로 맞춘다) 에 KIRO 의 비드·각·위빙, 볼록
    비드 (bead_geometry) 와 KIRO 의 겨냥 관행 (2026-10-13 밤)."""
    proc = WeldProcess(speed_limits=(0.6, 1.6))
    bm = proc.bead_model()
    f = synth_feats()
    plan = weld_plan_from_features(f, extent(f), SPEC, proc)
    # 우리 시편 (10T/30°/갭 ~5.2) 에서 기본 WPS (KIRO) → 한 비드 충전 3 층 (root + fill 2) + 위빙 캡 1 (KIRO 공정과 같은 배치)
    assert [p.kind for p in plan.passes] == ["fill", "fill", "fill", "cap"] and plan.bevel_side == "left"
    assert [p.role for p in plan.passes] == ["root", "fill", "fill", "cap"] and all(p.n_beads == 1 for p in plan.passes)
    assert plan.bevel_view == "right" and plan.length_mm == pytest.approx(L_MM)
    F = [ly.frac_top for ly in plan.topology.layers]
    for i, st in enumerate(plan.stations):
        sp = st.spec
        A = float(groove_area_below_mm2(sp, sp.thickness_mm))
        filled = sum(float(p.area_mm2[p.s_mm == st.s_mm][0]) for p in plan.passes[:3])
        assert filled == pytest.approx(F[-1] * A, rel=1e-6)                         # 단면마다 같은 단면적 비율 (bead_fill 정의)
        g, rf = 0.5 * sp.root_gap_mm, sp.root_face_mm
        beads = []
        for k, p in enumerate(plan.passes[:3]):
            j = int(np.flatnonzero(p.s_mm == st.s_mm)[0])
            b = p.beads[j]
            beads.append(b)
            assert b.area_mm2 == pytest.approx(p.area_mm2[j], rel=1e-6)            # the convex bead holds the pass's deposit
            assert b.toe_l[1] == pytest.approx(b.toe_r[1]) and plan.levels_mm[i][k] == pytest.approx(b.toe_l[1])   # level toe chord
            assert p.layer_to_mm[j] == pytest.approx(b.apex[1]) and p.layer_from_mm[j] == pytest.approx(float(np.min(b.under_z)))
            assert b.apex[1] - b.toe_l[1] == pytest.approx(proc.bead_aspect_ratio * b.chord_mm, rel=1e-9)    # the arc's height
            x, z = p.aim_xz_mm[j]
            # KIRO's practice (bevel on -x here): root = the root-face / bevel corner, fill 1 = the root bead's toe on the square
            # wall, fill 2 = midway between fill 1's toes on its surface
            want = [(-g, rf), (beads[0].toe_r[0], beads[0].toe_r[1]) if k > 0 else None,
                    (0.5 * (beads[1].toe_l[0] + beads[1].toe_r[0]), None) if k > 1 else None][k]
            assert x == pytest.approx(want[0], abs=1e-9)
            # (on an arc: the surface's polyline, sampled every SECTION_STEP_MM, lies under the arc by <= step^2 / (8 R) ~ 3e-4 mm)
            assert z == pytest.approx(want[1] if want[1] is not None else float(beads[1].top(x)), abs=1e-3)
            assert np.linalg.norm(p.aim_base_m[j] - st.to_base(x, z)) < 1e-12
            prm = proc.params(p.role)
            v_expect = prm.speed_mps * np.clip(bm.area_mm2(prm) / p.area_mm2[j], *proc.speed_limits)    # WFS 고정 → 단면적에 반비례
            assert p.speed_mps[j] == pytest.approx(v_expect)
            assert p.speed_required_mps[j] == pytest.approx(bm.speed_for_area(prm, p.area_mm2[j]))
        assert beads[0].toe_r[0] == pytest.approx(g) and beads[1].toe_l[1] > beads[0].apex[1]   # fill 1 rose over the root's crown
        cap = plan.passes[3]
        j = int(np.flatnonzero(cap.s_mm == st.s_mm)[0])
        xl, xr = (float(v) for v in wall_x_mm(sp, sp.thickness_mm))
        assert cap.weave_width_mm[j] == pytest.approx(xr - xl + 2 * proc.cap_overlap_mm)
        # 위빙 진폭 (단면 가로) = WPS 의 토치 y 방향 진폭 (KIRO ±4 mm) 의 사영
        assert cap.weave_amp_mm[j] == pytest.approx(4.0 * abs(float(cap.R_base_tcp[j][:, 1] @ st.w)))
        from scan_core.bead_geometry import groove_edges_mm
        xeL, _tL, xeR, _tR = groove_edges_mm(sp)
        assert cap.aim_xz_mm[j, 0] == pytest.approx(0.5 * (xeL + xeR))           # the groove top's lateral centre (user 2026-10-13 night)
        assert cap.aim_xz_mm[j, 1] == pytest.approx(float(beads[2].top(cap.aim_xz_mm[j, 0])), abs=1e-3)   # on fill 2's convex top
    assert [p.aim_lateral_rule for p in plan.passes] == ["kiro_practice"] * 4
    # 토치: 직교, z = 토치 축 — AWS A3.0 의 각: 진행각 = a 와 진행에 수직인 면 사이 각 (+ 미는 각, − 끄는 각), 작업각 = 그 면에서 −n 에서
    # 베벨 쪽 (−w, 베벨 판 왼쪽) 으로 WPS 값 (KIRO 9 월 용접, 베벨은 KIRO 진행의 왼쪽: root −42 / +9°, 첫 충전 −41 / −20°, 둘째 충전
    # −44 / +7°, 캡 −44 / +7° — 역할의 층 순서별 값), x = 진행 방향
    assert [(p.travel_angle_deg, p.work_angle_deg) for p in plan.passes] == [(-42.0, 9.0), (-41.0, -20.0), (-44.0, 7.0), (-44.0, 7.0)]
    for p in plan.passes:
        for i, R in list(enumerate(p.R_base_tcp))[::10]:
            st = plan.stations[int(np.flatnonzero([s.s_mm == p.s_mm[i] for s in plan.stations])[0])]
            a = R[:, 2]
            ds = st.u if p.direction == "S->E" else -st.u
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
            assert math.degrees(math.asin(float(a @ ds))) == pytest.approx(p.travel_angle_deg, abs=1e-6)
            assert math.degrees(math.atan2(float(-a @ st.w), float(-a @ st.n))) == pytest.approx(p.work_angle_deg, abs=1e-6)
            assert float(a @ ds) < 0.0 and float(R[:, 0] @ ds) == pytest.approx(math.sqrt(1.0 - float(a @ ds) ** 2))   # 끄는 각; x ⟂ a, 진행 쪽
    # 일정 속도 하나 (KIRO 실행기): 필요 단면적의 심 방향 평균 (사다리꼴) 에서 그 역할의 WFS 로, 단면별 필요 속도 범위 안
    for p in plan.passes:
        o = np.argsort(p.s_mm)
        a_mean = float(np.trapz(p.area_required_mm2[o], p.s_mm[o]) / (p.s_mm[o][-1] - p.s_mm[o][0]))
        assert p.area_mean_mm2 == pytest.approx(a_mean) and not p.speed_const_clamped
        assert p.speed_const_mps == pytest.approx(bm.speed_for_area(proc.params(p.role), a_mean))
        assert np.min(p.speed_required_mps) <= p.speed_const_mps <= np.max(p.speed_required_mps)
    assert any(line.startswith("  one constant speed per pass") for line in plan.summary_lines())


def test_speed_limits_and_travel_directions():
    f = synth_feats()
    wide = np.where(np.arange(f["root_gap_mm"].size) > 900, 8.0, 0.0)               # 한쪽 끝이 크게 벌어짐 (단면적 ×2)
    f["root_gap_mm"] = f["root_gap_mm"] + wide
    f["top_opening_mm"] = f["top_opening_mm"] + wide
    plan = weld_plan_from_features(f, extent(f), SPEC, WeldProcess(direction="alternate", speed_limits=(0.6, 1.6)))
    assert [p.direction for p in plan.passes] == ["S->E", "E->S", "S->E", "E->S"]
    p1 = plan.passes[0]
    assert p1.speed_clamped.any() and np.min(p1.speed_mps) == pytest.approx(0.005 * 0.6)
    assert any("need a speed outside the window" in line for line in plan.summary_lines())
    p2 = plan.passes[1]
    assert p2.s_mm[0] == pytest.approx(L_MM) and p2.dense_s_mm[0] == pytest.approx(L_MM) and p2.dense_s_mm[-1] == pytest.approx(0.0)
    # 3G 위보기 (uphill): E 끝이 아래면 E → S
    g = (U * 0.9 + N * 0.1)                                     # 중력이 +u 쪽 (E 가 아래)
    f0 = synth_feats()
    up = weld_plan_from_features(f0, extent(f0), SPEC, WeldProcess(direction="uphill"), gravity_axis_base=g / np.linalg.norm(g))
    assert all(p.direction == "E->S" for p in up.passes)


def test_dense_paths_fill_and_weave():
    proc = WeldProcess(start_inset_mm=2.0, end_inset_mm=3.0)
    f = synth_feats()
    plan = weld_plan_from_features(f, extent(f), SPEC, proc)
    p1 = plan.passes[0]
    assert p1.dense_s_mm[0] == pytest.approx(2.0) and p1.dense_s_mm[-1] == pytest.approx(L_MM - 3.0)
    assert np.all(np.diff(p1.dense_t_s) > 0) and np.max(np.abs(np.diff(p1.dense_s_mm))) <= proc.dense_step_mm + 1e-9
    v = np.interp(p1.dense_s_mm, np.sort(p1.s_mm), p1.speed_mps[np.argsort(p1.s_mm)])
    assert p1.duration_s == pytest.approx(float(np.trapz(1.0 / v, p1.dense_s_mm * 1e-3)), rel=2e-3)
    # 경로 점 = 단면 겨냥점의 보간 (단면 위치에서는 정확히)
    st = plan.stations[20]
    j = int(np.argmin(np.abs(p1.dense_s_mm - st.s_mm)))
    if abs(p1.dense_s_mm[j] - st.s_mm) < 1e-9:
        k = int(np.flatnonzero(p1.s_mm == st.s_mm)[0])
        assert np.linalg.norm(p1.dense_pos_m[j] - p1.aim_base_m[k]) < 1e-9
    cap = plan.passes[3]
    A = np.interp(cap.dense_s_mm, np.sort(cap.s_mm), cap.weave_amp_mm[np.argsort(cap.s_mm)])
    assert np.all(np.abs(cap.dense_lateral_mm) <= A + 1e-9) and np.all(np.diff(cap.dense_s_mm) >= 0)
    at_edge = np.abs(np.abs(cap.dense_lateral_mm) - A) < 1e-6
    v_cap = float(np.median(cap.speed_mps))
    cycles = (L_MM - 5.0) / proc.weave_pitch_mm
    assert at_edge.sum() == pytest.approx(cycles * 2 * proc.weave_dwell_s / proc.weave_sample_s, rel=0.15)   # 양 끝 머묾
    assert cap.dense_lateral_mm[-1] == 0.0 and cap.duration_s == pytest.approx((L_MM - 5.0) * 1e-3 / v_cap, rel=0.05)
    # 위빙 점은 캡 겨냥 높이 (지금 표면 = 마지막 충전 층 위) 에서 측면 w 로만 벗어난다
    st = plan.stations[40]
    j = int(np.argmin(np.abs(cap.dense_s_mm - st.s_mm)))
    off = cap.dense_pos_m[j] - st.origin
    jj = int(np.flatnonzero(cap.s_mm == st.s_mm)[0])
    assert abs(off @ st.n * 1e3 - float(cap.aim_xz_mm[jj, 1])) < 0.05
    for R in cap.dense_R[::500]:
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)


def test_save_summary_and_process_validation(tmp_path):
    f = synth_feats()
    plan = weld_plan_from_features(f, extent(f), SPEC, WeldProcess())
    files = plan.save(tmp_path)
    d = json.loads((tmp_path / "weld_plan.json").read_text(encoding="utf-8"))
    assert d["schema_version"] == 3 and len(d["passes"]) == 4 and len(d["stations"]) == 77
    assert len(d["topology"]["layers"]) == 3 and d["topology"]["n_cap"] == 1 and d["bead_model"]["model"] == "mass_balance"
    assert d["passes"][3]["kind"] == "cap" and len(d["passes"][3]["weave_amp_mm"]) == 3
    assert len(d["passes"][0]["sections"]["aim_base_m"]) == 77 and d["stations"][0]["levels_mm"][0] > 0
    sec0 = d["passes"][1]["sections"]                                          # the convex beads in the plan file (bead_geometry)
    assert len(sec0["bead_toe_left_xz_mm"]) == 77 and len(sec0["bead_crown_xz_mm"][0]) == 2 and set(sec0["bead_phase"]) == {"pool"}
    assert d["passes"][1]["bead_geometry"]["crown_ratio"][0] == pytest.approx(math.tan(math.radians(15.0)) / 2.0)
    assert "circular arc" in d["bead_geometry"]["rule"] and "toe angle 30 deg" in d["bead_geometry"]["crown_ratio"]
    assert d["bevel_plate_looking_S_to_E"] == "right"
    z = np.load(tmp_path / "weld_plan.npz")
    assert z["p4_pos_m"].shape[0] == z["p4_t_s"].shape[0] > 1000 and z["p1_R"].shape[1:] == (3, 3)
    lines = format_plan_summary(d)
    assert lines == plan.summary_lines() and lines[0].startswith("weld plan: 4 passes over 380.0 mm")
    assert any(" cap " in l_ for l_ in lines) and len(files) == 2
    with pytest.raises(ValueError, match="unknown keys"):
        WeldProcess.from_dict({"fill_fraction": [0.3]})
    with pytest.raises(ValueError, match="derived from the WPS"):               # 고정 층 비율은 이제 WPS 에서 유도된다
        WeldProcess.from_dict({"fill_fractions": [0.27, 0.57, 0.87]})
    with pytest.raises(ValueError, match="root, fill, cap"):
        WeldProcess(travel_angle_deg=(10.0, 10.0))
    with pytest.raises(ValueError, match="> 0"):
        WeldProcess(wfs_mpm=(6.0, 0.0, 4.5))
    assert WeldProcess.from_dict({"speed_mps": [0.004, 0.005, 0.006], "direction": "alternate"}).speed_mps == (0.004, 0.005, 0.006)
    plan.feasibility = [{"k": 1, "reachable": True, "min_limit_margin_rad": 1.2, "max_tcp_speed_mm_s": 5.0, "speed_over_limit": False},
                        {"k": 4, "reachable": True, "min_limit_margin_rad": 0.9, "max_tcp_speed_mm_s": 46.0, "speed_over_limit": True}]
    line = [l_ for l_ in plan.summary_lines() if "robot check" in l_]
    assert line and line[0].endswith("all 2 passes reachable, joint-limit margin >= 0.90 rad")      # 공정값 경고는 표에 없다 (json 에만)
    plan.feasibility[0].update(reachable=False, reason="no IK")
    assert "pass 1 NOT reachable (no IK)" in [l_ for l_ in plan.summary_lines() if "robot check" in l_][0]
    plan.feasibility[1].update(reachable=None, reason="no answer")      # 플랫폼이 답을 받지 못한 확인 = 하지 않음 (안 된다가 아니다)
    line = [l_ for l_ in plan.summary_lines() if "robot check" in l_][0]
    assert "pass 1 NOT reachable (no IK)" in line and "pass 4 NOT CHECKED (no answer)" in line and "pass 4 NOT reachable" not in line
    plan.feasibility[1]["pool_note"] = "OwnedPool lifetime rule partly off (test)"     # 확인의 프로세스 풀 기록도 요약에
    assert any(l_.endswith("robot check note: OwnedPool lifetime rule partly off (test)") for l_ in plan.summary_lines())
    png = plan.plot_sections(tmp_path / "s.png", n_sections=3)
    assert png is None or (tmp_path / "s.png").stat().st_size > 1000


# --------------------------------------------------------------------------------------
# WPS = KIRO 데이터셋 (2026-09-15/16 weld_testspecimen_logs; 사용자 승인 2026-10-12)
# --------------------------------------------------------------------------------------
# KIRO 로그의 패스별 실측 정상 상태 심 속도 [mm/s] 와 설정 WFS (fsv1_work/kiro_dataset/deposit_per_pass.csv), 질량 보존 단면적 (η 0.95)
KIRO_PASSES = [("root", 5.0106, 8.4, 30.02), ("fill", 7.0111, 10.8, 27.58), ("fill", 6.9962, 10.8, 27.64), ("cap", 5.9887, 10.2, 30.50)]
# k20_2_a (2026-10-12 실기, 우리 10T 시편) 의 기준 단면 — KIRO root + 충전 2 가 이 그루브를 채운다 (85.2 ≈ 85.5 mm²)
K20_REF = GrooveSpec(kind="single_bevel", thickness_mm=10.514106537212687, bevel_angle_deg=30.49794880937114, bevel_side="left",
                     root_gap_mm=5.199548388145472, root_face_mm=0.2804609679147437, hi_lo_mm=-0.6942210186257858)


def test_bead_model_reproduces_the_kiro_pass_areas():
    """비드 단면적 = η · (π d²/4) · WFS / v: KIRO 네 패스 (실측 심 속도) → 30.0 / 27.6 / 27.6 / 30.5 mm² (η 0.95), 설정 속도 (5 / 7 / 6 mm/s)
    에서는 기본 WPS 의 역할별 공칭 단면적 30.1 / 27.6 / 30.4 mm²."""
    from scan_core.bead_model import PassParams
    proc = WeldProcess()
    bm = proc.bead_model()
    assert proc.deposition_efficiency == 0.95 and proc.wire_diameter_mm == 1.2
    for role, v_mm_s, wfs, area in KIRO_PASSES:
        a = bm.area_mm2(PassParams(role, wfs, v_mm_s * 1e-3))
        assert a == pytest.approx(area, abs=0.01)
        assert a == pytest.approx(0.95 * math.pi * 1.2 ** 2 / 4 * wfs * 1000 / 60 / v_mm_s, rel=1e-12)
    nominal = [bm.area_mm2(proc.params(r)) for r in ("root", "fill", "cap")]
    assert nominal == pytest.approx([30.08, 27.63, 30.44], abs=0.01)
    # 역할별 설정값과 설정선 (WFS = 0.06 I − 3.0, U = 15 + 0.05 I) — 세 설정점에서 정확
    assert proc.current_a == (190.0, 230.0, 220.0) and proc.voltage_v == (24.5, 26.5, 26.0) and proc.wfs_mpm == (8.4, 10.8, 10.2)
    for i in range(3):
        assert proc.wfs_on_line(proc.current_a[i]) == pytest.approx(proc.wfs_mpm[i], abs=1e-9)
        assert proc.voltage_on_line(proc.current_a[i]) == pytest.approx(proc.voltage_v[i], abs=1e-9)
        assert proc.current_on_line(proc.wfs_mpm[i]) == pytest.approx(proc.current_a[i], abs=1e-9)
    assert proc.speed_mps == (0.005, 0.007, 0.006) and proc.source == "KIRO dataset 2026-09-15/16, weld_testspecimen_logs"
    assert (proc.lead_in_mm, proc.lead_out_mm, proc.arc_off_delay_ms) == ((1.0, 1.0, 0.0), (6.0, 6.0, 0.0), (500.0, 500.0, 0.0))
    assert (proc.weave_pattern, proc.weave_axis, proc.weave_amplitude_mm, proc.weave_pitch_mm, proc.weave_dwell_s,
            proc.weave_samples_per_cycle) == ("sine", "tool_y", 4.0, 7.8, 0.2, 12)


def test_kiro_wps_on_our_measured_groove_gives_root_two_fills_and_cap():
    """KIRO 설정 그대로 (속도 창 [1, 1]): 비드 = KIRO 의 단면적, 패스 수는 잰 그루브와 용접 설계의 캡 보강 (기본 1.87 ± 0.19 mm, KIRO 유도
    - 볼록 비드 기하 (토우각 30 deg) 로 KIRO 의 네 패스를 이 단면에 놓은 결과): 공차 안인 층 수, 없으면 가장 가까운 층 수 + 표시. k20_2_a
    기준 단면 (85.5 mm²) → root + 충전 2 + 위빙 캡 1, 보강 1.87 mm (공차 안). 설계 단면 (78.9 mm²) 에서도 4 패스 — 셋째 충전이 그루브 위로
    넘치고 (그 위를 캡이 덮는다) 보강 1.91 mm. 홈 갭 4 mm (69 mm²) 면 가장 가까운 배치 root + 충전 2 + 캡 (2.15 mm, 공차 밖 표시). 설계
    입력 (job weld_design) 이 고르기를 바꾼다."""
    tp = plan_topology(K20_REF, WeldProcess())
    assert [(ly.role, ly.n_beads) for ly in tp.layers] == [("root", 1), ("fill", 1), ("fill", 1)] and tp.n_cap == 1 and tp.feasible
    assert tp.groove_area_ref_mm2 == pytest.approx(85.51, abs=0.01) and tp.deposit_scale == 1.0 and tp.cap_scale == 1.0
    assert [ly.bead_area_ref_mm2 for ly in tp.layers] == pytest.approx([30.08, 27.63, 27.63], abs=0.01)
    assert tp.cap_area_ref_mm2 == pytest.approx(30.44, abs=0.01) and tp.reinforcement_ref_mm == pytest.approx(1.87, abs=0.01)
    # the reference section's convex beads (toe angle 30 deg): level toe chords at 4.03 / 7.04 / 9.56 mm, crowns 5.03 / 8.27 / 10.99 mm
    # (fill 2's crown 0.5 mm over the plate top: the cap covers it)
    assert [ly.z_top_ref_mm for ly in tp.layers] == pytest.approx([4.03, 7.04, 9.56], abs=0.01)
    assert [ly.z_crown_ref_mm for ly in tp.layers] == pytest.approx([5.03, 8.27, 10.99], abs=0.01)
    assert [b.phase for b in tp.beads_ref] == ["pool", "pool", "pool", "over"]
    design = plan_topology(GrooveSpec(bevel_side="left"), WeldProcess())
    assert [ly.role for ly in design.layers] == ["root", "fill", "fill"] and design.n_cap == 1 and design.feasible
    assert design.layers[-1].frac_top > 1.0 and design.target_met                                    # the third fill overflows
    assert design.reinforcement_ref_mm == pytest.approx(1.91, abs=0.01)
    small = plan_topology(GrooveSpec(bevel_side="left", root_gap_mm=4.0), WeldProcess())
    assert [ly.role for ly in small.layers] == ["root", "fill", "fill"] and small.reinforcement_ref_mm == pytest.approx(2.15, abs=0.01)
    assert not small.target_met and any("no layout gives the design cap reinforcement 1.87 +- 0.19 mm" in n_ for n_ in small.notes)
    assert tp.target_met and tp.design["cap_reinforcement_mm"] == {"nominal": 1.87, "tol": 0.19}
    # 설계 입력이 층 수를 정한다: 낮은 보강 (1.0 ± 0.5 mm) 을 원하면 설계 단면은 충전 1 개 (보강 0.96 mm, 공차 안)
    from scan_core.weld_plan import WeldDesign
    low = plan_topology(GrooveSpec(bevel_side="left"), WeldProcess(), design=WeldDesign(cap_reinforcement_mm=1.0, cap_reinforcement_tol_mm=0.5))
    assert [ly.role for ly in low.layers] == ["root", "fill"] and low.target_met and low.reinforcement_ref_mm == pytest.approx(0.96, abs=0.01)


def test_fixed_kiro_settings_layer_heights_follow_the_deposits():
    """설정이 고정이면 (KIRO, 속도 창 [1, 1]) 단면마다 용착 = 그 설정의 비드 단면적 (속도 = 설정), 층 높이 = 그 용착의 볼록 비드를 아래부터
    놓은 결과 (단면마다 그루브가 다르면 심 방향으로 변한다), 캡 보강 = 놓인 캡의 결과 - the same beads as laying the passes on the section
    by hand (one source). 필요 속도·단면적은 정보."""
    import scan_core.weld_plan as W
    from scan_core.bead_model import crown_ratio_of
    proc = WeldProcess()
    bm = proc.bead_model()
    f = synth_feats()
    plan = weld_plan_from_features(f, extent(f), SPEC, proc)
    assert [p.role for p in plan.passes] == ["root", "fill", "fill", "cap"]
    nominal = {r: bm.area_mm2(proc.params(r)) for r in ("root", "fill", "cap")}
    for p in plan.passes:
        assert np.allclose(p.area_mm2, nominal[p.role]) and np.allclose(p.speed_mps, proc.params(p.role).speed_mps)
        assert p.speed_const_mps == pytest.approx(proc.params(p.role).speed_mps)
        assert np.ptp(p.area_required_mm2) > 0.0                               # 필요 단면적은 그루브를 따라 변한다 (정보)
    span = W._section_span_mm(proc, bm, "1G")
    for i in (0, 33, 76):
        st = plan.stations[i]
        sec = W._Section(st.spec, span)
        for k, (role, occ) in enumerate((("root", 0), ("fill", 0), ("fill", 1))):
            pk = proc.params(role, occurrence=occ)
            b = sec.fill_layer(role, occ, 1, [nominal[role]], [crown_ratio_of(bm, pk)])[0][0]
            q = plan.passes[k]
            j = int(np.flatnonzero(q.s_mm == st.s_mm)[0])
            assert np.allclose([q.beads[j].toe_l, q.beads[j].toe_r], [b.toe_l, b.toe_r], atol=1e-9)
            assert plan.levels_mm[i][k] == pytest.approx(b.toe_l[1], abs=1e-9)
        cap = plan.passes[3]
        j = int(np.flatnonzero(cap.s_mm == st.s_mm)[0])
        pc = proc.params("cap", weave_amp_mm=float(cap.weave_amp_mm[j]))
        h = sec.cap_passes(1, [nominal["cap"]], [crown_ratio_of(bm, pc)], proc.cap_overlap_mm, float(cap.weave_amp_mm[j]))[0][4]
        assert cap.reinforcement_mm[j] == pytest.approx(h, abs=1e-9) and 1.5 < h < 3.5
    lv = np.asarray(plan.levels_mm)
    assert np.ptp(lv[:, 0]) > 0.01 and np.all(np.diff(lv, axis=1) > 0.0)       # 층 높이는 심 방향으로 변하고 층마다 오른다
    assert any("speed window fixed" in l_ for l_ in plan.summary_lines()) and any("cap reinforcement over the plate top" in l_
                                                                                for l_ in plan.summary_lines())


def test_job_template_weld_plan_block_is_the_default_wps():
    """1G 작업 템플릿의 weld_plan: 블록 = WeldProcess 기본값 (KIRO WPS, 같은 출처), weld_design: 블록 = WeldDesign 기본값 (캡 보강 공칭·
    공차, KIRO 유도) — 둘이 어긋나지 않게 (코드의 기본값은 템플릿의 값과 같고 출처를 지닌다)."""
    from scan_core.weld_plan import WeldDesign, weld_design_from_doc
    import yaml
    from pathlib import Path
    jobs = Path(__file__).resolve().parents[2] / "scan_l0" / "scan_l0" / "jobs"
    for name in ("groove_10T_1G.yaml", "example_1G.yaml"):
        doc = yaml.safe_load((jobs / name).read_text(encoding="utf-8"))
        block = dict(doc["weld_plan"])
        proc = WeldProcess.from_dict(block)
        ref = WeldProcess().to_dict()
        for k, v in block.items():
            assert proc.to_dict()[k] == ref[k], (name, k)
        assert proc.to_dict() == ref, name
        wd = weld_design_from_doc(doc, proc)
        assert doc["weld_design"]["cap_reinforcement_mm"] == {"nominal": 1.87, "tol": 0.19}
        assert block["bead_toe_angle_deg"] == 30.0 and "bead_aspect_ratio" not in block
        assert "aim_lateral_tcp_mm" not in block and block["work_angle_deg"] == [9.0, [-20.0, 7.0], 7.0]
        assert (wd.cap_reinforcement_mm, wd.cap_reinforcement_tol_mm, wd.source) == (
            WeldDesign().cap_reinforcement_mm, WeldDesign().cap_reinforcement_tol_mm, WeldDesign().source)
        assert doc["weld_design"]["end_inset_mm"] == {"start": 0.0, "goal": 0.0} and wd.end_insets_mm == (0.0, 0.0)   # 잰 끝 그대로
    # 3G 템플릿은 KIRO 데이터셋 전의 가정값을 명시해 그대로 (KIRO 데이터셋은 1G 만; 보강 목표 = 공정의 옛 키 1.5 mm, 공차 없음)
    for name in ("groove_10T_3G.yaml", "example_3G.yaml"):
        d3 = yaml.safe_load((jobs / name).read_text(encoding="utf-8"))
        p3 = WeldProcess.from_dict(d3["weld_plan"])
        wd3 = weld_design_from_doc(d3, p3)
        assert (wd3.cap_reinforcement_mm, wd3.cap_reinforcement_tol_mm) == (1.5, None), name
        assert (p3.wfs_mpm, p3.speed_mps, p3.travel_angle_deg, p3.work_angle_deg, p3.cap_reinforcement_mm, p3.weave_pitch_mm,
                p3.weave_dwell_s, p3.source, p3.speed_limits, p3.weave_amplitude_mm) == (
            (6.0, 6.6, 4.5), (0.005, 0.005, 0.0035), (10.0, 10.0, 10.0), (0.0, 0.0, 0.0), 1.5, 2.5, 0.1, "assumed (before the KIRO dataset)",
            (0.6, 1.6), None), name


def test_bead_shape_is_one_toe_angle_and_the_older_aspect_ratio_key_reads_as_it():
    """The convex bead's one shape parameter is the toe angle (user 2026-10-13 night; WPS bead_toe_angle_deg, default 30 deg - a
    typical literature value, to be calibrated): kappa = tan(toe / 2) / 2 for the free bead's width and every bead's dome. A job block
    with the older key bead_aspect_ratio still reads (kappa -> toe = 2 atan(2 kappa), said in the provenance); both keys -> refused;
    the key travels through to_dict / from_dict like the other WPS keys."""
    p = WeldProcess()
    assert p.bead_toe_angle_deg == 30.0 and p.bead_aspect_ratio == pytest.approx(math.tan(math.radians(15.0)) / 2.0)
    assert p.bead_model().toe_angle_deg == 30.0 and p.bead_model().aspect_ratio == pytest.approx(p.bead_aspect_ratio)
    assert "literature" in p.provenance["bead_toe_angle_deg"] and "calibrated" in p.provenance["bead_toe_angle_deg"]
    old = WeldProcess.from_dict({"bead_aspect_ratio": 0.3})
    assert old.bead_toe_angle_deg == pytest.approx(math.degrees(2.0 * math.atan(0.6))) and old.bead_aspect_ratio == pytest.approx(0.3)
    assert "bead_aspect_ratio 0.3 (the older key)" in old.provenance["bead_toe_angle_deg"]
    with pytest.raises(ValueError, match="both given"):
        WeldProcess.from_dict({"bead_aspect_ratio": 0.3, "bead_toe_angle_deg": 30.0})
    q = WeldProcess.from_dict({"bead_toe_angle_deg": 40.0})
    assert q.to_dict()["bead_toe_angle_deg"] == 40.0 and "bead_aspect_ratio" not in q.to_dict()
    assert WeldProcess.from_dict(q.to_dict()).bead_toe_angle_deg == 40.0 and q.provenance["bead_toe_angle_deg"] == "the job's value (no provenance given)"
    with pytest.raises(ValueError, match="toe angle"):
        WeldProcess(bead_toe_angle_deg=0.0)
    # a rounder bead (a larger toe angle) is narrower: fewer beads span a wide layer
    bm30, bm62 = WeldProcess().bead_model(), old.bead_model()
    prm = WeldProcess().params("fill")
    assert bm30.width_mm(prm) > bm62.width_mm(prm) and bm30.area_mm2(prm) == bm62.area_mm2(prm)


def test_old_job_process_blocks_still_read_and_per_layer_angles():
    """옛 작업의 weld_plan 블록 (역할마다 수 하나, 예전 키만) 은 그대로 읽힌다 (빠진 키 = KIRO WPS 기본). 각은 역할마다 수 또는 그 역할의
    층 순서별 목록 (마지막 값이 이어진다) — to_dict / from_dict 왕복."""
    old = {"wfs_mpm": [6.0, 6.6, 4.5], "speed_mps": [0.005, 0.005, 0.0035], "travel_angle_deg": [10.0, 10.0, 10.0],
           "work_angle_deg": [0.0, 0.0, 0.0], "weave_pitch_mm": 2.5, "weave_dwell_s": 0.1, "cap_reinforcement_mm": 1.5,
           "source": "assumed (before the KIRO dataset)"}
    p = WeldProcess.from_dict(old)
    assert p.role_angle("work_angle_deg", "fill", 5) == 0.0 and p.wfs_mpm == (6.0, 6.6, 4.5) and p.current_a == (190.0, 230.0, 220.0)
    from scan_core.weld_plan import WeldDesign, weld_design_from_doc
    assert weld_design_from_doc({"weld_plan": old}, p).cap_reinforcement_mm == 1.5                  # 옛 공정 키 = 설계 공칭
    assert WeldDesign.from_doc({"cap_reinforcement_mm": 2.0}).cap_reinforcement_mm == 2.0           # 수 하나도 읽는다 (공차 = 기본)
    with pytest.raises(ValueError, match="unknown keys"):
        WeldDesign.from_doc({"cap_reinforcement_mm": {"nominal": 3.0, "tolerance": 1.0}})
    # 끝 inset (사용자 2026-10-12): 설계 입력 weld_design.end_inset_mm {start, goal}; 옛 공정 키 start_inset_mm / end_inset_mm 는 설계가
    # 주지 않을 때만 (둘 다 주면 거부), 아무도 주지 않으면 0 (잰 끝 그대로)
    d = weld_design_from_doc({"weld_design": {"end_inset_mm": {"start": 10.0, "goal": 6.0}}}, WeldProcess())
    assert d.end_insets_mm == (10.0, 6.0) and d.to_dict()["end_inset_mm"]["source"] == "job weld_design.end_inset_mm"
    assert d.cap_reinforcement_mm == 1.87                                                           # 빠진 값 = 기본
    assert weld_design_from_doc({}, WeldProcess()).end_insets_mm == (0.0, 0.0)
    legacy = WeldProcess(start_inset_mm=2.0, end_inset_mm=3.0)
    assert weld_design_from_doc({}, legacy).end_insets_mm == (2.0, 3.0)
    assert WeldDesign.resolve(weld_design_from_doc({}, legacy), legacy).end_insets_mm == (2.0, 3.0)  # 확정한 설계를 다시 거쳐도 같다
    with pytest.raises(ValueError, match="given twice"):
        weld_design_from_doc({"weld_design": {"end_inset_mm": {"start": 10.0, "goal": 6.0}}}, legacy)
    for bad, msg in (({"start": -1.0, "goal": 0.0}, ">= 0"), ({"start": 1.0, "end": 2.0}, "unknown keys"), (5.0, "must be")):
        with pytest.raises(ValueError, match=msg):
            WeldDesign.from_doc({"end_inset_mm": bad})
    q = WeldProcess()
    assert [q.role_angle("work_angle_deg", "fill", i) for i in range(4)] == [-20.0, 7.0, 7.0, 7.0]
    assert q.role_angle("work_angle_deg", "root") == 9.0 and q.role_angle("work_angle_deg", "cap", 3) == 7.0
    assert [q.role_angle("travel_angle_deg", "fill", i) for i in range(3)] == [-41.0, -44.0, -44.0]
    d = q.to_dict()
    assert d["work_angle_deg"] == [9.0, [-20.0, 7.0], 7.0] and d["travel_angle_deg"] == [-42.0, [-41.0, -44.0], -44.0]
    assert WeldProcess.from_dict(json.loads(json.dumps(d))).to_dict() == d
    assert WeldProcess(work_angle_deg=(0.0, [5.0], 1.0)).work_angle_deg == (0.0, 5.0, 1.0)
    with pytest.raises(ValueError, match="empty per-layer list"):
        WeldProcess(work_angle_deg=(0.0, [], 1.0))
    with pytest.raises(ValueError, match="slope > 0"):
        WeldProcess(wfs_from_current=(0.0, 1.0))
    with pytest.raises(ValueError, match="lead_in_mm"):
        WeldProcess(lead_in_mm=(1.0, -1.0, 0.0))


@pytest.mark.parametrize("s_lower", [True, False], ids=["S_lower", "S_upper"])
def test_vertical_plate_3g_uphill_torch_and_weave(s_lower):
    """3G (세운 판, 2026-10-04): 판 법선 수평 (−y, 로봇 쪽), 용접선 수직. 'uphill' 이면 S 가 아래면 S→E, 위면 E→S (늘 위로);
    토치 축 = −n 에서 진행 (위) 쪽으로 10° (작업각 0 의 WPS — 진행각의 기하만 본다), 단면 가로축 (위빙) 은 수평, 층은 판 법선 방향으로 쌓인다."""
    Nv = np.array([0.0, -1.0, 0.0])
    Uv = np.array([0.0, 0.0, 1.0 if s_lower else -1.0])
    C0v = np.array([0.50, 0.70, 0.30 if s_lower else 0.68])
    f = synth_feats(U=Uv, N=Nv, C0=C0v)
    proc = WeldProcess(direction="uphill", travel_angle_deg=(10.0, 10.0, 10.0), work_angle_deg=(0.0, 0.0, 0.0))
    plan = weld_plan_from_features(f, extent(f), SPEC, proc, gravity_axis_base=(0.0, 0.0, -1.0), position="3G")
    want = "S->E" if s_lower else "E->S"
    assert [p.kind for p in plan.passes] == ["fill", "fill", "fill", "cap"] and all(p.direction == want for p in plan.passes)
    up = np.array([0.0, 0.0, 1.0])
    for p in plan.passes:
        for R in p.R_base_tcp[::10]:
            assert math.degrees(math.acos(float(np.clip(R[:, 2] @ -Nv, -1, 1)))) == pytest.approx(10.0, abs=1e-6)
            assert float(R[:, 2] @ up) == pytest.approx(math.sin(math.radians(10.0)), abs=1e-6)   # 위로 민다
            assert float(R[:, 0] @ up) > 0.98                                                  # x = 진행 방향 = 위
        assert np.all(np.diff(p.aim_base_m[:, 2]) > 0.0)                                       # 아래 → 위
    st = stations_from_features(f, SPEC, frame(f))
    for x in st[::10]:
        lat = x.to_base(1.0, 0.0) - x.to_base(0.0, 0.0)                    # 단면 x (위빙·벽 사이) = 수평
        dep = x.to_base(0.0, 1.0) - x.to_base(0.0, 0.0)                    # 단면 z (층) = 판 법선 (로봇 쪽)
        assert abs(lat[2]) < 1e-9 and np.linalg.norm(lat) == pytest.approx(1e-3)
        assert np.allclose(dep / np.linalg.norm(dep), Nv, atol=1e-9)
    cap = plan.passes[3]
    assert np.all(np.abs(np.diff(cap.dense_pos_m[:, 2])) < 0.01) and np.ptp(cap.dense_pos_m[:, 0]) > 0.005   # 위빙은 가로 (x)


# --------------------------------------------------------------------------------------
# S→E 계약 (2026-10-09, 실기 1G 배치 분석 3.4: 마지막 전체 패스가 E→S 인 실행의 계획이 'S->E' 표기로 E 에서 S 로 갔다)
# --------------------------------------------------------------------------------------
POSES = {"1G": dict(U=U, N=N, C0=C0, direction="S->E", g=None),
         "3G": dict(U=np.array([0.0, 0.0, 1.0]), N=np.array([0.0, -1.0, 0.0]), C0=np.array([0.50, 0.70, 0.30]), direction="uphill",
                    g=(0.0, 0.0, -1.0))}


def as_pass(f, travel_E_to_S: bool):
    """특징의 정련 직선 키를 한 패스의 것으로: c = 시편 가운데, u = 진행 방향 (E→S 패스면 E→S), s_line·s_a/s_b 도 그 좌표 — 온라인 병합이
    기준 패스의 fit 에서 넘기던 모양. 계획은 이 키를 쓰지 않아야 한다."""
    S, E = ends(f)
    u = (E - S) / np.linalg.norm(E - S) * (-1.0 if travel_E_to_S else 1.0)
    c = 0.5 * (S + E)
    g = dict(f, u_base=u, line_c_base_m=c)
    g["s_line_m"] = (np.asarray(f["root_polyline_base_m"]) - c) @ u
    g["s_a_m"], g["s_b_m"] = np.array(-0.5 * L_MM * 1e-3), np.array(0.5 * L_MM * 1e-3)
    return g


@pytest.mark.parametrize("pos", ["1G", "3G"])
@pytest.mark.parametrize("travel", ["S->E", "E->S"])
def test_plan_follows_the_observed_ends_whatever_the_pass_direction(pos, travel):
    """축과 단면 위치는 관측 끝 S, E (base 점) 에서만 나온다: 기준 패스가 E→S (정련 직선 u 가 E→S, c 가 시편 가운데) 여도 계획은 S→E 패스의
    것과 같다 — 첫 단면 = S, 마지막 = E, 단면 축 S→E, 같은 베벨 쪽, 패스마다 표기한 방향의 끝에서 시작. 1G·3G (위보기 uphill) 모두."""
    cfg = POSES[pos]
    f = synth_feats(U=cfg["U"], N=cfg["N"], C0=cfg["C0"])
    kw = dict(position=pos, gravity_axis_base=cfg["g"])
    ref = weld_plan_from_features(f, extent(f), SPEC, WeldProcess(direction=cfg["direction"]), **kw)
    plan = weld_plan_from_features(as_pass(f, travel == "E->S"), extent(f), SPEC, WeldProcess(direction=cfg["direction"]), **kw)
    S, E = ends(f)
    u = (E - S) / np.linalg.norm(E - S)
    O = np.array([st.origin for st in plan.stations])
    assert float((O[0] - S) @ u) == pytest.approx(0.0, abs=1e-9) and float((O[-1] - E) @ u) == pytest.approx(0.0, abs=1e-9)
    assert all(float(st.u @ (E - S)) > 0.0 for st in plan.stations) and plan.length_mm == pytest.approx(L_MM)
    assert plan.bevel_side == ref.bevel_side == "left" and plan.bevel_view == "right"
    np.testing.assert_allclose(O, [st.origin for st in ref.stations], rtol=0, atol=1e-12)
    for p, q in zip(plan.passes, ref.passes):
        assert p.direction == q.direction
        np.testing.assert_allclose(p.aim_base_m, q.aim_base_m, rtol=0, atol=1e-12)
        np.testing.assert_allclose(p.dense_pos_m, q.dense_pos_m, rtol=0, atol=1e-12)
        start = S if p.direction == "S->E" else E
        assert abs(float((p.aim_base_m[0] - start) @ u)) < 1e-9
    if pos == "3G":
        assert all(p.direction == "S->E" for p in plan.passes)                 # S 가 아래 (C0 z 0.30 → 위): 위보기는 S→E
    ch = plan.ends["checked"]
    assert abs(ch["first_section_from_S_mm"]) < 1e-6 and abs(ch["last_section_from_E_mm"]) < 1e-6
    assert ch["tol_S_mm"] == pytest.approx(0.3) and ch["k"] == 3.0                # k·σ_끝 (판정 배수 기본 3, σ 0.1 mm)
    assert plan.ends["S_base_m"] == pytest.approx(S.tolist()) and plan.ends["E_base_m"] == pytest.approx(E.tolist())


def test_contract_refuses_a_mirrored_or_mislabelled_plan():
    """계약 검사는 계획과 독립인 관측 끝으로 잰다: 단면이 E 에서 S 로 가는 계획 (예전 결함), 'S->E' 표기에 E→S 단면인 패스는 위반. 끝 점이
    없으면 (다른 직선 기준의 스칼라 's' 만), S = E, 끝 σ 를 모르면 계획을 내지 않는다 (PlanContractError). σ 가 없으면 착점 간격/√12."""
    f = synth_feats()
    S, E = ends(f)
    fr = plan_frame(f, S, E, sigma_S_m=1e-4, sigma_E_m=1e-4)
    mirrored = plan_weld(stations_from_features(f, SPEC, plan_frame(f, E, S)), WeldProcess())
    viol, chk = check_plan_contract(mirrored, fr, k=3.0)
    assert any(v.startswith("first section +380.00 mm from S") for v in viol), viol
    assert any("section axes point E->S" in v for v in viol)
    plan = weld_plan_from_features(f, extent(f), SPEC, WeldProcess())
    assert check_plan_contract(plan, fr)[0] == []
    p = plan.passes[0]
    p.aim_base_m, p.dense_pos_m = p.aim_base_m[::-1].copy(), p.dense_pos_m[::-1].copy()
    viol1, _ = check_plan_contract(plan, fr)
    assert any(v.startswith("pass 1 (S->E) first section +380.00 mm from S") for v in viol1), viol1
    with pytest.raises(PlanContractError, match="S end has no base point"):
        weld_plan_from_features(f, {"S": {"s": -0.19, "observed": True}, "E": {"s": 0.19, "observed": True}}, SPEC)
    ext = extent(f)
    ext["E"]["point"] = list(ext["S"]["point"])
    with pytest.raises(PlanContractError, match="do not span the seam"):
        weld_plan_from_features(f, ext, SPEC)
    ext = extent(f, sigma_m=float("nan"))
    del ext["spacing_mm"]
    with pytest.raises(PlanContractError, match="end uncertainty unknown"):
        weld_plan_from_features(f, ext, SPEC)
    pl = weld_plan_from_features(f, extent(f, sigma_m=float("nan")), SPEC, k=2.0)
    assert pl.ends["checked"]["tol_S_mm"] == pytest.approx(2.0 * 0.3 / math.sqrt(12.0))


def test_aim_is_the_wire_end_on_the_current_surface():
    """겨냥 (사용자 2026-10-13): 와이어 끝은 지금 표면에 닿는다 — the specimen before the pass: the plates, the backing and the
    earlier passes' convex beads (the weld scene shows the same solids); lateral = KIRO's practice, said per pass in the plan and the
    summary; the removed keys (aim_height_frac, the 09-22 table aim_lateral_tcp_mm / aim_lateral_axis) are refused with the reason."""
    from scan_core.weld_scene import pass_solids, surface_mm
    from scan_core.bead_geometry import TANGENT_POINTS
    f = synth_feats()
    plan = weld_plan_from_features(f, extent(f), SPEC, WeldProcess())
    root = plan.passes[0]
    assert root.role == "root" and np.allclose(root.aim_xz_mm[:, 1], [s_.spec.root_face_mm for s_ in plan.stations])
    for p in plan.passes:                                    # the aim point lies on the specimen's surface before the pass
        sol = pass_solids(plan, p)
        for i, st in list(enumerate(plan.stations))[::7]:
            j = int(np.flatnonzero(p.s_mm == st.s_mm)[0])
            assert np.linalg.norm(p.aim_base_m[j] - st.to_base(*p.aim_xz_mm[j])) < 1e-12
            x, z = p.aim_xz_mm[j]
            if p.k > 2:                                      # on an earlier bead's top (fill 1 aims at a toe on the square wall - a
                #                                              corner, where the scene's upper surface is the plate top): the scene's
                #                                              tangent polygon is within its bound
                top = surface_mm({so.name: so.poly[i] for so in sol}, x)
                assert top == pytest.approx(z, abs=0.05) and top >= z - 1e-9
            if p.k > 2:
                prev = plan.passes[p.k - 2].beads[int(np.flatnonzero(plan.passes[p.k - 2].s_mm == st.s_mm)[0])]
                assert z == pytest.approx(float(prev.top(x)), abs=1e-3)
    assert TANGENT_POINTS >= 8
    d = plan.to_dict()
    assert "KIRO's practice" in d["aim_rule"]["lateral"]
    assert "the corner of the root face and the bevel face" in d["aim_rule"]["lateral"] and "groove top" in d["aim_rule"]["lateral"]
    assert "geometric rule" in d["aim_rule"]["uncovered"] and "current surface" in d["aim_rule"]["vertical"]
    assert [r_["lateral"] for r_ in d["aim_rule"]["passes"]] == ["kiro_practice"] * 4
    assert d["aim_rule"]["tip_offset_mm"] == 20.0 and "no weld torch model: the stick-out alone" in d["aim_rule"]["tip_offset_source"]
    assert [p_["aim_lateral_rule"] for p_ in d["passes"]] == ["kiro_practice"] * 4
    assert any("pass 2 fill 1: the corner where the root bead meets the square face" in ln for ln in plan.summary_lines())
    with pytest.raises(ValueError, match="aim_height_frac: removed 2026-10-13"):
        WeldProcess.from_dict({"aim_height_frac": 0.5})
    with pytest.raises(ValueError, match="aim_lateral_tcp_mm: removed 2026-10-13 night"):
        WeldProcess.from_dict({"aim_lateral_tcp_mm": [[0.0], [-5.34, -1.63], [0.04]]})


def test_kiro_0922_offsets_are_a_verification_with_the_bevel_on_the_left():
    """KIRO's 09-22 per-pass offsets are no rule any more (user + KIRO 2026-10-13 night) - a verification only: re-read with the bevel
    on the LEFT of KIRO's travel they sit (relative to pass 1, at L = TCP -> wire end) fill 1 1.98 toward the square face, fill 2 1.64
    toward the bevel, the cap 0.13 toward the square face (stick-out 20 + recess 1.2); the plan reports its rule beside them."""
    from scan_core.weld_plan import KIRO_0922, kiro_0922_toward_bevel_mm
    assert kiro_0922_toward_bevel_mm("root", 21.2) == 0.0 and kiro_0922_toward_bevel_mm("fill 3", 21.2) is None
    assert kiro_0922_toward_bevel_mm("fill 1", 21.2) == pytest.approx(-1.98, abs=0.01)
    assert kiro_0922_toward_bevel_mm("fill 2", 21.2) == pytest.approx(1.64, abs=0.01)
    assert kiro_0922_toward_bevel_mm("cap", 21.2) == pytest.approx(-0.13, abs=0.01)
    assert kiro_0922_toward_bevel_mm("fill 1", 16.2) == pytest.approx(-0.25, abs=0.01)                 # stick-out 15
    assert KIRO_0922["rise_mm"]["fill 2"] == (5.2, 6.5) and KIRO_0922["rise_mm"]["cap"] == (13.1, 13.9)
    f = synth_feats()
    plan = weld_plan_from_features(f, extent(f), SPEC, WeldProcess())
    ck = plan.aim["check_0922"]
    assert ck["applies"] and [c["pass"] for c in ck["passes"]] == ["root", "fill 1", "fill 2", "cap"]
    st0 = plan.stations[0]
    g = 0.5 * st0.spec.root_gap_mm
    c1 = ck["passes"][1]
    # the rule puts fill 1 on the square wall, the root at the bevel corner: one root gap toward the square face
    assert c1["rule_toward_bevel_from_root_mm"] == pytest.approx(-2.0 * float(np.median([0.5 * s_.spec.root_gap_mm for s_ in plan.stations])),
                                                                 abs=0.01)
    assert c1["kiro_0922_toward_bevel_from_root_mm"] == pytest.approx(kiro_0922_toward_bevel_mm("fill 1", 20.0))
    assert c1["lateral_difference_mm"] == pytest.approx(c1["rule_toward_bevel_from_root_mm"] - c1["kiro_0922_toward_bevel_from_root_mm"])
    assert g > 0.0 and all(c["rise_outside_0922_mm"] >= 0.0 for c in ck["passes"])
    assert any(ln.startswith("  aim vs KIRO 09-22 (verification") for ln in plan.summary_lines())


def _mirror_bevel(f):
    """The same synthetic groove with the bevel on the other plate: the top centre shifted to +w instead of -w."""
    g = dict(f)
    N = np.asarray(f["n_base"], dtype=float)
    u = np.asarray(f["u_base"], dtype=float)
    w = np.cross(N, u)
    run = (np.asarray(f["top_opening_mm"]) - np.asarray(f["root_gap_mm"])) * 1e-3
    g["top_polyline_base_m"] = np.asarray(f["top_polyline_base_m"]) + np.outer(run, w)
    g["bevel_side"] = np.array(["right"] * len(run))
    return g


@pytest.mark.parametrize("bevel", ["left", "right"])
def test_kiro_aim_rule_and_work_angles_mirror_with_the_bevel_side(bevel):
    """KIRO's practice from the measured section, whichever plate is bevelled (the bevel side resolved from the scan: the side the top
    opening's centre is shifted to): root = the root-face / bevel-face corner (bevel side), fill 1 = the root bead's toe on the square
    wall (the other side), fill 2 = midway between fill 1's toes, cap = midway between the groove top edges. Work angles (+ = the torch
    axis toward the work tilts toward the bevel): root +9 and fill 2 / cap +7 lean the axis toward the bevel, fill 1 -20 toward the
    square face - in both mirror images."""
    from scan_core.bead_geometry import groove_edges_mm
    from scan_core.weld_torch import WeldTorch
    from test_weld_torch import NOZZLE_TEST_MM, TORCH
    f = synth_feats() if bevel == "left" else _mirror_bevel(synth_feats())
    st = stations_from_features(f, GrooveSpec(bevel_side=bevel), frame(f))
    torch = WeldTorch.from_dict(TORCH).with_nozzle_end(NOZZLE_TEST_MM, "test")
    plan = plan_weld(st, WeldProcess(), spec_design=GrooveSpec(bevel_side=bevel), torch=torch)
    assert plan.bevel_side == bevel and plan.bevel_view == {"left": "right", "right": "left"}[bevel]
    shift = float(np.median((np.asarray(f["top_polyline_base_m"]) - np.asarray(f["root_polyline_base_m"])) @ st[0].w))
    toward = math.copysign(1.0, shift)                               # the bevel plate's side in section x, from the geometry itself
    assert plan.aim["toward_bevel_x"] == toward and plan.aim["tip_offset_mm"] == pytest.approx(NOZZLE_TEST_MM + 20.0)
    assert [p.aim_lateral_rule for p in plan.passes] == ["kiro_practice"] * 4
    root, fill1, fill2, cap = plan.passes
    for i, s_ in enumerate(plan.stations):
        j = [int(np.flatnonzero(p.s_mm == s_.s_mm)[0]) for p in plan.passes]
        g, rf = 0.5 * s_.spec.root_gap_mm, s_.spec.root_face_mm
        assert np.allclose(root.aim_xz_mm[j[0]], (toward * g, rf))                                  # the bevel-side corner
        rb = root.beads[j[0]]
        sq = rb.toe_r if toward < 0 else rb.toe_l
        assert sq[0] == pytest.approx(-toward * g) and np.allclose(fill1.aim_xz_mm[j[1]], sq)        # the square wall at the root toe
        f1 = fill1.beads[j[1]]
        assert fill2.aim_xz_mm[j[2], 0] == pytest.approx(0.5 * (f1.toe_l[0] + f1.toe_r[0]))
        xeL, _tl, xeR, _tr = groove_edges_mm(s_.spec)
        assert cap.aim_xz_mm[j[3], 0] == pytest.approx(0.5 * (xeL + xeR))
        # the bevel widens the groove upward: the cap's centre lies further toward the bevel than fill 2's
        assert toward * (cap.aim_xz_mm[j[3], 0] - fill2.aim_xz_mm[j[2], 0]) > 0.0
        for p, jj, sign in zip(plan.passes, j, (1.0, -1.0, 1.0, 1.0)):
            a = p.R_base_tcp[jj][:, 2]
            assert sign * float(a @ (toward * s_.w)) > 0.0, (p.k, bevel)                              # the axis tilt's side
    assert [p.work_angle_deg for p in plan.passes] == [9.0, -20.0, 7.0, 7.0]


def test_uncovered_layers_keep_the_geometric_rule():
    """Layers outside KIRO's 4-pass practice keep the geometric rule, labelled per pass with why: on a wide single bevel the root and
    fill 1 are covered, then a multi-bead layer and everything after it (bands), several cap passes; every pass of a V groove (no single
    bevel). A single-bead layer: the groove's lateral centre at its current surface; a multi-bead layer: its equal-area bands left to
    right inside the walls."""
    from test_groove_families import FAMILIES, _stations
    for name in ("bevel45_25T_g6_R", "V30_20T_g3_rf2"):
        sp = FAMILIES[name]
        plan = plan_weld(_stations(sp), WeldProcess())
        rows = {r_["k"]: r_ for r_ in plan.aim["passes"]}
        multi_seen = False
        for p in plan.passes:
            multi_seen = multi_seen or p.n_beads > 1
            covered = sp.kind == "single_bevel" and not multi_seen and p.role in ("root", "fill") and \
                (p.role == "root" or rows[p.k]["occurrence"] <= 1)
            assert p.aim_lateral_rule == ("kiro_practice" if covered else "geometric"), (name, p.k, p.role, p.n_beads)
            if not covered:
                assert rows[p.k]["why"].startswith("the geometric rule"), rows[p.k]["why"]
        if sp.kind != "single_bevel":
            assert plan.aim["toward_bevel_x"] is None and not plan.aim["check_0922"]["applies"]
            assert all("not a single-bevel groove" in r_["why"] or "beads in the layer" in r_["why"] or "cap passes" in r_["why"]
                       for r_ in rows.values())
        else:
            assert any(p.aim_lateral_rule == "kiro_practice" for p in plan.passes)
        st = plan.stations[4]
        for ly in range(1, max(p.layer for p in plan.passes) + 1):
            beads = [p for p in plan.passes if p.layer == ly and p.kind == "fill"]
            if len(beads) < 2:
                continue
            j = int(np.flatnonzero(beads[0].s_mm == st.s_mm)[0])
            xs = [float(p.aim_xz_mm[j, 0]) for p in beads]
            assert np.all(np.diff(xs) > 0.0)                                    # the bands from the -x wall to the +x wall
            for p in beads:                                                     # each bead stays in its band (no overlap)
                assert p.beads[j].area_mm2 == pytest.approx(p.area_mm2[j], rel=1e-6)
