"""g1_outputs (G1_CORE_SPEC §5.4, §8.3, §9; WP4): 합성 SeamEstimate (곧은 심, 알려진 치수) → out_dir 에 sections.* + weld_plan.* 만,
내용 왕복, S→E 계약 통과, 설계 단면에서 4 패스 (충전 3 + 캡 1), 첫·마지막 스테이션 = r(e_S)·r(e_E), 베벨 쪽·hi-lo 부호, 빈 스테이션 보간,
계약 위반 → 계획 거부 (verification/ 에 남김)."""
import json
import math
import os
import zipfile

import numpy as np
import pytest

from g1_synth import synth_estimate, synth_prior
from scan_core.cross_sections import FACE_NAMES
from scan_core.g1_outputs import (G1WeldPlan, contract_frame, retract_weld_plan, sections_doc, weld_plan_from_estimate,
                                  weld_stations, write_outputs, write_sections)
from scan_core.g1_types import DIM_NAMES
from scan_core.weld_plan import PlanContractError, WeldProcess, check_plan_contract

OUT_FILES = ["sections.json", "sections.npz", "weld_plan.json", "weld_plan.npz"]


def _run(tmp_path, **kw):
    pr = synth_prior()
    est = synth_estimate(pr, **kw)
    res = write_outputs(est, pr, WeldProcess(), tmp_path / "out", base_frame="link0", k=pr.k)
    return pr, est, res


def test_only_the_four_files_in_out_dir(tmp_path):
    _, _, res = _run(tmp_path)
    # + both weld directions (user 2026-10-14): directions/S_to_E/ and directions/E_to_S/, each weld_plan.json / .npz
    assert sorted(os.listdir(tmp_path / "out")) == sorted(OUT_FILES + ["directions"])
    assert sorted(os.listdir(tmp_path / "out" / "directions")) == ["E_to_S", "S_to_E"]
    assert all(sorted(os.listdir(tmp_path / "out" / "directions" / d)) == ["weld_plan.json", "weld_plan.npz"] for d in ("E_to_S", "S_to_E"))
    assert sorted(os.path.basename(f) for f in res["files"]) == OUT_FILES
    with zipfile.ZipFile(tmp_path / "out" / "sections.npz") as z:                 # sections.npz: 압축 없음
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())


def test_sections_contents_roundtrip(tmp_path):
    pr, est, _ = _run(tmp_path)
    d = json.load(open(tmp_path / "out" / "sections.json", encoding="utf-8"))
    assert d["schema"] == "g1core.sections/1" and d["k"] == 3.0 and d["base_frame"] == "link0"
    assert d["accuracy"].startswith("relative")
    rS, rE = est.r(est.ends["S"].s_m), est.r(est.ends["E"].s_m)
    np.testing.assert_allclose(d["frame"]["S_m"], rS, atol=1e-12)
    np.testing.assert_allclose(d["frame"]["E_m"], rE, atol=1e-12)
    np.testing.assert_allclose(d["ends"]["S"]["point_m"], rS, atol=1e-12)
    assert d["length_mm"]["value"] == pytest.approx(380.0, abs=1e-6) and d["length_mm"]["design"] == 380.0
    assert d["ends"]["S"]["ratio"] == pytest.approx(0.2 / 0.25)               # σ / 1-σ precision target (user 2026-10-11)
    st = d["stations"]
    assert len(st) == 77 and st[0]["s_m"] == pytest.approx(0.0, abs=1e-12) and st[-1]["s_m"] == pytest.approx(0.380, abs=1e-9)
    np.testing.assert_allclose(np.diff([r["s_m"] for r in st]), 0.005, atol=1e-9)
    r0 = st[0]
    assert r0["root_ratio"] == pytest.approx(0.15 / 0.2) and r0["root_resolved"] is True
    assert r0["gap_mm"] == {"value": 5.0, "sigma": 0.05, "sigma_sys": 0.0, "n": 30, "tol": 0.5, "precision": 0.1,
                            "ratio": pytest.approx(0.5), "resolved": True}
    assert r0["hi_lo_mm"]["tol"] is None and r0["hi_lo_mm"]["precision"] is None and r0["hi_lo_mm"]["resolved"] is None
    assert d["precision"]["end_position_mm"] == 0.25 and d["precision"]["bevel_angle_deg"] == pytest.approx(
        math.degrees(math.atan(0.1 / (10.0 / math.cos(math.radians(30.0))))))
    assert r0["top_opening_mm"]["value"] == pytest.approx(5 + 10 * math.tan(math.radians(30)))
    assert set(r0["faces"]) == {"plate_top_left", "gap_bottom"} and r0["faces"]["gap_bottom"]["n_profiles"] == 30
    u, n, w = (np.array(r0[k]) for k in ("u", "n", "w"))
    np.testing.assert_allclose([u @ n, u @ w, n @ w], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(n, u), w, atol=1e-12)
    np.testing.assert_allclose(u, est.u, atol=1e-12)
    assert d["bevel_side"] == "left"                                              # bevel_sign_w +1 = +w = left looking S->E
    assert d["time_offset_ms"] == {"value": pytest.approx(3.7), "sigma": pytest.approx(1.0)}
    assert d["tolerances"]["end_position_mm"] == 1.0
    # npz (viz-compatible superset)
    with np.load(tmp_path / "out" / "sections.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    K, F = 77, len(FACE_NAMES)
    assert int(a["schema_version"]) == 101 and str(a["base_frame"]) == "link0" and float(a["k"]) == 3.0
    assert [str(x) for x in a["face_names"]] == list(FACE_NAMES)
    shapes = {"s_m": (K,), "s_range_m": (K, 2), "root_point_m": (K, 3), "root_sigma_m": (K, 2), "u": (3,), "n": (3,), "w": (3,),
              "normal": (K, F, 3), "point_m": (K, F, 3), "sigma_deg": (K, F), "n_profiles": (K, F), "ends_point_m": (2, 3),
              "ends_sigma_m": (2,)}
    for k_, shp in shapes.items():
        assert a[k_].shape == shp, k_
    for q in DIM_NAMES:
        assert a[q].shape == a[q + "_sigma"].shape == a[q + "_sigma_sys"].shape == (K,)
    np.testing.assert_allclose(a["gap_mm"], 5.0)
    np.testing.assert_allclose(a["root_point_m"], [r["root_point_m"] for r in st])
    np.testing.assert_allclose(a["ends_point_m"], [rS, rE], atol=1e-12)
    assert float(a["station_window_m"]) == 0.010
    qn = [str(x) for x in a["q_names"]]
    assert qn == ["root_lateral", "root_normal", "gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm"]
    assert a["resolved"].shape == (K, 6) and a["resolved"].all()
    j = FACE_NAMES.index("gap_bottom")
    np.testing.assert_allclose(a["point_m"][:, j], a["root_point_m"], atol=1e-12)
    assert np.isnan(a["normal"][:, FACE_NAMES.index("bevel_left")]).all()


def test_plan_four_passes_contract_and_ends(tmp_path):
    pr, est, res = _run(tmp_path)
    plan = res["plan"]
    assert isinstance(plan, G1WeldPlan)
    assert [p.kind for p in plan.passes] == ["fill", "fill", "fill", "cap"] and len(plan.passes) == 4
    fr = contract_frame(est)
    viol, checked = check_plan_contract(plan, fr, k=3.0)
    assert viol == [] and plan.ends["violations"] == []
    np.testing.assert_allclose(plan.stations[0].origin, est.r(est.ends["S"].s_m), atol=1e-12)
    np.testing.assert_allclose(plan.stations[-1].origin, est.r(est.ends["E"].s_m), atol=1e-12)
    assert plan.stations[0].s_mm == pytest.approx(0.0, abs=1e-9) and plan.stations[-1].s_mm == pytest.approx(380.0, abs=1e-6)
    sp = plan.stations[5].spec
    assert (sp.kind, sp.bevel_side) == ("single_bevel", "right") and plan.bevel_view == "left"
    assert sp.thickness_mm == pytest.approx(10.0) and sp.root_gap_mm == pytest.approx(5.0)
    assert sp.bevel_angle_deg == pytest.approx(30.0) and sp.hi_lo_mm == pytest.approx(0.0)
    assert sp.top_opening_mm == pytest.approx(plan.stations[5].measured["top_opening_mm"])
    # files round trip
    d = json.load(open(tmp_path / "out" / "weld_plan.json", encoding="utf-8"))
    assert d["schema_version"] == 3 and d["source"] == "g1core sections/1" and len(d["passes"]) == 4
    assert d["ends"]["checked"]["tol_S_mm"] == pytest.approx(0.6)
    for p in d["passes"]:
        assert len(p["aim_sigma_m"]["lateral"]) == 77 and p["aim_sigma_m"]["max"] == pytest.approx(1.5e-4)
    with np.load(tmp_path / "out" / "weld_plan.npz", allow_pickle=False) as z:
        for p in plan.passes:
            np.testing.assert_allclose(z[f"p{p.k}_pos_m"], p.dense_pos_m)
            assert str(z[f"p{p.k}_kind"]) == p.kind
    # S->E pass starts at the S end (aim offset is in the section plane)
    p1 = plan.passes[0]
    assert p1.direction == "S->E"
    assert float((p1.dense_pos_m[0] - fr.S) @ fr.u) == pytest.approx(0.0, abs=1e-9)
    assert float((p1.dense_pos_m[-1] - fr.E) @ fr.u) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("sign,side,t,hl", [(1, "right", 11.0, -1.0), (-1, "left", 10.0, 1.0)])
def test_bevel_side_and_hi_lo_sign(sign, side, t, hl):
    """H = 직각 쪽 상면 − 베벨 쪽 상면 (§4.6) = +1 mm → 단면 spec (x<0 'left', hi_lo = right − left)."""
    pr = synth_prior()
    dims = {"gap_mm": 5.0, "depth_mm": 10.0, "bevel_angle_deg": 30.0, "top_opening_mm": 10.77, "hi_lo_mm": 1.0, "root_face_mm": 0.0}
    est = synth_estimate(pr, bevel_sign_w=sign, dims=dims)
    st = weld_stations(est, pr.spec)
    sp = st[3].spec
    assert sp.bevel_side == side and sp.thickness_mm == pytest.approx(t) and sp.hi_lo_mm == pytest.approx(hl)
    top_bev = sp.thickness_mm + (sp.hi_lo_mm if side == "right" else 0.0)
    top_sq = sp.thickness_mm + (0.0 if side == "right" else sp.hi_lo_mm)
    assert top_bev == pytest.approx(10.0) and top_sq == pytest.approx(11.0)
    assert sections_doc(est, pr)["bevel_side"] == ("left" if sign > 0 else "right")
    plan = weld_plan_from_estimate(est, pr.spec, WeldProcess(), k=pr.k)
    assert plan.bevel_side == side


def test_station_without_profiles_is_interpolated(tmp_path):
    pr = synth_prior()
    dims = {"gap_mm": 5.0, "depth_mm": 10.0, "bevel_angle_deg": 30.0, "top_opening_mm": 10.77, "hi_lo_mm": 0.0, "root_face_mm": 0.0}
    est = synth_estimate(pr, dims=dims, missing=(10, 11))
    st = weld_stations(est, pr.spec)
    assert [i for i, s in enumerate(st) if s.interpolated] == [10, 11]
    assert st[10].spec.root_gap_mm == pytest.approx(5.0) and st[10].n_profiles == 0
    d = sections_doc(est, pr)
    assert d["stations"][10]["gap_mm"]["ratio"] is None or d["stations"][10]["gap_mm"]["ratio"] > 1   # σ = ∞ → not resolved
    assert d["stations"][10]["gap_mm"]["resolved"] is False and d["stations"][10]["faces"] == {}
    files = write_sections(d, tmp_path)
    with np.load(files[1], allow_pickle=False) as z:
        assert not z["resolved"][10, 2] and z["resolved"][9, 2] and np.isnan(z["gap_mm"][10])
    plan = weld_plan_from_estimate(est, pr.spec, WeldProcess(), k=pr.k)
    assert len(plan.passes) == 4 and any("interpolated" in n for n in plan.notes)


def test_contract_violation_refuses_and_keeps_plan_in_verification(tmp_path):
    pr = synth_prior()
    est = synth_estimate(pr)
    est.stations = est.stations[:-3]                       # stations stop 15 mm before the E end (k·σ = 0.6 mm)
    with pytest.raises(PlanContractError) as ei:
        weld_plan_from_estimate(est, pr.spec, WeldProcess(), k=pr.k)
    assert isinstance(ei.value.plan, G1WeldPlan) and ei.value.plan.ends["violations"]
    out = tmp_path / "out"
    with pytest.raises(PlanContractError):
        write_outputs(est, pr, WeldProcess(), out, k=pr.k)
    assert sorted(os.listdir(out)) == ["sections.json", "sections.npz", "verification"]
    assert sorted(os.listdir(out / "verification")) == ["weld_plan.json", "weld_plan.npz"]


def test_retract_weld_plan(tmp_path):
    _run(tmp_path)
    moved = retract_weld_plan(tmp_path / "out")
    assert len(moved) == 6                         # the primary weld_plan.* + both direction sets' (none delivered without it)
    assert sorted(os.listdir(tmp_path / "out")) == ["sections.json", "sections.npz", "verification"]
    assert sorted(os.listdir(tmp_path / "out" / "verification" / "directions")) == ["E_to_S", "S_to_E"]


def test_weld_length_is_the_seam_length_between_the_ends_not_the_taught_axis_difference():
    """kv_offA_a (2026-10-12): rough points ±30 mm to either side put the taught axis ~10.6° off the seam; the design-deviation WARNING
    said 'weld_length_mm measured 372.99' (the taught-axis difference s_E - s_S) while sections.json gave 379.46 mm. The measured length
    is the seam length between the delivered ends (sections.json's S->E contract length), its σ the ends' σ carried along the seam - the
    warning, the estimate log line and sections.json use one value (seam_estimate.delivered_length)."""
    from scan_core.seam_estimate import delivered_length, design_deviations
    pr = synth_prior()
    slope = math.tan(math.radians(10.6))
    for L_true, warn in ((0.37946, False), (0.3720, True)):
        est = synth_estimate(pr, length_m=L_true, a=(0.003, slope))
        ds = float(est.ends["E"].s_m - est.ends["S"].s_m)
        L, sig = delivered_length(est)
        assert L == pytest.approx(L_true, abs=1e-9) and ds < L_true - 0.006                     # the taught-axis difference is ~6.5 mm short
        g = float(np.linalg.norm(est.dr_ds(float(est.ends["S"].s_m))))                       # dL/ds of a straight line = |dr/ds|
        assert sig == pytest.approx(math.hypot(2e-4, 2e-4) * g, rel=1e-9) and g > 1.01
        doc = sections_doc(est, pr)
        assert doc["length_mm"]["value"] == pytest.approx(L * 1e3, abs=1e-9) and doc["length_mm"]["sigma"] == pytest.approx(sig * 1e3)
        fz = {"ends": {X: {"two_sided": True} for X in ("S", "E")}, "length_m": L, "length_sigma_m": sig}
        rows, lines = design_deviations(est, pr, fz)
        assert rows["weld_length_mm"]["measured"] == pytest.approx(L_true * 1e3, abs=1e-6)
        assert any(f"weld_length_mm measured {L_true * 1e3:.2f}" in ln for ln in lines) is warn, lines
    # an end that is not a point: no length (no warning)
    est.ends["E"] = type(est.ends["E"])(float("nan"), 2e-4, est.ends["E"].point_m, True, 2, [])
    assert all(math.isnan(v) for v in delivered_length(est))
