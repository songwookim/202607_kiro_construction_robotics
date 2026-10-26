"""시편 대비 설계 보고 (사용자 2026-10-02: 실제 시편의 갭 벌어짐 등 시공 오차는 정확히 재서 보고할 대상 — 판정 아님).

specimen_report: 패스의 구간 중앙값 (정련 심 직선 좌표) → S 판 끝에서 S→E 로 잰 거리 [mm], 판정 (verdict.specimen)·측정 잡음 포함.
format_specimen_table: 콘솔 표 (설계 ± tol 밖 값에 '*')."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scan_core.localize import fidelity_gate
from scan_core.specimen import GrooveSpec
from scan_l0.autoteach import format_specimen_table, specimen_report

EXT = {"first_observed": True, "last_observed": True, "length_mm": 380.0, "spacing_mm": 0.3,
       "S": {"point": [0.6, -0.19, 0.05]}, "E": {"point": [0.6, 0.19, 0.05]}}
Q = {"root_gap_mm": {"tol": 0.2}, "groove_depth_mm": {"tol": 0.2}, "top_opening_mm": {"tol": 0.2}, "bevel_angle_deg": {"tol": None},
     "hi_lo_mm": {"target_max": 1.0}}                                           # 작업 파일의 설계 허용치 (플랫폼 기본값 없음)


def _fit(u):
    """정련 직선 c = 시편 가운데, u (방향은 패스에 따라 ±); 구간은 그 직선 좌표 t 로 [-0.19, 0.19] 를 20 mm 씩. 갭은 t 가 커질수록 벌어진다."""
    bins = [{"s_from_m": -0.19 + 0.02 * j, "s_to_m": -0.17 + 0.02 * j, "n": 100, "root_gap_mm": 5.0 + 0.025 * j, "depth_mm": 10.0,
             "top_opening_mm": 11.3, "hi_lo_mm": -0.1, "bevel_angle_deg": 31.6} for j in range(19)]
    fm = {"root_gap_mm": 5.2, "depth_mm": 10.0, "top_opening_mm": 11.3, "bevel_angle_deg": 31.6, "hi_lo_mm": -0.1,
          "floor_visible_fraction": 1.0, "root_edges_fraction": 1.0, "ok_fraction": 0.8, "fov_margin_mm": 60.0, "sensor_z_mm": 341.0,
          "root_gap_along_min_mm": 5.0, "root_gap_along_max_mm": 5.45, "root_gap_noise_mm": 0.15, "depth_noise_mm": 0.04,
          "top_opening_noise_mm": 0.14, "hi_lo_noise_mm": 0.05}
    return SimpleNamespace(c=np.array([0.6, 0.0, 0.05]), u=np.asarray(u, float), feature_medians=fm, diag={"specimen_bins": bins})


COV = {"groove": {"n_stations": 77, "min_count": 60, "median_count": 60, "short": [], "ok": True, "need": 5}}   # 단면 창 (10 mm, 0.17 mm 간격) 60 프로파일


@pytest.mark.parametrize("u", [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
def test_specimen_report_positions_are_measured_from_the_S_end(u):
    fit = _fit(u)
    spec = GrooveSpec(bevel_side="auto")
    v = fidelity_gate(fit.feature_medians, {}, spec, "fidelity", extent=EXT, qc_targets=Q, coverage=COV)
    assert v.status == "PASS", v.summary()
    rep = specimen_report(fit, v, EXT, 2, "S->E", spec)
    rows = rep["bins"]
    assert len(rows) == 19 and rep["bin_mm"] == pytest.approx(20.0)
    assert [round(r["from_S_mm"]) for r in rows] == list(range(0, 380, 20)) and rows[-1]["to_S_mm"] == pytest.approx(380.0)
    # 갭은 직선 좌표 t 가 커지는 쪽으로 벌어진다: u = +y 면 E 쪽 (S 에서 멀수록), u = −y 면 S 쪽
    g = [r["root_gap_mm"] for r in rows]
    assert (g == sorted(g)) if u[1] > 0 else (g == sorted(g, reverse=True))
    assert rep["scan_status"] == "PASS" and rep["summary"]["root_gap_mm"]["along_within"] is False
    assert rep["summary"]["top_opening_mm"]["within"] is False and rep["measurement_noise_mm"]["root_gap_mm"] == 0.15
    assert rep["design"]["top_opening_mm"] == pytest.approx(spec.top_opening_mm) and rep["length_mm"] == 380.0
    json.dumps(rep)                                                            # JSON 으로 쓸 수 있다

    lines = format_specimen_table(rep)
    assert lines[0].startswith("specimen vs design (pass 2, report only - not a scan failure); bins of 20 mm from the S end")
    first = next(l_ for l_ in lines if l_.strip().startswith("0-20")).split()
    # 첫 구간 (S 끝 0–20 mm): 갭 (u = +y 면 t 가 가장 작은 구간 5.00, u = −y 면 가장 큰 구간 5.45 → 설계 + 0.2 밖 '*'), 개구 11.30 > 10.77 + 0.2
    assert first == ["0-20", "5.00" if u[1] > 0 else "5.45*", "10.00", "11.30*", "31.60", "-0.10", "100"]
    assert lines[-1] == "  outside design +- tol (construction deviation, reported): gap 5.00..5.45, opening median 11.30"
    assert next(l_ for l_ in lines if l_.strip().startswith("design")).split() == ["design", "5.00+-0.2", "10.00+-0.2", "10.77+-0.2",
                                                                                   "30.00", "<=1.00"]


def test_specimen_report_without_extent_has_no_S_positions():
    fit = _fit([0.0, 1.0, 0.0])
    spec = GrooveSpec(bevel_side="auto")
    v = fidelity_gate(fit.feature_medians, {}, spec, "fidelity", extent=EXT, qc_targets=Q, coverage=COV)
    rep = specimen_report(fit, v, None, 1, "E->S", spec)
    assert rep["length_mm"] is None and all(r["from_S_mm"] is None for r in rep["bins"])
    assert any(l_.strip().startswith("?") for l_ in format_specimen_table(rep))
    empty = specimen_report(SimpleNamespace(c=fit.c, u=fit.u, feature_medians={}, diag={}), v, EXT, 1, "S->E", None)
    assert empty["bins"] == [] and empty["bin_mm"] is None and empty["design"] is None
    assert len(format_specimen_table(empty)) == 5                               # 머리말·열 이름·중앙값·설계·편차 (구간 없음)
