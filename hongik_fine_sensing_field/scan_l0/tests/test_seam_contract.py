"""S→E 좌표 계약 (2026-10-09, 실기 1G 배치 분석 3.4) — 온라인 병합 경로 (autoteach.merged_pass_features → weld_plan_from_features).

실기 b1 t07·b3 t05: 마지막 전체 패스가 E→S 였고, 병합이 그 패스의 정련 직선 u (진행 방향 = E→S) 를 축으로 넘겨 용접 계획이 'S->E' 표기로
E 에서 S 로 갔다 (b1 t07: 첫 단면 S→E 좌표 +382.62, 마지막 +3.97 — S 쪽 3.97 mm 미용접, 베벨 보기 반대). 계약: 끝은 base 점 S, E 로만,
축 = unit(E − S). 기준 패스 진행 방향 (S→E / E→S) 과 자세 (1G 수평 판 / 3G 세운 판) 에 무관하게 같은 계획이어야 한다."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from scan_core.specimen import GrooveSpec
from scan_core.weld_plan import WeldProcess, end_insets_mm, weld_plan_from_features
from scan_l0 import scan_quality as SQ
from scan_l0.autoteach import merged_pass_features

SPEC = GrooveSpec(bevel_side="left")
LAYOUTS = {   # (S, E, 판 법선 (센서 쪽), 공정 방향, 중력)
    "1G": (np.array([1.435, 0.013, -0.144]), np.array([1.057, -0.024, -0.150]), np.array([-0.0166, 0.0035, 0.9999]), "S->E", None),
    "3G": (np.array([0.50, 0.70, 0.11]), np.array([0.50, 0.70, 0.49]), np.array([0.0, -1.0, 0.0]), "uphill", (0.0, 0.0, -1.0)),
}


def frame_of(S, E, n):
    n = n / np.linalg.norm(n)
    d = (E - S) - ((E - S) @ n) * n
    u = d / np.linalg.norm(d)
    return u, n, np.cross(n, u), float(np.linalg.norm(d))


def pass_feats(S, E, n, *, spacing=0.0003, gap=5.0):
    """한 패스의 프로파일별 특징 (루트·상부 중심 base 폴리라인, 개구·갭·깊이·베벨) — 베벨은 단면 −x (w 반대쪽), 판 끝 밖 10 mm 포함."""
    u, n, w, L = frame_of(S, E, n)
    s = np.arange(-0.010, L + 0.010, spacing)
    Pr = S + np.outer(s, u)
    O = 11.0 + 0.4 * s / L                                                     # 개구가 S→E 로 벌어진다 (거울상이면 반대)
    Pt = Pr + np.outer(-0.5 * (O - gap) * 1e-3, w) + 0.010 * n
    m = s.size
    return {"ok": (s >= 0.0) & (s <= L), "root_polyline_base_m": Pr, "top_polyline_base_m": Pt, "root_gap_mm": np.full(m, gap),
            "depth_mm": np.full(m, 10.0), "top_opening_mm": O, "bevel_angle_deg": np.degrees(np.arctan((O - gap) / 10.0)),
            "hi_lo_mm": np.full(m, -0.1), "bevel_side": np.array(["left"] * m), "n_base": n}


def ref_fit(S, E, n, travel: str):
    """기준 패스의 정련 직선 (fit_pass 와 같은 모양): u = 진행 방향, c = 시편 가운데, s_a/s_b 는 그 좌표의 정렬된 양끝."""
    u, n, _w, L = frame_of(S, E, n)
    u_fit = u if travel == "S->E" else -u
    return SimpleNamespace(ok=True, c=0.5 * (S + E), u=u_fit, n=n, s_a=-0.5 * L, s_b=0.5 * L)


def extent(S, E):
    return {"S": {"point": S.tolist(), "sigma_m": 0.086e-3, "observed": True}, "E": {"point": E.tolist(), "sigma_m": 0.086e-3,
                                                                                      "observed": True}, "spacing_mm": 0.3}


def plan_for(layout: str, travel: str):
    S, E, n, direction, g = LAYOUTS[layout]
    f1, f2 = pass_feats(S, E, n, gap=5.1), pass_feats(S, E, n)
    entries = [{"k": k, "feats": f, "q": SQ.profile_quality(f, SPEC), "posture": {"incidence_deg": 5.0}, "full": True}
               for k, f in ((1, f1), (2, f2))]
    merged, choice, bin_m = merged_pass_features(entries, extent(S, E), ref_fit(S, E, n, travel), WeldProcess())
    plan = weld_plan_from_features(merged, extent(S, E), SPEC, WeldProcess(direction=direction), position=layout,
                                   gravity_axis_base=g, k=3.0)
    return S, E, merged, choice, plan


@pytest.mark.parametrize("layout", ["1G", "3G"])
def test_merge_and_plan_are_the_same_whichever_way_the_reference_pass_ran(layout):
    S, E, m_fwd, c_fwd, p_fwd = plan_for(layout, "S->E")
    _S, _E, m_rev, c_rev, p_rev = plan_for(layout, "E->S")
    u, _n, _w, L = frame_of(S, E, LAYOUTS[layout][2])
    for merged, choice in ((m_fwd, c_fwd), (m_rev, c_rev)):
        assert float(np.asarray(merged["u_base"]) @ (E - S)) > 0.0 and np.allclose(merged["line_c_base_m"], S)
        assert float(merged["s_a_m"]) == 0.0 and float(merged["s_b_m"]) == pytest.approx(L, abs=1e-6)
        assert choice[0]["s_from_m"] == pytest.approx(0.0) and choice[-1]["s_to_m"] == pytest.approx(L, abs=1e-6)
    np.testing.assert_array_equal(m_fwd["s_line_m"], m_rev["s_line_m"])
    assert [c_["pass"] for c_ in c_fwd] == [c_["pass"] for c_ in c_rev]
    for p in (p_fwd, p_rev):
        O = np.array([st.origin for st in p.stations])
        assert abs(float((O[0] - S) @ u)) < 1e-9 and abs(float((O[-1] - E) @ u)) < 1e-9       # 첫 단면 = S, 마지막 = E
        assert p.bevel_side == "left" and p.bevel_view == "right"
        opening = [st.measured["top_opening_mm"] for st in p.stations]
        assert opening[0] < opening[-1]                                                        # S→E 로 벌어지는 실측 그대로
        for wp in p.passes:
            start = S if wp.direction == "S->E" else E
            assert abs(float((wp.aim_base_m[0] - start) @ u)) < 1e-9
            assert abs(float((wp.dense_pos_m[0] - start) @ u) - (1 if wp.direction == "S->E" else -1)
                       * end_insets_mm(p)[0] * 1e-3) < 1e-9
    np.testing.assert_allclose([st.origin for st in p_rev.stations], [st.origin for st in p_fwd.stations], rtol=0, atol=1e-12)
    for a, b in zip(p_rev.passes, p_fwd.passes):
        assert a.direction == b.direction
        np.testing.assert_allclose(a.dense_pos_m, b.dense_pos_m, rtol=0, atol=1e-12)
    if layout == "3G":
        assert all(wp.direction == "S->E" for wp in p_fwd.passes)                             # S 가 아래: 위보기 = S→E
    assert p_rev.ends["checked"]["tol_S_mm"] == pytest.approx(3.0 * 0.086) and math.isfinite(p_rev.ends["length_mm"])
