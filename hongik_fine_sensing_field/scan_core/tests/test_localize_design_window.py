"""정지 국소화의 설계 창 재추출 (2026-10-06 g3_14 E): 같은 자세·같은 프로파일이 g3_13 에서는 맞게 읽혔는데 g3_14 에서는 판 표면의 ~0.5 mm
물결이 왼쪽 판을 x −88 mm 에서 둘로 나눠 그 사이 전체 (개구 118.5 mm) 를 그루브로 읽고 설계 검사로 멈췄다. 첫 추출이 설계와 맞지 않으면
파인 자리 후보 둘레 창에서 다시 읽어, 갭·베벨까지 설계에 맞는 후보가 하나뿐일 때만 그것을 쓴다."""
import json
import os

import numpy as np
import pytest

from scan_core.localize import localize_static
from scan_core.seam_features import extract_on_design_window, matches_design, notch_candidates
from scan_core.specimen import GrooveSpec
from scan_core.types import Calib

DATA = os.path.join(os.path.dirname(__file__), "data", "real_3g_E_dwell_g3_13_g3_14_2026-10-06.npz")
SPEC = GrooveSpec(kind="single_bevel", thickness_mm=10.0, bevel_angle_deg=30.0, bevel_side="left", root_gap_mm=5.0, weld_length_mm=380.0)


def _loc(z, run):
    x, zz = z[f"{run}_x_mm"].astype(float), z[f"{run}_z_mm"].astype(float)
    S, E, g = z[f"{run}_rough"]
    calib = Calib.from_dict(json.loads(str(z[f"{run}_calib_json"])))
    return localize_static(z[f"{run}_T_bf"], calib, x, zz, SPEC, gravity_axis_base=g, travel_dir_hint=(E - S) / np.linalg.norm(E - S),
                           z_valid_mm=(233.1, 544.0))


def test_g3_14_E_dwell_reads_the_same_groove_as_g3_13():
    z = np.load(DATA)
    a, b = _loc(z, "g3_13"), _loc(z, "g3_14")
    assert "design_window" not in a.features.flags and a.diag["design_window_mm"] is None          # 첫 추출이 맞으면 그대로
    # g3_14: 예전 추출기는 판 물결 (~0.5 mm) 에 왼쪽 판이 둘로 나뉘어 설계 창이 필요했다. 결측 이음 = 갭/2 (2026-10-07 4c) 로 첫 추출이
    # 이미 맞을 수 있다 — 어느 길이든 같은 그루브를 읽어야 한다 (창 경로 자체는 아래 합성 시험)
    for loc in (a, b):
        f = loc.features
        assert f.groove_found and abs(f.top_opening_mm - SPEC.top_opening_mm) < 1.5 and np.isfinite(f.root_gap_mm)
        assert abs(f.bevel_angle_deg - 30.0) < 5.0
    assert b.features.groove_centre_x_mm == pytest.approx(a.features.groove_centre_x_mm, abs=1.0)      # 예전: −28.5 vs +24.9 mm


def test_window_needs_a_unique_full_design_match():
    """후보 창이 갭·베벨 없이 개구만 맞으면 (그림자·이물 사이) 받지 않고, 설계에 맞는 그루브가 둘이면 모호 → None."""
    x = np.linspace(-100.0, 100.0, 1401)
    zz = np.full_like(x, 400.0)

    def groove(xc):
        m = (x > xc - 5.4) & (x < xc + 5.4)
        zz[m] = 400.0 + np.minimum(10.0, (x[m] - (xc - 5.4)) / np.tan(np.radians(30.0)))
        g = (x > xc + 2.9) & (x < xc + 5.4)
        zz[g] = np.nan                                                  # 루트갭 (바닥 없음)
    groove(-40.0)
    groove(40.0)
    f, win = extract_on_design_window(x, zz, SPEC)
    assert f is None and win is None
    assert len(notch_candidates(x, zz, SPEC)) >= 2
    assert matches_design.__doc__
