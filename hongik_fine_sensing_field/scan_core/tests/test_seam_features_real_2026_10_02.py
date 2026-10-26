"""실기 회귀 (2026-10-02 teach-live 자세 스냅샷 17개, GainVal 250): 10T 단일 베벨 30°·갭 5·뒷댐재 시편을 352–495 mm, 입사 0–4.5°,
라인 요 −8.8…+8.5° 로 본 50 프레임 중앙값 프로파일 + 반사가 심한 자세(13·15·17) 의 단일 프레임.

이전 추출기는 같은 시편에서 깊이 10.4–22.9 mm (루트 모서리 반사 스파이크·베벨 발치 반사를 바닥으로), 갭 −12…22 mm, 개구 최대
63 mm (판 끝 몇~20 mm 가 그루브 내부로 섞임), 'both'/'none' 오판을 냈다. 자료: scan_data/2026-10-01/orient_study, 분석
docs/lab_2026-10-01_findings.md §6."""
import json
import os

import numpy as np
import pytest

from scan_core.seam_features import extract_cross_section_features
from scan_core.specimen import GrooveSpec

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_groove_orient_2026-10-02.npz")
SPEC = GrooveSpec()                          # 10T, 30°, gap 5 (= 시편 설계)
DESIGN_OPENING = SPEC.root_gap_mm + SPEC.thickness_mm * np.tan(np.radians(SPEC.bevel_angle_deg))


@pytest.fixture(scope="module")
def data():
    d = np.load(DATA)
    return d, json.loads(str(d["meta_json"]))


def _check(f, tag):
    assert f.ok and f.groove_found, (tag, f.report())
    assert 4.2 <= f.root_gap_mm <= 5.9, (tag, f.report())                       # 설계 5
    assert 9.3 <= f.depth_mm <= 10.6, (tag, f.report())                         # 설계 10 (뒷댐재 상면)
    assert 29.0 <= f.bevel_angle_deg <= 34.0, (tag, f.report())                  # 설계 30
    assert abs(f.top_opening_mm - DESIGN_OPENING) <= 1.0, (tag, f.report())     # 설계 10.77
    assert "wall_ambiguous" not in f.flags, (tag, f.report())


def test_all_17_median_profiles_match_the_design(data):
    d, meta = data
    sides, gaps, depths = set(), [], []
    for k in range(d["x_median"].shape[0]):
        f = extract_cross_section_features(d["x_median"][k], d["z_median"][k], SPEC)
        _check(f, f"snapshot {meta[k]['snapshot']} {meta[k]['orientation_deg']} {meta[k]['plate_distance_mm']} mm")
        sides.add(f.bevel_side)
        gaps.append(f.root_gap_mm)
        depths.append(f.depth_mm)
    assert len(sides) == 1 and sides <= {"left", "right"}                       # 같은 시편 → 같은 쪽 (센서 프레임)
    assert np.std(gaps) < 0.45 and np.std(depths) < 0.25                        # 자세가 달라도 일관


def test_root_corner_reflections_are_not_the_floor(data):
    """라인 요·진행 기울기 자세 (12–17): 바닥 아래 6–13 mm 의 좁은 반사 스파이크가 있어도 깊이는 뒷댐재 (이전 16–23 mm)."""
    d, meta = data
    for k in range(11, 17):
        f = extract_cross_section_features(d["x_median"][k], d["z_median"][k], SPEC)
        assert "below_floor_reflections" in f.flags and f.n_below_floor > 0, f.report()
        assert f.depth_mm < 10.6, f.report()


def test_single_frames_of_the_worst_poses(data):
    d, _meta = data
    for x, z, i in zip(d["frames_x"], d["frames_z"], d["frames_snapshot"]):
        _check(extract_cross_section_features(x, z, SPEC), f"frame of snapshot {int(i)}")


def test_result_does_not_depend_on_the_design_spec(data):
    """설계값은 비교에만 쓴다: spec 없이도 같은 치수 (바닥 = 가장 넓은 수평면, 설계 깊이로 고르지 않음)."""
    d, _meta = data
    for k in (0, 6, 12, 16):
        a = extract_cross_section_features(d["x_median"][k], d["z_median"][k], SPEC)
        b = extract_cross_section_features(d["x_median"][k], d["z_median"][k], None)
        assert b.depth_mm == pytest.approx(a.depth_mm, abs=1e-9) and b.bevel_side == a.bevel_side
        assert b.top_opening_mm == pytest.approx(a.top_opening_mm, abs=1e-9)
