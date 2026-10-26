"""KIRO groove estimator on a synthetic single-bevel groove (10 deep, 5 root gap, 30 deg bevel, 200 long)."""

import numpy as np
import pytest

from construct_robot.core.groove_estimate import GrooveEstimateError, _profile_medians, _ransac_line, estimate_groove, groove_qc


@pytest.mark.parametrize("shuffled", [False, True])
def test_profile_medians_preserve_independent_profiles_and_outliers(shuffled):
    # Missing frame IDs, uneven profile sizes and an outlier must keep the old exact median.
    index = np.array([2, 2, 2, 7, 7, 19, 19, 19, 19, 40])
    values = np.array([1.0, 3.0, 500.0, 4.0, 8.0, -6.0, -4.0, 2.0, 10.0, 9.0])
    if shuffled:
        order = np.random.default_rng(42).permutation(len(index))
        index, values = index[order], values[order]
    expected = np.array([np.median(values[index == frame]) for frame in np.unique(index)])
    np.testing.assert_array_equal(_profile_medians(values, index), expected)


def test_profile_medians_accept_no_profiles():
    assert _profile_medians(np.empty(0), np.empty(0, dtype=int)).shape == (0,)


def test_ransac_keeps_strict_distance_threshold_for_both_sides():
    class FixedPair:
        def choice(self, _size, _count, replace):
            assert replace is False
            return 0, 1
    on_line = np.c_[np.arange(20), np.zeros(20)]
    inside = np.array([[5, .149], [6, -.149]])
    boundary_and_outside = np.array([[7, .15], [8, -.15], [9, 3], [10, -3]])
    points = np.vstack([on_line, inside, boundary_and_outside])
    line = _ransac_line(points, (0, 8), FixedPair(), iters=1)
    assert line is not None
    assert line[2] == pytest.approx(22 / 26)
    np.testing.assert_allclose(line[0], np.vstack([on_line, inside]).mean(axis=0), atol=1e-12)


def synthetic_scan(start=-15.0, end=215.0, skew_deg=0.6, noise=0.03, outliers=0.03, seed=0, plate_len=200.0, mirror=False):
    """Profiles every 0.3 mm along the scan axis (World -Y); the seam is skewed a little against it."""
    rng = np.random.default_rng(seed)
    pts, idx, origins = [], [], []
    k = np.radians(skew_deg)
    for f, a in enumerate(np.arange(start, end, 0.3)):
        w = np.linspace(-30, 30, 1200)
        w_seam = w - np.tan(k) * a                                    # lateral in the seam frame
        t30 = np.tan(np.radians(30.0))
        h = np.full_like(w, 10.0)                                     # plate tops
        bevel = (w_seam > -2.5 - 10.0 * t30) & (w_seam < -2.5)
        h = np.where(bevel, (-2.5 - w_seam) / t30, h)                 # 30 deg bevel down to the root corner
        h = np.where((w_seam >= -2.5) & (w_seam <= 2.5), 0.0, h)     # bottom (root gap 5), vertical face at +2.5
        if mirror:                                                    # bevel on +lateral, vertical face at -2.5
            h = h[::-1]
        if not 0.0 <= a <= plate_len:                                 # past the plate ends: floor far below
            h = np.full_like(w, -40.0)
        h = h + rng.normal(0, noise, len(h))
        bad = rng.random(len(h)) < outliers
        h[bad] += rng.normal(0, 3, bad.sum())
        x = np.full_like(w, 0.0) + w
        p = np.c_[x * 1e-3, np.full_like(x, -a * 1e-3 - 1.5), h * 1e-3 + 0.56]
        pts.append(p)
        idx.append(np.full(len(p), f))
        origins.append([0.05, -a * 1e-3 - 1.5, 0.9])
    return np.vstack(pts), np.concatenate(idx), np.array(origins)


def test_root_corner_ends_and_groove_dims_are_recovered():
    P, fi, o = synthetic_scan()
    g = estimate_groove(P, fi, o)
    assert g["length_mm"] == pytest.approx(200.0, abs=1.0)
    assert g["depth_mm"] == pytest.approx(10.0, abs=0.3)
    assert g["bevel_deg_from_vertical"] == pytest.approx(30.0, abs=1.5)
    # corner 1 = bevel x bottom at lateral -2.5 (seam frame), followed along the skewed seam
    for key, a in (("corner1_S", 0.0), ("corner1_E", 200.0)):
        expected_x = (-2.5 + np.tan(np.radians(0.6)) * a) * 1e-3
        assert g[key][0] == pytest.approx(expected_x, abs=0.0004)
        assert g[key][2] == pytest.approx(0.56, abs=0.0004)


def test_a_scan_that_starts_on_the_plate_is_refused():
    P, fi, o = synthetic_scan(start=5.0)
    with pytest.raises(GrooveEstimateError, match="starts on the plate"):
        estimate_groove(P, fi, o)


def test_a_scan_line_skewed_like_the_cell_keeps_every_section_and_finds_the_gap():
    # the real scans run 2-3 deg off the seam: the groove drifts ~10 mm across the scan frame
    P, fi, o = synthetic_scan(skew_deg=3.0)
    g = estimate_groove(P, fi, o)
    assert g["bevel_side"] == "left" and g["sections_skipped"] == {}
    assert g["scan_skew_deg"] == pytest.approx(3.0, abs=0.3)
    assert g["root_gap_mm"] == pytest.approx(5.0, abs=0.3)
    assert g["length_mm"] == pytest.approx(200.0, abs=1.0)
    expected_x = (-2.5 + np.tan(np.radians(3.0)) * 200.0) * 1e-3
    assert g["corner1_E"][0] == pytest.approx(expected_x, abs=0.0005)


def test_the_bevel_side_follows_the_bevel_face_not_the_scan_frame():
    P, fi, o = synthetic_scan(skew_deg=2.0, mirror=True)
    g = estimate_groove(P, fi, o)
    assert g["bevel_side"] == "right"
    assert g["corner1_S"][0] == pytest.approx(0.0025, abs=0.0005)


def test_qc_compares_against_the_specimen_design():
    P, fi, o = synthetic_scan(plate_len=190.0)
    rows = {r["name"]: r for r in groove_qc(estimate_groove(P, fi, o))}
    assert rows["weld_length_mm"]["ok"] is False and rows["weld_length_mm"]["diff"] == pytest.approx(-10.0, abs=1.0)
    assert rows["groove_depth_mm"]["ok"] and rows["bevel_angle_deg"]["ok"]
