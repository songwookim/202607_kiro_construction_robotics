"""bead_geometry: convex beads (user + KIRO 2026-10-13 night) - the circular-arc free bead (one shape parameter, the bead model's
aspect ratio), its placement on the current surface (pool / spill / climb / over / band), mass conservation, the convex solid of the
weld scene and the crown ratio a post-weld profile would give back (the later calibration)."""
import math

import numpy as np
import pytest

from scan_core.bead_geometry import (BeadSection, Profile, bare_profile, crown_height_mm, crown_ratio_from_profiles, groove_edges_mm,
                                     place_bead, pool_cuts, pool_level)
from scan_core.bead_model import (MassBalanceBeadModel, PassParams, arc_area_ratio, arc_ratio_for_area_ratio, crown_ratio_for_toe_angle,
                                  crown_ratio_of, toe_angle_for_crown_ratio)
from scan_core.specimen import GrooveSpec

K20 = GrooveSpec(kind="single_bevel", thickness_mm=10.514106537212687, bevel_angle_deg=30.49794880937114, bevel_side="left",
                 root_gap_mm=5.199548388145472, root_face_mm=0.2804609679147437, hi_lo_mm=-0.6942210186257858)


def _flat(x0=-30.0, x1=30.0, z=0.0, step=0.1, wall=10.0):
    """A wide flat-bottomed trough: the floor at z between vertical walls at x0, x1 (the groove top edges), plates beyond."""
    x = np.linspace(x0, x1, int(round((x1 - x0) / step)) + 1)
    X = np.concatenate([[x0 - 20.0, x0], x, [x1, x1 + 20.0]])
    Z = np.concatenate([[z + wall, z + wall], np.full(x.size, z), [z + wall, z + wall]])
    return Profile(X, Z, (x0, x1))


def _area_between(before: Profile, after: Profile) -> float:
    xs = np.unique(np.concatenate([before.x, after.x]))
    xs = xs[(xs > max(before.x[0], after.x[0])) & (xs < min(before.x[-1], after.x[-1]))]
    lo, hi = before.span(float(xs[0]), float(xs[-1])), after.span(float(xs[0]), float(xs[-1]))
    return float(np.trapz(hi[1], hi[0]) - np.trapz(lo[1], lo[0]))


def test_the_circular_arc_bead_and_its_one_shape_parameter():
    """The free bead: a circular-arc segment of height kappa x width; area g(kappa) w^2 (the arc's segment area; ~ 2/3 kappa for a
    flat bead), the toe angle 2 atan(2 kappa) - the one shape parameter; the bead model's width and crown ratio follow (weave: the same
    area over the woven width - flatter)."""
    for k in (0.05, 0.3, 0.5):
        th = 2.0 * math.atan(2.0 * k)
        R = (0.25 + k * k) / (2.0 * k)                                  # for a unit chord
        assert arc_area_ratio(k) == pytest.approx(R * R * (th - math.sin(th) * math.cos(th)), rel=1e-12)
        assert arc_ratio_for_area_ratio(arc_area_ratio(k)) == pytest.approx(k, rel=1e-9)
        u = np.linspace(0.0, 1.0, 20001)
        assert float(np.trapz(crown_height_mm(u, 1.0, k), u)) == pytest.approx(arc_area_ratio(k), rel=1e-6)
        assert float(crown_height_mm(0.5, 1.0, k)) == pytest.approx(k) and float(crown_height_mm(0.0, 1.0, k)) == pytest.approx(0.0)
    assert arc_area_ratio(1e-4) == pytest.approx(2.0 / 3.0 * 1e-4, rel=1e-3)
    # the one shape parameter is the toe angle (user 2026-10-13 night): the arc meets its chord at it, kappa = tan(toe / 2) / 2
    for toe in (20.0, 30.0, 61.93):
        k = crown_ratio_for_toe_angle(toe)
        assert toe_angle_for_crown_ratio(k) == pytest.approx(toe, abs=1e-9)
        slope = float((crown_height_mm(1e-7, 1.0, k) - crown_height_mm(0.0, 1.0, k)) / 1e-7)
        assert math.degrees(math.atan(slope)) == pytest.approx(toe, abs=1e-3)
    bm = MassBalanceBeadModel()
    k30 = crown_ratio_for_toe_angle(30.0)
    assert bm.toe_angle_deg == 30.0 and bm.aspect_ratio == pytest.approx(k30)
    p = PassParams("fill", 10.8, 0.007)
    a, w = bm.area_mm2(p), bm.width_mm(p)
    assert a == pytest.approx(arc_area_ratio(k30) * w * w) and bm.crown_ratio(p) == pytest.approx(k30) == crown_ratio_of(bm, p)
    pw = PassParams("cap", 10.2, 0.006, weave_amp_mm=4.0)
    assert bm.width_mm(pw) == pytest.approx(math.sqrt(bm.area_mm2(pw) / arc_area_ratio(k30)) + 8.0)
    kw = bm.crown_ratio(pw)
    assert kw < k30 and arc_area_ratio(kw) * bm.width_mm(pw) ** 2 == pytest.approx(bm.area_mm2(pw), rel=1e-9)


def test_a_bead_on_a_flat_plate_is_the_free_bead_centred_on_its_aim():
    """On a flat floor wider than the bead (the pool spills over the flat at its level): toes on the floor, chord = the free width,
    height kappa x chord, centred on the aim; area exact; the surface after is the plate + the arc."""
    prof = _flat()
    a, k = 27.6, 0.3
    b, after = place_bead(prof, (3.0, 0.0), a, k)
    w = math.sqrt(a / arc_area_ratio(k))
    assert b.phase == "spill" and b.area_mm2 == pytest.approx(a, rel=1e-6)
    assert b.chord_mm == pytest.approx(w, rel=2e-3) and 0.5 * (b.toe_l[0] + b.toe_r[0]) == pytest.approx(3.0, abs=1e-6)
    assert b.toe_l[1] == b.toe_r[1] == 0.0 and b.apex[1] == pytest.approx(k * b.chord_mm, rel=1e-6)
    assert _area_between(prof, after) == pytest.approx(a, rel=1e-6)
    assert np.all(np.diff(after.x) >= 0.0)


def test_a_bead_in_a_groove_is_a_level_pool_with_the_arc_on_top():
    """In a groove the deposit fills from the bottom up to a level toe chord between the walls (toes on the walls at one height), the
    arc of height kappa x chord above it; the bead's area = its mass-balance area, whatever the aim inside the groove."""
    sp = GrooveSpec(bevel_side="left", root_gap_mm=5.0, root_face_mm=1.0)
    prof = bare_profile(sp, 20.0, 0.1)
    g = 2.5
    for aim in ((-g, 1.0), (0.0, 0.0), (g, 0.0)):                      # the bevel corner, the gap centre, the square wall's foot
        b, after = place_bead(prof, aim, 30.0, 0.3)
        assert b.phase == "pool" and b.toe_l[1] == pytest.approx(b.toe_r[1]) and b.area_mm2 == pytest.approx(30.0, rel=1e-6)
        z = b.toe_l[1]
        xl, xr = (float(v) for v in __import__("scan_core.bead_fill", fromlist=["wall_x_mm"]).wall_x_mm(sp, z))
        assert (b.toe_l[0], b.toe_r[0]) == pytest.approx((xl, xr), abs=1e-6)
        assert b.apex[1] == pytest.approx(z + 0.3 * (xr - xl), rel=1e-9)
        from scan_core.bead_fill import groove_area_below_mm2
        assert float(groove_area_below_mm2(sp, z)) + arc_area_ratio(0.3) * (xr - xl) ** 2 == pytest.approx(30.0, rel=2e-3)
        assert _area_between(prof, after) == pytest.approx(30.0, rel=1e-6)


def test_the_next_bead_fills_the_pockets_beside_the_crown_and_merges_over_it():
    """Aimed at the root bead's toe on the square wall (KIRO's fill 1), the pool fills that pocket first and merges over the root's
    crown into the bevel-side pocket: a level chord between the walls again. Aimed at the crown's top it goes down both ways."""
    prof = bare_profile(K20, 30.0, 0.1)
    root, p1 = place_bead(prof, (-0.5 * K20.root_gap_mm, K20.root_face_mm), 30.08, 0.3)
    fill1, p2 = place_bead(p1, root.toe_r, 27.63, 0.3)
    assert fill1.phase == "pool" and fill1.toe_l[1] == pytest.approx(fill1.toe_r[1]) and fill1.toe_l[1] > root.apex[1]
    alt, _ = place_bead(p1, (root.apex[0], float(p1.at(root.apex[0]))), 27.63, 0.3)       # the same bead from the crown's top
    assert np.allclose([alt.toe_l, alt.toe_r], [fill1.toe_l, fill1.toe_r], atol=1e-6)
    small, p3 = place_bead(p1, root.toe_r, 2.0, 0.3)                    # a small deposit stays in the square-wall pocket
    assert small.toe_r[0] == pytest.approx(root.toe_r[0]) and small.toe_l[0] > root.apex[0]
    assert _area_between(p1, p3) == pytest.approx(2.0, rel=1e-6)


def test_over_the_plate_tops_the_toe_climbs_the_higher_edge_then_both_move_out():
    """A deposit larger than the groove left: the pool rises to the lower groove top edge (hi-lo: the right plate 0.69 mm lower), the
    other toe climbs to its own edge, then both toes move out over the plates by equal lengths - the toe chord from plate to plate,
    tilted by the hi-lo. Area exact; the surface stays a function of x."""
    prof = bare_profile(K20, 40.0, 0.1)
    xeL, tL, xeR, tR = groove_edges_mm(K20)
    b, after = place_bead(prof, (0.0, 0.0), 100.0, 0.12)
    assert b.phase == "over" and b.area_mm2 == pytest.approx(100.0, rel=1e-6)
    assert b.toe_l[1] == pytest.approx(tL) and b.toe_r[1] == pytest.approx(tR)
    assert xeL - b.toe_l[0] == pytest.approx(b.toe_r[0] - xeR, rel=1e-6) and b.toe_r[0] > xeR
    assert _area_between(prof, after) == pytest.approx(100.0, rel=1e-6) and np.all(np.diff(after.x) >= 0.0)
    climb, c_after = place_bead(prof, (0.0, 0.0), 90.0, 0.12)         # between the two edges: the toe on the lower edge, the other climbing
    assert climb.phase == "climb" and np.allclose(climb.toe_r, (xeR, tR)) and tR < climb.toe_l[1] < tL
    assert _area_between(prof, c_after) == pytest.approx(90.0, rel=1e-6)
    below, _ = place_bead(prof, (0.0, 0.0), 85.0, 0.12)
    assert below.phase == "pool" and below.toe_l[1] == pytest.approx(below.toe_r[1]) and below.toe_l[1] < tR


def test_beads_of_a_band_stay_in_it_and_tile_the_layer():
    """A multi-bead layer: the pool at the layer's level cut into equal areas; each bead fills its band (the cuts act as walls) - the
    beads tile the layer without overlap, each with its own area."""
    sp = GrooveSpec(kind="single_v", thickness_mm=20.0, bevel_angle_deg=30.0, root_gap_mm=3.0, root_face_mm=2.0)
    prof = bare_profile(sp, 40.0, 0.1)
    z1, xl, xr = pool_level(prof, 3 * 25.0)
    cuts = pool_cuts(prof, z1, xl, xr, 3)
    assert xl < cuts[0] < cuts[1] < xr
    p = prof
    beads = []
    for b in range(3):
        band = (None if b == 0 else float(cuts[b - 1]), None if b == 2 else float(cuts[b]))
        lo = xl if b == 0 else cuts[b - 1]
        hi = xr if b == 2 else cuts[b]
        x = 0.5 * (lo + hi)
        bead, p2 = place_bead(p, (x, float(p.at(x))), 25.0, 0.3, band=band)
        assert bead.area_mm2 == pytest.approx(25.0, rel=1e-6) and _area_between(p, p2) == pytest.approx(25.0, rel=1e-6)
        if band[0] is not None:
            assert bead.toe_l[0] >= band[0] - 1e-9
        if band[1] is not None:
            assert bead.toe_r[0] <= band[1] + 1e-9
        beads.append(bead)
        p = p2
    assert _area_between(prof, p) == pytest.approx(75.0, rel=1e-6)


def test_the_convex_solid_contains_the_bead_and_its_top_is_the_arc():
    """The weld scene's solid of a bead: convex (CCW), contains the bead's region (the surface before under it, the arc above it), its
    top above the arc by at most the tangent polygon's bound, with a face straight up at the crown (weld_clearance's fast bound)."""
    prof = bare_profile(K20, 30.0, 0.1)
    _r, p1 = place_bead(prof, (-0.5 * K20.root_gap_mm, K20.root_face_mm), 30.08, 0.3)
    b, _p2 = place_bead(p1, _r.toe_r, 27.63, 0.3)
    H = b.hull
    e = np.roll(H, -1, axis=0) - H
    cross = e[:, 0] * np.roll(e, -1, axis=0)[:, 1] - e[:, 1] * np.roll(e, -1, axis=0)[:, 0]
    assert np.all(cross >= -1e-9)                                       # convex, counter-clockwise
    reg = b.region(200)
    n = np.stack([e[:, 1], -e[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1)[:, None]
    g = np.einsum("pk,qk->pq", reg, n) - np.sum(n * H, axis=1)[None, :]
    assert np.all(g <= 1e-6)                                            # every point of the bead inside every face
    xs = np.linspace(b.toe_l[0], b.toe_r[0], 301)[1:-1]
    c = b.chord_mm
    R = (0.25 * c * c + (b.kappa * c) ** 2) / (2.0 * b.kappa * c)
    from scan_core.bead_geometry import TANGENT_POINTS
    th = 2.0 * math.atan(2.0 * b.kappa)
    bound = R * (1.0 / math.cos(th / TANGENT_POINTS) - 1.0) / math.cos(th)      # off the arc along its normal, read vertically
    from scan_core.weld_scene import surface_mm
    gap = [surface_mm({"b": H}, x) - float(b.top(x)) for x in xs]
    assert min(gap) >= -1e-9 and max(gap) <= bound + 1e-9
    assert np.any(np.all(np.isclose(n, [0.0, 1.0], atol=1e-12), axis=1))


def test_a_post_weld_profile_gives_back_the_crown_ratio():
    """The calibration hook (no fit now): from the section before and after a pass, the toes, the chord and the crown ratio come back
    (the arc model's kappa_p) - what KIRO's per-pass post-weld scans will feed the bead model."""
    prof = bare_profile(K20, 30.0, 0.1)
    for k in (0.2, 0.3, 0.4):
        b, after = place_bead(prof, (0.0, 0.0), 30.0, k)
        xs = np.linspace(-12.0, 12.0, 4801)
        r = crown_ratio_from_profiles(xs, prof.at(xs), after.at(xs))
        assert r["crown_ratio"] == pytest.approx(k, abs=0.005) and r["chord_mm"] == pytest.approx(b.chord_mm, abs=0.02)
        assert r["area_mm2"] == pytest.approx(30.0, rel=2e-3)


def test_bead_section_record():
    b = BeadSection(1.0, 0.3, (0.0, 1.0), (10.0, 1.0), "pool")
    d = b.to_dict()
    assert d["crown_xz_mm"] == pytest.approx([5.0, 4.0]) and d["chord_mm"] == 10.0 and d["phase"] == "pool"
