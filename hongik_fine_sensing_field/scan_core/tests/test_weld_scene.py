"""weld_scene: the specimen as the torch meets it at the weld poses (user 2026-10-13) - plates on their own with the groove open,
backing, the beads deposited before each pass (the plan's own convex beads, 2026-10-13 night); mitered growth by the required clearance;
MoveIt meshes lofted through the stations."""
import numpy as np
import pytest

from scan_core.bead_fill import wall_x_mm
from scan_core.specimen import GrooveSpec
from scan_core.weld_plan import WeldProcess, weld_plan_from_features
from scan_core.weld_scene import (WELD_NAMESPACE, grow, pass_solids, plate_polys, scene_meshes, solid_planes, state_polys,
                                  surface_mm)

from test_weld_plan import SPEC, extent, synth_feats


def _ccw(P):
    x, z = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def test_plates_are_separate_with_the_groove_open_between_them():
    sp = GrooveSpec(kind="single_bevel", thickness_mm=10.0, bevel_angle_deg=30.0, bevel_side="left", root_gap_mm=5.0, root_face_mm=1.0,
                    hi_lo_mm=0.4)
    P = plate_polys(sp, (150.0, 140.0), 10.0)
    assert set(P) == {"left_plate", "right_plate", "backing"}
    L, R, B = P["left_plate"], P["right_plate"], P["backing"]
    assert _ccw(L) > 0 and _ccw(R) > 0 and _ccw(B) > 0                          # counter-clockwise (outward normals to the right)
    xl_t = float(wall_x_mm(sp, 10.0)[0])
    assert L.max(axis=0)[1] == 10.0 and L[:, 0].min() == pytest.approx(-2.5 - 150.0) and L[3] == pytest.approx([xl_t, 10.0])
    assert R.max(axis=0)[1] == pytest.approx(10.4) and R[:, 0].max() == pytest.approx(2.5 + 140.0)       # right top at t + hi-lo
    assert B[:, 1].min() == -10.0 and B[:, 1].max() == 0.0
    # the gap is open: the surface over the gap centre is the backing top, over the plates their tops, on the bevel its face
    assert surface_mm(P, 0.0) == pytest.approx(0.0)
    assert surface_mm(P, -50.0) == pytest.approx(10.0) and surface_mm(P, 50.0) == pytest.approx(10.4)
    x = 0.5 * (xl_t - 2.5)
    assert surface_mm(P, x) == pytest.approx(1.0 + (-2.5 - x) / np.tan(np.radians(30.0)))
    assert np.isnan(surface_mm({"left_plate": L}, 200.0))
    # the deposits before a pass: the earlier beads' convex solids (bead_geometry) - the surface over the gap is the bead's top
    from scan_core.bead_geometry import bare_profile, place_bead
    b, after = place_bead(bare_profile(sp, 20.0, 0.1), (-2.5, 1.0), 30.0, 0.3)
    S = state_polys(sp, (150.0, 140.0), 10.0, beads=[("bead_1", b)])
    assert set(S) == {"left_plate", "right_plate", "backing", "bead_1"} and _ccw(S["bead_1"]) > 0
    for x in (b.apex[0], -2.0, 1.0):
        assert surface_mm(S, x) == pytest.approx(float(b.top(x)), abs=0.02) and surface_mm(S, x) >= float(after.at(x)) - 1e-9


def test_growth_moves_every_face_out_by_r_along_its_normal():
    P = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [6.0, 8.0], [0.0, 8.0]])
    n, b, M = solid_planes(P)
    G = grow(P, 1.5)
    n2, b2, _ = solid_planes(G)
    assert np.allclose(n2, n) and np.allclose(b2, b + 1.5)                        # each face moved out by r, same normals
    assert np.allclose(np.einsum("kj,kj->k", np.roll(n, 1, axis=0), M), 1.0) and np.allclose(np.einsum("kj,kj->k", n, M), 1.0)
    # a zero-length edge (fixed topology) and a flat vertex are handled
    Q = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    n3, b3, M3 = solid_planes(Q)
    assert np.all(np.isfinite(n3)) and np.all(np.isfinite(M3))
    Gq = grow(Q, 2.0)
    assert Gq[:, 1].min() == pytest.approx(-2.0) and Gq[:, 0].max() == pytest.approx(12.0)


@pytest.fixture(scope="module")
def plan():
    f = synth_feats()
    return weld_plan_from_features(f, extent(f), SPEC, WeldProcess())


def test_pass_scene_holds_the_beads_deposited_before_the_pass(plan):
    """The scene before pass k = the plates, the backing and the plan's own beads of passes 1 .. k-1 (their convex solids: the same
    beads the plan's heights and aims come from - one source); the wire end of each pass is on that scene's surface."""
    names = [[so.name for so in pass_solids(plan, wp)] for wp in plan.passes]
    assert names[0] == ["left_plate", "right_plate", "backing"]                  # root: plates + backing, nothing deposited yet
    for k, n in enumerate(names[1:], start=2):
        assert n == ["left_plate", "right_plate", "backing"] + [f"bead_{q}" for q in range(1, k)]
    for wp in plan.passes:
        sol = {so.name: so for so in pass_solids(plan, wp)}
        assert all(so.poly.shape[0] == len(plan.stations) for so in sol.values())
        for q in plan.passes[:wp.k - 1]:                          # the scene's beads = the plan's beads (hulls, padded with repeats)
            by_s = {float(s): i for i, s in enumerate(q.s_mm)}
            for i, st in list(enumerate(plan.stations))[::9]:
                H = q.beads[by_s[float(st.s_mm)]].hull
                P = sol[f"bead_{q.k}"].poly[i]
                assert np.allclose(P[:len(H)], H) and np.allclose(P[len(H):], H[-1])
        for i, st in enumerate(plan.stations):                    # the wire end (aim point) is on the surface of this scene
            j = int(np.flatnonzero(wp.s_mm == st.s_mm)[0])
            x, z = wp.aim_xz_mm[j]
            top = surface_mm({k: so.poly[i] for k, so in sol.items()}, x)
            if wp.k == 2:                                         # fill 1 at the root's toe on the square wall: the corner of the scene
                prev = plan.passes[0].beads[int(np.flatnonzero(plan.passes[0].s_mm == st.s_mm)[0])]
                assert (x, z) == pytest.approx(prev.toe_r if x > 0 else prev.toe_l)
            else:                                                 # on the earlier beads' tops (their tangent polygons lie within 0.05 mm)
                assert top == pytest.approx(z, abs=0.05) and top >= z - 1e-9


def test_scene_meshes_are_closed_lofts_in_the_base_frame(plan):
    wp = plan.passes[1]
    sol = pass_solids(plan, wp)
    ms = scene_meshes(plan, sol, 2.0)
    assert [m["id"] for m in ms] == [WELD_NAMESPACE + so.name for so in sol]
    for m, so in zip(ms, sol):
        V, T = m["vertices_m"], m["triangles"]
        E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
        _, cnt = np.unique(E, axis=0, return_counts=True)
        assert np.all(cnt == 2)                                                   # closed (every edge in two triangles)
        st0 = plan.stations[0]
        loc = (V - st0.origin) @ np.column_stack([st0.w, st0.u, st0.n]) * 1e3
        assert loc[:, 1].min() == pytest.approx(-2.0, abs=1e-6)                  # the end moved out by r along the seam
        k = so.poly.shape[1]
        G = grow(so.poly[0], 2.0)
        assert np.allclose(loc[k:2 * k][:, [0, 2]], G, atol=1e-6)                # the first station's grown polygon
