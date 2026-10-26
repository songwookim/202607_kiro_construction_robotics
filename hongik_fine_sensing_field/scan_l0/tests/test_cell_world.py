"""scan_l0.cell_world.CellWorld — one world for the 1G core (interface design §2.15): the envelope reference and the MoveIt objects come
from the same solids (they agree), every object id is under 'hongik/', the floor stays C short of the robot's fixed body (no ladder,
no arm-state shaping), and a contact is named (which link inside which object). No robot, no ROS."""
from __future__ import annotations

import math

import numpy as np
import pytest

from scan_l0.cell_world import FLOOR_ID, NAMESPACE, SPECIMEN_ID, CellWorld, contact_pairs

UP = np.array([0.0, 0.0, 1.0])
T, B, W, L_D, C, REACH = 0.010, 0.006, 0.600, 0.380, 0.050, 0.500
KW = dict(up=UP, thickness_m=T, backing_m=B, width_m=W, design_length_m=L_D, clearance_m=C, reach_m=REACH)


def _box_points(obj) -> tuple:
    """(T, half sizes) of a MoveIt object dict."""
    return np.asarray(obj["T"], dtype=float), 0.5 * np.asarray(obj["size"], dtype=float)


def _inside(obj, P) -> np.ndarray:
    Tb, h = _box_points(obj)
    loc = (np.asarray(P, dtype=float).reshape(-1, 3) - Tb[:3, 3]) @ Tb[:3, :3]
    return np.all(np.abs(loc) <= h[None, :] + 1e-12, axis=1)


def _objs(w) -> dict:
    return {o["id"]: o for o in w.moveit_objects()}


def test_prior_from_level_rough_points_matches_the_design_rules():
    A, Bp = np.array([0.69, 0.0, -0.03]), np.array([1.01, 0.0, -0.03])
    t_hit = T                                                     # root gap > 0: the rough point may be at the plate bottom
    w = CellWorld.from_rough_points(A, Bp, hit_face_m=t_hit, **KW)
    env = w.envelope()
    assert w.version == 1 and env["plate_top_m"] == pytest.approx(-0.03 + t_hit)                       # higher hit + t_hit
    assert np.allclose(env["plate_normal"], UP)
    assert env["floor_m"] == pytest.approx(-0.03 + t_hit - T - B)                                        # lower hit + t_hit - t - b
    sp = w.specimen()
    ext = L_D - 0.32                                                                                       # design length not spanned
    assert sp.size[0] == pytest.approx(0.32 + 2 * ext) and sp.size[1] == pytest.approx(W) and sp.size[2] == pytest.approx(t_hit + T + B)
    # the rough points are inside the specimen solid (whether they are at the top or at the root)
    assert np.all(sp.contains(np.array([A, Bp])))


@pytest.mark.parametrize("measured", [False, True])
def test_envelope_and_moveit_objects_agree(measured):
    """The specimen object's top face is the envelope's plate plane grown by C, the floor object's top is the envelope's support
    height grown by C; MoveIt objects are exactly the solids grown by C."""
    th = math.radians(8.0)                                        # a plate sloped 8 deg along the seam, rolled 4 deg
    u = np.array([math.cos(th), 0.0, math.sin(th)])
    roll = math.radians(4.0)
    n = np.array([-math.sin(th), 0.0, math.cos(th)])
    n = math.cos(roll) * n + math.sin(roll) * np.cross(u, n)
    A = np.array([0.7, 0.05, -0.02])
    Bp = A + u * 0.33
    w = CellWorld.from_rough_points(A, Bp, hit_face_m=T, **KW)
    if measured:
        w.update_measured(top_point=A + n * 0.0005, sigma_top_m=0.0004, normal=n, k=3.0, root_lo=A - u * 0.03 - n * T,
                          root_hi=Bp + u * 0.02 - n * T, ends={"S": {}, "E": {}}, why="test")
        assert w.version == 2 and w.plate_sigma_m == pytest.approx(3.0 * 0.0004)
    env = w.envelope()
    objs = _objs(w)
    assert set(objs) == {SPECIMEN_ID, FLOOR_ID} and all(i.startswith(NAMESPACE) for i in objs)
    sp = w.specimen()
    p, nn = env["plate_point_m"], env["plate_normal"]
    # solid top face on the envelope plane
    top_c = sp.center + sp.R[:, 2] * (0.5 * sp.size[2])
    assert abs(float((top_c - p) @ nn)) < 1e-12 and np.allclose(sp.R[:, 2], nn)
    # MoveIt specimen object = solid grown by C (same pose)
    Tb, h = _box_points(objs[SPECIMEN_ID])
    assert np.allclose(Tb, sp.T()) and np.allclose(2 * h, sp.size + 2 * C)
    # points on the plate plane over the seam: inside the object; C (and a hair) above the plane: outside — the envelope's plane is
    # the object's face minus C
    a, b = w.line
    for s_ in np.linspace(0.0, 1.0, 7):
        q = a + (b - a) * s_
        assert _inside(objs[SPECIMEN_ID], q)[0] and _inside(objs[SPECIMEN_ID], q + nn * (C - 1e-6))[0]
        assert not _inside(objs[SPECIMEN_ID], q + nn * (C + 1e-6))[0]
    # floor object: horizontal, its top = support height + C; the support is below every corner of the specimen's top face
    Tf, hf = _box_points(objs[FLOOR_ID])
    assert np.allclose(Tf[:3, 2], UP) and float(Tf[:3, 3] @ UP) + hf[2] == pytest.approx(env["floor_m"] + C)
    assert env["floor_m"] <= min(float(c @ UP) for c in w._top_corners()) - T - B + 1e-12
    # the floor covers the specimen footprint + reach
    for c in sp.corners():
        for dx in (np.zeros(3), np.array([REACH - 1e-6, 0.0, 0.0]), np.array([0.0, -(REACH - 1e-6), 0.0])):
            foot = c + dx - UP * float(c @ UP) + UP * env["floor_m"]       # on the support plane under the footprint (+ reach)
            assert _inside(objs[FLOOR_ID], foot)[0]
    # the plate top used by the 1G plate-top rule = the highest point of the top face along the seam
    assert env["plate_top_m"] == pytest.approx(max(float(a @ UP), float(b @ UP), float(p @ UP)))
    d = w.to_dict()
    assert d["version"] == w.version and [o["id"] for o in d["objects"]] == [SPECIMEN_ID, FLOOR_ID]
    assert d["planes"][0]["id"] == NAMESPACE + "plate_top" and np.allclose(d["planes"][0]["normal"], nn)
    assert d["envelope"]["floor_m"] == pytest.approx(env["floor_m"]) and all(o["padding_m"] == C for o in d["objects"])


def test_floor_stops_C_short_of_the_robot_fixed_body_without_using_the_arm_state():
    """Fake-arm layout (test_m): the conservative floor is 35 mm below the robot base mounting plane, so the floor object (grown by C)
    would reach into link0. Its robot side stops C short of link0's collision box — the same world whatever the arm does."""
    A, Bp = np.array([0.69, 0.015, -0.03]), np.array([1.01, -0.010, -0.03])
    w = CellWorld.from_rough_points(A, Bp, hit_face_m=T, **KW)
    link0 = {"link": "link0", "lo": np.array([-0.1225, -0.1225, 0.0]), "hi": np.array([0.1225, 0.1272, 0.1065])}
    no_cut = _objs(w)[FLOOR_ID]
    corners = np.array([[x, y, z] for x in (link0["lo"][0], link0["hi"][0]) for y in (link0["lo"][1], link0["hi"][1])
                        for z in (link0["lo"][2], link0["hi"][2])])
    edge = np.array([[link0["hi"][0], 0.0, 0.005]])                       # the near edge of link0's box, inside the floor's height band
    assert _inside(no_cut, edge)[0]                                       # without the fixed body: link0 would be inside the floor
    w.set_fixed_bodies([link0, {"link": "base_plate", "lo": None, "hi": None, "why": "mesh not readable here"}])
    fl = _objs(w)[FLOOR_ID]
    Tf, hf = _box_points(fl)
    a = Tf[:3, :3][:, 0]                                                  # base → specimen (horizontal)
    near_face = float((Tf[:3, 3] - a * hf[0]) @ a)
    assert near_face == pytest.approx(float(np.max(corners @ a)) + C)    # C short of link0, not touching
    assert not np.any(_inside(fl, corners)) and not _inside(fl, edge)[0]
    d = w.to_dict()
    assert any("link0" in n_ for n_ in d["floor_cut"]) and any("base_plate" in n_ and "unknown" in n_ for n_ in d["floor_cut"])
    # the specimen object is unchanged by the fixed body and the far side of the floor too
    assert np.allclose(_objs(w)[SPECIMEN_ID]["T"], w.specimen().T())
    assert float((no_cut["T"][:3, 3] + no_cut["T"][:3, 0] * 0.5 * no_cut["size"][0]) @ a) == pytest.approx(
        float((Tf[:3, 3] + a * hf[0]) @ a))


def test_floor_stops_C_short_of_a_fixed_body_beyond_the_specimen_on_the_far_side():
    """KIRO layout (stand-in 2026-10-12): the robot model holds a test bed beyond the specimen (World y < -1.70 m, the specimen in front of
    it). The floor's far face stops C short of that body - the robot-side face stays at the robot's torso - so the floor still lies
    under the seam (before: the robot-side rule moved the near face past the test bed and the floor object vanished)."""
    A, Bp = np.array([0.062, -1.27, 0.703]), np.array([0.042, -1.59, 0.703])
    w = CellWorld.from_rough_points(A, Bp, hit_face_m=T, **KW)
    torso = {"link": "robot_base_link", "lo": np.array([-0.40, -0.80, 0.33]), "hi": np.array([0.40, -0.20, 1.40])}
    testbed = {"link": "testbed_link", "lo": np.array([-0.444, -2.20, 0.0]), "hi": np.array([0.456, -1.70, 2.11])}
    w.set_fixed_bodies([torso, testbed])
    fl = _objs(w)[FLOOR_ID]
    Tf, hf = _box_points(fl)
    a = Tf[:3, :3][:, 0]                                                  # base -> specimen (horizontal): about -y here
    assert a @ np.array([0.0, -1.0, 0.0]) > 0.99
    near, far = float((Tf[:3, 3] - a * hf[0]) @ a), float((Tf[:3, 3] + a * hf[0]) @ a)

    def corners(body):
        return np.array([[x, y, z] for x in (body["lo"][0], body["hi"][0]) for y in (body["lo"][1], body["hi"][1])
                         for z in (body["lo"][2], body["hi"][2])])
    assert near == pytest.approx(float(np.max(corners(torso) @ a)) + C, abs=1e-9)      # C beyond the torso (robot side)
    assert far == pytest.approx(float(np.min(corners(testbed) @ a)) - C, abs=1e-9)     # C before the test bed (far side)
    for body in (torso, testbed):                                         # neither body is inside the floor object
        assert not np.any(_inside(fl, corners(body)))
    assert near < float(A @ a) < far and near < float(Bp @ a) < far           # the floor lies under the seam between the two points
    cut = w.to_dict()["floor_cut"]
    assert any(n_.startswith("far side stops 50 mm short of testbed_link") for n_ in cut)
    assert any(n_.startswith("robot side stops 50 mm short of robot_base_link") for n_ in cut)


def test_floor_stops_C_short_of_a_fixed_body_beside_the_specimen_on_a_lateral_side():
    """KIRO stand-in layout 2 (2026-10-13): the specimen front-right of the robot (seam x -0.95, y -1.16 .. -1.54) and the test bed of
    the robot model off to the side (x -0.444 .. 0.456, y < -1.70): along the robot-specimen direction its box overlaps the specimen's
    range, so neither the robot side nor the far side keeps the floor under the specimen - the lateral face stops C short of it."""
    A, Bp = np.array([-0.95, -1.19, 0.90]), np.array([-0.95, -1.51, 0.90])
    w = CellWorld.from_rough_points(A, Bp, hit_face_m=T, **KW)
    testbed = {"link": "testbed_link", "lo": np.array([-0.444, -2.20, 0.0]), "hi": np.array([0.456, -1.70, 2.11])}
    w.set_fixed_bodies([testbed])
    fl = _objs(w)[FLOOR_ID]
    corners = np.array([[x, y, z] for x in (testbed["lo"][0], testbed["hi"][0]) for y in (testbed["lo"][1], testbed["hi"][1])
                        for z in (testbed["lo"][2], testbed["hi"][2])])
    assert not np.any(_inside(fl, corners))
    Tf, hf = _box_points(fl)
    for P_ in (A, Bp):                                                    # still under the seam
        assert _inside(fl, (P_ - UP * 0.04)[None, :])[0]
    cut = w.to_dict()["floor_cut"]
    assert len(cut) == 1 and cut[0].startswith("lateral side") and "testbed_link" in cut[0], cut


def test_floor_under_the_robot_when_it_is_far_below_the_base():
    """Lab layout (floor ~150 mm below the base): the fixed body is not within C of the floor's height band — no cut."""
    A, Bp = np.array([0.55, 0.0, -0.13]), np.array([0.93, 0.0, -0.13])
    w = CellWorld.from_rough_points(A, Bp, hit_face_m=T, **KW)
    before = _objs(w)[FLOOR_ID]
    w.set_fixed_bodies([{"link": "link0", "lo": np.array([-0.1225, -0.1225, 0.0]), "hi": np.array([0.1225, 0.1272, 0.1065])}])
    after = _objs(w)[FLOOR_ID]
    assert np.allclose(before["T"], after["T"]) and np.allclose(before["size"], after["size"]) and w.to_dict()["floor_cut"] == []


def test_contacts_are_named_link_and_object():
    w = CellWorld.from_rough_points([0.7, 0.0, 0.0], [1.0, 0.0, 0.0], hit_face_m=T, **KW)
    txt = w.explain([{"link": "eoat_base_link", "object": SPECIMEN_ID, "depth_m": 0.0123, "kind": "world"},
                     {"link": "link6", "object": FLOOR_ID, "depth_m": 0.0, "kind": "world"},
                     {"link": "link3", "object": "link5", "kind": "self"}])
    assert txt.startswith("eoat_base_link is inside hongik/specimen (the specimen (") and "12.3 mm deep" in txt
    assert "link6 is inside hongik/floor (the support below the specimen (top " in txt and "link3 touches link5 (self-collision)" in txt
    assert contact_pairs("point 0/0 invalid (contacts ['link6-hongik/floor', 'hongik/specimen-eoat_base_link', 'link2-link4'])") == [
        ("link6", FLOOR_ID), ("eoat_base_link", SPECIMEN_ID), ("link2", "link4")]
    assert contact_pairs("eoat_base_link-hongik/specimen") == [] and contact_pairs(None) == []


def test_measured_update_moves_the_ends_and_versions():
    A, Bp = np.array([0.69, 0.0, -0.03]), np.array([1.01, 0.0, -0.03])
    w = CellWorld.from_rough_points(A, Bp, hit_face_m=T, **KW)
    L0 = w.specimen().size[0]
    w.update_measured(top_point=[0.85, 0.0, -0.05], sigma_top_m=0.0002, normal=[0.0, 0.0, -1.0], k=3.0, root_lo=[0.66, 0.0, -0.06],
                      root_hi=[1.04, 0.0, -0.06], why="estimate 1")
    sp = w.specimen()
    assert w.version == 2 and np.allclose(w.plate_normal, UP)                  # normal turned toward up
    assert sp.size[0] == pytest.approx(0.38) and sp.size[0] < L0 and w.plate_top_m == pytest.approx(-0.05 + 3 * 0.0002)
    assert w.floor_m == pytest.approx(-0.05 + 3 * 0.0002 - T - B) and "estimate 1" in w.source
    assert w.summary().startswith("world v2 (measured: plate plane + k sigma_top")
    # line and normal estimated separately (not exactly perpendicular, lab g1c_01: u.n = 3e-5): the world uses the section frame of
    # spec §9 (u = the line, n made perpendicular to it) - the box is a rigid frame and its top face is the envelope plane
    from scan_core.seam_estimate import _section_basis
    lo, hi = np.array([0.66, 0.0, -0.06]), np.array([1.04, 0.002, -0.0605])
    n_est = np.array([3e-3, -1e-3, 1.0])
    w.update_measured(top_point=[0.85, 0.0, -0.05], sigma_top_m=0.0002, normal=n_est, k=3.0, root_lo=lo, root_hi=hi)
    u_b, n_b, w_b = _section_basis(hi - lo, n_est)
    sp = w.specimen()
    assert np.allclose(w.plate_normal, n_b, atol=1e-12) and np.allclose(sp.R, np.column_stack([u_b, w_b, n_b]), atol=1e-12)
    assert np.allclose(sp.R.T @ sp.R, np.eye(3), atol=1e-12) and abs(float(w.plate_normal @ (hi - lo))) < 1e-15


def test_plate_widths_each_plate_and_the_gap_placed_about_the_seam():
    """Specimen box across the seam = left plate + root gap + right plate (design widths of each plate): before the bevel side is
    resolved the wider plate on both sides; once known, each plate on its side (box centre offset from the seam)."""
    A, Bp = np.array([0.69, 0.0, -0.03]), np.array([1.01, 0.0, -0.03])          # S -> E along +x, so +w = +y (left looking S->E)
    kw = {k: v for k, v in KW.items() if k != "width_m"}
    w = CellWorld.from_rough_points(A, Bp, hit_face_m=T, plate_widths_m=(0.30, 0.20), gap_m=0.005, **kw)
    sp = w.specimen()
    assert sp.size[1] == pytest.approx(2 * (0.0025 + 0.30))                     # side unknown: the wider plate on both sides
    assert float(sp.center[1]) == pytest.approx(0.0, abs=1e-12)
    v0 = w.version
    w.set_plate_sides(+1)                                                       # the left (design) plate on +w
    sp = w.specimen()
    assert w.version == v0 + 1
    assert sp.size[1] == pytest.approx(0.30 + 0.005 + 0.20)
    lo, hi = float(sp.center[1]) - 0.5 * sp.size[1], float(sp.center[1]) + 0.5 * sp.size[1]
    assert hi == pytest.approx(0.0025 + 0.30) and lo == pytest.approx(-(0.0025 + 0.20))
    w.set_plate_sides(-1)
    sp = w.specimen()
    lo, hi = float(sp.center[1]) - 0.5 * sp.size[1], float(sp.center[1]) + 0.5 * sp.size[1]
    assert hi == pytest.approx(0.0025 + 0.20) and lo == pytest.approx(-(0.0025 + 0.30))
    d = w.to_dict()["planes"][0]["extent"]
    assert d["plate_widths_m"] == [0.30, 0.20] and d["left_plate_side_w"] == -1


def test_specimen_plate_widths_input_and_the_deprecated_total():
    from scan_core.specimen import GrooveSpec
    g = GrooveSpec.from_dict({"plate_width_mm": 600.0})                         # old key: the total of both plates
    assert g.plate_widths_mm == [300.0, 300.0] and g.plate_width_mm == 600.0
    assert any("deprecated" in n_ for n_ in g.deprecation_notes())
    g = GrooveSpec.from_dict({"plate_widths_mm": [300.0, 250.0], "root_gap_mm": 5.0})
    assert g.plate_widths_mm == [300.0, 250.0] and g.plate_width_mm == 550.0 and not g.deprecation_notes()
    assert g.specimen_width_mm() == pytest.approx(555.0)
    assert GrooveSpec.from_dict({"plate_widths_mm": 280.0}).plate_widths_mm == [280.0, 280.0]      # one value = both plates
    with pytest.raises(ValueError):
        GrooveSpec(plate_widths_mm=[300.0, -1.0])


def test_the_space_under_the_specimen_is_closed_down_to_the_arms_reach():
    """User 2026-10-14 (KIRO field): no motion under the specimen. With the platform's reach bound the world has a solid under the
    specimen's footprint from the support plane down to the arm's lowest reach (grown by C across and on top), flagged so the
    platform allows only the robot's fixed links (e.g. a test bed) in it; without a reach bound there is none (said, not invented)."""
    from scan_l0.cell_world import UNDER_ID
    A, Bp = np.array([0.69, 0.0, -0.03]), np.array([1.01, 0.0, -0.03])
    w = CellWorld.from_rough_points(A, Bp, hit_face_m=T, **KW)
    assert UNDER_ID not in _objs(w) and w.under() is None                  # reach unknown: no under solid
    v0 = w.version
    w.set_reach_below(2.0)
    assert w.version == v0                       # platform knowledge (as the fixed bodies), not a new world version
    o = _objs(w)[UNDER_ID]
    assert o.get("allow_fixed_links") is True and o["id"].startswith(NAMESPACE)
    sp = w.specimen()
    floor = w.envelope()["floor_m"]
    # every point under the specimen's top face, from just below the support plane to the arm's lowest reach, is inside
    a, b = w.line
    wv = np.cross(UP, b - a); wv /= np.linalg.norm(wv)
    for f_u in (0.0, 0.5, 1.0):
        for f_w in (-0.49, 0.0, 0.49):
            p = a + (b - a) * f_u + wv * (f_w * W)
            for depth in (0.001, 0.3, 1.9):
                q = p.copy(); q[2] = floor - depth
                assert _inside(o, q)[0], (f_u, f_w, depth)
    # grown by C across the seam and on top, not beyond the reach at the bottom
    assert _inside(o, a + wv * (0.5 * W + 0.9 * C) + UP * (floor - a[2] - 0.1))[0]
    assert not _inside(o, a + wv * (0.5 * W + 1.1 * C) + UP * (floor - a[2] - 0.1))[0]
    assert _inside(o, np.array([0.85, 0.0, floor + 0.9 * C]))[0] and not _inside(o, np.array([0.85, 0.0, floor + 1.1 * C]))[0]
    assert not _inside(o, np.array([0.85, 0.0, -2.0 - 0.01]))[0]           # base origin 0 - reach 2.0
    assert "no part of the arm goes there" in w.describe(UNDER_ID)
    assert w.to_dict()["reach_below_m"] == 2.0 and UNDER_ID in [ob["id"] for ob in w.to_dict()["objects"]]
    # a reach that does not get below the support: nothing to close
    w.set_reach_below(0.0)
    assert w.under() is None and UNDER_ID not in _objs(w)
    assert sp.size[1] == pytest.approx(W)
