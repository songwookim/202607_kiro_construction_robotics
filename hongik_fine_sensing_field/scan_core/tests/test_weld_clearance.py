"""weld_clearance: the tool <-> specimen clearance at the weld poses (user 2026-10-13) - exact for a triangle mesh against convex
section prisms grown with mitered faces (checked against dense sampling), the URDF tool geometry rigid with the TCP."""
import math

import numpy as np
import pytest

from scan_core.weld_clearance import PassGeometry, _g, clearance, tool_body_from_triangles, tool_body_from_urdf
from scan_core.weld_scene import solid_planes

PENT = np.array([[-60.0, 0.0], [-2.5, 0.0], [-2.5, 1.0], [-8.3, 10.0], [-60.0, 10.0]])      # a bevelled plate section [mm]
BOX = np.array([[2.5, 0.0], [60.0, 0.0], [60.0, 10.4], [2.5, 10.4]])


def geometry(polys, s_lo=0.0, s_hi=200.0, stations=(0.0, 100.0, 200.0)) -> PassGeometry:
    S = np.asarray(stations, dtype=float)
    planes = []
    for P in polys:
        Pn = np.repeat(np.asarray(P, dtype=float)[None], len(S), axis=0)
        Nrm, b, M = solid_planes(Pn)
        planes.append((Nrm, b, M, Pn))
    return PassGeometry(names=[f"s{i}" for i in range(len(polys))], planes=planes, ylo=s_lo - S, yhi=s_hi - S, A=np.eye(3),
                        origins=np.zeros((len(S), 3)), s_mm=S)


def brute(tris_mm, geo, P_sec, n=40) -> float:
    """min g over dense barycentric samples of every triangle at every station."""
    u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
    m = (u + v) <= 1.0 + 1e-12
    u, v = u[m], v[m]
    best = math.inf
    for T in tris_mm:
        X = T[0] + u[:, None] * (T[1] - T[0]) + v[:, None] * (T[2] - T[0])
        for i in range(len(geo.s_mm)):
            Q = X + P_sec[i]
            for Nrm, b, _M, _C in geo.planes:
                best = min(best, float(np.min(_g(Q, Nrm[i], b[i], float(geo.ylo[i]), float(geo.yhi[i])))))
    return best


def test_exact_against_dense_sampling_over_ridges_and_plate_ends():
    rng = np.random.default_rng(7)
    geo = geometry([PENT, BOX])
    n_clear = n_in = 0
    for trial in range(60):
        c = np.array([rng.uniform(-20, 15), rng.uniform(-30, 230), rng.uniform(8, 22)])
        tris = c + rng.normal(0.0, 12.0, size=(3, 3, 3))                          # three triangles of a few cm
        body = tool_body_from_triangles(tris * 1e-3, tcp_frame="tcp", source="test")
        P_sec = np.zeros((3, 3))
        P_sec[:, 1] = -geo.s_mm                                                     # the same body seen from each station
        c = clearance(body, geo, np.eye(3), P_sec)
        got = c["clearance_mm"]
        ref = brute(tris, geo, P_sec)
        L = max(np.linalg.norm(T[i] - T[j]) for T in tris for i in range(3) for j in range(3))
        if ref > L / 40 * 1.5:                                                      # clear: the exact first contact
            assert not c["intersects"] and got <= ref + 1e-6, (trial, got, ref)   # never above what any point shows
            assert ref - got <= L / 40 * 1.5 + 1e-6, (trial, got, ref)             # and the sampling gap bounds the difference
            n_clear += 1
        elif ref < 0.0:                                                             # inside: found, value <= 0
            assert c["intersects"] and got <= 0.0, (trial, got, ref)
            n_in += 1
        for thr in (0.0, 0.3, max(got, 0.0) + 0.5, max(got - 0.5, 0.0)):              # the feasibility test agrees with the value
            if abs(got - thr) > 1e-6:
                assert clearance(body, geo, np.eye(3), P_sec, threshold=thr)["below"] == (got < thr), (trial, got, thr)
    assert n_clear >= 10 and n_in >= 5, (n_clear, n_in)


def test_a_face_over_the_groove_edge_touches_although_no_vertex_does():
    """A flat triangle sagging over the bevelled plate's top edge (the ridge at x = -8.3, z = 10): its vertices are 2 mm above the
    plate top or over the open groove, the face itself is 1 mm below the ridge - the clearance is negative (the ridge pokes in)."""
    geo = geometry([PENT])
    tri = np.array([[-20.0, 80.0, 12.0], [-20.0, 120.0, 12.0], [0.0, 100.0, 4.0]])  # from 2 mm above the top down into the groove
    body = tool_body_from_triangles(tri[None] * 1e-3, tcp_frame="tcp", source="test")
    P_sec = np.array([[0.0, -s, 0.0] for s in geo.s_mm])
    c = clearance(body, geo, np.eye(3), P_sec)
    z_at_ridge = 12.0 + (4.0 - 12.0) * (-8.3 + 20.0) / 20.0                      # the face height over the ridge (y = 100 line)
    assert z_at_ridge < 10.0 and c["clearance_mm"] <= 0.0 and c["intersects"], c
    assert clearance(body, geo, np.eye(3), P_sec, threshold=0.0)["below"]          # the feasibility test sees it too
    # every vertex alone is outside
    for v in tri:
        assert all(float(_g(v[None] + P_sec[i], geo.planes[0][0][i], geo.planes[0][1][i], geo.ylo[i], geo.yhi[i])[0]) > 0 for i in range(3))


def test_beyond_the_plate_end_the_end_face_counts():
    """A tool part beyond the S end, below the plate top: clear by its distance to the end face (grown faces: mitered)."""
    geo = geometry([BOX])
    tri = np.array([[20.0, -6.0, 5.0], [30.0, -6.0, 5.0], [25.0, -15.0, 5.0]])     # 6 mm beyond the end, 5 mm below the top
    body = tool_body_from_triangles(tri[None] * 1e-3, tcp_frame="tcp", source="test")
    P_sec = np.array([[0.0, -s, 0.0] for s in geo.s_mm])
    assert clearance(body, geo, np.eye(3), P_sec)["clearance_mm"] == pytest.approx(6.0, abs=1e-9)


URDF = """<robot name="t">
  <link name="link5"><collision><geometry><box size="1 1 1"/></geometry></collision></link>
  <link name="link6"><collision><origin xyz="0 0 0.05" rpy="0 0 0"/><geometry><box size="0.02 0.02 0.02"/></geometry></collision></link>
  <link name="flange"/>
  <link name="tool"><collision><geometry><mesh filename="{mesh}" scale="0.001 0.001 0.001"/></geometry></collision></link>
  <link name="tcp"/>
  <joint name="j6" type="revolute"><parent link="link5"/><child link="link6"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
  <joint name="jf" type="fixed"><parent link="link6"/><child link="flange"/><origin xyz="0 0 0.1" rpy="0 0 1.5707963267948966"/></joint>
  <joint name="jt" type="fixed"><parent link="flange"/><child link="tool"/><origin xyz="0 0 0"/></joint>
  <joint name="jc" type="fixed"><parent link="tool"/><child link="tcp"/><origin xyz="0.01 0 0.3" rpy="0 0.5235987755982988 0"/></joint>
</robot>"""


def test_tool_body_is_the_collision_geometry_rigid_with_the_tcp(tmp_path):
    import struct
    tri = np.array([[[0, 0, 300], [10, 0, 300], [0, 10, 300]]], dtype="<f4")        # mm, one triangle at the TCP height
    rec = np.zeros(1, dtype=[("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")])
    rec["v"] = tri.reshape(1, 9)
    p = tmp_path / "tool.stl"
    p.write_bytes(b"\0" * 80 + struct.pack("<I", 1) + rec.tobytes())
    tb = tool_body_from_urdf(URDF.format(mesh=str(p)), "tcp")
    assert tb.links == ("link6", "tool")                                            # link5 moves (revolute): not rigid with the TCP
    assert len(tb.F) == 12 + 1
    # the tool triangle in the TCP frame: tcp = tool + (0.01, 0, 0.3) rotated 30 deg about y
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [0, math.radians(30), 0]).as_matrix()
    want = (tri[0].astype(float) * 1e-3 - [0.01, 0, 0.3]) @ R
    got = {tuple(np.round(v, 9)) for v in tb.V}
    assert all(tuple(np.round(w, 9)) in got for w in want)
    # a TCP the URDF does not carry: the tool rigid with the flange, expressed in the TCP frame through the given flange <- TCP
    from scipy.spatial.transform import Rotation as Rot
    T_ft = np.eye(4)
    T_ft[:3, :3] = Rot.from_euler("xyz", [0, math.radians(30), 0]).as_matrix()
    T_ft[:3, 3] = [0.01, 0, 0.3]
    tb2 = tool_body_from_urdf(URDF.format(mesh=str(p)), "kiro_ee_point", root_frame="flange", T_root_tcp=T_ft)
    assert tb2.links == ("link6", "tool") and np.allclose(np.sort(tb2.V, axis=0), np.sort(tb.V, axis=0), atol=1e-12)
    with pytest.raises(ValueError, match="not found here"):
        tool_body_from_urdf(URDF.format(mesh="package://no_such_pkg/x.stl"), "tcp")
