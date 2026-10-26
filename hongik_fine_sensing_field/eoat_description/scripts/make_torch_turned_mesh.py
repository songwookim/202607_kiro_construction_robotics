#!/usr/bin/env python3
"""EOAT meshes of KIRO's tool setup (identified 2026-10-13 from KIRO's September weld logs and a photo; the KIRO robot model
construct_robot_0528_hongik_v2 r4): the same bracket, torch and sensor as the lab EOAT, but the torch turned 180 deg in its clamp about
its straight body axis - the neck bends away from the sensor (at a weld pose the torch is below, the profiler above). The mounting yaw
on the flange is the lab's (-90 deg, the calibration file's eoat_yaw); this script only turns the torch inside the eoat frame.

meshes/eoat_base_link_dustplate.STL            ->  meshes/eoat_base_link_dustplate_torch180.STL            (visual)
meshes/eoat_base_link_dustplate_collision.STL  ->  meshes/eoat_base_link_dustplate_torch180_collision.STL  (collision, no wire stub)
(deterministic; test_eoat_frames regenerates both and compares bytes). The same rules as the KIRO URDF generator (fsv1_work/kiro_urdf/
v2/make_v2.py r4), so the triangles are those of the delivered r4 meshes:
  1. parts = vertex-connected components (vertices rounded to 1 nm);
  2. the torch = the components entirely at x_eoat < -20 mm that reach beyond z_eoat 100 mm (body, neck, gas nozzle, contact tip
     [+ the CAD wire stub in the visual mesh]); the holder (clamp halves, screws) = the components at x_eoat < -20 mm that stay at the
     clamp height - fixed, like the bracket arm, adapter / dust plate, sensor and guard;
  3. the straight torch body's axis = the line every side-face normal line of the body meets (Pluecker condition (p - c) . (a x n) = 0,
     area-weighted least squares over the body's side faces up to the start of the neck bend; start = the circle centre of the lowest
     vertex ring) - fitted on the collision mesh;
  4. the torch triangles turn by 180 deg about that axis (x -> R (x - c) + c); every triangle's normal is recomputed (float32 cross
     product, normalised), the attribute bytes are zero.
The torch frame weld_link and the TCP (eoat.xacro weld_joint . torch_tcp defaults: the CAD contact-tip end face centre, +z = the nozzle
axis toward the work, x in the bend plane) turn with the torch: tcp_eoat() / weld_link_eoat().

  python3 make_torch_turned_mesh.py [--check]      (--check: regenerate both in memory and compare with the committed files)
"""
from __future__ import annotations

import argparse
import math
import re
import struct
import sys
import xml.dom.minidom
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MESHES = HERE.parent / "meshes"
SRC_VISUAL = MESHES / "eoat_base_link_dustplate.STL"
SRC_COLLISION = MESHES / "eoat_base_link_dustplate_collision.STL"
OUT_VISUAL = MESHES / "eoat_base_link_dustplate_torch180.STL"
OUT_COLLISION = MESHES / "eoat_base_link_dustplate_torch180_collision.STL"
XACRO = HERE.parent / "urdf" / "eoat.xacro"
TURN_DEG = 180.0                                   # KIRO: the torch turned 180 deg in its clamp (user 2026-10-13, URDF v2 r4)
TORCH_X_MAX_M, TORCH_REACH_Z_M = -0.020, 0.100     # the torch: components entirely at x_eoat < -20 mm reaching beyond z_eoat 100 mm
HEADER_VISUAL = b"eoat_base_link + dust plate, torch turned 180 deg (make_torch_turned_mesh.py)"
HEADER_COLLISION = b"eoat_base_link + dust plate, collision, torch turned 180 deg (make_torch_turned_mesh.py)"
REC = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def read_triangles(path) -> np.ndarray:
    """Binary STL -> triangles (n, 3, 3) float64 [m]."""
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    if 84 + 50 * n != len(b):
        raise ValueError(f"{path}: not a binary STL of {n} triangles")
    return np.frombuffer(b[84:], dtype=REC)["v"].astype(np.float64)


def stl_bytes(T, header: bytes) -> bytes:
    """Triangles -> binary STL bytes: float32 vertices, normals from the float32 cross product (normalised in double), attribute 0."""
    T = np.asarray(T, dtype=np.float32).reshape(-1, 3, 3)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]).astype(np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    rec = np.zeros(len(T), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
    rec["n"], rec["v"] = n, T.reshape(-1, 9)
    return header[:80].ljust(80, b" ") + struct.pack("<I", len(T)) + rec.tobytes()


def mesh_parts(T) -> tuple:
    """Triangles -> (unique vertices V, vertex component labels, component per triangle)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    V, inv = np.unique(np.round(T.reshape(-1, 3), 9), axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _n, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))), directed=False)
    return V, lab, lab[F[:, 0]]


def torch_components(V, lab) -> tuple:
    """(the torch's components, the fixed holder's components) - both entirely at x_eoat < -20 mm; the torch reaches beyond z 100 mm."""
    nc = lab.max() + 1
    xmax = np.full(nc, -np.inf)
    np.maximum.at(xmax, lab, V[:, 0])
    zmax = np.full(nc, -np.inf)
    np.maximum.at(zmax, lab, V[:, 2])
    cand = np.flatnonzero(xmax < TORCH_X_MAX_M)
    return [int(c) for c in cand if zmax[c] > TORCH_REACH_Z_M], [int(c) for c in cand if zmax[c] <= TORCH_REACH_Z_M]


def fit_body_axis(T, tri_comp, torch_comps, V, lab) -> tuple:
    """The straight torch body's axis (point c at z_eoat 0, unit direction a; module docstring rule 3) + (side faces used, z range)."""
    from scipy.optimize import least_squares
    body = max(torch_comps, key=lambda c: int(np.sum(lab == c)))
    P = V[lab == body]
    tri = T[tri_comp == body]
    cen = tri.mean(axis=1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-30)[:, None]
    rings = []
    for z in np.unique(np.round(P[:, 2], 4)):
        Q = P[np.abs(P[:, 2] - z) < 5e-5]
        if len(Q) >= 8:
            A = np.c_[2 * Q[:, 0], 2 * Q[:, 1], np.ones(len(Q))]
            s = np.linalg.lstsq(A, (Q[:, :2] ** 2).sum(1), rcond=None)[0]
            rings.append((float(z), float(s[0]), float(s[1])))
    rings = np.array(rings)
    c0 = rings[0, 1:3]
    same = np.flatnonzero(np.hypot(rings[:, 1] - c0[0], rings[:, 2] - c0[1]) < 1e-4)
    z_straight = float(rings[same[-1], 0])
    sel = (tri[:, :, 2].max(axis=1) <= z_straight + 1e-6) & (np.abs(nrm[:, 2]) < 0.95) & (area > 1e-9)
    p, n, wgt = cen[sel], nrm[sel], np.sqrt(area[sel])

    def res(x):
        c = np.array([x[0], x[1], 0.0])
        a = np.array([x[2], x[3], 1.0])
        a /= np.linalg.norm(a)
        axn = np.cross(a, n)
        return wgt * np.einsum("ij,ij->i", p - c, axn) / np.maximum(np.linalg.norm(axn, axis=1), 1e-9)
    sol = least_squares(res, x0=[c0[0], c0[1], 0.0, 0.0], x_scale=[0.01, 0.01, 0.01, 0.01])
    a = np.array([sol.x[2], sol.x[3], 1.0])
    a /= np.linalg.norm(a)
    return np.array([sol.x[0], sol.x[1], 0.0]), a, int(sel.sum()), (float(rings[0, 0]), z_straight)


def rot_about(a, deg) -> np.ndarray:
    a = np.asarray(a, float) / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


def _T(xyz, rpy) -> np.ndarray:
    """URDF origin -> 4x4 (rpy = Rz(yaw) Ry(pitch) Rx(roll))."""
    r, p_, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p_), math.sin(p_), math.cos(y), math.sin(y)
    M = np.eye(4)
    M[:3, :3] = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                          [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                          [-sp, cp * sr, cp * cr]])
    M[:3, 3] = [float(v) for v in xyz]
    return M


def xacro_torch(xacro=None) -> tuple:
    """eoat.xacro: eoat <- weld_link (weld_joint origin) and weld_link <- torch_tcp (macro defaults torch_xyz / torch_rpy), 4x4 each."""
    dom = xml.dom.minidom.parse(str(XACRO if xacro is None else xacro))
    d = dict(re.findall(r"(\w+)='([^']*)'", dom.getElementsByTagName("xacro:macro")[0].getAttribute("params")))
    for j in dom.getElementsByTagName("joint"):
        if j.getAttribute("name").endswith("weld_joint"):
            o = j.getElementsByTagName("origin")[0]
            return (_T(o.getAttribute("xyz").split(), o.getAttribute("rpy").split()),
                    _T(d["torch_xyz"].split(), d["torch_rpy"].split()))
    raise KeyError("weld_joint")


def turn(src_visual=None, src_collision=None, turn_deg: float = TURN_DEG) -> dict:
    """The turned torch: {visual, collision (triangles), T_turn (4x4 in the eoat frame), axis {point_m, direction}, counts}."""
    Tc = read_triangles(SRC_COLLISION if src_collision is None else src_collision)
    Tv = read_triangles(SRC_VISUAL if src_visual is None else src_visual)
    Vc, labc, tric = mesh_parts(Tc)
    Vv, labv, triv = mesh_parts(Tv)
    torch_c, holder_c = torch_components(Vc, labc)
    torch_v, _holder_v = torch_components(Vv, labv)
    c, a, n_faces, z_range = fit_body_axis(Tc, tric, torch_c, Vc, labc)
    R = rot_about(a, turn_deg)
    T_turn = np.eye(4)
    T_turn[:3, :3], T_turn[:3, 3] = R, c - R @ c

    def turned(T, tri_comp, comps):
        out = T.copy()
        m = np.isin(tri_comp, comps)
        out[m] = ((T[m].reshape(-1, 3) - c) @ R.T + c).reshape(-1, 3, 3)
        return out, int(m.sum())
    Tc4, nc = turned(Tc, tric, torch_c)
    Tv4, nv = turned(Tv, triv, torch_v)
    return {"visual": Tv4, "collision": Tc4, "T_turn": T_turn, "turn_deg": float(turn_deg),
            "axis": {"point_m": c, "direction": a, "fit_side_faces": n_faces, "straight_z_m": z_range},
            "torch_components": torch_c, "holder_components": holder_c, "turned_triangles": {"visual": nv, "collision": nc}}


def weld_link_eoat(T_turn) -> np.ndarray:
    """eoat <- weld_link of the turned torch (4x4)."""
    T_ew, _T_wt = xacro_torch()
    return np.asarray(T_turn, dtype=float) @ T_ew


def tcp_eoat(T_turn) -> np.ndarray:
    """eoat <- torch_tcp of the turned torch (4x4): the contact-tip end face centre, +z = the nozzle axis toward the work."""
    T_ew, T_wt = xacro_torch()
    return np.asarray(T_turn, dtype=float) @ T_ew @ T_wt


def build() -> tuple:
    """-> (visual STL bytes, collision STL bytes, turn result)."""
    t = turn()
    return stl_bytes(t["visual"], HEADER_VISUAL), stl_bytes(t["collision"], HEADER_COLLISION), t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="regenerate in memory and compare with the committed meshes (no write)")
    a = ap.parse_args(argv)
    vis, col, t = build()
    T_tcp = tcp_eoat(t["T_turn"])
    print(f"torch body axis: point ({t['axis']['point_m'][0] * 1e3:.4f}, {t['axis']['point_m'][1] * 1e3:.4f}) mm, direction "
          f"{np.round(t['axis']['direction'], 6).tolist()}; turned triangles {t['turned_triangles']}; TCP eoat "
          f"{np.round(T_tcp[:3, 3] * 1e3, 3).tolist()} mm, axis {np.round(T_tcp[:3, 2], 4).tolist()}")
    if a.check:
        bad = [p.name for p, b in ((OUT_VISUAL, vis), (OUT_COLLISION, col)) if not p.is_file() or p.read_bytes() != b]
        print("check: " + ("OK" if not bad else f"DIFFERS: {bad}"))
        return 1 if bad else 0
    OUT_VISUAL.write_bytes(vis)
    OUT_COLLISION.write_bytes(col)
    print(f"wrote {OUT_VISUAL.name} ({len(vis)} bytes), {OUT_COLLISION.name} ({len(col)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
