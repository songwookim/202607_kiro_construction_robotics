#!/usr/bin/env python3
"""EOAT mesh with the dust plate (user 2026-10-12 night): in the lab and at KIRO a 5.0 mm dust plate sits between the profiler and the
bracket, so the real sensor is 5.0 mm further toward the work along +z_eoat than in the CAD (eoat_base_link.STL, kept as is).

meshes/eoat_base_link.STL  ->  meshes/eoat_base_link_dustplate.STL (deterministic; test_eoat_frames regenerates it and compares bytes):
  1. parts: the mock torch = vertex-connected components entirely at x_eoat < -20 mm (as for the torch TCP); every other triangle
     belongs to a closed shell (faces joined across edges used by exactly two faces - parts that only touch are separate shells);
  2. the mount interface = the back face of the profiler housing = the lowest z of the largest shell (the housing, z_eoat 57.0 mm);
  3. the sensor side = the non-torch shells that reach past that plane toward the work (housing, side plates, lower frame, guard
     plate, cable connector, screws); the bracket side = the shells entirely on the flange side (adapter plate, base plate, bracket
     arm and its flange) - unchanged, like the torch;
  4. the sensor side moves +5.0 mm along +z_eoat; the gap is filled with a 5.0 mm prism over the convex hull (x, y) of the housing's
     back faces in the interface plane (faces with normal -z there), so the collision body has no hole.
Not changed: the calibration (laser_link -> optical, measured - it already carries the plate), laser_link (the parent frame of that
calibration), the torch.

The collision mesh (user 2026-10-13: the weld-pose check uses the URDF collision geometry): meshes/eoat_base_link_dustplate_collision.STL
= the same mesh without the CAD wire stub - the wire is the process stick-out (job weld_plan.stick_out_mm), not a rigid body, and at a
weld pose its end meets the work by definition. The wire = the triangles of the contact-tip part (the part through the torch TCP of
eoat.xacro) that lie at or beyond the TCP plane (the contact-tip end face) and leave it; the hole it leaves in the end face is closed by
a fan of the wire's base ring (normal +z_TCP). The visual keeps the full mesh.

  python3 make_dust_plate_mesh.py [--check]      (--check: regenerate both in memory and compare with the committed files)
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "meshes" / "eoat_base_link.STL"
OUT = HERE.parent / "meshes" / "eoat_base_link_dustplate.STL"
OUT_COLLISION = HERE.parent / "meshes" / "eoat_base_link_dustplate_collision.STL"
XACRO = HERE.parent / "urdf" / "eoat.xacro"
HEADER_COLLISION = b"eoat_base_link + dust plate, collision: no wire stub (make_dust_plate_mesh.py)"
DUST_PLATE_M = 0.005                          # user 2026-10-12: the dust plate thickness (lab and KIRO)
TORCH_X_MAX_M = -0.020                        # the mock torch: components entirely at x_eoat < -20 mm (eoat.xacro torch_tcp)
HEADER = b"eoat_base_link + 5.0 mm dust plate (eoat_description/scripts/make_dust_plate_mesh.py)"
REC = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def read_stl(path) -> np.ndarray:
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    return np.frombuffer(b[84:84 + 50 * n], dtype=REC).copy()


def _components(n_nodes: int, pairs: np.ndarray) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_nodes, n_nodes))
    return connected_components(A, directed=False)[1]


def classify(rec: np.ndarray) -> dict:
    """→ {torch, sensor, bracket (bool per triangle), z_interface_m, housing (bool per triangle)}."""
    T = rec["v"].astype(float)
    V, inv = np.unique(T.reshape(-1, 3), axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    vlab = _components(len(V), np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]))
    flab = vlab[F[:, 0]]
    cmax = np.full(flab.max() + 1, -np.inf)
    np.maximum.at(cmax, flab, T.max(axis=1)[:, 0])
    torch = cmax[flab] < TORCH_X_MAX_M
    idx = np.where(~torch)[0]
    Fn = F[idx]
    E = np.sort(np.concatenate([Fn[:, [0, 1]], Fn[:, [1, 2]], Fn[:, [2, 0]]]), axis=1)
    fid = np.tile(np.arange(len(Fn)), 3)
    key = E[:, 0].astype(np.int64) * len(V) + E[:, 1]
    o = np.argsort(key, kind="stable")
    ks, fs = key[o], fid[o]
    _, start, cnt = np.unique(ks, return_index=True, return_counts=True)
    two = cnt == 2                                                  # manifold edges join faces of one closed shell
    pairs = np.stack([fs[start[two]], fs[start[two] + 1]], axis=1)
    slab = _components(len(Fn), pairs)
    ns = slab.max() + 1
    Tn = T[idx]
    zmin, zmax = np.full(ns, np.inf), np.full(ns, -np.inf)
    np.minimum.at(zmin, slab, Tn.min(axis=1)[:, 2])
    np.maximum.at(zmax, slab, Tn.max(axis=1)[:, 2])
    housing_shell = int(np.argmax(np.bincount(slab, minlength=ns)))
    zi = float(zmin[housing_shell])
    sensor_shell = zmax > zi + 1e-6
    sensor = np.zeros(len(rec), dtype=bool)
    sensor[idx] = sensor_shell[slab]
    housing = np.zeros(len(rec), dtype=bool)
    housing[idx] = slab == housing_shell
    return {"torch": torch, "sensor": sensor, "bracket": ~torch & ~sensor, "z_interface_m": zi, "housing": housing}


def _hull2d(P: np.ndarray) -> np.ndarray:
    """Convex hull (x, y), counter-clockwise, monotone chain (deterministic)."""
    pts = sorted(set(map(tuple, np.round(P, 9))))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1], dtype=float)


def plate_triangles(poly: np.ndarray, z_top: float, z_bottom: float) -> list:
    """Prism over a CCW polygon (x, y) between z_top (bracket side, normal -z) and z_bottom (sensor side, normal +z): (normal, tri)."""
    out = []
    k = len(poly)
    top = [np.array([x, y, z_top]) for x, y in poly]
    bot = [np.array([x, y, z_bottom]) for x, y in poly]
    for i in range(1, k - 1):
        out.append(((0.0, 0.0, -1.0), (top[0], top[i + 1], top[i])))      # facing -z (toward the flange)
        out.append(((0.0, 0.0, 1.0), (bot[0], bot[i], bot[i + 1])))       # facing +z (toward the sensor)
    for i in range(k):
        j = (i + 1) % k
        d = poly[j] - poly[i]
        nrm = (float(d[1]), float(-d[0]), 0.0)
        nn = float(np.hypot(nrm[0], nrm[1]))
        nrm = (nrm[0] / nn, nrm[1] / nn, 0.0)                             # outward for a CCW polygon
        out.append((nrm, (top[i], bot[i], bot[j])))
        out.append((nrm, (top[i], bot[j], top[j])))
    return out


def build(src=SRC) -> tuple:
    """→ (STL bytes, info dict)."""
    rec = read_stl(src)
    c = classify(rec)
    zi, dz = c["z_interface_m"], DUST_PLATE_M
    T = rec["v"].astype(float)
    nz = rec["n"][:, 2]
    back = c["sensor"] & (nz < -0.999) & np.all(np.abs(T[:, :, 2] - zi) < 1e-6, axis=1)    # the sensor's back faces at the interface
    poly = _hull2d(T[back][:, :, :2].reshape(-1, 2))
    out = rec.copy()
    moved = T.copy()
    moved[c["sensor"], :, 2] += dz
    out["v"] = moved.astype("<f4")
    plate = plate_triangles(poly, zi, zi + dz)
    prec = np.zeros(len(plate), dtype=REC)
    prec["n"] = np.array([p[0] for p in plate], dtype="<f4")
    prec["v"] = np.array([np.stack(p[1]) for p in plate], dtype="<f4")
    allrec = np.concatenate([out, prec])
    data = HEADER[:80].ljust(80, b"\0") + struct.pack("<I", len(allrec)) + allrec.tobytes()
    info = {"z_interface_mm": zi * 1e3, "dust_plate_mm": dz * 1e3, "n_sensor_tris": int(c["sensor"].sum()),
            "n_bracket_tris": int(c["bracket"].sum()), "n_torch_tris": int(c["torch"].sum()), "n_plate_tris": len(plate),
            "plate_footprint_mm": {"x": [float(poly[:, 0].min() * 1e3), float(poly[:, 0].max() * 1e3)],
                                   "y": [float(poly[:, 1].min() * 1e3), float(poly[:, 1].max() * 1e3)], "n_vertices": len(poly)}}
    return data, info


def eoat_from_torch_tcp(xacro=XACRO) -> np.ndarray:
    """eoat_base_link <- torch_tcp (4x4) from eoat.xacro: weld_joint origin x torch_tcp_joint origin (the macro defaults)."""
    import re
    import xml.dom.minidom
    from scipy.spatial.transform import Rotation
    dom = xml.dom.minidom.parse(str(xacro))
    params = dict(re.findall(r"(\w+)='([^']*)'", dom.getElementsByTagName("xacro:macro")[0].getAttribute("params")))

    def T(xyz, rpy):
        M = np.eye(4)
        M[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in rpy.split()]).as_matrix()
        M[:3, 3] = [float(v) for v in xyz.split()]
        return M
    for j in dom.getElementsByTagName("joint"):
        if j.getAttribute("name").endswith("weld_joint"):
            o = j.getElementsByTagName("origin")[0]
            return T(o.getAttribute("xyz"), o.getAttribute("rpy")) @ T(params["torch_xyz"], params["torch_rpy"])
    raise KeyError("weld_joint")


def build_collision(data: bytes, T_eoat_tcp=None) -> tuple:
    """The dust-plate mesh (STL bytes) without the wire stub, the contact-tip end face closed → (STL bytes, info dict)."""
    T_eoat_tcp = eoat_from_torch_tcp() if T_eoat_tcp is None else np.asarray(T_eoat_tcp, dtype=float)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=REC).copy()
    T = rec["v"].astype(float)
    V, inv = np.unique(T.reshape(-1, 3), axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    vlab = _components(len(V), np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]))
    o, a = T_eoat_tcp[:3, 3], T_eoat_tcp[:3, 2]
    tip = vlab[int(np.argmin(np.linalg.norm(V - o, axis=1)))]               # the contact-tip part (through the TCP)
    t = (V - o) @ a
    eps = 1e-6                                                              # on the TCP plane: 1 um (float32 STL, 6-digit xacro)
    wire = (vlab[F[:, 0]] == tip) & np.all(t[F] >= -eps, axis=1) & np.any(t[F] > eps, axis=1)
    if not wire.any():
        raise ValueError("no wire stub beyond the torch TCP in the mesh")
    keep = ~wire
    wv = np.unique(F[wire])
    ring = sorted(int(v) for v in wv[np.abs(t[wv]) <= eps])                   # the wire's base ring on the end face
    if len(ring) < 3 or not np.isin(ring, F[keep]).all():
        raise ValueError("the wire's base is not a ring on the TCP plane")
    x_ax, y_ax = T_eoat_tcp[:3, 0], T_eoat_tcp[:3, 1]
    ang = np.arctan2((V[ring] - o) @ y_ax, (V[ring] - o) @ x_ax)
    loop = [ring[i] for i in np.argsort(ang)]                                # counter-clockwise about +z_TCP
    cap = []
    for i in range(1, len(loop) - 1):
        cap.append((V[loop[0]], V[loop[i]], V[loop[i + 1]]))                  # normal +z_TCP (the end face faces the work)
    crec = np.zeros(len(cap), dtype=REC)
    crec["n"] = np.tile(a.astype("<f4"), (len(cap), 1))
    crec["v"] = np.array([np.stack(c) for c in cap], dtype="<f4")
    allrec = np.concatenate([rec[keep], crec])
    out = HEADER_COLLISION[:80].ljust(80, b"\0") + struct.pack("<I", len(allrec)) + allrec.tobytes()
    return out, {"wire_triangles_removed": int(wire.sum()), "end_face_fan_triangles": len(cap), "ring_vertices": len(loop),
                 "wire_length_mm": float(t[np.unique(F[wire])].max() * 1e3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="regenerate in memory and compare with the committed file")
    a = ap.parse_args(argv)
    data, info = build()
    coll, cinfo = build_collision(data)
    print(info)
    print(cinfo)
    if a.check:
        same = OUT.exists() and OUT.read_bytes() == data
        same_c = OUT_COLLISION.exists() and OUT_COLLISION.read_bytes() == coll
        print("identical" if same else f"DIFFERS from {OUT}")
        print("identical" if same_c else f"DIFFERS from {OUT_COLLISION}")
        return 0 if same and same_c else 1
    OUT.write_bytes(data)
    OUT_COLLISION.write_bytes(coll)
    print(f"wrote {OUT} ({len(data)} bytes), {OUT_COLLISION} ({len(coll)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
