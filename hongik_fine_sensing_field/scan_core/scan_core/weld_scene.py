"""The specimen as the torch meets it at the weld poses (user 2026-10-13) - the weld-pose scene of D14 and of the torch-orientation
search, and the surface the wire end touches (the plan's vertical aim).

The sensing world (scan_l0.cell_world) is a box grown by the policy clearance C: a sensing / positioning rule that does not apply at
the weld poses (the torch enters the groove). Here the specimen is the measured geometry the plan already holds, section by section
(the plan's stations, x along w from the root-gap centre, z along n from the backing-bar top, mm):

  left_plate, right_plate   each plate on its own, the groove open between them: root face (design, vertical) + bevel face (each
                            station's measured bevel angle) up to the plate top; the right plate's top at t + hi-lo (the station's
                            measured hi-lo, right - left - its groove-side wall continues to that top); outward to the design width of
                            each plate (spec_design.plate_widths_mm, from the root edge)
  backing                   the backing bar (design thickness) under both plates and the gap
  bead_<k>                  each earlier pass k's bead (user + KIRO 2026-10-13 night: beads are convex): the plan's own bead of that pass
                            in that section (scan_core.bead_geometry - the same beads the plan's heights and aims come from) as a convex
                            polygon: the hull of the bead with its top bounded by tangent lines of its arc (it contains the bead; the
                            rest of the hull lies below the surface - earlier beads or the plates)

The pass's own bead is not in the scene (the layers already deposited before each pass); the plates span the measured S .. E sections,
the deposits the same length. Every solid is a convex polygon per station (a straight prism across the seam's sections; a bead's
polygons are padded to one vertex count with repeated points - zero-length edges); MoveIt gets each solid lofted through the stations
(scene_meshes).

Required clearance r (uncertainty model, scan_core.weld_orient): every solid grown by r - each face moved out by r along its normal
(mitered corners; the ends moved out by r along the seam) - the same grown solids for the fast clearance model (weld_clearance) and for
MoveIt (scene_meshes), so both oracles see one geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bead_fill import wall_x_mm

__all__ = ["SectionSolid", "WELD_NAMESPACE", "plate_polys", "state_polys", "pass_solids", "specimen_solids", "surface_mm", "solid_planes",
           "grow", "scene_meshes", "STATE_RULE", "SOLID_SOURCES", "bead_source"]

WELD_NAMESPACE = "hongik/weld/"
STATE_RULE = "the beads deposited before the pass (the plan's convex beads; the pass's own bead is not in the scene)"


@dataclass
class SectionSolid:
    """One solid: a convex polygon per station (N, k, 2) [mm] counter-clockwise in the section (x along w, z along n), extruded along the
    seam over [s_lo, s_hi] (mm from S). Polygons may hold zero-length edges (fixed topology across the stations)."""
    name: str
    poly: np.ndarray
    s_lo: float
    s_hi: float
    source: str


def _box(x0: float, x1: float, z0: float, z1: float) -> np.ndarray:
    return np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]], dtype=float)


def plate_polys(sp, widths_mm, backing_mm: Optional[float]) -> dict:
    """The two plates and the backing bar of one section (GrooveSpec sp: the station's measured section) → {name: (k, 2) mm}."""
    g, rf, t = 0.5 * float(sp.root_gap_mm), float(sp.root_face_mm), float(sp.thickness_mm)
    tr = t + float(sp.hi_lo_mm)
    wl, wr = (float(v) for v in widths_mm)
    xo_l, xo_r = -g - wl, g + wr
    xl_t = float(wall_x_mm(sp, t)[0])
    xr_t = float(wall_x_mm(sp, max(tr, rf))[1])
    out = {"left_plate": np.array([[xo_l, 0.0], [-g, 0.0], [-g, rf], [xl_t, t], [xo_l, t]], dtype=float),
           "right_plate": np.array([[g, 0.0], [xo_r, 0.0], [xo_r, tr], [xr_t, tr], [g, rf]], dtype=float)}
    if backing_mm is not None and float(backing_mm) > 0.0:
        out["backing"] = _box(xo_l, xo_r, -float(backing_mm), 0.0)
    return out


def state_polys(sp, widths_mm, backing_mm, beads=()) -> dict:
    """The specimen of one section before a pass → {name: (k, 2) mm}: plates + backing + the earlier beads (beads = [(name,
    BeadSection)]: each bead's convex solid, BeadSection.hull)."""
    out = plate_polys(sp, widths_mm, backing_mm)
    for name, b in beads:
        out[name] = np.asarray(b.hull, dtype=float)
    return out


def surface_mm(polys, x_mm: float) -> float:
    """The height of the specimen's upper surface at x (the highest point of any solid over x) [mm]; NaN when no solid is under x."""
    x = float(x_mm)
    best = -math.inf
    for P in (polys.values() if isinstance(polys, dict) else polys):
        P = np.asarray(P, dtype=float)
        A, B = P, np.roll(P, -1, axis=0)
        lo, hi = np.minimum(A[:, 0], B[:, 0]), np.maximum(A[:, 0], B[:, 0])
        on = (lo <= x + 1e-12) & (x - 1e-12 <= hi)
        for a, b in zip(A[on], B[on]):
            dx = b[0] - a[0]
            z = max(a[1], b[1]) if abs(dx) < 1e-12 else a[1] + (b[1] - a[1]) * (x - a[0]) / dx
            best = max(best, float(z))
    return best if math.isfinite(best) else float("nan")


def _pad(P: np.ndarray, k: int) -> np.ndarray:
    """A convex polygon padded to k vertices with its last vertex repeated (zero-length edges)."""
    P = np.asarray(P, dtype=float)
    return P if len(P) >= k else np.vstack([P, np.repeat(P[-1:], k - len(P), axis=0)])


def bead_source(k: int, role: str) -> str:
    return (f"pass {int(k)}'s bead ({role}; the plan's convex bead - bead_geometry: its hull with the arc's tangent polygon on top, "
            "containing the bead)")


def pass_solids(plan, wp) -> list:
    """The specimen before pass wp of plan (module docstring) → [SectionSolid], polygons at the plan's stations (S → E order): the
    plates, the backing and the beads of the passes before wp (plan order)."""
    st = plan.stations
    S = np.array([s.s_mm for s in st], dtype=float)
    widths = plan.spec_design.plate_widths_mm
    back = plan.spec_design.backing_thickness_mm if plan.spec_design.backing else None
    per = [plate_polys(s.spec, widths, back) for s in st]
    out = [SectionSolid(name=n, poly=np.stack([p[n] for p in per]), s_lo=float(S.min()), s_hi=float(S.max()), source=SOLID_SOURCES[n])
           for n in per[0]]
    dep = getattr(plan, "deposits", None) or {}
    roles = {int(p.k): p.role for p in plan.passes}
    if dep:                                                   # the plan's record of every pass's beads (station order)
        earlier = [(int(k), list(v)) for k, v in sorted(dep.items()) if int(k) < int(wp.k)]
    else:                                                     # a plan without it: the beads of the passes it carries
        earlier = []
        for p in plan.passes:
            if int(p.k) < int(wp.k) and getattr(p, "beads", None):
                idx = {float(v): i for i, v in enumerate(p.s_mm)}
                earlier.append((int(p.k), [p.beads[idx[float(s)]] for s in S]))
    for k_, bs in earlier:
        hulls = [np.asarray(b.hull, dtype=float) for b in bs]
        k = max(len(h) for h in hulls)
        out.append(SectionSolid(name=f"bead_{k_}", poly=np.stack([_pad(h, k) for h in hulls]), s_lo=float(S.min()),
                                s_hi=float(S.max()), source=bead_source(k_, roles.get(k_, "pass"))))
    return out


SOLID_SOURCES = {"left_plate": "the measured sections' left plate (root face + bevel to the plate top; design width)",
                 "right_plate": "the measured sections' right plate (top at t + hi-lo; design width)",
                 "backing": "the backing bar (design thickness) under both plates"}


def specimen_solids(stations, spec_design) -> list:
    """The measured specimen with nothing deposited (the bare specimen, 2026-10-13): both plates with
    the measured groove open and the backing bar, at the given stations (GrooveStation: s_mm, spec) → [SectionSolid] - pass_solids of
    a first pass (no fill, no earlier bead), from the stations and the design alone (plate widths, backing thickness)."""
    st = list(stations)
    if not st:
        raise ValueError("no stations: the specimen is undefined")
    S = np.array([float(x.s_mm) for x in st], dtype=float)
    widths = spec_design.plate_widths_mm
    back = spec_design.backing_thickness_mm if spec_design.backing else None
    per = [plate_polys(x.spec, widths, back) for x in st]
    return [SectionSolid(name=n, poly=np.stack([p[n] for p in per]), s_lo=float(S.min()), s_hi=float(S.max()), source=SOLID_SOURCES[n])
            for n in per[0]]


def solid_planes(P: np.ndarray) -> tuple:
    """Polygon(s) (..., k, 2) CCW → (outward unit normals (..., k, 2) of edge j = v_j → v_j+1, offsets b (..., k): n·p <= b inside,
    miters (..., k, 2) of vertex j: n_(j-1)·m = n_j·m = 1, the vertex moves by r·m when every face moves out by r). A zero-length edge
    takes the normal of the next edge that has a length; a vertex between faces with the same normal moves along that normal."""
    P = np.asarray(P, dtype=float)
    D = np.roll(P, -1, axis=-2) - P
    L = np.linalg.norm(D, axis=-1)
    Nrm = np.stack([D[..., 1], -D[..., 0]], axis=-1) / np.maximum(L, 1e-300)[..., None]
    k = P.shape[-2]
    bad = L < 1e-9
    for _ in range(k):                                                   # zero-length edges: the next edge's normal
        if not bad.any():
            break
        nxt = np.roll(Nrm, -1, axis=-2)
        nb = np.roll(bad, -1, axis=-1)
        Nrm = np.where(bad[..., None], nxt, Nrm)
        bad = bad & nb
    b = np.sum(Nrm * P, axis=-1)
    Np = np.roll(Nrm, 1, axis=-2)                                         # normal of edge j-1 (ends at vertex j)
    det = Np[..., 0] * Nrm[..., 1] - Np[..., 1] * Nrm[..., 0]
    ok = np.abs(det) > 1e-12
    mx = np.where(ok, (Nrm[..., 1] - Np[..., 1]) / np.where(ok, det, 1.0), Nrm[..., 0])
    my = np.where(ok, (Np[..., 0] - Nrm[..., 0]) / np.where(ok, det, 1.0), Nrm[..., 1])
    return Nrm, b, np.stack([mx, my], axis=-1)


def grow(P: np.ndarray, r_mm: float) -> np.ndarray:
    """Every face of the convex polygon(s) moved out by r along its normal (mitered corners)."""
    _n, _b, M = solid_planes(P)
    return np.asarray(P, dtype=float) + float(r_mm) * M


def scene_meshes(plan, solids: list, r_mm: float, *, prefix: str = WELD_NAMESPACE) -> list:
    """The solids grown by r (module docstring), lofted through the plan's stations in its base frame → [{id, vertices_m (V, 3),
    triangles (T, 3), source}] for the MoveIt planning scene (the ends moved out by r along the seam)."""
    st = plan.stations
    O = np.array([s.origin for s in st], dtype=float)
    W = np.array([s.w for s in st], dtype=float)
    Nn = np.array([s.n for s in st], dtype=float)
    U = np.array([s.u for s in st], dtype=float)
    out = []
    r = float(r_mm)
    for so in solids:
        G = grow(so.poly, r)                                              # (N, k, 2)
        N, k = G.shape[:2]
        X = O[:, None, :] + (G[..., 0:1] * W[:, None, :] + G[..., 1:2] * Nn[:, None, :]) * 1e-3   # (N, k, 3)
        X = np.concatenate([X[:1] - U[:1, None, :] * (r * 1e-3), X, X[-1:] + U[-1:, None, :] * (r * 1e-3)], axis=0)
        M = X.shape[0]
        V = X.reshape(-1, 3)
        tri = []
        for i in range(M - 1):
            for j in range(k):
                a, b = i * k + j, i * k + (j + 1) % k
                c, d = (i + 1) * k + j, (i + 1) * k + (j + 1) % k
                tri += [[a, b, d], [a, d, c]]
        for j in range(1, k - 1):                                         # end caps (convex fan)
            tri.append([0, j + 1, j])
            tri.append([(M - 1) * k, (M - 1) * k + j, (M - 1) * k + j + 1])
        out.append({"id": prefix + so.name, "vertices_m": V, "triangles": np.asarray(tri, dtype=np.int64),
                    "source": f"{so.source}; grown by {r:.2f} mm (required clearance)"})
    return out
