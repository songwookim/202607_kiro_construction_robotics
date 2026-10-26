"""Clearance of the weld torch's tool from the specimen at the weld poses (user 2026-10-13) - the fast geometric oracle of the torch
orientation search (weld_orient) and of the virtual arm's D14; MoveIt's own collision check is the oracle on a MoveIt platform.

Tool (ToolBody, tool_body_from_urdf): the robot's own URDF collision geometry rigid with the weld torch's TCP - every link joined to
the TCP frame through fixed joints (the EOAT collision mesh - without the wire stub, eoat.xacro - and the wrist link it is bolted to),
as triangles in the TCP frame. The same geometry MoveIt checks.

Specimen (weld_scene.pass_solids): convex polygons per station (x along w, z along n, mm), each extruded along the seam between the
plate ends; the solids are grown by the required clearance r with every face moved out by r (mitered corners, ends moved by r) - the
same grown solids MoveIt gets (weld_scene.scene_meshes).

Clearance of a pose = the largest r for which the tool still clears every solid grown by r: for a solid with face planes n_f · p <= b_f
(the polygon's edges and the two end planes), g(p) = max_f (n_f · p - b_f) is the r at which the grown solid reaches p; the clearance is
the smallest r at which any part of the tool touches a grown solid (negative = the tool is inside the specimen by that much). Exact for
the triangle mesh against the convex prisms (each prism taken at its station's section): the first contact as r grows is one of
  (a) a tool vertex on a face of the grown solid - g at the vertex,
  (b) a tool edge on a grown ridge line along the seam (a polygon vertex moving by r · its miter),
  (b') a tool edge on a grown edge of an end face,
  (c') a corner of a grown end face on a tool triangle,
each event checked to be on the grown solid's boundary (g at the contact point = r). Each pose is evaluated in one station's section
(a row of the geometry): D14 evaluates the delivered weld line's samples (scan_core.weld_line, one base rotation) each in the section
of the station nearest its wire end - the rotation relative to the section then differs a little between rows (the measured sections
turn), so clearance() takes one rotation per row (rows_geometry / rows_poses / poses_clearance).
"""
from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .weld_scene import pass_solids, solid_planes

__all__ = ["ToolBody", "ToolRoles", "read_stl_triangles", "tool_body_from_triangles", "tool_body_from_urdf", "nozzle_end_mm",
           "tool_roles", "part_link", "PROCESS_ROLE", "OTHER_ROLE", "PassGeometry", "pass_geometry", "solids_geometry", "clearance",
           "poses_clearance", "nearest_station", "rows_geometry", "rows_poses", "default_resolve", "body_digest", "warm_caches"]

PROCESS_ROLE, OTHER_ROLE = "process", "other"     # the tool parts' roles in the weld-pose clearance (user 2026-10-13; tool_roles)

_EPS_MM = 1e-7


def read_stl_triangles(path) -> np.ndarray:
    """STL (binary or ASCII) → triangles (T, 3, 3) in the file's units."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n and n > 0:
            rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
            return rec["v"].reshape(-1, 3, 3).astype(np.float64)
    pts = [[float(x) for x in ln.split()[1:4]] for ln in data.decode("ascii", errors="replace").splitlines()
           if ln.strip().startswith("vertex")]
    if not pts or len(pts) % 3:
        raise ValueError(f"{path}: no triangles")
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3, 3)


@dataclass(frozen=True, eq=False)
class ToolBody:
    """Triangle mesh in the TCP frame [m]: unique vertices V (n, 3), faces F (f, 3), unique edges E (e, 2); face_link (f,) = the
    index in links of the URDF link each face belongs to (None = not tracked)."""
    V: np.ndarray
    F: np.ndarray
    E: np.ndarray
    tcp_frame: str
    source: str
    links: tuple = ()
    face_link: Optional[np.ndarray] = None

    def describe(self) -> str:
        return (f"{self.tcp_frame} tool: {len(self.F)} triangles ({', '.join(self.links) if self.links else 'mesh'}; {self.source})")


def tool_body_from_triangles(T_tcp: np.ndarray, *, tcp_frame: str, source: str, links=(), face_link=None) -> ToolBody:
    """Triangles (T, 3, 3) [m] in the TCP frame → ToolBody (vertices merged at 1 nm); face_link (T,) = each triangle's index in links."""
    T = np.asarray(T_tcp, dtype=np.float64).reshape(-1, 3, 3)
    V, inv = np.unique(np.round(T.reshape(-1, 3), 9), axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[keep]
    fl = None if face_link is None else np.asarray(face_link, dtype=np.int64).reshape(-1)[keep]
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    return ToolBody(V=V, F=F, E=E, tcp_frame=str(tcp_frame), source=str(source), links=tuple(links), face_link=fl)


def default_resolve(filename: str) -> Optional[str]:
    """URDF mesh name → local path: package://<pkg>/<rel> through the ament index, else this source tree (scan_stack/<pkg>/<rel>);
    file:// and plain paths as they are."""
    if filename.startswith("package://"):
        pkg, _, rel = filename[len("package://"):].partition("/")
        try:
            from ament_index_python.packages import get_package_share_directory
            p = Path(get_package_share_directory(pkg)) / rel
            if p.exists():
                return str(p)
        except Exception:  # noqa: BLE001 - no ament index here
            pass
        p = Path(__file__).resolve().parents[2] / pkg / rel
        return str(p) if p.exists() else None
    if filename.startswith("file://"):
        return filename[len("file://"):]
    return filename


def _origin(el) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in (el.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (el.get("rpy") or "0 0 0").split()]
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()            # URDF: R = Rz(yaw) Ry(pitch) Rx(roll)
    T[:3, 3] = xyz
    return T


def _box_tris(size) -> np.ndarray:
    h = 0.5 * np.asarray(size, dtype=float)
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float) * h
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tri = []
    for a, b, cc, d in quads:
        tri += [[c[a], c[b], c[cc]], [c[a], c[cc], c[d]]]
    return np.array(tri)


def tool_body_from_urdf(urdf, tcp_frame: str, *, resolve: Optional[Callable[[str], Optional[str]]] = None,
                        root_frame: Optional[str] = None, T_root_tcp: Optional[np.ndarray] = None) -> ToolBody:
    """The URDF's collision geometry of every link rigid with tcp_frame (joined through fixed joints, either direction) as one
    ToolBody in the tcp_frame. urdf: the XML text or a path. root_frame: take the links rigid with that frame (the flange the tool is
    bolted to) instead, and express them in the TCP frame with T_root_tcp (root ← TCP, e.g. the robot's TF) - for a TCP the URDF does
    not carry. Mesh (STL, with scale) and box geometry; anything else, or a mesh that cannot be read here, is refused (ValueError) -
    the check never runs on part of the tool."""
    resolve = resolve or default_resolve
    text = str(urdf)
    root = ET.fromstring(text) if text.lstrip().startswith("<") else ET.parse(text).getroot()
    joints = [(j.get("type"), j.find("parent").get("link"), j.find("child").get("link"), _origin(j.find("origin")))
              for j in root.findall("joint")]
    start = str(root_frame) if root_frame else str(tcp_frame)
    T_tcp_start = np.eye(4) if not root_frame else np.linalg.inv(np.asarray(T_root_tcp, dtype=float))
    T_of = {start: T_tcp_start}
    todo = [start]
    while todo:
        cur = todo.pop()
        for typ, par, chi, T in joints:
            if typ != "fixed":
                continue
            if par == cur and chi not in T_of:
                T_of[chi] = T_of[cur] @ T
                todo.append(chi)
            elif chi == cur and par not in T_of:
                T_of[par] = T_of[cur] @ np.linalg.inv(T)
                todo.append(par)
    if len(T_of) == 1 and not any(l_.get("name") == start for l_ in root.findall("link")):
        raise ValueError(f"the URDF has no link {start!r}")
    tris, links, srcs = [], [], []
    for el in root.findall("link"):
        name = el.get("name")
        if name not in T_of:
            continue
        for c in el.findall("collision"):
            T = T_of[name] @ _origin(c.find("origin"))
            g = c.find("geometry")
            g = list(g)[0] if g is not None and len(list(g)) else None
            if g is None:
                raise ValueError(f"link {name}: collision without geometry")
            if g.tag == "mesh":
                fn = g.get("filename") or ""
                p = resolve(fn)
                if not p or not Path(p).exists():
                    raise ValueError(f"link {name}: collision mesh {fn} not found here")
                sc = np.asarray([float(v) for v in (g.get("scale") or "1 1 1").split()], dtype=float)
                P = read_stl_triangles(p) * sc
                srcs.append(Path(p).name)
            elif g.tag == "box":
                P = _box_tris([float(v) for v in g.get("size").split()])
                srcs.append("box")
            else:
                raise ValueError(f"link {name}: collision {g.tag} not supported by the weld-pose clearance model")
            tris.append(P @ T[:3, :3].T + T[:3, 3])
            links.append(name)
    if not tris:
        raise ValueError(f"no collision geometry rigid with {tcp_frame}")
    names = tuple(dict.fromkeys(links))
    face_link = np.concatenate([np.full(len(P_), names.index(l_), dtype=np.int64) for P_, l_ in zip(tris, links)])
    return tool_body_from_triangles(np.concatenate(tris), tcp_frame=tcp_frame,
                                    source="URDF collision geometry rigid with the TCP: " + ", ".join(
                                        f"{l_} ({s_})" for l_, s_ in zip(links, srcs)), links=names, face_link=face_link)


def _axis_parts(body: ToolBody) -> tuple:
    """The tool's connected parts (shared vertices) and which of them the torch axis (the TCP z axis) passes through: (component
    label per vertex, number of parts, {part: (max z_TCP [m], the half-plane equations (a, b, c: a x + b y + c <= 0 inside) of the
    convex hull of its outline on the TCP xy plane)} of the parts whose outline encloses the axis point). Cached on the body (arrays
    only - the body crosses to the worker process)."""
    c = body.__dict__.get("_axis_cache")
    if c is None:
        from scipy.sparse import coo_matrix                              # noqa: PLC0415
        from scipy.sparse.csgraph import connected_components             # noqa: PLC0415
        from scipy.spatial import ConvexHull                              # noqa: PLC0415
        V, E = np.asarray(body.V, dtype=float), np.asarray(body.E)
        n_c, lab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))), directed=False)
        enc = {}
        for k in range(n_c):
            P = V[lab == k]
            if len(P) < 3:
                continue
            try:
                h = ConvexHull(P[:, :2])
            except Exception:  # noqa: BLE001 - a part whose outline along the axis is a line or a point cannot enclose it
                continue
            if np.all(h.equations[:, 2] < 0.0):                          # the axis point (0, 0) strictly inside the outline
                enc[k] = (float(P[:, 2].max()), np.array(h.equations, dtype=float))
        c = (lab, n_c, enc)
        object.__setattr__(body, "_axis_cache", c)
    return c


@dataclass(frozen=True, eq=False)
class ToolRoles:
    """The tool split by the role of its parts in the weld-pose clearance (user 2026-10-13, KIRO: in the root pass the gas nozzle almost
    touches the plate): process = the gas nozzle and the contact tip - the parts designed to approach the work (the wire is the
    process stick-out, not in the tool) - and other = every other part rigid with the TCP (sensor housing and guard, dust plate,
    bracket, clamp, torch body and neck, the wrist link). Each role has its own required clearance (weld_orient)."""
    process: Optional[ToolBody]
    other: ToolBody
    tcp_frame: str
    source: str
    process_links: tuple = ()
    how: str = ""

    def roles(self) -> list:
        return [(r, b) for r, b in ((PROCESS_ROLE, self.process), (OTHER_ROLE, self.other)) if b is not None]

    def describe(self) -> str:
        n_p = 0 if self.process is None else len(self.process.F)
        return (f"{self.tcp_frame} tool: {n_p + len(self.other.F)} triangles - process parts (gas nozzle + contact tip, "
                f"{', '.join(self.process_links) or '-'}): {n_p}, other parts: {len(self.other.F)} ({self.source})")


def _sub_body(body: ToolBody, fmask: np.ndarray, source: str) -> Optional[ToolBody]:
    if not fmask.any():
        return None
    return tool_body_from_triangles(body.V[body.F[fmask]], tcp_frame=body.tcp_frame, source=source, links=body.links,
                                    face_link=None if body.face_link is None else body.face_link[fmask])


def tool_roles(body) -> ToolRoles:
    """The tool's parts by role (ToolRoles), from its collision geometry alone: the gas nozzle = of the parts the torch axis passes
    through (their outline seen along the axis encloses it - _axis_parts) the one reaching farthest along +z_TCP (its rim is the nozzle
    end, nozzle_end_mm); the process parts = the nozzle and every part lying within the nozzle's outline seen along the axis (in its
    bore: the contact tip). Everything else - also axis-enclosing parts outside that outline (the torch body and neck, rings on the
    neck) and parts beside the axis (the sensor and its guard) - is 'other'. No dimension is picked by hand. A ToolRoles is returned
    as it is; a tool without any part around the axis → ValueError."""
    if isinstance(body, ToolRoles):
        return body
    cached = body.__dict__.get("_roles_cache")
    if cached is not None:
        return cached
    lab, n_c, enc = _axis_parts(body)
    if not enc:
        raise ValueError(f"no part of the tool encloses the torch axis ({body.describe()}): no gas nozzle / contact tip to tell apart")
    nozzle = max(enc, key=lambda k: enc[k][0])
    eq = enc[nozzle][1]
    V = np.asarray(body.V, dtype=float)
    proc = {nozzle}
    for k in range(n_c):
        if k == nozzle:
            continue
        P = V[lab == k][:, :2]
        if len(P) and np.all(P @ eq[:, :2].T + eq[:, 2] <= 0.0):        # within the nozzle's outline seen along the axis
            proc.add(k)
    fproc = np.isin(lab[body.F[:, 0]], sorted(proc))
    p_links = tuple(sorted({body.links[i] for i in np.unique(body.face_link[fproc])})) if body.face_link is not None and body.links else ()
    how = ("process parts = the gas nozzle (of the tool parts the torch axis passes through, the one reaching farthest along +z_TCP: "
           f"rim {enc[nozzle][0] * 1e3:.2f} mm) and the {len(proc) - 1} part(s) within its outline seen along the axis (the contact "
           f"tip in its bore): {int(fproc.sum())} triangles; other parts: the rest ({int((~fproc).sum())} triangles)")
    roles = ToolRoles(process=_sub_body(body, fproc, f"{body.source} - process parts (gas nozzle + contact tip)"),
                      other=_sub_body(body, ~fproc, f"{body.source} - other parts"),
                      tcp_frame=body.tcp_frame, source=body.source, process_links=p_links, how=how)
    object.__setattr__(body, "_roles_cache", roles)
    return roles


def part_link(body: ToolBody, point_tcp_mm) -> str:
    """The URDF link of the tool part at a point (TCP frame, mm): the link of the nearest tool vertex ('' if not tracked)."""
    if point_tcp_mm is None or body.face_link is None or not body.links:
        return ""
    p = np.asarray(point_tcp_mm, dtype=float) * 1e-3
    i = int(np.argmin(np.linalg.norm(np.asarray(body.V, dtype=float) - p, axis=1)))
    f = int(np.flatnonzero((body.F == i).any(axis=1))[0])
    return str(body.links[int(body.face_link[f])])


def body_digest(body: ToolBody) -> str:
    """A digest of the tool geometry the clearance evaluates (vertices, faces, edges - everything clearance() reads; its caches derive
    from them). Cached on the body (it crosses to the worker process with the body)."""
    d = body.__dict__.get("_digest_cache")
    if d is None:
        import hashlib                                                   # noqa: PLC0415
        h = hashlib.sha256(b"ToolBody V F E")
        for a, t in ((body.V, np.float64), (body.F, np.int64), (body.E, np.int64)):
            a = np.ascontiguousarray(a, dtype=t)
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        d = h.hexdigest()
        object.__setattr__(body, "_digest_cache", d)
    return d


def nozzle_end_mm(body: ToolBody) -> float:
    """The weld torch's nozzle end on its axis [mm along +z_TCP from the TCP] from the tool's collision geometry (the robot's URDF,
    in the TCP frame; user 2026-10-13: the wire stick-out is measured from the gas nozzle end): the farthest point along +z_TCP of
    the tool parts the torch axis passes through - every connected part whose outline seen along the axis (its convex hull projected
    on the TCP xy plane) encloses the axis: the gas nozzle around the recessed contact tip, the contact tip, the torch body behind.
    Parts beside the axis (the sensor and its guard, which may reach farther along +z_TCP) do not count. The collision geometry
    carries no wire (the wire is the process stick-out), so the farthest such point is the nozzle rim (our torch's CAD: 1.2 mm beyond
    the contact-tip end = the TCP). ValueError when no part encloses the axis."""
    if isinstance(body, ToolRoles):
        body = body.process if body.process is not None else body.other
    _lab, _n, enc = _axis_parts(body)
    if not enc:
        raise ValueError(f"no part of the tool encloses the torch axis ({body.describe()}): no nozzle to measure the stick-out from")
    return max(z for z, _h in enc.values()) * 1e3


# --------------------------------------------------------------------------------------
# the specimen of a pass, as arrays
# --------------------------------------------------------------------------------------
@dataclass
class PassGeometry:
    """The specimen before one pass in the stations' section frames (mm): per solid (name, normals (N, k, 2), offsets (N, k),
    miters (N, k, 2), vertices (N, k, 2)); ylo / yhi (N,) = the plate ends along the seam relative to each station; A (N, 3, 3) rows
    (w, u, n): each station's section axes in the base (a curved seam turns them); origins (N, 3) [m]."""
    names: list
    planes: list
    ylo: np.ndarray
    yhi: np.ndarray
    A: np.ndarray
    origins: np.ndarray
    s_mm: np.ndarray


def pass_geometry(plan, wp) -> PassGeometry:
    return solids_geometry(plan.stations, pass_solids(plan, wp))


def solids_geometry(stations, solids) -> PassGeometry:
    """Section solids (weld_scene.SectionSolid, polygons at the given stations) → PassGeometry (the stations' section frames)."""
    st = list(stations)
    S = np.array([s.s_mm for s in st], dtype=float)
    A = np.stack([np.stack([s.w, s.u, s.n]) for s in st]).astype(float)
    planes = []
    for so in solids:
        Nrm, b, M = solid_planes(so.poly)
        planes.append((Nrm, b, M, np.asarray(so.poly, dtype=float)))
    so = solids[-1]
    return PassGeometry(names=[so_.name for so_ in solids], planes=planes, ylo=so.s_lo - S, yhi=so.s_hi - S, A=A,
                        origins=np.array([s.origin for s in st], dtype=float), s_mm=S)


def nearest_station(geo: PassGeometry, P_base) -> np.ndarray:
    """For points (M, 3) [m, base] the index of the station whose section plane is nearest along the seam (|(P - origin_i) . u_i|
    smallest) - the section a tool pose is evaluated in (D14 evaluates the tool at each station with its wire end on that station's
    section)."""
    P = np.asarray(P_base, dtype=float).reshape(-1, 3)
    U = geo.A[:, 1, :]                                                     # (N, 3) the seam axis u of each station
    d = np.abs(np.einsum("nj,mnj->mn", U, P[:, None, :] - geo.origins[None, :, :]))
    return np.argmin(d, axis=1)


def rows_geometry(geo: PassGeometry, station) -> PassGeometry:
    """The geometry with one row per pose: row j = the section of station[j] (its index in geo; nearest_station) - its solids' planes,
    the plate ends relative to it, its frame. clearance() evaluates row j's pose against that section's prisms."""
    st = np.asarray(station, dtype=int).reshape(-1)
    return PassGeometry(names=list(geo.names), planes=[tuple(np.asarray(a)[st] for a in pl) for pl in geo.planes],
                        ylo=np.asarray(geo.ylo, dtype=float)[st], yhi=np.asarray(geo.yhi, dtype=float)[st], A=np.asarray(geo.A)[st],
                        origins=np.asarray(geo.origins, dtype=float)[st], s_mm=np.asarray(geo.s_mm, dtype=float)[st])


def rows_poses(rows: PassGeometry, R_base_tcp: np.ndarray, P_base_tcp) -> tuple:
    """TCP poses at ONE base rotation (R_base_tcp (3, 3), P_base_tcp (n, 3) [m]) in the section frames of the rows (rows_geometry) →
    (R_sec (n, 3, 3): each row's section <- TCP rotation - the measured sections turn, so it differs between rows -, P_sec (n, 3) [mm])."""
    Rb = np.asarray(R_base_tcp, dtype=float)
    P = np.asarray(P_base_tcp, dtype=float).reshape(-1, 3)
    R_sec = np.einsum("nij,jk->nik", rows.A, Rb)
    P_sec = np.einsum("nij,nj->ni", rows.A, P - rows.origins) * 1e3
    return R_sec, P_sec


def poses_clearance(body: ToolBody, geo: PassGeometry, R_base_tcp: np.ndarray, P_base_tcp, station, *,
                    threshold: Optional[float] = None, order: Optional[np.ndarray] = None) -> dict:
    """Clearance [mm] of the tool at TCP poses with ONE base rotation (a straight move at constant orientation - the delivered weld
    line that D14 checks): pose j = (R_base_tcp, P_base_tcp[j] m) evaluated against
    the specimen in the section of station[j] (its index in geo; nearest_station) - each station's prism over the plate ends. The
    rotation relative to a section differs between stations (the measured sections turn): one clearance() call with one rotation per
    row (rows_geometry / rows_poses). → the worst pose's record (clearance: clearance_mm, solid, kind, point_tcp_mm, below,
    intersects, s_mm = that station's, station = its index in geo) + 'sample' (the pose's index j) + n_samples; threshold as in
    clearance (stops at the first pose below; order = the poses to try first)."""
    P = np.asarray(P_base_tcp, dtype=float).reshape(-1, 3)
    st = np.asarray(station, dtype=int).reshape(-1)
    if len(st) != len(P):
        raise ValueError("one station index per pose")
    rows = rows_geometry(geo, st)
    R_sec, P_sec = rows_poses(rows, R_base_tcp, P)
    c = dict(clearance(body, rows, R_sec, P_sec, threshold=threshold, order=order))
    j = int(c.get("station", -1))
    c["sample"] = j if j >= 0 else 0
    c["station"] = int(st[j]) if j >= 0 else -1
    c["n_samples"] = int(len(P))
    return c


# --------------------------------------------------------------------------------------
# clearance
# --------------------------------------------------------------------------------------
def _g(P: np.ndarray, Nrm: np.ndarray, b: np.ndarray, ylo: float, yhi: float) -> np.ndarray:
    """g of points P (..., 3) [x, y, z mm] for one solid at one station (module docstring)."""
    g2 = np.max(P[..., None, 0] * Nrm[:, 0] + P[..., None, 2] * Nrm[:, 1] - b, axis=-1)
    return np.maximum(g2, np.maximum(P[..., 1] - yhi, ylo - P[..., 1]))


def _face_values(Q: np.ndarray, Nrm: np.ndarray, b: np.ndarray, ylo: np.ndarray, yhi: np.ndarray) -> np.ndarray:
    """The face planes' values (the solid's polygon edges, then its two end planes: n_f . p - b_f, p_y - yhi, ylo - p_y) at points Q
    (n, m, 3) [mm] of n rows, row r against its own section's solid (Nrm (n, k, 2), b (n, k), plate ends ylo / yhi (n,)) → (n, m,
    k + 2). Their max is g (_g row by row: the same arithmetic, element for element - max is exact)."""
    A = Q[..., None, 0] * Nrm[:, None, :, 0] + Q[..., None, 2] * Nrm[:, None, :, 1] - b[:, None, :]
    return np.concatenate([A, (Q[..., 1] - yhi[:, None])[..., None], (ylo[:, None] - Q[..., 1])[..., None]], axis=-1)


def _g_rows(Q: np.ndarray, Nrm: np.ndarray, b: np.ndarray, ylo: np.ndarray, yhi: np.ndarray) -> np.ndarray:
    """g of points Q (n, m, 3) [mm] of n rows (as _face_values) → (n, m): _g row by row."""
    return np.max(_face_values(Q, Nrm, b, ylo, yhi), axis=-1)


def _elements_lb(F: np.ndarray, g: np.ndarray, idx: np.ndarray, L: np.ndarray, above: float) -> np.ndarray:
    """Per row, a lower bound of g over the elements (triangles / edges: idx (e, c) their vertices' indices in the rows' points; F
    (n, m, f) the points' face values - _face_values -, g (n, m) their g; L (e,) the longest edge) good enough to tell whether it
    exceeds 'above' → (n,) (where it does not, a value <= 'above'; only that comparison is used). Two lower bounds of an element's
    least g, both valid: its vertices' least g minus its longest edge (g is 1-Lipschitz); and - g being the max of the face planes'
    affine functions, the element the convex hull of its vertices - the max over the planes of the plane's least value at the
    vertices. The second only where the first does not already exceed 'above'."""
    n = g.shape[0]
    if not len(idx):
        return np.full(n, np.inf)
    G = g[:, idx]                                                                   # (n, e, c)
    crude = G[..., 0]
    for j_ in range(1, idx.shape[1]):
        crude = np.minimum(crude, G[..., j_])
    crude = crude - L[None, :]                                                      # (n, e)
    out = np.min(crude, axis=1)
    rr, ee = np.nonzero(crude <= above)
    if rr.size:
        Fe = F[rr[:, None], idx[ee]]                                                # (k, c, f)
        fmin = Fe[:, 0]
        for j_ in range(1, idx.shape[1]):
            fmin = np.minimum(fmin, Fe[:, j_])
        tight = np.maximum(crude[rr, ee], np.max(fmin, axis=-1))
        lb = np.full(n, np.inf)
        np.minimum.at(lb, rr, tight)
        hit = np.zeros(n, dtype=bool)
        hit[rr] = True
        out[hit] = lb[hit]                    # (the rows' other elements already exceed 'above')
    return out


_ROWS_BATCH_ELEMS = 1 << 21      # rows evaluated together in clearance(): about this many face values at a time (memory only - the
#                                  same values whatever the grouping)


def _events(name, Nrm, b, M, C, ylo, yhi, A, D, Ta, Tb, Tc, ends: bool, best: tuple, ridges: bool = True) -> tuple:
    """Edge / corner events of one solid at one station (module docstring (b), (b'), (c')) for the tool edges A + τD and triangles
    (Ta, Tb, Tc) [mm, section frame] → the best (r, solid, kind, point) so far."""
    if ridges and len(A):
        # (b) ridge lines along the seam (polygon vertex c + r m) on tool edges
        det = -M[None, :, 0] * D[:, None, 2] + M[None, :, 1] * D[:, None, 0]
        ax = A[:, None, 0] - C[None, :, 0]
        az = A[:, None, 2] - C[None, :, 1]
        ok = np.abs(det) > 1e-12
        dd = np.where(ok, det, 1.0)
        r = (-ax * D[:, None, 2] + az * D[:, None, 0]) / dd
        tau = (M[None, :, 0] * az - M[None, :, 1] * ax) / dd
        ok &= (tau >= 0.0) & (tau <= 1.0) & (r < best[0])
        if ok.any():
            ei, kj = np.nonzero(ok)
            P = A[ei] + tau[ei, kj, None] * D[ei]
            P[:, 0] = C[kj, 0] + r[ei, kj] * M[kj, 0]
            P[:, 2] = C[kj, 1] + r[ei, kj] * M[kj, 1]
            rv = r[ei, kj]
            valid = _g(P, Nrm, b, ylo, yhi) <= rv + _EPS_MM
            if valid.any():
                m = int(np.argmin(np.where(valid, rv, np.inf)))
                best = (float(rv[m]), name, "ridge", P[m])
    if not ends:
        return best
    for s_, y_end in ((1.0, yhi), (-1.0, ylo)):
        if len(A):
            # (b') edges of a grown end face on tool edges: n_e·p_xz = b_e + r, s·p_y = s·y_end + r
            nD = D[:, None, 0] * Nrm[None, :, 0] + D[:, None, 2] * Nrm[None, :, 1]
            nA = A[:, None, 0] * Nrm[None, :, 0] + A[:, None, 2] * Nrm[None, :, 1]
            den = nD - s_ * D[:, None, 1]
            ok = np.abs(den) > 1e-12
            tau = (b[None, :] - s_ * y_end - nA + s_ * A[:, None, 1]) / np.where(ok, den, 1.0)
            r = nA + tau * nD - b[None, :]
            ok &= (tau >= 0.0) & (tau <= 1.0) & (r < best[0])
            if ok.any():
                ei, kj = np.nonzero(ok)
                P = A[ei] + tau[ei, kj, None] * D[ei]
                rv = r[ei, kj]
                valid = _g(P, Nrm, b, ylo, yhi) <= rv + _EPS_MM
                if valid.any():
                    m = int(np.argmin(np.where(valid, rv, np.inf)))
                    best = (float(rv[m]), name, "end edge", P[m])
        if len(Ta):
            # (c') corners of a grown end face on tool triangles: (c, y_end) + r (m, s) = Ta + β (Tb - Ta) + γ (Tc - Ta)
            k = C.shape[0]
            P0 = np.stack([C[:, 0], np.full(k, y_end), C[:, 1]], axis=-1)
            Vel = np.stack([M[:, 0], np.full(k, s_), M[:, 1]], axis=-1)
            e1, e2 = Tb - Ta, Tc - Ta
            rhs = Ta[:, None, :] - P0[None, :, :]
            c1 = -np.broadcast_to(e1[:, None, :], rhs.shape)
            c2 = -np.broadcast_to(e2[:, None, :], rhs.shape)
            c0 = np.broadcast_to(Vel[None, :, :], rhs.shape)
            x12 = np.cross(c1, c2)
            det = np.einsum("fkj,fkj->fk", c0, x12)
            ok = np.abs(det) > 1e-12
            dd = np.where(ok, det, 1.0)
            r = np.einsum("fkj,fkj->fk", rhs, x12) / dd
            be = np.einsum("fkj,fkj->fk", c0, np.cross(rhs, c2)) / dd
            ga = np.einsum("fkj,fkj->fk", c0, np.cross(c1, rhs)) / dd
            ok &= (be >= 0.0) & (ga >= 0.0) & (be + ga <= 1.0) & (r < best[0])
            if ok.any():
                fi, kj = np.nonzero(ok)
                rv = r[fi, kj]
                P = P0[kj] + rv[:, None] * Vel[kj]
                valid = _g(P, Nrm, b, ylo, yhi) <= rv + _EPS_MM
                if valid.any():
                    m = int(np.argmin(np.where(valid, rv, np.inf)))
                    best = (float(rv[m]), name, "end corner", P[m])
    return best


def _cells(body: ToolBody) -> tuple:
    """The vertices grouped in grid cells (about 256 per cell; a speed device only - every bound below holds for any grouping):
    (cell of each vertex (n,), cell centres (c, 3) [m], cell radii (c,) [m]). Cached on the body."""
    c = body.__dict__.get("_cell_cache")
    if c is None:
        V = np.asarray(body.V, dtype=float)
        lo, hi = V.min(axis=0), V.max(axis=0)
        n_cells = max(1.0, len(V) / 256.0)
        size = max(float(np.prod(np.maximum(hi - lo, 1e-6))) / n_cells, 1e-12) ** (1.0 / 3.0)
        key = np.floor((V - lo) / size).astype(np.int64)
        _, cell = np.unique(key, axis=0, return_inverse=True)
        cell = cell.ravel()
        nc = int(cell.max()) + 1
        cnt = np.bincount(cell, minlength=nc).astype(float)
        cen = np.stack([np.bincount(cell, weights=V[:, j], minlength=nc) for j in range(3)], axis=1) / cnt[:, None]
        rad = np.zeros(nc)
        np.maximum.at(rad, cell, np.linalg.norm(V - cen[cell], axis=1))
        c = (cell, cen, rad)
        object.__setattr__(body, "_cell_cache", c)
    return c


def _lengths(body: ToolBody) -> tuple:
    """Edge lengths and the longest edge of each triangle [mm]; bounding spheres of the edges and triangles (centres [m], radii
    [mm]) - cached on the body."""
    c = body.__dict__.get("_len_cache")
    if c is None:
        V = body.V * 1e3
        Le = np.linalg.norm(V[body.E[:, 1]] - V[body.E[:, 0]], axis=1)
        F = body.F
        Lf = np.max(np.stack([np.linalg.norm(V[F[:, j]] - V[F[:, (j + 1) % 3]], axis=1) for j in range(3)]), axis=0)
        ce = 0.5 * (body.V[body.E[:, 0]] + body.V[body.E[:, 1]])
        cf = body.V[F].mean(axis=1)
        rf = np.max(np.linalg.norm(V[F] - cf[:, None, :] * 1e3, axis=2), axis=1)
        c = (Le, Lf, ce, 0.5 * Le, cf, rf)
        object.__setattr__(body, "_len_cache", c)
    return c


def _vnorm(body: ToolBody) -> np.ndarray:
    """Each vertex's distance from the TCP [mm] - cached on the body."""
    vn = body.__dict__.get("_vnorm_cache")
    if vn is None:
        vn = np.linalg.norm(np.asarray(body.V), axis=1) * 1e3
        object.__setattr__(body, "_vnorm_cache", vn)
    return vn


def warm_caches(body) -> None:
    """The tool's per-body caches that clearance() and the evaluation key build on first use (vertex cells, element lengths and
    spheres, vertex distances, the geometry digest), built now - e.g. once in a process before it forks its searches, which then
    share them instead of each building its own. A ToolRoles: each role's body."""
    for b in ([b_ for _r, b_ in body.roles()] if isinstance(body, ToolRoles) else [body]):
        _cells(b)
        _lengths(b)
        _vnorm(b)
        body_digest(b)


def _sat_hits(Ta, Tb, Tc, Nrm, C, ylo: float, yhi: float) -> np.ndarray:
    """Which triangles (f, 3) x3 [mm] intersect the convex prism of polygon C (k, 2) (outward normals Nrm) over y in [ylo, yhi] with a
    positive overlap (touching = clear): separating-axis test - the prism's face normals, the triangle's normal and the cross products
    of the triangle's edges with the prism's edge directions (the polygon's edges and the seam axis). Exact for convex sets."""
    f = len(Ta)
    if f == 0:
        return np.zeros(0, dtype=bool)
    k = C.shape[0]
    PV = np.concatenate([np.stack([C[:, 0], np.full(k, ylo), C[:, 1]], axis=-1),
                         np.stack([C[:, 0], np.full(k, yhi), C[:, 1]], axis=-1)])                  # (2k, 3)
    T = np.stack([Ta, Tb, Tc], axis=1)                                                          # (f, 3, 3)
    eps = 1e-9
    sep = np.zeros(f, dtype=bool)
    # the prism's faces
    Af = np.concatenate([np.stack([Nrm[:, 0], np.zeros(k), Nrm[:, 1]], axis=-1), [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]])
    pP = PV @ Af.T                                                                              # (2k, na)
    pT = T @ Af.T                                                                               # (f, 3, na)
    sep |= np.any((pT.max(axis=1) <= pP.min(axis=0) + eps) | (pT.min(axis=1) >= pP.max(axis=0) - eps), axis=1)
    # the triangle's normal and the edge cross products (per triangle)
    E = np.stack([Tb - Ta, Tc - Tb, Ta - Tc], axis=1)                                          # (f, 3, 3)
    Dp = np.concatenate([np.stack([np.roll(C, -1, axis=0)[:, 0] - C[:, 0], np.zeros(k), np.roll(C, -1, axis=0)[:, 1] - C[:, 1]],
                                  axis=-1), [[0.0, 1.0, 0.0]]])                                # (k + 1, 3)
    Ax = np.concatenate([np.cross(E[:, 0], E[:, 1])[:, None, :],
                         np.cross(E[:, :, None, :], Dp[None, None, :, :]).reshape(f, -1, 3)], axis=1)   # (f, 1 + 3(k+1), 3)
    nrm = np.linalg.norm(Ax, axis=-1)
    good = nrm > 1e-9
    Ax = Ax / np.where(good, nrm, 1.0)[..., None]
    pP = np.einsum("vi,fai->fav", PV, Ax)
    pT = np.einsum("fti,fai->fat", T, Ax)
    s2 = (pT.max(axis=2) <= pP.min(axis=2) + eps) | (pT.min(axis=2) >= pP.max(axis=2) - eps)
    sep |= np.any(s2 & good, axis=1)
    return ~sep


def clearance(body: ToolBody, geo: PassGeometry, R_sec: np.ndarray, P_sec: np.ndarray, *, threshold: Optional[float] = None,
              order: Optional[np.ndarray] = None) -> dict:
    """Clearance [mm] of the tool at N poses (rows of geo: a station's section each): R_sec the section ← TCP rotation - (3, 3) the
    same at every row, or (N, 3, 3) one per row (poses at one base rotation in several stations' sections, rows_poses) -, P_sec (N, 3)
    the TCP origin in each row's section frame [mm]. With rows' own rotations every bound below is taken with the rotation of row 0
    and widened by the farthest any tool point can sit from that place in another row (chord of the largest rotation angle between
    the rows x the point's distance from the TCP: exact bounds, nothing skipped that could come below); every value is computed with
    the row's own rotation.
      threshold given (>= 0): the feasibility test - does the tool reach into any solid grown by the threshold? Exact (separating
      axes against the grown prisms); stops at the first station that it does ('below' True; stations in `order`).
      threshold None: the clearance value - exact when the tool clears the specimen (the first contact as the solids grow: module
      docstring (a), (b), (b'), (c')); when the tool is inside the specimen ('intersects' True, checked by separating axes) the value
      is <= 0 (its depth is not resolved beyond the events found).
    → {clearance_mm, station, s_mm, solid, kind, point_tcp_mm (the tool point, TCP frame), below, intersects, n_stations}.
    Speed without approximation: g is 1-Lipschitz (unit face normals), so a triangle whose vertices all have g > t + its longest edge
    cannot reach below t; a vertex above every solid's top face by more than t cannot either; anything below t lies in the solid
    grown by t (its bounding box) and, for the end faces, within the solid's half width of the end plane. The rows' vertex values are
    computed together (the same arithmetic as row by row) and, per row and solid, a lower bound of every value its triangles and edges
    can give (_elements_lb): their stages (separating axes, contact events) are skipped where that bound is above everything that can
    still matter by more than the events' acceptance slack (2 _EPS_MM) - the threshold, or (value) thr0: a vertex value of some row
    that the vertex stage examines, or 0 (the intersection test's level), so the result is never above it. Such elements could only
    give larger values (each event's value is g at its contact point on the element, accepted within _EPS_MM) and no intersection
    (they lie outside the solid). The vertex stage runs for every row and solid as before and the stages that run see a superset of
    the elements they saw, so the record - value, place, kind, the first of equal values in row and solid order, the intersection -
    is the same as without the skipping. (The grouping of rows is a memory device: _ROWS_BATCH_ELEMS.)"""
    Rin = np.asarray(R_sec, dtype=float)
    per_row = Rin.ndim == 3
    Rs = Rin[0] if per_row else Rin                 # the bounds' rotation (row 0)
    V = np.asarray(body.V)
    nV = V.shape[0]
    N = P_sec.shape[0]
    if per_row:
        if Rin.shape[0] != N:
            raise ValueError("one rotation per row")
        # every row's rotation is within theta of Rs: a tool point v (TCP frame) sits at most 2 sin(theta / 2) |v| from its Rs place
        ca = np.clip((np.einsum("nij,ij->n", Rin, Rs) - 1.0) / 2.0, -1.0, 1.0)
        chord = 2.0 * math.sin(0.5 * float(np.max(np.arccos(ca))))
    else:
        chord = 0.0
    vn = _vnorm(body)
    slack_v = chord * vn                                                             # [mm] per vertex (0 for one rotation)
    Le, Lf, ce, re_, cf, rf_ = _lengths(body)
    tops = []
    for Nrm_, b_, _M, _C in geo.planes:
        jz = np.argmax(Nrm_[..., 1], axis=-1)
        n_top = np.take_along_axis(Nrm_, jz[:, None, None], axis=1)[:, 0]
        b_top = np.take_along_axis(b_, jz[:, None], axis=1)[:, 0]
        tops.append((n_top, b_top))
    flat = all(np.allclose(nt, [0.0, 1.0], atol=1e-12) for nt, _ in tops)
    cz = min(float(np.min(P_sec[:, 2] - bt)) for _nt, bt in tops) if flat else None
    cell, cen, rad = _cells(body)
    Q0 = np.full((nV, 3), np.nan)
    have = np.zeros(nV, dtype=bool)

    def q0(idx):                                   # the vertices' coordinates relative to the TCP [mm] at Rs (bounds), on demand
        idx = np.asarray(idx, dtype=int)
        need = idx[~have[idx]]
        if need.size:
            Q0[need] = (V[need] @ Rs.T) * 1e3
            have[need] = True
        return Q0[idx]

    def qrow(idx, i):                              # exact: the vertices relative to the TCP in row i's section frame [mm]
        if not per_row:
            return q0(idx)
        return (V[np.asarray(idx, dtype=int)] @ Rin[int(i)].T) * 1e3

    def rrow(i):
        return Rin[int(i)] if per_row else Rs
    if flat:                                       # lower bound per vertex: the cell bound, the vertex's own where the cell may reach
        lb_c = ((cen @ Rs.T)[:, 2] - rad) * 1e3 + cz - chord * (np.linalg.norm(cen, axis=1) + rad) * 1e3
        lb_v = lb_c[cell].copy()
    else:                                          # a solid without a face straight up (a cap tangent polygon): every vertex
        q = q0(np.arange(nV))
        lb_v = np.full(nV, np.inf)
        for nt, bt in tops:
            for i in range(N):
                lb_v = np.minimum(lb_v, (q[:, 0] + P_sec[i, 0]) * nt[i, 0] + (q[:, 2] + P_sec[i, 2]) * nt[i, 1] - bt[i])
        lb_v = lb_v - slack_v
    order = np.arange(N) if order is None else np.asarray(order, dtype=int)

    def refine(thr):                               # the vertices of the cells that may reach below thr get their own bound
        if not flat:
            return
        sel = np.flatnonzero((lb_v <= thr + _EPS_MM) & ~have)
        if sel.size:
            lb_v[sel] = q0(sel)[:, 2] + cz - slack_v[sel]
    out = {"clearance_mm": math.inf, "station": -1, "s_mm": float("nan"), "solid": "", "kind": "", "point_tcp_mm": None,
           "below": False, "intersects": False, "n_stations": 0}

    def result(r, solid, kind, P, i, **kw):
        pt = None if P is None else (rrow(i).T @ (np.asarray(P, dtype=float) - P_sec[i]))
        out.update(clearance_mm=float(r), station=int(i), s_mm=float(geo.s_mm[i]), solid=solid, kind=kind,
                   point_tcp_mm=None if pt is None else [float(v) for v in pt], **kw)
        return out
    if threshold is None:                          # an upper bound from the lowest vertices first
        lo_c = np.argsort(lb_c)[:4] if flat else None
        if flat:
            refine(float(np.max(lb_c[lo_c])))
        low = np.argpartition(lb_v, min(255, nV - 1))[:256]
        ub = math.inf
        Ql = np.stack([qrow(low, i) + P_sec[i] for i in range(N)])                  # (every row at once: the same values)
        for (Nrm_, b_, _M, _C) in geo.planes:
            ub = min(ub, float(np.min(_g_rows(Ql, Nrm_, b_, np.asarray(geo.ylo, dtype=float), np.asarray(geo.yhi, dtype=float)))))
        thr0 = max(ub, 0.0)                        # the intersection test at 0 needs the elements that reach 0 too
    else:
        thr0 = float(threshold)
        refine(thr0)
        low = np.flatnonzero(lb_v <= thr0 + _EPS_MM)
        for i in (order[:1] if low.size else []):  # every vertex that can be below, at the first station: one inside decides it
            i = int(i)
            Qi = qrow(low, i) + P_sec[i]
            for name, (Nrm_, b_, _M, _C) in zip(geo.names, geo.planes):
                gv = _g(Qi, Nrm_[i], b_[i], float(geo.ylo[i]), float(geo.yhi[i]))
                j = int(np.argmin(gv))
                if gv[j] < thr0:
                    out["n_stations"] = 1
                    return result(gv[j], name, "vertex", Qi[j], i, below=True, intersects=bool(gv[j] < 0.0))
    refine(thr0)
    if flat:                                       # elements that can come below: their bounding spheres
        F0 = np.flatnonzero((cf @ Rs[2]) * 1e3 - rf_ + cz - chord * (np.linalg.norm(cf, axis=1) * 1e3 + rf_) <= thr0 + _EPS_MM)
        E0 = np.flatnonzero((ce @ Rs[2]) * 1e3 - re_ + cz - chord * (np.linalg.norm(ce, axis=1) * 1e3 + re_) <= thr0 + _EPS_MM)
    else:
        F0 = np.flatnonzero(np.min(lb_v[body.F], axis=1) - Lf <= thr0 + _EPS_MM)
        E0 = np.flatnonzero(np.min(lb_v[body.E], axis=1) - Le <= thr0 + _EPS_MM)
    vk = np.unique(np.concatenate([np.flatnonzero(lb_v <= thr0 + _EPS_MM), body.E[E0].ravel(), body.F[F0].ravel()]))
    pos = np.full(Q0.shape[0], -1)
    pos[vk] = np.arange(vk.size)
    E0l, F0l = pos[body.E[E0]], pos[body.F[F0]]
    Le0, Lf0 = Le[E0], Lf[F0]
    Qk0 = None if per_row else q0(vk)
    best_all = (math.inf, "", "", None, -1)
    inter = None                                   # (solid, triangle point, station) of an intersection at 0 (value mode)
    # the rows' vertex values per solid and the lower bound of their triangles' and edges' values (docstring), in groups of rows in
    # the order they are visited (the feasibility test usually stops at the first row)
    n_s, m_v = len(geo.planes), len(vk)
    Qv_all = np.zeros((N, m_v, 3))
    gv_all = np.zeros((n_s, N, m_v))
    elb = np.full((n_s, N), -np.inf)
    jmin = np.zeros((n_s, N), dtype=np.int64)
    ready = np.zeros(N, dtype=bool)
    ylo_a, yhi_a = np.asarray(geo.ylo, dtype=float), np.asarray(geo.yhi, dtype=float)
    k_max = max([int(np.asarray(pl[1]).shape[-1]) for pl in geo.planes] + [1]) + 2
    batch = max(1, _ROWS_BATCH_ELEMS // max(1, m_v * k_max, 3 * len(F0l), 2 * len(E0l)))

    ev = np.unique(np.concatenate([F0l.ravel(), E0l.ravel()])) if (len(F0l) or len(E0l)) else np.zeros(0, dtype=np.int64)

    def bounds(s_, rows, F_):                      # elb of the rows
        g_ = gv_all[s_, rows]
        # an element's bound is <= its vertices' least g: a row with an element vertex at or below skip_above keeps its stages
        # (its bound is that vertex's g); only the others need the element bounds
        lo_v = np.min(g_[:, ev], axis=1) if ev.size else np.full(len(rows), np.inf)
        elb[s_, rows] = lo_v
        need = lo_v > skip_above
        if need.any():
            F_n, g_n = F_[need], g_[need]
            elb[s_, rows[need]] = np.minimum(_elements_lb(F_n, g_n, F0l, Lf0, skip_above), _elements_lb(F_n, g_n, E0l, Le0, skip_above))

    def prepare(rows):
        rows = np.asarray([r_ for r_ in rows if not ready[r_]], dtype=np.int64)
        if not rows.size or not m_v:
            ready[rows] = True
            return
        Qv_all[rows] = np.stack([(qrow(vk, r_) if per_row else Qk0) + P_sec[r_] for r_ in rows])
        for s_, (Nrm_, b_, _M, _C) in enumerate(geo.planes):
            F_ = _face_values(Qv_all[rows], Nrm_[rows], b_[rows], ylo_a[rows], yhi_a[rows])
            g_ = np.max(F_, axis=-1)
            gv_all[s_, rows] = g_
            jmin[s_, rows] = np.argmin(g_, axis=1)
            bounds(s_, rows, F_)
        ready[rows] = True
    # what can still matter: the threshold (the feasibility test), or (value) thr0 - a vertex value of some row (or 0, the
    # intersection test's level) the result cannot exceed
    skip_above = (float(threshold) if threshold is not None else thr0) + 2.0 * _EPS_MM
    for i_o, i in enumerate(order):
        i = int(i)
        if not ready[i]:
            prepare(order[i_o:i_o + batch])
        ylo, yhi = float(geo.ylo[i]), float(geo.yhi[i])
        best = (math.inf, "", "", None)
        Qv = Qv_all[i]
        out["n_stations"] += 1
        for s_, (name, (Nrm_, b_, M_, C_)) in enumerate(zip(geo.names, geo.planes)):
            if not len(vk):
                break
            Nrm, b, M, C = Nrm_[i], b_[i], M_[i], C_[i]
            gv = gv_all[s_, i]
            j = int(jmin[s_, i])
            if gv[j] < best[0]:
                best = (float(gv[j]), name, "vertex", Qv[j].copy())
            if elb[s_, i] > skip_above:          # no triangle / edge of this solid here can matter (docstring): their stages skipped
                continue
            t_test = float(threshold) if threshold is not None else 0.0          # the separating-axis test: grown by t_test
            t_cull = float(threshold) if threshold is not None else min(thr0, best_all[0], best[0])
            t_cull = max(t_cull, t_test)
            fk = np.flatnonzero(np.min(gv[F0l], axis=1) - Lf0 <= t_cull + _EPS_MM) if len(F0l) else np.zeros(0, dtype=int)
            G = C + t_cull * M
            lo_x, hi_x, lo_z, hi_z = G[:, 0].min(), G[:, 0].max(), G[:, 1].min(), G[:, 1].max()

            def in_box(P_lo, P_hi):
                return ((P_hi[:, 0] >= lo_x - _EPS_MM) & (P_lo[:, 0] <= hi_x + _EPS_MM) & (P_hi[:, 2] >= lo_z - _EPS_MM)
                        & (P_lo[:, 2] <= hi_z + _EPS_MM) & (P_hi[:, 1] >= ylo - t_cull - _EPS_MM)
                        & (P_lo[:, 1] <= yhi + t_cull + _EPS_MM))
            Fa, Fb, Fc = (Qv[F0l[fk, j_]] for j_ in range(3))
            Flo, Fhi = np.minimum(np.minimum(Fa, Fb), Fc), np.maximum(np.maximum(Fa, Fb), Fc)
            mf = in_box(Flo, Fhi)
            Fa, Fb, Fc, Flo, Fhi = Fa[mf], Fb[mf], Fc[mf], Flo[mf], Fhi[mf]
            # exact intersection with the solid grown by t_test
            hit = _sat_hits(Fa, Fb, Fc, Nrm, C + t_test * M, ylo - t_test, yhi + t_test)
            if hit.any():
                h = int(np.flatnonzero(hit)[0])
                Ptri = np.stack([Fa[h], Fb[h], Fc[h]])
                gt = _g(Ptri, Nrm, b, ylo, yhi)
                if threshold is not None:
                    v_ = min(float(gt.min()), float(threshold) - _EPS_MM)
                    best_all = (v_, name, "triangle", Ptri[int(np.argmin(gt))], i)
                    r, solid, kind, P, ii = best_all
                    return result(r, solid, kind, P, ii, below=True, intersects=bool(t_test <= 0.0 or gt.min() < 0.0))
                if inter is None:
                    inter = (name, Ptri[int(np.argmin(gt))], i)
            if threshold is not None:
                continue
            # value: the first contact as the solid grows (exact while the tool clears it)
            ek = np.flatnonzero(np.min(gv[E0l], axis=1) - Le0 <= t_cull + _EPS_MM) if len(E0l) else np.zeros(0, dtype=int)
            Ae, Be = Qv[E0l[ek, 0]], Qv[E0l[ek, 1]]
            me = in_box(np.minimum(Ae, Be), np.maximum(Ae, Be))
            Ae, Be = Ae[me], Be[me]
            rho = 0.5 * min(float(np.ptp(C[:, 0])), float(np.ptp(C[:, 1])))       # >= the inradius: events have r >= -rho
            sel_e = ((np.maximum(Ae[:, 1], Be[:, 1]) >= yhi - rho - _EPS_MM) & (np.minimum(Ae[:, 1], Be[:, 1]) <= yhi + t_cull + _EPS_MM)) | \
                    ((np.minimum(Ae[:, 1], Be[:, 1]) <= ylo + rho + _EPS_MM) & (np.maximum(Ae[:, 1], Be[:, 1]) >= ylo - t_cull - _EPS_MM))
            sel_f = (((Fhi[:, 1] >= yhi - rho - _EPS_MM) & (Flo[:, 1] <= yhi + t_cull + _EPS_MM))
                     | ((Flo[:, 1] <= ylo + rho + _EPS_MM) & (Fhi[:, 1] >= ylo - t_cull - _EPS_MM)))
            best = _events(name, Nrm, b, M, C, ylo, yhi, Ae, Be - Ae, Fa[:0], Fb[:0], Fc[:0], False, best)
            if sel_e.any() or sel_f.any():
                best = _events(name, Nrm, b, M, C, ylo, yhi, Ae[sel_e], Be[sel_e] - Ae[sel_e], Fa[sel_f], Fb[sel_f], Fc[sel_f], True,
                               best, ridges=False)
        if best[0] < best_all[0]:
            best_all = (*best, i)
        if threshold is not None and best[0] < float(threshold):
            r, solid, kind, P, ii = best_all
            return result(r, solid, kind, P, ii, below=True, intersects=bool(r < 0.0))
    r, solid, kind, P, i = best_all
    if inter is not None and not r <= 0.0:         # inside the specimen, depth not resolved by the events: 0 (an upper bound)
        name, P, i = inter
        return result(0.0, name, "inside (depth not resolved)", P, i, below=False, intersects=True)
    if i >= 0:
        result(r, solid, kind, P, i, intersects=bool(inter is not None or r < 0.0))
    elif threshold is None:
        out["clearance_mm"] = float(thr0)
    out["below"] = bool(threshold is not None and out["clearance_mm"] < float(threshold))
    return out
