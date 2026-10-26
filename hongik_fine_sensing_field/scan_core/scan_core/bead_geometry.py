"""Convex bead cross-sections in the groove (user + KIRO 2026-10-13: beads are convex, not flat-filled layers) - the ONE bead geometry
of the weld plan: the layer plan's heights, the 'current surface' the aims meet, the D14 weld scene (scan_core.weld_scene), the plan
outputs and figures all come from the beads placed here.

Section frame (bead_fill / weld_plan): x along w from the root-gap centre, z along the plate normal from the backing-bar top [mm].

Profile        the section's current upper surface: a polyline with x non-decreasing (a vertical face = consecutive points with the
               same x - the square wall, the root faces, a bead's cliff), the plates' tops beyond the groove edges, the backing-bar top
               in the gap (bare_profile); each bead raises it.
Bead           the pass's deposit (its mass-balance area a, bead_model) between the current surface and a convex free surface: the
               circular arc over the toe chord with height kappa_p x chord above it (the free bead's shape - bead_model: the arc fits
               GMAW sections at wire-feed / travel ratios above 12.5, Xiong et al. 2013; kappa_p = the bead model's crown ratio of the
               pass, = tan(toe / 2) / 2 of WeldProcess.bead_toe_angle_deg without weave, the same area over the woven width with weave - the one shape
               parameter). The toes lie on the current surface; the bead fills the lowest region it wets:
  pool         the deposit levels (1G: gravity) - the toe chord is level at z, its ends where the surface rises through z on each side of
               the aim (the walls, earlier beads' flanks); the pool spreads over a hump under the aim (the aim on an earlier crown) and
               merges with the next pocket when it rises over the ridge between them (the multi-pass groove filling models fill the
               groove bottom-up the same way - equal-area layers - with flat tops; here the top is the free bead's arc);
  spill        when the area ends while the pool spills over a ridge or a flat at its level (a plateau - a flat floor wider than the
               bead), the open side(s) extend at that level by equal lengths from the aim side (a flat floor: the free bead centred on
               its aim, toes on the floor; a ridge: the spill front);
  climb, over  the pool reaches the lower groove top edge (the plate top): the toe stays on that edge while the other toe climbs its
               wall to its own edge (hi-lo), then both toes move out over the plate tops by equal lengths - the chord from toe to toe
               (tilted by the hi-lo; the plates' own heights), the arc over it (a cap; a fill that overflows the groove);
  band         a bead of a multi-bead layer or of a several-pass cap fills its band (the plan's cut lines act as walls - a provisional
               rule for layouts KIRO's practice does not cover).
Convex solid   the convex hull of the bead's region with its top replaced by tangent lines of the arc (the hull contains the bead; the
               extra part lies below the surface: earlier deposits or the plates) - one convex polygon per station for the weld scene.

Calibration (later, no fitting now): post-weld profiles of KIRO's welds per pass give each bead's toes and crown - crown_ratio_from_
profiles measures kappa_p (height over the toe chord / chord; toe angle 2 atan(2 kappa_p)); the bead model (bead_toe_angle_deg, or a data-based BeadModel with
crown_ratio(p)) takes it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .bead_fill import wall_x_mm

__all__ = ["Profile", "BeadSection", "bare_profile", "place_bead", "pool_level", "pool_cuts", "crown_height_mm", "crown_ratio_from_profiles",
           "groove_edges_mm", "PHASES", "TANGENT_POINTS", "BEAD_RULE"]

PHASES = ("pool", "spill", "climb", "over", "band")
# the tangent polygon of the arc (the scene's convex solid): points uniform along the arc + the apex - numerical resolution (the
# polygon lies off the arc by at most R (1 / cos(theta / TANGENT_POINTS) - 1), theta = 2 atan(2 kappa): 0.013 mm for a 10 mm, kappa 0.3
# bead)
TANGENT_POINTS = 16
BEAD_RULE = ("convex bead: the pass's mass-balance area between the current surface and a circular arc over its toe chord with height "
             "kappa_p x chord (the free bead's shape; kappa_p = the bead model's crown ratio: tan(toe / 2) / 2 of the bead toe angle, "
             "flatter with weave); "
             "toes on the current surface; the bead fills the lowest region it wets - a level pool from the aim (over a hump under the aim, "
             "merging over ridges), spilling at a flat or ridge, then over the plate tops (the toe chord from plate to plate)")


# --------------------------------------------------------------------------------------
# the arc
# --------------------------------------------------------------------------------------
def _arc(c: float, kappa: float) -> tuple:
    s = float(kappa) * float(c)
    if s <= 0.0 or c <= 0.0:
        return 0.0, math.inf
    return s, (0.25 * c * c + s * s) / (2.0 * s)


def crown_height_mm(u, c: float, kappa: float) -> np.ndarray:
    """Height of the arc above its chord at u in [0, c] from the left toe (chord c, height kappa c at the middle)."""
    u = np.asarray(u, dtype=float)
    s, R = _arc(c, kappa)
    if s <= 0.0:
        return np.zeros_like(u)
    v = np.clip(u, 0.0, c) - 0.5 * c
    return np.sqrt(np.maximum(R * R - v * v, 0.0)) - (R - s)


def _crown_slope(u, c: float, kappa: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    s, R = _arc(c, kappa)
    if s <= 0.0:
        return np.zeros_like(u)
    v = np.clip(u, 0.0, c) - 0.5 * c
    return -v / np.sqrt(np.maximum(R * R - v * v, 1e-300))


# --------------------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------------------
@dataclass
class Profile:
    """The section's upper surface: polyline x (non-decreasing; equal x = a vertical face), z [mm]. edges = the bare specimen's groove
    top edges x (left, right) - the plates' tops start there."""
    x: np.ndarray
    z: np.ndarray
    edges: tuple = (float("nan"), float("nan"))

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        if self.x.shape != self.z.shape or self.x.ndim != 1 or self.x.size < 2:
            raise ValueError("profile: x and z must be 1-D of the same length >= 2")
        if np.any(np.diff(self.x) < 0.0):
            raise ValueError("profile: x must be non-decreasing")

    def copy(self) -> "Profile":
        return Profile(self.x.copy(), self.z.copy(), tuple(self.edges))

    def at(self, xq) -> np.ndarray:
        """Height of the upper surface at x (the top of a vertical face where x is one) [mm]; NaN outside the profile."""
        xq = np.asarray(xq, dtype=float)
        X, Z = self.x, self.z
        i0 = np.searchsorted(X, xq, side="left")
        i1 = np.searchsorted(X, xq, side="right")
        out = np.full(xq.shape, np.nan)
        exact = i1 > i0
        if np.any(exact):
            # the highest point at that x (a vertical face's top)
            for idx in np.flatnonzero(exact.ravel()):
                a, b = int(i0.ravel()[idx]), int(i1.ravel()[idx])
                out.ravel()[idx] = float(np.max(Z[a:b]))
        inner = ~exact & (i0 > 0) & (i0 < X.size)
        if np.any(inner):
            j = i0[inner]
            x0, x1, z0, z1 = X[j - 1], X[j], Z[j - 1], Z[j]
            out[inner] = z0 + (z1 - z0) * (xq[inner] - x0) / (x1 - x0)
        return out

    def edge_heights(self) -> tuple:
        """The current surface at the groove top edges (left, right) [mm]."""
        return float(self.at(self.edges[0])), float(self.at(self.edges[1]))

    def insert(self, xs) -> "Profile":
        """A copy with points added at x (on the surface; where a point with that x exists - a vertical face too - none is added)."""
        xs = np.unique(np.asarray([float(v) for v in np.atleast_1d(xs)], dtype=float))
        xs = xs[(xs > self.x[0]) & (xs < self.x[-1])]
        i0 = np.searchsorted(self.x, xs, side="left")
        new = self.x[np.minimum(i0, self.x.size - 1)] != xs
        xs, i0 = xs[new], i0[new]
        if xs.size == 0:
            return Profile(self.x.copy(), self.z.copy(), tuple(self.edges))
        zs = self.z[i0 - 1] + (self.z[i0] - self.z[i0 - 1]) * (xs - self.x[i0 - 1]) / (self.x[i0] - self.x[i0 - 1])
        return Profile(np.insert(self.x, i0, xs), np.insert(self.z, i0, zs), tuple(self.edges))

    def span(self, x0: float, x1: float) -> tuple:
        """The polyline over [x0, x1] (x0 < x1 inside the profile): points (x, z) from the inner side of x0 to the inner side of x1 (at a
        vertical face the point on the [x0, x1] side)."""
        X, Z = self.x, self.z
        i0 = int(np.searchsorted(X, x0, side="right")) - 1
        i1 = int(np.searchsorted(X, x1, side="left"))
        i0, i1 = max(i0, 0), min(i1, X.size - 1)
        z0 = Z[i0] if X[i0] == x0 or i0 + 1 >= X.size else Z[i0] + (Z[i0 + 1] - Z[i0]) * (x0 - X[i0]) / (X[i0 + 1] - X[i0])
        z1 = Z[i1] if X[i1] == x1 or i1 == 0 else Z[i1 - 1] + (Z[i1] - Z[i1 - 1]) * (x1 - X[i1 - 1]) / (X[i1] - X[i1 - 1])
        return (np.concatenate([[x0], X[i0 + 1:i1], [x1]]), np.concatenate([[z0], Z[i0 + 1:i1], [z1]]))

    def area_below(self, x0: float, x1: float, z: float) -> float:
        """Area between the level z and the surface where the surface is below z, over [x0, x1] [mm^2] (exact on the polyline)."""
        xs, f = self.span(x0, x1)
        return _pos_area(xs, z - f)


def _pos_area(x: np.ndarray, d: np.ndarray) -> float:
    """Integral of max(d, 0) over the polyline (x, d) (d linear on each segment; zero-width segments add nothing)."""
    x, d = np.asarray(x, dtype=float), np.asarray(d, dtype=float)
    dx = np.diff(x)
    a, b = d[:-1], d[1:]
    both = (a >= 0.0) & (b >= 0.0)
    out = np.where(both, 0.5 * (a + b) * dx, 0.0)
    mix = (a > 0.0) != (b > 0.0)
    mix &= ~both
    if np.any(mix):
        p = np.maximum(a[mix], b[mix])
        q = np.abs(a[mix] - b[mix])
        out[mix] = 0.5 * p * p / np.where(q > 0.0, q, 1.0) * dx[mix]
    return float(np.sum(out))


def groove_edges_mm(spec) -> tuple:
    """(x_left_edge, z_left_top, x_right_edge, z_right_top): the groove top edges of a section (GrooveSpec) - the left plate top at t,
    the right at t + hi-lo (weld_scene.plate_polys)."""
    t = float(spec.thickness_mm)
    tr = t + float(spec.hi_lo_mm)
    rf = float(spec.root_face_mm)
    return float(wall_x_mm(spec, t)[0]), t, float(wall_x_mm(spec, max(tr, rf))[1]), tr


def bare_profile(spec, span_mm: float, step_mm: float) -> Profile:
    """The bare specimen's upper surface of a section (GrooveSpec: the station's measured section): the plate tops out to span_mm
    beyond each groove top edge, the groove walls (root faces vertical, bevel faces sloped, square walls vertical), the backing-bar top
    in the gap at z = 0 (the plan's floor also without backing - the groove area counts from there). step_mm: the sampling of the
    sloped and flat parts (numerical resolution)."""
    g, rf = 0.5 * float(spec.root_gap_mm), float(spec.root_face_mm)
    xeL, t, xeR, tr = groove_edges_mm(spec)
    xa, xb = xeL - float(span_mm), xeR + float(span_mm)
    path = [(xa, t), (xeL, t), (-g, rf), (-g, 0.0), (g, 0.0), (g, rf), (xeR, tr), (xb, tr)]
    if xeL == -g:                                   # a square left wall: the plate top straight down to the floor
        path[2] = (-g, min(rf, t))
    X, Z = [path[0][0]], [path[0][1]]
    st = float(step_mm)
    for (x0, z0), (x1, z1) in zip(path[:-1], path[1:]):
        if x1 > x0:
            n = max(1, int(math.ceil((x1 - x0) / st - 1e-9)))
            for k in range(1, n):
                f = k / n
                X.append(x0 + f * (x1 - x0))
                Z.append(z0 + f * (z1 - z0))
            X.append(x1)
            Z.append(z1)
        else:
            X.append(x1)
            Z.append(z1)
    X, Z = np.asarray(X), np.asarray(Z)
    keep = np.ones(X.size, dtype=bool)
    keep[1:] = ~((np.diff(X) == 0.0) & (np.diff(Z) == 0.0))   # repeated points
    return Profile(X[keep], Z[keep], (xeL, xeR))


# --------------------------------------------------------------------------------------
# a bead
# --------------------------------------------------------------------------------------
@dataclass
class BeadSection:
    """One bead in one section: its toes, the arc over the toe chord, its area and the phase that placed it; the surface before (the
    points under the bead) and the convex solid (hull: CCW, the scene)."""
    area_mm2: float
    kappa: float
    toe_l: tuple
    toe_r: tuple
    phase: str
    aim: tuple = (float("nan"), float("nan"))
    under_x: np.ndarray = field(default_factory=lambda: np.zeros(0))      # the surface before under the bead (x order)
    under_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    target_mm2: float = float("nan")
    _hull_cache: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def hull(self) -> np.ndarray:
        """The bead's convex solid (CCW): the hull of the surface points under it and the tangent polygon of its arc (contains it)."""
        if self._hull_cache is None:
            top = _tangent_top(self.toe_l, self.toe_r, self.kappa, self.apex[0])
            self._hull_cache = _hull(np.vstack([top, np.column_stack([self.under_x, self.under_z])]))
        return self._hull_cache

    @property
    def chord_mm(self) -> float:
        return float(self.toe_r[0] - self.toe_l[0])

    def top(self, xq) -> np.ndarray:
        """The bead's free surface at x in [toe_l, toe_r] (the chord line + the arc) [mm]."""
        xq = np.asarray(xq, dtype=float)
        c = self.chord_mm
        if c <= 0.0:
            return np.full(xq.shape, float(self.toe_l[1]))
        u = xq - self.toe_l[0]
        return self.toe_l[1] + (self.toe_r[1] - self.toe_l[1]) * u / c + crown_height_mm(u, c, self.kappa)

    @property
    def apex(self) -> tuple:
        """The highest point of the free surface (x, z) [mm] - the crown."""
        c = self.chord_mm
        if c <= 0.0:
            return float(self.toe_l[0]), float(self.toe_l[1])
        m = (self.toe_r[1] - self.toe_l[1]) / c
        s, R = _arc(c, self.kappa)
        if s <= 0.0:
            x = self.toe_l[0] if self.toe_l[1] >= self.toe_r[1] else self.toe_r[0]
        else:
            v = m * R / math.sqrt(1.0 + m * m)
            x = self.toe_l[0] + 0.5 * c + v if abs(v) <= 0.5 * c else (self.toe_r[0] if m > 0 else self.toe_l[0])
        return float(x), float(self.top(x))

    def region(self, n_top: int = 64) -> np.ndarray:
        """The bead's region as a polygon (CCW; the surface before along the bottom, the arc on top) - figures."""
        xt = np.linspace(self.toe_l[0], self.toe_r[0], max(2, int(n_top)))
        top = np.column_stack([xt[::-1], self.top(xt[::-1])])
        bot = np.column_stack([self.under_x, self.under_z])
        return np.vstack([bot, top])

    def to_dict(self) -> dict:
        ax, az = self.apex
        return {"area_mm2": float(self.area_mm2), "crown_ratio": float(self.kappa), "phase": self.phase,
                "toe_left_xz_mm": [float(v) for v in self.toe_l], "toe_right_xz_mm": [float(v) for v in self.toe_r],
                "crown_xz_mm": [ax, az], "chord_mm": self.chord_mm}


def _tangent_top(toe_l, toe_r, kappa, apex_x) -> np.ndarray:
    """Vertices (left -> right) of the polygon bounded by tangent lines of the bead's free surface (contains it): the toes, the tangent
    intersections; the apex tangent is horizontal (the scene's solids keep a face straight up at their top - weld_clearance)."""
    xl, zl = toe_l
    xr, zr = toe_r
    c = xr - xl
    if c <= 0.0:
        return np.array([[xl, zl], [xr, zr]], dtype=float)
    m = (zr - zl) / c
    s_, R = _arc(c, kappa)
    th = 2.0 * math.atan(2.0 * kappa) if s_ > 0.0 else 0.0
    us = list(0.5 * c + R * np.sin(np.linspace(-th, th, TANGENT_POINTS + 1))) if s_ > 0.0 else [0.0, c]   # uniform along the arc
    us[0], us[-1] = 0.0, c
    ua = float(apex_x) - xl
    if 0.0 < ua < c:
        us.append(ua)
    us = np.unique(np.clip(np.asarray(us, dtype=float), 0.0, c))
    zt = zl + m * us + crown_height_mm(us, c, kappa)
    sl = m + _crown_slope(us, c, kappa)
    if 0.0 < ua < c:
        sl[np.argmin(np.abs(us - ua))] = 0.0
    pts = [(xl, zl)]
    for k in range(us.size - 1):
        xa, za, sa = xl + us[k], zt[k], sl[k]
        xb, zb, sb = xl + us[k + 1], zt[k + 1], sl[k + 1]
        if abs(sa - sb) < 1e-15:
            pts.append((0.5 * (xa + xb), 0.5 * (za + zb)))
            continue
        x = (zb - za + sa * xa - sb * xb) / (sa - sb)
        x = min(max(x, xa), xb)
        pts.append((x, za + sa * (x - xa)))
    pts.append((xr, zr))
    return np.asarray(pts, dtype=float)


def _hull(P: np.ndarray) -> np.ndarray:
    """Convex hull (Andrew's monotone chain), CCW, no repeated points."""
    P = np.unique(np.round(np.asarray(P, dtype=float), 12), axis=0)
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(tuple(p))
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(tuple(p))
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


# --------------------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------------------
def _locate(prof: Profile, x: float, z: float) -> int:
    """Index j of the segment (point j -> j + 1) that holds the point (x, z) of the surface (the nearest one in z at x)."""
    X, Z = prof.x, prof.z
    j0 = max(int(np.searchsorted(X, x, side="left")) - 1, 0)
    j1 = min(int(np.searchsorted(X, x, side="right")), X.size - 1)
    best, bj = math.inf, max(min(j0, X.size - 2), 0)
    for j in range(j0, max(j1, j0 + 1)):
        if j + 1 >= X.size:
            break
        xa, xb, za, zb = X[j], X[j + 1], Z[j], Z[j + 1]
        if not (xa <= x <= xb):
            continue
        if xb > xa:
            d = abs(za + (zb - za) * (x - xa) / (xb - xa) - z)
        else:
            lo, hi = min(za, zb), max(za, zb)
            d = 0.0 if lo <= z <= hi else min(abs(z - lo), abs(z - hi))
        if d < best:
            best, bj = d, j
    return bj


@dataclass
class _Cut:
    """A bead's footprint on the path: the toes on segments (iL, iL + 1) and (iR - 1, iR) (path indices; the points strictly between
    are under the bead), the toes (x, z) and the old surface heights below them (lower than the toe at a cliff)."""
    iL: int
    iR: int
    toe_l: tuple
    toe_r: tuple
    old_l: float
    old_r: float


class _Path:
    """The surface as a path from left to right with the aim inserted (index pa; a weaving pass: its weave band's ends pa_l, pa_r -
    the deposit lands over the band); walls = heights as the pool sees them (a band of a multi-bead layer makes everything beyond its
    edges a wall: +inf)."""

    def __init__(self, prof: Profile, aim_xz, band=None, spread: float = 0.0):
        p = prof
        if band is not None:
            p = prof.insert([v for v in band if v is not None and math.isfinite(v)])
        ax, az = float(aim_xz[0]), float(aim_xz[1])
        sp_ = float(spread)
        if sp_ > 0.0:                                                     # the weave band's ends on the surface (inside the band)
            lo_b = band[0] if band is not None and band[0] is not None else -math.inf
            hi_b = band[1] if band is not None and band[1] is not None else math.inf
            eL, eR = (float(v) for v in p.edges)                          # the weave lands on the groove, not beyond its top edges
            xa_ = max(ax - sp_, lo_b, float(p.x[1]), min(eL, ax) if math.isfinite(eL) else -math.inf)
            xb_ = min(ax + sp_, hi_b, float(p.x[-2]), max(eR, ax) if math.isfinite(eR) else math.inf)
            p = p.insert([xa_, xb_])
            self.prof = p
            self.x, self.z = p.x.copy(), p.z.copy()
            self.pa_l = int(np.searchsorted(p.x, xa_, side="right")) - 1     # the inner of a vertical face at the band's end
            self.pa_r = int(np.searchsorted(p.x, xb_, side="left"))
            self.pa = self.pa_l
        else:
            self.prof = p
            j = _locate(p, ax, az)
            xa_, xb_, za_, zb_ = p.x[j], p.x[j + 1], p.z[j], p.z[j + 1]
            az = (za_ + (zb_ - za_) * (ax - xa_) / (xb_ - xa_) if xb_ > xa_ else          # the aim on the surface (its segment)
                  min(max(az, min(za_, zb_)), max(za_, zb_)))
            self.x = np.concatenate([p.x[:j + 1], [ax], p.x[j + 1:]])
            self.z = np.concatenate([p.z[:j + 1], [az], p.z[j + 1:]])
            self.pa = self.pa_l = self.pa_r = j + 1
        self.w = self.z.copy()
        if band is not None:
            bl, br = band
            if bl is not None and math.isfinite(bl):
                self.w[self.x < bl] = np.inf
            if br is not None and math.isfinite(br):
                self.w[self.x > br] = np.inf
        # the two sides as sequences from the aim outward, with suffix maxima (the flood level)
        self.idx = {-1: np.arange(self.pa_l, -1, -1), 1: np.arange(self.pa_r, self.x.size)}
        self.ws = {d: self.w[i] for d, i in self.idx.items()}
        self.smax = {d: np.maximum.accumulate(v[::-1])[::-1] for d, v in self.ws.items()}
        # the deposit flows downhill from the aim: on each side the run from the aim while the surface does not rise (an aim one
        # sampling step off a crown's top - the polyline's vertex - starts at that vertex: it is at the crown, both sides go down)
        self.run_end = {}
        self.run_start = {}
        for d, v in self.ws.items():
            zz = v
            s0 = 1 if (zz.size > 2 and zz[1] > zz[0] and zz[2] < zz[1]) else 0
            with np.errstate(invalid="ignore"):
                rise = np.flatnonzero(np.diff(zz[s0:]) > 0.0)
            self.run_start[d] = s0
            self.run_end[d] = s0 + (int(rise[0]) + 1 if rise.size else zz.size - s0)   # exclusive

    def _pocket_start(self, d: int, z: float, strict: bool = False) -> Optional[int]:
        """Index (in the side's sequence) where the pool at level z starts on side d: the aim if it is below z, else the first point
        below z on the downhill run from the aim; None when the run stays at or above z (no pocket this side)."""
        v = self.ws[d]
        if (v[0] <= z) if strict else (v[0] < z):
            return 0
        s0, e = self.run_start[d], self.run_end[d]
        run = v[s0:e]                                             # non-increasing
        k = int(np.searchsorted(-run, -z, side="left" if strict else "right"))
        return s0 + k if k < run.size else None

    def floods(self, z: float) -> bool:
        """Does the pool at level z spill out on a side (no wall after its pocket)? O(log n) per side."""
        for d in (-1, 1):
            k0 = self._pocket_start(d, z)
            if k0 is not None and self.smax[d][k0] < z:
                return True
        return False

    def _cross(self, i_wall: int, i_in: int, z: float) -> tuple:
        """Crossing of level z on the segment from the pocket point i_in to the wall point i_wall → (x, old height there)."""
        za, zb = float(self.w[i_in]), float(self.w[i_wall])
        xa, xb = float(self.x[i_in]), float(self.x[i_wall])
        if not math.isfinite(zb):                       # a band edge: the bead's side is the edge (the last point inside), a cliff
            return xa, float(self.z[i_in])
        f = min(max((z - za) / (zb - za), 0.0), 1.0) if zb != za else 1.0
        return xa + f * (xb - xa), za + f * (zb - za)

    def side_end(self, z: float, d: int, strict: bool = False):
        """One side of the pool at level z (d = -1 left, +1 right): (pocket start index, wall index, last pocket index), None when that
        side has no pocket (the downhill run from the aim stays at or above z - the deposit does not flow uphill), 'flood' when it has
        no wall."""
        k0 = self._pocket_start(d, z, strict)
        if k0 is None:
            return None
        idx, ws = self.idx[d], self.ws[d]
        rest = (ws[k0:] > z) if strict else (ws[k0:] >= z)
        if not rest.any():
            return "flood"
        kb = k0 + int(np.argmax(rest))
        return int(idx[k0]), int(idx[kb]), int(idx[kb - 1])

    def pool(self, z: float, strict: bool = False) -> Optional[_Cut]:
        """The pool at level z from the aim (module docstring): skip the hump under the aim (points at or above z) on the sides the
        surface goes down from the aim, the pocket below z,
        up to the first wall point at or above z (strict: above z - points at exactly z are covered) on each side. → the footprint, a
        zero footprint (no pocket on either side), or None (no wall on a side: the pool floods out)."""
        pa = self.pa_l
        L, R = self.side_end(z, -1, strict), self.side_end(z, 1, strict)
        if L == "flood" or R == "flood":
            return None
        if L is None and R is None:
            return _Cut(pa, pa, (self.x[pa], z), (self.x[pa], z), z, z)
        if L is not None and R is not None:
            iL, iR = L[1], R[1]
            xl, fl = self._cross(L[1], L[2], z)
            xr, fr = self._cross(R[1], R[2], z)
        elif R is not None:                             # only the right side has a pocket: its left end = where the hump comes down
            k0 = R[0]
            iL, iR = k0 - 1, R[1]
            xl, fl = self._cross(k0 - 1, k0, z)
            xr, fr = self._cross(R[1], R[2], z)
        else:
            k0 = L[0]
            iL, iR = L[1], k0 + 1
            xl, fl = self._cross(L[1], L[2], z)
            xr, fr = self._cross(k0 + 1, k0, z)
        return _Cut(iL, iR, (xl, z), (xr, z), min(fl, z), min(fr, z))

    def out_at(self, i_from: int, d: int, x: float) -> Optional[tuple]:
        """Walking outward (d = -1 left, +1 right) from path index i_from, the segment holding x → (index of its outer point, the
        surface height at x), or None when the path ends first."""
        X, Z = self.x, self.z
        if d < 0:
            j = int(np.searchsorted(X[:i_from + 1], x, side="right")) - 1     # the outer point: the last with X <= x
            if j < 0:
                return None
            if j == i_from:
                return (j, float(Z[j])) if X[j] == x else None
            xa, xb, za, zb = X[j], X[j + 1], Z[j], Z[j + 1]
        else:
            j = i_from + int(np.searchsorted(X[i_from:], x, side="left"))      # the outer point: the first with X >= x
            if j >= X.size:
                return None
            if j == i_from:
                return (j, float(Z[j])) if X[j] == x else None
            xa, xb, za, zb = X[j], X[j - 1], Z[j], Z[j - 1]
        h = float(max(za, zb)) if xb == xa else float(za + (zb - za) * (x - xa) / (xb - xa))
        return j, h

    def bead(self, cut: _Cut, kappa: float) -> tuple:
        """The bead on this footprint: (area, x, old z, new z) over the toes and the points between them."""
        x = np.concatenate([[cut.toe_l[0]], self.x[cut.iL + 1:cut.iR], [cut.toe_r[0]]])
        z_old = np.concatenate([[cut.old_l], self.z[cut.iL + 1:cut.iR], [cut.old_r]])
        xl, zl = cut.toe_l
        xr, zr = cut.toe_r
        c = xr - xl
        if c > 0.0:
            T = zl + (zr - zl) * (x - xl) / c + crown_height_mm(x - xl, c, kappa)
        else:
            T = np.full(x.shape, zl)
        T[0], T[-1] = zl, zr
        z_new = np.maximum(z_old, T)
        z_new[0], z_new[-1] = max(cut.old_l, zl), max(cut.old_r, zr)
        dx = np.diff(x)
        area = float(np.sum(0.5 * (z_new[1:] + z_new[:-1] - z_old[1:] - z_old[:-1]) * dx))
        return area, x, z_old, z_new


def _solve(fun, lo: float, hi: float, target: float, f_lo: Optional[float] = None, f_hi: Optional[float] = None,
           tol: float = 1e-9) -> tuple:
    """Monotone non-decreasing fun on [lo, hi], fun(lo) <= target <= fun(hi): a bracket [lo, hi] (width <= 2 tol) of fun = target -
    Brent's method on fun - target (for a left-continuous fun with jumps it closes on the jump)."""
    flo = fun(lo) if f_lo is None else f_lo
    fhi = fun(hi) if f_hi is None else f_hi
    if flo >= target:
        return lo, lo
    if fhi <= target or not math.isfinite(fhi):
        if not math.isfinite(fhi):                                     # an infinite end (a flood): bisect it away first
            for _ in range(200):
                m = 0.5 * (lo + hi)
                fm = fun(m)
                if not math.isfinite(fm):
                    hi, fhi = m, fm
                elif fm < target:
                    lo, flo = m, fm
                else:
                    hi, fhi = m, fm
                    break
                if hi - lo <= tol:
                    return lo, hi
        if fhi <= target:
            return hi, hi
    from scipy.optimize import brentq                                  # noqa: PLC0415
    r = brentq(lambda x: fun(x) - target, lo, hi, xtol=tol, rtol=4.0 * np.finfo(float).eps, maxiter=200)
    return max(lo, r - tol), min(hi, r + tol)


def place_bead(prof: Profile, aim_xz, area_mm2: float, kappa: float, *, band=None, spread_mm: float = 0.0) -> tuple:
    """Place one bead of area_mm2 [mm^2] with the crown ratio kappa (bead_model) on the surface prof, aimed at aim_xz (a point of the
    surface) - module docstring (pool / spill / climb / over; band = (x_left, x_right) or None: the bead stays in that band, its edges
    act as walls; spread_mm = a weaving pass's lateral weave amplitude: the deposit lands over aim x +- spread_mm). → (BeadSection,
    the surface after)."""
    a = float(area_mm2)
    kappa = float(kappa)
    if not a > 0.0:                                       # nothing deposited: a bead of no width at the aim, the surface unchanged
        ax, az = float(aim_xz[0]), float(aim_xz[1])
        return BeadSection(0.0, kappa, (ax, az), (ax, az), "pool", aim=(ax, az), under_x=np.array([ax, ax]),
                           under_z=np.array([az, az]), target_mm2=a), prof.copy()
    P = _Path(prof, aim_xz, band, spread_mm)
    banded = band is not None

    def A(z, strict=False):
        c = P.pool(z, strict)
        return math.inf if c is None else P.bead(c, kappa)[0]
    fin = P.w[np.isfinite(P.w)]
    z_lo = float(np.min(fin))
    cut, phase = None, "pool"
    # the highest level at which the pool from the aim stays between walls (above it, it spills out over a wall top: the plate tops)
    z_hi = float(np.max(fin)) + 1.0
    if P.floods(z_hi):
        lo_f, hi_f = z_lo, z_hi
        for _ in range(80):
            if hi_f - lo_f <= 1e-10:
                break
            m = 0.5 * (lo_f + hi_f)
            if P.floods(m):
                hi_f = m
            else:
                lo_f = m
        z_hi = lo_f
    elif banded:
        step = max(a / max(float(np.ptp(P.x[np.isfinite(P.w)])), 1e-6), 1e-6)
        while A(z_hi) < a:
            z_hi += step
            step *= 2.0
    if z_hi > z_lo and A(z_hi) >= a:
        lo, hi = _solve(A, z_lo, z_hi, a, f_lo=0.0)
        cand = fin[(fin >= lo) & (fin <= hi)]              # a jump at a point's height in [lo, hi]: a flat or a ridge at that level
        zj = float(np.min(cand)) if cand.size else None
        c_ge = P.pool(zj) if zj is not None else None
        if zj is not None and c_ge is not None and P.bead(c_ge, kappa)[0] < a:
            cut, phase = _spill(P, zj, c_ge, P.pool(zj, strict=True), a, kappa), "spill"
        else:
            cut = P.pool(hi)
    elif not banded:
        cut, phase = _over(P, z_lo, z_hi, a, kappa)
    else:                                                 # a band open to a flat (the plate top): spill over it at its level
        up = fin[fin >= z_hi]
        if up.size:
            zj = float(np.min(up))
            c_ge = P.pool(zj)
            if c_ge is not None:
                cut, phase = _spill(P, zj, c_ge, P.pool(zj, strict=True), a, kappa), "spill"
    if cut is None:
        raise ValueError(f"bead placement: no bead of {a:.2f} mm2 fits at the aim ({float(aim_xz[0]):.2f}, {float(aim_xz[1]):.2f}) mm")
    return _finish(P, "band" if banded else phase, cut, kappa, aim_xz, a)


def _spill(P: _Path, zj: float, c_ge: _Cut, c_gt: Optional[_Cut], a: float, kappa: float) -> _Cut:
    """The area ends while the pool spills at level zj (a flat at zj or a ridge top): the open side(s) extend at zj by equal lengths
    (each up to where the merged pool ends; the profile end when it floods out) - the toe on the flat, or a spill front."""
    ends = (float(P.x[0]), float(P.x[-1])) if c_gt is None else (c_gt.toe_l[0], c_gt.toe_r[0])
    dL = max(c_ge.toe_l[0] - ends[0], 0.0)
    dR = max(ends[1] - c_ge.toe_r[0], 0.0)

    def build(e):
        iL, xl, fl = c_ge.iL, c_ge.toe_l[0], c_ge.old_l
        iR, xr, fr = c_ge.iR, c_ge.toe_r[0], c_ge.old_r
        if dL > 0.0 and e > 0.0:
            xl = c_ge.toe_l[0] - min(e, dL)
            o = P.out_at(c_ge.iL + 1, -1, xl)
            if o is not None:
                iL, fl = o[0], min(o[1], zj)
        if dR > 0.0 and e > 0.0:
            xr = c_ge.toe_r[0] + min(e, dR)
            o = P.out_at(c_ge.iR - 1, 1, xr)
            if o is not None:
                iR, fr = o[0], min(o[1], zj)
        return _Cut(iL, iR, (xl, zj), (xr, zj), fl, fr)
    lo, hi = _solve(lambda e: P.bead(build(e), kappa)[0], 0.0, max(dL, dR), a)
    return build(hi)


def _over(P: _Path, z_lo: float, z_hi: float, a: float, kappa: float) -> tuple:
    """The pool reached the lower groove top edge (or nothing below it is reachable from the aim): climb (the toe on the lower edge
    stays, the other toe climbs its wall to its own edge), then over (both toes move out over the plate tops by equal lengths, the
    chord from toe to toe) - the smallest footprint that holds the area (the area need not grow monotonically while the toes come down
    the flanks of earlier beads over the edges: the first footprint that holds it)."""
    c0 = P.pool(z_hi) if z_hi > z_lo else None
    base = None
    if c0 is not None and c0.iL != c0.iR:
        base = c0
        # each side's wall top: the first point outward from the pool's end that the next point does not rise above
        tops = []
        for i0, d in ((c0.iL, -1), (c0.iR, 1)):
            i = i0
            while 0 <= i + d < P.x.size and P.z[i + d] > P.z[i]:
                i += d
            tops.append((i, float(P.z[i])))
        (iTL, hL), (iTR, hR) = tops
        low_left = hL <= hR
        t_lo, t_hi = min(hL, hR, z_hi), max(hL, hR)
        if t_hi > t_lo:
            d_hi = 1 if low_left else -1                                   # the side that climbs

            def climb(q):
                # the climbing side's toe: up its wall from the pool's end at z_hi (the first point at or above q outward)
                i_in = c0.iR - 1 if low_left else c0.iL + 1
                i = c0.iR if low_left else c0.iL
                while 0 <= i < P.x.size and P.w[i] < q:
                    i_in, i = i, i + d_hi
                if not 0 <= i < P.x.size:
                    return None
                x_c, f_c = P._cross(i, i_in, q)
                return (_Cut(c0.iL, i, c0.toe_l, (x_c, q), c0.old_l, min(f_c, q)) if low_left else
                        _Cut(i, c0.iR, (x_c, q), c0.toe_r, min(f_c, q), c0.old_r))
            # the climbing side's wall: up to its top (the pool would flood there: side_end of that side alone)
            end = climb(t_hi - 1e-12 * max(1.0, abs(t_hi)))
            if end is not None:
                if P.bead(end, kappa)[0] >= a:
                    lo, hi = _solve(lambda q: P.bead(climb(q), kappa)[0], t_lo, t_hi - 1e-12 * max(1.0, abs(t_hi)), a)
                    return climb(hi), "climb"
                base = end
    if base is None:                                                      # start at the groove top edges
        eL, eR = P.prof.edges
        oL, oR = P.out_at(P.pa_l, -1, eL), P.out_at(P.pa_r, 1, eR)
        if oL is None or oR is None:
            return None, "over"
        base = _Cut(oL[0], oR[0], (eL, oL[1]), (eR, oR[1]), oL[1], oR[1])

    def over(d):
        if d <= 0.0:
            return base
        xl, xr = base.toe_l[0] - d, base.toe_r[0] + d
        oL = P.out_at(base.iL + 1, -1, xl)
        oR = P.out_at(base.iR - 1, 1, xr)
        if oL is None or oR is None:
            return None
        return _Cut(oL[0], oR[0], (xl, oL[1]), (xr, oR[1]), oL[1], oR[1])

    def A(d):
        return P.bead(over(d), kappa)[0]
    if A(0.0) >= a:
        return base, "over"
    dx = np.diff(P.prof.x)
    step = float(np.median(dx[dx > 0.0])) if np.any(dx > 0.0) else 0.1   # the surface's own sampling
    d_end = min(base.toe_l[0] - float(P.x[0]), float(P.x[-1]) - base.toe_r[0])
    br = None
    d_from = 0.0
    while br is None and d_from < d_end:                                  # coarse steps (8 surface steps), eight at a time
        ds = d_from + 8.0 * step * np.arange(1, 9)
        ds = ds[ds <= d_end]
        if ds.size == 0:
            break
        Ad = _over_areas(P, base, ds, kappa)
        hit = np.flatnonzero(Ad >= a)
        if hit.size:
            k = int(hit[0])
            br = (float(ds[k - 1]) if k > 0 else d_from, float(ds[k]))
        d_from = float(ds[-1])
    if br is not None:                                                    # the surface's step inside the first coarse hit
        ds = br[0] + step * np.arange(1, 9)
        ds = ds[ds < br[1]]
        if ds.size:
            Ad = _over_areas(P, base, ds, kappa)
            hit = np.flatnonzero(Ad >= a)
            if hit.size:
                k = int(hit[0])
                br = (float(ds[k - 1]) if k > 0 else br[0], float(ds[k]))
            else:
                br = (float(ds[-1]), br[1])
    if br is None:
        return None, "over"
    d0, d1 = br
    lo, hi = _solve(A, d0, d1, a, tol=1e-7)
    return over(hi), "over"


def _over_areas(P: _Path, base: _Cut, ds: np.ndarray, kappa: float) -> np.ndarray:
    """The 'over' bead's area for each outward extension d in ds at once (toes on the surface at base toes -/+ d; the same bead as
    _over's over(d) and _Path.bead)."""
    ds = np.asarray(ds, dtype=float)
    xl = base.toe_l[0] - ds
    xr = base.toe_r[0] + ds
    XL, ZL = P.x[:base.iL + 1], P.z[:base.iL + 1]
    XR, ZR = P.x[base.iR:], P.z[base.iR:]
    zl = np.interp(xl, XL, ZL)
    zr = np.interp(xr, XR, ZR)
    lo = int(np.searchsorted(P.x, float(xl.min()), side="left"))
    hi = int(np.searchsorted(P.x, float(xr.max()), side="right"))
    X, Z = P.x[lo:hi], P.z[lo:hi]
    c = (xr - xl)[:, None]
    u = X[None, :] - xl[:, None]
    s_ = kappa * c
    Rr = (0.25 * c * c + s_ * s_) / (2.0 * np.maximum(s_, 1e-300))
    v = np.clip(u, 0.0, c) - 0.5 * c
    crown = np.where(s_ > 0.0, np.sqrt(np.maximum(Rr * Rr - v * v, 0.0)) - (Rr - s_), 0.0)
    T = zl[:, None] + (zr - zl)[:, None] * u / c + crown
    inside = (X[None, :] > xl[:, None]) & (X[None, :] < xr[:, None])
    D = np.where(inside, np.maximum(T - Z[None, :], 0.0), 0.0)
    both = inside[:, 1:] & inside[:, :-1]
    area = np.sum(np.where(both, 0.5 * (D[:, 1:] + D[:, :-1]) * np.diff(X)[None, :], 0.0), axis=1)
    j0 = np.argmax(inside, axis=1)
    j1 = inside.shape[1] - 1 - np.argmax(inside[:, ::-1], axis=1)
    has = inside.any(axis=1)
    r = np.arange(ds.size)
    area += np.where(has, 0.5 * D[r, j0] * (X[j0] - xl) + 0.5 * D[r, j1] * (xr - X[j1]), 0.0)
    return area


def _finish(P: _Path, phase: str, cut: _Cut, kappa: float, aim_xz, a_target: float) -> tuple:
    """The bead record and the surface after it (the old points outside the toes; a cliff where the old surface is below a toe)."""
    area, x, zo, zn = P.bead(cut, kappa)
    X = [P.x[:cut.iL + 1]]
    Z = [P.z[:cut.iL + 1]]
    if zo[0] < zn[0]:
        X.append([x[0]])
        Z.append([zo[0]])
    X.append(x)
    Z.append(zn)
    if zo[-1] < zn[-1]:
        X.append([x[-1]])
        Z.append([zo[-1]])
    X.append(P.x[cut.iR:])
    Z.append(P.z[cut.iR:])
    X, Z = np.concatenate(X), np.concatenate(Z)
    keep = np.ones(X.size, dtype=bool)
    keep[1:] = ~((np.diff(X) == 0.0) & (np.diff(Z) == 0.0))
    X, Z = X[keep], Z[keep]
    # the aim point inserted into the path is not a profile point: drop it where it is collinear (outside the bead)
    new = Profile(X, Z, tuple(P.prof.edges))
    bead = BeadSection(area, float(kappa), (float(cut.toe_l[0]), float(cut.toe_l[1])), (float(cut.toe_r[0]), float(cut.toe_r[1])),
                       phase, aim=(float(aim_xz[0]), float(aim_xz[1])), under_x=np.asarray(x, dtype=float),
                       under_z=np.asarray(zo, dtype=float), target_mm2=float(a_target))
    return bead, new


# --------------------------------------------------------------------------------------
# layers of several beads (bands), measurement
# --------------------------------------------------------------------------------------
def _layer_span(prof: Profile, z: float) -> Optional[tuple]:
    """The groove's pool at level z between its walls: (x_left, x_right) where the surface comes down below z scanning in from each
    groove top edge - at a top edge lower than z the edge itself (the pool rises between the higher edge's wall and the lower edge,
    as a bead climbs there: bead_geometry 'climb') -, (nan, nan) when nothing is below z, None when z is above both top edges."""
    X, Z = prof.x, prof.z
    eL, eR = prof.edges
    hL, hR = prof.edge_heights()
    if z > max(hL, hR):
        return None
    iL = int(np.searchsorted(X, eL, side="left"))
    iR = int(np.searchsorted(X, eR, side="right")) - 1
    inner = np.arange(iL, iR + 1)
    below = Z[inner] < z
    if not below.any():
        return float("nan"), float("nan")
    a, b = int(inner[np.argmax(below)]), int(inner[::-1][np.argmax(below[::-1])])
    if a == iL:
        xl = float(X[a])
    else:
        za, zb = Z[a - 1], Z[a]
        xl = float(X[a]) if zb == za else float(X[a - 1] + (z - za) / (zb - za) * (X[a] - X[a - 1]))
    if b == iR:
        xr = float(X[b])
    else:
        za, zb = Z[b], Z[b + 1]
        xr = float(X[b]) if zb == za else float(X[b] + (z - za) / (zb - za) * (X[b + 1] - X[b]))
    return xl, xr


def _pool_area(prof: Profile, z: float) -> float:
    s = _layer_span(prof, z)
    if s is None or not math.isfinite(s[0]):
        return 0.0
    return prof.area_below(s[0], s[1], z)


def pool_level(prof: Profile, area_mm2: float) -> Optional[tuple]:
    """The level z whose pool between the groove walls (below z, edge to edge, no crown; up to the higher top edge) holds area_mm2
    → (z, x_left, x_right), or None when the groove up to its higher top edge cannot hold it (the layer overflows)."""
    top = max(prof.edge_heights())
    if _pool_area(prof, top) < float(area_mm2):
        return None
    lo, hi = _solve(lambda z: _pool_area(prof, z), float(np.min(prof.z)), top, float(area_mm2), f_lo=0.0)
    s = _layer_span(prof, hi)
    return float(hi), s[0], s[1]


def pool_cuts(prof: Profile, z: float, x_left: float, x_right: float, n: int) -> np.ndarray:
    """The n - 1 vertical lines that cut the pool below z over [x_left, x_right] into n equal areas (a multi-bead layer's bands)."""
    if n <= 1:
        return np.zeros(0)
    xs = np.unique(np.concatenate([np.linspace(x_left, x_right, 801), prof.x[(prof.x > x_left) & (prof.x < x_right)]]))
    d = np.maximum(z - prof.at(xs), 0.0)
    C = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(xs))])
    return np.interp(np.arange(1, n) / n * C[-1], C, xs)


def crown_ratio_from_profiles(x, z_before, z_after) -> dict:
    """A bead's crown ratio from a section profile before and after the pass (the post-weld scans that will calibrate the bead shape):
    toes = where the surface rose (first / last x with z_after > z_before), chord between them, crown ratio = the free surface's height
    above that chord / the chord (the arc model's kappa_p). Information for later calibration - no fit here."""
    x = np.asarray(x, dtype=float)
    d = np.asarray(z_after, dtype=float) - np.asarray(z_before, dtype=float)
    res = 64.0 * np.finfo(float).eps * max(1.0, float(np.nanmax(np.abs(z_after))))   # the arithmetic's resolution
    up = np.flatnonzero(d > res)
    if up.size < 2:
        return {"crown_ratio": float("nan"), "chord_mm": 0.0, "height_mm": 0.0}
    i0, i1 = int(up[0]), int(up[-1])
    xl, xr = float(x[i0]), float(x[i1])                    # the outermost raised points (the toes, to the sampling)
    zl, zr = float(np.asarray(z_after, dtype=float)[i0]), float(np.asarray(z_after, dtype=float)[i1])
    c = float(xr - xl)
    sel = (x >= xl) & (x <= xr)
    h = float(np.max(np.asarray(z_after, dtype=float)[sel] - (zl + (zr - zl) * (x[sel] - xl) / c)))
    return {"crown_ratio": h / c if c > 0 else float("nan"), "chord_mm": c, "height_mm": h, "toe_left_xz_mm": [float(xl), zl],
            "toe_right_xz_mm": [float(xr), zr], "area_mm2": float(np.trapz(np.maximum(d, 0.0), x))}
