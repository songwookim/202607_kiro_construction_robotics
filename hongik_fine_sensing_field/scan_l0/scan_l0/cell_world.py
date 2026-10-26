"""One cell world for the 1G core (interface design r1 §2.15, r3 §6): the single description of the specimen and its support from
which BOTH the straight-leg envelope (path_gen.check_envelope via the reference job) and the MoveIt collision objects are derived.

Before any measurement it is built from the two rough points and the design; after every fused estimate it is rebuilt from the
measured specimen. Everything is in the robot base frame [m], up = -gravity.

  plate top     the plane the tool keeps its distance from. Prior: through the line of the two rough points, normal = up made
                perpendicular to that line (the plate contains the seam; roll unknown until measured), shifted by t_hit along the
                normal (the rough point may be at the root: t_hit = plate thickness with a root gap, else the groove depth) - the
                highest plate top the inputs allow. Measured: the fused model's highest plate-top point + k sigma_top along n^.
  specimen      a box whose top face lies on that plane, along the seam: prior = the rough points extended outward on both sides
                by the design length the points do not span (whichever end they sit at, it is covered); measured = the
                estimated ends -/+ k sigma_e. Across the seam: left plate + root gap + right plate (design widths of each plate),
                placed about the seam - each plate on its side once the estimate has resolved which side the bevel (design) plate
                is on; before that, or for a groove whose sides cannot be told apart, the wider plate on both sides. Down to the
                lowest plate bottom the inputs allow (prior: rough point - t - backing; measured: top - 2 k sigma_top - t - backing).
  support       the horizontal plane below the specimen (1G does not measure it): the lowest corner of the specimen's top face
                minus thickness and backing - the highest support that does not cut into the specimen. The floor object covers
                the specimen footprint plus the tool's reach (coarse standoff + safe clearance) on every side, as a slab of
                thickness C below that plane; toward the robot it stops C short of the robot's fixed body (the URDF collision
                geometry of the links no joint moves, given by the platform) - the robot's own body is never inside the world.
  under         the space below the specimen: its footprint (grown by C) from the support plane down to the lowest point the arm can
                reach (the platform's reach bound from the robot model) - the support / fixture under the specimen, never a place
                for the arm (user 2026-10-14). The robot's own fixed links may lie in it (a test bed of the robot model under the
                specimen): the platform allows exactly those contacts (they cannot move), so it is never cut like the floor.
  clearance     MoveIt objects are the solids grown by C (policy clearance_m) on every side (joint-space moves keep C); straight
                legs keep L = C + stopping distance from the planes (the envelope).

No ladder, no sizes beyond the design, the tool's working range and the measured geometry. The current arm state is never used to
shape the world: when the arm is inside it the caller refuses before motion and names the link and the object (explain()).
MoveIt object ids are under NAMESPACE ('hongik/'): we add and remove only our own objects. to_dict() is the scene-update data
(version, planes, objects) that can be handed to another scene owner (KIRO) as it is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from scan_core.frames import make_transform, unit

__all__ = ["NAMESPACE", "SPECIMEN_ID", "FLOOR_ID", "UNDER_ID", "SCHEMA", "Solid", "CellWorld", "contact_pairs"]

NAMESPACE = "hongik/"
SPECIMEN_ID = NAMESPACE + "specimen"
FLOOR_ID = NAMESPACE + "floor"
UNDER_ID = NAMESPACE + "under"
SCHEMA = "hongik.cell_world/1"


def _l(v) -> list:
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]


@dataclass
class Solid:
    """A box: centre, axes (columns of R, right-handed), edge lengths along those axes."""
    id: str
    center: np.ndarray
    R: np.ndarray
    size: np.ndarray
    source: str

    def T(self) -> np.ndarray:
        return make_transform(self.R, self.center)

    def corners(self) -> np.ndarray:
        h = 0.5 * np.asarray(self.size, dtype=float)
        s = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
        return self.center[None, :] + (s * h[None, :]) @ self.R.T

    def contains(self, P) -> np.ndarray:
        """Points (N, 3) inside the box (boundary included)."""
        loc = (np.asarray(P, dtype=float).reshape(-1, 3) - self.center) @ self.R
        return np.all(np.abs(loc) <= 0.5 * np.asarray(self.size)[None, :] + 1e-12, axis=1)


def _frame(u, n) -> np.ndarray:
    """Right-handed axes [u, w = n x u, n] with u made perpendicular to n."""
    n = unit(np.asarray(n, dtype=float))
    u = unit(np.asarray(u, dtype=float) - n * float(np.asarray(u, dtype=float) @ n))
    return np.column_stack([u, np.cross(n, u), n])


def contact_pairs(text: str) -> list:
    """'... (contacts ['link6-hongik/floor', ...])' (the state_collision text of the MoveIt adapter / virtual arm) → [(link, object)]
    (a pair is split at the object name when it is ours; otherwise at the first '-')."""
    import re
    m = re.search(r"contacts \[(.*)\]", str(text or ""))
    if not m:
        return []
    out = []
    for item in re.findall(r"'([^']*)'", m.group(1)):
        if item.startswith(NAMESPACE):
            j = item.find("-", len(NAMESPACE))
            obj, link = (item[:j], item[j + 1:]) if j > 0 else (item, "")
            out.append((link, obj))
            continue
        i = item.find("-" + NAMESPACE)
        out.append((item[:i], item[i + 1:]) if i > 0 else tuple(item.split("-", 1)) if "-" in item else (item, ""))
    return out


class CellWorld:
    """The 1G core's world (module docstring). Build with from_rough_points(), update with update_measured(); read envelope(),
    moveit_objects(), to_dict(); explain() names a contact."""

    def __init__(self, *, up, thickness_m: float, backing_m: float, design_length_m: float, clearance_m: float,
                 reach_m: float, base_frame: str = "link0", base_origin=(0.0, 0.0, 0.0), width_m: Optional[float] = None,
                 plate_widths_m: Optional[tuple] = None, gap_m: float = 0.0):
        """plate_widths_m = (left, right) design width of each plate, gap_m the root gap; width_m (older form) = the total of a
        symmetric specimen (two plates of width_m/2, no gap)."""
        self.up = unit(np.asarray(up, dtype=float).reshape(3))
        self.t = float(thickness_m)
        self.b = float(backing_m)
        if plate_widths_m is None:
            if width_m is None:
                raise ValueError("CellWorld: give plate_widths_m (left, right) or width_m")
            plate_widths_m = (0.5 * float(width_m), 0.5 * float(width_m))
        self.plate_widths = (float(plate_widths_m[0]), float(plate_widths_m[1]))
        self.gap = float(gap_m)
        self.left_sign_w: Optional[int] = None      # the side (sign of w = n x u, S->E) of the design's left plate; None = unknown
        self.L_design = float(design_length_m)
        self.C = float(clearance_m)
        self.reach = float(reach_m)
        self.base_frame = str(base_frame)
        self.base_origin = np.asarray(base_origin, dtype=float).reshape(3)
        self.version = 0
        self.source = ""
        self.fixed_bodies: list = []               # platform: the robot's fixed body boxes (fk_urdf.fixed_link_boxes rows)
        self.fixed_bodies_known = False
        self.reach_below_m: Optional[float] = None  # platform: how far below the base origin the arm can reach at most (None = unknown)
        self.plate_point = np.zeros(3)
        self.plate_normal = self.up.copy()
        self.plate_sigma_m: Optional[float] = None  # k sigma added to the plate top (measured), None for the prior
        self.line = (np.zeros(3), np.zeros(3))      # specimen top-face centre line (on the plate plane), S side first
        self.depth_below_top_m = 0.0               # specimen box thickness below its top face
        self.ends: dict = {}

    # ---- building ----------------------------------------------------------------------------------------------------
    @classmethod
    def from_rough_points(cls, A, B, *, hit_face_m: float, **kw) -> "CellWorld":
        """Prior world from the two rough points (in either order) and the design (module docstring)."""
        w = cls(**kw)
        w.set_prior(A, B, hit_face_m=hit_face_m)
        return w

    @classmethod
    def from_record(cls, rec: dict, *, thickness_m: float, backing_m: float, design_length_m: float, clearance_m: float,
                    reach_m: float, why: str = "") -> "CellWorld":
        """The measured world of a finished run from its record (to_dict(): the run log's last 'world') - the same plate plane
        (point with its k sigma_top, normal), top-face line, plate widths, gap and plate sides, ends; the solids' thickness from the
        design (thickness, backing) and the clearance / reach of the platform now. ValueError when the record is not a measured world
        of this schema."""
        if str((rec or {}).get("schema")) != SCHEMA:
            raise ValueError(f"cell world record: schema {(rec or {}).get('schema')!r} is not {SCHEMA}")
        top = next((p for p in rec.get("planes") or [] if p.get("id") == NAMESPACE + "plate_top"), None)
        if top is None or top.get("k_sigma_m") is None:
            raise ValueError("cell world record: no measured plate top (the run's world is still the prior of the rough points)")
        ext = top.get("extent") or {}
        line = np.asarray(ext.get("line_m"), dtype=float).reshape(2, 3)
        w = cls(up=rec["up"], thickness_m=thickness_m, backing_m=backing_m, design_length_m=design_length_m, clearance_m=clearance_m,
                reach_m=reach_m, base_frame=str(rec.get("base_frame") or "link0"), plate_widths_m=tuple(ext["plate_widths_m"]),
                gap_m=float(ext.get("gap_m") or 0.0))
        n = unit(np.asarray(top["normal"], dtype=float))
        ks = float(top["k_sigma_m"])
        # update_measured adds n k sigma_top to the top point: give it the recorded plane point less that, with k = 1
        w.update_measured(top_point=np.asarray(top["point_m"], dtype=float) - n * ks, sigma_top_m=ks, normal=n, k=1.0,
                          root_lo=line[0], root_hi=line[1], ends=dict(rec.get("ends") or {}),
                          why=why or f"the run's world v{rec.get('version')}")
        w.set_plate_sides(ext.get("left_plate_side_w"))
        return w

    def set_prior(self, A, B, *, hit_face_m: float) -> None:
        A = np.asarray(A, dtype=float).reshape(3)
        B = np.asarray(B, dtype=float).reshape(3)
        d = B - A
        L_hits = float(np.linalg.norm(d))
        if not L_hits > 0.0:
            raise ValueError("the two rough points coincide")
        u = d / L_hits
        n = unit(self.up - u * float(self.up @ u))
        t_hit = float(hit_face_m)
        ext = max(0.0, self.L_design - L_hits)
        hi = A if float(self.up @ A) >= float(self.up @ B) else B
        self.plate_point = hi + n * t_hit
        self.plate_normal = n
        self.plate_sigma_m = None
        self.line = (A - u * ext + n * t_hit, B + u * ext + n * t_hit)
        self.depth_below_top_m = t_hit + self.t + self.b
        self.ends = {"rough_points_m": [_l(A), _l(B)], "beyond_each_m": ext}
        self.source = (f"prior: the two rough points {L_hits * 1e3:.0f} mm apart, each extended outward by the design length they do not "
                       f"span ({ext * 1e3:.0f} mm), plate top = their line + t_hit {t_hit * 1e3:.0f} mm (the rough point may be at the root)")
        self.version += 1

    def update_measured(self, *, top_point, sigma_top_m: float, normal, k: float, root_lo, root_hi, ends: Optional[dict] = None,
                        why: str = "") -> None:
        """Measured world: plate top = top_point + n^ k sigma_top (n^ toward up), specimen along root_lo → root_hi (the estimated
        ends with -/+ k sigma_e, S side first) projected onto that plane. n^ is made perpendicular to the estimated line — the section
        frame of spec §9 (seam_estimate._section_basis: u^ = the line, n^ ⟂ u^): the line and the plate normal are estimated separately
        and are not exactly perpendicular, and the specimen box, the envelope plane and the pass frames must be one rigid frame."""
        lo = np.asarray(root_lo, dtype=float).reshape(3)
        hi = np.asarray(root_hi, dtype=float).reshape(3)
        n = unit(np.asarray(normal, dtype=float).reshape(3))
        d = hi - lo
        if float(np.linalg.norm(d)) > 0.0:
            u = unit(d)
            n = unit(n - u * float(n @ u))
        if float(n @ self.up) < 0.0:
            n = -n
        ks = float(k) * float(sigma_top_m)
        p = np.asarray(top_point, dtype=float).reshape(3) + n * ks
        self.plate_point = p
        self.plate_normal = n
        self.plate_sigma_m = ks
        self.line = (lo + n * float((p - lo) @ n), hi + n * float((p - hi) @ n))
        self.depth_below_top_m = 2.0 * ks + self.t + self.b
        self.ends = dict(ends or {})
        self.source = (f"measured: plate plane + k sigma_top ({ks * 1e3:.2f} mm), specimen between the estimated ends -/+ k sigma_e"
                       + (f" ({why})" if why else ""))
        self.version += 1

    def set_plate_sides(self, left_sign_w: Optional[int]) -> None:
        """Which side of the seam the design's left plate is on (+1 = +w = left looking S->E, -1, None = not known: the wider plate
        on both sides). A change makes a new world version."""
        v = None if left_sign_w is None else (1 if int(left_sign_w) > 0 else -1)
        if v != self.left_sign_w:
            self.left_sign_w = v
            self.version += 1

    def half_widths(self) -> tuple:
        """(extent on +w, extent on -w) from the seam centre [m]: half the gap + the plate on that side."""
        g = 0.5 * self.gap
        wl, wr = self.plate_widths
        if self.left_sign_w is None:
            m = max(wl, wr)
            return g + m, g + m
        return (g + wl, g + wr) if self.left_sign_w > 0 else (g + wr, g + wl)

    @property
    def W(self) -> float:
        """The specimen box width across the seam [m]."""
        hp, hm = self.half_widths()
        return hp + hm

    def set_fixed_bodies(self, rows) -> None:
        """The robot's fixed body (platform fixed_bodies(): [{link, lo, hi} | {link, lo: None, why}]) — the floor stops C short of it."""
        self.fixed_bodies = [dict(r) for r in (rows or [])]
        self.fixed_bodies_known = True

    def set_reach_below(self, depth_m: Optional[float]) -> None:
        """The platform's bound on how far below its base origin any part of the arm and tool can reach (the robot model: the summed
        joint offsets of the chain + the tool's extent) - the under solid goes down to there. None = unknown: no under solid."""
        v = None if depth_m is None or not math.isfinite(float(depth_m)) else float(depth_m)
        self.reach_below_m = v          # platform knowledge like the fixed bodies (set_fixed_bodies): not a new world version

    # ---- derived geometry ------------------------------------------------------------------------------------------------
    def specimen(self) -> Solid:
        a, b = self.line
        R = _frame(b - a, self.plate_normal)
        L = float(np.linalg.norm(b - a))
        D = float(self.depth_below_top_m)
        hp, hm = self.half_widths()
        return Solid(SPECIMEN_ID, 0.5 * (a + b) - R[:, 2] * (0.5 * D) + R[:, 1] * (0.5 * (hp - hm)), R, np.array([L, hp + hm, D]),
                     self.source)

    def under(self) -> Optional[Solid]:
        """The space below the specimen (module docstring 'under'): the specimen's footprint (horizontal axes along and across the
        seam) from the support plane down to the arm's lowest reach; None when that reach is unknown or not below the support."""
        if self.reach_below_m is None:
            return None
        bottom = float(self.base_origin @ self.up) - self.reach_below_m
        top = self.floor_m
        if not top > bottom:
            return None
        a, b = self.line
        ah = (b - a) - self.up * float((b - a) @ self.up)
        if not float(np.linalg.norm(ah)) > 1e-9:
            return None
        ah = unit(ah)
        bh = unit(np.cross(self.up, ah))
        P = self.specimen().corners()
        pa, pb = P @ ah, P @ bh
        c = ah * (0.5 * (pa.min() + pa.max())) + bh * (0.5 * (pb.min() + pb.max()))
        c = c - self.up * float(c @ self.up) + self.up * (0.5 * (top + bottom))
        return Solid(UNDER_ID, c, np.column_stack([ah, bh, self.up]),
                     np.array([float(pa.max() - pa.min()), float(pb.max() - pb.min()), top - bottom]),
                     f"under the specimen: its footprint from the support plane {top:.4f} m down to the arm's lowest reach {bottom:.3f} m")

    def _top_corners(self) -> np.ndarray:
        a, b = self.line
        w = _frame(b - a, self.plate_normal)[:, 1]
        hp, hm = self.half_widths()
        return np.array([a + w * hp, a - w * hm, b + w * hp, b - w * hm])

    @property
    def plate_top_m(self) -> float:
        """Height (up) of the specimen's top along its centre line (its highest point) — the 1G plate-top rule."""
        a, b = self.line
        return float(max(self.up @ a, self.up @ b, self.up @ self.plate_point))

    @property
    def floor_m(self) -> float:
        """Support height (up): the lowest corner of the specimen's top face - thickness - backing."""
        return float(np.min(self._top_corners() @ self.up)) - self.t - self.b

    def _ab(self) -> tuple:
        """Horizontal axes: a = from the base origin toward the specimen centre, b = up x a."""
        c = self.specimen().center - self.base_origin
        a = c - self.up * float(c @ self.up)
        if not float(np.linalg.norm(a)) > 1e-9:
            w = _frame(self.line[1] - self.line[0], self.plate_normal)[:, 1]
            a = w - self.up * float(w @ self.up)
        a = unit(a)
        return a, unit(np.cross(self.up, a))

    def _floor_ranges(self) -> tuple:
        """Floor solid ranges in (a, b, up): the specimen footprint + reach, slab C below the support plane."""
        a, b = self._ab()
        P = self.specimen().corners() - self.base_origin
        ca, cb = P @ a, P @ b
        h = self.floor_m - float(self.base_origin @ self.up)
        return ((float(ca.min()) - self.reach, float(ca.max()) + self.reach), (float(cb.min()) - self.reach, float(cb.max()) + self.reach),
                (h - self.C, h))

    def floor_object_ranges(self) -> tuple:
        """The floor MoveIt object (solid grown by C) in (a, b, up), stopped C short of every fixed body that reaches into it (within C):
        of its four vertical faces (robot side, far side, the two lateral sides) the one moved to C beyond that body is the one that
        removes the least floor - the robot's own base and torso cut the robot side, a fixture or test bed of the robot model beyond or
        beside the specimen cuts the far or a lateral side (before 2026-10-13 every body cut the robot side, and a body beyond the
        specimen took the whole floor away). The robot's own body is never inside the world → (ra, rb, ru, [cut notes])."""
        (a0, a1), (b0, b1), (u0, u1) = self._floor_ranges()
        C = self.C
        ra, rb, ru = [a0 - C, a1 + C], [b0 - C, b1 + C], (u0 - C, u1 + C)
        a, b = self._ab()
        notes = []
        for fb in self.fixed_bodies:
            if fb.get("lo") is None:
                notes.append(f"{fb.get('link')}: geometry unknown ({fb.get('why')}) - not used")
                continue
            lo, hi = np.asarray(fb["lo"], dtype=float), np.asarray(fb["hi"], dtype=float)
            X = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - self.base_origin
            pa, pb, pu = X @ a, X @ b, X @ self.up
            if (pu.max() <= ru[0] - C or pu.min() >= ru[1] + C or pb.max() <= rb[0] - C or pb.min() >= rb[1] + C
                    or pa.max() <= ra[0] - C or pa.min() >= ra[1] + C):
                continue                                           # not within C of the floor object
            la, lb = ra[1] - ra[0], rb[1] - rb[0]
            cuts = [(min(la, pa.max() + C - ra[0]) * lb, "robot side", "a0", float(pa.max()) + C),
                    (min(la, ra[1] - (pa.min() - C)) * lb, "far side", "a1", float(pa.min()) - C),
                    (min(lb, pb.max() + C - rb[0]) * la, "lateral side (-b)", "b0", float(pb.max()) + C),
                    (min(lb, rb[1] - (pb.min() - C)) * la, "lateral side (+b)", "b1", float(pb.min()) - C)]
            _loss, side, face, v = min(cuts, key=lambda c: c[0])
            if face == "a0":
                ra[0] = v
            elif face == "a1":
                ra[1] = v
            elif face == "b0":
                rb[0] = v
            else:
                rb[1] = v
            notes.append(f"{side} stops {C * 1e3:.0f} mm short of {fb.get('link')} (to {v:.3f} m from the base along the "
                         + ("robot-specimen direction)" if face.startswith("a") else "horizontal normal to it)"))
        return (ra[0], ra[1]), (rb[0], rb[1]), ru, notes

    # ---- consumers ---------------------------------------------------------------------------------------------------------
    def envelope(self) -> dict:
        """What the straight-leg envelope uses: plate plane (point, normal), plate top height, support height."""
        return {"plate_point_m": self.plate_point.copy(), "plate_normal": self.plate_normal.copy(), "plate_top_m": self.plate_top_m,
                "floor_m": self.floor_m}

    def moveit_objects(self) -> list:
        """[{id, size (3,), T (4x4), source}] — the solids grown by C (the floor's robot side C short of the fixed body)."""
        sp = self.specimen()
        objs = [{"id": sp.id, "size": _l(np.asarray(sp.size) + 2.0 * self.C), "T": sp.T(), "source": sp.source}]
        ra, rb, ru, _notes = self.floor_object_ranges()
        if ra[1] > ra[0]:
            a, b = self._ab()
            c = self.base_origin + a * (0.5 * (ra[0] + ra[1])) + b * (0.5 * (rb[0] + rb[1])) + self.up * (0.5 * (ru[0] + ru[1]))
            objs.append({"id": FLOOR_ID, "size": [ra[1] - ra[0], rb[1] - rb[0], ru[1] - ru[0]],
                         "T": make_transform(np.column_stack([a, b, self.up]), c),
                         "source": f"support below the specimen at {self.floor_m:.4f} m (lowest top corner - thickness - backing)"})
        uo = self.under_object(grow_top=True)
        if uo is not None:
            objs.append(uo)
        assert all(o["id"].startswith(NAMESPACE) for o in objs)
        return objs

    def under_object(self, *, grow_top: bool) -> Optional[dict]:
        """The under solid as a MoveIt object, grown by C across (and on top when grow_top; the bottom is beyond the arm's reach
        already). grow_top False = its top stays on the support plane: the weld-pose check (D14), where the torch works in the groove
        above that plane and the specimen itself is the weld scene's mesh. None when there is no under solid."""
        ud = self.under()
        if ud is None:
            return None
        g_top = self.C if grow_top else 0.0
        size = np.asarray(ud.size) + np.array([2.0 * self.C, 2.0 * self.C, g_top])
        return {"id": UNDER_ID, "size": _l(size), "T": make_transform(ud.R, ud.center + ud.R[:, 2] * (0.5 * g_top)),
                "source": ud.source + (f", grown by C {self.C * 1e3:.0f} mm" if grow_top else
                                       f", grown by C {self.C * 1e3:.0f} mm across, top on the support plane"),
                "allow_fixed_links": True}

    def describe(self, oid: str) -> str:
        C = self.C * 1e3
        if oid == SPECIMEN_ID:
            sp = self.specimen()
            return (f"the specimen ({sp.size[0] * 1e3:.0f} x {sp.size[1] * 1e3:.0f} x {sp.size[2] * 1e3:.0f} mm, top face on the plate "
                    f"plane at {self.plate_top_m:.4f} m) grown by C {C:.0f} mm")
        if oid == FLOOR_ID:
            return f"the support below the specimen (top {self.floor_m:.4f} m) grown by C {C:.0f} mm"
        if oid == UNDER_ID:
            return (f"the space under the specimen (support / fixture, from {self.floor_m:.4f} m down to the arm's lowest reach) grown "
                    f"by C {C:.0f} mm - no part of the arm goes there")
        if str(oid).startswith(NAMESPACE):
            return "an object of ours"
        return "an object that is not ours"

    def explain(self, contacts) -> str:
        """contacts [{link, object, depth_m, kind}] (platform state_contacts; kind 'self' = two robot links) or [(link, object)] →
        'link6 is inside hongik/floor (the support below the specimen …)'."""
        parts = []
        for c in contacts or []:
            if not isinstance(c, dict):
                c = {"link": c[0], "object": c[1] if len(c) > 1 else ""}
            link, obj = str(c.get("link") or "?"), str(c.get("object") or "")
            if c.get("kind") == "self":
                parts.append(f"{link} touches {obj} (self-collision)")
                continue
            depth = c.get("depth_m")
            d = f", {float(depth) * 1e3:.1f} mm deep" if depth is not None and math.isfinite(float(depth)) and float(depth) > 0.0 else ""
            parts.append(f"{link} is inside {obj} ({self.describe(obj)}{d})")
        return "; ".join(parts)

    def to_dict(self) -> dict:
        """Scene-update data (interface design §2.15 WorldUpdate): version, planes, objects (as applied to MoveIt: solid grown by
        C), the envelope values and the fixed-body cut."""
        sp = self.specimen()
        ra, rb, ru, notes = self.floor_object_ranges()

        def obj(o):
            from scipy.spatial.transform import Rotation
            T = np.asarray(o["T"], dtype=float)
            return {"id": o["id"], "shape": "box", "size_m": _l(o["size"]), "pose": {"position_m": _l(T[:3, 3]),
                    "quat_xyzw": _l(Rotation.from_matrix(T[:3, :3]).as_quat())}, "padding_m": self.C, "source": o["source"]}
        return {
            "schema": SCHEMA, "version": int(self.version), "base_frame": self.base_frame, "up": _l(self.up), "source": self.source,
            "planes": [{"id": NAMESPACE + "plate_top", "point_m": _l(self.plate_point), "normal": _l(self.plate_normal),
                        "k_sigma_m": self.plate_sigma_m, "extent": {"line_m": [_l(self.line[0]), _l(self.line[1])], "width_m": self.W,
                                                                    "half_widths_m": list(self.half_widths()),
                                                                    "plate_widths_m": list(self.plate_widths), "gap_m": self.gap,
                                                                    "left_plate_side_w": self.left_sign_w}},
                       {"id": NAMESPACE + "support", "point_m": _l(self.up * self.floor_m), "normal": _l(self.up),
                        "rule": "lowest corner of the specimen top face - thickness - backing"}],
            "objects": [obj(o) for o in self.moveit_objects()],
            "specimen_solid_size_m": _l(sp.size), "ends": self.ends,
            "envelope": {"plate_point_m": _l(self.plate_point), "plate_normal": _l(self.plate_normal), "plate_top_m": self.plate_top_m,
                         "floor_m": self.floor_m},
            "clearance_m": self.C, "reach_m": self.reach, "reach_below_m": self.reach_below_m,
            "fixed_bodies": [{"link": f.get("link"), "lo": None if f.get("lo") is None else _l(f["lo"]),
                              "hi": None if f.get("hi") is None else _l(f["hi"]), **({"why": f["why"]} if f.get("why") else {})}
                             for f in self.fixed_bodies],
            "floor_cut": notes,
        }

    def summary(self) -> str:
        """One line for the log."""
        objs = ", ".join(f"{o['id']} {np.round(o['size'], 3).tolist()}" for o in self.moveit_objects())
        _ra, _rb, _ru, notes = self.floor_object_ranges()
        return (f"world v{self.version} ({self.source}): plate plane point {np.round(self.plate_point, 4).tolist()} normal "
                f"{np.round(self.plate_normal, 4).tolist()}, plate top {self.plate_top_m:.4f} m, support {self.floor_m:.4f} m; MoveIt "
                f"objects (grown by C {self.C * 1e3:.0f} mm): {objs}" + (f"; floor: {'; '.join(notes)}" if notes else ""))
