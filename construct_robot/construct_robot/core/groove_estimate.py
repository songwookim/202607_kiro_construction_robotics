"""KIRO groove estimator for a single-bevel 1G groove from one scan (pure numpy, no ROS).

The groove faces are flat: fit them as planes over the whole scanned length (robust, reflections at the inner
corners are rejected), then
  corner 1 (root) = bevel plane x bottom plane         corner 2 = bottom plane x vertical (90 deg) plane
  seam frame      u = corner-1 line (scan direction), n = plate-top normal, w = n x u
  ends S / E      first / last laser profile that still sees both plate tops next to the groove (the scan must
                  start and end off the plate, otherwise the end is not observed)
Positions in metres (World); the report uses mm.
"""

import numpy as np


class GrooveEstimateError(ValueError):
    pass


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def fit_plane(points, iterations=6, k=2.5, floor=0.02):
    """Robust plane: (centroid, unit normal, inlier mask, rms) - iteratively re-fitted on |residual| < k * MAD."""
    points = np.asarray(points, float)
    if len(points) < 30:
        raise GrooveEstimateError(f"too few points for a plane ({len(points)})")
    keep = np.ones(len(points), bool)
    for _ in range(iterations):
        c = points[keep].mean(axis=0)
        _u, _s, vt = np.linalg.svd(points[keep] - c, full_matrices=False)
        normal = vt[2]
        r = (points - c) @ normal
        s = 1.4826 * np.median(np.abs(r[keep]))
        keep = np.abs(r) < max(k * s, floor)
    rms = float(np.sqrt(np.mean(((points[keep] - c) @ normal) ** 2)))
    return c, normal, keep, rms


def plane_line(p1, p2):
    """Intersection line of two planes (c, n): (point, unit direction)."""
    (c1, n1), (c2, n2) = p1, p2
    d = _unit(np.cross(n1, n2))
    A = np.array([n1, n2, d])
    b = np.array([n1 @ c1, n2 @ c2, d @ c1])
    return np.linalg.solve(A, b), d


def scan_axis(origins):
    """Travel axis of the recording, oriented along its first motion."""
    o = np.asarray(origins, float)
    c = o.mean(axis=0)
    _u, _s, vt = np.linalg.svd(o - c, full_matrices=False)
    u = vt[0]
    far = o[np.argmax(np.linalg.norm(o - o[0], axis=1))]          # robust to a standstill at the start
    if (far - o[0]) @ u < 0:
        u = -u
    return u


# the specimen (Hongik job groove_10T_1G "specimen" / "qc_targets"; the application reads the job file)
DEFAULT_SPEC = {"thickness_mm": 10.0, "bevel_angle_deg": 30.0, "root_gap_mm": 5.0, "weld_length_mm": 200.0}
DEFAULT_TOL = {"weld_length_mm": 1.0, "root_gap_mm": 0.5, "groove_depth_mm": 0.5, "bevel_angle_deg": 2.0}
LENGTH_AMBIGUOUS_MM = 5.0          # |length - design| above this: an end was not read correctly (the estimate is refused)


def _mode(values, step):
    values = np.asarray(values, float)
    hist, edges = np.histogram(values, bins=np.arange(values.min(), values.max() + step, step))
    return float(edges[np.argmax(hist)] + step / 2.0)


def _ransac_line(pts, angle_range, rng, tol=0.15, iters=300):
    """Best line through 2D points (lateral, height) with its angle from horizontal in angle_range [deg]:
    (centre, unit direction, inlier fraction, rms) or None."""
    if len(pts) < 20:
        return None
    best = None
    best_count = 0
    x, y = pts.T
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        if np.hypot(*d) < 0.5:
            continue
        ang = np.degrees(np.arctan2(abs(d[1]), abs(d[0])))
        if not angle_range[0] <= ang <= angle_range[1]:
            continue
        nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
        # Two coordinates only: avoid allocating an N x 2 residual matrix and
        # dispatching a BLAS product for every trial.
        inl = np.abs((x - x[i]) * nrm[0] + (y - y[i]) * nrm[1]) < tol
        count = np.count_nonzero(inl)
        if best is None or count > best_count:
            best = inl
            best_count = count
    if best is None or best_count < 15:
        return None
    q = pts[best]
    c = q.mean(axis=0)
    _u, _s, vt = np.linalg.svd(q - c, full_matrices=False)
    return c, vt[0], float(best.mean()), float(np.sqrt(np.mean(((q - c) @ vt[1]) ** 2)))


def _line_angle(line):
    d = line[1]
    return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))


def _cross(l1, l2):
    (c1, d1), (c2, d2) = l1[:2], l2[:2]
    t = np.linalg.solve(np.array([d1, -d2]).T, c2 - c1)
    return c1 + d1 * t[0]


def _sections(X, top, centre, spec, rng, length_mm=20.0):
    """Corners of each length_mm cross-section (X: along, lateral, height [mm] in a frame roughly along the seam).
    Every section finds its own plate top and groove position (the seam may run skewed to the frame / tilted)."""
    depth = spec["thickness_mm"]
    bevel_from_h = 90.0 - spec["bevel_angle_deg"]                # bevel face angle from the plate (60 deg)
    bevel_w = depth * np.tan(np.radians(spec["bevel_angle_deg"]))   # lateral width of the bevel face (5.8 mm)
    gap_d = spec["root_gap_mm"]
    band = (X[:, 2] > top - depth - 4.0) & (X[:, 2] < top + 4.0) & (np.abs(X[:, 1] - centre) < 35.0)
    a_lo, a_hi = np.percentile(X[band & (np.abs(X[:, 1] - centre) < 12.0), 0], [1, 99])
    rows, skipped = [], {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1
    for a0 in np.arange(a_lo, a_hi - 10.0, length_mm):
        sec = band & (X[:, 0] >= a0) & (X[:, 0] < a0 + length_mm)
        S3 = X[sec]
        S2 = S3[:, 1:3]
        if len(S2) < 200:
            skip("few points")
            continue
        near_top = np.abs(S2[:, 1] - top) < 3.0
        if near_top.sum() < 50:
            skip("no plate top")
            continue
        top_s = _mode(S2[near_top, 1], 0.25)
        deep = S2[(S2[:, 1] < top_s - 0.6 * depth) & (S2[:, 1] > top_s - depth - 3.0)]
        if len(deep) < 50:
            skip("no groove")
            continue
        c_loc = _mode(deep[:, 0], 1.0)
        bottom_s = _mode(deep[:, 1], 0.25)
        if not 0.7 * depth < top_s - bottom_s < 1.3 * depth:
            skip("depth")
            continue
        win = np.abs(S2[:, 0] - c_loc) < gap_d + bevel_w + 6.0
        bot_pts = S2[win & (np.abs(S2[:, 1] - bottom_s) < 1.0) & (np.abs(S2[:, 0] - c_loc) < gap_d + 2.0)]
        bot = _ransac_line(bot_pts, (0.0, 8.0), rng)
        walls = S2[win & (S2[:, 1] > bottom_s + 1.0) & (S2[:, 1] < top_s - 1.5)]
        bevel = _ransac_line(walls, (bevel_from_h - 12.0, bevel_from_h + 12.0), rng)
        if bot is None or bevel is None:
            skip("bottom" if bot is None else "bevel")
            continue
        c1 = _cross(bevel, bot)
        side = -1 if bevel[1][0] * bevel[1][1] < 0 else 1            # the bevel plate's side (where the bevel rises)
        # corner 2: the 90 deg face x bottom; where that face is hidden from the laser, the far end of the bottom
        nb = np.array([-bot[1][1], bot[1][0]])
        far = (bot_pts[:, 0] - c1[0]) * -side
        on_bot = (np.abs((bot_pts - bot[0]) @ nb) < 0.3) & (far > 0)
        gap_bottom = float(np.percentile(far[on_bot], 99.5)) if on_bot.sum() >= 10 else np.nan
        vert = _ransac_line(walls[(walls[:, 0] - c1[0]) * -side > 1.0], (80.0, 90.0), rng)
        c2 = _cross(vert, bot) if vert is not None else None
        gap = abs(c2[0] - c1[0]) if c2 is not None else np.nan
        if c2 is not None and (abs(gap - gap_d) > 3.0 or (np.isfinite(gap_bottom) and abs(gap - gap_bottom) > 1.0)):
            c2, gap = None, np.nan                                     # not this groove's 90 deg face
        gap_source = "face" if c2 is not None else None
        if c2 is None and np.isfinite(gap_bottom) and abs(gap_bottom - gap_d) <= 3.0:
            gap, gap_source = gap_bottom, "bottom end"
            c2 = np.array([c1[0] - side * gap, bot[0][1] + (c1[0] - side * gap - bot[0][0]) * bot[1][1] / bot[1][0]])
        rel = S2[:, 0] - c1[0]
        on_top = (np.abs(S2[:, 1] - top_s) < 1.0) & (np.abs(rel) < 25.0)
        tb = on_top & (rel * side > bevel_w + 1.5)
        tv = on_top & (rel * -side > (gap if np.isfinite(gap) else gap_d) + 1.5)
        if tb.sum() < 20 or tv.sum() < 20:
            skip("plate tops")
            continue
        rows.append({"along": a0 + length_mm / 2.0, "c1": c1, "c2": c2, "side": side, "gap": gap, "gap_source": gap_source,
                     "depth": 0.5 * (np.median(S2[tb, 1]) + np.median(S2[tv, 1])) - c1[1],
                     "bevel_deg": 90.0 - _line_angle(bevel), "bottom_rms": bot[3], "bevel_rms": bevel[3],
                     "bevel_inliers": bevel[2], "top_points": S3[tb | tv]})
    return rows, skipped


def _line3d(rows_mm, ref, basis, u_hint):
    """Robust 3D line through section corners (along, lateral, height [mm] in the frame basis) -> World."""
    pts_w = ref + rows_mm @ basis.T * 1e-3
    keep = np.ones(len(pts_w), bool)
    for _ in range(3):
        c = pts_w[keep].mean(axis=0)
        _u, _s, vt = np.linalg.svd(pts_w[keep] - c, full_matrices=False)
        d = vt[0] if vt[0] @ u_hint > 0 else -vt[0]
        r = np.linalg.norm(np.cross(pts_w - c, d), axis=1)
        keep = r < max(3 * np.median(r), 0.0003)
    return c, d, float(np.sqrt(np.mean(r[keep] ** 2)))


def _profile_medians(values, frame_index):
    """Exact per-profile medians, visiting each point once rather than masking the whole cloud per profile."""
    values, index = np.asarray(values), np.asarray(frame_index)
    if not len(index):
        return np.empty(0, dtype=float)
    if np.any(index[1:] < index[:-1]):
        order = np.argsort(index, kind="stable")
        values, index = values[order], index[order]
    starts = np.r_[0, np.flatnonzero(index[1:] != index[:-1]) + 1]
    stops = np.r_[starts[1:], len(index)]
    return np.asarray([np.median(values[a:b]) for a, b in zip(starts, stops)])


def estimate_groove(points, frame_index, origins, *, spec=None, depth_mm=None, min_plate_points=40):
    """Groove geometry of one scan. points (N, 3) World m, frame_index (N,), origins (F, 3) optical origins;
    spec: the specimen (DEFAULT_SPEC keys) - its bevel angle / depth / root gap set the search windows."""
    spec = {**DEFAULT_SPEC, **(spec or {})}
    if depth_mm is not None:
        spec["thickness_mm"] = float(depth_mm)
    P = np.asarray(points, float)
    fi = np.asarray(frame_index, int)
    up = np.array([0.0, 0.0, 1.0])
    ref = P.mean(axis=0)
    rng = np.random.default_rng(0)
    u_axis = scan_axis(origins)
    u0 = u_axis
    # two rounds: the scan axis may run a few degrees off the seam - the second round works in a frame along the
    # root line found in the first, so the groove stays put in the cross-sections
    for _round in range(2):
        w0 = _unit(np.cross(up, u0))
        n0 = np.cross(u0, w0)
        basis = np.c_[u0, w0, n0]
        X = (P - ref) @ basis * 1e3                                  # along, lateral, height [mm]
        # plate top = the dominant height (the plates are the largest surface)
        top = _mode(X[:, 2], 0.5)
        deep = X[(X[:, 2] < top - 0.6 * spec["thickness_mm"]) & (X[:, 2] > top - spec["thickness_mm"] - 3.0)]
        if len(deep) < 100:
            raise GrooveEstimateError("no groove found (no points about one groove depth below the plates)")
        centre = _mode(deep[:, 1], 1.0)
        rows, skipped = _sections(X, top, centre, spec, rng)
        if len(rows) < 3:
            raise GrooveEstimateError(f"the root corner was found in only {len(rows)} sections (skipped: {skipped})")
        sides = [r["side"] for r in rows]
        side = max(set(sides), key=sides.count)
        used = [r for r in rows if r["side"] == side]
        C1 = np.array([[r["along"], r["c1"][0], r["c1"][1]] for r in used])
        c1_point, u, c1_rms = _line3d(C1, ref, basis, u_axis)
        u0 = _unit(u - (u @ up) * up)
    bevel_side = "left" if side < 0 else "right"
    # plate normal from the plate tops next to the groove; w = n x u
    tops_w = ref + np.vstack([r["top_points"] for r in used]) @ basis.T * 1e-3
    _c, n_plate, _keep, plate_rms = fit_plane(tops_w)
    n_plate = n_plate if n_plate @ up > 0 else -n_plate
    n = _unit(n_plate - (n_plate @ u) * u)
    w = np.cross(n, u)
    C2 = [[r["along"], r["c2"][0], r["c2"][1]] for r in used if r["c2"] is not None]
    c2 = _line3d(np.array(C2), ref, basis, u_axis) if len(C2) >= 3 else None
    gaps = [r["gap"] for r in used if r["gap_source"] == "face"]          # the 90 deg face seen; else the bottom end
    gaps = gaps if len(gaps) >= 3 else [r["gap"] for r in used if np.isfinite(r["gap"])]
    depths = [r["depth"] for r in used]

    # ends: where the plate tops next to the groove start / stop, from the points of each side (a skewed laser line
    # does not shift it); the scan must start and end off the plate, otherwise that end is not observed
    Y = (P - c1_point) @ np.c_[u, w, n] * 1e3
    plate_h = float(np.median(depths))
    frame_along = _profile_medians(Y[:, 0], fi)
    scan_first, scan_last = float(frame_along.min()), float(frame_along.max())
    gap = float(np.median(gaps)) if gaps else spec["root_gap_mm"]
    bevel_w = plate_h * np.tan(np.radians(spec["bevel_angle_deg"]))
    ends = []
    for lo, hi, sign in ((bevel_w + 2.0, bevel_w + 12.0, side), (gap + 2.0, gap + 12.0, -side)):   # bevel / 90 deg plate
        rel = Y[:, 1] * sign
        sel = (rel > lo) & (rel < hi) & (np.abs(Y[:, 2] - plate_h) < 1.0)
        if sel.sum() < 200:
            raise GrooveEstimateError("too few plate-top points next to the groove")
        ends.append((np.percentile(Y[sel, 0], 0.2), np.percentile(Y[sel, 0], 99.8)))
    a_s = float(np.mean([e[0] for e in ends]))
    a_e = float(np.mean([e[1] for e in ends]))
    spacing = float(np.median(np.diff(np.sort(frame_along))))
    if a_s - scan_first < 2.0:
        raise GrooveEstimateError("the scan starts on the plate: teach the scan start (C) further past the plate end")
    if scan_last - a_e < 2.0:
        raise GrooveEstimateError("the scan ends on the plate: teach the scan end (D) further past the plate end")

    def on_line(line, a_mm):
        p, d = line
        t = ((c1_point + u * a_mm * 1e-3) - p) @ d
        return p + d * t

    out = {"u": u, "w": w, "n": n, "bevel_side": bevel_side,
           "S": on_line((c1_point, u), a_s), "E": on_line((c1_point, u), a_e),
           "corner1_S": on_line((c1_point, u), a_s), "corner1_E": on_line((c1_point, u), a_e),
           "length_mm": a_e - a_s, "end_resolution_mm": spacing / 2.0,
           "sections": len(used), "sections_skipped": skipped, "sections_other_side": len(rows) - len(used),
           "corner1_line_rms_mm": c1_rms * 1e3, "plate_rms_mm": plate_rms * 1e3,
           "scan_skew_deg": float(np.degrees(np.arccos(min(1.0, abs(u_axis @ _unit(u - (u @ up) * up)))))),
           "end_side_difference_mm": (float(ends[0][0] - ends[1][0]), float(ends[0][1] - ends[1][1])),
           "bottom_rms_mm": float(np.median([r["bottom_rms"] for r in used])),
           "bevel_rms_mm": float(np.median([r["bevel_rms"] for r in used])),
           "bevel_deg_from_vertical": float(np.median([r["bevel_deg"] for r in used])), "depth_mm": plate_h,
           "root_gap_sections": len(gaps),
           "root_gap_from_face": sum(r["gap_source"] == "face" for r in used)}
    if gaps:
        out["root_gap_mm"] = gap
    if c2 is not None:
        out["corner2_S"], out["corner2_E"] = on_line(c2[:2], a_s), on_line(c2[:2], a_e)
    return out


def groove_qc(groove, spec=None, tol=None):
    """Measured groove vs the specimen design: rows {name, measured, design, tol, diff, ok} (ok None: not measured)."""
    spec = {**DEFAULT_SPEC, **(spec or {})}
    tol = {**DEFAULT_TOL, **(tol or {})}
    rows = []
    for name, measured, design in (("weld_length_mm", groove.get("length_mm"), spec["weld_length_mm"]),
                                   ("root_gap_mm", groove.get("root_gap_mm"), spec["root_gap_mm"]),
                                   ("groove_depth_mm", groove.get("depth_mm"), spec["thickness_mm"]),
                                   ("bevel_angle_deg", groove.get("bevel_deg_from_vertical"), spec["bevel_angle_deg"])):
        diff = None if measured is None else float(measured) - float(design)
        rows.append({"name": name, "measured": None if measured is None else float(measured), "design": float(design),
                     "tol": float(tol[name]), "diff": diff, "ok": None if diff is None else abs(diff) <= tol[name]})
    return rows


def check_scan_covers_ends(points, frame_index, origins, *, depth_mm=10.0, edge_fraction=0.03, min_plate_points=40):
    """Fast check right after a scan: the first and the last profiles must not see the plate tops next to the groove
    (the scan has to pass both plate ends). Raises GrooveEstimateError with the side to extend."""
    P = np.asarray(points, float)
    fi = np.asarray(frame_index, int)
    u0 = scan_axis(origins)
    w0 = _unit(np.cross([0.0, 0.0, 1.0], u0))
    n0 = np.cross(u0, w0)
    X = (P - P.mean(axis=0)) @ np.c_[u0, w0, n0] * 1e3
    hist, edges = np.histogram(X[:, 2], bins=np.arange(X[:, 2].min(), X[:, 2].max() + 0.5, 0.5))
    top = float(edges[np.argmax(hist)] + 0.25)
    band = (X[:, 2] > top - depth_mm - 3.0) & (X[:, 2] < top - 0.6 * depth_mm)
    if band.sum() < 100:
        raise GrooveEstimateError("no groove found in the scan")
    centre = float(np.median(X[band, 1]))
    plate = (np.abs(X[:, 2] - top) < 1.5) & (np.abs(X[:, 1] - centre) > 6) & (np.abs(X[:, 1] - centre) < 25)
    if plate.sum() < 200:
        raise GrooveEstimateError("too few plate-top points next to the groove")
    counts = np.bincount(fi)
    along = np.bincount(fi, weights=X[:, 0])[counts > 0] / counts[counts > 0]   # each profile's position
    first, last = float(along.min()), float(along.max())
    plate_lo, plate_hi = np.percentile(X[plate, 0], [0.2, 99.8])
    if plate_lo - first < 2.0:
        raise GrooveEstimateError("the scan starts on the plate: teach the scan start (C) further past the plate end")
    if last - plate_hi < 2.0:
        raise GrooveEstimateError("the scan ends on the plate: teach the scan end (D) further past the plate end")
    return {"plate_start_mm": float(plate_lo - first), "plate_end_margin_mm": float(last - plate_hi)}
