"""Position correction of scan estimates (files + orchestration; math in core.scan_reference).

Files (construct_description/config/scan_teaching/):
  root1_reference.yaml   measured TCP poses (World <- right_manipulator_ee_point), per pass / end
  root1_correction.yaml  offsets (seam frame, per end), anchored on P1 or on P2's known position from the root
Corrected endpoints: <scan>/hongik_estimate/kiro_corrected/pass_<k>.yaml (KIRO schema, World, TCP poses).
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import yaml

from construct_robot.core.scan_reference import (
    ENDS,
    SeamFrame,
    aim_from_tcp,
    corrected_aim,
    offset_report,
    reference_offsets,
    tcp_from_aim,
    torch_angles,
    unit,
)
from construct_robot.core.scan_calib import same_calib
from construct_robot.io.scan_calib import recorded_calib
from construct_robot.io.teaching_yaml import atomic_yaml, teaching_config_dir

SCHEMA = "xy_endpoint_review_v1"
# right_manipulator_ee_point = the contact-tip end; the wire tip is Hongik's tcp_to_aim (21.2 mm) ahead along the
# torch axis (checked on the cell 2026-10-15: with the wire on the root corner the ee_point sits ~21 mm above the
# scanned corner). None = Hongik's value from the estimate; a number overrides it.
TCP_TO_WIRE_TIP_M = None


def scan_teaching_dir():
    return Path(teaching_config_dir()) / "scan_teaching"


def reference_path():
    return scan_teaching_dir() / "root1_reference.yaml"


def correction_path():
    return scan_teaching_dir() / "root1_correction.yaml"


def _pose_doc(position, quat):
    return {"position_m": {a: float(v) for a, v in zip("xyz", position)},
            "orientation_xyzw": {a: float(v) for a, v in zip("xyzw", quat)}}


def _pose_from_doc(doc):
    return (np.array([doc["position_m"][a] for a in "xyz"], float),
            np.array([doc["orientation_xyzw"][a] for a in "xyzw"], float))


def _read_reference_doc(path):
    doc = yaml.safe_load(path.read_text()) if path.is_file() else None
    doc = doc or {}
    if doc.get("format_version", 1) < 2:                 # v1: root start / goal only
        entries = [dict(doc[end], **{"pass": 1, "label": end}) for end in ENDS if end in doc]
        doc = {"entries": entries}
    doc.update(format_version=2, frame="World", tcp_frame="right_manipulator_ee_point")
    doc.setdefault("entries", [])
    return doc


SAME_END_M = 0.05     # references of one pass closer than this are the same end (the seam ends are ~200 mm apart)


def load_reference_set(path=None):
    """The active capture set, tied to one scan / specimen placement; legacy references have none."""
    return _read_reference_doc(Path(path or reference_path())).get("reference_set")


def begin_reference_set(scan_dir, relative="design_layout", *, baseline_relative=None):
    """Archive the previous teaching, then start an empty P1-P4 truth set. The active correction stays intact."""
    from construct_robot.application.kiro_estimate import check_estimate_length
    if relative not in RELATIVE_MODELS:
        raise ValueError(f"unknown relative model: {relative}")
    scan_dir = Path(scan_dir).expanduser().resolve()
    check_estimate_length(scan_dir)
    detail = load_estimate_detail(scan_dir)
    if [r["k"] for r in detail["passes"]] != [1, 2, 3, 4]:
        raise ValueError("a four-pass reference set requires an estimate with passes 1-4")
    baseline = load_correction()
    if baseline:
        check_correction_calib(baseline, detail)
    baseline_relative = baseline_relative or ((baseline or {}).get("reference_set") or {}).get("relative_model") or relative
    identifier = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + uuid4().hex[:8]
    backup = scan_teaching_dir() / "history" / f"{identifier}_before_four_pass_references"
    backup.mkdir(parents=True, exist_ok=False)
    for path in (reference_path(), correction_path(), design_layout_path(), pass_layout_path(),
                 scan_teaching_dir() / "pass_orientations.yaml"):
        if path.is_file():
            shutil.copy2(path, backup / path.name)
    # Read the plain YAML for a serializable baseline; omit nested capture histories.
    baseline = yaml.safe_load(correction_path().read_text()) if baseline else None
    if baseline:
        baseline.pop("reference_set", None)
        if baseline_relative == "measured_layout" and not baseline.get("measured_layout_m"):
            layout = load_pass_layout()
            if layout:
                baseline["measured_layout_m"] = {str(k): {end: v.tolist() for end, v in ends.items()}
                                                for k, ends in layout.items()}
    files = [scan_dir / "hongik_estimate" / "estimate.json"]
    files += sorted((scan_dir / "hongik_estimate" / "kiro").glob("pass_*.yaml"))
    files += [p for p in (scan_dir / "hongik_estimate" / "calib_used.json",) if p.is_file()]
    metadata = {"id": identifier, "source_scan": str(scan_dir), "relative_model": relative,
                "baseline_relative_model": baseline_relative,
                "created": datetime.now().isoformat(timespec="seconds"), "backup": str(backup),
                "baseline_correction": baseline,
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    atomic_yaml(reference_path(), {"format_version": 2, "frame": "World",
                                  "tcp_frame": "right_manipulator_ee_point", "reference_set": metadata,
                                  "entries": []})
    return metadata


def compute_reference_set_correction():
    """Fit the new P1-P4 captures to their own scan, preserving per-end residuals and the P1 anchor."""
    metadata = load_reference_set()
    if not metadata:
        raise ValueError("press New P1-P4 reference set first")
    for filename, digest in metadata["source_sha256"].items():
        path = Path(filename)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"reference-set estimate changed: {path}; start a new reference set")
    references = load_references()
    missing = [k for k in range(1, 5) if not any(r["pass"] == k for r in references)]
    if missing:
        raise ValueError("capture a correct point for every pass first; missing P" + ", P".join(map(str, missing)))
    # Captured attitudes describe the measurement. Position-only correction retains the welding attitudes.
    for ref in references:
        if "joints" not in ref:
            raise ValueError(f"P{ref['pass']} {ref['label']}: capture again to save its measured joint seed")
    if metadata["relative_model"] == "measured_layout":
        _learn_reference_set_layout(metadata, references)
    doc = compute_correction(metadata["source_scan"], references, anchor_pass=1, reference_set=metadata, persist=False)
    doc["notes"].append("Position-only truth correction: measurement attitudes are used for wire-tip conversion; "
                        "the existing per-pass welding attitudes and IK seeds are retained.")
    atomic_yaml(correction_path(), doc)
    return doc


def _learn_reference_set_layout(metadata, references):
    """Learn root-relative wire-tip positions; carry the previous end-to-end shape to unmeasured ends."""
    scan = metadata["source_scan"]
    detail = load_estimate_detail(scan)
    matched, _ = match_references(references, detail)
    # Root registration is independent of the pass layout. Use design_layout while learning a new measured layout.
    root = compute_correction(scan, references, anchor_pass=1,
                              reference_set=dict(metadata, relative_model="design_layout"), persist=False)
    frame = detail["frame"]
    baseline = metadata.get("baseline_correction")
    previous_model = metadata.get("baseline_relative_model", "design_layout")
    layout = {}
    for k in (2, 3, 4):
        L = next(r["L_m"] for r in detail["passes"] if r["k"] == k)
        ends = {}
        for end in ENDS:
            if (k, end) in matched:
                ref = matched[(k, end)]
                tip = aim_from_tcp(ref["position"], ref["quat"], L)
                ends[end] = frame.to_local(tip - pass_aim(detail, 1, end, root)[0])
        if len(ends) == 1:
            measured = next(iter(ends))
            other = "goal" if measured == "start" else "start"
            q = matched[(k, measured)]["quat"]
            old = {end: frame.to_local(pass_target_tcp(scan, k, end, q, baseline, previous_model, detail)[1]
                                      - pass_aim(detail, 1, end, baseline)[0]) for end in ENDS}
            ends[other] = ends[measured] + old[other] - old[measured]
        layout[str(k)] = {end: v.tolist() for end, v in ends.items()}
    atomic_yaml(pass_layout_path(), {"format_version": 1, "frame": "seam (along, lateral, normal) [m]",
                                    "source_scan": scan, "reference_set_id": metadata["id"],
                                    "computed": datetime.now().isoformat(timespec="seconds"), "passes": layout})


def check_reference_set_applied(correction):
    """Do not export / execute an older calibration while a new capture set is still pending."""
    metadata = load_reference_set()
    if not metadata:
        return
    fitted = (correction or {}).get("reference_set") or {}
    digest = hashlib.sha256(reference_path().read_bytes()).hexdigest()
    if fitted.get("id") != metadata["id"] or fitted.get("reference_sha256") != digest:
        raise ValueError("new / changed P1-P4 truth captures are pending: press Compute correction before Write / Build")


def save_reference(pass_k, label, position, quat, source, path=None, *, joints=None):
    """Store a measured correct TCP of pass k. label (start | goal) is what the operator chose; the end it belongs to
    is decided by position when the correction is computed (robust to a changed weld direction)."""
    if label not in ENDS:
        raise ValueError(f"label must be start | goal, got {label!r}")
    path = Path(path or reference_path())
    doc = _read_reference_doc(path)
    entry = dict(_pose_doc(position, quat), **{"pass": int(pass_k), "label": label, "source": str(source),
                                               "saved": datetime.now().isoformat(timespec="seconds")})
    if joints is not None:
        names, positions = joints
        expected = {f"right_manipulator_joint{i}" for i in range(1, 7)}
        if len(names) != 6 or set(names) != expected or len(positions) != 6 or not np.all(np.isfinite(positions)):
            raise ValueError("a reference needs the six named right-arm measured joints")
        entry["joint_state"] = {"names": list(names), "positions_rad": [float(v) for v in positions]}
    new = np.asarray(position, float)
    other = ENDS[1] if label == ENDS[0] else ENDS[0]
    kept = []
    for e in doc["entries"]:
        if int(e["pass"]) == int(pass_k):
            if float(np.linalg.norm(_pose_from_doc(e)[0] - new)) < SAME_END_M:
                continue                                 # the same end of this pass: replaced
            if e["label"] == label:                      # the other end carried this label: it is the other end
                e = dict(e, label=other)
        kept.append(e)
    doc["entries"] = kept + [entry]
    doc["entries"].sort(key=lambda e: (int(e["pass"]), e["label"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_yaml(path, doc)
    return path


def save_reference_end(end, position, quat, source, path=None):
    """Root-pass reference (pass 1)."""
    return save_reference(1, end, position, quat, source, path)


def load_references(path=None):
    """All measured references: [{pass, label, position (m), quat (xyzw), source, saved}]."""
    doc = _read_reference_doc(Path(path or reference_path()))
    out = []
    for e in doc["entries"]:
        position, quat = _pose_from_doc(e)
        ref = {"pass": int(e["pass"]), "label": e["label"], "position": position, "quat": quat,
               "source": e.get("source", ""), "saved": e.get("saved", "")}
        if "joint_state" in e:
            ref["joints"] = (tuple(e["joint_state"]["names"]), tuple(e["joint_state"]["positions_rad"]))
        out.append(ref)
    return out


def load_reference(path=None):
    """Root reference by label {start | goal: (position, quat)} (display / legacy)."""
    refs = {r["label"]: (r["position"], r["quat"]) for r in load_references(path) if r["pass"] == 1}
    if any(end not in refs for end in ENDS):
        raise ValueError(f"root-pass reference incomplete ({path or reference_path()}): measure both ends of pass 1")
    return refs


def match_references(references, detail):
    """Assign every reference to the nearer end (along the seam) of its pass in this estimate.
    Returns {(k, end): reference} (the latest saved wins) and notes."""
    frame = detail["frame"]
    rows = {r["k"]: r for r in detail["passes"]}
    matched, notes = {}, []
    for ref in sorted(references, key=lambda r: r["saved"]):
        row = rows.get(ref["pass"])
        if row is None:
            notes.append(f"pass {ref['pass']} {ref['label']}: the estimate has no pass {ref['pass']} - ignored")
            continue
        s = float(ref["position"] @ frame.u)
        end = min(ENDS, key=lambda e: abs(s - float(row[f"tcp_{e}"] @ frame.u)))
        if end != ref["label"]:
            notes.append(f"pass {ref['pass']}: reference saved as '{ref['label']}' lies at the plan's {end} end "
                         f"(weld direction) - used as the {end}")
        matched[(ref["pass"], end)] = ref
    return matched, notes


def _find(doc, key):
    if isinstance(doc, dict):
        if key in doc:
            return doc[key]
        for value in doc.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(doc, list):
        for value in doc:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def load_estimate_detail(scan_dir):
    """Seam frame and per pass: aim points, Hongik TCP poses, tcp_to_aim L of <scan_dir>/hongik_estimate."""
    out = Path(scan_dir).expanduser() / "hongik_estimate"
    est = json.loads((out / "estimate.json").read_text())
    frame = SeamFrame.from_estimate(est["u"], est["n_plate"])
    bevel = int(est.get("bevel_sign_w") or 1)
    passes = []
    for path in sorted((out / "kiro").glob("pass_[0-9]*.yaml")):
        if path.stem.endswith("_info"):
            continue
        doc = yaml.safe_load(path.read_text())
        info = yaml.safe_load(path.with_name(f"{path.stem}_info.yaml").read_text())
        aim = _find(info, "aim_point_World_m")
        L = _find(info, "tcp_to_aim_point_mm")
        if aim is None or L is None:
            raise ValueError(f"{path.name}: no aim point / tcp_to_aim in the info file")
        row = {"k": int(doc["pass"]), "role": _find(info, "role") or "", "hongik_L_m": float(L) * 1e-3,
               "L_m": wire_tip_model(float(L) * 1e-3)}
        for end in ENDS:
            row[f"tcp_{end}"], row[f"quat_{end}"] = _pose_from_doc(doc[f"current_{end}"])
            row[f"aim_{end}"] = np.asarray(aim[end], float)
        passes.append(row)
    if not passes:
        raise ValueError(f"no KIRO pass files in {out / 'kiro'}")
    return {"scan_dir": str(Path(scan_dir).expanduser()), "frame": frame,
            "plate_normal_world": unit(est["n_plate"]), "bevel_sign_w": bevel, "passes": passes,
            "calib": estimate_calib(scan_dir)}


def wire_tip_model(hongik_L_m):
    """TCP -> wire tip used for the targets: the calibrated TCP-frame vector (scan_teaching/wire_tip_offset.yaml)
    when present, else TCP_TO_WIRE_TIP_M, else Hongik's length along the torch axis."""
    from construct_robot.application.tcp_calibration import load_offset
    offset = load_offset()
    if offset is not None:
        return offset
    return float(hongik_L_m) if TCP_TO_WIRE_TIP_M is None else float(TCP_TO_WIRE_TIP_M)


def _wire_tip_doc(L):
    return [float(v) for v in np.asarray(L, float).reshape(-1)]


def estimate_calib(scan_dir):
    """{"file", "xyz", "rpy"} of the calib the estimate's poses are under (calib_used.json; older estimates: the
    recording's own calib). None when unknown."""
    used = Path(scan_dir).expanduser() / "hongik_estimate" / "calib_used.json"
    if used.is_file():
        return json.loads(used.read_text()).get("used")
    return recorded_calib(scan_dir)


def refresh_correction_for_wire_tip(path=None):
    """After a TCP (wire tip) calibration: recompute the stored correction on the scan it was computed on, with the
    same references and the new wire-tip model (only the model changed). Returns the new doc, or None when nothing
    to do (no correction / model unchanged / its scan is gone)."""
    doc = load_correction(path)
    if doc is None or not doc.get("source_scan") or not Path(doc["source_scan"]).is_dir():
        return None
    detail = load_estimate_detail(doc["source_scan"])
    model = doc.get("wire_tip_model_m")
    current = _wire_tip_doc(detail["passes"][0]["L_m"])
    if model is not None and len(model) == len(current) and np.allclose(model, current, atol=1e-7):
        return None
    if doc.get("reference_set"):
        metadata = load_reference_set()
        if not metadata or metadata["id"] != doc["reference_set"]["id"]:
            raise ValueError("the active correction belongs to another reference set; finish the new captures first")
        return compute_reference_set_correction()
    return compute_correction(doc["source_scan"], anchor_pass=doc.get("anchor_pass", 1))


def check_correction_calib(correction, detail):
    """A correction holds the scanner hand-eye error of its calib: refuse it on an estimate made with another one
    (the hand-eye part would be corrected twice / not at all). Corrections without a calib record are not checked."""
    model = (correction or {}).get("wire_tip_model_m")
    if model is not None and detail.get("passes"):
        current = _wire_tip_doc(detail["passes"][0]["L_m"])
        if len(model) != len(current) or not np.allclose(model, current, atol=1e-7):
            raise ValueError(f"the correction was computed with the wire-tip model {np.round(np.asarray(model) * 1e3, 2).tolist()}"
                             f" mm, now {np.round(np.asarray(current) * 1e3, 2).tolist()} mm (TCP calibration changed): "
                             "press Compute correction on this estimate (the references are kept)")
    a, b = (correction or {}).get("calib"), detail.get("calib")
    if not same_calib(a, b, tol=1e-7):
        raise ValueError(f"the correction was computed with the calib {a.get('file')} (xyz "
                         f"{[round(v * 1e3, 2) for v in a['xyz']]} mm) but this estimate used {b.get('file')} (xyz "
                         f"{[round(v * 1e3, 2) for v in b['xyz']]} mm): press Compute correction on this estimate "
                         "(the references are kept)")


def compute_correction(scan_dir, references=None, *, anchor_pass=None, reference_set=None, persist=True):
    """Correction of this estimate from the measured references (saved to root1_correction.yaml):
    root: wire-tip offsets per end (seam frame); passes 2..: the remaining difference to their measured TCP for every
    relative model (seam frame, per end; an end without its own measurement gets no pass correction).
    Pass 2 anchor: infer the root wire tip from measured P2 minus its known design offset. Other references supply
    root attitude only; their XYZ must not pull the calibrated groove back to an old position. This is setup
    calibration before execution, separate from sequential correction of the remaining welding passes.
    None reuses the saved anchor (legacy corrections use pass 1)."""
    from construct_robot.application.kiro_estimate import check_estimate_length
    if anchor_pass is None:
        anchor_pass = (load_correction() or {}).get("anchor_pass", 1)
    if anchor_pass not in (1, 2):
        raise ValueError("position calibration anchor must be pass 1 or pass 2")
    references = references if references is not None else load_references()
    check_estimate_length(scan_dir)              # a misread end would go into the correction (goal off by its error)
    detail = load_estimate_detail(scan_dir)
    frame, p1 = detail["frame"], detail["passes"][0]
    if p1["k"] != 1:
        raise ValueError("the estimate has no pass 1 (root)")
    matched, notes = match_references(references, detail)
    if reference_set and reference_set.get("baseline_correction") and len(matched) < 8:
        check_correction_calib(reference_set["baseline_correction"], detail)
    if reference_set and anchor_pass != 1:
        raise ValueError("four-pass truth uses P1 as the root anchor")
    missing_root_ends = [end for end in ENDS if (anchor_pass, end) not in matched]
    if missing_root_ends and not reference_set:
        raise ValueError(f"measure both ends of the calibration anchor (pass {anchor_pass} start and goal) first")
    anchor_data = {"anchor_pass": int(anchor_pass)}
    if anchor_pass == 1:
        root_ref = {end: (matched[(1, end)]["position"], matched[(1, end)]["quat"])
                    for end in ENDS if (1, end) in matched}
        if missing_root_ends:
            if not root_ref:
                raise ValueError("capture a correct P1 point first")
            baseline = reference_set.get("baseline_correction")
            if baseline:
                check_correction_calib(baseline, detail)
            delta = np.mean([aim_from_tcp(p, q, p1["L_m"]) - pass_aim(detail, 1, end, baseline)[0]
                             for end, (p, q) in root_ref.items()], axis=0)
            q = next(iter(root_ref.values()))[1]
            for end in missing_root_ends:
                aim = pass_aim(detail, 1, end, baseline)[0] + delta
                root_ref[end] = (tcp_from_aim(aim, q, p1["L_m"]), q)
                notes.append(f"P1 {end}: unmeasured; used the measured end's translation on the previous root model")
    else:
        layout = load_design_layout()
        offset = design_offset(detail, anchor_pass, layout)
        anchor_row = next(r for r in detail["passes"] if r["k"] == anchor_pass)
        root_ref, root_aims, anchor_refs = {}, {}, {}
        for end in ENDS:
            ref = matched[(anchor_pass, end)]
            root_aim = aim_from_tcp(ref["position"], ref["quat"], anchor_row["L_m"]) - frame.to_world(offset)
            q = matched[(1, end)]["quat"] if (1, end) in matched else p1[f"quat_{end}"]
            root_ref[end] = (tcp_from_aim(root_aim, q, p1["L_m"]), q)
            root_aims[end] = root_aim.tolist()
            anchor_refs[end] = dict(_pose_doc(ref["position"], ref["quat"]), source=ref["source"],
                                    saved=ref["saved"], saved_as=ref["label"])
        anchor_data.update(
            anchor_references=anchor_refs,
            anchor_design_m=layout[anchor_pass].tolist(),
            design_layout_m={str(k): v.tolist() for k, v in layout.items()},
            derived_root_aims_world_m=root_aims,
            correction_scope="groove origin from pass-2 design anchor; before execution",
        )
        notes.append("Position anchor P2: root wire tip = measured P2 wire tip - design offset; "
                     "P1 references supply attitude only, other reference positions are ignored. "
                     "Use design_layout for the root-relative bead positions.")
    offsets = reference_offsets(root_ref, p1, frame, p1["L_m"])
    bevel = detail["bevel_sign_w"]
    angles = {end: torch_angles(root_ref[end][1], frame, bevel_sign_w=bevel) for end in ENDS}
    plan_angles = torch_angles(p1["quat_start"], frame, bevel_sign_w=bevel)
    doc = {
        "format_version": 2,
        **anchor_data,
        "source_scan": detail["scan_dir"],
        "calib": detail["calib"],
        "computed": datetime.now().isoformat(timespec="seconds"),
        "compare_at": "wire tip (aim point) = TCP + rotated TCP-frame wire-tip vector (or L * torch axis); "
                      "root position from the selected anchor; pass remainders compared at TCP",
        "wire_tip_model_m": _wire_tip_doc(p1["L_m"]),
        "seam_frame": {"u": frame.u.tolist(), "w": frame.w.tolist(), "n": frame.n.tolist()},
        "plate_normal_world": detail["plate_normal_world"].tolist(),
        "offsets_seam_m": {end: offsets[end]["seam_m"].tolist() for end in ENDS},
        "offsets_world_m": {end: offsets[end]["world_m"].tolist() for end in ENDS},
        "angle_convention": "travel + push / - drag along the plan's travel; work + toward the bevel face (Hongik / AWS A3.0)",
        "reference_torch_angles_deg": {end: {"travel": a[0], "work": a[1]} for end, a in angles.items()},
        "hongik_root_torch_angles_deg": {"travel": plan_angles[0], "work": plan_angles[1]},
        "reference_orientation_xyzw": root_ref[ENDS[0]][1].tolist(),
        "reference_orientations_xyzw": {end: root_ref[end][1].tolist() for end in ENDS},
        "matches": [{"pass": k, "end": end, "saved_as": r["label"], "position_m": r["position"].tolist(),
                     "orientation_xyzw": r["quat"].tolist(), "source": r["source"], "saved": r["saved"]}
                    for (k, end), r in sorted(matched.items())],
        "notes": notes,
        "report": offset_report(offsets),
    }
    if reference_set:
        doc["reference_set"] = {k: v for k, v in reference_set.items() if k != "baseline_correction"}
        doc["reference_set"]["reference_sha256"] = hashlib.sha256(reference_path().read_bytes()).hexdigest()
        doc["inferred_ends"] = {str(k): [end for end in ENDS if (k, end) not in matched] for k in range(1, 5)}
        doc["design_layout_m"] = {str(k): v.tolist() for k, v in load_design_layout().items()}
        if reference_set["relative_model"] == "measured_layout":
            doc["measured_layout_m"] = {str(k): {end: v.tolist() for end, v in ends.items()}
                                        for k, ends in load_pass_layout().items()}
    root_only = dict(doc, offsets_seam_m={end: offsets[end]["seam_m"] for end in ENDS})
    deltas = {}
    for model in RELATIVE_MODELS:
        if model == "measured_layout" and load_pass_layout() is None:
            continue
        deltas[model] = {}
        for (k, end), ref in sorted(matched.items()):
            if k == 1:
                continue
            if anchor_pass != 1 and k != anchor_pass:
                continue                         # older pass positions must not override the new groove origin
            predicted, _aim = pass_target_tcp(scan_dir, k, end, ref["quat"], root_only, model, detail)
            deltas[model].setdefault(str(k), {})[end] = frame.to_local(ref["position"] - predicted).tolist()
        if reference_set:
            previous = ((reference_set.get("baseline_correction") or {}).get("pass_deltas_seam_m") or {}).get(model) or {}
            for k, ends in deltas[model].items():
                if len(ends) != 1:
                    continue
                measured = next(iter(ends))
                other = "goal" if measured == "start" else "start"
                old = previous.get(k) or {}
                if reference_set["relative_model"] == model == "measured_layout":
                    # The learned layout already carries the previous end-to-end geometry.
                    old = {}
                old_measured = np.asarray(old.get(measured, old.get(other, [0, 0, 0])), float)
                old_other = np.asarray(old.get(other, old.get(measured, [0, 0, 0])), float)
                # Keep the current pass shape; translate its unmeasured end by the same new residual.
                ends[other] = (old_other + np.asarray(ends[measured]) - old_measured).tolist()
    doc["pass_deltas_seam_m"] = deltas
    if reference_set and reference_set.get("endpoint_adjustments_world_m"):
        doc["endpoint_adjustments_world_m"] = reference_set["endpoint_adjustments_world_m"]
    if reference_set:
        model = reference_set["relative_model"]
        loaded = dict(doc, offsets_seam_m={end: np.asarray(v) for end, v in doc["offsets_seam_m"].items()})
        checks = []
        for (k, end), ref in sorted(matched.items()):
            predicted, _ = pass_target_tcp(scan_dir, k, end, ref["quat"], loaded, model, detail)
            adjustment = endpoint_adjustment(doc, k, end)
            expected = ref["position"] + adjustment
            error = float(np.linalg.norm(predicted - expected) * 1e3)
            checks.append({"pass": k, "end": end, "error_mm": error,
                           "operator_adjustment_world_m": adjustment.tolist(), "expected_tcp_world_m": expected.tolist(),
                           "reference_tcp_world_m": ref["position"].tolist(), "target_tcp_world_m": predicted.tolist()})
            doc["report"].append(f"P{k} {end}: measured truth vs corrected target {error:.6f} mm ({model})")
        doc["truth_checks"] = checks
    report_model = "measured_layout" if "measured_layout" in deltas else "hongik_tcp"
    for k, ends in sorted(deltas.get(report_model, {}).items()):
        for end, d in ends.items():
            doc["report"].append(f"pass {k} {end}: measured - predicted = along {d[0] * 1e3:+.1f} / "
                                 f"lateral {d[1] * 1e3:+.1f} / normal {d[2] * 1e3:+.1f} mm ({report_model})")
    if persist:
        correction_path().parent.mkdir(parents=True, exist_ok=True)
        atomic_yaml(correction_path(), doc)
    return doc


def load_correction(path=None):
    path = Path(path or correction_path())
    if not path.is_file():
        return None
    doc = yaml.safe_load(path.read_text())
    doc["offsets_seam_m"] = {end: np.asarray(v, float) for end, v in doc["offsets_seam_m"].items()}
    doc["pass_deltas_seam_m"] = {model: {str(k): {end: np.asarray(v, float) for end, v in ends.items()}
                                         for k, ends in (per_pass or {}).items()}
                                 for model, per_pass in (doc.get("pass_deltas_seam_m") or {}).items()}
    return doc


def pass_aim(detail, k, end, correction=None):
    """(Corrected) aim point of pass k at end."""
    row = next(r for r in detail["passes"] if r["k"] == int(k))
    aim = row[f"aim_{end}"]
    if correction is not None:
        check_correction_calib(correction, detail)
        aim = corrected_aim(aim, end, correction["offsets_seam_m"], detail["frame"])
    return aim, row


# How passes 2.. are placed relative to the (corrected) root pass:
#   hongik_tcp  Hongik's TCP displacement TCP_k - TCP_1 (seam frame, per end): reproduces the arrangement of Hongik's
#               TCP lines (their per-pass torch angles are inside it - fill 1 leans 29 deg differently from the root).
#   hongik_aim  Hongik's wire-tip displacement aim_k - aim_1, then TCP = aim - L * z(execution orientation).
#   measured_layout  the operator's measured pass layout (pass_layout.yaml: wire tip of pass k - pass 1 per end, seam
#               frame), TCP = aim - wire tip(execution orientation).
#   design_layout  the operator's design layout of the bead centres (pass_design_layout.yaml, mm from the root wire
#               tip: lateral + toward the 90 deg face, normal + up), TCP = aim - wire tip(execution orientation).
#               Passes with a measured reference are pinned to it by their per-pass remainder; the others (pass 3 -
#               same torch orientation as the root) follow the design exactly.
RELATIVE_MODELS = ("design_layout", "hongik_aim", "measured_layout", "hongik_tcp")      # first = the default

# operator 2026-10-15: P2 (lateral, along, up) = (2, 0, 5) mm from the pass-1 root wire tip.
DEFAULT_DESIGN_LAYOUT_MM = {2: (0.0, 2.0, 5.0), 3: (0.0, -4.0, 6.5), 4: (0.0, 0.0, 10.0)}   # (along, lateral, normal)


def design_layout_path():
    return scan_teaching_dir() / "pass_design_layout.yaml"


def load_design_layout(path=None):
    """{k: (along, lateral, normal) m} - lateral + toward the 90 deg face (the bevel's other side)."""
    path = Path(path or design_layout_path())
    mm = DEFAULT_DESIGN_LAYOUT_MM
    if path.is_file():
        mm = {int(k): tuple(float(x) for x in v) for k, v in (yaml.safe_load(path.read_text())["passes_mm"] or {}).items()}
    return {k: np.asarray(v, float) * 1e-3 for k, v in mm.items()}


def design_offset(detail, k, layout):
    """Signed seam (along, w, up) offset; lateral in the design is + toward the 90 degree face."""
    if int(k) not in layout:
        raise ValueError(f"no design position for pass {k} ({design_layout_path()})")
    along, lateral, normal = layout[int(k)]
    offset = np.asarray([along, lateral, normal], float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError(f"invalid design position for pass {k}")
    toward_face = -1.0 if int(detail.get("bevel_sign_w") or 1) > 0 else 1.0
    return offset * [1.0, toward_face, 1.0]


def pass_layout_path():
    return scan_teaching_dir() / "pass_layout.yaml"


def save_pass_layout(scan_dir, references=None, path=None):
    """Measured pass layout from references of passes 1..4 (both ends, same torch orientation per end): wire tip of
    pass k - pass 1, seam frame of this estimate. Returns {k: {end: (along, lateral, normal) m}}."""
    detail = load_estimate_detail(scan_dir)
    matched, _notes = match_references(load_references() if references is None else references, detail)
    frame, L = detail["frame"], detail["passes"][0]["L_m"]
    from construct_robot.core.scan_reference import aim_from_tcp
    layout = {}
    for k in (2, 3, 4):
        for end in ENDS:
            if (k, end) not in matched or (1, end) not in matched:
                continue
            tip_k = aim_from_tcp(matched[(k, end)]["position"], matched[(k, end)]["quat"], L)
            tip_1 = aim_from_tcp(matched[(1, end)]["position"], matched[(1, end)]["quat"], L)
            layout.setdefault(k, {})[end] = [float(v) for v in frame.to_local(tip_k - tip_1)]
    if not layout:
        raise ValueError("the layout needs measured pass 1 and pass 2-4 references at the same end")
    doc = {"format_version": 1, "frame": "seam (along, lateral, normal) [m]", "source_scan": str(scan_dir),
           "computed": datetime.now().isoformat(timespec="seconds"), "passes": layout}
    out = Path(path or pass_layout_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return layout


def load_pass_layout(path=None):
    path = Path(path or pass_layout_path())
    if not path.is_file():
        return None
    return {int(k): {end: np.asarray(v, float) for end, v in ends.items()}
            for k, ends in (yaml.safe_load(path.read_text())["passes"] or {}).items()}


def endpoint_adjustment(correction, k, end):
    """Explicit operator trim at one endpoint only (World metres); never inherit it from the other end."""
    per_pass = ((correction or {}).get("endpoint_adjustments_world_m") or {}).get(str(int(k))) or {}
    value = np.asarray(per_pass.get(end, [0, 0, 0]), float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid World endpoint adjustment for pass {k} {end}")
    return value


def pass_target_tcp(scan_dir, k, end, quat_xyzw, correction=None, relative="hongik_tcp", detail=None):
    """TCP position for pass k / end, executed with the torch orientation quat_xyzw.
    Root: (corrected) aim - L * z(quat). Passes 2..: the root target + Hongik's displacement (see RELATIVE_MODELS).
    Returns (tcp, aim) - aim = the wire tip the TCP implies with this orientation."""
    if relative not in RELATIVE_MODELS:
        raise ValueError(f"relative must be one of {RELATIVE_MODELS}")
    detail = detail or load_estimate_detail(scan_dir)
    aim1, row1 = pass_aim(detail, 1, end, correction)
    root_tcp = tcp_from_aim(aim1, quat_xyzw, row1["L_m"])
    row = next(r for r in detail["passes"] if r["k"] == int(k))
    if int(k) == 1:
        tcp = root_tcp
    elif relative == "measured_layout":
        snapshot = (correction or {}).get("measured_layout_m")
        layout = ({int(k): {e: np.asarray(v, float) for e, v in ends.items()} for k, ends in snapshot.items()}
                  if snapshot else load_pass_layout())
        if layout is None or int(k) not in layout or end not in layout[int(k)]:
            raise ValueError("no measured pass layout (scan_teaching/pass_layout.yaml)")
        aim = aim1 + detail["frame"].to_world(layout[int(k)][end])
        tcp = tcp_from_aim(aim, quat_xyzw, row["L_m"])
    elif relative == "design_layout":
        snapshot = (correction or {}).get("design_layout_m")
        layout = {int(k): np.asarray(v, float) for k, v in snapshot.items()} if snapshot else load_design_layout()
        aim = aim1 + detail["frame"].to_world(design_offset(detail, k, layout))
        tcp = tcp_from_aim(aim, quat_xyzw, row["L_m"])
    elif relative == "hongik_tcp":
        frame = detail["frame"]
        tcp = root_tcp + frame.to_world(frame.to_local(row[f"tcp_{end}"] - row1[f"tcp_{end}"]))
    else:
        aim, row = pass_aim(detail, k, end, correction)
        tcp = tcp_from_aim(aim, quat_xyzw, row["L_m"])
    if int(k) > 1 and correction is not None:          # measured per-pass remainder (seam frame)
        per_end = ((correction.get("pass_deltas_seam_m") or {}).get(relative) or {}).get(str(int(k))) or {}
        # each end its own measurement; an unmeasured end of a pass takes the other end's (operator, 2026-10-15)
        delta = per_end.get(end, per_end.get(ENDS[1] if end == ENDS[0] else ENDS[0]))
        if delta is not None:
            tcp = tcp + detail["frame"].to_world(delta)
    tcp = tcp + endpoint_adjustment(correction, k, end)
    from construct_robot.core.scan_reference import aim_from_tcp
    return tcp, aim_from_tcp(tcp, quat_xyzw, row["L_m"])


def reference_quat(correction, end):
    """The measured reference torch orientation at this end (older correction files: the start one)."""
    per_end = correction.get("reference_orientations_xyzw") or {}
    return np.asarray(per_end.get(end, correction["reference_orientation_xyzw"]), float)


def targets_table(scan_dir, quat_xyzw=None, correction=None):
    """Per pass / end: Hongik's TCP and the target TCP of every relative model (World m), for the operator.
    quat_xyzw None: the reference orientation of each end (needs the correction)."""
    detail = load_estimate_detail(scan_dir)
    rows = []
    for row in detail["passes"]:
        for end in ENDS:
            quat = quat_xyzw if quat_xyzw is not None else reference_quat(correction, end)
            entry = {"k": row["k"], "role": row["role"], "end": end, "raw_tcp": row[f"tcp_{end}"]}
            for model in RELATIVE_MODELS:
                entry[model] = pass_target_tcp(scan_dir, row["k"], end, quat, correction, model, detail)[0]
            rows.append(entry)
    return rows


def write_corrected_kiro(scan_dir, correction, quat_xyzw=None, relative="hongik_tcp"):
    """kiro_corrected/pass_<k>.yaml: corrected TCP poses with one torch orientation for every pass
    (default: the reference orientation). Returns the written files."""
    detail = load_estimate_detail(scan_dir)
    out = Path(detail["scan_dir"]) / "hongik_estimate" / "kiro_corrected"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for row in detail["passes"]:
        doc = {"schema": SCHEMA, "pass": row["k"], "frame_id": "World", "requires_ik": True}
        for end in ENDS:
            quat = np.asarray(quat_xyzw, float) if quat_xyzw is not None else reference_quat(correction, end)
            tcp, _aim = pass_target_tcp(scan_dir, row["k"], end, quat, correction, relative, detail)
            doc[f"current_{end}"] = _pose_doc(tcp, quat)
        path = out / f"pass_{row['k']}.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        files.append(str(path))
    (out / "README.md").write_text(
        "# Corrected pass endpoints\n\n"
        f"Hongik estimate {detail['scan_dir']} corrected with the measured root-pass reference "
        f"({correction.get('source_scan')}, computed {correction.get('computed')}).\n"
        f"Root: wire-tip aim + per-end seam-frame offset, TCP = aim - L * torch axis. Passes 2..: relative model "
        f"'{relative}'. Orientation: "
        + ("the measured reference orientation of each end" if quat_xyzw is None else f"xyzw {np.round(quat_xyzw, 4).tolist()}")
        + " for every pass.\n\nOffsets:\n"
        + "\n".join(f"- {line}" for line in correction.get("report", [])) + "\n")
    archive_to_history(scan_dir, extra_dirs=(out,), note=f"corrected with relative model {relative}")
    return files


def history_dir(scan_dir):
    return scan_teaching_dir() / "history" / Path(scan_dir).expanduser().name


def archive_to_history(scan_dir, extra_dirs=(), note=""):
    """Accumulate the estimate's small results in construct_description/config/scan_teaching/history/<scan>/:
    the KIRO pass files, the estimate's ends, the correction in use; the raw scan stays in the scan folder."""
    import shutil
    scan_dir = Path(scan_dir).expanduser()
    est_dir = scan_dir / "hongik_estimate"
    dest = history_dir(scan_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for sub in [est_dir / "kiro", *extra_dirs]:
        sub = Path(sub)
        if sub.is_dir():
            target = dest / sub.name
            target.mkdir(exist_ok=True)
            for f in sub.iterdir():
                if f.is_file():
                    shutil.copy2(f, target / f.name)
    summary = {"scan_dir": str(scan_dir), "archived": datetime.now().isoformat(timespec="seconds"), "note": note}
    if (est_dir / "estimate.json").is_file():
        ends = json.loads((est_dir / "estimate.json").read_text()).get("ends") or {}
        summary["ends_m"] = {k: {"point": v.get("point_m"), "sigma": v.get("sigma_m"), "two_sided": v.get("two_sided")}
                             for k, v in ends.items() if v}
    if correction_path().is_file():
        shutil.copy2(correction_path(), dest / correction_path().name)
    if reference_path().is_file():
        shutil.copy2(reference_path(), dest / reference_path().name)
    (dest / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return dest


def parse_pose_label(label):
    """'Pass 2 goal' -> (2, 'goal')."""
    parts = str(label).split()
    if len(parts) != 3 or parts[0] != "Pass" or parts[2] not in ENDS:
        raise ValueError(f"not a pass pose label: {label!r}")
    return int(parts[1]), parts[2]


def estimated_move_target(scan_dir, label, current_pose, correction=None, relative="hongik_tcp", orientations=None):
    """Target TCP Pose for the selected pass end with the pass's taught torch orientation (orientations: {k: {"quat"}}
    of scan_four_pass), else the CURRENT one: TCP = (corrected) aim - L * z(q). Returns (Pose, info) with the aim
    point, info["pass_orientation"] and, with a correction, the angle between the used torch and the reference torch
    (stick-out errors cancel only when they agree)."""
    import copy
    import math
    from construct_robot.core.scan_reference import torch_axis
    k, end = parse_pose_label(label)
    taught = (orientations or {}).get(k)
    quat = (tuple(float(v) for v in taught["quat"]) if taught is not None else
            (current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w))
    tcp, aim = pass_target_tcp(scan_dir, k, end, quat, correction, relative)
    target = copy.deepcopy(current_pose)
    target.position.x, target.position.y, target.position.z = (float(v) for v in tcp)
    if taught is not None:
        target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = quat
    info = {"aim_m": aim, "corrected": correction is not None, "relative": relative if k > 1 else "root",
            "pass_orientation": taught is not None}
    if correction is not None:
        ref_axis = torch_axis(correction["reference_orientation_xyzw"])
        info["angle_to_reference_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, float(torch_axis(quat) @ ref_axis)))))
    return target, info
