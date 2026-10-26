"""Four-pass ground truth capture: isolate specimen placements and reuse the working corrected geometry."""

import json

import numpy as np
import pytest
import yaml

from construct_robot.application import scan_correction as sc, scan_four_pass as fp, tcp_calibration as tc
from construct_robot.core.scan_reference import aim_from_tcp, tcp_from_aim
from test_scan_four_pass import NAMES, Q13, Q2, Q4, write_four_pass_estimate


@pytest.fixture
def cell(tmp_path, monkeypatch):
    teaching = tmp_path / "teaching"
    monkeypatch.setattr(sc, "scan_teaching_dir", lambda: teaching)
    monkeypatch.setattr(fp, "scan_teaching_dir", lambda: teaching)
    monkeypatch.setattr(tc, "load_offset", lambda path=None: np.array([0.002, -0.019, 0.029]))
    scan = write_four_pass_estimate(tmp_path)
    for k, q in ((1, Q13), (2, Q2), (3, Q13), (4, Q4)):
        fp.save_pass_orientation((k,), q, (NAMES, [0.1 * k] * 6), [0, -1.7, 0.6], "old attitude")
    initial = fp.four_pass_targets(scan, None, fp.load_pass_orientations())
    for k in range(1, 5):
        for end in sc.ENDS:
            # The previous calibration has an end-dependent P4 correction. A one-point update must retain it.
            extra = np.array([0.001, 0, 0.002]) if k == 4 and end == "goal" else np.zeros(3)
            p, q = initial[k][end]
            sc.save_reference(k, end, p + extra, q, "old specimen placement")
    sc.compute_correction(scan, anchor_pass=1)
    baseline = sc.load_correction()
    old_targets = fp.four_pass_targets(scan, baseline, fp.load_pass_orientations())
    return scan, teaching, old_targets


def capture(k, end, p, q):
    sc.save_reference(k, end, p, q, "new truth", joints=(NAMES, [0.1 * k] * 6))


def test_new_set_archives_old_truth_without_changing_the_active_calibration(cell):
    scan, teaching, _ = cell
    original = sc.correction_path().read_bytes()
    original_attitudes = fp.orientation_path().read_bytes()
    metadata = sc.begin_reference_set(scan)
    assert sc.load_references() == []
    assert sc.correction_path().read_bytes() == original
    assert fp.orientation_path().read_bytes() == original_attitudes
    backup = teaching / "history" / (metadata["id"] + "_before_four_pass_references")
    assert len(yaml.safe_load((backup / "root1_reference.yaml").read_text())["entries"]) == 8
    assert (backup / "root1_correction.yaml").read_bytes() == original
    assert sc.load_reference_set()["source_scan"] == str(scan.resolve())
    with pytest.raises(ValueError, match="pending"):
        sc.check_reference_set_applied(sc.load_correction())


def test_four_start_truths_reproduce_each_pass_and_translate_the_current_goal(cell):
    scan, _teaching, old = cell
    original = sc.correction_path().read_bytes()
    sc.begin_reference_set(scan)
    changes = {1: [0.001, 0.002, -0.001], 2: [-0.002, 0, 0.001],
               3: [0.003, 0, -0.002], 4: [0, 0, 0]}
    for k in range(1, 5):
        p, q = old[k]["start"]
        capture(k, "start", p + changes[k], q)
    # Captures are pending data until Compute; they must not move the working calibration.
    assert sc.correction_path().read_bytes() == original
    doc = sc.compute_reference_set_correction()
    assert doc["anchor_pass"] == 1
    assert doc["inferred_ends"] == {str(k): ["goal"] for k in range(1, 5)}
    assert max(r["error_mm"] for r in doc["truth_checks"]) < 1e-8
    correction = sc.load_correction()
    sc.check_reference_set_applied(correction)
    targets = fp.four_pass_targets(scan, correction, fp.load_pass_orientations())
    for k in range(1, 5):
        for end in sc.ENDS:
            np.testing.assert_allclose(targets[k][end][0], old[k][end][0] + changes[k], atol=1e-12)
        assert fp.load_pass_orientations()[k]["captured_for"] == (k,)
    # Editing just P3 must not corrupt the validated P1 / P4 geometry.
    capture(3, "start", old[3]["start"][0] + [0.004, 0, -0.002], old[3]["start"][1])
    with pytest.raises(ValueError, match="changed.*pending"):
        sc.check_reference_set_applied(correction)
    sc.compute_reference_set_correction()
    after = fp.four_pass_targets(scan, sc.load_correction(), fp.load_pass_orientations())
    for k in (1, 4):
        for end in sc.ENDS:
            np.testing.assert_allclose(after[k][end][0], targets[k][end][0], atol=1e-12)


def test_eight_truths_fit_both_ends_and_reach_native_teaching_files(cell):
    scan, teaching, old = cell
    metadata = sc.begin_reference_set(scan)
    measured = {}
    for k in range(1, 5):
        for end in sc.ENDS:
            p, q = old[k][end]
            delta = np.array([k * .0002, .0001, -.0003]) * (1 if end == "start" else -1)
            measured[k, end] = p + delta
            capture(k, end, measured[k, end], q)
    doc = sc.compute_reference_set_correction()
    assert not any(doc["inferred_ends"].values())
    folder, _ = fp.write_four_pass_teaching(scan, sc.load_correction(), out_dir=teaching / "export")
    for k in range(1, 5):
        written = yaml.safe_load((folder / f"pass_{k}.yaml").read_text())
        assert written["generated_from"]["reference_set_id"] == metadata["id"]
        for end, name in (("start", "weld_start"), ("goal", "weld_end")):
            position = written["poses"][name]["tcp_pose_world"]["position_m"]
            np.testing.assert_allclose([position[a] for a in "xyz"], measured[k, end], atol=1e-12)
    # Same calibration on another scan of a shifted specimen follows the scan, never the saved World positions.
    detail = sc.load_estimate_detail(scan)
    for row in detail["passes"]:
        row["aim_start"] += [0.015, 0, 0]
        row["aim_goal"] += [0.015, 0, 0]
    for k in range(1, 5):
        for end in sc.ENDS:
            q = old[k][end][1]
            actual, _ = sc.pass_target_tcp(scan, k, end, q, sc.load_correction(), "design_layout", detail)
            np.testing.assert_allclose(actual, measured[k, end] + [0.015, 0, 0], atol=1e-12)


def test_changed_scan_and_incomplete_captures_never_replace_active_calibration(cell):
    scan, _teaching, old = cell
    sc.begin_reference_set(scan)
    before = sc.correction_path().read_bytes()
    for k in (1, 2, 4):
        capture(k, "start", *old[k]["start"])
    with pytest.raises(ValueError, match="missing P3"):
        sc.compute_reference_set_correction()
    assert sc.correction_path().read_bytes() == before
    estimate = scan / "hongik_estimate/estimate.json"
    doc = json.loads(estimate.read_text())
    estimate.write_text(json.dumps(dict(doc, changed=True)))
    with pytest.raises(ValueError, match="estimate changed"):
        sc.compute_reference_set_correction()
    assert sc.correction_path().read_bytes() == before


def test_failed_calibration_save_keeps_the_previous_working_calibration(cell, monkeypatch):
    scan, _teaching, old = cell
    sc.begin_reference_set(scan)
    for k in range(1, 5):
        capture(k, "start", *old[k]["start"])
    before = sc.correction_path().read_bytes()
    original = sc.atomic_yaml
    def fail(path, doc):
        if path == sc.correction_path():
            raise OSError("disk unavailable")
        return original(path, doc)
    monkeypatch.setattr(sc, "atomic_yaml", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        sc.compute_reference_set_correction()
    assert sc.correction_path().read_bytes() == before
    with pytest.raises(ValueError, match="pending"):
        sc.check_reference_set_applied(sc.load_correction())


def test_learned_relative_positions_keep_welding_attitudes_and_allow_coincident_p3_p4(cell):
    scan, _teaching, old = cell
    before_attitudes = fp.orientation_path().read_bytes()
    sc.begin_reference_set(scan, "measured_layout", baseline_relative="design_layout")
    L = np.array([0.002, -0.019, 0.029])
    truth = {}
    for k in (1, 2, 3):
        p, q = old[k]["start"]
        truth[k] = aim_from_tcp(p, q, L) + [0.002, 0, 0.001 * k]
    truth[4] = truth[3].copy()       # operator explicitly confirmed P3 and P4 at the same point
    for k in range(1, 5):
        # Capture with the same measuring attitude; keep the existing pass-specific welding attitudes.
        capture(k, "start", tcp_from_aim(truth[k], Q13, L), Q13)
    doc = sc.compute_reference_set_correction()
    assert doc["reference_set"]["relative_model"] == "measured_layout"
    assert fp.orientation_path().read_bytes() == before_attitudes
    correction = sc.load_correction()
    targets = fp.four_pass_targets(scan, correction, fp.load_pass_orientations(), "measured_layout")
    for k in range(1, 5):
        for end in sc.ENDS:
            actual_tcp, q = targets[k][end]
            np.testing.assert_allclose(q, old[k][end][1], atol=1e-12)
            previous_tip = aim_from_tcp(old[k][end][0], old[k][end][1], L)
            previous_start = aim_from_tcp(old[k]["start"][0], old[k]["start"][1], L)
            np.testing.assert_allclose(aim_from_tcp(actual_tcp, q, L), previous_tip + truth[k] - previous_start, atol=1e-12)
    layout = sc.load_pass_layout()
    np.testing.assert_allclose(layout[3]["start"], layout[4]["start"], atol=1e-12)
    # The calibration owns a layout snapshot; editing the next nominal layout cannot alter this result.
    sc.pass_layout_path().write_text("passes: {}\n")
    after = fp.four_pass_targets(scan, correction, fp.load_pass_orientations(), "measured_layout")
    np.testing.assert_allclose(after[4]["start"][0], targets[4]["start"][0], atol=1e-12)


def test_start_height_trims_are_persistent_and_never_shift_goals(cell):
    scan, _teaching, old = cell
    sc.begin_reference_set(scan, "measured_layout", baseline_relative="design_layout")
    for k in range(1, 5):
        capture(k, "start", *old[k]["start"])
    sc.compute_reference_set_correction()
    before = fp.four_pass_targets(scan, sc.load_correction(), fp.load_pass_orientations(), "measured_layout")
    refdoc = yaml.safe_load(sc.reference_path().read_text())
    refdoc["reference_set"]["endpoint_adjustments_world_m"] = {
        "2": {"start": [0, 0, -.002]}, "3": {"start": [0, 0, -.002]}}
    sc.atomic_yaml(sc.reference_path(), refdoc)
    sc.compute_reference_set_correction()
    after = fp.four_pass_targets(scan, sc.load_correction(), fp.load_pass_orientations(), "measured_layout")
    for k in range(1, 5):
        for end in sc.ENDS:
            expected = [0, 0, -.002] if k in (2, 3) and end == "start" else [0, 0, 0]
            np.testing.assert_allclose(after[k][end][0] - before[k][end][0], expected, atol=1e-12)
    assert max(c["error_mm"] for c in sc.load_correction()["truth_checks"]) < 1e-8
    sc.compute_reference_set_correction()
    repeated = fp.four_pass_targets(scan, sc.load_correction(), fp.load_pass_orientations(), "measured_layout")
    np.testing.assert_allclose(repeated[2]["start"][0], after[2]["start"][0], atol=1e-12)


@pytest.mark.parametrize("revised", [False, True])
def test_world_trims_preserve_truth_height_trims_and_do_not_accumulate(cell, revised):
    scan, _teaching, old = cell
    sc.begin_reference_set(scan, "measured_layout", baseline_relative="design_layout")
    for k in range(1, 5):
        for end in sc.ENDS:
            capture(k, end, *old[k][end])
    refdoc = yaml.safe_load(sc.reference_path().read_text())
    trims = {str(k): {end: [0, 0, -.002 if k in (2, 3) and end == "start" else 0]
                     for end in sc.ENDS} for k in range(1, 5)}
    refdoc["reference_set"]["endpoint_adjustments_world_m"] = trims
    sc.atomic_yaml(sc.reference_path(), refdoc)
    sc.compute_reference_set_correction()
    before = fp.four_pass_targets(scan, sc.load_correction(), fp.load_pass_orientations(), "measured_layout")
    attitudes = fp.orientation_path().read_bytes()
    for k, ends in trims.items():
        for value in ends.values():
            value[0] = -.001 if revised and int(k) in (2, 4) else -.002
            if revised and int(k) == 4:
                value[2] = -.0015
    sc.atomic_yaml(sc.reference_path(), refdoc)
    for _ in range(2):
        correction = sc.compute_reference_set_correction()
        sc.check_reference_set_applied(correction)
        after = fp.four_pass_targets(scan, correction, fp.load_pass_orientations(), "measured_layout")
        for k in range(1, 5):
            for end in sc.ENDS:
                expected = [-.001 if revised and k in (2, 4) else -.002, 0,
                            -.0015 if revised and k == 4 else 0]
                np.testing.assert_allclose(after[k][end][0] - before[k][end][0], expected, atol=1e-12)
                assert correction["endpoint_adjustments_world_m"][str(k)][end][2] == trims[str(k)][end][2]
        assert yaml.safe_load(sc.reference_path().read_text())["entries"] == refdoc["entries"]
        assert fp.orientation_path().read_bytes() == attitudes


def test_goal_truths_and_replaced_p3_start_reach_the_wire_tip_at_welding_attitudes(cell):
    scan, _teaching, old = cell
    sc.begin_reference_set(scan, "measured_layout", baseline_relative="design_layout")
    L = np.array([0.002, -0.019, 0.029])
    tips = {}
    for k in range(1, 5):
        p, q = old[k]["start"]
        tips[k, "start"] = aim_from_tcp(p, q, L)
        capture(k, "start", tcp_from_aim(tips[k, "start"], Q13, L), Q13)
    sc.compute_reference_set_correction()
    # Later GOAL captures and P3 START replace the old inferred geometry. They
    # were all measured with P4's attitude; actual welding attitudes stay intact.
    before_attitudes = fp.orientation_path().read_bytes()
    for k in range(1, 5):
        p, q = old[k]["goal"]
        tips[k, "goal"] = aim_from_tcp(p, q, L) + [k * .0005, 0, -.001 * k]
        capture(k, "goal", tcp_from_aim(tips[k, "goal"], Q4, L), Q4)
    tips[3, "start"] += [-.002, 0, -.004]
    capture(3, "start", tcp_from_aim(tips[3, "start"], Q4, L), Q4)
    refdoc = yaml.safe_load(sc.reference_path().read_text())
    refdoc["reference_set"]["endpoint_adjustments_world_m"] = {
        "2": {"start": [0, 0, -.002]}, "3": {"start": [0, 0, -.002]}}
    sc.atomic_yaml(sc.reference_path(), refdoc)
    raw_truth = sc.reference_path().read_bytes()
    for _ in range(2):
        sc.compute_reference_set_correction()
        correction = sc.load_correction()
        sc.check_reference_set_applied(correction)
        assert not any(correction["inferred_ends"].values())
        targets = fp.four_pass_targets(scan, correction, fp.load_pass_orientations(), "measured_layout")
        for k in range(1, 5):
            for end in sc.ENDS:
                p, q = targets[k][end]
                trim = np.array([0, 0, -.002]) if k in (2, 3) and end == "start" else np.zeros(3)
                np.testing.assert_allclose(aim_from_tcp(p, q, L), tips[k, end] + trim, atol=1e-12)
                np.testing.assert_allclose(q, old[k][end][1], atol=1e-12)
        assert sc.reference_path().read_bytes() == raw_truth
        assert fp.orientation_path().read_bytes() == before_attitudes
