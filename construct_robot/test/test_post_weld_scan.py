"""Post-weld scan row in the Sequence Builder: the local scan C -> D -> C, files only (no estimate)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from construct_robot.application.sequence_executor import SequenceExecutor
from construct_robot.core.scan_motion import POST_WELD_SCAN, post_weld_scan_step
from construct_robot.core.sequence_model import validate_managed_weld_sequence
from construct_robot.gui.precision_sensing_test_panel import PrecisionSensingTestPanel
from construct_robot.io import scan_paths


def test_step_shape_and_speed_limits():
    step = post_weld_scan_step("/c.yaml", "/d.yaml", tcp_speed_m_s=0.03, velocity_scale=2.0, slot=7)
    assert step["type"] == POST_WELD_SCAN and step["parallel_slot"] == 7 and step["velocity_scale"] == 1.0
    with pytest.raises(ValueError):
        post_weld_scan_step("/c.yaml", "/d.yaml", tcp_speed_m_s=0.0, velocity_scale=0.3, slot=1)


def test_post_weld_scan_never_shares_a_slot_with_arc_on():
    scan = post_weld_scan_step("/c.yaml", "/d.yaml", tcp_speed_m_s=0.03, velocity_scale=0.3, slot=3)
    arc = {"type": "digital_weld", "command": "on", "parallel_slot": 3, "weld_scenario_id": "w"}
    with pytest.raises(ValueError):
        validate_managed_weld_sequence([arc, scan])


def test_executor_hands_the_row_to_the_host():
    host = SimpleNamespace(_run_post_weld_scan=Mock(return_value=(True, "ok")))
    step = post_weld_scan_step("/c.yaml", "/d.yaml", tcp_speed_m_s=0.03, velocity_scale=0.3, slot=1)
    assert SequenceExecutor(host, touch_io_backend="fastech").run_step(step, True) == (True, "ok")
    host._run_post_weld_scan.assert_called_once_with(step, True)


def test_post_weld_scans_have_their_own_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("KIRO_POST_WELD_SCAN_ROOT", str(tmp_path / "pw"))
    assert scan_paths.post_weld_scan_root() == tmp_path / "pw"
    monkeypatch.delenv("KIRO_POST_WELD_SCAN_ROOT")
    assert scan_paths.post_weld_scan_root().name == "post_weld_scan"


def panel(monkeypatch):
    p = object.__new__(PrecisionSensingTestPanel)
    p.busy, p.estimating, p.direct = False, False, None
    p.direct_status = SimpleNamespace(set=Mock())
    p.host = SimpleNamespace(post=lambda fn, *a: fn(*a), log=Mock())
    from construct_robot.gui import precision_sensing_test_panel as module
    monkeypatch.setattr(module, "load_scan_pose", lambda path: path)
    monkeypatch.setattr(module, "check_scan_line", lambda a, b: (0.23, 0.0, 0.0))
    monkeypatch.setattr(module, "export_post_weld_ply", Mock(return_value="/pw/x_scan/scan.ply"))
    return p, module


def test_plan_only_checks_the_poses(monkeypatch):
    p, _module = panel(monkeypatch)
    step = post_weld_scan_step("/c.yaml", "/d.yaml", tcp_speed_m_s=0.03, velocity_scale=0.3, slot=1)
    ok, message = p.run_post_weld_scan(step, False)
    assert ok and "no motion" in message and p.direct is None


def test_execute_records_c_d_c_into_the_post_weld_folder_without_an_estimate(monkeypatch):
    p, module = panel(monkeypatch)
    p._post_weld_sensor = lambda: {"calib_file": "c.yaml", "out_root": "/pw"}
    calls = {}

    class FakeDirect:
        def run(self, **kw):
            calls.update(kw)
            kw["on_finished"]({"status": "OK", "message": "Scan done", "folder": "/pw/x_scan"})
    p.direct = FakeDirect()
    p.start_estimate = Mock()
    step = post_weld_scan_step("/c.yaml", "/d.yaml", tcp_speed_m_s=0.03, velocity_scale=0.3, slot=1)
    ok, message = p.run_post_weld_scan(step, True)
    assert ok and "/pw/x_scan" in message
    assert calls["return_pass"] is True and calls["clearance_m"] == 0.0
    assert calls["sensor"]["out_root"] == "/pw" and calls["extra_meta"]["purpose"] == "post_weld_scan"
    p.start_estimate.assert_not_called()
    module.export_post_weld_ply.assert_called_once_with(p.host, "/pw/x_scan", 0)


def test_ply_failure_reports_failed_step_after_preserving_the_recorded_npz(monkeypatch):
    p, module = panel(monkeypatch)
    p._post_weld_sensor = lambda: {"calib_file": "c.yaml", "out_root": "/pw"}
    p.direct = SimpleNamespace(run=lambda **kw: kw["on_finished"](
        {"status": "OK", "message": "saved", "folder": "/pw/x_scan"}))
    module.export_post_weld_ply.side_effect = OSError("disk full")
    step = post_weld_scan_step("/c.yaml", "/d.yaml", tcp_speed_m_s=.03, velocity_scale=.3, slot=1)
    ok, message = p.run_post_weld_scan(step, True)
    assert not ok and "saved /pw/x_scan" in message and "PLY export failed: disk full" in message


def test_cycle_records_refused_estimate_ends_and_does_not_generate_weld_targets(monkeypatch, tmp_path):
    from construct_robot.io.scan_cycle_log import ScanCycleLog
    import json
    p, module = panel(monkeypatch)
    p.host.log, p.host.error = Mock(), Mock()
    p.host._scan_cycle_journal = ScanCycleLog(tmp_path / "cycles", [], {}, False)
    p._post_weld_sensor = lambda: {"calib_file": "calib.yaml"}
    p.direct = SimpleNamespace(run=lambda **kw: kw["on_finished"](
        {"status": "OK", "folder": str(tmp_path / "scan"), "message": "recorded"}))
    result = {"status": "REFUSED", "message": "PlanContractError: end uncertainty unknown",
              "ends": {"S": {"sigma_m": 0.000279, "two_sided": True},
                       "E": {"sigma_m": None, "two_sided": False}}}
    monkeypatch.setattr(module, "check_scan_coverage", Mock(return_value=(True, "covered")))
    monkeypatch.setattr(module, "run_estimate", Mock(return_value=result))
    writer = Mock()
    monkeypatch.setattr(module.scan_four_pass, "write_four_pass_teaching", writer)
    with pytest.raises(RuntimeError, match=r"unidentified seam end\(s\): E"):
        p.run_cycle_scan({"start_path": "c", "end_path": "d", "tcp_speed_m_s": 0.03, "velocity_scale": 0.3})
    writer.assert_not_called()
    events = [json.loads(line) for line in (p.host._scan_cycle_journal.directory / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "estimate_result"
    assert events[-1]["ends"]["E"]["sigma_m"] is None


def test_coarse_boundary_is_not_reported_as_a_delivered_precision_edge():
    from construct_robot.nodes.hongik_scan_estimate import end_diagnostics
    document = {"ends": {"E": {"two_sided": False, "sigma_m": None}}, "plate": {
        "plate_ends": {"E": {"inner_own_evidence": {"two_sided": True, "sigma_m": 0.000102}}},
        "end_faces": {"readings": [{"end": "S", "plate": "L"}], "plates": {"S:L": {}}}}}
    diag = end_diagnostics(document)
    assert diag["E"]["coarse_boundary_identified"] is True
    assert diag["E"]["precision_edge_readings"] == 0
    assert diag["E"]["precision_plates"] == []
    assert diag["E"]["delivered_identified"] is False
