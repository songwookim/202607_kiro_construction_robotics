"""Durable cycle logs on success, scan failure, STOP and save failure; no equipment."""

from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from construct_robot.io.scan_cycle_log import ScanCycleLog, record_cycle_event
from construct_robot.application.sequence_executor import attach_execution_conditions
from test_scan_weld_cycle import cycle
from construct_robot.application.scan_weld_cycle import expand_scan_weld_cycle
from test_sequence_executor import make_host, run


@pytest.fixture(autouse=True)
def log_root(monkeypatch, tmp_path):
    root = tmp_path / "cycle_logs"
    monkeypatch.setenv("KIRO_SCAN_CYCLE_LOG_ROOT", str(root))
    return root


def summary(host):
    return json.loads((host._scan_cycle_journal.directory / "summary.json").read_text())


def prepared_host(fake=False):
    host, executor = make_host(fake_arc=fake)
    host._prepare_scan_weld_cycle = lambda rows, execute: expand_scan_weld_cycle(rows)
    host._run_sequence_step = Mock(return_value=(True, "ok"))
    return host, executor


def test_completed_cycle_has_full_recipe_geometry_timing_and_latest_pointer(log_root):
    rows = cycle()
    host, executor = prepared_host(fake=True)
    attach_execution_conditions(rows, {"mode": "sequence_execute", "operator_setting": "frozen"})
    run(executor, rows)
    doc = summary(host)
    assert doc["status"] == "COMPLETED" and doc["fake_arc"] is True
    assert doc["execution_conditions"]["operator_setting"] == "frozen"
    assert doc["results"][0]["phase"] == "precision"
    assert len(doc["results"]) == len(doc["steps"]) + 1
    assert all(r["success"] and r["duration_s"] >= 0 for r in doc["results"])
    assert doc["not_run_steps"] == []
    assert {r["sequence_number"] for r in doc["results"]} == {1, *(r["sequence_number"] for r in doc["steps"])}
    arc = [r for r in doc["steps"] if r.get("weld_scenario_stage") == "arc_on"]
    assert [r["settings"]["current_a"] for r in arc] == [190, 230, 230, 220]
    weld = [r for r in doc["steps"] if r.get("weld_scenario_stage") == "weld_motion"]
    assert all(r["waypoints"] for r in weld)
    assert [r["weld_tcp_speed_mm_s"] for r in weld] == [5, 7, 7, 5]
    assert (log_root / "latest_scan_cycle.txt").read_text().strip() == str(host._scan_cycle_journal.directory)
    # Separate fake sessions are closed at each pass, retaining per-pass recipes.
    assert sum(c.args[0] == "completed (fake ARC)" for c in host._finish_weld_feedback_record.call_args_list) == 4


def test_per_pass_feedback_conditions_match_its_speed_and_weave():
    expanded = expand_scan_weld_cycle(cycle())
    attach_execution_conditions(expanded, {"weld_weave_enabled": True, "weld_tcp_speed_mm_s": 99})
    arcs = [r for r in expanded if r.get("weld_scenario_stage") == "arc_on"]
    assert [r["execution_conditions"]["weld_tcp_speed_mm_s"] for r in arcs] == [5, 7, 7, 5]
    assert [r["execution_conditions"]["weld_weave_enabled"] for r in arcs] == [False, False, False, True]
    assert [r["execution_conditions"]["scan_cycle_pass"] for r in arcs] == [1, 2, 3, 4]


def test_initial_scan_error_persists_failed_summary_before_any_weld():
    host, executor = prepared_host()
    host._prepare_scan_weld_cycle = Mock(side_effect=RuntimeError("sensor failed"))
    run(executor, cycle())
    doc = summary(host)
    assert doc["status"] == "FAILED" and "sensor failed" in doc["message"]
    assert doc["results"][0]["success"] is False
    host._run_sequence_step.assert_not_called()


def test_step_exception_and_remaining_passes_are_distinguished():
    host, executor = prepared_host()
    host._run_sequence_step = Mock(side_effect=RuntimeError("motion failed"))
    run(executor, cycle())
    doc = summary(host)
    assert doc["status"] == "FAILED"
    assert doc["results"][-1]["pass_number"] == 1
    assert doc["results"][-1]["message"] == "RuntimeError: motion failed"
    assert not any(r.get("pass_number") == 2 for r in doc["results"])
    assert any(r.get("scan_cycle_pass") == 2 for r in doc["not_run_steps"])


def test_stop_is_recorded_as_stopped_instead_of_completed():
    host, executor = prepared_host()
    def stop(step, execute):
        host.sequence_stop_requested = True
        return False, "operator stop"
    host._run_sequence_step = stop
    run(executor, cycle())
    assert summary(host)["status"] == "STOPPED"


def test_no_logging_for_plan_or_ordinary_sequence(log_root):
    host, executor = prepared_host()
    run(executor, cycle(), execute=False)
    assert not log_root.exists()
    run(executor, [{"type": "sleep", "seconds": 0}], execute=True)
    assert not log_root.exists()


def test_log_creation_error_prevents_scanning(monkeypatch):
    from construct_robot.application import sequence_executor as module
    monkeypatch.setattr(module, "ScanCycleLog", Mock(side_effect=OSError("disk full")))
    host, executor = prepared_host()
    host._prepare_scan_weld_cycle = Mock()
    run(executor, cycle())
    host._prepare_scan_weld_cycle.assert_not_called()
    assert host._sequence_finished.call_args.args[0] is False


def test_first_step_log_error_prevents_scanning(monkeypatch):
    original = ScanCycleLog.event
    def event(journal, kind, **data):
        if kind == "step_start":
            raise OSError("disk full before scan")
        return original(journal, kind, **data)
    monkeypatch.setattr(ScanCycleLog, "event", event)
    host, executor = prepared_host()
    host._prepare_scan_weld_cycle = Mock()
    run(executor, cycle())
    host._prepare_scan_weld_cycle.assert_not_called()
    assert summary(host)["status"] == "FAILED"


def test_mid_cycle_log_failure_stops_and_runs_existing_output_cleanup(monkeypatch):
    host, executor = prepared_host()
    def member(step, execute):
        host._scan_cycle_journal.event = Mock(side_effect=OSError("disk full during weld"))
        return True, "completed step"
    host._run_sequence_step = Mock(side_effect=member)
    run(executor, cycle())
    assert host._run_sequence_step.call_count == 1
    host.hicomm_client.inhibit_outputs.assert_called()
    host.hicomm_client.clear_outputs.assert_called()
    assert host._sequence_finished.call_args.args[0] is False


def test_parallel_events_and_late_feedback_artifact_preserve_closed_status(tmp_path):
    journal = ScanCycleLog(tmp_path, [], {}, False)
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda i: journal.event("step_end", sequence_number=i, success=True), range(16)))
    journal.finish("STOPPED", "operator stop")
    journal.event("artifact", kind="weld_feedback", pass_number=2, path="/history/p2.log")
    doc = json.loads((journal.directory / "summary.json").read_text())
    assert doc["status"] == "STOPPED" and len(doc["results"]) == 16
    assert doc["artifacts"][-1]["pass_number"] == 2
    events = [json.loads(line) for line in (journal.directory / "events.jsonl").read_text().splitlines()]
    assert len(events) == 19
    assert len({e["sequence_number"] for e in events if e["event"] == "step_end"}) == 16


def test_partial_execution_keeps_readable_events_and_running_summary(tmp_path):
    journal = ScanCycleLog(tmp_path, [], {}, False)
    journal.event("step_start", pass_number=1)
    assert json.loads((journal.directory / "summary.json").read_text())["status"] == "RUNNING"
    assert json.loads((journal.directory / "events.jsonl").read_text().splitlines()[-1])["event"] == "step_start"


def test_saved_weld_feedback_is_linked_even_after_stop_summary(tmp_path, monkeypatch):
    from test_weld_execution_controller import make_host as feedback_host, settings
    from construct_robot.application import weld_execution_controller as module
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=Mock(wait=Mock(return_value=0))))
    host, _controller, recorder = feedback_host(tmp_path)
    journal = ScanCycleLog(tmp_path / "cycles", [], {}, False)
    host._scan_cycle_journal = journal
    recorder.begin(settings(current_a=230), {"scan_cycle_log_directory": str(journal.directory), "scan_cycle_pass": 2})
    journal.finish("STOPPED", "operator stop")
    path = recorder.finish("operator stop")
    recorder.shutdown()
    doc = json.loads((journal.directory / "summary.json").read_text())
    assert doc["status"] == "STOPPED"
    assert doc["artifacts"] == [dict(kind="weld_feedback", path=str(path), status="operator stop", pass_number=2,
                                     tcp_samples=0, joint_samples=0)]


def test_result_folder_collects_initial_and_four_rescans_with_logs_and_late_feedback(tmp_path):
    import hashlib
    import numpy as np
    def scan(name):
        folder = tmp_path / "raw" / name
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "scan.npz", points_world_m=[[1., 2., 3.]])
        (folder / "scan.json").write_text('{"frames": 1}')
        return folder
    journal = ScanCycleLog(tmp_path / "cycles", [], {}, True)
    provisional = journal.directory
    initial = scan("20261015-180000_scan")
    journal.event("artifact", kind="initial_scan", pass_number=0, path=str(initial))
    assert journal.bundle_directory is None  # no raw data in the teaching history archive
    teaching = tmp_path / "weld_feedback" / f"scan_four_pass_{initial.name}"
    teaching.mkdir(parents=True)
    for k in range(1, 5):
        (teaching / f"pass_{k}.yaml").write_text(f"pass: {k}\n")
    journal.event("artifact", kind="four_pass_teaching", pass_number=0, path=str(teaching))
    assert journal.bundle_directory == teaching
    assert journal.directory == teaching and not provisional.exists()
    for k in range(1, 5):
        raw = scan(f"after_pass_{k}")
        journal.event("artifact", kind="post_weld_scan", pass_number=k, path=str(raw))
        assert (teaching / "npz" / f"pass_{k}_rescan.npz").read_bytes() == (raw / "scan.npz").read_bytes()
    journal.finish("COMPLETED", "complete")
    feedback = tmp_path / "feedback.log"
    feedback.write_text('[tcp_trajectory]\nACTUAL_TF\n[joint_trajectory]\nmeasured joints\n')
    journal.event("artifact", kind="weld_feedback", pass_number=4, path=str(feedback))
    assert (teaching / "pass_4_weld_feedback.log").read_bytes() == feedback.read_bytes()
    assert len(list((teaching / "npz").glob("*.npz"))) == 5
    np.testing.assert_array_equal(np.load(teaching / "npz" / "initial_scan.npz")["points_world_m"], [[1., 2., 3.]])
    for name in ("summary.json", "events.jsonl", "cycle.log"):
        assert (teaching / name).read_bytes() == (journal.directory / name).read_bytes()
    events = [json.loads(line) for line in (teaching / "events.jsonl").read_text().splitlines()]
    assert len(events) == len({json.dumps(e, sort_keys=True) for e in events})
    assert journal._latest_pointer.read_text().strip() == str(teaching)
    manifest = json.loads((teaching / "manifest.json").read_text())
    assert manifest["cycle"] == journal.cycle_id
    assert manifest["status"] == "COMPLETED" and not manifest["unavailable_artifacts"]
    for item in manifest["files"]:
        assert hashlib.sha256((teaching / item["file"]).read_bytes()).hexdigest() == item["sha256"]
    assert (teaching / "pass_4.yaml").read_text() == 'pass: 4\n'
    assert (initial / "scan.npz").exists() and feedback.exists()
    assert not (tmp_path / "npz").exists()


def test_legacy_root_setting_writes_one_result_folder_at_feedback_root(tmp_path, monkeypatch):
    legacy = tmp_path / "weld_feedback" / "scan_four_pass_cycles"
    monkeypatch.setenv("KIRO_SCAN_CYCLE_LOG_ROOT", str(legacy))
    journal = ScanCycleLog(legacy, [], {}, False)
    journal.finish("FAILED", "no scan")
    assert journal.directory.parent == legacy.parent
    assert journal.bundle_directory == journal.directory
    assert not legacy.exists()
    assert (legacy.parent / "latest_scan_cycle.txt").read_text().strip() == str(journal.directory)


def test_promotion_never_replaces_a_different_existing_journal(tmp_path):
    target = tmp_path / "result"
    target.mkdir()
    (target / "summary.json").write_text('another execution')
    journal = ScanCycleLog(tmp_path, [], {}, False)
    previous = journal.directory
    with pytest.raises(ValueError, match="existing cycle log"):
        journal.event("artifact", kind="four_pass_teaching", path=str(target))
    assert journal.directory == previous and (previous / "summary.json").is_file()
    assert (target / "summary.json").read_text() == 'another execution'


def test_partial_promotion_failure_restores_the_original_journal(tmp_path, monkeypatch):
    from pathlib import Path
    target = tmp_path / "result"
    target.mkdir()
    journal = ScanCycleLog(tmp_path, [], {}, False)
    previous = journal.directory
    before = {p.name: p.read_bytes() for p in previous.iterdir()}
    rename = Path.rename
    def fail_second(path, destination):
        if path.parent == previous and path.name == "cycle.log":
            raise OSError("disk failure during promotion")
        return rename(path, destination)
    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="promotion"):
        journal.event("artifact", kind="four_pass_teaching", path=str(target))
    assert journal.directory == previous and journal.bundle_directory is None
    assert {p.name: p.read_bytes() for p in previous.iterdir()} == before
    assert list(target.iterdir()) == []


def test_failed_cycle_still_collects_its_recorded_scan_without_teaching(tmp_path):
    raw = tmp_path / "raw" / "failed_estimate_scan"
    raw.mkdir(parents=True)
    (raw / "scan.npz").write_bytes(b"preserved raw scan")
    journal = ScanCycleLog(tmp_path / "cycles", [], {}, False)
    journal.event("artifact", kind="initial_scan", pass_number=0, path=str(raw))
    journal.finish("FAILED", "estimate refused")
    assert journal.bundle_directory.name == f"scan_four_pass_{raw.name}"
    assert (journal.bundle_directory / "npz" / "initial_scan.npz").read_bytes() == b"preserved raw scan"
    assert json.loads((journal.bundle_directory / "summary.json").read_text())["status"] == "FAILED"


def test_bundle_copy_failure_is_reported_and_source_scan_stays_intact(tmp_path, monkeypatch):
    from construct_robot.io import scan_cycle_log as module
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "scan.npz").write_bytes(b"scan data")
    teaching = tmp_path / "scan_four_pass_test_scan"
    teaching.mkdir()
    journal = ScanCycleLog(tmp_path / "cycles", [], {}, False)
    journal.event("artifact", kind="four_pass_teaching", pass_number=0, path=str(teaching))
    host = SimpleNamespace(_scan_cycle_journal=journal, _scan_cycle_log_error=None,
                           post=lambda f, *args: f(*args), error=Mock())
    monkeypatch.setattr(module.shutil, "copyfile", Mock(side_effect=OSError("disk full")))
    assert not record_cycle_event(host, "artifact", kind="post_weld_scan", pass_number=1, path=str(raw))
    assert "disk full" in host._scan_cycle_log_error
    assert (raw / "scan.npz").read_bytes() == b"scan data"
    assert not list(teaching.rglob("*.tmp"))


def test_feedback_save_error_is_reported_to_cycle_executor(tmp_path, monkeypatch):
    from test_weld_execution_controller import make_host as feedback_host, settings
    from construct_robot.application import weld_execution_controller as module
    monkeypatch.setattr(module, "save_weld_feedback_log", Mock(side_effect=OSError("feedback disk full")))
    host, _controller, recorder = feedback_host(tmp_path)
    journal = ScanCycleLog(tmp_path / "cycles", [], {}, False)
    host._scan_cycle_journal = journal
    recorder.begin(settings(), {"scan_cycle_log_directory": str(journal.directory), "scan_cycle_pass": 1})
    assert recorder.finish("completed") is None
    recorder.shutdown()
    assert "feedback disk full" in host._scan_cycle_log_error
    lines = [json.loads(line) for line in (journal.directory / "events.jsonl").read_text().splitlines()]
    assert lines[-1]["event"] == "feedback_save_failed"


def test_saved_scan_execution_skips_sensor_and_keeps_source_journal(tmp_path):
    from construct_robot.gui.weld_action_gui import WeldActionGui
    source = tmp_path / "previous_scan"
    source.mkdir()
    for k in range(1, 5):
        (source / f"pass_{k}.yaml").write_text(f"pass: {k}\n")
    (source / "summary.json").write_text("original summary")
    (source / "manifest.json").write_text('{"cycle":"previous"}')
    rows = cycle(rescan_after_passes=False, saved_folder=source)
    host, executor = prepared_host(fake=True)
    # No scanner/bridge object exists: this must only use the frozen saved inputs.
    host.precision_sensing_test_panel = object()
    host._prepare_scan_weld_cycle = lambda r, execute: WeldActionGui._prepare_scan_weld_cycle(host, r, execute)
    run(executor, rows)
    doc = summary(host)
    assert doc["status"] == "COMPLETED"
    assert doc["results"][0]["type"] == "saved_scan_load"
    target = host._scan_cycle_journal.directory
    assert target != source
    assert host._scan_cycle_journal.bundle_directory == target
    assert (source / "summary.json").read_text() == "original summary"
    assert all((target / f"pass_{k}.yaml").read_bytes() == (source / f"pass_{k}.yaml").read_bytes()
               for k in range(1, 5))
    assert (target / "manifest.json").is_file()
