import threading
from types import SimpleNamespace

from construct_robot.weld_action_gui import WeldActionGui


def fake_gui():
    calls = []
    gui = SimpleNamespace(
        fake_arc_enabled=SimpleNamespace(get=lambda: True),
        weld_motion_done_event=threading.Event(),
        weld_arc_on_done_event=threading.Event(),
        weld_arc_on_success=True,
        weld_motion_success=False,
        sequence_stop_requested=False,
    )

    def off(kind):
        calls.append(kind)
        return True, "FAKE ARC OFF"

    gui._execute_fake_arc = off
    return gui, calls


def test_fake_off_waits_for_motion_and_ignores_real_watcher_geometry():
    gui, calls = fake_gui()
    results = []
    worker = threading.Thread(target=lambda: results.append(
        WeldActionGui._execute_triggered_arc_off(
            gui, {"arc_off_delay_s": "invalid", "path_to_seam_speed_factor": None}
        )
    ))
    worker.start()
    try:
        assert not calls
        gui.weld_motion_success = True
    finally:
        gui.weld_motion_done_event.set()
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert calls == ["off"]
    assert results[0][0] is True


def test_fake_off_does_not_hide_failed_motion():
    gui, calls = fake_gui()
    gui.weld_motion_done_event.set()
    success, _ = WeldActionGui._execute_triggered_arc_off(gui, {})
    assert success is False
    assert calls == ["off"]


def test_fake_off_honors_operator_stop_without_motion_completion():
    gui, calls = fake_gui()
    gui.sequence_stop_requested = True
    success, _ = WeldActionGui._execute_triggered_arc_off(gui, {})
    assert success is False
    assert calls == ["off"]
