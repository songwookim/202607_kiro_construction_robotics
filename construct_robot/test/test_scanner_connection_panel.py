"""Independent laser and RViz button routing; no sensor/network I/O."""

import tkinter as tk
from types import SimpleNamespace
from unittest.mock import Mock

from construct_robot.gui.scanner_connection_panel import ScannerConnectionPanel


def panel_state(monkeypatch):
    interp = tk.Tcl()
    panel = object.__new__(ScannerConnectionPanel)
    live = SimpleNamespace(active=False, streaming=False, laser_mode=0,
                           set_laser=Mock(return_value="Laser ON · read-back confirmed"),
                           start=Mock(return_value={}), stop=Mock(return_value="Sensor stopped"), shutdown=Mock())
    panel.gui = SimpleNamespace(node=SimpleNamespace(scanner_live=live, precision_sensing_test=SimpleNamespace(busy=False)),
                                post=lambda callback, *args: callback(*args), log=Mock())
    panel.busy = False
    panel.ip, panel.port = tk.StringVar(interp, "192.168.3.15"), tk.StringVar(interp, "1096")
    panel.gain, panel.calib = tk.StringVar(interp, "0"), tk.StringVar(interp, "")
    panel.sensor_status, panel.laser_status = tk.StringVar(interp), tk.StringVar(interp)
    panel.check_button, panel.live_button = Mock(), Mock()
    panel.laser_on_button, panel.laser_off_button = Mock(), Mock()
    panel.indicator = Mock()
    monkeypatch.setattr("construct_robot.gui.scanner_connection_panel.threading.Thread",
                        lambda target, **kw: SimpleNamespace(start=target))
    return panel


def test_laser_button_does_not_start_visualization(monkeypatch):
    panel = panel_state(monkeypatch)
    panel.set_laser(True)
    panel.live.set_laser.assert_called_once_with("192.168.3.15", 1096, True)
    panel.live.start.assert_not_called()


def test_visualization_buttons_never_call_the_laser_command(monkeypatch):
    panel = panel_state(monkeypatch)
    panel.toggle_visualization()
    panel.live.start.assert_called_once()
    panel.live.set_laser.assert_not_called()
    panel.live.streaming = panel.live.active = True
    panel.toggle_visualization()
    panel.live.stop.assert_called_once_with()
    panel.live.set_laser.assert_not_called()


def test_laser_controls_stay_available_while_visualization_is_streaming(monkeypatch):
    panel = panel_state(monkeypatch)
    panel.live.streaming = panel.live.active = True
    panel._set_busy(False)
    assert panel.laser_on_button.configure.call_args.kwargs["state"] == tk.NORMAL
    assert panel.live_button.configure.call_args.kwargs["text"] == "RViz 시각화 OFF"
    panel.live.laser_mode = 1
    panel._set_busy(False)
    assert panel.laser_off_button.configure.call_args.kwargs["state"] == tk.NORMAL
    assert panel.laser_on_button.configure.call_args.kwargs["state"] == tk.DISABLED
    assert panel.laser_status.get() == "Laser: ON"


def test_emergency_stop_and_gui_close_reach_laser_only_shutdown(monkeypatch):
    panel = panel_state(monkeypatch)
    panel.live.laser_mode = 1
    panel.live.shutdown.return_value = "Laser OFF"
    panel.emergency_stop()
    panel.shutdown()
    assert panel.live.shutdown.call_count == 2
