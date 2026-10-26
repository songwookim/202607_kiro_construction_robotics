"""1. CONNECTION › Hongik scanner: sensor check, live RViz view, Hongik server status.

Sensor and server checks only; nothing here commands robot motion or welding.
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from construct_robot.io.scan_calib import default_calib_path
from construct_robot.nodes.scanner_live import PROFILE_TOPIC, RAW_PROFILE_TOPIC, SENSOR_FRAME

SENSOR_IP = "192.168.3.15"
SENSOR_PORT = 1096
STATUS_SERVICE = "/hongik_fine_sensing/hongik_auto_teach_server/status"


def _default_calib():
    """KIRO's hand-eye calib (scan_teaching/eoat_kiro_handeye.yaml) when present, else Hongik's eoat_kiro_01.yaml."""
    return default_calib_path()


class ScannerConnectionPanel:
    def __init__(self, gui, parent, indicator):
        self.gui, self.indicator = gui, indicator
        self.busy = False
        self.ip = tk.StringVar(value=SENSOR_IP)
        self.port = tk.StringVar(value=str(SENSOR_PORT))
        self.gain = tk.StringVar(value="0")
        self.calib = tk.StringVar(value=_default_calib())
        self.status_service = tk.StringVar(value=STATUS_SERVICE)
        self.sensor_status = tk.StringVar(value="Not checked")
        self.server_status = tk.StringVar(value="Not checked")
        self.laser_status = tk.StringVar(value="Laser: unknown")
        box = gui._create_fixed_section(parent, "Hongik Scanner · VC nano 3D-Z profiler / fine-sensing server")
        ttk.Label(box, text="IP").grid(row=0, column=0, padx=(6, 2), pady=4, sticky=tk.E)
        ttk.Entry(box, textvariable=self.ip, width=15).grid(row=0, column=1, padx=(2, 8), pady=4)
        ttk.Label(box, text="Port").grid(row=0, column=2, padx=(2, 2), pady=4, sticky=tk.E)
        ttk.Entry(box, textvariable=self.port, width=6).grid(row=0, column=3, padx=(2, 8), pady=4)
        self.check_button = ttk.Button(box, text="Check sensor", command=self.check_sensor)
        self.check_button.grid(row=0, column=4, padx=3, pady=4)
        self.laser_on_button = ttk.Button(box, text="Laser ON", command=lambda: self.set_laser(True))
        self.laser_on_button.grid(row=0, column=5, padx=3, pady=4)
        self.laser_off_button = ttk.Button(box, text="Laser OFF", command=lambda: self.set_laser(False))
        self.laser_off_button.grid(row=0, column=6, padx=3, pady=4)
        self.live_button = ttk.Button(box, text="RViz 시각화 ON", command=self.toggle_visualization)
        self.live_button.grid(row=0, column=7, padx=3, pady=4)
        ttk.Label(box, textvariable=self.sensor_status, wraplength=700).grid(
            row=0, column=8, columnspan=2, padx=(10, 6), pady=4, sticky=tk.W)
        ttk.Label(box, textvariable=self.laser_status).grid(row=1, column=9, padx=6, sticky=tk.W)

        ttk.Label(box, text="Calib YAML").grid(row=1, column=0, padx=(6, 2), pady=4, sticky=tk.E)
        ttk.Entry(box, textvariable=self.calib, width=60).grid(row=1, column=1, columnspan=5, padx=(2, 4), pady=4, sticky=tk.EW)
        ttk.Button(box, text="Browse", command=self.browse_calib).grid(row=1, column=6, padx=3, pady=4)
        ttk.Label(box, text="Gain (0 = Hongik standard)").grid(row=1, column=7, padx=(10, 2), pady=4, sticky=tk.E)
        ttk.Spinbox(box, from_=0, to=480, increment=10, textvariable=self.gain, width=6).grid(
            row=1, column=8, padx=(2, 6), pady=4, sticky=tk.W)

        ttk.Label(box, text="Server status").grid(row=2, column=0, padx=(6, 2), pady=4, sticky=tk.E)
        ttk.Entry(box, textvariable=self.status_service, width=60).grid(
            row=2, column=1, columnspan=5, padx=(2, 4), pady=4, sticky=tk.EW)
        self.server_button = ttk.Button(box, text="Check server", command=self.check_server)
        self.server_button.grid(row=2, column=6, padx=3, pady=4)
        ttk.Label(box, textvariable=self.server_status, wraplength=700).grid(
            row=2, column=7, columnspan=3, padx=(10, 6), pady=4, sticky=tk.W)
        ttk.Label(
            box,
            text=(f"RViz raw: PointCloud2 {RAW_PROFILE_TOPIC} in {SENSOR_FRAME} (all sensor columns; "
                  f"original raw_x_mm/raw_z_mm). Range-valid points: {PROFILE_TOPIC}. Calib YAML publishes "
                  "hongik_laser_link → optical frame (hongik_v4 model; not in kiro_legacy); otherwise set "
                  f"Fixed Frame = {SENSOR_FRAME}. The sensor accepts one client: stop live view before a scan. "
                  "Laser ON/OFF is independent of visualization. RViz ON preserves the laser state; "
                  "RViz OFF restores acquisition settings and keeps the last laser command. "
                  "Backup: ~/scan_data/sensor_backup."),
            foreground="#5f6368", wraplength=1100,
        ).grid(row=3, column=0, columnspan=10, padx=6, pady=(2, 4), sticky=tk.W)

    @property
    def live(self):
        return self.gui.node.scanner_live

    def _run(self, work, done, failed):
        def run():
            try:
                value = work()
            except Exception as error:
                self.gui.post(failed, str(error))
            else:
                self.gui.post(done, value)
        threading.Thread(target=run, daemon=True).start()

    def _set_indicator(self, ok, text):
        self.indicator.configure(text=f"SCANNER: {text}",
                                 bg="#e6f4ea" if ok else "#fce8e6", fg="#137333" if ok else "#b3261e")

    def _set_busy(self, busy):
        self.busy = busy
        streaming = self.live.streaming
        idle = tk.DISABLED if busy else tk.NORMAL
        self.check_button.configure(state=idle)
        self.live_button.configure(state=idle, text="RViz 시각화 OFF" if streaming else "RViz 시각화 ON")
        mode = self.live.laser_mode
        self.laser_on_button.configure(state=tk.DISABLED if busy or mode == 1 else tk.NORMAL)
        self.laser_off_button.configure(state=tk.DISABLED if busy or mode == 0 else tk.NORMAL)
        self.laser_status.set("Laser: ON" if mode == 1 else "Laser: OFF" if mode == 0 else "Laser: unknown")

    def set_laser(self, enabled):
        if self.busy:
            return
        try:
            ip, port = self.ip.get().strip(), int(self.port.get())
        except ValueError:
            self.sensor_status.set("ERROR · Port must be an integer")
            return
        self._set_busy(True)
        self.sensor_status.set(f"Setting laser {'ON' if enabled else 'OFF'}...")
        def done(report):
            self._set_busy(False)
            self.sensor_status.set(report)
            self._set_indicator(True, "LIVE" if self.live.active else "O")
        def failed(text):
            self._set_busy(False)
            self.sensor_status.set(f"ERROR · {text}")
            self._set_indicator(False, "!")
        self._run(lambda: self.live.set_laser(ip, port, enabled), done, failed)

    def toggle_visualization(self):
        if self.busy:
            return
        if self.live.streaming:
            self.stop_live()
        else:
            self.start_live()

    def check_sensor(self):
        if self.busy:
            return
        ip, port = self.ip.get().strip(), self.port.get().strip()
        self._set_busy(True)
        self.sensor_status.set(f"Checking {ip}:{port} (read-only)...")

        def done(info):
            self._set_busy(False)
            laser = "ON" if str(info.get("laser")) == "1" else "OFF"
            self.sensor_status.set(f"Connected · firmware {info.get('version')} · laser {laser} · "
                                   f"z {info['z_mm'][0]}–{info['z_mm'][1]} mm · x ±{info.get('x_mm')} mm")
            self._set_indicator(True, "O")

        def failed(text):
            self._set_busy(False)
            self.sensor_status.set(f"ERROR · {text}")
            self._set_indicator(False, "X")
        self._run(lambda: self.live.check(ip, port), done, failed)

    def start_live(self):
        if self.busy or self.live.active:
            return
        if self.gui.node.precision_sensing_test.busy:
            self.sensor_status.set("ERROR · A Hongik scan owns the sensor; wait for it to finish")
            return
        try:
            port, gain = int(self.port.get()), int(self.gain.get())
        except ValueError:
            self.sensor_status.set("ERROR · Port and gain must be integers")
            return
        args = dict(ip=self.ip.get().strip(), port=port, calib_file=self.calib.get().strip(), gain=gain,
                    backup_dir="~/scan_data/sensor_backup", on_status=self.sensor_status.set,
                    on_stopped=self._stopped)
        self._set_busy(True)
        self.sensor_status.set("Starting RViz acquisition (preserving laser state)...")

        def done(mismatch):
            self._set_busy(False)
            self._set_indicator(True, "LIVE")
            self.sensor_status.set("RViz streaming · " + ("laser ON" if self.live.laser_mode == 1 else "laser OFF; press Laser ON to measure") +
                                   (f" · not applied: {mismatch}" if mismatch else ""))
            self.gui.log(f"Scanner · raw live view started → {RAW_PROFILE_TOPIC}")

        def failed(text):
            self._set_busy(False)
            self.sensor_status.set(f"ERROR · {text}")
            self._set_indicator(False, "X")
        self._run(lambda: self.live.start(**args), done, failed)

    def stop_live(self):
        if self.busy or not self.live.active:
            return
        self._set_busy(True)
        self.sensor_status.set("Stopping and restoring sensor parameters...")
        self._run(self.live.stop, self._stopped, self._stopped)

    def _stopped(self, report):
        self._set_busy(False)
        self.sensor_status.set(report)
        ok = "NOT" not in report and "lost" not in report
        self._set_indicator(ok, "O" if ok else "!")
        self.gui.log(f"Scanner · {report}")

    def emergency_stop(self):
        self._set_busy(True)
        self._run(self.live.shutdown, self._stopped, self._stopped)

    def check_server(self):
        service = self.status_service.get().strip()
        self.server_button.configure(state=tk.DISABLED)
        self.server_status.set(f"Calling {service}...")

        def done(reply):
            self.server_button.configure(state=tk.NORMAL)
            success, _message, fields = reply
            self.server_status.set(
                f"{'OK' if success else 'FAILED'} · ready={fields.get('ready')} running={fields.get('running')} "
                f"base_frame={fields.get('base_frame')} adapter={fields.get('adapter')} "
                f"awaiting_confirm={fields.get('awaiting_confirm')}")

        def failed(text):
            self.server_button.configure(state=tk.NORMAL)
            self.server_status.set(f"ERROR · {text}")
        self._run(lambda: self.gui.node.precision_sensing_test.server_status(service), done, failed)

    def browse_calib(self):
        path = filedialog.askopenfilename(title="Hongik EOAT calibration", filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            self.calib.set(path)

    def shutdown(self):
        """GUI close / emergency stop: stop the laser and restore parameters (blocking, < 2 s)."""
        self.live.shutdown()
