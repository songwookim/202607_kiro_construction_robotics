"""Stand-alone scan recorder process (python3 -m construct_robot.nodes.scan_record_process ...).

The GUI process is CPU-bound; inside it the sensor thread and the TF listener lose most frames
(2026-10-14: ~380 of ~1950 frames, TF 0.8 s stale). This process owns the sensor and its own TF
listener for the duration of one scan line. Sensor only: it never commands robot motion.

Protocol on stdout (one line each): READY, then on SIGINT/SIGTERM: RESULT <json>.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from tf2_ros import Buffer, TransformListener

from construct_robot.nodes.scan_recorder import ScanRecorder
from construct_robot.core.scan_calib import calib_matrix
from construct_robot.io.scan_calib import load_calib_pose
from construct_robot.nodes.scanner_live import _driver, LASER_FRAME


def _say(tag, payload=""):
    sys.stdout.write(f"{tag} {payload}".rstrip() + "\n")
    sys.stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--gain", type=int, default=0)
    ap.add_argument("--backup-dir", default="~/scan_data/sensor_backup")
    ap.add_argument("--out-root", default=None, help="default: io.scan_paths.scan_results_root()")
    ap.add_argument("--meta-json", default="{}", help="metadata stored in scan.json")
    a = ap.parse_args(argv)
    import faulthandler
    faulthandler.register(signal.SIGUSR1, all_threads=True)    # kill -USR1 <pid>: dump every thread's stack (debugging)
    params, client_type, valid_mask = _driver()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())

    rclpy.init(args=None, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    node = rclpy.create_node("scan_record_process")
    node.tf_buffer = Buffer(cache_time=Duration(seconds=120))
    TransformListener(node.tf_buffer, node)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    # The calib is applied here (World <- laser_link from the robot TF @ calib), never through a static TF: Hongik's
    # server stack publishes its own calib as laser_link -> hongik_profiler_optical_frame and the last one would win.
    calib = load_calib_pose(a.calib)
    recorder = ScanRecorder(node, valid_mask, out_root=a.out_root, deferred=True,
                            laser_to_optical=calib_matrix(calib["xyz"], calib["rpy"]))
    result = {"status": "ERROR", "message": "", "folder": None}
    client = None
    try:
        # Frames stamped before the first buffered TF can never be resolved: wait for the robot TF first.
        node.tf_buffer.lookup_transform("World", LASER_FRAME, rclpy.time.Time(), timeout=Duration(seconds=10.0))
        init = dict(params.INIT_STANDARD)            # laser ON, mm mode
        if a.gain:
            init["GainVal"] = int(a.gain)
        client = client_type(a.ip, a.port, recv_timeout=5.0).connect()
        mismatch = client.init_session(init, backup_dir=os.path.expanduser(a.backup_dir))
        recorder.start()
        client.start()
        _say("READY", json.dumps({"not_applied": mismatch}, default=str))
        next_report = time.monotonic() + 2.0
        while not stop.is_set():
            if time.monotonic() >= next_report:
                next_report += 2.0
                sys.stderr.write(f"recorder: received {client.stats.get('frames')} stored {len(recorder._pending)}\n")
                sys.stderr.flush()
            try:
                record = client.get_frame(timeout=0.2)
            except Exception:
                if not client.connected:
                    raise RuntimeError(f"sensor connection lost: {client.last_error}")
                continue
            stamp = node.get_clock().now().nanoseconds
            if hasattr(record, "t_rx_ns"):
                stamp -= max(0, time.monotonic_ns() - int(record.t_rx_ns))
            recorder.add(record, stamp)
        result["status"] = "OK"
    except Exception as error:
        result["message"] = str(error)
    finally:
        sensor = {}
        if client is not None:
            sensor = {"driver_stats": dict(client.stats)}
            try:
                client.stop()
                time.sleep(0.2)
                client.drain()
                if client.param_backup:
                    _values, meta = params.parse_params_text_with_meta(client.param_backup_text or "")
                    restore = params.writable_params(client.param_backup, meta)
                    reply = client.set_params(restore, check=False)
                    _text, after = client.read_params()
                    mismatch = params.diff_params(restore, after)
                    sensor["restore"] = ("restored" if not (reply.error or mismatch)
                                         else f"NOT fully restored: error={reply.error} {mismatch}")
            except Exception as error:
                sensor["restore"] = f"failed: {error}"
            finally:
                client.close()
        stats = recorder.stop()
        result.update(stats=stats, sensor=sensor)
        if recorder.frames:
            meta = json.loads(a.meta_json)
            meta.update(status=meta.get("status", result["status"]), sensor_process=sensor, calib=calib)
            result["folder"] = recorder.save(meta)
        _say("RESULT", json.dumps(result, default=str))
        executor.shutdown(timeout_sec=2.0)
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    # The result is printed and the data saved; skip interpreter teardown (rclpy C++ objects can abort there).
    os._exit(0 if result["status"] == "OK" else 1)


if __name__ == "__main__":
    sys.exit(main())


class RecordSession:
    """GUI side: run this module as a child process for one scan line."""

    def __init__(self, sensor, meta, *, ready_timeout_s=20.0, result_timeout_s=180.0):
        self.sensor, self.meta = sensor, meta
        self.ready_timeout_s, self.result_timeout_s = ready_timeout_s, result_timeout_s
        self.process = None
        self._lines = []
        self._ready = threading.Event()
        self._done = threading.Event()

    def start(self):
        import subprocess
        cmd = [sys.executable, "-m", "construct_robot.nodes.scan_record_process",
               "--ip", str(self.sensor["ip"]), "--port", str(int(self.sensor["port"])),
               "--calib", str(self.sensor["calib_file"]), "--gain", str(int(self.sensor.get("gain") or 0)),
               "--backup-dir", str(self.sensor.get("backup_dir") or "~/scan_data/sensor_backup"),
               "--meta-json", json.dumps(self.meta, default=str)]
        if self.sensor.get("out_root"):
            cmd += ["--out-root", str(self.sensor["out_root"])]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        if not self._ready.wait(self.ready_timeout_s):
            self.abort()
            raise RuntimeError("scan recorder did not start: " + " | ".join(self._lines[-5:]))

    def _read(self):
        for line in self.process.stdout:
            line = line.rstrip()
            self._lines.append(line)
            if line.startswith("READY"):
                self._ready.set()
            if line.startswith("RESULT "):
                self._done.set()
        self._ready.set() if self.process.poll() is not None and not self._ready.is_set() else None
        self._done.set()

    def stop(self):
        """Stop recording; returns the child's RESULT dict (sensor restored, data saved)."""
        if self.process is None:
            return {"status": "ERROR", "message": "not started"}
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
        self._done.wait(self.result_timeout_s)
        result = next((json.loads(line[7:]) for line in reversed(self._lines) if line.startswith("RESULT ")), None)
        if result is None:
            self.abort()
            return {"status": "ERROR", "message": "scan recorder gave no result: " + " | ".join(self._lines[-5:])}
        return result

    def abort(self):
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10.0)
            except Exception:
                self.process.kill()
