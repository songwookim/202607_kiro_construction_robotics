"""VC nano 3D-Z connection check and live profile view (RViz) on the existing GUI node.

Sensor only: never commands robot motion. The sensor accepts one TCP client, so
live view and Hongik scans (which open the sensor in Hongik's server) exclude
each other. Streaming follows Hongik's vc3dz_driver sequence: back up ID 44 ->
INIT_STANDARD (preserve the current laser mode, mm mode) -> stream -> RESET ->
restore parameters. Laser ON/OFF is a separate, read-back-verified command.
"""

import math
import os
from functools import wraps
import threading
import time

import numpy as np
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# Our own name for the optical frame (laser_link -> optical from the calib chosen in 1. CONNECTION): Hongik's server
# stack publishes its calib as hongik_profiler_optical_frame; two static TFs for one frame -> the last one wins.
SENSOR_FRAME = "kiro_profiler_optical_frame"
LASER_FRAME = "hongik_laser_link"
PROFILE_TOPIC = "/hongik_scanner/profile"
RAW_PROFILE_TOPIC = "/hongik_scanner/raw_profile"
RAW_PROFILE_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("raw_x_mm", "<f4"), ("raw_z_mm", "<f4"),
    ("column_index", "<u4"), ("sensor_valid", "<u4"),
])


def raw_profile_points(x_mm, z_mm, valid_mask):
    """Keep every sensor column and its original mm values for inspection.

    XYZ is in optical-frame metres. Only nonfinite / nonpositive-range sentinels
    get NaN display coordinates. Positive returns outside the calibration range
    remain visible; sensor_valid is the driver's separate range-validity check.
    """
    x, z = np.asarray(x_mm, dtype=np.float32), np.asarray(z_mm, dtype=np.float32)
    if x.ndim != 1 or z.ndim != 1 or x.shape != z.shape:
        raise ValueError("Sensor x/z must be equal-length one-dimensional arrays")
    points = np.zeros(x.size, dtype=RAW_PROFILE_DTYPE)
    points["raw_x_mm"], points["raw_z_mm"] = x, z
    points["column_index"] = np.arange(x.size, dtype=np.uint32)
    points["sensor_valid"] = valid_mask(x, z) if x.size else 0
    drawable = np.isfinite(x) & np.isfinite(z) & (z > 0)
    points["x"], points["z"] = x * 1e-3, z * 1e-3
    for axis in "xyz":
        points[axis][~drawable] = np.nan
    quality = {"columns": x.size, "drawable": int(drawable.sum()),
               "zero_range": int((z == 0).sum()),
               "nonfinite": int((~np.isfinite(x) | ~np.isfinite(z)).sum())}
    return points, quality


def _driver():
    try:
        from vc3dz_driver import params
        from vc3dz_driver.client import VC3DZClient
        from vc3dz_driver.valid import valid_mask
    except ImportError as error:
        raise RuntimeError("vc3dz_driver is unavailable. Build/source Hongik's vc3dz_driver "
                           "package and restart the GUI.") from error
    return params, VC3DZClient, valid_mask


def profile_points(x_mm, z_mm, valid_mask):
    """Valid (x, z) mm samples -> Nx3 float32 metres in the optical frame (y = 0)."""
    x = np.asarray(x_mm, dtype=np.float32)
    z = np.asarray(z_mm, dtype=np.float32)
    mask = valid_mask(x, z) if x.size else np.zeros(0, bool)
    points = np.zeros((int(mask.sum()), 3), np.float32)
    points[:, 0] = x[mask] * 1e-3
    points[:, 2] = z[mask] * 1e-3
    return points, (float(mask.mean()) if mask.size else 0.0)


def calib_transform(path):
    """Hongik EOAT calibration YAML -> (xyz, quaternion xyzw) of laser_link -> optical."""
    with open(path, encoding="utf-8") as stream:
        calib = yaml.safe_load(stream)
    if calib.get("parent_frame") != "laser_link":
        raise ValueError(f"{path} must store laser_link -> optical (got {calib.get('parent_frame')!r})")
    r, p, y = (float(v) / 2.0 for v in calib["rpy"])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    quaternion = (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                  cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
    return tuple(float(v) for v in calib["xyz"]), quaternion


def _sensor_operation(operation):
    """Serialize sensor requests; expose ownership without blocking Tk on a socket wait."""
    @wraps(operation)
    def run(self, *args, **kwargs):
        with self._lock:
            self._operation_depth += 1
            try:
                return operation(self, *args, **kwargs)
            finally:
                self._operation_depth -= 1
    return run


class ScannerLiveBridge:
    def __init__(self, node):
        self.node = node
        self._lock = threading.RLock()
        self._operation_depth = 0
        self._client = None
        self._thread = None
        self._stop = threading.Event()
        self._static_tf = None
        self._laser_mode = None
        self._laser_override = None
        self._laser_commanded = False
        self._endpoint = None
        self.recorder = None        # e.g. ScanRecorder: gets every frame with its receive-time stamp
        self.publisher = node.create_publisher(PointCloud2, PROFILE_TOPIC, 5)
        self.raw_publisher = node.create_publisher(PointCloud2, RAW_PROFILE_TOPIC, 5)

    @property
    def active(self):
        return self._client is not None or self._operation_depth > 0

    @property
    def streaming(self):
        return self._client is not None

    @property
    def laser_mode(self):
        return self._laser_mode

    def _require_sensor_available(self):
        sensing = getattr(self.node, "precision_sensing_test", None)
        if sensing is not None and sensing.busy:
            raise RuntimeError("A Hongik scan owns the sensor; wait for it to finish")

    @_sensor_operation
    def set_laser(self, ip, port, enabled):
        """Set only LaserMode, using the live socket when one is already open."""
        _params, client_type, _valid = _driver()
        with self._lock:
            self._require_sensor_available()
            endpoint = (ip, int(port))
            owned = self._client is not None
            if owned and endpoint != self._endpoint:
                raise RuntimeError("Stop RViz visualization before changing the sensor IP/port")
            client = self._client if owned else client_type(ip, int(port), recv_timeout=5.0).connect()
            target = int(bool(enabled))
            self._endpoint, self._laser_mode, self._laser_commanded = endpoint, None, True
            if owned:
                self._laser_override = target
            try:
                reply = client.set_params({"LaserMode": target}, check=False)
                _text, after = client.read_params()
                self._laser_mode = int(after["LaserMode"])
                if reply.error or self._laser_mode != target:
                    raise RuntimeError(f"Laser command not confirmed: error={reply.error}, LaserMode={self._laser_mode}")
                return f"Laser {'ON' if target else 'OFF'} · read-back confirmed"
            finally:
                if not owned:
                    client.close()

    @_sensor_operation
    def check(self, ip, port):
        """Read-only ID 44 read. Called by a worker; never changes sensor parameters."""
        _params, client_type, _valid = _driver()
        with self._lock:
            self._require_sensor_available()
            streaming = self._client is not None
            if streaming and (ip, int(port)) != self._endpoint:
                raise RuntimeError("Stop RViz visualization before changing the sensor IP/port")
            self._laser_mode = None
            if streaming:
                _text, values = self._client.read_params()
            else:
                with client_type(ip, int(port), recv_timeout=5.0) as client:
                    _text, values = client.read_params()
            self._laser_mode = int(values["LaserMode"])
        return {
            "streaming": streaming,
            "version": values.get("MainVersion", values.get("Mainversion")),
            "laser": values.get("LaserMode"),
            "z_mm": (values.get("MIN_Z_MM"), values.get("MAX_Z_MM")),
            "x_mm": values.get("MAX_X_MM"),
        }

    @_sensor_operation
    def start(self, *, ip, port, calib_file, gain, backup_dir, on_status, on_stopped):
        """Called by a worker. Opens the sensor and starts the frame thread."""
        params, client_type, valid_mask = _driver()
        sensing = getattr(self.node, "precision_sensing_test", None)
        with self._lock:
            if self._client is not None:
                raise RuntimeError("Scanner live view is already running")
            if sensing is not None and sensing.busy:
                raise RuntimeError("A Hongik scan owns the sensor; wait for it to finish")
            if calib_file:
                self._publish_calib_tf(calib_file)
            init = dict(params.INIT_STANDARD)
            if int(gain):
                init["GainVal"] = int(gain)
            client = client_type(ip, int(port), recv_timeout=5.0).connect()
            try:
                _text, before = client.read_params()
                laser_mode = int(before["LaserMode"])
                if laser_mode not in (0, 1):
                    raise RuntimeError(f"Unsupported LaserMode={laser_mode}")
                init["LaserMode"] = laser_mode
                mismatch = client.init_session(init, backup_dir=os.path.expanduser(backup_dir))
                if "LaserMode" in mismatch:
                    raise RuntimeError(f"Laser mode preservation failed: {mismatch['LaserMode']}")
                client.start()
            except Exception:
                self._restore(client, params)
                raise
            self._client = client
            self._endpoint = (ip, int(port))
            self._laser_mode, self._laser_override = laser_mode, None
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._frames, args=(client, valid_mask, on_status, on_stopped), daemon=True)
            self._thread.start()
        return mismatch

    @_sensor_operation
    def stop(self, *, laser_off=False):
        """Stop RViz acquisition; preserve the operator's last laser command."""
        with self._lock:
            client, thread = self._client, self._thread
            self._client = self._thread = None
            self._stop.set()
            laser_mode = 0 if laser_off else self._laser_override
        if client is None:
            return "Scanner live view is not running"
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        params, _client_type, _valid = _driver()
        try:
            report = self._restore(client, params, laser_mode=laser_mode)
        except Exception:
            self._laser_mode = None
            raise
        finally:
            # An OFF visualization must not leave the last cloud visible indefinitely.
            self.publisher.publish(self._cloud(np.empty((0, 3), np.float32)))
            self.raw_publisher.publish(self._cloud(np.empty(0, RAW_PROFILE_DTYPE), raw=True))
        if "NOT fully restored" in report:
            self._laser_mode = None
        elif laser_mode is not None:
            self._laser_mode = laser_mode
        return report

    @_sensor_operation
    def shutdown(self):
        """GUI close / emergency stop: release acquisition and explicitly switch laser off."""
        with self._lock:
            if self._client is not None:
                return self.stop(laser_off=True)
            if self._laser_commanded and self._laser_mode != 0 and self._endpoint is not None:
                return self.set_laser(*self._endpoint, False)
            return "RViz OFF · no GUI-enabled laser to switch off"

    @staticmethod
    def _restore(client, params, *, laser_mode=None):
        try:
            client.stop()
            time.sleep(0.2)
            client.drain()
            if not client.param_backup:
                return "Sensor stopped (no parameter backup to restore)"
            _values, meta = params.parse_params_text_with_meta(client.param_backup_text or "")
            restore = dict(params.writable_params(client.param_backup, meta))
            if laser_mode is not None:
                restore["LaserMode"] = int(laser_mode)
            reply = client.set_params(restore, check=False)
            _text, after = client.read_params()
            mismatch = params.diff_params(restore, after)
            if reply.error or mismatch:
                return (f"Sensor stopped; parameters NOT fully restored (error={reply.error}, "
                        f"mismatch={mismatch}); restore from ~/scan_data/sensor_backup")
            return (f"Sensor stopped; {len(restore)} parameters restored "
                    f"(LaserMode={after.get('LaserMode')})")
        finally:
            client.close()

    def _publish_calib_tf(self, path):
        xyz, quaternion = calib_transform(path)
        transform = TransformStamped()
        transform.header.stamp = self.node.get_clock().now().to_msg()
        transform.header.frame_id = LASER_FRAME
        transform.child_frame_id = SENSOR_FRAME
        t, q = transform.transform.translation, transform.transform.rotation
        t.x, t.y, t.z = xyz
        q.x, q.y, q.z, q.w = quaternion
        if self._static_tf is None:
            self._static_tf = StaticTransformBroadcaster(self.node)
        self._static_tf.sendTransform(transform)

    def _frames(self, client, valid_mask, on_status, on_stopped):
        frames, last_publish, last_report = 0, 0.0, time.monotonic()
        valid, n_points, z_range = 0.0, 0, (math.nan, math.nan)
        reason = "stopped"
        while not self._stop.is_set():
            try:
                record = client.get_frame(timeout=0.5)
            except Exception:
                if not client.connected:
                    reason = f"connection lost: {client.last_error}"
                    break
                continue
            frames += 1
            n_points = int(record.n_points)
            recorder = self.recorder
            if recorder is not None:
                received = self.node.get_clock().now().nanoseconds
                if hasattr(record, "t_rx_ns"):
                    received -= max(0, time.monotonic_ns() - int(record.t_rx_ns))
                recorder.add(record, received)
            try:
                raw_points, quality = raw_profile_points(record.x, record.z, valid_mask)
                points, valid = profile_points(record.x, record.z, valid_mask)
            except ValueError as error:
                self.node.ui.post_latest("scanner_live_status", on_status, f"Malformed profile: {error}")
                continue
            z_range = (math.nan, math.nan)
            if len(points):
                z_range = (float(points[:, 2].min() * 1e3), float(points[:, 2].max() * 1e3))
            now = time.monotonic()
            if now - last_publish >= 1.0 / 30.0:
                last_publish = now
                stamp = self.node.get_clock().now()
                if hasattr(record, "t_rx_ns"):
                    stamp -= Duration(nanoseconds=max(0, time.monotonic_ns() - int(record.t_rx_ns)))
                self.raw_publisher.publish(self._cloud(raw_points, raw=True, stamp=stamp.to_msg()))
                self.publisher.publish(self._cloud(points, stamp=stamp.to_msg()))
            if now - last_report >= 1.0:
                self.node.ui.post_latest(
                    "scanner_live_status", on_status,
                    f"Streaming · {frames / (now - last_report):.0f} fps · raw {quality['drawable']}/{n_points} "
                    f"drawable · valid {100.0 * valid:.1f}% · zero {quality['zero_range']} · "
                    f"nonfinite {quality['nonfinite']} · z {z_range[0]:.1f}–{z_range[1]:.1f} mm · "
                    f"error={getattr(record, 'error', 0)} checksum={getattr(record, 'checksum_ok', True)}" +
                    (" · No measured points: check target / distance / laser signal" if not quality['drawable'] else ""),
                )
                frames, last_report = 0, now
        if reason != "stopped":
            report = self.stop()
            self.node.ui.post(on_stopped, f"Live view ended ({reason}). {report}")

    def _cloud(self, points, *, raw=False, stamp=None):
        message = PointCloud2()
        message.header.stamp = self.node.get_clock().now().to_msg() if stamp is None else stamp
        message.header.frame_id = SENSOR_FRAME
        message.height, message.width = 1, len(points)
        message.fields = [PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
                          for index, name in enumerate("xyz")]
        if raw:
            message.fields.extend([
                PointField(name=name, offset=offset, datatype=datatype, count=1)
                for name, offset, datatype in (("raw_x_mm", 12, PointField.FLOAT32),
                    ("raw_z_mm", 16, PointField.FLOAT32), ("column_index", 20, PointField.UINT32),
                    ("sensor_valid", 24, PointField.UINT32))
            ])
        message.is_bigendian, message.point_step = False, RAW_PROFILE_DTYPE.itemsize if raw else 12
        message.row_step, message.is_dense = message.point_step * len(points), not raw
        message.data = points.tobytes()
        return message
