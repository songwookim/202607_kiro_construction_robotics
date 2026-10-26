"""Record VC nano 3D-Z profiles with the robot pose at each frame and build a World point cloud.

ScannerLiveBridge hands every streamed frame (with its receive-time ROS stamp) to the attached
recorder. TF lookups (World <- optical frame at that stamp) run on this recorder's own worker so
the sensor receive thread never waits on TF. Sensor only: no robot motion.
"""

from datetime import datetime
import json
import os
import queue
import threading
import time

import numpy as np
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField

from construct_robot.nodes.scanner_live import LASER_FRAME, SENSOR_FRAME

SCAN_CLOUD_TOPIC = "/hongik_scanner/scan_cloud"
WORLD_FRAME = "World"


def transform_matrix(transform):
    """geometry_msgs/Transform -> 4x4."""
    t, q = transform.translation, transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = (t.x, t.y, t.z)
    return matrix


def world_points(x_mm, z_mm, valid, T_world_opt):
    """Valid (x, z) mm samples on the laser plane (optical y = 0) -> Nx3 World metres."""
    x = np.asarray(x_mm, dtype=np.float64)[valid] * 1e-3
    z = np.asarray(z_mm, dtype=np.float64)[valid] * 1e-3
    local = np.stack((x, np.zeros_like(x), z, np.ones_like(x)))
    return (T_world_opt @ local)[:3].T


class ScanRecorder:
    # Robot TF can trail the sensor stamps by ~1 s on this cell (2026-10-14: TF 0.8 s stale). Frames are
    # resolved in order, so one wait lets the following frames resolve at once.
    def __init__(self, node, valid_mask, *, out_root=None, tf_timeout_s=2.0,
                 drain_timeout_s=120.0, deferred=False, settle_s=0.3, laser_to_optical=None):
        """deferred: store frames while recording and resolve every pose once at stop() from the buffered TF
        (the TF buffer must keep the whole scan). Online resolving competes with the frame thread for the GIL
        and fell to ~13 frames/s on this PC (2026-10-14), leaving a backlog after the scan.
        laser_to_optical: 4x4 of the calib file in use: poses = TF World <- hongik_laser_link (robot model) @ it, so
        no static optical-frame TF is needed (Hongik's server stack publishes its own calib under that name)."""
        self.laser_to_optical = None if laser_to_optical is None else np.asarray(laser_to_optical, float)
        self.pose_frame = SENSOR_FRAME if self.laser_to_optical is None else LASER_FRAME
        self.deferred = bool(deferred)
        self.settle_s = float(settle_s)
        self._pending = []
        self.node = node
        self.valid_mask = valid_mask
        if out_root is None:
            from construct_robot.io.scan_paths import ensure_scan_results_root
            out_root = ensure_scan_results_root()
        self.out_root = os.path.expanduser(str(out_root))
        self.tf_timeout_s = float(tf_timeout_s)
        self.drain_timeout_s = float(drain_timeout_s)
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread = None
        self._active = False
        self.frames = []           # (stamp_ns, x_mm, z_mm, T_world_opt)
        self.points = []           # World Nx3 per frame
        self.tf_failures = 0
        self.unresolved = 0
        self.last_tf_error = ""
        self.publisher = node.create_publisher(
            PointCloud2, SCAN_CLOUD_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    @property
    def active(self):
        return self._active

    def start(self):
        with self._lock:
            if self._active:
                raise RuntimeError("Scan recording is already active")
            self.frames, self.points, self.tf_failures, self.unresolved = [], [], 0, 0
            self._queue, self._pending = queue.Queue(), []
            self._active = True
            if not self.deferred:
                self._thread = threading.Thread(target=self._work, daemon=True)
                self._thread.start()
        self.publisher.publish(self.cloud(np.empty((0, 3))))

    def add(self, record, stamp_ns):
        """Called by the sensor thread for every frame."""
        if self._active:
            item = (int(stamp_ns), np.array(record.x, np.float32), np.array(record.z, np.float32))
            if self.deferred:
                self._pending.append(item)
            else:
                self._queue.put(item)

    def stop(self):
        """Stop accepting frames, resolve every queued frame (bounded by drain_timeout_s), publish the cloud."""
        with self._lock:
            if not self._active:
                return self.stats()
            self._active = False
            thread = self._thread
        if self.deferred:
            time.sleep(self.settle_s)                # let the TF of the last frames arrive
            for item in self._pending:
                self._resolve(item, wait=False)
            self._pending = []
            self.publisher.publish(self.cloud(self.all_points()))
            return self.stats()
        self._queue.put(None)
        if thread is not None:
            thread.join(timeout=self.drain_timeout_s)
            if thread.is_alive():
                self.unresolved = self._queue.qsize()
        self.publisher.publish(self.cloud(self.all_points()))
        return self.stats()

    def _work(self):
        last_publish = 0
        while True:
            item = self._queue.get()
            if item is None:
                return
            if not self._resolve(item, wait=True):
                continue
            if len(self.frames) - last_publish >= 25:
                last_publish = len(self.frames)
                self.publisher.publish(self.cloud(self.all_points()))

    def _resolve(self, item, *, wait):
        stamp_ns, x, z = item
        try:
            transform = (self._lookup(stamp_ns) if wait else
                         self.node.tf_buffer.lookup_transform(WORLD_FRAME, self.pose_frame, Time(nanoseconds=stamp_ns)))
        except Exception as error:
            self.tf_failures += 1
            self.last_tf_error = f"{type(error).__name__}: {error}"[:300]
            return False
        T = transform_matrix(transform.transform)
        if self.laser_to_optical is not None:
            T = T @ self.laser_to_optical
        valid = self.valid_mask(x, z) if x.size else np.zeros(0, bool)
        self.frames.append((stamp_ns, x, z, T))
        self.points.append(world_points(x, z, valid, T))
        return True

    def _lookup(self, stamp_ns):
        """TF at the frame stamp. Waits only for TF that has not arrived yet; a stamp older than the
        buffered TF can never resolve and fails at once (a blocking wait there stalls the whole queue)."""
        buffer = self.node.tf_buffer
        stamp = Time(nanoseconds=stamp_ns)
        try:
            return buffer.lookup_transform(WORLD_FRAME, self.pose_frame, stamp)
        except Exception:
            latest = buffer.lookup_transform(WORLD_FRAME, self.pose_frame, Time())
            if stamp_ns <= Time.from_msg(latest.header.stamp).nanoseconds:
                raise
            return buffer.lookup_transform(WORLD_FRAME, self.pose_frame, stamp,
                                           timeout=Duration(seconds=self.tf_timeout_s))

    def all_points(self):
        return np.concatenate(self.points) if self.points else np.empty((0, 3))

    def stats(self):
        return {"frames": len(self.frames), "points": int(sum(len(p) for p in self.points)),
                "tf_failures": self.tf_failures, "unresolved_frames": self.unresolved}

    def cloud(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        message = PointCloud2()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = WORLD_FRAME
        message.height, message.width = 1, len(points)
        message.fields = [PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
                          for index, name in enumerate("xyz")]
        message.is_bigendian, message.point_step = False, 12
        message.row_step, message.is_dense = 12 * len(points), True
        message.data = points.tobytes()
        return message

    def save(self, meta):
        """Write <out_root>/<YYYYmmdd-HHMMSS>_scan/{scan.npz, scan.json}; returns the folder."""
        folder = os.path.join(self.out_root, datetime.now().strftime("%Y%m%d-%H%M%S") + "_scan")
        os.makedirs(folder, exist_ok=False)
        width = max((len(f[1]) for f in self.frames), default=0)
        x = np.full((len(self.frames), width), np.nan, np.float32)
        z = np.full_like(x, np.nan)
        for i, (_stamp, fx, fz, _T) in enumerate(self.frames):
            x[i, :len(fx)], z[i, :len(fz)] = fx, fz
        np.savez_compressed(
            os.path.join(folder, "scan.npz"),
            stamp_ns=np.array([f[0] for f in self.frames], np.int64),
            x_mm=x, z_mm=z,
            T_world_optical=np.array([f[3] for f in self.frames]).reshape(-1, 4, 4),
            points_world_m=self.all_points().astype(np.float32),
            points_frame_index=np.concatenate(
                [np.full(len(p), i, np.int32) for i, p in enumerate(self.points)]) if self.points
            else np.empty(0, np.int32),
        )
        document = dict(meta, **self.stats(), world_frame=WORLD_FRAME, sensor_frame=SENSOR_FRAME,
                        pose_source=(f"TF {WORLD_FRAME} <- {SENSOR_FRAME}" if self.laser_to_optical is None else
                                     f"TF {WORLD_FRAME} <- {LASER_FRAME} @ calib laser_link -> optical (local)"),
                        files={"scan.npz": "stamp_ns, x_mm, z_mm (raw sensor columns), T_world_optical, "
                                           "points_world_m (valid samples in World), points_frame_index"})
        with open(os.path.join(folder, "scan.json"), "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False, default=str)
        return folder
