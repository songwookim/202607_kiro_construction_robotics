"""hongik_profiler_node — VC nano 3D-Z 프로파일러 → ~/profile (hongik_scan_msgs/ProfileRaw, frame hongik_profiler_optical_frame).

vc3dz_driver (순수 Python TCP 클라이언트) 를 감싼다. **센서는 스트리밍하는 동안에만 잡는다** (센서 TCP 접속은 한 명만 — 2026-10-01):
  ~/start (std_srvs/Trigger)    센서 접속 → ID 44 백업 (세션 첫 init = 원본) → INIT_STANDARD (설정 sensor.init_params) → 스트리밍
                                (NbrLines −1) — 첫 프레임 대기 (first_frame_timeout_s) 후 성공
  ~/stop (std_srvs/Trigger)     스트리밍 정지 (RESET) → 원본 파라미터 복원 → **접속 해제 (센서 놓기)**. 이미 놓은 상태면 그대로 성공.
                                hongik_auto_teach_server 가 목표를 수행하기 전에 이 서비스를 불러 센서를 넘겨받는다.
  ~/backup_params (Trigger)     ID 44 판독 → 파일 (backup_dir) 백업, 메시지에 경로 (스트리밍 중이 아닐 때; 놓은 상태면 잠깐 접속)
  ~/restore_params (Trigger)    이 노드가 잡은 세션의 원본 파라미터 복원 (놓은 상태면 이미 복원됨)
  ~/sensor_owned (std_msgs/Bool, transient local) 이 노드가 지금 센서를 잡고 있는가.
autostart:=true 면 시작하자마자 ~/start 와 같다. 종료 시 ~/stop 과 같다.
"""
from __future__ import annotations

import array
import os
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from hongik_scan_msgs.msg import ProfileRaw
from scan_l0 import config as C
from scan_l0.session import SensorSession

__all__ = ["ProfilerNode", "main", "profile_msg"]


def _f32_seq(v) -> array.array:
    """float32[] 필드용 array('f') — rclpy 는 array('f') 를 검사 없이 받는다 (list 는 원소마다 형·범위 검사: 1408 점 × 2 에
    ~0.7 ms, array 는 ~0.03 ms; 이 함수는 센서 녹화 스레드에서 프레임마다 불린다)."""
    return array.array("f", np.ascontiguousarray(v, dtype=np.float32).tobytes())


def profile_msg(rec, frame_id: str, stamp) -> ProfileRaw:
    m = ProfileRaw()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.x_mm = _f32_seq(rec.x)
    m.z_mm = _f32_seq(rec.z)
    m.n_points = int(rec.n_points)
    m.result_counter = int(rec.result_counter) & 0xFFFFFFFF
    m.global_counter = int(rec.global_counter) & 0xFFFFFFFF
    m.sensor_ts_ms = int(rec.ts_ms)
    m.t_rx_monotonic_ns = int(rec.t_rx_ns)
    m.error = int(rec.error)
    m.checksum_ok = bool(rec.checksum_ok)
    return m


class ProfilerNode(Node):
    def __init__(self):
        super().__init__("hongik_profiler_node")
        P = self.declare_parameter
        cfg_path = P("platform_config", "").value
        settings = C.load_settings(cfg_path) if cfg_path else None
        sens = settings.sensor if settings is not None else {}
        self.sensor_ip = P("sensor_ip", "").value or str(C.unnull(sens.get("ip")) or "")
        self.sensor_port = int(P("sensor_port", 0).value or int(sens.get("port", 1096)))
        self.frame_id = P("frame_id", "hongik_profiler_optical_frame").value
        self.autostart = bool(P("autostart", False).value)
        self.backup_dir = P("backup_dir", os.path.expanduser("~/scan_data/sensor_backup")).value
        self.first_frame_timeout_s = float(P("first_frame_timeout_s", 2.0).value)
        self.apply_init = bool(P("apply_init", True).value)
        if not self.sensor_ip:
            raise RuntimeError("sensor ip is empty (platform_config sensor.ip or parameter sensor_ip)")
        self.pub = self.create_publisher(ProfileRaw, "~/profile", 50)
        self.pub_owned = self.create_publisher(Bool, "~/sensor_owned", QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._lock = threading.Lock()
        self._init_params = dict(sens.get("init_params") or {}) or None
        self.ss: Optional[SensorSession] = None          # 스트리밍하는 동안에만 접속 (센서는 한 명만 잡는다)
        self._publish_owned()
        self.create_service(Trigger, "~/start", self._on_start)
        self.create_service(Trigger, "~/stop", self._on_stop)
        self.create_service(Trigger, "~/backup_params", self._on_backup)
        self.create_service(Trigger, "~/restore_params", self._on_restore)
        self.get_logger().info(f"hongik_profiler_node: sensor {self.sensor_ip}:{self.sensor_port} frame {self.frame_id}")
        if self.autostart:
            self._start()

    def _on_frame(self, rec) -> None:
        age = max(0, time.monotonic_ns() - int(rec.t_rx_ns))
        try:
            self.pub.publish(profile_msg(rec, self.frame_id, (self.get_clock().now() - Duration(nanoseconds=age)).to_msg()))
        except Exception:  # noqa: BLE001
            pass

    def _publish_owned(self) -> None:
        try:
            self.pub_owned.publish(Bool(data=self.ss is not None))
        except Exception:  # noqa: BLE001
            pass

    def _open_locked(self) -> None:
        """센서 접속 + (apply_init) 원본 백업·INIT_STANDARD + 프레임 탭. 실패하면 접속을 닫고 예외."""
        ss = SensorSession(self.sensor_ip, self.sensor_port).open()
        try:
            if self.apply_init:
                mism = ss.init(self._init_params, backup_dir=self.backup_dir)
                for w in ss.warnings:
                    self.get_logger().warn(w)
                if mism:
                    self.get_logger().warn(f"sensor init mismatch (kept): {mism}")
            ss.add_tap(self._on_frame)
        except Exception:
            ss.close()
            raise
        self.ss = ss
        self._publish_owned()

    def _release_locked(self) -> str:
        """스트리밍 정지 → 원본 복원 → 접속 해제. 메시지 반환."""
        if self.ss is None:
            return "sensor already released"
        ss, msg = self.ss, []
        try:
            if ss.recording:
                recs = ss.stop_recording(settle_s=0.1)
                ss.records = []
                msg.append(f"stopped ({len(recs)} frames)")
            else:
                ss.client.stop()
            if ss.backup_taken and ss.restore_result is None:
                ss.restore_result = ss.restore_backup()
                msg.append(f"restored={bool(ss.restore_result.get('restored'))}")
        finally:
            try:
                ss.remove_tap(self._on_frame)
            except Exception:  # noqa: BLE001
                pass
            ss.close()
            self.ss = None
            self._publish_owned()
        msg.append("sensor released")
        return ", ".join(msg)

    def _start(self) -> tuple:
        with self._lock:
            try:
                if self.ss is None:
                    self._open_locked()
            except Exception as e:  # noqa: BLE001
                return False, f"sensor connect/init failed: {type(e).__name__}: {e}"
            if self.ss.recording:
                return True, "already streaming"
            self.ss.start_recording()
            if not self.ss.wait_first_frame(self.first_frame_timeout_s):
                self._release_locked()
                return False, f"no frame within {self.first_frame_timeout_s:.1f} s after start (sensor released)"
            return True, "streaming"

    def _on_start(self, _req, res):
        res.success, res.message = self._start()
        return res

    def _on_stop(self, _req, res):
        with self._lock:
            try:
                res.success, res.message = True, self._release_locked()
            except Exception as e:  # noqa: BLE001
                res.success, res.message = False, f"release failed: {type(e).__name__}: {e}"
        return res

    def _on_backup(self, _req, res):
        with self._lock:
            try:
                if self.ss is not None and self.ss.recording:
                    res.success, res.message = False, "stop streaming first"
                    return res
                os.makedirs(self.backup_dir, exist_ok=True)
                if self.ss is not None:
                    path = self.ss.client.backup_params_to_file(directory=self.backup_dir)
                else:
                    with SensorSession(self.sensor_ip, self.sensor_port) as tmp:     # 잠깐 접속 (판독만, 쓰기 없음)
                        path = tmp.client.backup_params_to_file(directory=self.backup_dir)
                res.success, res.message = True, path
            except Exception as e:  # noqa: BLE001
                res.success, res.message = False, str(e)
        return res

    def _on_restore(self, _req, res):
        with self._lock:
            if self.ss is None:
                res.success, res.message = True, ("sensor not held by this node (released sessions are restored on ~/stop); "
                                                  "to restore from a file use: python -m vc3dz_driver.cli --ip <sensor> restore --file <backup>")
                return res
            rr = self.ss.restore_backup()
            self.ss.restore_result = rr
            res.success = bool(rr.get("restored"))
            res.message = str(rr)
        return res

    def shutdown(self) -> None:
        try:
            with self._lock:
                self._release_locked()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"sensor shutdown failed: {e}")


def main(args=None) -> None:
    from .node_help import print_node_help
    if print_node_help("hongik_profiler_node", args):      # --help (English) before rclpy: no node, no connection
        return
    rclpy.init(args=args)
    node: Optional[ProfilerNode] = None
    try:
        node = ProfilerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
