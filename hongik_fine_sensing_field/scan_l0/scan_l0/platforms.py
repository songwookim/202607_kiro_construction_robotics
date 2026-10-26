"""플랫폼 선택: 플랫폼 YAML 의 pose_source / motion_backend → 어댑터 (ports.select_adapter).

  moveit_jtc (ros_tf + moveit_jtc) : hongik_scan_ros.moveit_adapter.MoveItPlatform — rclpy 노드가 필요하다: ROS 서버
                                     hongik_auto_teach_server 가 만들어 ScanSession(robot=...) 으로 넘긴다 (이 모듈은 열지 않는다).
2026-10-07 (사용자): RB 제어박스 어댑터 (cb_direct) 를 없앴다 — 모든 이동은 MoveIt 을 거친다.
scan_l0 은 ROS 없이 import 된다: MoveIt 어댑터 모듈은 필요할 때만 importlib 로 불러온다.
"""
from __future__ import annotations

import importlib

from .ports import PlatformConfigError, PlatformGates, select_adapter

__all__ = ["adapter_of", "gates_for", "MOVEIT_ADAPTER_MODULE"]

MOVEIT_ADAPTER_MODULE = "hongik_scan_ros.moveit_adapter"


def adapter_of(settings) -> str:
    """Settings → 'moveit_jtc'."""
    params = getattr(settings, "params", settings)
    return select_adapter(dict(params or {}))


def gates_for(settings) -> PlatformGates:
    """세션을 열기 전에 필요한 게이트 훅. MoveIt 어댑터는 hongik_scan_ros 가 있어야 한다."""
    a = adapter_of(settings)
    try:
        mod = importlib.import_module(MOVEIT_ADAPTER_MODULE)
    except ImportError as e:
        raise PlatformConfigError(f"platform adapter {a!r} needs the ROS package hongik_scan_ros ({e})") from e
    return mod.MoveItGates(settings)
