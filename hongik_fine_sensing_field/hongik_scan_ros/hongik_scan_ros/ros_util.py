"""rclpy 공용 도우미: 전용 노드 + 백그라운드 executor (RosContext), Future 대기, 자세/TF 변환.

어댑터·서버의 블로킹 호출 (MotionBackend.move_linear 등) 은 executor 스레드 밖에서 실행된다. 그래서 Future 는
spin_until_future_complete 가 아니라 add_done_callback + threading.Event 로 기다린다 (executor 는 계속 다른 스레드에서 돈다).

Service calls (call_service; KIRO field 2026-10-14): a request is sent once and waited for up to its timeout - the periodic re-send is
an explicit argument for cheap idempotent queries only (a 5 s planning request re-sent every 3 s queued ~10 copies in move_group,
which serves one request at a time, and /check_state_validity waited behind them). No answer = the pending request is removed and
PlatformUnavailable names the service and the wait. Every wait is stoppable: a stop registered on the RosContext of the client's node
(add_stop_source: the platform's stopped run, the server shutting down) ends it within one SPIN_PERIOD_S with MotionError('stopped
...'), and no request is sent while a stop is in force; clients made exempt (exempt_from_stop: the planning scene's) still complete,
so a stopped run removes its MoveIt objects.
"""
from __future__ import annotations

import math
import threading
import time
import weakref
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from scan_l0.motion_types import MotionError, PlatformUnavailable

NAMESPACE = "/hongik_fine_sensing"
SPIN_PERIOD_S = 0.1          # one wait of an executor thread (RosContext._spin) = the poll period of a stoppable service wait

__all__ = ["RosContext", "wait_future", "call_service", "transform_to_matrix", "matrix_to_pose", "pose_to_matrix",
           "stamp_to_ns", "NAMESPACE", "SPIN_PERIOD_S", "point_msg", "vector_msg"]

_CONTEXTS: "weakref.WeakSet" = weakref.WeakSet()      # live RosContexts: call_service finds the one whose node made the client


class RosContext:
    """전용 rclpy 노드 + MultiThreadedExecutor 백그라운드 스레드. node 를 주면 그 노드를 쓰고 (executor 는 호출자 몫), 아니면
    직접 만든다 (rclpy.init 이 안 돼 있으면 init 도). close() 는 직접 만든 것만 정리한다."""

    def __init__(self, name: str = "hongik_moveit_adapter", *, namespace: str = NAMESPACE, node: Optional[Node] = None,
                 use_sim_time: bool = False, num_threads: int = 6):
        self._owns_init = False
        self._owns_node = node is None
        self._closing = False
        self.n_spin_errors = 0
        self.last_spin_error = ""
        self.executor: Optional[MultiThreadedExecutor] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_sources: list = []
        self._stop_exempt: list = []
        if node is None:
            if not rclpy.ok():
                rclpy.init()
                self._owns_init = True
            node = Node(name, namespace=namespace,
                        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, bool(use_sim_time))])
            self.executor = MultiThreadedExecutor(num_threads=num_threads)
            self.executor.add_node(node)
        self.node = node
        if self.executor is not None:
            self._thread = threading.Thread(target=self._spin, name=f"{name}-executor", daemon=True)
            self._thread.start()
        _CONTEXTS.add(self)

    def add_stop_source(self, fn: Callable[[], Optional[str]]) -> None:
        """fn() → the reason while a stop is in force (a stopped run, the server shutting down), else None. While any source gives a
        reason, every service wait on a client of this node (call_service) ends within SPIN_PERIOD_S and no request is sent:
        MotionError('stopped ...'). Exempt clients (exempt_from_stop) are not stopped."""
        if fn not in self._stop_sources:
            self._stop_sources.append(fn)

    def exempt_from_stop(self, *clients) -> None:
        """Clients a stop does not end - the planning scene's (apply / get): a stopped run still removes its objects from the scene."""
        for c in clients:
            if not any(c is e for e in self._stop_exempt):
                self._stop_exempt.append(c)

    def stop_reason(self, client=None) -> Optional[str]:
        """Why a service wait of ``client`` must end now, or None: the node is closing (every client), else the first stop source
        with a reason (an exempt client: none)."""
        if self._closing:
            return "the ROS node of the platform is closing"
        if client is not None and any(client is e for e in self._stop_exempt):
            return None
        for fn in list(self._stop_sources):
            why = fn()
            if why:
                return str(why)
        return None

    def owns(self, client) -> bool:
        """True when ``client`` was made on this context's node."""
        try:
            return any(c is client for c in self.node.clients)
        except Exception:  # noqa: BLE001 - a destroyed node has no clients
            return False

    def _spin(self) -> None:
        """executor 를 끝까지 돌린다. 개별 콜백·대기 집합 예외 (엔티티 파괴 경쟁 등) 로 스레드가 조용히 죽으면 모든 Future 가
        멈추므로 (서비스 '응답 없음') 예외를 세고 계속 돈다."""
        self.n_spin_errors = 0
        while not self._closing and rclpy.ok(context=self.node.context):
            try:
                self.executor.spin_once(timeout_sec=SPIN_PERIOD_S)
            except Exception as e:  # noqa: BLE001
                if self._closing:
                    break
                self.n_spin_errors += 1
                self.last_spin_error = repr(e)
                time.sleep(0.001)

    def close(self) -> None:
        self._closing = True
        if self._owns_node:
            try:
                if self.executor is not None:
                    self.executor.shutdown(timeout_sec=1.0)
            except Exception:  # noqa: BLE001
                pass
            try:
                self.node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
            if self._thread is not None:
                self._thread.join(2.0)
        if self._owns_init:
            try:
                rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass


def wait_future(fut, timeout_s: float) -> bool:
    """Future 완료까지 (다른 스레드의 executor 가 처리) 최대 timeout_s 대기. 완료면 True."""
    if fut.done():
        return True
    ev = threading.Event()
    fut.add_done_callback(lambda _f: ev.set())
    return ev.wait(max(0.0, float(timeout_s))) or fut.done()


def _context_stop(client) -> Optional[Callable[[], Optional[str]]]:
    """The stop of the RosContext whose node made ``client`` (RosContext.add_stop_source), or None (no such context)."""
    for ctx in list(_CONTEXTS):
        if ctx.owns(client):
            return lambda: ctx.stop_reason(client)
    return None


def _withdraw(client, fut) -> None:
    """A request that will not be waited for any more: the client forgets it (a late response is dropped) and its future ends."""
    try:
        client.remove_pending_request(fut)
    except Exception:  # noqa: BLE001 - already removed / a client without the method
        pass
    try:
        fut.cancel()
    except Exception:  # noqa: BLE001
        pass


def _wait_response(fut, t_until: float, stop) -> Optional[str]:
    """Wait for ``fut`` until monotonic t_until → None (answered or out of time: see fut.done()), or the stop's reason (stop() polled
    every SPIN_PERIOD_S; stop None = no polling)."""
    if fut.done():
        return None
    ev = threading.Event()
    fut.add_done_callback(lambda _f: ev.set())
    while True:
        rem = t_until - time.monotonic()
        if rem <= 0.0 or fut.done():
            return None
        if ev.wait(rem if stop is None else min(rem, SPIN_PERIOD_S)) or fut.done():
            return None
        why = stop() if stop is not None else None
        if why:
            return why


def call_service(client, request, timeout_s: float, wait_service_s: float = 2.0, attempt_timeout_s: float = 3.0, *,
                 resend: bool = True, stop: Optional[Callable[[], Optional[str]]] = None):
    """Service call → the response; None when no server offers the service within wait_service_s (the caller says what that means).

    The request is waited for up to timeout_s. resend=True (cheap idempotent queries - state validity, IK, FK, scene reads and
    writes) sends the same request again every attempt_timeout_s inside timeout_s: the first request of a client DDS has just matched
    can lose its response (the server's response writer has not matched yet; wait_for_service sees the request side only).
    resend=False sends it once - for a request whose service time can exceed attempt_timeout_s (a planning request: move_group runs
    its attempts inside allowed_planning_time and serves one request at a time, so a re-sent copy only queues the same work again).
    No response within timeout_s → the pending request is removed (a late response is dropped) and PlatformUnavailable (a
    MotionError) names the service and the wait. stop() → a reason while a stop is in force (default: the stop of the RosContext
    whose node made the client, RosContext.add_stop_source): checked before every send and every SPIN_PERIOD_S of the wait -
    MotionError('stopped ...') with the request withdrawn."""
    name = str(getattr(client, "srv_name", "") or "service")
    stop = stop if stop is not None else _context_stop(client)
    why = stop() if stop is not None else None
    if why:
        raise MotionError(f"stopped ({why}): {name} not called")
    if not client.wait_for_service(timeout_sec=float(wait_service_s)):
        return None
    t_end = time.monotonic() + float(timeout_s)
    n_sent = 0
    while True:
        why = stop() if stop is not None else None
        if why:
            raise MotionError(f"stopped ({why}): {name} not called")
        fut = client.call_async(request)
        n_sent += 1
        t_until = min(t_end, time.monotonic() + float(attempt_timeout_s)) if resend else t_end
        why = _wait_response(fut, t_until, stop)
        if why:
            _withdraw(client, fut)
            raise MotionError(f"stopped ({why}) while waiting for {name}: request withdrawn")
        if fut.done() and not fut.cancelled():
            return fut.result()
        _withdraw(client, fut)
        if time.monotonic() >= t_end:
            sent = ("sent once" if n_sent == 1 else f"sent {n_sent} times, every {float(attempt_timeout_s):g} s")
            raise PlatformUnavailable(f"{name} did not answer within {float(timeout_s):g} s (request {sent}; the pending request "
                                      "was removed)")


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def transform_to_matrix(tf) -> np.ndarray:
    """geometry_msgs/TransformStamped 또는 Transform → 4x4."""
    t = tf.transform if hasattr(tf, "transform") else tf
    T = np.eye(4)
    q = t.rotation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return T


def matrix_to_pose(T):
    from geometry_msgs.msg import Pose
    T = np.asarray(T, dtype=float)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in q)
    return p


def pose_to_matrix(pose) -> np.ndarray:
    T = np.eye(4)
    q = pose.orientation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


def point_msg(p):
    from geometry_msgs.msg import Point
    m = Point()
    v = [float(x) for x in np.asarray(p, dtype=float).reshape(3)]
    m.x, m.y, m.z = (x if math.isfinite(x) else float("nan") for x in v)
    return m


def vector_msg(v):
    from geometry_msgs.msg import Vector3
    m = Vector3()
    m.x, m.y, m.z = (float(x) for x in np.asarray(v, dtype=float).reshape(3))
    return m


def monotonic_ns() -> int:
    return time.monotonic_ns()


class StampToMonotonic:
    """ROS header.stamp (벽시계 ns) → 호스트 monotonic 시간축 (프로파일 t_rx 와 같은 축).

    수신 시각은 DDS 전달 지연·부하에 따라 몰려서 들어온다 (예: 10 ms 간격으로 발행된 /joint_states 가 0 ms / 20 ms 간격으로 도착).
    그 수신 시각을 샘플 시각으로 쓰면 재구성 시간 동기와 속도 측정이 흔들린다 (30 mm/s 에서 20 ms 몰림 = 0.6 mm). 그래서
        t = stamp + offset,   offset = 창(window_s) 안의 min(t_rx − stamp)   (전달 지연의 하한 = 가장 빨리 도착한 메시지)
    로 발행 시각을 monotonic 축에 옮긴다. 정의상 t ≤ t_rx. 혼자 늦게 도착한 메시지도 발행 시각으로 옮겨진다 (그 자세가 속한 시각).
    스탬프가 없거나 (0) 매핑 시각이 뒤로 가면 (벽시계가 뒤로 조정된 직후 — 창이 새 지연으로 갱신될 때까지) 수신 시각으로 대체하고
    카운트한다 (fallbacks). use_sim_time 플랫폼에서는 쓰지 말 것 (sim 시각은 벽시계와 속도가 다를 수 있다) — 그때는 수신 시각을 쓴다."""

    def __init__(self, window_s: float = 2.0):
        from collections import deque
        self.window_ns = int(window_s * 1e9)
        self._mins = deque()          # (t_rx_ns, d_ns) — d 가 단조 증가하는 단조 덱 (슬라이딩 최소)
        self._last = None
        self.n = 0
        self.fallbacks = 0

    def offset_ns(self):
        return self._mins[0][1] if self._mins else None

    def map(self, stamp_ns: int, t_rx_ns: int) -> int:
        self.n += 1
        if stamp_ns <= 0:
            return self._fallback(t_rx_ns)
        d = int(t_rx_ns) - int(stamp_ns)
        while self._mins and self._mins[-1][1] >= d:
            self._mins.pop()
        self._mins.append((int(t_rx_ns), d))
        while self._mins and self._mins[0][0] < int(t_rx_ns) - self.window_ns:
            self._mins.popleft()
        t = int(stamp_ns) + self._mins[0][1]
        if self._last is not None and t <= self._last:
            return self._fallback(t_rx_ns)
        self._last = t
        return t

    def _fallback(self, t_rx_ns: int) -> int:
        self.fallbacks += 1
        t = int(t_rx_ns)
        if self._last is not None and t <= self._last:
            t = self._last + 1
        self._last = t
        return t
