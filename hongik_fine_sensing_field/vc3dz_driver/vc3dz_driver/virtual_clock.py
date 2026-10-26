"""시험 전용 가상 시계 (락스텝 이산 사건) — 생산 모듈은 이 파일을 import 하지 않는다 (린트 scan_l0/tests/test_clock_seams.py).

뜻 (명세 docs §2; 2026-10-09 사용자: "시뮬레이션에선 결코 실제 이동속도로 할 필요가 없다"):

* 시각 ``t_ns`` 는 정수 ns (시작 10**12, 실행마다 같다). 계산은 시뮬 시간을 0 s 쓴다 — 이동·촬영 격자·지연·타임아웃은 시뮬 초로 정확히 흐른다.
  배속 상수는 없다.
* 참여자 = 시계를 따르는 스레드 (``register`` (start 전에), ``acting()``, ``Link.receiving()``, install 한 스레드). 상태는 RUNNING (계산 중이거나
  시계가 아닌 대기 — 소켓·파일·``fut.result`` — 중) 또는 PARKED (시계 대기 안).
* 평가할 때마다 위에서부터:
    1. RUNNING 참여자가 있으면 멈춘다.
    2. 닫히지 않은 소켓 링크의 어느 방향이든 보냈지만 아직 받지 않은 메시지가 있으면 멈춘다.
    3. 술어가 참인 PARKED 대기자가 있으면 순번이 가장 작은 **하나**만 깨운다 (시간 그대로).
    4. 아니면 (마감, 순번) 이 가장 작은 **하나**를 깨운다 — ``t_ns = max(t_ns, 마감)``.
    5. 마감도 참인 술어도 없으면 교착: 벽시계로 잠깐 (non-participant 의 변화) 기다린 뒤 스택을 덤프하고 모든 대기자에게 SimDeadlock.
  깨운 스레드가 다시 대기하거나 끝날 때까지 다음을 깨우지 않으므로 참여자 실행은 완전히 직렬이고 결정적이다. 같은 시각은 먼저 등록된 쪽이 먼저.
* 술어는 시계 잠금 안에서 평가된다 — 잠금을 잡지 않는 읽기만: ``Event.is_set()``, ``bool(q.queue)``, 스레드 끝남, 링크 계수.
* 엄격 모드 (기본): 등록되지 않은 스레드의 sleep/wait/get/join 은 ClockMisuse. monotonic 읽기는 누구나.
* 감시 (진단 전용): 벽시계로 watchdog_s 동안 아무것도 깨우지 못하면 이유 (RUNNING 참여자의 스택, 또는 미수신 링크) 를 한 번 기록한다.
* shutdown: PARKED 대기자를 타임아웃으로 풀고, 그 뒤의 대기 호출은 실시계 위임 (새어 나간 데몬 스레드가 멈추거나 바쁘게 돌지 않게).
  살아 있는 참여자 목록을 돌려준다.
* fork 자식은 주변 시계가 REAL (이 모듈을 import 할 때 os.register_at_fork 로 건다).
"""
from __future__ import annotations

import ipaddress
import itertools
import os
import queue
import sys
import threading
import time
import traceback
import weakref
from contextlib import contextmanager
from typing import Callable, Optional

from . import clock as _clock_mod
from .clock import Clock

__all__ = ["VirtualClock", "ClockMisuse", "SimDeadlock", "T0_NS"]

T0_NS = 10 ** 12

_RUNNING = 0
_PARKED = 1


class ClockMisuse(RuntimeError):
    """등록되지 않은 스레드가 가상 시계로 기다리려 했다 (엄격 모드)."""


class SimDeadlock(RuntimeError):
    """모든 참여자가 마감 없는 대기에 있고 어떤 술어도 참이 아니다."""


os.register_at_fork(after_in_child=_clock_mod._reset_to_real_in_child)


class _Part:
    __slots__ = ("th", "name", "state", "deadline", "pred", "seq", "woken", "why", "cond")

    def __init__(self, th: threading.Thread, seq: int, lock):
        self.th = th
        self.name = th.name
        self.state = _RUNNING
        self.deadline: Optional[int] = None
        self.pred: Optional[Callable[[], bool]] = None
        self.seq = seq
        self.woken = False
        self.why: Optional[str] = None
        self.cond = threading.Condition(lock)


class _LinkState:
    __slots__ = ("key", "pending", "closed", "n_attached")

    def __init__(self, key):
        self.key = key
        self.pending: dict = {}
        self.closed = False
        self.n_attached = 0

    def blocking(self) -> bool:
        if self.closed:
            return False
        for v in self.pending.values():
            if v > 0:
                return True
        return False


class _VLink:
    """가상 시계의 소켓 링크 끝 하나 (attach 한 소켓의 (로컬, 상대) 끝점 쌍으로 상대 끝과 계수기를 나눈다)."""

    __slots__ = ("_c", "_st", "_out", "_in", "host", "port")

    def __init__(self, c: "VirtualClock", host, port):
        self._c = c
        self._st: Optional[_LinkState] = None
        self._out = None
        self._in = None
        self.host, self.port = host, port

    def attach(self, sock) -> None:
        me = tuple(sock.getsockname()[:2])
        peer = tuple(sock.getpeername()[:2])
        self._c._link_attach(self, me, peer)

    def sent(self) -> None:
        self._c._link_sent(self)

    def received(self) -> None:
        self._c._link_received(self)

    def receiving(self):
        return self._c._receiving(self)

    def close(self) -> None:
        self._c._link_close(self)


def _is_loopback(host) -> bool:
    h = str(host).strip().lower()
    if h in ("localhost", "localhost.localdomain", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(h.strip("[]")).is_loopback
    except ValueError:
        return False                      # 이름은 풀지 않는다 (DNS 없이 거부)


class VirtualClock(Clock):
    """락스텝 이산 사건 가상 시계 (모듈 문서)."""

    virtual = True

    def __init__(self, *, strict: bool = True, watchdog_s: float = 30.0, deadlock_grace_s: float = 2.0, trace: bool = False,
                 poll_wall_s: float = 0.5):
        self._lock = threading.Lock()
        self._t = T0_NS
        self._parts: dict = {}
        self._seq = itertools.count()
        self._links: dict = {}
        self._finished = weakref.WeakSet()
        self._shut = False
        self.strict = bool(strict)
        self.watchdog_s = float(watchdog_s)
        self.deadlock_grace_s = float(deadlock_grace_s)
        self.poll_wall_s = float(poll_wall_s)
        self.misuse: list = []
        self.deadlocks: list = []
        self.watchdog: list = []             # (kind, text): kind 'running' (긴 계산·시계 밖 대기 — 정보) | 'link' | 'parked'
        self.n_advances = 0
        self.n_wakes = 0
        self.trace: Optional[list] = [] if trace else None    # (t_ns, seq, why) — 결정성 시험용
        self._deadlock_since: Optional[float] = None
        self._last_progress_wall = time.monotonic()
        self._watch_reported = False
        self._watch_stop = threading.Event()
        self._watch_thr: Optional[threading.Thread] = None
        self.shutdown_report: Optional[dict] = None

    def __repr__(self) -> str:
        return f"VirtualClock(t={self._t * 1e-9:.6f} s, participants={len(self._parts)})"

    # ------------------------------------------------------------------ 시각
    def monotonic_ns(self) -> int:
        return self._t

    def monotonic(self) -> float:
        return self._t * 1e-9

    @property
    def shut(self) -> bool:
        return self._shut

    # ------------------------------------------------------------------ 대기
    def sleep(self, s: float) -> None:
        d = self._t + int(round(max(float(s), 0.0) * 1e9))
        if self._park(None, d, "sleep") is None:
            time.sleep(max(float(s), 0.0))

    def wait(self, event, timeout=None) -> bool:
        if event.is_set():
            return True
        d = None if timeout is None else self._t + int(round(max(float(timeout), 0.0) * 1e9))
        why = self._park(event.is_set, d, "wait")
        if why is None:
            return event.wait(timeout)
        return event.is_set()

    def get(self, q, timeout=None):
        if timeout is not None and timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        d = None if timeout is None else self._t + int(round(float(timeout) * 1e9))
        while True:
            try:
                return q.get_nowait()
            except queue.Empty:
                pass
            if d is not None and d <= self._t:
                raise queue.Empty
            why = self._park(lambda: bool(q.queue), d, "get")
            if why is None:                                  # 종료 뒤 (또는 비엄격 비참여) — 실시계 위임
                return q.get(timeout=None if d is None else max(0.0, (d - self._t) * 1e-9))
            if why == "shutdown":
                try:
                    return q.get_nowait()
                except queue.Empty:
                    raise queue.Empty from None

    def join(self, th, timeout=None) -> None:
        if th is threading.current_thread():
            raise RuntimeError("cannot join current thread")
        if self._done(th):
            th.join(timeout)
            return
        d = None if timeout is None else self._t + int(round(max(float(timeout), 0.0) * 1e9))
        why = self._park(lambda: self._done(th), d, "join")
        if why is None:
            th.join(timeout)
            return
        if self._done(th):
            th.join(1.0)                                     # run 은 끝났다 — 부트스트랩 정리만 (시간 정지 중, 짧다)

    def _done(self, th) -> bool:
        return th in self._finished or (th.ident is not None and not th.is_alive())

    # ------------------------------------------------------------------ 참여자
    def register(self, th: threading.Thread) -> threading.Thread:
        with self._lock:
            if self._shut or th in self._parts:
                return th
            self._parts[th] = _Part(th, next(self._seq), self._lock)
        orig = th.run

        def _run_registered():
            try:
                orig()
            finally:
                self._unregister(th, finished=True)
        th.run = _run_registered
        return th

    def _unregister(self, th, finished: bool) -> None:
        with self._lock:
            self._parts.pop(th, None)
            if finished:
                self._finished.add(th)
            self._evaluate()

    @contextmanager
    def _acting_cm(self):
        th = threading.current_thread()
        with self._lock:
            mine = not self._shut and th not in self._parts
            if mine:
                self._parts[th] = _Part(th, next(self._seq), self._lock)
        try:
            yield
        finally:
            if mine:
                self._unregister(th, finished=False)

    def acting(self):
        return self._acting_cm()

    def _install_enter(self, th: threading.Thread) -> None:
        with self._lock:
            if th not in self._parts:
                self._parts[th] = _Part(th, next(self._seq), self._lock)
        self._watch_thr = threading.Thread(target=self._watch_loop, name="virtual-clock-watchdog", daemon=True)
        self._watch_thr.start()

    def call_later(self, dt: float, fn: Callable[[], None]) -> threading.Thread:
        """시험 도우미: 시뮬 dt 초 뒤 fn() 을 부르는 등록된 타이머 스레드."""
        if self._shut:
            return _clock_mod.REAL.call_later(dt, fn)

        def body():
            self.sleep(dt)
            fn()
        th = threading.Thread(target=body, name="sim-call-later", daemon=True)
        self.register(th).start()
        return th

    # ------------------------------------------------------------------ 소켓 링크
    def link_to(self, host, port) -> _VLink:
        if not _is_loopback(host):
            raise ConnectionRefusedError(f"virtual clock: refusing {host}:{port} - the test clock only talks to local fakes (loopback)")
        return _VLink(self, host, port)

    def _link_attach(self, link: _VLink, me, peer) -> None:
        key = tuple(sorted([me, peer]))
        with self._lock:
            st = self._links.get(key)
            if st is None or (st.closed and st.n_attached >= 2):
                st = _LinkState(key)
                self._links[key] = st
            st.n_attached += 1
            link._st, link._out, link._in = st, (me, peer), (peer, me)

    def _link_sent(self, link: _VLink) -> None:
        st = link._st
        if st is None:
            return
        with self._lock:
            st.pending[link._out] = st.pending.get(link._out, 0) + 1

    def _dec(self, link: _VLink) -> None:
        st = link._st
        n = st.pending.get(link._in, 0)
        if n <= 0:
            self.misuse.append(f"link {st.key}: received() without a matching sent()")
        else:
            st.pending[link._in] = n - 1

    def _link_received(self, link: _VLink) -> None:
        if link._st is None:
            return
        with self._lock:
            self._dec(link)
            self._evaluate()

    @contextmanager
    def _receiving_cm(self, link: _VLink):
        th = threading.current_thread()
        with self._lock:                                     # 받음 계수와 임시 참여를 원자적으로 (그 사이 시간이 흐르지 않게)
            if link._st is not None:
                self._dec(link)
            mine = not self._shut and th not in self._parts
            if mine:
                self._parts[th] = _Part(th, next(self._seq), self._lock)
            elif link._st is not None:
                self._evaluate()
        try:
            yield
        finally:
            if mine:
                self._unregister(th, finished=False)

    def _receiving(self, link: _VLink):
        return self._receiving_cm(link)

    def _link_close(self, link: _VLink) -> None:
        st = link._st
        if st is None:
            return
        with self._lock:
            st.closed = True
            self._evaluate()

    # ------------------------------------------------------------------ 핵심
    def _park(self, pred, deadline: Optional[int], what: str) -> Optional[str]:
        """지금 스레드를 PARKED 로 두고 깨워질 때까지 (술어 참 / 마감 / 종료 / 교착). → 'pred' | 'deadline' | 'shutdown' | None (실시계 위임)."""
        th = threading.current_thread()
        with self._lock:
            if self._shut:
                return None
            p = self._parts.get(th)
            if p is None:
                msg = f"{what}() from unregistered thread {th.name!r} at t={self._t}"
                self.misuse.append(msg + "\n" + "".join(traceback.format_stack(limit=12)))
                if self.strict:
                    raise ClockMisuse(msg + " (register it before start, or use clock.acting())")
                return None
            p.pred, p.deadline = pred, deadline
            p.woken, p.why = False, None
            p.state = _PARKED
            self._evaluate()
            while not p.woken:
                p.cond.wait(self.poll_wall_s)
                if p.woken:
                    break
                if self._shut:
                    p.state, p.woken, p.why = _RUNNING, True, "shutdown"
                    break
                self._evaluate()                             # 비참여 스레드가 바꾼 것 (술어) · 교착 유예
            why = p.why
        if why == "deadlock":
            raise SimDeadlock(f"{what}() in {th.name!r}: every participant waits without a deadline and no predicate is true "
                              f"(t={self._t}); stacks: {self.deadlocks[-1] if self.deadlocks else ''}")
        return why

    def _wake(self, p: _Part, why: str) -> None:
        p.state, p.woken, p.why = _RUNNING, True, why
        p.pred = None
        p.deadline = None
        self.n_wakes += 1
        self._deadlock_since = None
        self._last_progress_wall = time.monotonic()
        self._watch_reported = False
        if self.trace is not None:
            self.trace.append((self._t, p.seq, why))
        p.cond.notify()

    def _evaluate(self) -> None:
        """잠금 안에서: 진행 규칙 1–5 (모듈 문서)."""
        if self._shut:
            return
        parts = list(self._parts.values())
        if not parts:
            return
        for p in parts:
            if p.state == _RUNNING:
                return
        for st in self._links.values():
            if st.blocking():
                return
        best = None
        for p in parts:
            if p.pred is not None:
                try:
                    ok = bool(p.pred())
                except Exception:  # noqa: BLE001 - let the waiter look at it
                    ok = True
                if ok and (best is None or p.seq < best.seq):
                    best = p
        if best is not None:
            self._wake(best, "pred")
            return
        for p in parts:
            if p.deadline is not None and (best is None or (p.deadline, p.seq) < (best.deadline, best.seq)):
                best = p
        if best is not None:
            if best.deadline > self._t:
                self._t = best.deadline
                self.n_advances += 1
            self._wake(best, "deadline")
            return
        now = time.monotonic()
        if self._deadlock_since is None:
            self._deadlock_since = now
            return
        if now - self._deadlock_since < self.deadlock_grace_s:
            return
        self.deadlocks.append(self._stacks(parts))
        for p in parts:
            self._wake(p, "deadlock")

    def _stacks(self, parts) -> str:
        frames = sys._current_frames()
        out = []
        for p in parts:
            f = frames.get(p.th.ident) if p.th.ident is not None else None
            st = "".join(traceback.format_stack(f, limit=10)) if f is not None else "  (no frame)\n"
            out.append(f"--- {p.name} seq {p.seq} state {'RUNNING' if p.state == _RUNNING else 'PARKED'} "
                       f"deadline {p.deadline} pred {p.pred is not None}\n{st}")
        return "".join(out)

    # ------------------------------------------------------------------ 감시·종료
    def _watch_loop(self) -> None:
        while not self._watch_stop.wait(1.0):
            with self._lock:
                if self._shut:
                    return
                if self._watch_reported or time.monotonic() - self._last_progress_wall < self.watchdog_s:
                    continue
                parts = list(self._parts.values())
                running = [p for p in parts if p.state == _RUNNING]
                links = [st for st in self._links.values() if st.blocking()]
                if running:
                    kind = "running"
                    text = f"no wake for {self.watchdog_s:.0f} s wall at t={self._t}: running participants\n" + self._stacks(running)
                elif links:
                    kind = "link"
                    text = f"no wake for {self.watchdog_s:.0f} s wall at t={self._t}: messages in flight " + \
                           "; ".join(f"{st.key} {dict(st.pending)}" for st in links)
                else:
                    kind = "parked"
                    text = f"no wake for {self.watchdog_s:.0f} s wall at t={self._t}: all parked\n" + self._stacks(parts)
                self.watchdog.append((kind, text))
                self._watch_reported = True
            print(f"[virtual clock watchdog] {kind}: {text[:2000]}", file=sys.stderr, flush=True)

    def shutdown(self) -> dict:
        """PARKED 대기자를 풀고 실시계 위임으로 바꾼다. → 보고 {alive_participants, misuse, deadlocks, watchdog, t_ns, wakes, advances}."""
        me = threading.current_thread()
        with self._lock:
            if self._shut:
                return self.shutdown_report or {}
            self._shut = True
            alive = [f"{p.name} (seq {p.seq})" for th, p in self._parts.items() if th is not me]
            for p in list(self._parts.values()):
                if p.state == _PARKED:
                    p.state, p.woken, p.why = _RUNNING, True, "shutdown"
                    p.cond.notify()
            self.shutdown_report = {"alive_participants": alive, "misuse": list(self.misuse), "deadlocks": list(self.deadlocks),
                                    "watchdog": list(self.watchdog), "t_ns": self._t, "wakes": self.n_wakes,
                                    "advances": self.n_advances}
        self._watch_stop.set()
        return self.shutdown_report
