"""시간 이음매 — 시계는 하나 (2026-10-09, 시험의 시뮬레이션 시계).

생산 코드는 시간을 직접 부르지 않고 이 모듈의 시계를 거친다. 기본 시계 ``REAL`` 은 실시계 그 자체다:

* ``REAL.monotonic`` / ``REAL.monotonic_ns`` / ``REAL.sleep`` 은 ``time.monotonic`` / ``time.monotonic_ns`` / ``time.sleep`` 과 **같은 객체** (``is``),
* ``wait`` / ``get`` / ``join`` 은 표준 호출에 같은 인자를 넘기는 한 줄 위임,
* ``register`` / ``acting`` / ``link_to`` / ``Link.*`` 는 공유 무동작.

그래서 실기 (그리고 시계를 바꾸지 않은 모든 실행) 의 동작은 이음매 전과 바이트까지 같다.

시험만 ``install(VirtualClock())`` 으로 주변 시계를 바꾼다 (``vc3dz_driver.virtual_clock`` — 생산 모듈은 그것을 import 하지 않는다;
``SCAN_SIM_CLOCK_ALLOW=1`` 이 있어야 설치된다 — 시험 conftest 만 설정한다). 객체는 **만들 때** ``get_clock()`` 을 잡아 두고 평생 그 시간축
하나만 쓴다. 표준 라이브러리만 쓴다 (가장 아래 패키지).
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager, nullcontext
from typing import Callable, Optional

__all__ = ["Clock", "RealClock", "REAL", "NULL_LINK", "get_clock", "install", "ALLOW_ENV"]

ALLOW_ENV = "SCAN_SIM_CLOCK_ALLOW"           # install() 허가 (시험 conftest 만 "1" 로 둔다)

_NULL_CTX = nullcontext()                    # 공유 무동작 컨텍스트 (nullcontext 는 다시 들어가도 된다)


class _NullLink:
    """실시계의 소켓 링크: 아무것도 하지 않는다 (공유 싱글턴 NULL_LINK)."""

    __slots__ = ()

    def attach(self, sock) -> None:
        return None

    def sent(self) -> None:
        return None

    def received(self) -> None:
        return None

    def receiving(self):
        return _NULL_CTX

    def close(self) -> None:
        return None


NULL_LINK = _NullLink()


class Clock:
    """시계 인터페이스 (설명용 기반 클래스).

    monotonic() / monotonic_ns()          지금 [s] / [ns]
    sleep(s)                              s 초 쉬기
    wait(event, timeout=None) -> bool     threading.Event.wait 와 같은 뜻
    get(q, timeout=None)                  queue.Queue.get(timeout=…) 와 같은 뜻 (없으면 queue.Empty)
    join(th, timeout=None)                Thread.join 과 같은 뜻
    register(th) -> th                    시계를 따르는 스레드로 등록 (start 전에)
    acting()                              컨텍스트: 그 범위 동안 지금 스레드가 시계를 따른다
    link_to(host, port) -> Link           소켓 링크 (메시지 송수신 계수; 실시계는 무동작)
    call_later(dt, fn)                    dt 초 뒤 fn() (시험 도우미)
    """

    virtual = False


class RealClock(Clock):
    """실시계 — 표준 시간 함수 그 자체 (기본, 실기)."""

    virtual = False
    monotonic = staticmethod(time.monotonic)
    monotonic_ns = staticmethod(time.monotonic_ns)
    sleep = staticmethod(time.sleep)

    @staticmethod
    def wait(event, timeout=None):
        return event.wait(timeout)

    @staticmethod
    def get(q, timeout=None):
        return q.get(timeout=timeout)

    @staticmethod
    def join(th, timeout=None):
        th.join(timeout)

    @staticmethod
    def register(th):
        return th

    @staticmethod
    def acting():
        return _NULL_CTX

    @staticmethod
    def link_to(host, port):
        return NULL_LINK

    @staticmethod
    def call_later(dt: float, fn: Callable[[], None]) -> threading.Timer:
        t = threading.Timer(float(dt), fn)
        t.daemon = True
        t.start()
        return t

    def __repr__(self) -> str:
        return "REAL"


REAL = RealClock()

_current: Clock = REAL
_install_lock = threading.Lock()


def get_clock() -> Clock:
    """주변 시계 (기본 REAL). 객체는 만들 때 한 번 잡아 둔다."""
    return _current


@contextmanager
def install(clock: Clock):
    """시험 전용: 주변 시계를 clock 으로 바꾼다. SCAN_SIM_CLOCK_ALLOW=1 이 없으면 거부, 중첩 금지. 들어갈 때 지금 스레드를 참여자 순번 0 으로,
    나올 때 clock.shutdown() 뒤 REAL 로 되돌린다 (shutdown 의 보고는 clock.shutdown_report)."""
    global _current
    if os.environ.get(ALLOW_ENV) != "1":
        raise RuntimeError(f"clock.install refused: {ALLOW_ENV}=1 is not set (only the test conftest sets it)")
    with _install_lock:
        if _current is not REAL:
            raise RuntimeError(f"clock.install refused: a clock is already installed ({_current!r})")
        _current = clock
    try:
        enter = getattr(clock, "_install_enter", None)
        if enter is not None:
            enter(threading.current_thread())
        yield clock
    finally:
        try:
            shut = getattr(clock, "shutdown", None)
            if shut is not None:
                shut()
        finally:
            with _install_lock:
                _current = REAL


def _reset_to_real_in_child() -> None:
    """fork 자식: 주변 시계를 REAL 로 (virtual_clock 이 import 될 때 등록한다)."""
    global _current
    _current = REAL


def installed() -> Optional[Clock]:
    """설치된 시계 (없으면 None) — 진단용."""
    return None if _current is REAL else _current
