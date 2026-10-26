"""시계 이음매 (vc3dz_driver.clock) 와 시험 전용 가상 시계 (vc3dz_driver.virtual_clock).

E1 실시계 동일성: REAL 의 시간 함수는 표준 함수 그 자체, 대기는 같은 인자의 한 줄 위임, 등록·링크는 공유 무동작.
E6 가상 시계 의미: 정확한 시각, 같은 시각은 순번 순서, 술어 깨움은 시간을 바꾸지 않고 하나씩, 미수신 메시지는 시간을 멈춤, 받는 범위 안의
sleep·join 진행 (RESET 모형), start 전 등록에 틈 없음, 엄격 모드, 비루프백 거부, 교착 감지, 종료 해제, fork 자식은 REAL, 결정성.
"""
from __future__ import annotations

import os
import queue
import random
import socket
import threading
import time

import pytest

from vc3dz_driver import clock as C
from vc3dz_driver.clock import NULL_LINK, REAL, get_clock, install
from vc3dz_driver.virtual_clock import ClockMisuse, SimDeadlock, VirtualClock


# ---------------------------------------------------------------------------------------------------------------- E1
def test_real_clock_time_functions_are_the_standard_objects():
    assert REAL.monotonic is time.monotonic
    assert REAL.monotonic_ns is time.monotonic_ns
    assert REAL.sleep is time.sleep
    assert get_clock() is REAL and REAL.virtual is False


class _Rec:
    def __init__(self, ret=None, exc=None):
        self.calls = []
        self.ret, self.exc = ret, exc

    def wait(self, *a, **k):
        self.calls.append(("wait", a, k))
        return self.ret

    def get(self, *a, **k):
        self.calls.append(("get", a, k))
        if self.exc is not None:
            raise self.exc
        return self.ret

    def join(self, *a, **k):
        self.calls.append(("join", a, k))
        return self.ret


def test_real_clock_waits_are_one_line_delegations():
    sentinel = object()
    e = _Rec(ret=sentinel)
    assert REAL.wait(e, 0.25) is sentinel and e.calls == [("wait", (0.25,), {})]
    e2 = _Rec(ret=True)
    assert REAL.wait(e2) is True and e2.calls == [("wait", (None,), {})]
    q = _Rec(ret=sentinel)
    assert REAL.get(q, 1.5) is sentinel and q.calls == [("get", (), {"timeout": 1.5})]
    qe = _Rec(exc=queue.Empty())
    with pytest.raises(queue.Empty):
        REAL.get(qe, 0.1)
    assert qe.calls == [("get", (), {"timeout": 0.1})]
    th = _Rec(ret="ignored")
    assert REAL.join(th, 2.0) is None and th.calls == [("join", (2.0,), {})]
    # 진짜 큐: 표준 get 과 같은 예외
    with pytest.raises(queue.Empty):
        REAL.get(queue.Queue(), 0.0)


def test_real_clock_registration_and_links_are_shared_no_ops():
    th = threading.Thread(target=lambda: None)
    assert REAL.register(th) is th and th.ident is None          # 시작하지 않는다
    assert REAL.acting() is REAL.acting() is NULL_LINK.receiving()
    with REAL.acting():
        pass
    with NULL_LINK.receiving():
        pass
    assert REAL.link_to("192.168.3.15", 1096) is NULL_LINK        # 실시계는 막지 않는다 (실기)
    for m in (NULL_LINK.sent, NULL_LINK.received, NULL_LINK.close):
        assert m() is None
    assert NULL_LINK.attach(object()) is None


def test_install_needs_the_test_permission(monkeypatch):
    monkeypatch.delenv(C.ALLOW_ENV, raising=False)
    with pytest.raises(RuntimeError, match="refused"):
        with install(VirtualClock()):
            pass
    assert get_clock() is REAL


def test_install_refuses_nesting():
    with install(VirtualClock()) as c:
        assert get_clock() is c and c.virtual
        with pytest.raises(RuntimeError, match="already installed"):
            with install(VirtualClock()):
                pass
        assert get_clock() is c
    assert get_clock() is REAL


# ---------------------------------------------------------------------------------------------------------------- E6
@pytest.fixture
def vclock():
    c = VirtualClock(deadlock_grace_s=0.3, poll_wall_s=0.05)
    with install(c):
        yield c
    assert c.shutdown_report is not None


def test_sleep_lands_on_the_exact_time(vclock):
    t0 = vclock.monotonic_ns()
    assert t0 == 10 ** 12
    vclock.sleep(0.123)
    assert vclock.monotonic_ns() - t0 == 123_000_000
    vclock.sleep(1 / 125.0)
    assert vclock.monotonic_ns() - t0 == 131_000_000
    assert vclock.monotonic() == pytest.approx(1000.131)


def test_same_time_goes_in_registration_order(vclock):
    order = []

    def mk(name):
        def body():
            vclock.sleep(0.05)
            order.append(name)
        return threading.Thread(target=body, daemon=True)
    ths = [vclock.register(mk(n)) for n in "abc"]
    for t in reversed(ths):
        t.start()
    vclock.sleep(0.05)
    order.append("main")
    for t in ths:
        vclock.join(t, 1.0)
    assert order == ["main", "a", "b", "c"]


def test_predicate_wake_keeps_the_time_and_wakes_one_at_a_time(vclock):
    ev = threading.Event()
    seen = []

    def waiter(tag):
        def body():
            assert vclock.wait(ev, 10.0)
            seen.append((tag, vclock.monotonic_ns()))
            time.sleep(0.05)                             # 벽시계로 바쁘다: 그동안 다른 대기자는 깨지 않는다
            seen.append((tag + "-end", vclock.monotonic_ns()))
        return threading.Thread(target=body, daemon=True)
    ws = [vclock.register(waiter(t)) for t in ("w1", "w2")]
    for w in ws:
        w.start()
    vclock.sleep(0.2)
    t_set = vclock.monotonic_ns()
    ev.set()
    vclock.sleep(1.0)
    assert seen == [("w1", t_set), ("w1-end", t_set), ("w2", t_set), ("w2-end", t_set)]
    assert vclock.monotonic_ns() - t_set == 1_000_000_000


def _pair(vclock):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    l1 = vclock.link_to("127.0.0.1", port)
    c = socket.create_connection(("127.0.0.1", port))
    l1.attach(c)
    s, addr = srv.accept()
    l2 = vclock.link_to(*addr[:2])
    l2.attach(s)
    srv.close()
    return c, l1, s, l2


def test_a_message_in_flight_stops_the_time(vclock):
    c, l1, s, l2 = _pair(vclock)
    q: queue.Queue = queue.Queue()

    def rx():                                            # 비참여 수신 스레드: 받은 뒤 벽시계로 50 ms 바쁘다
        data = s.recv(5)
        time.sleep(0.05)
        q.put(data)
        l2.received()
    threading.Thread(target=rx, daemon=True).start()
    t0 = vclock.monotonic_ns()
    l1.sent()
    c.sendall(b"hello")
    assert vclock.get(q, 0.001) == b"hello"             # 시뮬 1 ms 마감이 있어도 메시지가 날아가는 동안 시간은 멈춘다
    assert vclock.monotonic_ns() == t0
    for x in (c, s):
        x.close()
    l1.close()
    l2.close()


def test_sleep_and_join_inside_a_receiving_scope_progress(vclock):
    """RESET 모형: 받는 쪽이 받은 메시지를 처리하는 동안 (임시 참여자) 스트림을 멈추고 (join) 응답을 늦춘다 (sleep)."""
    c, l1, s, l2 = _pair(vclock)
    streaming = threading.Event()
    streaming.set()
    ticks = []

    def stream():
        while streaming.is_set():
            ticks.append(vclock.monotonic_ns())
            vclock.sleep(0.010)
    st = vclock.register(threading.Thread(target=stream, daemon=True))
    st.start()
    replies: queue.Queue = queue.Queue()

    def server():
        s.recv(5)
        with l2.receiving():
            streaming.clear()
            vclock.join(st, 1.0)
            vclock.sleep(0.8)
            l2.sent()
            s.sendall(b"ok")

    def client_rx():
        replies.put(c.recv(2))
        l1.received()
    threading.Thread(target=server, daemon=True).start()
    threading.Thread(target=client_rx, daemon=True).start()
    vclock.sleep(0.035)
    t0 = vclock.monotonic_ns()
    l1.sent()
    c.sendall(b"RESET")
    assert vclock.get(replies, 2.0) == b"ok"
    assert vclock.monotonic_ns() - t0 == 5_000_000 + 800_000_000     # 다음 스트림 틱 (40 ms) 에서 멈춤 + 응답 지연 0.8 s
    assert not st.is_alive() and len(ticks) == 4
    for x in (c, s):
        x.close()


def test_register_before_start_leaves_no_gap(vclock):
    hits = []

    def body():
        time.sleep(0.05)                                 # 시작이 벽시계로 늦어도 시간은 흐르지 않는다
        hits.append(vclock.monotonic_ns())
    t0 = vclock.monotonic_ns()
    th = vclock.register(threading.Thread(target=body, daemon=True))
    th.start()
    vclock.sleep(1.0)
    assert hits == [t0]


def test_unregistered_threads_may_read_but_not_wait(vclock):
    err, now = [], []

    def body():
        now.append(vclock.monotonic_ns())
        try:
            vclock.sleep(0.1)
        except ClockMisuse as e:
            err.append(e)
    th = threading.Thread(target=body)
    th.start()
    th.join()
    assert now == [10 ** 12] and len(err) == 1 and len(vclock.misuse) == 1
    vclock.misuse.clear()


def test_non_loopback_hosts_are_refused(vclock):
    for h in ("192.168.3.15", "192.168.0.50", "sensor.local", "10.0.0.1"):
        with pytest.raises(ConnectionRefusedError):
            vclock.link_to(h, 1096)
    for h in ("127.0.0.1", "localhost", "::1"):
        vclock.link_to(h, 1096)


def test_deadlock_is_detected(vclock):
    with pytest.raises(SimDeadlock):
        vclock.wait(threading.Event())                   # 마감 없음, 다른 참여자 없음
    assert vclock.deadlocks
    vclock.deadlocks.clear()


def test_shutdown_releases_parked_waiters_and_reports_them():
    c = VirtualClock(poll_wall_s=0.05)
    out = []
    ev = threading.Event()

    def body():
        out.append(c.wait(ev))
        c.sleep(0.01)                                    # 종료 뒤: 실시계 위임 (오류 없음)
        out.append("after")
    with install(c):
        th = c.register(threading.Thread(target=body, daemon=True, name="leaky"))
        th.start()
        t_end = time.monotonic() + 5.0
        while time.monotonic() < t_end and c._parts[th].state != 1:
            time.sleep(0.001)
    th.join(2.0)
    assert out == [False, "after"]
    assert c.shutdown_report["alive_participants"] == ["leaky (seq 1)"] and not c.misuse
    assert get_clock() is REAL


def test_a_fork_child_sees_the_real_clock(vclock):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                         # pragma: no cover - child
        try:
            os.write(w, b"1" if C.get_clock() is C.REAL else b"0")
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    assert os.read(r, 1) == b"1"
    os.close(r)
    os.close(w)


def _scenario(seed: int):
    c = VirtualClock(trace=True)
    log = []
    with install(c):
        q: queue.Queue = queue.Queue()
        done = threading.Event()

        def producer(k):
            rng = random.Random(seed * 100 + k)

            def body():
                for i in range(25):
                    c.sleep(rng.choice((0.001, 0.002, 0.005, 0.010)))
                    time.sleep(random.random() * 0.001)  # 벽시계 흔들림 (결과에 영향 없어야 한다)
                    q.put((k, i))
                    log.append(("p", k, i, c.monotonic_ns()))
            return threading.Thread(target=body, daemon=True)

        def consumer():
            while True:
                try:
                    item = c.get(q, 0.003)
                except queue.Empty:
                    if done.is_set():
                        return
                    log.append(("idle", c.monotonic_ns()))
                    continue
                log.append(("c", item, c.monotonic_ns()))
        ths = [c.register(producer(k)) for k in range(3)] + [c.register(threading.Thread(target=consumer, daemon=True))]
        for t in ths:
            t.start()
        for t in ths[:3]:
            c.join(t, 10.0)
        done.set()
        c.join(ths[3], 10.0)
    return log, c.trace


def test_random_multithread_scenario_is_deterministic():
    a = _scenario(7)
    b = _scenario(7)
    assert a == b and len(a[0]) > 75


# ---------------------------------------------------------------------------------------------------------------- S1
def test_s1_the_client_cannot_reach_a_non_loopback_host_under_the_virtual_clock(vclock, monkeypatch):
    calls = []

    def no_socket(*a, **k):
        calls.append(a)
        raise AssertionError("no socket may be opened")
    monkeypatch.setattr(socket, "create_connection", no_socket)
    from vc3dz_driver.client import VC3DZClient
    for host in ("192.168.3.15", "192.168.0.50"):
        c = VC3DZClient(host, 1096)
        with pytest.raises(ConnectionRefusedError):
            c.connect()
        assert c.sock is None
    assert calls == []


def test_e1_the_client_on_the_real_clock_holds_the_real_seams():
    from vc3dz_driver.client import VC3DZClient
    c = VC3DZClient("127.0.0.1", 1)
    assert c._clock is REAL and c._link is NULL_LINK
