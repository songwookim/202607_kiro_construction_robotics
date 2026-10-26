"""프로세스 풀의 수명 규칙 (scan_core.proc_pool.OwnedPool, 2026-10-09): 프로세스는 자기가 쓰는 파이프 끝만 든다 → 어느 쪽의 죽음이든 상대가
프로세스 상태로 바로 안다. fork·spawn 둘 다:
- 주인 쪽: 자식 하나가 죽는다 (SIGKILL) — 보내기 스레드가 가득 찬 호출 파이프에 쓰던 중이어도 — Future 는 바로 BrokenProcessPool, child_died(),
  shutdown(wait=True) 가 바로 끝나고 자식은 모두 끝나 거둬진다 (표준 실행기: 주인이 든 읽는 끝 때문에 shutdown 이 영원히 멈춘다).
- 자식 쪽: 주인이 갑자기 죽는다 (SIGKILL) — 자식은 호출 파이프의 끝 (EOF) 으로 알고 끝난다 (표준 spawn 실행기: 영원히 남는다)."""
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures.process import BrokenProcessPool

import pytest

from scan_core.proc_pool import OwnedPool

BIG = 4 << 20                       # 호출 하나의 바이트 (파이프 용량보다 훨씬 큼 — 보내기 스레드가 파이프에서 막힌다)


def _alive(pid: int) -> bool:
    """끝나지 않았나 (좀비 = 끝남)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _until(cond, limit_s: float) -> bool:
    t_end = time.monotonic() + limit_s
    while not cond() and time.monotonic() < t_end:
        time.sleep(0.005)
    return bool(cond())


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_the_owner_sees_a_dead_child_and_never_waits_for_it(method):
    ex = OwnedPool(max_workers=2, mp_context=multiprocessing.get_context(method))
    busy = [ex.submit(time.sleep, 2.0) for _ in range(2)]               # 두 자식이 바쁜 동안
    big = [ex.submit(len, b"x" * BIG) for _ in range(4)]                # 큰 호출이 호출 파이프에서 막힌다
    kids = ex.children()
    assert len(kids) == 2 and not ex.child_died()                       # 첫 submit 에 모두 떴다
    assert _until(lambda: ex._call_queue._thread is not None, 30.0)    # (시험만: 보내기 스레드가 쓰기 시작)
    time.sleep(0.3)
    t0 = time.monotonic()
    os.kill(kids[0].pid, signal.SIGKILL)
    with pytest.raises(BrokenProcessPool):
        big[-1].result(30.0)
    t_detect = time.monotonic() - t0
    assert ex.child_died()
    th = threading.Thread(target=ex.shutdown, kwargs={"wait": True}, daemon=True)
    th.start()
    th.join(30.0)
    assert not th.is_alive(), "shutdown(wait=True) waits for a feeder blocked on the call pipe"
    assert all(k.exitcode is not None for k in kids) and not any(_alive(k.pid) for k in kids)
    assert t_detect < 10.0, t_detect
    for f in busy + big:
        assert f.done()


_OWNER = r'''
import multiprocessing, os, sys, time
from scan_core.proc_pool import OwnedPool
if __name__ == "__main__":
    ex = OwnedPool(max_workers=2, mp_context=multiprocessing.get_context(sys.argv[1]))
    ex.submit(os.getpid).result()
    print(" ".join(str(k.pid) for k in ex.children()), flush=True)
    time.sleep(600)
'''


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_children_end_when_their_owner_dies(method, tmp_path):
    owner_py = tmp_path / "owner.py"
    owner_py.write_text(_OWNER, encoding="utf-8")
    o = subprocess.Popen([sys.executable, str(owner_py), method], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    kids = []
    try:
        kids = [int(s) for s in o.stdout.readline().split()]
        assert len(kids) == 2 and all(_alive(k) for k in kids), kids
        os.kill(o.pid, signal.SIGKILL)
        o.wait(30.0)
        assert _until(lambda: not any(_alive(k) for k in kids), 10.0), [k for k in kids if _alive(k)]
    finally:
        for k in kids:
            if _alive(k):
                os.kill(k, signal.SIGKILL)
        if o.poll() is None:
            o.kill()
            o.wait()


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_the_owner_releases_the_child_ends_once_and_keeps_their_numbers(method, monkeypatch):
    """주인이 자식만 쓰는 끝 (호출 파이프의 읽는 끝·결과 파이프의 쓰는 끝) 을 놓는 일 (_seal) 은 한 번뿐이다 — 여러 스레드의 첫 submit 이
    겹쳐도 — 그리고 fd 번호를 놓지 않는다 (/dev/null 로 바뀐다): 그 번호는 표준 라이브러리의 Connection 이 계속 갖고 나중에 다른 스레드 (큐의
    보내기 스레드) 에서 닫으므로, 번호를 놓았다면 그사이 다른 스레드가 연 파일이 그 번호를 받아 대신 닫힐 수 있다 (예전: close — 겹친 두
    스레드가 같은 번호를 두 번 닫을 수 있었다)."""
    import scan_core.proc_pool as PP
    calls: list = []
    real = PP.detach_fds
    monkeypatch.setattr(PP, "detach_fds", lambda fds: (calls.append(list(fds)), real(fds))[1])
    ex = OwnedPool(max_workers=2, mp_context=multiprocessing.get_context(method))
    ends = [ex._call_queue._reader, ex._result_queue._writer]
    numbers = [c.fileno() for c in ends]
    null = os.stat(os.devnull)
    bar = threading.Barrier(8)
    pids: list = []

    def first_submit():
        bar.wait()
        pids.append(ex.submit(os.getpid).result(60.0))
        ex._seal()
    ths = [threading.Thread(target=first_submit) for _ in range(8)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(60.0)
    assert len(pids) == 8 and set(pids) <= {k.pid for k in ex.children()}, pids
    assert calls == [numbers], calls                                    # 한 번, 두 끝
    for c, fd in zip(ends, numbers):
        s = os.fstat(fd)
        assert not c.closed and c.fileno() == fd and (s.st_dev, s.st_ino) == (null.st_dev, null.st_ino)
    ex.shutdown(wait=True)
    assert calls == [numbers] and all(k.exitcode == 0 for k in ex.children())


def _in_result_write(pid: int, nbytes: int) -> bool:
    """pid 가 지금 결과 파이프에 큰 결과를 쓰는 중인가 (리눅스 /proc/<pid>/syscall: write(fd, buf, count) 의 count 가 결과의 대부분)."""
    try:
        with open(f"/proc/{pid}/syscall", encoding="ascii") as fh:
            sc = fh.read().split()
    except OSError:
        return False
    return len(sc) > 3 and sc[0] == "1" and int(sc[3], 16) > nbytes // 4


@pytest.mark.parametrize("method", ["fork", "spawn"])
@pytest.mark.parametrize("n", [2, 4])
def test_a_child_dying_while_writing_its_result_breaks_the_pool_at_once(method, n):
    """자식이 결과를 쓰는 도중에 죽는다 (SIGKILL, 결과 파이프 write 안): 주인의 감시 스레드가 그 죽음을 센티널 (프로세스 상태) 로 알고 나머지
    자식을 끝낸다 → 남은 바이트를 기다리던 관리 스레드가 끝 (EOF) 을 보고 Future 는 바로 BrokenProcessPool, shutdown 도 바로 끝난다. 예전:
    자식이 2 개 이상이면 나머지가 결과 파이프의 쓰는 끝을 들고 있어 그 Future 가 끝나지 않았다 (기다리는 쪽의 시간 한계까지)."""
    threads0 = threading.active_count()
    ex = OwnedPool(max_workers=n, mp_context=multiprocessing.get_context(method))
    assert ex.rule_note is None
    assert [f.result(60.0) for f in [ex.submit(int) for _ in range(n)]] == [0] * n
    kids = ex.children()
    nbytes = 256 << 20
    f = ex.submit(bytes, nbytes)                                        # 큰 결과 (파이프 용량의 수천 배 — 쓰는 데 시간이 걸린다)
    victim, t_end = None, time.monotonic() + 60.0
    while victim is None and time.monotonic() < t_end:                  # 그 write 안에 있는 자식을 보는 즉시 끝낸다
        for k in kids:
            if _in_result_write(k.pid, nbytes):
                os.kill(k.pid, signal.SIGKILL)
                victim = k
                break
    assert victim is not None, "never caught a child inside its result write"
    t0 = time.monotonic()
    with pytest.raises(BrokenProcessPool):
        f.result(60.0)
    t_broken = time.monotonic() - t0
    assert _until(lambda: not any(_alive(k.pid) for k in kids), 30.0), [k.pid for k in kids if _alive(k.pid)]
    th = threading.Thread(target=ex.shutdown, kwargs={"wait": True}, daemon=True)
    th.start()
    th.join(30.0)
    assert not th.is_alive() and all(k.exitcode is not None for k in kids)
    assert t_broken < 30.0, t_broken
    assert _until(lambda: threading.active_count() <= threads0, 10.0), threading.enumerate()


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_a_stalled_child_is_bounded_by_one_limit_and_ended_by_the_waiter(method):
    """살아 있지만 멈춘 자식 (SIGSTOP — 프로세스 상태로는 알 수 없다): wait_within 은 Future 들 전체에 한계 한 번만 기다리고 (못 끝난 것을
    돌려준다), 기다린 쪽이 그 풀의 자식을 끝내면 (kill_children) 못 끝난 Future 는 BrokenProcessPool, shutdown 은 기다리지 않는다."""
    from scan_core.proc_pool import wait_within
    ex = OwnedPool(max_workers=2, mp_context=multiprocessing.get_context(method))
    assert [f.result(60.0) for f in [ex.submit(int) for _ in range(2)]] == [0, 0]
    kids = ex.children()
    futs = [ex.submit(time.sleep, 0.5) for _ in range(2)]               # 두 자식이 하나씩
    assert _until(lambda: all(_in_sleep(k.pid) for k in kids), 30.0)
    os.kill(kids[0].pid, signal.SIGSTOP)
    limit = 3.0                                                         # 시험 시간만 (한계의 값이 아니라 한계 뒤를 본다)
    t0 = time.monotonic()
    done, pending = wait_within(futs + [None], limit)
    dt = time.monotonic() - t0
    assert len(done) == 1 and len(pending) == 1 and limit <= dt < limit + 10.0, (len(done), len(pending), dt)
    ex.kill_children()
    with pytest.raises(BrokenProcessPool):
        next(iter(pending)).result(30.0)
    th = threading.Thread(target=ex.shutdown, kwargs={"wait": True}, daemon=True)
    th.start()
    th.join(30.0)
    assert not th.is_alive() and not any(_alive(k.pid) for k in kids)


def _in_sleep(pid: int) -> bool:
    """pid 가 일 (time.sleep) 을 하는 중인가 — 실행 중도, 호출 파이프를 읽으며 (read, 0) 일을 기다리는 중도 아닌 시스템 호출 안
    (리눅스 x86-64 /proc/<pid>/syscall)."""
    try:
        with open(f"/proc/{pid}/syscall", encoding="ascii") as fh:
            return fh.read().split()[0] not in ("running", "-1", "0")
    except (OSError, IndexError):
        return False


def test_the_stdlib_internals_are_checked_once_and_a_missing_one_turns_off_only_the_parts_that_need_it(monkeypatch, caplog):
    """표준 라이브러리 내부의 모양은 import 때 (클래스) 와 풀을 만들 때 (인스턴스) 확인한다. 이 파이썬에서는 모두 있다 (다른 파이썬으로 바뀌면 이
    시험이 먼저 말한다). 하나라도 다르면 조용히 건너뛰지 않는다: 빠진 것을 이름으로 적은 오류 로그 + rule_note, 그리고 그 내부가 필요한 규칙의
    부분만 꺼진다 — 큐의 끝이 없으면 봉인·감시만 꺼지고 자식 보기·끝내기는 그대로 (멈춘 자식도 끝낼 수 있어 종료가 멈추지 않는다); _processes
    가 없으면 자식을 볼 수 없어 죽음은 공개 예외 BrokenProcessPool 로만 안다. 계산은 같다."""
    import logging
    import scan_core.proc_pool as PP
    assert PP._CLASS_MISSING == {} and PP._missing_internals(OwnedPool(max_workers=1)) == {}
    monkeypatch.setattr(PP, "_INSTANCE_INTERNALS", PP._INSTANCE_INTERNALS + (("_call_queue._no_such_end", int),))
    lg = logging.getLogger("scan_core.proc_pool")                       # (ROS launch_testing 의 로거는 루트로 올리지 않는다 — 직접 붙인다)
    lg.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.ERROR, logger="scan_core.proc_pool"):
            ex = OwnedPool(max_workers=2, mp_context=multiprocessing.get_context("fork"))
    finally:
        lg.removeHandler(caplog.handler)
    assert ex.rule_note and "_call_queue._no_such_end" in ex.rule_note and "seal off" in ex.rule_note, ex.rule_note
    assert ex._can == {"see": True, "spawn_all": True, "seal": False}, ex._can
    assert any("_call_queue._no_such_end" in r.getMessage() for r in caplog.records), caplog.records
    assert [f.result(60.0) for f in [ex.submit(pow, 3, i) for i in range(6)]] == [3 ** i for i in range(6)]
    kids = ex.children()
    assert len(kids) == 2 and not ex.child_died() and ex._watch_thread is None
    for k in kids:                                                      # 살았지만 멈춘 자식도 끝낼 수 있다 (자식 보기는 켜져 있다)
        os.kill(k.pid, signal.SIGSTOP)
    ex.kill_children()
    assert _until(lambda: not any(_alive(k.pid) for k in kids), 30.0)
    th = threading.Thread(target=ex.shutdown, kwargs={"wait": True}, daemon=True)
    th.start()
    th.join(30.0)
    assert not th.is_alive()
    # _processes 가 없는 파이썬: 자식을 볼 수 없다 → 죽음은 공개 예외로만
    monkeypatch.setattr(PP, "_INSTANCE_INTERNALS", PP._INSTANCE_INTERNALS + (("_no_processes_here", dict),))
    monkeypatch.setattr(PP, "_PARTS", tuple((n, need + ("_no_processes_here",)) for n, need in PP._PARTS))
    ex = OwnedPool(max_workers=2, mp_context=multiprocessing.get_context("fork"))
    assert not any(ex._can.values()) and ex.children() == [] and not ex.child_died()
    os.kill(ex.submit(os.getpid).result(60.0), signal.SIGKILL)
    assert _until(lambda: ex.child_died() or not _probe_ok(ex), 30.0) and ex.child_died()
    ex.shutdown(wait=True)


def _probe_ok(ex) -> bool:
    """규칙 없는 풀에 작은 일을 하나 보내 본다 (깨졌으면 그 Future·submit 이 BrokenProcessPool — child_died 가 기록한다)."""
    try:
        ex.submit(int).result(30.0)
        return True
    except BrokenProcessPool:
        return False


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_a_pool_dropped_without_shutdown_is_collected_and_its_children_end(method):
    """shutdown 없이 버려진 풀: 감시 스레드는 풀을 약한 참조로만 보므로 풀이 수거되고, 표준 라이브러리의 수거 정리 (관리 스레드가 자식을 끝낸다)
    로 자식이 끝나고 감시 스레드도 끝난다 (예전: 감시 스레드의 대상이 풀의 메서드라 풀이 영원히 살아 자식이 남았다)."""
    import gc
    import weakref
    threads0 = threading.active_count()
    ex = OwnedPool(max_workers=2, mp_context=multiprocessing.get_context(method))
    assert [f.result(60.0) for f in [ex.submit(int) for _ in range(2)]] == [0, 0]
    kids = ex.children()
    ref = weakref.ref(ex)
    del ex
    gc.collect()
    assert ref() is None
    assert _until(lambda: not any(_alive(k.pid) for k in kids), 30.0), [k.pid for k in kids if _alive(k.pid)]
    assert _until(lambda: threading.active_count() <= threads0, 30.0), threading.enumerate()
