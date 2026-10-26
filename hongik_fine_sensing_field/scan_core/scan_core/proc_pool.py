"""우리가 만드는 프로세스 풀의 수명 규칙 하나 (2026-10-09) — 패스 뒤 처리 작업 프로세스 (scan_l0.post_scan, 주인 = 서버), 그 작업 프로세스
안의 특징 풀 (fork, 주인 = 작업 프로세스), 특징·작업 풀 (seam_features, spawn) 이 모두 OwnedPool 이다.

규칙: 프로세스는 자기가 쓰는 파이프 끝만 든다 — 그래야 어느 쪽의 죽음이든 상대에게 프로세스 상태 (센티널의 끝·파이프의 EPIPE/EOF) 로 바로
보이고, 죽음은 그 상태로 안다 (시간 초과로 알지 않는다; 시간 한계는 살아 있지만 멈춘 프로세스에만 — 그것은 상태로 알 수 없다).
- 주인 (풀을 만든 프로세스) 은 호출 파이프에 쓰고 결과 파이프를 읽기만 한다: 자식을 모두 띄우면 (첫 submit — 표준 라이브러리가 fork 에서 이미
  그렇게 하듯 spawn 도) 호출 파이프의 읽는 끝과 결과 파이프의 쓰는 끝을 놓는다. 예전에는 주인이 든 읽는 끝 때문에 자식이 모두 죽어도 실행기의
  보내기 스레드가 가득 찬 호출 파이프에서 EPIPE 를 받지 못하고 영원히 막혔다 → 깨진 풀의 정리 (관리 스레드) 와 그것을 기다리는 shutdown·with
  종료·프로세스 종료가 멈췄다 (작업 프로세스 안의 특징 풀: 자식 하나가 죽으면 패스 뒤 처리가 서버의 시간 한계 300 s 까지 멈춤). 결과 파이프의
  쓰는 끝도 같다: 자식이 결과를 쓰다 죽으면 주인의 읽기가 나머지 바이트를 영원히 기다리지 않고 끝 (EOF) 을 본다.
- 자식은 호출 파이프를 읽고 결과 파이프에 쓰기만 한다: 시작할 때 (_child_start) 호출 파이프의 쓰는 끝·결과 파이프의 읽는 끝을 놓는다 → 주인이
  죽으면 기다리던 호출 파이프가 끝 (EOF) 이 되어 자식도 끝난다 (예전: spawn 자식은 두 끝을 다 넘겨받아 들고 있어 주인이 갑자기 죽으면 영원히
  남았다). 주인이 자기 윗선 (자기 주인) 과의 연결을 fork 자식이 물려받지 않게 하는 것은 그 연결을 아는 주인의 initializer 몫이다
  (scan_l0.post_scan._fork_child_setup — 그 연결 fd 를 놓고 주인과 함께 죽는다).
- 자식의 죽음은 주인이 프로세스 상태로 안다: 실행기의 관리 스레드가 센티널로 (→ 그 풀의 Future 는 BrokenProcessPool), 그리고 child_died()
  (센티널을 기다리지 않고 본다, 거두지 않는다). 자식이 하나라도 끝난 풀은 깨진 것이라 shutdown 이 나머지를 끝낸다 (SIGKILL — 표준 라이브러리는
  SIGTERM 으로 끝내는데 멈춘 프로세스는 그것으로 끝나지 않아 정리가 영원히 기다릴 수 있다). 돌려받지 못한 계산은 주인 안에서 같은 계산으로
  (사용처: seam_features._features_for, scan_l0.post_scan.run_post_scan).

- 자식의 죽음은 기다리는 쪽을 막지 않는다: 자식을 모두 띄운 뒤 주인의 감시 스레드 (_watch) 가 자식의 센티널을 기다리다 shutdown 전에 하나라도
  끝나면 나머지를 바로 끝낸다. 실행기의 관리 스레드는 결과를 쓰다 죽은 자식의 남은 바이트를 기다리며 막힐 수 있는데 (나머지 자식이 결과
  파이프의 쓰는 끝을 들고 있어 끝 (EOF) 이 오지 않는다 — 자식 2 개 이상이면 그 풀의 Future 가 영원히 끝나지 않았다), 나머지가 끝나면 그
  읽기가 끝을 보고 모든 Future 가 BrokenProcessPool 이 되어 주인의 같은 계산 규칙이 바로 이어진다 (시간 한계를 기다리지 않는다).
- 기다림의 한계는 살아 있지만 멈춘 자식에게만: wait_within (RESULT_TIMEOUT_S, 이 저장소의 풀 대기가 모두 같은 값) — 그 안에 답이 없으면
  기다린 쪽이 그 풀의 자식을 끝내고 (kill_children) 같은 계산을 자기 안에서 하거나 '하지 않음' 으로 알린다.

표준 라이브러리 내부에 기대는 곳 (파이썬 3.10.12 concurrent.futures.process — 끝을 놓거나 모두 띄우는 공개 API 가 없다): _adjust_process_count 를
재정의하고 _spawn_process 로 자식을 모두 띄운다; _processes·_max_workers 로 띄운 자식을 본다; _call_queue._reader·_call_queue._writer·
_result_queue._reader·_result_queue._writer 가 놓을 끝이다. 그 모양 (이름·종류) 은 import 때 (클래스) 와 풀을 만들 때 (인스턴스) 한 번 확인한다
(_missing_internals): 하나라도 다르면 빠진 것을 이름으로 적은 오류를 로그에 남기고, 그 내부가 필요한 규칙의 부분만 꺼진다 (_PARTS — 자식
보기·끝내기는 _processes 만, 모두 띄우기는 _spawn_process 등, 끝 놓기·감시는 큐의 끝까지; 꺼진 부분은 표준 그대로) — rule_note 로 사용처가
결과의 기록에 남긴다; 계산과 결과는 같다 (자식을 볼 수 없으면 죽음은 BrokenProcessPool 로만 안다). 조용한 기본값은 없다.
자식 시작 (_child_start) 은 공개 initializer 를 감싸 하고, 자식의 상태는 공개 Process.sentinel·exitcode·kill 로 본다.
"""
from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing.connection
import os
import threading
import weakref
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Optional

__all__ = ["OwnedPool", "detach_fds", "wait_within", "RESULT_TIMEOUT_S"]

_LOG = logging.getLogger(__name__)
RESULT_TIMEOUT_S = 300.0         # 살아 있지만 멈춘 자식에게서 결과를 기다리는 한계 [s] — 이 저장소의 풀 대기가 모두 같은 값 (scan_l0.post_scan 의
#                                  작업 프로세스 한계도 이것). 죽은 자식은 프로세스 상태로 바로 안다 (이 한계를 기다리지 않는다)

# 기대는 표준 라이브러리 내부 (모듈 설명): 클래스의 메서드, 인스턴스의 속성 (점으로 이은 경로 → 기대하는 종류)
_CLASS_INTERNALS = ("_adjust_process_count", "_spawn_process")
_INSTANCE_INTERNALS = (("_processes", dict), ("_max_workers", int),
                       ("_call_queue._reader", multiprocessing.connection.Connection),
                       ("_call_queue._writer", multiprocessing.connection.Connection),
                       ("_result_queue._reader", multiprocessing.connection.Connection),
                       ("_result_queue._writer", multiprocessing.connection.Connection))
# 규칙의 부분마다 기대는 내부 — 빠진 내부가 있는 부분만 꺼지고 나머지는 그대로 (자식을 볼 수 있으면 끝낼 수도 있다). 끝 놓기·감시 ('seal',
# 모두 띄운 뒤) 는 확인하는 내부 전부에 기댄다
_PARTS = (("see", ("_processes",)),                                                       # 자식 보기·죽음 (센티널)·끝내기
          ("spawn_all", ("_adjust_process_count", "_spawn_process", "_processes", "_max_workers")))   # 첫 submit 에 모두 띄우기
_LOGGED: set = set()


def _missing_internals(pool=None) -> dict:
    """표준 라이브러리 내부의 모양 확인 → {빠진·다른 내부의 이름: 설명} (비면 모두 있다). pool 이 없으면 클래스만."""
    out = {n: f"ProcessPoolExecutor.{n} (method)" for n in _CLASS_INTERNALS if not callable(getattr(ProcessPoolExecutor, n, None))}
    if pool is not None:
        for path, kind in _INSTANCE_INTERNALS:
            obj, ok = pool, True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if not ok or not isinstance(obj, kind):
                out[path] = (f"ProcessPoolExecutor().{path} ({kind.__name__}{'' if ok else ', missing'}"
                             f"{'' if not ok else ', got ' + type(obj).__name__})")
    return out


def _log_once(msg: str) -> None:
    if msg not in _LOGGED:
        _LOGGED.add(msg)
        _LOG.error(msg)


_CLASS_MISSING = _missing_internals()
if _CLASS_MISSING:
    _log_once("scan_core.proc_pool: this Python's concurrent.futures.process lacks " + ", ".join(_CLASS_MISSING.values())
              + " - the parts of the OwnedPool lifetime rule that need them are off: "
              + ", ".join([n for n, need in _PARTS if any(x in _CLASS_MISSING for x in need)] + ["seal"]))


def wait_within(futures, limit_s: float) -> tuple:
    """Future 들을 한 번의 한계 안에서 기다린다 (Future 마다가 아니라 전체에 한 번) → (끝난 것, 못 끝난 것) — concurrent.futures.wait.
    OwnedPool 의 자식이 죽으면 그 풀의 Future 는 바로 끝나므로 (BrokenProcessPool — 감시 스레드), 한계에 걸리는 것은 살아 있지만 멈춘 자식의
    것뿐이다. None 은 건너뛴다."""
    fs = [f for f in futures if f is not None]
    if not fs:
        return set(), set()
    return concurrent.futures.wait(fs, timeout=limit_s)


def detach_fds(fds) -> None:
    """fds 를 /dev/null 로 바꾼다 — 원래 열린 파일·파이프는 놓고 fd 번호는 남긴다 (물려받은 파이썬 객체가 그 번호를 나중에 닫거나 써도 다른
    파일을 건드리지 않게). 표준 입출력 (0–2) 은 그대로."""
    fds = [int(fd) for fd in fds if int(fd) > 2]
    if not fds:
        return
    null = os.open(os.devnull, os.O_RDWR)
    try:
        for fd in fds:
            if fd != null:
                try:
                    os.dup2(null, fd, inheritable=False)
                except OSError:
                    pass
    finally:
        os.close(null)


def _fd_identity(fd: int) -> Optional[tuple]:
    """fd 가 가리키는 열린 끝: (st_dev, st_ino, 접근 모드) — 파이프의 두 끝은 inode 가 같고 접근 모드 (읽기/쓰기) 로 갈린다. 못 보면 None."""
    try:
        import fcntl
        s = os.fstat(fd)
        return s.st_dev, s.st_ino, fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
    except (OSError, ValueError, ImportError):
        return None


def _end(conn) -> Optional[tuple]:
    """multiprocessing Connection → (fd 번호, _fd_identity) 또는 None."""
    try:
        fd = conn.fileno()
    except (OSError, ValueError, AttributeError):
        return None
    ident = _fd_identity(fd)
    return (fd, ident) if ident is not None else None


def _child_start(ends: list, initializer, initargs: tuple) -> None:
    """자식 프로세스 시작 (OwnedPool 이 감싼 initializer): 이 풀에서 자식이 쓰지 않는 끝 — 호출 파이프의 쓰는 끝, 결과 파이프의 읽는 끝 — 을
    놓는다 (그 fd 번호가 지금도 같은 끝일 때만: fork 는 번호를 그대로 물려받고 spawn 은 넘긴 번호를 그대로 쓴다). 그다음 주인의 initializer."""
    detach_fds([fd for fd, ident in ends if _fd_identity(fd) == ident])
    if initializer is not None:
        initializer(*initargs)


def _watch(pool_ref, sentinels: list, wake_r: int) -> None:
    """주인 안 (데몬 스레드 — 자식을 모두 띄운 뒤부터 shutdown 까지): 자식의 센티널과 깨우기 파이프를 기다린다. shutdown 전에 자식이 하나라도
    끝나면 (깨진 풀 — 프로세스 상태) 나머지를 바로 끝낸다: 결과를 쓰다 죽은 자식의 남은 바이트를 기다리던 관리 스레드가 끝 (EOF) 을 보고
    모든 Future 가 BrokenProcessPool 이 된다 (모듈 설명). shutdown 이 깨우면 아무것도 하지 않고 끝난다. 풀은 약한 참조로만 본다: 이 스레드가
    풀을 붙들지 않아야 shutdown 없이 버려진 풀도 표준 라이브러리의 수거 정리 (관리 스레드가 자식을 끝낸다) 를 받는다 — 수거된 풀 = 닫는 중."""
    try:
        ready = multiprocessing.connection.wait(sentinels + [wake_r])
        pool = pool_ref()
        if pool is None:
            return
        with pool._own_lock:
            closing = pool._closing
        if not closing and any(s in ready for s in sentinels):
            pool.kill_children()
    finally:
        os.close(wake_r)

class OwnedPool(ProcessPoolExecutor):
    """ProcessPoolExecutor + 수명 규칙 (모듈 설명): 첫 submit 에 자식을 모두 띄우고 주인은 자식만 쓰는 끝을 놓는다; 자식은 시작할 때 주인만
    쓰는 끝을 놓는다; child_died() = 프로세스 상태; 자식이 끝난 풀의 shutdown 은 나머지를 끝낸다. 계산·결과·Future 는 표준과 같다."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")
        ends: list = []                                 # 자식이 놓을 끝 — 큐가 생긴 뒤 채운다 (자식은 첫 submit 에 뜬다: fork 는 그때 메모리, spawn 은 그때 피클)
        super().__init__(max_workers=max_workers, mp_context=mp_context, initializer=_child_start,
                         initargs=(ends, initializer, tuple(initargs)))
        self._own_lock = threading.Lock()
        self._kids: list = []                           # 띄운 자식 (multiprocessing.Process) — 표준 shutdown 이 목록을 놓은 뒤에도 상태를 본다
        self._sealed = False
        self._closing = False
        self._saw_broken = False                        # 규칙 없는 풀: Future·submit 이 BrokenProcessPool 을 냈나 (공개 예외로만 안다)
        self._watch_thread: Optional[threading.Thread] = None
        self._watch_w: Optional[weakref.finalize] = None   # 감시 스레드를 깨우는 파이프의 쓰는 끝을 닫는 것 (shutdown·수거 때 한 번)
        missing = _missing_internals(self)
        self._can = {name: not any(x in missing for x in need) for name, need in _PARTS}   # 규칙의 부분마다 켜졌나
        self._can["seal"] = not missing
        self.rule_note: Optional[str] = None            # 규칙의 일부가 꺼졌으면 그 까닭 (사용처가 결과의 기록에 남긴다)
        if missing:
            off = [n for n, on in self._can.items() if not on]
            self.rule_note = (f"OwnedPool lifetime rule partly off ({', '.join(off)} off; plain stdlib ProcessPoolExecutor for those) - "
                              f"missing stdlib internal(s): " + ", ".join(missing.values()))
            _log_once("scan_core.proc_pool: " + self.rule_note)
        if self._can["seal"]:                           # 자식이 놓을 끝 (봉인이 꺼지면 비어 있다: 자식 시작 = 주인의 initializer 그대로)
            for c in (self._call_queue._writer, self._result_queue._reader):
                e = _end(c)
                if e is not None:
                    ends.append(e)

    # ---- 자식 띄우기: 모두 한 번에 (표준 라이브러리 내부 재정의) -------------------------------------------------------------------
    def _adjust_process_count(self):
        """표준 (spawn·forkserver): submit 마다 쉬는 자식이 없으면 하나 더. 여기서는 처음에 모두 — fork 에서 표준이 이미 하듯 — 그래야 주인이 첫
        submit 뒤 자식만 쓰는 끝을 놓을 수 있다 (더 띄울 자식이 없다). 그 내부가 없으면 표준 그대로."""
        if not self._can["spawn_all"]:
            return super()._adjust_process_count()
        while len(self._processes) < self._max_workers:
            self._spawn_process()
        return None

    def submit(self, fn, /, *args, **kwargs):
        if not self._can["see"]:                        # 자식을 볼 수 없는 풀: 표준 + 깨짐을 공개 예외로 기록 (child_died)
            try:
                f = super().submit(fn, *args, **kwargs)
            except BrokenProcessPool:
                self._saw_broken = True
                raise
            f.add_done_callback(self._note_broken)
            return f
        f = super().submit(fn, *args, **kwargs)
        if self._can["seal"] and len(self.children()) >= self._max_workers:
            self._seal(watch=True)                      # 자식을 모두 띄웠다 — 더 띄울 일이 없다
        return f

    def _note_broken(self, f) -> None:
        if not f.cancelled() and isinstance(f.exception(), BrokenProcessPool):
            self._saw_broken = True

    def _seal(self, watch: bool = False) -> None:
        """주인: 자식만 쓰는 끝 — 호출 파이프의 읽는 끝, 결과 파이프의 쓰는 끝 — 을 놓는다 (자식을 모두 띄운 뒤·shutdown). 한 번만 (풀의 잠금
        안에서 — 여러 스레드의 첫 submit·shutdown 이 겹쳐도), 닫지 않고 /dev/null 로 바꾼다 (detach_fds, 자식 쪽과 같은 규칙): 그 fd 번호는
        표준 라이브러리의 Connection 이 계속 갖고 나중에 다른 스레드 (큐의 보내기 스레드·관리 스레드) 에서 닫는다 — 여기서 번호를 놓으면 그
        사이 다른 스레드가 연 파일이 그 번호를 받아 대신 닫힐 수 있다. 파이프의 끝은 바꾸는 순간 놓인다 (EPIPE·EOF 는 닫기와 같다).
        watch (첫 submit — shutdown 이 아니면): 자식의 죽음을 감시하는 스레드를 띄운다 (_watch)."""
        with self._own_lock:
            if self._sealed or not self._can["seal"]:
                return
            self._sealed = True
            fds = []
            for c in (self._call_queue._reader, self._result_queue._writer):
                if not c.closed:
                    fds.append(c.fileno())
            detach_fds(fds)
            if not watch or self._closing:
                return
            kids = list(self._kids)
            r, w = os.pipe()
            # 깨우기 파이프의 쓰는 끝은 풀이 갖는다: shutdown 이 닫고, shutdown 없이 버려진 풀은 수거될 때 닫힌다 (감시 스레드가 끝 (EOF) 을 본다)
            self._watch_w = weakref.finalize(self, os.close, w)
            self._watch_thread = threading.Thread(target=_watch, args=(weakref.ref(self), [k.sentinel for k in kids], r),
                                                  name="owned-pool-watch", daemon=True)
            self._watch_thread.start()

    # ---- 프로세스 상태 ------------------------------------------------------------------------------------------------------------
    def children(self) -> list:
        """띄운 자식 프로세스 (multiprocessing.Process; 끝난 것 포함). 그 내부 (_processes) 가 없으면 볼 수 없다 (빈 목록)."""
        if not self._can["see"]:
            return []
        procs = self._processes                         # shutdown 뒤에는 None (표준이 목록을 놓는다) — 그때는 기억한 것
        with self._own_lock:
            for p in (list(procs.values()) if procs else []):
                if not any(p is k for k in self._kids):
                    self._kids.append(p)
            return list(self._kids)

    def child_died(self) -> bool:
        """자식이 하나라도 끝났나 — 프로세스 상태 (센티널; 기다리지 않고, 거두지 않는다). shutdown 전에는 '죽었다' (자식은 shutdown 에만 끝난다).
        자식을 볼 수 없는 풀: 그 풀이 BrokenProcessPool 을 냈나 (공개 예외)."""
        if not self._can["see"]:
            return self._saw_broken
        sentinels = []
        for p in self.children():
            try:
                sentinels.append(p.sentinel)
            except ValueError:                          # 닫힌 Process 객체 = 이미 끝나고 정리됨
                return True
        return bool(sentinels) and bool(multiprocessing.connection.wait(sentinels, timeout=0))

    def kill_children(self) -> None:
        """자식을 모두 끝낸다 (SIGKILL; 이미 거둔 자식은 건드리지 않는다 — Process.kill). 그 끝은 실행기가 센티널로 알고 깨진 풀로 정리한다.
        자식을 볼 수 없는 풀 (_processes 가 없음 — rule_note) 에서만 아무것도 하지 않는다."""
        for p in self.children():
            try:
                p.kill()
            except Exception:  # noqa: BLE001 - already gone
                pass

    def shutdown(self, wait=True, *, cancel_futures=False):
        """표준 shutdown + 규칙: 처음 부를 때 감시 스레드를 깨워 끝내고 자식만 쓰는 끝을 놓고 (더 띄울 자식이 없다), 자식이 하나라도 끝났으면
        (깨진 풀 — 프로세스 상태) 나머지를 끝낸다 — 정리 (관리 스레드) 와 wait 가 어떤 자식도 기다리지 않게. wait 면 감시 스레드도 기다린다."""
        with self._own_lock:
            first, self._closing = not self._closing, True
            w, self._watch_w = self._watch_w, None
        if w is not None:
            w()                                         # 감시 스레드의 깨우기 파이프 끝을 닫는다 (EOF) — 정상 종료라 아무것도 끝내지 않는다
        if first:
            self._seal()
            if self.child_died():
                self.kill_children()
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        th = self._watch_thread
        if wait and th is not None and th is not threading.current_thread():
            th.join()
