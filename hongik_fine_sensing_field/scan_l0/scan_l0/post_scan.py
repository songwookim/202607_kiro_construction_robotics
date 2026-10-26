"""패스 뒤 처리 (원자료 저장 → 재구성 → QC → 재구성 저장 → fit_pass) 를 전용 작업 프로세스에서 (2026-10-05).

실기 1G: 스캔 이동 12–15 s 뒤 처리 12–14 s 인데 같은 코드·같은 원자료가 서버 밖에서는 4–5 s. MoveIt 가짜 팔 서버로 재현하니 서버 프로세스
안의 numpy 계산이 5–12 배 느렸고 (같은 시각 작업 프로세스에서는 정상), CPU 는 남아 있었다 — ROS 2 실행기 스레드 (초당 수백 번 깨어남) 와
같은 프로세스라 numpy 가 호출마다 GIL 을 놓았다 다시 잡을 때 뒤에서 기다린다 (전환 간격을 늘려도 그대로). 그래서 무거운 계산을 별도 프로세스로
옮긴다: 서버는 데이터만 넘기고 결과를 기다린다 (기다리는 동안 GIL 을 놓아 ROS 콜백은 그대로 돈다). 특징 추출의 병렬 풀은 작업 프로세스
안에서 만든다 (spawn, 중첩 허용). 작업 프로세스가 실패하면 같은 함수를 서버 안에서 돌린다 — 패스를 잃지 않는다.

패스 중 판 성장 스트림 (2026-10-08): 프로파일 특징 추출 시간의 ~70 % 인 판 성장 (seam_features.PlateSeed) 은 프로파일 점 (x, z) 과 설계·작업
허용치만의 함수라 (로봇 자세·시각 모형과 무관) 녹화되는 대로 계산할 수 있다. 서버의 SeedFeeder 스레드가 녹화 목록을 읽기만 해서 (녹화
스레드·가드는 건드리지 않는다) 새 프레임을 이름 있는 파이프 (FIFO) 에 바이트 그대로 쓰고, 작업 프로세스는 별도 fork 계산 프로세스에서
가장 큰 nice (리눅스 19 — 보통 시분할에서 가장 낮은 우선순위, 녹화·가드·ROS 아래; _lowest_priority) 로 판 성장을 계산해 둔다. 이동이 끝난
뒤 compute_pass 는 재구성 행의 점·유효 마스크가 스트림에 받은 것과 바이트까지 같고 설계·인자도 같은 행에만 그 결과를 쓰고 (같은 입력의 같은
순수 함수 → 같은 결과), 나머지는 예전처럼 처음부터 계산한다 — 그때 계산 중이던 프레임도 기다리지 않고 버린다 (가장 낮은 우선순위의 일을 보통
우선순위의 패스 뒤 처리가 기다리지 않게). 스트림이 어떻게 실패해도 결과는 같고 느려질 뿐이다.

서버 쪽 비용은 설계상 묶여 있다: 작업 프로세스와의 실행기 왕복은 패스마다 한 번 (stream_open), 프레임은 피클 없이 배열 메모리 그대로
writev (커널 복사 한 번), 응답·콜백이 없다. 피더는 파이프 한 통 분량이 녹화될 때마다 (파이프 용량 프레임 수 / 센서 프레임률) 깨어 그만큼을
한 번에 쓰므로 깨는 횟수는 패스 바이트 / 파이프 용량 정도 — 프레임 수·작업자 수와 따로다 (첫 판 2026-10-08 의 주기 = 작업자 수 / 프레임률
은 작업자 1 이면 프레임마다 깼다). 보고하는 서버 CPU 는 잰 것만 (피더 스레드 + 시작·멈춤 호출 + 그동안 실행기 스레드; 하나라도 못 재면 NaN).

프로세스 수명 (2026-10-09): 작업 프로세스가 fork 하는 계산 프로세스 (스트림·특징 풀) 는 서버와의 연결 fd (작업 프로세스가 spawn 때 받은 것:
실행기 호출·결과 파이프, 작업 프로세스의 죽음을 서버에 알리는 센티널 파이프) 를 놓고, 부모가 죽으면 같이 죽는다 (PR_SET_PDEATHSIG). 예전에는
그 fd 를 물려받아 작업 프로세스가 죽어도 서버 실행기의 센티널이 닫히지 않았고 (서버는 RESULT_TIMEOUT_S 300 s 를 기다리고, 계산 프로세스는 고아로
남아 서버 종료까지 막았다) — 이제 작업 프로세스의 죽음은 서버 실행기가 프로세스 상태 (센티널) 로 바로 알고 패스는 서버 안에서 같은 계산으로.
두 풀 (서버의 작업 프로세스 풀, 작업 프로세스 안의 특징 풀) 은 같은 규칙의 scan_core.proc_pool.OwnedPool 이다: 주인은 자식만 쓰는 파이프 끝을
들지 않고 (예전: 특징 풀의 자식 하나가 죽으면 주인 = 작업 프로세스가 든 호출 파이프의 읽는 끝 때문에 풀 정리가 영원히 멈춰 서버가 300 s 를
기다렸다), 자식의 죽음을 프로세스 상태로 안다 — 특징 풀의 자식이 죽으면 그 패스의 못 받은 덩어리는 작업 프로세스 안에서 바로 같은 계산
(seam_features._features_for). 서버는 '작업 프로세스가 죽었다' 를 예외 종류가 아니라 작업 프로세스 자신의 상태 (센티널) 로 가른다.
작업 프로세스가 끝난 실행기 (패스 사이에 죽은 경우 등) 는 다음에 쓸 때 새로 만든다 (post_scan_pool — 특징 풀과 같은 규칙).
작업 프로세스 자신도 같은 원시로 서버와 함께 죽는다 (S3, 2026-10-09: _worker_init — PDEATHSIG + 부모 재확인; 띄우는 스레드는 서버 수명 동안 산다).
"""
from __future__ import annotations

import atexit
import ctypes
import json
import math
import multiprocessing
import os
import pickle
import select
import signal
import socket
import struct
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from scan_core import proc_pool as _proc_pool
from scan_core.proc_pool import OwnedPool, detach_fds

__all__ = ["compute_pass", "run_post_scan", "post_scan_pool", "shutdown_post_scan_pool", "warm_post_scan_worker", "flush_saves",
           "submit_flush", "SeedFeeder", "stream_open", "stream_close", "vis_cloud", "submit_vis_cloud", "FeatureFeeder",
           "feature_stream_open", "feature_stream_batch", "feature_stream_close", "compute_track", "run_track", "refit_track", "run_refit",
           "Cancelled", "wait_result"]

_POOL: Optional[OwnedPool] = None
_LOCK = threading.Lock()
RESULT_TIMEOUT_S = _proc_pool.RESULT_TIMEOUT_S   # 살아 있지만 멈춘 작업 프로세스의 한계 (이 저장소의 풀 대기가 모두 같은 값 — scan_core.proc_pool)
#                                                — 죽은 작업 프로세스는 센티널 (프로세스 상태) 로 바로 안다
_PENDING: list = []              # 작업 프로세스 안: 판정 뒤로 미룬 파일 쓰기 (raw.npz·recon/) — flush_saves 가 쓴다
_STREAM: Optional[dict] = None   # 작업 프로세스 안: 지금 패스의 판 성장 스트림 (stream_open → 받기 스레드 … → compute_pass 가 닫는다)
_CHILD: Optional[tuple] = None   # 스트림 계산 프로세스 안: (설계, 추출 인자, 센서 유효 z 창) — fork 로 물려받는다
_HDR = struct.Struct("<qII")     # 스트림 프레임 머리: result_counter, x 점 수, z 점 수 (뒤에 x·z float32 바이트)
_MSG = struct.Struct("<Q")       # 계산 프로세스 → 작업 프로세스 메시지 머리: 피클 길이 (뒤에 피클)
_FIFO_NAME = "frames"            # 스트림 디렉터리 안의 FIFO 이름
_SERVER_FDS: dict = {}           # 작업 프로세스 안: 서버에서 물려받은 fd → (st_dev, st_ino) (시작할 때, _worker_init)
_REAP: set = set()               # 작업 프로세스 안: 죽였지만 아직 거두지 않은 스트림 계산 프로세스 pid
_DIRS: set = set()               # 서버 안: 만들고 아직 지우지 못한 스트림 디렉터리 (모든 종료 경로에서 지운다)
_DIRS_LOCK = threading.Lock()
try:
    _IOV_MAX = max(3, int(os.sysconf("SC_IOV_MAX")))   # writev 한 번의 조각 수 한계 (OS)
except (ValueError, OSError, AttributeError):
    _IOV_MAX = 16                                     # POSIX 가 보장하는 최소 (_XOPEN_IOV_MAX)
_VIS_SRC: Optional[tuple] = None   # 작업 프로세스 안: 처리된 패스 점구름을 만들 (재구성 결과, 점 간격 m) — vis_cloud 가 쓰고 비운다
try:   # 리눅스 prctl — fork 전에 (부모에서) 찾아 둔다 (자식에서 동적 적재하지 않게)
    _PRCTL = ctypes.CDLL(None, use_errno=True).prctl
    _PRCTL.restype = ctypes.c_int
    _PRCTL.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
except (OSError, AttributeError, TypeError):
    _PRCTL = None
_PR_SET_PDEATHSIG = 1            # <linux/prctl.h> (커널 ABI)


def post_scan_pool() -> OwnedPool:
    """전용 작업 프로세스 하나 (spawn, 서버 수명 동안 유지 — 모듈 import·특징 풀 시작은 한 번). OwnedPool: 서버는 그 실행기의 자식만 쓰는 파이프
    끝을 들지 않는다 (예: 작업 프로세스가 패스 원자료 ~15 MB 를 다 받기 전에 죽어도 보내던 쓰기가 EPIPE 로 끝나 정리·서버 종료가 멈추지 않는다).
    작업 프로세스가 끝난 실행기 (프로세스 상태 — 패스 사이 쉬는 동안 죽은 경우 등) 는 다음에 쓸 때 새로 만든다 (seam_features._feature_pool 과
    같은 규칙: 깨진 실행기를 계속 내주면 다음 패스의 데우기·스트림 시작이 실패하고 패스 뒤 처리가 서버 안에서 돈다).
    작업 프로세스는 서버와 함께 죽는다 (_worker_init: 계산 프로세스와 같은 원시 — PR_SET_PDEATHSIG + 부모 재확인). 리눅스의 '부모' 는 자식을 띄운
    **스레드** 라 (그 스레드가 끝나면 서버가 살아 있어도 신호가 온다), 띄우기는 서버가 사는 동안 끝나지 않는 스레드에서 한다 (_on_lasting_thread
    — 패스마다의 피더 스레드처럼 짧은 스레드가 첫 submit 을 해도 작업 프로세스가 그 스레드와 함께 죽지 않게)."""
    global _POOL
    with _LOCK:
        dead = _POOL if (_POOL is not None and _POOL.child_died()) else None
        if _POOL is None or dead is not None:
            pool = OwnedPool(max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init,
                             initargs=(os.getpid(),))
            _on_lasting_thread(lambda: pool.submit(os.getpid))   # 첫 submit = 자식 띄우기 (OwnedPool: 모두 한 번에) — 오래 사는 스레드에서
            _POOL = pool
        pool = _POOL
    if dead is not None:
        dead.shutdown(wait=False, cancel_futures=True)  # 죽은 작업 프로세스의 실행기 정리 (OwnedPool: 어떤 자식도 기다리지 않는다)
    return pool


_SPAWNER: Optional[tuple] = None   # 서버 안: (pid, 일감 큐) — 작업 프로세스를 띄우는 스레드 (_on_lasting_thread; fork 된 자식은 새로 만든다)
_SPAWNER_LOCK = threading.Lock()


def _spawner_loop(q) -> None:
    while True:
        fn, fut = q.get()
        try:
            fut.set_result(fn())
        except BaseException as e:  # noqa: BLE001 - handed to the caller
            fut.set_exception(e)


def _on_lasting_thread(fn):
    """fn() 을 이 프로세스가 사는 동안 끝나지 않는 스레드 (데몬 하나, 처음 부를 때 만든다) 에서 부르고 그 결과를 돌려준다 — 작업 프로세스를
    띄우는 곳 (post_scan_pool). 자식의 PR_SET_PDEATHSIG 는 띄운 스레드가 끝날 때 오므로 그 스레드의 수명 = 서버의 수명이어야 한다."""
    global _SPAWNER
    from concurrent.futures import Future
    with _SPAWNER_LOCK:
        if _SPAWNER is None or _SPAWNER[0] != os.getpid():
            import queue as _queue
            q = _queue.SimpleQueue()
            threading.Thread(target=_spawner_loop, args=(q,), name="post-scan-spawner", daemon=True).start()
            _SPAWNER = (os.getpid(), q)
        q = _SPAWNER[1]
    fut = Future()
    q.put((fn, fut))
    return fut.result()                                # 프로세스 하나를 띄우는 동안 (비참여 스레드 — R-IPC)


def shutdown_post_scan_pool(*, kill: bool = False, pool: Optional[OwnedPool] = None) -> None:
    """작업 프로세스 실행기를 내린다 (pool: 그 실행기 — 기본은 지금 것; 지금 것이면 다음 호출은 새로 만든다). 살아 있으면 하던 일 (미룬 파일
    쓰기) 을 마치고 끝나고, 죽었으면 OwnedPool.shutdown 이 정리한다. kill: 살았지만 응답이 없는 작업 프로세스 (RESULT_TIMEOUT_S) 를 끝낸다."""
    global _POOL
    with _LOCK:
        target = _POOL if pool is None else pool
        if target is not None and _POOL is target:
            _POOL = None
    if target is not None:
        if kill:
            target.kill_children()
        target.shutdown(wait=False, cancel_futures=True)


atexit.register(shutdown_post_scan_pool)


def _warm(feature_workers) -> dict:
    """작업 프로세스에서: 무거운 모듈 import (첫 패스가 기다리지 않게). 특징 추출 병렬은 패스마다 fork 풀 (compute_pass)."""
    t0 = time.perf_counter()
    from scan_core.seam_features import resolve_workers
    from . import autoteach  # noqa: F401
    return {"warm_s": time.perf_counter() - t0, "feature_workers": resolve_workers(feature_workers)}


def warm_post_scan_worker(feature_workers="auto"):
    """패스 전에 (접근·드웰 동안) 작업 프로세스를 데운다. Future 를 돌려준다 (실패해도 패스는 서버 안에서 돈다)."""
    try:
        return post_scan_pool().submit(_warm, feature_workers)
    except Exception:  # noqa: BLE001
        return None


# ---- 프로세스 수명: 작업 프로세스가 fork 하는 계산 프로세스 -----------------------------------------------------------------
def _open_fds() -> set:
    """이 프로세스의 열린 fd 번호 (리눅스 /proc/self/fd; 목록을 못 읽는 OS 면 빈 집합)."""
    for d in ("/proc/self/fd", "/dev/fd"):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        out = set()
        for s in names:
            try:
                fd = int(s)
                os.fstat(fd)                       # 목록을 읽던 디렉터리 fd 는 이미 닫혔다
            except (ValueError, OSError):
                continue
            out.add(fd)
        return out
    return set()


def _fd_ids(fds) -> dict:
    """fd → 그 fd 가 가리키는 열린 파일의 (st_dev, st_ino) (닫힌 fd 는 뺀다)."""
    out = {}
    for fd in fds:
        try:
            s = os.fstat(fd)
        except OSError:
            continue
        out[fd] = (s.st_dev, s.st_ino)
    return out


def _worker_init(parent: int) -> None:
    """작업 프로세스 시작 (spawn, 첫 작업 전): (1) 서버 (parent) 가 죽으면 같이 죽는다 — 계산 프로세스 (_fork_child_setup) 와 같은 원시:
    PR_SET_PDEATHSIG(SIGKILL), 이 설정 전에 서버가 이미 죽었으면 바로 끝낸다 (예전에는 이 프로세스에만 없어 서버가 죽거나 kill 되면 작업
    프로세스가 고아로 남아 메모리·CPU·스트림 디렉터리를 쥐었다 — 쉬는 동안은 호출 파이프의 끝 (EOF) 으로 끝나지만 계산·쓰기 중이거나 멈춘
    (SIGSTOP) 작업 프로세스는 남았다). 띄운 서버 스레드는 서버가 사는 동안 끝나지 않는다 (post_scan_pool). (2) 서버에서 물려받은 fd 를
    기억한다 — spawn 은 넘긴 fd 말고는 닫고 시작하므로 이것이 서버와의 연결 전부다 (실행기 호출·결과 파이프, 이 프로세스의 죽음을 서버에
    알리는 센티널 파이프, 부모 센티널, 자원 추적기). 이 프로세스가 fork 하는 계산 프로세스는 이것을 붙들지 않는다 (_fork_child_setup)."""
    global _SERVER_FDS
    if _PRCTL is not None:
        _PRCTL(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
    if os.getppid() != parent:
        os._exit(1)
    _SERVER_FDS = _fd_ids(_open_fds() - {0, 1, 2})


def _server_fds_now() -> list:
    """(작업 프로세스 안, fork 직전) 서버에서 물려받은 fd 중 지금도 같은 열린 파일을 가리키는 것."""
    now = _fd_ids(_SERVER_FDS)
    return [fd for fd, ident in _SERVER_FDS.items() if now.get(fd) == ident]


def _fork_child_setup(parent: int, detach) -> None:
    """작업 프로세스가 fork 한 계산 프로세스 시작: (1) 부모가 죽으면 같이 죽는다 (리눅스 PR_SET_PDEATHSIG; 이 설정 전에 부모가 이미 죽었으면
    바로 끝낸다 — fork 하는 스레드는 작업 프로세스의 주 스레드라 그 스레드의 끝 = 프로세스의 끝), (2) detach 의 fd 를 놓는다 — 서버와의 연결을
    붙들고 있으면 작업 프로세스가 죽어도 서버 실행기의 센티널이 닫히지 않아 서버가 그 죽음을 모른다."""
    if _PRCTL is not None:
        _PRCTL(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
    if os.getppid() != parent:
        os._exit(1)
    detach_fds(detach)


def _lowest_priority() -> str:
    """이 프로세스를 보통 시분할 (SCHED_OTHER) 의 가장 낮은 우선순위로: nice 를 1 씩 올려 더 오르지 않을 때까지 (범위 상수 없이; 끝에서 OS 가
    멈춘다 — 리눅스 19). 녹화·가드·ROS (nice 0) 보다 아래이고, 다른 배경 일괄 작업 (nice 19 시험 등) 과는 같다 — 'OS 의 가장 낮은 우선순위'
    는 아니다. SCHED_IDLE (권한 없이 가능, 더 낮음) 은 쓰지 않는다: g1_b 서버 역할 재생 (2026-10-09) 에서 SCHED_IDLE 계산 프로세스는 같은
    코드의 nice 19 보다 가드 흉내 지연 p99 를 ~2 배로 늘렸고 (작업자 1, 패스 3, 번갈아 3 회: 1.05–1.33 / 0.49–0.57 ms, 스트림 없음
    0.52–0.64), 배경 nice 19 부하 아래에서 따라가지 못해 이동 뒤로 넘긴 행이 163–194 / 1361 (nice 19: 0–76). → 적용한 것 ('nice N')."""
    try:
        prev, cur = None, os.nice(0)
        while cur != prev:
            prev, cur = cur, os.nice(1)
        return f"nice {cur}"
    except Exception:  # noqa: BLE001 - priority is a courtesy only
        return "unchanged"


def _reap() -> None:
    """작업 프로세스 안: 죽인 스트림 계산 프로세스를 기다리지 않고 거둔다 (아직 안 끝난 것은 다음 기회에)."""
    for pid in list(_REAP):
        try:
            done = os.waitpid(pid, os.WNOHANG)[0]
        except ChildProcessError:
            done = pid
        if done:
            _REAP.discard(pid)


# ---- 패스 중 판 성장 스트림: 전송 (서버 → 작업 프로세스) ---------------------------------------------------------------------
def _frame_iov(rec) -> list:
    """녹화 프레임 → 쓰기 조각 [머리, x 바이트, z 바이트]. 배열이 이미 연속 float32 (센서 프레임) 면 복사하지 않고 그 메모리 그대로 (피클 없음).
    재구성과 같은 값 (scan_core.reconstruct 는 프레임을 float32 로 쌓는다)."""
    x = np.ascontiguousarray(rec.x, dtype=np.float32).reshape(-1)
    z = np.ascontiguousarray(rec.z, dtype=np.float32).reshape(-1)
    return [_HDR.pack(int(rec.result_counter), x.size, z.size), memoryview(x).cast("B"), memoryview(z).cast("B")]


def _advance(iov: list, n: int) -> list:
    """쓰기 조각 목록에서 앞의 n 바이트 (이미 쓴 것) 를 뺀 나머지 (부분 쓰기 이어 쓰기)."""
    for i, b in enumerate(iov):
        m = memoryview(b)
        if n < m.nbytes:
            return ([m[n:]] if n else [b]) + iov[i + 1:]
        n -= m.nbytes
    return []


def _pipe_capacity(fd: int, need: Optional[int] = None) -> int:
    """파이프 용량 [바이트] (모르면 0). need 를 주면 적어도 그만큼으로 늘려 본다 (리눅스 F_SETPIPE_SZ, 사용자 한계 안) — 안 되면 그대로
    (쓰는 쪽은 쓰기 가능을 기다려 이어 쓴다)."""
    try:
        import fcntl
        get_, set_ = getattr(fcntl, "F_GETPIPE_SZ", None), getattr(fcntl, "F_SETPIPE_SZ", None)
        if get_ is None:
            return 0
        cap = int(fcntl.fcntl(fd, get_))
        if need is not None and set_ is not None and int(need) > cap:
            try:
                cap = int(fcntl.fcntl(fd, set_, int(need)))
            except OSError:
                pass
        return cap
    except Exception:  # noqa: BLE001 - capacity is a courtesy only
        return 0


def _remove_stream_dir(d: Optional[str]) -> bool:
    """스트림 디렉터리와 안의 FIFO 를 지운다 — 서버·작업 프로세스 둘 다 부른다 (먼저 하는 쪽이 지우고 나머지는 없음). → 이제 없는가."""
    if not d:
        return True
    for path, rm in ((os.path.join(d, _FIFO_NAME), os.unlink), (d, os.rmdir)):
        try:
            rm(path)
        except OSError:
            pass
    return not os.path.lexists(d)


def _server_stream_dir() -> str:
    """서버 안: 이 패스 스트림의 FIFO 를 둘 디렉터리를 만든다 (만든 쪽이 지울 책임 — _drop_server_dir, 실패 경로·종료 때 _sweep_server_dirs)."""
    d = tempfile.mkdtemp(prefix="fsv1-plate-stream-")
    with _DIRS_LOCK:
        _DIRS.add(d)
    return d


def _drop_server_dir(d: Optional[str]) -> None:
    if d and _remove_stream_dir(d):
        with _DIRS_LOCK:
            _DIRS.discard(d)


def _sweep_server_dirs() -> None:
    """서버 안: 아직 남은 스트림 디렉터리를 지운다 (작업 프로세스가 죽은 뒤·종료 때)."""
    with _DIRS_LOCK:
        ds = list(_DIRS)
    for d in ds:
        _drop_server_dir(d)


atexit.register(_sweep_server_dirs)


class _Wake:
    """깨우기 파이프: 한 스레드가 poll 로 기다리는 동안 다른 스레드·완료 콜백이 깨운다. 닫은 뒤의 깨우기는 아무것도 하지 않는다 (닫힌 fd 번호가
    다른 파일에 다시 쓰여도 거기에 쓰지 않게)."""

    def __init__(self) -> None:
        self.r, self.w = os.pipe()
        os.set_blocking(self.r, False)
        os.set_blocking(self.w, False)
        self._lock = threading.Lock()

    def set(self, *_a) -> None:
        with self._lock:
            if self.w >= 0:
                try:
                    os.write(self.w, b"\0")
                except OSError:          # 가득 = 이미 깨어 있다
                    pass

    def clear(self) -> None:
        try:
            while os.read(self.r, select.PIPE_BUF):
                pass
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            for fd in (self.r, self.w):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            self.r = self.w = -1


# ---- 패스 중 판 성장 스트림 (작업 프로세스 안) -------------------------------------------------------------------------------
def _stream_key(spec, kw) -> bytes:
    """판 성장 입력 (설계·추출 인자) 의 비교 열쇠 — 스트림을 연 값과 패스 뒤 fit_kwargs 가 다르면 스트림 결과를 쓰지 않는다."""
    return pickle.dumps((spec, dict(kw)), protocol=4)


def _seed_row(x32: np.ndarray, z32: np.ndarray) -> tuple:
    """스트림 계산 프로세스 안: 프레임 하나 → (유효 마스크, PlateSeed 또는 None, CPU 시간 [s]). 재구성과 같은 행 — float32 점
    (scan_core.reconstruct), 유효 마스크 (valid_mask, 같은 센서 창), 무효 점 NaN 인 float64 (extract_seam_features_along_scan 이 추출기에
    주는 행) — 에 같은 판 성장 함수."""
    from scan_core.seam_features import plate_seeds_chunk
    from scan_core.validity import valid_mask
    t0 = time.process_time()
    spec, kw, zv = _CHILD
    v = valid_mask(x32, z32, zv)
    X = np.where(v, np.asarray(x32, dtype=float), np.nan)
    Z = np.where(v, np.asarray(z32, dtype=float), np.nan)
    sd = plate_seeds_chunk([X], [Z], spec, kw)[0]
    return v, sd, time.process_time() - t0


def _send_msg(sock: socket.socket, obj) -> None:
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_MSG.pack(len(b)) + b)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytearray]:
    """n 바이트를 다 받을 때까지 (막힌다 — 계산 프로세스 쪽에서만 쓴다). 끝 (EOF) 이면 None."""
    b = bytearray(n)
    v = memoryview(b)
    got = 0
    while got < n:
        k = sock.recv_into(v[got:])
        if k == 0:
            return None
        got += k
    return b


def _seed_child(sock: socket.socket, parent: int, spec, kw: dict, z_valid_mm) -> None:
    """스트림 계산 프로세스 (작업 프로세스가 fork) 의 전부: 부모와 함께 죽고 자기 소켓 말고는 물려받은 fd 를 놓은 뒤 (_fork_child_setup) 가장
    큰 nice 로 (_lowest_priority; 첫 메시지로 적용한 것을 알린다) 프레임 바이트 (머리 + x·z float32) 를 받아 판 성장 → (counter, 유효 마스크,
    PlateSeed, CPU 초, 오류). 작업 프로세스 쪽이 닫히면 (EOF) 끝."""
    global _CHILD
    _fork_child_setup(parent, _open_fds() - {0, 1, 2, sock.fileno()})
    prio = _lowest_priority()
    _CHILD = (spec, dict(kw), tuple(z_valid_mm))
    _send_msg(sock, ("prio", prio))
    while True:
        h = _recv_exact(sock, _HDR.size)
        if h is None:
            return
        c, nx, nz = _HDR.unpack(h)
        d = _recv_exact(sock, 4 * (nx + nz))
        if d is None:
            return
        x = np.frombuffer(d, dtype=np.float32, count=nx)
        z = np.frombuffer(d, dtype=np.float32, count=nz, offset=4 * nx)
        try:
            v, sd, cpu = _seed_row(x, z)
            out = (c, v, sd, cpu, None)
        except Exception as e:  # noqa: BLE001 - reported to the worker, which stops the stream
            out = (c, None, None, 0.0, f"{type(e).__name__}: {e}"[:300])
        _send_msg(sock, out)


def _seed_proc_start(spec, kw: dict, z_valid_mm) -> dict:
    """스트림 계산 프로세스 하나를 fork 한다 (작업 프로세스의 주 스레드에서, 이 스트림의 스레드·FIFO 가 생기기 전). → 작업 프로세스 쪽 상태
    (pid, 논블로킹 소켓, 받은 바이트, 계산 중인 프레임)."""
    a, b = socket.socketpair()
    parent = os.getpid()
    try:
        pid = os.fork()
    except BaseException:
        a.close()
        b.close()
        raise
    if pid == 0:                                        # 자식: 여기서 부모의 코드로 돌아가지 않는다
        code = 1
        try:
            _seed_child(b, parent, spec, kw, z_valid_mm)
            code = 0
        except BaseException:  # noqa: BLE001 - the worker sees EOF and stops the stream
            pass
        finally:
            os._exit(code)
    b.close()
    a.setblocking(False)
    return {"pid": pid, "sock": a, "buf": bytearray(), "out": None, "busy": None, "prio": None, "dead": False}


def stream_open(token: str, spec, kw: dict, z_valid_mm, workers, fifo_dir: Optional[str] = None) -> dict:
    """작업 프로세스 안: 패스 녹화 시작 — 판 성장 스트림을 연다 (남아 있던 스트림은 닫는다). fifo_dir (서버가 만든 디렉터리; 없으면 여기서
    만든다) 안에 FIFO 를 만들어 연다. → {"fifo": 서버가 프레임을 쓸 FIFO 경로, "workers": 계산 프로세스 수, "pid": 이 프로세스, "procs":
    계산 프로세스 pid}. 판 성장 계산은 작업자 수와 무관하게 (1 도) 별도 fork 프로세스에서 가장 큰 nice (리눅스 19) 로 — 이 작업 프로세스의 받기
    스레드는 바이트를 받아 나눠 주기만 한다."""
    global _STREAM
    from scan_core.seam_features import resolve_workers
    stream_close()
    n = resolve_workers(workers)
    st = {"token": str(token), "key": _stream_key(spec, kw), "n": n, "procs": [], "dir": fifo_dir, "fifo": None, "fd": -1,
          "fd_keep": -1, "wake": None, "thr": None, "stop": False, "rows": {}, "received": 0, "frames": 0, "busy_s": 0.0,
          "unprocessed": 0, "abandoned": 0, "error": None}
    _STREAM = st
    try:
        if st["dir"] is None:
            st["dir"] = tempfile.mkdtemp(prefix="fsv1-plate-stream-")
        st["fifo"] = os.path.join(st["dir"], _FIFO_NAME)
        os.mkfifo(st["fifo"], 0o600)                   # 이름만 (fd 없음) — 서버가 이미 디렉터리를 지웠으면 (늦은 시작) 여기서 끝
        # 계산 프로세스를 fork 한다 — 이 스트림의 fd·스레드·깨우기 파이프가 생기기 전 (작업 프로세스는 ROS 스레드가 없는 단일 스레드 →
        # fork 안전; compute_pass 의 특징 풀과 같은 규칙)
        for _ in range(n):
            st["procs"].append(_seed_proc_start(spec, kw, z_valid_mm))
        st["fd"] = os.open(st["fifo"], os.O_RDONLY | os.O_NONBLOCK)
        # 이 프로세스도 쓰는 쪽을 하나 열어 둔다: 서버가 열기 전·닫은 뒤에도 읽기가 '끝 (EOF)' 이 되지 않는다 — 끝은 stop 으로만
        st["fd_keep"] = os.open(st["fifo"], os.O_WRONLY | os.O_NONBLOCK)
        st["wake"] = _Wake()
        st["thr"] = threading.Thread(target=_stream_reader, args=(st,), name="plate-stream-reader", daemon=True)
        st["thr"].start()
    except Exception:
        stream_close()
        raise
    return {"fifo": st["fifo"], "workers": n, "pid": os.getpid(), "procs": [p["pid"] for p in st["procs"]]}


def _read_frames(fd: int, buf: bytearray, chunk: int, pend: deque, st: dict) -> None:
    """논블로킹 FIFO 에서 지금 있는 바이트를 모두 읽어 완전한 프레임을 (counter, 프레임 바이트, x float32, z float32) 로 pend 에 (x·z 는 그
    바이트 위의 읽기 전용 배열; 남은 조각은 다음에 잇는다)."""
    while True:
        try:
            b = os.read(fd, chunk)
        except BlockingIOError:
            break
        if not b:
            break
        buf += b
    o, L, h = 0, len(buf), _HDR.size
    try:
        while L - o >= h:
            c, nx, nz = _HDR.unpack_from(buf, o)
            if nx != nz:
                raise ValueError(f"stream frame {c}: x/z lengths differ ({nx}, {nz})")
            end = o + h + 4 * (nx + nz)
            if end > L:
                break
            raw = bytes(buf[o:end])
            x = np.frombuffer(raw, dtype=np.float32, count=nx, offset=h)
            z = np.frombuffer(raw, dtype=np.float32, count=nz, offset=h + 4 * nx)
            pend.append((int(c), raw, x, z))
            st["received"] += 1
            o = end
    finally:
        del buf[:o]


def _proc_msg(p: dict, msg, st: dict) -> None:
    if msg[0] == "prio":
        p["prio"] = msg[1]
        return
    c, v, sd, cpu, err = msg
    busy, p["busy"] = p["busy"], None
    if err is not None:
        st["error"] = st["error"] or err
    elif busy is None or busy[0] != c:
        st["error"] = st["error"] or f"stream result {c} does not match the frame sent"
    else:
        st["rows"][c] = (busy[1], busy[2], v, sd)
        st["frames"] += 1
        st["busy_s"] += float(cpu)


def _proc_results(p: dict, st: dict) -> None:
    """계산 프로세스 소켓에 지금 와 있는 결과를 기다리지 않고 거둔다 (논블로킹). 그 프로세스가 끝났으면 (EOF) 스트림 오류 (멈춘다)."""
    if p["dead"]:
        return
    eof = False
    while True:
        try:
            b = p["sock"].recv(1 << 16)
        except (BlockingIOError, InterruptedError):
            break
        except OSError:
            b = b""
        if not b:
            eof = True
            break
        p["buf"] += b
    buf, h, o = p["buf"], _MSG.size, 0
    try:
        while len(buf) - o >= h:
            (n,) = _MSG.unpack_from(buf, o)
            if len(buf) - o - h < n:
                break
            msg = pickle.loads(buf[o + h:o + h + n])
            o += h + n
            _proc_msg(p, msg, st)
    finally:
        del buf[:o]
    if eof:
        p["dead"] = True
        st["error"] = st["error"] or f"stream compute process {p['pid']} ended"


def _send_more(p: dict) -> bool:
    """계산 프로세스에 보내던 프레임을 막히지 않는 만큼 더 보낸다 (소켓 버퍼보다 큰 프레임은 여러 번에). → 다 보냈나."""
    out = p["out"]
    while out is not None:
        try:
            k = p["sock"].send(out)
        except (BlockingIOError, InterruptedError):
            return False
        out = out[k:] if k < len(out) else None
        p["out"] = out
    return True


def _dispatch(procs: list, pend: deque, st: dict, poller) -> None:
    """쉬는 계산 프로세스마다 받은 프레임 하나 (계산 중인 것은 프로세스마다 하나뿐). 보내기는 막히지 않는다: 소켓에 다 안 들어가면 남은 바이트는
    그 프로세스가 읽어 자리가 날 때 (POLLOUT) 이어 보낸다 — 멈춘 계산 프로세스가 받기 스레드를 붙잡지 않는다."""
    for p in procs:
        if p["dead"]:
            continue
        if p["out"] is None and p["busy"] is None:
            if not pend or st["error"] is not None:
                continue
            c, raw, x, z = pend.popleft()
            p["busy"], p["out"] = (c, x, z), memoryview(raw)
        if p["out"] is not None:
            done = _send_more(p)
            poller.modify(p["sock"].fileno(), select.POLLIN if done else select.POLLIN | select.POLLOUT)


def _stream_reader(st: dict) -> None:
    """작업 프로세스 안 (스트림 받기 스레드): FIFO 의 프레임을 계산 프로세스에 하나씩 주고 결과를 거둔다 — 막히는 호출 없이 poll 로만 기다린다
    (새 바이트·계산 결과·계산 프로세스의 끝·stop). 받았지만 아직 주지 않은 프레임은 여기 쌓였다가 멈추면 패스 뒤 처리가 처음부터 계산한다."""
    fd, wake, procs = st["fd"], st["wake"], st["procs"]
    buf, pend = bytearray(), deque()
    chunk = _pipe_capacity(fd) or select.PIPE_BUF
    by_fd = {p["sock"].fileno(): p for p in procs}
    try:
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        poller.register(wake.r, select.POLLIN)
        for f in by_fd:
            poller.register(f, select.POLLIN)
        while not st["stop"] and st["error"] is None:
            for f, _e in poller.poll():
                if f in by_fd:
                    _proc_results(by_fd[f], st)
            wake.clear()
            _read_frames(fd, buf, chunk, pend, st)
            _dispatch(procs, pend, st, poller)
    except Exception as e:  # noqa: BLE001 - the stream is an optimisation only
        st["error"] = st["error"] or f"{type(e).__name__}: {e}"[:300]
    finally:
        try:
            if st["error"] is None:
                _read_frames(fd, buf, chunk, pend, st)      # 기록용: 멈출 때까지 받은 프레임 수
        except Exception:  # noqa: BLE001
            pass
        for p in procs:                                     # 이미 와 있는 결과만 (기다리지 않는다)
            try:
                _proc_results(p, st)
            except Exception:  # noqa: BLE001
                pass
        st["unprocessed"] = len(pend)
        st["abandoned"] = sum(1 for p in procs if p["busy"] is not None)


def _kill_procs(st: dict) -> None:
    """스트림 계산 프로세스를 끝낸다 (SIGKILL — 순수 계산이라 정리할 것이 없다; 기다리지 않고 거두기는 _reap)."""
    for p in st.get("procs") or []:
        if p.get("pid"):
            try:
                os.kill(p["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            _REAP.add(p["pid"])
            p["pid"] = 0
        try:
            p["sock"].close()
        except OSError:
            pass
    _reap()


def _stream_finish(st: dict) -> None:
    """받기를 멈춘다 — 아무것도 기다리지 않는다: 받기 스레드는 poll 에서만 기다리므로 깨우면 바로 끝나고, 이미 와 있는 계산 결과만 거둔다.
    계산 중이던 프레임 (가장 낮은 우선순위) 은 버린다 — 그 계산 프로세스를 끝내고, 그 행은 패스 뒤 처리가 보통 우선순위로 처음부터 (같은
    결과). 그 뒤 rows 는 바뀌지 않는다."""
    st["stop"] = True
    if st.get("wake") is not None:
        st["wake"].set()
    th = st.get("thr")
    if th is not None and th is not threading.current_thread():
        th.join()
    _kill_procs(st)


def stream_close() -> None:
    """작업 프로세스 안: 스트림 (받기 스레드·계산 프로세스·FIFO·디렉터리) 을 닫는다 (패스 뒤 처리, 새 스트림, 중단된 패스). 그 뒤 이
    프로세스에는 스트림의 스레드가 없다 (compute_pass 의 fork 풀 전에). 죽인 계산 프로세스는 기다리지 않고 거둔다."""
    global _STREAM
    st, _STREAM = _STREAM, None
    if st is None:
        _reap()
        return
    try:
        _stream_finish(st)
    finally:
        for k in ("fd", "fd_keep"):
            if st.get(k, -1) >= 0:
                try:
                    os.close(st[k])
                except OSError:
                    pass
                st[k] = -1
        if st.get("wake") is not None:
            st["wake"].close()
        _remove_stream_dir(st.get("dir"))
        _reap()


def _stream_seeds(token, res, spec, kw) -> tuple:
    """compute_pass 안: 재구성 행마다 스트림의 판 성장 (PlateSeed 또는 None) — 그 행의 float32 점 (x, z)·유효 마스크가 스트림에 받은 프레임과
    바이트까지 같고 설계·추출 인자가 같을 때만. 먼저 스트림을 멈춘다 (기다리지 않는다). → (seeds 또는 None, 시간 기록 dict)."""
    st = _STREAM
    info: dict = {}
    if token is None:
        return None, info
    if st is None or st["token"] != str(token):
        info["stream_note"] = "no stream for this pass"
        return None, info
    _stream_finish(st)
    prio = sorted({str(p["prio"]) for p in st["procs"] if p.get("prio")})
    info.update(stream_received=int(st["received"]), stream_frames=int(st["frames"]), stream_unprocessed=int(st["unprocessed"]),
                stream_abandoned=int(st["abandoned"]), stream_busy_s=float(st["busy_s"]), stream_priority="/".join(prio) or "unknown")
    if st["error"]:
        info["stream_error"] = st["error"]
    if st["key"] != _stream_key(spec, kw):
        info["stream_note"] = "design/extractor inputs differ from the stream - not used"
        return None, info
    rows = st["rows"]
    N = int(res.x_mm.shape[0])
    seeds: list = [None] * N
    n_used = 0
    for j in range(N):
        e = rows.get(int(res.result_counter[j]))
        if e is None or e[3] is None:
            continue
        a, b, m, sd = e
        xr, zr, vr = res.x_mm[j], res.z_mm[j], res.valid[j]
        if (a.dtype == xr.dtype and b.dtype == zr.dtype and a.shape == xr.shape and b.shape == zr.shape and m.shape == vr.shape
                and a.tobytes() == xr.tobytes() and b.tobytes() == zr.tobytes() and bool(np.array_equal(m, vr))):
            seeds[j] = sd
            n_used += 1
    info["stream_seeded"] = n_used
    info["stream_rows"] = N
    return (seeds if n_used else None), info


class _RawOnly(Exception):
    """원자료만 저장하는 구간 (재구성 없음)."""


def compute_pass(p: dict) -> dict:
    """작업 프로세스 (또는 대체로 서버 안) 에서: raw.npz → 재구성 → QC → recon/ + qc.json → (fit_kwargs 가 있으면) fit_pass.
    executor.save_raw + reconstruct_and_qc + autoteach.fit_pass 와 같은 계산·같은 파일. 반환 dict 는 피클 가능한 것만 (ScanResult 는 돌려주지
    않는다 — 서버는 meta 만 쓴다)."""
    from scan_core.io_npz import save_scan_result
    from scan_core.quality import integrity_report
    from scan_core.reconstruct import reconstruct_scan
    from scan_core.types import QCReport
    global _VIS_SRC
    _VIS_SRC = None                                         # 앞 패스의 남은 재구성 결과는 버린다 (점구름은 이 패스 것만)
    timing: dict = {}
    out = Path(p["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.npz"
    defer = bool(p.get("defer_save"))
    if not defer:
        t_ = time.perf_counter()
        np.savez_compressed(raw_path, **p["arrays"], **{k: np.array(v) for k, v in p["js"].items()})
        timing["save_raw_s"] = time.perf_counter() - t_
    res = None
    try:
        if p.get("reconstruct", True) is False:
            raise _RawOnly()
        frames, robot, job = p["frames"], p["robot"], p["job"]
        if len(frames) < 2 or len(robot) < 2:
            raise ValueError(f"not enough data for reconstruction: frames={len(frames)} robot_samples={len(robot)}")
        t_ = time.perf_counter()
        res = reconstruct_scan(frames, robot, p["calib"], job, fps_nominal=job.profile_fps, z_valid_mm=p["z_valid_mm"],
                               ts_tol_ms=p.get("ts_tol_ms"))
        timing["recon_reconstruct_s"] = time.perf_counter() - t_
        t_ = time.perf_counter()
        qc = integrity_report(res, auto_trigger_error=int(p.get("auto_trigger_error", 0)))
        res.qc = qc
        timing["recon_qc_s"] = time.perf_counter() - t_
    except _RawOnly:
        qc = QCReport(notes=["raw stream only (not reconstructed)"])
        res = None
    except Exception as e:  # noqa: BLE001 - QC must never mask the raw save (executor.reconstruct_and_qc 와 같은 규칙)
        import traceback
        qc = QCReport(verdict="UNJUDGED", failures=[f"reconstruction failed: {e}"], notes=[traceback.format_exc(limit=3)])
        res = None
    recon_dir = str(out / "recon") if res is not None else None
    if defer:
        # 판정에 필요 없는 압축 쓰기 (~2 s/패스) 는 결과를 돌려준 뒤 같은 작업 프로세스가 쓴다 (submit_flush → flush_saves)
        _PENDING.append({"raw_path": str(raw_path), "arrays": p["arrays"], "js": p["js"], "recon_dir": recon_dir, "res": res,
                         "extra_meta": p.get("extra_meta") or {}})
    elif res is not None:
        t_ = time.perf_counter()
        save_scan_result(out / "recon", res, extra_meta=p.get("extra_meta") or {})
        timing["recon_save_recon_s"] = time.perf_counter() - t_
    qc_path = out / "qc.json"
    with open(qc_path, "w", encoding="utf-8") as f:
        json.dump(qc.to_dict(), f, indent=1, ensure_ascii=False, default=str)
    fit_out = None
    try:
        if res is not None and p.get("fit_kwargs") is not None:
            from scan_core.seam_features import resolve_workers
            from .autoteach import fit_pass, pass_feature_inputs
            fp_t: dict = {}
            t_ = time.perf_counter()
            seeds = None
            if p.get("stream_token") is not None:
                try:
                    seeds, s_info = _stream_seeds(p.get("stream_token"), res, *pass_feature_inputs(p["fit_kwargs"]))
                except Exception as e:  # noqa: BLE001 - the stream is an optimisation only: compute every row as before
                    seeds, s_info = None, {"stream_error": f"{type(e).__name__}: {e}"[:300]}
                timing.update(s_info)
                timing["stream_match_s"] = time.perf_counter() - t_
            # 스트림 (받기 스레드·가장 낮은 우선순위 계산 프로세스) 은 여기까지 — 이 패스 것이든 남은 것이든 닫아 이 프로세스에 스레드가 없게
            # 한 뒤 아래 fork 풀 (보통 우선순위) 을 만든다
            stream_close()
            n = resolve_workers(p["fit_kwargs"].get("feature_workers", 1))
            if p.get("in_worker") and n > 1:
                # 작업 프로세스는 ROS 스레드가 없는 단일 스레드 프로세스 → fork 가 안전하고 빠르다 (import 없음). 패스마다 만들고 닫는다:
                # 오래 사는 중첩 spawn 풀은 종료 순간 손자 프로세스가 뜨는 중이면 멈출 수 있었다 (2026-10-05 시험). 자식은 서버와의 연결 fd 를
                # 놓고 이 프로세스와 함께 죽는다 (_fork_child_setup — 작업 프로세스가 죽으면 서버가 바로 안다). 이 프로세스가 주인인 OwnedPool:
                # 자식이 죽으면 그 덩어리는 기다리지 않고 여기서 같은 계산 (seam_features._features_for), with 의 끝은 어떤 자식도 기다리지 않는다
                with OwnedPool(max_workers=n, mp_context=multiprocessing.get_context("fork"), initializer=_fork_child_setup,
                               initargs=(os.getpid(), _server_fds_now())) as ex:
                    fit_out = fit_pass(res, timing=fp_t, feature_executor=ex, plate_seeds=seeds, **p["fit_kwargs"])
            else:
                fit_out = fit_pass(res, timing=fp_t, plate_seeds=seeds, **p["fit_kwargs"])
            timing["fit_pass_s"] = time.perf_counter() - t_
            timing.update({f"fit_{k}": v for k, v in fp_t.items()})
    finally:
        stream_close()                                  # 패스 뒤 처리 끝 = 이 패스 스트림의 끝 (풀·받은 프레임을 놓고 죽인 계산 프로세스를 거둔다)
    # 처리된 패스 점구름 (RViz; 보는 쪽이 있을 때만 cloud_step_m): 판정에 필요 없으니 여기서 만들지 않는다 (패스 결과를 늦추지 않게).
    # 작업 프로세스: 재구성 결과를 남겨 두고 서버가 결과를 받은 뒤 vis_cloud 를 맡긴다 (submit_vis_cloud — 미룬 쓰기보다 먼저).
    # 서버 안 (대체 경로): 같은 프로세스라 참조만 돌려준다 (호출자가 판정 뒤 만든다)
    cloud_pending, vis_src = False, None
    if res is not None and p.get("cloud_step_m"):
        if p.get("in_worker"):
            _VIS_SRC = (res, float(p["cloud_step_m"]))
            cloud_pending = True
        else:
            vis_src = res
    return {"raw_path": str(raw_path), "recon_dir": recon_dir, "qc_path": str(qc_path), "qc": qc, "recon_ok": res is not None,
            "result_meta": dict(res.meta or {}) if res is not None else {}, "timing": timing, "fit": fit_out, "deferred_save": defer,
            "result": res if p.get("return_result") else None, "cloud_pending": cloud_pending, "vis_src": vis_src}


def vis_cloud():
    """작업 프로세스 안: 마지막 패스의 재구성 결과 (실측 자세) → 처리된 패스 점구름 (scan_core.cross_sections.decimated_cloud).
    (점 (n, 3) float32, 만든 시간 s) 또는 None (남겨 둔 패스 없음)."""
    global _VIS_SRC
    src, _VIS_SRC = _VIS_SRC, None
    if src is None:
        return None
    from scan_core.cross_sections import decimated_cloud
    t_ = time.perf_counter()
    P = decimated_cloud(*src)
    return P, time.perf_counter() - t_


def submit_vis_cloud():
    """처리된 패스 점구름을 작업 프로세스에 맡긴다 (기다리지 않는다; 패스 결과를 받은 뒤 — 그 패스의 판정·다음 이동과 나란히). Future 또는 None."""
    try:
        return post_scan_pool().submit(vis_cloud)
    except Exception:  # noqa: BLE001
        return None


def flush_saves() -> list:
    """작업 프로세스 안: 미룬 raw.npz·recon/ 쓰기. [(raw_path, save_raw_s, save_recon_s, error)]."""
    from scan_core.io_npz import save_scan_result
    _reap()
    done = []
    while _PENDING:
        it = _PENDING.pop(0)
        err, t_raw, t_rec = None, float("nan"), float("nan")
        try:
            t_ = time.perf_counter()
            np.savez_compressed(it["raw_path"], **it["arrays"], **{k: np.array(v) for k, v in it["js"].items()})
            t_raw = time.perf_counter() - t_
            if it["res"] is not None and it["recon_dir"]:
                t_ = time.perf_counter()
                save_scan_result(it["recon_dir"], it["res"], extra_meta=it["extra_meta"])
                t_rec = time.perf_counter() - t_
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
        done.append((it["raw_path"], t_raw, t_rec, err))
    return done


def submit_flush():
    """미룬 쓰기를 작업 프로세스에 맡긴다 (기다리지 않는다 — 다음 패스 스캔 동안 쓴다). Future 또는 None."""
    try:
        return post_scan_pool().submit(flush_saves)
    except Exception:  # noqa: BLE001
        return None


class Cancelled(Exception):
    """실행 취소 (액션 취소·서버 종료 — 취소 경로가 cancel Future 를 끝낸다) 가 서버의 기다림에 닿았다: 기다림은 바로 끝나고 그 결과는 더 쓰지
    않는다 (서버 안 다시 계산도 없다). 인자 = 취소 사유."""


def wait_result(fut, cancel=None, limit_s: Optional[float] = None):
    """서버가 작업 프로세스 (또는 서버 쪽 풀) 의 일을 기다리는 규칙 하나 (S4, 2026-10-09): 셋 중 먼저 오는 것에서 끝난다 —
    결과 (예외 포함: 죽은 작업 프로세스는 실행기가 센티널로 바로 알아 BrokenProcessPool), 취소 (cancel = 취소 경로가 끝내는 Future; 여러 대상을
    한 번에 기다리는 concurrent.futures.wait FIRST_COMPLETED — 깨어 보는 주기 없음), 살았지만 멈춘 작업 프로세스의 한계 (limit_s, 기본
    RESULT_TIMEOUT_S). → fut 의 결과 (둘 다 끝났으면 결과가 먼저 — 받은 것은 잃지 않는다) | Cancelled(사유) | concurrent.futures.TimeoutError."""
    import concurrent.futures as _cf
    limit = RESULT_TIMEOUT_S if limit_s is None else float(limit_s)
    done, _ = _cf.wait([fut] if cancel is None else [fut, cancel], timeout=limit, return_when=_cf.FIRST_COMPLETED)
    if fut in done:
        return fut.result()                                # 끝난 Future (기다리지 않는다)
    if cancel is not None and cancel in done:
        why = cancel.exception() or cancel.result()        # 끝난 Future (기다리지 않는다)
        raise Cancelled(str(why or "cancelled"))
    raise _cf.TimeoutError(f"no answer within {limit:.0f} s (alive but stalled)")


def _run_on_worker(fn, payload: dict, local, *, what: str, log=None, cancel=None) -> dict:
    """서버 안: 작업 프로세스에 맡기는 계산의 기다림 — 규칙 하나 (run_post_scan·run_track·run_refit 이 모두 이것; scan_core.proc_pool).
    작업 프로세스가 죽으면 실행기가 그 프로세스의 센티널 (프로세스 상태) 로 바로 알고 (BrokenProcessPool — 그 자식들은 서버와의 연결을
    붙들지 않고 함께 죽는다), RESULT_TIMEOUT_S 는 살아 있지만 멈춘 작업 프로세스의 한계다. 실패는 예외 종류가 아니라 상태로 가른다 (작업
    프로세스 안에서 생겨 피클로 돌아온 BrokenProcessPool·TimeoutError 는 산 작업 프로세스의 계산 예외다): 작업 프로세스가 죽었다 (센티널) →
    실행기를 내린다 (OwnedPool 정리); 살았지만 한계 안에 그 일이 끝나지 않았다 (Future 가 끝나지 않음) → 끝내고 내린다; 살아서 끝난 일이
    예외를 돌려줬다 → 보통으로 내린다 (미룬 파일 쓰기를 마친다). 어느 쪽이든 그다음 local() — 서버 안에서 같은 계산.
    cancel (취소 경로가 끝내는 Future, S4): 기다림은 그것에서도 끝난다 (wait_result) — 그 일이 끝나지 않았으면 한계에서와 같은 규칙으로 작업
    프로세스를 끝내고 (멈췄거나 바쁜 작업 프로세스가 다음 실행을 붙들지 않게) local() 없이 Cancelled 를 올린다.
    → {'out', 'where' ('worker' | 'in_process'), 'worker_error', 'worker_state' ('died' | 'stalled' | 'alive' — 실패했을 때), 'pool_note'}."""
    pool, fut = None, None
    try:
        pool = post_scan_pool()
        fut = pool.submit(fn, payload)
        out = wait_result(fut, cancel)
        return {"out": out, "where": "worker", "worker_error": None, "worker_state": None, "pool_note": pool.rule_note}
    except Cancelled as e:
        if fut is not None and not fut.done():
            shutdown_post_scan_pool(kill=True, pool=pool)
            _sweep_server_dirs()
        if log is not None:
            log(f"{what}: cancelled while waiting for the worker ({e})")
        raise
    except Exception as e:  # noqa: BLE001 - a broken worker must never lose the computation
        died = pool is not None and pool.child_died()
        stalled = not died and fut is not None and not fut.done()
        state = "died" if died else "stalled" if stalled else "alive"
        shutdown_post_scan_pool(kill=stalled, pool=pool)
        _sweep_server_dirs()                            # 죽은 작업 프로세스의 스트림 FIFO 가 남았으면
        err = f"{type(e).__name__}: {e}"[:300]
        if log is not None:
            log(f"{what} worker {state} ({err}) - computing in the server process")
        return {"out": local(), "where": "in_process", "worker_error": err, "worker_state": state,
                "pool_note": pool.rule_note if pool is not None else None}


def run_post_scan(p: dict, *, use_worker: bool = True, log=None, cancel=None) -> dict:
    """compute_pass 를 작업 프로세스에서 (기다리는 동안 GIL 을 놓는다) — 기다림·실패의 규칙은 _run_on_worker (작업 프로세스에 맡기는 계산이
    모두 같은 규칙). 반환 dict 의 'where' = 'worker' | 'in_process' (+ 'worker_error', 'worker_state' = 작업 프로세스 자신의 상태로 가른
    'died' | 'stalled' | 'alive')."""
    t0 = time.perf_counter()
    if use_worker:
        r = _run_on_worker(compute_pass, dict(p, defer_save=True, in_worker=True), lambda: compute_pass(p), what="post-scan", log=log,
                           cancel=cancel)
        out = r["out"]
        out["where"] = r["where"]
        if r["where"] != "worker":
            out["worker_error"], out["worker_state"] = r["worker_error"], r["worker_state"]
        out["timing"]["post_scan_wall_s"] = time.perf_counter() - t0
        if r["pool_note"]:                              # 작업 프로세스 실행기가 수명 규칙 일부 없이 돈다 (표준 라이브러리 내부가 다름)
            out["timing"]["post_scan_pool_note"] = r["pool_note"]
        return out
    out = compute_pass(p)
    out["where"] = "in_process"
    out["timing"]["post_scan_wall_s"] = time.perf_counter() - t0
    return out


def _executor_threads(pool) -> Optional[list]:
    """실행기의 서버 쪽 스레드 (관리 스레드·호출 큐 피더 스레드) — 표준 라이브러리 내부 속성이라 이 파이썬에 없으면 None (측정 못 함)."""
    try:
        ts = (pool._executor_manager_thread, pool._call_queue._thread)
    except AttributeError:
        return None
    return [t for t in ts if isinstance(t, threading.Thread) and t.ident is not None]


def _thread_cpu(t: Optional[threading.Thread]) -> float:
    """다른 스레드의 CPU 시간 (사용자+커널) [s] (살아 있을 때만; 못 읽으면 NaN)."""
    if t is None or t.ident is None or not t.is_alive():
        return float("nan")
    try:
        return float(time.clock_gettime(time.pthread_getcpuclockid(t.ident)))
    except Exception:  # noqa: BLE001
        return float("nan")


class SeedFeeder:
    """서버 쪽: 패스 녹화 중 새 프레임을 작업 프로세스의 판 성장 스트림에 넘기는 스레드 (데몬) 하나.

    녹화 스레드·센서 가드는 건드리지 않는다 — 녹화 목록 (SensorSession.records) 을 읽기만 한다. 서버 쪽 비용은 설계상 묶여 있다: 작업
    프로세스와의 실행기 왕복은 패스마다 한 번 (stream_open — 서버가 만든 디렉터리에 작업 프로세스가 FIFO 를 만들어 연다), 프레임은 그 FIFO 에
    피클 없이 배열 메모리 그대로 writev (커널 복사 한 번) — 응답·완료 콜백이 없어 서버에서 스트림이 하는 일은 모두 이 스레드 안이다.
    쓰는 주기는 받는 쪽에서 정한다: 파이프 한 통 (용량에 들어가는 프레임 수 — OS 기본 용량, 계산 프로세스마다 한 프레임이 안 들어가면 그만큼
    늘린다) 이 녹화되는 시간 (그 프레임 수 / 센서 프레임률). 깨면 쌓인 프레임을 한 번에 쓰므로 깨는 횟수는 패스 바이트 / 파이프 용량 정도로
    작업자 수·프레임 수와 따로 묶인다. 파이프가 차면 (작업 프로세스가 못 따라가면) 쓰기 가능을 기다리고 그동안 프레임은 녹화 목록에 남는다
    (서버에 쌓이는 것 없음). 멈추면 (stop) 그때까지 녹화된 프레임을 기다리지 않고 한 번 더 쓰고 끝난다 — 못 넣은 프레임은 패스 뒤 처리가
    처음부터. 어떤 실패도 녹화·이동에 올라가지 않는다 (먹고 멈춘다). FIFO 의 이름은 양쪽이 연 뒤 바로 지운다 (어떻게 끝나도 남지 않게).

    보고 (stop): 잰 것만 — stream_feeder_cpu_s = 이 스레드의 CPU (사용자+커널: writev 의 커널 복사 포함), stream_calls_cpu_s = 시작·멈춤
    호출, stream_executor_cpu_s = 시작부터 멈춤까지 실행기 스레드의 CPU (그동안 그 스레드가 한 다른 일까지 넣은 상한; 그 스레드를 못 읽으면
    NaN), stream_server_cpu_s = 셋의 합 (하나라도 NaN 이면 NaN — 재지 못한 것을 0 으로 치지 않는다)."""

    def __init__(self, spec, kw: dict, *, z_valid_mm, workers, fps: float):
        from scan_core.seam_features import resolve_workers
        self.token = f"{os.getpid()}-{time.monotonic_ns()}"
        self.spec, self.kw, self.z_valid_mm, self.workers = spec, dict(kw), tuple(z_valid_mm), workers
        self.n = resolve_workers(workers)               # 스트림 계산 프로세스 수 (작업 프로세스의 stream_open 과 같은 유도)
        f = float(fps) if fps is not None else float("nan")
        self.fps = f if (np.isfinite(f) and f > 0.0) else float("nan")
        self.pipe_bytes = 0                             # FIFO 용량 [바이트] (첫 프레임을 쓸 때; 0 = 모름)
        self.pipe_frames = 0                            # 그 용량에 들어가는 프레임 수 (용량을 모르면 계산 프로세스 수)
        self.period_s = float("nan")                    # 쓰는 주기 = pipe_frames / fps (첫 프레임 전에는 프레임 주기로 기다린다)
        self.started = False
        self.closed_by_worker = False
        self.frames = 0                                 # 끝까지 쓴 프레임
        self.writes = 0                                 # writev 호출
        self.wakeups = 0                                # 피더가 깬 횟수 (poll 이 돌아온 횟수)
        self.round_trips = 0                            # 작업 프로세스 실행기 왕복 (stream_open, 중단이면 + stream_close)
        self.error: Optional[str] = None
        self.cpu_s = 0.0
        self._caller_cpu_s = 0.0
        self._ex_cpu0: Optional[dict] = None            # 실행기 스레드 → 시작 때 CPU (None = 측정 못 함)
        self._pool = None
        self._dir: Optional[str] = None
        self._stop = threading.Event()
        self._wake: Optional[_Wake] = None
        self._open = None
        self._cancelled = False
        self._thr: Optional[threading.Thread] = None
        self._records: Optional[list] = None

    def start(self, records: list) -> bool:
        """녹화가 시작된 뒤 (records = 녹화 스레드가 채우는 목록) 한 번. 스트림을 열고 스레드를 띄운다 (기다리지 않는다). 실패해도 False 만."""
        if self.started or not np.isfinite(self.fps):
            return False
        t0 = time.thread_time()
        try:
            self._pool = post_scan_pool()
            self._records = records
            self._wake = _Wake()
            self._dir = _server_stream_dir()
            before = _executor_threads(self._pool)
            cpu0 = {t: _thread_cpu(t) for t in before} if before is not None else None
            self._open = self._pool.submit(stream_open, self.token, self.spec, self.kw, self.z_valid_mm, self.workers, self._dir)
            self.round_trips += 1
            after = _executor_threads(self._pool)
            if cpu0 is not None and after is not None:
                for t in after:
                    cpu0.setdefault(t, 0.0)             # 이 호출이 띄운 스레드: 그 CPU 전부가 창 안
                self._ex_cpu0 = cpu0
            self._open.add_done_callback(self._wake.set)
            self._thr = threading.Thread(target=self._run, name="post-scan-seed-feeder", daemon=True)
            self._thr.start()
            self.started = True
        except Exception as e:  # noqa: BLE001 - never disturb the motion
            self.error = f"{type(e).__name__}: {e}"[:300]
            if self._thr is None:
                if self._wake is not None:
                    self._wake.close()
                _drop_server_dir(self._dir)
        finally:
            self._caller_cpu_s += time.thread_time() - t0
        return self.started

    def _cycle_s(self) -> float:
        """한 주기 [s]: 쓰는 주기 (첫 프레임 전에는 프레임 주기)."""
        return float(self.period_s) if np.isfinite(self.period_s) else 1.0 / self.fps

    def _run(self) -> None:
        t_cpu0 = time.thread_time()
        fd = -1
        wake = self._wake
        try:
            try:   # 읽는 쪽이 닫히면 이 스레드의 쓰기는 EPIPE 로만 끝난다 (프로세스에 SIGPIPE 가 가지 않게 — 파이썬은 원래 무시하지만)
                signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGPIPE})
            except Exception:  # noqa: BLE001
                pass
            poller = select.poll()
            poller.register(wake.r, select.POLLIN)
            while not self._open.done():                 # 작업 프로세스가 (앞 패스의 미룬 파일 쓰기 뒤) 스트림을 열 때까지 — stop 이 깨운다
                if self._stop.is_set():
                    return
                poller.poll()
                self.wakeups += 1
                wake.clear()
            if self._stop.is_set():
                return
            fd = os.open(self._open.result()["fifo"], os.O_WRONLY | os.O_NONBLOCK)
            _drop_server_dir(self._dir)                  # 양쪽이 열었다 — 이름은 더 필요 없다 (어느 쪽이 어떻게 끝나도 남지 않게)
            poller.register(fd, 0)
            w = {"i": 0, "iov": [], "n": 0}
            while not self._stop.is_set():
                if self._write(fd, w):                   # 한 번에 못 넣은 프레임이 더 있다 (따라잡기)
                    continue
                if w["iov"]:                             # 파이프가 찼다: 작업 프로세스가 읽을 때까지 (또는 stop)
                    poller.modify(fd, select.POLLOUT)
                    ev = poller.poll()
                else:                                    # 파이프 한 통 분량이 녹화될 때까지 (또는 stop)
                    poller.modify(fd, 0)
                    ev = poller.poll(max(1, int(np.ceil(self._cycle_s() * 1e3))))
                self.wakeups += 1
                if any(f_ == fd and e_ & (select.POLLERR | select.POLLHUP) for f_, e_ in ev):
                    raise BrokenPipeError("the post-scan worker closed the stream")
                wake.clear()
            while self._write(fd, w):                    # stop: 그때까지 녹화된 프레임을 기다리지 않고 들어가는 만큼 한 번 더
                pass
        except BrokenPipeError:
            self.closed_by_worker = True                  # 작업 프로세스가 스트림을 닫았다 (패스 뒤 처리 시작·중단·작업 프로세스의 끝) — 오류 아님
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"[:300]
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            wake.close()
            _drop_server_dir(self._dir)
            self.cpu_s = time.thread_time() - t_cpu0

    def _write(self, fd: int, w: dict) -> bool:
        """쌓인 프레임을 writev 한 번 (기다리지 않는다; 파이프가 차면 남은 조각은 w 에 두고 다음에 잇는다). → 다 쓰고도 녹화 목록에 프레임이
        더 있으면 True."""
        recs = self._records
        if not w["iov"]:
            j = min(len(recs), w["i"] + _IOV_MAX // 3)  # writev 한 번에 넣을 수 있는 프레임 (OS 조각 한계)
            if j <= w["i"]:
                return False
            for r in recs[w["i"]:j]:
                w["iov"] += _frame_iov(r)
            w["n"], w["i"] = j - w["i"], j
            if not self.pipe_frames:                     # 첫 프레임: 파이프 한 통 = 몇 프레임인가 → 쓰는 주기
                fb = sum(memoryview(b).nbytes for b in w["iov"][:3])
                self.pipe_bytes = _pipe_capacity(fd, self.n * fb)   # 계산 프로세스마다 한 프레임은 들어가게 (OS 가 허락하는 만큼)
                self.pipe_frames = max(1, self.pipe_bytes // fb) if self.pipe_bytes > 0 else self.n
                self.period_s = self.pipe_frames / self.fps
        try:
            k = os.writev(fd, w["iov"])
        except BlockingIOError:
            return False
        self.writes += 1
        w["iov"] = _advance(w["iov"], k)
        if w["iov"]:
            return False
        self.frames += w["n"]
        return len(recs) > w["i"]

    def _executor_cpu(self) -> float:
        """시작부터 지금까지 실행기 스레드의 CPU [s] — 실행기에 보낸 것이 없으면 0, 못 재면 (내부 속성 없음·창 안에서 끝난 스레드·시계 못
        읽음) NaN."""
        if self._open is None:
            return 0.0
        if self._ex_cpu0 is None:
            return float("nan")
        now = _executor_threads(self._pool)
        if now is None:
            return float("nan")
        total = 0.0
        for t in set(self._ex_cpu0) | set(now):
            total += _thread_cpu(t) - self._ex_cpu0.get(t, 0.0)   # 끝난 스레드 → NaN (그 CPU 는 잴 수 없다)
        return total

    def stop(self) -> dict:
        """녹화가 끝난 뒤: 피더가 그때까지 녹화된 프레임을 기다리지 않고 (파이프에 들어가는 만큼) 한 번 더 쓰고 끝난다 — 막히는 호출이 없어
        깨우면 한 바퀴 안에 끝나므로, 이동 뒤 처리를 붙잡지 않게 한 주기까지만 기다린다. 남은 프레임은 패스 뒤 처리가 처음부터.
        → 시간 기록 dict (쓴 프레임·writev·깬 횟수·왕복, 파이프 한 통·쓰는 주기, 잰 서버 CPU, 오류)."""
        t0 = time.thread_time()
        self._stop.set()
        if self._wake is not None:
            self._wake.set()
        running = False
        feeder = float(self.cpu_s)
        if self._thr is not None:
            self._thr.join(self._cycle_s())
            running = self._thr.is_alive()
            feeder = _thread_cpu(self._thr) if running else float(self.cpu_s)
        ex = self._executor_cpu()
        self._caller_cpu_s += time.thread_time() - t0
        calls = float(self._caller_cpu_s)
        out = {"stream_sent_frames": int(self.frames), "stream_sent_writes": int(self.writes), "stream_feeder_wakeups": int(self.wakeups),
               "stream_round_trips": int(self.round_trips), "stream_pipe_frames": int(self.pipe_frames),
               "stream_write_period_s": float(self.period_s), "stream_feeder_cpu_s": feeder, "stream_calls_cpu_s": calls,
               "stream_executor_cpu_s": ex, "stream_server_cpu_s": feeder + calls + ex}
        if running:
            out["stream_feeder_running"] = True
        if self.error:
            out["stream_feeder_error"] = self.error
        return out

    def cancel(self) -> None:
        """중단된 패스: 쓰기를 멈추고 작업 프로세스의 스트림 (계산 프로세스·받은 프레임·FIFO) 을 닫는다 (한 번만)."""
        self._stop.set()
        if self._wake is not None:
            self._wake.set()
        if self.started and not self._cancelled:
            self._cancelled = True
            try:
                post_scan_pool().submit(stream_close)
                self.round_trips += 1
            except Exception:  # noqa: BLE001
                pass



# ---- S 추적·트랜싯 특징 스트림 (P2, 2026-10-09) ---------------------------------------------------------------------------------
# S_track (S 첫 드웰 뒤 안쪽으로 짧게)·트랜싯 (S→E) 은 툴 자세를 유지하는 평행이동이라, 녹화되는 프레임의 그루브 특징 (패스와 같은 추출기·같은
# 인자, depth_scale 은 이동 내내 같다 — 자세 회전 없음) 을 이동 중에 계산할 수 있다. FeatureFeeder (서버 스레드) 가 녹화 목록을 읽기만 해서
# 이 작업 프로세스로 넘기고, 이동 뒤 compute_track 이 재구성 (같은 시계 동기·dt_s) 의 행마다 그 특징 (같은 float32 점·유효 마스크일 때만) 을
# 쓴다 — 스트림이 따라가지 못한 행 (피더 보폭은 부하에 따라 달라진다) 은 이동 뒤에 **모두** (보폭 1) 계산한다: 맞춤에 쓰는 행 집합은 늘
# 재구성의 모든 행이라 결과가 부하와 무관하다 (스트림은 지연만 줄인다). 맞춤 (scan_core.seam_track.fit_track) 도 작업 프로세스에서 (서버
# GIL 회피).
# 풀은 다른 모든 것과 같은 수명 규칙 (scan_core.proc_pool.OwnedPool — 1g-integ 2026-10-09): 스트림 계산 풀과 이동 뒤 특징 풀은 이 작업
# 프로세스가 주인인 fork OwnedPool (자식은 서버와의 연결을 놓고 부모와 함께 죽는다 — _fork_child_setup), 풀 결과의 기다림은 RESULT_TIMEOUT_S
# 안 (wait_within — 그 안에 답이 없는 풀의 자식은 끝낸다), 서버의 기다림은 다른 작업 프로세스 계산과 같은 _run_on_worker.
_FSTREAM: Optional[dict] = None


def _stream_rows(st: dict, xs, zs) -> tuple:
    """프레임 (x, z) → 재구성과 같은 행: float32 (scan_core.reconstruct._stack_frames), 유효 마스크 (valid_mask, 같은 센서 창), 무효 점 NaN
    인 float64 (extract_seam_features_along_scan 이 추출기에 주는 행)."""
    from scan_core.validity import valid_mask
    x32 = [np.asarray(x, dtype=np.float32).reshape(-1) for x in xs]
    z32 = [np.asarray(z, dtype=np.float32).reshape(-1) for z in zs]
    v = [valid_mask(a, b, st["z_valid_mm"]) for a, b in zip(x32, z32)]
    X = [np.where(m, np.asarray(a, dtype=float), np.nan) for a, m in zip(x32, v)]
    Z = [np.where(m, np.asarray(b, dtype=float), np.nan) for b, m in zip(z32, v)]
    return x32, z32, v, X, Z


def _track_features_chunk(X, Z, spec, kw, depth_scale) -> list:
    """프로파일 묶음 → CrossSectionFeatures 목록 (패스와 같은 추출기; depth_scale = 이동 내내 같은 레이저 평면 기울기 보정). 이동 뒤 계산
    (seam_features._features_for 의 DS = 같은 값의 배열) 과 같은 호출이다."""
    from scan_core.seam_features import extract_cross_section_features
    return [extract_cross_section_features(X[j], Z[j], spec, depth_scale=float(depth_scale), **kw) for j in range(len(X))]


def _features_rows(X, Z, spec, kw, depth_scale, workers, *, timing: dict) -> list:
    """이동 뒤 남은 행의 특징 (같은 순수 함수): 작업자 수 (seam_features.resolve_workers) 가 1 보다 크면 이 프로세스가 주인인 fork OwnedPool
    (임계 경로 — 보통 우선순위; 자식은 서버와의 연결을 놓고 함께 죽는다), 나누기·기다림은 패스의 특징 단계와 같은 함수
    (seam_features._features_for: RESULT_TIMEOUT_S 안에 돌려받지 못한 덩어리는 여기서 같은 계산, 멈춘 풀의 자식은 끝낸다); 아니면 이
    프로세스에서. 순서·값은 작업자 수와 무관하다. 실행 기록은 timing 의 track_features_* (패스의 fit_features_* 와 같은 항목)."""
    from scan_core import seam_features as _sf
    n = _sf.resolve_workers(workers) if workers is not None else 1
    DS = np.full(len(X), float(depth_scale))
    if n > 1:
        with OwnedPool(max_workers=n, mp_context=multiprocessing.get_context("fork"), initializer=_fork_child_setup,
                       initargs=(os.getpid(), _server_fds_now())) as ex:
            out = _sf._features_for(X, Z, spec, kw, n, executor=ex, DS=DS)
    else:
        out = _sf._features_for(X, Z, spec, kw, 1, DS=DS)
    run = _sf.FEATURE_RUN
    timing["track_features_mode"], timing["track_features_workers"] = run.get("mode"), run.get("workers")
    if run.get("error"):
        timing["track_features_pool_error"] = run.get("error")
    if run.get("note"):                                 # 프로세스 풀이 수명 규칙 일부 없이 돌았다 (scan_core.proc_pool)
        timing["track_features_pool_note"] = run.get("note")
    return out


def _stream_child_setup(parent: int, detach) -> None:
    """추적 특징 스트림 계산 프로세스 시작 (작업 프로세스의 fork): 서버와의 연결을 놓고 부모와 함께 죽는다 (_fork_child_setup — 이동 뒤
    특징 풀과 같은 규칙), 그다음 가장 낮은 우선순위 (_lowest_priority — 녹화·가드·ROS 아래; 판 성장 스트림과 같다)."""
    _fork_child_setup(parent, detach)
    _lowest_priority()


def feature_stream_open(token: str, spec, kw: dict, z_valid_mm, workers, depth_scale: float) -> dict:
    """작업 프로세스 안: 추적 특징 스트림을 연다 (남아 있던 것은 닫는다)."""
    global _FSTREAM
    from scan_core.seam_features import resolve_workers
    feature_stream_close()
    _FSTREAM = {"token": str(token), "spec": spec, "kw": dict(kw), "z_valid_mm": tuple(z_valid_mm), "n": resolve_workers(workers),
                "ds": float(depth_scale), "ex": None, "rows": {}, "frames": 0, "batches": 0, "busy_s": 0.0, "error": None, "note": None}
    return {"workers": _FSTREAM["n"]}


def feature_stream_batch(token: str, counters, xs, zs) -> tuple:
    """작업 프로세스 안: 프레임 묶음 → 특징 (작업자 수만큼 스트림 계산 풀에 나눠). → (계산한 프레임 수, 벽시계 [s]) — 피더가 보폭을 정한다.
    계산 풀 = 이 프로세스가 주인인 fork OwnedPool (가장 낮은 우선순위, 서버와의 연결을 놓고 함께 죽는다 — _stream_child_setup); 기다림은
    RESULT_TIMEOUT_S 안 (wait_within — 죽은 자식은 그 풀의 Future 가 바로 끝나 기다리지 않는다). 실패 (죽은 자식, 한계 안에 답이 없어 자식을
    끝낸 풀, 계산 예외) 는 스트림을 멈출 뿐 — 그 행은 이동 뒤에 계산한다."""
    st = _FSTREAM
    if st is None or st["token"] != str(token) or st["error"] is not None or not len(counters):
        return 0, 0.0
    t0 = time.perf_counter()
    try:
        x32, z32, v, X, Z = _stream_rows(st, xs, zs)
        B, n = len(X), int(st["n"])
        if n > 1 and B > 1:
            if st["ex"] is None:
                st["ex"] = OwnedPool(max_workers=n, mp_context=multiprocessing.get_context("fork"), initializer=_stream_child_setup,
                                     initargs=(os.getpid(), _server_fds_now()))
                st["note"] = st["ex"].rule_note
            bounds = np.linspace(0, B, min(B, n) + 1).astype(int)
            futs = [st["ex"].submit(_track_features_chunk, X[a:b], Z[a:b], st["spec"], st["kw"], st["ds"])
                    for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
            _done, stalled = _proc_pool.wait_within(futs, _proc_pool.RESULT_TIMEOUT_S)
            if stalled:                                 # 살았지만 한계 안에 답이 없는 풀 — 그 자식을 끝낸다 (주인의 정리가 기다리지 않게)
                st["ex"].kill_children()
                raise TimeoutError(f"no answer from the stream pool within {_proc_pool.RESULT_TIMEOUT_S:.0f} s (alive but stalled)")
            feats: list = []
            for f in futs:
                feats.extend(f.result())                # 이미 끝난 Future
        else:
            feats = _track_features_chunk(X, Z, st["spec"], st["kw"], st["ds"])
        for c, a, b, m, f in zip(counters, x32, z32, v, feats):
            st["rows"][int(c)] = (a, b, m, f)
        st["frames"] += B
        st["batches"] += 1
        return B, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - optimisation only: the rows are computed after the move
        st["error"] = f"{type(e).__name__}: {e}"[:300]
        return 0, 0.0
    finally:
        st["busy_s"] += time.perf_counter() - t0


def feature_stream_close() -> None:
    global _FSTREAM
    st, _FSTREAM = _FSTREAM, None
    if st is not None and st.get("ex") is not None:
        try:
            st["ex"].shutdown(wait=True)
        except Exception:  # noqa: BLE001
            pass


def compute_track(p: dict) -> dict:
    """작업 프로세스 (또는 대체로 서버 안) 에서: S_track/트랜싯 원자료 저장 (미룸) → 재구성 (시계 동기·dt_s, reconstruct_scan) → 행마다 특징
    (스트림 결과, 없는 행은 이동 뒤 계산 — 모든 행; p['workers'] > 1 이면 fork OwnedPool 에 나눠)
    → TrackRows → (p['fit'] 이 있으면) fit_track (앞 이동의 행 p['prior'] 와 이어서). p['stride'] (피더 보폭) 는 기록만 — 행 집합을 바꾸지
    않는다 (부하와 무관한 결과). 반환 {rows, est, outcome, info, timing, raw_path, deferred_save} (피클 가능)."""
    from scan_core.reconstruct import reconstruct_scan
    from scan_core.seam_track import concat_rows, fit_track, moving_mask, rows_from_features
    timing: dict = {}
    t_all = time.perf_counter()
    out = Path(p["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.npz"
    defer = bool(p.get("defer_save"))
    try:
        if defer:
            _PENDING.append({"raw_path": str(raw_path), "arrays": p["arrays"], "js": p["js"], "recon_dir": None, "res": None,
                             "extra_meta": {}})
        else:
            t_ = time.perf_counter()
            np.savez_compressed(raw_path, **p["arrays"], **{k: np.array(v) for k, v in p["js"].items()})
            timing["save_raw_s"] = time.perf_counter() - t_
        frames, robot, calib = p["frames"], p["robot"], p["calib"]
        if len(frames) < 2 or len(robot) < 2:
            raise ValueError(f"not enough data for the track: frames={len(frames)} robot_samples={len(robot)}")
        t_ = time.perf_counter()
        res = reconstruct_scan(frames, robot, calib, None, fps_nominal=float(p["fps"]), z_valid_mm=p["z_valid_mm"])
        timing["track_reconstruct_s"] = time.perf_counter() - t_
        N = int(res.x_mm.shape[0])
        X = np.where(res.valid, np.asarray(res.x_mm, dtype=float), np.nan)
        Z = np.where(res.valid, np.asarray(res.z_mm, dtype=float), np.nan)
        moving, _prog = moving_mask(res.T_base_opt[:, :3, 3], float(p["still_tol_m"]))
        stride = max(1, int(p.get("stride") or 1))
        st = _FSTREAM if (p.get("stream_token") is not None and _FSTREAM is not None and _FSTREAM["token"] == str(p["stream_token"])) else None
        feats: list = [None] * N
        n_stream = 0
        t_ = time.perf_counter()
        for j in range(N):
            e = st["rows"].get(int(res.result_counter[j])) if st is not None else None
            if e is not None:
                a, b, m, f = e
                if (a.shape == res.x_mm[j].shape and a.tobytes() == res.x_mm[j].tobytes() and b.tobytes() == res.z_mm[j].tobytes()
                        and bool(np.array_equal(m, res.valid[j]))):
                    feats[j] = f
                    n_stream += 1
        timing["track_stream_match_s"] = time.perf_counter() - t_
        if st is not None:
            timing.update(track_stream_frames=int(st["frames"]), track_stream_busy_s=float(st["busy_s"]))
            if st["error"]:
                timing["track_stream_error"] = st["error"]
            if st["note"]:
                timing["track_stream_pool_note"] = st["note"]
        # 스트림은 여기까지 (행을 다 맞춰 봤다): 닫아 이 프로세스에 스트림 풀의 스레드·자식이 없게 한 뒤 아래 fork 풀을 만든다 (compute_pass 와
        # 같은 순서)
        feature_stream_close()
        todo = [j for j in range(N) if feats[j] is None]          # 스트림이 주지 않은 행 전부 (보폭 1)
        t_ = time.perf_counter()
        if todo:
            for j, f in zip(todo, _features_rows(X[todo], Z[todo], p["spec"], p.get("kw") or {}, float(p.get("depth_scale", 1.0)),
                                                 p.get("workers"), timing=timing)):
                feats[j] = f
        timing["track_features_after_s"] = time.perf_counter() - t_
        timing.update(track_rows=N, track_from_stream=n_stream, track_computed_after=len(todo), track_feeder_stride=stride)
        rows = rows_from_features(feats, res.T_base_flange, X, Z, moving, p["spec"], part=int(p.get("part", 0)),
                                  counter=res.result_counter)
        est, outcome, info = None, "not_fitted", {}
        fit = p.get("fit")
        if fit is not None:
            t_ = time.perf_counter()
            allr = concat_rows([p.get("prior"), rows]) if p.get("prior") is not None else rows
            est, outcome, info = fit_track(allr, calib, **fit)
            timing["track_fit_s"] = time.perf_counter() - t_
        timing["track_total_s"] = time.perf_counter() - t_all
        return {"rows": rows, "est": est, "outcome": outcome, "info": info, "timing": timing, "raw_path": str(raw_path),
                "deferred_save": defer, "error": None}
    finally:
        feature_stream_close()


def run_track(p: dict, *, use_worker: bool = True, log=None, cancel=None) -> dict:
    """compute_track 를 작업 프로세스에서 — 기다림·실패의 규칙은 다른 작업 프로세스 계산과 같다 (_run_on_worker: 죽음은 프로세스 상태로 바로,
    RESULT_TIMEOUT_S 는 살았지만 멈춘 작업 프로세스의 한계, 실패하면 서버 안에서 같은 계산 — 서버 안은 스트림 없이 순차: ROS 스레드가 있는
    프로세스에서 fork 하지 않는다). 반환에 'where' (+ 'worker_error', 'worker_state'), timing 에 track_wall_s."""
    t0 = time.perf_counter()

    def local() -> dict:
        return compute_track(dict(p, stream_token=None, workers=None))     # 서버 안: fork 하지 않는다 (ROS 스레드)
    if use_worker:
        r = _run_on_worker(compute_track, dict(p, defer_save=True), local, what="track", log=log, cancel=cancel)
        out = r["out"]
        out["where"] = r["where"]
        if r["where"] != "worker":
            out["worker_error"], out["worker_state"] = r["worker_error"], r["worker_state"]
        if r["pool_note"]:
            out["timing"]["track_pool_note"] = r["pool_note"]
        out["timing"]["track_wall_s"] = time.perf_counter() - t0
        return out
    out = local()
    out["where"] = "in_process"
    out["timing"]["track_wall_s"] = time.perf_counter() - t0
    return out


def refit_track(rows_list: list, calib, fit: dict) -> tuple:
    """행 (센서 프레임 값) 을 다른 캘리브 (광학 분기 뒤집기) 또는 이어 붙인 이동 (S_track + S_track2) 으로 다시 맞춘다 → (est, outcome, info)."""
    from scan_core.seam_track import concat_rows, fit_track
    return fit_track(concat_rows(list(rows_list)), calib, **fit)


def _refit_job(p: dict) -> tuple:
    """작업 프로세스 안: refit_track (_run_on_worker 가 넘기는 dict 하나)."""
    return refit_track(p["rows"], p["calib"], p["fit"])


def run_refit(rows_list: list, calib, fit: dict, *, use_worker: bool = True, log=None, meta: Optional[dict] = None,
              cancel=None) -> tuple:
    """refit_track 를 작업 프로세스에서 — 기다림·실패의 규칙은 _run_on_worker (같은 규칙). meta (dict) 를 주면 where·worker_error·
    worker_state·timing (벽시계 — 결정 로그의 'timing' 아래에 둔다) 을 채운다. → (est, outcome, info)."""
    rows_list = list(rows_list)
    t0 = time.perf_counter()
    if not use_worker:
        out = refit_track(rows_list, calib, fit)
        if meta is not None:
            meta.update(where="in_process", timing={"track_refit_wall_s": time.perf_counter() - t0})
        return out
    r = _run_on_worker(_refit_job, {"rows": rows_list, "calib": calib, "fit": dict(fit)}, lambda: refit_track(rows_list, calib, fit),
                       what="track refit", log=log, cancel=cancel)
    if meta is not None:
        tim = {"track_refit_wall_s": time.perf_counter() - t0}
        if r["pool_note"]:
            tim["track_refit_pool_note"] = r["pool_note"]
        meta.update(where=r["where"], timing=tim)
        if r["where"] != "worker":
            meta.update(worker_error=r["worker_error"], worker_state=r["worker_state"])
    return r["out"]


class FeatureFeeder:
    """서버 쪽: S_track·트랜싯 녹화 중 새 프레임을 작업 프로세스의 추적 특징 스트림에 넘기는 스레드 (데몬; SeedFeeder 와 같은 규칙 — 녹화
    목록을 읽기만, 실패는 먹고 멈춘다). 보폭 = ⌈fps · t_frame⌉ (t_frame = 작업 프로세스가 잰 묶음당 프레임 벽시계 — 작업자 수만큼 병렬):
    작업 프로세스가 녹화를 따라잡는 가장 촘촘한 간격 (센서 fps 와 처리 시간에서 유도; 보통 1–2). 묶음 = 작업자당 한 프레임. 보폭은 이동 중
    미리 계산하는 행만 정한다 — 이동 뒤 compute_track 이 나머지 행을 모두 계산하므로 맞춤은 보폭 (부하) 과 무관하다."""

    def __init__(self, spec, kw: dict, *, z_valid_mm, workers, fps: float, depth_scale: float):
        from scan_core.seam_features import resolve_workers
        self.token = f"track-{os.getpid()}-{time.monotonic_ns()}"
        self.spec, self.kw, self.z_valid_mm, self.workers = spec, dict(kw), tuple(z_valid_mm), workers
        self.depth_scale = float(depth_scale)
        self.batch_max = resolve_workers(workers)
        f = float(fps) if fps is not None else float("nan")
        self.fps = f
        self.idle_s = self.batch_max / f if (np.isfinite(f) and f > 0.0) else float("nan")
        self.stride = 1
        self.started = False
        self.frames = 0
        self.batches = 0
        self.answered_frames = 0                        # 작업 프로세스가 계산해 돌려준 프레임 (기록)
        self.t_frame_s = float("nan")
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._cancelled = False
        self._thr: Optional[threading.Thread] = None
        self._records: Optional[list] = None

    def start(self, records: list) -> bool:
        if self.started or not np.isfinite(self.idle_s):
            return False
        try:
            self._records = records
            post_scan_pool().submit(feature_stream_open, self.token, self.spec, self.kw, self.z_valid_mm, self.workers, self.depth_scale)
            self._thr = threading.Thread(target=self._run, name="post-scan-track-feeder", daemon=True)
            self._thr.start()
            self.started = True
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"[:300]
        return self.started

    def _run(self) -> None:
        i = 0
        fut = None
        try:
            pool = post_scan_pool()
            while not self._stop.is_set():
                if fut is not None:
                    if not fut.done():
                        self._wake.wait()
                        self._wake.clear()
                        continue
                    B, wall = fut.result()
                    fut = None
                    self.answered_frames += int(B)
                    if B > 0 and wall > 0.0:
                        self.t_frame_s = float(wall) / float(B)
                        self.stride = max(1, int(math.ceil(self.fps * self.t_frame_s)))
                recs = self._records
                n = len(recs)
                pick = []
                while i < n and len(pick) < self.batch_max:
                    if i % self.stride == 0:
                        pick.append(recs[i])
                    i += 1
                if not pick:
                    self._wake.wait(self.idle_s)
                    self._wake.clear()
                    continue
                fut = pool.submit(feature_stream_batch, self.token, [int(r.result_counter) for r in pick], [r.x for r in pick],
                                  [r.z for r in pick])
                fut.add_done_callback(lambda _f: self._wake.set())
                self.frames += len(pick)
                self.batches += 1
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"[:300]

    def stop(self) -> dict:
        self._stop.set()
        self._wake.set()
        out = {"track_sent_frames": int(self.frames), "track_sent_batches": int(self.batches), "track_stride": int(self.stride),
               "track_answered_frames": int(self.answered_frames), "track_t_frame_s": float(self.t_frame_s)}
        if self.error:
            out["track_feeder_error"] = self.error
        return out

    def cancel(self) -> None:
        self._stop.set()
        self._wake.set()
        if self.started and not self._cancelled:
            self._cancelled = True
            try:
                post_scan_pool().submit(feature_stream_close)
            except Exception:  # noqa: BLE001
                pass
