"""시험용 ROS 도메인 할당 — 동시에 도는 시험 (다른 세션·worktree 의 scripts/run_tests.py, 직접 띄운 pytest) 끼리 ROS_DOMAIN_ID 를 절대
나눠 쓰지 않는다 (2026-10-08 ~21:04: 병렬 실행 5–6 개 중 하나가 남는 고정 구간이 없어 180 을 나눠 쓰고, 두 가짜 팔·센서가 섞여 test_d 가
DATA_FAIL. 고정 구간 100/140/180 은 실행 셋까지만 받고 넷째부터 '나눠 씀' 경고만 내고 계속 돌았다).

쓸 수 있는 도메인: DDSI-RTPS 기본 포트 사상의 포트가 UDP 범위 안에 들고 (→ ≤ 232), 커널의 임시 포트 범위
(/proc/sys/net/ipv4/ip_local_port_range) 와 겹치지 않는 도메인 (ROS 2 'About domain ID': Linux 기본값이면 0–101, 215–232) 에서
ROS 기본 도메인 (ROS_DOMAIN_ID 가 없는 모든 프로세스의 도메인) 과 호출한 쪽의 ROS_DOMAIN_ID (사용자의 대화형 도메인) 를 뺀 것.

도메인 d 가 비어 있다 = (1) 이 할당기의 잠금 파일 <LOCK_DIR>/fs_ros_domain_<d>.lock 을 아무도 쥐지 않았고 (2) 읽을 수 있는 살아 있는
프로세스 어느 것의 환경에도 ROS_DOMAIN_ID=d 나 d 를 담은 FS_TEST_DOMAINS 가 없다. (2) 는 할당기를 거치지 않은 실행 (옛 실행기, 손으로
ROS_DOMAIN_ID 를 준 실행) 과, 잠금을 쥔 프로세스가 죽은 뒤에도 그 도메인에 남은 프로세스 (고아 pytest·launch) 를 잡는다.
잠금은 도메인마다 따로, 실제로 띄운 작업자·그래프 수만큼만 쥔다. 비어 있는 것이 없으면 나눠 쓰지 않고 기다린다 (기다린다고 알린다).

남은 프로세스: 시험이 띄운 것 중 스스로 떨어져 나가는 것 (ros2 CLI 데몬 — `ros2 control …` 이 띄우고 새 세션으로 빠져 프로세스 그룹 정리에
안 걸린다) 은 끝나지 않고 도메인에 남는다. 그대로 두면 위 (2) 때문에 그 도메인을 영영 못 쓴다 (실행마다 풀이 줄어든다). 그래서 시험 프로세스
나무의 주인 (실행기의 작업자, 실행기 없이 띄운 pytest) 이 RUN_TAG 표시를 물려 주고, 나무의 뿌리 (시험 대상 하나 / pytest) 가 끝나면 그 표시를
가진 채 남은 프로세스를 멈춘다 (stop_processes: SIGTERM → 유예 → SIGKILL). 표시는 주인마다 달라 (pid:도메인) 다른 실행의 것은 건드리지 않는다.
표준 라이브러리만 쓴다 (scripts/run_tests.py 는 venv·ROS 없이 이 파일을 읽는다).
"""
from __future__ import annotations

import fcntl
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

# DDSI-RTPS 2.x §9.6.1.1 기본 포트 사상: 도메인 D 의 참가자 i 는 PB + DG·D + d0 / d2 (멀티캐스트 탐색 / 사용자),
# PB + DG·D + d1 / d3 + PG·i (유니캐스트 탐색 / 사용자) 포트를 쓴다.
RTPS_PB, RTPS_DG, RTPS_PG = 7400, 250, 2
RTPS_D0, RTPS_D1, RTPS_D2, RTPS_D3 = 0, 10, 1, 11
# CycloneDDS Discovery/MaxAutoParticipantIndex 기본값: 루프백 (ROS_LOCALHOST_ONLY, 멀티캐스트 없음) 에서는 참가자 번호를 0..9 에서 자동으로 고른다
CYCLONE_MAX_AUTO_PARTICIPANT_INDEX = 9
UDP_PORT_MAX = 65535
ROS_DEFAULT_DOMAIN = 0                # rmw: ROS_DOMAIN_ID 가 없거나 비면 0
EPHEMERAL_PORTS = Path("/proc/sys/net/ipv4/ip_local_port_range")
PROC = Path("/proc")
LOCK_DIR = Path("/tmp")               # 호스트 전체가 같은 UDP 포트를 쓰므로 모든 worktree·세션이 같은 곳을 본다
WAIT_POLL_S = 1.0                     # 기다릴 때 다시 보는 주기 — 도메인은 시험 파일 하나 (수 초~수 분) 동안 쥐므로 1 s 지연은 무시할 만하다
RUN_TAG = "FS_TEST_RUN"               # 시험 프로세스 나무의 주인 표시 (자식·떨어져 나간 데몬까지 환경으로 물려받는다)
STOP_GRACE_S = 5.0                    # 남은 프로세스: SIGTERM 뒤 SIGKILL 까지 (ros_harness.ProcessGroup 의 SIGTERM 단계와 같다)


def port_span(domain: int, max_participant_index: int = CYCLONE_MAX_AUTO_PARTICIPANT_INDEX) -> tuple:
    """도메인이 쓰는 UDP 포트의 [최소, 최대]."""
    base = RTPS_PB + RTPS_DG * int(domain)
    unicast = RTPS_PG * int(max_participant_index)
    return (base + min(RTPS_D0, RTPS_D1, RTPS_D2, RTPS_D3),
            base + max(RTPS_D0, RTPS_D2, RTPS_D1 + unicast, RTPS_D3 + unicast))


def ephemeral_port_range(path: Path = EPHEMERAL_PORTS) -> tuple:
    """커널이 임시로 나눠 주는 포트 범위 (다른 프로그램의 UDP 소켓이 언제든 잡을 수 있다)."""
    lo, hi = (int(v) for v in Path(path).read_text().split())
    return lo, hi


def valid_domains(ephemeral: Optional[tuple] = None) -> list:
    """포트가 UDP 범위 안이고, ephemeral (lo, hi) 이 주어지면 그 범위와 겹치지 않는 도메인 (0 부터 오름차순)."""
    out, d = [], 0
    while True:
        lo, hi = port_span(d)
        if hi > UDP_PORT_MAX:
            return out
        if ephemeral is None or hi < ephemeral[0] or lo > ephemeral[1]:
            out.append(d)
        d += 1


def parse_domain_id(value) -> Optional[int]:
    """ROS_DOMAIN_ID 값 → 도메인. 없거나 비면 ROS 기본 도메인, 정수가 아니면 None (rmw 가 거부하는 값)."""
    if value is None or not str(value).strip():
        return ROS_DEFAULT_DOMAIN
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def usable_domains(environ=None, ephemeral: Optional[tuple] = None) -> list:
    """시험에 줄 수 있는 도메인: valid_domains (임시 포트 범위 제외) 에서 ROS 기본 도메인과 호출한 쪽의 ROS_DOMAIN_ID 를 뺀 것."""
    environ = os.environ if environ is None else environ
    eph = ephemeral_port_range() if ephemeral is None else ephemeral
    skip = {ROS_DEFAULT_DOMAIN, parse_domain_id(environ.get("ROS_DOMAIN_ID"))}
    return [d for d in valid_domains(eph) if d not in skip]


def parse_domains(text) -> list:
    """FS_TEST_DOMAINS 형식 '7', '7-9', '7,12-14' → 도메인 목록 (순서 유지, 중복 없음)."""
    out: list = []
    for part in (p.strip() for p in str(text or "").split(",")):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        for d in (range(int(lo), int(hi) + 1) if sep else (int(lo),)):
            if d not in out:
                out.append(d)
    return out


def format_domains(domains) -> str:
    """parse_domains 의 역: 연속 구간은 'lo-hi' 로."""
    ds, parts, i = sorted(set(int(d) for d in domains)), [], 0
    while i < len(ds):
        j = i
        while j + 1 < len(ds) and ds[j + 1] == ds[j] + 1:
            j += 1
        parts.append(str(ds[i]) if i == j else f"{ds[i]}-{ds[j]}")
        i = j + 1
    return ",".join(parts)


def _environs(proc: Path = PROC):
    """(pid, 시작 때 환경 {bytes: bytes}) — 읽을 수 있는 살아 있는 프로세스 (다른 사용자·이미 끝난 것·좀비는 빠진다)."""
    try:
        entries = list(os.scandir(proc))
    except OSError:
        return
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = Path(entry.path, "environ").read_bytes()
        except OSError:
            continue
        if raw:
            yield int(entry.name), dict(item.split(b"=", 1) for item in raw.split(b"\0") if b"=" in item)


def scan_processes(proc: Path = PROC) -> dict:
    """{pid: (ROS_DOMAIN_ID 로 쓰는 도메인 또는 None, FS_TEST_DOMAINS 로 예약한 도메인 목록)} — 둘 중 하나라도 명시한 살아 있는 프로세스.
    ROS_DOMAIN_ID 가 없거나 빈 프로세스는 기본 도메인에 있고 그 도메인은 늘 빼므로 따로 적지 않는다. 다른 사용자의 프로세스는 읽을 수 없어 빠진다."""
    out: dict = {}
    for pid, env in _environs(proc):
        dom_raw, res_raw = env.get(b"ROS_DOMAIN_ID"), env.get(b"FS_TEST_DOMAINS")
        dom = parse_domain_id(dom_raw.decode(errors="replace")) if dom_raw is not None else None
        if dom == ROS_DEFAULT_DOMAIN:
            dom = None
        try:
            res = parse_domains(res_raw.decode(errors="replace")) if res_raw else []
        except ValueError:
            res = []
        if dom is not None or res:
            out[pid] = (dom, res)
    return out


def tagged_processes(tag: str, proc: Path = PROC) -> list:
    """RUN_TAG == tag 인 살아 있는 프로세스 (이 프로세스는 빼고)."""
    if not tag:
        return []
    me, want = os.getpid(), str(tag).encode()
    return sorted(pid for pid, env in _environs(proc) if pid != me and env.get(RUN_TAG.encode()) == want)


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            return f.read().rsplit(b")", 1)[1].split()[0] != b"Z"     # 좀비 = 이미 끝남 (거둘 부모를 기다릴 뿐)
    except (OSError, IndexError):
        return False


def _describe(pid: int) -> str:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmd = "?"
    return f"pid {pid} {cmd[:120]}"


def stop_processes(pids, grace_s: float = None) -> list:
    """프로세스들을 멈춘다: SIGTERM, grace_s 안에 안 끝나면 SIGKILL. 멈춘 것의 설명 ('pid N 명령') 목록."""
    grace_s = STOP_GRACE_S if grace_s is None else float(grace_s)
    desc = {p: _describe(p) for p in pids if _alive(p)}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        left = [p for p in desc if _alive(p)]
        for p in left:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        t_end = time.monotonic() + grace_s
        while any(_alive(p) for p in left) and time.monotonic() < t_end:
            time.sleep(0.05)
    return [desc[p] for p in sorted(desc)]


def domains_of(procs: dict) -> dict:
    """scan_processes 결과 → {도메인: {pid}} (쓰는 것 + 예약한 것)."""
    out: dict = {}
    for pid, (dom, res) in procs.items():
        for d in ([dom] if dom is not None else []) + list(res):
            out.setdefault(d, set()).add(pid)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 잠금 (도메인마다 파일 하나, flock — 쥔 프로세스가 끝나면 커널이 푼다)
# ----------------------------------------------------------------------------------------------------------------------
_held: dict = {}                      # 도메인 → 이 프로세스가 쥔 잠금 파일 fd
_held_lock = threading.Lock()         # 실행기의 작업자 스레드들이 함께 부른다


def _lock_path(domain: int, lock_dir) -> Path:
    return Path(lock_dir) / f"fs_ros_domain_{int(domain)}.lock"


def _try_lock(domain: int, lock_dir) -> bool:
    try:
        fd = os.open(_lock_path(domain, lock_dir), os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:                               # 다른 사용자의 잠금 파일 — 그쪽 시험의 도메인으로 본다
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False
    _held[int(domain)] = fd
    try:                                                  # 기다리는 쪽의 알림용: 누가 쥐었나 (실패해도 잠금은 그대로)
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()} in {os.getcwd()} ({' '.join(sys.argv[:2])})\n".encode(errors="replace"))
    except OSError:
        pass
    return True


def _holder(domain: int, lock_dir) -> str:
    try:
        return _lock_path(domain, lock_dir).read_text(errors="replace").strip() or "?"
    except OSError:
        return "?"


def held_domains() -> list:
    """이 프로세스가 지금 잠근 도메인."""
    with _held_lock:
        return sorted(_held)


def release_domain(domain: int) -> None:
    """acquire_domain 으로 잠근 도메인을 돌려준다 (이 프로세스가 쥐지 않은 것이면 아무것도 안 함)."""
    with _held_lock:
        fd = _held.pop(int(domain), None)
    if fd is not None:
        try:
            os.ftruncate(fd, 0)
        finally:
            os.close(fd)


def _busy_text(pool: list, lock_dir, used: dict) -> str:
    mine = [d for d in pool if d in _held]
    by_proc = [d for d in pool if d in used and d not in _held]
    by_lock = [d for d in pool if d not in used and d not in _held]
    parts = []
    if by_lock:
        parts.append(f"{len(by_lock)} locked by other test runs ({'; '.join(sorted({_holder(d, lock_dir) for d in by_lock}))})")
    if by_proc:
        pids = sorted({p for d in by_proc for p in used[d]})
        parts.append(f"{len(by_proc)} used by running processes (pids {', '.join(str(p) for p in pids[:8])}"
                     f"{', ...' if len(pids) > 8 else ''})")
    if mine:
        parts.append(f"{len(mine)} held by this process")
    return f"all {len(pool)} usable ROS domains ({format_domains(pool)}) are in use: {', '.join(parts)}"


def acquire_domain(pool=None, *, lock_dir=None, scan: Callable[[], dict] = None, give_up: Callable[[], bool] = None,
                   report: Callable[[str], None] = None, poll_s: float = None) -> Optional[int]:
    """비어 있는 도메인 하나를 잠가 돌려준다 (pool 순서 = 낮은 번호부터). 비어 있는 것이 없으면 나눠 쓰지 않고 기다린다 — 기다리기 시작할 때와
    얻었을 때 report 로 알린다. give_up() 이 참이 되면 (기다리던 일이 없어짐) 잠그지 않고 None.
    pool 기본 = usable_domains(), lock_dir 기본 = LOCK_DIR, scan 기본 = scan_processes (시험은 바꿔 끼운다)."""
    pool = usable_domains() if pool is None else [int(d) for d in pool]
    if not pool:
        raise RuntimeError("no usable ROS domain (all are outside the DDS port range, in the ephemeral port range or the caller's)")
    lock_dir = LOCK_DIR if lock_dir is None else lock_dir
    scan = scan_processes if scan is None else scan
    poll_s = WAIT_POLL_S if poll_s is None else float(poll_s)
    report = report or (lambda m: print(m, file=sys.stderr, flush=True))
    t_wait = None
    while True:
        if give_up is not None and give_up():
            return None
        used = domains_of(scan())
        got = None
        with _held_lock:
            got = next((d for d in pool if d not in used and d not in _held and _try_lock(d, lock_dir)), None)
            busy = _busy_text(pool, lock_dir, used) if got is None and t_wait is None else ""
        if got is not None:
            if t_wait is not None:
                report(f"got ROS domain {got} after waiting {time.monotonic() - t_wait:.0f} s")
            return got
        if t_wait is None:
            t_wait = time.monotonic()
            report(f"waiting for a free ROS domain: {busy}; not sharing - retrying every {poll_s:g} s")
        time.sleep(poll_s)
