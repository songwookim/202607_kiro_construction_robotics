"""시험용 ROS 도메인 할당기 (ros_domains, ros_harness.pick_domain) 시험 — ROS 없이 돈다.

동시에 도는 실행 (별도 프로세스) 은 서로 겹치지 않는 도메인을 받고, 빈 것이 없으면 나눠 쓰지 않고 기다리다 풀리는 것을 받는다 (2026-10-08:
고정 구간 셋이 다 차자 넷째 실행이 180 을 나눠 써 test_d 가 다른 가짜 팔·센서와 섞였다). 잠금 폴더는 tmp_path, 풀은 실제 시험이 쓰지 않는
번호 (9000 대) 라 다른 세션의 실제 실행과 엮이지 않는다.
"""
import os
import queue
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("fcntl")                                              # 잠금은 POSIX flock (Ubuntu 시험 PC)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_domains as R  # noqa: E402

FAST_POLL_S = 0.02

# 실행 하나를 흉내 내는 자식 프로세스: n 개를 잠그고 (GOT d), stdin 명령 'release d' / 'exit' 를 기다린다
CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
import ros_domains as R
pool = [int(x) for x in sys.argv[2].split(",")]
for _ in range(int(sys.argv[3])):
    d = R.acquire_domain(pool, lock_dir=sys.argv[4], scan=lambda: {}, poll_s=float(sys.argv[5]),
                         report=lambda m: print("MSG", m, flush=True))
    print("GOT", d, flush=True)
for line in sys.stdin:
    cmd = line.split()
    if cmd and cmd[0] == "release":
        R.release_domain(int(cmd[1]))
        print("RELEASED", cmd[1], flush=True)
    elif cmd and cmd[0] == "exit":
        break
"""


class Run:
    """자식 프로세스 하나 (= 동시에 도는 실행 하나) 와 그 출력 줄 큐."""

    def __init__(self, pool, n, lock_dir, env=None):
        self.proc = subprocess.Popen([sys.executable, "-u", "-c", CHILD, HERE, ",".join(str(d) for d in pool), str(n), str(lock_dir),
                                      str(FAST_POLL_S)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, env=env)
        self.lines: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))

    def next(self, prefix: str, timeout: float):
        """prefix 로 시작하는 다음 줄 (없으면 None). 다른 줄은 seen 에 모은다."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            try:
                line = self.lines.get(timeout=max(0.0, t_end - time.monotonic()))
            except queue.Empty:
                return None
            if line.startswith(prefix):
                return line
            if line.startswith("Traceback"):
                raise AssertionError(f"child failed: {line}")
        return None

    def got(self, n: int, timeout: float = 20.0) -> list:
        out = []
        for _ in range(n):
            line = self.next("GOT", timeout)
            assert line is not None, f"run did not get a domain within {timeout} s (got {out})"
            out.append(int(line.split()[1]))
        return out

    def send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def close(self):
        if self.proc.poll() is None:
            try:
                self.send("exit")
                self.proc.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait(timeout=5.0)


@pytest.fixture
def runs():
    made = []

    def start(*a, **k):
        r = Run(*a, **k)
        made.append(r)
        return r

    yield start
    for r in made:
        r.close()


# ----------------------------------------------------------------------------------------------------------------------
# 유효 범위 (DDS 포트 사상에서 유도)
# ----------------------------------------------------------------------------------------------------------------------
def test_valid_range_is_derived_from_the_rtps_port_mapping():
    doms = R.valid_domains()
    assert doms == list(range(0, 233))                                   # CycloneDDS / ROS 2: 도메인 ≤ 232
    assert R.port_span(232)[1] <= R.UDP_PORT_MAX < R.port_span(233)[1]
    # 커널 임시 포트 범위와 겹치는 도메인은 뺀다 — Linux 기본 32768–60999 이면 ROS 2 문서의 0–101, 215–232
    assert R.valid_domains((32768, 60999)) == list(range(0, 102)) + list(range(215, 233))
    # 범위를 다른 커널 설정에서 읽으면 결과도 따라간다 (상수 표가 아니다)
    narrow = R.valid_domains((40000, 50000))
    assert all(R.port_span(d)[1] < 40000 or R.port_span(d)[0] > 50000 for d in narrow) and 100 in narrow and 150 not in narrow


def test_usable_domains_exclude_the_ros_default_and_the_callers_domain():
    eph = (32768, 60999)
    assert R.usable_domains({}, eph) == list(range(1, 102)) + list(range(215, 233))
    assert R.usable_domains({"ROS_DOMAIN_ID": ""}, eph)[0] == 1           # 빈 값 = 기본 도메인 0
    u = R.usable_domains({"ROS_DOMAIN_ID": "7"}, eph)
    assert 0 not in u and 7 not in u and 8 in u and len(u) == 118


def test_fs_test_domains_lists_and_ranges():
    assert R.parse_domains("140-149") == list(range(140, 150))           # 옛 실행기 형식 그대로 읽는다
    assert R.parse_domains("7, 12-14,7") == [7, 12, 13, 14]
    assert R.parse_domains("") == []
    assert R.format_domains([14, 7, 12, 13, 1]) == "1,7,12-14"
    assert R.parse_domains(R.format_domains([3, 5, 6, 9])) == [3, 5, 6, 9]


# ----------------------------------------------------------------------------------------------------------------------
# 동시에 도는 실행: 겹치지 않음 · 다 쓰이면 기다림 (나눠 쓰지 않음)
# ----------------------------------------------------------------------------------------------------------------------
def test_concurrent_runs_get_disjoint_domains(runs, tmp_path):
    pool = list(range(9001, 9013))                                       # 12 = 실행 4 × 작업자 3
    rs = [runs(pool, 3, tmp_path) for _ in range(4)]                     # 동시에 띄운다
    got = [r.got(3) for r in rs]
    flat = [d for g in got for d in g]
    assert sorted(flat) == pool, got                                      # 모두 다르고, 풀을 정확히 나눴다
    # 다섯째 실행: 빈 것이 없으니 기다린다 (아무것도 받지 않음) — 한 실행이 끝나면 그 도메인을 받는다
    late = runs(pool, 1, tmp_path)
    msg = late.next("MSG", 10.0)
    assert msg is not None and "waiting for a free ROS domain" in msg and "12 locked by other test runs" in msg, msg
    assert late.next("GOT", 1.0) is None
    rs[2].close()
    msg = late.next("MSG", 20.0)
    assert msg is not None and msg.startswith("MSG got ROS domain") and "after waiting" in msg, msg
    d = late.got(1)[0]
    assert d in got[2] and msg.split()[4] == str(d), (d, got, msg)


def test_exhaustion_waits_and_never_shares(runs, tmp_path):
    pool = [9101, 9102]
    a = runs(pool, 2, tmp_path)
    assert sorted(a.got(2)) == pool
    b = runs(pool, 1, tmp_path)
    assert "waiting for a free ROS domain" in (b.next("MSG", 10.0) or "")
    assert b.next("GOT", 1.5) is None                                     # ~75 번 다시 봐도 쥔 것을 주지 않는다
    a.send("release 9102")
    assert a.next("RELEASED", 5.0) == "RELEASED 9102"
    assert b.got(1) == [9102]                                             # 풀린 것만 — a 가 아직 쥔 9101 은 아니다
    c = runs(pool, 1, tmp_path)
    assert "waiting" in (c.next("MSG", 10.0) or "") and c.next("GOT", 1.0) is None


def test_give_up_returns_none_without_locking(tmp_path):
    held = R.acquire_domain([9201], lock_dir=tmp_path, scan=lambda: {})
    try:
        msgs = []
        calls = []
        r = R.acquire_domain([9201], lock_dir=tmp_path, scan=lambda: {}, poll_s=FAST_POLL_S, report=msgs.append,
                             give_up=lambda: calls.append(1) or len(calls) > 3)
        assert r is None and len(msgs) == 1 and "1 held by this process" in msgs[0], msgs
    finally:
        R.release_domain(held)
    assert R.held_domains() == []


def test_domains_used_or_reserved_by_running_processes_are_busy(tmp_path):
    """할당기를 거치지 않은 프로세스 (손으로 ROS_DOMAIN_ID 를 준 실행, 옛 실행기의 FS_TEST_DOMAINS, 잠금을 놓친 고아 launch) 의 도메인은 주지 않는다."""
    env = dict(os.environ, ROS_DOMAIN_ID="9301", FS_TEST_DOMAINS="9302-9303")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env)
    try:
        t_end = time.monotonic() + 5.0
        while child.pid not in R.scan_processes() and time.monotonic() < t_end:
            time.sleep(0.05)
        assert R.scan_processes()[child.pid] == (9301, [9302, 9303])
        used = R.domains_of(R.scan_processes())
        assert all(child.pid in used[d] for d in (9301, 9302, 9303))
        d = R.acquire_domain([9301, 9302, 9303, 9304], lock_dir=tmp_path)
        assert d == 9304
        R.release_domain(d)
    finally:
        child.kill()
        child.wait()
    d = R.acquire_domain([9301, 9302, 9303, 9304], lock_dir=tmp_path)       # 그 프로세스가 끝나면 다시 쓸 수 있다
    assert d == 9301
    R.release_domain(d)


# ----------------------------------------------------------------------------------------------------------------------
# pick_domain (ros_harness): 작업자 도메인 먼저, 더 필요하면 전역 풀에서 잠근다
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def harness(monkeypatch, tmp_path):
    import ros_harness
    monkeypatch.setattr(R, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(R, "usable_domains", lambda *a, **k: [9402, 9403])
    monkeypatch.setattr(ros_harness, "_handed", set())
    yield ros_harness
    for d in R.held_domains():
        R.release_domain(d)


def test_pick_domain_uses_the_workers_domain_then_locks_more_for_more_graphs(harness, monkeypatch):
    monkeypatch.setenv("FS_TEST_DOMAINS", "9401")
    a = harness.pick_domain()
    assert a == 9401 and R.held_domains() == []                         # 작업자 도메인: 실행기가 쥐고 있다
    b = harness.pick_domain()                                            # 두 번째 그래프 (a 가 아직 살아 있음)
    assert b == 9402 and R.held_domains() == [9402]
    harness.release_domain(a)
    assert harness.pick_domain() == 9401                                  # a 를 정리한 뒤의 다음 시험은 작업자 도메인을 다시 쓴다
    harness.release_domain(b)
    assert R.held_domains() == []


def test_pick_domain_skips_its_domain_while_a_left_over_process_uses_it(harness, monkeypatch):
    monkeypatch.setenv("FS_TEST_DOMAINS", "9401")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=dict(os.environ, ROS_DOMAIN_ID="9401"))
    try:
        t_end = time.monotonic() + 5.0
        while child.pid not in R.scan_processes() and time.monotonic() < t_end:
            time.sleep(0.05)
        assert harness.pick_domain() == 9402                              # 앞 시험이 남긴 프로세스와 나눠 쓰지 않는다
    finally:
        child.kill()
        child.wait()


def test_pick_domain_without_the_runner_locks_from_the_pool(harness, monkeypatch):
    monkeypatch.delenv("FS_TEST_DOMAINS", raising=False)
    d = harness.pick_domain()
    assert d == 9402 and R.held_domains() == [9402]
    with harness.IsolatedEnv(d):
        assert os.environ["ROS_DOMAIN_ID"] == "9402"
    assert R.held_domains() == []                                         # IsolatedEnv 가 끝나며 돌려줬다


# ----------------------------------------------------------------------------------------------------------------------
# 남은 프로세스 정리: 시험이 띄우고 떨어져 나간 것 (ros2 CLI 데몬처럼 새 세션, SIGTERM 무시) 이 도메인을 영영 막지 않게
# ----------------------------------------------------------------------------------------------------------------------
def test_left_over_processes_of_a_finished_tree_are_stopped_and_free_their_domain(tmp_path):
    stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('up', flush=True); time.sleep(60)"
    mine = subprocess.Popen([sys.executable, "-u", "-c", stubborn], start_new_session=True, stdout=subprocess.PIPE, text=True,
                            env=dict(os.environ, ROS_DOMAIN_ID="9501", **{R.RUN_TAG: "4242:9501"}))
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             env=dict(os.environ, ROS_DOMAIN_ID="9502", **{R.RUN_TAG: "4343:9502"}))
    try:
        assert mine.stdout.readline().strip() == "up"                       # SIGTERM 무시를 건 뒤
        assert R.tagged_processes("4242:9501") == [mine.pid]                 # 표시가 같은 것만 (다른 실행의 것은 아님)
        assert R.tagged_processes("") == []
        assert 9501 in R.domains_of(R.scan_processes())
        t0 = time.monotonic()
        stopped = R.stop_processes(R.tagged_processes("4242:9501"), grace_s=0.3)
        assert len(stopped) == 1 and stopped[0].startswith(f"pid {mine.pid} ") and "SIG_IGN" in stopped[0], stopped
        assert time.monotonic() - t0 < 5.0                                   # SIGTERM 을 무시해도 SIGKILL 로
        mine.wait(timeout=5.0)
        assert other.poll() is None                                           # 다른 표시는 그대로
        d = R.acquire_domain([9501], lock_dir=tmp_path)                       # 도메인이 다시 빈다
        assert d == 9501
        R.release_domain(d)
    finally:
        for p in (mine, other):
            if p.poll() is None:
                p.kill()
            p.wait()
