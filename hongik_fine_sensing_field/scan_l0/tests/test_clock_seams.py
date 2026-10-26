"""E5 시계 이음매 린트 (명세 §3) — 시간·대기 호출은 시계 (vc3dz_driver.clock) 를 거치거나, 벽시계로 남는 이유 (분류 R-*) 가 허용 목록에 있다.

V (시계를 거침) = 세계의 시간: 팔 재생 틱, 촬영 박자·지연, 응답·첫 프레임·grab·이동 타임아웃, 녹화기 폴링, 프리롤, settle,
참여자 join, t_rx_ns, 자세 이력 시각축, 결정 로그 t, elapsed_s, motion_stop_to_data_s.
R (벽시계 그대로):
  R-측정   perf_counter / thread_time (계산 시간 측정)
  R-사람   확인 키 대기·남은 입력 비우기 (사람의 시간)
  R-IPC    다른 프로세스·비참여 스레드 대기 (post-scan fut.result, 피더, rx 스레드 join, 가짜 센서 accept 스레드, 프로세스 풀의 결과 대기 한계·
           자식 감시·감시 스레드 join — scan_core.proc_pool) — 그동안 시뮬 시간은 멈춘다
  R-ID     식별자·파일 이름
  R-문자열 str.join 등 시간과 무관한 같은 이름의 메서드
참여자가 R 대기에 들어가면 RUNNING 으로 남아 시간이 멈춘다 (= 계산 시간 0). 완료가 다른 참여자의 시뮬 진행에 달린 대기는 반드시 V.

생산 모듈은 vc3dz_driver.virtual_clock 을 import 하지 않고 clock.install 을 부르지 않는다.
"""
from __future__ import annotations

import ast
import os
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.dirname(os.path.dirname(HERE))

# 린트 대상 (명세 §3)
MODULES = {
    "client": os.path.join(STACK, "vc3dz_driver", "vc3dz_driver", "client.py"),
    "session": os.path.join(STACK, "scan_l0", "scan_l0", "session.py"),
    "autoteach": os.path.join(STACK, "scan_l0", "scan_l0", "autoteach.py"),
    "executor": os.path.join(STACK, "scan_l0", "scan_l0", "executor.py"),
    "cell_world": os.path.join(STACK, "scan_l0", "scan_l0", "cell_world.py"),
    # 시험 기반 (가짜 장치·가상 팔): 같은 규칙
    "fake_sensor": os.path.join(STACK, "vc3dz_driver", "vc3dz_driver", "fake_sensor.py"),
    "sim_arm": os.path.join(HERE, "sim_arm.py"),
    "sim_harness": os.path.join(HERE, "sim_harness.py"),
    "groove_world": os.path.join(HERE, "groove_world.py"),
    "render_prefetch": os.path.join(HERE, "render_prefetch.py"),
    # 프로세스 풀 (패스 뒤 처리·특징·도달성의 결과 대기는 모두 여기를 거친다)
    "proc_pool": os.path.join(STACK, "scan_core", "scan_core", "proc_pool.py"),
    "post_scan": os.path.join(STACK, "scan_l0", "scan_l0", "post_scan.py"),
    # 1G core: 오케스트레이터 (시계를 거친다) 와 작업 프로세스 (서버 쪽 기다림 = select({결과 소켓, 취소 fd}); 작업 프로세스 안의 스트림
    # 받기·덩어리 자식 거두기 = select / os.waitpid — 비참여 프로세스의 R-IPC, 이 린트가 보는 호출 형식이 아니다; 녹화기 탭은 기다리지 않는다)
    "teach_1g": os.path.join(STACK, "scan_l0", "scan_l0", "teach_1g.py"),
    "g1_worker": os.path.join(STACK, "scan_l0", "scan_l0", "g1_worker.py"),
}
# 시험 파일 (가상 시계로 도는 것): 시험 쪽 벽시계 주입기 (Timer·폴링 스레드) 도 시계를 거친다
SIM_TEST_FILES: list = []                 # (the lab line lints its legacy-pipeline simulation files here; none in this package)
for _f in SIM_TEST_FILES:
    MODULES[_f[:-3]] = os.path.join(HERE, _f)

# 허용 목록: (모듈, 함수, 호출문 (공백 정규화) 앞부분, 분류). 행 번호가 아니라 함수·호출문으로 맞춘다 (rebase 에 강하다).
ALLOW = [
    # client.py
    ("client", "VC3DZClient.close", "t.join(timeout=2.0)", "R-IPC"),            # rx 스레드 (비참여) join
    # autoteach.py — 사람의 시간 (확인 키)
    ("autoteach", "_read_confirm_key", "time.monotonic()", "R-사람"),
    ("autoteach", "_read_confirm_key", "time.sleep(0.01)", "R-사람"),
    # autoteach.py — post-scan 작업 프로세스·풀의 기다림은 모두 post_scan.wait_result (결과·죽음·취소, S4) 를 거친다 (아래 post_scan 항목)
    # 완료 콜백 안의 f.result(): 미래가 이미 끝났으므로 기다리지 않는다
    ("autoteach", "AutoTeach._emit_pass_cloud._done", "f.result()", "R-IPC"),
    # fake_sensor.py — accept·연결 스레드 (비참여: 소켓 대기) 정리, CLI 의 사람 시간
    ("fake_sensor", "FakeSensor.stop", "self._thread.join(timeout=2.0)", "R-IPC"),
    ("fake_sensor", "FakeSensor.stop", "th.join(timeout=2.0)", "R-IPC"),
    ("fake_sensor", "main", "time.sleep(1.0)", "R-사람"),
    # render_prefetch.py — 렌더 작업 프로세스의 결과 (스트림 스레드가 기다리는 동안 시뮬 시간은 멈춘다)
    ("render_prefetch", "RenderPrefetcher.take", "fut.result()", "R-IPC"),
    # proc_pool.py — 다른 프로세스 (풀의 자식) 의 결과·끝 (R-IPC: 계산하는 동안 시뮬 시간은 멈춘다 — 시뮬 시계로 재면 한계가 계산 중에 지난다)
    ("proc_pool", "wait_within", "concurrent.futures.wait(fs, timeout=limit_s)", "R-IPC"),
    ("proc_pool", "_watch", "multiprocessing.connection.wait(sentinels + [wake_r])", "R-IPC"),
    ("proc_pool", "OwnedPool.child_died", "multiprocessing.connection.wait(sentinels, timeout=0)", "R-IPC"),
    ("proc_pool", "OwnedPool.shutdown", "th.join()", "R-IPC"),                # 감시 스레드 (비참여)
    # post_scan.py — 작업 프로세스·그 계산 프로세스·피더 스레드 (비참여) 와의 대기 (한계 RESULT_TIMEOUT_S — scan_core.proc_pool)
    # 서버의 작업 프로세스 기다림 규칙 하나 (S4): 결과·작업 프로세스의 죽음·취소 Future 중 먼저 (여러 대상 한 번에), 한계 RESULT_TIMEOUT_S
    ("post_scan", "wait_result", "_cf.wait([fut] if cancel is None else [fut, cancel], timeout=limit", "R-IPC"),
    ("post_scan", "wait_result", "fut.result()", "R-IPC"),                  # 이미 끝난 Future (wait 뒤)
    ("post_scan", "wait_result", "cancel.result()", "R-IPC"),               # 이미 끝난 취소 Future
    ("post_scan", "_on_lasting_thread", "fut.result()", "R-IPC"),           # 작업 프로세스 띄우기 (비참여 스레드, S3)
    ("post_scan", "_stream_finish", "th.join()", "R-IPC"),                   # 작업 프로세스 안의 받기 스레드 (poll 만 — 깨우면 끝난다)
    ("post_scan", "feature_stream_batch", "f.result()", "R-IPC"),            # 이미 끝난 덩어리 Future (proc_pool.wait_within 뒤)
    ("post_scan", "SeedFeeder.stop", "self._thr.join(self._cycle_s())", "R-IPC"),
    ("post_scan", "SeedFeeder._run", "self._open.result()", "R-IPC"),        # 이미 끝난 Future (done 뒤)
    ("post_scan", "SeedFeeder.__init__", "time.monotonic_ns()", "R-ID"),      # 스트림 토큰
    ("post_scan", "FeatureFeeder.__init__", "time.monotonic_ns()", "R-ID"),   # 스트림 토큰
    ("post_scan", "FeatureFeeder._run", "fut.result()", "R-IPC"),            # 이미 끝난 Future (done 뒤)
    # 추적 특징 피더 (비참여 스레드): 작업 프로세스의 묶음 답 (완료 콜백·stop 이 깨운다), 녹화를 기다리는 쉼
    ("post_scan", "FeatureFeeder._run", "self._wake.wait()", "R-IPC"),
    ("post_scan", "FeatureFeeder._run", "self._wake.wait(self.idle_s)", "R-IPC"),
    # g1_worker.py — SIGKILL 한 작업 프로세스 거두기 (끝이 보장된 기다림), PYTHONPATH 문자열
    ("g1_worker", "G1Worker._reap", "self.proc.wait()", "R-IPC"),
    ("g1_worker", "G1Worker.start", "os.pathsep.join(", "R-문자열"),
]

_WALL_TIME = {"sleep", "monotonic", "monotonic_ns", "perf_counter", "time"}


def _src(node) -> str:
    return " ".join(ast.unparse(node).split())


def _qualnames(tree):
    """노드 → 둘러싼 함수의 qualname."""
    out = {}

    def walk(n, stack):
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(ch, stack + [ch.name])
            else:
                out[id(ch)] = ".".join(stack)
                walk(ch, stack)
    walk(tree, [])
    return out


def _via_clock(recv: str) -> bool:
    return "clock" in recv.lower() or "_cobj" in recv


def find_time_calls(src: str):
    """→ [(함수 qualname, 호출문, 행)] — 시계를 거치지 않는 시간·대기 호출."""
    tree = ast.parse(src)
    qn = _qualnames(tree)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        f = node.func
        recv = _src(f.value)
        name = f.attr
        flag = False
        if isinstance(f.value, ast.Name) and f.value.id == "time" and name in _WALL_TIME | {"thread_time"}:
            flag = name != "thread_time" and name != "perf_counter"    # 계산 시간 측정 (R-측정) 은 어디서나 허용
            if name == "perf_counter":
                flag = False
        elif isinstance(f.value, ast.Name) and f.value.id == "threading" and name == "Timer":
            flag = True
        elif name in ("wait", "join"):
            if isinstance(f.value, ast.Constant) or recv in ("os.path", "sep") or _via_clock(recv):
                flag = False
            elif name == "join" and isinstance(f.value, ast.JoinedStr):
                flag = False
            else:
                flag = True
        elif name == "get" and any(k.arg == "timeout" for k in node.keywords) and not _via_clock(recv):
            flag = True
        elif name == "result" and not node.keywords and len(node.args) <= 1 and not _via_clock(recv):
            flag = True                                                   # future.result([timeout]) — 허용 목록에서 R-IPC 로
        if flag:
            hits.append((qn.get(id(node), ""), _src(node), node.lineno))
    return hits


def violations(mod: str, src: str) -> list:
    out = []
    for fn, call, line in find_time_calls(src):
        ok = any(m == mod and fn == f and call.startswith(c) for m, f, c, _cls in ALLOW)
        if not ok:
            out.append(f"{mod}:{line} in {fn or '<module>'}: {call}")
    return out


@pytest.mark.parametrize("mod", sorted(MODULES))
def test_time_calls_go_through_the_clock_or_are_allowlisted(mod):
    src = open(MODULES[mod], encoding="utf-8").read()
    assert violations(mod, src) == []


def test_allowlist_entries_are_all_still_used():
    used = set()
    for mod, path in MODULES.items():
        for fn, call, _line in find_time_calls(open(path, encoding="utf-8").read()):
            for m, f, c, _cls in ALLOW:
                if m == mod and fn == f and call.startswith(c):
                    used.add((m, f, c))
    stale = [a[:3] for a in ALLOW if a[:3] not in used]
    assert stale == []


def test_meta_a_raw_sleep_in_a_production_module_fails_the_lint():
    src = open(MODULES["session"], encoding="utf-8").read()
    bad = src.replace("        self._clock.sleep(max(0.0, settle_s))", "        time.sleep(max(0.0, settle_s))")
    assert bad != src
    v = violations("session", bad)
    assert len(v) == 1 and "time.sleep(max(0.0, settle_s))" in v[0]
    bad2 = src.replace("            self._clock.join(self._rec, 2.0)", "            self._rec.join(2.0)")
    assert bad2 != src and len(violations("session", bad2)) == 1


def _production_files():
    for pkg in ("vc3dz_driver/vc3dz_driver", "scan_l0/scan_l0", "scan_core/scan_core", "hongik_scan_ros/hongik_scan_ros"):
        d = os.path.join(STACK, pkg)
        for name in sorted(os.listdir(d)):
            if name.endswith(".py") and name != "virtual_clock.py":
                yield os.path.join(d, name)


def test_production_never_imports_the_virtual_clock_nor_installs_a_clock():
    bad = []
    for path in _production_files():
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                if any("virtual_clock" in n for n in names):
                    bad.append(f"{path}:{node.lineno} imports the virtual clock")
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("clock") and any(a.name == "install" for a in node.names):
                    bad.append(f"{path}:{node.lineno} imports clock.install")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "install" \
                    and "clock" in _src(node.func.value).lower():
                bad.append(f"{path}:{node.lineno} calls clock.install")
    assert bad == []
