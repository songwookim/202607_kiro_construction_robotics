"""hongik ROS 노드 시험 (rclpy 프로세스 내, 격리 도메인): 센서 소유권 (프로파일러 노드는 스트리밍할 때만 센서를 잡고 ~/stop 으로 놓는다,
서버는 목표 전에 그것을 요청한다), 서버의 토픽·서비스 (센서 입출력과 액션 통신뿐). 실센서·실로봇 접속 없음 (가짜 센서 127.0.0.1).
"""
import os
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STACK = os.path.dirname(PKG)
for p in (PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"), os.path.join(STACK, "vc3dz_driver")):
    if p not in sys.path:
        sys.path.insert(0, p)

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("hongik_scan_msgs.msg")
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from hongik_scan_msgs.msg import ProfileRaw  # noqa: E402
from vc3dz_driver.fake_sensor import FakeSensor  # noqa: E402


def _wait(pred, timeout=5.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _call(node, name, timeout=10.0):
    cli = node.create_client(Trigger, name)
    assert cli.wait_for_service(timeout_sec=5.0), name
    fut = cli.call_async(Trigger.Request())
    assert _wait(fut.done, timeout), f"{name} timed out"
    return fut.result()


@pytest.fixture
def default_ros():
    """기본 rclpy 컨텍스트를 격리 도메인으로 (노드 생성자가 context 인자를 받지 않는 클래스용). 시험마다 init/shutdown, 끝에서 도메인 반납."""
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    from ros_harness import pick_domain, release_domain
    domain = pick_domain()
    os.environ["ROS_DOMAIN_ID"] = str(domain)
    yield
    if rclpy.ok():
        rclpy.shutdown()
    release_domain(domain)


def test_profiler_node_ownership_cycle(default_ros, tmp_path):
    from hongik_scan_ros.profiler_node import ProfilerNode
    with FakeSensor(params={"AutoTriggerFPS": 100.0}) as fs:
        rclpy.init(args=["--ros-args", "-p", "sensor_ip:=127.0.0.1", "-p", f"sensor_port:={fs.port}",
                         "-p", f"backup_dir:={tmp_path}"])
        prof = ProfilerNode()
        probe = Node("probe")
        owned, frames = [], []
        probe.create_subscription(Bool, "/hongik_profiler_node/sensor_owned", lambda m: owned.append(bool(m.data)), QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        probe.create_subscription(ProfileRaw, "/hongik_profiler_node/profile", lambda m: frames.append(m), 50)
        ex = MultiThreadedExecutor(num_threads=4)
        ex.add_node(prof)
        ex.add_node(probe)
        th = threading.Thread(target=ex.spin, daemon=True)
        th.start()
        try:
            # 시작 시: 센서를 잡지 않는다
            assert _wait(lambda: owned and owned[-1] is False), owned
            assert prof.ss is None and fs.stats["connections"] == 0
            # ~/start → 접속·init·스트리밍
            r = _call(probe, "/hongik_profiler_node/start")
            assert r.success, r.message
            assert _wait(lambda: owned and owned[-1] is True) and _wait(lambda: len(frames) >= 5)
            conns_after_start = fs.stats["connections"]
            assert conns_after_start >= 1
            # ~/stop → 정지·원본 복원·접속 해제
            r = _call(probe, "/hongik_profiler_node/stop")
            assert r.success and "released" in r.message, r.message
            assert _wait(lambda: owned[-1] is False) and prof.ss is None
            n = len(frames)
            time.sleep(0.3)
            assert len(frames) == n                                     # 더 이상 프레임 없음
            # 이미 놓은 상태의 ~/stop 도 성공 (서버가 목표마다 부른다)
            r = _call(probe, "/hongik_profiler_node/stop")
            assert r.success and "already released" in r.message
            # 다시 ~/start 가능 (새 접속)
            r = _call(probe, "/hongik_profiler_node/start")
            assert r.success and fs.stats["connections"] > conns_after_start
            r = _call(probe, "/hongik_profiler_node/stop")
            assert r.success
        finally:
            prof.shutdown()
            ex.shutdown(timeout_sec=2.0)
            prof.destroy_node()
            probe.destroy_node()


class _StubServer:
    """AutoTeachServer._release_profiler 만 시험 (서버 전체는 플랫폼 설정이 필요하다)."""

    def __init__(self, node, name):
        self._release_cli = node.create_client(Trigger, name)
        self._node = node

    def get_logger(self):
        return self._node.get_logger()


@pytest.mark.parametrize("behaviour", ["absent", "ok", "refuse"])
def test_server_asks_the_profiler_node_to_release_the_sensor(default_ros, behaviour):
    from hongik_scan_ros.auto_teach_server import AutoTeachServer
    rclpy.init()
    srv_node = Node("fake_profiler")
    calls = []

    def on_stop(_req, res):
        calls.append(1)
        res.success = behaviour == "ok"
        res.message = "sensor released" if res.success else "busy (test)"
        return res

    if behaviour != "absent":
        srv_node.create_service(Trigger, "/hongik_fine_sensing/hongik_profiler_node/stop", on_stop)
    client_node = Node("server_stub")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(srv_node)
    ex.add_node(client_node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    try:
        stub = _StubServer(client_node, "/hongik_fine_sensing/hongik_profiler_node/stop")
        ok, why = AutoTeachServer._release_profiler(stub)
        if behaviour == "absent":
            assert ok and why == "no profiler node" and not calls
        elif behaviour == "ok":
            assert ok and "released" in why and calls
        else:
            assert not ok and "sensor busy" in why and "refused" in why and calls
    finally:
        ex.shutdown(timeout_sec=2.0)
        srv_node.destroy_node()
        client_node.destroy_node()


def test_action_cancel_stops_the_run_and_completes_the_run_cancel_future():
    """S4: 목표 취소 = 그 실행의 즉시 정지 (platform.latch.fire; 래치 없음) + 지금 목표의 취소 Future 를 끝낸다 (실행의 작업 프로세스 기다림이 바로 끝난다 — scan_l0.post_scan.wait_result);
    다시 취소해도·서버 종료도 같은 Future (한 번만 끝난다). 실행 중이 아닌 목표 (거절되어 abort 로 가는 PrecisionSensing 목표) 의 취소는
    아무것도 건드리지 않는다 (r3: 목표는 늘 수락 — 두 번째 목표가 잠깐 있을 수 있다)."""
    from concurrent.futures import Future
    from types import SimpleNamespace
    from hongik_scan_ros.auto_teach_server import AutoTeachServer
    from rclpy.action import CancelResponse
    fired, said = [], []
    active, other = object(), object()
    for kind in ("auto_teach", "precision_sensing"):
        fired.clear()
        stub = SimpleNamespace(_cancel=threading.Event(), _cancel_fut=Future(),
                               platform=SimpleNamespace(latch=SimpleNamespace(fire=lambda r, s: fired.append((r, s)))),
                               _lock=threading.Lock(), _goal_handle=active, _goal_kind=kind, _say=lambda m, level="info": said.append(m))
        stub._complete_cancel = lambda why, st=stub: AutoTeachServer._complete_cancel(st, why)
        assert AutoTeachServer._on_cancel(stub, other) == CancelResponse.ACCEPT              # not the running goal: nothing happens
        assert not stub._cancel.is_set() and not fired and not stub._cancel_fut.done()
        assert AutoTeachServer._on_cancel(stub, active) == CancelResponse.ACCEPT
        why = f"{kind} action cancel requested"
        assert stub._cancel.is_set() and fired == [(why, "action_cancel")]
        assert stub._cancel_fut.done() and stub._cancel_fut.result() == why
        AutoTeachServer._on_cancel(stub, active)                                 # 두 번째 취소: 오류 없이, 처음 사유 그대로
        AutoTeachServer._complete_cancel(stub, "auto_teach server shutting down")
        assert stub._cancel_fut.result() == why


def test_server_refuses_to_start_without_the_kiro_action_type():
    """welding_interfaces cannot be loaded → one clear English line with the fix on stderr, exit code 2, no traceback, nothing started."""
    import subprocess
    code = ("import sys; sys.modules['welding_interfaces'] = None; "
            "from hongik_scan_ros.auto_teach_server import main; main([])")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"),
                                                        os.path.join(STACK, "vc3dz_driver"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env)
    assert p.returncode == 2, (p.returncode, p.stdout, p.stderr)
    err = p.stderr.strip().splitlines()
    assert len(err) == 1 and "Traceback" not in p.stderr, p.stderr
    assert err[0].startswith("hongik_auto_teach_server: cannot load the KIRO action type welding_interfaces/action/PrecisionSensing")
    assert "INSTALL.md" in err[0] and "colcon_meta_ament_cmake.py" in err[0]
    assert "ros2 interface show welding_interfaces/action/PrecisionSensing" in err[0]
    from hongik_scan_ros.auto_teach_server import kiro_action_type
    t, why = kiro_action_type()                                     # in this (overlay) environment it loads
    assert t is not None and why is None, why


def test_server_graph_is_sensor_io_and_the_actions_only(default_ros):
    """The server (constructed in its own process: isolated domain, the virtual-arm platform config, nothing to connect to) offers only
    the sensor I/O and the action communication: publishers ~/profile and ~/log, services ~/confirm_move and ~/status, the actions
    precision_sensing and auto_teach. It subscribes to nothing that stops or configures a robot controller (no stop-request topic,
    no controller status), and its MoveIt adapter calls no controller-manager service."""
    import json
    import subprocess
    cfg = os.path.join(STACK, "scan_l0", "tests", "data", "hongik_rb20_fake_moveit.yaml")
    calib = os.path.join(STACK, "eoat_description", "calib", "eoat_lab_01.yaml")
    code = f"""
import json, os, sys
import rclpy
rclpy.init(args=["--ros-args", "-p", "platform_config:={cfg}", "-p", "calib_file:={calib}"])
from hongik_scan_ros.auto_teach_server import AutoTeachServer
s = AutoTeachServer()
ad = s.platform.ctx.node
print(json.dumps({{"pub": sorted(p.topic_name for p in s.publishers), "srv": sorted(x.srv_name for x in s.services),
                  "sub": sorted(x.topic_name for x in s.subscriptions),
                  "ad_sub": sorted(x.topic_name for x in ad.subscriptions), "ad_cli": sorted(c.srv_name for c in ad.clients),
                  "mods": sorted(m for m in sys.modules if m.split(".")[0] in ("rb_cb_client", "rbpodo_msgs", "controller_manager_msgs"))}}),
      flush=True)
os._exit(0)
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"),
                                                        os.path.join(STACK, "vc3dz_driver"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert p.returncode == 0, (p.returncode, p.stdout[-2000:], p.stderr[-4000:])
    got = json.loads(p.stdout.strip().splitlines()[-1])
    ns = "/hongik_fine_sensing/hongik_auto_teach_server" if any(t.startswith("/hongik_fine_sensing/") for t in got["pub"]) else \
        "/hongik_auto_teach_server"
    own_pub = [t for t in got["pub"] if not t.startswith(("/rosout", "/parameter_events"))]
    assert sorted(own_pub) == sorted([f"{ns}/profile", f"{ns}/log"]), got["pub"]
    srv = [ns + x[1:] if x.startswith("~/") else "/" + x.lstrip("/") for x in got["srv"]]      # rclpy keeps '~/' unexpanded here
    assert f"{ns}/confirm_move" in srv and f"{ns}/status" in srv, got["srv"]
    assert not [t for t in got["sub"] + got["ad_sub"] if "estop" in t or "system_state" in t or "robot_status" in t], got
    assert sorted(got["ad_cli"]) == ["/apply_planning_scene", "/check_state_validity", "/compute_cartesian_path", "/compute_fk",
                                     "/compute_ik", "/get_planning_scene",
                                     "/joint_trajectory_controller/follow_joint_trajectory/_action/cancel_goal",
                                     "/plan_kinematic_path"], got["ad_cli"]            # move_group + our JTC goal only
    assert sorted(got["ad_sub"]) == ["/joint_states", "/joint_trajectory_controller/controller_state", "/robot_description", "/tf",
                                     "/tf_static"], got["ad_sub"]
    assert got["mods"] == [], got["mods"]


def test_server_log_lines_are_published_one_per_message():
    """~/log: every line on its own message (multi-line text split), nothing merged or dropped (r2 §4-1)."""
    from types import SimpleNamespace
    from hongik_scan_ros.auto_teach_server import AutoTeachServer
    got = []
    stub = SimpleNamespace(_log_lock=threading.Lock(), pub_log=SimpleNamespace(publish=lambda m: got.append(m.data)))
    AutoTeachServer._publish_lines(stub, "[dwell] S grab 1: ...\nsecond line\n\nfourth")
    AutoTeachServer._publish_lines(stub, "single")
    assert got == ["[dwell] S grab 1: ...", "second line", "", "fourth", "single"]


def test_server_say_logs_any_severity_sequence(default_ros):
    """_say with changing severities on a real rclpy logger (rclpy refuses a call site whose severity changes between calls - the first
    e2e run of the PrecisionSensing server died on 'precision_sensing result' after an info line from the same call site)."""
    from types import SimpleNamespace
    from hongik_scan_ros.auto_teach_server import AutoTeachServer
    rclpy.init()
    node = Node("say_probe")
    got = []
    try:
        stub = SimpleNamespace(get_logger=node.get_logger, _log_lock=threading.Lock(),
                               pub_log=SimpleNamespace(publish=lambda m: got.append(m.data)))
        stub._publish_lines = lambda text: AutoTeachServer._publish_lines(stub, text)
        for i, level in enumerate(("info", "warn", "error", "info", "warn")):
            AutoTeachServer._say(stub, f"line {i}", level)
        assert got == [f"line {i}" for i in range(5)]
    finally:
        node.destroy_node()


def test_client_ctrl_c_cancels_the_goal_on_the_server(default_ros):
    """hongik_auto_teach_client (--teach 없이) 의 Ctrl-C → 서버에 목표 취소 (= 래치 정지) 가 닿고 결과가 곧 온다 (2026-10-08: 기본 rclpy SIGINT
    처리기가 컨텍스트를 내려 취소가 가지 않았다 — 로봇은 계속 움직였을 것). 가짜 액션 서버 (프로세스 내) + 실제 클라이언트 (자식 프로세스)."""
    import signal
    import subprocess
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from hongik_scan_msgs.action import AutoTeach
    from scan_l0 import config as C
    rclpy.init()
    node = Node("fake_auto_teach_server", namespace="/hongik_fine_sensing")
    rec = {"goals": 0, "cancel": 0, "canceled": 0}

    def execute(gh):
        rec["goals"] += 1
        t_end = time.monotonic() + 60.0
        while time.monotonic() < t_end and not gh.is_cancel_requested:
            time.sleep(0.02)
        res = AutoTeach.Result()
        if gh.is_cancel_requested:
            gh.canceled()
            rec["canceled"] += 1
            res.status, res.reason = "ABORTED", "cancelled"
        else:
            gh.abort()
            res.status, res.reason = "ERROR", "no cancel within 60 s"
        return res

    def on_cancel(_gh):
        rec["cancel"] += 1
        return CancelResponse.ACCEPT

    cb = ReentrantCallbackGroup()
    srv = ActionServer(node, AutoTeach, "/hongik_fine_sensing/auto_teach", execute_callback=execute,
                       goal_callback=lambda _g: GoalResponse.ACCEPT, cancel_callback=on_cancel, callback_group=cb)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    proc = None
    try:
        job = str(C.JOBS_DIR / "groove_10T_1G.yaml")
        code = ("import sys; from hongik_scan_ros.auto_teach_client import main; "
                f"sys.exit(main(['--job', {job!r}, '--out', 'ctrl_c']))")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"),
                                                            os.path.join(STACK, "vc3dz_driver"), os.environ.get("PYTHONPATH", "")]))
        proc = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True, text=True)
        out = []

        def reader():
            for line in proc.stdout:
                out.append(line)

        rt = threading.Thread(target=reader, daemon=True)
        rt.start()
        assert _wait(lambda: rec["goals"] >= 1, 30.0), "".join(out)
        time.sleep(0.5)
        os.killpg(proc.pid, signal.SIGINT)                                 # 터미널의 Ctrl-C (프로세스 그룹에 SIGINT)
        assert _wait(lambda: rec["cancel"] >= 1, 10.0), "cancel never reached the server: " + "".join(out)
        assert proc.wait(timeout=20.0) == 1
        rt.join(timeout=2.0)
        text = "".join(out)
        assert rec["canceled"] == 1 and "Ctrl-C: cancelling the goal" in text, text
        assert "RESULT: ABORTED - cancelled" in text, text
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        ex.shutdown(timeout_sec=2.0)
        srv.destroy()
        node.destroy_node()
