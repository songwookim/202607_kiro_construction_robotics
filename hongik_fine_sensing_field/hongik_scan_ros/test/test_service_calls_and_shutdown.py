"""Service calls, stops and the server's start / shutdown (KIRO field 2026-10-14) without move_group: a planning request is sent
once and a silent service ends in PlatformUnavailable with the pending request removed; a stop ends a wait within one spin period;
the server's shutdown waits for the running goal (its MoveIt clean-up) before closing the platform; a cancelled or shut-down run
reports ABORTED; objects of ours left in the MoveIt scene are removed at start; the start-up record and the numeric fail-fast.
Fake clients / platforms (no ROS graph) except the two server-process tests (isolated domain, nothing to connect to)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STACK = os.path.dirname(PKG)
for p in (PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"), os.path.join(STACK, "vc3dz_driver")):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("rclpy")
from hongik_scan_ros import ros_util as RU  # noqa: E402
from scan_l0.executor import CANCEL_OPERATOR, CANCEL_SERVER_SHUTDOWN  # noqa: E402
from scan_l0.motion_types import MotionError, PlatformUnavailable  # noqa: E402

ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"),
                                                    os.path.join(STACK, "vc3dz_driver"), os.environ.get("PYTHONPATH", "")]))


class FakeClient:
    """A service client: offered or not, answers after `delay_s` (None = never), records sends and removed requests."""

    def __init__(self, name="/plan_kinematic_path", offered=True, delay_s=None, response="answer"):
        self.srv_name, self.offered, self.delay_s, self.response = name, offered, delay_s, response
        self.sent, self.removed = [], []

    def wait_for_service(self, timeout_sec=None):
        return self.offered

    def call_async(self, req):
        fut = Future()
        self.sent.append((time.monotonic(), req))
        if self.delay_s is not None:
            threading.Timer(self.delay_s, lambda: fut.done() or fut.set_result(self.response)).start()
        return fut

    def remove_pending_request(self, fut):
        self.removed.append(fut)


def test_a_planning_request_is_sent_once_and_a_silent_service_ends_in_platform_unavailable():
    """KIRO 3.5: a 5 s plan re-sent every 3 s queued ~10 copies in move_group. resend=False: one request for the whole timeout; no
    answer → the pending request is removed and PlatformUnavailable (a MotionError) names the service and the wait."""
    c = FakeClient()
    t0 = time.monotonic()
    with pytest.raises(PlatformUnavailable, match=r"/plan_kinematic_path did not answer within 0\.6 s \(request sent once"):
        RU.call_service(c, "req", 0.6, attempt_timeout_s=0.2, resend=False)
    assert len(c.sent) == 1 and len(c.removed) == 1 and time.monotonic() - t0 >= 0.6
    assert issubclass(PlatformUnavailable, MotionError)
    c = FakeClient("/check_state_validity")                         # cheap idempotent query: re-sent inside the timeout (first-contact loss)
    with pytest.raises(PlatformUnavailable, match=r"/check_state_validity did not answer within 0\.65 s \(request sent 4 times"):
        RU.call_service(c, "req", 0.65, attempt_timeout_s=0.2)
    assert len(c.sent) == 4 and len(c.removed) == 4
    assert RU.call_service(FakeClient(delay_s=0.05), "req", 2.0, resend=False) == "answer"
    assert RU.call_service(FakeClient(offered=False), "req", 2.0) is None          # not offered: the caller decides (unchanged)


def test_a_stop_ends_a_service_wait_within_one_spin_period_and_nothing_is_sent_while_it_holds():
    """A stop (the stopped run, the server shutting down) ends a wait within SPIN_PERIOD_S with MotionError('stopped ...') and the
    request withdrawn; while the stop holds no request is sent. The stop is found through the RosContext of the client's node; the
    planning-scene clients are exempt (a stopped run still removes its objects); a closing node stops every client."""
    stopped = {"why": None}
    c, scene = FakeClient(), FakeClient("/apply_planning_scene", delay_s=0.3)
    ctx = RU.RosContext(node=SimpleNamespace(clients=[c, scene]))
    ctx.add_stop_source(lambda: stopped["why"])
    ctx.exempt_from_stop(scene)
    threading.Timer(0.3, lambda: stopped.update(why="run stopped: precision_sensing action cancel requested")).start()
    t0 = time.monotonic()
    with pytest.raises(MotionError, match=r"^stopped \(run stopped: precision_sensing action cancel requested\) while waiting for "
                                          r"/plan_kinematic_path: request withdrawn"):
        RU.call_service(c, "req", 30.0, resend=False)
    dt = time.monotonic() - t0
    assert 0.3 <= dt <= 0.3 + 2 * RU.SPIN_PERIOD_S, dt
    assert len(c.sent) == 1 and len(c.removed) == 1
    with pytest.raises(MotionError, match=r"^stopped \(run stopped: .*\): /plan_kinematic_path not called"):
        RU.call_service(c, "req", 30.0, resend=False)
    assert len(c.sent) == 1                                          # nothing sent while the stop holds
    assert RU.call_service(scene, "req", 5.0) == "answer"           # exempt: the clean-up completes after the stop
    stopped["why"] = None
    ctx._closing = True                                              # the platform node is closing: every client, exempt or not
    with pytest.raises(MotionError, match="closing"):
        RU.call_service(scene, "req", 5.0)
    assert RU.call_service(FakeClient(delay_s=0.05), "req", 2.0) == "answer"      # a client of no context: no stop source


# ---- the server (stub: AutoTeachServer methods on an object without a ROS node) -------------------------------------------------
class FakePlatform:
    def __init__(self, scene_ids=(), events=None, offered=True):
        self.events = events if events is not None else []
        self.scene = list(scene_ids)
        self.fired = []
        self.closed = False
        self.mj = SimpleNamespace(get_scene_service="/get_planning_scene")
        plat = self

        class _Latch:
            is_set, reason = False, ""

            def fire(self, reason, source):
                plat.fired.append((reason, source))
                self.is_set, self.reason = True, reason

        self.latch = _Latch()

        class _SceneClient(FakeClient):
            def call_async(self, req):
                fut = Future()
                cos = [SimpleNamespace(id=i) for i in plat.scene]
                fut.set_result(SimpleNamespace(scene=SimpleNamespace(world=SimpleNamespace(collision_objects=cos))))
                return fut

        self.get_scene_client = _SceneClient("/get_planning_scene", offered=offered)

    def remove_scene_boxes(self, ids):
        self.events.append(("remove", sorted(ids), self.closed))
        self.scene = [i for i in self.scene if i not in ids]
        return True

    def close(self):
        self.events.append(("close",))
        self.closed = True


def _server(platform):
    from hongik_scan_ros.auto_teach_server import AutoTeachServer
    s = AutoTeachServer.__new__(AutoTeachServer)                     # no rclpy node: only the attributes these methods read
    s.said = []
    s._say = lambda line, level="info": s.said.append((level, line))
    s._shutdown, s._cancel, s._lock = threading.Event(), threading.Event(), threading.Lock()
    s._threads_cv, s._n_goal_threads = threading.Condition(), 0
    s._active, s._cancel_fut, s.platform = None, Future(), platform
    return s


def test_shutdown_waits_for_the_running_goal_before_closing_the_platform():
    """KIRO 3.3: the platform node was destroyed under a running goal (InvalidHandle, ERROR_INTERNAL, our boxes left in KIRO's scene).
    destroy(): the run is stopped like a cancel (latch + cancel Future = CANCEL_SERVER_SHUTDOWN), the server waits until the execute
    callback returned - the run's own clean-up (hongik/ objects) happens while the platform is open - then closes the platform."""
    plat = FakePlatform(scene_ids=["hongik/specimen", "hongik/floor", "kiro/table"])
    srv = _server(plat)
    srv._active = "precision_sensing"
    seen = {}

    def execute(_gh):                                                # a run: ends on its cancel Future, then its finally clears the cell
        seen["cause"] = srv._cancel_fut.result(timeout=10.0)
        time.sleep(0.3)                                              # the stop sequence / output writes of the run
        plat.remove_scene_boxes(["hongik/specimen", "hongik/floor"])
        return "result"

    th = threading.Thread(target=srv._goal_thread(execute), args=(object(),))
    th.start()
    assert _wait(lambda: srv.goal_threads() == 1)
    spins = []
    srv.destroy(spin=lambda: (spins.append(1), time.sleep(RU.SPIN_PERIOD_S)))
    th.join(1.0)
    assert not th.is_alive() and seen["cause"] == CANCEL_SERVER_SHUTDOWN and spins
    assert plat.fired == [(CANCEL_SERVER_SHUTDOWN, "server_shutdown")] and srv._cancel.is_set()
    assert plat.events == [("remove", ["hongik/floor", "hongik/specimen"], False), ("close",)], plat.events
    assert srv._claim("auto_teach", object()) == CANCEL_SERVER_SHUTDOWN                  # no new run after the shutdown began
    from hongik_scan_ros.auto_teach_server import claim_refusal
    assert claim_refusal(CANCEL_SERVER_SHUTDOWN) == "server shutting down: no new run (nothing sent)"
    assert claim_refusal("auto_teach goal running, phase pass") == \
        "busy: another sensing run is active (auto_teach goal running, phase pass; nothing sent)"
    # a goal that never returns: the second signal ends the waiting; objects of ours are still removed, then the platform closes
    plat = FakePlatform(scene_ids=["hongik/weld/pass_2_bead", "hongik/under"])
    srv = _server(plat)
    hold = threading.Event()
    th = threading.Thread(target=srv._goal_thread(lambda _gh: hold.wait(10.0)), args=(object(),), daemon=True)
    th.start()
    assert _wait(lambda: srv.goal_threads() == 1)
    srv.destroy(give_up=lambda: True)
    assert plat.events == [("remove", ["hongik/under", "hongik/weld/pass_2_bead"], False), ("close",)], plat.events
    assert any("not waiting for 1 running goal" in line for _lv, line in srv.said), srv.said
    hold.set()


def test_leftover_objects_of_ours_are_removed_and_nothing_else():
    """At start (and before every run): every MoveIt object under 'hongik/' (cell world + D14 weld scene) left by an earlier run is
    removed; objects of others stay; a scene that cannot be read is said, not fatal."""
    plat = FakePlatform(scene_ids=["hongik/specimen", "hongik/floor", "hongik/under", "hongik/weld/pass_1_bead", "kiro/fixture",
                                   "hongik_table"])
    srv = _server(plat)
    assert srv._clear_leftover_scene("server start") == ["hongik/floor", "hongik/specimen", "hongik/under", "hongik/weld/pass_1_bead"]
    assert plat.scene == ["kiro/fixture", "hongik_table"]
    assert srv.said[-1][0] == "warn" and srv.said[-1][1].startswith("MoveIt scene (server start): removed 4 object(s) of ours")
    n = len(srv.said)
    assert srv._clear_leftover_scene("before the run") == [] and len(srv.said) == n           # nothing left: no line
    srv = _server(FakePlatform(offered=False))
    assert srv._clear_leftover_scene("server start") == [] and "not offered" in srv.said[-1][1]


def test_a_cancelled_or_shut_down_run_reports_aborted_not_an_internal_error():
    """KIRO 3.3: InvalidHandle after the shutdown reached the generic handler → ERROR_INTERNAL. Now: after a stop from outside the
    exception is its consequence → ABORTED with the cause (no category, as a cancel); a silent planning service → motion category
    (REFUSED before the first motion); anything else stays ERROR / internal. The same rule on the server's side of a run."""
    from scan_l0.autoteach import AutoTeach
    from hongik_scan_ros.auto_teach_server import AutoTeachServer

    def run(cause=None, sent=False, latch=None):
        c = Future()
        if cause is not None:
            c.set_result(cause)
        be = SimpleNamespace(estopped=latch is not None, latch=SimpleNamespace(reason=latch or ""))
        st = SimpleNamespace(cancel=c, rs=SimpleNamespace(backend=be), _motion_sent=sent)
        st._external_stop = lambda: AutoTeach._external_stop(st)
        return st

    boom = RuntimeError("cannot use Destroyable because destruction was requested")
    assert AutoTeach._unexpected_failure(run(CANCEL_SERVER_SHUTDOWN), boom, " (D7)") == \
        ("ABORTED", "server shutting down (D7): RuntimeError: cannot use Destroyable because destruction was requested", None)
    assert AutoTeach._unexpected_failure(run("precision_sensing action cancel requested"), boom, " (D7)")[:2] == \
        ("ABORTED", f"{CANCEL_OPERATOR} (D7): RuntimeError: cannot use Destroyable because destruction was requested")
    assert AutoTeach._unexpected_failure(run(latch="server shutting down"), boom)[:2] == \
        ("ABORTED", "stopped (server shutting down): RuntimeError: cannot use Destroyable because destruction was requested")
    pu = PlatformUnavailable("/check_state_validity did not answer within 5 s (request sent twice)")
    assert AutoTeach._unexpected_failure(run(), pu) == \
        ("REFUSED", "PlatformUnavailable: /check_state_validity did not answer within 5 s (request sent twice) (nothing sent)", "motion")
    assert AutoTeach._unexpected_failure(run(sent=True), pu)[::2] == ("ABORTED", "motion")
    assert AutoTeach._unexpected_failure(run(), boom) == ("ERROR", f"RuntimeError: {boom}", "internal")
    srv = _server(FakePlatform())
    gh = SimpleNamespace(is_cancel_requested=False)
    assert AutoTeachServer._failure(srv, gh, boom) == ("ERROR", f"RuntimeError: {boom}", "internal")
    gh.is_cancel_requested = True
    assert AutoTeachServer._failure(srv, gh, boom) == ("ABORTED", f"{CANCEL_OPERATOR}: RuntimeError: {boom}", None)
    srv._shutdown.set()
    assert AutoTeachServer._failure(srv, gh, boom) == ("ABORTED", f"{CANCEL_SERVER_SHUTDOWN}: RuntimeError: {boom}", None)
    from scan_l0.teach_1g import _CancelAbort
    assert _CancelAbort("before 'pass_2'", CANCEL_SERVER_SHUTDOWN).reason == "server shutting down before 'pass_2' (D7)"
    assert _CancelAbort().reason == "cancel requested by the operator (D7)"


def _wait(pred, timeout=5.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ---- numeric packages ---------------------------------------------------------------------------------------------------------
def test_numeric_record_and_the_fail_fast_check():
    """KIRO NumPy 2: SciPy's compiled extensions cannot load against the loaded NumPy → one English line with the versions, the files
    and the fix (SystemExit, no traceback) before anything of the server is imported; a working pair passes and is recorded."""
    import numpy
    import scipy
    from hongik_scan_ros import numeric_env as NE
    assert NE.numeric_problem() is None
    rec = NE.require_numeric_stack("node")
    assert rec["numpy"] == {"version": numpy.__version__, "file": os.path.abspath(numpy.__file__)}
    assert rec["scipy"] == {"version": scipy.__version__, "file": os.path.abspath(scipy.__file__)}
    assert NE.numeric_record_text(rec) == (f"numpy {numpy.__version__} ({os.path.abspath(numpy.__file__)}), "
                                           f"scipy {scipy.__version__} ({os.path.abspath(scipy.__file__)})")

    def broken(mod):
        if mod == NE.CHECK_MODULE:
            raise ImportError("numpy.core.multiarray failed to import")
        return __import__(mod)
    assert NE.numeric_problem(broken) == "import scipy.spatial.transform: ImportError: numpy.core.multiarray failed to import"
    with pytest.raises(SystemExit) as ei:
        NE.require_numeric_stack("hongik_auto_teach_server", broken)
    msg = str(ei.value.code)
    assert msg.startswith("hongik_auto_teach_server: SciPy cannot load against the loaded NumPy (numpy ") and "\n" not in msg
    assert "numpy.core.multiarray failed to import" in msg and "INSTALL.md" in msg and "PYTHONNOUSERSITE=1" in msg
    code = ("import sys; sys.modules['scipy.spatial.transform'] = None; "    # the import fails as SciPy 1.8 does under NumPy 2
            "import hongik_scan_ros.auto_teach_server")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=ENV)
    err = p.stderr.strip().splitlines()
    assert p.returncode == 1 and len(err) == 1 and "Traceback" not in p.stderr, (p.returncode, p.stderr)
    assert err[0].startswith("hongik_auto_teach_server: SciPy cannot load against the loaded NumPy"), err


# ---- the server process: start-up record, SIGINT ------------------------------------------------------------------------------
def _server_args(tmp_path):
    cfg = os.path.join(STACK, "scan_l0", "tests", "data", "hongik_rb20_fake_moveit.yaml")
    calib = os.path.join(STACK, "eoat_description", "calib", "eoat_lab_01.yaml")
    return cfg, calib, ["--ros-args", "-p", f"platform_config:={cfg}", "-p", f"calib_file:={calib}", "-p",
                        f"out_root:={tmp_path / 'out'}", "-p", "kiro_confirm_mode:=none"]


@pytest.fixture
def domain_env():
    from ros_harness import pick_domain, release_domain
    d = pick_domain()
    yield dict(ENV, ROS_DOMAIN_ID=str(d), ROS_LOCALHOST_ONLY="1")
    release_domain(d)


def test_server_start_record_and_sigint_shutdown(tmp_path, domain_env):
    """The server process (isolated domain, virtual-arm config, no move_group): one line of the site parameters, one of the numeric
    packages loaded, the scene line (move_group not offered); SIGINT → 'server shutting down (SIGINT)', the platform closed, exit 0
    (rclpy's own handler no longer shuts the context down under a running goal)."""
    cfg, calib, args = _server_args(tmp_path)
    code = "import sys; from hongik_scan_ros.auto_teach_server import main; main(sys.argv[1:])"
    p = subprocess.Popen([sys.executable, "-u", "-c", code] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=domain_env, start_new_session=True)
    out = []
    rt = threading.Thread(target=lambda: [out.append(line) for line in p.stdout], daemon=True)
    rt.start()
    try:
        assert _wait(lambda: any("hongik_auto_teach_server ready" in line for line in out), 90.0), "".join(out)
        text = "".join(out)
        import numpy
        import scipy
        from scan_l0 import config as C
        job = str(C.JOBS_DIR / "groove_10T_1G.yaml")                 # the server's default job_file
        assert (f"site parameters: platform_config {cfg}, job_file {job}, calib_file {calib}, out_root {tmp_path / 'out'}, "
                "confirm_mode each, kiro_confirm_mode none") in text, text
        assert (f"numeric packages: numpy {numpy.__version__} ({os.path.abspath(numpy.__file__)}), scipy {scipy.__version__} "
                f"({os.path.abspath(scipy.__file__)})") in text, text
        assert "MoveIt scene (server start): /get_planning_scene not offered" in text, text
        os.killpg(p.pid, signal.SIGINT)
        assert p.wait(timeout=30.0) == 0, "".join(out)
        rt.join(2.0)
        text = "".join(out)
        assert "server shutting down (SIGINT)" in text and "server shutting down: no run active" in text, text
        assert "Traceback" not in text and "InvalidHandle" not in text, text
    finally:
        if p.poll() is None:
            p.kill()
