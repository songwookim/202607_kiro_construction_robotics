"""ROS 시험 하네스 — 시험 그래프를 격리된 ROS 도메인에서 띄우고 정리한다 (이 패키지의 노드 시험과 개발 저장소의 가짜 MoveIt 종단 시험
dev/kiro_e2e 가 함께 쓴다; 가짜 로봇 launch 와 mock 하드웨어 확인은 그쪽에 있다).

절대 안전 규칙 (2026-10-01): 실로봇·실센서 접속 금지 (FORBIDDEN_HOSTS).
  * 모든 프로세스는 ROS_LOCALHOST_ONLY=1 과 다른 어떤 실행과도 나눠 쓰지 않는 ROS_DOMAIN_ID (pick_domain, ros_domains) 로,
    자기 프로세스 그룹 (start_new_session) 으로 띄우고 끝나면 그룹 전체를 SIGINT → SIGKILL 로 정리한다.
ROS/MoveIt/우리 패키지가 없으면 ros_available() 이 사유를 돌려주고 시험은 깨끗이 skip 된다.
"""
from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional

import numpy as np

import ros_domains

FORBIDDEN_HOSTS = ("192.168.0.50", "192.168.3.15")


def ros_available(extra_packages=()) -> Optional[str]:
    """None = 사용 가능, 문자열 = skip 사유. extra_packages = 그 시험이 더 필요로 하는 ament 패키지 (예: 가짜 로봇의 MoveIt 설정)."""
    try:
        import rclpy  # noqa: F401
        from moveit_msgs.srv import GetCartesianPath  # noqa: F401
        from control_msgs.action import FollowJointTrajectory  # noqa: F401
        from hongik_scan_msgs.action import AutoTeach  # noqa: F401
        from ament_index_python.packages import get_package_share_directory
        for pkg in ("hongik_scan_ros", "eoat_description") + tuple(extra_packages):
            get_package_share_directory(pkg)
    except Exception as e:  # noqa: BLE001
        return f"ROS 2 / MoveIt / hongik packages not available: {type(e).__name__}: {e}"
    if shutil.which("ros2") is None:
        return "ros2 CLI not on PATH"
    return None


_handed: set = set()                  # 이 프로세스의 살아 있는 그래프에 준 도메인 (release_domain 으로 돌려받는다)


def _stop_own_leftovers() -> None:
    """실행기 없이 띄운 pytest 가 끝날 때: 이 프로세스가 띄우고 남은 것 (떨어져 나간 ros2 데몬 등) 을 멈춰 그 도메인을 비운다."""
    stopped = ros_domains.stop_processes(ros_domains.tagged_processes(os.environ.get(ros_domains.RUN_TAG, "")))
    if stopped:
        print(f"ros_harness: stopped {len(stopped)} left-over test process(es): {'; '.join(stopped)}", file=sys.stderr, flush=True)


def _tag_process_tree() -> None:
    """자식에게 물려줄 주인 표시 (ros_domains.RUN_TAG). 실행기가 이미 줬으면 그쪽이 대상마다 정리한다; 아니면 이 pytest 가 끝날 때 정리."""
    if not os.environ.get(ros_domains.RUN_TAG):
        os.environ[ros_domains.RUN_TAG] = f"{os.getpid()}:pytest"
        atexit.register(_stop_own_leftovers)


def pick_domain() -> int:
    """ROS 그래프 하나에 쓸 ROS_DOMAIN_ID — 다른 실행과도, 이 프로세스의 다른 살아 있는 그래프와도 나눠 쓰지 않는다.
    병렬 실행기 (scripts/run_tests.py) 가 작업자에게 잠가 준 도메인 (FS_TEST_DOMAINS) 부터 쓰고, 그것이 모두 쓰이는 중이면 (이 프로세스의 다른
    그래프, 또는 앞 시험이 남긴 프로세스) 전역 풀에서 하나 더 잠가 쓴다. 실행기 없이 띄운 pytest 는 처음부터 전역 풀. 비어 있는 것이 없으면 기다린다."""
    _tag_process_tree()
    own = ros_domains.parse_domains(os.environ.get("FS_TEST_DOMAINS", ""))
    if own:
        me = os.getpid()
        procs = ros_domains.scan_processes()
        for d in own:
            if d not in _handed and not any(dom == d for pid, (dom, _res) in procs.items() if pid != me):
                _handed.add(d)
                return d
    d = ros_domains.acquire_domain()
    _handed.add(d)
    return d


def release_domain(domain: int) -> None:
    """pick_domain 으로 받은 도메인을 돌려준다 (그 그래프의 프로세스를 다 정리한 뒤). 작업자 도메인은 실행기가 계속 쥐고, 전역 풀에서 더 받은 것은 잠금을 푼다."""
    _handed.discard(int(domain))
    ros_domains.release_domain(int(domain))


class IsolatedEnv:
    """os.environ 에 ROS_DOMAIN_ID / ROS_LOCALHOST_ONLY 를 걸고 (rclpy.init 전에) 끝에서 되돌리고 도메인을 돌려준다 (release_domain)."""

    def __init__(self, domain: int):
        self.domain = int(domain)
        self._saved: dict = {}

    def __enter__(self):
        for k, v in (("ROS_DOMAIN_ID", str(self.domain)), ("ROS_LOCALHOST_ONLY", "1"), ("DISPLAY", "")):
            self._saved[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        release_domain(self.domain)


class ProcessGroup:
    """명령 하나를 자기 세션/프로세스 그룹으로 띄우고 그룹 전체를 정리한다."""

    def __init__(self, cmd: list, log_path: str, env: Optional[dict] = None):
        for a in cmd:
            for h in FORBIDDEN_HOSTS:
                if h in str(a):
                    raise RuntimeError(f"refusing to start a process that names the real robot/sensor host {h}: {cmd}")
        self.cmd = list(cmd)
        self.log_path = log_path
        self._log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env or os.environ.copy(),
                                     start_new_session=True)
        self.pgid = os.getpgid(self.proc.pid)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout_s: float = 15.0) -> None:
        for sig, wait in ((signal.SIGINT, timeout_s), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(self.pgid, sig)
            except ProcessLookupError:
                break
            t_end = time.monotonic() + wait
            while time.monotonic() < t_end:
                self.proc.poll()          # 끝난 리더 (pytest 의 직계 자식) 를 거둔다 — 좀비면 killpg(pgid, 0) 이 그룹을 살아 있다고 본다
                if not _group_alive(self.pgid):
                    break
                time.sleep(0.1)
            self.proc.poll()
            if not _group_alive(self.pgid):
                break
        try:
            self.proc.wait(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        self._log.close()

    def tail(self, n: int = 60) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return ""


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# --------------------------------------------------------------------------------------
# rclpy 도우미 (시험 준비 전용 — 어댑터를 거치지 않는 관절 이동 / IK)
# --------------------------------------------------------------------------------------
def wait_robot_description(node, timeout_s: float = 60.0) -> str:
    import threading
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    got: dict = {}
    ev = threading.Event()

    def cb(m):
        got["x"] = m.data
        ev.set()

    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, "/robot_description", cb, qos)   # 파괴하지 않는다 (돌고 있는 executor 와의 경쟁 방지)
    ev.wait(timeout_s)
    return got.get("x", "")


def compute_ik(node, T, group: str, link: str, frame: str, joint_names, seed, timeout_s: float = 5.0):
    from moveit_msgs.srv import GetPositionIK
    from hongik_scan_ros.ros_util import call_service, matrix_to_pose
    cli = _client(node, GetPositionIK, "/compute_ik")
    if True:
        req = GetPositionIK.Request()
        req.ik_request.group_name = group
        req.ik_request.ik_link_name = link
        req.ik_request.pose_stamped.header.frame_id = frame
        req.ik_request.pose_stamped.pose = matrix_to_pose(T)
        req.ik_request.robot_state.joint_state.name = list(joint_names)
        req.ik_request.robot_state.joint_state.position = [float(v) for v in seed]
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 1
        r = call_service(cli, req, timeout_s, wait_service_s=10.0)
        if r is None or r.error_code.val != 1:
            return None
        names = list(r.solution.joint_state.name)
        return [float(r.solution.joint_state.position[names.index(n)]) for n in joint_names]


def move_joints(node, action: str, joint_names, positions, duration_s: float = 4.0, timeout_s: float = 30.0) -> bool:
    """시험 준비: JTC 로 관절 목표 (MoveIt 어댑터 를 거치지 않음)."""
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from hongik_scan_ros.ros_util import wait_future
    ac = _action_client(node, FollowJointTrajectory, action)
    if True:
        if not ac.wait_for_server(timeout_sec=20.0):
            return False
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = list(joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in positions]
        pt.velocities = [0.0] * len(positions)
        pt.time_from_start.sec = int(duration_s)
        pt.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1e9)
        g.trajectory.points = [pt]
        fut = ac.send_goal_async(g)
        if not wait_future(fut, 10.0) or not fut.result().accepted:
            return False
        rf = fut.result().get_result_async()
        return bool(wait_future(rf, timeout_s) and rf.result() is not None and rf.result().status == 4)


_CLIENTS: dict = {}


def _client(node, srv_type, name: str):
    """노드별로 한 번 만든 서비스 클라이언트를 재사용한다 (시험 중 파괴하지 않는다: 돌고 있는 executor 와의 경쟁 방지)."""
    key = (id(node), name)
    if key not in _CLIENTS or _CLIENTS[key][0] is not node:
        _CLIENTS[key] = (node, node.create_client(srv_type, name))
    return _CLIENTS[key][1]


def _action_client(node, action_type, name: str):
    from rclpy.action import ActionClient
    key = (id(node), "action:" + name)
    if key not in _CLIENTS or _CLIENTS[key][0] is not node:
        _CLIENTS[key] = (node, ActionClient(node, action_type, name))
    return _CLIENTS[key][1]


def rot_deg(Ra, Rb) -> float:
    R = np.asarray(Ra, dtype=float).T @ np.asarray(Rb, dtype=float)
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return float(np.degrees(np.arccos(c)))


# --------------------------------------------------------------------------------------
# fake sensor on a real clock (MoveIt fake-arm e2e, KIRO stand-in): stamping and content like the real sensor (2026-10-13)
# --------------------------------------------------------------------------------------
# Why. A frame_fn that reads TF 'latest' when the fake sensor calls it renders the pose of the render, not of the capture: on the
# 2026-10-13 fake-arm e2e the content trailed the host's capture time by 20–40 ms, varying per frame (the 100 Hz /joint_states step,
# delivery, and the clock sync of a send time that included the render: 1 ms at rest, 34 ms moving) — and the host calibration
# described the REAL sensor (edge response σ_L 0.47 mm, dt_s of the lab), not the exact ray cast. The end readings then scattered
# by ~1 mm per crossing and sat 0.3–0.6 mm outward (KIRO stand-in r13: E +1.38 mm). With the pieces below the fake stream is what
# the estimator models: content = the physical arm at the stamped capture instant, a fixed processing latency the host's dt_s
# knows, the fake's own edge response.
def fake_sensor_kwargs(fps: float) -> dict:
    """FakeSensor arguments of a real-clock fake sensor that stamps like the real one: the capture instants on its own grid, a late
    capture reported as a capture stall in the stamps (stall_late_ms, as sim_harness.SENSOR_REALTIME), frame_fn told the capture
    instant (timed_frame_fn), and a fixed processing latency of two frame periods. A real sensor's processing time is fixed, so its
    frames leave a fixed time after their capture (what the host calibrates as dt_s; the host's clock takes the lower envelope of the
    receptions). The fake's processing is the render, in the capture loop: one that runs past a period stalls the next capture
    (reported in the stamps), so a render takes up to about two periods (its own and the stalled one) — two periods of latency is the
    smallest that every such frame still meets. With one period, the CPU load of 2026-10-13 sent most moving frames late and the
    envelope clock put a pass up to 20 ms late (test_m: actual offset 25.6 ms at rest, 29–45 ms in the passes)."""
    return {"stall_late_ms": 4.0, "latency_ms": 2000.0 / float(fps), "timed_frame_fn": True}


def fake_sensor_calib_file(src: str, dst: str, latency_ms: float) -> str:
    """The host calibration of the fake sensor: the calibration file src with the fake's processing latency as its time offset (dt_s
    and the 1G block's g1.dt_s), the fake's edge response (groove_world.FAKE_SENSOR_EDGE_RESPONSE: an exact ray cast) and no END
    noise priors: src's g1.noise_priors end_crossing / end_reading are the lab robot + sensor population's end scatter (real edge
    response σ_L 0.47 mm at 150 fps); the fake's is the exact step sampled at its frame spacing (20 fps at 30 mm/s: 1.5 mm, shared by
    the two plates of a crossing, σ ≈ 0.4 mm), which no calibration describes — the end fusion then estimates it (uncalibrated prior).
    The leg-offset priors (the robot's pass-to-pass scatter: the fake arm's is smaller — conservative) and the speed range they hold
    stay. Every other entry (hand-eye, ...) as in src. Written to dst; returns dst."""
    import yaml
    from groove_world import FAKE_SENSOR_EDGE_RESPONSE
    with open(src, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    d["dt_s"] = float(latency_ms) * 1e-3
    g1 = dict(d.get("g1") or {})
    g1["dt_s"] = float(latency_ms) * 1e-3
    g1["edge_response"] = dict(FAKE_SENSOR_EDGE_RESPONSE)
    npri = dict(g1.get("noise_priors") or {})
    for c in ("end_crossing", "end_reading"):
        npri.pop(c, None)
    if npri:
        npri["note_fake_sensor"] = "end_crossing / end_reading removed: the fake sensor's end scatter is not the lab's"
        g1["noise_priors"] = npri
    else:
        g1.pop("noise_priors", None)
    d["g1"] = g1
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    return dst


class TfPoseAt:
    """GrooveWorld pose_at_fn: the physical flange pose at a capture instant (the fake sensor's monotonic clock) = TF base <- frame
    interpolated at that instant on the stamps' (wall) clock, lag_s earlier — the servo lag by which a physical arm trails a
    joint-state echo (the mock hardware's /joint_states is its command: the physical arm of a real echo trails it, lab CB
    2026-10-12 one 10–12 ms tick). Waits (timeout_s) for the stamps that bracket the instant; never extrapolates. The buffer's
    listener must be spun by another thread (RosContext). trace: (capture monotonic ns, query wall ns) per call."""

    def __init__(self, buf, base: str, frame: str, lag_s: float = 0.0, timeout_s: float = 2.0):
        self.buf, self.base, self.frame = buf, base, frame
        self.lag_ns = int(round(float(lag_s) * 1e9))
        self.timeout_s = float(timeout_s)
        self.trace: list = []

    def __call__(self, t_capture_ns: int) -> np.ndarray:
        from rclpy.duration import Duration
        from rclpy.time import Time
        from hongik_scan_ros.ros_util import transform_to_matrix
        t_q = int(t_capture_ns) + (time.time_ns() - time.monotonic_ns()) - self.lag_ns
        tr = self.buf.lookup_transform(self.base, self.frame, Time(nanoseconds=t_q), timeout=Duration(seconds=self.timeout_s))
        self.trace.append((int(t_capture_ns), t_q))
        return transform_to_matrix(tr)


def true_time_offset(fs, out_dir: str, lag_s: float) -> dict:
    """The run's ACTUAL time offset per profile of a fake-sensor run (FakeSensor(timed_frame_fn=True) + TfPoseAt): the host's capture
    time t_acq (verification/legs/*/recon.npz) minus the instant its content shows — the capture instant (fs.capture_log, the same
    monotonic clock) lag_s earlier, on the host's pose axis (the server maps a /joint_states stamp to stamp + the delivery's lower
    bound: raw.npz robot_t_rx_ns − robot_stamp_ns plus this host's wall − monotonic offset). → {'legs': {leg: (median, p10, p90)
    [s]}, 'moving_median_s', 'n'} over the profiles whose result counter is unique in the capture log."""
    import glob
    cl = np.asarray(fs.capture_log, dtype=np.int64).reshape(-1, 6)
    by_rc: dict = {}
    for row in cl:
        by_rc.setdefault(int(row[2]), []).append(int(row[4]))
    wall_minus_mono = time.time_ns() - time.monotonic_ns()
    legs, moving = {}, []
    for d in sorted(glob.glob(os.path.join(out_dir, "verification", "legs", "*"))):
        try:
            r = np.load(os.path.join(d, "recon.npz"))
            z = np.load(os.path.join(d, "raw.npz"))
        except OSError:
            continue
        if not z["robot_stamp_ns"].size:
            continue
        l_min = float(np.median(z["robot_t_rx_ns"].astype(float) - z["robot_stamp_ns"].astype(float))) + wall_minus_mono
        dl = np.array([float(r["t_acq_s"][k]) - ((by_rc[int(c)][0] + l_min) * 1e-9 - float(lag_s))
                       if len(by_rc.get(int(c), ())) == 1 else np.nan for k, c in enumerate(r["result_counter"])])
        ok = np.isfinite(dl)
        if not ok.any():
            continue
        name = os.path.basename(d)
        legs[name] = tuple(float(v) for v in np.percentile(dl[ok], [50, 10, 90]))
        if "pass" in name or "transit" in name:
            sp = np.asarray(r["speed_mps"], dtype=float)
            moving.extend(dl[ok & (np.abs(sp) > 0.5 * np.nanmax(np.abs(sp)))].tolist() if np.isfinite(sp).any() else [])
    return {"legs": legs, "moving_median_s": float(np.median(moving)) if moving else float("nan"), "n": len(moving)}
