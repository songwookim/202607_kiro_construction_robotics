"""The MoveIt adapter's pose stream (RosPoseSource) on a ROS graph with TF arriving 0.8 s late (user approval 2026-10-13: KIRO's weld logs
show TF reaching their nodes ~0.8 s late while /joint_states is fresh): the pose of every /joint_states message is FK of its joints with
the robot_description, available as soon as the message is — it never waits for TF; TF, once it arrives, equals it. Without a
robot_description the stream fails clearly (no TF fallback); a robot_description that is not the model behind the TF tree is caught by
the session's one-time cross-check; static transforms are read once, verified static and against the URDF, cached.

Graph (isolated ROS domain, ROS_LOCALHOST_ONLY): this process publishes /joint_states at 100 Hz (stamp = now) and the same messages 0.8 s
later on /joint_states_delayed; robot_state_publisher (lab RB20 + EOAT URDF, publish_frequency 1000 Hz so it publishes every message)
listens to the delayed topic — its TF is 0.8 s late, its /robot_description (transient local) is the model. No robot, no MoveIt."""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STACK = os.path.dirname(PKG)
for p in (HERE, PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"), os.path.join(STACK, "vc3dz_driver")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ros_harness import IsolatedEnv, ProcessGroup, pick_domain, ros_available  # noqa: E402

SKIP = ros_available()
pytestmark = pytest.mark.skipif(SKIP is not None, reason=SKIP or "")

URDF = os.path.join(STACK, "scan_l0", "tests", "data", "rb20_1900es_u_eoat.urdf")
FAKE_CFG = os.path.join(STACK, "scan_l0", "tests", "data", "hongik_rb20_fake_moveit.yaml")   # the fake-arm config is test data here
NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
Q0 = np.array([0.2, 0.1, 2.0, 0.3, 1.4, -0.5])
TF_DELAY_S = 0.8
RATE_HZ = 100.0


def _q(t: float) -> np.ndarray:
    """A smooth motion of all six joints (0.5 Hz, up to 0.25 rad)."""
    return Q0 + np.array([0.25, 0.1, -0.2, 0.15, 0.2, 0.25]) * math.sin(2.0 * math.pi * 0.5 * t)


class _JointStateFeed:
    """/joint_states now + the same message on /joint_states_delayed TF_DELAY_S later (robot_state_publisher's input)."""

    def __init__(self, node):
        from sensor_msgs.msg import JointState
        self.JointState = JointState
        self.node = node
        self.pub = node.create_publisher(JointState, "/joint_states", 50)
        self.pub_late = node.create_publisher(JointState, "/joint_states_delayed", 50)
        self.late: deque = deque()
        self.sent: dict = {}                     # stamp ns -> joint values
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.t0 = time.monotonic()

    def start(self):
        self.th.start()
        return self

    def stop(self):
        self._stop.set()
        self.th.join(2.0)

    def _run(self):
        period = 1.0 / RATE_HZ
        nxt = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= nxt:
                m = self.JointState()
                m.header.stamp = self.node.get_clock().now().to_msg()
                m.name = list(NAMES)
                q = _q(now - self.t0)
                m.position = [float(v) for v in q]
                self.pub.publish(m)
                self.sent[int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec)] = q
                self.late.append((now + TF_DELAY_S, m))
                nxt += period
            while self.late and self.late[0][0] <= now:
                self.pub_late.publish(self.late.popleft()[1])
            time.sleep(0.001)


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from hongik_scan_ros.moveit_adapter import RosPoseSource
    from hongik_scan_ros.platform_config import MoveItJtcConfig
    from hongik_scan_ros.ros_util import RosContext
    from rclpy.callback_groups import ReentrantCallbackGroup
    from scan_l0 import config as C
    logdir = tmp_path_factory.mktemp("pose_fk_ros")
    domain = pick_domain()
    env = IsolatedEnv(domain)
    env.__enter__()
    rclpy.init()
    feed_node = Node("pose_fk_test_feed")
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(feed_node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    mj = MoveItJtcConfig.from_params(C.load_settings(FAKE_CFG).params)
    ctx = RosContext("pose_fk_test_pose")
    pose = RosPoseSource(ctx, mj, ReentrantCallbackGroup())
    feed = _JointStateFeed(feed_node).start()
    try:
        yield SimpleNamespace(pose=pose, mj=mj, feed=feed, feed_node=feed_node, logdir=logdir, ctx=ctx,
                              start_rsp=lambda: _start_rsp(logdir))
    finally:
        feed.stop()
        ctx.close()
        ex.shutdown(timeout_sec=1.0)
        feed_node.destroy_node()
        rclpy.shutdown()
        for g in _RSP:
            g.stop()
        env.__exit__(None, None, None)


_RSP: list = []


def _start_rsp(logdir):
    with open(URDF, encoding="utf-8") as f:
        urdf = f.read()
    params = os.path.join(str(logdir), "rsp.yaml")
    with open(params, "w", encoding="utf-8") as f:
        yaml.safe_dump({"robot_state_publisher": {"ros__parameters": {"robot_description": urdf, "publish_frequency": 1000.0}}}, f)
    g = ProcessGroup(["ros2", "run", "robot_state_publisher", "robot_state_publisher", "--ros-args", "--params-file", params,
                      "-r", "joint_states:=/joint_states_delayed"], os.path.join(str(logdir), "rsp.log"))
    _RSP.append(g)
    return g


def _wait(cond, timeout_s: float) -> bool:
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_1_without_a_robot_description_the_stream_fails_clearly(graph):
    """/joint_states flow but no robot_description is published: no pose sample, the readiness reason names robot_description, the
    pose port raises — nothing falls back to TF."""
    from scan_l0.ports import MotionError
    p = graph.pose
    assert _wait(lambda: p.n_js > 50, 20.0), "no /joint_states received"
    why = p.fk_problem()
    assert why and "no robot_description on /robot_description" in why, why
    assert p.history(0) == [] and p.n_resolved == 0 and p.drop_reasons.get("no robot model (robot_description)", 0) > 50
    with pytest.raises(MotionError, match="robot_description"):
        p.latest_pose()
    with pytest.raises(MotionError, match="robot_description"):
        p.model()


def test_2_the_pose_stream_does_not_wait_for_a_late_tf(graph):
    """robot_state_publisher starts (its TF 0.8 s behind /joint_states). Every message becomes a pose sample as soon as it arrives —
    while TF cannot yet answer at its stamp — and the samples equal TF at the same stamps once TF has caught up."""
    from rclpy.time import Time
    from hongik_scan_ros.ros_util import transform_to_matrix
    p, mj = graph.pose, graph.mj
    rsp = graph.start_rsp()
    m = p.model(timeout_s=60.0)
    assert m.joint_names == tuple(NAMES) and m.source == "topic /robot_description", (m.source, rsp.tail())
    assert _wait(lambda: p.fk_problem() is None, 10.0), p.fk_problem()
    seen: dict = {}                                   # stamp -> (wall-clock lag at first sight [s], TF could answer then)
    t_end = time.monotonic() + 3.0
    n0 = len(p.history(0))
    while time.monotonic() < t_end:
        for s in p.history(0)[n0:]:
            if s.stamp_ns not in seen:
                ok = p.buffer.can_transform(mj.base_frame, mj.flange_frame, Time(nanoseconds=s.stamp_ns))
                seen[s.stamp_ns] = ((time.time_ns() - s.stamp_ns) * 1e-9, bool(ok))
        time.sleep(0.005)
    lags = np.array([v[0] for v in seen.values()])
    tf_ready = np.array([v[1] for v in seen.values()])
    print(f"{len(seen)} samples in 3 s: availability lag median {np.median(lags) * 1e3:.1f} ms, max {lags.max() * 1e3:.1f} ms; "
          f"TF could answer at first sight for {tf_ready.sum()}")
    assert len(seen) >= 0.8 * 3.0 * RATE_HZ, len(seen)
    assert np.median(lags) < TF_DELAY_S / 8 and lags.max() < TF_DELAY_S / 2, (np.median(lags), lags.max())
    assert tf_ready.mean() < 0.05, tf_ready.mean()                                  # the samples did not come from TF
    xc = p.tf_cross_check()
    assert xc["checked"] and xc["ok"], xc
    assert TF_DELAY_S * 0.6 < xc["tf_behind_joint_states_s"] < TF_DELAY_S * 1.6, xc
    assert xc["position_diff_m"] < 1e-9 and xc["rotation_diff_deg"] < 1e-6, xc
    # TF has caught up: FK of each message = TF at its stamp (robot_state_publisher published every message)
    time.sleep(TF_DELAY_S + 0.4)
    worst_p, worst_r, n = 0.0, 0.0, 0
    for s in p.history(0)[n0:]:
        if s.stamp_ns not in seen:
            continue
        T_tf = transform_to_matrix(p.buffer.lookup_transform(mj.base_frame, mj.flange_frame, Time(nanoseconds=s.stamp_ns)))
        dp, dr = (float(np.linalg.norm(T_tf[:3, 3] - s.T_base_flange[:3, 3])),
                  float(np.max(np.abs(T_tf[:3, :3] - s.T_base_flange[:3, :3]))))
        worst_p, worst_r, n = max(worst_p, dp), max(worst_r, dr), n + 1
        q_sent = graph.feed.sent.get(s.stamp_ns)
        assert q_sent is not None and np.array_equal(np.asarray(s.joints), q_sent)       # the sample keeps the message's joints
    print(f"FK vs TF at the same stamps: {n} samples, max position {worst_p:.2e} m, max rotation element {worst_r:.2e}")
    assert n >= 0.8 * len(seen) and worst_p < 1e-9 and worst_r < 1e-9
    rec = p.record()
    assert rec["method"] == "fk_from_joint_states" and rec["urdf_source"] == "topic /robot_description" and rec["n_pose_samples"] > 0


def test_3_static_transforms_are_read_once_verified_and_cached(graph):
    from geometry_msgs.msg import TransformStamped
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    from scan_l0.ports import MotionError
    p, mj = graph.pose, graph.mj
    p.model(timeout_s=60.0)
    T = p.static_transform("tcp", "eoat_base_link", 10.0)                           # in the URDF: fixed joints only
    Tu, mov = p.model().fixed_transform("tcp", "eoat_base_link")
    assert mov == [] and np.max(np.abs(T - Tu)) < 1e-12
    rec = p.record()["static_transforms"]["tcp <- eoat_base_link"]
    assert rec["urdf_checked"] is True and rec["tf_path"] == ["tcp -> eoat_base_link"], rec
    assert np.array_equal(p.static_transform("tcp", "eoat_base_link", 0.0), T)      # cached: no wait, no TF
    with pytest.raises(MotionError, match="is not static"):
        p.static_transform(mj.base_frame, mj.flange_frame, 5.0)                     # a moving frame is FK, never read from TF
    # a frame outside the URDF (world -> link0 like the lab launch's static_transform_publisher)
    br = StaticTransformBroadcaster(graph.feed_node)
    t = TransformStamped()
    t.header.stamp = graph.feed_node.get_clock().now().to_msg()
    t.header.frame_id, t.child_frame_id = "world", "link0"
    t.transform.rotation.w = 1.0
    br.sendTransform(t)
    Tw = p.static_transform("link0", "world", 10.0)
    assert np.allclose(Tw, np.eye(4)) and p.record()["static_transforms"]["link0 <- world"]["urdf_checked"] is False


def test_4_a_robot_description_that_is_not_the_tf_trees_model_is_caught(graph):
    """Our model from another URDF (the rbpodo upstream regression: shoulder origin 0 0 0 instead of 0 -0.2545 0) while the robot's TF comes
    from the right one: the one-time cross-check at TF's newest stamp shows the difference (session_init then refuses: not ready)."""
    import rclpy  # noqa: F401
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from hongik_scan_ros.moveit_adapter import RosPoseSource
    from hongik_scan_ros.ros_util import RosContext
    with open(URDF, encoding="utf-8") as f:
        text = f.read()
    i = text.index('<joint name="shoulder"')
    j = text.index("</joint>", i)
    wrong = text[:i] + text[i:j].replace('xyz="0 -0.2545 0"', 'xyz="0 0 0"') + text[j:]
    assert wrong != text
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = graph.feed_node.create_publisher(String, "/robot_description_other", qos)
    pub.publish(String(data=wrong))
    if not _RSP:
        graph.start_rsp()
    mj2 = dataclasses.replace(graph.mj, robot_description_topic="/robot_description_other")
    ctx2 = RosContext("pose_fk_test_pose_other")
    try:
        p2 = RosPoseSource(ctx2, mj2, ReentrantCallbackGroup())
        p2.model(timeout_s=30.0)
        assert _wait(lambda: p2.fk_problem() is None, 10.0), p2.fk_problem()
        assert _wait(lambda: p2.tf_cross_check().get("checked"), TF_DELAY_S + 10.0), p2.tf_cross_check()
        xc = p2.tf_cross_check()
        print(xc)
        assert not xc["ok"] and xc["position_diff_m"] > 0.1, xc
    finally:
        ctx2.close()
