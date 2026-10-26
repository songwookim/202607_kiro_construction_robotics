"""The MoveIt adapter's pose stream by FK of /joint_states (hongik_scan_ros.pose_fk, user approval 2026-10-13) — the model itself (no
ROS graph): joint values matched by name, the chain from the URDF the lab launch runs, equality with what the TF path recorded on the
real arm, static paths. The ROS-level stream (delayed TF, missing robot_description) is test_pose_fk_ros.py; FK vs TF on the fake MoveIt
arm is test_moveit_integration.py (TestAdapterB)."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STACK = os.path.dirname(PKG)
for p in (PKG, os.path.join(STACK, "scan_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hongik_scan_ros.pose_fk import METHOD, FkPoseModel, pose_difference, tf_dynamic_edges  # noqa: E402

URDF = os.path.join(STACK, "scan_l0", "tests", "data", "rb20_1900es_u_eoat.urdf")      # the lab RB20 + EOAT URDF (xacro output)
FIXTURE = os.path.join(HERE, "data", "k30_1_a_joint_states_tf.npz")
LAB_RUN = os.path.expanduser("~/scan_data/hongik_auto_teach/k30_1_a/verification/legs")
ROBOT_XACRO = os.path.join(STACK, "eoat_description", "urdf", "rb20_1900es_u_eoat.xacro")
CALIB = os.path.join(STACK, "eoat_description", "calib", "eoat_lab_01.yaml")
NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


def _model(text=None, base="link0", flange="tcp"):
    if text is None:
        with open(URDF, encoding="utf-8") as f:
            text = f.read()
    return FkPoseModel(text, base, flange, "test")


def _rsp_published(st: np.ndarray, interval_ns: int) -> np.ndarray:
    """robot_state_publisher (Humble) publishes a joint state's TF only when its stamp >= the last published stamp + 1 / publish_frequency
    (it receives them in stamp order; our history holds them in our arrival order). A recorded leg starts with a published message."""
    pub = np.zeros(len(st), bool)
    last = None
    for i in np.argsort(st, kind="stable"):
        if last is None or st[i] >= last + interval_ns:
            pub[i] = True
            last = st[i]
    return pub


def _follows_rsp_rule(st: np.ndarray, pub: np.ndarray, interval_ns: int) -> bool:
    """Flags of a segment cut from inside a leg: in stamp order every published stamp is >= the previous published + interval and every
    skipped one is < it (messages before the segment's first published one cannot be judged here)."""
    last = None
    for i in np.argsort(st, kind="stable"):
        if last is None:
            last = st[i] if pub[i] else None
            continue
        if bool(pub[i]) != bool(st[i] >= last + interval_ns):
            return False
        if pub[i]:
            last = st[i]
    return True


def _tf_of_messages(fk, Q, st, pub) -> np.ndarray:
    """What the TF path returns at each message's stamp: FK of the message for a published one; for a skipped one tf2 interpolates each
    joint transform between the published neighbours — for revolute joints = FK of the linearly interpolated joint values."""
    o = np.argsort(st[pub], kind="stable")
    ps, pq = st[pub][o], Q[pub][o]
    out = np.full((len(st), 4, 4), np.nan)
    for i in range(len(st)):
        if pub[i]:
            out[i] = fk(Q[i])
            continue
        k = int(np.searchsorted(ps, st[i]))
        if 0 < k < len(ps):
            a = (st[i] - ps[k - 1]) / (ps[k] - ps[k - 1])
            out[i] = fk(pq[k - 1] + a * (pq[k] - pq[k - 1]))
    return out


def _max_abs(A, B) -> float:
    ok = np.all(np.isfinite(A), axis=(1, 2))
    return float(np.max(np.abs(A[ok] - B[ok]))) if np.any(ok) else float("nan")


# --------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------
def test_pose_of_a_message_is_fk_of_its_joints_matched_by_name():
    m = _model()
    assert m.joint_names == tuple(NAMES) and METHOD == "fk_from_joint_states"
    assert [j.type for j in m.chain.joints][-1] == "fixed"                         # tcp_joint (flange face) — fixed joints included
    q = np.array([0.3, -0.4, 2.1, 0.2, 1.2, -0.7])
    T_ref = m.chain.fk(q)
    # another publisher's order with extra joints (KIRO: both arms + head in one /joint_states)
    names = ["head_1", "wrist3", "base", "left_j1", "elbow", "wrist1", "shoulder", "wrist2"]
    vals = {"head_1": 9.0, "left_j1": -9.0, **dict(zip(NAMES, q))}
    T, why = m.pose(names, [vals[n] for n in names])
    assert why == "" and np.max(np.abs(T - T_ref)) < 1e-12
    T2, _ = m.pose(names, [vals[n] for n in names])                                 # cached index, same answer
    assert np.array_equal(T, T2)
    # what gives no sample (and why)
    assert m.pose(NAMES[:5], list(q[:5])) == (None, "joints missing")
    assert m.missing_joints(NAMES[:5]) == ["wrist3"]
    assert m.pose(NAMES, list(q[:5]))[1] == "position count differs from the name count"
    bad = list(q)
    bad[2] = float("nan")
    assert m.pose(NAMES, bad) == (None, "non-finite joint value")
    rec = m.record()
    assert rec["method"] == METHOD and rec["urdf_source"] == "test" and len(rec["urdf_sha256"]) == 64
    assert rec["joint_names"] == NAMES and rec["chain"][-1] == "tcp_joint (fixed)"


def test_the_model_refuses_what_joint_states_cannot_drive():
    with pytest.raises(ValueError, match="not a URDF"):
        _model("<robot><joint")
    with pytest.raises(ValueError, match="cannot reach"):
        _model(base="tcp", flange="link0")                                          # the flange must be below the base
    with open(URDF, encoding="utf-8") as f:
        text = f.read()
    floating = text.replace('<joint name="wrist2" type="revolute">', '<joint name="wrist2" type="floating">')
    assert floating != text
    with pytest.raises(ValueError, match=r"wrist2 \(floating\)"):
        _model(floating)


def test_static_paths_of_the_urdf():
    m = _model()
    T, mov = m.fixed_transform("tcp", "eoat_base_link")
    j = m.joints
    eo = [x for x in j.values() if x.child == "eoat_base_link"][0]
    assert mov == [] and eo.parent == "tcp" and np.max(np.abs(T - eo.origin())) < 1e-15
    Ti, _ = m.fixed_transform("eoat_base_link", "tcp")
    assert np.max(np.abs(Ti @ T - np.eye(4))) < 1e-12
    assert m.fixed_transform("link0", "tcp") == (None, NAMES)                       # a moving path is never "static"
    assert m.fixed_transform("tcp", "world") == (None, [])                          # not in this URDF
    assert np.array_equal(m.fixed_transform("tcp", "tcp")[0], np.eye(4))


def test_tf_static_and_dynamic_edges_from_the_tf2_tree():
    tf2_py = pytest.importorskip("tf2_py")
    from geometry_msgs.msg import TransformStamped
    b = tf2_py.BufferCore()

    def ts(parent, child, sec):
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = parent, child
        t.header.stamp.sec = sec
        t.transform.rotation.w = 1.0
        return t
    b.set_transform_static(ts("world", "link0", 0), "test")
    b.set_transform(ts("link0", "link1", 5), "test")
    b.set_transform(ts("link0", "link1", 6), "test")
    b.set_transform_static(ts("link1", "tcp", 3), "test")                           # a static transform's stamp is not kept
    b.set_transform_static(ts("tcp", "optical", 0), "test")
    frames = yaml.safe_load(b.all_frames_as_yaml())
    path, dyn = tf_dynamic_edges(frames, "tcp", "optical")
    assert path == [("optical", "tcp")] and dyn == []
    path, dyn = tf_dynamic_edges(frames, "link0", "world")
    assert path == [("link0", "world")] and dyn == []
    path, dyn = tf_dynamic_edges(frames, "world", "optical")
    assert dyn == [("link1", "link0")] and len(path) == 4
    with pytest.raises(ValueError, match="not connected"):
        tf_dynamic_edges(frames, "world", "elsewhere")


# --------------------------------------------------------------------------------------
# equality with the TF path on the real arm (lab run k30_1_a, 2026-10-12)
# --------------------------------------------------------------------------------------
def test_fk_reproduces_the_tf_poses_recorded_on_the_real_arm():
    """Fixture = segments of k30_1_a (dwell, transit, two passes, each pass with its one message that arrived out of order): per message
    the joints, the header stamp, and the pose the TF path recorded. robot_state_publisher (publish_frequency 100 Hz) published 216 of the
    349 messages; their TF pose is FK of the message's joints to 1e-15. The other 133 were tf2's interpolation between published
    neighbours (= FK of the interpolated joints, 1e-15) — up to ~0.2 mm from that message's own joints on the lab's stepped joint stream,
    which the FK stream no longer has."""
    d = np.load(FIXTURE)
    m = _model()
    assert list(d["joint_names"]) == NAMES
    Q, st, T, pub, seg = d["q"], d["stamp_ns"].astype(np.int64), d["T"], d["published"], d["segment"]
    for k in np.unique(seg):                                       # the flags follow robot_state_publisher's rule
        s = seg == k
        assert _follows_rsp_rule(st[s], pub[s], int(d["rsp_interval_ns"]))
    F = np.array([m.pose(NAMES, q)[0] for q in Q])
    assert pub.sum() == 216 and (~pub).sum() == 133
    assert _max_abs(F[pub], T[pub]) < 1e-12
    E = np.concatenate([_tf_of_messages(m.fk, Q[seg == k], st[seg == k], pub[seg == k]) for k in np.unique(seg)])
    assert _max_abs(E[~pub], T[~pub]) < 1e-12
    gap = max(pose_difference(F[i], T[i])[0] for i in np.where(~pub)[0])
    assert gap > 1e-5, gap                                        # the TF path's interpolation was not this message's pose


@pytest.mark.skipif(not os.path.isdir(LAB_RUN), reason="lab run k30_1_a not on this PC")
def test_fk_reproduces_every_tf_pose_of_the_lab_run():
    """The whole run (7 legs, 5142 messages): published -> FK of the message, skipped -> FK of the interpolated joints, both <= 1e-12."""
    m = _model()
    n = {"published": 0, "skipped": 0}
    for f in sorted(glob.glob(os.path.join(LAB_RUN, "*", "raw.npz"))):
        z = np.load(f)
        names = list(z["robot_joint_names"])
        Q = z["robot_joint_positions_rad"][:, [names.index(x) for x in m.joint_names]]
        st, T = z["robot_stamp_ns"].astype(np.int64), z["robot_T_base_flange"]
        pub = _rsp_published(st, 10_000_000)
        F = np.array([m.pose(names, q)[0] for q in z["robot_joint_positions_rad"]])
        assert _max_abs(F[pub], T[pub]) < 1e-12, f
        assert _max_abs(_tf_of_messages(m.fk, Q, st, pub)[~pub], T[~pub]) < 1e-12, f
        n["published"] += int(pub.sum())
        n["skipped"] += int((~pub).sum())
    assert n == {"published": 3156, "skipped": 1986}, n


@pytest.mark.skipif(shutil.which("xacro") is None, reason="xacro not available")
def test_the_lab_launch_urdf_has_the_same_chain():
    """The URDF the lab launch publishes (eoat_description rb20_1900es_u_eoat.xacro + rbpodo rb20_1900es_u with our patches — shoulder
    0 -0.2545 0, flange = tcp; mappings of hongik_lab_rb20_moveit.launch.py) has the chain of the fixture URDF: same joints, same FK."""
    with open(CALIB, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    args = {"robot_ip": "127.0.0.1", "use_fake_hardware": "false", "fake_sensor_commands": "false", "cb_simulation": "true",
            "use_isaac_sim": "false", "eoat_yaw": repr(float(c["eoat_yaw"])), "opt_xyz": " ".join(repr(float(v)) for v in c["xyz"]),
            "opt_rpy": " ".join(repr(float(v)) for v in c["rpy"]), "payload": "none", "tcp": "none", "collision_detect": "true"}
    r = subprocess.run(["xacro", ROBOT_XACRO] + [f"{k}:={v}" for k, v in args.items()], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        pytest.skip(f"xacro failed here (packages not installed?): {r.stderr[-300:]}")
    lab, fix = _model(r.stdout), _model()
    assert [(j.name, j.type, j.parent, j.child, j.xyz, j.rpy, j.axis) for j in lab.chain.joints] == \
           [(j.name, j.type, j.parent, j.child, j.xyz, j.rpy, j.axis) for j in fix.chain.joints]
    sh = [j for j in lab.chain.joints if j.name == "shoulder"][0]
    assert sh.xyz == (0.0, -0.2545, 0.0)
    rng = np.random.default_rng(13)
    for q in rng.uniform(-3.0, 3.0, size=(50, 6)):
        assert np.max(np.abs(lab.fk(q) - fix.fk(q))) < 1e-15
