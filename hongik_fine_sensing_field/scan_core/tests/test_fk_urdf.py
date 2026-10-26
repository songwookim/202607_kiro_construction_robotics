import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from scan_core.fk_urdf import (
    RB20_1900ES_U_SHOULDER_Y,
    chain_from_joint_yaml,
    chain_from_urdf,
    load_chain,
    parse_urdf_joints,
    rb20_1900es_u_chain,
    rpy_to_matrix,
)


def _find_desc_dir():
    """rbpodo_description/robots (the RB20 reference URDF / joint.yaml) when it is installed here - else the URDF tests skip."""
    import os
    cands = [os.environ.get("RBPODO_DESCRIPTION_ROBOTS_DIR", "")]
    try:
        from ament_index_python.packages import get_package_share_directory
        cands.append(os.path.join(get_package_share_directory("rbpodo_description"), "robots"))
    except Exception:  # noqa: BLE001 - no ROS / package not built
        pass
    for c in cands:
        if c and os.path.exists(os.path.join(c, "rb20_1900es_u.urdf")):
            return c
    return None


@pytest.fixture(scope="session")
def urdf_path() -> str:
    import os
    d = _find_desc_dir()
    if d is None:
        pytest.skip("rbpodo_description (RB20 reference URDF) not installed")
    return os.path.join(d, "rb20_1900es_u.urdf")


@pytest.fixture(scope="session")
def joint_yaml_path() -> str:
    import os
    d = _find_desc_dir()
    if d is None:
        pytest.skip("rbpodo_description (RB20 joint.yaml) not installed")
    return os.path.join(d, "rb20_1900es_u", "joint.yaml")

ZERO_POSE_M = np.array([0.0, -0.067, 2.141])

BROKEN_YAML = """
base:
  origin: {xyz: 0 0 0.241, rpy: 0 0 0}
  parent: link0
  child: link1
  axis: {xyz: 0 0 1}
shoulder:
  origin: {xyz: 0 0 0, rpy: 0 0 0}
  parent: link1
  child: link2
  axis: {xyz: 0 1 0}
elbow:
  origin: {xyz: 0 0 0.885, rpy: 0 0 0}
  parent: link2
  child: link3
  axis: {xyz: 0 1 0}
wrist1:
  origin: {xyz: 0 0.1875 0, rpy: 0 0 0}
  parent: link3
  child: link4
  axis: {xyz: 0 0 1}
wrist2:
  origin: {xyz: 0 -0.1292 0.885, rpy: 0 0 0}
  parent: link4
  child: link5
  axis: {xyz: 0 1 0}
wrist3:
  origin: {xyz: 0 0.1292 0, rpy: 0 0 0}
  parent: link5
  child: link6
  axis: {xyz: 0 0 1}
"""


def test_rpy_matches_scipy():
    r, p, y = 0.3, -1.1, 2.0
    assert np.allclose(rpy_to_matrix(r, p, y), Rot.from_euler("xyz", [r, p, y]).as_matrix())


def test_hardcoded_zero_pose():
    ch = rb20_1900es_u_chain()
    T = ch.fk([0] * 6)
    assert np.allclose(T[:3, 3], ZERO_POSE_M, atol=1e-12)
    assert np.allclose(T[:3, :3], np.eye(3))
    assert ch.joint_names == ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
    assert ch.dof == 6


def test_urdf_zero_pose(urdf_path):
    ch = chain_from_urdf(urdf_path, "link0", "tcp")
    T = ch.fk(np.zeros(6))
    assert np.allclose(T[:3, 3], ZERO_POSE_M, atol=1e-12)
    assert np.allclose(T[:3, :3], np.eye(3))
    joints = parse_urdf_joints(urdf_path)
    assert joints["shoulder"].xyz[1] == pytest.approx(RB20_1900ES_U_SHOULDER_Y)
    assert joints["tcp_joint"].type == "fixed" and joints["tcp_joint"].xyz[2] == pytest.approx(0.13)
    # <ros2_control><joint> entries (no type/parent) must not be picked up
    assert all(j.parent for j in joints.values())


def test_joint_yaml_zero_pose(joint_yaml_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ch = chain_from_joint_yaml(joint_yaml_path)
    T = ch.fk(np.zeros(6))
    assert np.allclose(T[:3, 3], ZERO_POSE_M, atol=1e-12)


def test_broken_yaml_is_fixed_with_warning(tmp_path):
    p = tmp_path / "joint.yaml"
    p.write_text(BROKEN_YAML, encoding="utf-8")
    with pytest.warns(UserWarning, match="shoulder origin is 0 0 0"):
        ch = chain_from_joint_yaml(str(p))
    assert np.allclose(ch.fk(np.zeros(6))[:3, 3], ZERO_POSE_M)
    # without the fix the regression is reproduced (y = +0.1875)
    ch_raw = chain_from_joint_yaml(str(p), fix_shoulder=False)
    assert ch_raw.fk(np.zeros(6))[1, 3] == pytest.approx(0.1875)


def test_urdf_vs_yaml_vs_hardcoded_random(urdf_path, joint_yaml_path):
    ch_u = chain_from_urdf(urdf_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ch_y = chain_from_joint_yaml(joint_yaml_path)
    ch_h = rb20_1900es_u_chain()
    rng = np.random.default_rng(42)
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, 6)
        Tu, Ty, Th = ch_u.fk(q), ch_y.fk(q), ch_h.fk(q)
        assert np.allclose(Tu, Ty, atol=1e-12)
        assert np.allclose(Tu, Th, atol=1e-12)
        # rotation stays orthonormal
        R = Tu[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_fk_all_and_batch():
    ch = rb20_1900es_u_chain()
    frames = ch.fk_all([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert set(frames) >= {"link0", "link1", "link6", "tcp"}
    assert np.allclose(frames["link1"][:3, 3], [0, 0, 0.241])
    # tcp is 0.13 m along link6 z
    assert np.allclose(frames["tcp"][:3, 3], frames["link6"][:3, 3] + frames["link6"][:3, 2] * 0.13)
    Q = np.zeros((3, 6))
    assert ch.fk_batch(Q).shape == (3, 4, 4)
    with pytest.raises(ValueError):
        ch.fk([0, 0, 0])


def test_single_joint_directions():
    """Positive base rotation moves a +x offset toward +y (right-hand rule about +z)."""
    ch = rb20_1900es_u_chain()
    T = ch.fk([np.pi / 2, 0, 0, 0, 0, 0])
    # zero pose y = -0.067 -> after +90 deg about z becomes x = +0.067
    assert np.allclose(T[:3, 3], [0.067, 0.0, 2.141], atol=1e-12)
    # shoulder +90 deg about +y folds the arm toward +x
    T = ch.fk([0, np.pi / 2, 0, 0, 0, 0])
    assert T[0, 3] > 1.5 and abs(T[2, 3] - 0.241) < 1e-9


def test_load_chain_fallback():
    ch = load_chain(None, None)
    assert ch.dof == 6


# --------------------------------------------------------------------------------------
# 2026-10-02: batch FK (after a 342 mm real move, per-sample FK in measure_motion_speed held the GIL for 0.5 s and the
# sensor guard fell behind -> false G3 e-stop)
# --------------------------------------------------------------------------------------
def test_fk_batch_equals_fk_and_is_fast():
    import time
    from scan_core.fk_urdf import load_chain
    chain = load_chain(None, None)
    rng = np.random.default_rng(7)
    Q = rng.uniform(-3.0, 3.0, size=(300, chain.dof))
    A = chain.fk_batch(Q)
    B = np.stack([chain.fk(q) for q in Q])
    assert A.shape == (300, 4, 4) and np.abs(A - B).max() < 1e-12
    Qd = np.rad2deg(Q)
    assert np.abs(chain.fk_deg_batch(Qd) - A).max() < 1e-12
    big = rng.uniform(-3.0, 3.0, size=(3000, chain.dof))
    t0 = time.perf_counter()
    chain.fk_batch(big)
    assert time.perf_counter() - t0 < 0.2                     # 3000 samples: ~ms (per-sample loop was ~0.6 s)


def test_limit_problems_names_the_joint_and_the_overshoot():
    """시작 자세 교시 (2026-10-08 실기: 엘보 −152.07° 가 계획 한계 ±150° 밖): 어느 관절이 얼마나 넘었는지, 이름 순서와 무관하게."""
    import math
    from scan_core.fk_urdf import Joint, KinematicChain
    lim = math.radians(150.0)
    js = [Joint("shoulder", "revolute", "l0", "l1", (0, 0, 0), (0, 0, 0), lower=-2 * math.pi, upper=2 * math.pi),
          Joint("elbow", "revolute", "l1", "l2", (0, 0, 0.5), (0, 0, 0), lower=-lim, upper=lim)]
    ch = KinematicChain(js, "l0", "l2")
    bad = ch.limit_problems({"elbow": math.radians(-152.07), "shoulder": 0.2})
    assert len(bad) == 1 and bad[0].startswith("elbow -152.07 deg outside [-150, 150] deg") and "2.07 deg beyond" in bad[0], bad
    assert ch.limit_problems([0.2, math.radians(-149.46)]) == []
    m = ch.limit_margins([0.2, math.radians(-149.46)])
    assert math.degrees(m["elbow"]) == pytest.approx(0.54, abs=1e-6) and m["shoulder"] > 3.0
    assert ch.limit_problems({"elbow": 0.0}) == ["shoulder missing"]


def test_mounted_collision_points_are_the_fixed_subtree_below_the_mount_in_its_frame(tmp_path):
    """The tool on a flange as the robot model describes it (MoveIt adapter tool_geometry): every link below the mount through fixed
    joints (children direction only - not the arm link the flange is fixed to, not a sibling fixed to that arm link), its collision
    geometry as hull points in the mount frame (STL hull vertices with origin and scale; box / cylinder / sphere box corners)."""
    import io
    from scan_core.fk_urdf import mounted_collision_points
    stl = tmp_path / "tet.stl"                                               # one triangle fan (4 vertices, ASCII)
    stl.write_text("solid t\n" + "".join(f"facet normal 0 0 1\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in tri)
                                        + "endloop\nendfacet\n" for tri in (((0, 0, 0), (1, 0, 0), (0, 1, 0)),
                                                                          ((0, 0, 0), (0, 1, 0), (0, 0, 1)),
                                                                          ((0, 0, 0), (0, 0, 1), (1, 0, 0)),
                                                                          ((1, 0, 0), (0, 1, 0), (0, 0, 1))))
                   + "endsolid t\n")
    urdf = f"""<robot name="r">
      <link name="link6"><collision><geometry><box size="1 1 1"/></geometry></collision></link>
      <link name="flange"/>
      <link name="eoat"><collision><origin xyz="0 0 0.1" rpy="0 0 0"/><geometry><box size="0.2 0.4 0.02"/></geometry></collision></link>
      <link name="torch"><collision><origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry><mesh filename="file://{stl}" scale="0.01 0.01 0.01"/></geometry></collision></link>
      <link name="sibling"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
      <link name="frame_only"/>
      <joint name="j6f" type="fixed"><parent link="link6"/><child link="flange"/><origin xyz="0 0 0.06" rpy="0 0 0"/></joint>
      <joint name="jfe" type="fixed"><parent link="flange"/><child link="eoat"/><origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/></joint>
      <joint name="jet" type="fixed"><parent link="eoat"/><child link="torch"/><origin xyz="0.3 0 0.2" rpy="0 0 0"/></joint>
      <joint name="jef" type="fixed"><parent link="eoat"/><child link="frame_only"/><origin xyz="0 0 0.5" rpy="0 0 0"/></joint>
      <joint name="j6s" type="fixed"><parent link="link6"/><child link="sibling"/><origin xyz="0 0.5 0" rpy="0 0 0"/></joint>
    </robot>"""
    m = mounted_collision_points(io.StringIO(urdf), "flange")
    assert m["links"] == ["eoat", "torch"] and m["why"] == []
    P = m["points"]
    # the eoat box (yaw 90 deg: its 0.2 x 0.4 footprint turns to 0.4 x 0.2 in the flange frame), centre z 0.1
    box = P[:8]
    assert np.allclose(np.sort(np.unique(np.round(box[:, 0], 9))), [-0.2, 0.2]) and np.allclose(np.unique(np.round(box[:, 1], 9)), [-0.1, 0.1])
    assert np.allclose(np.unique(np.round(box[:, 2], 9)), [0.09, 0.11])
    # the mesh: its 4 hull vertices x 0.01, at eoat (0.3, 0, 0.2) -> flange (0, 0.3, 0.2) after the yaw
    tet = P[8:]
    assert tet.shape == (4, 3) and np.allclose(tet.min(axis=0), [-0.01, 0.3, 0.2]) and np.allclose(tet.max(axis=0), [0.0, 0.31, 0.21])
    # nothing below the mount = nothing (the arm's own links are not the tool); an unreadable mesh is named, not dropped silently
    assert mounted_collision_points(io.StringIO(urdf), "frame_only")["points"] is None
    m2 = mounted_collision_points(io.StringIO(urdf.replace(f"file://{stl}", "package://no_such_pkg/x.stl")), "flange",
                                  mesh_path=lambda f: None)
    assert m2["links"] == ["eoat"] and m2["why"] and "torch: mesh package://no_such_pkg/x.stl not readable here" in m2["why"][0]
