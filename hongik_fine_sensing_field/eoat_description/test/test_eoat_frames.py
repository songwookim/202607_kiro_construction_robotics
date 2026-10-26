"""eoat_description 검증 (ROS 불필요).

1. xacro 파일 XML 정형성, 캘리브 YAML 스키마.
2. eoat.xacro 의 laser_joint / weld_joint 원점과 매크로 기본값으로 프레임 체인을 수치 계산해
   설계서 §2.4 의 값과 일치하는지 확인:
     - profiler_optical_frame 원점 ≈ eoat_base_link 기준 (0.012, 0, 0.147) m
     - opt_rpy (π, 0, -π/2) → (X, Y, Z)_opt = (-y, +x, +z)_eoat
     - torch_tcp = 우리 토치의 CAD 컨택트 팁 끝 (STL 의 모형 토치 부품에서 유도: 노즐 축·팁 끝면), z_tcp = 노즐 축 (공작물 쪽),
       와이어 끝 = TCP + 20 mm z
3. xacro 가 있으면 eoat 매크로를 아무 부모 링크 (플랜지) 에 붙여 실제로 확장해 URDF 구조를 검사한다 — 다른 로봇 (KIRO) 의 URDF 에
   EOAT 를 넣을 때와 같은 사용법 (eoat_base_joint parent = 플랜지, xyz 0 0 0; STL 참조 (표시용·충돌용); 광학 프레임 위치).
"""

import math
import os
import re
import xml.dom.minidom

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation as R

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_DIR = os.path.join(PKG, "urdf")
EOAT_XACRO = os.path.join(URDF_DIR, "eoat.xacro")
CALIB = os.path.join(PKG, "calib", "eoat_lab_01.yaml")


def _vec(s):
    return np.array([float(v) for v in s.split()], dtype=float)


def _T(xyz, rpy):
    """URDF origin(xyz, rpy) → 4x4. rpy = Rz(yaw)·Ry(pitch)·Rx(roll)."""
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()  # extrinsic xyz == Rz·Ry·Rx
    T[:3, 3] = xyz
    return T


def _joint_origin(dom, joint_name):
    for j in dom.getElementsByTagName("joint"):
        if j.getAttribute("name").endswith(joint_name):
            o = j.getElementsByTagName("origin")[0]
            return o.getAttribute("xyz"), o.getAttribute("rpy")
    raise KeyError(joint_name)


def _macro_defaults(dom):
    m = dom.getElementsByTagName("xacro:macro")[0]
    params = m.getAttribute("params")
    out = {}
    for name, val in re.findall(r"(\w+)='([^']*)'", params):
        out[name] = val
    return out


# ---------- 1. well-formedness / schema ----------

@pytest.mark.parametrize("path", [EOAT_XACRO, os.path.join(PKG, "package.xml")])
def test_xml_well_formed(path):
    xml.dom.minidom.parse(path)


def test_calib_schema():
    with open(CALIB, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    for key in ("eoat_serial", "platform", "parent_frame", "child_frame", "xyz", "rpy",
                "eoat_yaw", "dt_s", "x_sign", "sensor_param_hash", "date", "residuals"):
        assert key in c, key
    assert c["parent_frame"] == "laser_link"
    assert c["child_frame"] == "profiler_optical_frame"
    assert len(c["xyz"]) == 3 and len(c["rpy"]) == 3
    assert c["x_sign"] in (-1, 0, 1)
    assert isinstance(c["residuals"], dict)
    assert c["dt_s"] == 0.0  # 초기값(미보정)


# ---------- 2. frame chain arithmetic ----------

def test_eoat_macro_structure():
    dom = xml.dom.minidom.parse(EOAT_XACRO)
    meshes = dom.getElementsByTagName("mesh")
    assert len(meshes) == 2  # eoat_base_link visual + collision only
    # the CAD with the dust plate (scripts/make_dust_plate_mesh.py; the CAD eoat_base_link.STL is its input); the collision without
    # the wire stub (2026-10-13: the wire is the process stick-out, its end meets the work at a weld pose)
    # (2026-10-13: the torch turned 180 deg in its clamp - the turned-torch meshes; the CAD-orientation meshes are the generator's input)
    vis = meshes[0].parentNode.parentNode
    assert vis.tagName == "visual"
    assert meshes[0].getAttribute("filename") == "package://eoat_description/meshes/eoat_base_link_dustplate_torch180.STL"
    col = meshes[1].parentNode.parentNode
    assert col.tagName == "collision"
    assert meshes[1].getAttribute("filename") == "package://eoat_description/meshes/eoat_base_link_dustplate_torch180_collision.STL"
    joints = [j.getAttribute("name") for j in dom.getElementsByTagName("joint")]
    for j in ("eoat_base_joint", "torch_clamp_joint", "weld_joint", "laser_joint", "profiler_optical_joint", "torch_tcp_joint"):
        assert any(n.endswith(j) for n in joints), j
    xyz, rpy = _joint_origin(dom, "eoat_base_joint")
    assert xyz == "0 0 0"
    assert "eoat_yaw" in rpy
    assert os.path.isfile(os.path.join(PKG, "meshes", "eoat_base_link.STL"))
    assert os.path.isfile(os.path.join(PKG, "meshes", "eoat_base_link_dustplate.STL"))
    assert os.path.isfile(os.path.join(PKG, "meshes", "eoat_base_link_dustplate_collision.STL"))


def test_optical_frame_chain():
    dom = xml.dom.minidom.parse(EOAT_XACRO)
    d = _macro_defaults(dom)
    lx, lr = _joint_origin(dom, "laser_joint")
    T_eoat_laser = _T(_vec(lx), _vec(lr))
    # laser_joint R = diag(-1, 1, -1)
    np.testing.assert_allclose(T_eoat_laser[:3, :3], np.diag([-1, 1, -1]), atol=1e-4)
    T_laser_opt = _T(_vec(d["opt_xyz"]), _vec(d["opt_rpy"]))
    T_eoat_opt = T_eoat_laser @ T_laser_opt
    # 창면 추정 (12, 0, 147) mm
    np.testing.assert_allclose(T_eoat_opt[:3, 3], [0.0123, 0.0, 0.147], atol=1e-3)
    Rm = T_eoat_opt[:3, :3]
    # (X, Y, Z)_opt = (+y, -x, +z)_eoat  (opt_rpy yaw = +pi/2 분기 — 연구실 센서 +X 부호 실측 확정 2026-10-02)
    np.testing.assert_allclose(Rm[:, 0], [0, 1, 0], atol=1e-4)
    np.testing.assert_allclose(Rm[:, 1], [-1, 0, 0], atol=1e-4)
    np.testing.assert_allclose(Rm[:, 2], [0, 0, 1], atol=1e-4)
    # 오른손계
    assert np.linalg.det(Rm) > 0.999


def test_optical_frame_other_branch_is_coupled():
    """이전 초기값 yaw = −pi/2 이면 (X, Y, Z)_opt = (−y, +x, +z)_eoat — X·Y 부호가 함께 바뀜 (광축 둘레 180°) 을 고정."""
    dom = xml.dom.minidom.parse(EOAT_XACRO)
    lx, lr = _joint_origin(dom, "laser_joint")
    T_eoat_laser = _T(_vec(lx), _vec(lr))
    T_laser_opt = _T([0.045, 0, -0.124], [np.pi, 0, -np.pi / 2])
    Rm = (T_eoat_laser @ T_laser_opt)[:3, :3]
    # laser_joint 의 3.1416 근사 → ~1e-5 잔차
    np.testing.assert_allclose(Rm[:, 0], [0, -1, 0], atol=1e-4)
    np.testing.assert_allclose(Rm[:, 1], [1, 0, 0], atol=1e-4)
    np.testing.assert_allclose(Rm[:, 2], [0, 0, 1], atol=1e-4)


def _stl(path):
    """이진 STL → (정점 (V, 3), 면 (F, 3) 정점 번호) — 같은 좌표의 정점은 하나로."""
    import struct
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    V, inv = np.unique(a["v"].astype(float).reshape(-1, 3).round(7), axis=0, return_inverse=True)
    return V, inv.reshape(-1, 3)


def _cylinder_axis(V, F):
    """원통 부품의 축 방향 = 옆면 법선 (면적 가중) 이 없는 방향 (법선 산포 행렬의 가장 작은 고유벡터), +z 쪽."""
    T = V[F]
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    A = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm = nrm / (2.0 * A[:, None])
    ax = np.linalg.eigh((nrm * A[:, None]).T @ nrm)[1][:, 0]
    side = np.abs(nrm @ ax) < 0.2
    ax = np.linalg.eigh((nrm[side] * A[side, None]).T @ nrm[side])[1][:, 0]
    return ax if ax[2] > 0 else -ax


def test_torch_tcp_is_the_cad_contact_tip_end():
    """torch_tcp (매크로 기본값) = 우리 토치의 CAD 컨택트 팁 끝 (2026-10-12; KIRO 의 TCP = 토치 팁 끝): STL 에서 그 점을 지나는 부품 (컨택트
    팁 + 모형 와이어) 과 가스 노즐의 원통 축이 같고 (노즐 축), TCP 원점이 그 축 위 컨택트 팁 끝면 (고리 반지름 0.6–3.1 mm) 의 중심, 그 앞
    모형 와이어 (지름 1.2 mm) 의 끝 = TCP + 20.0 mm z (포락선 키포인트의 공칭 형상), TCP z = 노즐 축 (공작물 쪽, 플랜지
    축에서 30° 굽음), x 는 굽힘 면 안 (예전 곧은 자리표시자 x = -x_eoat 의 최소 회전)."""
    pytest.importorskip("scipy")
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    dom = xml.dom.minidom.parse(EOAT_XACRO)
    d = _macro_defaults(dom)
    wx, wr = _joint_origin(dom, "weld_joint")
    T = _T(_vec(wx), _vec(wr)) @ _T(_vec(d["torch_xyz"]), _vec(d["torch_rpy"]))      # eoat <- torch_tcp
    # the CAD torch (the frames below torch_clamp_link on the turn generator's input mesh; the model = both turned, test 4)
    V, F = _stl(os.path.join(PKG, "meshes", "eoat_base_link_dustplate.STL"))
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))), directed=False)
    flab = lab[F[:, 0]]
    tip_part = lab[int(np.argmin(np.linalg.norm(V - T[:3, 3], axis=1)))]               # 컨택트 팁 + 모형 와이어
    a = _cylinder_axis(V, F[flab == tip_part])
    np.testing.assert_allclose(T[:3, 2], a, atol=1e-5)                                 # z_tcp = 노즐 축
    assert np.degrees(np.arccos(a[2])) == pytest.approx(30.0, abs=0.01) and abs(a[1]) < 1e-5
    P = V[np.unique(F[flab == tip_part])]
    t = (P - T[:3, 3]) @ a
    r = np.linalg.norm((P - T[:3, 3]) - np.outer(t, a), axis=1)
    on_face = np.abs(t) < 1e-6                                                         # TCP 면 = 컨택트 팁 끝면 (고리 0.6–3.1 mm)
    assert r[on_face].min() == pytest.approx(0.0006, abs=1e-5) and r[on_face].max() == pytest.approx(0.0031, abs=1e-5)
    assert t.max() == pytest.approx(0.020, abs=1e-5)                                   # 와이어 끝 = TCP + 20.0 mm z
    assert np.all(r[t > 1e-6] < 0.0006 + 1e-5)                                         # 그 사이는 지름 1.2 mm 와이어뿐
    wrap = [c for c in set(flab.tolist()) if c != tip_part                            # 같은 축을 감싸는 부품 (반지름 <= 10 mm)
            and np.max(np.linalg.norm(np.cross(V[np.unique(F[flab == c])] - T[:3, 3], a), axis=1)) < 0.0101]
    nozzle = max(wrap, key=lambda c: float(np.max((V[np.unique(F[flab == c])] - T[:3, 3]) @ a)))   # 끝에 가장 가까운 것 = 가스 노즐
    np.testing.assert_allclose(_cylinder_axis(V, F[flab == nozzle]), a, atol=1e-5)
    assert float(np.max((V[np.unique(F[flab == nozzle])] - T[:3, 3]) @ a)) == pytest.approx(0.0012, abs=1e-4)   # 노즐 끝 1.2 mm 앞
    np.testing.assert_allclose(T[:3, 3], [-0.046555, -0.000452, 0.297172], atol=2e-5)
    # x = 굽힘 면 안: 예전 곧은 자리표시자 축 diag(-1, -1, 1) 을 z_eoat x 노즐 축 둘레로 돌린 최소 회전
    np.testing.assert_allclose(T[:3, 0], [-a[2], 0.0, a[0]], atol=1e-5)
    np.testing.assert_allclose(T[:3, 1], [0.0, -1.0, 0.0], atol=1e-5)


def _dust_plate_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_dust_plate_mesh", os.path.join(PKG, "scripts", "make_dust_plate_mesh.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dust_plate_mesh_is_regenerated_bit_for_bit_from_the_cad():
    """eoat_base_link_dustplate.STL = scripts/make_dust_plate_mesh.py (the CAD eoat_base_link.STL in, deterministic)."""
    pytest.importorskip("scipy")
    M = _dust_plate_module()
    data, _info = M.build()
    with open(os.path.join(PKG, "meshes", "eoat_base_link_dustplate.STL"), "rb") as f:
        assert f.read() == data
    coll, _cinfo = M.build_collision(data)
    with open(os.path.join(PKG, "meshes", "eoat_base_link_dustplate_collision.STL"), "rb") as f:
        assert f.read() == coll


def test_collision_mesh_is_the_mesh_without_the_wire_stub():
    """충돌 메시 (2026-10-13) = 먼지판 메시에서 CAD 모형 와이어 토막 (TCP 면 너머 지름 1.2 mm, 20.0 mm) 만 뺀 것: 컨택트 팁 끝면은 닫고
    (TCP 면 위 부채꼴, 법선 +z_TCP), 나머지 삼각형은 그대로; TCP 면 너머에 남는 것은 노즐 (1.2 mm) 뿐."""
    pytest.importorskip("scipy")
    M = _dust_plate_module()
    full = M.read_stl(os.path.join(PKG, "meshes", "eoat_base_link_dustplate.STL"))
    coll = M.read_stl(os.path.join(PKG, "meshes", "eoat_base_link_dustplate_collision.STL"))
    _d, info = M.build_collision(open(os.path.join(PKG, "meshes", "eoat_base_link_dustplate.STL"), "rb").read())
    assert info["wire_triangles_removed"] == 34 and info["wire_length_mm"] == pytest.approx(20.0, abs=1e-3)
    n_keep = len(full) - info["wire_triangles_removed"]
    assert len(coll) == n_keep + info["end_face_fan_triangles"]
    key = lambda r: {tuple(np.round(t.ravel(), 9)) for t in r["v"].astype(float)}     # noqa: E731
    assert key(coll[:n_keep]) <= key(full)                                             # the rest unchanged
    dom = xml.dom.minidom.parse(EOAT_XACRO)
    d = _macro_defaults(dom)
    wx, wr = _joint_origin(dom, "weld_joint")
    T = _T(_vec(wx), _vec(wr)) @ _T(_vec(d["torch_xyz"]), _vec(d["torch_rpy"]))      # eoat <- torch_tcp
    V = coll["v"].reshape(-1, 3).astype(float)
    t = (V - T[:3, 3]) @ T[:3, 2]
    r = np.linalg.norm((V - T[:3, 3]) - np.outer(t, T[:3, 2]), axis=1)
    torch_ = r < 0.0101                                                                # the torch (within the nozzle radius)
    assert t[torch_].max() == pytest.approx(0.0012, abs=1e-4)                          # beyond the TCP plane: the nozzle end only
    assert np.all(r[t > 1e-6] > 0.003)                                                 # nothing on the wire's axis there
    assert t.max() == pytest.approx(0.01586, abs=1e-4) and r[np.argmax(t)] > 0.040     # the sensor guard: 15.9 mm beyond, > 40 mm off
    cap = coll[n_keep:]
    assert np.allclose(cap["n"].astype(float), T[:3, 2], atol=1e-6)                  # the closed end face faces the work
    tc = (cap["v"].reshape(-1, 3).astype(float) - T[:3, 3]) @ T[:3, 2]
    assert np.all(np.abs(tc) < 1e-6)
    # closed: every edge of the contact-tip part is used twice (the wire's hole is closed)
    Tv = coll["v"].astype(float)
    Vu, inv = np.unique(np.round(Tv.reshape(-1, 3), 9), axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    near = np.linalg.norm(Vu[F].mean(axis=1) - T[:3, 3], axis=1) < 0.0035
    E = np.sort(np.concatenate([F[near][:, [0, 1]], F[near][:, [1, 2]], F[near][:, [2, 0]]]), axis=1)
    _u, cnt = np.unique(E, axis=0, return_counts=True)
    ring_open = (cnt == 1)
    on_face = np.all(np.abs(((Vu[_u.ravel()] - T[:3, 3]) @ T[:3, 2]).reshape(-1, 2)) < 1e-6, axis=1)
    rr = np.linalg.norm(np.cross(Vu[_u.ravel()] - T[:3, 3], T[:3, 2]), axis=1).reshape(-1, 2)
    assert not np.any(ring_open & on_face & np.all(rr < 0.0006 + 1e-5, axis=1))          # no open edge left on the wire's base ring


def test_dust_plate_moves_the_sensor_5_mm_off_the_bracket_and_fills_the_gap():
    """먼지판 (사용자 2026-10-12 밤, 연구실·KIRO): 센서 쪽 부품의 등면 (하우징 뒷면, 마운트 면) 이 브라켓 밑면에서 정확히 5.0 mm (+z_eoat)
    떨어지고, 그 사이 5.0 mm 판이 뒷면 자국을 덮는다; 브라켓·어댑터판·토치 삼각형은 CAD 그대로, 센서 쪽 삼각형은 모두 +5.0 mm (z 만)."""
    pytest.importorskip("scipy")
    M = _dust_plate_module()
    cad = M.read_stl(os.path.join(PKG, "meshes", "eoat_base_link.STL"))
    new = M.read_stl(os.path.join(PKG, "meshes", "eoat_base_link_dustplate.STL"))
    c = M.classify(cad)
    n0 = len(cad)
    assert c["z_interface_m"] == pytest.approx(0.057, abs=1e-6)                       # the housing back face = the bracket bottom (CAD)
    d = new["v"][:n0].astype(float) - cad["v"].astype(float)
    assert np.array_equal(new["v"][:n0][~c["sensor"]], cad["v"][~c["sensor"]])           # bracket, adapter plate, torch untouched
    assert np.allclose(d[c["sensor"]][:, :, :2], 0.0, atol=0.0)
    assert np.allclose(d[c["sensor"]][:, :, 2], 0.005, atol=2e-8)                       # every sensor-side vertex +5.0 mm (float32)
    zi = c["z_interface_m"]
    Tn, nz = new["v"][:n0].astype(float), new["n"][:n0][:, 2]
    def plane(mask, sign, z):
        return mask & (nz * sign > 0.999) & np.all(np.abs(Tn[:, :, 2] - z) < 1e-6, axis=1)
    bracket_bottom = plane(c["bracket"], +1, zi)                                        # faces the work
    sensor_back = plane(c["sensor"] & c["housing"], -1, zi + 0.005)                     # faces the flange
    assert bracket_bottom.sum() > 10 and sensor_back.sum() > 10
    zb = float(np.max(Tn[bracket_bottom][:, :, 2]))
    zs = float(np.min(Tn[sensor_back][:, :, 2]))
    assert zs - zb == pytest.approx(0.0050, abs=1e-6)                                   # pinned: the sensor sits 5.0 mm off the bracket
    P = new["v"][n0:].astype(float)                                                     # the plate: a prism from the bracket to the sensor
    assert len(P) >= 8 and set(np.round(P[:, :, 2].ravel(), 6)) == {round(zb, 6), round(zs, 6)}
    lo, hi = P.reshape(-1, 3).min(axis=0), P.reshape(-1, 3).max(axis=0)
    B = Tn[plane(c["sensor"], -1, zi + 0.005)].reshape(-1, 3)                           # every sensor-side back face at the interface
    assert np.allclose(lo[:2], B.min(axis=0)[:2], atol=1e-6) and np.allclose(hi[:2], B.max(axis=0)[:2], atol=1e-6)   # covers the back face
    assert c["housing"].sum() > 1000 and np.all(c["sensor"][c["housing"]])


def _T_eoat_tcp_cad():
    """eoat_base_link <- torch_tcp of the CAD torch (eoat.xacro weld_joint . torch_tcp defaults)."""
    dom = xml.dom.minidom.parse(EOAT_XACRO)
    d = _macro_defaults(dom)
    wx, wr = _joint_origin(dom, "weld_joint")
    return _T(_vec(wx), _vec(wr)) @ _T(_vec(d["torch_xyz"]), _vec(d["torch_rpy"]))


def _mesh_on_tcp_urdf(mesh_path, T_eoat_tcp) -> str:
    """A minimal URDF: link eoat with the mesh as its collision geometry, a fixed joint eoat -> torch_tcp (for weld_clearance)."""
    rpy = R.from_matrix(T_eoat_tcp[:3, :3]).as_euler("xyz")
    return (f'<robot name="t"><link name="eoat"><collision><geometry><mesh filename="file://{mesh_path}"/></geometry></collision>'
            f'</link><link name="torch_tcp"/><joint name="j" type="fixed"><parent link="eoat"/><child link="torch_tcp"/>'
            f'<origin xyz="{" ".join(repr(float(v)) for v in T_eoat_tcp[:3, 3])}" rpy="{" ".join(repr(float(v)) for v in rpy)}"/>'
            f'</joint></robot>')


def _turn_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_torch_turned_mesh", os.path.join(PKG, "scripts", "make_torch_turned_mesh.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kiro_torch180_meshes_are_regenerated_bit_for_bit():
    """KIRO's tool setup (identified 2026-10-13, KIRO URDF v2 r4): eoat_base_link_dustplate_torch180.STL / _collision.STL =
    scripts/make_torch_turned_mesh.py (the two dust-plate meshes in, deterministic)."""
    pytest.importorskip("scipy")
    vis, col, _t = _turn_module().build()
    with open(os.path.join(PKG, "meshes", "eoat_base_link_dustplate_torch180.STL"), "rb") as f:
        assert f.read() == vis
    with open(os.path.join(PKG, "meshes", "eoat_base_link_dustplate_torch180_collision.STL"), "rb") as f:
        assert f.read() == col


def test_kiro_torch_is_turned_180_deg_about_its_straight_body_axis():
    """The turned torch (KIRO): the body axis is parallel to z_eoat (the CAD cylinder) through eoat x -142.2 mm; only the torch turns
    (body, neck, nozzle, contact tip [+ the visual's wire stub]) - the clamp, bracket, plates, sensor and guard keep every vertex; the
    torch is a rigid copy of the CAD torch turned by 180 deg; the TCP (eoat.xacro torch_tcp) turns with it to eoat (-237.86, -0.45,
    297.17) mm, its nozzle axis 30 deg off z_eoat now leaning to -x_eoat (away from the sensor at +x)."""
    pytest.importorskip("scipy")
    M = _turn_module()
    t = M.turn()
    a, c = t["axis"]["direction"], t["axis"]["point_m"]
    np.testing.assert_allclose(a, [0.0, 0.0, 1.0], atol=1e-6)
    assert c[0] * 1e3 == pytest.approx(-142.21, abs=0.01) and abs(c[1]) < 1e-3
    for key, src in (("visual", "eoat_base_link_dustplate.STL"), ("collision", "eoat_base_link_dustplate_collision.STL")):
        T0 = M.read_triangles(os.path.join(PKG, "meshes", src))
        T1 = t[key]
        moved = np.any(np.abs(T1 - T0) > 0.0, axis=(1, 2))
        assert int(moved.sum()) == t["turned_triangles"][key] > 1000
        assert np.all(T0[moved][:, :, 0] < -0.020) and float(T0[moved][:, :, 2].max()) > 0.100          # the torch only
        back = (T1[moved].reshape(-1, 3) - c) @ M.rot_about(a, 180.0).T + c                               # turning back = the CAD
        np.testing.assert_allclose(back.reshape(-1, 3, 3), T0[moved], atol=1e-9)
    T_tcp = M.tcp_eoat(t["T_turn"])
    np.testing.assert_allclose(T_tcp[:3, 3], [-0.237857, -0.000452, 0.297172], atol=2e-6)
    np.testing.assert_allclose(T_tcp[:3, 2], [-0.5, 0.0, math.sqrt(3.0) / 2.0], atol=2e-6)
    np.testing.assert_allclose(M.tcp_eoat(np.eye(4)), _T_eoat_tcp_cad(), atol=1e-12)                  # unturned = the CAD torch_tcp
    wc = pytest.importorskip("scan_core.weld_clearance")
    tcp = M.tcp_eoat(t["T_turn"])
    urdf = _mesh_on_tcp_urdf(os.path.join(PKG, "meshes", "eoat_base_link_dustplate_torch180_collision.STL"), tcp)
    assert wc.nozzle_end_mm(wc.tool_body_from_urdf(urdf, "torch_tcp")) == pytest.approx(1.2, abs=1e-3)       # the nozzle rim, as the CAD


# ---------- 3. real xacro expansion (pip xacro; skipped if unavailable) ----------

def _run_xacro(path, mappings):
    """xacro 를 UTF-8 강제 서브프로세스로 실행 (Windows cp949 로케일에서 한글 주석 디코드 오류 회피).

    Ubuntu 에서는 `xacro file k:=v` 와 동일. ament_index_python 이 없으면 conftest 의 셤 디렉터리를
    PYTHONPATH 로 넘겨 `$(find pkg)` 를 워크스페이스 src/ 에서 해결한다.
    """
    pytest.importorskip("xacro")
    import subprocess
    import sys

    # scan_stack/conftest.py 와 동일 판정. `import conftest` 는 다른 패키지의 tests/conftest.py 와
    # 모듈명이 충돌하므로(전체 트리 pytest 실행 시) 경로로 직접 해결한다.
    shim_dir = os.path.join(os.path.dirname(PKG), "_no_ros_shim")  # scan_stack/_no_ros_shim
    try:
        import ament_index_python.packages as _aip  # noqa: WPS433
        # 루트 conftest 가 이미 셤을 sys.path 에 넣었으면 import 는 성공하지만 파일 위치가 셤 아래다.
        mod_file = os.path.abspath(getattr(_aip, "__file__", "") or "")
        in_shim = mod_file.startswith(os.path.abspath(shim_dir) + os.sep)
        shim = shim_dir if in_shim else None  # Ubuntu/ROS 2: 실제 모듈 -> 셤 불필요
    except ImportError:
        shim = shim_dir if os.path.isdir(shim_dir) else None
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    if shim:
        env["PYTHONPATH"] = shim + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-X", "utf8", "-c", "import xacro; xacro.main()", path]
    cmd += ["%s:=%s" % (k, v) for k, v in mappings.items()]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    assert res.returncode == 0, "xacro failed:\n%s" % res.stderr
    return res.stdout


def test_eoat_macro_expands_on_any_flange():
    """다른 로봇의 URDF 에 EOAT 를 넣는 사용법: 플랜지 링크에 매크로를 붙이고 캘리브의 eoat_yaw / opt_xyz / opt_rpy 를 넘긴다."""
    import tempfile
    with open(CALIB, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    wrapper = f"""<?xml version="1.0"?>
<robot name="any_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="{EOAT_XACRO}"/>
  <link name="base_link"/>
  <link name="flange"/>
  <joint name="base_to_flange" type="fixed"><parent link="base_link"/><child link="flange"/><origin xyz="0 0 1" rpy="0 0 0"/></joint>
  <xacro:eoat parent="flange" eoat_yaw="{c['eoat_yaw']}" opt_xyz="{' '.join(str(v) for v in c['xyz'])}"
              opt_rpy="{' '.join(str(v) for v in c['rpy'])}"/>
</robot>
"""
    with tempfile.NamedTemporaryFile("w", suffix=".xacro", delete=False, encoding="utf-8") as f:
        f.write(wrapper)
        path = f.name
    try:
        urdf = _run_xacro(path, {})
    finally:
        os.unlink(path)
    dom = xml.dom.minidom.parseString(urdf)
    joints = {j.getAttribute("name"): j for j in dom.getElementsByTagName("joint")}
    j = joints["eoat_base_joint"]
    assert j.getElementsByTagName("parent")[0].getAttribute("link") == "flange"
    assert j.getElementsByTagName("child")[0].getAttribute("link") == "eoat_base_link"
    o = j.getElementsByTagName("origin")[0]
    np.testing.assert_allclose(_vec(o.getAttribute("xyz")), [0, 0, 0])
    np.testing.assert_allclose(_vec(o.getAttribute("rpy")), [0, 0, c["eoat_yaw"]], atol=1e-6)
    assert joints["profiler_optical_joint"].getElementsByTagName("parent")[0].getAttribute("link") == "laser_link"
    assert joints["torch_tcp_joint"].getElementsByTagName("parent")[0].getAttribute("link") == "weld_link"
    meshes = [m.getAttribute("filename") for m in dom.getElementsByTagName("mesh")]
    assert meshes == ["package://eoat_description/meshes/eoat_base_link_dustplate_torch180.STL",
                      "package://eoat_description/meshes/eoat_base_link_dustplate_torch180_collision.STL"]


# ---------- 4. the EOAT model = the torch turned 180 deg in its clamp (KIRO's setup, URDF v2 r4; the lab torch re-mounted the same way 2026-10-13) ----------

def _fk_fixed(dom, parent, child):
    """parent <- child through a (raw or expanded) URDF's joints (child below parent; the macro's ${...} origins resolved with its
    defaults)."""
    d = {}
    for m in dom.getElementsByTagName("xacro:macro"):
        d = dict(re.findall(r"(\w+)='([^']*)'", m.getAttribute("params")))
    js = {}
    for j in dom.getElementsByTagName("joint"):
        if not j.getElementsByTagName("child"):
            continue
        o = j.getElementsByTagName("origin")
        xyz = o[0].getAttribute("xyz") if o else "0 0 0"
        rpy = o[0].getAttribute("rpy") if o else "0 0 0"
        xyz, rpy = (re.sub(r"\$\{(\w+)\}", lambda m_: d[m_.group(1)], v) for v in (xyz, rpy))
        name = re.sub(r"\$\{prefix\}", "", j.getElementsByTagName("child")[0].getAttribute("link"))
        par = re.sub(r"\$\{prefix\}", "", j.getElementsByTagName("parent")[0].getAttribute("link"))
        js[name] = (par, _T(_vec(xyz or "0 0 0"), _vec(rpy or "0 0 0")))
    T, cur = np.eye(4), child
    while cur != parent:
        par, To = js[cur]
        T, cur = To @ T, par
    return T


def test_the_eoat_model_is_the_torch_turned_in_its_clamp():
    """eoat.xacro = the turned torch (no option): the turned-torch meshes (visual, collision), torch_clamp_joint = the script's turn
    (turn()['T_turn'], full precision) carrying the CAD torch frames - eoat <- weld_link / torch_tcp = the script's weld_link_eoat /
    tcp_eoat of the turn (TCP eoat (-237.86, -0.45, 297.17) mm, nozzle axis leaning away from the sensor); the CAD-orientation meshes are
    only the generator's input (not referenced)."""
    pytest.importorskip("scipy")
    M = _turn_module()
    t = M.turn()
    dom = xml.dom.minidom.parse(EOAT_XACRO)
    files = [m.getAttribute("filename") for m in dom.getElementsByTagName("mesh")]
    assert files == ["package://eoat_description/meshes/eoat_base_link_dustplate_torch180.STL",
                     "package://eoat_description/meshes/eoat_base_link_dustplate_torch180_collision.STL"]
    cx, cr = _joint_origin(dom, "torch_clamp_joint")
    np.testing.assert_allclose(_T(_vec(cx), _vec(cr)), t["T_turn"], atol=1e-12)
    np.testing.assert_allclose(_fk_fixed(dom, "eoat_base_link", "weld_link"), M.weld_link_eoat(t["T_turn"]), atol=1e-12)
    T_tcp = _fk_fixed(dom, "eoat_base_link", "torch_tcp")
    np.testing.assert_allclose(T_tcp, M.tcp_eoat(t["T_turn"]), atol=1e-12)
    np.testing.assert_allclose(T_tcp[:3, 3] * 1e3, [-237.86, -0.45, 297.17], atol=0.006)
    assert T_tcp[0, 2] < 0.0                                                          # the nozzle axis leans to -x_eoat (away from the sensor)
