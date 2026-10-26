"""우리 토치 모형 (2026-10-12: 용접 토치 = 우리 1차 브라켓 토치) 이 쓰이는 모든 곳이 한 모형 (eoat.xacro torch_tcp = CAD 컨택트 팁 끝,
포락선 키포인트 = 토치의 가장 먼 점 = CAD 의 모형 와이어 끝 — 공칭 돌출 형상; 계획의 돌출은 노즐 끝에서 잰 공정 값 stick_out_mm) 과 같은 값인가 — 수를 옮겨 적은 곳 (가상 팔 URDF, 설정 weld_torch.flange_from_tcp, 포락선
키포인트의 작업 torch_lateral_m·설정 safety.torch_protrusion_m) 이 모형에서 어긋나지 않게 (10-06 광축 원점 −25 mm 뒤 팁 키포인트가 24.5 mm
짧았던 것 같은 일)."""
import io
import math
import os
import re
import xml.dom.minidom

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from scan_core.fk_urdf import chain_from_joints, parse_urdf_joints
from scan_l0 import config as C
from scan_core.weld_torch import weld_torch_config
from sim_harness import CALIB

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.dirname(os.path.dirname(HERE))
URDF = os.path.join(HERE, "data", "rb20_1900es_u_eoat.urdf")
EOAT_XACRO = os.path.join(STACK, "eoat_description", "urdf", "eoat.xacro")
# the KIRO config ships in hongik_scan_ros/config; the lab and fake-arm configs are test data in this tree (kiro-1g)
CFG_PATHS = {"hongik_kiro.yaml": os.path.join(STACK, "hongik_scan_ros", "config", "hongik_kiro.yaml"),
             "hongik_lab_rb20_moveit.yaml": os.path.join(HERE, "data", "hongik_lab_rb20_values.yaml"),
             "hongik_rb20_fake_moveit.yaml": os.path.join(HERE, "data", "hongik_rb20_fake_moveit.yaml")}
JOBS = os.path.join(STACK, "scan_l0", "scan_l0", "jobs")
LAB_PLATFORMS = ("hongik_lab_rb20_moveit.yaml", "hongik_rb20_fake_moveit.yaml")
# the job templates with the lab torch's keypoint; groove_10T_1G (the default 1G job = KIRO's) carries KIRO's turned torch (r4)
LAB_TEMPLATES = ("groove_10T_3G.yaml", "example_1G.yaml", "example_3G.yaml")
KIRO_TEMPLATE = "groove_10T_1G.yaml"
KIRO_CALIB = os.path.join(STACK, "eoat_description", "calib", "eoat_kiro_01.yaml")
_TURN = {}


def _turned():
    """KIRO's torch (identified 2026-10-13, URDF v2 r4): eoat_description/scripts/make_torch_turned_mesh.py's turn of the CAD torch
    (180 deg about its straight body axis) - {'T_turn', 'T_eoat_tcp'} (eoat frame)."""
    if not _TURN:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_torch_turned_mesh", os.path.join(STACK, "eoat_description", "scripts", "make_torch_turned_mesh.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t = mod.turn()
        _TURN.update(T_turn=t["T_turn"], T_eoat_tcp=mod.tcp_eoat(t["T_turn"]))
    return _TURN
TOL_M = 0.5e-3                                   # 옮겨 적은 수의 자릿수 (0.1 mm 단위 작업 값) 안


def _fixed(a, b) -> np.ndarray:
    J = parse_urdf_joints(io.StringIO(open(URDF, "r", encoding="utf-8").read()))
    T = np.eye(4)
    for j in chain_from_joints(J, a, b).joints:
        T = T @ j.origin()
    return T


def _params(name):
    with open(CFG_PATHS[name], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["/**"]["ros__parameters"]


def _visual_mesh() -> str:
    """The virtual arm URDF's EOAT visual mesh (the turned-torch mesh since 2026-10-13) as a file of eoat_description."""
    for m in xml.dom.minidom.parse(URDF).getElementsByTagName("mesh"):
        fn = m.getAttribute("filename")
        if m.parentNode.parentNode.tagName == "visual" and "eoat_description/meshes/eoat_base_link" in fn:
            return os.path.join(STACK, "eoat_description", "meshes", fn.rsplit("/", 1)[1])
    raise AssertionError("no EOAT visual mesh in the virtual arm's URDF")


def _wire_end_eoat() -> np.ndarray:
    """eoat ← CAD 와이어 끝 (토치의 가장 먼 점, 공칭 돌출 형상): URDF torch_tcp 에서 그 z 축 (노즐 축) 을 따라 메시의 와이어 (축에서
    0.6 mm 안의 정점) 가 가장 멀리 간 점 — the URDF's visual mesh (the torch turned in its clamp; test_eoat_frames 가 20.0 mm 로 고정)."""
    import struct
    T = _fixed("eoat_base_link", "torch_tcp")
    b = open(_visual_mesh(), "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    V = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))["v"].reshape(-1, 3)
    d = V.astype(float) - T[:3, 3]
    t = d @ T[:3, 2]
    r = np.linalg.norm(d - np.outer(t, T[:3, 2]), axis=1)
    return T[:3, 3] + float(t[r < 0.0006 + 1e-6].max()) * T[:3, 2]


def test_the_virtual_arm_urdf_carries_the_xacro_torch_tcp():
    dom_x = xml.dom.minidom.parse(EOAT_XACRO)
    d = dom_x.getElementsByTagName("xacro:macro")[0].getAttribute("params")
    macro = dict(re.findall(r"(\w+)='([^']*)'", d))
    xj = {j.getAttribute("name").replace("${prefix}", ""): j for j in dom_x.getElementsByTagName("joint")}
    uj = {j.getAttribute("name"): j for j in xml.dom.minidom.parse(URDF).getElementsByTagName("joint")}
    o = uj["torch_tcp_joint"].getElementsByTagName("origin")[0]
    assert [float(v) for v in o.getAttribute("xyz").split()] == [float(v) for v in macro["torch_xyz"].split()]
    assert [float(v) for v in o.getAttribute("rpy").split()] == [float(v) for v in macro["torch_rpy"].split()]
    for name in ("torch_clamp_joint", "weld_joint"):                  # the torch turned in its clamp (2026-10-13), as the xacro
        a, b = xj[name].getElementsByTagName("origin")[0], uj[name].getElementsByTagName("origin")[0]
        assert [float(v) for v in a.getAttribute("xyz").split()] == [float(v) for v in b.getAttribute("xyz").split()]
        assert [float(v) for v in a.getAttribute("rpy").split()] == [float(v) for v in b.getAttribute("rpy").split()]
        assert (xj[name].getElementsByTagName("parent")[0].getAttribute("link").replace("${prefix}", "")
                == uj[name].getElementsByTagName("parent")[0].getAttribute("link"))
    xm = [m.getAttribute("filename") for m in dom_x.getElementsByTagName("mesh")]
    um = [m.getAttribute("filename") for m in xml.dom.minidom.parse(URDF).getElementsByTagName("mesh") if "eoat_description" in
          m.getAttribute("filename")]
    assert xm == um and all("torch180" in f for f in um)


@pytest.mark.parametrize("name", LAB_PLATFORMS)
def test_lab_weld_torch_is_the_urdf_torch_tcp(name):
    """연구실·가짜 팔 설정: TCP = 이 팔 URDF 의 torch_tcp, 설정의 flange ← TCP 기록 = URDF 사슬 (TF 와 같은 값), 끝 = 와이어 끝."""
    p = _params(name)
    wt = weld_torch_config(p)
    assert wt.tcp_frame == "torch_tcp" == p["torch_frame"] and wt.flange_frame == "tcp" == p["flange_frame"] and wt.on_this_arm
    assert np.allclose(wt.flange_from_tcp.matrix(), _fixed("tcp", "torch_tcp"), atol=1e-9)
    T = _fixed("eoat_base_link", "torch_tcp")                                                 # the CAD wire stub, 20.0 mm along the
    assert np.allclose(_wire_end_eoat(), T[:3, 3] + 0.020 * T[:3, 2], atol=2e-6)              # nozzle axis of the turned torch
    assert np.array_equal(wt.tcp_from_tip.t, np.zeros(3))                                     # the plan's stick-out is a process value


def test_the_nozzle_end_derived_from_the_urdf_collision_geometry_is_the_cad_nozzle_rim():
    """2026-10-13 (user): the wire stick-out is measured from the gas nozzle end; the TCP (the contact-tip end face centre) is recessed
    inside the nozzle. The system derives the nozzle end from the robot's URDF collision geometry (the D14 tool; weld_clearance.
    nozzle_end_mm) - on the virtual arm's URDF it is the CAD nozzle rim 1.2 mm beyond the TCP (eoat test_eoat_frames pins the rim on the
    mesh); a part beside the axis would not count (with the CAD orientation the sensor guard, 15.9 mm beyond the TCP plane)."""
    from scan_core.weld_clearance import nozzle_end_mm, tool_body_from_urdf
    from scan_core.weld_torch import torch_with_nozzle_end
    body = tool_body_from_urdf(open(URDF, "r", encoding="utf-8").read(), "torch_tcp")
    assert nozzle_end_mm(body) == pytest.approx(1.2, abs=1e-3)
    # the torch turned in its clamp (2026-10-13): the nozzle axis leans away from the sensor - no part (the guard included) reaches
    # beyond the gas nozzle rim along the torch axis any more (with the CAD orientation the guard was 15.9 mm beyond the TCP plane)
    assert float(body.V[:, 2].max()) * 1e3 == pytest.approx(nozzle_end_mm(body), abs=1e-6)
    wt = torch_with_nozzle_end(weld_torch_config(_params("hongik_lab_rb20_moveit.yaml")), lambda: body)
    assert wt.with_stick_out(20.0).tip_offset_mm == pytest.approx(21.2, abs=1e-3)              # TCP -> wire end at the default


def test_kiro_weld_torch_is_the_turned_torch_at_the_calibration_files_mounting():
    """KIRO config (2026-10-13, KIRO's setup identified from its September weld logs = URDF v2 r4): the same torch turned 180 deg in
    its clamp about its straight body axis (make_torch_turned_mesh's turn of the CAD torch = the virtual arm URDF's / eoat.xacro's
    torch_tcp since the lab torch was re-mounted the same way, 2026-10-13), the EOAT mounted at the KIRO calibration file's eoat_yaw
    (-90 deg, the lab's); the same TCP <- tip axes as the lab's torch."""
    import importlib.util
    kiro, lab = weld_torch_config(_params("hongik_kiro.yaml")), weld_torch_config(_params("hongik_lab_rb20_moveit.yaml"))
    spec = importlib.util.spec_from_file_location("make_torch_turned_mesh",
                                                  os.path.join(STACK, "eoat_description", "scripts", "make_torch_turned_mesh.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = _turned()
    assert np.allclose(t["T_eoat_tcp"], t["T_turn"] @ mod.tcp_eoat(np.eye(4)), atol=1e-9)    # the CAD torch, turned
    assert np.allclose(t["T_eoat_tcp"], _fixed("eoat_base_link", "torch_tcp"), atol=1e-9)    # = the virtual arm URDF's torch_tcp
    yaw = C.load_calib(KIRO_CALIB).eoat_yaw_rad
    assert yaw == pytest.approx(-math.pi / 2, abs=1e-7)
    Rz = np.eye(4)
    Rz[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()                        # KIRO mounting (rbpodo flange convention)
    assert np.allclose(kiro.flange_from_tcp.matrix(), Rz @ t["T_eoat_tcp"], atol=1e-9)
    np.testing.assert_allclose(kiro.flange_from_tcp.t * 1e3, [-0.45, 237.86, 297.17], atol=0.006)    # the r4 numbers
    np.testing.assert_allclose(kiro.flange_from_tcp.R, [[0.0, 1.0, 0.0], [-0.866025, 0.0, 0.5], [0.5, 0.0, 0.866025]], atol=1e-6)
    assert np.array_equal(kiro.tcp_from_tip.matrix(), lab.tcp_from_tip.matrix())
    assert kiro.tcp_frame == "right_manipulator_ee_point" and kiro.on_this_arm


def test_the_cad_torch_fixture_is_the_cad_torchs_wire_end():
    """torch_fixtures.CAD_TORCH_KEYPOINT (the explicit CAD-torch fixture of rule / planner tests) = the CAD torch's wire end (the turn
    generator's unturned torch_tcp + the CAD wire stub 20 mm) relative to the calibrated optical origin."""
    import importlib.util
    from torch_fixtures import CAD_TORCH_KEYPOINT
    spec = importlib.util.spec_from_file_location("make_torch_turned_mesh",
                                                  os.path.join(STACK, "eoat_description", "scripts", "make_torch_turned_mesh.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    T = mod.tcp_eoat(np.eye(4))
    want = T[:3, 3] + 0.020 * T[:3, 2]
    p_opt = C.load_calib(CALIB).p_opt_eoat
    tip = np.array([*CAD_TORCH_KEYPOINT["torch_lateral_m"], p_opt[2] + CAD_TORCH_KEYPOINT["torch_protrusion_m"]])
    assert np.max(np.abs(tip - want)) < TOL_M, (tip, want)


@pytest.mark.parametrize("platform", LAB_PLATFORMS)
@pytest.mark.parametrize("template", LAB_TEMPLATES)
def test_envelope_tip_keypoint_is_the_wire_end(platform, template):
    """포락선의 토치 팁 키포인트 (scan_core.path_gen.tool_keypoints: (torch_lateral_m, p_opt z + torch_protrusion_m), eoat) = 토치의 가장 먼
    점 = 와이어 끝 (torch_tcp + tip_in_tcp) — 작업 템플릿의 torch_lateral_m, 플랫폼 safety.torch_protrusion_m (작업 값을 덮어쓴다; 템플릿
    값도 같아야 한다), 캘리브 광학 원점 (eoat_lab_01)."""
    p = _params(platform)
    with open(os.path.join(JOBS, template), "r", encoding="utf-8") as f:
        job = yaml.safe_load(f)["job"]
    p_opt = C.load_calib(CALIB).p_opt_eoat
    want = _wire_end_eoat()                                                       # the torch in the eoat frame (any mounting)
    for prot in (float(p["safety"]["torch_protrusion_m"]), float(job["torch_protrusion_m"])):
        tip = np.array([job["torch_lateral_m"][0], job["torch_lateral_m"][1], p_opt[2] + prot])
        assert np.max(np.abs(tip - want)) < TOL_M, (tip, want)


def test_kiro_envelope_tip_keypoint_is_the_turned_torchs_wire_end():
    """KIRO (the default 1G job template and the KIRO config, the fallbacks of the robot-model keypoint): the envelope's torch tip =
    the wire end of KIRO's turned torch at the template's process stick-out = TCP + (nozzle end + weld_plan.stick_out_mm) along the
    torch axis - the point the run takes from the robot model (AutoTeach._apply_safety_policy) - relative to the KIRO calibration
    file's optical origin."""
    from scan_core.weld_clearance import nozzle_end_mm, tool_body_from_urdf
    with open(os.path.join(JOBS, KIRO_TEMPLATE), "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    job, stick = doc["job"], float(doc["weld_plan"]["stick_out_mm"])
    noz = nozzle_end_mm(tool_body_from_urdf(open(URDF, "r", encoding="utf-8").read(), "torch_tcp"))     # the torch's nozzle rim (1.2)
    T = _turned()["T_eoat_tcp"]
    want = T[:3, 3] + T[:3, 2] * (noz + stick) * 1e-3
    p_opt = C.load_calib(KIRO_CALIB).p_opt_eoat
    for prot in (float(_params("hongik_kiro.yaml")["safety"]["torch_protrusion_m"]), float(job["torch_protrusion_m"])):
        tip = np.array([job["torch_lateral_m"][0], job["torch_lateral_m"][1], p_opt[2] + prot])
        assert np.max(np.abs(tip - want)) < TOL_M, (tip, want)


def test_lab_cell_fixture_keypoint_is_the_lab_torchs_wire_end():
    """The virtual-arm sims' lab cell (groove_world.lab_job_doc: the template + the lab specimen + LAB_TORCH_KEYPOINT) keeps the lab
    arm's torch: the envelope keypoint = the CAD wire end of the virtual arm's URDF torch_tcp, as the lab platforms' protrusion."""
    from groove_world import lab_job_doc
    job = lab_job_doc()["job"]
    p_opt = C.load_calib(CALIB).p_opt_eoat
    want = _wire_end_eoat()
    for platform in LAB_PLATFORMS:
        for prot in (float(_params(platform)["safety"]["torch_protrusion_m"]), float(job["torch_protrusion_m"])):
            tip = np.array([job["torch_lateral_m"][0], job["torch_lateral_m"][1], p_opt[2] + prot])
            assert np.max(np.abs(tip - want)) < TOL_M, (tip, want)
