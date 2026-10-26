"""scan_core.weld_torch: 용접 계획의 토치 모형 = 우리 1차 브라켓 토치 (사용자·KIRO 합의 2026-10-12) — 설정 읽기·검사, 끝 틀 → TCP 자세,
플랜지 ← 끝, D14 는 용접 토치가 이 팔에 있을 때만 (설정이 없다고 하면 건너뛰고 기록 — 다른 토치로 대신하지 않는다)."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scan_core.weld_torch import (D14_NO_WELD_TORCH, D14_NOT_ON_THIS_ARM, WeldTorch, d14_on_this_arm, reach_tool, torch_with_nozzle_end,
                                   weld_torch_config)

# our torch's gas nozzle end along +z_TCP [mm] (the contact-tip end = the TCP is recessed inside the nozzle): a test input with the CAD's
# value - the system derives it from the robot's URDF collision geometry (weld_clearance.nozzle_end_mm; scan_l0 test_torch_model_consistency
# derives it from the lab URDF)
NOZZLE_TEST_MM = 1.2

# 연구실 설정의 weld_torch 와 같은 값 (hongik_scan_ros/test/test_kiro_io 가 설정 파일 그대로를 URDF 와 대조한다)
TORCH = {"tcp_frame": "torch_tcp",
         "tip_in_tcp": {"axes": {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]},
                        "source": "our torch's TCP axes (test)", "status": "axes (test)"},
         "flange_frame": "tcp",
         "flange_from_tcp": {"xyz_m": [-0.00045202636338089293, 0.046554996667270435, 0.2971720033278718],
                             "quat_xyzw": [0.18301239077290107, -0.1830124832423317, 0.6830128022457533, 0.6830127434898017],
                             "source": "lab EOAT CAD (test)", "status": "nominal"},
         "on_this_arm": True, "source": "our 1st-bracket torch (test)", "status": "test"}


def test_config_reads_our_torch_and_checks_it():
    assert weld_torch_config({}) is None
    t = weld_torch_config({"weld_torch": TORCH})
    assert t.tcp_frame == "torch_tcp" and t.flange_frame == "tcp" and t.on_this_arm is True
    assert np.allclose(t.flange_from_tcp.t * 1e3, [-0.45, 46.55, 297.17], atol=0.005)            # the CAD contact-tip end on the flange
    # rpy (0, -30, +90) deg: the nozzle axis (TCP z) is bent 30 deg from the flange axis
    dev = Rotation.from_euler("xyz", [0.0, -30.0, 90.0], degrees=True).inv() * Rotation.from_matrix(t.flange_from_tcp.R)
    assert np.degrees(dev.magnitude()) < 0.001
    assert np.degrees(np.arccos(t.flange_from_tcp.R[2, 2])) == pytest.approx(30.0, abs=1e-3)
    assert np.array_equal(t.tcp_from_tip.R, np.eye(3)) and np.array_equal(t.tcp_from_tip.t, np.zeros(3)) and t.stick_out_mm == 0.0
    assert t.nozzle_end_mm is None                                               # on this arm: derived from the robot's URDF later
    with pytest.raises(ValueError, match="its nozzle end is not known - the wire stick-out is measured from the gas nozzle end"):
        t.with_stick_out(20.0)                                                   # never measured from the contact tip instead
    tn = t.with_nozzle_end(NOZZLE_TEST_MM, "test (CAD)")
    assert np.array_equal(tn.tcp_from_tip.t, np.zeros(3)) and t.nozzle_end_mm is None                          # a copy, nothing placed
    s20 = tn.with_stick_out(20.0, "job weld_plan.stick_out_mm (test)")           # the process stick-out (WeldProcess.stick_out_mm)
    assert np.allclose(s20.tcp_from_tip.t, [0.0, 0.0, 0.0212], atol=1e-15) and s20.stick_out_mm == 20.0       # nozzle end + stick-out
    assert s20.tip_offset_mm == pytest.approx(21.2) and s20.to_dict()["tip_offset_mm"] == pytest.approx(21.2)
    assert s20.to_dict()["stick_out_source"] == "job weld_plan.stick_out_mm (test)" and tn.stick_out_mm == 0.0   # a copy
    assert s20.to_dict()["nozzle_end_mm"] == NOZZLE_TEST_MM and s20.to_dict()["stick_out_definition"].startswith("gas nozzle end")
    tip = s20.flange_from_tip()[:3, 3] * 1e3                                     # the plan's tip = the wire end, 21.2 mm along the axis
    assert np.allclose(tip, [-0.45, 46.55 - 0.5 * 21.2, 297.17 + 21.2 * np.cos(np.radians(30.0))], atol=0.005)
    s20b = s20.with_nozzle_end(2.0, "test")                                      # a new nozzle end re-places the placed tip
    assert np.allclose(s20b.tcp_from_tip.t, [0.0, 0.0, 0.022], atol=1e-15) and s20b.stick_out_mm == 20.0
    with pytest.raises(ValueError, match="stick_out_mm must be >= 0"):
        tn.with_stick_out(-1.0)
    with pytest.raises(ValueError, match="nozzle_end_mm must be >= 0"):
        t.with_nozzle_end(-0.1)
    # a torch not on this arm carries its nozzle end in the config; on this arm it is derived (a config value is refused)
    off = WeldTorch.from_dict(dict(TORCH, on_this_arm=False, nozzle_end_mm=1.5))
    assert off.nozzle_end_mm == 1.5 and off.nozzle_end_source == "platform config weld_torch.nozzle_end_mm"
    with pytest.raises(ValueError, match="nozzle_end_mm: the weld torch is on this arm - its nozzle end is derived"):
        WeldTorch.from_dict(dict(TORCH, nozzle_end_mm=1.2))
    with pytest.raises(ValueError, match="nozzle_end_mm must be >= 0"):
        WeldTorch.from_dict(dict(TORCH, on_this_arm=False, nozzle_end_mm=-1.0))
    assert "torch_tcp on tcp: (-0.45, +46.55, +297.17) mm" in t.describe()
    d = t.to_dict()
    assert d["tcp_frame"] == "torch_tcp" and d["on_this_arm"] is True and json.dumps(d)
    for key in ("tcp_frame", "flange_frame", "tip_in_tcp", "flange_from_tcp", "on_this_arm"):
        bad = {k: v for k, v in TORCH.items() if k != key}
        with pytest.raises(ValueError, match=f"weld_torch.{key} is required"):
            WeldTorch.from_dict(bad)
    with pytest.raises(ValueError, match="unknown keys"):
        WeldTorch.from_dict(dict(TORCH, ee_from_tip={}))
    with pytest.raises(ValueError, match="true or false"):
        WeldTorch.from_dict(dict(TORCH, on_this_arm="maybe"))
    bad = json.loads(json.dumps(TORCH))
    bad["tip_in_tcp"]["xyz_m"] = [0.0, 0.0, 0.020]                               # the tip offset is a process input now
    with pytest.raises(ValueError, match="tip_in_tcp.xyz_m: removed"):
        WeldTorch.from_dict(bad)
    bad = json.loads(json.dumps(TORCH))
    bad["tip_in_tcp"]["axes"]["y"] = [0.0, -1.0, 0.0]
    with pytest.raises(ValueError, match="not a rotation"):
        WeldTorch.from_dict(bad)
    bad = json.loads(json.dumps(TORCH))
    bad["flange_from_tcp"]["quat_xyzw"] = [0.0, 0.0, 0.5, 0.5]
    with pytest.raises(ValueError, match="unit quaternion"):
        WeldTorch.from_dict(bad)


# 일반 기제: TCP 축이 끝 틀과 다른 토치 (축 대응 + 끝 오프셋) 도 같은 식으로 (우리 토치는 항등 축 + 끝 = 토치 축 20 mm 앞)
_PERMUTED = {"x": [0.0, 0.0, -1.0], "y": [0.0, -1.0, 0.0], "z": [-1.0, 0.0, 0.0]}


@pytest.mark.parametrize("axes,stick_mm,roll", [(None, 20.0, 0.0), (None, 20.0, 180.0), (None, 0.0, 0.0), (_PERMUTED, 0.0, 0.0),
                                                (_PERMUTED, 15.0, 180.0)],
                         ids=["our_torch_stick_out", "our_torch_stick_out_roll180", "no_stick_out", "permuted_axes",
                              "permuted_axes_stick_out_roll180"])
def test_tip_poses_become_tcp_poses_and_flange_poses(axes, stick_mm, roll):
    """계획의 끝 자세 (와이어 끝 = 겨냥점, [x 진행, y, z 토치 축]) → 토치 TCP 자세 (TCP 롤 roll): TCP = 겨냥점 − stick_out · (토치 축),
    회전 = 끝 · Rz(roll) · (TCP ← 끝)ᵀ (우리 토치: 항등 축 — TCP z = 토치 축, x = ±진행); 플랜지 ← 끝 (롤) = 플랜지 ← TCP · TCP ← 끝 ·
    Rz(roll)⁻¹ — 플랜지 자세에 그것을 곱하면 끝 자세로 돌아온다."""
    d = json.loads(json.dumps(TORCH))
    if axes is not None:
        d["tip_in_tcp"]["axes"] = axes
    t = WeldTorch.from_dict(d).with_nozzle_end(NOZZLE_TEST_MM).with_stick_out(stick_mm)
    off = NOZZLE_TEST_MM + stick_mm                                              # TCP -> tip: nozzle end + stick-out
    from scan_core.weld_torch import roll_matrix
    rng = np.random.default_rng(3)
    R_tip = Rotation.random(5, random_state=7).as_matrix()
    P_tip = rng.normal(0.0, 0.5, (5, 3))
    P_tcp, R_tcp = t.tcp_poses(P_tip, R_tip, roll_deg=roll)
    A = np.column_stack([d["tip_in_tcp"]["axes"][c] for c in "xyz"]).astype(float)    # the tip frame's axes in the TCP frame
    assert np.allclose(R_tcp @ A, R_tip @ roll_matrix(roll), atol=1e-12)
    axis = R_tip[:, :, 2]                                                        # the torch axis (roll about it keeps it)
    assert np.allclose(P_tcp, P_tip - off * 1e-3 * axis, atol=1e-15)             # TCP = aim point - (nozzle end + stick_out) x axis
    if axes is None:
        assert np.allclose(R_tcp[:, :, 2], axis, atol=1e-15)                     # our torch: z_TCP = the torch axis
        assert np.allclose(R_tcp[:, :, 0], (1.0 if roll == 0.0 else -1.0) * R_tip[:, :, 0], atol=1e-12)   # x = +/- travel
    for i in range(5):
        T_tip = np.eye(4)
        T_tip[:3, :3], T_tip[:3, 3] = R_tip[i], P_tip[i]
        T_flange = T_tip @ np.linalg.inv(t.flange_from_tip(roll))
        assert np.allclose(T_flange @ t.flange_from_tip(roll), T_tip, atol=1e-12)
        T_tcp = np.eye(4)
        T_tcp[:3, :3], T_tcp[:3, 3] = R_tcp[i], P_tcp[i]
        assert np.allclose(T_flange @ t.flange_from_tcp.matrix(), T_tcp, atol=1e-12)
    p1, r1 = t.tcp_poses(P_tip[0], R_tip[0], roll_deg=roll)                      # one pose
    assert np.allclose(p1, P_tcp[0]) and np.allclose(r1, R_tcp[0])


def test_d14_runs_only_with_the_weld_torch_on_this_arm():
    """사용자 결정 2026-10-12: D14 (용접 계획 도달성) 는 용접 토치가 이 팔에 있을 때만 (연구실·KIRO: 우리 토치) — 그 로봇 TF 의 플랜지 ←
    TCP · TCP ← 끝, 도구 설명에 TF 에서 읽은 TCP 위치. 설정이 이 팔에 없다고 하면 하지 않고 기록 문구를 남긴다 (다른 토치로 대신하지
    않는다: reach_tool 이 거부, 아무것도 찾지 않는다)."""
    T_ft = np.eye(4)                                                             # the robot's own TF: tcp <- torch_tcp
    T_ft[:3, :3] = Rotation.from_euler("xyz", [0.0, -30.0, 90.0], degrees=True).as_matrix()
    T_ft[:3, 3] = [-0.00045, 0.04655, 0.29717]
    asked = []

    def lookup(a, b):
        asked.append((a, b))
        return {("tcp", "torch_tcp"): T_ft}[(a, b)]
    off = WeldTorch.from_dict(dict(TORCH, on_this_arm=False))
    assert d14_on_this_arm(off) == (False, D14_NOT_ON_THIS_ARM) and d14_on_this_arm(None) == (False, D14_NO_WELD_TORCH)
    assert D14_NOT_ON_THIS_ARM == ("weld-plan reachability (D14) not checked: the platform config says the weld torch is not mounted on "
                                   "this arm (weld_torch.on_this_arm false); it is checked on the arm that carries it")
    assert off.to_dict()["reachability_d14"] == D14_NOT_ON_THIS_ARM
    for t in (off, None):
        with pytest.raises(ValueError, match=r"reachability \(D14\) not checked"):
            reach_tool(t, lookup, "tcp")
    assert asked == []                                                           # nothing looked up: no other torch as a stand-in
    on = WeldTorch.from_dict(TORCH)
    assert d14_on_this_arm(on) == (True, "") and on.to_dict()["reachability_d14"].startswith("checked on this arm")
    on20 = on.with_nozzle_end(NOZZLE_TEST_MM).with_stick_out(20.0)
    T, why = reach_tool(on20, lookup, "tcp")
    assert asked == [("tcp", "torch_tcp")]
    assert why == ("the weld torch torch_tcp (weld_torch, on this arm: tcp <- torch_tcp from the robot's TF: (-0.45, +46.55, +297.17) mm)")
    assert np.allclose(T, T_ft @ on20.tcp_from_tip.matrix(), atol=1e-15)
    assert np.allclose(T[:3, :3], T_ft[:3, :3]) and np.allclose(T[:3, 3], T_ft[:3, 3] + 0.0212 * T_ft[:3, 2])  # tip = TCP + 21.2 mm z
    T180, _ = reach_tool(on20, lookup, "tcp", roll_deg=180.0)                    # the TCP rolled 180 deg about the torch axis
    assert np.allclose(T180[:3, 3], T[:3, 3]) and np.allclose(T180[:3, 2], T[:3, 2]) and np.allclose(T180[:3, 0], -T[:3, 0])




def _ring(r0, r1, z0, z1, n=24, cx=0.0, cy=0.0):
    """A closed tube (annulus r0..r1 between z0 and z1) around the z axis through (cx, cy) as triangles [m]."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    tri = []
    for i in range(n):
        j = (i + 1) % n
        for (ra, za), (rb, zb) in (((r1, z0), (r1, z1)), ((r0, z1), (r0, z0)), ((r0, z0), (r1, z0)), ((r1, z1), (r0, z1))):
            p = lambda r, z, k: [cx + r * np.cos(a[k]), cy + r * np.sin(a[k]), z]      # noqa: E731
            tri += [[p(ra, za, i), p(rb, zb, i), p(rb, zb, j)], [p(ra, za, i), p(rb, zb, j), p(ra, za, j)]]
    return np.array(tri)


def test_nozzle_end_is_the_farthest_point_of_the_parts_the_torch_axis_passes_through():
    """weld_clearance.nozzle_end_mm (user 2026-10-13: the stick-out is measured from the gas nozzle end): of the tool parts whose outline
    seen along the torch axis encloses it - the nozzle tube around the recessed contact tip, the contact tip - the farthest along +z_TCP;
    a part beside the axis reaching farther (the sensor guard) does not count; no part around the axis -> ValueError. torch_with_nozzle_end
    keeps a known end, derives it on this arm from the tool geometry, and refuses a torch not on this arm without the config value."""
    from scan_core.weld_clearance import nozzle_end_mm, tool_body_from_triangles
    nozzle = _ring(0.008, 0.010, -0.030, 0.0012)                                # the gas nozzle: rim 1.2 mm beyond the TCP plane
    tip = _ring(0.0006, 0.0031, -0.020, 0.0)                                    # the contact tip: its end face = the TCP plane
    guard = _ring(0.001, 0.005, -0.050, 0.016, cx=0.060)                        # beside the axis, 16 mm beyond the TCP plane
    body = tool_body_from_triangles(np.concatenate([nozzle, tip, guard]), tcp_frame="torch_tcp", source="test")
    assert nozzle_end_mm(body) == pytest.approx(1.2, abs=1e-9)
    with pytest.raises(ValueError, match="no part of the tool encloses the torch axis"):
        nozzle_end_mm(tool_body_from_triangles(guard, tcp_frame="torch_tcp", source="test"))
    on = WeldTorch.from_dict(TORCH)
    calls = []
    got = torch_with_nozzle_end(on, lambda: calls.append(1) or body)
    assert got.nozzle_end_mm == pytest.approx(1.2) and got.nozzle_end_source.startswith("derived from the robot's URDF collision geometry")
    assert torch_with_nozzle_end(got, lambda: calls.append(1) or body) is got and calls == [1]       # known: kept, nothing derived
    assert torch_with_nozzle_end(None, None) is None
    with pytest.raises(ValueError, match="gives no tool collision geometry"):
        torch_with_nozzle_end(on, None)
    with pytest.raises(ValueError, match="not on this arm - its nozzle end .* must be given in the platform config"):
        torch_with_nozzle_end(WeldTorch.from_dict(dict(TORCH, on_this_arm=False)), lambda: body)
    off = WeldTorch.from_dict(dict(TORCH, on_this_arm=False, nozzle_end_mm=1.5))
    assert torch_with_nozzle_end(off, None) is off
