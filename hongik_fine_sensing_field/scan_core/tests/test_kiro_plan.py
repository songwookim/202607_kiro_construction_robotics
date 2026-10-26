"""scan_core.kiro_plan: KIRO 형식 용접 계획 출력 (사용자 승인 2026-10-12 (A)) — KIRO 표본 (pass_1–4.yaml, xy_endpoint_review_v1) 과
글자까지 같은 형식 (층마다 직선 시작 → 끝 자세), 프레임 사슬 (고정 World ← base · ee ← 와이어 끝) 왕복, 자세 정책 (planned / reference /
auto), KIRO 로그 키 이름의 공유용 정보 파일 (명령 아님), 설정 검사, 오프라인 재생성 (추정·사전 JSON 왕복 + CLI)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from g1_synth import synth_estimate, synth_prior
from scan_core import kiro_plan as KP
from scan_core.g1_outputs import weld_plan_from_estimate, write_outputs
from scan_core.g1_types import SeamEstimate, SeamPrior
from scan_core.weld_plan import WeldProcess
from scan_core.weld_torch import WeldTorch

from test_weld_torch import NOZZLE_TEST_MM

SAMPLES = Path(__file__).resolve().parent / "data" / "kiro_samples"      # KIRO 의 pass_1–4.yaml (weld_testspecimen_logs, 2026-09-15/16)
# 설정 (KIRO 플랫폼 설정의 kiro_output·weld_torch 와 같은 값 — hongik_scan_ros/test/test_kiro_io 가 설정 파일 그대로를 본다)
CFG = {"enabled": True, "world_frame": "World",
       "world_from_base": {"xyz_m": [-0.24982842514061013, -0.49970893784963016, 1.330424871627074],
                           "quat_xyzw": [-6.094245571196422e-05, 3.808556282163003e-07, 0.00012641046925126396, 0.9999999901531326],
                           "source": "FK fit (test)", "status": "fitted"},
       }
# 용접 계획의 토치 모형 = 우리 1차 브라켓 토치 (사용자·KIRO 합의 2026-10-12; KIRO 설정: TCP right_manipulator_ee_point = 토치 팁 끝 =
# CAD 컨택트 팁 끝, 항등 축, 계획의 끝 = 와이어 끝 = TCP + 20 mm z; KIRO 장착 eoat_yaw +90° — 연구실 설정은 torch_tcp, −90°)
TORCH = {"tcp_frame": "right_manipulator_ee_point",
         "tip_in_tcp": {"axes": {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]},
                        "source": "our torch's TCP axes (test)", "status": "axes of our torch's TCP (test)"},
         "flange_frame": "hongik_flange",
         "flange_from_tcp": {"xyz_m": [0.00045202386850825524, -0.04655499669149436, 0.2971720033278718],
                             "quat_xyzw": [-0.18301248814612972, -0.18301238586910046, -0.6830127617910589, 0.6830127839444973],
                             "source": "lab EOAT CAD, KIRO mounting eoat_yaw +90 deg (test)", "status": "nominal CAD; KIRO re-calibrates the TCP on site"},
         "on_this_arm": True, "source": "our 1st-bracket torch (test)",
         "status": "nominal TCP = the CAD contact-tip end of our torch (KIRO's TCP = the torch tip end, KIRO 2026-10-12)"}
# KIRO 9 월 용접의 각 (WPS 진행각·작업각의 출처, 사용자 2026-10-13; fsv1_work/kiro_dataset2/toolid): 식별한 토치 = 우리 1차 브라켓을 장착
# −90° 로, 토치를 클램프에서 곧은 몸통 축 둘레로 173.9° 돌린 것 (CAD 30° 굽음) — KIRO 가 기록한 ee_point (옛 URDF 의 right_manipulator_ee_point,
# z = 플랜지 축) 좌표계에서 노즐 축 = Rz(−36.1°) Ry(−30°) z. 패스 (root, 충전 1, 충전 2, 캡) 마다 weld_start 의 AWS 각 (진행각 + 미는 각,
# 작업각 + = 진행의 오른쪽으로 팁이 기욺; 판 법선 = World +z, 용접 축 = 그 세션의 패스 1 직선)
LOGGED_EE_NOZZLE_RPY_DEG = (0.0, -30.0, -36.1)
KIRO_SEPT_0915 = [(-44.33, -11.47), (-44.50, 20.58), (-44.08, -7.02), (-44.24, -6.98)]     # 2026-09-15/16 (pass_1-4.yaml = SAMPLES)
KIRO_SEPT_0922 = [(-39.78, -6.59), (-38.17, 19.32), (-43.77, -7.06), (-44.75, -6.79)]      # 2026-09-22 (toolid, the same reading)


TIP_MM = 20.0 + NOZZLE_TEST_MM          # TCP -> wire end at the default stick-out (20 mm from the nozzle end) [mm]


def _cfg(cfg=None, torch=None):
    t = WeldTorch.from_dict(torch or TORCH)
    if t.nozzle_end_mm is None:                       # on a platform: derived from the robot's URDF (teach_1g); here the CAD value
        t = t.with_nozzle_end(NOZZLE_TEST_MM, "test (the CAD nozzle end)")
    return KP.KiroOutputConfig.from_dict(cfg or CFG, t)
# KIRO 로그의 키 ([commanded] 와 [execution_conditions] 에서 우리가 쓰는 것; 1.log / 4.log)
KIRO_COMMANDED = {"current_a", "voltage", "material", "diameter_mm", "mode", "gas"}
KIRO_EXECUTION = {"weld_tcp_speed_mm_s", "weld_weave_enabled", "weld_weave_pattern", "weld_weave_pitch_mm", "weld_weave_left_dwell_s",
                  "weld_weave_right_dwell_s", "weld_weave_amplitude_mm", "weld_weave_cycles", "weld_weave_samples_per_cycle",
                  "weld_weave_axis", "weld_lead_in_mm", "weld_lead_out_mm", "weld_arc_off_delay_ms"}


def _sig(v):
    """형 서명: dict → [(키, 서명)] (순서 포함), 스칼라 → 형 이름."""
    if isinstance(v, dict):
        return [(k, _sig(x)) for k, x in v.items()]
    return type(v).__name__


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    pr = synth_prior()
    est = synth_estimate(pr)
    out = tmp_path_factory.mktemp("g1_kiro")
    res = write_outputs(est, pr, WeldProcess(), out, k=pr.k, base_frame="link0")
    return res["plan"]


def _write(plan, out, cfg=None, torch=None):
    c = _cfg(cfg, torch)
    wt, notes = KP.resolve_world_from_base(c, "link0")
    return c, wt, KP.write_kiro_outputs(plan, out, c, world_from_base=wt, notes=notes)


# --------------------------------------------------------------------------------------
# 형식: KIRO 표본과 글자까지 같다
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_pass_yaml_reproduces_the_kiro_samples_character_for_character(k):
    f = SAMPLES / f"pass_{k}.yaml"
    r = KP.read_pass_yaml(f)
    doc = KP.pass_doc(r["pass"], r["frame_id"], *r["start"], *r["goal"])
    assert KP.format_pass_yaml(doc) == f.read_text(encoding="utf-8")
    assert _sig(yaml.safe_load(KP.format_pass_yaml(doc))) == _sig(yaml.safe_load(f.read_text(encoding="utf-8")))
    assert list(doc) == ["schema", "pass", "frame_id", "requires_ik", "current_start", "current_goal"]


def test_written_pass_files_have_kiro_keys_order_and_types(plan, tmp_path):
    _c, _wt, ko = _write(plan, tmp_path)
    ref = _sig(yaml.safe_load((SAMPLES / "pass_1.yaml").read_text(encoding="utf-8")))
    names = sorted(Path(f).name for f in ko["files"])
    assert names == sorted(["README.md"] + [f"pass_{k}.yaml" for k in (1, 2, 3, 4)] + [f"pass_{k}_info.yaml" for k in (1, 2, 3, 4)])
    for k in (1, 2, 3, 4):
        text = (tmp_path / "kiro" / f"pass_{k}.yaml").read_text(encoding="utf-8")
        d = yaml.safe_load(text)
        assert _sig(d) == ref and d["pass"] == k and d["schema"] == "xy_endpoint_review_v1" and d["frame_id"] == "World"
        assert d["requires_ik"] is True and text.endswith("\n") and "\t" not in text
        assert KP.format_pass_yaml(d) == text                                             # 왕복
    readme = (tmp_path / "kiro" / "README.md").read_text(encoding="utf-8")
    assert "they are not commands" in readme and "The weld torch is on the arm that made this run" in readme
    assert "`<dirname(data_file_path)>/kiro/pass_<k>.yaml`" in readme and "embeds the same yaml text" in readme
    assert "the TCP `right_manipulator_ee_point` = the torch tip end centre" in readme
    assert "moved back along -z_TCP (the torch axis) by 21.20 mm = the TCP's recess inside the gas nozzle 1.20 mm (test (the CAD" in readme
    assert ("+ the wire stick-out 20 mm measured from the gas nozzle end (job weld_plan.stick_out_mm 20 mm (the wire stick-out from the GAS "
            "NOZZLE END") in readme
    assert "KIRO's executor does not apply a" in readme and "paths are the aim-point (wire-tip) lines" in readme
    assert "(0.00, 0.00, 21.20) mm, axes identity" in readme.replace("+", "") and "KIRO re-calibrates the TCP on site" in readme
    # no torch model in this plan: no neck bend to orient, every roll fits - the NECK_RULE record says so (no silent default)
    assert "TCP roll about the torch axis: 0 deg - roll 0 deg: every roll fits these travel angles (NECK_RULE)" in readme
    assert "reference" not in readme
    assert ko["stick_out_mm"] == 20.0 and ko["stick_out_source"].startswith("job weld_plan.stick_out_mm 20 mm (the wire stick-out from the "
                                                                            "GAS NOZZLE END")
    assert ko["nozzle_end_mm"] == NOZZLE_TEST_MM and ko["tcp_to_aim_point_mm"] == pytest.approx(TIP_MM)
    assert "PrecisionSensing action result" in readme


# --------------------------------------------------------------------------------------
# 프레임 사슬: link0 의 합성 계획 → 맞춤 World ← link0 → World 자세 = 예상, 거꾸로 되돌리면 같은 끝점
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("stick_mm,roll", [(20.0, 0.0), (20.0, 180.0), (0.0, 0.0), (12.5, 180.0)],
                         ids=["stick20_roll0", "stick20_roll180", "no_stick_out", "stick12.5_roll180"])
def test_frame_round_trip_through_the_fitted_world_transform(plan, tmp_path, stick_mm, roll):
    """TCP (토치 팁 끝 중심) = 겨냥점 − (노즐 끝 + stick_out) · z_TCP (돌출 = 노즐 끝에서), 자세 = 계획 토치 자세 · Rz(롤) — 맞춤
    World ← link0 로 쓰고 거꾸로 되돌리면 같은 끝점; 토치 축은 롤과 무관하게 계획의 토치 축, x_TCP = ±진행."""
    import dataclasses
    from scan_core.weld_torch import roll_matrix
    c = _cfg()
    p2 = dataclasses.replace(plan, torch=c.torch.with_stick_out(stick_mm, "test"), torch_roll_deg=roll,
                             torch_roll={"deg": roll, "reason": f"test roll {roll:g}"})
    c, wt, ko = _write(p2, tmp_path)
    wb = CFG["world_from_base"]
    R_Wb = Rotation.from_quat(wb["quat_xyzw"]).as_matrix()                 # 독립 계산 (설정 값에서 바로)
    t_Wb = np.array(wb["xyz_m"])
    assert wt.source == "FK fit (test)" and wt.base_frame == "link0" and ko["stick_out_mm"] == stick_mm
    for wp in p2.passes:
        d = yaml.safe_load((tmp_path / "kiro" / f"pass_{wp.k}.yaml").read_text(encoding="utf-8"))
        ends = KP.pass_endpoints(p2, wp)
        R_bt = KP.planned_rotation(wp)
        R_W_ee = R_Wb @ R_bt @ roll_matrix(roll)
        for X, key in (("start", "current_start"), ("goal", "current_goal")):
            p = np.array([d[key]["position_m"][c_] for c_ in "xyz"])
            q = np.array([d[key]["orientation_xyzw"][c_] for c_ in "xyzw"])
            R = Rotation.from_quat(q).as_matrix()
            assert np.allclose(R, R_W_ee, atol=1e-12) and q[3] >= 0.0
            tip_W = R_Wb @ ends[X]["tip_base_m"] + t_Wb
            off = (NOZZLE_TEST_MM + stick_mm) * 1e-3                                           # nozzle end + stick-out
            assert np.allclose(p + off * R[:, 2], tip_W, atol=1e-12)                           # TCP + (nozzle + stick) x z_TCP = aim point
            back = R_Wb.T @ (p + off * R[:, 2] - t_Wb)                                         # 거꾸로: World TCP → 겨냥점 → base
            assert np.allclose(back, ends[X]["tip_base_m"], atol=1e-12)
            assert np.allclose(R_Wb.T @ R[:, 2], R_bt[:, 2], atol=1e-12)                       # z_TCP = the planned torch axis
            assert np.allclose(R_Wb.T @ R[:, 0], (1.0 if roll == 0.0 else -1.0) * R_bt[:, 0], atol=1e-12)   # x_TCP = +/- travel
        # 끝점 = 그 층 높이의 심 끝 (G1WeldPlan.layer_ends; 합성 = S, E) — 진행 방향의 시작 끝에서
        le = p2.layer_ends[wp.k]
        s0, s1 = (le["S"]["s_m"], le["E"]["s_m"]) if wp.direction == "S->E" else (le["E"]["s_m"], le["S"]["s_m"])
        assert ends["start"]["s_mm"] == pytest.approx(s0 * 1e3) and ends["goal"]["s_mm"] == pytest.approx(s1 * 1e3)
        row = [r for r in ko["passes"] if r["k"] == wp.k][0]
        assert row["torch_roll_deg"] == roll and row["written_travel_angle_deg"] == pytest.approx(wp.travel_angle_deg, abs=0.2)
        assert row["written_work_angle_deg"] == pytest.approx(wp.work_angle_deg, abs=0.2)
        info = yaml.safe_load((tmp_path / "kiro" / f"pass_{wp.k}_info.yaml").read_text(encoding="utf-8"))["derived"]
        assert info["orientation"]["torch_roll_deg"] == roll and info["orientation"]["torch_roll_reason"] == f"test roll {roll:g}"
        assert info["frames"]["stick_out_mm"] == stick_mm and np.allclose(info["frames"]["aim_point_World_m"]["start"],
                                                                          R_Wb @ ends["start"]["tip_base_m"] + t_Wb, atol=1e-12)
        rule = info["aim_offset_mm"]["rule"]                       # lateral = KIRO's practice (2026-10-13 night), the 09-22 data a check
        assert rule["lateral"].startswith("KIRO's practice") and rule["this_pass"] == "kiro_practice" and rule["why"] == wp.aim_why
        assert "current surface" in rule["vertical"] and rule["verification_0922"]["k"] == wp.k
        lm = info["layer_mm"]                                       # the pass's convex bead (bead_geometry)
        assert lm["top"]["median"] == pytest.approx(float(np.median(wp.layer_to_mm))) and lm["crown_ratio"]["median"] > 0.0
        assert lm["toe_left_z"]["median"] == pytest.approx(float(np.median(wp.bead_xz("toe_l")[:, 1])))


def test_lab_definition_world_is_the_base_and_the_tcp_is_our_torch_on_this_arm(plan, tmp_path):
    """연구실 설정 (World := base link0 항등, 용접 토치 = 이 팔에 달린 우리 토치 torch_tcp — 2026-10-12): 같은 코드 경로로 쓴 pass_<k>.yaml 은
    우리 base 프레임의 토치 TCP 자세 — 위치 = 끝점 (끝 = TCP 원점), 자세 = 계획 토치 자세의 회전 평균 (항등 축: TCP z = 노즐 축 = 계획 토치
    축, x = 진행). README 가 World 를 base 로 정의했다고, 그 토치가 이 팔에 있고 D14 를 이 토치로 확인했다고, TCP 가 공칭이라고 밝힌다.
    설정이 토치가 이 팔에 없다고 하면 (일반 기제) 그 기록 문구."""
    lab = json.loads(json.dumps(CFG))
    lab["world_from_base"].update(xyz_m=[0.0, 0.0, 0.0], quat_xyzw=[0.0, 0.0, 0.0, 1.0], source="lab definition (test)")
    torch = dict(json.loads(json.dumps(TORCH)), tcp_frame="torch_tcp", flange_frame="tcp",
                 flange_from_tcp={"xyz_m": [-0.00045202636338089293, 0.046554996667270435, 0.2971720033278718],
                                  "quat_xyzw": [0.18301239077290107, -0.1830124832423317, 0.6830128022457533, 0.6830127434898017],
                                  "source": "lab EOAT CAD (test)", "status": "nominal (CAD contact-tip end)"})
    c, _wt, ko = _write(plan, tmp_path, lab, torch=torch)
    for wp in plan.passes:
        d = yaml.safe_load((tmp_path / "kiro" / f"pass_{wp.k}.yaml").read_text(encoding="utf-8"))
        e = KP.pass_endpoints(plan, wp)
        R = Rotation.from_quat([d["current_goal"]["orientation_xyzw"][c_] for c_ in "xyzw"]).as_matrix()
        assert np.allclose(R, KP.planned_rotation(wp), atol=1e-12) and d["frame_id"] == "World"           # 항등 축: TCP 자세 = 끝 자세
        for X, key in (("start", "current_start"), ("goal", "current_goal")):   # TCP = 겨냥점 − (노즐 끝 1.2 + 돌출 20) mm 토치 축 (공정 기본)
            assert np.allclose([d[key]["position_m"][c_] for c_ in "xyz"], e[X]["tip_base_m"] - TIP_MM * 1e-3 * R[:, 2], atol=1e-15)
        assert np.allclose(R[:, 2], KP.planned_rotation(wp)[:, 2], atol=1e-12)                             # 토치 축 = z (노즐 축)
        info = yaml.safe_load((tmp_path / "kiro" / f"pass_{wp.k}_info.yaml").read_text(encoding="utf-8"))
        wtd = info["derived"]["frames"]["weld_torch"]
        assert wtd["on_this_arm"] is True and wtd["flange_frame"] == "tcp" and wtd["status"].startswith("nominal TCP")
        assert wtd["reachability_d14"].startswith("checked on this arm") and wtd["flange_from_tcp_status"].startswith("nominal")
        assert np.allclose(wtd["flange_from_tcp_xyz_mm"], [-0.45, 46.55, 297.17], atol=0.005)
        assert info["derived"]["frames"]["ee_from_tip_xyz_mm"] == pytest.approx([0.0, 0.0, TIP_MM])
        assert info["derived"]["frames"]["stick_out_mm"] == 20.0 and "torch tip end centre" in info["derived"]["frames"]["tcp_definition"]
        assert info["derived"]["frames"]["nozzle_end_mm"] == NOZZLE_TEST_MM and info["derived"]["frames"]["tcp_to_aim_point_mm"] == \
            pytest.approx(TIP_MM) and "measured from the gas nozzle end" in info["derived"]["frames"]["tcp_definition"]
        assert np.allclose(wtd["flange_from_tcp_rpy_deg"], [0.0, -30.0, 90.0], atol=0.001)
    readme = (tmp_path / "kiro" / "README.md").read_text(encoding="utf-8")
    assert "Here World is defined as the robot base frame `link0`" in readme and "`torch_tcp`" in readme
    assert "The weld torch is on the arm that made this run" in readme and "(-0.45, +46.55, +297.17) mm" in readme
    assert "replace with KIRO's value" in readme and "not checked" not in readme
    assert ko["weld_torch"]["on_this_arm"] is True and ko["tcp_frame"] == "torch_tcp"
    # 일반 기제: 설정이 용접 토치가 이 팔에 없다고 하면 README·info 에 그 문구
    c, _wt, ko = _write(plan, tmp_path / "off", lab, torch=dict(torch, on_this_arm=False))
    readme = (tmp_path / "off" / "kiro" / "README.md").read_text(encoding="utf-8")
    from scan_core.weld_torch import D14_NOT_ON_THIS_ARM
    assert "The platform config says the weld torch is not on the arm that made this run" in readme and D14_NOT_ON_THIS_ARM in readme
    info = yaml.safe_load((tmp_path / "off" / "kiro" / "pass_1_info.yaml").read_text(encoding="utf-8"))
    assert info["derived"]["frames"]["weld_torch"]["reachability_d14"] == D14_NOT_ON_THIS_ARM


def test_endpoints_carry_the_layer_aim_offsets_and_the_job_insets(tmp_path):
    """겨냥 오프셋 = 층 계획 (단면 x, z), 끝점 = 층 끝에서 작업 inset 만큼 안쪽 (진행 방향 기준; E->S 패스는 E 에서 시작)."""
    pr = synth_prior()
    est = synth_estimate(pr)
    plan = weld_plan_from_estimate(est, pr.spec, WeldProcess(direction="E->S", start_inset_mm=2.0, end_inset_mm=3.0), k=pr.k)
    for wp in plan.passes:
        e = KP.pass_endpoints(plan, wp)
        le = plan.layer_ends[wp.k]
        assert e["start"]["s_mm"] == pytest.approx(le["E"]["s_m"] * 1e3 - 2.0) and e["goal"]["s_mm"] == pytest.approx(le["S"]["s_m"] * 1e3 + 3.0)
        o = np.argsort(wp.s_mm)
        for X in ("start", "goal"):
            x, z = e[X]["aim_xz_mm"]
            assert x == pytest.approx(np.interp(e[X]["s_mm"], wp.s_mm[o], wp.aim_xz_mm[o, 0]))
            assert z == pytest.approx(np.interp(e[X]["s_mm"], wp.s_mm[o], wp.aim_xz_mm[o, 1]))


def test_wps_angles_are_kiros_september_welds_through_the_identified_tool():
    """WPS 진행각·작업각 (WeldProcess 기본값, 사용자 2026-10-13) = KIRO 가 9 월에 실제로 용접한 각: KIRO 표본 자세 (2026-09-15/16
    pass_1–4.yaml 의 weld_start) 를 식별한 토치의 노즐 축으로 읽으면 toolid 의 09-15/16 각이 나오고 (KIRO 의 부호: + = 진행의 오른쪽), 기본값
    = 두 세션 (09-15/16, 09-22) 평균을 1° 로 반올림한 값 — 작업각 부호: 베벨은 KIRO 진행의 왼쪽 (사용자 2026-10-13 밤; provenance 가 그렇게
    말한다) → 우리 부호 (+ = 베벨 쪽) = KIRO 부호의 반대: +9 / -20 / +7 / +7."""
    proc = WeldProcess()
    assert proc.provenance["angles"].startswith(
        "KIRO welds 2026-09-15/16 + 09-22, tool identified (mount -90 deg, torch flipped), nozzle axis; work-angle sign: the "
        "bevel on the LEFT of KIRO's travel (user 2026-10-13 night - corrects the earlier 'right')")
    r_, p_, y_ = LOGGED_EE_NOZZLE_RPY_DEG
    a_ee = Rotation.from_euler("ZYX", [y_, p_, r_], degrees=True).as_matrix()[:, 2]
    r1 = KP.read_pass_yaml(SAMPLES / "pass_1.yaml")
    d = r1["goal"][0] - r1["start"][0]                                        # the session's weld axis: the pass-1 line
    n = np.array([0.0, 0.0, 1.0])
    for k, ref in zip((1, 2, 3, 4), KIRO_SEPT_0915):
        R = Rotation.from_quat(KP.read_pass_yaml(SAMPLES / f"pass_{k}.yaml")["start"][1]).as_matrix()
        ang = KP._angles(R @ a_ee, d, n, np.cross(d, n))                      # KIRO's sense: + = toward the right of travel
        assert (ang["travel_angle_deg"], ang["work_angle_deg"]) == pytest.approx(ref, abs=0.1), (k, ang)
        bev = KP._angles(R @ a_ee, d, n, np.cross(n, d))                      # ours: + = toward the bevel = the left of travel
        assert bev["work_angle_deg"] == pytest.approx(-ref[1], abs=0.1), (k, bev)
    for (role, occ), a, b in zip((("root", 0), ("fill", 0), ("fill", 1), ("cap", 0)), KIRO_SEPT_0915, KIRO_SEPT_0922):
        assert proc.role_angle("travel_angle_deg", role, occ) == round(0.5 * (a[0] + b[0])), (role, occ)
        assert proc.role_angle("work_angle_deg", role, occ) == -round(0.5 * (a[1] + b[1])), (role, occ)


def test_kiros_logged_orientations_would_lay_our_torch_flat():
    """사용자 2026-10-12: KIRO 로그는 이 토치로 기록되지 않았다 — 로그의 용접 자세 (pass_1–4.yaml) 를 우리 토치의 TCP 규약 (+z = 노즐 축,
    공작물 쪽) 으로 읽으면 토치 축 z_ee 가 거의 수평 (아래에서 70° 넘게) 이고 진행 반대쪽을 향한다 — 토치가 심을 따라 눕는다. 그래서
    'reference' 정책 (KIRO 자세 그대로) 은 우리 토치에 뜻이 없고 (설정 기본은 planned), 로그의 프레임 규약은 토치 축에 쓰지 않는다."""
    c = _cfg()
    assert np.array_equal(c.ee_from_tip.R, np.eye(3))                            # the config: our torch's TCP axes = the tip frame
    for k in (1, 2, 3, 4):
        r = KP.read_pass_yaml(SAMPLES / f"pass_{k}.yaml")
        R = Rotation.from_quat(r["start"][1]).as_matrix()
        down = np.array([0.0, 0.0, -1.0])
        tilt_axis = math.degrees(math.acos(float((R @ c.ee_from_tip.R[:, 2]) @ down)))
        assert tilt_axis > 70.0, (k, tilt_axis)
        d = r["goal"][0] - r["start"][0]
        assert float(R[:, 2] @ (d / np.linalg.norm(d))) < -0.95                  # z_ee ≈ −travel: along the seam


# --------------------------------------------------------------------------------------
# 자세 정책
# --------------------------------------------------------------------------------------
def test_reference_orientation_policy_is_removed():
    """사용자 2026-10-12: 'reference' 자세 정책 (KIRO 가 준 pass_<k>.yaml 의 자세) 은 폐기된 KIRO 로그 프레임에 기대므로 없앴다 — 그 설정 키는
    거부되고 (이유를 말한다), 쓰는 자세는 늘 우리 계획 토치 자세."""
    for key, val in (("orientation_policy", "reference"), ("orientation_policy", "planned"), ("reference_dir", str(SAMPLES))):
        with pytest.raises(ValueError, match=r"removed \(user 2026-10-12\) - the 'reference' orientation"):
            _cfg(dict(CFG, **{key: val}))
    assert not hasattr(KP, "load_references") and not hasattr(KP, "POLICIES")


# --------------------------------------------------------------------------------------
# 공유용 정보 파일 (KIRO 로그 키 이름; 명령 아님)
# --------------------------------------------------------------------------------------
def test_info_file_lists_the_assumed_kiro_settings_and_the_derived_layer(plan, tmp_path):
    _c, _wt, ko = _write(plan, tmp_path)
    proc = plan.process
    for wp in plan.passes:
        d = yaml.safe_load((tmp_path / "kiro" / f"pass_{wp.k}_info.yaml").read_text(encoding="utf-8"))
        assert list(d) == ["schema", "pass", "pass_file", "pass_file_location", "purpose", "assumed_kiro_settings", "derived"]
        assert d["pass_file_location"].startswith(f"<dirname(data_file_path)>/kiro/pass_{wp.k}.yaml")
        assert "not commands" in d["purpose"] and d["pass_file"] == f"pass_{wp.k}.yaml" and d["schema"] == "hongik.kiro_pass_info/1"
        cm, ex, dv = d["assumed_kiro_settings"]["commanded"], d["assumed_kiro_settings"]["execution_conditions"], d["derived"]
        assert set(cm) == KIRO_COMMANDED and set(ex) == KIRO_EXECUTION
        i = ("root", "fill", "cap").index(wp.role)
        assert cm == {"current_a": int(proc.current_a[i]), "voltage": proc.voltage_v[i], "material": "FE-SOLID", "diameter_mm": 1.2,
                      "mode": "LSM", "gas": "CO2 100%"} and isinstance(cm["current_a"], int)
        assert ex["weld_tcp_speed_mm_s"] == pytest.approx(proc.speed_mps[i] * 1e3)                    # KIRO 설정 그대로 (5 / 7 / 6)
        assert (ex["weld_lead_in_mm"], ex["weld_lead_out_mm"], ex["weld_arc_off_delay_ms"]) == (
            proc.lead_in_mm[i], proc.lead_out_mm[i], proc.arc_off_delay_ms[i])
        L = dv["straight_line"]["length_mm"]
        if wp.kind == "cap":
            assert ex["weld_weave_enabled"] is True and ex["weld_weave_pattern"] == "sine" and ex["weld_weave_axis"] == "tool_y"
            assert ex["weld_weave_amplitude_mm"] == 4.0 and ex["weld_weave_cycles"] == 16 and ex["weld_weave_left_dwell_s"] == 0.2
            assert ex["weld_lead_in_mm"] == 0.0 and ex["weld_arc_off_delay_ms"] == 0.0
            assert dv["weave"]["cycles_at_the_set_pitch"] == math.ceil(L / 7.8) and dv["weave"]["pitch_with_the_cycles_mm"] == pytest.approx(L / 16)
            assert 0.0 < dv["weave"]["lateral_amplitude_mm"] <= 4.0 and dv["reinforcement_mm"]["median"] > 0.0
            assert dv["comparison_kiro_cap_vs_pass3"]["kiro_mm"] == {"lateral": 6.42, "up": 3.89}
        else:
            assert ex["weld_weave_enabled"] is False and ex["weld_weave_amplitude_mm"] == 0.0 and ex["weld_weave_cycles"] == 0
            assert ex["weld_lead_in_mm"] == 1.0 and ex["weld_lead_out_mm"] == 6.0 and ex["weld_arc_off_delay_ms"] == 500.0
        # 유도: 역할·층 단면적 (용착·필요), 층 높이, 겨냥 오프셋, 잰 심이 직선에서 벗어난 정도 (합성 = 곧은 심 → ~0), 정보용 속도·전류
        assert (dv["role"], dv["layer"], dv["bead"]) == (wp.role, wp.layer, wp.bead)
        assert dv["area_mm2"]["required_seam_mean"] == pytest.approx(wp.area_mean_mm2)
        assert dv["area_mm2"]["deposited_along_seam"]["median"] == pytest.approx(float(np.median(wp.area_mm2)))
        dev = dv["straight_line"]["measured_seam_deviation"]
        assert dev["n_sections"] > 50 and abs(dev["max_lateral_mm"]) < 0.05 and abs(dev["max_vertical_mm"]) < 0.5, dev
        sp = dv["speed_mm_s"]
        assert sp["assumed_setting"] == ex["weld_tcp_speed_mm_s"] and sp["required_along_seam"]["min"] <= sp["required_along_seam"]["max"]
        assert sp["current_for_required_mean_at_the_setting_speed_a"] == pytest.approx(
            proc.current_on_line(wp.area_mean_mm2 * proc.speed_mps[i] * 1e3 / (0.95 * math.pi * 0.36) * 0.06), rel=1e-9)
        assert set(dv["aim_offset_mm"]) == {"frame", "rule", "start", "goal", "x_along_seam", "z_along_seam"}
        assert dv["frames"]["stick_out_mm"] == 20.0 and dv["orientation"]["torch_roll_deg"] == 0.0
    assert "info_files" in ko and "not commands" in ko["info_files"]


# --------------------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------------------
def test_config_validation_and_world_transform_resolution():
    assert KP.kiro_output_config({}) is None
    assert KP.kiro_output_config({"kiro_output": dict(CFG, enabled=False), "weld_torch": TORCH}) is None
    c = KP.kiro_output_config({"kiro_output": CFG, "weld_torch": TORCH})
    assert c.world_frame == "World" and c.tcp_frame == "right_manipulator_ee_point"
    with pytest.raises(ValueError, match="needs the weld_torch section"):            # the poses' TCP is the weld torch model's
        KP.kiro_output_config({"kiro_output": CFG})
    for key, val in (("tcp_frame", "right_manipulator_ee_point"), ("ee_from_tip", {"xyz_m": [0, 0, 0]})):
        with pytest.raises(ValueError, match="moved to the weld_torch section"):
            _cfg(dict(CFG, **{key: val}))
    with pytest.raises(ValueError, match="world_frame is required"):
        _cfg(dict(CFG, world_frame=""))
    with pytest.raises(ValueError, match="unknown keys"):
        _cfg(dict(CFG, world_from_link0={}))
    with pytest.raises(ValueError, match="unit quaternion"):
        _cfg(dict(CFG, world_from_base=dict(CFG["world_from_base"], quat_xyzw=[0, 0, 0.5, 0.5])))
    with pytest.raises(ValueError, match="runtime MoveIt planning frame"):          # no base frame name in the config any more
        _cfg(dict(CFG, world_from_base=dict(CFG["world_from_base"], base_frame="robot_base_link")))
    bad = json.loads(json.dumps(TORCH))
    bad["tip_in_tcp"]["axes"]["y"] = [0.0, -1.0, 0.0]                               # 왼손 틀
    with pytest.raises(ValueError, match="not a rotation"):
        _cfg(torch=bad)
    # base = 실행 때의 계획 프레임: World 면 항등, 아니면 설정의 고정값 (이름과 무관) — TF 는 world_from_tf true 일 때만 덮어쓴다
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    wt, notes = KP.resolve_world_from_base(c, "World", lambda a, b: T)
    assert np.array_equal(wt.R, np.eye(3)) and np.array_equal(wt.t, np.zeros(3)) and "planning frame is World" in wt.source
    wt, notes = KP.resolve_world_from_base(c, "link0", lambda a, b: T)
    assert wt.source == "FK fit (test)" and np.allclose(wt.t, CFG["world_from_base"]["xyz_m"]) and wt.base_frame == "link0"
    assert math.degrees(Rotation.from_matrix(wt.R).magnitude()) == pytest.approx(0.0161, abs=1e-4)          # the fitted rotation
    c_tf = _cfg(dict(CFG, world_from_tf=True))
    wt, notes = KP.resolve_world_from_base(c_tf, "link0", lambda a, b: T)
    assert wt.source == "TF World <- link0 at output time" and np.allclose(wt.t, [1, 2, 3]) and notes == []

    def no_tf(a, b):
        raise LookupError(f'"{a}" passed to lookupTransform argument target_frame does not exist')
    wt, notes = KP.resolve_world_from_base(c_tf, "link0", no_tf)
    assert wt.source == "FK fit (test)" and "TF has no World <- link0" in notes[0]
    wt, notes = KP.resolve_world_from_base(c, "robot_base_link", no_tf)
    assert wt.source == "FK fit (test)" and wt.base_frame == "robot_base_link"
    assert KP.resolve_world_from_base(_cfg({k_: v for k_, v in CFG.items() if k_ != "world_from_base"}), "link0")[0] is None
    # compose: World <- planning frame ∘ planning frame <- plan base
    a = KP.RigidTransform(R=Rotation.from_euler("z", 30, degrees=True).as_matrix(), t=np.array([1.0, 0.0, 0.0]))
    b = KP.RigidTransform(R=Rotation.from_euler("x", 10, degrees=True).as_matrix(), t=np.array([0.0, 2.0, 0.0]), base_frame="link0")
    P = np.array([0.3, -0.2, 0.5])
    assert np.allclose(KP.compose(a, b).apply(P), a.apply(b.apply(P))) and KP.compose(a, b).base_frame == "link0"


# --------------------------------------------------------------------------------------
# 오프라인 재생성: 추정·사전 JSON 왕복 → 같은 계획, CLI
# --------------------------------------------------------------------------------------
def test_estimate_and_prior_json_round_trip_give_the_same_plan():
    pr = synth_prior()
    est = synth_estimate(pr)
    est2 = SeamEstimate.from_dict(json.loads(json.dumps(est.to_dict())))
    pr2 = SeamPrior.from_dict(json.loads(json.dumps(pr.to_dict())))
    assert pr2.spec == pr.spec and pr2.tol == pr.tol and pr2.precision == pr.precision and pr2.k == pr.k
    a = weld_plan_from_estimate(est, pr.spec, WeldProcess(), k=pr.k)
    b = weld_plan_from_estimate(est2, pr2.spec, WeldProcess(), k=pr2.k)
    for p, q in zip(a.passes, b.passes):
        np.testing.assert_allclose(p.aim_base_m, q.aim_base_m, rtol=0, atol=1e-12)
        np.testing.assert_allclose(p.dense_pos_m, q.dense_pos_m, rtol=0, atol=1e-12)
        assert p.speed_const_mps == pytest.approx(q.speed_const_mps, rel=1e-12)
    assert a.layer_ends[1]["S"]["s_m"] == pytest.approx(b.layer_ends[1]["S"]["s_m"])


def test_offline_cli_rebuilds_the_plan_and_the_kiro_files(tmp_path, capsys):
    pr = synth_prior()
    est = synth_estimate(pr)
    run = tmp_path / "run"
    (run / "verification" / "estimates").mkdir(parents=True)
    (run / "verification" / "estimates" / "est_2.json").write_text(json.dumps(est.to_dict()), encoding="utf-8")
    (run / "verification" / "estimates" / "est_10.json").write_text(json.dumps(est.to_dict()), encoding="utf-8")
    (run / "verification" / "run_log.json").write_text(json.dumps({"prior": pr.to_dict(), "job_rough": {
        "base_frame": "link0", "gravity_axis_base": [0.0, 0.0, -1.0]}}), encoding="utf-8")
    cfg = tmp_path / "platform.yaml"
    cfg.write_text(yaml.safe_dump({"/**": {"ros__parameters": {"kiro_output": CFG, "weld_torch": TORCH}}}), encoding="utf-8")
    assert KP.main([str(run), "--platform-config", str(cfg), "--out", str(run)]) == 2
    out = tmp_path / "regen"
    # the stick-out's reference (2026-10-13): the torch's nozzle end - the run's record (as the run derived it), else --urdf
    assert KP.main([str(run), "--platform-config", str(cfg), "--out", str(out)]) == 2
    assert "weld torch nozzle end: not known: the run's weld_plan.json has no weld_torch.nozzle_end_mm" in capsys.readouterr().err
    (run / "weld_plan.json").write_text(json.dumps({"weld_torch": {"nozzle_end_mm": NOZZLE_TEST_MM, "nozzle_end_source": "derived (test)"}}),
                                        encoding="utf-8")
    assert KP.main([str(run), "--platform-config", str(cfg), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "estimate est_10.json" in text and "weld plan: 4 passes" in text
    assert {"sections.json", "sections.npz", "weld_plan.json", "weld_plan.npz", "kiro"} <= {p.name for p in out.iterdir()}
    assert sorted(p.name for p in (out / "kiro").iterdir())[:2] == ["README.md", "pass_1.yaml"]
    assert not (run / "kiro").exists()                                           # 실행 폴더는 그대로
    # weld_plan.* 도 용접 토치 모형 (weld_torch) 의 TCP 자세를 지닌다: 우리 토치 = 항등 축 (TCP z = 노즐 축 = 계획 토치 축, x = 진행),
    # TCP = 끝 (와이어 끝) 에서 토치 축으로 20 mm 뒤
    wpj = json.loads((out / "weld_plan.json").read_text(encoding="utf-8"))
    assert wpj["weld_torch"]["tcp_frame"] == "right_manipulator_ee_point" and "p<k>_tcp_R" in wpj["tcp"]
    # the TCP roll is the NECK_RULE's for the WPS travel angles (2026-10-14: no silent roll 0 when D14 does not run - KIRO's field
    # offline plan had the torch neck bent against the drag): KIRO's drag angles -> roll 180 (TCP x = -travel)
    roll = float(wpj["torch_roll"]["deg"])
    assert "NECK_RULE" in wpj["torch_roll"]["reason"]
    assert all(float(p_["travel_nominal_deg"]) < 0 for p_ in wpj["passes"]) and roll == 180.0
    Rz = np.diag([-1.0, -1.0, 1.0])
    with np.load(out / "weld_plan.npz") as z:
        assert str(z["tcp_frame"]) == "right_manipulator_ee_point"
        for k in range(1, len(wpj["passes"]) + 1):
            R_tip, R_tcp = z[f"p{k}_R"], z[f"p{k}_tcp_R"]
            assert np.allclose(R_tcp, R_tip @ Rz, atol=1e-12)
            assert np.allclose(z[f"p{k}_tcp_pos_m"], z[f"p{k}_pos_m"] - TIP_MM * 1e-3 * R_tip[:, :, 2], atol=1e-15)
    assert wpj["weld_torch"]["nozzle_end_mm"] == NOZZLE_TEST_MM and wpj["weld_torch"]["nozzle_end_source"].startswith("the run's record")
