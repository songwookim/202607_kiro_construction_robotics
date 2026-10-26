"""hongik_scan_ros 순수 모듈 단위 시험 (ROS 불필요): 플랫폼 설정 검증, 직선 IK 추종·시간 매개화, 캘리브 ↔ TF,
포트 선택 (scan_l0.ports.select_adapter), 설정 파일 키 집합 (hongik_kiro ↔ 가상 팔 시험 설정 ↔ 연구실 값 시험 설정)."""
import copy
import json
import math
import os
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STACK = os.path.dirname(PKG)
for p in (PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"), os.path.join(STACK, "vc3dz_driver")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hongik_scan_ros.calib_tf import calib_consistency, calib_from_tf, rpy_static_args, static_tf_from_calib  # noqa: E402
from hongik_scan_ros.platform_config import REQUIRED_LIMITS, SECTION_KEYS, MoveItJtcConfig  # noqa: E402
from scan_core.trajectory import (TrapezoidProfile, cartesian_steps, interpolate_pose, on_path_distance,  # noqa: E402
                                        retime_straight_path, track_straight_line)
from scan_core.fk_urdf import rb20_1900es_u_chain  # noqa: E402
from scan_l0 import config as C  # noqa: E402
from scan_l0.ports import ADAPTERS, PlatformConfigError, select_adapter  # noqa: E402

CFG_DIR = os.path.join(PKG, "config")
KIRO = os.path.join(CFG_DIR, "hongik_kiro.yaml")
FAKE = os.path.join(STACK, "scan_l0", "tests", "data", "hongik_rb20_fake_moveit.yaml")      # 가상 팔 (시험 자료)
LAB = os.path.join(STACK, "scan_l0", "tests", "data", "hongik_lab_rb20_values.yaml")        # 연구실 플랫폼 값 (시험 자료)
CALIB = os.path.join(STACK, "eoat_description", "calib", "eoat_lab_01.yaml")
COMMON_SECTIONS = ("scan_defaults", "specimen", "auto_teach", "sensor", "time_model", "kiro_output")


def _params(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["/**"]["ros__parameters"]


def _flat(d, prefix=""):
    out = set()
    for k, v in d.items():
        key = prefix + str(k)
        out |= _flat(v, key + ".") if isinstance(v, dict) else {key}
    return out


# --------------------------------------------------------------------------------------
# 플랫폼 설정
# --------------------------------------------------------------------------------------
def test_kiro_config_selects_adapter_b_with_kiro_names():
    p = _params(KIRO)
    assert select_adapter(p) == "moveit_jtc" and C.load_settings(KIRO).adapter == "moveit_jtc"
    mj = MoveItJtcConfig.from_params(p)
    # KIRO's robot model with our EOAT (construct_robot_0528_hongik_v2): planning frame World, our frames hongik_*, the static optical TF
    assert (mj.base_frame, mj.flange_frame, mj.eoat_frame, mj.laser_frame, mj.optical_frame) == (
        "World", "hongik_flange", "hongik_eoat_base_link", "hongik_laser_link", "hongik_profiler_optical_frame")
    assert mj.gravity_frame == "World" and mj.weld_torch.tcp_frame == "right_manipulator_ee_point"
    assert mj.weld_torch.flange_frame == mj.flange_frame and mj.weld_torch.on_this_arm is True
    assert mj.move_group_name == "right_manipulator"
    assert mj.jtc_action == "/right_manipulator_controller/follow_joint_trajectory"
    assert mj.use_sim_time is False and mj.min_fraction >= 0.999                 # the real robot (true only with Isaac Sim /clock)
    assert set(p["moveit_jtc"]) <= set(SECTION_KEYS) and "kill_switch" not in p["safety"]
    assert p["scan_defaults"]["gravity_axis_base"] is None                  # 손으로 적지 않는다 (TF 유도)
    assert mj.gravity_frame and len(mj.gravity_axis) == 3
    for k in REQUIRED_LIMITS:
        assert getattr(mj, k) > 0


def test_lab_config_is_moveit_too():
    p = _params(LAB)
    assert select_adapter(p) == "moveit_jtc" and C.load_settings(LAB).adapter == "moveit_jtc"   # 연구실도 MoveIt 만 (제어박스 어댑터 제거)


def test_legacy_or_unknown_adapter_keys_are_refused():
    with pytest.raises(PlatformConfigError, match="pose_source must be one of"):
        select_adapter({"pose_source": "cb5001_fk", "motion_backend": "moveit_jtc"})     # 예전 제어박스 자세원
    with pytest.raises(PlatformConfigError, match="pose_source must be one of"):
        select_adapter({})
    with pytest.raises(PlatformConfigError, match="motion_backend must be one of"):
        select_adapter({"pose_source": "ros_tf", "motion_backend": "cb_move_l"})        # 예전 제어박스 move_l
    with pytest.raises(PlatformConfigError, match="pose_source must be one of"):
        select_adapter({"pose_source": "tf", "motion_backend": "moveit_jtc"})
    with pytest.raises(PlatformConfigError, match="motion_backend must be one of"):
        select_adapter({"pose_source": "ros_tf", "motion_backend": "moveit_lin"})
    assert set(ADAPTERS) == {"moveit_jtc"}


@pytest.mark.parametrize("key", REQUIRED_LIMITS)
def test_moveit_backend_refuses_to_start_without_declared_limits(key):
    p = copy.deepcopy(_params(KIRO))
    del p["moveit_jtc"]["limits"][key]
    with pytest.raises(PlatformConfigError, match=f"{key} is required"):
        MoveItJtcConfig.from_params(p)


def test_moveit_config_refuses_partial_lines_and_missing_names():
    p = copy.deepcopy(_params(KIRO))
    p["moveit_jtc"]["min_fraction"] = 0.95
    with pytest.raises(PlatformConfigError, match="never execute a partial"):
        MoveItJtcConfig.from_params(p)
    p = copy.deepcopy(_params(KIRO))
    p["moveit_jtc"]["move_group_name"] = ""
    with pytest.raises(PlatformConfigError, match="move_group_name is required"):
        MoveItJtcConfig.from_params(p)
    p = copy.deepcopy(_params(KIRO))
    p["gravity"]["frame"] = None
    with pytest.raises(PlatformConfigError, match="gravity.frame is required"):
        MoveItJtcConfig.from_params(p)
    p = copy.deepcopy(_params(KIRO))
    del p["use_sim_time"]
    with pytest.raises(PlatformConfigError, match="use_sim_time is required"):
        MoveItJtcConfig.from_params(p)


def test_config_key_sets_kiro_fake_and_lab_common_sections():
    kiro, fake, lab = _params(KIRO), _params(FAKE), _params(LAB)
    assert _flat(kiro) == _flat(fake)                                       # 시험 플랫폼 = KIRO 와 같은 키
    for sec in COMMON_SECTIONS:
        assert _flat(kiro[sec]) == _flat(lab[sec]), sec                    # 플랫폼 무관 섹션은 연구실과 같은 키
    # 같은 알고리즘 값 — 연구실의 1G 손목 형태 제한만 다르다 (사용자 결정 2026-10-09: 연구실 캘리브 후보를 맞춘 손목2 음수 형태; KIRO 는
    # 제한 없음 그대로)
    assert lab["auto_teach"]["wrist2_branch_1g"] == "neg" and kiro["auto_teach"]["wrist2_branch_1g"] == "any"
    assert dict(kiro["auto_teach"], wrist2_branch_1g="neg") == lab["auto_teach"]
    assert MoveItJtcConfig.from_params(fake).move_group_name == "mainpulation"


# --------------------------------------------------------------------------------------
# 직선 IK 추종 · 시간 매개화 (RB20 URDF 체인을 FK 모델로)
# --------------------------------------------------------------------------------------
def _rb20():
    ch = rb20_1900es_u_chain()
    q0 = np.array([0.122, -0.231, 2.365, 0.0, 1.008, -1.449])
    return ch, q0


def test_track_straight_line_follows_the_line_exactly_and_matches_moveit_steps():
    ch, q0 = _rb20()
    T0 = ch.fk(q0)
    T1 = T0.copy()
    T1[:3, 3] += [0.20, 0.03, -0.02]
    from scipy.spatial.transform import Rotation
    T1[:3, :3] = Rotation.from_rotvec([0.0, 0.0, math.radians(3.0)]).as_matrix() @ T0[:3, :3]
    Q, fr = track_straight_line(ch.fk, q0, T0, T1, 0.002)
    assert Q.shape[0] == cartesian_steps(T0, T1, 0.002) + 1 == int(math.floor(np.linalg.norm([0.2, 0.03, -0.02]) / 0.002)) + 2
    for q, f in zip(Q, fr):
        T = ch.fk(q)
        Tm = interpolate_pose(T0, T1, f)
        assert np.linalg.norm(T[:3, 3] - Tm[:3, 3]) < 1e-5
    assert np.linalg.norm(ch.fk(Q[-1])[:3, 3] - T1[:3, 3]) < 1e-6
    # 같은 경로 위의 점은 거리 ~0, 다른 분기 점은 멀다
    mid = 0.5 * (Q[10] + Q[11])
    assert on_path_distance(Q, mid) < 1e-9 and on_path_distance(Q, Q[5] + 0.3) > 0.1


def test_fast_fk_analytic_jacobian_and_rotation_vector_match_the_references():
    """fast_fk (FK 한 번에 해석적 야코비안) = 체인 FK·수치 미분 야코비안, 닫힌 식 회전 벡터 = scipy (180° 근처는 scipy 그대로) — 박스
    캘리브 시점 계획 (관절 시뮬레이션) 속도 개선 (2026-10-06) 이 직선 IK 결과를 바꾸지 않는다."""
    from scipy.spatial.transform import Rotation
    from scan_core.trajectory import _vee, fast_fk, rotation_vector
    ch, q0 = _rb20()
    fk = fast_fk(ch)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-3.0, 3.0, 6)
        T = fk(q)
        np.testing.assert_allclose(T, ch.fk(q), atol=1e-12)
        Jn = np.zeros((6, 6))
        for k in range(6):
            dq = q.copy()
            dq[k] += 1e-7
            Tk = fk(dq)
            Jn[:3, k] = (Tk[:3, 3] - T[:3, 3]) / 1e-7
            Jn[3:, k] = _vee(Tk[:3, :3] @ T[:3, :3].T) / 1e-7
        np.testing.assert_allclose(fk.jacobian(q), Jn, atol=1e-6)
    for ang in (0.0, 1e-9, 1e-4, 0.5, 2.0, math.radians(178.5), math.radians(179.5)):
        for _ in range(20):
            ax = rng.normal(size=3)
            rv = ax / np.linalg.norm(ax) * ang
            np.testing.assert_allclose(rotation_vector(Rotation.from_rotvec(rv).as_matrix()), rv, atol=1e-12)
    # 같은 직선 추종: 해석적 (fast_fk) = 수치 미분 (체인 FK)
    T0 = ch.fk(q0)
    T1 = T0.copy()
    T1[:3, 3] += [0.15, -0.05, 0.04]
    T1[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.4]).as_matrix() @ T0[:3, :3]
    Qa, _ = track_straight_line(fk, q0, T0, T1, 0.002)
    Qn, _ = track_straight_line(ch.fk, q0, T0, T1, 0.002)
    assert Qa.shape == Qn.shape and np.abs(Qa - Qn).max() < 1e-6


def test_track_straight_line_refuses_unreachable_and_joint_limits():
    ch, q0 = _rb20()
    T0 = ch.fk(q0)
    far = T0.copy()
    far[:3, 3] += [3.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="did not converge|jump"):
        track_straight_line(ch.fk, q0, T0, far, 0.002)
    T1 = T0.copy()
    T1[:3, 3] += [0.05, 0.0, 0.0]
    with pytest.raises(ValueError, match="URDF limit"):
        track_straight_line(ch.fk, q0, T0, T1, 0.002, lower=q0 - 1e-3, upper=q0 + 1e-3)


def test_retime_trapezoid_constant_cruise_and_joint_limit_stretch():
    prof = TrapezoidProfile(S=0.3, v=0.03, a=0.2)
    assert not prof.triangular and abs(prof.duration - (0.3 / 0.03 + 0.03 / 0.2)) < 1e-9
    assert prof.time_at(0.0) == 0.0 and abs(prof.time_at(0.3) - prof.duration) < 1e-9
    ch, q0 = _rb20()
    T0 = ch.fk(q0)
    T1 = T0.copy()
    T1[0, 3] += 0.3
    Q, _ = track_straight_line(ch.fk, q0, T0, T1, 0.002)
    rt = retime_straight_path(Q, 0.3, 0.0, 0.03, 0.2, max_joint_velocity=1.0, max_joint_accel=2.0)
    assert rt.time_scale == 1.0 and abs(rt.v_cruise - 0.03) < 1e-12 and np.all(np.diff(rt.times) > 0)
    # 등속 구간: 경로 매개변수 속도 = v (점 간격 2 mm / dt)
    ds = np.diff(rt.s)
    dt = np.diff(rt.times)
    mid = slice(len(dt) // 4, 3 * len(dt) // 4)
    assert np.allclose(ds[mid] / dt[mid], 0.03, rtol=1e-6)
    assert np.all(rt.velocities[0] == 0) and np.all(rt.velocities[-1] == 0)
    slow = retime_straight_path(Q, 0.3, 0.0, 0.03, 0.2, max_joint_velocity=0.005, max_joint_accel=2.0)
    assert slow.time_scale > 1.0 and slow.max_joint_velocity <= 0.005 + 1e-9 and slow.notes
    with pytest.raises(ValueError, match="joint-space jump"):
        retime_straight_path([[0.0] * 6, [0.5] + [0.0] * 5], 0.01, 0.0, 0.03, 0.2, max_joint_step_rad=0.1)


# --------------------------------------------------------------------------------------
# 캘리브
# --------------------------------------------------------------------------------------
def test_static_tf_from_calib_and_calib_from_tf_roundtrip():
    with open(CALIB, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    xyz, rpy = static_tf_from_calib(doc, "laser")
    assert np.allclose(xyz, doc["xyz"]) and np.allclose(rpy, doc["rpy"])
    file_calib = C.load_calib(CALIB)
    xyz_f, rpy_f = static_tf_from_calib(doc, "flange")
    from scan_core.frames import transform_from_xyz_rpy
    assert np.allclose(transform_from_xyz_rpy(xyz_f, rpy_f), file_calib.T_flange_opt, atol=1e-9)
    tf_calib = calib_from_tf(file_calib, file_calib.T_flange_opt, file_calib.T_flange_eoat)
    d = calib_consistency(file_calib, tf_calib)
    assert d["dp_mm"] < 1e-9 and d["dr_deg"] < 1e-6 and abs(d["tf_eoat_yaw_deg"] - math.degrees(file_calib.eoat_yaw_rad)) < 1e-9
    assert tf_calib.dt_s == file_calib.dt_s and tf_calib.x_sign == file_calib.x_sign and tf_calib.source.endswith("+tf")
    # the TF gives the two transforms (and eoat_yaw) only: every other field of the file calib reaches the run - the 1G 'g1:' block
    # (time offset, sigmas, edge response, noise priors), the tool body and any field Calib gains later (2026-10-12 morning lab runs:
    # a field-by-field copy dropped noise_priors / edge_response, so the root was never established and the end model estimated the
    # sensor response online)
    import dataclasses
    from scan_core.types import Calib
    assert file_calib.noise_priors and file_calib.edge_response and file_calib.he_sigma_m is not None
    file_calib.body_points_eoat, file_calib.body_source = np.eye(3), "test body"
    file_calib.residual_rms_m, file_calib.sensor_param_hash = 1.0e-4, "test hash"            # non-default values so a loss shows
    T_fo = file_calib.T_flange_opt.copy()
    T_fo[:3, 3] += 1e-3                                                       # a platform TF that differs from the file
    tf2 = calib_from_tf(file_calib, T_fo, file_calib.T_flange_eoat, note="x")
    np.testing.assert_allclose(tf2.T_flange_opt, T_fo)
    for f in dataclasses.fields(Calib):
        if f.name in ("T_flange_opt", "T_flange_eoat", "eoat_yaw_rad", "source"):
            continue
        a, b = getattr(file_calib, f.name), getattr(tf2, f.name)
        assert a is b or (np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b), f.name
    assert tf2.to_dict()["g1"] == file_calib.to_dict()["g1"]
    args = rpy_static_args(xyz, rpy, "laser_link", "hongik_profiler_optical_frame")
    assert args[-4:] == ["--frame-id", "laser_link", "--child-frame-id", "hongik_profiler_optical_frame"]
    with pytest.raises(ValueError):
        static_tf_from_calib({**doc, "parent_frame": "eoat_base_link"}, "laser")


# --------------------------------------------------------------------------------------
# /joint_states 시각축: 수신이 몰려도 (부하) 발행 간격을 복원한다 (2026-10-01 전체 시험 부하에서 측정 속도 2 배 오류의 원인)
# --------------------------------------------------------------------------------------
def test_stamp_to_monotonic_recovers_publish_cadence_under_bursty_delivery():
    import numpy as np
    from hongik_scan_ros.ros_util import StampToMonotonic
    rng = np.random.default_rng(0)
    wall0, mono0 = 1_790_000_000_000_000_000, 5_000_000_000
    period = 10_000_000                                   # 100 Hz
    m = StampToMonotonic(window_s=2.0)
    out, rx = [], []
    pending_until = 0
    for i in range(500):
        stamp = wall0 + i * period
        lat = 300_000 + int(rng.integers(0, 200_000))     # 0.3–0.5 ms 기본 지연
        t_pub_mono = mono0 + i * period
        if i % 7 == 3:                                    # 7 개마다 25 ms 정체 → 다음 몇 개가 몰려서 도착
            pending_until = t_pub_mono + 25_000_000
        t_rx = max(t_pub_mono + lat, pending_until)
        rx.append(t_rx)
        out.append(m.map(stamp, t_rx))
    out, rx = np.array(out), np.array(rx)
    d_rx = np.diff(rx) * 1e-6
    d_out = np.diff(out[50:]) * 1e-6                      # 창이 차고 나면
    assert d_rx.max() > 20 and d_rx.min() < 1            # 수신 간격은 몰림 (0 ~ 25 ms)
    assert abs(d_out - 10.0).max() < 0.25                # 매핑 간격 = 발행 간격 (±0.25 ms)
    assert np.all(out <= rx) and np.all(np.diff(out) > 0) and m.fallbacks == 0


def test_stamp_to_monotonic_falls_back_on_missing_or_stale_stamps():
    from hongik_scan_ros.ros_util import StampToMonotonic
    m = StampToMonotonic(window_s=2.0)
    assert m.map(0, 1_000) == 1_000 and m.fallbacks == 1                       # 스탬프 없음
    t = m.map(10_000_000_000, 10_000_500_000)                                   # 정상
    assert t == 10_000_500_000
    t2 = m.map(10_010_000_000, 11_000_000_000)          # 혼자 0.99 s 늦게 도착 → 창 최소 지연 유지 → 매핑 = 발행 + 0.5 ms
    assert t2 == 10_010_500_000 and m.fallbacks == 1
    t3 = m.map(10_005_000_000, 11_100_000_000)          # 벽시계가 뒤로 조정됨 (스탬프 역행) → 매핑 역행 → 수신 시각으로 대체
    assert t3 == 11_100_000_000 and m.fallbacks == 2


# --------------------------------------------------------------------------------------
# 클라이언트 · 서버 도우미
# --------------------------------------------------------------------------------------
def test_auto_teach_client_goal_and_result_lines(tmp_path):
    from types import SimpleNamespace
    from hongik_scan_ros.auto_teach_client import goal_from_job, result_lines
    from hongik_scan_ros.auto_teach_server import safe_out_name
    tpl = os.path.join(STACK, "scan_l0", "scan_l0", "jobs", "groove_10T_1G.yaml")
    with open(tpl, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc["job"].update(start=[1.07, -0.06, -0.15], end=[1.41, -0.03, -0.15], position="1G")
    doc["job"].pop("orientation", None)
    jp = tmp_path / "rough.yaml"
    jp.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    g = goal_from_job(jp, out_name="at09", dry_run=True)
    assert (g.rough_start.x, g.rough_end.y, g.position, g.confirm_mode, g.out_name, g.dry_run, g.frame_id) == \
        (1.07, -0.03, "1G", "none", "at09", True, "")
    assert g.job_file == str(jp.resolve())
    doc["job"].pop("position")
    jp2 = tmp_path / "nopos.yaml"
    jp2.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="position"):
        goal_from_job(jp2)
    assert goal_from_job(jp2, position="1G").position == "1G"
    with pytest.raises(ValueError, match="this package runs 1G jobs"):
        goal_from_job(jp2, position="3G")
    assert [safe_out_name(n) for n in ("at09", "../x", "a/b", "", "..", "ok_1.2")] == ["at09", "_x", "a_b", "", "", "ok_1.2"]
    out = tmp_path / "res"
    out.mkdir()
    rep = {"pass": 2, "bin_mm": 20.0, "measurement_noise_mm": {"root_gap_mm": 0.15, "depth_mm": 0.04, "top_opening_mm": 0.14},
           "summary": {"root_gap_mm": {"median": 5.0, "design": 5.0, "tol": 0.2, "within": True}}, "bins": []}
    (out / "specimen_report.json").write_text(json.dumps(rep), encoding="utf-8")
    from scan_core.weld_plan import GrooveStation, plan_weld
    from scan_core.specimen import GrooveSpec
    u, n = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    sts = [GrooveStation(s_mm=s_, origin=np.array([1.0 + s_ * 1e-3, 0.0, -0.17]), u=u, n=n, w=np.cross(n, u),
                         spec=GrooveSpec(bevel_side="left")) for s_ in (0.0, 190.0, 380.0)]
    plan_weld(sts).save(out)
    pt = SimpleNamespace(x=1.0, y=0.0, z=-0.17)
    res = SimpleNamespace(status="PASS", reason="pass 2 PASS", out_dir=str(out), refined_job_path=str(out / "job_refined.yaml"),
                          length_mm=380.49, ends_observed=True, n_passes=2, n_moves=14, elapsed_s=230.0, converged="true",
                          seam_start=pt, seam_end=pt)
    lines = result_lines(res)
    assert lines[0] == "RESULT: PASS - pass 2 PASS" and any("length 380.49 mm" in l_ for l_ in lines)
    assert any(l_.startswith("specimen vs design (pass 2") for l_ in lines)
    assert any(l_.startswith("weld plan: 4 passes over 380.0 mm") for l_ in lines) and any(" cap " in l_ for l_ in lines)
    assert not any("seam start" in l_ for l_ in lines)


def test_measure_speed_override_from_the_client_through_the_goal_to_the_server(tmp_path):
    """User 2026-10-12: the measurement speed per run (the passes and the precise dwell corrections), like --max-passes: the client's
    --measure-speed takes a speed in mm/s (default: not given; 2026-10-13: any speed - the 1G core checks it against the
    calibration's noise-prior speed range, test_job_overrides), goal_from_job carries it in OUR AutoTeach goal as
    measure_speed_mps [m/s] (0 = the job file's speed_mps), and the server reads it back as the run's speed (None = the job's) or
    refuses a negative / non-finite value before anything is sent. A goal without the field (the KIRO rough goal) reads as None."""
    from types import SimpleNamespace
    from hongik_scan_ros import auto_teach_client as cl
    from hongik_scan_ros.auto_teach_server import goal_measure_speed
    tpl = os.path.join(STACK, "scan_l0", "scan_l0", "jobs", "groove_10T_1G.yaml")
    with open(tpl, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc["job"].update(start=[1.07, -0.06, -0.15], end=[1.41, -0.03, -0.15], position="1G")
    doc["job"].pop("orientation", None)
    jp = tmp_path / "rough.yaml"
    jp.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    # CLI: a speed in mm/s (the calibrated range is the server's check), not given = None
    ap = cl.build_parser()
    base = ["--job", str(jp)]
    assert ap.parse_args(base).measure_speed is None
    assert [ap.parse_args(base + ["--measure-speed", v]).measure_speed for v in ("15", "30", "22.5", "60")] == [15.0, 30.0, 22.5, 60.0]
    with pytest.raises(SystemExit):
        ap.parse_args(base + ["--measure-speed", "fast"])
    for bad in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError, match="--measure-speed must be a speed > 0"):
            cl.measure_speed_mps(bad)
    # mm/s -> the goal's m/s (0 = not given)
    assert cl.measure_speed_mps(None) == 0.0 and cl.measure_speed_mps(15) == pytest.approx(0.015) \
        and cl.measure_speed_mps(30) == pytest.approx(0.030) and cl.measure_speed_mps(22.5) == pytest.approx(0.0225)
    g0 = cl.goal_from_job(jp)
    assert g0.measure_speed_mps == 0.0 and goal_measure_speed(g0) is None
    assert cl.measure_speed_text(g0.measure_speed_mps) == "measurement speed: the job file's speed_mps"
    g30 = cl.goal_from_job(jp, max_passes=3, measure_speed_mps=cl.measure_speed_mps(30))
    assert g30.measure_speed_mps == pytest.approx(0.030) and g30.max_passes == 3 and goal_measure_speed(g30) == pytest.approx(0.030)
    assert cl.measure_speed_text(g30.measure_speed_mps) == "measurement speed 30 mm/s (run override)"
    g15 = cl.goal_from_job(jp, measure_speed_mps=cl.measure_speed_mps(15))
    assert goal_measure_speed(g15) == pytest.approx(0.015)
    # the code stays general (any speed in m/s); only nonsense is refused, in the client and in the server
    assert goal_measure_speed(SimpleNamespace(measure_speed_mps=0.02)) == pytest.approx(0.02)
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="measure_speed_mps"):
            cl.goal_from_job(jp, measure_speed_mps=bad)
        with pytest.raises(ValueError, match="measure_speed_mps"):
            goal_measure_speed(SimpleNamespace(measure_speed_mps=bad))
    assert goal_measure_speed(SimpleNamespace(position="1G")) is None              # a goal without the field (KIRO rough goal)


def test_measure_speed_override_replaces_the_job_speed_for_the_run_only(tmp_path):
    """The run classes (AutoTeach / Teach1G) take the override as the job's speed_mps of this run (job0 - every speed-dependent
    quantity derives from it); the job file is not changed, and the record names the source (run log, setup line)."""
    from scan_l0.autoteach import AutoTeach
    from scan_l0.session import ScanSession
    tpl = os.path.join(STACK, "scan_l0", "scan_l0", "jobs", "groove_10T_1G.yaml")
    job, doc = C.load_job(tpl)
    st = C.load_settings(LAB)
    calib = C.load_calib(CALIB)
    sess = ScanSession(robot=None, sensor=None, owned=False)
    kw = dict(out_dir=str(tmp_path / "o"), session=sess)
    at0 = AutoTeach(st, job, doc, calib, **kw)
    # the template's measurement speed is 30 mm/s (user 2026-10-12 night); 15 mm/s is now the run override
    assert at0.job0.speed_mps == pytest.approx(0.03) and at0.measure_speed_record() == {
        "speed_mps": pytest.approx(0.03), "job_file_speed_mps": pytest.approx(0.03), "override_mps": None,
        "source": "job file speed_mps"}
    assert at0.measure_speed_text() == "the job's speed_mps"
    at15 = AutoTeach(st, job, doc, calib, measure_speed_mps=0.015, **kw)
    assert at15.job0.speed_mps == pytest.approx(0.015) and job.speed_mps == pytest.approx(0.03)      # the job itself is unchanged
    assert at15.job0.profile_fps == job.profile_fps and np.allclose(at15.job0.start, job.start)
    rec = at15.measure_speed_record()
    assert rec["speed_mps"] == pytest.approx(0.015) and rec["override_mps"] == pytest.approx(0.015) \
        and rec["job_file_speed_mps"] == pytest.approx(0.03) and rec["source"] == "run override (measure_speed_mps)"
    assert at15.measure_speed_text() == "run override measure_speed_mps; the job file's speed_mps 30.0 mm/s"
    for bad in (0.0, -0.03, float("nan")):
        with pytest.raises(ValueError, match="measure_speed_mps"):
            AutoTeach(st, job, doc, calib, measure_speed_mps=bad, **kw)


def test_auto_teach_client_prints_the_1g_core_sections_summary(tmp_path):
    """1G core (G1_CORE_SPEC §9): 결과 폴더에 sections.json 이 있으면 결과 줄에 단면 형상 요약 (판정·양끝·길이·단면 수) 이 들어간다."""
    from types import SimpleNamespace
    from hongik_scan_ros.auto_teach_client import result_lines, sections_lines
    doc = {"schema": "g1core.sections/1", "status": "PASS", "done": {"resolved": False, "max_ratio": 1.6, "unresolved": ["end_E"]},
           "ends": {"S": {"point_m": [0.6, -0.19, 0.04], "sigma_m": 0.00024, "two_sided": True},
                    "E": {"point_m": [0.6, 0.19, 0.04], "sigma_m": 0.00053, "two_sided": True}},
           "length_mm": {"value": 379.66, "sigma": 0.58, "design": 380.0},
           "stations": [{"s_m": 0.0, "gap_mm": {"value": 5.01}, "depth_mm": {"value": 10.02}},
                        {"s_m": 0.005, "gap_mm": {"value": 4.99}, "depth_mm": {"value": 10.0}}],
           "bevel_side": "left", "plate": {"roll_deg": 4.0, "misalignment_deg": 0.1}}
    lines = sections_lines(doc)
    assert lines[0].startswith("sections (g1core.sections/1): status PASS, resolved False, max sigma/precision 1.60, unresolved ['end_E']")
    assert "+- 0.24 mm (two-sided)" in lines[1] and "length 379.66 +- 0.58 mm (design 380.0 mm)" in lines[2]
    assert lines[3].startswith("  2 stations: median gap_mm") and "bevel side left" in lines[4]
    out = tmp_path / "res"
    out.mkdir()
    (out / "sections.json").write_text(json.dumps(doc), encoding="utf-8")
    res = SimpleNamespace(status="PASS", reason="fine-pass budget 4 used", out_dir=str(out), refined_job_path="", length_mm=379.66,
                          ends_observed=True, n_passes=5, n_moves=10, elapsed_s=120.0, converged="false")
    got = result_lines(res)
    assert any(l_.startswith("sections (g1core.sections/1)") for l_ in got) and any("cross sections:" in l_ for l_ in got)
    assert sections_lines({}) == ["sections (?): status -, resolved -, max sigma/precision -"]






def test_build_rough_job_takes_the_geometry_from_the_hit_points_only():
    """범용 시작 (2026-10-07): 1G 파일 + 3G 목표 → 3G 템플릿의 설계·공정 블록, 기하는 착점·자세·중력에서만 — 판 법선 = 로봇 쪽 수평
    (템플릿 축 아님), 판 상단·테이블은 비운다 (예전 '위쪽 점 + 0.10 m' 없음; auto-teach 가 착점·설계 길이·측정으로 정한다)."""
    from types import SimpleNamespace
    from geometry_msgs.msg import Point
    from hongik_scan_ros.auto_teach_server import build_rough_job
    from scan_l0 import config as C

    def goal(pos, S, E, job_file=""):
        return SimpleNamespace(rough_start=Point(x=S[0], y=S[1], z=S[2]), rough_end=Point(x=E[0], y=E[1], z=E[2]), position=pos,
                               job_file=job_file, weld_length_mm=0.0, thickness_mm=0.0, root_gap_mm=0.0, bevel_angle_deg=0.0,
                               groove_kind="", bevel_side="")
    one_g = str(C.JOBS_DIR / "groove_10T_1G.yaml")
    S, E = (0.50, 0.70, 0.33), (0.50, 0.70, 0.65)
    job, doc = build_rough_job(None, one_g, goal("3G", S, E), base_frame="link0", gravity_axis_base=(0.0, 0.0, -1.0))
    toward = -np.array([0.5, 0.7, 0.0]) / np.linalg.norm([0.5, 0.7])
    assert job.position == "3G" and job.orientation == "vertical_3G" and np.allclose(job.surface_normal, toward, atol=1e-9)
    assert job.specimen_top_z_m is None and job.table_z_m is None and any("uses" in n_ for n_ in doc["meta"]["goal_notes"])
    three_g = str(C.JOBS_DIR / "groove_10T_3G.yaml")
    job2, _ = build_rough_job(None, three_g, goal("3G", S, E, job_file=three_g), base_frame="link0")
    assert job2.specimen_top_z_m is None and np.allclose(job2.surface_normal, toward, atol=1e-9)    # 파일 값도 쓰지 않는다
    job4, _ = build_rough_job(None, one_g, goal("1G", (1.0, 0.0, -0.15), (1.38, 0.03, -0.15)), base_frame="link0")
    assert job4.position == "1G" and job4.specimen_top_z_m is None and np.allclose(job4.surface_normal, [0.0, 0.0, 1.0], atol=1e-9)


def test_client_explains_a_rejected_goal():
    """목표 거절 사유 (2026-10-05: 'REFUSED' 만 보여 원인을 몰랐다). 2026-10-10: 정지 래치 없음 — 준비 안 된 항목과 할 일을 보여 준다."""
    from hongik_scan_ros.auto_teach_client import rejection_lines
    st = ("adapter=moveit_jtc running=False goal= phase=idle pass=0 ready=False not_ready='JTC action /jtc/follow_joint_trajectory' "
          "run_stopped=False stop_reason='' awaiting_confirm=False last=")
    lines = rejection_lines(st)
    assert lines[0] == "REFUSED: the platform is not ready: JTC action /jtc/follow_joint_trajectory", lines
    assert "nothing to reset" in lines[1] and not any("reset_estop" in ln for ln in lines)
    ok = st.replace("ready=False", "ready=True").replace("not_ready='JTC action /jtc/follow_joint_trajectory'", "not_ready=''")
    assert "another auto-teach run is active" in rejection_lines(ok.replace("running=False", "running=True"))[0]
    assert "no status answer" in rejection_lines(None)[0]


@pytest.mark.parametrize("where", ["section", "limits"])
def test_a_key_this_package_does_not_read_is_refused(where):
    """moveit_jtc 의 키는 MoveItJtcConfig 가 읽는 것뿐 — 그 밖의 키 (옛 설정·오타) 는 조용히 무시하지 않고 시작 거부."""
    p = copy.deepcopy(_params(KIRO))
    (p["moveit_jtc"] if where == "section" else p["moveit_jtc"]["limits"])["not_a_setting"] = "x"
    key = "moveit_jtc.not_a_setting" if where == "section" else "moveit_jtc.limits.not_a_setting"
    with pytest.raises(PlatformConfigError, match=f"{key}: not a setting of this package"):
        MoveItJtcConfig.from_params(p)


def test_section_keys_are_exactly_the_keys_the_parser_reads():
    """SECTION_KEYS (받는 키 목록) = from_params 가 moveit_jtc 에서 읽는 키 — 새 키를 읽으면서 목록에 빠뜨리면 그 설정이 거부된다."""
    import inspect
    import re
    import hongik_scan_ros.platform_config as PC
    src = inspect.getsource(PC.MoveItJtcConfig.from_params)
    read = set(re.findall(r'\(mj, "(\w+)"', src)) | set(re.findall(r'mj\.get\("(\w+)"\)', src))
    assert read == set(SECTION_KEYS)


def test_duration_parts_never_goes_back_in_time():
    """2026-10-06: 정수 초 바로 아래 시각이 (sec, nanosec) 로 감겨 2.000 s 가 되어 JTC 가 목표를 거부했다 (시각이 뒤로)."""
    from scan_core.trajectory import duration_parts
    assert duration_parts(2.9999999997) == (3, 0)
    assert duration_parts(2.9999999994) == (2, 999999999)
    assert duration_parts(0.0) == (0, 0) and duration_parts(1.25) == (1, 250000000)
    ts = np.sort(np.concatenate([np.arange(0.0, 20.0, 0.0371), 3.0 - np.array([1e-10, 4e-10, 6e-10, 1e-9])]))
    ns = [s * 1_000_000_000 + n for s, n in (duration_parts(t) for t in ts)]
    assert all(0 <= duration_parts(t)[1] < 1_000_000_000 for t in ts) and all(b >= a for a, b in zip(ns, ns[1:]))


# ---- 인터페이스 r3 (2026-10-10): 입력 흐름 (재사용 질문 use / do not use + 시편 입력), 진행 줄, PASS_UNRESOLVED, 영어 문구 -------------
_HANGUL = __import__("re").compile(r"[가-힣]")


def _answers(*vals):
    """입력 주입: 차례대로 답한다 (묻는 문구는 asked 에)."""
    it = iter(vals)
    asked: list = []

    def ask(prompt):
        asked.append(prompt)
        return next(it)
    return ask, asked


def test_client_setup_specimen_input_and_reuse_question(tmp_path):
    """--setup (KIRO): 이전 입력이 없으면 시편 입력 (자세·루트 갭·개선각·용접 길이, Enter = 템플릿 값; 나머지는 보여 주기만) → job 파일.
    있으면 'Reuse the previous inputs? [u]se / [d]o not use' — use = 그대로, do not use = 다시 입력. 잘못된 답·값은 다시 묻는다."""
    from hongik_scan_ros.auto_teach_client import REUSE_PROMPT, main
    from scan_l0 import config as C
    job = tmp_path / "kiro" / "job_current.yaml"
    said: list = []
    ask, asked = _answers("", "abc", "4.0", "95", "", "390")          # 자세 Enter (1G), 루트 갭 'abc' → 다시 → 4.0, 개선각 95 → 다시 → Enter
    assert main(["--setup", "--job", str(job)], ask=ask, say=said.append) == 0
    j, doc = C.load_job(job)
    sp = C.specimen_from_doc(doc)
    assert j.position == "1G" and (sp.root_gap_mm, sp.bevel_angle_deg, sp.weld_length_mm) == (4.0, 30.0, 390.0)
    assert doc["qc_targets"]["end_position_mm"]["tol"] == 1.0 and doc["qc_targets"]["root_face_mm"] == {"design": 0.0, "tol": 0.2}
    assert not doc.get("teach") and "start_pose" not in doc
    # the template's run blocks travel with the specimen input: the 1-sigma precision targets of the done test, the WPS, the weld
    # design (the KIRO run measures and stops by them; 2026-10-12 the stand-in found the precision block missing)
    _tj, tdoc = C.load_job(os.path.join(STACK, "scan_l0", "scan_l0", "jobs", "groove_10T_1G.yaml"))
    for blk in ("precision", "weld_plan", "weld_design"):
        assert doc[blk] == tdoc[blk], blk
    assert doc["precision"]["end_position_mm"] == 0.10 and j.speed_mps == pytest.approx(0.03) and j.profile_fps == pytest.approx(150.0)
    assert asked[0].startswith("  welding position (1G) [1G]") and sum(a_.startswith("  root gap [mm]") for a_ in asked) == 2
    assert sum(a_.startswith("  bevel angle [deg]") for a_ in asked) == 2 and any("not a number: 'abc'" in s_ for s_ in said)
    assert any("thickness 10.0 mm" in s_ and "root face 0.0 +- 0.2 mm" in s_ and "backing 10.0 mm" in s_ for s_ in said), said
    assert any(s_.strip().startswith("tolerances:") and "end position +-1.0 mm" in s_ for s_ in said), said
    assert any("--edit-all" in s_ for s_ in said)
    # 있으면: 재사용 질문 (모르는 답은 다시) → use = 그대로
    before = job.read_text(encoding="utf-8")
    said.clear()
    ask, asked = _answers("x", "u")
    assert main(["--setup", "--job", str(job)], ask=ask, say=said.append) == 0
    assert asked == [REUSE_PROMPT, REUSE_PROMPT] == ["Reuse the previous inputs? [u]se / [d]o not use: "] * 2
    assert job.read_text(encoding="utf-8") == before and any(s_.startswith("kept ") for s_ in said)
    assert said[0].startswith(f"previous inputs ({job}, saved ") and "1G | root gap 4.0 mm | bevel angle 30.0 deg | weld length 390.0 mm" in said[1]
    # do not use → 다시 입력 (3G 는 이 패키지에서 고를 수 없다 — 다시 묻는다; 1G 템플릿의 값이 기본)
    ask, asked = _answers("d", "3G", "", "", "", "")
    assert main(["--setup", "--job", str(job)], ask=ask, say=said.append) == 0
    j3, doc3 = C.load_job(job)
    assert j3.position == "1G" and C.specimen_from_doc(doc3).root_gap_mm == 5.0
    assert any("welding position must be 1G, got '3G'" in s_ for s_ in said)
    # 개발 옵션 --edit-all: 나머지 값도 묻는다
    ask, asked = _answers("d", "1G", "", "", "", "12", "", "", "", "", "", "", "", "", "", "", "", "", "")
    assert main(["--setup", "--edit-all", "--job", str(job)], ask=ask, say=said.append) == 0
    assert C.specimen_from_doc(C.load_job(job)[1]).thickness_mm == 12.0 and any(a_.startswith("  end position tolerance +-") for a_ in asked)
    assert not any(_HANGUL.search(t_) for t_ in said + asked), [t_ for t_ in said + asked if _HANGUL.search(t_)]


def test_client_help_is_english(capsys):
    from hongik_scan_ros.auto_teach_client import main as client_main
    with pytest.raises(SystemExit):
        client_main(["--help"])
    out = capsys.readouterr().out
    assert "usage:" in out and not _HANGUL.search(out), [ln for ln in out.splitlines() if _HANGUL.search(ln)]
    assert "--teach" not in out and "--expected-mode" not in out            # 손 교시·제어기 모드는 이 패키지에 없다


def test_client_prints_every_progress_line_once():
    """r3 §8-1: 진행 줄을 잃지 않는다 — 로그 토픽이 있으면 그 줄 모두 (피드백은 같은 줄의 사본, 확인 안내도 로그 줄이라 찍지 않는다),
    없으면 피드백 문구 모두 (바로 앞 줄의 되풀이만 빼고). 예전 클라이언트는 서버가 0.2 s 안에 덮어쓴 줄을 받지 못했다."""
    from hongik_scan_ros.auto_teach_client import ProgressPrinter, log_topic_candidates
    lines = [f"[dwell] line {i}" for i in range(300)]
    got: list = []
    pr = ProgressPrinter(got.append, use_log=True)
    for i, ln in enumerate(lines):
        pr.on_log(ln)
        if i % 3 == 0:
            pr.on_feedback(ln)                                              # 피드백 (같은 줄의 사본, 일부만) 은 찍지 않는다
    hint = "-> in another terminal: ros2 service call /x std_srvs/srv/SetBool \"{data: true}\" (false = abort)"
    pr.on_log("CONFIRM move 'acquire_S'")
    pr.on_feedback("CONFIRM move 'acquire_S' " + hint, True)                 # 확인 안내도 서버 로그 줄 (srv) — 피드백은 찍지 않는다
    pr.on_log(hint)
    assert got == lines + ["CONFIRM move 'acquire_S'", hint]
    got2: list = []
    pf = ProgressPrinter(got2.append, use_log=False)
    for ln in lines:
        pf.on_feedback(ln)
        pf.on_feedback(ln)                                                  # 단계 사건이 같은 문구로 다시 낸 피드백
    assert got2 == lines
    assert log_topic_candidates("/hongik_fine_sensing/auto_teach") == ["/hongik_fine_sensing/hongik_auto_teach_server/log",
                                                                       "/hongik_fine_sensing/log"]


def test_client_result_lines_show_pass_unresolved(tmp_path):
    """r3 §8-6: PASS_UNRESOLVED = 출력은 썼고 공차 밖 양이 남았다 — 결과 둘째 줄에 미해결 목록 (sections.json done.unresolved)."""
    from types import SimpleNamespace
    from hongik_scan_ros.auto_teach_client import DELIVERED, result_lines
    out = tmp_path / "res"
    out.mkdir()
    (out / "sections.json").write_text(json.dumps({"schema": "g1core.sections/1", "status": "PASS_UNRESOLVED",
                                                   "done": {"resolved": False, "max_ratio": 1.6, "unresolved": ["end_E"]}}),
                                       encoding="utf-8")
    res = SimpleNamespace(status="PASS_UNRESOLVED", reason="pass budget 4 used; unresolved: ['end_E'] (max k sigma/tol 1.60)",
                          out_dir=str(out), refined_job_path="", length_mm=379.66, ends_observed=True, n_passes=4, n_moves=10,
                          elapsed_s=300.0, converged="false")
    lines = result_lines(res)
    assert lines[0] == "RESULT: PASS_UNRESOLVED - pass budget 4 used; unresolved: ['end_E'] (max k sigma/tol 1.60)"
    assert lines[1].startswith("  UNRESOLVED after the pass budget: ['end_E'] - the outputs are written")
    assert any("scan passes 4" in l_ for l_ in lines) and not any("refined job" in l_ for l_ in lines)
    assert "PASS_UNRESOLVED" in DELIVERED and "DATA_FAIL" not in DELIVERED
