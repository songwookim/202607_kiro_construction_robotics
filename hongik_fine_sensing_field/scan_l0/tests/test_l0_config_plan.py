"""설정/캘리브/작업 파일 입출력, 계획 (예제 1G/3G), R4 방향·3G 바닥 규칙 거부 (2026-10-07: CB CLI·스피드바 정책 제거 — 계획은 함수로 시험)."""
import math
import os

import numpy as np
import pytest
from scan_core.path_gen import EnvelopeError
import yaml
from conftest import CALIB, CONFIG, JOBS

from scan_core.types import Calib, JobError
from scan_l0 import config as C
from scan_l0.planning import make_plan


# --------------------------------------------------------------------------------------
# config / calib / job files
# --------------------------------------------------------------------------------------
def test_settings_from_platform_configs():
    s = C.load_settings(CONFIG)
    assert s.adapter == "moveit_jtc" and s.sensor_port == 1096
    kiro = C.load_settings(C.DEFAULT_CONFIG)
    assert kiro.platform == "hongik_kiro" and kiro.adapter == "moveit_jtc" and kiro.sensor_ip == "192.168.3.15"
    assert "quality_thresholds" not in s.params and not hasattr(s, "qc_thresholds")   # 구간 QC 판정은 은퇴 (4d P2)
    assert C.unnull("null") is None and C.unnull(None) is None and C.unnull(0.5) == 0.5


def test_load_calib_matches_lab_initial():
    c = C.load_calib(CALIB)
    ref = Calib.lab_initial(eoat_yaw_rad=-math.pi / 2, opt_yaw_branch=+1)     # +X 부호 실측 확정 (2026-10-02)
    # 두 방향 스캔 보정 (사용자 승인; eoat_lab_01.yaml 주석): 2026-10-04 광학 프레임에서 Ry(+1.305°) + x +1.33 mm,
    # 2026-10-05 진행 방향 y +11.09 mm (Ry 는 Y 를 바꾸지 않으므로 같은 프레임 이동으로 합쳐진다)
    corr = np.eye(4)
    th = math.radians(1.305)
    corr[:3, :3] = [[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]]
    corr[:3, 3] = [0.00133, 0.01109, 0.0]
    # 2026-10-05 미세 성분 (scan_l0.flip_calib, at21b/at20): 광학 프레임 Trans(+0.58, −0.28, 0) mm · Ry(−0.098°)
    fine = np.eye(4)
    e = math.radians(-0.09812208518039198)
    fine[:3, :3] = [[math.cos(e), 0, math.sin(e)], [0, 1, 0], [-math.sin(e), 0, math.cos(e)]]
    fine[:3, 3] = [0.5800574352363165e-3, -0.2796565436610307e-3, 0.0]
    # 2026-10-06 광축 방향 원점 −25 mm (박스 다자세 캘리브·g3_17 재처리·줄자, 사용자 승인): 광학 프레임 Trans(0, 0, −25) mm
    z25 = np.eye(4)
    z25[2, 3] = -0.025
    # 2026-10-12 레이저 선 축 (광학 X) 둘레 기울기 w_x −1.432° (실험 3 x34_1, 사용자 승인): 1G 패스의 판 위 range 334 mm 를 지나는 X 평행선
    # 둘레의 회전 — 광학 프레임 [Rx(w_x) | (0, D·sin w_x, D·(1 − cos w_x))], D = 0.334 m
    w, D = math.radians(-1.432), 0.334
    wx = np.eye(4)
    wx[1:3, 1:3] = [[math.cos(w), -math.sin(w)], [math.sin(w), math.cos(w)]]
    wx[1:3, 3] = [D * math.sin(w), D * (1.0 - math.cos(w))]
    np.testing.assert_allclose(c.T_flange_opt, ref.T_flange_opt @ corr @ fine @ z25 @ wx, atol=2e-5)
    # the change keeps a point on the optical axis at the pivot range where it was
    p = np.array([0.0, 0.0, D, 1.0])
    np.testing.assert_allclose(c.T_flange_opt @ p, ref.T_flange_opt @ corr @ fine @ z25 @ p, atol=2e-5)
    np.testing.assert_allclose(c.T_flange_eoat, ref.T_flange_eoat, atol=1e-7)
    assert c.dt_s == 0.0 and "eoat_lab_01" in c.source
    assert c.x_sign == -1 and "unconfirmed" not in c.source
    # the previous initial estimate is the same frame turned 180 deg about the optical axis (X, Y reversed; Z, origin kept)
    old = Calib.lab_initial(eoat_yaw_rad=-math.pi / 2, opt_yaw_branch=-1)
    np.testing.assert_allclose(ref.T_flange_opt[:3, 0], -old.T_flange_opt[:3, 0], atol=1e-7)
    np.testing.assert_allclose(ref.T_flange_opt[:3, 2:], old.T_flange_opt[:3, 2:], atol=1e-7)


def test_load_calib_to_dict_format(tmp_path):
    ref = Calib.lab_initial()
    p = tmp_path / "c.yaml"
    ref.save_yaml(p)
    c = C.load_calib(p)
    np.testing.assert_allclose(c.T_flange_opt, ref.T_flange_opt, atol=1e-9)


def test_job_from_settings_and_roundtrip(tmp_path):
    s = C.load_settings(CONFIG)
    job = C.job_from_settings(s, [0.6, -0.2, 0.05], [0.6, 0.3, 0.05], [0, 0, 1], job_id="t", specimen_top_z_m=0.05,
                              gravity_axis_base=[0.0, 0.0, -1.0])        # MoveIt 플랫폼: 중력은 TF 유도 (설정에는 null)
    assert job.standoff_m == 0.32 and job.speed_mps == 0.03 and job.safe_clearance_m == 0.10
    assert job.table_z_m is None and job.specimen_top_z_m == 0.05         # null -> None, override applied
    assert job.max_laser_tilt_deg == 40.0 and job.standoff_min_m == 0.245 and job.torch_protrusion_m == 0.1924  # 설정 safety (2026-10-05 30 → 40; 팁 = 와이어 끝 2026-10-12)
    assert job.vertical_direction == "down" and job.reference_point == "optical" and job.direction_tol_deg == 20.0
    assert job.progression_sign == 1 and job.base_frame == "link0"
    p = C.save_job(tmp_path / "j.yaml", job, teach={"start": {"z_c_mm": 320.0}}, meta={"x": 1})
    job2, doc = C.load_job(p)
    assert np.allclose(job2.start, job.start) and np.allclose(job2.end, job.end) and job2.job_id == "t"
    assert job2.table_z_m is None and doc["teach"]["start"]["z_c_mm"] == 320.0 and doc["meta"]["x"] == 1
    # flat dict with null strings and unknown keys is accepted
    d = C.job_to_dict(job)
    d["table_z_m"] = "null"
    d["unknown_key"] = 5
    with open(tmp_path / "flat.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f)
    job3, _ = C.load_job(tmp_path / "flat.yaml")
    assert job3.table_z_m is None and job3.length_m == pytest.approx(0.5)
    with pytest.raises(KeyError):
        C.job_from_settings(s, [0, 0, 0], [0, 0, 1], bogus=1)
    ov = C.parse_set_overrides(["speed_mps=0.02", "bidirectional=true", "specimen_top_z_m=null"])
    assert ov == {"speed_mps": 0.02, "bidirectional": True, "specimen_top_z_m": None}
    with pytest.raises(ValueError):
        C.parse_set_overrides(["nope=1"])


# --------------------------------------------------------------------------------------
# plan CLI on the example jobs (no CB, no sensor)
# --------------------------------------------------------------------------------------




def _write_job(tmp_path, name, **job):
    base = yaml.safe_load(open(os.path.join(JOBS, "example_3G.yaml"), encoding="utf-8"))
    base["job"].update(job)
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f)
    return str(p)








def test_make_plan_3g_up_direction_and_floor():
    calib = C.load_calib(CALIB)
    from torch_fixtures import CAD_TORCH_KEYPOINT                                    # the rule was built on the CAD torch's keypoint
    job, _ = C.load_job(os.path.join(JOBS, "example_3G.yaml"))
    job.vertical_direction = "up"
    job.torch_lateral_m = tuple(CAD_TORCH_KEYPOINT["torch_lateral_m"])
    job.torch_protrusion_m = CAD_TORCH_KEYPOINT["torch_protrusion_m"]
    # 3G 바닥은 토치 팁도 본다 (2026-10-05): 두 툴 진행 부호 중 토치가 레이저 위쪽인 쪽만 계획된다
    ok, refused = [], []
    for sign in (1, -1):
        job.progression_sign = sign
        try:
            ok.append((sign, make_plan(job, calib, None, v_eff_max_mps=0.05)))
        except EnvelopeError as e:
            refused.append((sign, str(e)))
    assert len(ok) == 1 and len(refused) == 1 and "rule=3G_floor: torch tip" in refused[0][1], (ok, refused)
    plan, stats = ok[0][1]
    names = [w.name for w in plan.waypoints]
    assert "scan_start" in names and "lead_out" in names and "lead_in" not in names
    assert stats["floor_margin_min"] >= -1e-6
    assert plan.travel_sign == -1     # start is the upper point -> actual travel end(lower) -> start(upper)


def test_make_plan_with_home_clamps_transit_speed():
    calib = C.load_calib(CALIB)
    job, _ = C.load_job(os.path.join(JOBS, "example_1G.yaml"))
    from scan_core.frames import transform_from_xyz_rpy
    T_home = transform_from_xyz_rpy([0.6, -0.4, 0.6], [math.pi, 0.0, 0.0])
    plan, stats = make_plan(job, calib, T_home, transit_speed_mps=0.2, descend_speed_mps=0.1, v_eff_max_mps=0.05)
    assert plan.waypoints[0].name == "home" and plan.waypoints[-1].name == "home"
    assert all(w.speed_mps <= 0.05 + 1e-12 for w in plan.waypoints if w.phase != "scan")
    assert stats["n_samples"] > 10 and stats["sample_step_m"] <= 0.005


# --------------------------------------------------------------------------------------
# speed policy math (R2) as used by scan_l0
# --------------------------------------------------------------------------------------




def test_scanjob_direction_validation_messages():
    s = C.load_settings(CONFIG)
    g = dict(gravity_axis_base=[0.0, 0.0, -1.0])
    job = C.job_from_settings(s, [0.5, 0.7, 0.6], [0.5, 0.7, 0.3], [0, -1, 0], orientation="vertical_3G",
                              specimen_top_z_m=0.7, **g)
    job.validate()
    bad = C.job_from_settings(s, [0.3, 0.7, 0.5], [0.7, 0.7, 0.5], [0, -1, 0], orientation="vertical_3G",
                              specimen_top_z_m=0.7, **g)
    with pytest.raises(JobError, match="3G requires vertical path"):
        bad.validate()
    bad1 = C.job_from_settings(s, [0.6, 0.0, 0.05], [0.6, 0.3, 0.15], [0, 0, 1], orientation="flat_1G", **g)
    with pytest.raises(JobError):
        bad1.validate()


# --------------------------------------------------------------------------------------
# groove specimen jobs (specimen block, incidence tilt, FOV check)
# --------------------------------------------------------------------------------------
def test_groove_jobs_load_with_specimen_and_qc_targets():
    # 1G: measurement 30 mm/s @ 150 fps (사용자 2026-10-12 밤: 패스·드웰 정밀 보정; 나머지 이동은 플랫폼 한계), 3G: 30 mm/s @ 100 fps.
    # Specimen: 1G = KIRO's site specimen 200 mm x plates 150 mm each (KIRO-only template values, user 2026-10-13; the lab line: 380 x
    # 300 + 300), 3G = the lab's 380 x 300 (3G is not part of this package's runs)
    for name, orient, tilt, v, fps, length_mm, widths_mm in (
            ("groove_10T_1G.yaml", "flat_1G", 0.0, 0.03, 150.0, 200.0, (150.0, 150.0)),
            ("groove_10T_3G.yaml", "vertical_3G", 10.0, 0.03, 100.0, 380.0, (150.0, 150.0))):
        job, doc = C.load_job(os.path.join(JOBS, name))
        assert job.standoff_m == 0.32 and job.speed_mps == v
        assert job.profile_fps == fps
        assert job.resolved_orientation() == orient and job.incidence_tilt_deg == tilt
        spec = C.specimen_from_doc(doc)
        assert spec is not None and spec.kind == "single_bevel" and spec.thickness_mm == 10.0
        assert spec.bevel_angle_deg == 30.0 and spec.root_gap_mm == 5.0 and spec.backing and spec.backing_thickness_mm == 10.0
        assert spec.weld_length_mm == length_mm and spec.bevel_side == "left"  # 2026-09-30 사용자 확인: 개선판 왼쪽
        assert job.length_m == pytest.approx(spec.weld_length_mm * 1e-3, abs=1e-9)   # the placeholder ends span the design length
        assert tuple(spec.plate_widths_mm) == widths_mm and spec.plate_width_mm == sum(widths_mm)
        assert spec.top_opening_mm == pytest.approx(10.77, abs=0.01)
        qt = C.qc_targets_effective(None, doc)
        assert qt["root_gap_mm"]["tol"] == 0.5 and qt["bevel_angle_deg"]["tol"] == 2.0 and qt["groove_depth_mm"]["tol"] == 0.5
        assert qt["hi_lo_mm"] == {"target_max": 1.0, "warn_above": 2.0}
    assert C.specimen_from_doc({"job": {}}) is None and C.qc_targets_effective(None, None) == {}


def test_save_job_keeps_specimen_block(tmp_path):
    s = C.load_settings(CONFIG)
    assert s.specimen["source"] == "job_file"
    job, doc = C.load_job(os.path.join(JOBS, "groove_10T_1G.yaml"))
    spec = C.specimen_from_doc(doc)
    p = C.save_job(tmp_path / "g.yaml", job, teach={"start": {}}, specimen=spec, qc_targets=C.qc_targets_effective(None, doc))
    job2, doc2 = C.load_job(p)
    assert C.specimen_from_doc(doc2) == spec and doc2["qc_targets"]["root_gap_mm"]["tol"] == 0.5
    assert job2.incidence_tilt_deg == 0.0
    # 용접 절차 (WPS) 블록도 템플릿 그대로 지닌다 (교시 도구·입력 job; 없으면 WeldProcess 기본값)
    assert "weld_plan" not in doc2
    p3 = C.save_job(tmp_path / "w.yaml", job, teach={}, specimen=spec, weld_plan=doc["weld_plan"], weld_design=doc["weld_design"])
    d3 = C.load_job(p3)[1]
    assert d3["weld_plan"] == doc["weld_plan"] and doc["weld_plan"]["work_angle_deg"] == [9.0, [-20.0, 7.0], 7.0]
    assert d3["weld_design"] == doc["weld_design"] and doc["weld_design"]["cap_reinforcement_mm"] == {"nominal": 1.87, "tol": 0.19}


def _job3g(**job):
    base = yaml.safe_load(open(os.path.join(JOBS, "example_3G.yaml"), encoding="utf-8"))
    base["job"].update(job)
    return C.job_from_dict(base["job"])


def test_plan_3g_horizontal_path_rejected():
    job = _job3g(start=[0.3, 0.7, 0.5], end=[0.7, 0.7, 0.5])
    with pytest.raises(JobError, match="3G requires vertical path"):
        make_plan(job, C.load_calib(CALIB), None)


def test_plan_1g_tilted_path_rejected_and_small_slope_planned():
    """1G 자세 확인 (허용 20°): 25° 경사는 '1G requires horizontal path' 로 거부, 12° 경사는 계획이 된다."""
    base = yaml.safe_load(open(os.path.join(JOBS, "example_1G.yaml"), encoding="utf-8"))
    calib = C.load_calib(CALIB)

    def job(deg):
        t = math.radians(deg)
        u = [0.0, math.cos(t), math.sin(t)]
        n = [0.0, -math.sin(t), math.cos(t)]
        from torch_fixtures import CAD_TORCH_KEYPOINT                                # the direction rule's test: the CAD torch (the
        return C.job_from_dict(dict(base["job"], start=[0.6, -0.2, 0.05], end=[0.6, -0.2 + 0.5 * u[1], 0.05 + 0.5 * u[2]],   # turned
                                    surface_normal=n, specimen_top_z_m=None, **CAD_TORCH_KEYPOINT))   # one reaches lower on a slope)
    with pytest.raises(JobError, match="1G requires horizontal path"):
        make_plan(job(25.0), calib, None)
    plan, stats = make_plan(job(12.0), calib, None)
    assert plan.waypoints and stats["n_samples"] > 10


def test_plan_3g_floor_violation_rejected():
    """판 법선이 4° 아래로 기울면 아래 점에서 광학 원점 (판 법선 방향 standoff) 이 3G 바닥 아래 → EnvelopeError rule=3G_floor."""
    t = math.radians(4.0)
    n = [0.0, -math.cos(t), -math.sin(t)]
    u = [0.0, -math.sin(t), math.cos(t)]
    start = [0.5, 0.7, 0.6]
    end = [start[0] - 0.3 * u[0], start[1] - 0.3 * u[1], start[2] - 0.3 * u[2]]
    with pytest.raises(EnvelopeError, match="3G_floor"):
        make_plan(_job3g(start=start, end=end, surface_normal=n), C.load_calib(CALIB), None)

